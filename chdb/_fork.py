"""POSIX fork() safety and engine-lifecycle tracking for chDB.

ClickHouse's embedded engine starts persistent C++ background thread pools
(``GlobalThreadPool`` and background maintenance threads) on first use.
POSIX ``fork()`` duplicates only the calling thread, leaving those thread
pools dead in the child process so subsequent queries or native teardown
deadlock indefinitely.

This module:
1. Tracks whether the native ClickHouse engine has been initialized in the
   current process (and whether any native operations/connections are open).
2. Registers an ``os.register_at_fork(after_in_child=...)`` handler that:
   - Resets Python-side locks and module globals so the child never inherits
     locked mutexes from another parent thread.
   - Leaves the child completely usable when the engine was NOT initialized
     before ``fork()`` (e.g. top-level ``import chdb`` before ``fork()``).
   - Marks the child as forked-after-init when the engine WAS initialized
     before ``fork()``, neutralizes inherited open handles without invoking
     native C++ destructors/``close()``, fails fast with ``RuntimeError`` on
     any subsequent chDB call, and prevents child shutdown deadlocks in
     ``_chdb`` exit handlers / C++ static destructors.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import os
import sys
import threading
import weakref
from typing import Any, Optional


class ForkAfterInitError(RuntimeError):
    """Raised when chDB is used in a child process forked after engine init."""


_engine_init_pid: Optional[int] = None
_active_native_ops: int = 0
_forked_after_init: bool = False
_fork_parent_pid: Optional[int] = None
_fork_child_pid: Optional[int] = None
_needs_fast_exit_on_shutdown: bool = False
_child_exit_code: Optional[int] = None
_orig_sys_exit: Any = None
_exit_hook_installed: bool = False

_live_handles: dict[int, weakref.ref] = {}
# Pin inherited native pybind11 handles in a forked-after-init child so GC
# never drops their refcount to 0 and triggers C++ ~Connection() -> deadlock.
_leaked_post_fork_handles: list[Any] = []

_chdb_module: Any = None
_installed: bool = False
_at_fork_registered: bool = False


def _fork_error_message() -> str:
    return (
        f"chDB cannot be used in child process (pid={_fork_child_pid}) forked from "
        f"parent process (pid={_fork_parent_pid}) because the ClickHouse engine was "
        f"already initialized in the parent (pid={_engine_init_pid}) before fork(). "
        "ClickHouse background threads do not survive POSIX fork(), which would deadlock. "
        "To use chDB with multiprocessing, either: "
        "(1) use multiprocessing.get_context('spawn') or 'forkserver', or "
        "(2) avoid running chDB queries or opening connections in the parent process before fork() "
        "(importing chdb before fork() is safe as long as the engine is initialized only inside worker processes)."
    )


def is_engine_initialized() -> bool:
    """Return True if the native ClickHouse engine has been initialized in this process."""
    return _engine_init_pid is not None and not _forked_after_init


def is_forked_after_init() -> bool:
    """Return True if this process was forked after the chDB engine was initialized."""
    return _forked_after_init


def should_fast_exit_on_shutdown() -> bool:
    """Return True if normal C/atexit teardown in this child would deadlock."""
    return _needs_fast_exit_on_shutdown


def _install_child_exit_hook() -> None:
    """Track sys.exit() status in a forked child that must bypass C++ teardown."""
    global _orig_sys_exit, _exit_hook_installed
    if _exit_hook_installed:
        return
    _exit_hook_installed = True
    _orig_sys_exit = sys.exit

    def _tracked_exit(status: Any = None) -> None:
        global _child_exit_code
        if status is None:
            _child_exit_code = 0
        elif isinstance(status, int):
            _child_exit_code = int(status)
        else:
            _child_exit_code = 1
        _orig_sys_exit(status)

    sys.exit = _tracked_exit


def fast_exit_child() -> None:
    """Flush Python stdio/logging and exit via os._exit() to skip deadlocking C++ teardown."""
    try:
        logging_mod = sys.modules.get("logging")
        if logging_mod is not None and hasattr(logging_mod, "shutdown"):
            logging_mod.shutdown()
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            try:
                stream.flush()
            except Exception:
                pass
    if getattr(sys, "last_value", None) is not None or getattr(sys, "last_exc", None) is not None:
        code = 1
    elif _child_exit_code is not None:
        code = _child_exit_code
    else:
        code = 0
    os._exit(code)


def _is_native_handle(obj: Any) -> bool:
    if obj is None:
        return False
    mod = getattr(type(obj), "__module__", "")
    return mod == "chdb._chdb" or mod.endswith("._chdb")


def _pin_native(obj: Any) -> bool:
    if _is_native_handle(obj):
        _leaked_post_fork_handles.append(obj)
        return True
    return False


def track_handle(obj: Any) -> None:
    """Track a Python wrapper object holding a native chDB handle."""
    key = id(obj)

    def _on_gc(_ref: weakref.ref, _key: int = key) -> None:
        if _live_handles.get(_key) is _ref:
            _live_handles.pop(_key, None)

    try:
        _live_handles[key] = weakref.ref(obj, _on_gc)
    except TypeError:
        pass


def neutralize_handle(obj: Any, _seen: Optional[set[int]] = None) -> bool:
    """Detach and pin native handles on ``obj`` without calling native close().

    Returns True if any live native ``_chdb`` handle was found and pinned.
    """
    if obj is None:
        return False
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return False
    _seen.add(obj_id)

    if _pin_native(obj):
        return True

    had_native = False
    obj_dict = getattr(obj, "__dict__", None)
    if not isinstance(obj_dict, dict):
        return False

    # Reset per-instance locks (e.g. ArrowBatchStream._lock) so the child never
    # blocks on a lock held by a vanished parent thread.
    if "_lock" in obj_dict:
        lock_obj = obj_dict["_lock"]
        if isinstance(lock_obj, type(threading.RLock())):
            obj_dict["_lock"] = threading.RLock()
        elif isinstance(lock_obj, type(threading.Lock())):
            obj_dict["_lock"] = threading.Lock()

    for attr in ("_cursor", "_active", "_conn", "_raw", "_result", "_c_inserter", "_session"):
        if attr in obj_dict:
            val = obj_dict[attr]
            if val is not None:
                if _pin_native(val):
                    had_native = True
                elif neutralize_handle(val, _seen):
                    had_native = True
                obj_dict[attr] = None

    if "_on_close" in obj_dict:
        obj_dict["_on_close"] = None
    if "_closed" in obj_dict:
        obj_dict["_closed"] = True
    if "_exhausted" in obj_dict:
        obj_dict["_exhausted"] = True
    if "_finished" in obj_dict:
        obj_dict["_finished"] = True
    if "_counted" in obj_dict:
        obj_dict["_counted"] = False

    return had_native


def _collect_gc_handles() -> list[Any]:
    """Safety-net scan for any open chDB wrapper objects not in ``_live_handles``."""
    wrapper_types: list[type] = []
    module_classes = (
        ("chdb.state.sqlitelike", ("Connection", "Cursor", "StreamingResult", "StreamingInserter")),
        ("chdb.arrow", ("ArrowSession", "ArrowBatchStream")),
        ("datastore.connection", ("Connection",)),
        ("chdb.durable.engine", ("ManagedConnection",)),
    )
    for mod_name, cls_names in module_classes:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for cls_name in cls_names:
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type):
                wrapper_types.append(cls)
    if not wrapper_types:
        return []
    type_tuple = tuple(wrapper_types)
    return [obj for obj in gc.get_objects() if isinstance(obj, type_tuple)]


def _reset_python_state_in_child(engine_was_initialized: bool) -> bool:
    """Reset Python-side locks and globals in a forked child without calling native close()."""
    had_open_handle = False

    chdb_mod = _chdb_module or sys.modules.get("chdb")
    if chdb_mod is not None:
        new_lock = threading.Lock()
        new_event = threading.Event()
        chdb_mod.g_conn_lock = new_lock
        chdb_mod.g_arrow_stream_active = new_event
        arrow_mod = sys.modules.get("chdb.arrow")
        if arrow_mod is not None:
            arrow_mod.g_conn_lock = new_lock
            arrow_mod.g_arrow_stream_active = new_event

    tool_mod = sys.modules.get("chdb.agents.tool")
    if tool_mod is not None:
        tool_mod._ABANDONED_LOCK = threading.Lock()
        abandoned = getattr(tool_mod, "_ABANDONED_SESSIONS", None)
        if abandoned:
            if engine_was_initialized:
                for sess in list(abandoned):
                    if neutralize_handle(sess):
                        had_open_handle = True
            abandoned.clear()

    ds_exec_mod = sys.modules.get("datastore.executor")
    if ds_exec_mod is not None:
        global_exec = getattr(ds_exec_mod, "_global_executor", None)
        if global_exec is not None:
            conn = getattr(global_exec, "connection", None)
            if conn is not None and engine_was_initialized:
                if neutralize_handle(conn):
                    had_open_handle = True
            ds_exec_mod._global_executor = None

    durable_mod = sys.modules.get("chdb.durable.engine")
    if durable_mod is not None:
        durable_mod._LIVE_CONNECTIONS = 0

    return had_open_handle


def _handle_detected_fork(current_pid: int) -> None:
    """Transition state when a fork is detected after engine initialization."""
    global _active_native_ops, _forked_after_init, _fork_parent_pid, _fork_child_pid
    global _needs_fast_exit_on_shutdown, _live_handles

    had_active_ops = _active_native_ops > 0
    _active_native_ops = 0

    engine_was_initialized = _engine_init_pid is not None or _forked_after_init
    had_open_handle = _reset_python_state_in_child(engine_was_initialized)

    refs = list(_live_handles.values())
    _live_handles = {}

    if not engine_was_initialized:
        return

    if not _forked_after_init:
        _fork_parent_pid = _engine_init_pid
    _forked_after_init = True
    _fork_child_pid = current_pid

    seen: set[int] = set()
    for ref in refs:
        obj = ref()
        if obj is not None and neutralize_handle(obj, seen):
            had_open_handle = True

    for obj in _collect_gc_handles():
        if neutralize_handle(obj, seen):
            had_open_handle = True

    if had_active_ops or had_open_handle:
        _needs_fast_exit_on_shutdown = True
        _install_child_exit_hook()


def _after_fork_in_child() -> None:
    _handle_detected_fork(os.getpid())


def check_fork_safety() -> None:
    """Fail fast if this process was forked after the ClickHouse engine initialized."""
    if _forked_after_init:
        raise ForkAfterInitError(_fork_error_message())
    if _engine_init_pid is not None:
        current_pid = os.getpid()
        if current_pid != _engine_init_pid:
            _handle_detected_fork(current_pid)
            raise ForkAfterInitError(_fork_error_message())


def mark_engine_initialized() -> None:
    """Check fork safety and record that the native engine is initialized in this PID."""
    global _engine_init_pid
    check_fork_safety()
    if _engine_init_pid is None:
        _engine_init_pid = os.getpid()


@contextlib.contextmanager
def native_operation():
    """Track an active native engine call so a concurrent fork() is detected."""
    global _active_native_ops
    mark_engine_initialized()
    _active_native_ops += 1
    try:
        yield
    finally:
        if _active_native_ops > 0:
            _active_native_ops -= 1


def _wrap_guarded(fn):
    @functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        check_fork_safety()
        return fn(self, *args, **kwargs)

    return _wrapper


def _wrap_safe_close(fn):
    @functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        if _forked_after_init:
            neutralize_handle(self)
            return None
        return fn(self, *args, **kwargs)

    return _wrapper


def patch_sqlitelike(module: Any) -> None:
    """Idempotently wrap ``chdb.state.sqlitelike`` classes for fork safety.

    Leaves ``chdb._chdb.connect`` untouched so ``type(conn._conn) is chdb._chdb.connect``
    and native ABI checks remain intact.
    """
    if getattr(module, "_chdb_fork_patched", False):
        return
    module._chdb_fork_patched = True

    conn_cls = getattr(module, "Connection", None)
    if conn_cls is not None:
        orig_init = conn_cls.__init__

        @functools.wraps(orig_init)
        def _conn_init(self, *args, **kwargs):
            with native_operation():
                orig_init(self, *args, **kwargs)
            track_handle(self)

        conn_cls.__init__ = _conn_init

        for name in ("query", "generate_sql", "ask", "set_progress_callback"):
            orig = getattr(conn_cls, name, None)
            if callable(orig):
                setattr(conn_cls, name, _wrap_guarded(orig))

        for name in ("cursor", "send_query", "send_insert"):
            orig = getattr(conn_cls, name, None)
            if callable(orig):
                def _make_tracking_wrapper(fn):
                    @functools.wraps(fn)
                    def _wrapper(self, *args, **kwargs):
                        check_fork_safety()
                        result = fn(self, *args, **kwargs)
                        track_handle(result)
                        return result

                    return _wrapper

                setattr(conn_cls, name, _make_tracking_wrapper(orig))

        if callable(getattr(conn_cls, "close", None)):
            conn_cls.close = _wrap_safe_close(conn_cls.close)

    cursor_cls = getattr(module, "Cursor", None)
    if cursor_cls is not None:
        for name in ("execute", "commit"):
            orig = getattr(cursor_cls, name, None)
            if callable(orig):
                setattr(cursor_cls, name, _wrap_guarded(orig))
        if callable(getattr(cursor_cls, "close", None)):
            cursor_cls.close = _wrap_safe_close(cursor_cls.close)

    stream_cls = getattr(module, "StreamingResult", None)
    if stream_cls is not None:
        for name in ("fetch", "__next__", "record_batch", "__arrow_c_stream__"):
            orig = getattr(stream_cls, name, None)
            if callable(orig):
                setattr(stream_cls, name, _wrap_guarded(orig))
        for name in ("cancel", "close"):
            orig = getattr(stream_cls, name, None)
            if callable(orig):
                setattr(stream_cls, name, _wrap_safe_close(orig))

    inserter_cls = getattr(module, "StreamingInserter", None)
    if inserter_cls is not None:
        for name in ("append", "finish"):
            orig = getattr(inserter_cls, name, None)
            if callable(orig):
                setattr(inserter_cls, name, _wrap_guarded(orig))
        for name in ("cancel", "close"):
            orig = getattr(inserter_cls, name, None)
            if callable(orig):
                setattr(inserter_cls, name, _wrap_safe_close(orig))


def ensure_sqlitelike_patched() -> None:
    """Patch ``chdb.state.sqlitelike`` if it has already been imported."""
    mod = sys.modules.get("chdb.state.sqlitelike")
    if mod is not None:
        patch_sqlitelike(mod)


class _PostImportLoader:
    """Wrap a module loader to run a callback right after exec_module."""

    def __init__(self, loader: Any, callback: Any):
        self._loader = loader
        self._callback = callback

    def create_module(self, spec: Any) -> Any:
        if hasattr(self._loader, "create_module"):
            return self._loader.create_module(spec)
        return None

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        self._callback(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _SqlitelikePatchFinder:
    """One-shot meta_path finder that patches chdb.state.sqlitelike on import."""

    _TARGET = "chdb.state.sqlitelike"

    def __init__(self) -> None:
        self._in_progress = False

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname != self._TARGET or self._in_progress:
            return None
        self._in_progress = True
        try:
            for finder in list(sys.meta_path):
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PostImportLoader(spec.loader, self._on_loaded)
                    return spec
            return None
        finally:
            self._in_progress = False

    def _on_loaded(self, module: Any) -> None:
        try:
            patch_sqlitelike(module)
        finally:
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass


def install(chdb_module: Any, raw_chdb_module: Any) -> None:
    """Install fork protection hooks for ``chdb`` and ``_chdb`` (idempotent)."""
    global _chdb_module, _installed, _at_fork_registered
    _chdb_module = chdb_module

    if not _installed:
        _installed = True

        # Wrap raw _chdb.query (a function, unlike _chdb.connect which is a class).
        raw_query = getattr(raw_chdb_module, "query", None)
        if callable(raw_query) and not getattr(raw_query, "_chdb_fork_wrapped", False):
            @functools.wraps(raw_query)
            def _guarded_raw_query(*args, **kwargs):
                with native_operation():
                    return raw_query(*args, **kwargs)

            _guarded_raw_query._chdb_fork_wrapped = True
            try:
                setattr(raw_chdb_module, "query", _guarded_raw_query)
            except Exception:
                pass

        if "chdb.state.sqlitelike" in sys.modules:
            patch_sqlitelike(sys.modules["chdb.state.sqlitelike"])
        else:
            sys.meta_path.insert(0, _SqlitelikePatchFinder())

    if not _at_fork_registered and hasattr(os, "register_at_fork"):
        _at_fork_registered = True
        os.register_at_fork(after_in_child=_after_fork_in_child)
