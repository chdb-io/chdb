"""Internal `:memory:` connection pool and stateful-subpackage lifecycle hooks.

`chdb-core` hosts a single `EmbeddedServer` bound to one storage path per
process. Repeatedly opening and closing the sole `:memory:` connection boots
and tears down `EmbeddedServer` on every `chdb.query()` call, while a single
module-level lock serializes threads and deadlocks re-entrant Python UDFs.

This module maintains a pool of reusable `:memory:` `_chdb.connect` handles for
stateless module-level queries and automatically drains idle handles whenever a
connection to another storage path (or with custom server startup options) is
opened. It also instruments `chdb.state.sqlitelike` on import so abandoned
`StreamingResult` iterators cancel their native stream and release pinned
`Python(df)` references on garbage collection.
"""

from __future__ import annotations

import importlib.abc
import os
import sys
import threading
import weakref
from typing import Any, Optional, Tuple


def normalize_conn_target(conn_str: Optional[str]) -> Tuple[str, bool]:
    """Return ``(normalized_path, has_query_params)`` matching ``LocalChdb.cpp``."""
    if not conn_str or conn_str == ":memory:":
        return ":memory:", False

    working = conn_str
    if working.startswith("file:"):
        working = working[5:]
        if working.startswith("///"):
            working = working[2:]

    if "?" in working:
        raw_path, query = working.split("?", 1)
        has_options = bool(query)
    else:
        raw_path = working
        has_options = False

    if not raw_path or raw_path == ":memory:":
        return ":memory:", has_options

    if not os.path.isabs(raw_path):
        try:
            raw_path = os.path.abspath(raw_path)
        except OSError:
            pass

    return raw_path, has_options


def is_memory_path(path: Optional[str]) -> bool:
    norm_path, has_options = normalize_conn_target(path)
    return norm_path == ":memory:" and not has_options


def classify_sql(conn: Any, sql: str) -> str:
    """Classify ``sql`` via ``conn.classify_query`` when supported by ``_chdb``."""
    classify = getattr(conn, "classify_query", None)
    if not callable(classify):
        return "READ_ONLY"
    try:
        report = classify(sql, "")
        qc = report.get("query_class")
        name = getattr(qc, "name", None)
        if isinstance(name, str):
            return name
        return str(qc)
    except Exception:
        return "UNKNOWN"


def _safe_close(conn: Any) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


class _UserConnToken:
    __slots__ = ("pool", "norm_path", "closed")

    def __init__(self, pool: "MemoryConnectionPool", norm_path: str):
        self.pool = pool
        self.norm_path = norm_path
        self.closed = False

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pool._unregister_user_path(self.norm_path)


class MemoryConnectionPool:
    """Thread-safe pool of clean `:memory:` ``_chdb.connect`` handles."""

    def __init__(self, raw_chdb: Any, max_idle: int = 16):
        self._chdb = raw_chdb
        self._max_idle = max_idle
        self._lock = threading.Lock()
        self._idle: list[Any] = []
        self._in_use = 0
        self._generation = 0
        self._shutting_down = False
        self._user_memory_conns = 0
        self._active_non_memory_paths: dict[str, int] = {}

    def borrow(self) -> Tuple[Any, int]:
        with self._lock:
            if self._shutting_down:
                raise RuntimeError("chDB is shutting down")
            conn = self._idle.pop() if self._idle else None
            self._in_use += 1
            gen = self._generation

        if conn is None:
            try:
                conn = self._chdb.connect(":memory:")
            except BaseException:
                with self._lock:
                    self._in_use -= 1
                raise
        return conn, gen

    def release(
        self,
        conn: Any,
        gen: int,
        *,
        sql_class: str = "READ_ONLY",
        had_error: bool = False,
    ) -> None:
        if not had_error and sql_class == "READ_ONLY":
            close_now = False
            with self._lock:
                self._in_use -= 1
                if (
                    not self._shutting_down
                    and gen == self._generation
                    and len(self._idle) < self._max_idle
                ):
                    self._idle.append(conn)
                else:
                    close_now = True
            if close_now:
                _safe_close(conn)
            return

        if not had_error and sql_class == "CONTROL":
            # Session-scoped mutation (SET / USE / TEMPORARY TABLE). Replace
            # `conn` with a fresh `:memory:` handle before closing `conn` when
            # nothing else holds `:memory:` open so `EmbeddedServer` stays warm
            # while no session settings leak to the next borrower.
            need_keeper = False
            with self._lock:
                if (
                    not self._shutting_down
                    and gen == self._generation
                    and not self._idle
                    and self._in_use == 1
                    and self._user_memory_conns == 0
                ):
                    need_keeper = True
            keeper = None
            if need_keeper:
                try:
                    keeper = self._chdb.connect(":memory:")
                except Exception:
                    keeper = None

            discard_keeper = False
            with self._lock:
                self._in_use -= 1
                if (
                    keeper is not None
                    and not self._shutting_down
                    and gen == self._generation
                    and len(self._idle) < self._max_idle
                ):
                    self._idle.append(keeper)
                else:
                    discard_keeper = True

            if discard_keeper and keeper is not None:
                _safe_close(keeper)
            _safe_close(conn)
            return

        # Error or catalog-mutating statement (`CREATE TABLE`, `CREATE FUNCTION`, ...).
        # Close `conn`, and when a stateless mutating query ran with no other
        # `:memory:` borrowers or user sessions open, drain idle handles too so
        # subsequent stateless queries see a clean `:memory:` catalog.
        to_close = [conn]
        with self._lock:
            self._in_use -= 1
            if (
                not had_error
                and sql_class in ("MUTATING", "MUTATING_GLOBAL")
                and self._in_use == 0
                and self._user_memory_conns == 0
                and self._idle
            ):
                to_close.extend(self._idle)
                self._idle.clear()
                self._generation += 1

        for item in to_close:
            _safe_close(item)

    def drain_idle(self) -> None:
        with self._lock:
            if not self._idle:
                self._generation += 1
                return
            conns = list(self._idle)
            self._idle.clear()
            self._generation += 1

        for conn in conns:
            _safe_close(conn)

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            conns = list(self._idle)
            self._idle.clear()
            self._generation += 1

        for conn in conns:
            _safe_close(conn)

    def prepare_for_connection(self, conn_str: Optional[str]) -> str:
        """Drain idle `:memory:` handles if ``conn_str`` needs a fresh engine."""
        norm_path, has_options = normalize_conn_target(conn_str)
        if norm_path != ":memory:":
            self.drain_idle()
            return norm_path

        if has_options:
            should_drain = False
            with self._lock:
                if self._idle and self._in_use == 0 and self._user_memory_conns == 0:
                    should_drain = True
            if should_drain:
                self.drain_idle()
        return norm_path

    def register_user_connection(self, owner: Any, norm_path: str) -> _UserConnToken:
        with self._lock:
            if norm_path == ":memory:":
                self._user_memory_conns += 1
            else:
                self._active_non_memory_paths[norm_path] = (
                    self._active_non_memory_paths.get(norm_path, 0) + 1
                )
        token = _UserConnToken(self, norm_path)
        weakref.finalize(owner, token.release)
        return token

    def _unregister_user_path(self, norm_path: str) -> None:
        with self._lock:
            if norm_path == ":memory:":
                if self._user_memory_conns > 0:
                    self._user_memory_conns -= 1
            else:
                remaining = self._active_non_memory_paths.get(norm_path, 0) - 1
                if remaining > 0:
                    self._active_non_memory_paths[norm_path] = remaining
                else:
                    self._active_non_memory_paths.pop(norm_path, None)

    def active_non_memory_path(self) -> Optional[str]:
        with self._lock:
            for path, count in self._active_non_memory_paths.items():
                if count > 0:
                    return path
        return None

    def resolve_engine_version(self, fallback: str) -> str:
        fn = getattr(self._chdb, "engine_version", None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass

        active_path = self.active_non_memory_path()
        if active_path is not None:
            conn = None
            try:
                conn = self._chdb.connect(active_path)
                raw = conn.query("SELECT version();", "CSV").bytes()
                return raw.decode("utf-8", errors="replace").strip().strip('"')
            except Exception:
                return fallback
            finally:
                _safe_close(conn)

        try:
            conn, gen = self.borrow()
        except Exception:
            return fallback

        had_error = False
        try:
            raw = conn.query("SELECT version();", "CSV").bytes()
            return raw.decode("utf-8", errors="replace").strip().strip('"')
        except Exception:
            had_error = True
            return fallback
        finally:
            self.release(conn, gen, sql_class="READ_ONLY", had_error=had_error)


# ---------------------------------------------------------------------------
# Stateful connection & StreamingResult lifecycle instrumentation
# ---------------------------------------------------------------------------


class _StreamState:
    __slots__ = ("stream_id", "c_result", "done")

    def __init__(self, stream_id: int, c_result: Any):
        self.stream_id = stream_id
        self.c_result = c_result
        self.done = False


class _ConnTracker:
    """Tracks active streams and in-flight C++ calls on one native connection.

    Allows `StreamingResult` garbage-collection finalizers to cancel abandoned
    streams without risking re-entrant `client_mutex` deadlocks or cancelling a
    newer stream that superseded an older one on the same connection.
    """

    def __init__(self, c_conn: Any, token: _UserConnToken):
        self.c_conn = c_conn
        self.token = token
        self.lock = threading.Lock()
        self.in_flight = 0
        self.next_stream_id = 0
        self.active_stream_id: Optional[int] = None
        self.pending_cancel: Any = None
        self.closed = False

    def enter_op(self, *, invalidates_stream: bool = False) -> None:
        pending = None
        with self.lock:
            if invalidates_stream:
                self.active_stream_id = None
            if self.in_flight == 0 and self.pending_cancel is not None:
                pending, self.pending_cancel = self.pending_cancel, None
            self.in_flight += 1

        if pending is not None and not self.closed:
            try:
                self.c_conn.streaming_cancel_query(pending)
            except Exception:
                pass

    def exit_op(self) -> None:
        while True:
            pending = None
            with self.lock:
                self.in_flight -= 1
                if (
                    self.in_flight == 0
                    and not self.closed
                    and self.pending_cancel is not None
                ):
                    pending, self.pending_cancel = self.pending_cancel, None
                    self.in_flight += 1
            if pending is None:
                return
            try:
                self.c_conn.streaming_cancel_query(pending)
            except Exception:
                pass

    def register_stream(self, c_result: Any) -> _StreamState:
        with self.lock:
            self.next_stream_id += 1
            stream_id = self.next_stream_id
            self.active_stream_id = stream_id
            return _StreamState(stream_id, c_result)

    def mark_stream_exhausted(self, state: _StreamState) -> None:
        with self.lock:
            state.done = True
            state.c_result = None
            if self.active_stream_id == state.stream_id:
                self.active_stream_id = None

    def cancel_stream_explicit(self, state: _StreamState) -> None:
        c_result = None
        with self.lock:
            if state.done:
                return
            state.done = True
            c_result, state.c_result = state.c_result, None
            if self.closed or self.active_stream_id != state.stream_id:
                return
            self.active_stream_id = None
            self.in_flight += 1

        try:
            if c_result is not None:
                self.c_conn.streaming_cancel_query(c_result)
        finally:
            self.exit_op()

    def cancel_stream_from_gc(self, state: _StreamState) -> None:
        c_result = None
        run_now = False
        with self.lock:
            if state.done:
                return
            state.done = True
            c_result, state.c_result = state.c_result, None
            if self.closed or self.active_stream_id != state.stream_id or c_result is None:
                return
            self.active_stream_id = None
            if self.in_flight == 0:
                self.in_flight += 1
                run_now = True
            else:
                self.pending_cancel = c_result

        if run_now:
            try:
                self.c_conn.streaming_cancel_query(c_result)
            except Exception:
                pass
            finally:
                self.exit_op()

    def mark_closed(self) -> None:
        with self.lock:
            self.closed = True
            self.active_stream_id = None
            self.pending_cancel = None
        self.token.release()


def _patch_sqlitelike(mod: Any, pool: MemoryConnectionPool) -> None:
    if getattr(mod, "_chdb_pool_patched", False):
        return
    mod._chdb_pool_patched = True

    Connection = mod.Connection
    StreamingResult = mod.StreamingResult
    ChdbError = mod.ChdbError

    orig_conn_init = Connection.__init__
    orig_conn_query = Connection.query
    orig_conn_send_query = Connection.send_query
    orig_conn_send_insert = Connection.send_insert
    orig_conn_close = Connection.close

    def conn_init(self, connection_string: str):
        norm_path = pool.prepare_for_connection(connection_string)
        orig_conn_init(self, connection_string)
        token = pool.register_user_connection(self, norm_path)
        self._chdb_tracker = _ConnTracker(self._conn, token)

    def conn_query(self, query: str, format: str = "CSV", params=None):
        tracker: Optional[_ConnTracker] = getattr(self, "_chdb_tracker", None)
        if tracker is None:
            return orig_conn_query(self, query, format, params=params)
        tracker.enter_op(invalidates_stream=True)
        try:
            return orig_conn_query(self, query, format, params=params)
        finally:
            tracker.exit_op()

    def conn_send_query(self, query: str, format: str = "CSV", params=None):
        tracker: Optional[_ConnTracker] = getattr(self, "_chdb_tracker", None)
        if tracker is None:
            return orig_conn_send_query(self, query, format, params=params)
        tracker.enter_op(invalidates_stream=True)
        try:
            stream = orig_conn_send_query(self, query, format, params=params)
            stream._bind_tracker(tracker)
            return stream
        finally:
            tracker.exit_op()

    def conn_send_insert(self, query: str, format: str = "CSV"):
        tracker: Optional[_ConnTracker] = getattr(self, "_chdb_tracker", None)
        if tracker is None:
            return orig_conn_send_insert(self, query, format)
        tracker.enter_op(invalidates_stream=False)
        try:
            return orig_conn_send_insert(self, query, format)
        finally:
            tracker.exit_op()

    def conn_close(self) -> None:
        tracker: Optional[_ConnTracker] = getattr(self, "_chdb_tracker", None)
        if tracker is not None:
            tracker.mark_closed()
        orig_conn_close(self)

    Connection.__init__ = conn_init
    Connection.query = conn_query
    Connection.send_query = conn_send_query
    Connection.send_insert = conn_send_insert
    Connection.close = conn_close

    orig_sr_init = StreamingResult.__init__

    def sr_init(
        self, c_result, conn, result_func, supports_record_batch, is_dataframe
    ):
        orig_sr_init(
            self, c_result, conn, result_func, supports_record_batch, is_dataframe
        )
        self._lock = threading.RLock()
        self._tracker: Optional[_ConnTracker] = None
        self._stream_state: Optional[_StreamState] = None
        self._finalizer: Optional[weakref.finalize] = None

    def sr_bind_tracker(self, tracker: _ConnTracker) -> None:
        state = tracker.register_stream(self._result)
        self._tracker = tracker
        self._stream_state = state
        self._finalizer = weakref.finalize(
            self, tracker.cancel_stream_from_gc, state
        )

    def sr_fetch(self):
        with self._lock:
            if self._exhausted:
                return None

            tracker = self._tracker
            state = self._stream_state
            if (
                tracker is not None
                and state is not None
                and tracker.active_stream_id != state.stream_id
            ):
                self._exhausted = True
                self._result = None
                state.done = True
                state.c_result = None
                self._cleanup_progress_callback_once()
                raise ChdbError("Streaming query failed: No active streaming query")

            if tracker is not None:
                tracker.enter_op(invalidates_stream=False)
            try:
                if self._is_dataframe:
                    result = self._conn.streaming_fetch_df(self._result)
                    empty = result is None or result.empty
                else:
                    result = self._conn.streaming_fetch_result(self._result)
                    empty = result is None or result.rows_read() == 0

                if empty:
                    self._exhausted = True
                    self._result = None
                    if tracker is not None and state is not None:
                        tracker.mark_stream_exhausted(state)
                    if self._finalizer is not None:
                        self._finalizer.detach()
                    self._cleanup_progress_callback_once()
                    return None
                return self._result_func(result)
            except Exception as e:
                self._exhausted = True
                self._result = None
                if tracker is not None and state is not None:
                    tracker.mark_stream_exhausted(state)
                if self._finalizer is not None:
                    self._finalizer.detach()
                self._cleanup_progress_callback_once()
                if isinstance(e, ChdbError):
                    raise
                raise ChdbError(f"Streaming query failed: {str(e)}") from e.with_traceback(None)
            finally:
                if tracker is not None:
                    tracker.exit_op()

    def sr_cancel(self):
        with self._lock:
            tracker = self._tracker
            state = self._stream_state
            if not self._exhausted:
                self._exhausted = True
                c_result, self._result = self._result, None
                if self._finalizer is not None:
                    self._finalizer.detach()
                try:
                    if tracker is not None and state is not None:
                        tracker.cancel_stream_explicit(state)
                    elif c_result is not None:
                        self._conn.streaming_cancel_query(c_result)
                except Exception as e:
                    raise ChdbError(
                        f"Failed to cancel streaming query: {str(e)}"
                    ) from e.with_traceback(None)
                finally:
                    self._cleanup_progress_callback_once()
            else:
                self._result = None
                if self._finalizer is not None:
                    self._finalizer.detach()
                self._cleanup_progress_callback_once()

    def sr_del(self):
        finalizer = getattr(self, "_finalizer", None)
        try:
            if finalizer is not None:
                finalizer()
            elif not getattr(self, "_exhausted", True):
                self.cancel()
        except Exception:
            pass
        finally:
            self._result = None

    StreamingResult.__init__ = sr_init
    StreamingResult._bind_tracker = sr_bind_tracker
    StreamingResult.fetch = sr_fetch
    StreamingResult.cancel = sr_cancel
    StreamingResult.__del__ = sr_del


def _patch_session_state(mod: Any, pool: MemoryConnectionPool) -> None:
    if getattr(mod, "_chdb_pool_patched", False):
        return
    mod._chdb_pool_patched = True

    Session = mod.Session
    orig_query = Session.query
    orig_send_query = Session.send_query
    orig_send_insert = getattr(Session, "send_insert", None)

    def session_query(self, sql, fmt="CSV", udf_path="", params=None):
        if self._conn is None:
            from chdb._exceptions import ChdbError

            raise ChdbError("Session is closed.")
        return orig_query(self, sql, fmt=fmt, udf_path=udf_path, params=params)

    def session_send_query(self, sql, fmt="CSV", params=None):
        if self._conn is None:
            from chdb._exceptions import ChdbError

            raise ChdbError("Session is closed.")
        return orig_send_query(self, sql, fmt=fmt, params=params)

    Session.query = session_query
    Session.sql = session_query
    Session.send_query = session_send_query

    if orig_send_insert is not None:

        def session_send_insert(self, query, fmt="CSV"):
            if self._conn is None:
                from chdb._exceptions import ChdbError

                raise ChdbError("Session is closed.")
            return orig_send_insert(self, query, fmt=fmt)

        Session.send_insert = session_send_insert


class _SubpackagePatchLoader(importlib.abc.Loader):
    def __init__(self, orig_loader: Any, callback):
        self._orig_loader = orig_loader
        self._callback = callback

    def create_module(self, spec):
        if hasattr(self._orig_loader, "create_module"):
            return self._orig_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._orig_loader.exec_module(module)
        self._callback(module)

    def __getattr__(self, name: str):
        return getattr(self._orig_loader, name)


class _SubpackagePatchFinder(importlib.abc.MetaPathFinder):
    _TARGETS = ("chdb.state.sqlitelike", "chdb.session.state")

    def __init__(self, pool: MemoryConnectionPool):
        self._pool = pool
        self._reentering = threading.local()

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in self._TARGETS:
            return None
        if getattr(self._reentering, "active", False):
            return None

        self._reentering.active = True
        try:
            for finder in sys.meta_path:
                if finder is self or isinstance(finder, _SubpackagePatchFinder):
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    cb = (
                        (lambda m: _patch_sqlitelike(m, self._pool))
                        if fullname == "chdb.state.sqlitelike"
                        else (lambda m: _patch_session_state(m, self._pool))
                    )
                    spec.loader = _SubpackagePatchLoader(spec.loader, cb)
                    return spec
            return None
        finally:
            self._reentering.active = False


def install_subpackage_hooks(pool: MemoryConnectionPool) -> None:
    if not any(isinstance(f, _SubpackagePatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _SubpackagePatchFinder(pool))

    sqlitelike = sys.modules.get("chdb.state.sqlitelike")
    if sqlitelike is not None:
        _patch_sqlitelike(sqlitelike, pool)

    session_state = sys.modules.get("chdb.session.state")
    if session_state is not None:
        _patch_session_state(session_state, pool)
