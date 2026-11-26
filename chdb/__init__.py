"""chdb - in-process OLAP SQL Engine powered by ClickHouse.

This __init__.py ensures chdb-core's engine is properly initialized regardless
of install method (regular pip install, editable install, or upgrade path).

When both chdb (wrapper) and chdb-core (engine) are installed, this file
takes precedence and bootstraps the engine from chdb-core's _chdb extension.
"""

import sys
import os
import threading
import importlib
import atexit
import gc

_this_dir = os.path.dirname(os.path.abspath(__file__))

# For editable installs the local chdb/ directory may shadow chdb-core's
# site-packages directory.  Ensure the latter is on __path__ so that
# `from . import _chdb` and subpackage imports resolve correctly.
for _sp in sys.path:
    if not _sp or not os.path.isdir(_sp):
        continue
    _core_dir = os.path.join(_sp, "chdb")
    if (
        os.path.isdir(_core_dir)
        and os.path.normpath(_core_dir) != os.path.normpath(_this_dir)
        and _core_dir not in __path__
    ):
        __path__.append(_core_dir)


# chdb-core keeps ChdbError and the polars conversion out of its own
# __init__.py, which this file shadows, so both are shared with the engine.
from ._exceptions import ChdbError  # noqa: E402
from .polars_io import to_polars, _require_polars  # noqa: E402

_arrow_format = set({"arrowtable"})
_polars_format = set({"polars"})
_df_format = set({"dataframe", "datastore"})
_process_result_format_funs = {
    "arrowtable": lambda x: to_arrowTable(x),
    "datastore": lambda x: to_datastore(x),
    "polars": lambda x: to_polars(x),
}

g_udf_path = ""

__version__ = "unknown"
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("chdb")
except Exception:
    pass

try:
    core_version = importlib.metadata.version("chdb-core")
except Exception:
    core_version = "unknown"

__chdb_core_version__ = core_version


def _preload_chdb_with_deepbind():
    if sys.platform != "linux":
        return
    import ctypes
    _RTLD_DEEPBIND = 0x00008
    for candidate_dir in [_this_dir] + [p for p in __path__ if p != _this_dir]:
        _abi3 = os.path.join(candidate_dir, "_chdb.abi3.so")
        if os.path.exists(_abi3):
            try:
                ctypes.CDLL(_abi3, mode=sys.getdlopenflags() | _RTLD_DEEPBIND)
            except OSError:
                pass
            return


if sys.version_info[:2] >= (3, 7):
    _preload_chdb_with_deepbind()
    try:
        _chdb = importlib.import_module(__name__ + "._chdb")
    except ModuleNotFoundError as e:
        if e.name != __name__ + "._chdb":
            raise
        raise ImportError(
            "chdb-core engine not found. Install it with: pip install chdb-core"
        ) from e
else:
    raise NotImplementedError("Python 3.6 or lower version is not supported")

from . import _fork  # noqa: E402

_fork.install(sys.modules[__name__], _chdb)


# chdb-core up to 26.9 tears down an engine still open at process exit during
# C++ static destruction, after the thread pools it needs are destroyed (heap
# corruption and SIGABRT on macOS, chdb-io/chdb-core#254). Python's atexit
# handlers run before that, so close whatever is still open there. Every
# long-lived engine connection (chdb.connect(), Session, dbapi, DataStore,
# durable) is owned by a chdb.state.sqlitelike.Connection, and instances of
# Python classes are always tracked by the garbage collector, so they can be
# found without wrapping anything in chdb-core. Not needed once
# chdb-core >= 26.11 is required.
def _close_open_connections():
    sqlitelike = sys.modules.get("chdb.state.sqlitelike")
    if sqlitelike is None:
        return
    connection_class = sqlitelike.Connection
    open_connections = [obj for obj in gc.get_objects() if issubclass(type(obj), connection_class)]
    for conn in open_connections:
        try:
            conn.close()
        except Exception:
            pass


def _on_process_exit():
    if _fork.should_fast_exit_on_shutdown():
        _fork.fast_exit_child()
        return
    if _fork.is_forked_after_init():
        return
    _close_open_connections()


atexit.register(_on_process_exit)

chdb_version = tuple(__version__.split("."))


def to_arrowTable(res):
    """Convert query result to PyArrow Table."""
    try:
        import pyarrow as pa
        import pandas as pd  # noqa
    except ImportError:
        print('Please install pyarrow and pandas via "pip install pyarrow pandas"')
        raise ImportError("Failed to import pyarrow or pandas") from None
    if len(res) == 0:
        return pa.Table.from_batches([], schema=pa.schema([]))
    memview = res.get_memview()
    return pa.RecordBatchFileReader(memview.view()).read_all()


def to_datastore(df):
    """Wrap a pandas DataFrame in a chdb DataStore.

    Requires pandas and pyarrow (declared dependencies of the ``chdb`` pip
    package, but absent on ``pip install --no-deps`` installs).
    """
    try:
        from chdb.datastore import DataStore
    except ImportError as e:
        raise ImportError(
            'DataStore output format requires pandas and pyarrow. '
            'Install them via "pip install pandas pyarrow".'
        ) from e
    return DataStore(df)


g_conn_lock = threading.Lock()
# The lock serializes module-level query startup and cleanup. A batch stream
# keeps the engine busy after startup, so reject overlapping calls rather than
# holding this non-reentrant lock across user code.
g_arrow_stream_active = threading.Event()

from .progress_display import (  # noqa: E402
    is_notebook as _is_notebook,
    create_auto_progress_callback as _create_auto_progress_callback,
)


def query(sql, output_format="CSV", path="", udf_path="", params=None, options=None):
    """Execute SQL query using chDB engine.

    Args:
        sql: SQL query string to execute
        output_format: Output format (CSV, JSON, Arrow, Parquet, DataFrame, ArrowTable,
            Polars, Debug)
        path: Database file path ("" for in-memory, or a file path)
        udf_path: Path to User-Defined Functions directory
        params: Named query parameters matching placeholders like {key:Type}
        options: Connection options passed to ClickHouse as startup arguments

    Returns:
        Query result in the specified format.

    Raises:
        ChdbError: If the SQL query execution fails
    """
    global g_udf_path
    _fork.check_fork_safety()
    params = params or {}
    options = dict(options or {})
    if udf_path != "":
        g_udf_path = udf_path
    conn_str = ":memory:" if path == "" else f"{path}"
    if g_udf_path != "":
        options["udf_path"] = g_udf_path
    if output_format == "Debug":
        output_format = "CSV"
        options.setdefault("verbose", "")
        options.setdefault("log-level", "test")
    progress_mode = options.get("progress")
    if isinstance(progress_mode, str):
        progress_mode = progress_mode.lower()
    if progress_mode == "auto":
        options.pop("progress", None)
        if not _is_notebook() and (sys.stdout.isatty() or sys.stderr.isatty()):
            options["progress"] = "tty"
    if options:
        parts = []
        for key, value in options.items():
            if value == "":
                parts.append(f"{key}")
            else:
                parts.append(f"{key}={value}")
        conn_str = f"{conn_str}?{'&'.join(parts)}"

    lower_output_format = output_format.lower()
    result_func = _process_result_format_funs.get(lower_output_format, lambda x: x)
    if lower_output_format in _polars_format:
        # Fail before running the query, like Connection.query does.
        _require_polars(f'output format "{output_format}"')
    if lower_output_format in _arrow_format or lower_output_format in _polars_format:
        output_format = "Arrow"

    with g_conn_lock:
        _fork.check_fork_safety()
        if g_arrow_stream_active.is_set():
            raise RuntimeError("Finish the active Arrow batch stream first")
        with _fork.native_operation():
            conn = _chdb.connect(conn_str)
            progress_callback = None
            if progress_mode == "auto":
                progress_callback = _create_auto_progress_callback()
                if progress_callback is not None:
                    conn.set_progress_callback(progress_callback)

            try:
                if lower_output_format in _df_format:
                    res = conn.query_df(sql, params=params)
                else:
                    res = conn.query(sql, output_format, params=params)
            except RuntimeError as e:
                if isinstance(e, ChdbError):
                    raise
                raise ChdbError(str(e)) from e.with_traceback(None)
            else:
                if lower_output_format not in _df_format and res.has_error():
                    raise ChdbError(res.error_message())
                return result_func(res)
            finally:
                if progress_callback is not None:
                    progress_callback.close()
                    conn.set_progress_callback(None)
                conn.close()


sql = query

PyReader = _chdb.PyReader

# UDF/UDAF module-level surfaces. Mirrored from chdb-core but guarded with
# hasattr so chdb keeps working against older chdb-core releases: the scalar
# names pre-date the `promote create_function to module-level API` refactor
# (e.g. 26.3.0), and the aggregate ones were added later still.
_udf_exports = []
for _name in (
    "create_function",
    "drop_function",
    "create_aggregate_function",
    "drop_aggregate_function",
    "NullHandling",
    "ExceptionHandling",
    "UDFType",
):
    if hasattr(_chdb, _name):
        globals()[_name] = getattr(_chdb, _name)
        _udf_exports.append(_name)

_lazy_submodules = {"agents", "arrow", "dbapi", "session", "udf", "utils"}

# Decorators re-exported from chdb.udf. `agg` is absent on chdb-core releases
# that pre-date Python UDAF support.
_udf_decorators = ("func", "agg")


def _import_udf_decorator(name):
    """Return ``chdb.udf.<name>``, or None when this chdb-core build lacks it."""
    try:
        return getattr(importlib.import_module(f"{__name__}.udf"), name)
    except (ImportError, AttributeError):
        return None


def _resolve_engine_version():
    native_version = getattr(_chdb, "engine_version", None)
    if callable(native_version):
        return str(native_version())
    _fork.check_fork_safety()
    with _fork.native_operation():
        try:
            conn = _chdb.connect()
        except RuntimeError as e:
            import re

            match = re.search(r"already initialized with path '([^']+)'", str(e))
            if not match:
                raise
            conn = _chdb.connect(match.group(1))
        try:
            return conn.query("SELECT version();", "CSV").bytes().decode("utf-8").strip().strip('"')
        finally:
            conn.close()


def __getattr__(name):
    if name == "engine_version":
        version = _resolve_engine_version()
        globals()["engine_version"] = version
        return version
    if name in _lazy_submodules:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        _fork.ensure_sqlitelike_patched()
        return module
    if name in _udf_decorators:
        decorator = _import_udf_decorator(name)
        if decorator is not None:
            globals()[name] = decorator
            return decorator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


def connect(*args, **kwargs):
    _fork.check_fork_safety()
    from .state import connect as _connect

    _fork.ensure_sqlitelike_patched()
    return _connect(*args, **kwargs)

for _name in _udf_decorators:
    _decorator = _import_udf_decorator(_name)
    if _decorator is not None:
        globals()[_name] = _decorator
        _udf_exports.append(_name)


def query_arrow(sql, *, path="", params=None, settings=None, options=None):
    """Return a PyArrow table via chDB's direct C Data result path.

    Unlike ``query(sql, 'ArrowTable')``, this does not encode Arrow IPC.
    Requires a chdb-core build with the native Python Arrow bridge.
    """
    from .arrow import query_arrow as _query_arrow

    return _query_arrow(sql, path=path, params=params, settings=settings, options=options)


def query_arrow_batches(sql, *, path="", params=None, settings=None, options=None):
    """Return a closable iterator of PyArrow batches from a streaming query."""
    from .arrow import query_arrow_batches as _query_arrow_batches

    return _query_arrow_batches(sql, path=path, params=params, settings=settings, options=options)

__all__ = [
    "_chdb",
    "PyReader",
    "ChdbError",
    "query",
    "query_arrow",
    "query_arrow_batches",
    "sql",
    "chdb_version",
    "engine_version",
    "to_arrowTable",
    "to_datastore",
    "to_polars",
    "dbapi",
    "session",
    "udf",
    "utils",
    "connect",
    "arrow",
    "agents",
] + _udf_exports
