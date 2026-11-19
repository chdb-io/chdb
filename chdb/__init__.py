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
    import importlib.machinery

    _RTLD_DEEPBIND = 0x00008
    for candidate_dir in [_this_dir] + [p for p in __path__ if p != _this_dir]:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_path = os.path.join(candidate_dir, "_chdb" + suffix)
            if os.path.exists(ext_path):
                try:
                    ctypes.CDLL(ext_path, mode=sys.getdlopenflags() | _RTLD_DEEPBIND)
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

from ._pool import (  # noqa: E402
    MemoryConnectionPool as _MemoryConnectionPool,
    classify_sql as _classify_sql,
    install_subpackage_hooks as _install_subpackage_hooks,
    is_memory_path as _is_memory_path,
)

_memory_pool = _MemoryConnectionPool(_chdb)
_install_subpackage_hooks(_memory_pool)

if callable(getattr(_chdb, "engine_version", None)):
    engine_version = str(_chdb.engine_version())


# Close pooled `:memory:` handles and any user-owned Python connections at
# interpreter exit before C++ static destruction runs.
def _close_open_connections():
    _memory_pool.shutdown()

    arrow_mod = sys.modules.get("chdb.arrow")
    if arrow_mod is not None:
        arrow_session_cls = getattr(arrow_mod, "ArrowSession", None)
        if arrow_session_cls is not None:
            for obj in [o for o in gc.get_objects() if isinstance(o, arrow_session_cls)]:
                try:
                    obj.close()
                except Exception:
                    pass

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


atexit.register(_close_open_connections)

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


g_conn_lock = threading.RLock()
_udf_lock = threading.Lock()

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
    params = params or {}
    options = dict(options or {})
    with _udf_lock:
        if udf_path != "":
            g_udf_path = udf_path
        active_udf_path = g_udf_path
    conn_str = ":memory:" if path == "" else f"{path}"
    if active_udf_path != "":
        options["udf_path"] = active_udf_path
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

    if progress_mode != "auto" and _is_memory_path(conn_str):
        conn, gen = _memory_pool.borrow()
        sql_class = _classify_sql(conn, sql)
        had_error = False
        try:
            if lower_output_format in _df_format:
                res = conn.query_df(sql, params=params)
            else:
                res = conn.query(sql, output_format, params=params)
        except BaseException as e:
            had_error = True
            if isinstance(e, RuntimeError) and not isinstance(e, ChdbError):
                raise ChdbError(str(e)) from e.with_traceback(None)
            raise
        else:
            if lower_output_format not in _df_format and res.has_error():
                had_error = True
                raise ChdbError(res.error_message())
            return result_func(res)
        finally:
            _memory_pool.release(conn, gen, sql_class=sql_class, had_error=had_error)

    _memory_pool.prepare_for_connection(conn_str)
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


def __getattr__(name):
    if name == "engine_version":
        ver = _memory_pool.resolve_engine_version(core_version)
        globals()["engine_version"] = ver
        return ver
    if name in _lazy_submodules:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _udf_decorators:
        decorator = _import_udf_decorator(name)
        if decorator is not None:
            globals()[name] = decorator
            return decorator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | _lazy_submodules | {"engine_version"})


def connect(*args, **kwargs):
    from .state import connect as _connect

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
