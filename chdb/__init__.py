"""chdb - in-process OLAP SQL Engine powered by ClickHouse.

This __init__.py ensures chdb-core's engine is properly initialized regardless
of install method (regular pip install, editable install, or upgrade path).

When both chdb (wrapper) and chdb-core (engine) are installed, this file
takes precedence and bootstraps the engine from chdb-core's _chdb extension.
"""

import sys
import os
import threading

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


class ChdbError(Exception):
    """Exception raised when chDB query execution fails."""


_arrow_format = set({"arrowtable"})
_process_result_format_funs = {
    "arrowtable": lambda x: to_arrowTable(x),
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

if sys.version_info[:2] >= (3, 7):
    cwd = os.getcwd()
    os.chdir(_this_dir)
    try:
        from . import _chdb  # noqa
    except ImportError:
        # _chdb not in local dir; search __path__ entries
        _found = False
        for _p in __path__:
            if _p == _this_dir:
                continue
            os.chdir(_p)
            try:
                from . import _chdb  # noqa

                _found = True
                break
            except ImportError:
                continue
        if not _found:
            os.chdir(cwd)
            raise ImportError(
                "chdb-core engine not found. Install it with: pip install chdb-core"
            )
    os.chdir(cwd)
    conn = _chdb.connect()
    engine_version = str(conn.query("SELECT version();", "CSV").bytes())[3:-4]
    conn.close()
else:
    raise NotImplementedError("Python 3.6 or lower version is not supported")

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


g_conn_lock = threading.Lock()

from .progress_display import (  # noqa: E402
    is_notebook as _is_notebook,
    create_auto_progress_callback as _create_auto_progress_callback,
)


def query(sql, output_format="CSV", path="", udf_path="", params=None, options=None):
    """Execute SQL query using chDB engine.

    Args:
        sql: SQL query string to execute
        output_format: Output format (CSV, JSON, Arrow, Parquet, DataFrame, ArrowTable, Debug)
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
    if lower_output_format in _arrow_format:
        output_format = "Arrow"

    with g_conn_lock:
        conn = _chdb.connect(conn_str)
        progress_callback = None
        if progress_mode == "auto":
            progress_callback = _create_auto_progress_callback()
            if progress_callback is not None:
                conn.set_progress_callback(progress_callback)

        try:
            if lower_output_format == "dataframe":
                res = conn.query_df(sql, params=params)
                return res

            res = conn.query(sql, output_format, params=params)

            if res.has_error():
                raise ChdbError(res.error_message())
            return result_func(res)
        finally:
            if progress_callback is not None:
                progress_callback.close()
                conn.set_progress_callback(None)
            conn.close()


sql = query

PyReader = _chdb.PyReader

from . import dbapi, session, udf, utils  # noqa: E402
from .state import connect  # noqa: E402

__all__ = [
    "_chdb",
    "PyReader",
    "ChdbError",
    "query",
    "sql",
    "chdb_version",
    "engine_version",
    "to_arrowTable",
    "dbapi",
    "session",
    "udf",
    "utils",
    "connect",
]
