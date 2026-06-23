"""
The ``client.chdb`` extension namespace.

chDB exposes capabilities that a remote ClickHouse server does not: querying in-process
Python objects through the ``Python()`` table function, Python UDFs, the native DB-API
cursor, and the on-disk vs in-memory session path. Per the design proposal these must NOT
be bolted onto clickhouse-connect's base ``Client`` (which would bloat the shared
interface); instead they live behind a namespace that exists *only* on a chDB client::

    client = clickhouse_connect.get_client(backend="chdb", path=":memory:")
    df = client.chdb.query_python("SELECT sum(a) FROM Python(my_df)", my_df=my_df)
    cur = client.chdb.cursor()
    path = client.chdb.session_path

On an HTTP client there is no ``.chdb`` attribute, so accessing it raises ``AttributeError``
-- exactly the intended behavior. This module is owned and versioned by the chDB team; new
chDB-only features are added here without any clickhouse-connect change.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas

    from chdb.cc_backend import ChdbClient

# Guards the brief window during which named frames are published into module globals so
# chDB's Python() table function can resolve them by name off the calling frame.
_python_tablefunc_lock = threading.Lock()


class ChdbExtension:
    """chDB-only API surface, reachable as ``client.chdb``."""

    def __init__(self, client: "ChdbClient"):
        self._client = client

    @property
    def session_path(self) -> str:
        """The chDB session path: ``":memory:"`` for in-memory, else the on-disk directory."""
        return self._client._chdb_path

    @property
    def connection(self):
        """The underlying chDB connection (escape hatch for advanced use)."""
        return self._client.chdb_connection

    def cursor(self):
        """Return chDB's native DB-API cursor for the underlying connection.

        This is the escape hatch to chDB's own cursor semantics; it bypasses
        clickhouse-connect's result machinery entirely.
        """
        return self._client.chdb_connection.cursor()

    def query_python(self, sql: str, fmt: str = "DataFrame", **frames: Any) -> "pandas.DataFrame | Any":
        """Run a query that references in-process Python objects via the ``Python()`` table function.

        Pass the objects (pandas DataFrames, Arrow tables, ...) as keyword arguments whose
        names match the identifiers used inside ``Python(<name>)`` in the SQL. The data is
        read in-process with no serialization round-trip::

            df = client.chdb.query_python("SELECT b, sum(a) FROM Python(t) GROUP BY b", t=my_df)

        :param sql: SQL referencing the frames through ``Python(<name>)``.
        :param fmt: ``"DataFrame"`` (default) returns a pandas DataFrame via the zero-copy
            Arrow path; any other value is passed to chDB as the output format and the raw
            result is returned.
        :param frames: name -> Python object bindings visible to the ``Python()`` table function.
        """
        conn = self._client.chdb_connection
        module_globals = globals()
        with _python_tablefunc_lock:
            # chDB's Python() table function resolves a name off the calling frame; this
            # method's f_globals is this module, so publish the frames here for the call.
            clashes = {name: module_globals[name] for name in frames if name in module_globals}
            module_globals.update(frames)
            try:
                if fmt == "DataFrame":
                    import chdb

                    res = conn.query(sql, "Arrow")
                    table = chdb.to_arrowTable(res)
                    return table.to_pandas()
                result = conn.query(sql, fmt)
                return result
            except Exception as ex:  # noqa: BLE001
                raise self._client.map_error(ex) from ex
            finally:
                for name in frames:
                    module_globals.pop(name, None)
                module_globals.update(clashes)

    def register_function(self, func=None, *, return_type: str = "String"):
        """Register a Python UDF usable from SQL on this client.

        Wraps chDB's ``chdb_udf`` decorator. Can be used directly on an already-decorated
        function, or as a decorator factory::

            @client.chdb.register_function(return_type="Int32")
            def plus_one(x):
                return int(x) + 1

        Note: chDB derives the UDF from the function's source, so ``func`` must be a
        module-level function whose source is importable (not a lambda or REPL closure).
        """
        from chdb.udf import chdb_udf

        def _wrap(fn):
            decorated = fn if getattr(fn, "_chdb_udf", False) else chdb_udf(return_type=return_type)(fn)
            self._client._integration_libs.add(f"udf:{getattr(fn, '__name__', 'anon')}")
            return decorated

        if func is not None:
            return _wrap(func)
        return _wrap
