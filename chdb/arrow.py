"""Direct Arrow C Data query helpers for the embedded chDB Python binding.

This module requires chdb-core 26.11.1 or newer for native Arrow query methods.
The existing ``query(..., 'ArrowTable')`` IPC interface remains unchanged.
"""

from __future__ import annotations

import threading
import uuid
import weakref
from typing import Any, Callable, Mapping, Sequence

from . import ChdbError, _chdb, g_arrow_stream_active, g_conn_lock


def _connection_string(path: str, settings: Mapping[str, Any] | None) -> str:
    conn_str = path or ":memory:"
    if settings:
        parts = [key if value == "" else f"{key}={value}" for key, value in settings.items()]
        conn_str += "?" + "&".join(parts)
    return conn_str


def _native_connection(path: str, settings: Mapping[str, Any] | None):
    conn = _chdb.connect(_connection_string(path, settings))
    if not all(callable(getattr(conn, name, None)) for name in (
        "query_arrow_reader", "send_query_arrow", "streaming_fetch_arrow"
    )):
        conn.close()
        raise RuntimeError(
            "Direct Arrow queries require chdb-core>=26.11.1 with "
            "query_arrow_reader/send_query_arrow; use ArrowTable for older builds"
        )
    return conn


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not identifier or "\x00" in identifier:
        raise ValueError("table and column names must be non-empty identifiers")
    return "`" + identifier.replace("\\", "\\\\").replace("`", "\\`") + "`"


class ArrowBatchStream:
    """Closable iterator of ``pyarrow.RecordBatch`` values from a native query.

    A stream owns its query handle, not the caller's session. Consuming only
    part of it requires ``close`` (or a ``with`` block) to cancel promptly.
    """

    def __init__(self, conn, result, on_close: Callable[[], None] | None = None):
        self._conn = conn
        self._result = result
        self._on_close = on_close
        self._lock = threading.RLock()
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._closed:
                raise StopIteration
            try:
                reader = self._conn.streaming_fetch_arrow(self._result)
                try:
                    batch = reader.read_next_batch()
                finally:
                    reader.close()
            except StopIteration:
                self.close()
                raise
            except BaseException:
                self.close()
                raise
            return batch

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self._result is not None:
                    self._conn.streaming_cancel_query(self._result)
            finally:
                self._result = None
                on_close, self._on_close = self._on_close, None
                if on_close is not None:
                    on_close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class ArrowSource:
    """A query-scoped name for an in-memory Arrow source.

    The Python table function resolves the opaque name in this module's
    globals. Closing during an active batch stream defers cleanup until that
    stream ends, so the producer remains alive for every native fetch.
    """

    def __init__(self, session: "ArrowSession", source):
        self._session = weakref.ref(session)
        self._name = "__chdb_arrow_" + uuid.uuid4().hex
        self._pending = False
        self._closed = False
        globals()[self._name] = source

    @property
    def sql_expr(self) -> str:
        if self._closed or self._pending:
            raise RuntimeError("ArrowSource is closed")
        return f"Python('{self._name}')"

    def _release(self):
        if not self._closed:
            globals().pop(self._name, None)
            self._closed = True
            session = self._session()
            if session is not None:
                session._sources.discard(self)

    def close(self):
        if self._closed:
            return
        session = self._session()
        if session is not None and session._active is not None:
            self._pending = True
            return
        self._release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class ArrowSession:
    """A chDB session with typed, non-IPC Arrow output.

    Keep a session on one calling thread and finish an active batch stream
    before starting another query. Python ``FROM Python(tbl)`` is still the
    in-memory Arrow input path; no extra IPC file is involved.
    """

    def __init__(self, path: str = "", *, settings: Mapping[str, Any] | None = None):
        self._conn = _native_connection(path, settings)
        self._active: ArrowBatchStream | None = None
        self._sources: set[ArrowSource] = set()
        self._closed = False

    def _ready(self):
        if self._closed:
            raise RuntimeError("ArrowSession is closed")
        if self._active is not None:
            raise RuntimeError("Finish the active Arrow batch stream first")

    def query_arrow(self, sql: str, *, params: Mapping[str, Any] | None = None,
                    options: Mapping[str, bool] | None = None):
        """Collect a native C Data query result as ``pyarrow.Table``."""
        self._ready()
        reader = self._conn.query_arrow_reader(sql, params=dict(params or {}), options=dict(options or {}))
        try:
            return reader.read_all()
        finally:
            reader.close()

    def register_arrow(self, source) -> ArrowSource:
        """Expose a PyArrow table or one-shot reader to `Python(source)` SQL."""
        self._ready()
        import pyarrow as pa

        if isinstance(source, pa.RecordBatch):
            source = pa.Table.from_batches([source])
        if not isinstance(source, pa.Table) and not (
            hasattr(source, "__arrow_c_stream__") and hasattr(source, "schema")
        ):
            raise TypeError("source must be a PyArrow table, batch, or Arrow C Stream producer with a schema")
        registration = ArrowSource(self, source)
        self._sources.add(registration)
        return registration

    def query_arrow_batches(self, sql: str, *, params: Mapping[str, Any] | None = None,
                            options: Mapping[str, bool] | None = None) -> ArrowBatchStream:
        """Run an incremental native C Data query; close on partial reads."""
        self._ready()
        result = self._conn.send_query_arrow(sql, params=dict(params or {}), options=dict(options or {}))

        owner = weakref.ref(self)

        def done():
            session = owner()
            if session is not None:
                session._active = None
                for source in tuple(session._sources):
                    if source._pending:
                        source._release()

        stream = ArrowBatchStream(self._conn, result, done)
        self._active = stream
        return stream

    def query(self, sql: str, output_format: str = "CSV", *, params: Mapping[str, Any] | None = None):
        """Run an existing chDB output format on this same session."""
        self._ready()
        result = self._conn.query(sql, output_format, params=dict(params or {}))
        if result.has_error():
            raise ChdbError(result.error_message())
        return result

    def execute(self, sql: str, *, params: Mapping[str, Any] | None = None):
        """Execute SQL without a result table on this same session."""
        return self.query(sql, "Null", params=params)

    def insert_arrow(self, table: str | Sequence[str], source, *, columns: Sequence[str] | None = None) -> int:
        """Insert a PyArrow table/reader through the in-memory Python source.

        The destination must already exist. A reader is consumed once; partial
        insert on error follows ordinary ClickHouse INSERT semantics.
        """
        self._ready()
        import pyarrow as pa

        if isinstance(source, pa.RecordBatch):
            source = pa.Table.from_batches([source])
        if not isinstance(source, pa.Table) and not (
            hasattr(source, "__arrow_c_stream__") and hasattr(source, "schema")
        ):
            raise TypeError("source must be a PyArrow table, batch, or Arrow C Stream producer with a schema")
        table_parts = (table,) if isinstance(table, str) else tuple(table)
        if not 1 <= len(table_parts) <= 2:
            raise ValueError("table must be a name or (database, name)")
        destination = ".".join(_quote_identifier(part) for part in table_parts)
        source_names = tuple(source.schema.names)
        selected = tuple(columns) if columns is not None else source_names
        if not selected or len(selected) != len(set(selected)) or any(name not in source_names for name in selected):
            raise ValueError("columns must be distinct names in the Arrow schema")
        projection = ", ".join(_quote_identifier(name) for name in selected)
        with self.register_arrow(source) as registration:
            sql = f"INSERT INTO {destination} ({projection}) SELECT {projection} FROM {registration.sql_expr}"
            result = self.execute(sql)
        return result.rows_written()

    def close(self):
        if self._closed:
            return
        try:
            if self._active is not None:
                self._active.close()
        finally:
            try:
                for source in tuple(self._sources):
                    source._release()
            finally:
                self._conn.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def connect(path: str = "", *, settings: Mapping[str, Any] | None = None) -> ArrowSession:
    return ArrowSession(path, settings=settings)


def query_arrow(sql: str, *, path: str = "", params: Mapping[str, Any] | None = None,
                settings: Mapping[str, Any] | None = None, options: Mapping[str, bool] | None = None):
    """Execute one query and return a ``pyarrow.Table`` without Arrow IPC."""
    with g_conn_lock:
        if g_arrow_stream_active.is_set():
            raise RuntimeError("Finish the active Arrow batch stream first")
        with connect(path, settings=settings) as session:
            return session.query_arrow(sql, params=params, options=options)


def query_arrow_batches(sql: str, *, path: str = "", params: Mapping[str, Any] | None = None,
                        settings: Mapping[str, Any] | None = None,
                        options: Mapping[str, bool] | None = None) -> ArrowBatchStream:
    """Return a closable batch iterator; use ``with`` for early termination."""
    with g_conn_lock:
        if g_arrow_stream_active.is_set():
            raise RuntimeError("Finish the active Arrow batch stream first")
        conn = None
        try:
            conn = _native_connection(path, settings)
            result = conn.send_query_arrow(sql, params=dict(params or {}), options=dict(options or {}))

            def done():
                with g_conn_lock:
                    try:
                        conn.close()
                    finally:
                        g_arrow_stream_active.clear()

            stream = ArrowBatchStream(conn, result, done)
            g_arrow_stream_active.set()
            return stream
        except BaseException:
            try:
                if conn is not None:
                    conn.close()
            finally:
                g_arrow_stream_active.clear()
            raise
