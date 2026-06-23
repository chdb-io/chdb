"""
chDB execution backend for clickhouse-connect.

This module makes chDB's in-process, embedded ClickHouse engine usable as a
clickhouse-connect backend::

    import clickhouse_connect
    client = clickhouse_connect.get_client(backend="chdb", path=":memory:")
    df = client.query_df("SELECT number FROM numbers(5)")

It lives in the chdb-io/chdb repository -- *not* in clickhouse-connect -- and registers
itself through the ``clickhouse_connect.backends`` entry point (see ``pyproject.toml``).
clickhouse-connect contains no chDB-specific code; the only coupling is the ``Backend``
factory contract in ``clickhouse_connect.driver.backend`` and the entry-point name
``chdb``. Everything here is owned and versioned by the chDB team, so chDB engine changes
never require a clickhouse-connect change.

``ChdbClient`` is a thin subclass of clickhouse-connect's ``Client``: it overrides only the
transport-shaped methods (``raw_query`` / ``raw_stream`` / ``raw_insert`` / ``command`` /
``_query_with_context``) plus the Arrow fast paths, and inherits the entire public API,
type system, settings handling, Native parser, and error model unchanged. The same Native
byte format the HTTP server emits is consumed verbatim, so result conversion is reused.

Capability: ``supports_zero_copy_arrow = True``. ``query_arrow`` / ``query_arrow_stream``
return PyArrow objects backed directly by chDB's in-process Arrow buffers (no wire bytes,
no socket), and ``query_df`` routes through that zero-copy Arrow path. chDB-only features
(the ``Python()`` table function, Python UDFs, the native DB-API cursor, the session path)
are exposed through the ``client.chdb`` extension namespace (see ``cc_extension.py``), not
bolted onto the base ``Client``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import weakref
import tempfile
import threading
from collections.abc import Generator, Iterable, Sequence
from datetime import tzinfo
from typing import TYPE_CHECKING, Any, BinaryIO

from clickhouse_connect import common
from clickhouse_connect.datatypes.registry import get_from_name
from clickhouse_connect.driver.binding import bind_query, quote_identifier
from clickhouse_connect.driver.client import Client
from clickhouse_connect.driver.common import StreamContext, coerce_int
from clickhouse_connect.driver.ctypes import RespBuffCls
from clickhouse_connect.driver.exceptions import (
    DatabaseError,
    NotSupportedError,
    ProgrammingError,
    StreamFailureError,
)
from clickhouse_connect.driver.external import ExternalData
from clickhouse_connect.driver.insert import InsertContext
from clickhouse_connect.driver.options import check_arrow
from clickhouse_connect.driver.query import QueryContext, QueryResult, TzMode, TzSource
from clickhouse_connect.driver.summary import QuerySummary
from clickhouse_connect.driver.transform import NativeTransform

if TYPE_CHECKING:
    import numpy
    import pandas
    import polars
    import pyarrow

logger = logging.getLogger(__name__)

_columns_only_re = re.compile(r"LIMIT 0\s*$", re.IGNORECASE)

# Pattern a ClickHouse setting name must match before we are willing to interpolate it into
# a ``SET <name> = <value>`` statement (see ChdbClient._validate_setting_name).
_SETTING_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_sql_string(text: str) -> str:
    """Single-quote a string literal for chDB SQL, escaping backslashes and quotes.

    Used for paths interpolated into ``INSERT INTO ... FROM INFILE '<path>'`` clauses --
    ``tempfile.NamedTemporaryFile`` respects the platform's TMPDIR which may contain
    apostrophes on user machines (typical on macOS when the temp dir is under
    ``/var/folders/.../T/`` -- but also possible under a user-set TMPDIR), so the path
    must be escaped the same way a value would be.
    """
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"

# chdb's `send_query` emits each ClickHouse block as a self-contained encoding in the
# requested format. For formats that have row-level (or block-level) self-description
# and no global header/footer/file structure, concatenating chunks yields a valid
# stream the caller's parser can consume directly. Other formats (Arrow, Parquet,
# JSON, *WithNames variants, ...) would emit duplicated headers / multiple file
# markers per chunk, which is not a valid larger stream. For those we fall back to a
# single non-streaming query so the result is one well-formed payload.
_STREAM_SAFE_FORMATS = frozenset(
    {
        "Native",
        "TabSeparated",
        "TSV",
        "CSV",
        "RowBinary",
        "JSONEachRow",
    }
)


def _finalize_stream(sr, lock, released_box):
    """Idempotently close a chdb StreamingResult and release the per-connection lock.

    Used both by an explicit ``close()`` call and by a ``weakref.finalize`` registered on
    the stream wrapper, so a stream object that is GC'd without being closed (e.g. test
    grabs the stream and then errors out) still releases the lock. Without this, a leaked
    stream holds the shared connection lock forever and every subsequent query on any
    ``ChdbClient`` using the same path deadlocks at ``with self._lock:``.

    ``released_box`` is a single-element list shared with the wrapper instance so both
    pathways agree on "already cleaned up".
    """
    if released_box[0]:
        return
    released_box[0] = True
    try:
        close = getattr(sr, "close", None)
        if close is not None:
            close()
    except Exception:  # noqa: BLE001
        logger.debug("Error closing chdb StreamingResult during finalize", exc_info=True)
    try:
        lock.release()
    except RuntimeError:
        pass


class _BytesSource:
    """
    Minimal stand-in for the HTTP `ResponseSource` that the response buffer
    expects. Yields a single chunk of bytes and exposes the attributes the
    transform layer reads.
    """

    __slots__ = ("data", "last_message", "exception_tag")

    def __init__(self, data: bytes):
        self.data = data
        self.last_message = None
        self.exception_tag = None

    @property
    def gen(self):
        def _gen():
            yield self.data

        return _gen()

    def close(self):
        return None


class _ChdbStreamSource:
    """
    Source for `ResponseBuffer` backed by a chdb `StreamingResult`. Yields each
    block's bytes and translates chdb's mid-stream RuntimeError into the
    `StreamFailureError` clickhouse-connect callers expect.
    """

    __slots__ = ("_sr", "_released", "_finalizer", "last_message", "exception_tag", "__weakref__")

    def __init__(self, streaming_result, lock: threading.Lock):
        self._sr = streaming_result
        self._released = [False]
        # weakref.finalize guarantees lock release even if close() is never called (e.g. the
        # stream is GC'd because a test errored out mid-iteration). See _finalize_stream.
        self._finalizer = weakref.finalize(self, _finalize_stream, streaming_result, lock, self._released)
        self.last_message = None
        self.exception_tag = None

    @property
    def gen(self):
        def _gen():
            try:
                while True:
                    try:
                        chunk = next(self._sr)
                    except StopIteration:
                        return
                    except Exception as ex:  # noqa: BLE001
                        raise StreamFailureError(_format_error_message(str(ex))) from ex
                    payload = chunk.bytes() if hasattr(chunk, "bytes") else bytes(chunk)
                    if payload:
                        yield payload
            finally:
                self.close()

        return _gen()

    def close(self):
        # Detach the finalizer and run the same cleanup directly (idempotent via the box).
        self._finalizer()


def _format_error_message(message: str) -> str:
    """Extract a clean ClickHouse exception message from a chdb error string."""
    if not message:
        return ""
    idx = message.find("Code: ")
    if idx > 0:
        return message[idx:].strip()
    return message.strip()


def _drain_to_bytes(block) -> bytes:
    """Collect any supported insert_block shape into a single bytes value."""
    if isinstance(block, (bytes, bytearray, memoryview)):
        return bytes(block)
    if isinstance(block, str):
        return block.encode()
    if hasattr(block, "to_pybytes"):
        return block.to_pybytes()
    if hasattr(block, "read"):
        return block.read()
    parts = []
    for chunk in block:
        parts.append(chunk if isinstance(chunk, (bytes, bytearray)) else chunk.encode())
    return b"".join(parts)


def _decompress(data: bytes, encoding: str) -> bytes:
    if encoding == "lz4":
        import lz4.frame

        return lz4.frame.decompress(data)
    if encoding == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(data)
    if encoding == "gzip":
        import gzip

        return gzip.decompress(data)
    if encoding == "br":
        try:
            import brotli
        except ImportError as ex:
            raise NotSupportedError("brotli is required to decompress 'br' for chdb raw_insert") from ex
        return brotli.decompress(data)
    if encoding == "deflate":
        import zlib

        return zlib.decompress(data)
    raise NotSupportedError(f"Unsupported compression {encoding!r} for chdb raw_insert")


# Process-wide cache of chdb connections, keyed by connection-string.
#
# chdb-core's embedded engine permits only one active engine instance per process per data
# directory: a second ``chdb.connect("...")`` while a prior connection on the same path is
# still open deadlocks inside the C extension. Because clickhouse-connect's tests freely
# instantiate multiple clients (sync + async fixtures, parametrized factories, ...), the
# backend must respect this and refcount-share a single underlying connection per path.
# ``close()`` decrements; the chdb connection itself is only closed when the refcount drops
# to zero. The lock guards the cache itself, not individual queries — each ChdbClient still
# has its own per-instance ``_lock`` to serialize operations on the shared connection.
_CONN_CACHE: dict[str, list] = {}  # conn_str -> [Connection, refcount, threading.Lock]
_CONN_CACHE_LOCK = threading.Lock()


def _acquire_chdb_connection(conn_str: str):
    """Return (connection, per-connection-lock) for ``conn_str``, sharing across clients."""
    import chdb

    with _CONN_CACHE_LOCK:
        entry = _CONN_CACHE.get(conn_str)
        if entry is None:
            entry = [chdb.connect(conn_str), 0, threading.Lock()]
            _CONN_CACHE[conn_str] = entry
        entry[1] += 1
        return entry[0], entry[2]


def _release_chdb_connection(conn_str: str) -> None:
    """Decrement the refcount; really close the chdb connection when the last user releases it."""
    with _CONN_CACHE_LOCK:
        entry = _CONN_CACHE.get(conn_str)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            try:
                entry[0].close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing shared chdb connection for %s", conn_str, exc_info=True)
            _CONN_CACHE.pop(conn_str, None)


def _build_conn_string(chdb_path: str, chdb_options: dict[str, Any] | None) -> str:
    path = chdb_path or ":memory:"
    if not chdb_options:
        return path
    from urllib.parse import urlencode

    query = urlencode({k: str(v) for k, v in chdb_options.items()})
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}{query}"


class ChdbClient(Client):
    """ClickHouse Connect client backed by the in-process chdb engine."""

    #: chDB's query_arrow / query_arrow_stream return PyArrow objects backed directly by
    #: in-process Arrow buffers, so query_df is routed through the zero-copy Arrow path.
    #: Read by ``clickhouse_connect.driver.backend.client_supports``.
    supports_zero_copy_arrow: bool = True

    backend_name = "chdb"

    # HTTP-style transport settings: accepted by setting validation but stripped
    # before being forwarded to chdb (they have no in-process equivalent).
    valid_transport_settings: set[str] = {
        "database",
        "client_protocol_version",
        "session_id",
        "session_timeout",
        "session_check",
        "query_id",
        "quota_key",
        "compress",
        "decompress",
        "wait_end_of_query",
        "buffer_size",
        "role",
        "send_progress_in_http_headers",
        "http_headers_progress_interval_ms",
        "enable_http_compression",
    }

    def __init__(
        self,
        chdb_path: str = ":memory:",
        chdb_options: dict[str, Any] | None = None,
        database: str | None = None,
        settings: dict[str, Any] | None = None,
        query_limit: int = 0,
        tz_source: TzSource | None = None,
        tz_mode: TzMode | None = None,
        show_clickhouse_errors: bool | None = None,
        **ignored,
    ):
        if sys.platform.startswith("win"):
            raise NotSupportedError("chdb backend is not supported on Windows")

        import chdb

        self._chdb_path = chdb_path or ":memory:"
        self._chdb_options = dict(chdb_options) if chdb_options else {}
        self._connection_string = _build_conn_string(self._chdb_path, self._chdb_options)
        self._chdb_module = chdb
        # chdb-core permits only one open connection per data directory per process; reuse a
        # refcounted shared connection across all ChdbClient instances for the same path.
        # The shared per-connection lock serializes operations on it (a second concurrent
        # chdb.send_query on the same connection deadlocks inside the engine).
        self._conn, self._lock = _acquire_chdb_connection(self._connection_string)
        self._closed = False
        # If any of the rest of __init__ raises (super().__init__, set_client_setting,
        # extension import, ...), self.close() never gets called and the shared connection's
        # refcount stays elevated -- on a :memory: path that would leak indefinitely. The
        # try/except releases the refcount once before re-raising, so the next ChdbClient on
        # the same path either reuses or recreates a clean connection.
        try:
            self._client_settings: dict[str, str] = {}
            self._initial_settings = dict(settings or {})
            # Backs the `database` property below; set before super().__init__ because the
            # base constructor assigns self.database (triggering the property setter).
            # `_active_database` tracks the database the session has actually been switched
            # to via USE.
            self._database: str | None = None
            self._active_database: str | None = None
            self._read_format = "Native"
            self._write_format = "Native"
            self._transform = NativeTransform()
            self._integration_libs: set[str] = set()
            self.uri = f"chdb://{self._chdb_path}"
            self.write_compression = None
            self.compression = None

            # coerce_int handles None-or-string flexibility
            super().__init__(
                database=database,
                uri=self.uri,
                query_limit=coerce_int(query_limit),
                query_retries=0,
                server_host_name=None,
                tz_source=tz_source,
                tz_mode=tz_mode,
                show_clickhouse_errors=show_clickhouse_errors,
                autoconnect=True,
            )

            for k, v in self._initial_settings.items():
                self.set_client_setting(k, v)

            # chDB-only API (Python() table function, UDFs, native cursor, session path) is
            # exposed through this namespace; the base Client has no `.chdb`, so accessing
            # it on an HTTP client raises AttributeError, exactly as intended.
            from chdb.cc_extension import ChdbExtension

            self.chdb = ChdbExtension(self)
        except BaseException:
            _release_chdb_connection(self._connection_string)
            self._closed = True
            raise

        logger.info(
            "ChdbClient connected: chdb=%s, server_version=%s, path=%s",
            getattr(chdb, "__version__", "?"),
            self.server_version,
            self._chdb_path,
        )

    # ---- helpers -------------------------------------------------------

    @property
    def chdb_connection(self):
        """Underlying chdb connection. Escape hatch for advanced users."""
        return self._conn

    def _ensure_open(self) -> None:
        if self._closed:
            raise ProgrammingError("ChdbClient is closed") from None

    def _filter_per_call_settings(self, settings: dict[str, Any] | None) -> dict[str, str]:
        """Validate per-call settings and drop transport-only ones."""
        out: dict[str, str] = {}
        if not settings:
            return out
        invalid_action = common.get_setting("invalid_setting_action")
        for k, v in settings.items():
            str_v = self._validate_setting(k, v, invalid_action)
            if str_v is None:
                continue
            if k in self.valid_transport_settings:
                continue
            out[k] = str_v
        return out

    @staticmethod
    def _validate_setting_name(key: str) -> str:
        """Reject a setting name that does not look like a ClickHouse identifier.

        The setting name is interpolated directly into a ``SET <name> = <value>`` statement,
        and even though ``_validate_setting`` filters against the known-settings catalogue when
        the user picks the strict ``invalid_setting_action``, the permissive mode lets arbitrary
        keys flow through. Restricting the name to the identifier shape ClickHouse itself uses
        (``[A-Za-z_][A-Za-z0-9_]*``) closes the SQL-shape hole without affecting any real setting.
        """
        if not isinstance(key, str) or not _SETTING_NAME_RE.match(key):
            raise ProgrammingError(f"Invalid setting name {key!r}: must match {_SETTING_NAME_RE.pattern}")
        return key

    @staticmethod
    def _quote_setting_value(value: str) -> str:
        """SQL-quote a setting value so chdb sees the expected literal type.

        Without quotes chdb parses bare numeric-looking strings as UInt64; if the
        setting is actually String-typed (e.g. `insert_deduplication_token`) this
        triggers `Bad get: has UInt64, requested String`. ClickHouse coerces
        single-quoted literals back to numeric types where needed, so quoting
        unconditionally is safe.
        """
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    def _append_settings_clause(self, sql, settings):
        if not settings:
            return sql
        extras = ", ".join(f"{self._validate_setting_name(k)} = {self._quote_setting_value(v)}" for k, v in settings.items())
        if isinstance(sql, bytes):
            # raw_query can receive a bytes SQL when binary parameter substitution
            # produced non-UTF-8 byte sequences. chdb accepts bytes natively, so
            # keep the bytes path and append the settings clause as bytes too.
            sep = b", " if b" SETTINGS " in sql.upper() else b" SETTINGS "
            return sql + sep + extras.encode()
        if " SETTINGS " in sql.upper():
            return f"{sql}, {extras}"
        return f"{sql} SETTINGS {extras}"

    def _persist_setting(self, key: str, value: str) -> None:
        """Apply a setting to the underlying chdb session via SET."""
        try:
            with self._lock:
                self._conn.query(f"SET {self._validate_setting_name(key)} = {self._quote_setting_value(value)}", "TabSeparated")
        except Exception as ex:  # noqa: BLE001
            logger.debug("Failed to apply SET %s=%s to chdb session: %s", key, value, ex)

    def _snapshot_settings(self, keys: Sequence[str]) -> dict[str, tuple[str, bool]]:
        """Read current value and 'changed' flag for each key from system.settings.

        Returns a dict: {name -> (value, was_explicitly_set)}.
        """
        if not keys:
            return {}
        quoted = ", ".join(f"'{k}'" for k in keys)
        body = self._exec_raw_query(
            f"SELECT name, value, changed FROM system.settings WHERE name IN ({quoted})",
            "TabSeparated",
        )
        result: dict[str, tuple[str, bool]] = {}
        if body:
            for line in body.decode().rstrip("\n").split("\n"):
                parts = line.split("\t")
                if len(parts) == 3:
                    name, value, changed = parts
                    result[name] = (value, changed == "1")
        return result

    def _restore_settings(self, snapshot: dict[str, tuple[str, bool]]) -> None:
        """Restore settings to the state captured by `_snapshot_settings`."""
        for name, (value, was_changed) in snapshot.items():
            try:
                if was_changed:
                    self._persist_setting(name, value)
                else:
                    with self._lock:
                        self._conn.query(f"SET {self._validate_setting_name(name)} = DEFAULT", "TabSeparated")
            except Exception:  # noqa: BLE001
                logger.debug("Failed to restore setting %s after command()", name, exc_info=True)

    @staticmethod
    def _strip_param_prefix(bind_params: dict[str, Any]) -> dict[str, Any]:
        """chdb's `params` kwarg expects bare names (`x`); bind_query produces `param_x`."""
        return {(k[6:] if k.startswith("param_") else k): v for k, v in bind_params.items()} if bind_params else {}

    def _exec_raw_query(
        self,
        sql: str,
        fmt: str = "Native",
        params: dict[str, Any] | None = None,
        *,
        use_database: bool = True,
    ) -> bytes:
        """Run a query against chdb under the per-client lock and return raw bytes.

        ``use_database`` matches the public ``raw_query`` / ``raw_stream`` flag: when ``False``,
        skip the lazy ``USE`` that would otherwise rebind the session to ``self.database`` before
        this query (the previously-applied database stays in effect; this query is not
        re-anchored to the configured one).
        """
        self._ensure_open()
        if use_database:
            self._apply_database()
        with self._lock:
            try:
                result = self._conn.query(sql, fmt, params=params or {})
            except Exception as ex:  # noqa: BLE001
                raise self._wrap_exception(ex) from ex
            return result.bytes() if hasattr(result, "bytes") else bytes(result)

    def _wrap_exception(self, ex: Exception) -> Exception:
        message = _format_error_message(str(ex))
        if not self.show_clickhouse_errors:
            message = "ClickHouse error"
        return DatabaseError(message)

    def map_error(self, exc: BaseException) -> Exception:
        """Backend protocol hook: translate a chdb error into a clickhouse-connect error."""
        if isinstance(exc, Exception):
            return self._wrap_exception(exc)
        return DatabaseError(str(exc))

    def _format_for_command(self) -> str:
        return "TabSeparated"

    # ---- abstract method implementations -------------------------------

    def set_client_setting(self, key: str, value: Any) -> None:
        str_value = self._validate_setting(key, value, common.get_setting("invalid_setting_action"))
        if str_value is None:
            return
        self._client_settings[key] = str_value
        if key in self.valid_transport_settings:
            return
        self._persist_setting(key, str_value)

    def get_client_setting(self, key: str) -> str | None:
        return self._client_settings.get(key)

    def set_access_token(self, access_token: str) -> None:
        # chdb has no auth concept; accept silently for HTTP-mode drop-in compatibility.
        return None

    def _query_with_context(self, context: QueryContext) -> QueryResult:
        self._ensure_open()
        self._apply_database()
        if context.external_data is not None:
            raise NotSupportedError("external_data is not supported by the chdb backend")
        # chdb's Native output does not include the 8-byte block_info prefix that the
        # HTTP server emits when client_protocol_version is set.
        context.block_info = False
        final_query = self._prep_query(context)
        if isinstance(final_query, bytes):
            final_query = final_query.decode()
        params = self._strip_param_prefix(context.bind_params)
        if not context.is_insert and _columns_only_re.search(context.uncommented_query):
            # chdb emits zero Native bytes for a LIMIT 0 query, so the Native parser
            # would return an empty result with no column metadata. Fetch the schema
            # via JSON instead, matching the HTTP client's columns-only fast path.
            return self._fetch_columns_only(context, final_query, params)
        if context.is_insert:
            # INSERT ... VALUES carries its data inline and has no result block to parse;
            # appending `FORMAT Native` to a VALUES statement is a syntax error.
            sql = self._append_settings_clause(final_query, self._filter_per_call_settings(context.settings))
            self._exec_raw_query(sql, "TabSeparated", params=params)
            return QueryResult([])
        sql = f"{final_query}\n FORMAT Native"
        sql = self._append_settings_clause(sql, self._filter_per_call_settings(context.settings))
        if context.streaming:
            # Use chdb's streaming `send_query` so mid-execution engine errors
            # (e.g. throwIf, division by zero on row N) surface during result
            # iteration as `StreamFailureError`, matching HTTP's contract. The
            # non-streaming `conn.query` would raise eagerly and lose lazy-error
            # semantics — we only opt into that for true streaming results, since
            # holding the per-client lock for the lifetime of a non-iterated
            # QueryResult would deadlock subsequent calls.
            self._ensure_open()
            self._lock.acquire()
            try:
                streaming = self._conn.send_query(sql, "Native", params=params or {})
            except Exception as ex:  # noqa: BLE001
                self._lock.release()
                raise self._wrap_exception(ex) from ex
            byte_source = RespBuffCls(_ChdbStreamSource(streaming, self._lock))
        else:
            data = self._exec_raw_query(sql, "Native", params=params)
            byte_source = RespBuffCls(_BytesSource(data))
        query_result = self._transform.parse_response(byte_source, context)
        query_result.summary = {}
        return query_result

    def _fetch_columns_only(self, context: QueryContext, final_query: str, params: dict[str, Any]) -> QueryResult:
        sql = self._append_settings_clause(f"{final_query}\n FORMAT JSON", self._filter_per_call_settings(context.settings))
        body = self._exec_raw_query(sql, "JSON", params=params)
        meta = json.loads(body)["meta"]
        renamer = context.column_renamer
        names: list[str] = []
        types = []
        for col in meta:
            name = col["name"]
            if renamer is not None:
                try:
                    name = renamer(name)
                except Exception as ex:  # noqa: BLE001
                    logger.debug("Failed to rename column %s: %s", name, ex)
            names.append(name)
            types.append(get_from_name(col["type"]))
        return QueryResult([], None, tuple(names), tuple(types))

    def raw_query(
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        fmt: str | None = None,
        use_database: bool = True,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> bytes:
        if external_data is not None:
            raise NotSupportedError("external_data is not supported by the chdb backend")
        final_query, bound = bind_query(query, parameters, self.server_tz)
        # chdb's conn.query accepts both str and bytes; preserve bytes when binary
        # parameter substitution (e.g. `$xx$` placeholders) yields non-UTF-8 SQL.
        final_query = self._append_settings_clause(final_query, self._filter_per_call_settings(settings))
        # HTTP path defaults to server's TabSeparated when no fmt is provided.
        return self._exec_raw_query(
            final_query, fmt or "TabSeparated", params=self._strip_param_prefix(bound), use_database=use_database
        )

    def raw_stream(
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        fmt: str | None = None,
        use_database: bool = True,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> io.IOBase:
        if external_data is not None:
            raise NotSupportedError("external_data is not supported by the chdb backend")
        final_query, bound = bind_query(query, parameters, self.server_tz)
        if isinstance(final_query, bytes):
            final_query = final_query.decode()
        final_query = self._append_settings_clause(final_query, self._filter_per_call_settings(settings))
        params = self._strip_param_prefix(bound)
        output_fmt = fmt or "TabSeparated"
        if output_fmt not in _STREAM_SAFE_FORMATS:
            # Formats with global structure (Arrow IPC, Parquet, JSON, *WithNames, ...)
            # can't be assembled from chdb's per-block chunks. Fetch as a single
            # well-formed payload and wrap as an in-memory stream.
            data = self._exec_raw_query(final_query, output_fmt, params=params, use_database=use_database)
            return io.BytesIO(data)
        self._ensure_open()
        if use_database:
            self._apply_database()
        # Acquire the lock for the lifetime of the streaming read so concurrent
        # callers don't interleave queries on the same chdb connection.
        self._lock.acquire()
        try:
            streaming = self._conn.send_query(final_query, output_fmt, params=params or {})
        except Exception as ex:  # noqa: BLE001
            self._lock.release()
            raise self._wrap_exception(ex) from ex
        return _ChdbStreamFile(streaming, self._lock)

    def command(
        self,
        cmd: str,
        parameters: Sequence | dict[str, Any] | None = None,
        data: str | bytes | None = None,
        settings: dict[str, Any] | None = None,
        use_database: bool = True,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> str | int | Sequence[str] | QuerySummary:
        if external_data is not None:
            raise NotSupportedError("external_data is not supported by the chdb backend")
        cmd, bound = bind_query(cmd, parameters, self.server_tz)
        if isinstance(cmd, bytes):
            cmd = cmd.decode()
        params = self._strip_param_prefix(bound)
        if data is not None:
            if isinstance(data, bytes):
                data_str = data.decode()
            else:
                data_str = data
            cmd = f"{cmd}\n{data_str}"
        per_call = self._filter_per_call_settings(settings)
        # ClickHouse DDL doesn't accept a SETTINGS clause; apply per-call settings to
        # the chdb session via SET before running the command, then restore them
        # afterwards so they don't leak into the session.
        snapshot: dict[str, tuple[str, bool]] = {}
        if per_call:
            snapshot = self._snapshot_settings(list(per_call.keys()))
            for k, v in per_call.items():
                self._persist_setting(k, v)
        try:
            body = self._exec_raw_query(cmd, self._format_for_command(), params=params)
        finally:
            if snapshot:
                self._restore_settings(snapshot)
        if not body:
            return QuerySummary({})
        try:
            text = body.decode()
        except UnicodeDecodeError:
            return str(body)
        # Match HTTP client semantics: strip trailing newline, split by tab, single
        # token tries to coerce to int.
        if text.endswith("\n"):
            text = text[:-1]
        result = text.split("\t")
        if len(result) == 1:
            try:
                return int(result[0])
            except ValueError:
                return result[0]
        return result

    def ping(self) -> bool:
        try:
            self._exec_raw_query("SELECT 1", "TabSeparated")
            return True
        except Exception:  # noqa: BLE001
            logger.debug("chdb ping failed", exc_info=True)
            return False

    def data_insert(self, context: InsertContext) -> QuerySummary:
        if context.empty:
            return QuerySummary()
        return self._insert_via_infile(context)

    def raw_insert(
        self,
        table: str | None = None,
        column_names: Sequence[str] | None = None,
        insert_block: str | bytes | Generator[bytes, None, None] | BinaryIO | None = None,
        settings: dict[str, Any] | None = None,
        fmt: str | None = None,
        compression: str | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> QuerySummary:
        if insert_block is None or not table:
            raise ProgrammingError("raw_insert requires a table and insert_block")
        if compression and compression != "identity":
            # HTTP carries this via Content-Encoding so the server decompresses.
            # chdb has no equivalent input stage, so the caller's pre-compressed
            # bytes must be drained and decompressed in the client before being
            # written to the INFILE temp file.
            insert_block = _drain_to_bytes(insert_block)
            insert_block = _decompress(insert_block, compression)
            compression = None

        fmt = fmt or self._write_format
        cols = ""
        if column_names:
            cols = f" ({', '.join(quote_identifier(c) for c in column_names)})"

        # Drain insert_block to a temp file, then INSERT FROM INFILE.
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt.lower()}", delete=False)
        try:
            try:
                if isinstance(insert_block, (bytes, bytearray, memoryview)):
                    tmp.write(bytes(insert_block))
                elif isinstance(insert_block, str):
                    tmp.write(insert_block.encode())
                elif hasattr(insert_block, "to_pybytes"):
                    # pyarrow.Buffer and friends — buffer protocol holder
                    tmp.write(insert_block.to_pybytes())
                elif hasattr(insert_block, "read"):
                    while True:
                        chunk = insert_block.read(1 << 20)
                        if not chunk:
                            break
                        tmp.write(chunk if isinstance(chunk, (bytes, bytearray)) else chunk.encode())
                else:
                    for chunk in insert_block:
                        tmp.write(chunk if isinstance(chunk, (bytes, bytearray)) else chunk.encode())
            finally:
                tmp.close()

            per_call = self._filter_per_call_settings(settings)
            settings_clause = (
                f" SETTINGS {', '.join(f'{self._validate_setting_name(k)} = {self._quote_setting_value(v)}' for k, v in per_call.items())}" if per_call else ""
            )
            sql = f"INSERT INTO {table}{cols} FROM INFILE {_quote_sql_string(tmp.name)}{settings_clause} FORMAT {fmt}"
            self._exec_raw_query(sql, "TabSeparated")
            return QuerySummary({})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Decrement the shared-connection refcount. The underlying chdb.Connection is only
        # actually closed when the last ChdbClient releases it (see _CONN_CACHE).
        _release_chdb_connection(self._connection_string)

    def close_connections(self) -> None:
        # chdb only has a single embedded connection per client.
        self.close()

    @property
    def database(self) -> str | None:
        return self._database

    @database.setter
    def database(self, value: str | None) -> None:
        """Set the session's current database.

        Over HTTP, ``client.database`` is sent as a per-request parameter, so assigning it is
        cheap and lazy — assigning a not-yet-created database is fine. chDB is session-scoped
        with no per-query database parameter, so the equivalent is to ``USE`` it; we do so
        lazily (see ``_apply_database``) before the next operation, which preserves the common
        bootstrap of "create the database on this client, then use it". Unqualified table
        references then resolve against the assigned database, matching the HTTP backend.
        """
        self._database = value

    def _apply_database(self) -> None:
        """Switch the chdb session to ``self._database`` if it differs and is available.

        Called before each operation. A USE that fails (database not created yet) is swallowed
        so the very command that creates the database can still run; the switch is retried on
        the next operation once the database exists.
        """
        db = self._database
        if not db or db == "__default__" or db == self._active_database or self._closed:
            return
        try:
            with self._lock:
                self._conn.query(f"USE {quote_identifier(db)}", "TabSeparated")
            self._active_database = db
        except Exception:  # noqa: BLE001
            logger.debug("Deferred USE %s (database not yet available)", db, exc_info=True)

    # ---- zero-copy Arrow fast paths ------------------------------------
    #
    # chDB writes the Arrow IPC stream into an in-process buffer that PyArrow reads
    # through the C Data Interface, so these return PyArrow objects without a wire
    # round-trip. query_df then rides the same path (Arrow -> pandas).

    def _arrow_sql(
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None,
        settings: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        final_query, bound = bind_query(query, parameters, self.server_tz)
        if isinstance(final_query, bytes):
            final_query = final_query.decode()
        final_query = self._append_settings_clause(final_query, self._filter_per_call_settings(settings))
        return final_query, self._strip_param_prefix(bound)

    def query_arrow(
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        use_strings: bool | None = None,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> "pyarrow.Table":
        if external_data is not None:
            raise NotSupportedError("external_data is not supported by the chdb backend")
        check_arrow()
        self._add_integration_tag("arrow")
        settings = self._update_arrow_settings(settings, use_strings)
        sql, params = self._arrow_sql(query, parameters, settings)
        self._ensure_open()
        self._apply_database()
        with self._lock:
            try:
                res = self._conn.query(sql, "Arrow", params=params or {})
            except Exception as ex:  # noqa: BLE001
                raise self._wrap_exception(ex) from ex
        return self._chdb_module.to_arrowTable(res)

    def query_arrow_stream(
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        use_strings: bool | None = None,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> StreamContext:
        if external_data is not None:
            raise NotSupportedError("external_data is not supported by the chdb backend")
        check_arrow()
        self._add_integration_tag("arrow")
        settings = self._update_arrow_settings(settings, use_strings)
        sql, params = self._arrow_sql(query, parameters, settings)
        self._ensure_open()
        self._apply_database()
        self._lock.acquire()
        try:
            streaming = self._conn.send_query(sql, "Arrow", params=params or {})
        except Exception as ex:  # noqa: BLE001
            self._lock.release()
            raise self._wrap_exception(ex) from ex
        source = _ChdbArrowStreamSource(streaming, self._lock)
        return StreamContext(source, source.gen())

    def query_df(
        self,
        query: str | None = None,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        query_formats: dict[str, str] | None = None,
        column_formats: dict[str, str] | None = None,
        encoding: str | None = None,
        use_none: bool | None = None,
        max_str_len: int | None = None,
        use_na_values: bool | None = None,
        query_tz: str | None = None,
        column_tzs: dict[str, str | tzinfo] | None = None,
        context: QueryContext | None = None,
        external_data: ExternalData | None = None,
        use_extended_dtypes: bool | None = None,
        transport_settings: dict[str, str] | None = None,
        tz_mode: TzMode | None = None,
    ) -> "pandas.DataFrame":
        # supports_zero_copy_arrow: route the DataFrame path through chDB's in-process
        # Arrow buffer instead of the Native parser. When the caller asks for Native-parser
        # dtype controls (query_formats / column_formats / use_none / numpy-extended dtypes
        # / per-column tz / an explicit context) we defer to the inherited Native path so
        # those options keep working; divergences between the two paths are catalogued in
        # the comparison suite as documented expected-fails.
        native_only = any(
            x is not None
            for x in (query_formats, column_formats, use_none, use_na_values, query_tz, column_tzs,
                      context, use_extended_dtypes, max_str_len, tz_mode)
        )
        if query is None or native_only:
            return super().query_df(
                query,
                parameters=parameters,
                settings=settings,
                query_formats=query_formats,
                column_formats=column_formats,
                encoding=encoding,
                use_none=use_none,
                max_str_len=max_str_len,
                use_na_values=use_na_values,
                query_tz=query_tz,
                column_tzs=column_tzs,
                context=context,
                external_data=external_data,
                use_extended_dtypes=use_extended_dtypes,
                transport_settings=transport_settings,
                tz_mode=tz_mode,
            )
        table = self.query_arrow(query, parameters=parameters, settings=settings, external_data=external_data)
        return table.to_pandas()

    # ---- insert implementations ----------------------------------------

    def _insert_via_infile(self, context: InsertContext) -> QuerySummary:
        tmp = tempfile.NamedTemporaryFile(suffix=".native", delete=False)
        try:
            try:
                first_chunk = True
                # NativeTransform.build_insert prepends an `INSERT INTO ... FORMAT Native\n`
                # statement to the first chunk for the HTTP request body. We're going to
                # write only the Native bytes to a file and INSERT FROM INFILE, so the
                # prefix must be skipped.
                for chunk in self._transform.build_insert(context):
                    if context.insert_exception is not None:
                        ex = context.insert_exception
                        context.insert_exception = None
                        raise ex
                    if first_chunk:
                        nl = chunk.find(b"\n")
                        if nl >= 0:
                            chunk = chunk[nl + 1 :]
                        first_chunk = False
                    tmp.write(chunk)
            finally:
                tmp.close()

            cols = ", ".join(quote_identifier(c) for c in context.column_names)
            per_call = self._filter_per_call_settings(context.settings)
            settings_clause = (
                f" SETTINGS {', '.join(f'{self._validate_setting_name(k)} = {self._quote_setting_value(v)}' for k, v in per_call.items())}" if per_call else ""
            )
            sql = f"INSERT INTO {context.table} ({cols}) FROM INFILE {_quote_sql_string(tmp.name)}{settings_clause} FORMAT Native"
            self._exec_raw_query(sql, "TabSeparated")
            return QuerySummary({})
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            context.data = None

    # ---- integration tagging ------------------------------------------

    def _add_integration_tag(self, name: str) -> None:
        # No User-Agent header to update for in-process chdb; just record for
        # potential future use.
        self._integration_libs.add(name)


class _ChdbArrowStreamSource:
    """Closable source feeding ``StreamContext`` with PyArrow tables per chdb Arrow block."""

    __slots__ = ("_sr", "_released", "_finalizer", "__weakref__")

    def __init__(self, streaming_result, lock: threading.Lock):
        self._sr = streaming_result
        self._released = [False]
        self._finalizer = weakref.finalize(self, _finalize_stream, streaming_result, lock, self._released)

    def gen(self):
        import pyarrow as pa

        try:
            reader = self._sr.record_batch()
            for batch in reader:
                yield pa.Table.from_batches([batch])
        except Exception as ex:  # noqa: BLE001
            raise StreamFailureError(_format_error_message(str(ex))) from ex
        finally:
            self.close()

    def close(self):
        self._finalizer()


class _ChdbStreamFile(io.RawIOBase):
    """
    File-like adapter wrapping chdb's StreamingResult iterator so callers in
    clickhouse-connect (which expect an io.IOBase / aiohttp-style stream) can
    iterate bytes block-by-block.

    Holds a per-client lock for its lifetime so the chdb connection is not used
    concurrently by another caller while a stream is in flight.
    """

    def __init__(self, streaming_result, lock: threading.Lock):
        super().__init__()
        self._sr = streaming_result
        # Mutable accumulator: repeated `bytes += chunk` would be O(n^2) for many small reads
        # because it builds a fresh bytes object every concatenation. A bytearray we extend
        # in place is amortized O(n).
        self._buf = bytearray()
        self._eof = False
        self._released = [False]
        # Guarantees lock release even if close() is never called; see _finalize_stream.
        self._finalizer = weakref.finalize(self, _finalize_stream, streaming_result, lock, self._released)

    def readable(self) -> bool:
        return True

    def _pull(self) -> bytes:
        while True:
            try:
                chunk = next(self._sr)
            except StopIteration:
                self._eof = True
                return b""
            except Exception as ex:  # noqa: BLE001
                # chdb wraps mid-stream engine errors as RuntimeError. Surface them
                # as StreamFailureError so callers can catch them with the same
                # exception type used by the HTTP backend's mid-stream failures.
                msg = _format_error_message(str(ex))
                self._eof = True
                raise StreamFailureError(msg) from ex
            payload = chunk.bytes() if hasattr(chunk, "bytes") else bytes(chunk)
            if payload:
                return payload

    def read(self, size: int | None = -1) -> bytes:
        if self._released[0]:
            return b""
        if size is None or size < 0:
            parts = [bytes(self._buf)]
            self._buf.clear()
            while not self._eof:
                chunk = self._pull()
                if not chunk:
                    break
                parts.append(chunk)
            return b"".join(parts)
        while len(self._buf) < size and not self._eof:
            chunk = self._pull()
            if not chunk:
                break
            self._buf.extend(chunk)
        if not self._buf:
            return b""
        out = bytes(self._buf[:size])
        del self._buf[:size]
        return out

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        n = len(data)
        if n:
            buf[:n] = data
        return n

    def close(self) -> None:
        self._finalizer()
        super().close()


class AsyncChdbClient(Client):
    """
    Async-facing client for the in-process chdb backend. Each public coroutine
    schedules the corresponding sync ChdbClient call on the default thread
    executor. Sync-only methods (settings, min_version) are passed through
    directly.

    chdb has no native async API, so this delegates to the wrapped sync client via
    ``run_in_executor``. Because ChdbClient serializes concurrent calls on a per-client
    ``threading.Lock``, gather()-style concurrency on a single AsyncChdbClient does not run
    in parallel — for true parallelism, create multiple clients.
    """

    backend_name = "chdb"
    supports_zero_copy_arrow: bool = True

    def __init__(self, sync: ChdbClient):
        self._sync = sync
        # Mirror attributes commonly read off the client object so user code that
        # touches them (server_version, server_tz, database, etc.) keeps working.
        self.server_tz = sync.server_tz
        self.server_version = sync.server_version
        self.server_settings = sync.server_settings
        self.database = sync.database
        self.uri = sync.uri
        self.query_limit = sync.query_limit
        self.query_retries = sync.query_retries
        self.tz_mode = sync.tz_mode
        self._tz_source = sync._tz_source
        self._apply_server_tz = sync._apply_server_tz
        self._dst_safe = sync._dst_safe
        self.show_clickhouse_errors = sync.show_clickhouse_errors
        self.protocol_version = sync.protocol_version
        self.write_compression = sync.write_compression
        self.compression = sync.compression
        self._read_format = sync._read_format
        self._write_format = sync._write_format
        self._transform = sync._transform

    @property
    def chdb_connection(self):
        return self._sync.chdb_connection

    @property
    def chdb(self):
        return self._sync.chdb

    @property
    def database(self) -> str | None:
        return self._sync.database

    @database.setter
    def database(self, value: str | None) -> None:
        # Delegate to the sync client so assignment issues USE on the shared connection.
        self._sync.database = value

    async def _run(self, func, *args, **kwargs):
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # ---- sync passthroughs (no I/O) ----

    def set_client_setting(self, key: str, value: Any) -> None:
        self._sync.set_client_setting(key, value)

    def get_client_setting(self, key: str) -> str | None:
        return self._sync.get_client_setting(key)

    def set_access_token(self, access_token: str) -> None:
        self._sync.set_access_token(access_token)

    def min_version(self, version_str: str) -> bool:
        return self._sync.min_version(version_str)

    def map_error(self, exc: BaseException) -> Exception:
        return self._sync.map_error(exc)

    # ---- async overrides ----

    async def _query_with_context(self, context: QueryContext) -> QueryResult:  # type: ignore[override]
        return await self._run(self._sync._query_with_context, context)

    async def query(  # type: ignore[override]
        self,
        query: str | None = None,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        query_formats: dict[str, str] | None = None,
        column_formats: dict[str, str | dict[str, str]] | None = None,
        encoding: str | None = None,
        use_none: bool | None = None,
        column_oriented: bool | None = None,
        use_numpy: bool | None = None,
        max_str_len: int | None = None,
        context: QueryContext | None = None,
        query_tz: str | tzinfo | None = None,
        column_tzs: dict[str, str | tzinfo] | None = None,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
        tz_mode: TzMode | None = None,
    ) -> QueryResult:
        return await self._run(
            lambda: self._sync.query(
                query=query,
                parameters=parameters,
                settings=settings,
                query_formats=query_formats,
                column_formats=column_formats,
                encoding=encoding,
                use_none=use_none,
                column_oriented=column_oriented,
                use_numpy=use_numpy,
                max_str_len=max_str_len,
                context=context,
                query_tz=query_tz,
                column_tzs=column_tzs,
                external_data=external_data,
                transport_settings=transport_settings,
                tz_mode=tz_mode,
            )
        )

    async def query_column_block_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_column_block_stream(*args, **kwargs))

    async def query_row_block_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_row_block_stream(*args, **kwargs))

    async def query_rows_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_rows_stream(*args, **kwargs))

    async def query_np(self, *args, **kwargs) -> "numpy.ndarray":
        return await self._run(lambda: self._sync.query_np(*args, **kwargs))

    async def query_np_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_np_stream(*args, **kwargs))

    async def query_df(self, *args, **kwargs) -> "pandas.DataFrame":
        return await self._run(lambda: self._sync.query_df(*args, **kwargs))

    async def query_df_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_df_stream(*args, **kwargs))

    async def query_arrow(self, *args, **kwargs) -> "pyarrow.Table":
        return await self._run(lambda: self._sync.query_arrow(*args, **kwargs))

    async def query_arrow_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_arrow_stream(*args, **kwargs))

    async def query_df_arrow(self, *args, **kwargs) -> "pandas.DataFrame | polars.DataFrame":
        return await self._run(lambda: self._sync.query_df_arrow(*args, **kwargs))

    async def query_df_arrow_stream(self, *args, **kwargs):  # type: ignore[override]
        return await self._run(lambda: self._sync.query_df_arrow_stream(*args, **kwargs))

    async def command(  # type: ignore[override]
        self,
        cmd: str,
        parameters: Sequence | dict[str, Any] | None = None,
        data: str | bytes | None = None,
        settings: dict[str, Any] | None = None,
        use_database: bool = True,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> str | int | Sequence[str] | QuerySummary:
        return await self._run(
            lambda: self._sync.command(
                cmd,
                parameters=parameters,
                data=data,
                settings=settings,
                use_database=use_database,
                external_data=external_data,
                transport_settings=transport_settings,
            )
        )

    async def ping(self) -> bool:  # type: ignore[override]
        return await self._run(self._sync.ping)

    async def raw_query(  # type: ignore[override]
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        fmt: str | None = None,
        use_database: bool = True,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> bytes:
        return await self._run(
            lambda: self._sync.raw_query(
                query,
                parameters=parameters,
                settings=settings,
                fmt=fmt,
                use_database=use_database,
                external_data=external_data,
                transport_settings=transport_settings,
            )
        )

    async def raw_stream(  # type: ignore[override]
        self,
        query: str,
        parameters: Sequence | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        fmt: str | None = None,
        use_database: bool = True,
        external_data: ExternalData | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> io.IOBase:
        return await self._run(
            lambda: self._sync.raw_stream(
                query,
                parameters=parameters,
                settings=settings,
                fmt=fmt,
                use_database=use_database,
                external_data=external_data,
                transport_settings=transport_settings,
            )
        )

    async def insert(  # type: ignore[override]
        self,
        table: str | None = None,
        data=None,
        column_names: str | Iterable[str] = "*",
        database: str | None = None,
        column_types=None,
        column_type_names=None,
        column_oriented: bool = False,
        settings: dict[str, Any] | None = None,
        context: InsertContext | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> QuerySummary:
        return await self._run(
            lambda: self._sync.insert(
                table=table,
                data=data,
                column_names=column_names,
                database=database,
                column_types=column_types,
                column_type_names=column_type_names,
                column_oriented=column_oriented,
                settings=settings,
                context=context,
                transport_settings=transport_settings,
            )
        )

    async def insert_df(self, *args, **kwargs) -> QuerySummary:  # type: ignore[override]
        return await self._run(lambda: self._sync.insert_df(*args, **kwargs))

    async def insert_arrow(self, *args, **kwargs) -> QuerySummary:  # type: ignore[override]
        return await self._run(lambda: self._sync.insert_arrow(*args, **kwargs))

    async def insert_df_arrow(self, *args, **kwargs) -> QuerySummary:  # type: ignore[override]
        return await self._run(lambda: self._sync.insert_df_arrow(*args, **kwargs))

    async def data_insert(self, context: InsertContext) -> QuerySummary:  # type: ignore[override]
        return await self._run(self._sync.data_insert, context)

    async def raw_insert(  # type: ignore[override]
        self,
        table: str | None = None,
        column_names: Sequence[str] | None = None,
        insert_block: str | bytes | Generator[bytes, None, None] | BinaryIO | None = None,
        settings: dict[str, Any] | None = None,
        fmt: str | None = None,
        compression: str | None = None,
        transport_settings: dict[str, str] | None = None,
    ) -> QuerySummary:
        return await self._run(
            lambda: self._sync.raw_insert(
                table=table,
                column_names=column_names,
                insert_block=insert_block,
                settings=settings,
                fmt=fmt,
                compression=compression,
                transport_settings=transport_settings,
            )
        )

    async def close(self) -> None:  # type: ignore[override]
        await self._run(self._sync.close)

    async def close_connections(self) -> None:  # type: ignore[override]
        await self._run(self._sync.close_connections)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False

    async def create_insert_context(self, *args, **kwargs) -> InsertContext:  # type: ignore[override]
        return await self._run(lambda: self._sync.create_insert_context(*args, **kwargs))

    def create_query_context(self, *args, **kwargs) -> QueryContext:
        return self._sync.create_query_context(*args, **kwargs)


# ChdbClient constructor arguments that may be passed through get_client(...).
_CHDB_CTOR_KWARGS = frozenset({"query_limit", "tz_source", "tz_mode", "show_clickhouse_errors"})


def _split_chdb_kwargs(database, settings, generic_args, kwargs):
    """Normalize get_client(...) arguments into ChdbClient constructor arguments.

    * ``path`` (the documented option), ``chdb_path`` and ``chdb_options`` configure the
      embedded engine; ``query_limit`` / ``tz_source`` / ``tz_mode`` /
      ``show_clickhouse_errors`` are forwarded to the constructor.
    * ``generic_args`` (the DBAPI connection-string path) follow create_client's HTTP
      convention: anything not recognized above becomes a ClickHouse setting (``ch_`` prefix
      stripped). This is how a DSN like ``?chdb_path=/db&ch_max_threads=4`` is parsed.
    * Any *other* explicit keyword argument is HTTP-transport noise (``username``,
      ``compress``, ``connect_timeout``, ``verify``, ``http_proxy``, ...) with no in-process
      equivalent, so it is dropped silently for drop-in compatibility.
    """
    settings = dict(settings or {})
    ctor_kwargs: dict[str, Any] = {}
    chdb_path = None
    chdb_options = None

    def _take(src: dict, *, treat_unknown_as_setting: bool) -> None:
        nonlocal chdb_path, chdb_options
        for key in list(src):
            value = src.pop(key)
            if key in ("path", "chdb_path"):
                if value:
                    chdb_path = value
            elif key == "chdb_options":
                chdb_options = value
            elif key in _CHDB_CTOR_KWARGS:
                ctor_kwargs[key] = value
            elif key == "database":
                pass  # handled by the explicit database argument
            elif treat_unknown_as_setting:
                settings[key[3:] if key.startswith("ch_") else key] = value
            # else: HTTP-only kwarg with no chdb equivalent -> dropped

    _take(dict(generic_args) if generic_args else {}, treat_unknown_as_setting=True)
    _take(kwargs, treat_unknown_as_setting=False)
    return chdb_path or ":memory:", chdb_options, database, settings, ctor_kwargs


class ChdbBackend:
    """Backend factory registered at the ``clickhouse_connect.backends`` entry point.

    Satisfies :class:`clickhouse_connect.driver.backend.Backend`: clickhouse-connect's
    ``get_client(backend="chdb", ...)`` loads this object and calls ``create_client`` /
    ``create_async_client``.
    """

    backend_name = "chdb"
    supports_zero_copy_arrow = True

    def create_client(self, *, database=None, settings=None, generic_args=None, **kwargs) -> ChdbClient:
        chdb_path, chdb_options, database, settings, ctor_kwargs = _split_chdb_kwargs(
            database, settings, generic_args, kwargs
        )
        if database == "__default__":
            database = None
        return ChdbClient(
            chdb_path=chdb_path,
            chdb_options=chdb_options,
            database=database,
            settings=settings,
            **ctor_kwargs,
        )

    def create_async_client(self, *, database=None, settings=None, generic_args=None, **kwargs) -> AsyncChdbClient:
        sync = self.create_client(database=database, settings=settings, generic_args=generic_args, **kwargs)
        return AsyncChdbClient(sync)


# The object the entry point resolves to. A module-level singleton is enough; the factory
# holds no per-client state.
backend = ChdbBackend()
