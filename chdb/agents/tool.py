"""ChDBTool — the canonical Python agent tool for chDB.

A thin, engine-correct wrapper that agent frameworks (LangChain, CrewAI,
Pydantic AI, AutoGen, smolagents, ...) shim in ~10-30 lines instead of each
re-implementing query + introspection + safety. Its behavior is defined by the
language-neutral CONTRACT.md and exercised by conformance/cases.jsonl, which the
TypeScript binding (chdb-node) runs against the same fixture so the two stay in
lock-step.

Four contract pillars:
  P1 read-only by the engine   (SET readonly=2 at session creation; opt out with read_only=False)
  P2 value binding             (values bound as chDB params, never concatenated; identifiers quoted)
  P3 result cap                (max_rows / max_bytes; truncation is flagged, never silent)
  P4 error-to-model            (call() returns an error envelope; query() raises typed errors)
"""

import asyncio
import json
import threading

from .descriptors import tool_specs as _tool_specs
from .errors import NETWORK_HINT, ChDBError, ChDBReadOnlyError, parse_error
from .safety import (
    FALLBACK_KNOWN_TABLE_FUNCTIONS,
    NETWORK_TABLE_FUNCTIONS,
    find_source_calls,
    path_allowed,
    quote_ident,
    quote_string,
)

__all__ = ["ChDBTool", "QueryResult"]

_MISSING = object()  # sentinel for "name absent from globals" in dataframe_query

# Sessions with a watchdog-abandoned call still in flight. Destroying one
# mid-call is UB, so they leak here for process lifetime instead of crashing.
_ABANDONED_SESSIONS = []


def _int_arg(value, name):
    """Coerce a numeric argument to int, or raise a typed INVALID_ARGUMENT.

    A non-numeric cap must fail loudly in every binding: silently ignoring it
    would disable the result cap (the TypeScript binding once did exactly that
    via NaN comparisons), and a bare ValueError would bypass the typed-error
    contract that lets the model read the failure.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ChDBError(
            "{} must be an integer, got {!r}".format(name, value), type="INVALID_ARGUMENT"
        )


# Attached to truncated envelope results so the model narrows instead of re-running.
TRUNCATION_HINT = (
    "Result truncated at the row/byte cap; more rows exist. "
    "If you need them, aggregate, filter with WHERE, or select fewer columns "
    "instead of re-running the same query."
)


class QueryResult:
    """Result of a query: decoded rows plus honest truncation / stat metadata."""

    __slots__ = ("rows", "row_count", "truncated", "column_names", "elapsed_s", "bytes_read")

    def __init__(self, rows, truncated, column_names, elapsed_s=None, bytes_read=None):
        self.rows = rows
        self.row_count = len(rows)
        self.truncated = truncated
        self.column_names = column_names
        self.elapsed_s = elapsed_s
        self.bytes_read = bytes_read

    def to_dict(self):
        d = {
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "column_names": self.column_names,
            "elapsed_s": self.elapsed_s,
            "bytes_read": self.bytes_read,
        }
        if self.truncated:
            d["hint"] = TRUNCATION_HINT
        return d

    def __repr__(self):
        t = " truncated" if self.truncated else ""
        return "<QueryResult {} rows{}>".format(self.row_count, t)


# Tool name -> method. Names match ClickHouse's MCP server (mcp-clickhouse) so
# the agent-facing corpus is consistent across the MCP and library surfaces.
_TOOL_METHODS = {
    "run_select_query": "query",
    "list_databases": "list_databases",
    "list_tables": "list_tables",
    "describe_table": "describe",
    "get_sample_data": "get_sample_data",
    "list_functions": "list_functions",
    "attach_file": "attach_file",
}


class ChDBTool:
    """Agent-safe query + introspection over one chDB session.

    chDB is single-engine-per-process, and `readonly` cannot be lowered once
    set, so `read_only` is fixed at construction (not per call).
    """

    def __init__(
        self,
        path=":memory:",
        *,
        read_only=True,
        max_rows=1000,
        max_bytes=1_000_000,
        max_execution_time=None,
        network_timeout=60,
        max_memory_usage=None,
        max_result_bytes=None,
        file_allowlist=None,
        attachments=None,
        session=None,
    ):
        self.read_only = bool(read_only)
        self.max_rows = max(1, _int_arg(max_rows, "max_rows"))
        self.max_bytes = max(1, _int_arg(max_bytes, "max_bytes"))
        self.max_execution_time = (
            None if max_execution_time is None else max(0, _int_arg(max_execution_time, "max_execution_time"))
        )
        # Deadline for queries referencing network sources (url()/s3()/...).
        # Binding-side: a firewalled endpoint can hang the engine past every
        # engine-side timeout (blocked TLS handshake). None/0 disables.
        self.network_timeout = (
            None if not network_timeout else max(1, _int_arg(network_timeout, "network_timeout"))
        )
        self._poisoned = False
        # Engine memory bound (bytes); exceeding raises MEMORY_LIMIT_EXCEEDED.
        self.max_memory_usage = (
            None if max_memory_usage is None else max(1, _int_arg(max_memory_usage, "max_memory_usage"))
        )
        # Engine result-size backstop (bytes). Break mode truncates WITHOUT a
        # flag — keep it well above max_bytes so the flagged cap fires first.
        self.max_result_bytes = (
            None if max_result_bytes is None else max(1, _int_arg(max_result_bytes, "max_result_bytes"))
        )
        # None = no allowlist (all paths allowed); a list = only these prefixes.
        self.file_allowlist = list(file_allowlist) if file_allowlist else None
        self._owns_session = session is None
        if session is not None:
            self._session = session
        else:
            # chdb.session.Session() with no path == ephemeral :memory:.
            from chdb.session import Session

            self._session = Session() if path in (":memory:", "", None) else Session(path)

        # If any setup below throws (a readonly mismatch, a bad attachment path,
        # an engine SET error), the constructor never returns, so a Session we
        # own would otherwise leak (the caller has no instance to close()).
        # Close it before re-throwing — mirrors the TypeScript binding.
        try:
            if not self._owns_session:
                # An external session's readonly state is probed, never mutated:
                # SET readonly=2 cannot be lowered again, so silently applying it
                # would irreversibly lock the caller's shared session (and any
                # other tool on it); silently skipping it would leave a tool that
                # claims read_only but isn't. A mismatch fails construction and
                # forces an explicit choice: let the tool own its session, pass
                # read_only=False, or hand in a session already at readonly=2.
                expected = 2 if self.read_only else 0
                actual = self._probe_readonly()
                if actual != expected:
                    raise ChDBError(
                        "external session has readonly={} but the tool was declared "
                        "read_only={} (expects readonly={}); pass a matching session, "
                        "change the read_only flag, or omit session so the tool owns one".format(
                            actual, self.read_only, expected
                        ),
                        type="CONFIG_MISMATCH",
                    )
            # Exact 64-bit integers survive JSON as strings rather than lossy floats.
            self._session.query("SET output_format_json_quote_64bit_integers=1", "CSV")
            # Engine-side row bound: without it the decode path buffers the whole
            # result before max_rows applies (OOM in small sandboxes). cap+1 under
            # 'break' keeps the truncated flag exact.
            self._session.query("SET max_block_size=8192", "CSV")
            self._session.query("SET result_overflow_mode='break'", "CSV")
            self._session.query("SET max_result_rows={}".format(self.max_rows + 1), "CSV")
            if self.max_result_bytes is not None:
                self._session.query("SET max_result_bytes={}".format(self.max_result_bytes), "CSV")
            if self.max_memory_usage is not None:
                # loud engine OOM guard: exceeding raises MEMORY_LIMIT_EXCEEDED
                self._session.query("SET max_memory_usage={}".format(self.max_memory_usage), "CSV")
            if self.max_execution_time is not None:
                # engine-side wall-clock bound; a runaway query raises TIMEOUT_EXCEEDED
                self._session.query("SET max_execution_time={}".format(self.max_execution_time), "CSV")
            if self.network_timeout is not None:
                # Fail fast on dead endpoints: one attempt, no HEAD probe. The
                # TLS handshake is bounded by max(send, receive) — NOT by
                # connection_timeout — and one attempt costs ~4-5x the setting
                # (verified against chdb-core main), so keep send/receive small.
                # The watchdog in query() is the backstop where none of this bites.
                cap_s = min(self.network_timeout, 10)
                self._session.query("SET http_connection_timeout={}".format(cap_s), "CSV")
                self._session.query("SET http_receive_timeout={}".format(cap_s), "CSV")
                self._session.query("SET http_send_timeout={}".format(cap_s), "CSV")
                self._session.query("SET http_max_tries=1", "CSV")
                self._session.query("SET http_make_head_request=0", "CSV")
            # Attachments must be materialized BEFORE the read-only lock, because
            # CREATE VIEW is a write that readonly=2 rejects. This is why read-only
            # tools declare files at construction rather than via attach_file().
            for name, spec in (attachments or {}).items():
                p, fmt = spec if isinstance(spec, (tuple, list)) else (spec, None)
                self._create_file_view(name, p, fmt)
            if self.read_only and self._owns_session:
                # readonly=2 (NOT 1): blocks INSERT/CREATE/ALTER/DROP while still
                # allowing SELECT and the file()/s3()/url() table functions that are
                # chDB's whole point. readonly=1 rejects those. Cannot be un-set.
                # (An external session was verified to be there already.)
                self._session.query("SET readonly=2", "CSV")
            # The allowlist gate judges table functions against what THIS engine
            # exposes, so new source functions are gated by default instead of
            # silently allowed by a stale hand-written list.
            self._known_table_functions = (
                self._snapshot_table_functions() if self.file_allowlist else None
            )
        except BaseException:
            if self._owns_session and self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = None
            raise

    def _probe_readonly(self):
        try:
            res = self._session.query("SELECT toInt32(getSetting('readonly'))", "CSV")
            return int(res.bytes().decode().strip().strip('"'))
        except ChDBError:
            raise
        except Exception as e:
            raise parse_error(e)

    def _snapshot_table_functions(self):
        """The live set of table-function names (lowercase), unioned with the
        static fallback; the fallback alone if the engine can't answer."""
        try:
            res = self._session.query("SELECT lower(name) FROM system.table_functions", "CSV")
            names = {line.strip().strip('"') for line in res.bytes().decode().splitlines()}
            return frozenset(n for n in names if n) | FALLBACK_KNOWN_TABLE_FUNCTIONS
        except Exception:
            return FALLBACK_KNOWN_TABLE_FUNCTIONS

    # ---- core query -------------------------------------------------------

    def query(self, sql, *, params=None, max_rows=None, _permit_fns=frozenset()):
        """Run read SQL. Values MUST be passed via `params` ({name:Type} + dict),
        never formatted into `sql`. Returns a `QueryResult`; raises `ChDBError`.
        (`_permit_fns` is internal: dataframe_query exempts the Python() table
        function it itself injects from the allowlist gate.)
        """
        if not isinstance(sql, str) or sql.strip() == "":
            raise ChDBError("sql must be a non-empty string")
        if self._poisoned:
            raise ChDBError(
                "a previous network-source query was abandoned after its deadline; "
                "this tool's engine session may be blocked — create a new ChDBTool",
                type="TOOL_ERROR",
            )
        self._enforce_allowlist(sql, permit=_permit_fns)
        # Clamped to the constructor cap: the engine bound was fixed there, so a
        # larger per-call cap would truncate silently.
        cap = (
            self.max_rows
            if max_rows is None
            else min(max(1, _int_arg(max_rows, "max_rows")), self.max_rows)
        )
        # Both the engine call and the decode go through parse_error: malformed or
        # non-JSON engine output (edge-case statements, empty results) becomes a
        # typed ChDBError rather than a bare JSONDecodeError leaking to the caller.
        try:
            if self.network_timeout and any(True for _ in find_source_calls(sql, NETWORK_TABLE_FUNCTIONS)):
                res = self._run_with_network_deadline(sql, params)
            else:
                res = self._session.query(sql, "JSON", params=params or {})
            obj = json.loads(res.bytes().decode() or "{}")
        except ChDBError:
            raise  # watchdog/typed errors pass through, never re-wrapped
        except Exception as e:
            raise parse_error(e)
        data = obj.get("data", []) or []
        meta = obj.get("meta", []) or []
        stats = obj.get("statistics", {}) or {}
        cols = [m.get("name") for m in meta]
        truncated = len(data) > cap
        rows = data[:cap] if truncated else data
        # Secondary byte guard, applied whether or not the row cap already fired:
        # a few very large rows under max_rows must still be capped by max_bytes.
        # Rows are measured in UTF-8 BYTES of their compact JSON encoding —
        # ensure_ascii would count "汉" as the 6 chars of "\\u6c49" while the
        # TypeScript binding counts UTF-16 units; UTF-8 bytes is the one measure
        # both bindings can produce identically (CONTRACT.md P3).
        if self.max_bytes:
            size = 0
            for i, r in enumerate(rows):
                size += len(json.dumps(r, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
                if size > self.max_bytes:
                    rows = rows[:i]
                    truncated = True
                    break
        return QueryResult(rows, truncated, cols, stats.get("elapsed"), stats.get("bytes_read"))

    def _run_with_network_deadline(self, sql, params):
        """Engine call with a deadline (CONTRACT P5). The blocked native call
        can't be cancelled, so on expiry it is abandoned and the tool poisoned.
        Runs on a daemon thread: an executor's non-daemon worker would make a
        truly stuck call block interpreter exit at the atexit join."""
        session = self._session
        outcome = {}
        done = threading.Event()

        def _call():
            try:
                outcome["res"] = session.query(sql, "JSON", params=params or {})
            except BaseException as e:
                outcome["exc"] = e
            finally:
                done.set()

        threading.Thread(target=_call, daemon=True, name="chdb-network-query").start()
        if not done.wait(self.network_timeout):
            self._poisoned = True
            _ABANDONED_SESSIONS.append(session)
            raise ChDBError(
                "network-source query did not return within {}s".format(self.network_timeout),
                type="NETWORK_TIMEOUT",
                hint=NETWORK_HINT,
            )
        if "exc" in outcome:
            exc = outcome["exc"]
            err = exc if isinstance(exc, ChDBError) else parse_error(exc)
            # Engine-side timeout on a network source (Poco::TimeoutException,
            # code 1001) deserves the same guidance as a watchdog expiry.
            if err.hint is None and "Poco::TimeoutException" in err.message:
                err.hint = NETWORK_HINT
            raise err
        return outcome["res"]

    async def aquery(self, sql, *, params=None, max_rows=None):
        """Async wrapper. chDB has no native async engine call, so this runs
        `query` in a worker thread (documented, not faked)."""
        return await asyncio.to_thread(self.query, sql, params=params, max_rows=max_rows)

    def _enforce_allowlist(self, sql, permit=frozenset()):
        """With a file_allowlist set, every non-safe table-function call in the
        SQL must carry a literal source argument inside the allowlist.

        The scan runs over masked SQL (string literals/comments blanked, quoted
        function names matched), against the table functions this engine
        actually exposes — so a call with a computed/concatenated source, or
        any external source function outside the allowlist, is rejected rather
        than slipping through a literal-only regex. readonly=2 remains the
        write backstop; OS sandboxing the filesystem backstop. No allowlist
        configured -> no restriction."""
        if not self.file_allowlist:
            return
        known = self._known_table_functions or FALLBACK_KNOWN_TABLE_FUNCTIONS
        for fn, arg in find_source_calls(sql, known):
            if fn in permit:
                continue
            if arg is None:
                raise ChDBError(
                    "table function {!r} without a literal source argument is not "
                    "allowed when file_allowlist is set".format(fn),
                    type="ALLOWLIST_DENIED",
                )
            if not path_allowed(arg, self.file_allowlist):
                raise ChDBError(
                    "source path not in file_allowlist: {!r}".format(arg),
                    type="ALLOWLIST_DENIED",
                )

    # ---- source catalog ---------------------------------------------------

    def _create_file_view(self, name, path, format=None):
        """CREATE VIEW <name> AS SELECT * FROM file(<path>[, <format>]).

        The view name is quoted as an identifier; the path/format are baked in as
        string literals (a stored view definition can't carry bound params), so
        they go through quote_string. Gated by the allowlist. This is a write, so
        it only succeeds before the read-only lock (see __init__ / attach_file).
        """
        if self.file_allowlist and not path_allowed(path, self.file_allowlist):
            raise ChDBError(
                "attach path not in file_allowlist: {!r}".format(path), type="ALLOWLIST_DENIED"
            )
        src = "file({}".format(quote_string(path))
        if format:
            src += ", {}".format(quote_string(format))
        src += ")"
        self._session.query(
            "CREATE VIEW {} AS SELECT * FROM {}".format(quote_ident(name), src), "CSV"
        )

    def attach_file(self, name, path, format=None):
        """Register a local file as a queryable named table (a view over file()).

        On a read-only tool this raises (CREATE VIEW is a write) — declare files
        via the `attachments=` constructor arg instead, which attaches them before
        the read-only lock. On a writable tool it works at any time.
        """
        if self.read_only:
            raise ChDBReadOnlyError(
                "attach_file needs a writable tool; for a read-only tool pass "
                "attachments={{'{}': '{}'}} to the constructor (attached before the "
                "read-only lock)".format(name, path),
                code=164,
                type="READONLY",
            )
        try:
            self._create_file_view(name, path, format)
        except ChDBError:
            raise
        except Exception as e:
            raise parse_error(e)
        return name

    def dataframe_query(self, sql, dataframes, *, max_rows=None):
        """Query in-process pandas DataFrames via chDB's `Python()` table function.

        Python-only / co-located capability (not part of the cross-language
        contract): `dataframes` maps the name used in `Python(<name>)` to a
        DataFrame. Reference them as `SELECT ... FROM Python(orders)` with
        `dataframes={'orders': df}`. The names are injected into this module's
        globals only for the duration of the call (that is where `Python()`
        resolves them from) and restored afterward.
        """
        if not isinstance(dataframes, dict) or not dataframes:
            raise ChDBError("dataframe_query requires a non-empty {name: DataFrame} mapping")
        g = globals()
        saved = {k: g.get(k, _MISSING) for k in dataframes}
        try:
            g.update(dataframes)
            # Python() resolves the names this method itself injected, so the
            # allowlist gate (which treats python as an RCE-class source) is
            # lifted for exactly this one function on this one call.
            return self.query(sql, max_rows=max_rows, _permit_fns=frozenset(("python",)))
        finally:
            for k, v in saved.items():
                if v is _MISSING:
                    g.pop(k, None)
                else:
                    g[k] = v

    # ---- introspection ----------------------------------------------------

    def list_databases(self):
        return [r["name"] for r in self.query("SHOW DATABASES").rows]

    def list_tables(self, database=None):
        if database is None:
            sql = "SELECT name FROM system.tables WHERE database = currentDatabase() ORDER BY name"
            return [r["name"] for r in self.query(sql).rows]
        sql = "SELECT name FROM system.tables WHERE database = {db:String} ORDER BY name"
        return [r["name"] for r in self.query(sql, params={"db": database}).rows]

    def _qualify(self, target, database=None):
        """Turn (target[, database]) into a safe SQL source reference.

        - `target` containing '(' is a table-function expression, passed through
          as SQL (its literal args are the one place a value rides in text —
          read-only + no-write makes that inert); a database qualifier is invalid.
        - otherwise `target` is a table identifier, backtick-quoted; when
          `database` is given each part is quoted independently as `db`.`table`
          (so a dotted name is never mis-quoted as a single identifier). This is
          what lets the mcp-clickhouse `(database, table)` tools map onto ChDBTool.
        """
        # `None` means "not provided"; any other value (including "") is a real
        # database argument and must be validated — an empty string flows into
        # quote_ident() and is rejected rather than silently treated as unqualified.
        if "(" in target:
            if database is not None:
                raise ChDBError("database qualifier is not valid for a table-function target")
            return target
        ident = quote_ident(target)
        return "{}.{}".format(quote_ident(database), ident) if database is not None else ident

    def describe(self, target, *, database=None, params=None):
        """Describe a table (optionally `database`-qualified) OR a table-function
        expression (e.g. file('x.parquet'))."""
        ref = self._qualify(target, database)
        rows = self.query("DESCRIBE TABLE {} ".format(ref), params=params).rows
        return [
            {"name": r.get("name"), "type": r.get("type"),
             "default_kind": r.get("default_type") or "", "comment": r.get("comment") or ""}
            for r in rows
        ]

    def get_sample_data(self, target, *, database=None, limit=5):
        ref = self._qualify(target, database)
        n = _int_arg(limit, "limit")
        return self.query(
            "SELECT * FROM {} LIMIT {{n:UInt32}}".format(ref),
            params={"n": n},
            max_rows=n,
        )

    def list_functions(self, *, like=None, limit=200):
        n = _int_arg(limit, "limit")
        if like:
            sql = "SELECT name FROM system.functions WHERE name ILIKE {like:String} ORDER BY name LIMIT {n:UInt32}"
            rows = self.query(sql, params={"like": like, "n": n}, max_rows=n).rows
        else:
            sql = "SELECT name FROM system.functions ORDER BY name LIMIT {n:UInt32}"
            rows = self.query(sql, params={"n": n}, max_rows=n).rows
        return [r["name"] for r in rows]

    # ---- agent integration ------------------------------------------------

    def tool_specs(self, dialect="anthropic"):
        """Tool definitions for auto-registration into any framework, generated
        from descriptors.json (the single source of the model-visible surface).
        `dialect` selects the shape: 'anthropic' | 'openai' | 'mcp'."""
        return _tool_specs(dialect)

    def call(self, name, arguments=None):
        """Dispatch a tool call, returning an error ENVELOPE instead of raising,
        so the model reads the engine message and can self-correct (P4)."""
        method_name = _TOOL_METHODS.get(name)
        if method_name is None:
            return {"ok": False, "error": {"code": 0, "type": "UNKNOWN_TOOL", "message": "unknown tool: " + str(name)}}
        # Caller mistakes on the dispatch path never throw (P4): a non-object
        # arguments payload comes back as an envelope, same as an unknown tool.
        # (dict("...") would raise here, and the TypeScript spread would
        # silently turn a string into {0: 'S', 1: 'E', ...} garbage.)
        if arguments is not None and not isinstance(arguments, dict):
            return {"ok": False, "error": {"code": 0, "type": "INVALID_ARGUMENT",
                                           "message": "arguments must be an object, got " + type(arguments).__name__}}
        args = dict(arguments or {})
        try:
            method = getattr(self, method_name)
            if method_name == "query":
                out = method(args.get("sql", ""), params=args.get("params"))
                result = out.to_dict()
            elif method_name == "describe":
                result = method(args["target"], database=args.get("database"))
            elif method_name == "get_sample_data":
                # In the model-facing envelope a JSON null argument means "omitted"
                # (models routinely send null for optional args); the direct
                # method path stays strict and raises INVALID_ARGUMENT on None.
                limit = args.get("limit")
                result = method(args["target"], database=args.get("database"), limit=5 if limit is None else limit).to_dict()
            elif method_name == "list_tables":
                result = method(args.get("database"))
            elif method_name == "list_functions":
                limit = args.get("limit")
                result = method(like=args.get("like"), limit=200 if limit is None else limit)
            elif method_name == "attach_file":
                result = method(args["name"], args["path"], args.get("format"))
            else:
                result = method()
            return {"ok": True, "result": result}
        except ChDBError as e:
            return {"ok": False, "error": e.to_dict()}
        except Exception as e:  # non-engine failure still reaches the model
            return {"ok": False, "error": {"code": 0, "type": "TOOL_ERROR", "message": str(e)}}

    async def acall(self, name, arguments=None):
        """Async form of `call` for async-first frameworks (AutoGen's run_json,
        Pydantic AI, ...). chDB has no native async engine call, so this runs
        `call` in a worker thread (documented, not faked) — same as `aquery`."""
        return await asyncio.to_thread(self.call, name, arguments)

    def close(self):
        if self._poisoned:
            # Abandoned call may still be inside the session; don't touch it.
            self._session = None
            return
        if self._owns_session and self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
