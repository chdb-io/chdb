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

from .errors import ChDBError, ChDBReadOnlyError, parse_error
from .safety import path_allowed, quote_ident, quote_string, scan_file_paths

__all__ = ["ChDBTool", "QueryResult"]

_MISSING = object()  # sentinel for "name absent from globals" in dataframe_query


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
        return {
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "column_names": self.column_names,
            "elapsed_s": self.elapsed_s,
            "bytes_read": self.bytes_read,
        }

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
        file_allowlist=None,
        attachments=None,
        session=None,
    ):
        self.read_only = bool(read_only)
        self.max_rows = max(1, int(max_rows))
        self.max_bytes = max(1, int(max_bytes))
        self.max_execution_time = None if max_execution_time is None else max(0, int(max_execution_time))
        # None = no allowlist (all paths allowed); a list = only these prefixes.
        self.file_allowlist = list(file_allowlist) if file_allowlist else None
        self._owns_session = session is None
        if session is not None:
            self._session = session
        else:
            # chdb.session.Session() with no path == ephemeral :memory:.
            from chdb.session import Session

            self._session = Session() if path in (":memory:", "", None) else Session(path)

        # Exact 64-bit integers survive JSON as strings rather than lossy floats.
        self._session.query("SET output_format_json_quote_64bit_integers=1", "CSV")
        if self.max_execution_time is not None:
            # engine-side wall-clock bound; a runaway query raises TIMEOUT_EXCEEDED
            self._session.query("SET max_execution_time={}".format(self.max_execution_time), "CSV")
        # Attachments must be materialized BEFORE the read-only lock, because
        # CREATE VIEW is a write that readonly=2 rejects. This is why read-only
        # tools declare files at construction rather than via attach_file().
        for name, spec in (attachments or {}).items():
            p, fmt = spec if isinstance(spec, (tuple, list)) else (spec, None)
            self._create_file_view(name, p, fmt)
        if self.read_only:
            # readonly=2 (NOT 1): blocks INSERT/CREATE/ALTER/DROP while still
            # allowing SELECT and the file()/s3()/url() table functions that are
            # chDB's whole point. readonly=1 rejects those. Cannot be un-set.
            self._session.query("SET readonly=2", "CSV")

    # ---- core query -------------------------------------------------------

    def query(self, sql, *, params=None, max_rows=None):
        """Run read SQL. Values MUST be passed via `params` ({name:Type} + dict),
        never formatted into `sql`. Returns a `QueryResult`; raises `ChDBError`.
        """
        if not isinstance(sql, str) or sql.strip() == "":
            raise ChDBError("sql must be a non-empty string")
        self._enforce_allowlist(sql)
        cap = self.max_rows if max_rows is None else max(1, int(max_rows))
        # Both the engine call and the decode go through parse_error: malformed or
        # non-JSON engine output (edge-case statements, empty results) becomes a
        # typed ChDBError rather than a bare JSONDecodeError leaking to the caller.
        try:
            res = self._session.query(sql, "JSON", params=params or {})
            obj = json.loads(res.bytes().decode() or "{}")
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
        if self.max_bytes:
            size = 0
            for i, r in enumerate(rows):
                size += len(json.dumps(r, separators=(",", ":")))
                if size > self.max_bytes:
                    rows = rows[:i]
                    truncated = True
                    break
        return QueryResult(rows, truncated, cols, stats.get("elapsed"), stats.get("bytes_read"))

    async def aquery(self, sql, *, params=None, max_rows=None):
        """Async wrapper. chDB has no native async engine call, so this runs
        `query` in a worker thread (documented, not faked)."""
        return await asyncio.to_thread(self.query, sql, params=params, max_rows=max_rows)

    def _enforce_allowlist(self, sql):
        """If a file_allowlist is set, reject file()/s3()/url() literal paths that
        fall outside it. Best-effort (literal args only); readonly=2 is the real
        write backstop. No allowlist configured -> no restriction."""
        if not self.file_allowlist:
            return
        for _fn, path in scan_file_paths(sql):
            if not path_allowed(path, self.file_allowlist):
                raise ChDBError(
                    "source path not in file_allowlist: {!r}".format(path),
                    type="ACCESS_DENIED",
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
                "attach path not in file_allowlist: {!r}".format(path), type="ACCESS_DENIED"
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
            return self.query(sql, max_rows=max_rows)
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
        if "(" in target:
            if database:
                raise ChDBError("database qualifier is not valid for a table-function target")
            return target
        ident = quote_ident(target)
        return "{}.{}".format(quote_ident(database), ident) if database else ident

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
        return self.query(
            "SELECT * FROM {} LIMIT {{n:UInt32}}".format(ref),
            params={"n": int(limit)},
            max_rows=int(limit),
        )

    def list_functions(self, *, like=None, limit=200):
        if like:
            sql = "SELECT name FROM system.functions WHERE name ILIKE {like:String} ORDER BY name LIMIT {n:UInt32}"
            rows = self.query(sql, params={"like": like, "n": int(limit)}, max_rows=int(limit)).rows
        else:
            sql = "SELECT name FROM system.functions ORDER BY name LIMIT {n:UInt32}"
            rows = self.query(sql, params={"n": int(limit)}, max_rows=int(limit)).rows
        return [r["name"] for r in rows]

    # ---- agent integration ------------------------------------------------

    def tool_specs(self):
        """JSON-schema tool definitions for the callables, for auto-registration
        into any framework. Names match mcp-clickhouse for corpus consistency."""
        s = lambda **p: {"type": "object", "properties": p}
        return [
            {"name": "run_select_query", "description": "Run a read-only ClickHouse SQL query via chDB and return rows.",
             "input_schema": s(sql={"type": "string"}, params={"type": "object"})},
            {"name": "list_databases", "description": "List databases.", "input_schema": s()},
            {"name": "list_tables", "description": "List tables in a database (current if omitted).",
             "input_schema": s(database={"type": "string"})},
            {"name": "describe_table", "description": "Describe a table (optionally database-qualified) or table function.",
             "input_schema": s(target={"type": "string"}, database={"type": "string"})},
            {"name": "get_sample_data", "description": "Return a few sample rows from a table or table function.",
             "input_schema": s(target={"type": "string"}, database={"type": "string"}, limit={"type": "integer"})},
            {"name": "list_functions", "description": "List available SQL functions.",
             "input_schema": s(like={"type": "string"}, limit={"type": "integer"})},
            {"name": "attach_file", "description": "Register a local file as a queryable named table (writable tools only).",
             "input_schema": s(name={"type": "string"}, path={"type": "string"}, format={"type": "string"})},
        ]

    def call(self, name, arguments=None):
        """Dispatch a tool call, returning an error ENVELOPE instead of raising,
        so the model reads the engine message and can self-correct (P4)."""
        args = dict(arguments or {})
        method_name = _TOOL_METHODS.get(name)
        if method_name is None:
            return {"ok": False, "error": {"code": 0, "type": "UNKNOWN_TOOL", "message": "unknown tool: " + str(name)}}
        try:
            method = getattr(self, method_name)
            if method_name == "query":
                out = method(args.get("sql", ""), params=args.get("params"))
                result = out.to_dict()
            elif method_name == "describe":
                result = method(args["target"], database=args.get("database"))
            elif method_name == "get_sample_data":
                result = method(args["target"], database=args.get("database"), limit=args.get("limit", 5)).to_dict()
            elif method_name == "list_tables":
                result = method(args.get("database"))
            elif method_name == "list_functions":
                result = method(like=args.get("like"), limit=args.get("limit", 200))
            elif method_name == "attach_file":
                result = method(args["name"], args["path"], args.get("format"))
            else:
                result = method()
            return {"ok": True, "result": result}
        except ChDBError as e:
            return {"ok": False, "error": e.to_dict()}
        except Exception as e:  # non-engine failure still reaches the model
            return {"ok": False, "error": {"code": 0, "type": "TOOL_ERROR", "message": str(e)}}

    def close(self):
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
