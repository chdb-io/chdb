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

from .errors import ChDBError, parse_error
from .safety import quote_ident

__all__ = ["ChDBTool", "QueryResult"]


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
        session=None,
    ):
        self.read_only = bool(read_only)
        self.max_rows = max(1, int(max_rows))
        self.max_bytes = max(1, int(max_bytes))
        self._owns_session = session is None
        if session is not None:
            self._session = session
        else:
            # chdb.session.Session() with no path == ephemeral :memory:.
            from chdb.session import Session

            self._session = Session() if path in (":memory:", "", None) else Session(path)

        # Exact 64-bit integers survive JSON as strings rather than lossy floats.
        self._session.query("SET output_format_json_quote_64bit_integers=1", "CSV")
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

    # ---- introspection ----------------------------------------------------

    def list_databases(self):
        return [r["name"] for r in self.query("SHOW DATABASES").rows]

    def list_tables(self, database=None):
        if database is None:
            sql = "SELECT name FROM system.tables WHERE database = currentDatabase() ORDER BY name"
            return [r["name"] for r in self.query(sql).rows]
        sql = "SELECT name FROM system.tables WHERE database = {db:String} ORDER BY name"
        return [r["name"] for r in self.query(sql, params={"db": database}).rows]

    def describe(self, target, *, params=None):
        """Describe a table OR a table-function expression (e.g. file('x.parquet')).

        A bare identifier is backtick-quoted; an expression containing '(' is a
        table function and passed through as SQL (its literal args are the one
        place a value rides in text — read-only + no-write makes that inert).
        """
        ref = target if "(" in target else quote_ident(target)
        rows = self.query("DESCRIBE TABLE {} ".format(ref), params=params).rows
        return [
            {"name": r.get("name"), "type": r.get("type"),
             "default_kind": r.get("default_type") or "", "comment": r.get("comment") or ""}
            for r in rows
        ]

    def get_sample_data(self, target, *, limit=5):
        ref = target if "(" in target else quote_ident(target)
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
            {"name": "describe_table", "description": "Describe a table or table function (columns and types).",
             "input_schema": s(target={"type": "string"})},
            {"name": "get_sample_data", "description": "Return a few sample rows from a table or table function.",
             "input_schema": s(target={"type": "string"}, limit={"type": "integer"})},
            {"name": "list_functions", "description": "List available SQL functions.",
             "input_schema": s(like={"type": "string"}, limit={"type": "integer"})},
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
                result = method(args["target"])
            elif method_name == "get_sample_data":
                result = method(args["target"], limit=args.get("limit", 5)).to_dict()
            elif method_name == "list_tables":
                result = method(args.get("database"))
            elif method_name == "list_functions":
                result = method(like=args.get("like"), limit=args.get("limit", 200))
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
