"""Python-specific unit tests for chdb.agents.ChDBTool.

The cross-language behaviors live in test_agents_conformance.py (shared fixture);
this file covers Python-only concerns: the writable opt-out, the error parser,
identifier quoting, tool_specs shape, and aquery.
"""

import asyncio
import os
import tempfile
import unittest

from chdb.agents import (
    CONTRACT_VERSION,
    ChDBTool,
    ChDBError,
    ChDBReadOnlyError,
    InvalidIdentifier,
    capabilities,
    load_descriptors,
    quote_ident,
    tool_specs,
)
from chdb.agents.errors import parse_error
from chdb.agents.safety import quote_string
from chdb.agents.tool import _TOOL_METHODS


def _sample_csv():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sample.csv")
    with open(p, "w") as fh:
        fh.write("id,name\n1,alice\n2,bob\n")
    return p


class TestErrorParser(unittest.TestCase):
    def test_parse_readonly(self):
        e = parse_error("Code: 164. DB::Exception: Cannot execute query in readonly mode. (READONLY)")
        self.assertIsInstance(e, ChDBReadOnlyError)
        self.assertEqual(e.code, 164)
        self.assertEqual(e.type, "READONLY")

    def test_parse_unknown_function(self):
        e = parse_error("Code: 46. DB::Exception: Function with name `x` does not exist. (UNKNOWN_FUNCTION)")
        self.assertEqual(e.code, 46)
        self.assertEqual(e.type, "UNKNOWN_FUNCTION")

    def test_parse_nonconforming(self):
        e = parse_error("some random text")
        self.assertIsInstance(e, ChDBError)
        self.assertEqual(e.type, "UNKNOWN")

    def test_parse_type_with_parenthesized_body(self):
        # a parenthesized UPPER_SNAKE inside the message body must NOT hijack the
        # trailing type (regression for the greedy-vs-non-greedy regex fix)
        e = parse_error(
            "Code: 62. DB::Exception: Cannot parse (SOME_ENUM value) near x. (SYNTAX_ERROR)"
        )
        self.assertEqual(e.code, 62)
        self.assertEqual(e.type, "SYNTAX_ERROR")
        self.assertIn("SOME_ENUM", e.message)


class TestQuoteIdent(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(quote_ident("events"), "`events`")

    def test_backtick_doubled(self):
        self.assertEqual(quote_ident("we`ird"), "`we``ird`")

    def test_reject_nul(self):
        with self.assertRaises(InvalidIdentifier):
            quote_ident("a\x00b")

    def test_reject_empty(self):
        with self.assertRaises(InvalidIdentifier):
            quote_ident("")


class TestToolSpecs(unittest.TestCase):
    def test_specs_shape(self):
        tool = ChDBTool(read_only=True)
        try:
            specs = tool.tool_specs()
            names = {s["name"] for s in specs}
            # canonical names must match mcp-clickhouse for corpus consistency
            self.assertTrue({"run_select_query", "list_databases", "list_tables",
                             "describe_table", "get_sample_data", "list_functions"} <= names)
            for s in specs:
                self.assertIn("description", s)
                self.assertEqual(s["input_schema"]["type"], "object")
        finally:
            tool.close()

    def test_dialects(self):
        # module-level generation from descriptors.json, no session needed
        anthropic = tool_specs("anthropic")
        openai = tool_specs("openai")
        mcp = tool_specs("mcp")
        self.assertEqual(len(anthropic), len(openai))
        self.assertEqual(len(anthropic), len(mcp))
        run = anthropic[0]
        self.assertEqual(run["name"], "run_select_query")
        self.assertEqual(run["input_schema"]["required"], ["sql"])
        self.assertIn("description", run["input_schema"]["properties"]["sql"])
        fn = openai[0]
        self.assertEqual(fn["type"], "function")
        self.assertEqual(fn["function"]["name"], "run_select_query")
        self.assertEqual(fn["function"]["parameters"], run["input_schema"])
        self.assertEqual(mcp[0]["inputSchema"], run["input_schema"])

    def test_unknown_dialect_is_typed_error(self):
        with self.assertRaises(ChDBError) as ctx:
            tool_specs("langchain")
        self.assertEqual(ctx.exception.type, "INVALID_ARGUMENT")

    def test_descriptors_cover_exactly_the_dispatch_table(self):
        # every descriptor is dispatchable and every dispatchable tool is described
        self.assertEqual(
            {t["name"] for t in load_descriptors()["tools"]}, set(_TOOL_METHODS)
        )

    def test_load_descriptors_returns_a_defensive_copy(self):
        # a caller mutating the result must not corrupt what tool_specs()/
        # capabilities() generate for everyone else in-process
        d = load_descriptors()
        d["contract_version"] = "9.9.9"
        d["tools"][0]["name"] = "mutated"
        d["tools"][0]["params"].clear()
        fresh = load_descriptors()
        self.assertEqual(fresh["contract_version"], CONTRACT_VERSION)
        self.assertEqual(fresh["tools"][0]["name"], "run_select_query")
        self.assertEqual(tool_specs()[0]["name"], "run_select_query")
        self.assertEqual(tool_specs()[0]["input_schema"]["required"], ["sql"])

    def test_contract_version_single_source(self):
        self.assertEqual(load_descriptors()["contract_version"], CONTRACT_VERSION)
        caps = capabilities()
        self.assertEqual(caps["contract_version"], CONTRACT_VERSION)
        self.assertTrue(caps["features"]["dataframe_query"])  # Python-only capability
        self.assertEqual(set(caps["tools"]), set(_TOOL_METHODS))


class TestReadOnlyOptOut(unittest.TestCase):
    def test_readonly_blocks_write(self):
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(ChDBReadOnlyError):
                tool.query("CREATE TABLE x (a Int32) ENGINE = Memory")
        finally:
            tool.close()

    def test_writable_allows_write(self):
        # NOTE: single-engine-per-process — run standalone. Uses read_only=False.
        tool = ChDBTool(read_only=False)
        try:
            tool.query("CREATE TABLE x (a Int32) ENGINE = Memory")
            tool.query("INSERT INTO x VALUES (1), (2)")
            r = tool.query("SELECT count() AS c FROM x")
            self.assertEqual(r.rows[0]["c"], "2")  # UInt64 -> exact string
        finally:
            tool.close()


class TestAquery(unittest.IsolatedAsyncioTestCase):
    async def test_aquery_matches_query(self):
        tool = ChDBTool(read_only=True)
        try:
            sync = tool.query("SELECT toInt32(1) AS x").rows
            asy = (await tool.aquery("SELECT toInt32(1) AS x")).rows
            self.assertEqual(sync, asy)
        finally:
            tool.close()

    async def test_acall_matches_call(self):
        tool = ChDBTool(read_only=True)
        try:
            args = {"sql": "SELECT toInt32(7) AS x"}
            sync = tool.call("run_select_query", args)
            asy = await tool.acall("run_select_query", args)
            self.assertTrue(sync["ok"] and asy["ok"])
            self.assertEqual(sync["result"]["rows"], asy["result"]["rows"])
            # the error envelope crosses the thread boundary intact (P4)
            bad = await tool.acall("run_select_query", {"sql": "SELECT BAD_FUNC()"})
            self.assertFalse(bad["ok"])
            self.assertEqual(bad["error"]["type"], "UNKNOWN_FUNCTION")
        finally:
            tool.close()


class TestArgumentValidation(unittest.TestCase):
    def test_non_numeric_max_rows_is_typed(self):
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(ChDBError) as ctx:
                tool.query("SELECT 1", max_rows="lots")
            self.assertEqual(ctx.exception.type, "INVALID_ARGUMENT")
        finally:
            tool.close()

    def test_non_numeric_constructor_cap_is_typed(self):
        with self.assertRaises(ChDBError) as ctx:
            ChDBTool(max_rows="lots")
        self.assertEqual(ctx.exception.type, "INVALID_ARGUMENT")

    def test_call_treats_null_limit_as_omitted(self):
        # models routinely send null for optional args on the envelope path
        tool = ChDBTool(read_only=True)
        try:
            out = tool.call("get_sample_data", {"target": "numbers(100)", "limit": None})
            self.assertTrue(out["ok"])
            self.assertEqual(out["result"]["row_count"], 5)
        finally:
            tool.close()

    def test_call_malformed_arguments_returns_envelope(self):
        # the dispatch path never throws for caller mistakes (P4): a non-object
        # arguments payload is an envelope, same as an unknown tool name
        tool = ChDBTool(read_only=True)
        try:
            for bad in ("SELECT 1", 42, ["sql"]):
                out = tool.call("run_select_query", bad)
                self.assertFalse(out["ok"])
                self.assertEqual(out["error"]["type"], "INVALID_ARGUMENT")
        finally:
            tool.close()


class TestQuoteString(unittest.TestCase):
    def test_escapes_quote_and_backslash(self):
        self.assertEqual(quote_string("a'b"), "'a\\'b'")
        self.assertEqual(quote_string("a\\b"), "'a\\\\b'")

    def test_reject_nul(self):
        with self.assertRaises(InvalidIdentifier):
            quote_string("a\x00b")


class TestAttachFile(unittest.TestCase):
    def test_attach_file_readonly_raises(self):
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(ChDBReadOnlyError):
                tool.attach_file("rep", _sample_csv())
        finally:
            tool.close()

    def test_attach_file_writable(self):
        tool = ChDBTool(read_only=False)
        try:
            tool.attach_file("rep_w", _sample_csv())
            self.assertEqual(tool.query("SELECT toInt32(count()) AS c FROM rep_w").rows, [{"c": 2}])
        finally:
            tool.close()


class TestQualifiedName(unittest.TestCase):
    def test_describe_and_sample_with_database(self):
        # (database, table) qualification is what lets mcp-clickhouse's
        # describe_table(database, table) / get_sample_data(...) map onto ChDBTool.
        tool = ChDBTool(read_only=False)
        try:
            tool.query("CREATE DATABASE dq")
            tool.query("CREATE TABLE dq.t (x Int32, y String) ENGINE = MergeTree ORDER BY x")
            tool.query("INSERT INTO dq.t VALUES (1, 'a'), (2, 'b')")
            cols = [c["name"] for c in tool.describe("t", database="dq")]
            self.assertEqual(cols, ["x", "y"])
            self.assertEqual(tool.get_sample_data("t", database="dq", limit=1).row_count, 1)
        finally:
            tool.close()

    def test_database_qualifier_rejected_for_table_function(self):
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(ChDBError):
                tool.describe("numbers(5)", database="dq")
        finally:
            tool.close()

    def test_empty_string_database_is_rejected_not_ignored(self):
        # an explicit "" must not be silently treated as unqualified
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(InvalidIdentifier):
                tool.describe("t", database="")
        finally:
            tool.close()


class TestSourceScanner(unittest.TestCase):
    # engine-free unit tests for the masked table-function scanner
    def _calls(self, sql):
        from chdb.agents.safety import FALLBACK_KNOWN_TABLE_FUNCTIONS, find_source_calls

        return find_source_calls(sql, FALLBACK_KNOWN_TABLE_FUNCTIONS)

    def test_plain_and_quoted_names_match(self):
        self.assertEqual(self._calls("SELECT * FROM file('/x')"), [("file", "/x")])
        self.assertEqual(self._calls("SELECT * FROM `file`('/x')"), [("file", "/x")])
        self.assertEqual(self._calls('SELECT * FROM "file"(\'/x\')'), [("file", "/x")])

    def test_comments_cannot_hide_a_call(self):
        self.assertEqual(self._calls("SELECT * FROM file/*c*/('/x')"), [("file", "/x")])
        self.assertEqual(self._calls("SELECT * FROM file--c\n('/x')"), [("file", "/x")])

    def test_string_literals_do_not_false_positive(self):
        self.assertEqual(self._calls("SELECT 'file(''/etc/passwd'')' AS s"), [])
        self.assertEqual(self._calls("SELECT '/* url(''x'') */' AS s"), [])

    def test_non_literal_arg_is_surfaced_as_none(self):
        self.assertEqual(self._calls("SELECT * FROM file(concat('a','b'))"), [("file", None)])
        self.assertEqual(self._calls("SELECT * FROM url(myvar)"), [("url", None)])

    def test_scalar_functions_never_flag(self):
        self.assertEqual(self._calls("SELECT sum(x), length(s) FROM t"), [])

    def test_literal_arg_is_unescaped(self):
        self.assertEqual(self._calls("SELECT * FROM file('a''b')"), [("file", "a'b")])
        self.assertEqual(self._calls("SELECT * FROM file('a\\'b')"), [("file", "a'b")])


class TestExternalSessionProbe(unittest.TestCase):
    def test_probe_mismatch_and_match(self):
        from chdb.session import Session

        s = Session()
        try:
            # a fresh session is readonly=0; declaring read_only=True must NOT
            # silently SET readonly=2 on the caller's session — it must refuse
            with self.assertRaises(ChDBError) as ctx:
                ChDBTool(session=s)
            self.assertEqual(ctx.exception.type, "CONFIG_MISMATCH")
            # the declared-writable form matches and works
            tool = ChDBTool(session=s, read_only=False)
            self.assertEqual(tool.query("SELECT toInt32(1) AS x").rows, [{"x": 1}])
            tool.close()  # must be a no-op on a session we do not own
            self.assertEqual(
                s.query("SELECT toInt32(2) AS x", "CSV").bytes().decode().strip(), "2"
            )
        finally:
            s.close()


class TestConstructorLeak(unittest.TestCase):
    def test_owned_session_closed_when_setup_throws(self):
        # a bad attachment under an allowlist throws ACCESS_DENIED during setup;
        # the Session the tool just created must be closed before the rethrow
        # (verified by the next construction working — chDB is single-engine)
        with self.assertRaises(ChDBError):
            ChDBTool(file_allowlist=["/allowed-prefix/"], attachments={"rep": "/elsewhere/x.csv"})
        tool = ChDBTool(read_only=True)
        try:
            self.assertEqual(tool.query("SELECT toInt32(1) AS x").rows, [{"x": 1}])
        finally:
            tool.close()


class TestDataFrameQuery(unittest.TestCase):
    def test_dataframe_query(self):
        import pandas as pd

        tool = ChDBTool(read_only=True)
        try:
            df = pd.DataFrame({"c": ["a", "a", "b"], "p": [1.0, 2.0, 3.0]})
            r = tool.dataframe_query(
                "SELECT c, sum(p) AS s FROM Python(orders) GROUP BY c ORDER BY c",
                {"orders": df},
            )
            self.assertEqual(r.rows, [{"c": "a", "s": 3}, {"c": "b", "s": 3}])
        finally:
            tool.close()

    def test_dataframe_query_permitted_under_allowlist(self):
        # the gate treats python as an RCE-class source; dataframe_query lifts
        # it only for the names it itself injects — raw query() stays denied
        import pandas as pd

        tool = ChDBTool(read_only=True, file_allowlist=["/nonexistent-prefix/"])
        try:
            df = pd.DataFrame({"a": [1, 2, 3]})
            r = tool.dataframe_query("SELECT toInt32(sum(a)) AS s FROM Python(t)", {"t": df})
            self.assertEqual(r.rows, [{"s": 6}])
            with self.assertRaises(ChDBError) as ctx:
                tool.query("SELECT * FROM Python(t)")
            self.assertEqual(ctx.exception.type, "ACCESS_DENIED")
        finally:
            tool.close()

    def test_dataframe_query_requires_mapping(self):
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(ChDBError):
                tool.dataframe_query("SELECT 1", {})
        finally:
            tool.close()


if __name__ == "__main__":
    unittest.main()
