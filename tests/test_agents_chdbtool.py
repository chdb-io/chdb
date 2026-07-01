"""Python-specific unit tests for chdb.agents.ChDBTool.

The cross-language behaviors live in test_agents_conformance.py (shared fixture);
this file covers Python-only concerns: the writable opt-out, the error parser,
identifier quoting, tool_specs shape, and aquery.
"""

import asyncio
import os
import tempfile
import unittest

from chdb.agents import ChDBTool, ChDBError, ChDBReadOnlyError, quote_ident, InvalidIdentifier
from chdb.agents.errors import parse_error
from chdb.agents.safety import quote_string


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

    def test_dataframe_query_requires_mapping(self):
        tool = ChDBTool(read_only=True)
        try:
            with self.assertRaises(ChDBError):
                tool.dataframe_query("SELECT 1", {})
        finally:
            tool.close()


if __name__ == "__main__":
    unittest.main()
