"""Tests for chdb.agents.smolagents (skipped when smolagents is not installed).

Instantiation itself is half the test surface: smolagents validates each
tool's `inputs` metadata against the `forward()` signature (key presence and
nullable agreement) at construction time.
"""

import json
import os
import tempfile
import unittest

try:
    import smolagents  # noqa: F401

    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False


READ_ONLY_TOOLS = [
    "run_select_query",
    "list_databases",
    "list_tables",
    "describe_table",
    "get_sample_data",
    "list_functions",
]


@unittest.skipUnless(HAS_SMOLAGENTS, "smolagents not installed")
class TestSmolagentsTools(unittest.TestCase):
    def test_suite_names_and_attach_file_gating(self):
        from chdb.agents.smolagents import chdb_smol_tools

        read_only = chdb_smol_tools()
        try:
            self.assertEqual([t.name for t in read_only], READ_ONLY_TOOLS)
        finally:
            read_only[0].engine.close()

        writable = chdb_smol_tools(read_only=False)
        try:
            self.assertEqual([t.name for t in writable][-1], "attach_file")
        finally:
            writable[0].engine.close()

    def test_descriptions_and_inputs_come_from_descriptors(self):
        from chdb.agents import load_descriptors
        from chdb.agents.smolagents import chdb_smol_tools

        descriptors = {t["name"]: t for t in load_descriptors()["tools"]}
        tools = chdb_smol_tools()
        try:
            for tool in tools:
                descriptor = descriptors[tool.name]
                self.assertEqual(tool.description, descriptor["description"])
                params = {p["name"]: p for p in descriptor.get("params", [])}
                self.assertEqual(set(tool.inputs), set(params))
                for name, entry in tool.inputs.items():
                    self.assertEqual(entry["type"], params[name]["type"])
                    self.assertEqual(
                        entry.get("nullable", False),
                        not params[name].get("required", False),
                    )
        finally:
            tools[0].engine.close()

    def test_call_path_returns_envelopes(self):
        from chdb.agents.smolagents import ChDBRunSelectQueryTool

        tool = ChDBRunSelectQueryTool()
        try:
            ok = json.loads(tool(sql="SELECT 21 * 2 AS x"))
            self.assertTrue(ok["ok"])
            self.assertEqual(ok["result"]["rows"], [{"x": 42}])

            err = json.loads(tool(sql="SELEC bad"))
            self.assertFalse(err["ok"])
            self.assertEqual(err["error"]["type"], "UNKNOWN_IDENTIFIER")

            readonly = json.loads(
                tool(sql="CREATE TABLE t (x Int64) ENGINE = Memory")
            )
            self.assertFalse(readonly["ok"])
        finally:
            tool.close()

    def test_attach_then_query_shares_engine(self):
        from chdb.agents.smolagents import chdb_smol_tools

        d = tempfile.mkdtemp()
        csv = os.path.join(d, "people.csv")
        with open(csv, "w") as fh:
            fh.write("name,age\nada,36\ngrace,45\n")

        tools = {t.name: t for t in chdb_smol_tools(read_only=False)}
        try:
            attached = json.loads(tools["attach_file"](name="people", path=csv))
            self.assertTrue(attached["ok"])
            queried = json.loads(
                tools["run_select_query"](sql="SELECT count() AS n FROM people")
            )
            self.assertTrue(queried["ok"])
            self.assertEqual(queried["result"]["rows"], [{"n": "2"}])
        finally:
            tools["run_select_query"].engine.close()

    def test_injected_engine_not_closed(self):
        from chdb.agents import ChDBTool
        from chdb.agents.smolagents import ChDBRunSelectQueryTool

        engine = ChDBTool()
        try:
            tool = ChDBRunSelectQueryTool(engine=engine)
            tool.close()
            self.assertTrue(engine.call("run_select_query", {"sql": "SELECT 1"})["ok"])
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
