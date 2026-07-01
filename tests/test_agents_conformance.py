"""Runs the language-neutral agent-tool conformance fixture against the Python
reference implementation (chdb.agents.ChDBTool).

This is the *reference* runner for chdb/agents/conformance/cases.jsonl. The
TypeScript binding ships an equivalent ~30-line runner over the same file, so the
two languages verify identical behavior. Keep this runner thin: all the
knowledge lives in the fixture and in ChDBTool, not here.
"""

import json
import os
import unittest

import chdb.agents
from chdb.agents import ChDBTool, ChDBError

# Locate the fixture next to the installed chdb.agents package, so this runner
# works whether run from the repo or against an installed wheel.
_AGENTS = os.path.join(os.path.dirname(os.path.abspath(chdb.agents.__file__)), "conformance")
_CASES = os.path.join(_AGENTS, "cases.jsonl")
_FIXTURES = os.path.abspath(os.path.join(_AGENTS, "fixtures"))


def _load_cases():
    cases = []
    with open(_CASES, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _sub(args):
    """Replace the {{fixtures}} token in any string arg with the abs path."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str):
            out[k] = v.replace("{{fixtures}}", _FIXTURES)
        elif isinstance(v, dict):
            out[k] = _sub(v)
        else:
            out[k] = v
    return out


class TestAgentsConformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One read-only tool for the whole run — every case is setup-free and
        # read-only-compatible (writes are expected to be blocked).
        cls.tool = ChDBTool(read_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.tool.close()

    def _invoke(self, case):
        method = case["method"]
        args = _sub(case.get("args", {}))
        # A case may declare its own tool config (for constructor-level features
        # like max_execution_time / file_allowlist / attachments); otherwise reuse
        # the shared read-only tool.
        if "tool" in case:
            tool = ChDBTool(**_sub(case["tool"]))
            self.addCleanup(tool.close)
        else:
            tool = self.tool
        if method == "call":
            return tool.call(args["name"], args.get("arguments"))
        if method == "query":
            return tool.query(args["sql"], params=args.get("params"), max_rows=args.get("max_rows"))
        if method == "list_databases":
            return tool.list_databases()
        if method == "list_tables":
            return tool.list_tables(args.get("database"))
        if method == "describe":
            return tool.describe(args["target"])
        if method == "get_sample_data":
            return tool.get_sample_data(args["target"], limit=args.get("limit", 5))
        if method == "list_functions":
            return tool.list_functions(like=args.get("like"), limit=args.get("limit", 200))
        self.fail("unknown method in case: " + method)

    def _assert(self, case, exp):
        # error_type on a raising method
        if "error_type" in exp and "envelope_ok" not in exp:
            with self.assertRaises(ChDBError) as ctx:
                self._invoke(case)
            self.assertEqual(ctx.exception.type, exp["error_type"])
            return
        result = self._invoke(case)
        if "envelope_ok" in exp:
            self.assertEqual(result["ok"], exp["envelope_ok"])
            if exp.get("error_type"):
                self.assertEqual(result["error"]["type"], exp["error_type"])
            return
        if "rows" in exp:
            self.assertEqual(result.rows, exp["rows"])
        if "truncated" in exp:
            self.assertEqual(result.truncated, exp["truncated"])
        if "row_count" in exp:
            rc = result.row_count if hasattr(result, "row_count") else len(result)
            self.assertEqual(rc, exp["row_count"])
        if "contains_all" in exp:
            for v in exp["contains_all"]:
                self.assertIn(v, result)
        if "min_len" in exp:
            self.assertGreaterEqual(len(result), exp["min_len"])
        if "describe_column" in exp:
            self.assertIn(exp["describe_column"], [c["name"] for c in result])

    def test_conformance_cases(self):
        cases = _load_cases()
        self.assertGreater(len(cases), 0, "no conformance cases loaded")
        for case in cases:
            with self.subTest(case=case["id"]):
                self._assert(case, case["expect"])


if __name__ == "__main__":
    unittest.main()
