"""pandas/pyarrow are runtime-optional for the chdb wrapper.

`import chdb` and every non-DataFrame/Arrow query path must work when
pandas, pyarrow and numpy are not installed (e.g. `pip install --no-deps`),
while the DataFrame/DataStore/ArrowTable paths must fail with informative
errors. Each case runs in a subprocess with an import hook simulating the
missing packages.
"""

import subprocess
import sys
import unittest

_BLOCKER_PRELUDE = """
import sys

class _Blocker:
    def __init__(self, blocked):
        self.blocked = blocked

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.blocked:
            raise ImportError(f"No module named {{name!r}} (blocked by test)")

sys.meta_path.insert(0, _Blocker({blocked!r}))
sys.path = {parent_sys_path!r} + sys.path
"""

_BODY_QUERY_PATHS_WORK = """
import chdb

# core_version is defined by the wrapper's __init__, so this proves the
# subprocess imported the chdb wrapper (not a bare chdb-core install).
assert hasattr(chdb, "core_version"), chdb.__file__

res = chdb.query("SELECT 1 as a, 'x' as b", "CSV")
assert res.bytes() == b'1,"x"\\n', res.bytes()

res = chdb.query("SELECT number FROM numbers(3)", "JSON")
assert '"number"' in str(res), str(res)

conn = chdb.connect(":memory:")
assert "42" in str(conn.query("SELECT 42", "CSV"))

cur = conn.cursor()
cur.execute("SELECT 1 as v, 'hello' as s")
assert cur.fetchall() == ((1, "hello"),), cur.fetchall()
conn.close()

from chdb import session
s = session.Session()
assert "7" in str(s.query("SELECT 7", "CSV"))
s.close()

from chdb.agents import ChDBTool  # noqa: F401

print("QUERY_PATHS_OK")
"""

_BODY_DF_PATHS_RAISE = """
import chdb

def assert_needs_deps(fn, label):
    try:
        fn()
    except Exception as e:
        msg = str(e).lower()
        assert any(m in msg for m in ("pandas", "numpy", "pyarrow")), (label, e)
    else:
        raise AssertionError(label + " should require pandas/pyarrow")

assert_needs_deps(lambda: chdb.query("SELECT 1", "DataFrame"), "DataFrame output")
assert_needs_deps(lambda: chdb.query("SELECT 1", "DataStore"), "DataStore output")
assert_needs_deps(lambda: chdb.query("SELECT 1", "ArrowTable"), "ArrowTable output")

def import_datastore():
    from chdb import datastore  # noqa: F401

assert_needs_deps(import_datastore, "chdb.datastore import")

print("DF_PATHS_RAISE_OK")
"""


class TestNoPandasPyarrow(unittest.TestCase):
    def _run_blocked(self, body):
        script = (
            _BLOCKER_PRELUDE.format(
                blocked=("pandas", "pyarrow", "numpy"),
                parent_sys_path=[p for p in sys.path if p],
            )
            + body
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        return proc.stdout

    def test_import_and_text_query_paths_work_without_pandas_pyarrow(self):
        out = self._run_blocked(_BODY_QUERY_PATHS_WORK)
        self.assertIn("QUERY_PATHS_OK", out)

    def test_dataframe_paths_raise_informative_errors_without_pandas_pyarrow(self):
        out = self._run_blocked(_BODY_DF_PATHS_RAISE)
        self.assertIn("DF_PATHS_RAISE_OK", out)


if __name__ == "__main__":
    unittest.main()
