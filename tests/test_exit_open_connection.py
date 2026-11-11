"""A process that exits with a chDB connection still open must exit cleanly.

`import chdb` starts the engine and closes it, so the next connection restarts
it inside the same process. With chdb-core 26.9.0, leaving that second engine
open until exit made its teardown run on already-destroyed C++ statics: on
Linux it printed "Exception in global_context->shutdown(): The
ShutdownTablesThreadPool is not initialized", on macOS it intermittently
aborted with SIGABRT and nothing on stderr. Each case runs in a fresh
interpreter and checks both the exit status and that stderr stays empty.
"""

import os
import subprocess
import sys
import unittest

_DATASTORE_GLOBAL_EXECUTOR = """
import pandas as pd
from datastore import DataStore

df = pd.DataFrame({'x': [1, 2, 3]})
ds = DataStore(df)
assert list(ds[ds['x'] > 1]['x']) == [2, 3]
"""

_DATASTORE_GLOBAL_EXECUTOR_SETTINGS = """
from datastore.config import set_chdb_setting
from datastore.executor import get_executor

set_chdb_setting('memory_worker_correct_memory_tracker', 0)
executor = get_executor()
executor._ensure_connected()
r = executor.connection._conn.query(
    "SELECT value FROM system.server_settings WHERE name = 'memory_worker_correct_memory_tracker'",
    'CSV',
)
assert '"0"' in str(r), r
"""

_CHDB_CONNECTION = """
import chdb

conn = chdb.connect()
conn.query("CREATE TABLE t (x Int32) ENGINE = Memory")
conn.query("INSERT INTO t VALUES (1), (2)")
assert str(conn.query("SELECT sum(x) FROM t", "CSV")).strip() == "3"
"""


def _run(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestExitWithOpenConnection(unittest.TestCase):
    def _check(self, code, runs=3):
        for _ in range(runs):
            proc = _run(code)
            self.assertEqual(proc.returncode, 0, f"exit {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
            self.assertEqual(proc.stderr, "")

    def test_datastore_dataframe_query(self):
        self._check(_DATASTORE_GLOBAL_EXECUTOR)

    def test_datastore_overridden_server_settings(self):
        self._check(_DATASTORE_GLOBAL_EXECUTOR_SETTINGS)

    def test_chdb_connection(self):
        self._check(_CHDB_CONNECTION)


if __name__ == "__main__":
    unittest.main()
