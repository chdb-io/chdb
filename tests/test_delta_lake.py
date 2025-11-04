#!python3

import unittest
import sys
import platform
import subprocess
import chdb
from chdb import session

def is_musl_linux():
    """Check if running on musl Linux"""
    if platform.system() != "Linux":
        return False
    try:
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True)
        print(f"stdout: {result.stdout.lower()}")
        print(f"stderr: {result.stderr.lower()}")
        # Check both stdout and stderr for musl
        output_text = (result.stdout + result.stderr).lower()
        return 'musl' in output_text
    except Exception as e:
        print(f"Exception in is_musl_linux: {e}")
        return False

@unittest.skipUnless(sys.platform.startswith("linux") and not is_musl_linux(), "Runs only on Linux platforms")
class TestDeltaLake(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_delta_lake(self):
        sess = session.Session()

        ret = sess.query(
            '''
            SELECT
                URL,
                UserAgent
            FROM deltaLake('https://clickhouse-public-datasets.s3.amazonaws.com/delta_lake/hits/')
            WHERE URL IS NULL
            LIMIT 2
            ''')
        self.assertEqual(ret.rows_read(), 0)

if __name__ == "__main__":
    unittest.main()
