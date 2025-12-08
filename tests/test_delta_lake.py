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

def should_skip_delta_lake_test():
    """Skip on Linux x86_64 (glibc) and musl Linux due to S3 permission issues"""
    if platform.system() != "Linux":
        return False  # Don't skip on macOS, Windows, etc.
    if is_musl_linux():
        return True  # Skip musl Linux
    if platform.machine() == "x86_64":
        return True  # Skip Linux x86_64 (glibc)
    return False  # Run on Linux arm64 (glibc), etc.

@unittest.skipIf(should_skip_delta_lake_test(), "Skipping on Linux x86_64 due to S3 access permissions")
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
