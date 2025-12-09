#!python3

import unittest
import sys
import chdb
from chdb import session
from utils import is_musl_linux


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
