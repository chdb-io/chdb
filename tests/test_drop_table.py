#!python3

import shutil
import unittest
from chdb import session

test_dir1 = "test_drop_table"

class TestDropTable(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(test_dir1, ignore_errors=True)
        return super().tearDown()

    def test_drop_table(self):
        shutil.rmtree(test_dir1, ignore_errors=True)
        sess = session.Session(test_dir1)

        sess.query('''
            CREATE TABLE test_table_1
            (
                value String
            ) ENGINE = MergeTree()
            ORDER BY value
        ''')
        sess.query("INSERT INTO test_table_1 VALUES ('test')")
        sess.query("DROP TABLE test_table_1")

        sess.close()

        sess = session.Session(test_dir1)

        sess.query('''
            CREATE TABLE test_table_2
            (
                value String
            ) ENGINE = MergeTree()
            ORDER BY value
        ''')
        sess.query("INSERT INTO test_table_2 VALUES ('test')")
        sess.query("DROP TABLE test_table_2 SYNC")

        sess.close()


if __name__ == '__main__':
    unittest.main()
