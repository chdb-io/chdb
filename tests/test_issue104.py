#!python3

import os
import time
import shutil
import unittest
import psutil
from chdb import session as chs

tmp_dir = ".state_tmp_auxten_issue104"


class TestIssue104(unittest.TestCase):
    def setUp(self) -> None:
        # shutil.rmtree(tmp_dir, ignore_errors=True)
        return super().setUp()

    def tearDown(self):
        # shutil.rmtree(tmp_dir, ignore_errors=True)
        return super().tearDown()

    def test_issue104(self):
        sess = chs.Session(tmp_dir)

        sess.query("CREATE DATABASE IF NOT EXISTS test_db ENGINE = Atomic;")
        # sess.query("CREATE DATABASE IF NOT EXISTS test_db ENGINE = Atomic;", "Debug")
        sess.query("CREATE TABLE IF NOT EXISTS test_db.test_table (x String, y String) ENGINE = MergeTree ORDER BY tuple()")
        sess.query("INSERT INTO test_db.test_table (x, y) VALUES ('A', 'B'), ('C', 'D');")

        # show final thread count
        print("Final thread count:", len(psutil.Process().threads()))

        print("Original values:")
        ret = sess.query("SELECT * FROM test_db.test_table", "Debug")
        print(ret)
        # self.assertEqual(str(ret), '"A","B"\n"C","D"\n')

        # show final thread count
        print("Final thread count:", len(psutil.Process().threads()))

        print('Values after ALTER UPDATE in same query expected:')
        ret = sess.query(
            "ALTER TABLE test_db.test_table UPDATE y = 'updated1' WHERE x = 'A';"
            "SELECT * FROM test_db.test_table WHERE x = 'A';")
        print(ret)
        self.assertEqual(str(ret), '"A","updated1"\n')

        # show final thread count
        print("Final thread count:", len(psutil.Process().threads()))

        # print("Values after UPDATE in same query (expected 'A', 'updated'):")
        # ret = sess.query(
        #     "UPDATE test_db.test_table SET y = 'updated2' WHERE x = 'A';"
        #     "SELECT * FROM test_db.test_table WHERE x = 'A';")
        # print(ret)
        # self.assertEqual(str(ret), '"A","updated2"\n')

        print('Values after UPDATE expected:')
        sess.query("ALTER TABLE test_db.test_table UPDATE y = 'updated2' WHERE x = 'A';"
                   "ALTER TABLE test_db.test_table UPDATE y = 'updated3' WHERE x = 'A'")
        ret = sess.query("SELECT * FROM test_db.test_table WHERE x = 'A'")
        print(ret)
        self.assertEqual(str(ret), '"A","updated3"\n')

        # show final thread count
        print("Final thread count:", len(psutil.Process().threads()))

        print("Values after DELETE expected:")
        sess.query("ALTER TABLE test_db.test_table DELETE WHERE x = 'A'")
        ret = sess.query("SELECT * FROM test_db.test_table")
        print(ret)
        self.assertEqual(str(ret), '"C","D"\n')

        # show final thread count
        print("Final thread count:", len(psutil.Process().threads()))

        print("Values after ALTER then OPTIMIZE expected:")
        sess.query("ALTER TABLE test_db.test_table DELETE WHERE x = 'C'; OPTIMIZE TABLE test_db.test_table FINAL")
        ret = sess.query("SELECT * FROM test_db.test_table")
        print(ret)
        self.assertEqual(str(ret), "")

        print("Inserting 1000 rows")
        sess.query("INSERT INTO test_db.test_table (x, y) SELECT toString(number), toString(number) FROM numbers(1000);")
        ret = sess.query("SELECT count() FROM test_db.test_table", "Debug")
        count = str(ret).count("\n")
        print("Number of newline characters:", count)

        # show final thread count
        print("Final thread count:", len(psutil.Process().threads()))

        time.sleep(3)
        print("Final thread count after 3s:", len(psutil.Process().threads()))
        self.assertEqual(len(psutil.Process().threads()), 1)


if __name__ == "__main__":
    unittest.main()
