#!python3

import time
import shutil
import psutil
import unittest
from chdb import session


test_state_dir = ".state_tmp_auxten_"
current_process = psutil.Process()
check_thread_count = False

class TestStateful(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(test_state_dir, ignore_errors=True)
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(test_state_dir, ignore_errors=True)
        return super().tearDown()

    def test_path(self):
        sess = session.Session(test_state_dir)
        sess.query("CREATE FUNCTION chdb_xxx AS () -> '0.12.0'", "CSV")
        ret = sess.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')

        sess.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic", "CSV")
        ret = sess.query("SHOW DATABASES", "CSV")
        self.assertIn("db_xxx", str(ret))

        sess.query(
            "CREATE TABLE IF NOT EXISTS db_xxx.log_table_xxx (x UInt8) ENGINE = Log;"
        )
        sess.query("INSERT INTO db_xxx.log_table_xxx VALUES (1), (2), (3), (4);")

        sess.query(
            "CREATE VIEW db_xxx.view_xxx AS SELECT * FROM db_xxx.log_table_xxx LIMIT 2;"
        )
        ret = sess.query("SELECT * FROM db_xxx.view_xxx", "CSV")
        self.assertEqual(str(ret), "1\n2\n")

        del sess # name sess dir will not be deleted

        sess = session.Session(test_state_dir)
        ret = sess.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')

        ret = sess.query("SHOW DATABASES", "CSV")
        self.assertIn("db_xxx", str(ret))

        ret = sess.query("SELECT * FROM db_xxx.log_table_xxx", "CSV")
        self.assertEqual(str(ret), "1\n2\n3\n4\n")

        # reuse session
        sess2 = session.Session(test_state_dir)

        ret = sess2.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')

        # remove session dir
        sess2.cleanup()
        ret = sess2.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), "")

    def test_tmp(self):
        sess = session.Session()
        sess.query("CREATE FUNCTION chdb_xxx AS () -> '0.12.0'", "CSV")
        ret = sess.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')
        del sess

        # another session
        sess2 = session.Session()
        ret = sess2.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), "")

    def test_zfree_thread_count(self):
        time.sleep(3)
        thread_count = current_process.num_threads()
        print("Number of threads using psutil library: ", thread_count)
        if check_thread_count:
            self.assertEqual(thread_count, 1)

if __name__ == "__main__":
    shutil.rmtree(test_state_dir, ignore_errors=True)
    check_thread_count = True
    unittest.main()
