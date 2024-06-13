#!python3

import time
import shutil
import psutil
import unittest
from chdb import session
import chdb


test_state_dir = ".state_tmp_auxten_test_stateful"
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
        ret = sess.query("SELECT chdb_xxx()", "Debug")
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

        del sess  # name sess dir will not be deleted

        sess = session.Session(test_state_dir)
        ret = sess.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')

        ret = sess.query("SHOW DATABASES", "CSV")
        self.assertIn("db_xxx", str(ret))

        ret = sess.query("SELECT * FROM db_xxx.log_table_xxx", "CSV")
        self.assertEqual(str(ret), "1\n2\n3\n4\n")
        ret.show()

        # reuse session
        sess2 = session.Session(test_state_dir)

        ret = sess2.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')

        # remove session dir
        sess2.cleanup()
        with self.assertRaises(Exception):
            ret = sess2.query("SELECT chdb_xxx()", "CSV")

    def test_mergetree(self):
        sess = session.Session()
        sess.query("CREATE DATABASE IF NOT EXISTS db_xxx_merge ENGINE = Atomic;", "CSV")
        sess.query(
            "CREATE TABLE IF NOT EXISTS db_xxx_merge.log_table_xxx (x String, y Int) ENGINE = MergeTree ORDER BY x;"
        )
        # insert 1000000 random rows
        sess.query(
            "INSERT INTO db_xxx_merge.log_table_xxx (x, y) SELECT toString(rand()), rand() FROM numbers(1000000);"
        )
        sess.query("Optimize TABLE db_xxx_merge.log_table_xxx;")
        ret = sess.query("SELECT count(*) FROM db_xxx_merge.log_table_xxx;")
        self.assertEqual(str(ret), "1000000\n")

    def test_tmp(self):
        sess = session.Session()
        sess.query("CREATE FUNCTION chdb_xxx AS () -> '0.12.0'", "CSV")
        ret = sess.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')
        del sess

        # another session
        sess2 = session.Session()
        with self.assertRaises(Exception):
            ret = sess2.query("SELECT chdb_xxx()", "CSV")

    def test_two_sessions(self):
        sess1 = session.Session()
        sess2 = session.Session()
        sess1.query("CREATE FUNCTION chdb_xxx AS () -> 'sess1'", "CSV")
        sess2.query("CREATE FUNCTION chdb_xxx AS () -> 'sess2'", "CSV")
        sess1.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic", "CSV")
        sess2.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic", "CSV")
        sess1.query("CREATE TABLE IF NOT EXISTS db_xxx.tbl1 (x UInt8) ENGINE = Log;")
        sess2.query("CREATE TABLE IF NOT EXISTS db_xxx.tbl2 (x UInt8) ENGINE = Log;")
        sess1.query("INSERT INTO db_xxx.tbl1 VALUES (1), (2), (3), (4);")
        sess2.query("INSERT INTO db_xxx.tbl2 VALUES (5), (6), (7), (8);")
        ret = sess1.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"sess1"\n')
        ret = sess2.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"sess2"\n')
        ret = sess1.query("SELECT * FROM db_xxx.tbl1", "CSV")
        self.assertEqual(str(ret), "1\n2\n3\n4\n")
        ret = sess2.query("SELECT * FROM db_xxx.tbl2", "CSV")
        self.assertEqual(str(ret), "5\n6\n7\n8\n")
        sess1.query(
            """
            SET input_format_csv_use_best_effort_in_schema_inference = 0;
            SET input_format_csv_skip_first_lines = 1;"""
        )
        # query level settings should not affect session level settings
        ret = sess1.query(
            "SELECT 123 SETTINGS input_format_csv_use_best_effort_in_schema_inference = 1;"
        )
        # check sess1 settings
        ret = sess1.query("""SELECT value, changed FROM system.settings
            WHERE name = 'input_format_csv_use_best_effort_in_schema_inference';""")
        self.assertEqual(str(ret), '"0",1\n')
        ret = sess1.query("""SELECT value, changed FROM system.settings
            WHERE name = 'input_format_csv_skip_first_lines';""")
        self.assertEqual(str(ret), '"1",1\n')

        # sess2 should not be affected
        ret = sess2.query("""SELECT value, changed FROM system.settings
            WHERE name = 'input_format_csv_use_best_effort_in_schema_inference';""")
        self.assertEqual(str(ret), '"1",0\n')
        ret = sess2.query("""SELECT value, changed FROM system.settings
            WHERE name = 'input_format_csv_skip_first_lines';""")
        self.assertEqual(str(ret), '"0",0\n')

        # stateless query should not be affected
        ret = chdb.query(
            """SELECT value, changed FROM system.settings
            WHERE name = 'input_format_csv_use_best_effort_in_schema_inference';"""
        )
        self.assertEqual(str(ret), '"1",0\n')
        ret = chdb.query(
            """SELECT value, changed FROM system.settings
            WHERE name = 'input_format_csv_skip_first_lines';"""
        )
        self.assertEqual(str(ret), '"0",0\n')

    def test_context_mgr(self):
        with session.Session() as sess:
            sess.query("CREATE FUNCTION chdb_xxx_mgr AS () -> '0.12.0_mgr'", "Debug")
            ret = sess.query("SELECT chdb_xxx_mgr()", "CSV")
            self.assertEqual(str(ret), '"0.12.0_mgr"\n')

        with session.Session() as sess:
            with self.assertRaises(Exception):
                ret = sess.query("SELECT chdb_xxx_notexist()", "CSV")

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
