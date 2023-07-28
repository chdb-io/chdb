#!python3

import time
import shutil
import unittest
import chdb
from chdb import session


test_state_dir = ".usedb_tmp_auxten_"
check_thread_count = False

class TestStateful(unittest.TestCase):    
    def test_no_path(self):
        chdb.query("CREATE DATABASE IF NOT EXISTS tmp_db_xxx ENGINE = Atomic", "Debug")

    def test_path(self):
        sess = session.Session(test_state_dir)
        sess.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic", "Debug")
        print(sess.query("USE db_xxx; SHOW tables", "CSV"))

if __name__ == "__main__":
    shutil.rmtree(test_state_dir, ignore_errors=True)
    unittest.main()
