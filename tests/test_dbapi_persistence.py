#!/usr/bin/env python3

import unittest
from chdb import dbapi

test_state_dir = ".state_tmp_auxten_dbapi"

class TestDBAPIPersistence(unittest.TestCase):
    def test_persistence(self):
        conn = dbapi.connect(path=test_state_dir)
        cur = con.cursor()
        cur.execute("CREATE DATABASE e ENGINE = Atomic;")
        cur.execute("CREATE TABLE e.hi (a String primary key, b Int32) Engine = MergeTree ORDER BY a;")
        cur.execute("INSERT INTO e.hi (a, b) VALUES (%s, %s);", ["he", 32])

        cur.close()
        conn.close()

        conn2 = dbapi.connect(path=test_state_dir)
        cur2 = conn2.cursor()
        cur2.execute('SELECT * FROM e.hi;')
        row = cur2.fetchone()
        self.assertEqual(('he', 32), row)


if __name__ == '__main__':
    unittest.main()