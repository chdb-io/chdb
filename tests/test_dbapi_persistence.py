#!/usr/bin/env python3

import shutil
import unittest
from chdb import dbapi

test_state_dir = ".state_tmp_auxten_dbapi"


class TestDBAPIPersistence(unittest.TestCase):

    def setUp(self) -> None:
        shutil.rmtree(test_state_dir, ignore_errors=True)
        return super().setUp()

    def tearDown(self):
        shutil.rmtree(test_state_dir, ignore_errors=True)
        return super().tearDown()

    def test_persistence(self):
        conn = dbapi.connect(path=test_state_dir)
        cur = conn.cursor()
        cur.execute("CREATE DATABASE e ENGINE = Atomic;")
        cur.execute(
            "CREATE TABLE e.hi (a String primary key, b Int32) Engine = MergeTree ORDER BY a;"
        )
        cur.execute("INSERT INTO e.hi (a, b) VALUES (%s, %s);", ["he", 32])

        cur.close()
        conn.close()

        conn2 = dbapi.connect(path=test_state_dir)
        cur2 = conn2.cursor()
        cur2.execute("SELECT * FROM e.hi;")
        row = cur2.fetchone()
        self.assertEqual(("he", 32), row)

    def test_placeholder1(self):
        conn = dbapi.connect(path=test_state_dir)
        cur = conn.cursor()

        cur.execute("CREATE DATABASE test ENGINE = Atomic;")
        cur.execute(
            "CREATE TABLE test.users (id UInt64, name String, age UInt32) "
            "ENGINE = MergeTree ORDER BY id;"
        )

        cur.execute("INSERT INTO test.users (id, name, age) VALUES (?, ?, ?)",
                   (1, 'Alice', 25))

        cur.execute("SELECT name, age FROM test.users WHERE id = ? AND age > ?",
                   (1, 20))
        row = cur.fetchone()
        self.assertEqual(("Alice", 25), row)

        data = [(2, 'Bob', 30), (3, 'Charlie', 35), (4, 'David', 28)]
        cur.executemany("INSERT INTO test.users (id, name, age) VALUES (?, ?, ?)",
                       data)

        cur.execute("SELECT COUNT(*) FROM test.users WHERE id > 1")
        count = cur.fetchone()[0]
        self.assertEqual(3, count)
        cur.execute("SELECT name FROM test.users WHERE age = ? ORDER BY id", (30,))
        result = cur.fetchone()
        self.assertEqual(("Bob",), result)
        cur.close()
        conn.close()

    def test_placeholder2(self):
        conn = dbapi.connect(path=test_state_dir)
        cur = conn.cursor()

        # Create table
        cur.execute("CREATE DATABASE compat ENGINE = Atomic;")
        cur.execute(
            "CREATE TABLE compat.test (id UInt64, value String) "
            "ENGINE = MergeTree ORDER BY id;"
        )

        # Test %s placeholders still work
        cur.execute("INSERT INTO compat.test (id, value) VALUES (%s, %s)",
                   (1, 'test_value'))

        cur.execute("SELECT value FROM compat.test")
        result = cur.fetchone()
        self.assertEqual(("test_value",), result)

        cur.close()
        conn.close()


if __name__ == "__main__":
    unittest.main()
