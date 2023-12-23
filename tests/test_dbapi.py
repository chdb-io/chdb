#!/usr/bin/env python3

import unittest
from chdb import dbapi

# version should be string split by '.'
# eg. '0.12.0' or '0.12.0rc1' or '0.12.0beta1' or '0.12.0alpha1' or '0.12.0a1'
expected_version_pattern = r'^\d+\.\d+\.\d+(.*)?$'
expected_clickhouse_version_pattern = r'^\d+\.\d+\.\d+.\d+$'


class TestDBAPI(unittest.TestCase):
    def test_select_version(self):
        conn = dbapi.connect()
        cur = conn.cursor()
        cur.execute('select version()')  # ClickHouse version
        description = cur.description
        data = cur.fetchone()
        cur.close()
        conn.close()

        # Add your assertions here to validate the description and data
        print(description)
        print(data)
        self.assertRegex(data[0], expected_clickhouse_version_pattern)

    def test_insert_and_read_data(self):
        conn = dbapi.connect()
        cur = conn.cursor()
        cur.execute("CREATE DATABASE IF NOT EXISTS test_db ENGINE = Atomic")
        cur.execute("USE test_db")
        cur.execute("""
        CREATE TABLE rate (
            day Date,
            value Int32
        ) ENGINE = Log""")

        # Insert single value
        cur.execute("INSERT INTO rate VALUES (%s, %s)", ("2021-01-01", 24))
        # Insert multiple values
        cur.executemany("INSERT INTO rate VALUES (%s, %s)", [("2021-01-02", 72), ("2021-01-03", 96)])

        # Read values
        cur.execute("SELECT value FROM rate ORDER BY day DESC")
        rows = cur.fetchall()
        assert rows==((96,), (72,), (24,))

    def test_select_chdb_version(self):
        ver = dbapi.get_client_info()  # chDB version liek '0.12.0'
        ver_tuple = dbapi.chdb_version  # chDB version tuple like ('0', '12', '0')
        print(ver)
        print(ver_tuple)
        self.assertEqual(ver, '.'.join(ver_tuple))
        self.assertRegex(ver, expected_version_pattern)


if __name__ == '__main__':
    unittest.main()
