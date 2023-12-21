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
        cur.execute("""
        CREATE TABLE rate (
            day Date,
            value Int32
        ) ENGINE = Memory""")

        # Insert values
        cur.execute("INSERT INTO rate VALUES ('2024-01-01', 24)")
        cur.execute("INSERT INTO rate VALUES ('2024-01-02', 72)")

        # Read values
        cur.execute("SELECT value FROM rate ORDER BY DAY DESC")
        rows = cur.fetchall()
        self.assertEqual(rows, [(72,), (24,)])

    def test_select_chdb_version(self):
        ver = dbapi.get_client_info()  # chDB version liek '0.12.0'
        ver_tuple = dbapi.chdb_version  # chDB version tuple like ('0', '12', '0')
        print(ver)
        print(ver_tuple)
        self.assertEqual(ver, '.'.join(ver_tuple))
        self.assertRegex(ver, expected_version_pattern)


if __name__ == '__main__':
    unittest.main()
