#!/usr/bin/env python3

import unittest
from chdb import dbapi

# version should be string split by '.'
# eg. '0.12.0' or '0.12.0rc1' or '0.12.0beta1' or '0.12.0alpha1' or '0.12.0a1'
expected_version_pattern = r'^\d+\.\d+\.\d+(.*)?$'


class TestDBAPI(unittest.TestCase):
    def test_get_client_info(self):
        driver_version = dbapi.get_client_info()
        self.assertRegex(driver_version, expected_version_pattern)

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
        self.assertRegex(data[0], expected_version_pattern)

    def test_select_chdb_version(self):
        ver = dbapi.get_client_info()  # chDB version liek '0.12.0'
        ver_tuple = dbapi.chdb_version  # chDB version tuple like ('0', '12', '0')
        self.assertEqual(ver, '.'.join(ver_tuple))
        self.assertRegex(ver, expected_version_pattern)


if __name__ == '__main__':
    unittest.main()
