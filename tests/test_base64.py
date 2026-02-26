#!python3

import unittest
import chdb


class TestBase64Functions(unittest.TestCase):
    """Test ClickHouse base64Encode and base64Decode functions."""

    def test_base64_encode(self):
        res = chdb.query("SELECT base64Encode('clickhouse')", "CSV")
        self.assertEqual(res.bytes().strip(), b'"Y2xpY2tob3VzZQ=="')

    def test_base64_decode(self):
        res = chdb.query("SELECT base64Decode('Y2xpY2tob3VzZQ==')", "CSV")
        self.assertEqual(res.bytes().strip(), b'"clickhouse"')

    def test_base64_roundtrip(self):
        res = chdb.query("SELECT base64Decode(base64Encode('hello world'))", "CSV")
        self.assertEqual(res.bytes().strip(), b'"hello world"')


if __name__ == "__main__":
    unittest.main()
