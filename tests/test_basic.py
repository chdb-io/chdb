#!python3

import unittest
import chdb

class TestBasic(unittest.TestCase):
    def test_basic(self):
        res = chdb.query("SELECT 1", "CSV")
        assert len(res.get_memview().tobytes()) == 1

if __name__ == '__main__':
    unittest.main()