#!python3

import os
import unittest
import pyarrow # type: ignore
import chdb
from format_output import format_output
from utils import data_file, reset_elapsed


class TestBasic(unittest.TestCase):
    def test_basic(self):
        res = chdb.query("SELECT 1", "CSV")
        self.assertEqual(len(res), 2) # "1\n"
        self.assertFalse(res.has_error())
        self.assertTrue(len(res.error_message()) == 0)
        with self.assertRaises(Exception):
            res = chdb.query("SELECT 1", "unknown_format")


class TestOutput(unittest.TestCase):
    def test_output(self):
        for format, output in format_output.items():
            if format != "ArrowTable":
                continue
            res = chdb.query("SELECT * FROM file('" + data_file + "', Parquet) limit 10", format)
            data = reset_elapsed(f"{res}")
            self.assertEqual(data, output["data"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
