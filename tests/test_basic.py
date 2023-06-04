#!python3

import os
import unittest
import chdb
from format_output import format_output
from utils import data_file, reset_elapsed

class TestBasic(unittest.TestCase):
    def test_basic(self):
        res = chdb.query("SELECT 1", "CSV")
        self.assertEqual(len(res.get_memview().tobytes()), 2) # "1\n"
class TestOutput(unittest.TestCase):
    def test_output(self):
        for format, output in format_output.items():
            res = chdb.query("SELECT * FROM file('" + data_file + "', Parquet) limit 10", format)
            if format == "ArrowTable":
                data = reset_elapsed(f"{res}")
            else:
                data = reset_elapsed(res.get_memview().tobytes())
            # Arrow format output is not deterministic
            if format in ("Arrow", "ArrowStream"):
                continue
            self.assertEqual(data, output["data"])


if __name__ == '__main__':
    unittest.main()
