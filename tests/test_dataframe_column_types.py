#!/usr/bin/env python3

import unittest
import pandas as pd
import chdb
from datetime import datetime, date
import numpy as np


class TestDataFrameColumnTypes(unittest.TestCase):

    def setUp(self):
        self.session = chdb.session.Session()

    def tearDown(self):
        self.session.close()

    def test_integer_types(self):
        ret = self.session.query("""
            SELECT
                toInt8(-128) as int8_val,
                toInt16(-32768) as int16_val,
                toInt32(-2147483648) as int32_val,
                toInt64(-9223372036854775808) as int64_val,
                toUInt8(255) as uint8_val,
                toUInt16(65535) as uint16_val,
                toUInt32(4294967295) as uint32_val,
                toUInt64(18446744073709551615) as uint64_val
        """, "DataFrame")

        self.assertEqual(ret.iloc[0]["int16_val"], -32768)
        self.assertEqual(ret.iloc[0]["int32_val"], -2147483648)
        self.assertEqual(ret.iloc[0]["int64_val"], -9223372036854775808)
        self.assertEqual(ret.iloc[0]["uint8_val"], 255)
        self.assertEqual(ret.iloc[0]["uint16_val"], 65535)
        self.assertEqual(ret.iloc[0]["uint32_val"], 4294967295)
        self.assertEqual(ret.iloc[0]["uint64_val"], 18446744073709551615)

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Precise data type validation
        expected_types = {
            "int8_val": "int8",
            "int16_val": "int16",
            "int32_val": "int32",
            "int64_val": "int64",
            "uint8_val": "uint8",
            "uint16_val": "uint16",
            "uint32_val": "uint32",
            "uint64_val": "uint64"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)


if __name__ == "__main__":
    unittest.main()
