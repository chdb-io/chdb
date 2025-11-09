#!/usr/bin/env python3

import unittest
import pandas as pd
import chdb
from datetime import datetime, date
import numpy as np
import math
import uuid
import ipaddress


class TestDataFrameColumnTypes(unittest.TestCase):

    def setUp(self):
        self.session = chdb.session.Session("./tmp")

    def tearDown(self):
        self.session.close()

    @unittest.skip("")
    def test_integer_types(self):
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toInt8(-128) as int8_val,
                    toInt16(-32768) as int16_val,
                    toInt32(-2147483648) as int32_val,
                    toInt64(-9223372036854775808) as int64_val,
                    toInt128('-170141183460469231731687303715884105728') as int128_val,
                    toInt256('-57896044618658097711785492504343953926634992332820282019728792003956564819968') as int256_val,
                    toUInt8(255) as uint8_val,
                    toUInt16(65535) as uint16_val,
                    toUInt32(4294967295) as uint32_val,
                    toUInt64(18446744073709551615) as uint64_val,
                    toUInt128('340282366920938463463374607431768211455') as uint128_val,
                    toUInt256('115792089237316195423570985008687907853269984665640564039457584007913129639935') as uint256_val
                UNION ALL
                SELECT
                    2 as row_id,
                    toInt8(127) as int8_val,
                    toInt16(32767) as int16_val,
                    toInt32(2147483647) as int32_val,
                    toInt64(9223372036854775807) as int64_val,
                    toInt128('170141183460469231731687303715884105727') as int128_val,
                    toInt256('57896044618658097711785492504343953926634992332820282019728792003956564819967') as int256_val,
                    toUInt8(254) as uint8_val,
                    toUInt16(65534) as uint16_val,
                    toUInt32(4294967294) as uint32_val,
                    toUInt64(18446744073709551614) as uint64_val,
                    toUInt128('340282366920938463463374607431768211454') as uint128_val,
                    toUInt256('115792089237316195423570985008687907853269984665640564039457584007913129639934') as uint256_val
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row (minimum/maximum values)
        self.assertEqual(ret.iloc[0]["int8_val"], -128)
        self.assertEqual(ret.iloc[0]["int16_val"], -32768)
        self.assertEqual(ret.iloc[0]["int32_val"], -2147483648)
        self.assertEqual(ret.iloc[0]["int64_val"], -9223372036854775808)
        self.assertEqual(ret.iloc[0]["int128_val"], float(-170141183460469231731687303715884105728))
        self.assertEqual(ret.iloc[0]["int256_val"], float(-57896044618658097711785492504343953926634992332820282019728792003956564819968))
        self.assertEqual(ret.iloc[0]["uint8_val"], 255)
        self.assertEqual(ret.iloc[0]["uint16_val"], 65535)
        self.assertEqual(ret.iloc[0]["uint32_val"], 4294967295)
        self.assertEqual(ret.iloc[0]["uint64_val"], 18446744073709551615)
        self.assertEqual(ret.iloc[0]["uint128_val"], float(340282366920938463463374607431768211455))
        self.assertEqual(ret.iloc[0]["uint256_val"], float(115792089237316195423570985008687907853269984665640564039457584007913129639935))

        # Test second row (maximum/near-maximum values)
        self.assertEqual(ret.iloc[1]["int8_val"], 127)
        self.assertEqual(ret.iloc[1]["int16_val"], 32767)
        self.assertEqual(ret.iloc[1]["int32_val"], 2147483647)
        self.assertEqual(ret.iloc[1]["int64_val"], 9223372036854775807)
        self.assertEqual(ret.iloc[1]["int128_val"], float(170141183460469231731687303715884105727))
        self.assertEqual(ret.iloc[1]["int256_val"], float(57896044618658097711785492504343953926634992332820282019728792003956564819967))
        self.assertEqual(ret.iloc[1]["uint8_val"], 254)
        self.assertEqual(ret.iloc[1]["uint16_val"], 65534)
        self.assertEqual(ret.iloc[1]["uint32_val"], 4294967294)
        self.assertEqual(ret.iloc[1]["uint64_val"], 18446744073709551614)
        self.assertEqual(ret.iloc[1]["uint128_val"], float(340282366920938463463374607431768211454))
        self.assertEqual(ret.iloc[1]["uint256_val"], float(115792089237316195423570985008687907853269984665640564039457584007913129639934))

        # Precise data type validation
        expected_types = {
            "int8_val": "int8",
            "int16_val": "int16",
            "int32_val": "int32",
            "int64_val": "int64",
            "int128_val": "float64",  # Int128 mapped to float64 in ClickHouse->pandas conversion
            "int256_val": "float64",  # Int256 mapped to float64 in ClickHouse->pandas conversion
            "uint8_val": "uint8",
            "uint16_val": "uint16",
            "uint32_val": "uint32",
            "uint64_val": "uint64",
            "uint128_val": "float64",  # UInt128 mapped to float64 in ClickHouse->pandas conversion
            "uint256_val": "float64"   # UInt256 mapped to float64 in ClickHouse->pandas conversion
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_float_types(self):
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toFloat32(3.14159265) as float32_val,
                    toFloat32(-3.40282347e+38) as float32_min,
                    toFloat32(3.40282347e+38) as float32_max,
                    toFloat64(2.718281828459045) as float64_val,
                    toFloat64(-1.7976931348623157e+308) as float64_min,
                    toFloat64(1.7976931348623157e+308) as float64_max,
                    toBFloat16(1.5) as bfloat16_val,
                    toBFloat16(-3.389531389e+38) as bfloat16_min,
                    toBFloat16(3.389531389e+38) as bfloat16_max
                UNION ALL
                SELECT
                    2 as row_id,
                    toFloat32(0.0) as float32_val,
                    toFloat32(1.175494351e-38) as float32_min,
                    toFloat32(-1.175494351e-38) as float32_max,
                    toFloat64(0.0) as float64_val,
                    toFloat64(2.2250738585072014e-308) as float64_min,
                    toFloat64(-2.2250738585072014e-308) as float64_max,
                    toBFloat16(0.0) as bfloat16_val,
                    toBFloat16(1.175494351e-38) as bfloat16_min,
                    toBFloat16(-1.175494351e-38) as bfloat16_max
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[1][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - regular and extreme values
        self.assertAlmostEqual(ret.iloc[0]["float32_val"], 3.14159265, places=6)
        self.assertAlmostEqual(ret.iloc[0]["float32_min"], -3.40282347e+38, delta=1e30)
        self.assertAlmostEqual(ret.iloc[0]["float32_max"], 3.40282347e+38, delta=1e30)
        self.assertAlmostEqual(ret.iloc[0]["float64_val"], 2.718281828459045, places=15)
        self.assertAlmostEqual(ret.iloc[0]["float64_min"], -1.7976931348623157e+308, delta=1e300)
        self.assertAlmostEqual(ret.iloc[0]["float64_max"], 1.7976931348623157e+308, delta=1e300)
        self.assertAlmostEqual(ret.iloc[0]["bfloat16_val"], 1.5, places=2)
        self.assertAlmostEqual(ret.iloc[0]["bfloat16_min"], -3.389531389e+38, delta=1e30)
        self.assertAlmostEqual(ret.iloc[0]["bfloat16_max"], 3.389531389e+38, delta=1e30)

        # Test second row - zero and small values
        self.assertEqual(ret.iloc[1]["float32_val"], 0.0)
        self.assertAlmostEqual(ret.iloc[1]["float32_min"], 1.175494351e-38, delta=1e-40)
        self.assertAlmostEqual(ret.iloc[1]["float32_max"], -1.175494351e-38, delta=1e-40)
        self.assertEqual(ret.iloc[1]["float64_val"], 0.0)
        self.assertAlmostEqual(ret.iloc[1]["float64_min"], 2.2250738585072014e-308, delta=1e-310)
        self.assertAlmostEqual(ret.iloc[1]["float64_max"], -2.2250738585072014e-308, delta=1e-310)
        self.assertEqual(ret.iloc[1]["bfloat16_val"], 0.0)
        self.assertAlmostEqual(ret.iloc[1]["bfloat16_min"], 1.175494351e-38, delta=1e-40)
        self.assertAlmostEqual(ret.iloc[1]["bfloat16_max"], -1.175494351e-38, delta=1e-40)

        # Precise data type validation
        expected_types = {
            "float32_val": "float32",
            "float32_min": "float32",
            "float32_max": "float32",
            "float64_val": "float64",
            "float64_min": "float64",
            "float64_max": "float64",
            "bfloat16_val": "float32",  # BFloat16 typically mapped to float32 in pandas
            "bfloat16_min": "float32",
            "bfloat16_max": "float32"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_float_special_values(self):
        """Test Infinity and NaN values for all float types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toFloat32(1.0/0.0) as float32_pos_inf,
                    toFloat32(-1.0/0.0) as float32_neg_inf,
                    toFloat32(0.0/0.0) as float32_nan,
                    toFloat64(1.0/0.0) as float64_pos_inf,
                    toFloat64(-1.0/0.0) as float64_neg_inf,
                    toFloat64(0.0/0.0) as float64_nan,
                    toBFloat16(1.0/0.0) as bfloat16_pos_inf,
                    toBFloat16(-1.0/0.0) as bfloat16_neg_inf,
                    toBFloat16(0.0/0.0) as bfloat16_nan
                UNION ALL
                SELECT
                    2 as row_id,
                    toFloat32(1.0/0.0) as float32_pos_inf,
                    toFloat32(-1.0/0.0) as float32_neg_inf,
                    toFloat32(0.0/0.0) as float32_nan,
                    toFloat64(1.0/0.0) as float64_pos_inf,
                    toFloat64(-1.0/0.0) as float64_neg_inf,
                    toFloat64(0.0/0.0) as float64_nan,
                    toBFloat16(1.0/0.0) as bfloat16_pos_inf,
                    toBFloat16(-1.0/0.0) as bfloat16_neg_inf,
                    toBFloat16(0.0/0.0) as bfloat16_nan
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test Float32 special values
        self.assertTrue(math.isinf(ret.iloc[0]["float32_pos_inf"]))
        self.assertTrue(ret.iloc[0]["float32_pos_inf"] > 0)  # positive infinity
        self.assertTrue(math.isinf(ret.iloc[0]["float32_neg_inf"]))
        self.assertTrue(ret.iloc[0]["float32_neg_inf"] < 0)  # negative infinity
        self.assertTrue(math.isnan(ret.iloc[0]["float32_nan"]))

        # Test Float64 special values
        self.assertTrue(math.isinf(ret.iloc[0]["float64_pos_inf"]))
        self.assertTrue(ret.iloc[0]["float64_pos_inf"] > 0)  # positive infinity
        self.assertTrue(math.isinf(ret.iloc[0]["float64_neg_inf"]))
        self.assertTrue(ret.iloc[0]["float64_neg_inf"] < 0)  # negative infinity
        self.assertTrue(math.isnan(ret.iloc[0]["float64_nan"]))

        # Test BFloat16 special values
        self.assertTrue(math.isinf(ret.iloc[0]["bfloat16_pos_inf"]))
        self.assertTrue(ret.iloc[0]["bfloat16_pos_inf"] > 0)  # positive infinity
        self.assertTrue(math.isinf(ret.iloc[0]["bfloat16_neg_inf"]))
        self.assertTrue(ret.iloc[0]["bfloat16_neg_inf"] < 0)  # negative infinity
        self.assertTrue(math.isnan(ret.iloc[0]["bfloat16_nan"]))

        # Test second row (same values, consistency check)
        self.assertTrue(math.isinf(ret.iloc[1]["float32_pos_inf"]))
        self.assertTrue(ret.iloc[1]["float32_pos_inf"] > 0)
        self.assertTrue(math.isinf(ret.iloc[1]["float32_neg_inf"]))
        self.assertTrue(ret.iloc[1]["float32_neg_inf"] < 0)
        self.assertTrue(math.isnan(ret.iloc[1]["float32_nan"]))

        self.assertTrue(math.isinf(ret.iloc[1]["float64_pos_inf"]))
        self.assertTrue(ret.iloc[1]["float64_pos_inf"] > 0)
        self.assertTrue(math.isinf(ret.iloc[1]["float64_neg_inf"]))
        self.assertTrue(ret.iloc[1]["float64_neg_inf"] < 0)
        self.assertTrue(math.isnan(ret.iloc[1]["float64_nan"]))

        self.assertTrue(math.isinf(ret.iloc[1]["bfloat16_pos_inf"]))
        self.assertTrue(ret.iloc[1]["bfloat16_pos_inf"] > 0)
        self.assertTrue(math.isinf(ret.iloc[1]["bfloat16_neg_inf"]))
        self.assertTrue(ret.iloc[1]["bfloat16_neg_inf"] < 0)
        self.assertTrue(math.isnan(ret.iloc[1]["bfloat16_nan"]))

        # Precise data type validation
        expected_types = {
            "float32_pos_inf": "float32",
            "float32_neg_inf": "float32",
            "float32_nan": "float32",
            "float64_pos_inf": "float64",
            "float64_neg_inf": "float64",
            "float64_nan": "float64",
            "bfloat16_pos_inf": "float32",  # BFloat16 typically mapped to float32 in pandas
            "bfloat16_neg_inf": "float32",
            "bfloat16_nan": "float32"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_decimal_types(self):
        """Test Decimal32, Decimal64, Decimal128, Decimal256 types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toDecimal32('123.456', 3) as decimal32_val,
                    toDecimal32('-999999.999', 3) as decimal32_min,
                    toDecimal32('999999.999', 3) as decimal32_max,
                    toDecimal64('123456.789012', 6) as decimal64_val,
                    toDecimal64('-999999999999.999999', 6) as decimal64_min,
                    toDecimal64('999999999999.999999', 6) as decimal64_max,
                    toDecimal128('12345678901234567890123456789.123456789', 9) as decimal128_val,
                    toDecimal128('-12345678901234567890123456789.123456789', 9) as decimal128_min,
                    toDecimal128('12345678901234567890123456789.123456789', 9) as decimal128_max,
                    toDecimal256('1234567890123456789012345678901234567890123456789012345678.123456789012345678', 18) as decimal256_val,
                    toDecimal256('-1234567890123456789012345678901234567890123456789012345678.123456789012345678', 18) as decimal256_min,
                    toDecimal256('1234567890123456789012345678901234567890123456789012345678.123456789012345678', 18) as decimal256_max
                UNION ALL
                SELECT
                    2 as row_id,
                    toDecimal32('0.001', 3) as decimal32_val,
                    toDecimal32('0.000', 3) as decimal32_min,
                    toDecimal32('1.000', 3) as decimal32_max,
                    toDecimal64('0.000001', 6) as decimal64_val,
                    toDecimal64('0.000000', 6) as decimal64_min,
                    toDecimal64('1.000000', 6) as decimal64_max,
                    toDecimal128('0.000000001', 9) as decimal128_val,
                    toDecimal128('0.000000000', 9) as decimal128_min,
                    toDecimal128('1.000000000', 9) as decimal128_max,
                    toDecimal256('0.000000000000000001', 18) as decimal256_val,
                    toDecimal256('0.000000000000000000', 18) as decimal256_min,
                    toDecimal256('1.000000000000000000', 18) as decimal256_max
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - regular and extreme decimal values (converted to float64)
        self.assertAlmostEqual(ret.iloc[0]["decimal32_val"], 123.456, places=3)
        self.assertAlmostEqual(ret.iloc[0]["decimal32_min"], -999999.999, places=3)
        self.assertAlmostEqual(ret.iloc[0]["decimal32_max"], 999999.999, places=3)

        self.assertAlmostEqual(ret.iloc[0]["decimal64_val"], 123456.789012, places=6)
        self.assertAlmostEqual(ret.iloc[0]["decimal64_min"], -999999999999.999999, places=6)
        self.assertAlmostEqual(ret.iloc[0]["decimal64_max"], 999999999999.999999, places=6)

        self.assertAlmostEqual(ret.iloc[0]["decimal128_val"], 12345678901234567890123456789.123456789, delta=1e20)
        self.assertAlmostEqual(ret.iloc[0]["decimal128_min"], -12345678901234567890123456789.123456789, delta=1e20)
        self.assertAlmostEqual(ret.iloc[0]["decimal128_max"], 12345678901234567890123456789.123456789, delta=1e20)

        self.assertAlmostEqual(ret.iloc[0]["decimal256_val"], 1234567890123456789012345678901234567890123456789012345678.123456789012345678, delta=1e50)
        self.assertAlmostEqual(ret.iloc[0]["decimal256_min"], -1234567890123456789012345678901234567890123456789012345678.123456789012345678, delta=1e50)
        self.assertAlmostEqual(ret.iloc[0]["decimal256_max"], 1234567890123456789012345678901234567890123456789012345678.123456789012345678, delta=1e50)

        # Test second row - small decimal values (converted to float64)
        self.assertAlmostEqual(ret.iloc[1]["decimal32_val"], 0.001, places=3)
        self.assertEqual(ret.iloc[1]["decimal32_min"], 0.000)
        self.assertAlmostEqual(ret.iloc[1]["decimal32_max"], 1.000, places=3)

        self.assertAlmostEqual(ret.iloc[1]["decimal64_val"], 0.000001, places=6)
        self.assertEqual(ret.iloc[1]["decimal64_min"], 0.000000)
        self.assertAlmostEqual(ret.iloc[1]["decimal64_max"], 1.000000, places=6)

        self.assertAlmostEqual(ret.iloc[1]["decimal128_val"], 0.000000001, places=9)
        self.assertEqual(ret.iloc[1]["decimal128_min"], 0.000000000)
        self.assertAlmostEqual(ret.iloc[1]["decimal128_max"], 1.000000000, places=9)

        self.assertAlmostEqual(ret.iloc[1]["decimal256_val"], 0.000000000000000001, places=18)
        self.assertEqual(ret.iloc[1]["decimal256_min"], 0.000000000000000000)
        self.assertAlmostEqual(ret.iloc[1]["decimal256_max"], 1.000000000000000000, places=18)

        # Precise data type validation
        expected_types = {
            "decimal32_val": "float64",  # Decimal types mapped to float64 in ClickHouse->pandas conversion
            "decimal32_min": "float64",
            "decimal32_max": "float64",
            "decimal64_val": "float64",
            "decimal64_min": "float64",
            "decimal64_max": "float64",
            "decimal128_val": "float64",
            "decimal128_min": "float64",
            "decimal128_max": "float64",
            "decimal256_val": "float64",
            "decimal256_min": "float64",
            "decimal256_max": "float64"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_string_types(self):
        """Test String, FixedString, and LowCardinality string types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toString('Hello World') as string_val,
                    toFixedString('Fixed', 10) as fixed_string_val,
                    toLowCardinality('Category A') as low_cardinality_val,
                    toString('') as empty_string,
                    toString('Unicode: 🌍 éñáíóú') as unicode_string,
                    toString('Special chars: \\t\\n\\r\\"\\\'') as special_chars,
                    toString('Very long string with many characters to test maximum length handling and memory allocation behavior') as long_string,
                    toFixedString('ABC', 5) as fixed_string_short,
                    toLowCardinality('') as low_cardinality_empty
                UNION ALL
                SELECT
                    2 as row_id,
                    toString('Another string') as string_val,
                    toFixedString('Test123', 10) as fixed_string_val,
                    toLowCardinality('Category B') as low_cardinality_val,
                    toString('Non-empty') as empty_string,
                    toString('More Unicode: 🚀 ñáéíóú àèìòù') as unicode_string,
                    toString('Line breaks:\\nTab:\\tQuote:\\"') as special_chars,
                    toString('Short') as long_string,
                    toFixedString('XYZZZ', 5) as fixed_string_short,
                    toLowCardinality('Option 2') as low_cardinality_empty
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - various string types
        self.assertEqual(ret.iloc[0]["string_val"], "Hello World")
        self.assertEqual(ret.iloc[0]["fixed_string_val"], "Fixed\x00\x00\x00\x00\x00")  # FixedString pads with null bytes
        self.assertEqual(ret.iloc[0]["low_cardinality_val"], "Category A")
        self.assertEqual(ret.iloc[0]["empty_string"], "")
        self.assertEqual(ret.iloc[0]["unicode_string"], "Unicode: 🌍 éñáíóú")
        self.assertEqual(ret.iloc[0]["special_chars"], "Special chars: \t\n\r\"'")  # ClickHouse interprets escape sequences
        self.assertEqual(ret.iloc[0]["long_string"], "Very long string with many characters to test maximum length handling and memory allocation behavior")
        self.assertEqual(ret.iloc[0]["fixed_string_short"], "ABC\x00\x00")  # Padded to 5 chars
        self.assertEqual(ret.iloc[0]["low_cardinality_empty"], "")

        # Test second row - different string values
        self.assertEqual(ret.iloc[1]["string_val"], "Another string")
        self.assertEqual(ret.iloc[1]["fixed_string_val"], "Test123\x00\x00\x00")  # Padded to 10 chars
        self.assertEqual(ret.iloc[1]["low_cardinality_val"], "Category B")
        self.assertEqual(ret.iloc[1]["empty_string"], "Non-empty")
        self.assertEqual(ret.iloc[1]["unicode_string"], "More Unicode: 🚀 ñáéíóú àèìòù")
        self.assertEqual(ret.iloc[1]["special_chars"], "Line breaks:\nTab:\tQuote:\"")  # ClickHouse interprets escape sequences
        self.assertEqual(ret.iloc[1]["long_string"], "Short")
        self.assertEqual(ret.iloc[1]["fixed_string_short"], "XYZZZ")  # Exactly 5 chars, no padding
        self.assertEqual(ret.iloc[1]["low_cardinality_empty"], "Option 2")

        # Precise data type validation
        expected_types = {
            "string_val": "object",  # String types mapped to object in pandas
            "fixed_string_val": "object",
            "low_cardinality_val": "object",
            "empty_string": "object",
            "unicode_string": "object",
            "special_chars": "object",
            "long_string": "object",
            "fixed_string_short": "object",
            "low_cardinality_empty": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_date_types(self):
        """Test Date and Date32 types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toDate('2023-12-25') as date_val,
                    toDate('1970-01-01') as date_min,
                    toDate('2149-06-06') as date_max,
                    toDate32('2023-12-25') as date32_val,
                    toDate32('1900-01-01') as date32_min,
                    toDate32('2299-12-31') as date32_max,
                    toDate('2000-02-29') as date_leap_year,
                    toDate32('2000-02-29') as date32_leap_year,
                    toDate32('1950-06-15') as date32_negative_1,
                    toDate32('1960-12-31') as date32_negative_2,
                    toDate32('1969-12-31') as date32_before_epoch
                UNION ALL
                SELECT
                    2 as row_id,
                    toDate('1970-01-01') as date_val,
                    toDate('2023-01-01') as date_min,
                    toDate('2023-12-31') as date_max,
                    toDate32('1970-01-01') as date32_val,
                    toDate32('2023-01-01') as date32_min,
                    toDate32('2023-12-31') as date32_max,
                    toDate('2024-02-29') as date_leap_year,
                    toDate32('2024-02-29') as date32_leap_year,
                    toDate32('1945-05-08') as date32_negative_1,
                    toDate32('1955-03-20') as date32_negative_2,
                    toDate32('1968-07-20') as date32_before_epoch
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - specific dates (Date types include time component 00:00:00)
        self.assertIn("2023-12-25", str(ret.iloc[0]["date_val"]))
        self.assertIn("1970-01-01", str(ret.iloc[0]["date_min"]))
        self.assertIn("2149-06-06", str(ret.iloc[0]["date_max"]))
        self.assertIn("2023-12-25", str(ret.iloc[0]["date32_val"]))
        self.assertIn("1900-01-01", str(ret.iloc[0]["date32_min"]))
        self.assertIn("2299-12-31", str(ret.iloc[0]["date32_max"]))
        self.assertIn("2000-02-29", str(ret.iloc[0]["date_leap_year"]))
        self.assertIn("2000-02-29", str(ret.iloc[0]["date32_leap_year"]))
        # Test Date32 negative values (before 1970 epoch)
        self.assertIn("1950-06-15", str(ret.iloc[0]["date32_negative_1"]))
        self.assertIn("1960-12-31", str(ret.iloc[0]["date32_negative_2"]))
        self.assertIn("1969-12-31", str(ret.iloc[0]["date32_before_epoch"]))

        # Test second row - different dates
        self.assertIn("1970-01-01", str(ret.iloc[1]["date_val"]))
        self.assertIn("2023-01-01", str(ret.iloc[1]["date_min"]))
        self.assertIn("2023-12-31", str(ret.iloc[1]["date_max"]))
        self.assertIn("1970-01-01", str(ret.iloc[1]["date32_val"]))
        self.assertIn("2023-01-01", str(ret.iloc[1]["date32_min"]))
        self.assertIn("2023-12-31", str(ret.iloc[1]["date32_max"]))
        self.assertIn("2024-02-29", str(ret.iloc[1]["date_leap_year"]))
        self.assertIn("2024-02-29", str(ret.iloc[1]["date32_leap_year"]))
        # Test Date32 negative values (before 1970 epoch) - second row
        self.assertIn("1945-05-08", str(ret.iloc[1]["date32_negative_1"]))
        self.assertIn("1955-03-20", str(ret.iloc[1]["date32_negative_2"]))
        self.assertIn("1968-07-20", str(ret.iloc[1]["date32_before_epoch"]))

        # Precise data type validation
        expected_types = {
            "date_val": "datetime64[s]",  # Date types mapped to datetime64[s] in pandas
            "date_min": "datetime64[s]",
            "date_max": "datetime64[s]",
            "date32_val": "datetime64[s]",
            "date32_min": "datetime64[s]",
            "date32_max": "datetime64[s]",
            "date_leap_year": "datetime64[s]",
            "date32_leap_year": "datetime64[s]",
            "date32_negative_1": "datetime64[s]",  # Date32 negative values (before 1970)
            "date32_negative_2": "datetime64[s]",
            "date32_before_epoch": "datetime64[s]"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_time_types(self):
        """Test Time and Time64 types"""
        # Enable Time and Time64 types
        self.session.query("SET enable_time_time64_type = 1")

        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    CAST('14:30:45' AS Time) as time_val,
                    CAST('00:00:00' AS Time) as time_min,
                    CAST('23:59:59' AS Time) as time_max,
                    CAST('14:30:45.123456' AS Time64(6)) as time64_val,
                    CAST('00:00:00.000000' AS Time64(6)) as time64_min,
                    CAST('23:59:59.999999' AS Time64(6)) as time64_max,
                    CAST('12:00:00.123' AS Time64(3)) as time64_ms,
                    CAST('18:45:30.987654321' AS Time64(9)) as time64_ns
                UNION ALL
                SELECT
                    2 as row_id,
                    CAST('09:15:30' AS Time) as time_val,
                    CAST('12:00:00' AS Time) as time_min,
                    CAST('18:45:15' AS Time) as time_max,
                    CAST('09:15:30.654321' AS Time64(6)) as time64_val,
                    CAST('12:30:45.500000' AS Time64(6)) as time64_min,
                    CAST('20:15:30.111111' AS Time64(6)) as time64_max,
                    CAST('08:30:15.500' AS Time64(3)) as time64_ms,
                    CAST('16:20:10.123456789' AS Time64(9)) as time64_ns
                UNION ALL
                SELECT
                    3 as row_id,
                    CAST(-3600 AS Time) as time_val,       -- -1 hour as negative seconds
                    CAST(-7200 AS Time) as time_min,       -- -2 hours as negative seconds
                    CAST(-1800 AS Time) as time_max,       -- -30 minutes as negative seconds
                    CAST(-3661.123456 AS Time64(6)) as time64_val,  -- -1h 1m 1.123456s
                    CAST(-7322.500000 AS Time64(6)) as time64_min,  -- -2h 2m 2.5s
                    CAST(-1801.999999 AS Time64(6)) as time64_max,  -- -30m 1.999999s
                    CAST(-3723.500 AS Time64(3)) as time64_ms,      -- -1h 2m 3.5s
                    CAST(-5434.123456789 AS Time64(9)) as time64_ns -- -1h 30m 34.123456789s
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - time values
        self.assertIn("14:30:45", str(ret.iloc[0]["time_val"]))
        self.assertIn("00:00:00", str(ret.iloc[0]["time_min"]))
        self.assertIn("23:59:59", str(ret.iloc[0]["time_max"]))
        self.assertIn("14:30:45", str(ret.iloc[0]["time64_val"]))
        self.assertIn("00:00:00", str(ret.iloc[0]["time64_min"]))
        self.assertIn("23:59:59", str(ret.iloc[0]["time64_max"]))
        self.assertIn("12:00:00", str(ret.iloc[0]["time64_ms"]))
        self.assertIn("18:45:30", str(ret.iloc[0]["time64_ns"]))

        # Test second row - different time values
        self.assertIn("09:15:30", str(ret.iloc[1]["time_val"]))
        self.assertIn("12:00:00", str(ret.iloc[1]["time_min"]))
        self.assertIn("18:45:15", str(ret.iloc[1]["time_max"]))
        self.assertIn("09:15:30", str(ret.iloc[1]["time64_val"]))
        self.assertIn("12:30:45", str(ret.iloc[1]["time64_min"]))
        self.assertIn("20:15:30", str(ret.iloc[1]["time64_max"]))
        self.assertIn("08:30:15", str(ret.iloc[1]["time64_ms"]))
        self.assertIn("16:20:10", str(ret.iloc[1]["time64_ns"]))

        # Test third row - negative time values (should be returned as string numbers)
        # Since Python time types don't support negative values, they are returned as numeric strings
        self.assertEqual(ret.iloc[2]["time_val"], "-3600")          # -1 hour
        self.assertEqual(ret.iloc[2]["time_min"], "-7200")          # -2 hours
        self.assertEqual(ret.iloc[2]["time_max"], "-1800")          # -30 minutes
        self.assertEqual(ret.iloc[2]["time64_val"], "-3661.123456") # -1h 1m 1.123456s
        self.assertEqual(ret.iloc[2]["time64_min"], "-7322.5")      # -2h 2m 2.5s
        self.assertEqual(ret.iloc[2]["time64_max"], "-1801.999999") # -30m 1.999999s
        self.assertEqual(ret.iloc[2]["time64_ms"], "-3723.5")       # -1h 2m 3.5s
        self.assertEqual(ret.iloc[2]["time64_ns"], "-5434.123456789") # -1h 30m 34.123456789s

        # Verify negative values are returned as strings (object dtype)
        for col in ["time_val", "time_min", "time_max", "time64_val", "time64_min", "time64_max", "time64_ms", "time64_ns"]:
            self.assertIsInstance(ret.iloc[2][col], str, f"{col} should be string for negative values")

        # Precise data type validation
        expected_types = {
            "time_val": "object",  # Time types mapped to object in pandas
            "time_min": "object",
            "time_max": "object",
            "time64_val": "object",
            "time64_min": "object",
            "time64_max": "object",
            "time64_ms": "object",
            "time64_ns": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_datetime_types(self):
        """Test DateTime and DateTime64 types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toDateTime('2023-12-25 14:30:45', 'Asia/Shanghai') as datetime_val,
                    toDateTime('1970-01-02 00:00:00', 'Asia/Shanghai') as datetime_min,
                    toDateTime('2106-02-07 06:28:15', 'Asia/Shanghai') as datetime_max,
                    toDateTime64('2023-12-25 14:30:45.123456', 6, 'Asia/Shanghai') as datetime64_val,
                    toDateTime64('1902-01-01 00:00:00.000000', 6, 'Asia/Shanghai') as datetime64_min,
                    toDateTime64('2099-12-31 10:59:59.999999', 6, 'Asia/Shanghai') as datetime64_max,
                    toDateTime64('2023-12-25 14:30:45.123456789', 9, 'Asia/Shanghai') as datetime64_ns,
                    toDateTime('2023-06-15 12:00:00', 'UTC') as datetime_utc,
                    toDateTime('2023-06-15 15:30:00', 'Europe/London') as datetime_london,
                    toDateTime64('2023-06-15 12:00:00.123', 3, 'Asia/Shanghai') as datetime64_tz_sh,
                    toDateTime64('2023-06-15 12:00:00.456', 3, 'America/New_York') as datetime64_tz_ny
                UNION ALL
                SELECT
                    2 as row_id,
                    toDateTime('2000-02-29 09:15:30', 'Asia/Shanghai') as datetime_val,
                    toDateTime('2023-01-01 12:30:45', 'Asia/Shanghai') as datetime_min,
                    toDateTime('2023-12-31 18:45:15', 'Asia/Shanghai') as datetime_max,
                    toDateTime64('2000-02-29 09:15:30.654321', 6, 'Asia/Shanghai') as datetime64_val,
                    toDateTime64('2023-01-01 08:00:00.111111', 6, 'Asia/Shanghai') as datetime64_min,
                    toDateTime64('2023-12-31 20:30:45.888888', 6, 'Asia/Shanghai') as datetime64_max,
                    toDateTime64('2000-02-29 09:15:30.987654321', 9, 'Asia/Shanghai') as datetime64_ns,
                    toDateTime('2024-01-15 08:30:00', 'UTC') as datetime_utc,
                    toDateTime('2024-01-15 20:00:00', 'Europe/London') as datetime_london,
                    toDateTime64('2024-01-15 16:45:30.789', 3, 'Asia/Shanghai') as datetime64_tz_sh,
                    toDateTime64('2024-01-15 09:15:45.987', 3, 'America/New_York') as datetime64_tz_ny
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Note: Historical timezone offsets vary for the same location across different periods.
        # For example, in 1900, Shanghai had a UTC offset of +8:05:43 (8 hours 5 minutes 43 seconds).
        # So executing session.query("select toDateTime64('1900-01-01 00:00:00.000000', 6, 'Asia/Shanghai')", "DataFrame")
        # would output 1900-01-01 00:00:17+08:06 in pandas instead of the standard +08:00


        # Test first row - exact datetime values
        # DateTime (second precision) - ClickHouse uses server timezone
        # Get system timezone dynamically
        actual_tz = "Asia/Shanghai"

        self.assertEqual(ret.iloc[0]["datetime_val"], pd.Timestamp('2023-12-25 14:30:45', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime_min"], pd.Timestamp('1970-01-02 00:00:00', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime_max"], pd.Timestamp('2106-02-07 06:28:15', tz=actual_tz))

        # DateTime64 (microsecond precision) - should use same timezone as ClickHouse server
        self.assertEqual(ret.iloc[0]["datetime64_val"], pd.Timestamp('2023-12-25 14:30:45.123456', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime64_min"], pd.Timestamp('1902-01-01 00:00:00.000000', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime64_max"], pd.Timestamp('2099-12-31 10:59:59.999999', tz=actual_tz))

        # DateTime64 (nanosecond precision) - should use same timezone as ClickHouse server
        self.assertEqual(ret.iloc[0]["datetime64_ns"], pd.Timestamp('2023-12-25 14:30:45.123456789', tz=actual_tz))

        # UTC timezone datetime
        expected_utc = pd.Timestamp('2023-06-15 12:00:00', tz='UTC')
        actual_utc = ret.iloc[0]["datetime_utc"]
        self.assertEqual(actual_utc, expected_utc)

        # Europe/London timezone datetime
        expected_london = pd.Timestamp('2023-06-15 15:30:00', tz='Europe/London')
        actual_london = ret.iloc[0]["datetime_london"]
        self.assertEqual(actual_london, expected_london)

        # Timezone-aware datetime64 - Asia/Shanghai
        expected_sh = pd.Timestamp('2023-06-15 12:00:00.123', tz='Asia/Shanghai')
        actual_sh = ret.iloc[0]["datetime64_tz_sh"]
        self.assertEqual(actual_sh, expected_sh)

        # Timezone-aware datetime64 - America/New_York
        expected_ny = pd.Timestamp('2023-06-15 12:00:00.456', tz='America/New_York')
        actual_ny = ret.iloc[0]["datetime64_tz_ny"]
        self.assertEqual(actual_ny, expected_ny)

        # Test second row - exact datetime values with ClickHouse server timezone
        self.assertEqual(ret.iloc[1]["datetime_val"], pd.Timestamp('2000-02-29 09:15:30', tz=actual_tz))
        self.assertEqual(ret.iloc[1]["datetime_min"], pd.Timestamp('2023-01-01 12:30:45', tz=actual_tz))
        self.assertEqual(ret.iloc[1]["datetime_max"], pd.Timestamp('2023-12-31 18:45:15', tz=actual_tz))
        self.assertEqual(ret.iloc[1]["datetime64_val"], pd.Timestamp('2000-02-29 09:15:30.654321', tz=actual_tz))
        self.assertEqual(ret.iloc[1]["datetime64_min"], pd.Timestamp('2023-01-01 08:00:00.111111', tz=actual_tz))
        self.assertEqual(ret.iloc[1]["datetime64_max"], pd.Timestamp('2023-12-31 20:30:45.888888', tz=actual_tz))
        self.assertEqual(ret.iloc[1]["datetime64_ns"], pd.Timestamp('2000-02-29 09:15:30.987654321', tz=actual_tz))

        # Second row timezone datetime tests
        expected_utc_2 = pd.Timestamp('2024-01-15 08:30:00', tz='UTC')
        actual_utc_2 = ret.iloc[1]["datetime_utc"]
        self.assertEqual(actual_utc_2, expected_utc_2)

        expected_london_2 = pd.Timestamp('2024-01-15 20:00:00', tz='Europe/London')
        actual_london_2 = ret.iloc[1]["datetime_london"]
        self.assertEqual(actual_london_2, expected_london_2)

        # Second row timezone tests (already converted by C++ code)
        expected_sh_2 = pd.Timestamp('2024-01-15 16:45:30.789', tz='Asia/Shanghai')
        actual_sh_2 = ret.iloc[1]["datetime64_tz_sh"]
        self.assertEqual(actual_sh_2, expected_sh_2)

        expected_ny_2 = pd.Timestamp('2024-01-15 09:15:45.987', tz='America/New_York')
        actual_ny_2 = ret.iloc[1]["datetime64_tz_ny"]
        self.assertEqual(actual_ny_2, expected_ny_2)

        # Precise data type validation
        expected_types = {
            "row_id": "uint8",
            "datetime_val": "datetime64[s, Asia/Shanghai]",      # DateTime types mapped to datetime64[s] (second precision)
            "datetime_min": "datetime64[s, Asia/Shanghai]",
            "datetime_max": "datetime64[s, Asia/Shanghai]",
            "datetime64_val": "datetime64[ns, Asia/Shanghai]",   # DateTime64 types mapped to datetime64[ns] (nanosecond precision)
            "datetime64_min": "datetime64[ns, Asia/Shanghai]",
            "datetime64_max": "datetime64[ns, Asia/Shanghai]",
            "datetime64_ns": "datetime64[ns, Asia/Shanghai]",    # DateTime64 with 9-digit precision (nanoseconds)
            "datetime_utc": "datetime64[s, UTC]",      # DateTime with timezone -> datetime64[s]
            "datetime64_tz_sh": "datetime64[ns, Asia/Shanghai]", # DateTime64 with Asia/Shanghai timezone
            "datetime64_tz_ny": "datetime64[ns, America/New_York]"  # DateTime64 with America/New_York timezone
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    @unittest.skip("")
    def test_enum_types(self):
        """Test Enum8 and Enum16 types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    CAST('hello' AS Enum8('hello' = 1, 'world' = 2)) as enum8_val,
                    CAST('small' AS Enum8('small' = -128, 'medium' = 0, 'large' = 127)) as enum8_range,
                    CAST('active' AS Enum16('active' = 1, 'inactive' = 2, 'pending' = 3, 'deleted' = -1)) as enum16_val,
                    CAST('north' AS Enum16('north' = 1, 'south' = 2, 'east' = 3, 'west' = 4, 'center' = 0)) as enum16_direction
                UNION ALL
                SELECT
                    2 as row_id,
                    CAST('world' AS Enum8('hello' = 1, 'world' = 2)) as enum8_val,
                    CAST('large' AS Enum8('small' = -128, 'medium' = 0, 'large' = 127)) as enum8_range,
                    CAST('deleted' AS Enum16('active' = 1, 'inactive' = 2, 'pending' = 3, 'deleted' = -1)) as enum16_val,
                    CAST('south' AS Enum16('north' = 1, 'south' = 2, 'east' = 3, 'west' = 4, 'center' = 0)) as enum16_direction
                UNION ALL
                SELECT
                    3 as row_id,
                    CAST('hello' AS Enum8('hello' = 1, 'world' = 2)) as enum8_val,
                    CAST('medium' AS Enum8('small' = -128, 'medium' = 0, 'large' = 127)) as enum8_range,
                    CAST('pending' AS Enum16('active' = 1, 'inactive' = 2, 'pending' = 3, 'deleted' = -1)) as enum16_val,
                    CAST('center' AS Enum16('north' = 1, 'south' = 2, 'east' = 3, 'west' = 4, 'center' = 0)) as enum16_direction
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row values
        self.assertEqual(ret.iloc[0]["enum8_val"], "hello")
        self.assertEqual(ret.iloc[0]["enum8_range"], "small")
        self.assertEqual(ret.iloc[0]["enum16_val"], "active")
        self.assertEqual(ret.iloc[0]["enum16_direction"], "north")

        # Test second row values
        self.assertEqual(ret.iloc[1]["enum8_val"], "world")
        self.assertEqual(ret.iloc[1]["enum8_range"], "large")
        self.assertEqual(ret.iloc[1]["enum16_val"], "deleted")
        self.assertEqual(ret.iloc[1]["enum16_direction"], "south")

        # Test third row values
        self.assertEqual(ret.iloc[2]["enum8_val"], "hello")
        self.assertEqual(ret.iloc[2]["enum8_range"], "medium")
        self.assertEqual(ret.iloc[2]["enum16_val"], "pending")
        self.assertEqual(ret.iloc[2]["enum16_direction"], "center")

        # Verify data types - Enum types should be mapped to object (string) dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "enum8_val": "object",      # Enum8 mapped to object (string) dtype
            "enum8_range": "object",    # Enum8 with negative/positive range
            "enum16_val": "object",     # Enum16 mapped to object (string) dtype
            "enum16_direction": "object" # Enum16 with multiple values
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

        # Verify all enum values are strings
        for col in ["enum8_val", "enum8_range", "enum16_val", "enum16_direction"]:
            for i in range(len(ret)):
                self.assertIsInstance(ret.iloc[i][col], str, f"Row {i}, column {col} should be string")

    @unittest.skip("")
    def test_uuid_types(self):
        """Test UUID data type"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toUUID('550e8400-e29b-41d4-a716-446655440000') as uuid_fixed1,
                    toUUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') as uuid_fixed2,
                    generateUUIDv4() as uuid_random1,
                    generateUUIDv4() as uuid_random2
                UNION ALL
                SELECT
                    2 as row_id,
                    toUUID('123e4567-e89b-12d3-a456-426614174000') as uuid_fixed1,
                    toUUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8') as uuid_fixed2,
                    generateUUIDv4() as uuid_random1,
                    generateUUIDv4() as uuid_random2
                UNION ALL
                SELECT
                    3 as row_id,
                    toUUID('00000000-0000-0000-0000-000000000000') as uuid_fixed1,
                    toUUID('ffffffff-ffff-ffff-ffff-ffffffffffff') as uuid_fixed2,
                    generateUUIDv4() as uuid_random1,
                    generateUUIDv4() as uuid_random2
            )
            ORDER BY row_id
        """, "DataFrame")

        # Verify we have 3 rows and 5 columns
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret.columns), 5)

        # Test first row fixed UUID values
        self.assertEqual(ret.iloc[0]["uuid_fixed1"], uuid.UUID("550e8400-e29b-41d4-a716-446655440000"))
        self.assertEqual(ret.iloc[0]["uuid_fixed2"], uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"))

        # Test second row fixed UUID values
        self.assertEqual(ret.iloc[1]["uuid_fixed1"], uuid.UUID("123e4567-e89b-12d3-a456-426614174000"))
        self.assertEqual(ret.iloc[1]["uuid_fixed2"], uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8"))

        # Test third row special UUID values (all zeros and all F's)
        self.assertEqual(ret.iloc[2]["uuid_fixed1"], uuid.UUID("00000000-0000-0000-0000-000000000000"))
        self.assertEqual(ret.iloc[2]["uuid_fixed2"], uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"))

        # Verify data types - UUID types should be mapped to object dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "uuid_fixed1": "object",     # UUID mapped to object dtype (contains UUID objects)
            "uuid_fixed2": "object",     # UUID mapped to object dtype (contains UUID objects)
            "uuid_random1": "object",    # Generated UUID mapped to object dtype (contains UUID objects)
            "uuid_random2": "object"     # Generated UUID mapped to object dtype (contains UUID objects)
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

        # Verify all UUID values are UUID objects and have valid format
        for col in ["uuid_fixed1", "uuid_fixed2", "uuid_random1", "uuid_random2"]:
            for i in range(len(ret)):
                uuid_value = ret.iloc[i][col]
                self.assertIsInstance(uuid_value, uuid.UUID, f"Row {i}, column {col} should be UUID object")
                # Verify UUID string representation has correct format
                uuid_str = str(uuid_value)
                self.assertEqual(len(uuid_str), 36, f"Row {i}, column {col} UUID string should be 36 characters")
                self.assertEqual(uuid_str.count('-'), 4, f"Row {i}, column {col} UUID should have 4 hyphens")

    @unittest.skip("")
    def test_ipv4_types(self):
        """Test IPv4 data type"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toIPv4('192.168.1.1') as ipv4_private,
                    toIPv4('8.8.8.8') as ipv4_public,
                    toIPv4('127.0.0.1') as ipv4_localhost,
                    toIPv4('0.0.0.0') as ipv4_zero,
                    toIPv4('255.255.255.255') as ipv4_broadcast
                UNION ALL
                SELECT
                    2 as row_id,
                    toIPv4('10.0.0.1') as ipv4_private,
                    toIPv4('1.1.1.1') as ipv4_public,
                    toIPv4('127.0.0.2') as ipv4_localhost,
                    toIPv4('172.16.0.1') as ipv4_zero,
                    toIPv4('203.0.113.1') as ipv4_broadcast
                UNION ALL
                SELECT
                    3 as row_id,
                    toIPv4('192.0.2.1') as ipv4_private,
                    toIPv4('208.67.222.222') as ipv4_public,
                    toIPv4('169.254.1.1') as ipv4_localhost,
                    toIPv4('224.0.0.1') as ipv4_zero,
                    toIPv4('239.255.255.255') as ipv4_broadcast
            )
            ORDER BY row_id
        """, "DataFrame")

        # Verify we have 3 rows and 6 columns
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret.columns), 6)

        # Test first row IPv4 values
        self.assertEqual(ret.iloc[0]["ipv4_private"], ipaddress.IPv4Address("192.168.1.1"))
        self.assertEqual(ret.iloc[0]["ipv4_public"], ipaddress.IPv4Address("8.8.8.8"))
        self.assertEqual(ret.iloc[0]["ipv4_localhost"], ipaddress.IPv4Address("127.0.0.1"))
        self.assertEqual(ret.iloc[0]["ipv4_zero"], ipaddress.IPv4Address("0.0.0.0"))
        self.assertEqual(ret.iloc[0]["ipv4_broadcast"], ipaddress.IPv4Address("255.255.255.255"))

        # Test second row IPv4 values
        self.assertEqual(ret.iloc[1]["ipv4_private"], ipaddress.IPv4Address("10.0.0.1"))
        self.assertEqual(ret.iloc[1]["ipv4_public"], ipaddress.IPv4Address("1.1.1.1"))
        self.assertEqual(ret.iloc[1]["ipv4_localhost"], ipaddress.IPv4Address("127.0.0.2"))
        self.assertEqual(ret.iloc[1]["ipv4_zero"], ipaddress.IPv4Address("172.16.0.1"))
        self.assertEqual(ret.iloc[1]["ipv4_broadcast"], ipaddress.IPv4Address("203.0.113.1"))

        # Test third row IPv4 values
        self.assertEqual(ret.iloc[2]["ipv4_private"], ipaddress.IPv4Address("192.0.2.1"))
        self.assertEqual(ret.iloc[2]["ipv4_public"], ipaddress.IPv4Address("208.67.222.222"))
        self.assertEqual(ret.iloc[2]["ipv4_localhost"], ipaddress.IPv4Address("169.254.1.1"))
        self.assertEqual(ret.iloc[2]["ipv4_zero"], ipaddress.IPv4Address("224.0.0.1"))
        self.assertEqual(ret.iloc[2]["ipv4_broadcast"], ipaddress.IPv4Address("239.255.255.255"))

        # Verify data types - IPv4 types should be mapped to object dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "ipv4_private": "object",    # IPv4Address mapped to object dtype
            "ipv4_public": "object",
            "ipv4_localhost": "object",
            "ipv4_zero": "object",
            "ipv4_broadcast": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

        # Verify all IPv4 values are IPv4Address objects
        for col in ["ipv4_private", "ipv4_public", "ipv4_localhost", "ipv4_zero", "ipv4_broadcast"]:
            for i in range(len(ret)):
                ipv4_value = ret.iloc[i][col]
                self.assertIsInstance(ipv4_value, ipaddress.IPv4Address, f"Row {i}, column {col} should be IPv4Address object")
                # Verify IPv4 string representation is valid
                ipv4_str = str(ipv4_value)
                self.assertEqual(len(ipv4_str.split('.')), 4, f"Row {i}, column {col} IPv4 should have 4 octets")

    @unittest.skip("")
    def test_ipv6_types(self):
        """Test IPv6 data type"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    toIPv6('2001:db8::1') as ipv6_standard,
                    toIPv6('::1') as ipv6_localhost,
                    toIPv6('::') as ipv6_zero,
                    toIPv6('2001:db8:85a3::8a2e:370:7334') as ipv6_full,
                    toIPv6('fe80::1') as ipv6_link_local
                UNION ALL
                SELECT
                    2 as row_id,
                    toIPv6('2001:db8::2') as ipv6_standard,
                    toIPv6('::2') as ipv6_localhost,
                    toIPv6('2001:db8::') as ipv6_zero,
                    toIPv6('2001:db8:85a3:0:0:8a2e:370:7335') as ipv6_full,
                    toIPv6('fe80::2') as ipv6_link_local
                UNION ALL
                SELECT
                    3 as row_id,
                    toIPv6('2001:0db8:0000:0000:0000:ff00:0042:8329') as ipv6_standard,
                    toIPv6('::ffff:192.0.2.1') as ipv6_localhost,
                    toIPv6('2001:db8:85a3::8a2e:370:7336') as ipv6_zero,
                    toIPv6('ff02::1') as ipv6_full,
                    toIPv6('2001:db8:85a3:8d3:1319:8a2e:370:7348') as ipv6_link_local
            )
            ORDER BY row_id
        """, "DataFrame")

        # Verify we have 3 rows and 6 columns
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret.columns), 6)

        # Test first row IPv6 values
        self.assertEqual(ret.iloc[0]["ipv6_standard"], ipaddress.IPv6Address("2001:db8::1"))
        self.assertEqual(ret.iloc[0]["ipv6_localhost"], ipaddress.IPv6Address("::1"))
        self.assertEqual(ret.iloc[0]["ipv6_zero"], ipaddress.IPv6Address("::"))
        self.assertEqual(ret.iloc[0]["ipv6_full"], ipaddress.IPv6Address("2001:db8:85a3::8a2e:370:7334"))
        self.assertEqual(ret.iloc[0]["ipv6_link_local"], ipaddress.IPv6Address("fe80::1"))

        # Test second row IPv6 values
        self.assertEqual(ret.iloc[1]["ipv6_standard"], ipaddress.IPv6Address("2001:db8::2"))
        self.assertEqual(ret.iloc[1]["ipv6_localhost"], ipaddress.IPv6Address("::2"))
        self.assertEqual(ret.iloc[1]["ipv6_zero"], ipaddress.IPv6Address("2001:db8::"))
        self.assertEqual(ret.iloc[1]["ipv6_full"], ipaddress.IPv6Address("2001:db8:85a3::8a2e:370:7335"))
        self.assertEqual(ret.iloc[1]["ipv6_link_local"], ipaddress.IPv6Address("fe80::2"))

        # Test third row IPv6 values
        self.assertEqual(ret.iloc[2]["ipv6_standard"], ipaddress.IPv6Address("2001:db8::ff00:42:8329"))
        self.assertEqual(ret.iloc[2]["ipv6_localhost"], ipaddress.IPv6Address("::ffff:192.0.2.1"))
        self.assertEqual(ret.iloc[2]["ipv6_zero"], ipaddress.IPv6Address("2001:db8:85a3::8a2e:370:7336"))
        self.assertEqual(ret.iloc[2]["ipv6_full"], ipaddress.IPv6Address("ff02::1"))
        self.assertEqual(ret.iloc[2]["ipv6_link_local"], ipaddress.IPv6Address("2001:db8:85a3:8d3:1319:8a2e:370:7348"))

        # Verify data types - IPv6 types should be mapped to object dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "ipv6_standard": "object",      # IPv6Address mapped to object dtype
            "ipv6_localhost": "object",
            "ipv6_zero": "object",
            "ipv6_full": "object",
            "ipv6_link_local": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

        # Verify all IPv6 values are IPv6Address objects
        for col in ["ipv6_standard", "ipv6_localhost", "ipv6_zero", "ipv6_full", "ipv6_link_local"]:
            for i in range(len(ret)):
                ipv6_value = ret.iloc[i][col]
                self.assertIsInstance(ipv6_value, ipaddress.IPv6Address, f"Row {i}, column {col} should be IPv6Address object")
                # Verify IPv6 address is valid by checking it can be converted back to string
                ipv6_str = str(ipv6_value)
                self.assertIn(":", ipv6_str, f"Row {i}, column {col} IPv6 should contain colons")


if __name__ == "__main__":
    unittest.main()
