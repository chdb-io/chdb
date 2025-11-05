#!/usr/bin/env python3

import unittest
import pandas as pd
import chdb
from datetime import datetime, date
import numpy as np
import math


class TestDataFrameColumnTypes(unittest.TestCase):

    def setUp(self):
        self.session = chdb.session.Session("./tmp")

    def tearDown(self):
        self.session.close()

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
                    toDateTime('2023-12-25 14:30:45') as datetime_val,
                    toDateTime('1970-01-01 00:00:00') as datetime_min,
                    toDateTime('2106-02-07 06:28:15') as datetime_max,
                    toDateTime64('2023-12-25 14:30:45.123456', 6) as datetime64_val,
                    toDateTime64('1900-01-01 00:00:00.000000', 6) as datetime64_min,
                    toDateTime64('2299-12-31 23:59:59.999999', 6) as datetime64_max,
                    toDateTime64('2023-12-25 14:30:45.123456789', 9) as datetime64_ns,
                    toDateTime('2023-06-15 12:00:00', 'UTC') as datetime_utc,
                    toDateTime('2023-06-15 15:30:00', 'Europe/London') as datetime_london,
                    toDateTime64('2023-06-15 12:00:00.123', 3, 'Asia/Shanghai') as datetime64_tz_sh,
                    toDateTime64('2023-06-15 12:00:00.456', 3, 'America/New_York') as datetime64_tz_ny
                UNION ALL
                SELECT
                    2 as row_id,
                    toDateTime('2000-02-29 09:15:30') as datetime_val,
                    toDateTime('2023-01-01 12:30:45') as datetime_min,
                    toDateTime('2023-12-31 18:45:15') as datetime_max,
                    toDateTime64('2000-02-29 09:15:30.654321', 6) as datetime64_val,
                    toDateTime64('2023-01-01 08:00:00.111111', 6) as datetime64_min,
                    toDateTime64('2023-12-31 20:30:45.888888', 6) as datetime64_max,
                    toDateTime64('2000-02-29 09:15:30.987654321', 9) as datetime64_ns,
                    toDateTime('2024-01-15 08:30:00', 'UTC') as datetime_utc,
                    toDateTime('2024-01-15 20:00:00', 'Europe/London') as datetime_london,
                    toDateTime64('2024-01-15 16:45:30.789', 3, 'Asia/Shanghai') as datetime64_tz_sh,
                    toDateTime64('2024-01-15 09:15:45.987', 3, 'America/New_York') as datetime64_tz_ny
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")


        # Test first row - exact datetime values
        # DateTime (second precision) - ClickHouse uses server timezone (likely Asia/Shanghai)
        # We need to check what timezone ClickHouse is actually using
        actual_tz = 'UTC'

        self.assertEqual(ret.iloc[0]["datetime_val"], pd.Timestamp('2023-12-25 14:30:45', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime_min"], pd.Timestamp('1970-01-01 00:00:00', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime_max"], pd.Timestamp('2106-02-07 06:28:15', tz=actual_tz))

        # DateTime64 (microsecond precision) - should use same timezone as ClickHouse server
        self.assertEqual(ret.iloc[0]["datetime64_val"], pd.Timestamp('2023-12-25 14:30:45.123456', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime64_min"], pd.Timestamp('1900-01-01 00:00:00.000000', tz=actual_tz))
        self.assertEqual(ret.iloc[0]["datetime64_max"], pd.Timestamp('2299-12-31 23:59:59.999999', tz=actual_tz))

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
            "row_id": "int64",
            "datetime_val": "datetime64[s]",      # DateTime types mapped to datetime64[s] (second precision)
            "datetime_min": "datetime64[s]",
            "datetime_max": "datetime64[s]",
            "datetime64_val": "datetime64[ns]",   # DateTime64 types mapped to datetime64[ns] (nanosecond precision)
            "datetime64_min": "datetime64[ns]",
            "datetime64_max": "datetime64[ns]",
            "datetime64_ns": "datetime64[ns]",    # DateTime64 with 9-digit precision (nanoseconds)
            "datetime_utc": "datetime64[s]",      # DateTime with timezone -> datetime64[s]
            "datetime64_tz_sh": "datetime64[ns]", # DateTime64 with Asia/Shanghai timezone
            "datetime64_tz_ny": "datetime64[ns]"  # DateTime64 with America/New_York timezone
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)


if __name__ == "__main__":
    unittest.main()
