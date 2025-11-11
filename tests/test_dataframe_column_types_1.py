#!/usr/bin/env python3

import unittest
import pandas as pd
import chdb
from datetime import datetime, date, timezone, timedelta
import numpy as np
import math
import uuid
import ipaddress


class TestDataFrameColumnTypes(unittest.TestCase):

    def setUp(self):
        self.session = chdb.session.Session()
        self.shanghai_tz = timezone(timedelta(hours=8))

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
                    toUInt8(255) as UInt8_val,
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
                    toUInt8(254) as UInt8_val,
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
        self.assertEqual(ret.iloc[0]["UInt8_val"], 255)
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
        self.assertEqual(ret.iloc[1]["UInt8_val"], 254)
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
            "UInt8_val": "uint8",
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

    def test_bool_types(self):
        """Test Bool and Nullable(Bool) types with various values"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    true as bool_true,
                    false as bool_false,
                    true::Bool as explicit_bool_true,
                    false::Bool as explicit_bool_false,
                    NULL::Nullable(Bool) as nullable_bool_null,
                    true::Nullable(Bool) as nullable_bool_true,
                    false::Nullable(Bool) as nullable_bool_false
                UNION ALL
                SELECT
                    2 as row_id,
                    false as bool_true,
                    true as bool_false,
                    false::Bool as explicit_bool_true,
                    true::Bool as explicit_bool_false,
                    true::Nullable(Bool) as nullable_bool_null,
                    NULL::Nullable(Bool) as nullable_bool_true,
                    true::Nullable(Bool) as nullable_bool_false
                UNION ALL
                SELECT
                    3 as row_id,
                    1 = 1 as bool_true,  -- expression result
                    1 = 0 as bool_false, -- expression result
                    (1 > 0)::Bool as explicit_bool_true,
                    (1 < 0)::Bool as explicit_bool_false,
                    false::Nullable(Bool) as nullable_bool_null,
                    false::Nullable(Bool) as nullable_bool_true,
                    NULL::Nullable(Bool) as nullable_bool_false
            )
            ORDER BY row_id
        """, "DataFrame")

        # Verify we have 3 rows and 8 columns
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret.columns), 8)

        # Test first row - basic Boolean values
        self.assertTrue(ret.iloc[0]["bool_true"])
        self.assertFalse(ret.iloc[0]["bool_false"])
        self.assertTrue(ret.iloc[0]["explicit_bool_true"])
        self.assertFalse(ret.iloc[0]["explicit_bool_false"])
        self.assertTrue(pd.isna(ret.iloc[0]["nullable_bool_null"]))
        self.assertTrue(ret.iloc[0]["nullable_bool_true"])
        self.assertFalse(ret.iloc[0]["nullable_bool_false"])

        # Test second row - inverted Boolean values
        self.assertFalse(ret.iloc[1]["bool_true"])
        self.assertTrue(ret.iloc[1]["bool_false"])
        self.assertFalse(ret.iloc[1]["explicit_bool_true"])
        self.assertTrue(ret.iloc[1]["explicit_bool_false"])
        self.assertTrue(ret.iloc[1]["nullable_bool_null"])
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_bool_true"]))
        self.assertTrue(ret.iloc[1]["nullable_bool_false"])

        # Test third row - expression results
        self.assertTrue(ret.iloc[2]["bool_true"])   # 1 = 1 is true
        self.assertFalse(ret.iloc[2]["bool_false"])  # 1 = 0 is false
        self.assertTrue(ret.iloc[2]["explicit_bool_true"])   # 1 > 0 is true
        self.assertFalse(ret.iloc[2]["explicit_bool_false"]) # 1 < 0 is false
        self.assertFalse(ret.iloc[2]["nullable_bool_null"])
        self.assertFalse(ret.iloc[2]["nullable_bool_true"])
        self.assertTrue(pd.isna(ret.iloc[2]["nullable_bool_false"]))

        # Test Python types - Bool values should be boolean types (Python bool or numpy bool_)
        for i in range(len(ret)):
            for col in ["bool_true", "bool_false", "explicit_bool_true", "explicit_bool_false"]:
                value = ret.iloc[i][col]
                # Accept both Python bool and numpy bool_ types
                self.assertTrue(isinstance(value, (bool, np.bool_)), f"Row {i}, column {col} should be boolean type, got {type(value)}")

            # Test nullable Bool columns - should be bool/numpy.bool_ or null
            for col in ["nullable_bool_null", "nullable_bool_true", "nullable_bool_false"]:
                if (pd.isna(ret.iloc[i][col])):
                    continue

                value = ret.iloc[i][col]
                self.assertTrue(isinstance(value, (bool, np.bool_)),
                              f"Row {i}, column {col} should be boolean type, got {type(value)}")

        # Verify data types - Bool types should be mapped to bool dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "bool_true": "bool",
            "bool_false": "bool",
            "explicit_bool_true": "bool",
            "explicit_bool_false": "bool",
            "nullable_bool_null": "boolean",
            "nullable_bool_true": "boolean",
            "nullable_bool_false": "boolean"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"Column {col} type mismatch")

    def test_tuple_types(self):
        """Test Tuple types with various element combinations"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    (1, 'hello') as tuple_int_str,
                    (true, false, true) as tuple_bool,
                    (1, 2.5, 'test') as tuple_mixed,
                    tuple(42, 'world', false) as tuple_explicit,
                    (1, (2, 3)) as tuple_nested,
                    ('a', 'b', 'c') as tuple_string,
                    (NULL, 1)::Tuple(Nullable(Int32), Int32) as tuple_nullable,
                    tuple() as tuple_empty
                UNION ALL
                SELECT
                    2 as row_id,
                    (100, 'goodbye') as tuple_int_str,
                    (false, true, false) as tuple_bool,
                    (10, -3.14, 'data') as tuple_mixed,
                    tuple(-5, 'universe', true) as tuple_explicit,
                    (5, (6, 7)) as tuple_nested,
                    ('x', 'y', 'z') as tuple_string,
                    (42, NULL)::Tuple(Int32, Nullable(Int32)) as tuple_nullable,
                    tuple() as tuple_empty
                UNION ALL
                SELECT
                    3 as row_id,
                    (-1, '') as tuple_int_str,
                    (true, false, false) as tuple_bool,
                    (0, 0.0, '') as tuple_mixed,
                    tuple(2147483647, 'edge_case', false) as tuple_explicit,
                    (99, (100, 101)) as tuple_nested,
                    ('🌍', 'Unicode', 'Test') as tuple_string,
                    (NULL, NULL)::Tuple(Nullable(Int32), Nullable(Int32)) as tuple_nullable,
                    tuple() as tuple_empty
            )
            ORDER BY row_id
        """, "DataFrame")

        # Verify we have 3 rows and 9 columns
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret.columns), 9)

        # Test first row - basic tuple values
        tuple_int_str = ret.iloc[0]["tuple_int_str"]
        self.assertIsInstance(tuple_int_str, np.ndarray)
        self.assertEqual(len(tuple_int_str), 2)
        self.assertEqual(tuple_int_str[0], 1)
        self.assertEqual(tuple_int_str[1], 'hello')

        tuple_bool = ret.iloc[0]["tuple_bool"]
        self.assertIsInstance(tuple_bool, np.ndarray)
        self.assertEqual(len(tuple_bool), 3)
        self.assertTrue(tuple_bool[0])
        self.assertFalse(tuple_bool[1])
        self.assertTrue(tuple_bool[2])

        tuple_mixed = ret.iloc[0]["tuple_mixed"]
        self.assertIsInstance(tuple_mixed, np.ndarray)
        self.assertEqual(len(tuple_mixed), 3)
        self.assertEqual(tuple_mixed[0], 1)
        self.assertEqual(tuple_mixed[1], 2.5)
        self.assertEqual(tuple_mixed[2], 'test')

        # Test nested tuples
        tuple_nested = ret.iloc[0]["tuple_nested"]
        self.assertIsInstance(tuple_nested, np.ndarray)
        self.assertEqual(len(tuple_nested), 2)
        self.assertEqual(tuple_nested[0], 1)
        self.assertIsInstance(tuple_nested[1], tuple)
        self.assertEqual(tuple_nested[1][0], 2)
        self.assertEqual(tuple_nested[1][1], 3)

        # Test nullable tuples
        tuple_nullable = ret.iloc[0]["tuple_nullable"]
        self.assertIsInstance(tuple_nullable, np.ndarray)
        self.assertEqual(len(tuple_nullable), 2)
        self.assertTrue(pd.isna(tuple_nullable[0]))  # NULL value
        self.assertEqual(tuple_nullable[1], 1)

        # Test empty tuple
        tuple_empty = ret.iloc[0]["tuple_empty"]
        self.assertIsInstance(tuple_empty, np.ndarray)
        self.assertEqual(len(tuple_empty), 0)

        # Test second row - different values
        tuple_int_str_2 = ret.iloc[1]["tuple_int_str"]
        self.assertEqual(tuple_int_str_2[0], 100)
        self.assertEqual(tuple_int_str_2[1], 'goodbye')

        tuple_nullable_2 = ret.iloc[1]["tuple_nullable"]
        self.assertEqual(tuple_nullable_2[0], 42)
        self.assertTrue(pd.isna(tuple_nullable_2[1]))  # NULL value

        # Test third row - edge cases
        tuple_bool_3 = ret.iloc[2]["tuple_bool"]
        self.assertIsInstance(tuple_bool_3, np.ndarray)
        self.assertEqual(len(tuple_bool_3), 3)
        self.assertTrue(tuple_bool_3[0])   # true
        self.assertFalse(tuple_bool_3[1])  # false
        self.assertFalse(tuple_bool_3[2])  # false

        tuple_nullable_3 = ret.iloc[2]["tuple_nullable"]
        self.assertTrue(pd.isna(tuple_nullable_3[0]))  # Both NULL
        self.assertTrue(pd.isna(tuple_nullable_3[1]))

        # Test string tuple with Unicode
        tuple_string_3 = ret.iloc[2]["tuple_string"]
        self.assertEqual(tuple_string_3[0], '🌍')
        self.assertEqual(tuple_string_3[1], 'Unicode')
        self.assertEqual(tuple_string_3[2], 'Test')

        # Test tuple element types
        for i in range(len(ret)):
            tuple_val = ret.iloc[i]["tuple_int_str"]
            self.assertIsInstance(tuple_val, np.ndarray, f"Row {i} tuple_int_str should be tuple")
            if len(tuple_val) >= 2:
                self.assertIsInstance(tuple_val[0], (int, np.integer), f"Row {i} first element should be integer")
                self.assertIsInstance(tuple_val[1], str, f"Row {i} second element should be string")

        # Verify data types - Tuple types should be mapped to object dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "tuple_int_str": "object",      # Tuple mapped to object dtype
            "tuple_bool": "object",         # Tuple mapped to object dtype
            "tuple_mixed": "object",        # Tuple mapped to object dtype
            "tuple_explicit": "object",     # Tuple mapped to object dtype
            "tuple_nested": "object",       # Nested Tuple mapped to object dtype
            "tuple_string": "object",       # Tuple mapped to object dtype
            "tuple_nullable": "object",     # Tuple with nullable elements mapped to object dtype
            "tuple_empty": "object"         # Empty Tuple mapped to object dtype
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"Column {col} type mismatch")

        # Test named tuples
        named_tuple_ret = self.session.query("""
            SELECT
                tuple(1, 'John', 25) as person_tuple,
                (42, 3.14159, 'pi') as unnamed_tuple
        """, "DataFrame")

        person_tuple = named_tuple_ret.iloc[0]["person_tuple"]
        self.assertIsInstance(person_tuple, np.ndarray)
        self.assertEqual(len(person_tuple), 3)
        self.assertEqual(person_tuple[0], 1)
        self.assertEqual(person_tuple[1], 'John')
        self.assertEqual(person_tuple[2], 25)

        unnamed_tuple = named_tuple_ret.iloc[0]["unnamed_tuple"]
        self.assertIsInstance(unnamed_tuple, np.ndarray)
        self.assertEqual(len(unnamed_tuple), 3)
        self.assertEqual(unnamed_tuple[0], 42)
        self.assertAlmostEqual(unnamed_tuple[1], 3.14159, places=5)
        self.assertEqual(unnamed_tuple[2], 'pi')

    def test_array_types(self):
        """Test Array types with various element types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    [1, 2, 3, 4, 5] as array_int32,
                    [1, 2, 3, 4, 5]::Array(UInt64) as array_uint64,
                    [1.1, 2.2, 3.3, 4.4, 5.5] as array_float64,
                    ['hello', 'world', 'clickhouse', 'array'] as array_string,
                    [true, false, true, false] as array_bool,
                    [toDate('2023-01-01'), toDate('2023-02-01'), toDate('2023-03-01')] as array_date,
                    [toDateTime('2023-01-01 10:00:00', 'Asia/Shanghai'), toDateTime('2023-01-01 11:00:00', 'Asia/Shanghai')] as array_datetime,
                    [[1, 2], [3, 4], [5, 6]] as array_nested_int,
                    [[100, 200], [300, 400], [500, 600]]::Array(Array(UInt32)) as array_nested_uint32,
                    [['a', 'b'], ['c', 'd']] as array_nested_string,
                    [] as array_empty_int,
                    ['']::Array(String) as array_empty_string_element,
                    [NULL, 1, NULL, 3]::Array(Nullable(Int32)) as array_nullable_int,
                    [NULL, 'test', NULL]::Array(Nullable(String)) as array_nullable_string
                UNION ALL
                SELECT
                    2 as row_id,
                    [10, 20, 30] as array_int32,
                    [100, 200, 300]::Array(UInt64) as array_uint64,
                    [10.5, 20.5] as array_float64,
                    ['test', 'array', 'data'] as array_string,
                    [false, false, true] as array_bool,
                    [toDate('2024-01-01'), toDate('2024-12-31')] as array_date,
                    [toDateTime('2024-06-15 14:30:00', 'Asia/Shanghai')] as array_datetime,
                    [[7, 8, 9], [10]] as array_nested_int,
                    [[700, 800], [900]]::Array(Array(UInt32)) as array_nested_uint32,
                    [['x'], ['y', 'z', 'w']] as array_nested_string,
                    [42] as array_empty_int,
                    ['single'] as array_empty_string_element,
                    [1, 2, 3]::Array(Nullable(Int32)) as array_nullable_int,
                    ['a', 'b']::Array(Nullable(String)) as array_nullable_string
                UNION ALL
                SELECT
                    3 as row_id,
                    [-1, 0, 1, 2147483647, -2147483648] as array_int32,
                    [0, 18446744073709551615]::Array(UInt64) as array_uint64,
                    [0.0, -1.5, 1.0/0.0, -1.0/0.0, 0.0/0.0] as array_float64,
                    ['Unicode: 🌍', 'Special: \t\n"''', ''] as array_string,
                    [true] as array_bool,
                    [toDate('1970-01-01'), toDate('2149-06-06')] as array_date,
                    [toDateTime('1970-01-02 00:00:00', 'Asia/Shanghai'), toDateTime('2106-02-07 06:28:15', 'Asia/Shanghai')] as array_datetime,
                    [[], [1], [2, 3, 4, 5]] as array_nested_int,
                    [[], [1000], [2000, 3000, 4000]]::Array(Array(UInt32)) as array_nested_uint32,
                    [[], ['single'], ['a', 'b', 'c']] as array_nested_string,
                    []::Array(Int32) as array_empty_int,
                    []::Array(String) as array_empty_string_element,
                    [NULL]::Array(Nullable(Int32)) as array_nullable_int,
                    [NULL, NULL]::Array(Nullable(String)) as array_nullable_string
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - basic arrays (converted to numpy arrays)
        np.testing.assert_array_equal(ret.iloc[0]["array_int32"], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(ret.iloc[0]["array_uint64"], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(ret.iloc[0]["array_float64"], [1.1, 2.2, 3.3, 4.4, 5.5])
        np.testing.assert_array_equal(ret.iloc[0]["array_string"], ['hello', 'world', 'clickhouse', 'array'])
        np.testing.assert_array_equal(ret.iloc[0]["array_bool"], [True, False, True, False])

        # Test date arrays (converted to numpy array of pandas timestamps)
        date_array = ret.iloc[0]["array_date"]
        self.assertIsInstance(date_array, np.ndarray)
        self.assertEqual(len(date_array), 3)
        self.assertEqual(date_array[0], pd.Timestamp('2023-01-01'))
        self.assertEqual(date_array[1], pd.Timestamp('2023-02-01'))
        self.assertEqual(date_array[2], pd.Timestamp('2023-03-01'))

        # Test datetime arrays (converted to numpy array of numpy.datetime64 in UTC)
        datetime_array = ret.iloc[0]["array_datetime"]
        self.assertIsInstance(datetime_array, np.ndarray)
        self.assertEqual(len(datetime_array), 2)
        # ClickHouse converts Asia/Shanghai time to UTC: 10:00:00 +0800 -> 02:00:00 UTC
        self.assertEqual(datetime_array[0], np.datetime64('2023-01-01T02:00:00'))
        self.assertEqual(datetime_array[1], np.datetime64('2023-01-01T03:00:00'))

        # Test nested arrays (numpy arrays containing numpy arrays)
        nested_int = ret.iloc[0]["array_nested_int"]
        self.assertIsInstance(nested_int, np.ndarray)
        self.assertEqual(len(nested_int), 3)
        np.testing.assert_array_equal(nested_int[0], [1, 2])
        np.testing.assert_array_equal(nested_int[1], [3, 4])
        np.testing.assert_array_equal(nested_int[2], [5, 6])

        nested_uint32 = ret.iloc[0]["array_nested_uint32"]
        self.assertIsInstance(nested_uint32, np.ndarray)
        self.assertEqual(len(nested_uint32), 3)
        np.testing.assert_array_equal(nested_uint32[0], [100, 200])
        np.testing.assert_array_equal(nested_uint32[1], [300, 400])
        np.testing.assert_array_equal(nested_uint32[2], [500, 600])

        nested_string = ret.iloc[0]["array_nested_string"]
        self.assertIsInstance(nested_string, np.ndarray)
        self.assertEqual(len(nested_string), 2)
        np.testing.assert_array_equal(nested_string[0], ['a', 'b'])
        np.testing.assert_array_equal(nested_string[1], ['c', 'd'])

        # Test empty arrays and arrays with empty string elements
        empty_int_array = ret.iloc[0]["array_empty_int"]
        self.assertIsInstance(empty_int_array, np.ndarray)
        self.assertEqual(len(empty_int_array), 0)

        string_element_array = ret.iloc[0]["array_empty_string_element"]
        self.assertIsInstance(string_element_array, np.ndarray)
        np.testing.assert_array_equal(string_element_array, [''])

        # Test nullable arrays (numpy arrays with None values)
        nullable_int = ret.iloc[0]["array_nullable_int"]
        self.assertIsInstance(nullable_int, np.ndarray)
        self.assertEqual(len(nullable_int), 4)
        self.assertTrue(nullable_int.mask[0])
        self.assertEqual(nullable_int[1], 1)
        self.assertTrue(nullable_int.mask[2])
        self.assertEqual(nullable_int[3], 3)

        nullable_string = ret.iloc[0]["array_nullable_string"]
        self.assertIsInstance(nullable_string, np.ndarray)
        self.assertEqual(len(nullable_string), 3)
        # self.assertTrue(nullable_string.mask[0])
        self.assertIsNone(nullable_string[0])
        self.assertEqual(nullable_string[1], 'test')
        # self.assertTrue(nullable_string.mask[2])
        self.assertIsNone(nullable_string[2])

        # Test second row - different arrays (numpy arrays)
        np.testing.assert_array_equal(ret.iloc[1]["array_int32"], [10, 20, 30])
        np.testing.assert_array_equal(ret.iloc[1]["array_uint64"], [100, 200, 300])
        np.testing.assert_array_equal(ret.iloc[1]["array_float64"], [10.5, 20.5])
        np.testing.assert_array_equal(ret.iloc[1]["array_string"], ['test', 'array', 'data'])
        np.testing.assert_array_equal(ret.iloc[1]["array_bool"], [False, False, True])

        # Test second row datetime array: 14:30:00 +0800 -> 06:30:00 UTC
        datetime_array_2 = ret.iloc[1]["array_datetime"]
        self.assertEqual(len(datetime_array_2), 1)
        self.assertEqual(datetime_array_2[0], np.datetime64('2024-06-15T06:30:00'))

        # Test third row - edge cases (numpy arrays)
        np.testing.assert_array_equal(ret.iloc[2]["array_int32"], [-1, 0, 1, 2147483647, -2147483648])
        np.testing.assert_array_equal(ret.iloc[2]["array_uint64"], [0, 18446744073709551615])

        # Test third row datetime array: Asia/Shanghai times converted to UTC
        datetime_array_3 = ret.iloc[2]["array_datetime"]
        self.assertEqual(len(datetime_array_3), 2)
        # 1970-01-02 00:00:00 +0800 -> 1970-01-01 16:00:00 UTC
        self.assertEqual(datetime_array_3[0], np.datetime64('1970-01-01T16:00:00'))
        # 2106-02-07 06:28:15 +0800 -> 2106-02-06 22:28:15 UTC
        self.assertEqual(datetime_array_3[1], np.datetime64('2106-02-06T22:28:15'))

        # Test float special values in array
        float_array = ret.iloc[2]["array_float64"]
        self.assertEqual(float_array[0], 0.0)
        self.assertEqual(float_array[1], -1.5)
        self.assertTrue(math.isinf(float_array[2]))  # positive infinity
        self.assertTrue(math.isinf(float_array[3]))  # negative infinity
        self.assertTrue(math.isnan(float_array[4]))  # NaN

        # Test string array with special characters (numpy array)
        string_array = ret.iloc[2]["array_string"]
        self.assertIsInstance(string_array, np.ndarray)
        self.assertEqual(string_array[0], 'Unicode: 🌍')
        self.assertEqual(string_array[1], "Special: \t\n\"'")  # ClickHouse interprets escape sequences
        self.assertEqual(string_array[2], '')

        # Test nested arrays with empty elements (numpy arrays)
        nested_int_3 = ret.iloc[2]["array_nested_int"]
        self.assertIsInstance(nested_int_3, np.ndarray)
        self.assertEqual(len(nested_int_3[0]), 0)  # empty array
        np.testing.assert_array_equal(nested_int_3[1], [1])  # single element
        np.testing.assert_array_equal(nested_int_3[2], [2, 3, 4, 5])  # multiple elements

        nested_uint32_3 = ret.iloc[2]["array_nested_uint32"]
        self.assertIsInstance(nested_uint32_3, np.ndarray)
        self.assertEqual(len(nested_uint32_3[0]), 0)  # empty array
        np.testing.assert_array_equal(nested_uint32_3[1], [1000])  # single element
        np.testing.assert_array_equal(nested_uint32_3[2], [2000, 3000, 4000])  # multiple elements

        # Test empty typed arrays
        self.assertEqual(len(ret.iloc[2]["array_empty_int"]), 0)
        self.assertEqual(len(ret.iloc[2]["array_empty_string_element"]), 0)

        # Test nullable arrays with only NULL values
        self.assertEqual(len(ret.iloc[2]["array_nullable_int"]), 1)
        self.assertTrue(ret.iloc[2]["array_nullable_int"].mask[0])

        self.assertEqual(len(ret.iloc[2]["array_nullable_string"]), 2)
        # self.assertTrue(ret.iloc[2]["array_nullable_string"].mask[0])
        # self.assertTrue(ret.iloc[2]["array_nullable_string"].mask[1])
        self.assertIsNone(ret.iloc[2]["array_nullable_string"][0])
        self.assertIsNone(ret.iloc[2]["array_nullable_string"][1])

        # Precise data type validation - Arrays should be mapped to object dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "array_int32": "object",           # Array(Int32) mapped to object dtype
            "array_uint64": "object",          # Array(UInt64) mapped to object dtype
            "array_float64": "object",         # Array(Float64) mapped to object dtype
            "array_string": "object",          # Array(String) mapped to object dtype
            "array_bool": "object",            # Array(Bool) mapped to object dtype
            "array_date": "object",            # Array(Date) mapped to object dtype
            "array_datetime": "object",        # Array(DateTime) mapped to object dtype
            "array_nested_int": "object",      # Array(Array(Int32)) mapped to object dtype
            "array_nested_uint32": "object",   # Array(Array(UInt32)) mapped to object dtype
            "array_nested_string": "object",   # Array(Array(String)) mapped to object dtype
            "array_empty_int": "object",       # Empty Array(Int32) mapped to object dtype
            "array_empty_string_element": "object",  # Array(String) with empty string mapped to object dtype
            "array_nullable_int": "object",    # Array(Nullable(Int32)) mapped to object dtype
            "array_nullable_string": "object"  # Array(Nullable(String)) mapped to object dtype
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

        # Verify all array columns contain numpy arrays
        array_columns = [col for col in ret.columns if col.startswith('array_')]
        for col in array_columns:
            for i in range(len(ret)):
                array_value = ret.iloc[i][col]
                # Check if it's a numpy array
                self.assertIsInstance(array_value, np.ndarray, f"Row {i}, column {col} should be numpy array")
                # Verify numpy array properties
                self.assertTrue(hasattr(array_value, '__len__'), f"Row {i}, column {col} should have length")
                self.assertTrue(hasattr(array_value, '__getitem__'), f"Row {i}, column {col} should be indexable")

    def test_map_types(self):
        """Test Map(K,V) types where K and V can be any types"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    -- Basic primitive type combinations with multiple key-value pairs
                    map('str_key1', 42, 'str_key2', 100, 'str_key3', -50)::Map(String, Int32) as map_str_int,
                    map(100, 'int_key1', 200, 'int_key2', -10, 'negative_key')::Map(Int32, String) as map_int_str,
                    map(true, 'bool_true', false, 'bool_false')::Map(Bool, String) as map_bool_str,
                    map('pi', 3.14, 'e', 2.718, 'phi', 1.618)::Map(String, Float64) as map_str_float,

                    -- DateTime and Date types as values with multiple pairs
                    map('created', toTimeZone('2023-01-15 10:30:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'), 'updated', toTimeZone('2024-03-20 14:45:30'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'), 'archived', toTimeZone('2024-12-01 09:15:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'))::Map(String, DateTime('Asia/Shanghai')) as map_str_datetime,
                    map('birth_date', '1990-05-15'::Date, 'start_date', '2020-01-01'::Date, 'end_date', '2025-12-31'::Date)::Map(String, Date) as map_str_date,
                    map('precise_time1', toTimeZone('2023-01-15 10:30:00.123456'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'), 'precise_time2', toTimeZone('2024-03-20 14:45:30.987654'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'), 'precise_time3', toTimeZone('2024-12-01 09:15:00.555555'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'))::Map(String, DateTime64(6, 'Asia/Shanghai')) as map_str_datetime64,
                    map('event_id', '1001', 'timestamp', '2023-06-10 16:20:45', 'event_id2', '1002', 'timestamp2', '2023-06-11 17:30:15')::Map(String, String) as map_mixed_datetime,

                    -- Decimal types as values with multiple pairs
                    map('price1', 99.99::Decimal(10,2), 'price2', 149.50::Decimal(10,2), 'discount', 15.75::Decimal(10,2))::Map(String, Decimal(10,2)) as map_str_decimal,

                    -- Array as Key and Value types
                    map([1,2], 'array_key')::Map(Array(Int32), String) as map_array_str,
                    map('array_val1', [10,20,30], 'array_val2', [40,50], 'empty_array', [])::Map(String, Array(Int32)) as map_str_array,

                    -- Tuple as Key and Value types
                    map((1,'tuple'), 'tuple_key')::Map(Tuple(Int32, String), String) as map_tuple_str,
                    map('tuple_val1', (100, 'data1'), 'tuple_val2', (200, 'data2'))::Map(String, Tuple(Int32, String)) as map_str_tuple,

                    -- Nested Map as Value with multiple entries
                    map('config1', map('timeout', 30, 'retries', 3), 'config2', map('timeout', 60, 'retries', 5))::Map(String, Map(String, Int32)) as map_nested,

                    -- Nullable types with multiple pairs
                    map('nullable1', NULL, 'nullable2', 'has_value', 'nullable3', NULL)::Map(String, Nullable(String)) as map_nullable
                UNION ALL
                SELECT
                    2 as row_id,
                    -- Different values with multiple pairs
                    map('key_a', 999, 'key_b', 888, 'key_c', 777)::Map(String, Int32) as map_str_int,
                    map(300, 'triple', 400, 'quad', 500, 'penta')::Map(Int32, String) as map_int_str,
                    map(false, 'false_key', true, 'true_key')::Map(Bool, String) as map_bool_str,
                    map('sqrt2', 1.414, 'sqrt3', 1.732, 'sqrt5', 2.236)::Map(String, Float64) as map_str_float,

                    -- Different datetime values
                    map('morning', toTimeZone('2024-01-01 08:00:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'), 'noon', toTimeZone('2024-01-01 12:00:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'), 'evening', toTimeZone('2024-01-01 18:00:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'))::Map(String, DateTime('Asia/Shanghai')) as map_str_datetime,
                    map('monday', '2024-01-01'::Date, 'friday', '2024-01-05'::Date, 'sunday', '2024-01-07'::Date)::Map(String, Date) as map_str_date,
                    map('morning_precise', toTimeZone('2024-01-01 08:00:00.111111'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'), 'noon_precise', toTimeZone('2024-01-01 12:00:00.222222'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'), 'evening_precise', toTimeZone('2024-01-01 18:00:00.333333'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'))::Map(String, DateTime64(6, 'Asia/Shanghai')) as map_str_datetime64,
                    map('log_entry1', 'ERROR: 2024-02-15 10:30:00', 'log_entry2', 'INFO: 2024-02-15 10:31:00')::Map(String, String) as map_mixed_datetime,

                    -- Different decimal values
                    map('tax', 8.25::Decimal(10,2), 'shipping', 12.99::Decimal(10,2), 'total', 199.99::Decimal(10,2))::Map(String, Decimal(10,2)) as map_str_decimal,

                    map([5,6,7], 'different_array')::Map(Array(Int32), String) as map_array_str,
                    map('values1', [100,200], 'values2', [300,400,500])::Map(String, Array(Int32)) as map_str_array,

                    map((2,'another'), 'another_tuple')::Map(Tuple(Int32, String), String) as map_tuple_str,
                    map('tuple_a', (200, 'test_a'), 'tuple_b', (300, 'test_b'))::Map(String, Tuple(Int32, String)) as map_str_tuple,

                    map('db_config', map('host', 1, 'port', 5432), 'cache_config', map('ttl', 300, 'size', 1000))::Map(String, Map(String, Int32)) as map_nested,

                    map('active', 'yes', 'inactive', NULL, 'pending', 'maybe')::Map(String, Nullable(String)) as map_nullable
                UNION ALL
                SELECT
                    3 as row_id,
                    -- Edge cases and special values with multiple pairs
                    map('min_int', -2147483648, 'max_int', 2147483647, 'zero', 0)::Map(String, Int32) as map_str_int,
                    map(-50, 'negative_int', 0, 'zero_int', 1000000, 'million')::Map(Int32, String) as map_int_str,
                    map(true, 'always_true', false, 'always_false')::Map(Bool, String) as map_bool_str,
                    map('inf', 1.0/0.0, 'neg_inf', -1.0/0.0, 'nan', 0.0/0.0)::Map(String, Float64) as map_str_float,

                    -- Extreme datetime values
                    map('epoch', toTimeZone('1970-01-01 00:00:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'), 'y2k', toTimeZone('2000-01-01 00:00:00'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'), 'future', toTimeZone('2099-12-31 23:59:59'::DateTime('Asia/Shanghai'), 'Asia/Shanghai'))::Map(String, DateTime('Asia/Shanghai')) as map_str_datetime,
                    map('past', '1900-01-01'::Date, 'present', today(), 'future', '2100-01-01'::Date)::Map(String, Date) as map_str_date,
                    map('epoch_precise', toTimeZone('1970-01-01 08:00:00.000001'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'), 'y2k_precise', toTimeZone('2000-01-01 00:00:00.999999'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'), 'future_precise', toTimeZone('2099-12-31 23:59:59.123456'::DateTime64(6, 'Asia/Shanghai'), 'Asia/Shanghai'))::Map(String, DateTime64(6, 'Asia/Shanghai')) as map_str_datetime64,
                    map('debug1', 'TRACE: 1970-01-01 00:00:01', 'debug2', 'DEBUG: 2038-01-19 03:14:07')::Map(String, String) as map_mixed_datetime,

                    -- Extreme decimal values
                    map('min_decimal', 0.01::Decimal(10,2), 'max_decimal', 99999999.99::Decimal(10,2), 'zero_decimal', 0.00::Decimal(10,2))::Map(String, Decimal(10,2)) as map_str_decimal,

                    map([], 'empty_array')::Map(Array(Int32), String) as map_array_str,
                    map('empty_val', [], 'single_val', [42], 'multi_val', [1,2,3,4,5])::Map(String, Array(Int32)) as map_str_array,

                    map((0,'zero'), 'zero_tuple')::Map(Tuple(Int32, String), String) as map_tuple_str,
                    map('empty_like', (0, ''), 'full_like', (999, 'full_string'))::Map(String, Tuple(Int32, String)) as map_str_tuple,

                    map('triple_nested', map('level2', 999))::Map(String, Map(String, Int32)) as map_nested,

                    map('null_again', NULL)::Map(String, Nullable(String)) as map_nullable
            )
            ORDER BY row_id
        """, "DataFrame")

        # Verify we have 3 rows and 16 columns
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(ret.columns), 16)

        # Test Row 1 - Basic primitive type combinations with multiple key-value pairs
        # Map(String, Int32) with multiple pairs
        map_str_int = ret.iloc[0]["map_str_int"]
        self.assertIsInstance(map_str_int, dict)
        self.assertEqual(len(map_str_int), 3)  # Should have 3 key-value pairs
        self.assertEqual(map_str_int['str_key1'], 42)
        self.assertEqual(map_str_int['str_key2'], 100)
        self.assertEqual(map_str_int['str_key3'], -50)

        # Map(Int32, String) with multiple pairs
        map_int_str = ret.iloc[0]["map_int_str"]
        self.assertIsInstance(map_int_str, dict)
        self.assertEqual(len(map_int_str), 3)  # Should have 3 key-value pairs
        self.assertEqual(map_int_str[100], 'int_key1')
        self.assertEqual(map_int_str[200], 'int_key2')
        self.assertEqual(map_int_str[-10], 'negative_key')

        # Map(Bool, String) with both true and false keys
        map_bool_str = ret.iloc[0]["map_bool_str"]
        self.assertIsInstance(map_bool_str, dict)
        self.assertEqual(len(map_bool_str), 2)  # Should have 2 key-value pairs
        self.assertEqual(map_bool_str[True], 'bool_true')
        self.assertEqual(map_bool_str[False], 'bool_false')

        # Map(String, Float64) with multiple mathematical constants
        map_str_float = ret.iloc[0]["map_str_float"]
        self.assertIsInstance(map_str_float, dict)
        self.assertEqual(len(map_str_float), 3)  # Should have 3 key-value pairs
        self.assertAlmostEqual(map_str_float['pi'], 3.14, places=2)
        self.assertAlmostEqual(map_str_float['e'], 2.718, places=3)
        self.assertAlmostEqual(map_str_float['phi'], 1.618, places=3)

        # Test DateTime and Date types as values
        # Map(String, DateTime) with multiple datetime values
        map_str_datetime = ret.iloc[0]["map_str_datetime"]
        self.assertIsInstance(map_str_datetime, dict)
        self.assertEqual(len(map_str_datetime), 3)  # Should have 3 key-value pairs
        # Verify datetime values (converted to python datetime objects with Shanghai timezone)
        self.assertIsInstance(map_str_datetime['created'], datetime)
        self.assertEqual(map_str_datetime['created'], datetime(2023, 1, 15, 10, 30, 0, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime['updated'], datetime)
        self.assertEqual(map_str_datetime['updated'], datetime(2024, 3, 20, 14, 45, 30, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime['archived'], datetime)
        self.assertEqual(map_str_datetime['archived'], datetime(2024, 12, 1, 9, 15, 0, tzinfo=self.shanghai_tz))

        # Map(String, Date) with multiple date values
        map_str_date = ret.iloc[0]["map_str_date"]
        self.assertIsInstance(map_str_date, dict)
        self.assertEqual(len(map_str_date), 3)  # Should have 3 key-value pairs
        # Verify date values (converted to python date objects)
        self.assertIsInstance(map_str_date['birth_date'], date)
        self.assertEqual(map_str_date['birth_date'], date(1990, 5, 15))
        self.assertIsInstance(map_str_date['start_date'], date)
        self.assertEqual(map_str_date['start_date'], date(2020, 1, 1))
        self.assertIsInstance(map_str_date['end_date'], date)
        self.assertEqual(map_str_date['end_date'], date(2025, 12, 31))

        # Test DateTime64 with microsecond precision
        # Map(String, DateTime64) with multiple datetime64 values
        map_str_datetime64 = ret.iloc[0]["map_str_datetime64"]
        self.assertIsInstance(map_str_datetime64, dict)
        self.assertEqual(len(map_str_datetime64), 3)  # Should have 3 key-value pairs
        # Verify datetime64 values (converted to python datetime objects with Shanghai timezone and microseconds)
        self.assertIsInstance(map_str_datetime64['precise_time1'], datetime)
        self.assertEqual(map_str_datetime64['precise_time1'], datetime(2023, 1, 15, 10, 30, 0, 123456, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime64['precise_time2'], datetime)
        self.assertEqual(map_str_datetime64['precise_time2'], datetime(2024, 3, 20, 14, 45, 30, 987654, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime64['precise_time3'], datetime)
        self.assertEqual(map_str_datetime64['precise_time3'], datetime(2024, 12, 1, 9, 15, 0, 555555, tzinfo=self.shanghai_tz))

        # Map(String, String) with mixed datetime strings
        map_mixed_datetime = ret.iloc[0]["map_mixed_datetime"]
        self.assertIsInstance(map_mixed_datetime, dict)
        self.assertEqual(len(map_mixed_datetime), 4)  # Should have 4 key-value pairs
        self.assertEqual(map_mixed_datetime['event_id'], '1001')
        self.assertIn('2023-06-10 16:20:45', map_mixed_datetime['timestamp'])

        # Map(String, Decimal) with multiple decimal values
        map_str_decimal = ret.iloc[0]["map_str_decimal"]
        self.assertIsInstance(map_str_decimal, dict)
        self.assertEqual(len(map_str_decimal), 3)  # Should have 3 key-value pairs
        # Verify decimal values (should be converted to float or Decimal)
        self.assertAlmostEqual(float(map_str_decimal['price1']), 99.99, places=2)
        self.assertAlmostEqual(float(map_str_decimal['price2']), 149.50, places=2)
        self.assertAlmostEqual(float(map_str_decimal['discount']), 15.75, places=2)

        # Test Array as Key/Value types
        # Map(Array(Int32), String) - Array as Key (non-hashable, uses keys/values structure)
        map_array_str = ret.iloc[0]["map_array_str"]
        self.assertIsInstance(map_array_str, dict)
        # Non-hashable keys create {keys: [...], values: [...]} structure
        self.assertIn('keys', map_array_str)
        self.assertIn('values', map_array_str)
        self.assertEqual(len(map_array_str['keys']), 1)
        self.assertEqual(len(map_array_str['values']), 1)
        # Verify the array key and its corresponding value
        array_key = map_array_str['keys'][0]
        self.assertIsInstance(array_key, list)
        np.testing.assert_array_equal(array_key, [1, 2])
        self.assertEqual(map_array_str['values'][0], 'array_key')

        # Map(String, Array(Int32)) - Array as Value with multiple pairs (hashable key, normal dict)
        map_str_array = ret.iloc[0]["map_str_array"]
        self.assertIsInstance(map_str_array, dict)
        self.assertEqual(len(map_str_array), 3)  # Should have 3 key-value pairs
        # Verify multiple array values
        array_value1 = map_str_array['array_val1']
        self.assertIsInstance(array_value1, list)
        np.testing.assert_array_equal(array_value1, [10, 20, 30])
        array_value2 = map_str_array['array_val2']
        self.assertIsInstance(array_value2, list)
        np.testing.assert_array_equal(array_value2, [40, 50])
        empty_array = map_str_array['empty_array']
        self.assertIsInstance(empty_array, list)
        self.assertEqual(len(empty_array), 0)

        # Test Tuple as Key/Value types
        # Map(Tuple(Int32, String), String) - Tuple as Key (non-hashable, uses keys/values structure)
        map_tuple_str = ret.iloc[0]["map_tuple_str"]
        self.assertIsInstance(map_tuple_str, dict)
        # Non-hashable keys create {keys: [...], values: [...]} structure
        self.assertIn('keys', map_tuple_str)
        self.assertIn('values', map_tuple_str)
        self.assertEqual(len(map_tuple_str['keys']), 1)
        self.assertEqual(len(map_tuple_str['values']), 1)
        # Verify the tuple key and its corresponding value
        tuple_key = map_tuple_str['keys'][0]
        self.assertIsInstance(tuple_key, tuple)
        self.assertEqual(map_tuple_str['values'][0], 'tuple_key')

        # Map(String, Tuple(Int32, String)) - Tuple as Value with multiple pairs (hashable key, normal dict)
        map_str_tuple = ret.iloc[0]["map_str_tuple"]
        self.assertIsInstance(map_str_tuple, dict)
        self.assertEqual(len(map_str_tuple), 2)  # Should have 2 key-value pairs
        # Verify multiple tuple values
        tuple_value1 = map_str_tuple['tuple_val1']
        self.assertIsInstance(tuple_value1, tuple)
        self.assertEqual(tuple_value1, (100, 'data1'))
        tuple_value2 = map_str_tuple['tuple_val2']
        self.assertIsInstance(tuple_value2, tuple)
        self.assertEqual(tuple_value2, (200, 'data2'))

        # Test Nested Map with multiple entries - Map(String, Map(String, Int32))
        map_nested = ret.iloc[0]["map_nested"]
        self.assertIsInstance(map_nested, dict)
        self.assertEqual(len(map_nested), 2)  # Should have 2 key-value pairs
        # Verify first nested map
        inner_map1 = map_nested['config1']
        self.assertIsInstance(inner_map1, dict)
        self.assertEqual(inner_map1['timeout'], 30)
        self.assertEqual(inner_map1['retries'], 3)
        # Verify second nested map
        inner_map2 = map_nested['config2']
        self.assertIsInstance(inner_map2, dict)
        self.assertEqual(inner_map2['timeout'], 60)
        self.assertEqual(inner_map2['retries'], 5)

        # Test Nullable Value with multiple pairs - Map(String, Nullable(String))
        map_nullable = ret.iloc[0]["map_nullable"]
        self.assertIsInstance(map_nullable, dict)
        self.assertEqual(len(map_nullable), 3)  # Should have 3 key-value pairs
        # Verify mixed null and non-null values
        self.assertTrue(pd.isna(map_nullable['nullable1']))
        self.assertEqual(map_nullable['nullable2'], 'has_value')
        self.assertTrue(pd.isna(map_nullable['nullable3']))

        # Test Row 2 - Different values with multiple pairs
        # Test Map(String, Int32) with different data (hashable key -> normal dict)
        map_str_int_2 = ret.iloc[1]["map_str_int"]
        self.assertIsInstance(map_str_int_2, dict)
        self.assertEqual(len(map_str_int_2), 3)  # Should have 3 key-value pairs
        self.assertEqual(map_str_int_2['key_a'], 999)
        self.assertEqual(map_str_int_2['key_b'], 888)
        self.assertEqual(map_str_int_2['key_c'], 777)

        # Test Map(Bool, String) with both keys (hashable key -> normal dict)
        map_bool_str_2 = ret.iloc[1]["map_bool_str"]
        self.assertIsInstance(map_bool_str_2, dict)
        self.assertEqual(len(map_bool_str_2), 2)  # Should have 2 key-value pairs
        self.assertEqual(map_bool_str_2[False], 'false_key')
        self.assertEqual(map_bool_str_2[True], 'true_key')

        # Test DateTime values in row 2
        map_str_datetime_2 = ret.iloc[1]["map_str_datetime"]
        self.assertIsInstance(map_str_datetime_2, dict)
        self.assertEqual(len(map_str_datetime_2), 3)  # Should have 3 key-value pairs
        self.assertIsInstance(map_str_datetime_2['morning'], datetime)
        self.assertEqual(map_str_datetime_2['morning'], datetime(2024, 1, 1, 8, 0, 0, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime_2['noon'], datetime)
        self.assertEqual(map_str_datetime_2['noon'], datetime(2024, 1, 1, 12, 0, 0, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime_2['evening'], datetime)
        self.assertEqual(map_str_datetime_2['evening'], datetime(2024, 1, 1, 18, 0, 0, tzinfo=self.shanghai_tz))

        # Test Date values in row 2
        map_str_date_2 = ret.iloc[1]["map_str_date"]
        self.assertIsInstance(map_str_date_2, dict)
        self.assertEqual(len(map_str_date_2), 3)  # Should have 3 key-value pairs
        self.assertIsInstance(map_str_date_2['monday'], date)
        self.assertEqual(map_str_date_2['monday'], date(2024, 1, 1))
        self.assertIsInstance(map_str_date_2['friday'], date)
        self.assertEqual(map_str_date_2['friday'], date(2024, 1, 5))
        self.assertIsInstance(map_str_date_2['sunday'], date)
        self.assertEqual(map_str_date_2['sunday'], date(2024, 1, 7))

        # Test DateTime64 values in row 2
        map_str_datetime64_2 = ret.iloc[1]["map_str_datetime64"]
        self.assertIsInstance(map_str_datetime64_2, dict)
        self.assertEqual(len(map_str_datetime64_2), 3)  # Should have 3 key-value pairs
        self.assertIsInstance(map_str_datetime64_2['morning_precise'], datetime)
        self.assertEqual(map_str_datetime64_2['morning_precise'], datetime(2024, 1, 1, 8, 0, 0, 111111, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime64_2['noon_precise'], datetime)
        self.assertEqual(map_str_datetime64_2['noon_precise'], datetime(2024, 1, 1, 12, 0, 0, 222222, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime64_2['evening_precise'], datetime)
        self.assertEqual(map_str_datetime64_2['evening_precise'], datetime(2024, 1, 1, 18, 0, 0, 333333, tzinfo=self.shanghai_tz))

        # Test Decimal values in row 2
        map_str_decimal_2 = ret.iloc[1]["map_str_decimal"]
        self.assertIsInstance(map_str_decimal_2, dict)
        self.assertEqual(len(map_str_decimal_2), 3)  # Should have 3 key-value pairs
        self.assertAlmostEqual(float(map_str_decimal_2['tax']), 8.25, places=2)
        self.assertAlmostEqual(float(map_str_decimal_2['shipping']), 12.99, places=2)
        self.assertAlmostEqual(float(map_str_decimal_2['total']), 199.99, places=2)

        # Test Map with nullable that has mixed values (hashable key -> normal dict)
        map_nullable_2 = ret.iloc[1]["map_nullable"]
        self.assertIsInstance(map_nullable_2, dict)
        self.assertEqual(len(map_nullable_2), 3)  # Should have 3 key-value pairs
        self.assertEqual(map_nullable_2['active'], 'yes')
        self.assertTrue(pd.isna(map_nullable_2['inactive']))
        self.assertEqual(map_nullable_2['pending'], 'maybe')

        # Test Array as key in row 2 (non-hashable -> keys/values structure)
        map_array_str_2 = ret.iloc[1]["map_array_str"]
        self.assertIn('keys', map_array_str_2)
        self.assertIn('values', map_array_str_2)
        array_key_2 = map_array_str_2['keys'][0]
        np.testing.assert_array_equal(array_key_2, [5, 6, 7])
        self.assertEqual(map_array_str_2['values'][0], 'different_array')

        # Test Array values in row 2 with multiple pairs
        map_str_array_2 = ret.iloc[1]["map_str_array"]
        self.assertIsInstance(map_str_array_2, dict)
        self.assertEqual(len(map_str_array_2), 2)  # Should have 2 key-value pairs
        np.testing.assert_array_equal(map_str_array_2['values1'], [100, 200])
        np.testing.assert_array_equal(map_str_array_2['values2'], [300, 400, 500])

        # Test Row 3 - Edge cases and special values with multiple pairs
        # Test extreme integer values (hashable keys -> normal dict)
        map_str_int_3 = ret.iloc[2]["map_str_int"]
        self.assertIsInstance(map_str_int_3, dict)
        self.assertEqual(len(map_str_int_3), 3)  # Should have 3 key-value pairs
        self.assertEqual(map_str_int_3['min_int'], -2147483648)
        self.assertEqual(map_str_int_3['max_int'], 2147483647)
        self.assertEqual(map_str_int_3['zero'], 0)

        map_int_str_3 = ret.iloc[2]["map_int_str"]
        self.assertIsInstance(map_int_str_3, dict)
        self.assertEqual(len(map_int_str_3), 3)  # Should have 3 key-value pairs
        self.assertEqual(map_int_str_3[-50], 'negative_int')
        self.assertEqual(map_int_str_3[0], 'zero_int')
        self.assertEqual(map_int_str_3[1000000], 'million')

        # Test special float values (infinity, negative infinity, NaN)
        map_str_float_3 = ret.iloc[2]["map_str_float"]
        self.assertIsInstance(map_str_float_3, dict)
        self.assertEqual(len(map_str_float_3), 3)  # Should have 3 key-value pairs
        self.assertTrue(math.isinf(map_str_float_3['inf']))
        self.assertTrue(map_str_float_3['inf'] > 0)  # Positive infinity
        self.assertTrue(math.isinf(map_str_float_3['neg_inf']))
        self.assertTrue(map_str_float_3['neg_inf'] < 0)  # Negative infinity
        self.assertTrue(math.isnan(map_str_float_3['nan']))

        # Test extreme datetime values in row 3
        map_str_datetime_3 = ret.iloc[2]["map_str_datetime"]
        self.assertIsInstance(map_str_datetime_3, dict)
        self.assertEqual(len(map_str_datetime_3), 3)  # Should have 3 key-value pairs
        self.assertIsInstance(map_str_datetime_3['epoch'], datetime)
        self.assertEqual(map_str_datetime_3['epoch'], datetime(1970, 1, 1, 8, 0, 0, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime_3['y2k'], datetime)
        print(map_str_datetime_3['y2k'])
        self.assertEqual(map_str_datetime_3['y2k'], datetime(2000, 1, 1, 0, 0, 0, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime_3['future'], datetime)
        self.assertEqual(map_str_datetime_3['future'], datetime(2099, 12, 31, 23, 59, 59, tzinfo=self.shanghai_tz))

        # Test extreme date values in row 3
        map_str_date_3 = ret.iloc[2]["map_str_date"]
        self.assertIsInstance(map_str_date_3, dict)
        self.assertEqual(len(map_str_date_3), 3)  # Should have 3 key-value pairs
        self.assertIsInstance(map_str_date_3['past'], date)
        self.assertEqual(map_str_date_3['past'], date(1970, 1, 1))
        self.assertIsInstance(map_str_date_3['present'], date)
        # Note: 'present' uses today() so we just check it's a date, not exact value
        self.assertIsInstance(map_str_date_3['future'], date)
        self.assertEqual(map_str_date_3['future'], date(2100, 1, 1))

        # Test extreme DateTime64 values in row 3
        map_str_datetime64_3 = ret.iloc[2]["map_str_datetime64"]
        self.assertIsInstance(map_str_datetime64_3, dict)
        self.assertEqual(len(map_str_datetime64_3), 3)  # Should have 3 key-value pairs
        self.assertIsInstance(map_str_datetime64_3['epoch_precise'], datetime)
        self.assertEqual(map_str_datetime64_3['epoch_precise'], datetime(1970, 1, 1, 8, 0, 0, 1, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime64_3['y2k_precise'], datetime)
        self.assertEqual(map_str_datetime64_3['y2k_precise'], datetime(2000, 1, 1, 0, 0, 0, 999999, tzinfo=self.shanghai_tz))
        self.assertIsInstance(map_str_datetime64_3['future_precise'], datetime)
        self.assertEqual(map_str_datetime64_3['future_precise'], datetime(2099, 12, 31, 23, 59, 59, 123456, tzinfo=self.shanghai_tz))

        # Test extreme decimal values in row 3
        map_str_decimal_3 = ret.iloc[2]["map_str_decimal"]
        self.assertIsInstance(map_str_decimal_3, dict)
        self.assertEqual(len(map_str_decimal_3), 3)  # Should have 3 key-value pairs
        self.assertAlmostEqual(float(map_str_decimal_3['min_decimal']), 0.01, places=2)
        self.assertAlmostEqual(float(map_str_decimal_3['max_decimal']), 99999999.99, places=2)
        self.assertAlmostEqual(float(map_str_decimal_3['zero_decimal']), 0.00, places=2)

        # Test Array values in row 3 with multiple pairs including edge cases
        map_str_array_3 = ret.iloc[2]["map_str_array"]
        self.assertIsInstance(map_str_array_3, dict)
        self.assertEqual(len(map_str_array_3), 3)  # Should have 3 key-value pairs
        # Empty array
        empty_array = map_str_array_3['empty_val']
        self.assertIsInstance(empty_array, list)
        self.assertEqual(len(empty_array), 0)
        # Single element array
        single_array = map_str_array_3['single_val']
        self.assertIsInstance(single_array, list)
        np.testing.assert_array_equal(single_array, [42])
        # Multi element array
        multi_array = map_str_array_3['multi_val']
        self.assertIsInstance(multi_array, list)
        np.testing.assert_array_equal(multi_array, [1, 2, 3, 4, 5])

        # Test Tuple values in row 3 with multiple pairs
        map_str_tuple_3 = ret.iloc[2]["map_str_tuple"]
        self.assertIsInstance(map_str_tuple_3, dict)
        self.assertEqual(len(map_str_tuple_3), 2)  # Should have 2 key-value pairs
        # Empty-like tuple
        empty_like_tuple = map_str_tuple_3['empty_like']
        self.assertIsInstance(empty_like_tuple, tuple)
        self.assertEqual(empty_like_tuple, (0, ''))
        # Full tuple
        full_like_tuple = map_str_tuple_3['full_like']
        self.assertIsInstance(full_like_tuple, tuple)
        self.assertEqual(full_like_tuple, (999, 'full_string'))

        # Test empty arrays (non-hashable key -> keys/values structure)
        map_array_str_3 = ret.iloc[2]["map_array_str"]
        self.assertIn('keys', map_array_str_3)
        self.assertIn('values', map_array_str_3)
        empty_array_key = map_array_str_3['keys'][0]
        self.assertIsInstance(empty_array_key, list)
        self.assertEqual(len(empty_array_key), 0)  # Empty array
        self.assertEqual(map_array_str_3['values'][0], 'empty_array')

        # Test empty array as value (hashable key -> normal dict)
        map_str_array_3 = ret.iloc[2]["map_str_array"]
        empty_array_value = map_str_array_3['empty_val']
        self.assertIsInstance(empty_array_value, list)
        self.assertEqual(len(empty_array_value), 0)

        # Comprehensive type validation for all Map variations
        for i in range(len(ret)):
            # Verify all Maps return dict objects
            for col in ['map_str_int', 'map_int_str', 'map_bool_str', 'map_str_float',
                       'map_array_str', 'map_str_array', 'map_tuple_str', 'map_str_tuple',
                       'map_nested', 'map_nullable']:
                map_value = ret.iloc[i][col]
                self.assertIsInstance(map_value, dict, f"Row {i}, column {col} should be dict")

            # Verify Map(String, Int32) key-value types
            str_int_map = ret.iloc[i]["map_str_int"]
            for key, value in str_int_map.items():
                self.assertIsInstance(key, str, f"Row {i} map_str_int key should be string")
                self.assertIsInstance(value, (int, np.integer), f"Row {i} map_str_int value should be integer")

            # Verify Map(Int32, String) key-value types
            int_str_map = ret.iloc[i]["map_int_str"]
            for key, value in int_str_map.items():
                self.assertIsInstance(key, (int, np.integer), f"Row {i} map_int_str key should be integer")
                self.assertIsInstance(value, str, f"Row {i} map_int_str value should be string")

            # Verify Map(Bool, String) key-value types
            bool_str_map = ret.iloc[i]["map_bool_str"]
            for key, value in bool_str_map.items():
                self.assertIsInstance(key, (bool, np.bool_), f"Row {i} map_bool_str key should be bool")
                self.assertIsInstance(value, str, f"Row {i} map_bool_str value should be string")

        # Verify data types - All Map types should be mapped to object dtype in pandas
        expected_types = {
            "row_id": "uint8",
            "map_str_int": "object",        # Map(String, Int32) mapped to object dtype
            "map_int_str": "object",        # Map(Int32, String) mapped to object dtype
            "map_bool_str": "object",       # Map(Bool, String) mapped to object dtype
            "map_str_float": "object",      # Map(String, Float64) mapped to object dtype
            "map_array_str": "object",      # Map(Array(Int32), String) mapped to object dtype
            "map_str_array": "object",      # Map(String, Array(Int32)) mapped to object dtype
            "map_tuple_str": "object",      # Map(Tuple(Int32, String), String) mapped to object dtype
            "map_str_tuple": "object",      # Map(String, Tuple(Int32, String)) mapped to object dtype
            "map_nested": "object",         # Map(String, Map(String, Int32)) mapped to object dtype
            "map_nullable": "object"        # Map(String, Nullable(String)) mapped to object dtype
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"Column {col} type mismatch")

        # Test Map functions and operations
        map_ops_ret = self.session.query("""
            SELECT
                map('a', 1, 'b', 2, 'c', 3) as test_map,
                mapKeys(map('x', 10, 'y', 20)) as map_keys,
                mapValues(map('p', 100, 'q', 200)) as map_values,
                length(map('one', 1, 'two', 2, 'three', 3)) as map_length
        """, "DataFrame")

        test_map = map_ops_ret.iloc[0]["test_map"]
        self.assertIsInstance(test_map, dict)
        self.assertEqual(len(test_map), 3)

        # mapKeys should return an array
        map_keys = map_ops_ret.iloc[0]["map_keys"]
        self.assertIsInstance(map_keys, np.ndarray)
        self.assertEqual(len(map_keys), 2)
        self.assertIn('x', map_keys)
        self.assertIn('y', map_keys)

        # mapValues should return an array
        map_values = map_ops_ret.iloc[0]["map_values"]
        self.assertIsInstance(map_values, np.ndarray)
        self.assertEqual(len(map_values), 2)
        self.assertIn(100, map_values)
        self.assertIn(200, map_values)

        # length should return integer
        map_length = map_ops_ret.iloc[0]["map_length"]
        self.assertEqual(map_length, 3)


if __name__ == "__main__":
    unittest.main()
