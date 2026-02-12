#!/usr/bin/env python3

import os
import shutil
import unittest
import time
from urllib.request import urlretrieve
import pandas as pd
import chdb
import json
import numpy as np
from datetime import timedelta

STRING_DTYPE = "str" if pd.__version__ >= "3" else "object"


class TestDataFrameLargeScale(unittest.TestCase):
    """Test DataFrame generation with large scale data (1M rows) and diverse data types"""

    def setUp(self):
        self.test_dir = ".tmp_test_dataframe_large_scale_1"
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.session = chdb.session.Session(self.test_dir)

    def tearDown(self):
        self.session.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_large_scale_dataframe_generation(self):
        """Test generating 1M rows DataFrame with diverse data types and validate correctness"""

        print("Starting 1M row DataFrame generation test...")
        start_time = time.time()

        ret = self.session.query("""
            SELECT
                number as row_id,

                -- Integer types
                toInt8(number % 127) as int8_col,
                toInt16(number % 32767) as int16_col,
                toInt32(number % 2147483647) as int32_col,
                toInt64(number) as int64_col,
                toUInt8(number % 255) as uint8_col,
                toUInt16(number % 65535) as uint16_col,
                toUInt32(number % 4294967295) as uint32_col,
                toUInt64(number) as uint64_col,

                -- Float types
                toFloat32(number * 3.14159 / 1000000) as float32_col,
                toFloat64(number * 2.718281828 / 1000000) as float64_col,

                -- Decimal types
                toDecimal32(number % 1000000 / 100.0, 2) as decimal32_col,
                toDecimal64(number / 1000000.0, 6) as decimal64_col,

                -- String types
                concat('row_', toString(number)) as string_col,
                toFixedString(concat('fix', toString(number % 100)), 10) as fixed_string_col,

                -- Date and DateTime types
                toDate('2024-01-01') + (number % 365) as date_col,
                toDateTime('2024-01-01 00:00:00', 'Asia/Shanghai') + (number % 86400) as datetime_col,

                -- Boolean type
                toBool((number % 2) = 0) as bool_col,

                -- UUID type
                generateUUIDv4() as uuid_col,

                -- Nullable types
                if(number % 10 = 0, NULL, toInt32(number % 1000)) as nullable_int_col,
                if(number % 15 = 0, NULL, toFloat64(number / 1000.0)) as nullable_float_col,
                if(number % 20 = 0, NULL, concat('nullable_', toString(number))) as nullable_string_col,

                -- Array types
                [toInt32(number % 100), toInt32((number + 1) % 100), toInt32((number + 2) % 100)] as array_int_col,
                [toString(number % 10), toString((number + 1) % 10)] as array_string_col,
                [toFloat64(number / 1000.0), toFloat64((number + 1) / 1000.0)] as array_float_col,

                -- Tuple types
                (toInt32(number % 1000), concat('tuple_', toString(number % 100))) as tuple_int_string_col,
                (toFloat64(number / 1000000.0), toDate('2024-01-01') + (number % 30), number % 2 = 0) as tuple_mixed_col,

                -- JSON type (simulate with Map)
                map(
                    'id', toString(number),
                    'name', concat('user_', toString(number % 10000)),
                    'score', toString(toFloat64(number % 100) / 10.0),
                    'active', toString(number % 3 = 0)
                ) as json_col,

                -- Interval types
                INTERVAL (number % 3600) SECOND as interval_seconds_col,
                INTERVAL (number % 24) HOUR as interval_hours_col,
                INTERVAL (number % 30) DAY as interval_days_col,

                -- Enum simulation with LowCardinality
                toLowCardinality(
                    multiIf(
                        number % 5 = 0, 'Level_A',
                        number % 5 = 1, 'Level_B',
                        number % 5 = 2, 'Level_C',
                        number % 5 = 3, 'Level_D',
                        'Level_E'
                    )
                ) as enum_col
            FROM numbers(1000000)
        """, "DataFrame")

        query_time = time.time() - start_time
        print(f"Query execution time: {query_time:.2f} seconds")

        # Validate DataFrame structure
        self.assertEqual(len(ret), 1000000, "Should have exactly 1M rows")
        self.assertEqual(len(ret.columns), 32, "Should have exactly 32 columns")

        # Validate data types
        expected_types = {
            'row_id': ['uint64'],
            'int8_col': ['int8'],
            'int16_col': ['int16'],
            'int32_col': ['int32'],
            'int64_col': ['int64'],
            'uint8_col': ['uint8'],
            'uint16_col': ['uint16'],
            'uint32_col': ['uint32'],
            'uint64_col': ['uint64'],
            'float32_col': ['float32'],
            'float64_col': ['float64'],
            'string_col': [STRING_DTYPE],
            'bool_col': ['bool'],
            'date_col': ['datetime64[s]'],
            'datetime_col': ['datetime64[s, Asia/Shanghai]'],
            'nullable_int_col': ['Int32'],
            'array_int_col': ['object'],
            'array_string_col': ['object'],
            'tuple_int_string_col': ['object'],
            'json_col': ['object'],
            'enum_col': [STRING_DTYPE],
            'interval_seconds_col': ['timedelta64[s]'],
        }

        print("\nData type validation:")
        for col, allowed_types in expected_types.items():
            if col in ret.columns:
                actual_type = str(ret.dtypes[col])
                self.assertIn(actual_type, allowed_types,
                             f"Column {col} has unexpected type {actual_type}, expected one of {allowed_types}")
                print(f" {col}: {actual_type}")

        # Validate sample data correctness
        print("\nData correctness validation:")

        # Check first row (number = 0)
        first_row = ret.iloc[0]
        self.assertEqual(first_row['row_id'], 0)
        self.assertEqual(first_row['int8_col'], 0)  # 0 % 127 = 0
        self.assertEqual(first_row['int16_col'], 0)  # 0 % 32767 = 0
        self.assertEqual(first_row['uint8_col'], 0)  # 0 % 255 = 0
        self.assertEqual(first_row['string_col'], 'row_0')
        self.assertEqual(first_row['bool_col'], True)  # 0 % 2 == 0
        # Check nullable column - might be NaN instead of None
        self.assertTrue(pd.isna(first_row['nullable_int_col']), f"nullable_int_col should be NULL/NaN, got {first_row['nullable_int_col']}")  # 0 % 10 == 0 -> NULL
        self.assertEqual(first_row['float32_col'], 0.0)  # 0 * 3.14159 / 1000000 = 0
        print("First row data validation passed")

        # Check middle row (number = 500000)
        middle_row = ret.iloc[500000]
        self.assertEqual(middle_row['row_id'], 500000)
        self.assertEqual(middle_row['int8_col'], 500000 % 127)  # 500000 % 127 = 73
        self.assertEqual(middle_row['uint8_col'], 500000 % 255)  # 500000 % 255 = 5
        self.assertEqual(middle_row['string_col'], 'row_500000')
        self.assertEqual(middle_row['bool_col'], True)  # 500000 % 2 == 0
        # 500000 % 10 == 0, so should be NULL
        self.assertTrue(pd.isna(middle_row['nullable_int_col']), "nullable_int_col should be NULL/NaN")
        # Check enum value: 500000 % 5 = 0 -> 'Level_A'
        self.assertEqual(middle_row['enum_col'], 'Level_A')
        print("Middle row data validation passed")

        # Check last row (number = 999999)
        last_row = ret.iloc[999999]
        self.assertEqual(last_row['row_id'], 999999)
        self.assertEqual(last_row['int8_col'], 999999 % 127)  # 999999 % 127 = 126
        self.assertEqual(last_row['uint8_col'], 999999 % 255)  # 999999 % 255 = 254
        self.assertEqual(last_row['string_col'], 'row_999999')
        self.assertEqual(last_row['bool_col'], False)  # 999999 % 2 == 1
        # 999999 % 10 != 0, so should have value
        self.assertFalse(pd.isna(last_row['nullable_int_col']), "nullable_int_col should not be NULL/NaN")
        self.assertEqual(last_row['nullable_int_col'], 999999 % 1000)  # 999
        # Check enum value: 999999 % 5 = 4 -> 'Level_E'
        self.assertEqual(last_row['enum_col'], 'Level_E')
        print("Last row data validation passed")

        # Validate nullable columns have some NULL values
        null_count_nullable_int = ret['nullable_int_col'].isna().sum()
        null_count_nullable_float = ret['nullable_float_col'].isna().sum()
        null_count_nullable_string = ret['nullable_string_col'].isna().sum()

        self.assertEqual(null_count_nullable_int, 100000, "nullable_int_col should have exactly 100k NULLs (every 10th row)")
        self.assertEqual(null_count_nullable_float, 66667, "nullable_float_col should have exactly 66,667 NULLs (every 15th row)")
        self.assertEqual(null_count_nullable_string, 50000, "nullable_string_col should have exactly 50k NULLs (every 20th row)")

        print(f"NULL value validation: int({null_count_nullable_int}), float({null_count_nullable_float}), string({null_count_nullable_string})")

        # Validate array columns (using row 100, number = 100)
        sample_array_int = ret.iloc[100]['array_int_col']
        sample_array_string = ret.iloc[100]['array_string_col']
        sample_array_float = ret.iloc[100]['array_float_col']

        self.assertIsInstance(sample_array_int, np.ndarray, "array_int_col should be array-like")
        self.assertIsInstance(sample_array_string, np.ndarray, "array_string_col should be array-like")
        self.assertEqual(len(sample_array_int), 3, "array_int_col should have 3 elements")
        self.assertEqual(len(sample_array_string), 2, "array_string_col should have 2 elements")

        # Validate specific array values for row 100 (number = 100)
        expected_int_array = [100 % 100, 101 % 100, 102 % 100]  # [0, 1, 2]
        expected_string_array = ['0', '1']  # [toString(100 % 10), toString(101 % 10)]

        np.testing.assert_array_equal(sample_array_int, expected_int_array)
        np.testing.assert_array_equal(sample_array_string, expected_string_array)

        print("Array column validation passed")

        # Validate tuple columns (using row 200, number = 200)
        sample_tuple = ret.iloc[200]['tuple_int_string_col']
        sample_tuple_mixed = ret.iloc[200]['tuple_mixed_col']

        self.assertIsInstance(sample_tuple, np.ndarray, "tuple_int_string_col should be array-like")
        self.assertEqual(len(sample_tuple), 2, "tuple should have 2 elements")

        # Validate tuple values for row 200 (number = 200)
        expected_tuple_int = 200 % 1000  # 200
        expected_tuple_string = 'tuple_0'  # concat('tuple_', toString(200 % 100))

        self.assertEqual(sample_tuple[0], expected_tuple_int)
        self.assertEqual(sample_tuple[1], expected_tuple_string)

        # Validate mixed tuple: (float, date, bool)
        self.assertEqual(len(sample_tuple_mixed), 3, "Mixed tuple should have 3 elements")
        expected_float = 200 / 1000000.0  # 0.0002
        expected_bool = (200 % 2 == 0)  # True

        self.assertAlmostEqual(sample_tuple_mixed[0], expected_float, places=7)
        self.assertEqual(sample_tuple_mixed[2], expected_bool)

        print("Tuple column validation passed")

        # Validate JSON-like column (Map) (using row 300, number = 300)
        sample_json = ret.iloc[300]['json_col']
        self.assertIsInstance(sample_json, dict, "json_col should be dict-like")
        self.assertIn('id', sample_json, "JSON should have 'id' key")
        self.assertIn('name', sample_json, "JSON should have 'name' key")
        self.assertIn('score', sample_json, "JSON should have 'score' key")
        self.assertIn('active', sample_json, "JSON should have 'active' key")

        # Validate specific JSON values for row 300 (number = 300)
        self.assertEqual(sample_json['id'], '300')
        self.assertEqual(sample_json['name'], 'user_300')  # concat('user_', toString(300 % 10000))
        self.assertEqual(sample_json['score'], '0')     # toString(toFloat64(300 % 100) / 10.0) = toString(0.0)
        self.assertEqual(sample_json['active'], '1')   # toString(300 % 3 = 0) = toString(true)

        print("JSON column validation passed")

        print("Large scale DataFrame test completed successfully!")


if __name__ == '__main__':
    unittest.main()
