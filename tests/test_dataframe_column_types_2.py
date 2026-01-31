#!/usr/bin/env python3

import shutil
import unittest
import uuid
import pandas as pd
import chdb
import json
import numpy as np
import datetime
from datetime import date, timedelta


class TestDataFrameColumnTypesTwo(unittest.TestCase):

    def setUp(self):
        self.test_dir = ".tmp_test_dataframe_column_types_2"
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.session = chdb.session.Session(self.test_dir)

    def tearDown(self):
        self.session.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_variant_types(self):
        """Test Variant type with mixed data types"""
        # Enable suspicious variant types to allow similar types like Int32 and Float64
        self.session.query("SET allow_suspicious_variant_types = 1")

        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    NULL::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    NULL::Variant(Float64, String, Bool) as variant_mixed,
                    NULL::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
                UNION ALL
                SELECT
                    2 as row_id,
                    42::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    -100.0::Variant(Float64, String, Bool) as variant_mixed,
                    'Hello World'::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
                UNION ALL
                SELECT
                    3 as row_id,
                    'Hello, World!'::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    3.14159::Variant(Float64, String, Bool) as variant_mixed,
                    ['a', 'b', 'c']::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
                UNION ALL
                SELECT
                    4 as row_id,
                    [1, 2, 3]::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    true::Variant(Float64, String, Bool) as variant_mixed,
                    ('tuple_str', 123)::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
                UNION ALL
                SELECT
                    5 as row_id,
                    9223372036854775807::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    'mixed_string'::Variant(Float64, String, Bool) as variant_mixed,
                    'Simple String'::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
                UNION ALL
                SELECT
                    6 as row_id,
                    'Another String'::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    false::Variant(Float64, String, Bool) as variant_mixed,
                    ['x', 'y']::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
                UNION ALL
                SELECT
                    7 as row_id,
                    [10, 20, 30, 40]::Variant(UInt64, String, Array(UInt64)) as variant_basic,
                    -2.71828::Variant(Float64, String, Bool) as variant_mixed,
                    ('another', 456)::Variant(String, Array(String), Tuple(String, Int32)) as variant_complex
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Verify we have 7 rows and 4 columns
        self.assertEqual(len(ret), 7)
        self.assertEqual(len(ret.columns), 4)

        # Test first row - all NULL values
        self.assertTrue(pd.isna(ret.iloc[0]["variant_basic"]))
        self.assertTrue(pd.isna(ret.iloc[0]["variant_mixed"]))
        self.assertTrue(pd.isna(ret.iloc[0]["variant_complex"]))

        # Test second row - basic types (UInt64, Float64, String)
        self.assertEqual(ret.iloc[1]["variant_basic"], 42)  # UInt64
        self.assertEqual(ret.iloc[1]["variant_mixed"], -100.0)  # Float64
        self.assertEqual(ret.iloc[1]["variant_complex"], "Hello World")  # String

        # Test third row - different types (String, Float64, Array(String))
        self.assertEqual(ret.iloc[2]["variant_basic"], "Hello, World!")  # String
        self.assertAlmostEqual(ret.iloc[2]["variant_mixed"], 3.14159, places=5)  # Float64
        # Array may be returned as numpy array or list
        array_val = ret.iloc[2]["variant_complex"]
        if isinstance(array_val, np.ndarray):
            np.testing.assert_array_equal(array_val, ['a', 'b', 'c'])
        else:
            self.assertEqual(array_val, ['a', 'b', 'c'])

        # Test fourth row - Array(UInt64), Bool, Tuple(String, Int32)
        array_basic = ret.iloc[3]["variant_basic"]
        if isinstance(array_basic, np.ndarray):
            np.testing.assert_array_equal(array_basic, [1, 2, 3])
        else:
            self.assertEqual(array_basic, [1, 2, 3])
        self.assertEqual(ret.iloc[3]["variant_mixed"], True)  # Bool
        # Tuple may be returned as numpy array or tuple
        tuple_val = ret.iloc[3]["variant_complex"]
        if isinstance(tuple_val, np.ndarray):
            self.assertEqual(tuple_val[0], 'tuple_str')
            self.assertEqual(tuple_val[1], 123)
        else:
            self.assertEqual(tuple_val, ('tuple_str', 123))

        # Test fifth row - large UInt64, String, String
        self.assertEqual(ret.iloc[4]["variant_basic"], 9223372036854775807)  # Large UInt64
        self.assertEqual(ret.iloc[4]["variant_mixed"], "mixed_string")  # String
        self.assertEqual(ret.iloc[4]["variant_complex"], "Simple String")  # String

        # Test sixth row - String, Bool, Array(String)
        self.assertEqual(ret.iloc[5]["variant_basic"], "Another String")  # String
        self.assertEqual(ret.iloc[5]["variant_mixed"], False)  # Bool
        array_val_6 = ret.iloc[5]["variant_complex"]
        if isinstance(array_val_6, np.ndarray):
            np.testing.assert_array_equal(array_val_6, ['x', 'y'])
        else:
            self.assertEqual(array_val_6, ['x', 'y'])

        # Test seventh row - Array(UInt64), Float64, Tuple(String, Int32)
        array_val_7 = ret.iloc[6]["variant_basic"]
        if isinstance(array_val_7, np.ndarray):
            np.testing.assert_array_equal(array_val_7, [10, 20, 30, 40])
        else:
            self.assertEqual(array_val_7, [10, 20, 30, 40])
        self.assertAlmostEqual(ret.iloc[6]["variant_mixed"], -2.71828, places=5)  # Float64
        tuple_val_7 = ret.iloc[6]["variant_complex"]
        if isinstance(tuple_val_7, np.ndarray):
            self.assertEqual(tuple_val_7[0], 'another')
            self.assertEqual(tuple_val_7[1], 456)
        else:
            self.assertEqual(tuple_val_7, ('another', 456))

        # Data type validation - Variant types mapped to object in pandas
        expected_types = {
            "row_id": "uint8",
            "variant_basic": "object",  # Variant(UInt64, String, Array(UInt64))
            "variant_mixed": "object",  # Variant(Float64, String, Bool)
            "variant_complex": "object"  # Variant(String, Array(String), Tuple(String, Int32))
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    def test_dynamic_types(self):
        """Test Dynamic type with schema evolution"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    'Static String'::Dynamic as dynamic_string,
                    100::Dynamic as dynamic_number,
                    [1, 2, 3, 4, 5]::Dynamic as dynamic_array,
                    'Alice'::Dynamic as dynamic_object,
                    NULL::Dynamic as dynamic_null
                UNION ALL
                SELECT
                    2 as row_id,
                    'Another Dynamic String'::Dynamic as dynamic_string,
                    -500::Dynamic as dynamic_number,
                    ['x', 'y', 'z', 'w']::Dynamic as dynamic_array,
                    'Engineer'::Dynamic as dynamic_object,
                    'Now not null'::Dynamic as dynamic_null
                UNION ALL
                SELECT
                    3 as row_id,
                    'evolvedSchema'::Dynamic as dynamic_string,  -- Schema evolution: string in string field
                    [10, 20, 30]::Dynamic as dynamic_number,  -- Schema evolution: array in number field
                    'Now a string'::Dynamic as dynamic_array,  -- Schema evolution: string in array field
                    42::Dynamic as dynamic_object,  -- Schema evolution: number in object field
                    'nested_value'::Dynamic as dynamic_null  -- Schema evolution: string value
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - initial dynamic types
        self.assertEqual(ret.iloc[0]["dynamic_string"], "Static String")
        self.assertEqual(ret.iloc[0]["dynamic_number"], '100')
        self.assertIn("[1,2,3,4,5]", str(ret.iloc[0]["dynamic_array"]).replace(" ", ""))
        self.assertEqual(ret.iloc[0]["dynamic_object"], "Alice")
        self.assertTrue(pd.isna(ret.iloc[0]["dynamic_null"]))

        # Test second row - different dynamic values
        self.assertEqual(ret.iloc[1]["dynamic_string"], "Another Dynamic String")
        self.assertEqual(ret.iloc[1]["dynamic_number"], '-500')
        self.assertIn("['x','y','z','w']", str(ret.iloc[1]["dynamic_array"]).replace(" ", "").replace('"', "'"))
        self.assertEqual(ret.iloc[1]["dynamic_object"], "Engineer")
        self.assertEqual(ret.iloc[1]["dynamic_null"], "Now not null")

        # Test third row - schema evolution
        self.assertEqual(ret.iloc[2]["dynamic_string"], "evolvedSchema")
        self.assertIn("[10,20,30]", str(ret.iloc[2]["dynamic_number"]).replace(" ", ""))
        self.assertEqual(ret.iloc[2]["dynamic_array"], "Now a string")
        self.assertEqual(ret.iloc[2]["dynamic_object"], '42')
        self.assertEqual(ret.iloc[2]["dynamic_null"], "nested_value")

        # Data type validation - Dynamic types may be mapped to object in pandas
        expected_types = {
            "dynamic_string": "object",
            "dynamic_number": "object",
            "dynamic_array": "object",
            "dynamic_object": "object",
            "dynamic_null": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    def test_json_types(self):
        """Test JSON type with complex nested structures"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    '{"name": "John", "age": 30, "city": "New York"}'::JSON as json_simple,
                    '{"items": [1, 2, 3], "metadata": {"created": "2023-01-01", "version": 1.0}}'::JSON as json_nested,
                    '{"array": [1, 2, 3, 4, 5]}'::JSON as json_array,
                    '{"active": true, "score": null, "tags": ["urgent", "new"]}'::JSON as json_mixed,
                    'null'::JSON as json_null,
                    '{"value": "simple string"}'::JSON as json_string,
                    '{"value": 42}'::JSON as json_number,
                    '{"value": true}'::JSON as json_boolean
                UNION ALL
                SELECT
                    2 as row_id,
                    '{"product": "laptop", "price": 1299.99, "in_stock": true}'::JSON as json_simple,
                    '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "total": 2}'::JSON as json_nested,
                    '{"mixed": ["text", 3.14, true, null]}'::JSON as json_array,
                    '{"config": {"timeout": 30, "retries": 3}, "enabled": false}'::JSON as json_mixed,
                    '{"complex": null, "array": [null, {"nested": null}]}'::JSON as json_null,
                    '{"value": "Unicode: 🌍 éñáíóú"}'::JSON as json_string,
                    '{"value": -123.456}'::JSON as json_number,
                    '{"value": false}'::JSON as json_boolean
                UNION ALL
                SELECT
                    3 as row_id,
                    '{"very": {"deeply": {"nested": {"structure": {"with": {"many": {"levels": "value"}}}}}}}'::JSON as json_simple,
                    '{"matrix": [[1, 2], [3, 4]], "config": {"debug": false, "timeout": 30}}'::JSON as json_nested,
                    '{"objects": [{"a": 1}, {"b": 2}, {"c": [1, 2, 3]}]}'::JSON as json_array,
                    '{"key1": "value1", "key2": 42, "key3": [1, 2, 3], "key4": {"nested": true}}'::JSON as json_mixed,
                    '{"nullValue": null, "notNull": "value", "arrayWithNull": [1, null, 3]}'::JSON as json_null,
                    '{"value": "Special chars with escapes"}'::JSON as json_string,
                    '{"value": 1.7976931348623157e+308}'::JSON as json_number,
                    '{"value": true}'::JSON as json_boolean
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - basic JSON structures
        json_simple_1 = ret.iloc[0]["json_simple"]
        self.assertEqual(json_simple_1['name'], 'John')
        self.assertEqual(json_simple_1['age'], 30)
        self.assertEqual(json_simple_1['city'], 'New York')

        json_nested_1 = ret.iloc[0]["json_nested"]
        self.assertEqual(json_nested_1['items'], [1, 2, 3])
        self.assertIn('metadata', json_nested_1)
        created_date = json_nested_1['metadata']['created']
        self.assertIsInstance(created_date, datetime.date)
        self.assertEqual(created_date, datetime.date(2023, 1, 1))

        json_array_1 = ret.iloc[0]["json_array"]
        self.assertEqual(json_array_1['array'], [1, 2, 3, 4, 5])

        json_mixed_1 = ret.iloc[0]["json_mixed"]
        self.assertIsInstance(json_mixed_1['active'], bool)
        self.assertTrue(json_mixed_1['active'])
        self.assertNotIn('score', json_mixed_1)  # null values don't create keys
        self.assertEqual(json_mixed_1['tags'], ['urgent', 'new'])

        self.assertIsNone(ret.iloc[0]["json_null"])
        json_string_1 = ret.iloc[0]["json_string"]
        self.assertEqual(json_string_1['value'], 'simple string')
        json_number_1 = ret.iloc[0]["json_number"]
        self.assertEqual(json_number_1['value'], 42)
        json_boolean_1 = ret.iloc[0]["json_boolean"]
        self.assertEqual(json_boolean_1['value'], True)

        # Test second row - more complex JSON structures
        json_simple_2 = ret.iloc[1]["json_simple"]
        self.assertEqual(json_simple_2['product'], 'laptop')
        self.assertEqual(json_simple_2['price'], 1299.99)
        self.assertEqual(json_simple_2['in_stock'], True)

        json_nested_2 = ret.iloc[1]["json_nested"]
        expected_users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        self.assertEqual(json_nested_2['users'], expected_users)
        self.assertEqual(json_nested_2['total'], 2)

        json_array_2 = ret.iloc[1]["json_array"]
        expected_mixed = ['text', 3.14, True, None]
        self.assertEqual(json_array_2['mixed'], expected_mixed)

        json_mixed_2 = ret.iloc[1]["json_mixed"]
        self.assertEqual(json_mixed_2['config']['timeout'], 30)
        self.assertEqual(json_mixed_2['config']['retries'], 3)
        self.assertEqual(json_mixed_2['enabled'], False)

        json_null_2 = ret.iloc[1]["json_null"]
        self.assertNotIn('complex', json_null_2)  # complex is null, so key doesn't exist
        expected_array = [None, None]
        self.assertEqual(json_null_2['array'], expected_array)

        json_string_2 = ret.iloc[1]["json_string"]
        self.assertIn("Unicode", json_string_2['value'])
        self.assertIn("🌍", json_string_2['value'])
        json_number_2 = ret.iloc[1]["json_number"]
        self.assertEqual(json_number_2['value'], -123.456)

        json_boolean_2 = ret.iloc[1]["json_boolean"]
        self.assertEqual(json_boolean_2['value'], False)

        # Test third row - very complex and edge cases
        json_simple_3 = ret.iloc[2]["json_simple"]
        self.assertEqual(json_simple_3['very']['deeply']['nested']['structure']['with']['many']['levels'], 'value')

        json_nested_3 = ret.iloc[2]["json_nested"]
        self.assertEqual(json_nested_3['matrix'], [[1, 2], [3, 4]])
        self.assertEqual(json_nested_3['config']['debug'], False)
        self.assertEqual(json_nested_3['config']['timeout'], 30)

        json_array_3 = ret.iloc[2]["json_array"]
        expected_objects = [{'a': 1}, {'b': 2}, {'c': [1, 2, 3]}]
        self.assertEqual(json_array_3['objects'], expected_objects)

        json_mixed_3 = ret.iloc[2]["json_mixed"]
        self.assertEqual(json_mixed_3['key1'], 'value1')
        self.assertEqual(json_mixed_3['key2'], 42)
        self.assertEqual(json_mixed_3['key3'], [1, 2, 3])
        self.assertEqual(json_mixed_3['key4']['nested'], True)

        json_null_3 = ret.iloc[2]["json_null"]
        self.assertNotIn('nullValue', json_null_3)  # null values don't create keys
        self.assertEqual(json_null_3['notNull'], 'value')
        self.assertEqual(json_null_3['arrayWithNull'], [1, None, 3])

        json_string_3 = ret.iloc[2]["json_string"]
        self.assertIn("Special chars with escapes", json_string_3['value'])

        json_number_3 = ret.iloc[2]["json_number"]
        self.assertEqual(json_number_3['value'], 1.7976931348623157e+308)  # Large number representation

        json_boolean_3 = ret.iloc[2]["json_boolean"]
        self.assertEqual(json_boolean_3['value'], True)

        # Data type validation - JSON types mapped to object in pandas
        expected_types = {
            "row_id": "uint8",
            "json_simple": "object",
            "json_nested": "object",
            "json_array": "object",
            "json_mixed": "object",
            "json_null": "object",
            "json_string": "object",
            "json_number": "object",
            "json_boolean": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    def test_nested_types(self):
        """Test Nested type with structured data"""
        ret = self.session.query("""
            SELECT * FROM (
                SELECT
                    1 as row_id,
                    [(1, 'Alice', 25.5), (2, 'Bob', 30.0), (3, 'Charlie', 35.5)]::Array(Tuple(UInt32, String, Float64))::Nested(id UInt32, name String, salary Float64) as employees,
                    [(100, 'Engineering'), (200, 'Marketing'), (300, 'Finance')]::Array(Tuple(UInt32, String))::Nested(dept_id UInt32, dept_name String) as departments,
                    [('2023-01-01', 1000.0), ('2023-02-01', 1500.0)]::Array(Tuple(Date, Float64))::Nested(date Date, amount Float64) as transactions
                UNION ALL
                SELECT
                    2 as row_id,
                    [(4, 'Diana', 45.0), (5, 'Eve', 28.5)]::Array(Tuple(UInt32, String, Float64))::Nested(id UInt32, name String, salary Float64) as employees,
                    [(400, 'Sales'), (500, 'HR')]::Array(Tuple(UInt32, String))::Nested(dept_id UInt32, dept_name String) as departments,
                    [('2023-03-01', 2000.0), ('2023-04-01', 2500.0), ('2023-05-01', 1800.0)]::Array(Tuple(Date, Float64))::Nested(date Date, amount Float64) as transactions
                UNION ALL
                SELECT
                    3 as row_id,
                    [(6, 'Frank', 55.0)]::Array(Tuple(UInt32, String, Float64))::Nested(id UInt32, name String, salary Float64) as employees,
                    [(600, 'Operations')]::Array(Tuple(UInt32, String))::Nested(dept_id UInt32, dept_name String) as departments,
                    [('2023-06-01', 3000.0)]::Array(Tuple(Date, Float64))::Nested(date Date, amount Float64) as transactions
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - multiple employees (Nested types are returned as numpy arrays of tuples)
        employees_1 = ret.iloc[0]["employees"]
        # Extract values by iterating through tuples and accessing by position
        ids_1 = [emp[0] for emp in employees_1]
        names_1 = [emp[1] for emp in employees_1]
        salaries_1 = [emp[2] for emp in employees_1]
        self.assertEqual(ids_1, [1, 2, 3])
        self.assertEqual(names_1, ['Alice', 'Bob', 'Charlie'])
        self.assertEqual(salaries_1, [25.5, 30.0, 35.5])

        departments_1 = ret.iloc[0]["departments"]
        # Extract dept_id and dept_name by position
        dept_ids_1 = [dept[0] for dept in departments_1]
        dept_names_1 = [dept[1] for dept in departments_1]
        self.assertEqual(dept_ids_1, [100, 200, 300])
        self.assertEqual(dept_names_1, ['Engineering', 'Marketing', 'Finance'])

        transactions_1 = ret.iloc[0]["transactions"]
        # Extract date and amount by position
        dates_1 = [trans[0] for trans in transactions_1]
        amounts_1 = [trans[1] for trans in transactions_1]
        expected_dates_1 = [datetime.date(2023, 1, 1), datetime.date(2023, 2, 1)]
        self.assertEqual(dates_1, expected_dates_1)
        self.assertEqual(amounts_1, [1000.0, 1500.0])

        # Test second row - different data
        employees_2 = ret.iloc[1]["employees"]
        ids_2 = [emp[0] for emp in employees_2]
        names_2 = [emp[1] for emp in employees_2]
        salaries_2 = [emp[2] for emp in employees_2]
        self.assertEqual(ids_2, [4, 5])
        self.assertEqual(names_2, ['Diana', 'Eve'])
        self.assertEqual(salaries_2, [45.0, 28.5])

        departments_2 = ret.iloc[1]["departments"]
        dept_ids_2 = [dept[0] for dept in departments_2]
        dept_names_2 = [dept[1] for dept in departments_2]
        self.assertEqual(dept_ids_2, [400, 500])
        self.assertEqual(dept_names_2, ['Sales', 'HR'])

        transactions_2 = ret.iloc[1]["transactions"]
        dates_2 = [trans[0] for trans in transactions_2]
        amounts_2 = [trans[1] for trans in transactions_2]
        expected_dates_2 = [datetime.date(2023, 3, 1), datetime.date(2023, 4, 1), datetime.date(2023, 5, 1)]
        self.assertEqual(dates_2, expected_dates_2)
        self.assertEqual(amounts_2, [2000.0, 2500.0, 1800.0])

        # Test third row - single employee
        employees_3 = ret.iloc[2]["employees"]
        ids_3 = [emp[0] for emp in employees_3]
        names_3 = [emp[1] for emp in employees_3]
        salaries_3 = [emp[2] for emp in employees_3]
        self.assertEqual(ids_3, [6])
        self.assertEqual(names_3, ['Frank'])
        self.assertEqual(salaries_3, [55.0])

        departments_3 = ret.iloc[2]["departments"]
        dept_ids_3 = [dept[0] for dept in departments_3]
        dept_names_3 = [dept[1] for dept in departments_3]
        self.assertEqual(dept_ids_3, [600])
        self.assertEqual(dept_names_3, ['Operations'])

        transactions_3 = ret.iloc[2]["transactions"]
        dates_3 = [trans[0] for trans in transactions_3]
        amounts_3 = [trans[1] for trans in transactions_3]
        expected_dates_3 = [datetime.date(2023, 6, 1)]
        self.assertEqual(dates_3, expected_dates_3)
        self.assertEqual(amounts_3, [3000.0])

        # Data type validation - Nested types should be mapped to object in pandas
        expected_types = {
            "row_id": "uint8",
            "employees": "object",
            "departments": "object",
            "transactions": "object"
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type)

    def test_interval_types(self):
        """Test various Interval types - time intervals in ClickHouse"""

        ret = self.session.query("""
            SELECT * FROM (
            SELECT
                1 as row_id,
                INTERVAL 30 SECOND as interval_seconds,
                INTERVAL 15 MINUTE as interval_minutes,
                INTERVAL 3 HOUR as interval_hours,
                INTERVAL 7 DAY as interval_days,
                INTERVAL 2 WEEK as interval_weeks,
                INTERVAL 6 MONTH as interval_months,
                INTERVAL 1 QUARTER as interval_quarters,
                INTERVAL 2 YEAR as interval_years
            UNION ALL
            SELECT
                2 as row_id,
                INTERVAL 90 SECOND as interval_seconds,
                INTERVAL 45 MINUTE as interval_minutes,
                INTERVAL 12 HOUR as interval_hours,
                INTERVAL 14 DAY as interval_days,
                INTERVAL 4 WEEK as interval_weeks,
                INTERVAL 18 MONTH as interval_months,
                INTERVAL 2 QUARTER as interval_quarters,
                INTERVAL 5 YEAR as interval_years
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - basic intervals
        self.assertEqual(ret.iloc[0]["row_id"], 1)

        # Interval values are typically returned as timedelta objects or integers representing units
        # Let's test the actual values
        intervals_1 = {
            "interval_seconds": ret.iloc[0]["interval_seconds"],
            "interval_minutes": ret.iloc[0]["interval_minutes"],
            "interval_hours": ret.iloc[0]["interval_hours"],
            "interval_days": ret.iloc[0]["interval_days"],
            "interval_weeks": ret.iloc[0]["interval_weeks"],
            "interval_months": ret.iloc[0]["interval_months"],
            "interval_quarters": ret.iloc[0]["interval_quarters"],
            "interval_years": ret.iloc[0]["interval_years"]
        }

        # Check interval values are not None and have expected types
        # Basic intervals should return timedelta64 type
        for interval_name, interval_value in intervals_1.items():
            self.assertIsNotNone(interval_value, f"{interval_name} should not be None")
            self.assertEqual(type(interval_value).__name__, 'Timedelta',
                           f"{interval_name} should be timedelta64, got {type(interval_value).__name__}")

        # Test second row - different interval values
        self.assertEqual(ret.iloc[1]["row_id"], 2)

        intervals_2 = {
            "interval_seconds": ret.iloc[1]["interval_seconds"],
            "interval_minutes": ret.iloc[1]["interval_minutes"],
            "interval_hours": ret.iloc[1]["interval_hours"],
            "interval_days": ret.iloc[1]["interval_days"],
            "interval_weeks": ret.iloc[1]["interval_weeks"],
            "interval_months": ret.iloc[1]["interval_months"],
            "interval_quarters": ret.iloc[1]["interval_quarters"],
            "interval_years": ret.iloc[1]["interval_years"]
        }

        # Check second row interval values - should also be timedelta64
        for interval_name, interval_value in intervals_2.items():
            self.assertIsNotNone(interval_value, f"{interval_name} should not be None")
            self.assertEqual(type(interval_value).__name__, 'Timedelta',
                           f"{interval_name} should be timedelta64, got {type(interval_value).__name__}")

        # Data type validation - Intervals should be mapped according to C++ NumpyType.cpp
        expected_interval_types = {
            "row_id": "uint8",
            "interval_seconds": "timedelta64[s]",      # Second -> timedelta64[s]
            "interval_minutes": "timedelta64[s]",      # Minute -> timedelta64[m]
            "interval_hours": "timedelta64[s]",        # Hour -> timedelta64[h]
            "interval_days": "timedelta64[s]",         # Day -> timedelta64[D]
            "interval_weeks": "timedelta64[s]",        # Week -> timedelta64[W]
            "interval_months": "timedelta64[s]",       # Month -> timedelta64[M]
            "interval_quarters": "timedelta64[s]",     # Quarter -> timedelta64[M] (numpy doesn't have quarter)
            "interval_years": "timedelta64[s]"         # Year -> timedelta64[Y]
        }

        for col, expected_type in expected_interval_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"{col} dtype should be {expected_type}, got {actual_type}")

        # Value assertions for first row
        self.assertEqual(intervals_1["interval_seconds"], pd.Timedelta(seconds=30))
        self.assertEqual(intervals_1["interval_minutes"], pd.Timedelta(minutes=15))
        self.assertEqual(intervals_1["interval_hours"], pd.Timedelta(hours=3))
        self.assertEqual(intervals_1["interval_days"], pd.Timedelta(days=7))
        self.assertEqual(intervals_1["interval_weeks"], pd.Timedelta(weeks=2))
        self.assertEqual(intervals_1["interval_months"], pd.Timedelta(days=180))  # Approximate months as days
        self.assertEqual(intervals_1["interval_quarters"], pd.Timedelta(days=1*90))  # Approximate quarters as days
        self.assertEqual(intervals_1["interval_years"], pd.Timedelta(days=2*365))  # Approximate years as days

        # Value assertions for second row
        self.assertEqual(intervals_2["interval_seconds"], pd.Timedelta(seconds=90))
        self.assertEqual(intervals_2["interval_minutes"], pd.Timedelta(minutes=45))
        self.assertEqual(intervals_2["interval_hours"], pd.Timedelta(hours=12))
        self.assertEqual(intervals_2["interval_days"], pd.Timedelta(days=14))
        self.assertEqual(intervals_2["interval_weeks"], pd.Timedelta(weeks=4))
        self.assertEqual(intervals_2["interval_months"], pd.Timedelta(days=540))  # Approximate months as days
        self.assertEqual(intervals_2["interval_quarters"], pd.Timedelta(days=180))  # Approximate quarters as days
        self.assertEqual(intervals_2["interval_years"], pd.Timedelta(days=365 * 5))  # Approximate years as days

    def test_nested_interval_types(self):
        """Test Interval types in nested structures like tuples - should return timedelta"""

        ret = self.session.query("""
            SELECT * FROM (
            SELECT
                1 as row_id,
                (INTERVAL 1000 NANOSECOND, INTERVAL 5000 NANOSECOND) as tuple_intervals_nanoseconds,
                (INTERVAL 500 MICROSECOND, INTERVAL 1500 MICROSECOND) as tuple_intervals_microseconds,
                (INTERVAL 100 MILLISECOND, INTERVAL 500 MILLISECOND) as tuple_intervals_milliseconds,
                (INTERVAL 30 SECOND, INTERVAL 90 SECOND) as tuple_intervals_seconds,
                (INTERVAL 15 MINUTE, INTERVAL 45 MINUTE) as tuple_intervals_minutes,
                (INTERVAL 2 HOUR, INTERVAL 6 HOUR) as tuple_intervals_hours,
                (INTERVAL 1 DAY, INTERVAL 3 DAY) as tuple_intervals_days,
                (INTERVAL 1 WEEK, INTERVAL 2 WEEK) as tuple_intervals_weeks,
                (INTERVAL 1 MONTH, INTERVAL 3 MONTH) as tuple_intervals_months,
                (INTERVAL 1 QUARTER, INTERVAL 2 QUARTER) as tuple_intervals_quarters,
                (INTERVAL 1 YEAR, INTERVAL 2 YEAR) as tuple_intervals_years
            UNION ALL
            SELECT
                2 as row_id,
                (INTERVAL 2000 NANOSECOND, INTERVAL 8000 NANOSECOND) as tuple_intervals_nanoseconds,
                (INTERVAL 800 MICROSECOND, INTERVAL 2000 MICROSECOND) as tuple_intervals_microseconds,
                (INTERVAL 200 MILLISECOND, INTERVAL 800 MILLISECOND) as tuple_intervals_milliseconds,
                (INTERVAL 60 SECOND, INTERVAL 120 SECOND) as tuple_intervals_seconds,
                (INTERVAL 30 MINUTE, INTERVAL 60 MINUTE) as tuple_intervals_minutes,
                (INTERVAL 4 HOUR, INTERVAL 8 HOUR) as tuple_intervals_hours,
                (INTERVAL 2 DAY, INTERVAL 5 DAY) as tuple_intervals_days,
                (INTERVAL 3 WEEK, INTERVAL 4 WEEK) as tuple_intervals_weeks,
                (INTERVAL 6 MONTH, INTERVAL 12 MONTH) as tuple_intervals_months,
                (INTERVAL 3 QUARTER, INTERVAL 4 QUARTER) as tuple_intervals_quarters,
                (INTERVAL 3 YEAR, INTERVAL 5 YEAR) as tuple_intervals_years
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - nested intervals in tuples
        self.assertEqual(ret.iloc[0]["row_id"], 1)

        # Nested intervals should return timedelta objects when in tuples
        tuple_intervals_1 = {
            "tuple_intervals_nanoseconds": ret.iloc[0]["tuple_intervals_nanoseconds"],
            "tuple_intervals_microseconds": ret.iloc[0]["tuple_intervals_microseconds"],
            "tuple_intervals_milliseconds": ret.iloc[0]["tuple_intervals_milliseconds"],
            "tuple_intervals_seconds": ret.iloc[0]["tuple_intervals_seconds"],
            "tuple_intervals_minutes": ret.iloc[0]["tuple_intervals_minutes"],
            "tuple_intervals_hours": ret.iloc[0]["tuple_intervals_hours"],
            "tuple_intervals_days": ret.iloc[0]["tuple_intervals_days"],
            "tuple_intervals_weeks": ret.iloc[0]["tuple_intervals_weeks"],
            "tuple_intervals_months": ret.iloc[0]["tuple_intervals_months"],
            "tuple_intervals_quarters": ret.iloc[0]["tuple_intervals_quarters"],
            "tuple_intervals_years": ret.iloc[0]["tuple_intervals_years"]
        }

        # Check nested interval tuples - elements should be timedelta
        for tuple_name, tuple_value in tuple_intervals_1.items():
            self.assertIsNotNone(tuple_value, f"{tuple_name} should not be None")
            # Should be a tuple containing interval values
            self.assertTrue(hasattr(tuple_value, '__iter__'), f"{tuple_name} should be iterable")

            # Check individual elements in the tuple - should be timedelta
            for i, interval_elem in enumerate(tuple_value):
                self.assertIsNotNone(interval_elem, f"{tuple_name}[{i}] should not be None")
                self.assertEqual(type(interval_elem).__name__, 'timedelta',
                                f"{tuple_name}[{i}] should be timedelta, got {type(interval_elem).__name__}")

        # Test second row
        self.assertEqual(ret.iloc[1]["row_id"], 2)

        tuple_intervals_2 = {
            "tuple_intervals_nanoseconds": ret.iloc[1]["tuple_intervals_nanoseconds"],
            "tuple_intervals_microseconds": ret.iloc[1]["tuple_intervals_microseconds"],
            "tuple_intervals_milliseconds": ret.iloc[1]["tuple_intervals_milliseconds"],
            "tuple_intervals_seconds": ret.iloc[1]["tuple_intervals_seconds"],
            "tuple_intervals_minutes": ret.iloc[1]["tuple_intervals_minutes"],
            "tuple_intervals_hours": ret.iloc[1]["tuple_intervals_hours"],
            "tuple_intervals_days": ret.iloc[1]["tuple_intervals_days"],
            "tuple_intervals_weeks": ret.iloc[1]["tuple_intervals_weeks"],
            "tuple_intervals_months": ret.iloc[1]["tuple_intervals_months"],
            "tuple_intervals_quarters": ret.iloc[1]["tuple_intervals_quarters"],
            "tuple_intervals_years": ret.iloc[1]["tuple_intervals_years"]
        }

        # Check second row nested interval tuples
        for tuple_name, tuple_value in tuple_intervals_2.items():
            self.assertIsNotNone(tuple_value, f"{tuple_name} should not be None")
            self.assertTrue(hasattr(tuple_value, '__iter__'), f"{tuple_name} should be iterable")

            for i, interval_elem in enumerate(tuple_value):
                self.assertIsNotNone(interval_elem, f"{tuple_name}[{i}] should not be None")
                self.assertEqual(type(interval_elem).__name__, 'timedelta',
                               f"{tuple_name}[{i}] should be timedelta, got {type(interval_elem).__name__}")

        # Data type validation - Tuple intervals should be object type containing tuples
        expected_nested_interval_types = {
            "row_id": "uint8",
            "tuple_intervals_nanoseconds": "object",
            "tuple_intervals_microseconds": "object",
            "tuple_intervals_milliseconds": "object",
            "tuple_intervals_seconds": "object",
            "tuple_intervals_minutes": "object",
            "tuple_intervals_hours": "object",
            "tuple_intervals_days": "object",
            "tuple_intervals_weeks": "object",
            "tuple_intervals_months": "object",
            "tuple_intervals_quarters": "object",
            "tuple_intervals_years": "object"
        }

        for col, expected_type in expected_nested_interval_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"{col} dtype should be {expected_type}, got {actual_type}")


        # Nanoseconds: (1000ns, 5000ns) -> microseconds = value / 1000
        self.assertEqual(tuple_intervals_1["tuple_intervals_nanoseconds"][0], timedelta(microseconds=1000/1000))
        self.assertEqual(tuple_intervals_1["tuple_intervals_nanoseconds"][1], timedelta(microseconds=5000/1000))

        # Microseconds: (500us, 1500us)
        self.assertEqual(tuple_intervals_1["tuple_intervals_microseconds"][0], timedelta(microseconds=500))
        self.assertEqual(tuple_intervals_1["tuple_intervals_microseconds"][1], timedelta(microseconds=1500))

        # Milliseconds: (100ms, 500ms)
        self.assertEqual(tuple_intervals_1["tuple_intervals_milliseconds"][0], timedelta(milliseconds=100))
        self.assertEqual(tuple_intervals_1["tuple_intervals_milliseconds"][1], timedelta(milliseconds=500))

        # Seconds: (30s, 90s)
        self.assertEqual(tuple_intervals_1["tuple_intervals_seconds"][0], timedelta(seconds=30))
        self.assertEqual(tuple_intervals_1["tuple_intervals_seconds"][1], timedelta(seconds=90))

        # Minutes: (15m, 45m)
        self.assertEqual(tuple_intervals_1["tuple_intervals_minutes"][0], timedelta(minutes=15))
        self.assertEqual(tuple_intervals_1["tuple_intervals_minutes"][1], timedelta(minutes=45))

        # Hours: (2h, 6h)
        self.assertEqual(tuple_intervals_1["tuple_intervals_hours"][0], timedelta(hours=2))
        self.assertEqual(tuple_intervals_1["tuple_intervals_hours"][1], timedelta(hours=6))

        # Days: (1d, 3d)
        self.assertEqual(tuple_intervals_1["tuple_intervals_days"][0], timedelta(days=1))
        self.assertEqual(tuple_intervals_1["tuple_intervals_days"][1], timedelta(days=3))

        # Weeks: (1w, 2w)
        self.assertEqual(tuple_intervals_1["tuple_intervals_weeks"][0], timedelta(weeks=1))
        self.assertEqual(tuple_intervals_1["tuple_intervals_weeks"][1], timedelta(weeks=2))

        # Months: (1 month, 3 months) -> days = value * 30
        self.assertEqual(tuple_intervals_1["tuple_intervals_months"][0], timedelta(days=1*30))
        self.assertEqual(tuple_intervals_1["tuple_intervals_months"][1], timedelta(days=3*30))

        # Quarters: (1 quarter, 2 quarters) -> days = value * 90
        self.assertEqual(tuple_intervals_1["tuple_intervals_quarters"][0], timedelta(days=1*90))
        self.assertEqual(tuple_intervals_1["tuple_intervals_quarters"][1], timedelta(days=2*90))

        # Years: (1 year, 2 years) -> days = value * 365
        self.assertEqual(tuple_intervals_1["tuple_intervals_years"][0], timedelta(days=1*365))
        self.assertEqual(tuple_intervals_1["tuple_intervals_years"][1], timedelta(days=2*365))

        # Value assertions for second row tuples
        # Nanoseconds: (2000ns, 8000ns) -> microseconds = value / 1000
        self.assertEqual(tuple_intervals_2["tuple_intervals_nanoseconds"][0], timedelta(microseconds=2000/1000))
        self.assertEqual(tuple_intervals_2["tuple_intervals_nanoseconds"][1], timedelta(microseconds=8000/1000))

        # Microseconds: (800us, 2000us)
        self.assertEqual(tuple_intervals_2["tuple_intervals_microseconds"][0], timedelta(microseconds=800))
        self.assertEqual(tuple_intervals_2["tuple_intervals_microseconds"][1], timedelta(microseconds=2000))

        # Milliseconds: (200ms, 800ms)
        self.assertEqual(tuple_intervals_2["tuple_intervals_milliseconds"][0], timedelta(milliseconds=200))
        self.assertEqual(tuple_intervals_2["tuple_intervals_milliseconds"][1], timedelta(milliseconds=800))

        # Seconds: (60s, 120s)
        self.assertEqual(tuple_intervals_2["tuple_intervals_seconds"][0], timedelta(seconds=60))
        self.assertEqual(tuple_intervals_2["tuple_intervals_seconds"][1], timedelta(seconds=120))

        # Minutes: (30m, 60m)
        self.assertEqual(tuple_intervals_2["tuple_intervals_minutes"][0], timedelta(minutes=30))
        self.assertEqual(tuple_intervals_2["tuple_intervals_minutes"][1], timedelta(minutes=60))

        # Hours: (4h, 8h)
        self.assertEqual(tuple_intervals_2["tuple_intervals_hours"][0], timedelta(hours=4))
        self.assertEqual(tuple_intervals_2["tuple_intervals_hours"][1], timedelta(hours=8))

        # Days: (2d, 5d)
        self.assertEqual(tuple_intervals_2["tuple_intervals_days"][0], timedelta(days=2))
        self.assertEqual(tuple_intervals_2["tuple_intervals_days"][1], timedelta(days=5))

        # Weeks: (3w, 4w)
        self.assertEqual(tuple_intervals_2["tuple_intervals_weeks"][0], timedelta(weeks=3))
        self.assertEqual(tuple_intervals_2["tuple_intervals_weeks"][1], timedelta(weeks=4))

        # Months: (6 months, 12 months) -> days = value * 30
        self.assertEqual(tuple_intervals_2["tuple_intervals_months"][0], timedelta(days=6*30))
        self.assertEqual(tuple_intervals_2["tuple_intervals_months"][1], timedelta(days=12*30))

        # Quarters: (3 quarters, 4 quarters) -> days = value * 90
        self.assertEqual(tuple_intervals_2["tuple_intervals_quarters"][0], timedelta(days=3*90))
        self.assertEqual(tuple_intervals_2["tuple_intervals_quarters"][1], timedelta(days=4*90))

        # Years: (3 years, 5 years) -> days = value * 365
        self.assertEqual(tuple_intervals_2["tuple_intervals_years"][0], timedelta(days=3*365))
        self.assertEqual(tuple_intervals_2["tuple_intervals_years"][1], timedelta(days=5*365))

    def test_nothing_types(self):
        """Test Nothing type - represents absence of value"""
        ret = self.session.query("""
            SELECT
                1 as row_id,
                array() as nothing_val
            FROM numbers(1)
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        nothing_val = ret.iloc[0]["nothing_val"]

        # Check if it's an empty array
        self.assertEqual(len(nothing_val), 0, "Should be empty array")

        # Data type validation
        expected_types = {
            "row_id": "uint8",
            "nothing_val": "object",
        }

        for col, expected_type in expected_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"{col} dtype should be {expected_type}, got {actual_type}")

    def test_geo_types(self):
        """Test native Point and Ring geo types"""
        ret = self.session.query("""
            SELECT * FROM (
            SELECT
                1 as row_id,
                (0.0, 0.0)::Point as point_origin,
                (37.7749, -122.4194)::Point as point_sf,
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]::Ring as ring_square
            UNION ALL
            SELECT
                2 as row_id,
                (-74.006, 40.7128)::Point as point_origin,
                (51.5074, -0.1278)::Point as point_sf,
                [(-1.0, -1.0), (0.0, -1.0), (0.0, 0.0), (-1.0, 0.0), (-1.0, -1.0)]::Ring as ring_square
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row geo values
        self.assertEqual(ret.iloc[0]["row_id"], 1)

        # Point is Tuple(Float64, Float64) - should be tuples with 2 float coordinates
        point_origin = ret.iloc[0]["point_origin"]
        self.assertIsInstance(point_origin, np.ndarray, "Point should be tuple")
        self.assertEqual(len(point_origin), 2, "Point should have 2 coordinates")
        self.assertEqual(point_origin[0], 0.0)
        self.assertEqual(point_origin[1], 0.0)

        point_sf = ret.iloc[0]["point_sf"]
        self.assertAlmostEqual(point_sf[0], 37.7749, places=4)
        self.assertAlmostEqual(point_sf[1], -122.4194, places=4)

        # Ring is Array(Point) - should be array of points
        ring_square = ret.iloc[0]["ring_square"]
        self.assertTrue(hasattr(ring_square, '__iter__'), "Ring should be iterable")
        self.assertEqual(len(ring_square), 5, "Square ring should have 5 points (closed)")

        # Each point in ring should be a tuple
        for point in ring_square:
            self.assertIsInstance(point, np.ndarray, "Each point in ring should be tuple")
            self.assertEqual(len(point), 2, "Each point should have 2 coordinates")

        # Test second row
        self.assertEqual(ret.iloc[1]["row_id"], 2)
        point_nyc = ret.iloc[1]["point_origin"]
        self.assertAlmostEqual(point_nyc[0], -74.006, places=3)
        self.assertAlmostEqual(point_nyc[1], 40.7128, places=4)

        # Data type validation - Geo types should be object
        expected_geo_types = {
            "row_id": "uint8",
            "point_origin": "object",
            "point_sf": "object",
            "ring_square": "object"
        }

        for col, expected_type in expected_geo_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"{col} dtype should be {expected_type}, got {actual_type}")

    def test_nested_geo_types(self):
        """Test Geo types nested in tuples"""
        ret = self.session.query("""
            SELECT * FROM (
            SELECT
                1 as row_id,
                ((0.0, 0.0)::Point, (1.0, 1.0)::Point) as tuple_two_points,
                ((37.7749, -122.4194)::Point, 'San Francisco') as tuple_point_with_name,
                ([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]::Ring, 'square') as tuple_ring_with_name
            UNION ALL
            SELECT
                2 as row_id,
                ((-74.006, 40.7128)::Point, (51.5074, -0.1278)::Point) as tuple_two_points,
                ((40.7589, -73.9851)::Point, 'Times Square') as tuple_point_with_name,
                ([(-1.0, -1.0), (0.0, -1.0), (0.0, 0.0), (-1.0, 0.0), (-1.0, -1.0)]::Ring, 'negative_square') as tuple_ring_with_name
            )
            ORDER BY row_id
        """, "DataFrame")

        # Test nested geo in tuples
        self.assertEqual(ret.iloc[0]["row_id"], 1)

        tuple_two_points = ret.iloc[0]["tuple_two_points"]
        self.assertIsInstance(tuple_two_points, np.ndarray, "Should be tuple")
        self.assertEqual(len(tuple_two_points), 2, "Should have 2 points")
        self.assertEqual(tuple_two_points[0], (0.0, 0.0))
        self.assertEqual(tuple_two_points[1], (1.0, 1.0))

        tuple_point_with_name = ret.iloc[0]["tuple_point_with_name"]
        self.assertEqual(len(tuple_point_with_name), 2)
        self.assertAlmostEqual(tuple_point_with_name[0][0], 37.7749, places=4)
        self.assertEqual(tuple_point_with_name[1], 'San Francisco')

        tuple_ring_with_name = ret.iloc[0]["tuple_ring_with_name"]
        self.assertEqual(len(tuple_ring_with_name), 2)
        self.assertEqual(len(tuple_ring_with_name[0]), 5)  # Ring with 5 points
        self.assertEqual(tuple_ring_with_name[1], 'square')

    def test_simple_aggregate_function_types(self):
        """Test SimpleAggregateFunction types with sum and max functions"""
        # Create a table using SimpleAggregateFunction
        self.session.query("DROP TABLE IF EXISTS test_simple_agg")
        self.session.query("""
            CREATE TABLE IF NOT EXISTS test_simple_agg (
                id UInt32,
                sum_val SimpleAggregateFunction(sum, UInt64),
                max_val SimpleAggregateFunction(max, Float64)
            ) ENGINE = AggregatingMergeTree() ORDER BY id
        """)

        # Insert test data
        self.session.query("""
            INSERT INTO test_simple_agg VALUES
            (1, 100, 10.5),
            (1, 200, 20.3),
            (2, 50, 5.7),
            (2, 150, 15.2)
        """)

        # Query the data
        ret = self.session.query("""
            SELECT
                id,
                sum(sum_val) as total_sum,
                max(max_val) as total_max
            FROM test_simple_agg
            GROUP BY id
            ORDER BY id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Verify aggregated results
        self.assertEqual(ret.iloc[0]["id"], 1)
        self.assertEqual(ret.iloc[0]["total_sum"], 300)  # 100 + 200
        self.assertAlmostEqual(ret.iloc[0]["total_max"], 20.3, places=1)

        self.assertEqual(ret.iloc[1]["id"], 2)
        self.assertEqual(ret.iloc[1]["total_sum"], 200)  # 50 + 150
        self.assertAlmostEqual(ret.iloc[1]["total_max"], 15.2, places=1)

        self.session.query("DROP TABLE IF EXISTS test_simple_agg")

    def test_aggregate_function_types(self):
        """Test AggregateFunction types with uniq and avgState functions"""
        # Create a table using AggregateFunction
        self.session.query("DROP TABLE IF EXISTS test_agg_func")
        self.session.query("""
            CREATE TABLE IF NOT EXISTS test_agg_func (
                id UInt32,
                uniq_state AggregateFunction(uniq, String),
                avg_state AggregateFunction(avgState, Float64)
            ) ENGINE = AggregatingMergeTree() ORDER BY id
        """)

        # Insert aggregate states
        self.session.query("""
            INSERT INTO test_agg_func
            SELECT
                1 as id,
                uniqState('a') as uniq_state,
                avgState(10.5) as avg_state
            UNION ALL
            SELECT
                1 as id,
                uniqState('b') as uniq_state,
                avgState(20.3) as avg_state
            UNION ALL
            SELECT
                2 as id,
                uniqState('c') as uniq_state,
                avgState(5.7) as avg_state
        """)

        # Query finalized results
        ret = self.session.query("""
            SELECT
                id,
                uniqMerge(uniq_state) as unique_count,
                avgMerge(avg_state) as average_value
            FROM test_agg_func
            GROUP BY id
            ORDER BY id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Verify aggregated results
        self.assertEqual(ret.iloc[0]["id"], 1)
        self.assertEqual(ret.iloc[0]["unique_count"], 2)  # 'a' and 'b'
        self.assertAlmostEqual(ret.iloc[0]["average_value"], 15.4, places=1)  # (10.5 + 20.3) / 2

        self.assertEqual(ret.iloc[1]["id"], 2)
        self.assertEqual(ret.iloc[1]["unique_count"], 1)  # 'c'
        self.assertAlmostEqual(ret.iloc[1]["average_value"], 5.7, places=1)

        self.session.query("DROP TABLE IF EXISTS test_agg_func")

    def test_low_cardinality_types(self):
        """Test LowCardinality types with String and various integer types"""
        ret = self.session.query("""
            SELECT * FROM (
            SELECT
                1 as row_id,
                toLowCardinality('red') as lc_string,
                toLowCardinality(toInt8(1)) as lc_int8,
                toLowCardinality(toInt32(100)) as lc_int32,
                toLowCardinality(toUInt16(65535)) as lc_uint16,
                toLowCardinality(toFloat32(3.14)) as lc_float32
            UNION ALL
            SELECT
                2 as row_id,
                toLowCardinality('blue') as lc_string,
                toLowCardinality(toInt8(2)) as lc_int8,
                toLowCardinality(toInt32(200)) as lc_int32,
                toLowCardinality(toUInt16(32768)) as lc_uint16,
                toLowCardinality(toFloat32(2.71)) as lc_float32
            UNION ALL
            SELECT
                3 as row_id,
                toLowCardinality('green') as lc_string,
                toLowCardinality(toInt8(1)) as lc_int8,  -- Repeat value to show low cardinality
                toLowCardinality(toInt32(100)) as lc_int32,  -- Repeat value
                toLowCardinality(toUInt16(65535)) as lc_uint16,  -- Repeat value
                toLowCardinality(toFloat32(3.14)) as lc_float32  -- Repeat value
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test values
        self.assertEqual(ret.iloc[0]["lc_string"], 'red')
        self.assertEqual(ret.iloc[0]["lc_int8"], 1)
        self.assertEqual(ret.iloc[0]["lc_int32"], 100)
        self.assertEqual(ret.iloc[0]["lc_uint16"], 65535)
        self.assertAlmostEqual(ret.iloc[0]["lc_float32"], 3.14, places=2)

        self.assertEqual(ret.iloc[1]["lc_string"], 'blue')
        self.assertEqual(ret.iloc[1]["lc_int8"], 2)
        self.assertEqual(ret.iloc[1]["lc_int32"], 200)
        self.assertEqual(ret.iloc[1]["lc_uint16"], 32768)
        self.assertAlmostEqual(ret.iloc[1]["lc_float32"], 2.71, places=2)

        # Test repeated values (showing low cardinality)
        self.assertEqual(ret.iloc[2]["lc_string"], 'green')
        self.assertEqual(ret.iloc[2]["lc_int8"], 1)  # Same as row 0
        self.assertEqual(ret.iloc[2]["lc_int32"], 100)  # Same as row 0

        # Data type validation - LowCardinality should typically be object for strings, specific types for numbers
        expected_lc_types = {
            "row_id": "uint8",
            "lc_string": "object",
            "lc_int8": "int8",
            "lc_int32": "int32",
            "lc_uint16": "uint16",
            "lc_float32": "float32"
        }

        for col, expected_type in expected_lc_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"{col} dtype should be {expected_type}, got {actual_type}")

    def test_nullable_types(self):
        """Test Nullable(T) with comprehensive type coverage from both test files"""
        ret = self.session.query("""
            SELECT * FROM (
            SELECT
                1 as row_id,
                -- Integer types
                toNullable(toInt8(127)) as nullable_int8,
                toNullable(toInt32(-2147483648)) as nullable_int32,
                toNullable(toInt64(9223372036854775807)) as nullable_int64,
                toNullable(toUInt16(65535)) as nullable_uint16,
                toNullable(toUInt64(18446744073709551615)) as nullable_uint64,
                -- Float types
                toNullable(toFloat32(3.14159)) as nullable_float32,
                toNullable(toFloat64(2.718281828)) as nullable_float64,
                -- Decimal types
                toNullable(toDecimal32(123.45, 2)) as nullable_decimal32,
                toNullable(toDecimal64(987654.321, 3)) as nullable_decimal64,
                -- String types
                toNullable('Hello World') as nullable_string,
                toNullable(toFixedString('Fixed', 5)) as nullable_fixed_string,
                -- Date/Time types
                toNullable(toDate('2023-12-25')) as nullable_date,
                toNullable(toDateTime('2023-12-25 18:30:45', 'Asia/Shanghai')) as nullable_datetime,
                toNullable(toDateTime64('2023-12-25 18:30:45.123', 3, 'Asia/Shanghai')) as nullable_datetime64,
                -- Enum types
                toNullable(CAST('red', 'Enum8(''red''=1, ''green''=2, ''blue''=3)')) as nullable_enum8,
                -- UUID type
                toNullable(toUUID('550e8400-e29b-41d4-a716-446655440000')) as nullable_uuid,
                -- IPv4/IPv6 types
                toNullable(toIPv4('192.168.1.1')) as nullable_ipv4,
                toNullable(toIPv6('2001:0db8:85a3:0000:0000:8a2e:0370:7334')) as nullable_ipv6,
                -- Bool type
                toNullable(true) as nullable_bool,
                -- JSON type
                toNullable(CAST('{"name": "Alice", "age": 30, "active": true}', 'JSON')) as nullable_json,
                -- Interval types
                toNullable(INTERVAL 3 YEAR) as nullable_interval_year,
                toNullable(INTERVAL 6 MONTH) as nullable_interval_month,
                toNullable(INTERVAL 15 DAY) as nullable_interval_day,
                toNullable(INTERVAL 2 HOUR) as nullable_interval_hour
            UNION ALL
            SELECT
                2 as row_id,
                -- Mix of NULL and non-NULL values
                NULL as nullable_int8,
                toNullable(toInt32(2147483647)) as nullable_int32,
                NULL as nullable_int64,
                toNullable(toUInt16(32768)) as nullable_uint16,
                NULL as nullable_uint64,
                toNullable(toFloat32(-3.14159)) as nullable_float32,
                NULL as nullable_float64,
                toNullable(toDecimal32(-456.78, 2)) as nullable_decimal32,
                NULL as nullable_decimal64,
                NULL as nullable_string,
                toNullable(toFixedString('NULL ', 5)) as nullable_fixed_string,
                NULL as nullable_date,
                toNullable(toDateTime('2024-01-01 00:00:00', 'Asia/Shanghai')) as nullable_datetime,
                NULL as nullable_datetime64,
                toNullable(CAST('blue', 'Enum8(''red''=1, ''green''=2, ''blue''=3)')) as nullable_enum8,
                NULL as nullable_uuid,
                toNullable(toIPv4('10.0.0.1')) as nullable_ipv4,
                NULL as nullable_ipv6,
                toNullable(false) as nullable_bool,
                toNullable(CAST('{"name": "Bob", "age": 25, "active": false}', 'JSON')) as nullable_json,
                -- Interval types
                toNullable(INTERVAL 1 YEAR) as nullable_interval_year,
                NULL as nullable_interval_month,
                toNullable(INTERVAL 7 DAY) as nullable_interval_day,
                NULL as nullable_interval_hour
            UNION ALL
            SELECT
                3 as row_id,
                -- All NULL values to test NULL handling
                NULL as nullable_int8,
                NULL as nullable_int32,
                NULL as nullable_int64,
                NULL as nullable_uint16,
                NULL as nullable_uint64,
                NULL as nullable_float32,
                NULL as nullable_float64,
                NULL as nullable_decimal32,
                NULL as nullable_decimal64,
                NULL as nullable_string,
                NULL as nullable_fixed_string,
                NULL as nullable_date,
                NULL as nullable_datetime,
                NULL as nullable_datetime64,
                NULL as nullable_enum8,
                NULL as nullable_uuid,
                NULL as nullable_ipv4,
                NULL as nullable_ipv6,
                NULL as nullable_bool,
                NULL as nullable_json,
                -- Interval types
                NULL as nullable_interval_year,
                NULL as nullable_interval_month,
                NULL as nullable_interval_day,
                NULL as nullable_interval_hour
            )
            ORDER BY row_id
        """, "DataFrame")

        for col in ret.columns:
            print(f"{col}: {ret.dtypes[col]} (actual value: {ret.iloc[0][col]}, Python type: {type(ret.iloc[0][col])})")

        # Test first row - all non-NULL values
        self.assertEqual(ret.iloc[0]["row_id"], 1)

        # Integer types
        self.assertEqual(ret.iloc[0]["nullable_int8"], 127)
        self.assertEqual(ret.iloc[0]["nullable_int32"], -2147483648)
        self.assertEqual(ret.iloc[0]["nullable_int64"], 9223372036854775807)
        self.assertEqual(ret.iloc[0]["nullable_uint16"], 65535)
        self.assertEqual(ret.iloc[0]["nullable_uint64"], 18446744073709551615)

        # Float types
        self.assertAlmostEqual(ret.iloc[0]["nullable_float32"], 3.14159, places=5)
        self.assertAlmostEqual(ret.iloc[0]["nullable_float64"], 2.718281828, places=9)

        # Decimal types
        self.assertAlmostEqual(ret.iloc[0]["nullable_decimal32"], 123.45, places=2)
        self.assertAlmostEqual(ret.iloc[0]["nullable_decimal64"], 987654.321, places=3)

        # String types
        self.assertEqual(ret.iloc[0]["nullable_string"], 'Hello World')
        self.assertEqual(ret.iloc[0]["nullable_fixed_string"], 'Fixed')

        # Date/Time types
        nullable_date = ret.iloc[0]["nullable_date"]
        self.assertIsInstance(nullable_date, pd.Timestamp)
        self.assertEqual(nullable_date.date(), date(2023, 12, 25))

        nullable_datetime = ret.iloc[0]["nullable_datetime"]
        self.assertIsInstance(nullable_datetime, pd.Timestamp)

        # Check if timezone info is preserved (may be naive depending on implementation)
        if nullable_datetime.tz is not None:
            self.assertEqual(nullable_datetime, pd.Timestamp('2023-12-25 18:30:45', tz='Asia/Shanghai'))
        else:
            # If timezone is lost, just check the datetime value without timezone
            self.assertEqual(nullable_datetime, pd.Timestamp('2023-12-25 10:30:45'))

        nullable_datetime64 = ret.iloc[0]["nullable_datetime64"]
        self.assertIsInstance(nullable_datetime64, pd.Timestamp)
        # Check if timezone info is preserved for DateTime64
        if nullable_datetime64.tz is not None:
            self.assertEqual(nullable_datetime64, pd.Timestamp('2023-12-25 18:30:45.123', tz='Asia/Shanghai'))
        else:
            # If timezone is lost, just check the datetime value without timezone
            self.assertEqual(nullable_datetime64, pd.Timestamp('2023-12-25 10:30:45.123'))

        # Enum, UUID, IP types
        self.assertEqual(ret.iloc[0]["nullable_enum8"], 'red')
        self.assertIsInstance(ret.iloc[0]["nullable_uuid"], uuid.UUID)
        self.assertEqual(ret.iloc[0]["nullable_uuid"], uuid.UUID('550e8400-e29b-41d4-a716-446655440000'))

        # IP types
        self.assertEqual(str(ret.iloc[0]["nullable_ipv4"]), '192.168.1.1')
        ipv6_str = str(ret.iloc[0]["nullable_ipv6"])
        self.assertIn('2001', ipv6_str)

        # Bool type
        self.assertEqual(ret.iloc[0]["nullable_bool"], True)

        # JSON type
        json_val = ret.iloc[0]["nullable_json"]
        self.assertIsInstance(json_val, dict)
        self.assertEqual(json_val["name"], "Alice")
        self.assertEqual(json_val["age"], 30)
        self.assertEqual(json_val["active"], True)

        # Interval types
        self.assertEqual(ret.iloc[0]["nullable_interval_year"], timedelta(days=3*365))
        self.assertEqual(ret.iloc[0]["nullable_interval_month"], timedelta(days=6*30))
        self.assertEqual(ret.iloc[0]["nullable_interval_day"], timedelta(days=15))
        self.assertEqual(ret.iloc[0]["nullable_interval_hour"], timedelta(hours=2))

        # Test second row - mix of NULL and non-NULL values
        self.assertEqual(ret.iloc[1]["row_id"], 2)

        # Test NULL values
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_int8"]), "Should be NULL/NaN")
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_int64"]), "Should be NULL/NaN")
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_uint64"]), "Should be NULL/NaN")
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_float64"]), "Should be NULL/NaN")
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_string"]), "Should be NULL/NaN")

        # Test non-NULL values in second row
        self.assertEqual(ret.iloc[1]["nullable_int32"], 2147483647)
        self.assertEqual(ret.iloc[1]["nullable_uint16"], 32768)
        self.assertAlmostEqual(ret.iloc[1]["nullable_float32"], -3.14159, places=5)
        self.assertEqual(ret.iloc[1]["nullable_fixed_string"], 'NULL ')
        self.assertEqual(ret.iloc[1]["nullable_bool"], False)

        # JSON type for second row
        json_val_2 = ret.iloc[1]["nullable_json"]
        self.assertIsInstance(json_val_2, dict)
        self.assertEqual(json_val_2["name"], "Bob")
        self.assertEqual(json_val_2["age"], 25)
        self.assertEqual(json_val_2["active"], False)

        # Interval types for second row
        self.assertEqual(ret.iloc[1]["nullable_interval_year"], timedelta(days=1*365))
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_interval_month"]), "Should be NULL/NaN")
        self.assertEqual(ret.iloc[1]["nullable_interval_day"], timedelta(days=7))
        self.assertTrue(pd.isna(ret.iloc[1]["nullable_interval_hour"]), "Should be NULL/NaN")

        # Test third row - all NULL values
        self.assertEqual(ret.iloc[2]["row_id"], 3)

        # All nullable columns should be NULL in third row
        nullable_columns = [col for col in ret.columns if col.startswith('nullable_')]
        for col in nullable_columns:
            value = ret.iloc[2][col]
            self.assertTrue(pd.isna(value) or value is None, f"{col} should be NULL/NaN in row 3")

        # Data type validation - Nullable types should maintain their underlying types or be object
        expected_nullable_types = {
            "row_id": "uint8",
            "nullable_int8": "Int8",
            "nullable_int32": "Int32",
            "nullable_int64": "Int64",
            "nullable_uint16": "UInt16",
            "nullable_uint64": "UInt64",
            "nullable_float32": "float32",
            "nullable_float64": "float64",
            "nullable_decimal32": "Float64",
            "nullable_decimal64": "Float64",
            "nullable_string": "object",
            "nullable_fixed_string": "object",
            "nullable_date": "datetime64[s]",
            "nullable_datetime": "datetime64[s, Asia/Shanghai]",
            "nullable_datetime64": "datetime64[ns, Asia/Shanghai]",
            "nullable_enum8": "object",
            "nullable_uuid": "object",
            "nullable_ipv4": "object",
            "nullable_ipv6": "object",
            "nullable_bool": "boolean",
            "nullable_json": "object",
            "nullable_interval_year": "timedelta64[s]",
            "nullable_interval_month": "timedelta64[s]",
            "nullable_interval_day": "timedelta64[s]",
            "nullable_interval_hour": "timedelta64[s]",
        }

        for col, expected_type in expected_nullable_types.items():
            actual_type = str(ret.dtypes[col])
            self.assertEqual(actual_type, expected_type, f"{col} dtype should be {expected_type}, got {actual_type}")

    def test_datetime_timezone_naive(self):
        """Test that timezone-naive datetime matches pandas behavior"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'dt': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 00:00:01', '2020-01-01 00:00:02'])
        })
        result = self.session.query("SELECT * FROM Python(df)", 'DataFrame')
        self.assertEqual(df['dt'].tolist(), result['dt'].tolist())
        self.assertEqual(df['dt'].dtype, result['dt'].dtype)

    def test_float_with_nan(self):
        """Test that float64 with NaN preserves dtype and values"""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "a": [1.0, 2.0, np.nan, 4.0]})
        result = self.session.query("SELECT * FROM Python(df) ORDER BY id", 'DataFrame')
        self.assertEqual(df['a'].dtype, result['a'].dtype)
        self.assertEqual(df['a'].isna().tolist(), result['a'].isna().tolist())

    def test_integer_with_none(self):
        """Test that integer-like column with None preserves dtype and values"""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "a": [1, 2, None, 4]})
        result = self.session.query("SELECT * FROM Python(df) ORDER BY id", 'DataFrame')
        self.assertEqual(df['a'].dtype, result['a'].dtype)
        self.assertEqual(df['a'].isna().tolist(), result['a'].isna().tolist())

    def test_binary_blob_and_string(self):
        result = self.session.query("SELECT 'hello' AS str_col, toFixedString('world', 5) AS fixed_str", 'DataFrame')
        self.assertIsInstance(result['str_col'].iloc[0], str)
        self.assertEqual(result['str_col'].iloc[0], 'hello')
        self.assertIsInstance(result['fixed_str'].iloc[0], str)
        self.assertEqual(result['fixed_str'].iloc[0], 'world')

        result = self.session.query(
            "SELECT unhex('0186000000') AS blob_col, toFixedString(unhex('ff00fe01'), 4) AS fixed_blob",
            'DataFrame'
        )
        self.assertIsInstance(result['blob_col'].iloc[0], bytearray)
        self.assertEqual(bytes(result['blob_col'].iloc[0]), b'\x01\x86\x00\x00\x00')
        self.assertIsInstance(result['fixed_blob'].iloc[0], bytearray)
        self.assertEqual(bytes(result['fixed_blob'].iloc[0]), b'\xff\x00\xfe\x01')


    def test_timedelta_input_from_pandas(self):
        """Test timedelta64 input from pandas DataFrame - aligned with pandas behavior"""
        # Create pandas DataFrame with timedelta columns
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'td_ns': pd.to_timedelta(['1 day', '2 hours', '30 minutes', '10 seconds']),
            'td_with_null': pd.to_timedelta(['1 day', None, '3 hours', '45 minutes']),
        })

        print("\nInput DataFrame:")
        print(df)
        print("\nInput dtypes:")
        print(df.dtypes)

        # Query using chDB - should handle timedelta input
        result = self.session.query("SELECT * FROM Python(df) ORDER BY id", 'DataFrame')

        print("\nResult DataFrame:")
        print(result)
        print("\nResult dtypes:")
        print(result.dtypes)

        # Verify row count
        self.assertEqual(len(result), 4)

        # Verify timedelta values are preserved
        # Note: ClickHouse Interval type may convert to different precision
        self.assertEqual(result.iloc[0]['id'], 1)
        self.assertEqual(result.iloc[1]['id'], 2)
        self.assertEqual(result.iloc[2]['id'], 3)
        self.assertEqual(result.iloc[3]['id'], 4)

        # Verify timedelta column values (comparing as timedelta)
        expected_td = [
            pd.Timedelta('1 day'),
            pd.Timedelta('2 hours'),
            pd.Timedelta('30 minutes'),
            pd.Timedelta('10 seconds'),
        ]

        for i, expected in enumerate(expected_td):
            actual = result.iloc[i]['td_ns']
            self.assertEqual(actual, expected, f"Row {i}: expected {expected}, got {actual}")

        # Verify NULL handling in timedelta column
        self.assertEqual(result.iloc[0]['td_with_null'], pd.Timedelta('1 day'))
        self.assertTrue(pd.isna(result.iloc[1]['td_with_null']), "Row 1 should be NULL")
        self.assertEqual(result.iloc[2]['td_with_null'], pd.Timedelta('3 hours'))
        self.assertEqual(result.iloc[3]['td_with_null'], pd.Timedelta('45 minutes'))

    def test_timedelta_various_precisions(self):
        """Test timedelta with different precisions from pandas"""
        # pandas timedelta64[ns] is the default
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'days': pd.to_timedelta([1, 2, 3, 4, 5], unit='D'),
            'hours': pd.to_timedelta([1, 2, 3, 4, 5], unit='h'),
            'minutes': pd.to_timedelta([1, 2, 3, 4, 5], unit='m'),
            'seconds': pd.to_timedelta([1, 2, 3, 4, 5], unit='s'),
            'milliseconds': pd.to_timedelta([1, 2, 3, 4, 5], unit='ms'),
        })

        print("\nInput DataFrame with various timedelta precisions:")
        print(df)
        print("\nInput dtypes:")
        print(df.dtypes)

        result = self.session.query("SELECT * FROM Python(df) ORDER BY id", 'DataFrame')

        print("\nResult DataFrame:")
        print(result)
        print("\nResult dtypes:")
        print(result.dtypes)

        # Verify values
        self.assertEqual(len(result), 5)

        # Check days column
        for i in range(5):
            expected_days = pd.Timedelta(days=i + 1)
            self.assertEqual(result.iloc[i]['days'], expected_days)

        # Check hours column
        for i in range(5):
            expected_hours = pd.Timedelta(hours=i + 1)
            self.assertEqual(result.iloc[i]['hours'], expected_hours)

        # Check seconds column
        for i in range(5):
            expected_seconds = pd.Timedelta(seconds=i + 1)
            self.assertEqual(result.iloc[i]['seconds'], expected_seconds)

    def test_timedelta_arithmetic_query(self):
        """Test timedelta values can be used in SQL arithmetic"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'duration': pd.to_timedelta(['1 hour', '2 hours', '3 hours']),
        })

        # Query that uses timedelta in SQL
        result = self.session.query("""
            SELECT id, duration FROM Python(df) ORDER BY id
        """, 'DataFrame')

        print("\nResult from arithmetic query:")
        print(result)

        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['duration'], pd.Timedelta('1 hour'))
        self.assertEqual(result.iloc[1]['duration'], pd.Timedelta('2 hours'))
        self.assertEqual(result.iloc[2]['duration'], pd.Timedelta('3 hours'))


if __name__ == '__main__':
    unittest.main()
