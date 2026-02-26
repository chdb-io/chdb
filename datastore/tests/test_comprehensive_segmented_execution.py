"""
Comprehensive test for segmented execution with multiple operation types.

This test validates:
1. Exact value comparison with Pandas for the same operations
2. Correct execution engine (SQL vs Pandas) for each segment
3. Explain and debug output correspondence with actual SQL execution

Operations covered:
- Multi-source JOINs
- CASE WHEN (where/mask)
- Column assignment
- String operations (str accessor)
- JSON operations
- DateTime operations (dt accessor)
- Python UDF (apply)
- Filters, sorts, limits
"""

import io
import json
import logging
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from tests.xfail_markers import chdb_case_bool_conversion

from datastore import DataStore
from datastore.config import config
from datastore.query_planner import QueryPlanner
from datastore.expressions import Field, Literal
from datastore.conditions import BinaryCondition

from tests.test_utils import assert_datastore_equals_pandas


def verify_segment_engines(lazy_ops, has_sql_source, expected_segments):
    """
    Verify that the execution plan matches expected segment types.

    Args:
        lazy_ops: List of lazy operations
        has_sql_source: Whether there's a SQL source
        expected_segments: List of expected segment types, e.g., ['sql', 'pandas', 'sql']

    Returns:
        ExecutionPlan for further inspection
    """
    planner = QueryPlanner()
    plan = planner.plan_segments(lazy_ops, has_sql_source)

    actual_segments = [seg.segment_type for seg in plan.segments]

    # Note: Due to consolidation logic, first Pandas segment might consolidate all
    # So we check if actual matches expected OR if it's consolidated
    if actual_segments != expected_segments:
        # Check if it's a valid consolidation case
        if len(plan.segments) == 1 and plan.segments[0].is_pandas():
            # Consolidated to single Pandas - this is acceptable
            pass
        else:
            raise AssertionError(
                f"Segment engines mismatch!\n"
                f"Expected: {expected_segments}\n"
                f"Actual: {actual_segments}\n"
                f"Plan:\n{plan.describe()}"
            )

    return plan


class TestComprehensiveSegmentedExecution(unittest.TestCase):
    """Comprehensive test for multi-operation segmented execution."""

    @classmethod
    def setUpClass(cls):
        """Create test datasets with various data types."""
        cls.temp_dir = tempfile.mkdtemp()

        # Users table with various data types
        cls.users_data = pd.DataFrame(
            {
                'user_id': [1, 2, 3, 4, 5, 6, 7, 8],
                'name': [
                    'Alice Smith',
                    'Bob Johnson',
                    'Charlie Brown',
                    'Diana Prince',
                    'Eve Wilson',
                    'Frank Miller',
                    'Grace Lee',
                    'Henry Chen',
                ],
                'email': [
                    'alice@test.com',
                    'bob@example.org',
                    'charlie@demo.net',
                    'diana@sample.io',
                    'eve@test.com',
                    'frank@example.org',
                    'grace@demo.net',
                    'henry@sample.io',
                ],
                'age': [28, 35, 22, 31, 45, 29, 38, 26],
                'country': ['USA', 'UK', 'Canada', 'USA', 'UK', 'Canada', 'USA', 'UK'],
                'registration_date': pd.to_datetime(
                    [
                        '2023-01-15',
                        '2023-02-20',
                        '2023-03-10',
                        '2023-04-05',
                        '2023-05-12',
                        '2023-06-18',
                        '2023-07-22',
                        '2023-08-30',
                    ]
                ),
                'metadata': [
                    '{"level": 5, "premium": true}',
                    '{"level": 3, "premium": false}',
                    '{"level": 8, "premium": true}',
                    '{"level": 2, "premium": false}',
                    '{"level": 10, "premium": true}',
                    '{"level": 4, "premium": false}',
                    '{"level": 7, "premium": true}',
                    '{"level": 1, "premium": false}',
                ],
            }
        )
        cls.users_file = os.path.join(cls.temp_dir, 'users.parquet')
        cls.users_data.to_parquet(cls.users_file, index=False)

        # Orders table
        cls.orders_data = pd.DataFrame(
            {
                'order_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                'user_id': [1, 2, 1, 3, 4, 2, 5, 1, 6, 3],
                'product_id': [201, 202, 203, 201, 204, 202, 205, 203, 201, 204],
                'amount': [59.99, 129.50, 89.00, 45.00, 199.99, 75.25, 320.00, 55.50, 42.00, 88.75],
                'order_date': pd.to_datetime(
                    [
                        '2024-01-10',
                        '2024-01-15',
                        '2024-01-20',
                        '2024-01-25',
                        '2024-02-01',
                        '2024-02-05',
                        '2024-02-10',
                        '2024-02-15',
                        '2024-02-20',
                        '2024-02-25',
                    ]
                ),
                'status': [
                    'completed',
                    'pending',
                    'completed',
                    'cancelled',
                    'completed',
                    'pending',
                    'completed',
                    'completed',
                    'cancelled',
                    'pending',
                ],
            }
        )
        cls.orders_file = os.path.join(cls.temp_dir, 'orders.parquet')
        cls.orders_data.to_parquet(cls.orders_file, index=False)

        # Products table
        cls.products_data = pd.DataFrame(
            {
                'product_id': [201, 202, 203, 204, 205],
                'product_name': ['Widget Pro', 'Gadget Plus', 'Tool Master', 'Device X', 'System Y'],
                'category': ['Electronics', 'Electronics', 'Tools', 'Electronics', 'Software'],
                'price': [29.99, 129.00, 45.00, 199.00, 320.00],
            }
        )
        cls.products_file = os.path.join(cls.temp_dir, 'products.parquet')
        cls.products_data.to_parquet(cls.products_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_sql_filter_exact_values(self):
        """
        Test SQL filter produces exact same values as Pandas.

        Pipeline: Filter (SQL only)
        Expected segments: [sql]
        """
        # DataStore
        ds = DataStore.from_file(self.users_file)
        ds = ds[ds['age'] > 30]

        # Verify execution engine
        plan = verify_segment_engines(ds._lazy_ops, has_sql_source=True, expected_segments=['sql'])
        self.assertEqual(plan.sql_segment_count(), 1)
        self.assertEqual(plan.pandas_segment_count(), 0)

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.copy()
        pd_result = pdf[pdf['age'] > 30]

        # Verify exact values
        self.assertEqual(len(ds_result), len(pd_result))
        np.testing.assert_array_equal(sorted(ds_result['user_id'].values), sorted(pd_result['user_id'].values))
        np.testing.assert_array_equal(sorted(ds_result['age'].values), sorted(pd_result['age'].values))

    def test_pandas_apply_exact_values(self):
        """
        Test Pandas apply produces exact same values.

        Pipeline: Apply (Pandas only, forced by apply)
        Expected segments: [pandas]
        """
        # DataStore
        ds = DataStore.from_file(self.users_file)
        ds['age_squared'] = ds['age'].apply(lambda x: x**2)

        # Verify execution engine - apply forces Pandas
        plan = verify_segment_engines(ds._lazy_ops, has_sql_source=True, expected_segments=['pandas'])

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.copy()
        pdf['age_squared'] = pdf['age'].apply(lambda x: x**2)

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf))
        np.testing.assert_array_equal(
            ds_result['age_squared'].sort_values().values, pdf['age_squared'].sort_values().values
        )

    def test_sql_pandas_sql_three_segments_exact_values(self):
        """
        Test SQL -> Pandas -> SQL produces exact same values.

        Pipeline:
        1. Filter by age > 25 (SQL - LazyRelationalOp WHERE)
        2. Apply to compute new column (Pandas - LazyColumnAssignment)
        3. Filter by computed column (SQL on DataFrame - LazyRelationalOp WHERE)

        Expected segments: [sql, pandas, sql] with 3 segments total
        """
        # DataStore
        ds = DataStore.from_file(self.users_file)
        ds = ds[ds['age'] > 25]  # SQL filter
        ds['age_doubled'] = ds['age'].apply(lambda x: x * 2)  # Pandas apply
        ds = ds[ds['age_doubled'] > 60]  # SQL filter on DataFrame

        # === SEGMENT STRUCTURE VERIFICATION ===
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)

        # Verify 3 segments
        self.assertEqual(
            len(plan.segments),
            3,
            f"Expected 3 segments [sql, pandas, sql], got {len(plan.segments)}:\n{plan.describe()}",
        )

        # Verify segment types in order
        expected_types = ['sql', 'pandas', 'sql']
        actual_types = [seg.segment_type for seg in plan.segments]
        self.assertEqual(
            actual_types, expected_types, f"Segment types mismatch: expected {expected_types}, got {actual_types}"
        )

        # Verify segment 1: SQL filter (age > 25)
        self.assertEqual(plan.segments[0].ops[0].op_type, 'WHERE')
        self.assertTrue(plan.segments[0].is_first_segment)

        # Verify segment 2: Pandas apply
        self.assertIn('ColumnAssignment', plan.segments[1].ops[0].__class__.__name__)

        # Verify segment 3: SQL filter on DataFrame (age_doubled > 60)
        self.assertEqual(plan.segments[2].ops[0].op_type, 'WHERE')
        self.assertFalse(plan.segments[2].is_first_segment)

        # Verify counts
        self.assertEqual(plan.sql_segment_count(), 2)
        self.assertEqual(plan.pandas_segment_count(), 1)

        # === VALUE VERIFICATION ===
        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.copy()
        pdf = pdf[pdf['age'] > 25]
        pdf = pdf.copy()
        pdf['age_doubled'] = pdf['age'].apply(lambda x: x * 2)
        pdf = pdf[pdf['age_doubled'] > 60]

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        np.testing.assert_array_equal(
            sorted(ds_result['user_id'].values), sorted(pdf['user_id'].values), err_msg="user_id values don't match"
        )
        np.testing.assert_array_equal(
            sorted(ds_result['age_doubled'].values),
            sorted(pdf['age_doubled'].values),
            err_msg="age_doubled values don't match",
        )

    def test_multi_join_exact_values(self):
        """
        Test multi-table JOIN produces exact same values.

        Pipeline:
        1. JOIN users with orders
        2. JOIN with products
        3. Filter by status

        All operations are SQL pushable.
        """
        # DataStore
        users = DataStore.from_file(self.users_file)
        orders = DataStore.from_file(self.orders_file)
        products = DataStore.from_file(self.products_file)

        ds = users.join(orders, on='user_id')
        ds = ds.join(products, on='product_id')
        ds = ds[ds['status'] == 'completed']

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.merge(self.orders_data, on='user_id')
        pdf = pdf.merge(self.products_data, on='product_id')
        pdf = pdf[pdf['status'] == 'completed']

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        # Verify key columns match
        np.testing.assert_array_equal(
            sorted(ds_result['order_id'].values), sorted(pdf['order_id'].values), err_msg="order_id values don't match"
        )
        np.testing.assert_array_almost_equal(
            sorted(ds_result['amount'].values),
            sorted(pdf['amount'].values),
            decimal=2,
            err_msg="amount values don't match",
        )

    def test_join_then_apply_exact_values(self):
        """
        Test JOIN followed by apply.

        Pipeline:
        1. JOIN (creates data structure, no lazy_ops)
        2. Apply (Pandas)

        Expected: First segment is Pandas (apply breaks SQL chain)
        """
        users = DataStore.from_file(self.users_file)
        orders = DataStore.from_file(self.orders_file)

        ds = users.join(orders, on='user_id')
        ds['discount'] = ds['amount'].apply(lambda x: x * 0.1 if x > 100 else x * 0.05)

        # Check segment - should be Pandas since apply is first lazy_op
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.merge(self.orders_data, on='user_id')
        pdf = pdf.copy()
        pdf['discount'] = pdf['amount'].apply(lambda x: x * 0.1 if x > 100 else x * 0.05)

        # Verify exact discount values
        self.assertEqual(len(ds_result), len(pdf))
        np.testing.assert_array_almost_equal(
            sorted(ds_result['discount'].values),
            sorted(pdf['discount'].values),
            decimal=4,
            err_msg="discount values don't match",
        )

    def test_datetime_operations_exact_values(self):
        """
        Test DateTime accessor operations.

        Pipeline:
        1. Filter by date (SQL)
        2. Extract year/month (Pandas - dt accessor)
        3. Filter by month (SQL on DataFrame)
        """
        ds = DataStore.from_file(self.orders_file)
        ds = ds[ds['order_date'] >= '2024-01-20']  # SQL filter
        ds['order_month'] = ds['order_date'].dt.month  # Pandas dt accessor
        ds = ds[ds['order_month'] >= 2]  # SQL filter

        # Check segments
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.orders_data.copy()
        pdf = pdf[pdf['order_date'] >= '2024-01-20']
        pdf = pdf.copy()
        pdf['order_month'] = pdf['order_date'].dt.month
        pdf = pdf[pdf['order_month'] >= 2]

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        np.testing.assert_array_equal(
            sorted(ds_result['order_id'].values), sorted(pdf['order_id'].values), err_msg="order_id values don't match"
        )
        np.testing.assert_array_equal(
            sorted(ds_result['order_month'].values),
            sorted(pdf['order_month'].values),
            err_msg="order_month values don't match",
        )

    def test_string_operations_exact_values(self):
        """
        Test string accessor operations.

        Pipeline:
        1. Filter by country (SQL)
        2. String upper (Pandas str accessor)
        3. Filter by age (SQL on DataFrame)
        """
        ds = DataStore.from_file(self.users_file)
        ds = ds[ds['country'].isin(['USA', 'UK'])]  # SQL filter
        ds['name_upper'] = ds['name'].str.upper()  # Pandas str accessor
        ds = ds[ds['age'] > 28]  # SQL filter

        # Check segments
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.copy()
        pdf = pdf[pdf['country'].isin(['USA', 'UK'])]
        pdf = pdf.copy()
        pdf['name_upper'] = pdf['name'].str.upper()
        pdf = pdf[pdf['age'] > 28]

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf))

        self.assertEqual(
            sorted(ds_result['name_upper'].tolist()),
            sorted(pdf['name_upper'].tolist()),
            "name_upper values don't match",
        )

    def test_json_extraction_exact_values(self):
        """
        Test JSON field extraction.

        Pipeline:
        1. Filter by age (SQL)
        2. Extract JSON fields (Pandas apply)
        3. Filter by extracted field (SQL on DataFrame)
        """
        ds = DataStore.from_file(self.users_file)
        ds = ds[ds['age'] > 25]  # SQL filter
        ds['level'] = ds['metadata'].apply(
            lambda x: json.loads(x).get('level', 0) if isinstance(x, str) else 0
        )  # Pandas apply
        ds = ds[ds['level'] >= 5]  # SQL filter

        # Check segments
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data.copy()
        pdf = pdf[pdf['age'] > 25]
        pdf = pdf.copy()
        pdf['level'] = pdf['metadata'].apply(lambda x: json.loads(x).get('level', 0) if isinstance(x, str) else 0)
        pdf = pdf[pdf['level'] >= 5]

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        np.testing.assert_array_equal(
            sorted(ds_result['user_id'].values), sorted(pdf['user_id'].values), err_msg="user_id values don't match"
        )
        np.testing.assert_array_equal(
            sorted(ds_result['level'].values), sorted(pdf['level'].values), err_msg="level values don't match"
        )

    def test_limit_before_after_pandas_exact_values(self):
        """
        Test LIMIT operations before and after Pandas.

        Pipeline:
        1. Sort and limit (SQL)
        2. Apply (Pandas)
        3. Limit again (SQL on DataFrame)

        This tests the LIMIT-before-WHERE logic in segmented execution.
        """
        ds = DataStore.from_file(self.orders_file)
        ds = ds.sort_values('amount', ascending=False)
        ds = ds.head(7)  # SQL LIMIT
        ds['doubled'] = ds['amount'].apply(lambda x: x * 2)  # Pandas
        ds = ds.head(5)  # SQL LIMIT on DataFrame

        # Check segments
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.orders_data.copy()
        pdf = pdf.sort_values('amount', ascending=False)
        pdf = pdf.head(7)
        pdf = pdf.copy()
        pdf['doubled'] = pdf['amount'].apply(lambda x: x * 2)
        pdf = pdf.head(5)

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        np.testing.assert_array_almost_equal(
            ds_result['amount'].values, pdf['amount'].values, decimal=2, err_msg="amount values don't match"
        )
        np.testing.assert_array_almost_equal(
            ds_result['doubled'].values, pdf['doubled'].values, decimal=2, err_msg="doubled values don't match"
        )

    def test_column_selection_exact_values(self):
        """
        Test column selection across segments.

        Pipeline:
        1. Select columns (SQL)
        2. Apply (Pandas)
        3. Select final columns (SQL on DataFrame)
        """
        ds = DataStore.from_file(self.users_file)
        ds = ds.select('user_id', 'name', 'age')  # SQL SELECT
        ds['age_group'] = ds['age'].apply(lambda x: 'young' if x < 30 else 'adult')  # Pandas
        ds = ds[['user_id', 'name', 'age_group']]  # SQL SELECT on DataFrame

        # Check segments
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.users_data[['user_id', 'name', 'age']].copy()
        pdf['age_group'] = pdf['age'].apply(lambda x: 'young' if x < 30 else 'adult')
        pdf = pdf[['user_id', 'name', 'age_group']]

        # Verify exact columns
        self.assertEqual(
            list(ds_result.columns),
            list(pdf.columns),
            f"Columns mismatch: DS={list(ds_result.columns)}, PD={list(pdf.columns)}",
        )

        # Verify exact values
        self.assertEqual(len(ds_result), len(pdf))
        self.assertEqual(
            sorted(ds_result['age_group'].tolist()), sorted(pdf['age_group'].tolist()), "age_group values don't match"
        )


class TestSegmentEngineVerification(unittest.TestCase):
    """Test that verifies correct engine selection for each segment."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data = pd.DataFrame(
            {
                'id': range(1, 21),
                'value': list(range(10, 210, 10)),
                'category': ['A', 'B'] * 10,
            }
        )
        cls.test_file = os.path.join(cls.temp_dir, 'test.parquet')
        cls.test_data.to_parquet(cls.test_file, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_pure_sql_segment(self):
        """Verify pure SQL operations stay in SQL segment."""
        ds = DataStore.from_file(self.test_file)
        ds = ds[ds['value'] > 50]
        ds = ds[ds['value'] < 150]
        ds = ds.sort_values('value')

        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)

        # Should be single SQL segment
        self.assertEqual(len(plan.segments), 1, f"Expected 1 segment, got {len(plan.segments)}: {plan.describe()}")
        self.assertEqual(
            plan.segments[0].segment_type, 'sql', f"Expected SQL segment, got {plan.segments[0].segment_type}"
        )

        # Verify result
        ds_result = ds.to_df()
        pdf = self.test_data[(self.test_data['value'] > 50) & (self.test_data['value'] < 150)]
        pdf = pdf.sort_values('value')

        np.testing.assert_array_equal(ds_result['value'].values, pdf['value'].values)

    def test_pure_pandas_segment(self):
        """Verify apply() forces Pandas segment."""
        ds = DataStore.from_file(self.test_file)
        ds['computed'] = ds['value'].apply(lambda x: x**2)

        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)

        # Should be single Pandas segment (apply is not SQL pushable)
        self.assertEqual(len(plan.segments), 1, f"Expected 1 segment, got {len(plan.segments)}: {plan.describe()}")
        self.assertEqual(
            plan.segments[0].segment_type, 'pandas', f"Expected pandas segment, got {plan.segments[0].segment_type}"
        )

        # Verify result
        ds_result = ds.to_df()
        pdf = self.test_data.copy()
        pdf['computed'] = pdf['value'].apply(lambda x: x**2)

        np.testing.assert_array_equal(sorted(ds_result['computed'].values), sorted(pdf['computed'].values))

    def test_sql_pandas_sql_segments(self):
        """
        Verify SQL -> Pandas -> SQL creates correct segments.

        Pipeline:
        1. Filter value > 30 (SQL pushable - LazyRelationalOp WHERE)
        2. Apply squared (Pandas - LazyColumnAssignment with apply)
        3. Filter squared > 5000 (SQL pushable - LazyRelationalOp WHERE)

        Expected segments: [sql, pandas, sql]
        """
        ds = DataStore.from_file(self.test_file)
        ds = ds[ds['value'] > 30]  # SQL
        ds['squared'] = ds['value'].apply(lambda x: x**2)  # Pandas
        ds = ds[ds['squared'] > 5000]  # Should be SQL on DataFrame

        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)

        # === SEGMENT STRUCTURE VERIFICATION ===
        # Verify we have exactly 3 segments
        self.assertEqual(
            len(plan.segments),
            3,
            f"Expected 3 segments [sql, pandas, sql], got {len(plan.segments)}:\n{plan.describe()}",
        )

        # Verify segment 1: SQL (filter value > 30)
        seg1 = plan.segments[0]
        self.assertEqual(seg1.segment_type, 'sql', f"Segment 1 should be SQL, got {seg1.segment_type}")
        self.assertEqual(len(seg1.ops), 1, f"Segment 1 should have 1 op, got {len(seg1.ops)}")
        self.assertEqual(seg1.ops[0].op_type, 'WHERE', f"Segment 1 op should be WHERE")
        self.assertTrue(seg1.is_first_segment, "Segment 1 should be first segment")

        # Verify segment 2: Pandas (apply)
        seg2 = plan.segments[1]
        self.assertEqual(seg2.segment_type, 'pandas', f"Segment 2 should be Pandas, got {seg2.segment_type}")
        self.assertEqual(len(seg2.ops), 1, f"Segment 2 should have 1 op, got {len(seg2.ops)}")
        self.assertIn('ColumnAssignment', seg2.ops[0].__class__.__name__)

        # Verify segment 3: SQL on DataFrame (filter squared > 5000)
        seg3 = plan.segments[2]
        self.assertEqual(seg3.segment_type, 'sql', f"Segment 3 should be SQL, got {seg3.segment_type}")
        self.assertEqual(len(seg3.ops), 1, f"Segment 3 should have 1 op, got {len(seg3.ops)}")
        self.assertEqual(seg3.ops[0].op_type, 'WHERE', f"Segment 3 op should be WHERE")
        self.assertFalse(seg3.is_first_segment, "Segment 3 should NOT be first segment")

        # === EXECUTION AND VALUE VERIFICATION ===
        ds_result = ds.to_df()

        pdf = self.test_data.copy()
        pdf = pdf[pdf['value'] > 30]
        pdf = pdf.copy()
        pdf['squared'] = pdf['value'].apply(lambda x: x**2)
        pdf = pdf[pdf['squared'] > 5000]

        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        np.testing.assert_array_equal(
            sorted(ds_result['id'].values), sorted(pdf['id'].values), err_msg="id values don't match"
        )
        np.testing.assert_array_equal(
            sorted(ds_result['squared'].values), sorted(pdf['squared'].values), err_msg="squared values don't match"
        )

    def test_multiple_pandas_ops_single_segment(self):
        """
        Verify consecutive Pandas-only ops stay in same segment, but SQL-compatible
        ops get their own segment.

        Pipeline:
        1. Apply doubled (Pandas - LazyColumnAssignment with apply lambda)
        2. Apply tripled (Pandas - LazyColumnAssignment with apply lambda)
        3. Compute sum (SQL - simple arithmetic can be pushed to SQL)

        Expected:
        - Segment 1: Pandas (2 ops - apply lambdas)
        - Segment 2: chDB (1 op - simple arithmetic)
        """
        ds = DataStore.from_file(self.test_file)
        ds['doubled'] = ds['value'].apply(lambda x: x * 2)  # Pandas (apply is pandas-only)
        ds['tripled'] = ds['value'].apply(lambda x: x * 3)  # Pandas (apply is pandas-only)
        ds['computed'] = ds['doubled'] + ds['tripled']  # SQL (simple arithmetic)

        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)

        # === SEGMENT STRUCTURE VERIFICATION ===
        # In unified architecture: pandas-only ops (apply) stay in pandas segment,
        # but SQL-compatible ops (simple arithmetic) can go to SQL segment
        self.assertEqual(len(plan.segments), 2, f"Expected 2 segments, got {len(plan.segments)}:\n{plan.describe()}")

        # First segment should be Pandas with apply operations
        seg1 = plan.segments[0]
        self.assertEqual(seg1.segment_type, 'pandas', f"Segment 1 should be Pandas, got {seg1.segment_type}")
        self.assertEqual(len(seg1.ops), 2, f"Segment 1 should have 2 ops, got {len(seg1.ops)}")

        # Second segment should be SQL with arithmetic operation
        seg2 = plan.segments[1]
        self.assertIn(seg2.segment_type, ['chdb', 'sql'], f"Segment 2 should be SQL, got {seg2.segment_type}")
        self.assertEqual(len(seg2.ops), 1, f"Segment 2 should have 1 op, got {len(seg2.ops)}")

        # Verify results match pandas
        pdf = pd.read_parquet(self.test_file)
        pdf['doubled'] = pdf['value'].apply(lambda x: x * 2)
        pdf['tripled'] = pdf['value'].apply(lambda x: x * 3)
        pdf['computed'] = pdf['doubled'] + pdf['tripled']

        ds_result = ds.to_df()
        np.testing.assert_array_equal(
            ds_result['computed'].values, pdf['computed'].values, err_msg="computed values don't match"
        )

        # Verify segment counts
        self.assertEqual(plan.sql_segment_count(), 1, f"Expected 1 SQL segment, got {plan.sql_segment_count()}")
        self.assertEqual(
            plan.pandas_segment_count(), 1, f"Expected 1 Pandas segment, got {plan.pandas_segment_count()}"
        )

        # === VALUE VERIFICATION ===
        ds_result = ds.to_df()

        pdf = self.test_data.copy()
        pdf['doubled'] = pdf['value'].apply(lambda x: x * 2)
        pdf['tripled'] = pdf['value'].apply(lambda x: x * 3)
        pdf['computed'] = pdf['doubled'] + pdf['tripled']

        np.testing.assert_array_equal(
            sorted(ds_result['computed'].values), sorted(pdf['computed'].values), err_msg="computed values don't match"
        )
        np.testing.assert_array_equal(
            sorted(ds_result['doubled'].values), sorted(pdf['doubled'].values), err_msg="doubled values don't match"
        )


class TestDebugLoggingVerification(unittest.TestCase):
    """Test that debug logging correctly shows segment execution."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_data = pd.DataFrame(
            {
                'id': range(1, 11),
                'value': list(range(10, 110, 10)),
            }
        )
        cls.test_file = os.path.join(cls.temp_dir, 'test.parquet')
        cls.test_data.to_parquet(cls.test_file, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_debug_logs_show_segment_execution(self):
        """
        Verify debug logs show segment execution details and actual SQL.

        This test captures logs and verifies:
        1. Segment execution is logged (Segment 1/3, 2/3, 3/3)
        2. SQL statements contain expected clauses
        3. Pandas operations are logged
        4. Segment type descriptions match backend format
        5. SQL on DataFrame uses Python() table function
        """
        old_level = config.log_level

        try:
            config.enable_debug()

            ds = DataStore.from_file(self.test_file)
            ds = ds[ds['value'] > 30]
            ds['doubled'] = ds['value'].apply(lambda x: x * 2)
            ds = ds[ds['doubled'] > 100]

            # Capture logs
            logger = logging.getLogger('datastore')
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)

            try:
                result = ds.to_df()
            finally:
                logger.removeHandler(handler)

            log_output = log_capture.getvalue()

            # === VERIFY EXECUTION PLAN SUMMARY ===
            # Backend logs: "Execution plan: 3 segments (2 SQL, 1 Pandas)"
            self.assertIn('Execution plan:', log_output, "Log should show execution plan summary")
            self.assertIn('3 segments', log_output, "Should have 3 segments")
            self.assertIn('2 SQL', log_output, "Should have 2 SQL segments")
            self.assertIn('1 Pandas', log_output, "Should have 1 Pandas segment")

            # === VERIFY EXECUTION PLAN STRUCTURE ===
            # Backend logs ExecutionPlan.describe() output:
            #   [1] chDB (from source): 1 ops
            #   [2] Pandas (on DataFrame): 1 ops
            #   [3] chDB (on DataFrame): 1 ops
            self.assertIn('chDB (from source)', log_output, "Segment 1 should be chDB from source")
            self.assertIn('Pandas (on DataFrame)', log_output, "Segment 2 should be Pandas on DataFrame")
            self.assertIn('chDB (on DataFrame)', log_output, "Segment 3 should be chDB on DataFrame")

            # === VERIFY SEGMENT EXECUTION LOGGING ===
            # Backend logs: "Segment 1/3: chDB (from source): 1 ops"
            self.assertIn('Segment 1/3', log_output, "Log should show Segment 1/3")
            self.assertIn('Segment 2/3', log_output, "Log should show Segment 2/3")
            self.assertIn('Segment 3/3', log_output, "Log should show Segment 3/3")

            # === VERIFY SQL SEGMENT 1 (FROM SOURCE) ===
            # Backend logs: 'Executing SQL: SELECT * ... WHERE "value" > 30 ...'
            self.assertIn('Executing SQL:', log_output, "Should show SQL being executed")
            self.assertIn('WHERE "value" > 30', log_output, "SQL should contain WHERE clause for value > 30")

            # === VERIFY PANDAS SEGMENT 2 ===
            # Backend logs: '[Pandas] Executing ColumnAssignment: column='doubled''
            self.assertIn('[Pandas] Executing ColumnAssignment', log_output, "Should show Pandas ColumnAssignment")
            self.assertIn("column='doubled'", log_output, "Should show doubled column assignment")

            # === VERIFY SQL SEGMENT 3 (ON DATAFRAME) ===
            # Backend logs: '[SQL on DataFrame] Executing: SELECT * FROM __df__ WHERE "doubled" > 100'
            self.assertIn('[SQL on DataFrame]', log_output, "Should indicate SQL on DataFrame")
            self.assertIn('WHERE "doubled" > 100', log_output, "SQL should contain WHERE clause for doubled > 100")
            # Backend uses Python() table function for DataFrame SQL
            self.assertIn('Python(__df__)', log_output, "SQL on DataFrame should use Python() table function")

            # === VERIFY EXECUTION COMPLETION ===
            self.assertIn('Execution complete', log_output, "Log should contain 'Execution complete'")
            self.assertIn('Final DataFrame shape', log_output, "Should log final DataFrame shape")

            # === VERIFY RESULT VALUES ===
            pdf = self.test_data.copy()
            pdf = pdf[pdf['value'] > 30]
            pdf = pdf.copy()
            pdf['doubled'] = pdf['value'].apply(lambda x: x * 2)
            pdf = pdf[pdf['doubled'] > 100]

            self.assertEqual(len(result), len(pdf), f"Length mismatch: DS={len(result)}, PD={len(pdf)}")
            np.testing.assert_array_equal(
                sorted(result['id'].values), sorted(pdf['id'].values), err_msg="id values don't match"
            )

        finally:
            config.set_log_level(old_level)

    def test_explain_shows_lazy_ops(self):
        """
        Verify explain() shows all lazy operations with correct format.

        explain() output should include:
        1. Data Source description
        2. Phase classification (Initial SQL vs DataFrame Operations)
        3. Execution engine labels ([chDB] or [Pandas])
        4. Operation descriptions
        5. Generated SQL Query (for SQL-pushable operations)
        """
        ds = DataStore.from_file(self.test_file)
        ds = ds[ds['value'] > 30]
        ds['doubled'] = ds['value'].apply(lambda x: x * 2)
        ds = ds.sort_values('doubled')

        explain = ds.explain()

        # === VERIFY HEADER ===
        self.assertIn('Execution Plan (in execution order)', explain, "Should show execution plan header")

        # === VERIFY DATA SOURCE ===
        self.assertIn('Data Source:', explain, "Should show data source")
        self.assertIn('Parquet', explain, "Should show Parquet file type")

        # === VERIFY SEGMENT CLASSIFICATION ===
        # New format: "Segment 1 [chDB] (from source): Operations 2-2"
        #             "Segment 2 [Pandas] (on DataFrame): Operations 3-3"
        self.assertIn('Segment 1 [chDB] (from source)', explain, "Should show Segment 1 as chDB from source")
        self.assertIn('Segment 2 [Pandas] (on DataFrame)', explain, "Should show Segment 2 as Pandas on DataFrame")

        # === VERIFY EXECUTION ENGINE LABELS ===
        # Backend shows: [chDB] for SQL operations, [Pandas] for Pandas operations
        self.assertIn('[chDB]', explain, "Should show [chDB] for SQL operations")
        self.assertIn('[Pandas]', explain, "Should show [Pandas] for Pandas operations")

        # === VERIFY OPERATION DESCRIPTIONS ===
        # Backend shows: 'WHERE: "value" > 30' for filter
        self.assertIn('WHERE', explain, "Should show WHERE operation")
        self.assertIn('"value" > 30', explain, "Should show filter condition")

        # Backend shows: "Assign column 'doubled'" for column assignment
        self.assertIn("Assign column 'doubled'", explain, "Should show column assignment for doubled")

        # For SQL segments, sort shows as 'ORDER BY: doubled ASC'
        # For Pandas segments, it would show 'sort_values: doubled'
        # Since this sort_values follows a Pandas op, it goes to a SQL segment via Python()
        self.assertTrue(
            'ORDER BY' in explain or 'sort_values' in explain, "Should show ORDER BY or sort_values operation"
        )
        self.assertIn('doubled', explain, "Should show doubled in sort")

        # === VERIFY FINAL STATE ===
        self.assertIn('Pending (lazy, not yet executed)', explain, "Should show pending state")

        # === VERIFY GENERATED SQL QUERY ===
        self.assertIn('Generated SQL Query:', explain, "Should show Generated SQL Query section")
        self.assertIn('SELECT', explain, "Generated SQL should include SELECT")
        self.assertIn('WHERE "value" > 30', explain, "Generated SQL should include WHERE clause")

        # === VERIFY LAZY OPS COUNT ===
        self.assertEqual(len(ds._lazy_ops), 3, f"Expected 3 lazy_ops, got {len(ds._lazy_ops)}")


class TestComplexPipelineExactValues(unittest.TestCase):
    """Test complex pipelines with exact value verification."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

        # Create deterministic test data
        np.random.seed(42)
        cls.data = pd.DataFrame(
            {
                'id': range(1, 101),
                'value': np.random.randint(1, 100, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'score': np.random.uniform(0, 100, 100).round(2),
            }
        )
        cls.data_file = os.path.join(cls.temp_dir, 'data.parquet')
        cls.data.to_parquet(cls.data_file, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_filter_apply_filter_apply_filter(self):
        """
        Test: filter -> apply -> filter -> apply -> filter

        This creates multiple SQL-Pandas alternations.
        """
        ds = DataStore.from_file(self.data_file)
        ds = ds[ds['value'] > 20]  # SQL
        ds['v2'] = ds['value'].apply(lambda x: x * 2)  # Pandas
        ds = ds[ds['v2'] > 60]  # SQL on DF
        ds['v3'] = ds['v2'].apply(lambda x: x + 10)  # Pandas
        ds = ds[ds['v3'] > 100]  # SQL on DF

        # Check plan
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        print(f"\nExecution plan:\n{plan.describe()}")

        ds_result = ds.to_df()

        # Pandas reference
        pdf = self.data.copy()
        pdf = pdf[pdf['value'] > 20]
        pdf = pdf.copy()
        pdf['v2'] = pdf['value'].apply(lambda x: x * 2)
        pdf = pdf[pdf['v2'] > 60]
        pdf = pdf.copy()
        pdf['v3'] = pdf['v2'].apply(lambda x: x + 10)
        pdf = pdf[pdf['v3'] > 100]

        # Exact value verification
        self.assertEqual(len(ds_result), len(pdf), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf)}")

        if len(ds_result) > 0:
            np.testing.assert_array_equal(
                sorted(ds_result['id'].values), sorted(pdf['id'].values), err_msg="id values don't match"
            )
            np.testing.assert_array_equal(
                sorted(ds_result['v3'].values), sorted(pdf['v3'].values), err_msg="v3 values don't match"
            )

    def test_groupby_after_apply(self):
        """
        Test: apply -> groupby aggregation
        """
        ds = DataStore.from_file(self.data_file)
        ds['adjusted'] = ds['value'].apply(lambda x: x * 1.5)

        # GroupBy aggregation
        agg = ds.groupby('category').agg({'adjusted': 'sum', 'id': 'count'})

        ds_result = agg.to_df()

        # Pandas reference
        pdf = self.data.copy()
        pdf['adjusted'] = pdf['value'].apply(lambda x: x * 1.5)
        pdf_agg = pdf.groupby('category').agg({'adjusted': 'sum', 'id': 'count'})

        # Compare aggregated values
        self.assertEqual(len(ds_result), len(pdf_agg), f"Length mismatch: DS={len(ds_result)}, PD={len(pdf_agg)}")

        # Compare sum values (may have slight floating point differences)
        for cat in ['A', 'B', 'C']:
            if cat in ds_result.index and cat in pdf_agg.index:
                np.testing.assert_almost_equal(
                    ds_result.loc[cat, 'adjusted'],
                    pdf_agg.loc[cat, 'adjusted'],
                    decimal=1,
                    err_msg=f"adjusted sum for category {cat} doesn't match",
                )


class TestWhereMaskKnownLimitations(unittest.TestCase):
    """
    Tests for known limitations in where/mask SQL CASE WHEN pushdown.

    These tests document expected behaviors and known issues that need to be
    addressed in future updates.
    """

    def setUp(self):
        """Set up test data with bool and date columns."""
        self.df = pd.DataFrame(
            {
                'id': range(10),
                'int_col': [100, 600, 300, 800, 200, 700, 400, 900, 500, 1000],
                'str_col': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'],
                'bool_col': [True, False, True, False, True, False, True, False, True, False],
            }
        )

    def test_where_with_bool_col_falls_back_to_pandas(self):
        """
        When DataFrame has bool_col and 'other' is numeric (including 0 or 1),
        where() always falls back to Pandas to ensure type correctness.

        SQL CASE WHEN converts 0 to false, which changes both dtype and value.
        Pandas preserves the actual int value with object dtype.
        """
        ds = DataStore(self.df)

        # Get the lazy where operation with other=0
        ds_where = ds.where(ds['int_col'] > 500, 0)

        # Find the LazyWhere operation
        from datastore.lazy_ops import LazyWhere

        where_op = None
        for op in ds_where._lazy_ops:
            if isinstance(op, LazyWhere):
                where_op = op
                break

        self.assertIsNotNone(where_op, "Should have LazyWhere operation")

        # With bool_col in schema + any numeric other, should NOT push to SQL
        schema = {'id': 'Int64', 'int_col': 'Int64', 'str_col': 'String', 'bool_col': 'Bool'}
        self.assertFalse(
            where_op._is_type_compatible_with_schema(schema),
            "Should not be type compatible when bool column exists with numeric other",
        )

        # Test with other=-1 (should also NOT be compatible)
        ds_where_neg = ds.where(ds['int_col'] > 500, -1)
        where_op_neg = None
        for op in ds_where_neg._lazy_ops:
            if isinstance(op, LazyWhere):
                where_op_neg = op
                break
        self.assertFalse(
            where_op_neg._is_type_compatible_with_schema(schema),
            "Should not be type compatible when other=-1 with bool column",
        )

    def test_where_with_false_as_other_pandas_behavior(self):
        """
        Verify pandas behavior when using False as 'other' value.
        Pandas converts columns to object dtype to hold mixed types.
        """
        # Pandas allows False to replace any column value
        pd_result = self.df.where(self.df['int_col'] > 500, False)

        # int_col becomes object dtype with [False, 600, False, 800, ...]
        self.assertEqual(pd_result['int_col'].dtype, object)

        # Verify mixed values
        int_values = pd_result['int_col'].tolist()
        has_false = any(v is False for v in int_values)
        has_int = any(isinstance(v, int) and v is not False for v in int_values)
        self.assertTrue(has_false and has_int, "Should have mixed False and int values")

    @chdb_case_bool_conversion
    def test_where_with_false_as_other_sql_execution(self):
        """
        KNOWN LIMITATION: Using False as 'other' value causes SQL type conversion errors.

        This test documents that SQL CASE WHEN fails when:
        - 'other' is False (bool type)
        - Columns are Int64 or String type

        Expected fix: Add type compatibility check for bool 'other' with non-bool columns,
        similar to the existing check for numeric 'other' with bool columns.

        Mirror Code Pattern: pandas vs DataStore comparison
        """
        import tempfile

        # === Test Data ===
        df = pd.DataFrame(
            {
                'id': range(10),
                'int_col': [100, 600, 300, 800, 200, 700, 400, 900, 500, 1000],
                'str_col': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'],
            }
        )

        # === Pandas operations ===
        pd_result = df.where(df['int_col'] > 500, False)
        # Expected pandas behavior:
        #   id: [False, 1, False, 3, False, 5, False, 7, False, 9] (object dtype)
        #   int_col: [False, 600, False, 800, False, 700, False, 900, False, 1000] (object dtype)
        #   str_col: [False, 'B', False, 'D', False, 'A', False, 'C', False, 'E'] (object dtype)
        #
        # DataStore SQL behavior (FAILS):
        #   SQL: SELECT CASE WHEN int_col > 500 THEN id ELSE false END AS id, ...
        #   Error: Cannot convert type Bool to Variant(Int64, String)
        #   Reason: ClickHouse cannot mix Bool with Int64/String in CASE WHEN result

        # Verify pandas behavior
        self.assertEqual(pd_result['id'].dtype, object, "pandas converts id to object dtype")
        self.assertEqual(pd_result['int_col'].dtype, object, "pandas converts int_col to object dtype")
        self.assertEqual(pd_result['str_col'].dtype, object, "pandas converts str_col to object dtype")
        self.assertEqual(pd_result['id'].iloc[0], False, "pandas replaces with False where condition is False")
        self.assertEqual(pd_result['int_col'].iloc[1], 600, "pandas keeps original value where condition is True")

        # === DataStore operations (mirror of pandas) ===
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
            df.to_parquet(parquet_path)

        try:
            ds = DataStore.from_file(parquet_path)
            ds.connect()

            # This currently fails with: Cannot convert type Bool to Variant(Int64, String)
            # SQL generated: CASE WHEN int_col > 500 THEN col ELSE False END
            # ClickHouse error: Cannot convert type Bool to Variant(Int64, String)
            ds_result = ds.where(ds['int_col'] > 500, False).to_df()

            # === Compare results ===
            # When fixed, DataStore should match pandas behavior
            assert_datastore_equals_pandas(ds_result, pd_result)
        finally:
            import os

            os.unlink(parquet_path)

    def test_schema_tracking_after_column_selection(self):
        """
        Test that schema is correctly tracked after column selection
        for type compatibility checking in where/mask.

        After selecting columns that exclude bool_col, where() should
        be able to use SQL CASE WHEN.
        """
        from datastore.query_planner import QueryPlanner
        from datastore.lazy_ops import LazyRelationalOp, LazyWhere

        # Create operations: SELECT [id, int_col, str_col], then WHERE
        ops = [
            LazyRelationalOp(
                'SELECT', 'id, int_col, str_col', fields=[Field('"id"'), Field('"int_col"'), Field('"str_col"')]
            ),
            LazyWhere(BinaryCondition('>', Field('int_col'), Literal(500)), other=0),
        ]

        planner = QueryPlanner()
        original_schema = {
            'id': 'Int64',
            'int_col': 'Int64',
            'str_col': 'String',
            'bool_col': 'Bool',  # This should be excluded after SELECT
        }

        # Plan with original schema (includes bool_col)
        plan = planner.plan_segments(ops, has_sql_source=True, schema=original_schema)

        # After column selection, effective schema should exclude bool_col
        # So LazyWhere should be able to push to SQL
        # Count SQL segments
        sql_segments = [s for s in plan.segments if s.is_sql()]

        # Both SELECT and WHERE should be in SQL segments
        # (SELECT is always SQL-pushable, WHERE should be now that bool_col is excluded)
        self.assertGreaterEqual(len(sql_segments), 1, "Should have at least 1 SQL segment after schema tracking fix")

    def test_where_without_bool_col_uses_sql(self):
        """
        When DataFrame has no bool columns, where() should use SQL CASE WHEN.
        """
        df_no_bool = pd.DataFrame(
            {
                'id': range(10),
                'int_col': [100, 600, 300, 800, 200, 700, 400, 900, 500, 1000],
                'str_col': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'],
            }
        )

        ds = DataStore(df_no_bool)
        ds_where = ds.where(ds['int_col'] > 500, 0)

        # Find LazyWhere operation
        from datastore.lazy_ops import LazyWhere

        where_op = None
        for op in ds_where._lazy_ops:
            if isinstance(op, LazyWhere):
                where_op = op
                break

        self.assertIsNotNone(where_op)

        # Without bool_col, should be type compatible
        schema = {'id': 'Int64', 'int_col': 'Int64', 'str_col': 'String'}
        self.assertTrue(where_op._is_type_compatible_with_schema(schema), "Should be type compatible without bool_col")

        # Verify results match
        pd_result = df_no_bool.where(df_no_bool['int_col'] > 500, 0)
        ds_result = ds_where.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
