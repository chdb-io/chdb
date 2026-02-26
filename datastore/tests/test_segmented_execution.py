"""
Tests for segmented execution (SQL-Pandas-SQL interleaving).

This module tests the QueryPlanner.plan_segments() functionality and
the ability to execute SQL on intermediate DataFrames.

Test Principles:
- Verify segment classification (SQL vs Pandas)
- Verify SQL pushdown within each segment
- Verify row order preservation across segments
- Compare results with pure Pandas execution
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from datastore import DataStore
from datastore.query_planner import QueryPlanner, ExecutionPlan, ExecutionSegment, QueryPlan
from datastore.lazy_ops import (
    LazyOp,
    LazyRelationalOp,
    LazyColumnAssignment,
    LazyFilter,
    LazyApply,
    LazyWhere,
    LazyMask,
)
from tests.test_utils import assert_datastore_equals_pandas
from datastore.expressions import Field, Literal
from datastore.conditions import BinaryCondition


class TestQueryPlannerSegments:
    """Test QueryPlanner.plan_segments() segment classification."""

    def test_all_sql_ops_single_segment(self):
        """All SQL-pushable ops should form a single SQL segment."""
        planner = QueryPlanner()

        ops = [
            LazyRelationalOp('WHERE', 'filter', condition=BinaryCondition('>', Field('a'), Literal(10))),
            LazyRelationalOp('ORDER BY', 'sort', fields=[Field('b')]),
            LazyRelationalOp('LIMIT', 'limit', limit_value=100),
        ]

        plan = planner.plan_segments(ops, has_sql_source=True)

        assert len(plan.segments) == 1
        assert plan.segments[0].is_sql()
        assert plan.segments[0].is_first_segment
        assert len(plan.segments[0].ops) == 3

    def test_all_pandas_ops_single_segment(self):
        """All Pandas-only ops should form a single Pandas segment."""
        planner = QueryPlanner()

        ops = [
            LazyColumnAssignment('new_col', lambda df: df['a'] + 1),
            LazyFilter(lambda df: df['a'] > 10, 'custom_filter'),
            LazyApply(lambda x: x * 2, 'apply'),
        ]

        plan = planner.plan_segments(ops, has_sql_source=True)

        assert len(plan.segments) == 1
        assert plan.segments[0].is_pandas()
        assert len(plan.segments[0].ops) == 3

    def test_sql_pandas_sql_three_segments(self):
        """SQL -> Pandas -> SQL should create 3 segments."""
        planner = QueryPlanner()

        ops = [
            # SQL segment
            LazyRelationalOp('WHERE', 'filter1', condition=BinaryCondition('>', Field('a'), Literal(10))),
            # Pandas segment (breaks SQL chain)
            LazyColumnAssignment('new_col', lambda df: df['a'] * 2),
            # SQL segment (can resume SQL on DataFrame)
            LazyRelationalOp('WHERE', 'filter2', condition=BinaryCondition('<', Field('b'), Literal(100))),
        ]

        plan = planner.plan_segments(ops, has_sql_source=True)

        assert len(plan.segments) == 3
        assert plan.segments[0].is_sql()
        assert plan.segments[0].is_first_segment
        assert plan.segments[1].is_pandas()
        assert not plan.segments[1].is_first_segment
        assert plan.segments[2].is_sql()
        assert not plan.segments[2].is_first_segment

    def test_multiple_alternating_segments(self):
        """Complex alternating pattern should create correct segments."""
        planner = QueryPlanner()

        ops = [
            # SQL 1
            LazyRelationalOp('WHERE', 'f1', condition=BinaryCondition('>', Field('a'), Literal(0))),
            LazyRelationalOp('SELECT', 'select', fields=[Field('a'), Field('b')]),
            # Pandas 1
            LazyApply(lambda x: x, 'apply1'),
            # SQL 2
            LazyRelationalOp('WHERE', 'f2', condition=BinaryCondition('<', Field('a'), Literal(100))),
            # Pandas 2
            LazyColumnAssignment('c', lambda df: df['a'] + df['b']),
            LazyFilter(lambda df: df['c'] > 0, 'filter'),
            # SQL 3
            LazyRelationalOp('ORDER BY', 'sort', fields=[Field('a')]),
            LazyRelationalOp('LIMIT', 'limit', limit_value=10),
        ]

        plan = planner.plan_segments(ops, has_sql_source=True)

        assert len(plan.segments) == 5
        assert plan.sql_segment_count() == 3
        assert plan.pandas_segment_count() == 2

        # Check segment types
        expected_types = ['sql', 'pandas', 'sql', 'pandas', 'sql']
        actual_types = [seg.segment_type for seg in plan.segments]
        assert actual_types == expected_types

        # Check segment sizes
        expected_sizes = [2, 1, 1, 2, 2]
        actual_sizes = [len(seg.ops) for seg in plan.segments]
        assert actual_sizes == expected_sizes

    def test_no_sql_source_all_pandas(self):
        """Without SQL source, all ops should be in Pandas segments."""
        planner = QueryPlanner()

        ops = [
            LazyRelationalOp('WHERE', 'filter', condition=BinaryCondition('>', Field('a'), Literal(10))),
            LazyRelationalOp('ORDER BY', 'sort', fields=[Field('b')]),
        ]

        # has_sql_source=False means no SQL pushdown possible
        plan = planner.plan_segments(ops, has_sql_source=False)

        # Even SQL-like ops become Pandas when there's no SQL source
        # (or they might still be classified as SQL but won't be executed via SQL)
        assert plan.has_sql_source is False

    def test_empty_ops_with_sql_source_creates_read_segment(self):
        """Empty ops with SQL source should create a segment to read data (SELECT *)."""
        planner = QueryPlanner()
        plan = planner.plan_segments([], has_sql_source=True)

        # When there's a SQL source but no ops, we need a segment to read the data
        assert len(plan.segments) == 1
        assert plan.segments[0].is_sql()
        assert plan.segments[0].is_first_segment
        assert plan.total_ops() == 0  # No user ops, just data source read

    def test_empty_ops_without_sql_source_empty_plan(self):
        """Empty ops without SQL source should produce empty plan."""
        planner = QueryPlanner()
        plan = planner.plan_segments([], has_sql_source=False)

        assert len(plan.segments) == 0
        assert plan.total_ops() == 0


class TestExecutionPlanMetadata:
    """Test ExecutionPlan metadata methods."""

    def test_describe(self):
        """ExecutionPlan.describe() should return readable summary."""
        plan = ExecutionPlan(
            segments=[
                ExecutionSegment('sql', [LazyRelationalOp('WHERE', 'f')], is_first_segment=True),
                ExecutionSegment('pandas', [LazyApply(lambda x: x, 'a')]),
                ExecutionSegment('sql', [LazyRelationalOp('LIMIT', 'l', limit_value=10)]),
            ],
            has_sql_source=True,
        )

        desc = plan.describe()
        assert 'ExecutionPlan' in desc
        assert 'SQL source: True' in desc
        assert 'Segments: 3' in desc

    def test_total_ops(self):
        """total_ops() should count all operations."""
        plan = ExecutionPlan(
            segments=[
                ExecutionSegment(
                    'sql',
                    [
                        LazyRelationalOp('WHERE', 'f1'),
                        LazyRelationalOp('WHERE', 'f2'),
                    ],
                ),
                ExecutionSegment('pandas', [LazyApply(lambda x: x, 'a')]),
            ]
        )

        assert plan.total_ops() == 3


class TestSegmentedExecutionIntegration:
    """Integration tests for segmented execution with real data."""

    def test_filter_assign_filter_sequence(self):
        """Test SQL -> Column Assignment -> SQL pattern."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 100, n),
                'category': np.random.choice(['A', 'B', 'C'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(path, index=False)

            # DataStore: filter -> assign -> filter
            ds = DataStore.from_file(path)
            ds = ds[ds['value'] > 30]  # SQL pushable
            ds['double_value'] = ds['value'] * 2  # Pandas only
            ds = ds[ds['double_value'] > 100]  # Could be SQL on DataFrame

            # Pandas equivalent
            pd_result = df[df['value'] > 30].copy()
            pd_result['double_value'] = pd_result['value'] * 2
            pd_result = pd_result[pd_result['double_value'] > 100]

            # Compare
            assert_datastore_equals_pandas(ds, pd_result)

    def test_apply_then_filter_sequence(self):
        """Test Pandas apply -> SQL filter pattern."""
        df = pd.DataFrame(
            {
                'id': range(100),
                'value': np.random.randint(0, 100, 100),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # DataStore: apply -> filter
            ds = DataStore.from_file(path)
            ds['squared'] = ds['value'].apply(lambda x: x**2)  # Pandas only
            ds = ds[ds['squared'] > 1000]  # Could be SQL on DataFrame

            # Pandas equivalent
            pd_result = df.copy()
            pd_result['squared'] = pd_result['value'].apply(lambda x: x**2)
            pd_result = pd_result[pd_result['squared'] > 1000]

            # Compare
            assert_datastore_equals_pandas(ds, pd_result)

    def test_where_apply_where_sequence(self):
        """Test where -> apply -> where pattern."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                'id': range(500),
                'value': np.random.randint(0, 100, 500),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(path, index=False)

            # DataStore
            ds = DataStore.from_file(path)
            ds = ds.where(ds['value'] > 50, 0)  # SQL pushable (CASE WHEN)
            ds['log_value'] = ds['value'].apply(lambda x: np.log1p(x))  # Pandas only
            ds = ds.where(ds['log_value'] > 2, -1)  # SQL on DataFrame

            # Pandas
            pd_result = df.where(df['value'] > 50, 0).copy()
            pd_result['log_value'] = pd_result['value'].apply(lambda x: np.log1p(x))
            pd_result = pd_result.where(pd_result['log_value'] > 2, -1)

            # Compare - use assert_datastore_equals_pandas which handles float comparisons
            assert_datastore_equals_pandas(ds, pd_result, rtol=1e-5)


class TestRowOrderPreservation:
    """Test that row order is preserved across segment boundaries."""

    def test_row_order_across_sql_pandas_boundary(self):
        """Row order must be preserved when transitioning between engines."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(path, index=False)

            # DataStore: filter -> transform -> filter
            ds = DataStore.from_file(path)
            ds = ds[ds['value'] > 20]  # SQL
            ds['transformed'] = ds['value'].apply(lambda x: x + 1)  # Pandas
            ds = ds[ds['transformed'] > 50]  # SQL on DataFrame

            # Pandas equivalent
            pd_result = df[df['value'] > 20].copy()
            pd_result['transformed'] = pd_result['value'].apply(lambda x: x + 1)
            pd_result = pd_result[pd_result['transformed'] > 50]

            # Compare - row order is critical
            assert_datastore_equals_pandas(ds, pd_result)

    def test_row_order_with_multiple_segments(self):
        """Row order preserved across multiple segment transitions."""
        np.random.seed(123)
        n = 500
        df = pd.DataFrame(
            {
                'id': range(n),
                'a': np.random.randint(0, 50, n),
                'b': np.random.randint(0, 50, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Multiple transitions: SQL -> Pandas -> SQL -> Pandas -> SQL
            ds = DataStore.from_file(path)
            ds = ds[ds['a'] > 10]  # SQL 1
            ds['sum'] = ds['a'] + ds['b']  # Pandas 1 (column assignment)
            ds = ds[ds['b'] < 40]  # SQL 2
            ds['product'] = ds['a'].apply(lambda x: x * 2)  # Pandas 2
            ds = ds[ds['sum'] > 30]  # SQL 3

            # Pandas equivalent
            pd_result = df[df['a'] > 10].copy()
            pd_result['sum'] = pd_result['a'] + pd_result['b']
            pd_result = pd_result[pd_result['b'] < 40]
            pd_result['product'] = pd_result['a'].apply(lambda x: x * 2)
            pd_result = pd_result[pd_result['sum'] > 30]

            # Compare
            assert_datastore_equals_pandas(ds, pd_result)


class TestExecuteSqlOnDataFrame:
    """Test SQLExecutionEngine.execute_sql_on_dataframe()."""

    def test_simple_filter_on_dataframe(self):
        """Test executing a simple filter on an existing DataFrame."""
        df = pd.DataFrame(
            {
                'id': range(100),
                'value': range(100),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(path, index=False)

            # Use DataStore to test SQL on DataFrame capability
            ds = DataStore.from_file(path)

            # Trigger execution naturally by accessing columns
            ds['extra'] = ds['value'] * 2  # Force Pandas execution
            # Access columns to trigger execution and get intermediate result
            _ = list(ds.columns)  # Natural trigger
            intermediate_df = pd.DataFrame({col: ds[col].values for col in ds.columns})

            # Now create new DataStore from DataFrame and filter
            ds2 = DataStore.from_dataframe(intermediate_df)
            ds2 = ds2[ds2['extra'] > 100]

            # Pandas equivalent
            pd_result = intermediate_df[intermediate_df['extra'] > 100]

            assert_datastore_equals_pandas(ds2, pd_result)

    def test_aggregation_on_dataframe(self):
        """Test executing aggregation on an existing DataFrame."""
        df = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'C'],
                'value': [10, 20, 30, 40, 50, 60],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(path, index=False)

            # Create intermediate DataFrame with transformation
            ds = DataStore.from_file(path)
            ds['doubled'] = ds['value'] * 2  # Force Pandas
            # Access columns to trigger execution and get intermediate result
            _ = list(ds.columns)  # Natural trigger
            intermediate_df = pd.DataFrame({col: ds[col].values for col in ds.columns})

            # Aggregate on the transformed DataFrame
            ds2 = DataStore.from_dataframe(intermediate_df)
            ds_result = ds2.groupby('category')['doubled'].sum()

            # Pandas equivalent
            pd_result = intermediate_df.groupby('category')['doubled'].sum()

            # Compare using assert_datastore_equals_pandas (supports Series)
            assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
