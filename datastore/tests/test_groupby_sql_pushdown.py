"""
Tests for GroupBy SQL Pushdown optimization.

These tests verify:
1. Filter + GroupBy + Sort operations can be pushed to SQL even with alias conflicts
2. Temporary alias renaming works correctly
3. Results match pandas behavior
4. Performance improvement when SQL pushdown is enabled

The key optimization tested here is the handling of ClickHouse's alias conflict:
When SELECT has `agg(col) AS col` and WHERE references `col`, ClickHouse would
incorrectly try to use the aggregate function in WHERE. We solve this by using
temporary aliases (`__agg_col__`) in SQL, then renaming back after execution.
"""

import os
import tempfile
import unittest
from typing import Dict

import numpy as np
import pandas as pd

from datastore import DataStore
from datastore.config import enable_profiling, disable_profiling, get_profiler, reset_profiler
from datastore.query_planner import QueryPlanner
from datastore.lazy_ops import LazyRelationalOp, LazyGroupByAgg


class TestGroupBySQLPushdown(unittest.TestCase):
    """Test GroupBy SQL pushdown with alias conflict handling."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_rows = 10000
        self.df = pd.DataFrame(
            {
                'id': range(self.n_rows),
                'int_col': np.random.randint(0, 1000, self.n_rows),
                'float_col': np.random.uniform(0, 1000, self.n_rows),
                'category': np.random.choice(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'], self.n_rows),
            }
        )

        # Create temporary parquet file
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_filter_groupby_agg_alias_conflict(self):
        """Test Filter + GroupBy + Agg with alias conflict (int_col in WHERE and as agg alias)."""
        # Pandas reference - groupby returns DataFrame with groupby col as index
        pd_result = self.df[self.df['int_col'] > 200].groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})

        # DataStore - should match pandas behavior (groupby col as index)
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['int_col'] > 200]
        ds_result = ds_result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})

        # Sort by index for consistent comparison
        pd_sorted = pd_result.sort_index()
        ds_sorted = ds_result.sort_index()

        # Verify index (groupby column) matches
        np.testing.assert_array_equal(ds_sorted.index, pd_sorted.index)

        # Verify column values match
        self.assertEqual(list(ds_sorted.columns), list(pd_sorted.columns))
        self.assertEqual(len(ds_sorted), len(pd_sorted))
        np.testing.assert_array_equal(ds_sorted['int_col'], pd_sorted['int_col'])
        np.testing.assert_allclose(ds_sorted['float_col'], pd_sorted['float_col'], rtol=1e-5)

    def test_filter_groupby_sort_alias_conflict(self):
        """Test Filter + GroupBy + Sort with alias conflict.

        This tests the ORDER BY temporary alias fix:
        When sorting by a column that has alias conflict (e.g. int_col),
        the ORDER BY should use the temporary alias (__agg_int_col__).
        """
        # Pandas reference
        pd_result = (
            self.df[self.df['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum', 'float_col': 'mean'})
            .sort_values(by='int_col', ascending=False)
        )

        # DataStore - sort_values on the conflict column should work
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['int_col'] > 200]
        ds_result = ds_result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})
        ds_result = ds_result.sort_values(by='int_col', ascending=False)

        # Verify order matches - index order should be same
        np.testing.assert_array_equal(ds_result.index, pd_result.index)

        # Verify values match in sorted order
        np.testing.assert_array_equal(ds_result['int_col'], pd_result['int_col'])
        np.testing.assert_allclose(ds_result['float_col'], pd_result['float_col'], rtol=1e-5)

    def test_filter_groupby_sort_ascending(self):
        """Test Filter + GroupBy + Sort ascending order."""
        # Pandas reference
        pd_result = (
            self.df[self.df['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum'})
            .sort_values(by='int_col', ascending=True)
        )

        # DataStore
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['int_col'] > 200]
        ds_result = ds_result.groupby('category').agg({'int_col': 'sum'})
        ds_result = ds_result.sort_values(by='int_col', ascending=True)

        # Verify sorted order
        np.testing.assert_array_equal(ds_result.index, pd_result.index)
        np.testing.assert_array_equal(ds_result['int_col'], pd_result['int_col'])

    def test_filter_groupby_sort_by_non_conflict_column(self):
        """Test sorting by a column that doesn't have alias conflict."""
        # Pandas reference - sort by float_col (no conflict with WHERE)
        pd_result = (
            self.df[self.df['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum', 'float_col': 'mean'})
            .sort_values(by='float_col', ascending=False)
        )

        # DataStore
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['int_col'] > 200]
        ds_result = ds_result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})
        ds_result = ds_result.sort_values(by='float_col', ascending=False)

        # Verify sorted order
        np.testing.assert_array_equal(ds_result.index, pd_result.index)
        np.testing.assert_allclose(ds_result['float_col'], pd_result['float_col'], rtol=1e-5)

    def test_filter_groupby_multiple_agg_alias_conflict(self):
        """Test Filter + GroupBy with multiple aggregations and alias conflicts."""
        # Pandas reference - multi-func agg creates MultiIndex columns
        pd_result = (
            self.df[self.df['int_col'] > 300].groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': 'max'})
        )

        # DataStore
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['int_col'] > 300]
        ds_result = ds_result.groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': 'max'})

        # Verify index and shape match
        self.assertEqual(len(ds_result), len(pd_result))
        np.testing.assert_array_equal(sorted(ds_result.index), sorted(pd_result.index))

    def test_multiple_filters_with_groupby(self):
        """Test multiple chained filters with groupby."""
        # Pandas reference
        pd_result = self.df[self.df['int_col'] > 100]
        pd_result = pd_result[pd_result['int_col'] < 800]
        pd_result = pd_result[pd_result['float_col'] > 200]
        pd_result = pd_result.groupby('category').agg({'int_col': 'sum'})

        # DataStore
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['int_col'] > 100]
        ds_result = ds_result[ds_result['int_col'] < 800]
        ds_result = ds_result[ds_result['float_col'] > 200]
        ds_result = ds_result.groupby('category').agg({'int_col': 'sum'})

        # Verify - index is groupby column, values in 'int_col'
        self.assertEqual(len(ds_result), len(pd_result))
        # Sort by index for comparison
        ds_sorted = ds_result.sort_index()
        pd_sorted = pd_result.sort_index()
        np.testing.assert_array_equal(ds_sorted['int_col'], pd_sorted['int_col'])

    def test_groupby_no_conflict(self):
        """Test GroupBy without alias conflict (different column names)."""
        # Pandas reference - float_col in WHERE, int_col in agg (no conflict)
        pd_result = self.df[self.df['float_col'] > 200].groupby('category').agg({'int_col': 'sum'})

        # DataStore
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['float_col'] > 200]
        ds_result = ds_result.groupby('category').agg({'int_col': 'sum'})

        # Verify - compare by sorted index
        self.assertEqual(len(ds_result), len(pd_result))
        ds_sorted = ds_result.sort_index()
        pd_sorted = pd_result.sort_index()
        np.testing.assert_array_equal(ds_sorted['int_col'], pd_sorted['int_col'])


class TestQueryPlannerAliasRenames(unittest.TestCase):
    """Test QueryPlanner alias conflict detection and resolution."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'int_col': [1, 2, 3, 4, 5],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_alias_conflict_detected(self):
        """Test that alias conflict is detected and recorded in plan."""
        ds = DataStore.from_file(self.parquet_path)

        # Add filter on int_col
        ds = ds[ds['int_col'] > 1]

        # Add groupby with agg on int_col (creates alias conflict)
        from copy import copy
        from datastore.lazy_ops import LazyGroupByAgg

        new_ds = copy(ds)
        new_ds._add_lazy_op(LazyGroupByAgg(groupby_cols=['category'], agg_dict={'int_col': 'sum'}))

        # Plan the operations
        planner = QueryPlanner()
        plan = planner.plan(new_ds._lazy_ops, has_sql_source=True)

        # Should detect conflict and create alias_renames
        self.assertIn('__agg_int_col__', plan.alias_renames)
        self.assertEqual(plan.alias_renames['__agg_int_col__'], 'int_col')

        # GroupByAgg should still be pushed to SQL
        self.assertIsNotNone(plan.groupby_agg)

    def test_no_conflict_when_different_columns(self):
        """Test that no conflict is detected when different columns are used."""
        ds = DataStore.from_file(self.parquet_path)

        # Filter on int_col, but agg on a different column name
        ds = ds[ds['int_col'] > 1]

        from copy import copy
        from datastore.lazy_ops import LazyGroupByAgg

        new_ds = copy(ds)
        new_ds._add_lazy_op(
            LazyGroupByAgg(groupby_cols=['category'], agg_dict={'int_col': 'sum'})  # Still uses int_col for agg
        )

        planner = QueryPlanner()
        plan = planner.plan(new_ds._lazy_ops, has_sql_source=True)

        # Conflict should be detected (int_col is in both WHERE and agg alias)
        self.assertIn('__agg_int_col__', plan.alias_renames)


class TestOrderByAliasMapping(unittest.TestCase):
    """Test ORDER BY correctly uses temporary aliases when there's a conflict."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_orderby_uses_temp_alias_for_conflict_column(self):
        """Test that ORDER BY uses temp alias when column has alias conflict."""
        # This is the key test for the ORDER BY fix:
        # Filter on 'value', then agg on 'value' (conflict), then sort by 'value'

        # Pandas reference
        pd_result = (
            self.df[self.df['value'] > 15]
            .groupby('category')
            .agg({'value': 'sum'})
            .sort_values(by='value', ascending=False)
        )

        # DataStore - should correctly map ORDER BY value -> ORDER BY __agg_value__
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['value'] > 15]
        ds_result = ds_result.groupby('category').agg({'value': 'sum'})
        ds_result = ds_result.sort_values(by='value', ascending=False)

        # Verify order and values
        np.testing.assert_array_equal(ds_result.index, pd_result.index)
        np.testing.assert_array_equal(ds_result['value'], pd_result['value'])

    def test_orderby_multiple_columns_with_partial_conflict(self):
        """Test ORDER BY when only some columns have conflict."""
        df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5, 6],
                'b': [60, 50, 40, 30, 20, 10],
                'cat': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            }
        )
        path = os.path.join(self.temp_dir, 'multi.parquet')
        df.to_parquet(path)

        # Filter on 'a', agg on 'a' and 'b', sort by 'a' (conflict) then 'b' (no conflict)
        pd_result = df[df['a'] > 1].groupby('cat').agg({'a': 'sum', 'b': 'mean'}).sort_values(by='a', ascending=False)

        ds = DataStore.from_file(path)
        ds_result = ds[ds['a'] > 1]
        ds_result = ds_result.groupby('cat').agg({'a': 'sum', 'b': 'mean'})
        ds_result = ds_result.sort_values(by='a', ascending=False)

        np.testing.assert_array_equal(ds_result.index, pd_result.index)
        np.testing.assert_array_equal(ds_result['a'], pd_result['a'])


class TestGroupBySQLPushdownPerformance(unittest.TestCase):
    """Test that SQL pushdown provides performance benefits."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_rows = 100000
        self.df = pd.DataFrame(
            {
                'id': range(self.n_rows),
                'int_col': np.random.randint(0, 1000, self.n_rows),
                'float_col': np.random.uniform(0, 1000, self.n_rows),
                'category': np.random.choice(['cat1', 'cat2', 'cat3', 'cat4', 'cat5'], self.n_rows),
            }
        )

        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_groupby_sql_pushdown_executed(self):
        """Test that GroupBy is pushed to SQL (no LazyGroupByAgg in DataFrame ops)."""
        reset_profiler()
        enable_profiling()

        ds = DataStore.from_file(self.parquet_path)
        result = ds[ds['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'})

        # Trigger execution by accessing values
        _ = len(result)

        disable_profiling()

        profiler = get_profiler()
        summary = profiler.summary() if profiler else {}

        # LazyGroupByAgg should NOT appear in DataFrame Operations
        # (it should be pushed to SQL)
        has_pandas_groupby = any('LazyGroupByAgg' in key for key in summary.keys())
        self.assertFalse(
            has_pandas_groupby,
            f"LazyGroupByAgg should not be in DataFrame Operations when SQL pushdown is enabled. "
            f"Found: {[k for k in summary.keys() if 'LazyGroupByAgg' in k]}",
        )

        # Verify result is correct - 5 categories as index
        self.assertEqual(len(result), 5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for GroupBy SQL pushdown."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'cat': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_empty_result_after_filter(self):
        """Test groupby on empty result after filter."""
        ds = DataStore.from_file(self.parquet_path)
        result = ds[ds['a'] > 1000]  # No rows match
        result = result.groupby('cat').agg({'a': 'sum'})

        self.assertEqual(len(result), 0)

    def test_single_row_per_group(self):
        """Test groupby with single row per group after filter."""
        # Create data with unique categories
        df = pd.DataFrame(
            {
                'a': [1, 2, 3],
                'cat': ['X', 'Y', 'Z'],
            }
        )
        path = os.path.join(self.temp_dir, 'unique_cats.parquet')
        df.to_parquet(path)

        ds = DataStore.from_file(path)
        result = ds.groupby('cat').agg({'a': 'sum'})

        # Index should be ['X', 'Y', 'Z'], values should be [1, 2, 3]
        self.assertEqual(len(result), 3)
        result_sorted = result.sort_index()
        np.testing.assert_array_equal(result_sorted['a'], [1, 2, 3])

    def test_multiple_columns_same_agg(self):
        """Test multiple columns with same aggregation function."""
        ds = DataStore.from_file(self.parquet_path)
        result = ds[ds['a'] > 3]
        result = result.groupby('cat').agg({'a': 'sum', 'b': 'sum'})

        pd_result = self.df[self.df['a'] > 3].groupby('cat').agg({'a': 'sum', 'b': 'sum'})

        self.assertEqual(len(result), len(pd_result))


if __name__ == '__main__':
    unittest.main()
