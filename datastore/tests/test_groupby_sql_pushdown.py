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
        exec_plan = planner.plan_segments(new_ds._lazy_ops, has_sql_source=True)
        plan = exec_plan.segments[0].plan

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
        exec_plan = planner.plan_segments(new_ds._lazy_ops, has_sql_source=True)
        plan = exec_plan.segments[0].plan

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


class TestPostAggregationFilterAfterLimit(unittest.TestCase):
    """Regression tests for filtering on an aggregated column after sort+head.

    Original report: chaining
        df[df['verified_purchase'] == True]
          .groupby('product_category')['star_rating']
          .agg(['mean','count'])
          .sort_values('count', ascending=False)
          .head(10)
        [<above>['count'] > N]
    used to emit nested SQL whose inner subquery dropped the GROUP BY +
    aggregation entirely, producing
        ``SELECT * FROM source WHERE ... ORDER BY count DESC LIMIT 10``
    and failing with ``UNKNOWN_IDENTIFIER`` for ``count``.

    These tests verify that boolean indexing, ``.loc[...]``, and
    ``.query(...)`` on the aggregated column after a sort+head all execute
    successfully and match pandas.
    """

    def setUp(self):
        """Set up test data with enough rows to make filtering meaningful."""
        np.random.seed(42)
        self.n_rows = 50000
        self.df = pd.DataFrame(
            {
                'product_category': np.random.choice(
                    ['Home', 'Books', 'Apparel', 'Beauty', 'Kitchen', 'Mobile_Apps', 'Sports'],
                    self.n_rows,
                ),
                'star_rating': np.random.choice([1, 2, 3, 4, 5], self.n_rows),
                'verified_purchase': np.random.choice([True, False], self.n_rows, p=[0.7, 0.3]),
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'amazon_sample.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _build_aggregated(self, source):
        """Mirror code: build the agg+sort+head pipeline on either engine."""
        return (
            source[source['verified_purchase'] == True]
            .groupby('product_category')['star_rating']
            .agg(['mean', 'count'])
            .sort_values('count', ascending=False)
            .head(n=10)
        )

    def test_boolean_mask_filter_on_agg_count_after_sort_head_matches_pandas(self):
        """``result[result['count'] > N]`` must keep the inner GROUP BY/aggregation."""
        ds = DataStore.from_file(self.parquet_path)

        pd_agg = self._build_aggregated(self.df)
        ds_agg = self._build_aggregated(ds)

        pd_filtered = pd_agg[pd_agg['count'] > 5000]
        ds_filtered = ds_agg[ds_agg['count'] > 5000]

        np.testing.assert_array_equal(ds_filtered.index, pd_filtered.index)
        self.assertEqual(list(ds_filtered.columns), list(pd_filtered.columns))
        np.testing.assert_array_equal(ds_filtered['count'], pd_filtered['count'])
        np.testing.assert_allclose(ds_filtered['mean'], pd_filtered['mean'], rtol=1e-5)

    def test_loc_filter_on_agg_count_after_sort_head_matches_pandas(self):
        """``result.loc[result['count'] > N]`` must produce the same data as pandas."""
        ds = DataStore.from_file(self.parquet_path)

        pd_agg = self._build_aggregated(self.df)
        ds_agg = self._build_aggregated(ds)

        pd_filtered = pd_agg.loc[pd_agg['count'] > 5000]
        ds_filtered = ds_agg.loc[ds_agg['count'] > 5000]

        np.testing.assert_array_equal(ds_filtered.index, pd_filtered.index)
        self.assertEqual(list(ds_filtered.columns), list(pd_filtered.columns))
        np.testing.assert_array_equal(ds_filtered['count'], pd_filtered['count'])
        np.testing.assert_allclose(ds_filtered['mean'], pd_filtered['mean'], rtol=1e-5)

    def test_query_filter_on_agg_count_after_sort_head_matches_pandas(self):
        """``result.query('count > N')`` must produce the same data as pandas."""
        ds = DataStore.from_file(self.parquet_path)

        pd_agg = self._build_aggregated(self.df)
        ds_agg = self._build_aggregated(ds)

        pd_filtered = pd_agg.query('count > 5000')
        ds_filtered = ds_agg.query('count > 5000')

        np.testing.assert_array_equal(ds_filtered.index, pd_filtered.index)
        self.assertEqual(list(ds_filtered.columns), list(pd_filtered.columns))
        np.testing.assert_array_equal(ds_filtered['count'], pd_filtered['count'])
        np.testing.assert_allclose(ds_filtered['mean'], pd_filtered['mean'], rtol=1e-5)

    def test_inner_layer_preserves_group_by_and_aggregation_in_sql(self):
        """Generated nested SQL must keep the GROUP BY + aggregation in the inner subquery.

        Without the fix, the inner subquery became
        ``SELECT * FROM source WHERE ... ORDER BY count DESC LIMIT 10`` -
        which fails because ``count`` is not a source column.
        """
        from datastore.query_planner import QueryPlanner
        from datastore.sql_executor import SQLExecutionEngine
        from datastore.lazy_ops import LazyGroupByAgg

        ds = DataStore.from_file(self.parquet_path)
        ds_agg = self._build_aggregated(ds)
        ds_filtered = ds_agg[ds_agg['count'] > 5000]

        planner = QueryPlanner()
        exec_plan = planner.plan_segments(
            ds_filtered._lazy_ops, has_sql_source=True, schema=None
        )
        # Expect a single SQL segment for this fully push-downable chain.
        self.assertEqual(len(exec_plan.segments), 1)
        seg = exec_plan.segments[0]
        self.assertEqual(seg.segment_type, 'sql')
        self.assertIsNotNone(seg.plan)
        # The LazyGroupByAgg must live inside layer 0 so that the per-layer
        # SQL builder dispatches GROUP BY into the innermost subquery.
        self.assertTrue(
            any(isinstance(op, LazyGroupByAgg) for op in seg.plan.layers[0]),
            f"Expected LazyGroupByAgg in inner layer, got: {seg.plan.layers}",
        )

        engine = SQLExecutionEngine(ds_filtered)
        sql_result = engine.build_sql_from_plan(seg.plan, schema={})
        sql = sql_result.sql

        # The inner subquery (before the outer WHERE on the agg result) MUST
        # contain GROUP BY and the aggregation functions for the SQL to be
        # well-formed and reference ``count``.
        self.assertIn('GROUP BY', sql)
        self.assertIn('avg("star_rating")', sql)
        self.assertIn('count("star_rating")', sql)
        # The outer WHERE references the aggregate alias.
        self.assertIn('> 5000', sql)


class TestGroupByInNonZeroLayer(unittest.TestCase):
    """Regression tests for LazyGroupByAgg living in a non-innermost layer.

    A real-world chain like
        df[x > N].head(K)[y > M].groupby('z').agg('sum')[(agg_col > V)]
    produces layers:
        layer 0: [WHERE x>N, LIMIT K]
        layer 1: [WHERE y>M, LazyGroupByAgg(z, sum), (optional) WHERE agg_col>V]
    The previous architecture stripped LazyGroupByAgg out of the layer
    contents and only re-injected it into the innermost layer's SQL builder.
    When the GroupByAgg lived in layer 1+, the wrapper builder silently
    dropped it and the query returned raw filtered rows instead of an
    aggregated result.

    These tests exercise that pattern and assert (a) the SQL contains
    GROUP BY in the correct subquery layer, and (b) the executed result
    matches pandas to the row/value.
    """

    def setUp(self):
        np.random.seed(42)
        self.n = 500
        self.df = pd.DataFrame(
            {
                'x': np.random.randint(0, 100, self.n),
                'y': np.random.randint(0, 100, self.n),
                'z': np.random.choice(['A', 'B', 'C', 'D'], self.n),
                'v': np.random.randint(0, 1000, self.n),
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def _planner_layers(self, ds_chain):
        from datastore.query_planner import QueryPlanner

        planner = QueryPlanner()
        exec_plan = planner.plan_segments(
            ds_chain._lazy_ops, has_sql_source=True, schema=None
        )
        self.assertEqual(len(exec_plan.segments), 1)
        seg = exec_plan.segments[0]
        self.assertEqual(seg.segment_type, 'sql')
        self.assertIsNotNone(seg.plan)
        return seg.plan

    def test_filter_limit_filter_groupby_agg_lives_in_wrapper_layer(self):
        """``df[x>N].head(K)[y>M].groupby('z').agg('sum')`` puts GroupByAgg
        into layer 1, and the SQL must wrap with GROUP BY in the outer query."""
        from datastore.lazy_ops import LazyGroupByAgg
        from datastore.sql_executor import SQLExecutionEngine

        ds = DataStore.from_file(self.parquet_path)
        ds_chain = (
            ds[ds['x'] > 20].head(100)
        )
        ds_chain = ds_chain[ds_chain['y'] > 30].groupby('z').agg({'v': 'sum'})

        plan = self._planner_layers(ds_chain)
        # Two layers, GroupByAgg lives in layer 1 (not layer 0)
        self.assertEqual(len(plan.layers), 2)
        self.assertFalse(
            any(isinstance(op, LazyGroupByAgg) for op in plan.layers[0]),
            f"Expected layer 0 to not contain LazyGroupByAgg, got {plan.layers[0]}",
        )
        self.assertTrue(
            any(isinstance(op, LazyGroupByAgg) for op in plan.layers[1]),
            f"Expected layer 1 to contain LazyGroupByAgg, got {plan.layers[1]}",
        )

        engine = SQLExecutionEngine(ds_chain)
        sql = engine.build_sql_from_plan(plan, schema={}).sql
        # The outer subquery (layer 1's SQL) must contain GROUP BY and the
        # aggregate. Without the fix, it would just be SELECT * FROM (...)
        # WHERE y > 30 with no aggregation.
        self.assertIn('GROUP BY', sql)
        self.assertIn('sum("v")', sql)
        # And the inner subquery is still the WHERE + LIMIT we expect
        self.assertIn('"x" > 20', sql)
        self.assertIn('LIMIT 100', sql)

    def test_filter_limit_filter_groupby_agg_result_matches_pandas(self):
        """End-to-end: same chain must return the same data as pandas."""
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['x'] > 20].head(100)
        ds_result = ds_result[ds_result['y'] > 30].groupby('z').agg({'v': 'sum'})

        pd_result = self.df[self.df['x'] > 20].head(100)
        pd_result = pd_result[pd_result['y'] > 30].groupby('z').agg({'v': 'sum'})

        ds_sorted = ds_result.sort_index()
        pd_sorted = pd_result.sort_index()
        np.testing.assert_array_equal(ds_sorted.index, pd_sorted.index)
        self.assertEqual(list(ds_sorted.columns), list(pd_sorted.columns))
        np.testing.assert_array_equal(ds_sorted['v'], pd_sorted['v'])

    def test_filter_limit_filter_groupby_agg_filter_agg_col_matches_pandas(self):
        """Same chain with an additional post-agg WHERE on the aggregated
        column - the post-agg WHERE becomes HAVING in the wrapper layer."""
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['x'] > 20].head(200)
        ds_result = ds_result[ds_result['y'] > 30].groupby('z').agg({'v': 'sum'})
        ds_result = ds_result[ds_result['v'] > 5000]

        pd_result = self.df[self.df['x'] > 20].head(200)
        pd_result = pd_result[pd_result['y'] > 30].groupby('z').agg({'v': 'sum'})
        pd_result = pd_result[pd_result['v'] > 5000]

        ds_sorted = ds_result.sort_index()
        pd_sorted = pd_result.sort_index()
        np.testing.assert_array_equal(ds_sorted.index, pd_sorted.index)
        self.assertEqual(list(ds_sorted.columns), list(pd_sorted.columns))
        if len(pd_sorted) > 0:
            np.testing.assert_array_equal(ds_sorted['v'], pd_sorted['v'])

    def test_dataframe_source_groupby_agg_in_layer_n_matches_pandas(self):
        """DataFrame-source equivalent of the layer-N GroupByAgg fix.

        Same nested chain but the source is an in-memory DataFrame
        (Python() table function), exercising the unified ``_build_layer``
        loop with ``first_layer_from_source='__df__'``. Previously the
        DataFrame path had its own ``_build_nested_sql_for_dataframe``
        which silently dropped a GroupByAgg living in layer 1+; the
        unified pipeline now dispatches it correctly here too.
        """
        ds = DataStore(self.df)
        ds_result = ds[ds['x'] > 20].head(100)
        ds_result = ds_result[ds_result['y'] > 30].groupby('z').agg({'v': 'sum'})

        pd_result = self.df[self.df['x'] > 20].head(100)
        pd_result = pd_result[pd_result['y'] > 30].groupby('z').agg({'v': 'sum'})

        ds_sorted = ds_result.sort_index()
        pd_sorted = pd_result.sort_index()
        np.testing.assert_array_equal(ds_sorted.index, pd_sorted.index)
        self.assertEqual(list(ds_sorted.columns), list(pd_sorted.columns))
        np.testing.assert_array_equal(ds_sorted['v'], pd_sorted['v'])


class TestCrossLayerAliasRewriting(unittest.TestCase):
    """Regression tests for cross-layer alias rewriting.

    When an inner subquery layer is forced to emit a column under a temp
    alias due to a source-col/agg-alias conflict (e.g. ``SUM(int_col) AS
    __agg_int_col__``), an outer wrapper layer that references the original
    name (``int_col``) used to fail because the inner subquery no longer
    exposed that name. The :func:`rewrite_column_refs_in_expression` /
    :func:`rewrite_column_refs_in_orderby` helpers plus the
    ``inner_visible_to_temp`` plumbing now rewrite the outer Field references
    onto the temp alias.
    """

    def setUp(self):
        np.random.seed(42)
        n = 1000
        self.df = pd.DataFrame(
            {
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
                'category': np.random.choice(['A', 'B', 'C'], n),
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_source_conflict_with_post_limit_filter_on_same_col(self):
        """Inner layer aliases ``int_col -> __agg_int_col__``; outer WHERE
        on ``int_col`` must be rewritten to use the temp alias."""
        pd_result = (
            self.df[self.df['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum'})
            .sort_values('int_col', ascending=False)
            .head(10)
        )
        pd_filter = pd_result[pd_result['int_col'] > 50000]

        ds = DataStore.from_file(self.parquet_path)
        ds_result = (
            ds[ds['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum'})
            .sort_values('int_col', ascending=False)
            .head(10)
        )
        ds_filter = ds_result[ds_result['int_col'] > 50000]

        np.testing.assert_array_equal(ds_filter.index, pd_filter.index)
        np.testing.assert_array_equal(ds_filter['int_col'], pd_filter['int_col'])

    def test_source_conflict_with_post_limit_orderby_on_same_col(self):
        """Same conflict pattern but the outer layer does ORDER BY on the
        renamed column instead of WHERE."""
        pd_base = (
            self.df[self.df['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum'})
            .sort_values('int_col', ascending=False)
            .head(10)
        )
        pd_sorted = pd_base.sort_values('int_col', ascending=True)

        ds = DataStore.from_file(self.parquet_path)
        ds_base = (
            ds[ds['int_col'] > 200]
            .groupby('category')
            .agg({'int_col': 'sum'})
            .sort_values('int_col', ascending=False)
            .head(10)
        )
        ds_sorted = ds_base.sort_values('int_col', ascending=True)

        np.testing.assert_array_equal(ds_sorted.index, pd_sorted.index)
        np.testing.assert_array_equal(ds_sorted['int_col'], pd_sorted['int_col'])

    def test_rewrite_column_refs_in_expression_returns_new_tree(self):
        """The helper must return a new tree without mutating the original."""
        from datastore.sql_executor import rewrite_column_refs_in_expression
        from datastore.expressions import Field, Literal
        from datastore.conditions import BinaryCondition, CompoundCondition

        original = CompoundCondition(
            'AND',
            BinaryCondition('>', Field('a'), Literal(1)),
            BinaryCondition('<', Field('b'), Literal(10)),
        )
        rewritten = rewrite_column_refs_in_expression(original, {'a': '__a__'})

        # Original tree untouched
        self.assertEqual(original.left.left.name, 'a')
        # Rewritten tree has the new name
        self.assertEqual(rewritten.left.left.name, '__a__')
        # Non-targeted field is unchanged
        self.assertEqual(rewritten.right.left.name, 'b')

    def test_rewrite_column_refs_handles_unary_in_between(self):
        """The helper recurses through UnaryCondition / InCondition /
        BetweenCondition with no exceptions and rewrites their nested
        Field references."""
        from datastore.sql_executor import rewrite_column_refs_in_expression
        from datastore.expressions import Field, Literal
        from datastore.conditions import (
            UnaryCondition,
            InCondition,
            BetweenCondition,
        )

        rename = {'col': '__col__'}

        unary = UnaryCondition('IS NULL', Field('col'))
        rewritten_unary = rewrite_column_refs_in_expression(unary, rename)
        self.assertEqual(rewritten_unary.expression.name, '__col__')

        in_cond = InCondition(Field('col'), [1, 2, 3])
        rewritten_in = rewrite_column_refs_in_expression(in_cond, rename)
        self.assertEqual(rewritten_in.expression.name, '__col__')

        between = BetweenCondition(Field('col'), Literal(1), Literal(10))
        rewritten_between = rewrite_column_refs_in_expression(between, rename)
        self.assertEqual(rewritten_between.expression.name, '__col__')


class TestDispatchByOpType(unittest.TestCase):
    """Unit tests for :meth:`SQLExecutionEngine._build_layer` op-type
    dispatch.

    These tests exercise the layer builder directly to confirm it dispatches
    on LazyGroupByAgg vs LazyWhere vs plain relational ops without relying
    on end-to-end execution.
    """

    def setUp(self):
        np.random.seed(0)
        self.n = 100
        self.df = pd.DataFrame(
            {
                'a': np.random.randint(0, 100, self.n),
                'b': np.random.randint(0, 100, self.n),
                'cat': np.random.choice(['X', 'Y'], self.n),
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_layer_with_groupby_agg_produces_group_by_sql(self):
        """A layer containing a WHERE + LazyGroupByAgg + ORDER BY must emit
        SQL containing GROUP BY and the aggregation function."""
        from datastore.lazy_ops import LazyRelationalOp, LazyGroupByAgg
        from datastore.sql_executor import SQLExecutionEngine
        from datastore.expressions import Field, Literal
        from datastore.conditions import BinaryCondition

        ds = DataStore.from_file(self.parquet_path)
        where = LazyRelationalOp(
            'WHERE', 'a > 5', condition=BinaryCondition('>', Field('a'), Literal(5))
        )
        agg = LazyGroupByAgg(groupby_cols=['cat'], agg_dict={'a': 'sum'})
        order = LazyRelationalOp('ORDER BY', '"cat"', fields=[Field('cat')], ascending=True)

        engine = SQLExecutionEngine(ds)
        result = engine._build_layer(
            [where, agg, order],
            from_source=None,
            is_first_layer=True,
            inherited_renames={},
            schema={},
            sql_select_fields=[],
        )
        sql = result.sql
        self.assertIn('GROUP BY', sql)
        self.assertIn('sum("a")', sql)
        self.assertIn('WHERE "a" > 5', sql)

    def test_layer_without_groupby_agg_skips_group_by(self):
        """A layer with only WHERE/ORDER BY/LIMIT must not emit GROUP BY."""
        from datastore.lazy_ops import LazyRelationalOp
        from datastore.sql_executor import SQLExecutionEngine
        from datastore.expressions import Field, Literal
        from datastore.conditions import BinaryCondition

        ds = DataStore.from_file(self.parquet_path)
        where = LazyRelationalOp(
            'WHERE', 'a > 5', condition=BinaryCondition('>', Field('a'), Literal(5))
        )
        limit = LazyRelationalOp('LIMIT', '10', limit_value=10)

        engine = SQLExecutionEngine(ds)
        result = engine._build_layer(
            [where, limit],
            from_source=None,
            is_first_layer=True,
            inherited_renames={},
            schema={},
            sql_select_fields=[],
        )
        sql = result.sql
        self.assertNotIn('GROUP BY', sql)
        self.assertIn('WHERE "a" > 5', sql)
        self.assertIn('LIMIT 10', sql)

    def test_extract_special_ops_finds_groupby_and_where_mask(self):
        """The helper that scans a layer for special ops should find
        LazyGroupByAgg and LazyWhere/LazyMask but not LazyRelationalOp."""
        from datastore.lazy_ops import (
            LazyRelationalOp,
            LazyGroupByAgg,
            LazyWhere,
            LazyMask,
        )
        from datastore.sql_executor import SQLExecutionEngine
        from datastore.expressions import Field, Literal
        from datastore.conditions import BinaryCondition

        ds = DataStore.from_file(self.parquet_path)
        engine = SQLExecutionEngine(ds)

        where_rel = LazyRelationalOp(
            'WHERE', 'a > 5', condition=BinaryCondition('>', Field('a'), Literal(5))
        )
        agg = LazyGroupByAgg(groupby_cols=['cat'], agg_func='sum')
        lazy_where = LazyWhere(
            BinaryCondition('>', Field('a'), Literal(0)), other=0
        )
        lazy_mask = LazyMask(
            BinaryCondition('>', Field('b'), Literal(0)), other=0
        )

        gb, wheres = engine._extract_special_ops_from_layer(
            [where_rel, agg, lazy_where, lazy_mask]
        )
        self.assertIs(gb, agg)
        self.assertEqual(wheres, [lazy_where, lazy_mask])

    def test_split_wheres_by_groupby_position(self):
        """WHEREs strictly before LazyGroupByAgg go to pre_agg (WHERE),
        WHEREs after go to post_agg (HAVING)."""
        from datastore.lazy_ops import LazyRelationalOp, LazyGroupByAgg
        from datastore.sql_executor import SQLExecutionEngine
        from datastore.expressions import Field, Literal
        from datastore.conditions import BinaryCondition

        ds = DataStore.from_file(self.parquet_path)
        engine = SQLExecutionEngine(ds)

        pre_where = LazyRelationalOp(
            'WHERE', 'a > 1', condition=BinaryCondition('>', Field('a'), Literal(1))
        )
        agg = LazyGroupByAgg(groupby_cols=['cat'], agg_dict={'a': 'sum'})
        post_where = LazyRelationalOp(
            'WHERE', 'a > 5', condition=BinaryCondition('>', Field('a'), Literal(5))
        )

        pre, post = engine._split_wheres_by_groupby_position(
            [pre_where, agg, post_where]
        )
        self.assertEqual(len(pre), 1)
        self.assertEqual(len(post), 1)
        # Pre-agg condition is on the source column
        self.assertIs(pre[0], pre_where.condition)
        # Post-agg condition is on the aggregation output
        self.assertIs(post[0], post_where.condition)


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
