"""
Real user journey: Mark Needham's Slack report on Amazon reviews pipeline.

Original report (paraphrased from Slack):
    import chdb.datastore as pd_chdb
    df_chdb = pd_chdb.read_parquet("amazon_sample.parquet")
    result = (df_chdb[df_chdb['verified_purchase'] == True]
        .groupby('product_category')['star_rating']
        .agg(['mean', 'count'])
        .sort_values('count', ascending=False)
        .head(n=10))
    # So far so good. But if I try to filter on count:
    result.filter(result['count'] > 50000)
    # E [chDB] Query failed: Unknown expression identifier `count` ...

He then tried boolean indexing, ``.loc[]``, and ``.query()`` looking for
something that would work. Three of those silently produced wrong /
crashed SQL because the dispatcher dropped the GROUP BY in the inner
subquery; ``.query()`` accidentally went through a different path that
happened to work.

This file mirrors Mark's exact code (renamed only the parquet column
generation to a deterministic synthetic dataset so the test is
self-contained) and pins down all four filter idioms plus a handful of
realistic follow-on operations a user would do after seeing the filter
work.

Once a real user pattern triggers a bug, it lives here permanently.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import chdb.datastore as pd_chdb
from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


class TestMarkSlackAmazonReviews(unittest.TestCase):
    """Verbatim regression of the Slack-reported amazon-reviews journey."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 50_000
        cls.df = pd.DataFrame(
            {
                'product_category': np.random.choice(
                    [
                        'Home',
                        'Digital_Ebook_Purchase',
                        'Apparel',
                        'Health & Personal Care',
                        'Books',
                        'Kitchen',
                        'Beauty',
                        'Mobile_Apps',
                        'Automotive',
                        'Home Improvement',
                        'Wireless',
                        'Sports',
                        'PC',
                        'Digital_Video_Download',
                    ],
                    n,
                ),
                'star_rating': np.random.choice([1, 2, 3, 4, 5], n),
                'verified_purchase': np.random.choice(
                    [True, False], n, p=[0.7, 0.3]
                ),
            }
        )
        cls.temp_dir = tempfile.mkdtemp()
        cls.parquet_path = os.path.join(cls.temp_dir, 'amazon_sample.parquet')
        cls.df.to_parquet(cls.parquet_path)

        # Mark's threshold scaled to the synthetic dataset (his was 50_000
        # on a 5M-row dump; we shrink for test speed). Picks roughly half
        # the categories so neither "empty result" nor "everything" hides
        # the bug.
        cls.threshold = 2500

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir)

    # ------- pandas reference, computed once -------------------------------

    def _pd_result(self):
        return (
            self.df[self.df['verified_purchase'] == True]
            .groupby('product_category')['star_rating']
            .agg(['mean', 'count'])
            .sort_values('count', ascending=False)
            .head(n=10)
        )

    def _ds_from_parquet(self):
        """Mirror of ``chdb.datastore.read_parquet(...)``."""
        return pd_chdb.read_parquet(self.parquet_path)

    def _ds_result_from_parquet(self):
        df_chdb = self._ds_from_parquet()
        return (
            df_chdb[df_chdb['verified_purchase'] == True]
            .groupby('product_category')['star_rating']
            .agg(['mean', 'count'])
            .sort_values('count', ascending=False)
            .head(n=10)
        )

    def _ds_result_from_dataframe(self):
        """Same pipeline but reading from an in-memory DataFrame source.

        Mark used ``read_parquet``; users who load via ``DataStore(df)``
        hit the Python() table function path. The same dispatch bug
        existed there too - this mirror covers both source types.
        """
        ds = DataStore(self.df)
        return (
            ds[ds['verified_purchase'] == True]
            .groupby('product_category')['star_rating']
            .agg(['mean', 'count'])
            .sort_values('count', ascending=False)
            .head(n=10)
        )

    # ------- the initial aggregate (the "So far so good" step) -------------

    def test_initial_aggregate_parquet_matches_pandas(self):
        assert_datastore_equals_pandas(
            self._ds_result_from_parquet(),
            self._pd_result(),
            check_index=True,
        )

    def test_initial_aggregate_dataframe_matches_pandas(self):
        assert_datastore_equals_pandas(
            self._ds_result_from_dataframe(),
            self._pd_result(),
            check_index=True,
        )

    # ------- the four filter idioms Mark tried in order -------------------

    def test_filter_with_bracket_mask_parquet(self):
        """``result[result['count'] > N]`` - Pandonic boolean indexing."""
        pd_filtered = self._pd_result()[self._pd_result()['count'] > self.threshold]
        ds_result = self._ds_result_from_parquet()
        ds_filtered = ds_result[ds_result['count'] > self.threshold]
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    def test_filter_with_loc_parquet(self):
        """``result.loc[result['count'] > N]`` - the documented pandas way."""
        pd_result = self._pd_result()
        pd_filtered = pd_result.loc[pd_result['count'] > self.threshold]
        ds_result = self._ds_result_from_parquet()
        ds_filtered = ds_result.loc[ds_result['count'] > self.threshold]
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    def test_filter_with_query_parquet(self):
        """``result.query('count > N')`` - the only one that accidentally
        used to work, because it took a different code path that did not
        share the broken layer dispatcher."""
        pd_result = self._pd_result()
        pd_filtered = pd_result.query(f'count > {self.threshold}')
        ds_result = self._ds_result_from_parquet()
        ds_filtered = ds_result.query(f'count > {self.threshold}')
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    def test_filter_with_filter_method_parquet(self):
        """``result.filter(result['count'] > N)`` - the call Mark tried
        first. In pandas, ``DataFrame.filter`` only takes labels and would
        reject a boolean mask; chdb-ds intentionally extends it to support
        boolean masks, so we compare against pandas boolean indexing
        (same semantics)."""
        pd_result = self._pd_result()
        pd_filtered = pd_result[pd_result['count'] > self.threshold]
        ds_result = self._ds_result_from_parquet()
        ds_filtered = ds_result.filter(ds_result['count'] > self.threshold)
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    # ------- same four idioms on the DataFrame source path ----------------

    def test_filter_with_bracket_mask_dataframe(self):
        pd_filtered = self._pd_result()[self._pd_result()['count'] > self.threshold]
        ds_result = self._ds_result_from_dataframe()
        ds_filtered = ds_result[ds_result['count'] > self.threshold]
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    def test_filter_with_loc_dataframe(self):
        pd_result = self._pd_result()
        pd_filtered = pd_result.loc[pd_result['count'] > self.threshold]
        ds_result = self._ds_result_from_dataframe()
        ds_filtered = ds_result.loc[ds_result['count'] > self.threshold]
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    def test_filter_with_query_dataframe(self):
        pd_result = self._pd_result()
        pd_filtered = pd_result.query(f'count > {self.threshold}')
        ds_result = self._ds_result_from_dataframe()
        ds_filtered = ds_result.query(f'count > {self.threshold}')
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    def test_filter_with_filter_method_dataframe(self):
        pd_result = self._pd_result()
        pd_filtered = pd_result[pd_result['count'] > self.threshold]
        ds_result = self._ds_result_from_dataframe()
        ds_filtered = ds_result.filter(ds_result['count'] > self.threshold)
        assert_datastore_equals_pandas(
            ds_filtered, pd_filtered, check_index=True
        )

    # ------- realistic things a user does AFTER the filter works ----------

    def test_filter_then_select_aggregate_col(self):
        """``result[mask]['count']`` - grab the count Series for further
        analysis. This exercises the column projection through the temp
        alias rewrite that was the trickiest part of the cross-layer
        rewrite fix."""
        pd_result = self._pd_result()
        pd_series = pd_result[pd_result['count'] > self.threshold]['count']
        ds_result = self._ds_result_from_parquet()
        ds_series = ds_result[ds_result['count'] > self.threshold]['count']
        np.testing.assert_array_equal(list(ds_series), list(pd_series))

    def test_filter_then_sort_by_other_agg_col(self):
        """Sort the filtered result by ``mean`` instead of ``count``."""
        pd_result = self._pd_result()
        pd_sorted = (
            pd_result[pd_result['count'] > self.threshold]
            .sort_values('mean', ascending=False)
        )
        ds_result = self._ds_result_from_parquet()
        ds_sorted = (
            ds_result[ds_result['count'] > self.threshold]
            .sort_values('mean', ascending=False)
        )
        assert_datastore_equals_pandas(
            ds_sorted, pd_sorted, check_index=True
        )

    def test_filter_then_head_smaller(self):
        """Filter then take the top 3 of what remains."""
        pd_result = self._pd_result()
        pd_top = pd_result[pd_result['count'] > self.threshold].head(3)
        ds_result = self._ds_result_from_parquet()
        ds_top = ds_result[ds_result['count'] > self.threshold].head(3)
        assert_datastore_equals_pandas(ds_top, pd_top, check_index=True)

    def test_chained_post_filter_operations(self):
        """A natural longer chain after the agg+filter step:
        ``result[result['count'] > N].sort_values('mean', desc).head(3)['mean']``."""
        pd_result = self._pd_result()
        pd_series = (
            pd_result[pd_result['count'] > self.threshold]
            .sort_values('mean', ascending=False)
            .head(3)['mean']
        )
        ds_result = self._ds_result_from_parquet()
        ds_series = (
            ds_result[ds_result['count'] > self.threshold]
            .sort_values('mean', ascending=False)
            .head(3)['mean']
        )
        np.testing.assert_allclose(
            list(ds_series), list(pd_series), rtol=1e-5
        )

    # ------- a related dispatcher bug uncovered by hypothesis -----------

    def test_head_before_agg_respects_limit(self):
        """``df[mask][mask].head(1).groupby('cat').agg({'v': 'sum'})``
        respects the ``head(1)`` and aggregates exactly that one row,
        just like pandas. Hypothesis discovered the original bug (LIMIT
        was folded into the same SQL layer as GROUP BY, limiting the
        number of GROUPS instead of input rows). Fixed by splitting
        ``LazyGroupByAgg`` into a new nested layer when it follows a
        LIMIT/OFFSET in ``QueryPlanner._build_layers``.
        """
        import numpy as np

        rng = np.random.default_rng(20260)
        n = 400
        df = pd.DataFrame(
            {
                'cat': rng.choice(['A', 'B', 'C', 'D'], n),
                'v': rng.integers(0, 100, n),
            }
        )
        pd_result = (
            df[df['v'] > 0][df[df['v'] > 0]['v'] > 0]
            .head(1)
            .groupby('cat')
            .agg({'v': 'sum'})
        )

        ds = DataStore(df)
        ds_result = (
            ds[ds['v'] > 0][ds[ds['v'] > 0]['v'] > 0]
            .head(1)
            .groupby('cat')
            .agg({'v': 'sum'})
        )
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_index=True, check_row_order=False
        )

    # ------- the structural assertion: SQL really has the GROUP BY -------

    def test_inner_subquery_keeps_group_by_after_post_agg_filter(self):
        """The original bug: the inner subquery dropped GROUP BY entirely.
        Snapshot the generated SQL and assert structural properties."""
        from datastore.query_planner import QueryPlanner
        from datastore.sql_executor import SQLExecutionEngine

        ds_result = self._ds_result_from_parquet()
        ds_filtered = ds_result[ds_result['count'] > self.threshold]

        planner = QueryPlanner()
        exec_plan = planner.plan_segments(
            ds_filtered._lazy_ops, has_sql_source=True, schema=None
        )
        self.assertEqual(len(exec_plan.segments), 1)
        seg = exec_plan.segments[0]
        self.assertEqual(seg.segment_type, 'sql')

        engine = SQLExecutionEngine(ds_filtered)
        sql = engine.build_sql_from_plan(seg.plan, schema={}).sql

        # The whole point of the bug: inner subquery must contain GROUP BY
        # and the aggregate calls, otherwise the outer WHERE on the agg
        # alias fails with UNKNOWN_IDENTIFIER.
        self.assertIn('GROUP BY', sql)
        self.assertIn('avg("star_rating")', sql)
        self.assertIn('count("star_rating")', sql)
        # And the outer filter is preserved.
        self.assertIn(f'> {self.threshold}', sql)


if __name__ == '__main__':
    unittest.main()
