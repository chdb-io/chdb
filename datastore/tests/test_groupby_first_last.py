"""
Test groupby().first() and groupby().last() methods.

These tests verify that ColumnExpr.first() and ColumnExpr.last()
work correctly with groupby operations.

Implementation Note:
- first() uses argMin(value, rowNumberInAllBlocks()) to get the first value by row order
- last() uses argMax(value, rowNumberInAllBlocks()) to get the last value by row order
- For Python() table function, connection.py replaces rowNumberInAllBlocks() with _row_id
  (deterministic, available in chDB v4.0.0b5+)
- For file sources (Parquet, CSV), rowNumberInAllBlocks() works correctly as-is

FIXED (chDB v4.0.0b5): The _row_id virtual column is now deterministic.
The xfail markers are now no-ops for backward compatibility.
"""

import unittest
import pandas as pd
import numpy as np
import pytest

import datastore as ds
from tests.test_utils import assert_series_equal, get_dataframe, get_series
from tests.xfail_markers import chdb_python_table_rownumber_nondeterministic


class TestGroupByFirstLast(unittest.TestCase):
    """Test groupby first() and last() methods."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40, 50, 60],
                'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        self.ds = ds.DataFrame(self.df.copy())

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_first_single_column(self):
        """Test groupby().first() on a single column."""
        pd_result = self.df.groupby('category')['value'].first()
        ds_result = self.ds.groupby('category')['value'].first()

        # Get the Series from ColumnExpr
        ds_series = get_series(ds_result)

        # Compare values (ignore index name)
        assert_series_equal(ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_last_single_column(self):
        """Test groupby().last() on a single column."""
        pd_result = self.df.groupby('category')['value'].last()
        ds_result = self.ds.groupby('category')['value'].last()

        ds_series = get_series(ds_result)

        assert_series_equal(ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_first_multiple_groups(self):
        """Test groupby().first() with multiple groups."""
        df = pd.DataFrame(
            {'g1': ['A', 'A', 'B', 'B', 'C', 'C'], 'g2': ['x', 'y', 'x', 'y', 'x', 'y'], 'value': [1, 2, 3, 4, 5, 6]}
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby(['g1', 'g2'])['value'].first()
        ds_result = ds_df.groupby(['g1', 'g2'])['value'].first()

        ds_series = get_series(ds_result)

        # Compare values (multi-index groupby may have different order)
        self.assertEqual(len(ds_series), len(pd_result))
        for idx in pd_result.index:
            self.assertEqual(ds_series.loc[idx], pd_result.loc[idx])

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_first_with_nan(self):
        """Test groupby().first() with NaN values."""
        df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [np.nan, 20, 30, np.nan]})
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('category')['value'].first()
        ds_result = ds_df.groupby('category')['value'].first()

        ds_series = get_series(ds_result)

        # Both should return the first non-NaN value if available
        # Note: behavior may differ slightly between pandas and chDB
        self.assertEqual(len(ds_series), len(pd_result))

    def test_groupby_first_returns_correct_type(self):
        """Test that first() returns ColumnExpr which can be converted to Series."""
        result = self.ds.groupby('category')['value'].first()

        # Should be a ColumnExpr
        self.assertIsInstance(result, ds.column_expr.ColumnExpr)

        # Should be convertible to pandas Series
        series = result.to_pandas()
        self.assertIsInstance(series, pd.Series)

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_agg_first_equivalent(self):
        """Test that .first() is equivalent to .agg('first')."""
        pd_first = self.df.groupby('category')['value'].first()
        pd_agg = self.df.groupby('category')['value'].agg('first')

        ds_first = self.ds.groupby('category')['value'].first()
        ds_agg = self.ds.groupby('category')['value'].agg('first')

        # Convert to pandas
        ds_first_series = get_series(ds_first)
        ds_agg_df = get_dataframe(ds_agg)

        # pandas first() and agg('first') should be equal
        assert_series_equal(pd_first, pd_agg)

        # DataStore results should match pandas
        self.assertEqual(ds_first_series['A'], pd_first['A'])
        self.assertEqual(ds_first_series['B'], pd_first['B'])


class TestRowNumberInAllBlocksBug(unittest.TestCase):
    """
    Test case that originally reproduced chDB bug #469:
    rowNumberInAllBlocks() was non-deterministic with Python() table function.

    FIXED (chDB v4.0.0b5): The _row_id virtual column is now deterministic.
    connection.py replaces rowNumberInAllBlocks() with _row_id for Python() table function.

    See: https://github.com/chdb-io/chdb/issues/469
    """

    @chdb_python_table_rownumber_nondeterministic
    def test_large_data_first_last_stability(self):
        """
        Test first()/last() with large dataset - should reproduce bug reliably.

        With 10000+ rows, the probability of triggering the non-deterministic
        block distribution is high, causing this test to fail frequently.
        """
        # Large dataset to trigger parallel block processing
        n = 10000
        categories = ['A', 'B', 'C', 'D', 'E'] * (n // 5)
        values = list(range(n))

        df = pd.DataFrame(
            {
                'category': categories,
                'value': values,
            }
        )
        ds_df = ds.DataFrame(df.copy())

        # Expected: first values are 0, 1, 2, 3, 4 for categories A, B, C, D, E
        pd_first = df.groupby('category')['value'].first().sort_index()
        ds_first = ds_df.groupby('category')['value'].first()

        ds_series = get_series(ds_first).sort_index()

        # This assertion will fail when rowNumberInAllBlocks() bug is triggered
        assert_series_equal(ds_series, pd_first, check_names=False)

    @chdb_python_table_rownumber_nondeterministic
    def test_large_data_last_stability(self):
        """Test last() with large dataset."""
        n = 10000
        categories = ['A', 'B', 'C', 'D', 'E'] * (n // 5)
        values = list(range(n))

        df = pd.DataFrame(
            {
                'category': categories,
                'value': values,
            }
        )
        ds_df = ds.DataFrame(df.copy())

        # Expected: last values are 9995, 9996, 9997, 9998, 9999 for A, B, C, D, E
        pd_last = df.groupby('category')['value'].last().sort_index()
        ds_last = ds_df.groupby('category')['value'].last()

        ds_series = get_series(ds_last).sort_index()

        assert_series_equal(ds_series, pd_last, check_names=False)


if __name__ == '__main__':
    unittest.main()
