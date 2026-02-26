"""
Tests for head() performance optimization.

The head() operation should NOT add ORDER BY when there's no filter (WHERE),
because the source naturally provides rows in order and adding ORDER BY
causes unnecessary sorting overhead.

However, filter + head() should still use ORDER BY to ensure correct results.
"""
import unittest
import tempfile
import os
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestHeadPerformanceOptimization(unittest.TestCase):
    """Test that head() optimization works correctly."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': ['x', 'y', 'z', 'w', 'v', 'u', 't', 's', 'r', 'q']
        })

    def test_simple_head_matches_pandas(self):
        """Simple head() should match pandas without ORDER BY."""
        pd_df = self.df.copy()
        ds = DataStore(pd_df)

        pd_result = pd_df.head(3)
        ds_result = ds.head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_simple_head_default_n(self):
        """head() with default n=5 should match pandas."""
        pd_df = self.df.copy()
        ds = DataStore(pd_df)

        pd_result = pd_df.head()
        ds_result = ds.head()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_head_matches_pandas(self):
        """filter + head should match pandas and preserve row order."""
        pd_df = self.df.copy()
        ds = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 5].head(3)
        ds_result = ds[ds['a'] > 5].head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_then_filter_matches_pandas(self):
        """head + filter should match pandas (take first n, then filter)."""
        pd_df = self.df.copy()
        ds = DataStore(pd_df)

        pd_result = pd_df.head(7)[pd_df.head(7)['a'] > 3]
        ds_result = ds.head(7)[ds.head(7)['a'] > 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_head_matches_pandas(self):
        """sort + head should match pandas (order by value, then take first n)."""
        pd_df = self.df.copy()
        ds = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', ascending=False).head(3)
        ds_result = ds.sort_values('a', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_on_empty_dataframe(self):
        """head() on empty DataFrame should match pandas."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds = DataStore(pd_df)

        pd_result = pd_df.head(5)
        ds_result = ds.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_larger_than_dataframe(self):
        """head(n) where n > len(df) should return all rows."""
        pd_df = self.df.copy()
        ds = DataStore(pd_df)

        pd_result = pd_df.head(100)
        ds_result = ds.head(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_with_parquet_file(self):
        """head() on Parquet file should match pandas."""
        # Create temp parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
        self.df.to_parquet(parquet_path)

        try:
            pd_df = pd.read_parquet(parquet_path)
            ds = DataStore(parquet_path)

            pd_result = pd_df.head(5)
            ds_result = ds.head(5)

            assert_datastore_equals_pandas(ds_result, pd_result)
        finally:
            os.unlink(parquet_path)

    def test_filter_head_parquet_matches_pandas(self):
        """filter + head on Parquet file should match pandas."""
        # Create temp parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
        self.df.to_parquet(parquet_path)

        try:
            pd_df = pd.read_parquet(parquet_path)
            ds = DataStore(parquet_path)

            pd_result = pd_df[pd_df['a'] > 3].head(4)
            ds_result = ds[ds['a'] > 3].head(4)

            assert_datastore_equals_pandas(ds_result, pd_result)
        finally:
            os.unlink(parquet_path)


class TestHeadRowOrderPreservation(unittest.TestCase):
    """Test that row order is preserved correctly with and without filters."""

    def test_head_preserves_original_order(self):
        """head() should preserve original row order from source."""
        # Create DataFrame with specific row order
        df = pd.DataFrame({
            'id': [10, 20, 30, 40, 50],
            'value': ['a', 'b', 'c', 'd', 'e']
        })
        ds = DataStore(df)

        pd_result = df.head(3)
        ds_result = ds.head(3)

        # Check exact row order
        assert_datastore_equals_pandas(ds_result, pd_result)
        # Explicitly verify order
        ds_df = ds_result.to_df()
        self.assertEqual(list(ds_df['id']), [10, 20, 30])
        self.assertEqual(list(ds_df['value']), ['a', 'b', 'c'])

    def test_filter_head_preserves_order_after_filter(self):
        """filter + head should preserve order of filtered rows."""
        # Create DataFrame with specific row order
        df = pd.DataFrame({
            'id': [10, 20, 30, 40, 50, 60],
            'value': [1, 5, 2, 6, 3, 7]
        })
        ds = DataStore(df)

        # Filter for value > 3, then head(2)
        # Original order of rows with value > 3: 20 (5), 40 (6), 60 (7)
        # head(2) should give: 20 (5), 40 (6)
        pd_result = df[df['value'] > 3].head(2)
        ds_result = ds[ds['value'] > 3].head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)
        ds_df = ds_result.to_df()
        self.assertEqual(list(ds_df['id']), [20, 40])
        self.assertEqual(list(ds_df['value']), [5, 6])


if __name__ == '__main__':
    unittest.main()
