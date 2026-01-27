"""
Tests for ColumnExpr.reset_index() method.

This tests the pandas-compatible reset_index() method on ColumnExpr,
particularly useful for groupby aggregation results.

Uses Mirror Code Pattern: exact same operations on both pandas and DataStore.
"""

import unittest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestColumnExprResetIndex(unittest.TestCase):
    """Test ColumnExpr.reset_index() alignment with pandas behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'C'],
            'subcategory': ['X', 'X', 'Y', 'Y', 'X', 'X'],
            'value': [10, 20, 30, 40, 50, 60]
        })

    def test_reset_index_with_name(self):
        """Test groupby().agg().reset_index(name='xxx') pattern."""
        # pandas
        pd_result = self.df.groupby('category')['value'].sum().reset_index(name='total')
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].sum().reset_index(name='total')
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_default(self):
        """Test groupby().agg().reset_index() without args."""
        # pandas
        pd_result = self.df.groupby('category')['value'].sum().reset_index()
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].sum().reset_index()
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_drop_true(self):
        """Test reset_index(drop=True) returns Series-like object."""
        # pandas
        pd_series = self.df.groupby('category')['value'].sum().reset_index(drop=True)
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].sum().reset_index(drop=True)
        
        # Compare values
        pd_values = pd_series.values
        ds_values = ds_result.values
        np.testing.assert_array_equal(np.sort(ds_values), np.sort(pd_values))

    def test_reset_index_mean(self):
        """Test reset_index with mean aggregation."""
        # pandas
        pd_result = self.df.groupby('category')['value'].mean().reset_index(name='avg_value')
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].mean().reset_index(name='avg_value')
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_count(self):
        """Test reset_index with count aggregation."""
        # pandas
        pd_result = self.df.groupby('category')['value'].count().reset_index(name='num_items')
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].count().reset_index(name='num_items')
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_multi_groupby(self):
        """Test reset_index with multi-column groupby."""
        # pandas
        pd_result = self.df.groupby(['category', 'subcategory'])['value'].sum().reset_index(name='total')
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby(['category', 'subcategory'])['value'].sum().reset_index(name='total')
        
        # Compare (row order may differ for multi-column groupby)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_reset_index_preserves_series_name(self):
        """Test that reset_index() without name arg uses series name."""
        # pandas
        pd_agg = self.df.groupby('category')['value'].sum()
        pd_result = pd_agg.reset_index()
        expected_columns = ['category', 'value']  # 'value' is the original series name
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_agg = ds.groupby('category')['value'].sum()
        ds_result = ds_agg.reset_index()
        
        # Check column names match
        self.assertEqual(list(ds_result.columns), expected_columns)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_inplace_raises(self):
        """Test that inplace=True raises ValueError."""
        ds = DataStore(self.df.copy())
        ds_agg = ds.groupby('category')['value'].sum()
        
        with self.assertRaises(ValueError) as context:
            ds_agg.reset_index(inplace=True)
        
        self.assertIn("inplace", str(context.exception).lower())

    def test_reset_index_max_aggregation(self):
        """Test reset_index with max aggregation."""
        # pandas
        pd_result = self.df.groupby('category')['value'].max().reset_index(name='max_val')
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].max().reset_index(name='max_val')
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_min_aggregation(self):
        """Test reset_index with min aggregation."""
        # pandas
        pd_result = self.df.groupby('category')['value'].min().reset_index(name='min_val')
        
        # DataStore (mirror)
        ds = DataStore(self.df.copy())
        ds_result = ds.groupby('category')['value'].min().reset_index(name='min_val')
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnExprResetIndexEdgeCases(unittest.TestCase):
    """Test edge cases for ColumnExpr.reset_index()."""

    def test_reset_index_empty_result(self):
        """Test reset_index on empty groupby result."""
        df = pd.DataFrame({'category': [], 'value': []})
        
        # pandas
        pd_result = df.groupby('category')['value'].sum().reset_index(name='total')
        
        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category')['value'].sum().reset_index(name='total')
        
        # Compare
        self.assertEqual(len(ds_result), 0)
        self.assertEqual(len(pd_result), 0)

    def test_reset_index_single_group(self):
        """Test reset_index with single group."""
        df = pd.DataFrame({'category': ['A', 'A', 'A'], 'value': [10, 20, 30]})
        
        # pandas
        pd_result = df.groupby('category')['value'].sum().reset_index(name='total')
        
        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category')['value'].sum().reset_index(name='total')
        
        # Compare
        assert_datastore_equals_pandas(ds_result, pd_result)

    @unittest.skip("Known limitation: chDB handles NaN in groupby differently than pandas (includes NaN groups)")
    def test_reset_index_with_nan_in_groupby(self):
        """Test reset_index when groupby column has NaN.
        
        Known limitation: chDB treats NaN as empty string in groupby,
        so it creates a group for NaN values while pandas excludes them by default.
        """
        df = pd.DataFrame({
            'category': ['A', 'B', None, 'A'],
            'value': [10, 20, 30, 40]
        })
        
        # pandas - dropna=True by default excludes NaN groups
        pd_result = df.groupby('category')['value'].sum().reset_index(name='total')
        
        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category')['value'].sum().reset_index(name='total')
        
        # Compare (NaN handling may differ, so just check non-NaN results)
        # Filter to only non-NaN categories for comparison
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


if __name__ == '__main__':
    unittest.main()
