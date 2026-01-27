"""
Tests for GroupBy aggregation MultiIndex column names.

These tests verify:
1. groupby().agg({col: [funcs]}) returns MultiIndex columns matching pandas
2. Different scenarios: single column, multiple columns, mixed functions
3. Named aggregation behavior
4. as_index parameter behavior

The key fix tested here is ensuring DataStore returns MultiIndex columns
when pandas does, using (column, function) tuples as column names.
"""

import unittest

import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal


class TestGroupByMultiIndexColumns(unittest.TestCase):
    """Test GroupBy aggregation with MultiIndex column names."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'subcategory': ['X', 'X', 'Y', 'Y', 'X'],
            'value': [10, 20, 30, 40, 50],
            'count': [1, 2, 3, 4, 5]
        })

    def test_single_func_per_column_no_multiindex(self):
        """Single function per column should NOT return MultiIndex."""
        pd_result = self.df.groupby('category').agg({'value': 'mean', 'count': 'sum'})
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category').agg({'value': 'mean', 'count': 'sum'}).to_df()

        # Check columns are NOT MultiIndex
        self.assertFalse(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertFalse(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check values match
        assert_frame_equal(ds_result, pd_result)

    def test_multiple_funcs_single_column_multiindex(self):
        """Multiple functions on single column should return MultiIndex."""
        pd_result = self.df.groupby('category').agg({'value': ['mean', 'sum']})
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category').agg({'value': ['mean', 'sum']}).to_df()

        # Check columns ARE MultiIndex
        self.assertTrue(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check values match
        assert_frame_equal(ds_result, pd_result)

    def test_multiple_funcs_multiple_columns_multiindex(self):
        """Multiple functions on multiple columns should return MultiIndex."""
        pd_result = self.df.groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': ['min', 'max']
        })
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': ['min', 'max']
        }).to_df()

        # Check columns ARE MultiIndex
        self.assertTrue(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check values match
        assert_frame_equal(ds_result, pd_result)

    def test_mixed_single_and_multiple_funcs_multiindex(self):
        """Mix of single and multiple functions should return MultiIndex."""
        pd_result = self.df.groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': 'max'
        })
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': 'max'
        }).to_df()

        # Check columns ARE MultiIndex (pandas returns MultiIndex if ANY column has multiple funcs)
        self.assertTrue(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check values match
        assert_frame_equal(ds_result, pd_result)

    def test_multiindex_column_access(self):
        """Accessing MultiIndex columns should work like pandas."""
        pd_result = self.df.groupby('category').agg({'value': ['mean', 'sum']})
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category').agg({'value': ['mean', 'sum']}).to_df()

        # Access via tuple
        assert_series_equal(
            ds_result[('value', 'mean')], 
            pd_result[('value', 'mean')]
        )
        
        # Access via hierarchical
        assert_series_equal(
            ds_result['value']['mean'], 
            pd_result['value']['mean']
        )

    def test_named_aggregation_no_multiindex(self):
        """Named aggregation should NOT return MultiIndex."""
        pd_result = self.df.groupby('category').agg(
            avg_val=('value', 'mean'), 
            total_cnt=('count', 'sum')
        )
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category').agg(
            avg_val=('value', 'mean'), 
            total_cnt=('count', 'sum')
        ).to_df()

        # Check columns are NOT MultiIndex
        self.assertFalse(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertFalse(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check index is set correctly
        self.assertEqual(pd_result.index.name, ds_result.index.name)
        self.assertEqual(pd_result.index.tolist(), ds_result.index.tolist())
        
        # Check values match
        assert_frame_equal(ds_result, pd_result)

    def test_named_aggregation_as_index_false(self):
        """Named aggregation with as_index=False should keep groupby col as column."""
        pd_result = self.df.groupby('category', as_index=False).agg(
            avg_val=('value', 'mean'), 
            total_cnt=('count', 'sum')
        )
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby('category', as_index=False).agg(
            avg_val=('value', 'mean'), 
            total_cnt=('count', 'sum')
        ).to_df()

        # Check columns match (category should be a column, not index)
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        self.assertIn('category', ds_result.columns)
        
        # Check values match
        assert_frame_equal(ds_result, pd_result)

    def test_multiindex_with_multiple_groupby_columns(self):
        """MultiIndex columns with multiple groupby keys should work."""
        pd_result = self.df.groupby(['category', 'subcategory']).agg({
            'value': ['mean', 'sum']
        })
        ds = DataStore.from_dataframe(self.df)
        ds_result = ds.groupby(['category', 'subcategory']).agg({
            'value': ['mean', 'sum']
        }).to_df()

        # Check columns ARE MultiIndex
        self.assertTrue(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check index is MultiIndex (groupby cols as index)
        self.assertTrue(isinstance(pd_result.index, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.index, pd.MultiIndex))
        
        # Check values match (sort for consistent comparison)
        assert_frame_equal(
            ds_result.sort_index(), 
            pd_result.sort_index()
        )


class TestGroupByMultiIndexFromFile(unittest.TestCase):
    """Test GroupBy MultiIndex columns when loading from file."""

    def setUp(self):
        """Set up test data and create temp parquet file."""
        import tempfile
        import os
        
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80],
            'count': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.parquet_path, index=False)

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_multiindex_from_parquet(self):
        """MultiIndex columns should work when loading from parquet."""
        pd_result = self.df.groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': ['min', 'max']
        })
        
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds.groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': ['min', 'max']
        }).to_df()

        # Check columns ARE MultiIndex
        self.assertTrue(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())
        
        # Check values match
        assert_frame_equal(
            ds_result.sort_index(), 
            pd_result.sort_index()
        )

    def test_multiindex_with_filter(self):
        """MultiIndex columns should work after filtering."""
        pd_result = self.df[self.df['value'] > 30].groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': 'max'
        })
        
        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[ds['value'] > 30].groupby('category').agg({
            'value': ['mean', 'sum'], 
            'count': 'max'
        }).to_df()

        # Check columns ARE MultiIndex
        self.assertTrue(isinstance(pd_result.columns, pd.MultiIndex))
        self.assertTrue(isinstance(ds_result.columns, pd.MultiIndex))
        self.assertEqual(pd_result.columns.tolist(), ds_result.columns.tolist())


if __name__ == '__main__':
    unittest.main()
