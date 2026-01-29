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


class TestMultiIndexColumnMerge(unittest.TestCase):
    """Test merging DataFrames with MultiIndex columns.
    
    pandas merge() doesn't support merging DataFrames with different column
    index levels. DataStore auto-flattens MultiIndex columns to enable merge.
    
    These tests follow the Mirror Code Pattern where applicable.
    """

    def setUp(self):
        """Set up test data."""
        self.users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        self.orders = pd.DataFrame({
            'order_id': [1, 2, 3, 4, 5],
            'user_id': [1, 1, 2, 2, 3],
            'amount': [100, 200, 150, 250, 300]
        })

    def test_multiindex_merge_with_flat_dataframe(self):
        """Test merging MultiIndex columns DataFrame with flat DataFrame."""
        # Create user stats with MultiIndex columns via groupby().agg()
        # pandas doesn't support this merge directly - it raises MergeError
        # DataStore should auto-flatten and succeed
        ds_users = DataStore(self.users)
        ds_orders = DataStore(self.orders)
        ds_user_stats = ds_orders.groupby('user_id').agg({'amount': ['sum', 'mean', 'count']})
        
        # Module-level merge
        from datastore import merge
        result = merge(ds_users, ds_user_stats, left_on='user_id', right_index=True)
        
        # Natural execution via .columns and len()
        expected_columns = ['user_id', 'name', 'amount_sum', 'amount_mean', 'amount_count']
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 3)
        
        # Check data values with natural execution via indexing
        # Alice: user_id=1 has orders 100, 200 -> sum=300, mean=150, count=2
        alice_row = result[result['name'] == 'Alice']
        self.assertEqual(alice_row['amount_sum'].values[0], 300)
        self.assertEqual(alice_row['amount_mean'].values[0], 150.0)
        self.assertEqual(alice_row['amount_count'].values[0], 2)
        
        # Bob: user_id=2 has orders 150, 250 -> sum=400, mean=200, count=2
        bob_row = result[result['name'] == 'Bob']
        self.assertEqual(bob_row['amount_sum'].values[0], 400)
        self.assertEqual(bob_row['amount_mean'].values[0], 200.0)
        self.assertEqual(bob_row['amount_count'].values[0], 2)
        
        # Charlie: user_id=3 has order 300 -> sum=300, mean=300, count=1
        charlie_row = result[result['name'] == 'Charlie']
        self.assertEqual(charlie_row['amount_sum'].values[0], 300)
        self.assertEqual(charlie_row['amount_mean'].values[0], 300.0)
        self.assertEqual(charlie_row['amount_count'].values[0], 1)

    def test_multiindex_merge_instance_method(self):
        """Test instance method merge with MultiIndex columns."""
        ds_users = DataStore(self.users)
        ds_orders = DataStore(self.orders)
        ds_user_stats = ds_orders.groupby('user_id').agg({'amount': ['sum', 'mean', 'count']})
        
        # Instance method merge
        result = ds_users.merge(ds_user_stats, left_on='user_id', right_index=True)
        
        # Natural execution via .columns and len()
        expected_columns = ['user_id', 'name', 'amount_sum', 'amount_mean', 'amount_count']
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(len(result), 3)
        
        # Check data values for Bob: user_id=2 has orders 150, 250 -> sum=400, mean=200, count=2
        bob_row = result[result['name'] == 'Bob']
        self.assertEqual(bob_row['amount_sum'].values[0], 400)
        self.assertEqual(bob_row['amount_mean'].values[0], 200.0)
        self.assertEqual(bob_row['amount_count'].values[0], 2)

    def test_multiindex_merge_after_reset_index(self):
        """Test merge after reset_index() which preserves MultiIndex columns."""
        ds_users = DataStore(self.users)
        ds_orders = DataStore(self.orders)
        ds_user_stats = ds_orders.groupby('user_id').agg({'amount': ['sum', 'mean', 'count']})
        
        # reset_index converts index to column but keeps MultiIndex columns
        ds_user_stats_reset = ds_user_stats.reset_index()
        
        # Merge on column instead of index
        from datastore import merge
        result = merge(ds_users, ds_user_stats_reset, on='user_id')
        
        # Natural execution via .columns and len()
        self.assertIn('user_id', list(result.columns))
        self.assertIn('name', list(result.columns))
        self.assertEqual(len(result), 3)
        
        # Verify flattened columns exist
        self.assertIn('amount_sum', list(result.columns))
        self.assertIn('amount_mean', list(result.columns))
        self.assertIn('amount_count', list(result.columns))
        
        # Verify data values via indexing (natural execution)
        # Alice: user_id=1 has orders 100, 200 -> sum=300, mean=150, count=2
        alice_row = result[result['name'] == 'Alice']
        self.assertEqual(alice_row['amount_sum'].values[0], 300)
        self.assertEqual(alice_row['amount_mean'].values[0], 150.0)
        self.assertEqual(alice_row['amount_count'].values[0], 2)
        
        # Charlie: user_id=3 has order 300 -> sum=300, mean=300, count=1
        charlie_row = result[result['name'] == 'Charlie']
        self.assertEqual(charlie_row['amount_sum'].values[0], 300)
        self.assertEqual(charlie_row['amount_mean'].values[0], 300.0)
        self.assertEqual(charlie_row['amount_count'].values[0], 1)

    def test_flat_columns_merge_unchanged(self):
        """Verify that merges with flat columns still work correctly.
        
        Uses mirror pattern - compare DataStore result with pandas.
        """
        # pandas: single func aggregation produces flat columns
        pd_user_stats = self.orders.groupby('user_id').agg({'amount': 'sum'})
        pd_result = self.users.merge(pd_user_stats, left_on='user_id', right_index=True)
        
        # DataStore (mirror)
        ds_users = DataStore(self.users)
        ds_orders = DataStore(self.orders)
        ds_user_stats = ds_orders.groupby('user_id').agg({'amount': 'sum'})
        
        from datastore import merge
        ds_result = merge(ds_users, ds_user_stats, left_on='user_id', right_index=True)
        
        # Natural execution via .columns
        self.assertEqual(list(ds_result.columns), ['user_id', 'name', 'amount'])
        self.assertEqual(len(ds_result), 3)
        
        # Verify data matches pandas (sort for consistent comparison)
        from tests.test_utils import assert_datastore_equals_pandas
        assert_datastore_equals_pandas(
            ds_result.sort_values('user_id').reset_index(drop=True), 
            pd_result.sort_values('user_id').reset_index(drop=True),
            check_row_order=True
        )

    def test_both_multiindex_same_levels(self):
        """Test merge when both DataFrames have MultiIndex columns with same levels.
        
        When both DataFrames have MultiIndex columns with same levels, the merge
        preserves the MultiIndex structure. Column names are tuples like 
        ('user_id', ''), ('amount', 'sum'), etc.
        """
        # Create two DataFrames with MultiIndex columns
        ds_orders = DataStore(self.orders)
        stats1 = ds_orders.groupby('user_id').agg({'amount': ['sum', 'mean']})
        stats2 = ds_orders.groupby('user_id').agg({'amount': ['min', 'max']})
        
        # Merge two MultiIndex column DataFrames
        from datastore import merge
        result = merge(stats1.reset_index(), stats2.reset_index(), on='user_id')
        
        # Natural execution via len() and columns
        self.assertEqual(len(result), 3)
        
        # When both have same MultiIndex levels, columns remain as tuples
        columns = list(result.columns)
        self.assertIn(('user_id', ''), columns)
        self.assertIn(('amount', 'sum'), columns)
        self.assertIn(('amount', 'mean'), columns)
        self.assertIn(('amount', 'min'), columns)
        self.assertIn(('amount', 'max'), columns)
        
        # Verify data values for each user via tuple column indexing (natural execution)
        # User 1: orders 100, 200 -> sum=300, mean=150, min=100, max=200
        user1_row = result[result[('user_id', '')] == 1]
        self.assertEqual(user1_row[('amount', 'sum')].values[0], 300)
        self.assertEqual(user1_row[('amount', 'mean')].values[0], 150.0)
        self.assertEqual(user1_row[('amount', 'min')].values[0], 100)
        self.assertEqual(user1_row[('amount', 'max')].values[0], 200)
        
        # User 2: orders 150, 250 -> sum=400, mean=200, min=150, max=250
        user2_row = result[result[('user_id', '')] == 2]
        self.assertEqual(user2_row[('amount', 'sum')].values[0], 400)
        self.assertEqual(user2_row[('amount', 'mean')].values[0], 200.0)
        self.assertEqual(user2_row[('amount', 'min')].values[0], 150)
        self.assertEqual(user2_row[('amount', 'max')].values[0], 250)
        
        # User 3: order 300 -> sum=300, mean=300, min=300, max=300
        user3_row = result[result[('user_id', '')] == 3]
        self.assertEqual(user3_row[('amount', 'sum')].values[0], 300)
        self.assertEqual(user3_row[('amount', 'mean')].values[0], 300.0)
        self.assertEqual(user3_row[('amount', 'min')].values[0], 300)
        self.assertEqual(user3_row[('amount', 'max')].values[0], 300)
