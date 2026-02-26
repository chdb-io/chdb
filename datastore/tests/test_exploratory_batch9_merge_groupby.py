"""
Exploratory tests: Batch 9 - Merge, GroupBy, and Complex Operations

This batch covers:
1. Merge/Join edge cases (pandas-style merge API)
2. Concat, apply, transform operations
3. Pivot, melt, stack operations
4. Window functions (rolling, expanding, ewm)
5. Complex groupby and chained operations

Discovery date: 2026-01-04
Fixed issues:
- groupby().ngroups returning string instead of int
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_series


class TestMergeEdgeCases:
    """Test merge/join operations with various edge cases."""

    def test_basic_inner_merge(self):
        """Test basic inner merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
        df2 = pd.DataFrame({'key': ['B', 'C', 'D', 'E'], 'value2': [20, 30, 40, 50]})
        ds1, ds2 = DataStore(df1), DataStore(df2)

        pd_result = pd.merge(df1, df2, on='key')
        ds_result = ds1.merge(ds2, on='key')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_left_merge(self):
        """Test left merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
        df2 = pd.DataFrame({'key': ['B', 'C', 'D', 'E'], 'value2': [20, 30, 40, 50]})
        ds1, ds2 = DataStore(df1), DataStore(df2)

        pd_result = pd.merge(df1, df2, on='key', how='left')
        ds_result = ds1.merge(ds2, on='key', how='left')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_outer_merge(self):
        """Test outer merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})
        df2 = pd.DataFrame({'key': ['B', 'C', 'D', 'E'], 'value2': [20, 30, 40, 50]})
        ds1, ds2 = DataStore(df1), DataStore(df2)

        pd_result = pd.merge(df1, df2, on='key', how='outer')
        ds_result = ds1.merge(ds2, on='key', how='outer')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_multiple_keys(self):
        """Test merge on multiple keys."""
        df7 = pd.DataFrame({
            'key1': ['A', 'A', 'B', 'B'],
            'key2': [1, 2, 1, 2],
            'val1': [10, 20, 30, 40]
        })
        df8 = pd.DataFrame({
            'key1': ['A', 'A', 'B'],
            'key2': [1, 2, 2],
            'val2': [100, 200, 400]
        })
        ds7, ds8 = DataStore(df7), DataStore(df8)

        pd_result = pd.merge(df7, df8, on=['key1', 'key2'])
        ds_result = ds7.merge(ds8, on=['key1', 'key2'])
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_cross_merge(self):
        """Test cross merge (cartesian product)."""
        df_a = pd.DataFrame({'a': [1, 2]})
        df_b = pd.DataFrame({'b': [10, 20, 30]})
        ds_a, ds_b = DataStore(df_a), DataStore(df_b)

        pd_result = pd.merge(df_a, df_b, how='cross')
        ds_result = ds_a.merge(ds_b, how='cross')
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestGroupByAdvanced:
    """Test advanced groupby operations."""

    def test_ngroups(self):
        """Test ngroups attribute."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [1, 2, 3, 4, 5]
        })
        ds = DataStore(df)

        pd_ngroups = df.groupby('group').ngroups
        ds_ngroups = ds.groupby('group').ngroups

        assert pd_ngroups == ds_ngroups == 3

    def test_ngroups_multiple_columns(self):
        """Test ngroups with multiple groupby columns."""
        df = pd.DataFrame({
            'group1': ['A', 'A', 'B', 'B'],
            'group2': ['X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4]
        })
        ds = DataStore(df)

        pd_ngroups = df.groupby(['group1', 'group2']).ngroups
        ds_ngroups = ds.groupby(['group1', 'group2']).ngroups

        assert pd_ngroups == ds_ngroups == 4

    def test_groupby_size(self):
        """Test groupby size method."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds = DataStore(df)

        pd_result = df.groupby('group').size()
        ds_result = ds.groupby('group').size()

        # Execute and compare
        ds_df = get_series(ds_result)

        assert isinstance(ds_df, pd.Series)
        assert ds_df.equals(pd_result)

    def test_groupby_agg_sum_then_sort(self):
        """Test groupby aggregate then sort."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds = DataStore(df)

        pd_result = df.groupby('group')['value'].sum().sort_values(ascending=False)
        ds_result = ds.groupby('group')['value'].sum().sort_values(ascending=False)

        # Execute and compare
        ds_df = get_series(ds_result)

        assert ds_df.equals(pd_result)


class TestWindowFunctions:
    """Test window functions (rolling, expanding, cumulative)."""

    def test_rolling_mean(self):
        """Test rolling mean."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds = DataStore(df)

        pd_result = df['value'].rolling(window=3).mean()
        ds_result = ds['value'].rolling(window=3).mean()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum(self):
        """Test cumulative sum."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].cumsum()
        ds_result = ds['value'].cumsum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift(self):
        """Test shift operation."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].shift(1)
        ds_result = ds['value'].shift(1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff(self):
        """Test diff operation."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].diff()
        ds_result = ds['value'].diff()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestChainedOperations:
    """Test complex chained operations."""

    def test_filter_groupby_agg(self):
        """Test filter -> groupby -> aggregate chain."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds = DataStore(df)

        pd_result = df[df['value'] > 15].groupby('group')['value'].sum()
        ds_result = ds[ds['value'] > 15].groupby('group')['value'].sum()
        
        # Execute ColumnExpr to get actual result
        ds_df = get_series(ds_result)
        
        # Compare Series
        assert ds_df.equals(pd_result), f"Results differ: DS={ds_df}, PD={pd_result}"

    def test_multiple_filters_and(self):
        """Test multiple filters with AND."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds = DataStore(df)

        pd_result = df[(df['value'] > 15) & (df['group'] == 'B')]
        ds_result = ds[(ds['value'] > 15) & (ds['group'] == 'B')]
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_sort_head(self):
        """Test filter -> sort -> head chain."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds = DataStore(df)

        pd_result = df[df['value'] > 15].sort_values('value', ascending=False).head(3)
        ds_result = ds[ds['value'] > 15].sort_values('value', ascending=False).head(3)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)
