"""
Tests for groupby().apply() MultiIndex preservation.

This module tests that DataStore correctly preserves MultiIndex structure
when groupby().apply() returns a DataFrame.

Key behaviors tested:
1. MultiIndex preservation with (groupby_key, original_index) structure
2. MultiIndex names are correctly set
3. Various apply functions returning DataFrames
4. reset_index() behavior on result

Reference: pandas groupby().apply() returns a DataFrame with MultiIndex
containing (groupby_key, original_row_index) when the function returns a DataFrame.
"""

import unittest

import numpy as np
import pandas as pd

import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas


class TestGroupByApplyMultiIndexPreservation(unittest.TestCase):
    """Test that groupby().apply() preserves MultiIndex structure."""

    def setUp(self):
        self.data = {
            'category': ['A', 'A', 'B', 'B', 'C'],
            'value': [1, 2, 3, 4, 5]
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = ds.DataStore.from_df(self.pd_df)

    def test_apply_head_multiindex(self):
        """Test groupby().apply(lambda x: x.head(2)) preserves MultiIndex."""
        # pandas
        pd_result = self.pd_df.groupby('category', dropna=True).apply(
            lambda x: x.head(2), include_groups=False
        )

        # DataStore (mirror)
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.head(2))

        # Verify MultiIndex is preserved
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_apply_nlargest_multiindex(self):
        """Test groupby().apply(lambda x: x.nlargest()) preserves MultiIndex."""
        data = {
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 5, 3, 2, 6, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.nlargest(2, 'value'), include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(lambda x: x.nlargest(2, 'value'))

        # Verify MultiIndex is preserved
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_apply_nsmallest_multiindex(self):
        """Test groupby().apply(lambda x: x.nsmallest()) preserves MultiIndex."""
        data = {
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 5, 3, 2, 6, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.nsmallest(2, 'value'), include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(lambda x: x.nsmallest(2, 'value'))

        # Verify MultiIndex is preserved
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_apply_filter_multiindex(self):
        """Test groupby().apply() with filter function preserves MultiIndex."""
        data = {
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 5, 3, 2, 6, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas - filter rows where value > 2
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x[x['value'] > 2], include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(lambda x: x[x['value'] > 2])

        # Verify MultiIndex is preserved
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_apply_sort_multiindex(self):
        """Test groupby().apply() with sort function preserves MultiIndex."""
        data = {
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 5, 6, 2, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas - sort within each group
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.sort_values('value'), include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(lambda x: x.sort_values('value'))

        # Verify MultiIndex is preserved
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestGroupByApplyMultiIndexNames(unittest.TestCase):
    """Test that groupby().apply() preserves MultiIndex names correctly."""

    def test_multiindex_names_single_groupby(self):
        """Test MultiIndex names with single groupby column."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.head(2), include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(lambda x: x.head(2))

        # Verify index names match
        assert pd_result.index.names == ds_result.index.names
        assert pd_result.index.names[0] == 'category'

    def test_multiindex_names_multiple_groupby(self):
        """Test MultiIndex names with multiple groupby columns."""
        data = {
            'cat1': ['A', 'A', 'B', 'B', 'A', 'B'],
            'cat2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'value': [1, 2, 3, 4, 5, 6]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby(['cat1', 'cat2'], dropna=True).apply(
            lambda x: x.head(2), include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby(['cat1', 'cat2']).apply(lambda x: x.head(2))

        # Verify index names match
        assert pd_result.index.names == ds_result.index.names
        assert pd_result.index.names[0] == 'cat1'
        assert pd_result.index.names[1] == 'cat2'


class TestGroupByApplyResetIndex(unittest.TestCase):
    """Test reset_index behavior on groupby().apply() results."""

    def test_reset_index_preserves_groupby_keys(self):
        """Test that reset_index() converts MultiIndex to columns."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.head(2), include_groups=False
        ).reset_index()

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(lambda x: x.head(2)).reset_index()

        # Verify reset_index converts MultiIndex to columns
        assert 'category' in pd_result.columns
        assert 'category' in ds_result.columns
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_reset_index_drop_false(self):
        """Test reset_index(drop=False) preserves groupby keys as columns."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.head(2), include_groups=False
        ).reset_index(drop=False)

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(
            lambda x: x.head(2)
        ).reset_index(drop=False)

        # Verify reset_index(drop=False) preserves groupby keys
        assert 'category' in pd_result.columns
        assert 'category' in ds_result.columns
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_reset_index_drop_true(self):
        """Test reset_index(drop=True) removes MultiIndex."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x.head(2), include_groups=False
        ).reset_index(drop=True)

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(
            lambda x: x.head(2)
        ).reset_index(drop=True)

        # Verify reset_index(drop=True) removes groupby keys
        assert 'category' not in pd_result.columns
        assert 'category' not in ds_result.columns
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestGroupByApplyEdgeCases(unittest.TestCase):
    """Test edge cases for groupby().apply() MultiIndex preservation."""

    def test_empty_group(self):
        """Test apply with some groups having no matching rows."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 10, 20]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # Filter that returns empty for group 'A'
        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x[x['value'] > 5], include_groups=False
        )

        ds_result = ds_df.groupby('category').apply(lambda x: x[x['value'] > 5])

        # Verify behavior matches pandas
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_single_row_groups(self):
        """Test apply where each group has only one row."""
        data = {
            'category': ['A', 'B', 'C'],
            'value': [1, 2, 3]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        pd_result = pd_df.groupby('category', dropna=True).apply(
            lambda x: x, include_groups=False
        )

        ds_result = ds_df.groupby('category').apply(lambda x: x)

        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_apply_with_column_selection(self):
        """Test apply on specific columns preserves MultiIndex."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value1': [1, 2, 3, 4],
            'value2': [10, 20, 30, 40]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas - apply to specific column group
        pd_result = pd_df.groupby('category', dropna=True)['value1'].apply(
            lambda x: x.head(2)
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category')['value1'].apply(lambda x: x.head(2))

        # Series result with MultiIndex
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)

    def test_apply_returning_modified_dataframe(self):
        """Test apply that modifies the DataFrame within groups."""
        data = {
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd_df)

        # pandas - double values within each group
        def double_values(x):
            result = x.copy()
            result['value'] = result['value'] * 2
            return result

        pd_result = pd_df.groupby('category', dropna=True).apply(
            double_values, include_groups=False
        )

        # DataStore (mirror)
        ds_result = ds_df.groupby('category').apply(double_values)

        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


if __name__ == '__main__':
    unittest.main()
