"""
Exploratory Batch 51: GroupBy Edge Cases, nth/head/tail Boundaries, Index Preservation

Tests discovered through source code analysis focusing on:
1. GroupBy parameter combinations (dropna + as_index + sort)
2. Empty DataFrame edge cases (0 rows, 0 groups, empty after filter)
3. nth() method with negative indices and dropna parameter
4. head()/tail() with small groups and n boundaries
5. size() and ngroups edge cases
6. Index preservation and restoration
7. Type coercion in aggregations with NULL values
"""

import numpy as np
from tests.xfail_markers import chdb_no_product_function
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
)


# =============================================================================
# GroupBy dropna + as_index + sort Parameter Combinations
# =============================================================================


class TestGroupByParameterCombinations:
    """Test groupby with various parameter combinations."""

    def test_groupby_dropna_true_as_index_true(self):
        """GroupBy dropna=True with as_index=True (default)."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', None, 'a', None, 'b'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key', dropna=True)['val'].sum()
        ds_result = ds_df.groupby('key', dropna=True)['val'].sum()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_dropna_false_as_index_false(self):
        """GroupBy dropna=False with as_index=False."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', None, 'a', None, 'b'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key', dropna=False, as_index=False)['val'].sum()
        ds_result = ds_df.groupby('key', dropna=False, as_index=False)['val'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_dropna_true_sort_false(self):
        """GroupBy dropna=True with sort=False."""
        pd_df = pd.DataFrame({
            'key': ['b', 'a', None, 'b', 'c', None],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key', dropna=True, sort=False)['val'].sum()
        ds_result = ds_df.groupby('key', dropna=True, sort=False)['val'].sum()

        # sort=False means order is undefined, compare values only
        ds_series = get_series(ds_result)
        assert set(ds_series.index) == set(pd_result.index)
        for key in pd_result.index:
            assert ds_series.loc[key] == pd_result.loc[key]

    def test_groupby_all_params_combined(self):
        """GroupBy with all parameters: dropna=True, as_index=False, sort=True."""
        pd_df = pd.DataFrame({
            'key': ['c', 'a', None, 'b', None, 'a'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key', dropna=True, as_index=False, sort=True)['val'].mean()
        ds_result = ds_df.groupby('key', dropna=True, as_index=False, sort=True)['val'].mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_multicolumn_key_dropna_true(self):
        """GroupBy with multiple keys and dropna=True."""
        pd_df = pd.DataFrame({
            'key1': ['a', 'a', None, 'b', 'b', None],
            'key2': ['x', None, 'y', 'x', 'y', None],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby(['key1', 'key2'], dropna=True)['val'].sum()
        ds_result = ds_df.groupby(['key1', 'key2'], dropna=True)['val'].sum()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_all_null_keys_dropna_true(self):
        """GroupBy where all keys are NULL with dropna=True should return empty."""
        pd_df = pd.DataFrame({
            'key': [None, None, None],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key', dropna=True)['val'].sum()
        ds_result = ds_df.groupby('key', dropna=True)['val'].sum()

        # Should be empty Series
        assert len(pd_result) == 0
        assert len(get_series(ds_result)) == 0


# =============================================================================
# Empty DataFrame with GroupBy
# =============================================================================


class TestEmptyDataFrameGroupBy:
    """Test groupby operations on empty DataFrames."""

    def test_empty_df_groupby_sum(self):
        """GroupBy sum on empty DataFrame."""
        pd_df = pd.DataFrame({'key': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        assert len(get_series(ds_result)) == 0
        assert len(pd_result) == 0

    def test_empty_df_groupby_count(self):
        """GroupBy count on empty DataFrame."""
        pd_df = pd.DataFrame({'key': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].count()
        ds_result = ds_df.groupby('key')['val'].count()

        assert len(get_series(ds_result)) == 0

    def test_empty_df_groupby_size(self):
        """GroupBy size on empty DataFrame."""
        pd_df = pd.DataFrame({'key': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').size()
        ds_result = ds_df.groupby('key').size()

        assert len(get_series(ds_result)) == 0

    def test_empty_df_ngroups(self):
        """ngroups property on empty DataFrame."""
        pd_df = pd.DataFrame({'key': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df)

        pd_groups = pd_df.groupby('key').ngroups
        ds_groups = ds_df.groupby('key').ngroups

        assert ds_groups == pd_groups == 0

    def test_empty_after_filter_groupby(self):
        """GroupBy on DataFrame that becomes empty after filter."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['val'] > 100]
        ds_filtered = ds_df[ds_df['val'] > 100]

        pd_result = pd_filtered.groupby('key')['val'].sum()
        ds_result = ds_filtered.groupby('key')['val'].sum()

        assert len(get_series(ds_result)) == 0


# =============================================================================
# GroupBy size() Edge Cases
# =============================================================================


class TestGroupBySizeEdgeCases:
    """Test groupby.size() edge cases."""

    def test_size_single_group(self):
        """size() with single group."""
        pd_df = pd.DataFrame({'key': ['a', 'a', 'a'], 'val': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').size()
        ds_result = ds_df.groupby('key').size()

        assert_series_equal(ds_result, pd_result)

    def test_size_single_row_groups(self):
        """size() where each group has one row."""
        pd_df = pd.DataFrame({'key': ['a', 'b', 'c'], 'val': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').size()
        ds_result = ds_df.groupby('key').size()

        assert_series_equal(ds_result, pd_result)

    def test_size_with_null_group_dropna_false(self):
        """size() with NULL group and dropna=False."""
        pd_df = pd.DataFrame({
            'key': ['a', None, 'b', None, 'a'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key', dropna=False).size()
        ds_result = ds_df.groupby('key', dropna=False).size()

        assert_series_equal(ds_result, pd_result)

    def test_size_after_filter(self):
        """size() after filtering."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b', 'c'],
            'val': [1, 10, 2, 20, 3]
        })
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['val'] > 5]
        ds_filtered = ds_df[ds_df['val'] > 5]

        pd_result = pd_filtered.groupby('key').size()
        ds_result = ds_filtered.groupby('key').size()

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# GroupBy ngroups Edge Cases
# =============================================================================


class TestGroupByNgroupsEdgeCases:
    """Test ngroups property edge cases."""

    def test_ngroups_single_group(self):
        """ngroups with single group."""
        pd_df = pd.DataFrame({'key': ['a', 'a', 'a'], 'val': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        assert ds_df.groupby('key').ngroups == pd_df.groupby('key').ngroups == 1

    def test_ngroups_all_unique(self):
        """ngroups where each row is a unique group."""
        pd_df = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'val': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        assert ds_df.groupby('key').ngroups == pd_df.groupby('key').ngroups == 4

    def test_ngroups_with_dropna_true(self):
        """ngroups with dropna=True excludes NULL group."""
        pd_df = pd.DataFrame({
            'key': ['a', None, 'b', None],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_groups = pd_df.groupby('key', dropna=True).ngroups
        ds_groups = ds_df.groupby('key', dropna=True).ngroups

        assert ds_groups == pd_groups == 2

    def test_ngroups_with_dropna_false(self):
        """ngroups with dropna=False includes NULL group."""
        pd_df = pd.DataFrame({
            'key': ['a', None, 'b', None],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_groups = pd_df.groupby('key', dropna=False).ngroups
        ds_groups = ds_df.groupby('key', dropna=False).ngroups

        assert ds_groups == pd_groups == 3


# =============================================================================
# GroupBy nth() Edge Cases
# =============================================================================


class TestGroupByNthEdgeCases:
    """Test groupby.nth() edge cases."""

    def test_nth_zero(self):
        """nth(0) returns first row per group."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b', 'b'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').nth(0)
        ds_result = ds_df.groupby('key').nth(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nth_negative_one(self):
        """nth(-1) returns last row per group."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b', 'b'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').nth(-1)
        ds_result = ds_df.groupby('key').nth(-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nth_out_of_bounds(self):
        """nth(n) where n exceeds group size returns empty for that group."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b'],  # 'a' has 2, 'b' has 1
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        # Request 11th element - No group has that many
        pd_result = pd_df.groupby('key').nth(10)
        ds_result = ds_df.groupby('key').nth(10)

        # Should return empty DataFrame
        assert len(get_dataframe(ds_result)) == 0
        assert len(pd_result) == 0

    def test_nth_single_element_groups(self):
        """nth() on groups with single elements."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').nth(0)
        ds_result = ds_df.groupby('key').nth(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nth_list_of_indices(self):
        """nth() with list of indices."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a', 'b', 'b', 'b'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').nth([0, 2])
        ds_result = ds_df.groupby('key').nth([0, 2])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nth_negative_list(self):
        """nth() with list containing negative indices."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a', 'b', 'b', 'b'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').nth([-1, -2])
        ds_result = ds_df.groupby('key').nth([-1, -2])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# GroupBy head()/tail() Edge Cases
# =============================================================================


class TestGroupByHeadTailEdgeCases:
    """Test groupby.head() and groupby.tail() edge cases."""

    def test_head_larger_than_group(self):
        """head(n) where n exceeds group size returns all rows."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b'],  # 'a' has 2, 'b' has 1
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').head(5)
        ds_result = ds_df.groupby('key').head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_larger_than_group(self):
        """tail(n) where n exceeds group size returns all rows."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').tail(5)
        ds_result = ds_df.groupby('key').tail(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """head(0) returns empty DataFrame."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').head(0)
        ds_result = ds_df.groupby('key').head(0)

        assert len(get_dataframe(ds_result)) == 0
        assert len(pd_result) == 0

    def test_tail_zero(self):
        """tail(0) returns empty DataFrame."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').tail(0)
        ds_result = ds_df.groupby('key').tail(0)

        assert len(get_dataframe(ds_result)) == 0
        assert len(pd_result) == 0

    def test_head_single_element_groups(self):
        """head() on single element groups."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').head(2)
        ds_result = ds_df.groupby('key').head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_preserves_original_index(self):
        """head() preserves original DataFrame index."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        }, index=[10, 20, 30, 40])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').head(1)
        ds_result = ds_df.groupby('key').head(1)

        # Should preserve original indices: 10, 30
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# GroupBy cumcount() Edge Cases
# =============================================================================


class TestGroupByCumcountEdgeCases:
    """Test groupby.cumcount() edge cases."""

    def test_cumcount_basic(self):
        """Basic cumcount()."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b', 'a'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').cumcount()
        ds_result = ds_df.groupby('key').cumcount()

        assert_series_equal(ds_result, pd_result)

    def test_cumcount_ascending_false(self):
        """cumcount(ascending=False)."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').cumcount(ascending=False)
        ds_result = ds_df.groupby('key').cumcount(ascending=False)

        assert_series_equal(ds_result, pd_result)

    def test_cumcount_single_element_groups(self):
        """cumcount() on single element groups."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').cumcount()
        ds_result = ds_df.groupby('key').cumcount()

        # Each group has single element, so cumcount should be [0, 0, 0]
        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Index Preservation Edge Cases
# =============================================================================


class TestIndexPreservationEdgeCases:
    """Test index preservation across operations."""

    def test_set_index_filter_reset(self):
        """set_index -> filter -> reset_index preserves data."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c', 'd'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('key').query('val > 2').reset_index()
        ds_result = ds_df.set_index('key')
        ds_result = ds_result[ds_result['val'] > 2].reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_custom_index_after_groupby(self):
        """Custom index after groupby aggregation with as_index=True."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        # Index should be ['a', 'b']
        assert_series_equal(ds_result, pd_result)

    def test_multiindex_set_reset(self):
        """MultiIndex set_index and reset_index."""
        pd_df = pd.DataFrame({
            'key1': ['a', 'a', 'b', 'b'],
            'key2': ['x', 'y', 'x', 'y'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index(['key1', 'key2'])
        pd_result = pd_result.reset_index()
        ds_result = ds_df.set_index(['key1', 'key2'])
        ds_result = ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_index_name_collision_with_column(self):
        """Index with same name as a column."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        # Set 'val' as index, then reset - 'val' should come back as column
        pd_result = pd_df.set_index('val').reset_index()
        ds_result = ds_df.set_index('val').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Type Coercion in Aggregations
# =============================================================================


class TestTypeCoercionAggregations:
    """Test type coercion in aggregations especially with NULL values."""

    def test_sum_all_nan(self):
        """sum() of all-NaN values returns 0 (skipna=True default)."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a'],
            'val': [np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        assert_series_equal(ds_result, pd_result)

    def test_mean_all_nan(self):
        """mean() of all-NaN values returns NaN."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a'],
            'val': [np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].mean()
        ds_result = ds_df.groupby('key')['val'].mean()

        # Both should be NaN
        assert np.isnan(get_series(ds_result).iloc[0])
        assert np.isnan(pd_result.iloc[0])

    def test_count_with_nan(self):
        """count() excludes NaN values."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a', 'b', 'b'],
            'val': [1, np.nan, 3, np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].count()
        ds_result = ds_df.groupby('key')['val'].count()

        assert_series_equal(ds_result, pd_result)

    def test_min_max_all_nan(self):
        """min/max of all-NaN returns NaN."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a'],
            'val': [np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_min = pd_df.groupby('key')['val'].min()
        ds_min = ds_df.groupby('key')['val'].min()

        pd_max = pd_df.groupby('key')['val'].max()
        ds_max = ds_df.groupby('key')['val'].max()

        assert np.isnan(get_series(ds_min).iloc[0])
        assert np.isnan(get_series(ds_max).iloc[0])

    def test_int_column_with_null_aggregation(self):
        """Integer column with nullable values in aggregation."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': pd.array([1, None, 3, 4], dtype='Int64')
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        # Values should match
        ds_series = get_series(ds_result)
        assert ds_series.loc['a'] == pd_result.loc['a'] == 1
        assert ds_series.loc['b'] == pd_result.loc['b'] == 7


# =============================================================================
# GroupBy agg() with Multiple Functions
# =============================================================================


class TestGroupByAggMultipleFunctions:
    """Test groupby.agg() with multiple functions."""

    def test_agg_dict_multi_column(self):
        """agg() with dict specifying functions per column."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').agg({'val1': 'sum', 'val2': 'mean'})
        ds_result = ds_df.groupby('key').agg({'val1': 'sum', 'val2': 'mean'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_list_of_functions(self):
        """agg() with list of functions."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].agg(['sum', 'mean', 'count'])
        ds_result = ds_df.groupby('key')['val'].agg(['sum', 'mean', 'count'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_empty_groups(self):
        """agg() on empty DataFrame."""
        pd_df = pd.DataFrame({'key': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].agg(['sum', 'mean'])
        ds_result = ds_df.groupby('key')['val'].agg(['sum', 'mean'])

        assert len(get_dataframe(ds_result)) == 0


# =============================================================================
# Complex Chains with GroupBy
# =============================================================================


class TestComplexGroupByChains:
    """Test complex chains involving groupby."""

    def test_filter_groupby_agg_sort_head(self):
        """filter -> groupby -> agg -> sort -> head chain."""
        pd_df = pd.DataFrame({
            'category': ['a', 'b', 'c', 'a', 'b', 'c', 'a'],
            'subcategory': ['x', 'x', 'y', 'y', 'x', 'y', 'x'],
            'value': [10, 20, 30, 40, 50, 60, 70]
        })
        ds_df = DataStore(pd_df)

        pd_result = (pd_df[pd_df['value'] > 15]
                     .groupby('category')['value']
                     .sum()
                     .sort_values(ascending=False)
                     .head(2))
        ds_result = (ds_df[ds_df['value'] > 15]
                     .groupby('category')['value']
                     .sum()
                     .sort_values(ascending=False)
                     .head(2))

        assert_series_equal(ds_result, pd_result)

    def test_assign_groupby_transform(self):
        """assign -> groupby -> transform chain."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(doubled=lambda x: x['val'] * 2)
        pd_result = pd_result.groupby('key')['doubled'].transform('sum')

        ds_result = ds_df.assign(doubled=lambda x: x['val'] * 2)
        ds_result = ds_result.groupby('key')['doubled'].transform('sum')

        assert_series_equal(ds_result, pd_result)

    def test_groupby_filter_on_agg_result(self):
        """GroupBy agg then filter based on aggregated value."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b', 'c', 'c'],
            'val': [1, 2, 10, 20, 100, 200]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        pd_result = pd_result[pd_result > 5]

        ds_result = ds_df.groupby('key')['val'].sum()
        # Filter the Series-like result
        ds_executed = get_series(ds_result)
        ds_filtered = ds_executed[ds_executed > 5]

        # Compare as Series
        assert_series_equal(ds_filtered, pd_result)


# =============================================================================
# Edge Cases in first() and last()
# =============================================================================


class TestGroupByFirstLastEdgeCases:
    """Test groupby.first() and groupby.last() edge cases."""

    def test_first_with_nan(self):
        """first() skips NaN values by default."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a'],
            'val': [np.nan, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].first()
        ds_result = ds_df.groupby('key')['val'].first()

        # first() should return 2 (first non-NaN)
        assert_series_equal(ds_result, pd_result)

    def test_last_with_nan(self):
        """last() skips NaN values by default."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a'],
            'val': [1, 2, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].last()
        ds_result = ds_df.groupby('key')['val'].last()

        # last() should return 2 (last non-NaN)
        assert_series_equal(ds_result, pd_result)

    def test_first_all_nan(self):
        """first() when all values are NaN."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a'],
            'val': [np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].first()
        ds_result = ds_df.groupby('key')['val'].first()

        # Should return NaN
        assert np.isnan(get_series(ds_result).iloc[0])
        assert np.isnan(pd_result.iloc[0])

    def test_first_single_row_groups(self):
        """first() on single row groups."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].first()
        ds_result = ds_df.groupby('key')['val'].first()

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Additional Edge Cases from Source Analysis
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge cases from source code analysis."""

    def test_groupby_on_computed_column(self):
        """GroupBy on a computed column."""
        pd_df = pd.DataFrame({
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        pd_df['bucket'] = pd_df['val'] // 2
        ds_df['bucket'] = ds_df['val'] // 2

        pd_result = pd_df.groupby('bucket')['val'].sum()
        ds_result = ds_df.groupby('bucket')['val'].sum()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_multiple_agg_same_column(self):
        """Multiple aggregations on same column."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key').agg({'val': ['sum', 'mean', 'min', 'max']})
        ds_result = ds_df.groupby('key').agg({'val': ['sum', 'mean', 'min', 'max']})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_std_single_value_groups(self):
        """var() and std() on single value groups."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_var = pd_df.groupby('key')['val'].var()
        ds_var = ds_df.groupby('key')['val'].var()

        pd_std = pd_df.groupby('key')['val'].std()
        ds_std = ds_df.groupby('key')['val'].std()

        # Single value groups should have NaN variance/std
        assert_series_equal(ds_var, pd_var)
        assert_series_equal(ds_std, pd_std)

    @chdb_no_product_function
    def test_prod_with_zeros(self):
        """prod() with zeros in data."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [0, 2, 3, 0]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].prod()
        ds_result = ds_df.groupby('key')['val'].prod()

        assert_series_equal(ds_result, pd_result)

    @chdb_no_product_function
    def test_prod_with_negatives(self):
        """prod() with negative numbers."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [-1, 2, -3, -4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].prod()
        ds_result = ds_df.groupby('key')['val'].prod()

        assert_series_equal(ds_result, pd_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
