"""
Exploratory Batch 41: Data Imputation, Deduplication, and Index-Based Aggregation Chains

Focus areas:
1. fillna/bfill/ffill + chain operations (filter, groupby, sort)
2. idxmax/idxmin with various scenarios
3. drop_duplicates + lazy operation chains
4. value_counts with chain operations
5. quantile edge cases and chains
6. nlargest/nsmallest in chains

Test design rationale:
- Mirror Code Pattern: Each test mirrors pandas operations exactly
- Chain Focus: Tests complex operation chains that haven't been covered
- Edge Cases: Empty DataFrames, single rows, all-NA columns
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Section 1: fillna/bfill/ffill + Chain Operations
# =============================================================================


class TestFillnaChains:
    """Tests for fillna/bfill/ffill combined with other operations."""

    def test_fillna_then_filter(self):
        """fillna followed by filter operation."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.fillna(0).query('A > 0')

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df.fillna(0).query('A > 0')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_then_groupby_sum(self):
        """fillna followed by groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        pd_result = pd_df.fillna(0).groupby('group')['value'].sum().reset_index()

        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        ds_result = ds_df.fillna(0).groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_then_sort_head(self):
        """fillna followed by sort and head."""
        pd_df = pd.DataFrame({
            'A': [np.nan, 2.0, np.nan, 4.0, 1.0],
            'B': ['e', 'd', 'c', 'b', 'a']
        })
        pd_result = pd_df.fillna(0).sort_values('A').head(3)

        ds_df = DataStore({
            'A': [np.nan, 2.0, np.nan, 4.0, 1.0],
            'B': ['e', 'd', 'c', 'b', 'a']
        })
        ds_result = ds_df.fillna(0).sort_values('A').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict_multicolumn(self):
        """fillna with dictionary for different columns."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [np.nan, 2.0, np.nan],
            'C': ['x', None, 'z']
        })
        pd_result = pd_df.fillna({'A': 0, 'B': -1, 'C': 'missing'})

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0],
            'B': [np.nan, 2.0, np.nan],
            'C': ['x', None, 'z']
        })
        ds_result = ds_df.fillna({'A': 0, 'B': -1, 'C': 'missing'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_fillna(self):
        """Filter followed by fillna."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['B'] > 15].fillna(99)

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df[ds_df['B'] > 15].fillna(99)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bfill_basic(self):
        """Basic backward fill operation."""
        pd_df = pd.DataFrame({
            'A': [np.nan, np.nan, 3.0, np.nan, 5.0],
        })
        pd_result = pd_df.bfill()

        ds_df = DataStore({
            'A': [np.nan, np.nan, 3.0, np.nan, 5.0],
        })
        ds_result = ds_df.bfill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ffill_basic(self):
        """Basic forward fill operation."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, 4.0, np.nan],
        })
        pd_result = pd_df.ffill()

        ds_df = DataStore({
            'A': [1.0, np.nan, np.nan, 4.0, np.nan],
        })
        ds_result = ds_df.ffill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ffill_then_filter(self):
        """Forward fill followed by filter."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, 4.0, np.nan],
            'B': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.ffill().query('A > 1')

        ds_df = DataStore({
            'A': [1.0, np.nan, np.nan, 4.0, np.nan],
            'B': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df.ffill().query('A > 1')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_empty_dataframe(self):
        """fillna on empty DataFrame."""
        pd_df = pd.DataFrame({'A': pd.Series([], dtype='float64')})
        pd_result = pd_df.fillna(0)

        ds_df = DataStore({'A': pd.Series([], dtype='float64')})
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_all_na_column(self):
        """fillna on column with all NA values."""
        pd_df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [1, 2, 3]
        })
        pd_result = pd_df.fillna(-999)

        ds_df = DataStore({
            'A': [np.nan, np.nan, np.nan],
            'B': [1, 2, 3]
        })
        ds_result = ds_df.fillna(-999)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 2: idxmax/idxmin Operations
# =============================================================================


class TestIdxMaxMin:
    """Tests for idxmax/idxmin operations."""

    def test_idxmax_basic(self):
        """Basic idxmax operation."""
        pd_df = pd.DataFrame({
            'A': [1.0, 5.0, 3.0, 2.0],
            'B': [10, 50, 30, 20]
        })
        pd_result = pd_df.idxmax()

        ds_df = DataStore({
            'A': [1.0, 5.0, 3.0, 2.0],
            'B': [10, 50, 30, 20]
        })
        ds_result = ds_df.idxmax()

        # idxmax returns Series
        assert list(ds_result.values) == list(pd_result.values)

    def test_idxmin_basic(self):
        """Basic idxmin operation."""
        pd_df = pd.DataFrame({
            'A': [1.0, 5.0, 3.0, 2.0],
            'B': [10, 50, 30, 20]
        })
        pd_result = pd_df.idxmin()

        ds_df = DataStore({
            'A': [1.0, 5.0, 3.0, 2.0],
            'B': [10, 50, 30, 20]
        })
        ds_result = ds_df.idxmin()

        assert list(ds_result.values) == list(pd_result.values)

    def test_idxmax_with_na(self):
        """idxmax with NA values - should skip NA by default."""
        pd_df = pd.DataFrame({
            'A': [np.nan, 5.0, 3.0, np.nan],
            'B': [10, np.nan, 30, 20]
        })
        pd_result = pd_df.idxmax()

        ds_df = DataStore({
            'A': [np.nan, 5.0, 3.0, np.nan],
            'B': [10, np.nan, 30, 20]
        })
        ds_result = ds_df.idxmax()

        assert list(ds_result.values) == list(pd_result.values)

    def test_idxmin_with_na(self):
        """idxmin with NA values."""
        pd_df = pd.DataFrame({
            'A': [np.nan, 5.0, 3.0, np.nan],
            'B': [10, np.nan, 30, 20]
        })
        pd_result = pd_df.idxmin()

        ds_df = DataStore({
            'A': [np.nan, 5.0, 3.0, np.nan],
            'B': [10, np.nan, 30, 20]
        })
        ds_result = ds_df.idxmin()

        assert list(ds_result.values) == list(pd_result.values)

    def test_series_idxmax(self):
        """Series idxmax."""
        pd_s = pd.Series([1.0, 5.0, 3.0, 2.0], name='A')
        pd_result = pd_s.idxmax()

        ds_df = DataStore({'A': [1.0, 5.0, 3.0, 2.0]})
        ds_result = ds_df['A'].idxmax()

        assert ds_result == pd_result

    def test_series_idxmin(self):
        """Series idxmin."""
        pd_s = pd.Series([1.0, 5.0, 3.0, 2.0], name='A')
        pd_result = pd_s.idxmin()

        ds_df = DataStore({'A': [1.0, 5.0, 3.0, 2.0]})
        ds_result = ds_df['A'].idxmin()

        assert ds_result == pd_result

    def test_idxmax_filter_chain(self):
        """idxmax after filter operation."""
        pd_df = pd.DataFrame({
            'A': [1.0, 5.0, 3.0, 2.0, 10.0],
            'B': [10, 50, 30, 20, 100]
        })
        # Filter then find idxmax
        pd_filtered = pd_df[pd_df['A'] < 6].reset_index(drop=True)
        pd_result = pd_filtered.idxmax()

        ds_df = DataStore({
            'A': [1.0, 5.0, 3.0, 2.0, 10.0],
            'B': [10, 50, 30, 20, 100]
        })
        ds_filtered = ds_df[ds_df['A'] < 6]
        ds_result = ds_filtered.idxmax()

        assert list(ds_result.values) == list(pd_result.values)


# =============================================================================
# Section 3: drop_duplicates + Chain Operations
# =============================================================================


class TestDropDuplicatesChains:
    """Tests for drop_duplicates combined with other operations."""

    def test_drop_duplicates_then_filter(self):
        """drop_duplicates followed by filter."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 10, 20, 30, 30]
        })
        pd_result = pd_df.drop_duplicates().query('A > 1')

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 10, 20, 30, 30]
        })
        ds_result = ds_df.drop_duplicates().query('A > 1')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset_then_groupby(self):
        """drop_duplicates with subset then groupby."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 1, 2, 3, 1]
        })
        pd_result = pd_df.drop_duplicates(subset=['group', 'value']).groupby('group').size().reset_index(name='count')

        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 1, 2, 3, 1]
        })
        ds_result = ds_df.drop_duplicates(subset=['group', 'value']).groupby('group').size().reset_index(name='count')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_drop_duplicates(self):
        """Filter followed by drop_duplicates."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3, 3],
            'B': [10, 10, 20, 20, 30, 30]
        })
        pd_result = pd_df[pd_df['A'] > 1].drop_duplicates()

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3, 3],
            'B': [10, 10, 20, 20, 30, 30]
        })
        ds_result = ds_df[ds_df['A'] > 1].drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """drop_duplicates with keep='last'."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 11, 20, 21, 30]
        })
        pd_result = pd_df.drop_duplicates(subset=['A'], keep='last')

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 11, 20, 21, 30]
        })
        ds_result = ds_df.drop_duplicates(subset=['A'], keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self):
        """drop_duplicates with keep=False (drop all duplicates)."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 3, 3],
            'B': [10, 11, 20, 30, 31]
        })
        pd_result = pd_df.drop_duplicates(subset=['A'], keep=False)

        ds_df = DataStore({
            'A': [1, 1, 2, 3, 3],
            'B': [10, 11, 20, 30, 31]
        })
        ds_result = ds_df.drop_duplicates(subset=['A'], keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_ignore_index(self):
        """drop_duplicates with ignore_index=True."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 10, 20, 20, 30]
        })
        pd_result = pd_df.drop_duplicates(ignore_index=True)

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 10, 20, 20, 30]
        })
        ds_result = ds_df.drop_duplicates(ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicated_then_filter(self):
        """duplicated() to identify duplicates then filter."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 10, 20, 30, 30]
        })
        pd_dup_mask = pd_df.duplicated()
        pd_result = pd_df[~pd_dup_mask]

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3],
            'B': [10, 10, 20, 30, 30]
        })
        ds_dup_mask = ds_df.duplicated()
        # For DataStore, we need to handle the mask appropriately
        ds_result = ds_df[~ds_dup_mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_empty_df(self):
        """drop_duplicates on empty DataFrame."""
        pd_df = pd.DataFrame({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})
        pd_result = pd_df.drop_duplicates()

        ds_df = DataStore({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 4: value_counts + Chain Operations
# =============================================================================


class TestValueCountsChains:
    """Tests for value_counts combined with other operations."""

    def test_value_counts_basic(self):
        """Basic value_counts operation."""
        pd_df = pd.DataFrame({'A': ['x', 'y', 'x', 'z', 'x', 'y']})
        pd_result = pd_df['A'].value_counts().reset_index()
        pd_result.columns = ['A', 'count']

        ds_df = DataStore({'A': ['x', 'y', 'x', 'z', 'x', 'y']})
        ds_result = ds_df['A'].value_counts().reset_index()
        ds_result.columns = ['A', 'count']

        # value_counts order may differ, so sort for comparison
        pd_result = pd_result.sort_values('A').reset_index(drop=True)
        ds_result_df = ds_result.sort_values('A')

        assert_datastore_equals_pandas(ds_result_df, pd_result, check_row_order=False)

    def test_value_counts_normalize(self):
        """value_counts with normalize=True."""
        pd_df = pd.DataFrame({'A': ['x', 'y', 'x', 'z', 'x', 'y']})
        pd_result = pd_df['A'].value_counts(normalize=True).reset_index()
        pd_result.columns = ['A', 'proportion']

        ds_df = DataStore({'A': ['x', 'y', 'x', 'z', 'x', 'y']})
        ds_result = ds_df['A'].value_counts(normalize=True).reset_index()
        ds_result.columns = ['A', 'proportion']

        # Sort for comparison
        pd_result = pd_result.sort_values('A').reset_index(drop=True)
        ds_result_df = ds_result.sort_values('A')

        assert_datastore_equals_pandas(ds_result_df, pd_result, check_row_order=False)

    def test_filter_then_value_counts(self):
        """Filter followed by value_counts."""
        pd_df = pd.DataFrame({
            'A': ['x', 'y', 'x', 'z', 'x', 'y'],
            'B': [1, 2, 3, 4, 5, 6]
        })
        pd_result = pd_df[pd_df['B'] > 2]['A'].value_counts().reset_index()
        pd_result.columns = ['A', 'count']

        ds_df = DataStore({
            'A': ['x', 'y', 'x', 'z', 'x', 'y'],
            'B': [1, 2, 3, 4, 5, 6]
        })
        ds_result = ds_df[ds_df['B'] > 2]['A'].value_counts().reset_index()
        ds_result.columns = ['A', 'count']

        pd_result = pd_result.sort_values('A').reset_index(drop=True)
        ds_result_df = ds_result.sort_values('A')

        assert_datastore_equals_pandas(ds_result_df, pd_result, check_row_order=False)

    def test_value_counts_dropna_false(self):
        """value_counts with dropna=False to include NA."""
        pd_df = pd.DataFrame({'A': ['x', None, 'x', 'y', None]})
        pd_result = pd_df['A'].value_counts(dropna=False).reset_index()
        pd_result.columns = ['A', 'count']

        ds_df = DataStore({'A': ['x', None, 'x', 'y', None]})
        ds_result = ds_df['A'].value_counts(dropna=False).reset_index()
        ds_result.columns = ['A', 'count']

        # Sort for comparison (handle None values)
        assert len(ds_result) == len(pd_result)

    def test_value_counts_ascending(self):
        """value_counts with ascending=True."""
        pd_df = pd.DataFrame({'A': ['x', 'y', 'x', 'z', 'x', 'y']})
        pd_result = pd_df['A'].value_counts(ascending=True).reset_index()
        pd_result.columns = ['A', 'count']

        ds_df = DataStore({'A': ['x', 'y', 'x', 'z', 'x', 'y']})
        ds_result = ds_df['A'].value_counts(ascending=True).reset_index()
        ds_result.columns = ['A', 'count']

        # Check values match (order by count ascending)
        assert list(ds_result['count'].values) == list(pd_result['count'].values)

    def test_dataframe_value_counts_subset(self):
        """DataFrame value_counts with subset parameter."""
        pd_df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y'],
            'B': [1, 1, 2, 3]
        })
        pd_result = pd_df.value_counts(subset=['A']).reset_index()
        pd_result.columns = ['A', 'count']

        ds_df = DataStore({
            'A': ['x', 'x', 'y', 'y'],
            'B': [1, 1, 2, 3]
        })
        ds_result = ds_df.value_counts(subset=['A']).reset_index()
        ds_result.columns = ['A', 'count']

        pd_result = pd_result.sort_values('A').reset_index(drop=True)
        ds_result_df = ds_result.sort_values('A')

        assert_datastore_equals_pandas(ds_result_df, pd_result, check_row_order=False)


# =============================================================================
# Section 5: quantile Edge Cases and Chains
# =============================================================================


class TestQuantileChains:
    """Tests for quantile operations and chains."""

    def test_quantile_basic(self):
        """Basic quantile operation."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        pd_result = pd_df.quantile(0.5)

        ds_df = DataStore({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_result = ds_df.quantile(0.5)

        # quantile returns Series
        np.testing.assert_almost_equal(ds_result['A'], pd_result['A'], decimal=5)

    def test_quantile_multiple(self):
        """quantile with multiple quantile values."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        pd_result = pd_df.quantile([0.25, 0.5, 0.75])

        ds_df = DataStore({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_result = ds_df.quantile([0.25, 0.5, 0.75])

        # Compare values
        np.testing.assert_almost_equal(
            list(ds_result['A'].values),
            list(pd_result['A'].values),
            decimal=5
        )

    def test_series_quantile(self):
        """Series quantile operation."""
        pd_s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pd_result = pd_s.quantile(0.5)

        ds_df = DataStore({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_result = ds_df['A'].quantile(0.5)

        np.testing.assert_almost_equal(ds_result, pd_result, decimal=5)

    def test_filter_then_quantile(self):
        """Filter followed by quantile."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'B': ['x', 'x', 'x', 'x', 'x', 'y', 'y', 'y', 'y', 'y']
        })
        pd_result = pd_df[pd_df['B'] == 'x']['A'].quantile(0.5)

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'B': ['x', 'x', 'x', 'x', 'x', 'y', 'y', 'y', 'y', 'y']
        })
        ds_result = ds_df[ds_df['B'] == 'x']['A'].quantile(0.5)

        np.testing.assert_almost_equal(ds_result, pd_result, decimal=5)

    def test_quantile_with_na(self):
        """quantile with NA values."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0, np.nan, 5.0]})
        pd_result = pd_df.quantile(0.5)

        ds_df = DataStore({'A': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_result = ds_df.quantile(0.5)

        np.testing.assert_almost_equal(ds_result['A'], pd_result['A'], decimal=5)


# =============================================================================
# Section 6: nlargest/nsmallest Chains
# =============================================================================


class TestNlargestNsmallestChains:
    """Tests for nlargest/nsmallest combined with other operations."""

    def test_nlargest_basic(self):
        """Basic nlargest operation."""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3, 2, 4],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.nlargest(3, 'A')

        ds_df = DataStore({
            'A': [1, 5, 3, 2, 4],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_result = ds_df.nlargest(3, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_basic(self):
        """Basic nsmallest operation."""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3, 2, 4],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.nsmallest(3, 'A')

        ds_df = DataStore({
            'A': [1, 5, 3, 2, 4],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_result = ds_df.nsmallest(3, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_then_filter(self):
        """nlargest followed by filter."""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3, 2, 4],
            'B': [10, 50, 30, 20, 40]
        })
        pd_result = pd_df.nlargest(4, 'A').query('B > 25')

        ds_df = DataStore({
            'A': [1, 5, 3, 2, 4],
            'B': [10, 50, 30, 20, 40]
        })
        ds_result = ds_df.nlargest(4, 'A').query('B > 25')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_nlargest(self):
        """Filter followed by nlargest."""
        pd_df = pd.DataFrame({
            'A': [1, 5, 3, 2, 4, 6],
            'B': [10, 50, 30, 20, 40, 60]
        })
        pd_result = pd_df[pd_df['B'] > 20].nlargest(2, 'A')

        ds_df = DataStore({
            'A': [1, 5, 3, 2, 4, 6],
            'B': [10, 50, 30, 20, 40, 60]
        })
        ds_result = ds_df[ds_df['B'] > 20].nlargest(2, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_multiple_columns(self):
        """nlargest with multiple columns for tie-breaking."""
        pd_df = pd.DataFrame({
            'A': [1, 5, 5, 2, 4],
            'B': [10, 50, 30, 20, 40]
        })
        pd_result = pd_df.nlargest(3, ['A', 'B'])

        ds_df = DataStore({
            'A': [1, 5, 5, 2, 4],
            'B': [10, 50, 30, 20, 40]
        })
        ds_result = ds_df.nlargest(3, ['A', 'B'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_with_na(self):
        """nsmallest with NA values."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 2.0, np.nan],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.nsmallest(2, 'A')

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, 2.0, np.nan],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_result = ds_df.nsmallest(2, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_nlargest(self):
        """Series nlargest operation."""
        pd_s = pd.Series([1, 5, 3, 2, 4])
        pd_result = pd_s.nlargest(3)

        ds_df = DataStore({'A': [1, 5, 3, 2, 4]})
        ds_result = ds_df['A'].nlargest(3)

        # Series nlargest returns Series with index
        assert list(ds_result.values) == list(pd_result.values)

    def test_series_nsmallest(self):
        """Series nsmallest operation."""
        pd_s = pd.Series([1, 5, 3, 2, 4])
        pd_result = pd_s.nsmallest(3)

        ds_df = DataStore({'A': [1, 5, 3, 2, 4]})
        ds_result = ds_df['A'].nsmallest(3)

        assert list(ds_result.values) == list(pd_result.values)


# =============================================================================
# Section 7: nunique Operations
# =============================================================================


class TestNuniqueChains:
    """Tests for nunique operations."""

    def test_nunique_basic(self):
        """Basic nunique operation."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': ['x', 'y', 'x', 'z', 'x']
        })
        pd_result = pd_df.nunique()

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3],
            'B': ['x', 'y', 'x', 'z', 'x']
        })
        ds_result = ds_df.nunique()

        assert list(ds_result.values) == list(pd_result.values)

    def test_nunique_dropna_false(self):
        """nunique with dropna=False."""
        pd_df = pd.DataFrame({
            'A': [1, 1, np.nan, 2, np.nan],
        })
        pd_result = pd_df.nunique(dropna=False)

        ds_df = DataStore({
            'A': [1, 1, np.nan, 2, np.nan],
        })
        ds_result = ds_df.nunique(dropna=False)

        assert list(ds_result.values) == list(pd_result.values)

    def test_groupby_nunique(self):
        """GroupBy nunique operation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 1, 2, 3, 3]
        })
        pd_result = pd_df.groupby('group')['value'].nunique().reset_index()

        ds_df = DataStore({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 1, 2, 3, 3]
        })
        ds_result = ds_df.groupby('group')['value'].nunique().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_nunique(self):
        """Filter followed by nunique."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3, 3],
            'B': [10, 20, 30, 40, 50, 60]
        })
        pd_result = pd_df[pd_df['B'] > 30].nunique()

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3, 3],
            'B': [10, 20, 30, 40, 50, 60]
        })
        ds_result = ds_df[ds_df['B'] > 30].nunique()

        assert list(ds_result.values) == list(pd_result.values)


# =============================================================================
# Section 8: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Edge cases for data imputation and deduplication."""

    def test_fillna_single_row(self):
        """fillna on single row DataFrame."""
        pd_df = pd.DataFrame({'A': [np.nan], 'B': [1]})
        pd_result = pd_df.fillna(0)

        ds_df = DataStore({'A': [np.nan], 'B': [1]})
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_single_row(self):
        """drop_duplicates on single row DataFrame."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        pd_result = pd_df.drop_duplicates()

        ds_df = DataStore({'A': [1], 'B': [2]})
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_all_same(self):
        """drop_duplicates when all rows are identical."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 1],
            'B': [2, 2, 2]
        })
        pd_result = pd_df.drop_duplicates()

        ds_df = DataStore({
            'A': [1, 1, 1],
            'B': [2, 2, 2]
        })
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_n_greater_than_rows(self):
        """nlargest with n > number of rows."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_result = pd_df.nlargest(10, 'A')

        ds_df = DataStore({'A': [1, 2, 3]})
        ds_result = ds_df.nlargest(10, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_value_counts_single_value(self):
        """value_counts with only one unique value."""
        pd_df = pd.DataFrame({'A': ['x', 'x', 'x']})
        pd_result = pd_df['A'].value_counts().reset_index()
        pd_result.columns = ['A', 'count']

        ds_df = DataStore({'A': ['x', 'x', 'x']})
        ds_result = ds_df['A'].value_counts().reset_index()
        ds_result.columns = ['A', 'count']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_quantile_single_value(self):
        """quantile on single value."""
        pd_df = pd.DataFrame({'A': [5.0]})
        pd_result = pd_df.quantile(0.5)

        ds_df = DataStore({'A': [5.0]})
        ds_result = ds_df.quantile(0.5)

        np.testing.assert_almost_equal(ds_result['A'], pd_result['A'], decimal=5)

    def test_idxmax_all_same_values(self):
        """idxmax when all values are the same."""
        pd_df = pd.DataFrame({'A': [5, 5, 5]})
        pd_result = pd_df.idxmax()

        ds_df = DataStore({'A': [5, 5, 5]})
        ds_result = ds_df.idxmax()

        # First occurrence should be returned
        assert ds_result['A'] == pd_result['A']

    def test_duplicated_no_duplicates(self):
        """duplicated when there are no duplicates."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        pd_result = pd_df.duplicated()

        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_result = ds_df.duplicated()

        # All should be False
        assert list(ds_result.values) == list(pd_result.values)


# =============================================================================
# Section 9: Complex Chains
# =============================================================================


class TestComplexChains:
    """Complex multi-operation chains."""

    def test_fillna_filter_groupby_nlargest(self):
        """Complex chain: fillna -> filter -> groupby -> agg -> nlargest."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A', 'B'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0]
        })
        pd_result = (pd_df
                     .fillna(0)
                     .query('value > 0')
                     .groupby('group')['value']
                     .sum()
                     .reset_index()
                     .nlargest(1, 'value'))

        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A', 'B'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0]
        })
        ds_result = (ds_df
                     .fillna(0)
                     .query('value > 0')
                     .groupby('group')['value']
                     .sum()
                     .reset_index()
                     .nlargest(1, 'value'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_sort_head(self):
        """Complex chain: drop_duplicates -> sort -> head."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3, 3, 4, 4],
            'B': [10, 11, 20, 21, 30, 31, 40, 41]
        })
        pd_result = pd_df.drop_duplicates(subset=['A']).sort_values('B', ascending=False).head(2)

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3, 3, 4, 4],
            'B': [10, 11, 20, 21, 30, 31, 40, 41]
        })
        ds_result = ds_df.drop_duplicates(subset=['A']).sort_values('B', ascending=False).head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_value_counts_nlargest(self):
        """Complex chain: filter -> value_counts -> head."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        pd_filtered = pd_df[pd_df['value'] > 2]
        pd_result = pd_filtered['category'].value_counts().head(2).reset_index()
        pd_result.columns = ['category', 'count']

        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        ds_filtered = ds_df[ds_df['value'] > 2]
        ds_result = ds_filtered['category'].value_counts().head(2).reset_index()
        ds_result.columns = ['category', 'count']

        # Results should have same counts
        assert len(ds_result) == len(pd_result)
        assert list(ds_result['count'].values) == list(pd_result['count'].values)

    def test_groupby_nunique_filter(self):
        """Complex chain: groupby -> nunique -> filter result."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'C'],
            'value': [1, 1, 2, 3, 3, 4]
        })
        pd_result = pd_df.groupby('group')['value'].nunique().reset_index()
        pd_result = pd_result[pd_result['value'] > 1]

        ds_df = DataStore({
            'group': ['A', 'A', 'A', 'B', 'B', 'C'],
            'value': [1, 1, 2, 3, 3, 4]
        })
        ds_result = ds_df.groupby('group')['value'].nunique().reset_index()
        ds_result = ds_result[ds_result['value'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)
