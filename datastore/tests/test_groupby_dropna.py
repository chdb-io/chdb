"""
Tests for groupby dropna parameter alignment with pandas.

This tests the fix for: groupby dropna parameter being ignored.
The default behavior should match pandas where dropna=True excludes NA groups.

See: tracking/discoveries/2026-01-06_groupby_dropna_alignment_research.md
"""

import pytest
import numpy as np
import pandas as pd

from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal, get_series


class TestGroupByDropna:
    """Test groupby dropna parameter for pandas alignment."""

    @pytest.fixture
    def df_with_nulls(self):
        """Create test data with NULL values in group column."""
        return pd.DataFrame(
            {'category': ['A', 'B', None, 'A', np.nan, 'B', 'C', None], 'value': [10, 20, 30, 40, 50, 60, 70, 80]}
        )

    def test_groupby_sum_dropna_true_default(self, df_with_nulls):
        """Test default dropna=True excludes NULL groups (pandas default)."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        # pandas default: dropna=True
        pd_result = pd_df.groupby('category')['value'].sum()
        ds_result = ds_df.groupby('category')['value'].sum()

        # Should only have A, B, C groups (no NULL group)
        assert len(pd_result) == 3
        # DataStore should match pandas
        assert_series_equal(ds_result, pd_result)

    def test_groupby_sum_dropna_true_explicit(self, df_with_nulls):
        """Test explicit dropna=True excludes NULL groups."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=True)['value'].sum()
        ds_result = ds_df.groupby('category', dropna=True)['value'].sum()

        assert len(pd_result) == 3
        assert_series_equal(ds_result, pd_result)

    def test_groupby_sum_dropna_false(self, df_with_nulls):
        """Test dropna=False includes NULL groups."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=False)['value'].sum()
        ds_result = ds_df.groupby('category', dropna=False)['value'].sum()

        # Should have A, B, C, NaN groups (4 groups)
        assert len(pd_result) == 4
        # DataStore should match pandas (including NaN group)
        assert_series_equal(ds_result, pd_result)

    def test_groupby_mean_dropna_true(self, df_with_nulls):
        """Test dropna=True with mean aggregation."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=True)['value'].mean()
        ds_result = ds_df.groupby('category', dropna=True)['value'].mean()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_mean_dropna_false(self, df_with_nulls):
        """Test dropna=False with mean aggregation."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=False)['value'].mean()
        ds_result = ds_df.groupby('category', dropna=False)['value'].mean()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_count_dropna_true(self, df_with_nulls):
        """Test dropna=True with count aggregation."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=True)['value'].count()
        ds_result = ds_df.groupby('category', dropna=True)['value'].count()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_count_dropna_false(self, df_with_nulls):
        """Test dropna=False with count aggregation."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=False)['value'].count()
        ds_result = ds_df.groupby('category', dropna=False)['value'].count()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_agg_dict_dropna_true(self, df_with_nulls):
        """Test dropna=True with agg dict."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=True).agg({'value': 'sum'})
        ds_result = ds_df.groupby('category', dropna=True).agg({'value': 'sum'})

        # DataStore returns DataFrame with index
        ds_executed = ds_result._get_df()

        # Both should have 3 groups (no NULL)
        assert len(pd_result) == 3
        assert len(ds_result) == 3

    def test_groupby_agg_dict_dropna_false(self, df_with_nulls):
        """Test dropna=False with agg dict."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_result = pd_df.groupby('category', dropna=False).agg({'value': 'sum'})
        ds_result = ds_df.groupby('category', dropna=False).agg({'value': 'sum'})

        ds_executed = ds_result._get_df()

        # Should have 4 groups including NaN
        assert len(pd_result) == 4
        assert len(ds_result) == 4

    def test_groupby_ngroups_dropna_true(self, df_with_nulls):
        """Test ngroups with dropna=True."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_ngroups = pd_df.groupby('category', dropna=True).ngroups
        ds_ngroups = ds_df.groupby('category', dropna=True).ngroups

        assert pd_ngroups == 3
        assert ds_ngroups == pd_ngroups

    def test_groupby_ngroups_dropna_false(self, df_with_nulls):
        """Test ngroups with dropna=False."""
        pd_df = df_with_nulls
        ds_df = DataStore(df_with_nulls)

        pd_ngroups = pd_df.groupby('category', dropna=False).ngroups
        ds_ngroups = ds_df.groupby('category', dropna=False).ngroups

        assert pd_ngroups == 4
        assert ds_ngroups == pd_ngroups


class TestGroupByDropnaMultiColumn:
    """Test dropna with multiple groupby columns."""

    @pytest.fixture
    def df_multi_null(self):
        """Create test data with NULL values in multiple group columns."""
        return pd.DataFrame(
            {
                'cat1': ['A', 'A', None, 'B', 'B', None],
                'cat2': ['X', None, 'Y', 'X', None, 'Z'],
                'value': [10, 20, 30, 40, 50, 60],
            }
        )

    def test_multi_groupby_dropna_true(self, df_multi_null):
        """Test dropna=True with multiple groupby columns."""
        pd_df = df_multi_null
        ds_df = DataStore(df_multi_null)

        pd_result = pd_df.groupby(['cat1', 'cat2'], dropna=True)['value'].sum()
        ds_result = ds_df.groupby(['cat1', 'cat2'], dropna=True)['value'].sum()

        # dropna=True should exclude rows where ANY groupby column is NULL
        # Only ('A', 'X') and ('B', 'X') should remain
        assert len(pd_result) == 2
        assert_series_equal(ds_result, pd_result)

    def test_multi_groupby_dropna_false(self, df_multi_null):
        """Test dropna=False with multiple groupby columns."""
        pd_df = df_multi_null
        ds_df = DataStore(df_multi_null)

        pd_result = pd_df.groupby(['cat1', 'cat2'], dropna=False)['value'].sum()
        ds_result = ds_df.groupby(['cat1', 'cat2'], dropna=False)['value'].sum()

        # dropna=False should include all groups - 6 unique combinations
        # Note: pandas treats None and NaN separately in some cases
        assert len(pd_result) >= 5  # At least 5 groups
        assert len(ds_result) == len(pd_result)


class TestGroupByDropnaEdgeCases:
    """Test edge cases for groupby dropna."""

    def test_groupby_all_null_dropna_true(self):
        """Test when all group values are NULL with dropna=True."""
        df = pd.DataFrame({'category': [None, np.nan, None], 'value': [10, 20, 30]})
        pd_df = df
        ds_df = DataStore(df)

        pd_result = pd_df.groupby('category', dropna=True)['value'].sum()
        ds_result = ds_df.groupby('category', dropna=True)['value'].sum()

        # Should be empty since all groups are NULL
        assert len(pd_result) == 0
        assert len(ds_result) == 0

    def test_groupby_no_null_dropna_true(self):
        """Test when no group values are NULL with dropna=True."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40]})
        pd_df = df
        ds_df = DataStore(df)

        pd_result = pd_df.groupby('category', dropna=True)['value'].sum()
        ds_result = ds_df.groupby('category', dropna=True)['value'].sum()

        assert_series_equal(ds_result, pd_result)

    def test_groupby_no_null_dropna_false(self):
        """Test when no group values are NULL with dropna=False."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40]})
        pd_df = df
        ds_df = DataStore(df)

        # Results should be identical regardless of dropna when no NULLs
        pd_true = pd_df.groupby('category', dropna=True)['value'].sum()
        pd_false = pd_df.groupby('category', dropna=False)['value'].sum()
        ds_true = ds_df.groupby('category', dropna=True)['value'].sum()
        ds_false = ds_df.groupby('category', dropna=False)['value'].sum()

        assert_series_equal(pd_true, pd_false)
        assert_series_equal(get_series(ds_true), get_series(ds_false))

    def test_groupby_single_null_group(self):
        """Test with a single NULL group."""
        df = pd.DataFrame({'category': ['A', None, 'A'], 'value': [10, 20, 30]})
        pd_df = df
        ds_df = DataStore(df)

        pd_result_true = pd_df.groupby('category', dropna=True)['value'].sum()
        ds_result_true = ds_df.groupby('category', dropna=True)['value'].sum()

        pd_result_false = pd_df.groupby('category', dropna=False)['value'].sum()
        ds_result_false = ds_df.groupby('category', dropna=False)['value'].sum()

        # dropna=True: only A group
        assert len(pd_result_true) == 1
        assert_series_equal(ds_result_true, pd_result_true)

        # dropna=False: A and NaN groups
        assert len(pd_result_false) == 2
        assert_series_equal(ds_result_false, pd_result_false)
