"""
Tests for groupby().nth() functionality.

Uses Mirror Pattern: DataStore behavior must match pandas exactly.
"""
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal, get_series


class TestGroupByNth:
    """Test suite for groupby().nth() method."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with groups."""
        return pd.DataFrame({
            'g1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [10, 20, 30, 40, 50, 60]
        })

    @pytest.fixture
    def sample_df_with_na(self):
        """Sample DataFrame with NA values."""
        return pd.DataFrame({
            'g1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [10.0, np.nan, 30.0, 40.0, 50.0, np.nan]
        })

    # ========== DataFrame nth tests ==========

    def test_nth_first_row(self, sample_df):
        """Test nth(0) - get first row of each group."""
        # pandas
        pd_result = sample_df.groupby('g1').nth(0)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth(0)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_nth_second_row(self, sample_df):
        """Test nth(1) - get second row of each group."""
        # pandas
        pd_result = sample_df.groupby('g1').nth(1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth(1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_nth_last_row_negative_index(self, sample_df):
        """Test nth(-1) - get last row of each group using negative index."""
        # pandas
        pd_result = sample_df.groupby('g1').nth(-1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth(-1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_nth_second_last_row(self, sample_df):
        """Test nth(-2) - get second last row of each group."""
        # pandas
        pd_result = sample_df.groupby('g1').nth(-2)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth(-2)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_nth_multiple_indices(self, sample_df):
        """Test nth([0, 2]) - get first and third rows of each group."""
        # pandas
        pd_result = sample_df.groupby('g1').nth([0, 2])

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth([0, 2])

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_nth_out_of_bounds(self, sample_df):
        """Test nth with index beyond group size - should return empty for that group."""
        # pandas
        pd_result = sample_df.groupby('g1').nth(10)  # Beyond any group size

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth(10)

        # Both should be empty
        ds_df = ds_result._get_df()
        assert len(ds_df) == 0
        assert len(pd_result) == 0

    # ========== Series nth tests ==========

    def test_series_nth_first(self, sample_df):
        """Test groupby column selection followed by nth(0)."""
        # pandas
        pd_result = sample_df.groupby('g1')['val'].nth(0)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1')['val'].nth(0)

        # ds_result is ColumnExpr, execute to get Series
        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result, check_names=True)

    def test_series_nth_second(self, sample_df):
        """Test groupby column selection followed by nth(1)."""
        # pandas
        pd_result = sample_df.groupby('g1')['val'].nth(1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1')['val'].nth(1)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result, check_names=True)

    def test_series_nth_negative(self, sample_df):
        """Test groupby column selection followed by nth(-1)."""
        # pandas
        pd_result = sample_df.groupby('g1')['val'].nth(-1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1')['val'].nth(-1)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result, check_names=True)

    # ========== Multi-column groupby ==========

    def test_nth_multi_column_groupby(self):
        """Test nth with multiple groupby columns."""
        df = pd.DataFrame({
            'g1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'g2': ['X', 'X', 'Y', 'X', 'Y', 'Y'],
            'val': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_result = df.groupby(['g1', 'g2']).nth(0)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby(['g1', 'g2']).nth(0)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    # ========== Edge cases ==========

    def test_nth_single_row_groups(self):
        """Test nth where each group has only one row."""
        df = pd.DataFrame({
            'g1': ['A', 'B', 'C'],
            'val': [10, 20, 30]
        })

        # pandas - nth(0) should return all rows
        pd_result = df.groupby('g1').nth(0)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('g1').nth(0)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

        # nth(1) should return empty (no second row in any group)
        pd_result_1 = df.groupby('g1').nth(1)
        ds_result_1 = ds.groupby('g1').nth(1)

        assert len(ds_result_1._get_df()) == 0
        assert len(pd_result_1) == 0

    def test_nth_preserves_original_index(self, sample_df):
        """Test that nth preserves original row indices."""
        # pandas
        pd_result = sample_df.groupby('g1').nth(1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('g1').nth(1)

        ds_df = ds_result._get_df()

        # Verify indices are preserved (1 and 4 for second rows of A and B)
        pd.testing.assert_index_equal(ds_df.index, pd_result.index)


class TestGroupByNthWithNA:
    """Test nth behavior with NA values."""

    @pytest.fixture
    def df_with_na(self):
        return pd.DataFrame({
            'g1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [10.0, np.nan, 30.0, 40.0, 50.0, np.nan]
        })

    def test_nth_includes_na_by_default(self, df_with_na):
        """Test that nth includes NA values by default."""
        # pandas - nth(1) should return NaN for group A (second row is NaN)
        pd_result = df_with_na.groupby('g1').nth(1)

        # DataStore (mirror)
        ds = DataStore(df_with_na.copy())
        ds_result = ds.groupby('g1').nth(1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)
