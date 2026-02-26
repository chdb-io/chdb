"""
Tests for groupby().head() and groupby().tail() functionality.

Uses Mirror Pattern: DataStore behavior must match pandas exactly.
"""
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from datastore import DataStore
from tests.test_utils import assert_frame_equal


class TestGroupByHead:
    """Test suite for groupby().head() method."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with groups."""
        return pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8]
        })

    @pytest.fixture
    def sample_df_with_na(self):
        """Sample DataFrame with NA values."""
        return pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [10.0, np.nan, 30.0, 40.0, 50.0, np.nan]
        })

    # ========== Basic head tests ==========

    def test_head_default(self, sample_df):
        """Test head() with default n=5."""
        # pandas
        pd_result = sample_df.groupby('category').head()

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').head()

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_head_n2(self, sample_df):
        """Test head(2) - get first 2 rows of each group."""
        # pandas
        pd_result = sample_df.groupby('category').head(2)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').head(2)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_head_n1(self, sample_df):
        """Test head(1) - get first row of each group."""
        # pandas
        pd_result = sample_df.groupby('category').head(1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').head(1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_head_larger_than_group(self, sample_df):
        """Test head(10) where n is larger than some groups."""
        # pandas
        pd_result = sample_df.groupby('category').head(10)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').head(10)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_head_preserves_original_index(self, sample_df):
        """Test that head preserves original row indices."""
        # pandas
        pd_result = sample_df.groupby('category').head(2)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').head(2)

        ds_df = ds_result._get_df()

        # Verify indices are preserved
        pd.testing.assert_index_equal(ds_df.index, pd_result.index)

    # ========== Multi-column groupby ==========

    def test_head_multi_column_groupby(self):
        """Test head with multiple groupby columns."""
        df = pd.DataFrame({
            'g1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'g2': ['X', 'X', 'Y', 'X', 'Y', 'Y'],
            'val': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_result = df.groupby(['g1', 'g2']).head(1)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby(['g1', 'g2']).head(1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    # ========== Edge cases ==========

    def test_head_single_row_groups(self):
        """Test head where each group has only one row."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })

        # pandas
        pd_result = df.groupby('category').head(2)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category').head(2)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_head_empty_dataframe(self):
        """Test head on empty DataFrame."""
        df = pd.DataFrame({
            'category': pd.Series([], dtype=str),
            'value': pd.Series([], dtype=int)
        })

        # pandas
        pd_result = df.groupby('category').head(2)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category').head(2)

        assert len(ds_result._get_df()) == 0
        assert len(pd_result) == 0


class TestGroupByTail:
    """Test suite for groupby().tail() method."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with groups."""
        return pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8]
        })

    # ========== Basic tail tests ==========

    def test_tail_default(self, sample_df):
        """Test tail() with default n=5."""
        # pandas
        pd_result = sample_df.groupby('category').tail()

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').tail()

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_tail_n2(self, sample_df):
        """Test tail(2) - get last 2 rows of each group."""
        # pandas
        pd_result = sample_df.groupby('category').tail(2)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').tail(2)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_tail_n1(self, sample_df):
        """Test tail(1) - get last row of each group."""
        # pandas
        pd_result = sample_df.groupby('category').tail(1)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').tail(1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_tail_larger_than_group(self, sample_df):
        """Test tail(10) where n is larger than some groups."""
        # pandas
        pd_result = sample_df.groupby('category').tail(10)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').tail(10)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_tail_preserves_original_index(self, sample_df):
        """Test that tail preserves original row indices."""
        # pandas
        pd_result = sample_df.groupby('category').tail(2)

        # DataStore (mirror)
        ds = DataStore(sample_df.copy())
        ds_result = ds.groupby('category').tail(2)

        ds_df = ds_result._get_df()

        # Verify indices are preserved
        pd.testing.assert_index_equal(ds_df.index, pd_result.index)

    # ========== Multi-column groupby ==========

    def test_tail_multi_column_groupby(self):
        """Test tail with multiple groupby columns."""
        df = pd.DataFrame({
            'g1': ['A', 'A', 'A', 'B', 'B', 'B'],
            'g2': ['X', 'X', 'Y', 'X', 'Y', 'Y'],
            'val': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_result = df.groupby(['g1', 'g2']).tail(1)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby(['g1', 'g2']).tail(1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    # ========== Edge cases ==========

    def test_tail_single_row_groups(self):
        """Test tail where each group has only one row."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })

        # pandas
        pd_result = df.groupby('category').tail(2)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category').tail(2)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result)

    def test_tail_empty_dataframe(self):
        """Test tail on empty DataFrame."""
        df = pd.DataFrame({
            'category': pd.Series([], dtype=str),
            'value': pd.Series([], dtype=int)
        })

        # pandas
        pd_result = df.groupby('category').tail(2)

        # DataStore (mirror)
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category').tail(2)

        assert len(ds_result._get_df()) == 0
        assert len(pd_result) == 0


class TestGroupByHeadTailComparison:
    """Test head and tail return complementary results."""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with groups of size 3."""
        return pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5, 6]
        })

    def test_head_tail_complementary(self, sample_df):
        """Test that head(1) and tail(2) for groups of 3 are complementary."""
        # For groups of size 3:
        # head(1) returns indices [0, 3] (first of each)
        # tail(2) returns indices [1, 2, 4, 5] (last 2 of each)

        ds = DataStore(sample_df.copy())
        
        ds_head = ds.groupby('category').head(1)._get_df()
        ds_tail = ds.groupby('category').tail(2)._get_df()
        
        pd_head = sample_df.groupby('category').head(1)
        pd_tail = sample_df.groupby('category').tail(2)

        # Verify DataStore matches pandas
        assert_frame_equal(ds_head, pd_head)
        assert_frame_equal(ds_tail, pd_tail)

        # Verify head and tail indices don't overlap
        assert set(ds_head.index) & set(ds_tail.index) == set()

    def test_head_equals_all_for_small_groups(self, sample_df):
        """Test head(n) where n >= group_size returns all rows."""
        ds = DataStore(sample_df.copy())
        
        ds_head = ds.groupby('category').head(5)._get_df()
        pd_head = sample_df.groupby('category').head(5)

        # All rows should be returned since each group has only 3 rows
        assert_frame_equal(ds_head, pd_head)
        assert len(ds_head) == len(sample_df)

    def test_head_vs_nth(self, sample_df):
        """Test head(1) is equivalent to rows from nth(0)."""
        ds = DataStore(sample_df.copy())
        
        ds_head = ds.groupby('category').head(1)._get_df()
        ds_nth = ds.groupby('category').nth(0)._get_df()

        # head(1) and nth(0) should return same values
        # Note: indices may differ slightly due to implementation
        assert list(ds_head['value'].values) == list(ds_nth['value'].values)

    def test_tail_vs_nth_negative(self, sample_df):
        """Test tail(1) is equivalent to rows from nth(-1)."""
        ds = DataStore(sample_df.copy())
        
        ds_tail = ds.groupby('category').tail(1)._get_df()
        ds_nth = ds.groupby('category').nth(-1)._get_df()

        # tail(1) and nth(-1) should return same values
        assert list(ds_tail['value'].values) == list(ds_nth['value'].values)
