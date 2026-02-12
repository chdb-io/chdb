"""
Tests for equals() and binary operation alignment semantics.

Key Behaviors Being Tested:

1. equals() method - considers both values AND index (pandas compatible)
2. values_equal() method - position-based comparison only (ignores index)
3. Binary operations - position alignment (DataStore design choice, differs from pandas)

Design Documentation:
- DataStore equals() follows pandas Series.equals() semantics: same values AND same index
- DataStore values_equal() provides position-based comparison for cases where index doesn't matter
- Binary operations (+-*/) use position alignment, NOT pandas label alignment
  - This is a deliberate design choice because SQL cannot do label alignment
  - Known limitation: cross-DataStore ops may produce different results than pandas
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import get_series


class TestEqualsMethodIndex:
    """Test ColumnExpr.equals() considers index correctly."""

    def test_equals_same_values_same_index(self):
        """equals() returns True when values and index match."""
        ds = DataStore({'a': [1, 2, 3]})
        pd_series = pd.Series([1, 2, 3], index=[0, 1, 2], name='a')

        # Both should be True
        assert ds['a'].equals(pd_series) == True

    def test_equals_same_values_different_index(self):
        """equals() returns False when values match but index differs.

        This is pandas-compatible behavior: Series.equals() considers index.
        """
        ds = DataStore({'a': [1, 2, 3]})
        pd_series = pd.Series([1, 2, 3], index=[10, 20, 30], name='a')

        # Should be False because index differs
        assert ds['a'].equals(pd_series) == False

    def test_equals_different_values_same_index(self):
        """equals() returns False when values differ."""
        ds = DataStore({'a': [1, 2, 3]})
        pd_series = pd.Series([1, 2, 4], index=[0, 1, 2], name='a')

        assert ds['a'].equals(pd_series) == False

    def test_equals_matches_pandas_semantics(self):
        """equals() mirrors pandas Series.equals() behavior."""
        # pandas reference
        pd_s1 = pd.Series([1, 2, 3], index=[0, 1, 2])
        pd_s2 = pd.Series([1, 2, 3], index=[0, 1, 2])
        pd_s3 = pd.Series([1, 2, 3], index=[10, 20, 30])

        # DataStore
        ds = DataStore({'a': [1, 2, 3]})

        # Both pandas and DataStore should agree
        assert pd_s1.equals(pd_s2) == ds['a'].equals(pd_s2)
        assert pd_s1.equals(pd_s3) == ds['a'].equals(pd_s3)


class TestValuesEqualMethod:
    """Test ColumnExpr.values_equal() for position-based comparison."""

    def test_values_equal_ignores_index(self):
        """values_equal() compares by position, ignoring index."""
        ds = DataStore({'a': [1, 2, 3]})
        pd_series = pd.Series([1, 2, 3], index=[10, 20, 30])

        # values_equal should return True (same values by position)
        assert ds['a'].values_equal(pd_series) == True

    def test_values_equal_different_values(self):
        """values_equal() returns False when values differ."""
        ds = DataStore({'a': [1, 2, 3]})
        pd_series = pd.Series([1, 2, 4], index=[0, 1, 2])

        assert ds['a'].values_equal(pd_series) == False

    def test_values_equal_with_float_tolerance(self):
        """values_equal() uses tolerance for float comparison."""
        ds = DataStore({'a': [1.0, 2.0, 3.0]})
        pd_series = pd.Series([1.0000001, 2.0, 3.0])

        # Should be True with default tolerance
        assert ds['a'].values_equal(pd_series) == True

    def test_values_equal_vs_equals_difference(self):
        """Demonstrate difference between equals() and values_equal()."""
        ds = DataStore({'a': [1, 2, 3]})
        pd_series = pd.Series([1, 2, 3], index=[10, 20, 30])

        # equals() considers index - returns False
        assert ds['a'].equals(pd_series) == False

        # values_equal() ignores index - returns True
        assert ds['a'].values_equal(pd_series) == True


class TestBinaryOperationAlignment:
    """Test binary operation alignment behavior.

    DESIGN NOTE: DataStore uses position alignment for binary operations,
    which differs from pandas' label alignment. This is a deliberate choice
    because SQL cannot do label alignment.
    """

    def test_same_index_alignment(self):
        """Binary ops work correctly when indices match."""
        ds = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})

        # Same DataStore columns should align correctly
        ds_result = get_series(ds['a'] + ds['b'])
        pd_result = pd.Series([1, 2, 3]) + pd.Series([10, 20, 30])

        assert list(ds_result) == list(pd_result)

    def test_default_rangeindex_alignment(self):
        """Binary ops with default RangeIndex work correctly."""
        ds1 = DataStore({'a': [1, 2, 3]})
        ds2 = DataStore({'b': [10, 20, 30]})

        ds_result = get_series(ds1['a'] + ds2['b'])

        # Position alignment: [1+10, 2+20, 3+30]
        assert list(ds_result) == [11, 22, 33]

    @pytest.mark.xfail(
        reason="DataStore uses position alignment, not pandas label alignment"
    )
    def test_mismatched_index_pandas_semantics(self):
        """Binary ops with mismatched index should align by label like pandas.

        NOTE: This test documents the behavior difference. DataStore uses
        position alignment, pandas uses label alignment.
        """
        pd_s1 = pd.Series([1, 2, 3], index=[1, 2, 3])
        pd_s2 = pd.Series([10, 20, 30], index=[2, 3, 4])

        ds1 = DataStore({'a': [1, 2, 3]}, index=[1, 2, 3])
        ds2 = DataStore({'b': [10, 20, 30]}, index=[2, 3, 4])

        pd_result = pd_s1 + pd_s2
        ds_result = get_series(ds1['a'] + ds2['b'])

        # pandas produces: {1: NaN, 2: 12.0, 3: 23.0, 4: NaN}
        # DataStore produces: [11, 22, 33] (position aligned)
        assert len(pd_result) == len(ds_result)
        assert list(pd_result.values) == list(ds_result)

    def test_position_alignment_behavior(self):
        """Document actual DataStore position alignment behavior."""
        ds1 = DataStore({'a': [1, 2, 3]}, index=[1, 2, 3])
        ds2 = DataStore({'b': [10, 20, 30]}, index=[2, 3, 4])

        ds_result = get_series(ds1['a'] + ds2['b'])

        # DataStore aligns by position: [1+10, 2+20, 3+30]
        # Index difference is ignored
        assert list(ds_result) == [11, 22, 33]

    def test_comparison_ops_position_aligned(self):
        """Comparison operations also use position alignment (within same DataStore)."""
        ds = DataStore({'a': [1, 2, 3], 'b': [0, 2, 5]})

        ds_gt = get_series(ds['a'] > ds['b'])
        ds_eq = get_series(ds['a'] == ds['b'])
        ds_lt = get_series(ds['a'] < ds['b'])

        # Position-aligned comparisons
        assert list(ds_gt) == [True, False, False]  # [1>0, 2>2, 3>5]
        assert list(ds_eq) == [False, True, False]  # [1==0, 2==2, 3==5]
        assert list(ds_lt) == [False, False, True]  # [1<0, 2<2, 3<5]

    def test_arithmetic_ops_same_datastore(self):
        """Arithmetic operations within same DataStore."""
        ds = DataStore({'a': [10, 20, 30], 'b': [2, 4, 5]})

        ds_add = get_series(ds['a'] + ds['b'])
        ds_sub = get_series(ds['a'] - ds['b'])
        ds_mul = get_series(ds['a'] * ds['b'])
        ds_div = get_series(ds['a'] / ds['b'])

        assert list(ds_add) == [12, 24, 35]
        assert list(ds_sub) == [8, 16, 25]
        assert list(ds_mul) == [20, 80, 150]
        assert list(ds_div) == [5.0, 5.0, 6.0]


class TestDataFrameEquals:
    """Test DataFrame-level equals behavior."""

    def test_dataframe_equals_identical(self):
        """Two identical DataFrames should be equal."""
        ds1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds2 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        df1 = ds1.to_df()
        df2 = ds2.to_df()

        assert df1.equals(df2) == True

    def test_dataframe_equals_different_index(self):
        """DataFrames with different index should not be equal."""
        ds1 = DataStore({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds2 = DataStore({'a': [1, 2, 3]}, index=[10, 20, 30])

        df1 = ds1.to_df()
        df2 = ds2.to_df()

        assert df1.equals(df2) == False


class TestEqualsWithNaN:
    """Test equals behavior with NaN values."""

    def test_equals_nan_positions_match(self):
        """equals() returns True when NaN positions match."""
        ds = DataStore({'a': [1.0, np.nan, 3.0]})
        pd_series = pd.Series([1.0, np.nan, 3.0])

        # pandas equals() treats NaN == NaN as True
        assert ds['a'].equals(pd_series) == True

    def test_equals_nan_positions_differ(self):
        """equals() returns False when NaN positions differ."""
        ds = DataStore({'a': [1.0, np.nan, 3.0]})
        pd_series = pd.Series([1.0, 2.0, np.nan])

        assert ds['a'].equals(pd_series) == False

    def test_values_equal_nan_handling(self):
        """values_equal() handles NaN correctly by position."""
        ds = DataStore({'a': [1.0, np.nan, 3.0]})
        pd_series = pd.Series([1.0, np.nan, 3.0], index=[10, 20, 30])

        # Different index but same values including NaN positions
        assert ds['a'].values_equal(pd_series) == True
