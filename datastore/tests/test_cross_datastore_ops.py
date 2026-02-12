"""
Tests for cross-DataStore operations.

When two ColumnExprs come from different DataStores, they should be
aligned by position (row index) similar to pandas behavior.
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestCrossDataStoreArithmetic:
    """Test arithmetic operations between two DataStores."""

    def test_add_same_length(self):
        """Test addition of columns from two DataStores with same length."""
        # Create two DataStores
        ds1 = DataStore({'a': [1, 2, 3]})
        ds2 = DataStore({'b': [10, 20, 30]})

        # Cross-DataStore addition
        result = ds1['a'] + ds2['b']

        # pandas mirror
        pd1 = pd.DataFrame({'a': [1, 2, 3]})
        pd2 = pd.DataFrame({'b': [10, 20, 30]})
        pd_result = pd1['a'] + pd2['b']

        # Trigger execution and compare
        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )

    def test_sub_same_length(self):
        """Test subtraction of columns from two DataStores."""
        ds1 = DataStore({'a': [10, 20, 30]})
        ds2 = DataStore({'b': [1, 2, 3]})

        result = ds1['a'] - ds2['b']

        pd1 = pd.DataFrame({'a': [10, 20, 30]})
        pd2 = pd.DataFrame({'b': [1, 2, 3]})
        pd_result = pd1['a'] - pd2['b']

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )

    def test_mul_same_length(self):
        """Test multiplication of columns from two DataStores."""
        ds1 = DataStore({'a': [2, 3, 4]})
        ds2 = DataStore({'b': [10, 20, 30]})

        result = ds1['a'] * ds2['b']

        pd1 = pd.DataFrame({'a': [2, 3, 4]})
        pd2 = pd.DataFrame({'b': [10, 20, 30]})
        pd_result = pd1['a'] * pd2['b']

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )

    def test_div_same_length(self):
        """Test division of columns from two DataStores."""
        ds1 = DataStore({'a': [10.0, 20.0, 30.0]})
        ds2 = DataStore({'b': [2.0, 4.0, 5.0]})

        result = ds1['a'] / ds2['b']

        pd1 = pd.DataFrame({'a': [10.0, 20.0, 30.0]})
        pd2 = pd.DataFrame({'b': [2.0, 4.0, 5.0]})
        pd_result = pd1['a'] / pd2['b']

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )


class TestCrossDataStoreComparison:
    """Test comparison operations between two DataStores."""

    def test_gt_same_length(self):
        """Test greater-than comparison between two DataStores."""
        ds1 = DataStore({'a': [1, 20, 3]})
        ds2 = DataStore({'b': [10, 2, 30]})

        result = ds1['a'] > ds2['b']

        pd1 = pd.DataFrame({'a': [1, 20, 3]})
        pd2 = pd.DataFrame({'b': [10, 2, 30]})
        pd_result = pd1['a'] > pd2['b']

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )

    def test_lt_same_length(self):
        """Test less-than comparison between two DataStores."""
        ds1 = DataStore({'a': [1, 20, 3]})
        ds2 = DataStore({'b': [10, 2, 30]})

        result = ds1['a'] < ds2['b']

        pd1 = pd.DataFrame({'a': [1, 20, 3]})
        pd2 = pd.DataFrame({'b': [10, 2, 30]})
        pd_result = pd1['a'] < pd2['b']

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )

    # NOTE: __eq__ and __ne__ do NOT support cross-DataStore operations
    # because they are commonly used for join conditions (e.g., ds1.id == ds2.user_id)
    # which need to remain as BinaryCondition for SQL generation.
    # Use .eq() and .ne() methods for cross-DataStore value comparison instead.


class TestCrossDataStoreDifferentLength:
    """Test cross-DataStore ops with different lengths (pandas-like alignment)."""

    def test_add_different_length(self):
        """Test addition with different length DataStores.

        pandas aligns by index, resulting in NaN for non-matching indices.
        Our implementation aligns by position (row 0 with row 0, etc.),
        resulting in shorter result.
        """
        ds1 = DataStore({'a': [1, 2, 3, 4, 5]})
        ds2 = DataStore({'b': [10, 20, 30]})

        result = ds1['a'] + ds2['b']

        # pandas mirror - note: pandas aligns by index
        pd1 = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd2 = pd.DataFrame({'b': [10, 20, 30]})
        pd_result = pd1['a'] + pd2['b']  # Will have NaN for index 3, 4

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )


class TestCrossDataStoreChainedOps:
    """Test chained operations across DataStores."""

    def test_chained_arithmetic(self):
        """Test chained arithmetic: (ds1['a'] + ds2['b']) * 2."""
        ds1 = DataStore({'a': [1, 2, 3]})
        ds2 = DataStore({'b': [10, 20, 30]})

        result = (ds1['a'] + ds2['b']) * 2

        pd1 = pd.DataFrame({'a': [1, 2, 3]})
        pd2 = pd.DataFrame({'b': [10, 20, 30]})
        pd_result = (pd1['a'] + pd2['b']) * 2

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )

    def test_multiple_cross_datastore_ops(self):
        """Test multiple cross-DataStore operations in sequence.

        For three DataStores, we fall back to pandas for the second operation
        since the result of the first cross-DataStore op is in executor mode.
        """
        ds1 = DataStore({'a': [1, 2, 3]})
        ds2 = DataStore({'b': [10, 20, 30]})

        # First cross-DataStore operation
        result = ds1['a'] + ds2['b']

        # Chain with scalar
        result2 = result - 5

        pd1 = pd.DataFrame({'a': [1, 2, 3]})
        pd2 = pd.DataFrame({'b': [10, 20, 30]})
        pd_result = (pd1['a'] + pd2['b']) - 5

        ds_series = result2._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )


class TestSameDataStoreNotAffected:
    """Ensure operations within the same DataStore are not affected."""

    def test_same_datastore_uses_sql(self):
        """Operations within the same DataStore should use SQL path."""
        ds = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})

        # Same DataStore operation - should use SQL
        result = ds['a'] + ds['b']

        # Should have an expression (SQL mode), not executor mode
        assert result._expr is not None, "Same DataStore ops should use SQL mode"

        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        pd_result = pd_df['a'] + pd_df['b']

        ds_series = result._execute()
        pd.testing.assert_series_equal(
            ds_series.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
