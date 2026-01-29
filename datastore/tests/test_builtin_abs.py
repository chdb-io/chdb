"""
Test Python's built-in abs() function with ColumnExpr.

This test verifies that abs(ds['col']) works the same as ds['col'].abs()
and matches pandas abs() behavior.

Mirror Pattern: Each test compares DataStore result with pandas result.
"""

import pandas as pd
import pytest
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_series_equal, assert_frame_equal


class TestBuiltinAbs:
    """Test built-in abs() function with ColumnExpr."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame with positive and negative values."""
        df = pd.DataFrame({
            'a': [-1, 2, -3, 4, -5],
            'b': [1.5, -2.5, 3.5, -4.5, 5.5],
            'c': [0, 0, 0, 0, 0],
        })
        ds = DataStore(df.copy())
        return df, ds

    def test_abs_basic_integers(self, sample_data):
        """Test abs() with integer column."""
        df, ds = sample_data

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_basic_floats(self, sample_data):
        """Test abs() with float column."""
        df, ds = sample_data

        # pandas
        pd_result = abs(df['b'])

        # DataStore
        ds_result = abs(ds['b'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_zeros(self, sample_data):
        """Test abs() with zeros."""
        df, ds = sample_data

        # pandas
        pd_result = abs(df['c'])

        # DataStore
        ds_result = abs(ds['c'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_in_filter(self, sample_data):
        """Test abs() in filter condition - the main use case from the bug report."""
        df, ds = sample_data

        # pandas
        pd_result = df[abs(df['a']) > 2]

        # DataStore
        ds_result = ds[abs(ds['a']) > 2]

        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True)
        )

    def test_abs_in_complex_filter(self, sample_data):
        """Test abs() combined with other conditions."""
        df, ds = sample_data

        # pandas
        pd_result = df[(abs(df['a']) > 2) & (df['b'] > 0)]

        # DataStore
        ds_result = ds[(abs(ds['a']) > 2) & (ds['b'] > 0)]

        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True)
        )

    def test_abs_equivalent_to_method(self, sample_data):
        """Verify abs(col) is equivalent to col.abs()."""
        df, ds = sample_data

        # Method version
        ds_method = ds['a'].abs()

        # Builtin version
        ds_builtin = abs(ds['a'])

        # Both should produce the same result
        assert list(ds_method) == list(ds_builtin)

    def test_abs_in_assignment(self, sample_data):
        """Test abs() result can be assigned to a new column."""
        df, ds = sample_data

        # pandas
        pd_result = df.assign(abs_a=abs(df['a']))

        # DataStore
        ds_result = ds.assign(abs_a=abs(ds['a']))

        # Compare - check_dtype=False due to chDB returning uint64 for abs
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_abs_chained_operations(self, sample_data):
        """Test abs() can be chained with other operations."""
        df, ds = sample_data

        # pandas: abs value + 10
        pd_result = abs(df['a']) + 10

        # DataStore: abs value + 10
        ds_result = abs(ds['a']) + 10

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_with_arithmetic_expression(self, sample_data):
        """Test abs() on arithmetic expression."""
        df, ds = sample_data

        # pandas: abs of (a - 3)
        pd_result = abs(df['a'] - 3)

        # DataStore: abs of (a - 3)
        ds_result = abs(ds['a'] - 3)

        # Compare - check_dtype=False due to chDB returning uint64 for abs
        # check_names=False because arithmetic expressions lose the column name in DataStore
        assert_series_equal(ds_result, pd_result, check_dtype=False, check_names=False)


class TestBuiltinAbsEdgeCases:
    """Test edge cases for abs() with ColumnExpr."""

    def test_abs_with_nulls(self):
        """Test abs() with NULL/NaN values."""
        df = pd.DataFrame({'a': [-1, None, -3, np.nan, -5]})
        ds = DataStore(df.copy())

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare - NaN handling should match
        assert_series_equal(ds_result, pd_result)

    def test_abs_all_negative(self):
        """Test abs() when all values are negative."""
        df = pd.DataFrame({'a': [-1, -2, -3, -4, -5]})
        ds = DataStore(df.copy())

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_all_positive(self):
        """Test abs() when all values are already positive."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df.copy())

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_single_row(self):
        """Test abs() with single row DataFrame."""
        df = pd.DataFrame({'a': [-42]})
        ds = DataStore(df.copy())

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_large_numbers(self):
        """Test abs() with large numbers."""
        df = pd.DataFrame({'a': [-1000000000, 1000000000, -999999999]})
        ds = DataStore(df.copy())

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare
        assert_series_equal(ds_result, pd_result)

    def test_abs_very_small_floats(self):
        """Test abs() with very small float values."""
        df = pd.DataFrame({'a': [-1e-10, 1e-10, -1e-15, 1e-15]})
        ds = DataStore(df.copy())

        # pandas
        pd_result = abs(df['a'])

        # DataStore
        ds_result = abs(ds['a'])

        # Compare
        assert_series_equal(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
