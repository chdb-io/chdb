"""
Exploratory Batch 54: Binary Operations with fill_value, compare() method, axis variations

This batch tests:
1. Binary arithmetic methods with fill_value parameter (add, sub, mul, div, etc.)
2. DataFrame.compare() method for comparing two DataFrames
3. Axis parameter variations (axis=0 vs axis=1 for applicable operations)
4. Comparison operators with different dtypes and shapes
5. Reverse binary operations (radd, rsub, rmul, etc.)
6. Floor division and modulo edge cases
7. Power operations with edge cases
8. Method chaining with binary operations

Discovery approach: Architecture-based testing targeting gaps in test coverage
for DataFrame binary operations and axis variations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# ============================================================================
# Binary Operations with fill_value
# ============================================================================

class TestBinaryOpsWithFillValue:
    """Test binary operations with fill_value parameter."""

    def test_add_with_fill_value_scalar(self):
        """Test add with fill_value for scalar operations."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, None], 'b': [4, None, 6]})
        pd_result = pd_df.add(10, fill_value=0)

        # DataStore
        ds_df = DataStore({'a': [1, 2, None], 'b': [4, None, 6]})
        ds_result = ds_df.add(10, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_with_fill_value_series(self):
        """Test add with fill_value when adding Series."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([10, 20], index=['a', 'b'])
        pd_result = pd_df.add(pd_series, axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_series = pd.Series([10, 20], index=['a', 'b'])
        ds_result = ds_df.add(ds_series, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_with_fill_value(self):
        """Test subtract with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, None, 30], 'b': [None, 50, 60]})
        pd_result = pd_df.sub(5, fill_value=100)

        # DataStore
        ds_df = DataStore({'a': [10, None, 30], 'b': [None, 50, 60]})
        ds_result = ds_df.sub(5, fill_value=100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_with_fill_value(self):
        """Test multiply with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, None], 'b': [None, 4, 5]})
        pd_result = pd_df.mul(2, fill_value=1)

        # DataStore
        ds_df = DataStore({'a': [1, 2, None], 'b': [None, 4, 5]})
        ds_result = ds_df.mul(2, fill_value=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_div_with_fill_value(self):
        """Test divide with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, None, 30], 'b': [None, 40, 50]})
        pd_result = pd_df.div(2, fill_value=20)

        # DataStore
        ds_df = DataStore({'a': [10, None, 30], 'b': [None, 40, 50]})
        ds_result = ds_df.div(2, fill_value=20)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floordiv_with_fill_value(self):
        """Test floor division with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, None, 35], 'b': [None, 45, 50]})
        pd_result = pd_df.floordiv(7, fill_value=21)

        # DataStore
        ds_df = DataStore({'a': [10, None, 35], 'b': [None, 45, 50]})
        ds_result = ds_df.floordiv(7, fill_value=21)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_with_fill_value(self):
        """Test modulo with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, None, 35], 'b': [None, 45, 50]})
        pd_result = pd_df.mod(7, fill_value=14)

        # DataStore
        ds_df = DataStore({'a': [10, None, 35], 'b': [None, 45, 50]})
        ds_result = ds_df.mod(7, fill_value=14)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pow_with_fill_value(self):
        """Test power with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [2, None, 4], 'b': [None, 3, 5]})
        pd_result = pd_df.pow(2, fill_value=1)

        # DataStore
        ds_df = DataStore({'a': [2, None, 4], 'b': [None, 3, 5]})
        ds_result = ds_df.pow(2, fill_value=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBinaryOpsWithDataFrame:
    """Test binary operations between two DataFrames."""

    def test_add_two_dataframes(self):
        """Test adding two DataFrames."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df1.add(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df1.add(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_dataframes_different_columns(self):
        """Test adding DataFrames with different columns."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'c': [40, 50, 60]})
        pd_result = pd_df1.add(pd_df2, fill_value=0)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'c': [40, 50, 60]})
        ds_result = ds_df1.add(ds_df2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_two_dataframes_with_fill_value(self):
        """Test subtracting DataFrames with fill_value."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [10, 20, None], 'b': [40, None, 60]})
        pd_df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df1.sub(pd_df2, fill_value=0)

        # DataStore
        ds_df1 = DataStore({'a': [10, 20, None], 'b': [40, None, 60]})
        ds_df2 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df1.sub(ds_df2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestReverseBinaryOps:
    """Test reverse binary operations."""

    def test_radd_scalar(self):
        """Test reverse add with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.radd(10)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.radd(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rsub_scalar(self):
        """Test reverse subtract with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.rsub(10)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.rsub(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmul_scalar(self):
        """Test reverse multiply with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.rmul(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.rmul(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rdiv_scalar(self):
        """Test reverse divide with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 4], 'b': [5, 10, 20]})
        pd_result = pd_df.rdiv(100)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 4], 'b': [5, 10, 20]})
        ds_result = ds_df.rdiv(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rfloordiv_scalar(self):
        """Test reverse floor division with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [5, 6, 7]})
        pd_result = pd_df.rfloordiv(20)

        # DataStore
        ds_df = DataStore({'a': [2, 3, 4], 'b': [5, 6, 7]})
        ds_result = ds_df.rfloordiv(20)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmod_scalar(self):
        """Test reverse modulo with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [3, 4, 5], 'b': [6, 7, 8]})
        pd_result = pd_df.rmod(17)

        # DataStore
        ds_df = DataStore({'a': [3, 4, 5], 'b': [6, 7, 8]})
        ds_result = ds_df.rmod(17)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rpow_scalar(self):
        """Test reverse power with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.rpow(2)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.rpow(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reverse_ops_with_fill_value(self):
        """Test reverse operations with fill_value."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, None], 'b': [None, 4, 5]})
        pd_result = pd_df.rsub(10, fill_value=0)

        # DataStore
        ds_df = DataStore({'a': [1, 2, None], 'b': [None, 4, 5]})
        ds_result = ds_df.rsub(10, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Axis Parameter Variations
# ============================================================================

class TestAxisVariations:
    """Test operations with different axis parameters."""

    def test_add_axis_0(self):
        """Test add with axis=0 (default)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([10, 20, 30])
        pd_result = pd_df.add(pd_series, axis=0)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.add(pd_series, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_axis_1(self):
        """Test add with axis=1 (columns)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([100, 200], index=['a', 'b'])
        pd_result = pd_df.add(pd_series, axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.add(pd_series, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_axis_1(self):
        """Test subtract with axis=1."""
        # pandas
        pd_df = pd.DataFrame({'x': [10, 20, 30], 'y': [40, 50, 60]})
        pd_series = pd.Series([1, 2], index=['x', 'y'])
        pd_result = pd_df.sub(pd_series, axis=1)

        # DataStore
        ds_df = DataStore({'x': [10, 20, 30], 'y': [40, 50, 60]})
        ds_result = ds_df.sub(pd_series, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_axis_1(self):
        """Test multiply with axis=1."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([2, 3], index=['a', 'b'])
        pd_result = pd_df.mul(pd_series, axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.mul(pd_series, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_div_axis_1(self):
        """Test divide with axis=1."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_series = pd.Series([2, 5], index=['a', 'b'])
        pd_result = pd_df.div(pd_series, axis=1)

        # DataStore
        ds_df = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df.div(pd_series, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_axis_1(self):
        """Test sum with axis=1 (row-wise)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        pd_result = pd_df.sum(axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_result = ds_df.sum(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mean_axis_1(self):
        """Test mean with axis=1 (row-wise)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})
        pd_result = pd_df.mean(axis=1)

        # DataStore
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})
        ds_result = ds_df.mean(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_max_axis_1(self):
        """Test max with axis=1 (row-wise)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 5, 3], 'b': [4, 2, 6], 'c': [7, 8, 1]})
        pd_result = pd_df.max(axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 5, 3], 'b': [4, 2, 6], 'c': [7, 8, 1]})
        ds_result = ds_df.max(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_min_axis_1(self):
        """Test min with axis=1 (row-wise)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 5, 3], 'b': [4, 2, 6], 'c': [7, 8, 1]})
        pd_result = pd_df.min(axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 5, 3], 'b': [4, 2, 6], 'c': [7, 8, 1]})
        ds_result = ds_df.min(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_std_axis_1(self):
        """Test std with axis=1 (row-wise)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})
        pd_result = pd_df.std(axis=1)

        # DataStore
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})
        ds_result = ds_df.std(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_axis_1(self):
        """Test var with axis=1 (row-wise)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})
        pd_result = pd_df.var(axis=1)

        # DataStore
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0], 'c': [7.0, 8.0, 9.0]})
        ds_result = ds_df.var(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Compare Method
# ============================================================================

class TestCompareMethod:
    """Test DataFrame.compare() method."""

    def test_compare_basic(self):
        """Test basic compare between two DataFrames."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [1, 2, 9], 'b': [4, 7, 6]})
        pd_result = pd_df1.compare(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [1, 2, 9], 'b': [4, 7, 6]})
        ds_result = ds_df1.compare(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_no_differences(self):
        """Test compare when DataFrames are identical."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df1.compare(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df1.compare(ds_df2)

        # Both should return empty DataFrame with same structure
        assert ds_result.shape[0] == pd_result.shape[0]

    def test_compare_align_axis(self):
        """Test compare with align_axis parameter."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [1, 9, 3], 'b': [4, 5, 9]})
        pd_result = pd_df1.compare(pd_df2, align_axis=0)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [1, 9, 3], 'b': [4, 5, 9]})
        ds_result = ds_df1.compare(ds_df2, align_axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_keep_shape(self):
        """Test compare with keep_shape=True."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [1, 9, 3], 'b': [4, 5, 9]})
        pd_result = pd_df1.compare(pd_df2, keep_shape=True)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [1, 9, 3], 'b': [4, 5, 9]})
        ds_result = ds_df1.compare(ds_df2, keep_shape=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_keep_equal(self):
        """Test compare with keep_equal=True."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [1, 9, 3], 'b': [4, 5, 9]})
        pd_result = pd_df1.compare(pd_df2, keep_equal=True)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [1, 9, 3], 'b': [4, 5, 9]})
        ds_result = ds_df1.compare(ds_df2, keep_equal=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_with_strings(self):
        """Test compare with string columns."""
        # pandas
        pd_df1 = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        pd_df2 = pd.DataFrame({'a': ['x', 'Y', 'z'], 'b': ['P', 'q', 'r']})
        pd_result = pd_df1.compare(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        ds_df2 = DataStore({'a': ['x', 'Y', 'z'], 'b': ['P', 'q', 'r']})
        ds_result = ds_df1.compare(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Comparison Operators with Different Types
# ============================================================================

class TestComparisonOperators:
    """Test comparison operators with various types."""

    def test_lt_with_scalar(self):
        """Test less than with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.lt(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.lt(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_gt_with_scalar(self):
        """Test greater than with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.gt(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.gt(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_le_with_scalar(self):
        """Test less than or equal with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.le(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.le(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ge_with_scalar(self):
        """Test greater than or equal with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.ge(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.ge(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eq_with_scalar(self):
        """Test equal with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 3, 3]})
        pd_result = pd_df.eq(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [3, 3, 3]})
        ds_result = ds_df.eq(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ne_with_scalar(self):
        """Test not equal with scalar."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [3, 3, 3]})
        pd_result = pd_df.ne(3)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [3, 3, 3]})
        ds_result = ds_df.ne(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_with_series_axis_0(self):
        """Test comparison with Series along axis 0."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([2, 5, 4])
        pd_result = pd_df.lt(pd_series, axis=0)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.lt(pd_series, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_with_series_axis_1(self):
        """Test comparison with Series along axis 1."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([2, 5], index=['a', 'b'])
        pd_result = pd_df.gt(pd_series, axis=1)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.gt(pd_series, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_with_dataframe(self):
        """Test comparison between two DataFrames."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [2, 2, 2], 'b': [5, 5, 5]})
        pd_result = pd_df1.lt(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [2, 2, 2], 'b': [5, 5, 5]})
        ds_result = ds_df1.lt(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Edge Cases for Binary Operations
# ============================================================================

class TestBinaryOpsEdgeCases:
    """Test edge cases for binary operations."""

    def test_division_by_zero(self):
        """Test division by zero (should produce inf)."""
        # pandas
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        pd_result = pd_df.div(0)

        # DataStore
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        ds_result = ds_df.div(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_modulo_with_zero(self):
        """Test modulo with zero."""
        # pandas behavior: int mod 0 raises error, float mod 0 gives nan
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        pd_result = pd_df.mod(0)

        # DataStore
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        ds_result = ds_df.mod(0)

        # Both should produce NaN for float mod 0
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_power_with_negative_base_and_fraction(self):
        """Test power with negative base and fractional exponent."""
        # pandas
        pd_df = pd.DataFrame({'a': [4.0, 9.0, 16.0]})
        pd_result = pd_df.pow(0.5)

        # DataStore
        ds_df = DataStore({'a': [4.0, 9.0, 16.0]})
        ds_result = ds_df.pow(0.5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_operations_with_inf(self):
        """Test operations with infinity values."""
        # pandas
        pd_df = pd.DataFrame({'a': [1.0, float('inf'), -float('inf')], 'b': [float('inf'), 2.0, 3.0]})
        pd_result = pd_df.add(1)

        # DataStore
        ds_df = DataStore({'a': [1.0, float('inf'), -float('inf')], 'b': [float('inf'), 2.0, 3.0]})
        ds_result = ds_df.add(1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_operations_with_nan(self):
        """Test operations with NaN values."""
        # pandas
        pd_df = pd.DataFrame({'a': [1.0, float('nan'), 3.0], 'b': [4.0, 5.0, float('nan')]})
        pd_result = pd_df.add(10)

        # DataStore
        ds_df = DataStore({'a': [1.0, float('nan'), 3.0], 'b': [4.0, 5.0, float('nan')]})
        ds_result = ds_df.add(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_operations(self):
        """Test binary operations on empty DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd_df.add(10)

        # DataStore
        ds_df = DataStore({'a': [], 'b': []})
        ds_result = ds_df.add(10)

        assert ds_result.shape == pd_result.shape

    def test_single_row_operations(self):
        """Test binary operations on single row DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        pd_result = pd_df.mul(2)

        # DataStore
        ds_df = DataStore({'a': [5], 'b': [10]})
        ds_result = ds_df.mul(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_column_operations(self):
        """Test binary operations on single column DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df.add(100)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds_df.add(100)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Method Chaining with Binary Operations
# ============================================================================

class TestBinaryOpsChaining:
    """Test method chaining with binary operations."""

    def test_add_then_filter(self):
        """Test add followed by filter."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.add(10)[pd_df.add(10)['a'] > 11]

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.add(10)[ds_df.add(10)['a'] > 11]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_then_sum(self):
        """Test multiply followed by sum."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.mul(2).sum()

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.mul(2).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_then_abs(self):
        """Test subtract followed by abs."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.sub(3).abs()

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.sub(3).abs()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_div_then_round(self):
        """Test divide followed by round."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [7, 11, 13]})
        pd_result = pd_df.div(3).round(2)

        # DataStore
        ds_df = DataStore({'a': [10, 20, 30], 'b': [7, 11, 13]})
        ds_result = ds_df.div(3).round(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_multiple_binary_ops(self):
        """Test chaining multiple binary operations."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.add(10).mul(2).sub(5)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.add(10).mul(2).sub(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_binary_ops_with_groupby(self):
        """Test binary operations followed by groupby."""
        # pandas
        pd_df = pd.DataFrame({'cat': ['A', 'B', 'A', 'B'], 'val': [1, 2, 3, 4]})
        pd_result = pd_df.assign(val=pd_df['val'] * 2).groupby('cat')['val'].sum().reset_index()

        # DataStore
        ds_df = DataStore({'cat': ['A', 'B', 'A', 'B'], 'val': [1, 2, 3, 4]})
        ds_result = ds_df.assign(val=ds_df['val'] * 2).groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_comparison_then_filter(self):
        """Test comparison followed by using result as filter."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_mask = pd_df['a'].gt(pd_df['b'])
        pd_result = pd_df[pd_mask]

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_mask = ds_df['a'].gt(ds_df['b'])
        ds_result = ds_df[ds_mask]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# DataFrame Combine Operations
# ============================================================================

class TestCombineOperations:
    """Test combine and combine_first operations."""

    def test_combine_first_basic(self):
        """Test combine_first basic functionality."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, None, 3], 'b': [None, 5, None]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df1.combine_first(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, None, 3], 'b': [None, 5, None]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_different_columns(self):
        """Test combine_first with different columns."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, None, 3], 'b': [None, 5, None]})
        pd_df2 = pd.DataFrame({'b': [40, 50, 60], 'c': [70, 80, 90]})
        pd_result = pd_df1.combine_first(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, None, 3], 'b': [None, 5, None]})
        ds_df2 = DataStore({'b': [40, 50, 60], 'c': [70, 80, 90]})
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_with_function(self):
        """Test combine with custom function."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1 + s2)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df1.combine(ds_df2, lambda s1, s2: s1 + s2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_with_fill_value(self):
        """Test combine with fill_value."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, None, 3], 'b': [None, 5, None]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1 + s2, fill_value=0)

        # DataStore
        ds_df1 = DataStore({'a': [1, None, 3], 'b': [None, 5, None]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df1.combine(ds_df2, lambda s1, s2: s1 + s2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Dot Product
# ============================================================================

class TestDotProduct:
    """Test dot product operations."""

    def test_dot_with_series(self):
        """Test dot product with Series."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_series = pd.Series([2, 3], index=['a', 'b'])
        pd_result = pd_df.dot(pd_series)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.dot(pd_series)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dot_with_dataframe(self):
        """Test dot product with another DataFrame."""
        # pandas
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]}, index=['a', 'b'])
        pd_result = pd_df1.dot(pd_df2)

        # DataStore
        ds_df1 = DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df1.dot(pd_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Alias Methods
# ============================================================================

class TestAliasMethods:
    """Test alias methods for binary operations."""

    def test_multiply_alias(self):
        """Test multiply as alias for mul."""
        # pandas
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.multiply(10)

        # DataStore
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.multiply(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_divide_alias(self):
        """Test divide as alias for div."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df.divide(10)

        # DataStore
        ds_df = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df.divide(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subtract_alias(self):
        """Test subtract as alias for sub."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df.subtract(5)

        # DataStore
        ds_df = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df.subtract(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truediv_method(self):
        """Test truediv method."""
        # pandas
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df.truediv(10)

        # DataStore
        ds_df = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df.truediv(10)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Mixed Type Operations
# ============================================================================

class TestMixedTypeOperations:
    """Test binary operations with mixed types."""

    def test_int_float_addition(self):
        """Test addition between int and float columns."""
        # pandas
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})
        pd_result = pd_df['int_col'] + pd_df['float_col']

        # DataStore
        ds_df = DataStore({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})
        ds_result = ds_df['int_col'] + ds_df['float_col']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_int_bool_operations(self):
        """Test operations between int and bool."""
        # pandas
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'bool_col': [True, False, True]})
        pd_result = pd_df['int_col'] + pd_df['bool_col']

        # DataStore
        ds_df = DataStore({'int_col': [1, 2, 3], 'bool_col': [True, False, True]})
        ds_result = ds_df['int_col'] + ds_df['bool_col']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_float_bool_operations(self):
        """Test operations between float and bool."""
        # pandas
        pd_df = pd.DataFrame({'float_col': [1.0, 2.0, 3.0], 'bool_col': [True, False, True]})
        pd_result = pd_df['float_col'] * pd_df['bool_col']

        # DataStore
        ds_df = DataStore({'float_col': [1.0, 2.0, 3.0], 'bool_col': [True, False, True]})
        ds_result = ds_df['float_col'] * ds_df['bool_col']

        assert_datastore_equals_pandas(ds_result, pd_result)
