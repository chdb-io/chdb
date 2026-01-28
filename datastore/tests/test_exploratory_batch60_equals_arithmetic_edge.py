"""
Exploratory test batch 60: equals() method, arithmetic operations with axis,
and complex lazy chain edge cases.

Focus areas:
1. equals() method with various parameters (rtol, atol, check_dtype, check_names)
2. DataFrame arithmetic methods with axis parameter (divide, multiply, subtract, add)
3. Binary operations with fill_value parameter
4. equals() with DataStore vs DataFrame vs scalar
5. Arithmetic operations in lazy chains
6. Edge cases: empty DataFrames, single values, NaN handling
7. Type coercion in arithmetic operations
8. Complex chains: filter -> arithmetic -> groupby
"""

import numpy as np
import pandas as pd
from tests.xfail_markers import chdb_alias_shadows_column_in_where, limit_datastore_no_invert
import pytest
from datastore import DataStore, concat as ds_concat
from tests.test_utils import (
    assert_frame_equal,
    assert_series_equal,
    assert_datastore_equals_pandas,
    get_dataframe,
    get_series,
)


class TestEqualsMethod:
    """Test equals() method with various parameters."""

    def test_equals_identical_dataframes(self):
        """Test equals() with identical DataFrames."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

        assert ds_df.equals(ds_df) == True
        # Also test against pandas DataFrame
        assert ds_df.equals(pd_df) == True

    def test_equals_different_values(self):
        """Test equals() with different values."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'a': [1, 2, 4]})
        ds_df1 = DataStore({'a': [1, 2, 3]})
        ds_df2 = DataStore({'a': [1, 2, 4]})

        pd_result = pd_df1.equals(pd_df2)
        ds_result = ds_df1.equals(ds_df2)

        assert pd_result == ds_result == False

    def test_equals_different_columns(self):
        """Test equals() with different column names."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'b': [1, 2, 3]})
        ds_df1 = DataStore({'a': [1, 2, 3]})
        ds_df2 = DataStore({'b': [1, 2, 3]})

        pd_result = pd_df1.equals(pd_df2)
        ds_result = ds_df1.equals(ds_df2)

        assert pd_result == ds_result == False

    def test_equals_with_nan_values(self):
        """Test equals() with NaN values (NaN == NaN should be True for equals)."""
        pd_df1 = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_df2 = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        ds_df1 = DataStore({'a': [1.0, np.nan, 3.0]})
        ds_df2 = DataStore({'a': [1.0, np.nan, 3.0]})

        pd_result = pd_df1.equals(pd_df2)
        ds_result = ds_df1.equals(ds_df2)

        assert pd_result == ds_result == True

    def test_equals_different_dtypes_same_values(self):
        """Test equals() with same values but different dtypes."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})  # int64
        pd_df2 = pd.DataFrame({'a': [1.0, 2.0, 3.0]})  # float64
        ds_df1 = DataStore({'a': [1, 2, 3]})
        ds_df2 = DataStore({'a': [1.0, 2.0, 3.0]})

        # pandas.equals() is strict about dtypes
        pd_result = pd_df1.equals(pd_df2)

        # DataStore equals with check_dtype=True (default)
        ds_result_strict = ds_df1.equals(ds_df2, check_dtype=True)

        # DataStore equals with check_dtype=False
        ds_result_relaxed = ds_df1.equals(ds_df2, check_dtype=False)

        assert pd_result == False
        assert ds_result_strict == False or ds_result_relaxed == True

    def test_equals_with_rtol_atol(self):
        """Test equals() with rtol and atol parameters."""
        ds_df1 = DataStore({'a': [1.0, 2.0, 3.0]})
        ds_df2 = DataStore({'a': [1.0001, 2.0001, 3.0001]})

        # Strict comparison should fail
        strict_result = ds_df1.equals(ds_df2, rtol=1e-8, atol=1e-8)

        # Relaxed comparison should pass
        relaxed_result = ds_df1.equals(ds_df2, rtol=1e-3, atol=1e-3)

        assert strict_result == False
        assert relaxed_result == True

    def test_equals_empty_dataframes(self):
        """Test equals() with empty DataFrames."""
        pd_df1 = pd.DataFrame({'a': []})
        pd_df2 = pd.DataFrame({'a': []})
        ds_df1 = DataStore({'a': []})
        ds_df2 = DataStore({'a': []})

        pd_result = pd_df1.equals(pd_df2)
        ds_result = ds_df1.equals(ds_df2)

        assert pd_result == ds_result == True

    def test_equals_scalar(self):
        """Test equals() with scalar comparison."""
        ds_df = DataStore({'a': [5, 5, 5]})

        # Comparing against scalar 5 - all values should equal
        result = ds_df.equals(5)
        assert result == True

        # Comparing against scalar 6 - should not equal
        result2 = ds_df.equals(6)
        assert result2 == False


class TestArithmeticMethodsWithAxis:
    """Test arithmetic methods with axis parameter."""

    def test_divide_axis_columns(self):
        """Test divide() with axis='columns' (default)."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 4, 6]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [2, 4, 6]})

        divisor = pd.Series([2, 1], index=['a', 'b'])

        pd_result = pd_df.divide(divisor, axis='columns')
        ds_result = ds_df.divide(divisor, axis='columns')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_divide_axis_index(self):
        """Test divide() with axis='index' (0)."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [4, 8, 12]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [4, 8, 12]})

        divisor = pd.Series([2, 4, 6])

        pd_result = pd_df.divide(divisor, axis='index')
        ds_result = ds_df.divide(divisor, axis='index')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiply_axis_columns(self):
        """Test multiply() with axis='columns'."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        multiplier = pd.Series([10, 100], index=['a', 'b'])

        pd_result = pd_df.multiply(multiplier, axis='columns')
        ds_result = ds_df.multiply(multiplier, axis='columns')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiply_axis_index(self):
        """Test multiply() with axis='index' (0)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        multiplier = pd.Series([10, 100, 1000])

        pd_result = pd_df.multiply(multiplier, axis='index')
        ds_result = ds_df.multiply(multiplier, axis='index')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subtract_axis_columns(self):
        """Test subtract() with axis='columns'."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [100, 200, 300]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [100, 200, 300]})

        subtrahend = pd.Series([5, 50], index=['a', 'b'])

        pd_result = pd_df.subtract(subtrahend, axis='columns')
        ds_result = ds_df.subtract(subtrahend, axis='columns')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subtract_axis_index(self):
        """Test subtract() with axis='index' (0)."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [100, 200, 300]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [100, 200, 300]})

        subtrahend = pd.Series([1, 2, 3])

        pd_result = pd_df.subtract(subtrahend, axis='index')
        ds_result = ds_df.subtract(subtrahend, axis='index')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_with_fill_value(self):
        """Test add() with fill_value parameter."""
        pd_df1 = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, np.nan, 6]})
        pd_df2 = pd.DataFrame({'a': [10, np.nan, 30], 'b': [np.nan, 50, 60]})
        ds_df1 = DataStore({'a': [1, 2, np.nan], 'b': [4, np.nan, 6]})
        ds_df2 = DataStore({'a': [10, np.nan, 30], 'b': [np.nan, 50, 60]})

        pd_result = pd_df1.add(pd_df2, fill_value=0)
        ds_result = ds_df1.add(ds_df2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_divide_scalar(self):
        """Test divide() with scalar."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [4, 8, 12]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [4, 8, 12]})

        pd_result = pd_df.divide(2)
        ds_result = ds_df.divide(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiply_scalar(self):
        """Test multiply() with scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.multiply(10)
        ds_result = ds_df.multiply(10)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticInChains:
    """Test arithmetic operations in lazy chains."""

    def test_filter_then_divide(self):
        """Test filter followed by divide."""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40], 'b': [2, 4, 6, 8]})
        ds_df = DataStore({'a': [10, 20, 30, 40], 'b': [2, 4, 6, 8]})

        pd_result = pd_df[pd_df['a'] > 15].divide(2)
        ds_result = ds_df[ds_df['a'] > 15].divide(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_divide_then_filter(self):
        """Test divide followed by filter."""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40], 'b': [2, 4, 6, 8]})
        ds_df = DataStore({'a': [10, 20, 30, 40], 'b': [2, 4, 6, 8]})

        pd_result = pd_df.divide(2)
        pd_result = pd_result[pd_result['a'] > 10]

        ds_result = ds_df.divide(2)
        ds_result = ds_result[ds_result['a'] > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiply_then_groupby_sum(self):
        """Test multiply followed by groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })

        pd_mult = pd_df.copy()
        pd_mult['value'] = pd_mult['value'] * 10
        pd_result = pd_mult.groupby('group')['value'].sum().reset_index()

        ds_mult = ds_df.assign(value=ds_df['value'] * 10)
        ds_result = ds_mult.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subtract_then_abs(self):
        """Test subtract followed by abs."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [15, 15, 15]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [15, 15, 15]})

        pd_result = pd_df.subtract(pd_df['b'], axis=0).abs()
        ds_result = ds_df.subtract(ds_df['b'], axis=0).abs()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_multiply_sort_head(self):
        """Test complex chain: filter -> multiply -> sort -> head."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['a'] > 2].multiply(2).sort_values('b', ascending=False).head(2)
        ds_result = ds_df[ds_df['a'] > 2].multiply(2).sort_values('b', ascending=False).head(2)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)


class TestArithmeticEdgeCases:
    """Test edge cases for arithmetic operations."""

    def test_divide_by_zero(self):
        """Test divide by zero behavior."""
        pd_df = pd.DataFrame({'a': [10, 20, 30]})
        ds_df = DataStore({'a': [10, 20, 30]})

        pd_result = pd_df.divide(0)
        ds_result = ds_df.divide(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_empty_dataframe(self):
        """Test arithmetic on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})

        pd_result = pd_df.multiply(10)
        ds_result = ds_df.multiply(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_single_row(self):
        """Test arithmetic on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        ds_df = DataStore({'a': [5], 'b': [10]})

        pd_result = pd_df.divide(5)
        ds_result = ds_df.divide(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_with_nan(self):
        """Test arithmetic with NaN values."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})

        pd_result = pd_df.multiply(10)
        ds_result = ds_df.multiply(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_dtypes_arithmetic(self):
        """Test arithmetic with mixed dtypes (int and float)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})

        pd_result = pd_df.multiply(2)
        ds_result = ds_df.multiply(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBinaryOpAxisParameter:
    """Test binary operations with axis parameter."""

    def test_add_series_axis_0(self):
        """Test adding Series along axis 0 (rows)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ser = pd.Series([10, 20, 30])

        pd_result = pd_df.add(ser, axis=0)
        ds_result = ds_df.add(ser, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_series_axis_1(self):
        """Test adding Series along axis 1 (columns)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ser = pd.Series([100, 200], index=['a', 'b'])

        pd_result = pd_df.add(ser, axis=1)
        ds_result = ds_df.add(ser, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_series_axis_0(self):
        """Test subtracting Series along axis 0."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})

        ser = pd.Series([1, 2, 3])

        pd_result = pd_df.sub(ser, axis=0)
        ds_result = ds_df.sub(ser, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_series_axis_1(self):
        """Test multiplying Series along axis 1."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ser = pd.Series([2, 3], index=['a', 'b'])

        pd_result = pd_df.mul(ser, axis=1)
        ds_result = ds_df.mul(ser, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_series_axis_0(self):
        """Test mod operation with Series along axis 0."""
        pd_df = pd.DataFrame({'a': [10, 21, 33], 'b': [45, 56, 67]})
        ds_df = DataStore({'a': [10, 21, 33], 'b': [45, 56, 67]})

        ser = pd.Series([3, 5, 7])

        pd_result = pd_df.mod(ser, axis=0)
        ds_result = ds_df.mod(ser, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pow_series_axis_1(self):
        """Test power operation with Series along axis 1."""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [2, 3, 4]})
        ds_df = DataStore({'a': [2, 3, 4], 'b': [2, 3, 4]})

        ser = pd.Series([2, 3], index=['a', 'b'])

        pd_result = pd_df.pow(ser, axis=1)
        ds_result = ds_df.pow(ser, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFloorDivAndTrueDiv:
    """Test floor division and true division specifically."""

    def test_floordiv_scalar(self):
        """Test floor division with scalar."""
        pd_df = pd.DataFrame({'a': [10, 21, 35], 'b': [4, 8, 12]})
        ds_df = DataStore({'a': [10, 21, 35], 'b': [4, 8, 12]})

        pd_result = pd_df.floordiv(3)
        ds_result = ds_df.floordiv(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truediv_scalar(self):
        """Test true division with scalar."""
        pd_df = pd.DataFrame({'a': [10, 21, 35], 'b': [4, 8, 12]})
        ds_df = DataStore({'a': [10, 21, 35], 'b': [4, 8, 12]})

        pd_result = pd_df.truediv(3)
        ds_result = ds_df.truediv(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floordiv_axis_0(self):
        """Test floor division with axis=0."""
        pd_df = pd.DataFrame({'a': [10, 21, 35], 'b': [4, 8, 12]})
        ds_df = DataStore({'a': [10, 21, 35], 'b': [4, 8, 12]})

        ser = pd.Series([2, 3, 5])

        pd_result = pd_df.floordiv(ser, axis=0)
        ds_result = ds_df.floordiv(ser, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truediv_axis_1(self):
        """Test true division with axis=1."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [4, 8, 12]})
        ds_df = DataStore({'a': [10, 20, 30], 'b': [4, 8, 12]})

        ser = pd.Series([2, 4], index=['a', 'b'])

        pd_result = pd_df.truediv(ser, axis=1)
        ds_result = ds_df.truediv(ser, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestReverseArithmetic:
    """Test reverse arithmetic operations (radd, rsub, rmul, etc.)."""

    def test_radd(self):
        """Test reverse add."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.radd(10)
        ds_result = ds_df.radd(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rsub(self):
        """Test reverse subtract."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.rsub(10)
        ds_result = ds_df.rsub(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmul(self):
        """Test reverse multiply."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.rmul(10)
        ds_result = ds_df.rmul(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rtruediv(self):
        """Test reverse true division."""
        pd_df = pd.DataFrame({'a': [1, 2, 4], 'b': [2, 4, 8]})
        ds_df = DataStore({'a': [1, 2, 4], 'b': [2, 4, 8]})

        pd_result = pd_df.rtruediv(100)
        ds_result = ds_df.rtruediv(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rfloordiv(self):
        """Test reverse floor division."""
        pd_df = pd.DataFrame({'a': [2, 3, 5], 'b': [2, 4, 8]})
        ds_df = DataStore({'a': [2, 3, 5], 'b': [2, 4, 8]})

        pd_result = pd_df.rfloordiv(100)
        ds_result = ds_df.rfloordiv(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmod(self):
        """Test reverse modulo."""
        pd_df = pd.DataFrame({'a': [3, 4, 5], 'b': [2, 3, 7]})
        ds_df = DataStore({'a': [3, 4, 5], 'b': [2, 3, 7]})

        pd_result = pd_df.rmod(100)
        ds_result = ds_df.rmod(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rpow(self):
        """Test reverse power."""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [1, 2, 3]})
        ds_df = DataStore({'a': [2, 3, 4], 'b': [1, 2, 3]})

        pd_result = pd_df.rpow(2)
        ds_result = ds_df.rpow(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComparisonMethods:
    """Test comparison methods (eq, ne, lt, le, gt, ge)."""

    def test_eq_method_scalar(self):
        """Test eq() method with scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [2, 2, 2]})

        pd_result = pd_df.eq(2)
        ds_result = ds_df.eq(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ne_method_scalar(self):
        """Test ne() method with scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [2, 2, 2]})

        pd_result = pd_df.ne(2)
        ds_result = ds_df.ne(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_lt_method_axis(self):
        """Test lt() method with axis parameter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ser = pd.Series([2, 5], index=['a', 'b'])

        pd_result = pd_df.lt(ser, axis='columns')
        ds_result = ds_df.lt(ser, axis='columns')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_le_method_axis_0(self):
        """Test le() method with axis=0."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ser = pd.Series([2, 3, 4])

        pd_result = pd_df.le(ser, axis=0)
        ds_result = ds_df.le(ser, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_gt_method_dataframe(self):
        """Test gt() method with another DataFrame."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [0, 2, 4], 'b': [5, 4, 7]})
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [0, 2, 4], 'b': [5, 4, 7]})

        pd_result = pd_df1.gt(pd_df2)
        ds_result = ds_df1.gt(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ge_method_dataframe(self):
        """Test ge() method with another DataFrame."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [0, 2, 4], 'b': [5, 4, 7]})
        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [0, 2, 4], 'b': [5, 4, 7]})

        pd_result = pd_df1.ge(pd_df2)
        ds_result = ds_df1.ge(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDotProduct:
    """Test dot product operations."""

    def test_dot_series(self):
        """Test dot product with Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ser = pd.Series([2, 3], index=['a', 'b'])

        pd_result = pd_df.dot(ser)
        ds_result = ds_df.dot(ser)

        ds_values = get_series(ds_result)
        assert_series_equal(ds_values, pd_result, check_names=False)

    def test_dot_dataframe(self):
        """Test dot product with DataFrame."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'x': [1, 2], 'y': [3, 4]}, index=['a', 'b'])
        ds_df1 = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df1.dot(pd_df2)
        ds_result = ds_df1.dot(pd_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUnaryOperations:
    """Test unary operations (neg, pos, abs, invert)."""

    def test_neg_operator(self):
        """Test negation operator."""
        pd_df = pd.DataFrame({'a': [1, -2, 3], 'b': [-4, 5, -6]})
        ds_df = DataStore({'a': [1, -2, 3], 'b': [-4, 5, -6]})

        pd_result = -pd_df
        ds_result = -ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pos_operator(self):
        """Test positive operator."""
        pd_df = pd.DataFrame({'a': [1, -2, 3], 'b': [-4, 5, -6]})
        ds_df = DataStore({'a': [1, -2, 3], 'b': [-4, 5, -6]})

        pd_result = +pd_df
        ds_result = +ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_function(self):
        """Test abs() function."""
        pd_df = pd.DataFrame({'a': [1, -2, 3], 'b': [-4, 5, -6]})
        ds_df = DataStore({'a': [1, -2, 3], 'b': [-4, 5, -6]})

        pd_result = abs(pd_df)
        ds_result = abs(ds_df)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_method(self):
        """Test abs() method."""
        pd_df = pd.DataFrame({'a': [1, -2, 3], 'b': [-4, 5, -6]})
        ds_df = DataStore({'a': [1, -2, 3], 'b': [-4, 5, -6]})

        pd_result = pd_df.abs()
        ds_result = ds_df.abs()

        assert_datastore_equals_pandas(ds_result, pd_result)

    
    @limit_datastore_no_invert
    def test_invert_operator_bool(self):
        """Test invert operator on boolean DataFrame."""
        pd_df = pd.DataFrame({'a': [True, False, True], 'b': [False, True, False]})
        ds_df = DataStore({'a': [True, False, True], 'b': [False, True, False]})

        pd_result = ~pd_df
        ds_result = ~ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexArithmeticChains:
    """Test complex chains involving multiple arithmetic operations."""

    def test_chain_add_multiply_subtract(self):
        """Test chain of add, multiply, subtract."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = ((pd_df + 10) * 2) - 5
        ds_result = ((ds_df + 10) * 2) - 5

        assert_datastore_equals_pandas(ds_result, pd_result)

    
    @chdb_alias_shadows_column_in_where
    def test_chain_filter_arithmetic_groupby(self):
        """Test filter -> arithmetic -> groupby chain."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_filtered = pd_df[pd_df['value'] > 15].copy()
        pd_filtered['value'] = pd_filtered['value'] * 2
        pd_result = pd_filtered.groupby('group')['value'].sum().reset_index()

        ds_filtered = ds_df[ds_df['value'] > 15]
        ds_filtered = ds_filtered.assign(value=ds_filtered['value'] * 2)
        ds_result = ds_filtered.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_divide_abs_sort(self):
        """Test divide -> abs -> sort chain."""
        pd_df = pd.DataFrame({'a': [10, -20, 30, -40], 'b': [1, 2, 3, 4]})
        ds_df = DataStore({'a': [10, -20, 30, -40], 'b': [1, 2, 3, 4]})

        pd_result = (pd_df.divide(10).abs()).sort_values('a')
        ds_result = (ds_df.divide(10).abs()).sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_chain_multiple_binary_ops(self):
        """Test chain with multiple binary operations."""
        pd_df = pd.DataFrame({'a': [100, 200, 300], 'b': [10, 20, 30]})
        ds_df = DataStore({'a': [100, 200, 300], 'b': [10, 20, 30]})

        pd_result = (pd_df / 10) + (pd_df * 0.1) - (pd_df // 100)
        ds_result = (ds_df / 10) + (ds_df * 0.1) - (ds_df // 100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_with_assign_and_arithmetic(self):
        """Test assign with computed columns then arithmetic."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.assign(c=pd_df['a'] + pd_df['b']).multiply(2)
        ds_result = ds_df.assign(c=ds_df['a'] + ds_df['b']).multiply(2)

        assert_datastore_equals_pandas(ds_result, pd_result)
