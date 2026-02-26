"""
Exploratory Test Batch 27: Reverse Operators and Edge Cases

Focus areas:
1. Reverse arithmetic operators (__radd__, __rsub__, __rmul__, __rtruediv__, etc.)
2. DataFrame binary operations with fill_value parameter
3. Method-based arithmetic (add, sub, mul, div with fill_value)
4. ColumnExpr reverse operators
5. Mixed scalar/DataFrame/Series operations
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


class TestReverseArithmeticOperators:
    """Test reverse arithmetic operators (scalar op DataFrame)."""

    def test_radd_scalar_plus_datastore(self):
        """Test scalar + DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 + pd_df
        ds_result = 10 + ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rsub_scalar_minus_datastore(self):
        """Test scalar - DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 - pd_df
        ds_result = 10 - ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmul_scalar_times_datastore(self):
        """Test scalar * DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 2 * pd_df
        ds_result = 2 * ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rtruediv_scalar_div_datastore(self):
        """Test scalar / DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 4], 'b': [4, 5, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 100 / pd_df
        ds_result = 100 / ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rfloordiv_scalar_floordiv_datastore(self):
        """Test scalar // DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 // pd_df
        ds_result = 10 // ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmod_scalar_mod_datastore(self):
        """Test scalar % DataStore."""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [5, 6, 7]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 % pd_df
        ds_result = 10 % ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rpow_scalar_pow_datastore(self):
        """Test scalar ** DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 2**pd_df
        ds_result = 2**ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMethodArithmeticWithFillValue:
    """Test method-based arithmetic with fill_value parameter."""

    def test_add_with_fill_value(self):
        """Test df.add() with fill_value for missing values."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'c': [7, 8, 9]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd_df1.add(pd_df2, fill_value=0)
        ds_result = ds_df1.add(ds_df2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_with_fill_value(self):
        """Test df.sub() with fill_value for missing values."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'c': [7, 8, 9]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd_df1.sub(pd_df2, fill_value=0)
        ds_result = ds_df1.sub(ds_df2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_with_fill_value(self):
        """Test df.mul() with fill_value."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [2, 3, 4], 'c': [5, 6, 7]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd_df1.mul(pd_df2, fill_value=1)
        ds_result = ds_df1.mul(ds_df2, fill_value=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_div_with_fill_value(self):
        """Test df.div() with fill_value."""
        pd_df1 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_df2 = pd.DataFrame({'a': [2, 4, 5], 'c': [5, 10, 15]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd_df1.div(pd_df2, fill_value=1)
        ds_result = ds_df1.div(ds_df2, fill_value=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_radd_method(self):
        """Test df.radd() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.radd(10)
        ds_result = ds_df.radd(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rsub_method(self):
        """Test df.rsub() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rsub(10)
        ds_result = ds_df.rsub(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rdiv_method(self):
        """Test df.rdiv() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 4], 'b': [5, 10, 20]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rdiv(100)
        ds_result = ds_df.rdiv(100)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnExprReverseOperators:
    """Test reverse operators on ColumnExpr (Series-like)."""

    def test_columnexpr_radd(self):
        """Test scalar + ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 + pd_df['a']
        ds_result = 10 + ds_df['a']

        # Execute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_columnexpr_rsub(self):
        """Test scalar - ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 - pd_df['a']
        ds_result = 10 - ds_df['a']

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_columnexpr_rmul(self):
        """Test scalar * ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 5 * pd_df['a']
        ds_result = 5 * ds_df['a']

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_columnexpr_rtruediv(self):
        """Test scalar / ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 5], 'b': [4, 5, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 100 / pd_df['a']
        ds_result = 100 / ds_df['a']

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_columnexpr_rfloordiv(self):
        """Test scalar // ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 // pd_df['a']
        ds_result = 10 // ds_df['a']

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_columnexpr_rmod(self):
        """Test scalar % ColumnExpr."""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [5, 6, 7]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 10 % pd_df['a']
        ds_result = 10 % ds_df['a']

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_columnexpr_rpow(self):
        """Test scalar ** ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = 2 ** pd_df['a']
        ds_result = 2 ** ds_df['a']

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)


class TestMixedOperations:
    """Test mixed operations between DataStore, ColumnExpr, and scalars."""

    def test_datastore_plus_columnexpr(self):
        """Test DataStore + ColumnExpr broadcasting."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        # Adding a Series to all columns
        pd_result = pd_df.add(pd_df['a'], axis=0)
        ds_result = ds_df.add(get_series(ds_df['a']), axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_complex_arithmetic_chain(self):
        """Test complex arithmetic chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = (pd_df * 2 + 10) / 5 - 1
        ds_result = (ds_df * 2 + 10) / 5 - 1

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_and_abs(self):
        """Test unary negation and abs."""
        pd_df = pd.DataFrame({'a': [-1, 2, -3], 'b': [4, -5, 6]})
        ds_df = DataStore(pd_df.copy())

        # Negation
        pd_neg = -pd_df
        ds_neg = -ds_df
        assert_datastore_equals_pandas(ds_neg, pd_neg)

        # Abs
        pd_abs = abs(pd_df)
        ds_abs = abs(ds_df)
        assert_datastore_equals_pandas(ds_abs, pd_abs)  # abs may return uint64

    def test_positive_operator(self):
        """Test unary positive operator."""
        pd_df = pd.DataFrame({'a': [-1, 2, -3], 'b': [4, -5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = +pd_df
        ds_result = +ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBinaryOpsWithNaN:
    """Test binary operations with NaN values."""

    def test_add_with_nan(self):
        """Test addition with NaN values."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df + 10
        ds_result = ds_df + 10

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sub_with_nan(self):
        """Test subtraction with NaN values."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df - 5
        ds_result = ds_df - 5

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_with_nan(self):
        """Test multiplication with NaN values."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df * 2
        ds_result = ds_df * 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_div_with_nan(self):
        """Test division with NaN values."""
        pd_df = pd.DataFrame({'a': [10, np.nan, 30], 'b': [40, 50, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df / 10
        ds_result = ds_df / 10

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBinaryOpsWithInf:
    """Test binary operations with infinity values."""

    def test_add_with_inf(self):
        """Test addition with infinity."""
        pd_df = pd.DataFrame({'a': [1, np.inf, 3], 'b': [4, 5, -np.inf]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df + 10
        ds_result = ds_df + 10

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mul_with_inf(self):
        """Test multiplication with infinity."""
        pd_df = pd.DataFrame({'a': [1, np.inf, 3], 'b': [4, 5, -np.inf]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df * 2
        ds_result = ds_df * 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_div_by_zero(self):
        """Test division by zero (should produce inf)."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [0, 5, 0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'] / pd_df['b']
        ds_result = get_series(ds_df['a']) / get_series(ds_df['b'])

        assert_series_equal(ds_result, pd_result)


class TestDataFrameScalarComparison:
    """Test DataFrame comparison with scalars."""

    def test_dataframe_eq_scalar(self):
        """Test DataFrame == scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df == 2
        ds_result = ds_df == 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_ne_scalar(self):
        """Test DataFrame != scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df != 2
        ds_result = ds_df != 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_gt_scalar(self):
        """Test DataFrame > scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df > 2
        ds_result = ds_df > 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_lt_scalar(self):
        """Test DataFrame < scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df < 3
        ds_result = ds_df < 3

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_ge_scalar(self):
        """Test DataFrame >= scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df >= 2
        ds_result = ds_df >= 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_le_scalar(self):
        """Test DataFrame <= scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df <= 3
        ds_result = ds_df <= 3

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDivideMultiplySubtractMethods:
    """Test divide(), multiply(), subtract() method aliases."""

    def test_divide_method(self):
        """Test df.divide() method."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.divide(10)
        ds_result = ds_df.divide(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiply_method(self):
        """Test df.multiply() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.multiply(10)
        ds_result = ds_df.multiply(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subtract_method(self):
        """Test df.subtract() method."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.subtract(5)
        ds_result = ds_df.subtract(5)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTruedivFloordivMethods:
    """Test truediv() and floordiv() method variants."""

    def test_truediv_method(self):
        """Test df.truediv() method."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.truediv(10)
        ds_result = ds_df.truediv(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floordiv_method(self):
        """Test df.floordiv() method."""
        pd_df = pd.DataFrame({'a': [10, 21, 32], 'b': [40, 53, 64]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.floordiv(10)
        ds_result = ds_df.floordiv(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_method(self):
        """Test df.mod() method."""
        pd_df = pd.DataFrame({'a': [10, 21, 32], 'b': [40, 53, 64]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.mod(7)
        ds_result = ds_df.mod(7)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pow_method(self):
        """Test df.pow() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.pow(2)
        ds_result = ds_df.pow(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestReverseMethods:
    """Test reverse method variants (rmul, rdiv, etc.)."""

    def test_rmul_method(self):
        """Test df.rmul() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rmul(10)
        ds_result = ds_df.rmul(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rtruediv_method(self):
        """Test df.rtruediv() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 5], 'b': [4, 5, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rtruediv(100)
        ds_result = ds_df.rtruediv(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rfloordiv_method(self):
        """Test df.rfloordiv() method."""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [5, 6, 7]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rfloordiv(20)
        ds_result = ds_df.rfloordiv(20)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rmod_method(self):
        """Test df.rmod() method."""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [5, 6, 7]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rmod(10)
        ds_result = ds_df.rmod(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rpow_method(self):
        """Test df.rpow() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rpow(2)
        ds_result = ds_df.rpow(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComparisonMethods:
    """Test comparison method variants (lt, gt, le, ge, eq, ne)."""

    def test_lt_method(self):
        """Test df.lt() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.lt(2)
        ds_result = ds_df.lt(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_gt_method(self):
        """Test df.gt() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.gt(2)
        ds_result = ds_df.gt(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_le_method(self):
        """Test df.le() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.le(2)
        ds_result = ds_df.le(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ge_method(self):
        """Test df.ge() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.ge(2)
        ds_result = ds_df.ge(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eq_method(self):
        """Test df.eq() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.eq(2)
        ds_result = ds_df.eq(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ne_method(self):
        """Test df.ne() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.ne(2)
        ds_result = ds_df.ne(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCases:
    """Test edge cases for arithmetic operations."""

    def test_empty_dataframe_arithmetic(self):
        """Test arithmetic on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df + 10
        ds_result = ds_df + 10

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_arithmetic(self):
        """Test arithmetic on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df * 3 + 5
        ds_result = ds_df * 3 + 5

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_column_arithmetic(self):
        """Test arithmetic on single-column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df**2
        ds_result = ds_df**2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_large_values_arithmetic(self):
        """Test arithmetic with large values."""
        pd_df = pd.DataFrame({'a': [1e15, 2e15, 3e15], 'b': [1e16, 2e16, 3e16]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df * 2
        ds_result = ds_df * 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_small_values_arithmetic(self):
        """Test arithmetic with small values."""
        pd_df = pd.DataFrame({'a': [1e-10, 2e-10, 3e-10], 'b': [1e-15, 2e-15, 3e-15]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df * 1e10
        ds_result = ds_df * 1e10

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
