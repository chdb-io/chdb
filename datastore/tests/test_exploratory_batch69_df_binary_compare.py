"""
Exploratory Batch 69: DataFrame Binary Operations and Comparison Edge Cases

Focus areas:
1. DataFrame binary arithmetic with mixed dtypes
2. Comparison operations with special values (NaN, None, inf)
3. Broadcasting behavior with scalars
4. DataFrame vs DataFrame operations
5. Boolean mask combinations

Discovery method: Architecture-based exploration focusing on column_expr.py and pandas_compat.py
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestDataFrameArithmeticMixedTypes:
    """Test arithmetic operations with mixed numeric types"""
    
    def test_add_int_float_columns(self):
        """Add integer column to float column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_subtract_float_from_int(self):
        """Subtract float column from integer column"""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [1.5, 2.5, 3.5]})
        pd_result = pd_df['a'] - pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] - ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_multiply_int_by_float(self):
        """Multiply integer column by float column"""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [1.5, 2.5, 3.5]})
        pd_result = pd_df['a'] * pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] * ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_divide_int_by_float(self):
        """Divide integer column by float column"""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2.0, 4.0, 5.0]})
        pd_result = pd_df['a'] / pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] / ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_floor_divide_mixed_types(self):
        """Floor divide with mixed types"""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [3.0, 7.0, 4.0]})
        pd_result = pd_df['a'] // pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] // ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_modulo_mixed_types(self):
        """Modulo with mixed types"""
        pd_df = pd.DataFrame({'a': [10, 20, 33], 'b': [3.0, 7.0, 4.0]})
        pd_result = pd_df['a'] % pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] % ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_power_int_float(self):
        """Power operation with int base and float exponent"""
        pd_df = pd.DataFrame({'a': [2, 3, 4], 'b': [0.5, 2.0, 0.5]})
        pd_result = pd_df['a'] ** pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] ** ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticWithNaN:
    """Test arithmetic operations with NaN values"""
    
    def test_add_with_nan(self):
        """Addition with NaN in column"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, 6.0]})
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_multiply_with_nan(self):
        """Multiplication with NaN in both columns"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, np.nan, 6.0]})
        pd_result = pd_df['a'] * pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] * ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_divide_by_nan(self):
        """Division where divisor contains NaN"""
        pd_df = pd.DataFrame({'a': [10.0, 20.0, 30.0], 'b': [2.0, np.nan, 5.0]})
        pd_result = pd_df['a'] / pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] / ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticWithInfinity:
    """Test arithmetic operations with infinity values"""
    
    def test_add_with_inf(self):
        """Addition with infinity"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, 3.0], 'b': [4.0, 5.0, 6.0]})
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_multiply_inf_by_zero(self):
        """Multiply infinity by zero (should be NaN)"""
        pd_df = pd.DataFrame({'a': [np.inf, -np.inf, 3.0], 'b': [0.0, 0.0, 6.0]})
        pd_result = pd_df['a'] * pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] * ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_inf_minus_inf(self):
        """Infinity minus infinity (should be NaN)"""
        pd_df = pd.DataFrame({'a': [np.inf, 1.0, 3.0], 'b': [np.inf, 2.0, 4.0]})
        pd_result = pd_df['a'] - pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] - ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_divide_by_zero(self):
        """Division by zero produces infinity"""
        pd_df = pd.DataFrame({'a': [10.0, -10.0, 0.0], 'b': [0.0, 0.0, 0.0]})
        pd_result = pd_df['a'] / pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] / ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestScalarBroadcasting:
    """Test broadcasting scalars to DataFrames/columns"""
    
    def test_add_scalar_to_column(self):
        """Add scalar to column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df['a'] + 10
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + 10
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_subtract_scalar_from_column(self):
        """Subtract scalar from column"""
        pd_df = pd.DataFrame({'a': [10, 20, 30]})
        pd_result = pd_df['a'] - 5
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] - 5
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_scalar_minus_column(self):
        """Scalar minus column (reverse operation)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = 100 - pd_df['a']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = 100 - ds_df['a']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_multiply_column_by_scalar(self):
        """Multiply column by scalar"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df['a'] * 2.5
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] * 2.5
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_divide_column_by_scalar(self):
        """Divide column by scalar"""
        pd_df = pd.DataFrame({'a': [10, 20, 30]})
        pd_result = pd_df['a'] / 2
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] / 2
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_scalar_divide_by_column(self):
        """Scalar divided by column (reverse)"""
        pd_df = pd.DataFrame({'a': [2, 4, 5]})
        pd_result = 100 / pd_df['a']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = 100 / ds_df['a']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_column_power_scalar(self):
        """Column raised to scalar power"""
        pd_df = pd.DataFrame({'a': [2, 3, 4]})
        pd_result = pd_df['a'] ** 2
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] ** 2
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_scalar_power_column(self):
        """Scalar raised to column power"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = 2 ** pd_df['a']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = 2 ** ds_df['a']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComparisonEdgeCases:
    """Test comparison operations with edge case values"""
    
    def test_compare_with_nan(self):
        """Compare column containing NaN"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_result = pd_df['a'] > 2
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] > 2
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_equal_nan_nan(self):
        """NaN == NaN (should be False in pandas)"""
        pd_df = pd.DataFrame({'a': [np.nan, 1.0, np.nan], 'b': [np.nan, np.nan, 1.0]})
        pd_result = pd_df['a'] == pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] == ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_inf_values(self):
        """Compare infinity with finite values"""
        pd_df = pd.DataFrame({'a': [np.inf, -np.inf, 0.0, 1e308]})
        pd_result = pd_df['a'] > 0
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] > 0
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_inf_equal_inf(self):
        """Infinity equals infinity"""
        pd_df = pd.DataFrame({'a': [np.inf, -np.inf, np.inf], 'b': [np.inf, np.inf, -np.inf]})
        pd_result = pd_df['a'] == pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] == ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_columns_equal(self):
        """Column equality comparison"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 3, 3]})
        pd_result = pd_df['a'] == pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] == ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_not_equal(self):
        """Column not-equal comparison"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 3, 3]})
        pd_result = pd_df['a'] != pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] != ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_less_than(self):
        """Column less-than comparison"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        pd_result = pd_df['a'] < pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] < ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_less_equal(self):
        """Column less-or-equal comparison"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        pd_result = pd_df['a'] <= pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] <= ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_greater_equal(self):
        """Column greater-or-equal comparison"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        pd_result = pd_df['a'] >= pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] >= ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanMaskOperations:
    """Test boolean mask combinations"""
    
    def test_and_masks(self):
        """Combine two boolean masks with AND"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_mask = (pd_df['a'] > 1) & (pd_df['b'] < 35)
        pd_result = pd_df[pd_mask]
        
        ds_df = DataStore(pd_df.copy())
        ds_mask = (ds_df['a'] > 1) & (ds_df['b'] < 35)
        ds_result = ds_df[ds_mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_or_masks(self):
        """Combine two boolean masks with OR"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_mask = (pd_df['a'] == 1) | (pd_df['b'] == 40)
        pd_result = pd_df[pd_mask]
        
        ds_df = DataStore(pd_df.copy())
        ds_mask = (ds_df['a'] == 1) | (ds_df['b'] == 40)
        ds_result = ds_df[ds_mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_not_mask(self):
        """Invert boolean mask"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        pd_mask = ~(pd_df['a'] > 2)
        pd_result = pd_df[pd_mask]
        
        ds_df = DataStore(pd_df.copy())
        ds_mask = ~(ds_df['a'] > 2)
        ds_result = ds_df[ds_mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_complex_boolean_expression(self):
        """Complex nested boolean expression"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})
        pd_mask = ((pd_df['a'] > 1) & (pd_df['b'] < 45)) | (pd_df['c'] == 500)
        pd_result = pd_df[pd_mask]
        
        ds_df = DataStore(pd_df.copy())
        ds_mask = ((ds_df['a'] > 1) & (ds_df['b'] < 45)) | (ds_df['c'] == 500)
        ds_result = ds_df[ds_mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_xor_masks(self):
        """XOR of two boolean masks"""
        pd_df = pd.DataFrame({'a': [True, True, False, False], 'b': [True, False, True, False]})
        pd_result = pd_df['a'] ^ pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] ^ ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEmptyDataFrameOperations:
    """Test operations on empty DataFrames"""
    
    def test_add_columns_empty_df(self):
        """Add columns of empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='int64')})
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b']
        
        # Empty DataFrames may have different dtypes between chdb and pandas
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)
    
    def test_compare_empty_column(self):
        """Comparison on empty column"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='float64')})
        pd_result = pd_df['a'] > 0
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] > 0
        
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)
    
    def test_filter_empty_df(self):
        """Filter empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64')})
        pd_result = pd_df[pd_df['a'] > 0]
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 0]
        
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


class TestSingleRowDataFrame:
    """Test operations on single-row DataFrames"""
    
    def test_arithmetic_single_row(self):
        """Arithmetic on single row DataFrame"""
        pd_df = pd.DataFrame({'a': [5], 'b': [3]})
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_compare_single_row(self):
        """Comparison on single row DataFrame"""
        pd_df = pd.DataFrame({'a': [5]})
        pd_result = pd_df['a'] > 3
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] > 3
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_filter_single_row_match(self):
        """Filter single row - match"""
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        pd_result = pd_df[pd_df['a'] > 3]
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 3]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_filter_single_row_no_match(self):
        """Filter single row - no match"""
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        pd_result = pd_df[pd_df['a'] > 10]
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 10]
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullableIntegerArithmetic:
    """Test arithmetic with nullable integer types"""
    
    def test_add_nullable_int(self):
        """Add nullable integer columns"""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, None], dtype='Int64'), 'b': pd.array([4, None, 6], dtype='Int64')})
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b']
        
        # Nullable types may differ between chdb and pandas
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)
    
    def test_multiply_nullable_int(self):
        """Multiply nullable integer columns"""
        pd_df = pd.DataFrame({'a': pd.array([2, 3, None], dtype='Int64'), 'b': pd.array([4, None, 6], dtype='Int64')})
        pd_result = pd_df['a'] * pd_df['b']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] * ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)
    
    def test_compare_nullable_int(self):
        """Compare nullable integer columns"""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, None, 4], dtype='Int64')})
        pd_result = pd_df['a'] > 2
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] > 2
        
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


class TestChainedArithmetic:
    """Test chained arithmetic operations"""
    
    def test_triple_add(self):
        """Chain three additions"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        pd_result = pd_df['a'] + pd_df['b'] + pd_df['c']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] + ds_df['b'] + ds_df['c']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_mixed_operations_chain(self):
        """Chain mixed operations (add, multiply, subtract)"""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 4, 5], 'c': [1, 1, 1]})
        pd_result = pd_df['a'] / pd_df['b'] - pd_df['c']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] / ds_df['b'] - ds_df['c']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_parenthesized_operations(self):
        """Parenthesized operations for order"""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 4, 5], 'c': [3, 3, 3]})
        pd_result = pd_df['a'] * (pd_df['b'] + pd_df['c'])
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'] * (ds_df['b'] + ds_df['c'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_complex_expression(self):
        """Complex multi-operator expression"""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 4, 5], 'c': [3, 3, 3], 'd': [1, 2, 3]})
        pd_result = (pd_df['a'] + pd_df['b']) * pd_df['c'] - pd_df['d']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = (ds_df['a'] + ds_df['b']) * ds_df['c'] - ds_df['d']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUnaryOperations:
    """Test unary operations"""
    
    def test_negation(self):
        """Negate column"""
        pd_df = pd.DataFrame({'a': [1, -2, 3]})
        pd_result = -pd_df['a']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = -ds_df['a']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    # Bug fixed: ColumnExpr now implements __pos__ (unary +)
    def test_positive(self):
        """Positive unary operator - BUG: not implemented"""
        pd_df = pd.DataFrame({'a': [1, -2, 3]})
        pd_result = +pd_df['a']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = +ds_df['a']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_abs(self):
        """Absolute value"""
        pd_df = pd.DataFrame({'a': [1, -2, 3, -4]})
        pd_result = pd_df['a'].abs()
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df['a'].abs()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_invert_bool_column(self):
        """Invert boolean column"""
        pd_df = pd.DataFrame({'a': [True, False, True]})
        pd_result = ~pd_df['a']
        
        ds_df = DataStore(pd_df.copy())
        ds_result = ~ds_df['a']
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAssignAndOperate:
    """Test assigning new columns then operating on them"""
    
    def test_assign_then_add(self):
        """Assign column then add to another"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = pd_df['a'] * 2
        pd_result = pd_df['a'] + pd_df['b']
        
        ds_df = DataStore({'a': [1, 2, 3]})
        ds_df['b'] = ds_df['a'] * 2
        ds_result = ds_df['a'] + ds_df['b']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_assign_chain_then_compare(self):
        """Assign multiple columns then compare"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        pd_df['b'] = pd_df['a'] * 2
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df[pd_df['c'] > 5]
        
        ds_df = DataStore({'a': [1, 2, 3, 4]})
        ds_df['b'] = ds_df['a'] * 2
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df[ds_df['c'] > 5]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_filter_then_assign(self):
        """Filter first then assign"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        pd_filtered['b'] = pd_filtered['a'] * 10
        pd_result = pd_filtered
        
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_filtered = ds_df[ds_df['a'] > 2]
        ds_filtered['b'] = ds_filtered['a'] * 10
        ds_result = ds_filtered
        
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
