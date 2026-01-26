"""
Exploratory Batch 46: Aggregation Edge Cases, Type Coercion Chains, Stack/Unstack

Focus areas:
1. Aggregation boundary cases - prod(), sem(), var() with edge cases
2. Complex lazy chains with type coercion - astype() interactions
3. Stack/Unstack operations with various index types
4. Index manipulation chains - set_index + operations + reset_index
5. Numeric operation edge cases - floordiv, mod with zeros and negatives
6. String accessor advanced cases - str methods with NaN, edge values
7. Datetime accessor edge cases - with timezone-naive and edge values
8. Correlation and covariance edge cases
9. Value transformation with clipping and rounding combinations
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, date

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_series
from tests.xfail_markers import chdb_no_product_function


class TestAggregationEdgeCases:
    """Test aggregation operations with edge cases."""

    def test_prod_basic(self):
        """Test basic product aggregation."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})

        pd_result = pd_df.prod()
        ds_result = ds_df.prod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_prod_with_zeros(self):
        """Test product with zeros in data."""
        pd_df = pd.DataFrame({'A': [1, 0, 3, 4], 'B': [2, 3, 0, 5]})
        ds_df = DataStore({'A': [1, 0, 3, 4], 'B': [2, 3, 0, 5]})

        pd_result = pd_df.prod()
        ds_result = ds_df.prod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_prod_with_negative(self):
        """Test product with negative values."""
        pd_df = pd.DataFrame({'A': [-1, 2, -3, 4], 'B': [2, -3, 4, -5]})
        ds_df = DataStore({'A': [-1, 2, -3, 4], 'B': [2, -3, 4, -5]})

        pd_result = pd_df.prod()
        ds_result = ds_df.prod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_prod_single_row(self):
        """Test product with single row."""
        pd_df = pd.DataFrame({'A': [5], 'B': [10]})
        ds_df = DataStore({'A': [5], 'B': [10]})

        pd_result = pd_df.prod()
        ds_result = ds_df.prod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_prod_all_same_values(self):
        """Test product with all same values."""
        pd_df = pd.DataFrame({'A': [2, 2, 2, 2], 'B': [3, 3, 3, 3]})
        ds_df = DataStore({'A': [2, 2, 2, 2], 'B': [3, 3, 3, 3]})

        pd_result = pd_df.prod()
        ds_result = ds_df.prod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_basic(self):
        """Test basic variance."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})

        pd_result = pd_df.var()
        ds_result = ds_df.var()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_ddof_0(self):
        """Test variance with ddof=0 (population variance)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})

        pd_result = pd_df.var(ddof=0)
        ds_result = ds_df.var(ddof=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_all_same_values(self):
        """Test variance with all same values (should be 0)."""
        pd_df = pd.DataFrame({'A': [5, 5, 5, 5], 'B': [3, 3, 3, 3]})
        ds_df = DataStore({'A': [5, 5, 5, 5], 'B': [3, 3, 3, 3]})

        pd_result = pd_df.var()
        ds_result = ds_df.var()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_single_value(self):
        """Test variance with single value (should be NaN)."""
        pd_df = pd.DataFrame({'A': [5], 'B': [3]})
        ds_df = DataStore({'A': [5], 'B': [3]})

        pd_result = pd_df.var()
        ds_result = ds_df.var()

        # With single value, var() returns NaN
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_std_basic(self):
        """Test basic standard deviation."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})

        pd_result = pd_df.std()
        ds_result = ds_df.std()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_std_ddof_0(self):
        """Test std with ddof=0."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})

        pd_result = pd_df.std(ddof=0)
        ds_result = ds_df.std(ddof=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_filter_var(self):
        """Test filter then variance."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [2, 4, 6, 8, 10, 12]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5, 6], 'B': [2, 4, 6, 8, 10, 12]})

        pd_result = pd_df[pd_df['A'] > 2].var()
        ds_result = ds_df[ds_df['A'] > 2].var()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_filter_prod(self):
        """Test filter then product."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})

        pd_result = pd_df[pd_df['A'] > 1].prod()
        ds_result = ds_df[ds_df['A'] > 1].prod()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTypeCoercionChains:
    """Test type coercion with lazy operation chains."""

    def test_astype_filter_chain(self):
        """Test astype followed by filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['1.5', '2.5', '3.5', '4.5', '5.5']})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': ['1.5', '2.5', '3.5', '4.5', '5.5']})

        pd_df['B'] = pd_df['B'].astype(float)
        pd_result = pd_df[pd_df['B'] > 3.0]

        ds_df['B'] = ds_df['B'].astype(float)
        ds_result = ds_df[ds_df['B'] > 3.0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_astype_chain(self):
        """Test filter followed by astype."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1.5, 2.5, 3.5, 4.5, 5.5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [1.5, 2.5, 3.5, 4.5, 5.5]})

        pd_result = pd_df[pd_df['A'] > 2]
        pd_result = pd_result.astype({'B': int})

        ds_result = ds_df[ds_df['A'] > 2]
        ds_result = ds_result.astype({'B': int})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_groupby_agg(self):
        """Test astype then groupby aggregation."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A'], 'value': ['1', '2', '3', '4', '5']})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B', 'A'], 'value': ['1', '2', '3', '4', '5']})

        pd_df['value'] = pd_df['value'].astype(int)
        pd_result = pd_df.groupby('category')['value'].sum().reset_index()

        ds_df['value'] = ds_df['value'].astype(int)
        ds_result = ds_df.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_astype_int_to_float(self):
        """Test astype from int to float."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df.astype(float)
        ds_result = ds_df.astype(float)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_float_to_int(self):
        """Test astype from float to int (truncation)."""
        pd_df = pd.DataFrame({'A': [1.9, 2.1, 3.5], 'B': [4.2, 5.8, 6.0]})
        ds_df = DataStore({'A': [1.9, 2.1, 3.5], 'B': [4.2, 5.8, 6.0]})

        pd_result = pd_df.astype(int)
        ds_result = ds_df.astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_str_to_int(self):
        """Test astype from string to int."""
        pd_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4', '5', '6']})
        ds_df = DataStore({'A': ['1', '2', '3'], 'B': ['4', '5', '6']})

        pd_result = pd_df.astype(int)
        ds_result = ds_df.astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_bool_conversion(self):
        """Test astype to bool."""
        pd_df = pd.DataFrame({'A': [0, 1, 2, 0, 1], 'B': [1, 0, 0, 1, 1]})
        ds_df = DataStore({'A': [0, 1, 2, 0, 1], 'B': [1, 0, 0, 1, 1]})

        pd_result = pd_df.astype(bool)
        ds_result = ds_df.astype(bool)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStackUnstackOperations:
    """Test stack and unstack operations."""

    def test_stack_basic(self):
        """Test basic stack operation."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df = ds_df.set_index(pd.Index(['x', 'y']))

        pd_result = pd_df.stack()
        ds_result = ds_df.stack()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_stack_dropna_false(self):
        """Test stack with dropna=False."""
        pd_df = pd.DataFrame({'A': [1, np.nan], 'B': [3, 4]}, index=['x', 'y'])
        ds_df = DataStore({'A': [1, np.nan], 'B': [3, 4]})
        ds_df = ds_df.set_index(pd.Index(['x', 'y']))

        pd_result = pd_df.stack(dropna=False)
        ds_result = ds_df.stack(dropna=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_stack_then_reset_index(self):
        """Test stack then reset_index."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df = ds_df.set_index(pd.Index(['x', 'y']))

        pd_result = pd_df.stack().reset_index()
        ds_result = ds_df.stack().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_unstack_basic(self):
        """Test basic unstack operation."""
        index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
        pd_series = pd.Series([10, 20, 30, 40], index=index)
        ds_series = get_series([10, 20, 30, 40])
        ds_series = ds_series.set_axis(index)

        pd_result = pd_series.unstack()
        ds_result = ds_series.unstack()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_unstack_fill_value(self):
        """Test unstack with fill_value."""
        index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)])
        pd_series = pd.Series([10, 20, 30], index=index)
        ds_series = get_series([10, 20, 30])
        ds_series = ds_series.set_axis(index)

        pd_result = pd_series.unstack(fill_value=0)
        ds_result = ds_series.unstack(fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIndexManipulationChains:
    """Test index manipulation with operation chains."""

    def test_set_index_filter_reset(self):
        """Test set_index then filter then reset_index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40], 'C': [100, 200, 300, 400]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40], 'C': [100, 200, 300, 400]})

        pd_result = pd_df.set_index('A')
        pd_result = pd_result[pd_result['B'] > 15]
        pd_result = pd_result.reset_index()

        ds_result = ds_df.set_index('A')
        ds_result = ds_result[ds_result['B'] > 15]
        ds_result = ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_groupby_chain(self):
        """Test set_index then groupby aggregation."""
        pd_df = pd.DataFrame(
            {'category': ['A', 'B', 'A', 'B', 'A'], 'id': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]}
        )
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B', 'A'], 'id': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})

        pd_result = pd_df.set_index('id').groupby('category')['value'].sum().reset_index()
        ds_result = ds_df.set_index('id').groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_reset_index_name_parameter(self):
        """Test reset_index with name parameter."""
        pd_series = pd.Series([1, 2, 3], index=['a', 'b', 'c'], name='values')
        ds_series = get_series([1, 2, 3])
        ds_series = ds_series.set_axis(pd.Index(['a', 'b', 'c']))
        ds_series.name = 'values'

        pd_result = pd_series.reset_index(name='new_values')
        ds_result = ds_series.reset_index(name='new_values')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_multicolumn(self):
        """Test set_index with multiple columns."""
        pd_df = pd.DataFrame({'A': ['x', 'x', 'y', 'y'], 'B': [1, 2, 1, 2], 'C': [10, 20, 30, 40]})
        ds_df = DataStore({'A': ['x', 'x', 'y', 'y'], 'B': [1, 2, 1, 2], 'C': [10, 20, 30, 40]})

        pd_result = pd_df.set_index(['A', 'B'])
        ds_result = ds_df.set_index(['A', 'B'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNumericOperationEdgeCases:
    """Test numeric operations with edge cases."""

    def test_floordiv_basic(self):
        """Test basic floor division."""
        pd_df = pd.DataFrame({'A': [10, 20, 30], 'B': [3, 4, 5]})
        ds_df = DataStore({'A': [10, 20, 30], 'B': [3, 4, 5]})

        pd_result = pd_df['A'] // pd_df['B']
        ds_result = ds_df['A'] // ds_df['B']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floordiv_negative(self):
        """Test floor division with negative values."""
        pd_df = pd.DataFrame({'A': [-10, 20, -30], 'B': [3, -4, 5]})
        ds_df = DataStore({'A': [-10, 20, -30], 'B': [3, -4, 5]})

        pd_result = pd_df['A'] // pd_df['B']
        ds_result = ds_df['A'] // ds_df['B']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floordiv_scalar(self):
        """Test floor division by scalar."""
        pd_df = pd.DataFrame({'A': [10, 20, 30, 40]})
        ds_df = DataStore({'A': [10, 20, 30, 40]})

        pd_result = pd_df['A'] // 3
        ds_result = ds_df['A'] // 3

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_basic(self):
        """Test basic modulo operation."""
        pd_df = pd.DataFrame({'A': [10, 20, 30], 'B': [3, 4, 5]})
        ds_df = DataStore({'A': [10, 20, 30], 'B': [3, 4, 5]})

        pd_result = pd_df['A'] % pd_df['B']
        ds_result = ds_df['A'] % ds_df['B']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_negative(self):
        """Test modulo with negative values."""
        pd_df = pd.DataFrame({'A': [-10, 20, -30], 'B': [3, -4, 5]})
        ds_df = DataStore({'A': [-10, 20, -30], 'B': [3, -4, 5]})

        pd_result = pd_df['A'] % pd_df['B']
        ds_result = ds_df['A'] % ds_df['B']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_scalar(self):
        """Test modulo with scalar."""
        pd_df = pd.DataFrame({'A': [10, 20, 30, 40]})
        ds_df = DataStore({'A': [10, 20, 30, 40]})

        pd_result = pd_df['A'] % 3
        ds_result = ds_df['A'] % 3

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_power_basic(self):
        """Test power operation."""
        pd_df = pd.DataFrame({'A': [2, 3, 4], 'B': [3, 2, 1]})
        ds_df = DataStore({'A': [2, 3, 4], 'B': [3, 2, 1]})

        pd_result = pd_df['A'] ** pd_df['B']
        ds_result = ds_df['A'] ** ds_df['B']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_power_scalar(self):
        """Test power with scalar exponent."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore({'A': [1, 2, 3, 4]})

        pd_result = pd_df['A'] ** 2
        ds_result = ds_df['A'] ** 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_power_fractional(self):
        """Test power with fractional exponent (square root)."""
        pd_df = pd.DataFrame({'A': [1.0, 4.0, 9.0, 16.0]})
        ds_df = DataStore({'A': [1.0, 4.0, 9.0, 16.0]})

        pd_result = pd_df['A'] ** 0.5
        ds_result = ds_df['A'] ** 0.5

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_round_basic(self):
        """Test basic rounding."""
        pd_df = pd.DataFrame({'A': [1.234, 2.567, 3.891]})
        ds_df = DataStore({'A': [1.234, 2.567, 3.891]})

        pd_result = pd_df.round(2)
        ds_result = ds_df.round(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_round_different_columns(self):
        """Test rounding with different decimals per column."""
        pd_df = pd.DataFrame({'A': [1.2345, 2.5678], 'B': [3.14159, 2.71828]})
        ds_df = DataStore({'A': [1.2345, 2.5678], 'B': [3.14159, 2.71828]})

        pd_result = pd_df.round({'A': 2, 'B': 3})
        ds_result = ds_df.round({'A': 2, 'B': 3})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringAccessorAdvanced:
    """Test string accessor with advanced/edge cases."""

    def test_str_strip_whitespace(self):
        """Test stripping various whitespace - chDB trimBoth only strips spaces."""
        pd_df = pd.DataFrame({'text': ['  hello  ', '\tworld\t', '\nhello world\n']})
        ds_df = DataStore({'text': ['  hello  ', '\tworld\t', '\nhello world\n']})

        pd_result = pd_df['text'].str.strip()
        ds_result = ds_df['text'].str.strip()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_regex(self):
        """Test string replace with regex."""
        pd_df = pd.DataFrame({'text': ['abc123', 'def456', 'ghi789']})
        ds_df = DataStore({'text': ['abc123', 'def456', 'ghi789']})

        pd_result = pd_df['text'].str.replace(r'\d+', 'NUM', regex=True)
        ds_result = ds_df['text'].str.replace(r'\d+', 'NUM', regex=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_split_basic(self):
        """Test string split - compare each element's list value."""
        pd_df = pd.DataFrame({'text': ['a,b,c', 'd,e,f', 'g,h,i']})
        ds_df = DataStore({'text': ['a,b,c', 'd,e,f', 'g,h,i']})

        pd_result = pd_df['text'].str.split(',')
        ds_result = ds_df['text'].str.split(',')

        # Compare list values element by element
        # Note: pandas returns list, DataStore may return numpy array
        assert len(ds_result) == len(pd_result)
        for ds_val, pd_val in zip(list(ds_result), list(pd_result)):
            # Convert to list for comparison
            ds_list = list(ds_val) if hasattr(ds_val, '__iter__') else [ds_val]
            pd_list = list(pd_val) if hasattr(pd_val, '__iter__') else [pd_val]
            assert ds_list == pd_list, f"Mismatch: {ds_list} != {pd_list}"

    def test_str_len_basic(self):
        """Test string length."""
        pd_df = pd.DataFrame({'text': ['hello', 'hi', 'goodbye']})
        ds_df = DataStore({'text': ['hello', 'hi', 'goodbye']})

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_case_insensitive(self):
        """Test string contains with case insensitivity."""
        pd_df = pd.DataFrame({'text': ['Hello', 'WORLD', 'hello world']})
        ds_df = DataStore({'text': ['Hello', 'WORLD', 'hello world']})

        pd_result = pd_df['text'].str.contains('hello', case=False)
        ds_result = ds_df['text'].str.contains('hello', case=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_startswith_endswith(self):
        """Test startswith and endswith - chDB returns uint8 instead of bool."""
        pd_df = pd.DataFrame({'text': ['hello world', 'world hello', 'hello hello']})
        ds_df = DataStore({'text': ['hello world', 'world hello', 'hello hello']})

        pd_start = pd_df['text'].str.startswith('hello')
        ds_start = ds_df['text'].str.startswith('hello')

        pd_end = pd_df['text'].str.endswith('hello')
        ds_end = ds_df['text'].str.endswith('hello')

        # chDB returns uint8 for boolean results
        assert_datastore_equals_pandas(ds_start, pd_start, check_nullable_dtype=False)
        assert_datastore_equals_pandas(ds_end, pd_end, check_nullable_dtype=False)

    def test_str_slice_basic(self):
        """Test string slicing."""
        pd_df = pd.DataFrame({'text': ['hello', 'world', 'python']})
        ds_df = DataStore({'text': ['hello', 'world', 'python']})

        pd_result = pd_df['text'].str.slice(0, 3)
        ds_result = ds_df['text'].str.slice(0, 3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_chain_operations(self):
        """Test chaining multiple string operations."""
        pd_df = pd.DataFrame({'text': ['  Hello World  ', '  PYTHON  ', '  pandas  ']})
        ds_df = DataStore({'text': ['  Hello World  ', '  PYTHON  ', '  pandas  ']})

        pd_result = pd_df['text'].str.strip().str.lower()
        ds_result = ds_df['text'].str.strip().str.lower()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDatetimeAccessorEdgeCases:
    """Test datetime accessor with edge cases."""

    def test_dt_year_month_day(self):
        """Test extracting year, month, day."""
        dates = pd.to_datetime(['2023-01-15', '2023-06-20', '2023-12-25'])
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        assert_datastore_equals_pandas(ds_df['date'].dt.year, pd_df['date'].dt.year)
        assert_datastore_equals_pandas(ds_df['date'].dt.month, pd_df['date'].dt.month)
        assert_datastore_equals_pandas(ds_df['date'].dt.day, pd_df['date'].dt.day)

    def test_dt_hour_minute_second(self):
        """Test extracting hour, minute, second."""
        dates = pd.to_datetime(['2023-01-15 10:30:45', '2023-06-20 14:20:30', '2023-12-25 23:59:59'])
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        assert_datastore_equals_pandas(ds_df['date'].dt.hour, pd_df['date'].dt.hour)
        assert_datastore_equals_pandas(ds_df['date'].dt.minute, pd_df['date'].dt.minute)
        assert_datastore_equals_pandas(ds_df['date'].dt.second, pd_df['date'].dt.second)

    def test_dt_dayofweek(self):
        """Test day of week extraction."""
        # Monday=0, Sunday=6
        dates = pd.to_datetime(['2023-01-16', '2023-01-17', '2023-01-22'])  # Mon, Tue, Sun
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        pd_result = pd_df['date'].dt.dayofweek
        ds_result = ds_df['date'].dt.dayofweek

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_quarter(self):
        """Test quarter extraction."""
        dates = pd.to_datetime(['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15'])
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        pd_result = pd_df['date'].dt.quarter
        ds_result = ds_df['date'].dt.quarter

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_month_start_end(self):
        """Test is_month_start and is_month_end."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-15', '2023-01-31'])
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        assert_datastore_equals_pandas(ds_df['date'].dt.is_month_start, pd_df['date'].dt.is_month_start)
        assert_datastore_equals_pandas(ds_df['date'].dt.is_month_end, pd_df['date'].dt.is_month_end)

    def test_dt_date(self):
        """Test extracting date component."""
        dates = pd.to_datetime(['2023-01-15 10:30:45', '2023-06-20 14:20:30'])
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        pd_result = pd_df['date'].dt.date
        ds_result = ds_df['date'].dt.date

        # Convert to string for comparison as date objects may differ
        pd_result_str = pd_result.astype(str)
        ds_result_str = ds_result.astype(str)

        assert_datastore_equals_pandas(ds_result_str, pd_result_str)


class TestClipAndRoundCombinations:
    """Test clip and round operation combinations."""

    def test_clip_then_round(self):
        """Test clip followed by round."""
        pd_df = pd.DataFrame({'A': [1.234, 5.678, 9.012, 0.123]})
        ds_df = DataStore({'A': [1.234, 5.678, 9.012, 0.123]})

        pd_result = pd_df.clip(lower=1.0, upper=8.0).round(1)
        ds_result = ds_df.clip(lower=1.0, upper=8.0).round(1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_round_then_clip(self):
        """Test round followed by clip."""
        pd_df = pd.DataFrame({'A': [1.234, 5.678, 9.012, 0.123]})
        ds_df = DataStore({'A': [1.234, 5.678, 9.012, 0.123]})

        pd_result = pd_df.round(1).clip(lower=1.0, upper=8.0)
        ds_result = ds_df.round(1).clip(lower=1.0, upper=8.0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_lower_only(self):
        """Test clip with only lower bound."""
        pd_df = pd.DataFrame({'A': [1, 5, 9, 0, -5]})
        ds_df = DataStore({'A': [1, 5, 9, 0, -5]})

        pd_result = pd_df.clip(lower=0)
        ds_result = ds_df.clip(lower=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_upper_only(self):
        """Test clip with only upper bound."""
        pd_df = pd.DataFrame({'A': [1, 5, 9, 15, 20]})
        ds_df = DataStore({'A': [1, 5, 9, 15, 20]})

        pd_result = pd_df.clip(upper=10)
        ds_result = ds_df.clip(upper=10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_filter_chain(self):
        """Test clip in a filter chain."""
        pd_df = pd.DataFrame({'A': [1, 5, 9, 15, 20], 'B': [100, 200, 300, 400, 500]})
        ds_df = DataStore({'A': [1, 5, 9, 15, 20], 'B': [100, 200, 300, 400, 500]})

        pd_result = pd_df[pd_df['B'] > 150].clip(lower=5, upper=15)
        ds_result = ds_df[ds_df['B'] > 150].clip(lower=5, upper=15)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCorrelationCovarianceEdgeCases:
    """Test correlation and covariance edge cases."""

    def test_corr_basic(self):
        """Test basic correlation."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]})

        pd_result = pd_df.corr()
        ds_result = ds_df.corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_corr_after_filter(self):
        """Test correlation after filtering."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6], 'B': [2, 4, 6, 8, 10, 12], 'C': [6, 5, 4, 3, 2, 1]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5, 6], 'B': [2, 4, 6, 8, 10, 12], 'C': [6, 5, 4, 3, 2, 1]})

        pd_result = pd_df[pd_df['A'] > 2].corr()
        ds_result = ds_df[ds_df['A'] > 2].corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cov_basic(self):
        """Test basic covariance."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10], 'C': [5, 4, 3, 2, 1]})

        pd_result = pd_df.cov()
        ds_result = ds_df.cov()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cov_ddof(self):
        """Test covariance with ddof parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [2, 4, 6, 8, 10]})

        pd_result = pd_df.cov(ddof=0)
        ds_result = ds_df.cov(ddof=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChains:
    """Test complex operation chains combining multiple operations."""

    def test_filter_astype_groupby_agg_sort(self):
        """Test filter -> astype -> groupby -> agg -> sort chain."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': ['1', '2', '3', '4', '5', '6']})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': ['1', '2', '3', '4', '5', '6']})

        # Filter to category A or values > 2
        pd_df['value'] = pd_df['value'].astype(int)
        pd_result = pd_df[pd_df['value'] > 2].groupby('category')['value'].sum().reset_index().sort_values('value')

        ds_df['value'] = ds_df['value'].astype(int)
        ds_result = ds_df[ds_df['value'] > 2].groupby('category')['value'].sum().reset_index().sort_values('value')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_round_filter_sort(self):
        """Test clip -> round -> filter -> sort chain."""
        pd_df = pd.DataFrame({'A': [1.234, 5.678, 9.012, 3.456, 7.890], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'A': [1.234, 5.678, 9.012, 3.456, 7.890], 'B': [10, 20, 30, 40, 50]})

        pd_result = pd_df.clip(lower=2.0, upper=8.0).round(1)
        pd_result = pd_result[pd_result['A'] > 3.0].sort_values('A')

        ds_result = ds_df.clip(lower=2.0, upper=8.0).round(1)
        ds_result = ds_result[ds_result['A'] > 3.0].sort_values('A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_filter_assign_reset(self):
        """Test set_index -> filter -> assign -> reset_index chain."""
        pd_df = pd.DataFrame({'id': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'id': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})

        pd_result = pd_df.set_index('id')
        pd_result = pd_result[pd_result['value'] > 20]
        pd_result['doubled'] = pd_result['value'] * 2
        pd_result = pd_result.reset_index()

        ds_result = ds_df.set_index('id')
        ds_result = ds_result[ds_result['value'] > 20]
        ds_result['doubled'] = ds_result['value'] * 2
        ds_result = ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_filter_groupby_count(self):
        """Test string operation -> filter -> groupby -> count chain."""
        pd_df = pd.DataFrame(
            {'text': ['hello', 'WORLD', 'Hello', 'world', 'HELLO'], 'category': ['A', 'B', 'A', 'B', 'A']}
        )
        ds_df = DataStore(
            {'text': ['hello', 'WORLD', 'Hello', 'world', 'HELLO'], 'category': ['A', 'B', 'A', 'B', 'A']}
        )

        pd_df['text_lower'] = pd_df['text'].str.lower()
        pd_result = pd_df[pd_df['text_lower'] == 'hello'].groupby('category').size().reset_index(name='count')

        ds_df['text_lower'] = ds_df['text'].str.lower()
        ds_result = ds_df[ds_df['text_lower'] == 'hello'].groupby('category').size().reset_index(name='count')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestEmptyDataFrameEdgeCases:
    """Test operations on empty DataFrames."""

    def test_empty_df_prod(self):
        """Test prod on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        pd_result = pd_df.prod()
        ds_result = ds_df.prod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_var(self):
        """Test var on empty DataFrame."""
        pd_df = pd.DataFrame({'A': pd.Series([], dtype=float), 'B': pd.Series([], dtype=float)})
        ds_df = DataStore({'A': pd.Series([], dtype=float), 'B': pd.Series([], dtype=float)})

        pd_result = pd_df.var()
        ds_result = ds_df.var()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_corr(self):
        """Test corr on empty DataFrame."""
        pd_df = pd.DataFrame({'A': pd.Series([], dtype=float), 'B': pd.Series([], dtype=float)})
        ds_df = DataStore({'A': pd.Series([], dtype=float), 'B': pd.Series([], dtype=float)})

        pd_result = pd_df.corr()
        ds_result = ds_df.corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_to_empty_then_agg(self):
        """Test filtering to empty and then aggregating."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df[pd_df['A'] > 100].sum()
        ds_result = ds_df[ds_df['A'] > 100].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByAggEdgeCases:
    """Test groupby aggregation edge cases."""

    @chdb_no_product_function
    def test_groupby_prod(self):
        """Test groupby product - chDB doesn't have prodIf function."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [2, 3, 4, 5]})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B'], 'value': [2, 3, 4, 5]})

        pd_result = pd_df.groupby('category')['value'].prod().reset_index()
        ds_result = ds_df.groupby('category')['value'].prod().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_var(self):
        """Test groupby variance."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4, 5, 6]})

        pd_result = pd_df.groupby('category')['value'].var().reset_index()
        ds_result = ds_df.groupby('category')['value'].var().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_std(self):
        """Test groupby standard deviation."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4, 5, 6]})

        pd_result = pd_df.groupby('category')['value'].std().reset_index()
        ds_result = ds_df.groupby('category')['value'].std().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_single_group(self):
        """Test groupby with only one group."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'A'], 'value': [1, 2, 3]})
        ds_df = DataStore({'category': ['A', 'A', 'A'], 'value': [1, 2, 3]})

        pd_result = pd_df.groupby('category')['value'].sum().reset_index()
        ds_result = ds_df.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_all_different(self):
        """Test groupby where each row is its own group."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore({'category': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]})

        pd_result = pd_df.groupby('category')['value'].mean().reset_index()
        ds_result = ds_df.groupby('category')['value'].mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
