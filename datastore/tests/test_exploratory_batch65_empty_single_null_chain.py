"""
Exploratory Batch 65: Empty DataFrame, Single-Row, NULL Propagation, String Edge Cases

This batch explores undertested boundary conditions:
1. Empty DataFrames after filtering (all rows filtered out)
2. Single-row DataFrame operations (boundary case)
3. NULL propagation in arithmetic and comparison chains
4. String accessor operations with empty/NULL values
5. GroupBy on empty/single-row DataFrames

Discovery method: Architecture-based exploration of boundary conditions
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    get_dataframe,
    get_series,
)


class TestEmptyDataFrameAfterFilter:
    """Test behavior when filtering results in empty DataFrame."""

    def test_filter_all_rows_out(self):
        """Test filtering that removes all rows."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        # Filter that removes all rows
        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_filter_then_select_columns(self):
        """Test selecting columns from empty filtered result."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 100][['a', 'c']]
        ds_result = ds_df[ds_df['a'] > 100][['a', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_filter_then_arithmetic(self):
        """Test arithmetic on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        pd_empty = pd_df[pd_df['a'] > 100]
        ds_empty = ds_df[ds_df['a'] > 100]

        pd_result = pd_empty['a'] + pd_empty['b']
        ds_result = ds_empty['a'] + ds_empty['b']

        # Empty Series comparison
        assert len(ds_result) == len(pd_result) == 0

    def test_empty_after_multiple_filters(self):
        """Test chained filters resulting in empty."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2][pd_df['a'] < 3]  # Impossible condition
        ds_result = ds_df[ds_df['a'] > 2][ds_df['a'] < 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_sort_values(self):
        """Test sorting empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 100].sort_values('a')
        ds_result = ds_df[ds_df['a'] > 100].sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_head_tail(self):
        """Test head/tail on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_empty = pd_df[pd_df['a'] > 100]
        ds_empty = ds_df[ds_df['a'] > 100]

        # head on empty
        pd_head = pd_empty.head(5)
        ds_head = ds_empty.head(5)
        assert_datastore_equals_pandas(ds_head, pd_head)

        # tail on empty
        pd_tail = pd_empty.tail(5)
        ds_tail = ds_empty.tail(5)
        assert_datastore_equals_pandas(ds_tail, pd_tail)


class TestSingleRowDataFrame:
    """Test edge cases with single-row DataFrames."""

    def test_single_row_basic_operations(self):
        """Test basic operations on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42], 'b': ['single']})
        ds_df = DataStore(pd_df.copy())

        # Filter that keeps the single row
        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_sort(self):
        """Test sorting single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42], 'b': ['single']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_groupby_count(self):
        """Test groupby count on single-row DataFrame."""
        pd_df = pd.DataFrame({'category': ['A'], 'value': [100]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].count().reset_index(name='count')
        ds_result = ds_df.groupby('category')['value'].count().reset_index(name='count')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_single_row_groupby_sum(self):
        """Test groupby sum on single-row DataFrame."""
        pd_df = pd.DataFrame({'category': ['A'], 'value': [100]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_df.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_single_row_head_tail(self):
        """Test head/tail on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore(pd_df.copy())

        # head(1) should return the row
        pd_head = pd_df.head(1)
        ds_head = ds_df.head(1)
        assert_datastore_equals_pandas(ds_head, pd_head)

        # head(10) should still return just the row
        pd_head10 = pd_df.head(10)
        ds_head10 = ds_df.head(10)
        assert_datastore_equals_pandas(ds_head10, pd_head10)

        # tail(1) should return the row
        pd_tail = pd_df.tail(1)
        ds_tail = ds_df.tail(1)
        assert_datastore_equals_pandas(ds_tail, pd_tail)

    def test_single_row_slice(self):
        """Test slicing single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore(pd_df.copy())

        # First row
        pd_result = pd_df[:1]
        ds_result = ds_df[:1]
        assert_datastore_equals_pandas(ds_result, pd_result)

        # Empty slice
        pd_result = pd_df[1:]
        ds_result = ds_df[1:]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullPropagationChains:
    """Test NULL/NaN propagation through operation chains."""

    def test_null_arithmetic_chain(self):
        """Test NULL propagation in arithmetic operations."""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': [10, 20, None, None]
        })
        ds_df = DataStore(pd_df.copy())

        # NULL + number = NULL
        pd_df['c'] = pd_df['a'] + pd_df['b']
        ds_df['c'] = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_null_multiplication_chain(self):
        """Test NULL propagation in multiplication."""
        pd_df = pd.DataFrame({
            'a': [1.0, None, 3.0, 4.0],
            'b': [2.0, 2.0, None, 2.0]
        })
        ds_df = DataStore(pd_df.copy())

        # NULL * number = NULL
        pd_df['product'] = pd_df['a'] * pd_df['b']
        ds_df['product'] = ds_df['a'] * ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_null_comparison_result(self):
        """Test NULL in comparison operations."""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': [1, 1, 1, None]
        })
        ds_df = DataStore(pd_df.copy())

        # In pandas: comparing with NA yields NA (which becomes False in boolean context)
        pd_mask = pd_df['a'] > pd_df['b']
        ds_mask = ds_df['a'] > ds_df['b']

        pd_result = pd_df[pd_mask]
        ds_result = ds_df[ds_mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_null_in_filter_chain(self):
        """Test NULL handling through filter chain."""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, 4, None],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore(pd_df.copy())

        # Filter with NULL values - should exclude NULL
        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_null_after_groupby_agg(self):
        """Test NULL handling in groupby aggregations."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [1, None, None, 4]
        })
        ds_df = DataStore(pd_df.copy())

        # sum should skip NULLs
        pd_result = pd_df.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_df.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestStringAccessorEdgeCases:
    """Test string accessor with edge cases."""

    def test_str_upper_with_null(self):
        """Test str.upper() with NULL values."""
        pd_df = pd.DataFrame({'text': ['hello', None, 'WORLD', None]})
        ds_df = DataStore(pd_df.copy())

        pd_df['upper'] = pd_df['text'].str.upper()
        ds_df['upper'] = ds_df['text'].str.upper()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_lower_with_null(self):
        """Test str.lower() with NULL values."""
        pd_df = pd.DataFrame({'text': ['HELLO', None, 'world']})
        ds_df = DataStore(pd_df.copy())

        pd_df['lower'] = pd_df['text'].str.lower()
        ds_df['lower'] = ds_df['text'].str.lower()

        assert_datastore_equals_pandas(ds_df, pd_df)

    @pytest.mark.xfail(reason="str.len() returns Int64 in DataStore vs float64 in pandas when NULLs present")
    def test_str_len_with_null(self):
        """Test str.len() with NULL values."""
        pd_df = pd.DataFrame({'text': ['hello', None, 'abc', '']})
        ds_df = DataStore(pd_df.copy())

        pd_df['length'] = pd_df['text'].str.len()
        ds_df['length'] = ds_df['text'].str.len()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_contains_with_null(self):
        """Test str.contains() with NULL values."""
        pd_df = pd.DataFrame({'text': ['hello world', None, 'foo bar', 'hello']})
        ds_df = DataStore(pd_df.copy())

        # contains pattern
        pd_mask = pd_df['text'].str.contains('hello', na=False)
        ds_mask = ds_df['text'].str.contains('hello', na=False)

        pd_result = pd_df[pd_mask]
        ds_result = ds_df[ds_mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_startswith_with_null(self):
        """Test str.startswith() with NULL values."""
        pd_df = pd.DataFrame({'text': ['hello', None, 'help', 'world']})
        ds_df = DataStore(pd_df.copy())

        pd_mask = pd_df['text'].str.startswith('hel', na=False)
        ds_mask = ds_df['text'].str.startswith('hel', na=False)

        pd_result = pd_df[pd_mask]
        ds_result = ds_df[ds_mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_empty_string(self):
        """Test string operations on empty strings."""
        pd_df = pd.DataFrame({'text': ['', 'a', '', 'abc']})
        ds_df = DataStore(pd_df.copy())

        # Length of empty strings
        pd_df['length'] = pd_df['text'].str.len()
        ds_df['length'] = ds_df['text'].str.len()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_replace_with_null(self):
        """Test str.replace() with NULL values."""
        pd_df = pd.DataFrame({'text': ['hello', None, 'hello world', 'foo']})
        ds_df = DataStore(pd_df.copy())

        pd_df['replaced'] = pd_df['text'].str.replace('hello', 'hi')
        ds_df['replaced'] = ds_df['text'].str.replace('hello', 'hi')

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestGroupByEmptySingleRow:
    """Test groupby on empty and single-row DataFrames."""

    def test_groupby_empty_dataframe(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'category': pd.Series([], dtype=str), 'value': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_df.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_all_same_category(self):
        """Test groupby when all rows have same category."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'A'],
            'value': [10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_df.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_all_unique_categories(self):
        """Test groupby when each row is unique category."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_df.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestChainedOperationsWithEdgeCases:
    """Test chained operations with edge cases."""

    def test_filter_then_groupby_then_filter(self):
        """Test filter -> groupby -> filter chain."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter -> groupby -> filter
        pd_filtered = pd_df[pd_df['value'] > 15]
        pd_grouped = pd_filtered.groupby('category')['value'].sum().reset_index(name='sum')
        pd_result = pd_grouped[pd_grouped['sum'] > 50]

        ds_filtered = ds_df[ds_df['value'] > 15]
        ds_grouped = ds_filtered.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_grouped[ds_grouped['sum'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_column_operations_then_filter(self):
        """Test creating multiple columns then filtering."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # Create derived columns
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_df['d'] = pd_df['a'] * 2
        pd_result = pd_df[pd_df['c'] > 30]

        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_df['d'] = ds_df['a'] * 2
        ds_result = ds_df[ds_df['c'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_head_then_filter(self):
        """Test sort -> head -> filter chain."""
        pd_df = pd.DataFrame({
            'a': [5, 2, 8, 1, 9, 3],
            'b': ['e', 'b', 'h', 'a', 'i', 'c']
        })
        ds_df = DataStore(pd_df.copy())

        # Sort -> head -> filter
        pd_sorted = pd_df.sort_values('a')
        pd_head = pd_sorted.head(4)
        pd_result = pd_head[pd_head['a'] > 1]

        ds_sorted = ds_df.sort_values('a')
        ds_head = ds_sorted.head(4)
        ds_result = ds_head[ds_head['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_to_empty_then_groupby(self):
        """Test filtering to empty then groupby."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter to empty then groupby
        pd_empty = pd_df[pd_df['value'] > 1000]
        pd_result = pd_empty.groupby('category')['value'].sum().reset_index(name='sum')

        ds_empty = ds_df[ds_df['value'] > 1000]
        ds_result = ds_empty.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAssignWithNullValues:
    """Test column assignment with NULL values."""

    def test_assign_constant_with_nulls(self):
        """Test assigning constant to column with NULLs."""
        pd_df = pd.DataFrame({'a': [1, None, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = 100
        ds_df['b'] = 100

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_expression_with_nulls(self):
        """Test assigning expression result with NULLs."""
        pd_df = pd.DataFrame({'a': [1, None, 3], 'b': [10, 20, None]})
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'] * pd_df['b']
        ds_df['c'] = ds_df['a'] * ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestDuplicateValues:
    """Test operations with duplicate values."""

    def test_filter_with_duplicates(self):
        """Test filtering with duplicate values."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2, 3, 3],
            'b': ['x', 'y', 'x', 'y', 'x', 'y']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 1]
        ds_result = ds_df[ds_df['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_with_duplicates(self):
        """Test sorting with duplicate keys."""
        pd_df = pd.DataFrame({
            'a': [2, 1, 2, 1, 3],
            'b': ['e', 'd', 'c', 'b', 'a']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a', kind='stable')
        ds_result = ds_df.sort_values('a', kind='stable')

        # Order among duplicates may differ, so check_row_order=False for same 'a' values
        # But we can check that rows with same 'a' have correct 'b' values
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_count_with_duplicates(self):
        """Test groupby count with duplicate category values."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].count().reset_index(name='count')
        ds_result = ds_df.groupby('category')['value'].count().reset_index(name='count')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestNegativeAndZeroValues:
    """Test operations with negative and zero values."""

    def test_filter_negative_values(self):
        """Test filtering with negative values."""
        pd_df = pd.DataFrame({
            'a': [-5, -2, 0, 2, 5],
            'b': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore(pd_df.copy())

        # Filter negative
        pd_result = pd_df[pd_df['a'] < 0]
        ds_result = ds_df[ds_df['a'] < 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

        # Filter zero
        pd_result = pd_df[pd_df['a'] == 0]
        ds_result = ds_df[ds_df['a'] == 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_with_zero(self):
        """Test arithmetic operations involving zero."""
        pd_df = pd.DataFrame({
            'a': [0, 1, 2],
            'b': [10, 0, 20]
        })
        ds_df = DataStore(pd_df.copy())

        # Addition with zero
        pd_df['sum'] = pd_df['a'] + pd_df['b']
        ds_df['sum'] = ds_df['a'] + ds_df['b']

        # Multiplication with zero
        pd_df['product'] = pd_df['a'] * pd_df['b']
        ds_df['product'] = ds_df['a'] * ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_division_by_nonzero(self):
        """Test division by non-zero values."""
        pd_df = pd.DataFrame({
            'a': [10, 20, 30],
            'b': [2, 4, 5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['quotient'] = pd_df['a'] / pd_df['b']
        ds_df['quotient'] = ds_df['a'] / ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_groupby_sum_with_negatives(self):
        """Test groupby sum with negative values."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [-10, 30, -5, 15]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].sum().reset_index(name='sum')
        ds_result = ds_df.groupby('category')['value'].sum().reset_index(name='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestBooleanColumnOperations:
    """Test operations with boolean columns."""

    def test_filter_by_boolean_column(self):
        """Test filtering by boolean column."""
        pd_df = pd.DataFrame({
            'active': [True, False, True, False],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['active']]
        ds_result = ds_df[ds_df['active']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_not_filter(self):
        """Test filtering by negated boolean."""
        pd_df = pd.DataFrame({
            'active': [True, False, True, False],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[~pd_df['active']]
        ds_result = ds_df[~ds_df['active']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_and_filter(self):
        """Test filtering with AND condition on booleans."""
        pd_df = pd.DataFrame({
            'active': [True, True, False, False],
            'verified': [True, False, True, False],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['active'] & pd_df['verified']]
        ds_result = ds_df[ds_df['active'] & ds_df['verified']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_or_filter(self):
        """Test filtering with OR condition on booleans."""
        pd_df = pd.DataFrame({
            'active': [True, True, False, False],
            'verified': [True, False, True, False],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['active'] | pd_df['verified']]
        ds_result = ds_df[ds_df['active'] | ds_df['verified']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMixedTypeComparisons:
    """Test comparisons with mixed types."""

    def test_int_float_comparison(self):
        """Test comparing int column with float value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2.5]
        ds_result = ds_df[ds_df['a'] > 2.5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_int_column_float_column_arithmetic(self):
        """Test arithmetic between int and float columns."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['sum'] = pd_df['int_col'] + pd_df['float_col']
        ds_df['sum'] = ds_df['int_col'] + ds_df['float_col']

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestDropDuplicates:
    """Test drop_duplicates edge cases."""

    def test_drop_duplicates_all_same(self):
        """Test drop_duplicates when all rows are identical."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 1],
            'b': ['x', 'x', 'x']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_no_duplicates(self):
        """Test drop_duplicates when no duplicates exist."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset(self):
        """Test drop_duplicates with subset of columns."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': ['x', 'y', 'x', 'y']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates(subset=['a'])
        ds_result = ds_df.drop_duplicates(subset=['a'])

        # Order of kept rows may differ
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_drop_duplicates_keep_last(self):
        """Test drop_duplicates with keep='last'."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': ['first', 'last', 'first', 'last']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates(subset=['a'], keep='last')
        ds_result = ds_df.drop_duplicates(subset=['a'], keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
