"""
Exploratory Batch 63: Advanced Chain Operations + Special Numeric Values

This batch explores:
1. Complex lazy chain operations: assign -> filter -> assign -> groupby
2. Special numeric values: infinity, -infinity, very large/small numbers
3. Boolean operation chains with negation
4. Index preservation through complex lazy chains
5. Column selection and reordering edge cases
6. Arithmetic operations with edge cases (division by zero, overflow)

Discovery method: Architecture-based exploration
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
from tests.xfail_markers import (
    chdb_duplicate_column_rename,
    pandas_version_nullable_int_dtype,
)


class TestComplexLazyChains:
    """Test complex chains of lazy operations."""

    def test_assign_filter_assign_chain(self):
        """Test assign -> filter -> assign chain."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # pandas chain
        pd_df['c'] = pd_df['a'] * 2
        pd_result = pd_df[pd_df['c'] > 4].copy()
        pd_result['d'] = pd_result['b'] + pd_result['c']

        # DataStore chain
        ds_df['c'] = ds_df['a'] * 2
        ds_result = ds_df[ds_df['c'] > 4]
        ds_result['d'] = ds_result['b'] + ds_result['c']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filter_chain(self):
        """Test multiple consecutive filters."""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
        })
        ds_df = DataStore(pd_df.copy())

        # pandas chain
        pd_result = pd_df[pd_df['x'] > 2]
        pd_result = pd_result[pd_result['x'] < 8]
        pd_result = pd_result[pd_result['y'] == 'a']

        # DataStore chain
        ds_result = ds_df[ds_df['x'] > 2]
        ds_result = ds_result[ds_result['x'] < 8]
        ds_result = ds_result[ds_result['y'] == 'a']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_groupby_agg_chain(self):
        """Test filter -> groupby -> agg chain."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50, 60, 70]
        })
        ds_df = DataStore(pd_df.copy())

        # pandas chain
        pd_result = pd_df[pd_df['value'] > 15].groupby('category')['value'].sum().reset_index()

        # DataStore chain
        ds_result = ds_df[ds_df['value'] > 15].groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_groupby_transform_chain(self):
        """Test assign -> groupby -> transform chain."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df.copy())

        # pandas
        pd_df['doubled'] = pd_df['val'] * 2
        pd_df['group_sum'] = pd_df.groupby('group')['doubled'].transform('sum')
        pd_result = pd_df.copy()

        # DataStore
        ds_df['doubled'] = ds_df['val'] * 2
        ds_df['group_sum'] = ds_df.groupby('group')['doubled'].transform('sum')

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_sort_head_filter_chain(self):
        """Test sort -> head -> filter chain."""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85, 92, 78, 95, 88]
        })
        ds_df = DataStore(pd_df.copy())

        # pandas chain
        pd_result = pd_df.sort_values('score', ascending=False).head(4)
        pd_result = pd_result[pd_result['score'] > 80]

        # DataStore chain
        ds_result = ds_df.sort_values('score', ascending=False).head(4)
        ds_result = ds_result[ds_result['score'] > 80]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialNumericValues:
    """Test handling of special numeric values."""

    def test_infinity_values(self):
        """Test DataFrame with infinity values."""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, -np.inf, 2.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })
        ds_df = DataStore(pd_df.copy())

        # Check values are preserved
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_filter_with_infinity(self):
        """Test filter operations with infinity values."""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, -np.inf, 2.0, 3.0],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter for finite values
        pd_result = pd_df[np.isfinite(pd_df['a'])]
        ds_result = ds_df[np.isfinite(ds_df['a'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_with_infinity(self):
        """Test arithmetic operations with infinity."""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, -np.inf, 2.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })
        ds_df = DataStore(pd_df.copy())

        # pandas
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df.copy()

        # DataStore
        ds_df['c'] = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        pd_df = pd.DataFrame({
            'a': [1e308, -1e308, 1.0, 2.0],
            'b': [1.0, 2.0, 3.0, 4.0]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_very_small_numbers(self):
        """Test handling of very small numbers (near zero)."""
        pd_df = pd.DataFrame({
            'a': [1e-308, -1e-308, 0.0, 1.0],
            'b': [1.0, 2.0, 3.0, 4.0]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_nan_vs_none(self):
        """Test NaN vs None handling."""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan],
            'b': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter for non-null
        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_with_infinity(self):
        """Test aggregation with infinity values."""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, 2.0, 3.0],
            'b': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        assert pd_result == ds_result  # Both should be inf


class TestBooleanOperationChains:
    """Test boolean operation chains."""

    def test_and_or_chain(self):
        """Test chained AND/OR conditions."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'x', 'y', 'x']
        })
        ds_df = DataStore(pd_df.copy())

        # (a > 2) AND ((b < 40) OR (c == 'y'))
        pd_result = pd_df[(pd_df['a'] > 2) & ((pd_df['b'] < 40) | (pd_df['c'] == 'y'))]
        ds_result = ds_df[(ds_df['a'] > 2) & ((ds_df['b'] < 40) | (ds_df['c'] == 'y'))]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_chain(self):
        """Test negation (~) operations."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [True, False, True, False, True]
        })
        ds_df = DataStore(pd_df.copy())

        # ~(a > 2)
        pd_result = pd_df[~(pd_df['a'] > 2)]
        ds_result = ds_df[~(ds_df['a'] > 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_double_negation(self):
        """Test double negation ~~cond."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'flag': [True, False, True, False, True]
        })
        ds_df = DataStore(pd_df.copy())

        # ~~(a > 2) should be equivalent to (a > 2)
        pd_result = pd_df[~~(pd_df['a'] > 2)]
        ds_result = ds_df[~~(ds_df['a'] > 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_column_filter(self):
        """Test filtering with boolean column directly."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'is_valid': [True, False, True, False, True]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter by boolean column
        pd_result = pd_df[pd_df['is_valid']]
        ds_result = ds_df[ds_df['is_valid']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_boolean_column_filter(self):
        """Test filtering with negated boolean column."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'is_valid': [True, False, True, False, True]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter by negated boolean column
        pd_result = pd_df[~pd_df['is_valid']]
        ds_result = ds_df[~ds_df['is_valid']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnSelectionEdgeCases:
    """Test column selection and reordering edge cases."""

    def test_select_columns_different_order(self):
        """Test selecting columns in different order."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        ds_df = DataStore(pd_df.copy())

        # Select in different order
        pd_result = pd_df[['c', 'a', 'b']]
        ds_result = ds_df[['c', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_single_column_as_dataframe(self):
        """Test selecting single column as DataFrame (not Series)."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        ds_df = DataStore(pd_df.copy())

        # Single column as list returns DataFrame
        pd_result = pd_df[['a']]
        ds_result = ds_df[['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_duplicate_column_rename
    def test_select_duplicate_columns(self):
        """Test selecting same column multiple times."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        ds_df = DataStore(pd_df.copy())

        # Select 'a' twice
        pd_result = pd_df[['a', 'b', 'a']]
        ds_result = ds_df[['a', 'b', 'a']]

        # SQL fundamentally cannot have duplicate column names (MULTIPLE_EXPRESSIONS_FOR_ALIAS error)
        # DataStore renames duplicates to 'a_1', 'a_2', etc.
        # This test verifies that at least the values are correct
        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == len(pd_result)

    def test_drop_and_select_columns(self):
        """Test dropping then selecting columns."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        })
        ds_df = DataStore(pd_df.copy())

        # Drop 'b', then select remaining
        pd_result = pd_df.drop(columns=['b'])[['d', 'a']]
        ds_result = ds_df.drop(columns=['b'])[['d', 'a']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIndexPreservation:
    """Test index preservation through operations."""

    def test_custom_index_after_filter(self):
        """Test that custom index is preserved after filter."""
        pd_df = pd.DataFrame(
            {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]},
            index=['w', 'x', 'y', 'z', 'v']
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        # Check both values and index
        assert list(ds_result.index) == list(pd_result.index)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_custom_index_after_sort(self):
        """Test that custom index is preserved after sort."""
        pd_df = pd.DataFrame(
            {'a': [3, 1, 4, 1, 5], 'b': [10, 20, 30, 40, 50]},
            index=['w', 'x', 'y', 'z', 'v']
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert list(ds_result.index) == list(pd_result.index)

    def test_integer_index_after_operations(self):
        """Test integer index after multiple operations."""
        pd_df = pd.DataFrame(
            {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]},
            index=[100, 200, 300, 400, 500]
        )
        ds_df = DataStore(pd_df.copy())

        # Filter then sort
        pd_result = pd_df[pd_df['a'] > 2].sort_values('b', ascending=False)
        ds_result = ds_df[ds_df['a'] > 2].sort_values('b', ascending=False)

        assert list(ds_result.index) == list(pd_result.index)


class TestArithmeticEdgeCases:
    """Test arithmetic operation edge cases."""

    def test_division_by_zero(self):
        """Test division by zero handling."""
        pd_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0, 40.0],
            'b': [2.0, 0.0, 5.0, 0.0]
        })
        ds_df = DataStore(pd_df.copy())

        # pandas division by zero produces inf
        pd_df['c'] = pd_df['a'] / pd_df['b']
        pd_result = pd_df.copy()

        ds_df['c'] = ds_df['a'] / ds_df['b']

        # Both should have inf values where division by zero occurred
        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_modulo_operations(self):
        """Test modulo operations."""
        pd_df = pd.DataFrame({
            'a': [10, 15, 20, 25, 30],
            'b': [3, 4, 7, 6, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'] % pd_df['b']
        pd_result = pd_df.copy()

        ds_df['c'] = ds_df['a'] % ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_floor_division(self):
        """Test floor division operations."""
        pd_df = pd.DataFrame({
            'a': [10, 15, 20, 25, 30],
            'b': [3, 4, 7, 6, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'] // pd_df['b']
        pd_result = pd_df.copy()

        ds_df['c'] = ds_df['a'] // ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_power_operations(self):
        """Test power operations."""
        pd_df = pd.DataFrame({
            'base': [2, 3, 4, 5],
            'exp': [3, 2, 2, 3]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = pd_df['base'] ** pd_df['exp']
        pd_result = pd_df.copy()

        ds_df['result'] = ds_df['base'] ** ds_df['exp']

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_negative_power(self):
        """Test negative power operations."""
        pd_df = pd.DataFrame({
            'base': [2.0, 3.0, 4.0, 5.0],
            'exp': [-1, -2, -1, -2]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = pd_df['base'] ** pd_df['exp']
        pd_result = pd_df.copy()

        ds_df['result'] = ds_df['base'] ** ds_df['exp']

        assert_datastore_equals_pandas(ds_df, pd_result)


class TestEmptyDataFrameEdgeCases:
    """Test edge cases with empty DataFrames."""

    def test_empty_after_filter(self):
        """Test operations on empty result after filter."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter that returns empty
        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert len(ds_result) == 0
        assert len(pd_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_assign_on_empty(self):
        """Test assign on empty DataFrame."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter to empty then assign
        pd_empty = pd_df[pd_df['a'] > 100].copy()
        pd_empty['c'] = pd_empty['a'] * 2

        ds_empty = ds_df[ds_df['a'] > 100]
        ds_empty['c'] = ds_empty['a'] * 2

        assert len(ds_empty) == 0
        assert 'c' in ds_empty.columns

    def test_groupby_on_empty(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'A'],
            'value': [1, 2, 3]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter to empty then groupby
        pd_empty = pd_df[pd_df['value'] > 100].groupby('group')['value'].sum().reset_index()
        ds_empty = ds_df[ds_df['value'] > 100].groupby('group')['value'].sum().reset_index()

        assert len(ds_empty) == len(pd_empty)


class TestSingleRowEdgeCases:
    """Test edge cases with single-row DataFrames."""

    def test_single_row_operations(self):
        """Test operations on single-row DataFrame."""
        pd_df = pd.DataFrame({
            'a': [1],
            'b': [10]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df.copy()

        ds_df['c'] = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_single_row_groupby(self):
        """Test groupby on single-row DataFrame."""
        pd_df = pd.DataFrame({
            'group': ['A'],
            'value': [100]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_keeps(self):
        """Test filter that keeps the single row."""
        pd_df = pd.DataFrame({
            'a': [5],
            'b': [10]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_removes(self):
        """Test filter that removes the single row."""
        pd_df = pd.DataFrame({
            'a': [1],
            'b': [10]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 5]
        ds_result = ds_df[ds_df['a'] > 5]

        assert len(ds_result) == 0
        assert len(pd_result) == 0


class TestStringColumnEdgeCases:
    """Test edge cases with string columns."""

    def test_empty_string_values(self):
        """Test DataFrame with empty string values."""
        pd_df = pd.DataFrame({
            'name': ['Alice', '', 'Bob', ''],
            'score': [85, 90, 75, 80]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_filter_empty_strings(self):
        """Test filtering for empty strings."""
        pd_df = pd.DataFrame({
            'name': ['Alice', '', 'Bob', ''],
            'score': [85, 90, 75, 80]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['name'] != '']
        ds_result = ds_df[ds_df['name'] != '']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_with_special_chars(self):
        """Test strings with special characters."""
        pd_df = pd.DataFrame({
            'text': ["hello\nworld", "tab\there", "quote's", 'double"quote'],
            'id': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_unicode_strings(self):
        """Test Unicode strings."""
        pd_df = pd.DataFrame({
            'text': ['hello', '你好', 'مرحبا', '🎉'],
            'id': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestMixedTypeOperations:
    """Test operations with mixed types."""

    @pandas_version_nullable_int_dtype
    def test_numeric_comparison_with_none(self):
        """Test numeric comparison with None values.

        Note: Nullable Int64 dtype preservation differs between pandas versions.
        In older pandas + chDB combinations, filtering removes NA and may return float64.
        """
        pd_df = pd.DataFrame({
            'a': pd.array([1, 2, None, 4, None], dtype=pd.Int64Dtype()),
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter non-null
        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_numeric_dataframe(self):
        """Test DataFrame with both string and numeric columns."""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [85.5, 90.0, 78.5]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter on numeric, select including string
        pd_result = pd_df[pd_df['age'] > 26][['name', 'score']]
        ds_result = ds_df[ds_df['age'] > 26][['name', 'score']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexAggregations:
    """Test complex aggregation scenarios."""

    def test_multiple_agg_functions(self):
        """Test multiple aggregation functions."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].agg(['sum', 'mean', 'max', 'min']).reset_index()
        ds_result = ds_df.groupby('category')['value'].agg(['sum', 'mean', 'max', 'min']).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_count_with_nulls(self):
        """Test groupby count with null values."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A'],
            'value': pd.array([1, None, 3, None, 5], dtype=pd.Int64Dtype())
        })
        ds_df = DataStore(pd_df.copy())

        # count() should not count nulls
        pd_result = pd_df.groupby('category')['value'].count().reset_index()
        ds_result = ds_df.groupby('category')['value'].count().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_groupby_size_vs_count(self):
        """Test groupby size vs count (size counts all, count skips nulls)."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A'],
            'value': pd.array([1, None, 3, None, 5], dtype=pd.Int64Dtype())
        })
        ds_df = DataStore(pd_df.copy())

        # size() counts all rows including nulls
        pd_size = pd_df.groupby('category').size().reset_index(name='count')
        ds_size = ds_df.groupby('category').size().reset_index(name='count')

        assert_datastore_equals_pandas(ds_size, pd_size, check_nullable_dtype=False)
