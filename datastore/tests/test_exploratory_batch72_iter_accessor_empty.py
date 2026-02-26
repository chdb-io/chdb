"""
Exploratory Batch 72: Iterator Protocol, Accessor Chaining, and Empty DataFrame Edge Cases

Focus areas:
1. Iterator protocol edge cases on DataStore and ColumnExpr
2. Accessor method chaining (str.upper().str.len() etc.)
3. Empty DataFrame combined with complex operations
4. Single row edge cases
5. Iterator behavior on filtered/grouped results
6. Multiple consecutive accessors of same type
7. Type coercion through accessor chains

Discovery method: Architecture-based exploration based on analysis of:
- datastore/core.py __iter__ implementation
- datastore/column_expr.py __iter__ implementation
- datastore/accessors/ - accessor implementations
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_datastore_equals_pandas_chdb_compat,
    assert_series_equal,
    get_dataframe,
    get_series
)


class TestIteratorProtocol:
    """Test iterator protocol edge cases"""

    def test_iter_over_datastore_yields_columns(self):
        """Iterating DataStore should yield column names like pandas"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df)

        pd_cols = list(pd_df)
        ds_cols = list(ds_df)

        assert pd_cols == ds_cols

    def test_iter_over_column_expr_yields_values(self):
        """Iterating ColumnExpr should yield values"""
        pd_df = pd.DataFrame({'a': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        pd_values = list(pd_df['a'])
        ds_values = list(ds_df['a'])

        assert pd_values == ds_values

    def test_iter_over_empty_datastore(self):
        """Iterating empty DataStore should yield empty list"""
        pd_df = pd.DataFrame()
        ds_df = DataStore(pd_df)

        pd_cols = list(pd_df)
        ds_cols = list(ds_df)

        assert pd_cols == ds_cols == []

    def test_iter_over_empty_column_expr(self):
        """Iterating empty column should yield empty list"""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df)

        pd_values = list(pd_df['a'])
        ds_values = list(ds_df['a'])

        assert pd_values == ds_values

    def test_iter_after_filter(self):
        """Iterate over DataStore columns after filtering"""
        pd_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['x'] > 1]
        ds_filtered = ds_df[ds_df['x'] > 1]

        pd_cols = list(pd_filtered)
        ds_cols = list(ds_filtered)

        assert pd_cols == ds_cols

    def test_iter_column_after_filter(self):
        """Iterate over column values after filtering"""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['x'] > 2]['x']
        ds_filtered = ds_df[ds_df['x'] > 2]['x']

        pd_values = list(pd_filtered)
        ds_values = list(ds_filtered)

        assert pd_values == ds_values

    def test_iter_preserves_order(self):
        """Iterator should preserve row order"""
        pd_df = pd.DataFrame({'val': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds_df = DataStore(pd_df)

        pd_values = list(pd_df['val'])
        ds_values = list(ds_df['val'])

        assert pd_values == ds_values

    def test_iter_with_nulls(self):
        """Iterator should handle NULL values"""
        pd_df = pd.DataFrame({'val': [1, None, 3, None, 5]})
        ds_df = DataStore(pd_df)

        pd_values = list(pd_df['val'])
        ds_values = list(ds_df['val'])

        # Compare handling NaN/None
        for pv, dv in zip(pd_values, ds_values):
            if pd.isna(pv):
                assert pd.isna(dv)
            else:
                assert pv == dv

    def test_multiple_iterations_same_datastore(self):
        """Multiple iterations over same DataStore should work"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        # First iteration
        first = list(ds_df)
        # Second iteration
        second = list(ds_df)

        assert first == second == ['a', 'b']

    def test_multiple_iterations_same_column(self):
        """Multiple iterations over same column should work"""
        pd_df = pd.DataFrame({'x': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        col = ds_df['x']
        first = list(col)
        second = list(col)

        assert first == second == [10, 20, 30]


class TestAccessorChaining:
    """Test accessor method chaining"""

    def test_str_upper_then_len(self):
        """Chained string operations: upper then len"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['name'].str.upper().str.len()
        ds_result = ds_df['name'].str.upper().str.len()

        pd_result = pd_result.reset_index(drop=True)
        ds_result = get_series(ds_result).reset_index(drop=True)

        assert_series_equal(ds_result, pd_result, check_names=False, check_dtype=False)

    def test_str_lower_then_replace(self):
        """Chained string operations: lower then replace"""
        pd_df = pd.DataFrame({'text': ['HELLO', 'WORLD', 'TEST']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.lower().str.replace('l', 'X')
        ds_result = ds_df['text'].str.lower().str.replace('l', 'X')

        pd_result = pd_result.reset_index(drop=True)
        ds_result = get_series(ds_result).reset_index(drop=True)

        assert_series_equal(ds_result, pd_result, check_names=False, check_dtype=False)

    def test_str_strip_then_upper(self):
        """Chained string operations: strip then upper"""
        pd_df = pd.DataFrame({'s': ['  hello  ', ' world ', '  test']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['s'].str.strip().str.upper()
        ds_result = ds_df['s'].str.strip().str.upper()

        pd_result = pd_result.reset_index(drop=True)
        ds_result = get_series(ds_result).reset_index(drop=True)

        assert_series_equal(ds_result, pd_result, check_names=False, check_dtype=False)

    def test_str_slice_then_upper(self):
        """Chained string operations: slice then upper"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['name'].str[:3].str.upper()
        ds_result = ds_df['name'].str[:3].str.upper()

        pd_result = pd_result.reset_index(drop=True)
        ds_result = get_series(ds_result).reset_index(drop=True)

        assert_series_equal(ds_result, pd_result, check_names=False, check_dtype=False)

    def test_str_contains_chain(self):
        """String contains followed by filtering"""
        pd_df = pd.DataFrame({'text': ['apple', 'banana', 'apricot', 'cherry']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['text'].str.startswith('a')]['text'].str.upper()
        ds_result = ds_df[ds_df['text'].str.startswith('a')]['text'].str.upper()

        pd_result = pd_result.reset_index(drop=True)
        ds_result = get_series(ds_result).reset_index(drop=True)

        assert_series_equal(ds_result, pd_result, check_names=False, check_dtype=False)


class TestEmptyDataFrameEdgeCases:
    """Test empty DataFrame combined with operations"""

    def test_empty_df_filter(self):
        """Filter on empty DataFrame"""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 0].reset_index(drop=True)
        ds_result = ds_df[ds_df['a'] > 0].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_assign(self):
        """Assign column on empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd_df['a'] * 2)
        ds_result = ds_df.assign(b=ds_df['a'] * 2)

        pd_result = pd_result.reset_index(drop=True)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_groupby(self):
        """GroupBy on empty DataFrame"""
        pd_df = pd.DataFrame({'cat': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('cat')['val'].sum().reset_index()
        ds_result = ds_df.groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_sort(self):
        """Sort empty DataFrame"""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a').reset_index(drop=True)
        ds_result = ds_df.sort_values('a').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_head_tail(self):
        """Head and tail on empty DataFrame"""
        pd_df = pd.DataFrame({'x': []})
        ds_df = DataStore(pd_df)

        pd_head = pd_df.head(5).reset_index(drop=True)
        ds_head = ds_df.head(5).reset_index(drop=True)
        assert_datastore_equals_pandas(ds_head, pd_head)

        pd_tail = pd_df.tail(5).reset_index(drop=True)
        ds_tail = ds_df.tail(5).reset_index(drop=True)
        assert_datastore_equals_pandas(ds_tail, pd_tail)

    def test_filter_to_empty_then_operations(self):
        """Filter to empty result then apply operations"""
        pd_df = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        # Filter to empty
        pd_empty = pd_df[pd_df['val'] > 100]
        ds_empty = ds_df[ds_df['val'] > 100]

        # Apply operations on empty result
        pd_result = pd_empty.assign(doubled=pd_empty['val'] * 2).reset_index(drop=True)
        ds_result = ds_empty.assign(doubled=ds_empty['val'] * 2).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSingleRowEdgeCases:
    """Test single row edge cases"""

    def test_single_row_filter(self):
        """Filter on single row DataFrame"""
        pd_df = pd.DataFrame({'a': [42], 'b': ['test']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 0].reset_index(drop=True)
        ds_result = ds_df[ds_df['a'] > 0].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_groupby(self):
        """GroupBy on single row"""
        pd_df = pd.DataFrame({'cat': ['A'], 'val': [100]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('cat')['val'].sum().reset_index()
        ds_result = ds_df.groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_sort(self):
        """Sort single row DataFrame"""
        pd_df = pd.DataFrame({'x': [1], 'y': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('x').reset_index(drop=True)
        ds_result = ds_df.sort_values('x').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_head_tail(self):
        """Head and tail on single row"""
        pd_df = pd.DataFrame({'val': [99]})
        ds_df = DataStore(pd_df)

        # head(1) should return the row
        pd_h1 = pd_df.head(1).reset_index(drop=True)
        ds_h1 = ds_df.head(1).reset_index(drop=True)
        assert_datastore_equals_pandas(ds_h1, pd_h1)

        # head(5) should return the row (only 1 exists)
        pd_h5 = pd_df.head(5).reset_index(drop=True)
        ds_h5 = ds_df.head(5).reset_index(drop=True)
        assert_datastore_equals_pandas(ds_h5, pd_h5)

        # tail(1) should return the row
        pd_t1 = pd_df.tail(1).reset_index(drop=True)
        ds_t1 = ds_df.tail(1).reset_index(drop=True)
        assert_datastore_equals_pandas(ds_t1, pd_t1)

    def test_single_row_assign(self):
        """Assign on single row DataFrame"""
        pd_df = pd.DataFrame({'a': [10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd_df['a'] * 3, c=pd_df['a'] + 5)
        ds_result = ds_df.assign(b=ds_df['a'] * 3, c=ds_df['a'] + 5)

        pd_result = pd_result.reset_index(drop=True)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAllNullsEdgeCases:
    """Test DataFrames with all NULL values"""

    def test_all_null_column_sum(self):
        """Sum of all-NULL column"""
        pd_df = pd.DataFrame({'val': [None, None, None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['val'].sum()
        ds_result = ds_df['val'].sum()

        # Use float() to trigger execution and extract scalar value naturally
        ds_val = float(ds_result)

        # pandas returns 0.0 for all-null sum, chDB may return NULL
        # Both 0 and NaN/None are acceptable behaviors
        if pd.isna(pd_result):
            assert pd.isna(ds_val) or ds_val == 0
        else:
            assert ds_val == pd_result or pd.isna(ds_val) or ds_val == 0

    def test_all_null_column_mean(self):
        """Mean of all-NULL column"""
        pd_df = pd.DataFrame({'val': [None, None, None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['val'].mean()
        ds_result = ds_df['val'].mean()

        # Use float() to trigger execution and extract scalar value naturally
        ds_val = float(ds_result)

        # Both should be NaN for all-null mean
        assert pd.isna(pd_result)
        assert pd.isna(ds_val)

    def test_all_null_filter(self):
        """Filter on all-NULL column"""
        pd_df = pd.DataFrame({'a': [None, None, None], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        # This filter should return empty result (NULL > 0 is NULL, not True)
        pd_result = pd_df[pd_df['a'] > 0].reset_index(drop=True)
        ds_result = ds_df[ds_df['a'] > 0].reset_index(drop=True)

        # Both should be empty
        assert len(get_dataframe(ds_result)) == len(pd_result)

    def test_all_null_groupby(self):
        """GroupBy with all NULL values in grouping column"""
        pd_df = pd.DataFrame({'cat': [None, None, None], 'val': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        # GroupBy with dropna=True (default) should give empty result
        pd_result = pd_df.groupby('cat', dropna=True)['val'].sum().reset_index()
        ds_result = ds_df.groupby('cat', dropna=True)['val'].sum().reset_index()

        # Both should be empty with dropna=True
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMixedTypeEdgeCases:
    """Test edge cases with mixed types"""

    def test_numeric_string_column_in_filter(self):
        """Filter with numeric column compared to numeric value"""
        pd_df = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['val'] > 2.5].reset_index(drop=True)
        ds_result = ds_df[ds_df['val'] > 2.5].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_int_float_column_operations(self):
        """Operations between int and float columns"""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(sum_col=pd_df['int_col'] + pd_df['float_col'])
        ds_result = ds_df.assign(sum_col=ds_df['int_col'] + ds_df['float_col'])

        pd_result = pd_result.reset_index(drop=True)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_int_column_divide_result_type(self):
        """Division of int column should produce float"""
        pd_df = pd.DataFrame({'val': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(divided=pd_df['val'] / 4)
        ds_result = ds_df.assign(divided=ds_df['val'] / 4)

        pd_result = pd_result.reset_index(drop=True)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnOperationChains:
    """Test complex column operation chains"""

    def test_arithmetic_chain(self):
        """Chain of arithmetic operations"""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(y=((pd_df['x'] + 10) * 2 - 5) / 3)
        ds_result = ds_df.assign(y=((ds_df['x'] + 10) * 2 - 5) / 3)

        pd_result = pd_result.reset_index(drop=True)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_chain(self):
        """Chain of comparison operations combined with filter"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df)

        # Complex condition
        pd_cond = (pd_df['a'] > 1) & (pd_df['b'] < 5) & ((pd_df['a'] + pd_df['b']) == 5)
        ds_cond = (ds_df['a'] > 1) & (ds_df['b'] < 5) & ((ds_df['a'] + ds_df['b']) == 5)

        pd_result = pd_df[pd_cond].reset_index(drop=True)
        ds_result = ds_df[ds_cond].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_column_assign_chain(self):
        """Multiple column assignments in chain"""
        pd_df = pd.DataFrame({'base': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        # Chain of assignments
        pd_result = pd_df.copy()
        pd_result['a'] = pd_result['base'] + 1
        pd_result['b'] = pd_result['a'] * 2
        pd_result['c'] = pd_result['b'] - pd_result['base']
        pd_result = pd_result.reset_index(drop=True)

        ds_result = ds_df.copy()
        ds_result['a'] = ds_result['base'] + 1
        ds_result['b'] = ds_result['a'] * 2
        ds_result['c'] = ds_result['b'] - ds_result['base']
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFilterThenGroupBy:
    """Test filter followed by groupby patterns"""

    def test_filter_then_groupby_sum(self):
        """Filter then groupby with sum"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['value'] > 20].groupby('category')['value'].sum().reset_index()
        ds_result = ds_df[ds_df['value'] > 20].groupby('category')['value'].sum().reset_index()

        # Sort for comparison (groupby order may vary)
        pd_result = pd_result.sort_values('category').reset_index(drop=True)
        ds_result = get_dataframe(ds_result).sort_values('category').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_filters_then_groupby(self):
        """Multiple filters then groupby"""
        pd_df = pd.DataFrame({
            'cat': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'val': [1, 2, 3, 4, 5, 6],
            'flag': [True, False, True, True, False, True]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[(pd_df['val'] > 1) & (pd_df['flag'] == True)].groupby('cat')['val'].mean().reset_index()
        ds_result = ds_df[(ds_df['val'] > 1) & (ds_df['flag'] == True)].groupby('cat')['val'].mean().reset_index()

        pd_result = pd_result.sort_values('cat').reset_index(drop=True)
        ds_result = get_dataframe(ds_result).sort_values('cat').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_to_single_group(self):
        """Filter that results in single group"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'B', 'C'],
            'val': [100, 50, 30]
        })
        ds_df = DataStore(pd_df)

        # Filter to only category A
        pd_result = pd_df[pd_df['val'] > 60].groupby('cat')['val'].sum().reset_index()
        ds_result = ds_df[ds_df['val'] > 60].groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestHeadTailEdgeCases:
    """Test head and tail edge cases"""

    def test_head_larger_than_dataframe(self):
        """Head with n larger than DataFrame size"""
        pd_df = pd.DataFrame({'x': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(100).reset_index(drop=True)
        ds_result = ds_df.head(100).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_larger_than_dataframe(self):
        """Tail with n larger than DataFrame size"""
        pd_df = pd.DataFrame({'x': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(100).reset_index(drop=True)
        ds_result = ds_df.tail(100).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """Head with n=0"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(0).reset_index(drop=True)
        ds_result = ds_df.head(0).reset_index(drop=True)

        assert len(get_dataframe(ds_result)) == len(pd_result) == 0

    def test_tail_zero(self):
        """Tail with n=0"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(0).reset_index(drop=True)
        ds_result = ds_df.tail(0).reset_index(drop=True)

        assert len(get_dataframe(ds_result)) == len(pd_result) == 0

    def test_head_after_filter(self):
        """Head after filter operation"""
        pd_df = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['val'] > 3].head(3).reset_index(drop=True)
        ds_result = ds_df[ds_df['val'] > 3].head(3).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_after_sort(self):
        """Tail after sort operation"""
        pd_df = pd.DataFrame({'val': [5, 2, 8, 1, 9, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('val').tail(3).reset_index(drop=True)
        ds_result = ds_df.sort_values('val').tail(3).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnReferenceAfterOperations:
    """Test column reference consistency after various operations"""

    def test_column_ref_after_rename(self):
        """Column reference after rename operation"""
        pd_df = pd.DataFrame({'old_name': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(columns={'old_name': 'new_name'})
        ds_result = ds_df.rename(columns={'old_name': 'new_name'})

        # Access renamed column
        pd_vals = list(pd_result['new_name'])
        ds_vals = list(ds_result['new_name'])

        assert pd_vals == ds_vals

    def test_column_ref_after_assign(self):
        """Column reference after assign operation"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd_df['a'] * 2)
        ds_result = ds_df.assign(b=ds_df['a'] * 2)

        # Access both original and new column
        assert list(pd_result['a']) == list(ds_result['a'])
        assert list(pd_result['b']) == list(ds_result['b'])

    def test_column_ref_after_filter(self):
        """Column reference after filter operation"""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['x'] > 2]
        ds_filtered = ds_df[ds_df['x'] > 2]

        # Access columns after filter
        pd_x = list(pd_filtered['x'])
        ds_x = list(ds_filtered['x'])
        pd_y = list(pd_filtered['y'])
        ds_y = list(ds_filtered['y'])

        assert pd_x == ds_x
        assert pd_y == ds_y


class TestDuplicateColumnHandling:
    """Test handling of duplicate column scenarios"""

    @pytest.mark.xfail(reason="DataStore renames duplicate columns to unique names (a, a_1) while pandas keeps both as 'a'")
    def test_select_same_column_twice(self):
        """Selecting same column multiple times - pandas keeps duplicate names, DataStore renames them"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        # This creates duplicate columns in pandas
        pd_result = pd_df[['a', 'a']]
        ds_result = ds_df[['a', 'a']]

        pd_result = pd_result.reset_index(drop=True)
        ds_df_result = get_dataframe(ds_result).reset_index(drop=True)

        # pandas has ['a', 'a'], DataStore has ['a', 'a_1']
        # This is a known difference in behavior
        assert list(pd_result.columns) == list(ds_df_result.columns)
        assert len(pd_result.columns) == 2

    def test_assign_existing_column_name(self):
        """Assign to existing column name (overwrite)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(a=pd_df['a'] * 10)
        ds_result = ds_df.assign(a=ds_df['a'] * 10)

        pd_result = pd_result.reset_index(drop=True)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)
