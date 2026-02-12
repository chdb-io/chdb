"""
Exploratory Batch 90: Indexing, Complex Chains, and Edge Cases
Date: 2026-01-16

Focus areas:
- iloc/loc indexing edge cases
- String accessor edge cases
- Datetime accessor edge cases
- Complex chain operations
- DataFrame copy semantics
- Column operations
- Iteration protocol
- Empty data edge cases
- Mixed type operations
- Method chaining
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestIlocEdgeCases:
    """Test iloc indexing edge cases."""

    def test_iloc_negative_index(self):
        """Test iloc with negative indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.iloc[-1]
        ds_result = ds_df.iloc[-1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative_slice(self):
        """Test iloc with negative slice."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.iloc[-3:-1]
        ds_result = ds_df.iloc[-3:-1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_step_slice(self):
        """Test iloc with step in slice."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [10, 20, 30, 40, 50, 60]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5, 6], 'b': [10, 20, 30, 40, 50, 60]})

        pd_result = pd_df.iloc[::2]
        ds_result = ds_df.iloc[::2]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_reverse_slice(self):
        """Test iloc with reverse slice."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.iloc[::-1]
        ds_result = ds_df.iloc[::-1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_row_and_column(self):
        """Test iloc with row and column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        pd_result = pd_df.iloc[0:2, 1:3]
        ds_result = ds_df.iloc[0:2, 1:3]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_single_row_returns_series(self):
        """Test that iloc single row returns Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.iloc[1]
        ds_result = ds_df.iloc[1]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestLocEdgeCases:
    """Test loc indexing edge cases."""

    def test_loc_with_boolean_series(self):
        """Test loc with boolean series filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.loc[pd_df['a'] > 2]
        ds_result = ds_df.loc[ds_df['a'] > 2]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_column_selection(self):
        """Test loc with column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        pd_result = pd_df.loc[:, ['a', 'c']]
        ds_result = ds_df.loc[:, ['a', 'c']]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_filter_and_column(self):
        """Test loc with filter and column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40], 'c': [100, 200, 300, 400]})
        ds_df = DataStore({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40], 'c': [100, 200, 300, 400]})

        pd_result = pd_df.loc[pd_df['a'] > 1, ['b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 1, ['b', 'c']]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringAccessorEdgeCases:
    """Test string accessor edge cases."""

    def test_str_contains_case_insensitive(self):
        """Test str.contains with case insensitive."""
        pd_df = pd.DataFrame({'text': ['Hello', 'WORLD', 'hello', 'World']})
        ds_df = DataStore({'text': ['Hello', 'WORLD', 'hello', 'World']})

        pd_result = pd_df[pd_df['text'].str.contains('hello', case=False)]
        ds_result = ds_df[ds_df['text'].str.contains('hello', case=False)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_strip_whitespace(self):
        """Test str.strip with whitespace."""
        pd_df = pd.DataFrame({'text': ['  hello  ', '  world', 'test  ']})
        ds_df = DataStore({'text': ['  hello  ', '  world', 'test  ']})

        pd_result = pd_df['text'].str.strip()
        ds_result = ds_df['text'].str.strip()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_split_expand(self):
        """Test str.split with expand."""
        pd_df = pd.DataFrame({'text': ['a,b,c', 'd,e,f', 'g,h,i']})
        ds_df = DataStore({'text': ['a,b,c', 'd,e,f', 'g,h,i']})

        pd_result = pd_df['text'].str.split(',', expand=True)
        ds_result = ds_df['text'].str.split(',', expand=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_regex(self):
        """Test str.replace with regex."""
        pd_df = pd.DataFrame({'text': ['abc123', 'def456', 'ghi789']})
        ds_df = DataStore({'text': ['abc123', 'def456', 'ghi789']})

        pd_result = pd_df['text'].str.replace(r'\d+', 'X', regex=True)
        ds_result = ds_df['text'].str.replace(r'\d+', 'X', regex=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len(self):
        """Test str.len."""
        pd_df = pd.DataFrame({'text': ['a', 'bb', 'ccc', 'dddd']})
        ds_df = DataStore({'text': ['a', 'bb', 'ccc', 'dddd']})

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDatetimeAccessorEdgeCases:
    """Test datetime accessor edge cases."""

    def test_dt_year_month_day(self):
        """Test dt accessor for year, month, day."""
        dates = pd.date_range('2023-01-15', periods=5, freq='M')
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        pd_result = pd_df['date'].dt.year
        ds_result = ds_df['date'].dt.year
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_dayofweek(self):
        """Test dt.dayofweek accessor."""
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        pd_result = pd_df['date'].dt.dayofweek
        ds_result = ds_df['date'].dt.dayofweek
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_strftime(self):
        """Test dt.strftime."""
        dates = pd.date_range('2023-01-15', periods=3, freq='D')
        pd_df = pd.DataFrame({'date': dates})
        ds_df = DataStore({'date': dates})

        pd_result = pd_df['date'].dt.strftime('%Y-%m-%d')
        ds_result = ds_df['date'].dt.strftime('%Y-%m-%d')
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChainOperations:
    """Test complex chain operations."""

    def test_filter_groupby_agg_sort(self):
        """Test filter -> groupby -> agg -> sort chain."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60],
            'flag': [True, True, False, True, True, False]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60],
            'flag': [True, True, False, True, True, False]
        })

        pd_result = pd_df[pd_df['flag']].groupby('category')['value'].sum().sort_values(ascending=False)
        ds_result = ds_df[ds_df['flag']].groupby('category')['value'].sum().sort_values(ascending=False)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_column_operations(self):
        """Test multiple column operations in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        ds_df = DataStore({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})

        pd_df = pd_df.copy()
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_df['d'] = pd_df['c'] * 2
        pd_result = pd_df[pd_df['d'] > 50]

        ds_df_copy = ds_df.copy()
        ds_df_copy['c'] = ds_df_copy['a'] + ds_df_copy['b']
        ds_df_copy['d'] = ds_df_copy['c'] * 2
        ds_result = ds_df_copy[ds_df_copy['d'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_select(self):
        """Test assign -> filter -> select chain."""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})

        pd_result = pd_df.assign(z=lambda d: d['x'] + d['y']).query('z > 5')[['x', 'z']]
        ds_result = ds_df.assign(z=lambda d: d['x'] + d['y']).query('z > 5')[['x', 'z']]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataFrameCopySemantics:
    """Test DataFrame copy semantics."""

    def test_copy_deep(self):
        """Test deep copy independence."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_copy = pd_df.copy(deep=True)
        ds_copy = ds_df.copy(deep=True)

        pd_copy['a'] = [10, 20, 30]
        ds_copy['a'] = [10, 20, 30]

        # Original should be unchanged
        assert_datastore_equals_pandas(ds_df, pd_df)
        # Copy should have new values
        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_default(self):
        """Test default copy behavior."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        assert_datastore_equals_pandas(ds_copy, pd_copy)


class TestColumnRenameReorder:
    """Test column rename and reorder operations."""

    def test_rename_columns(self):
        """Test column rename."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y'})
        ds_result = ds_df.rename(columns={'a': 'x', 'b': 'y'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """Test column reordering via reindex."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        pd_result = pd_df.reindex(columns=['c', 'a', 'b'])
        ds_result = ds_df.reindex(columns=['c', 'a', 'b'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_columns_order(self):
        """Test column selection preserves order."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        pd_result = pd_df[['c', 'a']]
        ds_result = ds_df[['c', 'a']]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIterationProtocol:
    """Test iteration protocol."""

    def test_iterrows(self):
        """Test iterrows iteration."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_rows = list(pd_df.iterrows())
        ds_rows = list(ds_df.iterrows())

        assert len(pd_rows) == len(ds_rows)
        for (pd_idx, pd_row), (ds_idx, ds_row) in zip(pd_rows, ds_rows):
            assert pd_idx == ds_idx
            assert_datastore_equals_pandas(ds_row, pd_row)

    def test_itertuples(self):
        """Test itertuples iteration."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_tuples = list(pd_df.itertuples())
        ds_tuples = list(ds_df.itertuples())

        assert len(pd_tuples) == len(ds_tuples)
        for pd_tup, ds_tup in zip(pd_tuples, ds_tuples):
            assert pd_tup == ds_tup

    def test_items(self):
        """Test items iteration over columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())

        assert len(pd_items) == len(ds_items)
        for (pd_name, pd_col), (ds_name, ds_col) in zip(pd_items, ds_items):
            assert pd_name == ds_name
            assert_datastore_equals_pandas(ds_col, pd_col)


class TestEdgeCasesWithEmptyData:
    """Test edge cases with empty data."""

    def test_empty_dataframe_columns(self):
        """Test empty DataFrame has correct columns."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})

        assert list(ds_df.columns) == list(pd_df.columns)
        assert len(ds_df) == len(pd_df)

    def test_empty_after_filter(self):
        """Test empty result after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert len(ds_result) == len(pd_result)
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_empty_groupby(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int), 'b': pd.Series([], dtype=int)})
        ds_df = DataStore({'a': [], 'b': []})

        pd_result = pd_df.groupby('a')['b'].sum()
        ds_result = ds_df.groupby('a')['b'].sum()

        assert len(ds_result) == len(pd_result)

    def test_single_row_dataframe(self):
        """Test operations on single row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore({'a': [1], 'b': [2]})

        pd_result = pd_df['a'] + pd_df['b']
        ds_result = ds_df['a'] + ds_df['b']
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMixedTypeOperations:
    """Test operations with mixed types."""

    def test_int_float_arithmetic(self):
        """Test arithmetic with int and float columns."""
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})
        ds_df = DataStore({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})

        pd_result = pd_df['int_col'] + pd_df['float_col']
        ds_result = ds_df['int_col'] + ds_df['float_col']
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_numeric_concat(self):
        """Test DataFrame with mixed string and numeric columns."""
        pd_df = pd.DataFrame({'name': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        ds_df = DataStore({'name': ['a', 'b', 'c'], 'value': [1, 2, 3]})

        pd_result = pd_df[pd_df['value'] > 1]
        ds_result = ds_df[ds_df['value'] > 1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_column_operations(self):
        """Test operations with boolean column."""
        pd_df = pd.DataFrame({'flag': [True, False, True], 'value': [1, 2, 3]})
        ds_df = DataStore({'flag': [True, False, True], 'value': [1, 2, 3]})

        pd_result = pd_df[pd_df['flag']]
        ds_result = ds_df[ds_df['flag']]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMethodChaining:
    """Test pandas-style method chaining."""

    def test_filter_sort_head(self):
        """Test filter -> sort -> head chain."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2], 'b': [50, 30, 10, 40, 20]})
        ds_df = DataStore({'a': [5, 3, 1, 4, 2], 'b': [50, 30, 10, 40, 20]})

        pd_result = pd_df[pd_df['a'] > 1].sort_values('a').head(3)
        ds_result = ds_df[ds_df['a'] > 1].sort_values('a').head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_drop_rename(self):
        """Test assign -> drop -> rename chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        pd_result = pd_df.assign(d=lambda x: x['a'] * 2).drop(columns=['c']).rename(columns={'a': 'A'})
        ds_result = ds_df.assign(d=lambda x: x['a'] * 2).drop(columns=['c']).rename(columns={'a': 'A'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_reset_index(self):
        """Test groupby -> agg -> reset_index chain."""
        pd_df = pd.DataFrame({'cat': ['A', 'B', 'A', 'B'], 'val': [1, 2, 3, 4]})
        ds_df = DataStore({'cat': ['A', 'B', 'A', 'B'], 'val': [1, 2, 3, 4]})

        pd_result = pd_df.groupby('cat')['val'].sum().reset_index()
        ds_result = ds_df.groupby('cat')['val'].sum().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNumericOperations:
    """Test numeric operations."""

    def test_abs(self):
        """Test abs operation."""
        pd_df = pd.DataFrame({'a': [-1, -2, 3, -4, 5]})
        ds_df = DataStore({'a': [-1, -2, 3, -4, 5]})

        pd_result = pd_df['a'].abs()
        ds_result = ds_df['a'].abs()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_round(self):
        """Test round operation."""
        pd_df = pd.DataFrame({'a': [1.234, 2.567, 3.891]})
        ds_df = DataStore({'a': [1.234, 2.567, 3.891]})

        pd_result = pd_df['a'].round(2)
        ds_result = ds_df['a'].round(2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip(self):
        """Test clip operation."""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})

        pd_result = pd_df['a'].clip(lower=5, upper=15)
        ds_result = ds_df['a'].clip(lower=5, upper=15)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between(self):
        """Test between operation."""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})

        pd_result = pd_df['a'].between(5, 15)
        ds_result = ds_df['a'].between(5, 15)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggregationEdgeCases:
    """Test aggregation edge cases."""

    def test_agg_with_multiple_funcs(self):
        """Test agg with multiple functions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].agg(['sum', 'mean', 'min', 'max'])
        ds_result = ds_df['a'].agg(['sum', 'mean', 'min', 'max'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe(self):
        """Test describe statistics."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'x', 'x', 'y', 'z']})
        ds_df = DataStore({'a': ['x', 'y', 'x', 'x', 'y', 'z']})

        pd_result = pd_df['a'].value_counts(normalize=True)
        ds_result = ds_df['a'].value_counts(normalize=True)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanOperations:
    """Test boolean operations."""

    def test_any_all(self):
        """Test any and all operations."""
        pd_df = pd.DataFrame({'a': [True, True, False], 'b': [True, True, True]})
        ds_df = DataStore({'a': [True, True, False], 'b': [True, True, True]})

        assert ds_df['a'].any() == pd_df['a'].any()
        assert ds_df['a'].all() == pd_df['a'].all()
        assert ds_df['b'].any() == pd_df['b'].any()
        assert ds_df['b'].all() == pd_df['b'].all()

    def test_isin(self):
        """Test isin operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].isin([2, 4])
        ds_result = ds_df['a'].isin([2, 4])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notna_isna(self):
        """Test notna and isna operations."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore({'a': [1, None, 3, None, 5]})

        pd_notna = pd_df['a'].notna()
        ds_notna = ds_df['a'].notna()
        assert_datastore_equals_pandas(ds_notna, pd_notna)

        pd_isna = pd_df['a'].isna()
        ds_isna = ds_df['a'].isna()
        assert_datastore_equals_pandas(ds_isna, pd_isna)


class TestSelectDtypes:
    """Test select_dtypes functionality."""

    def test_select_numeric(self):
        """Test select_dtypes for numeric columns."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        pd_result = pd_df.select_dtypes(include=['number'])
        ds_result = ds_df.select_dtypes(include=['number'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_exclude(self):
        """Test select_dtypes with exclude."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        pd_result = pd_df.select_dtypes(exclude=['object'])
        ds_result = ds_df.select_dtypes(exclude=['object'])
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
