"""
Exploratory Batch 25 - Boolean Indexing and Edge Cases

Tests focus on:
1. Boolean list indexing - index preservation
2. Boolean pandas.Series indexing
3. Chained accessor operations
4. GroupBy named aggregation
5. Various edge cases

This batch discovered and fixed:
- Boolean list indexing was resetting index (should preserve original indices)
- pandas.Series as boolean mask was not supported
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal, get_series, assert_datastore_equals_pandas


class TestBooleanListIndexing:
    """Test boolean list indexing with index preservation."""

    def test_boolean_list_preserves_index(self):
        """Boolean list indexing should preserve original indices."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        mask = [True, False, True, False, True]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_boolean_list_all_true(self):
        """All True mask returns all rows."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        mask = [True, True, True]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_boolean_list_all_false(self):
        """All False mask returns empty DataFrame with correct columns."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        mask = [False, False, False]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_boolean_list_first_only(self):
        """Keep only first row."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        mask = [True, False, False]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_boolean_list_last_only(self):
        """Keep only last row."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        mask = [False, False, True]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)


class TestBooleanSeriesIndexing:
    """Test pandas.Series as boolean mask."""

    def test_boolean_series_basic(self):
        """Basic boolean Series indexing."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        mask = pd.Series([True, False, True, False, True])
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_boolean_series_with_multiple_columns(self):
        """Boolean Series indexing with multiple columns."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [1.1, 2.2, 3.3]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [1.1, 2.2, 3.3]})

        mask = pd.Series([False, True, True])
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_boolean_series_chained_filter(self):
        """Boolean Series followed by another operation."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

        mask = pd.Series([True, False, True, False, True])
        pd_result = pd_df[mask][['A']]
        ds_result = ds_df[mask][['A']]

        assert_frame_equal(ds_result._get_df(), pd_result)


class TestChainedAccessorOperations:
    """Test chained accessor operations."""

    def test_str_accessor_triple_chain(self):
        """Multiple chained string operations."""
        # Mirror pattern: both directly get result from chained operations
        pd_df = pd.DataFrame({'text': ['  HELLO world  ', 'FOO bar', '  test  ']})
        ds_df = DataStore({'text': ['  HELLO world  ', 'FOO bar', '  test  ']})

        pd_result = pd_df['text'].str.strip().str.lower().str.replace('o', '0')
        ds_result = ds_df['text'].str.strip().str.lower().str.replace('o', '0')

        # Execute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_dt_accessor_with_filter(self):
        """DateTime accessor after filter."""
        # Mirror pattern: both directly get result from accessor
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        pd_df = pd.DataFrame({'date': dates, 'value': range(10)})
        ds_df = DataStore({'date': dates, 'value': range(10)})

        # Filter then extract
        pd_filtered = pd_df[pd_df['date'].dt.day > 5]
        ds_filtered = ds_df[ds_df['date'].dt.day > 5]

        pd_result = pd_filtered['date'].dt.day
        ds_result = ds_filtered['date'].dt.day

        # Execute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))


class TestGroupByNamedAggregation:
    """Test groupby with named aggregation."""

    def test_named_agg_with_index(self):
        """Named aggregation with group key as index."""
        pd_df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4], 'C': [10, 20, 30, 40]})
        ds_df = DataStore({'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4], 'C': [10, 20, 30, 40]})

        pd_result = pd_df.groupby('A').agg(B_sum=('B', 'sum'), C_mean=('C', 'mean'))
        ds_result = ds_df.groupby('A').agg(B_sum=('B', 'sum'), C_mean=('C', 'mean'))

        assert_frame_equal(ds_result._get_df().sort_index(), pd_result.sort_index())

    def test_named_agg_with_as_index_false(self):
        """Named aggregation with as_index=False."""
        pd_df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4], 'C': [10, 20, 30, 40]})
        ds_df = DataStore({'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4], 'C': [10, 20, 30, 40]})

        pd_result = pd_df.groupby('A', as_index=False).agg(B_sum=('B', 'sum'), C_mean=('C', 'mean'))
        ds_result = ds_df.groupby('A', as_index=False).agg(B_sum=('B', 'sum'), C_mean=('C', 'mean'))

        assert_frame_equal(
            ds_result._get_df().sort_values('A').reset_index(drop=True),
            pd_result.sort_values('A').reset_index(drop=True),
        )


class TestSeriesOperationsViaAssign:
    """Test Series operations via assign."""

    def test_series_clip(self):
        """Series.clip operation."""
        # Mirror pattern: both directly get result from clip
        pd_df = pd.DataFrame({'A': [-5, 0, 5, 10, 15]})
        ds_df = DataStore({'A': [-5, 0, 5, 10, 15]})

        pd_result = pd_df['A'].clip(lower=0, upper=10)
        ds_result = ds_df['A'].clip(lower=0, upper=10)

        # Execute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_series_between(self):
        """Series.between operation."""
        # Mirror pattern: both directly get result from between
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df['A'].between(2, 4)
        ds_result = ds_df['A'].between(2, 4)

        # Direct comparison - assert_series_equal auto-handles lazy objects
        assert_series_equal(ds_result, pd_result)

    def test_series_map_with_dict(self):
        """Series.map with dict."""
        # Mirror pattern: both directly get result from map
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c']})
        ds_df = DataStore({'A': ['a', 'b', 'c']})

        mapping = {'a': 1, 'b': 2, 'c': 3}
        pd_result = pd_df['A'].map(mapping)
        ds_result = ds_df['A'].map(mapping)

        # Execute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))


class TestReplaceOperations:
    """Test replace operations."""

    def test_replace_with_regex_false(self):
        """replace with regex=False."""
        pd_df = pd.DataFrame({'A': ['foo', 'bar', 'baz']})
        ds_df = DataStore({'A': ['foo', 'bar', 'baz']})

        pd_result = pd_df.replace('foo', 'FOO', regex=False)
        ds_result = ds_df.replace('foo', 'FOO', regex=False)

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_replace_with_list_of_values(self):
        """replace with list of values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df.replace([1, 3, 5], [10, 30, 50])
        ds_result = ds_df.replace([1, 3, 5], [10, 30, 50])

        assert_frame_equal(ds_result._get_df(), pd_result)


class TestWindowFunctions:
    """Test window functions."""

    def test_rolling_returns_pandas(self):
        """rolling operations return pandas Series."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df['A'].rolling(window=3, min_periods=1).sum()
        ds_result = ds_df['A'].rolling(window=3, min_periods=1).sum()

        assert isinstance(ds_result, pd.Series)
        assert_series_equal(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_expanding_returns_pandas(self):
        """expanding operations return pandas Series."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df['A'].expanding(min_periods=2).mean()
        ds_result = ds_df['A'].expanding(min_periods=2).mean()

        assert isinstance(ds_result, pd.Series)
        assert_series_equal(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))


class TestArithmeticOperations:
    """Test DataFrame arithmetic operations."""

    def test_dataframe_plus_scalar(self):
        """DataFrame + scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df + 10
        ds_result = ds_df + 10

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_dataframe_times_dataframe(self):
        """DataFrame * DataFrame."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [2, 2, 2], 'B': [3, 3, 3]})
        ds_df1 = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df2 = DataStore({'A': [2, 2, 2], 'B': [3, 3, 3]})

        pd_result = pd_df1 * pd_df2
        ds_result = ds_df1 * ds_df2

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_unary_negation(self):
        """Unary negation."""
        pd_df = pd.DataFrame({'A': [1, -2, 3], 'B': [-4, 5, -6]})
        ds_df = DataStore({'A': [1, -2, 3], 'B': [-4, 5, -6]})

        pd_result = -pd_df
        ds_result = -ds_df

        assert_frame_equal(ds_result._get_df(), pd_result)


class TestAdditionalEdgeCases:
    """Test additional edge cases."""

    def test_assign_lambda_cross_reference(self):
        """assign with lambda referencing another column."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [10, 20, 30]})

        pd_result = pd_df.assign(C=lambda x: x['A'] + x['B'])
        ds_result = ds_df.assign(C=lambda x: x['A'] + x['B'])

        assert_frame_equal(ds_result._get_df(), pd_result)

    def test_column_selection_then_sum(self):
        """Multiple column selection then sum."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

        pd_result = pd_df[['A', 'B']].sum()
        ds_result = ds_df[['A', 'B']].sum()

        assert isinstance(ds_result, pd.Series)
        assert_series_equal(ds_result, pd_result)

    def test_empty_dataframe_sum(self):
        """Empty DataFrame sum."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        pd_result = pd_df.sum()
        ds_result = ds_df.sum()

        assert isinstance(ds_result, pd.Series)
        assert_series_equal(ds_result, pd_result)


class TestCallableIndexing:
    """Test callable (lambda function) indexing."""

    def test_callable_basic_filter(self):
        """Basic callable indexing with simple condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[lambda x: x['a'] > 2]
        ds_result = ds_df[lambda x: x['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_compound_condition(self):
        """Callable with compound conditions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        pd_result = pd_df[lambda x: (x['a'] > 2) & (x['b'] < 4)]
        ds_result = ds_df[lambda x: (x['a'] > 2) & (x['b'] < 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_or_condition(self):
        """Callable with OR condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[lambda x: (x['a'] < 2) | (x['a'] > 4)]
        ds_result = ds_df[lambda x: (x['a'] < 2) | (x['a'] > 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_with_multiple_columns(self):
        """Callable that references multiple columns."""
        pd_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})
        ds_df = DataStore({'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9]})

        pd_result = pd_df[lambda df: df['x'] + df['y'] > 6]
        ds_result = ds_df[lambda df: df['x'] + df['y'] > 6]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_empty_result(self):
        """Callable that results in empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df[lambda x: x['a'] > 100]
        ds_result = ds_df[lambda x: x['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_all_rows(self):
        """Callable that matches all rows."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df[lambda x: x['a'] > 0]
        ds_result = ds_df[lambda x: x['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_preserves_index(self):
        """Callable indexing should preserve original indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])
        ds_df = DataStore(pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50]))

        pd_result = pd_df[lambda x: x['a'] > 2]
        ds_result = ds_df[lambda x: x['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_with_string_column(self):
        """Callable with string column operations."""
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})
        ds_df = DataStore({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})

        # Filter by age
        pd_result = pd_df[lambda x: x['age'] >= 30]
        ds_result = ds_df[lambda x: x['age'] >= 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_equality_filter(self):
        """Callable with equality condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 2]})
        ds_df = DataStore({'a': [1, 2, 2, 3, 2]})

        pd_result = pd_df[lambda x: x['a'] == 2]
        ds_result = ds_df[lambda x: x['a'] == 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_chained_operations(self):
        """Callable indexing followed by other operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df[lambda x: x['a'] > 2][['b']]
        ds_result = ds_df[lambda x: x['a'] > 2][['b']]

        assert_datastore_equals_pandas(ds_result, pd_result)
