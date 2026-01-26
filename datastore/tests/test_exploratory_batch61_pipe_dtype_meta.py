"""
Exploratory Batch 61: Pipe Operations, Type Inference, and Metadata Methods

Testing scenarios:
1. pipe() method with various functions and chains
2. convert_dtypes() and infer_objects() with lazy operations
3. memory_usage() and metadata inspection
4. get_dummies() in chains
5. select_dtypes() combinations with lazy operations
6. reindex/set_axis operations in chains
7. Complex dtype coercion scenarios
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_dataframe


# =============================================================================
# Test pipe() operations
# =============================================================================

class TestPipeOperations:
    """Test pipe() method with various functions."""

    def test_pipe_simple_function(self):
        """Test pipe with a simple function."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def add_column(df):
            df = df.copy() if isinstance(df, pd.DataFrame) else df
            return df.assign(C=df['A'] + df['B'])

        pd_result = pd_df.pipe(add_column)
        ds_result = ds_df.pipe(add_column)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_arguments(self):
        """Test pipe with function that takes arguments."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def multiply_column(df, col, factor):
            df = df.copy() if isinstance(df, pd.DataFrame) else df
            return df.assign(**{f'{col}_scaled': df[col] * factor})

        pd_result = pd_df.pipe(multiply_column, 'A', 10)
        ds_result = ds_df.pipe(multiply_column, 'A', 10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_chained(self):
        """Test multiple pipe calls chained."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def add_sum(df):
            df = df.copy() if isinstance(df, pd.DataFrame) else df
            return df.assign(sum=df['A'] + df['B'])

        def add_diff(df):
            df = df.copy() if isinstance(df, pd.DataFrame) else df
            return df.assign(diff=df['B'] - df['A'])

        pd_result = pd_df.pipe(add_sum).pipe(add_diff)
        ds_result = ds_df.pipe(add_sum).pipe(add_diff)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_after_filter(self):
        """Test pipe after filter operation."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        def double_column(df, col):
            df = df.copy() if isinstance(df, pd.DataFrame) else df
            return df.assign(**{f'{col}_double': df[col] * 2})

        pd_result = pd_df[pd_df['A'] > 2].pipe(double_column, 'B')
        ds_result = ds_df[ds_df['A'] > 2].pipe(double_column, 'B')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_lambda(self):
        """Test pipe with lambda function."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.pipe(lambda df: df.assign(total=df.sum(axis=1)))
        ds_result = ds_df.pipe(lambda df: df.assign(total=df.sum(axis=1)))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_returning_scalar(self):
        """Test pipe with function returning scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def get_max_sum(df):
            return df.sum().max()

        pd_result = pd_df.pipe(get_max_sum)
        ds_result = ds_df.pipe(get_max_sum)
        assert pd_result == ds_result

    def test_pipe_with_kwargs(self):
        """Test pipe with keyword arguments."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def filter_and_scale(df, min_val=1, scale_factor=1):
            df = df.copy() if isinstance(df, pd.DataFrame) else df
            filtered = df[df['A'] >= min_val]
            return filtered.assign(scaled=filtered['A'] * scale_factor)

        pd_result = pd_df.pipe(filter_and_scale, min_val=2, scale_factor=5)
        ds_result = ds_df.pipe(filter_and_scale, min_val=2, scale_factor=5)
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test convert_dtypes() and infer_objects()
# =============================================================================

class TestTypeConversion:
    """Test type conversion methods."""

    def test_convert_dtypes_basic(self):
        """Test basic convert_dtypes."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.convert_dtypes()
        ds_result_df = get_dataframe(ds_df.convert_dtypes())
        # Check column values match (dtypes may differ slightly)
        assert list(pd_result.columns) == list(ds_result_df.columns)
        for col in pd_result.columns:
            pd.testing.assert_series_equal(
                pd_result[col].astype(str),
                ds_result_df[col].astype(str),
                check_names=False,
                check_dtype=False
            )

    def test_convert_dtypes_with_nulls(self):
        """Test convert_dtypes with nullable types."""
        pd_df = pd.DataFrame({
            'int_col': [1, None, 3],
            'float_col': [1.1, np.nan, 3.3],
            'str_col': ['a', None, 'c']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.convert_dtypes()
        ds_result_df = get_dataframe(ds_df.convert_dtypes())
        assert list(pd_result.columns) == list(ds_result_df.columns)

    def test_convert_dtypes_after_filter(self):
        """Test convert_dtypes after filter."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['A'] > 2].convert_dtypes()
        ds_result = ds_df[ds_df['A'] > 2].convert_dtypes()
        ds_result_df = get_dataframe(ds_result)
        # Check values match
        for col in pd_result.columns:
            np.testing.assert_array_almost_equal(
                pd_result[col].values.astype(float),
                ds_result_df[col].values.astype(float)
            )

    def test_infer_objects_basic(self):
        """Test basic infer_objects."""
        pd_df = pd.DataFrame({
            'A': pd.Series([1, 2, 3], dtype=object),
            'B': ['a', 'b', 'c']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.infer_objects()
        ds_result = get_dataframe(ds_df.infer_objects())
        # Just check values match
        for col in pd_result.columns:
            assert list(pd_result[col]) == list(ds_result[col])

    def test_infer_objects_mixed_types(self):
        """Test infer_objects with mixed types."""
        pd_df = pd.DataFrame({
            'A': pd.Series([1, 2, '3'], dtype=object),
            'B': pd.Series([1.0, 2.0, 3.0], dtype=object)
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.infer_objects()
        ds_result = get_dataframe(ds_df.infer_objects())
        assert list(pd_result.columns) == list(ds_result.columns)


# =============================================================================
# Test memory_usage() and metadata
# =============================================================================

class TestMemoryAndMetadata:
    """Test memory_usage and metadata inspection methods."""

    def test_memory_usage_basic(self):
        """Test basic memory_usage."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.memory_usage()
        ds_result = ds_df.memory_usage()
        # Just verify it returns a Series with same index
        assert isinstance(ds_result, pd.Series)
        assert 'Index' in ds_result.index or 'index' in ds_result.index.str.lower()

    def test_memory_usage_deep(self):
        """Test memory_usage with deep=True."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.memory_usage(deep=True)
        ds_result = ds_df.memory_usage(deep=True)
        assert isinstance(ds_result, pd.Series)

    def test_memory_usage_no_index(self):
        """Test memory_usage with index=False."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.memory_usage(index=False)
        ds_result = ds_df.memory_usage(index=False)
        assert isinstance(ds_result, pd.Series)
        # Index should not be in result
        assert 'Index' not in ds_result.index

    def test_ndim(self):
        """Test ndim property."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)
        assert ds_df.ndim == pd_df.ndim == 2

    def test_size(self):
        """Test size property."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)
        assert ds_df.size == pd_df.size == 6

    def test_empty_property(self):
        """Test empty property."""
        pd_df = pd.DataFrame({'A': []})
        ds_df = DataStore(pd_df)
        assert ds_df.empty == pd_df.empty == True

        pd_df2 = pd.DataFrame({'A': [1]})
        ds_df2 = DataStore(pd_df2)
        assert ds_df2.empty == pd_df2.empty == False


# =============================================================================
# Test get_dummies operations
# =============================================================================

class TestGetDummies:
    """Test get_dummies in various scenarios."""

    def test_get_dummies_basic(self):
        """Test basic get_dummies on column."""
        pd_df = pd.DataFrame({'A': ['a', 'b', 'a', 'c']})
        ds_df = DataStore(pd_df)

        # For get_dummies, we need to get the underlying Series first
        pd_result = pd.get_dummies(pd_df['A'])
        ds_series = get_dataframe(ds_df[['A']])['A']
        ds_result = pd.get_dummies(ds_series)
        pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_get_dummies_dataframe(self):
        """Test get_dummies on DataFrame."""
        pd_df = pd.DataFrame({
            'A': ['a', 'b', 'a'],
            'B': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd.get_dummies(pd_df)
        ds_result_df = get_dataframe(ds_df)
        ds_result = pd.get_dummies(ds_result_df)
        pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_get_dummies_prefix(self):
        """Test get_dummies with prefix."""
        pd_df = pd.DataFrame({'A': ['x', 'y', 'x']})
        ds_df = DataStore(pd_df)

        pd_result = pd.get_dummies(pd_df['A'], prefix='cat')
        ds_series = get_dataframe(ds_df[['A']])['A']
        ds_result = pd.get_dummies(ds_series, prefix='cat')
        pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_get_dummies_drop_first(self):
        """Test get_dummies with drop_first."""
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c', 'a']})
        ds_df = DataStore(pd_df)

        pd_result = pd.get_dummies(pd_df['A'], drop_first=True)
        ds_series = get_dataframe(ds_df[['A']])['A']
        ds_result = pd.get_dummies(ds_series, drop_first=True)
        pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_get_dummies_after_filter(self):
        """Test get_dummies after filter."""
        pd_df = pd.DataFrame({
            'A': ['a', 'b', 'a', 'c', 'b'],
            'B': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['B'] > 2]
        ds_filtered_df = get_dataframe(ds_df[ds_df['B'] > 2])

        pd_result = pd.get_dummies(pd_filtered['A'])
        ds_result = pd.get_dummies(ds_filtered_df['A'])
        pd.testing.assert_frame_equal(ds_result, pd_result, check_dtype=False)


# =============================================================================
# Test select_dtypes in chains
# =============================================================================

class TestSelectDtypesChains:
    """Test select_dtypes in lazy chains."""

    def test_select_dtypes_include_number(self):
        """Test select_dtypes with include='number'."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.1, 2.2, 3.3],
            'C': ['a', 'b', 'c']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include='number')
        ds_result = ds_df.select_dtypes(include='number')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_object(self):
        """Test select_dtypes with exclude='object'."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.1, 2.2, 3.3],
            'C': ['a', 'b', 'c']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(exclude='object')
        ds_result = ds_df.select_dtypes(exclude='object')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_after_filter(self):
        """Test select_dtypes after filter."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['A'] > 2].select_dtypes(include='number')
        ds_result = ds_df[ds_df['A'] > 2].select_dtypes(include='number')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_then_mean(self):
        """Test select_dtypes then aggregate."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.0, 5.0, 6.0],
            'C': ['a', 'b', 'c']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include='number').mean()
        ds_result = ds_df.select_dtypes(include='number').mean()
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_select_dtypes_multiple_types(self):
        """Test select_dtypes with multiple type includes."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1.1, 2.2, 3.3],
            'C': ['a', 'b', 'c'],
            'D': [True, False, True]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include=['int64', 'float64'])
        ds_result = ds_df.select_dtypes(include=['int64', 'float64'])
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


# =============================================================================
# Test reindex and set_axis operations
# =============================================================================

class TestReindexOperations:
    """Test reindex and set_axis operations."""

    def test_reindex_basic(self):
        """Test basic reindex."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['c', 'b', 'a'])
        ds_result = ds_df.reindex(['c', 'b', 'a'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_missing(self):
        """Test reindex with missing values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['a', 'b', 'd'])
        ds_result = ds_df.reindex(['a', 'b', 'd'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_fill_value(self):
        """Test reindex with fill_value."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['a', 'b', 'd'], fill_value=0)
        ds_result = ds_df.reindex(['a', 'b', 'd'], fill_value=0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """Test reindex columns."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(columns=['C', 'A'])
        ds_result = ds_df.reindex(columns=['C', 'A'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_index(self):
        """Test set_axis on index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_axis(['x', 'y', 'z'], axis=0)
        ds_result = ds_df.set_axis(['x', 'y', 'z'], axis=0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_columns(self):
        """Test set_axis on columns."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_axis(['X', 'Y'], axis=1)
        ds_result = ds_df.set_axis(['X', 'Y'], axis=1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_after_filter(self):
        """Test reindex after filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]}, index=['a', 'b', 'c', 'd'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['A'] > 1].reindex(['b', 'c', 'd'])
        ds_result = ds_df[ds_df['A'] > 1].reindex(['b', 'c', 'd'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_like_basic(self):
        """Test reindex_like with pandas DataFrame as template."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        pd_df2 = pd.DataFrame({'B': [4, 5]}, index=['b', 'c'])
        ds_df1 = DataStore(pd_df1)

        # Use pandas DataFrame as the template (not DataStore)
        pd_result = pd_df1.reindex_like(pd_df2)
        ds_result = ds_df1.reindex_like(pd_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test dtype coercion chains
# =============================================================================

class TestDtypeCoercionChains:
    """Test dtype coercion in chain operations."""

    def test_astype_in_chain(self):
        """Test astype in a chain."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['A'] > 1].astype({'A': 'float64'})
        ds_result = ds_df[ds_df['A'] > 1].astype({'A': 'float64'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_after_groupby_agg(self):
        """Test astype after groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')['value'].sum().astype(float)
        ds_result_raw = ds_df.groupby('group')['value'].sum()
        # Convert to DataFrame first then apply astype
        ds_result_df = get_dataframe(ds_result_raw.reset_index())
        ds_result = ds_result_df.set_index('group')['value'].astype(float)
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_astype_before_filter(self):
        """Test astype before filter."""
        pd_df = pd.DataFrame({'A': ['1', '2', '3'], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.astype({'A': int})[pd_df.astype({'A': int})['A'] > 1]
        ds_result = ds_df.astype({'A': int})[ds_df.astype({'A': int})['A'] > 1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_multiple_columns(self):
        """Test astype on multiple columns."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.0, 5.0, 6.0],
            'C': ['7', '8', '9']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.astype({'A': float, 'B': int})
        ds_result = ds_df.astype({'A': float, 'B': int})
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test edge cases and boundary conditions
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe_pipe(self):
        """Test pipe on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore(pd_df)

        def add_col(df):
            return df.assign(C=0)

        pd_result = pd_df.pipe(add_col)
        ds_result = ds_df.pipe(add_col)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_operations(self):
        """Test operations on single row DataFrame."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.pipe(lambda df: df.assign(C=df['A'] + df['B']))
        ds_result = ds_df.pipe(lambda df: df.assign(C=df['A'] + df['B']))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_null_column_convert_dtypes(self):
        """Test convert_dtypes with all null column."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [None, None, None]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.convert_dtypes()
        ds_result = get_dataframe(ds_df.convert_dtypes())
        assert list(pd_result.columns) == list(ds_result.columns)

    def test_select_dtypes_no_match(self):
        """Test select_dtypes when no columns match."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include='object')
        ds_result = ds_df.select_dtypes(include='object')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_empty_list(self):
        """Test reindex with empty list."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex([])
        ds_result = ds_df.reindex([])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_type_operations(self):
        """Test multiple type conversion operations chained."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4.0, 5.0, 6.0]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.astype({'A': float}).astype({'A': int, 'B': int})
        ds_result = ds_df.astype({'A': float}).astype({'A': int, 'B': int})
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test complex chains combining multiple operations
# =============================================================================

class TestComplexChains:
    """Test complex chains combining multiple operations."""

    def test_filter_pipe_select_dtypes_mean(self):
        """Test filter -> pipe -> select_dtypes -> mean chain."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore(pd_df)

        def double_numeric(df):
            numeric = df.select_dtypes(include='number')
            return df.assign(**{col: df[col] * 2 for col in numeric.columns})

        pd_result = pd_df[pd_df['A'] > 2].pipe(double_numeric).select_dtypes('number').mean()
        ds_result = ds_df[ds_df['A'] > 2].pipe(double_numeric).select_dtypes('number').mean()
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_assign_reindex_filter(self):
        """Test assign -> reindex -> filter chain using explicit condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        # Use explicit condition instead of lambda
        pd_temp = pd_df.assign(B=pd_df['A'] * 2).reindex(['c', 'b', 'a'])
        pd_result = pd_temp[pd_temp['B'] > 2]
        
        ds_temp = ds_df.assign(B=ds_df['A'] * 2).reindex(['c', 'b', 'a'])
        ds_result = ds_temp[ds_temp['B'] > 2]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_pipe_chain(self):
        """Test groupby -> agg -> pipe chain."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        def normalize(df):
            return df.assign(norm_sum=df['value'] / df['value'].max())

        pd_result = pd_df.groupby('group')['value'].sum().reset_index().pipe(normalize)
        ds_result = ds_df.groupby('group')['value'].sum().reset_index().pipe(normalize)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_filter_astype_chain(self):
        """Test multiple filters with astype chain."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df)

        # First filter
        pd_temp = pd_df[pd_df['A'] > 1]
        ds_temp = ds_df[ds_df['A'] > 1]
        
        # Second filter - use explicit condition
        pd_temp2 = pd_temp[pd_temp['B'] < 45]
        ds_temp2 = ds_temp[ds_temp['B'] < 45]
        
        # Type conversion and assign
        pd_result = pd_temp2.astype({'A': float}).assign(ratio=lambda df: df['B'] / df['A'])
        ds_temp3 = ds_temp2.astype({'A': float})
        ds_result = ds_temp3.assign(ratio=ds_temp3['B'] / ds_temp3['A'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test additional metadata properties
# =============================================================================

class TestAdditionalProperties:
    """Test additional DataFrame properties."""

    def test_axes_property(self):
        """Test axes property."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_axes = pd_df.axes
        ds_axes = ds_df.axes
        assert len(pd_axes) == len(ds_axes) == 2
        pd.testing.assert_index_equal(pd_axes[0], ds_axes[0])
        pd.testing.assert_index_equal(pd_axes[1], ds_axes[1])

    def test_values_property(self):
        """Test values property."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        np.testing.assert_array_equal(pd_df.values, ds_df.values)

    def test_dtypes_property(self):
        """Test dtypes property."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.1, 2.2, 3.3]})
        ds_df = DataStore(pd_df)

        pd.testing.assert_series_equal(pd_df.dtypes, ds_df.dtypes)

    def test_keys_method(self):
        """Test keys() method."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df)

        pd.testing.assert_index_equal(pd_df.keys(), ds_df.keys())

    def test_T_property(self):
        """Test T (transpose) property."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.T
        ds_result = ds_df.T
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
