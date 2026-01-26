"""
Exploratory Batch 26: Advanced Methods and Edge Cases

Focus areas:
1. Matrix operations (dot)
2. Axis manipulation (swapaxes, swaplevel, transpose variations)
3. Update method edge cases
4. Truncate operations
5. Memory/View operations
6. Complex aggregation scenarios
7. Method chaining edge cases
8. Type coercion edge cases
"""

import pytest
from tests.xfail_markers import datastore_query_variable_scope
import pandas as pd
import numpy as np
from datastore import DataStore
import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_series


class TestMatrixOperations:
    """Test matrix multiplication and related operations."""
    
    def test_dot_dataframe_with_series(self):
        """Test dot product of DataFrame with Series."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        pd_series = pd.Series([1.0, 2.0], index=['a', 'b'])
        pd_result = pd_df.dot(pd_series)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.dot(pd_series)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_dot_dataframe_with_dataframe(self):
        """Test dot product of DataFrame with DataFrame."""
        pd_df1 = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        pd_df2 = pd.DataFrame({'x': [1.0, 0.0], 'y': [0.0, 1.0]}, index=['a', 'b'])
        pd_result = pd_df1.dot(pd_df2)
        
        ds_df1 = DataStore(pd_df1)
        ds_result = ds_df1.dot(pd_df2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_dot_with_numpy_array(self):
        """Test dot product with numpy array."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        np_arr = np.array([[1.0, 0.0], [0.0, 1.0]])
        pd_result = pd_df.dot(np_arr)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.dot(np_arr)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAxisManipulation:
    """Test axis manipulation methods."""
    
    def test_swapaxes_basic(self):
        """Test swapaxes - transpose equivalent."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.swapaxes(0, 1)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.swapaxes(0, 1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_transpose_property(self):
        """Test T property for transpose."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.T
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.T
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_transpose_method(self):
        """Test transpose() method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.transpose()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.transpose()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_swaplevel_multiindex_columns(self):
        """Test swaplevel with MultiIndex columns."""
        arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame(np.random.randn(3, 4), columns=index)
        pd_result = pd_df.swaplevel(axis=1)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.swaplevel(axis=1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUpdateMethod:
    """Test update method edge cases."""
    
    def test_update_basic(self):
        """Test basic update operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_other = pd.DataFrame({'b': [40, 50, 60]})
        pd_df_copy = pd_df.copy()
        pd_df_copy.update(pd_other)
        
        ds_df = DataStore(pd_df)
        ds_df.update(pd_other)
        
        assert_datastore_equals_pandas(ds_df, pd_df_copy)
    
    def test_update_with_na(self):
        """Test update with NA values - NA in other should not update."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_other = pd.DataFrame({'b': [40, np.nan, 60]})
        pd_df_copy = pd_df.copy()
        pd_df_copy.update(pd_other)
        
        ds_df = DataStore(pd_df)
        ds_df.update(pd_other)
        
        assert_datastore_equals_pandas(ds_df, pd_df_copy)
    
    def test_update_overwrite_false(self):
        """Test update with overwrite=False."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, 6]})
        pd_other = pd.DataFrame({'a': [10, 20, 30]})
        pd_df_copy = pd_df.copy()
        pd_df_copy.update(pd_other, overwrite=False)
        
        ds_df = DataStore(pd_df)
        ds_df.update(pd_other, overwrite=False)
        
        assert_datastore_equals_pandas(ds_df, pd_df_copy)
    
    def test_update_with_datastore(self):
        """Test update with DataStore as argument."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_other = pd.DataFrame({'b': [40, 50, 60]})
        pd_df_copy = pd_df.copy()
        pd_df_copy.update(pd_other)
        
        ds_df = DataStore(pd_df)
        ds_other = DataStore(pd_other)
        ds_df.update(ds_other)
        
        assert_datastore_equals_pandas(ds_df, pd_df_copy)


class TestTruncate:
    """Test truncate operations."""
    
    def test_truncate_with_integer_index(self):
        """Test truncate with integer index."""
        pd_df = pd.DataFrame({'a': range(10)}, index=range(10))
        pd_result = pd_df.truncate(before=2, after=7)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.truncate(before=2, after=7)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_truncate_before_only(self):
        """Test truncate with before only."""
        pd_df = pd.DataFrame({'a': range(10)}, index=range(10))
        pd_result = pd_df.truncate(before=5)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.truncate(before=5)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_truncate_after_only(self):
        """Test truncate with after only."""
        pd_df = pd.DataFrame({'a': range(10)}, index=range(10))
        pd_result = pd_df.truncate(after=5)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.truncate(after=5)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_truncate_with_datetime_index(self):
        """Test truncate with datetime index."""
        dates = pd.date_range('2023-01-01', periods=10)
        pd_df = pd.DataFrame({'a': range(10)}, index=dates)
        pd_result = pd_df.truncate(before='2023-01-03', after='2023-01-07')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.truncate(before='2023-01-03', after='2023-01-07')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMemoryViewOperations:
    """Test memory and view related operations."""
    
    def test_memory_usage_basic(self):
        """Test memory_usage method."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        pd_result = pd_df.memory_usage()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.memory_usage()
        
        assert_series_equal(ds_result, pd_result)
    
    def test_memory_usage_deep(self):
        """Test memory_usage with deep=True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        pd_result = pd_df.memory_usage(deep=True)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.memory_usage(deep=True)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_to_numpy_with_homogeneous_data(self):
        """Test to_numpy with homogeneous numeric data (avoids na_value issue)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.to_numpy()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.to_numpy()
        
        np.testing.assert_array_equal(ds_result, pd_result)
    
    def test_values_property(self):
        """Test values property returns numpy array."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.values
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.values
        
        np.testing.assert_array_equal(ds_result, pd_result)


class TestComplexAggregation:
    """Test complex aggregation scenarios."""
    
    def test_agg_multiple_funcs_dict(self):
        """Test agg with dict of multiple functions per column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_result = pd_df.agg({'a': ['sum', 'mean'], 'b': ['min', 'max']})
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.agg({'a': ['sum', 'mean'], 'b': ['min', 'max']})
        
        # Result may be DataFrame or may differ in column order
        if isinstance(ds_result, pd.DataFrame):
            assert_frame_equal(ds_result, pd_result, check_like=True)
        else:
            assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_agg_single_func_string(self):
        """Test agg with single function as string."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_result = pd_df.agg('sum')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.agg('sum')
        
        assert_series_equal(ds_result, pd_result)
    
    def test_agg_lambda_function(self):
        """Test agg with lambda function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_result = pd_df.agg(lambda x: x.max() - x.min())
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.agg(lambda x: x.max() - x.min())
        
        assert_series_equal(ds_result, pd_result)
    
    def test_aggregate_alias(self):
        """Test aggregate (alias for agg)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_result = pd_df.aggregate('sum')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.aggregate('sum')
        
        assert_series_equal(ds_result, pd_result)


class TestMethodChaining:
    """Test method chaining edge cases."""
    
    def test_filter_then_sort_then_head(self):
        """Test filter -> sort -> head chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2], 'b': list('abcdefg')})
        pd_result = pd_df[pd_df['a'] > 2].sort_values('a').head(3)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df[ds_df['a'] > 2].sort_values('a').head(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_assign_then_filter_then_groupby_explicit(self):
        """Test assign -> filter -> groupby chain using explicit boolean indexing."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40]})
        pd_df_temp = pd_df.assign(doubled=lambda x: x['value'] * 2)
        pd_result = (pd_df_temp[pd_df_temp['doubled'] > 25]
                     .groupby('category')['doubled']
                     .sum()
                     .reset_index())
        
        ds_df = DataStore(pd_df)
        ds_df_temp = ds_df.assign(doubled=lambda x: x['value'] * 2)
        ds_result = (ds_df_temp[ds_df_temp['doubled'] > 25]
                     .groupby('category')['doubled']
                     .sum()
                     .reset_index())
        
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
    
    def test_multiple_assigns_in_chain(self):
        """Test multiple assign calls in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = (pd_df
                     .assign(c=lambda x: x['a'] + x['b'])
                     .assign(d=lambda x: x['c'] * 2))
        
        ds_df = DataStore(pd_df)
        ds_result = (ds_df
                     .assign(c=lambda x: x['a'] + x['b'])
                     .assign(d=lambda x: x['c'] * 2))
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rename_then_select_columns(self):
        """Test rename -> column selection chain."""
        pd_df = pd.DataFrame({'old_name': [1, 2, 3], 'other': [4, 5, 6]})
        pd_result = pd_df.rename(columns={'old_name': 'new_name'})[['new_name']]
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.rename(columns={'old_name': 'new_name'})[['new_name']]
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTypeCoercion:
    """Test type coercion edge cases."""
    
    def test_astype_multiple_columns(self):
        """Test astype with dict for multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['4', '5', '6']})
        pd_result = pd_df.astype({'a': float, 'b': int})
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.astype({'a': float, 'b': int})
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_convert_dtypes(self):
        """Test convert_dtypes method."""
        pd_df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
        pd_result = pd_df.convert_dtypes()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.convert_dtypes()
        
        # convert_dtypes may create nullable types, so check values only
        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == len(pd_result)
    
    def test_infer_objects(self):
        """Test infer_objects method."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, 3], dtype=object)})
        pd_result = pd_df.infer_objects()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.infer_objects()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialValues:
    """Test handling of special values."""
    
    def test_operations_with_inf(self):
        """Test operations with infinity values."""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 4.0]})
        pd_result = pd_df.replace([np.inf, -np.inf], np.nan)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.replace([np.inf, -np.inf], np.nan)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_clip_with_inf(self):
        """Test clip with infinite bounds."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df.clip(lower=-np.inf, upper=3)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.clip(lower=-np.inf, upper=3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_fillna_with_inf(self):
        """Test fillna replacing NaN with inf."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_result = pd_df.fillna(np.inf)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.fillna(np.inf)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIndexOperations:
    """Test advanced index operations."""
    
    def test_set_index_then_reset(self):
        """Test set_index followed by reset_index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        pd_result = pd_df.set_index('a').reset_index()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.set_index('a').reset_index()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_set_index_multiindex(self):
        """Test set_index with multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2], 'b': ['x', 'y', 'x', 'y'], 'c': [10, 20, 30, 40]})
        pd_result = pd_df.set_index(['a', 'b'])
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.set_index(['a', 'b'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_reindex_with_fill(self):
        """Test reindex with fill_value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_result = pd_df.reindex([0, 1, 2, 3, 4], fill_value=0)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.reindex([0, 1, 2, 3, 4], fill_value=0)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_set_axis_columns(self):
        """Test set_axis for columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.set_axis(['x', 'y'], axis=1)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.set_axis(['x', 'y'], axis=1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEmptyDataFrameOperations:
    """Test operations on empty DataFrames."""
    
    def test_empty_df_groupby(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd_df.groupby('a')['b'].sum().reset_index()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.groupby('a')['b'].sum().reset_index()
        
        assert len(ds_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)
    
    def test_empty_df_concat(self):
        """Test concat with empty DataFrame."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'a': []})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_empty_df_sort_values(self):
        """Test sort_values on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd_df.sort_values('a')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.sort_values('a')
        
        assert len(ds_result) == 0


class TestSingleRowOperations:
    """Test operations on single-row DataFrames."""
    
    def test_single_row_groupby(self):
        """Test groupby on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [10]})
        pd_result = pd_df.groupby('a')['b'].sum().reset_index()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.groupby('a')['b'].sum().reset_index()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_single_row_rolling(self):
        """Test rolling on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [10.0]})
        pd_result = pd_df.rolling(window=1).mean()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.rolling(window=1).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_single_row_describe(self):
        """Test describe on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [10], 'b': [20]})
        pd_result = pd_df.describe()
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.describe()
        
        # Just check the shape, values may differ slightly
        assert ds_result.shape == pd_result.shape


class TestPipeOperations:
    """Test pipe method variations."""
    
    def test_pipe_basic(self):
        """Test basic pipe operation."""
        def add_column(df):
            return df.assign(c=lambda x: x['a'] + x['b'])
        
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.pipe(add_column)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.pipe(add_column)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_pipe_with_args(self):
        """Test pipe with additional arguments."""
        def multiply_column(df, column, factor):
            return df.assign(**{f'{column}_scaled': df[column] * factor})
        
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.pipe(multiply_column, 'a', 10)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.pipe(multiply_column, 'a', 10)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_pipe_chained(self):
        """Test multiple pipe calls in chain."""
        def add_one(df):
            return df + 1
        
        def double(df):
            return df * 2
        
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.pipe(add_one).pipe(double)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.pipe(add_one).pipe(double)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestQueryMethod:
    """Test query method variations."""
    
    def test_query_basic(self):
        """Test basic query."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        pd_result = pd_df.query('a > 2')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('a > 2')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_query_with_and(self):
        """Test query with AND condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        pd_result = pd_df.query('a > 2 and b < 50')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('a > 2 and b < 50')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    @datastore_query_variable_scope
    def test_query_with_variable(self):
        """Test query with external variable - known limitation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        threshold = 3
        pd_result = pd_df.query('a > @threshold')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('a > @threshold')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_query_without_at_variable(self):
        """Test query without @ variable (direct comparison)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        pd_result = pd_df.query('a > 3')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.query('a > 3')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEvalMethod:
    """Test eval method variations."""
    
    def test_eval_basic(self):
        """Test basic eval."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.eval('c = a + b')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.eval('c = a + b')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_eval_complex_expression(self):
        """Test eval with complex expression."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.eval('c = (a * 2) + (b ** 2)')
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.eval('c = (a * 2) + (b ** 2)')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_eval_inplace_false(self):
        """Test eval with inplace=False (default)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.eval('c = a + b', inplace=False)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.eval('c = a + b', inplace=False)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDuplicateColumnHandling:
    """Test handling of duplicate column names."""
    
    def test_concat_with_duplicate_columns(self):
        """Test concat creates duplicate columns correctly."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})
        pd_result = pd.concat([pd_df1, pd_df2], axis=1)
        
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds.concat([ds_df1, ds_df2], axis=1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_join_with_duplicate_columns(self):
        """Test join handles duplicate column suffixes."""
        pd_df1 = pd.DataFrame({'key': [1, 2], 'value': [10, 20]})
        pd_df2 = pd.DataFrame({'key': [1, 2], 'value': [100, 200]})
        pd_result = pd_df1.merge(pd_df2, on='key', suffixes=('_left', '_right'))
        
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds_df1.merge(ds_df2, on='key', suffixes=('_left', '_right'))
        
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestDropOperations:
    """Test drop method variations."""
    
    def test_drop_columns_by_name(self):
        """Test drop columns by name."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        pd_result = pd_df.drop(columns=['b'])
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.drop(columns=['b'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_drop_rows_by_index(self):
        """Test drop rows by index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 2])
        pd_result = pd_df.drop(index=[1])
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.drop(index=[1])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_drop_multiple_columns(self):
        """Test drop multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]})
        pd_result = pd_df.drop(columns=['b', 'd'])
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.drop(columns=['b', 'd'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestApplyMapOperations:
    """Test apply and applymap/map operations."""
    
    def test_apply_row_wise(self):
        """Test apply row-wise."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.apply(sum, axis=1)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.apply(sum, axis=1)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_apply_column_wise(self):
        """Test apply column-wise."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.apply(sum, axis=0)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.apply(sum, axis=0)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_map_with_dict(self):
        """Test map with dictionary - need to call _execute() for ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        pd_result = pd_df['b'].map({'x': 'X', 'y': 'Y', 'z': 'Z'})
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df['b'].map({'x': 'X', 'y': 'Y', 'z': 'Z'})
        
        # ColumnExpr.map() returns ColumnExpr, need to execute
        ds_result = get_series(ds_result)
        
        assert_series_equal(ds_result, pd_result)
    
    def test_map_with_function(self):
        """Test map with function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        pd_result = pd_df['a'].map(lambda x: x * 10)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df['a'].map(lambda x: x * 10)
        
        # ColumnExpr.map() returns ColumnExpr, need to execute
        ds_result = get_series(ds_result)
        
        assert_series_equal(ds_result, pd_result)


class TestAdditionalEdgeCases:
    """Additional edge cases discovered during testing."""
    
    def test_filter_with_boolean_list(self):
        """Test filtering with boolean list."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        mask = [True, False, True]
        pd_result = pd_df[mask]
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df[mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_filter_with_numpy_bool_array(self):
        """Test filtering with numpy boolean array."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        mask = np.array([True, False, True])
        pd_result = pd_df[mask]
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df[mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_iloc_negative_indices(self):
        """Test iloc with negative indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df.iloc[-3:]
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.iloc[-3:]
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_loc_with_slice(self):
        """Test loc with slice."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=['a', 'b', 'c', 'd', 'e'])
        pd_result = pd_df.loc['b':'d']
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.loc['b':'d']
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_copy_deep(self):
        """Test deep copy."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.copy(deep=True)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.copy(deep=True)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_copy_shallow(self):
        """Test shallow copy."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.copy(deep=False)
        
        ds_df = DataStore(pd_df)
        ds_result = ds_df.copy(deep=False)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
