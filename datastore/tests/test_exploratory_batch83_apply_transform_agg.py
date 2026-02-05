"""
Exploratory Batch 83: Apply, Transform, Agg Edge Cases

Focus areas:
1. apply() with various function types and axis options
2. transform() return type consistency
3. agg() with multiple aggregation functions
4. pipe() chaining
5. map()/applymap() element-wise operations
6. Complex chain operations with these methods
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestApplyEdgeCases:
    """Test apply() method edge cases"""

    def test_apply_axis0_simple(self):
        """apply() with axis=0 (column-wise) - simple sum"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.apply(sum, axis=0)
        ds_result = ds_df.apply(sum, axis=0)

        assert_series_equal(ds_result, pd_result)

    def test_apply_axis1_row_sum(self):
        """apply() with axis=1 (row-wise) - row sums"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.apply(sum, axis=1)
        ds_result = ds_df.apply(sum, axis=1)

        assert_series_equal(ds_result, pd_result)

    def test_apply_lambda_column(self):
        """apply() with lambda on columns"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [10, 20, 30]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [10, 20, 30]
        })

        pd_result = pd_df.apply(lambda col: col.max() - col.min(), axis=0)
        ds_result = ds_df.apply(lambda col: col.max() - col.min(), axis=0)

        assert_series_equal(ds_result, pd_result)

    def test_apply_returns_series(self):
        """apply() returning Series should have correct index"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.apply(np.mean)
        ds_result = ds_df.apply(np.mean)

        # Check both values and index
        assert list(pd_result.index) == list(ds_result.index)
        assert list(pd_result.values) == list(ds_result.values)

    def test_apply_with_result_type_expand(self):
        """apply() with result_type='expand'"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3]
        })
        ds_df = DataStore({
            'A': [1, 2, 3]
        })

        # Function that returns multiple values
        pd_result = pd_df.apply(lambda row: [row['A'], row['A'] * 2], axis=1, result_type='expand')
        ds_result = ds_df.apply(lambda row: [row['A'], row['A'] * 2], axis=1, result_type='expand')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_after_filter(self):
        """apply() after filter operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['A'] > 2].apply(sum, axis=0)
        ds_result = ds_df[ds_df['A'] > 2].apply(sum, axis=0)

        assert_series_equal(ds_result, pd_result)

    def test_apply_with_numpy_func(self):
        """apply() with numpy function"""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })

        pd_result = pd_df.apply(np.sqrt)
        ds_result = ds_df.apply(np.sqrt)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTransformEdgeCases:
    """Test transform() method edge cases"""

    def test_transform_simple(self):
        """transform() with simple function"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.transform(lambda x: x * 2)
        ds_result = ds_df.transform(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_multiple_funcs(self):
        """transform() with multiple functions"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.transform([np.sqrt, np.exp])
        ds_result = ds_df.transform([np.sqrt, np.exp])

        # Compare column structure
        assert list(pd_result.columns) == list(ds_result.columns)
        # Compare values
        np.testing.assert_array_almost_equal(pd_result.values, ds_result.values)

    def test_transform_preserves_shape(self):
        """transform() should preserve DataFrame shape"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.transform(lambda x: x - x.mean())
        ds_result = ds_df.transform(lambda x: x - x.mean())

        assert pd_result.shape == ds_result.shape
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_after_filter(self):
        """transform() after filter"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        pd_result = pd_df[pd_df['A'] > 1].transform(lambda x: x * 2)
        ds_result = ds_df[ds_df['A'] > 1].transform(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggEdgeCases:
    """Test agg() method edge cases"""

    def test_agg_single_func(self):
        """agg() with single function"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.agg('sum')
        ds_result = ds_df.agg('sum')

        assert_series_equal(ds_result, pd_result)

    def test_agg_multiple_funcs(self):
        """agg() with multiple functions"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.agg(['sum', 'mean', 'max'])
        ds_result = ds_df.agg(['sum', 'mean', 'max'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_dict_different_funcs_per_column(self):
        """agg() with dict specifying different functions per column"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.agg({'A': 'sum', 'B': 'mean'})
        ds_result = ds_df.agg({'A': 'sum', 'B': 'mean'})

        assert_series_equal(ds_result, pd_result)

    def test_agg_dict_multiple_funcs_per_column(self):
        """agg() with dict specifying multiple functions per column"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.agg({'A': ['sum', 'mean'], 'B': 'max'})
        ds_result = ds_df.agg({'A': ['sum', 'mean'], 'B': 'max'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_after_filter(self):
        """agg() after filter"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        pd_result = pd_df[pd_df['A'] > 2].agg(['sum', 'mean'])
        ds_result = ds_df[ds_df['A'] > 2].agg(['sum', 'mean'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_with_lambda(self):
        """agg() with lambda function"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        pd_result = pd_df.agg(lambda x: x.max() - x.min())
        ds_result = ds_df.agg(lambda x: x.max() - x.min())

        assert_series_equal(ds_result, pd_result)


class TestPipeEdgeCases:
    """Test pipe() method edge cases"""

    def test_pipe_simple_function(self):
        """pipe() with simple function"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        def double(df):
            return df * 2

        pd_result = pd_df.pipe(double)
        ds_result = ds_df.pipe(double)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_args(self):
        """pipe() with additional arguments"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        def multiply(df, factor):
            return df * factor

        pd_result = pd_df.pipe(multiply, factor=3)
        ds_result = ds_df.pipe(multiply, factor=3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_chained(self):
        """pipe() multiple pipes chained"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        def add_one(df):
            return df + 1

        def multiply_two(df):
            return df * 2

        pd_result = pd_df.pipe(add_one).pipe(multiply_two)
        ds_result = ds_df.pipe(add_one).pipe(multiply_two)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_returns_scalar(self):
        """pipe() with function that returns scalar"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        def total_sum(df):
            return df.sum().sum()

        pd_result = pd_df.pipe(total_sum)
        ds_result = ds_df.pipe(total_sum)

        assert pd_result == ds_result

    def test_pipe_after_filter(self):
        """pipe() after filter operation"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        def double(df):
            return df * 2

        pd_result = pd_df[pd_df['A'] > 2].pipe(double)
        ds_result = ds_df[ds_df['A'] > 2].pipe(double)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMapApplymapEdgeCases:
    """Test map()/applymap() method edge cases"""

    def test_applymap_simple(self):
        """applymap() with simple function"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        # applymap is deprecated in pandas 2.1+, removed in pandas 3.0
        import warnings
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='DataFrame.applymap has been deprecated')
            if pandas_version >= (3, 0):
                pd_result = pd_df.map(lambda x: x ** 2)
            else:
                pd_result = pd_df.applymap(lambda x: x ** 2)
            ds_result = ds_df.applymap(lambda x: x ** 2)
            # Assert inside the block because DataStore is lazy
            assert_datastore_equals_pandas(ds_result, pd_result)

    def test_applymap_string_operation(self):
        """applymap() with string operation"""
        pd_df = pd.DataFrame({
            'A': ['hello', 'world'],
            'B': ['foo', 'bar']
        })
        ds_df = DataStore({
            'A': ['hello', 'world'],
            'B': ['foo', 'bar']
        })

        # applymap is deprecated in pandas 2.1+, removed in pandas 3.0
        import warnings
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='DataFrame.applymap has been deprecated')
            if pandas_version >= (3, 0):
                pd_result = pd_df.map(str.upper)
            else:
                pd_result = pd_df.applymap(str.upper)
            ds_result = ds_df.applymap(str.upper)
            # Assert inside the block because DataStore is lazy
            assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_alias_for_applymap(self):
        """map() should work as alias for applymap()"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        # In pandas 2.0+, DataFrame.map is preferred over applymap
        pd_result = pd_df.map(lambda x: x * 2) if hasattr(pd_df, 'map') else pd_df.applymap(lambda x: x * 2)
        ds_result = ds_df.map(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChains:
    """Test complex chains involving apply/transform/agg"""

    def test_filter_apply_filter(self):
        """Filter -> Apply -> Filter chain"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        # Filter, apply transformation, then compare
        pd_filtered = pd_df[pd_df['A'] > 2]
        ds_filtered = ds_df[ds_df['A'] > 2]

        pd_result = pd_filtered.apply(lambda x: x * 2)
        ds_result = ds_filtered.apply(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_apply_chain(self):
        """GroupBy -> Apply chain with custom function"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [10, 20, 30, 40]
        })

        # Using sum builtin with apply triggers a FutureWarning about np.sum behavior
        # We test this pattern as users may use it; pandas recommends using 'sum' string instead
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The provided callable.*is currently using')
            pd_result = pd_df.groupby('cat')['val'].apply(sum)
            ds_result = ds_df.groupby('cat')['val'].apply(sum)
            # Assert inside the block because DataStore is lazy - execution happens at assertion
            assert_series_equal(ds_result, pd_result)

    def test_assign_transform_filter(self):
        """Assign -> Transform -> Filter chain"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4]
        })

        pd_result = (pd_df
                     .assign(B=lambda x: x['A'] * 2)
                     .transform(lambda x: x + 1))
        ds_result = (ds_df
                     .assign(B=lambda x: x['A'] * 2)
                     .transform(lambda x: x + 1))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_returns_series_or_dataframe(self):
        """Verify agg() return type matches pandas"""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })

        # Single function returns Series
        pd_single = pd_df.agg('sum')
        ds_single = ds_df.agg('sum')
        assert type(pd_single).__name__ == type(ds_single).__name__

        # Multiple functions returns DataFrame
        pd_multi = pd_df.agg(['sum', 'mean'])
        ds_multi = ds_df.agg(['sum', 'mean'])
        # Both should be DataFrame-like
        assert hasattr(pd_multi, 'columns')
        assert hasattr(ds_multi, 'columns')


class TestEdgeCasesWithNulls:
    """Test apply/transform/agg with NULL values"""

    def test_apply_with_nulls(self):
        """apply() correctly handles NULL values"""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan]
        })
        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan]
        })

        pd_result = pd_df.apply(np.nansum)
        ds_result = ds_df.apply(np.nansum)

        assert_series_equal(ds_result, pd_result)

    def test_transform_with_fillna(self):
        """transform() with fillna-like operation"""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan]
        })
        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan]
        })

        pd_result = pd_df.transform(lambda x: x.fillna(x.mean()))
        ds_result = ds_df.transform(lambda x: x.fillna(x.mean()))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_with_nulls(self):
        """agg() with NULL values"""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan]
        })
        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, np.nan]
        })

        pd_result = pd_df.agg(['sum', 'count'])
        ds_result = ds_df.agg(['sum', 'count'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEmptyDataFrameEdgeCases:
    """Test apply/transform/agg with empty DataFrames"""

    def test_apply_empty_dataframe(self):
        """apply() on empty DataFrame"""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        pd_result = pd_df.apply(sum, axis=0)
        ds_result = ds_df.apply(sum, axis=0)

        assert_series_equal(ds_result, pd_result)

    def test_agg_empty_dataframe(self):
        """agg() on empty DataFrame"""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        pd_result = pd_df.agg('sum')
        ds_result = ds_df.agg('sum')

        assert_series_equal(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
