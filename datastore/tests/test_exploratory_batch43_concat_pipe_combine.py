"""
Exploratory Batch 43: Concat, Pipe, Combine, and Logical Operations with Chains

This batch explores:
1. Concat with lazy operation chains
2. Pipe function in various scenarios
3. Combine/Combine_first edge cases
4. Complex logical operations (&, |, ~, ^) with chains
5. Add_prefix/Add_suffix lazy operations
6. Apply/Transform/Agg boundary cases
"""

import numpy as np
import pandas as pd
import pytest

import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas



# =============================================================================
# Section 1: Concat with Lazy Operation Chains
# =============================================================================


class TestConcatWithLazyChains:
    """Test concat function with subsequent lazy operations."""

    def test_concat_then_filter(self):
        """Concat followed by filter operation."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        pd_df2 = pd.DataFrame({'a': [4, 5, 6], 'b': ['p', 'q', 'r']})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_result[pd_result['a'] > 2]

        ds_df1 = ds.DataStore({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df2 = ds.DataStore({'a': [4, 5, 6], 'b': ['p', 'q', 'r']})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_groupby_agg(self):
        """Concat followed by groupby aggregation."""
        pd_df1 = pd.DataFrame({'cat': ['a', 'b', 'a'], 'val': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'cat': ['b', 'a', 'b'], 'val': [4, 5, 6]})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_result.groupby('cat')['val'].sum().reset_index()

        ds_df1 = ds.DataStore({'cat': ['a', 'b', 'a'], 'val': [1, 2, 3]})
        ds_df2 = ds.DataStore({'cat': ['b', 'a', 'b'], 'val': [4, 5, 6]})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_result.groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_concat_then_sort_head(self):
        """Concat followed by sort and head."""
        pd_df1 = pd.DataFrame({'val': [3, 1, 4]})
        pd_df2 = pd.DataFrame({'val': [1, 5, 9]})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_result.sort_values('val').head(4)

        ds_df1 = ds.DataStore({'val': [3, 1, 4]})
        ds_df2 = ds.DataStore({'val': [1, 5, 9]})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_result.sort_values('val').head(4)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_axis1_then_filter(self):
        """Concat along axis=1 followed by filter."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'b': [4, 5, 6]})
        pd_result = pd.concat([pd_df1, pd_df2], axis=1)
        pd_result = pd_result[pd_result['a'] > 1]

        ds_df1 = ds.DataStore({'a': [1, 2, 3]})
        ds_df2 = ds.DataStore({'b': [4, 5, 6]})
        ds_result = ds.concat([ds_df1, ds_df2], axis=1)
        ds_result = ds_result[ds_result['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_empty_dataframes(self):
        """Concat empty DataFrames."""
        pd_df1 = pd.DataFrame({'a': [], 'b': []})
        pd_df2 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)

        ds_df1 = ds.DataStore({'a': [], 'b': []})
        ds_df2 = ds.DataStore({'a': [1, 2], 'b': ['x', 'y']})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_single_dataframe(self):
        """Concat with single DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd.concat([pd_df], ignore_index=True)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds.concat([ds_df], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_mixed_types_columns(self):
        """Concat DataFrames with different column types."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [1.5, 2.5]})
        pd_df2 = pd.DataFrame({'a': [3, 4], 'b': [3.5, 4.5]})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_result[pd_result['a'] > 2]

        ds_df1 = ds.DataStore({'a': [1, 2], 'b': [1.5, 2.5]})
        ds_df2 = ds.DataStore({'a': [3, 4], 'b': [3.5, 4.5]})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_with_keys(self):
        """Concat with keys parameter."""
        pd_df1 = pd.DataFrame({'a': [1, 2]})
        pd_df2 = pd.DataFrame({'a': [3, 4]})
        pd_result = pd.concat([pd_df1, pd_df2], keys=['first', 'second'])

        ds_df1 = ds.DataStore({'a': [1, 2]})
        ds_df2 = ds.DataStore({'a': [3, 4]})
        ds_result = ds.concat([ds_df1, ds_df2], keys=['first', 'second'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 2: Pipe Function Tests
# =============================================================================


class TestPipeFunction:
    """Test pipe function in various scenarios."""

    def test_pipe_simple_function(self):
        """Pipe with a simple function."""
        def double_values(df):
            return df * 2

        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.pipe(double_values)

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.pipe(double_values)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_args(self):
        """Pipe with additional arguments."""
        def add_value(df, val):
            return df + val

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.pipe(add_value, 10)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.pipe(add_value, 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_kwargs(self):
        """Pipe with keyword arguments."""
        def multiply_cols(df, factor=1, col='a'):
            result = df.copy()
            result[col] = result[col] * factor
            return result

        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.pipe(multiply_cols, factor=3, col='a')

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.pipe(multiply_cols, factor=3, col='a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_chain(self):
        """Multiple pipe calls chained."""
        def add_one(df):
            return df + 1

        def double(df):
            return df * 2

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.pipe(add_one).pipe(double)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.pipe(add_one).pipe(double)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_after_filter(self):
        """Pipe after filter operation."""
        def sum_cols(df):
            return df.sum()

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        pd_result = pd_df[pd_df['a'] > 2].pipe(sum_cols)

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_result = ds_df[ds_df['a'] > 2].pipe(sum_cols)

        # Result should be a Series
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_returning_scalar(self):
        """Pipe function returning a scalar."""
        def total_sum(df):
            return df.values.sum()

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.pipe(total_sum)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.pipe(total_sum)

        assert pd_result == ds_result

    def test_pipe_tuple_callable(self):
        """Pipe with tuple (callable, data_keyword)."""
        def process(other_arg, data=None, multiplier=1):
            return data * multiplier + other_arg

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.pipe((process, 'data'), 5, multiplier=2)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.pipe((process, 'data'), 5, multiplier=2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 3: Combine and Combine_first Tests
# =============================================================================


class TestCombineOperations:
    """Test combine and combine_first operations."""

    def test_combine_basic(self):
        """Basic combine operation."""
        pd_df1 = pd.DataFrame({'a': [1, 2, np.nan], 'b': [4, np.nan, 6]})
        pd_df2 = pd.DataFrame({'a': [10, np.nan, 30], 'b': [np.nan, 50, 60]})
        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1.fillna(s2))

        ds_df1 = ds.DataStore({'a': [1, 2, np.nan], 'b': [4, np.nan, 6]})
        ds_df2 = ds.DataStore({'a': [10, np.nan, 30], 'b': [np.nan, 50, 60]})
        ds_result = ds_df1.combine(ds_df2, lambda s1, s2: s1.fillna(s2))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_with_fill_value(self):
        """Combine with fill_value parameter."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [10, 20], 'c': [30, 40]})
        pd_result = pd_df1.combine(pd_df2, np.minimum, fill_value=100)

        ds_df1 = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_df2 = ds.DataStore({'a': [10, 20], 'c': [30, 40]})
        ds_result = ds_df1.combine(ds_df2, np.minimum, fill_value=100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_basic(self):
        """Basic combine_first operation."""
        pd_df1 = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, np.nan]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df1.combine_first(pd_df2)

        ds_df1 = ds.DataStore({'a': [1, np.nan, 3], 'b': [np.nan, 2, np.nan]})
        ds_df2 = ds.DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_different_columns(self):
        """Combine_first with different columns."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, np.nan]})
        pd_df2 = pd.DataFrame({'b': [10, 20], 'c': [30, 40]})
        pd_result = pd_df1.combine_first(pd_df2)

        ds_df1 = ds.DataStore({'a': [1, 2], 'b': [3, np.nan]})
        ds_df2 = ds.DataStore({'b': [10, 20], 'c': [30, 40]})
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_then_filter(self):
        """Combine_first followed by filter."""
        pd_df1 = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, 3]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_result = pd_df1.combine_first(pd_df2)
        pd_result = pd_result[pd_result['a'] > 5]

        ds_df1 = ds.DataStore({'a': [1, np.nan, 3], 'b': [np.nan, 2, 3]})
        ds_df2 = ds.DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_result = ds_df1.combine_first(ds_df2)
        ds_result = ds_result[ds_result['a'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 4: Complex Logical Operations with Chains
# =============================================================================


class TestLogicalOperationsChains:
    """Test complex logical operations with chains."""

    def test_and_operation_chain(self):
        """Test & operation in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['b'] < 4)]

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_result = ds_df[(ds_df['a'] > 2) & (ds_df['b'] < 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_operation_chain(self):
        """Test | operation in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        pd_result = pd_df[(pd_df['a'] < 2) | (pd_df['a'] > 4)]

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        ds_result = ds_df[(ds_df['a'] < 2) | (ds_df['a'] > 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_not_operation_chain(self):
        """Test ~ (not) operation in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [True, False, True, False, True]})
        pd_result = pd_df[~pd_df['b']]

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': [True, False, True, False, True]})
        ds_result = ds_df[~ds_df['b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_xor_operation(self):
        """Test ^ (xor) operation."""
        pd_df = pd.DataFrame({'a': [True, True, False, False], 'b': [True, False, True, False]})
        pd_result = pd_df['a'] ^ pd_df['b']

        ds_df = ds.DataStore({'a': [True, True, False, False], 'b': [True, False, True, False]})
        ds_result = ds_df['a'] ^ ds_df['b']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_complex_logical_expression(self):
        """Test complex logical expression."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [True, False, True, False, True]
        })
        pd_result = pd_df[((pd_df['a'] > 2) & (pd_df['b'] < 40)) | pd_df['c']]

        ds_df = ds.DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [True, False, True, False, True]
        })
        ds_result = ds_df[((ds_df['a'] > 2) & (ds_df['b'] < 40)) | ds_df['c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_logical_then_groupby(self):
        """Logical filter followed by groupby."""
        pd_df = pd.DataFrame({
            'cat': ['a', 'b', 'a', 'b', 'a'],
            'val': [1, 2, 3, 4, 5]
        })
        pd_result = pd_df[(pd_df['val'] > 1) & (pd_df['val'] < 5)].groupby('cat')['val'].sum().reset_index()

        ds_df = ds.DataStore({
            'cat': ['a', 'b', 'a', 'b', 'a'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_result = ds_df[(ds_df['val'] > 1) & (ds_df['val'] < 5)].groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_logical_with_isin(self):
        """Logical operation with isin."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        pd_result = pd_df[(pd_df['a'].isin([1, 3, 5])) & (pd_df['b'] != 'z')]

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        ds_result = ds_df[(ds_df['a'].isin([1, 3, 5])) & (ds_df['b'] != 'z')]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 5: Add_prefix and Add_suffix Lazy Operations
# =============================================================================


class TestPrefixSuffixOperations:
    """Test add_prefix and add_suffix operations."""

    def test_add_prefix_basic(self):
        """Basic add_prefix operation."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_result = pd_df.add_prefix('col_')

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df.add_prefix('col_')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_suffix_basic(self):
        """Basic add_suffix operation."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_result = pd_df.add_suffix('_val')

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df.add_suffix('_val')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix_then_filter(self):
        """Add_prefix followed by filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.add_prefix('col_')
        pd_result = pd_result[pd_result['col_a'] > 1]

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.add_prefix('col_')
        ds_result = ds_result[ds_result['col_a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_suffix_then_groupby(self):
        """Add_suffix followed by groupby."""
        pd_df = pd.DataFrame({'cat': ['a', 'b', 'a'], 'val': [1, 2, 3]})
        pd_result = pd_df.add_suffix('_x')
        pd_result = pd_result.groupby('cat_x')['val_x'].sum().reset_index()

        ds_df = ds.DataStore({'cat': ['a', 'b', 'a'], 'val': [1, 2, 3]})
        ds_result = ds_df.add_suffix('_x')
        ds_result = ds_result.groupby('cat_x')['val_x'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_prefix_suffix_chain(self):
        """Chain add_prefix and add_suffix."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_result = pd_df.add_prefix('pre_').add_suffix('_post')

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df.add_prefix('pre_').add_suffix('_post')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_prefix(self):
        """Filter followed by add_prefix."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        pd_result = pd_df[pd_df['a'] > 2].add_prefix('x_')

        ds_df = ds.DataStore({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        ds_result = ds_df[ds_df['a'] > 2].add_prefix('x_')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 6: Apply, Transform, and Agg Boundary Cases
# =============================================================================


class TestApplyTransformAgg:
    """Test apply, transform, and agg boundary cases."""

    def test_apply_simple_function(self):
        """Apply a simple function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.apply(lambda x: x * 2)

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.apply(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_axis1(self):
        """Apply along axis=1."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.apply(lambda row: row['a'] + row['b'], axis=1)

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.apply(lambda row: row['a'] + row['b'], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_returning_series(self):
        """Apply function returning Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.apply(lambda col: pd.Series({'min': col.min(), 'max': col.max()}))

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.apply(lambda col: pd.Series({'min': col.min(), 'max': col.max()}))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_simple(self):
        """Simple transform operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.transform(lambda x: x - x.mean())

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.transform(lambda x: x - x.mean())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_with_string_func(self):
        """Transform with string function name."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.transform('sqrt')

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.transform('sqrt')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_multiple_functions(self):
        """Agg with multiple functions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.agg(['sum', 'mean', 'max'])

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.agg(['sum', 'mean', 'max'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_dict_of_functions(self):
        """Agg with dict of functions per column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.agg({'a': 'sum', 'b': 'mean'})

        ds_df = ds.DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds_df.agg({'a': 'sum', 'b': 'mean'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_after_filter(self):
        """Apply after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        pd_result = pd_df[pd_df['a'] > 2].apply(lambda x: x * 2)

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_result = ds_df[ds_df['a'] > 2].apply(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 7: Map and Applymap Operations
# =============================================================================


class TestMapApplymap:
    """Test map and applymap operations."""

    def test_map_dict(self):
        """Map with dictionary."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'z']})
        pd_result = pd_df['a'].map({'x': 1, 'y': 2, 'z': 3})

        ds_df = ds.DataStore({'a': ['x', 'y', 'z']})
        ds_result = ds_df['a'].map({'x': 1, 'y': 2, 'z': 3})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_function(self):
        """Map with function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df['a'].map(lambda x: x ** 2)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df['a'].map(lambda x: x ** 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_applymap_basic(self):
        """Basic applymap operation."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_result = pd_df.applymap(lambda x: x * 10)

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df.applymap(lambda x: x * 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_applymap_with_na_action(self):
        """Applymap with na_action parameter."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        pd_result = pd_df.applymap(lambda x: x * 2, na_action='ignore')

        ds_df = ds.DataStore({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
        ds_result = ds_df.applymap(lambda x: x * 2, na_action='ignore')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_after_filter(self):
        """Map after filter operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['w', 'x', 'y', 'z']})
        pd_result = pd_df[pd_df['a'] > 2]['b'].map(str.upper)

        ds_df = ds.DataStore({'a': [1, 2, 3, 4], 'b': ['w', 'x', 'y', 'z']})
        ds_result = ds_df[ds_df['a'] > 2]['b'].map(str.upper)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 8: Reindex and Align Operations
# =============================================================================


class TestReindexAlign:
    """Test reindex and align operations."""

    def test_reindex_basic(self):
        """Basic reindex operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        pd_result = pd_df.reindex(['x', 'z', 'w'])

        ds_df = ds.DataStore({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_result = ds_df.reindex(['x', 'z', 'w'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """Reindex columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        pd_result = pd_df.reindex(columns=['c', 'a', 'd'])

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_result = ds_df.reindex(columns=['c', 'a', 'd'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill_value(self):
        """Reindex with fill_value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_result = pd_df.reindex([0, 1, 2, 3, 4], fill_value=0)

        ds_df = ds.DataStore({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds_result = ds_df.reindex([0, 1, 2, 3, 4], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_then_filter(self):
        """Reindex followed by filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=[0, 1, 2, 3])
        pd_result = pd_df.reindex([3, 2, 1, 0])
        pd_result = pd_result[pd_result['a'] > 2]

        ds_df = ds.DataStore({'a': [1, 2, 3, 4]}, index=[0, 1, 2, 3])
        ds_result = ds_df.reindex([3, 2, 1, 0])
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_align_basic(self):
        """Basic align operation."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'b': [4, 5, 6]}, index=[1, 2, 3])
        pd_res1, pd_res2 = pd_df1.align(pd_df2)

        ds_df1 = ds.DataStore({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds_df2 = ds.DataStore({'b': [4, 5, 6]}, index=[1, 2, 3])
        ds_res1, ds_res2 = ds_df1.align(ds_df2)

        assert_datastore_equals_pandas(ds_res1, pd_res1)
        assert_datastore_equals_pandas(ds_res2, pd_res2)


# =============================================================================
# Section 9: Take, Truncate, and Set_axis Operations
# =============================================================================


class TestTakeTruncateSetAxis:
    """Test take, truncate, and set_axis operations."""

    def test_take_basic(self):
        """Basic take operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df.take([0, 2, 4])

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds_df.take([0, 2, 4])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_take_negative_indices(self):
        """Take with negative indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df.take([-1, -2, -3])

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds_df.take([-1, -2, -3])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_take_then_filter(self):
        """Take followed by filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e']})
        pd_result = pd_df.take([1, 2, 3])
        pd_result = pd_result[pd_result['a'] > 2]

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': ['a', 'b', 'c', 'd', 'e']})
        ds_result = ds_df.take([1, 2, 3])
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truncate_basic(self):
        """Basic truncate operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        pd_result = pd_df.truncate(before=2, after=4)

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        ds_result = ds_df.truncate(before=2, after=4)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_columns(self):
        """Set axis for columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_result = pd_df.set_axis(['x', 'y'], axis=1)

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df.set_axis(['x', 'y'], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_index(self):
        """Set axis for index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.set_axis(['x', 'y', 'z'], axis=0)

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.set_axis(['x', 'y', 'z'], axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 10: Squeeze and Other Edge Cases
# =============================================================================


class TestSqueezeAndEdgeCases:
    """Test squeeze and other edge cases."""

    def test_squeeze_single_column(self):
        """Squeeze single column DataFrame to Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.squeeze()

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.squeeze()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_squeeze_single_row(self):
        """Squeeze single row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        pd_result = pd_df.squeeze()

        ds_df = ds.DataStore({'a': [1], 'b': [2], 'c': [3]})
        ds_result = ds_df.squeeze()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_squeeze_scalar(self):
        """Squeeze to scalar."""
        pd_df = pd.DataFrame({'a': [42]})
        pd_result = pd_df.squeeze()

        ds_df = ds.DataStore({'a': [42]})
        ds_result = ds_df.squeeze()

        assert pd_result == ds_result

    def test_squeeze_no_change(self):
        """Squeeze on multi-row multi-column returns same shape."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_result = pd_df.squeeze()

        ds_df = ds.DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_result = ds_df.squeeze()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_keys_method(self):
        """Test keys() method returns columns."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        pd_result = pd_df.keys()

        ds_df = ds.DataStore({'a': [1], 'b': [2], 'c': [3]})
        ds_result = ds_df.keys()

        assert list(pd_result) == list(ds_result)

    def test_droplevel_basic(self):
        """Basic droplevel on MultiIndex."""
        arrays = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=index)
        pd_result = pd_df.droplevel('first')

        ds_df = ds.DataStore({'val': [10, 20, 30, 40]}, index=index)
        ds_result = ds_df.droplevel('first')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 11: Complex Multi-Operation Chains
# =============================================================================


class TestComplexChains:
    """Test complex multi-operation chains."""

    def test_filter_prefix_groupby_agg(self):
        """Filter -> add_prefix -> groupby -> agg chain."""
        pd_df = pd.DataFrame({
            'category': ['a', 'b', 'a', 'b', 'a'],
            'value': [1, 2, 3, 4, 5]
        })
        pd_result = pd_df[pd_df['value'] > 1].add_prefix('x_')
        pd_result = pd_result.groupby('x_category')['x_value'].sum().reset_index()

        ds_df = ds.DataStore({
            'category': ['a', 'b', 'a', 'b', 'a'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_result = ds_df[ds_df['value'] > 1].add_prefix('x_')
        ds_result = ds_result.groupby('x_category')['x_value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_concat_filter_sort_head(self):
        """Concat -> filter -> sort -> head chain."""
        pd_df1 = pd.DataFrame({'val': [3, 1, 4, 1]})
        pd_df2 = pd.DataFrame({'val': [5, 9, 2, 6]})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_result[pd_result['val'] > 2].sort_values('val').head(3)

        ds_df1 = ds.DataStore({'val': [3, 1, 4, 1]})
        ds_df2 = ds.DataStore({'val': [5, 9, 2, 6]})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)
        ds_result = ds_result[ds_result['val'] > 2].sort_values('val').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_transform_agg_chain(self):
        """Pipe -> transform -> agg chain."""
        def normalize(df):
            return (df - df.mean()) / df.std()

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        pd_result = pd_df.pipe(normalize)
        pd_agg = pd_result.agg(['mean', 'std'])

        ds_df = ds.DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_result = ds_df.pipe(normalize)
        ds_agg = ds_result.agg(['mean', 'std'])

        assert_datastore_equals_pandas(ds_agg, pd_agg)

    def test_logical_filter_combine_first_groupby(self):
        """Logical filter -> combine_first -> groupby chain."""
        pd_df1 = pd.DataFrame({
            'cat': ['a', 'b', 'a', 'b'],
            'val': [1, np.nan, 3, np.nan]
        })
        pd_df2 = pd.DataFrame({
            'cat': ['a', 'b', 'a', 'b'],
            'val': [10, 20, 30, 40]
        })
        pd_combined = pd_df1.combine_first(pd_df2)
        pd_result = pd_combined[pd_combined['val'] > 5].groupby('cat')['val'].sum().reset_index()

        ds_df1 = ds.DataStore({
            'cat': ['a', 'b', 'a', 'b'],
            'val': [1, np.nan, 3, np.nan]
        })
        ds_df2 = ds.DataStore({
            'cat': ['a', 'b', 'a', 'b'],
            'val': [10, 20, 30, 40]
        })
        ds_combined = ds_df1.combine_first(ds_df2)
        ds_result = ds_combined[ds_combined['val'] > 5].groupby('cat')['val'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_take_apply_suffix_sort(self):
        """Take -> apply -> add_suffix -> sort chain."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2], 'b': [50, 30, 10, 40, 20]})
        pd_result = pd_df.take([0, 2, 4])
        pd_result = pd_result.apply(lambda x: x * 2)
        pd_result = pd_result.add_suffix('_doubled')
        pd_result = pd_result.sort_values('a_doubled')

        ds_df = ds.DataStore({'a': [5, 3, 1, 4, 2], 'b': [50, 30, 10, 40, 20]})
        ds_result = ds_df.take([0, 2, 4])
        ds_result = ds_result.apply(lambda x: x * 2)
        ds_result = ds_result.add_suffix('_doubled')
        ds_result = ds_result.sort_values('a_doubled')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 12: Empty DataFrame Edge Cases
# =============================================================================


class TestEmptyDataFrameEdgeCases:
    """Test operations on empty DataFrames."""

    def test_concat_all_empty(self):
        """Concat all empty DataFrames."""
        pd_df1 = pd.DataFrame({'a': [], 'b': []})
        pd_df2 = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)

        ds_df1 = ds.DataStore({'a': [], 'b': []})
        ds_df2 = ds.DataStore({'a': [], 'b': []})
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_filter(self):
        """Empty DataFrame after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df[pd_df['a'] > 100]

        ds_df = ds.DataStore({'a': [1, 2, 3]})
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_on_empty(self):
        """Pipe on empty DataFrame."""
        def double_df(df):
            return df * 2

        pd_df = pd.DataFrame({'a': []})
        pd_result = pd_df.pipe(double_df)

        ds_df = ds.DataStore({'a': []})
        ds_result = ds_df.pipe(double_df)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix_empty(self):
        """Add_prefix on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd_df.add_prefix('col_')

        ds_df = ds.DataStore({'a': [], 'b': []})
        ds_result = ds_df.add_prefix('col_')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_empty(self):
        """Transform on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd_df.transform(lambda x: x * 2)

        ds_df = ds.DataStore({'a': [], 'b': []})
        ds_result = ds_df.transform(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_squeeze_empty(self):
        """Squeeze empty DataFrame."""
        pd_df = pd.DataFrame({'a': []})
        pd_result = pd_df.squeeze()

        ds_df = ds.DataStore({'a': []})
        ds_result = ds_df.squeeze()

        # Empty squeeze returns empty Series
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
