"""
Exploratory Batch 68: Callable Indexing, at/iat Accessors, Complex Chain Operations

This batch explores undertested boundary conditions:
1. Callable indexing with lambda functions
2. at/iat accessor edge cases
3. Complex chains with mixed operations after filter
4. Multi-column interdependent calculations
5. Chained groupby + agg + column modifications

Discovery method: Architecture-based exploration after reviewing pandas_compat.py and core.py
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
from pandas.testing import assert_frame_equal, assert_series_equal


class TestCallableIndexing:
    """Test callable (lambda) based indexing."""

    def test_callable_row_selection(self):
        """Test selecting rows with a callable."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[lambda df: df['a'] > 2]
        ds_result = ds_df[lambda df: df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_column_selection(self):
        """Test selecting columns with a callable."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        # Select columns that start with 'a' or 'b'
        pd_result = pd_df[[col for col in pd_df.columns if col in ['a', 'b']]]
        ds_result = ds_df[[col for col in ds_df.columns if col in ['a', 'b']]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_with_multiple_conditions(self):
        """Test callable with complex condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[lambda df: (df['a'] > 2) & (df['b'] < 4)]
        ds_result = ds_df[lambda df: (df['a'] > 2) & (df['b'] < 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_after_column_assignment(self):
        """Test callable indexing after column assignment."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = pd_df['a'] * 2
        ds_df['b'] = ds_df['a'] * 2

        pd_result = pd_df[lambda df: df['b'] > 5]
        ds_result = ds_df[lambda df: df['b'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_callable_chained(self):
        """Test chained callable indexing."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': range(10)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[lambda df: df['a'] > 3][lambda df: df['b'] < 7]
        ds_result = ds_df[lambda df: df['a'] > 3][lambda df: df['b'] < 7]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAtIatAccessors:
    """Test at and iat accessor edge cases."""

    def test_at_basic_access(self):
        """Test basic at accessor."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.at['y', 'a']
        ds_result = ds_df.at['y', 'a']

        assert pd_result == ds_result

    def test_iat_basic_access(self):
        """Test basic iat accessor."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iat[1, 0]
        ds_result = ds_df.iat[1, 0]

        assert pd_result == ds_result

    def test_at_with_integer_index(self):
        """Test at accessor with integer index."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.at[0, 'a']
        ds_result = ds_df.at[0, 'a']

        assert pd_result == ds_result

    def test_iat_last_element(self):
        """Test iat accessor for last element."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iat[-1, -1]
        ds_result = ds_df.iat[-1, -1]

        assert pd_result == ds_result

    def test_at_after_filter(self):
        """Test at accessor after filtering."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['a'] > 2]
        ds_filtered = ds_df[ds_df['a'] > 2]

        # Access specific element by index
        pd_result = pd_filtered.at[3, 'b']
        ds_result = ds_filtered.at[3, 'b']

        assert pd_result == ds_result


class TestComplexChainOperations:
    """Test complex operation chains."""

    def test_filter_then_multiple_assigns(self):
        """Test multiple column assignments after filtering."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        ds_filtered = ds_df[ds_df['a'] > 2]

        pd_filtered['c'] = pd_filtered['a'] + pd_filtered['b']
        ds_filtered['c'] = ds_filtered['a'] + ds_filtered['b']

        pd_filtered['d'] = pd_filtered['c'] * 2
        ds_filtered['d'] = ds_filtered['c'] * 2

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_assign_then_filter_then_assign(self):
        """Test assign -> filter -> assign chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = pd_df['a'] * 2
        ds_df['b'] = ds_df['a'] * 2

        pd_filtered = pd_df[pd_df['b'] > 4].copy()
        ds_filtered = ds_df[ds_df['b'] > 4]

        pd_filtered['c'] = pd_filtered['a'] + pd_filtered['b']
        ds_filtered['c'] = ds_filtered['a'] + ds_filtered['b']

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_multiple_filters_then_assign(self):
        """Test multiple filters followed by assign."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['a'] > 3][pd_df['b'] < 6].copy()
        ds_filtered = ds_df[ds_df['a'] > 3][ds_df['b'] < 6]

        pd_filtered['sum'] = pd_filtered['a'] + pd_filtered['b']
        ds_filtered['sum'] = ds_filtered['a'] + ds_filtered['b']

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_head_then_assign(self):
        """Test column assignment after head()."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_head = pd_df.head(3).copy()
        ds_head = ds_df.head(3)

        pd_head['c'] = pd_head['a'] * 10
        ds_head['c'] = ds_head['a'] * 10

        assert_datastore_equals_pandas(ds_head, pd_head)

    def test_tail_then_assign(self):
        """Test column assignment after tail()."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_tail = pd_df.tail(3).copy()
        ds_tail = ds_df.tail(3)

        pd_tail['c'] = pd_tail['b'] / 10
        ds_tail['c'] = ds_tail['b'] / 10

        assert_datastore_equals_pandas(ds_tail, pd_tail)

    def test_sample_then_sort(self):
        """Test sort after sample (with seed for reproducibility)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [50, 40, 30, 20, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_sampled = pd_df.sample(3, random_state=42).sort_values('a')
        ds_sampled = ds_df.sample(3, random_state=42).sort_values('a')

        assert_datastore_equals_pandas(ds_sampled, pd_sampled)


class TestMultiColumnInterdependencies:
    """Test complex interdependent column calculations."""

    def test_cascading_column_calculations(self):
        """Test cascading column calculations: a -> b -> c -> d."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = pd_df['a'] + 1
        ds_df['b'] = ds_df['a'] + 1

        pd_df['c'] = pd_df['b'] * 2
        ds_df['c'] = ds_df['b'] * 2

        pd_df['d'] = pd_df['c'] + pd_df['a']
        ds_df['d'] = ds_df['c'] + ds_df['a']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_bidirectional_column_dependencies(self):
        """Test columns that depend on each other."""
        pd_df = pd.DataFrame({'x': [10, 20, 30], 'y': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        # First update depends on both
        pd_df['z'] = pd_df['x'] + pd_df['y']
        ds_df['z'] = ds_df['x'] + ds_df['y']

        # Update x using z
        pd_df['x'] = pd_df['x'] + pd_df['z']
        ds_df['x'] = ds_df['x'] + ds_df['z']

        # Update y using new x
        pd_df['y'] = pd_df['x'] - pd_df['z']
        ds_df['y'] = ds_df['x'] - ds_df['z']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_complex_arithmetic_chain(self):
        """Test complex arithmetic with multiple operations."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [4.0, 3.0, 2.0, 1.0]})
        ds_df = DataStore(pd_df.copy())

        pd_df['sum'] = pd_df['a'] + pd_df['b']
        ds_df['sum'] = ds_df['a'] + ds_df['b']

        pd_df['diff'] = pd_df['a'] - pd_df['b']
        ds_df['diff'] = ds_df['a'] - ds_df['b']

        pd_df['prod'] = pd_df['a'] * pd_df['b']
        ds_df['prod'] = ds_df['a'] * ds_df['b']

        pd_df['ratio'] = pd_df['a'] / pd_df['b']
        ds_df['ratio'] = ds_df['a'] / ds_df['b']

        pd_df['complex'] = (pd_df['sum'] * pd_df['diff']) / (pd_df['prod'] + 1)
        ds_df['complex'] = (ds_df['sum'] * ds_df['diff']) / (ds_df['prod'] + 1)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_column_self_update_multiple_times(self):
        """Test updating a column multiple times with itself."""
        pd_df = pd.DataFrame({'val': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        # Multiple self-updates
        for _ in range(5):
            pd_df['val'] = pd_df['val'] + 1
            ds_df['val'] = ds_df['val'] + 1

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestGroupbyAggChains:
    """Test groupby + aggregation + further operations."""

    def test_groupby_agg_then_filter(self):
        """Test filtering after groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore(pd_df.copy())

        pd_agg = pd_df.groupby('group')['value'].sum().reset_index()
        ds_agg = ds_df.groupby('group')['value'].sum().reset_index()

        pd_result = pd_agg[pd_agg['value'] > 40]
        ds_result = ds_agg[ds_agg['value'] > 40]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_then_assign(self):
        """Test column assignment after groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df.copy())

        pd_agg = pd_df.groupby('group')['value'].mean().reset_index()
        ds_agg = ds_df.groupby('group')['value'].mean().reset_index()

        pd_agg['doubled'] = pd_agg['value'] * 2
        ds_agg['doubled'] = ds_agg['value'] * 2

        assert_datastore_equals_pandas(ds_agg, pd_agg)

    def test_groupby_multiple_aggs(self):
        """Test multiple aggregations in groupby."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'val1': [1, 2, 3, 4, 5, 6],
            'val2': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group').agg({
            'val1': 'sum',
            'val2': 'mean'
        }).reset_index()
        ds_result = ds_df.groupby('group').agg({
            'val1': 'sum',
            'val2': 'mean'
        }).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_then_sort(self):
        """Test sorting after groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['C', 'A', 'B', 'A', 'C', 'B'],
            'value': [60, 10, 30, 20, 50, 40]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group')['value'].sum().reset_index().sort_values('value')
        ds_result = ds_df.groupby('group')['value'].sum().reset_index().sort_values('value')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCasesWithNulls:
    """Test edge cases involving null/NaN values."""

    def test_assign_with_nulls(self):
        """Test column assignment when source has nulls."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'] + pd_df['b']
        ds_df['c'] = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_filter_with_null_condition(self):
        """Test filtering with condition involving nulls."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_null_column(self):
        """Test assigning a column to None/NaN."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = None
        ds_df['c'] = None

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_chain_operations_with_nulls(self):
        """Test chain of operations when data has nulls."""
        pd_df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [None, 20, 30, None, 50]
        })
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'].fillna(0) + pd_df['b'].fillna(0)
        ds_df['c'] = ds_df['a'].fillna(0) + ds_df['b'].fillna(0)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestLocIlocAdvanced:
    """Test advanced loc/iloc scenarios."""

    def test_loc_with_boolean_and_column(self):
        """Test loc with boolean condition and column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.loc[pd_df['a'] > 2, ['b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 2, ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative_indices(self):
        """Test iloc with negative indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[-3:]
        ds_result = ds_df.iloc[-3:]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_slice(self):
        """Test loc with slice."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=['w', 'x', 'y', 'z', 'v'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.loc['x':'z']
        ds_result = ds_df.loc['x':'z']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_step(self):
        """Test iloc with step."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[::2]  # Every other row
        ds_result = ds_df.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMethodChaining:
    """Test pandas-style method chaining."""

    def test_filter_assign_sort_head(self):
        """Test filter -> assign -> sort -> head chain."""
        pd_df = pd.DataFrame({
            'a': [5, 2, 8, 1, 9, 3, 7, 4, 6, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = (pd_df[pd_df['a'] > 3]
                     .assign(c=lambda df: df['a'] * df['b'])
                     .sort_values('c')
                     .head(3))
        ds_result = (ds_df[ds_df['a'] > 3]
                     .assign(c=lambda df: df['a'] * df['b'])
                     .sort_values('c')
                     .head(3))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_fillna_chain(self):
        """Test dropna and fillna in chain."""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, None, 5],
            'b': [None, 2, None, 4, None]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.fillna({'a': 0}).dropna(subset=['b'])
        ds_result = ds_df.fillna({'a': 0}).dropna(subset=['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_drop_chain(self):
        """Test rename and drop in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y'}).drop(columns=['c'])
        ds_result = ds_df.rename(columns={'a': 'x', 'b': 'y'}).drop(columns=['c'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticOperatorChains:
    """Test chained arithmetic operations."""

    def test_addition_chain(self):
        """Test chained addition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'] + pd_df['b'] + pd_df['c']
        ds_result = ds_df['a'] + ds_df['b'] + ds_df['c']

        assert_series_equal(get_series(ds_result), pd_result)

    def test_mixed_arithmetic_chain(self):
        """Test mixed arithmetic operations."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = (pd_df['a'] + pd_df['b']) * 2 - 5
        ds_result = (ds_df['a'] + ds_df['b']) * 2 - 5

        assert_series_equal(get_series(ds_result), pd_result)

    def test_division_with_zero_handling(self):
        """Test division that might produce inf."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 0, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'] / pd_df['b']
        ds_result = ds_df['a'] / ds_df['b']

        assert_series_equal(get_series(ds_result), pd_result)


class TestSelectDtypesAdvanced:
    """Test select_dtypes edge cases."""

    def test_select_dtypes_include_number(self):
        """Test select_dtypes with numeric include."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0],
            'c': ['x', 'y', 'z']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.select_dtypes(include='number')
        ds_result = ds_df.select_dtypes(include='number')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_object(self):
        """Test select_dtypes with object exclude."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [True, False, True]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.select_dtypes(exclude='object')
        ds_result = ds_df.select_dtypes(exclude='object')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCountsAdvanced:
    """Test value_counts edge cases."""

    def test_value_counts_with_bins(self):
        """Test value_counts with bins parameter."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].value_counts(bins=3)
        ds_result = ds_df['a'].value_counts(bins=3)

        # Compare with some tolerance due to potential floating point in bin edges
        assert_series_equal(get_series(ds_result), pd_result, check_names=False, check_index_type=False)

    def test_value_counts_normalize_with_dropna(self):
        """Test value_counts with normalize and dropna."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'x', None, 'y', 'y']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].value_counts(normalize=True, dropna=True)
        ds_result = ds_df['a'].value_counts(normalize=True, dropna=True)

        assert_series_equal(get_series(ds_result), pd_result)


class TestNuniqueAdvanced:
    """Test nunique edge cases."""

    def test_nunique_with_axis(self):
        """Test nunique along different axis."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': [1, 2, 2],
            'c': [1, 1, 1]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.nunique(axis=0)
        ds_result = ds_df.nunique(axis=0)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_nunique_per_column(self):
        """Test nunique per column."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': ['x', 'x', 'x', 'x']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(get_series(ds_result), pd_result)


class TestClipAdvanced:
    """Test clip edge cases."""

    def test_clip_with_dataframe_bounds(self):
        """Test clip with DataFrame as bounds."""
        pd_df = pd.DataFrame({'a': [1, 5, 10], 'b': [2, 6, 8]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.clip(lower=2, upper=7)
        ds_result = ds_df.clip(lower=2, upper=7)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_with_series(self):
        """Test clip on Series."""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].clip(lower=5, upper=15)
        ds_result = ds_df['a'].clip(lower=5, upper=15)

        assert_series_equal(get_series(ds_result), pd_result)
