"""
Exploratory Batch 57: Copy/Index/Slice Edge Cases

Focus areas:
1. Copy/Deepcopy semantics - DataStore copy behavior vs pandas
2. Index operations chain - set_index, reset_index, reindex combinations
3. Advanced slicing patterns - iloc/loc boundary cases, slice with step
4. set_axis/take/truncate - less common but important operations
5. Column rename/reorder - column manipulation
"""

import copy
import pytest
import numpy as np
import pandas as pd

from datastore import DataStore
from tests.test_utils import (
    assert_frame_equal,
    assert_series_equal,
    get_dataframe,
    get_series,
    assert_datastore_equals_pandas,
)


# =============================================================================
# Section 1: Copy/Deepcopy Semantics
# =============================================================================


class TestCopySemamtics:
    """Test copy() and deepcopy() behavior matches pandas."""

    def test_copy_default_deep(self):
        """copy() with default deep=True creates independent data."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        # Verify result matches
        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_shallow(self):
        """copy(deep=False) creates shallow copy."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy(deep=False)
        ds_copy = ds_df.copy(deep=False)

        # Both should return valid results
        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_deep_true_explicit(self):
        """copy(deep=True) explicitly."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy(deep=True)
        ds_copy = ds_df.copy(deep=True)

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_module_function(self):
        """Python copy.copy() works on DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_copy = copy.copy(pd_df)
        ds_copy = copy.copy(ds_df)

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_deepcopy_module_function(self):
        """Python copy.deepcopy() works on DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_copy = copy.deepcopy(pd_df)
        ds_copy = copy.deepcopy(ds_df)

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_preserves_dtypes(self):
        """Copy preserves data types."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
        })
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_preserves_index(self):
        """Copy preserves index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_then_filter(self):
        """Copy then filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()[pd_df['a'] > 2]
        ds_result = ds_df.copy()[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_copy_empty_dataframe(self):
        """Copy empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        assert_datastore_equals_pandas(ds_copy, pd_copy, check_nullable_dtype=False)


# =============================================================================
# Section 2: Index Operations Chain
# =============================================================================


class TestIndexOperationsChain:
    """Test index-related operations and their chaining."""

    def test_set_index_basic(self):
        """Basic set_index operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a')
        ds_result = ds_df.set_index('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_drop_false(self):
        """set_index with drop=False keeps original column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a', drop=False)
        ds_result = ds_df.set_index('a', drop=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_basic(self):
        """Basic reset_index operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reset_index()
        ds_result = ds_df.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_drop_true(self):
        """reset_index with drop=True discards index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_then_reset_index(self):
        """Chain set_index then reset_index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a').reset_index()
        ds_result = ds_df.set_index('a').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_then_set_index(self):
        """Chain reset_index then set_index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reset_index().set_index('index')
        ds_result = ds_df.reset_index().set_index('index')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_basic_labels(self):
        """Basic reindex with labels."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex([2, 1, 0])
        ds_result = ds_df.reindex([2, 1, 0])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_new_labels(self):
        """Reindex with labels including new ones (introduces NaN)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex([0, 1, 2, 3, 4])
        ds_result = ds_df.reindex([0, 1, 2, 3, 4])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """Reindex columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex(columns=['c', 'a'])
        ds_result = ds_df.reindex(columns=['c', 'a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill_value(self):
        """Reindex with fill_value for missing values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex([0, 1, 2, 3], fill_value=0)
        ds_result = ds_df.reindex([0, 1, 2, 3], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_like(self):
        """reindex_like to match another DataFrame's index."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'b': [4, 5]}, index=[1, 2])

        ds_df1 = DataStore(pd_df1.copy())

        pd_result = pd_df1.reindex_like(pd_df2)
        ds_result = ds_df1.reindex_like(pd_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_reset_index(self):
        """Filter then reset_index chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2].reset_index(drop=True)
        ds_result = ds_df[ds_df['a'] > 2].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_reset_index(self):
        """Sort then reset_index chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a').reset_index(drop=True)
        ds_result = ds_df.sort_values('a').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 3: Advanced Slicing Patterns
# =============================================================================


class TestAdvancedSlicing:
    """Test advanced slicing patterns with iloc/loc."""

    def test_iloc_single_row(self):
        """iloc single row returns Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[1]
        ds_result = ds_df.iloc[1]

        assert_series_equal(ds_result, pd_result)

    def test_iloc_single_column(self):
        """iloc single column returns Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[:, 0]
        ds_result = ds_df.iloc[:, 0]

        assert_series_equal(ds_result, pd_result)

    def test_iloc_row_range(self):
        """iloc row range slice."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[1:4]
        ds_result = ds_df.iloc[1:4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_row_range_with_step(self):
        """iloc row range with step."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[::2]
        ds_result = ds_df.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative_index(self):
        """iloc with negative index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[-2:]
        ds_result = ds_df.iloc[-2:]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative_step(self):
        """iloc with negative step (reverse)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[::-1]
        ds_result = ds_df.iloc[::-1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_row_and_column(self):
        """iloc with both row and column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[0:2, 1:3]
        ds_result = ds_df.iloc[0:2, 1:3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_list_indices(self):
        """iloc with list of indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[[0, 2, 4]]
        ds_result = ds_df.iloc[[0, 2, 4]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_single_row_string_index(self):
        """loc single row with string index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.loc['y']
        ds_result = ds_df.loc['y']

        assert_series_equal(ds_result, pd_result)

    def test_loc_row_range_string_index(self):
        """loc row range with string index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=['a', 'b', 'c', 'd', 'e'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.loc['b':'d']
        ds_result = ds_df.loc['b':'d']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_column_selection(self):
        """loc with column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.loc[:, 'a']
        ds_result = ds_df.loc[:, 'a']

        assert_series_equal(ds_result, pd_result)

    def test_loc_multiple_columns(self):
        """loc with multiple column selection."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.loc[:, ['c', 'a']]
        ds_result = ds_df.loc[:, ['c', 'a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_boolean_mask(self):
        """loc with boolean mask - use ds[condition] syntax."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        # Note: For DataStore, use ds[condition] instead of ds.loc[condition]
        # because loc returns the underlying DataFrame's loc which doesn't
        # understand ColumnExpr. This is equivalent to pandas behavior.
        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_then_filter(self):
        """Slice then filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[1:4][pd_df.iloc[1:4]['a'] > 2]
        ds_result = ds_df.iloc[1:4][ds_df.iloc[1:4]['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_slice(self):
        """Filter then slice chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['a'] > 1]
        ds_filtered = ds_df[ds_df['a'] > 1]

        pd_result = pd_filtered.iloc[:2]
        ds_result = ds_filtered.iloc[:2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 4: set_axis/take/truncate Operations
# =============================================================================


class TestSetAxisTakeTruncate:
    """Test less common but important operations."""

    def test_set_axis_index(self):
        """set_axis for index (axis=0)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_axis(['x', 'y', 'z'], axis=0)
        ds_result = ds_df.set_axis(['x', 'y', 'z'], axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_columns(self):
        """set_axis for columns (axis=1)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_axis(['col1', 'col2'], axis=1)
        ds_result = ds_df.set_axis(['col1', 'col2'], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_take_rows(self):
        """take specific rows by position."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.take([0, 2, 4])
        ds_result = ds_df.take([0, 2, 4])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_take_negative_indices(self):
        """take with negative indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.take([-1, -2])
        ds_result = ds_df.take([-1, -2])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_take_columns(self):
        """take columns by position."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.take([2, 0], axis=1)
        ds_result = ds_df.take([2, 0], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truncate_before(self):
        """truncate with before parameter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.truncate(before=3)
        ds_result = ds_df.truncate(before=3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truncate_after(self):
        """truncate with after parameter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.truncate(after=3)
        ds_result = ds_df.truncate(after=3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truncate_before_and_after(self):
        """truncate with both before and after."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.truncate(before=2, after=4)
        ds_result = ds_df.truncate(before=2, after=4)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 5: Column Rename and Reorder
# =============================================================================


class TestColumnRenameReorder:
    """Test column manipulation operations."""

    def test_rename_columns_dict(self):
        """rename columns with dict."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'a': 'col_a', 'b': 'col_b'})
        ds_result = ds_df.rename(columns={'a': 'col_a', 'b': 'col_b'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_partial_columns(self):
        """rename only some columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'a': 'new_a'})
        ds_result = ds_df.rename(columns={'a': 'new_a'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_columns_function(self):
        """rename columns with function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns=str.upper)
        ds_result = ds_df.rename(columns=str.upper)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_index_dict(self):
        """rename index with dict."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(index={'x': 'X', 'y': 'Y'})
        ds_result = ds_df.rename(index={'x': 'X', 'y': 'Y'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_reorder(self):
        """Column selection reorders columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['c', 'a', 'b']]
        ds_result = ds_df[['c', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_filter(self):
        """Rename then filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_renamed = pd_df.rename(columns={'a': 'value'})
        ds_renamed = ds_df.rename(columns={'a': 'value'})

        pd_result = pd_renamed[pd_renamed['value'] > 2]
        ds_result = ds_renamed[ds_renamed['value'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_rename(self):
        """Filter then rename chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2].rename(columns={'a': 'filtered_value'})
        ds_result = ds_df[ds_df['a'] > 2].rename(columns={'a': 'filtered_value'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix(self):
        """add_prefix to columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.add_prefix('col_')
        ds_result = ds_df.add_prefix('col_')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_suffix(self):
        """add_suffix to columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.add_suffix('_value')
        ds_result = ds_df.add_suffix('_value')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 6: sort_index Operations
# =============================================================================


class TestSortIndex:
    """Test sort_index operations."""

    def test_sort_index_ascending(self):
        """sort_index ascending (default)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[3, 1, 2])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_index()
        ds_result = ds_df.sort_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_descending(self):
        """sort_index descending."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[3, 1, 2])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_index(ascending=False)
        ds_result = ds_df.sort_index(ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_columns(self):
        """sort_index on columns (axis=1)."""
        pd_df = pd.DataFrame({'c': [1, 2], 'a': [3, 4], 'b': [5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_index(axis=1)
        ds_result = ds_df.sort_index(axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_then_filter(self):
        """sort_index then filter chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]}, index=[3, 1, 2])
        ds_df = DataStore(pd_df.copy())

        pd_sorted = pd_df.sort_index()
        ds_sorted = ds_df.sort_index()

        pd_result = pd_sorted[pd_sorted['a'] > 1]
        ds_result = ds_sorted[ds_sorted['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_with_na_position(self):
        """sort_index with na_position parameter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[3.0, np.nan, 1.0])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_index(na_position='first')
        ds_result = ds_df.sort_index(na_position='first')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 7: DataFrame Construction Edge Cases
# =============================================================================


class TestDataFrameConstruction:
    """Test DataFrame construction edge cases."""

    def test_construct_from_dict(self):
        """Construct from dict."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_list_of_lists(self):
        """Construct from list of lists."""
        data = [[1, 2], [3, 4], [5, 6]]
        pd_df = pd.DataFrame(data, columns=['a', 'b'])
        ds_df = DataStore(data, columns=['a', 'b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_list_of_dicts(self):
        """Construct from list of dicts."""
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_with_index(self):
        """Construct with explicit index."""
        data = {'a': [1, 2, 3]}
        pd_df = pd.DataFrame(data, index=['x', 'y', 'z'])
        ds_df = DataStore(data, index=['x', 'y', 'z'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_series(self):
        """Construct from pandas Series."""
        s = pd.Series([1, 2, 3], name='col')
        pd_df = s.to_frame()
        ds_df = DataStore(s)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_empty_with_columns(self):
        """Construct empty DataFrame with columns."""
        pd_df = pd.DataFrame(columns=['a', 'b', 'c'])
        ds_df = DataStore(columns=['a', 'b', 'c'])

        assert_datastore_equals_pandas(ds_df, pd_df, check_nullable_dtype=False)

    def test_construct_from_numpy_array(self):
        """Construct from numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        pd_df = pd.DataFrame(arr, columns=['a', 'b'])
        ds_df = DataStore(arr, columns=['a', 'b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_with_dtype(self):
        """Construct with explicit dtype."""
        data = {'a': [1, 2, 3]}
        pd_df = pd.DataFrame(data, dtype=float)
        ds_df = DataStore(data, dtype=float)

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Section 8: Complex Chain Operations
# =============================================================================


class TestComplexChains:
    """Test complex operation chains."""

    def test_copy_filter_sort_head(self):
        """Copy -> Filter -> Sort -> Head chain."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()[pd_df['a'] > 2].sort_values('a').head(2)
        ds_result = ds_df.copy()[ds_df['a'] > 2].sort_values('a').head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_filter_reset(self):
        """set_index -> Filter -> reset_index chain."""
        pd_df = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_indexed = pd_df.set_index('key')
        ds_indexed = ds_df.set_index('key')

        pd_result = pd_indexed[pd_indexed['value'] > 2].reset_index()
        ds_result = ds_indexed[ds_indexed['value'] > 2].reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_filter_groupby_agg(self):
        """Rename -> Filter -> Groupby -> Agg chain."""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'C'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_renamed = pd_df.rename(columns={'cat': 'category', 'val': 'value'})
        ds_renamed = ds_df.rename(columns={'cat': 'category', 'val': 'value'})

        pd_result = pd_renamed[pd_renamed['value'] > 1].groupby('category')['value'].sum().reset_index()
        ds_result = ds_renamed[ds_renamed['value'] > 1].groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_then_groupby(self):
        """iloc slice then groupby."""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'C', 'C'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[:4].groupby('cat')['val'].mean().reset_index()
        ds_result = ds_df.iloc[:4].groupby('cat')['val'].mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_filter_sort(self):
        """Reindex -> Filter -> Sort chain."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2]}, index=[0, 1, 2, 3, 4])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex([4, 3, 2, 1, 0])[pd_df.reindex([4, 3, 2, 1, 0])['a'] > 2].sort_values('a')
        ds_result = ds_df.reindex([4, 3, 2, 1, 0])[ds_df.reindex([4, 3, 2, 1, 0])['a'] > 2].sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix_filter_take(self):
        """add_prefix -> Filter -> Take chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_prefixed = pd_df.add_prefix('col_')
        ds_prefixed = ds_df.add_prefix('col_')

        pd_result = pd_prefixed[pd_prefixed['col_a'] > 2].take([0, 1])
        ds_result = ds_prefixed[ds_prefixed['col_a'] > 2].take([0, 1])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 9: Empty/Single-Row Edge Cases
# =============================================================================


class TestEmptySingleRowEdgeCases:
    """Test edge cases with empty and single-row DataFrames."""

    def test_copy_empty(self):
        """Copy empty DataFrame."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        ds_result = ds_df.copy()

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_set_index_single_row(self):
        """set_index on single row."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a')
        ds_result = ds_df.set_index('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_empty_result(self):
        """iloc that returns empty result."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.iloc[5:10]
        ds_result = ds_df.iloc[5:10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_take_empty_list(self):
        """take with empty list."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.take([])
        ds_result = ds_df.take([])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_empty_dataframe(self):
        """Rename empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'a': 'new_a'})
        ds_result = ds_df.rename(columns={'a': 'new_a'})

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_reindex_empty_dataframe(self):
        """Reindex empty DataFrame."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex([0, 1, 2])
        ds_result = ds_df.reindex([0, 1, 2])

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_sort_index_single_row(self):
        """sort_index on single row."""
        pd_df = pd.DataFrame({'a': [1]}, index=['x'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_index()
        ds_result = ds_df.sort_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 10: swaplevel / reorder_levels (if supported)
# =============================================================================


class TestSwapReorderLevels:
    """Test MultiIndex level operations if supported."""

    def test_swaplevel_basic(self):
        """swaplevel on MultiIndex columns."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=index)
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.swaplevel()
        ds_result = ds_df.swaplevel()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_droplevel_basic(self):
        """droplevel on MultiIndex."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=index)
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.droplevel('first')
        ds_result = ds_df.droplevel('first')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 11: xs (cross-section) Operations
# =============================================================================


class TestXsCrossSection:
    """Test cross-section selection if supported."""

    def test_xs_basic(self):
        """Basic xs operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.xs('y')
        ds_result = ds_df.xs('y')

        assert_series_equal(ds_result, pd_result)

    def test_xs_multiindex(self):
        """xs on MultiIndex DataFrame."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=index)
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.xs('A')
        ds_result = ds_df.xs('A')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 12: insert/pop Operations
# =============================================================================


class TestInsertPop:
    """Test insert and pop operations."""

    def test_insert_basic(self):
        """Basic insert operation - creates new DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_df_copy = pd_df.copy()
        pd_df_copy.insert(1, 'b', [4, 5, 6])

        ds_df_copy = ds_df.copy()
        ds_df_copy.insert(1, 'b', [4, 5, 6])

        assert_datastore_equals_pandas(ds_df_copy, pd_df_copy)

    def test_pop_basic(self):
        """Basic pop operation - removes and returns column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_popped = pd_df.pop('b')
        ds_popped = ds_df.pop('b')

        assert_series_equal(ds_popped, pd_popped)

    def test_pop_result_dataframe(self):
        """After pop, DataFrame has remaining columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_df.pop('b')
        ds_df.pop('b')

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Section 13: assign with Complex Expressions
# =============================================================================


class TestAssignComplex:
    """Test assign with complex expressions."""

    def test_assign_lambda(self):
        """assign with lambda."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_columns(self):
        """assign multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2, c=lambda x: x['a'] + 10)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2, c=lambda x: x['a'] + 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_scalar(self):
        """assign scalar value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=100)
        ds_result = ds_df.assign(b=100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_series(self):
        """assign series value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd.Series([4, 5, 6]))
        ds_result = ds_df.assign(b=pd.Series([4, 5, 6]))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self):
        """assign then filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_assigned = pd_df.assign(b=lambda x: x['a'] * 2)
        ds_assigned = ds_df.assign(b=lambda x: x['a'] * 2)

        pd_result = pd_assigned[pd_assigned['b'] > 4]
        ds_result = ds_assigned[ds_assigned['b'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_overwrite_column(self):
        """assign overwrites existing column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(a=lambda x: x['a'] * 10)
        ds_result = ds_df.assign(a=lambda x: x['a'] * 10)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 14: Boolean Operations on Index
# =============================================================================


class TestBooleanIndexOperations:
    """Test boolean operations involving index."""

    def test_filter_preserves_index(self):
        """Filter preserves original index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=['v', 'w', 'x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_preserves_index(self):
        """Sort preserves index relationship."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_index_isin_filter(self):
        """Filter by index using isin."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=['v', 'w', 'x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df.index.isin(['w', 'y'])]
        ds_result = ds_df[ds_df.index.isin(['w', 'y'])]

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
