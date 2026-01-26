"""
Exploratory Batch 45: Insert/Pop, Level Operations, Slice Edge Cases, Copy Semantics

Focus areas:
1. Insert column at specific position with lazy chains
2. Pop column and chain operations
3. Level operations (droplevel, swaplevel, reorder_levels)
4. Complex slice with negative step
5. Copy semantics with lazy operations
6. Inplace parameter edge cases
7. Duplicate index handling
8. Assignment with expansion
"""

import pandas as pd
import numpy as np
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


class TestInsertColumnChains:
    """Test insert column operations with lazy chains (inplace like pandas)."""

    def test_insert_basic(self):
        """Test basic column insertion (inplace)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
        pd_df.insert(1, 'B', [4, 5, 6])
        pd_result = pd_df.copy()

        # DataStore (inplace)
        ds_df = DataStore({'A': [1, 2, 3], 'C': [7, 8, 9]})
        ds_df.insert(1, 'B', [4, 5, 6])

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_insert_at_beginning(self):
        """Test inserting column at position 0 (inplace)."""
        # pandas
        pd_df = pd.DataFrame({'B': [2, 3, 4], 'C': [5, 6, 7]})
        pd_df.insert(0, 'A', [1, 2, 3])
        pd_result = pd_df.copy()

        # DataStore (inplace)
        ds_df = DataStore({'B': [2, 3, 4], 'C': [5, 6, 7]})
        ds_df.insert(0, 'A', [1, 2, 3])

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_insert_at_end(self):
        """Test inserting column at the end (inplace)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df.insert(2, 'C', [7, 8, 9])
        pd_result = pd_df.copy()

        # DataStore (inplace)
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df.insert(2, 'C', [7, 8, 9])

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_insert_then_filter(self):
        """Test inserting column and then filtering."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
        pd_df.insert(1, 'B', [4, 5, 6])
        pd_result = pd_df[pd_df['B'] > 4]

        # DataStore (inplace, then filter)
        ds_df = DataStore({'A': [1, 2, 3], 'C': [7, 8, 9]})
        ds_df.insert(1, 'B', [4, 5, 6])
        ds_result = ds_df[ds_df['B'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_insert_with_scalar(self):
        """Test inserting column with scalar value (inplace)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df.insert(1, 'B', 100)
        pd_result = pd_df.copy()

        # DataStore (inplace)
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df.insert(1, 'B', 100)

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_insert_with_series(self):
        """Test inserting column with Series (inplace)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        s = pd.Series([10, 20, 30])
        pd_df.insert(1, 'B', s)
        pd_result = pd_df.copy()

        # DataStore (inplace)
        ds_df = DataStore({'A': [1, 2, 3]})
        s = pd.Series([10, 20, 30])
        ds_df.insert(1, 'B', s)

        assert_datastore_equals_pandas(ds_df, pd_result)


class TestPopColumnChains:
    """Test pop column operations with lazy chains."""

    def test_pop_basic(self):
        """Test basic column pop."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        pd_popped = pd_df.pop('B')
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_popped = ds_df.pop('B')
        ds_result = ds_df

        # After pop, DataFrame should not have column B
        assert_datastore_equals_pandas(ds_result, pd_result)
        # Popped value should be Series
        assert_series_equal(ds_popped, pd_popped)

    def test_pop_first_column(self):
        """Test popping the first column."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_popped = pd_df.pop('A')
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_popped = ds_df.pop('A')
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert_series_equal(ds_popped, pd_popped)

    def test_pop_last_column(self):
        """Test popping the last column."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_popped = pd_df.pop('B')
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_popped = ds_df.pop('B')
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert_series_equal(ds_popped, pd_popped)


class TestSliceEdgeCases:
    """Test slice operations with edge cases."""

    def test_slice_negative_step(self):
        """Test slice with negative step (reverse)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.iloc[::-1]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df[::-1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_negative_step_with_range(self):
        """Test slice with negative step and range."""
        # pandas
        pd_df = pd.DataFrame({'A': list(range(10))})
        pd_result = pd_df.iloc[8:2:-2]

        # DataStore
        ds_df = DataStore({'A': list(range(10))})
        ds_result = ds_df[8:2:-2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_step_larger_than_length(self):
        """Test slice with step larger than DataFrame length."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.iloc[::10]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df[::10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_empty_result(self):
        """Test slice that results in empty DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.iloc[10:20]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df[10:20]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_then_filter(self):
        """Test slice followed by filter."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        pd_result = pd_df.iloc[::2][pd_df.iloc[::2]['A'] > 3]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_sliced = ds_df[::2]
        ds_result = ds_sliced[ds_sliced['A'] > 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_slice(self):
        """Test filter followed by slice."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        pd_result = pd_df[pd_df['A'] > 3].iloc[::2]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_result = ds_df[ds_df['A'] > 3][::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCopySemantics:
    """Test copy semantics with lazy operations."""

    def test_copy_basic(self):
        """Test basic copy operation."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df.copy()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_copy_deep_vs_shallow(self):
        """Test deep vs shallow copy."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_deep = pd_df.copy(deep=True)
        pd_shallow = pd_df.copy(deep=False)

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_deep = ds_df.copy(deep=True)
        ds_shallow = ds_df.copy(deep=False)

        assert_datastore_equals_pandas(ds_deep, pd_deep)
        assert_datastore_equals_pandas(ds_shallow, pd_shallow)

    def test_copy_after_filter(self):
        """Test copy after filter operation."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        pd_result = pd_df[pd_df['A'] > 2].copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_result = ds_df[ds_df['A'] > 2].copy()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_copy_then_modify(self):
        """Test modifying copy doesn't affect original."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_copy = pd_df.copy()
        pd_copy['B'] = [4, 5, 6]
        pd_original = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_copy = ds_df.copy()
        ds_copy['B'] = [4, 5, 6]
        ds_original = ds_df

        # Original should not have column B
        assert 'B' not in ds_original.columns
        assert_datastore_equals_pandas(ds_original, pd_original)


class TestDuplicateIndexHandling:
    """Test handling of duplicate index values."""

    def test_duplicate_index_filter(self):
        """Test filtering with duplicate index."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]}, index=[0, 0, 1, 1])
        pd_result = pd_df[pd_df['A'] > 2]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4]}, index=[0, 0, 1, 1])
        ds_result = ds_df[ds_df['A'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicate_index_groupby(self):
        """Test groupby with duplicate index."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['x', 'x', 'y', 'y']}, index=[0, 0, 1, 1])
        pd_result = pd_df.groupby('B')['A'].sum()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': ['x', 'x', 'y', 'y']}, index=[0, 0, 1, 1])
        ds_result = ds_df.groupby('B')['A'].sum()

        # groupby results may have different order
        # Trigger execution via values attribute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.sort_index(),
            pd_result.sort_index())

    def test_duplicate_index_sort(self):
        """Test sorting with duplicate index."""
        # pandas
        pd_df = pd.DataFrame({'A': [3, 1, 2, 4]}, index=[0, 0, 1, 1])
        pd_result = pd_df.sort_values('A')

        # DataStore
        ds_df = DataStore({'A': [3, 1, 2, 4]}, index=[0, 0, 1, 1])
        ds_result = ds_df.sort_values('A')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAssignmentExpansion:
    """Test assignment with different shapes and expansion."""

    def test_assign_scalar_broadcast(self):
        """Test assigning scalar broadcasts to all rows."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df['B'] = 100
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df['B'] = 100
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_list_matching_length(self):
        """Test assigning list with matching length."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df['B'] = [10, 20, 30]
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df['B'] = [10, 20, 30]
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_array_matching_length(self):
        """Test assigning numpy array with matching length."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df['B'] = np.array([10, 20, 30])
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df['B'] = np.array([10, 20, 30])
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_expression_result(self):
        """Test assigning result of column expression."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df['B'] = pd_df['A'] * 2
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df['B'] = ds_df['A'] * 2
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_after_filter_chain(self):
        """Test assigning to filtered DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        pd_filtered = pd_df[pd_df['A'] > 2].copy()
        pd_filtered['C'] = pd_filtered['A'] + pd_filtered['B']
        pd_result = pd_filtered

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_filtered = ds_df[ds_df['A'] > 2]
        ds_filtered['C'] = ds_filtered['A'] + ds_filtered['B']
        ds_result = ds_filtered

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestLevelOperations:
    """Test MultiIndex level operations."""

    def test_droplevel_index(self):
        """Test dropping index level."""
        # pandas
        arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'A': [10, 20, 30, 40]}, index=index)
        pd_result = pd_df.droplevel(0)

        # DataStore
        ds_df = DataStore({'A': [10, 20, 30, 40]}, index=index)
        ds_result = ds_df.droplevel(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_droplevel_by_name(self):
        """Test dropping index level by name."""
        # pandas
        arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'A': [10, 20, 30, 40]}, index=index)
        pd_result = pd_df.droplevel('first')

        # DataStore
        ds_df = DataStore({'A': [10, 20, 30, 40]}, index=index)
        ds_result = ds_df.droplevel('first')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_swaplevel_index(self):
        """Test swapping index levels."""
        # pandas
        arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
        pd_df = pd.DataFrame({'A': [10, 20, 30, 40]}, index=index)
        pd_result = pd_df.swaplevel(0, 1)

        # DataStore
        ds_df = DataStore({'A': [10, 20, 30, 40]}, index=index)
        ds_result = ds_df.swaplevel(0, 1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reorder_levels_index(self):
        """Test reordering index levels."""
        # pandas
        arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b'], ['x', 'y', 'x', 'y']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second', 'third'])
        pd_df = pd.DataFrame({'A': [10, 20, 30, 40]}, index=index)
        pd_result = pd_df.reorder_levels([2, 0, 1])

        # DataStore
        ds_df = DataStore({'A': [10, 20, 30, 40]}, index=index)
        ds_result = ds_df.reorder_levels([2, 0, 1])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestInplaceParameter:
    """Test inplace parameter edge cases."""

    def test_drop_inplace_false(self):
        """Test drop with inplace=False (default)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.drop(columns=['B'], inplace=False)
        pd_original = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df.drop(columns=['B'], inplace=False)
        ds_original = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Original should not be modified
        assert 'B' in list(ds_original.columns)

    def test_reset_index_inplace_false(self):
        """Test reset_index with inplace=False."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        pd_result = pd_df.reset_index(inplace=False)

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_result = ds_df.reset_index(inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_inplace_false(self):
        """Test rename with inplace=False."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_result = pd_df.rename(columns={'A': 'X'}, inplace=False)
        pd_original = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df.rename(columns={'A': 'X'}, inplace=False)
        ds_original = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Original should have column A
        assert 'A' in list(ds_original.columns)


class TestEmptyDataFrameOperations:
    """Test operations on empty DataFrames."""

    def test_empty_df_insert(self):
        """Test inserting column into empty DataFrame (inplace)."""
        # pandas
        pd_df = pd.DataFrame()
        pd_df.insert(0, 'A', [])
        pd_result = pd_df.copy()

        # DataStore (inplace)
        ds_df = DataStore()
        ds_df.insert(0, 'A', [])

        assert_datastore_equals_pandas(ds_df, pd_result)

    def test_empty_df_copy(self):
        """Test copying empty DataFrame."""
        # pandas
        pd_df = pd.DataFrame()
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore()
        ds_result = ds_df.copy()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_slice(self):
        """Test slicing empty DataFrame."""
        # pandas
        pd_df = pd.DataFrame({'A': []})
        pd_result = pd_df.iloc[::2]

        # DataStore
        ds_df = DataStore({'A': []})
        ds_result = ds_df[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestChainedOperationsComplex:
    """Test complex chained operations."""

    def test_insert_filter_groupby_sort(self):
        """Test insert -> filter -> groupby -> sort chain."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'C': ['x', 'y', 'x', 'y', 'x']})
        pd_df.insert(1, 'B', [10, 20, 30, 40, 50])
        pd_result = pd_df[pd_df['B'] > 15].groupby('C')['A'].sum().sort_index()

        # DataStore (inplace insert)
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'C': ['x', 'y', 'x', 'y', 'x']})
        ds_df.insert(1, 'B', [10, 20, 30, 40, 50])
        ds_filtered = ds_df[ds_df['B'] > 15]
        ds_result = ds_filtered.groupby('C')['A'].sum()

        # Trigger execution via values attribute and compare
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.sort_index(),
            pd_result)

    def test_copy_filter_assign_sort(self):
        """Test copy -> filter -> assign -> sort chain."""
        # pandas
        pd_df = pd.DataFrame({'A': [5, 3, 1, 4, 2], 'B': [10, 20, 30, 40, 50]})
        pd_copy = pd_df.copy()
        pd_filtered = pd_copy[pd_copy['A'] > 2]
        pd_filtered = pd_filtered.copy()  # Need copy to avoid SettingWithCopyWarning
        pd_filtered['C'] = pd_filtered['A'] * 100
        pd_result = pd_filtered.sort_values('A')

        # DataStore
        ds_df = DataStore({'A': [5, 3, 1, 4, 2], 'B': [10, 20, 30, 40, 50]})
        ds_copy = ds_df.copy()
        ds_filtered = ds_copy[ds_copy['A'] > 2]
        ds_filtered['C'] = ds_filtered['A'] * 100
        ds_result = ds_filtered.sort_values('A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_filter_slice(self):
        """Test slice -> filter -> slice chain."""
        # pandas
        pd_df = pd.DataFrame({'A': list(range(20))})
        pd_sliced1 = pd_df.iloc[::2]  # Every 2nd row
        pd_filtered = pd_sliced1[pd_sliced1['A'] > 5]
        pd_result = pd_filtered.iloc[::2]  # Every 2nd row of filtered

        # DataStore
        ds_df = DataStore({'A': list(range(20))})
        ds_sliced1 = ds_df[::2]
        ds_filtered = ds_sliced1[ds_sliced1['A'] > 5]
        ds_result = ds_filtered[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanIndexingEdge:
    """Test boolean indexing edge cases."""

    def test_boolean_mask_all_true(self):
        """Test boolean mask where all values are True."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mask = pd.Series([True, True, True])
        pd_result = pd_df[mask]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df[mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_mask_all_false(self):
        """Test boolean mask where all values are False."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mask = pd.Series([False, False, False])
        pd_result = pd_df[mask]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df[mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_column_as_mask(self):
        """Test using boolean column as mask."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, False, True]})
        pd_result = pd_df[pd_df['B']]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [True, False, True]})
        ds_result = ds_df[ds_df['B']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_boolean_column_as_mask(self):
        """Test using negated boolean column as mask."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [True, False, True]})
        pd_result = pd_df[~pd_df['B']]

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3], 'B': [True, False, True]})
        ds_result = ds_df[~ds_df['B']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNaNHandlingEdge:
    """Test NaN/None handling edge cases."""

    def test_assign_nan_to_column(self):
        """Test assigning NaN values to column."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df['B'] = np.nan
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df['B'] = np.nan
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_mixed_nan(self):
        """Test assigning list with NaN values."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df['B'] = [1.0, np.nan, 3.0]
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, 2, 3]})
        ds_df['B'] = [1.0, np.nan, 3.0]
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_copy_with_nan(self):
        """Test copying DataFrame with NaN values."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, np.nan]})
        pd_result = pd_df.copy()

        # DataStore
        ds_df = DataStore({'A': [1, np.nan, 3], 'B': [np.nan, 2, np.nan]})
        ds_result = ds_df.copy()

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
