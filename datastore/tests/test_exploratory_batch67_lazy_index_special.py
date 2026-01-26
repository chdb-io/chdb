"""
Exploratory Batch 67: Lazy Column Assignment, Index Preservation, Special Values

This batch explores undertested boundary conditions:
1. Complex lazy column assignment with computed column dependencies
2. Index preservation across multiple operation chains
3. Special floating point values (inf, -inf, very large numbers)
4. DataFrame constructor edge cases
5. Chained operations with mixed engines (SQL + Pandas)

Discovery method: Architecture-based exploration after reviewing lazy_ops.py and pandas_compat.py
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


class TestLazyColumnAssignmentChains:
    """Test complex lazy column assignment scenarios."""

    def test_self_referencing_column_update(self):
        """Test column that references itself in update."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        # Self-referencing update: a = a * 2
        pd_df['a'] = pd_df['a'] * 2
        ds_df['a'] = ds_df['a'] * 2

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_multiple_self_references(self):
        """Test multiple self-referencing updates."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        # Multiple self-references
        pd_df['a'] = pd_df['a'] + 1
        ds_df['a'] = ds_df['a'] + 1

        pd_df['a'] = pd_df['a'] * 2
        ds_df['a'] = ds_df['a'] * 2

        pd_df['a'] = pd_df['a'] - 5
        ds_df['a'] = ds_df['a'] - 5

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_chained_column_dependencies(self):
        """Test column assignments that depend on each other."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        # Chain: a -> b -> c
        pd_df['b'] = pd_df['a'] * 2
        ds_df['b'] = ds_df['a'] * 2

        pd_df['c'] = pd_df['b'] + 10
        ds_df['c'] = ds_df['b'] + 10

        pd_df['d'] = pd_df['c'] / pd_df['a']
        ds_df['d'] = ds_df['c'] / ds_df['a']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_circular_like_updates(self):
        """Test updates where columns depend on each other (not truly circular)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        # a depends on b, then b depends on a (but not circular - sequential)
        pd_df['a'] = pd_df['a'] + pd_df['b']
        ds_df['a'] = ds_df['a'] + ds_df['b']

        pd_df['b'] = pd_df['b'] + pd_df['a']
        ds_df['b'] = ds_df['b'] + ds_df['a']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_after_filter(self):
        """Test column assignment after filtering."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        ds_filtered = ds_df[ds_df['a'] > 2]

        pd_filtered['c'] = pd_filtered['a'] + pd_filtered['b']
        ds_filtered['c'] = ds_filtered['a'] + ds_filtered['b']

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)


class TestIndexPreservationChains:
    """Test index preservation across operation chains."""

    def test_index_after_multiple_filters(self):
        """Test index preservation after multiple filter operations."""
        pd_df = pd.DataFrame(
            {'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']},
            index=['i1', 'i2', 'i3', 'i4', 'i5']
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 1][pd_df['a'] < 5]
        ds_result = ds_df[ds_df['a'] > 1][ds_df['a'] < 5]

        # Check index preservation
        ds_executed = get_dataframe(ds_result)
        assert list(ds_executed.index) == list(pd_result.index)

    def test_index_after_sort(self):
        """Test index preservation after sorting."""
        pd_df = pd.DataFrame(
            {'a': [3, 1, 2], 'b': ['x', 'y', 'z']},
            index=['i3', 'i1', 'i2']
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        ds_executed = get_dataframe(ds_result)
        assert list(ds_executed.index) == list(pd_result.index)

    def test_index_after_column_assignment(self):
        """Test index preservation after column assignment."""
        pd_df = pd.DataFrame(
            {'a': [1, 2, 3]},
            index=['x', 'y', 'z']
        )
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = pd_df['a'] * 2
        ds_df['b'] = ds_df['a'] * 2

        ds_executed = get_dataframe(ds_df)
        assert list(ds_executed.index) == list(pd_df.index)


class TestSpecialFloatValues:
    """Test handling of special floating point values."""

    def test_infinity_values(self):
        """Test handling of infinity values."""
        pd_df = pd.DataFrame({'a': [1.0, float('inf'), -float('inf'), 2.0]})
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_infinity_arithmetic(self):
        """Test arithmetic with infinity values."""
        pd_df = pd.DataFrame({'a': [1.0, float('inf'), -float('inf'), 2.0]})
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = pd_df['a'] + 1
        ds_df['b'] = ds_df['a'] + 1

        # inf + 1 = inf, -inf + 1 = -inf
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_infinity_comparison(self):
        """Test comparison with infinity values."""
        pd_df = pd.DataFrame({'a': [1.0, float('inf'), -float('inf'), 2.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_values(self):
        """Test handling of NaN values."""
        pd_df = pd.DataFrame({'a': [1.0, float('nan'), 2.0, float('nan')]})
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_mixed_special_values(self):
        """Test mix of inf, -inf, and nan."""
        pd_df = pd.DataFrame({
            'a': [1.0, float('inf'), float('nan'), -float('inf'), 2.0]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_very_large_numbers(self):
        """Test very large numbers."""
        pd_df = pd.DataFrame({
            'a': [1e308, -1e308, 1e-308, -1e-308, 0.0]
        })
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestDataFrameConstructorEdgeCases:
    """Test DataFrame constructor edge cases."""

    def test_from_empty_dict(self):
        """Test creating DataStore from empty dict."""
        pd_df = pd.DataFrame({})
        ds_df = DataStore({})

        assert get_dataframe(ds_df).empty
        assert len(get_dataframe(ds_df).columns) == 0

    def test_from_dict_with_empty_lists(self):
        """Test creating DataStore from dict with empty lists."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})

        ds_executed = get_dataframe(ds_df)
        assert list(ds_executed.columns) == ['a', 'b']
        assert len(ds_executed) == 0

    def test_from_single_column(self):
        """Test creating DataStore from single column dict."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_from_list_of_dicts(self):
        """Test creating DataStore from list of dicts."""
        data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_from_list_of_dicts_uneven(self):
        """Test creating DataStore from list of dicts with different keys."""
        data = [{'a': 1, 'b': 2}, {'a': 3, 'c': 4}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_from_numpy_array(self):
        """Test creating DataStore from numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        pd_df = pd.DataFrame(arr, columns=['a', 'b'])
        ds_df = DataStore(arr, columns=['a', 'b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_from_series(self):
        """Test creating DataStore from pandas Series."""
        s = pd.Series([1, 2, 3], name='values')
        pd_df = s.to_frame()
        ds_df = DataStore(s)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestMixedEngineOperations:
    """Test operations that mix SQL and Pandas execution engines."""

    def test_filter_then_assign_then_filter(self):
        """Test filter -> assign -> filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 1].copy()
        ds_result = ds_df[ds_df['a'] > 1]

        pd_result['c'] = pd_result['b'] * 2
        ds_result['c'] = ds_result['b'] * 2

        pd_result = pd_result[pd_result['c'] > 50]
        ds_result = ds_result[ds_result['c'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_assign_then_head(self):
        """Test sort -> assign -> head chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a').copy()
        ds_result = ds_df.sort_values('a')

        pd_result['c'] = range(len(pd_result))
        ds_result['c'] = range(len(ds_result))

        pd_result = pd_result.head(3)
        ds_result = ds_result.head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByEdgeCases:
    """Test groupby edge cases."""

    def test_groupby_computed_column(self):
        """Test groupby on a computed column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [10, 20, 30, 40, 50, 60]})
        ds_df = DataStore(pd_df.copy())

        # Create computed column
        pd_df['category'] = pd_df['a'] // 2
        ds_df['category'] = ds_df['a'] // 2

        # Groupby on computed column
        pd_result = pd_df.groupby('category')['b'].sum()
        ds_result = ds_df.groupby('category')['b'].sum()

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)

    def test_groupby_with_filter_before(self):
        """Test groupby after filtering."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2, 3, 3],
            'b': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 1].groupby('a')['b'].mean()
        ds_result = ds_df[ds_df['a'] > 1].groupby('a')['b'].mean()

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)

    def test_groupby_single_group(self):
        """Test groupby that results in single group."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 1, 1],
            'b': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('a')['b'].sum()
        ds_result = ds_df.groupby('a')['b'].sum()

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)

    def test_groupby_all_unique(self):
        """Test groupby where each row is its own group."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('a')['b'].sum()
        ds_result = ds_df.groupby('a')['b'].sum()

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)


class TestSelectColumnEdgeCases:
    """Test column selection edge cases."""

    @pytest.mark.xfail(reason="SQL does not allow duplicate column names in SELECT; DataStore auto-renames to 'a_1'")
    def test_select_same_column_twice(self):
        """Test selecting the same column twice.

        Note: pandas allows duplicate column names, but SQL requires unique aliases.
        DataStore automatically renames duplicate columns (e.g., 'a' -> 'a_1').
        This is a known behavioral difference due to SQL limitations.
        """
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        # pandas allows selecting same column twice
        pd_result = pd_df[['a', 'a']]
        ds_result = ds_df[['a', 'a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_columns_different_order(self):
        """Test selecting columns in different order."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['c', 'a', 'b']]
        ds_result = ds_df[['c', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_single_column_as_df(self):
        """Test selecting single column as DataFrame (with double brackets)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['a']]
        ds_result = ds_df[['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticChains:
    """Test complex arithmetic chains."""

    def test_multiple_operator_chain(self):
        """Test chain of multiple arithmetic operators."""
        pd_df = pd.DataFrame({'a': [10, 20, 30], 'b': [2, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = (pd_df['a'] + pd_df['b']) * (pd_df['a'] - pd_df['b'])
        ds_df['result'] = (ds_df['a'] + ds_df['b']) * (ds_df['a'] - ds_df['b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_division_chain(self):
        """Test division operations chain."""
        pd_df = pd.DataFrame({'a': [100.0, 200.0, 300.0], 'b': [10.0, 20.0, 30.0]})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = pd_df['a'] / pd_df['b'] / 2
        ds_df['result'] = ds_df['a'] / ds_df['b'] / 2

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_floor_division(self):
        """Test floor division operator."""
        pd_df = pd.DataFrame({'a': [10, 21, 35], 'b': [3, 4, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = pd_df['a'] // pd_df['b']
        ds_df['result'] = ds_df['a'] // ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_modulo_operation(self):
        """Test modulo operator."""
        pd_df = pd.DataFrame({'a': [10, 21, 35], 'b': [3, 4, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_df['result'] = pd_df['a'] % pd_df['b']
        ds_df['result'] = ds_df['a'] % ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestStringColumnOperations:
    """Test operations on string columns."""

    def test_string_concatenation_columns(self):
        """Test concatenating string columns."""
        pd_df = pd.DataFrame({'a': ['hello', 'foo'], 'b': ['world', 'bar']})
        ds_df = DataStore(pd_df.copy())

        pd_df['c'] = pd_df['a'] + ' ' + pd_df['b']
        ds_df['c'] = ds_df['a'] + ' ' + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_string_with_null(self):
        """Test string column with null values."""
        pd_df = pd.DataFrame({'a': ['hello', None, 'world']})
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_string_filter(self):
        """Test filtering on string column."""
        pd_df = pd.DataFrame({'a': ['apple', 'banana', 'cherry'], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] == 'banana']
        ds_result = ds_df[ds_df['a'] == 'banana']

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanOperations:
    """Test boolean column operations."""

    def test_boolean_column_filter(self):
        """Test filtering by boolean column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'flag': [True, False, True]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['flag']]
        ds_result = ds_df[ds_df['flag']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_boolean_filter(self):
        """Test filtering by negated boolean column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'flag': [True, False, True]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[~pd_df['flag']]
        ds_result = ds_df[~ds_df['flag']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_to_boolean(self):
        """Test creating boolean from comparison."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df['is_big'] = pd_df['a'] > 3
        ds_df['is_big'] = ds_df['a'] > 3

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestAggregationEdgeCases:
    """Test aggregation edge cases."""

    def test_sum_empty_dataframe(self):
        """Test sum on empty DataFrame."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore({'a': []})

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        assert ds_result == pd_result

    def test_mean_single_value(self):
        """Test mean on single value."""
        pd_df = pd.DataFrame({'a': [42.0]})
        ds_df = DataStore({'a': [42.0]})

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()

        assert ds_result == pd_result

    def test_min_max_with_nulls(self):
        """Test min/max with null values."""
        pd_df = pd.DataFrame({'a': [1.0, None, 3.0, None, 5.0]})
        ds_df = DataStore(pd_df.copy())

        assert ds_df['a'].min() == pd_df['a'].min()
        assert ds_df['a'].max() == pd_df['a'].max()

    def test_count_with_nulls(self):
        """Test count with null values."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_count = pd_df['a'].count()
        ds_count = ds_df['a'].count()

        assert ds_count == pd_count
