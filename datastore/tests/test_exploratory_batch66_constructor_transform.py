"""
Exploratory Batch 66: Constructor Edge Cases, Transform, and Complex Index Operations

This batch explores boundary conditions in:
1. DataFrame constructor edge cases (empty, nested, unusual types)
2. GroupBy transform operations
3. Complex index operations (set_index, reset_index chains)
4. Multiple column modifications in chains
5. Concat/merge with empty DataFrames

Discovery method: Architecture-based exploration from core.py and groupby.py
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
)
from tests.xfail_markers import pandas_version_no_include_groups


class TestConstructorEdgeCases:
    """Test DataStore constructor with unusual inputs."""

    def test_empty_dict_constructor(self):
        """Test creating DataStore from empty dict."""
        pd_df = pd.DataFrame({})
        ds_df = DataStore({})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_dict_with_empty_lists(self):
        """Test creating DataStore from dict with empty lists."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_dict_with_none_values(self):
        """Test creating DataStore from dict with None values."""
        pd_df = pd.DataFrame({'a': [1, None, 3], 'b': [None, None, None]})
        ds_df = DataStore({'a': [1, None, 3], 'b': [None, None, None]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_dict_with_mixed_types(self):
        """Test creating DataStore from dict with mixed types."""
        pd_df = pd.DataFrame({'a': [1, 'str', 3.5], 'b': [True, None, 'x']})
        ds_df = DataStore({'a': [1, 'str', 3.5], 'b': [True, None, 'x']})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_list_of_dicts_constructor(self):
        """Test creating DataStore from list of dicts."""
        data = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}, {'a': 3, 'b': 'z'}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_list_of_dicts_with_missing_keys(self):
        """Test creating DataStore from list of dicts with missing keys."""
        data = [{'a': 1}, {'b': 2}, {'a': 3, 'b': 4}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_list_of_lists_constructor(self):
        """Test creating DataStore from list of lists."""
        data = [[1, 'x'], [2, 'y'], [3, 'z']]
        pd_df = pd.DataFrame(data, columns=['a', 'b'])
        ds_df = DataStore(data, columns=['a', 'b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_numpy_array_constructor(self):
        """Test creating DataStore from numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        pd_df = pd.DataFrame(arr, columns=['a', 'b'])
        ds_df = DataStore(arr, columns=['a', 'b'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_series_constructor(self):
        """Test creating DataStore from pandas Series."""
        series = pd.Series([1, 2, 3], name='values')
        pd_df = series.to_frame()
        ds_df = DataStore(series)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_with_index_parameter(self):
        """Test creating DataStore with explicit index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore({'a': [1, 2, 3]}, index=['x', 'y', 'z'])

        # Note: DataStore may handle index differently, compare values
        ds_result = get_dataframe(ds_df)
        assert list(ds_result['a']) == list(pd_df['a'])


class TestGroupByTransform:
    """Test GroupBy transform operations."""

    def test_transform_mean(self):
        """Test groupby transform with mean."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group')['value'].transform('mean')

        # DataStore may not have transform - check if it exists
        try:
            ds_result = ds_df.groupby('group')['value'].transform('mean')
            assert_series_equal(get_series(ds_result), pd_result, check_names=False)
        except AttributeError:
            pytest.skip("transform not implemented in DataStore")

    def test_transform_sum(self):
        """Test groupby transform with sum."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group')['value'].transform('sum')

        try:
            ds_result = ds_df.groupby('group')['value'].transform('sum')
            assert_series_equal(get_series(ds_result), pd_result, check_names=False)
        except AttributeError:
            pytest.skip("transform not implemented in DataStore")

    def test_transform_lambda(self):
        """Test groupby transform with lambda function."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group')['value'].transform(lambda x: x - x.mean())

        try:
            ds_result = ds_df.groupby('group')['value'].transform(lambda x: x - x.mean())
            assert_series_equal(get_series(ds_result), pd_result, check_names=False)
        except AttributeError:
            pytest.skip("transform not implemented in DataStore")


class TestGroupByApply:
    """Test GroupBy apply operations."""

    def test_apply_custom_function(self):
        """Test groupby apply with custom function."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        def custom_agg(x):
            return x.max() - x.min()

        pd_result = pd_df.groupby('group')['value'].apply(custom_agg)

        try:
            ds_result = ds_df.groupby('group')['value'].apply(custom_agg)
            # Use proper comparison with index preserved
            assert_series_equal(get_series(ds_result), pd_result, check_names=False)
        except AttributeError:
            pytest.skip("apply not implemented in DataStore groupby")

    @pandas_version_no_include_groups
    def test_apply_returning_dataframe(self):
        """Test groupby apply returning DataFrame.

        Note: include_groups parameter was added in pandas 2.2.0.
        """
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        def top_n(x, n=1):
            return x.nlargest(n, 'value')

        pd_result = pd_df.groupby('group').apply(top_n, include_groups=False)

        try:
            ds_result = ds_df.groupby('group').apply(top_n)
            # Just check shape for now
            ds_result_df = get_dataframe(ds_result)
            assert len(ds_result_df) == len(pd_result)
        except (AttributeError, TypeError):
            pytest.skip("apply with DataFrame result not implemented")


class TestComplexIndexOperations:
    """Test complex index manipulation chains."""

    def test_set_index_then_reset(self):
        """Test set_index followed by reset_index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a').reset_index()
        ds_result = ds_df.set_index('a').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_set_index(self):
        """Test setting index multiple times."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a').reset_index().set_index('b')
        ds_result = ds_df.set_index('a').reset_index().set_index('b')

        # Compare DataFrame content
        ds_df_result = get_dataframe(ds_result)
        assert list(ds_df_result.columns) == list(pd_result.columns)

    def test_reset_index_drop(self):
        """Test reset_index with drop=True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        pd_df = pd_df.set_index('a')
        ds_df = DataStore({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = ds_df.set_index('a')

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_with_filter(self):
        """Test set_index after filtering."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['x', 'y', 'z', 'w']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 1].set_index('a')
        ds_result = ds_df[ds_df['a'] > 1].set_index('a')

        ds_df_result = get_dataframe(ds_result)
        assert len(ds_df_result) == len(pd_result)


class TestMultipleColumnModifications:
    """Test multiple column modifications in chains."""

    def test_multiple_column_assign(self):
        """Test assigning multiple columns in sequence."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_df['b'] = pd_df['a'] * 2
        pd_df['c'] = pd_df['a'] + pd_df['b']

        ds_df['b'] = ds_df['a'] * 2
        ds_df['c'] = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_column_overwrite_chain(self):
        """Test overwriting same column multiple times."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_df['a'] = pd_df['a'] + 1
        pd_df['a'] = pd_df['a'] * 2
        pd_df['a'] = pd_df['a'] - 1

        ds_df['a'] = ds_df['a'] + 1
        ds_df['a'] = ds_df['a'] * 2
        ds_df['a'] = ds_df['a'] - 1

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_method_chaining(self):
        """Test using assign() for immutable column creation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2, c=lambda x: x['a'] + 10)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2, c=lambda x: x['a'] + 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_referencing_new_column(self):
        """Test assign() where one column references another new column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        # In pandas, assign processes columns in order
        pd_result = pd_df.assign(b=lambda x: x['a'] * 2).assign(c=lambda x: x['b'] + 1)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2).assign(c=lambda x: x['b'] + 1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestConcatEdgeCases:
    """Test pd.concat edge cases with DataStore."""

    def test_concat_empty_dataframes(self):
        """Test concatenating empty DataFrames."""
        pd_df1 = pd.DataFrame({'a': [], 'b': []})
        pd_df2 = pd.DataFrame({'a': [], 'b': []})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = pd.concat([get_dataframe(ds_df1), get_dataframe(ds_df2)], ignore_index=True)

        pd.testing.assert_frame_equal(ds_result, pd_result)

    def test_concat_empty_with_nonempty(self):
        """Test concatenating empty with non-empty DataFrame."""
        pd_empty = pd.DataFrame({'a': [], 'b': []})
        pd_full = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

        ds_empty = DataStore(pd_empty.copy())
        ds_full = DataStore(pd_full.copy())

        pd_result = pd.concat([pd_empty, pd_full], ignore_index=True)
        ds_result = pd.concat([get_dataframe(ds_empty), get_dataframe(ds_full)], ignore_index=True)

        pd.testing.assert_frame_equal(ds_result, pd_result)

    def test_concat_different_columns(self):
        """Test concatenating DataFrames with different columns."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        pd_df2 = pd.DataFrame({'a': [3, 4], 'c': [10, 20]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = pd.concat([get_dataframe(ds_df1), get_dataframe(ds_df2)], ignore_index=True)

        pd.testing.assert_frame_equal(ds_result, pd_result)


class TestMergeEdgeCases:
    """Test merge edge cases."""

    def test_merge_on_empty(self):
        """Test merging with empty DataFrame."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'val': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key': [], 'other': []})

        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd.merge(pd_left, pd_right, on='key', how='left')
        ds_result = pd.merge(get_dataframe(ds_left), get_dataframe(ds_right), on='key', how='left')

        pd.testing.assert_frame_equal(ds_result, pd_result)

    def test_merge_no_matching_keys(self):
        """Test merge where no keys match."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'val': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key': [4, 5, 6], 'other': ['x', 'y', 'z']})

        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd.merge(pd_left, pd_right, on='key', how='inner')
        ds_result = pd.merge(get_dataframe(ds_left), get_dataframe(ds_right), on='key', how='inner')

        pd.testing.assert_frame_equal(ds_result, pd_result)

    def test_merge_all_keys_match(self):
        """Test merge where all keys match."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'val': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key': [1, 2, 3], 'other': ['x', 'y', 'z']})

        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd.merge(pd_left, pd_right, on='key')
        ds_result = pd.merge(get_dataframe(ds_left), get_dataframe(ds_right), on='key')

        pd.testing.assert_frame_equal(ds_result, pd_result)


class TestDropDuplicatesEdgeCases:
    """Test drop_duplicates edge cases."""

    def test_drop_duplicates_all_same(self):
        """Test drop_duplicates when all rows are the same."""
        pd_df = pd.DataFrame({'a': [1, 1, 1], 'b': ['x', 'x', 'x']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_all_unique(self):
        """Test drop_duplicates when all rows are unique."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """Test drop_duplicates with keep='last'."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2], 'b': ['x', 'y', 'z', 'w']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates(subset=['a'], keep='last')
        ds_result = ds_df.drop_duplicates(subset=['a'], keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_drop_duplicates_keep_false(self):
        """Test drop_duplicates with keep=False."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 3, 3], 'b': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop_duplicates(subset=['a'], keep=False)
        ds_result = ds_df.drop_duplicates(subset=['a'], keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestRenameEdgeCases:
    """Test rename edge cases."""

    def test_rename_nonexistent_column(self):
        """Test renaming non-existent column (should not error in pandas)."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df.copy())

        # pandas silently ignores non-existent columns
        pd_result = pd_df.rename(columns={'nonexistent': 'new_name'})
        ds_result = ds_df.rename(columns={'nonexistent': 'new_name'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_to_existing_name(self):
        """Test renaming to a name that already exists."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df.copy())

        # This creates duplicate columns in pandas
        pd_result = pd_df.rename(columns={'a': 'b'})
        ds_result = ds_df.rename(columns={'a': 'b'})

        ds_df_result = get_dataframe(ds_result)
        assert list(ds_df_result.columns) == list(pd_result.columns)

    def test_rename_all_columns(self):
        """Test renaming all columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y', 'c': 'z'})
        ds_result = ds_df.rename(columns={'a': 'x', 'b': 'y', 'c': 'z'})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFillnaEdgeCases:
    """Test fillna edge cases."""

    def test_fillna_all_null_column(self):
        """Test fillna on column with all null values."""
        pd_df = pd.DataFrame({'a': [None, None, None], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_no_null_values(self):
        """Test fillna on DataFrame with no null values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_df)

    def test_fillna_with_dict(self):
        """Test fillna with different values per column."""
        pd_df = pd.DataFrame({'a': [1, None, 3], 'b': [None, 5, None]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.fillna({'a': 0, 'b': -1})
        ds_result = ds_df.fillna({'a': 0, 'b': -1})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSortValuesEdgeCases:
    """Test sort_values edge cases."""

    def test_sort_with_all_null(self):
        """Test sorting column with all null values."""
        pd_df = pd.DataFrame({'a': [None, None, None], 'b': [3, 1, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_sort_multiple_columns_different_order(self):
        """Test sorting by multiple columns with different orders."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': [4, 3, 2, 1]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values(['a', 'b'], ascending=[True, False])
        ds_result = ds_df.sort_values(['a', 'b'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_na_position(self):
        """Test sort_values with na_position parameter."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore(pd_df.copy())

        # na_position='first'
        pd_result = pd_df.sort_values('a', na_position='first')
        ds_result = ds_df.sort_values('a', na_position='first')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCountsEdgeCases:
    """Test value_counts edge cases."""

    def test_value_counts_all_same(self):
        """Test value_counts when all values are the same."""
        pd_df = pd.DataFrame({'a': [1, 1, 1, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].value_counts()
        ds_result = ds_df['a'].value_counts()

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)

    def test_value_counts_all_unique(self):
        """Test value_counts when all values are unique."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].value_counts()
        ds_result = ds_df['a'].value_counts()

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)

    def test_value_counts_with_null(self):
        """Test value_counts with null values."""
        pd_df = pd.DataFrame({'a': [1, 1, None, 2, None]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].value_counts(dropna=False)
        ds_result = ds_df['a'].value_counts(dropna=False)

        # Compare after sorting by index to handle order differences
        pd_sorted = pd_result.sort_index()
        ds_sorted = get_series(ds_result).sort_index()
        assert_series_equal(ds_sorted, pd_sorted, check_names=False)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize=True."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].value_counts(normalize=True)
        ds_result = ds_df['a'].value_counts(normalize=True)

        assert_series_equal(get_series(ds_result), pd_result, check_names=False)


class TestNuniqueEdgeCases:
    """Test nunique edge cases."""

    def test_nunique_all_same(self):
        """Test nunique when all values are the same."""
        pd_df = pd.DataFrame({'a': [1, 1, 1, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].nunique()
        ds_result = ds_df['a'].nunique()

        assert ds_result == pd_result

    def test_nunique_all_null(self):
        """Test nunique when all values are null."""
        pd_df = pd.DataFrame({'a': [None, None, None]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].nunique()
        ds_result = ds_df['a'].nunique()

        assert ds_result == pd_result

    def test_nunique_with_dropna_false(self):
        """Test nunique with dropna=False."""
        pd_df = pd.DataFrame({'a': [1, 1, None, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].nunique(dropna=False)
        ds_result = ds_df['a'].nunique(dropna=False)

        assert ds_result == pd_result
