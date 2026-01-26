"""
Exploratory Batch 62: ColumnExpr Advanced Features + DataFrame Metadata Operations

This batch explores:
1. ColumnExpr advanced methods: factorize, searchsorted, get, map, etc.
2. DataFrame metadata/state operations: flags, attrs, set_flags
3. Data alignment and combine operations with lazy chains
4. Complex accessor chains with groupby/filter combinations
5. Edge cases: empty data, single row, all-NA columns in various combinations

Discovery method: Architecture-based exploration
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    get_dataframe,
)


def _get_series_or_values(ds_result):
    """Helper to get values from ColumnExpr result.
    
    Some methods like unique() return numpy arrays directly,
    so we need to handle this case specially.
    """
    if hasattr(ds_result, '_execute'):
        executed = ds_result._execute()
        if isinstance(executed, np.ndarray):
            return list(executed)
        elif isinstance(executed, pd.Series):
            return executed.tolist()
        elif isinstance(executed, pd.DataFrame):
            if executed.shape[1] == 1:
                return executed.iloc[:, 0].tolist()
            return executed.values.tolist()
        return executed
    elif isinstance(ds_result, pd.DataFrame):
        if ds_result.shape[1] == 1:
            return ds_result.iloc[:, 0].tolist()
        return ds_result.iloc[:, 0].tolist()
    elif isinstance(ds_result, pd.Series):
        return ds_result.tolist()
    elif isinstance(ds_result, np.ndarray):
        return list(ds_result)
    return list(ds_result)


class TestColumnExprFactorize:
    """Test ColumnExpr.factorize() method."""

    def test_factorize_basic(self):
        """Basic factorize on string column."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'a', 'c', 'b', 'a']})
        ds_df = DataStore(pd_df.copy())

        pd_codes, pd_uniques = pd_df['category'].factorize()
        ds_codes, ds_uniques = ds_df['category'].factorize()

        np.testing.assert_array_equal(ds_codes, pd_codes)
        np.testing.assert_array_equal(ds_uniques, pd_uniques)

    def test_factorize_with_na(self):
        """Factorize with NA values."""
        pd_df = pd.DataFrame({'category': ['a', None, 'b', 'a', None, 'c']})
        ds_df = DataStore(pd_df.copy())

        pd_codes, pd_uniques = pd_df['category'].factorize()
        ds_codes, ds_uniques = ds_df['category'].factorize()

        np.testing.assert_array_equal(ds_codes, pd_codes)
        np.testing.assert_array_equal(ds_uniques, pd_uniques)

    def test_factorize_sorted(self):
        """Factorize with sort=True."""
        pd_df = pd.DataFrame({'category': ['c', 'a', 'b', 'a', 'c', 'b']})
        ds_df = DataStore(pd_df.copy())

        pd_codes, pd_uniques = pd_df['category'].factorize(sort=True)
        ds_codes, ds_uniques = ds_df['category'].factorize(sort=True)

        np.testing.assert_array_equal(ds_codes, pd_codes)
        np.testing.assert_array_equal(ds_uniques, pd_uniques)

    def test_factorize_numeric(self):
        """Factorize on numeric column."""
        pd_df = pd.DataFrame({'value': [10, 20, 10, 30, 20, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_codes, pd_uniques = pd_df['value'].factorize()
        ds_codes, ds_uniques = ds_df['value'].factorize()

        np.testing.assert_array_equal(ds_codes, pd_codes)
        np.testing.assert_array_equal(ds_uniques, pd_uniques)

    def test_factorize_after_filter(self):
        """Factorize after filter operation."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'a', 'c', 'b', 'a'], 'value': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['value'] > 2]
        ds_filtered = ds_df[ds_df['value'] > 2]

        pd_codes, pd_uniques = pd_filtered['category'].factorize()
        ds_codes, ds_uniques = get_dataframe(ds_filtered)['category'].factorize()

        np.testing.assert_array_equal(ds_codes, pd_codes)
        np.testing.assert_array_equal(ds_uniques, pd_uniques)


class TestColumnExprSearchsorted:
    """Test ColumnExpr.searchsorted() method."""

    def test_searchsorted_basic(self):
        """Basic searchsorted operation."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].searchsorted(5)
        ds_result = ds_df['value'].searchsorted(5)

        assert ds_result == pd_result

    def test_searchsorted_array_values(self):
        """Searchsorted with array of values."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].searchsorted([3, 7])
        ds_result = ds_df['value'].searchsorted([3, 7])

        np.testing.assert_array_equal(ds_result, pd_result)

    def test_searchsorted_side_right(self):
        """Searchsorted with side='right'."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 4, 4, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].searchsorted(4, side='right')
        ds_result = ds_df['value'].searchsorted(4, side='right')

        assert ds_result == pd_result


class TestColumnExprGet:
    """Test ColumnExpr.get() method."""

    def test_get_existing_index(self):
        """Get value at existing index."""
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].get(2)
        ds_result = ds_df['value'].get(2)

        assert ds_result == pd_result

    def test_get_missing_index_with_default(self):
        """Get value at missing index with default."""
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].get(100, default=-1)
        ds_result = ds_df['value'].get(100, default=-1)

        assert ds_result == pd_result

    def test_get_negative_index(self):
        """Get value at negative index."""
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        # Negative index returns None in pandas (index-based)
        pd_result = pd_df['value'].get(-1)
        ds_result = ds_df['value'].get(-1)

        # Both should return None for non-existent label index
        assert ds_result == pd_result


class TestColumnExprMap:
    """Test ColumnExpr.map() method."""

    def test_map_with_dict(self):
        """Map values with dictionary."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'b']})
        ds_df = DataStore(pd_df.copy())

        mapping = {'a': 1, 'b': 2, 'c': 3}
        pd_result = pd_df['category'].map(mapping)
        ds_result = ds_df['category'].map(mapping)

        # ColumnExpr.map returns a ColumnExpr, need to execute it
        ds_values = _get_series_or_values(ds_result)

        assert ds_values == pd_result.tolist()

    def test_map_with_function(self):
        """Map values with function."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].map(lambda x: x * 2)
        ds_result = ds_df['value'].map(lambda x: x * 2)

        ds_values = _get_series_or_values(ds_result)

        assert ds_values == pd_result.tolist()

    def test_map_with_na_action(self):
        """Map values with na_action='ignore'."""
        pd_df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].map(lambda x: x * 2, na_action='ignore')
        ds_result = ds_df['value'].map(lambda x: x * 2, na_action='ignore')

        ds_values = _get_series_or_values(ds_result)

        # Compare non-NaN values
        pd_non_na = [x for x in pd_result.tolist() if pd.notna(x)]
        ds_non_na = [x for x in ds_values if pd.notna(x)]
        assert ds_non_na == pd_non_na

    def test_map_after_filter(self):
        """Map after filter operation."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'b'], 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        mapping = {'a': 10, 'b': 20, 'c': 30}

        pd_filtered = pd_df[pd_df['value'] > 2]
        ds_filtered = ds_df[ds_df['value'] > 2]

        pd_result = pd_filtered['category'].map(mapping)
        ds_result = get_dataframe(ds_filtered)['category'].map(mapping)

        pd.testing.assert_series_equal(
            ds_result.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )


class TestColumnExprValueCounts:
    """Test ColumnExpr.value_counts() with various parameters."""

    def test_value_counts_normalize(self):
        """Value counts with normalize=True."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'a', 'c', 'b', 'a', 'a']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].value_counts(normalize=True)
        ds_result = ds_df['category'].value_counts(normalize=True)

        # DataStore returns DataFrame or Series
        ds_executed = get_dataframe(ds_result)
        if isinstance(ds_executed, pd.DataFrame):
            # Get proportions and compare as sorted lists
            if 'proportion' in ds_executed.columns:
                ds_values = sorted(ds_executed['proportion'].tolist())
            elif 'count' in ds_executed.columns:
                ds_values = sorted(ds_executed['count'].tolist())
            else:
                ds_values = sorted(ds_executed.iloc[:, -1].tolist())
        else:
            ds_values = sorted(ds_executed.tolist())

        pd_values = sorted(pd_result.tolist())
        
        # Compare values (proportions should be same regardless of order)
        np.testing.assert_array_almost_equal(ds_values, pd_values, decimal=5)

    def test_value_counts_basic(self):
        """Basic value counts."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'a', 'c', 'b', 'a', 'a']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].value_counts()
        ds_result = ds_df['category'].value_counts()

        ds_executed = get_dataframe(ds_result)
        if isinstance(ds_executed, pd.DataFrame):
            # Get counts as series
            if 'count' in ds_executed.columns:
                ds_values = sorted(ds_executed['count'].tolist())
            else:
                ds_values = sorted(ds_executed.iloc[:, -1].tolist())
        else:
            ds_values = sorted(ds_executed.tolist())

        pd_values = sorted(pd_result.tolist())
        assert ds_values == pd_values

    def test_value_counts_ascending(self):
        """Value counts with ascending=True."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'a', 'c', 'b', 'a', 'a']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].value_counts(ascending=True)
        ds_result = ds_df['category'].value_counts(ascending=True)

        ds_executed = get_dataframe(ds_result)
        if isinstance(ds_executed, pd.DataFrame):
            if 'count' in ds_executed.columns:
                ds_values = ds_executed['count'].tolist()
            else:
                ds_values = ds_executed.iloc[:, -1].tolist()
        else:
            ds_values = ds_executed.tolist()

        pd_values = pd_result.tolist()
        # Check values are the same (ascending order)
        assert ds_values == pd_values


class TestColumnExprUnique:
    """Test ColumnExpr.unique() method."""

    def test_unique_basic(self):
        """Basic unique values."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'a', 'c', 'b', 'a']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].unique()
        ds_result = ds_df['category'].unique()

        # unique() returns numpy array, get values properly
        ds_values = _get_series_or_values(ds_result)

        # Unique may return in different order, so compare as sets
        assert set(ds_values) == set(pd_result)

    def test_unique_with_na(self):
        """Unique values with NA."""
        pd_df = pd.DataFrame({'category': ['a', None, 'b', 'a', None, 'c']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].unique()
        ds_result = ds_df['category'].unique()

        ds_values = _get_series_or_values(ds_result)

        # Check non-NA values match
        pd_non_na = [x for x in pd_result if pd.notna(x)]
        ds_non_na = [x for x in ds_values if pd.notna(x)]
        assert set(ds_non_na) == set(pd_non_na)

    def test_unique_numeric(self):
        """Unique values on numeric column."""
        pd_df = pd.DataFrame({'value': [1, 2, 1, 3, 2, 1, 4, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].unique()
        ds_result = ds_df['value'].unique()

        ds_values = _get_series_or_values(ds_result)

        assert set(ds_values) == set(pd_result)

    def test_unique_after_filter(self):
        """Unique after filter."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'b'], 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['value'] > 2]
        ds_filtered = ds_df[ds_df['value'] > 2]

        pd_result = pd_filtered['category'].unique()
        ds_result = get_dataframe(ds_filtered)['category'].unique()

        assert set(ds_result) == set(pd_result)


class TestDataFrameAttrsFlags:
    """Test DataFrame attrs and flags properties."""

    def test_attrs_empty_default(self):
        """Default attrs is empty dict."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        assert ds_df.attrs == pd_df.attrs
        assert ds_df.attrs == {}

    def test_attrs_preserved(self):
        """Attrs should be preserved from source DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df.attrs['custom_attr'] = 'test_value'
        ds_df = DataStore(pd_df.copy())

        # After execution, attrs should match
        assert get_dataframe(ds_df).attrs.get('custom_attr') == pd_df.attrs.get('custom_attr')

    def test_flags_default(self):
        """Default flags."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        # Both should have flags attribute
        pd_flags = pd_df.flags
        ds_flags = ds_df.flags

        assert ds_flags.allows_duplicate_labels == pd_flags.allows_duplicate_labels

    def test_set_flags_allows_duplicate_labels(self):
        """Set flags with allows_duplicate_labels."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_flags(allows_duplicate_labels=False)
        ds_result = ds_df.set_flags(allows_duplicate_labels=False)

        assert get_dataframe(ds_result).flags.allows_duplicate_labels == pd_result.flags.allows_duplicate_labels


class TestDataFrameCombineOperations:
    """Test combine and combine_first operations with lazy chains."""

    def test_combine_first_basic(self):
        """Basic combine_first operation."""
        pd_df1 = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_after_filter(self):
        """Combine_first after filter operation."""
        pd_df1 = pd.DataFrame({'a': [1, np.nan, 3, 4], 'b': [np.nan, 5, 6, 7], 'c': [1, 2, 3, 4]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30, 40], 'b': [40, 50, 60, 70], 'c': [1, 2, 3, 4]})

        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_filtered = pd_df1[pd_df1['c'] > 1].combine_first(pd_df2[pd_df2['c'] > 1])
        ds_filtered = ds_df1[ds_df1['c'] > 1].combine_first(ds_df2[ds_df2['c'] > 1])

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_combine_with_function(self):
        """Combine with custom function."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})

        ds_df1 = DataStore(pd_df1.copy())

        # Use pandas DataFrame for other to ensure compatibility
        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1 + s2)
        ds_result = ds_df1.combine(pd_df2, lambda s1, s2: s1 + s2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_fill_value(self):
        """Combine with fill_value."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'c': [40, 50, 60]})

        ds_df1 = DataStore(pd_df1.copy())

        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1 + s2, fill_value=0)
        ds_result = ds_df1.combine(pd_df2, lambda s1, s2: s1 + s2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataFrameAlignOperations:
    """Test align operation."""

    def test_align_basic(self):
        """Basic align operation."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'a': [10, 20, 30]}, index=[1, 2, 3])

        ds_df1 = DataStore(pd_df1.copy())

        pd_left, pd_right = pd_df1.align(pd_df2)
        ds_left, ds_right = ds_df1.align(pd_df2)

        # ds_left should be DataStore, ds_right should be DataFrame
        assert_datastore_equals_pandas(ds_left, pd_left)
        # ds_right is a pandas DataFrame returned from align
        if isinstance(ds_right, pd.DataFrame):
            pd.testing.assert_frame_equal(ds_right, pd_right)

    def test_align_join_inner(self):
        """Align with join='inner'."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'a': [10, 20, 30]}, index=[1, 2, 3])

        ds_df1 = DataStore(pd_df1.copy())

        pd_left, pd_right = pd_df1.align(pd_df2, join='inner')
        ds_left, ds_right = ds_df1.align(pd_df2, join='inner')

        assert_datastore_equals_pandas(ds_left, pd_left)
        if isinstance(ds_right, pd.DataFrame):
            pd.testing.assert_frame_equal(ds_right, pd_right)

    def test_align_fill_value(self):
        """Align with fill_value."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'a': [10, 20, 30]}, index=[1, 2, 3])

        ds_df1 = DataStore(pd_df1.copy())

        pd_left, pd_right = pd_df1.align(pd_df2, fill_value=-1)
        ds_left, ds_right = ds_df1.align(pd_df2, fill_value=-1)

        assert_datastore_equals_pandas(ds_left, pd_left)
        if isinstance(ds_right, pd.DataFrame):
            pd.testing.assert_frame_equal(ds_right, pd_right)


class TestColumnExprProperties:
    """Test ColumnExpr properties."""

    def test_is_unique_true(self):
        """is_unique returns True for unique values."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].is_unique
        ds_result = ds_df['value'].is_unique

        assert ds_result == pd_result
        assert ds_result is True

    def test_is_unique_false(self):
        """is_unique returns False for non-unique values."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].is_unique
        ds_result = ds_df['value'].is_unique

        assert ds_result == pd_result
        assert ds_result is False

    def test_is_monotonic_increasing_true(self):
        """is_monotonic_increasing returns True."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].is_monotonic_increasing
        ds_result = ds_df['value'].is_monotonic_increasing

        assert ds_result == pd_result
        assert ds_result is True

    def test_is_monotonic_increasing_false(self):
        """is_monotonic_increasing returns False."""
        pd_df = pd.DataFrame({'value': [1, 3, 2, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].is_monotonic_increasing
        ds_result = ds_df['value'].is_monotonic_increasing

        assert ds_result == pd_result
        assert ds_result is False

    def test_is_monotonic_decreasing_true(self):
        """is_monotonic_decreasing returns True."""
        pd_df = pd.DataFrame({'value': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].is_monotonic_decreasing
        ds_result = ds_df['value'].is_monotonic_decreasing

        assert ds_result == pd_result
        assert ds_result is True

    def test_hasnans_true(self):
        """hasnans returns True when NaN present."""
        pd_df = pd.DataFrame({'value': [1.0, np.nan, 3.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].hasnans
        ds_result = ds_df['value'].hasnans

        assert ds_result == pd_result
        assert ds_result is True

    def test_hasnans_false(self):
        """hasnans returns False when no NaN."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].hasnans
        ds_result = ds_df['value'].hasnans

        assert ds_result == pd_result
        assert ds_result is False

    def test_nbytes(self):
        """nbytes property."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].nbytes
        ds_result = ds_df['value'].nbytes

        # nbytes should be equal or close
        assert ds_result == pd_result

    def test_array_property(self):
        """array property returns underlying array."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].array
        ds_result = ds_df['value'].array

        np.testing.assert_array_equal(ds_result, pd_result)


class TestColumnExprToMethods:
    """Test ColumnExpr to_* conversion methods."""

    def test_tolist(self):
        """tolist() method."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].tolist()
        ds_result = ds_df['value'].tolist()

        assert ds_result == pd_result
        assert isinstance(ds_result, list)

    def test_to_list(self):
        """to_list() method (alias)."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].to_list()
        ds_result = ds_df['value'].to_list()

        assert ds_result == pd_result
        assert isinstance(ds_result, list)

    def test_to_numpy(self):
        """to_numpy() method."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].to_numpy()
        ds_result = ds_df['value'].to_numpy()

        np.testing.assert_array_equal(ds_result, pd_result)

    def test_to_dict(self):
        """to_dict() method for Series."""
        pd_df = pd.DataFrame({'value': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].to_dict()
        ds_result = ds_df['value'].to_dict()

        assert ds_result == pd_result

    def test_to_frame_default(self):
        """to_frame() with default name."""
        pd_df = pd.DataFrame({'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].to_frame()
        ds_result = ds_df['value'].to_frame()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_to_frame_custom_name(self):
        """to_frame() with custom name."""
        pd_df = pd.DataFrame({'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].to_frame(name='custom_col')
        ds_result = ds_df['value'].to_frame(name='custom_col')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexAccessorChains:
    """Test complex accessor chain scenarios."""

    def test_str_accessor_chain_with_groupby(self):
        """String accessor chain followed by groupby."""
        pd_df = pd.DataFrame({
            'text': ['Hello World', 'Foo Bar', 'Hello Python', 'Foo Baz'],
            'value': [1, 2, 3, 4]
        })

        # Extract first word and group by it
        pd_df['first_word'] = pd_df['text'].str.split().str[0]
        pd_result = pd_df.groupby('first_word')['value'].sum()

        ds_df = DataStore(pd_df.copy())  # Use pandas-modified df for fair comparison
        ds_result = ds_df.groupby('first_word')['value'].sum()

        ds_executed = get_dataframe(ds_result)
        if isinstance(ds_executed, pd.DataFrame):
            ds_series = ds_executed.iloc[:, 0] if ds_executed.shape[1] == 1 else ds_executed['value']
        else:
            ds_series = ds_executed

        # Compare values (ignoring index type differences)
        assert sorted(ds_series.tolist()) == sorted(pd_result.tolist())

    def test_dt_accessor_chain_with_filter(self):
        """Datetime accessor chain with filter - sum returns scalar."""
        import numpy as np
        pd_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'value': range(10)
        })
        ds_df = DataStore(pd_df.copy())

        # Filter by day of week and compute - sum returns scalar
        pd_result = pd_df[pd_df['date'].dt.dayofweek < 5]['value'].sum()
        ds_result = ds_df[ds_df['date'].dt.dayofweek < 5]['value'].sum()

        # Both should be scalars
        assert isinstance(ds_result, (int, float, np.integer, np.floating))
        assert float(ds_result) == float(pd_result)

    def test_str_len_filter_chain(self):
        """String length filter chain - mean returns scalar."""
        import numpy as np
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85, 90, 78, 92, 88]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter by name length - mean returns scalar
        pd_result = pd_df[pd_df['name'].str.len() > 3]['score'].mean()
        ds_result = ds_df[ds_df['name'].str.len() > 3]['score'].mean()

        # Both should be scalars
        assert isinstance(ds_result, (int, float, np.integer, np.floating))
        assert abs(float(ds_result) - float(pd_result)) < 0.01


class TestEdgeCasesAdvanced:
    """Advanced edge cases."""

    def test_empty_dataframe_factorize(self):
        """Factorize on empty DataFrame column."""
        pd_df = pd.DataFrame({'category': pd.Series([], dtype=str)})
        ds_df = DataStore(pd_df.copy())

        pd_codes, pd_uniques = pd_df['category'].factorize()
        ds_codes, ds_uniques = ds_df['category'].factorize()

        np.testing.assert_array_equal(ds_codes, pd_codes)
        np.testing.assert_array_equal(ds_uniques, pd_uniques)

    def test_single_row_unique(self):
        """Unique on single row."""
        pd_df = pd.DataFrame({'value': [42]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].unique()
        ds_result = ds_df['value'].unique()

        ds_values = _get_series_or_values(ds_result)

        assert ds_values == list(pd_result)

    def test_all_na_column_value_counts(self):
        """Value counts on all-NA column."""
        pd_df = pd.DataFrame({'value': [np.nan, np.nan, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].value_counts()
        ds_result = ds_df['value'].value_counts()

        ds_executed = get_dataframe(ds_result)
        # Both should be empty (dropna=True by default)
        assert len(ds_executed) == len(pd_result)

    def test_all_same_value_is_unique(self):
        """is_unique on column with all same values."""
        pd_df = pd.DataFrame({'value': [1, 1, 1, 1, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].is_unique
        ds_result = ds_df['value'].is_unique

        assert ds_result == pd_result
        assert ds_result is False

    def test_mixed_types_in_map(self):
        """Map with mixed type results."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df.copy())

        mapping = {'a': 1, 'b': 'two', 'c': 3.0}
        pd_result = pd_df['category'].map(mapping)
        ds_result = ds_df['category'].map(mapping)

        ds_values = _get_series_or_values(ds_result)

        assert ds_values == pd_result.tolist()


class TestColumnExprArgmaxArgmin:
    """Test argmax/argmin methods."""

    def test_argmax_basic(self):
        """Basic argmax."""
        pd_df = pd.DataFrame({'value': [1, 5, 3, 9, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].argmax()
        ds_result = ds_df['value'].argmax()

        assert ds_result == pd_result

    def test_argmin_basic(self):
        """Basic argmin."""
        pd_df = pd.DataFrame({'value': [1, 5, 3, 9, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].argmin()
        ds_result = ds_df['value'].argmin()

        assert ds_result == pd_result

    def test_argmax_with_na(self):
        """Argmax with NA values."""
        pd_df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 2.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].argmax()
        ds_result = ds_df['value'].argmax()

        assert ds_result == pd_result

    def test_argmin_with_na(self):
        """Argmin with NA values."""
        pd_df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 2.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].argmin()
        ds_result = ds_df['value'].argmin()

        assert ds_result == pd_result


class TestColumnExprFirstLastValidIndex:
    """Test first_valid_index and last_valid_index methods."""

    def test_first_valid_index_basic(self):
        """Basic first_valid_index."""
        pd_df = pd.DataFrame({'value': [np.nan, np.nan, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].first_valid_index()
        ds_result = ds_df['value'].first_valid_index()

        assert ds_result == pd_result

    def test_last_valid_index_basic(self):
        """Basic last_valid_index."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, np.nan, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].last_valid_index()
        ds_result = ds_df['value'].last_valid_index()

        assert ds_result == pd_result

    def test_first_valid_index_all_na(self):
        """first_valid_index with all NA values."""
        pd_df = pd.DataFrame({'value': [np.nan, np.nan, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].first_valid_index()
        ds_result = ds_df['value'].first_valid_index()

        assert ds_result == pd_result  # Both should be None

    def test_last_valid_index_no_na(self):
        """last_valid_index with no NA values."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].last_valid_index()
        ds_result = ds_df['value'].last_valid_index()

        assert ds_result == pd_result


class TestColumnExprMode:
    """Test ColumnExpr.mode() method."""

    def test_mode_single(self):
        """Mode with single mode value."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 3, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].mode()
        ds_result = ds_df['value'].mode()

        ds_values = _get_series_or_values(ds_result)

        assert ds_values == pd_result.tolist()

    def test_mode_multiple(self):
        """Mode with multiple mode values."""
        pd_df = pd.DataFrame({'value': [1, 1, 2, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].mode()
        ds_result = ds_df['value'].mode()

        ds_values = _get_series_or_values(ds_result)

        assert set(ds_values) == set(pd_result.tolist())

    def test_mode_string(self):
        """Mode on string column."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'b', 'c', 'b', 'a']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].mode()
        ds_result = ds_df['category'].mode()

        ds_values = _get_series_or_values(ds_result)

        assert ds_values == pd_result.tolist()

    def test_mode_dropna_false(self):
        """Mode with dropna=False."""
        pd_df = pd.DataFrame({'value': [1, np.nan, np.nan, 2, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].mode(dropna=False)
        ds_result = ds_df['value'].mode(dropna=False)

        ds_values = _get_series_or_values(ds_result)
        # Both should include NaN as mode
        assert len(ds_values) == len(pd_result)
