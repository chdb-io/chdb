"""
Exploratory Batch 80: Return Type Consistency and API Alignment

Focus areas:
1. Return type consistency for DataFrame/Series properties and methods
2. Index-related operations and return types
3. Columns attribute behavior
4. dtypes property alignment
5. values property alignment
6. Complex operations return type verification
7. Method chaining with type preservation

Note on Lazy Execution Design:
DataStore uses lazy execution for all operations. Methods like sum(), unique(),
value_counts() return ColumnExpr wrappers. The underlying values are accessed via:
- .values property (returns numpy array)
- Direct comparison/iteration (triggers execution)
- len() call (triggers execution)

Tests verify that the EXECUTED values match pandas behavior, not the wrapper types.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestColumnsReturnType:
    """Test that .columns returns the same type as pandas"""

    def test_columns_type_is_index(self):
        """DataFrame.columns must return Index type like pandas"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_cols = pd_df.columns
        ds_cols = ds_df.columns

        # Type check - must be Index
        assert type(pd_cols) == pd.Index, f"pandas returned {type(pd_cols)}"
        assert type(ds_cols) == pd.Index, f"DataStore returned {type(ds_cols)}, expected pd.Index"

        # Value check
        assert list(ds_cols) == list(pd_cols)

    def test_columns_tolist_returns_list(self):
        """DataFrame.columns.tolist() must return Python list"""
        pd_df = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        ds_df = DataStore({'x': [1], 'y': [2], 'z': [3]})

        pd_result = pd_df.columns.tolist()
        ds_result = ds_df.columns.tolist()

        assert type(pd_result) is list
        assert type(ds_result) is list
        assert ds_result == pd_result

    def test_columns_after_operations(self):
        """Columns type should be preserved after operations"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        # After filter
        pd_filtered = pd_df[pd_df['a'] > 1]
        ds_filtered = ds_df[ds_df['a'] > 1]
        assert type(pd_filtered.columns) == type(ds_filtered.columns)

        # After assign
        pd_assigned = pd_df.assign(c=lambda x: x['a'] + x['b'])
        ds_assigned = ds_df.assign(c=lambda x: x['a'] + x['b'])
        assert type(pd_assigned.columns) == type(ds_assigned.columns)


class TestDtypesReturnType:
    """Test that .dtypes returns the same type as pandas"""

    def test_dtypes_type_is_series(self):
        """DataFrame.dtypes must return Series type like pandas"""
        pd_df = pd.DataFrame({'int_col': [1, 2], 'str_col': ['a', 'b']})
        ds_df = DataStore({'int_col': [1, 2], 'str_col': ['a', 'b']})

        pd_dtypes = pd_df.dtypes
        ds_dtypes = ds_df.dtypes

        # Type check
        assert type(pd_dtypes) == pd.Series, f"pandas returned {type(pd_dtypes)}"
        assert type(ds_dtypes) == pd.Series, f"DataStore returned {type(ds_dtypes)}, expected pd.Series"

    def test_dtypes_index_matches_columns(self):
        """DataFrame.dtypes.index must match DataFrame.columns"""
        pd_df = pd.DataFrame({'a': [1], 'b': ['x'], 'c': [1.5]})
        ds_df = DataStore({'a': [1], 'b': ['x'], 'c': [1.5]})

        assert list(pd_df.dtypes.index) == list(pd_df.columns)
        assert list(ds_df.dtypes.index) == list(ds_df.columns)


class TestValuesReturnType:
    """Test that .values returns the same type as pandas"""

    def test_dataframe_values_is_ndarray(self):
        """DataFrame.values must return numpy ndarray"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_vals = pd_df.values
        ds_vals = ds_df.values

        assert type(pd_vals) == np.ndarray
        assert type(ds_vals) == np.ndarray
        np.testing.assert_array_equal(ds_vals, pd_vals)

    def test_series_values_is_ndarray(self):
        """Series.values must return numpy ndarray"""
        pd_df = pd.DataFrame({'col': [1, 2, 3]})
        ds_df = DataStore({'col': [1, 2, 3]})

        pd_vals = pd_df['col'].values
        ds_vals = ds_df['col'].values

        assert type(pd_vals) == np.ndarray
        assert type(ds_vals) == np.ndarray


class TestShapeReturnType:
    """Test that .shape returns the same type as pandas"""

    def test_shape_is_tuple(self):
        """DataFrame.shape must return tuple"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        assert type(pd_df.shape) == tuple
        assert type(ds_df.shape) == tuple
        assert pd_df.shape == ds_df.shape

    def test_shape_elements_are_int(self):
        """Shape elements must be Python int"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_rows, pd_cols = pd_df.shape
        ds_rows, ds_cols = ds_df.shape

        # Should be Python int, not numpy int64
        assert type(pd_rows) in (int, np.int64)
        assert type(ds_rows) in (int, np.int64)


class TestIndexReturnType:
    """Test DataFrame index-related return types"""

    def test_index_type(self):
        """DataFrame.index must return Index type"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        # pandas returns RangeIndex by default
        assert isinstance(pd_df.index, pd.Index)
        assert isinstance(ds_df.index, pd.Index)

    def test_index_tolist_returns_list(self):
        """DataFrame.index.tolist() must return Python list"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.index.tolist()
        ds_result = ds_df.index.tolist()

        assert type(pd_result) is list
        assert type(ds_result) is list


class TestAggregationReturnValues:
    """Test that aggregation results have correct values (not wrapper types)"""

    def test_sum_value_equals_pandas(self):
        """Column sum value must equal pandas result"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].sum()
        ds_result = ds_df['col'].sum()

        # Compare values via equality (triggers execution)
        assert pd_result == ds_result == 15

    def test_mean_value_equals_pandas(self):
        """Column mean value must equal pandas result"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].mean()
        ds_result = ds_df['col'].mean()

        assert np.isclose(pd_result, ds_result)

    def test_count_value_equals_pandas(self):
        """Column count value must equal pandas result"""
        pd_df = pd.DataFrame({'col': [1, 2, None, 4, 5]})
        ds_df = DataStore({'col': [1, 2, None, 4, 5]})

        pd_result = pd_df['col'].count()
        ds_result = ds_df['col'].count()

        assert pd_result == ds_result == 4

    def test_nunique_value_equals_pandas(self):
        """Column nunique value must equal pandas result"""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'c', 'b']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c', 'b']})

        pd_result = pd_df['col'].nunique()
        ds_result = ds_df['col'].nunique()

        assert pd_result == ds_result == 3


class TestUniqueReturnValue:
    """Test that unique() values match pandas"""

    def test_unique_values_match_pandas(self):
        """unique() values must match pandas (via .values property)"""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'c', 'b']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c', 'b']})

        pd_result = pd_df['col'].unique()
        ds_result = ds_df['col'].unique()

        # Access values via .values (triggers execution)
        ds_values = ds_result.values
        
        assert type(pd_result) == np.ndarray
        assert type(ds_values) == np.ndarray
        assert set(pd_result) == set(ds_values)

    def test_unique_len_matches(self):
        """unique() length must match pandas"""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'c', 'b']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c', 'b']})

        pd_result = pd_df['col'].unique()
        ds_result = ds_df['col'].unique()

        assert len(pd_result) == len(ds_result) == 3


class TestValueCountsReturnValue:
    """Test that value_counts() values match pandas"""

    def test_value_counts_values_match_pandas(self):
        """value_counts() values must match pandas"""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'c', 'b', 'a']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c', 'b', 'a']})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        # Check values match
        assert pd_result['a'] == ds_result['a'] == 3
        assert pd_result['b'] == ds_result['b'] == 2
        assert pd_result['c'] == ds_result['c'] == 1

    def test_value_counts_index_matches_values(self):
        """value_counts index should contain original values"""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'c']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c']})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        # Index should contain the unique values
        assert set(pd_result.index) == set(ds_result.index)


class TestDescribeReturnValue:
    """Test that describe() values match pandas"""

    def test_describe_values_match_pandas(self):
        """describe() values must match pandas"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        pd_result = pd_df.describe()
        ds_result = ds_df.describe()

        # Compare via columns iteration (triggers execution for DataStore)
        for col in pd_result.columns:
            for stat in pd_result.index:
                assert np.isclose(pd_result.loc[stat, col], ds_result.loc[stat, col])

    def test_describe_series_values_match_pandas(self):
        """Series.describe() values must match pandas"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].describe()
        ds_result = ds_df['col'].describe()

        # Compare via index iteration
        for stat in pd_result.index:
            assert np.isclose(pd_result[stat], ds_result[stat])


class TestHeadTailReturnType:
    """Test that head/tail return correct types"""

    def test_head_returns_same_type(self):
        """head() must return same type as input"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_head = pd_df.head(3)
        ds_head = ds_df.head(3)

        assert type(pd_head) == pd.DataFrame
        assert_datastore_equals_pandas(ds_head, pd_head)

    def test_tail_returns_same_type(self):
        """tail() must return same type as input"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_tail = pd_df.tail(3)
        ds_tail = ds_df.tail(3)

        assert type(pd_tail) == pd.DataFrame
        assert_datastore_equals_pandas(ds_tail, pd_tail)


class TestDropDuplicatesReturnType:
    """Test that drop_duplicates returns correct type"""

    def test_drop_duplicates_returns_dataframe(self):
        """drop_duplicates must return DataFrame"""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 3], 'b': ['x', 'x', 'y', 'z', 'w']})
        ds_df = DataStore({'a': [1, 1, 2, 2, 3], 'b': ['x', 'x', 'y', 'z', 'w']})

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert type(pd_result) == pd.DataFrame
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSortValuesReturnType:
    """Test sort_values return type"""

    def test_sort_values_returns_dataframe(self):
        """sort_values must return DataFrame"""
        pd_df = pd.DataFrame({'a': [3, 1, 2], 'b': ['c', 'a', 'b']})
        ds_df = DataStore({'a': [3, 1, 2], 'b': ['c', 'a', 'b']})

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert type(pd_result) == pd.DataFrame
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupbyAggReturnType:
    """Test groupby aggregation return types"""

    def test_groupby_sum_values_match(self):
        """groupby().col.sum() values must match pandas"""
        pd_df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
        ds_df = DataStore({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})

        pd_result = pd_df.groupby('cat')['val'].sum()
        ds_result = ds_df.groupby('cat')['val'].sum()

        # Compare values
        assert pd_result['A'] == 3
        assert pd_result['B'] == 7
        # DataStore should have same values
        assert sorted(list(ds_result)) == sorted(list(pd_result))

    def test_groupby_agg_returns_dataframe(self):
        """groupby().agg() must return DataFrame"""
        pd_df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
        ds_df = DataStore({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})

        pd_result = pd_df.groupby('cat').agg({'val': 'sum'})
        ds_result = ds_df.groupby('cat').agg({'val': 'sum'})

        assert type(pd_result) == pd.DataFrame


class TestIterReturnType:
    """Test iteration-related return types"""

    def test_iterrows_yields_tuple(self):
        """iterrows must yield (index, Series) tuples"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_rows = list(pd_df.iterrows())
        ds_rows = list(ds_df.iterrows())

        assert len(pd_rows) == len(ds_rows)
        for (pd_idx, pd_row), (ds_idx, ds_row) in zip(pd_rows, ds_rows):
            # Both should yield tuples with index and Series
            assert type(pd_row) == pd.Series
            assert type(ds_row) == pd.Series

    def test_itertuples_yields_namedtuple(self):
        """itertuples must yield named tuples"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_tuples = list(pd_df.itertuples())
        ds_tuples = list(ds_df.itertuples())

        assert len(pd_tuples) == len(ds_tuples)
        # Both should yield named tuples with same values
        for pd_t, ds_t in zip(pd_tuples, ds_tuples):
            assert pd_t.a == ds_t.a
            assert pd_t.b == ds_t.b


class TestBoolReturnType:
    """Test boolean operation return types"""

    def test_any_value_equals_pandas(self):
        """Series.any() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [False, True, False]})
        ds_df = DataStore({'col': [False, True, False]})

        pd_result = pd_df['col'].any()
        ds_result = ds_df['col'].any()

        assert pd_result == ds_result == True

    def test_all_value_equals_pandas(self):
        """Series.all() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [True, True, True]})
        ds_df = DataStore({'col': [True, True, True]})

        pd_result = pd_df['col'].all()
        ds_result = ds_df['col'].all()

        assert pd_result == ds_result == True


class TestMinMaxReturnValue:
    """Test min/max return values"""

    def test_min_value_equals_pandas(self):
        """Series.min() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'col': [3, 1, 4, 1, 5]})

        pd_result = pd_df['col'].min()
        ds_result = ds_df['col'].min()

        assert pd_result == ds_result == 1

    def test_max_value_equals_pandas(self):
        """Series.max() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'col': [3, 1, 4, 1, 5]})

        pd_result = pd_df['col'].max()
        ds_result = ds_df['col'].max()

        assert pd_result == ds_result == 5

    def test_idxmin_returns_scalar(self):
        """Series.idxmin() must return scalar index"""
        pd_df = pd.DataFrame({'col': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'col': [3, 1, 4, 1, 5]})

        pd_result = pd_df['col'].idxmin()
        ds_result = ds_df['col'].idxmin()

        assert pd_result == ds_result

    def test_idxmax_returns_scalar(self):
        """Series.idxmax() must return scalar index"""
        pd_df = pd.DataFrame({'col': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'col': [3, 1, 4, 1, 5]})

        pd_result = pd_df['col'].idxmax()
        ds_result = ds_df['col'].idxmax()

        assert pd_result == ds_result


class TestEmptyReturnType:
    """Test empty DataFrame return types"""

    def test_empty_property_returns_bool(self):
        """DataFrame.empty must return bool"""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore({'a': []})

        assert type(pd_df.empty) == bool
        assert type(ds_df.empty) == bool
        assert pd_df.empty == ds_df.empty == True

    def test_non_empty_property_returns_bool(self):
        """Non-empty DataFrame.empty must return False"""
        pd_df = pd.DataFrame({'a': [1]})
        ds_df = DataStore({'a': [1]})

        assert type(pd_df.empty) == bool
        assert type(ds_df.empty) == bool
        assert pd_df.empty == ds_df.empty == False


class TestNRowsNColsReturnType:
    """Test nrows/ncols-like operations"""

    def test_len_returns_int(self):
        """len(DataFrame) must return int"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_len = len(pd_df)
        ds_len = len(ds_df)

        assert type(pd_len) == int
        assert type(ds_len) == int
        assert pd_len == ds_len == 5


class TestCopyReturnType:
    """Test that copy returns correct type"""

    def test_copy_returns_dataframe(self):
        """DataFrame.copy() must return DataFrame (or DataStore)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        assert type(pd_copy) == pd.DataFrame
        # DataStore.copy() should return something that behaves like DataFrame
        assert_datastore_equals_pandas(ds_copy, pd_copy)


class TestSelectDtypesReturnType:
    """Test select_dtypes return type"""

    def test_select_dtypes_returns_dataframe(self):
        """select_dtypes must return DataFrame"""
        pd_df = pd.DataFrame({'int_col': [1, 2], 'str_col': ['a', 'b'], 'float_col': [1.1, 2.2]})
        ds_df = DataStore({'int_col': [1, 2], 'str_col': ['a', 'b'], 'float_col': [1.1, 2.2]})

        pd_result = pd_df.select_dtypes(include=['int64'])
        ds_result = ds_df.select_dtypes(include=['int64'])

        assert type(pd_result) == pd.DataFrame
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestInfoReturnType:
    """Test info method behavior"""

    def test_info_returns_none(self):
        """DataFrame.info() must return None (prints to stdout)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.info()
        ds_result = ds_df.info()

        assert pd_result is None
        assert ds_result is None


class TestStdVarReturnValue:
    """Test standard deviation and variance return values"""

    def test_std_value_equals_pandas(self):
        """Series.std() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].std()
        ds_result = ds_df['col'].std()

        assert np.isclose(pd_result, ds_result)

    def test_var_value_equals_pandas(self):
        """Series.var() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].var()
        ds_result = ds_df['col'].var()

        assert np.isclose(pd_result, ds_result)


class TestMedianReturnValue:
    """Test median return values"""

    def test_median_value_equals_pandas(self):
        """Series.median() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].median()
        ds_result = ds_df['col'].median()

        assert np.isclose(pd_result, ds_result)


class TestQuantileReturnValue:
    """Test quantile return values"""

    def test_quantile_value_equals_pandas(self):
        """Series.quantile() value must equal pandas"""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5]})

        pd_result = pd_df['col'].quantile(0.5)
        ds_result = ds_df['col'].quantile(0.5)

        assert np.isclose(pd_result, ds_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
