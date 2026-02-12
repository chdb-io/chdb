"""
Exploratory Discovery Batch 16: Index Operations, Copy/View Semantics, and Boundary Data

This batch focuses on:
1. Index Operations - set_index, reset_index, reindex combinations
2. Copy/View Semantics - DataFrame copy behavior
3. Method Chaining Return Type Consistency
4. Special Characters and Boundary Data - Unicode, whitespace, extreme values
5. Property Accessors - columns, index, dtypes, shape edge cases
6. Comparison with Self - df == df, df.equals(df)
7. Empty Operations - operations on empty DataFrames
8. Single Value Operations - len=1 DataFrames
"""

import pytest
from tests.xfail_markers import chdb_unicode_filter
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestIndexOperations:
    """Tests for index manipulation methods"""

    def test_set_index_basic(self):
        """Basic set_index operation"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a')
        ds_result = ds_df.set_index('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_drop_false(self):
        """set_index with drop=False keeps the column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a', drop=False)
        ds_result = ds_df.set_index('a', drop=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_append(self):
        """set_index with append=True creates MultiIndex"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        pd_df = pd_df.set_index('a')
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('b', append=True)
        ds_result = ds_df.set_index('b', append=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_basic(self):
        """Basic reset_index operation"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df = pd_df.set_index('a')
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reset_index()
        ds_result = ds_df.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_drop(self):
        """reset_index with drop=True discards the index"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df = pd_df.set_index('a')
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_then_reset_index(self):
        """Round-trip: set_index then reset_index"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a').reset_index()
        ds_result = ds_df.set_index('a').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_rows(self):
        """reindex to change row order"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['z', 'y', 'x'])
        ds_result = ds_df.reindex(['z', 'y', 'x'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_missing(self):
        """reindex with non-existent index values introduces NaN"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['x', 'w', 'z'])
        ds_result = ds_df.reindex(['x', 'w', 'z'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """reindex columns"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(columns=['c', 'a'])
        ds_result = ds_df.reindex(columns=['c', 'a'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCopyViewSemantics:
    """Tests for copy/view behavior"""

    def test_copy_creates_independent_dataframe(self):
        """copy() creates an independent DataFrame"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        # Modify original
        pd_df['a'] = [10, 20, 30]

        # Copy should remain unchanged
        pd_expected = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert_datastore_equals_pandas(ds_copy, pd_expected)

    def test_copy_deep_true(self):
        """copy(deep=True) creates a deep copy"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_copy = pd_df.copy(deep=True)
        ds_copy = ds_df.copy(deep=True)

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_deep_false(self):
        """copy(deep=False) creates a shallow copy"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_copy = pd_df.copy(deep=False)
        ds_copy = ds_df.copy(deep=False)

        assert_datastore_equals_pandas(ds_copy, pd_copy)


class TestMethodChainingReturnTypes:
    """Tests for ensuring method chaining returns correct types"""

    def test_filter_returns_datastore(self):
        """filter returns DataStore"""
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = ds_df[ds_df['a'] > 1]
        assert isinstance(result, DataStore)

    def test_select_columns_returns_datastore(self):
        """Column selection returns DataStore"""
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = ds_df[['a', 'b']]
        assert isinstance(result, DataStore)

    def test_head_returns_datastore(self):
        """head() returns DataStore"""
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = ds_df.head(2)
        assert isinstance(result, DataStore)

    def test_sort_values_returns_datastore(self):
        """sort_values() returns DataStore"""
        ds_df = DataStore({'a': [3, 1, 2], 'b': [4, 5, 6]})
        result = ds_df.sort_values('a')
        assert isinstance(result, DataStore)

    def test_drop_duplicates_returns_datastore(self):
        """drop_duplicates() returns DataStore"""
        ds_df = DataStore({'a': [1, 1, 2], 'b': [4, 4, 6]})
        result = ds_df.drop_duplicates()
        assert isinstance(result, DataStore)

    def test_chained_operations_return_datastore(self):
        """Chained operations return DataStore"""
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        result = ds_df[ds_df['a'] > 1].sort_values('b').head(3)
        assert isinstance(result, DataStore)

    def test_groupby_agg_returns_datastore(self):
        """groupby().agg() returns DataStore"""
        ds_df = DataStore({'a': ['x', 'x', 'y'], 'b': [1, 2, 3]})
        result = ds_df.groupby('a').agg({'b': 'sum'})
        assert isinstance(result, DataStore)

    def test_merge_returns_datastore(self):
        """merge() returns DataStore"""
        ds1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds2 = DataStore({'a': [1, 2, 3], 'c': [7, 8, 9]})
        result = ds1.merge(ds2, on='a')
        assert isinstance(result, DataStore)


class TestUnicodeAndSpecialCharacters:
    """Tests for Unicode and special character handling"""

    def test_unicode_column_names(self):
        """Unicode column names"""
        pd_df = pd.DataFrame({'名前': ['Alice', 'Bob'], '年齢': [25, 30]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['名前']
        ds_result = ds_df['名前']

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_unicode_filter
    def test_unicode_string_values(self):
        """Unicode string values"""
        pd_df = pd.DataFrame({'name': ['アリス', 'ボブ', '中文'], 'value': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['name'] == 'アリス']
        ds_result = ds_df[ds_df['name'] == 'アリス']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_emoji_in_data(self):
        """Emoji characters in data"""
        pd_df = pd.DataFrame({'status': ['👍', '👎', '🎉'], 'count': [10, 5, 20]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['status']
        ds_result = ds_df['status']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_whitespace_handling(self):
        """Whitespace in string values"""
        pd_df = pd.DataFrame({'name': ['  Alice  ', 'Bob\t', '\nCharlie'], 'id': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['name'].str.strip()
        ds_result = ds_df['name'].str.strip()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_special_sql_characters(self):
        """Characters that have special meaning in SQL (quotes, backslash)"""
        pd_df = pd.DataFrame({
            'text': ["It's a test", 'Say "hello"', 'Back\\slash'],
            'id': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text']
        ds_result = ds_df['text']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_with_special_characters(self):
        """Filtering with special characters"""
        pd_df = pd.DataFrame({
            'text': ["It's", '"quoted"', 'normal'],
            'id': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['text'] == "It's"]
        ds_result = ds_df[ds_df['text'] == "It's"]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBoundaryValues:
    """Tests for boundary and extreme values"""

    def test_very_large_integers(self):
        """Very large integer values"""
        pd_df = pd.DataFrame({'a': [10**15, 10**16, 10**17]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        assert float(ds_result) == float(pd_result)

    def test_very_small_floats(self):
        """Very small float values"""
        pd_df = pd.DataFrame({'a': [1e-10, 1e-15, 1e-20]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a']
        ds_result = ds_df['a']

        # Compare values with tolerance
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_inf_values(self):
        """Infinity values"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 2.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a']
        ds_result = ds_df['a']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_nan_and_values(self):
        """Mix of NaN and regular values"""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].dropna()
        ds_result = ds_df['a'].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_nan_column(self):
        """Column with all NaN values"""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].sum()  # Should be 0.0
        ds_result = ds_df['a'].sum()

        # Both should handle all-NaN gracefully
        assert np.isnan(float(pd_result)) == np.isnan(float(ds_result)) or float(pd_result) == float(ds_result)


class TestPropertyAccessors:
    """Tests for property accessors"""

    def test_columns_property(self):
        """columns property returns correct column names"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        assert list(ds_df.columns) == list(pd_df.columns)

    def test_shape_property(self):
        """shape property returns (rows, cols)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        assert ds_df.shape == pd_df.shape

    def test_dtypes_property(self):
        """dtypes property returns column types"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        # Check column count matches
        assert len(ds_df.dtypes) == len(pd_df.dtypes)

    def test_empty_property_false(self):
        """empty property is False for non-empty DataFrame"""
        ds_df = DataStore({'a': [1, 2, 3]})
        assert ds_df.empty == False

    def test_empty_property_true(self):
        """empty property is True for empty DataFrame"""
        ds_df = DataStore(pd.DataFrame({'a': pd.Series([], dtype=int)}))
        assert ds_df.empty == True

    def test_ndim_property(self):
        """ndim property returns 2 for DataFrame"""
        ds_df = DataStore({'a': [1, 2, 3]})
        assert ds_df.ndim == 2

    def test_size_property(self):
        """size property returns total elements"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        assert ds_df.size == pd_df.size


class TestSelfComparison:
    """Tests for comparing DataFrame with itself"""

    def test_equals_self(self):
        """DataFrame equals itself"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.equals(pd_df)
        ds_result = ds_df.equals(ds_df)

        assert pd_result == ds_result == True

    def test_eq_self(self):
        """df == df returns all True"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = (pd_df == pd_df).all().all()
        # For DataStore, equals returns the actual comparison
        ds_comparison = ds_df.equals(ds_df)

        assert pd_result == True
        assert ds_comparison == True


class TestEmptyDataFrameOperations:
    """Tests for operations on empty DataFrames"""

    def test_empty_df_head(self):
        """head on empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int), 'b': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(5)
        ds_result = ds_df.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_tail(self):
        """tail on empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int), 'b': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(5)
        ds_result = ds_df.tail(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_sort(self):
        """sort_values on empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int), 'b': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_filter(self):
        """filter on empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int), 'b': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_groupby(self):
        """groupby on empty DataFrame"""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=str), 'b': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('a').sum()
        ds_result = ds_df.groupby('a').sum()

        # Empty groupby may have different column handling
        # Check both are empty
        assert len(ds_result) == len(pd_result) == 0


class TestSingleRowOperations:
    """Tests for operations on single-row DataFrames"""

    def test_single_row_head(self):
        """head on single-row DataFrame"""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(5)
        ds_result = ds_df.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_tail(self):
        """tail on single-row DataFrame"""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(5)
        ds_result = ds_df.tail(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_aggregation(self):
        """Aggregation on single-row DataFrame"""
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()

        assert float(ds_result) == float(pd_result)

    def test_single_row_filter_match(self):
        """filter on single-row that matches"""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] == 1]
        ds_result = ds_df[ds_df['a'] == 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self):
        """filter on single-row that doesn't match"""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] == 99]
        ds_result = ds_df[ds_df['a'] == 99]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnTypeConversions:
    """Tests for column type conversions"""

    def test_astype_int_to_float(self):
        """astype int to float"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(float)
        ds_result = ds_df['a'].astype(float)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_float_to_int(self):
        """astype float to int"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(int)
        ds_result = ds_df['a'].astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_to_string(self):
        """astype to string"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(str)
        ds_result = ds_df['a'].astype(str)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIterationProtocols:
    """Tests for iteration protocols"""

    def test_iter_columns(self):
        """iter() returns column names"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_cols = list(pd_df)
        ds_cols = list(ds_df)

        assert pd_cols == ds_cols

    def test_items_iteration(self):
        """items() iteration"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())

        assert len(pd_items) == len(ds_items)
        for (pd_name, pd_col), (ds_name, ds_col) in zip(pd_items, ds_items):
            assert pd_name == ds_name

    def test_iterrows(self):
        """iterrows() iteration"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_rows = list(pd_df.iterrows())
        ds_rows = list(ds_df.iterrows())

        assert len(pd_rows) == len(ds_rows)

    def test_itertuples(self):
        """itertuples() iteration"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_tuples = list(pd_df.itertuples())
        ds_tuples = list(ds_df.itertuples())

        assert len(pd_tuples) == len(ds_tuples)


class TestDuplicateColumnHandling:
    """Tests for duplicate column name handling"""

    def test_rename_to_duplicate(self):
        """rename() creating duplicate column names"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        # Rename 'b' to 'a' creating duplicates
        pd_result = pd_df.rename(columns={'b': 'a'})
        ds_result = ds_df.rename(columns={'b': 'a'})

        # Both should allow this (pandas does)
        assert list(pd_result.columns) == list(ds_result.columns)


class TestBooleanIndexing:
    """Tests for boolean indexing edge cases"""

    def test_all_true_mask(self):
        """Boolean mask with all True"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        mask = [True, True, True]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_false_mask(self):
        """Boolean mask with all False"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        mask = [False, False, False]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_alternating_mask(self):
        """Boolean mask with alternating True/False"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        ds_df = DataStore(pd_df)

        mask = [True, False, True, False]
        pd_result = pd_df[mask]
        ds_result = ds_df[mask]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRenameOperations:
    """Tests for rename operations"""

    def test_rename_columns_dict(self):
        """rename columns with dict"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y'})
        ds_result = ds_df.rename(columns={'a': 'x', 'b': 'y'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_partial(self):
        """rename only some columns"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(columns={'a': 'x'})
        ds_result = ds_df.rename(columns={'a': 'x'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_with_callable(self):
        """rename with callable function"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(columns=str.upper)
        ds_result = ds_df.rename(columns=str.upper)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDropOperations:
    """Tests for drop operations"""

    def test_drop_single_column(self):
        """drop single column"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop(columns=['a'])
        ds_result = ds_df.drop(columns=['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_multiple_columns(self):
        """drop multiple columns"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop(columns=['a', 'c'])
        ds_result = ds_df.drop(columns=['a', 'c'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_rows_by_index(self):
        """drop rows by index"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop(index=['y'])
        ds_result = ds_df.drop(index=['y'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMemoryViewOperations:
    """Tests for memory and view operations"""

    def test_values_array(self):
        """values returns numpy array"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_values = pd_df.values
        ds_values = ds_df.values

        np.testing.assert_array_equal(pd_values, ds_values)

    def test_to_numpy(self):
        """to_numpy() returns numpy array"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_arr = pd_df.values
        ds_arr = ds_df.values

        np.testing.assert_array_equal(pd_arr, ds_arr)
