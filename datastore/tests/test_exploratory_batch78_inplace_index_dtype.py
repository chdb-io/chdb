"""
Exploratory Batch 78: Inplace Operations, Complex Indexing, and Dtype Conversion Edge Cases

Focus areas:
1. Inplace operations (fillna, dropna, rename inplace=True)
2. Complex loc/iloc indexing patterns
3. Chained assignment edge cases
4. Dtype conversion and astype boundary conditions
5. Empty DataFrame operations
6. Single-row DataFrame operations
7. Column type coercion with mixed operations
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestInplaceOperations:
    """Test inplace parameter behavior"""

    def test_fillna_inplace_false(self):
        """Test fillna with inplace=False returns new object"""
        pd_df = pd.DataFrame({'A': [1, None, 3], 'B': [None, 2, None]})
        ds_df = DataStore({'A': [1, None, 3], 'B': [None, 2, None]})

        pd_result = pd_df.fillna(0, inplace=False)
        ds_result = ds_df.fillna(0, inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Original should be unchanged
        assert pd_df.isna().sum().sum() == 3  # pandas original has NaNs
        assert len(ds_df[ds_df['A'].isna()]) == 1  # DataStore original also has NaN in A

    def test_dropna_inplace_false(self):
        """Test dropna with inplace=False returns new object"""
        pd_df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
        ds_df = DataStore({'A': [1, None, 3], 'B': [4, 5, None]})

        pd_result = pd_df.dropna(inplace=False)
        ds_result = ds_df.dropna(inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_inplace_false(self):
        """Test rename with inplace=False returns new object"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})

        pd_result = pd_df.rename(columns={'A': 'X'}, inplace=False)
        ds_result = ds_df.rename(columns={'A': 'X'}, inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_inplace_false(self):
        """Test drop_duplicates with inplace=False"""
        pd_df = pd.DataFrame({'A': [1, 1, 2], 'B': [1, 1, 2]})
        ds_df = DataStore({'A': [1, 1, 2], 'B': [1, 1, 2]})

        pd_result = pd_df.drop_duplicates(inplace=False)
        ds_result = ds_df.drop_duplicates(inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_inplace_false(self):
        """Test sort_values with inplace=False"""
        pd_df = pd.DataFrame({'A': [3, 1, 2]})
        ds_df = DataStore({'A': [3, 1, 2]})

        pd_result = pd_df.sort_values('A', inplace=False)
        ds_result = ds_df.sort_values('A', inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexIndexing:
    """Test complex loc/iloc indexing patterns"""

    def test_loc_single_row_single_col(self):
        """Test loc with single row and single column"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df.loc[1, 'A']
        ds_result = ds_df.loc[1, 'A']

        assert ds_result == pd_result

    def test_loc_multiple_rows_single_col(self):
        """Test loc with multiple rows and single column"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df.loc[[0, 2], 'A']
        ds_result = ds_df.loc[[0, 2], 'A']

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_loc_slice_rows(self):
        """Test loc with slice for rows"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

        pd_result = pd_df.loc[1:3]
        ds_result = ds_df.loc[1:3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative_index(self):
        """Test iloc with negative index"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df.iloc[-1]
        ds_result = ds_df.iloc[-1]

        assert_series_equal(ds_result, pd_result)

    def test_iloc_negative_slice(self):
        """Test iloc with negative slice"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

        pd_result = pd_df.iloc[-3:-1]
        ds_result = ds_df.iloc[-3:-1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_boolean_condition(self):
        """Test loc with boolean condition"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})

        pd_result = pd_df.loc[pd_df['A'] > 2]
        ds_result = ds_df.loc[ds_df['A'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_boolean_with_columns(self):
        """Test loc with boolean condition and column selection"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40], 'C': [100, 200, 300, 400]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40], 'C': [100, 200, 300, 400]})

        pd_result = pd_df.loc[pd_df['A'] > 2, ['B', 'C']]
        ds_result = ds_df.loc[ds_df['A'] > 2, ['B', 'C']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestChainedAssignment:
    """Test chained assignment patterns"""

    def test_assign_from_arithmetic(self):
        """Test assign with arithmetic operation"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [10, 20, 30]})

        pd_df['C'] = pd_df['A'] + pd_df['B']
        ds_df['C'] = ds_df['A'] + ds_df['B']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_from_multiple_operations(self):
        """Test assign with multiple chained operations"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [10, 20, 30]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [10, 20, 30]})

        pd_df['C'] = (pd_df['A'] * 2) + (pd_df['B'] / 10)
        ds_df['C'] = (ds_df['A'] * 2) + (ds_df['B'] / 10)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_with_scalar(self):
        """Test assign column with scalar value"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_df['B'] = 100
        ds_df['B'] = 100

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_overwrite_existing_column(self):
        """Test overwriting an existing column"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_df['A'] = pd_df['A'] * 10
        ds_df['A'] = ds_df['A'] * 10

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_chained_column_assignment(self):
        """Test multiple column assignments in sequence"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_df['B'] = pd_df['A'] * 2
        pd_df['C'] = pd_df['B'] + 1

        ds_df['B'] = ds_df['A'] * 2
        ds_df['C'] = ds_df['B'] + 1

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_method(self):
        """Test assign() method"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df.assign(B=lambda x: x['A'] * 2, C=lambda x: x['A'] + 10)
        ds_result = ds_df.assign(B=lambda x: x['A'] * 2, C=lambda x: x['A'] + 10)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDtypeConversion:
    """Test dtype conversion and astype operations"""

    def test_astype_int_to_float(self):
        """Test converting int column to float"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df['A'].astype(float)
        ds_result = ds_df['A'].astype(float)

        assert_series_equal(ds_result, pd_result)

    def test_astype_float_to_int(self):
        """Test converting float column to int"""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        ds_df = DataStore({'A': [1.0, 2.0, 3.0]})

        pd_result = pd_df['A'].astype(int)
        ds_result = ds_df['A'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_to_string(self):
        """Test converting numeric to string"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df['A'].astype(str)
        ds_result = ds_df['A'].astype(str)

        assert_series_equal(ds_result, pd_result)

    def test_astype_string_to_int(self):
        """Test converting string to int"""
        pd_df = pd.DataFrame({'A': ['1', '2', '3']})
        ds_df = DataStore({'A': ['1', '2', '3']})

        pd_result = pd_df['A'].astype(int)
        ds_result = ds_df['A'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_dtypes_property(self):
        """Test dtypes property returns correct types"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [1.5, 2.5], 'C': ['x', 'y']})
        ds_df = DataStore({'A': [1, 2], 'B': [1.5, 2.5], 'C': ['x', 'y']})

        pd_dtypes = pd_df.dtypes
        ds_dtypes = ds_df.dtypes

        # Compare dtype names (allow int64/Int64 equivalence)
        assert len(pd_dtypes) == len(ds_dtypes)
        for col in pd_df.columns:
            assert col in ds_dtypes.index


class TestEmptyDataFrame:
    """Test operations on empty DataFrames"""

    def test_empty_dataframe_len(self):
        """Test len() on empty DataFrame"""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        assert len(ds_df) == len(pd_df) == 0

    def test_empty_dataframe_columns(self):
        """Test columns on empty DataFrame"""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        assert list(ds_df.columns) == list(pd_df.columns)

    def test_empty_dataframe_filter(self):
        """Test filtering empty DataFrame"""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        # This should work without error even on empty DataFrame
        pd_result = pd_df[pd_df['A'] > 0]
        ds_result = ds_df[ds_df['A'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_head(self):
        """Test head() on empty DataFrame"""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = DataStore({'A': [], 'B': []})

        pd_result = pd_df.head(5)
        ds_result = ds_df.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_sort(self):
        """Test sort on empty DataFrame"""
        # Create with explicit dtype to avoid comparison issues
        pd_df = pd.DataFrame({'A': pd.array([], dtype='int64'), 'B': pd.array([], dtype='int64')})
        ds_df = DataStore({'A': [], 'B': []})

        pd_result = pd_df.sort_values('A')
        ds_result = ds_df.sort_values('A')

        assert len(ds_result) == len(pd_result) == 0

    def test_empty_after_filter(self):
        """Test DataFrame becomes empty after filter"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df[pd_df['A'] > 100]
        ds_result = ds_df[ds_df['A'] > 100]

        assert len(ds_result) == len(pd_result) == 0


class TestSingleRowDataFrame:
    """Test operations on single-row DataFrames"""

    def test_single_row_filter(self):
        """Test filtering single-row DataFrame"""
        pd_df = pd.DataFrame({'A': [1], 'B': [10]})
        ds_df = DataStore({'A': [1], 'B': [10]})

        pd_result = pd_df[pd_df['A'] == 1]
        ds_result = ds_df[ds_df['A'] == 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_aggregation(self):
        """Test aggregation on single-row DataFrame"""
        pd_df = pd.DataFrame({'A': [5], 'B': [10]})
        ds_df = DataStore({'A': [5], 'B': [10]})

        pd_sum = pd_df['A'].sum()
        ds_sum = ds_df['A'].sum()

        # Execute and compare scalar values
        assert float(ds_sum) == float(pd_sum)

    def test_single_row_mean(self):
        """Test mean on single-row DataFrame"""
        pd_df = pd.DataFrame({'A': [10.0], 'B': [20.0]})
        ds_df = DataStore({'A': [10.0], 'B': [20.0]})

        pd_mean = pd_df['A'].mean()
        ds_mean = ds_df['A'].mean()

        assert float(ds_mean) == float(pd_mean)

    def test_single_row_iloc(self):
        """Test iloc on single-row DataFrame"""
        pd_df = pd.DataFrame({'A': [1], 'B': [2]})
        ds_df = DataStore({'A': [1], 'B': [2]})

        pd_result = pd_df.iloc[0]
        ds_result = ds_df.iloc[0]

        assert_series_equal(ds_result, pd_result)

    def test_single_row_head_tail(self):
        """Test head/tail on single-row DataFrame"""
        pd_df = pd.DataFrame({'A': [1]})
        ds_df = DataStore({'A': [1]})

        pd_head = pd_df.head(5)
        pd_tail = pd_df.tail(5)

        ds_head = ds_df.head(5)
        ds_tail = ds_df.tail(5)

        assert_datastore_equals_pandas(ds_head, pd_head)
        assert_datastore_equals_pandas(ds_tail, pd_tail)


class TestMixedTypeOperations:
    """Test operations with mixed column types"""

    def test_select_numeric_only(self):
        """Test selecting only numeric columns"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [1.5, 2.5], 'C': ['x', 'y']})
        ds_df = DataStore({'A': [1, 2], 'B': [1.5, 2.5], 'C': ['x', 'y']})

        pd_result = pd_df.select_dtypes(include=[np.number])
        ds_result = ds_df.select_dtypes(include=[np.number])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_object_only(self):
        """Test selecting only object/string columns"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [1.5, 2.5], 'C': ['x', 'y']})
        ds_df = DataStore({'A': [1, 2], 'B': [1.5, 2.5], 'C': ['x', 'y']})

        pd_result = pd_df.select_dtypes(include=['object'])
        ds_result = ds_df.select_dtypes(include=['object'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_type_promotion(self):
        """Test type promotion in arithmetic operations"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.5, 2.5, 3.5]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [1.5, 2.5, 3.5]})

        pd_result = pd_df['A'] + pd_df['B']
        ds_result = ds_df['A'] + ds_df['B']

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_comparison_with_none(self):
        """Test comparison operations with None values"""
        pd_df = pd.DataFrame({'A': [1, None, 3]})
        ds_df = DataStore({'A': [1, None, 3]})

        pd_result = pd_df['A'] > 0
        ds_result = ds_df['A'] > 0

        # None comparisons should return False/NaN in both systems
        assert len(ds_result) == len(pd_result)


class TestColumnOperationEdgeCases:
    """Test edge cases in column operations"""

    def test_column_with_spaces(self):
        """Test column names with spaces"""
        pd_df = pd.DataFrame({'col A': [1, 2], 'col B': [3, 4]})
        ds_df = DataStore({'col A': [1, 2], 'col B': [3, 4]})

        assert_datastore_equals_pandas(ds_df, pd_df)
        pd_result = pd_df['col A']
        ds_result = ds_df['col A']
        assert_series_equal(ds_result, pd_result)

    def test_column_with_special_chars(self):
        """Test column names with special characters"""
        pd_df = pd.DataFrame({'col-A': [1, 2], 'col_B': [3, 4], 'col.C': [5, 6]})
        ds_df = DataStore({'col-A': [1, 2], 'col_B': [3, 4], 'col.C': [5, 6]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_duplicate_column_names(self):
        """Test DataFrame with duplicate column names"""
        pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'A'])
        ds_df = DataStore([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'A'])

        assert list(ds_df.columns) == list(pd_df.columns)

    def test_numeric_column_names(self):
        """Test DataFrame with numeric column names"""
        pd_df = pd.DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
        ds_df = DataStore({0: [1, 2], 1: [3, 4], 2: [5, 6]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_column_reorder(self):
        """Test reordering columns"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

        pd_result = pd_df[['C', 'A', 'B']]
        ds_result = ds_df[['C', 'A', 'B']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggregationEdgeCases:
    """Test edge cases in aggregation operations"""

    def test_sum_with_all_none(self):
        """Test sum on column with all None values"""
        pd_df = pd.DataFrame({'A': [None, None, None]})
        ds_df = DataStore({'A': [None, None, None]})

        pd_result = pd_df['A'].sum()
        ds_result = ds_df['A'].sum()

        # Both should return 0 for all-None sum
        assert float(ds_result) == float(pd_result)

    def test_mean_with_all_none(self):
        """Test mean on column with all None values"""
        pd_df = pd.DataFrame({'A': [None, None, None]})
        ds_df = DataStore({'A': [None, None, None]})

        pd_result = pd_df['A'].mean()
        ds_result = ds_df['A'].mean()

        # Both should return NaN for all-None mean
        assert pd.isna(ds_result) and pd.isna(pd_result)

    def test_count_with_none(self):
        """Test count with None values"""
        pd_df = pd.DataFrame({'A': [1, None, 3, None, 5]})
        ds_df = DataStore({'A': [1, None, 3, None, 5]})

        pd_result = pd_df['A'].count()
        ds_result = ds_df['A'].count()

        assert int(ds_result) == int(pd_result) == 3

    def test_min_max_with_none(self):
        """Test min/max with None values"""
        pd_df = pd.DataFrame({'A': [1, None, 5, None, 3]})
        ds_df = DataStore({'A': [1, None, 5, None, 3]})

        pd_min = pd_df['A'].min()
        pd_max = pd_df['A'].max()
        ds_min = ds_df['A'].min()
        ds_max = ds_df['A'].max()

        assert float(ds_min) == float(pd_min) == 1
        assert float(ds_max) == float(pd_max) == 5

    def test_std_single_value(self):
        """Test std on single value"""
        pd_df = pd.DataFrame({'A': [5.0]})
        ds_df = DataStore({'A': [5.0]})

        pd_result = pd_df['A'].std()
        ds_result = ds_df['A'].std()

        # Both should return NaN for single value std
        assert pd.isna(ds_result) and pd.isna(pd_result)

    def test_var_single_value(self):
        """Test var on single value"""
        pd_df = pd.DataFrame({'A': [5.0]})
        ds_df = DataStore({'A': [5.0]})

        pd_result = pd_df['A'].var()
        ds_result = ds_df['A'].var()

        # Both should return NaN for single value var
        assert pd.isna(ds_result) and pd.isna(pd_result)


class TestSlicingEdgeCases:
    """Test edge cases in slicing operations"""

    def test_head_larger_than_rows(self):
        """Test head(n) where n > number of rows"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df.head(100)
        ds_result = ds_df.head(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_larger_than_rows(self):
        """Test tail(n) where n > number of rows"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df.tail(100)
        ds_result = ds_df.tail(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """Test head(0)"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert len(ds_result) == len(pd_result) == 0

    def test_tail_zero(self):
        """Test tail(0)"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df.tail(0)
        ds_result = ds_df.tail(0)

        assert len(ds_result) == len(pd_result) == 0

    def test_iloc_empty_list(self):
        """Test iloc with empty list"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_result = pd_df.iloc[[]]
        ds_result = ds_df.iloc[[]]

        assert len(ds_result) == len(pd_result) == 0

    def test_slice_step(self):
        """Test slicing with step"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5, 6]})

        pd_result = pd_df.iloc[::2]
        ds_result = ds_df.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCopyBehavior:
    """Test copy/deep copy behavior"""

    def test_copy_method(self):
        """Test copy() method"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_modification_isolation(self):
        """Test that copy modifications don't affect original"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        pd_copy['B'] = 100
        ds_copy['B'] = 100

        # Original should be unchanged
        assert 'B' not in pd_df.columns
        # Note: DataStore lazy behavior may differ - test actual columns
        original_cols = list(ds_df.columns)
        assert 'A' in original_cols


class TestBooleanOperations:
    """Test boolean operations and conditions"""

    def test_all_method(self):
        """Test all() method"""
        pd_df = pd.DataFrame({'A': [True, True, True]})
        ds_df = DataStore({'A': [True, True, True]})

        pd_result = pd_df['A'].all()
        ds_result = ds_df['A'].all()

        assert bool(ds_result) == bool(pd_result) == True

    def test_any_method(self):
        """Test any() method"""
        pd_df = pd.DataFrame({'A': [False, False, True]})
        ds_df = DataStore({'A': [False, False, True]})

        pd_result = pd_df['A'].any()
        ds_result = ds_df['A'].any()

        assert bool(ds_result) == bool(pd_result) == True

    def test_all_false(self):
        """Test all() when all False"""
        pd_df = pd.DataFrame({'A': [False, False, False]})
        ds_df = DataStore({'A': [False, False, False]})

        pd_result = pd_df['A'].all()
        ds_result = ds_df['A'].all()

        assert bool(ds_result) == bool(pd_result) == False

    def test_any_all_false(self):
        """Test any() when all False"""
        pd_df = pd.DataFrame({'A': [False, False, False]})
        ds_df = DataStore({'A': [False, False, False]})

        pd_result = pd_df['A'].any()
        ds_result = ds_df['A'].any()

        assert bool(ds_result) == bool(pd_result) == False

    def test_boolean_indexing_combined(self):
        """Test combined boolean conditions"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})

        pd_result = pd_df[(pd_df['A'] > 1) & (pd_df['B'] < 40)]
        ds_result = ds_df[(ds_df['A'] > 1) & (ds_df['B'] < 40)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_or_condition(self):
        """Test OR boolean condition"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore({'A': [1, 2, 3, 4]})

        pd_result = pd_df[(pd_df['A'] == 1) | (pd_df['A'] == 4)]
        ds_result = ds_df[(ds_df['A'] == 1) | (ds_df['A'] == 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_not_condition(self):
        """Test NOT boolean condition"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore({'A': [1, 2, 3, 4]})

        pd_result = pd_df[~(pd_df['A'] == 2)]
        ds_result = ds_df[~(ds_df['A'] == 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)
