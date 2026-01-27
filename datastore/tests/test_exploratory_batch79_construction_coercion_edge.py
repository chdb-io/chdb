"""
Exploratory Batch 79: DataFrame Construction, Type Coercion, and Edge Cases

Focus areas:
1. DataFrame construction with unusual input types
2. Type coercion in arithmetic operations
3. Conditional assignment (where/mask) with various fill values
4. String operations on boundary values (empty, whitespace, special chars)
5. GroupBy with multiple aggregations on same column
6. Arithmetic with None/NaN propagation
7. Column operations after filtering
8. Chained boolean operations with NULL handling
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestDataFrameConstructionEdgeCases:
    """Test DataFrame construction with unusual inputs"""

    def test_construct_from_dict_with_scalar_value(self):
        """Test construction from dict where one value is scalar (broadcasts)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': 'constant'})
        ds_df = DataStore({'a': [1, 2, 3], 'b': 'constant'})
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_dict_with_numpy_array(self):
        """Test construction from dict with numpy arrays"""
        arr = np.array([10, 20, 30])
        pd_df = pd.DataFrame({'values': arr, 'doubled': arr * 2})
        ds_df = DataStore({'values': arr, 'doubled': arr * 2})
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_dict_with_mixed_types(self):
        """Test construction from dict with mixed numeric types"""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'bool_col': [True, False, True]
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'bool_col': [True, False, True]
        })
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_list_of_dicts(self):
        """Test construction from list of dictionaries"""
        data = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30},
            {'name': 'Charlie', 'age': 35}
        ]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(pd.DataFrame(data))  # DataStore requires DataFrame
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_empty_dataframe(self):
        """Test construction of empty DataFrame"""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore({'a': [], 'b': []})
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_single_row(self):
        """Test construction of single-row DataFrame"""
        pd_df = pd.DataFrame({'x': [42], 'y': ['hello']})
        ds_df = DataStore({'x': [42], 'y': ['hello']})
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_single_column(self):
        """Test construction of single-column DataFrame"""
        pd_df = pd.DataFrame({'only_col': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'only_col': [1, 2, 3, 4, 5]})
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestTypeCoercionInOperations:
    """Test type coercion in various operations"""

    def test_int_plus_float_coercion(self):
        """Test int column + float constant -> float result"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_df['result'] = pd_df['a'] + 0.5
        ds_df['result'] = ds_df['a'] + 0.5

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_int_division_coercion(self):
        """Test integer division returns float"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_df['result'] = pd_df['a'] / 2
        ds_df['result'] = ds_df['a'] / 2

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_bool_to_int_coercion(self):
        """Test boolean column in arithmetic becomes int"""
        pd_df = pd.DataFrame({'flag': [True, False, True, False]})
        ds_df = DataStore({'flag': [True, False, True, False]})

        pd_result = pd_df['flag'].sum()
        ds_result = ds_df['flag'].sum()

        assert pd_result == ds_result

    def test_mixed_int_float_comparison(self):
        """Test comparison between int column and float value"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df['a'] > 2.5]
        ds_result = ds_df[ds_df['a'] > 2.5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_numeric_column_filter(self):
        """Test filtering with string representation of numbers"""
        pd_df = pd.DataFrame({'val': ['1', '2', '3', '10', '20']})
        ds_df = DataStore({'val': ['1', '2', '3', '10', '20']})

        pd_result = pd_df[pd_df['val'] == '2']
        ds_result = ds_df[ds_df['val'] == '2']

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWhereAndMaskOperations:
    """Test conditional assignment operations"""

    def test_where_with_scalar_other(self):
        """Test where() with scalar replacement value"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].where(pd_df['a'] > 2, other=-1)
        ds_result = ds_df['a'].where(ds_df['a'] > 2, other=-1)

        assert_series_equal(ds_result, pd_result)

    def test_mask_with_scalar_other(self):
        """Test mask() with scalar replacement value"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].mask(pd_df['a'] > 2, other=0)
        ds_result = ds_df['a'].mask(ds_df['a'] > 2, other=0)

        assert_series_equal(ds_result, pd_result)

    def test_where_with_nan_default(self):
        """Test where() defaults to NaN when condition is False"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].where(pd_df['a'] > 3)
        ds_result = ds_df['a'].where(ds_df['a'] > 3)

        assert_series_equal(ds_result, pd_result)

    def test_dataframe_where(self):
        """Test DataFrame-level where()"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })

        pd_result = pd_df.where(pd_df > 2, other=0)
        ds_result = ds_df.where(ds_df > 2, other=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringOperationEdgeCases:
    """Test string operations on boundary values"""

    def test_str_len_on_empty_strings(self):
        """Test string length on empty strings"""
        pd_df = pd.DataFrame({'text': ['hello', '', 'world', '']})
        ds_df = DataStore({'text': ['hello', '', 'world', '']})

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        assert_series_equal(ds_result, pd_result)

    def test_str_contains_on_empty_string(self):
        """Test str.contains on column with empty strings"""
        pd_df = pd.DataFrame({'text': ['apple', '', 'banana', 'grape']})
        ds_df = DataStore({'text': ['apple', '', 'banana', 'grape']})

        pd_result = pd_df[pd_df['text'].str.contains('a', regex=False)]
        ds_result = ds_df[ds_df['text'].str.contains('a', regex=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_strip_whitespace_only(self):
        """Test strip on whitespace-only strings"""
        pd_df = pd.DataFrame({'text': ['  hello  ', '   ', 'world', '\t\n']})
        ds_df = DataStore({'text': ['  hello  ', '   ', 'world', '\t\n']})

        pd_result = pd_df['text'].str.strip()
        ds_result = ds_df['text'].str.strip()

        assert_series_equal(ds_result, pd_result)

    def test_str_upper_on_mixed_case(self):
        """Test upper on mixed case strings"""
        pd_df = pd.DataFrame({'text': ['Hello', 'WORLD', 'PyThOn', '123abc']})
        ds_df = DataStore({'text': ['Hello', 'WORLD', 'PyThOn', '123abc']})

        pd_result = pd_df['text'].str.upper()
        ds_result = ds_df['text'].str.upper()

        assert_series_equal(ds_result, pd_result)

    def test_str_replace_no_match(self):
        """Test str.replace when pattern doesn't match"""
        pd_df = pd.DataFrame({'text': ['hello', 'world', 'python']})
        ds_df = DataStore({'text': ['hello', 'world', 'python']})

        pd_result = pd_df['text'].str.replace('xyz', 'abc', regex=False)
        ds_result = ds_df['text'].str.replace('xyz', 'abc', regex=False)

        assert_series_equal(ds_result, pd_result)

    @pytest.mark.xfail(reason="chDB returns uint8 for startsWith, pandas returns bool - known dtype issue")
    def test_str_startswith_empty_prefix(self):
        """Test startswith with empty string prefix"""
        pd_df = pd.DataFrame({'text': ['hello', 'world', '']})
        ds_df = DataStore({'text': ['hello', 'world', '']})

        pd_result = pd_df['text'].str.startswith('')
        ds_result = ds_df['text'].str.startswith('')

        assert_series_equal(ds_result, pd_result)


class TestGroupByMultipleAggregations:
    """Test GroupBy with multiple aggregations on same column"""

    def test_groupby_agg_dict_same_column(self):
        """Test groupby with multiple aggs on same column via dict"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.groupby('category')['value'].agg(['sum', 'mean', 'count'])
        ds_result = ds_df.groupby('category')['value'].agg(['sum', 'mean', 'count'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_named_agg(self):
        """Test groupby with named aggregations"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby('category').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )
        ds_result = ds_df.groupby('category').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_multiple_columns_agg(self):
        """Test groupby on multiple columns with aggregation"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'subcategory': ['X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B'],
            'subcategory': ['X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby(['category', 'subcategory'])['value'].sum()
        ds_result = ds_df.groupby(['category', 'subcategory'])['value'].sum()

        assert_series_equal(ds_result, pd_result, check_index=False)


class TestArithmeticWithNullPropagation:
    """Test arithmetic operations with NULL/NaN values"""

    def test_add_with_nan(self):
        """Test addition where one operand is NaN"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [10.0, 20.0, np.nan]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0], 'b': [10.0, 20.0, np.nan]})

        pd_df['result'] = pd_df['a'] + pd_df['b']
        ds_df['result'] = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_multiply_with_none(self):
        """Test multiplication with None in integer column"""
        pd_df = pd.DataFrame({'a': pd.array([1, None, 3], dtype=pd.Int64Dtype())})
        ds_df = DataStore({'a': pd.array([1, None, 3], dtype=pd.Int64Dtype())})

        pd_df['result'] = pd_df['a'] * 2
        ds_df['result'] = ds_df['a'] * 2

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_division_by_zero(self):
        """Test division by zero produces inf"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1.0, 0.0, 1.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [1.0, 0.0, 1.0]})

        pd_df['result'] = pd_df['a'] / pd_df['b']
        ds_df['result'] = ds_df['a'] / ds_df['b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_nan_in_comparison(self):
        """Test comparison with NaN (should be False)"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, 4.0]})

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnOperationsAfterFilter:
    """Test column operations on filtered DataFrames"""

    def test_add_column_after_filter(self):
        """Test adding new column after filtering"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        pd_filtered['c'] = pd_filtered['a'] + pd_filtered['b']

        ds_filtered = ds_df[ds_df['a'] > 2]
        ds_filtered['c'] = ds_filtered['a'] + ds_filtered['b']

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_modify_column_after_filter(self):
        """Test modifying existing column after filtering"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        pd_filtered['a'] = pd_filtered['a'] * 10

        ds_filtered = ds_df[ds_df['a'] > 2]
        ds_filtered['a'] = ds_filtered['a'] * 10

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_drop_column_after_filter(self):
        """Test dropping column after filtering"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})

        pd_result = pd_df[pd_df['a'] > 2].drop(columns=['c'])
        ds_result = ds_df[ds_df['a'] > 2].drop(columns=['c'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_filter_then_agg(self):
        """Test chaining filter with aggregation"""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A'], 'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'B', 'A'], 'value': [10, 20, 30, 40, 50]})

        pd_result = pd_df[pd_df['value'] > 15]['value'].sum()
        ds_result = ds_df[ds_df['value'] > 15]['value'].sum()

        assert pd_result == ds_result


class TestChainedBooleanOperationsWithNull:
    """Test chained boolean operations with NULL values"""

    def test_and_with_null(self):
        """Test AND operation where one condition involves NULL"""
        pd_df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })
        ds_df = DataStore({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] > 15)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] > 15)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_with_null(self):
        """Test OR operation where one condition involves NULL"""
        pd_df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })
        ds_df = DataStore({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [10.0, 20.0, 30.0, 40.0]
        })

        pd_result = pd_df[(pd_df['a'] > 3) | (pd_df['b'] > 35)]
        ds_result = ds_df[(ds_df['a'] > 3) | (ds_df['b'] > 35)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_not_on_condition_with_null(self):
        """Test NOT operation on condition involving NULL"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, 4.0]})

        pd_result = pd_df[~(pd_df['a'] > 2)]
        ds_result = ds_df[~(ds_df['a'] > 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialValueHandling:
    """Test handling of special values (inf, -inf, very large numbers)"""

    def test_filter_with_infinity(self):
        """Test filtering columns containing infinity"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 0.0, 5.0]})
        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, 0.0, 5.0]})

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_with_infinity(self):
        """Test sum of column containing infinity"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, 3.0]})
        ds_df = DataStore({'a': [1.0, np.inf, 3.0]})

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        assert pd_result == ds_result or (np.isinf(pd_result) and np.isinf(ds_result))

    def test_mean_with_infinity(self):
        """Test mean of column containing infinity"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, 3.0]})
        ds_df = DataStore({'a': [1.0, np.inf, 3.0]})

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()

        # Both should be inf
        assert np.isinf(pd_result) == np.isinf(ds_result)

    def test_arithmetic_with_very_large_numbers(self):
        """Test arithmetic with very large numbers"""
        large_val = 1e308
        pd_df = pd.DataFrame({'a': [large_val, large_val / 2, 1.0]})
        ds_df = DataStore({'a': [large_val, large_val / 2, 1.0]})

        pd_df['doubled'] = pd_df['a'] * 2
        ds_df['doubled'] = ds_df['a'] * 2

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestSelectAndProjectionEdgeCases:
    """Test column selection edge cases"""

    def test_select_single_column_returns_series(self):
        """Test that selecting single column returns Series-like"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df['a']
        ds_result = ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_select_columns_preserves_order(self):
        """Test that selecting columns preserves specified order"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6], 'd': [7, 8]})

        pd_result = pd_df[['d', 'b', 'a']]
        ds_result = ds_df[['d', 'b', 'a']]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert list(ds_result.columns) == ['d', 'b', 'a']

    def test_select_duplicate_column_names(self):
        """Test selecting when DataFrame has duplicate column names"""
        pd_df = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'a'])
        ds_df = DataStore(pd_df)

        # Both should return DataFrame with multiple columns
        pd_result = pd_df['a']
        ds_result = ds_df['a']

        # When there are duplicate columns, both pandas and datastore return DataFrame
        if isinstance(pd_result, pd.DataFrame):
            assert_datastore_equals_pandas(ds_result, pd_result)


class TestRenameOperations:
    """Test column rename operations"""

    def test_rename_single_column(self):
        """Test renaming a single column"""
        pd_df = pd.DataFrame({'old_name': [1, 2, 3]})
        ds_df = DataStore({'old_name': [1, 2, 3]})

        pd_result = pd_df.rename(columns={'old_name': 'new_name'})
        ds_result = ds_df.rename(columns={'old_name': 'new_name'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_multiple_columns(self):
        """Test renaming multiple columns at once"""
        pd_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ds_df = DataStore({'a': [1], 'b': [2], 'c': [3]})

        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y', 'c': 'z'})
        ds_result = ds_df.rename(columns={'a': 'x', 'b': 'y', 'c': 'z'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_nonexistent_column(self):
        """Test renaming column that doesn't exist (should be no-op in pandas)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.rename(columns={'nonexistent': 'new_name'})
        ds_result = ds_df.rename(columns={'nonexistent': 'new_name'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_to_existing_column_name(self):
        """Test renaming column to name that already exists"""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df.rename(columns={'a': 'b'})
        ds_result = ds_df.rename(columns={'a': 'b'})

        # This creates duplicate column names
        assert list(pd_result.columns) == list(ds_result.columns)


class TestAsTypeConversion:
    """Test astype conversion operations"""

    def test_astype_int_to_float(self):
        """Test converting int column to float"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df['a'].astype(float)
        ds_result = ds_df['a'].astype(float)

        assert_series_equal(ds_result, pd_result)

    def test_astype_float_to_int(self):
        """Test converting float column to int (truncates)"""
        pd_df = pd.DataFrame({'a': [1.7, 2.3, 3.9]})
        ds_df = DataStore({'a': [1.7, 2.3, 3.9]})

        pd_result = pd_df['a'].astype(int)
        ds_result = ds_df['a'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_int_to_str(self):
        """Test converting int column to string"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df['a'].astype(str)
        ds_result = ds_df['a'].astype(str)

        assert_series_equal(ds_result, pd_result)

    def test_astype_str_to_int(self):
        """Test converting string column to int"""
        pd_df = pd.DataFrame({'a': ['1', '2', '3']})
        ds_df = DataStore({'a': ['1', '2', '3']})

        pd_result = pd_df['a'].astype(int)
        ds_result = ds_df['a'].astype(int)

        assert_series_equal(ds_result, pd_result)


class TestSortOperations:
    """Test sort operations"""

    def test_sort_values_single_column(self):
        """Test sorting by single column"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5, 9, 2, 6]})

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_descending(self):
        """Test sorting descending"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.sort_values('a', ascending=False)
        ds_result = ds_df.sort_values('a', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_multiple_columns(self):
        """Test sorting by multiple columns"""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': [4, 3, 2, 1]
        })
        ds_df = DataStore({
            'a': [1, 1, 2, 2],
            'b': [4, 3, 2, 1]
        })

        pd_result = pd_df.sort_values(['a', 'b'])
        ds_result = ds_df.sort_values(['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_with_na(self):
        """Test sorting with NA values"""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 1.0, np.nan, 2.0]})
        ds_df = DataStore({'a': [3.0, np.nan, 1.0, np.nan, 2.0]})

        pd_result = pd_df.sort_values('a', na_position='last')
        ds_result = ds_df.sort_values('a', na_position='last')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCounts:
    """Test value_counts operations"""

    def test_value_counts_basic(self):
        """Test basic value_counts"""
        pd_df = pd.DataFrame({'a': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']})
        ds_df = DataStore({'a': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']})

        pd_result = pd_df['a'].value_counts()
        ds_result = ds_df['a'].value_counts()

        # value_counts order may differ for ties
        assert_series_equal(ds_result, pd_result, check_index=False)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize=True"""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'x', 'x', 'y']})
        ds_df = DataStore({'a': ['x', 'y', 'x', 'x', 'y']})

        pd_result = pd_df['a'].value_counts(normalize=True)
        ds_result = ds_df['a'].value_counts(normalize=True)

        assert_series_equal(ds_result, pd_result, check_index=False)

    def test_value_counts_with_na(self):
        """Test value_counts with NA values"""
        pd_df = pd.DataFrame({'a': ['x', None, 'x', None, 'y']})
        ds_df = DataStore({'a': ['x', None, 'x', None, 'y']})

        pd_result = pd_df['a'].value_counts(dropna=False)
        ds_result = ds_df['a'].value_counts(dropna=False)

        assert_series_equal(ds_result, pd_result, check_index=False)
