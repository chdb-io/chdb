"""
Exploratory Test Batch 48: Type Coercion Chains, Empty DataFrame Edge Cases,
Complex Operation Chains, and Boolean NULL Operations

This batch focuses on:
1. Type coercion chains - multiple sequential type conversions
2. Empty and single-row DataFrame operations
3. Complex lazy operation chains (filter -> assign -> groupby -> filter)
4. Boolean operations with NULL/NaN values
5. Aggregation edge cases (min_count, skipna variations)

Following Mirror Code Pattern - all tests compare DataStore behavior with pandas.
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_datastore_equals_pandas_chdb_compat,
)
from tests.xfail_markers import (
    chdb_category_type,
    chdb_timedelta_type,
)


# =============================================================================
# Type Coercion Chains
# =============================================================================


class TestTypeCoercionChains:
    """Test multiple sequential type conversions."""

    def test_int_to_float_to_str_chain(self):
        """Test int -> float -> str type chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.assign(
            b=lambda x: x['a'].astype('float64'),
            c=lambda x: x['a'].astype('float64').astype('str')
        )
        ds_result = ds_df.assign(
            b=lambda x: x['a'].astype('float64'),
            c=lambda x: x['a'].astype('float64').astype('str')
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_to_int_to_float_chain(self):
        """Test str -> int -> float type chain."""
        pd_df = pd.DataFrame({'a': ['1', '2', '3']})
        ds_df = DataStore({'a': ['1', '2', '3']})

        pd_result = pd_df.assign(
            b=lambda x: x['a'].astype('int64'),
            c=lambda x: x['a'].astype('int64').astype('float64')
        )
        ds_result = ds_df.assign(
            b=lambda x: x['a'].astype('int64'),
            c=lambda x: x['a'].astype('int64').astype('float64')
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_float_to_int_truncation(self):
        """Test float -> int truncation behavior."""
        pd_df = pd.DataFrame({'a': [1.1, 2.9, 3.5]})
        ds_df = DataStore({'a': [1.1, 2.9, 3.5]})

        pd_result = pd_df.assign(b=lambda x: x['a'].astype('int64'))
        ds_result = ds_df.assign(b=lambda x: x['a'].astype('int64'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_to_int_conversion(self):
        """Test bool -> int conversion."""
        pd_df = pd.DataFrame({'a': [True, False, True]})
        ds_df = DataStore({'a': [True, False, True]})

        pd_result = pd_df.assign(b=lambda x: x['a'].astype('int64'))
        ds_result = ds_df.assign(b=lambda x: x['a'].astype('int64'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_int_to_bool_conversion(self):
        """Test int -> bool conversion."""
        pd_df = pd.DataFrame({'a': [0, 1, 2, -1]})
        ds_df = DataStore({'a': [0, 1, 2, -1]})

        pd_result = pd_df.assign(b=lambda x: x['a'].astype('bool'))
        ds_result = ds_df.assign(b=lambda x: x['a'].astype('bool'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_type_coercion_with_none(self):
        """Test type coercion with None values."""
        pd_df = pd.DataFrame({'a': [1.0, None, 3.0]})
        ds_df = DataStore({'a': [1.0, None, 3.0]})

        # For int conversion with NaN, pandas uses Int64 nullable
        pd_result = pd_df.assign(b=lambda x: x['a'].astype('Int64'))
        ds_result = ds_df.assign(b=lambda x: x['a'].astype('Int64'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_with_type_promotion(self):
        """Test type promotion in arithmetic operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})

        pd_result = pd_df.assign(c=lambda x: x['a'] + x['b'])  # int + float -> float
        ds_result = ds_df.assign(c=lambda x: x['a'] + x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Empty DataFrame Operations
# =============================================================================


class TestEmptyDataFrameOperations:
    """Test operations on empty DataFrames."""

    def test_empty_df_column_selection(self):
        """Test column selection on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
        ds_df = DataStore({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})

        pd_result = pd_df[['a']]
        ds_result = ds_df[['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_assign(self):
        """Test assign on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64')})
        ds_df = DataStore({'a': pd.Series([], dtype='int64')})

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_filter(self):
        """Test filter on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64')})
        ds_df = DataStore({'a': pd.Series([], dtype='int64')})

        pd_result = pd_df[pd_df['a'] > 5]
        ds_result = ds_df[ds_df['a'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_groupby_agg(self):
        """Test groupby aggregation on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='str'), 'b': pd.Series([], dtype='int64')})
        ds_df = DataStore({'a': pd.Series([], dtype='str'), 'b': pd.Series([], dtype='int64')})

        pd_result = pd_df.groupby('a')['b'].sum().reset_index()
        ds_result = ds_df.groupby('a')['b'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_sort_values(self):
        """Test sort_values on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='str')})
        ds_df = DataStore({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='str')})

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


# =============================================================================
# Single Row DataFrame Operations
# =============================================================================


class TestSingleRowDataFrame:
    """Test operations on single-row DataFrames."""

    def test_single_row_aggregation(self):
        """Test aggregation on single row."""
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        ds_df = DataStore({'a': [5], 'b': [10]})

        pd_sum = pd_df['a'].sum()
        ds_sum = ds_df['a'].sum()

        # Compare scalar values
        assert float(ds_sum) == float(pd_sum)

    def test_single_row_groupby(self):
        """Test groupby on single row."""
        pd_df = pd.DataFrame({'group': ['A'], 'value': [100]})
        ds_df = DataStore({'group': ['A'], 'value': [100]})

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_match(self):
        """Test filter that matches the single row."""
        pd_df = pd.DataFrame({'a': [5]})
        ds_df = DataStore({'a': [5]})

        pd_result = pd_df[pd_df['a'] == 5]
        ds_result = ds_df[ds_df['a'] == 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self):
        """Test filter that doesn't match the single row (results in empty)."""
        pd_df = pd.DataFrame({'a': [5]})
        ds_df = DataStore({'a': [5]})

        pd_result = pd_df[pd_df['a'] == 10]
        ds_result = ds_df[ds_df['a'] == 10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_rolling(self):
        """Test rolling window on single row."""
        pd_df = pd.DataFrame({'a': [5]})
        ds_df = DataStore({'a': [5]})

        pd_result = pd_df.assign(rolling_mean=lambda x: x['a'].rolling(window=1).mean())
        ds_result = ds_df.assign(rolling_mean=lambda x: x['a'].rolling(window=1).mean())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Complex Operation Chains
# =============================================================================


class TestComplexOperationChains:
    """Test complex chains of multiple operations."""

    def test_filter_assign_filter_chain(self):
        """Test filter -> assign -> filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = (pd_df
            [pd_df['a'] > 1]
            .assign(c=lambda x: x['b'] * 2)
            .query('c > 50'))
        ds_result = (ds_df
            [ds_df['a'] > 1]
            .assign(c=lambda x: x['b'] * 2)
            .query('c > 50'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_groupby_agg_filter_chain(self):
        """Test assign -> groupby -> agg -> filter chain."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = (pd_df
            .assign(doubled=lambda x: x['value'] * 2)
            .groupby('group')['doubled'].sum()
            .reset_index())
        pd_result = pd_result[pd_result['doubled'] > 50]

        ds_result = (ds_df
            .assign(doubled=lambda x: x['value'] * 2)
            .groupby('group')['doubled'].sum()
            .reset_index())
        ds_result = ds_result[ds_result['doubled'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filter_combination(self):
        """Test combining multiple filters."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] > 1)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] > 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_filter_head_chain(self):
        """Test sort -> filter -> head chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5, 9, 2, 6]})

        pd_result = pd_df.sort_values('a')[pd_df.sort_values('a')['a'] > 2].head(3)
        ds_result = ds_df.sort_values('a')[ds_df.sort_values('a')['a'] > 2].head(3)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_assign_with_string_operations_chain(self):
        """Test assign with string operations chain."""
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
        ds_df = DataStore({'name': ['Alice', 'Bob', 'Charlie']})

        pd_result = (pd_df
            .assign(upper_name=lambda x: x['name'].str.upper())
            .assign(name_len=lambda x: x['name'].str.len()))
        ds_result = (ds_df
            .assign(upper_name=lambda x: x['name'].str.upper())
            .assign(name_len=lambda x: x['name'].str.len()))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Boolean Operations with NULL
# =============================================================================


class TestBooleanNullOperations:
    """Test boolean operations involving NULL/NaN values."""

    def test_and_with_null(self):
        """Test AND operation with NULL values."""
        pd_df = pd.DataFrame({
            'a': [True, True, False, False, None],
            'b': [True, False, True, False, True]
        })
        ds_df = DataStore({
            'a': [True, True, False, False, None],
            'b': [True, False, True, False, True]
        })

        pd_result = pd_df.assign(c=lambda x: x['a'] & x['b'])
        ds_result = ds_df.assign(c=lambda x: x['a'] & x['b'])

        # Values should match (dtype may differ due to nullable bool)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_with_null(self):
        """Test OR operation with NULL values."""
        pd_df = pd.DataFrame({
            'a': [True, True, False, False, None],
            'b': [True, False, True, False, True]
        })
        ds_df = DataStore({
            'a': [True, True, False, False, None],
            'b': [True, False, True, False, True]
        })

        pd_result = pd_df.assign(c=lambda x: x['a'] | x['b'])
        ds_result = ds_df.assign(c=lambda x: x['a'] | x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_not_with_null(self):
        """Test NOT operation with NULL values using nullable boolean type."""
        # Use nullable boolean dtype - regular object dtype with None fails in pandas
        pd_df = pd.DataFrame({'a': pd.array([True, False, None], dtype='boolean')})
        ds_df = DataStore({'a': pd.array([True, False, None], dtype='boolean')})

        pd_result = pd_df.assign(b=lambda x: ~x['a'])
        ds_result = ds_df.assign(b=lambda x: ~x['a'])

        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_comparison_with_null(self):
        """Test comparison operations with NULL values."""
        pd_df = pd.DataFrame({'a': [1, 2, None, 4, 5]})
        ds_df = DataStore({'a': [1, 2, None, 4, 5]})

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isnull_filter(self):
        """Test filtering by isnull()."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore({'a': [1, None, 3, None, 5]})

        pd_result = pd_df[pd_df['a'].isnull()]
        ds_result = ds_df[ds_df['a'].isnull()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notnull_filter(self):
        """Test filtering by notnull()."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore({'a': [1, None, 3, None, 5]})

        pd_result = pd_df[pd_df['a'].notnull()]
        ds_result = ds_df[ds_df['a'].notnull()]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Aggregation Edge Cases
# =============================================================================


class TestAggregationEdgeCases:
    """Test aggregation edge cases."""

    def test_sum_all_nan(self):
        """Test sum of all-NaN column."""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        ds_df = DataStore({'a': [np.nan, np.nan, np.nan]})

        pd_sum = pd_df['a'].sum()
        ds_sum = ds_df['a'].sum()

        # pandas sum of all-NaN returns 0.0, DataStore should match
        assert float(ds_sum) == float(pd_sum)

    def test_mean_all_nan(self):
        """Test mean of all-NaN column."""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        ds_df = DataStore({'a': [np.nan, np.nan, np.nan]})

        pd_mean = pd_df['a'].mean()
        ds_mean = ds_df['a'].mean()

        # pandas mean of all-NaN returns NaN, DataStore should match
        assert np.isnan(pd_mean)
        assert np.isnan(float(ds_mean))

    def test_min_max_all_nan(self):
        """Test min/max of all-NaN column."""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        ds_df = DataStore({'a': [np.nan, np.nan, np.nan]})

        pd_min = pd_df['a'].min()
        ds_min = ds_df['a'].min()

        pd_max = pd_df['a'].max()
        ds_max = ds_df['a'].max()

        # pandas min/max of all-NaN returns NaN
        assert np.isnan(pd_min)
        assert np.isnan(float(ds_min))
        assert np.isnan(pd_max)
        assert np.isnan(float(ds_max))

    def test_groupby_sum_with_nan_group(self):
        """Test groupby sum with NaN in group column."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', None, 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', None, 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        # Default dropna=True should exclude NaN groups
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_sum_with_nan_group_dropna_false(self):
        """Test groupby sum with NaN in group column, dropna=False."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', None, 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', None, 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        # dropna=False should include NaN as a group
        pd_result = pd_df.groupby('group', dropna=False)['value'].sum().reset_index()
        ds_result = ds_df.groupby('group', dropna=False)['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_count_with_null_values(self):
        """Test count (non-null count) with NULL values."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore({'a': [1, None, 3, None, 5]})

        pd_count = pd_df['a'].count()
        ds_count = ds_df['a'].count()

        assert int(ds_count) == int(pd_count)

    def test_nunique_with_null(self):
        """Test nunique with NULL values."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, None, 3]})
        ds_df = DataStore({'a': [1, 2, 2, None, 3]})

        pd_nunique = pd_df['a'].nunique()
        ds_nunique = ds_df['a'].nunique()

        # Default dropna=True in nunique
        assert int(ds_nunique) == int(pd_nunique)


# =============================================================================
# Numeric Edge Cases
# =============================================================================


class TestNumericEdgeCases:
    """Test numeric edge cases."""

    def test_large_integers(self):
        """Test operations with large integers."""
        large_int = 10**15
        pd_df = pd.DataFrame({'a': [large_int, large_int + 1, large_int + 2]})
        ds_df = DataStore({'a': [large_int, large_int + 1, large_int + 2]})

        pd_result = pd_df.assign(b=lambda x: x['a'] + 1)
        ds_result = ds_df.assign(b=lambda x: x['a'] + 1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negative_numbers(self):
        """Test operations with negative numbers."""
        pd_df = pd.DataFrame({'a': [-5, -3, -1, 0, 1, 3, 5]})
        ds_df = DataStore({'a': [-5, -3, -1, 0, 1, 3, 5]})

        pd_result = pd_df.assign(
            b=lambda x: x['a'] * -1,
            c=lambda x: x['a'] ** 2
        )
        ds_result = ds_df.assign(
            b=lambda x: x['a'] * -1,
            c=lambda x: x['a'] ** 2
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_division_by_zero_handling(self):
        """Test division by zero handling."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 0, 1]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [1, 0, 1]})

        # Division by zero should result in inf
        pd_result = pd_df.assign(c=lambda x: x['a'] / x['b'])
        ds_result = ds_df.assign(c=lambda x: x['a'] / x['b'])

        # Compare - inf values should match
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_float_precision(self):
        """Test float precision in calculations."""
        pd_df = pd.DataFrame({'a': [0.1, 0.2, 0.3]})
        ds_df = DataStore({'a': [0.1, 0.2, 0.3]})

        pd_result = pd_df.assign(b=lambda x: x['a'] * 3)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 3)

        # Use tolerance for float comparison
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-10, atol=1e-10)

    def test_modulo_operation(self):
        """Test modulo operation."""
        pd_df = pd.DataFrame({'a': [10, 11, 12, 13, 14]})
        ds_df = DataStore({'a': [10, 11, 12, 13, 14]})

        pd_result = pd_df.assign(b=lambda x: x['a'] % 3)
        ds_result = ds_df.assign(b=lambda x: x['a'] % 3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floor_division(self):
        """Test floor division operation."""
        pd_df = pd.DataFrame({'a': [10, 11, 12, 13, 14]})
        ds_df = DataStore({'a': [10, 11, 12, 13, 14]})

        pd_result = pd_df.assign(b=lambda x: x['a'] // 3)
        ds_result = ds_df.assign(b=lambda x: x['a'] // 3)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# String Edge Cases
# =============================================================================


class TestStringEdgeCases:
    """Test string operation edge cases."""

    def test_empty_string_operations(self):
        """Test operations on empty strings."""
        pd_df = pd.DataFrame({'a': ['', 'hello', '', 'world']})
        ds_df = DataStore({'a': ['', 'hello', '', 'world']})

        pd_result = pd_df.assign(
            len_a=lambda x: x['a'].str.len(),
            upper_a=lambda x: x['a'].str.upper()
        )
        ds_result = ds_df.assign(
            len_a=lambda x: x['a'].str.len(),
            upper_a=lambda x: x['a'].str.upper()
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_with_spaces(self):
        """Test string operations with leading/trailing spaces."""
        pd_df = pd.DataFrame({'a': ['  hello  ', 'world', '  test']})
        ds_df = DataStore({'a': ['  hello  ', 'world', '  test']})

        pd_result = pd_df.assign(stripped=lambda x: x['a'].str.strip())
        ds_result = ds_df.assign(stripped=lambda x: x['a'].str.strip())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_contains_special_chars(self):
        """Test string contains with special regex characters."""
        pd_df = pd.DataFrame({'a': ['hello.world', 'test*pattern', 'normal']})
        ds_df = DataStore({'a': ['hello.world', 'test*pattern', 'normal']})

        # Use regex=False to treat pattern as literal
        pd_result = pd_df[pd_df['a'].str.contains('.', regex=False)]
        ds_result = ds_df[ds_df['a'].str.contains('.', regex=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_null_handling(self):
        """Test string operations with NULL values."""
        pd_df = pd.DataFrame({'a': ['hello', None, 'world']})
        ds_df = DataStore({'a': ['hello', None, 'world']})

        pd_result = pd_df.assign(upper_a=lambda x: x['a'].str.upper())
        ds_result = ds_df.assign(upper_a=lambda x: x['a'].str.upper())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_concat_chain(self):
        """Test string concatenation chain."""
        pd_df = pd.DataFrame({
            'first': ['John', 'Jane'],
            'last': ['Doe', 'Smith']
        })
        ds_df = DataStore({
            'first': ['John', 'Jane'],
            'last': ['Doe', 'Smith']
        })

        pd_result = pd_df.assign(
            full_name=lambda x: x['first'] + ' ' + x['last']
        )
        ds_result = ds_df.assign(
            full_name=lambda x: x['first'] + ' ' + x['last']
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Index Preservation Tests
# =============================================================================


class TestIndexPreservation:
    """Test index preservation through various operations."""

    def test_filter_preserves_index(self):
        """Test that filter preserves index values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_preserves_index(self):
        """Test that assign preserves index values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore({'a': [1, 2, 3]}, index=['x', 'y', 'z'])

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_preserves_index(self):
        """Test that sort_values preserves index values."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]}, index=['c', 'a', 'b'])
        ds_df = DataStore({'a': [3, 1, 2]}, index=['c', 'a', 'b'])

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_chain_preserves_index(self):
        """Test that operation chain preserves index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])

        pd_result = (pd_df
            .assign(b=lambda x: x['a'] * 2)
            [pd_df['a'] > 1]
            .sort_values('a'))
        ds_result = (ds_df
            .assign(b=lambda x: x['a'] * 2)
            [ds_df['a'] > 1]
            .sort_values('a'))

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


# =============================================================================
# Duplicate Value Handling
# =============================================================================


class TestDuplicateValueHandling:
    """Test handling of duplicate values."""

    def test_duplicates_in_filter(self):
        """Test filter with duplicate values."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 3, 3]})
        ds_df = DataStore({'a': [1, 1, 2, 2, 3, 3]})

        pd_result = pd_df[pd_df['a'] == 2]
        ds_result = ds_df[ds_df['a'] == 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicates_in_groupby(self):
        """Test groupby with duplicate group values."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_basic(self):
        """Test drop_duplicates basic operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore({'a': [1, 2, 2, 3, 3, 3]})

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_drop_duplicates_subset(self):
        """Test drop_duplicates with subset."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': [1, 2, 1, 2]
        })
        ds_df = DataStore({
            'a': [1, 1, 2, 2],
            'b': [1, 2, 1, 2]
        })

        pd_result = pd_df.drop_duplicates(subset=['a'])
        ds_result = ds_df.drop_duplicates(subset=['a'])

        # Only check that we have the right number of unique 'a' values
        assert len(ds_result) == len(pd_result)

    def test_duplicated_detection(self):
        """Test duplicated() detection."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore({'a': [1, 2, 2, 3, 3, 3]})

        pd_result = pd_df.assign(is_dup=lambda x: x.duplicated())
        ds_result = ds_df.assign(is_dup=lambda x: x.duplicated())

        assert_datastore_equals_pandas(ds_result, pd_result)
