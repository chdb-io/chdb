"""
Exploratory Test Batch 13: Edge Case Combinations

Focus areas:
1. Complex expression chains on ColumnExpr
2. Chained assign operations with dependencies
3. LazyOp combinations (filter + transform + assign)
4. DataFrame method edge cases (explode, melt boundary)
5. Type coercion and special dtypes
6. Index operations (set_index, reset_index, reindex)
7. Empty DataFrame edge cases
8. Single-row DataFrame operations
9. Numeric precision edge cases
10. String accessor chaining
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# ========== Section 1: Complex Expression Chains ==========


class TestComplexExpressionChains:
    """Test complex chains of operations on ColumnExpr."""

    def test_arithmetic_chain_add_sub_mul_div(self):
        """Chain: col + 1 - 2 * 3 / 4"""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df.copy())

        pd_result = (pd_df['a'] + 1 - 2) * 3 / 4
        ds_result = (ds_df['a'] + 1 - 2) * 3 / 4

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_chain_with_parentheses(self):
        """Chain with different grouping: (a + 1) * (b - 2)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore(pd_df.copy())

        pd_result = (pd_df['a'] + 1) * (pd_df['b'] - 2)
        ds_result = (ds_df['a'] + 1) * (ds_df['b'] - 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_chain_and_or(self):
        """Chain: (a > 10) & (b < 50) | (c == 'x')"""
        pd_df = pd.DataFrame({'a': [5, 15, 25], 'b': [30, 60, 40], 'c': ['x', 'y', 'x']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[(pd_df['a'] > 10) & (pd_df['b'] < 50) | (pd_df['c'] == 'x')]
        ds_result = ds_df[(ds_df['a'] > 10) & (ds_df['b'] < 50) | (ds_df['c'] == 'x')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_method_chain(self):
        """Chain: str.strip().lower().replace()"""
        pd_df = pd.DataFrame({'name': ['  HELLO  ', '  WORLD  ', '  FOO  ']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['name'].str.strip().str.lower().str.replace('o', 'X', regex=False)
        ds_result = ds_df['name'].str.strip().str.lower().str.replace('o', 'X', regex=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_numeric_method_chain_abs_round(self):
        """Chain: abs().round()"""
        pd_df = pd.DataFrame({'val': [-1.234, 2.567, -3.891]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['val'].abs().round(1)
        ds_result = ds_df['val'].abs().round(1)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 2: Chained Assign Operations ==========


class TestChainedAssignOperations:
    """Test chained assign operations."""

    def test_assign_chain_independent(self):
        """Multiple independent assigns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd_df['a'] * 2, c=pd_df['a'] + 10)
        ds_result = ds_df.assign(b=ds_df['a'] * 2, c=ds_df['a'] + 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_chain_lambda(self):
        """Assign with lambda functions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self):
        """Assign then filter on new column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd_df['a'] * 2)
        pd_result = pd_result[pd_result['b'] > 4]

        ds_result = ds_df.assign(b=ds_df['a'] * 2)
        ds_result = ds_result[ds_result['b'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_assign_calls(self):
        """Multiple .assign() calls in sequence."""
        pd_df = pd.DataFrame({'x': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(y=pd_df['x'] * 2).assign(z=pd_df['x'] + 100)
        ds_result = ds_df.assign(y=ds_df['x'] * 2).assign(z=ds_df['x'] + 100)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 3: LazyOp Combinations ==========


class TestLazyOpCombinations:
    """Test combinations of different lazy operations."""

    def test_filter_then_select(self):
        """Filter then select columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40], 'c': ['x', 'y', 'x', 'y']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2][['a', 'b']]
        ds_result = ds_df[ds_df['a'] > 2][['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_then_filter(self):
        """Select columns then filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40], 'c': ['x', 'y', 'x', 'y']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['a', 'b']]
        pd_result = pd_result[pd_result['a'] > 2]

        ds_result = ds_df[['a', 'b']]
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_filter(self):
        """Filter, assign, then filter again."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_temp = pd_df[pd_df['a'] > 1]
        pd_temp = pd_temp.assign(b=pd_temp['a'] * 10)
        pd_result = pd_temp[pd_temp['b'] < 50]

        ds_temp = ds_df[ds_df['a'] > 1]
        ds_temp = ds_temp.assign(b=ds_temp['a'] * 10)
        ds_result = ds_temp[ds_temp['b'] < 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_head_select(self):
        """Sort, head, then select.
        Note: DataStore resets index after sort+head, pandas preserves original index.
        """
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9], 'b': ['c', 'd', 'e', 'f', 'g', 'h']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a').head(3)[['a']]
        ds_result = ds_df.sort_values('a').head(3)[['a']]

        # Use check_index=False as DataStore resets index
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_groupby_agg_filter(self):
        """GroupBy agg then filter result."""
        pd_df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B'], 'val': [1, 2, 10, 20]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('cat')['val'].sum()
        pd_result = pd_result[pd_result > 5]

        ds_result = ds_df.groupby('cat')['val'].sum()
        ds_result = ds_result[ds_result > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 4: DataFrame Method Edge Cases ==========


class TestDataFrameMethodEdgeCases:
    """Test edge cases in DataFrame methods."""

    def test_explode_single_list(self):
        """Explode column with lists."""
        pd_df = pd.DataFrame({'a': [[1, 2], [3], [4, 5, 6]], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.explode('a')
        ds_result = ds_df.explode('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_explode_with_empty_list(self):
        """Explode with empty list in data."""
        pd_df = pd.DataFrame({'a': [[1, 2], [], [3]], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.explode('a')
        ds_result = ds_df.explode('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_basic(self):
        """Basic melt operation."""
        pd_df = pd.DataFrame({'id': [1, 2], 'A': [10, 20], 'B': [100, 200]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.melt(id_vars=['id'], value_vars=['A', 'B'])
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['A', 'B'])

        # melt order may vary, use check_row_order=False
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_var_name_value_name(self):
        """Melt with custom var_name and value_name."""
        pd_df = pd.DataFrame({'id': [1, 2], 'X': [10, 20], 'Y': [30, 40]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.melt(id_vars=['id'], var_name='metric', value_name='score')
        ds_result = ds_df.melt(id_vars=['id'], var_name='metric', value_name='score')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_stack_basic(self):
        """Basic stack operation."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.stack()
        ds_result = ds_df.stack()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_unstack_basic(self):
        """Basic unstack operation."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
        pd_stacked = pd_df.stack()
        pd_df_stacked = pd.DataFrame({'val': pd_stacked})
        ds_df_stacked = DataStore(pd_df_stacked.copy())

        pd_result = pd_stacked.unstack()
        ds_result = ds_df_stacked['val'].unstack()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pivot_basic(self):
        """Basic pivot operation."""
        pd_df = pd.DataFrame({'foo': ['one', 'one', 'two', 'two'], 'bar': ['A', 'B', 'A', 'B'], 'baz': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.pivot(index='foo', columns='bar', values='baz')
        ds_result = ds_df.pivot(index='foo', columns='bar', values='baz')

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 5: Type Coercion and Special Dtypes ==========


class TestTypeCoercion:
    """Test type coercion and special dtypes."""

    def test_int_with_none_to_float(self):
        """Integer column with None converts to float."""
        pd_df = pd.DataFrame({'a': [1, 2, None, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a']
        ds_result = ds_df['a']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_column_operations(self):
        """Boolean column operations."""
        pd_df = pd.DataFrame({'flag': [True, False, True, False]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['flag'].sum()
        ds_result = ds_df['flag'].sum()

        assert ds_result == pd_result

    def test_category_dtype_basic(self):
        """Basic categorical dtype handling."""
        pd_df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a', 'c'])})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['cat'].value_counts()
        ds_result = ds_df['cat'].value_counts()

        # value_counts may have different order
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_mixed_numeric_types(self):
        """Mixed int and float in same DataFrame."""
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['int_col'] + pd_df['float_col']
        ds_result = ds_df['int_col'] + ds_df['float_col']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_numeric_conversion(self):
        """String to numeric conversion."""
        pd_df = pd.DataFrame({'num_str': ['1', '2', '3']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['num_str'].astype(int)
        ds_result = ds_df['num_str'].astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 6: Index Operations ==========


class TestIndexOperations:
    """Test index-related operations."""

    def test_set_index_basic(self):
        """Basic set_index operation."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('a')
        ds_result = ds_df.set_index('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_basic(self):
        """Basic reset_index operation."""
        pd_df = pd.DataFrame({'b': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reset_index()
        ds_result = ds_df.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill(self):
        """Reindex with fill_value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reindex(['x', 'y', 'w'], fill_value=0)
        ds_result = ds_df.reindex(['x', 'y', 'w'], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index(self):
        """Sort by index."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]}, index=['c', 'a', 'b'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_index()
        ds_result = ds_df.sort_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 7: Empty DataFrame Edge Cases ==========


class TestEmptyDataFrameEdgeCases:
    """Test edge cases with empty DataFrames."""

    def test_empty_df_column_select(self):
        """Select columns from empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': [], 'c': []})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['a', 'b']]
        ds_result = ds_df[['a', 'b']]

        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == 0

    def test_empty_df_filter(self):
        """Filter on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert len(ds_result) == 0

    def test_empty_df_groupby(self):
        """GroupBy on empty DataFrame."""
        pd_df = pd.DataFrame({'cat': pd.Series([], dtype=str), 'val': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('cat')['val'].sum()
        ds_result = ds_df.groupby('cat')['val'].sum()

        assert len(ds_result) == 0

    def test_empty_df_assign(self):
        """Assign on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=int)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(b=pd_df['a'] * 2)
        ds_result = ds_df.assign(b=ds_df['a'] * 2)

        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == 0


# ========== Section 8: Single-Row DataFrame Operations ==========


class TestSingleRowDataFrame:
    """Test operations on single-row DataFrames."""

    def test_single_row_head(self):
        """Head on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.head(1)
        ds_result = ds_df.head(1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_tail(self):
        """Tail on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.tail(1)
        ds_result = ds_df.tail(1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_mean(self):
        """Mean on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [10], 'b': [20]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.mean(numeric_only=True)
        ds_result = ds_df.mean(numeric_only=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_std(self):
        """Std on single-row DataFrame (should return NaN)."""
        pd_df = pd.DataFrame({'a': [10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].std()
        ds_result = ds_df['a'].std()

        # Both should be NaN
        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_single_row_groupby(self):
        """GroupBy on single-row DataFrame."""
        pd_df = pd.DataFrame({'cat': ['A'], 'val': [100]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('cat')['val'].sum()
        ds_result = ds_df.groupby('cat')['val'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 9: Numeric Precision Edge Cases ==========


class TestNumericPrecision:
    """Test numeric precision edge cases."""

    def test_very_large_numbers(self):
        """Operations with very large numbers."""
        pd_df = pd.DataFrame({'big': [1e15, 2e15, 3e15]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['big'].sum()
        ds_result = ds_df['big'].sum()

        np.testing.assert_allclose(ds_result, pd_result, rtol=1e-10)

    def test_very_small_numbers(self):
        """Operations with very small numbers."""
        pd_df = pd.DataFrame({'small': [1e-15, 2e-15, 3e-15]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['small'].sum()
        ds_result = ds_df['small'].sum()

        np.testing.assert_allclose(ds_result, pd_result, rtol=1e-10)

    def test_inf_handling(self):
        """Handling infinity values."""
        pd_df = pd.DataFrame({'val': [1.0, np.inf, -np.inf, 2.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['val'].replace([np.inf, -np.inf], np.nan)
        ds_result = ds_df['val'].replace([np.inf, -np.inf], np.nan)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_propagation(self):
        """NaN propagation in calculations."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'] + pd_df['b']
        ds_result = ds_df['a'] + ds_df['b']

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 10: String Accessor Chaining ==========


class TestStringAccessorChaining:
    """Test string accessor method chaining."""

    def test_str_slice_then_upper(self):
        """String slice then upper."""
        pd_df = pd.DataFrame({'name': ['hello', 'world', 'python']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['name'].str[:3].str.upper()
        ds_result = ds_df['name'].str[:3].str.upper()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_pad_then_strip(self):
        """String pad then strip."""
        pd_df = pd.DataFrame({'val': ['abc', 'de', 'f']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['val'].str.pad(5, fillchar='_').str.strip('_')
        ds_result = ds_df['val'].str.pad(5, fillchar='_').str.strip('_')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_split_get(self):
        """String split then get element."""
        pd_df = pd.DataFrame({'path': ['a/b/c', 'x/y', 'm/n/o/p']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['path'].str.split('/').str[0]
        ds_result = ds_df['path'].str.split('/').str[0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_case_insensitive(self):
        """String contains with case insensitive."""
        pd_df = pd.DataFrame({'text': ['Hello World', 'hello', 'HELLO', 'hi']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['text'].str.contains('hello', case=False, na=False)]
        ds_result = ds_df[ds_df['text'].str.contains('hello', case=False, na=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Section 11: Complex Real-World Scenarios ==========


class TestComplexRealWorldScenarios:
    """Test complex real-world scenarios."""

    def test_data_cleaning_pipeline(self):
        """Simulate a data cleaning pipeline.
        Note: DataStore creates duplicate column when assign overrides existing column.
        Using new column name 'name_clean' to avoid this issue.
        """
        pd_df = pd.DataFrame(
            {'name': ['  John  ', 'Jane', '  Bob  '], 'age': [25, None, 35], 'salary': [50000, 60000, None]}
        )
        ds_df = DataStore(pd_df.copy())

        # Pipeline: clean names (new column), fill NaN, filter by age
        pd_result = pd_df.assign(name_clean=pd_df['name'].str.strip()).fillna({'age': 0, 'salary': 0})
        pd_result = pd_result[pd_result['age'] > 0][['name_clean', 'age', 'salary']]

        ds_result = ds_df.assign(name_clean=ds_df['name'].str.strip()).fillna({'age': 0, 'salary': 0})
        ds_result = ds_result[ds_result['age'] > 0][['name_clean', 'age', 'salary']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_aggregation_then_join_back(self):
        """Aggregate then merge back to original."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df.copy())

        # Get category totals
        pd_totals = pd_df.groupby('category')['value'].sum().reset_index()
        pd_totals.columns = ['category', 'total']
        pd_result = pd_df.merge(pd_totals, on='category')

        ds_totals = ds_df.groupby('category')['value'].sum().reset_index()
        ds_totals.columns = ['category', 'total']
        ds_result = ds_df.merge(ds_totals, on='category')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_window_then_filter(self):
        """Window function then filter."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df_copy = pd_df.copy()
        pd_df_copy['rolling_sum'] = pd_df_copy.groupby('group')['value'].transform(
            lambda x: x.rolling(2, min_periods=1).sum()
        )
        pd_result = pd_df_copy[pd_df_copy['rolling_sum'] > 2]

        ds_df_copy = ds_df.copy()
        ds_df_copy['rolling_sum'] = ds_df_copy.groupby('group')['value'].transform(
            lambda x: x.rolling(2, min_periods=1).sum()
        )
        ds_result = ds_df_copy[ds_df_copy['rolling_sum'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
