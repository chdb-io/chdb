"""
Exploratory Test Batch 32: Complex Scenarios and Edge Cases

Focus areas:
1. Complex filter combinations with NaN
2. Multi-column operations
3. Chained method calls with different data types
4. Interval and period operations
5. Memory and performance edge cases
6. Complex aggregation scenarios
"""

import pytest
from tests.xfail_markers import chdb_datetime_range_comparison, chdb_dt_month_type
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


# =============================================================================
# Test Class: Complex Filter with NaN
# =============================================================================
class TestComplexFilterNaN:
    """Test filter operations with NaN values in various combinations."""

    def test_filter_nan_with_or(self):
        """Test filter with OR condition involving NaN."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0], 'b': ['x', 'y', None, 'z', 'w']})
        ds_df = DataStore(pd_df)

        # Filter: a > 2 OR a is NaN
        pd_result = pd_df[(pd_df['a'] > 2) | (pd_df['a'].isna())]
        ds_result = ds_df[(ds_df['a'] > 2) | (ds_df['a'].isna())]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_nan_with_and(self):
        """Test filter with AND condition involving NaN."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        # Filter: NOT NaN AND b > 20
        pd_result = pd_df[pd_df['a'].notna() & (pd_df['b'] > 20)]
        ds_result = ds_df[ds_df['a'].notna() & (ds_df['b'] > 20)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_then_filter(self):
        """Test fillna followed by filter."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.fillna({'a': 0})
        pd_result = pd_result[pd_result['a'] > 0]

        ds_result = ds_df.fillna({'a': 0})
        ds_result = ds_result[ds_result['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset_then_filter(self):
        """Test dropna with subset followed by filter."""
        pd_df = pd.DataFrame(
            {'a': [1.0, np.nan, 3.0, 4.0, np.nan], 'b': [np.nan, 2.0, 3.0, np.nan, 5.0], 'c': [1, 2, 3, 4, 5]}
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna(subset=['a'])
        pd_result = pd_result[pd_result['c'] > 2]

        ds_result = ds_df.dropna(subset=['a'])
        ds_result = ds_result[ds_result['c'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: Multi-Column Operations
# =============================================================================
class TestMultiColumnOperations:
    """Test operations involving multiple columns."""

    def test_multi_column_arithmetic(self):
        """Test arithmetic operations across multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(sum_abc=pd_df['a'] + pd_df['b'] + pd_df['c'])
        ds_result = ds_df.assign(sum_abc=ds_df['a'] + ds_df['b'] + ds_df['c'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_column_comparison(self):
        """Test comparison operations across multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 4, 2], 'b': [2, 3, 3, 4, 1], 'c': [3, 4, 3, 4, 3]})
        ds_df = DataStore(pd_df)

        # Filter where a < b and b <= c
        pd_result = pd_df[(pd_df['a'] < pd_df['b']) & (pd_df['b'] <= pd_df['c'])]
        ds_result = ds_df[(ds_df['a'] < ds_df['b']) & (ds_df['b'] <= ds_df['c'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_computed_columns(self):
        """Test selecting computed columns with different operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(
            product=pd_df['a'] * pd_df['b'], ratio=pd_df['b'] / pd_df['a'], diff=pd_df['b'] - pd_df['a']
        )[['product', 'ratio', 'diff']]

        ds_result = ds_df.assign(
            product=ds_df['a'] * ds_df['b'], ratio=ds_df['b'] / ds_df['a'], diff=ds_df['b'] - ds_df['a']
        )[['product', 'ratio', 'diff']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_multi_column_agg(self):
        """Test groupby with aggregation on multiple columns."""
        pd_df = pd.DataFrame(
            {
                'group': ['A', 'A', 'B', 'B', 'C'],
                'a': [1, 2, 3, 4, 5],
                'b': [10, 20, 30, 40, 50],
                'c': [100, 200, 300, 400, 500],
            }
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group').agg({'a': 'sum', 'b': 'mean', 'c': 'max'}).reset_index()
        ds_result = ds_df.groupby('group').agg({'a': 'sum', 'b': 'mean', 'c': 'max'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Class: Chained Methods with Different Types
# =============================================================================
class TestChainedMethodsTypes:
    """Test chained method calls with different data types."""

    def test_chain_numeric_string_ops(self):
        """Test chaining numeric and string operations."""
        pd_df = pd.DataFrame({'num': [1, 2, 3, 4, 5], 'text': ['apple', 'banana', 'cherry', 'date', 'elderberry']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['num'] > 2]
        pd_result = pd_result[pd_result['text'].str.len() > 4]

        ds_result = ds_df[ds_df['num'] > 2]
        ds_result = ds_result[ds_result['text'].str.len() > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_filter_assign_filter(self):
        """Test filter -> assign -> filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 3]
        pd_result = pd_result.assign(c=pd_result['a'] * 2)
        pd_result = pd_result[pd_result['c'] < 16]

        ds_result = ds_df[ds_df['a'] > 3]
        ds_result = ds_result.assign(c=ds_result['a'] * 2)
        ds_result = ds_result[ds_result['c'] < 16]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_sort_head_filter(self):
        """Test sort -> head -> filter chain."""
        pd_df = pd.DataFrame(
            {'a': [5, 2, 8, 1, 9, 3, 7, 4, 6, 10], 'b': ['e', 'b', 'h', 'a', 'i', 'c', 'g', 'd', 'f', 'j']}
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', ascending=False).head(7)
        pd_result = pd_result[pd_result['a'] > 4]

        ds_result = ds_df.sort_values('a', ascending=False).head(7)
        ds_result = ds_result[ds_result['a'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_groupby_filter_sort(self):
        """Test groupby -> agg -> filter -> sort chain."""
        pd_df = pd.DataFrame(
            {'group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'], 'value': [10, 20, 30, 40, 5, 15, 25, 35]}
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_result[pd_result['value'] > 20]
        pd_result = pd_result.sort_values('value', ascending=False)

        ds_result = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_result[ds_result['value'] > 20]
        ds_result = ds_result.sort_values('value', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: Date/Time Edge Cases
# =============================================================================
class TestDateTimeEdgeCases:
    """Test datetime operations with edge cases."""

    def test_date_filter_boundary(self):
        """Test filtering on date boundaries."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        pd_df = pd.DataFrame({'date': dates, 'value': range(10)})
        ds_df = DataStore(pd_df)

        boundary = pd.Timestamp('2023-01-05')
        pd_result = pd_df[pd_df['date'] >= boundary]
        ds_result = ds_df[ds_df['date'] >= boundary]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_datetime_range_comparison
    def test_date_between_filter(self):
        """Test filtering dates between two values."""
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        pd_df = pd.DataFrame({'date': dates, 'value': range(31)})
        ds_df = DataStore(pd_df)

        start = pd.Timestamp('2023-01-10')
        end = pd.Timestamp('2023-01-20')
        pd_result = pd_df[(pd_df['date'] >= start) & (pd_df['date'] <= end)]
        ds_result = ds_df[(ds_df['date'] >= start) & (ds_df['date'] <= end)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_datetime_groupby_date_part(self):
        """Test groupby with datetime part extraction."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        pd_df = pd.DataFrame({'date': dates, 'value': np.random.randint(1, 100, 100)})
        # NOTE: Must use .copy() to avoid shared DataFrame modification
        ds_df = DataStore(pd_df.copy())

        pd_df['month'] = pd_df['date'].dt.month
        pd_result = pd_df.groupby('month')['value'].sum().reset_index()

        ds_df = ds_df.assign(month=ds_df['date'].dt.month)
        ds_result = ds_df.groupby('month')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Class: Complex Aggregation Scenarios
# =============================================================================
class TestComplexAggregation:
    """Test complex aggregation scenarios."""

    def test_agg_with_custom_names(self):
        """Test aggregation with custom column names using named aggregation."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'C'], 'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = (
            pd_df.groupby('group')
            .agg(total=('value', 'sum'), average=('value', 'mean'), count=('value', 'count'))
            .reset_index()
        )

        ds_result = (
            ds_df.groupby('group')
            .agg(total=('value', 'sum'), average=('value', 'mean'), count=('value', 'count'))
            .reset_index()
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_agg_multiple_columns_same_func(self):
        """Test aggregation with same function on multiple columns."""
        pd_df = pd.DataFrame(
            {'group': ['A', 'A', 'B', 'B'], 'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40], 'c': [100, 200, 300, 400]}
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')[['a', 'b', 'c']].sum().reset_index()
        ds_result = ds_df.groupby('group')[['a', 'b', 'c']].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_agg_with_filter_after(self):
        """Test aggregation followed by filter on aggregated result."""
        pd_df = pd.DataFrame(
            {'group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'], 'value': [10, 20, 5, 10, 30, 40, 15, 25]}
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_result[pd_result['value'] > 25]

        ds_result = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_result[ds_result['value'] > 25]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_window_function_with_groupby(self):
        """Test window function with groupby."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'A', 'B', 'B', 'B'], 'value': [10, 20, 30, 15, 25, 35]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_result.groupby('group')['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result = ds_result.assign(cumsum=ds_df.groupby('group')['value'].cumsum())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: Edge Cases with Special Values
# =============================================================================
class TestSpecialValuesEdgeCases:
    """Test edge cases with special values like inf, -inf, very large/small numbers."""

    def test_inf_in_arithmetic(self):
        """Test arithmetic operations with infinity."""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 0.0, 5.0], 'b': [2.0, 3.0, 4.0, np.inf, -np.inf]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(sum_ab=pd_df['a'] + pd_df['b'])
        ds_result = ds_df.assign(sum_ab=ds_df['a'] + ds_df['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_inf_filter(self):
        """Test filtering infinite values."""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 0.0, 5.0, np.inf], 'b': [10, 20, 30, 40, 50, 60]})
        ds_df = DataStore(pd_df)

        # Filter out infinite values
        pd_result = pd_df[~np.isinf(pd_df['a'])]
        ds_result = ds_df[~np.isinf(ds_df['a'].to_pandas())]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_large_numbers(self):
        """Test operations with very large numbers."""
        pd_df = pd.DataFrame({'a': [1e100, 1e200, 1e300, 1e50, 1e10], 'b': [1e10, 1e20, 1e30, 1e40, 1e50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(product=pd_df['a'] * pd_df['b'])
        ds_result = ds_df.assign(product=ds_df['a'] * ds_df['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_small_numbers(self):
        """Test operations with very small numbers."""
        pd_df = pd.DataFrame({'a': [1e-100, 1e-200, 1e-300, 1e-50, 1e-10], 'b': [1e-10, 1e-20, 1e-30, 1e-40, 1e-50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(ratio=pd_df['a'] / pd_df['b'])
        ds_result = ds_df.assign(ratio=ds_df['a'] / ds_df['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: String Operations Edge Cases
# =============================================================================
class TestStringEdgeCases:
    """Test string operations with edge cases."""

    def test_string_with_special_chars(self):
        """Test string operations with special characters."""
        pd_df = pd.DataFrame({'text': ['hello\nworld', 'foo\tbar', 'a"b\'c', 'path/to/file', 'email@test.com']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(length=pd_df['text'].str.len())
        ds_result = ds_df.assign(length=ds_df['text'].str.len())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_empty_values(self):
        """Test string operations with empty strings."""
        pd_df = pd.DataFrame({'text': ['hello', '', 'world', '', 'test']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['text'].str.len() > 0]
        ds_result = ds_df[ds_df['text'].str.len() > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_case_insensitive_filter(self):
        """Test case-insensitive string filtering."""
        pd_df = pd.DataFrame({'text': ['Hello', 'WORLD', 'TeSt', 'foo', 'BAR']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['text'].str.lower().str.contains('o')]
        ds_result = ds_df[ds_df['text'].str.lower().str.contains('o')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_replace_chain(self):
        """Test chained string replace operations."""
        pd_df = pd.DataFrame({'text': ['a-b-c', 'd-e-f', 'g-h-i']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(modified=pd_df['text'].str.replace('-', '_').str.upper())
        ds_result = ds_df.assign(modified=ds_df['text'].str.replace('-', '_').str.upper())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: Index Operations
# =============================================================================
class TestIndexOperations:
    """Test various index operations."""

    def test_set_index_then_reset(self):
        """Test set_index followed by reset_index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v'], 'c': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a').reset_index()
        ds_result = ds_df.set_index('a').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill(self):
        """Test reindex with fill_value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]}, index=[0, 2, 4])
        ds_df = DataStore(pd_df)

        new_index = [0, 1, 2, 3, 4]
        pd_result = pd_df.reindex(new_index, fill_value=0)
        ds_result = ds_df.reindex(new_index, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_descending(self):
        """Test sort_index with descending order."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}, index=[3, 1, 4, 0, 2])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_index(ascending=False)
        ds_result = ds_df.sort_index(ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: Copy and Modification
# =============================================================================
class TestCopyModification:
    """Test copy and modification behavior."""

    def test_copy_independence(self):
        """Test that copies are independent."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        ds_copy = ds_df.copy()
        ds_filtered = ds_copy[ds_copy['a'] > 1]

        # Original should be unchanged
        assert len(ds_df) == 3
        assert len(ds_filtered) == 2

    def test_deep_copy(self):
        """Test deep copy behavior."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        ds_copy = ds_df.copy(deep=True)

        pd_result = pd_df.copy(deep=True)
        ds_result = ds_copy

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: Memory and Large Data
# =============================================================================
class TestMemoryLargeData:
    """Test memory operations and large data handling."""

    def test_memory_usage_basic(self):
        """Test memory_usage method."""
        pd_df = pd.DataFrame({'a': range(100), 'b': ['text'] * 100, 'c': np.random.random(100)})
        ds_df = DataStore(pd_df)

        pd_memory = pd_df.memory_usage()
        ds_memory = ds_df.memory_usage()

        # Memory usage should be comparable (not exact due to implementation)
        assert len(pd_memory) == len(ds_memory)

    def test_large_dataframe_filter(self):
        """Test filtering on larger DataFrame."""
        size = 10000
        pd_df = pd.DataFrame(
            {
                'a': np.random.randint(0, 100, size),
                'b': np.random.random(size),
                'c': ['cat_' + str(i % 10) for i in range(size)],
            }
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 50]
        ds_result = ds_df[ds_df['a'] > 50]

        assert len(ds_result) == len(pd_result)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_large_dataframe_groupby(self):
        """Test groupby on larger DataFrame."""
        size = 10000
        pd_df = pd.DataFrame(
            {'group': ['G' + str(i % 50) for i in range(size)], 'value': np.random.randint(0, 1000, size)}
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')['value'].sum().reset_index().sort_values('group').reset_index(drop=True)
        ds_result = ds_df.groupby('group')['value'].sum().reset_index().sort_values('group').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Class: DataFrame Construction Edge Cases
# =============================================================================
class TestConstructionEdgeCases:
    """Test DataFrame construction edge cases."""

    def test_construct_from_nested_dict(self):
        """Test construction from nested dictionary."""
        data = {'a': {'x': 1, 'y': 2, 'z': 3}, 'b': {'x': 10, 'y': 20, 'z': 30}}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_from_list_of_dicts(self):
        """Test construction from list of dictionaries."""
        data = [{'a': 1, 'b': 10}, {'a': 2, 'b': 20}, {'a': 3, 'b': 30}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_with_index(self):
        """Test construction with custom index."""
        data = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        index = ['x', 'y', 'z']
        pd_df = pd.DataFrame(data, index=index)
        ds_df = DataStore(data, index=index)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_with_dtype(self):
        """Test construction with explicit dtype."""
        data = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        pd_df = pd.DataFrame(data, dtype=float)
        ds_df = DataStore(data, dtype=float)

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Test Class: Method Return Types
# =============================================================================
class TestMethodReturnTypes:
    """Test that methods return correct types."""

    def test_head_returns_datastore(self):
        """Test that head returns DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        result = ds_df.head(3)
        assert isinstance(result, DataStore)
        assert len(result) == 3

    def test_filter_returns_datastore(self):
        """Test that filter returns DataStore."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        result = ds_df[ds_df['a'] > 2]
        assert isinstance(result, DataStore)
        assert len(result) == 3

    def test_groupby_agg_returns_datastore(self):
        """Test that groupby.agg returns DataStore."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B'], 'value': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        result = ds_df.groupby('group')['value'].sum().reset_index()
        assert isinstance(result, DataStore)

    def test_merge_returns_datastore(self):
        """Test that merge returns DataStore."""
        pd_df1 = pd.DataFrame({'key': [1, 2], 'a': [10, 20]})
        pd_df2 = pd.DataFrame({'key': [1, 2], 'b': [100, 200]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        result = ds_df1.merge(ds_df2, on='key')
        assert isinstance(result, DataStore)


# =============================================================================
# Test Class: Apply and Transform
# =============================================================================
class TestApplyTransform:
    """Test apply and transform operations."""

    def test_apply_lambda_to_column(self):
        """Test apply with lambda to single column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: x * 2)
        ds_result = ds_df.apply(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_axis_1(self):
        """Test apply along axis=1."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(sum, axis=1)
        ds_result = ds_df.apply(sum, axis=1)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_transform_with_function(self):
        """Test transform with function."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df.groupby('group')['value'].transform('mean')

        ds_result = get_series(ds_result)
        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Test Class: Concat and Merge Edge Cases
# =============================================================================
class TestConcatMergeEdgeCases:
    """Test concat and merge edge cases."""

    def test_concat_different_columns(self):
        """Test concat with different columns."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
        pd_df2 = pd.DataFrame({'a': [3, 4], 'c': [30, 40]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds_df1.concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_with_indicator(self):
        """Test merge with indicator column."""
        pd_df1 = pd.DataFrame({'key': [1, 2, 3], 'a': [10, 20, 30]})
        pd_df2 = pd.DataFrame({'key': [2, 3, 4], 'b': [200, 300, 400]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd_df1.merge(pd_df2, on='key', how='outer', indicator=True)
        ds_result = ds_df1.merge(ds_df2, on='key', how='outer', indicator=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_suffixes(self):
        """Test merge with custom suffixes."""
        pd_df1 = pd.DataFrame({'key': [1, 2], 'value': [10, 20]})
        pd_df2 = pd.DataFrame({'key': [1, 2], 'value': [100, 200]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd_df1.merge(pd_df2, on='key', suffixes=('_left', '_right'))
        ds_result = ds_df1.merge(ds_df2, on='key', suffixes=('_left', '_right'))

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
