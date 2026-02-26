"""
Exploratory Batch 81: Complex Operation Chains and SQL Pushdown Edge Cases

Focus areas:
1. Multi-step operation chains that should merge into single SQL
2. Filter -> GroupBy -> Agg -> Sort chains
3. Assign with computed columns and subsequent filters
4. Window functions with PARTITION BY
5. Edge cases with NULL handling in chains
6. Cumulative operations after filters
7. Multiple aggregations in single groupby
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestFilterGroupbyAggSortChain:
    """Test filter -> groupby -> agg -> sort chains"""

    def test_filter_groupby_sum_sort(self):
        """Filter -> GroupBy -> Sum -> Sort chain"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60, 15, 25],
            'flag': [True, False, True, True, False, True, True, False]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60, 15, 25],
            'flag': [True, False, True, True, False, True, True, False]
        })

        # Filter -> GroupBy -> Sum
        pd_result = (pd_df[pd_df['flag'] == True]
                     .groupby('category')['value']
                     .sum()
                     .sort_values())
        ds_result = (ds_df[ds_df['flag'] == True]
                     .groupby('category')['value']
                     .sum()
                     .sort_values())

        # Compare values
        assert list(pd_result.values) == list(ds_result.values)

    def test_multi_column_filter_groupby_multi_agg(self):
        """Multi-column filter -> groupby -> multiple aggregations"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value1': [10, 20, 30, 40, 50, 60],
            'value2': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value1': [10, 20, 30, 40, 50, 60],
            'value2': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        })

        pd_result = (pd_df[(pd_df['value1'] > 15) & (pd_df['value2'] < 6)]
                     .groupby('category')
                     .agg({'value1': 'sum', 'value2': 'mean'}))
        ds_result = (ds_df[(ds_df['value1'] > 15) & (ds_df['value2'] < 6)]
                     .groupby('category')
                     .agg({'value1': 'sum', 'value2': 'mean'}))

        # Compare values column by column
        for col in pd_result.columns:
            pd_vals = sorted(pd_result[col].values)
            ds_vals = sorted(ds_result[col].values)
            assert np.allclose(pd_vals, ds_vals)


class TestAssignFilterChain:
    """Test assign -> filter chains"""

    def test_assign_computed_column_then_filter(self):
        """Assign computed column -> filter on new column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = (pd_df
                     .assign(c=lambda x: x['a'] + x['b'])
                     [lambda x: x['c'] > 25])
        ds_result = (ds_df
                     .assign(c=lambda x: x['a'] + x['b'])
                     [lambda x: x['c'] > 25])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_assigns_then_filter(self):
        """Multiple assigns -> filter"""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [5, 4, 3, 2, 1]
        })
        ds_df = DataStore({
            'x': [1, 2, 3, 4, 5],
            'y': [5, 4, 3, 2, 1]
        })

        pd_result = (pd_df
                     .assign(sum_xy=lambda df: df['x'] + df['y'])
                     .assign(diff_xy=lambda df: df['x'] - df['y'])
                     .assign(product=lambda df: df['sum_xy'] * df['diff_xy'])
                     [lambda df: df['product'] > 0])
        ds_result = (ds_df
                     .assign(sum_xy=lambda df: df['x'] + df['y'])
                     .assign(diff_xy=lambda df: df['x'] - df['y'])
                     .assign(product=lambda df: df['sum_xy'] * df['diff_xy'])
                     [lambda df: df['product'] > 0])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWindowFunctionsWithPartition:
    """Test window functions with PARTITION BY"""

    def test_rank_with_partition_ascending(self):
        """Rank within partition (ascending)"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [30, 10, 20, 40, 20, 30]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [30, 10, 20, 40, 20, 30]
        })

        pd_result = pd_df.groupby('category')['value'].rank(method='first')
        ds_result = ds_df.groupby('category')['value'].rank(method='first')

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_rank_with_partition_descending(self):
        """Rank within partition (descending)"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [30, 10, 20, 40, 20, 30]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [30, 10, 20, 40, 20, 30]
        })

        pd_result = pd_df.groupby('category')['value'].rank(method='first', ascending=False)
        ds_result = ds_df.groupby('category')['value'].rank(method='first', ascending=False)

        assert_series_equal(ds_result, pd_result, check_names=False)


class TestNullHandlingInChains:
    """Test NULL/NaN handling in operation chains"""

    def test_filter_with_nulls_then_groupby(self):
        """Filter (including NULLs) -> GroupBy"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', None, 'A'],
            'value': [10.0, np.nan, 30.0, 40.0, 50.0, 60.0]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', None, 'A'],
            'value': [10.0, np.nan, 30.0, 40.0, 50.0, 60.0]
        })

        # Filter out nulls then groupby
        pd_result = pd_df[pd_df['category'].notna()].groupby('category')['value'].sum()
        ds_result = ds_df[ds_df['category'].notna()].groupby('category')['value'].sum()

        # Compare sums
        assert np.isclose(pd_result['A'], ds_result['A'])
        assert np.isclose(pd_result['B'], ds_result['B'])

    def test_fillna_then_filter(self):
        """FillNA -> Filter chain"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10.0, np.nan, np.nan, 40.0]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10.0, np.nan, np.nan, 40.0]
        })

        pd_result = pd_df.assign(value=pd_df['value'].fillna(0))[lambda x: x['value'] > 5]
        ds_result = ds_df.assign(value=ds_df['value'].fillna(0))[lambda x: x['value'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCumulativeAfterFilter:
    """Test cumulative operations after filtering"""

    def test_cumsum_after_filter(self):
        """Filter -> Cumsum chain"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df[pd_df['category'] == 'A']['value'].cumsum()
        ds_result = ds_df[ds_df['category'] == 'A']['value'].cumsum()

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cummax_after_filter(self):
        """Filter -> Cummax chain"""
        pd_df = pd.DataFrame({
            'value': [5, 2, 8, 3, 10, 1]
        })
        ds_df = DataStore({
            'value': [5, 2, 8, 3, 10, 1]
        })

        pd_result = pd_df[pd_df['value'] > 2]['value'].cummax()
        ds_result = ds_df[ds_df['value'] > 2]['value'].cummax()

        assert_series_equal(ds_result, pd_result, check_names=False)


class TestMultipleAggregationsSingleGroupby:
    """Test multiple aggregation functions in single groupby"""

    def test_groupby_multiple_agg_functions(self):
        """GroupBy with multiple named aggregations"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.groupby('category')['value'].agg(['sum', 'mean', 'min', 'max'])
        ds_result = ds_df.groupby('category')['value'].agg(['sum', 'mean', 'min', 'max'])

        # Compare values
        for col in pd_result.columns:
            pd_vals = sorted(pd_result[col].values)
            ds_vals = sorted(ds_result[col].values)
            assert np.allclose(pd_vals, ds_vals)


class TestSortAfterGroupby:
    """Test sort operations after groupby"""

    def test_groupby_sum_sort_index(self):
        """GroupBy -> Sum -> Sort by index"""
        pd_df = pd.DataFrame({
            'category': ['C', 'A', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['C', 'A', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.groupby('category')['value'].sum().sort_index()
        ds_result = ds_df.groupby('category')['value'].sum().sort_index()

        # Compare sorted results
        assert list(pd_result.index) == list(ds_result.index)
        assert list(pd_result.values) == list(ds_result.values)


class TestHeadTailAfterOperations:
    """Test head/tail after various operations"""

    def test_head_after_sort(self):
        """Sort -> Head chain"""
        pd_df = pd.DataFrame({
            'value': [5, 2, 8, 1, 9, 3]
        })
        ds_df = DataStore({
            'value': [5, 2, 8, 1, 9, 3]
        })

        pd_result = pd_df.sort_values('value').head(3)
        ds_result = ds_df.sort_values('value').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_after_filter(self):
        """Filter -> Tail chain"""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        ds_df = DataStore({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        pd_result = pd_df[pd_df['value'] > 3].tail(4)
        ds_result = ds_df[ds_df['value'] > 3].tail(4)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDropDuplicatesChain:
    """Test drop_duplicates in chains"""

    def test_filter_drop_duplicates_sort(self):
        """Filter -> Drop duplicates -> Sort"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
            'value': [10, 20, 10, 30, 20, 40, 30]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
            'value': [10, 20, 10, 30, 20, 40, 30]
        })

        pd_result = (pd_df[pd_df['value'] > 15]
                     .drop_duplicates(subset=['category'])
                     .sort_values('category'))
        ds_result = (ds_df[ds_df['value'] > 15]
                     .drop_duplicates(subset=['category'])
                     .sort_values('category'))

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRenameChain:
    """Test rename in chains"""

    def test_rename_then_filter(self):
        """Rename -> Filter chain"""
        pd_df = pd.DataFrame({
            'old_name': [1, 2, 3, 4, 5],
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore({
            'old_name': [1, 2, 3, 4, 5],
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })

        pd_result = (pd_df
                     .rename(columns={'old_name': 'new_name'})
                     [lambda x: x['new_name'] > 2])
        ds_result = (ds_df
                     .rename(columns={'old_name': 'new_name'})
                     [lambda x: x['new_name'] > 2])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSelectColumnsChain:
    """Test column selection in chains"""

    def test_filter_select_columns(self):
        """Filter -> Select columns chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df[pd_df['a'] > 2][['a', 'b']]
        ds_result = ds_df[ds_df['a'] > 2][['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_then_filter(self):
        """Select columns -> Filter chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df[['a', 'b']][lambda x: x['a'] > 2]
        ds_result = ds_df[['a', 'b']][lambda x: x['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexBooleanFilter:
    """Test complex boolean filters in chains"""

    def test_complex_boolean_then_groupby(self):
        """Complex boolean filter -> GroupBy"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value1': [10, 20, 30, 40, 50, 60],
            'value2': [5, 15, 25, 35, 45, 55],
            'flag': [True, False, True, False, True, False]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value1': [10, 20, 30, 40, 50, 60],
            'value2': [5, 15, 25, 35, 45, 55],
            'flag': [True, False, True, False, True, False]
        })

        condition = ((pd_df['value1'] > 15) & (pd_df['value2'] < 50)) | (pd_df['flag'] == True)
        pd_result = pd_df[condition].groupby('category')['value1'].sum()
        
        ds_condition = ((ds_df['value1'] > 15) & (ds_df['value2'] < 50)) | (ds_df['flag'] == True)
        ds_result = ds_df[ds_condition].groupby('category')['value1'].sum()

        # Compare sums
        assert sorted(list(pd_result)) == sorted(list(ds_result))


class TestSlicingChain:
    """Test slicing operations in chains"""

    def test_iloc_after_filter(self):
        """Filter -> iloc chain"""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        ds_df = DataStore({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

        pd_result = pd_df[pd_df['value'] > 3].iloc[1:4]
        ds_result = ds_df[ds_df['value'] > 3].iloc[1:4]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNrowsAfterOperations:
    """Test nrows/sample after operations"""

    def test_nlargest_after_filter(self):
        """Filter -> nlargest chain"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df[pd_df['category'] == 'A'].nlargest(2, 'value')
        ds_result = ds_df[ds_df['category'] == 'A'].nlargest(2, 'value')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_after_filter(self):
        """Filter -> nsmallest chain"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df[pd_df['category'] == 'B'].nsmallest(2, 'value')
        ds_result = ds_df[ds_df['category'] == 'B'].nsmallest(2, 'value')

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
