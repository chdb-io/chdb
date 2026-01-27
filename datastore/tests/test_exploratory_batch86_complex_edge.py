"""
Exploratory Batch 86: Complex Edge Cases and Advanced Operations

Focus areas:
1. Complex merge/join scenarios with filters
2. Multi-level groupby operations
3. Window functions with various partitions
4. String accessor chains
5. DateTime accessor chains
6. Complex expression evaluation
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestMergeComplexScenarios:
    """Test complex merge operations with various conditions"""

    def test_merge_with_pre_filter(self):
        """Merge after filtering both DataFrames"""
        pd_left = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'value1': [10, 20, 30, 40, 50]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'value2': [100, 200, 300, 400, 500]
        })

        ds_left = DataStore({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'value1': [10, 20, 30, 40, 50]
        })
        ds_right = DataStore({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'value2': [100, 200, 300, 400, 500]
        })

        pd_left_filtered = pd_left[pd_left['value1'] > 15]
        pd_right_filtered = pd_right[pd_right['value2'] < 450]
        pd_result = pd_left_filtered.merge(pd_right_filtered, on='key')

        ds_left_filtered = ds_left[ds_left['value1'] > 15]
        ds_right_filtered = ds_right[ds_right['value2'] < 450]
        ds_result = ds_left_filtered.merge(ds_right_filtered, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_with_post_filter(self):
        """Merge then filter the result"""
        pd_left = pd.DataFrame({
            'key': ['A', 'B', 'C'],
            'val1': [10, 20, 30]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'B', 'C'],
            'val2': [100, 200, 300]
        })

        ds_left = DataStore({
            'key': ['A', 'B', 'C'],
            'val1': [10, 20, 30]
        })
        ds_right = DataStore({
            'key': ['A', 'B', 'C'],
            'val2': [100, 200, 300]
        })

        pd_result = pd_left.merge(pd_right, on='key')
        pd_result = pd_result[pd_result['val1'] + pd_result['val2'] > 150]

        ds_result = ds_left.merge(ds_right, on='key')
        ds_result = ds_result[ds_result['val1'] + ds_result['val2'] > 150]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_with_different_key_names(self):
        """Merge with left_on and right_on"""
        pd_left = pd.DataFrame({
            'left_key': ['X', 'Y', 'Z'],
            'data1': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'right_key': ['X', 'Y', 'Z'],
            'data2': [10, 20, 30]
        })

        ds_left = DataStore({
            'left_key': ['X', 'Y', 'Z'],
            'data1': [1, 2, 3]
        })
        ds_right = DataStore({
            'right_key': ['X', 'Y', 'Z'],
            'data2': [10, 20, 30]
        })

        pd_result = pd_left.merge(pd_right, left_on='left_key', right_on='right_key')
        ds_result = ds_left.merge(ds_right, left_on='left_key', right_on='right_key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_chain_merge(self):
        """Multiple merges in chain"""
        pd_df1 = pd.DataFrame({'key': ['A', 'B'], 'v1': [1, 2]})
        pd_df2 = pd.DataFrame({'key': ['A', 'B'], 'v2': [10, 20]})
        pd_df3 = pd.DataFrame({'key': ['A', 'B'], 'v3': [100, 200]})

        ds_df1 = DataStore({'key': ['A', 'B'], 'v1': [1, 2]})
        ds_df2 = DataStore({'key': ['A', 'B'], 'v2': [10, 20]})
        ds_df3 = DataStore({'key': ['A', 'B'], 'v3': [100, 200]})

        pd_result = pd_df1.merge(pd_df2, on='key').merge(pd_df3, on='key')
        ds_result = ds_df1.merge(ds_df2, on='key').merge(ds_df3, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestMultiLevelGroupBy:
    """Test multi-level groupby operations"""

    def test_groupby_two_columns(self):
        """GroupBy with two columns"""
        pd_df = pd.DataFrame({
            'cat1': ['A', 'A', 'B', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'cat1': ['A', 'A', 'B', 'B', 'A', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.groupby(['cat1', 'cat2'])['value'].sum().reset_index()
        ds_result = ds_df.groupby(['cat1', 'cat2'])['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_two_columns_multiple_aggs(self):
        """GroupBy with two columns and multiple aggregations"""
        pd_df = pd.DataFrame({
            'region': ['East', 'East', 'West', 'West', 'East', 'West'],
            'product': ['A', 'B', 'A', 'B', 'A', 'B'],
            'sales': [100, 200, 150, 250, 300, 100]
        })
        ds_df = DataStore({
            'region': ['East', 'East', 'West', 'West', 'East', 'West'],
            'product': ['A', 'B', 'A', 'B', 'A', 'B'],
            'sales': [100, 200, 150, 250, 300, 100]
        })

        pd_result = pd_df.groupby(['region', 'product']).agg({
            'sales': ['sum', 'mean']
        }).reset_index()
        pd_result.columns = ['region', 'product', 'sales_sum', 'sales_mean']

        ds_result = ds_df.groupby(['region', 'product']).agg({
            'sales': ['sum', 'mean']
        }).reset_index()
        ds_result.columns = ['region', 'product', 'sales_sum', 'sales_mean']

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_then_filter_on_agg(self):
        """GroupBy then filter based on aggregated value"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'amount': [100, 200, 50, 75, 300, 400]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'amount': [100, 200, 50, 75, 300, 400]
        })

        pd_result = pd_df.groupby('category')['amount'].sum().reset_index()
        pd_result = pd_result[pd_result['amount'] > 200]

        ds_result = ds_df.groupby('category')['amount'].sum().reset_index()
        ds_result = ds_result[ds_result['amount'] > 200]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringAccessorChains:
    """Test string accessor operations in chains"""

    def test_str_upper_then_filter(self):
        """String upper then filter on result"""
        pd_df = pd.DataFrame({
            'name': ['alice', 'bob', 'charlie'],
            'score': [85, 90, 78]
        })
        ds_df = DataStore({
            'name': ['alice', 'bob', 'charlie'],
            'score': [85, 90, 78]
        })

        pd_df['name_upper'] = pd_df['name'].str.upper()
        pd_result = pd_df[pd_df['score'] > 80]

        ds_df['name_upper'] = ds_df['name'].str.upper()
        ds_result = ds_df[ds_df['score'] > 80]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_in_filter(self):
        """Use string length in filter condition"""
        pd_df = pd.DataFrame({
            'word': ['hi', 'hello', 'hey', 'howdy', 'hola'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'word': ['hi', 'hello', 'hey', 'howdy', 'hola'],
            'value': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df[pd_df['word'].str.len() > 3]
        ds_result = ds_df[ds_df['word'].str.len() > 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_filter(self):
        """Filter using str.contains"""
        pd_df = pd.DataFrame({
            'text': ['apple pie', 'banana bread', 'cherry cake', 'apple tart'],
            'price': [10, 15, 20, 12]
        })
        ds_df = DataStore({
            'text': ['apple pie', 'banana bread', 'cherry cake', 'apple tart'],
            'price': [10, 15, 20, 12]
        })

        pd_result = pd_df[pd_df['text'].str.contains('apple')]
        ds_result = ds_df[ds_df['text'].str.contains('apple')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_startswith_filter(self):
        """Filter using str.startswith"""
        pd_df = pd.DataFrame({
            'code': ['ABC123', 'DEF456', 'ABC789', 'GHI012'],
            'value': [100, 200, 300, 400]
        })
        ds_df = DataStore({
            'code': ['ABC123', 'DEF456', 'ABC789', 'GHI012'],
            'value': [100, 200, 300, 400]
        })

        pd_result = pd_df[pd_df['code'].str.startswith('ABC')]
        ds_result = ds_df[ds_df['code'].str.startswith('ABC')]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeAccessorChains:
    """Test datetime accessor operations in chains"""

    def test_dt_year_groupby(self):
        """Group by year extracted from datetime"""
        pd_df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10', '2021-08-25']),
            'value': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': pd.to_datetime(['2020-01-15', '2020-06-20', '2021-03-10', '2021-08-25']),
            'value': [100, 200, 150, 250]
        })

        pd_df['year'] = pd_df['date'].dt.year
        pd_result = pd_df.groupby('year')['value'].sum().reset_index()

        ds_df['year'] = ds_df['date'].dt.year
        ds_result = ds_df.groupby('year')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dt_month_filter(self):
        """Filter by month extracted from datetime"""
        pd_df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-15', '2020-06-20', '2020-01-25', '2020-08-10']),
            'value': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': pd.to_datetime(['2020-01-15', '2020-06-20', '2020-01-25', '2020-08-10']),
            'value': [100, 200, 150, 250]
        })

        pd_result = pd_df[pd_df['date'].dt.month == 1]
        ds_result = ds_df[ds_df['date'].dt.month == 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_day_sort(self):
        """Sort by day of month"""
        pd_df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-25', '2020-01-10', '2020-01-15', '2020-01-05']),
            'value': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': pd.to_datetime(['2020-01-25', '2020-01-10', '2020-01-15', '2020-01-05']),
            'value': [100, 200, 150, 250]
        })

        pd_df['day'] = pd_df['date'].dt.day
        pd_result = pd_df.sort_values('day')

        ds_df['day'] = ds_df['date'].dt.day
        ds_result = ds_df.sort_values('day')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexExpressions:
    """Test complex expression evaluation"""

    def test_arithmetic_chain(self):
        """Chain of arithmetic operations"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_df['c'] = pd_df['a'] * 2 + pd_df['b'] / 10
        pd_df['d'] = (pd_df['c'] - pd_df['a']) * 3

        ds_df['c'] = ds_df['a'] * 2 + ds_df['b'] / 10
        ds_df['d'] = (ds_df['c'] - ds_df['a']) * 3

        pd_result = pd_df[['a', 'b', 'c', 'd']]
        ds_result = ds_df[['a', 'b', 'c', 'd']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_comparison_chain(self):
        """Chain of comparison operations"""
        pd_df = pd.DataFrame({
            'x': [1, 5, 3, 7, 2],
            'y': [2, 4, 6, 3, 8],
            'z': [3, 3, 3, 3, 3]
        })
        ds_df = DataStore({
            'x': [1, 5, 3, 7, 2],
            'y': [2, 4, 6, 3, 8],
            'z': [3, 3, 3, 3, 3]
        })

        # Filter: x < y AND y > z
        pd_result = pd_df[(pd_df['x'] < pd_df['y']) & (pd_df['y'] > pd_df['z'])]
        ds_result = ds_df[(ds_df['x'] < ds_df['y']) & (ds_df['y'] > ds_df['z'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_operations(self):
        """Mixed arithmetic and comparison"""
        pd_df = pd.DataFrame({
            'price': [100, 200, 150, 300, 250],
            'quantity': [5, 3, 8, 2, 4]
        })
        ds_df = DataStore({
            'price': [100, 200, 150, 300, 250],
            'quantity': [5, 3, 8, 2, 4]
        })

        pd_df['total'] = pd_df['price'] * pd_df['quantity']
        pd_result = pd_df[pd_df['total'] > 500]

        ds_df['total'] = ds_df['price'] * ds_df['quantity']
        ds_result = ds_df[ds_df['total'] > 500]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSortingVariations:
    """Test various sorting scenarios"""

    def test_multi_column_sort(self):
        """Sort by multiple columns"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'B', 'A', 'B'],
            'val': [3, 1, 2, 3, 1, 2]
        })
        ds_df = DataStore({
            'cat': ['A', 'B', 'A', 'B', 'A', 'B'],
            'val': [3, 1, 2, 3, 1, 2]
        })

        pd_result = pd_df.sort_values(['cat', 'val'])
        ds_result = ds_df.sort_values(['cat', 'val'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_ascending(self):
        """Sort with mixed ascending/descending"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 1, 2, 1],
            'b': [5, 5, 3, 3, 4]
        })
        ds_df = DataStore({
            'a': [1, 2, 1, 2, 1],
            'b': [5, 5, 3, 3, 4]
        })

        pd_result = pd_df.sort_values(['a', 'b'], ascending=[True, False])
        ds_result = ds_df.sort_values(['a', 'b'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_after_groupby_agg(self):
        """Sort result of groupby aggregation"""
        pd_df = pd.DataFrame({
            'group': ['C', 'A', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'group': ['C', 'A', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_result.sort_values('value', ascending=False)

        ds_result = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_result.sort_values('value', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullHandling:
    """Test NULL/NaN handling in various operations"""

    def test_dropna_then_filter(self):
        """Drop NA then apply filter"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, 4.0, np.nan],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1.0, np.nan, 3.0, 4.0, np.nan],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.dropna()
        pd_result = pd_result[pd_result['a'] > 2]

        ds_result = ds_df.dropna()
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_then_groupby(self):
        """Fill NA then groupby"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10.0, np.nan, 30.0, np.nan]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10.0, np.nan, 30.0, np.nan]
        })

        pd_result = pd_df.fillna(0)
        pd_result = pd_result.groupby('group')['value'].sum().reset_index()

        ds_result = ds_df.fillna(0)
        ds_result = ds_result.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_isna_filter(self):
        """Filter using isna()"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['a'].isna()]
        ds_result = ds_df[ds_df['a'].isna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notna_filter(self):
        """Filter using notna()"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSlicingOperations:
    """Test slicing operations in chains"""

    def test_head_after_filter(self):
        """head() after filter"""
        pd_df = pd.DataFrame({
            'a': range(20),
            'b': range(20, 40)
        })
        ds_df = DataStore({
            'a': range(20),
            'b': range(20, 40)
        })

        pd_result = pd_df[pd_df['a'] > 5].head(5)
        ds_result = ds_df[ds_df['a'] > 5].head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_after_sort(self):
        """tail() after sort"""
        pd_df = pd.DataFrame({
            'a': [5, 2, 8, 1, 9, 3, 7],
            'b': [50, 20, 80, 10, 90, 30, 70]
        })
        ds_df = DataStore({
            'a': [5, 2, 8, 1, 9, 3, 7],
            'b': [50, 20, 80, 10, 90, 30, 70]
        })

        pd_result = pd_df.sort_values('a').tail(3)
        ds_result = ds_df.sort_values('a').tail(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_operation(self):
        """nlargest operation"""
        pd_df = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [100, 300, 200, 500, 400]
        })
        ds_df = DataStore({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [100, 300, 200, 500, 400]
        })

        pd_result = pd_df.nlargest(3, 'value')
        ds_result = ds_df.nlargest(3, 'value')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_operation(self):
        """nsmallest operation"""
        pd_df = pd.DataFrame({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [100, 300, 200, 500, 400]
        })
        ds_df = DataStore({
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [100, 300, 200, 500, 400]
        })

        pd_result = pd_df.nsmallest(3, 'value')
        ds_result = ds_df.nsmallest(3, 'value')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUniqueValueOperations:
    """Test unique value operations"""

    def test_drop_duplicates_subset(self):
        """Drop duplicates with subset"""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2, 3],
            'b': [10, 10, 20, 30, 30],
            'c': [100, 200, 300, 400, 500]
        })
        ds_df = DataStore({
            'a': [1, 1, 2, 2, 3],
            'b': [10, 10, 20, 30, 30],
            'c': [100, 200, 300, 400, 500]
        })

        pd_result = pd_df.drop_duplicates(subset=['a'])
        ds_result = ds_df.drop_duplicates(subset=['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """Drop duplicates keeping last"""
        pd_df = pd.DataFrame({
            'key': ['A', 'B', 'A', 'C', 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'key': ['A', 'B', 'A', 'C', 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df.drop_duplicates(subset=['key'], keep='last')
        ds_result = ds_df.drop_duplicates(subset=['key'], keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_value_counts_then_filter(self):
        """Value counts then filter high frequency"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A', 'B', 'A']
        })
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A', 'B', 'A']
        })

        pd_result = pd_df['category'].value_counts().reset_index()
        pd_result.columns = ['category', 'count']
        pd_result = pd_result[pd_result['count'] > 2]

        ds_result = ds_df['category'].value_counts().reset_index()
        ds_result.columns = ['category', 'count']
        ds_result = ds_result[ds_result['count'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestApplyTransform:
    """Test apply and transform operations"""

    def test_simple_apply(self):
        """Simple apply on column"""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'value': [1, 2, 3, 4, 5]
        })

        pd_df['doubled'] = pd_df['value'].apply(lambda x: x * 2)
        ds_df['doubled'] = ds_df['value'].apply(lambda x: x * 2)

        pd_result = pd_df[['value', 'doubled']]
        ds_result = ds_df[['value', 'doubled']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_then_filter(self):
        """Apply then filter on result"""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'value': [1, 2, 3, 4, 5]
        })

        pd_df['squared'] = pd_df['value'].apply(lambda x: x ** 2)
        pd_result = pd_df[pd_df['squared'] > 10]

        ds_df['squared'] = ds_df['value'].apply(lambda x: x ** 2)
        ds_result = ds_df[ds_df['squared'] > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)
