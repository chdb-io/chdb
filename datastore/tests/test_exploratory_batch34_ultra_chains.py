"""
Exploratory Batch 34: Ultra-long operation chains and SQL optimization verification

Focus areas:
1. Ultra-long lazy operation chains (10-15+ ops)
2. Complex multi-table operations with subsequent transformations
3. SQL optimization verification (chain merging)
4. Type coercion across complex chains
5. Memory efficiency with large intermediate results

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Test Group 1: Ultra-Long Filter Chains (10+ filters)
# =============================================================================


class TestUltraLongFilterChains:
    """Test very long chains of filter operations."""

    def test_ten_filter_chain_numeric(self):
        """Test 10 consecutive numeric filters."""
        data = {'a': list(range(100)), 'b': list(range(100, 200)), 'c': list(range(200, 300))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 10][pd_df['a'] < 90][pd_df['b'] > 110][pd_df['b'] < 190][pd_df['c'] > 210][
            pd_df['c'] < 290
        ][pd_df['a'] % 2 == 0][pd_df['b'] % 3 == 0][pd_df['a'] > 20][pd_df['b'] < 180]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 10][ds_df['a'] < 90][ds_df['b'] > 110][ds_df['b'] < 190][ds_df['c'] > 210][
            ds_df['c'] < 290
        ][ds_df['a'] % 2 == 0][ds_df['b'] % 3 == 0][ds_df['a'] > 20][ds_df['b'] < 180]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fifteen_filter_chain_mixed(self):
        """Test 15 consecutive filters with mixed conditions."""
        np.random.seed(42)
        data = {
            'x': np.random.randint(0, 100, 500),
            'y': np.random.randint(0, 100, 500),
            'z': np.random.randint(0, 100, 500),
            'name': [f'item_{i}' for i in range(500)],
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.copy()
        pd_result = pd_result[pd_result['x'] > 5]
        pd_result = pd_result[pd_result['x'] < 95]
        pd_result = pd_result[pd_result['y'] > 5]
        pd_result = pd_result[pd_result['y'] < 95]
        pd_result = pd_result[pd_result['z'] > 5]
        pd_result = pd_result[pd_result['z'] < 95]
        pd_result = pd_result[pd_result['x'] + pd_result['y'] > 50]
        pd_result = pd_result[pd_result['x'] + pd_result['y'] < 150]
        pd_result = pd_result[pd_result['y'] + pd_result['z'] > 50]
        pd_result = pd_result[pd_result['y'] + pd_result['z'] < 150]
        pd_result = pd_result[pd_result['x'] * 2 < 180]
        pd_result = pd_result[pd_result['y'] * 2 < 180]
        pd_result = pd_result[pd_result['z'] * 2 < 180]
        pd_result = pd_result[pd_result['x'] != pd_result['y']]
        pd_result = pd_result[pd_result['y'] != pd_result['z']]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.copy()
        ds_result = ds_result[ds_result['x'] > 5]
        ds_result = ds_result[ds_result['x'] < 95]
        ds_result = ds_result[ds_result['y'] > 5]
        ds_result = ds_result[ds_result['y'] < 95]
        ds_result = ds_result[ds_result['z'] > 5]
        ds_result = ds_result[ds_result['z'] < 95]
        ds_result = ds_result[ds_result['x'] + ds_result['y'] > 50]
        ds_result = ds_result[ds_result['x'] + ds_result['y'] < 150]
        ds_result = ds_result[ds_result['y'] + ds_result['z'] > 50]
        ds_result = ds_result[ds_result['y'] + ds_result['z'] < 150]
        ds_result = ds_result[ds_result['x'] * 2 < 180]
        ds_result = ds_result[ds_result['y'] * 2 < 180]
        ds_result = ds_result[ds_result['z'] * 2 < 180]
        ds_result = ds_result[ds_result['x'] != ds_result['y']]
        ds_result = ds_result[ds_result['y'] != ds_result['z']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_chain_with_string_ops(self):
        """Test filter chain involving string operations."""
        data = {
            'id': list(range(100)),
            'name': [f'user_{i:03d}' for i in range(100)],
            'email': [f'user{i}@example.com' for i in range(100)],
            'score': list(range(100)),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.copy()
        pd_result = pd_result[pd_result['id'] > 10]
        pd_result = pd_result[pd_result['id'] < 90]
        pd_result = pd_result[pd_result['score'] > 20]
        pd_result = pd_result[pd_result['score'] < 80]
        pd_result = pd_result[pd_result['name'].str.contains('user_0')]
        pd_result = pd_result[pd_result['email'].str.endswith('.com')]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.copy()
        ds_result = ds_result[ds_result['id'] > 10]
        ds_result = ds_result[ds_result['id'] < 90]
        ds_result = ds_result[ds_result['score'] > 20]
        ds_result = ds_result[ds_result['score'] < 80]
        ds_result = ds_result[ds_result['name'].str.contains('user_0')]
        ds_result = ds_result[ds_result['email'].str.endswith('.com')]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 2: Long Chain with Multiple Operation Types
# =============================================================================


class TestLongMixedOperationChains:
    """Test long chains mixing different operation types."""

    def test_filter_select_assign_chain(self):
        """Test chain of filter -> select -> assign operations."""
        data = {'a': list(range(50)), 'b': list(range(50, 100)), 'c': list(range(100, 150))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 10]
        pd_result = pd_result[['a', 'b']]
        pd_result = pd_result.assign(d=pd_result['a'] + pd_result['b'])
        pd_result = pd_result[pd_result['d'] > 80]
        pd_result = pd_result[['a', 'd']]
        pd_result = pd_result.assign(e=pd_result['d'] * 2)
        pd_result = pd_result[pd_result['e'] < 300]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 10]
        ds_result = ds_result[['a', 'b']]
        ds_result = ds_result.assign(d=ds_result['a'] + ds_result['b'])
        ds_result = ds_result[ds_result['d'] > 80]
        ds_result = ds_result[['a', 'd']]
        ds_result = ds_result.assign(e=ds_result['d'] * 2)
        ds_result = ds_result[ds_result['e'] < 300]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_head_filter_chain(self):
        """Test filter -> sort -> head -> filter chain."""
        np.random.seed(42)
        data = {'id': list(range(100)), 'value': np.random.randint(0, 1000, 100), 'category': ['A', 'B', 'C', 'D'] * 25}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['value'] > 100]
        pd_result = pd_result.sort_values('value', ascending=False)
        pd_result = pd_result.head(50)
        pd_result = pd_result[pd_result['category'].isin(['A', 'B'])]
        pd_result = pd_result.sort_values('id')
        pd_result = pd_result.head(20)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['value'] > 100]
        ds_result = ds_result.sort_values('value', ascending=False)
        ds_result = ds_result.head(50)
        ds_result = ds_result[ds_result['category'].isin(['A', 'B'])]
        ds_result = ds_result.sort_values('id')
        ds_result = ds_result.head(20)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_filter_sort_chain(self):
        """Test groupby -> agg -> filter -> sort chain."""
        data = {
            'group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'] * 5,
            'value': list(range(40)),
            'weight': [1.0, 1.5, 2.0, 2.5] * 10,
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('group').agg({'value': 'sum', 'weight': 'mean'}).reset_index()
        pd_result = pd_result[pd_result['value'] > 50]
        pd_result = pd_result.sort_values('weight', ascending=False)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('group').agg({'value': 'sum', 'weight': 'mean'}).reset_index()
        ds_result = ds_result[ds_result['value'] > 50]
        ds_result = ds_result.sort_values('weight', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_dropna_fillna_chain(self):
        """Test assign -> dropna -> fillna chain."""
        data = {
            'a': [1, 2, None, 4, 5, None, 7, 8, None, 10],
            'b': [None, 2, 3, None, 5, 6, None, 8, 9, None],
            'c': [1, None, 3, 4, None, 6, 7, None, 9, 10],
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(d=pd_df['a'].fillna(0) + pd_df['b'].fillna(0))
        pd_result = pd_result.dropna(subset=['c'])
        pd_result = pd_result.fillna({'a': -1, 'b': -1})
        pd_result = pd_result[pd_result['d'] > 0]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(d=ds_df['a'].fillna(0) + ds_df['b'].fillna(0))
        ds_result = ds_result.dropna(subset=['c'])
        ds_result = ds_result.fillna({'a': -1, 'b': -1})
        ds_result = ds_result[ds_result['d'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_sort_head_chain(self):
        """Test drop_duplicates -> sort -> head chain."""
        data = {'id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 5, 'value': list(range(50)), 'category': ['X', 'Y'] * 25}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.drop_duplicates(subset=['id'], keep='first')
        pd_result = pd_result.sort_values('value', ascending=False)
        pd_result = pd_result.head(3)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.drop_duplicates(subset=['id'], keep='first')
        ds_result = ds_result.sort_values('value', ascending=False)
        ds_result = ds_result.head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 3: Multi-Join with Subsequent Operations
# =============================================================================


class TestMultiJoinOperations:
    """Test multi-table joins followed by complex operations."""

    def test_three_table_join_then_groupby(self):
        """Test 3-table join followed by groupby aggregation."""
        # Table 1: Users
        users = {'user_id': [1, 2, 3, 4, 5], 'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']}
        # Table 2: Orders
        orders = {
            'order_id': list(range(10)),
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'amount': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        }
        # Table 3: Products
        products = {'order_id': list(range(10)), 'product': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']}

        # pandas
        pd_users = pd.DataFrame(users)
        pd_orders = pd.DataFrame(orders)
        pd_products = pd.DataFrame(products)
        pd_result = pd_users.merge(pd_orders, on='user_id').merge(pd_products, on='order_id')
        pd_result = pd_result.groupby('name').agg({'amount': 'sum', 'product': 'count'}).reset_index()
        pd_result = pd_result.sort_values('amount', ascending=False)

        # DataStore
        ds_users = DataStore(users)
        ds_orders = DataStore(orders)
        ds_products = DataStore(products)
        ds_result = ds_users.merge(ds_orders, on='user_id').merge(ds_products, on='order_id')
        ds_result = ds_result.groupby('name').agg({'amount': 'sum', 'product': 'count'}).reset_index()
        ds_result = ds_result.sort_values('amount', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_join_filter_assign_groupby(self):
        """Test join -> filter -> assign -> groupby chain."""
        df1 = {'key': [1, 2, 3, 4, 5], 'val1': [10, 20, 30, 40, 50]}
        df2 = {'key': [1, 2, 3, 4, 5], 'val2': [100, 200, 300, 400, 500], 'cat': ['A', 'B', 'A', 'B', 'A']}

        # pandas
        pd_df1 = pd.DataFrame(df1)
        pd_df2 = pd.DataFrame(df2)
        pd_result = pd_df1.merge(pd_df2, on='key')
        pd_result = pd_result[pd_result['val1'] > 15]
        pd_result = pd_result.assign(total=pd_result['val1'] + pd_result['val2'])
        pd_result = pd_result.groupby('cat').agg({'total': 'sum'}).reset_index()

        # DataStore
        ds_df1 = DataStore(df1)
        ds_df2 = DataStore(df2)
        ds_result = ds_df1.merge(ds_df2, on='key')
        ds_result = ds_result[ds_result['val1'] > 15]
        ds_result = ds_result.assign(total=ds_result['val1'] + ds_result['val2'])
        ds_result = ds_result.groupby('cat').agg({'total': 'sum'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_left_join_fillna_filter(self):
        """Test left join with missing values, then fillna and filter."""
        left = {'id': [1, 2, 3, 4, 5], 'val': [10, 20, 30, 40, 50]}
        right = {'id': [2, 4, 6], 'extra': [200, 400, 600]}

        # pandas
        pd_left = pd.DataFrame(left)
        pd_right = pd.DataFrame(right)
        pd_result = pd_left.merge(pd_right, on='id', how='left')
        pd_result = pd_result.fillna({'extra': 0})
        pd_result = pd_result[pd_result['extra'] > 0]

        # DataStore
        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds_left.merge(ds_right, on='id', how='left')
        ds_result = ds_result.fillna({'extra': 0})
        ds_result = ds_result[ds_result['extra'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_merge_select_columns(self):
        """Test multiple merges with column selection."""
        t1 = {'a': [1, 2, 3], 'b': [10, 20, 30]}
        t2 = {'a': [1, 2, 3], 'c': [100, 200, 300]}
        t3 = {'a': [1, 2, 3], 'd': [1000, 2000, 3000]}

        # pandas
        pd_t1 = pd.DataFrame(t1)
        pd_t2 = pd.DataFrame(t2)
        pd_t3 = pd.DataFrame(t3)
        pd_result = pd_t1.merge(pd_t2, on='a').merge(pd_t3, on='a')
        pd_result = pd_result[['a', 'b', 'd']]
        pd_result = pd_result.assign(total=pd_result['b'] + pd_result['d'])

        # DataStore
        ds_t1 = DataStore(t1)
        ds_t2 = DataStore(t2)
        ds_t3 = DataStore(t3)
        ds_result = ds_t1.merge(ds_t2, on='a').merge(ds_t3, on='a')
        ds_result = ds_result[['a', 'b', 'd']]
        ds_result = ds_result.assign(total=ds_result['b'] + ds_result['d'])

        # SQL JOIN operations don't guarantee row order without explicit ORDER BY
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 4: Concat with Subsequent Operations
# =============================================================================


class TestConcatOperationChains:
    """Test concat followed by various operations."""

    def test_concat_filter_groupby(self):
        """Test concat -> filter -> groupby chain."""
        df1 = {'id': [1, 2, 3], 'val': [10, 20, 30], 'src': ['A', 'A', 'A']}
        df2 = {'id': [4, 5, 6], 'val': [40, 50, 60], 'src': ['B', 'B', 'B']}
        df3 = {'id': [7, 8, 9], 'val': [70, 80, 90], 'src': ['C', 'C', 'C']}

        # pandas
        pd_df1 = pd.DataFrame(df1)
        pd_df2 = pd.DataFrame(df2)
        pd_df3 = pd.DataFrame(df3)
        pd_result = pd.concat([pd_df1, pd_df2, pd_df3], ignore_index=True)
        pd_result = pd_result[pd_result['val'] > 25]
        pd_result = pd_result.groupby('src').agg({'val': 'sum'}).reset_index()

        # DataStore
        ds_df1 = DataStore(df1)
        ds_df2 = DataStore(df2)
        ds_df3 = DataStore(df3)
        ds_result = ds.concat([ds_df1, ds_df2, ds_df3], ignore_index=True)
        ds_result = ds_result[ds_result['val'] > 25]
        ds_result = ds_result.groupby('src').agg({'val': 'sum'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_drop_duplicates_sort(self):
        """Test concat -> drop_duplicates -> sort chain."""
        df1 = {'key': [1, 2, 3], 'val': [10, 20, 30]}
        df2 = {'key': [2, 3, 4], 'val': [20, 30, 40]}
        df3 = {'key': [3, 4, 5], 'val': [30, 40, 50]}

        # pandas
        pd_df1 = pd.DataFrame(df1)
        pd_df2 = pd.DataFrame(df2)
        pd_df3 = pd.DataFrame(df3)
        pd_result = pd.concat([pd_df1, pd_df2, pd_df3], ignore_index=True)
        pd_result = pd_result.drop_duplicates(subset=['key'], keep='last')
        pd_result = pd_result.sort_values('val', ascending=False)

        # DataStore
        ds_df1 = DataStore(df1)
        ds_df2 = DataStore(df2)
        ds_df3 = DataStore(df3)
        ds_result = ds.concat([ds_df1, ds_df2, ds_df3], ignore_index=True)
        ds_result = ds_result.drop_duplicates(subset=['key'], keep='last')
        ds_result = ds_result.sort_values('val', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 5: Complex Aggregation Chains
# =============================================================================


class TestComplexAggregationChains:
    """Test complex aggregation operation chains."""

    def test_groupby_multiple_agg_filter_sort(self):
        """Test groupby with multiple aggregations, filter, and sort."""
        np.random.seed(42)
        data = {
            'group': ['A', 'B', 'C', 'D'] * 25,
            'value': np.random.randint(0, 100, 100),
            'weight': np.random.random(100),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('group').agg({'value': ['sum', 'mean', 'count'], 'weight': ['mean', 'std']})
        pd_result.columns = ['_'.join(col) for col in pd_result.columns]
        pd_result = pd_result.reset_index()
        pd_result = pd_result[pd_result['value_sum'] > 500]
        pd_result = pd_result.sort_values('value_mean', ascending=False)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('group').agg({'value': ['sum', 'mean', 'count'], 'weight': ['mean', 'std']})
        ds_result.columns = ['_'.join(col) for col in ds_result.columns]
        ds_result = ds_result.reset_index()
        ds_result = ds_result[ds_result['value_sum'] > 500]
        ds_result = ds_result.sort_values('value_mean', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nested_groupby_operations(self):
        """Test nested groupby operations (groupby result -> groupby)."""
        data = {
            'region': ['North', 'South', 'North', 'South'] * 10,
            'category': ['A', 'A', 'B', 'B'] * 10,
            'sales': list(range(40)),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        # First groupby
        pd_result = pd_df.groupby(['region', 'category']).agg({'sales': 'sum'}).reset_index()
        # Second groupby on the result
        pd_result = pd_result.groupby('region').agg({'sales': 'sum'}).reset_index()

        # DataStore
        ds_df = DataStore(data)
        # First groupby
        ds_result = ds_df.groupby(['region', 'category']).agg({'sales': 'sum'}).reset_index()
        # Second groupby on the result
        ds_result = ds_result.groupby('region').agg({'sales': 'sum'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 6: Type Coercion Across Chains
# =============================================================================


class TestTypeCoercionChains:
    """Test type coercion behavior across long operation chains."""

    def test_int_to_float_through_operations(self):
        """Test integer to float conversion through division."""
        data = {'a': [10, 20, 30, 40, 50], 'b': [3, 4, 5, 6, 7]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(ratio=pd_df['a'] / pd_df['b'])
        pd_result = pd_result[pd_result['ratio'] > 3.0]
        pd_result = pd_result.assign(ratio_squared=pd_result['ratio'] ** 2)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(ratio=ds_df['a'] / ds_df['b'])
        ds_result = ds_result[ds_result['ratio'] > 3.0]
        ds_result = ds_result.assign(ratio_squared=ds_result['ratio'] ** 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_type_arithmetic(self):
        """Test arithmetic with mixed int and float types."""
        data = {'int_col': [1, 2, 3, 4, 5], 'float_col': [1.5, 2.5, 3.5, 4.5, 5.5]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(
            sum_col=pd_df['int_col'] + pd_df['float_col'], product=pd_df['int_col'] * pd_df['float_col']
        )
        pd_result = pd_result[pd_result['sum_col'] > 5]
        pd_result = pd_result.assign(normalized=pd_result['product'] / pd_result['sum_col'])

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(
            sum_col=ds_df['int_col'] + ds_df['float_col'], product=ds_df['int_col'] * ds_df['float_col']
        )
        ds_result = ds_result[ds_result['sum_col'] > 5]
        ds_result = ds_result.assign(normalized=ds_result['product'] / ds_result['sum_col'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 7: Column Rename/Drop Chains
# =============================================================================


class TestColumnManipulationChains:
    """Test chains involving column rename and drop operations."""

    def test_rename_filter_rename_chain(self):
        """Test rename -> filter -> rename chain."""
        data = {'old_a': [1, 2, 3, 4, 5], 'old_b': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.rename(columns={'old_a': 'new_a', 'old_b': 'new_b'})
        pd_result = pd_result[pd_result['new_a'] > 2]
        pd_result = pd_result.rename(columns={'new_a': 'final_a'})

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.rename(columns={'old_a': 'new_a', 'old_b': 'new_b'})
        ds_result = ds_result[ds_result['new_a'] > 2]
        ds_result = ds_result.rename(columns={'new_a': 'final_a'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_assign_drop_chain(self):
        """Test drop -> assign -> drop chain."""
        data = {'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300], 'd': [1000, 2000, 3000]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.drop(columns=['d'])
        pd_result = pd_result.assign(e=pd_result['a'] + pd_result['b'])
        pd_result = pd_result.drop(columns=['c'])

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.drop(columns=['d'])
        ds_result = ds_result.assign(e=ds_result['a'] + ds_result['b'])
        ds_result = ds_result.drop(columns=['c'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix_suffix_chain(self):
        """Test add_prefix and add_suffix chains."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.add_prefix('pre_')
        pd_result = pd_result.add_suffix('_suf')

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.add_prefix('pre_')
        ds_result = ds_result.add_suffix('_suf')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 8: Edge Cases in Long Chains
# =============================================================================


class TestEdgeCasesInChains:
    """Test edge cases that might break long operation chains."""

    def test_empty_result_through_chain(self):
        """Test chain that results in empty DataFrame."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 10]  # Empty result
        pd_result = pd_result[['a', 'b']]
        pd_result = pd_result.assign(c=0)  # Assign on empty

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 10]
        ds_result = ds_result[['a', 'b']]
        ds_result = ds_result.assign(c=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_through_chain(self):
        """Test chain that results in single row."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] == 3]
        pd_result = pd_result.assign(c=pd_result['a'] * pd_result['b'])
        pd_result = pd_result[['a', 'c']]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] == 3]
        ds_result = ds_result.assign(c=ds_result['a'] * ds_result['b'])
        ds_result = ds_result[['a', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_rows_through_chain(self):
        """Test chain where filters don't remove any rows."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 0]  # All rows pass
        pd_result = pd_result[pd_result['b'] > 0]  # All rows pass
        pd_result = pd_result.assign(c=100)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 0]
        ds_result = ds_result[ds_df['b'] > 0]
        ds_result = ds_result.assign(c=100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_large_dataframe_chain(self):
        """Test chain on larger DataFrame (memory stress test)."""
        np.random.seed(42)
        n = 10000
        data = {
            'a': np.random.randint(0, 1000, n),
            'b': np.random.randint(0, 1000, n),
            'c': np.random.random(n),
            'cat': np.random.choice(['X', 'Y', 'Z'], n),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 100]
        pd_result = pd_result[pd_result['b'] < 900]
        pd_result = pd_result.assign(d=pd_result['a'] + pd_result['b'])
        pd_result = pd_result.groupby('cat').agg({'d': 'sum'}).reset_index()
        pd_result = pd_result.sort_values('d', ascending=False)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 100]
        ds_result = ds_result[ds_df['b'] < 900]
        ds_result = ds_result.assign(d=ds_result['a'] + ds_result['b'])
        ds_result = ds_result.groupby('cat').agg({'d': 'sum'}).reset_index()
        ds_result = ds_result.sort_values('d', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 9: Window Functions in Chains
# =============================================================================


class TestWindowFunctionChains:
    """Test window functions within operation chains."""

    def test_rolling_then_filter(self):
        """Test rolling operation followed by filter."""
        data = {'a': list(range(20)), 'b': [float(i * 10) for i in range(20)]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(rolling_b=pd_df['b'].rolling(3).mean())
        pd_result = pd_result.dropna()
        pd_result = pd_result[pd_result['rolling_b'] > 50]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(rolling_b=ds_df['b'].rolling(3).mean())
        ds_result = ds_result.dropna()
        ds_result = ds_result[ds_result['rolling_b'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_then_groupby(self):
        """Test cumsum followed by groupby."""
        data = {'group': ['A', 'A', 'A', 'B', 'B', 'B'] * 3, 'value': [1, 2, 3, 4, 5, 6] * 3}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(cumsum_val=pd_df['value'].cumsum())
        pd_result = pd_result.groupby('group').agg({'cumsum_val': 'max'}).reset_index()

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(cumsum_val=ds_df['value'].cumsum())
        ds_result = ds_result.groupby('group').agg({'cumsum_val': 'max'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_diff_chain(self):
        """Test shift and diff operations in chain."""
        data = {'a': [1, 3, 6, 10, 15, 21, 28]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(a_shift=pd_df['a'].shift(1), a_diff=pd_df['a'].diff())
        pd_result = pd_result.dropna()
        pd_result = pd_result[pd_result['a_diff'] > 3]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(a_shift=ds_df['a'].shift(1), a_diff=ds_df['a'].diff())
        ds_result = ds_result.dropna()
        ds_result = ds_result[ds_result['a_diff'] > 3]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 10: Apply/Transform in Chains
# =============================================================================


class TestApplyTransformChains:
    """Test apply and transform operations in chains."""

    def test_apply_then_filter(self):
        """Test apply followed by filter."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.apply(lambda x: x * 2)
        pd_result = pd_result[pd_result['a'] > 4]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.apply(lambda x: x * 2)
        ds_result = ds_result[ds_result['a'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_groupby_chain(self):
        """Test transform within groupby chain."""
        data = {'group': ['A', 'A', 'B', 'B', 'C', 'C'], 'value': [1, 2, 3, 4, 5, 6]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.copy()
        pd_result['normalized'] = pd_result.groupby('group')['value'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        pd_result = pd_result[pd_result['normalized'] > 0]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.copy()
        ds_result['normalized'] = ds_result.groupby('group')['value'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        ds_result = ds_result[ds_result['normalized'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 11: Numeric Precision in Chains
# =============================================================================


class TestNumericPrecisionChains:
    """Test numeric precision across operation chains."""

    def test_floating_point_accumulation(self):
        """Test floating point operations don't accumulate errors."""
        data = {'a': [0.1, 0.2, 0.3, 0.4, 0.5]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(b=pd_df['a'] * 10, c=pd_df['a'] * 100, d=pd_df['a'] * 1000)
        pd_result = pd_result.assign(ratio=pd_result['d'] / pd_result['b'])

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(b=ds_df['a'] * 10, c=ds_df['a'] * 100, d=ds_df['a'] * 1000)
        ds_result = ds_result.assign(ratio=ds_result['d'] / ds_result['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_division_by_near_zero(self):
        """Test division with values near zero."""
        data = {'a': [1.0, 2.0, 3.0, 4.0, 5.0], 'b': [0.001, 0.01, 0.1, 1.0, 10.0]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.assign(ratio=pd_df['a'] / pd_df['b'])
        pd_result = pd_result[pd_result['ratio'] < 1000]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.assign(ratio=ds_df['a'] / ds_df['b'])
        ds_result = ds_result[ds_result['ratio'] < 1000]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 12: Boolean Operations in Chains
# =============================================================================


class TestBooleanOperationChains:
    """Test boolean operations in chains."""

    def test_multiple_boolean_conditions(self):
        """Test multiple boolean conditions chained."""
        data = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'b': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['a'] < 8)]
        pd_result = pd_result[(pd_result['b'] > 3) & (pd_result['b'] < 8)]
        pd_result = pd_result[(pd_result['a'] + pd_result['b']) > 8]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[(ds_df['a'] > 2) & (ds_df['a'] < 8)]
        ds_result = ds_result[(ds_result['b'] > 3) & (ds_result['b'] < 8)]
        ds_result = ds_result[(ds_result['a'] + ds_result['b']) > 8]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_conditions_chain(self):
        """Test OR conditions in chain."""
        data = {'a': list(range(20)), 'b': list(range(20, 40))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[(pd_df['a'] < 5) | (pd_df['a'] > 15)]
        pd_result = pd_result[(pd_result['b'] < 25) | (pd_result['b'] > 35)]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[(ds_df['a'] < 5) | (ds_df['a'] > 15)]
        ds_result = ds_result[(ds_result['b'] < 25) | (ds_result['b'] > 35)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_in_chain(self):
        """Test negation operations in chain."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [True, False, True, False, True]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[~pd_df['b']]
        pd_result = pd_result[pd_result['a'] > 1]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[~ds_df['b']]
        ds_result = ds_result[ds_result['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)
