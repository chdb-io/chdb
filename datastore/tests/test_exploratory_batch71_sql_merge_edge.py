"""
Exploratory Batch 71: SQL Merge Edge Cases and Boundary Conditions

Focus areas:
1. Operations that break SQL merge (forcing pandas fallback)
2. Edge cases with chained assign-filter-assign operations
3. Column rename/reorder operations in chains
4. Operations with computed columns used in subsequent operations
5. Boundary conditions: empty results, single row, all nulls
6. Multiple consecutive filters vs single complex filter
7. Head/tail operations in the middle of chains

Discovery method: Architecture-based exploration based on sql_executor.py and query_planner.py
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestSQLMergeBreaking:
    """Test scenarios that should break SQL merge and fall back to pandas"""

    def test_assign_custom_function_breaks_sql(self):
        """Custom function in assign should force pandas fallback"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [10, 20, 30]
        })

        def custom_transform(x):
            return x ** 2 + 1

        pd_result = pd_df.copy()
        pd_result['transformed'] = pd_result['value'].apply(custom_transform)
        pd_result = pd_result[pd_result['transformed'] > 100]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['transformed'] = ds_result['value'].apply(custom_transform)
        ds_result = ds_result[ds_result['transformed'] > 100]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_assigns_then_filter(self):
        """Multiple column assignments followed by filter on new column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result['d'] = pd_result['c'] * 2
        pd_result['e'] = pd_result['d'] - pd_result['a']
        pd_result = pd_result[pd_result['e'] > 50]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['c'] = ds_result['a'] + ds_result['b']
        ds_result['d'] = ds_result['c'] * 2
        ds_result['e'] = ds_result['d'] - ds_result['a']
        ds_result = ds_result[ds_result['e'] > 50]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_filter_assign_pattern(self):
        """Interleaved filter and assign operations"""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        # Filter -> Assign -> Filter -> Assign
        pd_result = pd_df[pd_df['value'] > 15].copy()
        pd_result['doubled'] = pd_result['value'] * 2
        pd_result = pd_result[pd_result['doubled'] < 100]
        pd_result['final'] = pd_result['doubled'] + 10
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['value'] > 15]
        ds_result['doubled'] = ds_result['value'] * 2
        ds_result = ds_result[ds_result['doubled'] < 100]
        ds_result['final'] = ds_result['doubled'] + 10
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnRenameReorder:
    """Test column rename and reorder operations in chains"""

    def test_rename_then_filter_on_renamed(self):
        """Rename column then filter on the renamed column"""
        pd_df = pd.DataFrame({
            'old_name': [1, 2, 3, 4, 5],
            'other': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.rename(columns={'old_name': 'new_name'})
        pd_result = pd_result[pd_result['new_name'] > 2]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.rename(columns={'old_name': 'new_name'})
        ds_result = ds_result[ds_result['new_name'] > 2]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_subset_then_assign(self):
        """Select subset of columns then assign new column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        pd_result = pd_df[['a', 'b']].copy()
        pd_result['sum'] = pd_result['a'] + pd_result['b']

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[['a', 'b']]
        ds_result['sum'] = ds_result['a'] + ds_result['b']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reorder_columns_then_groupby(self):
        """Reorder columns then perform groupby"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value1': [1, 2, 3, 4],
            'value2': [10, 20, 30, 40]
        })
        # Reorder columns
        pd_result = pd_df[['value2', 'value1', 'group']]
        pd_result = pd_result.groupby('group').sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[['value2', 'value1', 'group']]
        ds_result = ds_result.groupby('group').sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_column_order=False)


class TestBoundaryConditions:
    """Test boundary conditions: empty, single row, all nulls"""

    def test_filter_results_empty(self):
        """Filter that produces empty result"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        pd_result = pd_df[pd_df['a'] > 100]  # No matches

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_chain_of_operations(self):
        """Chain of operations resulting in empty DataFrame"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['a'] > 2].copy()
        pd_result['c'] = pd_result['a'] * pd_result['b']
        pd_result = pd_result[pd_result['c'] > 1000]  # Should be empty

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result['c'] = ds_result['a'] * ds_result['b']
        ds_result = ds_result[ds_result['c'] > 1000]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_operations(self):
        """Operations on single-row DataFrame"""
        pd_df = pd.DataFrame({
            'a': [42],
            'b': ['single']
        })
        pd_result = pd_df.copy()
        pd_result['c'] = pd_result['a'] * 2
        pd_result = pd_result[pd_result['c'] > 50]

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['c'] = ds_result['a'] * 2
        ds_result = ds_result[ds_result['c'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_nulls_in_column_agg(self):
        """Aggregation on column with all null values"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B'],
            'value': [np.nan, np.nan, np.nan]
        })
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestConsecutiveFilters:
    """Test multiple consecutive filters vs single complex filter"""

    def test_two_consecutive_filters(self):
        """Two separate filter operations"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['a'] > 2]
        pd_result = pd_result[pd_result['b'] < 50]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result = ds_result[ds_result['b'] < 50]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_consecutive_filters(self):
        """Three separate filter operations"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8],
            'b': [10, 20, 30, 40, 50, 60, 70, 80],
            'c': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']
        })
        pd_result = pd_df[pd_df['a'] > 2]
        pd_result = pd_result[pd_result['b'] < 70]
        pd_result = pd_result[pd_result['c'] == 'x']
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result = ds_result[ds_result['b'] < 70]
        ds_result = ds_result[ds_result['c'] == 'x']
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combined_filter_vs_separate(self):
        """Compare combined AND filter vs separate filters - should give same result"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        # Combined filter
        pd_combined = pd_df[(pd_df['a'] > 2) & (pd_df['b'] < 50)].reset_index(drop=True)
        # Separate filters
        pd_separate = pd_df[pd_df['a'] > 2]
        pd_separate = pd_separate[pd_separate['b'] < 50].reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        # Separate filters with DataStore
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result = ds_result[ds_result['b'] < 50].reset_index(drop=True)

        # Both should match pandas combined result
        assert_datastore_equals_pandas(ds_result, pd_combined)


class TestHeadTailInChains:
    """Test head/tail operations in the middle of operation chains"""

    def test_filter_head_then_assign(self):
        """Filter, then head, then assign"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        pd_result = pd_df[pd_df['a'] > 3].head(4).copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 3].head(4)
        ds_result['c'] = ds_result['a'] + ds_result['b']
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_then_filter(self):
        """Head first, then filter"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        pd_result = pd_df.head(6)[pd_df.head(6)['a'] > 3]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_head = ds_df.head(6)
        ds_result = ds_head[ds_head['a'] > 3]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_then_groupby(self):
        """Tail first, then groupby"""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        pd_result = pd_df.tail(4).groupby('group')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.tail(4).groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestComputedColumnReference:
    """Test operations that reference computed columns"""

    def test_assign_then_reference_in_assign(self):
        """Assign a column then reference it in another assignment"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        pd_result = pd_df.copy()
        pd_result['b'] = pd_result['a'] * 2
        pd_result['c'] = pd_result['b'] + 10
        pd_result['d'] = pd_result['c'] * pd_result['a']

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['b'] = ds_result['a'] * 2
        ds_result['c'] = ds_result['b'] + 10
        ds_result['d'] = ds_result['c'] * ds_result['a']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_sort_by_new_column(self):
        """Assign a column then sort by it"""
        pd_df = pd.DataFrame({
            'name': ['Charlie', 'Alice', 'Bob', 'David'],
            'score': [75, 90, 85, 70]
        })
        pd_result = pd_df.copy()
        pd_result['adjusted'] = pd_result['score'] + 5
        pd_result = pd_result.sort_values('adjusted', ascending=False)
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['adjusted'] = ds_result['score'] + 5
        ds_result = ds_result.sort_values('adjusted', ascending=False)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_groupby_on_computed(self):
        """Assign a column then use it as groupby key"""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'score': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        pd_result = pd_df.copy()
        pd_result['bucket'] = (pd_result['value'] // 3).astype(int)
        pd_result = pd_result.groupby('bucket')['score'].mean().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['bucket'] = (ds_result['value'] // 3).astype(int)
        ds_result = ds_result.groupby('bucket')['score'].mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestComplexChains:
    """Test complex chains with 6+ operations"""

    def test_full_pipeline_etl_style(self):
        """Full ETL-style pipeline: filter, assign, rename, filter, groupby, sort"""
        pd_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
            'product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'amount': [100, 200, 150, 250, 300, 350, 50, 75],
            'status': ['complete', 'complete', 'pending', 'complete', 'complete', 'complete', 'pending', 'pending']
        })
        # Complex ETL pipeline
        pd_result = pd_df[pd_df['status'] == 'complete'].copy()
        pd_result['tax'] = pd_result['amount'] * 0.1
        pd_result['total'] = pd_result['amount'] + pd_result['tax']
        pd_result = pd_result.rename(columns={'user_id': 'customer'})
        pd_result = pd_result[pd_result['total'] > 100]
        pd_result = pd_result.groupby('customer')['total'].sum().reset_index()
        pd_result = pd_result.sort_values('total', ascending=False)
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['status'] == 'complete']
        ds_result['tax'] = ds_result['amount'] * 0.1
        ds_result['total'] = ds_result['amount'] + ds_result['tax']
        ds_result = ds_result.rename(columns={'user_id': 'customer'})
        ds_result = ds_result[ds_result['total'] > 100]
        ds_result = ds_result.groupby('customer')['total'].sum().reset_index()
        ds_result = ds_result.sort_values('total', ascending=False)
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_groupby_levels(self):
        """Two groupby operations in sequence (flatten then re-group)"""
        pd_df = pd.DataFrame({
            'region': ['East', 'East', 'West', 'West', 'East', 'West'],
            'city': ['NYC', 'Boston', 'LA', 'SF', 'NYC', 'LA'],
            'sales': [100, 200, 150, 250, 300, 350]
        })
        # First groupby by city
        pd_result = pd_df.groupby('city')['sales'].sum().reset_index()
        # Add region mapping
        city_to_region = {'NYC': 'East', 'Boston': 'East', 'LA': 'West', 'SF': 'West'}
        pd_result['region'] = pd_result['city'].map(city_to_region)
        # Second groupby by region
        pd_result = pd_result.groupby('region')['sales'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.groupby('city')['sales'].sum().reset_index()
        ds_result['region'] = ds_result['city'].map(city_to_region)
        ds_result = ds_result.groupby('region')['sales'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestOverwriteColumn:
    """Test operations that overwrite existing columns"""

    def test_overwrite_column_arithmetic(self):
        """Overwrite a column with arithmetic result"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.copy()
        pd_result['a'] = pd_result['a'] * 2
        pd_result = pd_result[pd_result['a'] > 4]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['a'] = ds_result['a'] * 2
        ds_result = ds_result[ds_result['a'] > 4]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_overwrite_column_multiple_times(self):
        """Overwrite same column multiple times"""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        pd_result = pd_df.copy()
        pd_result['value'] = pd_result['value'] * 2   # [2, 4, 6, 8, 10]
        pd_result['value'] = pd_result['value'] + 10  # [12, 14, 16, 18, 20]
        pd_result['value'] = pd_result['value'] // 2  # [6, 7, 8, 9, 10]

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['value'] = ds_result['value'] * 2
        ds_result['value'] = ds_result['value'] + 10
        ds_result['value'] = ds_result['value'] // 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_overwrite_filter(self):
        """Filter, overwrite column, then filter on overwritten column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8],
            'b': ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']
        })
        pd_result = pd_df[pd_df['a'] > 2].copy()
        pd_result['a'] = pd_result['a'] * 10
        pd_result = pd_result[pd_result['a'] > 50]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result['a'] = ds_result['a'] * 10
        ds_result = ds_result[ds_result['a'] > 50]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialValues:
    """Test handling of special values: inf, -inf, very large/small numbers"""

    def test_infinity_in_filter(self):
        """Filter with infinity values"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, 3.0, -np.inf, 5.0],
            'b': ['one', 'inf', 'three', 'neg_inf', 'five']
        })
        pd_result = pd_df[(pd_df['a'] != np.inf) & (pd_df['a'] != -np.inf)]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[(ds_df['a'] != np.inf) & (ds_df['a'] != -np.inf)]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_large_numbers(self):
        """Operations on very large numbers"""
        pd_df = pd.DataFrame({
            'a': [1e15, 2e15, 3e15],
            'b': [1, 2, 3]
        })
        pd_result = pd_df.copy()
        pd_result['c'] = pd_result['a'] / pd_result['b']
        pd_result = pd_result[pd_result['c'] > 1e15]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['c'] = ds_result['a'] / ds_result['b']
        ds_result = ds_result[ds_result['c'] > 1e15]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_nan_and_values(self):
        """Operations mixing NaN with valid values"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [10.0, 20.0, np.nan, 40.0, 50.0]
        })
        pd_result = pd_df.dropna().copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.dropna()
        ds_result['c'] = ds_result['a'] + ds_result['b']
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringOperationsInChains:
    """Test string operations mixed in operation chains"""

    def test_str_filter_then_numeric_agg(self):
        """Filter by string condition, then aggregate numeric"""
        pd_df = pd.DataFrame({
            'name': ['Alice Smith', 'Bob Jones', 'Charlie Smith', 'David Brown'],
            'score': [85, 90, 75, 88]
        })
        pd_result = pd_df[pd_df['name'].str.contains('Smith')].copy()
        pd_result = pd_result[['score']].mean()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['name'].str.contains('Smith')]
        ds_result = ds_result[['score']].mean()

        # Compare as scalars
        assert abs(float(ds_result['score']) - float(pd_result['score'])) < 0.01

    def test_str_transform_then_groupby(self):
        """String transformation followed by groupby"""
        pd_df = pd.DataFrame({
            'email': ['alice@example.com', 'bob@test.com', 'charlie@example.com', 'david@test.com'],
            'value': [100, 200, 150, 250]
        })
        pd_result = pd_df.copy()
        pd_result['domain'] = pd_result['email'].str.split('@').str[1]
        pd_result = pd_result.groupby('domain')['value'].sum().reset_index()

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.copy()
        ds_result['domain'] = ds_result['email'].str.split('@').str[1]
        ds_result = ds_result.groupby('domain')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestDropDuplicate:
    """Test drop_duplicates in chains"""

    def test_filter_then_drop_duplicates(self):
        """Filter then drop duplicates"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 2, 3, 3, 3, 4],
            'b': ['x', 'y', 'y', 'z', 'z', 'z', 'w']
        })
        pd_result = pd_df[pd_df['a'] > 1].drop_duplicates()
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 1].drop_duplicates()
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset_then_assign(self):
        """Drop duplicates on subset then assign"""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2, 3],
            'b': ['x', 'x', 'y', 'z', 'w'],
            'c': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.drop_duplicates(subset=['a']).copy()
        pd_result['d'] = pd_result['c'] * 2

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.drop_duplicates(subset=['a'])
        ds_result['d'] = ds_result['c'] * 2

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIlocInChains:
    """Test iloc operations in chains"""

    def test_filter_then_iloc(self):
        """Filter then select rows with iloc"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8],
            'b': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        })
        pd_result = pd_df[pd_df['a'] > 2].iloc[1:4]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df[ds_df['a'] > 2].iloc[1:4]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_then_assign(self):
        """Select rows with iloc then assign new column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.iloc[1:4].copy()
        pd_result['c'] = pd_result['a'] + pd_result['b']

        ds_df = DataStore(pd_df.copy())
        ds_result = ds_df.iloc[1:4]
        ds_result['c'] = ds_result['a'] + ds_result['b']

        assert_datastore_equals_pandas(ds_result, pd_result)
