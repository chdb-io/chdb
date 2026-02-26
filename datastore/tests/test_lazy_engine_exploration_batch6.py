"""
Exploratory tests for Lazy Execution Engine - Batch 6

Focus areas:
1. SQL merging verification - multiple operations merged into single SQL
2. Edge cases - nested subqueries for LIMIT-before-WHERE patterns
3. Mixed execution paths - SQL + Pandas alternation
4. Potential performance issues - unnecessary Pandas delegation

Uses Mirror Code Pattern: test DataStore vs pandas for correctness.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from io import StringIO

from datastore import DataStore
from datastore.config import get_logger
from tests.test_utils import get_dataframe, get_series, assert_datastore_equals_pandas, assert_series_equal


class TestSQLMergingVerification:
    """Test that multiple operations are correctly merged into single SQL."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [25, 30, 35, 40, 45],
                'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC'],
                'salary': [50000, 60000, 70000, 80000, 90000],
            }
        )
        self.ds = DataStore(self.df)

    def test_filter_select_sort_limit_merged(self):
        """filter + select + sort + limit should merge into single SQL."""
        # pandas
        pd_result = self.df[self.df['age'] > 25][['name', 'age']].sort_values('age').head(3)

        # DataStore - should merge all operations
        ds = self.ds
        ds_result = ds[ds['age'] > 25][['name', 'age']].sort_values('age').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_merged(self):
        """Multiple filters should merge with AND."""
        # pandas
        pd_result = self.df[(self.df['age'] > 25) & (self.df['salary'] > 60000)]

        # DataStore
        ds = self.ds
        ds_result = ds[(ds['age'] > 25) & (ds['salary'] > 60000)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_filters_merged(self):
        """Chained filter calls should merge."""
        # pandas
        pd_temp = self.df[self.df['age'] > 25]
        pd_result = pd_temp[pd_temp['salary'] > 60000]

        # DataStore - chained filters
        ds = self.ds
        ds_temp = ds[ds['age'] > 25]
        ds_result = ds_temp[ds_temp['salary'] > 60000]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_then_filter(self):
        """select then filter - order matters for column availability."""
        # pandas
        pd_result = self.df[['name', 'age']][self.df['age'] > 30]

        # DataStore
        ds = self.ds
        ds_result = ds[['name', 'age']][ds['age'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_head_sort(self):
        """sort -> head -> sort: requires nested subqueries."""
        # pandas
        pd_result = self.df.sort_values('age').head(4).sort_values('salary')

        # DataStore
        ds_result = self.ds.sort_values('age').head(4).sort_values('salary')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNestedSubqueryPatterns:
    """Test patterns that require nested subqueries."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'id': range(1, 11),
                'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            }
        )
        self.ds = DataStore(self.df)

    def test_head_then_filter(self):
        """head then filter: LIMIT before WHERE needs subquery."""
        # pandas: take first 6 rows, then filter
        pd_result = self.df.head(6)[self.df.head(6)['value'] > 30]

        # DataStore
        ds_head = self.ds.head(6)
        ds_result = ds_head[ds_head['value'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_then_filter(self):
        """iloc slice then filter: OFFSET+LIMIT before WHERE."""
        # pandas: rows 2-7, then filter
        pd_sliced = self.df.iloc[2:8]
        pd_result = pd_sliced[pd_sliced['value'] > 40]

        # DataStore
        ds_sliced = self.ds.iloc[2:8]
        ds_result = ds_sliced[ds_sliced['value'] > 40]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_head(self):
        """head().head(): multiple LIMITs need proper handling."""
        # pandas
        pd_result = self.df.head(7).head(4)

        # DataStore
        ds_result = self.ds.head(7).head(4)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_slice_combination(self):
        """head then iloc slice."""
        # pandas
        pd_result = self.df.head(8).iloc[2:6]

        # DataStore
        ds_result = self.ds.head(8).iloc[2:6]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_head_filter(self):
        """filter -> head -> filter: complex nesting."""
        # pandas
        pd_temp = self.df[self.df['value'] > 20].head(5)
        pd_result = pd_temp[pd_temp['value'] < 80]

        # DataStore
        ds_temp = self.ds[self.ds['value'] > 20].head(5)
        ds_result = ds_temp[ds_temp['value'] < 80]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMixedExecutionPaths:
    """Test SQL + Pandas mixed execution."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {'name': ['Alice', 'Bob', 'Charlie', 'David'], 'value': [100, 200, 300, 400], 'score': [1.5, 2.5, 3.5, 4.5]}
        )
        self.ds = DataStore(self.df)

    def test_filter_then_apply(self):
        """SQL filter then Pandas apply."""
        # pandas
        pd_filtered = self.df[self.df['value'] > 150]
        pd_result = pd_filtered.copy()
        pd_result['doubled'] = pd_filtered['value'].apply(lambda x: x * 2)

        # DataStore - filter (SQL) + apply (Pandas)
        ds_filtered = self.ds[self.ds['value'] > 150]
        ds_result = ds_filtered.assign(doubled=ds_filtered['value'].apply(lambda x: x * 2))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_filter(self):
        """filter -> assign -> filter: SQL-Pandas-SQL."""
        # pandas
        pd_temp = self.df[self.df['value'] > 100].copy()
        pd_temp['new_col'] = pd_temp['value'] * 2
        pd_result = pd_temp[pd_temp['new_col'] > 400]

        # DataStore
        ds_temp = self.ds[self.ds['value'] > 100]
        ds_temp = ds_temp.assign(new_col=ds_temp['value'] * 2)
        ds_result = ds_temp[ds_temp['new_col'] > 400]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_then_filter(self):
        """groupby agg then filter on result."""
        # pandas
        pd_agg = self.df.groupby('name')['value'].sum().reset_index()
        pd_result = pd_agg[pd_agg['value'] > 150]

        # DataStore
        ds_agg = self.ds.groupby('name')['value'].sum().reset_index()
        ds_result = ds_agg[ds_agg['value'] > 150]

        # groupby order undefined
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_transform_then_filter(self):
        """transform (Pandas) then filter."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['rank'] = pd_temp['value'].rank()
        pd_result = pd_temp[pd_temp['rank'] > 2]

        # DataStore - rank() might use pandas, then filter
        ds_temp = self.ds.assign(rank=self.ds['value'].rank())
        ds_result = ds_temp[ds_temp['rank'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByEdgeCases:
    """Test groupby edge cases for SQL merging."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
                'value': [10, 20, 30, 40, 50, 60],
                'count': [1, 2, 3, 4, 5, 6],
            }
        )
        self.ds = DataStore(self.df)

    def test_groupby_multiple_agg_functions(self):
        """groupby with multiple agg functions on same column."""
        # pandas
        pd_result = self.df.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()

        # DataStore
        ds_result = self.ds.groupby('category')['value'].agg(['sum', 'mean', 'count']).reset_index()

        # groupby order undefined
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_dict_agg(self):
        """groupby with dict specifying different functions per column."""
        # pandas
        pd_result = self.df.groupby('category').agg({'value': 'sum', 'count': 'mean'}).reset_index()

        # DataStore
        ds_result = self.ds.groupby('category').agg({'value': 'sum', 'count': 'mean'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_filter_on_agg(self):
        """Filter on aggregated values (HAVING equivalent)."""
        # pandas
        pd_agg = self.df.groupby('category')['value'].sum().reset_index()
        pd_result = pd_agg[pd_agg['value'] > 30]

        # DataStore - should translate to HAVING or filter after
        ds_agg = self.ds.groupby('category')['value'].sum().reset_index()
        ds_result = ds_agg[ds_agg['value'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_multiple_columns(self):
        """groupby on multiple columns."""
        # pandas
        pd_result = self.df.groupby(['category', 'subcategory'])['value'].sum().reset_index()

        # DataStore
        ds_result = self.ds.groupby(['category', 'subcategory'])['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_then_sort(self):
        """groupby then sort on aggregated values."""
        # pandas
        pd_result = self.df.groupby('category')['value'].sum().reset_index().sort_values('value')

        # DataStore
        ds_result = self.ds.groupby('category')['value'].sum().reset_index().sort_values('value')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticExpressionChaining:
    """Test arithmetic expression chaining for SQL generation."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})
        self.ds = DataStore(self.df)

    def test_simple_arithmetic(self):
        """Simple arithmetic should be SQL."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['sum'] = pd_temp['a'] + pd_temp['b']
        pd_result = pd_temp

        # DataStore
        ds_result = self.ds.assign(sum=self.ds['a'] + self.ds['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_complex_arithmetic(self):
        """Complex arithmetic chain."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['expr'] = (pd_temp['a'] * 2 + pd_temp['b']) / pd_temp['c']
        pd_result = pd_temp

        # DataStore
        ds_result = self.ds.assign(expr=(self.ds['a'] * 2 + self.ds['b']) / self.ds['c'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_on_arithmetic(self):
        """Filter based on arithmetic expression."""
        # pandas
        pd_result = self.df[(self.df['a'] + self.df['b']) > 30]

        # DataStore
        ds_result = self.ds[(self.ds['a'] + self.ds['b']) > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter_on_computed(self):
        """Assign computed column then filter on it."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['total'] = pd_temp['a'] + pd_temp['b']
        pd_result = pd_temp[pd_temp['total'] > 30]

        # DataStore
        ds_temp = self.ds.assign(total=self.ds['a'] + self.ds['b'])
        ds_result = ds_temp[ds_temp['total'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringOperationChaining:
    """Test string operation chaining for SQL vs Pandas."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'name': ['  Alice  ', '  BOB  ', '  Charlie  '],
                'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
            }
        )
        self.ds = DataStore(self.df)

    def test_trim_upper(self):
        """trim -> upper chain."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['clean'] = pd_temp['name'].str.strip().str.upper()
        pd_result = pd_temp

        # DataStore
        ds_result = self.ds.assign(clean=self.ds['name'].str.strip().str.upper())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_lower_contains_filter(self):
        """lower -> contains filter."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['lower_name'] = pd_temp['name'].str.lower()
        pd_result = pd_temp[pd_temp['lower_name'].str.contains('bob')]

        # DataStore
        ds_temp = self.ds.assign(lower_name=self.ds['name'].str.lower())
        ds_result = ds_temp[ds_temp['lower_name'].str.contains('bob')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_slice(self):
        """String slicing."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['domain'] = pd_temp['email'].str.split('@').str[1]
        pd_result = pd_temp

        # DataStore
        ds_result = self.ds.assign(domain=self.ds['email'].str.split('@').str[1])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWhereAndMaskOperations:
    """Test where/mask operations for SQL CASE WHEN generation."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'value': [10, -20, 30, -40, 50], 'category': ['A', 'B', 'A', 'B', 'A']})
        self.ds = DataStore(self.df)

    def test_where_simple(self):
        """Simple where operation."""
        # pandas
        pd_result = self.df['value'].where(self.df['value'] > 0, 0)

        # DataStore
        ds_result = self.ds['value'].where(self.ds['value'] > 0, 0)

        # Compare as Series
        assert_series_equal(
            get_series(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True), check_names=False
        )

    def test_mask_simple(self):
        """Simple mask operation."""
        # pandas
        pd_result = self.df['value'].mask(self.df['value'] < 0, 0)

        # DataStore
        ds_result = self.ds['value'].mask(self.ds['value'] < 0, 0)

        assert_series_equal(
            get_series(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True), check_names=False
        )

    def test_where_then_filter(self):
        """where then filter: chained operations."""
        # pandas
        pd_temp = self.df.copy()
        pd_temp['clean_value'] = pd_temp['value'].where(pd_temp['value'] > 0, 0)
        pd_result = pd_temp[pd_temp['clean_value'] > 0]

        # DataStore
        ds_temp = self.ds.assign(clean_value=self.ds['value'].where(self.ds['value'] > 0, 0))
        ds_result = ds_temp[ds_temp['clean_value'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexRealWorldChains:
    """Test complex chains similar to real-world usage."""

    def setup_method(self):
        """Create more realistic test data."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                'user_id': range(1, n + 1),
                'name': [f'User_{i}' for i in range(1, n + 1)],
                'age': np.random.randint(18, 65, n),
                'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n),
                'purchase_amount': np.random.uniform(10, 1000, n).round(2),
                'signup_year': np.random.choice([2020, 2021, 2022, 2023], n),
            }
        )
        self.ds = DataStore(self.df)

    def test_analytics_pipeline(self):
        """Typical analytics pipeline: filter -> groupby -> sort -> head."""
        # pandas
        pd_result = (
            self.df[self.df['age'] >= 25]
            .groupby('city')['purchase_amount']
            .sum()
            .reset_index()
            .sort_values('purchase_amount', ascending=False)
            .head(3)
        )

        # DataStore
        ds_result = (
            self.ds[self.ds['age'] >= 25]
            .groupby('city')['purchase_amount']
            .sum()
            .reset_index()
            .sort_values('purchase_amount', ascending=False)
            .head(3)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_feature_engineering_pipeline_simple(self):
        """Feature engineering: single assign + filter + column select."""
        # pandas - simple version with one computed column
        pd_temp = self.df.copy()
        pd_temp['purchase_per_age'] = pd_temp['purchase_amount'] / pd_temp['age']
        pd_result = pd_temp[pd_temp['purchase_per_age'] > 10][['name', 'city', 'purchase_per_age']]

        # DataStore
        ds_temp = self.ds.assign(purchase_per_age=self.ds['purchase_amount'] / self.ds['age'])
        ds_result = ds_temp[ds_temp['purchase_per_age'] > 10][['name', 'city', 'purchase_per_age']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_feature_engineering_pipeline_chained_assigns(self):
        """Feature engineering: chained assigns where one depends on another.

        This tests a complex scenario where:
        1. First assign creates a computed column (years_since_signup)
        2. Second assign uses that column (purchase_per_year = amount / years_since_signup)
        3. Then filter and column select

        Currently fails because when doing column selection, the SQL generator
        doesn't properly include all dependent computed columns in the query.
        """
        # pandas
        pd_temp = self.df.copy()
        pd_temp['years_since_signup'] = 2024 - pd_temp['signup_year']
        pd_temp['purchase_per_year'] = pd_temp['purchase_amount'] / pd_temp['years_since_signup']
        pd_result = pd_temp[pd_temp['purchase_per_year'] > 100][['name', 'city', 'purchase_per_year']]

        # DataStore
        ds_temp = self.ds.assign(years_since_signup=2024 - self.ds['signup_year'])
        ds_temp = ds_temp.assign(purchase_per_year=ds_temp['purchase_amount'] / ds_temp['years_since_signup'])
        ds_result = ds_temp[ds_temp['purchase_per_year'] > 100][['name', 'city', 'purchase_per_year']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_level_chained_dependency(self):
        """Three levels of chained computed columns: step1 -> step2 -> step3."""
        # Create simple test data
        df = pd.DataFrame({'base': [100, 200, 300, 400], 'rate': [0.1, 0.2, 0.15, 0.25]})

        # pandas
        pd_df = df.copy()
        pd_df['step1'] = pd_df['base'] * pd_df['rate']  # 10, 40, 45, 100
        pd_df['step2'] = pd_df['step1'] * 2  # 20, 80, 90, 200
        pd_df['step3'] = pd_df['step2'] + pd_df['base']  # 120, 280, 390, 600
        pd_result = pd_df[pd_df['step3'] > 200][['base', 'step3']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(step1=ds['base'] * ds['rate'])
        ds = ds.assign(step2=ds['step1'] * 2)
        ds = ds.assign(step3=ds['step2'] + ds['base'])
        ds_result = ds[ds['step3'] > 200][['base', 'step3']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_arithmetic_chained_assigns(self):
        """Chained assigns with mixed arithmetic operations (add, sub, mul, div)."""
        df = pd.DataFrame({'a': [10, 20, 30, 40], 'b': [2, 4, 5, 8], 'c': [100, 200, 300, 400]})

        # pandas
        pd_df = df.copy()
        pd_df['sum_ab'] = pd_df['a'] + pd_df['b']  # 12, 24, 35, 48
        pd_df['ratio'] = pd_df['c'] / pd_df['sum_ab']  # 8.33, 8.33, 8.57, 8.33
        pd_df['final'] = pd_df['ratio'] * pd_df['a']  # 83.3, 166.6, 257.1, 333.3
        pd_result = pd_df[pd_df['final'] > 150][['a', 'b', 'c', 'final']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(sum_ab=ds['a'] + ds['b'])
        ds = ds.assign(ratio=ds['c'] / ds['sum_ab'])
        ds = ds.assign(final=ds['ratio'] * ds['a'])
        ds_result = ds[ds['final'] > 150][['a', 'b', 'c', 'final']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_one_column_depends_on_multiple_computed(self):
        """One computed column depends on multiple other computed columns."""
        df = pd.DataFrame(
            {
                'price': [100, 200, 300, 400],
                'quantity': [10, 5, 8, 3],
                'discount_rate': [0.1, 0.2, 0.15, 0.05],
            }
        )

        # pandas
        pd_df = df.copy()
        pd_df['subtotal'] = pd_df['price'] * pd_df['quantity']  # 1000, 1000, 2400, 1200
        pd_df['discount'] = pd_df['subtotal'] * pd_df['discount_rate']  # 100, 200, 360, 60
        pd_df['final_price'] = pd_df['subtotal'] - pd_df['discount']  # 900, 800, 2040, 1140
        pd_result = pd_df[pd_df['final_price'] > 850][['price', 'quantity', 'final_price']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(subtotal=ds['price'] * ds['quantity'])
        ds = ds.assign(discount=ds['subtotal'] * ds['discount_rate'])
        ds = ds.assign(final_price=ds['subtotal'] - ds['discount'])
        ds_result = ds[ds['final_price'] > 850][['price', 'quantity', 'final_price']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diamond_dependency_pattern(self):
        """Diamond dependency: root -> branch1, branch2 -> merged."""
        df = pd.DataFrame({'x': [10, 20, 30, 40], 'y': [1, 2, 3, 4]})

        # pandas
        pd_df = df.copy()
        pd_df['branch1'] = pd_df['x'] * 2  # 20, 40, 60, 80
        pd_df['branch2'] = pd_df['x'] + 10  # 20, 30, 40, 50
        pd_df['merged'] = pd_df['branch1'] + pd_df['branch2']  # 40, 70, 100, 130
        pd_result = pd_df[pd_df['merged'] > 60][['x', 'merged']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(branch1=ds['x'] * 2)
        ds = ds.assign(branch2=ds['x'] + 10)
        ds = ds.assign(merged=ds['branch1'] + ds['branch2'])
        ds_result = ds[ds['merged'] > 60][['x', 'merged']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_functions_chained(self):
        """String functions with chained assigns."""
        df = pd.DataFrame(
            {
                'first_name': ['alice', 'bob', 'charlie'],
                'last_name': ['smith', 'jones', 'brown'],
                'score': [85, 92, 78],
            }
        )

        # pandas
        pd_df = df.copy()
        pd_df['first_upper'] = pd_df['first_name'].str.upper()
        pd_df['last_upper'] = pd_df['last_name'].str.upper()
        pd_result = pd_df[pd_df['score'] > 80][['first_upper', 'last_upper', 'score']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(first_upper=ds['first_name'].str.upper())
        ds = ds.assign(last_upper=ds['last_name'].str.upper())
        ds_result = ds[ds['score'] > 80][['first_upper', 'last_upper', 'score']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_complex_data_analysis_pipeline(self):
        """Complex analytics: multiple chained assigns + filter + column select."""
        # pandas
        pd_df = self.df.copy()
        pd_df['years_active'] = 2024 - pd_df['signup_year']
        pd_df['avg_annual_purchase'] = pd_df['purchase_amount'] / pd_df['years_active']
        pd_result = pd_df[pd_df['avg_annual_purchase'] > 150][['name', 'city', 'avg_annual_purchase']]

        # DataStore
        ds = DataStore(self.df)
        ds = ds.assign(years_active=2024 - ds['signup_year'])
        ds = ds.assign(avg_annual_purchase=ds['purchase_amount'] / ds['years_active'])
        ds_result = ds[ds['avg_annual_purchase'] > 150][['name', 'city', 'avg_annual_purchase']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_filter_simple(self):
        """Filter using a boolean computed column: ds[ds['is_high']]."""
        df = pd.DataFrame({'x': [100, 200, 300, 50], 'y': [2, 4, 3, 10]})

        # pandas
        pd_df = df.copy()
        pd_df['ratio'] = pd_df['x'] / pd_df['y']
        pd_df['is_high'] = pd_df['ratio'] > 50
        pd_result = pd_df[pd_df['is_high']][['x', 'y', 'ratio']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(ratio=ds['x'] / ds['y'])
        ds = ds.assign(is_high=ds['ratio'] > 50)
        ds_result = ds[ds['is_high']][['x', 'y', 'ratio']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_filter_complex(self):
        """Complex analytics with boolean computed column filter."""
        # pandas
        pd_df = self.df.copy()
        pd_df['years_active'] = 2024 - pd_df['signup_year']
        pd_df['avg_annual_purchase'] = pd_df['purchase_amount'] / pd_df['years_active']
        pd_df['is_valuable'] = pd_df['avg_annual_purchase'] > 150
        pd_result = pd_df[pd_df['is_valuable']][['name', 'city', 'avg_annual_purchase']]

        # DataStore
        ds = DataStore(self.df)
        ds = ds.assign(years_active=2024 - ds['signup_year'])
        ds = ds.assign(avg_annual_purchase=ds['purchase_amount'] / ds['years_active'])
        ds = ds.assign(is_valuable=ds['avg_annual_purchase'] > 150)
        ds_result = ds[ds['is_valuable']][['name', 'city', 'avg_annual_purchase']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_chained_filter(self):
        """Chain boolean computed column filter with additional condition."""
        df = pd.DataFrame(
            {
                'product': ['A', 'B', 'C', 'D', 'E'],
                'sales': [1000, 500, 1500, 300, 800],
                'region': ['East', 'West', 'East', 'West', 'East'],
            }
        )

        # pandas
        pd_df = df.copy()
        pd_df['is_high_sales'] = pd_df['sales'] > 700
        pd_result = pd_df[(pd_df['is_high_sales']) & (pd_df['region'] == 'East')][['product', 'sales']]

        # DataStore - chain filters
        ds = DataStore(df)
        ds = ds.assign(is_high_sales=ds['sales'] > 700)
        ds_result = ds[ds['is_high_sales']][ds['region'] == 'East'][['product', 'sales']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_multi_layer_dependency(self):
        """Boolean computed column depending on multiple layers of computed columns."""
        df = pd.DataFrame(
            {
                'base_price': [100, 200, 150, 80, 300],
                'markup_pct': [0.2, 0.3, 0.15, 0.25, 0.1],
                'tax_rate': [0.1, 0.1, 0.08, 0.1, 0.12],
            }
        )

        # pandas
        pd_df = df.copy()
        pd_df['markup'] = pd_df['base_price'] * pd_df['markup_pct']
        pd_df['price_with_markup'] = pd_df['base_price'] + pd_df['markup']
        pd_df['final_price'] = pd_df['price_with_markup'] * (1 + pd_df['tax_rate'])
        pd_df['is_expensive'] = pd_df['final_price'] > 200
        pd_result = pd_df[pd_df['is_expensive']][['base_price', 'final_price']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(markup=ds['base_price'] * ds['markup_pct'])
        ds = ds.assign(price_with_markup=ds['base_price'] + ds['markup'])
        ds = ds.assign(final_price=ds['price_with_markup'] * (1 + ds['tax_rate']))
        ds = ds.assign(is_expensive=ds['final_price'] > 200)
        ds_result = ds[ds['is_expensive']][['base_price', 'final_price']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_string_equality(self):
        """Boolean computed column using string equality."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50], 'category': ['A', 'B', 'A', 'B', 'A']})

        # pandas - test ==
        pd_df = df.copy()
        pd_df['is_category_a'] = pd_df['category'] == 'A'
        pd_result = pd_df[pd_df['is_category_a']][['value', 'category']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(is_category_a=ds['category'] == 'A')
        ds_result = ds[ds['is_category_a']][['value', 'category']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_string_inequality(self):
        """Boolean computed column using string inequality (!=)."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50], 'category': ['A', 'B', 'A', 'B', 'A']})

        # pandas - test !=
        pd_df = df.copy()
        pd_df['is_not_category_a'] = pd_df['category'] != 'A'
        pd_result = pd_df[pd_df['is_not_category_a']][['value', 'category']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(is_not_category_a=ds['category'] != 'A')
        ds_result = ds[ds['is_not_category_a']][['value', 'category']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_computed_column_gte_lte(self):
        """Boolean computed column using >= and <=."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

        # pandas - test >=
        pd_df = df.copy()
        pd_df['is_gte_30'] = pd_df['value'] >= 30
        pd_result_gte = pd_df[pd_df['is_gte_30']][['value']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(is_gte_30=ds['value'] >= 30)
        ds_result_gte = ds[ds['is_gte_30']][['value']]

        assert_datastore_equals_pandas(ds_result_gte, pd_result_gte)

        # pandas - test <=
        pd_df = df.copy()
        pd_df['is_lte_30'] = pd_df['value'] <= 30
        pd_result_lte = pd_df[pd_df['is_lte_30']][['value']]

        # DataStore
        ds = DataStore(df)
        ds = ds.assign(is_lte_30=ds['value'] <= 30)
        ds_result_lte = ds[ds['is_lte_30']][['value']]

        assert_datastore_equals_pandas(ds_result_lte, pd_result_lte)

    def test_cohort_analysis_pipeline(self):
        """Cohort analysis: groupby multiple columns -> agg -> filter."""
        # pandas
        pd_result = (
            self.df.groupby(['city', 'signup_year']).agg({'user_id': 'count', 'purchase_amount': 'mean'}).reset_index()
        )
        pd_result = pd_result[pd_result['user_id'] >= 3]

        # DataStore
        ds_agg = (
            self.ds.groupby(['city', 'signup_year']).agg({'user_id': 'count', 'purchase_amount': 'mean'}).reset_index()
        )
        ds_result = ds_agg[ds_agg['user_id'] >= 3]

        # groupby order undefined
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestEdgeCasesForSQLGeneration:
    """Test edge cases that might cause SQL generation issues."""

    def setup_method(self):
        """Create test data with edge cases."""
        self.df = pd.DataFrame(
            {
                'normal_col': [1, 2, 3],
                'col with space': [4, 5, 6],
                'col-with-dash': [7, 8, 9],
                'Col_With_Case': [10, 11, 12],
            }
        )
        self.ds = DataStore(self.df)

    def test_column_name_with_space(self):
        """Column name with space should be properly quoted."""
        # pandas
        pd_result = self.df[['col with space']]

        # DataStore
        ds_result = self.ds[['col with space']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_name_with_dash(self):
        """Column name with dash should be properly quoted."""
        # pandas
        pd_result = self.df[['col-with-dash']]

        # DataStore
        ds_result = self.ds[['col-with-dash']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_on_special_column_name(self):
        """Filter on column with special characters."""
        # pandas
        pd_result = self.df[self.df['col with space'] > 4]

        # DataStore
        ds_result = self.ds[self.ds['col with space'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEmptyResultHandling:
    """Test handling of empty results."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.ds = DataStore(self.df)

    def test_filter_returns_empty(self):
        """Filter that returns no rows."""
        # pandas
        pd_result = self.df[self.df['a'] > 100]

        # DataStore
        ds_result = self.ds[self.ds['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """head(0) should return empty DataFrame with columns."""
        # pandas
        pd_result = self.df.head(0)

        # DataStore
        ds_result = self.ds.head(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_then_operations(self):
        """Operations on empty result should not error."""
        # pandas
        pd_empty = self.df[self.df['a'] > 100]
        pd_result = pd_empty.sort_values('a').head(5)

        # DataStore
        ds_empty = self.ds[self.ds['a'] > 100]
        ds_result = ds_empty.sort_values('a').head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullHandlingInChains:
    """Test NULL/NaN handling in operation chains."""

    def setup_method(self):
        """Create test data with nulls."""
        self.df = pd.DataFrame(
            {'a': [1, None, 3, None, 5], 'b': [10, 20, None, 40, 50], 'c': ['x', None, 'z', None, 'w']}
        )
        self.ds = DataStore(self.df)

    def test_filter_with_null(self):
        """Filter on column with nulls."""
        # pandas
        pd_result = self.df[self.df['a'] > 2]

        # DataStore
        ds_result = self.ds[self.ds['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_filter(self):
        """dropna then filter."""
        # pandas
        pd_result = self.df.dropna(subset=['a'])[self.df.dropna(subset=['a'])['a'] > 2]

        # DataStore
        ds_clean = self.ds.dropna(subset=['a'])
        ds_result = ds_clean[ds_clean['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_then_filter(self):
        """fillna then filter."""
        # pandas
        pd_filled = self.df.fillna({'a': 0})
        pd_result = pd_filled[pd_filled['a'] > 0]

        # DataStore
        ds_filled = self.ds.fillna({'a': 0})
        ds_result = ds_filled[ds_filled['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
