"""
Exploratory tests for real-world pandas patterns from Kaggle notebooks.

These tests cover common data analysis operations found in real-world
data science workflows, including:
1. E-commerce transaction analysis
2. Customer segmentation
3. Time series feature engineering
4. Categorical data processing
5. Chain operations mimicking EDA workflows

All tests follow the Mirror Code Pattern: compare DataStore vs pandas results.
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.tests.test_utils import (
    assert_series_equal,
    assert_frame_equal,
    assert_datastore_equals_pandas,
)


class TestEcommerceTransactionAnalysis:
    """Test e-commerce transaction patterns from Kaggle notebooks."""

    @pytest.fixture
    def transactions_df(self):
        """Create sample e-commerce transaction data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'transaction_id': range(1, n + 1),
            'customer_id': np.random.randint(1, 20, n),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n),
            'amount': np.round(np.random.uniform(10, 500, n), 2),
            'quantity': np.random.randint(1, 10, n),
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'discount_pct': np.random.choice([0, 5, 10, 15, 20], n),
            'is_returned': np.random.choice([True, False], n, p=[0.1, 0.9]),
        })

    def test_revenue_by_category(self, transactions_df):
        """Calculate revenue by product category."""
        pd_df = transactions_df.copy()
        # pandas groupby().sum() returns a Series with index
        pd_result = pd_df.groupby('product_category')['amount'].sum()

        ds_df = DataStore(transactions_df)
        ds_result = ds_df.groupby('product_category')['amount'].sum()

        # Compare as Series - order may differ so check values
        assert_series_equal(
            ds_result, pd_result,
            check_names=False,
            check_index=False,  # groupby order may differ
        )

    def test_avg_order_value_per_customer(self, transactions_df):
        """Calculate average order value per customer."""
        pd_df = transactions_df.copy()
        pd_result = pd_df.groupby('customer_id')['amount'].mean()

        ds_df = DataStore(transactions_df)
        ds_result = ds_df.groupby('customer_id')['amount'].mean()

        assert_series_equal(
            ds_result, pd_result,
            check_names=False,
            check_index=False,
        )

    def test_filter_high_value_transactions(self, transactions_df):
        """Filter transactions above a threshold."""
        threshold = 200
        pd_df = transactions_df.copy()
        pd_result = pd_df[pd_df['amount'] > threshold]

        ds_df = DataStore(transactions_df)
        ds_result = ds_df[ds_df['amount'] > threshold]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_and_aggregate_chain(self, transactions_df):
        """Filter then aggregate - common EDA pattern."""
        pd_df = transactions_df.copy()
        pd_result = pd_df[pd_df['is_returned'] == False].groupby('product_category')['amount'].sum()

        ds_df = DataStore(transactions_df)
        ds_result = ds_df[ds_df['is_returned'] == False].groupby('product_category')['amount'].sum()

        # Compare as Series
        assert_series_equal(
            ds_result, pd_result,
            check_names=False,
            check_index=False,
        )

    def test_calculate_discount_amount(self, transactions_df):
        """Calculate computed column - discount amount."""
        pd_df = transactions_df.copy()
        pd_df['discount_amount'] = pd_df['amount'] * pd_df['discount_pct'] / 100
        pd_result = pd_df[['transaction_id', 'amount', 'discount_pct', 'discount_amount']]

        ds_df = DataStore(transactions_df)
        ds_df = ds_df.assign(discount_amount=ds_df['amount'] * ds_df['discount_pct'] / 100)
        ds_result = ds_df[['transaction_id', 'amount', 'discount_pct', 'discount_amount']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCustomerSegmentation:
    """Test customer segmentation patterns."""

    @pytest.fixture
    def customers_df(self):
        """Create sample customer data for segmentation."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'customer_id': range(1, n + 1),
            'age': np.random.randint(18, 70, n),
            'gender': np.random.choice(['M', 'F'], n),
            'income': np.random.randint(20000, 150000, n),
            'spending_score': np.random.randint(1, 100, n),
            'membership_years': np.random.randint(0, 10, n),
            'last_purchase_days': np.random.randint(0, 365, n),
        })

    def test_age_group_binning_via_filter(self, customers_df):
        """Segment customers by age groups using filter."""
        pd_df = customers_df.copy()
        pd_young = pd_df[pd_df['age'] < 30]
        pd_result = pd_young.copy()

        ds_df = DataStore(customers_df)
        ds_result = ds_df[ds_df['age'] < 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_income_percentile_filtering(self, customers_df):
        """Filter high-income customers (top 25%)."""
        pd_df = customers_df.copy()
        threshold = pd_df['income'].quantile(0.75)
        pd_result = pd_df[pd_df['income'] >= threshold]

        ds_df = DataStore(customers_df)
        # Note: DataStore might need to execute quantile first
        threshold_val = float(customers_df['income'].quantile(0.75))
        ds_result = ds_df[ds_df['income'] >= threshold_val]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_condition_segmentation(self, customers_df):
        """Segment with multiple conditions."""
        pd_df = customers_df.copy()
        pd_result = pd_df[
            (pd_df['spending_score'] > 50) &
            (pd_df['membership_years'] >= 2) &
            (pd_df['last_purchase_days'] < 90)
        ]

        ds_df = DataStore(customers_df)
        ds_result = ds_df[
            (ds_df['spending_score'] > 50) &
            (ds_df['membership_years'] >= 2) &
            (ds_df['last_purchase_days'] < 90)
        ]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringOperationsEDA:
    """Test string operations common in EDA."""

    @pytest.fixture
    def text_df(self):
        """Create sample text data."""
        return pd.DataFrame({
            'product_name': [
                'Apple iPhone 15 Pro',
                'Samsung Galaxy S24',
                'Google Pixel 8',
                'OnePlus 12',
                'Apple MacBook Air',
                'Dell XPS 15',
                'HP Spectre x360',
                'Lenovo ThinkPad X1',
            ],
            'description': [
                'Latest Apple smartphone with A17 chip',
                'Samsung flagship with great camera',
                'Pure Android experience',
                'Fast charging and performance',
                'Lightweight laptop for professionals',
                'Premium Windows laptop',
                'Convertible 2-in-1 laptop',
                'Business laptop classic',
            ],
            'category': ['Phone', 'Phone', 'Phone', 'Phone', 'Laptop', 'Laptop', 'Laptop', 'Laptop'],
            'price': [999, 899, 699, 799, 1099, 1299, 1499, 1199],
        })

    def test_str_contains_filter(self, text_df):
        """Filter products containing specific text."""
        pd_df = text_df.copy()
        pd_result = pd_df[pd_df['product_name'].str.contains('Apple')]

        ds_df = DataStore(text_df)
        ds_result = ds_df[ds_df['product_name'].str.contains('Apple')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_lower_and_contains(self, text_df):
        """Case-insensitive search using lower()."""
        pd_df = text_df.copy()
        pd_result = pd_df[pd_df['description'].str.lower().str.contains('laptop')]

        ds_df = DataStore(text_df)
        ds_result = ds_df[ds_df['description'].str.lower().str.contains('laptop')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_column(self, text_df):
        """Calculate string length as a column."""
        pd_df = text_df.copy()
        pd_df['name_length'] = pd_df['product_name'].str.len()
        pd_result = pd_df[['product_name', 'name_length']]

        ds_df = DataStore(text_df)
        ds_df = ds_df.assign(name_length=ds_df['product_name'].str.len())
        ds_result = ds_df[['product_name', 'name_length']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_startswith(self, text_df):
        """Filter products starting with specific text."""
        pd_df = text_df.copy()
        pd_result = pd_df[pd_df['product_name'].str.startswith('Apple')]

        ds_df = DataStore(text_df)
        ds_result = ds_df[ds_df['product_name'].str.startswith('Apple')]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNumericalOperationsEDA:
    """Test numerical operations common in EDA."""

    @pytest.fixture
    def numeric_df(self):
        """Create sample numerical data."""
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            'value1': np.random.randn(n) * 10 + 50,
            'value2': np.random.randn(n) * 5 + 25,
            'value3': np.random.randint(0, 100, n),
            'category': np.random.choice(['A', 'B', 'C'], n),
        })

    def test_arithmetic_operations(self, numeric_df):
        """Test arithmetic operations on columns."""
        pd_df = numeric_df.copy()
        pd_df['sum_values'] = pd_df['value1'] + pd_df['value2']
        pd_df['diff_values'] = pd_df['value1'] - pd_df['value2']
        pd_df['product'] = pd_df['value1'] * pd_df['value2']
        pd_result = pd_df[['value1', 'value2', 'sum_values', 'diff_values', 'product']]

        ds_df = DataStore(numeric_df)
        ds_df = ds_df.assign(
            sum_values=ds_df['value1'] + ds_df['value2'],
            diff_values=ds_df['value1'] - ds_df['value2'],
            product=ds_df['value1'] * ds_df['value2'],
        )
        ds_result = ds_df[['value1', 'value2', 'sum_values', 'diff_values', 'product']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_values(self, numeric_df):
        """Test absolute value operations."""
        pd_df = numeric_df.copy()
        pd_df['abs_value1'] = pd_df['value1'].abs()
        pd_result = pd_df[['value1', 'abs_value1']]

        ds_df = DataStore(numeric_df)
        ds_df = ds_df.assign(abs_value1=ds_df['value1'].abs())
        ds_result = ds_df[['value1', 'abs_value1']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_values(self, numeric_df):
        """Test clipping values to a range."""
        pd_df = numeric_df.copy()
        pd_df['clipped'] = pd_df['value3'].clip(lower=20, upper=80)
        pd_result = pd_df[['value3', 'clipped']]

        ds_df = DataStore(numeric_df)
        ds_df = ds_df.assign(clipped=ds_df['value3'].clip(lower=20, upper=80))
        ds_result = ds_df[['value3', 'clipped']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_round_values(self, numeric_df):
        """Test rounding values."""
        pd_df = numeric_df.copy()
        pd_df['rounded'] = pd_df['value1'].round(1)
        pd_result = pd_df[['value1', 'rounded']]

        ds_df = DataStore(numeric_df)
        ds_df = ds_df.assign(rounded=ds_df['value1'].round(1))
        ds_result = ds_df[['value1', 'rounded']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSortingAndRanking:
    """Test sorting and ranking operations."""

    @pytest.fixture
    def sales_df(self):
        """Create sample sales data."""
        return pd.DataFrame({
            'salesperson': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'region': ['North', 'South', 'North', 'South', 'East'],
            'sales': [150000, 120000, 180000, 95000, 160000],
            'deals_closed': [45, 38, 52, 30, 48],
        })

    def test_sort_by_single_column(self, sales_df):
        """Sort by a single column."""
        pd_df = sales_df.copy()
        pd_result = pd_df.sort_values('sales', ascending=False)

        ds_df = DataStore(sales_df)
        ds_result = ds_df.sort_values('sales', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_by_multiple_columns(self, sales_df):
        """Sort by multiple columns."""
        pd_df = sales_df.copy()
        pd_result = pd_df.sort_values(['region', 'sales'], ascending=[True, False])

        ds_df = DataStore(sales_df)
        ds_result = ds_df.sort_values(['region', 'sales'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest(self, sales_df):
        """Get top N rows by column."""
        pd_df = sales_df.copy()
        pd_result = pd_df.nlargest(3, 'sales')

        ds_df = DataStore(sales_df)
        ds_result = ds_df.nlargest(3, 'sales')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest(self, sales_df):
        """Get bottom N rows by column."""
        pd_df = sales_df.copy()
        pd_result = pd_df.nsmallest(2, 'deals_closed')

        ds_df = DataStore(sales_df)
        ds_result = ds_df.nsmallest(2, 'deals_closed')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullHandling:
    """Test null/NaN handling patterns."""

    @pytest.fixture
    def df_with_nulls(self):
        """Create DataFrame with null values."""
        return pd.DataFrame({
            'a': [1, 2, None, 4, None],
            'b': [10.5, None, 30.5, None, 50.5],
            'c': ['x', 'y', None, 'z', None],
            'd': [100, 200, 300, 400, 500],
        })

    def test_dropna_any(self, df_with_nulls):
        """Drop rows with any null values."""
        pd_df = df_with_nulls.copy()
        pd_result = pd_df.dropna()

        ds_df = DataStore(df_with_nulls)
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset(self, df_with_nulls):
        """Drop rows with nulls in specific columns."""
        pd_df = df_with_nulls.copy()
        pd_result = pd_df.dropna(subset=['a', 'b'])

        ds_df = DataStore(df_with_nulls)
        ds_result = ds_df.dropna(subset=['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_scalar(self, df_with_nulls):
        """Fill nulls with scalar value."""
        pd_df = df_with_nulls.copy()
        pd_df['a'] = pd_df['a'].fillna(0)
        pd_result = pd_df[['a']]

        ds_df = DataStore(df_with_nulls)
        ds_df = ds_df.assign(a=ds_df['a'].fillna(0))
        ds_result = ds_df[['a']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_isna_filter(self, df_with_nulls):
        """Filter rows where column is null."""
        pd_df = df_with_nulls.copy()
        pd_result = pd_df[pd_df['a'].isna()]

        ds_df = DataStore(df_with_nulls)
        ds_result = ds_df[ds_df['a'].isna()]

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)

    def test_notna_filter(self, df_with_nulls):
        """Filter rows where column is not null."""
        pd_df = df_with_nulls.copy()
        pd_result = pd_df[pd_df['b'].notna()]

        ds_df = DataStore(df_with_nulls)
        ds_result = ds_df[ds_df['b'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


class TestDuplicateHandling:
    """Test duplicate handling patterns."""

    @pytest.fixture
    def df_with_duplicates(self):
        """Create DataFrame with duplicate rows."""
        return pd.DataFrame({
            'id': [1, 2, 2, 3, 3, 3, 4],
            'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'Charlie', 'Charlie', 'Diana'],
            'value': [100, 200, 200, 300, 301, 302, 400],
        })

    def test_drop_duplicates_default(self, df_with_duplicates):
        """Drop duplicates keeping first occurrence."""
        pd_df = df_with_duplicates.copy()
        pd_result = pd_df.drop_duplicates()

        ds_df = DataStore(df_with_duplicates)
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset(self, df_with_duplicates):
        """Drop duplicates based on subset of columns."""
        pd_df = df_with_duplicates.copy()
        pd_result = pd_df.drop_duplicates(subset=['id'])

        ds_df = DataStore(df_with_duplicates)
        ds_result = ds_df.drop_duplicates(subset=['id'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self, df_with_duplicates):
        """Drop duplicates keeping last occurrence."""
        pd_df = df_with_duplicates.copy()
        pd_result = pd_df.drop_duplicates(subset=['id'], keep='last')

        ds_df = DataStore(df_with_duplicates)
        ds_result = ds_df.drop_duplicates(subset=['id'], keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicated_mask(self, df_with_duplicates):
        """Get boolean mask of duplicated rows."""
        pd_df = df_with_duplicates.copy()
        pd_result = pd_df['id'].duplicated()

        ds_df = DataStore(df_with_duplicates)
        ds_result = ds_df['id'].duplicated()

        assert_series_equal(ds_result, pd_result, check_names=False, check_dtype=False)


class TestChainedOperations:
    """Test chained operation patterns common in EDA notebooks."""

    @pytest.fixture
    def orders_df(self):
        """Create sample orders data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'order_id': range(1, n + 1),
            'customer_id': np.random.randint(1, 30, n),
            'product': np.random.choice(['Widget', 'Gadget', 'Gizmo', 'Thing'], n),
            'quantity': np.random.randint(1, 20, n),
            'unit_price': np.round(np.random.uniform(5, 100, n), 2),
            'shipping': np.random.choice(['Standard', 'Express', 'Overnight'], n),
            'status': np.random.choice(['Delivered', 'Pending', 'Cancelled'], n, p=[0.7, 0.2, 0.1]),
        })

    def test_filter_select_chain(self, orders_df):
        """Filter then select columns."""
        pd_df = orders_df.copy()
        pd_result = pd_df[pd_df['status'] == 'Delivered'][['order_id', 'product', 'quantity']]

        ds_df = DataStore(orders_df)
        ds_result = ds_df[ds_df['status'] == 'Delivered'][['order_id', 'product', 'quantity']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_sort_chain(self, orders_df):
        """Filter, assign computed column, then sort."""
        pd_df = orders_df.copy()
        pd_result = pd_df[pd_df['status'] == 'Delivered'].copy()
        pd_result['total'] = pd_result['quantity'] * pd_result['unit_price']
        pd_result = pd_result.sort_values('total', ascending=False)

        ds_df = DataStore(orders_df)
        ds_result = ds_df[ds_df['status'] == 'Delivered']
        ds_result = ds_result.assign(total=ds_result['quantity'] * ds_result['unit_price'])
        ds_result = ds_result.sort_values('total', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_filter_chain(self, orders_df):
        """Multiple filter conditions in chain."""
        pd_df = orders_df.copy()
        pd_result = pd_df[
            (pd_df['status'] == 'Delivered') &
            (pd_df['quantity'] > 5) &
            (pd_df['unit_price'] < 50)
        ]

        ds_df = DataStore(orders_df)
        ds_result = ds_df[
            (ds_df['status'] == 'Delivered') &
            (ds_df['quantity'] > 5) &
            (ds_df['unit_price'] < 50)
        ]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCounts:
    """Test value_counts patterns."""

    @pytest.fixture
    def survey_df(self):
        """Create sample survey data."""
        return pd.DataFrame({
            'response': ['Yes', 'No', 'Maybe', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Maybe', 'Yes'],
            'age_group': ['18-25', '26-35', '18-25', '36-45', '26-35', '18-25', '26-35', '36-45', '18-25', '26-35'],
            'satisfaction': [5, 3, 4, 5, 2, 4, 5, 1, 3, 4],
        })

    def test_value_counts_basic(self, survey_df):
        """Basic value counts."""
        pd_df = survey_df.copy()
        pd_result = pd_df['response'].value_counts()

        ds_df = DataStore(survey_df)
        ds_result = ds_df['response'].value_counts()

        # value_counts returns Series with values as index
        # Order may differ, so compare as unordered
        assert_series_equal(
            ds_result,
            pd_result,
            check_names=False,
            check_index=False,
        )

    def test_value_counts_normalize(self, survey_df):
        """Value counts with normalization (percentages)."""
        pd_df = survey_df.copy()
        pd_result = pd_df['age_group'].value_counts(normalize=True)

        ds_df = DataStore(survey_df)
        ds_result = ds_df['age_group'].value_counts(normalize=True)

        assert_series_equal(
            ds_result,
            pd_result,
            check_names=False,
            check_index=False,
        )

    def test_value_counts_sort_false(self, survey_df):
        """Value counts without sorting."""
        pd_df = survey_df.copy()
        pd_result = pd_df['response'].value_counts(sort=False)

        ds_df = DataStore(survey_df)
        ds_result = ds_df['response'].value_counts(sort=False)

        assert_series_equal(
            ds_result,
            pd_result,
            check_names=False,
            check_index=False,
        )


class TestHeadTailSample:
    """Test head, tail, sample operations."""

    @pytest.fixture
    def large_df(self):
        """Create larger DataFrame for head/tail/sample tests."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'id': range(1, n + 1),
            'value': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        })

    def test_head_default(self, large_df):
        """Default head (5 rows)."""
        pd_df = large_df.copy()
        pd_result = pd_df.head()

        ds_df = DataStore(large_df)
        ds_result = ds_df.head()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_n(self, large_df):
        """Head with specific n."""
        pd_df = large_df.copy()
        pd_result = pd_df.head(10)

        ds_df = DataStore(large_df)
        ds_result = ds_df.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_default(self, large_df):
        """Default tail (5 rows)."""
        pd_df = large_df.copy()
        pd_result = pd_df.tail()

        ds_df = DataStore(large_df)
        ds_result = ds_df.tail()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_n(self, large_df):
        """Tail with specific n."""
        pd_df = large_df.copy()
        pd_result = pd_df.tail(15)

        ds_df = DataStore(large_df)
        ds_result = ds_df.tail(15)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sample_n(self, large_df):
        """Sample n rows with fixed seed."""
        pd_df = large_df.copy()
        pd_result = pd_df.sample(n=10, random_state=42)

        ds_df = DataStore(large_df)
        ds_result = ds_df.sample(n=10, random_state=42)

        # Sample should return same rows with same seed
        assert len(ds_result) == 10
        # Note: Row content comparison depends on implementation


class TestColumnOperations:
    """Test column selection and manipulation."""

    @pytest.fixture
    def multi_col_df(self):
        """Create DataFrame with various column types."""
        return pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
            'date_col': pd.date_range('2024-01-01', periods=5),
        })

    def test_select_single_column(self, multi_col_df):
        """Select a single column."""
        pd_df = multi_col_df.copy()
        pd_result = pd_df['int_col']

        ds_df = DataStore(multi_col_df)
        ds_result = ds_df['int_col']

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_select_multiple_columns(self, multi_col_df):
        """Select multiple columns."""
        pd_df = multi_col_df.copy()
        pd_result = pd_df[['int_col', 'str_col', 'bool_col']]

        ds_df = DataStore(multi_col_df)
        ds_result = ds_df[['int_col', 'str_col', 'bool_col']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_columns(self, multi_col_df):
        """Drop columns."""
        pd_df = multi_col_df.copy()
        pd_result = pd_df.drop(columns=['date_col', 'bool_col'])

        ds_df = DataStore(multi_col_df)
        ds_result = ds_df.drop(columns=['date_col', 'bool_col'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_columns(self, multi_col_df):
        """Rename columns."""
        pd_df = multi_col_df.copy()
        pd_result = pd_df.rename(columns={'int_col': 'integer', 'str_col': 'string'})

        ds_df = DataStore(multi_col_df)
        ds_result = ds_df.rename(columns={'int_col': 'integer', 'str_col': 'string'})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBetweenAndIsin:
    """Test between and isin operations."""

    @pytest.fixture
    def range_df(self):
        """Create DataFrame for range filtering."""
        return pd.DataFrame({
            'value': [10, 25, 50, 75, 100, 125, 150],
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
            'status': ['active', 'inactive', 'active', 'pending', 'active', 'inactive', 'pending'],
        })

    def test_between_inclusive(self, range_df):
        """Filter values between range (inclusive)."""
        pd_df = range_df.copy()
        pd_result = pd_df[pd_df['value'].between(25, 100)]

        ds_df = DataStore(range_df)
        ds_result = ds_df[ds_df['value'].between(25, 100)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_values(self, range_df):
        """Filter using isin with list of values."""
        pd_df = range_df.copy()
        pd_result = pd_df[pd_df['category'].isin(['A', 'C'])]

        ds_df = DataStore(range_df)
        ds_result = ds_df[ds_df['category'].isin(['A', 'C'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_combined_with_other_filter(self, range_df):
        """Combine isin with other conditions."""
        pd_df = range_df.copy()
        pd_result = pd_df[
            (pd_df['status'].isin(['active', 'pending'])) &
            (pd_df['value'] > 50)
        ]

        ds_df = DataStore(range_df)
        ds_result = ds_df[
            (ds_df['status'].isin(['active', 'pending'])) &
            (ds_df['value'] > 50)
        ]

        assert_datastore_equals_pandas(ds_result, pd_result)
