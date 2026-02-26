"""
Exploratory Batch 38: Accessor + Chain + Aggregation Combinations

Focus areas:
1. String accessor operations in chains (str.upper + filter + groupby)
2. Datetime accessor operations with groupby and aggregation
3. Cumulative operations (cumsum, cummax) in groupby context
4. Multi-column aggregation with dict of functions
5. Select + accessor + computation chains
6. Multiple accessor uses in single pipeline
7. Accessor results as groupby keys

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
Discovery date: 2026-01-06
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import chdb_empty_df_str_dtype


# =============================================================================
# Test Group 1: String Accessor Chains
# =============================================================================


class TestStringAccessorChains:
    """Test string accessor operations in chain scenarios."""

    def test_str_upper_then_filter(self):
        """Test str.upper() followed by filter on different column."""
        df = pd.DataFrame({
            'name': ['alice', 'bob', 'charlie', 'david'],
            'score': [85, 92, 78, 95]
        })

        # pandas
        pd_df = df.copy()
        pd_df['name_upper'] = pd_df['name'].str.upper()
        pd_result = pd_df[pd_df['score'] > 80]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['name_upper'] = ds_df['name'].str.upper()
        ds_result = ds_df[ds_df['score'] > 80]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_filter_chain(self):
        """Test str.contains() as filter condition."""
        df = pd.DataFrame({
            'text': ['hello world', 'hi there', 'hello again', 'goodbye'],
            'value': [1, 2, 3, 4]
        })

        # pandas
        pd_df = df.copy()
        pd_result = pd_df[pd_df['text'].str.contains('hello')]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df[ds_df['text'].str.contains('hello')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_then_groupby(self):
        """Test str.len() followed by groupby."""
        df = pd.DataFrame({
            'name': ['a', 'bb', 'ccc', 'dd', 'eee'],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['name_len'] = pd_df['name'].str.len()
        pd_result = pd_df.groupby('category')['name_len'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['name_len'] = ds_df['name'].str.len()
        ds_result = ds_df.groupby('category')['name_len'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_str_slice_then_filter_then_sort(self):
        """Test str slicing followed by filter and sort."""
        df = pd.DataFrame({
            'code': ['ABC123', 'DEF456', 'ABC789', 'GHI012'],
            'amount': [100, 200, 150, 300]
        })

        # pandas
        pd_df = df.copy()
        pd_df['prefix'] = pd_df['code'].str[:3]
        pd_result = pd_df[pd_df['prefix'] == 'ABC'].sort_values('amount')

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prefix'] = ds_df['code'].str[:3]
        ds_result = ds_df[ds_df['prefix'] == 'ABC'].sort_values('amount')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_lower_groupby_agg(self):
        """Test str.lower() followed by groupby aggregation."""
        df = pd.DataFrame({
            'Product': ['Apple', 'APPLE', 'Banana', 'BANANA', 'apple'],
            'Sales': [100, 200, 150, 250, 300]
        })

        # pandas
        pd_df = df.copy()
        pd_df['product_lower'] = pd_df['Product'].str.lower()
        pd_result = pd_df.groupby('product_lower')['Sales'].agg(['sum', 'mean'])

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['product_lower'] = ds_df['Product'].str.lower()
        ds_result = ds_df.groupby('product_lower')['Sales'].agg(['sum', 'mean'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_str_ops_in_chain(self):
        """Test multiple string operations in sequence."""
        df = pd.DataFrame({
            'text': ['  Hello World  ', '  Python Programming  ', '  Data Science  '],
            'id': [1, 2, 3]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cleaned'] = pd_df['text'].str.strip().str.lower()
        pd_result = pd_df[['id', 'cleaned']]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cleaned'] = ds_df['text'].str.strip().str.lower()
        ds_result = ds_df[['id', 'cleaned']]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 2: Datetime Accessor Chains
# =============================================================================


class TestDatetimeAccessorChains:
    """Test datetime accessor operations in chain scenarios."""

    def test_dt_year_groupby_sum(self):
        """Test dt.year as groupby key for aggregation."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-15', '2023-06-20', '2024-02-10', '2024-08-25']),
            'sales': [100, 200, 150, 250]
        })

        # pandas
        pd_df = df.copy()
        pd_df['year'] = pd_df['date'].dt.year
        pd_result = pd_df.groupby('year')['sales'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['year'] = ds_df['date'].dt.year
        ds_result = ds_df.groupby('year')['sales'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dt_month_year_multi_groupby(self):
        """Test dt.year and dt.month as multi-level groupby keys."""
        df = pd.DataFrame({
            'date': pd.to_datetime([
                '2023-01-10', '2023-01-20', '2023-02-15',
                '2024-01-05', '2024-02-28'
            ]),
            'revenue': [1000, 1500, 2000, 1200, 1800]
        })

        # pandas
        pd_df = df.copy()
        pd_df['year'] = pd_df['date'].dt.year
        pd_df['month'] = pd_df['date'].dt.month
        pd_result = pd_df.groupby(['year', 'month'])['revenue'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['year'] = ds_df['date'].dt.year
        ds_df['month'] = ds_df['date'].dt.month
        ds_result = ds_df.groupby(['year', 'month'])['revenue'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dt_dayofweek_filter_then_agg(self):
        """Test dt.dayofweek filter followed by aggregation."""
        df = pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-01',  # Monday
                '2024-01-02',  # Tuesday
                '2024-01-06',  # Saturday
                '2024-01-07',  # Sunday
                '2024-01-08',  # Monday
            ]),
            'sales': [100, 200, 500, 600, 150]
        })

        # pandas
        pd_df = df.copy()
        pd_df['dow'] = pd_df['date'].dt.dayofweek
        # Filter weekdays (Monday=0 to Friday=4)
        pd_result = pd_df[pd_df['dow'] < 5]['sales'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['dow'] = ds_df['date'].dt.dayofweek
        ds_result = ds_df[ds_df['dow'] < 5]['sales'].sum()

        # Both should return scalar
        assert float(ds_result) == float(pd_result)

    def test_dt_quarter_groupby_multiple_aggs(self):
        """Test dt.quarter groupby with multiple aggregations."""
        df = pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-15', '2024-03-20', '2024-04-10',
                '2024-07-05', '2024-10-15', '2024-12-01'
            ]),
            'amount': [100, 200, 300, 400, 500, 600]
        })

        # pandas
        pd_df = df.copy()
        pd_df['quarter'] = pd_df['date'].dt.quarter
        pd_result = pd_df.groupby('quarter')['amount'].agg(['sum', 'mean', 'count'])

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['quarter'] = ds_df['date'].dt.quarter
        ds_result = ds_df.groupby('quarter')['amount'].agg(['sum', 'mean', 'count'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dt_hour_groupby_for_hourly_analysis(self):
        """Test dt.hour for hourly pattern analysis."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-01 09:30:00',
                '2024-01-01 10:15:00',
                '2024-01-01 10:45:00',
                '2024-01-01 14:00:00',
                '2024-01-01 14:30:00',
            ]),
            'requests': [10, 20, 15, 30, 25]
        })

        # pandas
        pd_df = df.copy()
        pd_df['hour'] = pd_df['timestamp'].dt.hour
        pd_result = pd_df.groupby('hour')['requests'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['hour'] = ds_df['timestamp'].dt.hour
        ds_result = ds_df.groupby('hour')['requests'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 3: Cumulative Operations
# =============================================================================


class TestCumulativeOperations:
    """Test cumulative operations in various scenarios."""

    def test_cumsum_simple(self):
        """Test simple cumulative sum."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cumsum'] = pd_df['value'].cumsum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumsum'] = ds_df['value'].cumsum()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_cumsum_with_filter(self):
        """Test cumsum after filter."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_filtered = pd_df[pd_df['category'] == 'A'].copy()
        pd_filtered['cumsum'] = pd_filtered['value'].cumsum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_filtered = ds_df[ds_df['category'] == 'A']
        ds_filtered['cumsum'] = ds_filtered['value'].cumsum()

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_cummax_simple(self):
        """Test simple cumulative max."""
        df = pd.DataFrame({
            'value': [3, 1, 4, 1, 5, 9, 2]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cummax'] = pd_df['value'].cummax()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cummax'] = ds_df['value'].cummax()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_cummin_simple(self):
        """Test simple cumulative min."""
        df = pd.DataFrame({
            'value': [5, 3, 8, 2, 9, 1, 7]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cummin'] = pd_df['value'].cummin()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cummin'] = ds_df['value'].cummin()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_cumprod_simple(self):
        """Test simple cumulative product."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cumprod'] = pd_df['value'].cumprod()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumprod'] = ds_df['value'].cumprod()

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Test Group 4: Multi-Column Aggregation with Dict
# =============================================================================


class TestMultiColumnAggregation:
    """Test multi-column aggregation with dict of functions."""

    def test_groupby_agg_dict_single_func_per_col(self):
        """Test groupby agg with dict specifying single function per column."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value1': [10, 20, 30, 40, 50],
            'value2': [100, 200, 300, 400, 500]
        })

        # pandas
        pd_result = df.groupby('category').agg({
            'value1': 'sum',
            'value2': 'mean'
        })

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df.groupby('category').agg({
            'value1': 'sum',
            'value2': 'mean'
        })

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_dict_multiple_funcs(self):
        """Test groupby agg with dict specifying multiple functions per column."""
        df = pd.DataFrame({
            'group': ['X', 'X', 'Y', 'Y'],
            'amount': [100, 200, 300, 400]
        })

        # pandas
        pd_result = df.groupby('group').agg({
            'amount': ['sum', 'mean', 'max', 'min']
        })

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df.groupby('group').agg({
            'amount': ['sum', 'mean', 'max', 'min']
        })

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_named_aggregation(self):
        """Test groupby with named aggregation syntax."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })

        # pandas
        pd_result = df.groupby('category').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df.groupby('category').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_agg_without_groupby(self):
        """Test DataFrame.agg() without groupby."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })

        # pandas
        pd_result = df.agg({
            'a': 'sum',
            'b': 'mean'
        })

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df.agg({
            'a': 'sum',
            'b': 'mean'
        })

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 5: Combined Accessor + Aggregation Chains
# =============================================================================


class TestCombinedAccessorAggChains:
    """Test combined accessor operations with aggregation."""

    def test_str_extract_groupby_agg(self):
        """Test string extraction followed by groupby aggregation."""
        df = pd.DataFrame({
            'product_code': ['A-001', 'A-002', 'B-001', 'B-002', 'A-003'],
            'quantity': [10, 20, 15, 25, 30]
        })

        # pandas
        pd_df = df.copy()
        pd_df['category'] = pd_df['product_code'].str.split('-').str[0]
        pd_result = pd_df.groupby('category')['quantity'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['category'] = ds_df['product_code'].str.split('-').str[0]
        ds_result = ds_df.groupby('category')['quantity'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dt_accessor_then_str_accessor(self):
        """Test datetime accessor followed by string operations on another column."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-02-20', '2024-01-25']),
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 150]
        })

        # pandas
        pd_df = df.copy()
        pd_df['month'] = pd_df['date'].dt.month
        pd_df['name_upper'] = pd_df['name'].str.upper()
        pd_result = pd_df[['month', 'name_upper', 'value']]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['month'] = ds_df['date'].dt.month
        ds_df['name_upper'] = ds_df['name'].str.upper()
        ds_result = ds_df[['month', 'name_upper', 'value']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_str_groupby_dt_agg(self):
        """Test filter by string, then groupby datetime field and aggregate."""
        df = pd.DataFrame({
            'date': pd.to_datetime([
                '2024-01-10', '2024-01-20', '2024-02-15',
                '2024-01-05', '2024-02-25'
            ]),
            'category': ['electronics', 'clothing', 'electronics', 'food', 'electronics'],
            'sales': [500, 300, 600, 200, 400]
        })

        # pandas
        pd_df = df.copy()
        pd_filtered = pd_df[pd_df['category'].str.contains('electro')]
        pd_filtered = pd_filtered.copy()
        pd_filtered['month'] = pd_filtered['date'].dt.month
        pd_result = pd_filtered.groupby('month')['sales'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_filtered = ds_df[ds_df['category'].str.contains('electro')]
        ds_filtered['month'] = ds_filtered['date'].dt.month
        ds_result = ds_filtered.groupby('month')['sales'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 6: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    # FIXED: Empty DataFrame str accessor now returns correct dtype
    def test_empty_df_str_accessor(self):
        """Test string accessor on empty DataFrame."""
        df = pd.DataFrame({
            'name': pd.Series([], dtype='str'),
            'value': pd.Series([], dtype='int64')
        })

        # pandas
        pd_df = df.copy()
        pd_df['name_upper'] = pd_df['name'].str.upper()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['name_upper'] = ds_df['name'].str.upper()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_single_row_cumsum(self):
        """Test cumsum on single row DataFrame."""
        df = pd.DataFrame({
            'value': [42]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cumsum'] = pd_df['value'].cumsum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumsum'] = ds_df['value'].cumsum()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_str_accessor_with_na_values(self):
        """Test string accessor with NA values."""
        df = pd.DataFrame({
            'text': ['hello', None, 'world', None, 'test'],
            'id': [1, 2, 3, 4, 5]
        })

        # pandas
        pd_df = df.copy()
        pd_df['text_upper'] = pd_df['text'].str.upper()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['text_upper'] = ds_df['text'].str.upper()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_dt_accessor_chain_with_filter(self):
        """Test datetime accessor with filter on extracted field."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-15 10:30:00',
                '2024-03-20 14:00:00',
                '2024-06-10 09:15:00',
                '2024-12-25 18:45:00'
            ]),
            'value': [100, 200, 300, 400]
        })

        # pandas - filter first half of year
        pd_df = df.copy()
        pd_df['month'] = pd_df['timestamp'].dt.month
        pd_result = pd_df[pd_df['month'] <= 6]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['month'] = ds_df['timestamp'].dt.month
        ds_result = ds_df[ds_df['month'] <= 6]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_groupby_agg_chains(self):
        """Test multiple groupby-agg operations in sequence."""
        df = pd.DataFrame({
            'region': ['East', 'East', 'West', 'West', 'East'],
            'product': ['A', 'B', 'A', 'B', 'A'],
            'sales': [100, 200, 150, 250, 300]
        })

        # pandas - first groupby
        pd_df = df.copy()
        pd_grouped = pd_df.groupby('region')['sales'].sum().reset_index()
        pd_result = pd_grouped.sort_values('sales', ascending=False)

        # DataStore
        ds_df = DataStore(df.copy())
        ds_grouped = ds_df.groupby('region')['sales'].sum().reset_index()
        ds_result = ds_grouped.sort_values('sales', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 7: Rank Operations
# =============================================================================


class TestRankOperations:
    """Test rank operations in various scenarios."""

    def test_rank_simple(self):
        """Test simple rank operation."""
        df = pd.DataFrame({
            'value': [3, 1, 4, 1, 5]
        })

        # pandas
        pd_df = df.copy()
        pd_df['rank'] = pd_df['value'].rank()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['rank'] = ds_df['value'].rank()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_rank_method_min(self):
        """Test rank with method='min' for ties."""
        df = pd.DataFrame({
            'score': [90, 85, 90, 80, 85]
        })

        # pandas
        pd_df = df.copy()
        pd_df['rank'] = pd_df['score'].rank(method='min')

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['rank'] = ds_df['score'].rank(method='min')

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_rank_ascending_false(self):
        """Test rank with ascending=False."""
        df = pd.DataFrame({
            'value': [10, 30, 20, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['rank'] = pd_df['value'].rank(ascending=False)

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['rank'] = ds_df['value'].rank(ascending=False)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_rank_with_filter(self):
        """Test rank after filtering."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'score': [85, 90, 95, 80, 88]
        })

        # pandas
        pd_df = df.copy()
        pd_filtered = pd_df[pd_df['category'] == 'A'].copy()
        pd_filtered['rank'] = pd_filtered['score'].rank()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_filtered = ds_df[ds_df['category'] == 'A']
        ds_filtered['rank'] = ds_filtered['score'].rank()

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)


# =============================================================================
# Test Group 8: Complex Chain Scenarios
# =============================================================================


class TestComplexChainScenarios:
    """Test complex multi-step chains combining various operations."""

    def test_str_dt_filter_groupby_agg_sort_head(self):
        """Test complex chain: str processing + dt extraction + filter + groupby + agg + sort + head."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-01-10 09:00:00', '2024-01-10 14:00:00',
                '2024-01-11 10:00:00', '2024-01-11 15:00:00',
                '2024-02-05 11:00:00', '2024-02-05 16:00:00',
            ]),
            'product': ['Widget A', 'Widget B', 'Widget A', 'Gadget X', 'Widget A', 'Gadget Y'],
            'amount': [100, 200, 150, 300, 250, 350]
        })

        # pandas
        pd_df = df.copy()
        pd_df['month'] = pd_df['timestamp'].dt.month
        pd_df['product_type'] = pd_df['product'].str.split().str[0]
        pd_filtered = pd_df[pd_df['product_type'] == 'Widget']
        pd_result = (
            pd_filtered
            .groupby('month')['amount']
            .sum()
            .reset_index()
            .sort_values('amount', ascending=False)
            .head(2)
        )

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['month'] = ds_df['timestamp'].dt.month
        ds_df['product_type'] = ds_df['product'].str.split().str[0]
        ds_filtered = ds_df[ds_df['product_type'] == 'Widget']
        ds_result = (
            ds_filtered
            .groupby('month')['amount']
            .sum()
            .reset_index()
            .sort_values('amount', ascending=False)
            .head(2)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_rank_in_same_df(self):
        """Test adding both cumsum and rank columns."""
        df = pd.DataFrame({
            'value': [10, 30, 20, 50, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cumsum'] = pd_df['value'].cumsum()
        pd_df['rank'] = pd_df['value'].rank()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumsum'] = ds_df['value'].cumsum()
        ds_df['rank'] = ds_df['value'].rank()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_accessor_result_in_computation(self):
        """Test using accessor result in arithmetic computation."""
        df = pd.DataFrame({
            'text': ['hello', 'world', 'hi'],
            'multiplier': [10, 20, 30]
        })

        # pandas
        pd_df = df.copy()
        pd_df['score'] = pd_df['text'].str.len() * pd_df['multiplier']

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['score'] = ds_df['text'].str.len() * ds_df['multiplier']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_dt_accessor_in_filter_expression(self):
        """Test datetime accessor directly in filter expression."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-06-20', '2024-12-25']),
            'value': [100, 200, 300]
        })

        # pandas
        pd_df = df.copy()
        pd_result = pd_df[pd_df['date'].dt.month > 6]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df[ds_df['date'].dt.month > 6]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_accessor_in_filter_expression(self):
        """Test string accessor directly in filter expression."""
        df = pd.DataFrame({
            'name': ['Alice Smith', 'Bob Jones', 'Charlie Brown'],
            'age': [25, 30, 35]
        })

        # pandas
        pd_df = df.copy()
        pd_result = pd_df[pd_df['name'].str.len() > 10]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_result = ds_df[ds_df['name'].str.len() > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)
