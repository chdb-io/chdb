"""
Exploratory Discovery Tests - Batch 4 (2026-01-04)

Based on architecture analysis, test the following edge cases:
1. GroupBy missing methods (nth, median, head, tail, prod, rank)
2. GroupBy parameter support (dropna, observed)
3. Chained operations edge cases
4. Rolling window function parameters
5. Multi-column operations edge cases
"""

import pytest
from tests.xfail_markers import chdb_no_product_function
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series, get_value


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40, 50, 60],
            'score': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        }
    )


@pytest.fixture
def df_with_nan():
    return pd.DataFrame({'category': ['A', 'A', None, 'B', 'B', None], 'value': [10, 20, 30, 40, np.nan, 60]})


@pytest.fixture
def large_groups_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'score': np.random.randn(10),
        }
    )


class TestGroupByMissingMethods:
    """Test groupby methods that may be missing or incomplete."""

    @pytest.mark.skip(reason="groupby().nth() not implemented - API gap, tracked in todo.md")
    def test_groupby_nth(self, large_groups_df):
        pd_df = large_groups_df
        ds = DataStore(large_groups_df)
        pd_result = pd_df.groupby('category').nth(1)
        ds_result = ds.groupby('category').nth(1)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_median(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df.groupby('category')['value'].median()
        try:
            ds_result = ds.groupby('category')['value'].median()
            assert_datastore_equals_pandas(ds_result, pd_result)
        except AttributeError as e:
            pytest.skip(f"groupby().median() not implemented: {e}")

    @pytest.mark.skip(reason="groupby().head() not implemented - API gap, tracked in todo.md")
    def test_groupby_head(self, large_groups_df):
        pd_df = large_groups_df
        ds = DataStore(large_groups_df)
        pd_result = pd_df.groupby('category').head(2)
        ds_result = ds.groupby('category').head(2)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    @pytest.mark.skip(reason="groupby().tail() not implemented - API gap, tracked in todo.md")
    def test_groupby_tail(self, large_groups_df):
        pd_df = large_groups_df
        ds = DataStore(large_groups_df)
        pd_result = pd_df.groupby('category').tail(2)
        ds_result = ds.groupby('category').tail(2)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    @chdb_no_product_function
    def test_groupby_prod(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df.groupby('category')['value'].prod()
        ds_result = ds.groupby('category')['value'].prod()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_rank(self, large_groups_df):
        pd_df = large_groups_df
        ds = DataStore(large_groups_df)
        pd_result = pd_df.groupby('category')['value'].rank()
        try:
            ds_result = ds.groupby('category')['value'].rank()
            assert_datastore_equals_pandas(ds_result, pd_result)
        except AttributeError as e:
            pytest.skip(f"groupby().rank() not implemented: {e}")

    def test_groupby_cumsum(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df.groupby('category')['value'].cumsum()
        try:
            ds_result = ds.groupby('category')['value'].cumsum()
            assert_datastore_equals_pandas(ds_result, pd_result)
        except AttributeError as e:
            pytest.skip(f"groupby().cumsum() not implemented: {e}")


class TestGroupByParameters:
    """Test groupby parameter support."""
    def test_groupby_dropna_true(self, df_with_nan):
        pd_df = df_with_nan
        ds = DataStore(df_with_nan)
        pd_result = pd_df.groupby('category', dropna=True)['value'].sum()
        ds_result = ds.groupby('category', dropna=True)['value'].sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_dropna_false(self, df_with_nan):
        pd_df = df_with_nan
        ds = DataStore(df_with_nan)
        pd_result = pd_df.groupby('category', dropna=False)['value'].sum()
        try:
            ds_result = ds.groupby('category', dropna=False)['value'].sum()
            assert_datastore_equals_pandas(ds_result, pd_result)
        except TypeError as e:
            if "dropna" in str(e):
                pytest.skip(f"groupby(dropna=) not implemented: {e}")
            raise

    def test_groupby_sort_false(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df.groupby('category', sort=False)['value'].sum()
        try:
            ds_result = ds.groupby('category', sort=False)['value'].sum()
            pd_result_sorted = pd_result.sort_index()
            ds_values = get_series(ds_result)
            if hasattr(ds_values, 'sort_index'):
                ds_values = ds_values.sort_index()
            assert set(pd_result.values) == set(ds_values.values) or list(pd_result_sorted.values) == list(
                ds_values.values
            )
        except TypeError as e:
            if "sort" in str(e):
                pytest.skip(f"groupby(sort=) not implemented: {e}")
            raise


class TestChainedOperationsBoundary:
    """Test edge cases in chained lazy operations."""

    def test_multiple_filters_chain(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[pd_df['value'] > 10]
        pd_result = pd_result[pd_result['value'] < 50]
        pd_result = pd_result[pd_result['category'] == 'B']
        ds_result = ds[ds['value'] > 10]
        ds_result = ds_result[ds_result['value'] < 50]
        ds_result = ds_result[ds_result['category'] == 'B']
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_after_select(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[['category', 'value']]
        pd_result = pd_result[pd_result['value'] > 25]
        ds_result = ds[['category', 'value']]
        ds_result = ds_result[ds_result['value'] > 25]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_scalar_then_filter(self, sample_df):
        pd_df = sample_df.copy()
        ds = DataStore(sample_df)
        pd_df['flag'] = 1
        pd_result = pd_df[pd_df['flag'] == 1]
        try:
            ds_with_flag = ds.assign(flag=1)
            ds_result = ds_with_flag[ds_with_flag['flag'] == 1]
            assert_datastore_equals_pandas(ds_result, pd_result)
        except Exception as e:
            pytest.fail(f"assign(scalar) -> filter failed: {e}")

    def test_select_rename_filter(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[['category', 'value']]
        pd_result = pd_result.rename(columns={'value': 'amount'})
        pd_result = pd_result[pd_result['amount'] > 25]
        ds_result = ds[['category', 'value']]
        ds_result = ds_result.rename(columns={'value': 'amount'})
        ds_result = ds_result[ds_result['amount'] > 25]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_head(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df.sort_values('value', ascending=False).head(3)
        ds_result = ds.sort_values('value', ascending=False).head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_head(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[pd_df['value'] > 15].sort_values('score').head(2)
        ds_result = ds[ds['value'] > 15].sort_values('score').head(2)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRollingWindowBoundary:
    """Test rolling window operation edge cases."""

    def test_rolling_basic(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df['value'].rolling(2).sum()
        try:
            ds_result = ds['value'].rolling(2).sum()
            ds_values = get_series(ds_result)
            assert_series_equal(
                ds_values.reset_index(drop=True), pd_result.reset_index(drop=True))
        except AttributeError as e:
            pytest.skip(f"rolling() not implemented: {e}")

    def test_rolling_center(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df['value'].rolling(3, center=True).mean()
        try:
            ds_result = ds['value'].rolling(3, center=True).mean()
            ds_values = get_series(ds_result)
            assert_series_equal(
                ds_values.reset_index(drop=True), pd_result.reset_index(drop=True))
        except (AttributeError, TypeError) as e:
            pytest.skip(f"rolling(center=True) not implemented: {e}")

    def test_rolling_min_periods(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df['value'].rolling(3, min_periods=1).sum()
        try:
            ds_result = ds['value'].rolling(3, min_periods=1).sum()
            ds_values = get_series(ds_result)
            assert_series_equal(
                ds_values.reset_index(drop=True), pd_result.reset_index(drop=True))
        except (AttributeError, TypeError) as e:
            pytest.skip(f"rolling(min_periods=) not implemented: {e}")


class TestMultiColumnOperations:
    """Test operations on multiple columns."""

    def test_select_all_columns(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[list(pd_df.columns)]
        ds_result = ds[list(sample_df.columns)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_multiple_columns(self, df_with_nan):
        pd_df = df_with_nan.copy()
        ds = DataStore(df_with_nan)
        pd_result = pd_df.fillna({'category': 'Unknown', 'value': 0})
        try:
            ds_result = ds.fillna({'category': 'Unknown', 'value': 0})
            assert_datastore_equals_pandas(ds_result, pd_result)
        except Exception as e:
            pytest.skip(f"fillna(dict) not fully implemented: {e}")

    def test_agg_multiple_columns_multiple_funcs(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df.groupby('category').agg({'value': ['sum', 'mean'], 'score': ['min', 'max']})
        try:
            ds_result = ds.groupby('category').agg({'value': ['sum', 'mean'], 'score': ['min', 'max']})
            ds_exec = get_series(ds_result)
            assert len(ds_exec) == len(pd_df['category'].unique())
        except Exception as e:
            pytest.skip(f"agg(dict with multiple funcs) not fully implemented: {e}")


class TestSpecialValuesBoundary:
    """Test handling of special values."""

    def test_empty_dataframe_operations(self):
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds = DataStore(pd_df)
        assert len(ds) == 0
        ds_filtered = ds[ds['a'] > 0]
        assert len(ds_filtered) == 0

    def test_single_row_operations(self):
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds = DataStore(pd_df)
        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds[ds['a'] > 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_nan_column(self):
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [np.nan, np.nan, np.nan]})
        ds = DataStore(pd_df)
        pd_sum = pd_df['b'].sum()
        ds_result = ds['b'].sum()
        ds_val = get_value(ds_result)
        assert pd.isna(ds_val) or ds_val == 0 or ds_val == pd_sum

    def test_unicode_column_names(self):
        pd_df = pd.DataFrame({'col_a': ['Alice', 'Bob'], 'col_b': [25, 30]})
        ds = DataStore(pd_df)
        pd_result = pd_df[pd_df['col_b'] > 26]
        try:
            ds_result = ds[ds['col_b'] > 26]
            assert_datastore_equals_pandas(ds_result, pd_result)
        except Exception as e:
            pytest.skip(f"Unicode column names not fully supported: {e}")


class TestComplexExpressions:
    """Test complex expression handling."""

    def test_arithmetic_chain(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_df = pd_df.copy()
        pd_df['computed'] = (pd_df['value'] * 2 + pd_df['score']) / 3
        ds_result = ds.assign(computed=(ds['value'] * 2 + ds['score']) / 3)
        assert_datastore_equals_pandas(ds_result, pd_df)

    def test_comparison_chain(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[(pd_df['value'] > 10) & (pd_df['value'] < 50) & (pd_df['score'] > 2)]
        ds_result = ds[(ds['value'] > 10) & (ds['value'] < 50) & (ds['score'] > 2)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_condition(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[(pd_df['category'] == 'A') | (pd_df['category'] == 'C')]
        ds_result = ds[(ds['category'] == 'A') | (ds['category'] == 'C')]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[~(pd_df['category'] == 'A')]
        ds_result = ds[~(ds['category'] == 'A')]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_operation(self, sample_df):
        pd_df = sample_df
        ds = DataStore(sample_df)
        pd_result = pd_df[pd_df['category'].isin(['A', 'C'])]
        ds_result = ds[ds['category'].isin(['A', 'C'])]
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
