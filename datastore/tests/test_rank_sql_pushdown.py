"""
Tests for rank() SQL window function pushdown.

These tests verify that:
1. rank(method='first') uses SQL ROW_NUMBER()
2. rank(method='min') uses SQL RANK()
3. rank(method='dense') uses SQL DENSE_RANK()
4. Other methods fall back to pandas
5. Groupby rank still works correctly
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestColumnExprRankSQLPushdown:
    """Test ColumnExpr.rank() with SQL window function pushdown."""

    @pytest.fixture
    def sample_df(self):
        """Sample data with ties for testing rank behavior.
        
        Use larger dataset (10000 rows) for stable testing - small datasets
        may not reliably expose non-deterministic row order issues.
        """
        np.random.seed(42)
        n = 10000
        return pd.DataFrame({
            'value': np.random.randint(10, 50, size=n),
            'category': np.random.choice(['A', 'B', 'C'], size=n)
        })

    def test_rank_method_first_sql(self, sample_df):
        """Test rank(method='first') uses SQL ROW_NUMBER()."""
        pd_result = sample_df['value'].rank(method='first')
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='first')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_method_min_sql(self, sample_df):
        """Test rank(method='min') uses SQL RANK()."""
        pd_result = sample_df['value'].rank(method='min')
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='min')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_method_dense_sql(self, sample_df):
        """Test rank(method='dense') uses SQL DENSE_RANK()."""
        pd_result = sample_df['value'].rank(method='dense')
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='dense')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_method_average_pandas(self, sample_df):
        """Test rank(method='average') falls back to pandas (no SQL equivalent)."""
        pd_result = sample_df['value'].rank(method='average')
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='average')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_method_max_pandas(self, sample_df):
        """Test rank(method='max') falls back to pandas (no SQL equivalent)."""
        pd_result = sample_df['value'].rank(method='max')
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='max')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_ascending_false(self, sample_df):
        """Test rank with ascending=False."""
        pd_result = sample_df['value'].rank(method='first', ascending=False)
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='first', ascending=False)
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_pct_pandas(self, sample_df):
        """Test rank with pct=True falls back to pandas."""
        pd_result = sample_df['value'].rank(method='first', pct=True)
        
        ds = DataStore(sample_df)
        ds_result = ds['value'].rank(method='first', pct=True)
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_na_option_bottom_pandas(self):
        """Test rank with na_option='bottom' falls back to pandas."""
        df = pd.DataFrame({'value': [30, None, 20, 20, 40]})
        pd_result = df['value'].rank(method='first', na_option='bottom')
        
        ds = DataStore(df)
        ds_result = ds['value'].rank(method='first', na_option='bottom')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_na_option_top_pandas(self):
        """Test rank with na_option='top' falls back to pandas."""
        df = pd.DataFrame({'value': [30, None, 20, 20, 40]})
        pd_result = df['value'].rank(method='first', na_option='top')
        
        ds = DataStore(df)
        ds_result = ds['value'].rank(method='first', na_option='top')
        
        assert_series_equal(ds_result, pd_result)


class TestColumnExprRankWithTies:
    """Test rank behavior with tied values."""

    @pytest.fixture
    def ties_df(self):
        """Data with multiple tie scenarios.
        
        Use larger dataset (10000 rows) for stable testing.
        """
        np.random.seed(42)
        # Generate data with many ties (only 5 distinct values in 10000 rows)
        values = np.random.choice([10, 20, 30, 40, 50], size=10000)
        return pd.DataFrame({
            'value': values
        })

    def test_rank_first_with_ties(self, ties_df):
        """ROW_NUMBER gives unique ranks even for ties."""
        pd_result = ties_df['value'].rank(method='first')
        
        ds = DataStore(ties_df)
        ds_result = ds['value'].rank(method='first')
        
        assert_series_equal(ds_result, pd_result)
        # Verify unique ranks
        assert len(set(ds_result)) == len(ds_result)

    def test_rank_min_with_ties(self, ties_df):
        """RANK gives same rank for ties, gaps after."""
        pd_result = ties_df['value'].rank(method='min')
        
        ds = DataStore(ties_df)
        ds_result = ds['value'].rank(method='min')
        
        assert_series_equal(ds_result, pd_result)

    def test_rank_dense_with_ties(self, ties_df):
        """DENSE_RANK gives same rank for ties, no gaps."""
        pd_result = ties_df['value'].rank(method='dense')
        
        ds = DataStore(ties_df)
        ds_result = ds['value'].rank(method='dense')
        
        assert_series_equal(ds_result, pd_result)


class TestGroupbyRank:
    """Test rank in groupby context."""

    @pytest.fixture
    def grouped_df(self):
        """Data for groupby rank tests.
        
        Use larger dataset (10000 rows) for stable testing.
        """
        np.random.seed(42)
        n = 10000
        return pd.DataFrame({
            'category': np.random.choice(['A', 'B', 'C'], size=n),
            'value': np.random.randint(10, 50, size=n)
        })

    def test_groupby_rank_default(self, grouped_df):
        """Test groupby rank with default method."""
        pd_result = grouped_df.groupby('category')['value'].rank()
        
        ds = DataStore(grouped_df)
        ds_result = ds.groupby('category')['value'].rank()
        
        assert_series_equal(ds_result, pd_result)

    def test_groupby_rank_method_first(self, grouped_df):
        """Test groupby rank with method='first'."""
        pd_result = grouped_df.groupby('category')['value'].rank(method='first')
        
        ds = DataStore(grouped_df)
        ds_result = ds.groupby('category')['value'].rank(method='first')
        
        assert_series_equal(ds_result, pd_result)

    def test_groupby_rank_method_min(self, grouped_df):
        """Test groupby rank with method='min'."""
        pd_result = grouped_df.groupby('category')['value'].rank(method='min')
        
        ds = DataStore(grouped_df)
        ds_result = ds.groupby('category')['value'].rank(method='min')
        
        assert_series_equal(ds_result, pd_result)

    def test_groupby_rank_method_dense(self, grouped_df):
        """Test groupby rank with method='dense'."""
        pd_result = grouped_df.groupby('category')['value'].rank(method='dense')
        
        ds = DataStore(grouped_df)
        ds_result = ds.groupby('category')['value'].rank(method='dense')
        
        assert_series_equal(ds_result, pd_result)


class TestRankWithAssign:
    """Test rank with assign for SQL pushdown verification."""

    @pytest.fixture
    def sample_df(self):
        """Use larger dataset (10000 rows) for stable testing."""
        np.random.seed(42)
        n = 10000
        return pd.DataFrame({
            'value': np.random.randint(10, 50, size=n)
        })

    def test_assign_rank_first(self, sample_df):
        """Test rank via assign - should use SQL window function."""
        pd_result = sample_df.assign(rank=sample_df['value'].rank(method='first'))
        
        ds = DataStore(sample_df)
        ds_result = ds.assign(rank=ds['value'].rank(method='first'))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_rank_min(self, sample_df):
        """Test rank via assign with method='min'."""
        pd_result = sample_df.assign(rank=sample_df['value'].rank(method='min'))
        
        ds = DataStore(sample_df)
        ds_result = ds.assign(rank=ds['value'].rank(method='min'))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_rank_dense(self, sample_df):
        """Test rank via assign with method='dense'."""
        pd_result = sample_df.assign(rank=sample_df['value'].rank(method='dense'))
        
        ds = DataStore(sample_df)
        ds_result = ds.assign(rank=ds['value'].rank(method='dense'))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_ranks(self, sample_df):
        """Test assigning multiple rank columns."""
        pd_result = sample_df.assign(
            rank_first=sample_df['value'].rank(method='first'),
            rank_min=sample_df['value'].rank(method='min'),
            rank_dense=sample_df['value'].rank(method='dense')
        )
        
        ds = DataStore(sample_df)
        ds_result = ds.assign(
            rank_first=ds['value'].rank(method='first'),
            rank_min=ds['value'].rank(method='min'),
            rank_dense=ds['value'].rank(method='dense')
        )
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRankWithExpressions:
    """Test rank on computed expressions."""

    @pytest.fixture
    def sample_df(self):
        """Use larger dataset (10000 rows) for stable testing."""
        np.random.seed(42)
        n = 10000
        return pd.DataFrame({
            'a': np.random.randint(1, 20, size=n),
            'b': np.random.randint(10, 100, size=n)
        })

    def test_rank_on_arithmetic_expression(self, sample_df):
        """Test rank on a computed column."""
        pd_result = (sample_df['a'] + sample_df['b']).rank(method='first')
        
        ds = DataStore(sample_df)
        ds_result = (ds['a'] + ds['b']).rank(method='first')
        
        assert_series_equal(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
