"""
Exploratory Batch 10: Window Functions and Time Series Operations

Focus areas:
1. Rolling window operations (sum, mean, std, min, max, count, apply)
2. Expanding window operations
3. EWM (Exponentially Weighted Moving) operations
4. Time series methods: shift, diff, pct_change
5. DataFrame comparison methods: eq, ne, gt, lt, ge, le
6. Advanced indexing: take, squeeze, pipe, eval
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series, get_value


# =============================================================================
# Part 1: Rolling Window Operations
# =============================================================================

class TestRollingBasic:
    """Test basic rolling window operations."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            'C': [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]
        })
    
    def test_rolling_mean(self, sample_df):
        """Test rolling mean."""
        pd_result = sample_df.rolling(window=3).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_sum(self, sample_df):
        """Test rolling sum."""
        pd_result = sample_df.rolling(window=3).sum()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).sum()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_std(self, sample_df):
        """Test rolling standard deviation."""
        pd_result = sample_df.rolling(window=3).std()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).std()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_min(self, sample_df):
        """Test rolling minimum."""
        pd_result = sample_df.rolling(window=3).min()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).min()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_max(self, sample_df):
        """Test rolling maximum."""
        pd_result = sample_df.rolling(window=3).max()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).max()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_count(self, sample_df):
        """Test rolling count (handles NaN)."""
        pd_result = sample_df.rolling(window=3).count()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).count()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_var(self, sample_df):
        """Test rolling variance."""
        pd_result = sample_df.rolling(window=3).var()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3).var()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_min_periods(self, sample_df):
        """Test rolling with min_periods parameter."""
        pd_result = sample_df.rolling(window=3, min_periods=1).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3, min_periods=1).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rolling_center(self, sample_df):
        """Test rolling with center=True."""
        pd_result = sample_df.rolling(window=3, center=True).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.rolling(window=3, center=True).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRollingSingleColumn:
    """Test rolling on single column."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
    
    def test_rolling_single_column_mean(self, sample_df):
        """Test rolling mean on single column."""
        pd_result = sample_df['A'].rolling(window=2).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds['A'].rolling(window=2).mean()
        
        # Compare as Series
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))
    
    def test_rolling_single_column_sum(self, sample_df):
        """Test rolling sum on single column."""
        pd_result = sample_df['A'].rolling(window=2).sum()
        
        ds = DataStore(sample_df)
        ds_result = ds['A'].rolling(window=2).sum()
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))


# =============================================================================
# Part 2: Expanding Window Operations
# =============================================================================

class TestExpandingBasic:
    """Test expanding window operations."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
    
    def test_expanding_mean(self, sample_df):
        """Test expanding mean (cumulative mean)."""
        pd_result = sample_df.expanding().mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.expanding().mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_expanding_sum(self, sample_df):
        """Test expanding sum (cumulative sum)."""
        pd_result = sample_df.expanding().sum()
        
        ds = DataStore(sample_df)
        ds_result = ds.expanding().sum()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_expanding_min(self, sample_df):
        """Test expanding min (cumulative min)."""
        pd_result = sample_df.expanding().min()
        
        ds = DataStore(sample_df)
        ds_result = ds.expanding().min()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_expanding_max(self, sample_df):
        """Test expanding max (cumulative max)."""
        pd_result = sample_df.expanding().max()
        
        ds = DataStore(sample_df)
        ds_result = ds.expanding().max()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_expanding_std(self, sample_df):
        """Test expanding standard deviation."""
        pd_result = sample_df.expanding().std()
        
        ds = DataStore(sample_df)
        ds_result = ds.expanding().std()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_expanding_min_periods(self, sample_df):
        """Test expanding with min_periods."""
        pd_result = sample_df.expanding(min_periods=2).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.expanding(min_periods=2).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 3: EWM (Exponentially Weighted Moving) Operations
# =============================================================================

class TestEWMBasic:
    """Test EWM operations."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
    
    def test_ewm_mean_span(self, sample_df):
        """Test EWM mean with span parameter."""
        pd_result = sample_df.ewm(span=3).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.ewm(span=3).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ewm_mean_alpha(self, sample_df):
        """Test EWM mean with alpha parameter."""
        pd_result = sample_df.ewm(alpha=0.5).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.ewm(alpha=0.5).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ewm_mean_com(self, sample_df):
        """Test EWM mean with com (center of mass) parameter."""
        pd_result = sample_df.ewm(com=2).mean()
        
        ds = DataStore(sample_df)
        ds_result = ds.ewm(com=2).mean()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ewm_std(self, sample_df):
        """Test EWM standard deviation."""
        pd_result = sample_df.ewm(span=3).std()
        
        ds = DataStore(sample_df)
        ds_result = ds.ewm(span=3).std()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ewm_var(self, sample_df):
        """Test EWM variance."""
        pd_result = sample_df.ewm(span=3).var()
        
        ds = DataStore(sample_df)
        ds_result = ds.ewm(span=3).var()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 4: Time Series Methods
# =============================================================================

class TestShift:
    """Test shift operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    def test_shift_positive(self, sample_df):
        """Test shift with positive periods (shift forward)."""
        pd_result = sample_df.shift(periods=1)
        
        ds = DataStore(sample_df)
        ds_result = ds.shift(periods=1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_shift_negative(self, sample_df):
        """Test shift with negative periods (shift backward)."""
        pd_result = sample_df.shift(periods=-1)
        
        ds = DataStore(sample_df)
        ds_result = ds.shift(periods=-1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_shift_multiple(self, sample_df):
        """Test shift with multiple periods."""
        pd_result = sample_df.shift(periods=2)
        
        ds = DataStore(sample_df)
        ds_result = ds.shift(periods=2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_shift_fill_value(self, sample_df):
        """Test shift with fill_value."""
        pd_result = sample_df.shift(periods=1, fill_value=0)
        
        ds = DataStore(sample_df)
        ds_result = ds.shift(periods=1, fill_value=0)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_shift_single_column(self, sample_df):
        """Test shift on single column."""
        pd_result = sample_df['A'].shift(periods=1)
        
        ds = DataStore(sample_df)
        ds_result = ds['A'].shift(periods=1)
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))


class TestDiff:
    """Test diff operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1.0, 3.0, 6.0, 10.0, 15.0],
            'B': [10.0, 30.0, 60.0, 100.0, 150.0]
        })
    
    def test_diff_default(self, sample_df):
        """Test diff with default period=1."""
        pd_result = sample_df.diff()
        
        ds = DataStore(sample_df)
        ds_result = ds.diff()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_diff_periods_2(self, sample_df):
        """Test diff with periods=2."""
        pd_result = sample_df.diff(periods=2)
        
        ds = DataStore(sample_df)
        ds_result = ds.diff(periods=2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_diff_negative(self, sample_df):
        """Test diff with negative periods."""
        pd_result = sample_df.diff(periods=-1)
        
        ds = DataStore(sample_df)
        ds_result = ds.diff(periods=-1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_diff_single_column(self, sample_df):
        """Test diff on single column."""
        pd_result = sample_df['A'].diff()
        
        ds = DataStore(sample_df)
        ds_result = ds['A'].diff()
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))


class TestPctChange:
    """Test pct_change operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [100.0, 110.0, 121.0, 133.1, 146.41],
            'B': [200.0, 220.0, 242.0, 266.2, 292.82]
        })
    
    def test_pct_change_default(self, sample_df):
        """Test pct_change with default period=1."""
        pd_result = sample_df.pct_change()
        
        ds = DataStore(sample_df)
        ds_result = ds.pct_change()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_pct_change_periods_2(self, sample_df):
        """Test pct_change with periods=2."""
        pd_result = sample_df.pct_change(periods=2)
        
        ds = DataStore(sample_df)
        ds_result = ds.pct_change(periods=2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_pct_change_single_column(self, sample_df):
        """Test pct_change on single column."""
        pd_result = sample_df['A'].pct_change()
        
        ds = DataStore(sample_df)
        ds_result = ds['A'].pct_change()
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))


# =============================================================================
# Part 5: DataFrame Comparison Methods
# =============================================================================

class TestComparisonMethods:
    """Test DataFrame comparison methods (eq, ne, gt, lt, ge, le)."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
    
    def test_eq_scalar(self, sample_df):
        """Test eq with scalar."""
        pd_result = sample_df.eq(3)
        
        ds = DataStore(sample_df)
        ds_result = ds.eq(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ne_scalar(self, sample_df):
        """Test ne with scalar."""
        pd_result = sample_df.ne(3)
        
        ds = DataStore(sample_df)
        ds_result = ds.ne(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_gt_scalar(self, sample_df):
        """Test gt with scalar."""
        pd_result = sample_df.gt(3)
        
        ds = DataStore(sample_df)
        ds_result = ds.gt(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_lt_scalar(self, sample_df):
        """Test lt with scalar."""
        pd_result = sample_df.lt(3)
        
        ds = DataStore(sample_df)
        ds_result = ds.lt(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ge_scalar(self, sample_df):
        """Test ge with scalar."""
        pd_result = sample_df.ge(3)
        
        ds = DataStore(sample_df)
        ds_result = ds.ge(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_le_scalar(self, sample_df):
        """Test le with scalar."""
        pd_result = sample_df.le(3)
        
        ds = DataStore(sample_df)
        ds_result = ds.le(3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_eq_dataframe(self, sample_df):
        """Test eq with another DataFrame."""
        other = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 5, 5, 5, 5]
        })
        pd_result = sample_df.eq(other)
        
        ds = DataStore(sample_df)
        ds_other = DataStore(other)
        ds_result = ds.eq(ds_other)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 6: Advanced Indexing Operations
# =============================================================================

class TestTake:
    """Test take operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
    
    def test_take_positive_indices(self, sample_df):
        """Test take with positive indices."""
        pd_result = sample_df.take([0, 2, 4])
        
        ds = DataStore(sample_df)
        ds_result = ds.take([0, 2, 4])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_take_negative_indices(self, sample_df):
        """Test take with negative indices."""
        pd_result = sample_df.take([-1, -2, -3])
        
        ds = DataStore(sample_df)
        ds_result = ds.take([-1, -2, -3])
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_take_mixed_indices(self, sample_df):
        """Test take with mixed positive/negative indices."""
        pd_result = sample_df.take([0, -1, 2])
        
        ds = DataStore(sample_df)
        ds_result = ds.take([0, -1, 2])
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSqueeze:
    """Test squeeze operation."""
    
    def test_squeeze_single_column(self):
        """Test squeeze on single column DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_result = pd_df.squeeze()
        
        ds = DataStore(pd_df)
        ds_result = ds.squeeze()
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))
    
    def test_squeeze_single_row(self):
        """Test squeeze on single row DataFrame."""
        pd_df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
        pd_result = pd_df.squeeze()
        
        ds = DataStore(pd_df)
        ds_result = ds.squeeze()
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))
    
    def test_squeeze_single_value(self):
        """Test squeeze on single value DataFrame."""
        pd_df = pd.DataFrame({'A': [42]})
        pd_result = pd_df.squeeze()
        
        ds = DataStore(pd_df)
        ds_result = ds.squeeze()
        
        # Single value squeeze should return scalar
        ds_val = get_value(ds_result)
        assert ds_val == pd_result
    
    def test_squeeze_no_effect(self):
        """Test squeeze on multi-row multi-column DataFrame (no effect)."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_result = pd_df.squeeze()
        
        ds = DataStore(pd_df)
        ds_result = ds.squeeze()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestPipe:
    """Test pipe operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    def test_pipe_simple_function(self, sample_df):
        """Test pipe with simple function."""
        def double_values(df):
            return df * 2
        
        pd_result = sample_df.pipe(double_values)
        
        ds = DataStore(sample_df)
        ds_result = ds.pipe(double_values)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_pipe_with_args(self, sample_df):
        """Test pipe with function that takes additional args."""
        def multiply_by(df, factor):
            return df * factor
        
        pd_result = sample_df.pipe(multiply_by, 3)
        
        ds = DataStore(sample_df)
        ds_result = ds.pipe(multiply_by, 3)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_pipe_chain(self, sample_df):
        """Test chained pipe operations."""
        def add_one(df):
            return df + 1
        
        def multiply_two(df):
            return df * 2
        
        pd_result = sample_df.pipe(add_one).pipe(multiply_two)
        
        ds = DataStore(sample_df)
        ds_result = ds.pipe(add_one).pipe(multiply_two)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEval:
    """Test eval operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    def test_eval_simple_expression(self, sample_df):
        """Test eval with simple arithmetic expression."""
        pd_result = sample_df.eval('C = A + B')
        
        ds = DataStore(sample_df)
        ds_result = ds.eval('C = A + B')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_eval_complex_expression(self, sample_df):
        """Test eval with complex expression."""
        pd_result = sample_df.eval('C = A * 2 + B / 10')
        
        ds = DataStore(sample_df)
        ds_result = ds.eval('C = A * 2 + B / 10')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_eval_returns_series(self, sample_df):
        """Test eval that returns a Series."""
        pd_result = sample_df.eval('A + B')
        
        ds = DataStore(sample_df)
        ds_result = ds.eval('A + B')
        
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), 
            pd_result.reset_index(drop=True))


class TestQuery:
    """Test query operation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'x', 'y', 'x']
        })
    
    def test_query_simple(self, sample_df):
        """Test query with simple condition."""
        pd_result = sample_df.query('A > 2')
        
        ds = DataStore(sample_df)
        ds_result = ds.query('A > 2')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_query_compound(self, sample_df):
        """Test query with compound condition."""
        pd_result = sample_df.query('A > 2 and B < 50')
        
        ds = DataStore(sample_df)
        ds_result = ds.query('A > 2 and B < 50')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_query_string_column(self, sample_df):
        """Test query with string column."""
        pd_result = sample_df.query('C == "x"')
        
        ds = DataStore(sample_df)
        ds_result = ds.query('C == "x"')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 7: Cumulative Operations
# =============================================================================

class TestCumulativeOps:
    """Test cumulative operations."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
    
    def test_cumsum(self, sample_df):
        """Test cumulative sum."""
        pd_result = sample_df.cumsum()
        
        ds = DataStore(sample_df)
        ds_result = ds.cumsum()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cumprod(self, sample_df):
        """Test cumulative product."""
        pd_result = sample_df.cumprod()
        
        ds = DataStore(sample_df)
        ds_result = ds.cumprod()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cummin(self, sample_df):
        """Test cumulative minimum."""
        # Use non-monotonic data
        df = pd.DataFrame({'A': [3, 1, 4, 1, 5], 'B': [9, 2, 6, 5, 3]})
        pd_result = df.cummin()
        
        ds = DataStore(df)
        ds_result = ds.cummin()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_cummax(self, sample_df):
        """Test cumulative maximum."""
        # Use non-monotonic data
        df = pd.DataFrame({'A': [3, 1, 4, 1, 5], 'B': [9, 2, 6, 5, 3]})
        pd_result = df.cummax()
        
        ds = DataStore(df)
        ds_result = ds.cummax()
        
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 8: Rank Operations
# =============================================================================

class TestRank:
    """Test rank operations."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [3, 1, 4, 1, 5, 9, 2, 6],
            'B': [10, 40, 20, 40, 30, 10, 50, 20]
        })
    
    def test_rank_default(self, sample_df):
        """Test rank with default parameters."""
        pd_result = sample_df.rank()
        
        ds = DataStore(sample_df)
        ds_result = ds.rank()
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rank_method_min(self, sample_df):
        """Test rank with method='min'."""
        pd_result = sample_df.rank(method='min')
        
        ds = DataStore(sample_df)
        ds_result = ds.rank(method='min')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rank_method_max(self, sample_df):
        """Test rank with method='max'."""
        pd_result = sample_df.rank(method='max')
        
        ds = DataStore(sample_df)
        ds_result = ds.rank(method='max')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rank_method_first(self, sample_df):
        """Test rank with method='first'."""
        pd_result = sample_df.rank(method='first')
        
        ds = DataStore(sample_df)
        ds_result = ds.rank(method='first')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rank_method_dense(self, sample_df):
        """Test rank with method='dense'."""
        pd_result = sample_df.rank(method='dense')
        
        ds = DataStore(sample_df)
        ds_result = ds.rank(method='dense')
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rank_ascending_false(self, sample_df):
        """Test rank with ascending=False."""
        pd_result = sample_df.rank(ascending=False)
        
        ds = DataStore(sample_df)
        ds_result = ds.rank(ascending=False)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_rank_pct(self, sample_df):
        """Test rank with pct=True (percentile rank)."""
        pd_result = sample_df.rank(pct=True)
        
        ds = DataStore(sample_df)
        ds_result = ds.rank(pct=True)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestComparisonMethodsDataStore:
    """Test comparison methods with DataStore as other parameter."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1]
        })
    
    @pytest.fixture
    def other_df(self):
        return pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 5, 5, 5, 5]
        })
    
    def test_ne_dataframe(self, sample_df, other_df):
        """Test ne with another DataStore."""
        pd_result = sample_df.ne(other_df)
        
        ds = DataStore(sample_df)
        ds_other = DataStore(other_df)
        ds_result = ds.ne(ds_other)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_gt_dataframe(self, sample_df, other_df):
        """Test gt with another DataStore."""
        pd_result = sample_df.gt(other_df)
        
        ds = DataStore(sample_df)
        ds_other = DataStore(other_df)
        ds_result = ds.gt(ds_other)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_lt_dataframe(self, sample_df, other_df):
        """Test lt with another DataStore."""
        pd_result = sample_df.lt(other_df)
        
        ds = DataStore(sample_df)
        ds_other = DataStore(other_df)
        ds_result = ds.lt(ds_other)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_ge_dataframe(self, sample_df, other_df):
        """Test ge with another DataStore."""
        pd_result = sample_df.ge(other_df)
        
        ds = DataStore(sample_df)
        ds_other = DataStore(other_df)
        ds_result = ds.ge(ds_other)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
    
    def test_le_dataframe(self, sample_df, other_df):
        """Test le with another DataStore."""
        pd_result = sample_df.le(other_df)
        
        ds = DataStore(sample_df)
        ds_other = DataStore(other_df)
        ds_result = ds.le(ds_other)
        
        assert_datastore_equals_pandas(ds_result, pd_result)
