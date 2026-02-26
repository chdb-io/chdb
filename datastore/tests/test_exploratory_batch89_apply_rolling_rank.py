"""
Exploratory Batch 89: Apply, Rolling Windows, Rank, and Time-Series Operations

Date: 2026-01-16
Exploration Method: Architecture-based exploratory testing

Focus areas:
1. Apply/Transform operations on columns and DataFrames
2. Rolling window operations (mean, sum, std)
3. Shift, diff, pct_change
4. Cumulative operations (cumsum, cummax, cummin, cumprod)
5. Rank operations with different methods
6. Duplicated and drop_duplicates edge cases
7. Complex method chains combining these operations
"""

import pytest
import numpy as np
import pandas as pd
from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal, assert_datastore_equals_pandas


class TestApplyOperations:
    """Test apply operations on columns and DataFrames."""

    def test_column_apply_simple_lambda(self):
        """Apply simple lambda to column."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].apply(lambda x: x * 2)
        ds_result = ds_df['value'].apply(lambda x: x * 2)

        assert_series_equal(ds_result, pd_result)

    def test_column_apply_with_numpy(self):
        """Apply numpy function to column."""
        pd_df = pd.DataFrame({'value': [1, 4, 9, 16, 25]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].apply(np.sqrt)
        ds_result = ds_df['value'].apply(np.sqrt)

        assert_series_equal(ds_result, pd_result)

    def test_column_apply_string_function(self):
        """Apply string transformation via apply."""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['name'].apply(str.upper)
        ds_result = ds_df['name'].apply(str.upper)

        assert_series_equal(ds_result, pd_result)

    def test_dataframe_apply_column_wise(self):
        """Apply function column-wise (axis=0)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(np.sum, axis=0)
        ds_result = ds_df.apply(np.sum, axis=0)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_dataframe_apply_row_wise(self):
        """Apply function row-wise (axis=1)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(np.sum, axis=1)
        ds_result = ds_df.apply(np.sum, axis=1)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_apply_with_none_values(self):
        """Apply function on column with None values."""
        pd_df = pd.DataFrame({'value': [1, None, 3, None, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].apply(lambda x: x * 2 if pd.notna(x) else None)
        ds_result = ds_df['value'].apply(lambda x: x * 2 if pd.notna(x) else None)

        assert_series_equal(ds_result, pd_result, check_dtype=False)


class TestRollingWindowOperations:
    """Test rolling window operations."""

    def test_rolling_mean_basic(self):
        """Basic rolling mean."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rolling(window=2).mean()
        ds_result = ds_df['value'].rolling(window=2).mean()

        assert_series_equal(ds_result, pd_result)

    def test_rolling_sum_with_min_periods(self):
        """Rolling sum with min_periods."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rolling(window=3, min_periods=1).sum()
        ds_result = ds_df['value'].rolling(window=3, min_periods=1).sum()

        assert_series_equal(ds_result, pd_result)

    def test_rolling_std(self):
        """Rolling standard deviation."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rolling(window=3).std()
        ds_result = ds_df['value'].rolling(window=3).std()

        assert_series_equal(ds_result, pd_result)

    def test_rolling_on_dataframe(self):
        """Rolling mean on full DataFrame."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0],
            'B': [10.0, 20.0, 30.0, 40.0]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rolling(window=2).mean()
        ds_result = ds_df.rolling(window=2).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_max_min(self):
        """Rolling max and min."""
        pd_df = pd.DataFrame({'value': [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]})
        ds_df = DataStore(pd_df)

        pd_max = pd_df['value'].rolling(window=3).max()
        ds_max = ds_df['value'].rolling(window=3).max()
        assert_series_equal(ds_max, pd_max)

        pd_min = pd_df['value'].rolling(window=3).min()
        ds_min = ds_df['value'].rolling(window=3).min()
        assert_series_equal(ds_min, pd_min)


class TestShiftDiffPctChange:
    """Test shift, diff, and pct_change operations."""

    def test_shift_positive(self):
        """Shift column forward."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].shift(1)
        ds_result = ds_df['value'].shift(1)

        assert_series_equal(ds_result, pd_result)

    def test_shift_negative(self):
        """Shift column backward."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].shift(-1)
        ds_result = ds_df['value'].shift(-1)

        assert_series_equal(ds_result, pd_result)

    def test_shift_with_fill_value(self):
        """Shift with fill_value."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].shift(2, fill_value=0)
        ds_result = ds_df['value'].shift(2, fill_value=0)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_diff_basic(self):
        """Basic diff operation."""
        pd_df = pd.DataFrame({'value': [1, 3, 6, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].diff()
        ds_result = ds_df['value'].diff()

        assert_series_equal(ds_result, pd_result)

    def test_diff_periods_2(self):
        """Diff with periods=2."""
        pd_df = pd.DataFrame({'value': [1, 2, 4, 7, 11]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].diff(periods=2)
        ds_result = ds_df['value'].diff(periods=2)

        assert_series_equal(ds_result, pd_result)

    def test_pct_change_basic(self):
        """Basic pct_change."""
        pd_df = pd.DataFrame({'value': [100.0, 110.0, 121.0, 133.1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].pct_change()
        ds_result = ds_df['value'].pct_change()

        assert_series_equal(ds_result, pd_result)

    def test_dataframe_shift(self):
        """Shift on full DataFrame."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.shift(1)
        ds_result = ds_df.shift(1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_diff(self):
        """Diff on full DataFrame."""
        pd_df = pd.DataFrame({
            'A': [1, 3, 6, 10],
            'B': [10, 20, 40, 80]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.diff()
        ds_result = ds_df.diff()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCumulativeOperations:
    """Test cumulative operations."""

    def test_cumsum_basic(self):
        """Basic cumulative sum."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].cumsum()
        ds_result = ds_df['value'].cumsum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cumsum_with_nulls(self):
        """Cumulative sum with null values."""
        pd_df = pd.DataFrame({'value': [1.0, None, 3.0, None, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].cumsum()
        ds_result = ds_df['value'].cumsum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cumprod_basic(self):
        """Basic cumulative product."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].cumprod()
        ds_result = ds_df['value'].cumprod()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cummax_cummin(self):
        """Cumulative max and min."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds_df = DataStore(pd_df)

        pd_max = pd_df['value'].cummax()
        ds_max = ds_df['value'].cummax()
        assert_series_equal(ds_max, pd_max, check_dtype=False)

        pd_min = pd_df['value'].cummin()
        ds_min = ds_df['value'].cummin()
        assert_series_equal(ds_min, pd_min, check_dtype=False)

    def test_dataframe_cumsum(self):
        """Cumulative sum on DataFrame."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.cumsum()
        ds_result = ds_df.cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


class TestRankOperations:
    """Test rank operations with different methods."""

    def test_rank_default(self):
        """Default rank (method='average')."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank()
        ds_result = ds_df['value'].rank()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_method_min(self):
        """Rank with method='min'."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank(method='min')
        ds_result = ds_df['value'].rank(method='min')

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_method_max(self):
        """Rank with method='max'."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank(method='max')
        ds_result = ds_df['value'].rank(method='max')

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_method_first(self):
        """Rank with method='first' (by appearance order)."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank(method='first')
        ds_result = ds_df['value'].rank(method='first')

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_ascending_false(self):
        """Rank in descending order."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank(ascending=False)
        ds_result = ds_df['value'].rank(ascending=False)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_pct(self):
        """Rank as percentage."""
        pd_df = pd.DataFrame({'value': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank(pct=True)
        ds_result = ds_df['value'].rank(pct=True)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rank_with_nulls(self):
        """Rank with null values."""
        pd_df = pd.DataFrame({'value': [3.0, None, 4.0, 1.0, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank()
        ds_result = ds_df['value'].rank()

        assert_series_equal(ds_result, pd_result, check_dtype=False)


class TestDuplicatedOperations:
    """Test duplicated and drop_duplicates operations."""

    def test_duplicated_default(self):
        """Default duplicated (keep='first')."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].duplicated()
        ds_result = ds_df['value'].duplicated()

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_keep_last(self):
        """Duplicated with keep='last'."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].duplicated(keep='last')
        ds_result = ds_df['value'].duplicated(keep='last')

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_keep_false(self):
        """Duplicated with keep=False (mark all duplicates)."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].duplicated(keep=False)
        ds_result = ds_df['value'].duplicated(keep=False)

        assert_series_equal(ds_result, pd_result)

    def test_dataframe_duplicated_subset(self):
        """DataFrame duplicated with subset."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'x']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.duplicated(subset=['A'])
        ds_result = ds_df.duplicated(subset=['A'])

        assert_series_equal(ds_result, pd_result)

    def test_drop_duplicates_default(self):
        """Default drop_duplicates (keep='first')."""
        pd_df = pd.DataFrame({'value': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        # DataStore does not preserve original index, so check_index=False
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_drop_duplicates_keep_last(self):
        """Drop duplicates keeping last occurrence."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop_duplicates(subset=['A'], keep='last')
        ds_result = ds_df.drop_duplicates(subset=['A'], keep='last')

        # DataStore does not preserve original index, so check_index=False
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_drop_duplicates_ignore_index(self):
        """Drop duplicates with ignore_index=True."""
        pd_df = pd.DataFrame({
            'value': [1, 2, 2, 3, 3, 3]
        }, index=[10, 20, 30, 40, 50, 60])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop_duplicates(ignore_index=True)
        ds_result = ds_df.drop_duplicates(ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChains:
    """Test complex chains combining time-series operations."""

    @pytest.mark.xfail(
        reason="chDB does not support nested window functions (e.g., shift().diff())",
        strict=True
    )
    def test_shift_then_diff(self):
        """Shift followed by diff - chDB nested window function limitation."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].shift(1).diff()
        ds_result = ds_df['value'].shift(1).diff()

        assert_series_equal(ds_result, pd_result)

    def test_rolling_then_rank(self):
        """Rolling mean followed by rank."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rolling(window=2).mean().rank()
        ds_result = ds_df['value'].rolling(window=2).mean().rank()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cumsum_then_pct_change(self):
        """Cumulative sum followed by pct_change."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].cumsum().pct_change()
        ds_result = ds_df['value'].cumsum().pct_change()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_filter_then_rolling(self):
        """Filter then rolling window."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'A', 'B', 'A'],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['category'] == 'A']
        ds_filtered = ds_df[ds_df['category'] == 'A']

        pd_result = pd_filtered['value'].rolling(window=2).mean()
        ds_result = ds_filtered['value'].rolling(window=2).mean()

        # DataStore filter may not preserve original index; compare values only
        assert_series_equal(ds_result, pd_result, check_index=False)

    @pytest.mark.xfail(
        reason="chDB cannot use window functions in WHERE clause (assign diff then filter)",
        strict=True
    )
    def test_assign_diff_then_filter(self):
        """Assign diff column then filter - chDB window function in WHERE limitation."""
        pd_df = pd.DataFrame({'value': [10, 15, 12, 18, 14, 20]})
        ds_df = DataStore(pd_df)

        pd_df_with_diff = pd_df.assign(change=pd_df['value'].diff())
        ds_df_with_diff = ds_df.assign(change=ds_df['value'].diff())

        pd_result = pd_df_with_diff[pd_df_with_diff['change'] > 0]
        ds_result = ds_df_with_diff[ds_df_with_diff['change'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_transform_rank(self):
        """GroupBy transform with rank."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 2, 6, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('group')['value'].rank()
        ds_result = ds_df.groupby('group')['value'].rank()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_rolling_assign_then_dropna(self):
        """Rolling assign then dropna."""
        pd_df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        ds_df = DataStore(pd_df)

        pd_df_with_rolling = pd_df.assign(rolling_mean=pd_df['value'].rolling(window=2).mean())
        ds_df_with_rolling = ds_df.assign(rolling_mean=ds_df['value'].rolling(window=2).mean())

        pd_result = pd_df_with_rolling.dropna()
        ds_result = ds_df_with_rolling.dropna()

        # DataStore dropna may not preserve original index; compare values only
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)


class TestEdgeCases:
    """Test edge cases for time-series operations."""

    def test_rolling_on_single_value(self):
        """Rolling on single value DataFrame."""
        pd_df = pd.DataFrame({'value': [42.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rolling(window=2).mean()
        ds_result = ds_df['value'].rolling(window=2).mean()

        assert_series_equal(ds_result, pd_result)

    def test_shift_larger_than_length(self):
        """Shift by more than DataFrame length."""
        pd_df = pd.DataFrame({'value': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].shift(5)
        ds_result = ds_df['value'].shift(5)

        assert_series_equal(ds_result, pd_result)

    def test_diff_on_constant_values(self):
        """Diff on column with constant values."""
        pd_df = pd.DataFrame({'value': [5, 5, 5, 5, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].diff()
        ds_result = ds_df['value'].diff()

        assert_series_equal(ds_result, pd_result)

    def test_rank_all_same_values(self):
        """Rank when all values are the same."""
        pd_df = pd.DataFrame({'value': [7, 7, 7, 7]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].rank()
        ds_result = ds_df['value'].rank()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cumsum_empty_dataframe(self):
        """Cumsum on empty DataFrame."""
        pd_df = pd.DataFrame({'value': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].cumsum()
        ds_result = ds_df['value'].cumsum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_pct_change_with_zeros(self):
        """Pct_change with zero values."""
        pd_df = pd.DataFrame({'value': [0.0, 1.0, 0.0, 2.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].pct_change()
        ds_result = ds_df['value'].pct_change()

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_all_unique(self):
        """Duplicated on all unique values."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].duplicated()
        ds_result = ds_df['value'].duplicated()

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_all_same(self):
        """Duplicated when all values are the same."""
        pd_df = pd.DataFrame({'value': [5, 5, 5, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['value'].duplicated()
        ds_result = ds_df['value'].duplicated()

        assert_series_equal(ds_result, pd_result)
