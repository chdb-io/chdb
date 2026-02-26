"""
Exploratory Batch 39: Shift, Diff, Assign, and Window Function Combinations

Focus areas:
1. shift() and diff() with various operations (filter, groupby, sort)
2. Complex assign() scenarios (multi-dependency, lambda chains)
3. pct_change() edge cases
4. Computed column references in chains
5. Window function results as filter conditions
6. Empty/single-row DataFrame edge cases with these operations
7. Nullable type handling with shift/diff

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
Discovery date: 2026-01-06
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas
# xfail markers removed - NULL comparison semantics fixed in conditions.py


# =============================================================================
# Test Group 1: Shift Operations with Chains
# =============================================================================


class TestShiftChains:
    """Test shift() operations in various chain scenarios."""

    @pytest.mark.xfail(
        reason="SQL limitation: window functions cannot be used in WHERE clause. "
               "Workaround: materialize intermediate result before filtering.",
        strict=True,
    )
    def test_shift_then_filter(self):
        """Test shift() followed by filter on shifted column."""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })

        # pandas
        pd_df = df.copy()
        pd_df['prev_value'] = pd_df['value'].shift(1)
        pd_result = pd_df[pd_df['prev_value'] > 15]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prev_value'] = ds_df['value'].shift(1)
        ds_result = ds_df[ds_df['prev_value'] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_then_arithmetic(self):
        """Test shift() followed by arithmetic operations."""
        df = pd.DataFrame({
            'price': [100.0, 105.0, 103.0, 110.0, 108.0]
        })

        # pandas
        pd_df = df.copy()
        pd_df['price_change'] = pd_df['price'] - pd_df['price'].shift(1)
        pd_result = pd_df.dropna()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['price_change'] = ds_df['price'] - ds_df['price'].shift(1)
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_negative_periods(self):
        """Test shift with negative periods (look ahead)."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'id': ['a', 'b', 'c', 'd', 'e']
        })

        # pandas
        pd_df = df.copy()
        pd_df['next_value'] = pd_df['value'].shift(-1)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['next_value'] = ds_df['value'].shift(-1)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_with_fill_value(self):
        """Test shift with fill_value parameter."""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['shifted'] = pd_df['value'].shift(1, fill_value=0)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['shifted'] = ds_df['value'].shift(1, fill_value=0)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_then_groupby(self):
        """Test shift() followed by groupby aggregation."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['prev_value'] = pd_df['value'].shift(1)
        pd_result = pd_df.groupby('group')['prev_value'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prev_value'] = ds_df['value'].shift(1)
        ds_result = ds_df.groupby('group')['prev_value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_shift_operations(self):
        """Test multiple shift operations in one DataFrame."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6]
        })

        # pandas
        pd_df = df.copy()
        pd_df['lag_1'] = pd_df['value'].shift(1)
        pd_df['lag_2'] = pd_df['value'].shift(2)
        pd_df['lead_1'] = pd_df['value'].shift(-1)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['lag_1'] = ds_df['value'].shift(1)
        ds_df['lag_2'] = ds_df['value'].shift(2)
        ds_df['lead_1'] = ds_df['value'].shift(-1)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 2: Diff Operations with Chains
# =============================================================================


class TestDiffChains:
    """Test diff() operations in various chain scenarios."""

    def test_diff_basic(self):
        """Test basic diff() operation."""
        df = pd.DataFrame({
            'value': [10, 15, 13, 20, 18]
        })

        # pandas
        pd_df = df.copy()
        pd_df['diff'] = pd_df['value'].diff()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['diff'] = ds_df['value'].diff()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_periods_2(self):
        """Test diff() with periods=2."""
        df = pd.DataFrame({
            'value': [100, 110, 105, 115, 120]
        })

        # pandas
        pd_df = df.copy()
        pd_df['diff_2'] = pd_df['value'].diff(periods=2)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['diff_2'] = ds_df['value'].diff(periods=2)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.xfail(
        reason="SQL limitation: window functions cannot be used in WHERE clause. "
               "Workaround: materialize intermediate result before filtering.",
        strict=True,
    )
    def test_diff_then_filter(self):
        """Test diff() followed by filter on diff result."""
        df = pd.DataFrame({
            'price': [100.0, 105.0, 103.0, 110.0, 108.0],
            'symbol': ['A', 'A', 'A', 'A', 'A']
        })

        # pandas
        pd_df = df.copy()
        pd_df['price_diff'] = pd_df['price'].diff()
        pd_result = pd_df[pd_df['price_diff'] > 0]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['price_diff'] = ds_df['price'].diff()
        ds_result = ds_df[ds_df['price_diff'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_then_abs(self):
        """Test diff() followed by abs() operation."""
        df = pd.DataFrame({
            'value': [10, 15, 12, 20, 18]
        })

        # pandas
        pd_df = df.copy()
        pd_df['abs_diff'] = pd_df['value'].diff().abs()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['abs_diff'] = ds_df['value'].diff().abs()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_cumsum_combination(self):
        """Test diff() and cumsum() combination for data validation."""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })

        # pandas: diff then cumsum should recover original (except first row)
        pd_df = df.copy()
        pd_df['reconstructed'] = pd_df['value'].diff().cumsum() + pd_df['value'].iloc[0]
        pd_result = pd_df.iloc[1:]  # Skip first row (NaN from diff)

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['reconstructed'] = ds_df['value'].diff().cumsum() + df['value'].iloc[0]
        ds_result = ds_df.iloc[1:]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 3: Pct_change Edge Cases
# =============================================================================


class TestPctChangeEdgeCases:
    """Test pct_change() edge cases."""

    def test_pct_change_basic(self):
        """Test basic pct_change() operation."""
        df = pd.DataFrame({
            'price': [100.0, 110.0, 105.0, 120.0]
        })

        # pandas
        pd_df = df.copy()
        pd_df['pct'] = pd_df['price'].pct_change()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['pct'] = ds_df['price'].pct_change()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change_with_zeros(self):
        """Test pct_change() with zero values (division by zero)."""
        df = pd.DataFrame({
            'value': [10.0, 0.0, 20.0, 0.0, 30.0]
        })

        # pandas
        pd_df = df.copy()
        pd_df['pct'] = pd_df['value'].pct_change()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['pct'] = ds_df['value'].pct_change()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change_then_filter(self):
        """Test pct_change() followed by filter."""
        df = pd.DataFrame({
            'price': [100.0, 105.0, 110.0, 108.0, 115.0],
            'symbol': ['X', 'X', 'X', 'X', 'X']
        })

        # pandas
        pd_df = df.copy()
        pd_df['return'] = pd_df['price'].pct_change()
        pd_result = pd_df[pd_df['return'] > 0.02]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['return'] = ds_df['price'].pct_change()
        ds_result = ds_df[ds_df['return'] > 0.02]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 4: Complex Assign Scenarios
# =============================================================================


class TestComplexAssign:
    """Test complex assign() scenarios."""

    def test_assign_referencing_computed_column(self):
        """Test assign where new column references previously computed column."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_df['d'] = pd_df['c'] * 2
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_df['d'] = ds_df['c'] * 2
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_chain_with_filter(self):
        """Test assign chain followed by filter on computed column."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['sum'] = pd_df['x'] + pd_df['y']
        pd_df['ratio'] = pd_df['x'] / pd_df['y']
        pd_result = pd_df[pd_df['sum'] > 25]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['sum'] = ds_df['x'] + ds_df['y']
        ds_df['ratio'] = ds_df['x'] / ds_df['y']
        ds_result = ds_df[ds_df['sum'] > 25]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_overwrite_column(self):
        """Test assign that overwrites an existing column."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [10, 20, 30, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['a'] = pd_df['a'] * 10  # Overwrite 'a'
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['a'] = ds_df['a'] * 10  # Overwrite 'a'
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_shift_reference(self):
        """Test assign using shift in computation."""
        df = pd.DataFrame({
            'value': [100, 110, 105, 115, 120]
        })

        # pandas
        pd_df = df.copy()
        pd_df['prev'] = pd_df['value'].shift(1)
        pd_df['change'] = pd_df['value'] - pd_df['prev']
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prev'] = ds_df['value'].shift(1)
        ds_df['change'] = ds_df['value'] - ds_df['prev']
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_interdependent_columns(self):
        """Test assign with multiple interdependent columns in chain."""
        df = pd.DataFrame({
            'base': [10, 20, 30, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['step1'] = pd_df['base'] * 2
        pd_df['step2'] = pd_df['step1'] + 5
        pd_df['step3'] = pd_df['step2'] / pd_df['base']
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['step1'] = ds_df['base'] * 2
        ds_df['step2'] = ds_df['step1'] + 5
        ds_df['step3'] = ds_df['step2'] / ds_df['base']
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 5: Window Function Results as Conditions
# =============================================================================


class TestWindowFunctionConditions:
    """Test using window function results as filter conditions."""

    def test_cumsum_as_filter_condition(self):
        """Test cumsum() result as filter condition."""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })

        # pandas
        pd_df = df.copy()
        pd_df['running_total'] = pd_df['value'].cumsum()
        pd_result = pd_df[pd_df['running_total'] > 50]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['running_total'] = ds_df['value'].cumsum()
        ds_result = ds_df[ds_df['running_total'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummax_as_filter_condition(self):
        """Test cummax() result as filter condition."""
        df = pd.DataFrame({
            'price': [100, 95, 110, 105, 120],
            'symbol': ['X', 'X', 'X', 'X', 'X']
        })

        # pandas
        pd_df = df.copy()
        pd_df['max_so_far'] = pd_df['price'].cummax()
        pd_result = pd_df[pd_df['price'] == pd_df['max_so_far']]  # New highs only

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['max_so_far'] = ds_df['price'].cummax()
        ds_result = ds_df[ds_df['price'] == ds_df['max_so_far']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummin_track_drawdown(self):
        """Test using cummin to track drawdown from peak."""
        df = pd.DataFrame({
            'value': [100, 110, 105, 115, 108, 120]
        })

        # pandas
        pd_df = df.copy()
        pd_df['peak'] = pd_df['value'].cummax()
        pd_df['trough'] = pd_df['value'].cummin()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['peak'] = ds_df['value'].cummax()
        ds_df['trough'] = ds_df['value'].cummin()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 6: Empty and Single-Row Edge Cases
# =============================================================================


class TestEmptySingleRowEdgeCases:
    """Test edge cases with empty or single-row DataFrames."""

    def test_shift_empty_dataframe(self):
        """Test shift on empty DataFrame."""
        df = pd.DataFrame({'value': pd.Series([], dtype=float)})

        # pandas
        pd_df = df.copy()
        pd_df['shifted'] = pd_df['value'].shift(1)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['shifted'] = ds_df['value'].shift(1)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_single_row(self):
        """Test diff on single-row DataFrame."""
        df = pd.DataFrame({'value': [100]})

        # pandas
        pd_df = df.copy()
        pd_df['diff'] = pd_df['value'].diff()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['diff'] = ds_df['value'].diff()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change_single_row(self):
        """Test pct_change on single-row DataFrame."""
        df = pd.DataFrame({'price': [100.0]})

        # pandas
        pd_df = df.copy()
        pd_df['pct'] = pd_df['price'].pct_change()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['pct'] = ds_df['price'].pct_change()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_empty_dataframe(self):
        """Test cumsum on empty DataFrame."""
        df = pd.DataFrame({'value': pd.Series([], dtype=float)})

        # pandas
        pd_df = df.copy()
        pd_df['cumsum'] = pd_df['value'].cumsum()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumsum'] = ds_df['value'].cumsum()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_single_row(self):
        """Test cumsum on single-row DataFrame."""
        df = pd.DataFrame({'value': [42]})

        # pandas
        pd_df = df.copy()
        pd_df['cumsum'] = pd_df['value'].cumsum()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumsum'] = ds_df['value'].cumsum()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 7: Nullable Types with Shift/Diff
# =============================================================================


class TestNullableTypeOperations:
    """Test shift/diff operations with nullable types."""

    @pytest.mark.xfail(
        reason="SQL window functions return float64, not nullable Int64. "
               "Pandas preserves nullable Int64 dtype for shift/diff on Int64 columns.",
        strict=True,
    )
    def test_shift_with_na_values(self):
        """Test shift on column with NA values."""
        df = pd.DataFrame({
            'value': pd.array([1, None, 3, None, 5], dtype=pd.Int64Dtype())
        })

        # pandas
        pd_df = df.copy()
        pd_df['shifted'] = pd_df['value'].shift(1)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['shifted'] = ds_df['value'].shift(1)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.xfail(
        reason="SQL window functions return float64, not nullable Int64. "
               "Pandas preserves nullable Int64 dtype for shift/diff on Int64 columns.",
        strict=True,
    )
    def test_diff_with_na_values(self):
        """Test diff on column with NA values."""
        df = pd.DataFrame({
            'value': pd.array([10, None, 30, None, 50], dtype=pd.Int64Dtype())
        })

        # pandas
        pd_df = df.copy()
        pd_df['diff'] = pd_df['value'].diff()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['diff'] = ds_df['value'].diff()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_with_na_values(self):
        """Test cumsum with NA values - pandas skipna behavior."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cumsum'] = pd_df['value'].cumsum()
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cumsum'] = ds_df['value'].cumsum()
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 8: Complex Chain Scenarios
# =============================================================================


class TestComplexChainScenarios:
    """Test complex multi-operation chain scenarios."""

    @pytest.mark.xfail(
        reason="SQL limitation: window functions cannot be used in WHERE clause. "
               "Workaround: materialize intermediate result before filtering.",
        strict=True,
    )
    def test_shift_diff_filter_sort_head(self):
        """Test chain: shift -> diff -> filter -> sort -> head."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'price': [100, 102, 99, 105, 103, 108, 110, 107, 112, 115]
        })

        # pandas
        pd_df = df.copy()
        pd_df['prev_price'] = pd_df['price'].shift(1)
        pd_df['price_diff'] = pd_df['price'].diff()
        pd_result = pd_df[pd_df['price_diff'] > 0].sort_values('price_diff', ascending=False).head(3)

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prev_price'] = ds_df['price'].shift(1)
        ds_df['price_diff'] = ds_df['price'].diff()
        ds_result = ds_df[ds_df['price_diff'] > 0].sort_values('price_diff', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_cumsum_groupby(self):
        """Test assign with cumsum followed by groupby."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_df = df.copy()
        pd_df['running_total'] = pd_df['value'].cumsum()
        pd_result = pd_df.groupby('category')['running_total'].max()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['running_total'] = ds_df['value'].cumsum()
        ds_result = ds_df.groupby('category')['running_total'].max()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_computed_columns_filter(self):
        """Test multiple computed columns with filter on any."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['sum_ab'] = pd_df['a'] + pd_df['b']
        pd_df['diff_ab'] = pd_df['b'] - pd_df['a']
        pd_df['ratio'] = pd_df['a'] / pd_df['b']
        pd_result = pd_df[(pd_df['sum_ab'] > 30) & (pd_df['diff_ab'] > 30)]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['sum_ab'] = ds_df['a'] + ds_df['b']
        ds_df['diff_ab'] = ds_df['b'] - ds_df['a']
        ds_df['ratio'] = ds_df['a'] / ds_df['b']
        ds_result = ds_df[(ds_df['sum_ab'] > 30) & (ds_df['diff_ab'] > 30)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_then_shift_then_compare(self):
        """Test rolling mean followed by shift for comparison.
        
        This tests NULL-safe comparison: pandas NaN > NaN returns False.
        DataStore now uses ifNull() wrapping to match pandas behavior.
        """
        df = pd.DataFrame({
            'value': [10, 15, 12, 18, 20, 16, 22, 25]
        })

        # pandas
        pd_df = df.copy()
        pd_df['ma3'] = pd_df['value'].rolling(3).mean()
        pd_df['ma3_prev'] = pd_df['ma3'].shift(1)
        pd_df['trend_up'] = pd_df['ma3'] > pd_df['ma3_prev']
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['ma3'] = ds_df['value'].rolling(3).mean()
        ds_df['ma3_prev'] = ds_df['ma3'].shift(1)
        ds_df['trend_up'] = ds_df['ma3'] > ds_df['ma3_prev']
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change_cumsum_combination(self):
        """Test pct_change and cumsum for cumulative returns calculation."""
        df = pd.DataFrame({
            'price': [100.0, 105.0, 103.0, 108.0, 110.0]
        })

        # pandas
        pd_df = df.copy()
        pd_df['daily_return'] = pd_df['price'].pct_change()
        pd_df['cum_return'] = (1 + pd_df['daily_return']).cumprod() - 1
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['daily_return'] = ds_df['price'].pct_change()
        ds_df['cum_return'] = (1 + ds_df['daily_return']).cumprod() - 1
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_filter_sort(self):
        """Test filter -> assign -> filter -> sort chain."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 15, 30, 25, 12, 35, 28],
            'flag': [True, False, True, True, False, False, True, True]
        })

        # pandas
        pd_df = df[df['flag']].copy()
        pd_df['value_doubled'] = pd_df['value'] * 2
        pd_result = pd_df[pd_df['value_doubled'] > 40].sort_values('value_doubled')

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df = ds_df[ds_df['flag']]
        ds_df['value_doubled'] = ds_df['value'] * 2
        ds_result = ds_df[ds_df['value_doubled'] > 40].sort_values('value_doubled')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 9: Edge Cases for Computed Column SQL Pushdown
# =============================================================================


class TestComputedColumnSQLPushdown:
    """Test computed column scenarios for SQL pushdown optimization."""

    def test_computed_column_in_where_same_op(self):
        """Test computed column used in where clause of same operation chain."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['z'] = pd_df['x'] * pd_df['y']
        pd_result = pd_df[pd_df['z'] > 50]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['z'] = ds_df['x'] * ds_df['y']
        ds_result = ds_df[ds_df['z'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_computed_column_groupby_key(self):
        """Test computed column as groupby key."""
        df = pd.DataFrame({
            'value': [10, 25, 15, 35, 20, 45],
            'amount': [100, 200, 150, 350, 200, 450]
        })

        # pandas
        pd_df = df.copy()
        pd_df['bucket'] = (pd_df['value'] // 20).astype(int)
        pd_result = pd_df.groupby('bucket')['amount'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['bucket'] = (ds_df['value'] // 20).astype(int)
        ds_result = ds_df.groupby('bucket')['amount'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_computed_column_sort_key(self):
        """Test computed column as sort key."""
        df = pd.DataFrame({
            'first': [10, 20, 30],
            'second': [5, 15, 10]
        })

        # pandas
        pd_df = df.copy()
        pd_df['ratio'] = pd_df['first'] / pd_df['second']
        pd_result = pd_df.sort_values('ratio')

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['ratio'] = ds_df['first'] / ds_df['second']
        ds_result = ds_df.sort_values('ratio')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 10: String Column Operations with Shift
# =============================================================================


class TestStringColumnShift:
    """Test shift operations on string columns."""

    def test_shift_string_column(self):
        """Test shift on string column."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'value': [1, 2, 3, 4]
        })

        # pandas
        pd_df = df.copy()
        pd_df['prev_name'] = pd_df['name'].shift(1)
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prev_name'] = ds_df['name'].shift(1)
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_string_compare_with_current(self):
        """Test comparing shifted string with current.
        
        This tests NULL-safe comparison: pandas 'string' != None returns True.
        DataStore now uses ifNull() wrapping to match pandas behavior.
        """
        df = pd.DataFrame({
            'status': ['pending', 'pending', 'approved', 'approved', 'rejected']
        })

        # pandas
        pd_df = df.copy()
        pd_df['prev_status'] = pd_df['status'].shift(1)
        pd_df['status_changed'] = pd_df['status'] != pd_df['prev_status']
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['prev_status'] = ds_df['status'].shift(1)
        ds_df['status_changed'] = ds_df['status'] != ds_df['prev_status']
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
