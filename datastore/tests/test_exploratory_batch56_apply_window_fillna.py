"""
Exploratory Batch 56: Apply/Applymap Edge Cases, Window Boundaries, Fillna/Dropna Parameters

Focus areas:
1. apply() and applymap() with various function types and parameters
2. Rolling/Expanding/EWM window boundary conditions (window=1, window=len(df))
3. Fillna/Dropna unusual parameter combinations and conflicts
4. Comparison operators with type mismatches
5. Accessor method chaining with type transitions
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import (
    limit_groupby_series_param,
    chdb_median_in_where,
    pandas_version_no_dataframe_map,
)


# ========== Apply/Applymap Edge Cases ==========

class TestApplyEdgeCases:
    """Test apply() with various function types and parameters."""

    def test_apply_lambda_scalar_return_axis0(self):
        """apply with lambda returning scalar, axis=0 (column-wise)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: x.sum(), axis=0)
        ds_result = ds_df.apply(lambda x: x.sum(), axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_lambda_scalar_return_axis1(self):
        """apply with lambda returning scalar, axis=1 (row-wise)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: x.sum(), axis=1)
        ds_result = ds_df.apply(lambda x: x.sum(), axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_lambda_series_return(self):
        """apply with lambda returning Series."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: pd.Series({'min': x.min(), 'max': x.max()}))
        ds_result = ds_df.apply(lambda x: pd.Series({'min': x.min(), 'max': x.max()}))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_with_raw_true_numeric(self):
        """apply with raw=True on numeric columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        # raw=True passes numpy arrays instead of Series
        pd_result = pd_df.apply(lambda x: x.sum(), axis=0, raw=True)
        ds_result = ds_df.apply(lambda x: x.sum(), axis=0, raw=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_with_raw_true_axis1(self):
        """apply with raw=True, axis=1."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(np.mean, axis=1, raw=True)
        ds_result = ds_df.apply(np.mean, axis=1, raw=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_returns_none_values(self):
        """apply with function that returns None for some rows."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        def conditional_return(x):
            return x['a'] if x['a'] > 2 else None

        pd_result = pd_df.apply(conditional_return, axis=1)
        ds_result = ds_df.apply(conditional_return, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_chained_with_filter(self):
        """apply chained after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 1].apply(lambda x: x.sum(), axis=0)
        ds_result = ds_df[ds_df['a'] > 1].apply(lambda x: x.sum(), axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_chained_with_sort(self):
        """apply chained after sort."""
        pd_df = pd.DataFrame({'a': [3, 1, 2], 'b': [30, 10, 20]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a').apply(lambda x: x.cumsum())
        ds_result = ds_df.sort_values('a').apply(lambda x: x.cumsum())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_single_column_dataframe(self):
        """apply on single-column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: x * 2)
        ds_result = ds_df.apply(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_empty_dataframe(self):
        """apply on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: x.sum())
        ds_result = ds_df.apply(lambda x: x.sum())

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestApplymapEdgeCases:
    """Test applymap() / map() with various scenarios.

    Note: DataFrame.map() was added in pandas 2.1.0. In pandas 2.0.x only applymap() exists.
    These tests use map() and are skipped on older pandas versions.
    """

    @pandas_version_no_dataframe_map
    def test_applymap_basic(self):
        """Basic applymap/map functionality."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.map(lambda x: x * 2)
        ds_result = ds_df.map(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_no_dataframe_map
    def test_applymap_type_change(self):
        """applymap that changes type (int -> str)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.map(str)
        ds_result = ds_df.map(str)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_no_dataframe_map
    def test_applymap_with_na_handling(self):
        """applymap with NA values."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.map(lambda x: x * 2 if pd.notna(x) else -1)
        ds_result = ds_df.map(lambda x: x * 2 if pd.notna(x) else -1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_no_dataframe_map
    def test_applymap_chained_operations(self):
        """applymap in chained operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.map(lambda x: x + 1).sum()
        ds_result = ds_df.map(lambda x: x + 1).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_no_dataframe_map
    def test_applymap_string_columns(self):
        """applymap on string columns."""
        pd_df = pd.DataFrame({'a': ['abc', 'def', 'ghi'], 'b': ['xyz', 'uvw', 'rst']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.map(lambda x: x.upper())
        ds_result = ds_df.map(lambda x: x.upper())

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Rolling/Expanding Window Boundary Conditions ==========

class TestRollingBoundaries:
    """Test rolling window with boundary conditions."""

    def test_rolling_window_1(self):
        """rolling with window=1 (identity-like operation)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=1).mean()
        ds_result = ds_df['a'].rolling(window=1).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_window_equals_length(self):
        """rolling with window equal to DataFrame length."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=5).mean()
        ds_result = ds_df['a'].rolling(window=5).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_window_greater_than_length(self):
        """rolling with window > DataFrame length (all NaN except maybe last)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=10).mean()
        ds_result = ds_df['a'].rolling(window=10).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_min_periods_1(self):
        """rolling with min_periods=1 (fill from start)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        # min_periods=1 allows partial windows
        pd_result = pd_df['a'].rolling(window=3, min_periods=1).sum()
        ds_result = ds_df['a'].rolling(window=3, min_periods=1).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_center_true_odd_window(self):
        """rolling with center=True and odd window size."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=3, center=True).mean()
        ds_result = ds_df['a'].rolling(window=3, center=True).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_center_true_even_window(self):
        """rolling with center=True and even window size."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=4, center=True).mean()
        ds_result = ds_df['a'].rolling(window=4, center=True).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_single_row_dataframe(self):
        """rolling on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=3).sum()
        ds_result = ds_df['a'].rolling(window=3).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_multiple_aggregations(self):
        """rolling with multiple aggregations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        for agg in ['sum', 'mean', 'std', 'min', 'max', 'count']:
            pd_result = getattr(pd_df['a'].rolling(window=2), agg)()
            ds_result = getattr(ds_df['a'].rolling(window=2), agg)()
            assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_chained_with_filter(self):
        """rolling chained after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 1]['a'].rolling(window=2).mean()
        ds_result = ds_df[ds_df['a'] > 1]['a'].rolling(window=2).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestExpandingBoundaries:
    """Test expanding window with boundary conditions."""

    def test_expanding_basic(self):
        """Basic expanding operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].expanding().mean()
        ds_result = ds_df['a'].expanding().mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_min_periods_greater_than_data(self):
        """expanding with min_periods > available data."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].expanding(min_periods=5).mean()
        ds_result = ds_df['a'].expanding(min_periods=5).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_min_periods_1(self):
        """expanding with min_periods=1."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].expanding(min_periods=1).sum()
        ds_result = ds_df['a'].expanding(min_periods=1).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_single_row(self):
        """expanding on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].expanding().mean()
        ds_result = ds_df['a'].expanding().mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_with_nan(self):
        """expanding with NaN values."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3, 4, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].expanding().sum()
        ds_result = ds_df['a'].expanding().sum()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEWMBoundaries:
    """Test EWM (exponential weighted) boundary conditions."""

    def test_ewm_span_1(self):
        """ewm with span=1 (all weight on current value)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].ewm(span=1, adjust=False).mean()
        ds_result = ds_df['a'].ewm(span=1, adjust=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_span_2(self):
        """ewm with span=2."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].ewm(span=2).mean()
        ds_result = ds_df['a'].ewm(span=2).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_large_span(self):
        """ewm with span much larger than data size."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].ewm(span=100).mean()
        ds_result = ds_df['a'].ewm(span=100).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_single_value(self):
        """ewm on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].ewm(span=3).mean()
        ds_result = ds_df['a'].ewm(span=3).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_alpha_parameter(self):
        """ewm with alpha parameter instead of span."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].ewm(alpha=0.5).mean()
        ds_result = ds_df['a'].ewm(alpha=0.5).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Fillna/Dropna Parameter Edge Cases ==========

class TestFillnaEdgeCases:
    """Test fillna with unusual parameter combinations."""

    def test_fillna_limit_1(self):
        """fillna with limit=1."""
        pd_df = pd.DataFrame({'a': [1, np.nan, np.nan, np.nan, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.ffill(limit=1)
        ds_result = ds_df.ffill(limit=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_limit_large(self):
        """fillna with limit larger than gap."""
        pd_df = pd.DataFrame({'a': [1, np.nan, np.nan, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.ffill(limit=10)
        ds_result = ds_df.ffill(limit=10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_all_nan_column(self):
        """fillna on column with all NaN values."""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, np.nan], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict_per_column(self):
        """fillna with dict specifying different values per column."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 5, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.fillna({'a': -1, 'b': -2})
        ds_result = ds_df.fillna({'a': -1, 'b': -2})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict_partial_columns(self):
        """fillna with dict specifying only some columns."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 5, np.nan], 'c': [np.nan, np.nan, 9]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.fillna({'a': -1})  # Only fill column 'a'
        ds_result = ds_df.fillna({'a': -1})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bfill_basic(self):
        """bfill (backward fill) basic."""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.bfill()
        ds_result = ds_df.bfill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ffill_basic(self):
        """ffill (forward fill) basic."""
        pd_df = pd.DataFrame({'a': [1, 2, np.nan, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.ffill()
        ds_result = ds_df.ffill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_chained_with_filter(self):
        """fillna chained after filter."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['b'] > 15].fillna(0)
        ds_result = ds_df[ds_df['b'] > 15].fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDropnaEdgeCases:
    """Test dropna with unusual parameter combinations."""

    def test_dropna_how_all(self):
        """dropna with how='all' (only drop if all values are NA)."""
        pd_df = pd.DataFrame({'a': [1, np.nan, np.nan], 'b': [np.nan, np.nan, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna(how='all')
        ds_result = ds_df.dropna(how='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_how_all_with_all_nan_row(self):
        """dropna with how='all' when a row is all NaN."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [2, np.nan, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna(how='all')
        ds_result = ds_df.dropna(how='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_thresh(self):
        """dropna with thresh parameter."""
        pd_df = pd.DataFrame({
            'a': [1, np.nan, np.nan],
            'b': [2, np.nan, 3],
            'c': [3, 4, np.nan]
        })
        ds_df = DataStore(pd_df)

        # Keep rows with at least 2 non-NA values
        pd_result = pd_df.dropna(thresh=2)
        ds_result = ds_df.dropna(thresh=2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset(self):
        """dropna with subset parameter."""
        pd_df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [np.nan, 5, 6],
            'c': [7, 8, np.nan]
        })
        ds_df = DataStore(pd_df)

        # Only consider columns a and b
        pd_result = pd_df.dropna(subset=['a', 'b'])
        ds_result = ds_df.dropna(subset=['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all_nan_dataframe(self):
        """dropna on DataFrame with all NaN."""
        pd_df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_no_nan(self):
        """dropna on DataFrame with no NaN values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_chained_with_sort(self):
        """dropna chained with sort."""
        pd_df = pd.DataFrame({'a': [3, np.nan, 1, 2], 'b': ['z', 'y', 'x', 'w']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna().sort_values('a')
        ds_result = ds_df.dropna().sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Comparison Operators with Type Mismatches ==========

class TestComparisonTypeMismatch:
    """Test comparison operators with type mismatches."""

    def test_compare_int_to_float(self):
        """Compare int column to float value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2.5]
        ds_result = ds_df[ds_df['a'] > 2.5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_float_to_int(self):
        """Compare float column to int value."""
        pd_df = pd.DataFrame({'a': [1.5, 2.5, 3.5, 4.5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_column_to_column(self):
        """Compare two columns of different types."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [1.5, 2.5, 2.5, 3.5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > pd_df['b']]
        ds_result = ds_df[ds_df['a'] > ds_df['b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_null_comparison_with_isna(self):
        """Compare using isna() vs == None."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'].isna()]
        ds_result = ds_df[ds_df['a'].isna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_null_comparison_with_notna(self):
        """Compare using notna()."""
        pd_df = pd.DataFrame({'a': [1, np.nan, 3, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_with_precomputed_scalar(self):
        """Compare column to precomputed scalar value (correct approach).
        
        When using an aggregation result in a filter condition, force execution
        first by converting to a Python scalar (float/int).
        """
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        # Correct approach: execute aggregation first
        pd_mean = pd_df['a'].mean()
        ds_mean = float(ds_df['a'].mean())  # Force execution to scalar

        pd_result = pd_df[pd_df['a'] > pd_mean]
        ds_result = ds_df[ds_df['a'] > ds_mean]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_with_lazy_aggregation(self):
        """Compare column to aggregation result.
        
        This now works because ds_df['col'].mean() returns scalar (matching pandas),
        avoiding the SQL aggregation in WHERE clause issue.
        """
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        # This generates invalid SQL: SELECT ... WHERE a > avg(a)
        pd_result = pd_df[pd_df['a'] > pd_df['a'].mean()]
        ds_result = ds_df[ds_df['a'] > ds_df['a'].mean()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_eq_with_different_types(self):
        """Test equality comparison with different types."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] == pd_df['b']]
        ds_result = ds_df[ds_df['a'] == ds_df['b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compare_ne_with_scalar(self):
        """Test not-equal comparison."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 2, 1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] != 2]
        ds_result = ds_df[ds_df['a'] != 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_comparisons(self):
        """Test chained comparisons (a < b < c style via &)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['a'] < 5)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['a'] < 5)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Accessor Method Chaining with Type Transitions ==========

class TestAccessorChaining:
    """Test accessor method chaining with type transitions."""

    def test_str_upper_then_filter(self):
        """String accessor then filter."""
        pd_df = pd.DataFrame({'a': ['abc', 'def', 'ghi'], 'b': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_df['upper'] = pd_df['a'].str.upper()
        ds_df['upper'] = ds_df['a'].str.upper()

        pd_result = pd_df[pd_df['b'] > 1]
        ds_result = ds_df[ds_df['b'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_then_filter(self):
        """String length then filter on length."""
        pd_df = pd.DataFrame({'a': ['a', 'ab', 'abc', 'abcd']})
        ds_df = DataStore(pd_df)

        pd_df['len'] = pd_df['a'].str.len()
        ds_df['len'] = ds_df['a'].str.len()

        pd_result = pd_df[pd_df['len'] > 2]
        ds_result = ds_df[ds_df['len'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_year_then_groupby_via_column(self):
        """Datetime year extraction then groupby (via assigned column)."""
        dates = pd.to_datetime(['2020-01-01', '2020-06-15', '2021-03-10', '2021-12-31'])
        pd_df = pd.DataFrame({'date': dates, 'value': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df)

        # Assign to column first then groupby by column name
        pd_df['year'] = pd_df['date'].dt.year
        ds_df['year'] = ds_df['date'].dt.year

        pd_result = pd_df.groupby('year')['value'].sum()
        ds_result = ds_df.groupby('year')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    @limit_groupby_series_param
    def test_dt_year_then_groupby_direct(self):
        """Datetime year extraction then groupby (direct - pandas style).

        DataStore limitation: groupby does not support direct ColumnExpr/Series parameter.
        Must use column name instead. Use the pattern:
            ds_df['year'] = ds_df['date'].dt.year
            ds_df.groupby('year')['value'].sum()
        """
        dates = pd.to_datetime(['2020-01-01', '2020-06-15', '2021-03-10', '2021-12-31'])
        pd_df = pd.DataFrame({'date': dates, 'value': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df)

        # Pandas allows direct Series in groupby
        pd_result = pd_df.groupby(pd_df['date'].dt.year)['value'].sum()
        ds_result = ds_df.groupby(ds_df['date'].dt.year)['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_month_then_filter(self):
        """Datetime month extraction then filter."""
        dates = pd.to_datetime(['2020-01-01', '2020-06-15', '2020-03-10', '2020-12-31'])
        pd_df = pd.DataFrame({'date': dates, 'value': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['date'].dt.month > 6]
        ds_result = ds_df[ds_df['date'].dt.month > 6]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_then_count(self):
        """String contains then count."""
        pd_df = pd.DataFrame({'a': ['apple', 'banana', 'apricot', 'berry']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'].str.contains('a')].shape[0]
        ds_result = len(ds_df[ds_df['a'].str.contains('a')])

        assert pd_result == ds_result

    def test_str_split_first_element(self):
        """String split and get first element."""
        pd_df = pd.DataFrame({'a': ['a-b', 'c-d-e', 'f-g']})
        ds_df = DataStore(pd_df)

        pd_df['first'] = pd_df['a'].str.split('-').str[0]
        ds_df['first'] = ds_df['a'].str.split('-').str[0]

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_multiple_str_operations(self):
        """Multiple string operations chained."""
        pd_df = pd.DataFrame({'a': ['  abc  ', '  DEF  ', '  ghi  ']})
        ds_df = DataStore(pd_df)

        pd_df['processed'] = pd_df['a'].str.strip().str.upper()
        ds_df['processed'] = ds_df['a'].str.strip().str.upper()

        assert_datastore_equals_pandas(ds_df, pd_df)


# ========== Edge Cases: Empty and Single-Row DataFrames ==========

class TestEmptySingleRowDataFrame:
    """Test operations on empty and single-row DataFrames."""

    def test_empty_df_groupby(self):
        """groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='float64')})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('a')['b'].sum()
        ds_result = ds_df.groupby('a')['b'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_sort(self):
        """sort on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64'), 'b': pd.Series([], dtype='str')})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_assign(self):
        """assign on empty DataFrame."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype='int64')})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_aggregation(self):
        """Aggregation on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [5], 'b': [10]})
        ds_df = DataStore(pd_df)

        for agg in ['sum', 'mean', 'min', 'max', 'std', 'var']:
            pd_result = getattr(pd_df, agg)()
            ds_result = getattr(ds_df, agg)()
            # Note: std/var on single value returns NaN in pandas
            assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_rolling(self):
        """Rolling on single-row DataFrame with window > 1."""
        pd_df = pd.DataFrame({'a': [5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].rolling(window=3).sum()
        ds_result = ds_df['a'].rolling(window=3).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_match(self):
        """Filter on single-row DataFrame that matches."""
        pd_df = pd.DataFrame({'a': [5], 'b': ['x']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self):
        """Filter on single-row DataFrame that doesn't match."""
        pd_df = pd.DataFrame({'a': [5], 'b': ['x']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Complex Chain Operations ==========

class TestComplexChains:
    """Test complex operation chains."""

    def test_filter_apply_sort_head(self):
        """Filter -> Apply -> Sort -> Head chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 1].apply(lambda x: x * 2).sort_values('a').head(3)
        ds_result = ds_df[ds_df['a'] > 1].apply(lambda x: x * 2).sort_values('a').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_fillna_groupby_agg(self):
        """Assign -> Fillna -> Groupby -> Agg chain."""
        pd_df = pd.DataFrame({'a': ['x', 'x', 'y', 'y'], 'b': [1, np.nan, 3, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(c=lambda x: x['b'] * 2).fillna(0).groupby('a')['c'].sum()
        ds_result = ds_df.assign(c=lambda x: x['b'] * 2).fillna(0).groupby('a')['c'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_sort_tail(self):
        """Dropna -> Sort -> Tail chain."""
        pd_df = pd.DataFrame({'a': [3, np.nan, 1, 2, np.nan], 'b': ['z', 'y', 'x', 'w', 'v']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.dropna().sort_values('a').tail(2)
        ds_result = ds_df.dropna().sort_values('a').tail(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_filter_assign(self):
        """Rolling -> Filter on rolling result -> Assign."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_df['rolling_mean'] = pd_df['a'].rolling(window=2).mean()
        ds_df['rolling_mean'] = ds_df['a'].rolling(window=2).mean()

        pd_result = pd_df[pd_df['rolling_mean'] > 2]
        ds_result = ds_df[ds_df['rolling_mean'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_and_sorts(self):
        """Multiple filters and sorts interleaved."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': ['x', 'y', 'x', 'y', 'x', 'y'],
            'c': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 1].sort_values('c', ascending=False)[pd_df['b'] == 'x'].head(2)
        ds_result = ds_df[ds_df['a'] > 1].sort_values('c', ascending=False)[ds_df['b'] == 'x'].head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
