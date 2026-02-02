"""
Exploratory Batch 42: Statistical Operations + Reshape + Complex Chains

Focus areas:
1. Correlation/Covariance + chains (corr, cov, corrwith)
2. Rank, Skew, Kurtosis with filters and groupby
3. Replace + transform chains with various patterns
4. Transpose/axis operations (T, transpose)
5. Mode operations with different data scenarios
6. Interpolate with different methods and chains
7. select_dtypes combinations
8. xs (cross-section) edge cases
9. Complex numeric reduction chains

Test design rationale:
- Mirror Code Pattern: Each test mirrors pandas operations exactly
- Chain Focus: Tests complex operation chains less covered
- Edge Cases: Empty DataFrames, single rows, mixed types
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal
from tests.xfail_markers import (
    chdb_integer_column_names,
)


# =============================================================================
# Section 1: Correlation and Covariance Chains
# =============================================================================


class TestCorrelationCovarianceChains:
    """Tests for correlation/covariance with chains."""

    def test_corr_basic(self):
        """Basic correlation matrix."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        pd_result = pd_df.corr()

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        ds_result = ds_df.corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_corr_after_filter(self):
        """Correlation after filtering rows."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'B': [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            'flag': [True, False, True, False, True, False]
        })
        pd_result = pd_df[pd_df['flag']][['A', 'B', 'C']].corr()

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'B': [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            'flag': [True, False, True, False, True, False]
        })
        ds_result = ds_df[ds_df['flag']][['A', 'B', 'C']].corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_corr_with_nan(self):
        """Correlation with NaN values."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, np.nan, 2.0, 1.0],
            'C': [2.0, 4.0, 6.0, np.nan, 10.0]
        })
        pd_result = pd_df.corr()

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, np.nan, 2.0, 1.0],
            'C': [2.0, 4.0, 6.0, np.nan, 10.0]
        })
        ds_result = ds_df.corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cov_basic(self):
        """Basic covariance matrix."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        pd_result = pd_df.cov()

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        ds_result = ds_df.cov()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cov_after_dropna(self):
        """Covariance after dropping NA values."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 4.0, np.nan],
            'B': [5.0, 4.0, 3.0, np.nan, 1.0],
            'C': [np.nan, 3.0, 4.0, 5.0, 6.0]
        })
        pd_result = pd_df.dropna().cov()

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, 4.0, np.nan],
            'B': [5.0, 4.0, 3.0, np.nan, 1.0],
            'C': [np.nan, 3.0, 4.0, 5.0, 6.0]
        })
        ds_result = ds_df.dropna().cov()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_corrwith_series(self):
        """Correlation with a Series."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        pd_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pd_result = pd_df.corrwith(pd_series)

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        ds_result = ds_df.corrwith(pd_series)

        # corrwith returns Series
        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Section 2: Rank, Skew, Kurtosis Chains
# =============================================================================


class TestRankSkewKurtChains:
    """Tests for rank, skew, kurtosis with chain operations."""

    def test_rank_basic(self):
        """Basic rank operation."""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        pd_result = pd_df.rank()

        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        ds_result = ds_df.rank()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_method_min(self):
        """Rank with method='min' for ties."""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        pd_result = pd_df.rank(method='min')

        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        ds_result = ds_df.rank(method='min')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_then_filter(self):
        """Rank followed by filter based on rank."""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        pd_df_ranked = pd_df.copy()
        pd_df_ranked['rank_A'] = pd_df['A'].rank()
        pd_result = pd_df_ranked[pd_df_ranked['rank_A'] <= 2]

        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        ds_df_ranked = ds_df.copy()
        ds_df_ranked['rank_A'] = ds_df['A'].rank()
        ds_result = ds_df_ranked[ds_df_ranked['rank_A'] <= 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_pct(self):
        """Rank with pct=True (percentile rank)."""
        pd_df = pd.DataFrame({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        pd_result = pd_df.rank(pct=True)

        ds_df = DataStore({
            'A': [3, 1, 4, 1, 5],
            'B': [9, 2, 6, 5, 3]
        })
        ds_result = ds_df.rank(pct=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_skew_basic(self):
        """Basic skewness calculation."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],
            'B': [1, 1, 1, 1, 1, 1]
        })
        pd_result = pd_df.skew()

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5, 100],
            'B': [1, 1, 1, 1, 1, 1]
        })
        ds_result = ds_df.skew()

        # skew returns Series
        assert_series_equal(ds_result, pd_result)

    def test_skew_after_filter(self):
        """Skewness after filtering."""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100, 200],
            'group': ['A', 'A', 'A', 'B', 'B', 'B', 'B']
        })
        pd_result = pd_df[pd_df['group'] == 'A'][['value']].skew()

        ds_df = DataStore({
            'value': [1, 2, 3, 4, 5, 100, 200],
            'group': ['A', 'A', 'A', 'B', 'B', 'B', 'B']
        })
        ds_result = ds_df[ds_df['group'] == 'A'][['value']].skew()

        assert_series_equal(ds_result, pd_result)

    def test_kurt_basic(self):
        """Basic kurtosis calculation."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],
            'B': [1, 2, 3, 4, 5, 6]
        })
        pd_result = pd_df.kurt()

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5, 100],
            'B': [1, 2, 3, 4, 5, 6]
        })
        ds_result = ds_df.kurt()

        assert_series_equal(ds_result, pd_result)

    def test_kurt_after_dropna(self):
        """Kurtosis after dropping NA."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, 4.0, 5.0, 100.0],
            'B': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
        })
        pd_result = pd_df.dropna().kurt()

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, 4.0, 5.0, 100.0],
            'B': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]
        })
        ds_result = ds_df.dropna().kurt()

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Section 3: Replace + Transform Chains
# =============================================================================


class TestReplaceTransformChains:
    """Tests for replace with various patterns and transform chains."""

    def test_replace_single_value(self):
        """Replace single value."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 2, 1],
            'B': ['x', 'y', 'z', 'y', 'x']
        })
        pd_result = pd_df.replace(2, 99)

        ds_df = DataStore({
            'A': [1, 2, 3, 2, 1],
            'B': ['x', 'y', 'z', 'y', 'x']
        })
        ds_result = ds_df.replace(2, 99)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict(self):
        """Replace with dictionary mapping."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 2, 1],
            'B': ['x', 'y', 'z', 'y', 'x']
        })
        pd_result = pd_df.replace({'x': 'X', 'y': 'Y'})

        ds_df = DataStore({
            'A': [1, 2, 3, 2, 1],
            'B': ['x', 'y', 'z', 'y', 'x']
        })
        ds_result = ds_df.replace({'x': 'X', 'y': 'Y'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_then_filter(self):
        """Replace followed by filter."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 2, 1],
            'B': [10, 20, 30, 20, 10]
        })
        pd_result = pd_df.replace(2, 99).query('A > 10')

        ds_df = DataStore({
            'A': [1, 2, 3, 2, 1],
            'B': [10, 20, 30, 20, 10]
        })
        ds_result = ds_df.replace(2, 99).query('A > 10')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_then_groupby_sum(self):
        """Replace followed by groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B', 'A'],
            'value': [-1, 2, 3, -1, 5]
        })
        # Replace -1 with 0, then group
        pd_result = pd_df.replace(-1, 0).groupby('group')['value'].sum().reset_index()

        ds_df = DataStore({
            'group': ['A', 'B', 'A', 'B', 'A'],
            'value': [-1, 2, 3, -1, 5]
        })
        ds_result = ds_df.replace(-1, 0).groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_to_value(self):
        """Replace list of values with single value."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.replace([1, 2, 3], 0)

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_result = ds_df.replace([1, 2, 3], 0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_na(self):
        """Replace NA values."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': ['x', None, 'z', None, 'w']
        })
        pd_result = pd_df.replace({np.nan: 0, None: 'missing'})

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': ['x', None, 'z', None, 'w']
        })
        ds_result = ds_df.replace({np.nan: 0, None: 'missing'})

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 4: Transpose and Axis Operations
# =============================================================================


class TestTransposeAxisOperations:
    """Tests for transpose and axis operations."""

    def test_transpose_basic(self):
        """Basic transpose operation."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        pd_result = pd_df.T

        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        ds_result = ds_df.T

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transpose_method(self):
        """Transpose using method."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        pd_result = pd_df.transpose()

        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        ds_result = ds_df.transpose()

        assert_datastore_equals_pandas(ds_result, pd_result)

    # FIXED: Integer column names now work via string conversion
    def test_transpose_then_filter_columns(self):
        """Transpose then select columns.

        Note: Transpose creates integer column names (0, 1, 2...).
        chDB Python() table function cannot handle integer column names,
        causing a KeyError when trying to access them.
        """
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        pd_result = pd_df.T[[0, 1]]

        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        ds_result = ds_df.T[[0, 1]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transpose_mixed_types(self):
        """Transpose with mixed column types."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        pd_result = pd_df.T

        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        ds_result = ds_df.T

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 5: Mode Operations
# =============================================================================


class TestModeOperations:
    """Tests for mode operations with various scenarios."""

    def test_mode_basic(self):
        """Basic mode operation."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 2, 3, 2],
            'B': [1, 1, 2, 2, 3]
        })
        pd_result = pd_df.mode()

        ds_df = DataStore({
            'A': [1, 2, 2, 3, 2],
            'B': [1, 1, 2, 2, 3]
        })
        ds_result = ds_df.mode()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mode_multimodal(self):
        """Mode with multiple modes (multimodal data)."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2, 3],  # bimodal: 1 and 2
            'B': [1, 2, 3, 4, 5]   # all unique
        })
        pd_result = pd_df.mode()

        ds_df = DataStore({
            'A': [1, 1, 2, 2, 3],
            'B': [1, 2, 3, 4, 5]
        })
        ds_result = ds_df.mode()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mode_with_na(self):
        """Mode with NA values."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 2.0, 2.0, np.nan],
            'B': [1.0, 1.0, np.nan, 1.0, 2.0]
        })
        pd_result = pd_df.mode(dropna=True)

        ds_df = DataStore({
            'A': [1.0, np.nan, 2.0, 2.0, np.nan],
            'B': [1.0, 1.0, np.nan, 1.0, 2.0]
        })
        ds_result = ds_df.mode(dropna=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mode_string_column(self):
        """Mode with string column."""
        pd_df = pd.DataFrame({
            'A': ['x', 'y', 'x', 'z', 'x'],
            'B': [1, 2, 1, 2, 1]
        })
        pd_result = pd_df.mode()

        ds_df = DataStore({
            'A': ['x', 'y', 'x', 'z', 'x'],
            'B': [1, 2, 1, 2, 1]
        })
        ds_result = ds_df.mode()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 6: Interpolate Operations
# =============================================================================


class TestInterpolateOperations:
    """Tests for interpolate with different methods."""

    def test_interpolate_linear_basic(self):
        """Basic linear interpolation."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, np.nan, 40.0, 50.0]
        })
        pd_result = pd_df.interpolate(method='linear')

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, np.nan, 40.0, 50.0]
        })
        ds_result = ds_df.interpolate(method='linear')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_then_filter(self):
        """Interpolate followed by filter."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        pd_result = pd_df.interpolate().query('A > 2')

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        ds_result = ds_df.interpolate().query('A > 2')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_ffill(self):
        """Forward fill interpolation."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, 4.0, np.nan],
            'B': [10.0, np.nan, 30.0, np.nan, 50.0]
        })
        # pandas 3.0 removed interpolate(method='ffill'), use ffill() instead
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        if pandas_version >= (3, 0):
            pd_result = pd_df.ffill()
        else:
            pd_result = pd_df.interpolate(method='ffill')

        ds_df = DataStore({
            'A': [1.0, np.nan, np.nan, 4.0, np.nan],
            'B': [10.0, np.nan, 30.0, np.nan, 50.0]
        })
        if pandas_version >= (3, 0):
            ds_result = ds_df.ffill()
        else:
            ds_result = ds_df.interpolate(method='ffill')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_limit(self):
        """Interpolation with limit parameter."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, np.nan, np.nan, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        pd_result = pd_df.interpolate(method='linear', limit=1)

        ds_df = DataStore({
            'A': [1.0, np.nan, np.nan, np.nan, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        ds_result = ds_df.interpolate(method='linear', limit=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 7: select_dtypes Operations
# =============================================================================


class TestSelectDtypesOperations:
    """Tests for select_dtypes with various combinations."""

    def test_select_dtypes_include_number(self):
        """Select numeric columns only."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        pd_result = pd_df.select_dtypes(include='number')

        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        ds_result = ds_df.select_dtypes(include='number')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_object(self):
        """Exclude object (string) columns."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        pd_result = pd_df.select_dtypes(exclude='object')

        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        ds_result = ds_df.select_dtypes(exclude='object')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_list(self):
        """Select with list of types."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        pd_result = pd_df.select_dtypes(include=['int64', 'float64'])

        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True]
        })
        ds_result = ds_df.select_dtypes(include=['int64', 'float64'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_then_operation(self):
        """Select dtypes then perform operation."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.select_dtypes(include='number').sum()

        ds_df = DataStore({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        ds_result = ds_df.select_dtypes(include='number').sum()

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Section 8: xs (Cross-Section) Operations
# =============================================================================


class TestXsOperations:
    """Tests for xs (cross-section) operations."""

    def test_xs_basic(self):
        """Basic xs operation on index."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        }, index=['x', 'y', 'z'])
        pd_result = pd_df.xs('y')

        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        }, index=['x', 'y', 'z'])
        ds_result = ds_df.xs('y')

        assert_series_equal(ds_result, pd_result)

    def test_xs_multiindex(self):
        """xs operation on MultiIndex."""
        arrays = [
            ['bar', 'bar', 'baz', 'baz'],
            ['one', 'two', 'one', 'two']
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8]
        }, index=index)
        pd_result = pd_df.xs('bar')

        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8]
        }, index=index)
        ds_result = ds_df.xs('bar')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_xs_level(self):
        """xs with specific level."""
        arrays = [
            ['bar', 'bar', 'baz', 'baz'],
            ['one', 'two', 'one', 'two']
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8]
        }, index=index)
        pd_result = pd_df.xs('one', level='second')

        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8]
        }, index=index)
        ds_result = ds_df.xs('one', level='second')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 9: Complex Numeric Reduction Chains
# =============================================================================


class TestComplexNumericReductionChains:
    """Tests for complex numeric reduction operation chains."""

    def test_filter_then_sum_mean_std(self):
        """Filter then multiple aggregations."""
        pd_df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'flag': [True, True, True, False, False, False]
        })
        pd_filtered = pd_df[pd_df['flag']]
        pd_sum = pd_filtered['value'].sum()
        pd_mean = pd_filtered['value'].mean()
        pd_std = pd_filtered['value'].std()

        ds_df = DataStore({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'flag': [True, True, True, False, False, False]
        })
        ds_filtered = ds_df[ds_df['flag']]
        ds_sum = float(ds_filtered['value'].sum())
        ds_mean = float(ds_filtered['value'].mean())
        ds_std = float(ds_filtered['value'].std())

        assert abs(ds_sum - pd_sum) < 1e-10
        assert abs(ds_mean - pd_mean) < 1e-10
        assert abs(ds_std - pd_std) < 1e-10

    def test_sem_basic(self):
        """Standard error of mean calculation."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        pd_result = pd_df.sem()

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        ds_result = ds_df.sem()

        assert_series_equal(ds_result, pd_result)

    def test_sem_after_dropna(self):
        """SEM after dropping NA values."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, np.nan, 40.0, 50.0]
        })
        pd_result = pd_df.dropna().sem()

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [10.0, 20.0, np.nan, 40.0, 50.0]
        })
        ds_result = ds_df.dropna().sem()

        assert_series_equal(ds_result, pd_result)

    def test_var_after_filter(self):
        """Variance after filter."""
        pd_df = pd.DataFrame({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            'outlier': [False, False, False, False, False, True]
        })
        pd_result = pd_df[~pd_df['outlier']]['value'].var()

        ds_df = DataStore({
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            'outlier': [False, False, False, False, False, True]
        })
        ds_result = float(ds_df[~ds_df['outlier']]['value'].var())

        assert abs(ds_result - pd_result) < 1e-10

    def test_describe_basic(self):
        """Basic describe operation."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df.describe()

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df.describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_after_filter(self):
        """Describe after filtering."""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100],
            'flag': [True, True, True, True, True, False]
        })
        pd_result = pd_df[pd_df['flag']][['value']].describe()

        ds_df = DataStore({
            'value': [1, 2, 3, 4, 5, 100],
            'flag': [True, True, True, True, True, False]
        })
        ds_result = ds_df[ds_df['flag']][['value']].describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_include_all(self):
        """Describe with include='all' for mixed types."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'str_col': ['a', 'b', 'a', 'b', 'c']
        })
        pd_result = pd_df.describe(include='all')

        ds_df = DataStore({
            'int_col': [1, 2, 3, 4, 5],
            'str_col': ['a', 'b', 'a', 'b', 'c']
        })
        ds_result = ds_df.describe(include='all')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 10: Edge Cases - Empty/Single Row DataFrames
# =============================================================================


class TestEdgeCasesEmptySingleRow:
    """Tests for edge cases with empty and single-row DataFrames."""

    def test_corr_single_row(self):
        """Correlation with single row returns NaN."""
        pd_df = pd.DataFrame({
            'A': [1.0],
            'B': [2.0]
        })
        pd_result = pd_df.corr()

        ds_df = DataStore({
            'A': [1.0],
            'B': [2.0]
        })
        ds_result = ds_df.corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_empty_df(self):
        """Rank on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        pd_result = pd_df.rank()

        ds_df = DataStore({'A': [], 'B': []})
        ds_result = ds_df.rank()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mode_empty_df(self):
        """Mode on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        pd_result = pd_df.mode()

        ds_df = DataStore({'A': [], 'B': []})
        ds_result = ds_df.mode()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transpose_single_row(self):
        """Transpose single row DataFrame."""
        pd_df = pd.DataFrame({
            'A': [1],
            'B': [2],
            'C': [3]
        })
        pd_result = pd_df.T

        ds_df = DataStore({
            'A': [1],
            'B': [2],
            'C': [3]
        })
        ds_result = ds_df.T

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_all_na(self):
        """Interpolate column with all NA values."""
        pd_df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [1.0, 2.0, 3.0]
        })
        pd_result = pd_df.interpolate()

        ds_df = DataStore({
            'A': [np.nan, np.nan, np.nan],
            'B': [1.0, 2.0, 3.0]
        })
        ds_result = ds_df.interpolate()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_empty_df(self):
        """Describe on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        pd_result = pd_df.describe()

        ds_df = DataStore({'A': [], 'B': []})
        ds_result = ds_df.describe()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 11: Complex Chain Operations
# =============================================================================


class TestComplexChainOperations:
    """Tests for complex multi-operation chains."""

    def test_filter_fillna_groupby_agg_sort(self):
        """Complex chain: filter -> fillna -> groupby -> agg -> sort."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
            'flag': [True, True, True, True, False, False]
        })
        pd_result = (pd_df[pd_df['flag']]
                     .fillna(0)
                     .groupby('group')['value']
                     .sum()
                     .reset_index()
                     .sort_values('value'))

        ds_df = DataStore({
            'group': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
            'flag': [True, True, True, True, False, False]
        })
        ds_result = (ds_df[ds_df['flag']]
                     .fillna(0)
                     .groupby('group')['value']
                     .sum()
                     .reset_index()
                     .sort_values('value'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_rank_filter(self):
        """Chain: replace -> rank -> filter."""
        pd_df = pd.DataFrame({
            'value': [-1, 2, 3, -1, 5],
            'label': ['a', 'b', 'c', 'd', 'e']
        })
        pd_df_processed = pd_df.replace(-1, 0).copy()
        pd_df_processed['rank'] = pd_df_processed['value'].rank()
        pd_result = pd_df_processed[pd_df_processed['rank'] <= 2]

        ds_df = DataStore({
            'value': [-1, 2, 3, -1, 5],
            'label': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df_processed = ds_df.replace(-1, 0).copy()
        ds_df_processed['rank'] = ds_df_processed['value'].rank()
        ds_result = ds_df_processed[ds_df_processed['rank'] <= 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_corr(self):
        """Chain: select_dtypes -> corr."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        pd_result = pd_df.select_dtypes(include='number').corr()

        ds_df = DataStore({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [5.0, 4.0, 3.0, 2.0, 1.0],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        ds_result = ds_df.select_dtypes(include='number').corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_interpolate_describe(self):
        """Chain: dropna on subset -> interpolate remaining -> describe."""
        pd_df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [np.nan, 2.0, np.nan, 4.0, np.nan],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        pd_result = pd_df.dropna(subset=['C']).interpolate().describe()

        ds_df = DataStore({
            'A': [1.0, np.nan, 3.0, np.nan, 5.0],
            'B': [np.nan, 2.0, np.nan, 4.0, np.nan],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        ds_result = ds_df.dropna(subset=['C']).interpolate().describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_then_groupby(self):
        """Assign multiple columns then groupby.

        Tests that column selection after groupby works correctly.
        Only selected columns should be aggregated.
        """
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })
        pd_result = (pd_df
                     .assign(doubled=lambda x: x['value'] * 2,
                             tripled=lambda x: x['value'] * 3)
                     .groupby('group')[['doubled', 'tripled']]
                     .sum()
                     .reset_index())

        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_result = (ds_df
                     .assign(doubled=lambda x: x['value'] * 2,
                             tripled=lambda x: x['value'] * 3)
                     .groupby('group')[['doubled', 'tripled']]
                     .sum()
                     .reset_index())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Section 12: Product and Cumulative Product
# =============================================================================


class TestProductOperations:
    """Tests for prod/product and cumprod operations."""

    def test_prod_basic(self):
        """Basic product operation."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 2, 2, 2, 2]
        })
        pd_result = pd_df.prod()

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 2, 2, 2, 2]
        })
        ds_result = ds_df.prod()

        assert_series_equal(ds_result, pd_result)

    def test_prod_with_na(self):
        """Product with NA values (skipna=True)."""
        pd_df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [2.0, np.nan, 2.0, 2.0, 2.0]
        })
        pd_result = pd_df.prod(skipna=True)

        ds_df = DataStore({
            'A': [1.0, 2.0, np.nan, 4.0, 5.0],
            'B': [2.0, np.nan, 2.0, 2.0, 2.0]
        })
        ds_result = ds_df.prod(skipna=True)

        assert_series_equal(ds_result, pd_result)

    def test_cumprod_basic(self):
        """Basic cumulative product."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 2, 2, 2, 2]
        })
        pd_result = pd_df.cumprod()

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [2, 2, 2, 2, 2]
        })
        ds_result = ds_df.cumprod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumprod_then_filter(self):
        """Cumulative product then filter."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        pd_df_cp = pd_df.copy()
        pd_df_cp['cumprod_A'] = pd_df['A'].cumprod()
        pd_result = pd_df_cp[pd_df_cp['cumprod_A'] <= 6]

        ds_df = DataStore({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        ds_df_cp = ds_df.copy()
        ds_df_cp['cumprod_A'] = ds_df['A'].cumprod()
        ds_result = ds_df_cp[ds_df_cp['cumprod_A'] <= 6]

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
