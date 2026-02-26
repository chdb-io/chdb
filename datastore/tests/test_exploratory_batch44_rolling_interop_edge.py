"""
Exploratory Batch 44: Rolling/Expanding Chains, DataFrame-Series Interop, Edge Cases

Focus areas:
1. Rolling/Expanding with chain operations (filter, groupby, sort)
2. DataFrame and Series interoperability
3. Duplicate column names and special character column names
4. Multi-function aggregation edge cases
5. Type coercion with operation chains
6. Single column DataFrame and empty column edge cases
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_dataframe


# =============================================================================
# 1. Rolling with Chain Operations
# =============================================================================


class TestRollingWithChains:
    """Test rolling operations combined with other lazy operations."""

    def test_rolling_mean_then_filter(self):
        """Rolling mean followed by filter on rolling result."""
        pd_df = pd.DataFrame(
            {'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']}
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['rolling_mean'] = pd_df['value'].rolling(3).mean()
        pd_result = pd_result[pd_result['rolling_mean'] > 3.0]

        ds_result = ds_df.copy()
        ds_result['rolling_mean'] = ds_df['value'].rolling(3).mean()
        ds_result = ds_result[ds_result['rolling_mean'] > 3.0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_sum_then_sort(self):
        """Rolling sum followed by sort."""
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40, 50], 'name': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['rolling_sum'] = pd_df['value'].rolling(2).sum()
        pd_result = pd_result.sort_values('rolling_sum', ascending=False).reset_index(drop=True)

        ds_result = ds_df.copy()
        ds_result['rolling_sum'] = ds_df['value'].rolling(2).sum()
        ds_result = ds_result.sort_values('rolling_sum', ascending=False).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_rolling(self):
        """Filter then rolling operation."""
        pd_df = pd.DataFrame(
            {'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'group': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']}
        )
        ds_df = DataStore(pd_df.copy())

        pd_filtered = pd_df[pd_df['group'] == 'A'].copy()
        pd_filtered['rolling_mean'] = pd_filtered['value'].rolling(2).mean()

        ds_filtered = ds_df[ds_df['group'] == 'A'].copy()
        ds_filtered['rolling_mean'] = ds_filtered['value'].rolling(2).mean()

        assert_datastore_equals_pandas(ds_filtered, pd_filtered, check_row_order=False)

    def test_rolling_multi_agg(self):
        """Rolling with multiple aggregations."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df.copy())

        pd_rolling = pd_df['value'].rolling(3)
        pd_result = pd.DataFrame(
            {'mean': pd_rolling.mean(), 'std': pd_rolling.std(), 'min': pd_rolling.min(), 'max': pd_rolling.max()}
        )

        ds_rolling = ds_df['value'].rolling(3)
        ds_result = DataStore(
            {'mean': ds_rolling.mean(), 'std': ds_rolling.std(), 'min': ds_rolling.min(), 'max': ds_rolling.max()}
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 2. Expanding with Chain Operations
# =============================================================================


class TestExpandingWithChains:
    """Test expanding operations combined with other lazy operations."""

    def test_expanding_mean_basic(self):
        """Basic expanding mean."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['expanding_mean'] = pd_df['value'].expanding().mean()

        ds_result = ds_df.copy()
        ds_result['expanding_mean'] = ds_df['value'].expanding().mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_sum_then_filter(self):
        """Expanding sum followed by filter."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['expanding_sum'] = pd_df['value'].expanding().sum()
        pd_result = pd_result[pd_result['expanding_sum'] > 10]

        ds_result = ds_df.copy()
        ds_result['expanding_sum'] = ds_df['value'].expanding().sum()
        ds_result = ds_result[ds_result['expanding_sum'] > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_min_periods(self):
        """Expanding with min_periods parameter."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['exp_mean'] = pd_df['value'].expanding(min_periods=3).mean()

        ds_result = ds_df.copy()
        ds_result['exp_mean'] = ds_df['value'].expanding(min_periods=3).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 3. EWM (Exponential Weighted) with Chains
# =============================================================================


class TestEWMWithChains:
    """Test exponentially weighted operations."""

    def test_ewm_mean_basic(self):
        """Basic EWM mean."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['ewm_mean'] = pd_df['value'].ewm(span=3).mean()

        ds_result = ds_df.copy()
        ds_result['ewm_mean'] = ds_df['value'].ewm(span=3).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_alpha_parameter(self):
        """EWM with alpha parameter."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['ewm_mean'] = pd_df['value'].ewm(alpha=0.5).mean()

        ds_result = ds_df.copy()
        ds_result['ewm_mean'] = ds_df['value'].ewm(alpha=0.5).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_then_filter(self):
        """EWM followed by filter."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['ewm_mean'] = pd_df['value'].ewm(span=3).mean()
        pd_result = pd_result[pd_result['ewm_mean'] > 3.0]

        ds_result = ds_df.copy()
        ds_result['ewm_mean'] = ds_df['value'].ewm(span=3).mean()
        ds_result = ds_result[ds_result['ewm_mean'] > 3.0]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 4. DataFrame and Series Interoperability
# =============================================================================


class TestDataFrameSeriesInterop:
    """Test DataFrame and Series interoperability."""

    def test_series_to_frame(self):
        """Series to_frame conversion."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_series = pd_df['a']
        pd_result = pd_series.to_frame()

        ds_series = ds_df['a']
        ds_result = ds_series.to_frame()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_to_frame_with_name(self):
        """Series to_frame with custom name."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_series = pd_df['a']
        pd_result = pd_series.to_frame(name='custom_name')

        ds_series = ds_df['a']
        ds_result = ds_series.to_frame(name='custom_name')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_squeeze_to_series(self):
        """DataFrame squeeze to Series when single column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # Both should be Series
        pd_values = pd_result.values if hasattr(pd_result, 'values') else pd_result
        ds_values = get_dataframe(ds_result).squeeze().values if hasattr(ds_result, '_execute') else ds_result.values

        np.testing.assert_array_equal(ds_values, pd_values)

    def test_series_operations_return_correct_type(self):
        """Series operations preserve type correctly."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        # Filter a Series
        pd_series = pd_df['a']
        pd_result = pd_series[pd_series > 2]

        ds_series = ds_df['a']
        ds_result = ds_series[ds_series > 2]

        pd_values = pd_result.values
        ds_df_result = get_dataframe(ds_result)
        if isinstance(ds_df_result, pd.DataFrame):
            ds_values = ds_df_result.squeeze().values
        else:
            ds_values = ds_df_result.values

        np.testing.assert_array_equal(ds_values, pd_values)

    def test_dataframe_single_row_squeeze(self):
        """DataFrame squeeze with single row."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # Should be a Series
        pd_values = pd_result.values
        ds_df_result = get_dataframe(ds_result)
        if isinstance(ds_df_result, pd.DataFrame):
            ds_values = ds_df_result.squeeze().values
        else:
            ds_values = ds_df_result.values

        np.testing.assert_array_equal(ds_values, pd_values)


# =============================================================================
# 5. Duplicate Column Names Edge Cases
# =============================================================================


class TestDuplicateColumnNames:
    """Test handling of duplicate column names."""

    def test_duplicate_column_creation(self):
        """Create DataFrame with duplicate column names."""
        # pandas allows this
        pd_df = pd.DataFrame([[1, 2, 3]], columns=['a', 'a', 'b'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df
        ds_result = get_dataframe(ds_df)

        # Check column names match
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_duplicate_column_sum(self):
        """Sum with duplicate columns."""
        pd_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'a', 'b'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sum()
        ds_result = ds_df.sum()

        ds_df_result = get_dataframe(ds_result)
        if isinstance(ds_df_result, pd.DataFrame):
            ds_series = ds_df_result.squeeze()
        else:
            ds_series = ds_df_result

        # Compare values (index might differ due to duplicates)
        np.testing.assert_array_equal(ds_series.values, pd_result.values)


# =============================================================================
# 6. Special Character Column Names
# =============================================================================


class TestSpecialCharColumnNames:
    """Test handling of special characters in column names."""

    def test_space_in_column_name(self):
        """Column name with space."""
        pd_df = pd.DataFrame({'col name': [1, 2, 3], 'other': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['col name'] > 1]
        ds_result = ds_df[ds_df['col name'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_special_chars_in_column_name(self):
        """Column name with special characters."""
        pd_df = pd.DataFrame({'col-name_1': [1, 2, 3], 'other.col': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['col-name_1', 'other.col']]
        ds_result = ds_df[['col-name_1', 'other.col']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_unicode_column_name(self):
        """Column name with unicode characters."""
        pd_df = pd.DataFrame({'value': [1, 2, 3], 'category': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'value': 'value_renamed'})
        ds_result = ds_df.rename(columns={'value': 'value_renamed'})

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 7. Multi-function Aggregation Edge Cases
# =============================================================================


class TestMultiFunctionAggEdge:
    """Test edge cases in multi-function aggregation."""

    def test_agg_with_list_single_function(self):
        """Agg with list containing single function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.agg(['sum'])
        ds_result = ds_df.agg(['sum'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_with_dict_partial_columns(self):
        """Agg with dict specifying only some columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.agg({'a': 'sum', 'c': 'mean'})
        ds_result = ds_df.agg({'a': 'sum', 'c': 'mean'})

        ds_df_result = get_dataframe(ds_result)
        if isinstance(ds_df_result, pd.DataFrame):
            ds_series = ds_df_result.squeeze()
        else:
            ds_series = ds_df_result

        # Compare values
        np.testing.assert_array_almost_equal(ds_series.values, pd_result.values)

    def test_groupby_agg_mixed_functions(self):
        """GroupBy agg with mixed function types."""
        pd_df = pd.DataFrame(
            {'group': ['A', 'A', 'B', 'B'], 'value1': [1.0, 2.0, 3.0, 4.0], 'value2': [10.0, 20.0, 30.0, 40.0]}
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('group').agg({'value1': ['sum', 'mean'], 'value2': 'max'}).reset_index()

        ds_result = ds_df.groupby('group').agg({'value1': ['sum', 'mean'], 'value2': 'max'}).reset_index()

        # Flatten column names for comparison
        pd_result.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in pd_result.columns]
        ds_df_result = get_dataframe(ds_result)
        ds_df_result.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col for col in ds_df_result.columns
        ]

        assert_datastore_equals_pandas(ds_df_result, pd_result, check_row_order=False)


# =============================================================================
# 8. Type Coercion with Operation Chains
# =============================================================================


class TestTypeCoercionChains:
    """Test type coercion behavior in operation chains."""

    def test_int_to_float_after_mean(self):
        """Integer column becomes float after mean - returns scalar like pandas."""
        import numpy as np
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()

        # Both should be scalar floats
        assert isinstance(ds_result, (int, float, np.integer, np.floating))
        assert ds_result == pd_result

    def test_bool_to_int_after_sum(self):
        """Boolean column becomes int after sum - returns scalar like pandas."""
        import numpy as np
        pd_df = pd.DataFrame({'a': [True, False, True, True, False]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        # Both should be scalar integers
        assert isinstance(ds_result, (int, np.integer))
        assert ds_result == pd_result

    def test_astype_in_chain(self):
        """Type conversion in operation chain - returns scalar like pandas."""
        import numpy as np
        pd_df = pd.DataFrame({'a': [1.5, 2.7, 3.2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].astype(int).sum()
        ds_result = ds_df['a'].astype(int).sum()

        # Both should be scalar integers
        assert isinstance(ds_result, (int, np.integer))
        assert ds_result == pd_result


# =============================================================================
# 9. Single Column DataFrame Edge Cases
# =============================================================================


class TestSingleColumnDataFrame:
    """Test edge cases with single column DataFrames."""

    def test_single_column_filter(self):
        """Filter on single column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_column_groupby(self):
        """GroupBy on single column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('a').size().reset_index(name='count')
        ds_result = ds_df.groupby('a').size().reset_index(name='count')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_single_column_sort(self):
        """Sort single column DataFrame."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('a').reset_index(drop=True)
        ds_result = ds_df.sort_values('a').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 10. Empty Column Edge Cases
# =============================================================================


class TestEmptyColumnEdge:
    """Test edge cases with empty data."""

    def test_empty_dataframe_operations(self):
        """Operations on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df.copy())

        # Should not error
        pd_result = pd_df.sum()
        ds_result = ds_df.sum()

        ds_df_result = get_dataframe(ds_result)
        if isinstance(ds_df_result, pd.DataFrame):
            ds_series = ds_df_result.squeeze()
        else:
            ds_series = ds_df_result

        np.testing.assert_array_equal(ds_series.values, pd_result.values)

    def test_empty_dataframe_mean(self):
        """Mean on empty DataFrame returns NaN - scalar like pandas."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=float)})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()

        # Both should be NaN scalars
        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_filter_to_empty(self):
        """Filter that results in empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 11. Complex Chain with Multiple Window Operations
# =============================================================================


class TestComplexWindowChains:
    """Test complex chains involving multiple window operations."""

    def test_rolling_then_expanding(self):
        """Rolling followed by expanding."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['rolling'] = pd_df['value'].rolling(2).mean()
        pd_result['expanding'] = pd_result['rolling'].expanding().sum()

        ds_result = ds_df.copy()
        ds_result['rolling'] = ds_df['value'].rolling(2).mean()
        ds_result['expanding'] = ds_result['rolling'].expanding().sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_then_diff(self):
        """Cumsum followed by diff."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df['value'].cumsum()
        pd_result['diff'] = pd_result['cumsum'].diff()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df['value'].cumsum()
        ds_result['diff'] = ds_result['cumsum'].diff()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_then_rolling(self):
        """Shift followed by rolling."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['shifted'] = pd_df['value'].shift(1)
        pd_result['rolling_shifted'] = pd_result['shifted'].rolling(2).mean()

        ds_result = ds_df.copy()
        ds_result['shifted'] = ds_df['value'].shift(1)
        ds_result['rolling_shifted'] = ds_result['shifted'].rolling(2).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 12. Index Operations Edge Cases
# =============================================================================


class TestIndexOperationsEdge:
    """Test edge cases with index operations."""

    def test_reset_index_drop(self):
        """Reset index with drop=True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_then_reset(self):
        """Set index then reset."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.set_index('b').reset_index()
        ds_result = ds_df.set_index('b').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_preserves_index(self):
        """Filter preserves original index values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[10, 20, 30, 40, 50])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 13. Comparison and Boolean Operations Edge Cases
# =============================================================================


class TestComparisonBooleanEdge:
    """Test edge cases with comparison and boolean operations."""

    def test_eq_ne_combined(self):
        """Combined eq and ne operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 2, 1], 'b': [1, 1, 1, 2, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[(pd_df['a'] == 2) | (pd_df['b'] != 1)]
        ds_result = ds_df[(ds_df['a'] == 2) | (ds_df['b'] != 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_conditions_and(self):
        """Multiple AND conditions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1], 'c': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] > 1) & (pd_df['c'] < 5)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] > 1) & (ds_df['c'] < 5)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_of_complex_condition(self):
        """Negation of complex boolean condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[~((pd_df['a'] > 2) & (pd_df['b'] > 2))]
        ds_result = ds_df[~((ds_df['a'] > 2) & (ds_df['b'] > 2))]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 14. Numeric Operations Edge Cases
# =============================================================================


class TestNumericOperationsEdge:
    """Test edge cases with numeric operations."""

    def test_division_by_zero(self):
        """Division by zero handling."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1.0, 0.0, 2.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['c'] = pd_df['a'] / pd_df['b']

        ds_result = ds_df.copy()
        ds_result['c'] = ds_df['a'] / ds_df['b']

        # Both should have inf for division by zero
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_modulo_operation(self):
        """Modulo operation."""
        pd_df = pd.DataFrame({'a': [10, 11, 12, 13, 14], 'b': [3, 3, 3, 3, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['mod'] = pd_df['a'] % pd_df['b']

        ds_result = ds_df.copy()
        ds_result['mod'] = ds_df['a'] % ds_df['b']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_power_operation(self):
        """Power operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['squared'] = pd_df['a'] ** 2

        ds_result = ds_df.copy()
        ds_result['squared'] = ds_df['a'] ** 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_on_negative_values(self):
        """Absolute value on negative values.

        DataStore now correctly converts unsigned integer results from chDB's abs()
        back to signed integers to match pandas behavior.
        """
        pd_df = pd.DataFrame({'a': [-3, -2, -1, 0, 1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['abs_a'] = pd_df['a'].abs()

        ds_result = ds_df.copy()
        ds_result['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 15. String Operations with Chains
# =============================================================================


class TestStringOperationsChains:
    """Test string operations in chains."""

    def test_str_upper_then_filter(self):
        """String upper followed by filter."""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie'], 'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['name_upper'] = pd_df['name'].str.upper()
        pd_result = pd_result[pd_result['value'] > 1]

        ds_result = ds_df.copy()
        ds_result['name_upper'] = ds_df['name'].str.upper()
        ds_result = ds_result[ds_result['value'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_in_filter(self):
        """String length used in filter condition."""
        pd_df = pd.DataFrame({'name': ['a', 'ab', 'abc', 'abcd', 'abcde']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['name'].str.len() > 2]
        ds_result = ds_df[ds_df['name'].str.len() > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_chain(self):
        """String contains with chain operations."""
        pd_df = pd.DataFrame({'name': ['apple', 'banana', 'cherry', 'apricot'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['name'].str.contains('a')].sort_values('value')
        ds_result = ds_df[ds_df['name'].str.contains('a')].sort_values('value')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 16. Datetime Operations Edge Cases
# =============================================================================


class TestDatetimeOperationsEdge:
    """Test datetime operation edge cases."""

    def test_dt_year_filter(self):
        """Datetime year extraction with filter."""
        pd_df = pd.DataFrame({'date': pd.to_datetime(['2020-01-15', '2021-06-20', '2022-12-25']), 'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['date'].dt.year >= 2021]
        ds_result = ds_df[ds_df['date'].dt.year >= 2021]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_dayofweek_groupby(self):
        """Datetime dayofweek used in groupby."""
        pd_df = pd.DataFrame(
            {
                'date': pd.to_datetime(
                    ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-08', '2024-01-09', '2024-01-10']
                ),
                'value': [1, 2, 3, 4, 5, 6],
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['dow'] = pd_df['date'].dt.dayofweek
        pd_result = pd_result.groupby('dow')['value'].sum().reset_index()

        ds_result = ds_df.copy()
        ds_result['dow'] = ds_df['date'].dt.dayofweek
        ds_result = ds_result.groupby('dow')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# 17. Memory and Info Operations
# =============================================================================


class TestMemoryInfoOperations:
    """Test memory and info operations."""

    def test_memory_usage_basic(self):
        """Basic memory usage call."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        # Should not error
        pd_mem = pd_df.memory_usage()
        ds_mem = ds_df.memory_usage()

        # Both should return Series with same index
        assert len(ds_mem) == len(pd_mem)

    def test_info_basic(self):
        """Basic info call."""
        import io

        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df.copy())

        # Should not error
        pd_buf = io.StringIO()
        ds_buf = io.StringIO()
        pd_df.info(buf=pd_buf)
        ds_df.info(buf=ds_buf)

        # Both should produce output
        assert len(pd_buf.getvalue()) > 0
        assert len(ds_buf.getvalue()) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
