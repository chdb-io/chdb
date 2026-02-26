"""
Exploratory Test Batch 47: Multi-column Operations, MultiIndex, Sparse Data, Sort Edge Cases

This batch focuses on:
1. Multi-column assign with complex dependencies
2. MultiIndex creation and operations with lazy chains
3. Sparse data handling (many NA values, empty strings, extreme values)
4. Sort stability and NaN positioning
5. Apply/Map with edge cases
6. Mixed-type operations and coercion chains

Following Mirror Code Pattern - all tests compare DataStore behavior with pandas.
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Multi-column Assign with Dependencies
# =============================================================================


class TestMultiColumnAssignDependencies:
    """Test assign operations where columns depend on each other."""

    def test_assign_chained_dependencies(self):
        """Test assign where new column depends on another new column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2).assign(c=lambda x: x['b'] + 1)
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2).assign(c=lambda x: x['b'] + 1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_columns_single_call(self):
        """Test assign with multiple columns in a single call."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.assign(c=lambda x: x['a'] + x['b'], d=lambda x: x['a'] * x['b'])
        ds_result = ds_df.assign(c=lambda x: x['a'] + x['b'], d=lambda x: x['a'] * x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_overwrite_existing_column(self):
        """Test assign that overwrites an existing column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.assign(a=lambda x: x['a'] * 10)
        ds_result = ds_df.assign(a=lambda x: x['a'] * 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_filter_chain(self):
        """Test assign followed by filter using assigned column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(b=lambda x: x['a'] * 2).query('b > 4')
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2).query('b > 4')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_groupby_on_assigned_column(self):
        """Test assign followed by groupby on the assigned column."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 3]})
        ds_df = DataStore({'a': [1, 1, 2, 2, 3]})

        pd_result = pd_df.assign(b=lambda x: x['a'] % 2).groupby('b').size().reset_index(name='count')
        ds_result = ds_df.assign(b=lambda x: x['a'] % 2).groupby('b').size().reset_index(name='count')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_scalar_and_series_mix(self):
        """Test assign with both scalar and Series values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.assign(scalar_col=100, series_col=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(scalar_col=100, series_col=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# MultiIndex Operations
# =============================================================================


class TestMultiIndexOperations:
    """Test MultiIndex creation and operations with lazy chains."""

    def test_set_index_multiindex_basic(self):
        """Test creating MultiIndex via set_index."""
        pd_df = pd.DataFrame({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 2, 1, 2],
            'c': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 2, 1, 2],
            'c': [10, 20, 30, 40]
        })

        pd_result = pd_df.set_index(['a', 'b'])
        ds_result = ds_df.set_index(['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiindex_reset_index(self):
        """Test reset_index on MultiIndex DataFrame."""
        pd_df = pd.DataFrame({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 2, 1, 2],
            'c': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 2, 1, 2],
            'c': [10, 20, 30, 40]
        })

        pd_result = pd_df.set_index(['a', 'b']).reset_index()
        ds_result = ds_df.set_index(['a', 'b']).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_creates_multiindex_in_result(self):
        """Test groupby that creates MultiIndex in result."""
        pd_df = pd.DataFrame({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 1, 2, 2],
            'c': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 1, 2, 2],
            'c': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby(['a', 'b'])['c'].sum().reset_index()
        ds_result = ds_df.groupby(['a', 'b'])['c'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiindex_filter_then_reset(self):
        """Test filter on MultiIndex DataFrame then reset."""
        pd_df = pd.DataFrame({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 2, 1, 2],
            'c': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'a': ['x', 'x', 'y', 'y'],
            'b': [1, 2, 1, 2],
            'c': [10, 20, 30, 40]
        })

        pd_result = pd_df.set_index(['a', 'b']).query('c > 15').reset_index()
        ds_result = ds_df.set_index(['a', 'b']).query('c > 15').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Sparse Data Handling
# =============================================================================


class TestSparseDataHandling:
    """Test handling of sparse data with many NA values."""

    def test_mostly_na_filter(self):
        """Test filter on column that is mostly NA."""
        pd_df = pd.DataFrame({'a': [None, None, None, 1, None]})
        ds_df = DataStore({'a': [None, None, None, 1, None]})

        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_na_aggregation(self):
        """Test aggregation on all-NA column."""
        pd_df = pd.DataFrame({'a': [None, None, None], 'b': [1, 2, 3]})
        ds_df = DataStore({'a': [None, None, None], 'b': [1, 2, 3]})

        pd_mean = pd_df['a'].mean()
        ds_mean = ds_df['a'].mean()

        # Both should return NaN
        assert pd.isna(pd_mean) and pd.isna(ds_mean)

    def test_sparse_groupby_agg(self):
        """Test groupby aggregation with sparse data."""
        pd_df = pd.DataFrame({
            'group': ['a', 'a', 'b', 'b'],
            'value': [1, None, None, 4]
        })
        ds_df = DataStore({
            'group': ['a', 'a', 'b', 'b'],
            'value': [1, None, None, 4]
        })

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        pd_df = pd.DataFrame({'a': ['', 'x', '', 'y', '']})
        ds_df = DataStore({'a': ['', 'x', '', 'y', '']})

        pd_result = pd_df[pd_df['a'] != '']
        ds_result = ds_df[ds_df['a'] != '']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_na_types_numeric(self):
        """Test numeric column with different NA representations."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 2.0, np.nan, 3.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 2.0, np.nan, 3.0]})

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_extremely_sparse_dataframe(self):
        """Test DataFrame where 90% of values are NA."""
        np.random.seed(42)
        data = np.random.choice([np.nan, 1.0], size=100, p=[0.9, 0.1])
        pd_df = pd.DataFrame({'a': data})
        ds_df = DataStore({'a': data.tolist()})

        pd_result = pd_df['a'].count()
        ds_result = ds_df['a'].count()

        assert pd_result == ds_result


# =============================================================================
# Sort Stability and NaN Positioning
# =============================================================================


class TestSortStabilityAndNaN:
    """Test sort behavior including stability and NaN handling."""

    def test_sort_with_nan_default(self):
        """Test sort_values with NaN (default na_position='last')."""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 1.0, np.nan, 2.0]})
        ds_df = DataStore({'a': [3.0, np.nan, 1.0, np.nan, 2.0]})

        pd_result = pd_df.sort_values('a').reset_index(drop=True)
        ds_result = ds_df.sort_values('a').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_with_nan_first(self):
        """Test sort_values with na_position='first'."""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 1.0, np.nan, 2.0]})
        ds_df = DataStore({'a': [3.0, np.nan, 1.0, np.nan, 2.0]})

        pd_result = pd_df.sort_values('a', na_position='first').reset_index(drop=True)
        ds_result = ds_df.sort_values('a', na_position='first').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending_with_nan(self):
        """Test descending sort with NaN values."""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 1.0, 2.0]})
        ds_df = DataStore({'a': [3.0, np.nan, 1.0, 2.0]})

        pd_result = pd_df.sort_values('a', ascending=False).reset_index(drop=True)
        ds_result = ds_df.sort_values('a', ascending=False).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_column_sort_with_nan(self):
        """Test multi-column sort with NaN in one column."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 2, 2],
            'b': [np.nan, 1.0, np.nan, 2.0]
        })
        ds_df = DataStore({
            'a': [1, 1, 2, 2],
            'b': [np.nan, 1.0, np.nan, 2.0]
        })

        pd_result = pd_df.sort_values(['a', 'b']).reset_index(drop=True)
        ds_result = ds_df.sort_values(['a', 'b']).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_string_with_nan(self):
        """Test sort on string column with NA values."""
        pd_df = pd.DataFrame({'a': ['c', None, 'a', None, 'b']})
        ds_df = DataStore({'a': ['c', None, 'a', None, 'b']})

        pd_result = pd_df.sort_values('a').reset_index(drop=True)
        ds_result = ds_df.sort_values('a').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_filter(self):
        """Test sort followed by filter."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.sort_values('a').query('b > 15').reset_index(drop=True)
        ds_result = ds_df.sort_values('a').query('b > 15').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Apply and Map Edge Cases
# =============================================================================


class TestApplyMapEdgeCases:
    """Test apply and map with various edge cases."""

    def test_apply_simple_lambda_column(self):
        """Test apply with simple lambda on column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].apply(lambda x: x ** 2)
        ds_result = ds_df['a'].apply(lambda x: x ** 2)

        # Compare as DataFrames
        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_apply_with_na_values(self):
        """Test apply handling NA values."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0]})

        pd_result = pd_df['a'].apply(lambda x: x * 2 if pd.notna(x) else x)
        ds_result = ds_df['a'].apply(lambda x: x * 2 if pd.notna(x) else x)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_map_dict(self):
        """Test map with dictionary."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'z', 'x']})
        ds_df = DataStore({'a': ['x', 'y', 'z', 'x']})

        mapping = {'x': 1, 'y': 2, 'z': 3}
        pd_result = pd_df['a'].map(mapping)
        ds_result = ds_df['a'].map(mapping)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_map_with_missing_key(self):
        """Test map with dictionary that doesn't cover all values."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'z', 'w']})
        ds_df = DataStore({'a': ['x', 'y', 'z', 'w']})

        mapping = {'x': 1, 'y': 2}  # 'z' and 'w' not in mapping
        pd_result = pd_df['a'].map(mapping)
        ds_result = ds_df['a'].map(mapping)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_apply_dataframe_axis0(self):
        """Test apply on DataFrame with axis=0 (columns)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.apply(sum, axis=0)
        ds_result = ds_df.apply(sum, axis=0)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result
        )

    def test_apply_dataframe_axis1(self):
        """Test apply on DataFrame with axis=1 (rows)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.apply(sum, axis=1)
        ds_result = ds_df.apply(sum, axis=1)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result
        )


# =============================================================================
# Extreme Values and Edge Cases
# =============================================================================


class TestExtremeValues:
    """Test handling of extreme values."""

    def test_very_large_integers(self):
        """Test operations with very large integers."""
        large = 10**15
        pd_df = pd.DataFrame({'a': [large, large + 1, large + 2]})
        ds_df = DataStore({'a': [large, large + 1, large + 2]})

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        assert pd_result == ds_result

    def test_very_small_floats(self):
        """Test operations with very small floats."""
        small = 1e-15
        pd_df = pd.DataFrame({'a': [small, small * 2, small * 3]})
        ds_df = DataStore({'a': [small, small * 2, small * 3]})

        pd_mean = pd_df['a'].mean()
        ds_mean = ds_df['a'].mean()

        np.testing.assert_almost_equal(ds_mean, pd_mean, decimal=20)

    def test_infinity_values(self):
        """Test handling of infinity values."""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 2.0]})
        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, 2.0]})

        pd_result = pd_df[pd_df['a'] != np.inf].reset_index(drop=True)
        ds_result = ds_df[ds_df['a'] != np.inf].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negative_zero(self):
        """Test handling of negative zero."""
        pd_df = pd.DataFrame({'a': [0.0, -0.0, 1.0]})
        ds_df = DataStore({'a': [0.0, -0.0, 1.0]})

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        assert pd_result == ds_result

    def test_mixed_sign_operations(self):
        """Test operations with mixed positive/negative values."""
        pd_df = pd.DataFrame({'a': [-5, -3, 0, 3, 5]})
        ds_df = DataStore({'a': [-5, -3, 0, 3, 5]})

        pd_result = pd_df.assign(abs_a=lambda x: x['a'].abs(), squared=lambda x: x['a'] ** 2)
        ds_result = ds_df.assign(abs_a=lambda x: x['a'].abs(), squared=lambda x: x['a'] ** 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Complex Chain Operations
# =============================================================================


class TestComplexChains:
    """Test complex chains of operations."""

    def test_filter_assign_groupby_sort_head(self):
        """Test chain: filter -> assign -> groupby -> sort -> head."""
        pd_df = pd.DataFrame({
            'category': ['a', 'b', 'a', 'b', 'a', 'b'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['a', 'b', 'a', 'b', 'a', 'b'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = (pd_df
                     .query('value > 15')
                     .assign(double_value=lambda x: x['value'] * 2)
                     .groupby('category')['double_value']
                     .sum()
                     .reset_index()
                     .sort_values('double_value', ascending=False)
                     .head(2))
        ds_result = (ds_df
                     .query('value > 15')
                     .assign(double_value=lambda x: x['value'] * 2)
                     .groupby('category')['double_value']
                     .sum()
                     .reset_index()
                     .sort_values('double_value', ascending=False)
                     .head(2))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_in_chain(self):
        """Test multiple filter operations in chain."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'x', 'y', 'x']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'x', 'y', 'x']
        })

        pd_result = pd_df.query('a > 1').query('b < 45').query('c == "x"')
        ds_result = ds_df.query('a > 1').query('b < 45').query('c == "x"')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_assign_chain(self):
        """Test assign -> filter -> assign chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = (pd_df
                     .assign(b=lambda x: x['a'] * 2)
                     .query('b > 4')
                     .assign(c=lambda x: x['b'] + x['a']))
        ds_result = (ds_df
                     .assign(b=lambda x: x['a'] * 2)
                     .query('b > 4')
                     .assign(c=lambda x: x['b'] + x['a']))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_filter_sort_chain(self):
        """Test sort -> filter -> sort chain."""
        pd_df = pd.DataFrame({
            'a': [5, 3, 4, 1, 2],
            'b': [50, 30, 40, 10, 20]
        })
        ds_df = DataStore({
            'a': [5, 3, 4, 1, 2],
            'b': [50, 30, 40, 10, 20]
        })

        pd_result = (pd_df
                     .sort_values('a')
                     .query('b > 15')
                     .sort_values('b', ascending=False)
                     .reset_index(drop=True))
        ds_result = (ds_df
                     .sort_values('a')
                     .query('b > 15')
                     .sort_values('b', ascending=False)
                     .reset_index(drop=True))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Type Coercion Edge Cases
# =============================================================================


class TestTypeCoercionEdgeCases:
    """Test type coercion in various scenarios."""

    def test_int_to_float_on_na_assignment(self):
        """Test int column becomes float when NA is assigned."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_df_copy = pd_df.copy()
        pd_df_copy.loc[0, 'a'] = np.nan

        ds_df_copy = ds_df.copy()
        ds_df_copy.loc[0, 'a'] = np.nan

        assert_datastore_equals_pandas(ds_df_copy, pd_df_copy)

    def test_astype_chain(self):
        """Test multiple astype operations in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df['a'].astype(float).astype(str)
        ds_result = ds_df['a'].astype(float).astype(str)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_mixed_numeric_operations(self):
        """Test operations between int and float columns."""
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})
        ds_df = DataStore({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})

        pd_result = pd_df.assign(sum_col=lambda x: x['int_col'] + x['float_col'])
        ds_result = ds_df.assign(sum_col=lambda x: x['int_col'] + x['float_col'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_to_int_conversion(self):
        """Test boolean to integer conversion."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(is_big=(pd_df['a'] > 2).astype(int))
        ds_result = ds_df.assign(is_big=(ds_df['a'] > 2).astype(int))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Column Operations Edge Cases
# =============================================================================


class TestColumnOperationsEdgeCases:
    """Test edge cases in column operations."""

    def test_select_single_column_as_dataframe(self):
        """Test selecting single column as DataFrame (double bracket)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df[['a']]
        ds_result = ds_df[['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reorder_columns(self):
        """Test reordering columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        pd_result = pd_df[['c', 'a', 'b']]
        ds_result = ds_df[['c', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_then_add_same_column_name(self):
        """Test dropping a column then adding one with the same name."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df.drop(columns=['b']).assign(b=lambda x: x['a'] * 10)
        ds_result = ds_df.drop(columns=['b']).assign(b=lambda x: x['a'] * 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_filter_on_new_name(self):
        """Test rename followed by filter on new column name."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.rename(columns={'a': 'new_a'}).query('new_a > 2')
        ds_result = ds_df.rename(columns={'a': 'new_a'}).query('new_a > 2')

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Window Operations with Edge Cases
# =============================================================================


class TestWindowEdgeCases:
    """Test window operations with edge cases."""

    def test_rolling_with_all_na_window(self):
        """Test rolling where some windows are all NA."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, np.nan, np.nan, 5.0]})
        ds_df = DataStore({'a': [1.0, np.nan, np.nan, np.nan, 5.0]})

        pd_result = pd_df['a'].rolling(2).mean()
        ds_result = ds_df['a'].rolling(2).mean()

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_cumsum_with_na(self):
        """Test cumsum with NA values."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = pd_df['a'].cumsum()
        ds_result = ds_df['a'].cumsum()

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_expanding_min_periods(self):
        """Test expanding with min_periods."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].expanding(min_periods=3).sum()
        ds_result = ds_df['a'].expanding(min_periods=3).sum()

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_diff_with_filter(self):
        """Test diff followed by filter."""
        pd_df = pd.DataFrame({'a': [1, 3, 6, 10, 15]})
        ds_df = DataStore({'a': [1, 3, 6, 10, 15]})

        pd_result = pd_df.assign(diff_a=lambda x: x['a'].diff()).query('diff_a > 3').reset_index(drop=True)
        ds_result = ds_df.assign(diff_a=lambda x: x['a'].diff()).query('diff_a > 3').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Boolean Indexing Edge Cases
# =============================================================================


class TestBooleanIndexingEdgeCases:
    """Test boolean indexing edge cases."""

    def test_boolean_with_all_false(self):
        """Test boolean indexing where all values are False."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_with_all_true(self):
        """Test boolean indexing where all values are True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df[pd_df['a'] < 100]
        ds_result = ds_df[ds_df['a'] < 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_compound_boolean_condition(self):
        """Test compound boolean condition with AND and OR."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['b'] < 4) | (pd_df['a'] == 1)]
        ds_result = ds_df[(ds_df['a'] > 2) & (ds_df['b'] < 4) | (ds_df['a'] == 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_with_na_in_condition_column(self):
        """Test boolean indexing when condition column has NA."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = pd_df[pd_df['a'] > 2]
        ds_result = ds_df[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Unique and Duplicate Handling
# =============================================================================


class TestUniqueAndDuplicates:
    """Test unique and duplicate handling."""

    def test_unique_with_na(self):
        """Test unique() with NA values."""
        pd_df = pd.DataFrame({'a': [1, 2, None, 2, None, 3]})
        ds_df = DataStore({'a': [1, 2, None, 2, None, 3]})

        pd_unique = pd_df['a'].unique()
        ds_unique = ds_df['a'].unique()

        # Compare as sets since order may vary
        pd_set = set(x for x in pd_unique if pd.notna(x))
        ds_set = set(x for x in ds_unique if pd.notna(x))
        assert pd_set == ds_set

        # Check NA count
        pd_na_count = sum(1 for x in pd_unique if pd.isna(x))
        ds_na_count = sum(1 for x in ds_unique if pd.isna(x))
        assert pd_na_count == ds_na_count

    def test_drop_duplicates_keep_first(self):
        """Test drop_duplicates with keep='first'."""
        pd_df = pd.DataFrame({'a': [1, 2, 1, 3, 2], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 1, 3, 2], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.drop_duplicates(subset=['a'], keep='first').reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset=['a'], keep='first').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """Test drop_duplicates with keep='last'."""
        pd_df = pd.DataFrame({'a': [1, 2, 1, 3, 2], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 1, 3, 2], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.drop_duplicates(subset=['a'], keep='last').reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset=['a'], keep='last').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self):
        """Test drop_duplicates with keep=False (remove all duplicates)."""
        pd_df = pd.DataFrame({'a': [1, 2, 1, 3, 2], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [1, 2, 1, 3, 2], 'b': [10, 20, 30, 40, 50]})

        pd_result = pd_df.drop_duplicates(subset=['a'], keep=False).reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset=['a'], keep=False).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicated_with_na(self):
        """Test duplicated() with NA values."""
        pd_df = pd.DataFrame({'a': [1, None, 2, None, 1]})
        ds_df = DataStore({'a': [1, None, 2, None, 1]})

        pd_result = pd_df.duplicated(subset=['a'])
        ds_result = ds_df.duplicated(subset=['a'])

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )


# =============================================================================
# String Operations Edge Cases
# =============================================================================


class TestStringOperationsEdgeCases:
    """Test string operations with edge cases."""

    def test_str_operations_on_empty_string(self):
        """Test string operations on empty strings."""
        pd_df = pd.DataFrame({'a': ['', 'hello', '', 'world']})
        ds_df = DataStore({'a': ['', 'hello', '', 'world']})

        pd_result = pd_df['a'].str.len()
        ds_result = ds_df['a'].str.len()

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_str_contains_with_na(self):
        """Test str.contains with NA values."""
        pd_df = pd.DataFrame({'a': ['hello', None, 'world', None]})
        ds_df = DataStore({'a': ['hello', None, 'world', None]})

        pd_result = pd_df['a'].str.contains('o', na=False)
        ds_result = ds_df['a'].str.contains('o', na=False)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_str_split_then_explode(self):
        """Test str.split followed by explode."""
        pd_df = pd.DataFrame({'a': ['a,b,c', 'd,e', 'f']})
        ds_df = DataStore({'a': ['a,b,c', 'd,e', 'f']})

        pd_result = pd_df['a'].str.split(',').explode().reset_index(drop=True)
        ds_result = ds_df['a'].str.split(',').explode().reset_index(drop=True)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )

    def test_str_replace_regex(self):
        """Test str.replace with regex."""
        pd_df = pd.DataFrame({'a': ['abc123', 'def456', 'ghi789']})
        ds_df = DataStore({'a': ['abc123', 'def456', 'ghi789']})

        pd_result = pd_df['a'].str.replace(r'\d+', 'NUM', regex=True)
        ds_result = ds_df['a'].str.replace(r'\d+', 'NUM', regex=True)

        pd.testing.assert_series_equal(
            pd.Series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result,
            pd_result,
            check_names=False
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
