"""
Exploratory Discovery Batch 21 - Category Accessor, Inplace Operations, and View Semantics
Date: 2026-01-04

Focus areas:
1. Category accessor operations (categories, codes, add_categories, remove_categories)
2. Inplace parameter consistency across methods
3. Chain assignment patterns and copy semantics
4. Type conversion edge cases
5. DataFrame memory and view behavior
6. Complex boolean indexing patterns
7. DataFrame arithmetic operators (+, -, *, /, **, %, //)

Known limitations (chDB engine):
- Categorical dtype not supported by chDB (CATEGORY numpy type)
"""

import pytest
from tests.xfail_markers import (
    chdb_category_type,
    chdb_category_to_object,
    datastore_loc_conditional_assignment,
)
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Category Accessor Operations - chDB now supports categorical SQL operations
# but converts dtype from category to object after execution
# =============================================================================

class TestCategoryAccessor:
    """Test Category accessor methods on categorical columns."""

    @chdb_category_type  # FIXED: SQL operations now work, values are correct
    def test_categorical_column_creation(self):
        """Create DataFrame with categorical column."""
        pd_df = pd.DataFrame({
            'category': pd.Categorical(['low', 'medium', 'high', 'low', 'medium'])
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['category']]
        ds_result = ds_df[['category']]

        assert list(ds_result['category'].values) == list(pd_result['category'].values)

    @chdb_category_type  # FIXED: SQL operations now work, values are correct
    def test_categorical_ordered(self):
        """Create ordered categorical column."""
        pd_df = pd.DataFrame({
            'size': pd.Categorical(
                ['S', 'M', 'L', 'XL', 'M'],
                categories=['S', 'M', 'L', 'XL'],
                ordered=True
            )
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['size']]
        ds_result = ds_df[['size']]

        assert list(ds_result['size'].values) == list(pd_result['size'].values)

    @chdb_category_to_object  # chDB converts category -> object dtype
    def test_filter_categorical_column(self):
        """Filter based on categorical values."""
        pd_df = pd.DataFrame({
            'category': pd.Categorical(['low', 'medium', 'high', 'low', 'medium']),
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['category'] == 'low']
        ds_result = ds_df[ds_df['category'] == 'low']

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_category_to_object  # chDB converts category -> object dtype
    def test_groupby_categorical(self):
        """GroupBy on categorical column."""
        pd_df = pd.DataFrame({
            'category': pd.Categorical(['A', 'B', 'A', 'B', 'A']),
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category', observed=True)['value'].sum().reset_index()
        ds_result = ds_df.groupby('category', observed=True)['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Inplace Parameter Consistency
# =============================================================================

class TestInplaceOperations:
    """Test that inplace parameter is handled consistently."""

    def test_drop_inplace_false(self):
        """drop() with inplace=False should return new DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.drop(columns=['B'], inplace=False)
        ds_result = ds_df.drop(columns=['B'], inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Original should be unchanged
        assert 'B' in ds_df.columns

    def test_fillna_inplace_false(self):
        """fillna() with inplace=False returns new DataFrame."""
        pd_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.fillna(0, inplace=False)
        ds_result = ds_df.fillna(0, inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_inplace_false(self):
        """rename() with inplace=False returns new DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.rename(columns={'A': 'X'}, inplace=False)
        ds_result = ds_df.rename(columns={'A': 'X'}, inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_inplace_false(self):
        """sort_values() with inplace=False returns new DataFrame."""
        pd_df = pd.DataFrame({'A': [3, 1, 2], 'B': [6, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sort_values('A', inplace=False)
        ds_result = ds_df.sort_values('A', inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_inplace_false(self):
        """reset_index() with inplace=False returns new DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.reset_index(inplace=False)
        ds_result = ds_df.reset_index(inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_inplace_false(self):
        """dropna() with inplace=False returns new DataFrame."""
        pd_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.dropna(inplace=False)
        ds_result = ds_df.dropna(inplace=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Copy and View Semantics
# =============================================================================

class TestCopySemantics:
    """Test copy/view behavior matches pandas."""

    def test_copy_creates_independent_df(self):
        """copy() should create independent DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_copy = pd_df.copy()
        ds_copy = ds_df.copy()

        # Modify copy, original should be unchanged
        pd_copy['A'] = [10, 20, 30]
        ds_copy = ds_copy.assign(A=[10, 20, 30])

        # Original unchanged
        assert list(ds_df['A'].values) == [1, 2, 3]

    def test_deep_copy_vs_shallow(self):
        """Test deep=True vs deep=False copy behavior."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        ds_deep = ds_df.copy(deep=True)
        ds_shallow = ds_df.copy(deep=False)

        # Both should have same data
        assert list(ds_deep['A'].values) == [1, 2, 3]
        assert list(ds_shallow['A'].values) == [1, 2, 3]

    def test_slice_returns_new_datastore(self):
        """Slicing should return new DataStore, not modify original."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_slice = pd_df[pd_df['A'] > 2]
        ds_slice = ds_df[ds_df['A'] > 2]

        assert_datastore_equals_pandas(ds_slice, pd_slice)
        # Original should still have all rows
        assert len(ds_df) == 5


# =============================================================================
# Type Conversion Edge Cases
# =============================================================================

class TestTypeConversion:
    """Test type conversion methods."""

    def test_astype_int_to_float(self):
        """Convert int column to float."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.astype({'A': 'float64'})
        ds_result = ds_df.astype({'A': 'float64'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_float_to_int(self):
        """Convert float column to int (no NaN)."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.astype({'A': 'int64'})
        ds_result = ds_df.astype({'A': 'int64'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_string(self):
        """Convert numeric to string."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.astype({'A': 'str'})
        ds_result = ds_df.astype({'A': 'str'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_convert_dtypes(self):
        """Test convert_dtypes() method."""
        pd_df = pd.DataFrame({
            'A': ['1', '2', '3'],
            'B': [1.0, 2.0, 3.0],
            'C': [True, False, True]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.convert_dtypes()
        ds_result = ds_df.convert_dtypes()

        # Just check values match, dtypes may differ
        for col in pd_result.columns:
            assert list(ds_result[col].values) == list(pd_result[col].values)

    def test_to_numeric_series(self):
        """Test pd.to_numeric on series."""
        pd_df = pd.DataFrame({'A': ['1', '2', '3']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(A_num=pd.to_numeric(pd_df['A']))
        ds_result = ds_df.assign(A_num=pd.to_numeric(ds_df['A'].to_pandas()))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Complex Boolean Indexing
# =============================================================================

class TestComplexBooleanIndexing:
    """Test complex boolean indexing patterns."""

    def test_multiple_conditions_and(self):
        """Multiple AND conditions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[(pd_df['A'] > 1) & (pd_df['A'] < 5) & (pd_df['B'] > 1)]
        ds_result = ds_df[(ds_df['A'] > 1) & (ds_df['A'] < 5) & (ds_df['B'] > 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_conditions_or(self):
        """Multiple OR conditions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['x', 'y', 'z', 'x', 'y']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[(pd_df['A'] == 1) | (pd_df['A'] == 5) | (pd_df['B'] == 'z')]
        ds_result = ds_df[(ds_df['A'] == 1) | (ds_df['A'] == 5) | (ds_df['B'] == 'z')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nested_conditions(self):
        """Nested AND/OR conditions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[((pd_df['A'] > 2) & (pd_df['B'] < 4)) | (pd_df['A'] == 1)]
        ds_result = ds_df[((ds_df['A'] > 2) & (ds_df['B'] < 4)) | (ds_df['A'] == 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_condition(self):
        """Negated boolean condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[~(pd_df['A'] > 3)]
        ds_result = ds_df[~(ds_df['A'] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_condition(self):
        """isin() in boolean condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['A'].isin([1, 3, 5])]
        ds_result = ds_df[ds_df['A'].isin([1, 3, 5])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_condition(self):
        """between() in boolean condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['A'].between(2, 4)]
        ds_result = ds_df[ds_df['A'].between(2, 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_condition_contains(self):
        """String contains in condition."""
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['name'].str.contains('a', case=False)]
        ds_result = ds_df[ds_df['name'].str.contains('a', case=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# DataFrame Arithmetic Operations - Fixed in this batch
# =============================================================================

class TestDataFrameArithmetic:
    """Test DataFrame-level arithmetic operations."""

    def test_add_scalar(self):
        """Add scalar to DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df + 10
        ds_result = ds_df + 10

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subtract_scalar(self):
        """Subtract scalar from DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df - 1
        ds_result = ds_df - 1

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiply_scalar(self):
        """Multiply DataFrame by scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df * 2
        ds_result = ds_df * 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_divide_scalar(self):
        """Divide DataFrame by scalar."""
        pd_df = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df / 10
        ds_result = ds_df / 10

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_power_scalar(self):
        """Raise DataFrame to power."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df ** 2
        ds_result = ds_df ** 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mod_scalar(self):
        """Modulo operation on DataFrame."""
        pd_df = pd.DataFrame({'A': [10, 11, 12], 'B': [13, 14, 15]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df % 3
        ds_result = ds_df % 3

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_floordiv_scalar(self):
        """Floor division on DataFrame."""
        pd_df = pd.DataFrame({'A': [10, 11, 12], 'B': [13, 14, 15]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df // 3
        ds_result = ds_df // 3

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negate_dataframe(self):
        """Negate DataFrame (unary minus)."""
        pd_df = pd.DataFrame({'A': [1, -2, 3], 'B': [-4, 5, -6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = -pd_df
        ds_result = -ds_df

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_dataframe(self):
        """Absolute value of DataFrame."""
        pd_df = pd.DataFrame({'A': [1, -2, 3], 'B': [-4, 5, -6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = abs(pd_df)
        ds_result = abs(ds_df)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# DataFrame Comparison Operations
# =============================================================================

class TestDataFrameComparison:
    """Test DataFrame-level comparison operations."""

    def test_gt_scalar(self):
        """Greater than with scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df > 2
        ds_result = ds_df > 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_lt_scalar(self):
        """Less than with scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df < 5
        ds_result = ds_df < 5

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ge_scalar(self):
        """Greater than or equal with scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df >= 3
        ds_result = ds_df >= 3

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_le_scalar(self):
        """Less than or equal with scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df <= 4
        ds_result = ds_df <= 4

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eq_scalar(self):
        """Equal with scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 2], 'B': [3, 3, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df == 2
        ds_result = ds_df == 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ne_scalar(self):
        """Not equal with scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 2], 'B': [3, 3, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df != 2
        ds_result = ds_df != 2

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Aggregation with Axis Parameter
# =============================================================================

class TestAggregationAxis:
    """Test aggregation operations with axis parameter."""

    def test_sum_axis_0(self):
        """Sum along rows (axis=0)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sum(axis=0)
        ds_result = ds_df.sum(axis=0)

        # Compare as Series
        assert ds_result['A'] == pd_result['A']
        assert ds_result['B'] == pd_result['B']

    def test_sum_axis_1(self):
        """Sum along columns (axis=1)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.sum(axis=1)
        ds_result = ds_df.sum(axis=1)

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_mean_axis_0(self):
        """Mean along rows (axis=0)."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.mean(axis=0)
        ds_result = ds_df.mean(axis=0)

        np.testing.assert_almost_equal(ds_result['A'], pd_result['A'])
        np.testing.assert_almost_equal(ds_result['B'], pd_result['B'])

    def test_mean_axis_1(self):
        """Mean along columns (axis=1)."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.mean(axis=1)
        ds_result = ds_df.mean(axis=1)

        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_min_axis_1(self):
        """Min along columns (axis=1)."""
        pd_df = pd.DataFrame({'A': [1, 5, 3], 'B': [4, 2, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.min(axis=1)
        ds_result = ds_df.min(axis=1)

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_max_axis_1(self):
        """Max along columns (axis=1)."""
        pd_df = pd.DataFrame({'A': [1, 5, 3], 'B': [4, 2, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.max(axis=1)
        ds_result = ds_df.max(axis=1)

        np.testing.assert_array_equal(ds_result.values, pd_result.values)


# =============================================================================
# Select/Filter Chaining Patterns
# =============================================================================

class TestSelectFilterChaining:
    """Test various select/filter chaining patterns."""

    def test_filter_then_select(self):
        """Filter rows then select columns."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['A'] > 1][['A', 'B']]
        ds_result = ds_df[ds_df['A'] > 1][['A', 'B']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_then_filter(self):
        """Select columns then filter rows."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['A', 'B']][pd_df['A'] > 1]
        ds_result = ds_df[['A', 'B']][ds_df['A'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters(self):
        """Apply multiple filters in sequence."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['A'] > 1][pd_df['B'] > 2]
        ds_result = ds_df[ds_df['A'] > 1][ds_df['B'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_head(self):
        """Filter, sort, then head."""
        pd_df = pd.DataFrame({'A': [3, 1, 4, 1, 5, 9, 2], 'B': [6, 5, 4, 3, 2, 1, 0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['A'] > 2].sort_values('A').head(3)
        ds_result = ds_df[ds_df['A'] > 2].sort_values('A').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_sort_tail(self):
        """Select columns, sort, then tail."""
        pd_df = pd.DataFrame({'A': [3, 1, 4, 1, 5], 'B': [6, 5, 4, 3, 2], 'C': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['A', 'B']].sort_values('B').tail(2)
        ds_result = ds_df[['A', 'B']].sort_values('B').tail(2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# DataFrame Apply Variations
# =============================================================================

class TestDataFrameApply:
    """Test DataFrame.apply() variations."""

    def test_apply_sum_axis_0(self):
        """Apply sum function along axis 0."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.apply(sum, axis=0)
        ds_result = ds_df.apply(sum, axis=0)

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_apply_sum_axis_1(self):
        """Apply sum function along axis 1."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.apply(sum, axis=1)
        ds_result = ds_df.apply(sum, axis=1)

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_apply_lambda(self):
        """Apply lambda function."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.apply(lambda x: x * 2)
        ds_result = ds_df.apply(lambda x: x * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_numpy_function(self):
        """Apply numpy function."""
        pd_df = pd.DataFrame({'A': [1, 4, 9], 'B': [16, 25, 36]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.apply(np.sqrt)
        ds_result = ds_df.apply(np.sqrt)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Column Name Edge Cases
# =============================================================================

class TestColumnNameEdgeCases:
    """Test edge cases with column names."""

    def test_column_with_space(self):
        """Column name with space."""
        pd_df = pd.DataFrame({'col A': [1, 2, 3], 'col B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['col A'] > 1]
        ds_result = ds_df[ds_df['col A'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_with_underscore(self):
        """Column name with underscore."""
        pd_df = pd.DataFrame({'col_a': [1, 2, 3], 'col_b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['col_a'] > 1]
        ds_result = ds_df[ds_df['col_a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_numeric_column_name(self):
        """Numeric column name (as string)."""
        pd_df = pd.DataFrame({'1': [1, 2, 3], '2': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['1'] > 1]
        ds_result = ds_df[ds_df['1'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_case_column_names(self):
        """Mixed case column names."""
        pd_df = pd.DataFrame({'ColA': [1, 2, 3], 'colB': [4, 5, 6], 'COLC': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[['ColA', 'colB']]
        ds_result = ds_df[['ColA', 'colB']]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Value Setting Operations
# =============================================================================

class TestValueSetting:
    """Test value setting operations."""

    def test_setitem_single_value(self):
        """Set single value using at."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_df.at[0, 'A'] = 100
        ds_df.at[0, 'A'] = 100

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_setitem_column_scalar(self):
        """Set entire column to scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_df['C'] = 10
        ds_df['C'] = 10

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_setitem_column_list(self):
        """Set column to list of values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_df['B'] = [4, 5, 6]
        ds_df['B'] = [4, 5, 6]

        assert_datastore_equals_pandas(ds_df, pd_df)

    @datastore_loc_conditional_assignment
    def test_setitem_conditional(self):
        """Set values based on condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_df.loc[pd_df['A'] > 3, 'A'] = 0
        ds_df.loc[ds_df['A'] > 3, 'A'] = 0

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Null Handling Variations
# =============================================================================

class TestNullHandling:
    """Test various null/NaN handling scenarios."""

    def test_isna_isnull_equivalence(self):
        """isna() and isnull() should be equivalent."""
        pd_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_isna = pd_df.isna()
        ds_isna = ds_df.isna()

        pd_isnull = pd_df.isnull()
        ds_isnull = ds_df.isnull()

        assert_datastore_equals_pandas(ds_isna, pd_isna)
        assert_datastore_equals_pandas(ds_isnull, pd_isnull)

    def test_notna_notnull_equivalence(self):
        """notna() and notnull() should be equivalent."""
        pd_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 2, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_notna = pd_df.notna()
        ds_notna = ds_df.notna()

        pd_notnull = pd_df.notnull()
        ds_notnull = ds_df.notnull()

        assert_datastore_equals_pandas(ds_notna, pd_notna)
        assert_datastore_equals_pandas(ds_notnull, pd_notnull)

    def test_dropna_how_all(self):
        """dropna with how='all'."""
        pd_df = pd.DataFrame({
            'A': [1, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan],
            'C': [3, np.nan, np.nan]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.dropna(how='all')
        ds_result = ds_df.dropna(how='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_thresh(self):
        """dropna with thresh parameter."""
        pd_df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [np.nan, np.nan, 6],
            'C': [7, np.nan, 9]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.dropna(thresh=2)
        ds_result = ds_df.dropna(thresh=2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_ffill(self):
        """fillna with method='ffill'."""
        pd_df = pd.DataFrame({'A': [1, np.nan, np.nan, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.ffill()
        ds_result = ds_df.ffill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_bfill(self):
        """fillna with method='bfill'."""
        pd_df = pd.DataFrame({'A': [np.nan, np.nan, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.bfill()
        ds_result = ds_df.bfill()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# DataFrame Construction Edge Cases
# =============================================================================

class TestDataFrameConstruction:
    """Test DataFrame construction edge cases."""

    def test_from_dict_orient_index(self):
        """Create from dict with orient='index'."""
        data = {'row1': [1, 2, 3], 'row2': [4, 5, 6]}
        pd_df = pd.DataFrame.from_dict(data, orient='index', columns=['A', 'B', 'C'])
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_from_records(self):
        """Create from records."""
        records = [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}]
        pd_df = pd.DataFrame.from_records(records)
        ds_df = DataStore(pd_df.copy())

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_with_specified_index(self):
        """Create with specified index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())

        # Values should match
        assert list(ds_df['A'].values) == list(pd_df['A'].values)
        # Index should match
        assert list(ds_df.index) == list(pd_df.index)

    def test_with_datetime_index(self):
        """Create with datetime index."""
        dates = pd.date_range('2024-01-01', periods=3)
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=dates)
        ds_df = DataStore(pd_df.copy())

        assert list(ds_df['A'].values) == list(pd_df['A'].values)


# =============================================================================
# Head/Tail Edge Cases
# =============================================================================

class TestHeadTailEdgeCases:
    """Test head() and tail() edge cases."""

    def test_head_larger_than_df(self):
        """head(n) where n > len(df)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.head(10)
        ds_result = ds_df.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_larger_than_df(self):
        """tail(n) where n > len(df)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.tail(10)
        ds_result = ds_df.tail(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """head(0) returns empty DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert len(ds_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_tail_zero(self):
        """tail(0) returns empty DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.tail(0)
        ds_result = ds_df.tail(0)

        assert len(ds_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_head_negative(self):
        """head(-n) excludes last n rows."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.head(-2)
        ds_result = ds_df.head(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_negative(self):
        """tail(-n) excludes first n rows."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.tail(-2)
        ds_result = ds_df.tail(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
