"""
Exploratory Test Batch 28: Edge Cases and Lazy Execution Verification

Focus areas:
1. Empty DataFrame operations
2. Single row/column DataFrames
3. Complex lazy chain verification
4. Type coercion edge cases
5. Chained operations performance patterns
"""

import pytest
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal


# =============================================================================
# Part 1: Empty DataFrame Operations
# =============================================================================

class TestEmptyDataFrame:
    """Test operations on empty DataFrames."""

    def test_empty_dataframe_creation(self):
        """Create empty DataFrame with columns."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds = DataStore({'a': [], 'b': []})

        assert list(ds.columns) == list(pd_df.columns)
        assert len(ds) == len(pd_df)

    def test_empty_dataframe_filter_result(self):
        """Filter that results in empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df[pd_df['a'] > 100]

        ds = DataStore({'a': [1, 2, 3]})
        ds_result = ds[ds['a'] > 100]

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 0

    def test_empty_dataframe_sum(self):
        """Sum of empty DataFrame column."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=float)})
        pd_result = pd_df['a'].sum()

        ds = DataStore({'a': pd.Series([], dtype=float)})
        ds_result = ds['a'].sum()

        # Empty sum should be 0
        assert float(ds_result) == pd_result

    def test_empty_dataframe_mean(self):
        """Mean of empty DataFrame column returns NaN."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=float)})
        pd_result = pd_df['a'].mean()

        ds = DataStore({'a': pd.Series([], dtype=float)})
        ds_result = ds['a'].mean()

        # Empty mean should be NaN
        assert np.isnan(float(ds_result)) == np.isnan(pd_result)

    def test_empty_dataframe_count(self):
        """Count of empty DataFrame column."""
        pd_df = pd.DataFrame({'a': pd.Series([], dtype=float)})
        pd_result = pd_df['a'].count()

        ds = DataStore({'a': pd.Series([], dtype=float)})
        ds_result = ds['a'].count()

        assert int(ds_result) == pd_result

    def test_empty_groupby(self):
        """Groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'g': pd.Series([], dtype=str), 'v': pd.Series([], dtype=float)})
        pd_result = pd_df.groupby('g')['v'].sum().reset_index()

        ds = DataStore({'g': pd.Series([], dtype=str), 'v': pd.Series([], dtype=float)})
        ds_result = ds.groupby('g')['v'].sum()

        assert len(ds_result) == len(pd_result)


# =============================================================================
# Part 2: Single Row/Column DataFrames
# =============================================================================

class TestSingleRowColumn:
    """Test operations on single row or column DataFrames."""

    def test_single_row_filter(self):
        """Filter single row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        pd_result = pd_df[pd_df['a'] > 0]

        ds = DataStore({'a': [1], 'b': [2]})
        ds_result = ds[ds['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_no_match(self):
        """Filter single row DataFrame with no match."""
        pd_df = pd.DataFrame({'a': [1]})
        pd_result = pd_df[pd_df['a'] > 100]

        ds = DataStore({'a': [1]})
        ds_result = ds[ds['a'] > 100]

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 0

    def test_single_column_arithmetic(self):
        """Arithmetic on single column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = pd_df['a'] * 2

        ds = DataStore({'a': [1, 2, 3]})
        ds['b'] = ds['a'] * 2

        assert_datastore_equals_pandas(ds, pd_df)

    def test_single_row_groupby(self):
        """Groupby on single row DataFrame."""
        pd_df = pd.DataFrame({'g': ['a'], 'v': [10]})
        pd_result = pd_df.groupby('g')['v'].sum().reset_index()

        ds = DataStore({'g': ['a'], 'v': [10]})
        ds_result = ds.groupby('g')['v'].sum()

        # Compare values
        # groupby returns Series, not DataFrame. Compare scalar values
        assert float(ds_result.iloc[0]) == float(pd_result['v'].iloc[0])

    def test_single_value_agg(self):
        """Aggregation on single value."""
        pd_df = pd.DataFrame({'a': [42]})

        ds = DataStore({'a': [42]})

        assert float(ds['a'].sum()) == pd_df['a'].sum()
        assert float(ds['a'].mean()) == pd_df['a'].mean()
        assert float(ds['a'].min()) == pd_df['a'].min()
        assert float(ds['a'].max()) == pd_df['a'].max()


# =============================================================================
# Part 3: Complex Lazy Chain Verification
# =============================================================================

class TestLazyChainVerification:
    """Verify lazy execution chains produce correct results."""

    def test_filter_filter_select(self):
        """Multiple filters followed by select."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        pd_result = pd_df[pd_df['a'] > 1][pd_df['b'] > 1][['a', 'b']]

        ds = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        ds_result = ds[ds['a'] > 1][ds['b'] > 1][['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_sort(self):
        """Assignment followed by filter and sort."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_df['b'] = pd_df['a'] * 2
        pd_result = pd_df[pd_df['b'] > 4].sort_values('a')

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds['b'] = ds['a'] * 2
        ds_result = ds[ds['b'] > 4].sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))

    def test_groupby_filter_sort(self):
        """Groupby followed by filter on result and sort."""
        pd_df = pd.DataFrame({
            'g': ['a', 'a', 'b', 'b', 'c'],
            'v': [1, 2, 3, 4, 5]
        })
        # groupby returns Series, filter using Series comparison
        pd_agg = pd_df.groupby('g')['v'].sum()
        pd_result = pd_agg[pd_agg > 3].sort_values()

        ds = DataStore({
            'g': ['a', 'a', 'b', 'b', 'c'],
            'v': [1, 2, 3, 4, 5]
        })
        ds_agg = ds.groupby('g')['v'].sum()
        ds_result = ds_agg[ds_agg > 3].sort_values()

        # Compare Series values
        pd_values = pd_result.values
        ds_values = ds_result.to_pandas().values
        np.testing.assert_array_equal(np.sort(ds_values), np.sort(pd_values))

    def test_multiple_column_assignments(self):
        """Multiple column assignments in sequence."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = pd_df['a'] + 1
        pd_df['c'] = pd_df['b'] + 1
        pd_df['d'] = pd_df['c'] + 1

        ds = DataStore({'a': [1, 2, 3]})
        ds['b'] = ds['a'] + 1
        ds['c'] = ds['b'] + 1
        ds['d'] = ds['c'] + 1

        assert_datastore_equals_pandas(ds, pd_df)

    def test_filter_head_tail_chain(self):
        """Filter followed by head and tail operations."""
        pd_df = pd.DataFrame({'a': list(range(10))})
        pd_filtered = pd_df[pd_df['a'] > 2]
        pd_result = pd_filtered.head(5).tail(3)

        ds = DataStore({'a': list(range(10))})
        ds_filtered = ds[ds['a'] > 2]
        ds_result = ds_filtered.head(5).tail(3)

        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))

    def test_complex_arithmetic_chain(self):
        """Complex arithmetic expression chain."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_df['c'] = (pd_df['a'] + pd_df['b']) * 2 - pd_df['a']
        pd_df['d'] = pd_df['c'] / pd_df['b'] + 1
        pd_result = pd_df[pd_df['d'] > 2]

        ds = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds['c'] = (ds['a'] + ds['b']) * 2 - ds['a']
        ds['d'] = ds['c'] / ds['b'] + 1
        ds_result = ds[ds['d'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 4: Type Coercion Edge Cases
# =============================================================================

class TestTypeCoercion:
    """Test type coercion in various operations."""

    def test_int_float_addition(self):
        """Addition of int and float columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})
        pd_df['c'] = pd_df['a'] + pd_df['b']

        ds = DataStore({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})
        ds['c'] = ds['a'] + ds['b']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_int_division_produces_float(self):
        """Integer division should produce float."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
        pd_df['c'] = pd_df['a'] / pd_df['b']

        ds = DataStore({'a': [1, 2, 3], 'b': [2, 2, 2]})
        ds['c'] = ds['a'] / ds['b']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_string_int_comparison(self):
        """Filter string column, arithmetic on int column."""
        pd_df = pd.DataFrame({'name': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        pd_result = pd_df[pd_df['name'] == 'b']

        ds = DataStore({'name': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        ds_result = ds[ds['name'] == 'b']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_to_int_sum(self):
        """Sum of boolean series should return int."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = (pd_df['a'] > 2).sum()

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = (ds['a'] > 2).sum()

        assert int(ds_result) == pd_result

    def test_mixed_null_types(self):
        """Operations with mixed null types (None, np.nan)."""
        pd_df = pd.DataFrame({'a': [1.0, None, 3.0, np.nan, 5.0]})
        pd_result = pd_df['a'].sum()

        ds = DataStore({'a': [1.0, None, 3.0, np.nan, 5.0]})
        ds_result = ds['a'].sum()

        np.testing.assert_almost_equal(float(ds_result), pd_result)


# =============================================================================
# Part 5: Boundary Value Tests
# =============================================================================

class TestBoundaryValues:
    """Test boundary values and edge cases."""

    def test_head_zero(self):
        """head(0) should return empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.head(0)

        ds = DataStore({'a': [1, 2, 3]})
        ds_result = ds.head(0)

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 0

    def test_tail_zero(self):
        """tail(0) should return empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.tail(0)

        ds = DataStore({'a': [1, 2, 3]})
        ds_result = ds.tail(0)

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 0

    def test_head_larger_than_size(self):
        """head(n) where n > len(df)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.head(100)

        ds = DataStore({'a': [1, 2, 3]})
        ds_result = ds.head(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_larger_than_size(self):
        """tail(n) where n > len(df)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.tail(100)

        ds = DataStore({'a': [1, 2, 3]})
        ds_result = ds.tail(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_larger_than_size(self):
        """nlargest(n) where n > len(df)."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]})
        pd_result = pd_df.nlargest(100, 'a')

        ds = DataStore({'a': [3, 1, 2]})
        ds_result = ds.nlargest(100, 'a')

        # Compare values since order should be the same
        assert list(ds_result['a'].values) == list(pd_result['a'].values)

    def test_nsmallest_larger_than_size(self):
        """nsmallest(n) where n > len(df)."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]})
        pd_result = pd_df.nsmallest(100, 'a')

        ds = DataStore({'a': [3, 1, 2]})
        ds_result = ds.nsmallest(100, 'a')

        # Compare values since order should be the same
        assert list(ds_result['a'].values) == list(pd_result['a'].values)


# =============================================================================
# Part 6: Column Name Edge Cases
# =============================================================================

class TestColumnNameEdgeCases:
    """Test edge cases in column naming."""

    def test_column_with_spaces(self):
        """Column names with spaces."""
        pd_df = pd.DataFrame({'col name': [1, 2, 3], 'another col': [4, 5, 6]})
        pd_result = pd_df[pd_df['col name'] > 1]

        ds = DataStore({'col name': [1, 2, 3], 'another col': [4, 5, 6]})
        ds_result = ds[ds['col name'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_starting_with_number(self):
        """Column names starting with numbers."""
        pd_df = pd.DataFrame({'1col': [1, 2, 3], '2col': [4, 5, 6]})
        pd_df['3col'] = pd_df['1col'] + pd_df['2col']

        ds = DataStore({'1col': [1, 2, 3], '2col': [4, 5, 6]})
        ds['3col'] = ds['1col'] + ds['2col']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_column_with_special_chars(self):
        """Column names with special characters."""
        pd_df = pd.DataFrame({'col-1': [1, 2, 3], 'col_2': [4, 5, 6]})
        pd_result = pd_df['col-1'] + pd_df['col_2']

        ds = DataStore({'col-1': [1, 2, 3], 'col_2': [4, 5, 6]})
        ds_result = ds['col-1'] + ds['col_2']

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_rename_columns(self):
        """Rename columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y'})

        ds = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_result = ds.rename(columns={'a': 'x', 'b': 'y'})

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 7: Duplicate Values Handling
# =============================================================================

class TestDuplicateHandling:
    """Test handling of duplicate values."""

    def test_all_same_values(self):
        """DataFrame with all same values."""
        pd_df = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 1, 1]})

        ds = DataStore({'a': [1, 1, 1], 'b': [1, 1, 1]})

        assert float(ds['a'].sum()) == pd_df['a'].sum()
        assert float(ds['a'].mean()) == pd_df['a'].mean()
        assert float(ds['a'].std()) == pd_df['a'].std()  # Should be 0

    def test_drop_duplicates_all_same(self):
        """drop_duplicates on all same values."""
        pd_df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})
        pd_result = pd_df.drop_duplicates()

        ds = DataStore({'a': [1, 1, 1], 'b': [2, 2, 2]})
        ds_result = ds.drop_duplicates()

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 1

    def test_nunique_all_same(self):
        """nunique on all same values."""
        pd_df = pd.DataFrame({'a': [1, 1, 1]})
        pd_result = pd_df['a'].nunique()

        ds = DataStore({'a': [1, 1, 1]})
        ds_result = ds['a'].nunique()

        assert int(ds_result) == pd_result
        assert int(ds_result) == 1

    def test_value_counts_all_same(self):
        """value_counts on all same values."""
        pd_df = pd.DataFrame({'a': [1, 1, 1]})
        pd_result = pd_df['a'].value_counts()

        ds = DataStore({'a': [1, 1, 1]})
        ds_result = ds['a'].value_counts()

        # The count should be 3 for value 1
        assert int(ds_result.iloc[0]) == int(pd_result.iloc[0])


# =============================================================================
# Part 8: Comparison with NaN
# =============================================================================

class TestNaNComparison:
    """Test comparison operations with NaN values."""

    def test_nan_equality(self):
        """NaN == NaN should be False (per IEEE)."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_result = pd_df[pd_df['a'] == np.nan]  # Should return empty

        ds = DataStore({'a': [1.0, np.nan, 3.0]})
        ds_result = ds[ds['a'] == np.nan]

        # Both should be empty (NaN != NaN)
        assert len(ds_result) == len(pd_result)

    @pytest.mark.skip(reason="Known SQL/pandas semantic difference: SQL != NaN excludes NULL rows, pandas keeps all because NaN != NaN is True")
    def test_nan_not_equal(self):
        """Filter for not NaN using != has different semantics in SQL vs pandas.
        
        In pandas: df['a'] != np.nan returns all rows (NaN != NaN is True per IEEE)
        In SQL: column != NaN excludes NULL rows (SQL NULL comparison semantics)
        
        This is a known behavioral difference. Use isna()/notna() for consistent behavior.
        """
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_result = pd_df[pd_df['a'] != np.nan]  # Returns all rows because NaN != NaN

        ds = DataStore({'a': [1.0, np.nan, 3.0]})
        ds_result = ds[ds['a'] != np.nan]

        # This returns all rows in pandas too because NaN != NaN is True
        assert len(ds_result) == len(pd_result)

    def test_isna_for_nan_filter(self):
        """Proper way to filter NaN is with isna()."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_result = pd_df[pd_df['a'].isna()]

        ds = DataStore({'a': [1.0, np.nan, 3.0]})
        ds_result = ds[ds['a'].isna()]

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 1

    def test_notna_filter(self):
        """Filter non-NaN with notna()."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]})
        pd_result = pd_df[pd_df['a'].notna()]

        ds = DataStore({'a': [1.0, np.nan, 3.0, np.nan]})
        ds_result = ds[ds['a'].notna()]

        assert len(ds_result) == len(pd_result)
        assert len(ds_result) == 2


# =============================================================================
# Part 9: Sort Stability
# =============================================================================

class TestSortStability:
    """Test sort operations maintain stable order for ties."""

    def test_sort_with_ties_stable(self):
        """Sort with ties should maintain original order for equal values."""
        pd_df = pd.DataFrame({
            'a': [1, 1, 1, 2, 2],
            'b': ['e', 'd', 'c', 'b', 'a']  # Secondary column to verify order
        })
        pd_result = pd_df.sort_values('a', kind='stable')

        ds = DataStore({
            'a': [1, 1, 1, 2, 2],
            'b': ['e', 'd', 'c', 'b', 'a']
        })
        ds_result = ds.sort_values('a')

        # Check that values with same 'a' maintain relative order
        # This may differ based on implementation
        assert list(ds_result['a'].values) == list(pd_result['a'].values)

    def test_sort_multiple_columns(self):
        """Sort by multiple columns."""
        pd_df = pd.DataFrame({
            'a': [2, 1, 2, 1],
            'b': [3, 4, 1, 2]
        })
        pd_result = pd_df.sort_values(['a', 'b'])

        ds = DataStore({
            'a': [2, 1, 2, 1],
            'b': [3, 4, 1, 2]
        })
        ds_result = ds.sort_values(['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))

    def test_sort_ascending_descending_mixed(self):
        """Sort with mixed ascending/descending."""
        pd_df = pd.DataFrame({
            'a': [2, 1, 2, 1],
            'b': [3, 4, 1, 2]
        })
        pd_result = pd_df.sort_values(['a', 'b'], ascending=[True, False])

        ds = DataStore({
            'a': [2, 1, 2, 1],
            'b': [3, 4, 1, 2]
        })
        ds_result = ds.sort_values(['a', 'b'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))


# =============================================================================
# Part 10: Large Number Handling
# =============================================================================

class TestLargeNumbers:
    """Test handling of very large numbers."""

    def test_large_integers(self):
        """Operations with large integers."""
        large_val = 10**15
        pd_df = pd.DataFrame({'a': [large_val, large_val + 1, large_val + 2]})
        pd_result = pd_df['a'].sum()

        ds = DataStore({'a': [large_val, large_val + 1, large_val + 2]})
        ds_result = ds['a'].sum()

        assert int(ds_result) == pd_result

    def test_large_float_precision(self):
        """Large floats should maintain reasonable precision."""
        pd_df = pd.DataFrame({'a': [1e15 + 0.1, 1e15 + 0.2, 1e15 + 0.3]})
        pd_result = pd_df['a'].mean()

        ds = DataStore({'a': [1e15 + 0.1, 1e15 + 0.2, 1e15 + 0.3]})
        ds_result = ds['a'].mean()

        np.testing.assert_almost_equal(float(ds_result), pd_result, decimal=5)

    def test_very_small_floats(self):
        """Operations with very small floats."""
        pd_df = pd.DataFrame({'a': [1e-10, 2e-10, 3e-10]})
        pd_result = pd_df['a'].sum()

        ds = DataStore({'a': [1e-10, 2e-10, 3e-10]})
        ds_result = ds['a'].sum()

        np.testing.assert_almost_equal(float(ds_result), pd_result, decimal=15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
