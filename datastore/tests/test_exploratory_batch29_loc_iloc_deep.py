"""
Exploratory Batch 29: loc/iloc Edge Cases, Deep Lazy Chains, Copy Semantics

This batch focuses on:
1. loc/iloc accessor edge cases
2. Deep operation chains with multiple lazy ops
3. Copy/view semantics verification
4. Mixed accessor operations
5. Boundary conditions for indexers
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal


# =======================
# Test Fixtures
# =======================


@pytest.fixture
def df_basic():
    return pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10.0, 20.0, 30.0, 40.0, 50.0], 'C': ['a', 'b', 'c', 'd', 'e']})


@pytest.fixture
def df_multitype():
    return pd.DataFrame(
        {
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, np.nan, 4.4, 5.5],
            'str_col': ['foo', 'bar', 'baz', 'qux', 'quux'],
            'bool_col': [True, False, True, False, True],
        }
    )


@pytest.fixture
def df_with_custom_index():
    return pd.DataFrame({'value': [100, 200, 300, 400, 500]}, index=['x', 'y', 'z', 'w', 'v'])


@pytest.fixture
def df_empty():
    return pd.DataFrame({'A': [], 'B': []}, dtype=float)


@pytest.fixture
def df_single_row():
    return pd.DataFrame({'A': [42], 'B': ['only_row']})


# =======================
# Part 1: loc Edge Cases
# =======================


class TestLocEdgeCases:
    """Test loc accessor edge cases."""

    def test_loc_single_label(self, df_basic):
        """Test loc with single row label."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[2]
        ds_result = ds_df.loc[2]

        assert_series_equal(pd_result, ds_result)

    def test_loc_label_list(self, df_basic):
        """Test loc with list of labels."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[[0, 2, 4]]
        ds_result = ds_df.loc[[0, 2, 4]]

        assert_frame_equal(pd_result, ds_result)

    def test_loc_label_slice(self, df_basic):
        """Test loc with slice of labels."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[1:3]
        ds_result = ds_df.loc[1:3]

        assert_frame_equal(pd_result, ds_result)

    def test_loc_row_col(self, df_basic):
        """Test loc with row and column selection."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[1:3, ['A', 'C']]
        ds_result = ds_df.loc[1:3, ['A', 'C']]

        assert_frame_equal(pd_result, ds_result)

    def test_loc_single_value(self, df_basic):
        """Test loc for single scalar value."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[2, 'B']
        ds_result = ds_df.loc[2, 'B']

        assert pd_result == ds_result

    def test_loc_with_boolean_series(self, df_basic):
        """Test loc with boolean Series."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        mask = pd_df['A'] > 2
        pd_result = pd_df.loc[mask]
        ds_result = ds_df.loc[mask]

        assert_frame_equal(pd_result, ds_result)

    def test_loc_custom_index(self, df_with_custom_index):
        """Test loc with string index."""
        pd_df = df_with_custom_index.copy()
        ds_df = DataStore(df_with_custom_index.copy())

        pd_result = pd_df.loc['y':'w']
        ds_result = ds_df.loc['y':'w']

        assert_frame_equal(pd_result, ds_result)

    def test_loc_nonexistent_label(self, df_basic):
        """Test loc with non-existent label raises KeyError."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        with pytest.raises(KeyError):
            _ = pd_df.loc[100]

        with pytest.raises(KeyError):
            _ = ds_df.loc[100]


# =======================
# Part 2: iloc Edge Cases
# =======================


class TestIlocEdgeCases:
    """Test iloc accessor edge cases."""

    def test_iloc_single_int(self, df_basic):
        """Test iloc with single integer."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[2]
        ds_result = ds_df.iloc[2]

        assert_series_equal(pd_result, ds_result)

    def test_iloc_int_list(self, df_basic):
        """Test iloc with list of integers."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[[0, 2, 4]]
        ds_result = ds_df.iloc[[0, 2, 4]]

        assert_frame_equal(pd_result, ds_result)

    def test_iloc_slice(self, df_basic):
        """Test iloc with slice."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[1:4]
        ds_result = ds_df.iloc[1:4]

        assert_frame_equal(pd_result, ds_result)

    def test_iloc_negative_index(self, df_basic):
        """Test iloc with negative index."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[-1]
        ds_result = ds_df.iloc[-1]

        assert_series_equal(pd_result, ds_result)

    def test_iloc_negative_slice(self, df_basic):
        """Test iloc with negative slice."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[-3:]
        ds_result = ds_df.iloc[-3:]

        assert_frame_equal(pd_result, ds_result)

    def test_iloc_step(self, df_basic):
        """Test iloc with step."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[::2]
        ds_result = ds_df.iloc[::2]

        assert_frame_equal(pd_result, ds_result)

    def test_iloc_row_col(self, df_basic):
        """Test iloc with row and column selection."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[1:4, [0, 2]]
        ds_result = ds_df.iloc[1:4, [0, 2]]

        assert_frame_equal(pd_result, ds_result)

    def test_iloc_single_value(self, df_basic):
        """Test iloc for single scalar value."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iloc[2, 1]
        ds_result = ds_df.iloc[2, 1]

        assert pd_result == ds_result

    def test_iloc_out_of_bounds(self, df_basic):
        """Test iloc with out of bounds index raises IndexError."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        with pytest.raises(IndexError):
            _ = pd_df.iloc[100]

        with pytest.raises(IndexError):
            _ = ds_df.iloc[100]


# =======================
# Part 3: at/iat Edge Cases
# =======================


class TestAtIatEdgeCases:
    """Test at and iat scalar accessors."""

    def test_at_single_value(self, df_basic):
        """Test at accessor for single value."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.at[2, 'B']
        ds_result = ds_df.at[2, 'B']

        assert pd_result == ds_result

    def test_at_custom_index(self, df_with_custom_index):
        """Test at with custom string index."""
        pd_df = df_with_custom_index.copy()
        ds_df = DataStore(df_with_custom_index.copy())

        pd_result = pd_df.at['z', 'value']
        ds_result = ds_df.at['z', 'value']

        assert pd_result == ds_result

    def test_iat_single_value(self, df_basic):
        """Test iat accessor for single value."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iat[2, 1]
        ds_result = ds_df.iat[2, 1]

        assert pd_result == ds_result

    def test_iat_negative_index(self, df_basic):
        """Test iat with negative indices."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.iat[-1, -1]
        ds_result = ds_df.iat[-1, -1]

        assert pd_result == ds_result


# =======================
# Part 4: Deep Lazy Operation Chains
# =======================


class TestDeepLazyChains:
    """Test deeply nested lazy operation chains."""

    def test_multiple_assigns_in_chain(self, df_basic):
        """Test multiple column assignments in sequence."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Pandas
        pd_df['D'] = pd_df['A'] * 2
        pd_df['E'] = pd_df['B'] + 100
        pd_df['F'] = pd_df['D'] + pd_df['E']
        pd_result = pd_df

        # DataStore
        ds_df['D'] = ds_df['A'] * 2
        ds_df['E'] = ds_df['B'] + 100
        ds_df['F'] = ds_df['D'] + ds_df['E']
        ds_result = ds_df.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_assign(self, df_basic):
        """Test filter followed by column assignment."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Pandas
        pd_filtered = pd_df[pd_df['A'] > 2].copy()
        pd_filtered['D'] = pd_filtered['A'] * 10
        pd_result = pd_filtered

        # DataStore
        ds_filtered = ds_df[ds_df['A'] > 2]
        ds_filtered['D'] = ds_filtered['A'] * 10
        ds_result = ds_filtered.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_select(self, df_basic):
        """Test assign followed by column selection."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Pandas
        pd_df['D'] = pd_df['A'] * 2
        pd_result = pd_df[['A', 'D']]

        # DataStore
        ds_df['D'] = ds_df['A'] * 2
        ds_result = ds_df[['A', 'D']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters(self, df_basic):
        """Test multiple filters in sequence."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Pandas
        pd_result = pd_df[pd_df['A'] > 1]
        pd_result = pd_result[pd_result['A'] < 5]
        pd_result = pd_result[pd_result['B'] > 15]

        # DataStore
        ds_result = ds_df[ds_df['A'] > 1]
        ds_result = ds_result[ds_result['A'] < 5]
        ds_result = ds_result[ds_result['B'] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_select(self, df_basic):
        """Test rename followed by column selection."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Pandas
        pd_result = pd_df.rename(columns={'A': 'X', 'B': 'Y'})[['X', 'Y']]

        # DataStore
        ds_result = ds_df.rename(columns={'A': 'X', 'B': 'Y'})[['X', 'Y']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_head(self, df_basic):
        """Test sort followed by head."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Pandas
        pd_result = pd_df.sort_values('A', ascending=False).head(3)

        # DataStore
        ds_result = ds_df.sort_values('A', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_then_dropna(self, df_multitype):
        """Test fillna followed by dropna on different column."""
        pd_df = df_multitype.copy()
        ds_df = DataStore(df_multitype.copy())

        # Add another column with NaN
        pd_df['extra'] = [1, np.nan, 3, 4, np.nan]
        ds_df['extra'] = [1, np.nan, 3, 4, np.nan]

        # Pandas
        pd_result = pd_df.fillna({'float_col': 0}).dropna(subset=['extra'])

        # DataStore
        ds_result = ds_df.fillna({'float_col': 0}).dropna(subset=['extra'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 5: Copy Semantics
# =======================


class TestCopySemantics:
    """Test that copy operations work correctly."""

    def test_copy_independence(self, df_basic):
        """Test that copy creates independent DataFrame."""
        ds_original = DataStore(df_basic.copy())
        ds_copy = ds_original.copy()

        # Modify copy
        ds_copy['A'] = ds_copy['A'] * 100

        # Original should be unchanged
        original_df = ds_original.to_df()
        copy_df = ds_copy.to_df()

        assert original_df['A'].iloc[0] == 1
        assert copy_df['A'].iloc[0] == 100

    def test_deep_copy(self, df_basic):
        """Test deep copy."""
        ds_original = DataStore(df_basic.copy())
        ds_copy = ds_original.copy(deep=True)

        # Modify copy
        ds_copy['A'] = ds_copy['A'] * 100

        # Original should be unchanged
        original_df = ds_original.to_df()
        copy_df = ds_copy.to_df()

        assert original_df['A'].iloc[0] == 1
        assert copy_df['A'].iloc[0] == 100

    def test_operation_does_not_modify_original(self, df_basic):
        """Test that operations return new DataStore, not modify original."""
        ds_original = DataStore(df_basic.copy())
        original_cols = list(ds_original.columns)

        # Perform operations
        ds_filtered = ds_original[ds_original['A'] > 2]
        ds_renamed = ds_original.rename(columns={'A': 'X'})

        # Original should be unchanged
        assert list(ds_original.columns) == original_cols
        assert 'A' in ds_original.columns


# =======================
# Part 6: Empty/Single Row Edge Cases
# =======================


class TestEmptyAndSingleRow:
    """Test edge cases with empty and single row DataFrames."""

    def test_empty_df_loc(self, df_empty):
        """Test loc on empty DataFrame."""
        pd_df = df_empty.copy()
        ds_df = DataStore(df_empty.copy())

        # Empty slice should return empty
        pd_result = pd_df.loc[:]
        ds_result = ds_df.loc[:]

        assert_frame_equal(ds_result, pd_result)

    def test_empty_df_iloc(self, df_empty):
        """Test iloc on empty DataFrame."""
        pd_df = df_empty.copy()
        ds_df = DataStore(df_empty.copy())

        # Empty slice should return empty
        pd_result = pd_df.iloc[:]
        ds_result = ds_df.iloc[:]

        assert_frame_equal(ds_result, pd_result)

    def test_single_row_loc(self, df_single_row):
        """Test loc on single row DataFrame."""
        pd_df = df_single_row.copy()
        ds_df = DataStore(df_single_row.copy())

        pd_result = pd_df.loc[0]
        ds_result = ds_df.loc[0]

        assert_series_equal(ds_result, pd_result)

    def test_single_row_iloc(self, df_single_row):
        """Test iloc on single row DataFrame."""
        pd_df = df_single_row.copy()
        ds_df = DataStore(df_single_row.copy())

        pd_result = pd_df.iloc[0]
        ds_result = ds_df.iloc[0]

        assert_series_equal(ds_result, pd_result)

    def test_empty_df_filter(self, df_empty):
        """Test filter on empty DataFrame."""
        pd_df = df_empty.copy()
        ds_df = DataStore(df_empty.copy())

        # Filter on empty should return empty
        pd_result = pd_df[pd_df['A'] > 0]
        ds_result = ds_df[ds_df['A'] > 0]

        assert_frame_equal(ds_result.to_df(), pd_result)

    def test_single_row_filter_match(self, df_single_row):
        """Test filter that matches single row."""
        pd_df = df_single_row.copy()
        ds_df = DataStore(df_single_row.copy())

        pd_result = pd_df[pd_df['A'] == 42]
        ds_result = ds_df[ds_df['A'] == 42]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self, df_single_row):
        """Test filter that doesn't match single row."""
        pd_df = df_single_row.copy()
        ds_df = DataStore(df_single_row.copy())

        pd_result = pd_df[pd_df['A'] == 999]
        ds_result = ds_df[ds_df['A'] == 999]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 7: Mixed Operations
# =======================


class TestMixedOperations:
    """Test combinations of different operation types."""

    def test_loc_after_filter(self, df_basic):
        """Test loc after filter operation."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Filter then access via loc
        pd_filtered = pd_df[pd_df['A'] > 2]
        pd_result = pd_filtered.loc[:, ['A', 'B']]

        ds_filtered = ds_df[ds_df['A'] > 2]
        ds_result = ds_filtered.loc[:, ['A', 'B']]

        assert_frame_equal(ds_result, pd_result)

    def test_iloc_after_sort(self, df_basic):
        """Test iloc after sort operation."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Sort then access via iloc
        pd_sorted = pd_df.sort_values('A', ascending=False)
        pd_result = pd_sorted.iloc[0:2]

        ds_sorted = ds_df.sort_values('A', ascending=False)
        ds_result = ds_sorted.iloc[0:2]

        assert_frame_equal(ds_result, pd_result)

    def test_assign_after_rename(self, df_basic):
        """Test column assignment after rename."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Rename then assign
        pd_renamed = pd_df.rename(columns={'A': 'X'})
        pd_renamed['Y'] = pd_renamed['X'] * 2
        pd_result = pd_renamed

        ds_renamed = ds_df.rename(columns={'A': 'X'})
        ds_renamed['Y'] = ds_renamed['X'] * 2
        ds_result = ds_renamed.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_then_filter(self, df_basic):
        """Test groupby aggregation then filter result."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'C'], 'value': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        # Groupby then filter
        pd_grouped = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_grouped[pd_grouped['value'] > 25]

        ds_grouped = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_grouped[ds_grouped['value'] > 25]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 8: Index Manipulation
# =======================


class TestIndexManipulation:
    """Test index-related operations."""

    def test_set_index_then_loc(self, df_basic):
        """Test set_index followed by loc access."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Set index then access
        pd_indexed = pd_df.set_index('C')
        pd_result = pd_indexed.loc['b':'d']

        ds_indexed = ds_df.set_index('C')
        ds_result = ds_indexed.loc['b':'d']

        assert_frame_equal(ds_result, pd_result)

    def test_reset_index_after_filter(self, df_basic):
        """Test reset_index after filter."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        # Filter creates gaps in index, reset fills them
        pd_result = pd_df[pd_df['A'] > 2].reset_index(drop=True)
        ds_result = ds_df[ds_df['A'] > 2].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex(self, df_basic):
        """Test reindex operation."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        new_index = [4, 3, 2, 1, 0, 5]  # 5 doesn't exist

        pd_result = pd_df.reindex(new_index)
        ds_result = ds_df.reindex(new_index)

        assert_frame_equal(ds_result.to_df(), pd_result)


# =======================
# Part 9: Type Coercion in Operations
# =======================


class TestTypeCoercion:
    """Test type coercion in various operations."""

    def test_int_float_arithmetic(self, df_multitype):
        """Test arithmetic between int and float columns."""
        pd_df = df_multitype.copy()
        ds_df = DataStore(df_multitype.copy())

        # Int + Float
        pd_df['result'] = pd_df['int_col'] + pd_df['float_col']
        pd_result = pd_df

        ds_df['result'] = ds_df['int_col'] + ds_df['float_col']
        ds_result = ds_df.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_concatenation_via_add(self):
        """Test string 'addition' behavior.

        DataStore automatically converts string '+' to concat() function for chDB.
        """
        pd_df = pd.DataFrame({'A': ['foo', 'bar'], 'B': ['baz', 'qux']})
        ds_df = DataStore(pd_df.copy())

        # String concatenation
        pd_df['C'] = pd_df['A'] + pd_df['B']
        pd_result = pd_df

        ds_df['C'] = ds_df['A'] + ds_df['B']
        ds_result = ds_df.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_to_int_arithmetic(self):
        """Test boolean treated as int in arithmetic."""
        pd_df = pd.DataFrame({'A': [True, False, True], 'B': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        # Bool + Int
        pd_df['C'] = pd_df['A'] + pd_df['B']
        pd_result = pd_df

        ds_df['C'] = ds_df['A'] + ds_df['B']
        ds_result = ds_df.to_df()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =======================
# Part 10: Stress Tests
# =======================


class TestStressConditions:
    """Test under stress conditions."""

    def test_many_columns_selection(self):
        """Test selecting from DataFrame with many columns."""
        data = {f'col_{i}': range(10) for i in range(50)}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(pd_df.copy())

        # Select subset
        cols = [f'col_{i}' for i in range(0, 50, 5)]
        pd_result = pd_df[cols]
        ds_result = ds_df[cols]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_many_rows_filter(self):
        """Test filtering DataFrame with many rows."""
        pd_df = pd.DataFrame({'A': range(10000), 'B': np.random.randn(10000)})
        ds_df = DataStore(pd_df.copy())

        # Filter
        pd_result = pd_df[pd_df['A'] > 9000]
        ds_result = ds_df[ds_df['A'] > 9000]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_of_10_operations(self):
        """Test chain of 10 operations."""
        pd_df = pd.DataFrame({'A': range(100), 'B': range(100, 200), 'C': ['x'] * 50 + ['y'] * 50})
        ds_df = DataStore(pd_df.copy())

        # Chain of operations
        pd_result = (
            pd_df[pd_df['A'] > 10][pd_df['A'] < 90]
            .rename(columns={'A': 'X'})
            .sort_values('X')
            .head(50)[['X', 'B']]
            .reset_index(drop=True)
        )

        ds_result = (
            ds_df[ds_df['A'] > 10][ds_df['A'] < 90]
            .rename(columns={'A': 'X'})
            .sort_values('X')
            .head(50)[['X', 'B']]
            .reset_index(drop=True)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
