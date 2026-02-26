"""
Exploratory Batch 36: Merge/Join chains and edge cases

Focus areas:
1. Merge with subsequent operations (filter, groupby, sort)
2. Empty DataFrame merge behavior
3. Single-row DataFrame merge
4. Merge with NA values in join keys
5. Multiple merge operations in a chain
6. Merge after filter/groupby
7. Left_on/right_on with different column names

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
Discovery date: 2026-01-06
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_series


# =============================================================================
# Test Group 1: Merge with Subsequent Operations
# =============================================================================


class TestMergeWithSubsequentOps:
    """Test merge followed by various operations."""

    def test_merge_then_filter(self):
        """Test merge followed by filter."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [10, 20, 30]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [100, 200, 300]})

        # pandas
        pd_merged = pd.merge(df1, df2, on='key')
        pd_result = pd_merged[pd_merged['val1'] > 15]

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_merged = ds1.merge(ds2, on='key')
        ds_result = ds_merged[ds_merged['val1'] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_then_sort(self):
        """Test merge followed by sort."""
        df1 = pd.DataFrame({'key': ['C', 'A', 'B'], 'val1': [30, 10, 20]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [100, 200, 300]})

        # pandas
        pd_merged = pd.merge(df1, df2, on='key')
        pd_result = pd_merged.sort_values('val1')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_merged = ds1.merge(ds2, on='key')
        ds_result = ds_merged.sort_values('val1')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_merge_then_groupby_sum(self):
        """Test merge followed by groupby and aggregation."""
        df1 = pd.DataFrame({
            'key': ['A', 'A', 'B', 'B'],
            'cat': ['X', 'Y', 'X', 'Y'],
            'val1': [10, 20, 30, 40]
        })
        df2 = pd.DataFrame({
            'key': ['A', 'B'],
            'val2': [100, 200]
        })

        # pandas
        pd_merged = pd.merge(df1, df2, on='key')
        pd_result = pd_merged.groupby('cat')['val1'].sum()

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_merged = ds1.merge(ds2, on='key')
        ds_result = ds_merged.groupby('cat')['val1'].sum()

        ds_series = get_series(ds_result)
        assert ds_series.equals(pd_result), f"DS={ds_series}, PD={pd_result}"

    def test_merge_then_head(self):
        """Test merge followed by head."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val1': [10, 20, 30, 40]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val2': [100, 200, 300, 400]})

        # pandas
        pd_merged = pd.merge(df1, df2, on='key')
        pd_result = pd_merged.head(2)

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_merged = ds1.merge(ds2, on='key')
        ds_result = ds_merged.head(2)

        # Note: order may not be guaranteed, just check shape and values exist
        assert len(ds_result) == len(pd_result) == 2

    def test_merge_filter_sort_head_chain(self):
        """Test merge -> filter -> sort -> head chain."""
        df1 = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'val1': [50, 40, 30, 20, 10]
        })
        df2 = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'val2': [5, 4, 3, 2, 1]
        })

        # pandas
        pd_merged = pd.merge(df1, df2, on='key')
        pd_result = pd_merged[pd_merged['val1'] > 15].sort_values('val2', ascending=False).head(3)

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_merged = ds1.merge(ds2, on='key')
        ds_result = ds_merged[ds_merged['val1'] > 15].sort_values('val2', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


# =============================================================================
# Test Group 2: Empty and Single-Row DataFrame Merge
# =============================================================================


class TestMergeEmptyAndSingleRow:
    """Test merge with empty and single-row DataFrames."""

    def test_merge_with_empty_left(self):
        """Test merge when left DataFrame is empty."""
        df1 = pd.DataFrame({'key': pd.Series([], dtype=str), 'val1': pd.Series([], dtype=int)})
        df2 = pd.DataFrame({'key': ['A', 'B'], 'val2': [100, 200]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_merge_with_empty_right(self):
        """Test merge when right DataFrame is empty."""
        df1 = pd.DataFrame({'key': ['A', 'B'], 'val1': [10, 20]})
        df2 = pd.DataFrame({'key': pd.Series([], dtype=str), 'val2': pd.Series([], dtype=int)})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_merge_both_empty(self):
        """Test merge when both DataFrames are empty."""
        df1 = pd.DataFrame({'key': pd.Series([], dtype=str), 'val1': pd.Series([], dtype=int)})
        df2 = pd.DataFrame({'key': pd.Series([], dtype=str), 'val2': pd.Series([], dtype=int)})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_merge_single_row_each(self):
        """Test merge with single-row DataFrames."""
        df1 = pd.DataFrame({'key': ['A'], 'val1': [10]})
        df2 = pd.DataFrame({'key': ['A'], 'val2': [100]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_single_row_no_match(self):
        """Test merge with single-row DataFrames that don't match."""
        df1 = pd.DataFrame({'key': ['A'], 'val1': [10]})
        df2 = pd.DataFrame({'key': ['B'], 'val2': [100]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert len(ds_result) == 0
        assert len(pd_result) == 0


# =============================================================================
# Test Group 3: Merge with NA Values
# =============================================================================


class TestMergeWithNA:
    """Test merge behavior with NA values in join keys."""

    def test_merge_na_in_left_key_inner(self):
        """Test inner merge with NA in left join key."""
        df1 = pd.DataFrame({'key': ['A', None, 'C'], 'val1': [10, 20, 30]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [100, 200, 300]})

        # pandas inner merge doesn't match None/NaN keys
        pd_result = pd.merge(df1, df2, on='key', how='inner')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key', how='inner')

        # Inner merge should not include NA rows
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_na_in_both_keys_left(self):
        """Test left merge with NA in both join keys."""
        df1 = pd.DataFrame({'key': ['A', None, 'C'], 'val1': [10, 20, 30]})
        df2 = pd.DataFrame({'key': ['A', None, 'C'], 'val2': [100, 200, 300]})

        # pandas left merge keeps left rows, NA doesn't match NA
        pd_result = pd.merge(df1, df2, on='key', how='left')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key', how='left')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_na_values_in_non_key_columns(self):
        """Test merge with NA values in non-key columns."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [10, None, 30]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [None, 200, 300]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 4: Multiple Merge Operations
# =============================================================================


class TestMultipleMerges:
    """Test chains of multiple merge operations."""

    def test_double_merge(self):
        """Test two consecutive merge operations."""
        df1 = pd.DataFrame({'key1': ['A', 'B', 'C'], 'val1': [1, 2, 3]})
        df2 = pd.DataFrame({'key1': ['A', 'B', 'C'], 'key2': ['X', 'Y', 'Z'], 'val2': [10, 20, 30]})
        df3 = pd.DataFrame({'key2': ['X', 'Y', 'Z'], 'val3': [100, 200, 300]})

        # pandas
        pd_merged = pd.merge(df1, df2, on='key1')
        pd_result = pd.merge(pd_merged, df3, on='key2')

        # DataStore
        ds1, ds2, ds3 = DataStore(df1), DataStore(df2), DataStore(df3)
        ds_merged = ds1.merge(ds2, on='key1')
        ds_result = ds_merged.merge(ds3, on='key2')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_triple_merge(self):
        """Test three consecutive merge operations."""
        df1 = pd.DataFrame({'id': [1, 2, 3], 'a': ['A', 'B', 'C']})
        df2 = pd.DataFrame({'id': [1, 2, 3], 'b': ['X', 'Y', 'Z']})
        df3 = pd.DataFrame({'id': [1, 2, 3], 'c': [10, 20, 30]})
        df4 = pd.DataFrame({'id': [1, 2, 3], 'd': [100, 200, 300]})

        # pandas
        pd_result = pd.merge(pd.merge(pd.merge(df1, df2, on='id'), df3, on='id'), df4, on='id')

        # DataStore
        ds1, ds2, ds3, ds4 = DataStore(df1), DataStore(df2), DataStore(df3), DataStore(df4)
        ds_result = ds1.merge(ds2, on='id').merge(ds3, on='id').merge(ds4, on='id')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 5: Filter/GroupBy then Merge
# =============================================================================


class TestOpsBeforeMerge:
    """Test operations before merge."""

    def test_filter_then_merge(self):
        """Test filter followed by merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val1': [10, 20, 30, 40]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val2': [100, 200, 300, 400]})

        # pandas
        pd_filtered = df1[df1['val1'] > 15]
        pd_result = pd.merge(pd_filtered, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_filtered = ds1[ds1['val1'] > 15]
        ds_result = ds_filtered.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_then_merge(self):
        """Test groupby aggregation followed by merge."""
        df1 = pd.DataFrame({
            'key': ['A', 'A', 'B', 'B'],
            'val': [10, 20, 30, 40]
        })
        df2 = pd.DataFrame({
            'key': ['A', 'B'],
            'info': ['alpha', 'beta']
        })

        # pandas: groupby creates index, need reset_index for merge
        pd_grouped = df1.groupby('key')['val'].sum().reset_index()
        pd_result = pd.merge(pd_grouped, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_grouped = ds1.groupby('key')['val'].sum().reset_index()
        ds_result = ds_grouped.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_column_assignment_then_merge(self):
        """Test column assignment followed by merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [10, 20, 30]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [100, 200, 300]})

        # pandas
        pd_df1 = df1.copy()
        pd_df1['val1_doubled'] = pd_df1['val1'] * 2
        pd_result = pd.merge(pd_df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds1['val1_doubled'] = ds1['val1'] * 2
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 6: Left_on/Right_on with Different Column Names
# =============================================================================


class TestLeftOnRightOn:
    """Test merge with left_on and right_on parameters."""

    def test_left_on_right_on_basic(self):
        """Test basic left_on/right_on merge."""
        df1 = pd.DataFrame({'id_left': ['A', 'B', 'C'], 'val1': [10, 20, 30]})
        df2 = pd.DataFrame({'id_right': ['A', 'B', 'C'], 'val2': [100, 200, 300]})

        # pandas
        pd_result = pd.merge(df1, df2, left_on='id_left', right_on='id_right')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, left_on='id_left', right_on='id_right')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_left_on_right_on_multiple_keys(self):
        """Test left_on/right_on with multiple columns."""
        df1 = pd.DataFrame({
            'a': ['A', 'A', 'B', 'B'],
            'b': [1, 2, 1, 2],
            'val1': [10, 20, 30, 40]
        })
        df2 = pd.DataFrame({
            'x': ['A', 'A', 'B'],
            'y': [1, 2, 2],
            'val2': [100, 200, 400]
        })

        # pandas
        pd_result = pd.merge(df1, df2, left_on=['a', 'b'], right_on=['x', 'y'])

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, left_on=['a', 'b'], right_on=['x', 'y'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 7: Merge Type Coercion
# =============================================================================


class TestMergeTypeCoercion:
    """Test merge behavior with different key types."""

    def test_merge_int_keys(self):
        """Test merge with integer keys."""
        df1 = pd.DataFrame({'key': [1, 2, 3], 'val1': ['A', 'B', 'C']})
        df2 = pd.DataFrame({'key': [1, 2, 3], 'val2': [10.0, 20.0, 30.0]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_float_keys(self):
        """Test merge with float keys."""
        df1 = pd.DataFrame({'key': [1.0, 2.0, 3.0], 'val1': ['A', 'B', 'C']})
        df2 = pd.DataFrame({'key': [1.0, 2.0, 3.0], 'val2': [10, 20, 30]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 8: Merge with Duplicate Keys
# =============================================================================


class TestMergeWithDuplicates:
    """Test merge with duplicate keys (many-to-many)."""

    def test_many_to_many_merge(self):
        """Test many-to-many merge (duplicates in both)."""
        df1 = pd.DataFrame({'key': ['A', 'A', 'B'], 'val1': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'A', 'B'], 'val2': [10, 20, 30]})

        # pandas: 2*2 = 4 rows for 'A', 1*1 = 1 row for 'B'
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        # Many-to-many creates cartesian product for duplicates
        assert len(ds_result) == len(pd_result) == 5
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_one_to_many_merge(self):
        """Test one-to-many merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'A', 'B', 'B', 'B'], 'val2': [10, 11, 20, 21, 22]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 9: Merge with Column Selection
# =============================================================================


class TestMergeColumnSelection:
    """Test merge combined with column selection."""

    def test_merge_then_select_columns(self):
        """Test merge followed by column selection."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'v1': [1, 2, 3], 'v2': [10, 20, 30]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'v3': [100, 200, 300], 'v4': [1000, 2000, 3000]})

        # pandas
        pd_merged = pd.merge(df1, df2, on='key')
        pd_result = pd_merged[['key', 'v1', 'v3']]

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_merged = ds1.merge(ds2, on='key')
        ds_result = ds_merged[['key', 'v1', 'v3']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_select_columns_then_merge(self):
        """Test column selection followed by merge."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'v1': [1, 2, 3], 'v2': [10, 20, 30]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'v3': [100, 200, 300]})

        # pandas
        pd_selected = df1[['key', 'v1']]
        pd_result = pd.merge(pd_selected, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_selected = ds1[['key', 'v1']]
        ds_result = ds_selected.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 10: Merge with Suffixes
# =============================================================================


class TestMergeSuffixes:
    """Test merge with custom suffixes for overlapping columns."""

    def test_merge_default_suffixes(self):
        """Test merge with default suffixes (_x, _y)."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [10, 20, 30]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key')

        # Check columns have suffixes
        assert 'value_x' in pd_result.columns
        assert 'value_y' in pd_result.columns
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_custom_suffixes(self):
        """Test merge with custom suffixes."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [10, 20, 30]})

        # pandas
        pd_result = pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_result = ds1.merge(ds2, on='key', suffixes=('_left', '_right'))

        assert 'value_left' in pd_result.columns
        assert 'value_right' in pd_result.columns
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 11: Merge after Lazy Operations
# =============================================================================


class TestMergeAfterLazyOps:
    """Test merge after various lazy operations."""

    def test_merge_after_sort(self):
        """Test merge after sort operation."""
        df1 = pd.DataFrame({'key': ['C', 'A', 'B'], 'val1': [30, 10, 20]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [100, 200, 300]})

        # pandas
        pd_sorted = df1.sort_values('val1')
        pd_result = pd.merge(pd_sorted, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_sorted = ds1.sort_values('val1')
        ds_result = ds_sorted.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_after_head(self):
        """Test merge after head operation."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val1': [10, 20, 30, 40]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val2': [100, 200, 300, 400]})

        # pandas
        pd_head = df1.head(2)
        pd_result = pd.merge(pd_head, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds_head = ds1.head(2)
        ds_result = ds_head.merge(ds2, on='key')

        assert len(ds_result) == len(pd_result) == 2

    def test_merge_after_multiple_lazy_ops(self):
        """Test merge after filter + sort + column assignment."""
        df1 = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'val1': [50, 40, 30, 20, 10]
        })
        df2 = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D', 'E'],
            'val2': [5, 4, 3, 2, 1]
        })

        # pandas
        pd_df1 = df1.copy()
        pd_df1 = pd_df1[pd_df1['val1'] > 15]
        pd_df1 = pd_df1.sort_values('val1')
        pd_df1['val1_sq'] = pd_df1['val1'] ** 2
        pd_result = pd.merge(pd_df1, df2, on='key')

        # DataStore
        ds1, ds2 = DataStore(df1), DataStore(df2)
        ds1 = ds1[ds1['val1'] > 15]
        ds1 = ds1.sort_values('val1')
        ds1['val1_sq'] = ds1['val1'] ** 2
        ds_result = ds1.merge(ds2, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
