"""
Exploratory Batch 64: Advanced Join Operations + GroupBy Cumulative

This batch explores undertested areas identified through architecture analysis:
1. Self-joins (merging a DataFrame with itself)
2. Right outer join operations
3. Full outer join operations
4. Join with empty DataFrames
5. Join with duplicate keys on both sides
6. Cumulative operations within groupby groups
7. Complex merge scenarios with indexes

Discovery method: Architecture-based exploration
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    get_dataframe,
    get_series,
)


class TestSelfJoins:
    """Test self-joins (merging a DataFrame with itself)."""

    def test_self_join_basic(self):
        """Test basic self-join on a single column."""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'parent_id': [None, 1, 1],
            'name': ['Root', 'Child1', 'Child2']
        })
        ds_df = DataStore(pd_df.copy())

        # Self-join to get parent names
        pd_result = pd_df.merge(
            pd_df[['id', 'name']],
            left_on='parent_id',
            right_on='id',
            how='left',
            suffixes=('', '_parent')
        )

        ds_result = ds_df.merge(
            ds_df[['id', 'name']],
            left_on='parent_id',
            right_on='id',
            how='left',
            suffixes=('', '_parent')
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_self_join_multiple_keys(self):
        """Test self-join with multiple join keys."""
        pd_df = pd.DataFrame({
            'dept': ['A', 'A', 'B', 'B'],
            'level': [1, 2, 1, 2],
            'employee': ['Alice', 'Bob', 'Charlie', 'David']
        })
        ds_df = DataStore(pd_df.copy())

        # Self-join same department different levels
        pd_result = pd_df.merge(
            pd_df,
            on='dept',
            suffixes=('_self', '_other')
        )
        pd_result = pd_result[pd_result['level_self'] < pd_result['level_other']]

        ds_result = ds_df.merge(
            ds_df,
            on='dept',
            suffixes=('_self', '_other')
        )
        ds_result = ds_result[ds_result['level_self'] < ds_result['level_other']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_self_join_filter_before_join(self):
        """Test self-join with filter applied before join."""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter then self-join
        pd_filtered = pd_df[pd_df['value'] > 15]
        pd_result = pd_filtered.merge(
            pd_filtered,
            on='category',
            suffixes=('_1', '_2')
        )
        pd_result = pd_result[pd_result['id_1'] < pd_result['id_2']]

        ds_filtered = ds_df[ds_df['value'] > 15]
        ds_result = ds_filtered.merge(
            ds_filtered,
            on='category',
            suffixes=('_1', '_2')
        )
        ds_result = ds_result[ds_result['id_1'] < ds_result['id_2']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestOuterJoins:
    """Test outer join operations."""

    def test_right_join_basic(self):
        """Test basic right outer join."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B', 'C'],
            'left_val': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'key': ['B', 'C', 'D'],
            'right_val': [20, 30, 40]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='right')
        ds_result = ds_left.merge(ds_right, on='key', how='right')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_outer_join_basic(self):
        """Test basic full outer join."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B'],
            'left_val': [1, 2]
        })
        pd_right = pd.DataFrame({
            'key': ['B', 'C'],
            'right_val': [20, 30]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='outer')
        ds_result = ds_left.merge(ds_right, on='key', how='outer')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_outer_join_with_nulls_in_key(self):
        """Test outer join where key column has NULL values."""
        pd_left = pd.DataFrame({
            'key': ['A', None, 'C'],
            'left_val': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'B', None],
            'right_val': [10, 20, 30]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='outer')
        ds_result = ds_left.merge(ds_right, on='key', how='outer')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_outer_join_no_common_keys(self):
        """Test outer join where there are no common keys."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B'],
            'left_val': [1, 2]
        })
        pd_right = pd.DataFrame({
            'key': ['C', 'D'],
            'right_val': [30, 40]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='outer')
        ds_result = ds_left.merge(ds_right, on='key', how='outer')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_right_join_all_keys_match(self):
        """Test right join where all keys match."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B', 'C'],
            'left_val': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'B', 'C'],
            'right_val': [10, 20, 30]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='right')
        ds_result = ds_left.merge(ds_right, on='key', how='right')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestJoinWithDuplicateKeys:
    """Test joins with duplicate keys on both sides."""

    def test_join_duplicate_keys_left(self):
        """Test join where left has duplicate keys."""
        pd_left = pd.DataFrame({
            'key': ['A', 'A', 'B'],
            'left_val': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'B'],
            'right_val': [10, 20]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key')
        ds_result = ds_left.merge(ds_right, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_join_duplicate_keys_right(self):
        """Test join where right has duplicate keys."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B'],
            'left_val': [1, 2]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'A', 'B'],
            'right_val': [10, 15, 20]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key')
        ds_result = ds_left.merge(ds_right, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_join_duplicate_keys_both_sides(self):
        """Test join where both sides have duplicate keys (cartesian product per key)."""
        pd_left = pd.DataFrame({
            'key': ['A', 'A', 'B'],
            'left_val': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'key': ['A', 'A', 'B'],
            'right_val': [10, 15, 20]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key')
        ds_result = ds_left.merge(ds_right, on='key')

        # Should be 4 rows for A (2x2) + 1 row for B = 5 rows
        assert len(get_dataframe(ds_result)) == 5
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestJoinWithEmptyDataFrames:
    """Test joins involving empty DataFrames."""

    def test_left_join_empty_left(self):
        """Test left join with empty left DataFrame."""
        pd_left = pd.DataFrame({'key': pd.Series([], dtype='object'), 'left_val': pd.Series([], dtype='float64')})
        pd_right = pd.DataFrame({
            'key': ['A', 'B'],
            'right_val': [10, 20]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='left')
        ds_result = ds_left.merge(ds_right, on='key', how='left')

        assert len(get_dataframe(ds_result)) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_left_join_empty_right(self):
        """Test left join with empty right DataFrame."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B'],
            'left_val': [1, 2]
        })
        pd_right = pd.DataFrame({'key': pd.Series([], dtype='object'), 'right_val': pd.Series([], dtype='float64')})
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='left')
        ds_result = ds_left.merge(ds_right, on='key', how='left')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_inner_join_both_empty(self):
        """Test inner join with both DataFrames empty."""
        pd_left = pd.DataFrame({'key': pd.Series([], dtype='object'), 'left_val': pd.Series([], dtype='float64')})
        pd_right = pd.DataFrame({'key': pd.Series([], dtype='object'), 'right_val': pd.Series([], dtype='float64')})
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on='key', how='inner')
        ds_result = ds_left.merge(ds_right, on='key', how='inner')

        assert len(get_dataframe(ds_result)) == 0


class TestJoinWithMultipleKeys:
    """Test joins with multiple join keys."""

    def test_join_two_keys(self):
        """Test join on two columns."""
        pd_left = pd.DataFrame({
            'key1': ['A', 'A', 'B'],
            'key2': [1, 2, 1],
            'left_val': [10, 20, 30]
        })
        pd_right = pd.DataFrame({
            'key1': ['A', 'B', 'B'],
            'key2': [1, 1, 2],
            'right_val': [100, 200, 300]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, on=['key1', 'key2'])
        ds_result = ds_left.merge(ds_right, on=['key1', 'key2'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_join_different_column_names(self):
        """Test join with different column names on left and right."""
        pd_left = pd.DataFrame({
            'left_key': ['A', 'B', 'C'],
            'left_val': [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            'right_key': ['B', 'C', 'D'],
            'right_val': [20, 30, 40]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(pd_right, left_on='left_key', right_on='right_key')
        ds_result = ds_left.merge(ds_right, left_on='left_key', right_on='right_key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_join_multiple_different_column_names(self):
        """Test join with multiple different column names."""
        pd_left = pd.DataFrame({
            'lk1': ['A', 'A', 'B'],
            'lk2': [1, 2, 1],
            'left_val': [10, 20, 30]
        })
        pd_right = pd.DataFrame({
            'rk1': ['A', 'B', 'B'],
            'rk2': [1, 1, 2],
            'right_val': [100, 200, 300]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        pd_result = pd_left.merge(
            pd_right,
            left_on=['lk1', 'lk2'],
            right_on=['rk1', 'rk2']
        )
        ds_result = ds_left.merge(
            ds_right,
            left_on=['lk1', 'lk2'],
            right_on=['rk1', 'rk2']
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestGroupByCumulative:
    """Test cumulative operations within groupby groups."""

    def test_groupby_cumsum_basic(self):
        """Test cumsum within groups."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df.groupby('group')['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df.groupby('group')['value'].cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_cummax_basic(self):
        """Test cummax within groups."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 4, 10, 5, 20]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cummax'] = pd_df.groupby('group')['value'].cummax()

        ds_result = ds_df.copy()
        ds_result['cummax'] = ds_df.groupby('group')['value'].cummax()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_cummin_basic(self):
        """Test cummin within groups."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 4, 10, 5, 20]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cummin'] = pd_df.groupby('group')['value'].cummin()

        ds_result = ds_df.copy()
        ds_result['cummin'] = ds_df.groupby('group')['value'].cummin()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_cumprod_basic(self):
        """Test cumprod within groups."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 2, 3, 4]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumprod'] = pd_df.groupby('group')['value'].cumprod()

        ds_result = ds_df.copy()
        ds_result['cumprod'] = ds_df.groupby('group')['value'].cumprod()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_cumsum_with_nulls(self):
        """Test cumsum within groups with NULL values."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1.0, None, 3.0, 10.0, 20.0, None]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df.groupby('group')['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df.groupby('group')['value'].cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_cumulative_multiple_groups(self):
        """Test cumulative with multiple groupby columns."""
        pd_df = pd.DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'B', 'B'],
            'g2': ['X', 'X', 'Y', 'Y', 'X', 'X'],
            'value': [1, 2, 10, 20, 100, 200]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df.groupby(['g1', 'g2'])['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df.groupby(['g1', 'g2'])['value'].cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCumulativeEdgeCases:
    """Test edge cases for cumulative operations."""

    def test_cumsum_single_row_groups(self):
        """Test cumsum with single-row groups."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df.groupby('group')['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df.groupby('group')['value'].cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumulative_empty_dataframe(self):
        """Test cumulative on empty DataFrame."""
        pd_df = pd.DataFrame({'group': pd.Series([], dtype='object'), 'value': pd.Series([], dtype='float64')})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df.groupby('group')['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df.groupby('group')['value'].cumsum()

        assert len(get_dataframe(ds_result)) == 0

    def test_cumsum_with_all_nulls_in_group(self):
        """Test cumsum where entire group is NULL."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [None, None, 10.0, 20.0]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cumsum'] = pd_df.groupby('group')['value'].cumsum()

        ds_result = ds_df.copy()
        ds_result['cumsum'] = ds_df.groupby('group')['value'].cumsum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummax_with_negative_values(self):
        """Test cummax with negative values."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A'],
            'value': [-5, -3, -10]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['cummax'] = pd_df.groupby('group')['value'].cummax()

        ds_result = ds_df.copy()
        ds_result['cummax'] = ds_df.groupby('group')['value'].cummax()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDiffPctChangeAdvanced:
    """Test diff and pct_change edge cases."""

    def test_diff_period_2(self):
        """Test diff with period > 1."""
        pd_df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['diff2'] = pd_df['value'].diff(periods=2)

        ds_result = ds_df.copy()
        ds_result['diff2'] = ds_df['value'].diff(periods=2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_negative_period(self):
        """Test diff with negative period (backward diff)."""
        pd_df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['diff_neg'] = pd_df['value'].diff(periods=-1)

        ds_result = ds_df.copy()
        ds_result['diff_neg'] = ds_df['value'].diff(periods=-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change_with_zeros(self):
        """Test pct_change with zero values (division by zero)."""
        pd_df = pd.DataFrame({
            'value': [0.0, 10.0, 0.0, 20.0]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['pct'] = pd_df['value'].pct_change()

        ds_result = ds_df.copy()
        ds_result['pct'] = ds_df['value'].pct_change()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pct_change_period_2(self):
        """Test pct_change with period > 1."""
        pd_df = pd.DataFrame({
            'value': [100.0, 110.0, 121.0, 133.1, 146.41]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.copy()
        pd_result['pct2'] = pd_df['value'].pct_change(periods=2)

        ds_result = ds_df.copy()
        ds_result['pct2'] = ds_df['value'].pct_change(periods=2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCountsAdvanced:
    """Test value_counts with advanced options."""

    def test_value_counts_bins(self):
        """Test value_counts with bins parameter."""
        pd_df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].value_counts(bins=3, sort=False)
        ds_result = ds_df['value'].value_counts(bins=3, sort=False)

        # Check same bins and counts (dtype may differ due to interval)
        pd_vals = pd_result.values
        ds_vals = get_series(ds_result).values
        np.testing.assert_array_equal(ds_vals, pd_vals)

    def test_value_counts_normalize_true(self):
        """Test value_counts with normalize=True."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'A', 'B', 'C']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].value_counts(normalize=True)
        ds_result = ds_df['category'].value_counts(normalize=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_value_counts_ascending(self):
        """Test value_counts with ascending=True."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'A', 'B', 'C']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].value_counts(ascending=True)
        ds_result = ds_df['category'].value_counts(ascending=True)

        # Order matters for ascending
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_value_counts_dropna_false(self):
        """Test value_counts with dropna=False."""
        pd_df = pd.DataFrame({
            'category': ['A', 'B', None, 'A', None, 'C']
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['category'].value_counts(dropna=False)
        ds_result = ds_df['category'].value_counts(dropna=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestIdxmaxIdxminAdvanced:
    """Test idxmax/idxmin with various index types."""

    def test_idxmax_string_index(self):
        """Test idxmax with string index."""
        pd_df = pd.DataFrame({
            'value': [10, 30, 20]
        }, index=['first', 'second', 'third'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].idxmax()
        ds_result = ds_df['value'].idxmax()

        assert ds_result == pd_result

    def test_idxmin_string_index(self):
        """Test idxmin with string index."""
        pd_df = pd.DataFrame({
            'value': [10, 30, 20]
        }, index=['first', 'second', 'third'])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].idxmin()
        ds_result = ds_df['value'].idxmin()

        assert ds_result == pd_result

    def test_idxmax_with_ties(self):
        """Test idxmax when there are ties (should return first occurrence)."""
        pd_df = pd.DataFrame({
            'value': [10, 30, 30, 20]
        })
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['value'].idxmax()
        ds_result = ds_df['value'].idxmax()

        assert ds_result == pd_result


class TestJoinAfterOperations:
    """Test joins after other operations have been applied."""

    def test_join_after_filter(self):
        """Test join after filter operation."""
        pd_left = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D'],
            'val': [1, 2, 3, 4]
        })
        pd_right = pd.DataFrame({
            'key': ['B', 'C', 'D', 'E'],
            'other': [20, 30, 40, 50]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        # Filter then join
        pd_filtered = pd_left[pd_left['val'] > 1]
        pd_result = pd_filtered.merge(pd_right, on='key')

        ds_filtered = ds_left[ds_left['val'] > 1]
        ds_result = ds_filtered.merge(ds_right, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_join_after_groupby_agg(self):
        """Test join after groupby aggregation."""
        pd_detail = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        pd_lookup = pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'name': ['Group A', 'Group B', 'Group C']
        })
        ds_detail = DataStore(pd_detail.copy())
        ds_lookup = DataStore(pd_lookup.copy())

        # Aggregate then join
        pd_agg = pd_detail.groupby('group')['value'].sum().reset_index()
        pd_result = pd_agg.merge(pd_lookup, on='group')

        ds_agg = ds_detail.groupby('group')['value'].sum().reset_index()
        ds_result = ds_agg.merge(ds_lookup, on='group')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_join_then_groupby(self):
        """Test groupby after join."""
        pd_left = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': ['A', 'A', 'B', 'B']
        })
        pd_right = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'value': [100, 200, 300, 400]
        })
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())

        # Join then groupby
        pd_joined = pd_left.merge(pd_right, on='id')
        pd_result = pd_joined.groupby('category')['value'].sum().reset_index()

        ds_joined = ds_left.merge(ds_right, on='id')
        ds_result = ds_joined.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
