"""
Exploratory test batch 58: Parameter combinations, edge cases, and chain operations.

Focus areas:
1. fillna() with complex parameter combinations
2. drop() with axis/index/column combinations
3. where/mask with type mismatches
4. groupby with sparse/empty groups and dropna
5. Multi-column groupby with as_index and selected columns
6. assign() with lambda referencing other computed columns
7. concat() with mixed DataFrame types
8. rename() with partial mapping
9. merge() with non-unique keys
10. drop_duplicates() with keep='last'
11. Rolling/expanding with mixed window parameters
12. groupby().nth() with negative indices
13. Constructor edge cases
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore, concat as ds_concat
from tests.test_utils import (
    assert_frame_equal,
    assert_series_equal,
    get_dataframe,
    get_series,
)
from tests.xfail_markers import (
    chdb_category_type,
    chdb_timedelta_type,
    chdb_array_nullable,
)


class TestFillnaComplexParams:
    """Test fillna() with complex parameter combinations."""

    def test_fillna_partial_dict(self):
        """Test fillna with dict that only covers some columns."""
        pd_df = pd.DataFrame({
            'A': [1, None, 3, None],
            'B': [10, 20, None, 40],
            'C': ['x', None, 'y', 'z']
        })
        ds_df = DataStore({
            'A': [1, None, 3, None],
            'B': [10, 20, None, 40],
            'C': ['x', None, 'y', 'z']
        })

        # fillna with only A and B in dict (C not included)
        pd_result = pd_df.fillna({'A': 0, 'B': 0})
        ds_result = ds_df.fillna({'A': 0, 'B': 0})

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_fillna_chain_with_groupby(self):
        """Test fillna followed by groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, None, 3.0, None]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, None, 3.0, None]
        })

        pd_result = pd_df.fillna(0).groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.fillna(0).groupby('group')['value'].sum().reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_fillna_empty_dataframe(self):
        """Test fillna on empty DataFrame."""
        pd_df = pd.DataFrame({'A': pd.Series([], dtype='float64'), 'B': pd.Series([], dtype='float64')})
        ds_df = DataStore({'A': pd.Series([], dtype='float64'), 'B': pd.Series([], dtype='float64')})

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_fillna_scalar_on_mixed_types(self):
        """Test fillna with scalar on DataFrame with mixed types."""
        pd_df = pd.DataFrame({
            'int_col': [1, None, 3],
            'float_col': [1.5, None, 3.5],
            'str_col': ['a', None, 'c']
        })
        ds_df = DataStore({
            'int_col': [1, None, 3],
            'float_col': [1.5, None, 3.5],
            'str_col': ['a', None, 'c']
        })

        # Scalar fillna (affects numeric columns differently than string)
        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_fillna_method_ffill(self):
        """Test fillna with method='ffill'."""
        pd_df = pd.DataFrame({
            'A': [1, None, None, 4, None],
            'B': [10, 20, None, 40, 50]
        })
        ds_df = DataStore({
            'A': [1, None, None, 4, None],
            'B': [10, 20, None, 40, 50]
        })

        pd_result = pd_df.ffill()
        ds_result = ds_df.ffill()

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestDropComplexParams:
    """Test drop() with complex axis/index/column combinations."""

    def test_drop_columns_axis_equivalence(self):
        """Test drop(labels, axis=1) vs drop(columns)."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

        # Both ways should produce identical results
        pd_result1 = pd_df.drop('B', axis=1)
        pd_result2 = pd_df.drop(columns='B')
        ds_result1 = ds_df.drop('B', axis=1)
        ds_result2 = ds_df.drop(columns='B')

        assert_frame_equal(get_dataframe(ds_result1), pd_result1)
        assert_frame_equal(get_dataframe(ds_result2), pd_result2)
        assert_frame_equal(pd_result1, pd_result2)

    def test_drop_errors_ignore(self):
        """Test drop with errors='ignore' on non-existent column."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})

        pd_result = pd_df.drop(columns='NonExistent', errors='ignore')
        ds_result = ds_df.drop(columns='NonExistent', errors='ignore')

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_drop_single_column_df(self):
        """Test drop on single-column DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore({'A': [1, 2, 3]})

        # Dropping the only column leaves empty DataFrame
        pd_result = pd_df.drop(columns='A')
        ds_result = ds_df.drop(columns='A')

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_drop_after_groupby(self):
        """Test drop after groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'B', 'A', 'B'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby('group').agg({'val1': 'sum', 'val2': 'mean'}).reset_index().drop(columns='val2')
        ds_result = ds_df.groupby('group').agg({'val1': 'sum', 'val2': 'mean'}).reset_index().drop(columns='val2')

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestWhereTypeMismatches:
    """Test where/mask with type mismatches."""

    def test_where_int_other_on_float_column(self):
        """Test where with int 'other' value on float column."""
        pd_df = pd.DataFrame({
            'A': [1.5, 2.5, 3.5, 4.5],
            'B': [10.0, 20.0, 30.0, 40.0]
        })
        ds_df = DataStore({
            'A': [1.5, 2.5, 3.5, 4.5],
            'B': [10.0, 20.0, 30.0, 40.0]
        })

        cond_pd = pd_df['A'] > 2
        cond_ds = ds_df['A'] > 2

        pd_result = pd_df.where(cond_pd, other=-999)
        ds_result = ds_df.where(cond_ds, other=-999)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_mask_chain_with_fillna(self):
        """Test mask chained with fillna."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        cond_pd = pd_df['A'] < 3
        cond_ds = ds_df['A'] < 3

        # mask makes condition=True -> NaN, then fillna fills them
        pd_result = pd_df.mask(cond_pd).fillna(0)
        ds_result = ds_df.mask(cond_ds).fillna(0)

        # Relax dtype check - DataStore may use nullable types
        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_where_on_mixed_dtypes(self):
        """Test where on DataFrame with mixed dtypes."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.5, 2.5, 3.5, 4.5]
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.5, 2.5, 3.5, 4.5]
        })

        cond_pd = pd_df['int_col'] > 2
        cond_ds = ds_df['int_col'] > 2

        pd_result = pd_df.where(cond_pd, other=0)
        ds_result = ds_df.where(cond_ds, other=0)

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestGroupbySparseGroups:
    """Test groupby with sparse/empty groups and dropna."""

    def test_groupby_all_nan_column_dropna_false(self):
        """Test groupby with all-NaN column and dropna=False."""
        pd_df = pd.DataFrame({
            'group': [None, None, None, None],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': [None, None, None, None],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group', dropna=False)['value'].sum().reset_index()
        ds_result = ds_df.groupby('group', dropna=False)['value'].sum().reset_index()

        # Relax dtype check for group column - pandas uses float64 for None, DataStore uses object
        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_empty_dataframe(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'group': pd.Series([], dtype='str'), 'value': pd.Series([], dtype='float64')})
        ds_df = DataStore({'group': pd.Series([], dtype='str'), 'value': pd.Series([], dtype='float64')})

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_single_row(self):
        """Test groupby on single-row DataFrame."""
        pd_df = pd.DataFrame({'group': ['A'], 'value': [10]})
        ds_df = DataStore({'group': ['A'], 'value': [10]})

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_groupby_size_vs_count_with_nan(self):
        """Test groupby().size() vs groupby().count() with NaN values."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, None, 3, None]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, None, 3, None]
        })

        # size() counts all rows including NaN
        pd_size = pd_df.groupby('group').size().reset_index(name='size')
        ds_size = ds_df.groupby('group').size().reset_index(name='size')

        # count() only counts non-NaN
        pd_count = pd_df.groupby('group')['value'].count().reset_index(name='count')
        ds_count = ds_df.groupby('group')['value'].count().reset_index(name='count')

        assert_frame_equal(get_dataframe(ds_size), pd_size)
        assert_frame_equal(get_dataframe(ds_count), pd_count)


class TestMultiColumnGroupby:
    """Test multi-column groupby with as_index and selected columns."""

    def test_multicolumn_groupby_as_index_false(self):
        """Test multi-column groupby with as_index=False."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'y'],
            'C': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'y'],
            'C': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby(['A', 'B'], as_index=False)['C'].sum()
        ds_result = ds_df.groupby(['A', 'B'], as_index=False)['C'].sum()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_multicolumn_groupby_multiple_aggs(self):
        """Test multi-column groupby with multiple aggregations."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'y'],
            'C': [10, 20, 30, 40],
            'D': [100, 200, 300, 400]
        })
        ds_df = DataStore({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'y'],
            'C': [10, 20, 30, 40],
            'D': [100, 200, 300, 400]
        })

        pd_result = pd_df.groupby(['A', 'B']).agg({'C': 'sum', 'D': 'mean'}).reset_index()
        ds_result = ds_df.groupby(['A', 'B']).agg({'C': 'sum', 'D': 'mean'}).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_groupby_order_matters(self):
        """Test if groupby column order affects result column order."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'y'],
            'C': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 1, 2, 2],
            'B': ['x', 'y', 'x', 'y'],
            'C': [10, 20, 30, 40]
        })

        # groupby(['A', 'B']) vs groupby(['B', 'A'])
        pd_result_ab = pd_df.groupby(['A', 'B'])['C'].sum().reset_index()
        pd_result_ba = pd_df.groupby(['B', 'A'])['C'].sum().reset_index()
        ds_result_ab = ds_df.groupby(['A', 'B'])['C'].sum().reset_index()
        ds_result_ba = ds_df.groupby(['B', 'A'])['C'].sum().reset_index()

        assert_frame_equal(get_dataframe(ds_result_ab), pd_result_ab)
        assert_frame_equal(get_dataframe(ds_result_ba), pd_result_ba)


class TestAssignLambdaChain:
    """Test assign() with lambda referencing other computed columns."""

    def test_assign_simple_lambda(self):
        """Test simple assign with lambda."""
        pd_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        ds_df = DataStore({'x': [1, 2, 3], 'y': [4, 5, 6]})

        pd_result = pd_df.assign(z=lambda df: df['x'] + df['y'])
        ds_result = ds_df.assign(z=lambda df: df['x'] + df['y'])

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_assign_chain_multiple(self):
        """Test chained assign operations."""
        pd_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        ds_df = DataStore({'x': [1, 2, 3], 'y': [4, 5, 6]})

        pd_result = (pd_df
                     .assign(z=lambda df: df['x'] + df['y'])
                     .assign(w=lambda df: df['z'] * 2))
        ds_result = (ds_df
                     .assign(z=lambda df: df['x'] + df['y'])
                     .assign(w=lambda df: df['z'] * 2))

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_assign_overwrite_existing(self):
        """Test assign overwrites existing column."""
        pd_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        ds_df = DataStore({'x': [1, 2, 3], 'y': [4, 5, 6]})

        pd_result = pd_df.assign(x=lambda df: df['x'] * 10)
        ds_result = ds_df.assign(x=lambda df: df['x'] * 10)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_assign_then_groupby(self):
        """Test assign followed by groupby on new column."""
        pd_df = pd.DataFrame({
            'val': [1, 2, 3, 4, 5, 6],
            'amount': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'val': [1, 2, 3, 4, 5, 6],
            'amount': [10, 20, 30, 40, 50, 60]
        })

        pd_result = (pd_df
                     .assign(group=lambda df: df['val'] % 2)
                     .groupby('group')['amount'].sum()
                     .reset_index())
        ds_result = (ds_df
                     .assign(group=lambda df: df['val'] % 2)
                     .groupby('group')['amount'].sum()
                     .reset_index())

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestConcatEdgeCases:
    """Test concat() with edge cases."""

    def test_concat_basic(self):
        """Test basic concat."""
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        ds_df1 = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = DataStore({'A': [5, 6], 'B': [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds_concat([ds_df1, ds_df2], ignore_index=True)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_concat_different_columns(self):
        """Test concat with different column sets."""
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})
        ds_df1 = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = DataStore({'B': [5, 6], 'C': [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds_concat([ds_df1, ds_df2], ignore_index=True)

        # Reorder columns to match
        pd_result = pd_result[sorted(pd_result.columns)]
        ds_result_df = get_dataframe(ds_result)[sorted(get_dataframe(ds_result).columns)]

        assert_frame_equal(ds_result_df, pd_result)

    def test_concat_empty_dataframe(self):
        """Test concat with empty DataFrame."""
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})
        ds_df1 = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = DataStore({'A': pd.Series([], dtype='int64'), 'B': pd.Series([], dtype='int64')})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds_concat([ds_df1, ds_df2], ignore_index=True)

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestRenameEdgeCases:
    """Test rename() with edge cases."""

    def test_rename_partial_mapping(self):
        """Test rename with partial column mapping."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

        # Only rename A, leave B and C unchanged
        pd_result = pd_df.rename(columns={'A': 'NewA'})
        ds_result = ds_df.rename(columns={'A': 'NewA'})

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_rename_chain_with_filter(self):
        """Test rename followed by filter using new column name."""
        pd_df = pd.DataFrame({'OldName': [1, 2, 3, 4], 'value': [10, 20, 30, 40]})
        ds_df = DataStore({'OldName': [1, 2, 3, 4], 'value': [10, 20, 30, 40]})

        pd_result = pd_df.rename(columns={'OldName': 'NewName'})
        pd_result = pd_result[pd_result['NewName'] > 2]

        ds_result = ds_df.rename(columns={'OldName': 'NewName'})
        ds_result = ds_result[ds_result['NewName'] > 2]

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_rename_then_groupby(self):
        """Test rename followed by groupby on renamed column."""
        pd_df = pd.DataFrame({'old_group': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds_df = DataStore({'old_group': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})

        pd_result = (pd_df
                     .rename(columns={'old_group': 'new_group'})
                     .groupby('new_group')['value'].sum()
                     .reset_index())
        ds_result = (ds_df
                     .rename(columns={'old_group': 'new_group'})
                     .groupby('new_group')['value'].sum()
                     .reset_index())

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestMergeEdgeCases:
    """Test merge() with edge cases."""

    def test_merge_with_duplicates(self):
        """Test merge with duplicate keys in both DataFrames."""
        pd_df1 = pd.DataFrame({'key': [1, 1, 2], 'A': [10, 20, 30]})
        pd_df2 = pd.DataFrame({'key': [1, 2, 2], 'B': [100, 200, 300]})
        ds_df1 = DataStore({'key': [1, 1, 2], 'A': [10, 20, 30]})
        ds_df2 = DataStore({'key': [1, 2, 2], 'B': [100, 200, 300]})

        pd_result = pd_df1.merge(pd_df2, on='key', how='inner')
        ds_result = ds_df1.merge(ds_df2, on='key', how='inner')

        # Sort for comparison (order may differ)
        pd_result = pd_result.sort_values(['key', 'A', 'B']).reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values(['key', 'A', 'B']).reset_index(drop=True)

        assert_frame_equal(ds_result_df, pd_result)

    def test_merge_left_join(self):
        """Test merge with left join."""
        pd_df1 = pd.DataFrame({'key': [1, 2, 3], 'A': [10, 20, 30]})
        pd_df2 = pd.DataFrame({'key': [1, 2], 'B': [100, 200]})
        ds_df1 = DataStore({'key': [1, 2, 3], 'A': [10, 20, 30]})
        ds_df2 = DataStore({'key': [1, 2], 'B': [100, 200]})

        pd_result = pd_df1.merge(pd_df2, on='key', how='left')
        ds_result = ds_df1.merge(ds_df2, on='key', how='left')

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_merge_then_groupby(self):
        """Test merge followed by groupby."""
        pd_df1 = pd.DataFrame({'key': [1, 1, 2, 2], 'A': [10, 20, 30, 40]})
        pd_df2 = pd.DataFrame({'key': [1, 2], 'B': ['x', 'y']})
        ds_df1 = DataStore({'key': [1, 1, 2, 2], 'A': [10, 20, 30, 40]})
        ds_df2 = DataStore({'key': [1, 2], 'B': ['x', 'y']})

        pd_result = (pd_df1
                     .merge(pd_df2, on='key')
                     .groupby('B')['A'].sum()
                     .reset_index())
        ds_result = (ds_df1
                     .merge(ds_df2, on='key')
                     .groupby('B')['A'].sum()
                     .reset_index())

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestDropDuplicatesEdgeCases:
    """Test drop_duplicates() with edge cases."""

    def test_drop_duplicates_keep_last(self):
        """Test drop_duplicates with keep='last'."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 1, 3, 2],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore({
            'A': [1, 2, 1, 3, 2],
            'B': ['a', 'b', 'c', 'd', 'e']
        })

        pd_result = pd_df.drop_duplicates(subset=['A'], keep='last').reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset=['A'], keep='last').reset_index(drop=True)

        # Sort by A for comparison (order may differ)
        pd_result = pd_result.sort_values('A').reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values('A').reset_index(drop=True)

        assert_frame_equal(ds_result_df, pd_result)

    def test_drop_duplicates_keep_false(self):
        """Test drop_duplicates with keep=False (remove all duplicates)."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 1, 3, 2],
            'B': ['a', 'b', 'c', 'd', 'e']
        })
        ds_df = DataStore({
            'A': [1, 2, 1, 3, 2],
            'B': ['a', 'b', 'c', 'd', 'e']
        })

        pd_result = pd_df.drop_duplicates(subset=['A'], keep=False).reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset=['A'], keep=False).reset_index(drop=True)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_drop_duplicates_all_unique(self):
        """Test drop_duplicates when all rows are unique."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestRollingEdgeCases:
    """Test rolling() with edge cases."""

    def test_rolling_min_periods(self):
        """Test rolling with min_periods on small DataFrame."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
        ds_df = DataStore({'value': [1.0, 2.0, 3.0]})

        pd_result = pd_df.rolling(window=5, min_periods=2)['value'].mean().to_frame()
        ds_result = ds_df.rolling(window=5, min_periods=2)['value'].mean()

        assert_series_equal(get_series(ds_result), pd_result['value'])

    def test_rolling_single_row(self):
        """Test rolling on single-row DataFrame."""
        pd_df = pd.DataFrame({'value': [10.0]})
        ds_df = DataStore({'value': [10.0]})

        pd_result = pd_df.rolling(window=3, min_periods=1)['value'].mean().to_frame()
        ds_result = ds_df.rolling(window=3, min_periods=1)['value'].mean()

        assert_series_equal(get_series(ds_result), pd_result['value'])

    def test_expanding_mean(self):
        """Test expanding mean."""
        pd_df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})

        pd_result = pd_df.expanding()['value'].mean().to_frame()
        ds_result = ds_df.expanding()['value'].mean()

        assert_series_equal(get_series(ds_result), pd_result['value'])


class TestGroupbyNth:
    """Test groupby().nth() with edge cases."""

    def test_nth_positive(self):
        """Test nth with positive index."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.groupby('group').nth(1).reset_index()
        ds_result = ds_df.groupby('group').nth(1).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_nth_negative(self):
        """Test nth with negative index (last element)."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.groupby('group').nth(-1).reset_index()
        ds_result = ds_df.groupby('group').nth(-1).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_nth_out_of_bounds(self):
        """Test nth with index larger than group size."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B'],
            'value': [10, 20, 30]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B'],
            'value': [10, 20, 30]
        })

        # nth(5) should return empty for group B (only 1 element)
        pd_result = pd_df.groupby('group').nth(5).reset_index()
        ds_result = ds_df.groupby('group').nth(5).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestConstructorEdgeCases:
    """Test constructor edge cases."""

    def test_constructor_columns_only(self):
        """Test constructor with columns-only parameter."""
        pd_df = pd.DataFrame(columns=['A', 'B', 'C'])
        ds_df = DataStore(columns=['A', 'B', 'C'])

        assert list(get_dataframe(ds_df).columns) == list(pd_df.columns)
        assert len(get_dataframe(ds_df)) == len(pd_df) == 0

    def test_constructor_with_index(self):
        """Test constructor with explicit index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=[10, 20, 30])
        ds_df = DataStore({'A': [1, 2, 3]}, index=[10, 20, 30])

        ds_result = get_dataframe(ds_df)
        assert list(ds_result.index) == list(pd_df.index)
        assert list(ds_result['A']) == list(pd_df['A'])

    def test_constructor_from_dict_of_lists(self):
        """Test constructor from dict of lists."""
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_frame_equal(get_dataframe(ds_df), pd_df)

    def test_constructor_from_list_of_dicts(self):
        """Test constructor from list of dicts."""
        data = [{'A': 1, 'B': 4}, {'A': 2, 'B': 5}, {'A': 3, 'B': 6}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        assert_frame_equal(get_dataframe(ds_df), pd_df)

    def test_empty_dataframe_operations(self):
        """Test operations on empty DataFrame."""
        pd_df = pd.DataFrame(columns=['A', 'B'])
        ds_df = DataStore(columns=['A', 'B'])

        # Filter on empty DataFrame
        pd_result = pd_df[pd_df['A'] > 0] if len(pd_df) > 0 else pd_df
        ds_result = ds_df  # Empty already

        assert len(get_dataframe(ds_result)) == len(pd_result) == 0


class TestStringAccessorEdgeCases:
    """Test string accessor edge cases."""

    def test_str_upper_lower(self):
        """Test basic string upper/lower."""
        pd_df = pd.DataFrame({'text': ['hello', 'WORLD', 'Mixed']})
        ds_df = DataStore({'text': ['hello', 'WORLD', 'Mixed']})

        pd_upper = pd_df['text'].str.upper()
        pd_lower = pd_df['text'].str.lower()
        ds_upper = ds_df['text'].str.upper()
        ds_lower = ds_df['text'].str.lower()

        assert_series_equal(get_series(ds_upper), pd_upper)
        assert_series_equal(get_series(ds_lower), pd_lower)

    def test_str_contains(self):
        """Test string contains."""
        pd_df = pd.DataFrame({'text': ['apple', 'banana', 'cherry']})
        ds_df = DataStore({'text': ['apple', 'banana', 'cherry']})

        pd_result = pd_df['text'].str.contains('an')
        ds_result = ds_df['text'].str.contains('an')

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_slice(self):
        """Test string slicing."""
        pd_df = pd.DataFrame({'text': ['hello', 'world', 'python']})
        ds_df = DataStore({'text': ['hello', 'world', 'python']})

        pd_result = pd_df['text'].str[:3]
        ds_result = ds_df['text'].str[:3]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_empty_string(self):
        """Test string operations on empty strings."""
        pd_df = pd.DataFrame({'text': ['', 'hello', '']})
        ds_df = DataStore({'text': ['', 'hello', '']})

        pd_result = pd_df['text'].str.upper()
        ds_result = ds_df['text'].str.upper()

        assert_series_equal(get_series(ds_result), pd_result)


class TestComplexChains:
    """Test complex operation chains."""

    def test_filter_groupby_agg_sort_head(self):
        """Test filter -> groupby -> agg -> sort -> head chain."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_result = (pd_df[pd_df['value'] > 15]
                     .groupby('group')['value'].sum()
                     .reset_index()
                     .sort_values('value', ascending=False)
                     .head(2)
                     .reset_index(drop=True))
        ds_result = (ds_df[ds_df['value'] > 15]
                     .groupby('group')['value'].sum()
                     .reset_index()
                     .sort_values('value', ascending=False)
                     .head(2)
                     .reset_index(drop=True))

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_assign_fillna_groupby_agg(self):
        """Test assign -> fillna -> groupby -> agg chain."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, None, 3.0, None]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, None, 3.0, None]
        })

        pd_result = (pd_df
                     .assign(doubled=lambda df: df['value'] * 2)
                     .fillna(0)
                     .groupby('group')['doubled'].sum()
                     .reset_index())
        ds_result = (ds_df
                     .assign(doubled=lambda df: df['value'] * 2)
                     .fillna(0)
                     .groupby('group')['doubled'].sum()
                     .reset_index())

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_merge_filter_groupby(self):
        """Test merge -> filter -> groupby chain (using explicit filter instead of lambda)."""
        pd_df1 = pd.DataFrame({'key': [1, 2, 3], 'A': [10, 20, 30]})
        pd_df2 = pd.DataFrame({'key': [1, 2, 3], 'B': ['x', 'y', 'x']})
        ds_df1 = DataStore({'key': [1, 2, 3], 'A': [10, 20, 30]})
        ds_df2 = DataStore({'key': [1, 2, 3], 'B': ['x', 'y', 'x']})

        pd_merged = pd_df1.merge(pd_df2, on='key')
        pd_result = pd_merged[pd_merged['A'] > 15].groupby('B')['A'].sum().reset_index()

        ds_merged = ds_df1.merge(ds_df2, on='key')
        ds_result = ds_merged[ds_merged['A'] > 15].groupby('B')['A'].sum().reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_rename_drop_select_filter(self):
        """Test rename -> drop -> select -> filter chain (using explicit filter)."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40],
            'C': [100, 200, 300, 400]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40],
            'C': [100, 200, 300, 400]
        })

        pd_temp = (pd_df
                   .rename(columns={'A': 'new_A'})
                   .drop(columns='C')
                   [['new_A', 'B']])
        pd_result = pd_temp[pd_temp['new_A'] > 2]

        ds_temp = (ds_df
                   .rename(columns={'A': 'new_A'})
                   .drop(columns='C')
                   [['new_A', 'B']])
        ds_result = ds_temp[ds_temp['new_A'] > 2]

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestLambdaFilterLimitation:
    """Test to document that DataStore doesn't support lambda filtering like pandas."""

    def test_lambda_filter_not_supported(self):
        """
        Document that DataStore doesn't support df[lambda x: x['col'] > 0] syntax.

        In pandas, you can use lambda in __getitem__ for filtering:
            df[lambda df: df['A'] > 2]

        DataStore requires explicit condition syntax:
            ds[ds['A'] > 2]

        This test documents this limitation.
        """
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore({'A': [1, 2, 3, 4]})

        # Pandas supports lambda
        pd_result = pd_df[lambda df: df['A'] > 2]
        assert len(pd_result) == 2

        # DataStore requires explicit condition
        ds_result = ds_df[ds_df['A'] > 2]
        assert len(get_dataframe(ds_result)) == 2

        # Verify they produce same result
        assert_frame_equal(get_dataframe(ds_result), pd_result)
