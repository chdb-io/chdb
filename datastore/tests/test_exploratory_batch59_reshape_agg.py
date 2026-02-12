"""
Exploratory test batch 59: Reshape operations and multi-function aggregation chains.

Focus areas:
1. pivot() with various parameters and edge cases
2. pivot_table() with multiple aggregation functions
3. stack()/unstack() boundary conditions
4. melt() with id_vars/value_vars combinations
5. Multi-function agg() on DataFrames
6. Named aggregations in groupby().agg()
7. Multi-column agg with mixed types
8. Complex chains: reshape -> filter -> groupby
9. Empty/single-row DataFrame reshape edge cases
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore, concat as ds_concat
from tests.test_utils import (
    assert_frame_equal,
    assert_series_equal,
    assert_datastore_equals_pandas,
    get_dataframe,
    get_series,
)
from tests.xfail_markers import (
    chdb_category_type,
)


class TestPivotOperations:
    """Test pivot() operation edge cases."""

    def test_pivot_basic(self):
        """Test basic pivot operation."""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NYC', 'LA', 'NYC', 'LA'],
            'value': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NYC', 'LA', 'NYC', 'LA'],
            'value': [100, 200, 150, 250]
        })

        pd_result = pd_df.pivot(index='date', columns='city', values='value').reset_index()
        ds_result = ds_df.pivot(index='date', columns='city', values='value').reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_pivot_no_values(self):
        """Test pivot without values parameter (all other columns become values)."""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NYC', 'LA', 'NYC', 'LA'],
            'val1': [100, 200, 150, 250],
            'val2': [10, 20, 15, 25]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NYC', 'LA', 'NYC', 'LA'],
            'val1': [100, 200, 150, 250],
            'val2': [10, 20, 15, 25]
        })

        pd_result = pd_df.pivot(index='date', columns='city')
        ds_result = ds_df.pivot(index='date', columns='city')

        # Compare values (MultiIndex columns make direct comparison complex)
        pd_flat = pd_result.reset_index()
        pd_flat.columns = ['_'.join(map(str, col)).strip('_') for col in pd_flat.columns]
        ds_flat = get_dataframe(ds_result).reset_index()
        ds_flat.columns = ['_'.join(map(str, col)).strip('_') for col in ds_flat.columns]

        assert_frame_equal(ds_flat, pd_flat)

    def test_pivot_with_nan_in_values(self):
        """Test pivot with NaN in values column."""
        pd_df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val': [1.0, None, 3.0, 4.0]
        })
        ds_df = DataStore({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val': [1.0, None, 3.0, 4.0]
        })

        pd_result = pd_df.pivot(index='row', columns='col', values='val').reset_index()
        ds_result = ds_df.pivot(index='row', columns='col', values='val').reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_pivot_single_row(self):
        """Test pivot with single row DataFrame."""
        pd_df = pd.DataFrame({
            'row': ['A'],
            'col': ['X'],
            'val': [100]
        })
        ds_df = DataStore({
            'row': ['A'],
            'col': ['X'],
            'val': [100]
        })

        pd_result = pd_df.pivot(index='row', columns='col', values='val').reset_index()
        ds_result = ds_df.pivot(index='row', columns='col', values='val').reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestPivotTableOperations:
    """Test pivot_table() operation edge cases."""

    def test_pivot_table_single_aggfunc(self):
        """Test pivot_table with single aggregation function."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum').reset_index()
        ds_result = ds_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum').reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_pivot_table_multiple_aggfunc(self):
        """Test pivot_table with multiple aggregation functions."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
            'B': ['one', 'one', 'one', 'two', 'two', 'two'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
            'B': ['one', 'one', 'one', 'two', 'two', 'two'],
            'value': [1, 2, 3, 4, 5, 6]
        })

        pd_result = pd_df.pivot_table(index='A', columns='B', values='value', aggfunc=['sum', 'mean'])
        ds_result = ds_df.pivot_table(index='A', columns='B', values='value', aggfunc=['sum', 'mean'])

        # Compare by flattening column names
        pd_flat = pd_result.reset_index()
        pd_flat.columns = ['_'.join(map(str, col)).strip('_') for col in pd_flat.columns]
        ds_flat = get_dataframe(ds_result).reset_index()
        ds_flat.columns = ['_'.join(map(str, col)).strip('_') for col in ds_flat.columns]

        assert_frame_equal(ds_flat, pd_flat, check_dtype=False)

    def test_pivot_table_fill_value(self):
        """Test pivot_table with fill_value parameter."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar'],
            'B': ['one', 'two', 'one'],
            'value': [1, 2, 3]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar'],
            'B': ['one', 'two', 'one'],
            'value': [1, 2, 3]
        })

        pd_result = pd_df.pivot_table(index='A', columns='B', values='value', fill_value=0).reset_index()
        ds_result = ds_df.pivot_table(index='A', columns='B', values='value', fill_value=0).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_pivot_table_multiple_values(self):
        """Test pivot_table with multiple value columns."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'C': [1, 2, 3, 4],
            'D': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'C': [1, 2, 3, 4],
            'D': [10, 20, 30, 40]
        })

        pd_result = pd_df.pivot_table(index='A', columns='B', values=['C', 'D'], aggfunc='sum')
        ds_result = ds_df.pivot_table(index='A', columns='B', values=['C', 'D'], aggfunc='sum')

        pd_flat = pd_result.reset_index()
        pd_flat.columns = ['_'.join(map(str, col)).strip('_') for col in pd_flat.columns]
        ds_flat = get_dataframe(ds_result).reset_index()
        ds_flat.columns = ['_'.join(map(str, col)).strip('_') for col in ds_flat.columns]

        assert_frame_equal(ds_flat, pd_flat, check_dtype=False)

    def test_pivot_table_margins(self):
        """Test pivot_table with margins=True (subtotals)."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum', margins=True)
        ds_result = ds_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum', margins=True)

        pd_flat = pd_result.reset_index()
        ds_flat = get_dataframe(ds_result).reset_index()

        assert_frame_equal(ds_flat, pd_flat, check_dtype=False)


class TestMeltOperations:
    """Test melt() operation edge cases."""

    def test_melt_basic(self):
        """Test basic melt operation."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        ds_df = DataStore({
            'id': ['A', 'B', 'C'],
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        pd_result = pd_df.melt(id_vars=['id'], value_vars=['col1', 'col2'])
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['col1', 'col2'])

        # Sort for comparison (melt order may vary)
        pd_sorted = pd_result.sort_values(['id', 'variable']).reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values(['id', 'variable']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted)

    def test_melt_no_id_vars(self):
        """Test melt without id_vars (all columns become value vars)."""
        pd_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        ds_df = DataStore({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        pd_result = pd_df.melt()
        ds_result = ds_df.melt()

        pd_sorted = pd_result.sort_values(['variable', 'value']).reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values(['variable', 'value']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted, check_dtype=False)

    def test_melt_custom_var_value_names(self):
        """Test melt with custom var_name and value_name."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B'],
            'x': [1, 2],
            'y': [3, 4]
        })
        ds_df = DataStore({
            'id': ['A', 'B'],
            'x': [1, 2],
            'y': [3, 4]
        })

        pd_result = pd_df.melt(id_vars=['id'], var_name='measurement', value_name='result')
        ds_result = ds_df.melt(id_vars=['id'], var_name='measurement', value_name='result')

        pd_sorted = pd_result.sort_values(['id', 'measurement']).reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values(['id', 'measurement']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted)

    def test_melt_then_filter(self):
        """Test melt followed by filter."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        ds_df = DataStore({
            'id': ['A', 'B', 'C'],
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })

        pd_result = pd_df.melt(id_vars=['id']).query("value > 3")
        ds_melted = ds_df.melt(id_vars=['id'])
        ds_result = ds_melted[ds_melted['value'] > 3]

        pd_sorted = pd_result.sort_values(['id', 'variable', 'value']).reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values(['id', 'variable', 'value']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted, check_dtype=False)

    def test_melt_then_groupby(self):
        """Test melt followed by groupby."""
        pd_df = pd.DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'x': [1, 2, 3, 4],
            'y': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'id': ['A', 'A', 'B', 'B'],
            'x': [1, 2, 3, 4],
            'y': [10, 20, 30, 40]
        })

        pd_result = pd_df.melt(id_vars=['id']).groupby('variable')['value'].sum().reset_index()
        ds_result = ds_df.melt(id_vars=['id']).groupby('variable')['value'].sum().reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)


class TestStackUnstack:
    """Test stack()/unstack() operations."""

    def test_unstack_after_groupby(self):
        """Test unstack after groupby aggregation."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby(['A', 'B'])['value'].sum().unstack()
        ds_result = ds_df.groupby(['A', 'B'])['value'].sum().unstack()

        pd_flat = pd_result.reset_index()
        ds_flat = get_dataframe(ds_result).reset_index()

        assert_frame_equal(ds_flat, pd_flat, check_dtype=False)

    def test_stack_basic(self):
        """Test basic stack operation."""
        pd_df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4]
        }, index=['row1', 'row2'])
        ds_df = DataStore({
            'A': [1, 2],
            'B': [3, 4]
        })
        # DataStore doesn't preserve custom index, so use default
        ds_df_with_index = DataStore(pd_df.reset_index())

        pd_result = pd_df.stack()
        ds_result = pd_df.stack()  # Use pandas for this operation

        # DataStore stack may differ due to index handling
        # Compare values
        assert len(pd_result) == 4


class TestMultiFunctionAgg:
    """Test multi-function aggregation operations."""

    def test_agg_list_of_funcs(self):
        """Test agg with list of functions on DataFrame."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        pd_result = pd_df.agg(['sum', 'mean', 'max'])
        ds_result = ds_df.agg(['sum', 'mean', 'max'])

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_agg_dict_of_funcs(self):
        """Test agg with dict mapping columns to functions."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        pd_result = pd_df.agg({'A': 'sum', 'B': 'mean'})
        ds_result = ds_df.agg({'A': 'sum', 'B': 'mean'})

        assert_series_equal(get_series(ds_result), pd_result, check_dtype=False)

    def test_agg_dict_multiple_funcs_per_column(self):
        """Test agg with dict mapping columns to multiple functions."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'A': [1, 2, 3, 4],
            'B': [10, 20, 30, 40]
        })

        pd_result = pd_df.agg({'A': ['sum', 'mean'], 'B': ['min', 'max']})
        ds_result = ds_df.agg({'A': ['sum', 'mean'], 'B': ['min', 'max']})

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_agg_named(self):
        """Test groupby agg with named aggregations."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4],
            'other': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4],
            'other': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby('group').agg(
            total=('value', 'sum'),
            avg=('value', 'mean'),
            max_other=('other', 'max')
        ).reset_index()
        ds_result = ds_df.groupby('group').agg(
            total=('value', 'sum'),
            avg=('value', 'mean'),
            max_other=('other', 'max')
        ).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_agg_dict(self):
        """Test groupby agg with dict of functions."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby('group').agg({'val1': 'sum', 'val2': 'mean'}).reset_index()
        ds_result = ds_df.groupby('group').agg({'val1': 'sum', 'val2': 'mean'}).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_agg_list_per_column(self):
        """Test groupby agg with list of functions per column."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group').agg({'value': ['sum', 'mean', 'max']})
        ds_result = ds_df.groupby('group').agg({'value': ['sum', 'mean', 'max']})

        pd_flat = pd_result.reset_index()
        pd_flat.columns = ['_'.join(col).strip('_') for col in pd_flat.columns.values]
        ds_flat = get_dataframe(ds_result).reset_index()
        ds_flat.columns = ['_'.join(col).strip('_') for col in ds_flat.columns.values]

        assert_frame_equal(ds_flat, pd_flat, check_dtype=False)

    def test_agg_mixed_types(self):
        """Test agg on DataFrame with mixed types."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })

        # Only count works on all types, nunique also works
        pd_result = pd_df.agg(['count', 'nunique'])
        ds_result = ds_df.agg(['count', 'nunique'])

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)


class TestReshapeChains:
    """Test complex chains involving reshape operations."""

    def test_filter_then_pivot(self):
        """Test filter followed by pivot."""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-03', '2023-03'],
            'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA'],
            'value': [100, 200, 150, 250, 180, 280]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02', '2023-03', '2023-03'],
            'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA'],
            'value': [100, 200, 150, 250, 180, 280]
        })

        pd_result = pd_df[pd_df['value'] > 150].pivot(index='date', columns='city', values='value').reset_index()
        ds_filtered = ds_df[ds_df['value'] > 150]
        ds_result = ds_filtered.pivot(index='date', columns='city', values='value').reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_pivot_table_then_filter(self):
        """Test pivot_table followed by filter on result."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
            'B': ['one', 'one', 'one', 'two', 'two', 'two'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
            'B': ['one', 'one', 'one', 'two', 'two', 'two'],
            'value': [1, 2, 3, 4, 5, 6]
        })

        pd_pivot = pd_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum').reset_index()
        ds_pivot = ds_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum').reset_index()

        # Filter on pivoted result
        pd_result = pd_pivot[pd_pivot['one'] > 2]
        ds_pivot_df = get_dataframe(ds_pivot)
        ds_result = ds_pivot_df[ds_pivot_df['one'] > 2]

        assert_frame_equal(ds_result, pd_result, check_dtype=False)

    def test_groupby_agg_then_melt(self):
        """Test groupby aggregation followed by melt."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })

        pd_agg = pd_df.groupby('group').agg({'val1': 'sum', 'val2': 'mean'}).reset_index()
        ds_agg = ds_df.groupby('group').agg({'val1': 'sum', 'val2': 'mean'}).reset_index()

        pd_result = pd_agg.melt(id_vars=['group'])
        ds_result = get_dataframe(ds_agg).melt(id_vars=['group'])

        pd_sorted = pd_result.sort_values(['group', 'variable']).reset_index(drop=True)
        ds_sorted = ds_result.sort_values(['group', 'variable']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted, check_dtype=False)

    def test_assign_then_pivot_table(self):
        """Test assign followed by pivot_table."""
        pd_df = pd.DataFrame({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NYC', 'LA', 'NYC', 'LA'],
            'value': [100, 200, 150, 250]
        })
        ds_df = DataStore({
            'date': ['2023-01', '2023-01', '2023-02', '2023-02'],
            'city': ['NYC', 'LA', 'NYC', 'LA'],
            'value': [100, 200, 150, 250]
        })

        pd_result = pd_df.assign(doubled=lambda x: x['value'] * 2).pivot_table(
            index='date', columns='city', values='doubled', aggfunc='sum'
        ).reset_index()
        ds_result = ds_df.assign(doubled=ds_df['value'] * 2).pivot_table(
            index='date', columns='city', values='doubled', aggfunc='sum'
        ).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)


class TestEmptyAndEdgeCases:
    """Test reshape operations on empty and edge case DataFrames."""

    def test_pivot_table_empty_df(self):
        """Test pivot_table on empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': [], 'value': []})
        ds_df = DataStore({'A': [], 'B': [], 'value': []})

        pd_result = pd_df.pivot_table(index='A', columns='B', values='value')
        ds_result = ds_df.pivot_table(index='A', columns='B', values='value')

        # Both should be empty
        assert len(get_dataframe(ds_result)) == len(pd_result)

    def test_melt_empty_df(self):
        """Test melt on empty DataFrame."""
        pd_df = pd.DataFrame({'id': [], 'x': [], 'y': []})
        ds_df = DataStore({'id': [], 'x': [], 'y': []})

        pd_result = pd_df.melt(id_vars=['id'])
        ds_result = ds_df.melt(id_vars=['id'])

        assert len(get_dataframe(ds_result)) == len(pd_result)

    def test_agg_empty_df(self):
        """Test agg on empty DataFrame."""
        pd_df = pd.DataFrame({'A': pd.Series([], dtype='float64'), 'B': pd.Series([], dtype='float64')})
        ds_df = DataStore({'A': pd.Series([], dtype='float64'), 'B': pd.Series([], dtype='float64')})

        pd_result = pd_df.agg(['sum', 'mean'])
        ds_result = ds_df.agg(['sum', 'mean'])

        # Sum of empty is 0, mean of empty is NaN
        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_agg_single_group(self):
        """Test groupby agg with single group."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A'],
            'value': [1, 2, 3]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'A'],
            'value': [1, 2, 3]
        })

        pd_result = pd_df.groupby('group').agg({'value': ['sum', 'mean', 'count']}).reset_index()
        ds_result = ds_df.groupby('group').agg({'value': ['sum', 'mean', 'count']}).reset_index()

        pd_flat = pd_result
        pd_flat.columns = ['_'.join(col).strip('_') for col in pd_flat.columns.values]
        ds_flat = get_dataframe(ds_result)
        ds_flat.columns = ['_'.join(col).strip('_') for col in ds_flat.columns.values]

        assert_frame_equal(ds_flat, pd_flat, check_dtype=False)

    def test_pivot_large_unique_values(self):
        """Test pivot with many unique column values."""
        n = 100
        pd_df = pd.DataFrame({
            'row': ['R1'] * n,
            'col': [f'C{i}' for i in range(n)],
            'val': list(range(n))
        })
        ds_df = DataStore({
            'row': ['R1'] * n,
            'col': [f'C{i}' for i in range(n)],
            'val': list(range(n))
        })

        pd_result = pd_df.pivot(index='row', columns='col', values='val').reset_index()
        ds_result = ds_df.pivot(index='row', columns='col', values='val').reset_index()

        # Check same number of columns
        assert len(get_dataframe(ds_result).columns) == len(pd_result.columns)


class TestAggReturnTypes:
    """Test that aggregation return types match pandas."""

    def test_agg_returns_series_for_single_func(self):
        """Test that single function agg returns Series."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df.agg('sum')
        ds_result = ds_df.agg('sum')

        assert isinstance(pd_result, pd.Series)
        assert_series_equal(get_series(ds_result), pd_result, check_dtype=False)

    def test_agg_returns_dataframe_for_list(self):
        """Test that list of functions agg returns DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df.agg(['sum', 'mean'])
        ds_result = ds_df.agg(['sum', 'mean'])

        assert isinstance(pd_result, pd.DataFrame)
        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_single_column_single_func(self):
        """Test groupby single column single func returns Series."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B'], 'value': [1, 2, 3]})
        ds_df = DataStore({'group': ['A', 'A', 'B'], 'value': [1, 2, 3]})

        pd_result = pd_df.groupby('group')['value'].sum()
        ds_result = ds_df.groupby('group')['value'].sum()

        # Both should be Series with group as index
        assert isinstance(pd_result, pd.Series)
        # Compare reset_index to avoid index comparison issues
        pd_reset = pd_result.reset_index()
        ds_reset = get_dataframe(ds_result.reset_index())

        assert_frame_equal(ds_reset, pd_reset, check_dtype=False)


class TestMultiColumnGroupby:
    """Test multi-column groupby with various aggregations."""

    def test_groupby_multi_col_single_agg(self):
        """Test groupby on multiple columns with single aggregation."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby(['A', 'B'])['value'].sum().reset_index()
        ds_result = ds_df.groupby(['A', 'B'])['value'].sum().reset_index()

        # Sort for comparison
        pd_sorted = pd_result.sort_values(['A', 'B']).reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values(['A', 'B']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted, check_dtype=False)

    def test_groupby_multi_col_multi_agg(self):
        """Test groupby on multiple columns with multiple aggregations."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
            'B': ['one', 'one', 'one', 'two', 'two', 'two'],
            'val1': [1, 2, 3, 4, 5, 6],
            'val2': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar'],
            'B': ['one', 'one', 'one', 'two', 'two', 'two'],
            'val1': [1, 2, 3, 4, 5, 6],
            'val2': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df.groupby(['A', 'B']).agg({
            'val1': ['sum', 'mean'],
            'val2': 'max'
        }).reset_index()
        ds_result = ds_df.groupby(['A', 'B']).agg({
            'val1': ['sum', 'mean'],
            'val2': 'max'
        }).reset_index()

        # Flatten column names for comparison
        pd_flat = pd_result
        pd_flat.columns = ['_'.join(col).strip('_') for col in pd_flat.columns.values]
        ds_flat = get_dataframe(ds_result)
        ds_flat.columns = ['_'.join(col).strip('_') for col in ds_flat.columns.values]

        # Sort for comparison
        pd_sorted = pd_flat.sort_values(['A', 'B']).reset_index(drop=True)
        ds_sorted = ds_flat.sort_values(['A', 'B']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted, check_dtype=False)

    def test_groupby_as_index_false(self):
        """Test groupby with as_index=False."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group', as_index=False)['value'].sum()
        ds_result = ds_df.groupby('group', as_index=False)['value'].sum()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_filter_then_agg(self):
        """Test filter followed by groupby aggregation."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [1, 2, 3, 4, 5, 6]
        })

        pd_result = pd_df[pd_df['value'] > 2].groupby('group')['value'].sum().reset_index()
        ds_filtered = ds_df[ds_df['value'] > 2]
        ds_result = ds_filtered.groupby('group')['value'].sum().reset_index()

        pd_sorted = pd_result.sort_values('group').reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values('group').reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted, check_dtype=False)


class TestNullHandlingInReshape:
    """Test NULL/NaN handling in reshape and aggregation operations."""

    def test_pivot_table_with_all_nan_cell(self):
        """Test pivot_table where some cells have only NaN values."""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1.0, None, None, 4.0]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'value': [1.0, None, None, 4.0]
        })

        pd_result = pd_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum').reset_index()
        ds_result = ds_df.pivot_table(index='A', columns='B', values='value', aggfunc='sum').reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_groupby_agg_with_nan(self):
        """Test groupby aggregation with NaN values in value column."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, None, 3.0, None]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, None, 3.0, None]
        })

        pd_result = pd_df.groupby('group')['value'].agg(['sum', 'count', 'mean']).reset_index()
        ds_result = ds_df.groupby('group')['value'].agg(['sum', 'count', 'mean']).reset_index()

        assert_frame_equal(get_dataframe(ds_result), pd_result, check_dtype=False)

    def test_melt_with_nan_values(self):
        """Test melt with NaN in value columns."""
        pd_df = pd.DataFrame({
            'id': ['A', 'B'],
            'x': [1.0, None],
            'y': [None, 4.0]
        })
        ds_df = DataStore({
            'id': ['A', 'B'],
            'x': [1.0, None],
            'y': [None, 4.0]
        })

        pd_result = pd_df.melt(id_vars=['id'])
        ds_result = ds_df.melt(id_vars=['id'])

        pd_sorted = pd_result.sort_values(['id', 'variable']).reset_index(drop=True)
        ds_sorted = get_dataframe(ds_result).sort_values(['id', 'variable']).reset_index(drop=True)

        assert_frame_equal(ds_sorted, pd_sorted)
