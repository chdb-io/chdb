"""
Exploratory Batch 37: GroupBy + Apply/Transform + Reshape Chain Edge Cases

Focus areas:
1. GroupBy.apply() followed by pivot_table() with NA values
2. GroupBy.transform() with NaN handling in complex aggregation chains
3. Multi-level groupby with subsequent melt/stack operations
4. Type preservation when apply() returns Series with different dtype than input
5. Empty DataFrame edge cases for apply/transform
6. Apply with inconsistent return types
7. GroupBy agg() with multiple functions followed by reshape

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
Discovery date: 2026-01-06
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_series

# =============================================================================
# Test Group 1: GroupBy Transform Chains
# =============================================================================


class TestGroupByTransformChains:
    """Test groupby.transform() in various chain scenarios."""

    def test_transform_with_filter(self):
        """Test groupby transform followed by filter."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['group_mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_result = pd_df[pd_df['value'] > pd_df['group_mean']]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['group_mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df[ds_df['value'] > ds_df['group_mean']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_transform_sum_with_groupby_again(self):
        """Test groupby transform followed by another groupby."""
        df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'subcat': ['X', 'Y', 'X', 'Y'],
            'value': [10, 20, 30, 40]
        })

        # pandas
        pd_df = df.copy()
        pd_df['cat_sum'] = pd_df.groupby('cat')['value'].transform('sum')
        pd_result = pd_df.groupby('subcat')['cat_sum'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['cat_sum'] = ds_df.groupby('cat')['value'].transform('sum')
        ds_result = ds_df.groupby('subcat')['cat_sum'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_transforms_in_chain(self):
        """Test multiple transform operations in sequence."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 10, 20]
        })

        # pandas
        pd_df = df.copy()
        pd_df['mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_df['std'] = pd_df.groupby('group')['value'].transform('std')
        pd_df['normalized'] = (pd_df['value'] - pd_df['mean']) / pd_df['std']
        pd_result = pd_df[['group', 'value', 'normalized']]

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_df['std'] = ds_df.groupby('group')['value'].transform('std')
        ds_df['normalized'] = (ds_df['value'] - ds_df['mean']) / ds_df['std']
        ds_result = ds_df[['group', 'value', 'normalized']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False, rtol=1e-5)

    def test_transform_with_na_values(self):
        """Test transform with NA values in data."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10.0, np.nan, 30.0, 40.0]
        })

        # pandas
        pd_df = df.copy()
        pd_df['group_mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['group_mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 2: GroupBy Apply Scenarios
# Note: groupby.apply() with custom functions has limited support in DataStore.
# Most apply use cases are better served by built-in aggregations.
# =============================================================================


class TestGroupByApply:
    """Test groupby.apply() with various scenarios."""

    def test_apply_lambda_returning_scalar(self):
        """Test apply with lambda that returns scalar."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_result = df.groupby('group')['value'].apply(lambda x: x.max() - x.min())

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group')['value'].apply(lambda x: x.max() - x.min())

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_apply_with_custom_function(self):
        """Test apply with a custom aggregation function."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 10, 20]
        })

        def custom_agg(x):
            return x.sum() / len(x) * 2

        # pandas
        pd_result = df.groupby('group')['value'].apply(custom_agg)

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group')['value'].apply(custom_agg)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_apply_filter_then_groupby_agg(self):
        """Test filter followed by groupby agg (instead of apply)."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        # pandas - use sum() instead of apply(sum)
        pd_filtered = df[df['value'] > 15]
        pd_result = pd_filtered.groupby('group')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_filtered = ds_df[ds_df['value'] > 15]
        ds_result = ds_filtered.groupby('group')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 3: GroupBy Agg + Reshape
# =============================================================================


class TestGroupByAggReshape:
    """Test groupby aggregation followed by reshape operations."""

    def test_groupby_agg_dict_then_reset_index(self):
        """Test groupby with dict agg followed by reset_index."""
        df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val1': [10, 20, 30, 40],
            'val2': [100, 200, 300, 400]
        })

        # pandas
        pd_agg = df.groupby('cat').agg({'val1': 'sum', 'val2': 'mean'})
        pd_result = pd_agg.reset_index()

        # DataStore
        ds_df = DataStore(df)
        ds_agg = ds_df.groupby('cat').agg({'val1': 'sum', 'val2': 'mean'})
        ds_result = ds_agg.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_multiple_agg_then_filter_on_reset(self):
        """Test groupby with multiple aggregations followed by filter after reset_index.
        
        Note: Filtering on aggregated column names ('sum', 'mean') in the lazy result
        requires reset_index first since the aggregated columns have internal aliases.
        """
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_agg = df.groupby('group')['value'].agg(['sum', 'mean', 'count']).reset_index()
        pd_result = pd_agg[pd_agg['sum'] > 50]

        # DataStore - need reset_index to materialize column names for filter
        ds_df = DataStore(df)
        ds_agg = ds_df.groupby('group')['value'].agg(['sum', 'mean', 'count']).reset_index()
        ds_result = ds_agg[ds_agg['sum'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_then_sort(self):
        """Test groupby aggregation followed by sort."""
        df = pd.DataFrame({
            'cat': ['C', 'A', 'B', 'C', 'A', 'B'],
            'value': [1, 2, 3, 4, 5, 6]
        })

        # pandas
        pd_agg = df.groupby('cat')['value'].sum()
        pd_result = pd_agg.sort_values(ascending=False)

        # DataStore
        ds_df = DataStore(df)
        ds_agg = ds_df.groupby('cat')['value'].sum()
        ds_result = ds_agg.sort_values(ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


# =============================================================================
# Test Group 4: Pivot Table Scenarios
# =============================================================================


class TestPivotTableEdgeCases:
    """Test pivot_table edge cases."""

    def test_pivot_table_basic(self):
        """Test basic pivot_table."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val': [1, 2, 3, 4]
        })

        # pandas
        pd_result = df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_table_with_fill_value(self):
        """Test pivot_table with missing combinations and fill_value."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B'],
            'col': ['X', 'Y', 'X'],
            'val': [1, 2, 3]
        })

        # pandas
        pd_result = df.pivot_table(values='val', index='row', columns='col', aggfunc='sum', fill_value=0)

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values='val', index='row', columns='col', aggfunc='sum', fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_table_with_multiple_values(self):
        """Test pivot_table with multiple value columns."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val1': [1, 2, 3, 4],
            'val2': [10, 20, 30, 40]
        })

        # pandas
        pd_result = df.pivot_table(values=['val1', 'val2'], index='row', columns='col', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values=['val1', 'val2'], index='row', columns='col', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_then_pivot(self):
        """Test filter followed by pivot_table."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B', 'C', 'C'],
            'col': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'val': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_filtered = df[df['val'] > 25]
        pd_result = pd_filtered.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_filtered = ds_df[ds_df['val'] > 25]
        ds_result = ds_filtered.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 5: Melt Operations
# =============================================================================


class TestMeltEdgeCases:
    """Test melt operation edge cases."""

    def test_melt_basic(self):
        """Test basic melt."""
        df = pd.DataFrame({
            'id': ['A', 'B'],
            'x': [1, 2],
            'y': [3, 4]
        })

        # pandas
        pd_result = df.melt(id_vars=['id'], value_vars=['x', 'y'])

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['x', 'y'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_with_var_name(self):
        """Test melt with custom var_name and value_name."""
        df = pd.DataFrame({
            'id': ['A', 'B'],
            'col1': [10, 20],
            'col2': [30, 40]
        })

        # pandas
        pd_result = df.melt(id_vars=['id'], value_vars=['col1', 'col2'],
                           var_name='metric', value_name='measurement')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['col1', 'col2'],
                              var_name='metric', value_name='measurement')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_then_groupby(self):
        """Test melt followed by groupby."""
        df = pd.DataFrame({
            'id': ['A', 'A', 'B', 'B'],
            'x': [1, 2, 3, 4],
            'y': [10, 20, 30, 40]
        })

        # pandas
        pd_melted = df.melt(id_vars=['id'], value_vars=['x', 'y'])
        pd_result = pd_melted.groupby('variable')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_melted = ds_df.melt(id_vars=['id'], value_vars=['x', 'y'])
        ds_result = ds_melted.groupby('variable')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_then_melt(self):
        """Test filter followed by melt."""
        df = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'x': [10, 20, 30],
            'y': [100, 200, 300]
        })

        # pandas
        pd_filtered = df[df['x'] > 15]
        pd_result = pd_filtered.melt(id_vars=['id'], value_vars=['x', 'y'])

        # DataStore
        ds_df = DataStore(df)
        ds_filtered = ds_df[ds_df['x'] > 15]
        ds_result = ds_filtered.melt(id_vars=['id'], value_vars=['x', 'y'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 6: Stack/Unstack Operations
# =============================================================================


class TestStackUnstackEdgeCases:
    """Test stack/unstack edge cases."""

    def test_unstack_after_groupby(self):
        """Test unstack after groupby aggregation."""
        df = pd.DataFrame({
            'cat1': ['A', 'A', 'B', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4]
        })

        # pandas
        pd_grouped = df.groupby(['cat1', 'cat2'])['value'].sum()
        pd_result = pd_grouped.unstack()

        # DataStore
        ds_df = DataStore(df)
        ds_grouped = ds_df.groupby(['cat1', 'cat2'])['value'].sum()
        ds_result = ds_grouped.unstack()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_unstack_with_fill_value(self):
        """Test unstack with fill_value for missing combinations."""
        df = pd.DataFrame({
            'cat1': ['A', 'A', 'B'],
            'cat2': ['X', 'Y', 'X'],
            'value': [1, 2, 3]
        })

        # pandas
        pd_grouped = df.groupby(['cat1', 'cat2'])['value'].sum()
        pd_result = pd_grouped.unstack(fill_value=0)

        # DataStore
        ds_df = DataStore(df)
        ds_grouped = ds_df.groupby(['cat1', 'cat2'])['value'].sum()
        ds_result = ds_grouped.unstack(fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_as_unstack_alternative(self):
        """Test pivot_table as alternative to groupby().unstack()."""
        df = pd.DataFrame({
            'cat1': ['A', 'A', 'B', 'B'],
            'cat2': ['X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4]
        })

        # pandas using pivot_table
        pd_result = df.pivot_table(values='value', index='cat1', columns='cat2', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values='value', index='cat1', columns='cat2', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 7: Empty DataFrame Edge Cases
# =============================================================================


class TestEmptyDataFrameEdgeCases:
    """Test operations on empty DataFrames."""

    def test_groupby_on_empty(self):
        """Test groupby on empty DataFrame."""
        df = pd.DataFrame({'group': [], 'value': []})
        df['group'] = df['group'].astype(str)
        df['value'] = df['value'].astype(float)

        # pandas
        pd_result = df.groupby('group')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_transform_on_empty(self):
        """Test transform on empty DataFrame."""
        df = pd.DataFrame({'group': [], 'value': []})
        df['group'] = df['group'].astype(str)
        df['value'] = df['value'].astype(float)

        # pandas
        pd_df = df.copy()
        pd_df['mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_to_empty_then_groupby(self):
        """Test filter that produces empty result followed by groupby."""
        df = pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })

        # pandas
        pd_filtered = df[df['value'] > 100]  # Empty result
        pd_result = pd_filtered.groupby('group')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_filtered = ds_df[ds_df['value'] > 100]
        ds_result = ds_filtered.groupby('group')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_on_empty(self):
        """Test pivot_table on empty DataFrame."""
        df = pd.DataFrame({'row': [], 'col': [], 'val': []})
        df['row'] = df['row'].astype(str)
        df['col'] = df['col'].astype(str)
        df['val'] = df['val'].astype(float)

        # pandas
        pd_result = df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 8: Single Row Edge Cases
# =============================================================================


class TestSingleRowEdgeCases:
    """Test operations on single-row DataFrames."""

    def test_groupby_single_row(self):
        """Test groupby on single row."""
        df = pd.DataFrame({'group': ['A'], 'value': [100]})

        # pandas
        pd_result = df.groupby('group')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_transform_single_row(self):
        """Test transform on single row."""
        df = pd.DataFrame({'group': ['A'], 'value': [100.0]})

        # pandas
        pd_df = df.copy()
        pd_df['mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_single_row(self):
        """Test pivot_table with single row."""
        df = pd.DataFrame({
            'row': ['A'],
            'col': ['X'],
            'val': [100]
        })

        # pandas
        pd_result = df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 9: Complex Chain Operations
# =============================================================================


class TestComplexChains:
    """Test complex operation chains."""

    def test_filter_groupby_agg_sort_head(self):
        """Test complex chain: filter -> groupby -> agg -> sort -> head."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
            'subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Z'],
            'amount': [100, 200, 300, 400, 500, 600, 700]
        })

        # pandas
        pd_result = (df[df['amount'] > 150]
                     .groupby('category')['amount']
                     .sum()
                     .sort_values(ascending=False)
                     .head(2))

        # DataStore
        ds_df = DataStore(df)
        ds_result = (ds_df[ds_df['amount'] > 150]
                     .groupby('category')['amount']
                     .sum()
                     .sort_values(ascending=False)
                     .head(2))

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_transform_filter_groupby_agg(self):
        """Test chain: groupby -> transform -> filter -> groupby -> agg."""
        df = pd.DataFrame({
            'cat1': ['A', 'A', 'B', 'B', 'B'],
            'cat2': ['X', 'X', 'Y', 'Y', 'Y'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_df = df.copy()
        pd_df['group_mean'] = pd_df.groupby('cat1')['value'].transform('mean')
        pd_filtered = pd_df[pd_df['value'] > pd_df['group_mean']]
        pd_result = pd_filtered.groupby('cat2')['value'].sum()

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['group_mean'] = ds_df.groupby('cat1')['value'].transform('mean')
        ds_filtered = ds_df[ds_df['value'] > ds_df['group_mean']]
        ds_result = ds_filtered.groupby('cat2')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_groupby_pivot(self):
        """Test chain: melt -> groupby -> pivot (round-trip-ish)."""
        df = pd.DataFrame({
            'id': ['A', 'B'],
            'x': [10, 20],
            'y': [30, 40]
        })

        # pandas
        pd_melted = df.melt(id_vars=['id'], value_vars=['x', 'y'])
        pd_grouped = pd_melted.groupby(['id', 'variable'])['value'].sum().reset_index()
        pd_result = pd_grouped.pivot_table(values='value', index='id', columns='variable', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_melted = ds_df.melt(id_vars=['id'], value_vars=['x', 'y'])
        ds_grouped = ds_melted.groupby(['id', 'variable'])['value'].sum().reset_index()
        ds_result = ds_grouped.pivot_table(values='value', index='id', columns='variable', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_assign_groupby_sort(self):
        """Test chain: filter -> assign column -> groupby -> sort."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        # pandas
        pd_df = df[df['value'] > 15].copy()
        pd_df['doubled'] = pd_df['value'] * 2
        pd_result = pd_df.groupby('category')['doubled'].sum().sort_values()

        # DataStore
        ds_df = DataStore(df)
        ds_filtered = ds_df[ds_df['value'] > 15]
        ds_filtered['doubled'] = ds_filtered['value'] * 2
        ds_result = ds_filtered.groupby('category')['doubled'].sum().sort_values()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


# =============================================================================
# Test Group 10: NA Value Handling in Chains
# =============================================================================


class TestNAValueHandling:
    """Test NA value handling in various chain operations."""
    def test_groupby_with_na_in_key_dropna_true(self):
        """Test groupby with NA values in grouping key (dropna=True).
        
        Note: chDB treats NULL differently from pandas in groupby.
        """
        df = pd.DataFrame({
            'group': ['A', 'A', np.nan, 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas (default dropna=True)
        pd_result = df.groupby('group', dropna=True)['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group', dropna=True)['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_without_na_in_key(self):
        """Test groupby without NA values in grouping key - simple case."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })

        # pandas
        pd_result = df.groupby('group')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_transform_na_in_values(self):
        """Test transform with NA values in value column."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10.0, np.nan, 30.0, np.nan]
        })

        # pandas
        pd_df = df.copy()
        pd_df['filled'] = pd_df.groupby('group')['value'].transform(
            lambda x: x.fillna(x.mean())
        )
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['filled'] = ds_df.groupby('group')['value'].transform(
            lambda x: x.fillna(x.mean())
        )
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_pivot_with_na_values(self):
        """Test pivot_table with NA values."""
        df = pd.DataFrame({
            'row': ['A', 'A', 'B', 'B'],
            'col': ['X', 'Y', 'X', 'Y'],
            'val': [1.0, np.nan, 3.0, 4.0]
        })

        # pandas
        pd_result = df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.pivot_table(values='val', index='row', columns='col', aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_with_na_in_values(self):
        """Test melt with NA values in melted columns."""
        df = pd.DataFrame({
            'id': ['A', 'B'],
            'x': [1.0, np.nan],
            'y': [np.nan, 4.0]
        })

        # pandas
        pd_result = df.melt(id_vars=['id'], value_vars=['x', 'y'])

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.melt(id_vars=['id'], value_vars=['x', 'y'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 11: Type Preservation
# =============================================================================


class TestTypePreservation:
    """Test type preservation through chain operations."""

    def test_integer_preserved_through_groupby(self):
        """Test integer type preserved through groupby operations."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        # pandas
        pd_result = df.groupby('group')['value'].sum()

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('group')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_float_preserved_through_transform(self):
        """Test float type preserved through transform."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.5, 2.5, 3.5, 4.5]
        })

        # pandas
        pd_df = df.copy()
        pd_df['mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_result = pd_df

        # DataStore
        ds_df = DataStore(df.copy())
        ds_df['mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_string_preserved_through_operations(self):
        """Test string type preserved through operations."""
        df = pd.DataFrame({
            'category': ['apple', 'banana', 'apple', 'cherry'],
            'value': [10, 20, 30, 40]
        })

        # pandas
        pd_result = df.groupby('category')['value'].sum().reset_index()

        # DataStore
        ds_df = DataStore(df)
        ds_result = ds_df.groupby('category')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
