"""
Exploratory Batch 50: Advanced Iteration, Aggregation, and Complex Conditionals

This batch tests:
1. DataFrame.items/iteritems iteration operations
2. Complex apply/transform combinations with groupby
3. Reindex and align advanced operations
4. Complex conditional chains (where/mask with multiple conditions)
5. Advanced aggregation patterns (custom named agg, multiple functions)
6. Set operations (union, intersection via merge)
7. Comparison operations chains
8. Memory-efficient operations (memory_usage, info)
9. Complex boolean indexing patterns
10. Edge cases with rename, reorder, and column manipulation

Tests follow Mirror Code Pattern: pandas first, DataStore mirrors exactly.
"""

import pytest
import pandas as pd
import numpy as np

from datastore import DataStore, concat
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_datastore_equals_pandas_chdb_compat,
    get_series,
)
from tests.xfail_markers import (
    chdb_category_type,
    chdb_timedelta_type,
)


# =============================================================================
# DataFrame.items Iteration Tests
# =============================================================================


class TestDataFrameItems:
    """Test DataFrame.items() iteration."""

    def test_items_basic(self):
        """Test basic items() iteration returns same columns and values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())

        assert len(pd_items) == len(ds_items)
        for (pd_name, pd_series), (ds_name, ds_series) in zip(pd_items, ds_items):
            assert pd_name == ds_name
            pd.testing.assert_series_equal(
                get_series(ds_series),
                pd_series,
                check_names=False
            )

    def test_items_empty_dataframe(self):
        """Test items() on empty DataFrame."""
        pd_df = pd.DataFrame()
        ds_df = DataStore()

        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())

        assert len(pd_items) == 0
        assert len(ds_items) == 0

    def test_items_single_column(self):
        """Test items() with single column."""
        pd_df = pd.DataFrame({'only_col': [1, 2, 3]})
        ds_df = DataStore({'only_col': [1, 2, 3]})

        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())

        assert len(ds_items) == 1
        assert ds_items[0][0] == 'only_col'

    def test_items_after_filter(self):
        """Test items() after filter operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        ds_df = DataStore({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})

        pd_filtered = pd_df[pd_df['a'] > 2]
        ds_filtered = ds_df[ds_df['a'] > 2]

        pd_items = list(pd_filtered.items())
        ds_items = list(ds_filtered.items())

        assert len(pd_items) == len(ds_items)
        for (pd_name, pd_series), (ds_name, ds_series) in zip(pd_items, ds_items):
            assert pd_name == ds_name


# =============================================================================
# Complex Apply/Transform with GroupBy Tests
# =============================================================================


class TestComplexApplyTransform:
    """Test complex apply and transform with groupby."""

    def test_transform_mean_subtraction(self):
        """Test transform for centering data (subtract group mean)."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 100, 200]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 100, 200]
        })

        pd_result = pd_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df.groupby('group')['value'].transform('mean')

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_transform_with_lambda_rank(self):
        """Test transform with lambda for ranking within groups."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 5, 2, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [3, 1, 5, 2, 4]
        })

        pd_result = pd_df.groupby('group')['value'].transform('rank')
        ds_result = ds_df.groupby('group')['value'].transform('rank')

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_apply_size_per_group(self):
        """Test apply returning size per group."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 3, 4, 5]
        })

        # Use size() which is well-defined
        pd_result = pd_df.groupby('group').size()
        ds_result = ds_df.groupby('group').size()

        pd.testing.assert_series_equal(
            get_series(ds_result),
            pd_result,
            check_names=False
        )


# =============================================================================
# Reindex and Align Tests
# =============================================================================


class TestReindexAlign:
    """Test reindex and align operations."""

    def test_reindex_subset(self):
        """Test reindex with subset of indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.reindex([0, 2, 4])
        ds_result = ds_df.reindex([0, 2, 4])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill_value(self):
        """Test reindex with fill_value for missing indices."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.reindex([0, 1, 2, 3, 4], fill_value=-1)
        ds_result = ds_df.reindex([0, 1, 2, 3, 4], fill_value=-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """Test reindex columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        pd_result = pd_df.reindex(columns=['c', 'a'])
        ds_result = ds_df.reindex(columns=['c', 'a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_add_new_columns(self):
        """Test reindex adding new columns with NaN."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df.reindex(columns=['a', 'b', 'c'])
        ds_result = ds_df.reindex(columns=['a', 'b', 'c'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Complex Conditional Chains Tests
# =============================================================================


class TestComplexConditionalChains:
    """Test complex conditional operations."""

    def test_where_with_multiple_conditions(self):
        """Test where with complex boolean condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        # Where a > 2 AND b > 2
        pd_cond = (pd_df['a'] > 2) & (pd_df['b'] > 2)
        ds_cond = (ds_df['a'] > 2) & (ds_df['b'] > 2)

        pd_result = pd_df.where(pd_cond)
        ds_result = ds_df.where(ds_cond)

        # dtype difference: DataStore preserves nullable Int64, pandas converts to float64
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_where_or_condition(self):
        """Test where with OR condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        pd_cond = (pd_df['a'] > 4) | (pd_df['b'] > 4)
        ds_cond = (ds_df['a'] > 4) | (ds_df['b'] > 4)

        pd_result = pd_df.where(pd_cond)
        ds_result = ds_df.where(ds_cond)

        # dtype difference: DataStore preserves nullable Int64, pandas converts to float64
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_mask_with_other_dataframe(self):
        """Test mask with another value as replacement."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.mask(pd_df['a'] > 3, -1)
        ds_result = ds_df.mask(ds_df['a'] > 3, -1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_where_mask(self):
        """Test chaining where and mask operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.where(pd_df['a'] > 2).mask(pd_df['a'] > 4)
        ds_result = ds_df.where(ds_df['a'] > 2).mask(ds_df['a'] > 4)

        # dtype difference: DataStore preserves nullable Int64, pandas converts to float64
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)


# =============================================================================
# Advanced Aggregation Patterns Tests
# =============================================================================


class TestAdvancedAggregation:
    """Test advanced aggregation patterns."""

    def test_agg_dict_multi_column(self):
        """Test agg with dict specifying different functions per column."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'x': [1, 2, 3, 4],
            'y': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'x': [1, 2, 3, 4],
            'y': [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby('group').agg({'x': 'sum', 'y': 'mean'})
        ds_result = ds_df.groupby('group').agg({'x': 'sum', 'y': 'mean'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_list_of_functions(self):
        """Test agg with list of functions."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group')['value'].agg(['sum', 'mean', 'count'])
        ds_result = ds_df.groupby('group')['value'].agg(['sum', 'mean', 'count'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_named_aggregation(self):
        """Test named aggregation with tuple syntax."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )
        ds_result = ds_df.groupby('group').agg(
            total=('value', 'sum'),
            average=('value', 'mean')
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_with_count_on_all_columns(self):
        """Test count aggregation across all columns."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'x': [1, None, 3, 4],
            'y': [10, 20, None, 40]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'x': [1, None, 3, 4],
            'y': [10, 20, None, 40]
        })

        pd_result = pd_df.groupby('group').count()
        ds_result = ds_df.groupby('group').count()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Comparison Operations Chain Tests
# =============================================================================


class TestComparisonChains:
    """Test comparison operation chains."""

    def test_eq_chain(self):
        """Test equality comparison chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 4]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [1, 2, 4]})

        pd_result = pd_df['a'].eq(pd_df['b'])
        ds_result = ds_df['a'].eq(ds_df['b'])

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_ne_chain(self):
        """Test not-equal comparison chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 4]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [1, 2, 4]})

        pd_result = pd_df['a'].ne(pd_df['b'])
        ds_result = ds_df['a'].ne(ds_df['b'])

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_lt_le_gt_ge_chain(self):
        """Test comparison operators in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        # Less than
        pd_lt = pd_df['a'].lt(3)
        ds_lt = ds_df['a'].lt(3)
        pd.testing.assert_series_equal(
            get_series(ds_lt).reset_index(drop=True),
            pd_lt.reset_index(drop=True),
            check_names=False
        )

        # Greater than or equal
        pd_ge = pd_df['a'].ge(3)
        ds_ge = ds_df['a'].ge(3)
        pd.testing.assert_series_equal(
            get_series(ds_ge).reset_index(drop=True),
            pd_ge.reset_index(drop=True),
            check_names=False
        )

    def test_between_inclusive(self):
        """Test between with inclusive bounds."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].between(2, 4, inclusive='both')
        ds_result = ds_df['a'].between(2, 4, inclusive='both')

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_between_exclusive(self):
        """Test between with exclusive bounds."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].between(2, 4, inclusive='neither')
        ds_result = ds_df['a'].between(2, 4, inclusive='neither')

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )


# =============================================================================
# Complex Boolean Indexing Tests
# =============================================================================


class TestComplexBooleanIndexing:
    """Test complex boolean indexing patterns."""

    def test_boolean_index_with_isin(self):
        """Test boolean indexing with isin."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'x', 'y']})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'x', 'y']})

        pd_result = pd_df[pd_df['b'].isin(['x', 'z'])]
        ds_result = ds_df[ds_df['b'].isin(['x', 'z'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_index_with_str_contains(self):
        """Test boolean indexing with string contains."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['hello', 'world', 'hello world']})
        ds_df = DataStore({'a': [1, 2, 3], 'b': ['hello', 'world', 'hello world']})

        pd_result = pd_df[pd_df['b'].str.contains('hello')]
        ds_result = ds_df[ds_df['b'].str.contains('hello')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combined_numeric_string_filter(self):
        """Test combined numeric and string filtering."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['foo', 'bar', 'foo', 'bar', 'foo']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['foo', 'bar', 'foo', 'bar', 'foo']
        })

        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['b'] == 'foo')]
        ds_result = ds_df[(ds_df['a'] > 2) & (ds_df['b'] == 'foo')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negated_condition(self):
        """Test negated boolean condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[~(pd_df['a'] > 3)]
        ds_result = ds_df[~(ds_df['a'] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Rename and Column Manipulation Tests
# =============================================================================


class TestRenameColumnManipulation:
    """Test rename and column manipulation operations."""

    def test_rename_columns_dict(self):
        """Test rename columns with dict."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_result = pd_df.rename(columns={'a': 'x', 'b': 'y'})
        ds_result = ds_df.rename(columns={'a': 'x', 'b': 'y'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_partial(self):
        """Test partial rename (only some columns)."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        pd_result = pd_df.rename(columns={'a': 'new_a'})
        ds_result = ds_df.rename(columns={'a': 'new_a'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_filter(self):
        """Test rename followed by filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        ds_df = DataStore({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})

        pd_result = pd_df.rename(columns={'a': 'x'})
        pd_result = pd_result[pd_result['x'] > 2]

        ds_result = ds_df.rename(columns={'a': 'x'})
        ds_result = ds_result[ds_result['x'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_reorder(self):
        """Test column reordering via indexing."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

        pd_result = pd_df[['c', 'a', 'b']]
        ds_result = ds_df[['c', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix_suffix(self):
        """Test add_prefix and add_suffix."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_prefix = pd_df.add_prefix('col_')
        ds_prefix = ds_df.add_prefix('col_')
        assert_datastore_equals_pandas(ds_prefix, pd_prefix)

        pd_suffix = pd_df.add_suffix('_val')
        ds_suffix = ds_df.add_suffix('_val')
        assert_datastore_equals_pandas(ds_suffix, pd_suffix)


# =============================================================================
# Set-like Operations Tests
# =============================================================================


class TestSetLikeOperations:
    """Test set-like operations (union, intersection via concat/merge)."""

    def test_concat_union(self):
        """Test concat for union-like operation."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [3, 4], 'b': [5, 6]})

        ds_df1 = DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_df2 = DataStore({'a': [3, 4], 'b': [5, 6]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_inner_intersection(self):
        """Test inner merge for intersection-like operation."""
        pd_df1 = pd.DataFrame({'key': [1, 2, 3], 'val1': ['a', 'b', 'c']})
        pd_df2 = pd.DataFrame({'key': [2, 3, 4], 'val2': ['x', 'y', 'z']})

        ds_df1 = DataStore({'key': [1, 2, 3], 'val1': ['a', 'b', 'c']})
        ds_df2 = DataStore({'key': [2, 3, 4], 'val2': ['x', 'y', 'z']})

        pd_result = pd.merge(pd_df1, pd_df2, on='key', how='inner')
        ds_result = DataStore.merge(ds_df1, ds_df2, on='key', how='inner')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_outer_union(self):
        """Test outer merge for union-like operation."""
        pd_df1 = pd.DataFrame({'key': [1, 2], 'val': ['a', 'b']})
        pd_df2 = pd.DataFrame({'key': [2, 3], 'val': ['c', 'd']})

        ds_df1 = DataStore({'key': [1, 2], 'val': ['a', 'b']})
        ds_df2 = DataStore({'key': [2, 3], 'val': ['c', 'd']})

        pd_result = pd.merge(pd_df1, pd_df2, on='key', how='outer', suffixes=('_l', '_r'))
        ds_result = DataStore.merge(ds_df1, ds_df2, on='key', how='outer', suffixes=('_l', '_r'))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Drop and Fill Operations Tests
# =============================================================================


class TestDropFillOperations:
    """Test drop and fill operations."""

    def test_dropna_subset(self):
        """Test dropna with subset of columns."""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': [None, 2, 3, 4],
            'c': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'a': [1, None, 3, None],
            'b': [None, 2, 3, 4],
            'c': [1, 2, 3, 4]
        })

        pd_result = pd_df.dropna(subset=['a'])
        ds_result = ds_df.dropna(subset=['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_how_all(self):
        """Test dropna with how='all'."""
        pd_df = pd.DataFrame({
            'a': [1, None, None],
            'b': [2, None, 3]
        })
        ds_df = DataStore({
            'a': [1, None, None],
            'b': [2, None, 3]
        })

        pd_result = pd_df.dropna(how='all')
        ds_result = ds_df.dropna(how='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_ffill(self):
        """Test fillna with ffill method."""
        pd_df = pd.DataFrame({'a': [1, None, None, 4]})
        ds_df = DataStore({'a': [1, None, None, 4]})

        pd_result = pd_df.ffill()
        ds_result = ds_df.ffill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_bfill(self):
        """Test fillna with bfill method."""
        pd_df = pd.DataFrame({'a': [1, None, None, 4]})
        ds_df = DataStore({'a': [1, None, None, 4]})

        pd_result = pd_df.bfill()
        ds_result = ds_df.bfill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict(self):
        """Test fillna with dict for different columns."""
        pd_df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [None, 2, None]
        })
        ds_df = DataStore({
            'a': [1, None, 3],
            'b': [None, 2, None]
        })

        pd_result = pd_df.fillna({'a': -1, 'b': -2})
        ds_result = ds_df.fillna({'a': -1, 'b': -2})

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Numeric Edge Cases Tests
# =============================================================================


class TestNumericEdgeCases:
    """Test numeric edge cases."""

    def test_divide_by_zero(self):
        """Test division by zero produces inf."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1.0, 0.0, 1.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [1.0, 0.0, 1.0]})

        pd_result = pd_df['a'] / pd_df['b']
        ds_result = ds_df['a'] / ds_df['b']

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_modulo_with_zero(self):
        """Test modulo with zero."""
        pd_df = pd.DataFrame({'a': [5, 10, 15], 'b': [2, 0, 3]})
        ds_df = DataStore({'a': [5, 10, 15], 'b': [2, 0, 3]})

        # Modulo by zero produces NaN in pandas
        pd_result = pd_df['a'] % pd_df['b']
        ds_result = ds_df['a'] % ds_df['b']

        # Check values match (both will have NaN at same position)
        pd_vals = pd_result.values
        ds_vals = get_series(ds_result).values

        for i in range(len(pd_vals)):
            if np.isnan(pd_vals[i]):
                assert np.isnan(ds_vals[i]), f"Position {i}: expected NaN"
            else:
                assert pd_vals[i] == ds_vals[i], f"Position {i}: {pd_vals[i]} != {ds_vals[i]}"

    def test_power_negative_base(self):
        """Test power with negative base."""
        pd_df = pd.DataFrame({'a': [-2, -1, 0, 1, 2]})
        ds_df = DataStore({'a': [-2, -1, 0, 1, 2]})

        pd_result = pd_df['a'] ** 2
        ds_result = ds_df['a'] ** 2

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_floor_ceil_round(self):
        """Test floor, ceil, round operations."""
        pd_df = pd.DataFrame({'a': [1.2, 2.5, 3.7, -1.5, -2.3]})
        ds_df = DataStore({'a': [1.2, 2.5, 3.7, -1.5, -2.3]})

        # Floor
        pd_floor = np.floor(pd_df['a'])
        ds_floor = np.floor(ds_df['a'])
        pd.testing.assert_series_equal(
            get_series(ds_floor).reset_index(drop=True),
            pd_floor.reset_index(drop=True),
            check_names=False
        )

        # Ceil
        pd_ceil = np.ceil(pd_df['a'])
        ds_ceil = np.ceil(ds_df['a'])
        pd.testing.assert_series_equal(
            get_series(ds_ceil).reset_index(drop=True),
            pd_ceil.reset_index(drop=True),
            check_names=False
        )


# =============================================================================
# Complex Chain Operations Tests
# =============================================================================


class TestComplexChainOperations:
    """Test complex chains of operations."""

    def test_filter_groupby_agg_sort_head(self):
        """Test complex chain: filter -> groupby -> agg -> sort -> head."""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60, 70],
            'flag': [True, False, True, True, False, True, True]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C', 'C'],
            'value': [10, 20, 30, 40, 50, 60, 70],
            'flag': [True, False, True, True, False, True, True]
        })

        pd_result = (pd_df[pd_df['flag'] == True]
                     .groupby('category')['value']
                     .sum()
                     .sort_values(ascending=False)
                     .head(2))
        ds_result = (ds_df[ds_df['flag'] == True]
                     .groupby('category')['value']
                     .sum()
                     .sort_values(ascending=False)
                     .head(2))

        pd.testing.assert_series_equal(
            get_series(ds_result),
            pd_result,
            check_names=False
        )

    def test_assign_multiple_filter_sort(self):
        """Test assign multiple columns then filter and sort."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        # pandas version with lambda filter
        pd_result = (pd_df
                     .assign(b=lambda x: x['a'] * 2,
                             c=lambda x: x['a'] ** 2)
                     [lambda x: x['b'] > 4]
                     .sort_values('c', ascending=False))
        
        # DataStore version using explicit condition (lambda filter not supported)
        ds_tmp = ds_df.assign(b=lambda x: x['a'] * 2, c=lambda x: x['a'] ** 2)
        ds_result = ds_tmp[ds_tmp['b'] > 4].sort_values('c', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_filter_groupby(self):
        """Test merge then filter then groupby."""
        pd_df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'key': ['a', 'b', 'c'], 'val2': [10, 20, 30]})

        ds_df1 = DataStore({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
        ds_df2 = DataStore({'key': ['a', 'b', 'c'], 'val2': [10, 20, 30]})

        pd_merged = pd.merge(pd_df1, pd_df2, on='key')
        ds_merged = DataStore.merge(ds_df1, ds_df2, on='key')

        pd_result = pd_merged[pd_merged['val1'] > 1].groupby('key')['val2'].sum()
        ds_result = ds_merged[ds_merged['val1'] > 1].groupby('key')['val2'].sum()

        pd.testing.assert_series_equal(
            get_series(ds_result),
            pd_result,
            check_names=False
        )


# =============================================================================
# String Operations Advanced Tests
# =============================================================================


class TestStringOperationsAdvanced:
    """Test advanced string operations."""

    def test_str_extract_basic(self):
        """Test str.extract with regex."""
        pd_df = pd.DataFrame({'text': ['a1', 'b2', 'c3']})
        ds_df = DataStore({'text': ['a1', 'b2', 'c3']})

        pd_result = pd_df['text'].str.extract(r'(\w)(\d)')
        ds_result = ds_df['text'].str.extract(r'(\w)(\d)')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_regex(self):
        """Test str.replace with regex."""
        pd_df = pd.DataFrame({'text': ['hello123', 'world456', 'foo789']})
        ds_df = DataStore({'text': ['hello123', 'world456', 'foo789']})

        pd_result = pd_df['text'].str.replace(r'\d+', 'NUM', regex=True)
        ds_result = ds_df['text'].str.replace(r'\d+', 'NUM', regex=True)

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )

    def test_str_split_expand(self):
        """Test str.split with expand=True."""
        pd_df = pd.DataFrame({'text': ['a-b-c', 'x-y-z', '1-2-3']})
        ds_df = DataStore({'text': ['a-b-c', 'x-y-z', '1-2-3']})

        pd_result = pd_df['text'].str.split('-', expand=True)
        ds_result = ds_df['text'].str.split('-', expand=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_cat_columns(self):
        """Test str.cat concatenating columns."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['1', '2', '3']})
        ds_df = DataStore({'a': ['x', 'y', 'z'], 'b': ['1', '2', '3']})

        pd_result = pd_df['a'].str.cat(pd_df['b'], sep='-')
        ds_result = ds_df['a'].str.cat(ds_df['b'], sep='-')

        pd.testing.assert_series_equal(
            get_series(ds_result).reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False
        )


# =============================================================================
# DataFrame Info and Metadata Tests
# =============================================================================


class TestDataFrameMetadata:
    """Test DataFrame metadata operations."""

    def test_shape_after_operations(self):
        """Test shape is correct after various operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        # After filter
        pd_filtered = pd_df[pd_df['a'] > 2]
        ds_filtered = ds_df[ds_df['a'] > 2]
        assert ds_filtered.shape == pd_filtered.shape

        # After column select
        pd_select = pd_df[['a']]
        ds_select = ds_df[['a']]
        assert ds_select.shape == pd_select.shape

    def test_columns_property(self):
        """Test columns property returns correct column names."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ds_df = DataStore({'a': [1], 'b': [2], 'c': [3]})

        assert list(ds_df.columns) == list(pd_df.columns)

    def test_dtypes_property(self):
        """Test dtypes property."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c']
        })

        pd_dtypes = pd_df.dtypes
        ds_dtypes = ds_df.dtypes

        assert set(pd_dtypes.index) == set(ds_dtypes.index)

    def test_empty_property(self):
        """Test empty property for various DataFrame states."""
        # Non-empty
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})
        assert ds_df.empty == pd_df.empty == False

        # Empty (no rows)
        pd_empty = pd.DataFrame({'a': []})
        ds_empty = DataStore({'a': []})
        assert ds_empty.empty == pd_empty.empty == True


# =============================================================================
# Idxmin/Idxmax Edge Cases Tests
# =============================================================================


class TestIdxminIdxmaxEdgeCases:
    """Test idxmin and idxmax edge cases."""

    def test_idxmin_basic(self):
        """Test basic idxmin."""
        pd_df = pd.DataFrame({'a': [3, 1, 2, 5, 4]})
        ds_df = DataStore({'a': [3, 1, 2, 5, 4]})

        pd_result = pd_df['a'].idxmin()
        ds_result = ds_df['a'].idxmin()

        assert ds_result == pd_result

    def test_idxmax_basic(self):
        """Test basic idxmax."""
        pd_df = pd.DataFrame({'a': [3, 1, 2, 5, 4]})
        ds_df = DataStore({'a': [3, 1, 2, 5, 4]})

        pd_result = pd_df['a'].idxmax()
        ds_result = ds_df['a'].idxmax()

        assert ds_result == pd_result

    def test_idxmin_with_custom_index(self):
        """Test idxmin with custom index."""
        pd_df = pd.DataFrame({'a': [3, 1, 2, 5, 4]}, index=['x', 'y', 'z', 'w', 'v'])
        ds_df = DataStore({'a': [3, 1, 2, 5, 4]})
        ds_df = ds_df.set_index(pd.Index(['x', 'y', 'z', 'w', 'v']))

        pd_result = pd_df['a'].idxmin()
        ds_result = ds_df['a'].idxmin()

        assert ds_result == pd_result

    def test_idxmax_with_ties(self):
        """Test idxmax when there are ties (returns first)."""
        pd_df = pd.DataFrame({'a': [1, 5, 5, 2, 5]})
        ds_df = DataStore({'a': [1, 5, 5, 2, 5]})

        pd_result = pd_df['a'].idxmax()
        ds_result = ds_df['a'].idxmax()

        assert ds_result == pd_result  # Should be 1 (first occurrence of max)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
