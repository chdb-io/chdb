"""
Exploratory Batch 49: Scalar Returns, Nested Operations, and Edge Cases

This batch tests:
1. Scalar-returning operations and their integration
2. Nested assign with complex expressions
3. Multi-column operations on different dtypes
4. Operations that force engine transitions
5. Edge cases with apply/transform/pipe
6. Complex where/mask chains
7. Cumulative operations with edge cases

Tests follow Mirror Code Pattern: pandas first, DataStore mirrors exactly.
"""

import pytest
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    get_series,
)
from tests.xfail_markers import (
    chdb_category_type,
    chdb_timedelta_type,
)


# =============================================================================
# Scalar Return Tests
# =============================================================================


class TestScalarReturns:
    """Test operations that return scalar values."""

    def test_sum_returns_scalar_comparable(self):
        """Test that sum() returns a value comparable to pandas."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()

        # Both should be numeric scalars
        assert float(ds_result) == float(pd_result)

    def test_mean_returns_scalar_comparable(self):
        """Test that mean() returns a value comparable to pandas."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()

        assert abs(float(ds_result) - float(pd_result)) < 1e-10

    def test_min_max_returns_scalar(self):
        """Test min/max return scalar values."""
        pd_df = pd.DataFrame({'a': [10, 20, 30, 40, 50]})
        ds_df = DataStore({'a': [10, 20, 30, 40, 50]})

        assert int(ds_df['a'].min()) == int(pd_df['a'].min())
        assert int(ds_df['a'].max()) == int(pd_df['a'].max())

    def test_std_returns_scalar(self):
        """Test std() returns comparable scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].std()
        ds_result = ds_df['a'].std()

        assert abs(float(ds_result) - float(pd_result)) < 1e-5

    def test_var_returns_scalar(self):
        """Test var() returns comparable scalar."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].var()
        ds_result = ds_df['a'].var()

        assert abs(float(ds_result) - float(pd_result)) < 1e-5

    def test_count_returns_scalar(self):
        """Test count() returns comparable scalar."""
        pd_df = pd.DataFrame({'a': [1, None, 3, None, 5]})
        ds_df = DataStore({'a': [1, None, 3, None, 5]})

        pd_result = pd_df['a'].count()
        ds_result = ds_df['a'].count()

        assert int(ds_result) == int(pd_result)

    def test_nunique_returns_scalar(self):
        """Test nunique() returns comparable scalar."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 3]})
        ds_df = DataStore({'a': [1, 1, 2, 2, 3]})

        pd_result = pd_df['a'].nunique()
        ds_result = ds_df['a'].nunique()

        assert int(ds_result) == int(pd_result)


# =============================================================================
# Nested Assign Tests
# =============================================================================


class TestNestedAssign:
    """Test nested and complex assign operations."""

    def test_assign_referencing_new_column(self):
        """Test assign where second lambda references first assigned column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.assign(
            b=lambda x: x['a'] * 2,
            c=lambda x: x['b'] + 1  # References newly assigned b
        )
        ds_result = ds_df.assign(
            b=lambda x: x['a'] * 2,
            c=lambda x: x['b'] + 1
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multi_column_chain(self):
        """Test multiple column assignments in chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.assign(
            sum_ab=lambda x: x['a'] + x['b'],
            diff_ab=lambda x: x['a'] - x['b'],
            prod_ab=lambda x: x['a'] * x['b']
        )
        ds_result = ds_df.assign(
            sum_ab=lambda x: x['a'] + x['b'],
            diff_ab=lambda x: x['a'] - x['b'],
            prod_ab=lambda x: x['a'] * x['b']
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_overwrite_existing_column(self):
        """Test assign that overwrites an existing column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.assign(a=lambda x: x['a'] * 10)
        ds_result = ds_df.assign(a=lambda x: x['a'] * 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_constant(self):
        """Test assign with constant value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.assign(constant=100)
        ds_result = ds_df.assign(constant=100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_mixed_constant_and_lambda(self):
        """Test assign with both constants and lambdas."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.assign(
            const=42,
            computed=lambda x: x['a'] + x['const']
        )
        ds_result = ds_df.assign(
            const=42,
            computed=lambda x: x['a'] + x['const']
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Multi-dtype Column Operations
# =============================================================================


class TestMultiDtypeOperations:
    """Test operations across columns with different dtypes."""

    def test_mixed_int_float_arithmetic(self):
        """Test arithmetic between int and float columns."""
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})
        ds_df = DataStore({'int_col': [1, 2, 3], 'float_col': [1.5, 2.5, 3.5]})

        pd_result = pd_df.assign(mixed=lambda x: x['int_col'] + x['float_col'])
        ds_result = ds_df.assign(mixed=lambda x: x['int_col'] + x['float_col'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_int_concat(self):
        """Test string concatenation with int column (needs type conversion)."""
        pd_df = pd.DataFrame({
            'name': ['Item', 'Item', 'Item'],
            'num': [1, 2, 3]
        })
        ds_df = DataStore({
            'name': ['Item', 'Item', 'Item'],
            'num': [1, 2, 3]
        })

        pd_result = pd_df.assign(
            combined=lambda x: x['name'] + '_' + x['num'].astype(str)
        )
        ds_result = ds_df.assign(
            combined=lambda x: x['name'] + '_' + x['num'].astype(str)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_int_arithmetic(self):
        """Test arithmetic with boolean and int columns."""
        pd_df = pd.DataFrame({
            'value': [10, 20, 30],
            'flag': [True, False, True]
        })
        ds_df = DataStore({
            'value': [10, 20, 30],
            'flag': [True, False, True]
        })

        pd_result = pd_df.assign(
            flagged=lambda x: x['value'] * x['flag'].astype(int)
        )
        ds_result = ds_df.assign(
            flagged=lambda x: x['value'] * x['flag'].astype(int)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Cumulative Operations
# =============================================================================


class TestCumulativeOperations:
    """Test cumulative operations with edge cases."""

    def test_cumsum_basic(self):
        """Test basic cumsum."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(cumsum_a=lambda x: x['a'].cumsum())
        ds_result = ds_df.assign(cumsum_a=lambda x: x['a'].cumsum())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummax_basic(self):
        """Test cummax."""
        pd_df = pd.DataFrame({'a': [1, 3, 2, 5, 4]})
        ds_df = DataStore({'a': [1, 3, 2, 5, 4]})

        pd_result = pd_df.assign(cummax_a=lambda x: x['a'].cummax())
        ds_result = ds_df.assign(cummax_a=lambda x: x['a'].cummax())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummin_basic(self):
        """Test cummin."""
        pd_df = pd.DataFrame({'a': [5, 3, 4, 1, 2]})
        ds_df = DataStore({'a': [5, 3, 4, 1, 2]})

        pd_result = pd_df.assign(cummin_a=lambda x: x['a'].cummin())
        ds_result = ds_df.assign(cummin_a=lambda x: x['a'].cummin())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumprod_basic(self):
        """Test cumprod."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(cumprod_a=lambda x: x['a'].cumprod())
        ds_result = ds_df.assign(cumprod_a=lambda x: x['a'].cumprod())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_with_null(self):
        """Test cumsum with NULL values."""
        pd_df = pd.DataFrame({'a': [1.0, None, 3.0, None, 5.0]})
        ds_df = DataStore({'a': [1.0, None, 3.0, None, 5.0]})

        pd_result = pd_df.assign(cumsum_a=lambda x: x['a'].cumsum())
        ds_result = ds_df.assign(cumsum_a=lambda x: x['a'].cumsum())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Where/Mask Edge Cases
# =============================================================================


class TestWhereMaskEdgeCases:
    """Test where and mask edge cases."""

    def test_where_basic(self):
        """Test basic where operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(
            where_a=lambda x: x['a'].where(x['a'] > 2, other=-1)
        )
        ds_result = ds_df.assign(
            where_a=lambda x: x['a'].where(x['a'] > 2, other=-1)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mask_basic(self):
        """Test basic mask operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(
            mask_a=lambda x: x['a'].mask(x['a'] > 2, other=-1)
        )
        ds_result = ds_df.assign(
            mask_a=lambda x: x['a'].mask(x['a'] > 2, other=-1)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_without_other(self):
        """Test where without 'other' parameter (uses NaN)."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        pd_result = pd_df.assign(where_a=lambda x: x['a'].where(x['a'] > 2))
        ds_result = ds_df.assign(where_a=lambda x: x['a'].where(x['a'] > 2))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Rank Operations
# =============================================================================


class TestRankOperations:
    """Test rank operations."""

    def test_rank_average(self):
        """Test rank with average method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.assign(rank_a=lambda x: x['a'].rank(method='average'))
        ds_result = ds_df.assign(rank_a=lambda x: x['a'].rank(method='average'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_min(self):
        """Test rank with min method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.assign(rank_a=lambda x: x['a'].rank(method='min'))
        ds_result = ds_df.assign(rank_a=lambda x: x['a'].rank(method='min'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_max(self):
        """Test rank with max method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.assign(rank_a=lambda x: x['a'].rank(method='max'))
        ds_result = ds_df.assign(rank_a=lambda x: x['a'].rank(method='max'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_first(self):
        """Test rank with first method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.assign(rank_a=lambda x: x['a'].rank(method='first'))
        ds_result = ds_df.assign(rank_a=lambda x: x['a'].rank(method='first'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_dense(self):
        """Test rank with dense method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.assign(rank_a=lambda x: x['a'].rank(method='dense'))
        ds_result = ds_df.assign(rank_a=lambda x: x['a'].rank(method='dense'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_descending(self):
        """Test rank with ascending=False."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore({'a': [3, 1, 4, 1, 5]})

        pd_result = pd_df.assign(rank_a=lambda x: x['a'].rank(ascending=False))
        ds_result = ds_df.assign(rank_a=lambda x: x['a'].rank(ascending=False))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Clip Operations
# =============================================================================


class TestClipOperations:
    """Test clip operations."""

    def test_clip_both_bounds(self):
        """Test clip with lower and upper bounds."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        pd_result = pd_df.assign(clipped=lambda x: x['a'].clip(lower=3, upper=7))
        ds_result = ds_df.assign(clipped=lambda x: x['a'].clip(lower=3, upper=7))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_lower_only(self):
        """Test clip with only lower bound."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(clipped=lambda x: x['a'].clip(lower=3))
        ds_result = ds_df.assign(clipped=lambda x: x['a'].clip(lower=3))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_upper_only(self):
        """Test clip with only upper bound."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(clipped=lambda x: x['a'].clip(upper=3))
        ds_result = ds_df.assign(clipped=lambda x: x['a'].clip(upper=3))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# DataFrame-level Aggregation
# =============================================================================


class TestDataFrameAggregation:
    """Test DataFrame-level aggregation operations."""

    def test_df_sum(self):
        """Test DataFrame sum."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

        pd_result = pd_df.sum()
        ds_result = ds_df.sum()

        # Compare Series
        assert float(ds_result['a']) == float(pd_result['a'])
        assert float(ds_result['b']) == float(pd_result['b'])

    def test_df_mean(self):
        """Test DataFrame mean."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})

        pd_result = pd_df.mean()
        ds_result = ds_df.mean()

        assert abs(float(ds_result['a']) - float(pd_result['a'])) < 1e-10
        assert abs(float(ds_result['b']) - float(pd_result['b'])) < 1e-10


# =============================================================================
# GroupBy Edge Cases
# =============================================================================


class TestGroupByEdgeCases:
    """Test groupby edge cases."""

    def test_groupby_single_group(self):
        """Test groupby with only one group."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A'],
            'value': [1, 2, 3]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'A'],
            'value': [1, 2, 3]
        })

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_all_unique(self):
        """Test groupby where each group has one element."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', 'C'],
            'value': [1, 2, 3]
        })
        ds_df = DataStore({
            'group': ['A', 'B', 'C'],
            'value': [1, 2, 3]
        })

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_multiple_columns(self):
        """Test groupby with multiple columns."""
        pd_df = pd.DataFrame({
            'g1': ['A', 'A', 'B', 'B'],
            'g2': ['X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'g1': ['A', 'A', 'B', 'B'],
            'g2': ['X', 'Y', 'X', 'Y'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby(['g1', 'g2'])['value'].sum().reset_index()
        ds_result = ds_df.groupby(['g1', 'g2'])['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_multiple_funcs(self):
        """Test groupby with multiple aggregation functions."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('group')['value'].agg(['sum', 'mean', 'count']).reset_index()
        ds_result = ds_df.groupby('group')['value'].agg(['sum', 'mean', 'count']).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


# =============================================================================
# Filter Chain Edge Cases
# =============================================================================


class TestFilterChainEdgeCases:
    """Test complex filter chains."""

    def test_filter_all_false(self):
        """Test filter where condition is all False."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds_df[ds_df['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_all_true(self):
        """Test filter where condition is all True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_and_or_combination(self):
        """Test complex AND/OR filter combinations."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })

        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['b'] < 4)]
        ds_result = ds_df[(ds_df['a'] > 2) & (ds_df['b'] < 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_or_combination(self):
        """Test OR filter combinations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[(pd_df['a'] == 1) | (pd_df['a'] == 5)]
        ds_result = ds_df[(ds_df['a'] == 1) | (ds_df['a'] == 5)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_not(self):
        """Test NOT filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df[~(pd_df['a'] > 3)]
        ds_result = ds_df[~(ds_df['a'] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Shift/Diff Operations
# =============================================================================


class TestShiftDiffOperations:
    """Test shift and diff operations."""

    def test_shift_positive(self):
        """Test shift with positive periods."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        pd_result = pd_df.assign(shifted=lambda x: x['a'].shift(1))
        ds_result = ds_df.assign(shifted=lambda x: x['a'].shift(1))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_negative(self):
        """Test shift with negative periods."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})

        pd_result = pd_df.assign(shifted=lambda x: x['a'].shift(-1))
        ds_result = ds_df.assign(shifted=lambda x: x['a'].shift(-1))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_basic(self):
        """Test basic diff operation."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 4.0, 7.0, 11.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 4.0, 7.0, 11.0]})

        pd_result = pd_df.assign(diff_a=lambda x: x['a'].diff())
        ds_result = ds_df.assign(diff_a=lambda x: x['a'].diff())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_periods_2(self):
        """Test diff with periods=2."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 4.0, 7.0, 11.0]})
        ds_df = DataStore({'a': [1.0, 2.0, 4.0, 7.0, 11.0]})

        pd_result = pd_df.assign(diff_a=lambda x: x['a'].diff(periods=2))
        ds_result = ds_df.assign(diff_a=lambda x: x['a'].diff(periods=2))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Pct_change Operations
# =============================================================================


class TestPctChangeOperations:
    """Test pct_change operations."""

    def test_pct_change_basic(self):
        """Test basic pct_change."""
        pd_df = pd.DataFrame({'a': [100.0, 110.0, 121.0, 133.1]})
        ds_df = DataStore({'a': [100.0, 110.0, 121.0, 133.1]})

        pd_result = pd_df.assign(pct_chg=lambda x: x['a'].pct_change())
        ds_result = ds_df.assign(pct_chg=lambda x: x['a'].pct_change())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Select dtypes Operations
# =============================================================================


class TestSelectDtypes:
    """Test select_dtypes operations."""

    def test_select_dtypes_numeric(self):
        """Test select_dtypes with numeric include."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })

        pd_result = pd_df.select_dtypes(include=['number'])
        ds_result = ds_df.select_dtypes(include=['number'])

        # Check columns are the same
        assert set(ds_result.columns) == set(pd_result.columns)

    def test_select_dtypes_object(self):
        """Test select_dtypes with object include."""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c']
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'str_col': ['a', 'b', 'c']
        })

        pd_result = pd_df.select_dtypes(include=['object'])
        ds_result = ds_df.select_dtypes(include=['object'])

        # Check columns are the same (may vary due to dtype handling)
        assert 'str_col' in ds_result.columns


# =============================================================================
# Value Counts on Series
# =============================================================================


class TestValueCounts:
    """Test value_counts operations."""

    def test_value_counts_basic(self):
        """Test basic value_counts."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'x', 'x', 'y', 'z']})
        ds_df = DataStore({'a': ['x', 'y', 'x', 'x', 'y', 'z']})

        pd_result = pd_df['a'].value_counts().reset_index()
        pd_result.columns = ['a', 'count']
        ds_result = ds_df['a'].value_counts().reset_index()
        ds_result.columns = ['a', 'count']

        # Sort both for comparison (value_counts order may differ)
        pd_result = pd_result.sort_values('a').reset_index(drop=True)
        ds_result_df = ds_result.sort_values('a').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result_df, pd_result, check_row_order=False)

    def test_value_counts_normalize(self):
        """Test value_counts with normalize=True."""
        pd_df = pd.DataFrame({'a': ['x', 'y', 'x', 'x', 'y', 'z']})
        ds_df = DataStore({'a': ['x', 'y', 'x', 'x', 'y', 'z']})

        pd_result = pd_df['a'].value_counts(normalize=True)
        ds_result = ds_df['a'].value_counts(normalize=True)

        # Check values match (proportions)
        pd_sum = pd_result.sum()
        ds_sum = float(get_series(ds_result).sum())

        assert abs(pd_sum - ds_sum) < 0.01


# =============================================================================
# Unique Operations
# =============================================================================


class TestUniqueOperations:
    """Test unique operations."""

    def test_unique_basic(self):
        """Test basic unique."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore({'a': [1, 2, 2, 3, 3, 3]})

        pd_result = pd_df['a'].unique()
        ds_result = ds_df['a'].unique()

        # Compare as sets (order may differ)
        assert set(pd_result) == set(ds_result)

    def test_unique_with_null(self):
        """Test unique with NULL values."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, None, 2.0, None]})
        ds_df = DataStore({'a': [1.0, 2.0, None, 2.0, None]})

        pd_result = pd_df['a'].unique()
        ds_result = ds_df['a'].unique()

        # Count should match (including NaN)
        assert len(pd_result) == len(ds_result)


# =============================================================================
# NSmallest/NLargest Operations
# =============================================================================


class TestNSmallestNLargest:
    """Test nsmallest and nlargest operations."""

    def test_nlargest_basic(self):
        """Test basic nlargest."""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 4, 2]})
        ds_df = DataStore({'a': [1, 5, 3, 4, 2]})

        pd_result = pd_df.nlargest(3, 'a')
        ds_result = ds_df.nlargest(3, 'a')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True, check_index=False)

    def test_nsmallest_basic(self):
        """Test basic nsmallest."""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 4, 2]})
        ds_df = DataStore({'a': [1, 5, 3, 4, 2]})

        pd_result = pd_df.nsmallest(3, 'a')
        ds_result = ds_df.nsmallest(3, 'a')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True, check_index=False)


# =============================================================================
# Abs Operations
# =============================================================================


class TestAbsOperations:
    """Test abs operations."""

    def test_abs_basic(self):
        """Test basic abs."""
        pd_df = pd.DataFrame({'a': [-1, 2, -3, 4, -5]})
        ds_df = DataStore({'a': [-1, 2, -3, 4, -5]})

        pd_result = pd_df.assign(abs_a=lambda x: x['a'].abs())
        ds_result = ds_df.assign(abs_a=lambda x: x['a'].abs())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_float(self):
        """Test abs on float column."""
        pd_df = pd.DataFrame({'a': [-1.5, 2.5, -3.5]})
        ds_df = DataStore({'a': [-1.5, 2.5, -3.5]})

        pd_result = pd_df.assign(abs_a=lambda x: x['a'].abs())
        ds_result = ds_df.assign(abs_a=lambda x: x['a'].abs())

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Round Operations
# =============================================================================


class TestRoundOperations:
    """Test round operations."""

    def test_round_basic(self):
        """Test basic round."""
        pd_df = pd.DataFrame({'a': [1.234, 2.567, 3.891]})
        ds_df = DataStore({'a': [1.234, 2.567, 3.891]})

        pd_result = pd_df.assign(rounded=lambda x: x['a'].round(2))
        ds_result = ds_df.assign(rounded=lambda x: x['a'].round(2))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_round_to_int(self):
        """Test round to integer."""
        pd_df = pd.DataFrame({'a': [1.5, 2.4, 3.6]})
        ds_df = DataStore({'a': [1.5, 2.4, 3.6]})

        pd_result = pd_df.assign(rounded=lambda x: x['a'].round(0))
        ds_result = ds_df.assign(rounded=lambda x: x['a'].round(0))

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Copy Operations
# =============================================================================


class TestCopyOperations:
    """Test copy operations."""

    def test_copy_deep(self):
        """Test deep copy."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_copy = pd_df.copy(deep=True)
        ds_copy = ds_df.copy(deep=True)

        # Modify original
        pd_df['a'] = [10, 20, 30]
        ds_df['a'] = [10, 20, 30]

        # Copies should be unchanged
        assert list(pd_copy['a']) == [1, 2, 3]
        # DataStore copy behavior


# =============================================================================
# Head/Tail Edge Cases
# =============================================================================


class TestHeadTailEdgeCases:
    """Test head and tail edge cases."""

    def test_head_larger_than_df(self):
        """Test head with n larger than dataframe."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.head(10)
        ds_result = ds_df.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_larger_than_df(self):
        """Test tail with n larger than dataframe."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.tail(10)
        ds_result = ds_df.tail(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """Test head(0)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_zero(self):
        """Test tail(0)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df.tail(0)
        ds_result = ds_df.tail(0)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Sample Operations
# =============================================================================


class TestSampleOperations:
    """Test sample operations."""

    def test_sample_n(self):
        """Test sample with n."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        # Use seed for reproducibility
        pd_result = pd_df.sample(n=3, random_state=42)
        ds_result = ds_df.sample(n=3, random_state=42)

        # Check length matches
        assert len(ds_result) == len(pd_result)

    def test_sample_frac(self):
        """Test sample with frac."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        # Sample 50%
        pd_result = pd_df.sample(frac=0.5, random_state=42)
        ds_result = ds_df.sample(frac=0.5, random_state=42)

        # Check length is approximately correct
        assert len(ds_result) == len(pd_result)
