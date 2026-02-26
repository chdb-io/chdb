"""
Exploratory Batch 73: DataFrame.update(), Numeric Edge Cases, and Complex Boolean Chains

Focus areas:
1. DataFrame.update() with different shapes, dtypes, and overwrite behavior
2. Numeric edge cases: inf, -inf, very large/small numbers, precision
3. Complex boolean indexing chains with multiple conditions
4. Aggregations on computed columns in chains
5. assign() with callable that references multiple columns
6. Clip operations with various bounds including inf

Discovery method: Architecture-based exploration targeting:
- datastore/core.py update() implementation
- Numeric handling in SQL vs pandas
- Boolean condition chaining in lazy evaluation
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_datastore_equals_pandas_chdb_compat,
    assert_series_equal,
    get_dataframe,
    get_series
)


class TestDataFrameUpdate:
    """Test DataFrame.update() method edge cases"""

    def test_update_basic(self):
        """Basic update with matching indices"""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30]})
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [10, 20, 30]})
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_partial_index_overlap(self):
        """Update with partial index overlap"""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'a': [100, 200]}, index=[1, 2])
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2_dict = {'a': [np.nan, 100, 200]}  # Position 0 has no update
        ds_df2 = DataStore(ds_df2_dict)
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_with_nan_values(self):
        """Update should not overwrite with NaN by default"""
        pd_df1 = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        pd_df2 = pd.DataFrame({'a': [10.0, np.nan, 30.0]})
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        ds_df2 = DataStore({'a': [10.0, np.nan, 30.0]})
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_overwrite_true(self):
        """Update with overwrite=True should overwrite existing values"""
        pd_df1 = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        pd_df2 = pd.DataFrame({'a': [np.nan, 20.0, np.nan]})
        pd_df1.update(pd_df2, overwrite=True)

        ds_df1 = DataStore({'a': [1.0, 2.0, 3.0]})
        ds_df2 = DataStore({'a': [np.nan, 20.0, np.nan]})
        ds_df1.update(ds_df2, overwrite=True)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_filter_func(self):
        """Update with filter_func to conditionally update"""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        pd_df1.update(pd_df2, filter_func=lambda x: x > 2)

        ds_df1 = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df1.update(ds_df2, filter_func=lambda x: x > 2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)

    def test_update_with_new_column(self):
        """Update from other with a column not in self (should be ignored)"""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'c': [100, 200, 300]})
        pd_df1.update(pd_df2)

        ds_df1 = DataStore({'a': [1, 2, 3]})
        ds_df2 = DataStore({'a': [10, 20, 30], 'c': [100, 200, 300]})
        ds_df1.update(ds_df2)

        assert_datastore_equals_pandas(ds_df1, pd_df1)


class TestNumericEdgeCases:
    """Test numeric edge cases: inf, -inf, very large numbers"""

    def test_inf_in_dataframe(self):
        """DataStore should handle infinity values"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 4.0]})
        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, 4.0]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_inf_in_arithmetic(self):
        """Arithmetic with infinity"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 4.0]})
        pd_result = pd_df['a'] + 10

        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, 4.0]})
        ds_result = ds_df['a'] + 10

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_inf_in_filter(self):
        """Filter with infinity comparison"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 4.0, 100.0]})
        pd_result = pd_df[pd_df['a'] > 10].reset_index(drop=True)

        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, 4.0, 100.0]})
        ds_result = ds_df[ds_df['a'] > 10].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_inf_mean_calculation(self):
        """Mean with infinity values"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, np.inf, 4.0]})
        pd_result = pd_df['a'].mean()

        ds_df = DataStore({'a': [1.0, 2.0, np.inf, 4.0]})
        ds_result = ds_df['a'].mean()

        assert pd_result == ds_result

    def test_large_numbers_sum(self):
        """Sum of very large numbers"""
        large = 1e18
        pd_df = pd.DataFrame({'a': [large, large, large]})
        pd_result = pd_df['a'].sum()

        ds_df = DataStore({'a': [large, large, large]})
        ds_result = ds_df['a'].sum()

        assert abs(pd_result - ds_result) / pd_result < 1e-10

    def test_small_numbers_precision(self):
        """Very small numbers precision"""
        small = 1e-15
        pd_df = pd.DataFrame({'a': [small, small * 2, small * 3]})
        pd_result = pd_df['a'].sum()

        ds_df = DataStore({'a': [small, small * 2, small * 3]})
        ds_result = ds_df['a'].sum()

        assert abs(pd_result - ds_result) / pd_result < 1e-10

    def test_isinf_method(self):
        """Test isinf detection"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, np.nan, 4.0]})
        pd_result = np.isinf(pd_df['a'])

        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, np.nan, 4.0]})
        ds_result = np.isinf(get_series(ds_df['a']))

        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)


class TestClipOperations:
    """Test clip with various bounds including inf"""

    def test_clip_basic(self):
        """Basic clip operation"""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        pd_result = pd_df['a'].clip(lower=5, upper=15)

        ds_df = DataStore({'a': [1, 5, 10, 15, 20]})
        ds_result = ds_df['a'].clip(lower=5, upper=15)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_lower_only(self):
        """Clip with only lower bound"""
        pd_df = pd.DataFrame({'a': [-10, -5, 0, 5, 10]})
        pd_result = pd_df['a'].clip(lower=0)

        ds_df = DataStore({'a': [-10, -5, 0, 5, 10]})
        ds_result = ds_df['a'].clip(lower=0)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_upper_only(self):
        """Clip with only upper bound"""
        pd_df = pd.DataFrame({'a': [-10, 0, 10, 20, 30]})
        pd_result = pd_df['a'].clip(upper=10)

        ds_df = DataStore({'a': [-10, 0, 10, 20, 30]})
        ds_result = ds_df['a'].clip(upper=10)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_with_inf(self):
        """Clip with infinity values"""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 10.0]})
        pd_result = pd_df['a'].clip(lower=-100, upper=100)

        ds_df = DataStore({'a': [1.0, np.inf, -np.inf, 10.0]})
        ds_result = ds_df['a'].clip(lower=-100, upper=100)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_with_nan(self):
        """Clip with NaN values (NaN should pass through)"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 10.0, np.nan, 20.0]})
        pd_result = pd_df['a'].clip(lower=5, upper=15)

        ds_df = DataStore({'a': [1.0, np.nan, 10.0, np.nan, 20.0]})
        ds_result = ds_df['a'].clip(lower=5, upper=15)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_dataframe(self):
        """Clip on DataFrame"""
        pd_df = pd.DataFrame({'a': [1, 5, 10], 'b': [20, 25, 30]})
        pd_result = pd_df.clip(lower=5, upper=25)

        ds_df = DataStore({'a': [1, 5, 10], 'b': [20, 25, 30]})
        ds_result = ds_df.clip(lower=5, upper=25)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexBooleanChains:
    """Test complex boolean indexing chains"""

    def test_multiple_and_conditions(self):
        """Multiple AND conditions"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'x', 'y']
        })
        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] < 40) & (pd_df['c'] == 'y')]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'x', 'y']
        })
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] < 40) & (ds_df['c'] == 'y')]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_or_conditions(self):
        """Multiple OR conditions"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[(pd_df['a'] == 1) | (pd_df['a'] == 3) | (pd_df['a'] == 5)]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df[(ds_df['a'] == 1) | (ds_df['a'] == 3) | (ds_df['a'] == 5)]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nested_and_or_conditions(self):
        """Nested AND/OR conditions"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [10, 20, 30, 40, 50, 60]
        })
        # (a > 2 AND b < 50) OR (a == 1)
        pd_result = pd_df[((pd_df['a'] > 2) & (pd_df['b'] < 50)) | (pd_df['a'] == 1)]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [10, 20, 30, 40, 50, 60]
        })
        ds_result = ds_df[((ds_df['a'] > 2) & (ds_df['b'] < 50)) | (ds_df['a'] == 1)]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_in_boolean_chain(self):
        """Negation in boolean chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[~((pd_df['a'] > 2) & (pd_df['b'] < 40))]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df[~((ds_df['a'] > 2) & (ds_df['b'] < 40))]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_in_chain(self):
        """Between used in boolean chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_result = pd_df[pd_df['a'].between(2, 4) & (pd_df['b'] != 30)]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_result = ds_df[ds_df['a'].between(2, 4) & (ds_df['b'] != 30)]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_in_boolean_chain(self):
        """isin used in boolean chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'x', 'y']
        })
        pd_result = pd_df[(pd_df['a'] > 1) & pd_df['b'].isin(['x', 'z'])]
        pd_result = pd_result.reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'x', 'y']
        })
        ds_result = ds_df[(ds_df['a'] > 1) & ds_df['b'].isin(['x', 'z'])]
        ds_result = ds_result.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggregationOnComputedColumns:
    """Test aggregation on columns created in the same chain"""

    def test_assign_then_sum(self):
        """Assign new column then sum it"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_df = pd_df.copy()
        pd_df['b'] = pd_df['a'] * 2
        pd_result = pd_df['b'].sum()

        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_df['b'] = ds_df['a'] * 2
        ds_result = ds_df['b'].sum()

        assert pd_result == ds_result

    def test_assign_then_mean(self):
        """Assign new column then mean it"""
        pd_df = pd.DataFrame({'a': [10.0, 20.0, 30.0, 40.0]})
        pd_df = pd_df.copy()
        pd_df['b'] = pd_df['a'] / 10
        pd_result = pd_df['b'].mean()

        ds_df = DataStore({'a': [10.0, 20.0, 30.0, 40.0]})
        ds_df['b'] = ds_df['a'] / 10
        ds_result = ds_df['b'].mean()

        assert pd_result == ds_result

    def test_assign_then_groupby_agg(self):
        """Assign new column then groupby aggregate"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        pd_df = pd_df.copy()
        pd_df['doubled'] = pd_df['value'] * 2
        pd_result = pd_df.groupby('group')['doubled'].sum().reset_index()

        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        ds_df['doubled'] = ds_df['value'] * 2
        ds_result = ds_df.groupby('group')['doubled'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_assign_then_agg(self):
        """Filter, then assign, then aggregate"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [10, 20, 30, 40, 50, 60]
        })
        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        pd_filtered['c'] = pd_filtered['a'] + pd_filtered['b']
        pd_result = pd_filtered['c'].sum()

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5, 6],
            'b': [10, 20, 30, 40, 50, 60]
        })
        ds_filtered = ds_df[ds_df['a'] > 2]
        ds_filtered['c'] = ds_filtered['a'] + ds_filtered['b']
        ds_result = ds_filtered['c'].sum()

        assert pd_result == ds_result

    def test_chained_assigns_then_agg(self):
        """Multiple chained assigns then aggregate"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_df = pd_df.copy()
        pd_df['b'] = pd_df['a'] * 2
        pd_df['c'] = pd_df['b'] + 10
        pd_df['d'] = pd_df['c'] ** 2
        pd_result = pd_df['d'].max()

        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_df['b'] = ds_df['a'] * 2
        ds_df['c'] = ds_df['b'] + 10
        ds_df['d'] = ds_df['c'] ** 2
        ds_result = ds_df['d'].max()

        assert pd_result == ds_result


class TestAssignWithCallable:
    """Test assign() method with callable functions"""

    def test_assign_lambda_single_column(self):
        """assign with lambda referencing single column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df.assign(b=lambda x: x['a'] * 2)

        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds_df.assign(b=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_lambda_multiple_columns(self):
        """assign with lambda referencing multiple columns"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        pd_result = pd_df.assign(c=lambda x: x['a'] + x['b'])

        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_result = ds_df.assign(c=lambda x: x['a'] + x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_chained_lambdas(self):
        """Multiple chained assigns with lambdas"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.assign(
            b=lambda x: x['a'] * 2,
            c=lambda x: x['b'] + 10  # References 'b' created in same assign
        )

        ds_df = DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.assign(
            b=lambda x: x['a'] * 2,
            c=lambda x: x['b'] + 10
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_mixed_values_and_callable(self):
        """assign with both direct values and callables"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_result = pd_df.assign(
            b=[10, 20, 30],
            c=lambda x: x['a'] + x['b']
        )

        ds_df = DataStore({'a': [1, 2, 3]})
        ds_result = ds_df.assign(
            b=[10, 20, 30],
            c=lambda x: x['a'] + x['b']
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCaseCombinations:
    """Test edge case combinations"""

    def test_empty_dataframe_operations(self):
        """Operations on empty DataFrame"""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        pd_result = pd_df[pd_df['a'] > 0]

        ds_df = DataStore({'a': [], 'b': []})
        ds_result = ds_df[ds_df['a'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_operations(self):
        """Operations on single row DataFrame"""
        pd_df = pd.DataFrame({'a': [1], 'b': [10]})
        pd_result = pd_df[pd_df['a'] > 0]['b'].sum()

        ds_df = DataStore({'a': [1], 'b': [10]})
        ds_result = ds_df[ds_df['a'] > 0]['b'].sum()

        assert pd_result == ds_result

    def test_all_null_column(self):
        """Operations on all-null column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [np.nan, np.nan, np.nan]})
        pd_result = pd_df['b'].sum()

        ds_df = DataStore({'a': [1, 2, 3], 'b': [np.nan, np.nan, np.nan]})
        ds_result = ds_df['b'].sum()

        assert pd_result == ds_result

    def test_filter_to_empty_then_agg(self):
        """Filter to empty then aggregate"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        pd_filtered = pd_df[pd_df['a'] > 100]  # Empty result
        pd_result = pd_filtered['b'].sum()

        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_filtered = ds_df[ds_df['a'] > 100]
        ds_result = ds_filtered['b'].sum()

        assert pd_result == ds_result

    def test_mixed_dtypes_in_operations(self):
        """Operations with mixed dtypes"""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        pd_result = pd_df.assign(
            sum_col=lambda x: x['int_col'] + x['float_col']
        )

        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['a', 'b', 'c']
        })
        ds_result = ds_df.assign(
            sum_col=lambda x: x['int_col'] + x['float_col']
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSelectAfterComputation:
    """Test column selection after various computations"""

    def test_select_computed_column(self):
        """Select only the computed column"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        pd_df = pd_df.copy()
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df[['c']]

        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df[['c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_after_filter_and_compute(self):
        """Select after filter and compute"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        pd_filtered['c'] = pd_filtered['a'] * 10
        pd_result = pd_filtered[['a', 'c']].reset_index(drop=True)

        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_filtered = ds_df[ds_df['a'] > 2]
        ds_filtered['c'] = ds_filtered['a'] * 10
        ds_result = ds_filtered[['a', 'c']].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_original_keep_computed(self):
        """Drop original columns, keep computed"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        pd_df = pd_df.copy()
        pd_df['c'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df.drop(columns=['a', 'b'])

        ds_df = DataStore({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds_df['c'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df.drop(columns=['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)
