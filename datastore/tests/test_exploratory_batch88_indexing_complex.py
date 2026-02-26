"""
Exploratory Batch 88: Complex Indexing, Type Coercion, and Nested Operations

Focus areas:
1. Complex iloc/loc edge cases with various indexers
2. Type coercion in mixed-type operations
3. Multi-level nested operations (assign + groupby + transform + filter)
4. Sparse data patterns and extreme NULL scenarios
5. Categorical and timedelta operations
6. Boolean indexing edge cases
7. Multi-column iloc/loc selections
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestComplexIlocEdgeCases:
    """Test complex iloc indexing scenarios"""

    def test_iloc_single_row_single_col(self):
        """iloc with single row and single column"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.iloc[2, 1]
        ds_result = ds_df.iloc[2, 1]

        assert ds_result == pd_result

    def test_iloc_slice_rows_list_cols(self):
        """iloc with row slice and column list"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500]
        })

        pd_result = pd_df.iloc[1:4, [0, 2]]
        ds_result = ds_df.iloc[1:4, [0, 2]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative_indices(self):
        """iloc with negative indices"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.iloc[-3:-1]
        ds_result = ds_df.iloc[-3:-1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_step_slicing(self):
        """iloc with step in slice"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8],
            'b': [10, 20, 30, 40, 50, 60, 70, 80]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5, 6, 7, 8],
            'b': [10, 20, 30, 40, 50, 60, 70, 80]
        })

        pd_result = pd_df.iloc[::2]  # every other row
        ds_result = ds_df.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_list_of_indices(self):
        """iloc with list of specific row indices"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df.iloc[[0, 2, 4]]
        ds_result = ds_df.iloc[[0, 2, 4]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_empty_slice(self):
        """iloc with empty slice result"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })

        pd_result = pd_df.iloc[5:10]  # beyond bounds
        ds_result = ds_df.iloc[5:10]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexLocEdgeCases:
    """Test complex loc indexing scenarios"""

    def test_loc_boolean_array(self):
        """loc with boolean array indexer"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        mask = [True, False, True, False, True]
        pd_result = pd_df.loc[mask]
        ds_result = ds_df.loc[mask]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_with_column_selection(self):
        """loc with row filter and column selection"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df.loc[pd_df['a'] > 2, ['a', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 2, ['a', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_single_column_selection(self):
        """loc with condition and single column as string"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.loc[pd_df['a'] > 2, 'b']
        ds_result = ds_df.loc[ds_df['a'] > 2, 'b']

        assert_series_equal(ds_result, pd_result)


class TestMixedTypeCoercion:
    """Test type coercion in mixed-type operations"""

    def test_int_float_addition(self):
        """Addition of int and float columns"""
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.5, 2.5, 3.5, 4.5]
        })
        ds_df = DataStore({
            'int_col': [1, 2, 3, 4],
            'float_col': [1.5, 2.5, 3.5, 4.5]
        })

        pd_df['sum'] = pd_df['int_col'] + pd_df['float_col']
        ds_df['sum'] = ds_df['int_col'] + ds_df['float_col']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_int_string_comparison(self):
        """Filter with int column compared to float literal"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df[pd_df['a'] > 2.5]
        ds_result = ds_df[ds_df['a'] > 2.5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_arithmetic(self):
        """Arithmetic on boolean results"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })

        # Sum of boolean conditions
        pd_df['matches'] = (pd_df['a'] > 2).astype(int) + (pd_df['b'] > 2).astype(int)
        ds_df['matches'] = (ds_df['a'] > 2).astype(int) + (ds_df['b'] > 2).astype(int)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestNestedOperationChains:
    """Test multi-level nested operations"""

    def test_assign_filter_groupby_agg(self):
        """Chain: assign -> filter -> groupby -> agg"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60]
        })

        pd_df['doubled'] = pd_df['value'] * 2
        pd_result = pd_df[pd_df['doubled'] > 40].groupby('category')['doubled'].sum().reset_index()

        ds_df['doubled'] = ds_df['value'] * 2
        ds_result = ds_df[ds_df['doubled'] > 40].groupby('category')['doubled'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_assign_sort_head(self):
        """Chain: filter -> assign -> sort -> head"""
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85, 92, 78, 95, 88]
        })
        ds_df = DataStore({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85, 92, 78, 95, 88]
        })

        pd_df = pd_df[pd_df['score'] > 80]
        pd_df['grade'] = pd_df['score'] // 10
        pd_result = pd_df.sort_values('score', ascending=False).head(3)

        ds_df = ds_df[ds_df['score'] > 80]
        ds_df['grade'] = ds_df['score'] // 10
        ds_result = ds_df.sort_values('score', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_transform_filter_sort(self):
        """Chain: groupby transform -> filter -> sort"""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_df['group_mean'] = pd_df.groupby('group')['value'].transform('mean')
        pd_result = pd_df[pd_df['value'] > pd_df['group_mean']].sort_values('value')

        ds_df['group_mean'] = ds_df.groupby('group')['value'].transform('mean')
        ds_result = ds_df[ds_df['value'] > ds_df['group_mean']].sort_values('value')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_assign_chain(self):
        """Chain: multiple sequential assigns"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5]
        })

        pd_df['b'] = pd_df['a'] * 2
        pd_df['c'] = pd_df['b'] + pd_df['a']
        pd_df['d'] = pd_df['c'] ** 2
        pd_df['e'] = pd_df['d'] - pd_df['a']

        ds_df['b'] = ds_df['a'] * 2
        ds_df['c'] = ds_df['b'] + ds_df['a']
        ds_df['d'] = ds_df['c'] ** 2
        ds_df['e'] = ds_df['d'] - ds_df['a']

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestSparseDataPatterns:
    """Test sparse data and NULL handling"""

    def test_mostly_null_column_filter(self):
        """Filter on mostly-NULL column"""
        pd_df = pd.DataFrame({
            'a': [1, None, None, None, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, None, None, None, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_null_column_operations(self):
        """Operations on all-NULL column"""
        pd_df = pd.DataFrame({
            'a': [None, None, None],
            'b': [1, 2, 3]
        })
        ds_df = DataStore({
            'a': [None, None, None],
            'b': [1, 2, 3]
        })

        pd_result = pd_df.dropna(subset=['a'])
        ds_result = ds_df.dropna(subset=['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_null_pattern(self):
        """Mixed NULL pattern across columns"""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, None, 5],
            'b': [None, 2, None, 4, None],
            'c': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, None, 3, None, 5],
            'b': [None, 2, None, 4, None],
            'c': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.dropna(how='all', subset=['a', 'b'])
        ds_result = ds_df.dropna(how='all', subset=['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_with_computed_value(self):
        """fillna with computed value from another column"""
        pd_df = pd.DataFrame({
            'a': [1, None, 3, None, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [1, None, 3, None, 5],
            'b': [10, 20, 30, 40, 50]
        })

        pd_df['a'] = pd_df['a'].fillna(pd_df['b'] / 10)
        ds_df['a'] = ds_df['a'].fillna(ds_df['b'] / 10)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestBooleanIndexingEdgeCases:
    """Test boolean indexing edge cases"""

    def test_complex_boolean_expression(self):
        """Complex boolean expression with multiple conditions"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': ['x', 'y', 'x', 'y', 'x']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': ['x', 'y', 'x', 'y', 'x']
        })

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] < 5) & (pd_df['c'] == 'x')]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] < 5) & (ds_df['c'] == 'x')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_or_expression(self):
        """Boolean OR expression"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['A', 'B', 'C', 'A', 'B']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['A', 'B', 'C', 'A', 'B']
        })

        pd_result = pd_df[(pd_df['a'] < 2) | (pd_df['a'] > 4)]
        ds_result = ds_df[(ds_df['a'] < 2) | (ds_df['a'] > 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negation_condition(self):
        """Negation of condition"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df[~(pd_df['a'] > 3)]
        ds_result = ds_df[~(ds_df['a'] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.xfail(reason="chDB limitation: IN clause requires constant or table expression, empty list not supported")
    def test_isin_with_empty_list(self):
        """isin with empty list"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })

        pd_result = pd_df[pd_df['a'].isin([])]
        ds_result = ds_df[ds_df['a'].isin([])]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCategoricalOperations:
    """Test categorical dtype operations"""

    # chDB v4.x supports categorical SQL operations but converts dtype to object
    @pytest.mark.xfail(
        reason="chDB converts categorical to object dtype after SQL execution - VALUES ARE CORRECT",
        strict=True,
    )
    def test_categorical_groupby(self):
        """GroupBy on categorical column"""
        pd_df = pd.DataFrame({
            'cat': pd.Categorical(['A', 'B', 'A', 'B', 'C']),
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'cat': pd.Categorical(['A', 'B', 'A', 'B', 'C']),
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.groupby('cat', observed=True)['value'].sum().reset_index()
        ds_result = ds_df.groupby('cat', observed=True)['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    # chDB v4.x supports categorical SQL operations but converts dtype to object
    @pytest.mark.xfail(
        reason="chDB converts categorical to object dtype after SQL execution - VALUES ARE CORRECT",
        strict=True,
    )
    def test_categorical_filter(self):
        """Filter on categorical column"""
        pd_df = pd.DataFrame({
            'cat': pd.Categorical(['A', 'B', 'A', 'B', 'C']),
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'cat': pd.Categorical(['A', 'B', 'A', 'B', 'C']),
            'value': [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df['cat'] == 'A']
        ds_result = ds_df[ds_df['cat'] == 'A']

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTimedeltaOperations:
    """Test timedelta operations"""

    @pytest.mark.xfail(reason="DataStore issue: timedelta result unit mismatch (seconds vs nanoseconds)")
    def test_timedelta_arithmetic(self):
        """Arithmetic with timedelta columns"""
        pd_df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'end': pd.to_datetime(['2024-01-05', '2024-01-06', '2024-01-10'])
        })
        ds_df = DataStore({
            'start': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'end': pd.to_datetime(['2024-01-05', '2024-01-06', '2024-01-10'])
        })

        pd_df['duration'] = pd_df['end'] - pd_df['start']
        ds_df['duration'] = ds_df['end'] - ds_df['start']

        assert_datastore_equals_pandas(ds_df, pd_df)

    @pytest.mark.xfail(reason="DataStore issue: timedelta result unit mismatch (seconds vs nanoseconds)")
    def test_timedelta_filter(self):
        """Filter based on timedelta comparison"""
        pd_df = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'end': pd.to_datetime(['2024-01-05', '2024-01-06', '2024-01-10'])
        })
        ds_df = DataStore({
            'start': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'end': pd.to_datetime(['2024-01-05', '2024-01-06', '2024-01-10'])
        })

        pd_df['duration'] = pd_df['end'] - pd_df['start']
        ds_df['duration'] = ds_df['end'] - ds_df['start']

        pd_result = pd_df[pd_df['duration'] > pd.Timedelta(days=5)]
        ds_result = ds_df[ds_df['duration'] > pd.Timedelta(days=5)]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMultiColumnOperations:
    """Test operations involving multiple columns"""

    def test_sum_across_columns(self):
        """Sum across multiple columns"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30],
            'c': [100, 200, 300]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [10, 20, 30],
            'c': [100, 200, 300]
        })

        pd_df['total'] = pd_df['a'] + pd_df['b'] + pd_df['c']
        ds_df['total'] = ds_df['a'] + ds_df['b'] + ds_df['c']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_max_across_columns(self):
        """Find max across multiple columns per row"""
        pd_df = pd.DataFrame({
            'a': [1, 5, 3],
            'b': [4, 2, 6],
            'c': [3, 4, 1]
        })
        ds_df = DataStore({
            'a': [1, 5, 3],
            'b': [4, 2, 6],
            'c': [3, 4, 1]
        })

        pd_df['max_val'] = pd_df[['a', 'b', 'c']].max(axis=1)
        ds_df['max_val'] = ds_df[['a', 'b', 'c']].max(axis=1)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_min_across_columns(self):
        """Find min across multiple columns per row"""
        pd_df = pd.DataFrame({
            'a': [1, 5, 3],
            'b': [4, 2, 6],
            'c': [3, 4, 1]
        })
        ds_df = DataStore({
            'a': [1, 5, 3],
            'b': [4, 2, 6],
            'c': [3, 4, 1]
        })

        pd_df['min_val'] = pd_df[['a', 'b', 'c']].min(axis=1)
        ds_df['min_val'] = ds_df[['a', 'b', 'c']].min(axis=1)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestEdgeCaseChains:
    """Test unusual but valid operation chains"""

    def test_head_then_tail(self):
        """head followed by tail"""
        pd_df = pd.DataFrame({
            'a': list(range(20)),
            'b': list(range(100, 120))
        })
        ds_df = DataStore({
            'a': list(range(20)),
            'b': list(range(100, 120))
        })

        pd_result = pd_df.head(10).tail(3)
        ds_result = ds_df.head(10).tail(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_then_head(self):
        """tail followed by head"""
        pd_df = pd.DataFrame({
            'a': list(range(20)),
            'b': list(range(100, 120))
        })
        ds_df = DataStore({
            'a': list(range(20)),
            'b': list(range(100, 120))
        })

        pd_result = pd_df.tail(10).head(3)
        ds_result = ds_df.tail(10).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_to_empty_then_operations(self):
        """Filter to empty then continue operations"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [10, 20, 30]
        })

        pd_result = pd_df[pd_df['a'] > 100].head(5)
        ds_result = ds_df[ds_df['a'] > 100].head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_twice_different_columns(self):
        """Sort twice on different columns"""
        pd_df = pd.DataFrame({
            'a': [3, 1, 2, 1, 3],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'a': [3, 1, 2, 1, 3],
            'b': [10, 20, 30, 40, 50]
        })

        # Sort by a, then by b (effectively re-sorts by b)
        pd_result = pd_df.sort_values('a').sort_values('b')
        ds_result = ds_df.sort_values('a').sort_values('b')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_filter_on_new_name(self):
        """Rename column then filter using new name"""
        pd_df = pd.DataFrame({
            'old_name': [1, 2, 3, 4, 5],
            'other': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'old_name': [1, 2, 3, 4, 5],
            'other': [10, 20, 30, 40, 50]
        })

        pd_df = pd_df.rename(columns={'old_name': 'new_name'})
        pd_result = pd_df[pd_df['new_name'] > 2]

        ds_df = ds_df.rename(columns={'old_name': 'new_name'})
        ds_result = ds_df[ds_df['new_name'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestConcatAndAppend:
    """Test concat operations"""

    def test_concat_two_datastores_vertical(self):
        """Vertical concat of two DataStores"""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        ds_df1 = DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_df2 = DataStore({'a': [5, 6], 'b': [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = pd.concat([ds_df1._get_df(), ds_df2._get_df()], ignore_index=True)
        ds_result = DataStore(ds_result)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_with_different_columns(self):
        """Concat DataFrames with different columns"""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        pd_df2 = pd.DataFrame({'a': [5, 6], 'c': [7, 8]})

        ds_df1 = DataStore({'a': [1, 2], 'b': [3, 4]})
        ds_df2 = DataStore({'a': [5, 6], 'c': [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = pd.concat([ds_df1._get_df(), ds_df2._get_df()], ignore_index=True)
        ds_result = DataStore(ds_result)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNumpyInterop:
    """Test numpy interop operations"""

    def test_create_from_numpy_array(self):
        """Create DataStore from numpy array"""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        pd_df = pd.DataFrame(arr, columns=['a', 'b', 'c'])
        ds_df = DataStore(arr, columns=['a', 'b', 'c'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_numpy_values_property(self):
        """Access .values property"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        ds_df = DataStore({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

        pd_values = pd_df['a'].values
        ds_values = ds_df['a'].values

        np.testing.assert_array_equal(ds_values, pd_values)


class TestClipOperation:
    """Test clip operations"""

    def test_clip_single_column(self):
        """Clip values in single column"""
        pd_df = pd.DataFrame({
            'a': [-5, 0, 5, 10, 15, 20],
            'b': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'a': [-5, 0, 5, 10, 15, 20],
            'b': [1, 2, 3, 4, 5, 6]
        })

        pd_df['clipped'] = pd_df['a'].clip(lower=0, upper=10)
        ds_df['clipped'] = ds_df['a'].clip(lower=0, upper=10)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_clip_lower_only(self):
        """Clip with lower bound only"""
        pd_df = pd.DataFrame({
            'a': [-5, 0, 5, 10],
        })
        ds_df = DataStore({
            'a': [-5, 0, 5, 10],
        })

        pd_df['clipped'] = pd_df['a'].clip(lower=0)
        ds_df['clipped'] = ds_df['a'].clip(lower=0)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestRoundOperations:
    """Test rounding operations"""

    def test_round_column(self):
        """Round column to specific decimals"""
        pd_df = pd.DataFrame({
            'a': [1.234, 2.567, 3.891, 4.123]
        })
        ds_df = DataStore({
            'a': [1.234, 2.567, 3.891, 4.123]
        })

        pd_df['rounded'] = pd_df['a'].round(2)
        ds_df['rounded'] = ds_df['a'].round(2)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_floor_ceil(self):
        """Floor and ceiling operations"""
        pd_df = pd.DataFrame({
            'a': [1.2, 2.5, 3.8, 4.1]
        })
        ds_df = DataStore({
            'a': [1.2, 2.5, 3.8, 4.1]
        })

        pd_df['floored'] = np.floor(pd_df['a'])
        pd_df['ceiled'] = np.ceil(pd_df['a'])

        # Mirror pattern: same code as pandas - numpy interop works via __array__ protocol
        ds_df['floored'] = np.floor(ds_df['a'])
        ds_df['ceiled'] = np.ceil(ds_df['a'])

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestAbsoluteValue:
    """Test absolute value operations"""

    def test_abs_column(self):
        """Absolute value of column"""
        pd_df = pd.DataFrame({
            'a': [-5, -2, 0, 3, 7]
        })
        ds_df = DataStore({
            'a': [-5, -2, 0, 3, 7]
        })

        pd_df['abs_a'] = pd_df['a'].abs()
        ds_df['abs_a'] = ds_df['a'].abs()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_abs_expression(self):
        """Absolute value of expression"""
        pd_df = pd.DataFrame({
            'a': [1, 5, 3, 7, 2],
            'b': [3, 2, 5, 1, 4]
        })
        ds_df = DataStore({
            'a': [1, 5, 3, 7, 2],
            'b': [3, 2, 5, 1, 4]
        })

        pd_df['diff'] = (pd_df['a'] - pd_df['b']).abs()
        ds_df['diff'] = (ds_df['a'] - ds_df['b']).abs()

        assert_datastore_equals_pandas(ds_df, pd_df)
