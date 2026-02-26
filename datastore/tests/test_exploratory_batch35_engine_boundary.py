"""
Exploratory Batch 35: Engine boundary and type preservation edge cases

Focus areas:
1. Multiple SQL-Pandas engine switches in a single chain
2. Type preservation across engine boundaries
3. Nullable types in complex aggregation chains
4. Edge cases when lazy ops interact with groupby results
5. Column assignment conflicts with aggregation aliases

Tests use Mirror Code Pattern: compare DataStore results with pandas results.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import (
    pandas_version_nullable_int_dtype,
    pandas_version_nullable_bool_sql,
)


# =============================================================================
# Test Group 1: Multiple Engine Switches
# =============================================================================


class TestMultipleEngineSwitches:
    """Test chains that switch between SQL and Pandas multiple times."""

    def test_sql_pandas_sql_chain_simple(self):
        """Test SQL -> Pandas (apply) -> SQL pattern."""
        data = {'a': [10, 20, 30, 40, 50], 'b': [1, 2, 3, 4, 5]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 15]  # SQL: filter
        pd_result = pd_result.apply(lambda row: row, axis=1)  # Pandas: apply
        pd_result = pd_result[pd_result['b'] > 2]  # Should go back to SQL

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 15]
        ds_result = ds_result.apply(lambda row: row, axis=1)
        ds_result = ds_result[ds_result['b'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sql_pandas_sql_chain_with_column_assignment(self):
        """Test SQL -> Pandas (apply) -> column assignment -> SQL filter."""
        data = {'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['x'] > 1]
        pd_result['z'] = pd_result.apply(lambda row: row['x'] + row['y'], axis=1)
        pd_result = pd_result[pd_result['z'] > 30]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['x'] > 1]
        ds_result['z'] = ds_result.apply(lambda row: row['x'] + row['y'], axis=1)
        ds_result = ds_result[ds_result['z'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_engine_switches(self):
        """Test pattern: SQL -> Pandas -> SQL -> Pandas."""
        data = {'a': list(range(1, 11)), 'b': list(range(10, 20)), 'c': ['A', 'B'] * 5}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 2]  # SQL
        pd_result['custom'] = pd_result.apply(lambda r: r['a'] * 10, axis=1)  # Pandas
        pd_result = pd_result[pd_result['b'] < 18]  # SQL
        pd_result['final'] = pd_result['custom'].apply(lambda x: x + 1)  # Pandas

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 2]
        ds_result['custom'] = ds_result.apply(lambda r: r['a'] * 10, axis=1)
        ds_result = ds_result[ds_result['b'] < 18]
        ds_result['final'] = ds_result['custom'].apply(lambda x: x + 1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_engine_switch_preserves_dtypes(self):
        """Test that data types are preserved across engine switches."""
        data = {
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['int_col'] > 1]  # SQL
        pd_result = pd_result.apply(lambda row: row, axis=1)  # Pandas (no-op)
        pd_result = pd_result[pd_result['int_col'] < 5]  # SQL

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['int_col'] > 1]
        ds_result = ds_result.apply(lambda row: row, axis=1)
        ds_result = ds_result[ds_result['int_col'] < 5]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 2: Nullable Types in Complex Chains
# =============================================================================


class TestNullableTypesInChains:
    """Test nullable types through complex operation chains.

    Note: Nullable integer dtype (Int64) preservation differs between pandas versions:
    - pandas 2.0.x: May return float64 for SQL operations with NULL/NA values
    - pandas 2.1+: Better nullable type preservation

    These tests are skipped on older pandas versions where dtype mismatch is expected.
    """

    @pandas_version_nullable_int_dtype
    def test_nullable_int_through_filter_chain(self):
        """Test nullable Int64 through multiple filters."""
        data = {'a': pd.array([1, 2, pd.NA, 4, 5, pd.NA, 7, 8], dtype='Int64'), 'b': list(range(8))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['b'] > 1]
        pd_result = pd_result[pd_df['b'] < 7]
        pd_result = pd_result[pd_result['a'].notna()]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['b'] > 1]
        ds_result = ds_result[ds_df['b'] < 7]
        ds_result = ds_result[ds_result['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_nullable_bool_sql
    def test_nullable_bool_in_where_chain(self):
        """Test nullable boolean in filter operation chain."""
        data = {
            'val': [1, 2, 3, 4, 5],
            'flag': pd.array([True, False, pd.NA, True, False], dtype='boolean'),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['val'] > 1]  # Filter step 1
        pd_result = pd_result[pd_result['flag'].fillna(False)]  # Filter step 2: nullable bool

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['val'] > 1]
        ds_result = ds_result[ds_result['flag'].fillna(False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_nullable_int_dtype
    def test_nullable_in_groupby_agg_chain(self):
        """Test nullable types through groupby -> aggregation -> filter chain."""
        data = {
            'group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'value': pd.array([1, pd.NA, 3, 4, pd.NA, 6], dtype='Int64'),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_result[pd_result['value'] > 3]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_result[ds_result['value'] > 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nullable_float_arithmetic_chain(self):
        """Test nullable Float64 through arithmetic operations."""
        data = {
            'a': pd.array([1.0, 2.0, pd.NA, 4.0, 5.0], dtype='Float64'),
            'b': pd.array([10.0, pd.NA, 30.0, 40.0, 50.0], dtype='Float64'),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_df['sum'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df[pd_df['sum'].notna()]

        # DataStore
        ds_df = DataStore(data)
        ds_df['sum'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df[ds_df['sum'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 3: GroupBy Result Operations
# =============================================================================


class TestGroupByResultOperations:
    """Test operations on groupby aggregation results."""

    def test_groupby_result_filter_and_sort(self):
        """Test filtering and sorting groupby results."""
        data = {'group': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'], 'value': [10, 20, 30, 40, 50, 60, 70, 80]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_result[pd_result['value'] > 50]
        pd_result = pd_result.sort_values('value', ascending=False)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_result[ds_result['value'] > 50]
        ds_result = ds_result.sort_values('value', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_result_column_assignment(self):
        """Test assigning new columns to groupby results."""
        data = {'category': ['X', 'X', 'Y', 'Y'], 'amount': [100, 200, 300, 400]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('category')['amount'].mean().reset_index()
        pd_result['doubled'] = pd_result['amount'] * 2

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('category')['amount'].mean().reset_index()
        ds_result['doubled'] = ds_result['amount'] * 2

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_multiple_aggs_then_filter(self):
        """Test multiple aggregations followed by filter."""
        data = {'g': ['A', 'A', 'A', 'B', 'B', 'C'], 'v1': [1, 2, 3, 4, 5, 6], 'v2': [10, 20, 30, 40, 50, 60]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('g').agg({'v1': 'sum', 'v2': 'mean'}).reset_index()
        pd_result = pd_result[pd_result['v1'] > 5]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('g').agg({'v1': 'sum', 'v2': 'mean'}).reset_index()
        ds_result = ds_result[ds_result['v1'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_count_with_nullable(self):
        """Test groupby count with nullable values."""
        data = {
            'group': ['A', 'A', 'B', 'B', 'B'],
            'value': pd.array([1, pd.NA, 3, pd.NA, pd.NA], dtype='Int64'),
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('group')['value'].count().reset_index()

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('group')['value'].count().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# Test Group 4: Head/Tail After Complex Operations
# =============================================================================


class TestHeadTailAfterComplexOps:
    """Test head/tail operations following complex chains."""

    def test_head_after_groupby_filter(self):
        """Test head after groupby and filter."""
        data = {'cat': ['A', 'B', 'C', 'A', 'B', 'C'] * 3, 'val': list(range(18))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('cat')['val'].sum().reset_index()
        pd_result = pd_result[pd_result['val'] > 10]
        pd_result = pd_result.head(2)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('cat')['val'].sum().reset_index()
        ds_result = ds_result[ds_result['val'] > 10]
        ds_result = ds_result.head(2)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_tail_after_sort_filter(self):
        """Test tail after sort and filter."""
        data = {'id': list(range(20)), 'score': list(range(100, 120))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.sort_values('score', ascending=False)
        pd_result = pd_result[pd_result['id'] > 5]
        pd_result = pd_result.tail(5)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.sort_values('score', ascending=False)
        ds_result = ds_result[ds_result['id'] > 5]
        ds_result = ds_result.tail(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_with_column_assignment_then_filter(self):
        """Test head followed by column assignment and filter."""
        data = {'a': list(range(50)), 'b': list(range(50, 100))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.head(30)
        pd_result['c'] = pd_result['a'] + pd_result['b']
        pd_result = pd_result[pd_result['c'] > 70]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.head(30)
        ds_result['c'] = ds_result['a'] + ds_result['b']
        ds_result = ds_result[ds_result['c'] > 70]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 5: Edge Cases with Empty Results
# =============================================================================


class TestEmptyResultEdgeCases:
    """Test edge cases that may produce empty results."""

    def test_filter_to_empty_then_groupby(self):
        """Test groupby on empty filtered result."""
        data = {'group': ['A', 'B', 'C'], 'value': [1, 2, 3]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['value'] > 100]  # Empty
        pd_result = pd_result.groupby('group')['value'].sum().reset_index()

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['value'] > 100]
        ds_result = ds_result.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_multiple_filters(self):
        """Test empty result after multiple filters."""
        data = {'x': [1, 2, 3], 'y': [10, 20, 30]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['x'] > 0][pd_df['x'] < 10][pd_df['y'] > 1000]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['x'] > 0][ds_df['x'] < 10][ds_df['y'] > 1000]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_result_column_assignment(self):
        """Test column assignment on empty result."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 100]  # Empty
        pd_result = pd_result.assign(c=pd_result['a'] + pd_result['b'])

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 100]
        ds_result = ds_result.assign(c=ds_result['a'] + ds_result['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 6: Complex Type Coercion Chains
# =============================================================================


class TestTypeCoercionChains:
    """Test type coercion through complex operation chains."""

    def test_int_to_float_in_arithmetic(self):
        """Test int -> float coercion in arithmetic."""
        data = {'int_col': [1, 2, 3, 4, 5], 'divisor': [2, 2, 2, 2, 2]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_df['result'] = pd_df['int_col'] / pd_df['divisor']
        pd_result = pd_df[pd_df['result'] > 1.0]

        # DataStore
        ds_df = DataStore(data)
        ds_df['result'] = ds_df['int_col'] / ds_df['divisor']
        ds_result = ds_df[ds_df['result'] > 1.0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_type_aggregation(self):
        """Test aggregation with mixed types."""
        data = {
            'group': ['A', 'A', 'B', 'B'],
            'int_val': [1, 2, 3, 4],
            'float_val': [1.5, 2.5, 3.5, 4.5],
        }

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('group').agg({'int_val': 'sum', 'float_val': 'mean'}).reset_index()

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('group').agg({'int_val': 'sum', 'float_val': 'mean'}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_string_numeric_column_interaction(self):
        """Test operations involving both string and numeric columns."""
        data = {'name': ['alice', 'bob', 'charlie'], 'score': [85, 92, 78], 'grade': ['B', 'A', 'C']}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['score'] > 80]
        pd_result = pd_result[pd_result['name'].str.len() > 3]
        pd_result['bonus'] = pd_result['score'] + 10

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['score'] > 80]
        ds_result = ds_result[ds_result['name'].str.len() > 3]
        ds_result['bonus'] = ds_result['score'] + 10

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 7: Slice and Index Operations
# =============================================================================


class TestSliceAndIndexOperations:
    """Test slice and index operations in chains."""

    def test_iloc_after_filter(self):
        """Test iloc after filter operation."""
        data = {'a': list(range(20)), 'b': list(range(20, 40))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df[pd_df['a'] > 5]
        pd_result = pd_result.iloc[2:8]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df[ds_df['a'] > 5]
        ds_result = ds_result.iloc[2:8]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_with_step_after_sort(self):
        """Test slice with step after sort."""
        data = {'val': [5, 3, 8, 1, 9, 2, 7, 4, 6, 0]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.sort_values('val')
        pd_result = pd_result.iloc[::2]  # Every other row

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.sort_values('val')
        ds_result = ds_result.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_tail_combination(self):
        """Test head followed by tail."""
        data = {'x': list(range(100))}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.head(50).tail(10)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.head(50).tail(10)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 8: Where/Mask with Complex Conditions
# =============================================================================


class TestWhereMaskComplexConditions:
    """Test where/mask with complex conditions and chains."""

    def test_where_with_computed_column(self):
        """Test where using a computed column condition."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_df['sum'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df.where(pd_df['sum'] > 5, 0)

        # DataStore
        ds_df = DataStore(data)
        ds_df['sum'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df.where(ds_df['sum'] > 5, 0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mask_then_filter(self):
        """Test mask followed by filter."""
        data = {'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.mask(pd_df['x'] < 3, 0)
        pd_result = pd_result[pd_result['x'] > 0]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.mask(ds_df['x'] < 3, 0)
        ds_result = ds_result[ds_result['x'] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_where_mask(self):
        """Test chained where and mask operations."""
        data = {'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.where(pd_df['val'] > 3, 0)
        pd_result = pd_result.mask(pd_result['val'] > 7, -1)

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.where(ds_df['val'] > 3, 0)
        ds_result = ds_result.mask(ds_result['val'] > 7, -1)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 9: Rename and Drop After Complex Operations
# =============================================================================


class TestRenameDropAfterComplexOps:
    """Test rename and drop column operations after complex chains."""

    def test_rename_after_groupby(self):
        """Test rename columns after groupby aggregation."""
        data = {'category': ['A', 'A', 'B', 'B'], 'amount': [10, 20, 30, 40]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.groupby('category')['amount'].sum().reset_index()
        pd_result = pd_result.rename(columns={'amount': 'total_amount', 'category': 'cat'})

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.groupby('category')['amount'].sum().reset_index()
        ds_result = ds_result.rename(columns={'amount': 'total_amount', 'category': 'cat'})

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_drop_after_column_assignment(self):
        """Test drop columns after column assignment."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_df['d'] = pd_df['a'] + pd_df['b']
        pd_result = pd_df.drop(columns=['b', 'c'])

        # DataStore
        ds_df = DataStore(data)
        ds_df['d'] = ds_df['a'] + ds_df['b']
        ds_result = ds_df.drop(columns=['b', 'c'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_filter(self):
        """Test rename followed by filter on renamed column."""
        data = {'old_name': [1, 2, 3, 4, 5], 'other': [10, 20, 30, 40, 50]}

        # pandas
        pd_df = pd.DataFrame(data)
        pd_result = pd_df.rename(columns={'old_name': 'new_name'})
        pd_result = pd_result[pd_result['new_name'] > 2]

        # DataStore
        ds_df = DataStore(data)
        ds_result = ds_df.rename(columns={'old_name': 'new_name'})
        ds_result = ds_result[ds_result['new_name'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Test Group 10: Complex Join Chains
# =============================================================================


class TestComplexJoinChains:
    """Test complex operations before and after joins."""

    def test_filter_before_merge(self):
        """Test filter before merge operation."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'val1': [1, 2, 3, 4]})
        df2 = pd.DataFrame({'key': ['B', 'C', 'D', 'E'], 'val2': [10, 20, 30, 40]})

        # pandas
        pd_result = df1[df1['val1'] > 1].merge(df2, on='key', how='inner')

        # DataStore
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)
        ds_result = ds1[ds1['val1'] > 1].merge(ds2, on='key', how='inner')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_after_merge(self):
        """Test filter after merge operation."""
        df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val1': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['A', 'B', 'C'], 'val2': [10, 20, 30]})

        # pandas
        pd_result = df1.merge(df2, on='key', how='inner')
        pd_result = pd_result[pd_result['val1'] + pd_result['val2'] > 15]

        # DataStore
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)
        ds_result = ds1.merge(ds2, on='key', how='inner')
        ds_result = ds_result[ds_result['val1'] + ds_result['val2'] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_assignment_after_merge(self):
        """Test column assignment after merge."""
        df1 = pd.DataFrame({'id': [1, 2, 3], 'a': [10, 20, 30]})
        df2 = pd.DataFrame({'id': [1, 2, 3], 'b': [100, 200, 300]})

        # pandas
        pd_result = df1.merge(df2, on='id')
        pd_result['total'] = pd_result['a'] + pd_result['b']

        # DataStore
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)
        ds_result = ds1.merge(ds2, on='id')
        ds_result['total'] = ds_result['a'] + ds_result['b']

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
