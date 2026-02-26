"""
Test loc condition pushdown to SQL.

This module tests the lazy execution of loc[condition, columns] pattern
that converts to SQL: SELECT col1, col2 WHERE condition
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal


# =======================
# Test Fixtures
# =======================


@pytest.fixture
def df_basic():
    return pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50],
        'c': ['x', 'y', 'z', 'w', 'v']
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        'a': [1, 2, None, 4, 5],
        'b': [10.0, np.nan, 30.0, 40.0, 50.0],
        'c': ['x', 'y', None, 'w', 'v']
    })


@pytest.fixture
def parquet_file(df_basic):
    """Create a temporary parquet file for testing file-based SQL pushdown."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        parquet_path = f.name
        df_basic.to_parquet(parquet_path, index=False)
    yield parquet_path
    os.unlink(parquet_path)


# =======================
# Part 1: Lazy Operation Creation
# =======================


class TestLocLazyOpCreation:
    """Test that loc creates lazy operations instead of immediate execution."""

    def test_loc_condition_creates_where_op(self, df_basic):
        """Test that loc[condition] creates a WHERE lazy op."""
        ds = DataStore(df_basic)
        result = ds.loc[ds['a'] > 2]

        # Should be a DataStore
        assert isinstance(result, DataStore)

        # Should have lazy ops including WHERE
        where_ops = [op for op in result._lazy_ops if hasattr(op, 'op_type') and op.op_type == 'WHERE']
        assert len(where_ops) >= 1

    def test_loc_condition_columns_creates_where_and_select(self, df_basic):
        """Test that loc[condition, columns] creates WHERE + SELECT lazy ops."""
        ds = DataStore(df_basic)
        result = ds.loc[ds['a'] > 2, ['b', 'c']]

        # Should be a DataStore
        assert isinstance(result, DataStore)

        # Should have both WHERE and SELECT ops
        where_ops = [op for op in result._lazy_ops if hasattr(op, 'op_type') and op.op_type == 'WHERE']
        select_ops = [op for op in result._lazy_ops if hasattr(op, 'op_type') and op.op_type == 'SELECT']

        assert len(where_ops) >= 1
        assert len(select_ops) >= 1

    def test_loc_condition_single_column_string(self, df_basic):
        """Test loc[condition, single_column] with string column."""
        ds = DataStore(df_basic)
        result = ds.loc[ds['a'] > 2, 'b']

        # Should be a DataStore (column selection returns DataStore with one column)
        assert isinstance(result, DataStore)


# =======================
# Part 2: Results Match Pandas
# =======================


class TestLocConditionPushdownResults:
    """Test that loc condition pushdown produces correct results."""

    def test_loc_greater_than(self, df_basic):
        """Test loc[col > value, columns] matches pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 2, ['b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 2, ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_less_than_equal(self, df_basic):
        """Test loc[col <= value, columns] matches pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] <= 3, ['a', 'b']]
        ds_result = ds_df.loc[ds_df['a'] <= 3, ['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_equal(self, df_basic):
        """Test loc[col == value, columns] matches pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] == 3, ['b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] == 3, ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_not_equal(self, df_basic):
        """Test loc[col != value, columns] matches pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] != 3, ['a', 'c']]
        ds_result = ds_df.loc[ds_df['a'] != 3, ['a', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_compound_and(self, df_basic):
        """Test loc[(cond1) & (cond2), columns] matches pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[(pd_df['a'] > 1) & (pd_df['a'] < 5), ['b', 'c']]
        ds_result = ds_df.loc[(ds_df['a'] > 1) & (ds_df['a'] < 5), ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_compound_or(self, df_basic):
        """Test loc[(cond1) | (cond2), columns] matches pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[(pd_df['a'] == 1) | (pd_df['a'] == 5), ['b', 'c']]
        ds_result = ds_df.loc[(ds_df['a'] == 1) | (ds_df['a'] == 5), ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_condition_only(self, df_basic):
        """Test loc[condition] without column selection."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 2]
        ds_result = ds_df.loc[ds_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_all_columns(self, df_basic):
        """Test loc[condition, all_columns] selects all columns."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 2, ['a', 'b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 2, ['a', 'b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)


# =======================
# Part 3: SQL Pushdown Verification
# =======================


class TestLocSQLPushdown:
    """Test that loc operations are pushed down to SQL."""

    def test_sql_contains_where(self, parquet_file):
        """Test that generated SQL contains WHERE clause."""
        ds = DataStore.from_file(parquet_file)
        result = ds.loc[ds['a'] > 2, ['b', 'c']]

        sql = result.to_sql(execution_format=True)

        # SQL should contain WHERE clause
        assert 'WHERE' in sql
        assert '"a" > 2' in sql

    def test_sql_contains_column_selection(self, parquet_file):
        """Test that generated SQL selects specified columns."""
        ds = DataStore.from_file(parquet_file)
        result = ds.loc[ds['a'] > 2, ['b', 'c']]

        sql = result.to_sql(execution_format=True)

        # SQL should select specific columns
        assert '"b"' in sql
        assert '"c"' in sql

    def test_chained_loc_operations(self, parquet_file):
        """Test chained loc operations create combined SQL."""
        ds = DataStore.from_file(parquet_file)
        result = ds.loc[ds['a'] > 1].loc[ds['a'] < 5, ['b', 'c']]

        sql = result.to_sql(execution_format=True)

        # Should have WHERE conditions for both filters
        assert 'WHERE' in sql


# =======================
# Part 4: Edge Cases
# =======================


class TestLocEdgeCases:
    """Test edge cases for loc condition pushdown."""

    def test_loc_empty_result(self, df_basic):
        """Test loc that returns empty result."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 100, ['b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 100, ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)
        assert len(ds_result) == 0

    def test_loc_all_match(self, df_basic):
        """Test loc that matches all rows."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 0, ['b', 'c']]
        ds_result = ds_df.loc[ds_df['a'] > 0, ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)
        assert len(ds_result) == len(df_basic)

    def test_loc_single_column_list(self, df_basic):
        """Test loc with single column in list."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 2, ['b']]
        ds_result = ds_df.loc[ds_df['a'] > 2, ['b']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_preserves_column_order(self, df_basic):
        """Test that loc preserves requested column order."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 2, ['c', 'b', 'a']]
        ds_result = ds_df.loc[ds_df['a'] > 2, ['c', 'b', 'a']]

        # Check column order matches
        assert list(pd_result.columns) == list(ds_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)


# =======================
# Part 5: Fallback to Pandas
# =======================


class TestLocFallbackToPandas:
    """Test that non-pushable loc patterns fall back to pandas."""

    def test_loc_label_slice_fallback(self, df_basic):
        """Test that label slice falls back to pandas (returns raw DataFrame)."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[1:3, ['a', 'b']]
        ds_result = ds_df.loc[1:3, ['a', 'b']]

        # Fallback returns raw pandas DataFrame, not DataStore
        assert_frame_equal(pd_result, ds_result)

    def test_loc_label_list_fallback(self, df_basic):
        """Test that label list falls back to pandas (returns raw DataFrame)."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[[0, 2, 4]]
        ds_result = ds_df.loc[[0, 2, 4]]

        # Fallback returns raw pandas DataFrame, not DataStore
        assert_frame_equal(pd_result, ds_result)

    def test_loc_single_label_fallback(self, df_basic):
        """Test that single label falls back to pandas."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[2]
        ds_result = ds_df.loc[2]

        assert_series_equal(pd_result, ds_result)


# =======================
# Part 6: Integration Tests
# =======================


class TestLocIntegration:
    """Integration tests combining loc with other operations."""

    def test_loc_then_head(self, df_basic):
        """Test loc followed by head operation."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 1, ['b', 'c']].head(2)
        ds_result = ds_df.loc[ds_df['a'] > 1, ['b', 'c']].head(2)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_loc_then_sort(self, df_basic):
        """Test loc followed by sort operation."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_result = pd_df.loc[pd_df['a'] > 1, ['a', 'b']].sort_values('b', ascending=False)
        ds_result = ds_df.loc[ds_df['a'] > 1, ['a', 'b']].sort_values('b', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_filter_then_loc(self, df_basic):
        """Test filter followed by loc."""
        pd_df = df_basic.copy()
        ds_df = DataStore(df_basic.copy())

        pd_filtered = pd_df[pd_df['a'] > 1]
        pd_result = pd_filtered.loc[pd_filtered['a'] < 5, ['b', 'c']]

        ds_filtered = ds_df[ds_df['a'] > 1]
        ds_result = ds_filtered.loc[ds_filtered['a'] < 5, ['b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)
