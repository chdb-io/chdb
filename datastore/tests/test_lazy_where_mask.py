"""
Tests for LazyWhere and LazyMask SQL pushdown functionality.

These tests verify:
1. LazyWhere/LazyMask can be pushed to SQL using CASE WHEN
2. The ClickHouse WHERE/CASE WHEN alias conflict is properly handled with subqueries
3. Results match pandas behavior for value replacement operations
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.lazy_ops import LazyWhere, LazyMask
from tests.test_utils import assert_datastore_equals_pandas


DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")


def dataset_path(filename: str) -> str:
    return os.path.join(DATASET_DIR, filename)


class TestLazyWhereBasic:
    """Basic LazyWhere functionality tests."""

    def test_where_simple_condition(self):
        """where() with simple condition replaces values correctly."""
        df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5],
                'b': [10, 20, 30, 40, 50],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['a'] > 2, 0)

            pd_result = df.where(df['a'] > 2, 0)

            # Values should match (ignoring dtype differences)
            np.testing.assert_array_equal(ds_result, pd_result)

    def test_where_replaces_false_values(self):
        """where() replaces values where condition is False."""
        df = pd.DataFrame(
            {
                'x': [100, 200, 300, 400, 500],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['x'] > 250, -1)

            pd_result = df.where(df['x'] > 250, -1)

            np.testing.assert_array_equal(ds_result, pd_result)

    def test_where_with_none_other(self):
        """where() with other=None uses NULL/NaN."""
        df = pd.DataFrame(
            {
                'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['a'] > 3)

            pd_result = df.where(df['a'] > 3)

            # Both should have NaN/NULL for first 3 rows
            # SQL uses NULL which becomes NaN in pandas
            assert pd.isna(ds_result['a'].iloc[0]) or ds_result['a'].iloc[0] is None
            assert pd.isna(ds_result['a'].iloc[1]) or ds_result['a'].iloc[1] is None
            assert pd.isna(ds_result['a'].iloc[2]) or ds_result['a'].iloc[2] is None

            # Last 2 rows should have original values
            assert ds_result['a'].iloc[3] == 4.0
            assert ds_result['a'].iloc[4] == 5.0


class TestLazyMaskBasic:
    """Basic LazyMask functionality tests."""

    def test_mask_simple_condition(self):
        """mask() with simple condition replaces values correctly."""
        df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5],
                'b': [10, 20, 30, 40, 50],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['a'] > 2, 0)

            pd_result = df.mask(df['a'] > 2, 0)

            np.testing.assert_array_equal(ds_result, pd_result)

    def test_mask_replaces_true_values(self):
        """mask() replaces values where condition is True (inverse of where)."""
        df = pd.DataFrame(
            {
                'x': [100, 200, 300, 400, 500],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['x'] > 250, -1)

            pd_result = df.mask(df['x'] > 250, -1)

            assert list(ds_result['x']) == list(pd_result['x'])


class TestWhereMaskSQLPushdown:
    """Test SQL pushdown for where/mask operations."""

    def test_where_generates_case_when_sql(self):
        """where() generates CASE WHEN SQL."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'value': [100, 200, 300],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_where = ds.where(ds['value'] > 150, 0)

            # Check that LazyWhere was added
            assert len(ds_where._lazy_ops) == 1
            assert isinstance(ds_where._lazy_ops[0], LazyWhere)

            # Execute and verify result
            result = ds_where
            expected = df.where(df['value'] > 150, 0)
            np.testing.assert_array_equal(result, expected)

    def test_mask_generates_case_when_sql(self):
        """mask() generates CASE WHEN SQL."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'value': [100, 200, 300],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_mask = ds.mask(ds['value'] > 150, 0)

            # Check that LazyMask was added
            assert len(ds_mask._lazy_ops) == 1
            assert isinstance(ds_mask._lazy_ops[0], LazyMask)

            # Execute and verify result
            result = ds_mask
            expected = df.mask(df['value'] > 150, 0)
            np.testing.assert_array_equal(result, expected)


class TestWhereFilterChain:
    """Test where + filter chain with subquery isolation.

    This tests the ClickHouse-specific fix where WHERE and CASE WHEN
    must be isolated in subqueries to avoid alias conflicts.
    """

    def test_where_then_filter_row_count(self):
        """where() + filter should return correct row count.

        This is the key test for the ClickHouse alias conflict fix.
        Without the subquery fix, ClickHouse would return wrong row count.
        """
        df = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'value': [100, 200, 300, 400, 500],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)

            # where + filter chain
            ds_result = ds.where(ds['value'] > 250, 0)[ds['category'] == 'A']

            # Pandas equivalent: where then filter using ORIGINAL df for condition
            pd_temp = df.where(df['value'] > 250, 0)
            pd_result = pd_temp[df['category'] == 'A']

            # Row count must match
            assert len(ds_result) == len(
                pd_result
            ), f"Row count mismatch: DataStore={len(ds_result)}, Pandas={len(pd_result)}"

            # Should be 3 rows (id=0,2,4 where original category='A')
            assert len(ds_result) == 3

    def test_where_then_filter_values(self):
        """where() + filter should have correct values."""
        df = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'value': [100, 200, 300, 400, 500],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 250, 0)[ds['category'] == 'A']

            pd_temp = df.where(df['value'] > 250, 0)
            pd_result = pd_temp[df['category'] == 'A'].reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # id and value columns should match exactly
            np.testing.assert_array_equal(ds_result['id'], pd_result['id'])
            np.testing.assert_array_equal(ds_result['value'], pd_result['value'])

    def test_filter_then_where(self):
        """filter then where should work correctly."""
        df = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'value': [100, 200, 300, 400, 500],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)

            # filter then where
            ds_result = ds[ds['category'] == 'A'].where(ds['value'] > 250, 0)

            # Pandas: filter first, then where
            pd_result = df[df['category'] == 'A'].where(df['value'] > 250, 0)

            # Should have 3 rows (category='A' rows)
            assert len(ds_result) == 3

            # With Variant type, SQL now preserves mixed types like pandas
            # Both should have: id=[0, 2, 4], value=[0, 300, 500], category=[0, 'A', 'A']
            np.testing.assert_array_equal(ds_result, pd_result)

    def test_mask_then_filter(self):
        """mask() + filter should work correctly."""
        df = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'value': [100, 200, 300, 400, 500],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)

            # mask + filter chain
            ds_result = ds.mask(ds['value'] > 250, -1)[ds['category'] == 'A']

            # Pandas equivalent
            pd_result = df.mask(df['value'] > 250, -1)[df['category'] == 'A']

            # Row count must match
            assert len(ds_result) == len(pd_result)
            assert len(ds_result) == 3


class TestWhereWithDifferentTypes:
    """Test where/mask with different data types and 'other' values."""

    def test_where_string_column_int_other(self):
        """where() on string column with int other value."""
        df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'score': [100, 50, 75],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # This should convert 0 to '0' for string column in SQL
            ds_result = ds.where(ds['score'] > 60, 0)

            pd_result = df.where(df['score'] > 60, 0)

            # Score column should match
            np.testing.assert_array_equal(ds_result['score'], pd_result['score'])

    def test_where_preserves_column_order(self):
        """where() preserves original column order."""
        df = pd.DataFrame(
            {
                'z': [1, 2, 3],
                'a': [4, 5, 6],
                'm': [7, 8, 9],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['a'] > 4, 0)

            # Column order should be preserved
            assert list(ds_result.columns) == ['z', 'a', 'm']


class TestLazyWhereImmutability:
    """Test that where/mask operations maintain DataStore immutability."""

    def test_where_returns_new_datastore(self):
        """where() returns a new DataStore, original unchanged."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_new = ds.where(ds['a'] > 1, 0)

            # Should be different objects
            assert ds is not ds_new

            # Original should have no lazy ops
            assert len(ds._lazy_ops) == 0

            # New should have LazyWhere
            assert len(ds_new._lazy_ops) == 1

    def test_mask_returns_new_datastore(self):
        """mask() returns a new DataStore, original unchanged."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_new = ds.mask(ds['a'] > 1, 0)

            assert ds is not ds_new
            assert len(ds._lazy_ops) == 0
            assert len(ds_new._lazy_ops) == 1


class TestWhereWithMultipleColumns:
    """Test where/mask with multiple columns of various types."""

    def test_where_on_mixed_columns(self):
        """where() on dataset with mixed column types (all SQL-compatible)."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3, 4, 5],
                'score': [85, 92, 78, 65, 88],
                'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['score'] > 80, 0)
            pd_result = df.where(df['score'] > 80, 0)

            # Score and id columns should match (numeric)
            np.testing.assert_array_equal(ds_result['score'], pd_result['score'])
            np.testing.assert_array_equal(ds_result['id'], pd_result['id'])

    def test_where_filter_chain_mixed_types(self):
        """where() + filter chain with string filtering."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3, 4, 5],
                'score': [85, 92, 78, 65, 88],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)

            # where + filter chain
            ds_result = ds.where(ds['score'] > 80, 0)[ds['category'] == 'A']

            pd_temp = df.where(df['score'] > 80, 0)
            pd_result = pd_temp[df['category'] == 'A']

            # Row counts should match
            assert len(ds_result) == len(pd_result)

            # Numeric columns should match
            np.testing.assert_array_equal(ds_result['score'], pd_result['score'])


class TestWhereMaskEngineSwitch:
    """Test function-level backend switching for where/mask operations."""

    def test_where_default_uses_chdb(self):
        """By default, where() should use chDB (SQL CASE WHEN)."""
        from datastore.function_executor import function_config

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Reset to default
            function_config.reset()

            ds = DataStore.from_file(path)
            lazy_where = ds.where(ds['a'] > 2, 0)

            # Check LazyWhere can push to SQL
            assert len(lazy_where._lazy_ops) == 1
            assert lazy_where._lazy_ops[0].can_push_to_sql() is True

    def test_where_switch_to_pandas(self):
        """When configured, where() should use Pandas instead of SQL."""
        from datastore.function_executor import function_config

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            try:
                # Configure to use Pandas for 'where'
                function_config.use_pandas('where')

                ds = DataStore.from_file(path)
                lazy_where = ds.where(ds['a'] > 2, 0)

                # Check LazyWhere cannot push to SQL (should use Pandas)
                assert len(lazy_where._lazy_ops) == 1
                assert lazy_where._lazy_ops[0].can_push_to_sql() is False

                # Result should still be correct
                result = lazy_where
                expected = df.where(df['a'] > 2, 0)
                np.testing.assert_array_equal(result, expected)
            finally:
                # Reset config
                function_config.reset()

    def test_mask_switch_to_pandas(self):
        """When configured, mask() should use Pandas instead of SQL."""
        from datastore.function_executor import function_config

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            try:
                # Configure to use Pandas for 'mask'
                function_config.use_pandas('mask')

                ds = DataStore.from_file(path)
                lazy_mask = ds.mask(ds['a'] > 2, 0)

                # Check LazyMask cannot push to SQL
                assert len(lazy_mask._lazy_ops) == 1
                assert lazy_mask._lazy_ops[0].can_push_to_sql() is False

                # Result should still be correct
                result = lazy_mask
                expected = df.mask(df['a'] > 2, 0)
                np.testing.assert_array_equal(result, expected)
            finally:
                function_config.reset()

    def test_where_switch_back_to_chdb(self):
        """After switching to Pandas, can switch back to chDB."""
        from datastore.function_executor import function_config

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            try:
                # First use Pandas
                function_config.use_pandas('where')
                ds = DataStore.from_file(path)
                lazy_where1 = ds.where(ds['a'] > 2, 0)
                assert lazy_where1._lazy_ops[0].can_push_to_sql() is False

                # Switch back to chDB
                function_config.use_chdb('where')
                ds2 = DataStore.from_file(path)
                lazy_where2 = ds2.where(ds2['a'] > 2, 0)
                assert lazy_where2._lazy_ops[0].can_push_to_sql() is True
            finally:
                function_config.reset()

    def test_prefer_pandas_affects_where_mask(self):
        """prefer_pandas() should affect both where and mask."""
        from datastore.function_executor import function_config

        df = pd.DataFrame({'a': [1, 2, 3]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            try:
                # Set global preference to Pandas
                function_config.prefer_pandas()

                ds = DataStore.from_file(path)
                lazy_where = ds.where(ds['a'] > 1, 0)
                lazy_mask = ds.mask(ds['a'] > 1, 0)

                # Both should not push to SQL
                assert lazy_where._lazy_ops[0].can_push_to_sql() is False
                assert lazy_mask._lazy_ops[0].can_push_to_sql() is False

                # Results should still be correct
                where_result = lazy_where
                mask_result = lazy_mask

                where_expected = df.where(df['a'] > 1, 0)
                mask_expected = df.mask(df['a'] > 1, 0)

                np.testing.assert_array_equal(where_result, where_expected)
                np.testing.assert_array_equal(mask_result, mask_expected)
            finally:
                function_config.reset()

    def test_independent_where_mask_config(self):
        """where and mask can be configured independently."""
        from datastore.function_executor import function_config

        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            try:
                # where -> Pandas, mask -> chDB
                function_config.use_pandas('where')
                function_config.use_chdb('mask')

                ds = DataStore.from_file(path)
                lazy_where = ds.where(ds['a'] > 2, 0)
                lazy_mask = ds.mask(ds['a'] > 2, 0)

                # where uses Pandas, mask uses chDB
                assert lazy_where._lazy_ops[0].can_push_to_sql() is False
                assert lazy_mask._lazy_ops[0].can_push_to_sql() is True
            finally:
                function_config.reset()

    def test_engine_config_with_filter_chain(self):
        """Engine config works correctly with where + filter chain.

        Note: When using Pandas execution for where, the filter condition
        applies to the RESULT of where (where category may have changed),
        not the original data. This is different from SQL pushdown where
        filter applies to original column values.
        """
        from datastore.function_executor import function_config

        df = pd.DataFrame(
            {
                'id': [0, 1, 2, 3, 4],
                'value': [100, 200, 300, 400, 500],
                'category': ['A', 'B', 'A', 'B', 'A'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            try:
                # Force Pandas execution for where
                function_config.use_pandas('where')

                ds = DataStore.from_file(path)
                ds_result = ds.where(ds['value'] > 250, 0)[ds['category'] == 'A']

                # When Pandas executes where, then filter applies to RESULT
                # pd_temp[pd_temp['category'] == 'A'] (not pd_temp[df['category'] == 'A'])
                pd_temp = df.where(df['value'] > 250, 0)
                pd_result = pd_temp[pd_temp['category'] == 'A']

                # Row count and values should match
                assert len(ds_result) == len(pd_result)
                # Should be 2 rows (only id=2,4 where category is still 'A')
                assert len(ds_result) == 2
            finally:
                function_config.reset()


class TestWhereWithVariousDataTypes:
    """Test where/mask with various data types to verify Variant type handling."""

    def test_where_float_column_int_other(self):
        """where() on float column with int other value."""
        df = pd.DataFrame({'value': [1.5, 2.5, 3.5, 4.5, 5.5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 3.0, 0)
            pd_result = df.where(df['value'] > 3.0, 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_int_column_float_other(self):
        """where() on int column with float other value.

        Auto-fallback: Float other with Int column causes NO_COMMON_TYPE in SQL,
        so DataStore automatically falls back to Pandas execution.
        """
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # This will auto-fallback to Pandas (float other + int column)
            ds_result = ds.where(ds['value'] > 3, -1.5)
            pd_result = df.where(df['value'] > 3, -1.5)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_string_column_string_other(self):
        """where() on string column with string other value.

        Auto-fallback: String other with numeric columns causes NO_COMMON_TYPE in SQL,
        so DataStore automatically falls back to Pandas execution.
        """
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'], 'score': [85, 92, 78, 65, 88]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # This will auto-fallback to Pandas (string other + numeric column)
            ds_result = ds.where(ds['score'] > 80, 'FAIL')
            pd_result = df.where(df['score'] > 80, 'FAIL')

            # name column: should have 'FAIL' where score <= 80
            np.testing.assert_array_equal(ds_result['name'].values, pd_result['name'].values)

    def test_where_bool_column_int_other(self):
        """where() on bool column with int other value."""
        df = pd.DataFrame({'flag': [True, False, True, False, True], 'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 2, 0)
            pd_result = df.where(df['value'] > 2, 0)

            np.testing.assert_array_equal(ds_result['value'].values, pd_result['value'].values)

    def test_where_datetime_column_preserves_type(self):
        """where() on datetime column handles type correctly."""
        df = pd.DataFrame(
            {'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']), 'value': [100, 200, 300]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # For datetime, numeric other should become NULL
            ds_result = ds.where(ds['value'] > 150)  # other=None (default)
            pd_result = df.where(df['value'] > 150)

            # value column should match
            np.testing.assert_array_equal(ds_result['value'].values, pd_result['value'].values)

    def test_where_negative_other(self):
        """where() with negative other value."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 25, -999)
            pd_result = df.where(df['value'] > 25, -999)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_zero_other(self):
        """where() with zero as other value."""
        df = pd.DataFrame({'value': [-2, -1, 0, 1, 2]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 0, 0)
            pd_result = df.where(df['value'] > 0, 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_large_numeric_other(self):
        """where() with large numeric other value."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 3, 999999999)
            pd_result = df.where(df['value'] > 3, 999999999)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)


class TestWhereWithComplexConditions:
    """Test where/mask with complex conditions."""

    def test_where_and_condition(self):
        """where() with AND condition."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where((ds['a'] > 2) & (ds['b'] < 45), 0)
            pd_result = df.where((df['a'] > 2) & (df['b'] < 45), 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_or_condition(self):
        """where() with OR condition."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [50, 40, 30, 20, 10]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where((ds['a'] > 4) | (ds['b'] > 40), -1)
            pd_result = df.where((df['a'] > 4) | (df['b'] > 40), -1)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_not_condition(self):
        """where() with NOT condition."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(~(ds['value'] > 3), 0)
            pd_result = df.where(~(df['value'] > 3), 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_comparison_between_columns(self):
        """where() with condition comparing two columns."""
        df = pd.DataFrame({'a': [1, 5, 3, 8, 2], 'b': [3, 2, 3, 5, 7]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['a'] > ds['b'], 0)
            pd_result = df.where(df['a'] > df['b'], 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_chained_where_operations(self):
        """Multiple chained where() operations."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 1, 0).where(ds['value'] < 5, 99)
            pd_result = df.where(df['value'] > 1, 0).where(df['value'] < 5, 99)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_chained_where_mask(self):
        """Chained where() and mask() operations."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 1, 0).mask(ds['value'] > 4, -1)
            pd_result = df.where(df['value'] > 1, 0).mask(df['value'] > 4, -1)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)


class TestWhereEdgeCases:
    """Test where/mask edge cases and boundary conditions."""

    def test_where_single_row(self):
        """where() on single row DataFrame."""
        df = pd.DataFrame({'value': [42]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 50, 0)
            pd_result = df.where(df['value'] > 50, 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_all_true(self):
        """where() when all conditions are True (no replacements)."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 0, -1)
            pd_result = df.where(df['value'] > 0, -1)

            # All values should be unchanged
            np.testing.assert_array_equal(ds_result.values, pd_result.values)
            np.testing.assert_array_equal(ds_result['value'].values, [10, 20, 30, 40, 50])

    def test_where_all_false(self):
        """where() when all conditions are False (all replaced)."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 100, -1)
            pd_result = df.where(df['value'] > 100, -1)

            # All values should be replaced
            np.testing.assert_array_equal(ds_result.values, pd_result.values)
            np.testing.assert_array_equal(ds_result['value'].values, [-1, -1, -1, -1, -1])

    def test_where_with_null_values_in_data(self):
        """where() on data containing NULL/NaN values."""
        df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 2, 0)
            pd_result = df.where(df['value'] > 2, 0)

            # Compare non-NaN values
            ds_vals = ds_result['value'].values
            pd_vals = pd_result['value'].values

            for i in range(len(ds_vals)):
                if pd.isna(pd_vals[i]):
                    assert pd.isna(ds_vals[i]) or ds_vals[i] == 0
                else:
                    assert ds_vals[i] == pd_vals[i]

    def test_where_many_columns(self):
        """where() on DataFrame with many columns."""
        df = pd.DataFrame({f'col_{i}': list(range(i, i + 5)) for i in range(10)})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['col_5'] > 7, 0)
            pd_result = df.where(df['col_5'] > 7, 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_mask_all_true(self):
        """mask() when all conditions are True (all replaced)."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['value'] > 0, -1)
            pd_result = df.mask(df['value'] > 0, -1)

            # All values should be replaced
            np.testing.assert_array_equal(ds_result.values, pd_result.values)
            np.testing.assert_array_equal(ds_result['value'].values, [-1, -1, -1, -1, -1])

    def test_mask_all_false(self):
        """mask() when all conditions are False (no replacements)."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['value'] > 100, -1)
            pd_result = df.mask(df['value'] > 100, -1)

            # All values should be unchanged
            np.testing.assert_array_equal(ds_result.values, pd_result.values)
            np.testing.assert_array_equal(ds_result['value'].values, [1, 2, 3, 4, 5])


class TestWhereMixedTypeScenarios:
    """Test where/mask with mixed type scenarios - Variant type behavior."""

    def test_where_mixed_types_multiple_string_columns(self):
        """where() on DataFrame with multiple string columns."""
        df = pd.DataFrame(
            {'name': ['Alice', 'Bob', 'Charlie'], 'city': ['NYC', 'LA', 'Chicago'], 'score': [85, 60, 75]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['score'] > 70, 0)
            pd_result = df.where(df['score'] > 70, 0)

            # All columns should match pandas (falls back to Pandas for mixed types)
            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_string_column_preserves_int_type(self):
        """Verify where() preserves integer type for string columns (pandas alignment).

        When DataFrame has string columns and numeric 'other', DataStore falls back
        to Pandas execution to preserve mixed type behavior (int 0 in object dtype).
        """
        df = pd.DataFrame({'name': ['A', 'B', 'C'], 'value': [1, 2, 3]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 1, 0)

            # For rows where value <= 1, name should be 0 (int), not '0' (str)
            # This matches pandas behavior where object dtype preserves int
            name_values = list(ds_result['name'].values)
            assert name_values[0] == 0  # int, not '0'
            assert isinstance(name_values[0], (int, np.integer))
            assert name_values[1] == 'B'
            assert name_values[2] == 'C'

    def test_where_float_other_on_string_column(self):
        """where() with float other on string column.

        Auto-fallback: Float other with Int column causes NO_COMMON_TYPE in SQL,
        so DataStore automatically falls back to Pandas execution.
        """
        df = pd.DataFrame({'name': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # This will auto-fallback to Pandas (float other + int column)
            ds_result = ds.where(ds['value'] > 2, 1.5)
            pd_result = df.where(df['value'] > 2, 1.5)

            # name column should have 1.5 (float) for replaced values
            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_mask_mixed_types_preserves_types(self):
        """mask() also preserves types with Variant."""
        df = pd.DataFrame({'category': ['X', 'Y', 'Z'], 'value': [10, 20, 30]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['value'] > 15, -1)
            pd_result = df.mask(df['value'] > 15, -1)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_where_numeric_columns_no_variant(self):
        """where() on pure numeric DataFrame doesn't need Variant."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [100, 200, 300, 400, 500]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['a'] > 2, 0)
            pd_result = df.where(df['a'] > 2, 0)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)


class TestWhereWithFilterCombinations:
    """Test various combinations of where/mask with filter operations."""

    def test_multiple_filters_then_where(self):
        """Multiple filter operations followed by where."""
        df = pd.DataFrame({'id': range(10), 'value': [i * 10 for i in range(10)], 'category': ['A', 'B'] * 5})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # Filter twice, then where
            ds_result = ds[ds['category'] == 'A'][ds['value'] > 20].where(ds['id'] > 3, 0)

            pd_filtered = df[df['category'] == 'A']
            # Note: Using original df for subsequent filters triggers UserWarning about reindexing
            # This tests the edge case where user applies filter from original DataFrame to filtered result
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Boolean Series key will be reindexed')
                pd_filtered = pd_filtered[df['value'] > 20]
                pd_result = pd_filtered.where(df['id'] > 3, 0)

            assert len(ds_result) == len(pd_result)

    def test_where_between_filters(self):
        """where() operation between two filter operations."""
        df = pd.DataFrame({'id': range(10), 'value': [i * 10 for i in range(10)], 'flag': [True, False] * 5})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # filter -> where -> filter
            ds_result = ds[ds['flag'] == True].where(ds['value'] > 30, 0)[ds['id'] < 8]

            pd_temp = df[df['flag'] == True]
            pd_temp = pd_temp.where(df['value'] > 30, 0)
            # Note: Using original df for filter triggers UserWarning about reindexing
            # This tests the edge case where user applies filter from original DataFrame
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Boolean Series key will be reindexed')
                pd_result = pd_temp[df['id'] < 8]

            assert len(ds_result) == len(pd_result)

    def test_mask_with_string_filter(self):
        """mask() combined with string column filter."""
        df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve'],
                'score': [85, 72, 91, 68, 79],
                'grade': ['A', 'C', 'A', 'D', 'B'],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            # mask high scores, then filter by grade
            ds_result = ds.mask(ds['score'] > 80, 0)[ds['grade'].isin(['A', 'B'])]

            pd_temp = df.mask(df['score'] > 80, 0)
            pd_result = pd_temp[df['grade'].isin(['A', 'B'])]

            assert len(ds_result) == len(pd_result)

    def test_head_after_where(self):
        """head() after where() operation."""
        df = pd.DataFrame({'value': list(range(100))})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 50, 0).head(10)

            pd_result = df.where(df['value'] > 50, 0).head(10)

            np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_sort_after_where(self):
        """sort_values() after where() operation."""
        df = pd.DataFrame({'id': [5, 3, 1, 4, 2], 'value': [50, 30, 10, 40, 20]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 25, 0).sort_values('id')

            pd_result = df.where(df['value'] > 25, 0).sort_values('id')

            np.testing.assert_array_equal(
                ds_result.reset_index(drop=True).values, pd_result.reset_index(drop=True).values
            )


class TestWhereWithComputedColumns:
    """Test where/mask with lazy column assignments (computed columns).

    This tests the fix for the issue where where() referencing a computed column
    would fail with 'Unknown expression identifier' because the computed column
    hadn't been materialized before the SQL execution.
    """

    def test_where_with_computed_column_condition(self):
        """where() condition referencing a lazy assigned column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_df['b'] = pd_df['a'] * 2
        pd_result = pd_df.where(pd_df['b'] > 5)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds['b'] = ds['a'] * 2
        ds_result = ds.where(ds['b'] > 5)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_mask_with_computed_column_condition(self):
        """mask() condition referencing a lazy assigned column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_df['b'] = pd_df['a'] * 2
        pd_result = pd_df.mask(pd_df['b'] > 5, -1)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds['b'] = ds['a'] * 2
        ds_result = ds.mask(ds['b'] > 5, -1)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_where_with_computed_column_and_filter(self):
        """Filter followed by where with computed column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        pd_df['b'] = pd_df['a'] * 2
        pd_filtered = pd_df[pd_df['a'] > 2]
        pd_result = pd_filtered.where(pd_filtered['b'] > 10)

        ds = DataStore({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds['b'] = ds['a'] * 2
        ds_filtered = ds[ds['a'] > 2]
        ds_result = ds_filtered.where(ds_filtered['b'] > 10)

        # Compare with check_index=False since filter changes indices
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_where_with_multiple_computed_columns(self):
        """where() with multiple computed columns in the condition."""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        pd_df['y'] = pd_df['x'] + 1
        pd_df['z'] = pd_df['x'] * pd_df['y']
        pd_result = pd_df.where(pd_df['z'] > 6)

        ds = DataStore({'x': [1, 2, 3, 4, 5]})
        ds['y'] = ds['x'] + 1
        ds['z'] = ds['x'] * ds['y']
        ds_result = ds.where(ds['z'] > 6)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_where_computed_column_with_other_value(self):
        """where() with computed column and explicit other value."""
        pd_df = pd.DataFrame({'val': [10, 20, 30, 40, 50]})
        pd_df['doubled'] = pd_df['val'] * 2
        pd_result = pd_df.where(pd_df['doubled'] > 50, 0)

        ds = DataStore({'val': [10, 20, 30, 40, 50]})
        ds['doubled'] = ds['val'] * 2
        ds_result = ds.where(ds['doubled'] > 50, 0)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)

    def test_where_chain_with_computed_column(self):
        """where() chained with other operations after computed column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8]})
        pd_df['b'] = pd_df['a'] + 10
        pd_result = pd_df.where(pd_df['b'] > 13).dropna()

        ds = DataStore({'a': [1, 2, 3, 4, 5, 6, 7, 8]})
        ds['b'] = ds['a'] + 10
        ds_result = ds.where(ds['b'] > 13).dropna()

        # Compare with check_index=False since dropna changes indices
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=False)
