"""
Tests for multi-datasource coexistence and cross-source operations.

This module tests:
1. Multiple DataStore instances with different data sources can coexist
2. The 'database' parameter semantic fix (remote db name vs local chDB path)
3. Cross-source operations when possible

Key Fix Verified:
- For table function sources (file, s3, mysql, clickhouse, etc.), self.database
  should always be ':memory:' to avoid chDB path conflicts
- The remote database name is correctly passed to table functions
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile

from datastore import DataStore
from datastore.executor import reset_executor
from tests.test_utils import assert_datastore_equals_pandas


class TestDatabaseParameterSemantic(unittest.TestCase):
    """Test the database parameter semantic fix."""

    def setUp(self):
        """Reset executor before each test."""
        reset_executor()

    def test_dataframe_source_uses_memory(self):
        """DataFrame DataStore should use :memory: for chDB."""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        ds = DataStore(df)

        self.assertEqual(ds.database, ':memory:')
        self.assertEqual(ds.source_type, 'dataframe')
        self.assertIsNone(ds._table_function)

    def test_file_source_uses_memory(self):
        """File DataStore should use :memory: for chDB."""
        ds = DataStore("file", path="/tmp/test.parquet")

        self.assertEqual(ds.database, ':memory:')
        self.assertEqual(ds.source_type, 'file')
        self.assertIsNotNone(ds._table_function)

    def test_s3_source_uses_memory(self):
        """S3 DataStore should use :memory: for chDB."""
        ds = DataStore("s3", url="s3://bucket/data.parquet", nosign=True)

        self.assertEqual(ds.database, ':memory:')
        self.assertEqual(ds.source_type, 's3')
        self.assertIsNotNone(ds._table_function)

    def test_mysql_source_uses_memory_with_correct_remote_db(self):
        """MySQL DataStore should use :memory: for chDB, with database passed to table function."""
        ds = DataStore.from_mysql("host:3306", "mydb", "users", "root", "pass")

        # chDB path should be :memory:
        self.assertEqual(ds.database, ':memory:')
        # Remote database name should be in table function params
        self.assertEqual(ds._table_function.params.get('database'), 'mydb')

    def test_clickhouse_source_uses_memory_with_correct_remote_db(self):
        """ClickHouse DataStore should use :memory: for chDB, with database passed to table function."""
        ds = DataStore.from_clickhouse("host:9000", "pypi", "downloads", "demo", "")

        # chDB path should be :memory:
        self.assertEqual(ds.database, ':memory:')
        # Remote database name should be in table function params
        self.assertEqual(ds._table_function.params.get('database'), 'pypi')

    def test_postgresql_source_uses_memory_with_correct_remote_db(self):
        """PostgreSQL DataStore should use :memory: for chDB, with database passed to table function."""
        ds = DataStore.from_postgresql("host:5432", "mydb", "users", "postgres", "pass")

        self.assertEqual(ds.database, ':memory:')
        self.assertEqual(ds._table_function.params.get('database'), 'mydb')

    def test_explicit_chdb_path_preserved(self):
        """Explicit chDB path should be preserved for non-table-function sources."""
        ds = DataStore(table='test', database='/tmp/custom_db')

        # This is a direct chDB table access, so database path should be preserved
        self.assertEqual(ds.database, '/tmp/custom_db')
        self.assertIsNone(ds._table_function)


class TestMultiDataSourceCoexistence(unittest.TestCase):
    """Test that multiple DataStore instances with different sources can coexist."""

    def setUp(self):
        """Reset executor before each test."""
        reset_executor()

    def test_dataframe_and_file_coexist(self):
        """DataFrame and file DataStores can coexist."""
        # Create DataFrame DataStore
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        ds_df = DataStore(df)

        # Create file DataStore (doesn't need to exist for this test)
        ds_file = DataStore("file", path="/tmp/test.parquet")

        # Both should use :memory:
        self.assertEqual(ds_df.database, ':memory:')
        self.assertEqual(ds_file.database, ':memory:')

        # DataFrame should still be usable
        self.assertEqual(list(ds_df.columns), ['id', 'name'])

    def test_multiple_dataframes_coexist(self):
        """Multiple DataFrame DataStores can coexist and operate independently."""
        # pandas reference
        pd_df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        pd_df2 = pd.DataFrame({'id': [4, 5, 6], 'score': [40, 50, 60]})

        # DataStore mirror
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        # Filter operations
        pd_result1 = pd_df1[pd_df1['value'] > 15]
        pd_result2 = pd_df2[pd_df2['score'] > 45]

        ds_result1 = ds_df1[ds_df1['value'] > 15]
        ds_result2 = ds_df2[ds_df2['score'] > 45]

        # Verify results
        assert_datastore_equals_pandas(ds_result1, pd_result1)
        assert_datastore_equals_pandas(ds_result2, pd_result2)

    def test_dataframe_with_real_file(self):
        """DataFrame and real file DataStores can coexist and both work."""
        # Create a real test file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            temp_path = f.name

        try:
            # Create test data
            file_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
            file_df.to_parquet(temp_path)

            # Create DataFrame DataStore
            memory_df = pd.DataFrame({'a': [10, 20], 'b': [30, 40]})
            ds_memory = DataStore(memory_df)

            # Create file DataStore
            ds_file = DataStore("file", path=temp_path)
            ds_file.connect()

            # Both should use :memory:
            self.assertEqual(ds_memory.database, ':memory:')
            self.assertEqual(ds_file.database, ':memory:')

            # Both should be queryable
            memory_result = ds_memory.filter(ds_memory['a'] > 10)
            file_result = ds_file.filter(ds_file['x'] > 1)

            # Verify DataFrame result
            pd_expected = memory_df[memory_df['a'] > 10]
            assert_datastore_equals_pandas(memory_result, pd_expected)

            # Verify file result
            pd_file_expected = file_df[file_df['x'] > 1]
            assert_datastore_equals_pandas(file_result, pd_file_expected)
        finally:
            os.unlink(temp_path)


import pytest


@pytest.mark.network
class TestRemoteClickHouseIntegration(unittest.TestCase):
    """
    Integration tests with real remote ClickHouse server.

    Uses the public ClickHouse demo server:
    - Host: sql-clickhouse.clickhouse.com:9440
    - Database: pypi
    - Table: pypi_downloads
    - User: demo
    - Password: (empty)
    - Secure: True (required for port 9440)

    These tests require network access and are marked with @pytest.mark.network.
    Run with: pytest -m network tests/test_multi_datasource.py
    Skip with: pytest -m "not network" tests/
    """

    def setUp(self):
        """Reset executor before each test."""
        reset_executor()

    def test_remote_clickhouse_database_param(self):
        """Remote ClickHouse connection should use correct database param."""
        ds = DataStore.from_clickhouse(
            'sql-clickhouse.clickhouse.com:9440', 'pypi', 'pypi_downloads', 'demo', '', secure=True
        )

        # chDB path should be :memory:
        self.assertEqual(ds.database, ':memory:')
        # Remote database should be 'pypi'
        self.assertEqual(ds._table_function.params.get('database'), 'pypi')

    def test_remote_clickhouse_with_local_dataframe(self):
        """Remote ClickHouse and local DataFrame can coexist."""
        # Create local DataFrame first
        local_df = pd.DataFrame({'project': ['numpy', 'pandas', 'scipy'], 'local_score': [100, 200, 150]})
        ds_local = DataStore(local_df)

        # Create remote ClickHouse connection
        ds_remote = DataStore.from_clickhouse(
            'sql-clickhouse.clickhouse.com:9440', 'pypi', 'pypi_downloads', 'demo', '', secure=True
        )

        # Both should use :memory:
        self.assertEqual(ds_local.database, ':memory:')
        self.assertEqual(ds_remote.database, ':memory:')

        # Connect to remote
        ds_remote.connect()

        # Query remote
        remote_result = ds_remote.select('project', 'count').filter(ds_remote['project'] == 'numpy').limit(1)

        # Should have results
        self.assertGreater(len(remote_result), 0)
        self.assertEqual(list(remote_result.columns), ['project', 'count'])

        # Local should still work
        local_result = ds_local.filter(ds_local['project'] == 'numpy')
        self.assertEqual(len(local_result), 1)
        self.assertEqual(local_result['local_score'].iloc[0], 100)

    def test_query_remote_clickhouse_basic(self):
        """Basic query to remote ClickHouse server."""
        ds = DataStore.from_clickhouse(
            'sql-clickhouse.clickhouse.com:9440', 'pypi', 'pypi_downloads', 'demo', '', secure=True
        )
        ds.connect()

        # Simple query
        result = ds.select('project', 'count').limit(5)

        # Should have 5 rows
        self.assertEqual(len(result), 5)
        # Should have correct columns
        self.assertEqual(list(result.columns), ['project', 'count'])

    def test_query_remote_clickhouse_with_filter(self):
        """Query remote ClickHouse with filter."""
        ds = DataStore.from_clickhouse(
            'sql-clickhouse.clickhouse.com:9440', 'pypi', 'pypi_downloads', 'demo', '', secure=True
        )
        ds.connect()

        # Query with filter
        result = ds.select('project', 'count').filter(ds['project'] == 'requests').limit(3)

        # All results should have project = 'requests'
        self.assertTrue(all(result['project'] == 'requests'))

    def test_query_remote_clickhouse_with_aggregation(self):
        """Query remote ClickHouse with aggregation."""
        ds = DataStore.from_clickhouse(
            'sql-clickhouse.clickhouse.com:9440', 'pypi', 'pypi_downloads', 'demo', '', secure=True
        )
        ds.connect()

        # Aggregation query
        result = ds.filter(ds['project'] == 'numpy').groupby('project').agg(total_count=('count', 'sum'))

        # Should have aggregated result
        self.assertEqual(len(result), 1)
        self.assertIn('total_count', result.columns)


if __name__ == '__main__':
    unittest.main()
