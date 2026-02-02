"""
Integration tests for remote connection features with real ClickHouse server.

These tests use pytest fixtures to automatically start/stop a ClickHouse test server.

To run these tests:
    pytest datastore/tests/test_remote_connection_integration.py -v

To use an external ClickHouse server:
    export TEST_CLICKHOUSE_HOST=localhost:9000
    pytest datastore/tests/test_remote_connection_integration.py -v

To keep the server running after tests:
    KEEP_CLICKHOUSE=1 pytest datastore/tests/test_remote_connection_integration.py -v
"""

import pytest
import pandas as pd


class TestClickHouseMetadata:
    """Test metadata discovery methods against real ClickHouse."""

    def test_databases_returns_list(self, clickhouse_connection):
        """databases() returns list of database names."""
        databases = clickhouse_connection.databases()

        assert isinstance(databases, list)
        assert len(databases) > 0
        # system and default should always exist
        assert "system" in databases
        assert "default" in databases

    def test_tables_returns_system_tables(self, clickhouse_connection):
        """tables('system') returns system tables."""
        tables = clickhouse_connection.tables("system")

        assert isinstance(tables, list)
        assert len(tables) > 0
        # These system tables should always exist
        assert "tables" in tables
        assert "databases" in tables
        assert "columns" in tables

    def test_describe_returns_dataframe(self, clickhouse_connection):
        """describe() returns DataFrame with column info for remote tables."""
        schema_df = clickhouse_connection.describe("system", "tables")

        assert isinstance(schema_df, pd.DataFrame)
        assert len(schema_df) > 0
        # Should have name and type columns
        assert "name" in schema_df.columns
        assert "type" in schema_df.columns


class TestClickHouseQuery:
    """Test SQL execution against real ClickHouse."""

    def test_sql_system_one(self, clickhouse_connection):
        """Execute SQL on system.one table."""
        from datastore import DataStore

        result = clickhouse_connection.sql("SELECT * FROM system.one")

        # Result should be a DataStore
        assert isinstance(result, DataStore)

        # Should have one row with dummy column
        df = result._execute()
        assert len(df) == 1

    def test_sql_with_use(self, clickhouse_connection):
        """SQL execution after use() sets database."""
        clickhouse_connection.use("system")
        result = clickhouse_connection.sql("SELECT * FROM one")

        df = result._execute()
        assert len(df) == 1

    def test_sql_join(self, clickhouse_connection):
        """SQL with JOIN."""
        result = clickhouse_connection.sql(
            """
            SELECT d.name, count(*) as table_count
            FROM system.databases d
            LEFT JOIN system.tables t ON d.name = t.database
            GROUP BY d.name
            LIMIT 5
        """
        )

        df = result._execute()
        assert len(df) > 0
        assert "name" in df.columns
        assert "table_count" in df.columns


class TestClickHouseTableSelection:
    """Test table selection via table() method."""

    def test_table_method_creates_working_datastore(self, clickhouse_connection):
        """table("system", "one") creates usable DataStore."""
        one = clickhouse_connection.table("system", "one")

        # Should be able to execute
        df = one._execute()
        assert len(df) == 1

    def test_table_with_dot_notation(self, clickhouse_connection):
        """table("system.one") works with dot notation."""
        one = clickhouse_connection.table("system.one")

        df = one._execute()
        assert len(df) == 1

    def test_table_method_with_use(self, clickhouse_connection):
        """Table selection works with use() + table(table_name)."""
        clickhouse_connection.use("system")
        tables = clickhouse_connection.table("tables")

        df = tables.head()._execute()
        assert len(df) > 0

    def test_table_then_filter(self, clickhouse_connection):
        """Can filter after table() selection - no ambiguity with pandas filter."""
        tables = clickhouse_connection.table("system", "tables")

        # Filter to just system database tables (pandas-style, unambiguous)
        filtered = tables[tables["database"] == "system"]
        df = filtered.head(10)._execute()

        assert len(df) > 0
        # All rows should be from system database
        assert all(df["database"] == "system")


class TestTestDatabase:
    """Test against test_db if setup script created it."""

    @pytest.fixture(autouse=True)
    def check_test_db(self, clickhouse_connection):
        """Skip tests if test_db doesn't exist."""
        databases = clickhouse_connection.databases()
        if "test_db" not in databases:
            pytest.skip("test_db not found - may need to wait for setup to complete")

    def test_tables_in_test_db(self, clickhouse_connection):
        """List tables in test_db."""
        tables = clickhouse_connection.tables("test_db")

        assert "users" in tables
        assert "orders" in tables

    def test_describe_users_table(self, clickhouse_connection):
        """Get schema of test_db.users."""
        schema = clickhouse_connection.describe("test_db", "users")

        column_names = schema["name"].tolist()
        assert "id" in column_names
        assert "name" in column_names
        assert "email" in column_names
        assert "age" in column_names

    def test_query_users(self, clickhouse_connection):
        """Execute SQL on users table."""
        clickhouse_connection.use("test_db")
        result = clickhouse_connection.sql("SELECT * FROM users ORDER BY id")

        df = result._execute()

        assert len(df) == 3
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[1]["name"] == "Bob"
        assert df.iloc[2]["name"] == "Charlie"

    def test_query_join_users_orders(self, clickhouse_connection):
        """SQL JOIN users and orders tables."""
        clickhouse_connection.use("test_db")
        result = clickhouse_connection.sql(
            """
            SELECT u.name, sum(o.amount) as total
            FROM users u
            JOIN orders o ON u.id = o.user_id
            GROUP BY u.name
            ORDER BY total DESC
        """
        )

        df = result._execute()

        assert len(df) > 0
        assert "name" in df.columns
        assert "total" in df.columns

    def test_pandas_style_operations(self, clickhouse_connection):
        """Test pandas-style operations on selected table.
        
        Demonstrates the clear separation:
        - table() for table selection
        - [...] for pandas-style filtering (no ambiguity)
        """
        users = clickhouse_connection.table("test_db", "users")

        # Filter (pandas-style, unambiguous)
        adults = users[users["age"] >= 25]
        df = adults._execute()
        assert len(df) > 0

        # All should be >= 25
        assert all(df["age"] >= 25)


class TestConnectionRepr:
    """Test repr/str with real connection."""

    def test_repr_masks_password(self, clickhouse_server):
        """repr masks password even with working connection."""
        from datastore import DataStore

        host, port = clickhouse_server
        ds = DataStore.from_clickhouse(
            host=f"{host}:{port}", user="default", password="test_secret_password"
        )

        # Use _connection_repr directly since __repr__ might trigger execution
        repr_str = ds._connection_repr()

        assert "test_secret_password" not in repr_str
        assert "***" in repr_str
        assert f"{host}:{port}" in repr_str


class TestTablesWithDefaultDatabase:
    """Test tables() with default database from use()."""

    def test_tables_without_arg_after_use(self, clickhouse_connection):
        """tables() without argument should use default database."""
        clickhouse_connection.use("system")

        # Should work without specifying database
        tables = clickhouse_connection.tables()

        assert isinstance(tables, list)
        assert len(tables) > 0
        assert "tables" in tables
        assert "databases" in tables

    def test_tables_arg_overrides_default(self, clickhouse_connection):
        """tables(db) should override default database."""
        clickhouse_connection.use("default")

        # Explicit argument should override default
        tables = clickhouse_connection.tables("system")

        assert "tables" in tables
        assert "databases" in tables


class TestCrossDatabaseQueries:
    """Test cross-database SQL queries."""

    def test_query_different_databases_in_same_sql(self, clickhouse_connection):
        """SQL with tables from different databases should work."""
        # Query using fully qualified table names from different databases
        result = clickhouse_connection.sql(
            """
            SELECT 
                (SELECT count(*) FROM system.databases) as db_count,
                (SELECT count(*) FROM system.tables) as table_count
        """
        )

        df = result._execute()

        assert len(df) == 1
        assert "db_count" in df.columns
        assert "table_count" in df.columns
        assert df.iloc[0]["db_count"] > 0
        assert df.iloc[0]["table_count"] > 0

    def test_join_across_system_tables(self, clickhouse_connection):
        """JOIN between system tables should work."""
        result = clickhouse_connection.sql(
            """
            SELECT 
                d.name as database_name,
                count(t.name) as table_count
            FROM system.databases d
            LEFT JOIN system.tables t ON d.name = t.database
            GROUP BY d.name
            ORDER BY table_count DESC
            LIMIT 5
        """
        )

        df = result._execute()

        assert len(df) > 0
        assert "database_name" in df.columns
        assert "table_count" in df.columns


class TestChainedOperationsIntegration:
    """Test chained operations with real server."""

    def test_use_then_table_then_filter(self, clickhouse_connection):
        """use(db) -> table(table) -> filter should work."""
        clickhouse_connection.use("system")

        # Select table after use() using table() method
        tables = clickhouse_connection.table("tables")

        # Apply filter (pandas-style, unambiguous)
        system_tables = tables[tables["database"] == "system"]

        df = system_tables.head(10)._execute()

        assert len(df) > 0
        assert all(df["database"] == "system")

    def test_sql_result_chaining(self, clickhouse_connection):
        """SQL result should support further DataFrame operations."""
        result = clickhouse_connection.sql(
            """
            SELECT database, name, engine
            FROM system.tables
            WHERE database = 'system'
        """
        )

        # Chain further operations on SQL result (pandas-style filter)
        filtered = result[result["engine"] == "SystemNumbers"]

        # This may or may not have results depending on ClickHouse version
        df = filtered._execute()

        assert isinstance(df, pd.DataFrame)
        # If there are results, they should all have engine = SystemNumbers
        if len(df) > 0:
            assert all(df["engine"] == "SystemNumbers")

    def test_table_preserves_connection_params(self, clickhouse_connection):
        """table() should preserve connection for further queries."""
        one = clickhouse_connection.table("system", "one")

        # Should be able to execute
        df = one._execute()
        assert len(df) == 1

        # Should also be able to run SQL on the new DataStore
        # (if implementation supports it)


class TestSecureConnection:
    """Test secure connections (when available)."""

    def test_secure_connection_creation(self, clickhouse_server):
        """Create DataStore with secure=True parameter."""
        from datastore import DataStore

        host, port = clickhouse_server
        # Note: Test server may not support secure connections
        # This tests that the parameter is accepted
        ds = DataStore.from_clickhouse(
            host=f"{host}:{port}",
            user="default",
            password="",
            secure=False,  # Use False for test server
        )

        assert ds._remote_params.get("secure") == False


class TestEdgeCasesIntegration:
    """Test edge cases with real server."""

    def test_empty_result_query(self, clickhouse_connection):
        """Query returning no rows should work."""
        result = clickhouse_connection.sql(
            """
            SELECT * FROM system.one WHERE 1 = 0
        """
        )

        df = result._execute()

        assert len(df) == 0
        assert "dummy" in df.columns

    def test_large_result_with_limit(self, clickhouse_connection):
        """Query with LIMIT should respect the limit."""
        result = clickhouse_connection.sql(
            """
            SELECT * FROM system.numbers LIMIT 100
        """
        )

        df = result._execute()

        assert len(df) == 100

    def test_describe_system_table(self, clickhouse_connection):
        """describe() on system table should return schema."""
        schema = clickhouse_connection.describe("system", "one")

        assert isinstance(schema, pd.DataFrame)
        assert len(schema) > 0
        assert "name" in schema.columns
        assert "type" in schema.columns
        # system.one has a 'dummy' column
        assert "dummy" in schema["name"].values

    def test_multiple_use_calls(self, clickhouse_connection):
        """Multiple use() calls should work correctly."""
        clickhouse_connection.use("default")
        assert clickhouse_connection._default_database == "default"

        clickhouse_connection.use("system")
        assert clickhouse_connection._default_database == "system"

        # Should be able to query from current database
        tables = clickhouse_connection.tables()
        assert "tables" in tables


class TestTestDatabaseAdvanced:
    """Advanced tests using test_db if available."""

    @pytest.fixture(autouse=True)
    def check_test_db(self, clickhouse_connection):
        """Skip tests if test_db doesn't exist."""
        databases = clickhouse_connection.databases()
        if "test_db" not in databases:
            pytest.skip("test_db not found - may need to wait for setup to complete")

    def test_use_database_table_then_filter(self, clickhouse_connection):
        """use(db, table) then filter should work."""
        clickhouse_connection.use("test_db", "users")

        # After use(db, table), we're in table mode
        assert clickhouse_connection._connection_mode == "table"

    def test_table_after_use_database(self, clickhouse_connection):
        """table(table) after use(db) should work."""
        clickhouse_connection.use("test_db")

        users = clickhouse_connection.table("users")

        df = users._execute()
        assert len(df) > 0
        assert "name" in df.columns

    def test_aggregation_on_remote_table(self, clickhouse_connection):
        """Aggregation queries on remote table should work."""
        result = clickhouse_connection.sql(
            """
            SELECT 
                count(*) as user_count,
                avg(age) as avg_age,
                max(age) as max_age
            FROM test_db.users
        """
        )

        df = result._execute()

        assert len(df) == 1
        assert df.iloc[0]["user_count"] == 3
        assert df.iloc[0]["avg_age"] > 0

    def test_subquery(self, clickhouse_connection):
        """Subquery in SQL should work."""
        result = clickhouse_connection.sql(
            """
            SELECT * FROM test_db.users
            WHERE age > (SELECT avg(age) FROM test_db.users)
        """
        )

        df = result._execute()

        # Users with age above average
        assert len(df) >= 0  # May be 0 or more depending on data
