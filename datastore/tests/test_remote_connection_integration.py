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
    """Test table selection via __getitem__."""

    def test_select_table_creates_working_datastore(self, clickhouse_connection):
        """ds["system.one"] creates usable DataStore."""
        one = clickhouse_connection["system.one"]

        # Should be able to execute
        df = one._execute()
        assert len(df) == 1

    def test_select_table_with_use(self, clickhouse_connection):
        """Table selection works with use()."""
        clickhouse_connection.use("system")
        tables = clickhouse_connection["tables"]

        df = tables.head()._execute()
        assert len(df) > 0

    def test_select_and_filter(self, clickhouse_connection):
        """Can filter after table selection."""
        tables = clickhouse_connection["system.tables"]

        # Filter to just system database tables
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
        """Test pandas-style operations on selected table."""
        users = clickhouse_connection["test_db.users"]

        # Filter
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
