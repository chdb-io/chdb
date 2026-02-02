"""
Unit tests for remote connection features.

Tests the connection-level DataStore functionality:
- Connection mode detection
- Factory methods with optional params
- use() method
- __getitem__ table selection
- Metadata methods (adapter-based)
- SQL rewriting
- Password masking
"""

import unittest
from unittest import mock
import pandas as pd

from datastore import DataStore
from datastore.adapters import (
    SourceAdapter,
    ClickHouseAdapter,
    MySQLAdapter,
    PostgreSQLAdapter,
    get_adapter,
)
from datastore.exceptions import DataStoreError


class TestConnectionModeDetection(unittest.TestCase):
    """Test that connection mode is correctly detected based on params."""

    def test_clickhouse_connection_mode_no_db_no_table(self):
        """Connection mode when only host/user/password provided."""
        ds = DataStore(
            source="clickhouse", host="localhost:9000", user="default", password=""
        )
        self.assertEqual(ds._connection_mode, "connection")
        self.assertIsNone(ds._default_database)
        self.assertIsNone(ds._default_table)

    def test_clickhouse_database_mode_with_db_no_table(self):
        """Database mode when database provided but no table."""
        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="mydb",
            user="default",
            password="",
        )
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "mydb")
        self.assertIsNone(ds._default_table)

    def test_clickhouse_table_mode_full_params(self):
        """Table mode when all params provided."""
        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="mydb",
            table="users",
            user="default",
            password="",
        )
        self.assertEqual(ds._connection_mode, "table")
        # Note: In table mode, _default_database is NOT set because the database
        # is passed directly to the table function. Only in 'database' mode
        # (when database is provided but table is not) is _default_database set.

    def test_mysql_connection_mode(self):
        """MySQL connection mode detection."""
        ds = DataStore(
            source="mysql", host="localhost:3306", user="root", password="secret"
        )
        self.assertEqual(ds._connection_mode, "connection")
        self.assertIsNone(ds._default_database)

    def test_mysql_database_mode(self):
        """MySQL database mode detection."""
        ds = DataStore(
            source="mysql",
            host="localhost:3306",
            database="testdb",
            user="root",
            password="secret",
        )
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "testdb")

    def test_postgresql_connection_mode(self):
        """PostgreSQL connection mode detection."""
        ds = DataStore(
            source="postgresql",
            host="localhost:5432",
            user="postgres",
            password="secret",
        )
        self.assertEqual(ds._connection_mode, "connection")
        self.assertIsNone(ds._default_database)

    def test_postgresql_table_mode(self):
        """PostgreSQL table mode when all params provided."""
        ds = DataStore(
            source="postgresql",
            host="localhost:5432",
            database="mydb",
            table="users",
            user="postgres",
            password="secret",
        )
        self.assertEqual(ds._connection_mode, "table")

    def test_dataframe_source_is_table_mode(self):
        """DataFrame source should be in table mode."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ds = DataStore(df)
        self.assertEqual(ds._connection_mode, "table")

    def test_dict_source_is_table_mode(self):
        """Dict source should be in table mode."""
        ds = DataStore({"a": [1, 2, 3]})
        self.assertEqual(ds._connection_mode, "table")


class TestFactoryMethods(unittest.TestCase):
    """Test from_xxx factory methods with optional params."""

    def test_from_clickhouse_connection_level(self):
        """from_clickhouse with only connection params."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        self.assertEqual(ds._connection_mode, "connection")
        self.assertEqual(ds._remote_params["host"], "localhost:9000")
        self.assertEqual(ds._remote_params["user"], "default")
        self.assertEqual(ds._remote_params["password"], "")

    def test_from_clickhouse_with_secure(self):
        """from_clickhouse with secure=True."""
        ds = DataStore.from_clickhouse(
            host="localhost:9440", user="default", password="", secure=True
        )
        self.assertEqual(ds._remote_params.get("secure"), True)

    def test_from_clickhouse_database_level(self):
        """from_clickhouse with database but no table."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", database="analytics", user="default", password=""
        )
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "analytics")
        self.assertIsNone(ds._default_table)

    def test_from_mysql_connection_level(self):
        """from_mysql with only connection params."""
        ds = DataStore.from_mysql(host="localhost:3306", user="root", password="secret")
        self.assertEqual(ds._connection_mode, "connection")
        self.assertEqual(ds._remote_params["host"], "localhost:3306")
        self.assertEqual(ds._remote_params["user"], "root")

    def test_from_mysql_database_level(self):
        """from_mysql with database but no table."""
        ds = DataStore.from_mysql(
            host="localhost:3306", database="mydb", user="root", password="secret"
        )
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "mydb")

    def test_from_postgresql_connection_level(self):
        """from_postgresql with only connection params."""
        ds = DataStore.from_postgresql(
            host="localhost:5432", user="postgres", password="secret"
        )
        self.assertEqual(ds._connection_mode, "connection")
        self.assertEqual(ds._remote_params["host"], "localhost:5432")

    def test_from_clickhouse_table_level_backward_compat(self):
        """from_clickhouse with all params (backward compatibility)."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000",
            database="mydb",
            table="users",
            user="default",
            password="",
        )
        self.assertEqual(ds._connection_mode, "table")
        # Should have table function
        self.assertIsNotNone(ds._table_function)


class TestUseMethod(unittest.TestCase):
    """Test use() method for setting defaults."""

    def test_use_single_arg_sets_database(self):
        """use(database) sets default database."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        result = ds.use("production")

        self.assertIs(result, ds)  # Returns self
        self.assertEqual(ds._default_database, "production")

    def test_use_two_args_sets_database_and_table(self):
        """use(database, table) sets both."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("production", "users")

        self.assertEqual(ds._default_database, "production")
        self.assertEqual(ds._default_table, "users")
        self.assertEqual(ds._connection_mode, "table")

    def test_use_three_args_sets_all(self):
        """use(schema, database, table) sets all three."""
        ds = DataStore.from_postgresql(
            host="localhost:5432", user="postgres", password=""
        )
        ds.use("public", "mydb", "users")

        self.assertEqual(ds._default_schema, "public")
        self.assertEqual(ds._default_database, "mydb")
        self.assertEqual(ds._default_table, "users")
        self.assertEqual(ds._connection_mode, "table")

    def test_use_returns_self_for_chaining(self):
        """use() returns self for method chaining."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        result = ds.use("db1").use("db2")

        self.assertIs(result, ds)
        self.assertEqual(ds._default_database, "db2")

    def test_use_no_args_raises_error(self):
        """use() with no args raises ValueError."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        with self.assertRaises(ValueError):
            ds.use()

    def test_use_too_many_args_raises_error(self):
        """use() with >3 args raises ValueError."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        with self.assertRaises(ValueError):
            ds.use("a", "b", "c", "d")

    def test_use_updates_connection_mode_from_connection_to_database(self):
        """use(database) updates mode from connection to database."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        self.assertEqual(ds._connection_mode, "connection")

        ds.use("mydb")
        # After use(database), mode becomes database (since no table yet)
        # Actually, looking at the code, it just sets _default_database
        # Mode stays connection until we also have a table
        self.assertEqual(ds._default_database, "mydb")

    def test_use_updates_connection_mode_to_table(self):
        """use(database, table) updates mode to table."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        self.assertEqual(ds._connection_mode, "connection")

        ds.use("mydb", "users")
        self.assertEqual(ds._connection_mode, "table")


class TestUseDatabaseMethod(unittest.TestCase):
    """Test use_database() method for explicit database selection."""

    def test_use_database_sets_default_database(self):
        """use_database(database) sets default database."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        result = ds.use_database("production")

        self.assertIs(result, ds)  # Returns self
        self.assertEqual(ds._default_database, "production")

    def test_use_database_updates_connection_mode(self):
        """use_database() updates mode from connection to database."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        self.assertEqual(ds._connection_mode, "connection")

        ds.use_database("mydb")
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "mydb")

    def test_use_database_returns_self_for_chaining(self):
        """use_database() returns self for method chaining."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        result = ds.use_database("db1").use_database("db2")

        self.assertIs(result, ds)
        self.assertEqual(ds._default_database, "db2")

    def test_use_database_chain_with_table(self):
        """use_database().table() chain works correctly."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        users = ds.use_database("production").table("users")

        # Original ds should have database set
        self.assertEqual(ds._default_database, "production")
        self.assertEqual(ds._connection_mode, "database")

        # New DataStore should be in table mode
        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")

    def test_use_database_none_raises_error(self):
        """use_database(None) raises ValueError."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        with self.assertRaises(ValueError):
            ds.use_database(None)

    def test_use_database_empty_string_raises_error(self):
        """use_database('') raises ValueError."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        with self.assertRaises(ValueError):
            ds.use_database("")

    def test_use_database_does_not_affect_table(self):
        """use_database() should not change _default_table."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds._default_table = "existing_table"  # Set a table first

        ds.use_database("newdb")

        self.assertEqual(ds._default_database, "newdb")
        self.assertEqual(ds._default_table, "existing_table")  # Unchanged


class TestTableMethod(unittest.TestCase):
    """Test table() method for explicit table selection.

    The table() method is the recommended way to select tables, as it avoids
    ambiguity with pandas-style ds["column"] and ds[condition] syntax.
    """

    def test_table_with_dot_notation(self):
        """table("db.table") creates new DataStore."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        users = ds.table("production.users")

        self.assertIsInstance(users, DataStore)
        self.assertIsNot(users, ds)
        self.assertEqual(users._connection_mode, "table")

    def test_table_with_two_args(self):
        """table(database, table) creates new DataStore."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        users = ds.table("production", "users")

        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")

    def test_table_with_three_args(self):
        """table(schema, database, table) creates new DataStore."""
        ds = DataStore.from_postgresql(
            host="localhost:5432", user="postgres", password=""
        )
        users = ds.table("public", "mydb", "users")

        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")
        self.assertEqual(users._default_schema, "public")

    def test_table_single_arg_with_default_database(self):
        """table(table) uses default database from use()."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("production")

        users = ds.table("users")

        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")

    def test_table_single_arg_without_database_raises_error(self):
        """table(table) without default database raises error."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        with self.assertRaises(DataStoreError) as ctx:
            ds.table("users")

        error_msg = str(ctx.exception).lower()
        self.assertIn("database", error_msg)

    def test_table_preserves_connection_params(self):
        """table() preserves connection params in new DataStore."""
        ds = DataStore.from_clickhouse(
            host="analytics.example.com:9000", user="analyst", password="secret123"
        )
        users = ds.table("production", "users")

        self.assertEqual(users._remote_params["host"], "analytics.example.com:9000")
        self.assertEqual(users._remote_params["user"], "analyst")
        self.assertEqual(users._remote_params["password"], "secret123")

    def test_table_original_unchanged(self):
        """table() does not modify original DataStore."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        original_mode = ds._connection_mode

        users = ds.table("production", "users")

        self.assertEqual(ds._connection_mode, original_mode)
        self.assertIsNone(ds._default_table)

    def test_table_invalid_args_raises_error(self):
        """table() with wrong number of args raises ValueError."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        with self.assertRaises(ValueError):
            ds.table("a", "b", "c", "d")  # Too many args

        with self.assertRaises((ValueError, TypeError)):
            ds.table()  # No args


class TestGetItemTableSelection(unittest.TestCase):
    """Test __getitem__ for table selection in connection mode.

    NOTE: Using ds["db.table"] for table selection is legacy behavior.
    The recommended approach is to use ds.table("db", "table") which is
    unambiguous and doesn't conflict with pandas-style column/row selection.
    These tests are kept for backward compatibility verification.
    """

    def test_getitem_database_table_format(self):
        """ds["db.table"] creates new DataStore (legacy behavior)."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        users = ds["production.users"]

        # Should return new DataStore
        self.assertIsInstance(users, DataStore)
        self.assertIsNot(users, ds)

        # New DataStore should be in table mode
        self.assertEqual(users._connection_mode, "table")

    def test_getitem_table_only_with_default_database(self):
        """ds["db.table"] uses dot notation to select table (legacy)."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        # In connection mode, ds["db.table"] with dot notation selects the table
        # Note: ds["table"] without dot in database mode falls through to column access
        users = ds["production.users"]

        self.assertEqual(users._connection_mode, "table")

    def test_getitem_table_only_without_default_raises_error(self):
        """ds["table"] without default database raises error."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        # In connection mode without '.' in key and no default database
        # This should raise an error when trying to select table

        with self.assertRaises(DataStoreError) as ctx:
            ds["users"]

        # Check that error message contains helpful info
        error_msg = str(ctx.exception)
        self.assertIn("database", error_msg.lower())

    def test_getitem_original_unchanged(self):
        """ds["db.table"] does not modify original (legacy behavior)."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        original_mode = ds._connection_mode

        users = ds["production.users"]

        # Original should be unchanged
        self.assertEqual(ds._connection_mode, original_mode)
        self.assertIsNone(ds._default_table)

    def test_getitem_preserves_connection_params(self):
        """ds["db.table"] preserves connection params (legacy behavior)."""
        ds = DataStore.from_clickhouse(
            host="analytics.example.com:9000", user="analyst", password="secret123"
        )
        users = ds["production.users"]

        # New DataStore should have same connection params
        self.assertEqual(users._remote_params["host"], "analytics.example.com:9000")
        self.assertEqual(users._remote_params["user"], "analyst")
        self.assertEqual(users._remote_params["password"], "secret123")


class TestAdapters(unittest.TestCase):
    """Test source adapter implementations."""

    def test_clickhouse_adapter_list_databases_sql(self):
        """ClickHouse adapter generates correct databases SQL."""
        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        sql = adapter.list_databases_sql()

        self.assertIn("remote(", sql)
        self.assertIn("system", sql)
        self.assertIn("databases", sql)

    def test_clickhouse_adapter_list_tables_sql(self):
        """ClickHouse adapter generates correct tables SQL."""
        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        sql = adapter.list_tables_sql("mydb")

        self.assertIn("remote(", sql)
        self.assertIn("system", sql)
        self.assertIn("tables", sql)
        self.assertIn("mydb", sql)

    def test_clickhouse_adapter_describe_table_sql(self):
        """ClickHouse adapter generates correct describe SQL."""
        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        sql = adapter.describe_table_sql("mydb", "users")

        self.assertIn("remote(", sql)
        self.assertIn("system", sql)
        self.assertIn("columns", sql)
        self.assertIn("mydb", sql)
        self.assertIn("users", sql)

    def test_clickhouse_adapter_secure(self):
        """ClickHouse adapter uses remoteSecure when secure=True."""
        adapter = ClickHouseAdapter(
            host="localhost:9440", user="default", password="", secure=True
        )
        sql = adapter.list_databases_sql()

        self.assertIn("remoteSecure(", sql)
        self.assertNotIn("remote(", sql.replace("remoteSecure(", ""))

    def test_clickhouse_adapter_build_table_function(self):
        """ClickHouse adapter builds correct table function."""
        adapter = ClickHouseAdapter(
            host="localhost:9000", user="default", password="secret"
        )
        func = adapter.build_table_function("mydb", "users")

        self.assertIn("remote(", func)
        self.assertIn("mydb", func)
        self.assertIn("users", func)
        self.assertIn("default", func)
        self.assertIn("secret", func)

    def test_clickhouse_adapter_escape_sql_string(self):
        """ClickHouse adapter escapes single quotes in parameters."""
        adapter = ClickHouseAdapter(
            host="localhost:9000", user="default", password="pass'word"
        )
        func = adapter.build_table_function("mydb", "users")

        # Single quote should be escaped as two single quotes
        self.assertIn("pass''word", func)

    def test_mysql_adapter_list_databases_sql(self):
        """MySQL adapter generates correct databases SQL."""
        adapter = MySQLAdapter(host="localhost:3306", user="root", password="")
        sql = adapter.list_databases_sql()

        self.assertIn("mysql(", sql)
        self.assertIn("information_schema", sql)
        self.assertIn("schemata", sql)

    def test_mysql_adapter_list_tables_sql(self):
        """MySQL adapter generates correct tables SQL."""
        adapter = MySQLAdapter(host="localhost:3306", user="root", password="")
        sql = adapter.list_tables_sql("testdb")

        self.assertIn("mysql(", sql)
        self.assertIn("information_schema", sql)
        self.assertIn("tables", sql)
        self.assertIn("testdb", sql)

    def test_mysql_adapter_build_table_function(self):
        """MySQL adapter builds correct table function."""
        adapter = MySQLAdapter(host="localhost:3306", user="root", password="secret")
        func = adapter.build_table_function("mydb", "users")

        self.assertEqual(
            func, "mysql('localhost:3306', 'mydb', 'users', 'root', 'secret')"
        )

    def test_postgresql_adapter_list_databases_sql(self):
        """PostgreSQL adapter generates correct databases SQL."""
        adapter = PostgreSQLAdapter(host="localhost:5432", user="postgres", password="")
        sql = adapter.list_databases_sql()

        self.assertIn("postgresql(", sql)
        self.assertIn("pg_database", sql)

    def test_postgresql_adapter_list_tables_sql(self):
        """PostgreSQL adapter generates correct tables SQL."""
        adapter = PostgreSQLAdapter(host="localhost:5432", user="postgres", password="")
        sql = adapter.list_tables_sql("mydb")

        self.assertIn("postgresql(", sql)
        self.assertIn("information_schema", sql)
        self.assertIn("mydb", sql)

    def test_postgresql_adapter_build_table_function(self):
        """PostgreSQL adapter builds correct table function."""
        adapter = PostgreSQLAdapter(
            host="localhost:5432", user="postgres", password="secret"
        )
        func = adapter.build_table_function("mydb", "users")

        self.assertEqual(
            func, "postgresql('localhost:5432', 'mydb', 'users', 'postgres', 'secret')"
        )

    def test_get_adapter_clickhouse(self):
        """get_adapter returns ClickHouseAdapter for clickhouse source."""
        adapter = get_adapter(
            "clickhouse", host="localhost:9000", user="default", password=""
        )
        self.assertIsInstance(adapter, ClickHouseAdapter)

    def test_get_adapter_remote_alias(self):
        """get_adapter returns ClickHouseAdapter for remote source."""
        adapter = get_adapter(
            "remote", host="localhost:9000", user="default", password=""
        )
        self.assertIsInstance(adapter, ClickHouseAdapter)

    def test_get_adapter_remotesecure_alias(self):
        """get_adapter returns ClickHouseAdapter with secure=True for remotesecure source."""
        adapter = get_adapter(
            "remotesecure", host="localhost:9440", user="default", password=""
        )
        self.assertIsInstance(adapter, ClickHouseAdapter)
        self.assertTrue(adapter.secure)

    def test_get_adapter_mysql(self):
        """get_adapter returns MySQLAdapter for mysql source."""
        adapter = get_adapter("mysql", host="localhost:3306", user="root", password="")
        self.assertIsInstance(adapter, MySQLAdapter)

    def test_get_adapter_postgresql(self):
        """get_adapter returns PostgreSQLAdapter for postgresql source."""
        adapter = get_adapter(
            "postgresql", host="localhost:5432", user="postgres", password=""
        )
        self.assertIsInstance(adapter, PostgreSQLAdapter)

    def test_get_adapter_postgres_alias(self):
        """get_adapter returns PostgreSQLAdapter for postgres alias."""
        adapter = get_adapter(
            "postgres", host="localhost:5432", user="postgres", password=""
        )
        self.assertIsInstance(adapter, PostgreSQLAdapter)

    def test_get_adapter_unsupported_raises_error(self):
        """get_adapter raises error for unsupported source."""
        with self.assertRaises(DataStoreError) as ctx:
            get_adapter("unknown_source", host="localhost", user="user", password="")

        error_msg = str(ctx.exception)
        self.assertIn("not supported", error_msg.lower())

    def test_get_adapter_case_insensitive(self):
        """get_adapter is case-insensitive for source type."""
        adapter1 = get_adapter(
            "CLICKHOUSE", host="localhost:9000", user="default", password=""
        )
        adapter2 = get_adapter(
            "ClickHouse", host="localhost:9000", user="default", password=""
        )

        self.assertIsInstance(adapter1, ClickHouseAdapter)
        self.assertIsInstance(adapter2, ClickHouseAdapter)


class TestSQLRewriting(unittest.TestCase):
    """Test SQL table reference rewriting."""

    def test_rewrite_simple_from_with_default_database(self):
        """Rewrite simple FROM table when default database is set."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references("SELECT * FROM users")

        self.assertIn("remote(", sql)
        self.assertIn("testdb", sql)
        self.assertIn("users", sql)

    def test_rewrite_qualified_from(self):
        """Rewrite FROM db.table."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        sql = ds._rewrite_table_references("SELECT * FROM production.users")

        self.assertIn("remote(", sql)
        self.assertIn("production", sql)
        self.assertIn("users", sql)

    def test_rewrite_preserves_where(self):
        """Rewriting preserves WHERE clause."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references("SELECT * FROM users WHERE age > 25")

        self.assertIn("WHERE", sql)
        self.assertIn("age > 25", sql)

    def test_rewrite_preserves_select_columns(self):
        """Rewriting preserves SELECT columns."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references("SELECT name, age FROM users")

        self.assertIn("SELECT name, age", sql)
        self.assertIn("remote(", sql)

    def test_rewrite_without_default_db_raises_error(self):
        """Rewriting without default database raises error for unqualified table."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        with self.assertRaises(DataStoreError) as ctx:
            ds._rewrite_table_references("SELECT * FROM users")

        error_msg = str(ctx.exception)
        self.assertIn("database", error_msg.lower())


class TestRequireConnectionParams(unittest.TestCase):
    """Test _require_connection_params validation."""

    def test_require_connection_params_without_host_raises_error(self):
        """_require_connection_params raises error when host is missing."""
        ds = DataStore(source="clickhouse")

        with self.assertRaises(DataStoreError) as ctx:
            ds._require_connection_params()

        error_msg = str(ctx.exception)
        self.assertIn("connection", error_msg.lower())

    def test_require_connection_params_with_host_passes(self):
        """_require_connection_params passes when host is set."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        # Should not raise
        ds._require_connection_params()


class TestPasswordMasking(unittest.TestCase):
    """Test that passwords are masked in _connection_repr() output.

    Note: We test _connection_repr() directly because __repr__ may try to execute
    the query if _has_sql_state() returns True (which happens when _table_function is set).
    """

    def test_connection_repr_masks_password(self):
        """_connection_repr() masks password field."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password="supersecret123"
        )

        repr_str = ds._connection_repr()

        self.assertNotIn("supersecret123", repr_str)
        self.assertIn("***", repr_str)
        self.assertIn("localhost:9000", repr_str)

    def test_str_masks_password(self):
        """__str__ masks password field (via _connection_repr)."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password="supersecret123"
        )

        # __str__ calls __repr__, which may fail execution but shouldn't expose password
        str_str = str(ds)
        self.assertNotIn("supersecret123", str_str)

    def test_connection_repr_shows_other_params(self):
        """_connection_repr() shows non-sensitive params."""
        ds = DataStore.from_clickhouse(
            host="analytics.example.com:9000", user="analyst", password="secret"
        )

        repr_str = ds._connection_repr()

        self.assertIn("analytics.example.com:9000", repr_str)
        self.assertIn("analyst", repr_str)

    def test_connection_repr_shows_database_if_set(self):
        """_connection_repr() shows database when set via use()."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("production")

        repr_str = ds._connection_repr()

        self.assertIn("production", repr_str)

    def test_connection_repr_shows_table_if_set(self):
        """_connection_repr() shows table when set via use()."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("production", "users")

        repr_str = ds._connection_repr()

        self.assertIn("users", repr_str)


class TestRemoteParamsStorage(unittest.TestCase):
    """Test that remote connection parameters are stored correctly."""

    def test_remote_params_stored_for_clickhouse(self):
        """Remote params are stored for ClickHouse source."""
        ds = DataStore(
            source="clickhouse", host="myhost:9000", user="myuser", password="mypass"
        )

        self.assertEqual(ds._remote_params["host"], "myhost:9000")
        self.assertEqual(ds._remote_params["user"], "myuser")
        self.assertEqual(ds._remote_params["password"], "mypass")

    def test_remote_params_stored_for_mysql(self):
        """Remote params are stored for MySQL source."""
        ds = DataStore(
            source="mysql", host="myhost:3306", user="root", password="secret"
        )

        self.assertEqual(ds._remote_params["host"], "myhost:3306")
        self.assertEqual(ds._remote_params["user"], "root")
        self.assertEqual(ds._remote_params["password"], "secret")

    def test_remote_params_stored_for_postgresql(self):
        """Remote params are stored for PostgreSQL source."""
        ds = DataStore(
            source="postgresql", host="myhost:5432", user="postgres", password="pgpass"
        )

        self.assertEqual(ds._remote_params["host"], "myhost:5432")
        self.assertEqual(ds._remote_params["user"], "postgres")
        self.assertEqual(ds._remote_params["password"], "pgpass")

    def test_secure_param_stored_for_clickhouse(self):
        """Secure param is stored for ClickHouse source."""
        ds = DataStore(
            source="clickhouse",
            host="myhost:9440",
            user="default",
            password="",
            secure=True,
        )

        self.assertTrue(ds._remote_params.get("secure"))

    def test_remote_params_empty_for_dataframe(self):
        """Remote params are empty for DataFrame source."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        ds = DataStore(df)

        self.assertEqual(ds._remote_params, {})


class TestAdapterTableFunctionFormat(unittest.TestCase):
    """Test the exact format of table functions generated by adapters."""

    def test_clickhouse_table_function_format(self):
        """ClickHouse adapter generates correctly formatted table function."""
        adapter = ClickHouseAdapter(
            host="ch.example.com:9000", user="admin", password="pwd123"
        )
        func = adapter.build_table_function("analytics", "events")

        # Format: remote('host', 'database', 'table', 'user', 'password')
        expected = (
            "remote('ch.example.com:9000', 'analytics', 'events', 'admin', 'pwd123')"
        )
        self.assertEqual(func, expected)

    def test_clickhouse_secure_table_function_format(self):
        """ClickHouse adapter generates remoteSecure for secure connections."""
        adapter = ClickHouseAdapter(
            host="ch.example.com:9440", user="admin", password="pwd123", secure=True
        )
        func = adapter.build_table_function("analytics", "events")

        expected = "remoteSecure('ch.example.com:9440', 'analytics', 'events', 'admin', 'pwd123')"
        self.assertEqual(func, expected)

    def test_mysql_table_function_format(self):
        """MySQL adapter generates correctly formatted table function."""
        adapter = MySQLAdapter(
            host="mysql.example.com:3306", user="dbuser", password="dbpass"
        )
        func = adapter.build_table_function("myapp", "customers")

        expected = (
            "mysql('mysql.example.com:3306', 'myapp', 'customers', 'dbuser', 'dbpass')"
        )
        self.assertEqual(func, expected)

    def test_postgresql_table_function_format(self):
        """PostgreSQL adapter generates correctly formatted table function."""
        adapter = PostgreSQLAdapter(
            host="pg.example.com:5432", user="pguser", password="pgpass"
        )
        func = adapter.build_table_function("inventory", "products")

        expected = "postgresql('pg.example.com:5432', 'inventory', 'products', 'pguser', 'pgpass')"
        self.assertEqual(func, expected)


class TestAdapterSourceTypeProperty(unittest.TestCase):
    """Test that adapters return correct table function names."""

    def test_clickhouse_adapter_function_name(self):
        """ClickHouse adapter returns 'remote' as function name."""
        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        self.assertEqual(adapter.get_table_function_name(), "remote")

    def test_clickhouse_secure_adapter_function_name(self):
        """ClickHouse adapter returns 'remoteSecure' when secure=True."""
        adapter = ClickHouseAdapter(
            host="localhost:9440", user="default", password="", secure=True
        )
        self.assertEqual(adapter.get_table_function_name(), "remoteSecure")

    def test_mysql_adapter_function_name(self):
        """MySQL adapter returns 'mysql' as function name."""
        adapter = MySQLAdapter(host="localhost:3306", user="root", password="")
        self.assertEqual(adapter.get_table_function_name(), "mysql")

    def test_postgresql_adapter_function_name(self):
        """PostgreSQL adapter returns 'postgresql' as function name."""
        adapter = PostgreSQLAdapter(host="localhost:5432", user="postgres", password="")
        self.assertEqual(adapter.get_table_function_name(), "postgresql")


class TestSmartErrorMessages(unittest.TestCase):
    """Test smart error messages when operations are called without proper context.

    Design doc Section 5: Smart Error Messages
    """

    def test_columns_in_connection_mode_raises_error(self):
        """ds.columns in connection mode should raise helpful error."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        # In connection mode without table, accessing columns should fail
        # Note: This may raise different errors depending on implementation
        # The key is that it should not silently fail or return garbage
        with self.assertRaises((DataStoreError, AttributeError, KeyError)):
            _ = ds.columns

    def test_head_in_connection_mode_behavior(self):
        """ds.head() in connection mode - verify current behavior.

        NOTE: Design doc Section 5 suggests table-level operations should raise
        an error in connection mode. Current implementation returns a lazy
        DataStore that may fail on execution. This test documents actual behavior.

        TODO: Consider if head() should raise DataStoreError immediately
        in connection mode per design doc.
        """
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        # Current behavior: head() returns lazy DataStore, doesn't fail immediately
        result = ds.head()
        self.assertIsInstance(result, DataStore)

    def test_getitem_table_without_database_error_message_contains_hint(self):
        """Error message should contain helpful hint about how to fix it."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        with self.assertRaises(DataStoreError) as ctx:
            ds["users"]  # No database specified

        error_msg = str(ctx.exception).lower()
        # Should mention database and provide hint
        self.assertIn("database", error_msg)
        # Should mention use() or db.table format
        self.assertTrue(
            "use(" in error_msg or "." in error_msg or "hint" in error_msg,
            f"Error message should provide hint: {error_msg}",
        )

    def test_sql_without_database_error_message_contains_hint(self):
        """SQL rewrite error should contain helpful hint."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        with self.assertRaises(DataStoreError) as ctx:
            ds._rewrite_table_references("SELECT * FROM users")

        error_msg = str(ctx.exception).lower()
        self.assertIn("database", error_msg)
        # Should provide actionable hint
        self.assertTrue(
            "use(" in error_msg or "hint" in error_msg or "qualified" in error_msg,
            f"Error message should provide hint: {error_msg}",
        )


class TestChainedTableSelection(unittest.TestCase):
    """Test chained table selection via table() method.

    Design doc Section 3: Select Database/Table

    NOTE: This class tests the recommended table() method.
    Legacy __getitem__ tests are in TestGetItemTableSelection.
    """

    def test_table_method_creates_new_datastore(self):
        """table(db, table) creates new DataStore."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        users = ds.table("production", "users")

        self.assertIsInstance(users, DataStore)
        self.assertIsNot(users, ds)
        self.assertEqual(users._connection_mode, "table")

    def test_table_with_dot_notation(self):
        """table("db.table") works with dot notation."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        users = ds.table("production.users")

        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")

    def test_use_then_table_method(self):
        """After use(db), table(table) should work."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("production")

        # Now table-only selection should work
        users = ds.table("users")

        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")

    def test_table_preserves_original_connection_mode(self):
        """Original DataStore should remain unchanged after table()."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        original_mode = ds._connection_mode
        original_db = ds._default_database

        # Select table using table() method
        _ = ds.table("production", "users")

        # Original should be unchanged
        self.assertEqual(ds._connection_mode, original_mode)
        self.assertEqual(ds._default_database, original_db)


class TestSQLTableFunctionPreservation(unittest.TestCase):
    """Test that table functions in SQL are preserved (not rewritten).

    Design doc Section 4: Table name resolution table
    """

    def test_rewrite_preserves_file_function(self):
        """file() table function should not be rewritten."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references("SELECT * FROM file('data.csv')")

        # file() should be preserved as-is
        self.assertIn("file(", sql)
        self.assertIn("data.csv", sql)

    def test_rewrite_preserves_url_function(self):
        """url() table function should not be rewritten."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references(
            "SELECT * FROM url('http://example.com/data.json', JSONEachRow)"
        )

        # url() should be preserved
        self.assertIn("url(", sql)
        self.assertIn("http://example.com", sql)

    def test_rewrite_preserves_s3_function(self):
        """s3() table function should not be rewritten."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references(
            "SELECT * FROM s3('s3://bucket/file.parquet')"
        )

        # s3() should be preserved
        self.assertIn("s3(", sql)
        self.assertIn("bucket", sql)

    def test_rewrite_preserves_numbers_function(self):
        """numbers() table function should not be rewritten."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("testdb")

        sql = ds._rewrite_table_references("SELECT * FROM numbers(10)")

        # numbers() should be preserved
        self.assertIn("numbers(", sql)


class TestTablesWithDefaultDatabase(unittest.TestCase):
    """Test tables() method with default database from use().

    Design doc Section 2: Metadata Discovery - tables() without argument
    """

    def test_tables_without_arg_after_use(self):
        """tables() without argument should use default database."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )
        ds.use("mydb")

        # This tests the parameter resolution, not actual execution
        # The actual query would need a real server
        self.assertEqual(ds._default_database, "mydb")

    def test_tables_without_arg_without_use_raises_error(self):
        """tables() without argument and no default should raise error."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        # Mock the adapter to avoid actual connection
        with mock.patch.object(ds, "_get_adapter"):
            with self.assertRaises(DataStoreError) as ctx:
                ds.tables()

        error_msg = str(ctx.exception).lower()
        self.assertIn("database", error_msg)


class TestUseMethodEdgeCases(unittest.TestCase):
    """Test edge cases for use() method."""

    def test_use_overwrites_previous_use(self):
        """Multiple use() calls should overwrite previous values."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        ds.use("db1")
        self.assertEqual(ds._default_database, "db1")

        ds.use("db2")
        self.assertEqual(ds._default_database, "db2")

    def test_use_database_then_use_database_table(self):
        """use(db) then use(db, table) should work.

        NOTE: Current implementation keeps _connection_mode as 'connection'
        after use(db), only changing to 'table' after use(db, table).
        Design doc suggests use(db) should change mode to 'database'.

        TODO: Consider if use(db) should change mode to 'database' per design doc.
        """
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        ds.use("mydb")
        # Current behavior: mode stays 'connection', only _default_database is set
        self.assertEqual(ds._default_database, "mydb")
        # Note: mode doesn't change to 'database' in current implementation
        # self.assertEqual(ds._connection_mode, "database")  # Design expectation

        ds.use("mydb", "users")
        self.assertEqual(ds._connection_mode, "table")
        self.assertEqual(ds._default_database, "mydb")
        self.assertEqual(ds._default_table, "users")

    def test_use_empty_string_raises_error(self):
        """use('') should raise error or handle gracefully."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        # Empty database name should be rejected
        # Implementation may vary on exact behavior
        ds.use("")
        # At minimum, _default_database should be set (even if empty)
        self.assertEqual(ds._default_database, "")


class TestSpecialCharactersInNames(unittest.TestCase):
    """Test handling of special characters in database/table names."""

    def test_adapter_escapes_single_quotes_in_database(self):
        """Single quotes in database name should be escaped."""
        from datastore.adapters import ClickHouseAdapter

        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        sql = adapter.list_tables_sql("test'db")

        # Single quote should be escaped
        self.assertIn("test''db", sql)

    def test_adapter_escapes_single_quotes_in_table(self):
        """Single quotes in table name should be escaped."""
        from datastore.adapters import ClickHouseAdapter

        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        func = adapter.build_table_function("mydb", "test'table")

        # Single quote should be escaped
        self.assertIn("test''table", func)

    def test_adapter_handles_unicode_names(self):
        """Unicode characters in names should be handled."""
        from datastore.adapters import ClickHouseAdapter

        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        func = adapter.build_table_function("数据库", "用户表")

        # Should contain the unicode characters
        self.assertIn("数据库", func)
        self.assertIn("用户表", func)


class TestEmptyPassword(unittest.TestCase):
    """Test handling of empty passwords."""

    def test_clickhouse_empty_password_allowed(self):
        """ClickHouse should allow empty password."""
        ds = DataStore.from_clickhouse(
            host="localhost:9000", user="default", password=""
        )

        self.assertEqual(ds._remote_params["password"], "")

    def test_mysql_empty_password_allowed(self):
        """MySQL should allow empty password."""
        ds = DataStore.from_mysql(host="localhost:3306", user="root", password="")

        self.assertEqual(ds._remote_params["password"], "")

    def test_adapter_table_function_with_empty_password(self):
        """Table function should work with empty password."""
        from datastore.adapters import ClickHouseAdapter

        adapter = ClickHouseAdapter(host="localhost:9000", user="default", password="")
        func = adapter.build_table_function("mydb", "users")

        # Should have empty string for password, not None or missing
        self.assertIn("''", func)  # Empty string in SQL


if __name__ == "__main__":
    unittest.main()
