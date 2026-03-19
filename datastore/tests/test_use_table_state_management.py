"""
Tests for use() -> table() state management correctness (CH-3).

Verifies that connection/database/table mode transitions behave correctly,
that table() creates independent DataStore instances, and that use()'s
mutable semantics don't cause unintended shared-state bugs.
"""

import unittest

from datastore import DataStore
from datastore.exceptions import DataStoreError


def _make_ch_connection(**kwargs):
    """Helper to create a ClickHouse connection-level DataStore."""
    defaults = dict(host="localhost:9000", user="default", password="")
    defaults.update(kwargs)
    return DataStore.from_clickhouse(**defaults)


def _make_pg_connection(**kwargs):
    """Helper to create a PostgreSQL connection-level DataStore."""
    defaults = dict(host="localhost:5432", user="postgres", password="")
    defaults.update(kwargs)
    return DataStore.from_postgresql(**defaults)


class TestUseTableChaining(unittest.TestCase):
    """ds.use("db").table("tbl") chain creates correct new DataStore."""

    def test_use_then_table_creates_new_ds(self):
        ds = _make_ch_connection()
        tbl = ds.use("mydb").table("users")

        self.assertIsNot(tbl, ds)
        self.assertIsInstance(tbl, DataStore)
        self.assertEqual(tbl._connection_mode, "table")

    def test_use_then_table_new_ds_has_table_name(self):
        ds = _make_ch_connection()
        tbl = ds.use("mydb").table("users")

        self.assertEqual(tbl.table_name, "users")

    def test_use_then_table_new_ds_has_table_function(self):
        ds = _make_ch_connection()
        tbl = ds.use("mydb").table("users")

        self.assertIsNotNone(tbl._table_function)
        sql = tbl._table_function.to_sql()
        self.assertIn("users", sql)
        self.assertIn("mydb", sql)

    def test_use_then_table_preserves_connection_params(self):
        ds = _make_ch_connection(
            host="analytics.example.com:9000", user="analyst", password="s3cret"
        )
        tbl = ds.use("prod").table("events")

        self.assertEqual(tbl._remote_params["host"], "analytics.example.com:9000")
        self.assertEqual(tbl._remote_params["user"], "analyst")
        self.assertEqual(tbl._remote_params["password"], "s3cret")

    def test_use_then_table_original_mode_is_database(self):
        """use("db") mutates ds to database mode; table() doesn't change ds."""
        ds = _make_ch_connection()
        ds.use("mydb")
        _ = ds.table("users")

        # use() mutated ds to database mode, table() didn't change it further
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "mydb")


class TestGetItemTableSelection(unittest.TestCase):
    """ds.use("db")["tbl"] selects table via __getitem__."""

    def test_getitem_after_use_returns_new_ds(self):
        ds = _make_ch_connection()
        ds.use("mydb")
        tbl = ds["users"]

        self.assertIsNot(tbl, ds)
        self.assertIsInstance(tbl, DataStore)
        self.assertEqual(tbl._connection_mode, "table")

    def test_getitem_after_use_has_correct_table(self):
        ds = _make_ch_connection()
        ds.use("mydb")
        tbl = ds["users"]

        self.assertEqual(tbl.table_name, "users")

    def test_getitem_dot_notation_in_connection_mode(self):
        ds = _make_ch_connection()
        tbl = ds["prod.orders"]

        self.assertIsNot(tbl, ds)
        self.assertEqual(tbl._connection_mode, "table")
        self.assertEqual(tbl.table_name, "orders")

    def test_getitem_does_not_modify_original(self):
        ds = _make_ch_connection()
        ds.use("mydb")
        original_mode = ds._connection_mode
        original_db = ds._default_database
        _ = ds["users"]

        self.assertEqual(ds._connection_mode, original_mode)
        self.assertEqual(ds._default_database, original_db)
        self.assertIsNone(ds._default_table)


class TestUseTwoArgsBindsTable(unittest.TestCase):
    """ds.use("db", "tbl") sets both database and table, binds table context."""

    def test_use_two_args_sets_connection_mode_to_table(self):
        ds = _make_ch_connection()
        ds.use("mydb", "users")

        self.assertEqual(ds._connection_mode, "table")

    def test_use_two_args_sets_default_database_and_table(self):
        ds = _make_ch_connection()
        ds.use("mydb", "users")

        self.assertEqual(ds._default_database, "mydb")
        self.assertEqual(ds._default_table, "users")

    def test_use_two_args_binds_table_name(self):
        ds = _make_ch_connection()
        ds.use("mydb", "users")

        self.assertEqual(ds.table_name, "users")

    def test_use_two_args_creates_table_function(self):
        ds = _make_ch_connection()
        ds.use("mydb", "users")

        self.assertIsNotNone(ds._table_function)
        sql = ds._table_function.to_sql()
        self.assertIn("mydb", sql)
        self.assertIn("users", sql)

    def test_use_two_args_has_sql_state(self):
        ds = _make_ch_connection()
        ds.use("mydb", "users")

        self.assertTrue(ds._has_sql_state())

    def test_use_two_args_subsequent_table_overrides(self):
        """After use(db, tbl), calling table() returns an independent DS."""
        ds = _make_ch_connection()
        ds.use("mydb", "users")

        orders = ds.table("mydb.orders")
        # ds still points to users
        self.assertEqual(ds.table_name, "users")
        self.assertEqual(ds._default_table, "users")
        # orders is a new DS pointing to orders
        self.assertEqual(orders.table_name, "orders")


class TestTableDotNotation(unittest.TestCase):
    """ds.table("db.tbl") correctly parses dot notation."""

    def test_dot_notation_parses_correctly(self):
        ds = _make_ch_connection()
        tbl = ds.table("prod.events")

        self.assertEqual(tbl._connection_mode, "table")
        self.assertEqual(tbl.table_name, "events")

    def test_dot_notation_includes_db_in_table_function(self):
        ds = _make_ch_connection()
        tbl = ds.table("prod.events")

        sql = tbl._table_function.to_sql()
        self.assertIn("prod", sql)
        self.assertIn("events", sql)

    def test_three_part_dot_notation(self):
        ds = _make_pg_connection()
        tbl = ds.table("public.mydb.users")

        self.assertEqual(tbl._connection_mode, "table")
        self.assertEqual(tbl._default_schema, "public")
        self.assertEqual(tbl.table_name, "users")

    def test_dot_notation_does_not_modify_original(self):
        ds = _make_ch_connection()
        _ = ds.table("prod.events")

        self.assertEqual(ds._connection_mode, "connection")
        self.assertIsNone(ds._default_table)


class TestTableTwoArgs(unittest.TestCase):
    """ds.table("db", "tbl") with two arguments works correctly."""

    def test_two_args_creates_new_ds(self):
        ds = _make_ch_connection()
        tbl = ds.table("staging", "orders")

        self.assertIsNot(tbl, ds)
        self.assertEqual(tbl._connection_mode, "table")
        self.assertEqual(tbl.table_name, "orders")

    def test_two_args_table_function_references_both(self):
        ds = _make_ch_connection()
        tbl = ds.table("staging", "orders")

        sql = tbl._table_function.to_sql()
        self.assertIn("staging", sql)
        self.assertIn("orders", sql)


class TestMultipleUseDoesNotAffectPriorTables(unittest.TestCase):
    """Switching database with use() does not affect previously created tables."""

    def test_table_from_first_db_unaffected_by_second_use(self):
        ds = _make_ch_connection()

        ds.use("db1")
        tbl1 = ds.table("users")

        ds.use("db2")
        tbl2 = ds.table("orders")

        # tbl1 should still reference db1.users
        sql1 = tbl1._table_function.to_sql()
        self.assertIn("db1", sql1)
        self.assertIn("users", sql1)

        # tbl2 should reference db2.orders
        sql2 = tbl2._table_function.to_sql()
        self.assertIn("db2", sql2)
        self.assertIn("orders", sql2)

    def test_table_objects_are_independent(self):
        ds = _make_ch_connection()
        ds.use("db1")
        tbl1 = ds.table("users")

        ds.use("db2")
        tbl2 = ds.table("orders")

        # Verify they are distinct objects
        self.assertIsNot(tbl1, tbl2)
        self.assertIsNot(tbl1, ds)
        self.assertIsNot(tbl2, ds)

    def test_use_mutation_does_not_leak_to_table_instances(self):
        """table() creates brand new DS; later use() on parent doesn't affect it."""
        ds = _make_ch_connection()
        ds.use("original_db")
        tbl = ds.table("users")

        # Now mutate ds
        ds.use("new_db", "different_table")

        # tbl should be unaffected
        self.assertEqual(tbl.table_name, "users")
        sql = tbl._table_function.to_sql()
        self.assertIn("original_db", sql)
        self.assertNotIn("new_db", sql)
        self.assertNotIn("different_table", sql)


class TestTableIndependence(unittest.TestCase):
    """ds.table("tbl1") and ds.table("tbl2") are fully independent."""

    def test_two_tables_same_db_independent(self):
        ds = _make_ch_connection()
        ds.use("mydb")

        users = ds.table("users")
        orders = ds.table("orders")

        self.assertIsNot(users, orders)
        self.assertEqual(users.table_name, "users")
        self.assertEqual(orders.table_name, "orders")

    def test_independent_tables_have_own_table_functions(self):
        ds = _make_ch_connection()
        ds.use("mydb")

        users = ds.table("users")
        orders = ds.table("orders")

        users_sql = users._table_function.to_sql()
        orders_sql = orders._table_function.to_sql()

        self.assertIn("users", users_sql)
        self.assertNotIn("orders", users_sql)
        self.assertIn("orders", orders_sql)
        self.assertNotIn("users", orders_sql)

    def test_independent_tables_share_connection_params(self):
        ds = _make_ch_connection(host="host1:9000", user="u1", password="p1")
        ds.use("mydb")

        users = ds.table("users")
        orders = ds.table("orders")

        self.assertEqual(users._remote_params["host"], "host1:9000")
        self.assertEqual(orders._remote_params["host"], "host1:9000")

    def test_independent_tables_do_not_share_sql_state(self):
        ds = _make_ch_connection()
        ds.use("mydb")

        users = ds.table("users")
        orders = ds.table("orders")

        # They should have independent _lazy_ops
        self.assertIsNot(users._lazy_ops, orders._lazy_ops)

    def test_independent_tables_do_not_share_cached_results(self):
        ds = _make_ch_connection()
        ds.use("mydb")

        users = ds.table("users")
        orders = ds.table("orders")

        # Setting cached result on one should not affect the other
        users._cached_result = "fake_users"
        self.assertIsNone(orders._cached_result)


class TestUseMutableSemantics(unittest.TestCase):
    """use() returns self (mutable). ds2 = ds.use("db") -> ds and ds2 are same object."""

    def test_use_returns_self(self):
        ds = _make_ch_connection()
        ds2 = ds.use("mydb")

        self.assertIs(ds, ds2)

    def test_use_mutation_visible_through_both_references(self):
        ds = _make_ch_connection()
        ds2 = ds.use("db1")

        # Both references see the same state
        self.assertEqual(ds._default_database, "db1")
        self.assertEqual(ds2._default_database, "db1")

        # Mutating through ds2 is visible on ds
        ds2.use("db2")
        self.assertEqual(ds._default_database, "db2")

    def test_use_chaining_all_same_object(self):
        ds = _make_ch_connection()
        ds2 = ds.use("db1")
        ds3 = ds2.use("db2")

        self.assertIs(ds, ds2)
        self.assertIs(ds2, ds3)
        self.assertEqual(ds._default_database, "db2")

    def test_use_mutation_risk_shared_references(self):
        """Demonstrates the risk: two references to same DS after use()."""
        ds = _make_ch_connection()

        # User might think ds2 is independent
        ds2 = ds.use("db1")
        # But mutating ds changes ds2 too
        ds.use("db2")
        self.assertEqual(ds2._default_database, "db2")  # Not db1!

    def test_table_creates_independent_copy_unlike_use(self):
        """table() returns a new DS, so it's safe from use() mutations."""
        ds = _make_ch_connection()
        ds.use("db1")
        tbl = ds.table("users")  # New independent DS

        ds.use("db2")

        # tbl is unaffected by ds.use("db2")
        sql = tbl._table_function.to_sql()
        self.assertIn("db1", sql)


class TestConnectionModeTransitions(unittest.TestCase):
    """Verify correct mode transitions across all paths."""

    def test_connection_to_database_via_use(self):
        ds = _make_ch_connection()
        self.assertEqual(ds._connection_mode, "connection")

        ds.use("mydb")
        self.assertEqual(ds._connection_mode, "database")

    def test_connection_to_table_via_use_two_args(self):
        ds = _make_ch_connection()
        self.assertEqual(ds._connection_mode, "connection")

        ds.use("mydb", "users")
        self.assertEqual(ds._connection_mode, "table")

    def test_database_to_table_via_use_two_args(self):
        ds = _make_ch_connection()
        ds.use("db1")
        self.assertEqual(ds._connection_mode, "database")

        ds.use("db1", "users")
        self.assertEqual(ds._connection_mode, "table")

    def test_table_method_always_returns_table_mode(self):
        ds = _make_ch_connection()
        tbl = ds.table("prod.events")
        self.assertEqual(tbl._connection_mode, "table")

    def test_getitem_in_database_mode_returns_table_mode(self):
        ds = _make_ch_connection()
        ds.use("mydb")
        tbl = ds["users"]
        self.assertEqual(tbl._connection_mode, "table")

    def test_use_database_method_transitions_to_database_mode(self):
        ds = _make_ch_connection()
        self.assertEqual(ds._connection_mode, "connection")

        ds.use_database("mydb")
        self.assertEqual(ds._connection_mode, "database")

    def test_multiple_use_database_stays_in_database_mode(self):
        ds = _make_ch_connection()
        ds.use_database("db1")
        ds.use_database("db2")
        self.assertEqual(ds._connection_mode, "database")
        self.assertEqual(ds._default_database, "db2")


class TestUseOverwrite(unittest.TestCase):
    """Multiple use() calls correctly overwrite previous state."""

    def test_use_overwrites_database(self):
        ds = _make_ch_connection()
        ds.use("db1")
        ds.use("db2")
        self.assertEqual(ds._default_database, "db2")

    def test_use_two_args_overwrites_table(self):
        ds = _make_ch_connection()
        ds.use("db1", "tbl1")
        ds.use("db2", "tbl2")

        self.assertEqual(ds._default_database, "db2")
        self.assertEqual(ds._default_table, "tbl2")
        self.assertEqual(ds.table_name, "tbl2")

    def test_use_db_after_use_db_table_keeps_table_mode(self):
        """use(db) after use(db, table) keeps _connection_mode as 'table'."""
        ds = _make_ch_connection()
        ds.use("db1", "tbl1")
        self.assertEqual(ds._connection_mode, "table")

        # use(db) only sets _default_database, doesn't downgrade mode
        ds.use("db2")
        self.assertEqual(ds._connection_mode, "table")

    def test_use_database_after_use_table_does_not_downgrade(self):
        """use_database() after mode='table' keeps table mode."""
        ds = _make_ch_connection()
        ds.use("db1", "tbl1")
        self.assertEqual(ds._connection_mode, "table")

        ds.use_database("db2")
        self.assertEqual(ds._connection_mode, "table")


class TestThreeArgUse(unittest.TestCase):
    """use(schema, database, table) for PostgreSQL-style schemas."""

    def test_three_args_sets_all_fields(self):
        ds = _make_pg_connection()
        ds.use("public", "mydb", "users")

        self.assertEqual(ds._default_schema, "public")
        self.assertEqual(ds._default_database, "mydb")
        self.assertEqual(ds._default_table, "users")

    def test_three_args_mode_is_table(self):
        ds = _make_pg_connection()
        ds.use("public", "mydb", "users")

        self.assertEqual(ds._connection_mode, "table")

    def test_three_args_binds_table(self):
        ds = _make_pg_connection()
        ds.use("public", "mydb", "users")

        self.assertEqual(ds.table_name, "users")
        self.assertIsNotNone(ds._table_function)


class TestTableThreeArgs(unittest.TestCase):
    """table(schema, database, table) for PostgreSQL-style schemas."""

    def test_three_args_creates_new_ds(self):
        ds = _make_pg_connection()
        tbl = ds.table("public", "mydb", "users")

        self.assertIsNot(tbl, ds)
        self.assertEqual(tbl._connection_mode, "table")
        self.assertEqual(tbl._default_schema, "public")

    def test_three_args_table_function(self):
        ds = _make_pg_connection()
        tbl = ds.table("public", "mydb", "users")

        self.assertIsNotNone(tbl._table_function)
        sql = tbl._table_function.to_sql()
        self.assertIn("users", sql)


class TestErrorPaths(unittest.TestCase):
    """Error cases for use() and table()."""

    def test_table_without_database_raises_error(self):
        ds = _make_ch_connection()
        with self.assertRaises(DataStoreError):
            ds.table("users")  # No database set

    def test_use_no_args_raises(self):
        ds = _make_ch_connection()
        with self.assertRaises(ValueError):
            ds.use()

    def test_use_four_args_raises(self):
        ds = _make_ch_connection()
        with self.assertRaises(ValueError):
            ds.use("a", "b", "c", "d")

    def test_table_four_args_raises(self):
        ds = _make_ch_connection()
        with self.assertRaises(ValueError):
            ds.table("a", "b", "c", "d")

    def test_table_no_args_raises(self):
        ds = _make_ch_connection()
        with self.assertRaises((ValueError, TypeError)):
            ds.table()

    def test_getitem_in_connection_mode_without_dot_raises(self):
        """In connection mode, ds["users"] without db should raise."""
        ds = _make_ch_connection()
        with self.assertRaises(DataStoreError):
            ds["users"]  # connection mode, no dot, no default db


class TestComplexWorkflows(unittest.TestCase):
    """Multi-step workflows that combine use(), table(), __getitem__."""

    def test_use_then_multiple_tables_independent(self):
        """Create multiple table instances from same connection, all independent."""
        ds = _make_ch_connection()
        ds.use("analytics")

        users = ds.table("users")
        events = ds.table("events")
        orders = ds.table("orders")

        # All are independent
        self.assertEqual(users.table_name, "users")
        self.assertEqual(events.table_name, "events")
        self.assertEqual(orders.table_name, "orders")

        # All reference analytics database
        for tbl in [users, events, orders]:
            sql = tbl._table_function.to_sql()
            self.assertIn("analytics", sql)

    def test_switch_database_create_tables_from_each(self):
        ds = _make_ch_connection()

        ds.use("db1")
        t1 = ds.table("users")

        ds.use("db2")
        t2 = ds.table("users")

        # Both named users but different databases
        self.assertEqual(t1.table_name, "users")
        self.assertEqual(t2.table_name, "users")

        t1_sql = t1._table_function.to_sql()
        t2_sql = t2._table_function.to_sql()
        self.assertIn("db1", t1_sql)
        self.assertIn("db2", t2_sql)
        self.assertNotIn("db2", t1_sql)
        self.assertNotIn("db1", t2_sql)

    def test_use_database_then_getitem_table_selection(self):
        """use_database() followed by __getitem__ table selection."""
        ds = _make_ch_connection()
        ds.use_database("production")

        users = ds["users"]
        self.assertIsInstance(users, DataStore)
        self.assertEqual(users._connection_mode, "table")
        self.assertEqual(users.table_name, "users")

    def test_chained_use_table_head_pattern(self):
        """ds.use("db").table("tbl") pattern produces valid SQL state."""
        ds = _make_ch_connection()
        tbl = ds.use("mydb").table("events")

        # Should have valid SQL state for execution
        self.assertTrue(tbl._has_sql_state())
        self.assertIsNotNone(tbl._table_function)
        self.assertEqual(tbl.table_name, "events")

    def test_table_from_database_mode_ds(self):
        """DataStore created with database= parameter, then table()."""
        ds = DataStore(
            source="clickhouse",
            host="localhost:9000",
            database="mydb",
            user="default",
            password="",
        )
        self.assertEqual(ds._connection_mode, "database")

        tbl = ds.table("users")
        self.assertEqual(tbl._connection_mode, "table")
        self.assertEqual(tbl.table_name, "users")


if __name__ == "__main__":
    unittest.main()
