"""
Tests for INSERT, UPDATE, DELETE operations (ClickHouse style)
"""

import unittest
from datastore import DataStore
from datastore.expressions import Field
from datastore.functions import Count


class TestInsert(unittest.TestCase):
    """Test INSERT operations."""

    def test_insert_with_values_simple(self):
        """Test simple INSERT with VALUES."""
        ds = DataStore(table="users")
        query = ds.insert_into('id', 'name', 'age').insert_values(1, 'Alice', 25)

        expected = 'INSERT INTO "users" ("id", "name", "age") VALUES (1, \'Alice\', 25)'
        self.assertEqual(expected, query.to_sql())

    def test_insert_with_values_multiple_rows(self):
        """Test INSERT with multiple rows."""
        ds = DataStore(table="users")
        query = ds.insert_into('id', 'name', 'age').insert_values((1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35))

        expected = 'INSERT INTO "users" ("id", "name", "age") VALUES (1, \'Alice\', 25), (2, \'Bob\', 30), (3, \'Charlie\', 35)'
        self.assertEqual(expected, query.to_sql())

    def test_insert_with_values_chained(self):
        """Test INSERT with chained values() calls."""
        ds = DataStore(table="users")
        query = ds.insert_into('id', 'name').insert_values(1, 'Alice').insert_values(2, 'Bob')

        expected = 'INSERT INTO "users" ("id", "name") VALUES (1, \'Alice\'), (2, \'Bob\')'
        self.assertEqual(expected, query.to_sql())

    def test_insert_with_select(self):
        """Test INSERT with SELECT subquery."""
        ds_target = DataStore(table="users_backup")
        ds_source = DataStore(table="users")

        query = ds_target.insert_into('id', 'name', 'age').select_from(
            ds_source.select('id', 'name', 'age').filter(ds_source.active == True)
        )

        expected = 'INSERT INTO "users_backup" ("id", "name", "age") SELECT "id", "name", "age" FROM "users" WHERE "active" = TRUE'
        self.assertEqual(expected, query.to_sql())

    def test_insert_with_null_values(self):
        """Test INSERT with NULL values."""
        ds = DataStore(table="users")
        query = ds.insert_into('id', 'name', 'email').insert_values(1, 'Alice', None)

        expected = 'INSERT INTO "users" ("id", "name", "email") VALUES (1, \'Alice\', NULL)'
        self.assertEqual(expected, query.to_sql())

    def test_insert_with_boolean_values(self):
        """Test INSERT with boolean values."""
        ds = DataStore(table="users")
        query = ds.insert_into('id', 'name', 'active').insert_values(1, 'Alice', True)

        expected = 'INSERT INTO "users" ("id", "name", "active") VALUES (1, \'Alice\', 1)'
        self.assertEqual(expected, query.to_sql())


class TestUpdate(unittest.TestCase):
    """Test UPDATE operations (ClickHouse ALTER TABLE ... UPDATE style)."""

    def test_update_simple(self):
        """Test simple UPDATE."""
        ds = DataStore(table="users")
        query = ds.update_set(age=26).filter(ds.id == 1)

        expected = 'ALTER TABLE "users" UPDATE "age"=26 WHERE "id" = 1'
        self.assertEqual(expected, query.to_sql())

    def test_update_multiple_fields(self):
        """Test UPDATE with multiple fields."""
        ds = DataStore(table="users")
        query = ds.update_set(age=26, city='NYC', active=True).filter(ds.id == 1)

        # Note: order of fields may vary due to dict iteration
        sql = query.to_sql()
        self.assertIn('ALTER TABLE "users" UPDATE', sql)
        self.assertIn('"age"=26', sql)
        self.assertIn('"city"=\'NYC\'', sql)
        self.assertIn('"active"=1', sql)
        self.assertIn('WHERE "id" = 1', sql)

    def test_update_with_string_escaping(self):
        """Test UPDATE with string value that needs escaping."""
        ds = DataStore(table="users")
        query = ds.update_set(name="O'Brien").filter(ds.id == 1)

        expected = 'ALTER TABLE "users" UPDATE "name"=\'O\'\'Brien\' WHERE "id" = 1'
        self.assertEqual(expected, query.to_sql())

    def test_update_with_null(self):
        """Test UPDATE with NULL value."""
        ds = DataStore(table="users")
        query = ds.update_set(email=None).filter(ds.id == 1)

        expected = 'ALTER TABLE "users" UPDATE "email"=NULL WHERE "id" = 1'
        self.assertEqual(expected, query.to_sql())

    def test_update_with_complex_condition(self):
        """Test UPDATE with complex WHERE condition."""
        ds = DataStore(table="users")
        query = ds.update_set(active=False).filter((ds.age > 65) & (ds.status == 'inactive'))

        # Note: CompoundCondition generates SQL with parentheses around each part
        expected = 'ALTER TABLE "users" UPDATE "active"=0 WHERE ("age" > 65 AND "status" = \'inactive\')'
        self.assertEqual(expected, query.to_sql())


class TestDelete(unittest.TestCase):
    """Test DELETE operations (ClickHouse ALTER TABLE ... DELETE style)."""

    def test_delete_simple(self):
        """Test simple DELETE."""
        ds = DataStore(table="users")
        query = ds.delete_rows().filter(ds.id == 1)

        expected = 'ALTER TABLE "users" DELETE WHERE "id" = 1'
        self.assertEqual(expected, query.to_sql())

    def test_delete_with_complex_condition(self):
        """Test DELETE with complex WHERE condition."""
        ds = DataStore(table="users")
        query = ds.delete_rows().filter((ds.age < 18) | (ds.age > 65))

        # Note: CompoundCondition generates SQL with parentheses around the whole expression
        expected = 'ALTER TABLE "users" DELETE WHERE ("age" < 18 OR "age" > 65)'
        self.assertEqual(expected, query.to_sql())

    def test_delete_requires_where(self):
        """Test that DELETE without WHERE raises an error."""
        ds = DataStore(table="users")
        query = ds.delete_rows()

        with self.assertRaises(Exception) as context:
            query.to_sql()

        self.assertIn("WHERE clause", str(context.exception))

    def test_delete_all_with_where_1_equals_1(self):
        """Test DELETE all rows using WHERE 1=1."""
        ds = DataStore(table="users")
        query = ds.delete_rows().filter(Field('1') == 1)

        expected = 'ALTER TABLE "users" DELETE WHERE "1" = 1'
        self.assertEqual(expected, query.to_sql())


class TestInsertExecution(unittest.TestCase):
    """Test INSERT execution with chdb."""

    def setUp(self):
        """Set up test environment."""
        self.ds = DataStore(table="test_insert_users")
        self.ds.connect()

        # Create table
        self.ds.create_table(
            {"id": "UInt64", "name": "String", "age": "UInt8", "email": "Nullable(String)", "active": "UInt8"}
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS test_insert_users")
        except Exception:
            pass
        self.ds.close()

    def test_insert_with_values_execution(self):
        """Test INSERT with VALUES execution."""
        # Insert data
        query = self.ds.insert_into('id', 'name', 'age').insert_values((1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35))
        query.execute()

        # Verify data was inserted
        result = self.ds.select('*').execute()
        self.assertEqual(3, len(result))

        # Check specific values
        rows = result.to_dict(orient='records')
        self.assertEqual('Alice', rows[0]['name'])
        self.assertEqual(25, rows[0]['age'])

    def test_insert_with_null_execution(self):
        """Test INSERT with NULL values."""
        query = self.ds.insert_into('id', 'name', 'email').insert_values((1, 'Alice', None), (2, 'Bob', 'bob@test.com'))
        query.execute()

        result = self.ds.select('*').execute()
        self.assertEqual(2, len(result))

        rows = result.to_dict(orient='records')
        self.assertIsNone(rows[0]['email'])
        self.assertEqual('bob@test.com', rows[1]['email'])

    def test_insert_with_boolean_execution(self):
        """Test INSERT with boolean values."""
        query = self.ds.insert_into('id', 'name', 'active').insert_values((1, 'Alice', True), (2, 'Bob', False))
        query.execute()

        result = self.ds.select('name', 'active').execute()
        rows = result.to_dict(orient='records')

        self.assertEqual(1, rows[0]['active'])  # True -> 1
        self.assertEqual(0, rows[1]['active'])  # False -> 0

    def test_insert_from_select_execution(self):
        """Test INSERT INTO ... SELECT execution."""
        # First, insert some source data
        self.ds.insert(
            [
                {"id": 1, "name": "Alice", "age": 25, "active": 1},
                {"id": 2, "name": "Bob", "age": 30, "active": 0},
                {"id": 3, "name": "Charlie", "age": 35, "active": 1},
            ]
        )

        # Create target table
        ds_backup = DataStore(table="test_insert_backup")
        ds_backup.connect()
        ds_backup.create_table({"id": "UInt64", "name": "String", "age": "UInt8"})

        # Insert from SELECT with filter
        query = ds_backup.insert_into('id', 'name', 'age').select_from(
            self.ds.select('id', 'name', 'age').filter(self.ds.active == 1)
        )
        query.execute()

        # Verify only active users were copied
        result = ds_backup.select('*').execute()
        self.assertEqual(2, len(result))  # Alice and Charlie

        rows = result.to_dict(orient='records')
        names = [row['name'] for row in rows]
        self.assertIn('Alice', names)
        self.assertIn('Charlie', names)
        self.assertNotIn('Bob', names)

        ds_backup.close()


class TestUpdateExecution(unittest.TestCase):
    """Test UPDATE execution with chdb (ClickHouse ALTER TABLE ... UPDATE)."""

    def setUp(self):
        """Set up test environment."""
        self.ds = DataStore(table="test_update_users")
        self.ds.connect()

        # Create table (use MergeTree for mutations)
        schema_sql = """
        CREATE TABLE test_update_users (
            id UInt64,
            name String,
            age UInt8,
            city String,
            active UInt8
        ) ENGINE = MergeTree() ORDER BY id
        """
        self.ds._executor.execute(schema_sql)

        # Insert test data
        self.ds.insert(
            [
                {"id": 1, "name": "Alice", "age": 25, "city": "NYC", "active": 1},
                {"id": 2, "name": "Bob", "age": 30, "city": "LA", "active": 1},
                {"id": 3, "name": "Charlie", "age": 35, "city": "NYC", "active": 0},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS test_update_users")
        except Exception:
            pass
        self.ds.close()

    def test_update_single_field_execution(self):
        """Test UPDATE single field execution."""
        # Update Alice's age
        query = self.ds.update_set(age=26).filter(self.ds.id == 1)
        query.execute()

        # Verify update
        result = self.ds.select('name', 'age').filter(self.ds.id == 1).execute()
        rows = result.to_dict(orient='records')
        self.assertEqual('Alice', rows[0]['name'])
        self.assertEqual(26, rows[0]['age'])

    def test_update_multiple_fields_execution(self):
        """Test UPDATE multiple fields execution."""
        # Update Bob's age and city
        query = self.ds.update_set(age=31, city='SF').filter(self.ds.id == 2)
        query.execute()

        # Verify update
        result = self.ds.select('*').filter(self.ds.id == 2).execute()
        rows = result.to_dict(orient='records')
        self.assertEqual(31, rows[0]['age'])
        self.assertEqual('SF', rows[0]['city'])

    def test_update_with_condition_execution(self):
        """Test UPDATE with complex condition."""
        # Deactivate all NYC users over 30
        query = self.ds.update_set(active=0).filter((self.ds.city == 'NYC') & (self.ds.age > 30))
        query.execute()

        # Verify - only Charlie should be affected
        result = self.ds.select('name', 'active').filter(self.ds.active == 0).execute()
        rows = result.to_dict(orient='records')

        # Charlie was already inactive, should still be there
        names = [row['name'] for row in rows]
        self.assertIn('Charlie', names)

    def test_update_with_string_escaping_execution(self):
        """Test UPDATE with string that needs escaping."""
        query = self.ds.update_set(name="O'Brien").filter(self.ds.id == 1)
        query.execute()

        result = self.ds.select('name').filter(self.ds.id == 1).execute()
        rows = result.to_dict(orient='records')
        self.assertEqual("O'Brien", rows[0]['name'])


class TestDeleteExecution(unittest.TestCase):
    """Test DELETE execution with chdb (ClickHouse ALTER TABLE ... DELETE)."""

    def setUp(self):
        """Set up test environment."""
        self.ds = DataStore(table="test_delete_users")
        self.ds.connect()

        # Create table (use MergeTree for mutations)
        schema_sql = """
        CREATE TABLE test_delete_users (
            id UInt64,
            name String,
            age UInt8,
            city String
        ) ENGINE = MergeTree() ORDER BY id
        """
        self.ds._executor.execute(schema_sql)

        # Insert test data
        self.ds.insert(
            [
                {"id": 1, "name": "Alice", "age": 17, "city": "NYC"},
                {"id": 2, "name": "Bob", "age": 30, "city": "LA"},
                {"id": 3, "name": "Charlie", "age": 70, "city": "NYC"},
                {"id": 4, "name": "Diana", "age": 25, "city": "SF"},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS test_delete_users")
        except Exception:
            pass
        self.ds.close()

    def test_delete_single_row_execution(self):
        """Test DELETE single row execution."""
        # Delete Bob
        query = self.ds.delete_rows().filter(self.ds.id == 2)
        query.execute()

        # Verify deletion
        result = self.ds.select('*').execute()
        self.assertEqual(3, len(result))

        rows = result.to_dict(orient='records')
        names = [row['name'] for row in rows]
        self.assertNotIn('Bob', names)

    def test_delete_with_condition_execution(self):
        """Test DELETE with condition execution."""
        # Delete users outside age range 18-65
        query = self.ds.delete_rows().filter((self.ds.age < 18) | (self.ds.age > 65))
        query.execute()

        # Verify deletion
        result = self.ds.select('*').execute()
        self.assertEqual(2, len(result))  # Bob and Diana remain

        rows = result.to_dict(orient='records')
        names = [row['name'] for row in rows]
        self.assertIn('Bob', names)
        self.assertIn('Diana', names)
        self.assertNotIn('Alice', names)  # Too young
        self.assertNotIn('Charlie', names)  # Too old

    def test_delete_with_multiple_conditions_execution(self):
        """Test DELETE with multiple AND conditions."""
        # Delete NYC users under 20
        query = self.ds.delete_rows().filter((self.ds.city == 'NYC') & (self.ds.age < 20))
        query.execute()

        result = self.ds.select('*').execute()
        self.assertEqual(3, len(result))  # Only Alice deleted

        rows = result.to_dict(orient='records')
        names = [row['name'] for row in rows]
        self.assertNotIn('Alice', names)
        self.assertIn('Charlie', names)  # NYC but not under 20


if __name__ == '__main__':
    unittest.main()
