"""
Test immutability of DataStore operations - migrated from pypika test_immutability.py

DataStore uses @immutable decorator to ensure all operations return new instances.
"""

import unittest
from datastore import DataStore, Field
from datastore.functions import Count, Sum


class DataStoreImmutabilityTests(unittest.TestCase):
    """Test that DataStore operations are immutable"""

    def test_select_returns_new_instance(self):
        """Test that select() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("foo")
        ds2 = ds1.select("bar")

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertEqual('SELECT "foo" FROM "test"', ds1.to_sql())
        self.assertEqual('SELECT "foo", "bar" FROM "test"', ds2.to_sql())

    def test_filter_returns_new_instance(self):
        """Test that filter() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("*")
        ds2 = ds1.filter(Field("age") > 18)

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertEqual('SELECT * FROM "test"', ds1.to_sql())
        self.assertIn('WHERE', ds2.to_sql())

    def test_groupby_returns_new_instance(self):
        """Test that groupby() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("category")
        ds2 = ds1.groupby("category")

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotIn('GROUP BY', ds1.to_sql())
        self.assertIn('GROUP BY', ds2.to_sql())

    def test_having_returns_new_instance(self):
        """Test that having() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("category", Count("*")).groupby("category")
        ds2 = ds1.having(Count("*") > 5)

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotIn('HAVING', ds1.to_sql())
        self.assertIn('HAVING', ds2.to_sql())

    def test_sort_returns_new_instance(self):
        """Test that sort() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("name")
        ds2 = ds1.sort("name")

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotIn('ORDER BY', ds1.to_sql())
        self.assertIn('ORDER BY', ds2.to_sql())

    def test_limit_returns_new_instance(self):
        """Test that limit() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("*")
        ds2 = ds1.limit(10)

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotIn('LIMIT', ds1.to_sql())
        self.assertIn('LIMIT', ds2.to_sql())

    def test_offset_returns_new_instance(self):
        """Test that offset() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("*")
        ds2 = ds1.offset(10)

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotIn('OFFSET', ds1.to_sql())
        self.assertIn('OFFSET', ds2.to_sql())

    def test_distinct_returns_new_instance(self):
        """Test that distinct() returns a new DataStore instance"""
        ds1 = DataStore(table="test").select("category")
        ds2 = ds1.distinct()

        self.assertIsNot(ds1, ds2)
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotIn('DISTINCT', ds1.to_sql())
        self.assertIn('DISTINCT', ds2.to_sql())

    def test_chained_operations_return_new_instances(self):
        """Test that chained operations each return new instances"""
        ds0 = DataStore(table="test")
        ds1 = ds0.select("name", "age")
        ds2 = ds1.filter(Field("age") > 18)
        ds3 = ds2.sort("name")
        ds4 = ds3.limit(10)

        # All should be different instances
        self.assertIsNot(ds0, ds1)
        self.assertIsNot(ds1, ds2)
        self.assertIsNot(ds2, ds3)
        self.assertIsNot(ds3, ds4)

        # Each should have different SQL
        self.assertNotEqual(ds0.to_sql(), ds1.to_sql())
        self.assertNotEqual(ds1.to_sql(), ds2.to_sql())
        self.assertNotEqual(ds2.to_sql(), ds3.to_sql())
        self.assertNotEqual(ds3.to_sql(), ds4.to_sql())

    def test_original_datastore_unchanged_after_operations(self):
        """Test that original DataStore is not modified by operations"""
        original = DataStore(table="users").select("id", "name")
        original_sql = original.to_sql()

        # Perform various operations
        _ = original.filter(Field("age") > 18)
        _ = original.sort("name")
        _ = original.limit(10)
        _ = original.offset(5)
        _ = original.distinct()

        # Original should still be unchanged
        self.assertEqual(original_sql, original.to_sql())
        self.assertEqual('SELECT "id", "name" FROM "users"', original.to_sql())

    def test_branching_queries(self):
        """Test branching queries from a base query"""
        base = DataStore(table="products").select("id", "name", "price")

        # Branch 1: expensive products
        expensive = base.filter(Field("price") > 100).sort("price", ascending=False)

        # Branch 2: cheap products
        cheap = base.filter(Field("price") < 20).sort("price")

        # Base should be unchanged
        self.assertNotIn('WHERE', base.to_sql())
        self.assertNotIn('ORDER BY', base.to_sql())

        # Branches should be different
        self.assertIn('price" > 100', expensive.to_sql())
        self.assertIn('DESC', expensive.to_sql())

        self.assertIn('price" < 20', cheap.to_sql())
        self.assertIn('ASC', cheap.to_sql())

    def test_slice_notation_returns_new_instance(self):
        """Test that slice notation returns a new instance (pandas-like immutable behavior)."""
        ds = DataStore(table="test").select("*")

        # Store original SQL
        original_sql = ds.to_sql()

        # LIMIT 10 - returns new instance, original unchanged
        result = ds[:10]

        # Should be different instances (pandas-like immutable behavior)
        self.assertIsNot(ds, result)

        # Original should be unchanged
        self.assertNotIn('LIMIT', ds.to_sql())
        self.assertEqual(original_sql, ds.to_sql())

        # New instance should have LIMIT
        self.assertIn('LIMIT', result.to_sql())

    def test_multiple_filter_calls_accumulate(self):
        """Test that multiple filter calls are ANDed together"""
        ds = DataStore(table="users")
        ds1 = ds.select("*")
        ds2 = ds1.filter(Field("age") > 18)
        ds3 = ds2.filter(Field("city") == "NYC")

        # Each should be different
        self.assertIsNot(ds1, ds2)
        self.assertIsNot(ds2, ds3)

        # ds1 has no WHERE
        self.assertNotIn('WHERE', ds1.to_sql())

        # ds2 has one condition
        self.assertIn('WHERE', ds2.to_sql())
        self.assertIn('"age" > 18', ds2.to_sql())
        self.assertNotIn('"city"', ds2.to_sql())

        # ds3 has both conditions
        self.assertIn('WHERE', ds3.to_sql())
        self.assertIn('"age" > 18', ds3.to_sql())
        self.assertIn('"city" = \'NYC\'', ds3.to_sql())
        self.assertIn('AND', ds3.to_sql())


class ExpressionImmutabilityTests(unittest.TestCase):
    """Test immutability of expression operations"""

    def test_field_operations_return_new_instances(self):
        """Test that field operations return new expression instances"""
        field = Field("value")

        # Arithmetic operations
        expr1 = field + 10
        expr2 = field - 5
        expr3 = field * 2
        expr4 = field / 2

        # All should be different from original field
        self.assertIsNot(field, expr1)
        self.assertIsNot(field, expr2)
        self.assertIsNot(field, expr3)
        self.assertIsNot(field, expr4)

        # Original field should still be just a field
        self.assertEqual('"value"', field.to_sql())

    def test_condition_operations_return_new_instances(self):
        """Test that condition operations return new condition instances"""
        field = Field("age")

        cond1 = field > 18
        cond2 = field < 65
        cond3 = cond1 & cond2

        # All should be different instances
        self.assertIsNot(cond1, cond2)
        self.assertIsNot(cond1, cond3)
        self.assertIsNot(cond2, cond3)

        # cond1 and cond2 should be unchanged
        self.assertEqual('"age" > 18', cond1.to_sql())
        self.assertEqual('"age" < 65', cond2.to_sql())

    def test_field_alias_returns_new_instance(self):
        """Test that adding alias returns new instance"""
        field1 = Field("name")
        field2 = field1.as_("full_name")

        self.assertIsNot(field1, field2)
        self.assertEqual('"name"', field1.to_sql())
        self.assertEqual('"name" AS "full_name"', field2.to_sql(with_alias=True))


if __name__ == '__main__':
    unittest.main()
