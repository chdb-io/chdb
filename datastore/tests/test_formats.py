"""
Test SQL formatting options - migrated from pypika test_formats.py

Tests quote character customization and SQL formatting.
"""

import unittest
from datastore import DataStore, Field
from datastore.functions import Sum


class QuoteCharTests(unittest.TestCase):
    """Test custom quote characters"""

    def test_default_double_quote(self):
        """Test default double quote character"""
        ds = DataStore(table="test")
        sql = ds.select("foo", "bar").to_sql()
        self.assertIn('"test"', sql)
        self.assertIn('"foo"', sql)
        self.assertIn('"bar"', sql)

    def test_custom_backtick_quote(self):
        """Test MySQL-style backtick quote"""
        ds = DataStore(table="test")
        sql = ds.select("foo", "bar").to_sql(quote_char='`')
        self.assertEqual('SELECT `foo`, `bar` FROM `test`', sql)

    def test_no_quote_char(self):
        """Test no quote character (Oracle style)"""
        ds = DataStore(table="test")
        sql = ds.select("foo", "bar").to_sql(quote_char='')
        self.assertEqual('SELECT foo, bar FROM test', sql)

    def test_quote_char_in_complex_query(self):
        """Test quote character in complex query"""
        ds = DataStore(table="sales")
        sql = (
            ds.select("category", Sum(ds.amount).as_("total"))
            .groupby("category")
            .having(Sum(ds.amount) > 1000)
            .sort("total", ascending=False)
            .to_sql(quote_char='`')
        )

        expected = 'SELECT `category`, SUM(`amount`) AS `total` FROM `sales` GROUP BY `category` HAVING SUM(`amount`) > 1000 ORDER BY `total` DESC'
        self.assertEqual(expected, sql)

    def test_quote_char_with_conditions(self):
        """Test quote character with WHERE conditions"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.age > 18).filter(ds.status == 'active').to_sql(quote_char='')
        self.assertEqual("SELECT * FROM users WHERE (age > 18 AND status = 'active')", sql)


class QuoteCharExecutionTests(unittest.TestCase):
    """Test that different quote characters produce equivalent results"""

    def test_quote_equivalence(self):
        """Test that different quote chars produce logically equivalent SQL"""
        ds = DataStore(table="test")
        query = ds.select("id", "name").filter(ds.id > 5).sort("name")

        sql_double = query.to_sql(quote_char='"')
        sql_backtick = query.to_sql(quote_char='`')
        sql_none = query.to_sql(quote_char='')

        # All should have the same structure
        self.assertIn('SELECT', sql_double)
        self.assertIn('SELECT', sql_backtick)
        self.assertIn('SELECT', sql_none)

        self.assertIn('id', sql_double)
        self.assertIn('id', sql_backtick)
        self.assertIn('id', sql_none)


if __name__ == '__main__':
    unittest.main()
