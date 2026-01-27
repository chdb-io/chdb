"""
Test BETWEEN conditions - migrated from pypika test_criterions.py BetweenTests

Comprehensive BETWEEN condition tests with chdb execution.
"""

import unittest
from datetime import date, datetime

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field


# ========== SQL Generation Tests ==========


class BetweenBasicTests(unittest.TestCase):
    """Basic BETWEEN tests"""

    def test_between_numbers(self):
        """Test BETWEEN with numbers"""
        cond = Field("foo").between(0, 1)
        self.assertEqual('"foo" BETWEEN 0 AND 1', cond.to_sql())

    def test_between_decimals(self):
        """Test BETWEEN with decimal numbers"""
        cond = Field("price").between(10.5, 99.99)
        self.assertEqual('"price" BETWEEN 10.5 AND 99.99', cond.to_sql())

    def test_between_negative_numbers(self):
        """Test BETWEEN with negative numbers"""
        cond = Field("temperature").between(-10, 30)
        self.assertEqual('"temperature" BETWEEN -10 AND 30', cond.to_sql())

    def test_between_dates(self):
        """Test BETWEEN with dates"""
        cond = Field("created").between(date(2023, 1, 1), date(2023, 12, 31))
        self.assertEqual('"created" BETWEEN \'2023-01-01\' AND \'2023-12-31\'', cond.to_sql())

    def test_between_datetimes(self):
        """Test BETWEEN with datetimes"""
        cond = Field("timestamp").between(datetime(2023, 1, 1, 0, 0, 0), datetime(2023, 12, 31, 23, 59, 59))
        self.assertEqual('"timestamp" BETWEEN \'2023-01-01 00:00:00\' AND \'2023-12-31 23:59:59\'', cond.to_sql())

    def test_between_strings(self):
        """Test BETWEEN with strings (alphabetic range)"""
        cond = Field("name").between('A', 'M')
        self.assertEqual('"name" BETWEEN \'A\' AND \'M\'', cond.to_sql())


class BetweenInQueryTests(unittest.TestCase):
    """Test BETWEEN in complete queries"""

    def test_between_in_where(self):
        """Test BETWEEN in WHERE clause"""
        ds = DataStore(table="products")
        sql = ds.select("*").filter(ds.price.between(10, 100)).to_sql()
        self.assertEqual('SELECT * FROM "products" WHERE "price" BETWEEN 10 AND 100', sql)

    def test_between_combined_with_other_conditions(self):
        """Test BETWEEN combined with other conditions"""
        ds = DataStore(table="orders")
        sql = ds.select("*").filter((ds.amount.between(100, 1000)) & (ds.status == 'completed')).to_sql()
        self.assertIn('BETWEEN', sql)
        self.assertIn('AND "status"', sql)

    def test_multiple_between_conditions(self):
        """Test multiple BETWEEN conditions"""
        ds = DataStore(table="products")
        sql = ds.select("*").filter(ds.price.between(10, 100)).filter(ds.quantity.between(5, 50)).to_sql()
        # Both BETWEEN conditions should be present
        sql_parts = sql.split('BETWEEN')
        self.assertEqual(3, len(sql_parts))  # 1 before + 2 BETWEEN = 3 parts

    def test_between_with_order_by(self):
        """Test BETWEEN with ORDER BY"""
        ds = DataStore(table="users")
        sql = ds.select("name", "age").filter(ds.age.between(18, 65)).sort("age").to_sql()
        self.assertIn('BETWEEN', sql)
        self.assertIn('ORDER BY', sql)


class BetweenEdgeCasesTests(unittest.TestCase):
    """Test BETWEEN edge cases"""

    def test_between_same_value(self):
        """Test BETWEEN with same lower and upper bound"""
        cond = Field("value").between(5, 5)
        self.assertEqual('"value" BETWEEN 5 AND 5', cond.to_sql())

    def test_between_zero_range(self):
        """Test BETWEEN 0 to 0"""
        cond = Field("count").between(0, 0)
        self.assertEqual('"count" BETWEEN 0 AND 0', cond.to_sql())

    def test_between_large_numbers(self):
        """Test BETWEEN with large numbers"""
        cond = Field("population").between(1000000, 10000000)
        self.assertEqual('"population" BETWEEN 1000000 AND 10000000', cond.to_sql())


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class BetweenExecutionTests(unittest.TestCase):
    """Test BETWEEN execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_between_exec (
            id UInt32,
            value Int32,
            price Float64,
            created_date Date,
            name String
        ) ENGINE = Memory;
        
        INSERT INTO test_between_exec VALUES
            (1, 5, 15.50, '2023-01-15', 'Alice'),
            (2, 15, 25.00, '2023-03-20', 'Bob'),
            (3, 25, 35.75, '2023-06-10', 'Charlie'),
            (4, 35, 45.00, '2023-09-05', 'David'),
            (5, 45, 55.25, '2023-11-25', 'Eve'),
            (6, -10, 65.00, '2023-12-01', 'Frank'),
            (7, 100, 5.50, '2023-02-14', 'Grace');
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session"""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute(self, sql):
        """Helper to execute SQL and return CSV result"""
        sql_no_quotes = sql.replace('"', '')
        result = self.session.query(sql_no_quotes, 'CSV')
        return result.bytes().decode('utf-8').strip().replace('"', '')

    def test_between_integer_execution(self):
        """Test BETWEEN with integers"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id").filter(ds.value.between(10, 30)).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # values 15, 25 are in range [10, 30]
        self.assertEqual(['2', '3'], lines)

    def test_between_float_execution(self):
        """Test BETWEEN with floats"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id").filter(ds.price.between(20.0, 50.0)).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # prices 25.00, 35.75, 45.00 are in range
        self.assertEqual(['2', '3', '4'], lines)

    def test_between_dates_execution(self):
        """Test BETWEEN with dates"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id").filter(ds.created_date.between('2023-03-01', '2023-09-30')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # dates in Mar-Sep: 2023-03-20, 2023-06-10, 2023-09-05
        self.assertEqual(['2', '3', '4'], lines)

    def test_between_with_negative_execution(self):
        """Test BETWEEN including negative numbers"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id").filter(ds.value.between(-20, 20)).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # values: 5, 15, -10
        self.assertEqual(['1', '2', '6'], lines)

    def test_between_same_value_execution(self):
        """Test BETWEEN with same lower and upper bound"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id").filter(ds.value.between(25, 25)).to_sql()
        result = self._execute(sql)
        # Only exact match to 25
        self.assertEqual('3', result)

    def test_between_with_other_conditions_execution(self):
        """Test BETWEEN combined with other conditions"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id", "name").filter((ds.value.between(10, 40)) & (ds.price > 30)).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # value in [10,40] AND price > 30: rows 3 (25, 35.75), 4 (35, 45.00)
        self.assertEqual(['3,Charlie', '4,David'], lines)

    def test_between_strings_execution(self):
        """Test BETWEEN with strings (alphabetic)"""
        ds = DataStore(table="test_between_exec")
        sql = ds.select("id", "name").filter(ds.name.between('A', 'D')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Names A-D: Alice, Bob, Charlie, David (not Eve, Frank, Grace)
        self.assertIn('Alice', result)
        self.assertIn('Bob', result)
        self.assertIn('Charlie', result)


if __name__ == '__main__':
    unittest.main()
