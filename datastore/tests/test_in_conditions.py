"""
Test IN/NOT IN conditions - extended tests from pypika test_criterions.py

Comprehensive IN condition tests with various data types and chdb execution.
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


class InConditionDataTypesTests(unittest.TestCase):
    """Test IN with various data types"""

    def test_in_integers(self):
        """Test IN with integers"""
        cond = Field("id").isin([1, 2, 3, 4, 5])
        self.assertEqual('"id" IN (1,2,3,4,5)', cond.to_sql())

    def test_in_floats(self):
        """Test IN with float numbers"""
        cond = Field("price").isin([1.5, 2.5, 3.5])
        self.assertEqual('"price" IN (1.5,2.5,3.5)', cond.to_sql())

    def test_in_strings(self):
        """Test IN with strings"""
        cond = Field("status").isin(['active', 'pending', 'completed'])
        self.assertEqual('"status" IN (\'active\',\'pending\',\'completed\')', cond.to_sql())

    def test_in_dates(self):
        """Test IN with dates"""
        cond = Field("created").isin([date(2023, 1, 1), date(2023, 12, 31)])
        self.assertEqual('"created" IN (\'2023-01-01\',\'2023-12-31\')', cond.to_sql())

    def test_in_datetimes(self):
        """Test IN with datetimes"""
        cond = Field("timestamp").isin([datetime(2023, 1, 1, 0, 0, 0), datetime(2023, 12, 31, 23, 59, 59)])
        self.assertEqual('"timestamp" IN (\'2023-01-01 00:00:00\',\'2023-12-31 23:59:59\')', cond.to_sql())

    def test_in_single_value(self):
        """Test IN with single value"""
        cond = Field("id").isin([42])
        self.assertEqual('"id" IN (42)', cond.to_sql())

    def test_in_empty_list(self):
        """Test IN with empty list"""
        cond = Field("id").isin([])
        self.assertEqual('"id" IN ()', cond.to_sql())


class NotInConditionTests(unittest.TestCase):
    """Test NOT IN conditions"""

    def test_notin_integers(self):
        """Test NOT IN with integers"""
        cond = Field("id").notin([1, 2, 3])
        self.assertEqual('"id" NOT IN (1,2,3)', cond.to_sql())

    def test_notin_strings(self):
        """Test NOT IN with strings"""
        cond = Field("status").notin(['inactive', 'deleted'])
        self.assertEqual('"status" NOT IN (\'inactive\',\'deleted\')', cond.to_sql())

    def test_notin_dates(self):
        """Test NOT IN with dates"""
        cond = Field("holiday").notin([date(2023, 1, 1), date(2023, 12, 25)])
        self.assertEqual('"holiday" NOT IN (\'2023-01-01\',\'2023-12-25\')', cond.to_sql())


class InConditionInQueryTests(unittest.TestCase):
    """Test IN in complete queries"""

    def test_in_in_where(self):
        """Test IN in WHERE clause"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(ds.role.isin(['admin', 'moderator'])).to_sql()
        self.assertEqual('SELECT * FROM "users" WHERE "role" IN (\'admin\',\'moderator\')', sql)

    def test_in_with_multiple_conditions(self):
        """Test IN combined with other conditions"""
        ds = DataStore(table="products")
        sql = ds.select("*").filter((ds.category.isin(['A', 'B', 'C'])) & (ds.price > 50)).to_sql()
        self.assertIn('IN', sql)
        self.assertIn('AND', sql)

    def test_notin_in_where(self):
        """Test NOT IN in WHERE clause"""
        ds = DataStore(table="orders")
        sql = ds.select("*").filter(ds.status.notin(['cancelled', 'failed'])).to_sql()
        self.assertIn('NOT IN', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class InConditionExecutionTests(unittest.TestCase):
    """Test IN condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_in_exec (
            id UInt32,
            name String,
            status String,
            score Int32,
            price Float64,
            created_date Date
        ) ENGINE = Memory;
        
        INSERT INTO test_in_exec VALUES
            (1, 'Alice', 'active', 85, 100.50, '2023-01-15'),
            (2, 'Bob', 'pending', 72, 75.00, '2023-02-20'),
            (3, 'Charlie', 'active', 93, 150.75, '2023-03-10'),
            (4, 'David', 'inactive', 65, 50.00, '2023-04-05'),
            (5, 'Eve', 'active', 88, 125.00, '2023-05-18'),
            (6, 'Frank', 'deleted', 45, 25.50, '2023-06-22'),
            (7, 'Grace', 'pending', 79, 90.00, '2023-07-14');
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

    def test_in_integers_execution(self):
        """Test IN with integer list"""
        ds = DataStore(table="test_in_exec")
        sql = ds.select("id", "name").filter(ds.id.isin([1, 3, 5, 7])).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        self.assertEqual(['1,Alice', '3,Charlie', '5,Eve', '7,Grace'], lines)

    def test_in_strings_execution(self):
        """Test IN with string list"""
        ds = DataStore(table="test_in_exec")
        sql = ds.select("id").filter(ds.status.isin(['active', 'pending'])).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with status 'active' or 'pending': 1, 2, 3, 5, 7
        self.assertEqual(['1', '2', '3', '5', '7'], lines)

    def test_notin_execution(self):
        """Test NOT IN execution"""
        ds = DataStore(table="test_in_exec")
        sql = ds.select("id").filter(ds.status.notin(['inactive', 'deleted'])).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Exclude inactive (4) and deleted (6)
        self.assertEqual(['1', '2', '3', '5', '7'], lines)

    def test_in_with_range_filter_execution(self):
        """Test IN combined with range filter"""
        ds = DataStore(table="test_in_exec")
        sql = (
            ds.select("id", "name", "score")
            .filter((ds.status.isin(['active', 'pending'])) & (ds.score > 80))
            .sort("score", ascending=False)
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split('\n')
        # Charlie (active, 93), Eve (active, 88), Alice (active, 85)
        self.assertEqual(3, len(lines))

    def test_in_dates_execution(self):
        """Test IN with dates"""
        ds = DataStore(table="test_in_exec")
        sql = (
            ds.select("id", "name")
            .filter(ds.created_date.isin(['2023-01-15', '2023-03-10', '2023-05-18']))
            .sort("id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs 1, 3, 5
        self.assertEqual(['1,Alice', '3,Charlie', '5,Eve'], lines)

    def test_multiple_in_conditions_execution(self):
        """Test multiple IN conditions"""
        ds = DataStore(table="test_in_exec")
        sql = (
            ds.select("id")
            .filter(ds.status.isin(['active', 'pending']))
            .filter(ds.score.isin([85, 88, 93]))
            .sort("id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split('\n')
        # Alice (85), Charlie (93), Eve (88) - all active
        self.assertEqual(['1', '3', '5'], lines)

    def test_in_with_limit_execution(self):
        """Test IN with LIMIT"""
        ds = DataStore(table="test_in_exec")
        sql = ds.select("id", "name").filter(ds.status.isin(['active', 'pending'])).sort("id")[:3].to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # First 3 matching records
        self.assertEqual(3, len(lines))


if __name__ == '__main__':
    unittest.main()
