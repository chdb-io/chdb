"""
Test condition combinations - migrated from pypika test_criterions.py ComplexCriterionTests

Tests AND, OR, XOR, NOT and complex nested conditions.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field


# ========== SQL Generation Tests ==========


class AndOrXorTests(unittest.TestCase):
    """Test AND, OR, XOR operators"""

    def test_and(self):
        """Test AND operator"""
        c = (Field("foo") == 1) & (Field("bar") == 2)
        self.assertIn('AND', c.to_sql())
        self.assertIn('"foo" = 1', c.to_sql())
        self.assertIn('"bar" = 2', c.to_sql())

    def test_or(self):
        """Test OR operator"""
        c = (Field("foo") == 1) | (Field("bar") == 2)
        self.assertIn('OR', c.to_sql())
        self.assertIn('"foo" = 1', c.to_sql())
        self.assertIn('"bar" = 2', c.to_sql())

    def test_xor(self):
        """Test XOR operator"""
        c = (Field("foo") == 1) ^ (Field("bar") == 2)
        self.assertIn('XOR', c.to_sql())
        self.assertIn('"foo" = 1', c.to_sql())
        self.assertIn('"bar" = 2', c.to_sql())


class NestedConditionsTests(unittest.TestCase):
    """Test nested condition combinations"""

    def test_nested_and(self):
        """Test chained AND conditions"""
        c = (Field("foo") == 1) & (Field("bar") == 2) & (Field("buz") == 3)
        sql = c.to_sql()
        self.assertIn('AND', sql)
        # All three conditions should be present
        self.assertIn('"foo" = 1', sql)
        self.assertIn('"bar" = 2', sql)
        self.assertIn('"buz" = 3', sql)

    def test_nested_or(self):
        """Test chained OR conditions"""
        c = (Field("foo") == 1) | (Field("bar") == 2) | (Field("buz") == 3)
        sql = c.to_sql()
        self.assertIn('OR', sql)

    def test_nested_mixed(self):
        """Test mixed AND/OR conditions"""
        c = ((Field("foo") == 1) & (Field("bar") == 2)) | (Field("buz") == 3)
        sql = c.to_sql()
        self.assertIn('AND', sql)
        self.assertIn('OR', sql)
        # Should have proper grouping
        self.assertIn('(', sql)

    def test_complex_nesting(self):
        """Test complex nested conditions"""
        c = (((Field("a") == 1) & (Field("b") == 2)) | ((Field("c") == 3) & (Field("d") == 4))) & (Field("e") == 5)
        sql = c.to_sql()
        # Should contain all operators
        self.assertIn('AND', sql)
        self.assertIn('OR', sql)


class NotConditionsTests(unittest.TestCase):
    """Test NOT operator"""

    def test_not_simple_condition(self):
        """Test NOT on simple condition"""
        c = ~(Field("foo") == 1)
        sql = c.to_sql()
        self.assertIn('NOT', sql)

    def test_not_or_criterion(self):
        """Test NOT on OR condition"""
        c = ~((Field("foo") == 1) | (Field("bar") == 2))
        sql = c.to_sql()
        self.assertIn('NOT', sql)
        self.assertIn('OR', sql)

    def test_not_and_criterion(self):
        """Test NOT on AND condition"""
        c = ~((Field("foo") == 1) & (Field("bar") == 2))
        sql = c.to_sql()
        self.assertIn('NOT', sql)
        self.assertIn('AND', sql)

    def test_double_negation(self):
        """Test double NOT"""
        c = ~(~(Field("foo") == 1))
        sql = c.to_sql()
        # Should have two NOT
        self.assertEqual(2, sql.count('NOT'))


class CombinedAdvancedConditionsTests(unittest.TestCase):
    """Test combinations of advanced conditions"""

    def test_between_and_isin(self):
        """Test BETWEEN combined with IN"""
        c = Field("foo").isin([1, 2, 3]) & Field("bar").between(0, 1)
        sql = c.to_sql()
        self.assertIn('IN', sql)
        self.assertIn('BETWEEN', sql)
        self.assertIn('AND', sql)

    def test_isnull_or_in(self):
        """Test IS NULL combined with IN using OR"""
        c = Field("foo").isnull() | Field("bar").isin([1, 2, 3])
        sql = c.to_sql()
        self.assertIn('IS NULL', sql)
        self.assertIn('IN', sql)
        self.assertIn('OR', sql)

    def test_like_and_between(self):
        """Test LIKE combined with BETWEEN"""
        c = (Field("name").like('A%')) & (Field("age").between(18, 65))
        sql = c.to_sql()
        self.assertIn('LIKE', sql)
        self.assertIn('BETWEEN', sql)

    def test_not_in_and_not_null(self):
        """Test NOT IN combined with NOT NULL"""
        c = Field("status").notin(['deleted', 'banned']) & Field("email").notnull()
        sql = c.to_sql()
        self.assertIn('NOT IN', sql)
        self.assertIn('IS NOT NULL', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class ConditionCombinationExecutionTests(unittest.TestCase):
    """Test condition combination execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_conditions (
            id UInt32,
            name String,
            age UInt32,
            city String,
            status String,
            score Nullable(UInt32)
        ) ENGINE = Memory;
        
        INSERT INTO test_conditions VALUES
            (1, 'Alice', 25, 'NYC', 'active', 85),
            (2, 'Bob', 30, 'LA', 'inactive', NULL),
            (3, 'Charlie', 35, 'NYC', 'active', 90),
            (4, 'David', 22, 'Chicago', 'pending', 75),
            (5, 'Eve', 45, 'LA', 'active', 95),
            (6, 'Frank', 28, 'NYC', 'deleted', 60);
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
        return result.bytes().decode('utf-8').strip().replace('"', '').replace('\\N', 'NULL')

    def test_and_condition_execution(self):
        """Test AND condition execution"""
        ds = DataStore(table="test_conditions")
        sql = ds.select("id", "name").filter((ds.age > 25) & (ds.city == 'NYC')).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Charlie (35, NYC), Frank (28, NYC)
        self.assertEqual(['3,Charlie', '6,Frank'], lines)

    def test_or_condition_execution(self):
        """Test OR condition execution"""
        ds = DataStore(table="test_conditions")
        sql = ds.select("id").filter((ds.age < 25) | (ds.age > 40)).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # David (22), Eve (45)
        self.assertEqual(['4', '5'], lines)

    def test_complex_and_or_execution(self):
        """Test complex AND/OR combination"""
        ds = DataStore(table="test_conditions")
        sql = (
            ds.select("id")
            .filter(((ds.city == 'NYC') | (ds.city == 'LA')) & (ds.status == 'active'))
            .sort("id")
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # Alice (NYC, active), Charlie (NYC, active), Eve (LA, active)
        self.assertEqual(['1', '3', '5'], lines)

    def test_not_condition_execution(self):
        """Test NOT condition execution"""
        ds = DataStore(table="test_conditions")
        sql = ds.select("id").filter(~(ds.status == 'deleted')).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # All except Frank (deleted)
        self.assertEqual(['1', '2', '3', '4', '5'], lines)

    def test_null_and_status_execution(self):
        """Test NULL check combined with status"""
        ds = DataStore(table="test_conditions")
        sql = ds.select("id", "name").filter(ds.score.isnull() | (ds.status == 'deleted')).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Bob (NULL score), Frank (deleted)
        self.assertEqual(['2,Bob', '6,Frank'], lines)

    def test_in_and_range_execution(self):
        """Test IN combined with range"""
        ds = DataStore(table="test_conditions")
        sql = ds.select("id").filter(ds.city.isin(['NYC', 'LA']) & ds.age.between(25, 35)).sort("id").to_sql()

        result = self._execute(sql)
        lines = result.split('\n')
        # Alice (25, NYC), Bob (30, LA), Charlie (35, NYC), Frank (28, NYC)
        self.assertGreater(len(lines), 0)


if __name__ == '__main__':
    unittest.main()
