"""
Test advanced criteria/conditions - migrated from pypika test_criterions.py

Contains:
1. SQL generation tests for advanced conditions
2. Execution tests with chdb
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.conditions import UnaryCondition, InCondition, BetweenCondition, LikeCondition


# ========== SQL Generation Tests ==========


class NullConditionTests(unittest.TestCase):
    """isNull / isNotNull condition tests"""

    def test_isnull(self):
        """Test isNull() condition"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.foo.isnull()).to_sql()
        # toBool wrapper ensures bool dtype compatibility with pandas
        self.assertEqual('SELECT * FROM "test" WHERE toBool(isNull("foo")) = 1', sql)

    def test_notnull(self):
        """Test isNotNull() condition"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.foo.notnull()).to_sql()
        # toBool wrapper ensures bool dtype compatibility with pandas
        self.assertEqual('SELECT * FROM "test" WHERE toBool(isNotNull("foo")) = 1', sql)

    def test_isnull_with_field_object(self):
        """Test IS NULL with explicit Field"""
        field = Field("name")
        cond = field.isnull()
        self.assertEqual('"name" IS NULL', cond.to_sql())

    def test_notnull_with_field_object(self):
        """Test IS NOT NULL with explicit Field"""
        field = Field("name")
        cond = field.notnull()
        self.assertEqual('"name" IS NOT NULL', cond.to_sql())


class InConditionTests(unittest.TestCase):
    """IN / NOT IN condition tests"""

    def test_isin_numbers(self):
        """Test IN with numbers"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.id.isin([1, 2, 3])).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "id" IN (1,2,3)', sql)

    def test_isin_strings(self):
        """Test IN with strings"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.isin(['Alice', 'Bob'])).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" IN (\'Alice\',\'Bob\')', sql)

    def test_notin_numbers(self):
        """Test NOT IN with numbers"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.id.notin([1, 2, 3])).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "id" NOT IN (1,2,3)', sql)

    def test_notin_strings(self):
        """Test NOT IN with strings"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.notin(['Alice', 'Bob'])).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" NOT IN (\'Alice\',\'Bob\')', sql)

    def test_isin_single_value(self):
        """Test IN with single value"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.id.isin([1])).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "id" IN (1)', sql)


class BetweenConditionTests(unittest.TestCase):
    """BETWEEN condition tests"""

    def test_between_numbers(self):
        """Test BETWEEN with numbers"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.age.between(18, 65)).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "age" BETWEEN 18 AND 65', sql)

    def test_between_decimals(self):
        """Test BETWEEN with decimal numbers"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.price.between(10.5, 99.99)).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "price" BETWEEN 10.5 AND 99.99', sql)

    def test_between_with_field_object(self):
        """Test BETWEEN with explicit Field"""
        field = Field("score")
        cond = field.between(0, 100)
        self.assertEqual('"score" BETWEEN 0 AND 100', cond.to_sql())


class LikeConditionTests(unittest.TestCase):
    """LIKE / NOT LIKE / ILIKE condition tests"""

    def test_like_starts_with(self):
        """Test LIKE for starts with pattern"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.like('John%')).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" LIKE \'John%\'', sql)

    def test_like_ends_with(self):
        """Test LIKE for ends with pattern"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.like('%son')).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" LIKE \'%son\'', sql)

    def test_like_contains(self):
        """Test LIKE for contains pattern"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.like('%oh%')).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" LIKE \'%oh%\'', sql)

    def test_notlike(self):
        """Test NOT LIKE"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.notlike('Test%')).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" NOT LIKE \'Test%\'', sql)

    def test_ilike_case_insensitive(self):
        """Test ILIKE (case-insensitive)"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.ilike('john%')).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE "name" ILIKE \'john%\'', sql)


class CombinedConditionTests(unittest.TestCase):
    """Test combining advanced conditions"""

    def test_isnull_and_isin(self):
        """Test isNull() combined with IN"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter(ds.name.isnull() | ds.id.isin([1, 2, 3])).to_sql()
        # toBool wrapper ensures bool dtype compatibility with pandas
        self.assertEqual('SELECT * FROM "test" WHERE (toBool(isNull("name")) = 1 OR "id" IN (1,2,3))', sql)

    def test_between_and_like(self):
        """Test BETWEEN combined with LIKE"""
        ds = DataStore(table="test")
        sql = ds.select("*").filter((ds.age.between(18, 65)) & (ds.name.like('A%'))).to_sql()
        self.assertEqual('SELECT * FROM "test" WHERE ("age" BETWEEN 18 AND 65 AND "name" LIKE \'A%\')', sql)

    def test_complex_condition(self):
        """Test complex condition combination"""
        ds = DataStore(table="test")
        cond = (ds.age > 18) & (ds.name.notnull()) & (ds.city.isin(['NYC', 'LA']))
        sql = ds.select("*").filter(cond).to_sql()
        # notnull() returns ColumnExpr wrapping toBool(isNotNull()), which is converted to toBool(isNotNull()) = 1
        # toBool wrapper ensures bool dtype compatibility with pandas
        self.assertEqual(
            'SELECT * FROM "test" WHERE (("age" > 18 AND toBool(isNotNull("name")) = 1) AND "city" IN (\'NYC\',\'LA\'))',
            sql,
        )


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class NullConditionExecutionTests(unittest.TestCase):
    """Test NULL condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table with NULL values"""
        cls.init_sql = """
        CREATE TABLE test_nulls (
            id UInt32,
            name Nullable(String),
            value Nullable(UInt32)
        ) ENGINE = Memory;
        
        INSERT INTO test_nulls VALUES
            (1, 'Alice', 100),
            (2, NULL, 200),
            (3, 'Charlie', NULL),
            (4, NULL, NULL),
            (5, 'Eve', 500);
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
        csv_result = result.bytes().decode('utf-8').strip()
        return csv_result.replace('"', '').replace('\\N', 'NULL')

    def test_isnull_execution(self):
        """Test IS NULL execution"""
        ds = DataStore(table="test_nulls")
        sql = ds.select("id").filter(ds.name.isnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs 2 and 4 have NULL names
        self.assertEqual(['2', '4'], lines)

    def test_notnull_execution(self):
        """Test IS NOT NULL execution"""
        ds = DataStore(table="test_nulls")
        sql = ds.select("id").filter(ds.name.notnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs 1, 3, 5 have non-NULL names
        self.assertEqual(['1', '3', '5'], lines)

    def test_value_null_execution(self):
        """Test IS NULL on value column"""
        ds = DataStore(table="test_nulls")
        sql = ds.select("id").filter(ds.value.isnull()).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs 3 and 4 have NULL values
        self.assertEqual(['3', '4'], lines)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class InConditionExecutionTests(unittest.TestCase):
    """Test IN condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_in (
            id UInt32,
            category String,
            value UInt32
        ) ENGINE = Memory;
        
        INSERT INTO test_in VALUES
            (1, 'A', 10),
            (2, 'B', 20),
            (3, 'A', 30),
            (4, 'C', 40),
            (5, 'B', 50);
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

    def test_isin_execution(self):
        """Test IN execution"""
        ds = DataStore(table="test_in")
        sql = ds.select("id").filter(ds.id.isin([1, 3, 5])).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        self.assertEqual(['1', '3', '5'], lines)

    def test_isin_strings_execution(self):
        """Test IN with strings execution"""
        ds = DataStore(table="test_in")
        sql = ds.select("id").filter(ds.category.isin(['A', 'C'])).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs 1, 3 are category 'A', ID 4 is category 'C'
        self.assertEqual(['1', '3', '4'], lines)

    def test_notin_execution(self):
        """Test NOT IN execution"""
        ds = DataStore(table="test_in")
        sql = ds.select("id").filter(ds.id.notin([2, 4])).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        self.assertEqual(['1', '3', '5'], lines)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class BetweenConditionExecutionTests(unittest.TestCase):
    """Test BETWEEN condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_between (
            id UInt32,
            score UInt32,
            price Float64
        ) ENGINE = Memory;
        
        INSERT INTO test_between VALUES
            (1, 45, 15.5),
            (2, 78, 25.0),
            (3, 92, 35.75),
            (4, 33, 45.0),
            (5, 88, 55.25);
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

    def test_between_execution(self):
        """Test BETWEEN execution"""
        ds = DataStore(table="test_between")
        sql = ds.select("id").filter(ds.score.between(40, 80)).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with score between 40 and 80: 1 (45), 2 (78)
        self.assertEqual(['1', '2'], lines)

    def test_between_decimals_execution(self):
        """Test BETWEEN with decimals execution"""
        ds = DataStore(table="test_between")
        sql = ds.select("id").filter(ds.price.between(20.0, 40.0)).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with price between 20 and 40: 2 (25.0), 3 (35.75)
        self.assertEqual(['2', '3'], lines)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class LikeConditionExecutionTests(unittest.TestCase):
    """Test LIKE condition execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_like (
            id UInt32,
            name String,
            email String
        ) ENGINE = Memory;
        
        INSERT INTO test_like VALUES
            (1, 'John Smith', 'john@example.com'),
            (2, 'Jane Johnson', 'jane@test.com'),
            (3, 'Bob Jackson', 'bob@example.com'),
            (4, 'Alice Anderson', 'alice@demo.com'),
            (5, 'John Doe', 'jdoe@example.com');
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

    def test_like_starts_with_execution(self):
        """Test LIKE starts with execution"""
        ds = DataStore(table="test_like")
        sql = ds.select("id").filter(ds.name.like('John%')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with name starting with 'John': 1, 5
        self.assertEqual(['1', '5'], lines)

    def test_like_ends_with_execution(self):
        """Test LIKE ends with execution"""
        ds = DataStore(table="test_like")
        sql = ds.select("id").filter(ds.name.like('%son')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with name ending with 'son': 2 (Johnson), 3 (Jackson), 4 (Anderson)
        self.assertEqual(['2', '3', '4'], lines)

    def test_like_contains_execution(self):
        """Test LIKE contains execution"""
        ds = DataStore(table="test_like")
        sql = ds.select("id").filter(ds.email.like('%example.com')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with email ending with 'example.com': 1, 3, 5
        self.assertEqual(['1', '3', '5'], lines)

    def test_notlike_execution(self):
        """Test NOT LIKE execution"""
        ds = DataStore(table="test_like")
        sql = ds.select("id").filter(ds.name.notlike('John%')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # IDs with name NOT starting with 'John': 2, 3, 4
        self.assertEqual(['2', '3', '4'], lines)


if __name__ == '__main__':
    unittest.main()
