"""
Test LIKE pattern matching - extended tests from pypika test_criterions.py

Comprehensive LIKE/NOT LIKE/ILIKE pattern tests with chdb execution.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field


# ========== SQL Generation Tests ==========


class LikePatternTests(unittest.TestCase):
    """Test various LIKE patterns"""

    def test_like_starts_with(self):
        """Test LIKE starts with pattern"""
        cond = Field("name").like("John%")
        self.assertEqual('"name" LIKE \'John%\'', cond.to_sql())

    def test_like_ends_with(self):
        """Test LIKE ends with pattern"""
        cond = Field("email").like("%@gmail.com")
        self.assertEqual('"email" LIKE \'%@gmail.com\'', cond.to_sql())

    def test_like_contains(self):
        """Test LIKE contains pattern"""
        cond = Field("description").like("%keyword%")
        self.assertEqual('"description" LIKE \'%keyword%\'', cond.to_sql())

    def test_like_exact_length(self):
        """Test LIKE for exact length (3 chars)"""
        cond = Field("code").like("___")
        self.assertEqual('"code" LIKE \'___\'', cond.to_sql())

    def test_like_pattern_with_underscore(self):
        """Test LIKE pattern with underscore wildcard"""
        cond = Field("phone").like("555-____")
        self.assertEqual('"phone" LIKE \'555-____\'', cond.to_sql())

    def test_like_complex_pattern(self):
        """Test LIKE with complex pattern"""
        cond = Field("sku").like("A_B%C")
        self.assertEqual('"sku" LIKE \'A_B%C\'', cond.to_sql())


class NotLikePatternTests(unittest.TestCase):
    """Test NOT LIKE patterns"""

    def test_notlike_starts_with(self):
        """Test NOT LIKE starts with"""
        cond = Field("name").notlike("Test%")
        self.assertEqual('"name" NOT LIKE \'Test%\'', cond.to_sql())

    def test_notlike_contains(self):
        """Test NOT LIKE contains"""
        cond = Field("text").notlike("%spam%")
        self.assertEqual('"text" NOT LIKE \'%spam%\'', cond.to_sql())

    def test_notlike_ends_with(self):
        """Test NOT LIKE ends with"""
        cond = Field("filename").notlike("%.tmp")
        self.assertEqual('"filename" NOT LIKE \'%.tmp\'', cond.to_sql())


class ILikePatternTests(unittest.TestCase):
    """Test ILIKE (case-insensitive) patterns"""

    def test_ilike_starts_with(self):
        """Test ILIKE starts with (case-insensitive)"""
        cond = Field("name").ilike("john%")
        self.assertEqual('"name" ILIKE \'john%\'', cond.to_sql())

    def test_ilike_contains(self):
        """Test ILIKE contains"""
        cond = Field("text").ilike("%keyword%")
        self.assertEqual('"text" ILIKE \'%keyword%\'', cond.to_sql())

    def test_ilike_ends_with(self):
        """Test ILIKE ends with"""
        cond = Field("domain").ilike("%@company.com")
        self.assertEqual('"domain" ILIKE \'%@company.com\'', cond.to_sql())


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class LikeExecutionTests(unittest.TestCase):
    """Test LIKE pattern execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_like_exec (
            id UInt32,
            name String,
            email String,
            phone String,
            code String
        ) ENGINE = Memory;
        
        INSERT INTO test_like_exec VALUES
            (1, 'John Smith', 'john@example.com', '555-1234', 'ABC'),
            (2, 'Jane Johnson', 'jane@gmail.com', '555-5678', 'DEF'),
            (3, 'Bob Jackson', 'bob@test.org', '666-1111', 'GHI'),
            (4, 'Alice Anderson', 'alice@example.com', '555-9999', 'JKL'),
            (5, 'John Doe', 'jdoe@company.net', '777-2222', 'MNO'),
            (6, 'Sarah Johnson', 'sarah@gmail.com', '555-3333', 'PQR');
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
        ds = DataStore(table="test_like_exec")
        sql = ds.select("id", "name").filter(ds.name.like('John%')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # John Smith, John Doe
        self.assertEqual(['1,John Smith', '5,John Doe'], lines)

    def test_like_ends_with_execution(self):
        """Test LIKE ends with execution"""
        ds = DataStore(table="test_like_exec")
        sql = ds.select("id").filter(ds.email.like('%@gmail.com')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # jane@gmail.com, sarah@gmail.com
        self.assertEqual(['2', '6'], lines)

    def test_like_contains_execution(self):
        """Test LIKE contains execution"""
        ds = DataStore(table="test_like_exec")
        sql = ds.select("id").filter(ds.name.like('%son%')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Johnson (2, 6), Jackson (3), Anderson (4)
        self.assertEqual(['2', '3', '4', '6'], lines)

    def test_like_exact_length_execution(self):
        """Test LIKE for exact length"""
        ds = DataStore(table="test_like_exec")
        sql = ds.select("id").filter(ds.code.like('___')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # All codes are 3 characters
        self.assertEqual(['1', '2', '3', '4', '5', '6'], lines)

    def test_notlike_execution(self):
        """Test NOT LIKE execution"""
        ds = DataStore(table="test_like_exec")
        sql = ds.select("id").filter(ds.phone.notlike('555-%')).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Not starting with 555-: 666-1111 (3), 777-2222 (5)
        self.assertEqual(['3', '5'], lines)

    def test_like_with_multiple_conditions_execution(self):
        """Test LIKE with multiple conditions"""
        ds = DataStore(table="test_like_exec")
        sql = (
            ds.select("id", "name")
            .filter((ds.name.like('%John%')) & (ds.email.like('%@example.com')))
            .sort("id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split('\n')
        # John Smith with @example.com
        self.assertEqual(['1,John Smith'], lines)


if __name__ == '__main__':
    unittest.main()
