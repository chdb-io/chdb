"""
Test string functions - Upper, Lower, Concat and more

Tests string manipulation functions with chdb execution.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.functions import Upper, Lower, Concat, CustomFunction


# ========== SQL Generation Tests ==========


class StringFunctionTests(unittest.TestCase):
    """Test string function SQL generation"""

    def test_upper_function(self):
        """Test UPPER function"""
        func = Upper(Field("name"))
        self.assertEqual('UPPER("name")', func.to_sql())

    def test_lower_function(self):
        """Test LOWER function"""
        func = Lower(Field("name"))
        self.assertEqual('LOWER("name")', func.to_sql())

    def test_concat_two_fields(self):
        """Test CONCAT with two fields"""
        func = Concat(Field("first_name"), Field("last_name"))
        self.assertEqual('CONCAT("first_name","last_name")', func.to_sql())

    def test_concat_three_fields(self):
        """Test CONCAT with three fields"""
        func = Concat(Field("first"), Field("middle"), Field("last"))
        self.assertEqual('CONCAT("first","middle","last")', func.to_sql())

    def test_upper_in_select(self):
        """Test UPPER in SELECT"""
        ds = DataStore(table="users")
        sql = ds.select(Upper(ds.name).as_("upper_name")).to_sql()
        self.assertEqual('SELECT UPPER("name") AS "upper_name" FROM "users"', sql)

    def test_lower_in_select(self):
        """Test LOWER in SELECT"""
        ds = DataStore(table="users")
        sql = ds.select(Lower(ds.email).as_("lower_email")).to_sql()
        self.assertEqual('SELECT LOWER("email") AS "lower_email" FROM "users"', sql)

    def test_concat_in_select(self):
        """Test CONCAT in SELECT"""
        ds = DataStore(table="users")
        sql = ds.select(Concat(ds.first_name, ds.last_name).as_("full_name")).to_sql()
        self.assertEqual('SELECT CONCAT("first_name","last_name") AS "full_name" FROM "users"', sql)

    def test_string_function_in_where(self):
        """Test string function in WHERE"""
        ds = DataStore(table="users")
        sql = ds.select("*").filter(Upper(ds.name) == 'JOHN').to_sql()
        self.assertIn('WHERE UPPER("name") = \'JOHN\'', sql)

    def test_string_function_in_orderby(self):
        """Test string function in ORDER BY"""
        ds = DataStore(table="users")
        sql = ds.select("*").sort(Lower(ds.name)).to_sql()
        self.assertIn('ORDER BY LOWER("name")', sql)


class CustomStringFunctionTests(unittest.TestCase):
    """Test custom string functions"""

    def test_trim_function(self):
        """Test TRIM custom function"""
        Trim = CustomFunction("TRIM", ["str"])
        func = Trim(Field("text"))
        self.assertEqual('TRIM("text")', func.to_sql())

    def test_substring_function(self):
        """Test SUBSTRING custom function"""
        Substring = CustomFunction("SUBSTRING", ["str", "start", "length"])
        func = Substring(Field("text"), 1, 10)
        self.assertEqual('SUBSTRING("text",1,10)', func.to_sql())

    def test_length_function(self):
        """Test LENGTH custom function"""
        Length = CustomFunction("LENGTH", ["str"])
        func = Length(Field("name"))
        # LENGTH is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('toInt64(LENGTH("name"))', func.to_sql())


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class StringFunctionExecutionTests(unittest.TestCase):
    """Test string function execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_strings (
            id UInt32,
            first_name String,
            last_name String,
            email String,
            city String
        ) ENGINE = Memory;
        
        INSERT INTO test_strings VALUES
            (1, 'John', 'Smith', 'john@test.com', 'new york'),
            (2, 'jane', 'DOE', 'JANE@TEST.COM', 'los angeles'),
            (3, 'Bob', 'johnson', 'bob@test.com', 'CHICAGO'),
            (4, 'alice', 'WILSON', 'alice@test.com', 'Seattle');
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

    def test_upper_execution(self):
        """Test UPPER function execution"""
        ds = DataStore(table="test_strings")
        sql = ds.select(Upper(ds.first_name).as_("upper_name")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        self.assertEqual(['JOHN', 'JANE', 'BOB', 'ALICE'], lines)

    def test_lower_execution(self):
        """Test LOWER function execution"""
        ds = DataStore(table="test_strings")
        sql = ds.select(Lower(ds.last_name).as_("lower_name")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        self.assertEqual(['smith', 'doe', 'johnson', 'wilson'], lines)

    def test_concat_execution(self):
        """Test CONCAT function execution"""
        ds = DataStore(table="test_strings")
        sql = ds.select(Concat(ds.first_name, ds.last_name).as_("full_name")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Note: CONCAT without separator
        self.assertEqual(['JohnSmith', 'janeDOE', 'Bobjohnson', 'aliceWILSON'], lines)

    def test_upper_in_filter_execution(self):
        """Test UPPER in WHERE clause"""
        ds = DataStore(table="test_strings")
        sql = ds.select("id").filter(Upper(ds.city) == 'CHICAGO').to_sql()
        result = self._execute(sql)
        # Chicago (ID 3)
        self.assertEqual('3', result)

    def test_length_execution(self):
        """Test LENGTH custom function execution"""
        ds = DataStore(table="test_strings")
        Length = CustomFunction("LENGTH", ["str"])
        sql = ds.select("id", Length(ds.first_name).as_("name_len")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # John=4, jane=4, Bob=3, alice=5
        self.assertEqual(['1,4', '2,4', '3,3', '4,5'], lines)

    def test_string_function_in_groupby_execution(self):
        """Test string function in GROUP BY"""
        ds = DataStore(table="test_strings")
        from datastore.functions import Count

        sql = (
            ds.select(Upper(ds.city).as_("city_upper"), Count("*").as_("count"))
            .groupby(Upper(ds.city))
            .sort(Upper(ds.city))
            .to_sql()
        )

        result = self._execute(sql)
        lines = result.split('\n')
        # 4 unique cities when uppercased
        self.assertEqual(4, len(lines))


if __name__ == '__main__':
    unittest.main()
