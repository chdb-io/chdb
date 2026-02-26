"""
Test custom functions - migrated from pypika test_custom_functions.py

Tests the CustomFunction factory for creating user-defined SQL functions.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.functions import CustomFunction
from datastore.exceptions import ValidationError


# ========== SQL Generation Tests ==========


class CustomFunctionBasicTests(unittest.TestCase):
    """Test CustomFunction creation and SQL generation"""

    def test_custom_function_no_args(self):
        """Test custom function with no arguments"""
        CurrentDate = CustomFunction("CURRENT_DATE")
        func = CurrentDate()
        self.assertEqual("CURRENT_DATE()", func.to_sql())

    def test_custom_function_with_args(self):
        """Test custom function with arguments"""
        DateDiff = CustomFunction("DATE_DIFF", ["interval", "start_date", "end_date"])
        func = DateDiff("day", "2023-01-01", "2023-12-31")
        self.assertEqual("DATE_DIFF('day','2023-01-01','2023-12-31')", func.to_sql())

    def test_custom_function_with_field_args(self):
        """Test custom function with Field arguments"""
        DateDiff = CustomFunction("DATE_DIFF", ["interval", "start_date", "end_date"])
        func = DateDiff("day", Field("created_date"), Field("updated_date"))
        self.assertEqual('DATE_DIFF(\'day\',"created_date","updated_date")', func.to_sql())

    def test_custom_function_with_wrong_arg_count(self):
        """Test custom function with wrong number of arguments raises error"""
        DateDiff = CustomFunction("DATE_DIFF", ["interval", "start_date", "end_date"])

        with self.assertRaises(ValidationError):
            DateDiff("day")  # Missing 2 arguments

    def test_custom_function_with_alias(self):
        """Test custom function with alias"""
        Sha256 = CustomFunction("SHA256", ["text"])
        func = Sha256("hello").as_("hash")
        self.assertEqual('SHA256(\'hello\') AS "hash"', func.to_sql(with_alias=True))

    def test_custom_function_in_select(self):
        """Test custom function in SELECT clause"""
        ds = DataStore(table="service")
        DateDiff = CustomFunction("DATE_DIFF", ["interval", "start_date", "end_date"])

        sql = ds.select(DateDiff("day", ds.created_date, ds.updated_date).as_("days")).to_sql()
        self.assertEqual('SELECT DATE_DIFF(\'day\',"created_date","updated_date") AS "days" FROM "service"', sql)

    def test_custom_function_with_mixed_args(self):
        """Test custom function with mixed argument types"""
        Coalesce = CustomFunction("COALESCE", ["value1", "value2", "default"])
        func = Coalesce(Field("email"), Field("phone"), "N/A")
        self.assertEqual('COALESCE("email","phone",\'N/A\')', func.to_sql())

    def test_custom_function_with_numeric_args(self):
        """Test custom function with numeric arguments"""
        Power = CustomFunction("POWER", ["base", "exponent"])
        func = Power(Field("value"), 2)
        self.assertEqual('POWER("value",2)', func.to_sql())

    def test_custom_function_case_sensitive_name(self):
        """Test custom function preserves case in name"""
        MyFunc = CustomFunction("MyCustomFunc", ["arg1"])
        func = MyFunc("test")
        self.assertEqual("MyCustomFunc('test')", func.to_sql())

    def test_custom_function_no_required_args(self):
        """Test custom function with no required arguments (variadic)"""
        Concat = CustomFunction("CONCAT")
        func = Concat("a", "b", "c")
        self.assertEqual("CONCAT('a','b','c')", func.to_sql())


class CustomFunctionUsageTests(unittest.TestCase):
    """Test using custom functions in queries"""

    def test_custom_function_in_where(self):
        """Test custom function in WHERE clause"""
        ds = DataStore(table="users")
        Length = CustomFunction("LENGTH", ["str"])

        sql = ds.select("name").filter(Length(ds.name) > 10).to_sql()
        # LENGTH is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('SELECT "name" FROM "users" WHERE toInt64(LENGTH("name")) > 10', sql)

    def test_custom_function_in_groupby(self):
        """Test custom function in GROUP BY"""
        ds = DataStore(table="events")
        DateTrunc = CustomFunction("DATE_TRUNC", ["unit", "timestamp"])

        truncated_date = DateTrunc("day", ds.created_at)
        sql = ds.select(truncated_date.as_("date")).groupby(truncated_date).to_sql()
        self.assertEqual(
            'SELECT DATE_TRUNC(\'day\',"created_at") AS "date" FROM "events" GROUP BY DATE_TRUNC(\'day\',"created_at")',
            sql,
        )

    def test_custom_function_in_orderby(self):
        """Test custom function in ORDER BY"""
        ds = DataStore(table="products")
        Length = CustomFunction("LENGTH", ["str"])

        sql = ds.select("name").sort(Length(ds.name), ascending=False).to_sql()
        # LENGTH is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('SELECT "name" FROM "products" ORDER BY toInt64(LENGTH("name")) DESC', sql)

    def test_nested_custom_functions(self):
        """Test nested custom functions"""
        Upper = CustomFunction("UPPER", ["str"])
        Trim = CustomFunction("TRIM", ["str"])

        func = Upper(Trim(Field("name")))
        self.assertEqual('UPPER(TRIM("name"))', func.to_sql())


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class CustomFunctionExecutionTests(unittest.TestCase):
    """Test custom function execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_custom (
            id UInt32,
            name String,
            value UInt32,
            created_date Date,
            updated_date Date
        ) ENGINE = Memory;
        
        INSERT INTO test_custom VALUES
            (1, 'test', 100, '2023-01-01', '2023-01-10'),
            (2, 'hello world', 200, '2023-01-05', '2023-01-15'),
            (3, 'x', 300, '2023-01-10', '2023-01-20');
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

    def test_length_function_execution(self):
        """Test LENGTH custom function execution"""
        ds = DataStore(table="test_custom")
        Length = CustomFunction("LENGTH", ["str"])

        sql = ds.select(Length(ds.name).as_("len")).filter(ds.id <= 3).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # lengths: 'test'=4, 'hello world'=11, 'x'=1
        self.assertEqual(['4', '11', '1'], lines)

    def test_upper_function_execution(self):
        """Test UPPER custom function execution"""
        ds = DataStore(table="test_custom")
        Upper = CustomFunction("UPPER", ["str"])

        sql = ds.select(Upper(ds.name).as_("upper_name")).filter(ds.id <= 3).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        self.assertEqual(['TEST', 'HELLO WORLD', 'X'], lines)

    def test_custom_function_in_filter_execution(self):
        """Test custom function in WHERE clause execution"""
        ds = DataStore(table="test_custom")
        Length = CustomFunction("LENGTH", ["str"])

        sql = ds.select("id").filter(Length(ds.name) > 5).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Only 'hello world' has length > 5
        self.assertEqual(['2'], lines)

    def test_datediff_execution(self):
        """Test DATE_DIFF custom function execution"""
        ds = DataStore(table="test_custom")
        DateDiff = CustomFunction("dateDiff", ["unit", "start", "end"])

        # ClickHouse uses dateDiff (camelCase)
        sql = (
            ds.select("id", DateDiff("day", ds.created_date, ds.updated_date).as_("diff"))
            .filter(ds.id <= 3)
            .sort("id")
            .to_sql()
        )
        result = self._execute(sql)
        lines = result.split('\n')
        # Differences: 9 days, 10 days, 10 days
        expected_diffs = ['1,9', '2,10', '3,10']
        self.assertEqual(expected_diffs, lines)

    def test_coalesce_execution(self):
        """Test COALESCE custom function execution"""
        # Add NULL values for testing
        self.session.query("INSERT INTO test_custom VALUES (4, '', 400, '2023-01-01', '2023-01-01')")

        ds = DataStore(table="test_custom")
        Coalesce = CustomFunction("coalesce", ["value1", "value2"])

        sql = ds.select("id", Coalesce(ds.name, "default").as_("result")).filter(ds.id == 4).to_sql()
        result = self._execute(sql)
        # Empty string should be returned (or 'default' if NULL)
        self.assertIn('4', result)


if __name__ == '__main__':
    unittest.main()
