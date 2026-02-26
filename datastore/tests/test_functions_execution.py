"""
Test function execution on chDB - real query execution tests
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore.expressions import Field, Literal
from datastore.functions import Function, Sum, Count, Avg, Min, Max, Upper, Lower, Concat, CustomFunction


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestAggregateFunctionExecution(unittest.TestCase):
    """Test aggregate functions with real execution on chDB."""

    @classmethod
    def setUpClass(cls):
        """Create test table with sample data."""
        # chDB works with in-memory data, create and populate in one session
        cls.init_sql = """
        CREATE TABLE test_orders (
            id UInt32,
            customer_id UInt32,
            amount Float64,
            price Float64,
            quantity UInt32
        ) ENGINE = Memory;
        
        INSERT INTO test_orders VALUES
            (1, 101, 100.0, 10.0, 10),
            (2, 102, 200.0, 20.0, 10),
            (3, 101, 150.0, 15.0, 10),
            (4, 103, 300.0, 30.0, 10),
            (5, 102, 250.0, 25.0, 10);
        """

        # Create a session for this test class
        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session."""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute_select(self, *expressions, table='test_orders'):
        """Helper to execute SELECT with expressions."""
        # Build SQL
        if not expressions:
            select_sql = '*'
        else:
            select_sql = ', '.join(
                expr.to_sql(quote_char='') if hasattr(expr, 'to_sql') else str(expr) for expr in expressions
            )

        sql = f"SELECT {select_sql} FROM {table}"
        result = self.session.query(sql, 'CSV')
        return result.bytes().decode('utf-8').strip()

    def test_sum_execution(self):
        """Test SUM function execution."""
        func = Sum(Field('amount'))
        result = self._execute_select(func)
        # Sum: 100 + 200 + 150 + 300 + 250 = 1000
        self.assertEqual('1000', result)

    def test_count_star_execution(self):
        """Test COUNT(*) execution."""
        func = Count('*')
        result = self._execute_select(func)
        # 5 rows
        self.assertEqual('5', result)

    def test_count_field_execution(self):
        """Test COUNT(field) execution."""
        func = Count(Field('id'))
        result = self._execute_select(func)
        self.assertEqual('5', result)

    def test_avg_execution(self):
        """Test AVG function execution."""
        func = Avg(Field('amount'))
        result = self._execute_select(func)
        # Avg: 1000 / 5 = 200
        self.assertEqual('200', result)

    def test_min_execution(self):
        """Test MIN function execution."""
        func = Min(Field('amount'))
        result = self._execute_select(func)
        self.assertEqual('100', result)

    def test_max_execution(self):
        """Test MAX function execution."""
        func = Max(Field('amount'))
        result = self._execute_select(func)
        self.assertEqual('300', result)

    def test_multiple_aggregates_execution(self):
        """Test multiple aggregate functions together."""
        sum_func = Sum(Field('amount'))
        count_func = Count('*')
        avg_func = Avg(Field('price'))

        result = self._execute_select(sum_func, count_func, avg_func)
        # Expected: 1000, 5, 20.0
        values = result.split(',')
        self.assertEqual('1000', values[0])
        self.assertEqual('5', values[1])
        self.assertEqual('20', values[2])


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestStringFunctionExecution(unittest.TestCase):
    """Test string functions with real execution on chDB."""

    @classmethod
    def setUpClass(cls):
        """Create test table with string data."""
        cls.init_sql = """
        CREATE TABLE test_users (
            id UInt32,
            first_name String,
            last_name String,
            email String
        ) ENGINE = Memory;
        
        INSERT INTO test_users VALUES
            (1, 'John', 'Doe', 'john@example.com'),
            (2, 'Jane', 'Smith', 'jane@example.com'),
            (3, 'Bob', 'Johnson', 'bob@example.com');
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session."""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute_select(self, *expressions, table='test_users', where=None):
        """Helper to execute SELECT with expressions."""
        if not expressions:
            select_sql = '*'
        else:
            select_sql = ', '.join(
                expr.to_sql(quote_char='') if hasattr(expr, 'to_sql') else str(expr) for expr in expressions
            )

        sql = f"SELECT {select_sql} FROM {table}"
        if where:
            sql += f" WHERE {where}"

        result = self.session.query(sql, 'CSV')
        return result.bytes().decode('utf-8').strip()

    def test_upper_execution(self):
        """Test UPPER function execution."""
        func = Upper(Field('first_name'))
        result = self._execute_select(func, where='id = 1')
        # CSV format adds quotes to strings
        self.assertEqual('"JOHN"', result)

    def test_lower_execution(self):
        """Test LOWER function execution."""
        func = Lower(Field('first_name'))
        result = self._execute_select(func, where='id = 1')
        # CSV format adds quotes to strings
        self.assertEqual('"john"', result)

    def test_concat_execution(self):
        """Test CONCAT function execution."""
        func = Concat(Field('first_name'), Literal(' '), Field('last_name'))
        result = self._execute_select(func, where='id = 1')
        # CSV format adds quotes to strings
        self.assertEqual('"John Doe"', result)

    def test_nested_string_functions_execution(self):
        """Test nested string functions."""
        # UPPER(CONCAT(first_name, ' ', last_name))
        inner = Concat(Field('first_name'), Literal(' '), Field('last_name'))
        outer = Upper(inner)
        result = self._execute_select(outer, where='id = 1')
        # CSV format adds quotes to strings
        self.assertEqual('"JOHN DOE"', result)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestFunctionCompositionExecution(unittest.TestCase):
    """Test function composition with real execution."""

    @classmethod
    def setUpClass(cls):
        """Create test table."""
        cls.init_sql = """
        CREATE TABLE test_sales (
            id UInt32,
            product String,
            revenue Float64,
            cost Float64,
            quantity UInt32
        ) ENGINE = Memory;
        
        INSERT INTO test_sales VALUES
            (1, 'Product A', 1000.0, 600.0, 10),
            (2, 'Product B', 2000.0, 1200.0, 20),
            (3, 'Product C', 1500.0, 900.0, 15);
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session."""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def _execute_select(self, *expressions, table='test_sales'):
        """Helper to execute SELECT."""
        if not expressions:
            select_sql = '*'
        else:
            select_sql = ', '.join(
                expr.to_sql(quote_char='') if hasattr(expr, 'to_sql') else str(expr) for expr in expressions
            )

        sql = f"SELECT {select_sql} FROM {table}"
        result = self.session.query(sql, 'CSV')
        return result.bytes().decode('utf-8').strip()

    def test_arithmetic_with_aggregates_execution(self):
        """Test arithmetic operations with aggregate functions."""
        # SUM(revenue) - SUM(cost) = Total Profit
        from datastore.expressions import ArithmeticExpression

        revenue_sum = Sum(Field('revenue'))
        cost_sum = Sum(Field('cost'))
        profit = revenue_sum - cost_sum

        result = self._execute_select(profit)
        # (1000 + 2000 + 1500) - (600 + 1200 + 900) = 4500 - 2700 = 1800
        self.assertEqual('1800', result)

    def test_avg_calculation_execution(self):
        """Test average calculation with division."""
        # SUM(revenue) / COUNT(*) = Average revenue
        total = Sum(Field('revenue'))
        count = Count('*')
        avg_manual = total / count

        result = self._execute_select(avg_manual)
        # 4500 / 3 = 1500
        self.assertEqual('1500', result)


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestCustomFunctionExecution(unittest.TestCase):
    """Test custom functions with real execution."""

    def test_custom_function_with_stateless_query(self):
        """Test custom function with stateless chDB query."""
        # Use ClickHouse's toYear function with proper Date conversion
        ToYear = CustomFunction('toYear')
        ToDate = CustomFunction('toDate')

        # First convert string to Date, then extract year
        date_expr = ToDate(Literal('2024-01-15'))
        func = ToYear(date_expr)

        sql = f"SELECT {func.to_sql(quote_char='')} AS year"
        result = chdb.query(sql, 'CSV')
        self.assertEqual('2024', result.bytes().decode('utf-8').strip())

    def test_custom_date_function_execution(self):
        """Test custom date function."""
        # Use ClickHouse's dateDiff function
        DateDiff = CustomFunction('dateDiff', ['unit', 'date1', 'date2'])
        ToDate = CustomFunction('toDate')

        # Test with proper Date conversion (stateless query)
        date1 = ToDate(Literal('2024-01-01'))
        date2 = ToDate(Literal('2024-01-15'))
        func = DateDiff(Literal('day'), date1, date2)

        sql = f"SELECT {func.to_sql(quote_char='')} AS days"
        result = chdb.query(sql, 'CSV')
        # dateDiff('day', toDate('2024-01-01'), toDate('2024-01-15')) = 14
        self.assertEqual('14', result.bytes().decode('utf-8').strip())

    def test_custom_function_sql_generation(self):
        """Test that CustomFunction generates correct SQL."""
        MyFunc = CustomFunction('myCustomFunc', ['arg1', 'arg2'])
        func = MyFunc(Field('x'), Literal(42))

        expected_sql = "myCustomFunc(x,42)"
        self.assertEqual(expected_sql, func.to_sql(quote_char=''))


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class TestFunctionWithGroupBy(unittest.TestCase):
    """Test functions with GROUP BY execution."""

    @classmethod
    def setUpClass(cls):
        """Create test table for grouping."""
        cls.init_sql = """
        CREATE TABLE test_transactions (
            id UInt32,
            category String,
            amount Float64,
            quantity UInt32
        ) ENGINE = Memory;
        
        INSERT INTO test_transactions VALUES
            (1, 'Electronics', 1000.0, 5),
            (2, 'Electronics', 1500.0, 7),
            (3, 'Books', 200.0, 10),
            (4, 'Books', 300.0, 15),
            (5, 'Clothing', 500.0, 8);
        """

        cls.session = chdb.session.Session()
        cls.session.query(cls.init_sql)

    @classmethod
    def tearDownClass(cls):
        """Clean up session."""
        if hasattr(cls, 'session'):
            cls.session.cleanup()

    def test_sum_with_group_by_execution(self):
        """Test SUM with GROUP BY."""
        sql = """
        SELECT category, SUM(amount) as total
        FROM test_transactions
        GROUP BY category
        ORDER BY category
        """
        result = self.session.query(sql, 'CSV')
        lines = result.bytes().decode('utf-8').strip().split('\n')

        # Expected results (sorted by category):
        # Books: 200 + 300 = 500
        # Clothing: 500
        # Electronics: 1000 + 1500 = 2500
        # CSV format adds quotes to strings
        self.assertEqual('"Books",500', lines[0])
        self.assertEqual('"Clothing",500', lines[1])
        self.assertEqual('"Electronics",2500', lines[2])

    def test_multiple_aggregates_with_group_by(self):
        """Test multiple aggregates with GROUP BY."""
        sql = """
        SELECT 
            category,
            SUM(amount) as total,
            COUNT(*) as count,
            AVG(amount) as avg_amount
        FROM test_transactions
        GROUP BY category
        ORDER BY category
        """
        result = self.session.query(sql, 'CSV')
        lines = result.bytes().decode('utf-8').strip().split('\n')

        # Books: sum=500, count=2, avg=250
        values = lines[0].split(',')
        # CSV format adds quotes to strings
        self.assertEqual('"Books"', values[0])
        self.assertEqual('500', values[1])
        self.assertEqual('2', values[2])
        self.assertEqual('250', values[3])


if __name__ == '__main__':
    # Print whether chDB is available
    if CHDB_AVAILABLE:
        print("✅ chDB is available - running execution tests")
    else:
        print("⚠️  chDB not available - skipping execution tests")
        print("   Install with: pip install chdb")

    unittest.main()
