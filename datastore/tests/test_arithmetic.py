"""
Test arithmetic operations - migrated from pypika test_functions.py ArithmeticTests

Tests arithmetic operators (+, -, *, /, %, <<, >>) on fields and expressions.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.expressions import Literal


# ========== SQL Generation Tests ==========


class ArithmeticAdditionTests(unittest.TestCase):
    """Test addition operator"""

    def test_addition_fields(self):
        """Test adding two fields"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a + ds.b).to_sql()
        self.assertEqual('SELECT ("a"+"b") FROM "abc"', sql)

    def test_addition_number(self):
        """Test adding field and number"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a + 1).to_sql()
        self.assertEqual('SELECT ("a"+1) FROM "abc"', sql)

    def test_addition_decimal(self):
        """Test adding field and decimal"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a + 1.0).to_sql()
        self.assertEqual('SELECT ("a"+1.0) FROM "abc"', sql)

    def test_addition_right(self):
        """Test reverse addition (number + field)"""
        ds = DataStore(table="abc")
        sql = ds.select(1 + ds.a).to_sql()
        self.assertEqual('SELECT (1+"a") FROM "abc"', sql)


class ArithmeticSubtractionTests(unittest.TestCase):
    """Test subtraction operator"""

    def test_subtraction_fields(self):
        """Test subtracting two fields"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a - ds.b).to_sql()
        self.assertEqual('SELECT ("a"-"b") FROM "abc"', sql)

    def test_subtraction_number(self):
        """Test subtracting number from field"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a - 1).to_sql()
        self.assertEqual('SELECT ("a"-1) FROM "abc"', sql)

    def test_subtraction_decimal(self):
        """Test subtracting decimal from field"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a - 1.0).to_sql()
        self.assertEqual('SELECT ("a"-1.0) FROM "abc"', sql)

    def test_subtraction_right(self):
        """Test reverse subtraction (number - field)"""
        ds = DataStore(table="abc")
        sql = ds.select(1 - ds.a).to_sql()
        self.assertEqual('SELECT (1-"a") FROM "abc"', sql)


class ArithmeticMultiplicationTests(unittest.TestCase):
    """Test multiplication operator"""

    def test_multiplication_fields(self):
        """Test multiplying two fields"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a * ds.b).to_sql()
        self.assertEqual('SELECT ("a"*"b") FROM "abc"', sql)

    def test_multiplication_number(self):
        """Test multiplying field by number"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a * 1).to_sql()
        self.assertEqual('SELECT ("a"*1) FROM "abc"', sql)

    def test_multiplication_decimal(self):
        """Test multiplying field by decimal"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a * 1.0).to_sql()
        self.assertEqual('SELECT ("a"*1.0) FROM "abc"', sql)

    def test_multiplication_right(self):
        """Test reverse multiplication (number * field)"""
        ds = DataStore(table="abc")
        sql = ds.select(1 * ds.a).to_sql()
        self.assertEqual('SELECT (1*"a") FROM "abc"', sql)


class ArithmeticDivisionTests(unittest.TestCase):
    """Test division operator"""

    def test_division_fields(self):
        """Test dividing two fields"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a / ds.b).to_sql()
        self.assertEqual('SELECT ("a"/"b") FROM "abc"', sql)

    def test_division_number(self):
        """Test dividing field by number"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a / 1).to_sql()
        self.assertEqual('SELECT ("a"/1) FROM "abc"', sql)

    def test_division_decimal(self):
        """Test dividing field by decimal"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a / 1.0).to_sql()
        self.assertEqual('SELECT ("a"/1.0) FROM "abc"', sql)

    def test_division_right(self):
        """Test reverse division (number / field)"""
        ds = DataStore(table="abc")
        sql = ds.select(1 / ds.a).to_sql()
        self.assertEqual('SELECT (1/"a") FROM "abc"', sql)


class ArithmeticModuloTests(unittest.TestCase):
    """Test modulo operator"""

    def test_modulo_fields(self):
        """Test modulo of two fields"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a % ds.b).to_sql()
        self.assertEqual('SELECT ("a"%"b") FROM "abc"', sql)

    def test_modulo_number(self):
        """Test modulo with number"""
        ds = DataStore(table="abc")
        sql = ds.select(ds.a % 10).to_sql()
        self.assertEqual('SELECT ("a"%10) FROM "abc"', sql)


class ComplexArithmeticTests(unittest.TestCase):
    """Test complex arithmetic expressions"""

    def test_nested_arithmetic(self):
        """Test nested arithmetic operations"""
        ds = DataStore(table="abc")
        sql = ds.select((ds.a + ds.b) * ds.c).to_sql()
        self.assertEqual('SELECT (("a"+"b")*"c") FROM "abc"', sql)

    def test_arithmetic_with_parentheses(self):
        """Test arithmetic preserves operation order"""
        ds = DataStore(table="abc")
        sql = ds.select((ds.a + ds.b) / (ds.c - ds.d)).to_sql()
        self.assertEqual('SELECT (("a"+"b")/("c"-"d")) FROM "abc"', sql)

    def test_multiple_operations(self):
        """Test chain of operations"""
        ds = DataStore(table="abc")
        expr = ds.a + ds.b - ds.c * ds.d / ds.e
        sql = ds.select(expr).to_sql()
        # Should maintain operation order
        self.assertIn('+', sql)
        self.assertIn('-', sql)
        self.assertIn('*', sql)
        self.assertIn('/', sql)

    def test_arithmetic_with_literals(self):
        """Test arithmetic mixing fields and literals"""
        ds = DataStore(table="sales")
        # Calculate discounted price: price * (1 - discount_rate)
        expr = ds.price * (Literal(1) - ds.discount_rate)
        sql = ds.select(expr.as_("final_price")).to_sql()
        self.assertIn('*', sql)
        self.assertIn('-', sql)
        self.assertIn('1', sql)


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class ArithmeticExecutionTests(unittest.TestCase):
    """Test arithmetic operations execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_arithmetic (
            id UInt32,
            a Int32,
            b Int32,
            c Float64,
            d Float64
        ) ENGINE = Memory;
        
        INSERT INTO test_arithmetic VALUES
            (1, 10, 5, 2.5, 1.5),
            (2, 20, 4, 5.0, 2.0),
            (3, 15, 3, 3.0, 1.0);
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

    def test_addition_execution(self):
        """Test addition execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (ds.a + ds.b).as_("sum")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 10+5=15, 20+4=24, 15+3=18
        self.assertEqual(['1,15', '2,24', '3,18'], lines)

    def test_subtraction_execution(self):
        """Test subtraction execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (ds.a - ds.b).as_("diff")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 10-5=5, 20-4=16, 15-3=12
        self.assertEqual(['1,5', '2,16', '3,12'], lines)

    def test_multiplication_execution(self):
        """Test multiplication execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (ds.a * ds.b).as_("product")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 10*5=50, 20*4=80, 15*3=45
        self.assertEqual(['1,50', '2,80', '3,45'], lines)

    def test_division_execution(self):
        """Test division execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (ds.a / ds.b).as_("quotient")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 10/5=2, 20/4=5, 15/3=5
        self.assertEqual(['1,2', '2,5', '3,5'], lines)

    def test_complex_arithmetic_execution(self):
        """Test complex arithmetic expression execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", ((ds.a + ds.b) * ds.c).as_("result")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # (10+5)*2.5=37.5, (20+4)*5.0=120, (15+3)*3.0=54
        self.assertEqual(['1,37.5', '2,120', '3,54'], lines)

    def test_division_with_decimals_execution(self):
        """Test decimal division execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (ds.c / ds.d).as_("ratio")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 2.5/1.5≈1.667, 5.0/2.0=2.5, 3.0/1.0=3.0
        self.assertIn('1,', lines[0])
        self.assertIn('2,2.5', lines[1])
        self.assertIn('3,3', lines[2])

    def test_modulo_execution(self):
        """Test modulo execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (ds.a % ds.b).as_("remainder")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 10%5=0, 20%4=0, 15%3=0
        self.assertEqual(['1,0', '2,0', '3,0'], lines)

    def test_arithmetic_in_where_execution(self):
        """Test arithmetic in WHERE clause execution"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id").filter((ds.a + ds.b) > 20).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Only row 2 has a+b=24 > 20
        self.assertEqual(['2'], lines)

    def test_reverse_arithmetic_execution(self):
        """Test reverse arithmetic operations"""
        ds = DataStore(table="test_arithmetic")
        sql = ds.select("id", (Literal(100) - ds.a).as_("result")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # 100-10=90, 100-20=80, 100-15=85
        self.assertEqual(['1,90', '2,80', '3,85'], lines)


if __name__ == '__main__':
    unittest.main()
