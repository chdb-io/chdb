"""
Test negation operations - migrated from pypika test_negation.py

Tests negation operator (-) on fields, expressions, and functions.
"""

import unittest

try:
    import chdb

    CHDB_AVAILABLE = True
except ImportError:
    CHDB_AVAILABLE = False

from datastore import DataStore, Field
from datastore.expressions import Literal
from datastore.functions import Sum


# ========== SQL Generation Tests ==========


class NegationTests(unittest.TestCase):
    """Test negation operator"""

    def test_negate_literal_int(self):
        """Test negating integer literal"""
        lit = -Literal(1)
        # DataStore implements negation as (0-x)
        self.assertEqual("(0-1)", lit.to_sql())

    def test_negate_literal_float(self):
        """Test negating float literal"""
        lit = -Literal(1.0)
        self.assertEqual("(0-1.0)", lit.to_sql())

    def test_negate_field(self):
        """Test negating field"""
        field = -Field("value")
        self.assertEqual('(0-"value")', field.to_sql())

    def test_negate_field_in_select(self):
        """Test negating field in SELECT"""
        ds = DataStore(table="test")
        sql = ds.select(-ds.value).to_sql()
        self.assertEqual('SELECT (0-"value") FROM "test"', sql)

    def test_negate_function(self):
        """Test negating function"""
        func = -Sum(Field("amount"))
        self.assertEqual('(0-SUM("amount"))', func.to_sql())

    def test_negate_function_in_select(self):
        """Test negating function in SELECT"""
        ds = DataStore(table="sales")
        sql = ds.select((-Sum(ds.amount)).as_("negative_total")).to_sql()
        self.assertEqual('SELECT (0-SUM("amount")) AS "negative_total" FROM "sales"', sql)

    def test_negate_arithmetic_expression(self):
        """Test negating arithmetic expression"""
        field = Field("a")
        expr = -(field + 10)
        self.assertEqual('(0-("a"+10))', expr.to_sql())

    def test_double_negation(self):
        """Test double negation"""
        field = Field("value")
        expr = -(-field)
        self.assertEqual('(0-(0-"value"))', expr.to_sql())

    def test_negate_in_arithmetic(self):
        """Test negation in arithmetic expression"""
        field = Field("value")
        expr = field + (-Literal(10))
        self.assertEqual('("value"+(0-10))', expr.to_sql())


# ========== Execution Tests with chdb ==========


@unittest.skipIf(not CHDB_AVAILABLE, "chDB not installed")
class NegationExecutionTests(unittest.TestCase):
    """Test negation execution on chdb"""

    @classmethod
    def setUpClass(cls):
        """Create test table"""
        cls.init_sql = """
        CREATE TABLE test_negation (
            id UInt32,
            value Int32,
            amount Float64
        ) ENGINE = Memory;
        
        INSERT INTO test_negation VALUES
            (1, 10, 100.5),
            (2, -20, -50.25),
            (3, 30, 75.0);
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

    def test_negate_field_execution(self):
        """Test negating field execution"""
        ds = DataStore(table="test_negation")
        sql = ds.select("id", (-ds.value).as_("neg_value")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Negations: -10, 20, -30
        self.assertEqual(['1,-10', '2,20', '3,-30'], lines)

    def test_negate_aggregate_execution(self):
        """Test negating aggregate function execution"""
        ds = DataStore(table="test_negation")
        sql = ds.select((-Sum(ds.value)).as_("neg_sum")).to_sql()
        result = self._execute(sql)
        # Sum: 10 + (-20) + 30 = 20, negated = -20
        self.assertEqual('-20', result)

    def test_arithmetic_with_negation_execution(self):
        """Test arithmetic with negation execution"""
        ds = DataStore(table="test_negation")
        sql = ds.select("id", (ds.value + (-Literal(5))).as_("result")).sort("id").to_sql()
        result = self._execute(sql)
        lines = result.split('\n')
        # Results: 10-5=5, -20-5=-25, 30-5=25
        self.assertEqual(['1,5', '2,-25', '3,25'], lines)


if __name__ == '__main__':
    unittest.main()
