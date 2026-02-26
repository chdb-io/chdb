"""
Test expression system with chdb execution - converted from pypika test_terms.py

Tests both SQL generation AND actual execution on chdb.
"""

import unittest
from datastore import DataStore
from datastore.expressions import Field, Literal, ArithmeticExpression
from datastore.conditions import BinaryCondition


class TestField(unittest.TestCase):
    """Test Field class with SQL generation."""

    def test_field_simple(self):
        """Test basic field creation."""
        f = Field('name')
        self.assertEqual('"name"', f.to_sql())

    def test_field_with_table(self):
        """Test field with table prefix."""
        f = Field('name', table='customers')
        self.assertEqual('"customers"."name"', f.to_sql())

    def test_field_with_alias(self):
        """Test field with alias."""
        f = Field('name', alias='customer_name')
        self.assertEqual('"name" AS "customer_name"', f.to_sql(with_alias=True))

    def test_field_without_quotes(self):
        """Test field without quote characters."""
        f = Field('name')
        self.assertEqual('name', f.to_sql(quote_char=''))


class TestLiteral(unittest.TestCase):
    """Test Literal class."""

    def test_literal_int(self):
        """Test integer literal."""
        lit = Literal(42)
        self.assertEqual('42', lit.to_sql())

    def test_literal_float(self):
        """Test float literal."""
        lit = Literal(3.14)
        self.assertEqual('3.14', lit.to_sql())

    def test_literal_string(self):
        """Test string literal."""
        lit = Literal('hello')
        self.assertEqual("'hello'", lit.to_sql())

    def test_literal_string_with_quotes(self):
        """Test string literal with quotes (escaping)."""
        lit = Literal("it's")
        self.assertEqual("'it''s'", lit.to_sql())

    def test_literal_none(self):
        """Test NULL literal."""
        lit = Literal(None)
        self.assertEqual('NULL', lit.to_sql())

    def test_literal_bool_true(self):
        """Test boolean TRUE literal."""
        lit = Literal(True)
        self.assertEqual('TRUE', lit.to_sql())

    def test_literal_bool_false(self):
        """Test boolean FALSE literal."""
        lit = Literal(False)
        self.assertEqual('FALSE', lit.to_sql())

    def test_literal_with_expression_value(self):
        """Test Literal wrapping an Expression delegates to_sql correctly."""
        # This can happen in edge cases - Literal should handle it gracefully
        inner_field = Field('age')
        lit = Literal(inner_field)
        # Should delegate to the expression's to_sql
        self.assertEqual('"age"', lit.to_sql())

    def test_literal_with_column_expr_value(self):
        """Test Literal wrapping a ColumnExpr delegates to_sql without infinite recursion."""
        import pandas as pd
        from datastore.column_expr import ColumnExpr
        from datastore.lazy_ops import LazyDataFrameSource

        # Create a ColumnExpr
        df = pd.DataFrame({'value': [1, 2, 3]})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df)]
        col_expr = ds['value']

        # Wrap in Literal (edge case that could cause infinite recursion)
        lit = Literal(col_expr)

        # Should NOT cause infinite recursion - should delegate to ColumnExpr.to_sql()
        sql = lit.to_sql()
        self.assertEqual('"value"', sql)

    def test_literal_with_nested_expression(self):
        """Test Literal with nested arithmetic expression."""
        # Create (a + 1)
        expr = ArithmeticExpression('+', Field('a'), Literal(1))
        lit = Literal(expr)
        # Should delegate to expression's to_sql
        self.assertEqual('("a"+1)', lit.to_sql())


class TestArithmeticExpression(unittest.TestCase):
    """Test arithmetic operations."""

    def test_addition(self):
        """Test addition operation."""
        expr = ArithmeticExpression('+', Field('a'), Literal(1))
        self.assertEqual('("a"+1)', expr.to_sql())

    def test_subtraction(self):
        """Test subtraction operation."""
        expr = ArithmeticExpression('-', Field('a'), Literal(1))
        self.assertEqual('("a"-1)', expr.to_sql())

    def test_multiplication(self):
        """Test multiplication operation."""
        expr = ArithmeticExpression('*', Field('a'), Literal(2))
        self.assertEqual('("a"*2)', expr.to_sql())

    def test_division(self):
        """Test division operation."""
        expr = ArithmeticExpression('/', Field('a'), Literal(2))
        self.assertEqual('("a"/2)', expr.to_sql())

    def test_nested_arithmetic(self):
        """Test nested arithmetic expressions."""
        # (a + 1) * 2
        inner = ArithmeticExpression('+', Field('a'), Literal(1))
        outer = ArithmeticExpression('*', inner, Literal(2))
        self.assertEqual('(("a"+1)*2)', outer.to_sql())


class TestOperatorOverloading(unittest.TestCase):
    """Test operator overloading on expressions."""

    def test_addition_operator(self):
        """Test + operator."""
        expr = Field('a') + 1
        self.assertIsInstance(expr, ArithmeticExpression)
        self.assertEqual('("a"+1)', expr.to_sql())

    def test_subtraction_operator(self):
        """Test - operator."""
        expr = Field('a') - 1
        self.assertEqual('("a"-1)', expr.to_sql())

    def test_multiplication_operator(self):
        """Test * operator."""
        expr = Field('price') * 1.1
        self.assertEqual('("price"*1.1)', expr.to_sql())

    def test_division_operator(self):
        """Test / operator."""
        expr = Field('total') / 2
        self.assertEqual('("total"/2)', expr.to_sql())

    def test_reverse_addition(self):
        """Test reverse addition (1 + field)."""
        expr = 100 + Field('price')
        self.assertEqual('(100+"price")', expr.to_sql())

    def test_comparison_equal(self):
        """Test == operator."""
        cond = Field('age') == 18
        self.assertIsInstance(cond, BinaryCondition)
        self.assertEqual('"age" = 18', cond.to_sql())

    def test_comparison_not_equal(self):
        """Test != operator."""
        cond = Field('status') != 'inactive'
        self.assertEqual('"status" != \'inactive\'', cond.to_sql())

    def test_comparison_greater_than(self):
        """Test > operator."""
        cond = Field('age') > 18
        self.assertEqual('"age" > 18', cond.to_sql())

    def test_comparison_less_than(self):
        """Test < operator."""
        cond = Field('price') < 100
        self.assertEqual('"price" < 100', cond.to_sql())

    def test_chained_operations(self):
        """Test chained operations."""
        expr = (Field('a') + 1) * 2 - 3
        self.assertEqual('((("a"+1)*2)-3)', expr.to_sql())


class TestArithmeticExecution(unittest.TestCase):
    """Test arithmetic expressions with chdb execution."""

    def setUp(self):
        """Create test table with numeric data."""
        self.ds = DataStore(table="numbers")
        self.ds.connect()
        self.ds.create_table({"a": "Int32", "b": "Int32", "c": "Float64", "name": "String"})
        self.ds.insert(
            [
                {"a": 10, "b": 5, "c": 3.5, "name": "row1"},
                {"a": 20, "b": 10, "c": 7.2, "name": "row2"},
                {"a": 30, "b": 15, "c": 12.8, "name": "row3"},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS numbers")
        except Exception:
            pass
        self.ds.close()

    def test_addition_execution(self):
        """Test addition with execution."""
        result = self.ds.select((self.ds.a + self.ds.b).as_("sum")).execute()

        # Verify SQL
        sql = self.ds.select((self.ds.a + self.ds.b).as_("sum")).to_sql()
        self.assertIn("+", sql)

        # Verify execution results
        self.assertEqual(3, len(result))
        self.assertIn("sum", result.column_names)

        # Check calculated values
        sums = [row[0] for row in result.rows]
        self.assertEqual([15, 30, 45], sums)  # 10+5, 20+10, 30+15

    def test_subtraction_execution(self):
        """Test subtraction with execution."""
        result = self.ds.select((self.ds.a - self.ds.b).as_("diff")).execute()

        # Verify results
        self.assertEqual(3, len(result))
        diffs = [row[0] for row in result.rows]
        self.assertEqual([5, 10, 15], diffs)  # 10-5, 20-10, 30-15

    def test_multiplication_execution(self):
        """Test multiplication with execution."""
        result = self.ds.select((self.ds.a * 2).as_("double")).execute()

        # Verify results
        self.assertEqual(3, len(result))
        doubles = [row[0] for row in result.rows]
        self.assertEqual([20, 40, 60], doubles)  # 10*2, 20*2, 30*2

    def test_division_execution(self):
        """Test division with execution."""
        result = self.ds.select((self.ds.a / 2).as_("half")).execute()

        # Verify results
        self.assertEqual(3, len(result))
        halves = [row[0] for row in result.rows]
        self.assertEqual([5, 10, 15], halves)  # 10/2, 20/2, 30/2

    def test_complex_arithmetic_execution(self):
        """Test complex arithmetic expression with execution."""
        # (a + b) * 2 - 5
        expr = ((self.ds.a + self.ds.b) * 2 - 5).as_("result")
        result = self.ds.select(expr).execute()

        # Verify results
        self.assertEqual(3, len(result))
        results = [row[0] for row in result.rows]
        # row1: (10+5)*2-5 = 25
        # row2: (20+10)*2-5 = 55
        # row3: (30+15)*2-5 = 85
        self.assertEqual([25, 55, 85], results)

    def test_float_arithmetic_execution(self):
        """Test float arithmetic with execution."""
        result = self.ds.select((self.ds.c * 2).as_("doubled")).execute()

        # Verify results
        self.assertEqual(3, len(result))
        doubled = [row[0] for row in result.rows]
        # Allow for floating point precision
        self.assertAlmostEqual(7.0, doubled[0], places=1)  # 3.5*2
        self.assertAlmostEqual(14.4, doubled[1], places=1)  # 7.2*2
        self.assertAlmostEqual(25.6, doubled[2], places=1)  # 12.8*2

    def test_mixed_field_and_literal_execution(self):
        """Test mixing fields and literals in arithmetic."""
        result = self.ds.select((self.ds.a + 100).as_("plus100")).execute()

        # Verify results
        self.assertEqual(3, len(result))
        results = [row[0] for row in result.rows]
        self.assertEqual([110, 120, 130], results)  # 10+100, 20+100, 30+100


class TestComparisonExecution(unittest.TestCase):
    """Test comparison operations with chdb execution."""

    def setUp(self):
        """Create test table."""
        self.ds = DataStore(table="data")
        self.ds.connect()
        self.ds.create_table({"id": "UInt32", "value": "Int32", "status": "String"})
        self.ds.insert(
            [
                {"id": 1, "value": 100, "status": "active"},
                {"id": 2, "value": 50, "status": "inactive"},
                {"id": 3, "value": 150, "status": "active"},
                {"id": 4, "value": 75, "status": "pending"},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS data")
        except Exception:
            pass
        self.ds.close()

    def test_equal_comparison_execution(self):
        """Test == comparison with execution."""
        result = self.ds.select("*").filter(self.ds.value == 100).execute()

        # Verify SQL
        sql = self.ds.select("*").filter(self.ds.value == 100).to_sql()
        self.assertIn("=", sql)
        self.assertIn("100", sql)

        # Verify results
        self.assertEqual(1, len(result))
        self.assertEqual(100, result.rows[0][1])  # value column

    def test_not_equal_comparison_execution(self):
        """Test != comparison with execution."""
        result = self.ds.select("*").filter(self.ds.status != "active").execute()

        # Verify results
        self.assertEqual(2, len(result))  # inactive and pending

        # Check none of the results have "active" status
        for row in result.rows:
            status_idx = result.column_names.index("status")
            self.assertNotEqual("active", row[status_idx])

    def test_greater_than_execution(self):
        """Test > comparison with execution."""
        result = self.ds.select("*").filter(self.ds.value > 75).execute()

        # Verify results
        self.assertEqual(2, len(result))  # 100 and 150

        # Check all results have value > 75
        for row in result.rows:
            value_idx = result.column_names.index("value")
            self.assertGreater(row[value_idx], 75)

    def test_less_than_execution(self):
        """Test < comparison with execution."""
        result = self.ds.select("*").filter(self.ds.value < 100).execute()

        # Verify results
        self.assertEqual(2, len(result))  # 50 and 75

        # Check all results have value < 100
        for row in result.rows:
            value_idx = result.column_names.index("value")
            self.assertLess(row[value_idx], 100)

    def test_greater_or_equal_execution(self):
        """Test >= comparison with execution."""
        result = self.ds.select("*").filter(self.ds.value >= 100).execute()

        # Verify results
        self.assertEqual(2, len(result))  # 100 and 150

    def test_less_or_equal_execution(self):
        """Test <= comparison with execution."""
        result = self.ds.select("*").filter(self.ds.value <= 75).execute()

        # Verify results
        self.assertEqual(2, len(result))  # 50 and 75

    def test_string_comparison_execution(self):
        """Test string comparison with execution."""
        result = self.ds.select("*").filter(self.ds.status == "active").execute()

        # Verify results
        self.assertEqual(2, len(result))

        # Check all results have "active" status
        for row in result.rows:
            status_idx = result.column_names.index("status")
            self.assertEqual("active", row[status_idx])


class TestComplexExpressionExecution(unittest.TestCase):
    """Test complex expressions combining arithmetic and comparisons."""

    def setUp(self):
        """Create test table."""
        self.ds = DataStore(table="sales")
        self.ds.connect()
        self.ds.create_table({"price": "Float64", "quantity": "UInt32", "discount": "Float64"})
        self.ds.insert(
            [
                {"price": 100.0, "quantity": 2, "discount": 0.1},
                {"price": 50.0, "quantity": 5, "discount": 0.2},
                {"price": 200.0, "quantity": 1, "discount": 0.0},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS sales")
        except Exception:
            pass
        self.ds.close()

    def test_calculated_total_execution(self):
        """Test calculating total with discount."""
        # total = price * quantity * (1 - discount)
        total_expr = (self.ds.price * self.ds.quantity * (1 - self.ds.discount)).as_("total")
        result = self.ds.select(total_expr).execute()

        # Verify results
        self.assertEqual(3, len(result))
        totals = [row[0] for row in result.rows]

        # row1: 100 * 2 * (1 - 0.1) = 180
        # row2: 50 * 5 * (1 - 0.2) = 200
        # row3: 200 * 1 * (1 - 0.0) = 200
        self.assertAlmostEqual(180.0, totals[0], places=1)
        self.assertAlmostEqual(200.0, totals[1], places=1)
        self.assertAlmostEqual(200.0, totals[2], places=1)

    def test_filter_with_calculated_value(self):
        """Test filtering based on calculated value."""
        # Find items where price * quantity > 150
        result = self.ds.select("*").filter((self.ds.price * self.ds.quantity) > 150).execute()

        # Verify results
        # row1: 100 * 2 = 200 > 150 ✓
        # row2: 50 * 5 = 250 > 150 ✓
        # row3: 200 * 1 = 200 > 150 ✓
        self.assertEqual(3, len(result))

    def test_select_multiple_calculations(self):
        """Test selecting multiple calculated fields."""
        result = self.ds.select(
            (self.ds.price * self.ds.quantity).as_("subtotal"),
            (self.ds.price * self.ds.quantity * self.ds.discount).as_("discount_amount"),
        ).execute()

        # Verify results
        self.assertEqual(3, len(result))
        self.assertEqual(["subtotal", "discount_amount"], result.column_names)

        # Check first row calculations
        # subtotal: 100 * 2 = 200
        # discount_amount: 100 * 2 * 0.1 = 20
        self.assertAlmostEqual(200.0, result.rows[0][0], places=1)
        self.assertAlmostEqual(20.0, result.rows[0][1], places=1)


class TestFieldAliasExecution(unittest.TestCase):
    """Test field aliases in execution."""

    def setUp(self):
        """Create test table."""
        self.ds = DataStore(table="users")
        self.ds.connect()
        self.ds.create_table({"first_name": "String", "last_name": "String", "age": "UInt8"})
        self.ds.insert(
            [
                {"first_name": "John", "last_name": "Doe", "age": 30},
                {"first_name": "Jane", "last_name": "Smith", "age": 25},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation."""
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS users")
        except Exception:
            pass
        self.ds.close()

    def test_field_alias_in_result(self):
        """Test that field aliases appear in result column names."""
        result = self.ds.select(self.ds.first_name.as_("fname"), self.ds.last_name.as_("lname")).execute()

        # Verify column names
        self.assertIn("fname", result.column_names)
        self.assertIn("lname", result.column_names)

        # Verify data
        self.assertEqual(2, len(result))

    def test_expression_alias_in_result(self):
        """Test that expression aliases work in results."""
        result = self.ds.select((self.ds.age + 10).as_("age_plus_10")).execute()

        # Verify column name
        self.assertIn("age_plus_10", result.column_names)

        # Verify calculated values
        self.assertEqual(2, len(result))
        ages = [row[0] for row in result.rows]
        self.assertEqual([40, 35], ages)  # 30+10, 25+10


if __name__ == '__main__':
    unittest.main()
