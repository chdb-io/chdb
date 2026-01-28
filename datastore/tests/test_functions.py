"""
Test function system - converted from pypika test_functions.py
"""

import unittest
from datastore.expressions import Field, Literal
from datastore.functions import Function, CustomFunction, Sum, Count, Avg, Min, Max, Upper, Lower, Concat


class TestBasicFunction(unittest.TestCase):
    """Test basic Function class."""

    def test_function_no_args(self):
        """Test function with no arguments."""
        func = Function('NOW')
        self.assertEqual('NOW()', func.to_sql())

    def test_function_one_arg(self):
        """Test function with one argument."""
        func = Function('UPPER', Field('name'))
        self.assertEqual('UPPER("name")', func.to_sql())

    def test_function_multiple_args(self):
        """Test function with multiple arguments."""
        func = Function('CONCAT', Field('first_name'), Literal(' '), Field('last_name'))
        self.assertEqual('CONCAT("first_name",\' \',"last_name")', func.to_sql())

    def test_function_with_alias(self):
        """Test function with alias."""
        func = Function('COUNT', Field('id'), alias='total')
        # COUNT is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('toInt64(COUNT("id")) AS "total"', func.to_sql(with_alias=True))


class TestAggregateFunction(unittest.TestCase):
    """Test aggregate functions."""

    def test_sum(self):
        """Test SUM function."""
        func = Sum(Field('amount'))
        self.assertEqual('SUM("amount")', func.to_sql())
        self.assertTrue(func.is_aggregate)

    def test_count(self):
        """Test COUNT function."""
        func = Count(Field('id'))
        # COUNT is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('toInt64(COUNT("id"))', func.to_sql())
        self.assertTrue(func.is_aggregate)

    def test_count_star(self):
        """Test COUNT(*) function."""
        func = Count('*')
        # COUNT is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('toInt64(COUNT(*))', func.to_sql())

    def test_avg(self):
        """Test AVG function."""
        func = Avg(Field('price'))
        self.assertEqual('AVG("price")', func.to_sql())

    def test_min(self):
        """Test MIN function."""
        func = Min(Field('price'))
        self.assertEqual('MIN("price")', func.to_sql())

    def test_max(self):
        """Test MAX function."""
        func = Max(Field('price'))
        self.assertEqual('MAX("price")', func.to_sql())

    def test_aggregate_with_alias(self):
        """Test aggregate function with alias."""
        func = Sum(Field('amount'), alias='total')
        self.assertEqual('SUM("amount") AS "total"', func.to_sql(with_alias=True))


class TestStringFunction(unittest.TestCase):
    """Test string functions."""

    def test_upper(self):
        """Test UPPER function."""
        func = Upper(Field('name'))
        self.assertEqual('UPPER("name")', func.to_sql())

    def test_lower(self):
        """Test LOWER function."""
        func = Lower(Field('name'))
        self.assertEqual('LOWER("name")', func.to_sql())

    def test_concat(self):
        """Test CONCAT function."""
        func = Concat(Field('first_name'), Literal(' '), Field('last_name'))
        self.assertEqual('CONCAT("first_name",\' \',"last_name")', func.to_sql())


class TestCustomFunction(unittest.TestCase):
    """Test CustomFunction factory."""

    def test_custom_function_no_params(self):
        """Test custom function without parameter validation."""
        MyFunc = CustomFunction('MY_FUNC')
        func = MyFunc(Field('x'), Field('y'))
        self.assertEqual('MY_FUNC("x","y")', func.to_sql())

    def test_custom_function_with_params(self):
        """Test custom function with parameter validation."""
        DateDiff = CustomFunction('DATE_DIFF', ['interval', 'start_date', 'end_date'])
        func = DateDiff(Literal('day'), Field('created_at'), Field('updated_at'))
        self.assertEqual('DATE_DIFF(\'day\',"created_at","updated_at")', func.to_sql())

    def test_custom_function_wrong_param_count(self):
        """Test custom function with wrong parameter count."""
        DateDiff = CustomFunction('DATE_DIFF', ['interval', 'start', 'end'])
        with self.assertRaises(Exception):
            DateDiff(Literal('day'), Field('created_at'))  # Missing one argument

    def test_custom_function_with_alias(self):
        """Test custom function with alias."""
        MyFunc = CustomFunction('MY_FUNC')
        func = MyFunc(Field('x'), alias='result')
        self.assertEqual('MY_FUNC("x") AS "result"', func.to_sql(with_alias=True))


class TestFunctionComposition(unittest.TestCase):
    """Test composing functions."""

    def test_function_in_arithmetic(self):
        """Test function in arithmetic expression."""
        expr = Sum(Field('amount')) / Count(Field('id'))
        # COUNT is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual('(SUM("amount")/toInt64(COUNT("id")))', expr.to_sql())

    def test_nested_functions(self):
        """Test nested function calls."""
        inner = Upper(Field('name'))
        outer = Concat(inner, Literal(' Jr.'))
        self.assertEqual('CONCAT(UPPER("name"),\' Jr.\')', outer.to_sql())


if __name__ == '__main__':
    unittest.main()
