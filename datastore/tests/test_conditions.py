"""
Test condition system - converted from pypika test_criterions.py
"""

import unittest
from datastore.expressions import Field, Literal
from datastore.conditions import BinaryCondition, CompoundCondition, NotCondition, Condition


class TestBinaryCondition(unittest.TestCase):
    """Test binary comparison conditions."""

    def test_equal_condition(self):
        """Test equality condition."""
        cond = BinaryCondition('=', Field('age'), Literal(18))
        self.assertEqual('"age" = 18', cond.to_sql())

    def test_not_equal_condition(self):
        """Test inequality condition."""
        cond = BinaryCondition('!=', Field('status'), Literal('inactive'))
        self.assertEqual('"status" != \'inactive\'', cond.to_sql())

    def test_greater_than(self):
        """Test greater than condition."""
        cond = BinaryCondition('>', Field('price'), Literal(100))
        self.assertEqual('"price" > 100', cond.to_sql())

    def test_less_than_or_equal(self):
        """Test less than or equal condition."""
        cond = BinaryCondition('<=', Field('age'), Literal(65))
        self.assertEqual('"age" <= 65', cond.to_sql())


class TestCompoundCondition(unittest.TestCase):
    """Test compound conditions (AND, OR)."""

    def test_and_condition(self):
        """Test AND condition."""
        cond1 = Field('age') > 18
        cond2 = Field('age') < 65
        compound = CompoundCondition('AND', cond1, cond2)
        self.assertEqual('("age" > 18 AND "age" < 65)', compound.to_sql())

    def test_or_condition(self):
        """Test OR condition."""
        cond1 = Field('status') == 'active'
        cond2 = Field('status') == 'pending'
        compound = CompoundCondition('OR', cond1, cond2)
        self.assertEqual('("status" = \'active\' OR "status" = \'pending\')', compound.to_sql())

    def test_and_operator(self):
        """Test & operator for AND."""
        cond = (Field('age') > 18) & (Field('city') == 'NYC')
        self.assertEqual('("age" > 18 AND "city" = \'NYC\')', cond.to_sql())

    def test_or_operator(self):
        """Test | operator for OR."""
        cond = (Field('status') == 'active') | (Field('status') == 'pending')
        self.assertEqual('("status" = \'active\' OR "status" = \'pending\')', cond.to_sql())

    def test_complex_condition(self):
        """Test complex nested conditions."""
        cond = ((Field('age') > 18) & (Field('age') < 65)) | (Field('status') == 'premium')
        expected = '(("age" > 18 AND "age" < 65) OR "status" = \'premium\')'
        self.assertEqual(expected, cond.to_sql())


class TestNotCondition(unittest.TestCase):
    """Test NOT condition."""

    def test_not_condition(self):
        """Test NOT condition."""
        cond = NotCondition(Field('active') == True)
        self.assertEqual('ifNull(NOT ("active" = TRUE), 1)', cond.to_sql())

    def test_not_operator(self):
        """Test ~ operator for NOT."""
        cond = ~(Field('age') > 18)
        self.assertEqual('ifNull(NOT ("age" > 18), 1)', cond.to_sql())

    def test_not_compound(self):
        """Test NOT on compound condition."""
        cond = ~((Field('age') > 18) & (Field('city') == 'NYC'))
        self.assertEqual("ifNull(NOT ((\"age\" > 18 AND \"city\" = 'NYC')), 1)", cond.to_sql())


class TestConditionHelpers(unittest.TestCase):
    """Test Condition helper methods."""

    def test_all_single_condition(self):
        """Test Condition.all with single condition."""
        cond = Condition.all([Field('age') > 18])
        self.assertEqual('"age" > 18', cond.to_sql())

    def test_all_multiple_conditions(self):
        """Test Condition.all with multiple conditions."""
        conditions = [Field('age') > 18, Field('age') < 65, Field('status') == 'active']
        cond = Condition.all(conditions)
        expected = '(("age" > 18 AND "age" < 65) AND "status" = \'active\')'
        self.assertEqual(expected, cond.to_sql())

    def test_any_multiple_conditions(self):
        """Test Condition.any with multiple conditions."""
        conditions = [Field('status') == 'active', Field('status') == 'pending', Field('status') == 'trial']
        cond = Condition.any(conditions)
        expected = '(("status" = \'active\' OR "status" = \'pending\') OR "status" = \'trial\')'
        self.assertEqual(expected, cond.to_sql())


if __name__ == '__main__':
    unittest.main()

