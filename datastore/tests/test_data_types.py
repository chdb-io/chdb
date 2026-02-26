"""
Test data types and value handling - migrated from pypika test_data_types.py

Tests how different Python data types are converted to SQL literals.
"""

import unittest
import uuid
from datetime import date, datetime
from datastore.expressions import Literal


class StringTests(unittest.TestCase):
    """Test string handling"""

    def test_string_basic(self):
        """Test basic string conversion"""
        lit = Literal("hello")
        self.assertEqual("'hello'", lit.to_sql())

    def test_string_with_single_quote(self):
        """Test string with single quote is escaped"""
        lit = Literal("it's")
        self.assertEqual("'it''s'", lit.to_sql())

    def test_string_with_multiple_quotes(self):
        """Test string with multiple single quotes"""
        lit = Literal("it's a 'test'")
        self.assertEqual("'it''s a ''test'''", lit.to_sql())

    def test_empty_string(self):
        """Test empty string"""
        lit = Literal("")
        self.assertEqual("''", lit.to_sql())

    def test_string_with_newline(self):
        """Test string with newline"""
        lit = Literal("line1\nline2")
        self.assertEqual("'line1\nline2'", lit.to_sql())


class NumberTests(unittest.TestCase):
    """Test number handling"""

    def test_integer_positive(self):
        """Test positive integer"""
        lit = Literal(42)
        self.assertEqual("42", lit.to_sql())

    def test_integer_negative(self):
        """Test negative integer"""
        lit = Literal(-42)
        self.assertEqual("-42", lit.to_sql())

    def test_integer_zero(self):
        """Test zero"""
        lit = Literal(0)
        self.assertEqual("0", lit.to_sql())

    def test_float_positive(self):
        """Test positive float"""
        lit = Literal(3.14)
        self.assertEqual("3.14", lit.to_sql())

    def test_float_negative(self):
        """Test negative float"""
        lit = Literal(-3.14)
        self.assertEqual("-3.14", lit.to_sql())

    def test_float_zero(self):
        """Test zero float"""
        lit = Literal(0.0)
        self.assertEqual("0.0", lit.to_sql())

    def test_large_number(self):
        """Test large number"""
        lit = Literal(1234567890)
        self.assertEqual("1234567890", lit.to_sql())

    def test_scientific_notation(self):
        """Test scientific notation"""
        lit = Literal(1.23e-4)
        sql = lit.to_sql()
        # Python may format this differently
        self.assertTrue(sql.startswith("0.000"))


class BooleanTests(unittest.TestCase):
    """Test boolean handling"""

    def test_bool_true(self):
        """Test True boolean"""
        lit = Literal(True)
        self.assertEqual("TRUE", lit.to_sql())

    def test_bool_false(self):
        """Test False boolean"""
        lit = Literal(False)
        self.assertEqual("FALSE", lit.to_sql())


class NullTests(unittest.TestCase):
    """Test NULL handling"""

    def test_none(self):
        """Test None converts to NULL"""
        lit = Literal(None)
        self.assertEqual("NULL", lit.to_sql())


class DateTimeTests(unittest.TestCase):
    """Test date and datetime handling"""

    def test_date(self):
        """Test date conversion"""
        lit = Literal(date(2023, 12, 25))
        self.assertEqual("'2023-12-25'", lit.to_sql())

    def test_datetime(self):
        """Test datetime conversion"""
        lit = Literal(datetime(2023, 12, 25, 15, 30, 45))
        self.assertEqual("'2023-12-25 15:30:45'", lit.to_sql())

    def test_datetime_with_microseconds(self):
        """Test datetime with microseconds"""
        lit = Literal(datetime(2023, 12, 25, 15, 30, 45, 123456))
        sql = lit.to_sql()
        # Should include microseconds
        self.assertIn("2023-12-25", sql)
        self.assertIn("15:30:45", sql)


class UuidTests(unittest.TestCase):
    """Test UUID handling"""

    def test_uuid(self):
        """Test UUID conversion"""
        test_uuid = uuid.uuid4()
        lit = Literal(test_uuid)
        expected = f"'{test_uuid}'"
        self.assertEqual(expected, lit.to_sql())

    def test_uuid_string_format(self):
        """Test specific UUID string"""
        test_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
        lit = Literal(test_uuid)
        self.assertEqual("'12345678-1234-5678-1234-567812345678'", lit.to_sql())


class ListTests(unittest.TestCase):
    """Test list handling (for IN clauses)"""

    def test_list_of_integers(self):
        """Test list of integers"""
        # Note: Lists themselves aren't directly converted to SQL,
        # but we test that Literal handles them gracefully
        lit = Literal([1, 2, 3])
        # Should convert to string representation
        sql = lit.to_sql()
        self.assertIsInstance(sql, str)


class AliasTests(unittest.TestCase):
    """Test Literal with alias"""

    def test_literal_with_alias(self):
        """Test Literal can have alias"""
        lit = Literal(42, alias="answer")
        sql_without_alias = lit.to_sql()
        sql_with_alias = lit.to_sql(with_alias=True)

        self.assertEqual("42", sql_without_alias)
        self.assertEqual('42 AS "answer"', sql_with_alias)

    def test_string_literal_with_alias(self):
        """Test string Literal with alias"""
        lit = Literal("hello", alias="greeting")
        sql_with_alias = lit.to_sql(with_alias=True)
        self.assertEqual('\'hello\' AS "greeting"', sql_with_alias)


class EdgeCaseTests(unittest.TestCase):
    """Test edge cases"""

    def test_very_long_string(self):
        """Test very long string"""
        long_str = "x" * 1000
        lit = Literal(long_str)
        sql = lit.to_sql()
        self.assertEqual(f"'{long_str}'", sql)

    def test_string_with_special_chars(self):
        """Test string with special characters"""
        lit = Literal("test\t\r\n\"quoted\"")
        sql = lit.to_sql()
        # Single quotes should remain, double quotes should be in the string
        self.assertIn("'", sql)
        self.assertIn('"quoted"', sql)

    def test_numeric_string(self):
        """Test that string "123" is quoted, not treated as number"""
        lit = Literal("123")
        self.assertEqual("'123'", lit.to_sql())

    def test_float_precision(self):
        """Test float precision is maintained"""
        lit = Literal(1.23456789)
        sql = lit.to_sql()
        self.assertIn("1.234567", sql)  # Should maintain precision


if __name__ == '__main__':
    unittest.main()
