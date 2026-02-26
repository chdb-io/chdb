"""
Tests for dynamic Pandas method invocation.

When a function is not registered, DataStore tries to call corresponding Pandas method.
"""

import unittest
from datastore import DataStore as ds
from datastore.functions import Function
from datastore.expressions import Field


class TestDynamicPandasMethods(unittest.TestCase):
    """Test dynamic Pandas method invocation for unregistered functions."""

    def test_str_removeprefix(self):
        """Test str.removeprefix - unregistered, uses dynamic invocation."""
        nat = ds.from_file('tests/dataset/users.csv').limit(3)
        nat['no_prefix'] = Function('removeprefix', Field('name'), 'Alice ')
        df = nat.to_df()
        self.assertEqual(df[df['name'] == 'Alice Smith']['no_prefix'].iloc[0], 'Smith')

    def test_str_removesuffix(self):
        """Test str.removesuffix - unregistered, uses dynamic invocation."""
        nat = ds.from_file('tests/dataset/users.csv').limit(3)
        nat['no_suffix'] = Function('removesuffix', Field('name'), ' Smith')
        df = nat.to_df()
        self.assertEqual(df[df['name'] == 'Alice Smith']['no_suffix'].iloc[0], 'Alice')

    def test_series_add(self):
        """Test Series.add - unregistered, uses dynamic invocation."""
        nat = ds.from_numbers(5)
        nat['plus_100'] = Function('add', Field('number'), 100)
        df = nat.to_df()
        self.assertEqual(df['plus_100'].tolist(), [100, 101, 102, 103, 104])

    def test_series_mul(self):
        """Test Series.mul - unregistered, uses dynamic invocation."""
        nat = ds.from_numbers(5)
        nat['times_10'] = Function('mul', Field('number'), 10)
        df = nat.to_df()
        self.assertEqual(df['times_10'].tolist(), [0, 10, 20, 30, 40])

    def test_series_pow(self):
        """Test Series.pow - unregistered, uses dynamic invocation."""
        nat = ds.from_numbers(5)
        nat['squared'] = Function('pow', Field('number'), 2)
        df = nat.to_df()
        self.assertEqual([int(x) for x in df['squared'].tolist()], [0, 1, 4, 9, 16])

    def test_str_slice(self):
        """Test str.slice - unregistered, uses dynamic invocation."""
        nat = ds.from_file('tests/dataset/users.csv').limit(3)
        # slice(0, 5) gets first 5 characters
        nat['first5'] = Function('slice', Field('name'), 0, 5)
        df = nat.to_df()
        self.assertEqual(df['first5'].iloc[0], 'Alice')  # 'Alice Smith' -> 'Alice'


class TestDynamicFallbackToChdb(unittest.TestCase):
    """Test that unknown functions fallback to chDB."""

    def test_chdb_specific_function(self):
        """Test ClickHouse-specific function via fallback."""
        nat = ds.from_file('tests/dataset/users.csv').orderby('user_id').limit(3)
        nat['name_len'] = Function('lengthUTF8', Field('name'))
        df = nat.to_df()
        self.assertEqual(df['name_len'].iloc[0], 11)  # 'Alice Smith'

    def test_nonexistent_function_raises_error(self):
        """Test that a non-existent function raises an error."""
        nat = ds.from_file('tests/dataset/users.csv').limit(2)
        nat['bad'] = Function('totally_fake_xyz', Field('name'))
        with self.assertRaises(Exception):
            nat.to_df()


class TestDynamicWithNumbers(unittest.TestCase):
    """Test dynamic method on numbers table."""

    def test_series_floordiv(self):
        """Test Series.floordiv - unregistered, uses dynamic invocation."""
        nat = ds.from_numbers(5)
        nat['value'] = Field('number') + 10  # 10, 11, 12, 13, 14
        nat['div3'] = Function('floordiv', Field('value'), 3)
        df = nat.to_df()
        self.assertEqual(df['div3'].tolist(), [3, 3, 4, 4, 4])

    def test_series_mod_via_chdb(self):
        """Test modulo - falls back to chDB since mod exists there."""
        nat = ds.from_numbers(5)
        nat['value'] = Field('number') + 10  # 10, 11, 12, 13, 14
        nat['mod3'] = Function('modulo', Field('value'), 3)  # ClickHouse function
        df = nat.to_df()
        self.assertEqual(df['mod3'].tolist(), [1, 2, 0, 1, 2])


if __name__ == '__main__':
    unittest.main()
