#!/usr/bin/env python3

import unittest
import chdb
from chdb import session


class TestDuplicateColumnNames(unittest.TestCase):
    """Test cases for queries with duplicate column names."""

    def test_duplicate_column_names_dataframe(self):
        """Test that DataFrame output preserves all columns with duplicate names."""
        result = chdb.query("SELECT count(), count(), count(), number FROM numbers(5) group by number", "DataFrame")
        self.assertEqual(result.shape, (5, 4))
        columns = list(result.columns)
        self.assertIn('count()', columns)
        self.assertIn('count()_1', columns)
        self.assertIn('count()_2', columns)
        self.assertIn('number', columns)
        self.assertTrue((result['count()'] == 1).all())
        self.assertTrue((result['count()_1'] == 1).all())
        self.assertTrue((result['count()_2'] == 1).all())

        result = chdb.query("SELECT count(), count(), count(), number as `count()_1` FROM numbers(5) group by number", "DataFrame")
        self.assertEqual(result.shape, (5, 4))

        result = chdb.query("SELECT count(), number as `count()_1`, count(), count() FROM numbers(5) group by number", "DataFrame")
        self.assertEqual(result.shape, (5, 4))

        result = chdb.query("SELECT count(), count(), count(), number as `COUNT()` FROM numbers(5) group by number order by number", "DataFrame")
        self.assertEqual(result.shape, (5, 4))
        columns = list(result.columns)
        self.assertIn('count()', columns)
        self.assertIn('count()_1', columns)
        self.assertIn('count()_2', columns)
        self.assertIn('COUNT()', columns)
        self.assertEqual(result['count()'].values[0], 1)
        self.assertEqual(result['count()_1'].values[0], 1)
        self.assertEqual(result['count()_2'].values[0], 1)
        self.assertEqual(result['COUNT()'].values.tolist(), [0, 1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main(verbosity=2)
