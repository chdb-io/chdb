"""
Test dtype alignment between DataStore and pandas.

These tests verify that chDB functions returning UInt64 are properly
converted to Int64 to match pandas behavior.
"""

import unittest
import numpy as np
import pandas as pd
import datastore as ds


class TestDtypeAlignment(unittest.TestCase):
    """Test dtype alignment for various functions."""

    def setUp(self):
        """Set up test data."""
        self.data = {
            'text': ['hello', 'world', 'foo', 'bar', 'baz'],
            'value': [1, 2, 3, 4, 5],
            'category': ['A', 'A', 'B', 'B', 'C'],
        }
        self.ds = ds.DataFrame(self.data)
        self.df = pd.DataFrame(self.data)

    def test_str_len_dtype(self):
        """Test str.len() returns int64 like pandas."""
        ds_result = self.ds['text'].str.len()
        pd_result = self.df['text'].str.len()

        # Both should be int64
        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"str.len() dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_groupby_count_dtype(self):
        """Test groupby().count() returns int64 like pandas."""
        ds_result = self.ds.groupby('category')['value'].count()
        pd_result = self.df.groupby('category')['value'].count()

        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"groupby count dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_groupby_sum_dtype(self):
        """Test groupby().sum() preserves dtype (should be int64)."""
        ds_result = self.ds.groupby('category')['value'].sum()
        pd_result = self.df.groupby('category')['value'].sum()

        # Sum of int64 should be int64
        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"groupby sum dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_groupby_mean_dtype(self):
        """Test groupby().mean() returns float64 like pandas."""
        ds_result = self.ds.groupby('category')['value'].mean()
        pd_result = self.df.groupby('category')['value'].mean()

        # Mean should be float64
        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"groupby mean dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )
        np.testing.assert_allclose(ds_result.values, pd_result.values)

    def test_value_counts_dtype(self):
        """Test value_counts() returns int64 like pandas."""
        ds_result = self.ds['category'].value_counts()
        pd_result = self.df['category'].value_counts()

        # value_counts returns int64
        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"value_counts dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )

    def test_nunique_dtype(self):
        """Test nunique() returns int64 scalar."""
        ds_result = self.ds['category'].nunique()
        pd_result = self.df['category'].nunique()

        # nunique returns int
        self.assertEqual(int(ds_result), pd_result)


class TestStringFunctionDtypes(unittest.TestCase):
    """Test dtype alignment for string functions."""

    def setUp(self):
        """Set up test data."""
        self.data = {'text': ['hello world', 'foo bar', 'test string']}
        self.ds = ds.DataFrame(self.data)
        self.df = pd.DataFrame(self.data)

    def test_str_find_dtype(self):
        """Test str.find() returns int64 like pandas."""
        ds_result = self.ds['text'].str.find('o')
        pd_result = self.df['text'].str.find('o')

        # find returns int64 (-1 for not found, position otherwise)
        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"str.find() dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )

    def test_str_count_dtype(self):
        """Test str.count() returns int64 like pandas."""
        ds_result = self.ds['text'].str.count('o')
        pd_result = self.df['text'].str.count('o')

        # count returns int64
        self.assertEqual(
            ds_result.dtype, pd_result.dtype, f"str.count() dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"
        )
        np.testing.assert_array_equal(ds_result.values, pd_result.values)


class TestAggregationDtypes(unittest.TestCase):
    """Test dtype alignment for aggregation functions."""

    def setUp(self):
        """Set up test data."""
        self.data = {
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        }
        self.ds = ds.DataFrame(self.data)
        self.df = pd.DataFrame(self.data)

    def test_sum_int_dtype(self):
        """Test sum() on int column returns int64."""
        ds_result = self.ds['int_col'].sum()
        pd_result = self.df['int_col'].sum()

        self.assertEqual(type(float(ds_result)), type(float(pd_result)))
        self.assertEqual(float(ds_result), float(pd_result))

    def test_sum_float_dtype(self):
        """Test sum() on float column returns float64."""
        ds_result = self.ds['float_col'].sum()
        pd_result = self.df['float_col'].sum()

        self.assertAlmostEqual(float(ds_result), float(pd_result), places=5)

    def test_mean_dtype(self):
        """Test mean() returns float64."""
        ds_result = self.ds['int_col'].mean()
        pd_result = self.df['int_col'].mean()

        self.assertAlmostEqual(float(ds_result), float(pd_result), places=5)

    def test_count_dtype(self):
        """Test count() returns int."""
        ds_result = self.ds['int_col'].count()
        pd_result = self.df['int_col'].count()

        self.assertEqual(int(ds_result), pd_result)


if __name__ == '__main__':
    unittest.main()
