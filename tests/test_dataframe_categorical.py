#!/usr/bin/env python3

import unittest
import pandas as pd
import chdb
import numpy as np
from datetime import datetime


class TestDataFrameCategorical(unittest.TestCase):
    def setUp(self):
        self.conn = chdb.connect()

    def tearDown(self):
        self.conn.close()

    def test_string_category(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical(['apple', 'banana', 'apple', 'cherry', 'banana'])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 5)
        self.assertEqual(result['category'].tolist(), ['apple', 'banana', 'apple', 'cherry', 'banana'])

    def test_string_category_with_null(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical(['apple', 'banana', None, 'cherry', None])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 5)
        values = result['category'].tolist()
        self.assertEqual(values[0], 'apple')
        self.assertEqual(values[1], 'banana')
        self.assertTrue(pd.isna(values[2]))
        self.assertEqual(values[3], 'cherry')
        self.assertTrue(pd.isna(values[4]))

    def test_int_category(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical([10, 20, 10, 30, 20])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 5)
        self.assertEqual(result['category'].tolist(), [10, 20, 10, 30, 20])

    def test_int_category_with_null(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical([10, 20, None, 30, None])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 5)
        values = result['category'].tolist()
        self.assertEqual(values[0], 10)
        self.assertEqual(values[1], 20)
        self.assertTrue(pd.isna(values[2]))
        self.assertEqual(values[3], 30)
        self.assertTrue(pd.isna(values[4]))

    def test_datetime_category(self):
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 6, 15),
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
            datetime(2024, 6, 15)
        ]
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical(dates)
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 5)
        result_dates = result['category'].tolist()
        self.assertEqual(result_dates[0], dates[0])
        self.assertEqual(result_dates[1], dates[1])
        self.assertEqual(result_dates[2], dates[2])
        self.assertEqual(result_dates[3], dates[3])
        self.assertEqual(result_dates[4], dates[4])

    def test_string_category_filter(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical(['apple', 'banana', 'apple', 'cherry', 'banana'])
        })

        result = self.conn.query(
            "SELECT * FROM Python(df) WHERE category = 'apple' ORDER BY id",
            "DataFrame"
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result['id'].tolist(), [1, 3])

    def test_string_category_group_by(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': pd.Categorical(['apple', 'banana', 'apple', 'cherry', 'banana']),
            'value': [10, 20, 30, 40, 50]
        })

        result = self.conn.query(
            "SELECT category, SUM(value) as total FROM Python(df) GROUP BY category ORDER BY category",
            "DataFrame"
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result['category'].tolist(), ['apple', 'banana', 'cherry'])
        self.assertEqual(result['total'].tolist(), [40, 70, 40])

    def test_mixed_type_category_error(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': pd.Categorical(['apple', 'banana', 3.14, 42])
        })

        with self.assertRaises(Exception) as context:
            self.conn.query("SELECT * FROM Python(df)", "DataFrame")

        self.assertIn("non-string categories", str(context.exception))

    def test_large_category_count(self):
        n = 1000
        categories = [f"cat_{i % 100}" for i in range(n)]
        df = pd.DataFrame({
            'id': list(range(n)),
            'category': pd.Categorical(categories)
        })

        result = self.conn.query("SELECT COUNT(DISTINCT category) as cnt FROM Python(df)", "DataFrame")

        self.assertEqual(result['cnt'].tolist()[0], 100)

    def test_category_with_special_chars(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': pd.Categorical(['hello world', 'foo\tbar', 'test\nnewline', 'quote"test'])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 4)
        values = result['category'].tolist()
        self.assertEqual(values[0], 'hello world')
        self.assertEqual(values[1], 'foo\tbar')
        self.assertEqual(values[2], 'test\nnewline')
        self.assertEqual(values[3], 'quote"test')

    def test_category_empty_string(self):
        df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'category': pd.Categorical(['apple', '', 'banana', ''])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 4)
        values = result['category'].tolist()
        self.assertEqual(values[0], 'apple')
        self.assertEqual(values[1], '')
        self.assertEqual(values[2], 'banana')
        self.assertEqual(values[3], '')

    def test_category_all_null(self):
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'category': pd.Categorical([None, None, None])
        })

        result = self.conn.query("SELECT * FROM Python(df) ORDER BY id", "DataFrame")

        self.assertEqual(len(result), 3)
        for val in result['category'].tolist():
            self.assertTrue(pd.isna(val))


if __name__ == '__main__':
    unittest.main()
