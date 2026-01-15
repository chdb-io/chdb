#!/usr/bin/env python3
import unittest
import pandas as pd
import numpy as np
import chdb


def to_list(series):
    return [None if pd.isna(x) else x for x in series]


class TestDataFrameSliced(unittest.TestCase):

    def setUp(self):
        self.conn = chdb.connect()

    def tearDown(self):
        self.conn.close()

    def test_sliced_int(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7], 'a': [1, 2, 3, 4, 5, 6, 7, 8]})
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), [1, 3, 5, 7])

    def test_sliced_nullable_int(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([1, 2, None, 4, None, 6, 7, 8], dtype='Int64')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [1, None, None, 7])

    def test_sliced_float(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7], 'a': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]})
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), [1.1, 3.3, 5.5, 7.7])

    def test_sliced_float_with_nan(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, np.nan, 8.8]
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [1.1, 3.3, 5.5, None])

    def test_sliced_string(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7], 'a': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']})
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), ['a', 'c', 'e', 'g'])

    def test_sliced_string_with_none(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7], 'a': ['a', None, 'c', None, 'e', 'f', None, 'h']})
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), ['a', 'c', 'e', None])

    def test_sliced_datetime(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04',
                                 '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08'])
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        expected = [pd.Timestamp(d).date() for d in ['2024-01-01', '2024-01-03', '2024-01-05', '2024-01-07']]
        actual = [x.date() for x in result['a']]
        self.assertEqual(actual, expected)

    def test_sliced_datetime_with_nat(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.to_datetime(['2024-01-01', pd.NaT, '2024-01-03', pd.NaT,
                                 '2024-01-05', '2024-01-06', pd.NaT, '2024-01-08'])
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        expected = ['2024-01-01', '2024-01-03', '2024-01-05', None]
        for i, exp in enumerate(expected):
            if exp is None:
                self.assertTrue(pd.isna(result['a'].iloc[i]))
            else:
                self.assertEqual(result['a'].iloc[i].date(), pd.Timestamp(exp).date())

    def test_sliced_bool(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7], 'a': [True, False, True, False, True, False, False, False]})
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), [True, True, True, False])

    def test_sliced_nullable_bool(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([True, False, None, False, True, None, True, False], dtype='boolean')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [True, None, True, True])

    def test_fancy_indexing(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4], 'a': pd.array([10, 20, 30, 40, 50], dtype='Int64')})
        df_fancy = df.iloc[[0, 2, 4]]
        result = self.conn.query('SELECT a FROM Python(df_fancy) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), [10, 30, 50])

    def test_fancy_indexing_with_none(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4], 'a': pd.array([10, None, 30, None, 50], dtype='Int64')})
        df_fancy = df.iloc[[0, 1, 4]]
        result = self.conn.query('SELECT a FROM Python(df_fancy) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [10, None, 50])

    def test_mixed_columns_sliced(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5],
            'int_col': [1, 2, 3, 4, 5, 6],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
            'str_col': ['a', 'b', 'c', 'd', 'e', 'f'],
            'nullable_int': pd.array([1, None, 3, None, 5, 6], dtype='Int64'),
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT int_col, float_col, str_col, nullable_int FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['int_col'].tolist(), [1, 3, 5])
        self.assertEqual(result['float_col'].tolist(), [1.1, 3.3, 5.5])
        self.assertEqual(result['str_col'].tolist(), ['a', 'c', 'e'])
        self.assertEqual(to_list(result['nullable_int']), [1, 3, 5])

    def test_step_3_slice(self):
        df = pd.DataFrame({'id': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'a': pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='Int64')})
        df_sliced = df.iloc[::3]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), [1, 4, 7])

    def test_sliced_nullable_int8(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([1, 2, None, 4, None, 6, 7, 8], dtype='Int8')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [1, None, None, 7])

    def test_sliced_nullable_int16(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([100, 200, None, 400, None, 600, 700, 800], dtype='Int16')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [100, None, None, 700])

    def test_sliced_nullable_int32(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([1000, 2000, None, 4000, None, 6000, 7000, 8000], dtype='Int32')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [1000, None, None, 7000])

    def test_sliced_nullable_uint8(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([1, 2, None, 4, None, 6, 7, 8], dtype='UInt8')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [1, None, None, 7])

    def test_sliced_nullable_uint64(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([10000, 20000, None, 40000, None, 60000, 70000, 80000], dtype='UInt64')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [10000, None, None, 70000])

    def test_sliced_nullable_string(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': ['a', 'b', None, 'd', None, 'f', 'g', 'h']
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), ['a', None, None, 'g'])

    def test_sliced_all_null_mask(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'a': pd.array([None, None, None, None], dtype='Int64')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(to_list(result['a']), [None, None])

    def test_sliced_no_null_mask(self):
        df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6, 7],
            'a': pd.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='Int64')
        })
        df_sliced = df.iloc[::2]
        result = self.conn.query('SELECT a FROM Python(df_sliced) ORDER BY id', 'DataFrame')
        self.assertEqual(result['a'].tolist(), [1, 3, 5, 7])


if __name__ == '__main__':
    unittest.main(verbosity=2)
