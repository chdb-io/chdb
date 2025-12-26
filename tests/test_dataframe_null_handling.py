#!/usr/bin/env python3
import unittest
import pandas as pd
import numpy as np
import chdb


class TestDataFrameNullHandling(unittest.TestCase):
    """Test various null value handling between pandas and chdb"""

    def setUp(self):
        self.conn = chdb.connect()

    def tearDown(self):
        self.conn.close()

    def test_string_nan_is_null(self):
        """Test that Null in string column is recognized as NULL"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5, 6],
            'str_col': ['A', np.nan, pd.NaT, None, pd.NA, 'B'],
        })
        print('str_col dtype:', df['str_col'].dtype)

        result = self.conn.query('SELECT str_col, isNull(str_col) as is_null FROM Python(df) ORDER BY int_col', 'DataFrame')
        print("Result DataFrame:")
        print(result)

        self.assertEqual(result['str_col'].iloc[0], 'A')
        self.assertTrue(pd.isna(result['str_col'].iloc[1]))
        self.assertTrue(pd.isna(result['str_col'].iloc[2]))
        self.assertTrue(pd.isna(result['str_col'].iloc[3]))
        self.assertTrue(pd.isna(result['str_col'].iloc[4]))
        self.assertEqual(result['str_col'].iloc[5], 'B')

        self.assertEqual(result['is_null'].iloc[0], 0)
        self.assertEqual(result['is_null'].iloc[1], 1)
        self.assertEqual(result['is_null'].iloc[2], 1)
        self.assertEqual(result['is_null'].iloc[3], 1)
        self.assertEqual(result['is_null'].iloc[4], 1)
        self.assertEqual(result['is_null'].iloc[5], 0)

    def test_float_nan_is_null(self):
        """Test that Null in float column is recognized as NULL"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5, 6],
            'float_col1': [3.5, np.nan, pd.NaT, None, pd.NA, 6.7], # object
            'float_col2': [3.6, np.nan, np.nan, np.nan, np.nan, 6.8], # float
        })
        print('float_col1 dtype:', df['float_col1'].dtype)
        print('float_col2 dtype:', df['float_col2'].dtype)


        result = self.conn.query('SELECT float_col1, float_col2, isNull(float_col1) as is_null1 FROM Python(df) ORDER BY int_col', 'DataFrame')
        print("Result DataFrame:")
        print(result)

        self.assertEqual(result['float_col1'].iloc[0], 3.5)
        self.assertTrue(pd.isna(result['float_col1'].iloc[1]))
        self.assertTrue(pd.isna(result['float_col1'].iloc[2]))
        self.assertTrue(pd.isna(result['float_col1'].iloc[3]))
        self.assertTrue(pd.isna(result['float_col1'].iloc[4]))
        self.assertEqual(result['float_col1'].iloc[5], 6.7)

        self.assertEqual(result['float_col2'].iloc[0], 3.6)
        self.assertTrue(pd.isna(result['float_col2'].iloc[1]))
        self.assertTrue(pd.isna(result['float_col2'].iloc[2]))
        self.assertTrue(pd.isna(result['float_col2'].iloc[3]))
        self.assertTrue(pd.isna(result['float_col2'].iloc[4]))
        self.assertEqual(result['float_col2'].iloc[5], 6.8)

        self.assertEqual(result['is_null1'].iloc[0], 0)
        self.assertEqual(result['is_null1'].iloc[1], 1)
        self.assertEqual(result['is_null1'].iloc[2], 1)
        self.assertEqual(result['is_null1'].iloc[3], 1)
        self.assertEqual(result['is_null1'].iloc[4], 1)
        self.assertEqual(result['is_null1'].iloc[5], 0)

        sum_result = self.conn.query('SELECT sum(float_col1) as sum1, sum(float_col2) as sum2 FROM Python(df)', 'DataFrame')
        self.assertAlmostEqual(sum_result['sum1'].iloc[0], 10.2, places=5)  # 3.5 + 6.7
        self.assertAlmostEqual(sum_result['sum2'].iloc[0], 10.4, places=5)  # 3.6 + 6.8

    def test_int_nan_is_null(self):
        """Test that Null in int column is recognized as NULL"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5, 6],
            'int_col1': [3, np.nan, pd.NaT, None, pd.NA, 6], # object
            'int_col2': [4, np.nan, np.nan, np.nan, np.nan, 7], # float
        })
        print('int_col1 dtype:', df['int_col1'].dtype)
        print('int_col2 dtype:', df['int_col2'].dtype)

        result = self.conn.query('SELECT int_col1, int_col2, isNull(int_col1) as is_null1 FROM Python(df) ORDER BY int_col', 'DataFrame')
        print("Result DataFrame:")
        print(result)

        self.assertEqual(result['int_col1'].iloc[0], 3)
        self.assertTrue(pd.isna(result['int_col1'].iloc[1]))
        self.assertTrue(pd.isna(result['int_col1'].iloc[2]))
        self.assertTrue(pd.isna(result['int_col1'].iloc[3]))
        self.assertTrue(pd.isna(result['int_col1'].iloc[4]))
        self.assertEqual(result['int_col1'].iloc[5], 6)

        self.assertEqual(result['int_col2'].iloc[0], 4)
        self.assertTrue(pd.isna(result['int_col2'].iloc[1]))
        self.assertTrue(pd.isna(result['int_col2'].iloc[2]))
        self.assertTrue(pd.isna(result['int_col2'].iloc[3]))
        self.assertTrue(pd.isna(result['int_col2'].iloc[4]))
        self.assertEqual(result['int_col2'].iloc[5], 7)

        self.assertEqual(result['is_null1'].iloc[0], 0)
        self.assertEqual(result['is_null1'].iloc[1], 1)
        self.assertEqual(result['is_null1'].iloc[2], 1)
        self.assertEqual(result['is_null1'].iloc[3], 1)
        self.assertEqual(result['is_null1'].iloc[4], 1)
        self.assertEqual(result['is_null1'].iloc[5], 0)

    def test_int_and_float_nan_is_null(self):
        """Test that Null in int and float columns are recognized as NULL"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5, 6],
            'mixed_col1': [1, np.nan, pd.NaT, None, pd.NA, 6.7], # object
            'mixed_col2': [2, np.nan, np.nan, np.nan, np.nan, 6.8], # float
        })
        print('mixed_col1 dtype:', df['mixed_col1'].dtype)
        print('mixed_col2 dtype:', df['mixed_col2'].dtype)

        result = self.conn.query('SELECT mixed_col1, mixed_col2, isNull(mixed_col1) as is_null1 FROM Python(df) ORDER BY int_col', 'DataFrame')
        print("Result DataFrame:")
        print(result)

        self.assertEqual(result['mixed_col1'].iloc[0], 1)
        self.assertTrue(pd.isna(result['mixed_col1'].iloc[1]))
        self.assertTrue(pd.isna(result['mixed_col1'].iloc[2]))
        self.assertTrue(pd.isna(result['mixed_col1'].iloc[3]))
        self.assertTrue(pd.isna(result['mixed_col1'].iloc[4]))
        self.assertEqual(result['mixed_col1'].iloc[5], 6.7)

        self.assertEqual(result['mixed_col2'].iloc[0], 2)
        self.assertTrue(pd.isna(result['mixed_col2'].iloc[1]))
        self.assertTrue(pd.isna(result['mixed_col2'].iloc[2]))
        self.assertTrue(pd.isna(result['mixed_col2'].iloc[3]))
        self.assertTrue(pd.isna(result['mixed_col2'].iloc[4]))
        self.assertEqual(result['mixed_col2'].iloc[5], 6.8)

        self.assertEqual(result['is_null1'].iloc[0], 0)
        self.assertEqual(result['is_null1'].iloc[1], 1)
        self.assertEqual(result['is_null1'].iloc[2], 1)
        self.assertEqual(result['is_null1'].iloc[3], 1)
        self.assertEqual(result['is_null1'].iloc[4], 1)
        self.assertEqual(result['is_null1'].iloc[5], 0)

    def test_datetime_nan_is_null(self):
        """Test that Null in datetime column is recognized as NULL"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5, 6],
            'dt_col1': [pd.Timestamp('2024-01-01'), pd.NaT, pd.NaT, None, None, pd.Timestamp('2024-01-06')],  # datetime64[ns]
            'dt_col2': [np.datetime64('2024-01-02'), np.datetime64('NaT'), np.datetime64('NaT'), np.datetime64('NaT'), np.datetime64('NaT'), np.datetime64('2024-01-07')],  # datetime64[ns]
        })
        print('dt_col1 dtype:', df['dt_col1'].dtype)
        print('dt_col2 dtype:', df['dt_col2'].dtype)

        result = self.conn.query('SELECT dt_col1, dt_col2, isNull(dt_col1) as is_null1, isNull(dt_col2) as is_null2 FROM Python(df) ORDER BY int_col', 'DataFrame')
        print("Result DataFrame:")
        print(result)

        self.assertEqual(result['dt_col1'].iloc[0].date(), pd.Timestamp('2024-01-01').date())
        self.assertTrue(pd.isna(result['dt_col1'].iloc[1]))
        self.assertTrue(pd.isna(result['dt_col1'].iloc[2]))
        self.assertTrue(pd.isna(result['dt_col1'].iloc[3]))
        self.assertTrue(pd.isna(result['dt_col1'].iloc[4]))
        self.assertEqual(result['dt_col1'].iloc[5].date(), pd.Timestamp('2024-01-06').date())

        self.assertEqual(result['dt_col2'].iloc[0].date(), pd.Timestamp('2024-01-02').date())
        self.assertTrue(pd.isna(result['dt_col2'].iloc[1]))
        self.assertTrue(pd.isna(result['dt_col2'].iloc[2]))
        self.assertTrue(pd.isna(result['dt_col2'].iloc[3]))
        self.assertTrue(pd.isna(result['dt_col2'].iloc[4]))
        self.assertEqual(result['dt_col2'].iloc[5].date(), pd.Timestamp('2024-01-07').date())

        self.assertEqual(result['is_null1'].iloc[0], 0)
        self.assertEqual(result['is_null1'].iloc[1], 1)
        self.assertEqual(result['is_null1'].iloc[2], 1)
        self.assertEqual(result['is_null1'].iloc[3], 1)
        self.assertEqual(result['is_null1'].iloc[4], 1)
        self.assertEqual(result['is_null1'].iloc[5], 0)

        self.assertEqual(result['is_null2'].iloc[0], 0)
        self.assertEqual(result['is_null2'].iloc[1], 1)
        self.assertEqual(result['is_null2'].iloc[2], 1)
        self.assertEqual(result['is_null2'].iloc[3], 1)
        self.assertEqual(result['is_null2'].iloc[4], 1)
        self.assertEqual(result['is_null2'].iloc[5], 0)

    def test_json_nan_is_null(self):
        """Test that Null in JSON column is recognized as NULL"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5, 6],
            'json_col1': [{'a': 1}, np.nan, pd.NaT, None, pd.NA, {'b': 2}],
        })

        print('json_col1 dtype:', df['json_col1'].dtype)

        result = self.conn.query('SELECT json_col1.a as a, isNull(json_col1) as is_null1 FROM Python(df) ORDER BY int_col', 'DataFrame')
        print("Result DataFrame:")
        print(result)

        self.assertEqual(result['a'].iloc[0], 1)
        self.assertTrue(pd.isna(result['a'].iloc[1]))
        self.assertTrue(pd.isna(result['a'].iloc[2]))
        self.assertTrue(pd.isna(result['a'].iloc[3]))
        self.assertTrue(pd.isna(result['a'].iloc[4]))
        self.assertTrue(pd.isna(result['a'].iloc[5]))

        self.assertEqual(result['is_null1'].iloc[0], 0)
        self.assertEqual(result['is_null1'].iloc[1], 1)
        self.assertEqual(result['is_null1'].iloc[2], 1)
        self.assertEqual(result['is_null1'].iloc[3], 1)
        self.assertEqual(result['is_null1'].iloc[4], 1)
        self.assertEqual(result['is_null1'].iloc[5], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
