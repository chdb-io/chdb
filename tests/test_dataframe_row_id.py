#!/usr/bin/env python3

import unittest
import chdb
import pandas as pd
import numpy as np

ROW_COUNT = 500000
df = pd.DataFrame({
    'id': np.arange(ROW_COUNT, dtype=np.int64),
    'value': ['a' if i % 2 == 0 else 'b' for i in range(ROW_COUNT)],
})


class TestDataFrameRowId(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = chdb.connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_row_id_basic(self):
        result = self.conn.query(
            "SELECT _row_id, id FROM Python(df) WHERE _row_id > 20 LIMIT 10 SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(len(result), 10)
        for i in range(10):
            self.assertEqual(result['_row_id'][i], result['id'][i])

    def test_row_id_at_different_positions(self):
        """Test _row_id at different column positions in SELECT."""
        # _row_id first
        result1 = self.conn.query(
            "SELECT _row_id, id, value FROM Python(df) LIMIT 5 SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(list(result1.columns), ['_row_id', 'id', 'value'])
        self.assertTrue((result1['_row_id'] == result1['id']).all())

        # _row_id middle
        result2 = self.conn.query(
            "SELECT id, _row_id, value FROM Python(df) LIMIT 5 SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(list(result2.columns), ['id', '_row_id', 'value'])
        self.assertTrue((result2['_row_id'] == result2['id']).all())

        # _row_id last
        result3 = self.conn.query(
            "SELECT id, value, _row_id FROM Python(df) LIMIT 5 SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(list(result3.columns), ['id', 'value', '_row_id'])
        self.assertTrue((result3['_row_id'] == result3['id']).all())

    def test_row_id_only(self):
        """Test selecting only _row_id column."""
        result = self.conn.query(
            "SELECT _row_id FROM Python(df) WHERE _row_id < 10 SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(len(result), 10)
        expected = list(range(10))
        actual = sorted(result['_row_id'].tolist())
        self.assertEqual(actual, expected)

    def test_row_id_type(self):
        """Test _row_id column type is UInt64."""
        result = self.conn.query(
            "SELECT toTypeName(_row_id) as type_name FROM Python(df) LIMIT 1 SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(result['type_name'][0], 'UInt64')

    def test_row_id_order_by(self):
        """Test ORDER BY _row_id."""
        result = self.conn.query(
            "SELECT _row_id, id FROM Python(df) ORDER BY _row_id SETTINGS max_threads=5",
            'DataFrame'
        )
        self.assertEqual(len(result), 500000)
        expected = list(range(500000))
        actual = result['_row_id'].tolist()
        self.assertEqual(actual, expected)
        self.assertTrue((result['_row_id'] == result['id']).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
