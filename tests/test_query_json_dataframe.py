#!python3

import math
import random
import uuid
import unittest
import numpy as np
import pandas as pd
import chdb.session as chs
from datetime import date, datetime, time, timedelta
from decimal import Decimal


EXPECTED2 = '"apple1",3,\\N\n\\N,4,2\n'

complex_dict = {
    "date_col": [
        {"date": date(2023, 5, 15)},
        {"date": date(2024, 1, 1)}
    ],
    'datetime_col': [
        {"datetime": datetime(2024, 5, 30, 14, 30)},
        {"datetime": datetime(2023, 12, 31, 23, 59, 59)}
    ],
    'time_col': [
        {"time": time(12, 30)},
        {"time": time(0, 0)}
    ],
    'timedelta_col': [
        {"timedelta": timedelta(days=1, hours=12, minutes=30)},
        {"timedelta": timedelta(days=2, hours=3, minutes=20)}
    ],
    'bytes_col': [
        {"bytes": b'\xe8\x8b\xb9\xe6\x9e\x9c'},
        {"bytes": bytes([0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79])}
    ],
    'bytearray_col': [
        {"bytearray": bytearray(b"bytearray1")},
        {"bytearray": bytearray('苹果'.encode('utf-8'))}
    ],
    'memoryview_col': [
        {"memoryview": memoryview(b"memory1")},
        {"memoryview": memoryview('苹果'.encode('utf-8'))}
    ],
    'uuid_col': [
        {"uuid": uuid.UUID('00000000-0000-0000-0000-000000000001')},
        {"uuid": uuid.UUID('00000000-0000-0000-0000-000000000000')}
    ],
    "string_col": [
        {"str": "hello"},
        {"str": "苹果"}
    ],
    'special_values_col': [
        {"null_val": None, "bool_val": True, "empty_list": [], "empty_dict": {}, "empty_array": np.array([])},
    ],
    'special_num_col': [
        {"special_num_val": [0.0, -1.1, float('nan'), float('inf'), float('-inf'), math.inf, math.nan]}
    ],
    'numpy_num_col': [
        {"numpy_val": np.array([np.int32(42), np.int64(-1), np.float32(3.14), np.float64(2.718)])},
    ],
    'numpy_bool_col': [
        {"numpy_val": np.array([np.bool_(True), np.bool_(False)])},
    ],
    'numpy_datetime_col': [
        {"numpy_val": np.datetime64('2025-05-30T20:08:08.123')}
    ],
    'decimal_col': [
        {"decimal_val": Decimal('123456789.1234567890')},
        {"decimal_val": Decimal('-987654321.9876543210')},
        {"decimal_val": Decimal('0.0000000001')}
    ]
}

dict3 = {
    "c1": [1, 2, 3, 4],
    "c2": ["banana", "water", "apple", "water"],
    "c3": [
        {"deep": {"level2": {"level3": 100.3}}},
        {"mixed_list": [{"a": 1}, {"a": 666}]},
        {"nested_int": 1, "mixed": "text", "float_val": 3.14},
        {"list_val": [1,2,3], "tuple_val": (4,5)},
    ],
    "c4": [
        {"coordinates": [1.1, 2.2], "tags": ("urgent", "important")},
        {"metadata": {"created_at": "2024-01-01", "active": True}},
        {"scores": [85.5, 92.3, 77.8], "status": "pass"},
        {"nested_list": [[1,2], [3,4], [5,6]]},
    ]
}

df2 = pd.DataFrame({
    "c1": dict3["c1"],
    "c2": dict3["c2"],
    "c3": dict3["c3"],
    "c4": dict3["c4"]
})


class TestQueryJSONDataFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.sess = chs.Session()
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        return super().tearDown()

    def test_query_df1(self):
        data = {
            'dict_col1': [
                {'id1': 1, 'name1': 'apple1' },
                {'id2': 2, 'name2': 'apple2' }
            ],
            'dict_col2': [
                {'id': 3, 'name': 'apple3' },
                {'id': 4, 'name': 'apple4' }
            ],
        }

        df_object = pd.DataFrame(data)

        ret = self.sess.query("SELECT dict_col1.name1, dict_col2.id, dict_col1.id2  FROM Python(df_object)")
        self.assertEqual(str(ret), EXPECTED2)

    def test_pandas_analyze_sample(self):
        data = {
            'int_col': [i for i in range(1, 20001)],
            'obj_col': [
                {'id': i, 'value': f'value_{i}', 'flag': random.choice([True, False])}
                if i != 2 else 100
                for i in range(1, 20001)
            ]
        }
        df = pd.DataFrame(data)

        self.sess.query("SET pandas_analyze_sample = 20000")
        with self.assertRaises(Exception):
            self.sess.query("SELECT obj_col.id FROM Python(df) ORDER BY int_col LIMIT 1")

        self.sess.query("SET pandas_analyze_sample = 10000")
        ret = self.sess.query("SELECT obj_col.id FROM Python(df) ORDER BY int_col LIMIT 1")
        self.assertEqual(str(ret), '1\n')

        self.sess.query("SET pandas_analyze_sample = 0")
        with self.assertRaises(Exception):
            self.sess.query("SELECT obj_col.id FROM Python(df) ORDER BY int_col LIMIT 1")

        self.sess.query("SET pandas_analyze_sample = -1")
        ret = self.sess.query("SELECT obj_col.id FROM Python(df) ORDER BY int_col LIMIT 1")
        self.assertEqual(str(ret), '1\n')

    def test_date_data_types(self):
        df = pd.DataFrame({
            'datetime': complex_dict['datetime_col'],
            'time': complex_dict['time_col'],
            'date': complex_dict['date_col'],
            'timedelta': complex_dict['timedelta_col']
        })

        ret = self.sess.query("""
            SELECT date.date, time.time, datetime.datetime, timedelta.timedelta
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"2023-05-15","12:30:00","2024-05-30 14:30:00","1 day, 12:30:00"\n"2024-01-01","00:00:00","2023-12-31 23:59:59","2 days, 3:20:00"\n')

    def test_binary_data_types(self):
        df = pd.DataFrame({
            'bytes': complex_dict['bytes_col'],
            'bytearray': complex_dict['bytearray_col'],
            'memoryview': complex_dict['memoryview_col'],
            'uuid': complex_dict['uuid_col']
        })

        ret = self.sess.query("""
            SELECT bytes.bytes, bytearray.bytearray, memoryview.memoryview, uuid.uuid
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"苹果","bytearray1","memory1","00000000-0000-0000-0000-000000000001"\n"memory","苹果","苹果","00000000-0000-0000-0000-000000000000"\n')

    def test_none_data_types(self):
        df = pd.DataFrame({
            'special_values': complex_dict['special_values_col']
        })

        ret = self.sess.query("""
            SELECT special_values.null_val
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '\\N\n')

    def test_bool_data_types(self):
        df = pd.DataFrame({
            'special_values': complex_dict['special_values_col']
        })

        ret = self.sess.query("""
            SELECT special_values.bool_val
            FROM Python(df)
        """)
        self.assertEqual(str(ret), 'true\n')

    def test_empty_list_data_types(self):
        df = pd.DataFrame({
            'special_values': complex_dict['special_values_col']
        })

        ret = self.sess.query("""
            SELECT special_values.empty_list
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"[]"\n')

        ret = self.sess.query("""
            SELECT special_values.empty_dict
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '\\N\n')

        ret = self.sess.query("""
            SELECT special_values.empty_array
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"[]"\n')

    def test_special_num_data_types(self):
        df = pd.DataFrame({
            'special_num': complex_dict['special_num_col']
        })

        ret = self.sess.query("""
            SELECT special_num.special_num_val
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"[0,-1.1,NULL,NULL,NULL,NULL,NULL]"\n')

    def test_decimal_data_types(self):
        df = pd.DataFrame({
            'decimals': complex_dict['decimal_col']
        })

        self.sess.query("SET allow_suspicious_types_in_order_by = 1")
        ret = self.sess.query("""
            SELECT decimals.decimal_val
            FROM Python(df)
            ORDER BY decimals.decimal_val DESC
        """)
        self.assertEqual(str(ret), '"1E-10"\n"123456789.1234567890"\n"-987654321.9876543210"\n')

    def test_string_data_types(self):
        df = pd.DataFrame({
            'string_values': complex_dict['string_col']
        })

        self.sess.query("SET allow_suspicious_types_in_order_by = 1")
        ret = self.sess.query("""
            SELECT string_values.str
            FROM Python(df)
            ORDER BY string_values.str
        """)
        self.assertEqual(str(ret), '"hello"\n"苹果"\n')

    def test_special_numpy_types(self):
        df = pd.DataFrame({
            'numpy': complex_dict['numpy_num_col']
        })

        ret = self.sess.query("""
            SELECT numpy.numpy_val
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"[42,-1,3.140000104904175,2.718]"\n')

        df = pd.DataFrame({
            'numpy': complex_dict['numpy_bool_col']
        })

        ret = self.sess.query("""
            SELECT numpy.numpy_val
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"[true,false]"\n')

        df = pd.DataFrame({
            'numpy': complex_dict['numpy_datetime_col']
        })

        ret = self.sess.query("""
            SELECT numpy.numpy_val
            FROM Python(df)
        """)
        self.assertEqual(str(ret), '"2025-05-30 20:08:08.123000000"\n')


if __name__ == "__main__":
    unittest.main(verbosity=2)
