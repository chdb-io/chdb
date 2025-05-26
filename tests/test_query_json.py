#!python3

import io
import json
import random
import shutil
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import chdb
import chdb.session as chs
from datetime import date

test_json_query_dir = ".tmp_test_json_query"

EXPECTED1 = """"['urgent','important']",100.3,"[]"
\\N,\\N,"[1,666]"
"""

EXPECTED2 = '"apple1",3,\\N\n\\N,4,2\n'

dict1 = {
    "c1": [1, 2, 3, 4, 5, 6, 7, 8],
    "c2": ["banana", "water", "apple", "water", "apple", "banana", "apple", "banana"],
    "c3": [
        {"deep": {"level2": {"level3": 100.3}}},
        {"mixed_list": [{"a": 1}, {"a": 666}]},
        {"nested_int": 1, "mixed": "text", "float_val": 3.14},
        {"list_val": [1,2,3], "tuple_val": (4,5)},
        {"multi_type": [1, "two", 3.0, True]},
        None,
        1,
        'a'
    ],
    "c4": [
        {"coordinates": [1.1, 2.2], "tags": ("urgent", "important")},
        {"metadata": {"created_at": "2024-01-01", "active": True}},
        {"scores": [85.5, 92.3, 77.8], "status": "pass"},
        {"nested_list": [[1,2], [3,4], [5,6]]},
        {"mixed_types": {"int": 42, "str": "answer", "float": 3.14159}},
        {},
        None,
        date(2023, 5, 15)
    ]
}

df1 = pd.DataFrame({
    "c1": dict1["c1"],
    "c2": dict1["c2"],
    "c3": dict1["c3"],
    "c4": dict1["c4"]
})

dict2 = {
    "c1": df1['c1'],
    "c2": df1['c2'],
    "c3": df1['c3'],
    "c4": df1['c4']
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

arrow_table1 = pa.Table.from_pandas(df2)

class MyReader(chdb.PyReader):
    def __init__(self, data):
        self.data = data
        self.cursor = 0
        super().__init__(data)

    def read(self, col_names, count):
        # print("Python func read", col_names, count, self.cursor)
        if self.cursor >= len(self.data["c1"]):
            return []
        block = [self.data[col] for col in col_names]
        self.cursor += len(block[0])
        return block

    def get_schema(self):
        return [
            ("c1", "int"),
            ("c2", "str"),
            ("c3", "json"),
            ("c4", "json")
        ]

class TestQueryJSON(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(test_json_query_dir, ignore_errors=True)
        self.sess = chs.Session(test_json_query_dir)
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        shutil.rmtree(test_json_query_dir, ignore_errors=True)
        return super().tearDown()

    def test_query_py_reader1(self):
        reader1 = MyReader(dict1)

        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(reader1) WHERE c1 <= 2 ORDER BY c1")

        # print(ret)
        self.assertEqual(str(ret), EXPECTED1)

    def test_query_py_reader2(self):
        reader2 = MyReader(dict2)

        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(reader2) WHERE c1 <= 2 ORDER BY c1")

        # print(ret)
        self.assertEqual(str(ret), EXPECTED1)

    def test_query_dict1(self):
        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(dict1) WHERE c1 <= 2 ORDER BY c1")

        # print(ret)
        self.assertEqual(str(ret), EXPECTED1)

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

        # print(ret)
        self.assertEqual(str(ret), EXPECTED2)

    def test_query_df2(self):
        self.sess.query("SET pandas_analyze_sample = 1")
        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(dict1) WHERE c1 <= 2 ORDER BY c1")

        # print(ret)
        self.assertEqual(str(ret), EXPECTED1)

    def test_query_pyarrow_table1(self):
        self.sess.query("SET pandas_analyze_sample = 1")
        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(arrow_table1) WHERE c1 <= 2 ORDER BY c1")

        # print(ret)
        self.assertEqual(str(ret), EXPECTED1)

if __name__ == "__main__":
    unittest.main(verbosity=3)
