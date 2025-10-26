#!python3

import unittest
import chdb.session as chs
import chdb
from datetime import date


EXPECTED1 = """"['urgent','important']",100.3,"[]"
\\N,\\N,"[1,666]"
"""

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

dict2 = {
    "c1": dict1['c1'],
    "c2": dict1['c2'],
    "c3": dict1['c3'],
    "c4": dict1['c4']
}


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
        self.sess = chs.Session()
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        return super().tearDown()

    def test_query_py_reader1(self):
        reader1 = MyReader(dict1)

        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(reader1) WHERE c1 <= 2 ORDER BY c1")

        self.assertEqual(str(ret), EXPECTED1)

    def test_query_py_reader2(self):
        reader2 = MyReader(dict2)

        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(reader2) WHERE c1 <= 2 ORDER BY c1")

        self.assertEqual(str(ret), EXPECTED1)

    def test_query_dict1(self):
        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(dict1) WHERE c1 <= 2 ORDER BY c1")

        self.assertEqual(str(ret), EXPECTED1)

    def test_query_dict2(self):
        self.sess.query("SET pandas_analyze_sample = 1")
        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list[].a FROM Python(dict1) WHERE c1 <= 2 ORDER BY c1")
        self.assertEqual(str(ret), EXPECTED1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
