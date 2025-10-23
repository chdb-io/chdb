#!python3

import unittest
import pyarrow as pa
import chdb.session as chs

EXPECTED1 = """"['urgent','important']",100.3,"[]"
"[]",0,"[1,666]"
"""

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

arrow_table1 = pa.table({
    "c1": dict3["c1"],
    "c2": dict3["c2"],
    "c3": dict3["c3"],
    "c4": dict3["c4"]
})


class TestQueryJSONArrow(unittest.TestCase):
    def setUp(self) -> None:
        self.sess = chs.Session()
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        return super().tearDown()

    def test_query_pyarrow_table1(self):
        ret = self.sess.query("SELECT c4.tags, c3.deep.level2.level3, c3.mixed_list.a FROM Python(arrow_table1) WHERE c1 <= 2 ORDER BY c1")

        self.assertEqual(str(ret), EXPECTED1)

    def test_pyarrow_complex_types(self):
        struct_type = pa.struct([
            pa.field('level1', pa.struct([
                pa.field('level2', pa.string())
            ])),
            pa.field('array_col', pa.list_(pa.int32()))
        ])

        data = [
            {'level1': {'level2': 'value1'}, 'array_col': [1,2]},
            {'level1': {'level2': None}, 'array_col': []}
        ]

        arrow_table = pa.Table.from_arrays([
            pa.array(data, type=struct_type)
        ], names=["struct_col"])

        ret = self.sess.query("SELECT struct_col.level1.level2 FROM Python(arrow_table)")
        self.assertEqual(str(ret), '"value1"\n\\N\n')


if __name__ == "__main__":
    unittest.main(verbosity=2)
