#!python3

import random
import unittest
import chdb


EXPECTED = """"auxten",9
"jerry",7
"tom",5
"""


class myReader(chdb.PyReader):
    def __init__(self, data):
        self.data = data
        self.cursor = 0
        super().__init__(data)

    def read(self, col_names, count):
        if self.cursor >= len(self.data["a"]):
            return []
        block = [self.data[col] for col in col_names]
        self.cursor += len(block[0])
        return block


class TestQueryPy(unittest.TestCase):
    def test_query_py(self):
        reader = myReader(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        ret = chdb.query("SELECT b, sum(a) FROM Python(reader) GROUP BY b ORDER BY b")
        self.assertEqual(str(ret), EXPECTED)

    def test_string_with_null_character(self):
        """Test basic string with null character in the middle"""
        res = chdb.query("SELECT 'hello\0world' as test_string", "CSV")
        self.assertFalse(res.has_error())
        result_data = res.bytes().decode('utf-8')
        self.assertIn('hello\0world', result_data)

    def test_random_float(self):
        x = {"col1": [random.uniform(0, 1) for _ in range(0, 100000)]}
        ret = chdb.sql(
            """
        select avg(col1)
        FROM Python(x)
        """
        )
        self.assertAlmostEqual(float(ret.bytes()), 0.5, delta=0.01)

    def test_query_dict(self):
        data = {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
        }

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(data) GROUP BY b ORDER BY b"
        )
        self.assertEqual(str(ret), EXPECTED)

    def test_query_dict_int(self):
        data = {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [1, 2, 3, 1, 2, 3],
        }

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(data) GROUP BY b ORDER BY b"
        )
        self.assertEqual(
            str(ret),
            """1,5
2,7
3,9
""",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
