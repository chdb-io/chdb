#!python3

import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
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
        print("Python func read", col_names, count, self.cursor)
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

    def test_query_df(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        ret = chdb.query("SELECT b, sum(a) FROM Python(df) GROUP BY b ORDER BY b")
        self.assertEqual(str(ret), EXPECTED)

    def test_query_arrow(self):
        table = pa.table(
            {
                "a": pa.array([1, 2, 3, 4, 5, 6]),
                "b": pa.array(["tom", "jerry", "auxten", "tom", "jerry", "auxten"]),
            }
        )

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(table) GROUP BY b ORDER BY b", "debug"
        )
        self.assertEqual(str(ret), EXPECTED)

    def test_query_arrow2(self):
        t2 = pa.table(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(t2) GROUP BY b ORDER BY b", "debug"
        )
        self.assertEqual(str(ret), EXPECTED)

    # def test_query_np(self):
    #     t3 = {
    #         "a": np.array([1, 2, 3, 4, 5, 6]),
    #         "b": np.array(["tom", "jerry", "auxten", "tom", "jerry", "auxten"]),
    #     }

    #     ret = chdb.query(
    #         "SELECT b, sum(a) FROM Python(t3) GROUP BY b ORDER BY b", "debug"
    #     )
    #     self.assertEqual(str(ret), EXPECTED)

    # def test_query_dict(self):
    #     data = {
    #         "a": [1, 2, 3, 4, 5, 6],
    #         "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
    #     }

    #     ret = chdb.query("SELECT b, sum(a) FROM Python(data) GROUP BY b ORDER BY b")
    #     self.assertEqual(str(ret), EXPECTED)


if __name__ == "__main__":
    unittest.main()
