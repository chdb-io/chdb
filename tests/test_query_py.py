#!python3

from io import StringIO
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import chdb


EXPECTED = """"auxten",9
"jerry",7
"tom",5
"""


SMALL_CSV = """score1,score2,score3
70906,0.9166144356547409,draw
580525,0.9944755780981678,lose
254703,0.5290208413632235,lose
522924,0.9837867058675329,lose
382278,0.4781036385988161,lose
380893,0.48907718034312386,draw
221497,0.32603538643678,draw
446521,0.1386178708257899,win
522220,0.6633602572635723,draw
717410,0.6095994785374601,draw
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

    def test_query_dict(self):
        data = {
            "a": [1, 2, 3, 4, 5, 6],
            "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
        }

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(data) GROUP BY b ORDER BY b", "debug"
        )
        self.assertEqual(str(ret), EXPECTED)

    #     def test_query_dict_int(self):
    #         data = {
    #             "a": [1, 2, 3, 4, 5, 6],
    #             "b": [1, 2, 3, 1, 2, 3],
    #         }

    #         ret = chdb.query(
    #             "SELECT b, sum(a) FROM Python(data) GROUP BY b ORDER BY b", "debug"
    #         )
    #         self.assertEqual(
    #             str(ret),
    #             """1,5
    # 2,7
    # 3,9
    # """,
    #         )

    def test_query_pd_csv(self):
        csv_data = pd.read_csv(StringIO(SMALL_CSV))
        ret = chdb.query(
            """
            SELECT sum(score1), avg(score1), median(score1),
                sum(toFloat32(score2)), avg(toFloat32(score2)), median(toFloat32(score2)),
                countIf(score3 = 'win') AS wins,
                countIf(score3 = 'draw') AS draws,
                countIf(score3 = 'lose') AS losses,
                count()
            FROM Python(csv_data)
            """,
        )
        self.assertEqual(
            str(ret),
            """4099877,409987.7,414399.5,6.128691345453262,0.6128691345453262,0.5693101584911346,1,5,4,10
""",
        )


if __name__ == "__main__":
    unittest.main()
