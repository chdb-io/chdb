#!python3

import io
import json
import random
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv
import pyarrow.json
import pyarrow.parquet
import chdb


EXPECTED = """"auxten",9
"jerry",7
"tom",5
"""

EXPECTED_MULTILPE_TABLES = """1,"tom"
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


class TestQueryPyDataFrame(unittest.TestCase):
    def test_query_df(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        ret = chdb.query("SELECT b, sum(a) FROM Python(df) GROUP BY b ORDER BY b")
        self.assertEqual(str(ret), EXPECTED)

    def test_query_df_with_index(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            },
            index=[3, 1, 2, 4, 5, 6],
        )

        ret = chdb.query("SELECT * FROM Python(df)")
        self.assertIn("tom", str(ret))

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            },
            index=[0, 1, 2, 4, 5, 6],
        )

        ret = chdb.query("SELECT * FROM Python(df)")
        self.assertIn("tom", str(ret))

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            },
            index=['a', 1, 2, 4, 5, 6],
        )

        ret = chdb.query("SELECT * FROM Python(df)")
        self.assertIn("tom", str(ret))

    def test_query_pd_csv(self):
        csv_data = pd.read_csv(io.StringIO(SMALL_CSV))
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
            "4099877,409987.7,414399.5,6.128691345453262,0.6128691345453262,0.5693101584911346,1,5,4,10\n",
        )

    def test_query_multiple_df(self):
        df1 = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        df2 = pd.DataFrame(
            {
                "a": [7, 8, 9, 10, 11, 12],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        df3 = pd.DataFrame(
            {
                "a": [13, 14, 15, 16, 17, 18],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        ret = chdb.query(
            """
            SELECT * FROM python(df1) WHERE a = 1
            UNION ALL
            SELECT * FROM python(df2) WHERE a = 98
            UNION ALL
            SELECT * FROM python(df3) WHERE a = 198
            """)

        self.assertEqual(str(ret), EXPECTED_MULTILPE_TABLES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
