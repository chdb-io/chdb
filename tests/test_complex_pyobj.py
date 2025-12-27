import unittest
import pandas as pd
import chdb

df_with_na = pd.DataFrame(
    {
        "A": [1, 2, 3, pd.NA],
        "B": [4.0, 5.0, 6.0, pd.NA],
        "C": [True, False, True, pd.NA],
        "D": ["a", "b", "c", pd.NA],
        "E": [pd.NA, pd.NA, pd.NA, pd.NA],
        "F": [[1, 2], [3, 4], [5, 6], pd.NA],
        "G": [{"a": 1, "b": 2}, {"c": 3, "d": 4}, {"e": 5, "f": 6}, pd.NA],
    }
)

df_without_na = pd.DataFrame(
    {
        "A": [1, 2, 3, 4],
        "B": [4.0, 5.0, 6.0, 7.0],
        "C": [True, False, True, False],
        "D": ["a", "b", "c", "d"],
        "E": ["a", "b", "c", "d"],
        "F": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "G": [{"a": 1, "b": 2}, {"c": 3, "d": 4}, {"e": 5, "f": 6}, {"g": 7, "h": 8}],
    }
)


class TestComplexPyObj(unittest.TestCase):
    def test_df_with_na(self):
        ret = chdb.query(
            """
            select * from Python(df_with_na) limit 10
            """,
            "dataframe",
        )
        self.assertEqual(ret.dtypes["A"], "Int32")
        self.assertEqual(ret.dtypes["B"], "float64")
        self.assertEqual(ret.dtypes["C"], "object")
        self.assertEqual(ret.dtypes["D"], "object")
        self.assertEqual(ret.dtypes["E"], "object")
        self.assertEqual(ret.dtypes["F"], "object")
        self.assertEqual(ret.dtypes["G"], "object")
        self.assertEqual(ret.shape, (4, 7))

        # Row 0
        self.assertEqual(ret.iloc[0]["A"], 1)
        self.assertEqual(ret.iloc[0]["B"], 4.0)
        self.assertEqual(ret.iloc[0]["C"], 'True')
        self.assertEqual(ret.iloc[0]["D"], 'a')
        self.assertTrue(pd.isna(ret.iloc[0]["E"]))
        self.assertEqual(ret.iloc[0]["F"], '[1, 2]')
        self.assertEqual(ret.iloc[0]["G"], {"a": 1, "b": 2})

        # Row 1
        self.assertEqual(ret.iloc[1]["A"], 2)
        self.assertEqual(ret.iloc[1]["B"], 5.0)
        self.assertEqual(ret.iloc[1]["C"], 'False')
        self.assertEqual(ret.iloc[1]["D"], 'b')
        self.assertTrue(pd.isna(ret.iloc[1]["E"]))
        self.assertEqual(ret.iloc[1]["F"], '[3, 4]')
        self.assertEqual(ret.iloc[1]["G"], {"c": 3, "d": 4})

        # Row 2
        self.assertEqual(ret.iloc[2]["A"], 3)
        self.assertEqual(ret.iloc[2]["B"], 6.0)
        self.assertEqual(ret.iloc[2]["C"], 'True')
        self.assertEqual(ret.iloc[2]["D"], 'c')
        self.assertTrue(pd.isna(ret.iloc[2]["E"]))
        self.assertEqual(ret.iloc[2]["F"], '[5, 6]')
        self.assertEqual(ret.iloc[2]["G"], {"e": 5, "f": 6})

        # Row 3
        self.assertTrue(pd.isna(ret.iloc[3]["A"]))
        self.assertTrue(pd.isna(ret.iloc[3]["B"]))
        self.assertTrue(pd.isna(ret.iloc[3]["C"]))
        self.assertTrue(pd.isna(ret.iloc[3]["D"]))
        self.assertTrue(pd.isna(ret.iloc[3]["E"]))
        self.assertTrue(pd.isna(ret.iloc[3]["F"]))
        self.assertTrue(pd.isna(ret.iloc[3]["G"]))

    def test_df_without_na(self):
        ret = chdb.query(
            """
            select * from Python(df_without_na) limit 10
            """,
            "dataframe",
        )
        self.assertEqual(ret.dtypes["A"], "int64")
        self.assertEqual(ret.dtypes["B"], "float64")
        self.assertEqual(ret.dtypes["C"], "uint8")
        self.assertEqual(ret.dtypes["D"], "object")
        self.assertEqual(ret.dtypes["E"], "object")
        self.assertEqual(ret.dtypes["F"], "object")
        self.assertEqual(ret.dtypes["G"], "object")

        self.assertEqual(ret.shape, (4, 7))

        # Row 0
        self.assertEqual(ret.iloc[0]["A"], 1)
        self.assertEqual(ret.iloc[0]["B"], 4.0)
        self.assertEqual(ret.iloc[0]["C"], 1)
        self.assertEqual(ret.iloc[0]["D"], "a")
        self.assertEqual(ret.iloc[0]["E"], "a")
        self.assertEqual(ret.iloc[0]["F"], '[1, 2]')
        self.assertEqual(ret.iloc[0]["G"], {"a": 1, "b": 2})

        # Row 1
        self.assertEqual(ret.iloc[1]["A"], 2)
        self.assertEqual(ret.iloc[1]["B"], 5.0)
        self.assertEqual(ret.iloc[1]["C"], 0)
        self.assertEqual(ret.iloc[1]["D"], "b")
        self.assertEqual(ret.iloc[1]["E"], "b")
        self.assertEqual(ret.iloc[1]["F"], '[3, 4]')
        self.assertEqual(ret.iloc[1]["G"], {"c": 3, "d": 4})

        # Row 2
        self.assertEqual(ret.iloc[2]["A"], 3)
        self.assertEqual(ret.iloc[2]["B"], 6.0)
        self.assertEqual(ret.iloc[2]["C"], 1)
        self.assertEqual(ret.iloc[2]["D"], "c")
        self.assertEqual(ret.iloc[2]["E"], "c")
        self.assertEqual(ret.iloc[2]["F"], '[5, 6]')
        self.assertEqual(ret.iloc[2]["G"], {"e": 5, "f": 6})

        # Row 3
        self.assertEqual(ret.iloc[3]["A"], 4)
        self.assertEqual(ret.iloc[3]["B"], 7.0)
        self.assertEqual(ret.iloc[3]["C"], 0)
        self.assertEqual(ret.iloc[3]["D"], "d")
        self.assertEqual(ret.iloc[3]["E"], "d")
        self.assertEqual(ret.iloc[3]["F"], '[7, 8]')
        self.assertEqual(ret.iloc[3]["G"], {"g": 7, "h": 8})


if __name__ == "__main__":
    unittest.main()
