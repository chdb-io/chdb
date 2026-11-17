"""chdb.query(..., "polars") through the chdb wrapper.

The wrapper ships its own chdb/__init__.py, which shadows chdb-core's, so the
"polars" output format chdb-core 26.11 added to its query() only reaches users
of the chdb package if this file maps it too.
"""

import unittest

import pytest

import chdb

pl = pytest.importorskip("polars")


class TestPolarsOutput(unittest.TestCase):
    def test_dataframe_in_dataframe_out(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = chdb.query("SELECT sum(a) AS total FROM Python(df)", "polars")
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result.columns, ["total"])
        self.assertEqual(result["total"].to_list(), [6])

    def test_format_name_is_case_insensitive(self):
        result = chdb.query("SELECT 1 AS x", "Polars")
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(result["x"].to_list(), [1])

    def test_lazyframe_and_series_input(self):
        lf = pl.LazyFrame({"a": [1, 2, 3, 4]}).filter(pl.col("a") > 1)
        result = chdb.query("SELECT a FROM Python(lf) ORDER BY a", "polars")
        self.assertEqual(result["a"].to_list(), [2, 3, 4])

        s = pl.Series("v", [10, 20, 30])
        result = chdb.query("SELECT sum(v) AS total FROM Python(s)", "polars")
        self.assertEqual(result["total"].to_list(), [60])

    def test_multi_block_result_is_complete(self):
        result = chdb.query("SELECT number FROM numbers(1000000)", "polars")
        self.assertEqual(result.height, 1000000)
        self.assertEqual(result["number"].sum(), 999999 * 1000000 // 2)

    def test_column_types_and_order(self):
        result = chdb.query(
            "SELECT toInt32(1) AS i, 'x' AS s, 1.5 AS f, NULL::Nullable(Int64) AS n",
            "polars",
        )
        self.assertEqual(result.columns, ["i", "s", "f", "n"])
        self.assertEqual(result.dtypes, [pl.Int32, pl.String, pl.Float64, pl.Int64])
        self.assertEqual(result.row(0), (1, "x", 1.5, None))

    def test_query_error_is_chdb_error(self):
        with self.assertRaises(chdb.ChdbError):
            chdb.query("SELECT * FROM table_that_does_not_exist", "polars")


if __name__ == "__main__":
    unittest.main()
