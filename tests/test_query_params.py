import unittest

import pandas as pd

import chdb
from chdb import session
from chdb.state import connect


class TestQueryParams(unittest.TestCase):
    def test_connection_query_with_params(self):
        df = chdb.query(
            "SELECT toDate({base_date:String}) + number AS d "
            "FROM numbers({total_days:UInt64}) "
            "LIMIT {items_per_page:UInt64}",
            "DataFrame",
            params={"base_date": "2025-01-01", "total_days": 10, "items_per_page": 2},
        )
        self.assertListEqual(df["d"].astype(str).tolist(), ["2025-01-01", "2025-01-02"])

    def test_connection_send_query_with_params(self):
        conn = connect(":memory:")
        try:
            stream = conn.send_query(
                "SELECT {v:Int32} AS v", format="dataframe", params={"v": 9}
            )
            with stream:
                chunk = stream.fetch()
                self.assertIsNotNone(chunk)
                self.assertListEqual(chunk["v"].tolist(), [9])
                self.assertIsNone(stream.fetch())
        finally:
            conn.close()

    def test_connection_query_with_params_csv(self):
        res = chdb.query("SELECT {x:UInt64} AS v", params={"x": 11})
        self.assertEqual(res.bytes(), b"11\n")

    def test_session_query_with_params(self):
        sess = session.Session(":memory:")
        try:
            df = sess.query(
                "SELECT {x:UInt64} + {y:UInt64} AS total, {s:String} AS label",
                fmt="dataframe",
                params={"x": 5, "y": 7, "s": "ok"},
            )
            self.assertListEqual(df["total"].tolist(), [12])
            self.assertListEqual(df["label"].tolist(), ["ok"])
        finally:
            sess.close()

    def test_session_query_with_params_csv(self):
        sess = session.Session(":memory:")
        try:
            res = sess.query("SELECT {x:UInt64} AS v", fmt="CSV", params={"x": 13})
            self.assertEqual(res.bytes(), b"13\n")
        finally:
            sess.close()

    def test_session_send_query_with_params(self):
        sess = session.Session(":memory:")
        try:
            stream = sess.send_query(
                "SELECT number AS n FROM numbers({n:UInt64} * {m:UInt64}) ORDER BY n",
                fmt="DataFrame",
                params={"m": 2, "n": 4},
            )
            chunks = []
            with stream:
                chunk = stream.fetch()
                while chunk is not None:
                    chunks.append(chunk)
                    chunk = stream.fetch()

            df = (
                pd.concat(chunks, ignore_index=True)
                if chunks
                else pd.DataFrame(columns=["n"])
            )
            self.assertListEqual(df["n"].tolist(), [0, 1, 2, 3, 4, 5, 6, 7])
        finally:
            sess.close()

    def test_query_with_null_param(self):
        df = chdb.query(
            "SELECT {n:Nullable(UInt64)} AS n",
            "DataFrame",
            params={"n": "\\N"},
        )
        self.assertTrue(pd.isna(df["n"][0]))

    def test_query_with_bool_param(self):
        df_true = chdb.query(
            "SELECT {flag:Bool} AS flag",
            "DataFrame",
            params={"flag": True},
        )
        df_false = chdb.query(
            "SELECT {flag:Bool} AS flag",
            "DataFrame",
            params={"flag": False},
        )
        self.assertListEqual(df_true["flag"].tolist(), [True])
        self.assertListEqual(df_false["flag"].tolist(), [False])

    def test_query_param_with_equals_sign(self):
        df = chdb.query(
            "SELECT {s:String} AS s",
            "DataFrame",
            params={"s": "a=b=c"},
        )
        self.assertListEqual(df["s"].tolist(), ["a=b=c"])

    def test_query_param_key_with_equals_sign(self):
        with self.assertRaises(RuntimeError) as exc:
            chdb.query("SELECT {a=b:UInt64}", "DataFrame", params={"a=b": 1})
        self.assertIn("SYNTAX_ERROR", str(exc.exception))

    def test_query_with_array_param(self):
        df = chdb.query(
            "SELECT {v:Array(UInt64)} AS v",
            "DataFrame",
            params={"v": [1, 2, 3]},
        )
        self.assertListEqual([list(df["v"].tolist()[0])], [[1, 2, 3]])

    def test_query_with_tuple_param_string_encoded(self):
        df = chdb.query(
            "SELECT {v:Tuple(UInt64, String)} AS v",
            "DataFrame",
            params={"v": "(7,'x')"},
        )
        self.assertListEqual([list(df["v"].tolist()[0])], [[7, "x"]])

    def test_query_with_params_missing_value(self):
        with self.assertRaises(RuntimeError) as exc:
            chdb.query("SELECT {x:UInt64} AS v")
        self.assertIn("Substitution `x` is not set", str(exc.exception))

    def test_query_with_params_invalid_type(self):
        with self.assertRaises(RuntimeError) as exc:
            chdb.query("SELECT {x:UInt64} AS v", params={"x": "not-a-number"})
        self.assertIn("cannot be parsed as UInt64", str(exc.exception))

    def test_session_send_query_with_params_missing_value_stream(self):
        sess = session.Session(":memory:")
        try:
            stream = sess.send_query("SELECT {n:UInt64} AS v")
            with stream:
                with self.assertRaises(RuntimeError) as exc:
                    stream.fetch()
            self.assertIn("Substitution `n` is not set", str(exc.exception))
        finally:
            sess.close()

    def test_session_send_query_with_params_invalid_type_stream(self):
        sess = session.Session(":memory:")
        try:
            stream = sess.send_query("SELECT {n:UInt64} AS v", params={"n": "bad"})
            with stream:
                with self.assertRaises(RuntimeError) as exc:
                    stream.fetch()
            self.assertIn("cannot be parsed as UInt64", str(exc.exception))
        finally:
            sess.close()


if __name__ == "__main__":
    unittest.main()
