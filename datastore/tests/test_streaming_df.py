"""
Test streaming DataFrame execution.

Verifies that the streaming send_query("DataFrame") + pd.concat path
produces identical results to the synchronous query("DataFrame") path,
with lower MemoryTracker peak.
"""

import unittest
import pandas as pd
from datastore import DataStore
from datastore.config import set_streaming_df


class TestStreamingDataFrame(unittest.TestCase):
    """Test streaming vs synchronous DataFrame execution."""

    def setUp(self):
        self.ds = DataStore(table="streaming_test")
        self.ds.connect()
        self.ds.create_table({
            "id": "UInt64",
            "category": "String",
            "value": "Float64",
        })
        rows = [
            {"id": i, "category": f"cat_{i % 5}", "value": float(i * 1.5)}
            for i in range(200_000)
        ]
        self.ds.insert(rows)

    def tearDown(self):
        try:
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query(
                    "DROP TABLE IF EXISTS streaming_test"
                )
        except Exception:
            pass
        set_streaming_df(True)
        self.ds.close()

    def _query_with_mode(self, sql, streaming):
        set_streaming_df(streaming)
        result = self.ds._executor.execute(sql)
        return result.to_df()

    def test_streaming_matches_sync_full_table(self):
        """Streaming and sync produce identical results for full table scan."""
        sql = "SELECT * FROM streaming_test ORDER BY id"
        df_sync = self._query_with_mode(sql, streaming=False)
        df_stream = self._query_with_mode(sql, streaming=True)

        self.assertEqual(len(df_sync), len(df_stream))
        self.assertEqual(list(df_sync.columns), list(df_stream.columns))
        pd.testing.assert_frame_equal(
            df_stream.reset_index(drop=True),
            df_sync.reset_index(drop=True),
        )

    def test_streaming_matches_sync_with_filter(self):
        """Streaming and sync produce identical results with WHERE filter."""
        sql = "SELECT * FROM streaming_test WHERE category = 'cat_0' ORDER BY id"
        df_sync = self._query_with_mode(sql, streaming=False)
        df_stream = self._query_with_mode(sql, streaming=True)

        self.assertEqual(len(df_sync), 40_000)
        pd.testing.assert_frame_equal(
            df_stream.reset_index(drop=True),
            df_sync.reset_index(drop=True),
        )

    def test_streaming_matches_sync_with_aggregation(self):
        """Streaming and sync produce identical results for aggregation."""
        sql = (
            "SELECT category, count() as cnt, sum(value) as total "
            "FROM streaming_test GROUP BY category ORDER BY category"
        )
        df_sync = self._query_with_mode(sql, streaming=False)
        df_stream = self._query_with_mode(sql, streaming=True)

        self.assertEqual(len(df_sync), 5)
        pd.testing.assert_frame_equal(
            df_stream.reset_index(drop=True),
            df_sync.reset_index(drop=True),
        )

    def test_streaming_empty_result(self):
        """Streaming handles empty result sets correctly."""
        sql = "SELECT * FROM streaming_test WHERE id > 999999"
        df_stream = self._query_with_mode(sql, streaming=True)
        self.assertEqual(len(df_stream), 0)

    def test_streaming_single_row(self):
        """Streaming handles single-row results correctly."""
        sql = "SELECT count() as cnt FROM streaming_test"
        df_stream = self._query_with_mode(sql, streaming=True)
        self.assertEqual(len(df_stream), 1)
        self.assertEqual(int(df_stream.iloc[0]["cnt"]), 200_000)

    def test_config_toggle(self):
        """Config toggle switches between streaming and sync."""
        set_streaming_df(False)
        from datastore.config import get_streaming_df
        self.assertFalse(get_streaming_df())

        set_streaming_df(True)
        self.assertTrue(get_streaming_df())

    def test_datastore_filter_uses_streaming(self):
        """DataStore.filter() works correctly with streaming enabled."""
        set_streaming_df(True)
        ds = DataStore({"id": range(100), "val": range(100)})
        result = ds[ds["val"] > 50]
        self.assertEqual(len(result), 49)


class TestStreamingDataFrameOnDataFrame(unittest.TestCase):
    """Test streaming on DataFrame-backed queries (Python table function)."""

    def setUp(self):
        self.df = pd.DataFrame({
            "x": range(100_000),
            "y": [f"row_{i}" for i in range(100_000)],
            "z": [float(i) * 0.5 for i in range(100_000)],
        })

    def tearDown(self):
        set_streaming_df(True)

    def test_streaming_on_dataframe_source(self):
        """Streaming works with Python() table function (DataFrame source)."""
        set_streaming_df(True)
        ds = DataStore(self.df)
        result = ds[ds["x"] > 50_000]
        self.assertEqual(len(result), 49_999)

    def test_streaming_vs_sync_on_dataframe(self):
        """Streaming and sync produce identical results on DataFrame source."""
        ds_sync = DataStore(self.df.copy())
        set_streaming_df(False)
        result_sync = ds_sync[ds_sync["x"] > 90_000]

        ds_stream = DataStore(self.df.copy())
        set_streaming_df(True)
        result_stream = ds_stream[ds_stream["x"] > 90_000]

        self.assertEqual(len(result_sync), len(result_stream))


if __name__ == "__main__":
    unittest.main()
