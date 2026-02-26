#!python3

import unittest
import chdb


class TestProgressCallback(unittest.TestCase):
    def test_query_with_progress_tty(self):
        conn = chdb.connect(":memory:?progress=tty")
        try:
            result = conn.query("SELECT count() FROM numbers(1000000)", "CSV")
        finally:
            conn.close()

        self.assertEqual(result.data().strip(), "1000000")

    def test_query_with_progress_err(self):
        conn = chdb.connect(":memory:?progress=err")
        try:
            result = conn.query("SELECT count() FROM numbers(1000000)", "CSV")
        finally:
            conn.close()

        self.assertEqual(result.data().strip(), "1000000")

    def test_query_with_progress_table_tty(self):
        conn = chdb.connect(":memory:?progress-table=tty")
        try:
            result = conn.query("SELECT count() FROM numbers(1000000)", "CSV")
        finally:
            conn.close()

        self.assertEqual(result.data().strip(), "1000000")

    def test_streaming_query_progress_callback_with_auto_mode(self):
        conn = chdb.connect(":memory:?progress=auto")
        progress_events = []

        def progress_callback(payload):
            progress_events.append(
                (
                    int(payload.get("read_rows", 0)),
                    int(payload.get("elapsed_ns", 0)),
                )
            )

        conn.set_progress_callback(progress_callback)
        try:
            with conn.send_query("SELECT number FROM numbers_mt(50000000)", "CSV") as stream:
                for _ in stream:
                    pass
        finally:
            conn.set_progress_callback(None)
            conn.close()

        self.assertGreater(len(progress_events), 0)
        self.assertTrue(all(read_rows >= 0 for read_rows, _ in progress_events))
        self.assertTrue(all(elapsed_ns >= 0 for _, elapsed_ns in progress_events))

    def test_query_progress_callback_with_auto_mode(self):
        conn = chdb.connect(":memory:?progress=auto")
        progress_events = []

        def progress_callback(payload):
            progress_events.append(
                (
                    int(payload.get("read_rows", 0)),
                    int(payload.get("elapsed_ns", 0)),
                )
            )

        conn.set_progress_callback(progress_callback)
        try:
            conn.query(
                "SELECT sum(number) FROM numbers_mt(1000000000) GROUP BY number % 10 SETTINGS max_threads=4",
                "CSV",
            )
        finally:
            conn.set_progress_callback(None)
            conn.close()

        self.assertGreater(len(progress_events), 0)
        self.assertTrue(all(read_rows >= 0 for read_rows, _ in progress_events))
        self.assertTrue(all(elapsed_ns >= 0 for _, elapsed_ns in progress_events))


if __name__ == "__main__":
    unittest.main()
