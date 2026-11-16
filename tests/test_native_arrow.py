"""Python surface tests for native C Data Arrow query helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pyarrow as pa

import chdb
from chdb import arrow


class FakeReader:
    def __init__(self, batches):
        self.batches = list(batches)
        self.closed = False

    def read_all(self):
        return pa.Table.from_batches(self.batches)

    def read_next_batch(self):
        if not self.batches:
            raise StopIteration
        return self.batches.pop(0)

    def close(self):
        self.closed = True


class FakeResult:
    def has_error(self):
        return False

    def rows_written(self):
        return 2


class FakeConnection:
    def __init__(self):
        self.closed = False
        self.cancelled = False
        self.calls = []
        self.reader = None

    def query_arrow_reader(self, sql, *, params, options):
        self.calls.append((sql, params, options))
        self.reader = FakeReader([pa.record_batch({"n": [1, 2]})])
        return self.reader

    def send_query_arrow(self, sql, *, params, options):
        self.calls.append((sql, params, options))
        self.batches = [pa.record_batch({"n": [1, 2]})]
        return object()

    def streaming_fetch_arrow(self, result):
        return FakeReader([self.batches.pop(0)] if self.batches else [])

    def streaming_cancel_query(self, result):
        self.cancelled = True

    def query(self, sql, fmt, *, params=None):
        self.calls.append((sql, fmt, params))
        return FakeResult()

    def close(self):
        self.closed = True


class TestNativeArrowHelpers(unittest.TestCase):
    def test_materialized_result_and_options(self):
        native = FakeConnection()
        with patch.object(arrow, "_native_connection", return_value=native):
            with arrow.connect() as session:
                table = session.query_arrow("SELECT n", params={"p": 1}, options={"string_as_string": False})
        self.assertEqual(table.column("n").to_pylist(), [1, 2])
        self.assertEqual(native.calls, [("SELECT n", {"p": 1}, {"string_as_string": False})])
        self.assertTrue(native.reader.closed)
        self.assertTrue(native.closed)

    def test_partial_stream_close_cancels_and_releases_session(self):
        native = FakeConnection()
        with patch.object(arrow, "_native_connection", return_value=native):
            with arrow.query_arrow_batches("SELECT n") as stream:
                self.assertEqual(next(stream).column("n").to_pylist(), [1, 2])
            self.assertTrue(native.cancelled)
            self.assertTrue(native.closed)
            self.assertEqual(list(stream), [])
            self.assertTrue(arrow.g_conn_lock.acquire(blocking=False))
            arrow.g_conn_lock.release()

    def test_active_stream_prevents_overlapping_query(self):
        native = FakeConnection()
        with patch.object(arrow, "_native_connection", return_value=native):
            with arrow.connect() as session:
                with session.query_arrow_batches("SELECT n"):
                    with self.assertRaisesRegex(RuntimeError, "active Arrow batch stream"):
                        session.query_arrow("SELECT 1")

    def test_exhausted_stream_closes_connection(self):
        native = FakeConnection()
        with patch.object(arrow, "_native_connection", return_value=native):
            stream = arrow.query_arrow_batches("SELECT n")
            self.assertEqual([b.num_rows for b in stream], [2])
            self.assertTrue(native.closed)
            self.assertTrue(arrow.g_conn_lock.acquire(blocking=False))
            arrow.g_conn_lock.release()

    def test_stream_start_error_releases_global_lock(self):
        native = FakeConnection()

        def fail(*args, **kwargs):
            raise RuntimeError("query rejected")

        native.send_query_arrow = fail
        with patch.object(arrow, "_native_connection", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "query rejected"):
                arrow.query_arrow_batches("SELECT 1")
        self.assertTrue(native.closed)
        self.assertTrue(arrow.g_conn_lock.acquire(blocking=False))
        arrow.g_conn_lock.release()

    def test_insert_uses_source_names_and_quotes_identifiers(self):
        native = FakeConnection()
        with patch.object(arrow, "_native_connection", return_value=native):
            with arrow.connect() as session:
                count = session.insert_arrow(("db", "events"), pa.table({"id": [1, 2], "we`ird": ["a", "b"]}))
        self.assertEqual(count, 2)
        self.assertIn("INSERT INTO `db`.`events` (`id`, `we\\`ird`) SELECT `id`, `we\\`ird`", native.calls[0][0])
        self.assertRegex(native.calls[0][0], r"FROM Python\('__chdb_arrow_[0-9a-f]{32}'\)$")

    def test_missing_native_methods_is_explicit(self):
        native = FakeConnection()
        native.query_arrow_reader = None
        with patch.object(arrow._chdb, "connect", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "does not expose the Python native Arrow bridge"):
                arrow.connect()
        self.assertTrue(native.closed)

    def test_insert_arrow_reuses_existing_python_arrow_scan(self):
        native = chdb._chdb.connect(":memory:")
        # The installed engine predates the new direct-output bridge. This
        # focused test exercises only the already-supported Python input path.
        session = object.__new__(arrow.ArrowSession)
        session._conn = native
        session._active = None
        session._sources = set()
        session._closed = False
        try:
            native.query("CREATE TABLE arrow_insert_test (id Int64, name String) ENGINE=Memory", "Null")
            source = pa.table({"name": ["é", "你好"], "id": [1, 2]})
            written = session.insert_arrow("arrow_insert_test", source, columns=("name", "id"))
            self.assertEqual(written, 2)
            result = native.query("SELECT id, name FROM arrow_insert_test ORDER BY id", "TabSeparated")
            self.assertEqual(result.bytes().decode(), "1\té\n2\t你好\n")
        finally:
            session.close()

    def test_registered_table_is_scoped_and_queryable(self):
        native = chdb._chdb.connect(":memory:")
        session = object.__new__(arrow.ArrowSession)
        session._conn = native
        session._active = None
        session._sources = set()
        session._closed = False
        try:
            with session.register_arrow(pa.table({"id": [1, 2]})) as source:
                sql = f"SELECT sum(id) FROM {source.sql_expr}"
                self.assertEqual(session.query(sql, "CSV").bytes().strip(), b"3")
                self.assertIn(source._name, vars(arrow))
            self.assertNotIn(source._name, vars(arrow))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                _ = source.sql_expr
        finally:
            session.close()

    def test_registered_reader_is_one_shot(self):
        native = chdb._chdb.connect(":memory:")
        session = object.__new__(arrow.ArrowSession)
        session._conn = native
        session._active = None
        session._sources = set()
        session._closed = False
        reader = pa.RecordBatchReader.from_batches(
            pa.schema([pa.field("id", pa.int64())]),
            [pa.record_batch({"id": [1, 2]})],
        )
        try:
            with session.register_arrow(reader) as source:
                sql = f"SELECT count() FROM {source.sql_expr}"
                self.assertEqual(session.query(sql, "CSV").bytes().strip(), b"2")
                self.assertEqual(session.query(sql, "CSV").bytes().strip(), b"0")
        finally:
            session.close()

    def test_close_source_during_stream_defers_cleanup(self):
        native = FakeConnection()
        with patch.object(arrow, "_native_connection", return_value=native):
            with arrow.connect() as session:
                source = session.register_arrow(pa.table({"id": [1]}))
                with session.query_arrow_batches("SELECT n"):
                    source.close()
                    self.assertIn(source._name, vars(arrow))
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        _ = source.sql_expr
                self.assertNotIn(source._name, vars(arrow))

    def test_public_direct_arrow_end_to_end_when_bridge_is_available(self):
        probe = chdb._chdb.connect(":memory:")
        available = hasattr(probe, "query_arrow_reader")
        probe.close()
        if not available:
            self.skipTest("chdb-core Python native Arrow bridge is unavailable")

        source = pa.table({"id": [1, 2], "name": ["é", None]})
        with arrow.connect() as session:
            with session.register_arrow(source) as registration:
                table = session.query_arrow(
                    f"SELECT id, name FROM {registration.sql_expr} ORDER BY id"
                )
                self.assertEqual(table.column("id").to_pylist(), [1, 2])
                self.assertEqual(table.column("name").to_pylist(), ["é", None])

            with session.query_arrow_batches("SELECT number FROM numbers(1000)") as stream:
                batches = list(stream)
            self.assertEqual(sum(batch.num_rows for batch in batches), 1000)
            self.assertEqual(batches[0].column(0)[0].as_py(), 0)

            once = pa.RecordBatchReader.from_batches(
                pa.schema([pa.field("id", pa.int64())]),
                [pa.record_batch({"id": [3, 4]})],
            )
            registration = session.register_arrow(once)
            with session.query_arrow_batches(f"SELECT id FROM {registration.sql_expr}") as stream:
                registration.close()  # Deferred until the active scan ends.
                self.assertIn(registration._name, vars(arrow))
                self.assertEqual([v for batch in stream for v in batch.column(0).to_pylist()], [3, 4])
            self.assertNotIn(registration._name, vars(arrow))

            session.execute("CREATE TABLE native_arrow_dst (id Int64, name Nullable(String)) ENGINE=Memory")
            self.assertEqual(session.insert_arrow("native_arrow_dst", source), 2)
            self.assertEqual(
                session.query_arrow("SELECT id, name FROM native_arrow_dst ORDER BY id")
                .column("name").to_pylist(), ["é", None]
            )

        self.assertEqual(chdb.query_arrow("SELECT 42 AS answer").column("answer").to_pylist(), [42])
        with chdb.query_arrow_batches("SELECT number FROM numbers(1000)") as stream:
            self.assertGreater(next(stream).num_rows, 0)


if __name__ == "__main__":
    unittest.main()
