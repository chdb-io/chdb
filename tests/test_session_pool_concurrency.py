"""Regression tests for zero-boot import, `:memory:` connection pooling,
path switching, UDF re-entrancy, thread-safe `ChDBTool.dataframe_query`, and
`StreamingResult` / `ArrowBatchStream` finalization.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import gc
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import weakref
from unittest.mock import patch

import pandas as pd
import pyarrow as pa

import chdb
from chdb import arrow, dbapi
from chdb.agents import ChDBTool
from chdb.session import Session


def _run_subprocess(code: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestZeroBootImportAndDeepbind(unittest.TestCase):
    def test_import_chdb_does_not_boot_engine(self):
        proc = _run_subprocess(
            """
            import importlib.abc
            import os
            import sys

            connect_calls = []

            class _SpyLoader(importlib.abc.Loader):
                def __init__(self, orig):
                    self._orig = orig
                def create_module(self, spec):
                    return self._orig.create_module(spec) if hasattr(self._orig, "create_module") else None
                def exec_module(self, module):
                    self._orig.exec_module(module)
                    orig_connect = module.connect
                    def spy_connect(*a, **kw):
                        connect_calls.append((a, kw))
                        return orig_connect(*a, **kw)
                    spy_connect.__wrapped__ = orig_connect
                    module._orig_connect = orig_connect

            import chdb

            assert chdb._memory_pool._idle == [], chdb._memory_pool._idle
            assert chdb._memory_pool._in_use == 0
            assert "engine_version" not in chdb.__dict__
            if os.path.isdir("/proc/self/task"):
                assert len(os.listdir("/proc/self/task")) == 1, os.listdir("/proc/self/task")

            assert isinstance(chdb.__version__, str) and chdb.__version__
            assert isinstance(chdb.chdb_version, tuple) and len(chdb.chdb_version) >= 1
            assert isinstance(chdb.core_version, str) and chdb.core_version != "unknown"
            assert chdb.__chdb_core_version__ == chdb.core_version
            assert "engine_version" in dir(chdb)

            # Lazy resolution works on demand and caches the string.
            ev = chdb.engine_version
            assert isinstance(ev, str) and "." in ev, ev
            assert chdb.__dict__["engine_version"] == ev
            print("ZERO_BOOT_OK", ev)
            """
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertIn("ZERO_BOOT_OK", proc.stdout)

    def test_engine_version_resolves_when_non_memory_session_already_open(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "ver_db")
            proc = _run_subprocess(
                f"""
                import chdb
                from chdb.session import Session

                assert "engine_version" not in chdb.__dict__
                with Session({db_path!r}) as sess:
                    sess.query("CREATE TABLE t (x Int32) ENGINE = MergeTree ORDER BY x")
                    ev = chdb.engine_version
                    assert isinstance(ev, str) and "." in ev, ev
                print("NON_MEM_VER_OK", ev)
                """
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("NON_MEM_VER_OK", proc.stdout)

    def test_preload_deepbind_supports_free_threaded_extension_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ft_suffix = ".cpython-314t-x86_64-linux-gnu.so"
            fake_so = os.path.join(tmpdir, "_chdb" + ft_suffix)
            with open(fake_so, "wb") as fh:
                fh.write(b"")

            loaded = []

            def fake_cdll(path, mode=0):
                loaded.append((path, mode))
                return object()

            with (
                patch.object(chdb.sys, "platform", "linux"),
                patch.object(chdb, "_this_dir", tmpdir),
                patch.object(chdb, "__path__", [tmpdir]),
                patch("importlib.machinery.EXTENSION_SUFFIXES", [ft_suffix, ".abi3.so", ".so"]),
                patch.object(ctypes, "CDLL", side_effect=fake_cdll),
            ):
                chdb._preload_chdb_with_deepbind()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0][0], fake_so)
            self.assertTrue(loaded[0][1] & 0x00008)


class TestMemoryConnectionPoolAndConcurrency(unittest.TestCase):
    def tearDown(self):
        chdb._memory_pool.drain_idle()

    def test_concurrent_query_query_arrow_and_batches(self):
        barrier = threading.Barrier(9)

        def sql_worker(tid: int):
            df = pd.DataFrame({"k": [1, 2, 3, 4], "v": [tid * 10 + i for i in range(4)]})  # noqa: F841
            barrier.wait(timeout=10)
            for _ in range(8):
                out_df = chdb.query(
                    "SELECT sum(v) AS total FROM Python(df)", "DataFrame"
                )
                expected = sum(tid * 10 + i for i in range(4))
                assert int(out_df["total"].iloc[0]) == expected

        def arrow_worker(tid: int):
            barrier.wait(timeout=10)
            for i in range(8):
                tbl = chdb.query_arrow(
                    "SELECT {val:Int64} AS v", params={"val": tid * 100 + i}
                )
                assert tbl.column("v").to_pylist() == [tid * 100 + i]

        def stream_worker(tid: int):
            barrier.wait(timeout=10)
            for _ in range(4):
                with chdb.query_arrow_batches(
                    "SELECT number + {off:UInt64} AS n FROM numbers(500)",
                    params={"off": tid},
                ) as stream:
                    vals = [v for batch in stream for v in batch.column("n").to_pylist()]
                assert len(vals) == 500
                assert vals[0] == tid and vals[-1] == tid + 499

        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as ex:
            futs = []
            for i in range(3):
                futs.append(ex.submit(sql_worker, i))
                futs.append(ex.submit(arrow_worker, i))
                futs.append(ex.submit(stream_worker, i))
            for f in futs:
                f.result(timeout=30)

        self.assertGreaterEqual(len(chdb._memory_pool._idle), 1)

    def test_reentrant_query_inside_python_udf_does_not_deadlock(self):
        @chdb.func(["Int64"], "Int64")
        def udf_reentrant_lookup(x):
            inner = chdb.query(
                "SELECT {v:Int64} * 7 AS r", "CSV", params={"v": int(x)}
            )
            arrow_tbl = chdb.query_arrow(
                "SELECT {v:Int64} + 1 AS r", params={"v": int(inner.bytes().strip())}
            )
            return int(arrow_tbl.column("r")[0].as_py())

        try:
            df = chdb.query(
                "SELECT udf_reentrant_lookup(toInt64(number)) AS y FROM numbers(5)",
                "DataFrame",
            )
            self.assertEqual(df["y"].tolist(), [i * 7 + 1 for i in range(5)])
        finally:
            chdb.drop_function("udf_reentrant_lookup")

    def test_pooled_queries_do_not_leak_session_settings_or_tables(self):
        # 1. SET on one query must not leak to subsequent pooled queries.
        iso_res = chdb.query(
            "SET date_time_output_format = 'iso'; "
            "SELECT toDateTime('2025-01-02 03:04:05', 'UTC') AS ts",
            "CSV",
        )
        self.assertIn(b"2025-01-02T03:04:05Z", iso_res.bytes())

        default_res = chdb.query(
            "SELECT toDateTime('2025-01-02 03:04:05', 'UTC') AS ts",
            "CSV",
        )
        self.assertIn(b"2025-01-02 03:04:05", default_res.bytes())
        self.assertNotIn(b"T03:04:05Z", default_res.bytes())

        # 2. SET readonly = 2 must not poison the pooled connection.
        chdb.query("SET readonly = 2; SELECT 1", "CSV")
        ro_val = chdb.query("SELECT getSetting('readonly')", "CSV").bytes().strip()
        self.assertEqual(ro_val, b"0")

        # 3. Stateless DDL/DML on `:memory:` must not collide across calls.
        r1 = chdb.query(
            "CREATE TABLE pool_iso_t (x Int32) ENGINE = Memory; "
            "INSERT INTO pool_iso_t VALUES (11); "
            "SELECT x FROM pool_iso_t",
            "CSV",
        )
        self.assertEqual(r1.bytes().strip(), b"11")

        r2 = chdb.query(
            "CREATE TABLE pool_iso_t (x Int32) ENGINE = Memory; "
            "INSERT INTO pool_iso_t VALUES (22); "
            "SELECT x FROM pool_iso_t",
            "CSV",
        )
        self.assertEqual(r2.bytes().strip(), b"22")

    def test_switching_between_pooled_query_and_persistent_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = os.path.join(tmpdir, "db1")
            dir2 = os.path.join(tmpdir, "db2")
            dir3 = os.path.join(tmpdir, "db3")
            dir4 = os.path.join(tmpdir, "db4")

            # Warm the `:memory:` pool first.
            self.assertEqual(chdb.query("SELECT 10", "CSV").bytes().strip(), b"10")
            self.assertEqual(chdb.query_arrow("SELECT 20 AS x").column("x").to_pylist(), [20])
            self.assertGreater(len(chdb._memory_pool._idle), 0)

            # Opening Session(dir1) must drain idle `:memory:` connections automatically.
            with Session(dir1) as s1:
                s1.query("CREATE TABLE items (id Int32) ENGINE = MergeTree ORDER BY id")
                s1.query("INSERT INTO items VALUES (1), (2)")
                self.assertEqual(s1.query("SELECT count() FROM items", "CSV").bytes().strip(), b"2")

            # Back to `:memory:` pooled queries, then chdb.connect(dir2), dbapi.connect(dir3), arrow.connect(dir4).
            self.assertEqual(chdb.query("SELECT 30", "CSV").bytes().strip(), b"30")

            with chdb.connect(dir2) as c2:
                c2.query("CREATE TABLE t2 (v Int32) ENGINE = MergeTree ORDER BY v")
                c2.query("INSERT INTO t2 VALUES (99)")

            db_conn = dbapi.connect(dir3)
            try:
                cur = db_conn.cursor()
                cur.execute("SELECT 77")
                self.assertEqual(cur.fetchone(), (77,))
                cur.close()
            finally:
                db_conn.close()

            self.assertEqual(chdb.query("SELECT 40", "CSV").bytes().strip(), b"40")

            with arrow.connect(dir4) as a4:
                a4.execute("CREATE TABLE t4 (id Int64) ENGINE = MergeTree ORDER BY id")
                a4.insert_arrow("t4", pa.table({"id": [5, 6]}))
                self.assertEqual(a4.query_arrow("SELECT sum(id) AS s FROM t4").column("s").to_pylist(), [11])

            # Direct `chdb.query(..., path=dir1)` after `:memory:` pool activity also drains `:memory:` cleanly.
            self.assertEqual(chdb.query("SELECT 50", "CSV").bytes().strip(), b"50")
            persisted = chdb.query("SELECT count() FROM items", "CSV", path=dir1)
            self.assertEqual(persisted.bytes().strip(), b"2")

    def test_create_table_combined_with_set_does_not_leak_catalog(self):
        # `classify_query` reports `CONTROL` when `CREATE TABLE` is combined with `SET`.
        chdb.query(
            "CREATE TABLE transient_ctrl_tbl (x Int32) ENGINE = Memory; "
            "SET max_threads = 1"
        )
        cnt = chdb.query(
            "SELECT count() FROM system.tables "
            "WHERE database = currentDatabase() AND name = 'transient_ctrl_tbl'",
            "CSV",
        )
        self.assertEqual(cnt.bytes().strip(), b"0")

        # Recreating the same table in a subsequent stateless call must also succeed.
        chdb.query(
            "CREATE TABLE transient_ctrl_tbl (x Int32) ENGINE = Memory; "
            "SET max_threads = 2"
        )

    def test_mutating_query_while_another_borrower_active_cleans_catalog(self):
        reader_entered = threading.Event()

        @chdb.udf.func(return_type="UInt64")
        def hold_borrower(x):
            reader_entered.set()
            import time

            time.sleep(0.15)
            return int(x) + 1

        errors = []

        def slow_reader():
            try:
                res = chdb.query("SELECT hold_borrower(41)", "CSV")
                self.assertEqual(res.bytes().strip(), b"42")
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=slow_reader)
        t.start()
        self.assertTrue(reader_entered.wait(timeout=10))

        # Mutate `:memory:` while `slow_reader` still holds an active borrowed handle.
        chdb.query(
            "CREATE TABLE race_mutated_tbl (x Int32) ENGINE = Memory; "
            "INSERT INTO race_mutated_tbl VALUES (7)"
        )
        t.join(timeout=10)
        self.assertEqual(errors, [])

        # Once the surviving reader releases, the dirty `:memory:` generation must be retired.
        cnt = chdb.query(
            "SELECT count() FROM system.tables "
            "WHERE database = currentDatabase() AND name = 'race_mutated_tbl'",
            "CSV",
        )
        self.assertEqual(cnt.bytes().strip(), b"0")

    def test_concurrent_queries_across_different_persistent_paths(self):
        with (
            tempfile.TemporaryDirectory() as dir1,
            tempfile.TemporaryDirectory() as dir2,
            tempfile.TemporaryDirectory() as dir3,
        ):
            for idx, d in enumerate((dir1, dir2, dir3), start=1):
                chdb.query(
                    f"CREATE TABLE p (v Int32) ENGINE = MergeTree ORDER BY v; "
                    f"INSERT INTO p VALUES ({idx * 10})",
                    path=d,
                )

            barrier = threading.Barrier(8)
            results = [None] * 8
            errors = []

            def worker(slot: int):
                try:
                    barrier.wait(timeout=10)
                    for _ in range(3):
                        mod = slot % 4
                        if mod == 0:
                            r = chdb.query("SELECT sum(v) FROM p", "CSV", path=dir1)
                            val = int(r.bytes().strip())
                            self.assertEqual(val, 10)
                        elif mod == 1:
                            r = chdb.query("SELECT sum(v) FROM p", "CSV", path=dir2)
                            val = int(r.bytes().strip())
                            self.assertEqual(val, 20)
                        elif mod == 2:
                            tbl = chdb.query_arrow("SELECT sum(v) AS s FROM p", path=dir3)
                            val = tbl.column("s").to_pylist()[0]
                            self.assertEqual(val, 30)
                        else:
                            r = chdb.query("SELECT 99", "CSV")
                            val = int(r.bytes().strip())
                            self.assertEqual(val, 99)
                    results[slot] = True
                except BaseException as exc:
                    errors.append((slot, exc))

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futs = [pool.submit(worker, i) for i in range(8)]
                for f in futs:
                    f.result(timeout=30)

            self.assertEqual(errors, [])
            self.assertEqual(results, [True] * 8)

            # Long-lived user Session on dir1 must still fail-fast (not deadlock)
            # if the same thread attempts to open a different path while dir1 stays open.
            with Session(dir1):
                with self.assertRaises(Exception) as ctx:
                    Session(dir2)
                self.assertIn("different path", str(ctx.exception))

    def test_custom_memory_startup_options_after_pooled_query_emit_no_warning(self):
        proc = _run_subprocess(
            """
            import warnings
            import chdb

            assert chdb.query("SELECT 1", "CSV").bytes().strip() == b"1"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with chdb.connect(":memory:?memory_worker_correct_memory_tracker=0") as conn:
                    res = conn.query(
                        "SELECT value FROM system.server_settings "
                        "WHERE name = 'memory_worker_correct_memory_tracker'",
                        "CSV",
                    )
                    assert '"0"' in str(res), res
            runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
            assert not runtime_warnings, [str(w.message) for w in runtime_warnings]
            print("NO_WARN_OK")
            """
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("NO_WARN_OK", proc.stdout)


class TestChDBToolThreadSafeDataFrameQuery(unittest.TestCase):
    def test_concurrent_dataframe_query_same_alias_no_globals_mutation(self):
        import chdb.agents.tool as tool_mod

        sentinel = object()
        tool_mod.__dict__["orders"] = sentinel
        try:
            barrier = threading.Barrier(6)

            def worker_own_tool(worker_id: int):
                df = pd.DataFrame(
                    {"k": ["a", "b"], "amount": [worker_id * 10 + 1, worker_id * 10 + 2]}
                )
                with ChDBTool(read_only=True) as tool:
                    barrier.wait(timeout=10)
                    for _ in range(10):
                        assert tool_mod.__dict__.get("orders") is sentinel
                        res = tool.dataframe_query(
                            "SELECT toInt64(sum(amount)) AS total FROM Python(orders)",
                            {"orders": df},
                        )
                        assert int(res.rows[0]["total"]) == worker_id * 20 + 3
                        assert tool_mod.__dict__.get("orders") is sentinel

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
                futs = [ex.submit(worker_own_tool, i) for i in range(6)]
                for f in futs:
                    f.result(timeout=30)

            self.assertIs(tool_mod.__dict__.get("orders"), sentinel)
        finally:
            tool_mod.__dict__.pop("orders", None)
            chdb._memory_pool.drain_idle()

    def test_shared_chdbtool_concurrent_dataframe_query_and_asyncio(self):
        with ChDBTool(read_only=True) as shared_tool:
            barrier = threading.Barrier(4)

            def worker_shared(worker_id: int):
                df = pd.DataFrame({"v": [worker_id + 1, worker_id + 2, worker_id + 3]})
                barrier.wait(timeout=10)
                for _ in range(8):
                    res = shared_tool.dataframe_query(
                        "SELECT toInt64(sum(v)) AS total FROM Python(shared_tbl)",
                        {"shared_tbl": df},
                    )
                    assert int(res.rows[0]["total"]) == 3 * worker_id + 6

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(worker_shared, i * 10) for i in range(4)]
                for f in futs:
                    f.result(timeout=30)

            async def run_async():
                async def one(idx: int):
                    df = pd.DataFrame({"n": [idx, idx * 2]})
                    res = await asyncio.to_thread(
                        shared_tool.dataframe_query,
                        "SELECT toInt64(sum(n)) AS s FROM Python(orders)",
                        {"orders": df},
                    )
                    return int(res.rows[0]["s"])

                return await asyncio.gather(*(one(i) for i in range(1, 7)))

            totals = asyncio.run(run_async())
            self.assertEqual(totals, [i * 3 for i in range(1, 7)])


class TestStreamingResultAndArrowStreamLifecycle(unittest.TestCase):
    def tearDown(self):
        chdb._memory_pool.drain_idle()

    def test_abandoned_streaming_result_cancels_and_releases_dataframe(self):
        def _start_and_abandon(s: Session):
            df = pd.DataFrame({"x": list(range(200_000))})
            ref = weakref.ref(df)
            stream = s.send_query("SELECT * FROM Python(df)", "CSV")
            self.assertIsNotNone(stream.fetch())
            # Return without closing `stream`; exiting this frame clears CPython's
            # `frame->f_locals` snapshot created during `Python(df)` lookup.
            return ref

        with Session() as sess:
            df_ref = _start_and_abandon(sess)
            gc.collect()

            self.assertIsNone(
                df_ref(),
                "Abandoned StreamingResult kept Python(df) pinned in PythonTableCache",
            )

            # The session must remain usable right after GC cleanup.
            self.assertEqual(sess.query("SELECT 42", "CSV").bytes().strip(), b"42")

    def test_superseded_stream_close_or_gc_does_not_cancel_newer_stream(self):
        with chdb.connect(":memory:") as conn:
            stream1 = conn.send_query("SELECT number FROM numbers(500000)", "CSV")
            self.assertIsNotNone(stream1.fetch())

            # Starting stream2 supersedes stream1 on the same connection.
            stream2 = conn.send_query("SELECT number FROM numbers(1000)", "CSV")
            self.assertIsNotNone(stream2.fetch())

            # Explicitly closing or GC-collecting stream1 must NOT kill stream2.
            stream1.close()
            del stream1
            gc.collect()

            rows_remaining = sum(chunk.rows_read() for chunk in stream2)
            self.assertGreaterEqual(rows_remaining, 0)
            stream2.close()

    def test_abandoned_arrow_batch_stream_releases_session_and_pool(self):
        with arrow.connect() as sess:
            src = sess.register_arrow(pa.table({"x": list(range(50_000))}))
            stream = sess.query_arrow_batches(f"SELECT x FROM {src.sql_expr}")
            first = next(stream)
            self.assertGreater(first.num_rows, 0)
            src.close()  # Deferred while stream is active.
            self.assertIn(src._name, vars(arrow))

            del first
            del stream
            gc.collect()

            # Abandoning the stream must clear `sess._active` and release deferred sources.
            self.assertIsNone(sess._active_stream())
            self.assertNotIn(src._name, vars(arrow))
            self.assertEqual(sess.query_arrow("SELECT 9 AS v").column("v").to_pylist(), [9])

        # Module-level abandoned ArrowBatchStream returns/retires its handle cleanly too.
        mod_stream = chdb.query_arrow_batches("SELECT number FROM numbers(100000)")
        self.assertGreater(next(mod_stream).num_rows, 0)
        del mod_stream
        gc.collect()
        self.assertEqual(chdb._memory_pool._in_use, 0)

    def test_concurrent_send_query_on_single_connection_registers_latest_stream(self):
        with chdb.connect(":memory:") as conn:
            barrier = threading.Barrier(4)
            streams = [None] * 4
            errors = []

            def start_stream(i: int):
                try:
                    barrier.wait(timeout=10)
                    streams[i] = conn.send_query(
                        f"SELECT number + {i * 1000} AS v FROM numbers(50000)", "CSV"
                    )
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=start_stream, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertEqual(errors, [])

            # Exactly one stream is the active latest stream; fetching it must succeed,
            # and all superseded streams must raise ChdbError cleanly.
            active_id = conn._chdb_tracker.active_stream_id
            winners = [s for s in streams if s._stream_state.stream_id == active_id]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            chunk = winner.fetch()
            self.assertIsNotNone(chunk)
            self.assertGreater(chunk.rows_read(), 0)

            for s in streams:
                if s is not winner:
                    with self.assertRaises(chdb.ChdbError):
                        s.fetch()
            winner.close()

    def test_reentrant_cyclic_stream_gc_on_same_connection_does_not_deadlock(self):
        with chdb.connect(":memory:") as conn:
            stream1 = conn.send_query("SELECT number FROM numbers(100000)", "CSV")
            self.assertIsNotNone(stream1.fetch())
            # Create a reference cycle so `stream1` is only collected by cyclic GC,
            # not immediately on `del stream1`.
            cycle = [stream1]
            cycle.append(cycle)
            state1 = stream1._stream_state
            del stream1, cycle

            # Simulate cyclic GC running on the same thread while `tracker.lock`
            # and `tracker.op_lock` are already held during a subsequent operation.
            tracker = conn._chdb_tracker
            tracker.enter_op(invalidates_stream=True)
            try:
                with tracker.lock:
                    gc.collect()
                    tracker.cancel_stream_from_gc(state1)
            finally:
                tracker.exit_op()

            stream2 = conn.send_query("SELECT 7 AS v", "CSV")
            chunk = stream2.fetch()
            self.assertIsNotNone(chunk)
            self.assertEqual(chunk.bytes().strip(), b"7")
            stream2.close()

    def test_preflight_validation_failure_preserves_active_stream(self):
        from chdb.state import sqlitelike

        with chdb.connect(":memory:") as conn:
            stream = conn.send_query("SELECT number FROM numbers(300000)", "CSV")
            first = stream.fetch()
            self.assertIsNotNone(first)
            self.assertGreater(first.rows_read(), 0)

            # Preflight AttributeError (`format` not a str) and pybind11 TypeError
            # (`query` not a str / `params` not a dict) must NOT retire `stream`.
            with self.assertRaises(AttributeError):
                conn.query("SELECT 1", format=123)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                conn.query(123, format="CSV")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                conn.send_query("SELECT 1", format="CSV", params="bad")  # type: ignore[arg-type]
            with patch.object(
                sqlitelike,
                "_require_polars",
                side_effect=ImportError("polars unavailable"),
            ):
                with self.assertRaises(ImportError):
                    conn.query("SELECT 1", format="polars")
                with self.assertRaises(ImportError):
                    conn.send_query("SELECT 1", format="polars")

            # `stream` must still fetch remaining rows and close cleanly.
            next_chunk = stream.fetch()
            self.assertIsNotNone(next_chunk)
            self.assertGreater(next_chunk.rows_read(), 0)
            stream.close()

    def test_cross_thread_arrow_batch_stream_close_and_path_switch_no_deadlock(self):
        with tempfile.TemporaryDirectory() as db_dir:
            chdb.query(
                "CREATE TABLE p (x Int32) ENGINE = MergeTree ORDER BY x; "
                "INSERT INTO p VALUES (11)",
                path=db_dir,
            )

            # Thread A opens S1, Thread B opens S2 and hands S2 to Thread A.
            # Thread A closes S2 while still holding S1, then attempts a persistent-path
            # query on Thread A: must fail-fast instead of deadlocking forever.
            s1 = chdb.query_arrow_batches("SELECT number FROM numbers(100000)")
            self.assertGreater(next(s1).num_rows, 0)

            box = []

            def open_s2():
                s2 = chdb.query_arrow_batches("SELECT number FROM numbers(100000)")
                next(s2)
                box.append(s2)

            tb = threading.Thread(target=open_s2)
            tb.start()
            tb.join(timeout=10)
            s2 = box.pop()
            s2.close()

            with self.assertRaises(RuntimeError) as ctx:
                chdb.query("SELECT sum(x) FROM p", "CSV", path=db_dir)
            self.assertIn("different path", str(ctx.exception))

            # Handing S1 to a consumer thread that runs `chdb.query(...)` before closing S1
            # must also complete without deadlocking.
            consumer_out = []

            def consume_and_query(stream):
                batch = next(stream)
                sub = chdb.query("SELECT 21 * 2", "CSV")
                consumer_out.append((batch.num_rows > 0, sub.bytes().strip()))
                stream.close()

            tc = threading.Thread(target=consume_and_query, args=(s1,))
            tc.start()
            tc.join(timeout=10)
            self.assertFalse(tc.is_alive())
            self.assertEqual(consumer_out, [(True, b"42")])
            self.assertEqual(chdb._memory_pool._in_use, 0)
            self.assertEqual(chdb._memory_pool._user_memory_conns, 0)

            # Once S1 is closed, switching to `db_dir` succeeds cleanly.
            res = chdb.query("SELECT sum(x) FROM p", "CSV", path=db_dir)
            self.assertEqual(res.bytes().strip(), b"11")

    def test_keyboard_interrupt_during_classify_sql_releases_leases(self):
        with tempfile.TemporaryDirectory() as db_dir:
            chdb.query(
                "CREATE TABLE p (x Int32) ENGINE = MergeTree ORDER BY x; "
                "INSERT INTO p VALUES (5)",
                path=db_dir,
            )

            with patch.object(chdb, "_classify_sql", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    chdb.query("SELECT 1")

            with patch.object(arrow, "_classify_sql", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    chdb.query_arrow("SELECT 1")
                with self.assertRaises(KeyboardInterrupt):
                    chdb.query_arrow_batches("SELECT 1")

            self.assertEqual(chdb._memory_pool._in_use, 0)
            self.assertEqual(chdb._memory_pool._in_use_by_thread, {})
            self.assertEqual(chdb._memory_pool._user_memory_conns, 0)

            # Another thread switching to `db_dir` must not hang on leaked reservations.
            out = []

            def check_persistent():
                r = chdb.query("SELECT sum(x) FROM p", "CSV", path=db_dir)
                out.append(r.bytes().strip())

            t = threading.Thread(target=check_persistent)
            t.start()
            t.join(timeout=10)
            self.assertFalse(t.is_alive())
            self.assertEqual(out, [b"5"])


if __name__ == "__main__":
    unittest.main()
