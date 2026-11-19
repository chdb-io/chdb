"""
Tests for thread-safety, per-thread executor isolation, and concurrent streaming.
"""

import concurrent.futures
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest

import pandas as pd

from datastore import DataStore
from datastore.config import get_streaming_df, set_streaming_df
from datastore.executor import get_executor, reset_executor
from tests.test_utils import assert_frame_equal, assert_series_equal


class TestConcurrency(unittest.TestCase):
    """Test concurrent execution scenarios."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()

        cls.files = []
        cls.reference_dfs = []
        for i in range(5):
            csv_file = os.path.join(cls.temp_dir, f"data_{i}.csv")
            with open(csv_file, "w") as f:
                f.write("id,value,category\n")
                for j in range(10):
                    f.write(f"{j},{j * (i + 1)},cat_{j % 3}\n")
            cls.files.append(csv_file)
            cls.reference_dfs.append(pd.read_csv(csv_file))

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        reset_executor()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def tearDown(self):
        reset_executor()

    def test_unique_variable_names(self):
        """Test that each DataStore instance gets a unique variable name."""
        ds1 = DataStore.from_file(self.files[0])
        ds2 = DataStore.from_file(self.files[1])
        ds3 = DataStore.from_file(self.files[2])

        self.assertNotEqual(ds1._df_var_name, ds2._df_var_name)
        self.assertNotEqual(ds2._df_var_name, ds3._df_var_name)
        self.assertNotEqual(ds1._df_var_name, ds3._df_var_name)

        self.assertTrue(ds1._df_var_name.startswith("__ds_df_"))
        self.assertTrue(ds2._df_var_name.startswith("__ds_df_"))
        self.assertTrue(ds3._df_var_name.startswith("__ds_df_"))

    def test_variable_name_preserved_in_chain(self):
        """Test that variable name is preserved in lazy operation chains."""
        ds = DataStore.from_file(self.files[0])
        original_var_name = ds._df_var_name

        ds1 = ds.select("*").filter(ds.value > 5)
        self.assertEqual(ds1._df_var_name, original_var_name)

        ds2 = ds1.add_prefix("x_")
        self.assertEqual(ds2._df_var_name, original_var_name)

        self.assertIsNot(ds1, ds2)

    def test_concurrent_queries(self):
        """Test concurrent execution of queries on different files."""

        def process_file(idx):
            file_path = self.files[idx]
            ds = DataStore.from_file(file_path)
            try:
                result = (
                    ds.select("*")
                    .filter(ds.value > 3)
                    .add_prefix("col_")
                    .filter(ds.col_value > 5)
                    .to_df()
                )
                return idx, result
            finally:
                ds.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_file, i) for i in range(len(self.files))]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 5)

        for idx, result in results:
            pdf = self.reference_dfs[idx]
            expected = pdf[pdf["value"] > 3].add_prefix("col_")
            expected = expected[expected["col_value"] > 5]
            assert_frame_equal(result, expected)

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed SQL and pandas operations."""

        def complex_operation(idx, multiplier):
            ds = DataStore.from_file(self.files[idx])
            try:
                result = (
                    ds.filter(ds.value > 2)
                    .assign(doubled=lambda x: x["value"] * multiplier)
                    .filter(ds.doubled > 10)
                    .sort_values("doubled", ascending=False)
                    .head(3)
                    .to_df()
                )
                return idx, multiplier, result
            finally:
                ds.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(complex_operation, i, i + 1)
                for i in range(len(self.files))
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 5)

        for idx, multiplier, result in results:
            pdf = self.reference_dfs[idx]
            expected = pdf[pdf["value"] > 2].assign(
                doubled=lambda x, m=multiplier: x["value"] * m
            )
            expected = (
                expected[expected["doubled"] > 10]
                .sort_values("doubled", ascending=False, kind="mergesort")
                .head(3)
            )
            assert_frame_equal(result, expected)

    def test_no_variable_name_collision(self):
        """Test that concurrent operations do not collide or cross-contaminate results."""
        seen_names = set()
        names_lock = threading.Lock()
        collision_detected = [False]

        def check_operation(idx):
            ds = DataStore.from_file(self.files[idx])
            try:
                with names_lock:
                    if ds._df_var_name in seen_names:
                        collision_detected[0] = True
                    seen_names.add(ds._df_var_name)

                ds_mat = ds.add_prefix("x_")
                result = ds_mat.filter(ds.x_value > 5).to_df()
                return idx, result
            finally:
                ds.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(check_operation, idx)
                for _ in range(20)
                for idx in range(len(self.files))
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertFalse(collision_detected[0], "Variable name collision detected!")
        self.assertEqual(len(results), 100)

        for idx, result in results:
            pdf = self.reference_dfs[idx].add_prefix("x_")
            expected = pdf[pdf["x_value"] > 5]
            assert_frame_equal(result, expected)


class TestConcurrentStreamingAndExecutor(unittest.TestCase):
    """Regression tests for multithreaded executor isolation, streaming, and reset_executor()."""

    def setUp(self):
        self._orig_streaming = get_streaming_df()
        set_streaming_df(True)
        gc.enable()
        reset_executor()

    def tearDown(self):
        set_streaming_df(self._orig_streaming)
        gc.enable()
        reset_executor()

    def test_concurrent_global_executor_streaming_preserves_gc(self):
        """Concurrent multi-chunk streaming queries on get_executor() must not cancel each other or disable GC."""
        barrier = threading.Barrier(6)
        gc_states = []
        gc_lock = threading.Lock()

        def stream_worker(worker_id):
            barrier.wait(timeout=10)
            offset = worker_id * 1000
            with gc_lock:
                gc_states.append(gc.isenabled())
            res_df = (
                get_executor()
                .execute(
                    f"SELECT number AS id, number * 2 AS doubled "
                    f"FROM numbers(250000) WHERE number >= {offset} ORDER BY id"
                )
                .to_df()
            )
            with gc_lock:
                gc_states.append(gc.isenabled())
            return worker_id, res_df

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(stream_worker, i) for i in range(6)]
            outputs = [f.result() for f in futures]

        self.assertTrue(gc.isenabled())
        self.assertTrue(all(gc_states), "GC was unexpectedly disabled during streaming")

        for worker_id, res_df in outputs:
            offset = worker_id * 1000
            expected_len = 250_000 - offset
            self.assertEqual(len(res_df), expected_len)
            self.assertEqual(list(res_df.columns), ["id", "doubled"])
            self.assertEqual(int(res_df["id"].iloc[0]), offset)
            self.assertEqual(int(res_df["doubled"].iloc[-1]), (250_000 - 1) * 2)

    def test_concurrent_shared_file_datastore_streaming(self):
        """Concurrent streaming queries sharing one connected file DataStore must serialize safely per connection."""
        temp_dir = tempfile.mkdtemp()
        parquet_path = os.path.join(temp_dir, "large.parquet")
        pdf = pd.DataFrame(
            {
                "id": range(200_000),
                "category": [f"cat_{i % 4}" for i in range(200_000)],
                "amount": [float(i * 1.25) for i in range(200_000)],
            }
        )
        pdf.to_parquet(parquet_path, index=False)

        ds = DataStore("file", path=parquet_path)
        ds.connect()
        try:
            barrier = threading.Barrier(4)

            def query_shared(worker_id):
                barrier.wait(timeout=10)
                cat = f"cat_{worker_id % 4}"
                min_id = worker_id * 10_000
                got = ds[(ds["category"] == cat) & (ds["id"] >= min_id)].to_df()
                return worker_id, got

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(query_shared, i) for i in range(8)]
                # Run two waves of 4 barrier-synchronized threads
                results = [f.result() for f in futures]

            for worker_id, got in results:
                cat = f"cat_{worker_id % 4}"
                min_id = worker_id * 10_000
                expected = pdf[(pdf["category"] == cat) & (pdf["id"] >= min_id)]
                assert_frame_equal(got, expected)
        finally:
            ds.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concurrent_dataframe_groupby_and_filter_no_cross_thread_aliasing(self):
        """Concurrent DataStore(df) pipelines must isolate Python(__df__) per thread."""
        num_workers = 6
        barrier = threading.Barrier(num_workers)

        def run_pipeline(worker_id):
            base = (worker_id + 1) * 100
            pd_left = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5, 6],
                    "cat": ["a", "b", "a", "b", "a", "c"],
                    "val": [base + 1, base + 2, base + 3, base + 4, base + 5, base + 6],
                }
            )
            pd_right = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "weight": [2, 3, 2, 3, 2],
                }
            )

            barrier.wait(timeout=10)

            ds_left = DataStore(pd_left)
            ds_right = DataStore(pd_right)
            ds_merged = ds_left[ds_left["val"] > base + 1].merge(ds_right, on="id")
            ds_WithScore = ds_merged.assign(weighted=ds_merged["val"] * ds_merged["weight"])
            ds_grouped = ds_WithScore.groupby("cat")["weighted"].sum()

            pd_merged = pd_left[pd_left["val"] > base + 1].merge(pd_right, on="id")
            pd_with_score = pd_merged.assign(weighted=pd_merged["val"] * pd_merged["weight"])
            pd_grouped = pd_with_score.groupby("cat")["weighted"].sum()

            return ds_grouped.to_pandas(), pd_grouped

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(run_pipeline, i) for i in range(num_workers)]
            for fut in futures:
                got_series, expected_series = fut.result()
                assert_series_equal(got_series, expected_series)

    def test_concurrent_python_table_function_from_worker_threads(self):
        """PythonTableFunction must resolve registered DataFrames on non-__main__ worker threads."""

        def run_python_tf(worker_id):
            pdf = pd.DataFrame(
                {
                    "k": ["x", "y", "x", "z"],
                    "v": [worker_id + 10, worker_id + 40, worker_id + 20, worker_id + 30],
                }
            )
            ds = DataStore("python", df=pdf)
            try:
                got = ds.sort_values("v", ascending=False).head(3).head(2).to_df()
                expected = (
                    pdf.sort_values("v", ascending=False, kind="mergesort")
                    .head(3)
                    .head(2)
                )
                return got, expected
            finally:
                ds.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_python_tf, i * 100) for i in range(8)]
            for fut in futures:
                got, expected = fut.result()
                assert_frame_equal(got, expected)

    def test_concurrent_reset_executor_during_queries(self):
        """Calling reset_executor() concurrently with active worker queries must not fail queries or leak handles."""
        stop_reset = threading.Event()
        errors = []

        def resetter():
            while not stop_reset.is_set():
                try:
                    reset_executor()
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

        def query_worker(iteration):
            pdf = pd.DataFrame(
                {
                    "grp": ["g1", "g2", "g1", "g2", "g1"],
                    "num": [iteration + 1, iteration + 2, iteration + 3, iteration + 4, iteration + 5],
                }
            )
            ds = DataStore(pdf)
            got = ds[ds["num"] > iteration + 1].groupby("grp")["num"].sum().to_pandas()
            expected = pdf[pdf["num"] > iteration + 1].groupby("grp")["num"].sum()
            assert_series_equal(got, expected)

            stream_df = (
                get_executor()
                .execute(f"SELECT number FROM numbers(50000) WHERE number < {100 + (iteration % 50)}")
                .to_df()
            )
            self.assertEqual(len(stream_df), 100 + (iteration % 50))

        reset_thread = threading.Thread(target=resetter)
        reset_thread.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(query_worker, i) for i in range(24)]
                for fut in futures:
                    fut.result()
        finally:
            stop_reset.set()
            reset_thread.join(timeout=10)
            reset_executor()

        self.assertEqual(errors, [])

    def test_reset_executor_allows_custom_path_session_after_multithreaded_queries(self):
        """reset_executor() must deterministically close all live and exited thread connections."""
        # Run in a fresh subprocess because unrelated earlier test modules in the full suite
        # may keep module-level ':memory:' DataStore/Session objects alive across modules.
        script = textwrap.dedent(
            """
            import concurrent.futures
            import shutil
            import tempfile
            import threading
            import chdb.session
            import pandas as pd
            from datastore import DataStore
            from datastore.executor import reset_executor

            def worker(idx):
                pdf = pd.DataFrame({"a": [idx, idx + 1, idx + 2], "b": ["x", "y", "x"]})
                ds = DataStore(pdf)
                got = ds.groupby("b")["a"].sum().to_pandas()
                expected = pdf.groupby("b")["a"].sum()
                pd.testing.assert_series_equal(got, expected)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for fut in [pool.submit(worker, i * 10) for i in range(8)]:
                    fut.result()

            short_lived = [threading.Thread(target=worker, args=(i * 100,)) for i in range(4)]
            for t in short_lived:
                t.start()
            for t in short_lived:
                t.join(timeout=10)

            # Without gc.collect(), reset_executor() alone must close every thread's ':memory:'
            # connection so chDB permits opening a persistent Session on a custom directory.
            reset_executor()

            sess_dir = tempfile.mkdtemp(prefix="datastore_conc_sess_")
            try:
                with chdb.session.Session(sess_dir) as sess:
                    sess.query("CREATE TABLE t (id Int64, v String) ENGINE = MergeTree ORDER BY id")
                    sess.query("INSERT INTO t VALUES (1, 'ok'), (2, 'done')")
                    out = str(sess.query("SELECT count() FROM t", "CSV")).strip()
                    assert out == "2", f"Unexpected count: {out!r}"
            finally:
                shutil.rmtree(sess_dir, ignore_errors=True)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Subprocess failed (exit {proc.returncode}):\nstdout: {proc.stdout}\nstderr: {proc.stderr}",
        )

    def test_departed_worker_threads_reclaim_executor_connections(self):
        """Connections created for short-lived worker threads must be closed even on the initialized-thread fast path."""
        reset_executor()
        executor = get_executor()
        df = pd.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})

        def sql_sum() -> int:
            ds = DataStore.from_df(df)
            res = ds.sql("SELECT sum(v) AS s FROM __df__ WHERE k >= 1").to_df()
            return int(res["s"].iloc[0])

        # Initialize the main thread's slot BEFORE spawning short-lived workers so the
        # subsequent main-thread query exercises `_get_or_create_slot`'s fast path.
        self.assertEqual(sql_sum(), 60)
        self.assertEqual(len(executor._thread_slots), 1)

        for i in range(12):
            out = []

            def run_once(idx=i):
                out.append(sql_sum() + idx)

            t = threading.Thread(target=run_once)
            t.start()
            t.join(timeout=10)
            self.assertEqual(out, [60 + i])

        # Re-querying on the already-initialized main thread must drain retired worker slots.
        self.assertEqual(sql_sum(), 60)
        self.assertEqual(
            len(executor._thread_slots),
            1,
            f"Departed worker slots leaked in _thread_slots: {len(executor._thread_slots)}",
        )

    def test_retired_global_executor_closed_mid_acquire_delegates_to_replacement(self):
        """A thread holding a reference to a global Executor retired mid-call must retry on the new Executor."""
        reset_executor()
        stale_exec = get_executor()
        df = pd.DataFrame({"x": [2, 4, 6]})

        orig_get_or_create = stale_exec._get_or_create_slot

        def race_reset_then_fail():
            reset_executor()
            return orig_get_or_create()

        stale_exec._get_or_create_slot = race_reset_then_fail
        out = stale_exec.query_dataframe("SELECT sum(x) AS total FROM Python(__df__)", df)
        self.assertEqual(int(out["total"].iloc[0]), 12)
        self.assertIsNot(get_executor(), stale_exec)


class TestVariableNameGeneration(unittest.TestCase):
    """Test variable name generation."""

    def test_variable_name_format(self):
        """Test that variable names follow expected format."""
        temp_dir = tempfile.mkdtemp()
        csv_file = os.path.join(temp_dir, "test.csv")

        with open(csv_file, "w") as f:
            f.write("a,b\n1,2\n")

        try:
            ds = DataStore.from_file(csv_file)

            var_name = ds._df_var_name
            self.assertTrue(var_name.startswith("__ds_df_"))
            self.assertTrue(var_name.endswith("__"))
            self.assertTrue(var_name.isidentifier())
            self.assertGreater(len(var_name), 40)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_lazy_operations_preserve_variable_name(self):
        """Test that pandas-style operations preserve variable name but return new object."""
        temp_dir = tempfile.mkdtemp()
        csv_file = os.path.join(temp_dir, "test.csv")

        with open(csv_file, "w") as f:
            f.write("a,b\n1,2\n3,4\n")

        try:
            ds = DataStore.from_file(csv_file)
            var1 = ds._df_var_name

            ds2 = ds.add_prefix("x_")
            var2 = ds2._df_var_name

            self.assertEqual(var1, var2)
            self.assertIsNot(ds, ds2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
