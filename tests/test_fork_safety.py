"""Regression tests for lazy engine boot and POSIX fork() safety.

Each scenario runs in a fresh subprocess with a strict timeout so any deadlock
fails fast instead of hanging the test suite.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_TIMEOUT = 25


def _run_subprocess(code: str, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_ok(code: str, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess:
    proc = _run_subprocess(code, timeout=timeout)
    assert proc.returncode == 0, (
        f"subprocess exited with {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


class TestLazyEngineBoot:
    def test_import_chdb_does_not_boot_engine_or_spawn_threads(self):
        proc = _assert_ok(
            """
import sys

def get_threads():
    if sys.platform != "linux":
        return None
    with open("/proc/self/status", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return None

threads_before = get_threads()
import chdb
threads_after = get_threads()

assert not chdb._fork.is_engine_initialized(), "import chdb must not boot ClickHouse"
assert "engine_version" not in chdb.__dict__, "engine_version must be lazy"
assert "engine_version" in dir(chdb), "dir(chdb) should still advertise engine_version"
assert not chdb._fork.is_engine_initialized(), "dir(chdb) must not boot ClickHouse"

assert isinstance(chdb.__version__, str) and chdb.__version__
assert isinstance(chdb.chdb_version, tuple) and len(chdb.chdb_version) >= 1
assert isinstance(chdb.core_version, str) and chdb.core_version
assert chdb.__chdb_core_version__ == chdb.core_version

if threads_before is not None:
    assert threads_before == 1, threads_before
    assert threads_after == 1, f"import chdb spawned threads: {threads_before} -> {threads_after}"
print("LAZY_IMPORT_OK")
"""
        )
        assert "LAZY_IMPORT_OK" in proc.stdout

    def test_import_datastore_does_not_boot_engine(self):
        proc = _assert_ok(
            """
import chdb
import datastore  # noqa: F401
from chdb import datastore as chdb_ds  # noqa: F401

assert not chdb._fork.is_engine_initialized(), "import datastore must not boot ClickHouse"
assert "engine_version" not in chdb.__dict__
print("DATASTORE_LAZY_OK")
"""
        )
        assert "DATASTORE_LAZY_OK" in proc.stdout

    def test_lazy_engine_version_resolves_and_caches(self):
        proc = _assert_ok(
            """
import chdb

assert "engine_version" not in chdb.__dict__
v1 = chdb.engine_version
assert isinstance(v1, str) and "." in v1, repr(v1)
assert chdb.__dict__["engine_version"] == v1
v2 = chdb.engine_version
assert v2 == v1
print("VERSION_OK", v1)
"""
        )
        assert "VERSION_OK" in proc.stdout

    def test_lazy_engine_version_works_with_open_file_connection(self):
        proc = _assert_ok(
            """
import tempfile
import chdb

with tempfile.TemporaryDirectory() as d:
    conn = chdb.connect(f"{d}/test.db")
    try:
        v = chdb.engine_version
        assert isinstance(v, str) and "." in v, repr(v)
        print("FILE_VERSION_OK", v)
    finally:
        conn.close()
"""
        )
        assert "FILE_VERSION_OK" in proc.stdout


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork() required")
class TestForkSafety:
    def test_import_then_fork_child_queries_succeed(self):
        """Repro A: top-level import chdb -> os.fork() -> child uses all chDB APIs cleanly."""
        proc = _assert_ok(
            """
import os
import sys
import chdb

assert not chdb._fork.is_engine_initialized()

pid = os.fork()
if pid == 0:
    res = chdb.query("SELECT 1", "CSV")
    assert res.bytes().strip() == b"1", res.bytes()

    with chdb.connect() as conn:
        assert conn.query("SELECT 2", "CSV").bytes().strip() == b"2"

    from chdb.session import Session
    with Session() as sess:
        assert sess.query("SELECT 3", "CSV").bytes().strip() == b"3"

    import pandas as pd
    from datastore import DataStore
    ds = DataStore(pd.DataFrame({"x": [1, 2, 3]}))
    assert list(ds[ds["x"] > 1]["x"]) == [2, 3]
    print("CHILD_ALL_APIS_OK", flush=True)
    sys.exit(0)

_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
# Parent can still initialize and run queries afterward.
assert chdb.query("SELECT 42", "CSV").bytes().strip() == b"42"
print("PARENT_OK")
"""
        )
        assert "CHILD_ALL_APIS_OK" in proc.stdout
        assert "PARENT_OK" in proc.stdout

    def test_import_then_process_pool_executor_fork(self):
        """Repro A (ProcessPoolExecutor): import chdb -> fork-based worker pool."""
        proc = _assert_ok(
            """
import concurrent.futures
import multiprocessing
import chdb

def _worker(n):
    q = int(chdb.query(f"SELECT {n} * 10", "CSV").bytes().strip())
    with chdb.connect() as conn:
        c = int(conn.query(f"SELECT {n} + 1", "CSV").bytes().strip())
    from chdb.session import Session
    with Session() as s:
        sv = int(s.query(f"SELECT {n} + 2", "CSV").bytes().strip())
    return (q, c, sv)

ctx = multiprocessing.get_context("fork")
with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
    results = list(pool.map(_worker, [1, 2, 3, 4]))

assert results == [(10, 2, 3), (20, 3, 4), (30, 4, 5), (40, 5, 6)], results
assert chdb.query("SELECT 99", "CSV").bytes().strip() == b"99"
print("POOL_OK")
"""
        )
        assert "POOL_OK" in proc.stdout

    def test_import_datastore_then_fork_child_succeeds(self):
        proc = _assert_ok(
            """
import os
import sys
import pandas as pd
import chdb
from datastore import DataStore

assert not chdb._fork.is_engine_initialized()

pid = os.fork()
if pid == 0:
    ds = DataStore(pd.DataFrame({"v": [10, 20, 30]}))
    assert list(ds[ds["v"] >= 20]["v"]) == [20, 30]
    print("DS_CHILD_OK", flush=True)
    sys.exit(0)

_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
"""
        )
        assert "DS_CHILD_OK" in proc.stdout

    def test_open_connection_across_fork_child_exits_cleanly(self):
        """Repro B: parent holds open chdb.connect() -> fork -> child sys.exit(0) does not deadlock."""
        proc = _assert_ok(
            """
import gc
import os
import sys
import chdb

conn = chdb.connect()
assert conn.query("SELECT 1", "CSV").bytes().strip() == b"1"

# Child 1: direct sys.exit(0) without touching chdb
p1 = os.fork()
if p1 == 0:
    sys.exit(0)
_, st1 = os.waitpid(p1, 0)
assert os.waitstatus_to_exitcode(st1) == 0, st1

# Child 2: explicit del + gc.collect() + custom exit code
p2 = os.fork()
if p2 == 0:
    del conn
    gc.collect()
    sys.exit(42)
_, st2 = os.waitpid(p2, 0)
assert os.waitstatus_to_exitcode(st2) == 42, st2

# Child 3: explicit conn.close() is a safe no-op
p3 = os.fork()
if p3 == 0:
    conn.close()
    sys.exit(0)
_, st3 = os.waitpid(p3, 0)
assert os.waitstatus_to_exitcode(st3) == 0, st3

# Parent connection remains healthy after forks
assert conn.query("SELECT 2", "CSV").bytes().strip() == b"2"
conn.close()
print("REPRO_B_OK")
"""
        )
        assert "REPRO_B_OK" in proc.stdout

    def test_open_session_dbapi_arrow_datastore_across_fork_exit_cleanly(self):
        """Child exits cleanly when parent holds open Session, dbapi, ArrowSession, or DataStore."""
        cases = [
            (
                "session",
                """
from chdb.session import Session
handle = Session()
assert handle.query("SELECT 1", "CSV").bytes().strip() == b"1"
""",
                "handle.close()",
            ),
            (
                "dbapi",
                """
from chdb import dbapi
handle = dbapi.connect()
cur = handle.cursor()
cur.execute("SELECT 1")
assert cur.fetchone() == (1,)
""",
                "handle.close()",
            ),
            (
                "arrow",
                """
from chdb import arrow
handle = arrow.connect()
assert handle.query_arrow("SELECT 1 AS n").column("n").to_pylist() == [1]
""",
                "handle.close()",
            ),
            (
                "datastore",
                """
import pandas as pd
from datastore import DataStore
ds = DataStore(pd.DataFrame({"x": [1, 2, 3]}))
assert list(ds[ds["x"] > 1]["x"]) == [2, 3]
""",
                "",
            ),
        ]
        for label, setup, cleanup in cases:
            proc = _assert_ok(
                f"""
import os
import sys
{setup}
pid = os.fork()
if pid == 0:
    sys.exit(0)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, ({label!r}, status)
{cleanup}
print("CLEAN_EXIT_{label}")
"""
            )
            assert f"CLEAN_EXIT_{label}" in proc.stdout

    def test_fork_after_init_fails_fast_with_runtime_error(self):
        """After engine init in parent, every chDB entry point in child raises RuntimeError."""
        proc = _assert_ok(
            """
import os
import sys
import chdb

# Initialize engine via chdb.query (closes its connection before fork)
assert chdb.query("SELECT 1", "CSV").bytes().strip() == b"1"

pid = os.fork()
if pid == 0:
    def check_fails_fast(label, fn):
        try:
            fn()
        except RuntimeError as e:
            msg = str(e).lower()
            assert "fork" in msg, (label, msg)
            assert "spawn" in msg or "forkserver" in msg, (label, msg)
        else:
            raise AssertionError(f"{label} did not raise RuntimeError after fork")

    check_fails_fast("chdb.query", lambda: chdb.query("SELECT 1", "CSV"))
    check_fails_fast("chdb.connect", lambda: chdb.connect())
    check_fails_fast("Session", lambda: chdb.session.Session())
    check_fails_fast("arrow.connect", lambda: chdb.arrow.connect())
    check_fails_fast("query_arrow", lambda: chdb.query_arrow("SELECT 1"))
    check_fails_fast("engine_version", lambda: chdb.engine_version)

    import pandas as pd
    from datastore import DataStore
    check_fails_fast(
        "DataStore",
        lambda: DataStore(pd.DataFrame({"x": [1, 2, 3]}))[lambda d: d["x"] > 1].to_df(),
    )
    print("FAIL_FAST_ALL_OK", flush=True)
    sys.exit(0)

_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
"""
        )
        assert "FAIL_FAST_ALL_OK" in proc.stdout

    def test_fork_after_open_conn_unhandled_error_exits_nonzero(self):
        """Unhandled RuntimeError in child with inherited open conn exits 1 without hanging."""
        proc = _assert_ok(
            """
import os
import chdb

conn = chdb.connect()
assert conn.query("SELECT 1", "CSV").bytes().strip() == b"1"

pid = os.fork()
if pid == 0:
    conn.query("SELECT 2", "CSV")

_, status = os.waitpid(pid, 0)
code = os.waitstatus_to_exitcode(status)
conn.close()
assert code == 1, f"expected child exit code 1, got {code}"
print("UNHANDLED_EXIT_1_OK")
"""
        )
        assert "UNHANDLED_EXIT_1_OK" in proc.stdout

    def test_background_thread_active_query_lock_reset_on_fork(self):
        """Deadlock Path C: parent thread holds g_conn_lock during fork() -> child does not deadlock."""
        proc = _assert_ok(
            """
import os
import sys
import threading
import time
import chdb

started = threading.Event()
errors = []

def run_slow_query():
    started.set()
    try:
        chdb.query("SELECT sleep(0.5)", "CSV")
    except Exception as e:
        errors.append(e)

t = threading.Thread(target=run_slow_query)
t.start()
started.wait()
time.sleep(0.15)

pid = os.fork()
if pid == 0:
    # Locks must have been reset so we fail fast instead of deadlocking on g_conn_lock.
    for label, fn in (
        ("chdb.query", lambda: chdb.query("SELECT 1", "CSV")),
        ("chdb.query_arrow", lambda: chdb.query_arrow("SELECT 1")),
    ):
        try:
            fn()
        except RuntimeError as e:
            assert "fork" in str(e).lower(), (label, str(e))
        else:
            raise AssertionError(f"{label} should raise RuntimeError")
    print("LOCK_RESET_CHILD_OK", flush=True)
    sys.exit(0)

_, status = os.waitpid(pid, 0)
t.join(timeout=10)
assert not errors, errors
assert os.waitstatus_to_exitcode(status) == 0, status
print("LOCK_RESET_PARENT_OK")
"""
        )
        assert "LOCK_RESET_CHILD_OK" in proc.stdout
        assert "LOCK_RESET_PARENT_OK" in proc.stdout

    def test_active_arrow_batch_stream_across_fork(self):
        proc = _assert_ok(
            """
import os
import sys
import chdb

stream = chdb.query_arrow_batches("SELECT number FROM numbers(1000)")
pid = os.fork()
if pid == 0:
    try:
        next(stream)
    except RuntimeError as e:
        assert "fork" in str(e).lower(), str(e)
    else:
        raise AssertionError("next(stream) should fail fast")
    stream.close()
    try:
        chdb.query("SELECT 1", "CSV")
    except RuntimeError as e:
        assert "fork" in str(e).lower(), str(e)
    else:
        raise AssertionError("chdb.query should fail fast")
    print("ARROW_STREAM_CHILD_OK", flush=True)
    sys.exit(0)

_, status = os.waitpid(pid, 0)
total = sum(batch.num_rows for batch in stream)
assert total == 1000, total
assert os.waitstatus_to_exitcode(status) == 0, status
"""
        )
        assert "ARROW_STREAM_CHILD_OK" in proc.stdout

    def test_nested_forks(self):
        """Both (no-init -> fork -> init -> fork) and (init -> fork -> fork) behave safely."""
        proc = _assert_ok(
            """
import os
import sys
import chdb

# Case 1: parent has NOT initialized -> child initializes -> grandchild must fail fast
p1 = os.fork()
if p1 == 0:
    conn = chdb.connect()
    assert conn.query("SELECT 7", "CSV").bytes().strip() == b"7"
    gp = os.fork()
    if gp == 0:
        try:
            chdb.query("SELECT 1", "CSV")
        except RuntimeError as e:
            assert "fork" in str(e).lower(), str(e)
        else:
            raise AssertionError("grandchild should fail fast")
        sys.exit(0)
    _, gst = os.waitpid(gp, 0)
    conn.close()
    sys.exit(os.waitstatus_to_exitcode(gst))

_, st1 = os.waitpid(p1, 0)
assert os.waitstatus_to_exitcode(st1) == 0, st1

# Case 2: parent initializes -> child forks grandchild -> grandchild still fails fast & exits cleanly
conn = chdb.connect()
assert conn.query("SELECT 8", "CSV").bytes().strip() == b"8"
p2 = os.fork()
if p2 == 0:
    gp = os.fork()
    if gp == 0:
        try:
            chdb.connect()
        except RuntimeError as e:
            assert "fork" in str(e).lower(), str(e)
        else:
            raise AssertionError("grandchild should fail fast")
        sys.exit(0)
    _, gst = os.waitpid(gp, 0)
    sys.exit(os.waitstatus_to_exitcode(gst))

_, st2 = os.waitpid(p2, 0)
conn.close()
assert os.waitstatus_to_exitcode(st2) == 0, st2
print("NESTED_FORKS_OK")
"""
        )
        assert "NESTED_FORKS_OK" in proc.stdout

    def test_dead_weakref_proxy_in_heap_across_fork(self):
        """Retained dead weakref.proxy in heap must not abort after_in_child or cause deadlock."""
        proc = _assert_ok(
            """
import gc
import os
import weakref
import chdb

class _Target:
    pass

target = _Target()
dead_proxy = weakref.proxy(target)
del target
gc.collect()

conn = chdb.connect()
assert conn.query("SELECT 1", "CSV").bytes().strip() == b"1"

# Child 1: normal return (implicit exit 0) with dead_proxy still live in heap
p1 = os.fork()
if p1 == 0:
    assert dead_proxy is not None
    assert chdb._fork.is_forked_after_init()
    assert chdb._fork.should_fast_exit_on_shutdown()
    os._exit(0)
_, st1 = os.waitpid(p1, 0)
assert os.waitstatus_to_exitcode(st1) == 0, st1

# Child 2: real Python shutdown through atexit -> fast_exit_child()
p2 = os.fork()
if p2 == 0:
    assert chdb._fork.should_fast_exit_on_shutdown()
    raise SystemExit(0)
_, st2 = os.waitpid(p2, 0)
assert os.waitstatus_to_exitcode(st2) == 0, st2

# Keep dead_proxy referenced until after forks complete
assert type(dead_proxy) is weakref.ProxyType
conn.close()
print("DEAD_PROXY_OK")
"""
        )
        assert "DEAD_PROXY_OK" in proc.stdout
        assert "ReferenceError" not in proc.stderr
        assert "Exception ignored" not in proc.stderr

    def test_fork_child_system_exit_and_alias_preserve_exit_status(self):
        """Direct raise SystemExit(...) and pre-fork sys.exit aliases preserve exact exit code and stderr."""
        proc = _assert_ok(
            """
import os
import sys
import threading
import chdb

# Capture sys.exit alias BEFORE fork (and before engine init)
exit_fn = sys.exit

conn = chdb.connect()
assert conn.query("SELECT 1", "CSV").bytes().strip() == b"1"

def run_child(fn):
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        os.dup2(w_fd, 2)
        os.close(w_fd)
        fn()
    os.close(w_fd)
    chunks = []
    while True:
        chunk = os.read(r_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r_fd)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), b"".join(chunks).decode("utf-8", "replace")

# 1. raise SystemExit(42) directly
def _raise_42():
    raise SystemExit(42)

rc, err = run_child(_raise_42)
assert rc == 42, (rc, err)

# 2. raise SystemExit("error message") -> exit 1 + message on stderr
def _raise_msg():
    raise SystemExit("error message")

rc, err = run_child(_raise_msg)
assert rc == 1, (rc, err)
assert "error message" in err, repr(err)

# 3. raise SystemExit() -> exit 0
def _raise_none():
    raise SystemExit()

rc, err = run_child(_raise_none)
assert rc == 0, (rc, err)

# 4. Pre-fork alias exit_fn(42)
rc, err = run_child(lambda: exit_fn(42))
assert rc == 42, (rc, err)

# 5. Pre-fork alias exit_fn("aliased error") -> exit 1 + message on stderr
rc, err = run_child(lambda: exit_fn("aliased error"))
assert rc == 1, (rc, err)
assert "aliased error" in err, repr(err)

# 6. Caught SystemExit(42) followed by SystemExit(0) -> exit 0
def _caught_then_ok():
    try:
        raise SystemExit(42)
    except SystemExit:
        pass
    raise SystemExit(0)

rc, err = run_child(_caught_then_ok)
assert rc == 0, (rc, err)

# 7. try ... finally propagating SystemExit(42) -> exit 42
def _finally_se():
    try:
        raise SystemExit(42)
    finally:
        _ = 1

rc, err = run_child(_finally_se)
assert rc == 42, (rc, err)

# 8. Background thread SystemExit(42) does not override main thread exit 0
def _bg_thread_se():
    t = threading.Thread(target=lambda: sys.exit(42))
    t.start()
    t.join()
    raise SystemExit(0)

rc, err = run_child(_bg_thread_se)
assert rc == 0, (rc, err)

conn.close()
print("SYSTEM_EXIT_ALL_OK")
"""
        )
        assert "SYSTEM_EXIT_ALL_OK" in proc.stdout

    def test_fork_child_caught_system_exit_normal_fallthrough(self):
        """A child that catches SystemExit(42) and returns normally from <module> exits 0."""
        proc = _assert_ok(
            """
import os
import chdb

conn = chdb.connect()
assert conn.query("SELECT 1", "CSV").bytes().strip() == b"1"

pid = os.fork()
if pid == 0:
    try:
        raise SystemExit(42)
    except SystemExit:
        pass
else:
    _, status = os.waitpid(pid, 0)
    conn.close()
    assert os.waitstatus_to_exitcode(status) == 0, status
    print("CAUGHT_FALLTHROUGH_OK")
"""
        )
        assert "CAUGHT_FALLTHROUGH_OK" in proc.stdout

    def test_fork_child_exit_codes_trace_fallback(self):
        """Exercise the sys.settrace fallback path (used on Python <3.12 or busy monitoring)."""
        proc = _assert_ok(
            """
import os
import sys
import chdb
import chdb._fork as fork_mod

exit_fn = sys.exit
fork_mod._install_monitoring_exit_tracker = lambda: False

conn = chdb.connect()
assert conn.query("SELECT 1", "CSV").bytes().strip() == b"1"

# 1. raise SystemExit(42) at top level
p1 = os.fork()
if p1 == 0:
    raise SystemExit(42)
_, st1 = os.waitpid(p1, 0)
assert os.waitstatus_to_exitcode(st1) == 42, st1

# 2. raise SystemExit("trace error") -> exit 1 + message on stderr
r_fd, w_fd = os.pipe()
p2 = os.fork()
if p2 == 0:
    os.close(r_fd)
    os.dup2(w_fd, 2)
    os.close(w_fd)
    raise SystemExit("trace error")
os.close(w_fd)
err = os.read(r_fd, 4096).decode("utf-8", "replace")
os.close(r_fd)
_, st2 = os.waitpid(p2, 0)
assert os.waitstatus_to_exitcode(st2) == 1, (st2, err)
assert "trace error" in err, repr(err)

# 3. Pre-fork alias exit_fn(42)
p3 = os.fork()
if p3 == 0:
    exit_fn(42)
_, st3 = os.waitpid(p3, 0)
assert os.waitstatus_to_exitcode(st3) == 42, st3

# 4. Caught SystemExit(42) falling through normally -> exit 0
p4 = os.fork()
if p4 == 0:
    try:
        raise SystemExit(42)
    except SystemExit:
        pass
else:
    _, st4 = os.waitpid(p4, 0)
    conn.close()
    assert os.waitstatus_to_exitcode(st4) == 0, st4
    print("TRACE_FALLBACK_OK")
"""
        )
        assert "TRACE_FALLBACK_OK" in proc.stdout
