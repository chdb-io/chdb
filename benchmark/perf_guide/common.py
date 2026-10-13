"""Shared harness for the performance-guide benchmarks.

Every case runs in a fresh subprocess so peak RSS (VmHWM) belongs to that case
alone. The parent records wall time, peak RSS, and whatever extra numbers the
case returns, repeats the case, and keeps the median wall time.

Layout:

    BENCH_DATA     directory holding prepared datasets (prepare_data.py)
    BENCH_RESULTS  directory receiving <suite>.json
    BENCH_THREADS  max_threads applied through the session connection string
    BENCH_REPEAT   repetitions per case (default 3)
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional

DATA = os.environ.get("BENCH_DATA", os.path.expanduser("~/chdb-bench/data"))
RESULTS = os.environ.get("BENCH_RESULTS", os.path.expanduser("~/chdb-bench/results"))
THREADS = int(os.environ.get("BENCH_THREADS", "16"))
REPEAT = int(os.environ.get("BENCH_REPEAT", "3"))

TAXI_GLOB = os.path.join(DATA, "taxi", "*.parquet")
TAXI_MONTH = os.path.join(DATA, "taxi", "yellow_tripdata_2024-01.parquet")
TAXI_MONTH_CSV = os.path.join(DATA, "taxi", "yellow_tripdata_2024-01.csv")
# ClickHouse globs use {a..b}; py_glob() translates them for Python's glob module.
TAXI_QUARTER = os.path.join(DATA, "taxi", "yellow_tripdata_2024-0{1..3}.parquet")
TICKS_WEEK = os.path.join(DATA, "ticks", "day=2024-01-0{1..7}", "symbol=*", "trades.parquet")
TICKS_HIVE = os.path.join(DATA, "ticks", "**", "*.parquet")
TICKS_DIR = os.path.join(DATA, "ticks")
KLINES_HIVE = os.path.join(DATA, "klines", "**", "*.parquet")
MERGETREE_DIR = os.path.join(DATA, "mergetree")


def session(path: str = ":memory:", threads: Optional[int] = None, **settings):
    """Open a chdb session with max_threads pinned for the whole benchmark."""
    from chdb import session as chs

    params = {"max_threads": THREADS if threads is None else threads}
    params.update(settings)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    if path == ":memory:":
        return chs.Session(f":memory:?{query}")
    return chs.Session(f"file:{path}?{query}")


def peak_rss_kb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss // 1024 if platform.system() == "Darwin" else rss


def current_rss_kb() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    return peak_rss_kb()


def engine_counters(conn) -> Dict[str, int]:
    """Cumulative read counters from system.events (process-wide)."""
    text = (
        conn.query(
            """
        SELECT
            sumIf(value, event = 'SelectedRows'),
            sumIf(value, event = 'SelectedBytes'),
            sumIf(value, event = 'SelectedParts'),
            sumIf(value, event = 'SelectedMarks')
        FROM system.events
        WHERE event IN ('SelectedRows', 'SelectedBytes', 'SelectedParts', 'SelectedMarks')
        """,
            "TSV",
        )
        .bytes()
        .decode()
        .strip()
    )
    rows, bytes_, parts, marks = (int(x) for x in text.split("\t"))
    return {"rows": rows, "bytes": bytes_, "parts": parts, "marks": marks}


def counter_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {f"read_{k}": after[k] - before[k] for k in before}


class MemoryTrackingSampler:
    """Sample system.metrics MemoryTracking from a second session."""

    def __init__(self, interval: float = 0.05):
        import threading

        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._conn = None

    def _loop(self):
        from chdb import session as chs

        self._conn = chs.Session(":memory:?max_threads=1")
        while not self._stop.wait(self.interval):
            try:
                value = (
                    self._conn.query(
                        "SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'",
                        "CSV",
                    )
                    .bytes()
                    .strip()
                )
                self.samples.append((time.perf_counter(), int(value), current_rss_kb()))
            except Exception:
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        if self._conn is not None:
            self._conn.close()

    def peak_mb(self) -> Optional[float]:
        if not self.samples:
            return None
        return max(s[1] for s in self.samples) / 1e6


def timed(fn: Callable[[], Any]) -> Dict[str, Any]:
    """Run fn in this process and return wall time plus whatever it returns."""
    rss_start = current_rss_kb()
    start = time.perf_counter()
    extra = fn() or {}
    wall = time.perf_counter() - start
    out = {
        "wall_s": round(wall, 4),
        "peak_rss_mb": round(peak_rss_kb() / 1024, 1),
        "rss_start_mb": round(rss_start / 1024, 1),
    }
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------


def run_case(
    suite: str,
    case: str,
    args: Optional[Dict[str, Any]] = None,
    repeat: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run suite.case(**args) in a subprocess `repeat` times; return the summary."""
    args = args or {}
    repeat = REPEAT if repeat is None else repeat
    runs = []
    for _ in range(repeat):
        payload = json.dumps({"suite": suite, "case": case, "args": args})
        child_env = dict(os.environ)
        child_env.update(env or {})
        completed = subprocess.run(
            [sys.executable, "-m", "benchmark.perf_guide.worker", payload],
            capture_output=True,
            text=True,
            env=child_env,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        if completed.returncode != 0:
            runs.append({"error": completed.stderr.strip()[-2000:]})
            break
        line = [ln for ln in completed.stdout.splitlines() if ln.startswith("@@RESULT@@")]
        if not line:
            runs.append({"error": "no result line", "stdout": completed.stdout[-2000:]})
            break
        runs.append(json.loads(line[-1][len("@@RESULT@@") :]))

    summary: Dict[str, Any] = {"case": case, "args": args, "runs": runs}
    good = [r for r in runs if "error" not in r]
    if good:
        summary["wall_s"] = round(statistics.median(r["wall_s"] for r in good), 4)
        summary["peak_rss_mb"] = round(max(r["peak_rss_mb"] for r in good), 1)
        for key in good[0]:
            if key not in ("wall_s", "peak_rss_mb") and isinstance(good[0][key], (int, float, str)):
                summary[key] = good[0][key]
    else:
        summary["error"] = runs[-1].get("error", "unknown")
    label = f"{suite}.{case} {args}"
    if "error" in summary:
        print(f"  FAIL {label}: {summary['error'][:300]}", flush=True)
    else:
        print(
            f"  {label}: wall={summary['wall_s']}s peak_rss={summary['peak_rss_mb']}MB",
            flush=True,
        )
    return summary


def machine_info() -> Dict[str, Any]:
    import chdb

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "chdb_engine": chdb.engine_version,
        "chdb_core": getattr(chdb, "core_version", "unknown"),
        "max_threads": THREADS,
        "repeat": REPEAT,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    try:
        import pandas

        info["pandas"] = pandas.__version__
    except ImportError:
        pass
    try:
        import pyarrow

        info["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        info["cpu_threads"] = os.cpu_count()
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            info["mem_total_gb"] = round(int(meminfo.readline().split()[1]) / 1024 / 1024, 1)
        info["loadavg"] = os.getloadavg()
    except (FileNotFoundError, OSError):
        pass
    return info


def save(suite: str, cases: list, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{suite}.json")
    payload = {"suite": suite, "machine": machine_info(), "cases": cases}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, default=str)
    print(f"wrote {path}")
    return path


def py_glob(pattern: str) -> list:
    """Expand a ClickHouse-style glob ({1..3}, {a,b}) with Python's glob."""
    import glob
    import re

    pattern = re.sub(r"\{(\d)\.\.(\d)\}", r"[\1-\2]", pattern)
    pattern = re.sub(r"\{([^{}]*,[^{}]*)\}", lambda m: "[" + m.group(1).replace(",", "") + "]", pattern)
    return sorted(glob.glob(pattern, recursive=True))


def file_size_mb(path_or_glob: str) -> float:
    total = 0
    for path in py_glob(path_or_glob):
        total += os.path.getsize(path)
    return round(total / 1e6, 1)
