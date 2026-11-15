"""Monitoring cost and fidelity: sampler overhead, MemoryTracking vs RSS trace.

Dataset: NYC taxi 2024 (12 Parquet files).
"""

import sys
import time

from benchmark.perf_guide import common

TAXI = common.TAXI_GLOB
HIGH_CARD_SQL = f"""
SELECT count() FROM (
    SELECT tpep_pickup_datetime, PULocationID, DOLocationID, count() AS c, sum(total_amount) AS s
    FROM file('{TAXI}', Parquet)
    GROUP BY tpep_pickup_datetime, PULocationID, DOLocationID
)
"""
GROUPBY_SQL = f"""
SELECT passenger_count, payment_type, count(), sum(total_amount)
FROM file('{TAXI}', Parquet)
GROUP BY passenger_count, payment_type
"""


def sampler_overhead(interval=0.0, queries=5):
    """Run the same aggregate N times with and without a background sampler."""
    conn = common.session()
    start = time.perf_counter()
    if interval > 0:
        with common.MemoryTrackingSampler(interval) as sampler:
            for _ in range(queries):
                conn.query(GROUPBY_SQL, "Null")
        samples = len(sampler.samples)
    else:
        for _ in range(queries):
            conn.query(GROUPBY_SQL, "Null")
        samples = 0
    elapsed = time.perf_counter() - start
    conn.close()
    return {"query_wall_s": round(elapsed, 3), "samples": samples}


def probe_cost(calls=200):
    """How long one system.metrics / system.events probe takes."""
    conn = common.session(threads=1)
    conn.query("SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'", "CSV")
    start = time.perf_counter()
    for _ in range(calls):
        conn.query("SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'", "CSV")
    metrics_us = (time.perf_counter() - start) / calls * 1e6
    start = time.perf_counter()
    for _ in range(calls):
        common.engine_counters(conn)
    events_us = (time.perf_counter() - start) / calls * 1e6
    conn.close()
    return {"metrics_probe_us": int(metrics_us), "events_probe_us": int(events_us)}


def memory_trace(external_bytes=0):
    """Return the MemoryTracking / RSS time series for a heavy GROUP BY."""
    conn = common.session()
    settings = f"SETTINGS max_bytes_before_external_group_by={external_bytes}" if external_bytes else ""
    with common.MemoryTrackingSampler(0.05) as sampler:
        t0 = time.perf_counter()
        groups = int(conn.query(f"{HIGH_CARD_SQL} {settings}", "CSV").bytes().strip())
        t1 = time.perf_counter()
    conn.close()
    trace = [
        (round(ts - t0, 3), round(mem / 1e6, 1), round(rss / 1024, 1))
        for ts, mem, rss in sampler.samples
        if ts <= t1 + 0.2
    ]
    # thin to about 60 points so the JSON stays readable
    step = max(1, len(trace) // 60)
    return {
        "groups": groups,
        "query_wall_s": round(t1 - t0, 3),
        "engine_peak_mb": sampler.peak_mb(),
        "rss_peak_mb": round(max(s[2] for s in sampler.samples) / 1024, 1) if sampler.samples else None,
        "trace": trace[::step],
    }


def main():
    suite = "bench_monitoring"
    cases = []

    print("== sampler overhead")
    for interval in (0.0, 0.1, 0.01):
        cases.append(common.run_case(suite, "sampler_overhead", {"interval": interval, "queries": 20}))
    cases.append(common.run_case(suite, "probe_cost", {}))

    print("== memory trace")
    cases.append(common.run_case(suite, "memory_trace", {"external_bytes": 0}, repeat=1))
    cases.append(common.run_case(suite, "memory_trace", {"external_bytes": 1 << 30}, repeat=1))

    common.save(suite, cases)


if __name__ == "__main__":
    sys.exit(main())
