#!/usr/bin/env python3
"""Sample engine memory and process RSS while a query is running.

system.metrics MemoryTracking is the engine's current allocation.
system.events counters are cumulative for the process. system.query_log is not
populated on an embedded 26.9 session; system.user_query_log stays empty after
SYSTEM FLUSH LOGS, so this sampler does not read it.

Samples are taken from a second session so they are not queued behind the
workload query, then inserted into a MergeTree table.
"""

import threading
import time

from chdb import session


def rss_bytes():
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def sample(conn):
    text = conn.query(
        """
        SELECT
            (SELECT value FROM system.metrics WHERE metric = 'MemoryTracking') AS memory_tracking,
            (SELECT value FROM system.events WHERE event = 'Query') AS queries,
            (SELECT value FROM system.events WHERE event = 'SelectedRows') AS selected_rows
        """,
        "TSV",
    ).bytes().decode().strip()
    memory_tracking, queries, selected_rows = text.split("\t")
    return {
        "memory_tracking": int(memory_tracking),
        "queries": int(queries),
        "selected_rows": int(selected_rows),
        "rss": rss_bytes(),
    }


def main():
    conn = session.Session()
    monitor = session.Session()
    conn.query(
        """
        CREATE TABLE resource_samples (
            ts DateTime64(3),
            memory_tracking UInt64,
            rss UInt64,
            queries UInt64,
            selected_rows UInt64
        )
        ENGINE = MergeTree
        ORDER BY ts
        """
    )

    stop = threading.Event()
    samples = []

    def loop():
        while not stop.wait(0.05):
            try:
                samples.append(sample(monitor))
            except Exception as exc:  # the in-flight query can reject a second one
                samples.append({"error": str(exc)})

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    conn.query(
        """
        SELECT sleep(0.4), length(groupArray(number))
        FROM numbers(1500000)
        SETTINGS max_threads = 1
        """,
        "Null",
    )
    stop.set()
    worker.join()

    good = [row for row in samples if "memory_tracking" in row]
    if not good:
        raise SystemExit(f"no samples collected: {samples[:3]}")

    values = []
    for index, row in enumerate(good):
        values.append(
            "("
            f"now64(3) + INTERVAL {index} MILLISECOND, "
            f"{row['memory_tracking']}, {row['rss']}, {row['queries']}, {row['selected_rows']}"
            ")"
        )
    conn.query(f"INSERT INTO resource_samples VALUES {', '.join(values)}")
    print(conn.query(
        """
        SELECT
            count() AS samples,
            min(memory_tracking) AS min_engine_bytes,
            max(memory_tracking) AS max_engine_bytes,
            max(rss) AS max_rss_bytes,
            max(selected_rows) AS selected_rows
        FROM resource_samples
        """,
        "PrettyCompact",
    ))
    stored = int(conn.query("SELECT count() FROM resource_samples", "CSV").bytes().strip())
    conn.close()
    monitor.close()
    if stored < 2:
        raise SystemExit(f"expected several samples, stored {stored}: {samples[:3]}")
    if stored != len(good):
        raise SystemExit(f"stored {stored} rows, sampled {len(good)}")
    print(f"stored={stored}")


if __name__ == "__main__":
    main()
