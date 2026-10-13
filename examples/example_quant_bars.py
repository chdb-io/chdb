#!/usr/bin/env python3
"""Aggregate synthetic ticks into minute bars inside chDB.

Ticks are written as Parquet by SQL, then reduced with toStartOfInterval and
argMin/argMax. A second query ASOF-joins each trade to the latest quote. The
script also reports peak RSS for the SQL aggregation and for loading the same
file into pandas, in separate processes.
"""

import os
import subprocess
import sys
import tempfile

ROWS = 2_000_000


def generate(directory):
    from chdb import session

    ticks = os.path.join(directory, "ticks.parquet")
    quotes = os.path.join(directory, "quotes.parquet")
    conn = session.Session()
    conn.query(
        f"""
        INSERT INTO FUNCTION file('{ticks}', 'Parquet')
        SELECT
            toDate('2024-01-02') AS day,
            ['AAPL', 'MSFT', 'NVDA', 'AMZN'][1 + (number % 4)] AS symbol,
            toDateTime64('2024-01-02 09:30:00', 3) + (number % 23400) AS ts,
            100 + (number % 50) + randCanonical() AS px,
            1 + (number % 20) AS qty,
            randCanonical() AS signal_a,
            randCanonical() AS signal_b
        FROM numbers({ROWS})
        """
    )
    conn.query(
        f"""
        INSERT INTO FUNCTION file('{quotes}', 'Parquet')
        SELECT
            ['AAPL', 'MSFT', 'NVDA', 'AMZN'][1 + (number % 4)] AS symbol,
            toDateTime64('2024-01-02 09:30:00', 3) + number * 5 AS ts,
            100 + (number % 40) AS bid
        FROM numbers(20000)
        """
    )
    conn.close()
    return ticks, quotes


def sql_bars(ticks):
    from chdb import session

    conn = session.Session()
    result = conn.query(
        f"""
        SELECT
            symbol,
            toStartOfInterval(ts, INTERVAL 1 MINUTE) AS bar,
            argMin(px, ts) AS open,
            max(px) AS high,
            min(px) AS low,
            argMax(px, ts) AS close,
            sum(qty) AS volume
        FROM file('{ticks}', Parquet)
        GROUP BY symbol, bar
        ORDER BY symbol, bar
        LIMIT 3
        """,
        "PrettyCompact",
    )
    print(result)
    conn.close()


def sql_asof(ticks, quotes):
    from chdb import session

    conn = session.Session()
    result = conn.query(
        f"""
        SELECT t.symbol, t.ts, t.px, q.bid
        FROM file('{ticks}', Parquet) AS t
        ASOF LEFT JOIN file('{quotes}', Parquet) AS q
            ON t.symbol = q.symbol AND t.ts >= q.ts
        ORDER BY t.symbol, t.ts
        LIMIT 3
        """,
        "PrettyCompact",
    )
    print(result)
    conn.close()


def peak_rss_kb():
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss


def run_sql_child(ticks):
    from chdb import session

    conn = session.Session()
    conn.query(
        f"""
        SELECT symbol, toStartOfInterval(ts, INTERVAL 1 MINUTE) AS bar, sum(qty) AS volume
        FROM file('{ticks}', Parquet)
        GROUP BY symbol, bar
        """,
        "Null",
    )
    conn.close()
    print(peak_rss_kb())


def run_pandas_child(ticks):
    import pandas as pd

    frame = pd.read_parquet(ticks)
    frame["bar"] = frame["ts"].dt.floor("min")
    frame.groupby(["symbol", "bar"], as_index=False)["qty"].sum()
    print(peak_rss_kb())


def measure(mode, ticks):
    completed = subprocess.run(
        [sys.executable, __file__, "--child", mode, ticks],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        mode, ticks = sys.argv[2], sys.argv[3]
        if mode == "sql":
            run_sql_child(ticks)
        elif mode == "pandas":
            run_pandas_child(ticks)
        else:
            raise SystemExit(f"unknown mode {mode}")
        return

    with tempfile.TemporaryDirectory() as directory:
        ticks, quotes = generate(directory)
        print(f"ticks={os.path.getsize(ticks) / 1e6:.1f} MB rows={ROWS}")
        sql_bars(ticks)
        sql_asof(ticks, quotes)
        sql_kb = measure("sql", ticks)
        pandas_kb = measure("pandas", ticks)
        print(f"peak_rss_kib sql={sql_kb} pandas={pandas_kb}")


if __name__ == "__main__":
    main()
