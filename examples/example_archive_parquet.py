#!/usr/bin/env python3
"""Move a MergeTree partition out to Hive-style Parquet, then drop it.

The table keeps recent days. One day is written with PARTITION BY into
day=<date>/ticks.parquet (zstd, the 26.9 default compression) and removed from
MergeTree with ALTER TABLE DROP PARTITION. The archived files stay readable
through file() and use_hive_partitioning, which defaults to 1.
"""

import os
import tempfile

from chdb import session


def main():
    with tempfile.TemporaryDirectory() as directory:
        conn = session.Session(os.path.join(directory, "db"))
        conn.query(
            """
            CREATE TABLE ticks (
                day Date,
                symbol LowCardinality(String),
                ts DateTime64(3),
                px Float64,
                qty UInt32
            )
            ENGINE = MergeTree
            PARTITION BY day
            ORDER BY (symbol, ts)
            """
        )
        conn.query(
            """
            INSERT INTO ticks
            SELECT
                toDate('2024-01-01') + intDiv(number, 100) AS day,
                if(number % 2 = 0, 'AAPL', 'MSFT') AS symbol,
                toDateTime64('2024-01-01 09:30:00', 3) + number AS ts,
                100 + (number % 17) AS px,
                1 + (number % 9) AS qty
            FROM numbers(250)
            """
        )
        before = int(conn.query("SELECT count() FROM ticks", "CSV").bytes().strip())

        archive = os.path.join(directory, "archive", "day={_partition_id}", "ticks.parquet")
        conn.query(
            f"""
            INSERT INTO FUNCTION file('{archive}', 'Parquet')
            PARTITION BY toString(day)
            SELECT day, symbol, ts, px, qty
            FROM ticks
            WHERE day = toDate('2024-01-01')
            ORDER BY symbol, ts
            SETTINGS output_format_parquet_compression_method = 'zstd'
            """
        )
        conn.query("ALTER TABLE ticks DROP PARTITION '2024-01-01'")
        after = int(conn.query("SELECT count() FROM ticks", "CSV").bytes().strip())

        glob = os.path.join(directory, "archive", "**", "*.parquet")
        archived = conn.query(
            f"""
            SELECT day, count() AS rows
            FROM file('{glob}', Parquet)
            GROUP BY day
            ORDER BY day
            SETTINGS use_hive_partitioning = 1
            """,
            "TSV",
        ).bytes().decode().strip()
        print(f"mergetree_before={before} mergetree_after={after}")
        print(f"archived={archived}")
        conn.close()

        if before != 250 or after != 150:
            raise SystemExit("unexpected MergeTree row counts")
        if archived != "2024-01-01\t100":
            raise SystemExit(f"unexpected archive contents: {archived}")


if __name__ == "__main__":
    main()
