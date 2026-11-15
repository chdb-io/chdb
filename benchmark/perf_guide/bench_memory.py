"""Process memory: where bytes go, and what each knob changes.

Dataset: NYC yellow taxi 2024 (12 Parquet files, ~41M rows, 19 columns).
Each case runs in its own process; the parent records wall time and VmHWM.
"""

import sys

from benchmark.perf_guide import common

TAXI = common.TAXI_GLOB
MONTH = common.TAXI_MONTH
MONTH_CSV = common.TAXI_MONTH_CSV
QUARTER = common.TAXI_QUARTER

GROUPBY_SQL = """
SELECT passenger_count, payment_type, count() AS trips, sum(total_amount) AS revenue
FROM file('{path}', Parquet)
GROUP BY passenger_count, payment_type
"""

# --------------------------------------------------------------------------
# cases (run inside the worker process)
# --------------------------------------------------------------------------


def baseline_import():
    import chdb  # noqa: F401

    return {}


def baseline_session():
    conn = common.session()
    value = int(conn.query("SELECT value FROM system.metrics WHERE metric = 'MemoryTracking'", "CSV").bytes().strip())
    conn.close()
    return {"memory_tracking_mb": round(value / 1e6, 1)}


def groupby_pandas(path=TAXI):
    import glob
    import pandas as pd

    frames = [pd.read_parquet(p) for p in sorted(glob.glob(path))]
    df = pd.concat(frames, ignore_index=True)
    del frames
    out = df.groupby(["passenger_count", "payment_type"], dropna=False).agg(
        trips=("total_amount", "size"), revenue=("total_amount", "sum")
    )
    return {"rows_in": len(df), "groups": len(out), "df_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1)}


def groupby_pandas_columns(path=TAXI):
    import glob
    import pandas as pd

    cols = ["passenger_count", "payment_type", "total_amount"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted(glob.glob(path))], ignore_index=True)
    out = df.groupby(["passenger_count", "payment_type"], dropna=False).agg(
        trips=("total_amount", "size"), revenue=("total_amount", "sum")
    )
    return {"rows_in": len(df), "groups": len(out), "df_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1)}


def groupby_sql(path=TAXI, threads=None):
    conn = common.session(threads=threads)
    before = common.engine_counters(conn)
    df = conn.query(GROUPBY_SQL.format(path=path), "DataFrame")
    after = common.engine_counters(conn)
    conn.close()
    out = {"groups": len(df)}
    out.update(common.counter_delta(before, after))
    return out


def groupby_datastore(path=TAXI, mode="pandas"):
    import chdb.datastore as pd
    from chdb.datastore.config import config

    if mode == "performance":
        config.use_performance_mode()
    else:
        config.use_pandas_compat()
    ds = pd.read_parquet(path)
    result = ds.groupby(["passenger_count", "payment_type"]).agg({"total_amount": ["count", "sum"]})
    df = result.to_df()
    return {"groups": len(df)}


def materialize_dataframe(path=QUARTER):
    conn = common.session()
    df = conn.query(f"SELECT * FROM file('{path}', Parquet)", "DataFrame")
    total = float(df["total_amount"].sum())
    conn.close()
    return {"rows": len(df), "checksum": round(total, 2), "df_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1)}


def materialize_arrow_table(path=QUARTER):
    conn = common.session()
    table = conn.query(f"SELECT * FROM file('{path}', Parquet)", "ArrowTable")
    import pyarrow.compute as pc

    total = pc.sum(table["total_amount"]).as_py()
    conn.close()
    return {"rows": table.num_rows, "checksum": round(total, 2), "arrow_mb": round(table.nbytes / 1e6, 1)}


def stream_arrow(path=QUARTER, batch_size=65536, threads=None, parsing_threads=None):
    settings = {}
    if parsing_threads is not None:
        settings = {"max_parsing_threads": parsing_threads, "max_download_threads": parsing_threads}
    conn = common.session(threads=threads, **settings)
    stream = conn.send_query(f"SELECT * FROM file('{path}', Parquet) SETTINGS max_block_size={batch_size}", "Arrow")
    import pyarrow.compute as pc

    rows = 0
    total = 0.0
    batches = 0
    for batch in stream.record_batch(rows_per_batch=batch_size):
        rows += batch.num_rows
        total += pc.sum(batch.column("total_amount")).as_py()
        batches += 1
    stream.close()
    conn.close()
    return {"rows": rows, "batches": batches, "checksum": round(total, 2)}


def stream_pipeline(path=QUARTER, batch_size=65536, output="arrow"):
    from chdb.pipeline import iter_batches

    rows = 0
    conn = common.session()
    for batch in iter_batches(
        f"SELECT * FROM file('{path}', Parquet)", batch_size=batch_size, conn=conn, output=output
    ):
        if output == "arrow":
            rows += batch.num_rows
        elif output == "numpy":
            rows += len(batch["total_amount"])
        else:
            rows += len(batch)
    conn.close()
    return {"rows": rows}


HIGH_CARD_SQL = """
SELECT tpep_pickup_datetime, PULocationID, DOLocationID, count() AS c, sum(total_amount) AS s
FROM file('{path}', Parquet)
GROUP BY tpep_pickup_datetime, PULocationID, DOLocationID
"""


def high_cardinality_groupby(path=TAXI, external_bytes=0, max_memory=0, threads=None):
    conn = common.session(threads=threads)
    before = common.engine_counters(conn)
    settings = f"SETTINGS max_bytes_before_external_group_by={external_bytes}, max_memory_usage={max_memory}"
    with common.MemoryTrackingSampler(0.1) as sampler:
        try:
            groups = int(
                conn.query(f"SELECT count() FROM ({HIGH_CARD_SQL.format(path=path)}) {settings}", "CSV").bytes().strip()
            )
            engine_error = ""
        except Exception as exc:  # MEMORY_LIMIT_EXCEEDED is an expected outcome
            groups = -1
            engine_error = str(exc).split("\n")[0][:160]
    after = common.engine_counters(conn)
    conn.close()
    out = {"groups": groups, "engine_peak_mb": sampler.peak_mb(), "engine_error": engine_error}
    out.update(common.counter_delta(before, after))
    return out


def csv_vs_parquet(fmt="Parquet"):
    conn = common.session()
    path = MONTH_CSV if fmt == "CSVWithNames" else MONTH
    df = conn.query(
        f"SELECT passenger_count, count(), sum(total_amount) FROM file('{path}', {fmt}) GROUP BY passenger_count",
        "DataFrame",
    )
    conn.close()
    return {"groups": len(df)}


def column_pruning(mode="sql_two_columns"):
    conn = common.session()
    before = common.engine_counters(conn)
    if mode == "sql_two_columns":
        value = conn.query(f"SELECT avg(total_amount), avg(trip_distance) FROM file('{TAXI}', Parquet)", "CSV")
        out = {"value": value.bytes().decode().strip()}
    elif mode == "sql_select_star":
        df = conn.query(f"SELECT * FROM file('{QUARTER}', Parquet)", "DataFrame")
        out = {"value": f"{df['total_amount'].mean():.4f}", "rows": len(df)}
    else:
        raise ValueError(mode)
    after = common.engine_counters(conn)
    conn.close()
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    suite = "bench_memory"
    cases = []
    run = lambda case, **args: cases.append(common.run_case(suite, case, args))  # noqa: E731

    print("== baseline")
    run("baseline_import")
    run("baseline_session")

    print("== groupby over 12 months")
    run("groupby_pandas")
    run("groupby_pandas_columns")
    run("groupby_sql")
    run("groupby_datastore", mode="pandas")
    run("groupby_datastore", mode="performance")

    print("== materialize vs stream (3 months)")
    run("materialize_dataframe")
    run("materialize_arrow_table")
    for batch in (8192, 65536, 1048576):
        run("stream_arrow", batch_size=batch)
    for threads in (1, 4):
        run("stream_arrow", batch_size=65536, threads=threads)
    run("stream_arrow", batch_size=65536, threads=1, parsing_threads=1)
    run("stream_arrow", path=TAXI, batch_size=65536)
    run("stream_arrow", path=TAXI, batch_size=65536, threads=1, parsing_threads=1)
    run("materialize_dataframe", path=TAXI)
    for output in ("arrow", "numpy", "pandas"):
        run("stream_pipeline", batch_size=65536, output=output)

    print("== threads")
    for threads in (1, 2, 4, 8, 16, 32, 64):
        run("groupby_sql", threads=threads)

    print("== high-cardinality GROUP BY, memory limit, external aggregation")
    run("high_cardinality_groupby")
    run("high_cardinality_groupby", external_bytes=1 << 30)
    run("high_cardinality_groupby", max_memory=4 << 30)
    run("high_cardinality_groupby", max_memory=4 << 30, external_bytes=1 << 30)
    run("high_cardinality_groupby", max_memory=1 << 30, external_bytes=256 << 20)
    run("high_cardinality_groupby", max_memory=4 << 30, external_bytes=1 << 30, threads=4)

    print("== CSV vs Parquet, one month")
    run("csv_vs_parquet", fmt="Parquet")
    run("csv_vs_parquet", fmt="CSVWithNames")

    print("== column pruning")
    run("column_pruning", mode="sql_two_columns")
    run("column_pruning", mode="sql_select_star")

    common.save(
        suite,
        cases,
        {
            "datasets": {
                "taxi_year_mb": common.file_size_mb(TAXI),
                "taxi_quarter_mb": common.file_size_mb(QUARTER),
                "taxi_month_parquet_mb": common.file_size_mb(MONTH),
                "taxi_month_csv_mb": common.file_size_mb(MONTH_CSV),
            }
        },
    )


if __name__ == "__main__":
    sys.exit(main())
