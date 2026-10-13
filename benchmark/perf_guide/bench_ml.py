"""Training-data pipelines: streaming throughput, DataLoader workers, per-epoch cost.

Dataset: NYC yellow taxi 2024, first quarter (3 Parquet files, ~9M rows).
Features are numeric trip columns; the label is total_amount.
"""

import os
import sys
import time

from benchmark.perf_guide import common

QUARTER = common.TAXI_QUARTER
YEAR = common.TAXI_GLOB
FEATURES = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
]
LABEL = "total_amount"
FEATURE_SQL = f"""
SELECT
    toFloat32(ifNull(trip_distance, 0)) AS trip_distance,
    toFloat32(ifNull(passenger_count, 0)) AS passenger_count,
    toFloat32(PULocationID) AS PULocationID,
    toFloat32(DOLocationID) AS DOLocationID,
    toFloat32(fare_amount) AS fare_amount,
    toFloat32(tip_amount) AS tip_amount,
    toFloat32(tolls_amount) AS tolls_amount,
    toFloat32({LABEL}) AS {LABEL},
    cityHash64(tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID, fare_amount, total_amount) AS rk
FROM file('{{path}}', Parquet)
"""


def feature_sql(scope="quarter"):
    return FEATURE_SQL.format(path=QUARTER if scope == "quarter" else YEAR)


ENGINEERED_SQL = f"""
SELECT
    toFloat32(ifNull(trip_distance, 0)) AS trip_distance,
    toFloat32(log1p(ifNull(trip_distance, 0))) AS log_distance,
    toFloat32(toHour(tpep_pickup_datetime)) AS hour,
    toFloat32(toDayOfWeek(tpep_pickup_datetime)) AS dow,
    toFloat32(dateDiff('second', tpep_pickup_datetime, tpep_dropoff_datetime)) AS duration_s,
    toFloat32(ifNull(trip_distance, 0) / greatest(dateDiff('second', tpep_pickup_datetime, tpep_dropoff_datetime), 1) * 3600) AS mph,
    toFloat32(PULocationID) AS PULocationID,
    toFloat32(DOLocationID) AS DOLocationID,
    toFloat32(fare_amount) AS fare_amount,
    toFloat32({LABEL}) AS {LABEL}
FROM file('{QUARTER}', Parquet)
"""
HEAVY_SQL = f"""
SELECT
    toFloat32(ifNull(trip_distance, 0)) AS trip_distance,
    toFloat32(toHour(tpep_pickup_datetime)) AS hour,
    toFloat32(PULocationID) AS PULocationID,
    toFloat32(fare_amount) AS fare_amount,
    toFloat32(avg(fare_amount) OVER w) AS zone_fare_ma,
    toFloat32(fare_amount - avg(fare_amount) OVER w) AS fare_minus_ma,
    toFloat32(count() OVER w) AS zone_recent_trips,
    toFloat32({LABEL}) AS {LABEL}
FROM file('{QUARTER}', Parquet)
WINDOW w AS (PARTITION BY PULocationID ORDER BY tpep_pickup_datetime ROWS BETWEEN 200 PRECEDING AND CURRENT ROW)
"""
ENGINEERED_FEATURES = [
    "trip_distance",
    "log_distance",
    "hour",
    "dow",
    "duration_s",
    "mph",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
]
PRECOMPUTED = os.path.join(common.DATA, "ml", "features.parquet")
PRECOMPUTED_HEAVY = os.path.join(common.DATA, "ml", "features_heavy.parquet")


# --------------------------------------------------------------------------
# streaming throughput
# --------------------------------------------------------------------------


def iter_batches_throughput(batch_size=65536, output="arrow", scope="quarter", threads=None):
    from chdb.pipeline import iter_batches

    conn = common.session(threads=threads)
    rows = 0
    batches = 0
    start = time.perf_counter()
    for batch in iter_batches(feature_sql(scope), batch_size=batch_size, conn=conn, output=output):
        batches += 1
        if output == "arrow":
            rows += batch.num_rows
        elif output == "numpy":
            rows += len(batch[LABEL])
        else:
            rows += len(batch)
    elapsed = time.perf_counter() - start
    conn.close()
    return {"rows": rows, "batches": batches, "rows_per_s": int(rows / elapsed)}


# --------------------------------------------------------------------------
# PyTorch
# --------------------------------------------------------------------------


def torch_dataloader(num_workers=0, batch_size=65536, scope="quarter", threads=None):
    import resource

    import torch
    from torch.utils.data import DataLoader

    from chdb.pipeline import TorchIterableDataset

    threads = common.THREADS if threads is None else threads
    sql = feature_sql(scope) + f" SETTINGS max_threads={threads}, max_block_size={batch_size}"
    dataset = TorchIterableDataset(sql, FEATURES, LABEL, batch_size=batch_size, key="rk" if num_workers else None)
    kwargs = {"batch_size": None}
    if num_workers:
        kwargs.update(num_workers=num_workers, multiprocessing_context="spawn", persistent_workers=False)
    loader = DataLoader(dataset, **kwargs)
    rows = 0
    total = torch.zeros((), dtype=torch.float64)
    start = time.perf_counter()
    for features, label in loader:
        rows += features.shape[0]
        total += label.double().sum()
    elapsed = time.perf_counter() - start
    del loader
    children_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return {
        "rows": rows,
        "rows_per_s": int(rows / elapsed),
        "checksum": round(total.item(), 2),
        "worker_peak_rss_mb": round(children_kb / 1024, 1),
    }


def torch_eager(batch_size=65536, scope="quarter"):
    """Load everything with to_df(), convert once, slice tensors per batch."""
    import torch

    conn = common.session()
    df = conn.query(feature_sql(scope), "DataFrame")
    conn.close()
    x = torch.from_numpy(df[FEATURES].to_numpy(dtype="float32"))
    y = torch.from_numpy(df[LABEL].to_numpy(dtype="float32"))
    del df
    rows = 0
    total = torch.zeros((), dtype=torch.float64)
    start = time.perf_counter()
    for i in range(0, x.shape[0], batch_size):
        rows += x[i : i + batch_size].shape[0]
        total += y[i : i + batch_size].double().sum()
    elapsed = time.perf_counter() - start
    return {"rows": rows, "rows_per_s": int(rows / elapsed), "checksum": round(total.item(), 2)}


def torch_pandas_eager(batch_size=65536, scope="quarter"):
    """pandas.read_parquet + feature engineering in pandas, then slice."""
    import glob

    import numpy as np
    import pandas as pd
    import torch

    cols = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "tip_amount",
        "tolls_amount",
        LABEL,
    ]
    df = pd.concat(
        [pd.read_parquet(p, columns=cols) for p in common.py_glob(QUARTER if scope == "quarter" else YEAR)],
        ignore_index=True,
    )
    df["trip_distance"] = df["trip_distance"].fillna(0)
    df["passenger_count"] = df["passenger_count"].fillna(0)
    x = torch.from_numpy(df[FEATURES].to_numpy(dtype=np.float32))
    y = torch.from_numpy(df[LABEL].to_numpy(dtype=np.float32))
    del df
    rows = 0
    total = torch.zeros((), dtype=torch.float64)
    for i in range(0, x.shape[0], batch_size):
        rows += x[i : i + batch_size].shape[0]
        total += y[i : i + batch_size].double().sum()
    return {"rows": rows, "checksum": round(total.item(), 2)}


# --------------------------------------------------------------------------
# per-epoch cost: recompute features vs precomputed Parquet
# --------------------------------------------------------------------------


def _consume(sql, batch_size=65536):
    from chdb.pipeline import iter_batches

    conn = common.session()
    rows = 0
    for batch in iter_batches(sql, batch_size=batch_size, conn=conn, output="numpy"):
        rows += len(batch[LABEL])
    conn.close()
    return rows


def epochs_recompute(epochs=3, heavy=False):
    rows = 0
    for _ in range(epochs):
        rows += _consume(HEAVY_SQL if heavy else ENGINEERED_SQL)
    return {"rows": rows}


def precompute_features(heavy=False):
    path = PRECOMPUTED_HEAVY if heavy else PRECOMPUTED
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = common.session()
    conn.query(f"INSERT INTO FUNCTION file('{path}', Parquet) {HEAVY_SQL if heavy else ENGINEERED_SQL}")
    conn.close()
    return {"features_mb": common.file_size_mb(path)}


def epochs_precomputed(epochs=3, heavy=False):
    path = PRECOMPUTED_HEAVY if heavy else PRECOMPUTED
    rows = 0
    for _ in range(epochs):
        rows += _consume(f"SELECT * FROM file('{path}', Parquet)")
    return {"rows": rows}


# --------------------------------------------------------------------------
# train/validation split by hash
# --------------------------------------------------------------------------


def split_scan(mode="full"):
    conn = common.session()
    before = common.engine_counters(conn)
    if mode == "full":
        sql = f"SELECT count(), sum(total_amount) FROM ({feature_sql()})"
    elif mode == "train_80":
        sql = f"SELECT count(), sum(total_amount) FROM ({feature_sql()}) WHERE rk % 10 < 8"
    elif mode == "valid_20":
        sql = f"SELECT count(), sum(total_amount) FROM ({feature_sql()}) WHERE rk % 10 >= 8"
    elif mode == "shard_0_of_4":
        from chdb.pipeline import shard_sql

        sql = f"SELECT count(), sum(total_amount) FROM ({shard_sql(feature_sql(), 0, 4, 'rk')})"
    else:
        raise ValueError(mode)
    text = conn.query(sql, "TSV").bytes().decode().strip()
    after = common.engine_counters(conn)
    conn.close()
    count, total = text.split("\t")
    out = {"rows_out": int(count), "checksum": round(float(total), 2)}
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    suite = "bench_ml"
    cases = []
    run = lambda case, **args: cases.append(common.run_case(suite, case, args))  # noqa: E731

    print("== iter_batches throughput")
    for batch_size in (4096, 65536, 1048576):
        run("iter_batches_throughput", batch_size=batch_size, output="arrow")
    for output in ("numpy", "pandas"):
        run("iter_batches_throughput", batch_size=65536, output=output)
    for threads in (1, 2, 4):
        run("iter_batches_throughput", batch_size=65536, output="arrow", threads=threads)

    print("== PyTorch, one quarter (9M rows)")
    run("torch_pandas_eager")
    run("torch_eager")
    for workers in (0, 2, 4):
        run("torch_dataloader", num_workers=workers)

    print("== PyTorch, full year (41M rows)")
    run("torch_pandas_eager", scope="year")
    run("torch_eager", scope="year")
    run("torch_dataloader", num_workers=0, scope="year")
    run("torch_dataloader", num_workers=0, scope="year", threads=2)
    run("torch_dataloader", num_workers=4, scope="year", threads=2)

    print("== epochs: recompute vs precomputed")
    run("epochs_recompute", epochs=3)
    cases.append(common.run_case(suite, "precompute_features", {}, repeat=1))
    run("epochs_precomputed", epochs=3)
    run("epochs_recompute", epochs=3, heavy=True)
    cases.append(common.run_case(suite, "precompute_features", {"heavy": True}, repeat=1))
    run("epochs_precomputed", epochs=3, heavy=True)

    print("== hash split")
    for mode in ("full", "train_80", "valid_20", "shard_0_of_4"):
        run("split_scan", mode=mode)

    common.save(
        suite,
        cases,
        {"datasets": {"taxi_quarter_mb": common.file_size_mb(QUARTER), "taxi_year_mb": common.file_size_mb(YEAR)}},
    )


if __name__ == "__main__":
    sys.exit(main())
