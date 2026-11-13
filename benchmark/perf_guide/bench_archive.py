"""Archiving: Parquet compression, row-group size, sort order, partitioned writes, DROP PARTITION.

Dataset: 7 days of BTCUSDT+ETHUSDT aggTrades (~30M rows) for the write cases;
NYC taxi January (~3M rows) for the partitioned-write case.
"""

import glob
import os
import shutil
import sys

from benchmark.perf_guide import common

WEEK_GLOB = common.TICKS_WEEK
OUT_DIR = os.path.join(common.DATA, "archive")
MT_DIR = os.path.join(common.DATA, "mergetree_archive")
DAY = "2024-01-04"
RANGE_WHERE = (
    f"WHERE symbol = 'BTCUSDT' AND ts >= toDateTime64('{DAY} 12:00:00', 3) AND ts < toDateTime64('{DAY} 13:00:00', 3)"
)


def _out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def _write(conn, path, settings="", order="ORDER BY symbol, ts, agg_trade_id"):
    if os.path.exists(path):
        os.remove(path)
    conn.query(
        f"""
        INSERT INTO FUNCTION file('{path}', Parquet)
        SELECT symbol, ts, agg_trade_id, price, qty, is_buyer_maker
        FROM file('{WEEK_GLOB}', Parquet)
        {order}
        {settings}
        """
    )


# --------------------------------------------------------------------------
# compression codecs
# --------------------------------------------------------------------------


def write_compression(method="zstd"):
    conn = common.session()
    path = _out(f"week_{method}.parquet")
    _write(conn, path, f"SETTINGS output_format_parquet_compression_method = '{method}'")
    conn.close()
    return {"file_mb": common.file_size_mb(path)}


def read_compression(method="zstd"):
    conn = common.session()
    path = _out(f"week_{method}.parquet")
    text = conn.query(f"SELECT count(), sum(qty) FROM file('{path}', Parquet)", "TSV").bytes().decode().strip()
    conn.close()
    count, qty = text.split("\t")
    return {"rows": int(count), "checksum": round(float(qty), 3)}


# --------------------------------------------------------------------------
# row-group size and sort order
# --------------------------------------------------------------------------


def write_row_group(rows=1000000, sorted_=True):
    conn = common.session()
    name = f"week_rg{rows}_{'sorted' if sorted_ else 'shuffled'}.parquet"
    path = _out(name)
    order = "ORDER BY symbol, ts, agg_trade_id" if sorted_ else "ORDER BY rand()"
    _write(
        conn,
        path,
        f"SETTINGS output_format_parquet_row_group_size = {rows}, output_format_parquet_compression_method = 'zstd'",
        order,
    )
    conn.close()
    return {"file_mb": common.file_size_mb(path)}


def range_query(rows=1000000, sorted_=True):
    conn = common.session()
    name = f"week_rg{rows}_{'sorted' if sorted_ else 'shuffled'}.parquet"
    path = _out(name)
    before = common.engine_counters(conn)
    text = (
        conn.query(f"SELECT count(), sum(qty) FROM file('{path}', Parquet) {RANGE_WHERE}", "TSV")
        .bytes()
        .decode()
        .strip()
    )
    after = common.engine_counters(conn)
    conn.close()
    count, qty = text.split("\t")
    out = {"rows_out": int(count), "checksum": round(float(qty), 3)}
    out.update(common.counter_delta(before, after))
    return out


def point_lookup(rows=1000000, sorted_=True):
    """Single-key lookup: bloom filter and min/max statistics at work."""
    conn = common.session()
    name = f"week_rg{rows}_{'sorted' if sorted_ else 'shuffled'}.parquet"
    path = _out(name)
    key = (
        conn.query(
            f"SELECT agg_trade_id FROM file('{WEEK_GLOB}', Parquet) WHERE symbol = 'BTCUSDT' AND day = '{DAY}' ORDER BY ts LIMIT 1 OFFSET 500000",
            "CSV",
        )
        .bytes()
        .decode()
        .strip()
    )
    before = common.engine_counters(conn)
    text = (
        conn.query(
            f"SELECT count() FROM file('{path}', Parquet) WHERE agg_trade_id = {key} AND symbol = 'BTCUSDT'", "TSV"
        )
        .bytes()
        .decode()
        .strip()
    )
    after = common.engine_counters(conn)
    conn.close()
    out = {"rows_out": int(text)}
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# partitioned writes
# --------------------------------------------------------------------------


def partitioned_write(mode="partition_by"):
    conn = common.session()
    target = os.path.join(OUT_DIR, f"taxi_{mode}")
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)
    if mode == "partition_by":
        conn.query(
            f"""
            INSERT INTO FUNCTION file('{target}/day={{_partition_id}}/trips.parquet', Parquet)
            PARTITION BY toDate(tpep_pickup_datetime)
            SELECT * FROM file('{common.TAXI_MONTH}', Parquet)
            WHERE toYYYYMM(tpep_pickup_datetime) = 202401
            SETTINGS output_format_parquet_compression_method = 'zstd'
            """
        )
    elif mode == "loop":
        days = (
            conn.query(
                f"SELECT DISTINCT toDate(tpep_pickup_datetime) AS d FROM file('{common.TAXI_MONTH}', Parquet) WHERE toYYYYMM(tpep_pickup_datetime) = 202401 ORDER BY d",
                "TSV",
            )
            .bytes()
            .decode()
            .split()
        )
        for day in days:
            os.makedirs(f"{target}/day={day}", exist_ok=True)
            conn.query(
                f"""
                INSERT INTO FUNCTION file('{target}/day={day}/trips.parquet', Parquet)
                SELECT * FROM file('{common.TAXI_MONTH}', Parquet)
                WHERE toDate(tpep_pickup_datetime) = '{day}'
                SETTINGS output_format_parquet_compression_method = 'zstd'
                """
            )
    else:
        raise ValueError(mode)
    conn.close()
    files = glob.glob(os.path.join(target, "**", "*.parquet"), recursive=True)
    return {"files": len(files), "total_mb": common.file_size_mb(os.path.join(target, "**", "*.parquet"))}


# --------------------------------------------------------------------------
# MergeTree retention: DROP PARTITION, export to Parquet
# --------------------------------------------------------------------------


def mergetree_build():
    shutil.rmtree(MT_DIR, ignore_errors=True)
    conn = common.session(path=MT_DIR)
    conn.query("CREATE DATABASE IF NOT EXISTS bench")
    conn.query(
        """
        CREATE TABLE bench.ticks
        (symbol LowCardinality(String), ts DateTime64(3), agg_trade_id Int64, price Float64, qty Float64, is_buyer_maker Bool)
        ENGINE = MergeTree PARTITION BY toYYYYMMDD(ts) ORDER BY (symbol, ts, agg_trade_id)
        """
    )
    conn.query(
        f"INSERT INTO bench.ticks SELECT symbol, ts, agg_trade_id, price, qty, is_buyer_maker FROM file('{WEEK_GLOB}', Parquet)"
    )
    parts = (
        conn.query(
            "SELECT count(), sum(bytes_on_disk) FROM system.parts WHERE database='bench' AND table='ticks' AND active",
            "TSV",
        )
        .bytes()
        .decode()
        .strip()
        .split("\t")
    )
    conn.close()
    return {"parts": int(parts[0]), "mergetree_mb": round(int(parts[1]) / 1e6, 1)}


def export_partition():
    conn = common.session(path=MT_DIR)
    path = _out("exported_20240101.parquet")
    if os.path.exists(path):
        os.remove(path)
    conn.query(
        f"""
        INSERT INTO FUNCTION file('{path}', Parquet)
        SELECT * FROM bench.ticks WHERE toYYYYMMDD(ts) = 20240101
        SETTINGS output_format_parquet_compression_method = 'zstd'
        """
    )
    rows = int(conn.query(f"SELECT count() FROM file('{path}', Parquet)", "CSV").bytes().strip())
    conn.close()
    return {"rows": rows, "file_mb": common.file_size_mb(path)}


def drop_partition():
    conn = common.session(path=MT_DIR)
    before = int(conn.query("SELECT count() FROM bench.ticks", "CSV").bytes().strip())
    conn.query("ALTER TABLE bench.ticks DROP PARTITION 20240101")
    after = int(conn.query("SELECT count() FROM bench.ticks", "CSV").bytes().strip())
    conn.close()
    return {"rows_before": before, "rows_after": after}


def delete_rows():
    """The slow alternative: a mutation that rewrites parts."""
    conn = common.session(path=MT_DIR)
    before = int(conn.query("SELECT count() FROM bench.ticks", "CSV").bytes().strip())
    conn.query("ALTER TABLE bench.ticks DELETE WHERE toYYYYMMDD(ts) = 20240102 SETTINGS mutations_sync = 2")
    after = int(conn.query("SELECT count() FROM bench.ticks", "CSV").bytes().strip())
    conn.close()
    return {"rows_before": before, "rows_after": after}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    suite = "bench_archive"
    cases = []
    run = lambda case, **args: cases.append(common.run_case(suite, case, args))  # noqa: E731
    once = lambda case, **args: cases.append(common.run_case(suite, case, args, repeat=1))  # noqa: E731

    print("== compression codecs (7 days, two symbols)")
    for method in ("none", "snappy", "lz4", "zstd", "gzip"):
        run("write_compression", method=method)
        run("read_compression", method=method)

    print("== row-group size and sort order")
    for rows in (100000, 1000000, 10000000):
        run("write_row_group", rows=rows, sorted_=True)
        run("range_query", rows=rows, sorted_=True)
        run("point_lookup", rows=rows, sorted_=True)
    run("write_row_group", rows=1000000, sorted_=False)
    run("range_query", rows=1000000, sorted_=False)
    run("point_lookup", rows=1000000, sorted_=False)

    print("== partitioned write of one taxi month")
    run("partitioned_write", mode="partition_by")
    run("partitioned_write", mode="loop")

    print("== MergeTree retention")
    once("mergetree_build")
    once("export_partition")
    once("drop_partition")
    once("delete_rows")

    common.save(
        suite,
        cases,
        {
            "datasets": {
                "week_mb": common.file_size_mb(WEEK_GLOB),
                "taxi_month_mb": common.file_size_mb(common.TAXI_MONTH),
            }
        },
    )
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    shutil.rmtree(MT_DIR, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
