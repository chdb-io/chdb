"""SQL techniques that change how much work the engine does.

Datasets: NYC taxi 2024 (12 Parquet files, ~41M rows) and a MergeTree copy of
one week of aggTrades built by this suite.
"""

import os
import shutil
import sys

from benchmark.perf_guide import common

TAXI = common.TAXI_GLOB
MONTH = common.TAXI_MONTH
WEEK_GLOB = common.TICKS_WEEK
MT_DIR = os.path.join(common.DATA, "mergetree_sql")


def _measure(conn, sql, fmt="TSV"):
    before = common.engine_counters(conn)
    text = conn.query(sql, fmt).bytes().decode().strip()
    after = common.engine_counters(conn)
    out = {"value": text[:120]}
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# MergeTree fixture (ticks, one week) and taxi variants
# --------------------------------------------------------------------------


def build_tables():
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
    # taxi: String vs LowCardinality, Nullable vs plain
    conn.query(
        """
        CREATE TABLE bench.taxi_plain
        (pickup DateTime, vendor String, payment String, passenger_count Nullable(Float64), total_amount Float64, pu Int32)
        ENGINE = MergeTree ORDER BY pickup
        """
    )
    conn.query(
        """
        CREATE TABLE bench.taxi_tuned
        (pickup DateTime, vendor LowCardinality(String), payment LowCardinality(String), passenger_count Float64, total_amount Float64, pu Int32)
        ENGINE = MergeTree ORDER BY pickup
        """
    )
    select = f"""
        SELECT tpep_pickup_datetime, concat('vendor_', toString(VendorID)), concat('pay_', toString(payment_type)),
               passenger_count, total_amount, PULocationID
        FROM file('{TAXI}', Parquet)
    """
    conn.query(f"INSERT INTO bench.taxi_plain {select}")
    conn.query(
        f"""
        INSERT INTO bench.taxi_tuned
        SELECT tpep_pickup_datetime, concat('vendor_', toString(VendorID)), concat('pay_', toString(payment_type)),
               ifNull(passenger_count, 0), total_amount, PULocationID
        FROM file('{TAXI}', Parquet)
        """
    )
    sizes = (
        conn.query(
            "SELECT table, sum(bytes_on_disk) FROM system.parts WHERE database = 'bench' AND active GROUP BY table ORDER BY table",
            "TSV",
        )
        .bytes()
        .decode()
        .strip()
        .splitlines()
    )
    conn.close()
    return {f"{name}_mb": round(int(size) / 1e6, 1) for name, size in (line.split("\t") for line in sizes)}


# --------------------------------------------------------------------------
# PREWHERE
# --------------------------------------------------------------------------


def prewhere(mode="auto"):
    conn = common.session(path=MT_DIR)
    settings = {"auto": "", "off": "SETTINGS optimize_move_to_prewhere = 0", "explicit": ""}[mode]
    clause = "PREWHERE qty > 5" if mode == "explicit" else "WHERE qty > 5"
    sql = f"SELECT count(), sum(price * qty) FROM bench.ticks {clause} {settings}"
    out = _measure(conn, sql)
    conn.close()
    return out


# --------------------------------------------------------------------------
# last row per group
# --------------------------------------------------------------------------


def last_per_minute(mode="argmax"):
    conn = common.session(path=MT_DIR)
    if mode == "argmax":
        sql = """
            SELECT count(), sum(last_price) FROM (
                SELECT symbol, toStartOfMinute(ts) AS m, argMax(price, (ts, agg_trade_id)) AS last_price
                FROM bench.ticks GROUP BY symbol, m)
        """
    elif mode == "limit_by":
        sql = """
            SELECT count(), sum(price) FROM (
                SELECT symbol, toStartOfMinute(ts) AS m, price
                FROM bench.ticks ORDER BY ts DESC, agg_trade_id DESC LIMIT 1 BY symbol, m)
        """
    elif mode == "row_number":
        sql = """
            SELECT count(), sum(price) FROM (
                SELECT symbol, toStartOfMinute(ts) AS m, price,
                       row_number() OVER (PARTITION BY symbol, toStartOfMinute(ts) ORDER BY ts DESC, agg_trade_id DESC) AS rn
                FROM bench.ticks) WHERE rn = 1
        """
    elif mode == "self_join":
        sql = """
            SELECT count(), sum(t.price) FROM bench.ticks AS t
            INNER JOIN (
                SELECT symbol, toStartOfMinute(ts) AS m, max(ts) AS mts FROM bench.ticks GROUP BY symbol, m
            ) AS last ON t.symbol = last.symbol AND t.ts = last.mts
        """
    else:
        raise ValueError(mode)
    out = _measure(conn, sql)
    conn.close()
    return out


# --------------------------------------------------------------------------
# approximate aggregates on 41M taxi rows
# --------------------------------------------------------------------------


def distinct_count(mode="uniq"):
    conn = common.session()
    fn = {"uniq": "uniq", "uniqExact": "uniqExact", "uniqCombined": "uniqCombined", "count_distinct": "count(DISTINCT"}[
        mode
    ]
    expr = f"{fn} tpep_pickup_datetime)" if mode == "count_distinct" else f"{fn}(tpep_pickup_datetime)"
    out = _measure(conn, f"SELECT {expr} FROM file('{TAXI}', Parquet)")
    conn.close()
    return out


def quantiles(mode="quantile"):
    conn = common.session()
    fn = {"quantile": "quantiles", "quantileTDigest": "quantilesTDigest", "quantileExact": "quantilesExact"}[mode]
    out = _measure(conn, f"SELECT {fn}(0.5, 0.9, 0.99)(total_amount) FROM file('{TAXI}', Parquet)")
    conn.close()
    return out


# --------------------------------------------------------------------------
# schema choices: LowCardinality, Nullable
# --------------------------------------------------------------------------


def schema_groupby(table="taxi_plain"):
    conn = common.session(path=MT_DIR)
    out = _measure(
        conn,
        f"SELECT vendor, payment, count(), sum(total_amount), avg(passenger_count) FROM bench.{table} GROUP BY vendor, payment ORDER BY vendor, payment",
    )
    conn.close()
    return out


def schema_filter(table="taxi_plain"):
    conn = common.session(path=MT_DIR)
    out = _measure(
        conn, f"SELECT count(), sum(total_amount) FROM bench.{table} WHERE payment = 'pay_1' AND passenger_count > 2"
    )
    conn.close()
    return out


# --------------------------------------------------------------------------
# reading in primary-key order, LIMIT, count
# --------------------------------------------------------------------------


def order_limit(mode="in_order"):
    conn = common.session(path=MT_DIR)
    setting = "" if mode == "in_order" else "SETTINGS optimize_read_in_order = 0"
    out = _measure(
        conn, f"SELECT ts, price FROM bench.ticks WHERE symbol = 'BTCUSDT' ORDER BY ts DESC LIMIT 10 {setting}"
    )
    conn.close()
    return out


def count_rows(mode="count_star"):
    conn = common.session()
    if mode == "count_star":
        sql = f"SELECT count() FROM file('{TAXI}', Parquet)"
    elif mode == "count_column":
        sql = f"SELECT count(total_amount) FROM file('{TAXI}', Parquet)"
    elif mode == "count_filtered":
        sql = f"SELECT count() FROM file('{TAXI}', Parquet) WHERE total_amount > 20"
    else:
        raise ValueError(mode)
    out = _measure(conn, sql)
    conn.close()
    return out


# --------------------------------------------------------------------------
# DataStore: boolean index vs ds.sql()
# --------------------------------------------------------------------------


def datastore_filter(mode="boolean_index"):
    import chdb.datastore as pd
    from chdb.datastore.config import config

    config.use_performance_mode()
    conn = common.session()
    before = common.engine_counters(conn)
    ds = pd.read_parquet(TAXI)
    if mode == "boolean_index":
        sub = ds[(ds["total_amount"] > 200) & (ds["passenger_count"] >= 4)][["total_amount", "trip_distance"]]
        n = len(sub)
    elif mode == "sql":
        sub = ds.sql("SELECT total_amount, trip_distance FROM __df__ WHERE total_amount > 200 AND passenger_count >= 4")
        n = len(sub)
    else:
        raise ValueError(mode)
    after = common.engine_counters(conn)
    conn.close()
    out = {"rows_out": n}
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    suite = "bench_sql"
    cases = []
    run = lambda case, **args: cases.append(common.run_case(suite, case, args))  # noqa: E731

    print("== build fixtures")
    cases.append(common.run_case(suite, "build_tables", {}, repeat=1))

    print("== PREWHERE")
    for mode in ("off", "auto", "explicit"):
        run("prewhere", mode=mode)

    print("== last row per group")
    for mode in ("argmax", "limit_by", "row_number", "self_join"):
        run("last_per_minute", mode=mode)

    print("== approximate aggregates")
    for mode in ("uniqExact", "count_distinct", "uniq", "uniqCombined"):
        run("distinct_count", mode=mode)
    for mode in ("quantileExact", "quantile", "quantileTDigest"):
        run("quantiles", mode=mode)

    print("== schema choices")
    for table in ("taxi_plain", "taxi_tuned"):
        run("schema_groupby", table=table)
        run("schema_filter", table=table)

    print("== order/limit/count")
    for mode in ("in_order", "no_in_order"):
        run("order_limit", mode=mode)
    for mode in ("count_star", "count_column", "count_filtered"):
        run("count_rows", mode=mode)

    print("== DataStore filters")
    for mode in ("boolean_index", "sql"):
        run("datastore_filter", mode=mode)

    common.save(
        suite,
        cases,
        {"datasets": {"taxi_year_mb": common.file_size_mb(TAXI), "week_mb": common.file_size_mb(WEEK_GLOB)}},
    )
    shutil.rmtree(MT_DIR, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
