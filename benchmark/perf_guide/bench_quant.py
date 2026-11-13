"""Quant workloads: bars, partition pruning, MergeTree vs Parquet, ASOF JOIN, rolling windows.

Dataset: Binance spot aggTrades (BTCUSDT, ETHUSDT) for January 2024 as
Hive-partitioned Parquet, plus BTCUSDT 1-second klines.
"""

import glob
import os
import sys

from benchmark.perf_guide import common

TICKS = common.TICKS_HIVE
TICKS_DIR = common.TICKS_DIR
KLINES = common.KLINES_HIVE
MT_DIR = common.MERGETREE_DIR
DAY = "2024-01-15"
DAY_BTC = os.path.join(TICKS_DIR, f"day={DAY}", "symbol=BTCUSDT", "trades.parquet")
DAY_KLINES = os.path.join(common.DATA, "klines", f"day={DAY}", "klines.parquet")
BTC_GLOB = os.path.join(TICKS_DIR, "day=*", "symbol=BTCUSDT", "trades.parquet")

BARS_SQL = """
SELECT
    toStartOfMinute(ts) AS minute,
    argMin(price, (ts, agg_trade_id)) AS open,
    max(price) AS high,
    min(price) AS low,
    argMax(price, (ts, agg_trade_id)) AS close,
    sum(qty) AS volume,
    sum(price * qty) / sum(qty) AS vwap,
    count() AS trades
FROM {source}
GROUP BY minute
ORDER BY minute
"""


def _btc_frames(columns=("ts", "price", "qty", "agg_trade_id")):
    import pyarrow.parquet as pq

    tables = [pq.read_table(p, columns=list(columns)) for p in sorted(glob.glob(BTC_GLOB))]
    return tables


# --------------------------------------------------------------------------
# 1-minute OHLCV bars over 31 days of BTCUSDT
# --------------------------------------------------------------------------


def bars_sql():
    conn = common.session()
    before = common.engine_counters(conn)
    df = conn.query(BARS_SQL.format(source=f"file('{BTC_GLOB}', Parquet)"), "DataFrame")
    after = common.engine_counters(conn)
    conn.close()
    out = {"bars": len(df), "checksum": round(float(df["volume"].sum()), 3)}
    out.update(common.counter_delta(before, after))
    return out


def bars_pandas():
    import pyarrow as pa

    tables = _btc_frames()
    df = pa.concat_tables(tables).to_pandas()
    del tables
    df = df.sort_values(["ts", "agg_trade_id"])
    df["notional"] = df["price"] * df["qty"]
    grouped = df.set_index("ts").resample("1min")
    bars = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        notional=("notional", "sum"),
        trades=("price", "size"),
    )
    bars = bars[bars["trades"] > 0]
    bars["vwap"] = bars["notional"] / bars["volume"]
    return {
        "bars": len(bars),
        "checksum": round(float(bars["volume"].sum()), 3),
        "df_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }


def bars_pandas_per_day():
    """Same as bars_pandas but one day at a time (what a careful pandas user does)."""
    import pandas as pd
    import pyarrow.parquet as pq

    pieces = []
    for path in sorted(glob.glob(BTC_GLOB)):
        df = pq.read_table(path, columns=["ts", "price", "qty", "agg_trade_id"]).to_pandas()
        df = df.sort_values(["ts", "agg_trade_id"])
        df["notional"] = df["price"] * df["qty"]
        bars = (
            df.set_index("ts")
            .resample("1min")
            .agg(
                open=("price", "first"),
                high=("price", "max"),
                low=("price", "min"),
                close=("price", "last"),
                volume=("qty", "sum"),
                notional=("notional", "sum"),
                trades=("price", "size"),
            )
        )
        pieces.append(bars[bars["trades"] > 0])
    bars = pd.concat(pieces)
    return {"bars": len(bars), "checksum": round(float(bars["volume"].sum()), 3)}


def bars_stream_sql(batch_size=65536):
    """Bars via SQL, but consumed as a stream (bounded Python-side memory)."""
    from chdb.pipeline import iter_batches

    conn = common.session()
    rows = 0
    volume = 0.0
    for batch in iter_batches(
        BARS_SQL.format(source=f"file('{BTC_GLOB}', Parquet)"), batch_size=batch_size, conn=conn, output="numpy"
    ):
        rows += len(batch["minute"])
        volume += float(batch["volume"].sum())
    conn.close()
    return {"bars": rows, "checksum": round(volume, 3)}


# --------------------------------------------------------------------------
# partition pruning on Hive layout
# --------------------------------------------------------------------------


def one_day(mode="hive_filter"):
    conn = common.session()
    before = common.engine_counters(conn)
    if mode == "hive_filter":
        sql = f"SELECT count(), sum(qty) FROM file('{TICKS}', Parquet) WHERE day = '{DAY}' AND symbol = 'BTCUSDT'"
    elif mode == "ts_filter":
        sql = (
            f"SELECT count(), sum(qty) FROM file('{TICKS}', Parquet) "
            f"WHERE ts >= toDateTime64('{DAY} 00:00:00', 3) AND ts < toDateTime64('{DAY} 00:00:00', 3) + INTERVAL 1 DAY AND symbol = 'BTCUSDT'"
        )
    elif mode == "ts_filter_no_pushdown":
        sql = (
            f"SELECT count(), sum(qty) FROM file('{TICKS}', Parquet) "
            f"WHERE ts >= toDateTime64('{DAY} 00:00:00', 3) AND ts < toDateTime64('{DAY} 00:00:00', 3) + INTERVAL 1 DAY AND symbol = 'BTCUSDT' "
            "SETTINGS input_format_parquet_filter_push_down = 0"
        )
    elif mode == "explicit_path":
        sql = f"SELECT count(), sum(qty) FROM file('{DAY_BTC}', Parquet)"
    elif mode == "all_days":
        sql = f"SELECT count(), sum(qty) FROM file('{TICKS}', Parquet) WHERE symbol = 'BTCUSDT'"
    elif mode == "unprunable_filter":
        sql = f"SELECT count(), sum(qty) FROM file('{TICKS}', Parquet) WHERE symbol = 'BTCUSDT' AND qty > 50"
    elif mode == "full_scan":
        sql = f"SELECT count(), sum(qty) FROM file('{TICKS}', Parquet) WHERE toDate(ts) = '{DAY}' AND symbol = 'BTCUSDT' SETTINGS input_format_parquet_filter_push_down = 0"
    else:
        raise ValueError(mode)
    text = conn.query(sql, "TSV").bytes().decode().strip()
    after = common.engine_counters(conn)
    conn.close()
    count, qty = text.split("\t")
    out = {"rows_out": int(count), "checksum": round(float(qty), 3)}
    out.update(common.counter_delta(before, after))
    return out


def one_hour(mode="parquet"):
    """A one-hour range for one symbol: Parquet row-group statistics vs MergeTree primary key."""
    start = f"toDateTime64('{DAY} 12:00:00', 3)"
    end = f"toDateTime64('{DAY} 13:00:00', 3)"
    if mode == "parquet":
        conn = common.session()
        source = f"file('{TICKS}', Parquet)"
        where = f"WHERE day = '{DAY}' AND symbol = 'BTCUSDT' AND ts >= {start} AND ts < {end}"
    elif mode == "parquet_no_hive":
        conn = common.session()
        source = f"file('{TICKS}', Parquet)"
        where = f"WHERE symbol = 'BTCUSDT' AND ts >= {start} AND ts < {end}"
    elif mode == "mergetree":
        conn = common.session(path=MT_DIR)
        source = "bench.ticks"
        where = f"WHERE symbol = 'BTCUSDT' AND ts >= {start} AND ts < {end}"
    else:
        raise ValueError(mode)
    before = common.engine_counters(conn)
    text = conn.query(f"SELECT count(), sum(qty), max(price) FROM {source} {where}", "TSV").bytes().decode().strip()
    after = common.engine_counters(conn)
    conn.close()
    count, qty, mx = text.split("\t")
    out = {"rows_out": int(count), "checksum": round(float(qty), 3)}
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# MergeTree: load, size, full scan
# --------------------------------------------------------------------------


def mergetree_load():
    import shutil

    shutil.rmtree(MT_DIR, ignore_errors=True)
    conn = common.session(path=MT_DIR)
    conn.query("CREATE DATABASE IF NOT EXISTS bench")
    conn.query(
        """
        CREATE TABLE bench.ticks
        (
            symbol LowCardinality(String),
            ts DateTime64(3),
            agg_trade_id Int64,
            price Float64,
            qty Float64,
            is_buyer_maker Bool
        )
        ENGINE = MergeTree
        PARTITION BY toYYYYMMDD(ts)
        ORDER BY (symbol, ts, agg_trade_id)
        """
    )
    conn.query(
        f"""
        INSERT INTO bench.ticks
        SELECT symbol, ts, agg_trade_id, price, qty, is_buyer_maker
        FROM file('{TICKS}', Parquet)
        """
    )
    parts_sql = "SELECT sum(rows), sum(bytes_on_disk), count() FROM system.parts WHERE database = 'bench' AND table = 'ticks' AND active"
    rows, bytes_on_disk, parts = conn.query(parts_sql, "TSV").bytes().decode().strip().split("\t")
    import time

    start = time.perf_counter()
    conn.query("OPTIMIZE TABLE bench.ticks FINAL")
    optimize_s = time.perf_counter() - start
    _, bytes_final, parts_final = conn.query(parts_sql, "TSV").bytes().decode().strip().split("\t")
    conn.close()
    return {
        "rows": int(rows),
        "mergetree_mb": round(int(bytes_on_disk) / 1e6, 1),
        "parts": int(parts),
        "mergetree_mb_after_optimize": round(int(bytes_final) / 1e6, 1),
        "parts_after_optimize": int(parts_final),
        "optimize_s": round(optimize_s, 3),
        "parquet_mb": common.file_size_mb(TICKS),
    }


def bars_mergetree():
    conn = common.session(path=MT_DIR)
    before = common.engine_counters(conn)
    df = conn.query(BARS_SQL.format(source="bench.ticks WHERE symbol = 'BTCUSDT'"), "DataFrame")
    after = common.engine_counters(conn)
    conn.close()
    out = {"bars": len(df), "checksum": round(float(df["volume"].sum()), 3)}
    out.update(common.counter_delta(before, after))
    return out


# --------------------------------------------------------------------------
# ASOF JOIN trades -> 1s klines (one day)
# --------------------------------------------------------------------------


def asof_sql():
    conn = common.session()
    text = (
        conn.query(
            f"""
        SELECT count(), avg(t.price - k.close), max(t.ts - k.ts)
        FROM (SELECT ts, price, 'BTCUSDT' AS symbol FROM file('{DAY_BTC}', Parquet)) AS t
        ASOF LEFT JOIN (SELECT ts, close, 'BTCUSDT' AS symbol FROM file('{DAY_KLINES}', Parquet)) AS k
        ON t.symbol = k.symbol AND t.ts >= k.ts
        """,
            "TSV",
        )
        .bytes()
        .decode()
        .strip()
    )
    conn.close()
    count, diff, _ = text.split("\t")
    return {"rows": int(count), "checksum": round(float(diff), 6)}


def asof_pandas():
    import pandas as pd
    import pyarrow.parquet as pq

    trades = pq.read_table(DAY_BTC, columns=["ts", "price"]).to_pandas().sort_values("ts")
    klines = pq.read_table(DAY_KLINES, columns=["ts", "close"]).to_pandas().sort_values("ts")
    merged = pd.merge_asof(trades, klines, on="ts", direction="backward")
    diff = (merged["price"] - merged["close"]).mean()
    return {"rows": len(merged), "checksum": round(float(diff), 6)}


# --------------------------------------------------------------------------
# rolling window over one day of trades
# --------------------------------------------------------------------------


def rolling_sql(window=1000):
    conn = common.session()
    text = (
        conn.query(
            f"""
        SELECT count(), avg(ma), max(abs(price - ma))
        FROM (
            SELECT price,
                   avg(price) OVER (ORDER BY ts, agg_trade_id ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS ma
            FROM file('{DAY_BTC}', Parquet)
        )
        """,
            "TSV",
        )
        .bytes()
        .decode()
        .strip()
    )
    conn.close()
    count, avg_ma, _ = text.split("\t")
    return {"rows": int(count), "checksum": round(float(avg_ma), 4)}


def rolling_pandas(window=1000):
    import pyarrow.parquet as pq

    df = pq.read_table(DAY_BTC, columns=["ts", "agg_trade_id", "price"]).to_pandas()
    df = df.sort_values(["ts", "agg_trade_id"])
    ma = df["price"].rolling(window, min_periods=1).mean()
    return {"rows": len(df), "checksum": round(float(ma.mean()), 4)}


def rolling_bars_sql():
    """20/60-bar moving averages and 1-minute returns over the 31-day bar series."""
    conn = common.session()
    df = conn.query(
        f"""
        WITH bars AS ({BARS_SQL.format(source=f"file('{BTC_GLOB}', Parquet)")})
        SELECT
            minute, close,
            avg(close) OVER (ORDER BY minute ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma20,
            avg(close) OVER (ORDER BY minute ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS ma60,
            close / lagInFrame(close, 1) OVER (ORDER BY minute) - 1 AS ret
        FROM bars
        """,
        "DataFrame",
    )
    conn.close()
    return {"bars": len(df), "checksum": round(float(df["ma20"].mean()), 4)}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    suite = "bench_quant"
    cases = []
    run = lambda case, **args: cases.append(common.run_case(suite, case, args))  # noqa: E731

    print("== 1-minute bars, 31 days BTCUSDT")
    run("bars_sql")
    run("bars_stream_sql")
    run("bars_pandas")
    run("bars_pandas_per_day")

    print("== one day out of 31 (partition pruning)")
    for mode in (
        "all_days",
        "unprunable_filter",
        "explicit_path",
        "hive_filter",
        "ts_filter",
        "ts_filter_no_pushdown",
        "full_scan",
    ):
        run("one_day", mode=mode)

    print("== MergeTree")
    cases.append(common.run_case(suite, "mergetree_load", {}, repeat=1))
    run("bars_mergetree")
    for mode in ("parquet", "parquet_no_hive", "mergetree"):
        run("one_hour", mode=mode)

    print("== ASOF JOIN, one day")
    run("asof_sql")
    run("asof_pandas")

    print("== rolling windows")
    for window in (100, 1000):
        run("rolling_sql", window=window)
        run("rolling_pandas", window=window)
    run("rolling_bars_sql")

    common.save(
        suite,
        cases,
        {
            "datasets": {
                "ticks_mb": common.file_size_mb(TICKS),
                "btc_mb": common.file_size_mb(BTC_GLOB),
                "day_btc_mb": common.file_size_mb(DAY_BTC),
                "klines_mb": common.file_size_mb(KLINES),
            }
        },
    )


if __name__ == "__main__":
    sys.exit(main())
