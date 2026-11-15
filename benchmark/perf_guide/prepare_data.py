#!/usr/bin/env python3
"""Download and lay out the datasets used by the performance-guide benchmarks.

Public sources, no credentials:

- Binance spot aggTrades for BTCUSDT and ETHUSDT, one file per day, written as
  Hive-partitioned Parquet: ticks/day=YYYY-MM-DD/symbol=SYM/trades.parquet
- Binance 1-second klines for BTCUSDT: klines/day=YYYY-MM-DD/klines.parquet
- NYC TLC yellow taxi trips, one Parquet file per month: taxi/*.parquet, plus a
  CSV copy of January for the CSV-versus-Parquet case.

All conversion is done by chdb itself (INSERT INTO FUNCTION file(...)).

    python -m benchmark.perf_guide.prepare_data --days 31 --months 12
"""

import argparse
import concurrent.futures
import datetime as dt
import os
import subprocess
import sys
import zipfile

from benchmark.perf_guide import common

BINANCE = "https://data.binance.vision/data/spot/daily"
TAXI = "https://d37ci6vzurychx.cloudfront.net/trip-data"

TRADES_SCHEMA = (
    "agg_trade_id Int64, price Float64, qty Float64, first_trade_id Int64, "
    "last_trade_id Int64, ts_ms Int64, is_buyer_maker Bool, is_best_match Bool"
)
KLINES_SCHEMA = (
    "open_time Int64, open Float64, high Float64, low Float64, close Float64, "
    "volume Float64, close_time Int64, quote_volume Float64, trades Int64, "
    "taker_buy_volume Float64, taker_buy_quote_volume Float64, ignore String"
)


def download(url: str, dest: str) -> str:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", tmp, url], check=True)
    os.replace(tmp, dest)
    return dest


def unzip_csv(zip_path: str) -> str:
    csv_path = zip_path[:-4] + ".csv"
    if os.path.exists(csv_path):
        return csv_path
    with zipfile.ZipFile(zip_path) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as src, open(csv_path + ".part", "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
    os.replace(csv_path + ".part", csv_path)
    return csv_path


def csv_has_header(path: str) -> bool:
    with open(path, "rb") as handle:
        first = handle.read(1)
    return not first.isdigit()


def convert_trades(conn, csv_path: str, day: str, symbol: str) -> str:
    out_dir = os.path.join(common.TICKS_DIR, f"day={day}", f"symbol={symbol}")
    out = os.path.join(out_dir, "trades.parquet")
    if os.path.exists(out):
        return out
    os.makedirs(out_dir, exist_ok=True)
    fmt = "CSVWithNames" if csv_has_header(csv_path) else "CSV"
    conn.query(
        f"""
        INSERT INTO FUNCTION file('{out}', 'Parquet')
        SELECT
            agg_trade_id,
            price,
            qty,
            fromUnixTimestamp64Milli(ts_ms) AS ts,
            is_buyer_maker
        FROM file('{csv_path}', '{fmt}', '{TRADES_SCHEMA}')
        ORDER BY ts, agg_trade_id
        SETTINGS output_format_parquet_compression_method = 'zstd'
        """
    )
    return out


def convert_klines(conn, csv_path: str, day: str) -> str:
    out_dir = os.path.join(common.DATA, "klines", f"day={day}")
    out = os.path.join(out_dir, "klines.parquet")
    if os.path.exists(out):
        return out
    os.makedirs(out_dir, exist_ok=True)
    fmt = "CSVWithNames" if csv_has_header(csv_path) else "CSV"
    conn.query(
        f"""
        INSERT INTO FUNCTION file('{out}', 'Parquet')
        SELECT
            fromUnixTimestamp64Milli(open_time) AS ts,
            open, high, low, close, volume, trades
        FROM file('{csv_path}', '{fmt}', '{KLINES_SCHEMA}')
        ORDER BY ts
        SETTINGS output_format_parquet_compression_method = 'zstd'
        """
    )
    return out


def binance_jobs(days: int, symbols):
    start = dt.date(2024, 1, 1)
    jobs = []
    for offset in range(days):
        day = (start + dt.timedelta(days=offset)).isoformat()
        for symbol in symbols:
            jobs.append(("trades", symbol, day))
        jobs.append(("klines", "BTCUSDT", day))
    return jobs


def fetch_binance(job):
    kind, symbol, day = job
    raw = os.path.join(common.DATA, "raw")
    if kind == "trades":
        url = f"{BINANCE}/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"
        dest = os.path.join(raw, "aggTrades", f"{symbol}-aggTrades-{day}.zip")
    else:
        url = f"{BINANCE}/klines/{symbol}/1s/{symbol}-1s-{day}.zip"
        dest = os.path.join(raw, "klines", f"{symbol}-1s-{day}.zip")
    return job, unzip_csv(download(url, dest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=31, help="days of Binance data from 2024-01-01")
    parser.add_argument("--months", type=int, default=12, help="months of 2024 taxi data")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--keep-csv", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    os.makedirs(common.DATA, exist_ok=True)
    conn = common.session()

    jobs = binance_jobs(args.days, symbols)
    print(f"binance: {len(jobs)} files", flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        for job, csv_path in pool.map(fetch_binance, jobs):
            kind, symbol, day = job
            if kind == "trades":
                convert_trades(conn, csv_path, day, symbol)
            else:
                convert_klines(conn, csv_path, day)
            if not args.keep_csv:
                os.remove(csv_path)
            print(f"  {kind} {symbol} {day}", flush=True)

    taxi_dir = os.path.join(common.DATA, "taxi")
    os.makedirs(taxi_dir, exist_ok=True)
    for month in range(1, args.months + 1):
        name = f"yellow_tripdata_2024-{month:02d}.parquet"
        download(f"{TAXI}/{name}", os.path.join(taxi_dir, name))
        print(f"  taxi {name}", flush=True)

    if not os.path.exists(common.TAXI_MONTH_CSV):
        conn.query(
            f"""
            INSERT INTO FUNCTION file('{common.TAXI_MONTH_CSV}', 'CSVWithNames')
            SELECT * FROM file('{common.TAXI_MONTH}', Parquet)
            """
        )
        print("  taxi january CSV", flush=True)

    print(
        "ticks rows:",
        conn.query(
            f"SELECT symbol, count() FROM file('{common.TICKS_HIVE}', Parquet) GROUP BY symbol ORDER BY symbol",
            "TSV",
        )
        .bytes()
        .decode()
        .strip()
        .replace("\n", " | "),
    )
    print(
        "klines rows:",
        conn.query(f"SELECT count() FROM file('{common.KLINES_HIVE}', Parquet)", "CSV").bytes().decode().strip(),
    )
    print(
        "taxi rows:",
        conn.query(f"SELECT count() FROM file('{common.TAXI_GLOB}', Parquet)", "CSV").bytes().decode().strip(),
    )
    print(
        "sizes MB: ticks",
        common.file_size_mb(common.TICKS_HIVE),
        "klines",
        common.file_size_mb(common.KLINES_HIVE),
        "taxi",
        common.file_size_mb(common.TAXI_GLOB),
    )
    conn.close()


if __name__ == "__main__":
    sys.exit(main())
