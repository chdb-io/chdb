#!/usr/bin/env python3
"""
chdb vs pandas — Peak memory benchmark (VmHWM / ru_maxrss, subprocess isolation)

Generates test data automatically and runs 10 benchmark scenarios comparing
chdb SQL-pushdown vs pandas in-memory processing on time and peak memory.

Usage:
    python chdb_vs_pandas_memory.py              # default 1M rows
    python chdb_vs_pandas_memory.py --rows 10M   # 10 million rows
    python chdb_vs_pandas_memory.py --rows 50M -o results.json
"""
import os, sys, json, time, math, subprocess, tempfile, platform
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="chdb vs pandas peak-memory benchmark")
parser.add_argument("--rows", default="10M",
                    help="Number of rows to generate, e.g. 1M, 10M, 50M (default: 10M)")
parser.add_argument("--python", default=sys.executable,
                    help="Python interpreter for subprocess (default: current)")
parser.add_argument("-o", "--output", default=None,
                    help="Output JSON path (default: bench_results_<rows>.json)")
parser.add_argument("--data-dir", default=None,
                    help="Directory for generated Parquet files (default: tempdir)")
args = parser.parse_args()


def parse_row_count(s: str) -> int:
    s = s.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


NROWS = parse_row_count(args.rows)
VENV_PYTHON = args.python
OUTPUT_JSON = args.output or f"bench_results_{args.rows.lower()}.json"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(data_dir: str, nrows: int):
    """Generate fact + dimension Parquet files for benchmarking."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)

    pq_path = os.path.join(data_dir, "bench_data.parquet")
    dim_path = os.path.join(data_dir, "bench_dim.parquet")

    if os.path.exists(pq_path) and os.path.exists(dim_path):
        existing = pd.read_parquet(pq_path, columns=["id"])
        if len(existing) == nrows:
            print(f"  Reusing existing data ({nrows:,} rows)")
            return pq_path, dim_path

    print(f"  Generating {nrows:,} rows ... ", end="", flush=True)
    t0 = time.perf_counter()

    regions = [f"R{i:03d}" for i in range(100)]
    categories = [f"C{i:04d}" for i in range(10_000)]

    df = pd.DataFrame({
        "id": np.arange(nrows),
        "region": rng.choice(regions, nrows),
        "category": rng.choice(categories, nrows),
        "user_id": rng.integers(1, nrows // 10 + 1, nrows),
        "amount": rng.uniform(1, 1000, nrows).round(2),
        "price": rng.uniform(1, 500, nrows).round(2),
        "quantity": rng.integers(1, 100, nrows),
        "flag": rng.integers(0, 2, nrows),
        "ts": rng.integers(1_577_836_800, 1_704_067_200, nrows),  # 2020-2024
    })
    df.to_parquet(pq_path, index=False)

    dim = pd.DataFrame({
        "category": categories,
        "cat_weight": rng.uniform(0.1, 5.0, len(categories)).round(3),
    })
    dim.to_parquet(dim_path, index=False)

    elapsed = time.perf_counter() - t0
    pq_mb = os.path.getsize(pq_path) / 1e6
    print(f"done ({elapsed:.1f}s, {pq_mb:.0f} MB)")
    return pq_path, dim_path


# ---------------------------------------------------------------------------
# Subprocess wrapper — measures VmHWM (Linux) or ru_maxrss (macOS)
# ---------------------------------------------------------------------------
WRAPPER = '''\
import gc, time, json, sys, platform, resource

def peak_mem_kb():
    """Return peak RSS in KB. Linux: VmHWM from /proc. macOS: ru_maxrss (bytes->KB)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except FileNotFoundError:
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        rss //= 1024  # macOS returns bytes, convert to KB
    return rss

gc.collect()
baseline = peak_mem_kb()
t0 = time.perf_counter()
error = None
try:
    exec(open(CODE_FILE).read())
except Exception as e:
    import traceback; traceback.print_exc()
    error = str(e)
elapsed = time.perf_counter() - t0
gc.collect()
peak = peak_mem_kb()
r = {"elapsed_s": round(elapsed, 3),
     "baseline_mb": round(baseline/1024, 1),
     "peak_mb": round(peak/1024, 1),
     "delta_mb": round((peak - baseline)/1024, 1),
     "error": error}
print("__R__" + json.dumps(r))
'''


def run_sub(code, timeout=600):
    """Write code to temp file and execute in isolated subprocess."""
    tmp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=tmp_dir) as cf:
        cf.write(code)
        code_path = cf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=tmp_dir) as wf:
        wf.write(f'CODE_FILE = "{code_path}"\n')
        wf.write(WRAPPER)
        wrapper_path = wf.name
    try:
        proc = subprocess.run([VENV_PYTHON, wrapper_path],
                              capture_output=True, text=True, timeout=timeout)
        out = proc.stdout + proc.stderr
        for line in out.split("\n"):
            if "__R__" in line:
                return json.loads(line.split("__R__", 1)[1])
        return {"elapsed_s": 0, "baseline_mb": 0, "peak_mb": 0, "delta_mb": 0,
                "error": f"exit={proc.returncode} | {out[-300:]}"}
    except subprocess.TimeoutExpired:
        return {"elapsed_s": timeout, "baseline_mb": 0, "peak_mb": 0, "delta_mb": 0,
                "error": "TIMEOUT"}
    finally:
        os.unlink(code_path)
        os.unlink(wrapper_path)


# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
print("=" * 90)
print("chdb vs pandas — Peak memory benchmark")
print("=" * 90)

if args.data_dir:
    DATA_DIR = os.path.abspath(args.data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    _cleanup_data = False
else:
    DATA_DIR = tempfile.mkdtemp(prefix="chdb_bench_")
    _cleanup_data = True

PQ, DIM = generate_data(DATA_DIR, NROWS)

pq_size_mb = os.path.getsize(PQ) / 1e6
try:
    total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    ram_info = f" | RAM {total_ram:.0f} GB"
except (ValueError, OSError, AttributeError):
    ram_info = ""
print(f"  Data: {NROWS:,} rows | Parquet {pq_size_mb:.0f} MB{ram_info}")
print(f"  Platform: {platform.system()} {platform.machine()}")
print()


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
TESTS = []

def T(name, desc, pd_code, ch_code):
    TESTS.append((name, desc, pd_code, ch_code))

T("T01_read_count", "Read Parquet + count",
f"""
import pandas as pd
df = pd.read_parquet("{PQ}")
n = len(df)
del df
""",
f"""
import chdb
r = chdb.query("SELECT count() FROM file('{PQ}', Parquet)", "CSV")
n = int(r.bytes().decode().strip())
""")

T("T02_filter_groupby_sum", "Filter + GroupBy + sum",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}")
r = df[df["flag"] == 1].groupby("region")["amount"].sum().reset_index()
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT region, sum(amount) as total_amount
    FROM file('{PQ}', Parquet)
    WHERE flag = 1
    GROUP BY region ORDER BY region
''', "DataFrame")
""")

T("T03_multi_groupby", "Multi-key GroupBy + 4 aggs",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}")
r = df.groupby(["region", "category"]).agg(
    total_amount=("amount", "sum"),
    avg_price=("price", "mean"),
    max_qty=("quantity", "max"),
    row_count=("id", "count"),
).reset_index()
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT region, category,
           sum(amount) as total_amount, avg(price) as avg_price,
           max(quantity) as max_qty, count(id) as row_count
    FROM file('{PQ}', Parquet)
    GROUP BY region, category
''', "DataFrame")
""")

T("T04_join_aggregate", "Two-table JOIN + weighted agg",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}")
dim = pd.read_parquet("{DIM}")
r = df.merge(dim, on="category", how="inner")
r["weighted_amount"] = r["amount"] * r["cat_weight"]
result = r.groupby("region")["weighted_amount"].sum().reset_index()
del df, dim, r; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT f.region, sum(f.amount * d.cat_weight) as weighted_amount
    FROM file('{PQ}', Parquet) AS f
    INNER JOIN file('{DIM}', Parquet) AS d ON f.category = d.category
    GROUP BY f.region ORDER BY f.region
''', "DataFrame")
""")

T("T05_topn_per_group", "Top-10 per region (window func)",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}", columns=["region", "user_id", "amount"])
r = (df.sort_values(["region", "amount"], ascending=[True, False])
       .groupby("region").head(10).reset_index(drop=True))
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT region, user_id, amount FROM (
        SELECT region, user_id, amount,
               ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) as rn
        FROM file('{PQ}', Parquet)
    ) WHERE rn <= 10
    ORDER BY region, amount DESC
''', "DataFrame")
""")

T("T06_quantile_p95", "P95 quantile by region",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}", columns=["region", "amount"])
r = df.groupby("region")["amount"].quantile(0.95).reset_index()
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT region, quantile(0.95)(amount) as p95_amount
    FROM file('{PQ}', Parquet)
    GROUP BY region ORDER BY region
''', "DataFrame")
""")

T("T07_derived_filter_sort", "Derived cols + filter + Top-1000",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}", columns=["id", "amount", "price", "quantity", "region"])
df["revenue"] = df["price"] * df["quantity"]
df["margin"] = df["revenue"] - df["amount"]
r = df[df["margin"] > 500].nlargest(1000, "margin")[["id", "region", "revenue", "margin"]]
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT id, region, price * quantity as revenue, price * quantity - amount as margin
    FROM file('{PQ}', Parquet)
    WHERE price * quantity - amount > 500
    ORDER BY margin DESC LIMIT 1000
''', "DataFrame")
""")

T("T08_count_distinct", "Count Distinct user_id by region",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}", columns=["region", "user_id"])
r = df.groupby("region")["user_id"].nunique().reset_index()
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT region, uniqExact(user_id) as unique_users
    FROM file('{PQ}', Parquet)
    GROUP BY region ORDER BY region
''', "DataFrame")
""")

T("T09_inmemory_ops", "In-memory DataFrame -> chdb SQL",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}")
r = (df.query("flag == 1 and amount > 50")
       .groupby("region")
       .agg(total=("amount", "sum"), cnt=("id", "count"))
       .reset_index())
del df; gc.collect()
""",
f"""
import pandas as pd, chdb, gc
df = pd.read_parquet("{PQ}")
import chdb.session as chs
sess = chs.Session()
r = sess.query('''
    SELECT region, sum(amount) as total, count(id) as cnt
    FROM Python(df)
    WHERE flag = 1 AND amount > 50
    GROUP BY region ORDER BY region
''', "DataFrame")
del df; gc.collect()
""")

T("T10_timeseries_agg", "Time-series monthly agg",
f"""
import pandas as pd, gc
df = pd.read_parquet("{PQ}", columns=["ts", "amount"])
df["date"] = pd.to_datetime(df["ts"], unit="s")
df["month"] = df["date"].dt.to_period("M")
r = df.groupby("month")["amount"].agg(["sum", "mean", "count"]).reset_index()
del df; gc.collect()
""",
f"""
import chdb
r = chdb.query('''
    SELECT toStartOfMonth(toDateTime(ts)) as month,
           sum(amount) as sum, avg(amount) as mean, count() as count
    FROM file('{PQ}', Parquet)
    GROUP BY month ORDER BY month
''', "DataFrame")
""")


# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
results = []
for name, desc, pd_code, ch_code in TESTS:
    print(f"{'─'*80}")
    print(f"{name}: {desc}")

    sys.stdout.write("  pandas ... "); sys.stdout.flush()
    pr = run_sub(pd_code)
    if pr["error"]:
        print(f"FAIL: {pr['error'][:70]}")
    else:
        print(f"{pr['elapsed_s']:>6.2f}s | peak {pr['peak_mb']:>7.0f} MB "
              f"(delta {pr['delta_mb']:>+7.0f} MB)")

    sys.stdout.write("  chdb   ... "); sys.stdout.flush()
    cr = run_sub(ch_code)
    if cr["error"]:
        print(f"FAIL: {cr['error'][:70]}")
    else:
        print(f"{cr['elapsed_s']:>6.2f}s | peak {cr['peak_mb']:>7.0f} MB "
              f"(delta {cr['delta_mb']:>+7.0f} MB)")

    results.append({
        "test": name, "desc": desc,
        "pd_time": pr["elapsed_s"], "ch_time": cr["elapsed_s"],
        "pd_peak_mb": pr["peak_mb"], "ch_peak_mb": cr["peak_mb"],
        "pd_delta_mb": pr["delta_mb"], "ch_delta_mb": cr["delta_mb"],
        "pd_err": pr["error"], "ch_err": cr["error"],
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
mem_source = "VmHWM" if platform.system() == "Linux" else "ru_maxrss"
print(f"\n\n{'='*115}")
print(f"Summary — Peak memory ({mem_source}), {NROWS:,} rows")
print(f"{'='*115}")
hdr = (f"{'Test':<26} {'Scenario':<24} {'pd time':>9} {'ch time':>9} {'Speedup':>7} "
       f"{'pd peak':>9} {'ch peak':>9} {'Mem save':>8} {'Status':>6}")
print(hdr)
print("─" * 115)

ok = []
for r in results:
    pt, ct = r["pd_time"], r["ch_time"]
    pp, cp = r["pd_peak_mb"], r["ch_peak_mb"]
    err = []
    if r["pd_err"]: err.append("PD")
    if r["ch_err"]: err.append("CH")
    status = "+".join(err) if err else "OK"

    if not err and ct > 0.001:
        spd = f"{pt/ct:.1f}x"
        mem = f"{pp/cp:.2f}x" if cp > 0 else "-"
        ok.append(r)
    else:
        spd, mem = "-", "-"

    print(f"{r['test']:<26} {r['desc'][:22]:<24} {pt:>8.2f}s {ct:>8.2f}s {spd:>7} "
          f"{pp:>8.0f} {cp:>8.0f} {mem:>8} {status:>6}")

print("─" * 115)

if ok:
    geo_spd = math.exp(sum(math.log(r["pd_time"]/r["ch_time"]) for r in ok) / len(ok))
    geo_mem = math.exp(sum(math.log(r["pd_peak_mb"]/r["ch_peak_mb"])
                           for r in ok if r["ch_peak_mb"] > 0) / len(ok))
    max_spd = max(r["pd_time"]/r["ch_time"] for r in ok)
    min_spd = min(r["pd_time"]/r["ch_time"] for r in ok)
    print(f"\nPassed: {len(ok)}/{len(results)} | "
          f"Speed geo-mean: {geo_spd:.1f}x (range {min_spd:.1f}x ~ {max_spd:.1f}x) | "
          f"Peak memory geo-mean: pandas is {geo_mem:.1f}x of chdb")

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to: {os.path.abspath(OUTPUT_JSON)}")

# Cleanup temp data if we created it
if _cleanup_data:
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    print(f"Cleaned up temp data: {DATA_DIR}")
