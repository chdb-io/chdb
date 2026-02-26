import pandas as pd
import duckdb
import polars as pl
import chdb
import time

# Generate more data rows
n = 100_000_000
df = pd.DataFrame({"a": range(n), "b": ["c", "d"] * (n // 2)})

query = "SELECT a + 2 AS c FROM df WHERE b = 'd'"
chdb_query = "SELECT a + 2 AS c FROM Python(df) WHERE b = 'd'"

# DuckDB
t0 = time.time()
duck_res = duckdb.sql(query).df()
print(f"DuckDB: {time.time() - t0:.3f}s, rows: {len(duck_res)}")

# Polars
t0 = time.time()
pl_res = pl.sql(query, eager=True).to_pandas()
print(f"Polars: {time.time() - t0:.3f}s, rows: {len(pl_res)}")

# chDB
t0 = time.time()
chdb_res = chdb.query(chdb_query, 'DataFrame')
print(f"chDB:   {time.time() - t0:.3f}s, rows: {len(chdb_res)}")

print(f"\nDuckDB dtype: {duck_res['c'].dtype}, head: {duck_res['c'].head(3).tolist()}")
print(f"Polars dtype: {pl_res['c'].dtype}, head: {pl_res['c'].head(3).tolist()}")
print(f"chDB   dtype: {chdb_res['c'].dtype}, head: {chdb_res['c'].head(3).tolist()}")

assert duck_res.equals(pl_res)
print("\nDuckDB == Polars ✓")

# chDB may not preserve order, sort and compare
duck_sorted = duck_res.sort_values('c').reset_index(drop=True)
chdb_sorted = chdb_res.sort_values('c').reset_index(drop=True)
assert duck_sorted.equals(chdb_sorted)
print("DuckDB == chDB ✓ (sorted)")
