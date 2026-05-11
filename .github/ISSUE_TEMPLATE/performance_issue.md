---
name: Performance issue
about: Report a query, operation, or workload that's slower than expected
labels: performance
---

## What's slow

<!-- The query, DataStore operation, or workload that's underperforming.
     Include enough context to tell whether the bottleneck is in chdb-core
     (engine), chdb-ds (DataStore), or the pandas interop layer. -->

## Dataset characteristics

<!-- Shape and source matter a lot for engine-side perf reports. -->

- Row count: <!-- e.g. 10M rows -->
- Column count / types: <!-- e.g. 20 columns, mix of String/Int64/Float64 -->
- Source format: <!-- Parquet / CSV / pandas DataFrame / Arrow Table / etc. -->
- File size on disk (if applicable): <!-- e.g. 1.2 GB Parquet -->

## How to reproduce

<!-- Smallest script that shows the slowness. If the data is public,
     point at it; if not, describe the schema + how to synthesise it. -->

```python
import chdb
import time

# Setup (data generation / loading)
# ...

start = time.perf_counter()
result = chdb.query("SELECT ...")   # or DataStore equivalent
elapsed = time.perf_counter() - start
print(f"elapsed: {elapsed:.3f}s")
```

## Expected vs observed

- Observed: <!-- e.g. 12.4 s on 10M rows, single query -->
- Expected: <!-- what you compared against — DuckDB, Polars, prior chdb
    version, an order-of-magnitude estimate, etc. -->

## Environment

- chDB version: <!-- python -c "import chdb; print(chdb.__version__)" -->
- chdb-core version: <!-- python -c "import chdb; print(getattr(chdb, 'core_version', 'unknown'))" -->
- Python version: <!-- python --version -->
- OS / architecture: <!-- e.g. macOS arm64 M2, 16 GB RAM -->
- Hardware notes (if relevant): <!-- CPU model, RAM, disk type (NVMe / spinning) -->

## Profiling output (optional but very helpful)

<!-- If you can capture any of:
     - `EXPLAIN` / `EXPLAIN PIPELINE` for SQL queries
     - py-spy / cProfile output for DataStore code paths
     - chdb settings used (e.g. via `chdb.query(..., 'PrettyJSONEachRow')`)
     paste them here. -->

```
<paste here>
```

## Additional context

<!-- Anything else: workaround you found, comparable timing on the same
     box with another tool, related issues. -->
