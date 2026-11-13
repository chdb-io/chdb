ClickBench results for chDB are maintained in a separate repository: https://github.com/ClickHouse/ClickBench

`perf_guide/` holds the benchmarks behind `dev-docs/performance/`. Each case runs in its own
subprocess and records wall time, peak RSS, and `system.events` read counters.

```bash
pip install -e . pandas pyarrow psutil torch        # torch only for bench_ml
python -m benchmark.perf_guide.prepare_data         # Binance aggTrades/klines + NYC taxi 2024, ~1.3 GB
BENCH_THREADS=16 python -m benchmark.perf_guide.bench_memory
python -m benchmark.perf_guide.report               # Markdown tables from perf_guide/results/*.json
```

`BENCH_DATA`, `BENCH_RESULTS`, `BENCH_THREADS`, and `BENCH_REPEAT` override the data directory,
output directory, `max_threads`, and repetitions.
