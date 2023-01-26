#!/usr/bin/env python3
import concurrent.futures
import time
import sys
import _chdb as chdb

# run query parallel in n thread and benchmark
thread_count = 1
query_count = 1000

if len(sys.argv) == 2:
    thread_count = int(sys.argv[1])
elif len(sys.argv) == 3:
    thread_count = int(sys.argv[1])
    query_count = int(sys.argv[2])

thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)

def run_query(query, format):
    res = chdb.query(query, format)
    print(len(res.get_memview().tobytes()))

def run_queries(query, format, count = query_count):
    for i in range(count):
        run_query(query, format)

def run_queries_parallel(query, format, parallel = thread_count, count = query_count):
    for i in range(parallel):
        thread_pool.submit(run_queries, query, format, count // parallel)

def wait():
    thread_pool.shutdown(wait=True)

def benchmark(query, format, parallel = thread_count, count = query_count):
    time_start = time.time()
    run_queries_parallel(query, format, parallel, count)
    wait()
    time_end = time.time()
    print("Time cost:", time_end - time_start, "s")

if __name__ == '__main__':
    benchmark("SELECT * FROM file('/home/Clickhouse/bench/result.parquet', Parquet) LIMIT 10", "Arrow")
