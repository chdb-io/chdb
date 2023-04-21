#!python3
import concurrent.futures
import time
import sys
import os
import chdb
import unittest

# run query parallel in n thread and benchmark
thread_count = 10
query_count = 1000
current_dir=os.path.dirname(os.path.abspath(__file__))
data_file=os.path.join(current_dir, "../contrib/arrow/cpp/submodules/parquet-testing/data/alltypes_dictionary.parquet")

if len(sys.argv) == 2:
    thread_count = int(sys.argv[1])
elif len(sys.argv) == 3:
    thread_count = int(sys.argv[1])
    query_count = int(sys.argv[2])

thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)

def run_query(query, format):
    res = chdb.query(query, format)
    assert len(res.get_memview().tobytes()) == 2290

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
    print("QPS:", count / (time_end - time_start))

class TestParallel(unittest.TestCase):
    def test_parallel(self):
        benchmark(f"SELECT * FROM file('{data_file}', Parquet) LIMIT 10", "Arrow")

if __name__ == '__main__':
    unittest.main()
