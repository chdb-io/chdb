#!python3
import unittest
import traceback
import concurrent.futures
import uuid
import chdb
import pandas as pd

def worker(worker_id):
    try:
        # Create a DataFrame
        df = pd.DataFrame({
            'id': list(range(10)),
            'value': [i * worker_id for i in range(10)],
            'category': [f'cat_{i % 3}' for i in range(10)]
        })

        # Generate unique variable name
        var_name = f"__test_df_{uuid.uuid4().hex}__"

        # Register DataFrame in global namespace
        globals()[var_name] = df

        try:
            # Execute SQL query using Python() table function
            sql = f"SELECT * FROM Python({var_name}) WHERE value > 3"
            conn = chdb.connect()
            result = conn.query(sql, 'DataFrame')
            conn.close()
            return result
        finally:
            # Clean up global namespace
            if var_name in globals():
                del globals()[var_name]

    except Exception as e:
        print(f"[Worker {worker_id}] EXCEPTION: {e}")
        traceback.print_exc()
        return None

def test_connection_per_thread():
    num_workers = 5
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]

            completed = []
            for future in concurrent.futures.as_completed(futures, timeout=15):
                try:
                    result = future.result()
                    completed.append(result)
                except Exception as e:
                    print(f"Worker raised exception: {e}")
                    return False
        if len(completed) != num_workers:
            print(f"ERROR: Expected {num_workers} results, got {len(completed)}")
            return False

        # Validate each result is a DataFrame with correct data
        for i, result in enumerate(completed):
            if result is None:
                print(f"ERROR: Result {i} is None")
                return False

            if not isinstance(result, pd.DataFrame):
                print(f"ERROR: Result {i} is not a DataFrame, got {type(result)}")
                return False

        return True

    except concurrent.futures.TimeoutError:
        print("TIMEOUT!")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

class TestParallelQuery(unittest.TestCase):
    def test_parallel_query(self):
        self.assertTrue(test_connection_per_thread())


if __name__ == '__main__':
    unittest.main(verbosity=2)