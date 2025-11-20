#!/usr/bin/env python3

import unittest
import time
import shutil
import pandas as pd
import chdb
from chdb import session


class TestStreamingDataFrame(unittest.TestCase):
    """Test streaming DataFrame generation with large scale data"""

    def setUp(self):
        self.test_dir = ".tmp_test_streaming_dataframe"
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.sess = session.Session(self.test_dir)

    def tearDown(self):
        self.sess.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_streaming_dataframe_1million_rows(self):
        """Test streaming 1M rows as DataFrame with int and string columns"""

        print("Starting 1M row streaming DataFrame test...")
        start_time = time.time()

        # Query to generate 1M rows with int and string columns
        query = """
        SELECT
            number as id,
            concat('row_', toString(number)) as name
        FROM numbers(1000000)
        """

        # Test streaming fetch using streaming_fetch_result
        streaming_result = self.sess.send_query(query, "DataFrame")

        total_rows = 0
        dataframes = []

        try:
            while True:
                # Fetch next batch as DataFrame
                batch_df = streaming_result.fetch()
                if batch_df is None:
                    break

                # Verify it's a pandas DataFrame
                self.assertIsInstance(batch_df, pd.DataFrame)

                # Verify column structure
                expected_columns = ['id', 'name']
                self.assertEqual(list(batch_df.columns), expected_columns)

                # Verify column types
                self.assertTrue(pd.api.types.is_integer_dtype(batch_df['id']))
                self.assertTrue(pd.api.types.is_object_dtype(batch_df['name']))

                # Add to total count
                batch_rows = len(batch_df)
                total_rows += batch_rows
                dataframes.append(batch_df)

                # print(f"Received batch with {batch_rows} rows, total: {total_rows}")

                # Verify first few rows format for each batch
                if len(batch_df) > 0:
                    first_id = batch_df.iloc[0]['id']
                    first_name = batch_df.iloc[0]['name']
                    expected_name = f'row_{first_id}'
                    self.assertEqual(first_name, expected_name)

        finally:
            streaming_result.close()

        elapsed_time = time.time() - start_time
        print(f"Streaming DataFrame test completed in {elapsed_time:.2f} seconds")
        print(f"Total rows received: {total_rows}")
        print(f"Total batches: {len(dataframes)}")

        # Verify we got exactly 1M rows
        self.assertEqual(total_rows, 1000000)

        # Verify sequential IDs when combining all dataframes
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Verify total rows
            self.assertEqual(len(combined_df), 1000000)

            # Verify ID sequence (should be 0 to 999999)
            expected_ids = list(range(1000000))
            actual_ids = sorted(combined_df['id'].tolist())
            self.assertEqual(actual_ids, expected_ids)

            # Verify name format for sample rows
            sample_indices = [0, 100000, 500000, 999999]
            for i in sample_indices:
                if i < len(combined_df):
                    row = combined_df.iloc[i]
                    expected_name = f'row_{row["id"]}'
                    self.assertEqual(row['name'], expected_name)

        print("All validations passed!")

    def test_streaming_dataframe_empty_result(self):
        """Test streaming DataFrame with empty result set"""

        query = "SELECT number as id, toString(number) as value FROM numbers(0)"

        streaming_result = self.sess.send_query(query, "DataFrame")

        try:
            batch_df = streaming_result.fetch()
            # Should return None for empty result
            self.assertIsNone(batch_df)
        finally:
            streaming_result.close()
            pass

    def test_cancel_streaming_query(self):
        stream_result =  self.sess.send_query("SELECT number FROM numbers(10)", "DataFrame")
        stream_result.cancel()

        result = self.sess.query("SELECT number FROM numbers(10)")
        self.assertEqual(result.rows_read(), 10)

        stream_result = self.sess.send_query("SELECT number FROM numbers(10)", "DataFrame")
        chunks = list(stream_result)
        self.assertEqual(len(chunks), 1)

        stream_result = self.sess.send_query("SELECT number FROM numbers(10)", "DataFrame")
        stream_result.cancel()

    def test_large_dataframe(self):
        total_rows = 0
        with self.sess.send_query("SELECT * FROM numbers(1073741824)", "DataFrame") as stream:
            for df in stream:
                total_rows += len(df)
        self.assertEqual(total_rows, 1073741824)


if __name__ == "__main__":
    unittest.main()
