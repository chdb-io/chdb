#!/usr/bin/env python3

import os
import shutil
import unittest
import pyarrow as pa
import chdb
from chdb import session
from deltalake import write_deltalake


class TestArrowRecordReaderDeltaLake(unittest.TestCase):
    def setUp(self) -> None:
        # Create test directory in current directory
        self.test_dir = os.path.join(os.getcwd(), "chdb_arrow_test_temp")
        self.delta_table_path = os.path.join(self.test_dir, "test_delta_table")

        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)

        self.sess = session.Session()
        return super().setUp()

    def tearDown(self) -> None:
        self.sess.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        return super().tearDown()

    def test_arrow_record_reader(self):
        """Test arrow_record_reader basic functionality"""

        stream_result = self.sess.send_query(
            "SELECT number as id, toString(number) as name FROM numbers(500000) WHERE id < 0",
            "arrow",
        )
        empty_batch_reader = stream_result.record_batch()

        self.assertIsInstance(empty_batch_reader, pa.RecordBatchReader)

        batches = list(empty_batch_reader)
        total_rows = sum(batch.num_rows for batch in batches)

        self.assertEqual(total_rows, 0)
        self.assertEqual(len(batches), 0)

        stream_result = self.sess.send_query(
            "SELECT number as id, toString(number) as name FROM numbers(500000)",
            "ArrowTable"
        )
        with self.assertRaises(ValueError) as context:
            batch_reader = stream_result.record_batch()
        stream_result.close()

        with self.sess.send_query(
            "SELECT number as id, toString(number) as name FROM numbers(500000)",
            "JSON"
        ) as stream_result:
            with self.assertRaises(ValueError) as context:
                batch_reader = stream_result.record_batch()

        stream_result = self.sess.send_query(
            "SELECT number as id, toString(number) as name FROM numbers(500000)",
            "arrow"
        )
        batch_reader = stream_result.record_batch(100000)

        self.assertIsInstance(batch_reader, pa.RecordBatchReader)

        batches = list(batch_reader)
        total_rows = sum(batch.num_rows for batch in batches)

        self.assertEqual(total_rows, 500000)
        # print(len(batches))

        # Verify schema
        schema = batches[0].schema
        self.assertEqual(len(schema), 2)
        self.assertEqual(schema[0].name, 'id')
        self.assertEqual(schema[1].name, 'name')

    def test_deltalake_integration(self):
        """Test writing RecordBatchReader to Delta Lake and reading back"""

        delta_table_path = self.delta_table_path
        write_query = f"""
            SELECT
                number as id,
                toString(number) as name,
                number * 1.5 as value,
                if(number % 2 = 0, 'even', 'odd') as category
            FROM numbers(500000)
        """

        # Create test data with RecordBatchReader
        stream_result = self.sess.send_query(write_query, "arrow")
        batch_reader = stream_result.record_batch(rows_per_batch=100000)
        stream_result.close()

        with self.sess.send_query(write_query, "arrow") as stream_result:
            batch_reader = stream_result.record_batch(rows_per_batch=100000)

        stream_result = self.sess.send_query(write_query, "arrow")

        # Get RecordBatchReader with smaller batch size for testing
        batch_reader = stream_result.record_batch(rows_per_batch=100000)

        # Write to Delta Lake
        write_deltalake(
            table_or_uri=delta_table_path,
            data=batch_reader,
            mode="overwrite"
        )

        # Verify files were created
        self.assertTrue(os.path.exists(delta_table_path))
        self.assertTrue(any(f.endswith('.parquet') for f in os.listdir(delta_table_path)))

        # Read back using chdb to verify data integrity
        read_query = f"""
            SELECT
                id, name, value, category,
                COUNT(*) as total_count
            FROM file('{delta_table_path}/*.parquet', 'Parquet')
            GROUP BY id, name, value, category
            ORDER BY id
        """

        read_table = self.sess.query(read_query, "ArrowTable")

        # Verify data integrity
        self.assertEqual(len(read_table), 500000, "Should have 1000 records")

        # Check first few records
        first_row = read_table.slice(0, 1).to_pydict()
        self.assertEqual(first_row['id'][0], 0)
        self.assertEqual(first_row['name'][0], '0')
        self.assertEqual(first_row['value'][0], 0.0)
        self.assertEqual(first_row['category'][0], 'even')
        self.assertEqual(first_row['total_count'][0], 1)

        # Check last record
        last_row = read_table.slice(499999, 1).to_pydict()
        self.assertEqual(last_row['id'][0], 499999)
        self.assertEqual(last_row['name'][0], '499999')
        self.assertEqual(last_row['value'][0], 749998.5)
        self.assertEqual(last_row['category'][0], 'odd')

        # Verify schema consistency
        expected_columns = {'id', 'name', 'value', 'category', 'total_count'}
        actual_columns = set(read_table.column_names)
        self.assertEqual(actual_columns, expected_columns)


if __name__ == "__main__":
    unittest.main()
