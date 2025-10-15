#!/usr/bin/env python3

import unittest
import tempfile
import os
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import chdb
from chdb import session
from urllib.request import urlretrieve

# Clean up and create session in the test methods instead of globally

class TestChDBArrowTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Download parquet file if it doesn't exist
        cls.parquet_file = "hits_0.parquet"
        if not os.path.exists(cls.parquet_file):
            print(f"Downloading {cls.parquet_file}...")
            url = "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet"
            urlretrieve(url, cls.parquet_file)
            print("Download complete!")

        # Load parquet as PyArrow table
        cls.arrow_table = pq.read_table(cls.parquet_file)
        cls.table_size = cls.arrow_table.nbytes
        cls.num_rows = cls.arrow_table.num_rows
        cls.num_columns = cls.arrow_table.num_columns

        print(f"Loaded Arrow table: {cls.num_rows} rows, {cls.num_columns} columns, {cls.table_size} bytes")

        if os.path.exists(".test_chdb_arrow_table"):
            shutil.rmtree(".test_chdb_arrow_table", ignore_errors=True)
        cls.sess = session.Session(".test_chdb_arrow_table")

    @classmethod
    def tearDownClass(cls):
        # Clean up session directory
        if os.path.exists(".test_chdb_arrow_table"):
            shutil.rmtree(".test_chdb_arrow_table", ignore_errors=True)
        cls.sess.close()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_arrow_table_basic_info(self):
        """Test basic Arrow table information"""
        self.assertEqual(self.table_size, 729898624)
        self.assertEqual(self.num_rows, 1000000)
        self.assertEqual(self.num_columns, 105)

    def test_arrow_table_count(self):
        """Test counting rows in Arrow table"""
        my_arrow_table = self.arrow_table
        result = self.sess.query("SELECT COUNT(*) as row_count FROM Python(my_arrow_table)", "CSV")
        lines = str(result).strip().split('\n')
        count = int(lines[0])
        self.assertEqual(count, self.num_rows, f"Count should match table rows: {self.num_rows}")

    def test_arrow_table_schema(self):
        """Test querying Arrow table schema information"""
        my_arrow_table = self.arrow_table
        result = self.sess.query("DESCRIBE Python(my_arrow_table)", "CSV")
        # print(result)
        self.assertIn('WatchID', str(result))
        self.assertIn('URLHash', str(result))

    def test_arrow_table_limit(self):
        """Test LIMIT queries on Arrow table"""
        my_arrow_table = self.arrow_table
        result = self.sess.query("SELECT * FROM Python(my_arrow_table) LIMIT 5", "CSV")
        lines = str(result).strip().split('\n')
        self.assertEqual(len(lines), 5, "Should have 5 data rows")

    def test_arrow_table_select_columns(self):
        """Test selecting specific columns from Arrow table"""
        my_arrow_table = self.arrow_table
        # Get first few column names from schema
        schema = self.arrow_table.schema
        first_col = schema.field(0).name
        second_col = schema.field(1).name if len(schema) > 1 else first_col

        result = self.sess.query(f"SELECT {first_col}, {second_col} FROM Python(my_arrow_table) LIMIT 3", "CSV")
        lines = str(result).strip().split('\n')
        self.assertEqual(len(lines), 3, "Should have 3 data rows")

    def test_arrow_table_where_clause(self):
        """Test WHERE clause filtering on Arrow table"""
        my_arrow_table = self.arrow_table
        # Find a numeric column for filtering
        numeric_col = None
        for field in self.arrow_table.schema:
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                numeric_col = field.name
                break

        result = self.sess.query(f"SELECT COUNT(*) FROM Python(my_arrow_table) WHERE {numeric_col} > 1", "CSV")
        lines = str(result).strip().split('\n')
        count = int(lines[0])
        self.assertEqual(count, 1000000)

    def test_arrow_table_group_by(self):
        """Test GROUP BY queries on Arrow table"""
        my_arrow_table = self.arrow_table
        # Find a string column for grouping
        string_col = None
        for field in self.arrow_table.schema:
            if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
                string_col = field.name
                break

        result = self.sess.query(f"SELECT {string_col}, COUNT(*) as cnt FROM Python(my_arrow_table) GROUP BY {string_col} ORDER BY cnt DESC LIMIT 5", "CSV")
        lines = str(result).strip().split('\n')
        self.assertEqual(len(lines), 5)

    def test_arrow_table_aggregations(self):
        """Test aggregation functions on Arrow table"""
        my_arrow_table = self.arrow_table
        # Find a numeric column for aggregation
        numeric_col = None
        for field in self.arrow_table.schema:
            if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                numeric_col = field.name
                break

        result = self.sess.query(f"SELECT AVG({numeric_col}) as avg_val, MIN({numeric_col}) as min_val, MAX({numeric_col}) as max_val FROM Python(my_arrow_table)", "CSV")
        lines = str(result).strip().split('\n')
        self.assertEqual(len(lines), 1)

    def test_arrow_table_order_by(self):
        """Test ORDER BY queries on Arrow table"""
        my_arrow_table = self.arrow_table
        # Use first column for ordering
        first_col = self.arrow_table.schema.field(0).name

        result = self.sess.query(f"SELECT {first_col} FROM Python(my_arrow_table) ORDER BY {first_col} LIMIT 10", "CSV")
        lines = str(result).strip().split('\n')
        self.assertEqual(len(lines), 10)

    def test_arrow_table_subquery(self):
        """Test subqueries with Arrow table"""
        my_arrow_table = self.arrow_table
        result = self.sess.query("""
            SELECT COUNT(*) as total_count
            FROM (
                SELECT * FROM Python(my_arrow_table)
                WHERE WatchID IS NOT NULL
                LIMIT 1000
            ) subq
        """, "CSV")
        lines = str(result).strip().split('\n')
        self.assertEqual(len(lines), 1)
        count = int(lines[0])
        self.assertEqual(count, 1000)

    def test_arrow_table_multiple_tables(self):
        """Test using multiple Arrow tables in one query"""
        my_arrow_table = self.arrow_table
        # Create a smaller subset table
        subset_table = my_arrow_table.slice(0, min(100, my_arrow_table.num_rows))

        result = self.sess.query("""
            SELECT
                (SELECT COUNT(*) FROM Python(my_arrow_table)) as full_count,
                (SELECT COUNT(*) FROM Python(subset_table)) as subset_count
        """, "CSV")
        self.assertEqual(str(result).strip(), '1000000,100')


if __name__ == '__main__':
    unittest.main(verbosity=2)
