import unittest
import os
import shutil
from typing import List, Any, Dict

from chdb import connect

db_path = "test_db_3fdds"


class TestCHDB(unittest.TestCase):
    def setUp(self):
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

    def test_conn_query_without_receiving_result(self):
        conn = connect()
        conn.query("SELECT 1", "CSV")
        conn.query("SELECT 1", "Null")
        conn.query("SELECT 1", "Null")
        conn.close()

    def test_basic_operations(self):
        conn = connect(":memory:")
        cursor = conn.cursor()
        # Create a table
        cursor.execute(
            """
            CREATE TABLE users (
                id Int32,
                name String,
                scores Array(UInt8)
            ) ENGINE = Memory
            """
        )

        # Insert test data
        cursor.execute(
            """
            INSERT INTO users VALUES
            (1, 'Alice', [95, 87, 92]),
            (2, 'Bob', [88, 85, 90]),
            (3, 'Charlie', [91, 89, 94])
            """
        )

        # Test fetchone
        cursor.execute("SELECT * FROM users WHERE id = 1")
        row = cursor.fetchone()
        print(row)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], "Alice")
        self.assertEqual(row[2], [95, 87, 92])

        cursor.execute("SELECT * FROM users WHERE id = 2")
        row = cursor.fetchone()
        print(row)
        self.assertEqual(row[0], 2)
        self.assertEqual(row[1], "Bob")
        self.assertEqual(row[2], [88, 85, 90])

        row = cursor.fetchone()
        self.assertIsNone(row)

        # Test fetchall
        print("fetchall")
        cursor.execute("SELECT * FROM users ORDER BY id")
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][1], "Bob")

        # Test iteration
        cursor.execute("SELECT * FROM users ORDER BY id")
        rows = [row for row in cursor]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2][1], "Charlie")
        cursor.close()
        conn.close()

    def test_connection_management(self):
        # Test file-based connection
        file_conn = connect(f"file:{db_path}")
        self.assertIsNotNone(file_conn)
        file_conn.close()

        # Test connection with parameters
        readonly_conn = connect(f"file:{db_path}?mode=ro")
        self.assertIsNotNone(readonly_conn)
        with self.assertRaises(Exception):
            cur = readonly_conn.cursor()
            cur.execute("CREATE TABLE test (id Int32) ENGINE = Memory")
        readonly_conn.close()

        # Test create dir fails
        with self.assertRaises(Exception):
            # try to create a directory with this test file name
            # which will fail surely
            connect("test_conn_cursor.py")

    def test_cursor_error_handling(self):
        conn = connect(":memory:")
        cursor = conn.cursor()
        try:
            # Test syntax error
            with self.assertRaises(Exception):
                cursor.execute("INVALID SQL QUERY")

            # Test table not found error
            with self.assertRaises(Exception):
                cursor.execute("SELECT * FROM nonexistent_table")
        finally:
            cursor.close()
            conn.close()

    def test_transaction_behavior(self):
        # Create test table
        conn = connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE test_transactions (
                id Int32,
                value String
            ) ENGINE = Memory
        """
        )

        # Test basic insert
        cursor.execute("INSERT INTO test_transactions VALUES (1, 'test')")
        cursor.commit()  # Should work even though Memory engine doesn't support transactions

        # Verify data
        cursor.execute("SELECT * FROM test_transactions")
        row = cursor.fetchone()
        self.assertEqual(row, (1, "test"))

    def test_cursor_data_types(self):
        conn = connect(":memory:")
        cursor = conn.cursor()
        # Test various data types
        cursor.execute(
            """
            CREATE TABLE type_test (
                int_val Int32,
                float_val Float64,
                string_val String,
                array_val Array(Int32),
                nullable_val Nullable(String),
                date_val Date,
                datetime_val DateTime
            ) ENGINE = Memory
        """
        )

        cursor.execute(
            """
            INSERT INTO type_test VALUES
            (42, 3.14, 'hello', [1,2,3], NULL, '2023-01-01', '2023-01-01 12:00:00')
        """
        )

        cursor.execute("SELECT * FROM type_test")
        row = cursor.fetchone()
        self.assertEqual(row[0], 42)
        self.assertAlmostEqual(row[1], 3.14)
        self.assertEqual(row[2], "hello")
        self.assertEqual(row[3], [1, 2, 3])
        self.assertIsNone(row[4])

    def test_cursor_multiple_results(self):
        conn = connect(":memory:")
        cursor = conn.cursor()
        # Create test data
        cursor.execute(
            """
            CREATE TABLE multi_test (id Int32, value String) ENGINE = Memory;
            INSERT INTO multi_test VALUES (1, 'one'), (2, 'two'), (3, 'three');
        """
        )

        # Test partial fetching
        cursor.execute("SELECT * FROM multi_test ORDER BY id")
        first_row = cursor.fetchone()
        self.assertEqual(first_row, (1, "one"))

        remaining_rows = cursor.fetchall()
        self.assertEqual(len(remaining_rows), 2)
        self.assertEqual(remaining_rows[0], (2, "two"))

    def test_query_formats(self):
        conn = connect(":memory:")
        cursor = conn.cursor()
        # Create test data
        cursor.execute(
            """
            CREATE TABLE format_test (id Int32, value String) ENGINE = Memory;
            INSERT INTO format_test VALUES (1, 'test');
        """
        )

        # Test different output formats
        csv_result = conn.query("SELECT * FROM format_test", format="CSV")
        self.assertIsNotNone(csv_result)

        arrow_result = conn.query("SELECT * FROM format_test", format="ArrowStream")
        self.assertIsNotNone(arrow_result)

    def test_cursor_statistics(self):
        conn = connect(":memory:?verbose&log-level=test")
        cursor = conn.cursor()
        # Create and populate test table
        cursor.execute(
            """
            CREATE TABLE stats_test (id Int32, value String) ENGINE = Memory;
            INSERT INTO stats_test SELECT number, toString(number)
            FROM numbers(1000);
        """
        )

        # Execute query and check statistics
        cursor.execute("SELECT * FROM stats_test")
        self.assertGreater(cursor._cursor.rows_read(), 0)
        self.assertGreater(cursor._cursor.bytes_read(), 0)
        self.assertGreater(cursor._cursor.elapsed(), 0)

    def test_memory_management(self):
        conn = connect(":memory:")
        cursor = conn.cursor()
        # Test multiple executions
        for i in range(10):
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            self.assertIsNotNone(cursor.fetchone())

        # Test large result sets
        cursor.execute(
            """
            SELECT number, toString(number) as str_num
            FROM numbers(1000000)
        """
        )
        rows = cursor.fetchall()
        self.assertEqual(len(rows), 1000000)

    def test_multiple_connections(self):
        conn1 = connect(":memory:")
        conn2 = connect(":memory:")
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()

        with self.assertRaises(Exception):
            connect("file:test.db")

        # Create table in first connection
        cursor1.execute(
            """
            CREATE TABLE test_table (id Int32, value String) ENGINE = Memory
        """
        )

        # Insert data in second connection
        cursor2.execute("INSERT INTO test_table VALUES (1, 'test')")
        cursor2.commit()

        # Query data from first connection
        cursor1.execute("SELECT * FROM test_table")
        row = cursor1.fetchone()
        self.assertEqual(row, (1, "test"))

        conn1.close()
        conn2.close()

    def test_connection_properties(self):
        # conn = connect("{db_path}?log_queries=1&verbose&log-level=test")
        with self.assertRaises(Exception):
            conn = connect(f"{db_path}?not_exist_flag=1")
        with self.assertRaises(Exception):
            conn = connect(f"{db_path}?verbose=1")

        conn = connect(f"{db_path}?verbose&log-level=test")
        ret = conn.query("SELECT 123", "CSV")
        print(ret)
        print(len(ret))
        self.assertEqual(str(ret), "123\n")
        ret = conn.query("show tables in system", "CSV")
        self.assertGreater(len(ret), 10)

        conn.close()

    def test_create_func(self):
        conn = connect(f"file:{db_path}")
        ret = conn.query("CREATE FUNCTION chdb_xxx AS () -> '0.12.0'", "CSV")
        ret = conn.query("SELECT chdb_xxx()", "CSV")
        self.assertEqual(str(ret), '"0.12.0"\n')
        conn.close()


if __name__ == "__main__":
    unittest.main()
