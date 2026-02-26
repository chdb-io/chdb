"""
Test chdb integration with DataStore.

These tests demonstrate real query execution using chdb (embedded ClickHouse).
"""

import unittest
from datastore import DataStore, Sum, Count, Avg


class TestChdbBasics(unittest.TestCase):
    """Test basic chdb functionality"""

    def setUp(self):
        """Set up a test table with data"""
        self.ds = DataStore(table="users")
        self.ds.connect()

        # Create table
        self.ds.create_table({"id": "UInt64", "name": "String", "age": "UInt8", "city": "String"})

        # Insert test data
        self.ds.insert(
            [
                {"id": 1, "name": "Alice", "age": 25, "city": "NYC"},
                {"id": 2, "name": "Bob", "age": 30, "city": "LA"},
                {"id": 3, "name": "Charlie", "age": 35, "city": "NYC"},
                {"id": 4, "name": "Diana", "age": 28, "city": "SF"},
                {"id": 5, "name": "Eve", "age": 32, "city": "NYC"},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation"""
        try:
            # Drop table before closing to ensure clean state
            if self.ds._connection and self.ds._connection._conn:
                self.ds._connection._conn.query("DROP TABLE IF EXISTS users")
        except Exception:
            pass
        self.ds.close()

    def test_select_all(self):
        """Test SELECT * query"""
        result = self.ds.select("*").execute()

        self.assertEqual(5, len(result))
        self.assertEqual(4, len(result.column_names))
        self.assertIn("name", result.column_names)
        self.assertIn("age", result.column_names)

    def test_select_specific_columns(self):
        """Test SELECT specific columns"""
        result = self.ds.select("name", "age").execute()

        self.assertEqual(5, len(result))
        self.assertEqual(2, len(result.column_names))
        self.assertEqual(["name", "age"], result.column_names)

    def test_where_condition(self):
        """Test WHERE clause"""
        result = self.ds.select("*").filter(self.ds.age > 30).execute()

        self.assertEqual(2, len(result))  # Charlie and Eve

        # Check that all results have age > 30
        for row in result:
            age_idx = result.column_names.index("age")
            self.assertGreater(row[age_idx], 30)

    def test_where_string_equality(self):
        """Test WHERE with string equality"""
        result = self.ds.select("*").filter(self.ds.city == "NYC").execute()

        self.assertEqual(3, len(result))  # Alice, Charlie, Eve

    def test_where_and_condition(self):
        """Test WHERE with AND"""
        result = self.ds.select("*").filter((self.ds.age >= 30) & (self.ds.city == "NYC")).execute()

        self.assertEqual(2, len(result))  # Charlie and Eve

    def test_where_or_condition(self):
        """Test WHERE with OR"""
        result = self.ds.select("*").filter((self.ds.age < 28) | (self.ds.city == "LA")).execute()

        self.assertEqual(2, len(result))  # Alice and Bob

    def test_groupby_count(self):
        """Test GROUP BY with COUNT"""
        result = self.ds.select(self.ds.city, Count("*").as_("count")).groupby(self.ds.city).execute()

        self.assertEqual(3, len(result))  # 3 cities

        # Convert to dict for easier checking
        results_dict = {row[0]: row[1] for row in result.rows}
        self.assertEqual(3, results_dict["NYC"])
        self.assertEqual(1, results_dict["LA"])
        self.assertEqual(1, results_dict["SF"])

    def test_groupby_avg(self):
        """Test GROUP BY with AVG"""
        result = self.ds.select(self.ds.city, Avg(self.ds.age).as_("avg_age")).groupby(self.ds.city).execute()

        self.assertEqual(3, len(result))

        # Check that NYC has average age of (25 + 35 + 32) / 3 = 30.666...
        results_dict = {row[0]: row[1] for row in result.rows}
        self.assertAlmostEqual(30.666, results_dict["NYC"], places=2)

    def test_having_clause(self):
        """Test HAVING clause"""
        result = (
            self.ds.select(self.ds.city, Count("*").as_("count")).groupby(self.ds.city).having(Count("*") > 1).execute()
        )

        self.assertEqual(1, len(result))  # Only NYC has count > 1
        self.assertEqual("NYC", result.rows[0][0])

    def test_order_by(self):
        """Test ORDER BY"""
        result = self.ds.select("name", "age").sort(self.ds.age).execute()

        self.assertEqual(5, len(result))

        # Check that results are sorted by age ascending
        ages = [row[1] for row in result.rows]
        self.assertEqual(ages, sorted(ages))

    def test_order_by_desc(self):
        """Test ORDER BY DESC"""
        result = self.ds.select("name", "age").sort(self.ds.age, ascending=False).execute()

        self.assertEqual(5, len(result))

        # Check that results are sorted by age descending
        ages = [row[1] for row in result.rows]
        self.assertEqual(ages, sorted(ages, reverse=True))

    def test_limit(self):
        """Test LIMIT"""
        result = self.ds.select("*").limit(3).execute()

        self.assertEqual(3, len(result))

    def test_offset(self):
        """Test OFFSET"""
        result = self.ds.select("*").offset(2).execute()

        self.assertEqual(3, len(result))  # 5 total - 2 offset = 3

    def test_limit_and_offset(self):
        """Test LIMIT and OFFSET together"""
        result = self.ds.select("*").limit(2).offset(1).execute()

        self.assertEqual(2, len(result))

    def test_distinct(self):
        """Test DISTINCT"""
        result = self.ds.select(self.ds.city).distinct().execute()

        self.assertEqual(3, len(result))  # 3 unique cities

    def test_result_to_dict(self):
        """Test converting result to list of dicts (using orient='records')"""
        result = self.ds.select("name", "age").limit(2).execute()

        dicts = result.to_dict(orient='records')  # explicitly request records format
        self.assertEqual(2, len(dicts))
        self.assertIn("name", dicts[0])
        self.assertIn("age", dicts[0])

    def test_result_iteration(self):
        """Test iterating over result"""
        result = self.ds.select("name").execute()

        names = [row[0] for row in result]
        self.assertEqual(5, len(names))
        self.assertIn("Alice", names)


class TestChdbComplexQueries(unittest.TestCase):
    """Test more complex queries"""

    def setUp(self):
        """Set up test tables"""
        # Create orders table
        self.orders_ds = DataStore(table="orders")
        self.orders_ds.connect()

        self.orders_ds.create_table(
            {"order_id": "UInt64", "user_id": "UInt64", "amount": "Float64", "status": "String"}
        )

        self.orders_ds.insert(
            [
                {"order_id": 1, "user_id": 1, "amount": 100.0, "status": "completed"},
                {"order_id": 2, "user_id": 2, "amount": 200.0, "status": "completed"},
                {"order_id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
                {"order_id": 4, "user_id": 3, "amount": 50.0, "status": "pending"},
                {"order_id": 5, "user_id": 2, "amount": 300.0, "status": "completed"},
            ]
        )

    def tearDown(self):
        """Clean up - drop table to ensure test isolation"""
        try:
            if self.orders_ds._connection and self.orders_ds._connection._conn:
                self.orders_ds._connection._conn.query("DROP TABLE IF EXISTS orders")
        except Exception:
            pass
        self.orders_ds.close()

    def test_aggregation_sum(self):
        """Test SUM aggregation"""
        result = (
            self.orders_ds.select(Sum(self.orders_ds.amount).as_("total"))
            .filter(self.orders_ds.status == "completed")
            .execute()
        )

        self.assertEqual(1, len(result))
        self.assertEqual(750.0, result.rows[0][0])  # 100 + 200 + 150 + 300

    def test_groupby_sum(self):
        """Test GROUP BY with SUM"""
        result = (
            self.orders_ds.select(self.orders_ds.user_id, Sum(self.orders_ds.amount).as_("total"))
            .groupby(self.orders_ds.user_id)
            .execute()
        )

        self.assertEqual(3, len(result))  # 3 users

        # Convert to dict
        results_dict = {row[0]: row[1] for row in result.rows}
        self.assertEqual(250.0, results_dict[1])  # user 1: 100 + 150
        self.assertEqual(500.0, results_dict[2])  # user 2: 200 + 300
        self.assertEqual(50.0, results_dict[3])  # user 3: 50

    def test_multiple_aggregations(self):
        """Test multiple aggregations in one query"""
        result = (
            self.orders_ds.select(
                self.orders_ds.user_id,
                Count("*").as_("order_count"),
                Sum(self.orders_ds.amount).as_("total_amount"),
                Avg(self.orders_ds.amount).as_("avg_amount"),
            )
            .groupby(self.orders_ds.user_id)
            .execute()
        )

        self.assertEqual(3, len(result))

        # Check column names
        self.assertIn("user_id", result.column_names)
        self.assertIn("order_count", result.column_names)
        self.assertIn("total_amount", result.column_names)
        self.assertIn("avg_amount", result.column_names)


class TestChdbSqlGeneration(unittest.TestCase):
    """Test that generated SQL works with chdb"""

    def test_sql_without_execution(self):
        """Test that we can generate SQL without executing"""
        ds = DataStore(table="test")

        sql = ds.select("a", "b").filter(ds.c > 10).to_sql()

        self.assertIn("SELECT", sql)
        self.assertIn("FROM", sql)
        self.assertIn("WHERE", sql)

    def test_complex_sql_generation(self):
        """Test complex SQL generation"""
        ds = DataStore(table="test")

        sql = (
            ds.select(ds.category, Count("*").as_("count"))
            .filter((ds.price > 100) & (ds.status == "active"))
            .groupby(ds.category)
            .having(Count("*") > 5)
            .sort(ds.category)
            .limit(10)
            .to_sql()
        )

        self.assertIn("SELECT", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("HAVING", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("LIMIT", sql)


class TestWhereAliasConflictIntegration(unittest.TestCase):
    """
    Integration tests for ClickHouse WHERE alias conflict fix.

    This tests the bug where ClickHouse incorrectly uses a SELECT alias
    value in WHERE clause when the alias shadows an original column name.

    Example (buggy behavior):
        SELECT value*2 AS value FROM table WHERE value > 15
        # ClickHouse uses value*2 in WHERE instead of original value

    Fix: wrap to apply WHERE first in inner subquery, then compute in outer.
    """

    def test_filter_before_same_name_assign_from_file(self):
        """
        Integration test: filter -> assign(same name) from file source.

        Verifies that filter uses original column value, not computed alias.
        """
        import pandas as pd
        import tempfile
        import os

        # Create test data
        pd_df = pd.DataFrame({
            'name': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie'],
            'value': [10, 20, 30, 40, 50]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd_df.to_csv(f, index=False)
            csv_path = f.name

        try:
            # Expected pandas result
            pd_result = pd_df[pd_df['value'] > 15].copy()  # [20, 30, 40, 50]
            pd_result['value'] = pd_result['value'] * 2    # [40, 60, 80, 100]

            # DataStore
            ds = DataStore.from_file(csv_path)
            ds = ds[ds['value'] > 15]  # filter on ORIGINAL value
            ds['value'] = ds['value'] * 2  # then compute

            # Execute and compare
            ds_result = repr(ds)  # Triggers execution

            # Verify row count (should be 4, not 5)
            self.assertEqual(len(ds), len(pd_result),
                             f"Row count mismatch: ds={len(ds)}, pd={len(pd_result)}")

            # Verify values match pandas
            ds_values = ds['value']._execute().tolist()
            pd_values = pd_result['value'].tolist()
            self.assertEqual(ds_values, pd_values,
                             f"Values mismatch: ds={ds_values}, pd={pd_values}")

        finally:
            os.unlink(csv_path)

    def test_filter_assign_groupby_chain_from_file(self):
        """
        Integration test: filter -> assign(same name) -> groupby from file.

        This was the original bug scenario reported by the user.
        """
        import pandas as pd
        import tempfile
        import os

        pd_df = pd.DataFrame({
            'name': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie'],
            'value': [10, 20, 30, 40, 50]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            pd_df.to_csv(f, index=False)
            csv_path = f.name

        try:
            # Expected pandas result
            pd_filtered = pd_df[pd_df['value'] > 15].copy()
            pd_filtered['value'] = pd_filtered['value'] * 2
            pd_grouped = pd_filtered.groupby('name')['value'].sum()

            # DataStore
            ds = DataStore.from_file(csv_path)
            ds = ds[ds['value'] > 15]
            ds['value'] = ds['value'] * 2
            ds_grouped = ds.groupby('name')['value'].sum()

            # Compare results
            ds_executed = ds_grouped._execute()
            self.assertEqual(ds_executed['Alice'], pd_grouped['Alice'],
                             f"Alice mismatch: ds={ds_executed['Alice']}, pd={pd_grouped['Alice']}")
            self.assertEqual(ds_executed['Bob'], pd_grouped['Bob'],
                             f"Bob mismatch: ds={ds_executed['Bob']}, pd={pd_grouped['Bob']}")
            self.assertEqual(ds_executed['Charlie'], pd_grouped['Charlie'],
                             f"Charlie mismatch: ds={ds_executed['Charlie']}, pd={pd_grouped['Charlie']}")

        finally:
            os.unlink(csv_path)


if __name__ == '__main__':
    unittest.main()
