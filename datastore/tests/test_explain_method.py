"""
Tests for the explain() method.
"""

import unittest
import tempfile
import os
from datastore import DataStore


class TestExplainMethod(unittest.TestCase):
    """Test the explain() method."""

    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, "test.csv")

        with open(self.csv_file, "w") as f:
            f.write("id,name,age,salary\n")
            f.write("1,Alice,28,65000\n")
            f.write("2,Bob,32,70000\n")
            f.write("3,Charlie,26,55000\n")
            f.write("4,David,35,80000\n")

    def tearDown(self):
        """Clean up test data."""
        if os.path.exists(self.csv_file):
            os.unlink(self.csv_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_explain_returns_string(self):
        """Test that explain() returns a string."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*')

        output = result.explain()
        self.assertIsInstance(output, str)
        self.assertIn("Execution Plan", output)

    def test_explain_pure_sql(self):
        """Test explain() with pure SQL operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)

        output = result.explain()
        self.assertIn("Operations", output)
        self.assertIn("SELECT", output)
        self.assertIn("WHERE", output)

    def test_explain_mixed_operations(self):
        """Test explain() with mixed SQL and Pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_').filter(ds.p1_salary > 55000)

        output = result.explain()
        self.assertIn("Execution Plan", output)
        # Should show some operations
        self.assertIn("[1]", output)

    def test_explain_verbose_mode(self):
        """Test explain() with verbose=True."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_')

        normal_output = result.explain()
        verbose_output = result.explain(verbose=True)

        # Verbose should have more content
        self.assertGreaterEqual(len(verbose_output), len(normal_output))
        # Both should have basic info
        self.assertIn("Execution Plan", verbose_output)

    def test_explain_does_not_execute(self):
        """Test that explain() does not execute the query."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)

        # Call explain - should not execute
        output = result.explain()

        # Should show pending state
        self.assertIn("Pending", output)
        self.assertIn("Execution Plan", output)

    def test_explain_after_pandas_operations(self):
        """Test explain() after pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').add_prefix('p1_')

        output = result.explain()
        # Should show the operation history
        self.assertIn("Execution Plan", output)

    def test_explain_shows_sql_query(self):
        """Test that explain() shows the SQL query for unexecuted queries."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).filter(ds.salary > 60000)

        output = result.explain()
        self.assertIn("Generated SQL Query", output)
        self.assertIn("SELECT", output)
        self.assertIn("FROM", output)
        self.assertIn("WHERE", output)

    def test_explain_tracks_operations(self):
        """Test that explain() tracks operations correctly."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)
        result['doubled'] = result['age'] * 2
        result = result.filter(ds.salary > 55000)

        output = result.explain()

        # Should have numbered operations
        self.assertIn("[1]", output)
        self.assertIn("[2]", output)
        self.assertIn("[3]", output)
        self.assertIn("[4]", output)

    def test_explain_operation_order(self):
        """Test that explain() shows operations in original order."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_').filter(ds.p1_salary > 55000)

        output = result.explain()

        # Operations should appear in order
        select_idx = output.find("SELECT:")
        filter_idx = output.find("WHERE:")
        prefix_idx = output.find("Add prefix")

        self.assertLess(select_idx, filter_idx)
        self.assertLess(filter_idx, prefix_idx)

    def test_explain_with_no_operations(self):
        """Test explain() with a DataStore that has no operations."""
        ds = DataStore.from_file(self.csv_file)

        output = ds.explain()
        self.assertIn("Execution Plan", output)

    def test_explain_pandas_operation(self):
        """Test explain() with pandas operation (now lazy)."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').add_prefix('emp_').filter(ds.emp_age > 28)

        output = result.explain()
        # add_prefix is now lazy, should show in operations
        self.assertIn("Add prefix", output)

    def test_explain_extreme_many_operations(self):
        """Test explain() with 100+ mixed operations (only tests explain, not execution)."""
        ds = DataStore.from_file(self.csv_file)

        # Build a chain with 100 operations - but don't execute it!
        # Just track the operations for explain()
        result = ds.select('*')

        # 10 SQL filter operations (different filters to avoid redundancy)
        for i in range(10):
            result = result.filter(ds.age > 20)  # Same filter is OK for explain testing

        # Trigger execution
        result = result.add_prefix('p1_')

        # 40 mixed operations (SQL + Pandas)
        for i in range(20):
            # Track these operations without executing
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "p1_age" > {20 + i}',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"rename(id_{i})",
                    'details': {'shape': (2, 4)},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )

        # 50 more Pandas operations
        for i in range(25):
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"add_suffix('_s{i}')",
                    'details': {'shape': (2, 4)},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"add_prefix('p{i}_')",
                    'details': {'shape': (2, 4)},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )

        # explain should handle many operations without crashing
        output = result.explain()

        # Verify basic structure exists
        self.assertIn("Execution Plan", output)
        self.assertIn("Final State", output)

        # Verify operation count roughly correct
        # Numbers should go from [1] to [50+]
        self.assertIn("[1]", output)
        self.assertIn("[50]", output)

    def test_explain_extreme_deep_nesting(self):
        """Test explain() with deeply nested operations (explain only, not execution)."""
        ds = DataStore.from_file(self.csv_file)

        # Build a chain with many operations
        result = ds.select('*')

        # 25 SQL filter operations
        for i in range(25):
            result = result.filter(ds.age > 20 + i)

        # Execution
        result = result.add_prefix('mid_')

        # 25 more operations after execution (added to history)
        for i in range(25):
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "mid_age" > {20 + i}',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )

        output = result.explain()
        self.assertIn("Execution Plan", output)
        # Should have many operations (data source + select + 25 filters + execution + 25 post-mat)
        self.assertIn("[25]", output)  # Should have operation #25

    def test_explain_extreme_alternating_sql_pandas(self):
        """Test explain() with alternating SQL and Pandas operations (explain only)."""
        ds = DataStore.from_file(self.csv_file)

        # Start with SQL
        result = ds.select('*').filter(ds.age > 25)

        # Simulate alternating operations without executing
        # First pandas triggers execution
        result = result.add_prefix('p0_')

        # Add 24 more alternating operations to history (48 ops total)
        for i in range(1, 25):
            result._operation_history.append(
                {
                    'type': 'pandas',
                    'description': f"add_prefix('p{i}_')",
                    'details': {'shape': (2, 4), 'on_cached_df': True},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "p{i}_age" > 20',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )

        output = result.explain()

        # Should show correct execution and subsequent operations
        self.assertIn("Execution Point", output)
        self.assertIn("Post-Execution Operations", output)

        # Verify has 50+ operations
        self.assertIn("[50]", output)

    def test_explain_extreme_performance(self):
        """Test that explain() performs well with many operations (100+ ops)."""
        import time

        ds = DataStore.from_file(self.csv_file)

        # Build a chain with many operations using real API
        result = ds.select('*')

        # 50 SQL filter operations
        for i in range(50):
            result = result.filter(ds.age > 20 + (i % 10))

        # Trigger execution
        result = result.add_prefix('p1_')

        # 50 more operations after execution (added to history)
        for i in range(50):
            result._operation_history.append(
                {
                    'type': 'sql',
                    'description': f'WHERE "p1_age" > {20 + (i % 10)}',
                    'details': {'on_dataframe': True},
                    'is_on_dataframe': True,
                    'executed_at_call': True,
                }
            )

        # explain() should complete quickly (<2 seconds)
        start = time.time()
        output = result.explain()
        duration = time.time() - start

        self.assertLess(duration, 2.0, "explain() should complete in less than 2 seconds")
        # Should have many operations
        self.assertIn("[50]", output)

    def test_explain_only_pandas_operations(self):
        """Test explain() with only pandas operations (no explicit SQL)."""
        ds = DataStore.from_file(self.csv_file)

        # Only Pandas operations
        result = ds.add_prefix('p1_').add_suffix('_s1').rename(columns={'p1_id_s1': 'new_id'}).add_prefix('p2_')

        output = result.explain()

        # Should have execution plan
        self.assertIn("Execution Plan", output)

    def test_explain_shows_original_order(self):
        """Test that explain() shows operations in their original definition order."""
        ds = DataStore.from_file(self.csv_file)

        # Define operations in a specific order
        result = ds.select('name', 'age')
        result = result.filter(ds.age > 25)
        result['doubled'] = result['age'] * 2
        result = result.filter(ds.age < 50)
        result['tripled'] = result['age'] * 3
        result = result.sort('age', ascending=False)
        result = result.limit(10)

        output = result.explain()

        # Find positions of operations
        # Note: SQL operations use SQL terminology, Pandas operations use pandas terminology
        select_pos = output.find("SELECT:")
        filter1_pos = output.find("WHERE:")  # First filter is SQL
        doubled_pos = output.find("doubled")
        tripled_pos = output.find("tripled")
        # ORDER BY in Pandas phase uses "sort_values" terminology
        order_pos = output.find("sort_values:")
        if order_pos == -1:
            order_pos = output.find("ORDER BY:")  # Fallback for SQL phase
        # LIMIT in Pandas phase uses "head" terminology
        limit_pos = output.find("head:")
        if limit_pos == -1:
            limit_pos = output.find("LIMIT:")  # Fallback for SQL phase

        # Verify order: SELECT < FILTER < doubled < FILTER < tripled < ORDER BY < LIMIT
        self.assertLess(select_pos, filter1_pos, "SELECT should come before WHERE")
        self.assertLess(filter1_pos, doubled_pos, "WHERE should come before doubled assignment")
        self.assertLess(doubled_pos, tripled_pos, "doubled should come before tripled")
        self.assertLess(tripled_pos, order_pos, "tripled should come before sort")
        self.assertLess(order_pos, limit_pos, "sort should come before limit")


if __name__ == '__main__':
    unittest.main()
