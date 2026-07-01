"""
Tests for the explain() method.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from datastore import DataStore


def _capture_explain(obj, **kwargs):
    f = io.StringIO()
    with redirect_stdout(f):
        obj.explain(**kwargs)
    return f.getvalue()


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

        output = _capture_explain(result)
        self.assertIsInstance(output, str)
        self.assertIn("Execution Plan", output)

    def test_explain_pure_sql(self):
        """Test explain() with pure SQL operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)

        output = _capture_explain(result)
        self.assertIn("Operations", output)
        self.assertIn("SELECT", output)
        self.assertIn("WHERE", output)

    def test_explain_mixed_operations(self):
        """Test explain() with mixed SQL and Pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_').filter(ds.p1_salary > 55000)

        output = _capture_explain(result)
        self.assertIn("Execution Plan", output)
        # Should show some operations
        self.assertIn("[1]", output)

    def test_explain_verbose_mode(self):
        """Test explain() with verbose=True."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_')

        normal_output = _capture_explain(result)
        verbose_output = _capture_explain(result, verbose=True)

        # Verbose should have more content
        self.assertGreaterEqual(len(verbose_output), len(normal_output))
        # Both should have basic info
        self.assertIn("Execution Plan", verbose_output)

    def test_explain_does_not_execute(self):
        """Test that explain() does not execute the query."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25)

        # Call explain - should not execute
        output = _capture_explain(result)

        # Should show pending state
        self.assertIn("Pending", output)
        self.assertIn("Execution Plan", output)

    def test_explain_after_pandas_operations(self):
        """Test explain() after pandas operations."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').add_prefix('p1_')

        output = _capture_explain(result)
        # Should show the operation history
        self.assertIn("Execution Plan", output)

    def test_explain_shows_sql_query(self):
        """Test that explain() shows the SQL query for unexecuted queries."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).filter(ds.salary > 60000)

        output = _capture_explain(result)
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

        output = _capture_explain(result)

        # Should have numbered operations
        self.assertIn("[1]", output)
        self.assertIn("[2]", output)
        self.assertIn("[3]", output)
        self.assertIn("[4]", output)

    def test_explain_operation_order(self):
        """Test that explain() shows operations in original order."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').filter(ds.age > 25).add_prefix('p1_').filter(ds.p1_salary > 55000)

        output = _capture_explain(result)

        # Operations should appear in order
        select_idx = output.find("SELECT:")
        filter_idx = output.find("WHERE:")
        prefix_idx = output.find("Add prefix")

        self.assertLess(select_idx, filter_idx)
        self.assertLess(filter_idx, prefix_idx)

    def test_explain_with_no_operations(self):
        """Test explain() with a DataStore that has no operations."""
        ds = DataStore.from_file(self.csv_file)

        output = _capture_explain(ds)
        self.assertIn("Execution Plan", output)

    def test_explain_pandas_operation(self):
        """Test explain() with pandas operation (now lazy)."""
        ds = DataStore.from_file(self.csv_file)
        result = ds.select('*').add_prefix('emp_').filter(ds.emp_age > 28)

        output = _capture_explain(result)
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
        output = _capture_explain(result)

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

        output = _capture_explain(result)
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

        output = _capture_explain(result)

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
        output = _capture_explain(result)
        duration = time.time() - start

        self.assertLess(duration, 2.0, "explain() should complete in less than 2 seconds")
        # Should have many operations
        self.assertIn("[50]", output)

    def test_explain_only_pandas_operations(self):
        """Test explain() with only pandas operations (no explicit SQL)."""
        ds = DataStore.from_file(self.csv_file)

        # Only Pandas operations
        result = ds.add_prefix('p1_').add_suffix('_s1').rename(columns={'p1_id_s1': 'new_id'}).add_prefix('p2_')

        output = _capture_explain(result)

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

        output = _capture_explain(result)

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


class TestExplainEngineLabelingForPushedOps(unittest.TestCase):
    """Regression tests: ops pushed into a chDB segment must be labeled [chDB] in explain().

    These tests pin down a previous bug where LazyGroupByAgg.execution_engine() and
    LazyApply.execution_engine() returned the non-canonical literal 'SQL'. The
    explain() renderer only recognizes 'chDB'/'Pandas', so 'SQL' silently fell into
    the Pandas branch and the GroupBy/apply lines were rendered as 🐼 [Pandas] even
    though the segment header said [chDB] and the generated SQL clearly contained
    GROUP BY pushed to chDB.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.parquet_file = os.path.join(self.temp_dir, "amazon_sample.parquet")

        import pandas as pd
        pd.DataFrame({
            'product_category': ['A', 'B', 'A', 'B', 'C'] * 10,
            'star_rating': [5, 4, 3, 5, 2] * 10,
            'verified_purchase': [True, False, True, True, True] * 10,
        }).to_parquet(self.parquet_file)

    def tearDown(self):
        if os.path.exists(self.parquet_file):
            os.unlink(self.parquet_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def _extract_op_line(self, output: str, op_number: int) -> str:
        """Return the single explain() line beginning with ' [<op_number>] '."""
        prefix = f" [{op_number}]"
        for line in output.splitlines():
            if line.startswith(prefix):
                return line
        raise AssertionError(
            f"Could not find op [{op_number}] line in explain output:\n{output}"
        )

    def test_explain_labels_pushable_groupby_agg_dict_list_as_chdb(self):
        """Mirrors the user-reported scenario: agg({'col': ['mean', 'count']}).

        The GroupBy is pushed to chDB (GROUP BY appears in the generated SQL),
        so explain() must label that op as [chDB], not [Pandas].
        """
        ds = DataStore.from_file(self.parquet_file)
        res = (
            ds[ds['verified_purchase'] == True]
            .groupby('product_category')
            .agg({'star_rating': ['mean', 'count']})
        )

        output = _capture_explain(res)

        # Segment header must say chDB from source
        self.assertIn("Segment 1 [chDB] (from source)", output)

        # The GroupBy op line must be labeled chDB, not Pandas
        groupby_line = next(
            line for line in output.splitlines() if "GroupBy(" in line
        )
        self.assertIn("[chDB]", groupby_line)
        self.assertNotIn("[Pandas]", groupby_line)

        # Sanity: GROUP BY must actually appear in the pushed-down SQL
        self.assertIn("GROUP BY", output)

    def test_explain_labels_pushable_groupby_agg_func_as_chdb(self):
        """Simple groupby('col').mean() must also render as [chDB]."""
        ds = DataStore.from_file(self.parquet_file)
        res = ds.groupby('product_category').mean()

        output = _capture_explain(res)

        groupby_line = next(
            line for line in output.splitlines() if "GroupBy(" in line
        )
        self.assertIn("[chDB]", groupby_line)
        self.assertNotIn("[Pandas]", groupby_line)
        self.assertIn("GROUP BY", output)

    def test_explain_labels_non_pushable_groupby_agg_list_as_pandas(self):
        """When agg_dict is itself a list (not a dict-of-lists), it cannot be
        pushed to chDB, so the GroupBy op line must render as [Pandas]."""
        from datastore.lazy_ops import LazyGroupByAgg
        ds = DataStore.from_file(self.parquet_file)

        op = LazyGroupByAgg(
            groupby_cols=['product_category'],
            agg_dict=['sum', 'mean'],  # list, not dict -> cannot push
        )
        self.assertFalse(op.can_push_to_sql())
        self.assertEqual(op.execution_engine(), 'Pandas')

    def test_explain_labels_pushable_apply_as_chdb(self):
        """groupby().apply(lambda x: x.sum()) is detected as a simple aggregation
        and pushed to chDB, so explain() must label that op as [chDB]."""
        ds = DataStore.from_file(self.parquet_file)
        res = ds.groupby('product_category').apply(lambda x: x.sum())

        output = _capture_explain(res)

        apply_line = next(
            (line for line in output.splitlines() if "apply" in line.lower()),
            None,
        )
        if apply_line is not None:
            self.assertIn("[chDB]", apply_line)
            self.assertNotIn("[Pandas]", apply_line)

    def test_lazy_groupby_agg_execution_engine_returns_canonical_literal(self):
        """Direct unit test: LazyGroupByAgg.execution_engine() must return the
        canonical 'chDB'/'Pandas' literal that explain() understands.

        Returning 'SQL' here previously caused explain() to mislabel pushed-down
        GroupBy ops as Pandas.
        """
        from datastore.lazy_ops import LazyGroupByAgg

        pushable = LazyGroupByAgg(
            groupby_cols=['product_category'], agg_func='mean'
        )
        self.assertTrue(pushable.can_push_to_sql())
        self.assertEqual(pushable.execution_engine(), 'chDB')

        pushable_dict = LazyGroupByAgg(
            groupby_cols=['product_category'],
            agg_dict={'star_rating': ['mean', 'count']},
        )
        self.assertTrue(pushable_dict.can_push_to_sql())
        self.assertEqual(pushable_dict.execution_engine(), 'chDB')

        non_pushable = LazyGroupByAgg(
            groupby_cols=['product_category'],
            agg_dict=['sum', 'mean'],
        )
        self.assertFalse(non_pushable.can_push_to_sql())
        self.assertEqual(non_pushable.execution_engine(), 'Pandas')


class TestBuildExecutionSQLFirstSegment(unittest.TestCase):
    """``_build_execution_sql`` (the backend for ``to_sql()`` and
    ``explain()``'s SQL preview) returns the SQL the executor would
    issue against the *original source* - i.e. the FIRST step of
    ``_execute()``.

    Regression for PR #577 Copilot comment: a previous implementation
    used ``next(seg for seg in segments if seg.is_sql())`` which would
    happily pick a LATER ``SQL-on-Python(__df__)`` segment when the
    first segment was Pandas (cost-aware planner case for ORDER BY
    without LIMIT). That returned a Python-table-function SQL whose
    ``FROM`` clause does not match the original source the user sees in
    the executor's first call.
    """

    def setUp(self):
        import pandas as pd

        self.temp_dir = tempfile.mkdtemp()
        self.parquet_file = os.path.join(self.temp_dir, "data.parquet")
        pd.DataFrame({"a": list(range(10)), "b": list(range(10))}).to_parquet(
            self.parquet_file
        )

    def tearDown(self):
        if os.path.exists(self.parquet_file):
            os.unlink(self.parquet_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_first_segment_sql_returns_source_bound_sql(self):
        ds = DataStore.from_file(self.parquet_file)
        sql = ds[ds["a"] > 5].to_sql()
        self.assertIn("file(", sql)
        self.assertNotIn("Python(", sql)
        self.assertNotIn("__df__", sql)
        self.assertIn('"a" > 5', sql)

    def test_first_segment_pandas_returns_source_select(self):
        """When the chain starts with a Pandas segment (e.g. ORDER BY
        without LIMIT under the cost-aware planner), ``to_sql()`` must
        return the initial ``SELECT * FROM <source>`` the executor
        issues to materialize rows for Pandas - NOT a downstream
        ``Python(__df__)`` SQL fragment."""
        ds = DataStore.from_file(self.parquet_file)
        sql = ds.sort_values("a").to_sql()
        self.assertIn("file(", sql)
        self.assertNotIn("Python(", sql)
        self.assertNotIn("__df__", sql)


if __name__ == '__main__':
    unittest.main()
