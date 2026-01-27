"""
Test explain() output for SQL-Pandas-SQL interleaved execution.

This test verifies that explain() accurately reflects the segmented execution plan,
showing proper SQL-Pandas-SQL interleaving instead of the old single-boundary approach.

Reference: example_daily_usage.ipynb scenario
"""

import unittest
import pandas as pd

from datastore import DataStore as ds
from tests.test_utils import assert_datastore_equals_pandas


class TestExplainSegmentedExecution(unittest.TestCase):
    """Test explain() output for segmented execution."""

    def test_sql_pandas_sql_interleaving_explain(self):
        """
        Test that explain() shows proper SQL-Pandas-SQL interleaving.

        Scenario from example_daily_usage.ipynb:
        1. filter (SQL)
        2. column assignment (Pandas)
        3. add_prefix (Pandas)
        4. filter (SQL on DataFrame)
        5. filter (SQL on DataFrame)
        6. column assignment (Pandas)
        7. sql() (chDB)
        8. filter (SQL on DataFrame)
        9. cast (chDB)

        The explain output should show multiple segments, not just 2 phases.
        """
        # Build the pipeline
        nat = ds.from_file('tests/dataset/users.csv')
        nat = nat.filter(nat.age < 35)
        nat['age_minus_10'] = nat['age'] - 10
        nat = nat.add_prefix('col_').filter(nat.col_age > 25).filter(nat.col_country == 'USA')
        nat['doubled'] = nat['col_age'] * 2
        nat2 = nat.sql('doubled > 35')
        nat2 = nat2.filter(nat2.doubled > 25)
        nat2['float_age'] = nat2.doubled.cast('Float64')

        # Get explain output
        explain_output = nat2.explain()

        # Verify multiple segments are shown (not just 2 phases)
        self.assertIn('Segment 1', explain_output)
        self.assertIn('Segment 2', explain_output)
        self.assertIn('Segment 3', explain_output)  # Should have at least 3 segments

        # Verify segment types are shown
        self.assertIn('[chDB]', explain_output)
        self.assertIn('[Pandas]', explain_output)

        # Verify the note about Python() table function
        self.assertIn('Python() table function', explain_output)

        # Verify operations are listed with correct engines
        self.assertIn('🚀 [chDB] WHERE: "age" < 35', explain_output)
        self.assertIn('🐼 [Pandas] Assign column', explain_output)
        self.assertIn('🐼 [Pandas] Add prefix', explain_output)

        # Key verification: filters after add_prefix should still use chDB
        # because they can be executed via Python() table function
        self.assertIn('🚀 [chDB] WHERE: "col_age" > 25', explain_output)
        self.assertIn('🚀 [chDB] WHERE: "col_country"', explain_output)

    def test_explain_matches_execution_result(self):
        """
        Test that the execution result matches pandas equivalent.

        Mirror Code Pattern: DataStore and pandas operations are mirrored.
        Focus on numeric columns to avoid date format differences.
        """
        # === DataStore operations ===
        ds_df = ds.from_file('tests/dataset/users.csv')
        ds_df = ds_df.filter(ds_df.age < 35)
        ds_df['age_minus_10'] = ds_df['age'] - 10
        ds_df = ds_df.add_prefix('col_').filter(ds_df.col_age > 25).filter(ds_df.col_country == 'USA')
        ds_df['doubled'] = ds_df['col_age'] * 2
        ds_result = ds_df.sql('doubled > 35')
        ds_result = ds_result.filter(ds_result.doubled > 25)
        ds_result['float_age'] = ds_result.doubled.cast('Float64')

        # === Pandas operations (mirror of DataStore) ===
        pd_df = pd.read_csv('tests/dataset/users.csv')
        pd_df = pd_df[pd_df['age'] < 35]
        pd_df['age_minus_10'] = pd_df['age'] - 10
        pd_df = pd_df.add_prefix('col_')
        pd_df = pd_df[pd_df['col_age'] > 25]
        pd_df = pd_df[pd_df['col_country'] == 'USA']
        pd_df['doubled'] = pd_df['col_age'] * 2
        pd_df = pd_df[pd_df['doubled'] > 35]
        pd_df = pd_df[pd_df['doubled'] > 25]
        pd_df['float_age'] = pd_df['doubled'].astype('float64')

        # Compare using assert_datastore_equals_pandas (complete comparison)
        # Note: after add_prefix('col_'), age_minus_10 becomes col_age_minus_10
        # Select only numeric columns to avoid date format differences from chDB
        numeric_cols = ['col_user_id', 'col_age', 'col_age_minus_10', 'doubled', 'float_age']
        pd_result = pd_df[numeric_cols]
        ds_result_selected = ds_result[numeric_cols]
        
        assert_datastore_equals_pandas(ds_result_selected, pd_result)
        
        # Verify row count matches
        self.assertEqual(len(ds_result), 3)  # Alice, Diana, Grace

    def test_explain_segment_numbering(self):
        """
        Test that segment operation numbers match display numbers.

        The segment summary should use the same numbering as the operation list,
        accounting for [1] being the data source.
        """
        nat = ds.from_file('tests/dataset/users.csv')
        nat = nat.filter(nat.age < 35)
        nat['age_minus_10'] = nat['age'] - 10

        explain_output = nat.explain()

        # Segment 1 should start at Operation 2 (after data source [1])
        self.assertIn('Segment 1', explain_output)
        self.assertIn('Operations 2-', explain_output)

        # Operation [2] should be the first filter
        self.assertIn('[2] 🚀 [chDB]', explain_output)

    def test_all_sql_no_pandas_segments(self):
        """
        Test explain output when all operations are SQL-pushable.
        """
        nat = ds.from_file('tests/dataset/users.csv')
        nat = nat.filter(nat.age < 35).filter(nat.age > 20)

        explain_output = nat.explain()

        # Should show only one SQL segment
        self.assertIn('Segment 1 [chDB] (from source)', explain_output)
        # Should not have any Pandas segments
        self.assertNotIn('Segment 2 [Pandas]', explain_output)

    def test_pandas_only_no_sql_source(self):
        """
        Test explain output when starting from DataFrame (no SQL source).
        """
        # Create DataStore from DataFrame directly
        pdf = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        nat = ds(pdf)
        nat = nat.filter(nat.age > 20)

        explain_output = nat.explain()

        # With DataFrame source, operations go through Python() table function
        # which is still SQL-capable
        self.assertIn('Segment 1', explain_output)


class TestSegmentedExecutionDailyUsage(unittest.TestCase):
    """Test the daily usage example specifically."""

    def test_daily_usage_example(self):
        """
        Reproduce the exact scenario from example_daily_usage.ipynb.

        This is the reference test for the explain() fix.
        """
        nat = ds.from_file('tests/dataset/users.csv')
        nat.sql("select 1").execute()  # Initial SQL test

        nat = nat.filter(nat.age < 35)
        nat['age_minus_10'] = nat['age'] - 10
        nat = nat.add_prefix('col_').filter(nat.col_age > 25).filter(nat.col_country == 'USA')
        nat['doubled'] = nat['col_age'] * 2

        nat2 = nat.sql("doubled > 35")
        nat2 = nat2.filter(nat2.doubled > 25)
        nat2["float_age"] = nat2.doubled.cast("Float64")

        # Get explain output
        explain_output = nat2.explain()

        # Verify the key fix: should have multiple segments
        segment_count = explain_output.count('Segment ')
        self.assertGreaterEqual(segment_count, 3, "Should have at least 3 segments for SQL-Pandas-SQL interleaving")

        # Verify filters after add_prefix use SQL (via Python() table function)
        # These were incorrectly shown as Pandas in the old implementation
        self.assertIn('🚀 [chDB] WHERE: "col_age" > 25', explain_output)
        self.assertIn('🚀 [chDB] WHERE: "col_country"', explain_output)

        # Verify execution produces correct result using natural triggers
        self.assertEqual(len(nat2), 3)  # Alice, Diana, Grace - natural trigger via len()
        self.assertIn('float_age', list(nat2.columns))  # Natural trigger via columns property


if __name__ == '__main__':
    unittest.main()

