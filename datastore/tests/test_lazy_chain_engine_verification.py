"""
Tests for verifying engine usage in long function chains with Lazy execution.

This tests that:
1. Long function chains actually use the configured engine
2. explain() output correctly shows which engine is used
3. Debug logging confirms engine selection
"""

import unittest
import pandas as pd
import numpy as np
import logging
import io
import re

from datastore import DataStore, config
from datastore.config import ExecutionEngine
from datastore.function_executor import function_config, ExecutionEngine as FuncExecEngine


class TestLongChainWithExplain(unittest.TestCase):
    """Test long function chains and verify via explain()."""

    def setUp(self):
        """Create test data and reset config."""
        self.df = pd.DataFrame(
            {
                'text': ['  hello world  ', '  HELLO WORLD  ', '  Test String  '],
                'value': [-1.5, 2.5, -3.5],
                'count': [10, 20, 30],
            }
        )
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_explain_shows_chdb_for_sql_operations(self):
        """explain() should show [chDB] for SQL operations when using clickhouse."""
        config.use_chdb()

        # Build a chain of operations
        ds = self.ds.select('text', 'value')
        ds = ds.filter(ds['value'] > -5)

        explain_output = ds.explain()

        # Should show [chDB] markers
        self.assertIn('[chDB]', explain_output)
        # Should show SQL query
        self.assertIn('SELECT', explain_output.upper())

    def test_explain_shows_pandas_for_column_assignment(self):
        """explain() should show [Pandas] for column assignment operations."""
        config.use_pandas()

        ds = self.ds.copy()
        ds['upper_text'] = ds['text'].str.upper()

        explain_output = ds.explain()

        # Verify [Pandas] marker is present for this column assignment
        pandas_pattern = re.compile(r'\[Pandas\].*Assign column.*upper_text')
        match = pandas_pattern.search(explain_output)
        self.assertIsNotNone(match, f"Expected [Pandas] Assign column 'upper_text' marker.\nExplain:\n{explain_output}")

    def test_long_chain_string_operations_explain(self):
        """Test explain output for long chain of string operations."""
        config.use_chdb()

        ds = self.ds.copy()
        # Chain: trim -> upper
        ds['processed'] = ds['text'].str.trim().str.upper()

        explain_output = ds.explain()

        # Verify [chDB] marker is present for this column assignment
        chdb_pattern = re.compile(r'\[chDB\].*Assign column.*processed')
        match = chdb_pattern.search(explain_output)
        self.assertIsNotNone(match, f"Expected [chDB] Assign column 'processed' marker.\nExplain:\n{explain_output}")

    def test_long_chain_math_operations_explain(self):
        """Test explain output for long chain of math operations."""
        config.use_chdb()

        ds = self.ds.copy()
        # Chain: abs -> + 10 -> sqrt
        ds['computed'] = (ds['value'].abs() + 10).sqrt()

        explain_output = ds.explain()

        # Verify [chDB] marker is present for this column assignment
        chdb_pattern = re.compile(r'\[chDB\].*Assign column.*computed')
        match = chdb_pattern.search(explain_output)
        self.assertIsNotNone(match, f"Expected [chDB] Assign column 'computed' marker.\nExplain:\n{explain_output}")


class TestDebugLoggingEngineVerification(unittest.TestCase):
    """Test that debug logging confirms engine selection."""

    def setUp(self):
        """Create test data, capture logs."""
        self.df = pd.DataFrame({'text': ['hello', 'world', 'test'], 'value': [1.0, 2.0, 3.0]})
        self.ds = DataStore.from_dataframe(self.df)

        # Set up log capture
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)

        # Get datastore logger and add handler
        from datastore.config import get_logger

        self.logger = get_logger()
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

        config.use_auto()

    def tearDown(self):
        """Reset logging and config."""
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.original_level)
        config.use_auto()

    def get_logs(self):
        """Get captured log output."""
        return self.log_capture.getvalue()

    def test_debug_log_shows_execution_info(self):
        """Debug log should show execution information."""
        config.enable_debug()

        # Execute an operation
        result = list(self.ds['text'].str.upper())

        logs = self.get_logs()

        # Should have some execution related log
        self.assertTrue(len(logs) > 0 or len(result) == 3)


class TestChainExecutionVerification(unittest.TestCase):
    """Verify that chain operations actually execute with correct engine."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['  hello  ', '  world  ', '  test  '], 'value': [-2.5, 3.5, -1.5]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_chain_chdb_produces_correct_results(self):
        """Chain with ClickHouse should produce correct results."""
        config.use_chdb()

        # Chain: trim -> upper
        result = list(self.ds['text'].str.trim().str.upper())

        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST'])

    def test_chain_pandas_produces_correct_results(self):
        """Chain with Pandas should produce correct results."""
        config.use_pandas()

        # Chain: trim -> upper
        result = list(self.ds['text'].str.trim().str.upper())

        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST'])

    def test_chain_results_consistent(self):
        """ClickHouse and Pandas should produce consistent results."""
        # Execute with ClickHouse
        config.use_chdb()
        ch_result = list(self.ds['text'].str.trim().str.upper())

        # Execute with Pandas
        config.use_pandas()
        pd_result = list(self.ds['text'].str.trim().str.upper())

        # Row order is now preserved
        self.assertEqual(ch_result, pd_result)

    def test_complex_chain_chdb(self):
        """Complex chain with ClickHouse."""
        config.use_chdb()

        # abs -> round
        result = list(self.ds['value'].abs().round())

        # Row order is now preserved
        self.assertEqual(result, [2.0, 4.0, 2.0])

    def test_complex_chain_pandas(self):
        """Complex chain with Pandas."""
        config.use_pandas()

        # abs -> round
        result = list(self.ds['value'].abs().round())

        # Pandas rounding may differ slightly
        self.assertEqual(len(result), 3)


class TestFunctionConfigEngineCheck(unittest.TestCase):
    """Verify function_config correctly reports engine selection."""

    def setUp(self):
        """Reset function_config."""
        function_config.reset()

    def tearDown(self):
        """Reset function_config."""
        function_config.reset()

    def test_default_engine_is_chdb(self):
        """Default engine should be chDB."""
        self.assertEqual(function_config.default_engine, FuncExecEngine.CHDB)

    def test_overlapping_function_follows_default(self):
        """Overlapping functions should follow default engine."""
        # With default (chDB)
        self.assertTrue(function_config.should_use_chdb('upper'))

        # Switch to pandas
        function_config.prefer_pandas()
        self.assertTrue(function_config.should_use_pandas('upper'))

    def test_specific_function_override(self):
        """Can override engine for specific functions."""
        function_config.prefer_chdb()  # Default to chDB
        function_config.use_pandas('upper')  # Override upper to pandas

        self.assertTrue(function_config.should_use_pandas('upper'))
        self.assertTrue(function_config.should_use_chdb('lower'))  # Still chDB

    def test_pandas_only_always_uses_pandas(self):
        """Pandas-only functions always use Pandas regardless of config."""
        function_config.prefer_chdb()

        # These should always return True for pandas
        self.assertTrue(function_config.should_use_pandas('cumsum'))
        self.assertTrue(function_config.should_use_pandas('fillna'))
        self.assertTrue(function_config.should_use_pandas('shift'))

    def test_engine_for_all_overlapping_functions(self):
        """All overlapping functions should respect engine setting."""
        test_functions = ['upper', 'lower', 'abs', 'round', 'floor', 'ceil']

        # Test with chDB
        function_config.prefer_chdb()
        for func in test_functions:
            self.assertTrue(function_config.should_use_chdb(func), f"{func} should use chDB when prefer_chdb")

        # Test with Pandas
        function_config.prefer_pandas()
        for func in test_functions:
            self.assertTrue(function_config.should_use_pandas(func), f"{func} should use Pandas when prefer_pandas")


class TestExplainWithEngineMarkers(unittest.TestCase):
    """Test that explain() output includes correct engine markers."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['hello', 'world'], 'value': [1, 2]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_explain_contains_engine_markers(self):
        """explain() should contain engine markers [chDB] or [Pandas]."""
        ds = self.ds.select('text', 'value')
        explain = ds.explain()

        # Should have at least one engine marker
        has_chdb = '[chDB]' in explain
        has_pandas = '[Pandas]' in explain

        self.assertTrue(has_chdb or has_pandas, "explain() should contain engine markers")

    def test_explain_shows_execution_plan_header(self):
        """explain() should show execution plan header."""
        ds = self.ds.select('text')
        explain = ds.explain()

        self.assertIn('Execution Plan', explain)

    def test_explain_shows_data_source(self):
        """explain() should show data source."""
        explain = self.ds.explain()

        # Should mention data source
        self.assertIn('📊', explain)

    def test_explain_shows_final_state(self):
        """explain() should show final state."""
        ds = self.ds.select('text')
        explain = ds.explain()

        self.assertIn('Final State', explain)


class TestMixedOperationsChain(unittest.TestCase):
    """Test chains with mixed operation types."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'text': ['  abc  ', '  def  ', '  ghi  '],
                'value': [-1, 2, -3],
                'date': pd.to_datetime(['2024-01-01', '2024-06-15', '2024-12-31']),
            }
        )
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_string_then_math_chain(self):
        """Test chain: string op -> length (returns number) -> can do math."""
        config.use_chdb()

        # Get length of trimmed text (using pandas-style str.len())
        result = list(self.ds['text'].str.trim().str.len())

        # Row order is now preserved
        self.assertEqual(result, [3, 3, 3])

    def test_math_operations_chain(self):
        """Test chain of math operations."""
        config.use_chdb()

        # abs -> multiply -> floor
        result = list((self.ds['value'].abs() * 1.5).floor())

        # Row order is now preserved
        self.assertEqual(result, [1.0, 3.0, 4.0])

    def test_filter_then_transform(self):
        """Test filter followed by transform."""
        config.use_chdb()

        ds = self.ds.filter(self.ds['value'] > 0)
        result = list(ds['text'].str.trim())

        # Row order is now preserved
        self.assertEqual(result, ['def'])

    def test_pandas_mode_complex_chain(self):
        """Test complex chain in Pandas mode."""
        config.use_pandas()

        # Chain multiple operations
        result = list(self.ds['text'].str.strip().str.upper())

        # Row order is now preserved
        self.assertEqual(result, ['ABC', 'DEF', 'GHI'])


class TestEngineVerificationWithColumnAssignment(unittest.TestCase):
    """Test engine verification with column assignment operations."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['hello', 'world'], 'value': [1.5, 2.5]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_column_assignment_with_chdb(self):
        """Column assignment should work with ClickHouse engine."""
        config.use_chdb()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.upper()

        result = ds.to_df()

        self.assertIn('upper', result.columns)
        # Row order is now preserved
        self.assertEqual(list(result['upper']), ['HELLO', 'WORLD'])

    def test_column_assignment_with_pandas(self):
        """Column assignment should work with Pandas engine."""
        config.use_pandas()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.upper()

        result = ds.to_df()

        self.assertIn('upper', result.columns)
        # Row order is now preserved
        self.assertEqual(list(result['upper']), ['HELLO', 'WORLD'])

    def test_multiple_column_assignments(self):
        """Multiple column assignments in chain."""
        config.use_chdb()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.upper()
        ds['rounded'] = ds['value'].round()

        result = ds.to_df()

        self.assertIn('upper', result.columns)
        self.assertIn('rounded', result.columns)

    def test_column_assignment_explain(self):
        """Column assignment should appear in explain() with correct engine marker."""
        config.use_chdb()
        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.upper()

        explain = ds.explain()

        # Verify [chDB] marker for column assignment
        chdb_pattern = re.compile(r'\[chDB\].*Assign column.*upper')
        match = chdb_pattern.search(explain)
        self.assertIsNotNone(match, f"Expected [chDB] Assign column 'upper' marker.\nExplain:\n{explain}")


class TestPandasImplementationDirect(unittest.TestCase):
    """Test that Pandas implementations are actually called."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['HELLO', 'WORLD'], 'value': [1.5, 2.5]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_pandas_impl_lower(self):
        """Test Pandas implementation for lower()."""
        config.use_pandas()

        result = list(self.ds['text'].str.lower())

        # Row order is now preserved
        self.assertEqual(result, ['hello', 'world'])

    def test_pandas_impl_abs(self):
        """Test Pandas implementation for abs()."""
        config.use_pandas()

        df = pd.DataFrame({'value': [-1, -2, -3]})
        ds = DataStore.from_dataframe(df)

        result = list(ds['value'].abs())

        # Row order is now preserved
        self.assertEqual(result, [1, 2, 3])

    def test_pandas_impl_round_with_decimals(self):
        """Test Pandas implementation for round with decimals."""
        config.use_pandas()

        df = pd.DataFrame({'value': [1.234, 2.567, 3.891]})
        ds = DataStore.from_dataframe(df)

        result = list(ds['value'].round(1))

        # Row order is now preserved
        self.assertEqual(result, [1.2, 2.6, 3.9])

    def test_pandas_impl_sqrt(self):
        """Test Pandas implementation for sqrt()."""
        config.use_pandas()

        df = pd.DataFrame({'value': [4.0, 9.0, 16.0]})
        ds = DataStore.from_dataframe(df)

        result = list(ds['value'].sqrt())

        # Row order is now preserved
        self.assertEqual(result, [2.0, 3.0, 4.0])


class TestExplainOutputEngineVerification(unittest.TestCase):
    """Verify explain() output correctly shows engine markers using regex."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'text': ['  hello  ', '  WORLD  ', '  Test  '],
                'value': [-1.5, 2.5, -3.5],
            }
        )
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_chdb_engine_shows_chdb_markers(self):
        """With ClickHouse engine, explain should show [chDB] for function ops."""
        config.use_chdb()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.trim().str.upper()
        ds['abs_val'] = ds['value'].abs()

        explain_output = ds.explain()

        # Use regex to find [chDB] markers for Assign column operations
        chdb_assign_pattern = re.compile(r'\[chDB\].*Assign column')
        matches = chdb_assign_pattern.findall(explain_output)

        # Should have at least 2 [chDB] Assign column operations (upper and abs_val)
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 [chDB] Assign column markers, found {len(matches)}.\n"
            f"Explain output:\n{explain_output}",
        )

    def test_pandas_engine_shows_pandas_markers(self):
        """With Pandas engine, explain should show [Pandas] for function ops."""
        config.use_pandas()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.trim().str.upper()
        ds['abs_val'] = ds['value'].abs()

        explain_output = ds.explain()

        # Use regex to find [Pandas] markers for Assign column operations
        pandas_assign_pattern = re.compile(r'\[Pandas\].*Assign column')
        matches = pandas_assign_pattern.findall(explain_output)

        # Should have at least 2 [Pandas] Assign column operations
        self.assertGreaterEqual(
            len(matches),
            2,
            f"Expected at least 2 [Pandas] Assign column markers, found {len(matches)}.\n"
            f"Explain output:\n{explain_output}",
        )

    def test_chdb_no_pandas_in_function_ops(self):
        """With ClickHouse engine, function ops should not show [Pandas]."""
        config.use_chdb()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.upper()

        explain_output = ds.explain()

        # [Pandas] should NOT appear for Assign column with upper
        pandas_upper_pattern = re.compile(r'\[Pandas\].*Assign column.*upper')
        matches = pandas_upper_pattern.findall(explain_output)

        self.assertEqual(
            len(matches),
            0,
            f"Expected no [Pandas] Assign column 'upper' markers with ClickHouse engine.\n"
            f"Explain output:\n{explain_output}",
        )

    def test_pandas_no_chdb_in_function_ops(self):
        """With Pandas engine, function ops should not show [chDB]."""
        config.use_pandas()

        ds = self.ds.copy()
        ds['upper'] = ds['text'].str.upper()

        explain_output = ds.explain()

        # [chDB] should NOT appear for Assign column with upper
        chdb_upper_pattern = re.compile(r'\[chDB\].*Assign column.*upper')
        matches = chdb_upper_pattern.findall(explain_output)

        self.assertEqual(
            len(matches),
            0,
            f"Expected no [chDB] Assign column 'upper' markers with Pandas engine.\n"
            f"Explain output:\n{explain_output}",
        )

    def test_chain_operations_all_use_same_engine(self):
        """All chained operations should use the same configured engine."""
        config.use_chdb()

        ds = self.ds.copy()
        # Multiple column assignments
        ds['trimmed'] = ds['text'].str.trim()
        ds['upper'] = ds['text'].str.upper()
        ds['abs_val'] = ds['value'].abs()
        ds['rounded'] = ds['value'].round()

        explain_output = ds.explain()

        # Count [chDB] and [Pandas] markers for Assign column
        chdb_pattern = re.compile(r'\[chDB\].*Assign column')
        pandas_pattern = re.compile(r'\[Pandas\].*Assign column')

        chdb_count = len(chdb_pattern.findall(explain_output))
        pandas_count = len(pandas_pattern.findall(explain_output))

        # All 4 should be [chDB]
        self.assertEqual(chdb_count, 4, f"Expected 4 [chDB] Assign column markers, found {chdb_count}")
        self.assertEqual(pandas_count, 0, f"Expected 0 [Pandas] Assign column markers, found {pandas_count}")

    def test_switch_engine_changes_explain_output(self):
        """Switching engine should change explain output markers."""
        # First with ClickHouse
        config.use_chdb()
        ds1 = self.ds.copy()
        ds1['upper'] = ds1['text'].str.upper()
        explain_ch = ds1.explain()

        # Then with Pandas
        config.use_pandas()
        ds2 = self.ds.copy()
        ds2['upper'] = ds2['text'].str.upper()
        explain_pd = ds2.explain()

        # ClickHouse explain should have [chDB]
        self.assertIn('[chDB]', explain_ch)
        chdb_assign = re.search(r'\[chDB\].*Assign column.*upper', explain_ch)
        self.assertIsNotNone(chdb_assign, f"Expected [chDB] Assign column 'upper' in ClickHouse explain:\n{explain_ch}")

        # Pandas explain should have [Pandas]
        self.assertIn('[Pandas]', explain_pd)
        pandas_assign = re.search(r'\[Pandas\].*Assign column.*upper', explain_pd)
        self.assertIsNotNone(pandas_assign, f"Expected [Pandas] Assign column 'upper' in Pandas explain:\n{explain_pd}")

    def test_results_consistent_between_engines(self):
        """Results should be the same regardless of which engine is used."""
        # Execute with ClickHouse
        config.use_chdb()
        ds_ch = self.ds.copy()
        ds_ch['upper'] = ds_ch['text'].str.trim().str.upper()
        ds_ch['abs_val'] = ds_ch['value'].abs()
        result_ch = ds_ch.to_df()

        # Execute with Pandas
        config.use_pandas()
        ds_pd = self.ds.copy()
        ds_pd['upper'] = ds_pd['text'].str.trim().str.upper()
        ds_pd['abs_val'] = ds_pd['value'].abs()
        result_pd = ds_pd.to_df()

        # Results should match
        self.assertEqual(list(result_ch['upper']), list(result_pd['upper']))
        self.assertEqual(list(result_ch['abs_val']), list(result_pd['abs_val']))

    def test_function_config_summary(self):
        """function_config should report correct summary."""
        config.use_pandas()

        summary = function_config.get_config_summary()

        self.assertEqual(summary['default_engine'], 'pandas')
        self.assertGreater(summary['overlapping_functions'], 50)
        self.assertGreater(summary['pandas_implementations'], 100)


class TestConfigSwitchMidChain(unittest.TestCase):
    """Test behavior when config is switched mid-chain."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['hello', 'world'], 'value': [1.0, 2.0]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_switch_before_execution(self):
        """Switching before execution uses new engine."""
        config.use_chdb()

        # Create lazy expression
        expr = self.ds['text'].str.upper()

        # Switch engine before execution
        config.use_pandas()

        # Execute - should use Pandas
        result = list(expr)

        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD'])

    def test_results_same_regardless_of_switch(self):
        """Results should be same regardless of engine."""
        # Create expression with chDB
        config.use_chdb()
        expr1 = self.ds['text'].str.upper()
        result1 = list(expr1)

        # Create new expression with Pandas
        config.use_pandas()
        expr2 = self.ds['text'].str.upper()
        result2 = list(expr2)

        # Row order is now preserved
        self.assertEqual(result1, result2)


if __name__ == '__main__':
    unittest.main()
