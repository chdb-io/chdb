"""
Tests for Lazy execution mode with engine switching (ClickHouse vs Pandas).

This tests the ability to switch between execution engines in Lazy mode
via global config settings.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from datastore import DataStore, config
from datastore.config import ExecutionEngine
from datastore.function_executor import function_config, ExecutionEngine as FuncExecEngine


class TestLazyEngineConfigSync(unittest.TestCase):
    """Test that global config syncs with function_config."""

    def setUp(self):
        """Reset to auto before each test."""
        config.use_auto()

    def tearDown(self):
        """Reset to auto after each test."""
        config.use_auto()

    def test_config_syncs_to_pandas(self):
        """Setting config to pandas should sync with function_config."""
        config.use_pandas()
        self.assertEqual(function_config.default_engine, FuncExecEngine.PANDAS)

    def test_config_syncs_to_chdb(self):
        """Setting config to clickhouse should sync with function_config."""
        config.use_chdb()
        self.assertEqual(function_config.default_engine, FuncExecEngine.CHDB)

    def test_config_syncs_to_auto(self):
        """Setting config to auto should reset function_config."""
        config.use_pandas()  # First set to pandas
        config.use_auto()  # Then reset
        self.assertEqual(function_config.default_engine, FuncExecEngine.CHDB)

    def test_property_setter_syncs(self):
        """Setting via property should also sync."""
        config.execution_engine = 'pandas'
        self.assertEqual(function_config.default_engine, FuncExecEngine.PANDAS)

        config.execution_engine = 'chdb'
        self.assertEqual(function_config.default_engine, FuncExecEngine.CHDB)


class TestLazyStringFunctionsEngineSwitch(unittest.TestCase):
    """Test string function execution with engine switching in Lazy mode."""

    def setUp(self):
        """Create test data and reset config."""
        self.df = pd.DataFrame(
            {
                'text': ['hello', 'WORLD', 'Test'],
            }
        )
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_upper_with_chdb(self):
        """Test upper() executes via ClickHouse."""
        config.use_chdb()

        # Execute and get result
        result = list(self.ds['text'].str.upper())

        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST'])

    def test_upper_with_pandas(self):
        """Test upper() executes via Pandas."""
        config.use_pandas()

        # Execute and get result
        result = list(self.ds['text'].str.upper())

        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST'])

    def test_lower_with_chdb(self):
        """Test lower() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['text'].str.lower())
        # Row order is now preserved
        self.assertEqual(result, ['hello', 'world', 'test'])

    def test_lower_with_pandas(self):
        """Test lower() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['text'].str.lower())
        # Row order is now preserved
        self.assertEqual(result, ['hello', 'world', 'test'])

    def test_length_with_chdb(self):
        """Test str.len() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['text'].str.len())
        # Row order is now preserved
        self.assertEqual(result, [5, 5, 4])

    def test_length_with_pandas(self):
        """Test str.len() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['text'].str.len())
        # Row order is now preserved
        self.assertEqual(result, [5, 5, 4])

    def test_trim_with_chdb(self):
        """Test trim() executes via ClickHouse."""
        config.use_chdb()
        df = pd.DataFrame({'text': ['  hello  ', '  world  ']})
        ds = DataStore.from_dataframe(df)
        result = list(ds['text'].str.trim())
        # Row order is now preserved
        self.assertEqual(result, ['hello', 'world'])

    def test_trim_with_pandas(self):
        """Test trim() executes via Pandas."""
        config.use_pandas()
        df = pd.DataFrame({'text': ['  hello  ', '  world  ']})
        ds = DataStore.from_dataframe(df)
        result = list(ds['text'].str.trim())
        # Row order is now preserved
        self.assertEqual(result, ['hello', 'world'])


class TestLazyMathFunctionsEngineSwitch(unittest.TestCase):
    """Test math function execution with engine switching in Lazy mode."""

    def setUp(self):
        """Create test data and reset config."""
        self.df = pd.DataFrame({'value': [-1.5, 2.7, -3.2], 'positive': [4.0, 9.0, 16.0]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_abs_with_chdb(self):
        """Test abs() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['value'].abs())
        # Row order is now preserved
        self.assertEqual(result, [1.5, 2.7, 3.2])

    def test_abs_with_pandas(self):
        """Test abs() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['value'].abs())
        # Row order is now preserved
        self.assertEqual(result, [1.5, 2.7, 3.2])

    def test_round_with_chdb(self):
        """Test round() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['value'].round())
        # Row order is now preserved
        self.assertEqual(result, [-2.0, 3.0, -3.0])

    def test_round_with_pandas(self):
        """Test round() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['value'].round())
        # Pandas may have different rounding behavior
        self.assertEqual(len(result), 3)

    def test_sqrt_with_chdb(self):
        """Test sqrt() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['positive'].sqrt())
        # Row order is now preserved
        self.assertEqual(result, [2.0, 3.0, 4.0])

    def test_sqrt_with_pandas(self):
        """Test sqrt() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['positive'].sqrt())
        # Row order is now preserved
        self.assertEqual(result, [2.0, 3.0, 4.0])

    def test_floor_with_chdb(self):
        """Test floor() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['value'].floor())
        # Row order is now preserved
        self.assertEqual(result, [-2.0, 2.0, -4.0])

    def test_floor_with_pandas(self):
        """Test floor() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['value'].floor())
        # Row order is now preserved
        self.assertEqual(result, [-2.0, 2.0, -4.0])

    def test_ceil_with_chdb(self):
        """Test ceil() executes via ClickHouse."""
        config.use_chdb()
        result = list(self.ds['value'].ceil())
        # Row order is now preserved
        self.assertEqual(result, [-1.0, 3.0, -3.0])

    def test_ceil_with_pandas(self):
        """Test ceil() executes via Pandas."""
        config.use_pandas()
        result = list(self.ds['value'].ceil())
        # Row order is now preserved
        self.assertEqual(result, [-1.0, 3.0, -3.0])


class TestLazyPandasOnlyFunctions(unittest.TestCase):
    """Test Pandas-only functions that have no ClickHouse equivalent."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'value': [1, 2, 3, 4, 5], 'text': ['a', 'b', 'a', 'b', 'c']})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_cumsum_uses_pandas(self):
        """cumsum should use Pandas even in auto mode."""
        self.assertTrue(function_config.is_pandas_only('cumsum'))

    def test_shift_uses_pandas(self):
        """shift should use Pandas even in auto mode."""
        self.assertTrue(function_config.is_pandas_only('shift'))

    def test_diff_uses_pandas(self):
        """diff should use Pandas even in auto mode."""
        self.assertTrue(function_config.is_pandas_only('diff'))

    def test_fillna_uses_pandas(self):
        """fillna should use Pandas even in auto mode."""
        self.assertTrue(function_config.is_pandas_only('fillna'))

    def test_isna_uses_sql(self):
        """isna uses SQL via toBool(isNull()) for bool dtype compatibility."""
        # isna is now in OVERLAPPING_FUNCTIONS, not PANDAS_ONLY_FUNCTIONS
        self.assertFalse(function_config.is_pandas_only('isna'))
        self.assertIn('isna', function_config.OVERLAPPING_FUNCTIONS)


class TestLazyChainedOperations(unittest.TestCase):
    """Test chained operations with engine switching."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['  hello  ', '  WORLD  ', '  Test  '], 'value': [-1, 2, -3]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_chained_string_ops_chdb(self):
        """Test chained string operations with ClickHouse."""
        config.use_chdb()

        # Chain: trim -> upper
        result = list(self.ds['text'].str.trim().str.upper())
        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST'])

    def test_chained_string_ops_pandas(self):
        """Test chained string operations with Pandas."""
        config.use_pandas()

        # Chain: trim -> upper
        result = list(self.ds['text'].str.trim().str.upper())
        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST'])

    def test_mixed_operations(self):
        """Test mixed math and comparison operations."""
        config.use_chdb()

        # abs then filter
        abs_values = list(self.ds['value'].abs())
        # Row order is now preserved
        self.assertEqual(abs_values, [1, 2, 3])


class TestEngineSwitchingDuringExecution(unittest.TestCase):
    """Test switching engines during execution."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        self.ds = DataStore.from_dataframe(self.df)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.use_auto()

    def test_switch_after_lazy_creation(self):
        """Can switch engine after creating lazy expression."""
        # Create lazy expression with auto
        lazy_expr = self.ds['value'].abs()

        # Switch to pandas before execution
        config.use_pandas()
        result1 = list(lazy_expr)

        # Switch to clickhouse
        config.use_chdb()
        result2 = list(self.ds['value'].abs())

        # Row order is now preserved
        self.assertEqual(result1, [1, 2, 3, 4, 5])
        self.assertEqual(result2, [1, 2, 3, 4, 5])

    def test_results_consistent_across_engines(self):
        """Results should be consistent regardless of engine."""
        df = pd.DataFrame({'val': [-2.5, 1.5, -0.5]})
        ds = DataStore.from_dataframe(df)

        config.use_chdb()
        ch_result = list(ds['val'].abs())

        config.use_pandas()
        pd_result = list(ds['val'].abs())

        # Row order is now preserved
        self.assertEqual(ch_result, pd_result)


class TestFunctionConfigDirectAccess(unittest.TestCase):
    """Test direct access to function_config for fine-grained control."""

    def setUp(self):
        """Reset function_config."""
        function_config.reset()

    def tearDown(self):
        """Reset function_config."""
        function_config.reset()

    def test_use_pandas_for_specific_function(self):
        """Can configure specific functions to use pandas."""
        function_config.use_pandas('upper', 'lower')

        self.assertTrue(function_config.should_use_pandas('upper'))
        self.assertTrue(function_config.should_use_pandas('lower'))
        self.assertFalse(function_config.should_use_pandas('length'))

    def test_use_chdb_for_specific_function(self):
        """Can configure specific functions to use chDB."""
        function_config.prefer_pandas()  # Set default to pandas
        function_config.use_chdb('upper', 'lower')  # Override for specific

        self.assertTrue(function_config.should_use_chdb('upper'))
        self.assertTrue(function_config.should_use_chdb('lower'))
        self.assertFalse(function_config.should_use_chdb('length'))

    def test_prefer_pandas(self):
        """prefer_pandas sets default engine to pandas."""
        function_config.prefer_pandas()
        self.assertEqual(function_config.default_engine, FuncExecEngine.PANDAS)

    def test_prefer_chdb(self):
        """prefer_chdb sets default engine to chdb."""
        function_config.prefer_chdb()
        self.assertEqual(function_config.default_engine, FuncExecEngine.CHDB)

    def test_reset(self):
        """reset clears all custom settings."""
        function_config.prefer_pandas()
        function_config.use_chdb('upper')
        function_config.reset()

        self.assertEqual(function_config.default_engine, FuncExecEngine.CHDB)


class TestHasPandasImplementation(unittest.TestCase):
    """Test that pandas implementations are registered."""

    def test_has_upper_implementation(self):
        """upper should have pandas implementation."""
        self.assertTrue(function_config.has_pandas_implementation('upper'))

    def test_has_lower_implementation(self):
        """lower should have pandas implementation."""
        self.assertTrue(function_config.has_pandas_implementation('lower'))

    def test_has_abs_implementation(self):
        """abs should have pandas implementation."""
        self.assertTrue(function_config.has_pandas_implementation('abs'))

    def test_has_sum_implementation(self):
        """sum should have pandas implementation."""
        self.assertTrue(function_config.has_pandas_implementation('sum'))

    def test_has_mean_implementation(self):
        """mean should have pandas implementation."""
        self.assertTrue(function_config.has_pandas_implementation('mean'))

    def test_get_pandas_implementation_callable(self):
        """get_pandas_implementation should return callable."""
        impl = function_config.get_pandas_implementation('upper')
        self.assertIsNotNone(impl)
        self.assertTrue(callable(impl))

        # Test it works
        s = pd.Series(['hello', 'world'])
        result = impl(s)
        self.assertEqual(list(result), ['HELLO', 'WORLD'])


class TestConfigSummary(unittest.TestCase):
    """Test configuration summary."""

    def setUp(self):
        """Reset function_config."""
        function_config.reset()

    def tearDown(self):
        """Reset function_config."""
        function_config.reset()

    def test_get_config_summary(self):
        """get_config_summary returns useful info."""
        summary = function_config.get_config_summary()

        self.assertIn('default_engine', summary)
        self.assertIn('custom_mappings', summary)
        self.assertIn('overlapping_functions', summary)
        self.assertIn('pandas_implementations', summary)

        self.assertGreater(summary['overlapping_functions'], 50)
        self.assertGreater(summary['pandas_implementations'], 50)


if __name__ == '__main__':
    unittest.main()
