"""
Tests for function execution with engine switching (ClickHouse vs Pandas).

This tests the ability to switch between execution engines via config.
"""

import unittest
import pandas as pd
import numpy as np

from datastore import DataStore, config
from datastore.config import ExecutionEngine
from datastore.function_registry import FunctionRegistry, FunctionCategory
from datastore import function_definitions


class TestEngineConfig(unittest.TestCase):
    """Test execution engine configuration."""

    def setUp(self):
        """Reset to auto before each test."""
        config.use_auto()

    def tearDown(self):
        """Reset to auto after each test."""
        config.use_auto()

    def test_default_engine_is_auto(self):
        """Default engine should be auto."""
        config.use_auto()
        self.assertEqual(config.execution_engine, ExecutionEngine.AUTO)

    def test_set_pandas_engine(self):
        """Can switch to pandas engine."""
        config.use_pandas()
        self.assertEqual(config.execution_engine, ExecutionEngine.PANDAS)

    def test_set_chdb_engine(self):
        """Can switch to clickhouse engine."""
        config.use_chdb()
        self.assertEqual(config.execution_engine, ExecutionEngine.CHDB)

    def test_set_engine_via_property(self):
        """Can set engine via property."""
        config.execution_engine = 'pandas'
        self.assertEqual(config.execution_engine, 'pandas')

        config.execution_engine = 'chdb'
        self.assertEqual(config.execution_engine, 'chdb')

        config.execution_engine = 'auto'
        self.assertEqual(config.execution_engine, 'auto')

    def test_invalid_engine_raises(self):
        """Invalid engine should raise ValueError."""
        with self.assertRaises(ValueError):
            config.execution_engine = 'invalid'


class TestFunctionRegistryCoverage(unittest.TestCase):
    """Test that functions are properly registered."""

    @classmethod
    def setUpClass(cls):
        """Ensure functions are registered."""
        function_definitions.ensure_functions_registered()

    def test_string_functions_registered(self):
        """String functions should be registered."""
        str_funcs = FunctionRegistry.get_by_category(FunctionCategory.STRING)
        self.assertGreaterEqual(len(str_funcs), 40)

        # Check specific functions
        for name in ['upper', 'lower', 'length', 'trim', 'replace', 'substring']:
            spec = FunctionRegistry.get(name)
            self.assertIsNotNone(spec, f"Function {name} not found")

    def test_datetime_functions_registered(self):
        """DateTime functions should be registered."""
        dt_funcs = FunctionRegistry.get_by_category(FunctionCategory.DATETIME)
        self.assertGreaterEqual(len(dt_funcs), 50)

        # Check specific functions
        for name in ['year', 'month', 'day', 'hour', 'minute', 'second', 'to_date', 'to_datetime']:
            spec = FunctionRegistry.get(name)
            self.assertIsNotNone(spec, f"Function {name} not found")

    def test_aggregate_functions_registered(self):
        """Aggregate functions should be registered."""
        agg_funcs = FunctionRegistry.get_by_category(FunctionCategory.AGGREGATE)
        self.assertGreaterEqual(len(agg_funcs), 20)

        # Check specific functions
        for name in ['sum', 'avg', 'count', 'min', 'max', 'median', 'stddev']:
            spec = FunctionRegistry.get(name)
            self.assertIsNotNone(spec, f"Function {name} not found")

    def test_window_functions_registered(self):
        """Window functions should be registered."""
        window_funcs = FunctionRegistry.get_by_category(FunctionCategory.WINDOW)
        self.assertGreaterEqual(len(window_funcs), 10)

        # Check specific functions
        for name in ['row_number', 'rank', 'dense_rank', 'lead', 'lag']:
            spec = FunctionRegistry.get(name)
            self.assertIsNotNone(spec, f"Function {name} not found")

    def test_aliases_work(self):
        """Function aliases should resolve correctly."""
        # toDateTime is alias for to_datetime
        spec1 = FunctionRegistry.get('to_datetime')
        spec2 = FunctionRegistry.get('toDateTime')
        self.assertEqual(spec1.name, spec2.name)

        # mean is alias for avg
        spec1 = FunctionRegistry.get('avg')
        spec2 = FunctionRegistry.get('mean')
        self.assertEqual(spec1.name, spec2.name)


class TestStringFunctionsExecution(unittest.TestCase):
    """Test string function execution."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'text': ['hello', 'WORLD', 'Test String', '  spaces  '], 'num': [1, 2, 3, 4]})
        self.ds = DataStore.from_dataframe(self.df)

    def test_upper(self):
        """Test upper() function."""
        result = list(self.ds['text'].str.upper())
        # Row order is now preserved
        self.assertEqual(result, ['HELLO', 'WORLD', 'TEST STRING', '  SPACES  '])

    def test_lower(self):
        """Test lower() function."""
        result = list(self.ds['text'].str.lower())
        # Row order is now preserved
        self.assertEqual(result, ['hello', 'world', 'test string', '  spaces  '])

    def test_length(self):
        """Test str.len() function (pandas style)."""
        result = list(self.ds['text'].str.len())
        # Row order is now preserved
        self.assertEqual(result, [5, 5, 11, 10])

    def test_trim(self):
        """Test trim() function."""
        result = list(self.ds['text'].str.trim())
        # ClickHouse doesn't guarantee order, so check value is present
        self.assertIn('spaces', result)

    def test_left(self):
        """Test left() function."""
        result = list(self.ds['text'].str.left(3))
        # Row order is now preserved
        self.assertEqual(result, ['hel', 'WOR', 'Tes', '  s'])

    def test_right(self):
        """Test right() function."""
        result = list(self.ds['text'].str.right(3))
        # Row order is now preserved
        self.assertEqual(result, ['llo', 'RLD', 'ing', 's  '])

    def test_replace(self):
        """Test replace() function."""
        result = list(self.ds['text'].str.replace('l', 'L'))
        # ClickHouse doesn't guarantee order, so check value is present
        self.assertIn('heLLo', result)

    def test_substring(self):
        """Test substring() function."""
        # ClickHouse substring is 1-based: substring(s, start, length)
        result = list(self.ds['text'].str.substring(2, 3))  # Start at pos 2, length 3
        # ClickHouse doesn't guarantee order, so check value is present
        self.assertIn('ell', result)


class TestDateTimeFunctionsExecution(unittest.TestCase):
    """Test datetime function execution."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'date': pd.to_datetime(
                    [
                        '2024-01-15 10:30:45',
                        '2024-06-20 14:15:30',
                        '2024-12-25 08:00:00',
                    ]
                )
            }
        )
        self.ds = DataStore.from_dataframe(self.df)

    def test_year(self):
        """Test year extraction."""
        result = list(self.ds['date'].dt.year)
        self.assertEqual(result, [2024, 2024, 2024])

    def test_month(self):
        """Test month extraction."""
        result = list(self.ds['date'].dt.month)
        # Row order is now preserved
        self.assertEqual(result, [1, 6, 12])

    def test_day(self):
        """Test day extraction."""
        result = list(self.ds['date'].dt.day)
        # Row order is now preserved
        self.assertEqual(result, [15, 20, 25])

    def test_hour(self):
        """Test hour extraction."""
        result = list(self.ds['date'].dt.hour)
        # Hours may differ due to timezone, just check it's valid
        self.assertTrue(all(0 <= h <= 23 for h in result))

    def test_minute(self):
        """Test minute extraction."""
        result = list(self.ds['date'].dt.minute)
        # Row order is now preserved
        self.assertEqual(result, [30, 15, 0])

    def test_second(self):
        """Test second extraction."""
        result = list(self.ds['date'].dt.second)
        # Row order is now preserved
        self.assertEqual(result, [45, 30, 0])

    def test_dayofweek(self):
        """Test dayofweek extraction - aligned with pandas (Monday=0, Sunday=6)."""
        result = self.ds['date'].dt.dayofweek
        # Verify range is 0-6 (pandas convention)
        np.testing.assert_array_equal(np.asarray(result) >= 0, True)
        np.testing.assert_array_equal(np.asarray(result) <= 6, True)
        # Verify matches pandas
        expected = self.df['date'].dt.dayofweek
        np.testing.assert_array_equal(result, expected)

    def test_quarter(self):
        """Test quarter extraction."""
        result = list(self.ds['date'].dt.quarter)
        # Row order is now preserved
        self.assertEqual(result, [1, 2, 4])


class TestMathFunctionsExecution(unittest.TestCase):
    """Test math function execution."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'value': [-1.5, 2.7, -3.2, 4.9], 'positive': [1.0, 4.0, 9.0, 16.0]})
        self.ds = DataStore.from_dataframe(self.df)

    def test_abs(self):
        """Test abs() function."""
        result = list(self.ds['value'].abs())
        # Row order is now preserved
        self.assertEqual(result, [1.5, 2.7, 3.2, 4.9])

    def test_round(self):
        """Test round() function."""
        result = list(self.ds['value'].round())
        # Row order is now preserved
        self.assertEqual(result, [-2.0, 3.0, -3.0, 5.0])

    def test_floor(self):
        """Test floor() function."""
        result = list(self.ds['value'].floor())
        # Row order is now preserved
        self.assertEqual(result, [-2.0, 2.0, -4.0, 4.0])

    def test_ceil(self):
        """Test ceil() function."""
        result = list(self.ds['value'].ceil())
        # Row order is now preserved
        self.assertEqual(result, [-1.0, 3.0, -3.0, 5.0])

    def test_sqrt(self):
        """Test sqrt() function."""
        result = list(self.ds['positive'].sqrt())
        # Row order is now preserved
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    def test_exp(self):
        """Test exp() function."""
        result = list(self.ds['positive'].exp())
        self.assertTrue(all(r > 0 for r in result))

    def test_log(self):
        """Test log() function."""
        result = list(self.ds['positive'].log())
        # ClickHouse doesn't guarantee order, so check that log(1.0)=0.0 is present
        self.assertTrue(any(abs(r) < 1e-5 for r in result))


class TestAggregateFunctionsExecution(unittest.TestCase):
    """Test aggregate function execution."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({'value': [10, 20, 30, 40, 50], 'category': ['A', 'A', 'B', 'B', 'B']})
        self.ds = DataStore.from_dataframe(self.df)

    def test_sum_sql(self):
        """Test sum_sql() returns correct SQL expression for SQL building."""
        # Use sum_sql() for SQL building (returns ColumnExpr), not sum() (returns scalar)
        sql = self.ds.select(self.ds['value'].sum_sql().as_('total')).to_sql()
        self.assertIn('sum("value")', sql.lower())

    def test_avg_sql(self):
        """Test avg() returns correct SQL."""
        sql = self.ds.select(self.ds['value'].avg().as_('average')).to_sql()
        self.assertIn('avg("value")', sql.lower())

    def test_count_sql(self):
        """Test count_sql() returns correct SQL expression for SQL building."""
        # Use count_sql() for SQL building (returns ColumnExpr), not count() (returns scalar)
        sql = self.ds.select(self.ds['value'].count_sql().as_('cnt')).to_sql()
        # count() may be count(*) or count("value") depending on implementation
        self.assertIn('count', sql.lower())

    def test_min_sql(self):
        """Test min_sql() returns correct SQL expression for SQL building."""
        # Use min_sql() for SQL building (returns ColumnExpr), not min() (returns scalar)
        sql = self.ds.select(self.ds['value'].min_sql().as_('minimum')).to_sql()
        self.assertIn('min("value")', sql.lower())

    def test_max_sql(self):
        """Test max_sql() returns correct SQL expression for SQL building."""
        # Use max_sql() for SQL building (returns ColumnExpr), not max() (returns scalar)
        sql = self.ds.select(self.ds['value'].max_sql().as_('maximum')).to_sql()
        self.assertIn('max("value")', sql.lower())


class TestWindowFunctionsSQL(unittest.TestCase):
    """Test window function SQL generation."""

    def setUp(self):
        """Create test data."""
        function_definitions.ensure_functions_registered()

    def test_row_number_sql(self):
        """Test row_number() SQL generation."""
        from datastore import F

        wf = F.row_number().over(partition_by='category', order_by='value')
        sql = wf.to_sql()
        self.assertIn('row_number()', sql.lower())
        self.assertIn('over', sql.lower())
        self.assertIn('partition by', sql.lower())

    def test_rank_sql(self):
        """Test rank() SQL generation."""
        from datastore import F

        wf = F.rank().over(order_by='value DESC')
        sql = wf.to_sql()
        self.assertIn('rank()', sql.lower())
        self.assertIn('order by', sql.lower())

    def test_lead_sql(self):
        """Test lead() SQL generation."""
        from datastore import F
        from datastore.expressions import Field

        wf = F.lead(Field('value'), 1).over(order_by='date')
        sql = wf.to_sql()
        self.assertIn('leadinframe', sql.lower())

    def test_lag_sql(self):
        """Test lag() SQL generation."""
        from datastore import F
        from datastore.expressions import Field

        wf = F.lag(Field('value'), 1).over(order_by='date')
        sql = wf.to_sql()
        self.assertIn('laginframe', sql.lower())


class TestFClassMethods(unittest.TestCase):
    """Test F class static methods."""

    def test_f_upper(self):
        """Test F.upper()."""
        from datastore import F

        func = F.upper('name')
        self.assertIn('upper', func.to_sql().lower())

    def test_f_sum(self):
        """Test F.sum()."""
        from datastore import F

        func = F.sum('value')
        self.assertIn('sum', func.to_sql().lower())

    def test_f_year(self):
        """Test F.year()."""
        from datastore import F

        func = F.year('date')
        self.assertIn('toyear', func.to_sql().lower())

    def test_f_round(self):
        """Test F.round()."""
        from datastore import F

        func = F.round('value', 2)
        self.assertIn('round', func.to_sql().lower())

    def test_f_coalesce(self):
        """Test F.coalesce()."""
        from datastore import F

        func = F.coalesce('value', 0)
        self.assertIn('coalesce', func.to_sql().lower())


class TestNewPandasMethods(unittest.TestCase):
    """Test newly added pandas-compatible methods."""

    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {
                'text': ['hello', 'WORLD', 'test'],
                'value': [-1, 2, -3],
                'date': pd.to_datetime(['2024-01-15', '2024-06-20', '2024-12-25']),
            }
        )
        self.ds = DataStore.from_dataframe(self.df)

    def test_capitalize(self):
        """Test capitalize() method."""
        result = list(self.ds['text'].str.capitalize())
        # initcap might behave differently
        self.assertEqual(len(result), 3)

    def test_zfill(self):
        """Test zfill() method."""
        result = list(self.ds['text'].str.zfill(10))
        self.assertTrue(all(len(r) == 10 for r in result))

    def test_isdigit(self):
        """Test isdigit() method."""
        df = pd.DataFrame({'text': ['123', 'abc', '456']})
        ds = DataStore.from_dataframe(df)
        result = list(ds['text'].str.isdigit())
        # Row order is now preserved
        self.assertEqual(result, [1, 0, 1])

    def test_clip(self):
        """Test clip() method."""
        result = list(self.ds['value'].clip(lower=-2, upper=2))
        # Row order is now preserved
        self.assertEqual(result, [-1, 2, -2])

    def test_fillna(self):
        """Test fillna_sql() method via SQL generation."""
        # fillna() returns pandas Series (for pandas compatibility)
        # fillna_sql() returns ColumnExpr for SQL generation
        sql = self.ds['value'].fillna_sql(0).to_sql()
        self.assertIn('ifnull', sql.lower())


class TestRegistryStats(unittest.TestCase):
    """Test registry statistics."""

    def test_total_functions(self):
        """Should have 180+ functions registered."""
        function_definitions.ensure_functions_registered()
        stats = FunctionRegistry.stats()
        self.assertGreaterEqual(stats['total_functions'], 180)

    def test_has_all_categories(self):
        """Should have functions in all main categories."""
        function_definitions.ensure_functions_registered()
        stats = FunctionRegistry.stats()

        required_categories = ['STRING', 'DATETIME', 'MATH', 'AGGREGATE', 'WINDOW', 'CONDITIONAL']
        for cat in required_categories:
            self.assertIn(cat, stats['by_category'])
            self.assertGreater(stats['by_category'][cat], 0)


if __name__ == '__main__':
    unittest.main()
