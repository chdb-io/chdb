"""
Comprehensive tests for DataStore CASE WHEN expressions.

Tests the ds.when().when().otherwise() API for:
1. Basic string categorization
2. Numeric transformations
3. Multiple conditions
4. Column expressions as values
5. Null handling
6. Edge cases
7. SQL generation
8. Pandas execution compatibility
"""

import unittest
import numpy as np
import pandas as pd
import datastore as ds


class TestCaseWhenBasic(unittest.TestCase):
    """Basic CASE WHEN functionality tests."""

    def test_simple_grade_classification(self):
        """Test simple grade classification with string values."""
        pdf = pd.DataFrame({'score': [95, 85, 75, 65, 55]})
        store = ds.DataStore.from_df(pdf)

        # DataStore case when
        store['grade'] = (
            store.when(store['score'] >= 90, 'A')
            .when(store['score'] >= 80, 'B')
            .when(store['score'] >= 70, 'C')
            .when(store['score'] >= 60, 'D')
            .otherwise('F')
        )

        result = store.to_df()

        # Expected: A, B, C, D, F
        expected = ['A', 'B', 'C', 'D', 'F']
        self.assertEqual(list(result['grade']), expected)

    def test_numeric_classification(self):
        """Test classification with numeric values."""
        pdf = pd.DataFrame({'value': [150, 80, 50, 20, -10]})
        store = ds.DataStore.from_df(pdf)

        store['category'] = (
            store.when(store['value'] >= 100, 3).when(store['value'] >= 50, 2).when(store['value'] >= 0, 1).otherwise(0)
        )

        result = store.to_df()

        # Expected: 150>=100->3, 80>=50->2, 50>=50->2, 20>=0->1, -10<0->0
        expected = [3, 2, 2, 1, 0]
        self.assertEqual(list(result['category']), expected)

    def test_binary_classification(self):
        """Test simple binary (if-else) classification."""
        pdf = pd.DataFrame({'age': [15, 18, 25, 65, 70]})
        store = ds.DataStore.from_df(pdf)

        store['is_adult'] = store.when(store['age'] >= 18, 'Yes').otherwise('No')

        result = store.to_df()

        expected = ['No', 'Yes', 'Yes', 'Yes', 'Yes']
        self.assertEqual(list(result['is_adult']), expected)

    def test_pandas_comparison(self):
        """Test that DataStore case when matches pandas np.select."""
        pdf = pd.DataFrame({'score': [95, 85, 75, 65, 55]})
        store = ds.DataStore.from_df(pdf)

        # Pandas with np.select
        conditions = [
            pdf['score'] >= 90,
            pdf['score'] >= 80,
            pdf['score'] >= 70,
            pdf['score'] >= 60,
        ]
        choices = ['A', 'B', 'C', 'D']
        pdf['grade'] = np.select(conditions, choices, default='F')

        # DataStore case when
        store['grade'] = (
            store.when(store['score'] >= 90, 'A')
            .when(store['score'] >= 80, 'B')
            .when(store['score'] >= 70, 'C')
            .when(store['score'] >= 60, 'D')
            .otherwise('F')
        )

        ds_result = store.to_df()

        # Compare
        np.testing.assert_array_equal(ds_result['grade'].values, pdf['grade'].values)


class TestCaseWhenWithExpressions(unittest.TestCase):
    """Tests for CASE WHEN with column expressions as values."""

    def test_column_as_value(self):
        """Test using another column as the THEN value."""
        pdf = pd.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3], 'c': [100, 200, 300]})
        store = ds.DataStore.from_df(pdf)

        # Use column expression as value
        store['result'] = store.when(store['a'] >= 20, store['c']).otherwise(store['b'])

        result = store.to_df()

        # a >= 20: [False, True, True] -> result: [b[0], c[1], c[2]] = [1, 200, 300]
        expected = [1, 200, 300]
        self.assertEqual(list(result['result']), expected)

    def test_expression_as_value(self):
        """Test using arithmetic expression as the THEN value."""
        pdf = pd.DataFrame({'value': [10, 50, 100]})
        store = ds.DataStore.from_df(pdf)

        # Double large values, halve small values
        store['adjusted'] = store.when(store['value'] >= 50, store['value'] * 2).otherwise(store['value'] / 2)

        result = store.to_df()

        # value >= 50: [False, True, True] -> [10/2, 50*2, 100*2] = [5, 100, 200]
        expected = [5.0, 100.0, 200.0]
        self.assertEqual(list(result['adjusted']), expected)


class TestCaseWhenMultipleConditions(unittest.TestCase):
    """Tests for complex conditions in CASE WHEN."""

    def test_compound_conditions(self):
        """Test CASE WHEN with compound (AND/OR) conditions."""
        pdf = pd.DataFrame({'age': [25, 35, 45, 55, 65], 'income': [30000, 50000, 70000, 90000, 40000]})
        store = ds.DataStore.from_df(pdf)

        # Complex categorization
        store['segment'] = (
            store.when((store['age'] >= 60) | (store['income'] >= 80000), 'Premium')
            .when((store['age'] >= 40) & (store['income'] >= 50000), 'Standard')
            .otherwise('Basic')
        )

        result = store.to_df()

        # Row 0: age=25, income=30000 -> Basic
        # Row 1: age=35, income=50000 -> Basic (age < 40)
        # Row 2: age=45, income=70000 -> Standard
        # Row 3: age=55, income=90000 -> Premium (income >= 80000)
        # Row 4: age=65, income=40000 -> Premium (age >= 60)
        expected = ['Basic', 'Basic', 'Standard', 'Premium', 'Premium']
        self.assertEqual(list(result['segment']), expected)

    def test_multiple_columns_in_condition(self):
        """Test conditions involving multiple columns."""
        pdf = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]})
        store = ds.DataStore.from_df(pdf)

        store['comparison'] = (
            store.when(store['x'] > store['y'], 'x_larger').when(store['x'] < store['y'], 'y_larger').otherwise('equal')
        )

        result = store.to_df()

        expected = ['y_larger', 'y_larger', 'equal', 'x_larger', 'x_larger']
        self.assertEqual(list(result['comparison']), expected)


class TestCaseWhenNullHandling(unittest.TestCase):
    """Tests for NULL/NaN handling in CASE WHEN."""

    def test_null_values_in_condition(self):
        """Test CASE WHEN with NULL values in the condition column."""
        pdf = pd.DataFrame({'value': [10, np.nan, 30, np.nan, 50]})
        store = ds.DataStore.from_df(pdf)

        store['category'] = store.when(store['value'] >= 30, 'high').otherwise('low')

        result = store.to_df()

        # NaN comparisons are False, so they get 'low'
        expected = ['low', 'low', 'high', 'low', 'high']
        self.assertEqual(list(result['category']), expected)

    def test_null_as_otherwise_value(self):
        """Test using None/NULL as the otherwise value."""
        pdf = pd.DataFrame({'status': ['active', 'inactive', 'active', 'pending']})
        store = ds.DataStore.from_df(pdf)

        store['is_active'] = store.when(store['status'] == 'active', True).otherwise(None)

        result = store.to_df()

        # Expected: True, None, True, None
        self.assertEqual(result['is_active'].iloc[0], True)
        self.assertTrue(pd.isna(result['is_active'].iloc[1]))
        self.assertEqual(result['is_active'].iloc[2], True)
        self.assertTrue(pd.isna(result['is_active'].iloc[3]))


class TestCaseWhenEdgeCases(unittest.TestCase):
    """Edge cases and error handling tests."""

    def test_single_when(self):
        """Test CASE WHEN with only one condition."""
        pdf = pd.DataFrame({'flag': [True, False, True]})
        store = ds.DataStore.from_df(pdf)

        store['result'] = store.when(store['flag'] == True, 'yes').otherwise('no')  # noqa: E712

        result = store.to_df()
        expected = ['yes', 'no', 'yes']
        self.assertEqual(list(result['result']), expected)

    def test_many_when_conditions(self):
        """Test CASE WHEN with many conditions."""
        pdf = pd.DataFrame({'month': list(range(1, 13))})
        store = ds.DataStore.from_df(pdf)

        store['season'] = (
            store.when(store['month'].isin([12, 1, 2]), 'Winter')
            .when(store['month'].isin([3, 4, 5]), 'Spring')
            .when(store['month'].isin([6, 7, 8]), 'Summer')
            .when(store['month'].isin([9, 10, 11]), 'Fall')
            .otherwise('Unknown')
        )

        result = store.to_df()

        expected = [
            'Winter',
            'Winter',
            'Spring',
            'Spring',
            'Spring',
            'Summer',
            'Summer',
            'Summer',
            'Fall',
            'Fall',
            'Fall',
            'Winter',
        ]
        self.assertEqual(list(result['season']), expected)

    def test_otherwise_completes_expression(self):
        """Test that otherwise() returns a usable expression."""
        pdf = pd.DataFrame({'value': [1, 2, 3]})
        store = ds.DataStore.from_df(pdf)

        # Builder without otherwise() is not usable
        builder = store.when(store['value'] > 1, 'big')
        self.assertIsInstance(builder, ds.CaseWhenBuilder)

        # After otherwise(), it becomes a CaseWhenExpr
        expr = builder.otherwise('small')
        self.assertIsInstance(expr, ds.CaseWhenExpr)

        # CaseWhenExpr can be assigned to column
        store['result'] = expr
        result = store.to_df()

        expected = ['small', 'big', 'big']
        self.assertEqual(list(result['result']), expected)

    def test_empty_dataframe(self):
        """Test CASE WHEN on empty DataFrame."""
        pdf = pd.DataFrame({'value': pd.Series([], dtype=float)})
        store = ds.DataStore.from_df(pdf)

        store['category'] = store.when(store['value'] > 0, 'positive').otherwise('non-positive')

        result = store.to_df()

        self.assertEqual(len(result), 0)
        self.assertIn('category', result.columns)


class TestCaseWhenSQL(unittest.TestCase):
    """Tests for SQL generation from CASE WHEN."""

    def test_to_sql_simple(self):
        """Test SQL generation for simple CASE WHEN."""
        pdf = pd.DataFrame({'score': [90, 80, 70]})
        store = ds.DataStore.from_df(pdf)

        expr = store.when(store['score'] >= 90, 'A').when(store['score'] >= 80, 'B').otherwise('C')

        sql = expr.to_sql()

        # Should contain CASE WHEN structure
        self.assertIn('CASE', sql)
        self.assertIn('WHEN', sql)
        self.assertIn('THEN', sql)
        self.assertIn('ELSE', sql)
        self.assertIn('END', sql)
        self.assertIn("'A'", sql)
        self.assertIn("'B'", sql)
        self.assertIn("'C'", sql)

    def test_to_sql_with_numeric_values(self):
        """Test SQL generation with numeric THEN values."""
        pdf = pd.DataFrame({'value': [100, 50, 0]})
        store = ds.DataStore.from_df(pdf)

        expr = store.when(store['value'] >= 50, 100).when(store['value'] >= 0, 50).otherwise(0)

        sql = expr.to_sql()

        # Should contain numeric values without quotes
        self.assertIn('100', sql)
        self.assertIn('50', sql)
        self.assertIn('0', sql)


class TestCaseWhenIntegration(unittest.TestCase):
    """Integration tests with other DataStore operations."""

    def test_case_when_with_filter(self):
        """Test CASE WHEN after filter operation."""
        pdf = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David'], 'age': [25, 35, 45, 55]})
        store = ds.DataStore.from_df(pdf)

        # Filter first, then apply case when
        store = store.filter(store['age'] >= 30)
        store['age_group'] = (
            store.when(store['age'] >= 50, 'senior').when(store['age'] >= 40, 'middle').otherwise('young')
        )

        result = store.to_df()

        self.assertEqual(len(result), 3)  # Bob, Charlie, David
        self.assertEqual(list(result['age_group']), ['young', 'middle', 'senior'])

    def test_case_when_with_groupby(self):
        """Test CASE WHEN result used in groupby."""
        pdf = pd.DataFrame({'score': [95, 85, 92, 78, 88], 'subject': ['Math', 'Math', 'Science', 'Science', 'Math']})
        store = ds.DataStore.from_df(pdf)

        # Add grade classification
        store['grade'] = store.when(store['score'] >= 90, 'A').when(store['score'] >= 80, 'B').otherwise('C')

        # Now group by grade
        result = store.to_df()

        # Count grades
        grade_counts = result['grade'].value_counts()
        self.assertEqual(grade_counts['A'], 2)  # 95, 92
        self.assertEqual(grade_counts['B'], 2)  # 85, 88
        self.assertEqual(grade_counts['C'], 1)  # 78

    def test_case_when_preserves_index(self):
        """Test that CASE WHEN preserves DataFrame index."""
        pdf = pd.DataFrame({'value': [10, 20, 30]}, index=['a', 'b', 'c'])
        store = ds.DataStore.from_df(pdf)

        store['category'] = store.when(store['value'] >= 20, 'high').otherwise('low')

        result = store.to_df()

        self.assertEqual(list(result.index), ['a', 'b', 'c'])


class TestCaseWhenTypes(unittest.TestCase):
    """Tests for different data types in CASE WHEN."""

    def test_boolean_values(self):
        """Test CASE WHEN with boolean result values."""
        pdf = pd.DataFrame({'value': [10, 20, 30]})
        store = ds.DataStore.from_df(pdf)

        store['is_large'] = store.when(store['value'] >= 20, True).otherwise(False)

        result = store.to_df()
        expected = [False, True, True]
        self.assertEqual(list(result['is_large']), expected)

    def test_float_values(self):
        """Test CASE WHEN with float result values."""
        pdf = pd.DataFrame({'category': ['A', 'B', 'C']})
        store = ds.DataStore.from_df(pdf)

        store['multiplier'] = (
            store.when(store['category'] == 'A', 1.5).when(store['category'] == 'B', 1.2).otherwise(1.0)
        )

        result = store.to_df()
        expected = [1.5, 1.2, 1.0]
        self.assertEqual(list(result['multiplier']), expected)


class TestCaseWhenNumpyAlignment(unittest.TestCase):
    """Tests to verify alignment with np.where() and np.select()."""

    def test_simple_binary_matches_np_where(self):
        """Test simple binary condition matches np.where."""
        pdf = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        store = ds.DataStore.from_df(pdf)

        np_result = np.where(pdf['value'] >= 30, 'high', 'low')

        store['category'] = store.when(store['value'] >= 30, 'high').otherwise('low')
        ds_result = store.to_df()['category'].values

        np.testing.assert_array_equal(ds_result, np_result)

    def test_nested_np_where_matches_chained_when(self):
        """Test nested np.where matches chained when()."""
        pdf = pd.DataFrame({'score': [95, 85, 75, 65, 55]})
        store = ds.DataStore.from_df(pdf)

        # Nested np.where
        np_result = np.where(
            pdf['score'] >= 90,
            'A',
            np.where(
                pdf['score'] >= 80, 'B', np.where(pdf['score'] >= 70, 'C', np.where(pdf['score'] >= 60, 'D', 'F'))
            ),
        )

        # Chained when
        store['grade'] = (
            store.when(store['score'] >= 90, 'A')
            .when(store['score'] >= 80, 'B')
            .when(store['score'] >= 70, 'C')
            .when(store['score'] >= 60, 'D')
            .otherwise('F')
        )
        ds_result = store.to_df()['grade'].values

        np.testing.assert_array_equal(ds_result, np_result)

    def test_numeric_result_matches_np_where(self):
        """Test numeric results match np.where."""
        pdf = pd.DataFrame({'x': [-5, 0, 5, 10, 15]})
        store = ds.DataStore.from_df(pdf)

        np_result = np.where(pdf['x'] > 0, pdf['x'] * 2, 0)

        store['doubled'] = store.when(store['x'] > 0, store['x'] * 2).otherwise(0)
        ds_result = store.to_df()['doubled'].values

        np.testing.assert_array_equal(ds_result, np_result)

    def test_column_as_else_matches_np_where(self):
        """Test using column as else value matches np.where."""
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        store = ds.DataStore.from_df(pdf)

        np_result = np.where(pdf['a'] > 3, pdf['b'], pdf['a'])

        store['result'] = store.when(store['a'] > 3, store['b']).otherwise(store['a'])
        ds_result = store.to_df()['result'].values

        np.testing.assert_array_equal(ds_result, np_result)

    def test_nan_handling_matches_np_where(self):
        """Test NaN handling matches np.where behavior."""
        pdf = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})
        store = ds.DataStore.from_df(pdf)

        # NaN comparisons return False in np.where
        np_result = np.where(pdf['value'] > 2, 'big', 'small')

        store['size'] = store.when(store['value'] > 2, 'big').otherwise('small')
        ds_result = store.to_df()['size'].values

        np.testing.assert_array_equal(ds_result, np_result)

    def test_matches_np_select(self):
        """Test that chained when matches np.select."""
        pdf = pd.DataFrame({'temp': [-10, 0, 15, 25, 35]})
        store = ds.DataStore.from_df(pdf)

        conditions = [pdf['temp'] < 0, pdf['temp'] < 20, pdf['temp'] < 30]
        choices = ['freezing', 'cold', 'warm']
        np_result = np.select(conditions, choices, default='hot')

        store['weather'] = (
            store.when(store['temp'] < 0, 'freezing')
            .when(store['temp'] < 20, 'cold')
            .when(store['temp'] < 30, 'warm')
            .otherwise('hot')
        )
        ds_result = store.to_df()['weather'].values

        np.testing.assert_array_equal(ds_result, np_result)


class TestCaseWhenExecutionEngine(unittest.TestCase):
    """Tests for CASE WHEN execution engine configuration."""

    def setUp(self):
        """Reset function_config before each test."""
        from datastore.function_executor import function_config

        function_config.reset()

    def tearDown(self):
        """Reset function_config after each test."""
        from datastore.function_executor import function_config

        function_config.reset()

    def test_default_engine_is_chdb(self):
        """Test that default execution engine for CASE WHEN is chDB."""
        from datastore.function_executor import function_config, ExecutionEngine

        # Default engine should be chDB
        engine = function_config.get_engine('when')
        self.assertEqual(engine, ExecutionEngine.CHDB)
        self.assertFalse(function_config.should_use_pandas('when'))

    def test_expression_reports_chdb_by_default(self):
        """Test that CaseWhenExpr.execution_engine() returns 'chDB' by default."""
        pdf = pd.DataFrame({'score': [90, 80, 70]})
        store = ds.DataStore.from_df(pdf)

        expr = store.when(store['score'] >= 90, 'A').otherwise('B')

        self.assertEqual(expr.execution_engine(), 'chDB')

    def test_config_switch_to_pandas(self):
        """Test switching execution engine to Pandas via function_config.

        Validates via:
        1. explain() output shows [Pandas] for the CASE WHEN operation
        2. Debug log shows 'Using Pandas engine'
        """
        import io
        import sys
        import re
        import logging
        from datastore.function_executor import function_config

        # Switch to Pandas
        function_config.use_pandas('when')
        self.assertTrue(function_config.should_use_pandas('when'))

        # Create expression
        pdf = pd.DataFrame({'score': [90, 80, 70]})
        store = ds.DataStore.from_df(pdf)
        store['grade'] = store.when(store['score'] >= 90, 'A').otherwise('B')

        # Verify via explain() output
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        store.explain()
        sys.stdout = old_stdout
        explain_output = buffer.getvalue()

        # Should show [Pandas] for CASE WHEN assignment
        self.assertRegex(
            explain_output,
            r'\[Pandas\].*Assign column.*grade.*CASE WHEN',
            "explain() should show [Pandas] for CASE WHEN operation",
        )

        # Verify via debug log during execution
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger('datastore')
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            _ = store.to_df()  # Trigger execution
            log_output = log_buffer.getvalue()
            # Should log 'Using Pandas engine (np.select)'
            self.assertRegex(
                log_output,
                r'Using Pandas engine.*np\.select',
                "Debug log should show 'Using Pandas engine (np.select)'",
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    def test_config_switch_back_to_chdb(self):
        """Test switching execution engine back to chDB.

        Validates via:
        1. explain() output shows [chDB] for the CASE WHEN operation
        2. Debug log shows 'Using SQL engine'
        """
        import io
        import sys
        import re
        import logging
        from datastore.function_executor import function_config

        # Switch to Pandas, then back to chDB
        function_config.use_pandas('when')
        function_config.use_chdb('when')

        # Create expression
        pdf = pd.DataFrame({'score': [90, 80, 70]})
        store = ds.DataStore.from_df(pdf)
        store['grade'] = store.when(store['score'] >= 90, 'A').otherwise('B')

        # Verify via explain() output
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        store.explain()
        sys.stdout = old_stdout
        explain_output = buffer.getvalue()

        # Should show [chDB] for CASE WHEN assignment
        self.assertRegex(
            explain_output,
            r'\[chDB\].*Assign column.*grade.*CASE WHEN',
            "explain() should show [chDB] for CASE WHEN operation",
        )

        # Verify via debug log during execution
        log_buffer = io.StringIO()
        handler = logging.StreamHandler(log_buffer)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger('datastore')
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            _ = store.to_df()  # Trigger execution
            log_output = log_buffer.getvalue()
            # Should log 'Using SQL engine (chDB)'
            self.assertRegex(log_output, r'Using SQL engine.*chDB', "Debug log should show 'Using SQL engine (chDB)'")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)

    def test_chdb_and_pandas_produce_same_results(self):
        """Test that chDB and Pandas engines produce identical results."""
        from datastore.function_executor import function_config

        pdf = pd.DataFrame({'score': [95, 85, 75, 65, 55]})

        # Test with chDB (default)
        function_config.use_chdb('when')
        store_chdb = ds.DataStore.from_df(pdf)
        store_chdb['grade'] = (
            store_chdb.when(store_chdb['score'] >= 90, 'A')
            .when(store_chdb['score'] >= 80, 'B')
            .when(store_chdb['score'] >= 70, 'C')
            .otherwise('F')
        )
        result_chdb = store_chdb.to_df()

        # Test with Pandas
        function_config.use_pandas('when')
        store_pandas = ds.DataStore.from_df(pdf)
        store_pandas['grade'] = (
            store_pandas.when(store_pandas['score'] >= 90, 'A')
            .when(store_pandas['score'] >= 80, 'B')
            .when(store_pandas['score'] >= 70, 'C')
            .otherwise('F')
        )
        result_pandas = store_pandas.to_df()

        # Results should be identical
        np.testing.assert_array_equal(result_chdb['grade'].values, result_pandas['grade'].values)

    def test_chdb_and_pandas_numeric_values(self):
        """Test both engines with numeric values."""
        from datastore.function_executor import function_config

        pdf = pd.DataFrame({'value': [100, 50, 25, 0, -10]})

        # chDB
        function_config.use_chdb('when')
        store1 = ds.DataStore.from_df(pdf)
        store1['category'] = (
            store1.when(store1['value'] >= 75, 3)
            .when(store1['value'] >= 25, 2)
            .when(store1['value'] >= 0, 1)
            .otherwise(0)
        )
        result_chdb = store1.to_df()

        # Pandas
        function_config.use_pandas('when')
        store2 = ds.DataStore.from_df(pdf)
        store2['category'] = (
            store2.when(store2['value'] >= 75, 3)
            .when(store2['value'] >= 25, 2)
            .when(store2['value'] >= 0, 1)
            .otherwise(0)
        )
        result_pandas = store2.to_df()

        np.testing.assert_array_equal(result_chdb['category'].values, result_pandas['category'].values)

    def test_lazy_column_assignment_reports_correct_engine(self):
        """Test that LazyColumnAssignment correctly reports the execution engine."""
        from datastore.function_executor import function_config
        from datastore.lazy_ops import LazyColumnAssignment

        pdf = pd.DataFrame({'score': [90, 80]})
        store = ds.DataStore.from_df(pdf)
        expr = store.when(store['score'] >= 90, 'A').otherwise('B')

        # Create LazyColumnAssignment and check engine
        op = LazyColumnAssignment('grade', expr)

        # Default should be chDB
        function_config.use_chdb('when')
        self.assertEqual(op.execution_engine(), 'chDB')

        # After switching config
        function_config.use_pandas('when')
        self.assertEqual(op.execution_engine(), 'Pandas')

    def test_chdb_execution_with_complex_conditions(self):
        """Test chDB execution with complex AND/OR conditions."""
        from datastore.function_executor import function_config

        pdf = pd.DataFrame({'age': [25, 35, 45, 55, 65], 'income': [30000, 50000, 80000, 100000, 60000]})

        # Use chDB
        function_config.use_chdb('when')
        store = ds.DataStore.from_df(pdf)

        # Complex conditions: high income young or any senior
        store['priority'] = (
            store.when((store['age'] < 30) & (store['income'] > 25000), 'young_high_income')
            .when(store['age'] >= 60, 'senior')
            .when(store['income'] >= 75000, 'high_income')
            .otherwise('standard')
        )

        result = store.to_df()

        expected = ['young_high_income', 'standard', 'high_income', 'high_income', 'senior']
        self.assertEqual(list(result['priority']), expected)

    def test_pandas_execution_preserves_index(self):
        """Test that Pandas execution preserves DataFrame index."""
        from datastore.function_executor import function_config

        pdf = pd.DataFrame({'score': [90, 80, 70]}, index=['a', 'b', 'c'])

        function_config.use_pandas('when')
        store = ds.DataStore.from_df(pdf)
        store['grade'] = store.when(store['score'] >= 90, 'A').otherwise('B')
        result = store.to_df()

        # Check index is preserved
        self.assertEqual(list(result.index), ['a', 'b', 'c'])

    def test_chdb_execution_preserves_index(self):
        """Test that chDB execution preserves DataFrame index."""
        from datastore.function_executor import function_config

        pdf = pd.DataFrame({'score': [90, 80, 70]}, index=['x', 'y', 'z'])

        function_config.use_chdb('when')
        store = ds.DataStore.from_df(pdf)
        store['grade'] = store.when(store['score'] >= 90, 'A').otherwise('B')
        result = store.to_df()

        # Check index is preserved
        self.assertEqual(list(result.index), ['x', 'y', 'z'])

    def test_case_when_in_overlapping_functions(self):
        """Test that 'when' is in OVERLAPPING_FUNCTIONS set."""
        from datastore.function_executor import FunctionExecutorConfig

        self.assertIn('when', FunctionExecutorConfig.OVERLAPPING_FUNCTIONS)
        self.assertIn('case_when', FunctionExecutorConfig.OVERLAPPING_FUNCTIONS)


if __name__ == '__main__':
    unittest.main()
