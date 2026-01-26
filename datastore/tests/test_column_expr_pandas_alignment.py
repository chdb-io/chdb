"""
Tests for ColumnExpr pandas alignment.

These tests verify that DataStore's ColumnExpr behaves similarly to pandas
when accessing columns and performing operations.

Key behaviors tested:
- ds["col"] displays actual values like pandas Series
- ds["col"] + 1 returns computed values like pandas
- ds["col"].str.upper() returns string operation results
- ds["col"] > 10 returns Condition (for filtering, unlike pandas boolean Series)
"""

import unittest
import pandas as pd
import numpy as np

from datastore import DataStore, Field, ColumnExpr
from datastore.conditions import BinaryCondition, Condition
from datastore.lazy_ops import LazyDataFrameSource
from tests.test_utils import assert_series_equal


class TestColumnExprPandasAlignment(unittest.TestCase):
    """Test ColumnExpr alignment with pandas behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
                'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
                'department': ['HR', 'Engineering', 'Engineering', 'Management', 'HR'],
                'active': [True, True, False, True, True],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    # ========== Basic Column Access ==========

    def test_column_access_returns_column_expr(self):
        """Test that ds['col'] returns ColumnExpr."""
        ds = self.create_ds()
        result = ds['age']
        self.assertIsInstance(result, ColumnExpr)

    def test_column_access_wraps_field(self):
        """Test that ColumnExpr wraps a Field."""
        ds = self.create_ds()
        result = ds['age']
        self.assertIsInstance(result._expr, Field)
        self.assertEqual(result._expr.name, 'age')

    def test_column_access_executes_correctly(self):
        """Test that column access executes to correct values."""
        ds = self.create_ds()
        result = ds['age']
        expected = self.df['age']
        # Natural trigger via np.testing
        np.testing.assert_array_equal(result, expected)

    def test_attribute_access_returns_column_expr(self):
        """Test that ds.col returns ColumnExpr."""
        ds = self.create_ds()
        result = ds.age
        self.assertIsInstance(result, ColumnExpr)

    def test_attribute_access_executes_correctly(self):
        """Test that attribute access executes to correct values."""
        ds = self.create_ds()
        result = ds.age
        expected = self.df['age']
        # Natural trigger via np.testing
        np.testing.assert_array_equal(result, expected)

    # ========== Arithmetic Operations ==========

    def test_addition(self):
        """Test column + scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] + 10
        pd_result = self.df['age'] + 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_subtraction(self):
        """Test column - scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] - 5
        pd_result = self.df['age'] - 5
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_multiplication(self):
        """Test column * scalar."""
        ds = self.create_ds()
        ds_result = ds['salary'] * 1.1
        pd_result = self.df['salary'] * 1.1
        np.testing.assert_allclose(ds_result, pd_result)

    def test_division(self):
        """Test column / scalar."""
        ds = self.create_ds()
        ds_result = ds['salary'] / 1000
        pd_result = self.df['salary'] / 1000
        np.testing.assert_allclose(ds_result, pd_result)

    def test_floor_division(self):
        """Test column // scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] // 10
        pd_result = self.df['age'] // 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_modulo(self):
        """Test column % scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] % 10
        pd_result = self.df['age'] % 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_power(self):
        """Test column ** scalar."""
        ds = self.create_ds()
        ds_result = ds['age'] ** 2
        pd_result = self.df['age'] ** 2
        np.testing.assert_array_equal(ds_result, pd_result)

    # ========== Reverse Arithmetic Operations ==========

    def test_reverse_addition(self):
        """Test scalar + column."""
        ds = self.create_ds()
        ds_result = 100 + ds['age']
        pd_result = 100 + self.df['age']
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_reverse_subtraction(self):
        """Test scalar - column."""
        ds = self.create_ds()
        ds_result = 1000 - ds['age']
        pd_result = 1000 - self.df['age']
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_reverse_multiplication(self):
        """Test scalar * column."""
        ds = self.create_ds()
        ds_result = 2 * ds['salary']
        pd_result = 2 * self.df['salary']
        np.testing.assert_allclose(ds_result, pd_result)

    # ========== Unary Operations ==========

    def test_negation(self):
        """Test -column."""
        ds = self.create_ds()
        ds_result = -ds['age']
        pd_result = -self.df['age']
        np.testing.assert_array_equal(ds_result, pd_result)

    # ========== Column-Column Operations ==========

    def test_column_column_addition(self):
        """Test column + column."""
        ds = self.create_ds()
        ds_result = ds['age'] + ds['salary'] / 1000
        pd_result = self.df['age'] + self.df['salary'] / 1000
        np.testing.assert_allclose(ds_result, pd_result)

    # ========== Chained Operations ==========

    def test_chained_arithmetic(self):
        """Test (column - scalar) * scalar + scalar."""
        ds = self.create_ds()
        ds_result = (ds['age'] - 20) * 2 + 10
        pd_result = (self.df['age'] - 20) * 2 + 10
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_complex_chained_operations(self):
        """Test complex chained operations."""
        ds = self.create_ds()
        ds_result = ds['salary'] / 1000 - ds['age']
        pd_result = self.df['salary'] / 1000 - self.df['age']
        np.testing.assert_allclose(ds_result, pd_result)


class TestColumnExprStringOperations(unittest.TestCase):
    """Test ColumnExpr string operations alignment with pandas."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'text': ['Hello World', 'UPPER CASE', 'lower case', 'Mixed Case', '  spaces  '],
                'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_str_upper(self):
        """Test str.upper()."""
        ds = self.create_ds()
        ds_result = ds['name'].str.upper()
        pd_result = self.df['name'].str.upper()
        self.assertTrue(ds_result == pd_result)

    def test_str_lower(self):
        """Test str.lower()."""
        ds = self.create_ds()
        ds_result = ds['name'].str.lower()
        pd_result = self.df['name'].str.lower()
        self.assertTrue(ds_result == pd_result)

    def test_str_length(self):
        """Test str.len() (pandas style)."""
        ds = self.create_ds()
        ds_result = ds['name'].str.len()
        pd_result = self.df['name'].str.len()
        self.assertTrue(ds_result == pd_result)

    def test_str_trim(self):
        """Test str.trim() / str.strip()."""
        ds = self.create_ds()
        ds_result = ds['text'].str.trim()
        pd_result = self.df['text'].str.strip()
        self.assertTrue(ds_result == pd_result)

    def test_str_left(self):
        """Test str.left(n)."""
        ds = self.create_ds()
        ds_result = ds['text'].str.left(5)
        pd_result = self.df['text'].str[:5]
        self.assertTrue(ds_result == pd_result)

    def test_str_right(self):
        """Test str.right(n)."""
        ds = self.create_ds()
        ds_result = ds['text'].str.right(3)
        pd_result = self.df['text'].str[-3:]
        self.assertTrue(ds_result == pd_result)

    def test_str_reverse(self):
        """Test str.reverse()."""
        ds = self.create_ds()
        ds_result = ds['text'].str.reverse()
        pd_result = self.df['text'].apply(lambda x: x[::-1])
        self.assertTrue(ds_result == pd_result)


class TestColumnExprComparisonOperations(unittest.TestCase):
    """Test that comparison operations return Conditions (not boolean Series)."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_greater_than_returns_column_expr(self):
        """Test that > returns ColumnExpr wrapping Condition."""
        ds = self.create_ds()
        result = ds['age'] > 25
        self.assertIsInstance(result, ColumnExpr)
        self.assertIsInstance(result._expr, Condition)
        # Natural trigger via np.testing
        np.testing.assert_array_equal(result, [True, True, True, True, False])

    def test_greater_equal_returns_column_expr(self):
        """Test that >= returns ColumnExpr wrapping Condition."""
        ds = self.create_ds()
        result = ds['age'] >= 28
        self.assertIsInstance(result, ColumnExpr)
        self.assertIsInstance(result._expr, Condition)

    def test_less_than_returns_column_expr(self):
        """Test that < returns ColumnExpr wrapping Condition."""
        ds = self.create_ds()
        result = ds['age'] < 30
        self.assertIsInstance(result, ColumnExpr)
        self.assertIsInstance(result._expr, Condition)

    def test_less_equal_returns_column_expr(self):
        """Test that <= returns ColumnExpr wrapping Condition."""
        ds = self.create_ds()
        result = ds['age'] <= 29
        self.assertIsInstance(result, ColumnExpr)
        self.assertIsInstance(result._expr, Condition)

    def test_equal_returns_column_expr(self):
        """Test that == returns ColumnExpr wrapping Condition."""
        ds = self.create_ds()
        result = ds['age'] == 28
        self.assertIsInstance(result, ColumnExpr)
        self.assertIsInstance(result._expr, Condition)

    def test_not_equal_returns_column_expr(self):
        """Test that != returns ColumnExpr wrapping Condition."""
        ds = self.create_ds()
        result = ds['age'] != 28
        self.assertIsInstance(result, ColumnExpr)
        self.assertIsInstance(result._expr, Condition)

    def test_column_expr_condition_value_counts(self):
        """Test that ColumnExpr wrapping Condition supports value_counts()."""
        ds = self.create_ds()
        result = ds['age'] > 30
        counts = result.value_counts()
        # age values: [28, 31, 29, 45, 22] -> only 31, 45 are > 30
        self.assertEqual(counts[True], 2)
        self.assertEqual(counts[False], 3)

    def test_column_expr_condition_sum(self):
        """Test that ColumnExpr wrapping Condition supports sum() (counts True)."""
        ds = self.create_ds()
        result = ds['age'] > 30
        # age values: [28, 31, 29, 45, 22] -> only 31, 45 are > 30
        self.assertEqual(result.sum(), 2)


class TestColumnExprNullMethods(unittest.TestCase):
    """Test isnull/notnull methods on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [100, 200, 150, 300, 50],
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_notnull_returns_column_expr(self):
        """Test that notnull() returns ColumnExpr wrapping isNotNull()."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].notnull()
        # notnull() returns ColumnExpr wrapping isNotNull() function
        self.assertIsInstance(result, ColumnExpr)

    def test_isnull_returns_column_expr(self):
        """Test that isnull() returns ColumnExpr wrapping isNull()."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].isnull()
        # isnull() returns ColumnExpr wrapping isNull() function
        self.assertIsInstance(result, ColumnExpr)

    def test_notnull_astype_int_matches_pandas(self):
        """Test notnull().astype(int) matches pandas behavior (no NaN in data)."""
        ds = self.create_ds()
        result = ds['value'].notnull().astype(int)
        expected = self.df['value'].notnull().astype(int)
        # Compare values - LazySeries implements __array__
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result.index.equals(expected.index))

    def test_isnull_astype_int_matches_pandas(self):
        """Test isnull().astype(int) matches pandas behavior (no NaN in data)."""
        ds = self.create_ds()
        result = ds['value'].isnull().astype(int)
        expected = self.df['value'].isnull().astype(int)
        # Compare values - LazySeries implements __array__
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result.index.equals(expected.index))

    def test_notnull_condition_for_filtering(self):
        """Test notnull_condition() returns Condition for filtering."""
        ds = self.create_ds()
        result = ds['value'].notnull_condition()
        # Should be a Condition, not ColumnExpr
        from datastore.conditions import Condition

        self.assertIsInstance(result, Condition)

    def test_isnull_condition_for_filtering(self):
        """Test isnull_condition() returns Condition for filtering."""
        ds = self.create_ds()
        result = ds['value'].isnull_condition()
        from datastore.conditions import Condition

        self.assertIsInstance(result, Condition)


class TestColumnExprConditionMethods(unittest.TestCase):
    """Test condition methods on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [100, 200, 150, 300, 50],
                'category': ['A', 'B', 'A', 'C', 'B'],
                'text': ['Hello World', 'test', 'example', 'world', 'hello'],
                'nullable': [1.0, None, 3.0, None, 5.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_isin_returns_condition(self):
        """Test that isin() returns LazyCondition with IN SQL support."""
        ds = self.create_ds()
        result = ds['category'].isin(['A', 'B'])
        # LazyCondition has to_sql() method for SQL generation
        self.assertIn('IN', result.to_sql())
        # Also verify it can be executed as boolean Series
        self.assertEqual(len(result.to_pandas()), len(self.df))

    def test_between_returns_condition(self):
        """Test that between() returns LazyCondition with BETWEEN SQL support."""
        ds = self.create_ds()
        result = ds['value'].between(100, 200)
        # LazyCondition has to_sql() method for SQL generation
        self.assertIn('BETWEEN', result.to_sql())
        # Also verify it can be executed as boolean Series
        self.assertEqual(len(result.to_pandas()), len(self.df))

    def test_like_returns_condition(self):
        """Test that like() returns Condition with LIKE."""
        ds = self.create_ds()
        result = ds['text'].like('%World%')
        self.assertIn('LIKE', str(result))

    def test_isnull_matches_pandas(self):
        """Test that isnull() result matches pandas when executed."""
        ds = self.create_ds()
        # Execute and compare with pandas via .values/.index
        ds_result = ds['nullable'].isnull().astype(int)
        pd_result = self.df['nullable'].isnull().astype(int)
        np.testing.assert_array_equal(ds_result, pd_result)
        self.assertTrue(ds_result.index.equals(pd_result.index))

    def test_notnull_matches_pandas(self):
        """Test that notnull() result matches pandas when executed."""
        ds = self.create_ds()
        # Execute and compare with pandas via .values/.index
        ds_result = ds['nullable'].notnull().astype(int)
        pd_result = self.df['nullable'].notnull().astype(int)
        np.testing.assert_array_equal(ds_result, pd_result)
        self.assertTrue(ds_result.index.equals(pd_result.index))

    def test_isnull_generates_sql(self):
        """Test that isnull() generates proper SQL with isNull function."""
        ds = self.create_ds()
        result = ds['nullable'].isnull()
        sql = result.to_sql()
        self.assertIn('isNull', sql)

    def test_notnull_generates_sql(self):
        """Test that notnull() generates proper SQL with isNotNull function."""
        ds = self.create_ds()
        result = ds['nullable'].notnull()
        sql = result.to_sql()
        self.assertIn('isNotNull', sql)


class TestColumnExprTypeConversion(unittest.TestCase):
    """Test type conversion operations on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'int_col': [1, 2, 3, 4, 5],
                'float_col': [1.5, 2.5, 3.5, 4.5, 5.5],
                'str_col': ['10', '20', '30', '40', '50'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_cast_to_float(self):
        """Test cast to Float64."""
        ds = self.create_ds()
        result = list(ds['int_col'].cast('Float64'))
        self.assertTrue(all(isinstance(x, float) for x in result))

    def test_to_string(self):
        """Test to_string()."""
        ds = self.create_ds()
        result = list(ds['int_col'].to_string())
        self.assertTrue(all(isinstance(x, str) for x in result))
        self.assertEqual(result, ['1', '2', '3', '4', '5'])


class TestColumnExprAggregateFunctions(unittest.TestCase):
    """Test aggregate functions return correct values like pandas."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [28, 31, 29, 45, 22],
                'with_nan': [28.0, 31.0, np.nan, 45.0, 22.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_mean_returns_scalar(self):
        """Test mean() returns a scalar matching pandas behavior."""
        import numpy as np

        ds = self.create_ds()
        result = ds['value'].mean()
        expected = self.df['value'].mean()
        # mean() without groupby returns scalar (matches pandas)
        self.assertIsInstance(result, (int, float, np.integer, np.floating))
        self.assertEqual(type(result), type(expected))
        self.assertAlmostEqual(float(result), expected)

    def test_mean_with_nan_skipna(self):
        """Test mean() correctly skips NaN values like pandas."""
        ds = self.create_ds()
        result = ds['with_nan'].mean()
        expected = self.df['with_nan'].mean()  # pandas skips NaN by default
        self.assertAlmostEqual(result, expected)

    def test_sum_returns_scalar(self):
        """Test sum() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].sum()
        expected = self.df['value'].sum()
        self.assertEqual(result, expected)

    def test_min_returns_scalar(self):
        """Test min() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].min()
        expected = self.df['value'].min()
        self.assertEqual(result, expected)

    def test_max_returns_scalar(self):
        """Test max() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].max()
        expected = self.df['value'].max()
        self.assertEqual(result, expected)

    def test_std_returns_scalar(self):
        """Test std() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].std()
        expected = self.df['value'].std()
        self.assertAlmostEqual(result, expected, places=4)

    def test_count_returns_int(self):
        """Test count() returns an integer like pandas."""
        ds = self.create_ds()
        result = ds['value'].count()
        expected = self.df['value'].count()
        self.assertEqual(result, expected)

    def test_median_returns_scalar(self):
        """Test median() returns a scalar like pandas."""
        ds = self.create_ds()
        result = ds['value'].median()
        expected = self.df['value'].median()
        self.assertEqual(result, expected)

    def test_round_mean(self):
        """Test round(mean(), 2) works with scalar result."""
        ds = self.create_ds()
        result = round(ds['value'].mean(), 2)
        expected = round(self.df['value'].mean(), 2)
        self.assertEqual(result, expected)

    def test_mean_sql_returns_column_expr(self):
        """Test mean_sql() returns ColumnExpr for SQL building."""
        ds = self.create_ds()
        result = ds['value'].mean_sql()
        self.assertIsInstance(result, ColumnExpr)
        self.assertIn('avg', result.to_sql().lower())


class TestColumnExprMathFunctions(unittest.TestCase):
    """Test math functions on ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [100.5, -200.3, 150.7, -300.2, 50.1],
                'positive': [4.0, 9.0, 16.0, 25.0, 36.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_abs(self):
        """Test abs()."""
        ds = self.create_ds()
        ds_result = ds['value'].abs()
        pd_result = self.df['value'].abs()
        np.testing.assert_allclose(ds_result, pd_result)

    def test_round(self):
        """Test round()."""
        ds = self.create_ds()
        ds_result = ds['value'].round(0)
        pd_result = self.df['value'].round(0)
        np.testing.assert_allclose(ds_result, pd_result)

    def test_sqrt(self):
        """Test sqrt()."""
        ds = self.create_ds()
        ds_result = ds['positive'].sqrt()
        pd_result = np.sqrt(self.df['positive'])
        np.testing.assert_allclose(ds_result, pd_result)

    def test_builtin_round(self):
        """Test Python's built-in round() function on ColumnExpr."""
        ds = self.create_ds()
        ds_result = round(ds['value'], 1)
        # Should return ColumnExpr
        self.assertIsInstance(ds_result, ColumnExpr)
        # Execute and compare
        pd_result = self.df['value'].round(1)
        np.testing.assert_allclose(list(ds_result), list(pd_result))

    def test_builtin_round_no_decimals(self):
        """Test round() without decimal places."""
        ds = self.create_ds()
        ds_result = round(ds['value'])
        self.assertIsInstance(ds_result, ColumnExpr)
        pd_result = self.df['value'].round(0)
        np.testing.assert_allclose(list(ds_result), list(pd_result))

    def test_builtin_round_on_aggregate(self):
        """Test Python round() on scalar aggregate result like mean()."""
        import numpy as np
        ds = self.create_ds()
        # mean() returns scalar (matches pandas)
        mean_result = ds['value'].mean()
        self.assertIsInstance(mean_result, (int, float, np.integer, np.floating))
        # Python round() works on scalar
        rounded = round(float(mean_result), 2)
        expected = round(self.df['value'].mean(), 2)
        self.assertAlmostEqual(rounded, expected)

    def test_fillna_with_aggregate_expression(self):
        """Test fillna() with aggregate expression like mean()."""
        ds = self.create_ds()
        # fillna with mean should work - returns LazySeries
        result = ds['value'].fillna(ds['value'].mean())
        # Natural trigger via len()
        self.assertEqual(len(result), 5)


class TestColumnExprFillna(unittest.TestCase):
    """Test fillna() method for proper NaN handling."""

    def test_fillna_string_column(self):
        """Test fillna() on string column with NaN values."""
        df = pd.DataFrame(
            {
                'Cabin': ['A1', np.nan, 'B2', np.nan, 'C3'],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Cabin'].fillna('Unknown')
        expected = df['Cabin'].fillna('Unknown')

        np.testing.assert_array_equal(result, expected)

    def test_fillna_numeric_column(self):
        """Test fillna() on numeric column with NaN values."""
        df = pd.DataFrame(
            {
                'Age': [28.0, np.nan, 29.0, np.nan, 22.0],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Age'].fillna(0)
        expected = df['Age'].fillna(0)

        np.testing.assert_array_equal(result, expected)

    def test_fillna_with_mean(self):
        """Test fillna() with mean value."""
        df = pd.DataFrame(
            {
                'Age': [28.0, np.nan, 29.0, np.nan, 22.0],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Age'].fillna(ds['Age'].mean())
        expected = df['Age'].fillna(df['Age'].mean())

        np.testing.assert_array_almost_equal(result, expected)

    def test_fillna_with_mode(self):
        """Test fillna() with mode value for string column."""
        df = pd.DataFrame(
            {
                'Embarked': ['S', 'C', np.nan, 'S', np.nan, 'S'],
            }
        )
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        result = ds['Embarked'].fillna(ds['Embarked'].mode()[0])
        expected = df['Embarked'].fillna(df['Embarked'].mode()[0])

        np.testing.assert_array_equal(result, expected)


class TestColumnExprDisplayBehavior(unittest.TestCase):
    """Test that ColumnExpr displays like pandas Series."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'value': [100.5, 200.3, 150.7],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_repr_shows_values(self):
        """Test that repr() shows actual values."""
        ds = self.create_ds()
        result = repr(ds['value'])
        self.assertIn('100.5', result)
        self.assertIn('200.3', result)

    def test_str_shows_values(self):
        """Test that str() shows actual values."""
        ds = self.create_ds()
        result = str(ds['value'])
        self.assertIn('100.5', result)

    def test_len(self):
        """Test len() returns correct length."""
        ds = self.create_ds()
        self.assertEqual(len(ds['name']), 3)

    def test_iteration(self):
        """Test iteration over ColumnExpr."""
        ds = self.create_ds()
        values = list(ds['name'])
        self.assertEqual(values, ['Alice', 'Bob', 'Charlie'])

    def test_tolist(self):
        """Test tolist() method."""
        ds = self.create_ds()
        values = ds['name'].tolist()
        self.assertEqual(values, ['Alice', 'Bob', 'Charlie'])

    def test_getitem_index(self):
        """Test subscripting with integer index."""
        ds = self.create_ds()
        # ds['col'][0] should return first value
        first_value = ds['value'][0]
        self.assertEqual(first_value, 100.5)

    def test_getitem_slice(self):
        """Test subscripting with slice."""
        ds = self.create_ds()
        values = ds['value'][:2]
        self.assertEqual(len(values), 2)

    def test_mode_returns_series(self):
        """Test mode() returns LazySeries that can be executed."""
        ds = self.create_ds()
        result = ds['name'].mode()
        # Natural trigger via len()
        self.assertGreater(len(result), 0)

    def test_mode_subscript(self):
        """Test mode()[0] pattern - regression test for TypeError."""
        ds = self.create_ds()
        # This used to fail with: TypeError: 'ColumnExpr' object is not subscriptable
        first_mode = ds['name'].mode()[0]
        # Should return a scalar value
        self.assertIsInstance(first_mode, str)


class TestColumnExprFilterIntegration(unittest.TestCase):
    """Test that ColumnExpr works correctly with filter()."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_filter_with_column_expr_comparison(self):
        """Test filter with ds['col'] > value."""
        ds = self.create_ds()
        filtered = ds.filter(ds['age'] > 28)
        expected = self.df[self.df['age'] > 28]
        np.testing.assert_array_equal(filtered['age'], expected['age'])

    def test_filter_with_attribute_comparison(self):
        """Test filter with ds.col > value."""
        ds = self.create_ds()
        filtered = ds.filter(ds.age >= 29)
        expected = self.df[self.df['age'] >= 29]
        np.testing.assert_array_equal(filtered['age'], expected['age'])

    def test_filter_with_multiple_conditions(self):
        """Test filter with combined conditions."""
        ds = self.create_ds()
        filtered = ds.filter((ds.age > 25) & (ds.salary > 50000))
        expected = self.df[(self.df['age'] > 25) & (self.df['salary'] > 50000)]
        np.testing.assert_array_equal(filtered['name'], expected['name'])


class TestColumnExprAssignment(unittest.TestCase):
    """Test column assignment with ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [28, 31, 29],
                'salary': [50000.0, 75000.0, 60000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_assign_arithmetic_result(self):
        """Test ds['new'] = ds['col'] * 2."""
        ds = self.create_ds()
        ds['age_doubled'] = ds['age'] * 2
        expected = self.df['age'] * 2
        np.testing.assert_array_equal(ds['age_doubled'], expected)

    def test_assign_complex_expression(self):
        """Test ds['new'] = (col1 / 1000) + (col2 * 2)."""
        ds = self.create_ds()
        ds['complex'] = (ds['salary'] / 1000) + (ds['age'] * 2)
        result = ds.to_df()
        expected = (self.df['salary'] / 1000) + (self.df['age'] * 2)
        np.testing.assert_allclose(result['complex'], expected)

    def test_assign_string_operation(self):
        """Test ds['new'] = ds['col'].str.upper()."""
        ds = self.create_ds()
        ds['name_upper'] = ds['name'].str.upper()
        expected = self.df['name'].str.upper()
        np.testing.assert_array_equal(ds['name_upper'], expected)

    def test_assign_type_cast(self):
        """Test ds['new'] = ds['col'].cast('Float64')."""
        ds = self.create_ds()
        ds['age_float'] = ds['age'].cast('Float64')
        result = ds.to_df()
        self.assertTrue(all(isinstance(x, float) for x in result['age_float']))


class TestColumnExprCombinedPipeline(unittest.TestCase):
    """Test combined pipelines with ColumnExpr."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [28, 31, 29, 45, 22],
                'salary': [50000.0, 75000.0, 60000.0, 120000.0, 45000.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_filter_then_assign(self):
        """Test filter -> assign pipeline."""
        ds = self.create_ds()
        filtered = ds.filter(ds['salary'] > 50000)
        filtered['bonus'] = filtered['salary'] * 0.1
        result = filtered.to_df()

        # Verify filter
        expected_df = self.df[self.df['salary'] > 50000].copy()
        expected_df['bonus'] = expected_df['salary'] * 0.1

        np.testing.assert_allclose(result['bonus'], expected_df['bonus'])

    def test_assign_then_filter(self):
        """Test assign -> filter pipeline."""
        ds = self.create_ds()
        ds['age_doubled'] = ds['age'] * 2
        filtered = ds.filter(ds['age_doubled'] > 50)

        # Verify
        temp_df = self.df.copy()
        temp_df['age_doubled'] = temp_df['age'] * 2
        expected = temp_df[temp_df['age_doubled'] > 50]

        np.testing.assert_array_equal(filtered['name'], expected['name'])

    def test_access_column_from_filtered_result(self):
        """Test accessing column from filtered DataStore."""
        ds = self.create_ds()
        filtered = ds.filter(ds.salary > 50000)
        col_result = filtered['salary']
        expected = self.df[self.df['salary'] > 50000]['salary']
        # Natural trigger via np.testing
        np.testing.assert_allclose(col_result, expected)


class TestLazySlice(unittest.TestCase):
    """Test LazySeries class for lazy head()/tail() operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
                'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_column_head_returns_column_expr_method_mode(self):
        """Test that ColumnExpr.head() returns ColumnExpr (method mode)."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].head(5)

        # Should return ColumnExpr in method mode, not pd.Series
        self.assertIsInstance(result, ColumnExpr)
        self.assertEqual(result._exec_mode, 'method')

    def test_column_tail_returns_column_expr_method_mode(self):
        """Test that ColumnExpr.tail() returns ColumnExpr (method mode)."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].tail(5)

        # Should return ColumnExpr in method mode, not pd.Series
        self.assertIsInstance(result, ColumnExpr)
        self.assertEqual(result._exec_mode, 'method')

    def test_lazy_series_method_executes_on_repr(self):
        """Test that LazySeries executes when displayed."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # repr() should trigger execution
        repr_str = repr(result)

        # Should show actual values
        self.assertIn('10', repr_str)

    def test_lazy_series_method_to_pandas(self):
        """Test explicit execution with to_pandas()."""
        ds = self.create_ds()
        result = ds['value'].head(3).to_pandas()

        # Should return pd.Series
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)

    def test_column_expr_method_chainable(self):
        """Test that ColumnExpr.head() and tail() return new ColumnExpr (method mode)."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds['value'].head(5).tail(2)

        # Should still be ColumnExpr in method mode
        self.assertIsInstance(result, ColumnExpr)
        self.assertEqual(result._exec_mode, 'method')

        # Should work correctly
        final = result.to_pandas()
        self.assertEqual(len(final), 2)

    def test_lazy_series_method_iteration(self):
        """Test iteration triggers execution."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # Iteration should work
        values = list(result)
        self.assertEqual(len(values), 3)

    def test_lazy_series_method_indexing(self):
        """Test indexing triggers execution."""
        ds = self.create_ds()
        result = ds['value'].head(5)

        # Indexing should work
        first_value = result[0]
        self.assertEqual(first_value, 10)

    def test_lazy_series_method_len(self):
        """Test len() triggers execution."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        self.assertEqual(len(result), 3)

    def test_lazy_series_method_properties(self):
        """Test properties like len, index, dtype."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # Natural triggers
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result.index), 3)
        self.assertIsNotNone(result.dtype)

    def test_lazy_series_method_arithmetic(self):
        """Test arithmetic on LazySeries."""
        ds = self.create_ds()
        result = ds['value'].head(3)

        # Arithmetic should work
        doubled = result * 2
        np.testing.assert_array_equal(doubled, [20, 40, 60])

    def test_groupby_aggregate_head(self):
        """Test that groupby().agg().head() returns ColumnExpr (method mode)."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        agg_result = ds.groupby('category')['value'].mean()

        # Should be ColumnExpr in aggregation mode
        self.assertIsInstance(agg_result, ColumnExpr)
        self.assertEqual(agg_result._exec_mode, 'agg')

        # head() should return ColumnExpr in method mode
        head_result = agg_result.head(3)
        self.assertIsInstance(head_result, ColumnExpr)
        self.assertEqual(head_result._exec_mode, 'method')

        # Executed result should have correct length
        self.assertEqual(len(head_result), 3)

    def test_groupby_aggregate_tail(self):
        """Test that groupby().agg().tail() returns ColumnExpr (method mode)."""
        from datastore.column_expr import ColumnExpr

        ds = self.create_ds()
        result = ds.groupby('category')['value'].sum().tail(2)

        self.assertIsInstance(result, ColumnExpr)
        self.assertEqual(result._exec_mode, 'method')
        self.assertEqual(len(result), 2)

    def test_scalar_aggregate_no_head_method(self):
        """Test scalar aggregate (no groupby) returns scalar, not ColumnExpr with head()."""
        import numpy as np
        ds = self.create_ds()
        result = ds['value'].mean()

        # Scalar aggregates return actual scalar (matches pandas)
        # Scalars don't have head() method
        expected = self.df['value'].mean()
        self.assertIsInstance(result, (int, float, np.integer, np.floating))
        self.assertAlmostEqual(float(result), expected)

    def test_column_expr_method_aggregations(self):
        """Test calling aggregation methods on ColumnExpr (method mode)."""
        ds = self.create_ds()
        slice_result = ds['value'].head(5)

        # Aggregation methods should work - use float() to trigger execution
        self.assertEqual(float(slice_result.sum()), 10 + 20 + 30 + 40 + 50)
        self.assertEqual(float(slice_result.mean()), 30.0)
        self.assertEqual(float(slice_result.min()), 10)
        self.assertEqual(float(slice_result.max()), 50)


class TestColumnExprPlot(unittest.TestCase):
    """Test ColumnExpr plot accessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
                'name': ['a', 'b', 'c', 'd', 'e'],
                'category': ['X', 'X', 'Y', 'Y', 'Z'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_plot_property_exists(self):
        """Test that plot property exists on ColumnExpr."""
        ds = self.create_ds()
        col_expr = ds['value']
        self.assertTrue(hasattr(col_expr, 'plot'))

    def test_plot_returns_plot_accessor(self):
        """Test that plot returns pandas PlotAccessor."""
        from pandas.plotting._core import PlotAccessor

        ds = self.create_ds()
        col_expr = ds['value']
        plot_accessor = col_expr.plot

        self.assertIsInstance(plot_accessor, PlotAccessor)

    def test_plot_callable(self):
        """Test that plot accessor is callable."""
        ds = self.create_ds()
        col_expr = ds['value']

        # The plot accessor should be callable
        self.assertTrue(callable(col_expr.plot))

    def test_plot_has_common_methods(self):
        """Test that plot accessor has common plotting methods."""
        ds = self.create_ds()
        col_expr = ds['value']
        plot = col_expr.plot

        # Check common plot methods exist
        self.assertTrue(hasattr(plot, 'bar'))
        self.assertTrue(hasattr(plot, 'barh'))
        self.assertTrue(hasattr(plot, 'box'))
        self.assertTrue(hasattr(plot, 'hist'))
        self.assertTrue(hasattr(plot, 'kde'))
        self.assertTrue(hasattr(plot, 'density'))
        self.assertTrue(hasattr(plot, 'line'))
        self.assertTrue(hasattr(plot, 'pie'))

    def test_plot_with_string_column(self):
        """Test that plot works with string columns too."""
        ds = self.create_ds()
        col_expr = ds['name']

        # Should still have plot accessor
        self.assertTrue(hasattr(col_expr, 'plot'))

    def test_plot_after_arithmetic(self):
        """Test that plot works after arithmetic operations."""
        ds = self.create_ds()
        col_expr = ds['value'] * 2

        # Should have plot accessor after arithmetic
        self.assertTrue(hasattr(col_expr, 'plot'))

        from pandas.plotting._core import PlotAccessor

        self.assertIsInstance(col_expr.plot, PlotAccessor)

    def test_plot_data_matches_executed(self):
        """Test that plot uses correctly executed data."""
        ds = self.create_ds()
        col_expr = ds['value']

        # PlotAccessor wraps the Series, we can verify via _parent
        plot_accessor = col_expr.plot

        # Natural trigger: compare plot's data with expected pandas Series
        expected = self.df['value']
        assert_series_equal(plot_accessor._parent, expected)


class TestColumnExprPandasProperties(unittest.TestCase):
    """Test ColumnExpr pandas-compatible properties."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
                'name': ['a', 'b', 'c', 'd', 'e'],
                'category': ['X', 'X', 'Y', 'Y', 'Z'],
                'with_nan': [1.0, np.nan, 3.0, 4.0, 5.0],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_dtype_property(self):
        """Test dtype property returns correct type."""
        ds = self.create_ds()
        self.assertEqual(ds['value'].dtype, np.dtype('int64'))
        self.assertEqual(ds['name'].dtype, np.dtype('object'))

    def test_dtypes_property(self):
        """Test dtypes property (alias for dtype)."""
        ds = self.create_ds()
        self.assertEqual(ds['value'].dtypes, ds['value'].dtype)

    def test_shape_property(self):
        """Test shape property returns correct tuple."""
        ds = self.create_ds()
        self.assertEqual(ds['value'].shape, (5,))

    def test_ndim_property(self):
        """Test ndim is always 1 for Series."""
        ds = self.create_ds()
        self.assertEqual(ds['value'].ndim, 1)
        self.assertEqual(ds['name'].ndim, 1)

    def test_index_property(self):
        """Test index property returns correct index."""
        ds = self.create_ds()
        index = ds['value'].index
        np.testing.assert_array_equal(index, [0, 1, 2, 3, 4])

    def test_empty_property(self):
        """Test empty property."""
        ds = self.create_ds()
        self.assertFalse(ds['value'].empty)

    def test_T_property(self):
        """Test T (transpose) property."""
        ds = self.create_ds()
        np.testing.assert_array_equal(ds['value'].T, np.array([10, 20, 30, 40, 50]))

    def test_axes_property(self):
        """Test axes property returns list of axis."""
        ds = self.create_ds()
        axes = ds['value'].axes
        self.assertEqual(len(axes), 1)

    def test_nbytes_property(self):
        """Test nbytes property returns memory size."""
        ds = self.create_ds()
        nbytes = ds['value'].nbytes
        self.assertGreater(nbytes, 0)
        self.assertEqual(nbytes, 40)  # 5 int64 values = 5 * 8 bytes

    def test_hasnans_property(self):
        """Test hasnans property."""
        ds = self.create_ds()
        self.assertFalse(ds['value'].hasnans)
        self.assertTrue(ds['with_nan'].hasnans)

    def test_is_unique_property(self):
        """Test is_unique property."""
        ds = self.create_ds()
        self.assertTrue(ds['value'].is_unique)
        self.assertFalse(ds['category'].is_unique)  # Has duplicates

    def test_is_monotonic_increasing_property(self):
        """Test is_monotonic_increasing property."""
        ds = self.create_ds()
        self.assertTrue(ds['value'].is_monotonic_increasing)

    def test_is_monotonic_decreasing_property(self):
        """Test is_monotonic_decreasing property."""
        ds = self.create_ds()
        self.assertFalse(ds['value'].is_monotonic_decreasing)

    def test_array_property(self):
        """Test array property returns underlying array."""
        ds = self.create_ds()
        arr = ds['value'].array
        self.assertEqual(len(arr), 5)


class TestColumnExprPandasMethods(unittest.TestCase):
    """Test ColumnExpr pandas-compatible methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
                'name': ['a', 'b', 'c', 'd', 'e'],
                'category': ['X', 'X', 'Y', 'Y', 'Z'],
            }
        )

    def create_ds(self):
        """Create a DataStore with the test DataFrame."""
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(self.df.copy())]
        return ds

    def test_to_list_method(self):
        """Test to_list method."""
        ds = self.create_ds()
        result = ds['value'].to_list()
        self.assertEqual(result, [10, 20, 30, 40, 50])

    def test_to_dict_method(self):
        """Test to_dict method."""
        ds = self.create_ds()
        result = ds['value'].to_dict()
        self.assertEqual(result, {0: 10, 1: 20, 2: 30, 3: 40, 4: 50})

    def test_to_frame_method(self):
        """Test to_frame method returns DataStore."""
        ds = self.create_ds()
        result = ds['value'].to_frame()
        self.assertIsInstance(result, DataStore)
        # The column might be named 'value' or None depending on execution
        self.assertEqual(len(result.columns), 1)

    def test_copy_method(self):
        """Test copy method returns LazySeries that executes to Series values."""
        ds = self.create_ds()
        result = ds['value'].copy()
        # copy() now returns LazySeries for lazy execution
        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(result, [10, 20, 30, 40, 50])

    def test_describe_method(self):
        """Test describe method."""
        ds = self.create_ds()
        result = ds['value'].describe()
        self.assertIn('mean', result.index)
        self.assertIn('std', result.index)
        self.assertIn('min', result.index)
        self.assertIn('max', result.index)

    def test_sample_method(self):
        """Test sample method with random_state."""
        ds = self.create_ds()
        result = ds['value'].sample(3, random_state=42)
        self.assertEqual(len(result), 3)

    def test_nlargest_method(self):
        """Test nlargest method."""
        ds = self.create_ds()
        result = ds['value'].nlargest(3)
        np.testing.assert_array_equal(result, [50, 40, 30])

    def test_nsmallest_method(self):
        """Test nsmallest method."""
        ds = self.create_ds()
        result = ds['value'].nsmallest(3)
        np.testing.assert_array_equal(result, [10, 20, 30])

    def test_drop_duplicates_method(self):
        """Test drop_duplicates method."""
        ds = self.create_ds()
        result = ds['category'].drop_duplicates()
        np.testing.assert_array_equal(result, ['X', 'Y', 'Z'])

    def test_duplicated_method(self):
        """Test duplicated method."""
        ds = self.create_ds()
        result = ds['category'].duplicated()
        expected = [False, True, False, True, False]
        np.testing.assert_array_equal(result, expected)

    def test_agg_single_func(self):
        """Test agg with single function."""
        ds = self.create_ds()
        result = ds['value'].agg('mean')
        # Use float() to trigger execution for ColumnExpr result
        self.assertEqual(float(result), 30.0)

    def test_agg_multiple_funcs(self):
        """Test agg with multiple functions."""
        ds = self.create_ds()
        result = ds['value'].agg(['sum', 'mean'])
        self.assertEqual(result['sum'], 150)
        self.assertEqual(result['mean'], 30.0)

    def test_aggregate_alias(self):
        """Test aggregate is alias for agg."""
        ds = self.create_ds()
        result = ds['value'].aggregate('mean')
        self.assertEqual(result, 30.0)

    def test_where_method(self):
        """Test where method."""
        ds = self.create_ds()
        # Use comparison which returns ColumnExpr (condition)
        cond = ds['value'] > 25
        result = ds['value'].where(cond, 0)
        expected = [0, 0, 30, 40, 50]
        np.testing.assert_array_equal(result, expected)

    def test_argsort_method(self):
        """Test argsort method."""
        ds = self.create_ds()
        result = ds['value'].argsort()
        # Already sorted, so argsort returns [0, 1, 2, 3, 4]
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])

    def test_sort_index_method(self):
        """Test sort_index method."""
        ds = self.create_ds()
        result = ds['value'].sort_index(ascending=False)
        np.testing.assert_array_equal(result.index, [4, 3, 2, 1, 0])

    def test_info_method(self):
        """Test info method runs without error."""
        import io

        ds = self.create_ds()
        buf = io.StringIO()
        ds['value'].info(buf=buf)
        output = buf.getvalue()
        self.assertIn('int64', output)


class TestColumnExprCatSparseAccessors(unittest.TestCase):
    """Test ColumnExpr cat and sparse accessors."""

    def test_cat_accessor_exists(self):
        """Test that cat accessor exists on ColumnExpr."""
        # Create categorical data
        df = pd.DataFrame({'category': pd.Categorical(['a', 'b', 'c', 'a', 'b'])})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        col_expr = ds['category']
        self.assertTrue(hasattr(col_expr, 'cat'))

    def test_cat_accessor_categories(self):
        """Test cat.categories returns correct categories."""
        df = pd.DataFrame({'category': pd.Categorical(['a', 'b', 'c', 'a', 'b'])})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        categories = ds['category'].cat.categories
        np.testing.assert_array_equal(categories, ['a', 'b', 'c'])

    def test_cat_accessor_codes(self):
        """Test cat.codes returns correct codes."""
        df = pd.DataFrame({'category': pd.Categorical(['a', 'b', 'c', 'a', 'b'])})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        codes = ds['category'].cat.codes
        np.testing.assert_array_equal(codes, [0, 1, 2, 0, 1])

    def test_cat_accessor_ordered(self):
        """Test cat.ordered property."""
        df = pd.DataFrame({'category': pd.Categorical(['a', 'b', 'c'], ordered=True)})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        self.assertTrue(ds['category'].cat.ordered)

    def test_cat_accessor_raises_for_non_categorical(self):
        """Test that cat accessor raises error for non-categorical data."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        with self.assertRaises(AttributeError):
            _ = ds['value'].cat.categories

    def test_sparse_accessor_exists(self):
        """Test that sparse accessor exists on ColumnExpr."""
        # Create sparse data
        sparse_arr = pd.arrays.SparseArray([0, 0, 1, 0, 2])
        df = pd.DataFrame({'sparse_col': sparse_arr})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        col_expr = ds['sparse_col']
        self.assertTrue(hasattr(col_expr, 'sparse'))

    def test_sparse_accessor_density(self):
        """Test sparse.density returns correct value."""
        sparse_arr = pd.arrays.SparseArray([0, 0, 1, 0, 2])
        df = pd.DataFrame({'sparse_col': sparse_arr})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        density = ds['sparse_col'].sparse.density
        self.assertAlmostEqual(density, 0.4)  # 2 non-zero out of 5

    def test_sparse_accessor_fill_value(self):
        """Test sparse.fill_value returns correct value."""
        sparse_arr = pd.arrays.SparseArray([0, 0, 1, 0, 2], fill_value=0)
        df = pd.DataFrame({'sparse_col': sparse_arr})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        fill_value = ds['sparse_col'].sparse.fill_value
        self.assertEqual(fill_value, 0)

    def test_sparse_accessor_raises_for_non_sparse(self):
        """Test that sparse accessor raises error for non-sparse data."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore('chdb')
        ds._lazy_ops = [LazyDataFrameSource(df.copy())]

        with self.assertRaises(AttributeError):
            _ = ds['value'].sparse.density


if __name__ == '__main__':
    unittest.main()
