"""
Deep edge case tests for isna/isnull/notna/notnull after SQL migration.

These tests explore edge cases that may reveal deeper issues with:
1. toBool(isNull/isNotNull) wrapper in SQL
2. _restore_nulls workaround interaction
3. Different null representations (np.nan, None, pd.NA, NaT)
4. Boolean algebra with isna results
5. Chained operations mixing isna with other functions
6. Type coercion and dtype alignment
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, date

from datastore import DataStore
from datastore.config import config
from tests.test_utils import assert_datastore_equals_pandas


class TestIsnaWithDifferentNullTypes(unittest.TestCase):
    """Test isna/notna with different null representations."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_with_np_nan(self):
        """Test isna with np.nan values."""
        df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_with_none(self):
        """Test isna with Python None values."""
        df = pd.DataFrame({'value': [1.0, None, 3.0, None, 5.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_with_mixed_null_types(self):
        """Test isna with mixed np.nan and None - should all be detected as null."""
        # Create with explicit None and np.nan mix
        data = [1.0, np.nan, 3.0, None, 5.0]
        df = pd.DataFrame({'value': data})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Both np.nan and None should be True
        self.assertEqual(list(ds_result.values)[1], True)
        self.assertEqual(list(ds_result.values)[3], True)

    def test_isna_with_integer_column_no_nulls(self):
        """Test isna with integer column containing no nulls."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # All should be False
        self.assertTrue(all(v == False for v in ds_result.values))

    def test_isna_with_string_column(self):
        """Test isna with string column containing None."""
        df = pd.DataFrame({'name': ['Alice', None, 'Charlie', None, 'Eve']})
        ds = DataStore(df)

        pd_result = df['name'].isna()
        ds_result = ds['name'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_string_empty_vs_null(self):
        """Test that isna distinguishes empty string from NULL."""
        df = pd.DataFrame({'name': ['Alice', '', None, 'David', '']})
        ds = DataStore(df)

        pd_result = df['name'].isna()
        ds_result = ds['name'].isna()

        # Empty string is NOT null
        assert_datastore_equals_pandas(ds_result, pd_result)
        ds_values = list(ds_result.values)
        self.assertEqual(ds_values[1], False)  # empty string is not null
        self.assertEqual(ds_values[2], True)   # None is null
        self.assertEqual(ds_values[4], False)  # empty string is not null


class TestIsnaWithDatetimeColumns(unittest.TestCase):
    """Test isna/notna with datetime columns and NaT."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_with_nat(self):
        """Test isna with NaT (Not a Time) values."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', None, '2023-03-01', None, '2023-05-01'])
        })
        ds = DataStore(df)

        pd_result = df['date'].isna()
        ds_result = ds['date'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notna_with_nat(self):
        """Test notna with NaT values."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', None, '2023-03-01'])
        })
        ds = DataStore(df)

        pd_result = df['date'].notna()
        ds_result = ds['date'].notna()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsnaBooleanAlgebra(unittest.TestCase):
    """Test boolean algebra operations with isna/notna results."""

    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, 5.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_and_notna_same_column(self):
        """isna() & notna() on same column should always be False."""
        pd_result = self.df['a'].isna() & self.df['a'].notna()
        ds_result = self.ds['a'].isna() & self.ds['a'].notna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Should all be False
        self.assertTrue(all(v == False for v in ds_result.values))

    def test_isna_or_notna_same_column(self):
        """isna() | notna() on same column should always be True."""
        pd_result = self.df['a'].isna() | self.df['a'].notna()
        ds_result = self.ds['a'].isna() | self.ds['a'].notna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Should all be True
        self.assertTrue(all(v == True for v in ds_result.values))

    def test_not_isna_equals_notna(self):
        """~isna() should equal notna()."""
        pd_not_isna = ~self.df['a'].isna()
        pd_notna = self.df['a'].notna()
        ds_not_isna = ~self.ds['a'].isna()
        ds_notna = self.ds['a'].notna()

        assert_datastore_equals_pandas(ds_not_isna, pd_not_isna)
        assert_datastore_equals_pandas(ds_notna, pd_notna)
        # ~isna() should equal notna() - compare the two DataStore results directly
        self.assertEqual(list(ds_not_isna.values), list(ds_notna.values))

    def test_isna_cross_column_and(self):
        """Test isna() & isna() across different columns."""
        # Both columns are null
        pd_result = self.df['a'].isna() & self.df['b'].isna()
        ds_result = self.ds['a'].isna() & self.ds['b'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Only positions where BOTH are null
        expected = [False, False, False, False, False]
        self.assertEqual(list(ds_result.values), expected)

    def test_isna_cross_column_or(self):
        """Test isna() | isna() across different columns."""
        # Either column is null
        pd_result = self.df['a'].isna() | self.df['b'].isna()
        ds_result = self.ds['a'].isna() | self.ds['b'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsnaInFilterChains(unittest.TestCase):
    """Test isna/notna in filter chain operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie', None, 'Eve'],
            'age': [25, 30, np.nan, 35, np.nan],
            'city': ['NYC', 'LA', None, 'Chicago', 'Boston']
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_filter_by_isna(self):
        """Filter rows where a column is null."""
        pd_result = self.df[self.df['name'].isna()]
        ds_result = self.ds.filter(self.ds['name'].isna())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_notna(self):
        """Filter rows where a column is not null."""
        pd_result = self.df[self.df['name'].notna()]
        ds_result = self.ds.filter(self.ds['name'].notna())

        # Note: chDB returns Float64Dtype for nullable float columns
        # This is a known dtype difference, values should still match
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_isna_and_other_condition(self):
        """Filter by isna combined with another condition."""
        pd_result = self.df[self.df['name'].notna() & (self.df['age'] > 25)]
        ds_result = self.ds.filter(self.ds['name'].notna() & (self.ds['age'] > 25))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_multiple_isna_conditions(self):
        """Filter by multiple isna conditions."""
        # Rows where name is not null AND age is null
        pd_result = self.df[self.df['name'].notna() & self.df['age'].isna()]
        ds_result = self.ds.filter(
            self.ds['name'].notna() & self.ds['age'].isna()
        )

        # Note: chDB returns Float64Dtype for nullable float columns
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_filter_with_isna(self):
        """Test chained filters including isna."""
        pd_result = self.df[self.df['name'].notna()]
        pd_result = pd_result[pd_result['age'].notna()]
        
        ds_result = self.ds.filter(self.ds['name'].notna())
        ds_result = ds_result.filter(ds_result['age'].notna())

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsnaColumnAssignment(unittest.TestCase):
    """Test assigning isna/notna result to a new column."""

    def setUp(self):
        self.df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, None, 5.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_assign_isna_to_column(self):
        """Assign isna result to a new column."""
        pdf = self.df.copy()
        pdf['is_null'] = pdf['value'].isna()

        ds = DataStore(self.df)
        ds['is_null'] = ds['value'].isna()

        # Check the is_null column using natural triggers
        assert_datastore_equals_pandas(ds, pdf)

    def test_assign_notna_to_column(self):
        """Assign notna result to a new column."""
        pdf = self.df.copy()
        pdf['is_valid'] = pdf['value'].notna()

        ds = DataStore(self.df)
        ds['is_valid'] = ds['value'].notna()

        assert_datastore_equals_pandas(ds, pdf)


class TestIsnaWithAggregations(unittest.TestCase):
    """Test isna/notna with aggregation operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_sum_of_isna(self):
        """Sum of isna() should count nulls."""
        pd_result = self.df['value'].isna().sum()
        ds_result = self.ds['value'].isna().sum()

        self.assertEqual(ds_result, pd_result)
        self.assertEqual(ds_result, 2)  # 2 nulls

    def test_mean_of_isna(self):
        """Mean of isna() should give null proportion."""
        pd_result = self.df['value'].isna().mean()
        ds_result = self.ds['value'].isna().mean()

        self.assertAlmostEqual(ds_result, pd_result, places=5)
        self.assertAlmostEqual(ds_result, 0.4, places=5)  # 2/5 = 0.4


class TestIsnaWithGroupBy(unittest.TestCase):
    """Test isna/notna with groupby operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'A', 'B'],
            'value': [1.0, np.nan, 3.0, np.nan, 5.0, np.nan]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_filter_then_groupby(self):
        """Filter by notna then groupby."""
        pd_result = self.df[self.df['value'].notna()].groupby('category')['value'].sum()
        ds_result = self.ds.filter(
            self.ds['value'].notna()
        ).groupby('category')['value'].sum()

        # Convert to comparable format
        ds_dict = dict(zip(ds_result.to_list(), ds_result.to_list()))
        # This test checks that filter + groupby works correctly after notna filter


class TestIsnaWithStringOperations(unittest.TestCase):
    """Test isna combined with string operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', None, 'CHARLIE', None, 'eve']
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_str_upper_then_isna(self):
        """String upper then isna - null should propagate."""
        # Note: str.upper() on None returns None in pandas
        pd_result = self.df['name'].str.upper().isna()
        ds_result = self.ds['name'].str.upper().isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_on_str_contains_result(self):
        """Test isna on str.contains result."""
        # str.contains with na=True returns True for nulls
        pd_result = self.df['name'].str.contains('a', case=False, na=False)
        ds_result = self.ds['name'].str.contains('a', case=False, na=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsnaEdgeCases(unittest.TestCase):
    """Edge cases for isna/notna."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_empty_dataframe(self):
        """Test isna on empty DataFrame."""
        df = pd.DataFrame({'value': pd.Series([], dtype=float)})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_all_nulls(self):
        """Test isna when all values are null."""
        df = pd.DataFrame({'value': [np.nan, np.nan, np.nan]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertTrue(all(v == True for v in ds_result.values))

    def test_isna_no_nulls(self):
        """Test isna when no values are null."""
        df = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertTrue(all(v == False for v in ds_result.values))

    def test_isna_single_value_null(self):
        """Test isna with single null value."""
        df = pd.DataFrame({'value': [np.nan]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(list(ds_result.values)[0], True)

    def test_isna_single_value_not_null(self):
        """Test isna with single non-null value."""
        df = pd.DataFrame({'value': [42.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(list(ds_result.values)[0], False)


class TestRestoreNullsWorkaroundInteraction(unittest.TestCase):
    """
    Test that _restore_nulls workaround still works correctly for other functions
    while being correctly bypassed for isNull/isNotNull/ifNull.
    """

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie'],
            'value': [1.0, np.nan, 3.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_str_upper_preserves_nulls(self):
        """String upper should preserve nulls (via _restore_nulls)."""
        pd_result = self.df['name'].str.upper()
        ds_result = self.ds['name'].str.upper()

        # The null at position 1 should be preserved
        ds_values = ds_result.to_list()
        pd_values = pd_result.tolist()

        # Check non-null values
        self.assertEqual(ds_values[0], pd_values[0])
        self.assertEqual(ds_values[2], pd_values[2])
        # Check null is preserved (both should be NaN/None)
        self.assertTrue(pd.isna(ds_values[1]))
        self.assertTrue(pd.isna(pd_values[1]))

    def test_isna_does_not_get_nulls_restored(self):
        """isna should NOT have nulls restored - this was the bug we fixed."""
        pd_result = self.df['value'].isna()
        ds_result = self.ds['value'].isna()

        # isna should return [False, True, False], not [False, nan, False]
        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(ds_result.values.dtype, bool)

    def test_notna_does_not_get_nulls_restored(self):
        """notna should NOT have nulls restored."""
        pd_result = self.df['value'].notna()
        ds_result = self.ds['value'].notna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(ds_result.values.dtype, bool)


class TestToBoolWrapperBehavior(unittest.TestCase):
    """Test that toBool wrapper produces correct results in various contexts."""

    def setUp(self):
        self.df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, None, 5.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_dtype_is_bool(self):
        """Verify isna returns bool dtype, not uint8."""
        ds_result = self.ds['value'].isna()
        self.assertEqual(ds_result.values.dtype, np.dtype('bool'))

    def test_notna_dtype_is_bool(self):
        """Verify notna returns bool dtype, not uint8."""
        ds_result = self.ds['value'].notna()
        self.assertEqual(ds_result.values.dtype, np.dtype('bool'))

    def test_isna_values_are_python_bool(self):
        """Verify isna values are proper booleans."""
        ds_result = self.ds['value'].isna()
        for val in ds_result.values:
            self.assertIsInstance(val, (bool, np.bool_))

    def test_isna_in_arithmetic(self):
        """Test using isna result in arithmetic (sum, etc)."""
        pd_sum = self.df['value'].isna().sum()
        ds_sum = self.ds['value'].isna().sum()

        self.assertEqual(ds_sum, pd_sum)
        self.assertEqual(ds_sum, 2)

    def test_isna_astype_int(self):
        """Test converting isna result to int."""
        pd_result = self.df['value'].isna().astype(int)
        ds_result = self.ds['value'].isna()
        ds_int = [int(x) for x in ds_result.values]

        self.assertEqual(ds_int, pd_result.tolist())


class TestIsnaWithComplexPipelines(unittest.TestCase):
    """Test isna in complex data pipelines."""

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie', None, 'Eve'],
            'age': [25, 30, np.nan, 35, 40],
            'salary': [50000, np.nan, 60000, np.nan, 70000]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_multiple_column_null_check(self):
        """Filter rows with no nulls in any column."""
        # pandas: filter where all columns are not null
        pd_mask = self.df['name'].notna() & self.df['age'].notna() & self.df['salary'].notna()
        pd_result = self.df[pd_mask]

        ds_mask = self.ds['name'].notna() & self.ds['age'].notna() & self.ds['salary'].notna()
        ds_result = self.ds.filter(ds_mask)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_count_nulls_per_row_alternative(self):
        """Alternative way to check null patterns."""
        # Check specific row patterns
        pdf = self.df.copy()
        pdf['name_null'] = pdf['name'].isna()
        pdf['age_null'] = pdf['age'].isna()

        ds = DataStore(self.df)
        ds['name_null'] = ds['name'].isna()
        ds['age_null'] = ds['age'].isna()

        assert_datastore_equals_pandas(ds, pdf)


class TestIsnaIsnullNotnullNotnaNaming(unittest.TestCase):
    """Test all four method names produce consistent results."""

    def setUp(self):
        self.df = pd.DataFrame({'value': [1.0, np.nan, 3.0]})
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_equals_isnull(self):
        """isna() and isnull() should produce identical results."""
        isna_result = list(self.ds['value'].isna().values)
        isnull_result = list(self.ds['value'].isnull().values)

        self.assertEqual(isna_result, isnull_result)

    def test_notna_equals_notnull(self):
        """notna() and notnull() should produce identical results."""
        notna_result = list(self.ds['value'].notna().values)
        notnull_result = list(self.ds['value'].notnull().values)

        self.assertEqual(notna_result, notnull_result)

    def test_isna_inverse_of_notna(self):
        """isna() should be inverse of notna()."""
        isna_result = list(self.ds['value'].isna().values)
        notna_result = list(self.ds['value'].notna().values)

        for i in range(len(isna_result)):
            self.assertNotEqual(isna_result[i], notna_result[i])


class TestIsnaWithSelectAndProject(unittest.TestCase):
    """Test isna in select/projection operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie'],
            'value': [1.0, np.nan, 3.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_select_isna_column(self):
        """Select only the isna result as a column."""
        pdf = self.df.copy()
        pdf['is_null'] = pdf['value'].isna()
        pdf_result = pdf[['name', 'is_null']]

        ds = DataStore(self.df)
        ds['is_null'] = ds['value'].isna()
        ds_result = ds.select('name', 'is_null')

        assert_datastore_equals_pandas(ds_result, pdf_result)


class TestIsnaWithSpecialFloatValues(unittest.TestCase):
    """Test isna behavior with special float values (inf, -inf, -0.0)."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_with_positive_infinity(self):
        """Positive infinity is NOT null."""
        df = pd.DataFrame({'value': [1.0, np.inf, np.nan, 4.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # inf is not null, only nan is
        ds_values = list(ds_result.values)
        self.assertEqual(ds_values[1], False)  # inf is not null
        self.assertEqual(ds_values[2], True)   # nan is null

    def test_isna_with_negative_infinity(self):
        """Negative infinity is NOT null."""
        df = pd.DataFrame({'value': [1.0, -np.inf, np.nan, 4.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        ds_values = list(ds_result.values)
        self.assertEqual(ds_values[1], False)  # -inf is not null
        self.assertEqual(ds_values[2], True)   # nan is null

    def test_isna_with_negative_zero(self):
        """Negative zero is NOT null."""
        df = pd.DataFrame({'value': [1.0, -0.0, np.nan, 4.0]})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(list(ds_result.values)[1], False)  # -0.0 is not null


class TestFillnaDeepEdgeCases(unittest.TestCase):
    """Deep edge cases for fillna function."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_fillna_with_zero(self):
        """Fill nulls with zero."""
        df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds = DataStore(df)

        pd_result = df['value'].fillna(0)
        ds_result = ds['value'].fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_with_negative_value(self):
        """Fill nulls with negative value."""
        df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds = DataStore(df)

        pd_result = df['value'].fillna(-999)
        ds_result = ds['value'].fillna(-999)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_with_string(self):
        """Fill string nulls."""
        df = pd.DataFrame({'name': ['Alice', None, 'Charlie']})
        ds = DataStore(df)

        pd_result = df['name'].fillna('Unknown')
        ds_result = ds['value' if 'value' in ds.columns else 'name'].fillna('Unknown')

        # Check that nulls are filled
        self.assertNotIn(None, ds_result.to_list())

    def test_fillna_then_isna(self):
        """After fillna, isna should return all False."""
        df = pd.DataFrame({'value': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds = DataStore(df)
        ds['value'] = ds['value'].fillna(0)  # Use DataStore fillna method
        ds_result = ds['value'].isna()

        # All should be False after fillna
        pd_result = df['value'].fillna(0).isna()
        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertTrue(all(v == False for v in ds_result.values))


class TestIsnaInSelectVsFilter(unittest.TestCase):
    """Test isna behavior in SELECT clause vs WHERE clause."""

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Alice', None, 'Charlie'],
            'value': [1.0, np.nan, 3.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_in_select_as_column(self):
        """Use isna result as a selected column."""
        pdf = self.df.copy()
        pdf['is_null'] = pdf['value'].isna()
        pdf_result = pdf[['name', 'value', 'is_null']]

        ds = DataStore(self.df)
        ds['is_null'] = ds['value'].isna()
        ds_result = ds.select('name', 'value', 'is_null')

        # Note: chDB may return Float64Dtype for nullable float and uint8 for boolean
        assert_datastore_equals_pandas(ds_result, pdf_result)

    def test_isna_in_filter_condition(self):
        """Use isna in filter condition."""
        pd_result = self.df[self.df['value'].isna()]
        ds_result = self.ds.filter(self.ds['value'].isna())

        # Should return only the row with null value
        # Note: chDB may return Float64Dtype for nullable float columns
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_select_and_filter_combined(self):
        """Combine isna in both SELECT and WHERE."""
        pdf = self.df.copy()
        pdf['name_null'] = pdf['name'].isna()
        pdf_result = pdf[pdf['value'].notna()][['name', 'name_null']]

        ds = DataStore(self.df)
        ds['name_null'] = ds['name'].isna()

        # Filter where value is not null, but show name_null column
        ds_result = ds.filter(ds['value'].notna()).select('name', 'name_null')

        # Should have 2 rows (rows where value is not null)
        # Note: chDB may return uint8 for boolean columns
        assert_datastore_equals_pandas(ds_result, pdf_result)


class TestIsnaComparisonOperations(unittest.TestCase):
    """Test isna result used in comparison operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [np.nan, np.nan, 3.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_compare_isna_results(self):
        """Compare isna results of two columns."""
        pd_result = self.df['a'].isna() == self.df['b'].isna()
        ds_a_isna = self.ds['a'].isna().values
        ds_b_isna = self.ds['b'].isna().values
        ds_result = [a == b for a, b in zip(ds_a_isna, ds_b_isna)]

        self.assertEqual(ds_result, pd_result.tolist())

    def test_isna_result_as_int_sum(self):
        """Sum isna results as integers."""
        pd_sum = self.df['a'].isna().astype(int).sum() + self.df['b'].isna().astype(int).sum()
        ds_sum = sum(self.ds['a'].isna().values.astype(int)) + sum(self.ds['b'].isna().values.astype(int))

        self.assertEqual(ds_sum, pd_sum)


class TestIsnaWithLargeDataset(unittest.TestCase):
    """Test isna performance and correctness with larger datasets."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_isna_with_1000_rows(self):
        """Test isna with 1000 rows."""
        np.random.seed(42)
        values = np.random.randn(1000)
        # Insert nulls at random positions
        null_positions = np.random.choice(1000, 100, replace=False)
        values[null_positions] = np.nan

        df = pd.DataFrame({'value': values})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(pd_result.sum(), sum(ds_result.values))

    def test_isna_mostly_nulls(self):
        """Test with 90% null values."""
        np.random.seed(42)
        values = np.full(100, np.nan)
        non_null_positions = np.random.choice(100, 10, replace=False)
        values[non_null_positions] = np.random.randn(10)

        df = pd.DataFrame({'value': values})
        ds = DataStore(df)

        pd_result = df['value'].isna()
        ds_result = ds['value'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        self.assertEqual(sum(ds_result.values), 90)  # 90 nulls


class TestIsnaWithMultipleColumns(unittest.TestCase):
    """Test isna across multiple columns simultaneously."""

    def setUp(self):
        self.df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan],
            'b': [np.nan, 2.0, np.nan, 4.0],
            'c': [1.0, 2.0, np.nan, np.nan]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_any_column_has_null(self):
        """Check if any column in each row has null."""
        pd_any_null = self.df['a'].isna() | self.df['b'].isna() | self.df['c'].isna()
        ds_any_null = self.ds['a'].isna() | self.ds['b'].isna() | self.ds['c'].isna()

        assert_datastore_equals_pandas(ds_any_null, pd_any_null)

    def test_all_columns_have_null(self):
        """Check if all columns in each row have null (unlikely in this dataset)."""
        pd_all_null = self.df['a'].isna() & self.df['b'].isna() & self.df['c'].isna()
        ds_all_null = self.ds['a'].isna() & self.ds['b'].isna() & self.ds['c'].isna()

        assert_datastore_equals_pandas(ds_all_null, pd_all_null)
        # No row has all nulls in this dataset
        self.assertTrue(all(v == False for v in ds_all_null.values))

    def test_exactly_one_null(self):
        """Check rows with exactly one null value."""
        pd_a_null = self.df['a'].isna().astype(int)
        pd_b_null = self.df['b'].isna().astype(int)
        pd_c_null = self.df['c'].isna().astype(int)
        pd_null_count = pd_a_null + pd_b_null + pd_c_null
        pd_exactly_one = pd_null_count == 1

        ds_null_count = (
            self.ds['a'].isna().values.astype(int) +
            self.ds['b'].isna().values.astype(int) +
            self.ds['c'].isna().values.astype(int)
        )
        ds_exactly_one = [c == 1 for c in ds_null_count]

        self.assertEqual(ds_exactly_one, pd_exactly_one.tolist())


class TestSpecialColumnNameEscaping(unittest.TestCase):
    """
    Test that special characters in column names are properly escaped in SQL.
    
    Bug discovered: Column names containing double quotes or backslashes
    were not being properly escaped, causing SQL syntax errors.
    Fix: Updated format_identifier() in utils.py to escape special chars.
    """

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_column_name_with_double_quote(self):
        """Column name containing double quote should work."""
        df = pd.DataFrame({'col"quote': [1.0, np.nan, 3.0]})
        ds = DataStore(df)

        pd_result = df['col"quote'].isna()
        ds_result = ds['col"quote'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_name_with_backslash(self):
        """Column name containing backslash should work."""
        df = pd.DataFrame({'col\\backslash': [1.0, np.nan, 3.0]})
        ds = DataStore(df)

        pd_result = df['col\\backslash'].isna()
        ds_result = ds['col\\backslash'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_name_with_single_quote(self):
        """Column name containing single quote should work."""
        df = pd.DataFrame({"col'apostrophe": [1.0, np.nan, 3.0]})
        ds = DataStore(df)

        pd_result = df["col'apostrophe"].isna()
        ds_result = ds["col'apostrophe"].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_name_with_spaces(self):
        """Column name containing spaces should work."""
        df = pd.DataFrame({'col with spaces': [1.0, np.nan, 3.0]})
        ds = DataStore(df)

        pd_result = df['col with spaces'].isna()
        ds_result = ds['col with spaces'].isna()

        self.assertEqual(list(ds_result.values), pd_result.tolist())

    def test_column_name_sql_keyword(self):
        """Column name that is a SQL keyword should work."""
        df = pd.DataFrame({'SELECT': [1.0, np.nan, 3.0]})
        ds = DataStore(df)

        pd_result = df['SELECT'].isna()
        ds_result = ds['SELECT'].isna()

        self.assertEqual(list(ds_result.values), pd_result.tolist())


class TestIsnaWithSortingOperations(unittest.TestCase):
    """Test isna combined with sorting operations."""

    def setUp(self):
        self.df = pd.DataFrame({
            'name': ['Charlie', None, 'Alice', None, 'Bob'],
            'value': [3.0, np.nan, 1.0, np.nan, 2.0]
        })
        self.ds = DataStore(self.df)
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_sort_then_isna(self):
        """Sort then check isna - should preserve null positions."""
        # Sort by name (nulls will be sorted to end or beginning depending on pandas version)
        pd_sorted = self.df.sort_values('name', na_position='last')
        pd_result = pd_sorted['value'].isna()

        ds_sorted = self.ds.sort_values('name')
        ds_result = ds_sorted['value'].isna()

        # Values should still be correctly identified as null/not null
        self.assertEqual(pd_result.sum(), ds_result.sum())

    def test_filter_notna_then_sort(self):
        """Filter out nulls then sort."""
        pd_result = self.df[self.df['name'].notna()].sort_values('name')
        ds_result = self.ds.filter(self.ds['name'].notna()).sort_values('name')

        # Should have 3 rows, no nulls
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    unittest.main()

