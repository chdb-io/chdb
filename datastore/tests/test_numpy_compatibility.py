"""
NumPy Compatibility Test Suite for DataStore

Tests verify that DataStore and ColumnExpr work seamlessly with NumPy functions,
providing the same level of compatibility as Pandas DataFrame/Series.

Key features tested:
- __array__ interface implementation
- Direct usage with NumPy statistical functions (mean, sum, std, etc.)
- Direct usage with NumPy array functions (concatenate, dot, corrcoef, etc.)
- Comparison functions (allclose, equal, isclose)
- .values property and .to_numpy() method
"""

import unittest
import numpy as np
import pandas as pd

from datastore import DataStore


class TestNumpyArrayInterface(unittest.TestCase):
    """Test __array__ interface implementation."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1.0, 2.0, 3.0, 4.0, 5.0],
                'b': [2.0, 4.0, 6.0, 8.0, 10.0],
                'c': [1.1, 2.1, 3.1, 4.1, 5.1],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_column_expr_has_array_interface(self):
        """Test that ColumnExpr has __array__ method."""
        self.assertTrue(hasattr(self.ds['a'], '__array__'))

    def test_datastore_has_array_interface(self):
        """Test that DataStore has __array__ method."""
        self.assertTrue(hasattr(self.ds, '__array__'))

    def test_np_array_on_column_expr(self):
        """Test np.array() on ColumnExpr."""
        result = np.array(self.ds['a'])
        expected = np.array(self.df['a'])
        np.testing.assert_array_equal(result, expected)

    def test_np_array_on_datastore(self):
        """Test np.array() on DataStore."""
        result = np.array(self.ds.select('a', 'b', 'c').to_df())
        expected = np.array(self.df[['a', 'b', 'c']])
        np.testing.assert_array_equal(result, expected)

    def test_values_property(self):
        """Test .values property returns numpy array."""
        result = self.ds['a'].values
        expected = self.df['a'].values
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)

    def test_to_numpy_method(self):
        """Test .to_numpy() method."""
        result = self.ds['a'].to_numpy()
        expected = self.df['a'].to_numpy()
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)


class TestNumpyStatisticalFunctions(unittest.TestCase):
    """Test NumPy statistical functions work directly with DataStore."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1.0, 2.0, 3.0, 4.0, 5.0],
                'b': [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_np_mean_direct(self):
        """Test np.mean() works directly on ColumnExpr."""
        result = np.mean(self.ds['a'])
        expected = np.mean(self.df['a'])
        self.assertAlmostEqual(result, expected)

    def test_np_sum_direct(self):
        """Test np.sum() works directly on ColumnExpr."""
        result = np.sum(self.ds['a'])
        expected = np.sum(self.df['a'])
        self.assertAlmostEqual(result, expected)

    def test_np_std_direct(self):
        """Test np.std() works directly on ColumnExpr."""
        result = np.std(self.ds['a'])
        expected = np.std(self.df['a'])
        self.assertAlmostEqual(result, expected, places=10)

    def test_np_var_direct(self):
        """Test np.var() works directly on ColumnExpr."""
        result = np.var(self.ds['a'])
        expected = np.var(self.df['a'])
        self.assertAlmostEqual(result, expected, places=10)

    def test_np_min_direct(self):
        """Test np.min() works directly on ColumnExpr."""
        result = np.min(self.ds['a'])
        expected = np.min(self.df['a'])
        self.assertEqual(result, expected)

    def test_np_max_direct(self):
        """Test np.max() works directly on ColumnExpr."""
        result = np.max(self.ds['a'])
        expected = np.max(self.df['a'])
        self.assertEqual(result, expected)

    def test_np_median_direct(self):
        """Test np.median() works directly on ColumnExpr."""
        result = np.median(self.ds['a'])
        expected = np.median(self.df['a'])
        self.assertAlmostEqual(result, expected)

    def test_np_prod_direct(self):
        """Test np.prod() works directly on ColumnExpr."""
        result = np.prod(self.ds['a'])
        expected = np.prod(self.df['a'])
        self.assertAlmostEqual(result, expected)

    def test_np_argmin_direct(self):
        """Test np.argmin() works directly on ColumnExpr."""
        result = np.argmin(self.ds['a'])
        expected = np.argmin(self.df['a'])
        self.assertEqual(result, expected)

    def test_np_argmax_direct(self):
        """Test np.argmax() works directly on ColumnExpr."""
        result = np.argmax(self.ds['a'])
        expected = np.argmax(self.df['a'])
        self.assertEqual(result, expected)

    def test_np_cumsum_direct(self):
        """Test np.cumsum() works directly on ColumnExpr."""
        result = np.cumsum(self.ds['a'])
        expected = np.cumsum(self.df['a'])
        np.testing.assert_allclose(result, expected)

    def test_np_cumprod_direct(self):
        """Test np.cumprod() works directly on ColumnExpr."""
        result = np.cumprod(self.ds['a'])
        expected = np.cumprod(self.df['a'])
        np.testing.assert_allclose(result, expected)


class TestNumpyComparisonFunctions(unittest.TestCase):
    """Test NumPy comparison functions."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1.0, 2.0, 3.0, 4.0],
                'b': [1.0, 2.0, 3.0, 4.0],
                'c': [1.1, 2.1, 3.1, 4.1],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_np_allclose_equal_columns(self):
        """Test np.allclose() with equal columns."""
        result = np.allclose(self.ds['a'], self.ds['b'])
        self.assertTrue(result)

    def test_np_allclose_close_columns(self):
        """Test np.allclose() with close but not equal columns."""
        result = np.allclose(self.ds['a'], self.ds['c'], rtol=0.1)
        expected = np.allclose(self.df['a'], self.df['c'], rtol=0.1)
        self.assertEqual(result, expected)

    def test_np_allclose_not_close(self):
        """Test np.allclose() with non-close columns."""
        result = np.allclose(self.ds['a'], self.ds['c'], rtol=0.01)
        expected = np.allclose(self.df['a'], self.df['c'], rtol=0.01)
        self.assertEqual(result, expected)

    def test_np_equal(self):
        """Test np.equal() on ColumnExpr."""
        result = np.equal(self.ds['a'], self.ds['b'])
        expected = np.equal(self.df['a'], self.df['b'])
        np.testing.assert_array_equal(result, expected)

    def test_np_isclose(self):
        """Test np.isclose() on ColumnExpr."""
        result = np.isclose(self.ds['a'], self.ds['c'], rtol=0.1)
        expected = np.isclose(self.df['a'], self.df['c'], rtol=0.1)
        np.testing.assert_array_equal(result, expected)


class TestNumpyArrayOperations(unittest.TestCase):
    """Test NumPy array operations."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1.0, 2.0, 3.0, 4.0, 5.0],
                'b': [2.0, 4.0, 6.0, 8.0, 10.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_np_dot_direct(self):
        """Test np.dot() works directly on ColumnExpr."""
        result = np.dot(self.ds['a'], self.ds['b'])
        expected = np.dot(self.df['a'], self.df['b'])
        self.assertAlmostEqual(result, expected)

    def test_np_corrcoef_direct(self):
        """Test np.corrcoef() works directly on ColumnExpr."""
        result = np.corrcoef(self.ds['a'], self.ds['b'])
        expected = np.corrcoef(self.df['a'], self.df['b'])
        np.testing.assert_allclose(result, expected)

    def test_np_concatenate_direct(self):
        """Test np.concatenate() works directly on ColumnExpr."""
        result = np.concatenate([self.ds['a'], self.ds['b']])
        expected = np.concatenate([self.df['a'], self.df['b']])
        np.testing.assert_array_equal(result, expected)

    def test_np_sort_direct(self):
        """Test np.sort() works directly on ColumnExpr."""
        result = np.sort(self.ds['a'])
        expected = np.sort(self.df['a'])
        np.testing.assert_array_equal(result, expected)

    def test_np_unique_direct(self):
        """Test np.unique() works directly on ColumnExpr."""
        result = np.unique(self.ds['a'])
        expected = np.unique(self.df['a'])
        np.testing.assert_array_equal(result, expected)

    def test_np_percentile_direct(self):
        """Test np.percentile() works directly on ColumnExpr."""
        result = np.percentile(self.ds['a'], [25, 50, 75])
        expected = np.percentile(self.df['a'], [25, 50, 75])
        np.testing.assert_allclose(result, expected)

    def test_np_histogram_direct(self):
        """Test np.histogram() works directly on ColumnExpr."""
        result_hist, result_bins = np.histogram(self.ds['a'], bins=3)
        expected_hist, expected_bins = np.histogram(self.df['a'], bins=3)
        np.testing.assert_array_equal(result_hist, expected_hist)
        np.testing.assert_allclose(result_bins, expected_bins)

    def test_np_where_direct(self):
        """Test np.where() works directly on ColumnExpr."""
        result = np.where(np.asarray(self.ds['a']) > 3)
        expected = np.where(np.asarray(self.df['a']) > 3)
        np.testing.assert_array_equal(result[0], expected[0])

    def test_np_column_stack(self):
        """Test np.column_stack() with ColumnExpr."""
        result = np.column_stack([self.ds['a'], self.ds['b']])
        expected = np.column_stack([self.df['a'], self.df['b']])
        np.testing.assert_array_equal(result, expected)


class TestNumpyWithSQLFiltering(unittest.TestCase):
    """Test NumPy functions work after SQL filtering."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
                'age': [25, 30, 35, 40, 45],
                'salary': [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_np_mean_after_filter(self):
        """Test np.mean() after SQL filter."""
        filtered_ds = self.ds.filter(self.ds['age'] > 30)
        filtered_df = self.df[self.df['age'] > 30]

        result = np.mean(filtered_ds['salary'])
        expected = np.mean(filtered_df['salary'])
        self.assertAlmostEqual(result, expected)

    def test_np_std_after_filter(self):
        """Test np.std() after SQL filter."""
        filtered_ds = self.ds.filter(self.ds['age'] >= 30)
        filtered_df = self.df[self.df['age'] >= 30]

        result = np.std(filtered_ds['salary'])
        expected = np.std(filtered_df['salary'])
        self.assertAlmostEqual(result, expected, places=10)

    def test_np_corrcoef_after_filter(self):
        """Test np.corrcoef() after SQL filter."""
        filtered_ds = self.ds.filter(self.ds['age'] > 25)
        filtered_df = self.df[self.df['age'] > 25]

        result = np.corrcoef(filtered_ds['age'], filtered_ds['salary'])[0, 1]
        expected = np.corrcoef(filtered_df['age'], filtered_df['salary'])[0, 1]
        self.assertAlmostEqual(result, expected)

    def test_np_percentile_after_filter(self):
        """Test np.percentile() after SQL filter."""
        filtered_ds = self.ds.filter(self.ds['salary'] > 55000)
        filtered_df = self.df[self.df['salary'] > 55000]

        result = np.percentile(filtered_ds['salary'], 50)
        expected = np.percentile(filtered_df['salary'], 50)
        self.assertAlmostEqual(result, expected)


class TestNumpyDataNormalization(unittest.TestCase):
    """Test NumPy-based data normalization patterns."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'value': [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_z_score_normalization(self):
        """Test Z-score normalization using NumPy."""
        # DataStore
        ds_values = self.ds['value']
        ds_normalized = (np.asarray(ds_values) - np.mean(ds_values)) / np.std(ds_values)

        # Pandas
        pd_values = self.df['value']
        pd_normalized = (np.asarray(pd_values) - np.mean(pd_values)) / np.std(pd_values)

        np.testing.assert_allclose(ds_normalized, pd_normalized)

    def test_min_max_normalization(self):
        """Test min-max normalization using NumPy."""
        # DataStore
        ds_values = self.ds['value']
        ds_min, ds_max = np.min(ds_values), np.max(ds_values)
        ds_normalized = (np.asarray(ds_values) - ds_min) / (ds_max - ds_min)

        # Pandas
        pd_values = self.df['value']
        pd_min, pd_max = np.min(pd_values), np.max(pd_values)
        pd_normalized = (np.asarray(pd_values) - pd_min) / (pd_max - pd_min)

        np.testing.assert_allclose(ds_normalized, pd_normalized)


class TestNumpyPtpAndOtherFunctions(unittest.TestCase):
    """Test additional NumPy functions."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1.0, 5.0, 3.0, 9.0, 2.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_np_ptp_direct(self):
        """Test np.ptp() (peak-to-peak) works on ColumnExpr."""
        result = np.ptp(self.ds['a'])
        expected = np.ptp(self.df['a'])
        self.assertAlmostEqual(result, expected)

    def test_np_diff_direct(self):
        """Test np.diff() works on ColumnExpr."""
        result = np.diff(self.ds['a'])
        expected = np.diff(self.df['a'])
        np.testing.assert_array_equal(result, expected)

    def test_np_clip_direct(self):
        """Test np.clip() works on ColumnExpr."""
        result = np.clip(self.ds['a'], 2, 7)
        expected = np.clip(self.df['a'], 2, 7)
        np.testing.assert_array_equal(result, expected)

    def test_np_abs_direct(self):
        """Test np.abs() works on ColumnExpr."""
        df_neg = pd.DataFrame({'a': [-1.0, 2.0, -3.0, 4.0, -5.0]})
        ds_neg = DataStore.from_df(df_neg)

        result = np.abs(ds_neg['a'])
        expected = np.abs(df_neg['a'])
        np.testing.assert_array_equal(result, expected)

    def test_np_sqrt_direct(self):
        """Test np.sqrt() works on ColumnExpr."""
        result = np.sqrt(self.ds['a'])
        expected = np.sqrt(self.df['a'])
        np.testing.assert_allclose(result, expected)

    def test_np_log_direct(self):
        """Test np.log() works on ColumnExpr."""
        result = np.log(self.ds['a'])
        expected = np.log(self.df['a'])
        np.testing.assert_allclose(result, expected)

    def test_np_exp_direct(self):
        """Test np.exp() works on ColumnExpr."""
        result = np.exp(self.ds['a'])
        expected = np.exp(self.df['a'])
        np.testing.assert_allclose(result, expected)


class TestColumnExprStatisticalMethods(unittest.TestCase):
    """Test that ColumnExpr statistical methods match NumPy behavior."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_mean_method(self):
        """Test ColumnExpr.mean() method."""
        result = self.ds['a'].mean()
        expected = self.df['a'].mean()
        self.assertAlmostEqual(result, expected)

    def test_sum_method(self):
        """Test ColumnExpr.sum() method."""
        result = self.ds['a'].sum()
        expected = self.df['a'].sum()
        self.assertAlmostEqual(result, expected)

    def test_std_method(self):
        """Test ColumnExpr.std() method."""
        result = self.ds['a'].std()
        expected = self.df['a'].std()
        self.assertAlmostEqual(result, expected, places=10)

    def test_var_method(self):
        """Test ColumnExpr.var() method."""
        result = self.ds['a'].var()
        expected = self.df['a'].var()
        self.assertAlmostEqual(result, expected, places=10)

    def test_min_method(self):
        """Test ColumnExpr.min() method."""
        result = self.ds['a'].min()
        expected = self.df['a'].min()
        self.assertEqual(result, expected)

    def test_max_method(self):
        """Test ColumnExpr.max() method."""
        result = self.ds['a'].max()
        expected = self.df['a'].max()
        self.assertEqual(result, expected)

    def test_median_method(self):
        """Test ColumnExpr.median() method."""
        result = self.ds['a'].median()
        expected = self.df['a'].median()
        self.assertAlmostEqual(result, expected)

    def test_prod_method(self):
        """Test ColumnExpr.prod() method."""
        result = self.ds['a'].prod()
        expected = self.df['a'].prod()
        self.assertAlmostEqual(result, expected)


class TestNumpyEdgeCases(unittest.TestCase):
    """Test edge cases for NumPy compatibility."""

    def test_empty_dataframe(self):
        """Test NumPy functions with empty DataFrame."""
        df = pd.DataFrame({'a': []})
        ds = DataStore.from_df(df)

        # np.array should return empty array
        result = np.array(ds['a'])
        self.assertEqual(len(result), 0)

    def test_single_row(self):
        """Test NumPy functions with single row."""
        df = pd.DataFrame({'a': [42.0]})
        ds = DataStore.from_df(df)

        result = np.mean(ds['a'])
        expected = np.mean(df['a'])
        self.assertEqual(result, expected)

    def test_with_nan_values(self):
        """Test NumPy functions with NaN values."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds = DataStore.from_df(df)

        # np.nanmean should handle NaN
        result = np.nanmean(ds['a'])
        expected = np.nanmean(df['a'])
        self.assertAlmostEqual(result, expected)

    def test_integer_column(self):
        """Test NumPy functions with integer column."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore.from_df(df)

        result = np.mean(ds['a'])
        expected = np.mean(df['a'])
        self.assertAlmostEqual(result, expected)


class TestColumnExprBooleanOpsWithNumpyArray(unittest.TestCase):
    """
    Test that ColumnExpr boolean operations (&, |, ^) work with numpy arrays.
    
    This is critical for compatibility with libraries like seaborn that use
    numpy arrays for boolean masking, e.g.:
        data[row & col & hue & self._not_na]
    where self._not_na is a numpy ndarray.
    
    Regression test for: TypeError: Cannot AND ColumnExpr with ndarray
    """

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'x', 'y'],
            'c': [10, 20, 30, 40, 50]
        })
        self.ds = DataStore.from_df(self.df)

    def test_column_expr_and_ndarray(self):
        """Test ColumnExpr & ndarray returns ColumnExpr and works correctly."""
        # pandas
        pd_mask = self.df['a'] > 2
        np_mask = np.array([True, True, False, True, False])
        pd_combined = pd_mask & np_mask
        pd_result = self.df[pd_combined]
        
        # DataStore
        ds_mask = self.ds['a'] > 2
        ds_combined = ds_mask & np_mask
        ds_result = self.ds[ds_combined]
        
        # Verify combined mask returns ColumnExpr
        from datastore.column_expr import ColumnExpr
        self.assertIsInstance(ds_combined, ColumnExpr)
        
        # Verify results match
        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)
        np.testing.assert_array_equal(ds_result['c'].values, pd_result['c'].values)

    def test_ndarray_and_column_expr(self):
        """Test ndarray & ColumnExpr (reverse order)."""
        # pandas
        pd_mask = self.df['a'] > 2
        np_mask = np.array([True, True, False, True, False])
        pd_combined = np_mask & pd_mask
        pd_result = self.df[pd_combined]
        
        # DataStore - note: numpy __and__ takes precedence, result is ndarray
        ds_mask = self.ds['a'] > 2
        ds_combined = np_mask & ds_mask
        ds_result = self.ds[ds_combined]
        
        # Verify results match
        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_column_expr_or_ndarray(self):
        """Test ColumnExpr | ndarray works correctly."""
        # pandas
        pd_mask = self.df['a'] > 4
        np_mask = np.array([True, False, False, False, False])
        pd_combined = pd_mask | np_mask
        pd_result = self.df[pd_combined]
        
        # DataStore
        ds_mask = self.ds['a'] > 4
        ds_combined = ds_mask | np_mask
        ds_result = self.ds[ds_combined]
        
        # Verify results match
        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_column_expr_xor_ndarray(self):
        """Test ColumnExpr ^ ndarray works correctly."""
        # pandas
        pd_mask = self.df['a'] > 2
        np_mask = np.array([True, True, True, True, True])
        pd_combined = pd_mask ^ np_mask
        pd_result = self.df[pd_combined]
        
        # DataStore
        ds_mask = self.ds['a'] > 2
        ds_combined = ds_mask ^ np_mask
        ds_result = self.ds[ds_combined]
        
        # Verify results match
        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_seaborn_facetgrid_pattern(self):
        """
        Test the exact pattern used by seaborn's FacetGrid.facet_data():
            data[row & col & hue & self._not_na]
        where row, col, hue can be ColumnExpr and self._not_na is ndarray.
        """
        ds = DataStore({
            'Pclass': [1, 2, 3, 1, 2, 3],
            'Survived': [1, 0, 1, 0, 1, 0],
            'Sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'Embarked': ['S', 'C', 'Q', 'S', 'C', 'Q']
        })
        df = pd.DataFrame({
            'Pclass': [1, 2, 3, 1, 2, 3],
            'Survived': [1, 0, 1, 0, 1, 0],
            'Sex': ['male', 'female', 'male', 'female', 'male', 'female'],
            'Embarked': ['S', 'C', 'Q', 'S', 'C', 'Q']
        })
        
        # Simulate seaborn's mask creation pattern
        # row = (ds['Embarked'] == 'S')  # ColumnExpr
        # col = all True (numpy array)
        # hue = (ds['Sex'] == 'male')   # ColumnExpr
        # not_na = numpy array of True
        
        pd_row = df['Embarked'] == 'S'
        pd_col = np.ones(len(df), dtype=bool)
        pd_hue = df['Sex'] == 'male'
        not_na = np.array([True, True, True, True, True, True])
        
        ds_row = ds['Embarked'] == 'S'
        ds_col = np.ones(len(ds), dtype=bool)
        ds_hue = ds['Sex'] == 'male'
        
        # The seaborn pattern: data[row & col & hue & self._not_na]
        pd_combined = pd_row & pd_col & pd_hue & not_na
        pd_result = df[pd_combined]
        
        ds_combined = ds_row & ds_col & ds_hue & not_na
        ds_result = ds[ds_combined]
        
        # Verify results match
        self.assertEqual(len(ds_result), len(pd_result))
        np.testing.assert_array_equal(ds_result['Pclass'].values, pd_result['Pclass'].values)
        np.testing.assert_array_equal(ds_result['Survived'].values, pd_result['Survived'].values)
        np.testing.assert_array_equal(ds_result['Sex'].values, pd_result['Sex'].values)
        np.testing.assert_array_equal(ds_result['Embarked'].values, pd_result['Embarked'].values)

    def test_column_expr_and_pandas_series(self):
        """Test ColumnExpr & pd.Series works correctly."""
        # pandas
        pd_mask = self.df['a'] > 2
        series_mask = pd.Series([True, True, False, True, False])
        pd_combined = pd_mask & series_mask
        pd_result = self.df[pd_combined]
        
        # DataStore
        ds_mask = self.ds['a'] > 2
        ds_combined = ds_mask & series_mask
        ds_result = self.ds[ds_combined]
        
        # Verify results match
        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_chained_boolean_ops_with_mixed_types(self):
        """Test chaining multiple boolean ops with mixed ColumnExpr and ndarray."""
        # pandas
        pd_mask1 = self.df['a'] > 1
        pd_mask2 = self.df['a'] < 5
        np_mask = np.array([True, True, True, True, False])
        pd_combined = pd_mask1 & pd_mask2 & np_mask
        pd_result = self.df[pd_combined]
        
        # DataStore
        ds_mask1 = self.ds['a'] > 1
        ds_mask2 = self.ds['a'] < 5
        ds_combined = ds_mask1 & ds_mask2 & np_mask
        ds_result = self.ds[ds_combined]
        
        # Verify results match
        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)


if __name__ == '__main__':
    unittest.main()
