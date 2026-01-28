"""
Advanced Pandas Compatibility Test Suite

Tests advanced pandas operations commonly used in real-world data science projects:
- Cumulative operations (cumsum, cumprod, cummin, cummax)
- Rolling window operations
- Expanding window operations
- Shift and diff operations
- Advanced groupby (transform, filter)
- Pivot tables and reshaping
- Conditional logic (where, mask, clip, between)
- Data validation (isin, duplicated, nlargest, nsmallest)
- Statistical correlation
- Advanced indexing (at, iat)

Based on real-world patterns from financial analysis, time series, and customer analytics.
"""

import unittest
import numpy as np
import pandas as pd

import datastore as ds
from tests.test_utils import assert_frame_equal


class TestCumulativeOperations(unittest.TestCase):
    """Test cumulative operations (cumsum, cumprod, cummax, cummin)."""

    def setUp(self):
        self.ts_data_pd = pd.DataFrame(
            {'value': [10, 20, 15, 30, 25, 35], 'price': [100.0, 102.5, 101.0, 105.0, 103.0, 107.5]}
        )
        self.ts_data_ds = ds.DataFrame(
            {'value': [10, 20, 15, 30, 25, 35], 'price': [100.0, 102.5, 101.0, 105.0, 103.0, 107.5]}
        )

    def test_cumsum(self):
        """Test cumulative sum with cumsum()."""
        pd_result = self.ts_data_pd['value'].cumsum()
        ds_result = self.ts_data_ds['value'].cumsum()
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_cumprod(self):
        """Test cumulative product with cumprod()."""
        pd_result = pd.Series([1.1, 1.2, 0.9, 1.05]).cumprod()
        ds_result = ds.Series([1.1, 1.2, 0.9, 1.05]).cumprod()
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_cummax(self):
        """Test cumulative maximum with cummax()."""
        pd_result = self.ts_data_pd['value'].cummax()
        ds_result = self.ts_data_ds['value'].cummax()
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_cummin(self):
        """Test cumulative minimum with cummin()."""
        pd_result = self.ts_data_pd['value'].cummin()
        ds_result = self.ts_data_ds['value'].cummin()
        np.testing.assert_array_equal(ds_result.values, pd_result.values)


class TestShiftDiffOperations(unittest.TestCase):
    """Test shift and diff operations for time series."""

    def setUp(self):
        self.ts_data_pd = pd.DataFrame(
            {'value': [10, 20, 15, 30, 25, 35], 'price': [100.0, 102.5, 101.0, 105.0, 103.0, 107.5]}
        )
        self.ts_data_ds = ds.DataFrame(
            {'value': [10, 20, 15, 30, 25, 35], 'price': [100.0, 102.5, 101.0, 105.0, 103.0, 107.5]}
        )

    def test_shift(self):
        """Test shift values by 1 period."""
        pd_result = self.ts_data_pd['value'].shift(1)
        ds_result = self.ts_data_ds['value'].shift(1)
        # Handle NaN comparison
        np.testing.assert_array_equal(pd.isna(ds_result.values), pd.isna(pd_result.values))
        # Compare non-NaN values
        mask = ~pd.isna(pd_result.values)
        np.testing.assert_array_equal(ds_result.values[mask], pd_result.values[mask])

    def test_diff(self):
        """Test calculate differences with diff()."""
        pd_result = self.ts_data_pd['price'].diff()
        ds_result = self.ts_data_ds['price'].diff()
        # Handle NaN comparison
        mask = ~pd.isna(pd_result.values)
        np.testing.assert_array_almost_equal(ds_result.values[mask], pd_result.values[mask])

    def test_pct_change(self):
        """Test percentage change with pct_change()."""
        pd_result = self.ts_data_pd['price'].pct_change()
        ds_result = self.ts_data_ds['price'].pct_change()
        # Handle NaN comparison
        mask = ~pd.isna(pd_result.values)
        np.testing.assert_array_almost_equal(ds_result.values[mask], pd_result.values[mask])


class TestAdvancedGroupByOperations(unittest.TestCase):
    """Test advanced groupby operations (transform, filter)."""

    def setUp(self):
        self.group_data_pd = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40, 50, 60],
                'weight': [1, 2, 3, 4, 5, 6],
            }
        )
        self.group_data_ds = ds.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40, 50, 60],
                'weight': [1, 2, 3, 4, 5, 6],
            }
        )

    def test_transform_normalize(self):
        """Test groupby().transform() - normalize within groups."""
        pd_result = self.group_data_pd.groupby('category')['value'].transform(lambda x: x / x.sum())
        ds_result = self.group_data_ds.groupby('category')['value'].transform(lambda x: x / x.sum())
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_transform_center(self):
        """Test groupby().transform() - center within groups."""
        pd_result = self.group_data_pd.groupby('category')['value'].transform(lambda x: x - x.mean())
        ds_result = self.group_data_ds.groupby('category')['value'].transform(lambda x: x - x.mean())
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_transform_zscore(self):
        """Test groupby().transform() - z-score standardization."""
        pd_result = self.group_data_pd.groupby('category')['value'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        ds_result = self.group_data_ds.groupby('category')['value'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        np.testing.assert_array_almost_equal(ds_result.to_pandas().fillna(0).values, pd_result.fillna(0).values)

    def test_transform_rank(self):
        """Test groupby().transform() - rank within groups."""
        pd_result = self.group_data_pd.groupby('category')['value'].transform(lambda x: x.rank())
        ds_result = self.group_data_ds.groupby('category')['value'].transform(lambda x: x.rank())
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_transform_cumsum(self):
        """Test groupby().transform() - cumsum within groups."""
        pd_result = self.group_data_pd.groupby('category')['value'].transform(lambda x: x.cumsum())
        ds_result = self.group_data_ds.groupby('category')['value'].transform(lambda x: x.cumsum())
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_transform_string_func(self):
        """Test groupby().transform() with string function name."""
        pd_result = self.group_data_pd.groupby('category')['value'].transform('mean')
        ds_result = self.group_data_ds.groupby('category')['value'].transform('mean')
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_filter_by_mean(self):
        """Test groupby().filter() - filter groups by mean condition."""
        pd_result = self.group_data_pd.groupby('category').filter(lambda x: x['value'].mean() > 35)
        ds_result = self.group_data_ds.groupby('category').filter(lambda x: x['value'].mean() > 35)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_filter_by_size(self):
        """Test groupby().filter() - filter groups by size."""
        pd_result = self.group_data_pd.groupby('category').filter(lambda x: len(x) >= 3)
        ds_result = self.group_data_ds.groupby('category').filter(lambda x: len(x) >= 3)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_filter_compound_condition(self):
        """Test groupby().filter() - compound condition."""
        pd_result = self.group_data_pd.groupby('category').filter(
            lambda x: x['value'].mean() > 25 and x['value'].max() < 55
        )
        ds_result = self.group_data_ds.groupby('category').filter(
            lambda x: x['value'].mean() > 25 and x['value'].max() < 55
        )
        assert_frame_equal(ds_result.to_pandas(), pd_result)


class TestPivotAndReshaping(unittest.TestCase):
    """Test pivot tables and reshaping operations."""

    def setUp(self):
        self.pivot_data_pd = pd.DataFrame(
            {
                'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
                'product': ['A', 'B', 'A', 'B'],
                'sales': [100, 150, 120, 180],
            }
        )
        self.pivot_data_ds = ds.DataFrame(
            {
                'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
                'product': ['A', 'B', 'A', 'B'],
                'sales': [100, 150, 120, 180],
            }
        )

    def test_pivot_table(self):
        """Test pivot_table() - create pivot table."""
        pd_result = self.pivot_data_pd.pivot_table(values='sales', index='date', columns='product', aggfunc='sum')
        ds_result = self.pivot_data_ds.pivot_table(values='sales', index='date', columns='product', aggfunc='sum')
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_melt(self):
        """Test melt() - unpivot DataFrame."""
        pd_result = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).melt()
        ds_result = ds.DataFrame({'A': [1, 2], 'B': [3, 4]}).melt()
        assert_frame_equal(ds_result.to_pandas(), pd_result)


class TestConditionalLogic(unittest.TestCase):
    """Test conditional logic operations (where, mask, clip, between)."""

    def setUp(self):
        self.cond_data_pd = pd.DataFrame({'value': [10, 20, -5, 30, -10, 40]})
        self.cond_data_ds = ds.DataFrame({'value': [10, 20, -5, 30, -10, 40]})

    def test_where(self):
        """Test where() - conditional replacement."""
        pd_result = self.cond_data_pd['value'].where(self.cond_data_pd['value'] > 0, 0)
        ds_result = self.cond_data_ds['value'].where(self.cond_data_ds['value'] > 0, 0)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_mask(self):
        """Test mask() - inverse of where."""
        pd_result = self.cond_data_pd['value'].mask(self.cond_data_pd['value'] < 0, 0)
        ds_result = self.cond_data_ds['value'].mask(self.cond_data_ds['value'] < 0, 0)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_clip(self):
        """Test clip() - limit values to range."""
        pd_result = self.cond_data_pd['value'].clip(lower=0, upper=25)
        ds_result = self.cond_data_ds['value'].clip(lower=0, upper=25)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_between(self):
        """Test between() - check if values in range."""
        pd_result = self.cond_data_pd['value'].between(10, 30)
        ds_result = self.cond_data_ds['value'].between(10, 30)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)


class TestDataValidation(unittest.TestCase):
    """Test data validation operations."""

    def setUp(self):
        self.valid_data_pd = pd.DataFrame({'id': [1, 2, 3, 2, 4, 3], 'category': ['A', 'B', 'C', 'B', 'D', 'C']})
        self.valid_data_ds = ds.DataFrame({'id': [1, 2, 3, 2, 4, 3], 'category': ['A', 'B', 'C', 'B', 'D', 'C']})
        self.ts_data_pd = pd.DataFrame(
            {'value': [10, 20, 15, 30, 25, 35], 'price': [100.0, 102.5, 101.0, 105.0, 103.0, 107.5]}
        )
        self.ts_data_ds = ds.DataFrame(
            {'value': [10, 20, 15, 30, 25, 35], 'price': [100.0, 102.5, 101.0, 105.0, 103.0, 107.5]}
        )

    def test_isin(self):
        """Test isin() - check membership in list."""
        pd_result = self.valid_data_pd['category'].isin(['A', 'C'])
        ds_result = self.valid_data_ds['category'].isin(['A', 'C'])
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_nlargest(self):
        """Test nlargest() - get n largest values."""
        pd_result = self.ts_data_pd.nlargest(3, 'value')
        ds_result = self.ts_data_ds.nlargest(3, 'value')
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_nsmallest(self):
        """Test nsmallest() - get n smallest values."""
        pd_result = self.ts_data_pd.nsmallest(2, 'value')
        ds_result = self.ts_data_ds.nsmallest(2, 'value')
        assert_frame_equal(ds_result.to_pandas(), pd_result)


class TestAdvancedIndexing(unittest.TestCase):
    """Test advanced indexing operations (at, iat)."""

    def setUp(self):
        self.idx_data_pd = pd.DataFrame({'A': range(10), 'B': range(10, 20)}, index=list('abcdefghij'))
        self.idx_data_ds = ds.DataFrame({'A': range(10), 'B': range(10, 20)}, index=list('abcdefghij'))

    def test_at_scalar_access(self):
        """Test at[] - fast scalar value access."""
        pd_result = self.idx_data_pd.at['a', 'A']
        ds_result = self.idx_data_ds.at['a', 'A']
        self.assertEqual(ds_result, pd_result)

    def test_iat_integer_access(self):
        """Test iat[] - fast integer location access."""
        pd_result = self.idx_data_pd.iat[0, 0]
        ds_result = self.idx_data_ds.iat[0, 0]
        self.assertEqual(ds_result, pd_result)


if __name__ == '__main__':
    unittest.main()
