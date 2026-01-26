"""
Mirror Test for Perth House Prices EDA Notebook
==============================================

Tests pandas operations found in the perth-house-prices-eda.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Operations tested:
- fillna() with scalar and mode()
- drop(columns=...)
- to_datetime() (via pd.to_datetime)
- Column arithmetic
- corr()
- isna().sum()
- describe()
- info() (structural check only)

Design Principle:
    Tests use natural execution triggers following the lazy execution design.
    Avoid explicit _execute() calls - use natural triggers instead.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# ============================================================================
# Fixtures mimicking Perth house prices data structure
# ============================================================================


@pytest.fixture
def perth_sample_df():
    """Sample data mimicking Perth house prices dataset structure."""
    return pd.DataFrame({
        'ADDRESS': ['1 Main St', '2 Oak Ave', '3 Pine Rd', '4 Elm Ln', '5 Cedar Ct'],
        'SUBURB': ['Perth', 'Fremantle', 'Perth', 'Midland', 'Fremantle'],
        'PRICE': [500000, 650000, 420000, 380000, 720000],
        'BEDROOMS': [3, 4, 2, 3, 4],
        'BATHROOMS': [2, 2, 1, 1, 3],
        'GARAGE': [2.0, np.nan, 1.0, np.nan, 2.0],  # Has NaN values like original
        'LAND_AREA': [500, 600, 400, 550, 700],
        'FLOOR_AREA': [150, 200, 100, 130, 250],
        'BUILD_YEAR': [2000.0, np.nan, 1980.0, 1975.0, np.nan],  # Has NaN values
        'CBD_DIST': [5000, 15000, 3000, 20000, 16000],
        'NEAREST_SCH_RANK': [50.0, np.nan, 30.0, np.nan, 45.0],  # Mostly NaN like original
    })


@pytest.fixture
def numeric_df():
    """DataFrame for numeric operations like corr()."""
    return pd.DataFrame({
        'PRICE': [500000, 650000, 420000, 380000, 720000],
        'LAND_AREA': [500, 600, 400, 550, 700],
        'FLOOR_AREA': [150, 200, 100, 130, 250],
        'BEDROOMS': [3, 4, 2, 3, 4],
    })


# ============================================================================
# Test Classes
# ============================================================================


class TestFillnaOperations:
    """Tests for fillna operations as used in the notebook."""

    def test_fillna_with_string_value(self, perth_sample_df):
        """fillna('UNKNOWN') as used in notebook for GARAGE column."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['GARAGE'] = pd_df['GARAGE'].fillna('UNKNOWN')

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['GARAGE'] = ds['GARAGE'].fillna('UNKNOWN')

        # Compare - note dtype may differ due to mixed types
        assert_datastore_equals_pandas(ds, pd_df)

    def test_fillna_with_mode(self, perth_sample_df):
        """fillna(column.mode()[0]) as used in notebook for BUILD_YEAR."""
        # pandas
        pd_df = perth_sample_df.copy()
        mode_val = pd_df['BUILD_YEAR'].mode()[0]
        pd_df['BUILD_YEAR'] = pd_df['BUILD_YEAR'].fillna(mode_val)

        # DataStore - use same mode value for consistency
        ds = DataStore(perth_sample_df.copy())
        ds['BUILD_YEAR'] = ds['BUILD_YEAR'].fillna(mode_val)

        assert_datastore_equals_pandas(ds, pd_df)

    def test_fillna_numeric_value(self, perth_sample_df):
        """fillna with a numeric value."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['GARAGE'] = pd_df['GARAGE'].fillna(0)

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['GARAGE'] = ds['GARAGE'].fillna(0)

        assert_datastore_equals_pandas(ds, pd_df)


class TestDropOperations:
    """Tests for drop operations."""

    def test_drop_single_column(self, perth_sample_df):
        """drop(columns=['NEAREST_SCH_RANK']) as used in notebook."""
        # pandas
        pd_df = perth_sample_df.drop(columns=['NEAREST_SCH_RANK'])

        # DataStore
        ds = DataStore(perth_sample_df)
        ds_result = ds.drop(columns=['NEAREST_SCH_RANK'])

        assert_datastore_equals_pandas(ds_result, pd_df)

    def test_drop_multiple_columns(self, perth_sample_df):
        """Drop multiple columns at once."""
        # pandas
        pd_df = perth_sample_df.drop(columns=['NEAREST_SCH_RANK', 'ADDRESS'])

        # DataStore
        ds = DataStore(perth_sample_df)
        ds_result = ds.drop(columns=['NEAREST_SCH_RANK', 'ADDRESS'])

        assert_datastore_equals_pandas(ds_result, pd_df)


class TestColumnArithmetic:
    """Tests for column arithmetic operations."""

    def test_column_addition(self, perth_sample_df):
        """df['TotalRooms'] = df['BEDROOMS'] + df['BATHROOMS'] as in notebook."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['TotalRooms'] = pd_df['BEDROOMS'] + pd_df['BATHROOMS']

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['TotalRooms'] = ds['BEDROOMS'] + ds['BATHROOMS']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_column_subtraction(self, perth_sample_df):
        """Test column subtraction."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['room_diff'] = pd_df['BEDROOMS'] - pd_df['BATHROOMS']

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['room_diff'] = ds['BEDROOMS'] - ds['BATHROOMS']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_column_multiplication(self, perth_sample_df):
        """Test column multiplication with scalar."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['price_per_sqm'] = pd_df['PRICE'] / pd_df['FLOOR_AREA']

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['price_per_sqm'] = ds['PRICE'] / ds['FLOOR_AREA']

        assert_datastore_equals_pandas(ds, pd_df)


class TestIsnaSum:
    """Tests for isna().sum() pattern as used in notebook."""

    def test_isna_sum_single_column(self, perth_sample_df):
        """Test isna().sum() on single column."""
        # pandas
        pd_result = perth_sample_df['GARAGE'].isna().sum()

        # DataStore
        ds = DataStore(perth_sample_df)
        ds_result = ds['GARAGE'].isna().sum()

        assert ds_result == pd_result

    def test_isna_sum_all_columns(self, perth_sample_df):
        """Test isna().sum() on entire DataFrame."""
        # pandas
        pd_result = perth_sample_df.isna().sum()

        # DataStore
        ds = DataStore(perth_sample_df)
        ds_result = ds.isna().sum()

        # Compare values
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCorrelation:
    """Tests for correlation operations."""

    def test_corr_numeric_columns(self, numeric_df):
        """Test corr() on numeric columns as used in notebook."""
        # pandas
        pd_result = numeric_df.corr()

        # DataStore
        ds = DataStore(numeric_df)
        ds_result = ds.corr()

        # Correlation values should match within tolerance
        # Note: correlation is always between -1 and 1
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_corr_subset_columns(self, perth_sample_df):
        """Test corr() on subset of columns."""
        # pandas
        cols = ['PRICE', 'LAND_AREA', 'FLOOR_AREA']
        pd_result = perth_sample_df[cols].corr()

        # DataStore
        ds = DataStore(perth_sample_df)
        ds_result = ds[cols].corr()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestDescribe:
    """Tests for describe() operation."""

    def test_describe_default(self, numeric_df):
        """Test describe() with default parameters."""
        # pandas
        pd_result = numeric_df.describe()

        # DataStore
        ds = DataStore(numeric_df)
        ds_result = ds.describe()

        # Note: describe() returns statistics - compare with tolerance
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestInfoStructure:
    """Tests for structural operations like columns, len, etc."""

    def test_columns_property(self, perth_sample_df):
        """Test that columns property returns correct column names."""
        ds = DataStore(perth_sample_df)

        assert list(ds.columns) == list(perth_sample_df.columns)

    def test_len_function(self, perth_sample_df):
        """Test len() returns correct row count."""
        ds = DataStore(perth_sample_df)

        assert len(ds) == len(perth_sample_df)

    def test_shape_property(self, perth_sample_df):
        """Test shape property returns correct dimensions."""
        ds = DataStore(perth_sample_df)

        # Shape is a natural trigger
        assert ds.shape == perth_sample_df.shape


class TestCombinedOperations:
    """Tests for combined operations as seen in notebook flow."""

    def test_fillna_then_drop(self, perth_sample_df):
        """fillna + drop sequence as in notebook."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['GARAGE'] = pd_df['GARAGE'].fillna('UNKNOWN')
        pd_df['BUILD_YEAR'] = pd_df['BUILD_YEAR'].fillna(pd_df['BUILD_YEAR'].mode()[0])
        pd_df = pd_df.drop(columns=['NEAREST_SCH_RANK'])

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['GARAGE'] = ds['GARAGE'].fillna('UNKNOWN')
        ds['BUILD_YEAR'] = ds['BUILD_YEAR'].fillna(ds['BUILD_YEAR'].mode()[0])
        ds = ds.drop(columns=['NEAREST_SCH_RANK'])

        assert_datastore_equals_pandas(ds, pd_df)

    def test_add_column_then_filter(self, perth_sample_df):
        """Add computed column then filter."""
        # pandas
        pd_df = perth_sample_df.copy()
        pd_df['TotalRooms'] = pd_df['BEDROOMS'] + pd_df['BATHROOMS']
        pd_result = pd_df[pd_df['TotalRooms'] >= 5]

        # DataStore
        ds = DataStore(perth_sample_df.copy())
        ds['TotalRooms'] = ds['BEDROOMS'] + ds['BATHROOMS']
        ds_result = ds[ds['TotalRooms'] >= 5]

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
