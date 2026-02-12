"""
Mirror Test for Daily Climate EDA Notebook
==========================================

Tests pandas operations found in the daily-climate-eda.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Operations tested:
- Basic DataFrame operations (columns, len, shape)
- isna().sum()
- duplicated().sum()
- describe()
- sort_values()
- corr()
- Column selection

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
# Fixtures mimicking Delhi climate data structure
# ============================================================================


@pytest.fixture
def climate_df():
    """Sample data mimicking Delhi climate dataset structure."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2017-01-01', '2017-01-02', '2017-01-03',
                                '2017-01-04', '2017-01-05', '2017-01-06',
                                '2017-01-07', '2017-01-08', '2017-01-09', '2017-01-10']),
        'meantemp': [15.9, 18.5, 17.1, 18.7, 18.4, 16.2, 19.3, 20.1, 17.8, 15.5],
        'humidity': [85.9, 77.2, 81.9, 70.1, 74.9, 82.3, 68.5, 65.2, 79.8, 88.1],
        'wind_speed': [2.7, 2.9, 4.0, 4.5, 3.3, 3.1, 5.2, 4.8, 3.6, 2.4],
        'meanpressure': [1015.0, 1018.3, 1018.3, 1015.7, 1014.3, 1016.5, 1012.8, 1011.2, 1017.1, 1019.5],
    })


@pytest.fixture
def climate_with_nulls():
    """Climate data with some null values for testing isna()."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2017-01-01', '2017-01-02', '2017-01-03',
                                '2017-01-04', '2017-01-05']),
        'meantemp': [15.9, np.nan, 17.1, 18.7, np.nan],
        'humidity': [85.9, 77.2, np.nan, 70.1, 74.9],
        'wind_speed': [2.7, 2.9, 4.0, np.nan, 3.3],
        'meanpressure': [1015.0, 1018.3, 1018.3, 1015.7, 1014.3],
    })


@pytest.fixture
def climate_with_duplicates():
    """Climate data with duplicate rows."""
    return pd.DataFrame({
        'date': pd.to_datetime(['2017-01-01', '2017-01-02', '2017-01-01',  # duplicate
                                '2017-01-03', '2017-01-02']),  # duplicate
        'meantemp': [15.9, 18.5, 15.9, 17.1, 18.5],  # matching duplicates
        'humidity': [85.9, 77.2, 85.9, 81.9, 77.2],
    })


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicOperations:
    """Tests for basic DataFrame inspection operations."""

    def test_columns_property(self, climate_df):
        """Test columns property returns correct column names."""
        ds = DataStore(climate_df)
        assert list(ds.columns) == list(climate_df.columns)

    def test_len_function(self, climate_df):
        """Test len() returns correct row count."""
        ds = DataStore(climate_df)
        assert len(ds) == len(climate_df)

    def test_shape_property(self, climate_df):
        """Test shape property returns correct dimensions."""
        ds = DataStore(climate_df)
        assert ds.shape == climate_df.shape


class TestIsnaOperations:
    """Tests for isna() operations as used in notebook."""

    def test_isna_sum_no_nulls(self, climate_df):
        """isna().sum() when no null values exist."""
        # pandas
        pd_result = climate_df.isna().sum()

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds.isna().sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_sum_with_nulls(self, climate_with_nulls):
        """isna().sum() when null values exist."""
        # pandas
        pd_result = climate_with_nulls.isna().sum()

        # DataStore
        ds = DataStore(climate_with_nulls)
        ds_result = ds.isna().sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_single_column(self, climate_with_nulls):
        """Test isna() on single column."""
        # pandas
        pd_result = climate_with_nulls['meantemp'].isna()

        # DataStore
        ds = DataStore(climate_with_nulls)
        ds_result = ds['meantemp'].isna()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDuplicatedOperations:
    """Tests for duplicated() operations."""

    def test_duplicated_sum_no_duplicates(self, climate_df):
        """duplicated().sum() when no duplicates exist."""
        # pandas
        pd_result = climate_df.duplicated().sum()

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds.duplicated().sum()

        assert ds_result == pd_result

    def test_duplicated_sum_with_duplicates(self, climate_with_duplicates):
        """duplicated().sum() when duplicates exist."""
        # pandas
        pd_result = climate_with_duplicates.duplicated().sum()

        # DataStore
        ds = DataStore(climate_with_duplicates)
        ds_result = ds.duplicated().sum()

        assert ds_result == pd_result

    def test_duplicated_returns_series(self, climate_with_duplicates):
        """Test duplicated() returns boolean series."""
        # pandas
        pd_result = climate_with_duplicates.duplicated()

        # DataStore
        ds = DataStore(climate_with_duplicates)
        ds_result = ds.duplicated()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDescribeOperations:
    """Tests for describe() operation."""

    def test_describe_numeric_columns(self, climate_df):
        """Test describe() on numeric columns."""
        # pandas - select only numeric columns for describe
        numeric_cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
        pd_result = climate_df[numeric_cols].describe()

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds[numeric_cols].describe()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestSortOperations:
    """Tests for sort_values() operations."""

    def test_sort_by_single_column(self, climate_df):
        """sort_values('date') as used in notebook."""
        # pandas
        pd_result = climate_df.sort_values('date')

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds.sort_values('date')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_by_numeric_column(self, climate_df):
        """Sort by numeric column."""
        # pandas
        pd_result = climate_df.sort_values('meantemp')

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds.sort_values('meantemp')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending(self, climate_df):
        """Sort in descending order."""
        # pandas
        pd_result = climate_df.sort_values('humidity', ascending=False)

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds.sort_values('humidity', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCorrelationOperations:
    """Tests for correlation operations."""

    def test_corr_all_numeric(self, climate_df):
        """corr() on numeric columns as used in notebook."""
        # pandas - select only numeric columns
        numeric_cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
        pd_result = climate_df[numeric_cols].corr()

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds[numeric_cols].corr()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestColumnSelection:
    """Tests for column selection operations."""

    def test_select_single_column(self, climate_df):
        """Select single column."""
        # pandas
        pd_result = climate_df['meantemp']

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds['meantemp']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_multiple_columns(self, climate_df):
        """Select multiple columns as in notebook: df[['meantemp', 'humidity', ...]]"""
        # pandas
        cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
        pd_result = climate_df[cols]

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds[cols]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCombinedOperations:
    """Tests for combined operation chains as seen in notebook."""

    def test_sort_then_select(self, climate_df):
        """Sort then select columns."""
        # pandas
        pd_result = climate_df.sort_values('date')[['date', 'meantemp']]

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds.sort_values('date')[['date', 'meantemp']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_then_corr(self, climate_df):
        """Select columns then compute correlation."""
        # pandas
        cols = ['meantemp', 'humidity']
        pd_result = climate_df[cols].corr()

        # DataStore
        ds = DataStore(climate_df)
        ds_result = ds[cols].corr()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
