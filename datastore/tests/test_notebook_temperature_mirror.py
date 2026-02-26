"""
Mirror Test for Daily Temperature Forecasting Notebooks
========================================================

Tests pandas operations found in temperature forecasting notebooks,
comparing DataStore behavior with pandas for API consistency.

Operations tested:
- Column selection with list
- pd.to_datetime() conversion
- dropna()
- set_index()
- resample('D').mean() - daily resampling with aggregation
- reset_index()
- Column renaming via df.columns assignment
- Column arithmetic (subtraction)
- head() / tail()
- Negative slicing
- .values property
- abs().mean() chain
- idxmin() / min()

Design Principle:
    Tests use natural execution triggers following the lazy execution design.
    Avoid explicit _execute() calls - use natural triggers instead.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore, to_datetime
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


# ============================================================================
# Fixtures mimicking temperature data structure
# ============================================================================


@pytest.fixture
def temperature_df():
    """
    Sample data mimicking temperature dataset structure with multiple locations.
    Temperature values are in Kelvin (like the notebook).
    Creates hourly data for a few days to test resampling.
    """
    # Create hourly timestamps for 5 days
    dates = pd.date_range(start='2023-01-01', periods=120, freq='h')
    np.random.seed(42)  # For reproducibility

    # Temperature in Kelvin (around 280-290K = 7-17C)
    new_york_temps = np.random.normal(285, 5, 120)
    los_angeles_temps = np.random.normal(290, 3, 120)
    chicago_temps = np.random.normal(280, 7, 120)
    houston_temps = np.random.normal(288, 4, 120)

    # Add some NaN values like in real data
    new_york_temps[10] = np.nan
    new_york_temps[50] = np.nan
    los_angeles_temps[30] = np.nan

    return pd.DataFrame({
        'datetime': dates,
        'New York': new_york_temps,
        'Los Angeles': los_angeles_temps,
        'Chicago': chicago_temps,
        'Houston': houston_temps,
    })


@pytest.fixture
def temperature_with_nulls():
    """Temperature data with null values for testing dropna."""
    dates = pd.date_range('2023-01-01', periods=10, freq='h')
    return pd.DataFrame({
        'datetime': dates,
        'New York': [270.0, 271.0, np.nan, 273.0, 274.0, np.nan, 276.0, 277.0, 278.0, np.nan],
        'Los Angeles': [285.0, np.nan, 287.0, 288.0, np.nan, 290.0, 291.0, 292.0, np.nan, 294.0],
    })


@pytest.fixture
def simple_resample_df():
    """Simple DataFrame for resample tests."""
    # Create data with clear daily patterns for easier verification
    dates = pd.date_range('2023-01-01', periods=48, freq='h')  # 2 days of hourly data
    return pd.DataFrame({
        'datetime': dates,
        'value': list(range(48)),  # Simple increasing values
    })


@pytest.fixture
def temperature_df_string_datetime():
    """Temperature data with datetime as string (like CSV read)."""
    dates = pd.date_range(start='2023-01-01', periods=120, freq='h')
    np.random.seed(42)

    new_york_temps = np.random.normal(285, 5, 120)
    los_angeles_temps = np.random.normal(290, 3, 120)
    chicago_temps = np.random.normal(280, 7, 120)

    new_york_temps[10] = np.nan
    new_york_temps[50] = np.nan
    los_angeles_temps[30] = np.nan

    return pd.DataFrame({
        'datetime': dates.astype(str),  # String format like CSV
        'New York': new_york_temps,
        'Los Angeles': los_angeles_temps,
        'Chicago': chicago_temps,
    })


# ============================================================================
# Test Classes
# ============================================================================


class TestColumnSelection:
    """Tests for column selection operations as used in notebook."""

    def test_select_single_column(self, temperature_df):
        """Select single column from DataFrame."""
        # pandas
        pd_result = temperature_df['New York']

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds['New York']

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_multiple_columns_list(self, temperature_df):
        """df[['datetime', 'New York']] as used in notebook."""
        # pandas
        pd_result = temperature_df[['datetime', 'New York']]

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds[['datetime', 'New York']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_columns_property(self, temperature_df):
        """Test columns property for list comprehension filtering."""
        # pandas
        pd_cols = list(temperature_df.columns)

        # DataStore
        ds = DataStore(temperature_df)
        ds_cols = list(ds.columns)

        assert ds_cols == pd_cols

    def test_column_filter_comprehension(self, temperature_df):
        """[col for col in df.columns if col != 'datetime'] as used in notebook."""
        # pandas
        pd_cols = [col for col in temperature_df.columns if col != 'datetime']

        # DataStore
        ds = DataStore(temperature_df)
        ds_cols = [col for col in ds.columns if col != 'datetime']

        assert ds_cols == pd_cols

    def test_select_and_dropna(self, temperature_df):
        """df[['datetime', location]].dropna() pattern."""
        location = 'New York'

        # pandas
        pd_result = temperature_df[['datetime', location]].dropna()

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds[['datetime', location]].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDropnaOperations:
    """Tests for dropna operations as used in notebook."""

    def test_dropna_single_column_selection(self, temperature_with_nulls):
        """df[['datetime', location]].dropna() as used in notebook."""
        # pandas
        pd_result = temperature_with_nulls[['datetime', 'New York']].dropna()

        # DataStore
        ds = DataStore(temperature_with_nulls)
        ds_result = ds[['datetime', 'New York']].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_multiple_columns(self, temperature_with_nulls):
        """Test dropna on multiple columns."""
        # pandas
        pd_result = temperature_with_nulls[['datetime', 'New York', 'Los Angeles']].dropna()

        # DataStore
        ds = DataStore(temperature_with_nulls)
        ds_result = ds[['datetime', 'New York', 'Los Angeles']].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIndexOperations:
    """Tests for index operations (set_index, reset_index)."""

    def test_set_index(self, temperature_df):
        """df.set_index('datetime') as used in notebook."""
        # pandas
        pd_df = temperature_df[['datetime', 'New York']].copy()
        pd_result = pd_df.set_index('datetime')

        # DataStore
        ds = DataStore(temperature_df[['datetime', 'New York']].copy())
        ds_result = ds.set_index('datetime')

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_reset_index(self, temperature_df):
        """Test reset_index after set_index."""
        # pandas
        pd_df = temperature_df[['datetime', 'New York']].copy()
        pd_result = pd_df.set_index('datetime').reset_index()

        # DataStore
        ds = DataStore(temperature_df[['datetime', 'New York']].copy())
        ds_result = ds.set_index('datetime').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_after_resample(self, temperature_df):
        """Full pipeline: select -> dropna -> set_index -> resample -> reset_index."""
        location = 'New York'

        # pandas
        pd_df = temperature_df[['datetime', location]].dropna()
        pd_df = pd_df.set_index('datetime')
        pd_df = pd_df.resample('D').mean()
        pd_result = pd_df.reset_index()

        # DataStore
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', location]].dropna()
        ds_df = ds_df.set_index('datetime')
        ds_df = ds_df.resample('D').mean()
        ds_result = ds_df.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestResampleOperations:
    """Tests for resample operations as used in notebook."""

    def test_resample_daily_mean(self, simple_resample_df):
        """df.resample('D').mean() as used in notebook for daily temperature average."""
        # pandas
        pd_df = simple_resample_df.copy()
        pd_df = pd_df.set_index('datetime')
        pd_result = pd_df.resample('D').mean()

        # DataStore
        ds = DataStore(simple_resample_df.copy())
        ds = ds.set_index('datetime')
        ds_result = ds.resample('D').mean()

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_resample_daily_mean_with_reset_index(self, simple_resample_df):
        """resample('D').mean().reset_index() as used in notebook."""
        # pandas
        pd_df = simple_resample_df.copy()
        pd_df = pd_df.set_index('datetime')
        pd_result = pd_df.resample('D').mean().reset_index()

        # DataStore
        ds = DataStore(simple_resample_df.copy())
        ds = ds.set_index('datetime')
        ds_result = ds.resample('D').mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_resample_with_column_selection(self, temperature_df):
        """Full pattern: select columns, set_index, resample, reset_index."""
        # pandas
        pd_df = temperature_df[['datetime', 'New York']].copy()
        pd_df = pd_df.set_index('datetime')
        pd_result = pd_df.resample('D').mean().reset_index()

        # DataStore
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', 'New York']]
        ds_df = ds_df.set_index('datetime')
        ds_result = ds_df.resample('D').mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnRenaming:
    """Tests for column renaming operations."""

    def test_rename_columns_via_assignment(self, simple_resample_df):
        """df.columns = ['ds', 'y'] as used in notebook."""
        # pandas
        pd_df = simple_resample_df.copy()
        pd_df = pd_df.set_index('datetime')
        pd_df = pd_df.resample('D').mean().reset_index()
        pd_df.columns = ['ds', 'y']

        # DataStore
        ds = DataStore(simple_resample_df.copy())
        ds = ds.set_index('datetime')
        ds = ds.resample('D').mean().reset_index()
        ds.columns = ['ds', 'y']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_rename_method(self, temperature_df):
        """Test rename method as alternative to columns assignment."""
        # pandas
        pd_df = temperature_df[['datetime', 'New York']].copy()
        pd_result = pd_df.rename(columns={'datetime': 'ds', 'New York': 'y'})

        # DataStore
        ds = DataStore(temperature_df[['datetime', 'New York']].copy())
        ds_result = ds.rename(columns={'datetime': 'ds', 'New York': 'y'})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticOperations:
    """Tests for column arithmetic operations."""

    def test_column_subtraction_scalar(self, temperature_df):
        """df['y'] = df['y'] - 273.15 (Kelvin to Celsius conversion) as used in notebook."""
        # pandas
        pd_df = temperature_df[['datetime', 'New York']].copy()
        pd_df['New York'] = pd_df['New York'] - 273.15

        # DataStore
        ds = DataStore(temperature_df[['datetime', 'New York']].copy())
        ds['New York'] = ds['New York'] - 273.15

        assert_datastore_equals_pandas(ds, pd_df)

    def test_column_addition_scalar(self, temperature_df):
        """Test column addition with scalar."""
        # pandas
        pd_df = temperature_df[['datetime', 'New York']].copy()
        pd_df['adjusted'] = pd_df['New York'] + 10

        # DataStore
        ds = DataStore(temperature_df[['datetime', 'New York']].copy())
        ds['adjusted'] = ds['New York'] + 10

        assert_datastore_equals_pandas(ds, pd_df)

    def test_full_pipeline_with_conversion(self, temperature_df):
        """df['y'] = df['y'] - 273.15 pattern (Kelvin to Celsius) in full pipeline."""
        location = 'New York'

        # pandas
        pd_df = temperature_df[['datetime', location]].dropna()
        pd_df = pd_df.set_index('datetime')
        pd_df = pd_df.resample('D').mean()
        pd_df = pd_df.reset_index()
        pd_df.columns = ['ds', 'y']
        pd_df['y'] = pd_df['y'] - 273.15

        # DataStore
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', location]].dropna()
        ds_df = ds_df.set_index('datetime')
        ds_df = ds_df.resample('D').mean()
        ds_df = ds_df.reset_index()
        ds_df.columns = ['ds', 'y']
        ds_df['y'] = ds_df['y'] - 273.15

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestHeadTailOperations:
    """Tests for head() and tail() operations as used in notebook."""

    def test_head_default(self, temperature_df):
        """df.head() returns first 5 rows by default."""
        # pandas
        pd_result = temperature_df.head()

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds.head()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_with_n(self, temperature_df):
        """df.head(10) returns first 10 rows."""
        # pandas
        pd_result = temperature_df.head(10)

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_default(self, temperature_df):
        """df.tail() returns last 5 rows by default."""
        # pandas
        pd_result = temperature_df.tail()

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds.tail()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_with_n(self, temperature_df):
        """df.tail(10) returns last 10 rows."""
        # pandas
        pd_result = temperature_df.tail(10)

        # DataStore
        ds = DataStore(temperature_df)
        ds_result = ds.tail(10)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDatetimeConversion:
    """Tests for datetime conversion operations."""

    def test_to_datetime_column(self):
        """pd.to_datetime(df['datetime']) as used in notebook."""
        # Create DataFrame with string dates
        df = pd.DataFrame({
            'datetime': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'value': [1, 2, 3],
        })

        # pandas
        pd_df = df.copy()
        pd_df['datetime'] = pd.to_datetime(pd_df['datetime'])

        # DataStore
        ds = DataStore(df.copy())
        ds['datetime'] = to_datetime(ds['datetime'])

        assert_datastore_equals_pandas(ds, pd_df)

    def test_to_datetime_string_column(self, temperature_df_string_datetime):
        """pd.to_datetime(df['datetime']) conversion from string."""
        # pandas
        pd_df = temperature_df_string_datetime.copy()
        pd_df['datetime'] = pd.to_datetime(pd_df['datetime'])

        # DataStore - note: DataStore uses pd.to_datetime internally
        ds = DataStore(temperature_df_string_datetime.copy())
        ds['datetime'] = pd.to_datetime(ds['datetime'])

        assert_datastore_equals_pandas(ds, pd_df)


class TestSlicingOperations:
    """Tests for slicing operations as used in notebook."""

    def test_negative_slice_series(self, temperature_df):
        """df['y'][-365:] pattern to get last N rows."""
        location = 'New York'

        # pandas - use smaller slice since we have only 5 days
        pd_df = temperature_df[['datetime', location]].dropna()
        pd_df = pd_df.set_index('datetime')
        pd_df = pd_df.resample('D').mean()
        pd_df = pd_df.reset_index()
        pd_df.columns = ['ds', 'y']
        pd_df['y'] = pd_df['y'] - 273.15
        pd_result = pd_df['y'][-3:]

        # DataStore
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', location]].dropna()
        ds_df = ds_df.set_index('datetime')
        ds_df = ds_df.resample('D').mean()
        ds_df = ds_df.reset_index()
        ds_df.columns = ['ds', 'y']
        ds_df['y'] = ds_df['y'] - 273.15
        ds_result = ds_df['y'][-3:]

        # Compare Series
        assert_series_equal(ds_result, pd_result, check_names=False, check_index=False)

    def test_series_slice_values(self, temperature_df):
        """df['column'][-10:].values as used in notebook."""
        # pandas
        pd_values = temperature_df['New York'][-10:].values

        # DataStore
        ds = DataStore(temperature_df)
        ds_values = ds['New York'][-10:].values

        np.testing.assert_array_almost_equal(ds_values, pd_values)


class TestValueAccess:
    """Tests for accessing values from Series as used for MAE calculation."""

    def test_series_values_property(self, temperature_df):
        """df['column'].values as used in notebook for MAE calculation."""
        # pandas
        pd_values = temperature_df['New York'].values

        # DataStore
        ds = DataStore(temperature_df)
        ds_values = ds['New York'].values

        np.testing.assert_array_almost_equal(ds_values, pd_values)

    def test_values_after_pipeline(self, temperature_df):
        """df['y'].values pattern after full pipeline."""
        location = 'New York'

        # pandas
        pd_df = temperature_df[['datetime', location]].dropna()
        pd_df = pd_df.set_index('datetime')
        pd_df = pd_df.resample('D').mean()
        pd_df = pd_df.reset_index()
        pd_df.columns = ['ds', 'y']
        pd_df['y'] = pd_df['y'] - 273.15
        pd_values = pd_df['y'].values

        # DataStore
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', location]].dropna()
        ds_df = ds_df.set_index('datetime')
        ds_df = ds_df.resample('D').mean()
        ds_df = ds_df.reset_index()
        ds_df.columns = ['ds', 'y']
        ds_df['y'] = ds_df['y'] - 273.15
        ds_values = ds_df['y'].values

        # Both should be numpy arrays with same values
        np.testing.assert_array_almost_equal(ds_values, pd_values)


class TestAbsoluteMean:
    """Tests for abs().mean() chain as used in seasonality calculation."""

    def test_column_abs_mean(self):
        """df['column'].abs().mean() as used for seasonality calculation."""
        df = pd.DataFrame({
            'weekly': [-1.5, 2.3, -0.5, 1.2, -3.0],
            'yearly': [10.0, -5.0, 8.0, -2.0, 4.0],
        })

        # pandas
        pd_result = df['weekly'].abs().mean()

        # DataStore
        ds = DataStore(df)
        ds_result = ds['weekly'].abs().mean()

        assert abs(float(ds_result) - pd_result) < 1e-10

    def test_dataframe_abs_mean(self):
        """df[['col1', 'col2']].abs().mean() for multiple columns."""
        df = pd.DataFrame({
            'weekly': [-1.5, 2.3, -0.5, 1.2, -3.0],
            'yearly': [10.0, -5.0, 8.0, -2.0, 4.0],
        })

        # pandas
        pd_result = df[['weekly', 'yearly']].abs().mean()

        # DataStore
        ds = DataStore(df)
        ds_result = ds[['weekly', 'yearly']].abs().mean()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataFrameFromDict:
    """Tests for creating DataFrame from list of dicts as used in summary aggregation."""

    def test_dataframe_from_list_of_dicts(self):
        """pd.DataFrame(list_of_dicts) as used in temperature_forecast summary."""
        summary = [
            {'location': 'New York', 'mae': 1.5, 'weekly_seasonality': 0.3, 'yearly_seasonality': 5.2},
            {'location': 'Los Angeles', 'mae': 1.2, 'weekly_seasonality': 0.2, 'yearly_seasonality': 4.8},
            {'location': 'Chicago', 'mae': 1.8, 'weekly_seasonality': 0.4, 'yearly_seasonality': 6.1},
        ]

        # pandas
        pd_df = pd.DataFrame(summary)

        # DataStore
        ds = DataStore(summary)

        assert_datastore_equals_pandas(ds, pd_df)


class TestMinMaxComparison:
    """Tests for finding min/max values as used in finding most predictable location."""

    def test_find_min_value_in_series(self):
        """Find minimum MAE value as done in temperature_forecast."""
        df = pd.DataFrame({
            'location': ['New York', 'Los Angeles', 'Chicago'],
            'mae': [1.5, 1.2, 1.8],
        })

        # pandas
        pd_min = df['mae'].min()

        # DataStore
        ds = DataStore(df)
        ds_min = ds['mae'].min()

        assert abs(float(ds_min) - pd_min) < 1e-10

    def test_idxmin_operation(self):
        """df['mae'].idxmin() to find location with lowest MAE."""
        df = pd.DataFrame({
            'location': ['New York', 'Los Angeles', 'Chicago'],
            'mae': [1.5, 1.2, 1.8],
        })

        # pandas
        pd_idx = df['mae'].idxmin()

        # DataStore
        ds = DataStore(df)
        ds_idx = ds['mae'].idxmin()

        assert ds_idx == pd_idx


class TestFullPipeline:
    """Tests for the full notebook pipeline pattern."""

    def test_full_temperature_preprocessing_pipeline(self, temperature_df):
        """
        Full pipeline as used in notebook:
        1. Select columns [datetime, location]
        2. dropna()
        3. set_index('datetime')
        4. resample('D').mean()
        5. reset_index()
        6. rename columns to ['ds', 'y']
        7. subtract 273.15 for Kelvin to Celsius
        """
        location = 'New York'

        # pandas
        pd_df = temperature_df[['datetime', location]].dropna().copy()
        pd_df.set_index('datetime', inplace=True)
        pd_daily = pd_df.resample('D').mean()
        pd_daily = pd_daily.reset_index()
        pd_daily.columns = ['ds', 'y']
        pd_daily['y'] = pd_daily['y'] - 273.15

        # DataStore - mirroring the same operations
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', location]].dropna()
        ds_df = ds_df.set_index('datetime')
        ds_daily = ds_df.resample('D').mean()
        ds_daily = ds_daily.reset_index()
        ds_daily.columns = ['ds', 'y']
        ds_daily['y'] = ds_daily['y'] - 273.15

        assert_datastore_equals_pandas(ds_daily, pd_daily)

    def test_pipeline_without_inplace(self, temperature_df):
        """Same pipeline but without inplace operations."""
        location = 'New York'

        # pandas - chain without inplace
        pd_df = temperature_df[['datetime', location]].dropna()
        pd_df = pd_df.set_index('datetime')
        pd_daily = pd_df.resample('D').mean().reset_index()
        pd_daily.columns = ['ds', 'y']
        pd_daily['y'] = pd_daily['y'] - 273.15

        # DataStore
        ds = DataStore(temperature_df)
        ds_df = ds[['datetime', location]].dropna()
        ds_df = ds_df.set_index('datetime')
        ds_daily = ds_df.resample('D').mean().reset_index()
        ds_daily.columns = ['ds', 'y']
        ds_daily['y'] = ds_daily['y'] - 273.15

        assert_datastore_equals_pandas(ds_daily, pd_daily)

    def test_multiple_locations_processing(self, temperature_df):
        """Test processing for different locations."""
        for location in ['New York', 'Los Angeles', 'Chicago']:
            # pandas
            pd_df = temperature_df[['datetime', location]].dropna()
            pd_df = pd_df.set_index('datetime')
            pd_df = pd_df.resample('D').mean()
            pd_df = pd_df.reset_index()
            pd_df.columns = ['ds', 'y']
            pd_df['y'] = pd_df['y'] - 273.15

            # DataStore
            ds = DataStore(temperature_df)
            ds_df = ds[['datetime', location]].dropna()
            ds_df = ds_df.set_index('datetime')
            ds_df = ds_df.resample('D').mean()
            ds_df = ds_df.reset_index()
            ds_df.columns = ['ds', 'y']
            ds_df['y'] = ds_df['y'] - 273.15

            assert_datastore_equals_pandas(ds_df, pd_df, msg=f"Failed for location: {location}")


class TestMultiLocationIteration:
    """Tests for iterating over multiple locations as used in temperature_forecast function."""

    def test_iterate_columns_process_each(self, temperature_df):
        """
        Process each location column independently as done in temperature_forecast.

        Pattern:
        for location in locations:
            df[['datetime', location]].dropna()
        """
        locations = ['New York', 'Los Angeles', 'Chicago', 'Houston']

        for location in locations:
            # pandas
            pd_result = temperature_df[['datetime', location]].dropna()

            # DataStore
            ds = DataStore(temperature_df)
            ds_result = ds[['datetime', location]].dropna()

            assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_operations_multiple_locations(self, temperature_df):
        """Test the pattern used in temperature_forecast function."""
        locations = [col for col in temperature_df.columns if col != 'datetime']

        for location in locations:
            # pandas
            pd_selected = temperature_df[['datetime', location]].dropna()
            pd_selected = pd_selected.set_index('datetime')
            pd_daily = pd_selected.resample('D').mean()
            pd_daily = pd_daily.reset_index()
            pd_daily.columns = ['ds', 'y']
            pd_daily['y'] = pd_daily['y'] - 273.15

            # DataStore
            ds = DataStore(temperature_df)
            ds_selected = ds[['datetime', location]].dropna()
            ds_selected = ds_selected.set_index('datetime')
            ds_daily = ds_selected.resample('D').mean()
            ds_daily = ds_daily.reset_index()
            ds_daily.columns = ['ds', 'y']
            ds_daily['y'] = ds_daily['y'] - 273.15

            assert_datastore_equals_pandas(ds_daily, pd_daily, msg=f"Failed for {location}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
