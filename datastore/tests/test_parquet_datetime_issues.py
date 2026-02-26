"""
Tests for Parquet file reading issues discovered during benchmarking.

Issues covered:
1. Row order preservation - chDB may read Parquet row groups in parallel, causing
   non-deterministic row order. Fixed by setting input_format_parquet_preserve_order=1.

2. DateTime timezone handling - New versions of chDB return tz-aware datetime
   (e.g., +08:00 system timezone), while Pandas returns tz-naive datetime.
   Fixed by setting session_timezone='UTC'.

3. DateTime column replacement in where/mask - When condition is False, Pandas
   replaces datetime values with scalar (e.g., 0), creating object column with
   mixed types. DataStore returns NaT instead. This is a design difference that
   is currently bypassed in benchmark verification.

Reference: benchmark_datastore_vs_pandas.py verification
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from tests.xfail_markers import design_datetime_fillna_nat

from datastore import DataStore


class TestParquetRowOrderPreservation:
    """
    Test that Parquet file row order is preserved when reading.

    chDB may read row groups in parallel, which can reorder rows.
    This is fixed by setting input_format_parquet_preserve_order=1
    in DataStore.from_file() for Parquet files.
    """

    def test_parquet_row_order_preserved_small(self):
        """Test row order preservation on small dataset (within single row group)."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Read with DataStore - must select columns explicitly
            ds = DataStore.from_file(path)
            ds_result = ds[['id', 'value']].to_df()

            # Row order should match
            np.testing.assert_array_equal(ds_result['id'].values, df['id'].values)

    def test_parquet_row_order_preserved_large(self):
        """Test row order preservation on large dataset (multiple row groups)."""
        np.random.seed(42)
        n = 100000  # Large enough to span multiple row groups
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Read with DataStore - must select columns explicitly
            ds = DataStore.from_file(path)
            ds_result = ds[['id', 'value', 'str_col']].to_df()

            # Row order should match
            np.testing.assert_array_equal(ds_result['id'].values, df['id'].values)

    def test_parquet_row_order_preserved_after_filter(self):
        """Test row order preservation after filtering."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'value': np.random.randint(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas filter
            pd_result = df[df['value'] > 500]

            # DataStore filter
            ds = DataStore.from_file(path)
            ds_result = ds[ds['value'] > 500].to_df()

            # Row order should match after filter
            # Note: DataStore returns continuous index (0,1,2,...) while pandas
            # preserves original index. This is a known design difference because
            # SQL has no index concept. We reset_index to compare row ORDER not index.
            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

    def test_parquet_row_order_preserved_chain_filters(self):
        """Test row order preservation after chained filters."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'float_col': np.random.uniform(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas - chain 5 filters
            pd_result = df[df['int_col'] > 200]
            pd_result = pd_result[pd_result['int_col'] < 800]
            pd_result = pd_result[pd_result['float_col'] > 100]
            pd_result = pd_result[pd_result['float_col'] < 900]
            pd_result = pd_result[pd_result['str_col'].isin(['A', 'B', 'C'])]

            # DataStore - chain 5 filters
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 200]
            ds_result = ds_result[ds_result['int_col'] < 800]
            ds_result = ds_result[ds_result['float_col'] > 100]
            ds_result = ds_result[ds_result['float_col'] < 900]
            ds_result = ds_result[ds_result['str_col'].isin(['A', 'B', 'C'])].to_df()

            # Row order should match
            # Note: reset_index compares row ORDER (DataStore has no pandas index concept)
            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)


class TestDatetimeTimezoneHandling:
    """
    Test that datetime values are correctly handled with timezone settings.

    New versions of chDB may return tz-aware datetime values using the system
    timezone (e.g., +08:00). Pandas returns tz-naive datetime (implicit UTC).
    This is fixed by setting session_timezone='UTC' in DataStore.from_file().
    """

    def test_datetime_column_matches_pandas_utc(self):
        """Test datetime column values match Pandas (tz-naive UTC)."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Read with DataStore - must select columns explicitly
            ds = DataStore.from_file(path)
            ds_result = ds[['id', 'date_col', 'value']].to_df()

            # Datetime values should match
            pd_ts = pd.to_datetime(df['date_col'])
            ds_ts = pd.to_datetime(ds_result['date_col'])

            # If DataStore returns tz-aware, convert to tz-naive for comparison
            if hasattr(ds_ts.dt, 'tz') and ds_ts.dt.tz is not None:
                ds_ts = ds_ts.dt.tz_convert('UTC').dt.tz_localize(None)

            np.testing.assert_array_equal(ds_ts.values, pd_ts.values)

    def test_datetime_column_after_filter(self):
        """Test datetime column values after filtering."""
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame(
            {
                'id': range(n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
                'value': np.random.randint(0, 1000, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas filter
            pd_result = df[df['value'] > 500]

            # DataStore filter
            ds = DataStore.from_file(path)
            ds_result = ds[ds['value'] > 500].to_df()

            # Reset index for comparison (DataStore returns continuous index, not pandas original)
            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Compare datetime values
            pd_ts = pd.to_datetime(pd_result['date_col'])
            ds_ts = pd.to_datetime(ds_result['date_col'])

            if hasattr(ds_ts.dt, 'tz') and ds_ts.dt.tz is not None:
                ds_ts = ds_ts.dt.tz_convert('UTC').dt.tz_localize(None)

            np.testing.assert_array_equal(ds_ts.values, pd_ts.values)

    def test_datetime_column_after_sort(self):
        """Test datetime column values after sorting."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                'id': range(n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas sort
            pd_result = df.sort_values('value', ascending=False, kind='stable').head(100)

            # DataStore sort
            ds = DataStore.from_file(path)
            ds_result = ds.sort_values('value', ascending=False, kind='stable').head(100).to_df()

            # Reset index for comparison (DataStore returns continuous index, not pandas original)
            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Compare datetime values
            pd_ts = pd.to_datetime(pd_result['date_col'])
            ds_ts = pd.to_datetime(ds_result['date_col'])

            if hasattr(ds_ts.dt, 'tz') and ds_ts.dt.tz is not None:
                ds_ts = ds_ts.dt.tz_convert('UTC').dt.tz_localize(None)

            np.testing.assert_array_equal(ds_ts.values, pd_ts.values)


class TestDatetimeWhereMaskDesignDifference:
    """
    Test datetime column behavior in where/mask operations.

    Design difference:
    - Pandas: replaces datetime values with scalar (e.g., 0), creating object column
    - DataStore: replaces datetime values with NaT

    These tests are marked xfail because this is a known design difference
    that is currently bypassed in the benchmark verification.
    """

    @design_datetime_fillna_nat
    def test_where_datetime_column_replaced_with_zero(self):
        """
        Test where() on DataFrame with datetime column - Pandas replaces with 0.

        Pandas behavior: datetime values where condition is False are replaced
        with the scalar value (0), creating an object column with mixed types.

        DataStore behavior: datetime values are replaced with NaT.

        This is a design difference that we document but don't fix.
        """
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                'id': range(n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Store original condition mask before applying where operation
            condition_mask = df['value'] > 50

            # Pandas where - replaces datetime with 0 (creates object column)
            pd_result = df.where(condition_mask, 0)

            # DataStore where
            ds = DataStore.from_file(path)
            ds_result = ds.where(ds['value'] > 50, 0).to_df()

            # This will fail because Pandas has 0 and DataStore has NaT
            # for datetime column where condition is False
            assert pd_result['date_col'].dtype == object  # Pandas: mixed type
            # DataStore: datetime with NaT or similar

            # Check actual values match (this is where the difference is)
            for i in range(len(pd_result)):
                if not condition_mask.iloc[i]:
                    # Pandas has 0, DataStore has NaT - this will fail
                    assert (
                        pd_result['date_col'].iloc[i] == ds_result['date_col'].iloc[i]
                    ), f"Row {i}: Pandas={pd_result['date_col'].iloc[i]}, DataStore={ds_result['date_col'].iloc[i]}"

    @design_datetime_fillna_nat
    def test_mask_datetime_column_replaced_with_minus_one(self):
        """
        Test mask() on DataFrame with datetime column - Pandas replaces with -1.

        Similar design difference as where().
        """
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                'id': range(n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
                'value': np.random.randint(0, 100, n),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Store original condition mask before applying mask operation
            condition_mask = df['value'] > 50

            # Pandas mask - replaces datetime with -1 where condition is True
            pd_result = df.mask(condition_mask, -1)

            # DataStore mask
            ds = DataStore.from_file(path)
            ds_result = ds.mask(ds['value'] > 50, -1).to_df()

            # This will fail because Pandas has -1 and DataStore has NaT
            # for datetime column where condition is True
            for i in range(len(pd_result)):
                if condition_mask.iloc[i]:
                    # Pandas has -1, DataStore has NaT - this will fail
                    assert (
                        pd_result['date_col'].iloc[i] == ds_result['date_col'].iloc[i]
                    ), f"Row {i}: Pandas={pd_result['date_col'].iloc[i]}, DataStore={ds_result['date_col'].iloc[i]}"


class TestFromFileSetsCorrectSettings:
    """
    Test that from_file() correctly sets the required format settings
    to handle Parquet row order and datetime timezone issues.
    """

    def test_from_file_sets_session_timezone_utc(self):
        """Test that from_file() sets session_timezone='UTC'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)

            ds = DataStore.from_file(path)

            # Check that session_timezone is set to UTC
            assert ds._format_settings.get('session_timezone') == 'UTC'

    def test_from_file_sets_parquet_preserve_order(self):
        """Test that from_file() sets input_format_parquet_preserve_order=1 for Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)

            ds = DataStore.from_file(path)

            # Check that preserve order is set for Parquet
            assert ds._format_settings.get('input_format_parquet_preserve_order') == 1

    def test_from_file_csv_no_preserve_order(self):
        """Test that from_file() does NOT set preserve_order for CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            pd.DataFrame({'a': [1, 2, 3]}).to_csv(path, index=False)

            ds = DataStore.from_file(path)

            # CSV files should not have preserve_order setting
            # (only Parquet needs it due to row group parallelism)
            assert ds._format_settings.get('input_format_parquet_preserve_order') is None

    def test_from_file_csv_still_has_timezone_utc(self):
        """Test that from_file() sets session_timezone='UTC' for CSV too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            pd.DataFrame({'a': [1, 2, 3]}).to_csv(path, index=False)

            ds = DataStore.from_file(path)

            # All file types should have UTC timezone
            assert ds._format_settings.get('session_timezone') == 'UTC'


class TestIntegrationBenchmarkScenarios:
    """
    Integration tests reproducing exact scenarios from benchmark
    that were failing due to these issues.
    """

    def test_filter_with_datetime_column(self):
        """Test filter operation with datetime column - from benchmark 'Filter (single)'."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df[df['int_col'] > 500]

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 500].to_df()

            # Reset index for comparison (DataStore returns continuous index, not pandas original)
            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Compare id (verifies row order)
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

            # Compare datetime - handle potential timezone difference
            pd_ts = pd.to_datetime(pd_result['date_col'])
            ds_ts = pd.to_datetime(ds_result['date_col'])
            if hasattr(ds_ts.dt, 'tz') and ds_ts.dt.tz is not None:
                ds_ts = ds_ts.dt.tz_convert('UTC').dt.tz_localize(None)
            np.testing.assert_array_equal(ds_ts.values, pd_ts.values)

    def test_head_limit_with_datetime_column(self):
        """Test head/limit operation with datetime column - from benchmark 'Head/Limit'."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas
            pd_result = df.head(1000)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds.head(1000).to_df()

            # Compare id
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

            # Compare datetime
            pd_ts = pd.to_datetime(pd_result['date_col'])
            ds_ts = pd.to_datetime(ds_result['date_col'])
            if hasattr(ds_ts.dt, 'tz') and ds_ts.dt.tz is not None:
                ds_ts = ds_ts.dt.tz_convert('UTC').dt.tz_localize(None)
            np.testing.assert_array_equal(ds_ts.values, pd_ts.values)

    def test_combined_ops_with_datetime_column(self):
        """Test combined operations with datetime column - from benchmark 'Combined ops'."""
        np.random.seed(42)
        n = 100000
        df = pd.DataFrame(
            {
                'id': range(n),
                'int_col': np.random.randint(0, 1000, n),
                'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
                'float_col': np.random.uniform(0, 1000, n),
                'date_col': pd.date_range('2020-01-01', periods=n, freq='s'),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            # Pandas - combined ops
            pd_result = df[df['int_col'] > 200]
            pd_result = pd_result[['id', 'int_col', 'str_col', 'float_col', 'date_col']]
            pd_result = pd_result.sort_values('int_col', ascending=False, kind='stable')
            pd_result = pd_result.head(100)

            # DataStore
            ds = DataStore.from_file(path)
            ds_result = ds[ds['int_col'] > 200]
            ds_result = ds_result[['id', 'int_col', 'str_col', 'float_col', 'date_col']]
            ds_result = ds_result.sort_values('int_col', ascending=False, kind='stable')
            ds_result = ds_result.head(100).to_df()

            # Reset index for comparison (DataStore returns continuous index, not pandas original)
            pd_result = pd_result.reset_index(drop=True)
            ds_result = ds_result.reset_index(drop=True)

            # Compare id
            np.testing.assert_array_equal(ds_result['id'].values, pd_result['id'].values)

            # Compare datetime
            pd_ts = pd.to_datetime(pd_result['date_col'])
            ds_ts = pd.to_datetime(ds_result['date_col'])
            if hasattr(ds_ts.dt, 'tz') and ds_ts.dt.tz is not None:
                ds_ts = ds_ts.dt.tz_convert('UTC').dt.tz_localize(None)
            np.testing.assert_array_equal(ds_ts.values, pd_ts.values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
