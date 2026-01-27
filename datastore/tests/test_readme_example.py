"""
Test for README example - verify consistency with pandas results.

This test validates that the README example produces identical results
to pandas, including:
- Numeric values
- Column names
- Column types
- Index values
"""

from tests.test_utils import get_dataframe, get_series
import numpy as np
import pandas as pd
import pytest

import datastore as ds


@pytest.fixture
def employee_data(tmp_path):
    """Create test employee data CSV file."""
    data = pd.DataFrame(
        {
            'age': [30, 25, 35, 28, 32, 26, 40, 22, 38, 29],
            'salary': [60000, 48000, 75000, 55000, 70000, 52000, 85000, 45000, 80000, 58000],
            'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA', 'NYC', 'LA'],
        }
    )
    csv_path = tmp_path / "employee_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path, data


class TestReadmeExample:
    """Test the main README example for pandas consistency."""

    def test_readme_example_full_pipeline(self, employee_data):
        """
        Test the complete README example pipeline:

        filtered = df[(df['age'] > 25) & (df['salary'] > 50000)]
        grouped = filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        sorted_df = grouped.sort_values('mean', ascending=False)
        result = sorted_df.head(10)
        """
        csv_path, raw_data = employee_data

        # === DataStore execution ===
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_sorted = ds_grouped.sort_values('mean', ascending=False)
        ds_result = ds_sorted.head(10)

        # Trigger execution and get DataFrame
        ds_result_df = ds_result.to_df()

        # === Pandas execution ===
        pd_df = raw_data.copy()
        pd_filtered = pd_df[(pd_df['age'] > 25) & (pd_df['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        pd_sorted = pd_grouped.sort_values('mean', ascending=False)
        pd_result = pd_sorted.head(10)

        # === Verify consistency ===

        # 1. Check column names
        assert list(ds_result_df.columns) == list(
            pd_result.columns
        ), f"Column names mismatch: {list(ds_result_df.columns)} vs {list(pd_result.columns)}"

        # 2. Check index values (city names)
        assert list(ds_result_df.index) == list(
            pd_result.index
        ), f"Index values mismatch: {list(ds_result_df.index)} vs {list(pd_result.index)}"

        # 3. Check index name
        assert (
            ds_result_df.index.name == pd_result.index.name
        ), f"Index name mismatch: {ds_result_df.index.name} vs {pd_result.index.name}"

        # 4. Check numeric values
        for col in pd_result.columns:
            np.testing.assert_array_almost_equal(
                ds_result_df[col].values, pd_result[col].values, decimal=5, err_msg=f"Values mismatch in column '{col}'"
            )

        # 5. Check column dtypes (numeric types should be compatible)
        for col in pd_result.columns:
            ds_dtype = ds_result_df[col].dtype
            pd_dtype = pd_result[col].dtype
            # Both should be numeric
            assert np.issubdtype(ds_dtype, np.number), f"Column '{col}' in datastore is not numeric: {ds_dtype}"
            assert np.issubdtype(pd_dtype, np.number), f"Column '{col}' in pandas is not numeric: {pd_dtype}"

    def test_readme_example_filter_only(self, employee_data):
        """Test just the filter operation."""
        csv_path, raw_data = employee_data

        # DataStore
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_result = ds_filtered.to_df()

        # Pandas
        pd_filtered = raw_data[(raw_data['age'] > 25) & (raw_data['salary'] > 50000)]

        # Reset index for comparison (SQL doesn't preserve original index)
        ds_result = ds_result.reset_index(drop=True)
        pd_result = pd_filtered.reset_index(drop=True)

        # Sort by all columns to ensure consistent ordering
        ds_result = ds_result.sort_values(by=list(ds_result.columns)).reset_index(drop=True)
        pd_result = pd_result.sort_values(by=list(pd_result.columns)).reset_index(drop=True)

        # Check column names
        assert list(ds_result.columns) == list(pd_result.columns)

        # Check values - handle numeric and non-numeric columns differently
        for col in pd_result.columns:
            if np.issubdtype(pd_result[col].dtype, np.number):
                np.testing.assert_array_almost_equal(
                    ds_result[col].values,
                    pd_result[col].values,
                    decimal=5,
                    err_msg=f"Values mismatch in column '{col}'",
                )
            else:
                # For string columns, compare directly
                np.testing.assert_array_equal(
                    ds_result[col].values, pd_result[col].values, err_msg=f"Values mismatch in column '{col}'"
                )

        # Check dtypes - numeric types should be numeric, string types should be object/string
        for col in pd_result.columns:
            ds_dtype = ds_result[col].dtype
            pd_dtype = pd_result[col].dtype
            if np.issubdtype(pd_dtype, np.number):
                assert np.issubdtype(ds_dtype, np.number), f"Dtype mismatch in column '{col}': {ds_dtype} vs {pd_dtype}"
            else:
                # String columns might have different representations (object vs string)
                assert (
                    ds_dtype == pd_dtype or ds_dtype.kind == pd_dtype.kind
                ), f"Dtype mismatch in column '{col}': {ds_dtype} vs {pd_dtype}"

    def test_readme_example_groupby_single_agg(self, employee_data):
        """Test groupby with single aggregation function."""
        csv_path, raw_data = employee_data

        # DataStore
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].mean()
        ds_result = get_dataframe(ds_grouped).values

        # Pandas
        pd_filtered = raw_data[(raw_data['age'] > 25) & (raw_data['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].mean()

        # Compare values (sorted by index for consistent ordering)
        ds_series = get_series(ds_grouped)

        # Sort both for comparison
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_grouped.sort_index()

        np.testing.assert_array_almost_equal(
            np.array(ds_sorted.values if hasattr(ds_sorted, 'values') else ds_sorted).flatten(),
            pd_sorted.values,
            decimal=5,
        )

    def test_readme_example_groupby_multi_agg(self, employee_data):
        """Test groupby with multiple aggregation functions."""
        csv_path, raw_data = employee_data

        # DataStore
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_result = ds_grouped.to_df()

        # Pandas
        pd_filtered = raw_data[(raw_data['age'] > 25) & (raw_data['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])

        # Sort both by index for comparison
        ds_result = ds_result.sort_index()
        pd_grouped = pd_grouped.sort_index()

        # Check column names
        assert list(ds_result.columns) == list(
            pd_grouped.columns
        ), f"Columns: {list(ds_result.columns)} vs {list(pd_grouped.columns)}"

        # Check index
        assert list(ds_result.index) == list(
            pd_grouped.index
        ), f"Index: {list(ds_result.index)} vs {list(pd_grouped.index)}"

        # Check values
        for col in pd_grouped.columns:
            np.testing.assert_array_almost_equal(
                ds_result[col].values, pd_grouped[col].values, decimal=5, err_msg=f"Values mismatch in column '{col}'"
            )

    def test_readme_example_sort_values(self, employee_data):
        """Test sort_values after groupby."""
        csv_path, raw_data = employee_data

        # DataStore
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_sorted = ds_grouped.sort_values('mean', ascending=False)
        ds_result = ds_sorted.to_df()

        # Pandas
        pd_filtered = raw_data[(raw_data['age'] > 25) & (raw_data['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        pd_sorted = pd_grouped.sort_values('mean', ascending=False)

        # Check order is the same
        assert list(ds_result.index) == list(
            pd_sorted.index
        ), f"Sort order mismatch: {list(ds_result.index)} vs {list(pd_sorted.index)}"

        # Check values
        for col in pd_sorted.columns:
            np.testing.assert_array_almost_equal(ds_result[col].values, pd_sorted[col].values, decimal=5)

    def test_readme_example_head(self, employee_data):
        """Test head() limit."""
        csv_path, raw_data = employee_data

        # DataStore
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_sorted = ds_grouped.sort_values('mean', ascending=False)
        ds_result = ds_sorted.head(1).to_df()

        # Pandas
        pd_filtered = raw_data[(raw_data['age'] > 25) & (raw_data['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        pd_sorted = pd_grouped.sort_values('mean', ascending=False)
        pd_result = pd_sorted.head(1)

        # Check only 1 row returned
        assert len(ds_result) == 1
        assert len(pd_result) == 1

        # Check it's the same row
        assert list(ds_result.index) == list(pd_result.index)

        # Check values
        for col in pd_result.columns:
            np.testing.assert_array_almost_equal(ds_result[col].values, pd_result[col].values, decimal=5)


class TestReadmeExampleSQLCompilation:
    """Test that the README example compiles to SQL correctly."""

    def test_uses_sql_engine(self, employee_data):
        """Verify that the pipeline uses SQL engine (not pandas fallback)."""
        csv_path, _ = employee_data

        # Build the pipeline
        ds_df = ds.read_csv(str(csv_path))

        # Check that we have a table function (SQL source)
        assert ds_df._table_function is not None, "read_csv should use SQL engine by default"

        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]

        # Check filtered still has table function
        assert ds_filtered._table_function is not None, "Filter should preserve SQL source"

        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])

        # Check grouped is a DataStore (not LazySeries)
        assert hasattr(ds_grouped, '_table_function'), "Groupby agg should return DataStore for SQL compilation"

    def test_lazy_ops_chain(self, employee_data):
        """Verify that operations are recorded as lazy ops."""
        csv_path, _ = employee_data

        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_sorted = ds_grouped.sort_values('mean', ascending=False)
        ds_result = ds_sorted.head(10)

        # Check lazy ops are recorded
        lazy_ops = ds_result._lazy_ops
        op_descriptions = [op.describe() for op in lazy_ops]

        # Should have WHERE, GroupBy, ORDER BY, LIMIT
        assert any('WHERE' in desc for desc in op_descriptions), f"Missing WHERE in lazy ops: {op_descriptions}"
        assert any('GroupBy' in desc for desc in op_descriptions), f"Missing GroupBy in lazy ops: {op_descriptions}"
        assert any('ORDER BY' in desc for desc in op_descriptions), f"Missing ORDER BY in lazy ops: {op_descriptions}"
        assert any('LIMIT' in desc for desc in op_descriptions), f"Missing LIMIT in lazy ops: {op_descriptions}"


class TestReadmeExampleEdgeCases:
    """Test edge cases for the README example."""

    def test_empty_result(self, employee_data):
        """Test when filter produces no results."""
        csv_path, raw_data = employee_data

        # DataStore - filter that matches nothing
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 100) & (ds_df['salary'] > 1000000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_result = ds_grouped.to_df()

        # Pandas
        pd_filtered = raw_data[(raw_data['age'] > 100) & (raw_data['salary'] > 1000000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])

        # Both should be empty
        assert len(ds_result) == len(pd_grouped) == 0

    def test_single_group(self, tmp_path):
        """Test when there's only one group."""
        # Create data with single city
        data = pd.DataFrame({'age': [30, 35, 40], 'salary': [60000, 75000, 85000], 'city': ['NYC', 'NYC', 'NYC']})
        csv_path = tmp_path / "single_city.csv"
        data.to_csv(csv_path, index=False)

        # DataStore
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 25) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_sorted = ds_grouped.sort_values('mean', ascending=False)
        ds_result = ds_sorted.head(10).to_df()

        # Pandas
        pd_filtered = data[(data['age'] > 25) & (data['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        pd_sorted = pd_grouped.sort_values('mean', ascending=False)
        pd_result = pd_sorted.head(10)

        # Check results
        assert len(ds_result) == len(pd_result) == 1
        assert list(ds_result.index) == list(pd_result.index)

        for col in pd_result.columns:
            np.testing.assert_array_almost_equal(ds_result[col].values, pd_result[col].values, decimal=5)

    def test_all_rows_filtered(self, tmp_path):
        """Test when all rows match the filter."""
        data = pd.DataFrame({'age': [30, 35, 40], 'salary': [60000, 75000, 85000], 'city': ['NYC', 'LA', 'NYC']})
        csv_path = tmp_path / "all_match.csv"
        data.to_csv(csv_path, index=False)

        # DataStore - filter that matches all
        ds_df = ds.read_csv(str(csv_path))
        ds_filtered = ds_df[(ds_df['age'] > 20) & (ds_df['salary'] > 50000)]
        ds_grouped = ds_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        ds_sorted = ds_grouped.sort_values('mean', ascending=False)
        ds_result = ds_sorted.head(10).to_df()

        # Pandas
        pd_filtered = data[(data['age'] > 20) & (data['salary'] > 50000)]
        pd_grouped = pd_filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        pd_sorted = pd_grouped.sort_values('mean', ascending=False)
        pd_result = pd_sorted.head(10)

        # Check results
        assert len(ds_result) == len(pd_result)

        # Sort by index for comparison
        ds_result = ds_result.sort_index()
        pd_result = pd_result.sort_index()

        for col in pd_result.columns:
            np.testing.assert_array_almost_equal(ds_result[col].values, pd_result[col].values, decimal=5)
