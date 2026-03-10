"""
Tests for pristine source metadata optimization.

Verifies that dtypes, columns, shape, size, empty, ndim do NOT trigger
full data loading when accessing pristine (unmodified) DataStore sources.

Covers three source types:
1. DataFrame source — reads metadata directly from the in-memory DataFrame
2. File source — uses DESCRIBE / COUNT(*) / LIMIT 0 via chDB SQL
3. Remote ClickHouse source — same SQL metadata path over the network

Uses the clickhouse_server fixture (auto-starts local ClickHouse) for
remote source tests.
"""

import os
import time
import tempfile

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dataset_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "dataset", name)


# ---------------------------------------------------------------------------
# 1. DataFrame source (pristine)
# ---------------------------------------------------------------------------

class TestPristineDataFrameSource:
    """Metadata access on DataStore(df) should NOT go through _execute()."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [90.5, 85.0, 92.3, 78.0, 95.5],
            "active": [True, False, True, True, False],
        })

    def test_columns_without_execution(self, df):
        ds = DataStore(df)
        assert list(ds.columns) == list(df.columns)

    def test_dtypes_without_execution(self, df):
        ds = DataStore(df)
        pd.testing.assert_series_equal(ds.dtypes, df.dtypes)

    def test_shape_without_execution(self, df):
        ds = DataStore(df)
        assert ds.shape == df.shape

    def test_size_without_execution(self, df):
        ds = DataStore(df)
        assert ds.size == df.size

    def test_empty_without_execution(self, df):
        ds = DataStore(df)
        assert ds.empty == df.empty

    def test_ndim_always_two(self, df):
        ds = DataStore(df)
        assert ds.ndim == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="float64")})
        ds = DataStore(df)
        assert ds.empty is True
        assert ds.shape == (0, 2)
        assert ds.size == 0
        assert list(ds.columns) == ["a", "b"]

    def test_pristine_reads_source_directly(self, df):
        """Verify the fast path: _get_source_df_if_pristine returns the source."""
        ds = DataStore(df)
        src = ds._get_source_df_if_pristine()
        assert src is not None
        assert src is df

    def test_cached_result_counts_as_pristine(self, df):
        """After execution + cache, metadata should still be fast."""
        ds = DataStore(df)
        _ = repr(ds)
        src = ds._get_source_df_if_pristine()
        assert src is not None
        assert list(ds.columns) == list(df.columns)
        assert ds.shape == df.shape

    def test_with_ops_not_pristine(self, df):
        """Once ops are added, _get_source_df_if_pristine should return None."""
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 80]
        assert ds2._get_source_df_if_pristine() is None


# ---------------------------------------------------------------------------
# 2. File source (pristine)
# ---------------------------------------------------------------------------

class TestPristineFileSource:
    """Metadata access on DataStore.from_file() should use SQL metadata queries."""

    @pytest.fixture
    def parquet_file(self, tmp_path):
        df = pd.DataFrame({
            "id": range(1000),
            "value": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        })
        path = str(tmp_path / "test_data.parquet")
        df.to_parquet(path, index=False)
        return path, df

    @pytest.fixture
    def csv_file(self, tmp_path):
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [90.5, 85.0, 92.3],
        })
        path = str(tmp_path / "test_data.csv")
        df.to_csv(path, index=False)
        return path, df

    def test_columns_from_parquet(self, parquet_file):
        path, df = parquet_file
        ds = DataStore.from_file(path)
        assert list(ds.columns) == list(df.columns)

    def test_shape_from_parquet(self, parquet_file):
        path, df = parquet_file
        ds = DataStore.from_file(path)
        assert ds.shape == df.shape

    def test_empty_from_parquet(self, parquet_file):
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        assert ds.empty is False

    def test_size_from_parquet(self, parquet_file):
        path, df = parquet_file
        ds = DataStore.from_file(path)
        assert ds.size == df.size

    def test_ndim_from_parquet(self, parquet_file):
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        assert ds.ndim == 2

    def test_dtypes_from_parquet(self, parquet_file):
        path, df = parquet_file
        ds = DataStore.from_file(path)
        assert list(ds.dtypes.index) == list(df.dtypes.index)
        assert len(ds.dtypes) == len(df.dtypes)

    def test_columns_from_csv(self, csv_file):
        path, df = csv_file
        ds = DataStore.from_file(path)
        assert list(ds.columns) == list(df.columns)

    def test_is_pristine_sql_source(self, parquet_file):
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        assert ds._is_pristine_sql_source() is True
        assert ds._get_source_df_if_pristine() is None

    def test_with_ops_not_pristine_sql(self, parquet_file):
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        ds2 = ds[ds["value"] > 0]
        assert ds2._is_pristine_sql_source() is False

    def test_metadata_does_not_load_all_data(self, parquet_file):
        """Metadata access should not populate _cached_result."""
        path, df = parquet_file
        ds = DataStore.from_file(path)
        assert ds._cached_result is None
        _ = ds.columns
        _ = ds.shape
        _ = ds.dtypes
        _ = ds.empty
        assert ds._cached_result is None


# ---------------------------------------------------------------------------
# 3. Remote ClickHouse source (pristine) — uses clickhouse_server fixture
# ---------------------------------------------------------------------------

class TestPristineRemoteClickHouseSource:
    """Metadata access on session.table() should use SQL metadata queries."""

    def test_columns_from_system_one(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        cols = ds.columns
        assert "dummy" in list(cols)

    def test_shape_from_system_one(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        rows, ncols = ds.shape
        assert rows == 1
        assert ncols >= 1

    def test_empty_from_system_one(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        assert ds.empty is False

    def test_ndim_from_remote(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        assert ds.ndim == 2

    def test_size_from_system_one(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        assert ds.size >= 1

    def test_dtypes_from_system_one(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        dtypes = ds.dtypes
        assert len(dtypes) >= 1
        assert "dummy" in dtypes.index

    def test_columns_from_system_tables(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "tables")
        cols = ds.columns
        assert "database" in list(cols)
        assert "name" in list(cols)
        assert "engine" in list(cols)

    def test_shape_from_system_tables(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "tables")
        rows, ncols = ds.shape
        assert rows > 0
        assert ncols > 5

    def test_is_pristine_sql_source_remote(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "one")
        assert ds._is_pristine_sql_source() is True

    def test_after_filter_not_pristine(self, clickhouse_connection):
        ds = clickhouse_connection.table("system", "tables")
        ds2 = ds[ds["database"] == "system"]
        assert ds2._is_pristine_sql_source() is False


class TestPristineRemoteTestDB:
    """Test with test_db tables for richer schema verification."""

    @pytest.fixture(autouse=True)
    def check_test_db(self, clickhouse_connection):
        databases = clickhouse_connection.databases()
        if "test_db" not in databases:
            pytest.skip("test_db not available")

    def test_columns_from_users_table(self, clickhouse_connection):
        ds = clickhouse_connection.table("test_db", "users")
        cols = list(ds.columns)
        assert "id" in cols
        assert "name" in cols
        assert "email" in cols
        assert "age" in cols

    def test_dtypes_from_users_table(self, clickhouse_connection):
        ds = clickhouse_connection.table("test_db", "users")
        dtypes = ds.dtypes
        assert len(dtypes) >= 4
        assert "id" in dtypes.index
        assert "name" in dtypes.index

    def test_shape_from_users_table(self, clickhouse_connection):
        ds = clickhouse_connection.table("test_db", "users")
        rows, ncols = ds.shape
        assert rows == 3
        assert ncols >= 4

    def test_size_from_users_table(self, clickhouse_connection):
        ds = clickhouse_connection.table("test_db", "users")
        assert ds.size == 3 * len(ds.columns)

    def test_empty_from_users_table(self, clickhouse_connection):
        ds = clickhouse_connection.table("test_db", "users")
        assert ds.empty is False

    def test_metadata_matches_sql_schema(self, clickhouse_connection):
        """Metadata from pristine path must match DESCRIBE schema."""
        ds = clickhouse_connection.table("test_db", "users")
        meta_cols = list(ds.columns)
        meta_dtypes = ds.dtypes

        schema_df = clickhouse_connection.describe("test_db", "users")
        schema_cols = schema_df["name"].tolist()

        assert meta_cols == schema_cols
        assert len(meta_dtypes) == len(schema_cols)
        assert list(meta_dtypes.index) == schema_cols

    def test_metadata_does_not_trigger_full_load(self, clickhouse_connection):
        """Accessing metadata should not populate _cached_result."""
        ds = clickhouse_connection.table("test_db", "users")
        assert ds._cached_result is None
        _ = ds.columns
        _ = ds.shape
        _ = ds.dtypes
        _ = ds.empty
        assert ds._cached_result is None
