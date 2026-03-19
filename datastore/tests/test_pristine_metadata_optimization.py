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

Scenario 5 additions: mock-based verification that _execute() is never
called for pristine metadata access, LIMIT 0 query verification for
_probe_dtypes_from_sql_source(), and post-filter metadata correctness.
"""

import os
import time
import tempfile
from unittest.mock import patch, MagicMock, call

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


# ---------------------------------------------------------------------------
# 4. Zero-overhead mock verification (Scenario 5)
# ---------------------------------------------------------------------------

class TestZeroOverheadDataFrameSource:
    """Verify _execute() is never called for pristine DataFrame metadata."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [90.5, 85.0, 92.3, 78.0, 95.5],
            "active": [True, False, True, True, False],
        })

    def test_columns_no_execute(self, df):
        """Access .columns on DataStore(df) must not call _execute."""
        ds = DataStore(df)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            cols = ds.columns
            mock_exec.assert_not_called()
        assert list(cols) == list(df.columns)

    def test_dtypes_no_execute(self, df):
        """Access .dtypes on DataStore(df) must not call _execute."""
        ds = DataStore(df)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            dtypes = ds.dtypes
            mock_exec.assert_not_called()
        pd.testing.assert_series_equal(dtypes, df.dtypes)

    def test_shape_no_execute(self, df):
        """Access .shape on DataStore(df) must not call _execute."""
        ds = DataStore(df)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            shape = ds.shape
            mock_exec.assert_not_called()
        assert shape == df.shape

    def test_size_no_execute(self, df):
        """Access .size on DataStore(df) must not call _execute."""
        ds = DataStore(df)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            size = ds.size
            mock_exec.assert_not_called()
        assert size == df.size

    def test_empty_no_execute(self, df):
        """Access .empty on DataStore(df) must not call _execute."""
        ds = DataStore(df)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            empty = ds.empty
            mock_exec.assert_not_called()
        assert empty == df.empty

    def test_all_metadata_no_execute_combined(self, df):
        """All metadata properties together must not trigger _execute."""
        ds = DataStore(df)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            _ = ds.columns
            _ = ds.dtypes
            _ = ds.shape
            _ = ds.size
            _ = ds.empty
            _ = ds.ndim
            mock_exec.assert_not_called()


class TestZeroOverheadFileSource:
    """Verify _execute() is never called for pristine file source metadata."""

    @pytest.fixture
    def parquet_file(self, tmp_path):
        df = pd.DataFrame({
            "id": range(100),
            "value": np.random.randn(100),
            "category": np.random.choice(["A", "B", "C"], 100),
        })
        path = str(tmp_path / "test_data.parquet")
        df.to_parquet(path, index=False)
        return path, df

    @pytest.fixture
    def csv_file(self, tmp_path):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4.0, 5.0, 6.0],
        })
        path = str(tmp_path / "test_data.csv")
        df.to_csv(path, index=False)
        return path, df

    def test_columns_no_execute_parquet(self, parquet_file):
        """Access .columns on from_file() parquet must not call _execute."""
        path, df = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            cols = ds.columns
            mock_exec.assert_not_called()
        assert list(cols) == list(df.columns)

    def test_dtypes_no_execute_parquet(self, parquet_file):
        """Access .dtypes on from_file() parquet must not call _execute."""
        path, df = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            dtypes = ds.dtypes
            mock_exec.assert_not_called()
        assert list(dtypes.index) == list(df.dtypes.index)

    def test_shape_no_execute_parquet(self, parquet_file):
        """Access .shape on from_file() parquet must not call _execute."""
        path, df = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            shape = ds.shape
            mock_exec.assert_not_called()
        assert shape == df.shape

    def test_size_no_execute_parquet(self, parquet_file):
        """Access .size on from_file() parquet must not call _execute."""
        path, df = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            size = ds.size
            mock_exec.assert_not_called()
        assert size == df.size

    def test_empty_no_execute_parquet(self, parquet_file):
        """Access .empty on from_file() parquet must not call _execute."""
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            empty = ds.empty
            mock_exec.assert_not_called()
        assert empty is False

    def test_all_metadata_no_execute_file(self, parquet_file):
        """All metadata properties together must not trigger _execute on file source."""
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            _ = ds.columns
            _ = ds.dtypes
            _ = ds.shape
            _ = ds.size
            _ = ds.empty
            _ = ds.ndim
            mock_exec.assert_not_called()
        assert ds._cached_result is None

    def test_columns_no_execute_csv(self, csv_file):
        """Access .columns on from_file() CSV must not call _execute."""
        path, df = csv_file
        ds = DataStore.from_file(path)
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            cols = ds.columns
            mock_exec.assert_not_called()
        assert list(cols) == list(df.columns)


# ---------------------------------------------------------------------------
# 5. _probe_dtypes_from_sql_source() LIMIT 0 verification (Scenario 5)
# ---------------------------------------------------------------------------

class TestProbeDtypesLimit0:
    """Verify _probe_dtypes_from_sql_source uses LIMIT 0 query."""

    @pytest.fixture
    def parquet_file(self, tmp_path):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "z"],
        })
        path = str(tmp_path / "probe_test.parquet")
        df.to_parquet(path, index=False)
        return path, df

    def test_probe_dtypes_uses_limit_0(self, parquet_file):
        """_probe_dtypes_from_sql_source must execute a LIMIT 0 query."""
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        ds.connect()

        original_execute = ds._executor.execute
        executed_sqls = []

        def capture_sql(sql, *args, **kwargs):
            executed_sqls.append(sql)
            return original_execute(sql, *args, **kwargs)

        with patch.object(ds._executor, "execute", side_effect=capture_sql):
            dtypes = ds._probe_dtypes_from_sql_source()

        limit_0_sqls = [s for s in executed_sqls if "LIMIT 0" in s]
        assert len(limit_0_sqls) > 0, (
            f"Expected LIMIT 0 query, got: {executed_sqls}"
        )
        # The query should be SELECT * FROM ... LIMIT 0
        assert "SELECT * FROM" in limit_0_sqls[0]
        assert isinstance(dtypes, pd.Series)
        assert len(dtypes) == 3

    def test_probe_dtypes_returns_correct_columns(self, parquet_file):
        """_probe_dtypes_from_sql_source must return dtypes for all columns."""
        path, df = parquet_file
        ds = DataStore.from_file(path)
        dtypes = ds._probe_dtypes_from_sql_source()
        assert list(dtypes.index) == list(df.columns)

    def test_dtypes_property_calls_probe_for_file_source(self, parquet_file):
        """Accessing .dtypes on file source must route to _probe_dtypes_from_sql_source."""
        path, _ = parquet_file
        ds = DataStore.from_file(path)
        with patch.object(
            ds, "_probe_dtypes_from_sql_source",
            wraps=ds._probe_dtypes_from_sql_source,
        ) as mock_probe:
            _ = ds.dtypes
            mock_probe.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Post-filter metadata correctness (Scenario 5)
# ---------------------------------------------------------------------------

class TestPostFilterMetadata:
    """After filter, metadata must NOT use pristine path but still be correct."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [90.5, 85.0, 92.3, 78.0, 95.5],
        })

    def test_filter_then_columns(self, df):
        """After filter, .columns should return correct columns via execution."""
        pd_result = df[df["score"] > 80]
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 80]
        assert ds2._get_source_df_if_pristine() is None
        assert list(ds2.columns) == list(pd_result.columns)

    def test_filter_then_shape(self, df):
        """After filter, .shape should return correct shape via execution."""
        pd_result = df[df["score"] > 80]
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 80]
        assert ds2.shape == pd_result.shape

    def test_filter_then_dtypes(self, df):
        """After filter, .dtypes should still match pandas."""
        pd_result = df[df["score"] > 80]
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 80]
        pd.testing.assert_series_equal(ds2.dtypes, pd_result.dtypes)

    def test_filter_then_size(self, df):
        """After filter, .size should match pandas."""
        pd_result = df[df["score"] > 80]
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 80]
        assert ds2.size == pd_result.size

    def test_filter_then_empty(self, df):
        """After filter, .empty should match pandas."""
        pd_result = df[df["score"] > 80]
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 80]
        assert ds2.empty == pd_result.empty

    def test_filter_empty_result(self, df):
        """Filter that excludes all rows: metadata should reflect empty result."""
        pd_result = df[df["score"] > 999]
        ds = DataStore(df)
        ds2 = ds[ds["score"] > 999]
        assert ds2.empty == pd_result.empty
        assert ds2.shape == pd_result.shape
        assert ds2.size == pd_result.size

    def test_filter_on_file_source_not_pristine(self, tmp_path):
        """After filter on file source, pristine path must not be used."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
        path = str(tmp_path / "filter_test.parquet")
        df.to_parquet(path, index=False)

        pd_result = df[df["x"] > 2]
        ds = DataStore.from_file(path)
        ds2 = ds[ds["x"] > 2]
        assert ds2._is_pristine_sql_source() is False
        assert ds2._get_source_df_if_pristine() is None
        assert list(ds2.columns) == list(pd_result.columns)
        assert ds2.shape == pd_result.shape


# ---------------------------------------------------------------------------
# 7. Remote source DESCRIBE verification (Scenario 5)
# ---------------------------------------------------------------------------

class TestRemoteDescribeNotSelectStar:
    """Pristine remote .columns must use DESCRIBE/schema, not SELECT *."""

    def test_columns_uses_schema_path(self, clickhouse_connection):
        """Columns access on pristine remote source must route through schema()."""
        ds = clickhouse_connection.table("system", "one")
        with patch.object(ds, "schema", wraps=ds.schema) as mock_schema:
            with patch.object(
                ds, "_execute", wraps=ds._execute
            ) as mock_exec:
                cols = ds.columns
                # schema() should be called (DESCRIBE path)
                mock_schema.assert_called()
                # _execute() should NOT be called
                mock_exec.assert_not_called()
        assert "dummy" in list(cols)

    def test_dtypes_uses_probe_path(self, clickhouse_connection):
        """Dtypes access on pristine remote source must use _probe_dtypes_from_sql_source."""
        ds = clickhouse_connection.table("system", "one")
        with patch.object(
            ds, "_probe_dtypes_from_sql_source",
            wraps=ds._probe_dtypes_from_sql_source,
        ) as mock_probe:
            with patch.object(
                ds, "_execute", wraps=ds._execute
            ) as mock_exec:
                dtypes = ds.dtypes
                mock_probe.assert_called_once()
                mock_exec.assert_not_called()
        assert "dummy" in dtypes.index

    def test_shape_uses_count_rows_not_execute(self, clickhouse_connection):
        """Shape access on pristine remote source must use count_rows, not _execute."""
        ds = clickhouse_connection.table("system", "one")
        with patch.object(ds, "count_rows", wraps=ds.count_rows) as mock_count:
            with patch.object(
                ds, "_execute", wraps=ds._execute
            ) as mock_exec:
                shape = ds.shape
                mock_count.assert_called()
                mock_exec.assert_not_called()
        assert shape == (1, 1)

    def test_empty_uses_count_rows_not_execute(self, clickhouse_connection):
        """Empty check on pristine remote source must use count_rows, not _execute."""
        ds = clickhouse_connection.table("system", "one")
        with patch.object(ds, "count_rows", wraps=ds.count_rows) as mock_count:
            with patch.object(
                ds, "_execute", wraps=ds._execute
            ) as mock_exec:
                empty = ds.empty
                mock_count.assert_called()
                mock_exec.assert_not_called()
        assert empty is False

    def test_no_execute_for_all_metadata_remote(self, clickhouse_connection):
        """All metadata properties on pristine remote source must skip _execute."""
        ds = clickhouse_connection.table("system", "one")
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            _ = ds.columns
            _ = ds.dtypes
            _ = ds.shape
            _ = ds.size
            _ = ds.empty
            _ = ds.ndim
            mock_exec.assert_not_called()
        assert ds._cached_result is None

    def test_after_filter_remote_uses_execute(self, clickhouse_connection):
        """After filter on remote source, metadata MUST go through execution."""
        ds = clickhouse_connection.table("system", "tables")
        ds2 = ds[ds["database"] == "system"]
        assert ds2._is_pristine_sql_source() is False
        # Accessing shape on filtered remote source should work correctly
        rows, ncols = ds2.shape
        assert rows > 0
        assert ncols > 5
