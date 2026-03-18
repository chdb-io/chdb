"""
Tests for INSERT INTO remote ClickHouse via TABLE FUNCTION remote().

Covers:
- SQL generation for INSERT INTO TABLE FUNCTION remote(...)
- Type conversions: int, float, string, datetime, bool, NaN, None, NaT
- NULL handling
- Large batch writes
- Append vs overwrite
- insert_into().select_from() from one remote table to another
- Write-then-read-back verification

SQL generation tests run without a server.
Integration tests require a real ClickHouse server (via clickhouse_server fixture).
"""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.table_functions import RemoteTableFunction
from datastore.exceptions import QueryError


# ============================================================================
# SQL Generation Tests (no server needed)
# ============================================================================


class TestInsertRemoteSQLGeneration:
    """Test INSERT INTO TABLE FUNCTION remote(...) SQL generation."""

    def _make_remote_ds(self, table="users"):
        """Helper to create a DataStore with remote table function."""
        return DataStore(
            "clickhouse",
            host="localhost:9000",
            database="default",
            table=table,
            user="default",
            password="",
        )

    def test_basic_insert_values(self):
        """Basic INSERT INTO remote() with VALUES."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "name", "age").insert_values(
            (1, "Alice", 25)
        )
        sql = query.to_sql()

        assert "INSERT INTO TABLE FUNCTION remote(" in sql
        assert "'localhost:9000'" in sql
        assert "'default'" in sql
        assert "'users'" in sql
        assert '("id", "name", "age")' in sql
        assert "VALUES (1, 'Alice', 25)" in sql

    def test_insert_multiple_rows(self):
        """INSERT with multiple rows in VALUES."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "name").insert_values(
            (1, "Alice"), (2, "Bob"), (3, "Charlie")
        )
        sql = query.to_sql()

        assert "VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')" in sql

    def test_insert_chained_values(self):
        """INSERT with chained insert_values() calls."""
        ds = self._make_remote_ds()
        query = (
            ds.insert_into("id", "name")
            .insert_values(1, "Alice")
            .insert_values(2, "Bob")
        )
        sql = query.to_sql()

        assert "VALUES (1, 'Alice'), (2, 'Bob')" in sql

    def test_insert_null_values(self):
        """INSERT with None -> NULL."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "name", "email").insert_values(
            (1, "Alice", None)
        )
        sql = query.to_sql()

        assert "VALUES (1, 'Alice', NULL)" in sql

    def test_insert_nan_becomes_null(self):
        """INSERT with float('nan') -> NULL."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "val").insert_values((1, float("nan")))
        sql = query.to_sql()

        assert "VALUES (1, NULL)" in sql

    def test_insert_numpy_nan_becomes_null(self):
        """INSERT with np.nan -> NULL."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "val").insert_values((1, np.nan))
        sql = query.to_sql()

        assert "VALUES (1, NULL)" in sql

    def test_insert_nat_becomes_null(self):
        """INSERT with pd.NaT -> NULL."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "ts").insert_values((1, pd.NaT))
        sql = query.to_sql()

        assert "VALUES (1, NULL)" in sql

    def test_insert_boolean_values(self):
        """INSERT with bool -> 1/0."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "active").insert_values(
            (1, True), (2, False)
        )
        sql = query.to_sql()

        assert "VALUES (1, 1), (2, 0)" in sql

    def test_insert_datetime_values(self):
        """INSERT with datetime -> quoted string."""
        ds = self._make_remote_ds()
        ts = datetime(2024, 1, 15, 10, 30, 0)
        query = ds.insert_into("id", "ts").insert_values((1, ts))
        sql = query.to_sql()

        assert "VALUES (1, '2024-01-15 10:30:00')" in sql

    def test_insert_date_values(self):
        """INSERT with date -> quoted string."""
        ds = self._make_remote_ds()
        d = date(2024, 1, 15)
        query = ds.insert_into("id", "d").insert_values((1, d))
        sql = query.to_sql()

        assert "VALUES (1, '2024-01-15')" in sql

    def test_insert_float_values(self):
        """INSERT with float values."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "price").insert_values((1, 99.99))
        sql = query.to_sql()

        assert "VALUES (1, 99.99)" in sql

    def test_insert_string_escaping(self):
        """INSERT with string that needs escaping."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "name").insert_values((1, "O'Brien"))
        sql = query.to_sql()

        assert "VALUES (1, 'O''Brien')" in sql

    def test_insert_select_from_remote(self):
        """INSERT INTO remote() SELECT FROM another remote table."""
        ds_target = self._make_remote_ds("users_backup")
        ds_source = self._make_remote_ds("users")

        query = ds_target.insert_into("id", "name", "age").select_from(
            ds_source.select("id", "name", "age").filter(ds_source.age > 18)
        )
        sql = query.to_sql()

        assert "INSERT INTO TABLE FUNCTION remote(" in sql
        assert "'users_backup'" in sql
        assert "SELECT" in sql
        assert '"age" > 18' in sql

    def test_insert_select_from_local_table(self):
        """INSERT INTO remote() SELECT FROM local table."""
        ds_target = self._make_remote_ds("users_backup")
        ds_source = DataStore(table="local_users")

        query = ds_target.insert_into("id", "name").select_from(
            ds_source.select("id", "name")
        )
        sql = query.to_sql()

        assert "INSERT INTO TABLE FUNCTION remote(" in sql
        assert 'SELECT "id", "name" FROM "local_users"' in sql

    def test_insert_with_secure_remote(self):
        """INSERT INTO remoteSecure() for ClickHouse Cloud."""
        ds = DataStore(
            "clickhouse",
            host="abc123.clickhouse.cloud",
            database="default",
            table="users",
            user="default",
            password="secret",
        )
        query = ds.insert_into("id", "name").insert_values((1, "Alice"))
        sql = query.to_sql()

        assert "INSERT INTO TABLE FUNCTION remoteSecure(" in sql
        assert "'abc123.clickhouse.cloud:9440'" in sql

    def test_insert_mixed_types_in_row(self):
        """INSERT with mixed types in a single row."""
        ds = self._make_remote_ds()
        ts = datetime(2024, 6, 15, 12, 0, 0)
        query = ds.insert_into(
            "id", "name", "score", "active", "created_at", "email"
        ).insert_values((1, "Alice", 95.5, True, ts, None))
        sql = query.to_sql()

        assert "VALUES (1, 'Alice', 95.5, 1, '2024-06-15 12:00:00', NULL)" in sql

    def test_insert_numpy_int64(self):
        """INSERT with numpy int64."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "val").insert_values(
            (np.int64(1), np.int64(42))
        )
        sql = query.to_sql()

        assert "VALUES (1, 42)" in sql

    def test_insert_numpy_float64(self):
        """INSERT with numpy float64."""
        ds = self._make_remote_ds()
        query = ds.insert_into("id", "val").insert_values(
            (1, np.float64(3.14))
        )
        sql = query.to_sql()

        assert "VALUES (1, 3.14)" in sql

    def test_insert_requires_insert_into_first(self):
        """insert_values() without insert_into() raises error."""
        ds = self._make_remote_ds()
        with pytest.raises(QueryError, match="Must call insert_into"):
            ds.insert_values(1, "Alice")

    def test_select_from_requires_insert_into_first(self):
        """select_from() without insert_into() raises error."""
        ds = self._make_remote_ds()
        source = DataStore(table="users")
        with pytest.raises(QueryError, match="Must call insert_into"):
            ds.select_from(source.select("*"))

    def test_insert_immutability(self):
        """insert_into/insert_values return new DataStore, not mutate."""
        ds = self._make_remote_ds()
        ds2 = ds.insert_into("id", "name")
        ds3 = ds2.insert_values(1, "Alice")

        # Original ds should NOT have insert state
        assert not ds._insert_columns
        assert not ds._insert_values

        # ds2 should have columns but no values
        assert ds2._insert_columns == ["id", "name"]
        assert not ds2._insert_values

        # ds3 should have both
        assert ds3._insert_columns == ["id", "name"]
        assert len(ds3._insert_values) == 1

    def test_large_batch_sql_generation(self):
        """SQL generation for large batch (1000 rows)."""
        ds = self._make_remote_ds()
        rows = [(i, f"user_{i}", i * 1.5) for i in range(1000)]
        query = ds.insert_into("id", "name", "score").insert_values(*rows)
        sql = query.to_sql()

        assert "INSERT INTO TABLE FUNCTION remote(" in sql
        assert sql.count("(") >= 1001  # 1000 value groups + columns group
        assert "user_0" in sql
        assert "user_999" in sql


# ============================================================================
# Integration Tests (require real ClickHouse server)
# ============================================================================

# Path to ClickHouse client binary used by the test infrastructure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CH_BINARY = os.path.join(SCRIPT_DIR, ".clickhouse", "clickhouse")


class TestInsertRemoteIntegration:
    """Integration tests: INSERT into real ClickHouse via remote() table function."""

    @pytest.fixture(autouse=True)
    def setup_connection(self, clickhouse_server):
        """Set up connection params from server fixture."""
        self.host, self.port = clickhouse_server
        self.conn_str = f"{self.host}:{self.port}"

    def _make_remote_ds(self, table, database="default"):
        """Create DataStore pointing to remote table."""
        return DataStore(
            "clickhouse",
            host=self.conn_str,
            database=database,
            table=table,
            user="default",
            password="",
        )

    def _run_ddl(self, sql):
        """Run DDL on the remote ClickHouse server via clickhouse client."""
        result = subprocess.run(
            [CH_BINARY, "client", f"--port={self.port}", f"--query={sql}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"DDL failed: {result.stderr}\nSQL: {sql}"
            )

    def _drop_table(self, table_name, database="default"):
        """Drop a table on the remote server."""
        self._run_ddl(f"DROP TABLE IF EXISTS {database}.{table_name}")

    def test_basic_insert_and_readback(self, clickhouse_server):
        """INSERT VALUES into remote table, then read back and verify."""
        table = "test_insert_basic"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, name String, age UInt8) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            query = ds.insert_into("id", "name", "age").insert_values(
                (1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)
            )
            query.execute()

            # Read back
            result = ds.select("*")._execute()
            assert len(result) == 3

            rows = result.sort_values("id").to_dict(orient="records")
            assert rows[0]["name"] == "Alice"
            assert rows[0]["age"] == 25
            assert rows[1]["name"] == "Bob"
            assert rows[2]["name"] == "Charlie"
        finally:
            self._drop_table(table)

    def test_type_conversions(self, clickhouse_server):
        """Test pandas types -> ClickHouse types through INSERT."""
        table = "test_insert_types"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, int_val Int64, float_val Float64, "
                f"str_val String, dt_val DateTime) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            ts = datetime(2024, 6, 15, 12, 30, 0)
            query = ds.insert_into(
                "id", "int_val", "float_val", "str_val", "dt_val"
            ).insert_values(
                (1, 42, 3.14, "hello", ts),
                (2, -100, 0.001, "world", datetime(2025, 1, 1, 0, 0, 0)),
            )
            query.execute()

            result = ds.select("*")._execute().sort_values("id")
            assert len(result) == 2

            row0 = result.iloc[0]
            assert row0["int_val"] == 42
            assert abs(row0["float_val"] - 3.14) < 0.001
            assert row0["str_val"] == "hello"

            row1 = result.iloc[1]
            assert row1["int_val"] == -100
            assert row1["str_val"] == "world"
        finally:
            self._drop_table(table)

    def test_null_handling(self, clickhouse_server):
        """Test None/NaN/NaT -> ClickHouse NULL."""
        table = "test_insert_nulls"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, name Nullable(String), "
                f"score Nullable(Float64), ts Nullable(DateTime)) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            query = ds.insert_into("id", "name", "score", "ts").insert_values(
                (1, "Alice", 95.5, datetime(2024, 1, 1, 0, 0, 0)),
                (2, None, float("nan"), pd.NaT),
                (3, "Charlie", np.nan, None),
            )
            query.execute()

            result = ds.select("*")._execute().sort_values("id")
            assert len(result) == 3

            # Row 1: all values present
            assert result.iloc[0]["name"] == "Alice"
            assert result.iloc[0]["score"] == 95.5

            # Row 2: all nullable columns are NULL
            assert pd.isna(result.iloc[1]["name"])
            assert pd.isna(result.iloc[1]["score"])
            assert pd.isna(result.iloc[1]["ts"])

            # Row 3: mixed NULL
            assert result.iloc[2]["name"] == "Charlie"
            assert pd.isna(result.iloc[2]["score"])
            assert pd.isna(result.iloc[2]["ts"])
        finally:
            self._drop_table(table)

    def test_large_batch_insert(self, clickhouse_server):
        """INSERT 10k+ rows and verify correctness."""
        table = "test_insert_large"
        n_rows = 10000
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, name String, value Float64) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            rows = [(i, f"user_{i}", float(i) * 0.1) for i in range(n_rows)]
            query = ds.insert_into("id", "name", "value").insert_values(*rows)
            query.execute()

            # Verify count
            result = ds.select("*")._execute()
            assert len(result) == n_rows

            # Verify boundary values
            sorted_result = result.sort_values("id")
            assert sorted_result.iloc[0]["name"] == "user_0"
            assert sorted_result.iloc[-1]["name"] == f"user_{n_rows - 1}"
            assert abs(sorted_result.iloc[100]["value"] - 10.0) < 0.01
        finally:
            self._drop_table(table)

    def test_append_write(self, clickhouse_server):
        """INSERT twice into same table -> data appended."""
        table = "test_insert_append"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, name String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)

            # First insert
            ds.insert_into("id", "name").insert_values(
                (1, "Alice"), (2, "Bob")
            ).execute()

            # Second insert (append)
            ds.insert_into("id", "name").insert_values(
                (3, "Charlie"), (4, "Diana")
            ).execute()

            # Verify all 4 rows
            result = ds.select("*")._execute()
            assert len(result) == 4

            names = sorted(result["name"].tolist())
            assert names == ["Alice", "Bob", "Charlie", "Diana"]
        finally:
            self._drop_table(table)

    def test_insert_select_from_remote_to_remote(self, clickhouse_server):
        """INSERT INTO remote_table SELECT FROM another remote_table."""
        src_table = "test_insert_select_src"
        dst_table = "test_insert_select_dst"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{src_table} "
                f"(id UInt64, name String, age UInt8) "
                f"ENGINE = MergeTree() ORDER BY id"
            )
            self._run_ddl(
                f"CREATE TABLE default.{dst_table} "
                f"(id UInt64, name String, age UInt8) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            # Insert source data
            ds_src = self._make_remote_ds(src_table)
            ds_src.insert_into("id", "name", "age").insert_values(
                (1, "Alice", 25),
                (2, "Bob", 17),
                (3, "Charlie", 30),
                (4, "Diana", 15),
            ).execute()

            # INSERT INTO dst SELECT FROM src WHERE age >= 18
            ds_dst = self._make_remote_ds(dst_table)
            ds_dst.insert_into("id", "name", "age").select_from(
                ds_src.select("id", "name", "age").filter(ds_src.age >= 18)
            ).execute()

            # Verify only adults were copied
            result = ds_dst.select("*")._execute()
            assert len(result) == 2

            names = sorted(result["name"].tolist())
            assert names == ["Alice", "Charlie"]
        finally:
            self._drop_table(src_table)
            self._drop_table(dst_table)

    def test_write_then_read_consistency(self, clickhouse_server):
        """Write data, read back, verify exact match with original."""
        table = "test_insert_consistency"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, name String, score Float64, active UInt8) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            original_data = [
                (1, "Alice", 95.5, True),
                (2, "Bob", 87.3, False),
                (3, "Charlie", 72.1, True),
                (4, "Diana", 99.9, True),
                (5, "Eve", 60.0, False),
            ]

            ds = self._make_remote_ds(table)
            ds.insert_into("id", "name", "score", "active").insert_values(
                *original_data
            ).execute()

            result = ds.select("*")._execute().sort_values("id")

            for i, (oid, oname, oscore, oactive) in enumerate(original_data):
                row = result.iloc[i]
                assert row["id"] == oid
                assert row["name"] == oname
                assert abs(row["score"] - oscore) < 0.01
                assert row["active"] == (1 if oactive else 0)
        finally:
            self._drop_table(table)

    def test_string_with_special_characters(self, clickhouse_server):
        """INSERT strings with quotes and whitespace characters."""
        table = "test_insert_special_str"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, text String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            ds.insert_into("id", "text").insert_values(
                (1, "O'Brien"),
                (2, "hello world"),
                (3, "tabs here"),
            ).execute()

            result = ds.select("*")._execute().sort_values("id")
            assert len(result) == 3
            assert result.iloc[0]["text"] == "O'Brien"
            assert result.iloc[1]["text"] == "hello world"
            assert result.iloc[2]["text"] == "tabs here"
        finally:
            self._drop_table(table)

    def test_insert_empty_string(self, clickhouse_server):
        """INSERT with empty string values."""
        table = "test_insert_empty_str"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, name String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            ds.insert_into("id", "name").insert_values(
                (1, ""), (2, "non-empty")
            ).execute()

            result = ds.select("*")._execute().sort_values("id")
            assert result.iloc[0]["name"] == ""
            assert result.iloc[1]["name"] == "non-empty"
        finally:
            self._drop_table(table)

    def test_insert_boolean_roundtrip(self, clickhouse_server):
        """INSERT booleans into UInt8, read back as integers."""
        table = "test_insert_bool"
        try:
            self._run_ddl(
                f"CREATE TABLE default.{table} "
                f"(id UInt64, flag UInt8) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = self._make_remote_ds(table)
            ds.insert_into("id", "flag").insert_values(
                (1, True), (2, False), (3, True)
            ).execute()

            result = ds.select("*")._execute().sort_values("id")
            assert result.iloc[0]["flag"] == 1
            assert result.iloc[1]["flag"] == 0
            assert result.iloc[2]["flag"] == 1
        finally:
            self._drop_table(table)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
