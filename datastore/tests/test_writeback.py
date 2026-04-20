"""
Tests for writeback APIs: to_clickhouse, create_view, create_materialized_view, save.

All tests run against a remote ClickHouse server. Connection is configured
via environment variables:
  CHDB_TEST_HOST     — host:port  (e.g. "myserver:9000")
  CHDB_TEST_USER     — ClickHouse user
  CHDB_TEST_PASSWORD — password
  CHDB_TEST_DATABASE — target database

Tests are skipped automatically when the env vars are missing or the
server is unreachable.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import chdb
from datastore import DataStore
from datastore.expressions import col
from datastore.exceptions import QueryError, ExecutionError

HOST = os.environ.get("CHDB_TEST_HOST", "")
USER = os.environ.get("CHDB_TEST_USER", "")
PASSWORD = os.environ.get("CHDB_TEST_PASSWORD", "")
DATABASE = os.environ.get("CHDB_TEST_DATABASE", "")


def _remote_func(db, table):
    return f"remote('{HOST}', '{db}', '{table}', '{USER}', '{PASSWORD}')"


def _query_remote(sql):
    """Execute a SQL query via chdb using remote() for the test server."""
    return chdb.query(sql, output_format="DataFrame")


def _ddl(sql):
    """Execute DDL on the remote server via remote(query=...)."""
    escaped = sql.replace("\\", "\\\\").replace("'", "\\'")
    remote_sql = (
        f"SELECT * FROM remote('{HOST}', "
        f"query = '{escaped}', '{USER}', '{PASSWORD}')"
    )
    chdb.query(remote_sql, "TabSeparated")


def _drop(table, db=DATABASE):
    _ddl(f'DROP TABLE IF EXISTS "{db}"."{table}"')


def _count(table, db=DATABASE):
    result = _query_remote(
        f"SELECT count() AS cnt FROM {_remote_func(db, table)}"
    )
    return int(result.iloc[0, 0])


def _select_all(table, db=DATABASE, order_by=None):
    sql = f"SELECT * FROM {_remote_func(db, table)}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    return _query_remote(sql)


def _make_ds(table=None, db=DATABASE):
    """Create a DataStore pointing to the remote test server."""
    return DataStore.from_clickhouse(
        host=HOST, database=db, table=table, user=USER, password=PASSWORD
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _check_server():
    """Skip all tests if env vars are missing or the remote server is unreachable."""
    if not all([HOST, USER, DATABASE]):
        pytest.skip(
            "Remote ClickHouse env vars not set "
            "(CHDB_TEST_HOST, CHDB_TEST_USER, CHDB_TEST_DATABASE)"
        )
    try:
        _query_remote(
            f"SELECT 1 FROM remote('{HOST}', 'system', 'one', '{USER}', '{PASSWORD}')"
        )
    except Exception as e:
        pytest.skip(f"Remote ClickHouse server not available: {e}")


# ============================================================================
# Setup: create a source table with sample data
# ============================================================================


@pytest.fixture()
def source_table():
    """Create a source table with sample data and clean up after test."""
    table = "wb_source"
    _drop(table)
    _ddl(
        f'CREATE TABLE "{DATABASE}"."{table}" '
        f"(city String, amount Float64, status String) "
        f"ENGINE = MergeTree() ORDER BY city"
    )
    _ddl(
        f'INSERT INTO "{DATABASE}"."{table}" VALUES '
        f"('Beijing', 100.0, 'completed'), "
        f"('Beijing', 200.0, 'completed'), "
        f"('Beijing', 50.0, 'pending'), "
        f"('Shanghai', 300.0, 'completed'), "
        f"('Shanghai', 150.0, 'pending'), "
        f"('Guangzhou', 400.0, 'completed'), "
        f"('Guangzhou', 100.0, 'completed')"
    )
    yield table
    _drop(table)


# ============================================================================
# to_clickhouse tests
# ============================================================================


class TestToClickHouse:
    """Test to_clickhouse() method."""

    def test_basic_write_new_table(self, source_table):
        target = "wb_target_basic"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}", engine="MergeTree()", order_by="city")

            assert _count(target) == 7
            result = _select_all(target, order_by="city, amount")
            pd_expected = _select_all(source_table, order_by="city, amount")
            pd.testing.assert_frame_equal(
                result.reset_index(drop=True), pd_expected.reset_index(drop=True)
            )
        finally:
            _drop(target)

    def test_if_exists_fail(self, source_table):
        target = "wb_target_fail"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}", engine="MergeTree()", order_by="city")

            with pytest.raises(QueryError, match="already exists"):
                ds.to_clickhouse(f"{DATABASE}.{target}")
        finally:
            _drop(target)

    def test_if_exists_replace(self, source_table):
        target = "wb_target_replace"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}", engine="MergeTree()", order_by="city")
            assert _count(target) == 7

            ds_filtered = ds.filter(ds["status"] == "completed")
            ds_filtered.to_clickhouse(
                f"{DATABASE}.{target}", if_exists="replace",
                engine="MergeTree()", order_by="city",
            )
            assert _count(target) == 5

            result = _select_all(target, order_by="city, amount")
            assert all(result["status"] == "completed")
        finally:
            _drop(target)

    def test_if_exists_append(self, source_table):
        target = "wb_target_append"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}", engine="MergeTree()", order_by="city")
            assert _count(target) == 7

            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists="append")
            assert _count(target) == 14
        finally:
            _drop(target)

    def test_write_with_filter_and_sort(self, source_table):
        target = "wb_target_filter_sort"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds_processed = (
                ds.filter(ds["status"] == "completed")
                .sort("amount", ascending=False)
            )
            ds_processed.to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city"
            )

            result = _select_all(target, order_by="amount DESC")
            assert len(result) == 5
            assert result.iloc[0]["amount"] == 400.0
        finally:
            _drop(target)

    def test_write_with_groupby_agg(self, source_table):
        target = "wb_target_groupby"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            summary = (
                ds.filter(ds["status"] == "completed")
                .groupby("city")
                .agg(total=col("amount").sum(), cnt=col("amount").count())
            )
            summary.to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city"
            )

            result = _select_all(target, order_by="city")
            assert len(result) == 3

            bj = result[result["city"] == "Beijing"]
            assert float(bj["total"].iloc[0]) == 300.0
            assert int(bj["cnt"].iloc[0]) == 2

            gz = result[result["city"] == "Guangzhou"]
            assert float(gz["total"].iloc[0]) == 500.0
        finally:
            _drop(target)

    def test_write_back_to_source(self, source_table):
        """Write back to source table with replace (atomic)."""
        ds = _make_ds(source_table)
        ds_filtered = ds.filter(ds["status"] == "completed")
        ds_filtered.to_clickhouse(
            f"{DATABASE}.{source_table}",
            if_exists="replace", engine="MergeTree()", order_by="city"
        )

        assert _count(source_table) == 5
        result = _select_all(source_table)
        assert all(result["status"] == "completed")

    def test_order_by_list(self, source_table):
        target = "wb_target_order_list"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by=["city", "amount"],
            )
            assert _count(target) == 7
        finally:
            _drop(target)


# ============================================================================
# create_view tests
# ============================================================================


class TestCreateView:
    """Test create_view() method."""

    def test_basic_view(self, source_table):
        view_name = "wb_view_basic"
        _drop(view_name)
        try:
            ds = _make_ds(source_table)
            ds_filtered = ds.filter(ds["status"] == "completed")
            ds_filtered.create_view(f"{DATABASE}.{view_name}")

            result = _select_all(view_name, order_by="city, amount")
            assert len(result) == 5
            assert all(result["status"] == "completed")
        finally:
            _drop(view_name)

    def test_view_with_groupby(self, source_table):
        view_name = "wb_view_groupby"
        _drop(view_name)
        try:
            ds = _make_ds(source_table)
            summary = (
                ds.filter(ds["status"] == "completed")
                .groupby("city")
                .agg(total=col("amount").sum())
            )
            summary.create_view(f"{DATABASE}.{view_name}")

            result = _select_all(view_name, order_by="city")
            assert len(result) == 3

            bj = result[result["city"] == "Beijing"]
            assert float(bj["total"].iloc[0]) == 300.0
        finally:
            _drop(view_name)

    def test_view_replace(self, source_table):
        view_name = "wb_view_replace"
        _drop(view_name)
        try:
            ds = _make_ds(source_table)

            ds.filter(ds["status"] == "completed").create_view(
                f"{DATABASE}.{view_name}"
            )
            result1 = _select_all(view_name)
            assert len(result1) == 5

            ds.filter(ds["status"] == "pending").create_view(
                f"{DATABASE}.{view_name}", replace=True
            )
            result2 = _select_all(view_name)
            assert len(result2) == 2
            assert all(result2["status"] == "pending")
        finally:
            _drop(view_name)

    def test_view_reflects_source_changes(self, source_table):
        """Views should reflect changes in the source table."""
        view_name = "wb_view_live"
        _drop(view_name)
        try:
            ds = _make_ds(source_table)
            ds.create_view(f"{DATABASE}.{view_name}")

            count_before = len(_select_all(view_name))

            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Shenzhen', 999.0, 'completed')"
            )
            count_after = len(_select_all(view_name))
            assert count_after == count_before + 1
        finally:
            _drop(view_name)


# ============================================================================
# create_materialized_view tests
# ============================================================================


class TestCreateMaterializedView:
    """Test create_materialized_view() — TO form, target table is explicit."""

    def test_basic_mv_auto_creates_target(self, source_table):
        mv_name = "wb_mv_basic"
        target = "wb_mv_basic_target"
        _drop(mv_name)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            summary = (
                ds.filter(ds["status"] == "completed")
                .groupby("city")
                .agg(total=col("amount").sum(), cnt=col("amount").count())
            )
            summary.create_materialized_view(
                name=f"{DATABASE}.{mv_name}",
                to=f"{DATABASE}.{target}",
                engine="SummingMergeTree()",
                order_by="city",
                populate=True,
            )

            # populate=True backfilled existing rows into the target.
            result = _select_all(target, order_by="city")
            assert len(result) >= 3
            # Querying the MV name itself routes through the trigger to the
            # same target table, so counts must match.
            assert _count(mv_name) == _count(target)
        finally:
            _drop(mv_name)
            _drop(target)

    def test_mv_incremental_update_lands_in_target(self, source_table):
        """New rows inserted into the source must show up in the target table
        (and therefore in the MV) without manual intervention."""
        mv_name = "wb_mv_incremental"
        target = "wb_mv_incremental_target"
        _drop(mv_name)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.create_materialized_view(
                name=f"{DATABASE}.{mv_name}",
                to=f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by="city",
            )

            count_before = _count(target)

            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Shenzhen', 888.0, 'new')"
            )

            assert _count(target) == count_before + 1
        finally:
            _drop(mv_name)
            _drop(target)

    def test_mv_fail_when_target_exists_without_append(self, source_table):
        """if_target_exists='fail' (default) must reject a pre-existing target."""
        mv_name = "wb_mv_target_fail"
        target = "wb_mv_target_fail_target"
        _drop(mv_name)
        _drop(target)
        try:
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" (city String, x Int64) '
                f"ENGINE = MergeTree() ORDER BY city"
            )
            ds = _make_ds(source_table)
            with pytest.raises(QueryError, match="already exists"):
                ds.create_materialized_view(
                    name=f"{DATABASE}.{mv_name}",
                    to=f"{DATABASE}.{target}",
                    engine="MergeTree()",
                    order_by="city",
                )
        finally:
            _drop(mv_name)
            _drop(target)

    def test_mv_fail_when_mv_already_exists(self, source_table):
        """A second create_materialized_view with the same MV name must
        always fail — replace is intentionally not supported until ClickHouse
        ships CREATE OR REPLACE MATERIALIZED VIEW (PR #100539)."""
        mv_name = "wb_mv_already_exists"
        target = "wb_mv_already_exists_target"
        _drop(mv_name)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.create_materialized_view(
                name=f"{DATABASE}.{mv_name}",
                to=f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by="city",
            )
            with pytest.raises(QueryError, match="already exists"):
                ds.create_materialized_view(
                    name=f"{DATABASE}.{mv_name}",
                    to=f"{DATABASE}.{target}",
                    engine="MergeTree()",
                    order_by="city",
                    if_target_exists="append",
                )
        finally:
            _drop(mv_name)
            _drop(target)


# ============================================================================
# save() unified entry point tests
# ============================================================================


class TestSave:
    """Test save() unified entry point."""

    def test_save_as_table(self, source_table):
        target = "wb_save_table"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.save(f"{DATABASE}.{target}")

            assert _count(target) == 7
        finally:
            _drop(target)

    def test_save_as_table_replace(self, source_table):
        target = "wb_save_table_replace"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.save(f"{DATABASE}.{target}")
            assert _count(target) == 7

            ds.filter(ds["status"] == "completed").save(
                f"{DATABASE}.{target}", if_exists="replace"
            )
            assert _count(target) == 5
        finally:
            _drop(target)

    def test_save_as_table_append(self, source_table):
        target = "wb_save_table_append"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.save(f"{DATABASE}.{target}")
            ds.save(f"{DATABASE}.{target}", if_exists="append")
            assert _count(target) == 14
        finally:
            _drop(target)

    def test_save_as_view(self, source_table):
        view_name = "wb_save_view"
        _drop(view_name)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "completed").save(
                f"{DATABASE}.{view_name}", type="view", if_exists="replace"
            )

            result = _select_all(view_name)
            assert len(result) == 5
        finally:
            _drop(view_name)

    def test_save_as_mv(self, source_table):
        mv_name = "wb_save_mv"
        target = "wb_save_mv_target"
        _drop(mv_name)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.save(
                f"{DATABASE}.{mv_name}",
                type="materialized_view",
                to=f"{DATABASE}.{target}",
            )

            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Hangzhou', 777.0, 'completed')"
            )
            assert _count(target) >= 1
        finally:
            _drop(mv_name)
            _drop(target)

    def test_save_as_mv_requires_to(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="requires `to=`"):
            ds.save("some_mv", type="materialized_view")

    def test_save_as_mv_rejects_replace(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="if_exists='fail'"):
            ds.save(
                "some_mv",
                type="materialized_view",
                if_exists="replace",
                to="some_target",
            )

    def test_save_invalid_type(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Invalid type"):
            ds.save("some_table", type="invalid")

    def test_save_view_invalid_if_exists(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Views only support"):
            ds.save("some_view", type="view", if_exists="append")


# ============================================================================
# Schema evolution tests
# ============================================================================


class TestPureRemoteExecution:
    """Test that pure-remote (same server) pipelines run entirely on the remote server."""

    def test_replace_atomic_same_server(self, source_table):
        """replace on same-server uses CREATE OR REPLACE TABLE ... AS SELECT (atomic)."""
        target = "wb_pure_remote_replace"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds_filtered = ds.filter(ds["status"] == "completed")
            ds_filtered.to_clickhouse(
                f"{DATABASE}.{target}",
                if_exists="replace",
                engine="MergeTree()",
                order_by="city",
            )
            assert _count(target) == 5

            ds_all = _make_ds(source_table)
            ds_all.to_clickhouse(
                f"{DATABASE}.{target}",
                if_exists="replace",
                engine="MergeTree()",
                order_by="city",
            )
            assert _count(target) == 7
        finally:
            _drop(target)

    def test_fail_same_server(self, source_table):
        """fail on same-server creates table + inserts via remote SQL."""
        target = "wb_pure_remote_fail"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by="city",
            )
            assert _count(target) == 7
        finally:
            _drop(target)

    def test_append_same_server(self, source_table):
        """append on same-server inserts via remote SQL without local data transit."""
        target = "wb_pure_remote_append"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by="city",
            )
            assert _count(target) == 7

            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists="append")
            assert _count(target) == 14
        finally:
            _drop(target)

    def test_write_back_to_source_replace_atomic(self, source_table):
        """Writing back to source with replace is atomic — no data loss window."""
        ds = _make_ds(source_table)
        original_count = _count(source_table)
        assert original_count == 7

        ds_filtered = ds.filter(ds["status"] == "completed")
        ds_filtered.to_clickhouse(
            f"{DATABASE}.{source_table}",
            if_exists="replace", engine="MergeTree()", order_by="city"
        )
        assert _count(source_table) == 5
        result = _select_all(source_table)
        assert all(result["status"] == "completed")

    def test_groupby_agg_same_server(self, source_table):
        """groupby().agg() pipeline runs as pure remote SQL."""
        target = "wb_pure_remote_groupby"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            summary = (
                ds.filter(ds["status"] == "completed")
                .groupby("city")
                .agg(total=col("amount").sum(), cnt=col("amount").count())
            )
            summary.to_clickhouse(
                f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by="city",
            )
            result = _select_all(target, order_by="city")
            assert len(result) == 3

            bj = result[result["city"] == "Beijing"]
            assert float(bj["total"].iloc[0]) == 300.0
            assert int(bj["cnt"].iloc[0]) == 2
        finally:
            _drop(target)


class TestSchemaEvolution:
    """Test enable_schema_evolution in to_clickhouse."""

    def test_extra_column_auto_added(self):
        """Source has extra columns → target gets ALTER TABLE ADD COLUMN."""
        source = "wb_schema_source"
        target = "wb_schema_target"
        _drop(source)
        _drop(target)
        try:
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{source}" '
                f"(id UInt32, name String, email String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )
            _ddl(
                f'INSERT INTO "{DATABASE}"."{source}" VALUES '
                f"(1, 'Alice', 'alice@example.com'), (2, 'Bob', 'bob@example.com')"
            )
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" '
                f"(id UInt32, name String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = _make_ds(source)
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                if_exists="append",
                enable_schema_evolution=True,
            )

            schema = _query_remote(
                f"SELECT name, type FROM remote('{HOST}', 'system', 'columns', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND table = '{target}' "
                f"ORDER BY position"
            )
            col_names = list(schema["name"])
            assert "email" in col_names

            assert _count(target) == 2
        finally:
            _drop(source)
            _drop(target)


# ============================================================================
# Edge cases and error handling
# ============================================================================


class TestMixedPipelineWrite:
    """Test to_clickhouse when pipeline contains Pandas-only operations."""

    def test_write_after_apply_with_python_udf(self, source_table):
        """Pipeline with apply(callable) should fall back to DataFrame upload."""
        target = "wb_mixed_apply"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds_processed = ds.filter(ds["status"] == "completed")
            ds_processed = ds_processed.transform(lambda df: df.assign(
                amount_doubled=df["amount"] * 2
            ))
            ds_processed.to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city"
            )

            result = _select_all(target, order_by="city, amount")
            assert len(result) == 5

            bj = result[result["city"] == "Beijing"].sort_values("amount")
            assert float(bj.iloc[0]["amount_doubled"]) == 200.0
            assert float(bj.iloc[1]["amount_doubled"]) == 400.0
        finally:
            _drop(target)

    def test_write_after_boolean_mask_filter(self, source_table):
        """Pipeline with boolean mask (pandas array) should fall back to DataFrame upload."""
        target = "wb_mixed_mask"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            # Use pandas-style boolean mask: ds[mask_series]
            mask = ds["amount"] > 150
            ds_filtered = ds[mask]
            ds_filtered.to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city"
            )

            result = _select_all(target, order_by="city, amount")
            assert len(result) > 0
            assert all(result["amount"] > 150)
        finally:
            _drop(target)

    def test_mixed_pipeline_append_existing(self, source_table):
        """Mixed pipeline with if_exists='append' to existing table."""
        target = "wb_mixed_append"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city"
            )
            assert _count(target) == 7

            ds2 = _make_ds(source_table)
            ds_transformed = ds2.transform(lambda df: df.assign(
                amount=df["amount"].cumsum()
            ))
            ds_transformed.to_clickhouse(
                f"{DATABASE}.{target}", if_exists="append"
            )
            assert _count(target) == 14  # 7 original + 7 transformed
        finally:
            _drop(target)

    def test_verify_schema_with_describe(self, source_table):
        """Verify that DESCRIBE-based schema produces correct ClickHouse types."""
        target = "wb_describe_schema"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            summary = (
                ds.filter(ds["status"] == "completed")
                .groupby("city")
                .agg(total=col("amount").sum(), cnt=col("amount").count())
            )
            summary.to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city"
            )

            # Verify schema types on the remote table
            schema = _query_remote(
                f"SELECT name, type FROM remote('{HOST}', 'system', 'columns', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND table = '{target}' "
                f"ORDER BY position"
            )
            schema_dict = dict(zip(schema["name"], schema["type"]))

            assert schema_dict["city"] == "String"
            assert schema_dict["total"] == "Float64"
            assert schema_dict["cnt"] == "Int64"

            result = _select_all(target, order_by="city")
            assert len(result) == 3
        finally:
            _drop(target)


class TestWritebackEdgeCases:
    """Test error handling and edge cases."""

    def test_no_connection_params(self):
        """to_clickhouse on a local DataFrame DataStore should fail gracefully."""
        ds = DataStore(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(Exception):
            ds.to_clickhouse("some_table")

    def test_invalid_if_exists(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Invalid if_exists"):
            ds.to_clickhouse("some_table", if_exists="invalid")

    def test_truncate_mode_removed(self, source_table):
        """truncate mode is no longer supported."""
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Invalid if_exists"):
            ds.to_clickhouse("some_table", if_exists="truncate")
