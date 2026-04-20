"""
Tests for writeback APIs: to_clickhouse, create_view, create_materialized_view, save.

All tests run against a remote ClickHouse server. Connection is configured
via environment variables.

Source server (always required — most tests skip without it):
  CHDB_TEST_HOST            host:port
  CHDB_TEST_USER            user
  CHDB_TEST_PASSWORD        password
  CHDB_TEST_DATABASE        database

Target server (optional — only the cross-server suite needs it). Set this to a
*different* ClickHouse instance with *different* credentials so a misrouted
connection (source creds sent to target, etc.) gets rejected immediately
instead of silently passing:

  CHDB_TEST_TARGET_HOST     host:port  (different port from source)
  CHDB_TEST_TARGET_USER     user
  CHDB_TEST_TARGET_PASSWORD password
  CHDB_TEST_TARGET_DATABASE database

Tests requiring the target server skip automatically when these are unset.
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
from datastore.exceptions import (
    DataStoreError,
    ExecutionError,
    QueryError,
)


# ---------------------------------------------------------------------------
# Server config (env-driven)
# ---------------------------------------------------------------------------


class _Server:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def configured(self):
        return bool(self.host and self.user and self.database)


SOURCE = _Server(
    host=os.environ.get("CHDB_TEST_HOST", ""),
    user=os.environ.get("CHDB_TEST_USER", ""),
    password=os.environ.get("CHDB_TEST_PASSWORD", ""),
    database=os.environ.get("CHDB_TEST_DATABASE", ""),
)
TARGET = _Server(
    host=os.environ.get("CHDB_TEST_TARGET_HOST", ""),
    user=os.environ.get("CHDB_TEST_TARGET_USER", ""),
    password=os.environ.get("CHDB_TEST_TARGET_PASSWORD", ""),
    database=os.environ.get("CHDB_TEST_TARGET_DATABASE", ""),
)

# Back-compat shortcuts so single-server helpers stay terse.
HOST, USER, PASSWORD, DATABASE = SOURCE.host, SOURCE.user, SOURCE.password, SOURCE.database

requires_target = pytest.mark.skipif(
    not TARGET.configured(),
    reason="CHDB_TEST_TARGET_* not set — cross-server tests skipped",
)


# ---------------------------------------------------------------------------
# Server-aware helpers
# ---------------------------------------------------------------------------


def _remote_func(srv: _Server, db: str, table: str) -> str:
    return f"remote('{srv.host}', '{db}', '{table}', '{srv.user}', '{srv.password}')"


def _ddl(sql: str, srv: _Server = SOURCE) -> None:
    """Execute DDL on a server via remote(query=...)."""
    escaped = sql.replace("\\", "\\\\").replace("'", "\\'")
    chdb.query(
        f"SELECT * FROM remote('{srv.host}', "
        f"query = '{escaped}', '{srv.user}', '{srv.password}')",
        "TabSeparated",
    )


def _q(sql: str):
    return chdb.query(sql, output_format="DataFrame")


def _drop(table: str, db: str = DATABASE, srv: _Server = SOURCE) -> None:
    _ddl(f'DROP TABLE IF EXISTS "{db}"."{table}"', srv=srv)


def _drop_view(name: str, db: str = DATABASE, srv: _Server = SOURCE) -> None:
    _ddl(f'DROP VIEW IF EXISTS "{db}"."{name}"', srv=srv)


def _count(table: str, db: str = DATABASE, srv: _Server = SOURCE) -> int:
    result = _q(f"SELECT count() FROM {_remote_func(srv, db, table)}")
    return int(result.iloc[0, 0])


def _select_all(table: str, db: str = DATABASE, order_by: str = None,
                srv: _Server = SOURCE):
    sql = f"SELECT * FROM {_remote_func(srv, db, table)}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    return _q(sql)


def _columns(table: str, db: str = DATABASE, srv: _Server = SOURCE):
    """Return [(name, type), ...] from system.columns on the given server."""
    sql = (
        f"SELECT name, type FROM remote('{srv.host}', 'system', 'columns', "
        f"'{srv.user}', '{srv.password}') "
        f"WHERE database = '{db}' AND table = '{table}' ORDER BY position"
    )
    df = _q(sql)
    return list(zip(df["name"], df["type"]))


def _make_ds(table: str = None, db: str = DATABASE, srv: _Server = SOURCE):
    return DataStore.from_clickhouse(
        host=srv.host, database=db, table=table,
        user=srv.user, password=srv.password,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _check_source_server():
    if not SOURCE.configured():
        pytest.skip(
            "Source ClickHouse env vars not set "
            "(CHDB_TEST_HOST, CHDB_TEST_USER, CHDB_TEST_DATABASE)"
        )
    try:
        _q(f"SELECT 1 FROM remote('{SOURCE.host}', 'system', 'one', "
           f"'{SOURCE.user}', '{SOURCE.password}')")
    except Exception as e:
        pytest.skip(f"Source ClickHouse unreachable: {e}")


@pytest.fixture()
def source_table():
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
# Parameter / connection validation (does not need a working pipeline)
# ============================================================================


class TestParameterValidation:
    def test_to_clickhouse_rejects_unknown_if_exists(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Invalid if_exists"):
            ds.to_clickhouse(f"{DATABASE}.t", if_exists="bogus")

    def test_to_clickhouse_rejects_legacy_truncate(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Invalid if_exists"):
            ds.to_clickhouse(f"{DATABASE}.t", if_exists="truncate")

    def test_save_rejects_unknown_type(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Invalid type"):
            ds.save(f"{DATABASE}.t", type="bogus")

    def test_save_view_rejects_append(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="Views only support"):
            ds.save(f"{DATABASE}.t", type="view", if_exists="append")

    def test_save_mv_rejects_replace(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="if_exists='fail'"):
            ds.save(f"{DATABASE}.t", type="materialized_view",
                    if_exists="replace", to=f"{DATABASE}.tgt")

    def test_save_mv_requires_to(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="requires `to=`"):
            ds.save(f"{DATABASE}.t", type="materialized_view")

    def test_create_mv_rejects_unknown_if_target_exists(self, source_table):
        ds = _make_ds(source_table)
        with pytest.raises(ValueError, match="if_target_exists must be"):
            ds.create_materialized_view(
                name=f"{DATABASE}.t",
                to=f"{DATABASE}.tgt",
                if_target_exists="bogus",
            )

    def test_local_dataframe_datastore_has_no_connection(self):
        """Writeback APIs need a remote connection — pure-pandas DataStores
        must fail with DataStoreError, not silently no-op."""
        ds = DataStore(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(DataStoreError):
            ds.to_clickhouse(f"{DATABASE}.t")
        with pytest.raises(DataStoreError):
            ds.create_view(f"{DATABASE}.v")
        with pytest.raises(DataStoreError):
            ds.create_materialized_view(name=f"{DATABASE}.mv",
                                        to=f"{DATABASE}.tgt")


# ============================================================================
# Internal helpers — small but exercised on every writeback call
# ============================================================================


class TestInternalHelpers:
    def test_parse_target_name_db_dot_table(self, source_table):
        ds = _make_ds(source_table)
        assert ds._parse_target_name("a.b") == ("a", "b")

    def test_parse_target_name_falls_back_to_table_function_db(self, source_table):
        """from_clickhouse's database flows into _parse_target_name when no
        explicit db is given."""
        ds = _make_ds(source_table)
        db, table = ds._parse_target_name("just_a_name")
        assert db == DATABASE
        assert table == "just_a_name"

    def test_parse_target_name_unresolvable_db_raises(self):
        """A DataStore with no source-table context can't resolve a default db."""
        ds = DataStore.from_clickhouse(
            host=SOURCE.host, user=SOURCE.user, password=SOURCE.password,
        )
        with pytest.raises(QueryError, match="Cannot resolve database"):
            ds._parse_target_name("just_a_name")

    def test_check_remote_table_exists_true_and_false(self, source_table):
        ds = _make_ds(source_table)
        assert ds._check_remote_table_exists(DATABASE, source_table) is True
        assert ds._check_remote_table_exists(
            DATABASE, "wb_definitely_does_not_exist_xyz"
        ) is False


# ============================================================================
# to_clickhouse — same-server fully-SQL paths
# ============================================================================


class TestToClickHouseSameServer:
    """Pipeline lives on the source server and is fully SQL-pushable.
    Covers the fast path: pure remote DDL + INSERT...SELECT, no local
    DataFrame round trip."""

    def test_fail_creates_new_table(self, source_table):
        target = "wb_ss_fail_new"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}",
                             engine="MergeTree()", order_by="city")
            assert _count(target) == 7
        finally:
            _drop(target)

    def test_fail_when_target_exists_raises(self, source_table):
        """fail mode lets ClickHouse reject CREATE TABLE naturally —
        manifests as ExecutionError (not QueryError) since no library-side
        pre-check is performed for fail."""
        target = "wb_ss_fail_existing"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}",
                             engine="MergeTree()", order_by="city")
            with pytest.raises(ExecutionError, match="already exists"):
                ds.to_clickhouse(f"{DATABASE}.{target}",
                                 engine="MergeTree()", order_by="city")
        finally:
            _drop(target)

    def test_replace_new_table_via_create_or_replace(self, source_table):
        target = "wb_ss_replace_new"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                if_exists="replace", engine="MergeTree()", order_by="city",
            )
            assert _count(target) == 7
        finally:
            _drop(target)

    def test_replace_overwrites_existing_atomically(self, source_table):
        target = "wb_ss_replace_existing"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}",
                             engine="MergeTree()", order_by="city")
            assert _count(target) == 7

            ds.filter(ds["status"] == "completed").to_clickhouse(
                f"{DATABASE}.{target}",
                if_exists="replace", engine="MergeTree()", order_by="city",
            )
            assert _count(target) == 5
            result = _select_all(target)
            assert all(result["status"] == "completed")
        finally:
            _drop(target)

    def test_replace_writes_back_to_source_atomically(self, source_table):
        ds = _make_ds(source_table)
        ds.filter(ds["status"] == "completed").to_clickhouse(
            f"{DATABASE}.{source_table}",
            if_exists="replace", engine="MergeTree()", order_by="city",
        )
        assert _count(source_table) == 5
        result = _select_all(source_table)
        assert all(result["status"] == "completed")

    def test_append_creates_then_inserts(self, source_table):
        target = "wb_ss_append_create"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists="append",
                             engine="MergeTree()", order_by="city")
            assert _count(target) == 7
            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists="append")
            assert _count(target) == 14
        finally:
            _drop(target)

    def test_filter_sort_pipeline(self, source_table):
        target = "wb_ss_filter_sort"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds.filter(ds["status"] == "completed")
               .sort("amount", ascending=False)).to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city",
            )
            result = _select_all(target, order_by="amount DESC")
            assert len(result) == 5
            assert float(result.iloc[0]["amount"]) == 400.0
        finally:
            _drop(target)

    def test_groupby_agg_pipeline(self, source_table):
        target = "wb_ss_groupby"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            summary = (ds.filter(ds["status"] == "completed")
                         .groupby("city")
                         .agg(total=col("amount").sum(),
                              cnt=col("amount").count()))
            summary.to_clickhouse(f"{DATABASE}.{target}",
                                  engine="MergeTree()", order_by="city")
            result = _select_all(target, order_by="city")
            assert len(result) == 3

            bj = result[result["city"] == "Beijing"]
            assert float(bj["total"].iloc[0]) == 300.0
            assert int(bj["cnt"].iloc[0]) == 2

            schema = dict(_columns(target))
            assert schema["city"].endswith("String")
            assert "Float64" in schema["total"]
        finally:
            _drop(target)

    def test_order_by_list_and_partition_by(self, source_table):
        target = "wb_ss_order_partition"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                engine="MergeTree()",
                order_by=["city", "amount"],
                partition_by="status",
            )
            assert _count(target) == 7

            sql_parts = _q(
                f"SELECT engine_full FROM remote('{HOST}', 'system', 'tables',"
                f" '{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND name = '{target}'"
            )
            engine_full = str(sql_parts.iloc[0, 0])
            assert "PARTITION BY" in engine_full
            assert "status" in engine_full
        finally:
            _drop(target)

    def test_default_engine_is_mergetree(self, source_table):
        target = "wb_ss_default_engine"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}", order_by="city")
            engine = _q(
                f"SELECT engine FROM remote('{HOST}', 'system', 'tables', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND name = '{target}'"
            ).iloc[0, 0]
            assert engine == "MergeTree"
        finally:
            _drop(target)

    def test_summing_mergetree(self, source_table):
        target = "wb_ss_summing"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds.groupby("city").agg(total=col("amount").sum())
             ).to_clickhouse(
                f"{DATABASE}.{target}",
                engine="SummingMergeTree()", order_by="city",
            )
            engine = _q(
                f"SELECT engine FROM remote('{HOST}', 'system', 'tables', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND name = '{target}'"
            ).iloc[0, 0]
            assert engine == "SummingMergeTree"
        finally:
            _drop(target)

    def test_unqualified_target_uses_default_database(self, source_table):
        """Target name without 'db.' should pick up from_clickhouse's database."""
        target = "wb_ss_unqualified"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(target, engine="MergeTree()", order_by="city")
            assert _count(target) == 7
        finally:
            _drop(target)


# ============================================================================
# to_clickhouse — DataFrame-upload fallback (pandas-only ops in the pipeline)
# ============================================================================


class TestToClickHouseDataFrameUpload:
    """Pipelines that contain pandas-only ops can't be SQL-pushed; the impl
    materializes locally and re-uploads via the Python() table function.
    These exercise the non-fully_sql branch end-to-end."""

    def test_python_udf_via_transform(self, source_table):
        target = "wb_df_udf"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds_udf = (ds.filter(ds["status"] == "completed")
                        .transform(lambda df: df.assign(
                            amount_doubled=df["amount"] * 2)))
            ds_udf.to_clickhouse(f"{DATABASE}.{target}",
                                 engine="MergeTree()", order_by="city")

            schema = dict(_columns(target))
            assert "amount_doubled" in schema

            result = _select_all(target, order_by="city, amount")
            assert len(result) == 5
            assert all(result["amount_doubled"] == result["amount"] * 2)
        finally:
            _drop(target)

    def test_replace_dataframe_path_when_target_missing_creates_it(
        self, source_table
    ):
        """Pandas-only pipeline + replace + target absent → DROP+CREATE+INSERT
        is allowed (no atomic CREATE OR REPLACE here, but no data to lose
        either)."""
        target = "wb_df_replace_new"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds.transform(lambda df: df.assign(extra=df["amount"] + 1))
             ).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="replace",
                engine="MergeTree()", order_by="city",
            )
            assert _count(target) == 7
            assert "extra" in dict(_columns(target))
        finally:
            _drop(target)

    def test_replace_dataframe_path_refuses_to_overwrite_existing(
        self, source_table
    ):
        """Pandas-only pipeline + replace + target present → no atomic path
        available, so we refuse rather than risk a non-atomic DROP+CREATE
        window."""
        target = "wb_df_replace_existing"
        _drop(target)
        try:
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" (a Int64) '
                f"ENGINE = MergeTree() ORDER BY a"
            )
            ds = _make_ds(source_table)
            with pytest.raises(QueryError,
                               match="Cannot atomically replace"):
                (ds.transform(lambda df: df.assign(extra=1))
                 ).to_clickhouse(
                    f"{DATABASE}.{target}", if_exists="replace",
                    engine="MergeTree()", order_by="city",
                )
        finally:
            _drop(target)

    def test_append_dataframe_path_to_existing(self, source_table):
        target = "wb_df_append"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(f"{DATABASE}.{target}",
                             engine="MergeTree()", order_by="city")
            (ds.transform(lambda df: df.assign(amount=df["amount"] + 1))
             ).to_clickhouse(f"{DATABASE}.{target}", if_exists="append")
            assert _count(target) == 14
        finally:
            _drop(target)

    def test_index_true_writes_index_column(self, source_table):
        """index=True must surface the pandas index as a real column on the
        target table (default name = 'index')."""
        target = "wb_df_index"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds.transform(lambda df: df.assign(rn=range(len(df))).set_index("rn"))
             ).to_clickhouse(
                f"{DATABASE}.{target}", index=True,
                engine="MergeTree()", order_by="city",
            )
            cols = dict(_columns(target))
            assert "rn" in cols
            assert _count(target) == 7
        finally:
            _drop(target)

    def test_index_label_renames_index_column(self, source_table):
        target = "wb_df_index_label"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds.transform(lambda df:
                          df.assign(_rn=range(len(df))).set_index("_rn"))
             ).to_clickhouse(
                f"{DATABASE}.{target}", index=True, index_label="row_id",
                engine="MergeTree()", order_by="city",
            )
            cols = dict(_columns(target))
            assert "row_id" in cols
            assert "_rn" not in cols
        finally:
            _drop(target)


# ============================================================================
# Schema evolution
# ============================================================================


class TestSchemaEvolution:
    def test_missing_columns_added_via_alter(self):
        """Pipeline produces a column the target table doesn't have →
        ALTER TABLE ADD COLUMN before INSERT."""
        source = "wb_evo_source"
        target = "wb_evo_target"
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
                f"(1, 'Alice', 'a@x'), (2, 'Bob', 'b@x')"
            )
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" '
                f"(id UInt32, name String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )
            _ddl(
                f'INSERT INTO "{DATABASE}"."{target}" VALUES '
                f"(99, 'pre-existing')"
            )

            _make_ds(source).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="append",
                enable_schema_evolution=True,
            )

            cols = dict(_columns(target))
            assert "email" in cols, "schema evolution did not add the new column"
            assert _count(target) == 3

            existing_email = _q(
                f"SELECT email FROM {_remote_func(SOURCE, DATABASE, target)} "
                f"WHERE id = 99"
            )
            # CH default zero value for String is empty string; no Nullable wrap.
            assert existing_email.iloc[0, 0] == ""
        finally:
            _drop(source)
            _drop(target)

    def test_evolution_uses_pipeline_output_not_source_schema(self):
        """Renames / projections in the pipeline must drive evolution —
        comparing against the *source* schema would miss them."""
        source = "wb_evo_pipeline_src"
        target = "wb_evo_pipeline_tgt"
        _drop(source)
        _drop(target)
        try:
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{source}" (id UInt32, v Int64) '
                f"ENGINE = MergeTree() ORDER BY id"
            )
            _ddl(
                f'INSERT INTO "{DATABASE}"."{source}" VALUES (1, 10), (2, 20)'
            )
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" (id UInt32) '
                f"ENGINE = MergeTree() ORDER BY id"
            )

            ds = _make_ds(source)
            (ds.select("id", (col("v") * 2).as_("v2"))
             ).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="append",
                enable_schema_evolution=True,
            )

            cols = dict(_columns(target))
            assert "v2" in cols
            assert "v" not in cols
        finally:
            _drop(source)
            _drop(target)


# ============================================================================
# create_view
# ============================================================================


class TestCreateView:
    def test_basic_view_reflects_filter(self, source_table):
        view = "wb_view_basic"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "completed").create_view(
                f"{DATABASE}.{view}"
            )
            result = _select_all(view, order_by="city, amount")
            assert len(result) == 5
            assert all(result["status"] == "completed")
        finally:
            _drop_view(view)

    def test_view_with_groupby(self, source_table):
        view = "wb_view_groupby"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            (ds.filter(ds["status"] == "completed")
               .groupby("city")
               .agg(total=col("amount").sum())
            ).create_view(f"{DATABASE}.{view}")
            result = _select_all(view, order_by="city")
            bj = result[result["city"] == "Beijing"]
            assert float(bj["total"].iloc[0]) == 300.0
        finally:
            _drop_view(view)

    def test_view_replace_swaps_definition(self, source_table):
        view = "wb_view_replace"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "completed").create_view(
                f"{DATABASE}.{view}"
            )
            assert len(_select_all(view)) == 5

            ds.filter(ds["status"] == "pending").create_view(
                f"{DATABASE}.{view}", replace=True,
            )
            after = _select_all(view)
            assert len(after) == 2
            assert all(after["status"] == "pending")
        finally:
            _drop_view(view)

    def test_view_replace_false_on_existing_raises(self, source_table):
        view = "wb_view_no_replace"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            ds.create_view(f"{DATABASE}.{view}")
            with pytest.raises(ExecutionError, match="already exists"):
                ds.create_view(f"{DATABASE}.{view}", replace=False)
        finally:
            _drop_view(view)

    def test_view_reflects_subsequent_source_inserts(self, source_table):
        view = "wb_view_live"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            ds.create_view(f"{DATABASE}.{view}")
            before = _count(view)

            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Shenzhen', 999.0, 'completed')"
            )
            assert _count(view) == before + 1
        finally:
            _drop_view(view)

    def test_view_pandas_only_pipeline_rejected(self, source_table):
        """Views save SQL — a pipeline that requires Python execution can't
        be pushed and must error out cleanly, not silently materialize."""
        view = "wb_view_pandas_only"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            with pytest.raises((QueryError, DataStoreError)):
                (ds.transform(lambda df: df.assign(x=1))
                 ).create_view(f"{DATABASE}.{view}")
        finally:
            _drop_view(view)


# ============================================================================
# create_materialized_view
# ============================================================================


class TestCreateMaterializedView:
    """All MV tests use the TO form; replace semantics are intentionally
    unsupported (see method docstring)."""

    def test_default_engine_is_mergetree(self, source_table):
        mv, target = "wb_mv_engine_default", "wb_mv_engine_default_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            _make_ds(source_table).create_materialized_view(
                name=f"{DATABASE}.{mv}",
                to=f"{DATABASE}.{target}",
                order_by="city",
            )
            engine = _q(
                f"SELECT engine FROM remote('{HOST}', 'system', 'tables', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND name = '{target}'"
            ).iloc[0, 0]
            assert engine == "MergeTree"
        finally:
            _drop_view(mv)
            _drop(target)

    def test_incremental_writes_land_in_target(self, source_table):
        mv, target = "wb_mv_incr", "wb_mv_incr_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            _make_ds(source_table).create_materialized_view(
                name=f"{DATABASE}.{mv}",
                to=f"{DATABASE}.{target}",
                engine="MergeTree()", order_by="city",
            )
            before = _count(target)
            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Shenzhen', 888.0, 'new')"
            )
            assert _count(target) == before + 1
        finally:
            _drop_view(mv)
            _drop(target)

    def test_populate_backfills_existing_rows(self, source_table):
        mv, target = "wb_mv_populate", "wb_mv_populate_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            _make_ds(source_table).create_materialized_view(
                name=f"{DATABASE}.{mv}",
                to=f"{DATABASE}.{target}",
                engine="MergeTree()", order_by="city",
                populate=True,
            )
            assert _count(target) == 7
        finally:
            _drop_view(mv)
            _drop(target)

    def test_partition_by_propagates_to_target(self, source_table):
        mv, target = "wb_mv_partition", "wb_mv_partition_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            _make_ds(source_table).create_materialized_view(
                name=f"{DATABASE}.{mv}",
                to=f"{DATABASE}.{target}",
                engine="MergeTree()", order_by="city",
                partition_by="status",
            )
            engine_full = _q(
                f"SELECT engine_full FROM remote('{HOST}', 'system', 'tables', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND name = '{target}'"
            ).iloc[0, 0]
            assert "PARTITION BY" in str(engine_full)
        finally:
            _drop_view(mv)
            _drop(target)

    def test_target_exists_fail_raises(self, source_table):
        mv, target = "wb_mv_tgt_fail", "wb_mv_tgt_fail_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" (city String, x Int64) '
                f"ENGINE = MergeTree() ORDER BY city"
            )
            with pytest.raises(QueryError, match="already exists"):
                _make_ds(source_table).create_materialized_view(
                    name=f"{DATABASE}.{mv}",
                    to=f"{DATABASE}.{target}",
                    engine="MergeTree()", order_by="city",
                )
        finally:
            _drop_view(mv)
            _drop(target)

    def test_target_exists_append_fan_in(self, source_table):
        """if_target_exists='append' lets multiple MVs feed one target table."""
        mv1, mv2, target = "wb_mv_fanin1", "wb_mv_fanin2", "wb_mv_fanin_tgt"
        _drop_view(mv1)
        _drop_view(mv2)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "completed").create_materialized_view(
                name=f"{DATABASE}.{mv1}", to=f"{DATABASE}.{target}",
                engine="MergeTree()", order_by="city", populate=True,
            )
            ds.filter(ds["status"] == "pending").create_materialized_view(
                name=f"{DATABASE}.{mv2}", to=f"{DATABASE}.{target}",
                if_target_exists="append",
                engine="MergeTree()", order_by="city", populate=True,
            )
            assert _count(target) == 7
        finally:
            _drop_view(mv1)
            _drop_view(mv2)
            _drop(target)

    def test_mv_already_exists_always_fails(self, source_table):
        mv, target = "wb_mv_dup", "wb_mv_dup_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.create_materialized_view(
                name=f"{DATABASE}.{mv}", to=f"{DATABASE}.{target}",
                engine="MergeTree()", order_by="city",
            )
            with pytest.raises(QueryError, match="already exists"):
                ds.create_materialized_view(
                    name=f"{DATABASE}.{mv}", to=f"{DATABASE}.{target}",
                    if_target_exists="append",
                    engine="MergeTree()", order_by="city",
                )
        finally:
            _drop_view(mv)
            _drop(target)

    def test_pandas_only_pipeline_rejected(self, source_table):
        mv, target = "wb_mv_pandas_only", "wb_mv_pandas_only_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            with pytest.raises((QueryError, DataStoreError)):
                (ds.transform(lambda df: df.assign(x=1))
                 ).create_materialized_view(
                    name=f"{DATABASE}.{mv}", to=f"{DATABASE}.{target}",
                    engine="MergeTree()", order_by="city",
                )
        finally:
            _drop_view(mv)
            _drop(target)


# ============================================================================
# save() — delegation matrix
# ============================================================================


class TestSaveDelegation:
    def test_table_default_fail(self, source_table):
        target = "wb_save_table"
        _drop(target)
        try:
            _make_ds(source_table).save(f"{DATABASE}.{target}")
            assert _count(target) == 7
        finally:
            _drop(target)

    def test_table_replace(self, source_table):
        target = "wb_save_table_replace"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.save(f"{DATABASE}.{target}")
            ds.filter(ds["status"] == "completed").save(
                f"{DATABASE}.{target}", if_exists="replace"
            )
            assert _count(target) == 5
        finally:
            _drop(target)

    def test_table_append_enables_schema_evolution(self):
        """save(type='table', if_exists='append') must auto-enable schema
        evolution — that's its whole differentiation from to_clickhouse."""
        source = "wb_save_evo_src"
        target = "wb_save_evo_tgt"
        _drop(source)
        _drop(target)
        try:
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{source}" '
                f"(id UInt32, name String, age UInt32) "
                f"ENGINE = MergeTree() ORDER BY id"
            )
            _ddl(
                f'INSERT INTO "{DATABASE}"."{source}" VALUES '
                f"(1, 'Alice', 30)"
            )
            _ddl(
                f'CREATE TABLE "{DATABASE}"."{target}" '
                f"(id UInt32, name String) "
                f"ENGINE = MergeTree() ORDER BY id"
            )
            _make_ds(source).save(f"{DATABASE}.{target}", if_exists="append")
            assert "age" in dict(_columns(target))
        finally:
            _drop(source)
            _drop(target)

    def test_view_replace(self, source_table):
        view = "wb_save_view"
        _drop_view(view)
        try:
            (_make_ds(source_table)
                .filter(col("status") == "completed")
            ).save(f"{DATABASE}.{view}", type="view", if_exists="replace")
            assert _count(view) == 5
        finally:
            _drop_view(view)

    def test_materialized_view(self, source_table):
        mv, target = "wb_save_mv", "wb_save_mv_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            _make_ds(source_table).save(
                f"{DATABASE}.{mv}",
                type="materialized_view",
                to=f"{DATABASE}.{target}",
            )
            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Hangzhou', 777.0, 'completed')"
            )
            assert _count(target) >= 1
        finally:
            _drop_view(mv)
            _drop(target)


# ============================================================================
# Cross-server writeback (target host explicitly different from source)
# ============================================================================


@requires_target
class TestCrossServerToClickHouse:
    """Pipeline source lives on SOURCE; target lives on TARGET — different
    server, different credentials. Exercises the cross-server branches of
    _to_clickhouse_impl."""

    @pytest.fixture(autouse=True)
    def _check_target(self):
        try:
            _q(
                f"SELECT 1 FROM remote('{TARGET.host}', 'system', 'one', "
                f"'{TARGET.user}', '{TARGET.password}')"
            )
        except Exception as e:
            pytest.skip(f"Target ClickHouse unreachable: {e}")

    def _drop_target(self, name):
        _drop(name, db=TARGET.database, srv=TARGET)

    def test_fail_creates_table_on_target_server(self, source_table):
        target = "wb_cs_fail_new"
        self._drop_target(target)
        try:
            _make_ds(source_table).to_clickhouse(
                f"{TARGET.database}.{target}",
                engine="MergeTree()", order_by="city",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            assert _count(target, db=TARGET.database, srv=TARGET) == 7
            # And the table must NOT exist on the source server.
            assert _count_or_zero(target, db=DATABASE, srv=SOURCE) == 0
        finally:
            self._drop_target(target)

    def test_replace_fully_sql_uses_create_or_replace_on_target(
        self, source_table
    ):
        target = "wb_cs_replace_fully_sql"
        self._drop_target(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{TARGET.database}.{target}",
                engine="MergeTree()", order_by="city",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            ds.filter(ds["status"] == "completed").to_clickhouse(
                f"{TARGET.database}.{target}",
                if_exists="replace",
                engine="MergeTree()", order_by="city",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            assert _count(target, db=TARGET.database, srv=TARGET) == 5
        finally:
            self._drop_target(target)

    def test_replace_dataframe_path_refuses_existing_target(self, source_table):
        target = "wb_cs_replace_df_existing"
        self._drop_target(target)
        try:
            _ddl(
                f'CREATE TABLE "{TARGET.database}"."{target}" (a Int64) '
                f"ENGINE = MergeTree() ORDER BY a", srv=TARGET,
            )
            ds = _make_ds(source_table)
            with pytest.raises(QueryError, match="Cannot atomically replace"):
                (ds.transform(lambda df: df.assign(extra=1))
                 ).to_clickhouse(
                    f"{TARGET.database}.{target}",
                    if_exists="replace",
                    engine="MergeTree()", order_by="city",
                    host=TARGET.host, user=TARGET.user,
                    password=TARGET.password,
                )
        finally:
            self._drop_target(target)

    def test_append_cross_server(self, source_table):
        target = "wb_cs_append"
        self._drop_target(target)
        try:
            ds = _make_ds(source_table)
            ds.to_clickhouse(
                f"{TARGET.database}.{target}",
                engine="MergeTree()", order_by="city",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            ds.to_clickhouse(
                f"{TARGET.database}.{target}",
                if_exists="append",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            assert _count(target, db=TARGET.database, srv=TARGET) == 14
        finally:
            self._drop_target(target)

    def test_dataframe_path_cross_server(self, source_table):
        """Pandas-only pipeline + cross-server target → DataFrame is uploaded
        directly to the target."""
        target = "wb_cs_df_upload"
        self._drop_target(target)
        try:
            ds = _make_ds(source_table)
            (ds.transform(lambda df:
                          df.assign(amount_doubled=df["amount"] * 2))
             ).to_clickhouse(
                f"{TARGET.database}.{target}",
                engine="MergeTree()", order_by="city",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            cols = dict(_columns(target, db=TARGET.database, srv=TARGET))
            assert "amount_doubled" in cols
            assert _count(target, db=TARGET.database, srv=TARGET) == 7
        finally:
            self._drop_target(target)


@requires_target
class TestCrossServerCreateView:
    @pytest.fixture(autouse=True)
    def _check_target(self):
        try:
            _q(
                f"SELECT 1 FROM remote('{TARGET.host}', 'system', 'one', "
                f"'{TARGET.user}', '{TARGET.password}')"
            )
        except Exception as e:
            pytest.skip(f"Target ClickHouse unreachable: {e}")

    def test_view_lives_on_target_server(self, source_table):
        view = "wb_cs_view"
        _drop_view(view, db=TARGET.database, srv=TARGET)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "completed").create_view(
                f"{TARGET.database}.{view}",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
            )
            assert _count(view, db=TARGET.database, srv=TARGET) == 5
            # View must NOT exist on the source server.
            assert _count_or_zero(view, db=DATABASE, srv=SOURCE) == 0
        finally:
            _drop_view(view, db=TARGET.database, srv=TARGET)


@requires_target
class TestCrossServerCreateMaterializedView:
    @pytest.fixture(autouse=True)
    def _check_target(self):
        try:
            _q(
                f"SELECT 1 FROM remote('{TARGET.host}', 'system', 'one', "
                f"'{TARGET.user}', '{TARGET.password}')"
            )
        except Exception as e:
            pytest.skip(f"Target ClickHouse unreachable: {e}")

    def test_cross_server_mv_rejected_by_clickhouse(self, source_table):
        """Cross-server MV is conceptually broken: an MV is triggered by
        INSERTs on its source table, but if the source lives on another
        server the trigger never fires. ClickHouse itself rejects this with
        ``QUERY_IS_NOT_SUPPORTED_IN_MATERIALIZED_VIEW`` because MV's SELECT
        cannot reference a ``remote()`` table function. We just propagate
        that error so the user gets a deterministic failure rather than a
        silently dead MV."""
        mv, target = "wb_cs_mv_rejected", "wb_cs_mv_rejected_tgt"
        _drop_view(mv, db=TARGET.database, srv=TARGET)
        _drop(target, db=TARGET.database, srv=TARGET)
        try:
            with pytest.raises(
                ExecutionError,
                match=r"QUERY_IS_NOT_SUPPORTED_IN_MATERIALIZED_VIEW",
            ):
                _make_ds(source_table).create_materialized_view(
                    name=f"{TARGET.database}.{mv}",
                    to=f"{TARGET.database}.{target}",
                    engine="MergeTree()", order_by="city",
                    host=TARGET.host, user=TARGET.user,
                    password=TARGET.password,
                )
        finally:
            _drop_view(mv, db=TARGET.database, srv=TARGET)
            _drop(target, db=TARGET.database, srv=TARGET)


# ---------------------------------------------------------------------------
# Helper used only by cross-server tests — silently returns 0 if the table
# doesn't exist (we use this to assert *absence* on the wrong server).
# ---------------------------------------------------------------------------


def _count_or_zero(table: str, db: str, srv: _Server) -> int:
    try:
        return _count(table, db=db, srv=srv)
    except Exception:
        return 0
