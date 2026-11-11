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
from tests.test_utils import assert_frame_equal
from datastore.exceptions import (
    DataStoreError,
    ExecutionError,
    QueryError,
)
from tests.test_utils import assert_frame_equal


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
    # Escape as a ClickHouse string literal the same way production code does
    # (escape backslashes, then double single quotes) rather than backslash-
    # escaping quotes — independent of any backslash-escape parser setting.
    escaped = sql.replace("\\", "\\\\").replace("'", "''")
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


def _assert_written(table: str, expected: pd.DataFrame, key: list) -> None:
    """Compare a written table with its pandas mirror. A table has no row
    order, so both sides are ordered by ``key``, which must be unique."""
    result = _select_all(table, order_by=", ".join(key))
    assert_frame_equal(result, expected.sort_values(key, ignore_index=True))


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

    def test_projection_writes_only_selected_columns(self, source_table):
        """A column projection must persist only the selected columns. The
        writeback SELECT has to honor lazy ops (execution_format); the flat
        SELECT generator emits a bare ``SELECT * FROM source`` and would
        silently write every source column."""
        target = "wb_ss_projection"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds[["city", "amount"]].to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city",
            )
            cols = {name for name, _ in _columns(target)}
            assert cols == {"city", "amount"}
            assert "status" not in cols
            assert _count(target) == 7
            result = _select_all(target, order_by="amount DESC")
            assert float(result.iloc[0]["amount"]) == 400.0
        finally:
            _drop(target)

    def test_limit_before_filter_preserves_pipeline_order(self, source_table):
        """``head()`` before ``filter()`` must run as LIMIT-then-WHERE (pandas
        order), not the flat WHERE-then-LIMIT. The two highest amounts (400,
        300) are both 'completed', so a subsequent status=='pending' filter
        yields zero rows. The buggy flat SQL (WHERE status='pending' ORDER BY
        amount DESC LIMIT 2) would instead persist the two 'pending' rows."""
        target = "wb_ss_limit_order"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds.sort("amount", ascending=False)
               .head(2)
               .filter(ds["status"] == "pending")).to_clickhouse(
                f"{DATABASE}.{target}", engine="MergeTree()", order_by="city",
            )
            assert _count(target) == 0
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
# Assignments that read or overwrite a column assigned just before
# ============================================================================


def _dependent_assignment(frame):
    frame["c"] = frame["amount"] * 2
    frame["d"] = frame["c"] + 1
    return frame


def _reassign_twice(frame):
    frame["c"] = frame["amount"] * 2
    frame["c"] = frame["c"] + 1
    return frame


LAYERED_CHAINS = [
    pytest.param(_dependent_assignment, id="d_reads_c"),
    pytest.param(_reassign_twice, id="c_reassigned"),
]


class TestLayeredAssignmentPipeline:
    """An assignment that reads or overwrites a column assigned in the same
    query gets its own subquery layer, so the whole chain is one SQL query.
    Writeback must emit that full layered query: a single flat SELECT would
    bind the later expressions to the source columns."""

    @pytest.mark.parametrize("if_exists", ["fail", "replace", "append"])
    @pytest.mark.parametrize("chain", LAYERED_CHAINS)
    def test_to_clickhouse_matches_pandas(self, source_table, chain, if_exists):
        target = "wb_layered"
        _drop(target)
        try:
            expected = chain(_select_all(source_table, order_by="city, amount"))
            ds = chain(_make_ds(source_table))
            assert ds._is_fully_sql_pushable()
            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists=if_exists)
            result = _select_all(target, order_by="city, amount")
            assert_frame_equal(result, expected)
        finally:
            _drop(target)

    @pytest.mark.parametrize("chain", LAYERED_CHAINS)
    def test_save_table_matches_pandas(self, source_table, chain):
        target = "wb_layered_save"
        _drop(target)
        try:
            expected = chain(_select_all(source_table, order_by="city, amount"))
            chain(_make_ds(source_table)).save(f"{DATABASE}.{target}")
            result = _select_all(target, order_by="city, amount")
            assert_frame_equal(result, expected)
        finally:
            _drop(target)

    @pytest.mark.parametrize("chain", LAYERED_CHAINS)
    def test_create_view_matches_pandas(self, source_table, chain):
        view = "wb_layered_view"
        _drop_view(view)
        try:
            expected = chain(_select_all(source_table, order_by="city, amount"))
            chain(_make_ds(source_table)).create_view(f"{DATABASE}.{view}")
            result = _select_all(view, order_by="city, amount")
            assert_frame_equal(result, expected)
        finally:
            _drop_view(view)

    @pytest.mark.parametrize("chain", LAYERED_CHAINS)
    def test_create_materialized_view_matches_pandas(self, source_table, chain):
        mv = "wb_layered_mv"
        target = "wb_layered_mv_target"
        _drop_view(mv)
        _drop(target)
        try:
            chain(_make_ds(source_table)).create_materialized_view(
                f"{DATABASE}.{mv}", to=f"{DATABASE}.{target}",
            )
            _ddl(
                f'INSERT INTO "{DATABASE}"."{source_table}" VALUES '
                f"('Hangzhou', 777.0, 'completed')"
            )
            inserted = _q(
                f"SELECT * FROM {_remote_func(SOURCE, DATABASE, source_table)} "
                f"WHERE city = 'Hangzhou'"
            )
            assert_frame_equal(_select_all(target), chain(inserted))
        finally:
            _drop_view(mv)
            _drop(target)

    @pytest.mark.parametrize("chain", LAYERED_CHAINS)
    def test_replace_existing_target_matches_pandas(self, source_table, chain):
        """The query runs server-side, so replacing an existing target is
        allowed and swaps in the full result."""
        target = "wb_layered_existing"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "pending").to_clickhouse(
                f"{DATABASE}.{target}"
            )
            expected = chain(_select_all(source_table, order_by="city, amount"))
            chain(_make_ds(source_table)).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="replace",
            )
            result = _select_all(target, order_by="city, amount")
            assert_frame_equal(result, expected)
        finally:
            _drop(target)

    def test_independent_assignments_stay_server_side(self, source_table):
        target = "wb_independent_assignments"
        _drop(target)
        try:
            expected = _select_all(source_table, order_by="city, amount")
            expected["c"] = expected["amount"] * 2
            expected["d"] = expected["amount"] + 1
            ds = _make_ds(source_table)
            ds["c"] = ds["amount"] * 2
            ds["d"] = ds["amount"] + 1
            assert ds._is_fully_sql_pushable()
            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists="replace")
            result = _select_all(target, order_by="city, amount")
            assert_frame_equal(result, expected)
        finally:
            _drop(target)

    def test_column_selection_after_dependent_assignment(self, source_table):
        target = "wb_layered_projection"
        _drop(target)
        try:
            expected = _dependent_assignment(
                _select_all(source_table, order_by="city, amount")
            )[["city", "amount", "d"]]
            ds = _dependent_assignment(_make_ds(source_table))
            ds[["city", "amount", "d"]].to_clickhouse(f"{DATABASE}.{target}")
            result = _select_all(target, order_by="city, amount")
            assert_frame_equal(result, expected)
        finally:
            _drop(target)

    def test_schema_evolution_append_adds_dependent_columns(self, source_table):
        """Appending with schema evolution must add both computed columns.
        The append runs server-side, so the new columns take the pipeline's
        types and the pre-existing rows read back as their zero default."""
        target = "wb_layered_evolution"
        _drop(target)
        try:
            _make_ds(source_table).to_clickhouse(f"{DATABASE}.{target}")
            _dependent_assignment(_make_ds(source_table)).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="append",
                enable_schema_evolution=True,
            )
            source = _select_all(source_table, order_by="city, amount")
            existing = source.assign(c=0.0, d=0.0)
            appended = _dependent_assignment(source.copy())
            expected = pd.concat([existing, appended])
            _assert_written(target, expected, ["city", "amount", "c"])
        finally:
            _drop(target)


# ============================================================================
# Pipelines the executor runs in several stages
# ============================================================================


def _sql_filter_then_dependent_assignment(ds):
    ds = ds.sql("amount > 150")
    return _dependent_assignment(ds)


class TestMultiStagePipeline:
    """``ds.sql(...)`` runs on the first stage's result, so the assignments
    after it run as a later stage. No single SELECT reproduces such a plan,
    so writeback must not emit one that covers only the first stage."""

    def test_to_clickhouse_matches_pandas(self, source_table):
        target = "wb_multi_stage"
        _drop(target)
        try:
            source = _select_all(source_table, order_by="city, amount")
            expected = _dependent_assignment(source[source["amount"] > 150].copy())
            ds = _sql_filter_then_dependent_assignment(_make_ds(source_table))
            assert not ds._is_fully_sql_pushable()
            ds.to_clickhouse(f"{DATABASE}.{target}")
            _assert_written(target, expected, ["city", "amount"])
        finally:
            _drop(target)

    def test_create_view_rejected(self, source_table):
        view = "wb_multi_stage_view"
        _drop_view(view)
        try:
            ds = _sql_filter_then_dependent_assignment(_make_ds(source_table))
            with pytest.raises(QueryError, match="single SQL query"):
                ds.create_view(f"{DATABASE}.{view}")
            assert not ds._check_remote_table_exists(DATABASE, view)
        finally:
            _drop_view(view)

    def test_create_materialized_view_rejected(self, source_table):
        mv = "wb_multi_stage_mv"
        target = "wb_multi_stage_mv_target"
        _drop_view(mv)
        _drop(target)
        try:
            ds = _sql_filter_then_dependent_assignment(_make_ds(source_table))
            with pytest.raises(QueryError, match="single SQL query"):
                ds.create_materialized_view(
                    f"{DATABASE}.{mv}", to=f"{DATABASE}.{target}",
                )
            assert not ds._check_remote_table_exists(DATABASE, mv)
            assert not ds._check_remote_table_exists(DATABASE, target)
        finally:
            _drop_view(mv)
            _drop(target)

    def test_replace_existing_target_refused_and_left_unchanged(self, source_table):
        """The upload path cannot replace atomically, so an existing target
        must be refused and keep its rows rather than get a partial write."""
        target = "wb_multi_stage_existing"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.filter(ds["status"] == "pending").to_clickhouse(
                f"{DATABASE}.{target}"
            )
            before = _select_all(target, order_by="city, amount")
            with pytest.raises(QueryError, match="Cannot atomically replace"):
                _sql_filter_then_dependent_assignment(
                    _make_ds(source_table)
                ).to_clickhouse(f"{DATABASE}.{target}", if_exists="replace")
            assert_frame_equal(_select_all(target, order_by="city, amount"), before)
            assert {name for name, _ in _columns(target)} == {
                "city", "amount", "status",
            }
        finally:
            _drop(target)


# ============================================================================
# LazySQLQuery pipelines (ds.sql(...), SQL-style groupby().agg())
# ============================================================================


class TestLazySQLQueryWriteback:
    """``ds.sql(...)`` carries its query only in the lazy op, so the flat
    SELECT does not render it; such pipelines must be materialized rather
    than written as the bare source. A SQL-style ``groupby().agg()`` is
    rendered by the flat SELECT and stays server-side."""

    def test_sql_filter_matches_pandas(self, source_table):
        target = "wb_sql_filter"
        _drop(target)
        try:
            source = _select_all(source_table, order_by="city, amount")
            expected = source[source["amount"] > 150]
            _make_ds(source_table).sql("amount > 150").to_clickhouse(
                f"{DATABASE}.{target}"
            )
            _assert_written(target, expected, ["city", "amount"])
        finally:
            _drop(target)

    def test_sql_filter_then_assignment_matches_pandas(self, source_table):
        target = "wb_sql_filter_assign"
        _drop(target)
        try:
            source = _select_all(source_table, order_by="city, amount")
            expected = source[source["amount"] > 150].copy()
            expected["c"] = expected["amount"] * 2
            ds = _make_ds(source_table).sql("amount > 150")
            ds["c"] = ds["amount"] * 2
            ds.to_clickhouse(f"{DATABASE}.{target}", if_exists="replace")
            _assert_written(target, expected, ["city", "amount"])
        finally:
            _drop(target)

    def test_sql_filter_view_rejected(self, source_table):
        view = "wb_sql_filter_view"
        _drop_view(view)
        try:
            ds = _make_ds(source_table).sql("amount > 150")
            with pytest.raises(QueryError, match="Cannot create VIEW"):
                ds.create_view(f"{DATABASE}.{view}")
            assert not ds._check_remote_table_exists(DATABASE, view)
        finally:
            _drop_view(view)

    def test_sql_filter_materialized_view_rejected(self, source_table):
        mv = "wb_sql_filter_mv"
        target = "wb_sql_filter_mv_target"
        _drop_view(mv)
        _drop(target)
        try:
            ds = _make_ds(source_table).sql("amount > 150")
            with pytest.raises(QueryError, match="Cannot create MATERIALIZED VIEW"):
                ds.create_materialized_view(
                    f"{DATABASE}.{mv}", to=f"{DATABASE}.{target}",
                )
            assert not ds._check_remote_table_exists(DATABASE, target)
        finally:
            _drop_view(mv)
            _drop(target)

    def test_filter_then_sql_agg_stays_server_side(self, source_table):
        target = "wb_sql_agg_server_side"
        _drop(target)
        try:
            source = _select_all(source_table, order_by="city, amount")
            completed = source[source["status"] == "completed"]
            expected = completed.groupby("city", as_index=False).agg(
                total=("amount", "sum")
            )
            ds = _make_ds(source_table)
            summary = (ds.filter(ds["status"] == "completed")
                         .groupby("city")
                         .agg(total=col("amount").sum()))
            assert summary._is_fully_sql_pushable()
            summary.to_clickhouse(f"{DATABASE}.{target}")
            _assert_written(target, expected, ["city"])
        finally:
            _drop(target)

    def test_filter_applied_to_sql_agg_result_matches_pandas(self, source_table):
        """A filter on the aggregated result is not part of the flat SELECT
        (which would turn it into a pre-aggregation WHERE)."""
        target = "wb_sql_agg_then_filter"
        _drop(target)
        try:
            source = _select_all(source_table, order_by="city, amount")
            totals = source.groupby("city", as_index=False).agg(
                total=("amount", "sum")
            )
            expected = totals[totals["total"] > 400]
            summary = _make_ds(source_table).groupby("city").agg(
                total=col("amount").sum()
            )
            summary[summary["total"] > 400].to_clickhouse(f"{DATABASE}.{target}")
            _assert_written(target, expected, ["city"])
        finally:
            _drop(target)


# ============================================================================
# to_clickhouse — pure in-memory DataFrame source (no ClickHouse connection)
# ============================================================================


def _local_df():
    """A small in-memory DataFrame standing in for 'data the user built in
    Python', with no ClickHouse connection behind it."""
    return pd.DataFrame({
        "city": ["Beijing", "Shanghai", "Guangzhou"],
        "amount": [100.0, 300.0, 400.0],
        "status": ["completed", "pending", "completed"],
    })


class TestToClickHouseLocalDataFrame:
    """A DataStore wrapping a pure in-memory DataFrame (source_type ==
    'dataframe') persists to ClickHouse when an explicit target server is
    given. The data is materialized locally and uploaded via Python(df) — the
    same DataFrame-upload path a Pandas-only pipeline takes, just with no
    source server in play."""

    def test_requires_explicit_host(self):
        """No target host → there is no server to write to. Must raise a clear
        DataStoreError (not silently no-op, not the generic source guard).
        Both an omitted host and a blank host string are rejected the same way."""
        ds = DataStore(_local_df())
        with pytest.raises(DataStoreError, match="in-memory DataFrame"):
            ds.to_clickhouse(f"{DATABASE}.wb_localdf_nohost")
        with pytest.raises(DataStoreError, match="in-memory DataFrame"):
            ds.to_clickhouse(f"{DATABASE}.wb_localdf_nohost",
                             host="", user=USER, password=PASSWORD)
        # the failed attempts must not mutate the store's identity
        assert ds.source_type == "dataframe"

    def test_create_view_local_dataframe_with_host_still_rejected(self):
        """Scope guard: the relaxation is for to_clickhouse only. create_view
        on a pure df is rejected even WITH an explicit host — a local df has no
        server-side SQL source for a view definition."""
        ds = DataStore(_local_df())
        with pytest.raises(QueryError, match="requires a ClickHouse connection"):
            ds.create_view(f"{DATABASE}.wb_localdf_view",
                           host=HOST, user=USER, password=PASSWORD)
        assert ds.source_type == "dataframe"

    def test_create_materialized_view_local_dataframe_with_host_still_rejected(self):
        """Scope guard: create_materialized_view on a pure df is rejected even
        WITH an explicit host."""
        ds = DataStore(_local_df())
        with pytest.raises(QueryError, match="requires a ClickHouse connection"):
            ds.create_materialized_view(
                name=f"{DATABASE}.wb_localdf_mv",
                to=f"{DATABASE}.wb_localdf_mv_tgt",
                host=HOST, user=USER, password=PASSWORD,
            )
        assert ds.source_type == "dataframe"

    def test_save_view_and_mv_local_dataframe_rejected(self):
        """Scope guard: save() has no host param (it writes to the source
        server), so a pure df has nowhere to go — save(type='view'/'
        materialized_view') is rejected. Only to_clickhouse (data upload) is
        relaxed for local DataFrames."""
        ds = DataStore(_local_df())
        with pytest.raises(DataStoreError):
            ds.save(f"{DATABASE}.wb_localdf_save_view", type="view")
        with pytest.raises(DataStoreError):
            ds.save(f"{DATABASE}.wb_localdf_save_mv",
                    type="materialized_view", to=f"{DATABASE}.wb_localdf_save_tgt")
        assert ds.source_type == "dataframe"

    def test_fail_creates_and_inserts_exact_values(self):
        target = "wb_localdf_fail"
        _drop(target)
        try:
            DataStore(_local_df()).to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD, order_by="city",
            )
            assert _count(target) == 3
            got = _select_all(target, order_by="city")
            assert list(got["city"]) == ["Beijing", "Guangzhou", "Shanghai"]
            assert list(got["amount"]) == [100.0, 400.0, 300.0]
            assert list(got["status"]) == ["completed", "completed", "pending"]
        finally:
            _drop(target)

    @pytest.mark.parametrize("index", [False, True])
    def test_multiindex_columns_are_flattened(self, index):
        """agg(['sum', 'max']) columns are written as amount_sum /
        amount_max, the names the aggregation has in SQL."""
        target = f"wb_localdf_multiindex_{int(index)}"
        _drop(target)
        try:
            frame = _local_df()[["status", "amount"]]
            DataStore(frame).groupby("status").agg(["sum", "max"]).to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD, index=index,
            )
            expected = frame.groupby("status").agg(["sum", "max"])
            expected.columns = ["amount_sum", "amount_max"]
            expected = expected.reset_index(drop=not index)
            sort_key = "status" if index else "amount_sum"
            _assert_written(target, expected, [sort_key])
        finally:
            _drop(target)

    def test_multiindex_flattened_name_collision_raises(self):
        """A key column amount_sum next to ('amount', 'sum') would write two
        columns with one name."""
        target = "wb_localdf_multiindex_clash"
        _drop(target)
        try:
            frame = _local_df()[["status", "amount"]].assign(amount_sum=1)
            with pytest.raises(DataStoreError, match="collide"):
                DataStore(frame).groupby(["status", "amount_sum"]).agg(
                    ["sum", "max"]
                ).to_clickhouse(
                    f"{DATABASE}.{target}",
                    host=HOST, user=USER, password=PASSWORD, index=True,
                )
            assert not _q(
                f"SELECT name FROM remote('{HOST}', 'system', 'tables', "
                f"'{USER}', '{PASSWORD}') "
                f"WHERE database = '{DATABASE}' AND name = '{target}'"
            ).shape[0]
        finally:
            _drop(target)

    def test_fail_when_target_exists_raises(self):
        target = "wb_localdf_exists"
        _drop(target)
        try:
            DataStore(_local_df()).to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD, order_by="city",
            )
            with pytest.raises(ExecutionError, match="already exists"):
                DataStore(_local_df()).to_clickhouse(
                    f"{DATABASE}.{target}",
                    host=HOST, user=USER, password=PASSWORD, order_by="city",
                )
        finally:
            _drop(target)

    def test_replace_new_creates_then_refuses_existing(self):
        """replace + target absent → create+insert; replace + target present →
        the non-atomic upload path refuses rather than risk a data-loss window,
        and the temporary source_type override is restored after the raise."""
        target = "wb_localdf_replace"
        _drop(target)
        try:
            DataStore(_local_df()).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="replace",
                host=HOST, user=USER, password=PASSWORD, order_by="city",
            )
            assert _count(target) == 3
            got = _select_all(target, order_by="city")
            assert list(got["city"]) == ["Beijing", "Guangzhou", "Shanghai"]
            assert list(got["amount"]) == [100.0, 400.0, 300.0]

            ds = DataStore(_local_df())
            with pytest.raises(QueryError, match="Cannot atomically replace"):
                ds.to_clickhouse(
                    f"{DATABASE}.{target}", if_exists="replace",
                    host=HOST, user=USER, password=PASSWORD, order_by="city",
                )
            assert ds.source_type == "dataframe"   # restored after exception
            assert _count(target) == 3             # original data untouched
        finally:
            _drop(target)

    def test_append_accumulates(self):
        target = "wb_localdf_append"
        _drop(target)
        try:
            DataStore(_local_df()).to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD, order_by="city",
            )
            DataStore(_local_df()).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="append",
                host=HOST, user=USER, password=PASSWORD,
            )
            assert _count(target) == 6
            got = _select_all(target, order_by="city, amount")
            # each of the 3 source rows now appears twice
            assert list(got["city"]) == [
                "Beijing", "Beijing", "Guangzhou", "Guangzhou",
                "Shanghai", "Shanghai",
            ]
            assert list(got["amount"]) == [100.0, 100.0, 400.0, 400.0, 300.0, 300.0]
        finally:
            _drop(target)

    def test_engine_order_by_list_and_partition_by(self):
        target = "wb_localdf_engine"
        _drop(target)
        try:
            DataStore(_local_df()).to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD,
                engine="MergeTree()", order_by=["city", "status"],
                partition_by="status",
            )
            assert _count(target) == 3
            engine_full = str(_q(
                f"SELECT engine_full FROM remote('{HOST}','system','tables',"
                f"'{USER}','{PASSWORD}') "
                f"WHERE database='{DATABASE}' AND name='{target}'"
            ).iloc[0, 0]).upper()
            assert "PARTITION BY" in engine_full
            assert "ORDER BY" in engine_full
            got = _select_all(target, order_by="city")
            assert list(got["city"]) == ["Beijing", "Guangzhou", "Shanghai"]
            assert list(got["amount"]) == [100.0, 400.0, 300.0]
            assert list(got["status"]) == ["completed", "completed", "pending"]
        finally:
            _drop(target)

    def test_index_true_with_index_label(self):
        target = "wb_localdf_index"
        _drop(target)
        try:
            df = _local_df().set_index(pd.Index([10, 20, 30], name="rid"))
            DataStore(df).to_clickhouse(
                f"{DATABASE}.{target}", index=True, index_label="row_id",
                host=HOST, user=USER, password=PASSWORD, order_by="row_id",
            )
            cols = dict(_columns(target))
            assert "row_id" in cols and "rid" not in cols
            got = _select_all(target, order_by="row_id")
            assert list(got["row_id"]) == [10, 20, 30]
        finally:
            _drop(target)

    def test_pandas_transform_on_local_df_flows_through(self):
        """A computed column added via a Pandas transform on a local df still
        lands on the server (it's all the same upload path)."""
        target = "wb_localdf_computed"
        _drop(target)
        try:
            ds = DataStore(_local_df()).transform(
                lambda df: df.assign(amount_x2=df["amount"] * 2)
            )
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD, order_by="city",
            )
            got = _select_all(target, order_by="city")
            assert "amount_x2" in got.columns
            assert all(got["amount_x2"] == got["amount"] * 2)
        finally:
            _drop(target)

    def test_source_type_restored_and_store_still_usable(self):
        """The temporary source_type override must not leak: the store is still
        a 'dataframe' store afterwards and remains usable as a local df."""
        target = "wb_localdf_restore"
        _drop(target)
        try:
            ds = DataStore(_local_df())
            assert ds.source_type == "dataframe"
            ds.to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD, order_by="city",
            )
            assert ds.source_type == "dataframe"
            assert len(ds) == 3
        finally:
            _drop(target)

    def test_from_df_and_from_dataframe_factories(self):
        """A DataFrame wrapped via the from_df / from_dataframe factories (which
        keep source_type='chdb' and stash the frame in _source_df, unlike the
        DataStore(df) constructor which uses source_type='dataframe') must write
        just the same. Both representations are 'a local frame with no source
        server', so to_clickhouse routes both through the upload path."""
        for factory in (DataStore.from_df, DataStore.from_dataframe):
            target = f"wb_localdf_{factory.__name__}"
            _drop(target)
            try:
                ds = factory(_local_df())
                # sanity: this is the non-'dataframe' representation
                assert ds.source_type != "dataframe"
                ds.to_clickhouse(
                    f"{DATABASE}.{target}",
                    host=HOST, user=USER, password=PASSWORD, order_by="city",
                )
                got = _select_all(target, order_by="city")
                assert len(got) == 3
                assert list(got["city"]) == ["Beijing", "Guangzhou", "Shanghai"]
                assert list(got["amount"]) == [100.0, 400.0, 300.0]
                assert list(got["status"]) == ["completed", "completed", "pending"]
                # the temporary source_type override must be restored
                assert ds.source_type != "dataframe"
            finally:
                _drop(target)

    def test_from_df_factory_requires_explicit_host(self):
        """The no-host guard applies to the from_df representation too."""
        ds = DataStore.from_df(_local_df())
        with pytest.raises(DataStoreError, match="in-memory DataFrame"):
            ds.to_clickhouse(f"{DATABASE}.wb_localdf_fromdf_nohost")

    @pytest.mark.parametrize("ds_chain, pd_chain", [
        pytest.param(
            lambda d: d.assign(k=d["amount"] * 2).sort_values("amount"),
            lambda d: d.assign(k=d["amount"] * 2).sort_values("amount"),
            id="assign_sort",
        ),
        pytest.param(
            lambda d: d.assign(k=col("amount") * 2),
            lambda d: d.assign(k=d["amount"] * 2),
            id="assign_col",
        ),
        pytest.param(
            lambda d: d[d["amount"] > 150],
            lambda d: d[d["amount"] > 150],
            id="filter",
        ),
    ])
    def test_from_df_sql_ops_are_uploaded(self, ds_chain, pd_chain):
        """SQL-compatible ops on a from_df source run against Python(df), which
        only this process can read, so they must be uploaded rather than sent
        to the server as a SELECT."""
        target = "wb_localdf_fromdf_sql_ops"
        _drop(target)
        try:
            df = pd.DataFrame({
                "city": ["Shanghai", "Beijing", "Guangzhou"],
                "amount": [300.0, 100.0, 400.0],
            })
            expected = pd_chain(df.copy())
            ds_chain(DataStore.from_df(df)).to_clickhouse(
                f"{DATABASE}.{target}",
                host=HOST, user=USER, password=PASSWORD,
            )
            _assert_written(target, expected, ["city"])
        finally:
            _drop(target)


@requires_target
class TestToClickHouseLocalDataFrameCrossServer:
    """Persist a pure local DataFrame onto a *different* server — the
    'push to ClickHouse Cloud' shape: a distinct host with distinct
    credentials, reached over the network."""

    def test_uploads_to_target_server_only(self):
        target = "wb_localdf_cross"
        _drop(target, db=TARGET.database, srv=TARGET)
        try:
            DataStore(_local_df()).to_clickhouse(
                f"{TARGET.database}.{target}",
                host=TARGET.host, user=TARGET.user, password=TARGET.password,
                order_by="city",
            )
            assert _count(target, db=TARGET.database, srv=TARGET) == 3
            got = _select_all(
                target, db=TARGET.database, order_by="city", srv=TARGET
            )
            assert list(got["city"]) == ["Beijing", "Guangzhou", "Shanghai"]
            assert list(got["amount"]) == [100.0, 400.0, 300.0]
            assert list(got["status"]) == ["completed", "completed", "pending"]

            # The table must NOT have been created on the source server.
            on_source = _q(
                f"SELECT count() FROM remote('{HOST}','system','tables',"
                f"'{USER}','{PASSWORD}') WHERE database='{TARGET.database}' "
                f"AND name='{target}'"
            ).iloc[0, 0]
            assert int(on_source) == 0
        finally:
            _drop(target, db=TARGET.database, srv=TARGET)


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
# groupby() aggregations — written column names match the pandas result
# ============================================================================


def _source_table_pdf():
    """pandas mirror of the ``source_table`` fixture rows."""
    return pd.DataFrame({
        "city": ["Beijing", "Beijing", "Beijing", "Shanghai", "Shanghai",
                 "Guangzhou", "Guangzhou"],
        "amount": [100.0, 200.0, 50.0, 300.0, 150.0, 400.0, 100.0],
        "status": ["completed", "completed", "pending", "completed",
                   "pending", "completed", "completed"],
    })


def _assert_written_equals(written: pd.DataFrame, expected: pd.DataFrame):
    # The pandas side keeps group keys in the index; a SQL table has no
    # index, so the expected frame carries them as regular columns.
    assert_frame_equal(written, expected)


class TestGroupByAggWritebackColumnNames:
    """Aggregations whose output alias shadows a source column are built
    with temporary SQL aliases; the server-side SELECT must still expose the
    user-visible names (regression for ``__agg_amount__`` columns)."""

    def test_agg_dict_replace(self, source_table):
        target = "wb_agg_dict_replace"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.groupby("city").agg({"amount": "sum"}).to_clickhouse(
                f"{DATABASE}.{target}", if_exists="replace"
            )
            expected = (_source_table_pdf().groupby("city")
                        .agg({"amount": "sum"}).reset_index())
            _assert_written_equals(_select_all(target, order_by="city"), expected)
        finally:
            _drop(target)

    def test_filter_then_multi_column_agg_dict(self, source_table):
        target = "wb_agg_dict_multi"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds[ds["amount"] > 60]
               .groupby("city")
               .agg({"amount": "sum", "status": "count"})
            ).to_clickhouse(f"{DATABASE}.{target}",
                            engine="MergeTree()", order_by="city")
            pdf = _source_table_pdf()
            expected = (pdf[pdf["amount"] > 60]
                        .groupby("city")
                        .agg({"amount": "sum", "status": "count"})
                        .reset_index())
            _assert_written_equals(_select_all(target, order_by="city"), expected)
            assert [name for name, _ in _columns(target)] == ["city", "amount", "status"]
        finally:
            _drop(target)

    def test_as_index_false_append(self, source_table):
        target = "wb_agg_as_index_false"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            pipeline = ds.groupby("city", as_index=False).agg({"amount": "max"})
            pipeline.to_clickhouse(f"{DATABASE}.{target}", if_exists="append",
                                   engine="MergeTree()", order_by="city")
            pipeline.to_clickhouse(f"{DATABASE}.{target}", if_exists="append")
            once = _source_table_pdf().groupby("city", as_index=False).agg(
                {"amount": "max"}
            )
            expected = pd.concat([once, once]).sort_values(
                "city", kind="stable"
            ).reset_index(drop=True)
            _assert_written_equals(_select_all(target, order_by="city"), expected)
        finally:
            _drop(target)

    def test_agg_then_filter_sort_head(self, source_table):
        target = "wb_agg_having_sort_head"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            totals = ds.groupby("city").agg({"amount": "sum"})
            (totals[totals["amount"] > 340]
               .sort_values("amount", ascending=False)
               .head(2)
            ).to_clickhouse(f"{DATABASE}.{target}",
                            engine="MergeTree()", order_by="city")
            pd_totals = _source_table_pdf().groupby("city").agg({"amount": "sum"})
            expected = (pd_totals[pd_totals["amount"] > 340]
                        .sort_values("amount", ascending=False)
                        .head(2)
                        .reset_index())
            # A MergeTree table has no row order; read it back in the order
            # the pandas pipeline sorted by.
            _assert_written_equals(
                _select_all(target, order_by="amount DESC"), expected
            )
        finally:
            _drop(target)

    def test_named_agg_reusing_source_column_name(self, source_table):
        """Guard, not a regression test: this shape never used a temp alias
        and already wrote the right columns before the fix."""
        target = "wb_agg_named"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.groupby("city").agg(
                amount=("amount", "sum"), n=("status", "count")
            ).to_clickhouse(f"{DATABASE}.{target}",
                            engine="MergeTree()", order_by="city")
            expected = (_source_table_pdf().groupby("city")
                        .agg(amount=("amount", "sum"), n=("status", "count"))
                        .reset_index())
            _assert_written_equals(_select_all(target, order_by="city"), expected)
        finally:
            _drop(target)

    def test_filter_then_named_agg_reusing_source_column_name(self, source_table):
        target = "wb_agg_filter_named"
        _drop(target)
        try:
            ds = _make_ds(source_table)
            (ds[ds["amount"] > 60]
               .groupby("city")
               .agg(amount=("amount", "sum"))
            ).to_clickhouse(f"{DATABASE}.{target}",
                            engine="MergeTree()", order_by="city")
            pdf = _source_table_pdf()
            expected = (pdf[pdf["amount"] > 60]
                        .groupby("city")
                        .agg(amount=("amount", "sum"))
                        .reset_index())
            _assert_written_equals(_select_all(target, order_by="city"), expected)
        finally:
            _drop(target)

    def test_create_view(self, source_table):
        view = "wb_agg_view"
        _drop_view(view)
        try:
            ds = _make_ds(source_table)
            ds.groupby("city").agg({"amount": "sum"}).create_view(
                f"{DATABASE}.{view}"
            )
            expected = (_source_table_pdf().groupby("city")
                        .agg({"amount": "sum"}).reset_index())
            _assert_written_equals(_select_all(view, order_by="city"), expected)
        finally:
            _drop_view(view)

    def test_materialized_view_populate(self, source_table):
        mv, target = "wb_agg_mv", "wb_agg_mv_tgt"
        _drop_view(mv)
        _drop(target)
        try:
            ds = _make_ds(source_table)
            ds.groupby("city").agg({"amount": "sum"}).create_materialized_view(
                name=f"{DATABASE}.{mv}",
                to=f"{DATABASE}.{target}",
                engine="MergeTree()", order_by="city",
                populate=True,
            )
            expected = (_source_table_pdf().groupby("city")
                        .agg({"amount": "sum"}).reset_index())
            _assert_written_equals(_select_all(target, order_by="city"), expected)
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
