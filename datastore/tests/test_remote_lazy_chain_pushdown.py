"""
Test CH-8: Remote ClickHouse Lazy Chain Complete Pushdown Verification.

Verifies that multi-step lazy operation chains on remote ClickHouse
are fully pushed down to a single SQL query, with no intermediate
materialization or pandas fallback.

Each test verifies:
1. Single SQL segment via QueryPlanner.plan_segments()
2. Single SQL execution via debug log capture
3. Correct SQL clauses (WHERE, ORDER BY, LIMIT, GROUP BY)
4. Results match equivalent pandas operations (Mirror Code Pattern)
"""

import os
import subprocess
import pytest
import logging
import io
import pandas as pd
import numpy as np

from datastore import DataStore
from datastore.query_planner import QueryPlanner
from tests.test_utils import assert_datastore_equals_pandas

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CH_DIR = os.path.join(SCRIPT_DIR, ".clickhouse")
CH_BINARY = os.path.join(CH_DIR, "clickhouse")

TEST_DB = "test_pushdown_ch8"
TEST_TABLE = "lazy_chain_data"

TEST_ROWS = [
    (1, 10, "x", 100),
    (2, 20, "x", 200),
    (3, 30, "y", 300),
    (4, 40, "y", 400),
    (5, 50, "x", 500),
    (6, 60, "z", 600),
    (7, 70, "z", 700),
    (8, 80, "y", 800),
    (9, 90, "x", 900),
    (10, 100, "z", 1000),
]


def run_ch_sql(port, sql):
    """Execute SQL on the test ClickHouse server via clickhouse-client."""
    result = subprocess.run(
        [CH_BINARY, "client", "--port", str(port), "--query", sql],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ClickHouse SQL failed: {result.stderr}\nSQL: {sql}")
    return result.stdout


def capture_execution_log(ds):
    """Execute DataStore and capture debug log.

    Returns:
        (result_dataframe, log_text)
    """
    logger = logging.getLogger("datastore")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        df = ds._execute()
        log_text = stream.getvalue()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    return df, log_text


def extract_sqls(log_text):
    """Extract SQL statements from debug log."""
    sqls = []
    for line in log_text.split("\n"):
        if "Executing SQL:" in line:
            sql = line.split("Executing SQL:", 1)[1].strip()
            sqls.append(sql)
    return sqls


@pytest.fixture(scope="module")
def pushdown_table(clickhouse_server):
    """Create test database and table with data for pushdown tests."""
    host, port = clickhouse_server

    if not os.path.exists(CH_BINARY):
        pytest.skip(f"ClickHouse binary not found at {CH_BINARY}")

    run_ch_sql(port, f"CREATE DATABASE IF NOT EXISTS {TEST_DB}")
    run_ch_sql(
        port,
        f"CREATE TABLE IF NOT EXISTS {TEST_DB}.{TEST_TABLE} "
        f"(a Int32, b Int32, g String, v Int32) "
        f"ENGINE = MergeTree() ORDER BY a",
    )
    run_ch_sql(port, f"TRUNCATE TABLE {TEST_DB}.{TEST_TABLE}")

    values = ", ".join(
        f"({a}, {b}, '{g}', {v})" for a, b, g, v in TEST_ROWS
    )
    run_ch_sql(port, f"INSERT INTO {TEST_DB}.{TEST_TABLE} VALUES {values}")

    yield host, port

    try:
        run_ch_sql(port, f"DROP DATABASE IF EXISTS {TEST_DB}")
    except Exception:
        pass


@pytest.fixture
def ds_table(pushdown_table):
    """DataStore pointing at the remote test table."""
    host, port = pushdown_table
    ds = DataStore.from_clickhouse(
        host=f"{host}:{port}", user="default", password=""
    )
    return ds.table(TEST_DB, TEST_TABLE)


@pytest.fixture
def pd_df():
    """Equivalent pandas DataFrame for mirror comparison."""
    return pd.DataFrame(
        {
            "a": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int32"),
            "b": pd.array(
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype="int32"
            ),
            "g": ["x", "x", "y", "y", "x", "z", "z", "y", "x", "z"],
            "v": pd.array(
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                dtype="int32",
            ),
        }
    )


class TestRemoteLazyChainPushdown:
    """Verify lazy chains on remote ClickHouse push down to single SQL."""

    def _assert_single_sql_segment(self, ds):
        """Verify the lazy ops form a single SQL segment with no pandas fallback."""
        planner = QueryPlanner()
        plan = planner.plan_segments(ds._lazy_ops, has_sql_source=True)
        assert plan.sql_segment_count() == 1, (
            f"Expected 1 SQL segment, got {plan.sql_segment_count()}. "
            f"Plan:\n{plan.describe()}"
        )
        assert plan.pandas_segment_count() == 0, (
            f"Expected 0 Pandas segments, got {plan.pandas_segment_count()}. "
            f"Plan:\n{plan.describe()}"
        )

    def _assert_single_sql_execution(self, ds):
        """Execute DataStore and verify only 1 SQL was executed.

        Returns:
            (result_df, list_of_sql_strings)
        """
        df, log_text = capture_execution_log(ds)
        sqls = extract_sqls(log_text)
        assert len(sqls) == 1, (
            f"Expected 1 SQL execution, got {len(sqls)}. SQLs:\n"
            + "\n".join(sqls)
        )
        return df, sqls

    def test_filter_select_sort_head(self, ds_table, pd_df):
        """ds[cond].select('a','b').sort_values('a').head(5) -> single SQL.

        Expected SQL: SELECT a, b FROM remote(...) WHERE a > 3 ORDER BY a LIMIT 5
        """
        # DataStore lazy chain
        ds_result = (
            ds_table[ds_table["a"] > 3]
            .select("a", "b")
            .sort_values("a")
            .head(5)
        )

        # Verify single SQL segment (before execution)
        self._assert_single_sql_segment(ds_result)

        # Execute and verify single SQL
        ds_df, sqls = self._assert_single_sql_execution(ds_result)
        sql = sqls[0].upper()
        assert "WHERE" in sql, f"Missing WHERE: {sqls[0]}"
        assert "ORDER BY" in sql, f"Missing ORDER BY: {sqls[0]}"
        assert "LIMIT" in sql, f"Missing LIMIT: {sqls[0]}"

        # Mirror pandas
        pd_result = pd_df[pd_df["a"] > 3][["a", "b"]].sort_values("a").head(5)

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_groupby_agg_sort(self, ds_table, pd_df):
        """ds[cond].groupby('g').agg({'v':'sum'}).sort_values('v') -> single SQL.

        Expected SQL: SELECT g, SUM(v) AS v FROM remote(...)
                      WHERE a > 2 GROUP BY g ORDER BY v
        """
        # DataStore lazy chain
        ds_result = (
            ds_table[ds_table["a"] > 2]
            .groupby("g")
            .agg({"v": "sum"})
            .sort_values("v")
        )

        # Verify single SQL segment
        self._assert_single_sql_segment(ds_result)

        # Execute and verify single SQL
        ds_df, sqls = self._assert_single_sql_execution(ds_result)
        sql = sqls[0].upper()
        assert "WHERE" in sql, f"Missing WHERE: {sqls[0]}"
        assert "GROUP BY" in sql, f"Missing GROUP BY: {sqls[0]}"
        assert "ORDER BY" in sql, f"Missing ORDER BY: {sqls[0]}"

        # Mirror pandas
        pd_result = (
            pd_df[pd_df["a"] > 2]
            .groupby("g")
            .agg({"v": "sum"})
            .sort_values("v")
        )

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_groupby_agg(self, ds_table, pd_df):
        """ds.assign(new_col=ds['a']*2)[cond].groupby('g').agg({...}) -> single SQL.

        Expected SQL: SELECT g, AVG(v), AVG(a*2) AS new_col
                      FROM remote(...) WHERE a > 3 GROUP BY g
        """
        # DataStore lazy chain with computed column
        ds_with_col = ds_table.assign(new_col=ds_table["a"] * 2)
        ds_result = (
            ds_with_col[ds_with_col["a"] > 3]
            .groupby("g")
            .agg({"v": "mean", "new_col": "mean"})
        )

        # Verify single SQL segment
        self._assert_single_sql_segment(ds_result)

        # Execute and verify single SQL
        ds_df, sqls = self._assert_single_sql_execution(ds_result)
        sql = sqls[0].upper()
        assert "WHERE" in sql, f"Missing WHERE: {sqls[0]}"
        assert "GROUP BY" in sql, f"Missing GROUP BY: {sqls[0]}"

        # Mirror pandas
        pd_with_col = pd_df.assign(new_col=pd_df["a"] * 2)
        pd_result = (
            pd_with_col[pd_with_col["a"] > 3]
            .groupby("g")
            .agg({"v": "mean", "new_col": "mean"})
        )

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multi_filter_groupby_multi_agg(self, ds_table, pd_df):
        """ds[cond1][cond2].groupby('g').agg({...}) -> single SQL.

        Tests: multi WHERE + multi-column agg + alias conflict resolution.
        Expected SQL: SELECT g, SUM(v) AS v, AVG(a) AS a
                      FROM remote(...) WHERE a > 2 AND b < 80 GROUP BY g
        """
        # DataStore lazy chain with two separate filter conditions
        ds_result = (
            ds_table[ds_table["a"] > 2][ds_table["b"] < 80]
            .groupby("g")
            .agg({"v": "sum", "a": "mean"})
        )

        # Verify single SQL segment
        self._assert_single_sql_segment(ds_result)

        # Execute and verify single SQL
        ds_df, sqls = self._assert_single_sql_execution(ds_result)
        sql = sqls[0].upper()
        assert "WHERE" in sql, f"Missing WHERE: {sqls[0]}"
        assert "GROUP BY" in sql, f"Missing GROUP BY: {sqls[0]}"

        # Mirror pandas (combined condition for correct semantics)
        pd_result = (
            pd_df[(pd_df["a"] > 2) & (pd_df["b"] < 80)]
            .groupby("g")
            .agg({"v": "sum", "a": "mean"})
        )

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
