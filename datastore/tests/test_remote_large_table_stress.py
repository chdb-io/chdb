"""
CH-10: Remote ClickHouse large table stress test.

Verifies that common DataStore operations on a 1M-row remote table
do not OOM or exceed timeout limits. Uses numbers() to generate test
data directly in ClickHouse.

To run these tests:
    pytest datastore/tests/test_remote_large_table_stress.py -v

To use an external ClickHouse server:
    export TEST_CLICKHOUSE_HOST=localhost:9000
    pytest datastore/tests/test_remote_large_table_stress.py -v
"""

import os
import subprocess
import time
import io
import pytest
import pandas as pd

from datastore import DataStore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CH_DIR = os.path.join(SCRIPT_DIR, ".clickhouse")
CH_BINARY = os.path.join(CH_DIR, "clickhouse")

TEST_DB = "test_stress_ch10"
LARGE_TABLE = "large_table"
MEDIUM_TABLE = "medium_table"
LARGE_ROW_COUNT = 1_000_000
MEDIUM_ROW_COUNT = 100_000


def run_ch_sql(port, sql):
    """Execute SQL on the test ClickHouse server via clickhouse-client."""
    result = subprocess.run(
        [CH_BINARY, "client", "--port", str(port), "--query", sql],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ClickHouse SQL failed: {result.stderr}\nSQL: {sql}")
    return result.stdout


def _create_table(port, table_name, row_count):
    """Create a test table with the given number of rows."""
    run_ch_sql(
        port,
        f"CREATE TABLE IF NOT EXISTS {TEST_DB}.{table_name} ("
        f"  id UInt64,"
        f"  val_int Int32,"
        f"  val_float Float64,"
        f"  category String,"
        f"  flag UInt8,"
        f"  score Float32,"
        f"  tag String,"
        f"  amount Int64,"
        f"  ratio Float64,"
        f"  label String"
        f") ENGINE = MergeTree() ORDER BY id",
    )

    count_str = run_ch_sql(
        port, f"SELECT count() FROM {TEST_DB}.{table_name}"
    ).strip()
    if int(count_str) < row_count:
        run_ch_sql(port, f"TRUNCATE TABLE {TEST_DB}.{table_name}")
        run_ch_sql(
            port,
            f"INSERT INTO {TEST_DB}.{table_name} "
            f"SELECT "
            f"  number AS id, "
            f"  toInt32(number % 1000) AS val_int, "
            f"  number * 0.1 AS val_float, "
            f"  arrayElement(['alpha','beta','gamma','delta'], (number % 4) + 1) AS category, "
            f"  number % 2 AS flag, "
            f"  toFloat32(number % 100) AS score, "
            f"  arrayElement(['X','Y','Z'], (number % 3) + 1) AS tag, "
            f"  toInt64(number * 10) AS amount, "
            f"  number / ({row_count} + 0.0) AS ratio, "
            f"  concat('row_', toString(number % 10000)) AS label "
            f"FROM numbers({row_count})",
        )


@pytest.fixture(scope="module")
def stress_tables(clickhouse_server):
    """Create test tables: 1M-row (large) and 100K-row (medium)."""
    host, port = clickhouse_server

    if not os.path.exists(CH_BINARY):
        pytest.skip(f"ClickHouse binary not found at {CH_BINARY}")

    run_ch_sql(port, f"CREATE DATABASE IF NOT EXISTS {TEST_DB}")
    _create_table(port, LARGE_TABLE, LARGE_ROW_COUNT)
    _create_table(port, MEDIUM_TABLE, MEDIUM_ROW_COUNT)

    yield host, port

    try:
        run_ch_sql(port, f"DROP DATABASE IF EXISTS {TEST_DB}")
    except Exception:
        pass


@pytest.fixture
def ds_large(stress_tables):
    """DataStore connected to the 1M-row remote table."""
    host, port = stress_tables
    ds = DataStore.from_clickhouse(
        host=f"{host}:{port}", user="default", password=""
    )
    return ds.table(TEST_DB, LARGE_TABLE)


@pytest.fixture
def ds_medium(stress_tables):
    """DataStore connected to the 100K-row remote table."""
    host, port = stress_tables
    ds = DataStore.from_clickhouse(
        host=f"{host}:{port}", user="default", password=""
    )
    return ds.table(TEST_DB, MEDIUM_TABLE)


class TestLargeTableFastOps:
    """Operations that MUST complete quickly on a 1M-row table.

    These operations use SQL pushdown and should NOT download all data.
    """

    def test_head_under_1s(self, ds_large):
        """ds.head(10) must return in < 1 second."""
        start = time.monotonic()
        result = ds_large.head(10)
        df = result._execute()
        elapsed = time.monotonic() - start

        assert len(df) == 10
        assert list(df.columns) == [
            "id", "val_int", "val_float", "category", "flag",
            "score", "tag", "amount", "ratio", "label",
        ]
        assert elapsed < 1.0, f"head(10) took {elapsed:.2f}s, expected < 1s"

    def test_columns_under_1s(self, ds_large):
        """ds.columns must return in < 1 second (pristine, no data load)."""
        start = time.monotonic()
        cols = ds_large.columns
        elapsed = time.monotonic() - start

        assert len(cols) == 10
        assert "id" in cols
        assert "category" in cols
        assert elapsed < 1.0, f"columns took {elapsed:.2f}s, expected < 1s"

    def test_dtypes_under_1s(self, ds_large):
        """ds.dtypes must return in < 1 second (pristine, LIMIT 0 probe)."""
        start = time.monotonic()
        dt = ds_large.dtypes
        elapsed = time.monotonic() - start

        assert len(dt) == 10
        assert elapsed < 1.0, f"dtypes took {elapsed:.2f}s, expected < 1s"

    def test_shape_under_5s(self, ds_large):
        """ds.shape must return in < 5 seconds (needs COUNT)."""
        start = time.monotonic()
        rows, cols = ds_large.shape
        elapsed = time.monotonic() - start

        assert rows == LARGE_ROW_COUNT
        assert cols == 10
        assert elapsed < 5.0, f"shape took {elapsed:.2f}s, expected < 5s"

    def test_count_rows_under_2s(self, ds_large):
        """ds.count_rows() must return in < 2 seconds."""
        start = time.monotonic()
        count = ds_large.count_rows()
        elapsed = time.monotonic() - start

        assert count == LARGE_ROW_COUNT
        assert elapsed < 2.0, f"count_rows() took {elapsed:.2f}s, expected < 2s"

    def test_filter_head_under_2s(self, ds_large):
        """ds[cond].head(10) must return in < 2 seconds (SQL pushdown)."""
        start = time.monotonic()
        filtered = ds_large[ds_large["val_int"] > 500].head(10)
        df = filtered._execute()
        elapsed = time.monotonic() - start

        assert len(df) == 10
        assert all(df["val_int"] > 500)
        assert elapsed < 2.0, f"filter+head took {elapsed:.2f}s, expected < 2s"

    def test_groupby_count_under_5s(self, ds_large):
        """ds.groupby('col').count() must return in < 5 seconds (SQL pushdown)."""
        start = time.monotonic()
        result = ds_large.groupby("category").count()
        df = result._execute()
        elapsed = time.monotonic() - start

        assert len(df) == 4  # alpha, beta, gamma, delta
        assert "id" in df.columns
        assert elapsed < 5.0, f"groupby+count took {elapsed:.2f}s, expected < 5s"


class TestLargeTableSQLOptimized:
    """Operations that SHOULD use SQL optimization (no full download)."""

    def test_tail_via_sql_offset(self, ds_large):
        """ds.tail(5) should use SQL OFFSET/LIMIT on pristine source."""
        assert ds_large._can_use_sql_tail()
        start = time.monotonic()
        result = ds_large.tail(5)
        df = result._execute()
        elapsed = time.monotonic() - start

        assert len(df) == 5
        assert elapsed < 5.0, f"tail(5) took {elapsed:.2f}s, expected < 5s with SQL opt"

    def test_info_no_counts(self, ds_large):
        """ds.info(show_counts=False) should be fast (metadata only)."""
        assert ds_large._can_use_sql_no_load()
        buf = io.StringIO()
        start = time.monotonic()
        ds_large.info(buf=buf, show_counts=False)
        elapsed = time.monotonic() - start

        output = buf.getvalue()
        assert "id" in output
        assert "category" in output
        assert elapsed < 5.0, f"info(show_counts=False) took {elapsed:.2f}s, expected < 5s"


class TestLargeTableFilterVariants:
    """Test various filter patterns on large data (all use SQL pushdown)."""

    def test_multi_condition_filter(self, ds_large):
        """Multi-condition filter with head() stays fast."""
        start = time.monotonic()
        result = ds_large[
            (ds_large["val_int"] > 100) & (ds_large["category"] == "alpha")
        ].head(20)
        df = result._execute()
        elapsed = time.monotonic() - start

        assert len(df) <= 20
        assert all(df["val_int"] > 100)
        assert all(df["category"] == "alpha")
        assert elapsed < 3.0, f"multi-filter+head took {elapsed:.2f}s"

    def test_groupby_agg_sum(self, ds_large):
        """groupby + sum should push down to SQL."""
        start = time.monotonic()
        result = ds_large.groupby("category")["val_int"].sum()
        series = result._execute()
        elapsed = time.monotonic() - start

        assert len(series) == 4
        assert elapsed < 5.0, f"groupby+sum took {elapsed:.2f}s"

    def test_groupby_agg_mean(self, ds_large):
        """groupby + mean should push down to SQL."""
        start = time.monotonic()
        result = ds_large.groupby("tag")["score"].mean()
        series = result._execute()
        elapsed = time.monotonic() - start

        assert len(series) == 3  # X, Y, Z
        assert elapsed < 5.0, f"groupby+mean took {elapsed:.2f}s"

    def test_select_filter_limit(self, ds_large):
        """select + filter + limit chain stays fast."""
        start = time.monotonic()
        result = ds_large[["id", "category", "val_int"]]
        result = result[result["val_int"] < 50].head(15)
        df = result._execute()
        elapsed = time.monotonic() - start

        assert len(df) == 15
        assert list(df.columns) == ["id", "category", "val_int"]
        assert all(df["val_int"] < 50)
        assert elapsed < 3.0, f"select+filter+limit took {elapsed:.2f}s"


class TestLargeTableSQLPushdown:
    """Verify SQL pushdown internals work correctly for common patterns."""

    def test_count_rows_uses_sql(self, ds_large):
        """count_rows() should use SQL COUNT(*), not load all data."""
        assert ds_large._can_sql_pushdown()
        count = ds_large.count_rows()
        assert count == LARGE_ROW_COUNT

    def test_shape_uses_sql_for_pristine(self, ds_large):
        """shape should use SQL COUNT + DESCRIBE for pristine source."""
        assert ds_large._is_pristine_sql_source()
        rows, cols = ds_large.shape
        assert rows == LARGE_ROW_COUNT
        assert cols == 10

    def test_columns_uses_schema_for_pristine(self, ds_large):
        """columns should use schema() for pristine source, no data load."""
        assert ds_large._is_pristine_sql_source()
        cols = ds_large.columns
        assert len(cols) == 10

    def test_filter_count_rows_pushdown(self, ds_large):
        """Filtered count_rows() should push down to SQL."""
        filtered = ds_large[ds_large["flag"] == 1]
        assert filtered._can_sql_pushdown()
        count = filtered.count_rows()
        assert count == LARGE_ROW_COUNT // 2

    def test_tail_uses_sql_optimization(self, ds_large):
        """tail() on pristine source should use SQL OFFSET/LIMIT."""
        assert ds_large._can_use_sql_tail()
        result = ds_large.tail(5)
        df = result._execute()
        assert len(df) == 5


class TestMediumTableFullDownload:
    """Full data download tests on 100K-row table (feasible size)."""

    def test_to_df_completes(self, ds_medium):
        """ds.to_df() should complete on 100K rows."""
        start = time.monotonic()
        df = ds_medium.to_df()
        elapsed = time.monotonic() - start

        assert len(df) == MEDIUM_ROW_COUNT
        assert len(df.columns) == 10
        print(f"to_df() of {MEDIUM_ROW_COUNT} rows took {elapsed:.2f}s")

    def test_info_with_counts(self, ds_medium):
        """ds.info() with non-null counts on medium table."""
        buf = io.StringIO()
        start = time.monotonic()
        ds_medium.info(buf=buf)
        elapsed = time.monotonic() - start

        output = buf.getvalue()
        assert "id" in output
        assert "category" in output
        print(f"info() of {MEDIUM_ROW_COUNT} rows took {elapsed:.2f}s")

    def test_tail_result_correctness(self, ds_medium):
        """tail() results should match the actual last rows."""
        result = ds_medium.tail(5)
        df = result._execute()
        assert len(df) == 5
        # Last row should have id = MEDIUM_ROW_COUNT - 1
        assert df["id"].max() == MEDIUM_ROW_COUNT - 1

    def test_filter_then_to_df(self, ds_medium):
        """Filter then to_df on medium table."""
        result = ds_medium[ds_medium["val_int"] < 10]
        df = result.to_df()
        # val_int = number % 1000, so values 0-9 appear for each 1000-block
        # 100K rows / 1000 * 10 = 1000 rows
        assert len(df) == 1000
        assert all(df["val_int"] < 10)
