"""
Tests for Remote Table Function operation robustness.

Verifies that operations on remote data sources (MySQL, PostgreSQL, SQLite,
remote ClickHouse) do not inject rowNumberInAllBlocks(), and that filter/sort/agg
chains generate correct SQL.

The root cause: prior to v4.1.0, row order handling logic was never tested
against remote sources. rowNumberInAllBlocks() on a remote() source forces
a full table scan before any WHERE filtering, which is catastrophic for
large tables.
"""

import pytest
import pandas as pd

from datastore import DataStore
from datastore.sql_executor import SQLExecutionEngine
from datastore.table_functions import (
    MySQLTableFunction,
    PostgreSQLTableFunction,
    SQLiteTableFunction,
    RemoteTableFunction,
    MongoDBTableFunction,
    RedisTableFunction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ds_with_table_function(tf, schema=None):
    """
    Create a DataStore backed by the given table function.

    Constructs a minimal DataStore and wires the table function directly,
    bypassing remote connectivity. Schema is set so that SQL generation
    works without needing to query the remote source for column metadata.
    """
    ds = DataStore({"_placeholder": [1]})  # seed with dummy DataFrame
    # Override internals to use the table function as data source
    ds._table_function = tf
    ds._format_settings = {}
    ds._source_df = None
    ds._source_df_name = None
    ds.table_name = None
    ds.source_type = "remote"
    # Clear DataFrame-based lazy ops so the planner treats this as a SQL source
    ds._lazy_ops = []
    if schema:
        ds._schema = schema
    return ds


REMOTE_TABLE_FUNCTIONS = [
    pytest.param(
        MySQLTableFunction(
            host="mysql.example.com:3306",
            database="testdb",
            table="orders",
            user="reader",
            password="pass",
        ),
        id="mysql",
    ),
    pytest.param(
        PostgreSQLTableFunction(
            host="pg.example.com:5432",
            database="analytics",
            table="events",
            user="analyst",
            password="pass",
        ),
        id="postgresql",
    ),
    pytest.param(
        SQLiteTableFunction(
            database_path="/data/local.db",
            table="metrics",
        ),
        id="sqlite",
    ),
    pytest.param(
        RemoteTableFunction(
            host="ch-replica.example.com:9000",
            database="prod",
            table="logs",
            user="default",
            password="",
        ),
        id="remote_clickhouse",
    ),
    pytest.param(
        MongoDBTableFunction(
            host="mongo.example.com:27017",
            database="app",
            collection="users",
            user="ro",
            password="pass",
        ),
        id="mongodb",
    ),
    pytest.param(
        RedisTableFunction(
            host="redis.example.com:6379",
            key="id",
            structure="id UInt64, name String, value Float64",
        ),
        id="redis",
    ),
]

SCHEMA = {"id": "UInt64", "name": "String", "value": "Float64", "category": "String"}


# ===========================================================================
# 1. preserves_row_order() returns True for every remote source
# ===========================================================================


class TestRemotePreservesRowOrder:
    """All remote / external DB table functions must report preserves_row_order=True."""

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_preserves_row_order_returns_true(self, tf):
        assert tf.preserves_row_order() is True

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_preserves_row_order_with_empty_settings(self, tf):
        assert tf.preserves_row_order({}) is True

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_preserves_row_order_with_none_settings(self, tf):
        assert tf.preserves_row_order(None) is True

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_source_preserves_row_order_via_engine(self, tf):
        """SQLExecutionEngine.source_preserves_row_order() must return True."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        engine = SQLExecutionEngine(ds)
        assert engine.source_preserves_row_order() is True


# ===========================================================================
# 2. SQL must never contain rowNumberInAllBlocks() for remote sources
# ===========================================================================


class TestNoRowNumberInAllBlocks:
    """Verify generated SQL never injects rowNumberInAllBlocks() for remote sources."""

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_plain_select_no_row_number(self, tf):
        """Plain SELECT * must not contain rowNumberInAllBlocks()."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        sql = ds._build_execution_sql()
        if sql:
            assert "rowNumberInAllBlocks" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_filter_no_row_number(self, tf):
        """WHERE clause must not inject rowNumberInAllBlocks()."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_filtered = ds[ds["value"] > 100]
        sql = ds_filtered._build_execution_sql()
        assert sql is not None, "Expected SQL to be generated for filtered remote source"
        assert "rowNumberInAllBlocks" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_sort_no_row_number(self, tf):
        """ORDER BY must not inject rowNumberInAllBlocks()."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_sorted = ds.sort_values("value")
        sql = ds_sorted._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_head_no_row_number(self, tf):
        """LIMIT must not inject rowNumberInAllBlocks()."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_head = ds.head(10)
        sql = ds_head._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_filter_sort_head_chain_no_row_number(self, tf):
        """filter -> sort -> head chain must not inject rowNumberInAllBlocks()."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_result = ds[ds["value"] > 0].sort_values("name").head(20)
        sql = ds_result._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_groupby_agg_no_row_number(self, tf):
        """GROUP BY + aggregation must not inject rowNumberInAllBlocks()."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_grouped = ds.groupby("category").agg({"value": "sum"})
        sql = ds_grouped._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql


# ===========================================================================
# 3. SQL correctness: filter -> sort -> head chain
# ===========================================================================


class TestFilterSortHeadSQL:
    """Verify SQL structure for filter -> sort -> head on remote sources."""

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_filter_generates_where(self, tf):
        """Filter should produce a WHERE clause."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_filtered = ds[ds["value"] > 100]
        sql = ds_filtered._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "WHERE" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_sort_generates_order_by(self, tf):
        """sort_values should produce an ORDER BY clause."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_sorted = ds.sort_values("value", ascending=False)
        sql = ds_sorted._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "ORDER BY" in sql_upper
        assert "DESC" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_head_generates_limit(self, tf):
        """head(n) should produce a LIMIT clause."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_head = ds.head(5)
        sql = ds_head._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "LIMIT" in sql_upper
        assert "5" in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_filter_sort_head_chain_correct(self, tf):
        """filter -> sort -> head should produce WHERE + ORDER BY + LIMIT."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_result = ds[ds["value"] > 0].sort_values("name").head(20)
        sql = ds_result._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "WHERE" in sql_upper
        assert "ORDER BY" in sql_upper
        assert "LIMIT" in sql_upper
        assert "20" in sql
        # No full-table-scan wrapper
        assert "rowNumberInAllBlocks" not in sql
        assert "__orig_row_num__" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_filter_no_full_table_scan_subquery(self, tf):
        """
        Filter on remote source should NOT wrap with a subquery that selects *
        from the entire table first (the rowNumberInAllBlocks pattern).
        """
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_filtered = ds[ds["value"] > 100]
        sql = ds_filtered._build_execution_sql()
        assert sql is not None
        # The problematic pattern wraps the source in:
        #   SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM <source>
        # which forces a full table scan BEFORE WHERE.
        assert "__orig_row_num__" not in sql


# ===========================================================================
# 4. SQL correctness: GROUP BY + aggregation
# ===========================================================================


class TestGroupByAggSQL:
    """Verify SQL structure for GROUP BY + aggregation on remote sources."""

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_groupby_sum(self, tf):
        """GROUP BY + sum() should generate correct SQL."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_grouped = ds.groupby("category").agg({"value": "sum"})
        sql = ds_grouped._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "GROUP BY" in sql_upper
        assert "SUM" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_groupby_mean(self, tf):
        """GROUP BY + mean() should generate correct SQL."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_grouped = ds.groupby("category").agg({"value": "mean"})
        sql = ds_grouped._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "GROUP BY" in sql_upper
        assert "AVG" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_groupby_count(self, tf):
        """GROUP BY + count() should generate correct SQL."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_grouped = ds.groupby("category").agg({"value": "count"})
        sql = ds_grouped._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "GROUP BY" in sql_upper
        assert "COUNT" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_groupby_multiple_aggs(self, tf):
        """GROUP BY with multiple aggregation functions."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_grouped = ds.groupby("category").agg({"value": ["sum", "mean", "count"]})
        sql = ds_grouped._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "GROUP BY" in sql_upper
        assert "SUM" in sql_upper
        assert "AVG" in sql_upper
        assert "COUNT" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_filter_then_groupby(self, tf):
        """filter -> groupby should produce WHERE + GROUP BY."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_result = ds[ds["value"] > 0].groupby("category").agg({"value": "sum"})
        sql = ds_result._build_execution_sql()
        assert sql is not None
        sql_upper = sql.upper()
        assert "WHERE" in sql_upper
        assert "GROUP BY" in sql_upper
        assert "SUM" in sql_upper
        # Must not inject row ordering for aggregated results
        assert "rowNumberInAllBlocks" not in sql
        assert "__orig_row_num__" not in sql


# ===========================================================================
# 5. RemoteTableFunction-specific: to_sql correctness
# ===========================================================================


class TestRemoteTableFunctionToSQL:
    """Verify to_sql() output for remote table functions is well-formed."""

    def test_mysql_to_sql(self):
        tf = MySQLTableFunction(
            host="db.example.com:3306",
            database="mydb",
            table="users",
            user="admin",
            password="secret",
        )
        sql = tf.to_sql()
        assert sql.startswith("mysql(")
        assert "'db.example.com:3306'" in sql
        assert "'mydb'" in sql
        assert "'users'" in sql
        assert "'admin'" in sql

    def test_postgresql_to_sql(self):
        tf = PostgreSQLTableFunction(
            host="pg.example.com:5432",
            database="analytics",
            table="events",
            user="readonly",
            password="pw",
        )
        sql = tf.to_sql()
        assert sql.startswith("postgresql(")
        assert "'pg.example.com:5432'" in sql
        assert "'analytics'" in sql
        assert "'events'" in sql

    def test_sqlite_to_sql(self):
        tf = SQLiteTableFunction(
            database_path="/tmp/test.db",
            table="metrics",
        )
        sql = tf.to_sql()
        assert sql.startswith("sqlite(")
        assert "'/tmp/test.db'" in sql
        assert "'metrics'" in sql

    def test_remote_clickhouse_to_sql(self):
        tf = RemoteTableFunction(
            host="ch.example.com:9000",
            database="default",
            table="hits",
            user="default",
            password="",
        )
        sql = tf.to_sql()
        assert sql.startswith("remote(")
        assert "'ch.example.com:9000'" in sql
        assert "'default'" in sql
        assert "'hits'" in sql

    def test_remote_secure_to_sql(self):
        tf = RemoteTableFunction(
            host="ch.example.com:9440",
            database="prod",
            table="logs",
            user="admin",
            password="pw",
            secure=True,
        )
        sql = tf.to_sql()
        assert sql.startswith("remoteSecure(")

    def test_mongodb_to_sql(self):
        tf = MongoDBTableFunction(
            host="mongo.example.com:27017",
            database="app",
            collection="docs",
            user="viewer",
            password="pw",
        )
        sql = tf.to_sql()
        assert sql.startswith("mongodb(")
        assert "'app'" in sql
        assert "'docs'" in sql


# ===========================================================================
# 6. Edge cases
# ===========================================================================


class TestRemoteEdgeCases:
    """Edge cases for remote table function handling."""

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_multiple_filters_no_row_number(self, tf):
        """Chained boolean filter should not inject rowNumberInAllBlocks."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_result = ds[(ds["value"] > 0) & (ds["value"] < 1000)]
        sql = ds_result._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql
        assert "__orig_row_num__" not in sql
        sql_upper = sql.upper()
        assert "WHERE" in sql_upper

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_sort_ascending_and_descending(self, tf):
        """Both ASC and DESC sort should produce clean SQL."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)

        ds_asc = ds.sort_values("value", ascending=True)
        sql_asc = ds_asc._build_execution_sql()
        assert sql_asc is not None
        assert "rowNumberInAllBlocks" not in sql_asc

        ds_desc = ds.sort_values("value", ascending=False)
        sql_desc = ds_desc._build_execution_sql()
        assert sql_desc is not None
        assert "rowNumberInAllBlocks" not in sql_desc
        assert "DESC" in sql_desc.upper()

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_head_zero(self, tf):
        """head(0) should produce LIMIT 0."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_head = ds.head(0)
        sql = ds_head._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql

    @pytest.mark.parametrize("tf", REMOTE_TABLE_FUNCTIONS)
    def test_column_selection_no_row_number(self, tf):
        """Selecting specific columns should not inject rowNumberInAllBlocks."""
        ds = _make_ds_with_table_function(tf, schema=SCHEMA)
        ds_cols = ds[["id", "name"]]
        sql = ds_cols._build_execution_sql()
        assert sql is not None
        assert "rowNumberInAllBlocks" not in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
