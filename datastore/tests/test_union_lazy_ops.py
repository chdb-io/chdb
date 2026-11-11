"""
union() must carry each side's own lazy operations (issue #668).

Each side of ``a.union(b)`` is evaluated with its complete pipeline, and
neither side's operations leak onto the union result. Covers SQL-composed
unions (both pipelines are single SQL queries) and the pandas fallback.

All tests use the mirror pattern: DataStore result == pandas result.
"""

import os
import uuid

import pandas as pd
import pytest

import chdb
from datastore import DataStore, concat as ds_concat
from datastore.exceptions import ExecutionError
from tests.test_utils import assert_datastore_equals_pandas


SALES = pd.DataFrame(
    {
        "region": ["a", "b", "a", "c", "b"],
        "price": [1.5, 2.25, 3.5, 4.75, 2.25],
        "qty": [1, 2, 3, 4, 2],
    }
)


@pytest.fixture
def sales_files(tmp_path):
    path = str(tmp_path / "sales.parquet")
    SALES.to_parquet(path)
    reordered = str(tmp_path / "sales_reordered.parquet")
    SALES[["qty", "region", "price"]].to_parquet(reordered)
    return path, reordered


class TestUnionCarriesLazyOps:
    def test_issue_chain_filter_left_assign_right(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 2]
        pb = SALES.copy()
        pb["price"] = pb["price"] * 10
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        b = DataStore.from_file(path)
        b["price"] = b["price"] * 10
        ds_result = a.union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_ops_only_on_left(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 2]
        pd_result = pd.concat([pa, SALES], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        ds_result = a.union(DataStore.from_file(path), all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ops_only_on_right(self, sales_files):
        path, _ = sales_files
        pb = SALES.copy()
        pb["price"] = pb["price"] * 10
        pd_result = pd.concat([SALES, pb], ignore_index=True)

        b = DataStore.from_file(path)
        b["price"] = b["price"] * 10
        ds_result = DataStore.from_file(path).union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ops_on_both_sides(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["region"] != "b"].copy()
        pa["total"] = pa["price"] * pa["qty"]
        pb = SALES[SALES["qty"] < 3].copy()
        pb["total"] = pb["qty"] + 100
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["region"] != "b"]
        a["total"] = a["price"] * a["qty"]
        b = DataStore.from_file(path)
        b = b[b["qty"] < 3]
        b["total"] = b["qty"] + 100
        ds_result = a.union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_limit_on_right(self, sales_files):
        path, _ = sales_files
        pd_result = pd.concat([SALES, SALES.head(2)], ignore_index=True)

        ds_result = DataStore.from_file(path).union(
            DataStore.from_file(path).head(2), all=True
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_union_distinct_with_ops(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 1]
        pb = SALES.copy()
        pb.loc[pb["region"] == "c", "price"] = 0.5
        pd_result = pd.concat([pa, pb], ignore_index=True).drop_duplicates()

        a = DataStore.from_file(path)
        a = a[a["qty"] > 1]
        b = DataStore.from_file(path)
        b["price"] = b["price"].where(b["region"] != "c", 0.5)
        ds_result = a.union(b)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_union_distinct_without_ops(self, sales_files):
        path, _ = sales_files
        pd_result = pd.concat([SALES, SALES], ignore_index=True).drop_duplicates()

        ds_result = DataStore.from_file(path).union(DataStore.from_file(path))

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_side_with_computed_column_referenced_by_filter(self, sales_files):
        path, _ = sales_files
        pa = SALES.copy()
        pa["total"] = pa["price"] * pa["qty"]
        pa = pa[pa["total"] > 4]
        pb = SALES.copy()
        pb["total"] = pb["price"] + pb["qty"]
        pb = pb[pb["total"] < 6]
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = DataStore.from_file(path)
        a["total"] = a["price"] * a["qty"]
        a = a[a["total"] > 4]
        b = DataStore.from_file(path)
        b["total"] = b["price"] + b["qty"]
        b = b[b["total"] < 6]
        ds_result = a.union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_side_with_invalid_sql_raises_at_execution(self, tmp_path):
        path = str(tmp_path / "codes.csv")
        pd.DataFrame({"code": ["x1", "x2"], "n": [1, 2]}).to_csv(path, index=False)

        b = DataStore.from_file(path)
        b["code"] = b["code"] + 1
        ds_result = DataStore.from_file(path).union(b, all=True)

        with pytest.raises(ExecutionError):
            ds_result.to_pandas()

    def test_different_column_order_with_ops(self, sales_files):
        path, reordered = sales_files
        right = SALES[["qty", "region", "price"]]
        pa = SALES[SALES["qty"] > 2]
        pb = right[right["qty"] < 3].copy()
        pb["price"] = pb["price"] * 10
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        b = DataStore.from_file(reordered)
        b = b[b["qty"] < 3]
        b["price"] = b["price"] * 10
        ds_result = a.union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_right_side_with_pandas_only_op(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 2]
        pb = SALES.copy()
        pb["price"] = pb["price"].apply(lambda x: x * 3)
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        b = DataStore.from_file(path)
        b["price"] = b["price"].apply(lambda x: x * 3)
        ds_result = a.union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ops_after_union(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 2]
        pb = SALES.copy()
        pb["price"] = pb["price"] * 10
        pd_union = pd.concat([pa, pb], ignore_index=True)
        pd_result = pd_union[pd_union["price"] > 4]

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        b = DataStore.from_file(path)
        b["price"] = b["price"] * 10
        ds_union = a.union(b, all=True)
        ds_result = ds_union[ds_union["price"] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_operands_unchanged(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 2]
        pb = SALES.copy()
        pb["price"] = pb["price"] * 10

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        b = DataStore.from_file(path)
        b["price"] = b["price"] * 10
        a.union(b, all=True)

        assert_datastore_equals_pandas(a, pa)
        assert_datastore_equals_pandas(b, pb)

    def test_concat_ignore_index_with_ops(self, sales_files):
        path, _ = sales_files
        pa = SALES[SALES["qty"] > 2]
        pb = SALES.copy()
        pb["price"] = pb["price"] * 10
        pc = SALES[SALES["region"] == "b"]
        pd_result = pd.concat([pa, pb, pc], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["qty"] > 2]
        b = DataStore.from_file(path)
        b["price"] = b["price"] * 10
        c = DataStore.from_file(path)
        c = c[c["region"] == "b"]
        ds_result = ds_concat([a, b, c], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUnionOperandState:
    """A union run in pandas reads the other side as it was at union()."""

    @pytest.mark.parametrize("other_kind", ["dataframe", "file"])
    def test_other_changed_after_union(self, sales_files, other_kind):
        path, _ = sales_files
        pd_result = pd.concat([SALES, SALES], ignore_index=True)

        other = DataStore(SALES.copy()) if other_kind == "dataframe" else DataStore.from_file(path)
        ds_result = DataStore(SALES.copy()).union(other, all=True)
        other["qty"] = other["qty"] * 100

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUnionHelperColumnNames:
    """The SQL union orders rows with helper columns; user columns may share
    their names."""

    @pytest.mark.parametrize(
        "columns",
        [
            ["__union_side__"],
            ["__union_row__"],
            ["__union_side__", "__union_side_1__", "__union_row__"],
        ],
    )
    def test_column_named_like_helper(self, tmp_path, columns):
        df = SALES.copy()
        for i, name in enumerate(columns):
            df[name] = df["qty"] * (10 + i)
        path = str(tmp_path / "helper_names.parquet")
        df.to_parquet(path)
        pa = df[df["qty"] > 1]
        pb = df.copy()
        pb[columns[0]] = pb[columns[0]] + 1
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = DataStore.from_file(path)
        a = a[a["qty"] > 1]
        b = DataStore.from_file(path)
        b[columns[0]] = b[columns[0]] + 1
        ds_result = a.union(b, all=True)

        assert "UNION ALL" in ds_result.to_sql()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


# ---------------------------------------------------------------------------
# Remote ClickHouse tables (same env-driven server setup as test_writeback.py)
# ---------------------------------------------------------------------------

HOST = os.environ.get("CHDB_TEST_HOST", "")
USER = os.environ.get("CHDB_TEST_USER", "")
PASSWORD = os.environ.get("CHDB_TEST_PASSWORD", "")


def _ddl(sql: str) -> None:
    escaped = sql.replace("\\", "\\\\").replace("'", "''")
    chdb.query(
        f"SELECT * FROM remote('{HOST}', query = '{escaped}', '{USER}', '{PASSWORD}')",
        "TabSeparated",
    )


@pytest.fixture
def remote_sales_table():
    if not (HOST and USER):
        pytest.skip("Source ClickHouse env vars not set (CHDB_TEST_HOST, CHDB_TEST_USER)")
    try:
        chdb.query(
            f"SELECT 1 FROM remote('{HOST}', 'system', 'one', '{USER}', '{PASSWORD}')"
        )
    except Exception as e:
        pytest.skip(f"Source ClickHouse unreachable: {e}")

    database = f"test_union_668_{uuid.uuid4().hex[:8]}"
    _ddl(f'CREATE DATABASE "{database}"')
    try:
        _ddl(
            f'CREATE TABLE "{database}"."sales" '
            f"(id Int64, region String, price Float64, qty Int64) "
            f"ENGINE = MergeTree() ORDER BY id"
        )
        rows = ", ".join(
            f"({i}, '{r.region}', {r.price}, {r.qty})"
            for i, r in enumerate(SALES.itertuples())
        )
        _ddl(f'INSERT INTO "{database}"."sales" VALUES {rows}')
        yield database
    finally:
        _ddl(f'DROP DATABASE IF EXISTS "{database}"')


class TestRemoteUnionCarriesLazyOps:
    def _make_ds(self, database):
        return DataStore.from_clickhouse(
            host=HOST, database=database, table="sales", user=USER, password=PASSWORD
        )

    def test_issue_chain_on_remote_tables(self, remote_sales_table):
        sales = SALES.assign(id=range(len(SALES)))[["id", "region", "price", "qty"]]
        pa = sales[sales["qty"] > 2]
        pb = sales.copy()
        pb["price"] = pb["price"] * 10
        pd_result = pd.concat([pa, pb], ignore_index=True)

        a = self._make_ds(remote_sales_table)
        a = a[a["qty"] > 2]
        b = self._make_ds(remote_sales_table)
        b["price"] = b["price"] * 10
        ds_result = a.union(b, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)
