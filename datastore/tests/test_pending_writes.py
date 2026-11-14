"""
execute() runs a pending write (INSERT, UPDATE, DELETE), whatever read
operations the DataStore had before it.

A filter, sort or limit before insert_into() / update_set() / delete_rows()
must not turn execute() into a read: the write happens, which the tests
verify by reading the target back. Sources: a file, a chDB table, a table
on a ClickHouse server, and a DataFrame (which can't be written).
"""

import os
import subprocess
import uuid

import pandas as pd
import pytest

from datastore import DataStore
from datastore.exceptions import QueryError


ROWS = pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})

# Read operations a write may follow
BEFORE = {
    "none": lambda d: d,
    "getitem_filter": lambda d: d[d["v"] > 10],
    "filter": lambda d: d.filter(d["v"] > 10),
    "sort_head": lambda d: d.sort_values("v").head(2),
}


@pytest.mark.parametrize("before", list(BEFORE), ids=list(BEFORE))
class TestFileWrites:
    @pytest.fixture
    def path(self, tmp_path):
        path = str(tmp_path / "rows.csv")
        ROWS.to_csv(path, index=False)
        return path

    def test_insert_values(self, path, before):
        expected = pd.concat([ROWS, pd.DataFrame({"id": [4], "v": [40]})], ignore_index=True)

        ds = BEFORE[before](DataStore.from_file(path))
        ds.insert_into("id", "v").insert_values((4, 40)).execute()

        pd.testing.assert_frame_equal(pd.read_csv(path), expected)

    def test_insert_select(self, path, tmp_path, before):
        other = str(tmp_path / "other.csv")
        pd.DataFrame({"id": [7, 8], "v": [70, 80]}).to_csv(other, index=False)
        expected = pd.concat(
            [ROWS, pd.DataFrame({"id": [8], "v": [80]})], ignore_index=True
        )

        source = DataStore.from_file(other)
        ds = BEFORE[before](DataStore.from_file(path))
        ds.insert_into("id", "v").select_from(source.filter(source["v"] > 70)).execute()

        pd.testing.assert_frame_equal(pd.read_csv(path), expected)

    @pytest.mark.parametrize(
        "write",
        [lambda d: d.update_set(v=0), lambda d: d.delete_rows()],
        ids=["update", "delete"],
    )
    def test_update_delete_raise(self, path, before, write):
        # A file has no ALTER TABLE: the write fails instead of reading rows
        ds = BEFORE[before](DataStore.from_file(path))

        with pytest.raises(QueryError):
            write(ds).execute()

        pd.testing.assert_frame_equal(pd.read_csv(path), ROWS)


@pytest.fixture
def chdb_table():
    """A chDB MergeTree table holding ROWS; yields its name."""
    name = f"t_writes_{uuid.uuid4().hex[:8]}"
    ds = DataStore(table=name)
    ds.connect()
    ds.create_table({"id": "UInt64", "v": "Int64"})
    ds.insert_into("id", "v").insert_values(*map(tuple, ROWS.values.tolist())).execute()
    yield name
    ds._connection._conn.query(f"DROP TABLE IF EXISTS {name}")


def _table_rows(name):
    df = DataStore(table=name).select("*").execute().to_df()
    return df.sort_values("id").reset_index(drop=True).astype("int64")


@pytest.mark.parametrize("before", ["none", "getitem_filter", "filter"])
class TestTableWrites:
    def test_insert(self, chdb_table, before):
        expected = pd.concat([ROWS, pd.DataFrame({"id": [4], "v": [40]})], ignore_index=True)

        ds = BEFORE[before](DataStore(table=chdb_table))
        ds.insert_into("id", "v").insert_values((4, 40)).execute()

        pd.testing.assert_frame_equal(_table_rows(chdb_table), expected)

    def test_update(self, chdb_table, before):
        if before == "none":
            pytest.skip("ALTER TABLE ... UPDATE needs a WHERE clause")
        expected = ROWS.copy()
        expected.loc[expected["v"] > 10, "v"] = 0

        ds = BEFORE[before](DataStore(table=chdb_table))
        ds.update_set(v=0).execute()

        pd.testing.assert_frame_equal(_table_rows(chdb_table), expected)

    def test_delete(self, chdb_table, before):
        if before == "none":
            pytest.skip("DELETE needs a WHERE clause")
        expected = ROWS[ROWS["v"] <= 10].reset_index(drop=True)

        ds = BEFORE[before](DataStore(table=chdb_table))
        ds.delete_rows().execute()

        pd.testing.assert_frame_equal(_table_rows(chdb_table), expected)


def test_dataframe_source_write_raises():
    ds = DataStore(ROWS.copy())

    with pytest.raises(QueryError):
        ds[ds["v"] > 10].insert_into("id", "v").insert_values((4, 40)).execute()


CH_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".clickhouse", "clickhouse")


@pytest.mark.parametrize("before", ["none", "getitem_filter", "filter"])
def test_server_table_insert(clickhouse_server, before):
    host, port = clickhouse_server
    table = f"t_writes_{uuid.uuid4().hex[:8]}"

    def ddl(sql):
        subprocess.run(
            [CH_BINARY, "client", f"--port={port}", f"--query={sql}"],
            check=True,
            capture_output=True,
            timeout=30,
        )

    ddl(f"CREATE TABLE default.{table} (id UInt64, v Int64) ENGINE = MergeTree ORDER BY id")
    try:
        ddl(f"INSERT INTO default.{table} VALUES (1, 10), (2, 20), (3, 30)")
        remote = dict(host=f"{host}:{port}", database="default", table=table, user="default", password="")
        expected = pd.concat([ROWS, pd.DataFrame({"id": [4], "v": [40]})], ignore_index=True)

        ds = BEFORE[before](DataStore("clickhouse", **remote))
        ds.insert_into("id", "v").insert_values((4, 40)).execute()

        read = DataStore("clickhouse", **remote).to_pandas()
        read = read.sort_values("id").reset_index(drop=True).astype("int64")
        pd.testing.assert_frame_equal(read, expected)
    finally:
        ddl(f"DROP TABLE IF EXISTS default.{table}")
