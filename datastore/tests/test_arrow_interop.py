"""
Tests for zero-copy Arrow interop on DataStore:

- ``DataStore.to_arrow()``          -> native-typed ``pyarrow.Table`` (no pandas round-trip)
- ``DataStore.__arrow_c_stream__()``-> Arrow PyCapsule interface (pandas 3.0 standard),
                                       consumed zero-copy by pyarrow / Polars / DuckDB /
                                       ``pandas.DataFrame.from_arrow``
- ``DataStore.from_arrow()``        -> ingest any Arrow PyCapsule producer

The headline win is on export: for a pipeline that fully pushes down to one chDB SQL
query, chDB emits Arrow directly, so ClickHouse-native Arrow types survive (``UInt64`` ->
``uint64``) with no numpy materialization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datastore import DataStore


@pytest.fixture
def parquet_path(tmp_path):
    """A parquet file with varied types, incl. an unsigned int and a nullable int."""
    tbl = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.uint64()),
            "city": ["NY", "LA", "NY", "SF", "LA"],
            "score": [1.5, 2.5, 3.5, 4.5, 5.5],
            "opt": pa.array([10, None, 30, None, 50], pa.int64()),
        }
    )
    path = tmp_path / "people.parquet"
    pq.write_table(tbl, path)
    return str(path)


# --------------------------------------------------------------------------------------
# Export: to_arrow() native path
# --------------------------------------------------------------------------------------
class TestToArrowNative:
    def test_returns_pyarrow_table(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        tbl = ds.to_arrow()
        assert isinstance(tbl, pa.Table)
        assert tbl.num_rows == 5
        assert tbl.column_names == ["id", "city", "score", "opt"]

    def test_preserves_native_unsigned_type(self, parquet_path):
        """The whole point: chDB emits Arrow directly, so UInt64 stays uint64
        instead of being coerced by a pandas round-trip."""
        ds = DataStore.from_file(parquet_path)
        tbl = ds.to_arrow()
        assert tbl.schema.field("id").type == pa.uint64()

    def test_filter_is_pushed_down(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        tbl = ds.filter(ds.id > 3).to_arrow()
        assert tbl.num_rows == 2
        assert tbl.column("id").to_pylist() == [4, 5]
        # native type survives through the pushed-down filter
        assert tbl.schema.field("id").type == pa.uint64()

    def test_select_projection(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        tbl = ds.select("id", "city").to_arrow()
        assert tbl.column_names == ["id", "city"]

    def test_groupby_aggregation_values(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        tbl = ds.groupby("city").agg({"score": "sum"}).to_arrow()
        got = dict(zip(tbl.column("city").to_pylist(), tbl.column("score").to_pylist()))
        assert got == {"NY": 5.0, "LA": 8.0, "SF": 4.5}

    def test_native_and_pandas_flavored_have_same_rows(self, parquet_path):
        """native and pandas-flavored Arrow must carry the SAME rows/columns
        (comparing as Arrow avoids NaN-vs-NA / dtype-backend noise)."""
        ds = DataStore.from_file(parquet_path)
        native = ds.filter(ds.id > 1).to_arrow(types="native")
        ds2 = DataStore.from_file(parquet_path)
        pandas_flavored = ds2.filter(ds2.id > 1).to_arrow(types="pandas")
        assert native.column_names == pandas_flavored.column_names
        assert native.to_pydict() == pandas_flavored.to_pydict()

    def test_nullable_int_null_survives(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        tbl = ds.to_arrow()
        assert tbl.column("opt").to_pylist() == [10, None, 30, None, 50]

    def test_invalid_types_arg_raises(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        with pytest.raises(ValueError):
            ds.to_arrow(types="bogus")


# --------------------------------------------------------------------------------------
# Export: fallback path (DataFrame source / pandas-side ops) still yields a Table
# --------------------------------------------------------------------------------------
class TestToArrowFallback:
    def test_from_df_source_returns_table(self):
        ds = DataStore.from_df(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        tbl = ds.filter(ds.a > 1).to_arrow()
        assert isinstance(tbl, pa.Table)
        assert tbl.column("a").to_pylist() == [2, 3]

    def test_types_pandas_matches_to_df_dtypes(self, parquet_path):
        """types='pandas' routes through to_df(), so Arrow dtypes line up with it."""
        ds = DataStore.from_file(parquet_path)
        pandas_flavored = ds.to_arrow(types="pandas")
        # Values still correct
        assert pandas_flavored.num_rows == 5
        # And equivalent to converting to_df() ourselves
        expected = pa.Table.from_pandas(
            DataStore.from_file(parquet_path).to_df(), preserve_index=False
        )
        assert pandas_flavored.schema == expected.schema


# --------------------------------------------------------------------------------------
# Export: Arrow PyCapsule interface (__arrow_c_stream__) — zero glue for consumers
# --------------------------------------------------------------------------------------
class TestArrowCStream:
    def test_pyarrow_consumes_datastore(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        tbl = pa.table(ds)  # uses __arrow_c_stream__
        assert tbl.num_rows == 5
        assert tbl.schema.field("id").type == pa.uint64()

    def test_pandas_from_arrow_consumes_datastore(self, parquet_path):
        ds = DataStore.from_file(parquet_path)
        pdf = pd.DataFrame.from_arrow(ds.filter(ds.id > 3))  # pandas 3.0 standard
        assert pdf["id"].tolist() == [4, 5]

    def test_polars_consumes_datastore(self, parquet_path):
        pl = pytest.importorskip("polars")
        ds = DataStore.from_file(parquet_path)
        plf = pl.from_arrow(ds.filter(ds.id > 3))
        assert plf.shape == (2, 4)
        assert str(plf.schema["id"]) == "UInt64"  # native type survives into Polars

    def test_duckdb_consumes_datastore(self, parquet_path):
        duckdb = pytest.importorskip("duckdb")
        ds = DataStore.from_file(parquet_path)
        (count, total) = duckdb.sql("SELECT count(*), sum(id) FROM ds").fetchone()
        assert count == 5
        assert total == 15


# --------------------------------------------------------------------------------------
# Import: from_arrow() ingests any Arrow PyCapsule producer
# --------------------------------------------------------------------------------------
class TestFromArrow:
    def test_from_pyarrow_table(self):
        tbl = pa.table({"n": [1, 2, 3], "s": ["a", "b", "c"]})
        ds = DataStore.from_arrow(tbl)
        assert isinstance(ds, DataStore)
        assert ds.to_df()["n"].tolist() == [1, 2, 3]

    def test_from_record_batch(self):
        batch = pa.record_batch({"n": [1, 2, 3]})
        ds = DataStore.from_arrow(batch)
        assert ds.to_df()["n"].tolist() == [1, 2, 3]

    def test_from_record_batch_reader(self):
        tbl = pa.table({"n": [1, 2, 3]})
        reader = tbl.to_reader()
        ds = DataStore.from_arrow(reader)
        assert ds.to_df()["n"].tolist() == [1, 2, 3]

    def test_from_pandas(self):
        ds = DataStore.from_arrow(pd.DataFrame({"n": [1, 2], "s": ["a", "b"]}))
        assert ds.to_df()["n"].tolist() == [1, 2]

    def test_from_polars(self):
        pl = pytest.importorskip("polars")
        ds = DataStore.from_arrow(pl.DataFrame({"n": [1, 2, 3], "s": ["a", "b", "c"]}))
        assert ds.to_df()["s"].tolist() == ["a", "b", "c"]

    def test_from_datastore(self, parquet_path):
        src = DataStore.from_file(parquet_path)
        ds = DataStore.from_arrow(src.filter(src.id > 3))
        assert ds.to_df()["id"].tolist() == [4, 5]

    def test_bad_input_raises_typeerror(self):
        with pytest.raises(TypeError):
            DataStore.from_arrow(object())


# --------------------------------------------------------------------------------------
# Round-trips
# --------------------------------------------------------------------------------------
class TestRoundTrip:
    def test_datastore_arrow_datastore(self, parquet_path):
        """DataStore -> Arrow -> DataStore -> Arrow preserves values, columns and
        nullability (compared as Arrow to stay dtype-backend agnostic)."""
        original = DataStore.from_file(parquet_path).to_arrow()
        roundtrip = DataStore.from_arrow(original).to_arrow()
        assert roundtrip.column_names == original.column_names
        assert roundtrip.to_pydict() == original.to_pydict()

    def test_polars_roundtrip(self, parquet_path):
        pl = pytest.importorskip("polars")
        ds = DataStore.from_file(parquet_path)
        back = DataStore.from_arrow(pl.from_arrow(ds))
        assert back.to_df()["id"].tolist() == [1, 2, 3, 4, 5]
