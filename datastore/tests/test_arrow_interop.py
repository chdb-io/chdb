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
from datastore.pandas_col_compat import PANDAS_3_PLUS

# CI runs the suite under both pandas 2.x and pandas 3.x. A few things are pandas
# 3.0-only: the ``pd.DataFrame.from_arrow`` constructor and the default string dtype.
requires_pandas3 = pytest.mark.skipif(
    not PANDAS_3_PLUS,
    reason="requires pandas 3.0+ (pd.DataFrame.from_arrow / default string dtype)",
)


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

    @requires_pandas3
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


# --------------------------------------------------------------------------------------
# pandas 3.x compatibility
#
# Arrow interop is the pandas 3.0 interchange standard. These lock in behaviour for
# pandas-3.x dtypes as from_arrow() INPUT, pandas as a from_arrow CONSUMER, and the
# ClickHouse-native Arrow types that survive on export.
# --------------------------------------------------------------------------------------
class TestPandas3xIngestDtypes:
    """pandas 3.x dtypes as input to from_arrow() (values + null positions preserved)."""

    def test_nullable_int_boolean_float(self):
        df = pd.DataFrame(
            {
                "i": pd.array([1, None, 3], dtype="Int64"),
                "b": pd.array([True, None, False], dtype="boolean"),
                "f": pd.array([1.5, None, 3.5], dtype="Float64"),
            }
        )
        tbl = DataStore.from_arrow(df).to_arrow()
        assert tbl.column("i").to_pylist() == [1, None, 3]
        assert tbl.column("b").to_pylist() == [True, None, False]
        assert tbl.column("f").to_pylist() == [1.5, None, 3.5]

    @requires_pandas3
    def test_pandas3_default_string_dtype(self):
        df = pd.DataFrame({"s": ["a", "b", "c"]})
        # pandas 3.0 defaults object strings to the new string dtype
        assert "str" in str(df["s"].dtype) or "string" in str(df["s"].dtype)
        tbl = DataStore.from_arrow(df).to_arrow()
        assert tbl.column("s").to_pylist() == ["a", "b", "c"]

    def test_arrow_backed_dtype(self):
        df = pd.DataFrame(
            {
                "x": pd.array([1, 2, 3], dtype="int64[pyarrow]"),
                "s": pd.array(["a", "b", "c"], dtype="string[pyarrow]"),
            }
        )
        tbl = DataStore.from_arrow(df).to_arrow()
        assert tbl.column("x").to_pylist() == [1, 2, 3]
        assert tbl.column("s").to_pylist() == ["a", "b", "c"]

    def test_datetime(self):
        df = pd.DataFrame({"t": pd.to_datetime(["2020-01-01", "2020-06-15"])})
        tbl = DataStore.from_arrow(df).to_arrow()
        assert pa.types.is_timestamp(tbl.schema.field("t").type)

    def test_tz_aware_datetime(self):
        df = pd.DataFrame({"t": pd.to_datetime(["2020-01-01"]).tz_localize("UTC")})
        typ = DataStore.from_arrow(df).to_arrow().schema.field("t").type
        assert pa.types.is_timestamp(typ) and typ.tz == "UTC"

    def test_categorical(self):
        df = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
        tbl = DataStore.from_arrow(df).to_arrow()
        assert tbl.column("c").to_pylist() == ["a", "b", "a"]


@requires_pandas3
class TestPandas3xConsumer:
    """pandas 3.x consuming a DataStore via ``pd.DataFrame.from_arrow`` (the 3.0 API)."""

    def test_uint64_stays_unsigned(self, parquet_path):
        pdf = pd.DataFrame.from_arrow(DataStore.from_file(parquet_path))
        assert "uint" in str(pdf["id"].dtype).lower()

    def test_null_positions_preserved(self, parquet_path):
        pdf = pd.DataFrame.from_arrow(DataStore.from_file(parquet_path))
        # opt = [10, None, 30, None, 50] in the fixture
        assert pdf["opt"].isna().tolist() == [False, True, False, True, False]


class TestNativeArrowTypes:
    """ClickHouse-native Arrow types that survive on export (no pandas coercion)."""

    def _write(self, tmp_path, table):
        path = tmp_path / "typed.parquet"
        pq.write_table(table, path)
        return str(path)

    def test_decimal128_preserved(self, tmp_path):
        import decimal

        path = self._write(
            tmp_path,
            pa.table(
                {"d": pa.array([decimal.Decimal("1.50"), decimal.Decimal("2.25")], pa.decimal128(5, 2))}
            ),
        )
        typ = DataStore.from_file(path).to_arrow().schema.field("d").type
        assert pa.types.is_decimal(typ)

    def test_list_preserved(self, tmp_path):
        path = self._write(
            tmp_path, pa.table({"a": pa.array([[1, 2], [3]], pa.list_(pa.int64()))})
        )
        tbl = DataStore.from_file(path).to_arrow()
        assert pa.types.is_list(tbl.schema.field("a").type)
        assert tbl.column("a").to_pylist() == [[1, 2], [3]]

    def test_assign_computed_column(self, tmp_path):
        path = self._write(tmp_path, pa.table({"a": pa.array([1, 2, 3], pa.int64())}))
        ds = DataStore.from_file(path)
        tbl = ds.assign(b=ds.a * 2).to_arrow()
        assert tbl.column("b").to_pylist() == [2, 4, 6]

    def test_empty_table(self, tmp_path):
        path = self._write(tmp_path, pa.table({"a": pa.array([], pa.int64())}))
        assert DataStore.from_file(path).to_arrow().num_rows == 0

    def test_all_null_column(self, tmp_path):
        path = self._write(tmp_path, pa.table({"a": pa.array([None, None], pa.int64())}))
        assert DataStore.from_file(path).to_arrow().column("a").to_pylist() == [None, None]


# --------------------------------------------------------------------------------------
# Nested / binary / awkward-name types
#
# Type fidelity is verified via the NATIVE from_file path (chDB emits Arrow directly),
# which is independent of the installed pandas version.
# --------------------------------------------------------------------------------------
class TestNestedAndBinaryTypes:
    def _write(self, tmp_path, table):
        path = tmp_path / "nested.parquet"
        pq.write_table(table, path)
        return str(path)

    def test_map_type(self, tmp_path):
        vals = [[("a", 1), ("b", 2)], [("c", 3)]]
        path = self._write(
            tmp_path, pa.table({"m": pa.array(vals, pa.map_(pa.string(), pa.int64()))})
        )
        tbl = DataStore.from_file(path).to_arrow()
        assert pa.types.is_map(tbl.schema.field("m").type)
        assert tbl.column("m").to_pylist() == vals

    def test_struct_type(self, tmp_path):
        vals = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        path = self._write(
            tmp_path,
            pa.table({"s": pa.array(vals, pa.struct([("x", pa.int64()), ("y", pa.string())]))}),
        )
        tbl = DataStore.from_file(path).to_arrow()
        assert pa.types.is_struct(tbl.schema.field("s").type)
        assert tbl.column("s").to_pylist() == vals

    def test_fixed_size_binary(self, tmp_path):
        vals = [
            b"1234567890123456",
            b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff",
        ]
        path = self._write(tmp_path, pa.table({"b": pa.array(vals, pa.binary(16))}))
        tbl = DataStore.from_file(path).to_arrow()
        assert pa.types.is_fixed_size_binary(tbl.schema.field("b").type)
        assert tbl.column("b").to_pylist() == vals

    def test_awkward_column_names(self, tmp_path):
        tbl_in = pa.Table.from_arrays(
            [pa.array([1, 2]), pa.array([3, 4]), pa.array([5, 6])],
            names=["a b", "名前", "with-dash"],
        )
        path = self._write(tmp_path, tbl_in)
        out = DataStore.from_file(path).to_arrow()
        assert out.column_names == ["a b", "名前", "with-dash"]
        assert out.column("名前").to_pylist() == [3, 4]


# --------------------------------------------------------------------------------------
# Streaming + requested_schema (Arrow C stream). Pure pyarrow, pandas-version agnostic.
# --------------------------------------------------------------------------------------
class TestArrowStreaming:
    def test_multibatch_reader_ingest(self):
        tbl = pa.table({"n": list(range(30))})
        reader = pa.RecordBatchReader.from_batches(tbl.schema, tbl.to_batches(max_chunksize=10))
        ds = DataStore.from_arrow(reader)  # must drain all batches
        assert sorted(ds.to_df()["n"].tolist()) == list(range(30))

    def test_c_stream_consumable_via_reader(self, tmp_path):
        path = tmp_path / "stream.parquet"
        pq.write_table(pa.table({"n": list(range(1000))}), path)
        ds = DataStore.from_file(str(path))
        result = pa.RecordBatchReader.from_stream(ds).read_all()
        assert result.num_rows == 1000

    def test_requested_schema_compatible_cast(self, tmp_path):
        path = tmp_path / "cast.parquet"
        pq.write_table(pa.table({"v": pa.array([1, 2, 3], pa.int64())}), path)
        ds = DataStore.from_file(str(path))
        out = pa.RecordBatchReader.from_stream(
            ds, schema=pa.schema([("v", pa.int32())])
        ).read_all()
        assert out.schema.field("v").type == pa.int32()
        assert out.column("v").to_pylist() == [1, 2, 3]

    def test_requested_schema_incompatible_raises(self, tmp_path):
        path = tmp_path / "bad.parquet"
        pq.write_table(pa.table({"v": pa.array([1, 2, 3], pa.int64())}), path)
        ds = DataStore.from_file(str(path))
        with pytest.raises(pa.lib.ArrowException):
            pa.RecordBatchReader.from_stream(
                ds, schema=pa.schema([("v", pa.list_(pa.int64()))])
            ).read_all()


class TestIbisInterop:
    def test_ingest_from_ibis(self):
        ibis = pytest.importorskip("ibis")
        # ibis produces Arrow; DataStore.from_arrow ingests it.
        expr = ibis.memtable(pa.table({"n": [1, 2, 3]}))
        ds = DataStore.from_arrow(expr.to_pyarrow())
        assert ds.to_df()["n"].tolist() == [1, 2, 3]
