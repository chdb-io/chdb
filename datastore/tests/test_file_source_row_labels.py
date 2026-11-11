"""
Rows read from a file keep their file positions as labels (issues #735, #718).

pandas labels the rows it reads from a file 0..n-1 and keeps those labels
through filters, sorts and limits. A DataStore over a file must label the
same rows the same way, whether the operations run in SQL or in pandas.

All tests use the mirror pattern and compare the index.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas as _assert_values_equal


def assert_datastore_equals_pandas(ds_result, pd_result):
    """Compare columns, values and row order, then the row labels."""
    _assert_values_equal(ds_result, pd_result)
    # RangeIndex and an Index of the same integers are the same labels
    pd.testing.assert_index_equal(ds_result.index, pd_result.index, exact=False)


SALES = pd.DataFrame(
    {
        "region": list("nsnsnsns"),
        "price": [8, 7, 6, 5, 4, 3, 2, 1],
        "qty": [5, 20, 15, 30, 11, 12, 1, 40],
    }
)


def _write(df, path, fmt):
    if fmt == "parquet":
        df.to_parquet(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", lines=True)


@pytest.fixture(params=["dataframe", "parquet", "csv", "jsonl"])
def make_source(request, tmp_path):
    """Return a factory building a DataStore over ``df`` from one source kind."""

    def make(df):
        if request.param == "dataframe":
            return DataStore(df.copy())
        path = str(tmp_path / f"data.{request.param}")
        _write(df, path, request.param)
        return DataStore.from_file(path)

    return make


class TestFileSourceRowLabels:
    def test_issue_735_sort_head(self, make_source):
        df = pd.DataFrame({"k": [5, 3, 4, 1, 2, 0], "v": list("abcdef")})
        pd_result = df.sort_values("k").head(3)

        ds_result = make_source(df).sort_values("k").head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_issue_718_sort_filter_select_head_head(self, make_source):
        def chain(d):
            d = d.sort_values("price", ascending=False)
            d = d[d["qty"] > 10][["region", "qty"]]
            return d.head(3).head(2)

        pd_result = chain(SALES.copy())

        ds_result = chain(make_source(SALES))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter(self, make_source):
        pd_result = SALES[SALES["qty"] > 10]

        ds = make_source(SALES)
        ds_result = ds[ds["qty"] > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self, make_source):
        pd_df = SALES.copy()
        pd_df["total"] = pd_df["price"] * pd_df["qty"]
        pd_result = pd_df[pd_df["total"] > 50]

        ds = make_source(SALES)
        ds["total"] = ds["price"] * ds["qty"]
        ds_result = ds[ds["total"] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_then_filter(self, make_source):
        pd_head = SALES.head(5)
        pd_result = pd_head[pd_head["qty"] > 10]

        ds_head = make_source(SALES).head(5)
        ds_result = ds_head[ds_head["qty"] > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_with_ties_head(self, make_source):
        pd_filtered = SALES[SALES["qty"] > 1]
        pd_result = pd_filtered.sort_values("region").head(4)

        ds = make_source(SALES)
        ds_filtered = ds[ds["qty"] > 1]
        ds_result = ds_filtered.sort_values("region").head(4)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_rows(self, make_source):
        pd_result = SALES.iloc[2:5]

        ds_result = make_source(SALES).iloc[2:5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_select_columns(self, make_source):
        pd_result = SALES[SALES["region"] == "s"][["qty"]]

        ds = make_source(SALES)
        ds_result = ds[ds["region"] == "s"][["qty"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates(self, make_source):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3, 1], "b": list("xxyyzx")})
        pd_result = df.drop_duplicates()

        ds_result = make_source(df).drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset_after_filter(self, make_source):
        pd_filtered = SALES[SALES["qty"] > 5]
        pd_result = pd_filtered.drop_duplicates(subset=["region"])

        ds = make_source(SALES)
        ds_result = ds[ds["qty"] > 5].drop_duplicates(subset=["region"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail(self, make_source):
        pd_result = SALES.tail(3)

        ds_result = make_source(SALES).tail(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_tail(self, make_source):
        pd_result = SALES[SALES["qty"] > 5].tail(2)

        ds = make_source(SALES)
        ds_result = ds[ds["qty"] > 5].tail(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_offset_limit(self, make_source):
        pd_result = SALES.iloc[2:5]

        ds_result = make_source(SALES).offset(2).limit(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.parametrize(
        "chain",
        [
            lambda d: d.head(5)[["region", "qty"]].groupby("region").first(),
            lambda d: d.tail(5)[["region", "qty"]].groupby("region").last(),
            lambda d: d.head(5)[["region", "qty"]].groupby("region", sort=False).first(),
            lambda d: d[d["qty"] > 5][["region", "price"]].head(4).groupby("region").last(),
        ],
        ids=["head_first", "tail_last", "head_first_unsorted", "filter_head_last"],
    )
    def test_projection_then_first_last(self, make_source, chain):
        pd_result = chain(SALES)

        ds_result = chain(make_source(SALES))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filtered_series_labels(self, make_source):
        pd_result = SALES[SALES["qty"] > 10]["price"]

        ds = make_source(SALES)
        ds_result = ds[ds["qty"] > 10]["price"]

        assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("fmt", ["parquet", "csv", "jsonl"])
class TestLargeFileRowLabels:
    """Files read in many blocks, row groups and parsing chunks."""

    N = 120_000

    @pytest.fixture
    def large(self, tmp_path, fmt):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"k": rng.permutation(self.N), "v": np.arange(self.N)})
        path = str(tmp_path / f"large.{fmt}")
        if fmt == "parquet":
            df.to_parquet(path, row_group_size=5_000)
        else:
            _write(df, path, fmt)
        return df, path

    def test_filter(self, large):
        df, path = large
        pd_result = df[df["k"] % 7 == 0]

        ds = DataStore.from_file(path)
        ds_result = ds[ds["k"] % 7 == 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_on_sorted_column(self, large):
        df, path = large
        pd_result = df[(df["v"] > 100_000) & (df["k"] < 30_000)]

        ds = DataStore.from_file(path)
        ds_result = ds[(ds["v"] > 100_000) & (ds["k"] < 30_000)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_head(self, large):
        df, path = large
        pd_filtered = df[df["v"] > 60_000]
        pd_result = pd_filtered.sort_values("k").head(20)

        ds = DataStore.from_file(path)
        ds_filtered = ds[ds["v"] > 60_000]
        ds_result = ds_filtered.sort_values("k").head(20)

        assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
class TestTextFileParsedInChunks:
    """Text files larger than a parallel-parsing chunk (10 MB)."""

    N = 600_000

    @pytest.fixture
    def chunked(self, tmp_path, fmt):
        rng = np.random.default_rng(11)
        df = pd.DataFrame(
            {
                "g": rng.integers(0, 50, self.N),
                "v": np.arange(self.N),
                "pad": "abcdefghijklmnopqrstuvwxyz",
            }
        )
        path = str(tmp_path / f"chunked.{fmt}")
        _write(df, path, fmt)
        return df, path

    @pytest.mark.parametrize("func", ["first", "last"])
    def test_groupby_first_last(self, chunked, func):
        df, path = chunked
        pd_result = getattr(df.groupby("g")["v"], func)()

        ds_result = getattr(DataStore.from_file(path).groupby("g")["v"], func)()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter(self, chunked):
        df, path = chunked
        pd_result = df[df["g"] == 7]

        ds = DataStore.from_file(path)
        ds_result = ds[ds["g"] == 7]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSourceFormats:
    def test_compressed_csv(self, tmp_path):
        path = str(tmp_path / "sales.csv.gz")
        SALES.to_csv(path, index=False)
        pd_result = SALES[SALES["qty"] > 10].sort_values("price")

        ds = DataStore.from_file(path)
        ds_result = ds[ds["qty"] > 10].sort_values("price")

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.parametrize("ext", ["parquet", "csv"])
    def test_format_omitted(self, tmp_path, ext):
        path = str(tmp_path / f"sales.{ext}")
        _write(SALES, path, ext)
        pd_result = SALES[SALES["qty"] > 10].tail(3)

        ds = DataStore("file", path=path)
        ds_result = ds[ds["qty"] > 10].tail(3)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnNameCollisions:
    def test_column_named_like_label_column(self, tmp_path):
        df = pd.DataFrame({"__row_label__": [10, 20, 30, 40], "x": [1, 2, 3, 4]})
        path = str(tmp_path / "c.parquet")
        df.to_parquet(path)
        pd_result = df[df["x"] > 1]

        ds = DataStore.from_file(path)
        ds_result = ds[ds["x"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_named_row_number_read(self, tmp_path):
        df = pd.DataFrame({"_row_number": [100, 200, 300, 400], "x": [1, 2, 3, 4]})
        path = str(tmp_path / "c.parquet")
        df.to_parquet(path)

        ds_result = DataStore.from_file(path)

        assert_datastore_equals_pandas(ds_result, df)

    def test_column_named_row_number_csv_filter(self, tmp_path):
        df = pd.DataFrame({"_row_number": [100, 200, 300, 400], "x": [1, 2, 3, 4]})
        path = str(tmp_path / "c.csv")
        df.to_csv(path, index=False)
        pd_result = df[df["x"] > 1]

        ds = DataStore.from_file(path)
        ds_result = ds[ds["x"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_named_row_number_filter(self, tmp_path):
        # The column hides the Parquet reader's _row_number, so the file has
        # no row positions: its rows are labelled 0..n-1, values stay intact
        df = pd.DataFrame({"_row_number": [100, 200, 300, 400], "x": [1, 2, 3, 4]})
        path = str(tmp_path / "c.parquet")
        df.to_parquet(path)
        pd_result = df[df["x"] > 1]

        ds = DataStore.from_file(path)
        ds_result = ds[ds["x"] > 1]

        _assert_values_equal(ds_result, pd_result)


@pytest.fixture
def executed_sql():
    """SQL the executions log ("Executing SQL: ...")."""
    logger = logging.getLogger("datastore")
    level = logger.level
    sqls = []
    handler = logging.Handler()
    handler.emit = lambda record: sqls.append(record.getMessage())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield sqls
    logger.removeHandler(handler)
    logger.setLevel(level)


def _data_queries(messages):
    prefix = "  Executing SQL: "
    return [m[len(prefix):] for m in messages if m.startswith(prefix)]


class TestRowLabelCost:
    """Labels cost only what the operations need (no sort or serial read)."""

    @pytest.fixture
    def csv_path(self, tmp_path):
        path = str(tmp_path / "sales.csv")
        SALES.to_csv(path, index=False)
        return path

    def test_head_reads_no_positions(self, csv_path, executed_sql):
        pd_result = SALES.head(3)

        ds_result = DataStore.from_file(csv_path).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)
        queries = _data_queries(executed_sql)
        assert queries
        assert all("ORDER BY" not in sql and "_row_number" not in sql for sql in queries)

    def test_filter_head_does_not_sort(self, csv_path, executed_sql):
        pd_result = SALES[SALES["qty"] > 10].head(2)

        ds = DataStore.from_file(csv_path)
        ds_result = ds[ds["qty"] > 10].head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)
        queries = _data_queries(executed_sql)
        assert queries
        assert all("ORDER BY" not in sql for sql in queries)


class TestGroupFirstAppearance:
    """groupby(sort=False) orders the groups by their first row in the file."""

    N = 200_000

    @staticmethod
    def _spread_keys(n):
        # A new key every 100 rows: first appearances span the whole file
        keys = np.random.default_rng(12).permutation(n // 100)
        return pd.DataFrame({"g": keys[np.arange(n) // 100], "v": np.arange(n)})

    @pytest.mark.parametrize("fmt", ["parquet", "csv", "jsonl"])
    @pytest.mark.parametrize(
        "chain",
        [
            lambda d: d.groupby("g", sort=False).agg({"v": "sum"}),
            lambda d: d.head(150_000).groupby("g", sort=False)["v"].sum(),
        ],
        ids=["agg", "head_then_agg"],
    )
    def test_first_appearance_order(self, tmp_path, fmt, chain):
        df = self._spread_keys(self.N)
        path = str(tmp_path / f"spread.{fmt}")
        _write(df, path, fmt)
        pd_result = chain(df)

        # Rows read in parallel reach the query threads in varying order
        for _ in range(5):
            ds_result = chain(DataStore.from_file(path))
            assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.parametrize(
        "fmt, ordered_read",
        [("csv", "parallelize_output_from_storages=0"), ("parquet", "_row_number")],
    )
    @pytest.mark.parametrize(
        "chain",
        [
            lambda d: d.groupby("region", sort=False).agg({"qty": "sum"}),
            lambda d: d.head(6).groupby("region", sort=False)["qty"].sum(),
        ],
        ids=["agg", "head_then_agg"],
    )
    def test_positions_read_from_file(self, tmp_path, executed_sql, fmt, ordered_read, chain):
        path = str(tmp_path / f"sales.{fmt}")
        _write(SALES, path, fmt)
        pd_result = chain(SALES)

        ds_result = chain(DataStore.from_file(path))

        assert_datastore_equals_pandas(ds_result, pd_result)
        queries = _data_queries(executed_sql)
        assert queries
        assert all(ordered_read in sql for sql in queries)


@pytest.fixture
def connection_sql(monkeypatch):
    """SQL the connection runs, other than schema probes."""
    from datastore.connection import Connection

    sqls = []

    def capturing(method):
        def capture(self, sql, *args, **kwargs):
            if not sql.startswith("DESCRIBE"):
                sqls.append(sql)
            return method(self, sql, *args, **kwargs)

        return capture

    monkeypatch.setattr(Connection, "execute", capturing(Connection.execute))
    monkeypatch.setattr(Connection, "query_arrow", capturing(Connection.query_arrow))
    return sqls


class TestFileOrderOutsideLabelledQueries:
    """Every read of a file returns its rows in file order."""

    @pytest.fixture
    def ties(self, tmp_path):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"k": rng.integers(0, 50, 2_000), "t": rng.integers(0, 3, 2_000)})
        path = str(tmp_path / "ties.csv")
        df.to_csv(path, index=False)
        return df, path

    @pytest.mark.parametrize(
        "chain",
        [
            lambda d: d.sort_values("k", kind="stable"),
            lambda d: d.sort_values(["k", "t"], kind="stable"),
        ],
        ids=["one_key", "two_keys"],
    )
    def test_pandas_first_sort(self, ties, connection_sql, chain):
        # The sort runs in pandas on the file's rows: their order breaks ties
        df, path = ties
        pd_result = chain(df)

        ds_result = chain(DataStore.from_file(path))

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert connection_sql
        assert all("parallelize_output_from_storages=0" in sql for sql in connection_sql)

    def test_to_arrow_of_filter(self, ties, connection_sql):
        df, path = ties
        pd_result = df[df["t"] == 1].reset_index(drop=True)

        ds = DataStore.from_file(path)
        table = ds[ds["t"] == 1].to_arrow()

        pd.testing.assert_frame_equal(table.to_pandas(), pd_result, check_dtype=False)
        assert connection_sql
        assert all("parallelize_output_from_storages=0" in sql for sql in connection_sql)

    def test_union_all(self, ties):
        df, path = ties
        pd_result = pd.concat([df[df["t"] == 1], df[df["t"] == 2]], ignore_index=True)

        left = DataStore.from_file(path)
        right = DataStore.from_file(path)
        ds_result = left[left["t"] == 1].union(right[right["t"] == 2], all=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSourceLocations:
    @pytest.fixture
    def served(self, tmp_path):
        """An HTTP server for tmp_path; yields its base URL."""
        import functools
        import http.server
        import threading

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)
        )
        handler.log_message = lambda *args: None
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}"
        server.shutdown()

    def test_url_with_query_string(self, tmp_path, served):
        df = pd.DataFrame({"k": [5, 3, 4, 1, 2, 0]})
        df.to_parquet(str(tmp_path / "s.parquet"))
        pd_result = df[df["k"] < 3]

        ds = DataStore.from_url(f"{served}/s.parquet?token=abc", format="Parquet")
        ds_result = ds[ds["k"] < 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.parametrize(
        "location, has_positions",
        [
            ("https://host/data.parquet?token=abc&x=1", True),
            ("https://host/data.parquet#part", True),
            ("https://host/part-*.parquet", False),
            ("https://host/part-{1..3}.parquet", False),
            ("s3://bucket/dir/data.parquet?versionId=7", True),
            ("s3://bucket/dir/{a,b}.parquet", False),
            ("/data/part_?.parquet", False),
            ("/data/part_{1..9}.parquet", False),
            ("/data/part.parquet", True),
        ],
    )
    def test_glob_locations(self, location, has_positions):
        from datastore.table_functions import FileTableFunction, S3TableFunction, UrlTableFunction

        if location.startswith("s3://"):
            source = S3TableFunction(url=location, format="Parquet")
        elif "://" in location:
            source = UrlTableFunction(url=location, format="Parquet")
        else:
            source = FileTableFunction(path=location, format="Parquet")

        assert source.has_row_positions(["k"]) is has_positions
