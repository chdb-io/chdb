"""
A boolean mask applies with the values it was taken with, matched to the
rows by index label, as pandas applies a boolean Series; to_sql() shows the
query that runs. Every case runs on a DataFrame source and on a file source.
"""

import itertools
import warnings

import chdb
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas

_counter = itertools.count()

NEEDS_FILE_ROW_LABELS = pytest.mark.xfail(
    strict=True,
    reason="#735: a filter on a file source relabels the rows 0..n-1, so a "
    "mask of the unfiltered frame is matched to the wrong rows",
)


def _write(df, tmp_path):
    path = str(tmp_path / f"t{next(_counter)}.parquet")
    df.to_parquet(path)
    return path


@pytest.fixture(params=["dataframe", "file"])
def source(request, tmp_path):
    """Build a DataStore over ``df`` from the parametrized kind of source."""

    def make(df):
        if request.param == "dataframe":
            return DataStore(df.copy())
        return DataStore.from_file(_write(df, tmp_path))

    make.kind = request.param
    return make


def _frame():
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [40, 30, 20, 10]})


def _overwrite(d):
    """A mask on ``a``, then ``a`` overwritten in place."""
    if isinstance(d, pd.DataFrame):
        d = d.copy()
    m = d["a"] > 2
    d["a"] = d["a"] * 0
    return d, m


class TestMaskCombinedWithMaskOfNewValues:
    """A mask taken before an overwrite combined with a mask on the new values."""

    @pytest.mark.parametrize("op", ["&", "|", "^"])
    def test_with_overwritten_copy(self, source, op):
        def chain(d):
            m = d["a"] > 1
            d2 = d.assign(a=d["a"] * 100)
            new = d2["a"] > 300
            return d2[{"&": m & new, "|": m | new, "^": m ^ new}[op]]

        df = _frame()
        assert_datastore_equals_pandas(chain(source(df)), chain(df))

    def test_in_place(self, source):
        def chain(d):
            if isinstance(d, pd.DataFrame):
                d = d.copy()
            m = d["a"] > 1
            d["a"] = d["a"] * 100
            return d[m & (d["a"] > 300)]

        df = _frame()
        assert_datastore_equals_pandas(chain(source(df)), chain(df))


class TestMaskOfValuesOverwrittenSince:
    """#739: the mask keeps the values it was taken with, whatever follows."""

    @pytest.mark.parametrize(
        "then",
        [
            lambda d, m: d[m],
            lambda d, m: d.sort_values("b")[m],
            lambda d, m: d.head(3)[m],
            lambda d, m: d.loc[m],
            lambda d, m: d.loc[m, ["b"]],
            lambda d, m: d.fillna(0)[m],
            lambda d, m: d.where(m, -1),
            lambda d, m: d.mask(m, -1),
            lambda d, m: d[["b"]][m],
        ],
        ids=["getitem", "sort", "head", "loc", "loc_columns", "fillna", "where", "mask", "projection"],
    )
    def test_then(self, source, then):
        df = _frame()
        assert_datastore_equals_pandas(then(*_overwrite(source(df))), then(*_overwrite(df)))

    def test_then_filter(self, source, request):
        if source.kind == "file":
            request.applymarker(NEEDS_FILE_ROW_LABELS)
        df = _frame()

        def chain(d):
            d, m = _overwrite(d)
            return d[d["b"] < 35][m]

        assert_datastore_equals_pandas(chain(source(df)), chain(df))

    def test_isin_mask(self, source):
        def chain(d):
            if isinstance(d, pd.DataFrame):
                d = d.copy()
            m = d["a"].isin([3, 4])
            d["a"] = d["a"] * 0
            return d.sort_values("b")[m]

        df = _frame()
        assert_datastore_equals_pandas(chain(source(df)), chain(df))

    def test_mask_of_other_rows_is_unalignable(self, source):
        df = _frame()
        with pytest.raises(pd.errors.IndexingError):
            df[df.head(2)["a"] > 0]
        d = source(df)
        with pytest.raises(pd.errors.IndexingError):
            d[d.head(2)["a"] > 0].to_pandas()


class TestMaskReadOnFrame:
    """Masks whose values the frame still holds stay a filter of its rows."""

    def test_after_projection_and_filter(self, source):
        df = _frame()
        d = source(df)
        assert_datastore_equals_pandas(
            d[["b"]][d["b"] > 15][d["a"] > 1], df[["b"]][df["b"] > 15][df["a"] > 1]
        )

    def test_name_of_a_column_added_later(self, source):
        # The mask names a column its own state doesn't have: it is the
        # frame's column (chdb-ds builder style).
        df = _frame()
        d = source(df)
        got = d.add_prefix("x_").filter(d.x_a > 2)
        expected = df.add_prefix("x_")
        assert_datastore_equals_pandas(got, expected[expected["x_a"] > 2])


class TestToSqlRunsLikeExecution:
    """The SQL to_sql() shows returns what execution returns."""

    @staticmethod
    def _assert_sql_runs_like(d):
        sql_rows = chdb.query(d.to_sql(), "DataFrame")
        executed = d.to_pandas()
        assert list(sql_rows.columns) == list(executed.columns)
        assert sql_rows.values.tolist() == executed.values.tolist()

    @pytest.mark.parametrize(
        "chain",
        [
            lambda d: d.head(3)[d["a"] > 1],
            lambda d: d.select("a", "b").select("b"),
            lambda d: d.limit(3).offset(1),
            lambda d: d.offset(1).limit(2),
            lambda d: d.select("a", "b").filter(d.a > 1).sort("b").limit(2),
        ],
        ids=["head_then_filter", "select_twice", "limit_then_offset", "offset_then_limit", "builder_chain"],
    )
    def test_file_source(self, tmp_path, chain):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._assert_sql_runs_like(chain(DataStore.from_file(_write(_frame(), tmp_path))))

    def test_head_then_filter_matches_pandas(self, source):
        df = _frame()
        d = source(df)
        assert_datastore_equals_pandas(d.head(3)[d["a"] > 1], df.head(3)[df["a"] > 1])


class TestMaskOfColumnsDroppedSince:
    """A mask reading a column the frame dropped since keeps its values."""

    @staticmethod
    def _dropped(d, how):
        if isinstance(d, pd.DataFrame):
            d = d.copy()
        m = d["a"] > 2
        if how == "del":
            del d["a"]
        elif how == "drop":
            d = d.drop(columns=["a"])
        else:
            d = d[["b"]]
        return d, m

    @pytest.mark.parametrize("how", ["del", "drop", "projection"])
    @pytest.mark.parametrize(
        "then",
        [
            lambda d, m: d.where(m, 0),
            lambda d, m: d.mask(m, 0),
            lambda d, m: d[m],
            lambda d, m: d.loc[m],
            lambda d, m: d.loc[m, ["b"]],
            lambda d, m: d.sort_values("b")[m],
            lambda d, m: d.head(3)[m],
        ],
        ids=["where", "mask", "getitem", "loc", "loc_columns", "sort", "head"],
    )
    def test_then(self, source, how, then):
        df = _frame()
        assert_datastore_equals_pandas(
            then(*self._dropped(source(df), how)), then(*self._dropped(df, how))
        )

    def test_filter_reading_projected_out_column_stays_sql(self, tmp_path):
        # A filter in the projection's SQL layer reads the projection's input
        ds = DataStore.from_file(_write(_frame(), tmp_path))
        got = ds.select("b").filter(ds.a > 2)
        assert 'WHERE "a" > 2' in got.to_sql()
        df = _frame()
        assert_datastore_equals_pandas(got, df[["b"]][df["a"] > 2])


class TestNullableBooleanMask:
    """A mask with NA selects no row where it is NA, as in pandas."""

    @staticmethod
    def _frame():
        return pd.DataFrame(
            {"a": pd.array([1, None, 3, 4], dtype="Int64"), "b": [40, 30, 20, 10]}
        )

    @pytest.mark.parametrize("overwrite", [False, True], ids=["same_state", "overwritten_since"])
    @pytest.mark.parametrize(
        "then",
        [
            lambda d, m: d[m],
            lambda d, m: d.loc[m],
            lambda d, m: d.where(m, 0),
            lambda d, m: d.mask(m, 0),
            lambda d, m: d.sort_values("b")[m],
        ],
        ids=["getitem", "loc", "where", "mask", "sort"],
    )
    def test_then(self, source, overwrite, then):
        def chain(d):
            if isinstance(d, pd.DataFrame):
                d = d.copy()
            m = d["a"] > 2
            if overwrite:
                d["a"] = d["a"] * 0
            return then(d, m)

        df = self._frame()
        assert_datastore_equals_pandas(chain(source(df)), chain(df))


class TestMaskOfColumnProjectedOutEarlier:
    """A filter reads the columns entering its SQL layer, not only those of
    the nearest select()."""

    @staticmethod
    def _chain(d):
        if isinstance(d, pd.DataFrame):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e = d[["b"]].iloc[:3].iloc[1:][["b"]]
                return e[d["a"] > 2]
        return d.select("b").limit(3).offset(1).select("b").filter(d.a > 2)

    def test_mask_of_column_dropped_by_earlier_projection(self, source, request):
        if source.kind == "file":
            request.applymarker(NEEDS_FILE_ROW_LABELS)
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [60, 50, 40, 30, 20, 10]})
        assert_datastore_equals_pandas(self._chain(source(df)), self._chain(df))

    def test_no_sql_filter_on_a_dropped_column(self):
        # Even with the source's schema unknown, the projections tell that
        # the filter's layer reads only b.
        ds = DataStore(table="t")
        assert '"a" > 2' not in self._chain(ds).to_sql()


class TestExecuteReturnsWhatToPandasReturns:
    """execute() returns the pipeline's result, also when it isn't one query."""

    @staticmethod
    def _rows(result):
        return [list(row) for row in result.rows]

    @pytest.mark.parametrize(
        "chain",
        [
            lambda d: d.select("a")[::2].limit(2),
            lambda d: d.sql("a > 2"),
            lambda d: d.sql("SELECT 1 AS one"),
            lambda d: d.select("a", "b").filter(d.a > 1).sort("b").limit(2),
            lambda d: d,
        ],
        ids=["pandas_step_between_sql_steps", "sql_filter", "sql_without_from", "one_query", "source"],
    )
    def test_execute(self, source, chain):
        d = chain(source(_frame()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert self._rows(d.execute()) == d.to_pandas().values.tolist()

    def test_pandas_step_between_sql_steps_matches_pandas(self, source):
        df = _frame()
        d = source(df).select("a")[::2].limit(2)
        assert self._rows(d.execute()) == df[["a"]].iloc[::2].head(2).values.tolist()

    def test_to_sql_of_pipeline_continuing_in_pandas_is_its_first_step(self, tmp_path):
        ds = DataStore.from_file(_write(_frame(), tmp_path))
        first_step = ds.select("a").to_sql(execution_format=True)
        assert ds.select("a")[::2].limit(2).to_sql(execution_format=True) == first_step
