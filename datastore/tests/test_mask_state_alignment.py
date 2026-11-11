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


def _masks_of_two_states(d):
    """``a`` taken before ``a`` is overwritten, ``b`` and ``c`` after."""
    if isinstance(d, pd.DataFrame):
        d = d.copy()
    a = d["a"] > 1
    d["a"] = d["a"] * 10
    return d, a, d["a"] > 25, d["b"] > 15


class TestCombiningMasksOfDifferentStates:
    """&, |, ^ and ~ of masks of different states, in any nesting, keep each
    mask's values."""

    @pytest.mark.parametrize(
        "combine",
        [
            lambda a, b, c: (a ^ b) ^ c,
            lambda a, b, c: c ^ (a ^ b),
            lambda a, b, c: ~(a & b),
            lambda a, b, c: ~(a | b),
            lambda a, b, c: ~~(a & b),
            lambda a, b, c: (a & b) | c,
            lambda a, b, c: c & (a | b),
            lambda a, b, c: ~(a ^ b) & c,
            lambda a, b, c: (a & b) == c,
            lambda a, b, c: ~a & b,
        ],
        ids=["xor_xor", "xor_reflected", "invert_and", "invert_or", "invert_twice",
             "and_or", "and_reflected", "invert_xor_and", "compare", "invert_first"],
    )
    def test_combine(self, source, combine):
        df = _frame()
        d, a, b, c = _masks_of_two_states(source(df))
        e, pa, pb, pc = _masks_of_two_states(df)
        assert_datastore_equals_pandas(d[combine(a, b, c)], e[combine(pa, pb, pc)])

    def test_combine_with_isin_of_other_state(self, source):
        df = _frame()
        d, a, b, _ = _masks_of_two_states(source(df))
        e, pa, pb, _ = _masks_of_two_states(df)
        assert_datastore_equals_pandas(
            d[~(a & b) | d["b"].isin([10, 20])], e[~(pa & pb) | e["b"].isin([10, 20])]
        )


class TestToSqlWithoutReachingTheSource:
    """to_sql() renders the pipeline when the source can't be reached."""

    @staticmethod
    def _unreachable():
        return [
            DataStore.from_file("/nonexistent/chdb-ds-test/missing.parquet"),
            DataStore.from_clickhouse(
                host="127.0.0.1:1", database="db", table="t", user="u", password="p"
            ),
        ]

    @pytest.mark.parametrize(
        "chain, rendered",
        [
            (lambda d: d.assign(c=d["a"] + 1), '("a"+1) AS "c"'),
            (lambda d: d.select("a", "b").assign(a=d["a"] + 1), 'SELECT ("a"+1) AS "a", "b"'),
            (lambda d: d.head(3)[d["a"] > 1], 'LIMIT 3) AS __subq1__ WHERE "a" > 1'),
            (lambda d: d.select("a", "b").select("b"), 'SELECT "b" FROM'),
            (lambda d: d.select("a").filter(d.a > 1), 'WHERE "a" > 1'),
        ],
        ids=["assign", "overwrite_after_projection", "head_then_filter", "select_twice", "builder"],
    )
    def test_renders(self, chain, rendered):
        for ds in self._unreachable():
            assert rendered in chain(ds).to_sql()

    def test_execution_still_needs_the_source(self):
        for ds in self._unreachable():
            with pytest.raises(Exception):
                ds.assign(c=ds["a"] + 1).to_pandas()


class TestDuplicateColumnsInOneSelect:
    """A column named twice in one selection is selected twice, as in
    df[['a', 'a']] and SQL."""

    def test_select_same_column_twice(self, source):
        df = _frame()
        d = source(df).select("a", "a")
        assert '"a", "a"' in d.to_sql()
        result = d.to_pandas()
        assert len(result.columns) == 2
        assert result.values.tolist() == df[["a", "a"]].values.tolist()

    def test_getitem_same_column_twice(self, source):
        df = _frame()
        result = source(df)[["a", "a"]].to_pandas()
        assert len(result.columns) == 2
        assert result.values.tolist() == df[["a", "a"]].values.tolist()

    def test_later_select_still_replaces(self, source):
        df = _frame()
        d = source(df).select("a", "b").select("b", "b")
        result = d.to_pandas()
        assert result.values.tolist() == df[["a", "b"]][["b", "b"]].values.tolist()


class TestMaskArgumentOfSeriesMethod:
    """A mask or column passed to a Series method stays a Series, also with
    one row, and a mask computed in pandas is applied by mask() as by
    where()."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.where(s.isin([1]), 0),
            lambda s: s.where(s.cumsum() > 0, 0),
            lambda s: s.mask(s.isin([1]), 9),
            lambda s: s.mask(s.cumsum() > 0, 9),
            lambda s: s.fillna(s * 2),
        ],
        ids=["where_isin", "where_cumsum", "mask_isin", "mask_cumsum", "fillna_series"],
    )
    @pytest.mark.parametrize("values", [[1], [1, 2, 3]], ids=["one_row", "three_rows"])
    def test_call(self, source, call, values):
        df = pd.DataFrame({"x": values})
        d = source(df)
        pd.testing.assert_series_equal(call(d["x"]).to_pandas(), call(df["x"]), check_names=False)
