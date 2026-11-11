"""
A boolean mask applies with the values it was taken with, matched to the
rows by index label, as pandas applies a boolean Series. Every case runs on
a DataFrame source and on a file source.
"""

import itertools

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
