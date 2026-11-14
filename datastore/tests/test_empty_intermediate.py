"""
A step that returns no rows hands an empty frame, with its columns and
dtypes, to the next step: later steps run on it instead of reading the
source again. Every case runs on a DataFrame source and on a file source.
"""

import itertools

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas

_counter = itertools.count()


@pytest.fixture(params=["dataframe", "file"])
def source(request, tmp_path):
    """Build a DataStore over ``df`` from the parametrized kind of source."""

    def make(df):
        if request.param == "dataframe":
            return DataStore(df.copy())
        path = str(tmp_path / f"t{next(_counter)}.parquet")
        df.to_parquet(path)
        return DataStore.from_file(path)

    return make


def _frame():
    return pd.DataFrame(
        {"g": ["x", "y", "x", "y"], "a": [1, 2, 3, 4], "b": [40, 30, 20, 10]}
    )


def _empty_then_pandas_step(d):
    """No rows after a filter, then a step that runs in pandas (cumsum)."""
    d = d[d["a"] > 100]
    return d.assign(c=d["a"].cumsum())


class TestEmptyIntermediate:
    def test_mask_of_overwritten_column_then_select(self, source):
        # A mask taken before an overwrite is evaluated on its own state, in
        # pandas; the later projection runs on the empty result.
        def chain(d):
            m = d["a"] > 100
            if isinstance(d, pd.DataFrame):
                return d.assign(a=d["a"] * 0)[m][["b"]]
            return d.assign(a=d["a"] * 0).filter(m).select("b")

        df = _frame()
        assert_datastore_equals_pandas(chain(source(df)), chain(df))

    @pytest.mark.parametrize(
        "then",
        [
            lambda d: d,
            lambda d: d[["a", "c"]],
            lambda d: d.sort_values("b"),
            lambda d: d.assign(t=d["a"] * 2),
            lambda d: d[d["b"] > 15],
            lambda d: d.head(2),
        ],
        ids=["nothing", "select", "sort", "assign", "filter", "head"],
    )
    def test_then(self, source, then):
        df = _frame()
        assert_datastore_equals_pandas(
            then(_empty_then_pandas_step(source(df))), then(_empty_then_pandas_step(df))
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "count"])
    def test_then_groupby(self, source, agg):
        df = _frame()
        assert_datastore_equals_pandas(
            getattr(_empty_then_pandas_step(source(df)).groupby("g"), agg)(),
            getattr(_empty_then_pandas_step(df).groupby("g"), agg)(),
            check_index=True,
        )

    def test_then_merge(self, source):
        df = _frame()
        right = pd.DataFrame({"g": ["x", "y"], "label": ["first", "second"]})
        ds_right = DataStore(right)
        got = _empty_then_pandas_step(source(df)).merge(ds_right, on="g")
        expected = _empty_then_pandas_step(df).merge(right, on="g")
        assert_datastore_equals_pandas(got, expected)

    def test_empty_sql_step_then_pandas_then_sql(self, source):
        # The empty result of the first SQL step stays empty through a
        # pandas step and a SQL step on its result.
        df = _frame()

        def chain(d):
            d = d[["a", "b"]]
            d = d[d["a"] > 100]
            d = d.assign(c=d["a"].cumsum())
            return d.mask(d["a"] > 10, 0)

        assert_datastore_equals_pandas(chain(source(df)), chain(df))
