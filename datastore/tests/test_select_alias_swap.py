"""
select() fields read the columns before the select, also when another
field's alias replaces the column they read (#768).
"""

import pandas as pd
import pytest

from datastore import DataStore, col
from tests.test_utils import assert_frame_equal, assert_series_equal


DF = pd.DataFrame(
    {
        "price": [3.0, 1.0, 2.0, 5.0],
        "qty": [10, 30, 20, 40],
        "k": ["a", "b", "a", "b"],
    }
)


def _source(kind, df, tmp_path):
    if kind == "dataframe":
        return DataStore(df.copy())
    path = str(tmp_path / "t.parquet")
    df.to_parquet(path)
    return DataStore.from_file(path)


SOURCES = ["dataframe", "parquet"]


def _assert_matches(ds_result, pd_result):
    result = ds_result.to_pandas()
    if isinstance(pd_result, pd.DataFrame):
        assert_frame_equal(result, pd_result)
    else:
        assert_series_equal(result, pd_result)


def _swap(df):
    return df[["qty", "price"]].set_axis(["price", "qty"], axis=1)


@pytest.mark.parametrize("source", SOURCES)
def test_issue_chain(source, tmp_path):
    df = pd.DataFrame({"price": [1.0, 2.0], "qty": [10, 20]})
    ds = _source(source, df, tmp_path)
    result = ds.select(col("qty").as_("price"), col("price").as_("qty"))
    _assert_matches(result, _swap(df))


SWAP = (col("qty").as_("price"), col("price").as_("qty"))

CHAINS = [
    pytest.param(
        lambda d: d.select("k", *SWAP),
        lambda d: pd.concat([d[["k"]], _swap(d)], axis=1),
        id="with-plain-column",
    ),
    pytest.param(
        lambda d: d.select(col("qty").as_("price"), (col("price") * 2).as_("double")),
        lambda d: pd.DataFrame({"price": d["qty"], "double": d["price"] * 2}),
        id="expression-reads-replaced-column",
    ),
    pytest.param(
        lambda d: d.select((col("qty") + 1).as_("price"), (col("price") * 2).as_("qty")),
        lambda d: pd.DataFrame({"price": d["qty"] + 1, "qty": d["price"] * 2}),
        id="swap-expressions",
    ),
    pytest.param(
        lambda d: d.select(col("qty").as_("price"), col("k").as_("qty"), col("price").as_("k")),
        lambda d: d[["qty", "k", "price"]].set_axis(["price", "qty", "k"], axis=1),
        id="rotate-three",
    ),
    pytest.param(
        lambda d: d.select(d["qty"].as_("price"), d["price"].as_("qty")),
        _swap,
        id="column-expr-fields",
    ),
    pytest.param(
        lambda d: d.select(*SWAP).select(*SWAP),
        lambda d: d[["price", "qty"]],
        id="swap-twice",
    ),
    pytest.param(
        lambda d: d.assign(t=col("price") + 1).select(col("t").as_("price"), col("price").as_("t")),
        lambda d: (lambda x: pd.DataFrame({"price": x["t"], "t": x["price"]}))(d.assign(t=d["price"] + 1)),
        id="assign-then-swap",
    ),
    pytest.param(
        lambda d: d.head(2).select(*SWAP),
        lambda d: _swap(d.head(2)),
        id="head-then-swap",
    ),
    pytest.param(
        lambda d: d.select("k", *SWAP).groupby("k")["price"].sum(),
        lambda d: pd.concat([d[["k"]], _swap(d)], axis=1).groupby("k")["price"].sum(),
        id="swap-then-groupby",
    ),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("ds_chain, pd_chain", CHAINS)
def test_chain(source, ds_chain, pd_chain, tmp_path):
    _assert_matches(ds_chain(_source(source, DF, tmp_path)), pd_chain(DF.copy()))


# A filtered or sorted file source comes back with a fresh RangeIndex, so the
# chains that move rows run on a DataFrame source.
ROW_CHAINS = [
    pytest.param(
        lambda d: d[d["price"] > 1.5].select(*SWAP),
        lambda d: _swap(d[d["price"] > 1.5]),
        id="filter-then-swap",
    ),
    pytest.param(
        lambda d: (lambda x: x[x["price"] > 15])(d.select(*SWAP)),
        lambda d: (lambda x: x[x["price"] > 15])(_swap(d)),
        id="swap-then-filter",
    ),
    pytest.param(
        lambda d: d.sort_values("price").select(*SWAP),
        lambda d: _swap(d.sort_values("price")),
        id="sort-then-swap",
    ),
    pytest.param(
        lambda d: d.select(*SWAP).sort_values("qty", ascending=False).head(2),
        lambda d: _swap(d).sort_values("qty", ascending=False).head(2),
        id="swap-then-sort-head",
    ),
]


@pytest.mark.parametrize("ds_chain, pd_chain", ROW_CHAINS)
def test_row_chain(ds_chain, pd_chain):
    _assert_matches(ds_chain(DataStore(DF.copy())), pd_chain(DF.copy()))


def test_columns():
    ds = DataStore(DF.copy()).select("k", *SWAP)
    assert list(ds.columns) == ["k", "price", "qty"]
