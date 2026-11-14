"""
groupby()[col].agg('<reduction>') returns a Series named after the column,
as groupby()[col].<reduction>() and pandas do (#764).
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal


ISSUE_DF = pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3]})

DF = pd.DataFrame(
    {
        "g": ["b", "a", "b", None, "a", "c", "a"],
        "h": ["x", "x", "y", "x", "y", "x", "x"],
        "v": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0],
        "w": [3, 1, 4, 1, 5, 9, 2],
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
    assert type(result) is type(pd_result)
    if isinstance(pd_result, pd.DataFrame):
        assert_frame_equal(result, pd_result)
    else:
        assert_series_equal(result, pd_result)


@pytest.mark.parametrize("source", SOURCES)
def test_issue_chain(source, tmp_path):
    ds_result = _source(source, ISSUE_DF, tmp_path).groupby("g")["v"].agg("sum")
    pd_result = ISSUE_DF.groupby("g")["v"].agg("sum")
    _assert_matches(ds_result, pd_result)


FUNCS = ["sum", "mean", "min", "max", "count", "std", "var", "median", "first", "last", "nunique", "quantile"]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("func", FUNCS)
def test_reduction(source, func, tmp_path):
    ds_result = _source(source, DF, tmp_path).groupby("g")["v"].agg(func)
    _assert_matches(ds_result, DF.groupby("g")["v"].agg(func))


CHAINS = [
    pytest.param(lambda d: d.groupby("g")["v"].aggregate("sum"), id="aggregate"),
    pytest.param(lambda d: d.groupby("g")["w"].agg("sum"), id="int-column"),
    pytest.param(lambda d: d.groupby("g").w.agg("mean"), id="attribute-access"),
    pytest.param(lambda d: d.groupby("g", as_index=False)["v"].agg("sum"), id="keys-as-columns"),
    pytest.param(lambda d: d.groupby("g", dropna=False)["v"].agg("count"), id="keep-na-key"),
    pytest.param(lambda d: d.groupby(["g", "h"])["w"].agg("sum"), id="two-keys"),
    pytest.param(lambda d: d[d["w"] > 1].groupby("g")["w"].agg("sum"), id="filter-sum"),
    pytest.param(lambda d: d.groupby("g")["w"].agg("sum").sort_values(), id="sum-sort-values"),
    pytest.param(lambda d: d.groupby("g")["w"].agg("sum").reset_index(), id="sum-reset-index"),
    pytest.param(lambda d: d.groupby("g")["w"].agg(["sum", "mean"]), id="list-stays-frame"),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("chain", CHAINS)
def test_chain(source, chain, tmp_path):
    _assert_matches(chain(_source(source, DF, tmp_path)), chain(DF.copy()))
