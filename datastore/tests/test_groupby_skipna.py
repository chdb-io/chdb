"""
groupby()[col] reductions with skipna=False give NaN for a group holding a
missing value, as pandas does (#763). A column reduction without groupby
honors skipna=False too.
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.config import get_execution_engine, set_execution_engine, use_pandas
from datastore.pandas_col_compat import PANDAS_3_PLUS
from tests.test_utils import assert_frame_equal, assert_series_equal

groupby_skipna = pytest.mark.skipif(not PANDAS_3_PLUS, reason="pandas < 3 groupby reductions don't take skipna")


ISSUE_DF = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, np.nan, 2.0, 3.0]})

DF = pd.DataFrame(
    {
        "g": ["a", "b", "a", "c", "b", "a", None, "b"],
        "h": ["x", "x", "y", "x", "y", "x", "y", "y"],
        "v": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 4.0],
        "w": [1, 2, 3, 4, 5, 6, 7, 8],
        "s": ["p", "q", None, "r", "s", "t", "u", "v"],
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
    if isinstance(pd_result, pd.DataFrame):
        assert_frame_equal(ds_result.to_pandas(), pd_result)
    else:
        assert_series_equal(ds_result, pd_result)


@groupby_skipna
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("method, b", [("sum", 5.0), ("mean", 2.5), ("max", 3.0)])
def test_issue_chain(source, method, b, tmp_path):
    ds_result = getattr(_source(source, ISSUE_DF, tmp_path).groupby("g")["v"], method)(skipna=False)
    pd_result = getattr(ISSUE_DF.groupby("g")["v"], method)(skipna=False)
    assert pd.isna(pd_result["a"]) and pd_result["b"] == b
    _assert_matches(ds_result, pd_result)


CHAINS = [
    pytest.param(lambda d: d.groupby("g")["v"].sum(skipna=False), id="sum"),
    pytest.param(lambda d: d.groupby("g")["v"].mean(skipna=False), id="mean"),
    pytest.param(lambda d: d.groupby("g")["v"].max(skipna=False), id="max"),
    pytest.param(lambda d: d.groupby("g")["v"].min(skipna=False), id="min"),
    pytest.param(lambda d: d.groupby("g")["v"].std(skipna=False), id="std"),
    pytest.param(lambda d: d.groupby("g")["v"].var(skipna=False), id="var"),
    pytest.param(lambda d: d.groupby("g")["v"].median(skipna=False), id="median"),
    pytest.param(lambda d: d.groupby("g")["w"].sum(skipna=False), id="int-column"),
    pytest.param(lambda d: d.groupby("g")["s"].max(skipna=False), id="str-max"),
    pytest.param(lambda d: d.groupby("g")["s"].min(skipna=False), id="str-min"),
    pytest.param(lambda d: d.groupby("g", as_index=False)["v"].sum(skipna=False), id="keys-as-columns"),
    pytest.param(lambda d: d.groupby("g", dropna=False)["v"].max(skipna=False), id="keep-na-key"),
    pytest.param(lambda d: d.groupby(["g", "h"])["v"].mean(skipna=False), id="two-keys"),
    pytest.param(lambda d: d[d["w"] > 2].groupby("g")["v"].sum(skipna=False), id="filter-sum"),
    pytest.param(lambda d: d.groupby("g")["v"].sum(skipna=False).sort_values(), id="sum-sort-values"),
]


@groupby_skipna
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("chain", CHAINS)
def test_chain(source, chain, tmp_path):
    _assert_matches(chain(_source(source, DF, tmp_path)), chain(DF.copy()))


@pytest.fixture
def pandas_engine():
    engine = get_execution_engine()
    use_pandas()
    yield
    set_execution_engine(engine)


@groupby_skipna
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("method", ["sum", "mean", "max"])
def test_pandas_engine(source, method, tmp_path, pandas_engine):
    ds_result = getattr(_source(source, DF, tmp_path).groupby("g")["v"], method)(skipna=False)
    _assert_matches(ds_result, getattr(DF.groupby("g")["v"], method)(skipna=False))


@pytest.mark.parametrize("method", ["sum", "mean", "max", "min", "std", "var", "median"])
@pytest.mark.parametrize("skipna", [False, True])
def test_column_reduction(method, skipna):
    ds_result = getattr(DataStore(DF.copy())["v"], method)(skipna=skipna)
    pd_result = getattr(DF["v"], method)(skipna=skipna)
    assert pd.isna(pd_result) != skipna
    if skipna:
        assert ds_result == pytest.approx(pd_result)
    else:
        assert pd.isna(ds_result)
