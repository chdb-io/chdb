"""
groupby() on a categorical key of an in-memory frame orders the groups by
category and keeps the categorical dtype of the keys, as pandas does (#762).
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal


def _issue_df():
    cat = pd.Categorical(["lo", "hi", "mid", "lo", "hi"], categories=["lo", "mid", "hi"], ordered=True)
    return pd.DataFrame({"g": cat, "v": [1, 2, 3, 4, 5]})


def _df(ordered=True):
    # "none" is never observed; one key is missing.
    g = pd.Categorical(
        ["lo", "hi", None, "lo", "hi", "mid"], categories=["lo", "mid", "hi", "none"], ordered=ordered
    )
    return pd.DataFrame(
        {
            "g": g,
            "h": list("babbab"),
            "v": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "w": [1, 2, 3, 4, 5, 6],
        }
    )


def test_issue_chain():
    ds_result = DataStore(_issue_df()).groupby("g", observed=True)["v"].sum()
    pd_result = _issue_df().groupby("g", observed=True)["v"].sum()
    assert list(pd_result.index) == ["lo", "mid", "hi"]
    assert_series_equal(ds_result, pd_result)


SERIES_CHAINS = [
    pytest.param(lambda d: d.groupby("g", observed=True)["v"].sum(), id="sum"),
    pytest.param(lambda d: d.groupby("g", observed=False)["v"].sum(), id="sum-unobserved"),
    pytest.param(lambda d: d.groupby("g", observed=True, sort=False)["v"].mean(), id="mean-unsorted"),
    pytest.param(lambda d: d.groupby("g", observed=True, dropna=False)["w"].sum(), id="sum-keep-na"),
    pytest.param(lambda d: d.groupby("g", observed=True)["w"].std(ddof=0), id="std-ddof0"),
    pytest.param(lambda d: d.groupby("g", observed=True)["v"].first(), id="first"),
    pytest.param(lambda d: d.groupby("g", observed=True)["h"].nunique(), id="nunique"),
    pytest.param(lambda d: d.groupby(["h", "g"], observed=True)["w"].sum(), id="two-keys"),
    pytest.param(lambda d: d.groupby("g", observed=True)["w"].sum().sort_values(), id="sum-sort-values"),
    pytest.param(lambda d: d.groupby("g", observed=True).size(), id="size"),
    pytest.param(lambda d: d.groupby("g", observed=False).size(), id="size-unobserved"),
]


@pytest.mark.parametrize("ordered", [True, False], ids=["ordered", "unordered"])
@pytest.mark.parametrize("chain", SERIES_CHAINS)
def test_series_result(chain, ordered):
    assert_series_equal(chain(DataStore(_df(ordered))), chain(_df(ordered)))


FRAME_CHAINS = [
    pytest.param(lambda d: d.groupby("g", observed=True)[["v", "w"]].sum(), id="sum"),
    pytest.param(lambda d: d.groupby("g", observed=True).agg({"w": "sum", "v": "max"}), id="agg-dict"),
    pytest.param(lambda d: d.groupby("g", observed=True).agg(total=("w", "sum")), id="named-agg"),
    pytest.param(lambda d: d.groupby("g", observed=False).agg({"w": "sum"}), id="agg-unobserved"),
    pytest.param(lambda d: d.groupby("g", observed=True, sort=False).agg({"w": "sum"}), id="agg-unsorted"),
    pytest.param(lambda d: d.groupby("g", observed=True, as_index=False).agg({"w": "sum"}), id="agg-keys-as-columns"),
    pytest.param(lambda d: d.groupby("g", observed=True, as_index=False)["w"].count(), id="count-keys-as-columns"),
    pytest.param(lambda d: d[d["w"] > 1].groupby("g", observed=True).agg({"w": "sum"}), id="filter-agg"),
    pytest.param(lambda d: d.groupby("g", observed=True).agg({"w": "sum"}).sort_values("w"), id="agg-sort-values"),
    pytest.param(lambda d: d.groupby(["g", "h"], observed=True).agg({"w": "sum"}), id="two-keys"),
]


@pytest.mark.parametrize("ordered", [True, False], ids=["ordered", "unordered"])
@pytest.mark.parametrize("chain", FRAME_CHAINS)
def test_frame_result(chain, ordered):
    assert_frame_equal(chain(DataStore(_df(ordered))).to_pandas(), chain(_df(ordered)))


@pytest.mark.parametrize("source", ["dataframe", "parquet"])
def test_groupby_on_cut_bins(source, tmp_path):
    """Bins made by pd.cut inside the chain are categorical whatever the source."""
    df = _df()[["h", "w"]]
    if source == "dataframe":
        ds = DataStore(df.copy())
    else:
        path = str(tmp_path / "t.parquet")
        df.to_parquet(path)
        ds = DataStore.from_file(path)

    def chain(d):
        d = d.assign(b=lambda x: pd.cut(x["w"], bins=[0, 2, 4, 6], labels=["low", "mid", "high"]))
        return d.groupby("b", observed=True)

    assert_series_equal(chain(ds)["w"].sum(), chain(df)["w"].sum())
    assert_frame_equal(chain(ds).agg({"w": "sum"}).to_pandas(), chain(df).agg({"w": "sum"}))
