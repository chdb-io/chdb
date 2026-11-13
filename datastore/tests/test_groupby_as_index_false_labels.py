"""
groupby(as_index=False) labels its rows 0..n-1 by group, and a later sort,
filter or slice keeps those labels, as in pandas (#761).
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


ISSUE_DF = pd.DataFrame({"g": ["a", "b", "c", "a", "b"], "v": [5, 1, 3, 5, 1]})

DF = pd.DataFrame(
    {
        "g": ["a", "b", "c", "a", "b", "d", None, "c"],
        "h": ["x", "y", "x", "x", "y", "y", "x", "y"],
        "v": [5, 1, 4, 6, 2, 7, 8, 9],
    }
)


def _source(kind, df, tmp_path):
    if kind == "dataframe":
        return DataStore(df.copy())
    path = str(tmp_path / f"t.{kind}")
    if kind == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
    return DataStore.from_file(path)


SOURCES = ["dataframe", "parquet", "csv"]


def _assert_matches(ds_result, pd_result):
    assert_datastore_equals_pandas(ds_result, pd_result)
    assert list(ds_result.index) == list(pd_result.index)


def _agg(d):
    return d.groupby("g", as_index=False).agg(s=("v", "sum"))


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(
    "op, labels",
    [
        pytest.param(lambda a: a.sort_values("s"), [1, 2, 0], id="sort_values"),
        pytest.param(lambda a: a[a["s"] > 2], [0, 2], id="filter"),
    ],
)
def test_issue_chain(source, op, labels, tmp_path):
    ds_result = op(_agg(_source(source, ISSUE_DF, tmp_path)))
    pd_result = op(_agg(ISSUE_DF.copy()))
    assert list(pd_result.index) == labels
    _assert_matches(ds_result, pd_result)


AGGS = [
    pytest.param(_agg, id="named_agg"),
    pytest.param(lambda d: d.groupby("g", as_index=False).agg({"v": "sum"}), id="agg_dict"),
    pytest.param(lambda d: d[["g", "v"]].groupby("g", as_index=False).sum(), id="sum"),
    pytest.param(lambda d: d.groupby(["h", "g"], as_index=False).agg(s=("v", "sum")), id="two_keys"),
    pytest.param(
        lambda d: d.groupby("g", as_index=False, dropna=False).agg(s=("v", "sum")), id="dropna_false"
    ),
]


OPS = [
    pytest.param(lambda a, c: a.sort_values(c), id="sort"),
    pytest.param(lambda a, c: a.sort_values(c, ascending=False), id="sort_desc"),
    pytest.param(lambda a, c: a[a[c] > 3], id="filter"),
    pytest.param(lambda a, c: a.sort_values(c).head(2), id="sort_head"),
    pytest.param(lambda a, c: a[a[c] > 3].sort_values(c), id="filter_sort"),
    pytest.param(lambda a, c: a.sort_values(c).sort_values("g", ascending=False), id="sort_sort"),
    pytest.param(lambda a, c: a[1:], id="slice_from"),
    pytest.param(lambda a, c: a[1:3], id="slice"),
    pytest.param(lambda a, c: a.sort_values(c)[1:], id="sort_slice"),
    pytest.param(lambda a, c: a[a[c] > 3].assign(t=lambda x: x[c] + 1), id="filter_assign"),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("agg", AGGS)
def test_labels_after_agg(source, agg, op, tmp_path):
    pd_agg = agg(DF.copy())
    column = pd_agg.columns[-1]
    pd_result = op(pd_agg, column)
    ds_result = op(agg(_source(source, DF, tmp_path)), column)
    _assert_matches(ds_result, pd_result)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(
    "op",
    [
        pytest.param(lambda a: a.sort_values("s"), id="sort"),
        pytest.param(lambda a: a[a["s"] > 0], id="filter"),
        pytest.param(lambda a: a[1:], id="slice"),
        pytest.param(lambda a: a[a["s"] > 0].assign(t=lambda x: x["s"] + 1), id="filter_assign"),
    ],
)
def test_no_groups(source, op, tmp_path):
    """A step after an aggregation with no groups runs on that empty result,
    not on the source rows."""

    def chain(d):
        return op(_agg(d[d["v"] > 100]))

    pd_result = chain(DF.copy())
    ds_result = chain(_source(source, DF, tmp_path))
    assert len(pd_result) == 0
    _assert_matches(ds_result, pd_result)


@pytest.mark.parametrize("source", SOURCES)
def test_as_index_true_keeps_key_labels(source, tmp_path):
    def chain(d):
        a = d.groupby("g").agg(s=("v", "sum"))
        return a[a["s"] > 3].sort_values("s")

    pd_result = chain(DF.copy())
    ds_result = chain(_source(source, DF, tmp_path))
    _assert_matches(ds_result, pd_result)
