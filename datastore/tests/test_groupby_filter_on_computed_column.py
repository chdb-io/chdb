"""
A filter on a computed column followed by groupby().agg().

When the assignment, the filter and the GROUP BY share one SQL layer, the
GROUP BY SELECT holds only keys and aggregates, so the WHERE must read the
computed column's expression, as the aggregates do, rather than its alias.
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


def _frame():
    return pd.DataFrame({"g": ["x", "y", "x", "y", "x"], "v": [1, 2, 3, 4, 5], "w": [5, 4, 3, 2, 1]})


@pytest.fixture(params=["file", "dataframe"])
def make_ds(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(_frame())
    path = str(tmp_path / "data.parquet")
    _frame().to_parquet(path)
    return lambda: DataStore.from_file(path)


def _setitem_filter(d):
    d["a"] = d["v"] * 2
    d = d[d["a"] > 2]
    return d.groupby("g").agg({"a": "sum"})


def _assign_filter(d):
    d = d.assign(a=d["v"] * 2)
    d = d[d["a"] > 2]
    return d.groupby("g").agg({"a": "sum"})


def _overwrite_then_filter(d):
    d["a"] = d["v"] * 2
    d["a"] = d["a"] + 1
    d = d[d["a"] > 4]
    return d.groupby("g").agg({"a": "sum"})


def _isin(d):
    d["a"] = d["v"] * 2
    d = d[d["a"].isin([4, 6, 10])]
    return d.groupby("g").agg({"a": "sum", "w": "max"})


def _between(d):
    d["a"] = d["v"] + d["w"] * 10
    d = d[d["a"].between(20, 45)]
    return d.groupby("g").agg({"a": "min"})


def _negated(d):
    d["a"] = d["v"] - d["w"]
    d = d[~(d["a"] > 0)]
    return d.groupby("g").agg({"v": "sum", "a": "sum"})


def _computed_key_and_filter(d):
    d["k"] = d["v"] % 2
    d["a"] = d["w"] * 3
    d = d[d["a"] < 12]
    return d.groupby("k").agg({"a": "sum"})


def _named_agg_as_columns(d):
    d["a"] = d["v"] * 2
    d = d[d["a"] >= 4]
    return d.groupby("g", as_index=False).agg(total=("a", "sum"), n=("v", "count"))


@pytest.mark.parametrize(
    "chain",
    [
        _setitem_filter,
        _assign_filter,
        _overwrite_then_filter,
        _isin,
        _between,
        _negated,
        _computed_key_and_filter,
        _named_agg_as_columns,
    ],
)
def test_matches_pandas(make_ds, chain):
    assert_datastore_equals_pandas(chain(make_ds()), chain(_frame()))
