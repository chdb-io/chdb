"""
join()/merge() of two file sources run the left side's lazy ops before the
join, as pandas merges the frame they produce (#754).
"""

import os

import pandas as pd
import pytest

from datastore import DataStore
from datastore.exceptions import QueryError
from tests.test_utils import assert_datastore_equals_pandas


LEFT = pd.DataFrame({"k": [1, 2, 3, 4], "a": [10, 20, 30, 40], "x": [7, 8, 9, 0]})
RIGHT = pd.DataFrame({"k": [1, 1, 2, 2, 3, 4], "b": [1, 2, 3, 4, 5, 6]})
THIRD = pd.DataFrame({"k": [1, 2, 3], "c": [100, 200, 300]})


@pytest.fixture(params=["parquet", "csv"])
def files(tmp_path, request):
    paths = {}
    for name, df in [("l", LEFT), ("r", RIGHT), ("c", THIRD)]:
        path = os.path.join(tmp_path, f"{name}.{request.param}")
        if request.param == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        paths[name] = path
    return paths


def test_head_then_join(files):
    pd_result = LEFT.head(2).merge(RIGHT, on="k")
    ds_result = DataStore.from_file(files["l"]).head(2).join(
        DataStore.from_file(files["r"]), on="k"
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_column_selection_then_join(files):
    pd_result = LEFT[["k", "a"]].merge(RIGHT, on="k")
    ds_result = DataStore.from_file(files["l"])[["k", "a"]].join(
        DataStore.from_file(files["r"]), on="k"
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_head_then_merge(files):
    pd_result = LEFT.head(2).merge(RIGHT, on="k")
    ds_result = DataStore.from_file(files["l"]).head(2).merge(
        DataStore.from_file(files["r"]), on="k"
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_filter_then_left_join(files):
    pd_left = LEFT[LEFT["a"] > 15]
    pd_result = pd_left.merge(RIGHT, on="k", how="left")

    ds_left = DataStore.from_file(files["l"])
    ds_left = ds_left[ds_left["a"] > 15]
    ds_result = ds_left.join(DataStore.from_file(files["r"]), on="k", how="left")
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_assign_then_join(files):
    pd_result = LEFT.assign(z=LEFT["a"] * 2).head(3).merge(RIGHT, on="k")

    ds_left = DataStore.from_file(files["l"])
    ds_result = (
        ds_left.assign(z=ds_left["a"] * 2)
        .head(3)
        .join(DataStore.from_file(files["r"]), on="k")
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_head_then_join_on_left_on_right_on(files):
    cols = ["k", "a", "x", "b"]
    pd_result = LEFT.head(2).merge(RIGHT, left_on="k", right_on="k")[cols]
    ds_result = DataStore.from_file(files["l"]).head(2).join(
        DataStore.from_file(files["r"]), left_on="k", right_on="k"
    )[cols]
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_join_head_then_join(files):
    pd_result = LEFT.head(2).merge(RIGHT, on="k").head(3).merge(THIRD, on="k")
    ds_result = (
        DataStore.from_file(files["l"])
        .head(2)
        .join(DataStore.from_file(files["r"]), on="k")
        .head(3)
        .join(DataStore.from_file(files["c"]), on="k")
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_groupby_agg_then_join(files):
    # The aggregation runs in pandas, so the join is a pandas merge
    pd_result = LEFT.groupby("k").agg({"a": "sum"}).merge(RIGHT, on="k")
    ds_result = (
        DataStore.from_file(files["l"])
        .groupby("k")
        .agg({"a": "sum"})
        .join(DataStore.from_file(files["r"]), on="k")
    )
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_sort_then_join(files):
    # The sort runs in pandas, so the join is a pandas merge
    pd_result = LEFT.sort_values("a", ascending=False).merge(RIGHT, on="k")
    ds_result = (
        DataStore.from_file(files["l"])
        .sort_values("a", ascending=False)
        .join(DataStore.from_file(files["r"]), on="k")
    )
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_condition_join_needs_sql_left_side(files):
    left = DataStore.from_file(files["l"]).sort_values("a")
    right = DataStore.from_file(files["r"])
    with pytest.raises(QueryError):
        left.join(right, on=left.k == right.k)
