"""
merge(), union() and join() with pandas DataFrame operands stay lazy and use
the DataFrame's data at call time, as pandas does (#773).
"""

import os

import pandas as pd
import pytest

import datastore
import datastore.core as core
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


LEFT = pd.DataFrame(
    {"k": [3, 1, 2, 2, 5], "v": [10, 20, 30, 40, 50], "x": ["a", "b", "c", "d", "e"]}
)
RIGHT = pd.DataFrame({"k": [1, 2, 3, 4], "v": [1.5, 2.5, 3.5, 4.5], "y": [7, 8, 9, 10]})


@pytest.fixture
def execute_calls(monkeypatch):
    calls = []
    original = core.DataStore._execute

    def counting_execute(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(core.DataStore, "_execute", counting_execute)
    return calls


@pytest.fixture
def files(tmp_path):
    paths = {}
    for name, df in [("l", LEFT), ("r", RIGHT)]:
        path = os.path.join(tmp_path, f"{name}.parquet")
        df.to_parquet(path, index=False)
        paths[name] = path
    return paths


def test_module_merge_dataframe_left_is_lazy(files, execute_calls):
    ds_result = datastore.merge(LEFT, DataStore.from_file(files["r"]), on="k")
    assert execute_calls == []
    assert_datastore_equals_pandas(ds_result, pd.merge(LEFT, RIGHT, on="k"))


def test_module_merge_two_files_is_lazy(files, execute_calls):
    ds_result = datastore.merge(
        DataStore.from_file(files["l"]),
        DataStore.from_file(files["r"]),
        on="k",
        how="left",
    )
    assert execute_calls == []
    assert_datastore_equals_pandas(ds_result, pd.merge(LEFT, RIGHT, on="k", how="left"))


def test_merge_dataframe_right_is_lazy(files, execute_calls):
    ds_result = DataStore.from_file(files["l"]).merge(
        RIGHT, on="k", suffixes=("_l", "_r")
    )
    assert execute_calls == []
    assert_datastore_equals_pandas(
        ds_result, LEFT.merge(RIGHT, on="k", suffixes=("_l", "_r"))
    )


def test_merge_reads_dataframe_at_call_time(files):
    right = RIGHT.copy()
    ds_result = DataStore.from_file(files["l"]).merge(right, on="k")
    pd_result = LEFT.merge(right, on="k")
    right.loc[0, "y"] = 999
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_module_merge_reads_dataframe_at_call_time(files):
    left = LEFT.copy()
    ds_result = datastore.merge(left, DataStore.from_file(files["r"]), on="k")
    pd_result = pd.merge(left, RIGHT, on="k")
    left.loc[0, "v"] = 999
    assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("all", [False, True])
def test_union_reads_dataframe_at_call_time(all):
    other = pd.DataFrame({"k": [7, 1], "v": [9, 20], "x": ["z", "b"]})
    ds_result = DataStore(LEFT).union(other, all=all)
    pd_result = pd.concat([LEFT, other], ignore_index=True)
    if not all:
        pd_result = pd_result.drop_duplicates()
    other.loc[0, "v"] = 999
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_join_dataframe():
    right = RIGHT.copy()
    ds_result = DataStore(LEFT).join(right, on="k")
    pd_result = LEFT.merge(right, on="k")
    right.loc[0, "y"] = 999
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_join_dataframe_left_on_right_on(files):
    right = RIGHT.rename(columns={"k": "key"})
    ds_result = DataStore.from_file(files["l"]).join(
        right, left_on="k", right_on="key", how="left"
    )
    pd_result = LEFT.merge(right, left_on="k", right_on="key", how="left")
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_assign_overwrite_after_merge(files):
    pd_merged = LEFT.merge(RIGHT, on="k")
    pd_result = pd_merged.assign(x=pd_merged["x"] + "!")
    ds_merged = DataStore.from_file(files["l"]).merge(RIGHT, on="k")
    ds_result = ds_merged.assign(x=ds_merged["x"] + "!")
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_select_then_assign_after_merge():
    pd_result = LEFT.merge(RIGHT, on="k")[["k", "y"]].assign(y=lambda t: t["y"] * 3)
    ds_result = (
        DataStore.from_df(LEFT)
        .merge(RIGHT, on="k")[["k", "y"]]
        .assign(y=lambda t: t["y"] * 3)
    )
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_select_then_assign_after_join(files):
    pd_result = LEFT.merge(RIGHT, on="k")[["k", "y"]].assign(y=lambda t: t["y"] * 3)
    ds_result = (
        DataStore.from_df(LEFT)
        .join(DataStore.from_file(files["r"]), on="k")[["k", "y"]]
        .assign(y=lambda t: t["y"] * 3)
    )
    assert_datastore_equals_pandas(ds_result, pd_result)

