"""
Joins, merges and unions take their other operand as it is at call time,
whichever path runs them: a DataStore operand is frozen and a pandas
DataFrame operand is copied, so later in-place changes don't reach the
result (pandas reads the operand when called). An anti merge run lazily in
pandas reports its result columns to the ops after it on pandas 2 too.
"""

import pandas as pd
import pytest

import datastore
from datastore import DataStore
from datastore.lazy_ops import LazyJoin
from datastore.pandas_compat import _pandas_merge
from tests.test_utils import assert_datastore_equals_pandas

LEFT = pd.DataFrame({"id": [1, 2, 3, 4], "a": [10, 20, 30, 40]})
RIGHT = pd.DataFrame({"id": [2, 3, 5], "b": [200, 300, 500], "a": [7, 8, 9]})


@pytest.fixture(params=["dataframe", "file"])
def make(request, tmp_path):
    def make(df, name):
        if request.param == "dataframe":
            return DataStore(df.copy())
        path = str(tmp_path / f"{name}.parquet")
        df.to_parquet(path)
        return DataStore.from_file(path)

    return make


def test_merge_right_changed_after(make):
    right = make(RIGHT, "r")
    right = right[right["b"] > 250]
    # The shared non-key column makes merge() run in pandas
    ds_result = make(LEFT, "l").merge(right, on="id", how="left", suffixes=("", "_r"))
    right["b"] = right["b"] * 0

    filtered = RIGHT[RIGHT["b"] > 250]
    pd_result = LEFT.merge(filtered, on="id", how="left", suffixes=("", "_r"))
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_join_dataframe_changed_after(make):
    right = RIGHT[["id", "b"]].copy()
    ds_result = make(LEFT, "l").join(right, on="id", how="left")
    right.loc[:, "b"] = -1

    pd_result = LEFT.merge(RIGHT[["id", "b"]], on="id", how="left")
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_union_dataframe_changed_after(make):
    other = LEFT.copy()
    ds_result = make(LEFT, "l").union(other, all=True)
    other.loc[:, "a"] = -1

    pd_result = pd.concat([LEFT, LEFT], ignore_index=True)
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_module_merge_dataframe_changed_after(make):
    left = LEFT.copy()
    ds_result = datastore.merge(left, make(RIGHT, "r"), on="id", how="left")
    left.loc[:, "a"] = -1

    pd_result = LEFT.merge(RIGHT, on="id", how="left")
    assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("how", ["left_anti", "right_anti"])
def test_anti_merge_then_column_ops(make, how):
    ds = make(LEFT, "l").merge(RIGHT, on="id", how=how, suffixes=("_l", "_r"))
    ds_result = ds[ds["id"] > 0][["id", "a_l", "a_r"]]

    merged = _pandas_merge(LEFT, RIGHT, on="id", how=how, suffixes=("_l", "_r"))
    pd_result = merged[merged["id"] > 0][["id", "a_l", "a_r"]]
    assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("how", ["left_anti", "right_anti"])
def test_anti_merge_predicts_its_columns(how):
    # The planner reads these without executing the merge
    join = LazyJoin(RIGHT, on="id", how=how, suffixes=("_l", "_r"))
    merged = _pandas_merge(LEFT, RIGHT, on="id", how=how, suffixes=("_l", "_r"))
    assert join.transform_columns(list(LEFT.columns)) == list(merged.columns)
