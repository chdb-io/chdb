"""
merge() of two file sources gives pandas' column layout: left_on/right_on
naming the same columns keep one column per key, and columns added on the left
come before the right side's columns (#756).
"""

import os

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


L = pd.DataFrame({"k": [1, 2, 3, 4], "a": [10.5, 20.5, 30.5, 40.5]})
R = pd.DataFrame({"k": [3, 2, 1, 5], "b": [300.5, 200.5, 100.5, 500.5]})
L2 = pd.DataFrame({"k": [1, 2, 3, 4], "j": [1, 1, 2, 2], "a": [10, 20, 30, 40]})
R2 = pd.DataFrame({"k": [3, 2, 1, 5], "j": [2, 1, 1, 2], "b": [300, 200, 100, 500]})
R3 = pd.DataFrame({"kk": [3, 2, 1, 5], "jj": [2, 1, 1, 2], "b": [300, 200, 100, 500]})
R4 = pd.DataFrame({"k": [3, 2, 1, 5], "id": [1, 2, 3, 9], "b": [300, 200, 100, 500]})


@pytest.fixture(params=["parquet", "csv"])
def files(tmp_path, request):
    paths = {}
    for name, df in [("l", L), ("r", R), ("l2", L2), ("r2", R2), ("r3", R3), ("r4", R4)]:
        path = os.path.join(tmp_path, f"{name}.{request.param}")
        if request.param == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        paths[name] = path
    return paths


def test_assign_then_merge(files):
    pd_result = L.assign(c=L["a"] * 2).merge(R, on="k")

    ds_left = DataStore.from_file(files["l"])
    ds_result = ds_left.assign(c=ds_left["a"] * 2).merge(DataStore.from_file(files["r"]), on="k")
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_merge_left_on_right_on_same_name(files, how):
    pd_result = L.merge(R, left_on="k", right_on="k", how=how)
    ds_result = DataStore.from_file(files["l"]).merge(
        DataStore.from_file(files["r"]), left_on="k", right_on="k", how=how
    )
    assert 'USING ("k")' in ds_result.to_sql()
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_merge_left_on_right_on_same_names(files):
    pd_result = L2.merge(R2, left_on=["k", "j"], right_on=["k", "j"])
    ds_result = DataStore.from_file(files["l2"]).merge(
        DataStore.from_file(files["r2"]), left_on=["k", "j"], right_on=["k", "j"]
    )
    assert 'USING ("k", "j")' in ds_result.to_sql()
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_merge_left_on_right_on_lists(files):
    pd_result = L2.merge(R3, left_on=["k", "j"], right_on=["kk", "jj"])
    ds_result = DataStore.from_file(files["l2"]).merge(
        DataStore.from_file(files["r3"]), left_on=["k", "j"], right_on=["kk", "jj"]
    )
    assert 'ON ("l2"."k" = "r3"."kk" AND "l2"."j" = "r3"."jj")' in ds_result.to_sql()
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_merge_left_on_right_on_partly_same_names(files):
    pd_result = L2.merge(R3.rename(columns={"kk": "k"}), left_on=["k", "j"], right_on=["k", "jj"])
    ds_right = DataStore.from_file(files["r3"]).rename(columns={"kk": "k"})
    ds_result = DataStore.from_file(files["l2"]).merge(ds_right, left_on=["k", "j"], right_on=["k", "jj"])
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


def test_merge_left_key_name_is_right_column(files):
    # pandas suffixes k on both sides, as the right k is not a key
    pd_result = L.merge(R4, left_on="k", right_on="id")
    ds_result = DataStore.from_file(files["l"]).merge(
        DataStore.from_file(files["r4"]), left_on="k", right_on="id"
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)
