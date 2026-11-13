"""
An outer merge() of two file sources gives the columns it fills with missing
values pandas' dtypes: float64 for integers and object for bools, not nullable
Int64 / boolean (#757).
"""

import os

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


L = pd.DataFrame({"k": [1, 2, 3], "a": [7, 8, 9], "f": [True, False, True]})
R = pd.DataFrame({"k": [1, 2, 4], "v": [10, 20, 40], "b": [True, False, True]})
R2 = pd.DataFrame({"rk": [1, 2, 4], "v": [10, 20, 40]})
R3 = pd.DataFrame({"k": [1, 3], "w": [100, 300]})
RN = pd.DataFrame({"k": [1, 2, 4], "v": pd.array([10, None, 40], dtype="Int64")})


@pytest.fixture(params=["parquet", "csv"])
def files(tmp_path, request):
    paths = {}
    for name, df in [("l", L), ("r", R), ("r2", R2), ("r3", R3), ("rn", RN)]:
        path = os.path.join(tmp_path, f"{name}.{request.param}")
        if request.param == "parquet":
            df.to_parquet(path, index=False)
        else:
            # chDB infers CSV bools from lowercase true/false
            bools = df.select_dtypes(bool).columns
            df.assign(**{c: df[c].map({True: "true", False: "false"}) for c in bools}).to_csv(
                path, index=False
            )
        paths[name] = path
    return paths


def test_left_merge_issue_example(tmp_path):
    l = pd.DataFrame({"k": [1, 2, 3]})
    r = pd.DataFrame({"k": [1, 2], "v": [10, 20]})
    lp, rp = os.path.join(tmp_path, "l.parquet"), os.path.join(tmp_path, "r.parquet")
    l.to_parquet(lp)
    r.to_parquet(rp)

    pd_result = l.merge(r, on="k", how="left")
    ds_result = DataStore.from_file(lp).merge(DataStore.from_file(rp), on="k", how="left")
    assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=True)
    assert ds_result.to_pandas()["v"].dtype == "float64"


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_merge_on(files, how):
    pd_result = L.merge(R, on="k", how=how)
    ds_result = DataStore.from_file(files["l"]).merge(
        DataStore.from_file(files["r"]), on="k", how=how
    )
    assert_datastore_equals_pandas(
        ds_result, pd_result, check_row_order=False, check_nullable_dtype=True
    )


@pytest.mark.parametrize("how", ["left", "right"])
def test_merge_left_on_right_on(files, how):
    pd_result = L.merge(R2, left_on="k", right_on="rk", how=how)
    ds_result = DataStore.from_file(files["l"]).merge(
        DataStore.from_file(files["r2"]), left_on="k", right_on="rk", how=how
    )
    assert_datastore_equals_pandas(
        ds_result, pd_result, check_row_order=False, check_nullable_dtype=True
    )


def test_left_merge_keeps_int64_with_own_nulls(files):
    ds_right = DataStore.from_file(files["rn"])
    pd_result = L.merge(ds_right.to_pandas(), on="k", how="left")
    ds_result = DataStore.from_file(files["l"]).merge(ds_right, on="k", how="left")
    assert_datastore_equals_pandas(
        ds_result, pd_result, check_row_order=False, check_nullable_dtype=True
    )
    assert ds_result.to_pandas()["v"].dtype == "Int64"


def test_chained_left_merges(files):
    pd_result = L.merge(R, on="k", how="left").merge(R3, on="k", how="left")
    ds_result = (
        DataStore.from_file(files["l"])
        .merge(DataStore.from_file(files["r"]), on="k", how="left")
        .merge(DataStore.from_file(files["r3"]), on="k", how="left")
    )
    assert_datastore_equals_pandas(
        ds_result, pd_result, check_row_order=False, check_nullable_dtype=True
    )


def test_left_merge_then_select_and_sort(files):
    pd_result = L.merge(R, on="k", how="left")[["k", "v", "b"]].sort_values("k")
    ds_merged = DataStore.from_file(files["l"]).merge(
        DataStore.from_file(files["r"]), on="k", how="left"
    )
    ds_result = ds_merged[["k", "v", "b"]].sort_values("k")
    assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=True)


def test_left_merge_column(files):
    pd_result = L.merge(R, on="k", how="left").sort_values("k")["v"]
    ds_result = (
        DataStore.from_file(files["l"])
        .merge(DataStore.from_file(files["r"]), on="k", how="left")
        .sort_values("k")["v"]
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=True)
