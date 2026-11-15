"""
Arithmetic assigned with assign() or ``ds[col] = ...`` and computed in SQL
keeps the dtype pandas gives it where ClickHouse would widen it: int8 + 1
stays int8 and wraps around, float32 * 1e100 stays float32 and overflows to
inf (#766).
"""

import os

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


DF = pd.DataFrame(
    {
        "i8": np.array([1, 2, 3, 100], dtype="int8"),
        "u8": np.array([250, 2, 3, 0], dtype="uint8"),
        "i16": np.array([1, 2, 3, -300], dtype="int16"),
        "i32": np.array([2**31 - 1, 2, 3, -5], dtype="int32"),
        "u64": np.array([0, 2, 3, 2**64 - 1], dtype="uint64"),
        "i64": np.array([1, 2, 3, 4], dtype="int64"),
        "f32": np.array([1.5, 2.5, 3.5, 0.1], dtype="float32"),
    }
)

EXPRESSIONS = [
    pytest.param(lambda t: t["i8"] + 1, id="i8+1"),
    pytest.param(lambda t: t["i8"] * 100, id="i8*100"),
    pytest.param(lambda t: t["u8"] + 10, id="u8+10"),
    pytest.param(lambda t: t["u8"] - 251, id="u8-251"),
    pytest.param(lambda t: -t["i8"], id="-i8"),
    pytest.param(lambda t: (t["i8"] + 1) * 2, id="(i8+1)*2"),
    pytest.param(lambda t: t["i8"] + t["i16"], id="i8+i16"),
    pytest.param(lambda t: t["i8"] - t["u8"], id="i8-u8"),
    pytest.param(lambda t: t["i32"] + 1, id="i32+1"),
    pytest.param(lambda t: t["u64"] - 1, id="u64-1"),
    pytest.param(lambda t: t["i64"] + 1, id="i64+1"),
    pytest.param(lambda t: t["i8"] + np.int64(1), id="i8+np.int64"),
    pytest.param(lambda t: t["f32"] * 1e100, id="f32*1e100"),
    pytest.param(lambda t: t["f32"] + 1, id="f32+1"),
    pytest.param(lambda t: t["f32"] / 3, id="f32/3"),
    pytest.param(lambda t: t["f32"] + 0.1, id="f32+0.1"),
    pytest.param(lambda t: t["f32"] + t["i8"], id="f32+i8"),
    pytest.param(lambda t: t["f32"] * t["f32"] - t["f32"], id="f32*f32-f32"),
    pytest.param(lambda t: t["i8"] / 2, id="i8/2"),
    pytest.param(lambda t: t["i8"] + 1.5, id="i8+1.5"),
]


@pytest.fixture(params=["dataframe", "parquet"])
def source(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(DF.copy())
    path = os.path.join(tmp_path, "data.parquet")
    DF.to_parquet(path, index=False)
    return lambda: DataStore.from_file(path)


def pandas_result(fn):
    with np.errstate(over="ignore"):
        return fn(DF.copy())


def test_issue_example():
    df = pd.DataFrame(
        {
            "i8": np.array([1, 2, 3], dtype="int8"),
            "u8": np.array([250, 2, 3], dtype="uint8"),
            "f32": np.array([1.5, 2.5, 3.5], dtype="float32"),
        }
    )
    for fn in [
        lambda t: t["i8"] + 1,
        lambda t: t["i8"] * 100,
        lambda t: t["u8"] + 10,
        lambda t: t["f32"] * 1e100,
    ]:
        with np.errstate(over="ignore"):
            pd_result = df.assign(x=fn(df))["x"]
        ds = DataStore(df)
        ds_result = ds.assign(x=fn(ds))["x"]
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=True)


@pytest.mark.parametrize("fn", EXPRESSIONS)
def test_assign(source, fn):
    pd_result = pandas_result(lambda df: df.assign(x=fn(df)))
    ds = source()
    assert_datastore_equals_pandas(ds.assign(x=fn(ds)), pd_result, check_nullable_dtype=True)


@pytest.mark.parametrize("fn", EXPRESSIONS)
def test_setitem(source, fn):
    def assign(t):
        t["x"] = fn(t)
        return t

    pd_result = pandas_result(assign)
    assert_datastore_equals_pandas(assign(source()), pd_result, check_nullable_dtype=True)


def test_filter_on_wrapped_value(source):
    pd_df = DF.assign(x=DF["i8"] * 100)
    pd_result = pd_df[pd_df["x"] < 0]
    ds = source()
    ds = ds.assign(x=ds["i8"] * 100)
    ds_result = ds[ds["x"] < 0]
    assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=True)


def test_setitem_reads_computed_column(source):
    def assign(t):
        t["y"] = t["i8"] + 1
        t["z"] = t["y"] * 100
        t["i8"] = t["i8"] * 3
        return t

    pd_result = assign(DF.copy())
    assert_datastore_equals_pandas(assign(source()), pd_result, check_nullable_dtype=True)


def test_assign_callable_reads_earlier_column(source):
    def fns():
        return dict(x=lambda t: t["i8"] + 1, y=lambda t: t["x"] * 100, f32=lambda t: t["f32"] / 3)

    pd_result = DF.assign(**fns())
    assert_datastore_equals_pandas(source().assign(**fns()), pd_result, check_nullable_dtype=True)


def test_nullable_integer_column():
    df = pd.DataFrame(
        {
            "n8": pd.array([1, None, 3, 120], dtype="Int8"),
            "i8": np.array([1, 2, 3, 2], dtype="int8"),
        }
    )
    pd_result = df.assign(x=df["n8"] + 10, y=df["n8"] * df["i8"])
    ds = DataStore(df)
    ds_result = ds.assign(x=ds["n8"] + 10, y=ds["n8"] * ds["i8"])
    assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=True)
