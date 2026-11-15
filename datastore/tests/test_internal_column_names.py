"""
Internal temporary columns never take the place of a user column: a group
key, source column or output named like one (``__agg_0__``, ``__value__``,
``__tmp_x__``, ``__count__``) keeps its own values. Each case runs on a
DataFrame source and on a file source; group keys are categorical (the
aggregates then run in pandas) and plain (they run in SQL).
"""

import logging

import pandas as pd
import pytest

from datastore import DataStore
from datastore.expressions import col
from tests.test_utils import assert_datastore_equals_pandas

INTERNAL_NAMES = ["__agg_0__", "__agg_1__", "__value__", "__key_k__"]


def _make(kind, df, tmp_path, categorical=()):
    if kind == "dataframe":
        return DataStore(df.copy())
    path = str(tmp_path / "t.parquet")
    df.assign(**{name: df[name].astype(str) for name in categorical}).to_parquet(path)
    ds = DataStore.from_file(path)
    if categorical:
        ds = ds.astype({name: df[name].dtype for name in categorical})
    return ds


@pytest.fixture(params=["dataframe", "file"])
def kind(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["categorical", "plain"])
def categorical(request):
    return request.param


def _keys(values, categorical):
    return pd.Categorical(values, categories=["b", "a"]) if categorical else values


@pytest.mark.parametrize("name", INTERNAL_NAMES)
def test_group_key_named_like_internal_column(kind, categorical, name, tmp_path):
    df = pd.DataFrame({name: _keys(["a", "b", "a"], categorical), "v": [1, 2, 3]})
    ds = _make(kind, df, tmp_path, [name] if categorical else ())

    ds_result = ds.groupby(name, observed=True).agg(t=col("v").sum(), k=col(name))

    pd_result = df.groupby(name, observed=True).agg(t=("v", "sum"))
    pd_result["k"] = pd_result.index
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("name", INTERNAL_NAMES)
def test_source_column_named_like_internal_column(kind, categorical, name, tmp_path):
    df = pd.DataFrame(
        {"k": _keys(["a", "b", "a"], categorical), name: [10, 20, 30], "v": [1, 2, 3]}
    )
    ds = _make(kind, df, tmp_path, ["k"] if categorical else ())

    ds_result = ds.groupby("k", observed=True).agg(
        t=col("v").sum(), u=col(name).max() - col("v").min()
    )

    grouped = df.groupby("k", observed=True)
    pd_result = pd.DataFrame(
        {"t": grouped["v"].sum(), "u": grouped[name].max() - grouped["v"].min()}
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("name", INTERNAL_NAMES)
def test_output_named_like_internal_column(kind, categorical, name, tmp_path):
    df = pd.DataFrame({"k": _keys(["a", "b", "a"], categorical), "v": [1, 2, 3]})
    ds = _make(kind, df, tmp_path, ["k"] if categorical else ())

    ds_result = ds.groupby("k", observed=True).agg(
        **{name: col("v").sum(), "m": col("v").max() * 10}
    )

    grouped = df.groupby("k", observed=True)
    pd_result = pd.DataFrame({name: grouped["v"].sum(), "m": grouped["v"].max() * 10})
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("method", ["where", "mask"])
def test_where_with_column_named_like_temp_alias(kind, method, tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "__tmp_x__": [10, 20, 30]})
    ds = _make(kind, df, tmp_path)

    ds_result = getattr(ds, method)(ds["x"] > 1, 0)

    pd_result = getattr(df, method)(df["x"] > 1, 0)
    assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("name", ["__count__", "__value__"])
def test_value_counts_of_column_named_like_result_column(kind, name, tmp_path, caplog):
    df = pd.DataFrame({name: ["a", "b", "a", "c", "a"]})
    ds = _make(kind, df, tmp_path)

    with caplog.at_level(logging.ERROR):
        ds_result = ds[name].value_counts()
        pd_result = df[name].value_counts()
        assert_datastore_equals_pandas(ds_result, pd_result)

    # Counted in SQL, without a failed query falling back to pandas
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
