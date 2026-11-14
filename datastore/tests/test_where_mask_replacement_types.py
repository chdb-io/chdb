"""
where()/mask() with a scalar replacement match pandas for every pairing of
column dtype and replacement type: SQL computes them only when the result
keeps the column's dtype, and pandas computes a replacement that changes it
(a str into numbers, a number into bools, a float into ints, a value into a
nullable column) or raises as pandas does.
"""

import pandas as pd
import pytest

import datastore.core as core
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


def _frame():
    return pd.DataFrame(
        {
            "i": [1, 2, 3, 4],
            "f": [1.5, 2.5, 3.5, 4.5],
            "b": [True, False, True, False],
            "s": ["a", "b", "c", "d"],
            "n": pd.array([1, None, 3, 4], dtype="Int64"),
            "k": [1, 2, 3, 4],
        }
    )


@pytest.fixture(params=["dataframe", "parquet"])
def make_ds(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(_frame())
    path = str(tmp_path / "frame.parquet")
    _frame().to_parquet(path)
    return lambda: DataStore.from_file(path)


COLUMNS = ["i", "f", "b", "s", "n"]
REPLACEMENTS = {"str": "x", "int": 7, "float": 0.5, "bool": True}


def _mirror(ds_call, pd_call, compare):
    try:
        expected = pd_call()
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            compare(ds_call(), None)
        return
    compare(ds_call(), expected)


def _series_equal(result, expected):
    series = get_series(result)
    if expected is not None:
        assert_series_equal(series, expected)


def _frame_equal(result, expected):
    if expected is None:
        get_series(result.iloc[:, 0])
        return
    assert_datastore_equals_pandas(result, expected)


@pytest.mark.parametrize("method", ["mask", "where"])
@pytest.mark.parametrize("other", sorted(REPLACEMENTS))
@pytest.mark.parametrize("column", COLUMNS)
def test_series_method(make_ds, column, other, method):
    value = REPLACEMENTS[other]
    d = make_ds()
    p = _frame()
    _mirror(
        lambda: getattr(d[column], method)(d["k"] > 2, value),
        lambda: getattr(p[column], method)(p["k"] > 2, value),
        _series_equal,
    )


@pytest.mark.parametrize("method", ["mask", "where"])
@pytest.mark.parametrize("other", sorted(REPLACEMENTS))
@pytest.mark.parametrize("column", COLUMNS)
def test_series_method_assigned(make_ds, column, other, method):
    value = REPLACEMENTS[other]
    p = _frame()
    try:
        p["r"] = getattr(p[column], method)(p["k"] > 2, value)
    except (TypeError, ValueError) as error:
        d = make_ds()
        d["r"] = getattr(d[column], method)(d["k"] > 2, value)
        with pytest.raises(type(error)):
            get_series(d["r"])
        return
    d = make_ds()
    d["r"] = getattr(d[column], method)(d["k"] > 2, value)
    assert_datastore_equals_pandas(d, p)


@pytest.mark.parametrize("method", ["mask", "where"])
@pytest.mark.parametrize("other", sorted(REPLACEMENTS))
@pytest.mark.parametrize("column", COLUMNS)
def test_frame_method(make_ds, column, other, method):
    value = REPLACEMENTS[other]
    d = make_ds()
    p = _frame()
    _mirror(
        lambda: getattr(d[[column, "k"]], method)(d["k"] > 2, value, axis=0),
        lambda: getattr(p[[column, "k"]], method)(p["k"] > 2, value, axis=0),
        _frame_equal,
    )


@pytest.fixture
def execute_calls(monkeypatch):
    calls = []
    original = core.DataStore._execute

    def counting_execute(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(core.DataStore, "_execute", counting_execute)
    return calls


@pytest.mark.parametrize("other", sorted(REPLACEMENTS))
def test_building_does_not_execute(make_ds, execute_calls, other):
    d = make_ds()
    d["k"].mask(d["k"] > 2, REPLACEMENTS[other])
    d[["i", "k"]].where(d["k"] > 2, REPLACEMENTS[other], axis=0)
    assert execute_calls == []
