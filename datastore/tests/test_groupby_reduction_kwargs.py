"""
groupby()[col] reductions honour their pandas arguments: nunique(dropna=False)
counts a missing value as one more value, and an argument the SQL
aggregation step doesn't implement (std ddof=2, quantile interpolation) runs
the reduction in pandas instead of being dropped. Each case runs on a
DataFrame source and on Parquet and CSV files.
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.config import ExecutionEngine, set_execution_engine
from tests.test_utils import assert_series_equal, get_series

KEYS = ["a", "a", "a", "b", "b", "c"]

COLUMNS = {
    "int": pd.Series([1, 1, 2, 3, 4, 5], dtype="int64"),
    "float": pd.Series([1.0, np.nan, 1.0, 2.0, np.nan, np.nan]),
    "nullable_int": pd.Series([1, None, 2, 3, 3, None], dtype="Int64"),
    "str": pd.Series(["x", None, "x", "y", "z", None], dtype=object),
}


def _make(kind, df, tmp_path):
    if kind == "dataframe":
        return DataStore(df.copy())
    path = str(tmp_path / f"t.{kind}")
    if kind == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
    return DataStore.from_file(path)


@pytest.fixture(params=["dataframe", "parquet", "csv"])
def kind(request):
    return request.param


def _expected(df, chain):
    result = chain(df)
    # A file has no pandas dtype to restore; nunique is int64 everywhere
    return result


@pytest.mark.parametrize("column", list(COLUMNS))
@pytest.mark.parametrize("dropna", [True, False])
def test_nunique_dropna(kind, column, dropna, tmp_path):
    df = pd.DataFrame({"k": KEYS, "v": COLUMNS[column]})
    ds = _make(kind, df, tmp_path)
    assert_series_equal(
        get_series(ds.groupby("k")["v"].nunique(dropna=dropna)),
        df.groupby("k")["v"].nunique(dropna=dropna),
    )


@pytest.mark.parametrize("dropna", [True, False])
def test_nunique_dropna_via_agg(kind, dropna, tmp_path):
    df = pd.DataFrame({"k": KEYS, "v": COLUMNS["float"]})
    ds = _make(kind, df, tmp_path)
    assert_series_equal(
        get_series(ds.groupby("k")["v"].agg("nunique", dropna=dropna)),
        df.groupby("k")["v"].agg("nunique", dropna=dropna),
    )


@pytest.mark.parametrize("dropna", [True, False])
def test_nunique_dropna_as_index_false(kind, dropna, tmp_path):
    df = pd.DataFrame({"k": KEYS, "v": COLUMNS["float"]})
    ds = _make(kind, df, tmp_path)
    ds_result = ds.groupby("k", as_index=False)["v"].nunique(dropna=dropna)
    pd_result = df.groupby("k", as_index=False)["v"].nunique(dropna=dropna)
    pd.testing.assert_frame_equal(
        ds_result.to_pandas().reset_index(drop=True), pd_result.reset_index(drop=True)
    )


VALUES = pd.Series([1.0, 4.0, 9.0, 2.0, 8.0, 5.0])


@pytest.mark.parametrize(
    "reduce",
    [
        pytest.param(lambda g: g.std(ddof=2), id="std_ddof2"),
        pytest.param(lambda g: g.var(ddof=2), id="var_ddof2"),
        pytest.param(lambda g: g.std(ddof=0), id="std_ddof0"),
        pytest.param(lambda g: g.quantile(0.5, interpolation="lower"), id="quantile_lower"),
        pytest.param(lambda g: g.quantile(0.5, interpolation="nearest"), id="quantile_nearest"),
    ],
)
def test_reduction_arguments(kind, reduce, tmp_path):
    df = pd.DataFrame({"k": KEYS, "v": VALUES})
    ds = _make(kind, df, tmp_path)
    assert_series_equal(get_series(reduce(ds.groupby("k")["v"])), reduce(df.groupby("k")["v"]))


@pytest.mark.parametrize(
    "reduce",
    [
        pytest.param(lambda g: g.std(ddof=2), id="std_ddof2"),
        pytest.param(lambda g: g.quantile(0.5, interpolation="lower"), id="quantile_lower"),
    ],
)
@pytest.mark.parametrize("alias", [None, "w"])
def test_reduction_arguments_of_computed_column(kind, reduce, alias, tmp_path):
    # w is also a column of the frame: an alias names the result, it is not read
    df = pd.DataFrame({"k": KEYS, "v": VALUES, "w": VALUES * 100})
    ds = _make(kind, df, tmp_path)
    computed = ds.groupby("k")["v"] + 1
    if alias is not None:
        computed = computed.as_(alias)
    ds_result = get_series(reduce(computed))
    pd_result = reduce((df["v"] + 1).groupby(df["k"]))
    np.testing.assert_allclose(ds_result.to_numpy(), pd_result.to_numpy())
    assert list(ds_result.index) == list(pd_result.index)
    # The reduction keeps the name the supported path gives the same column
    assert ds_result.name == get_series(computed.std()).name



@pytest.mark.parametrize(
    "reduce",
    [
        pytest.param(lambda g: g.sum(min_count=2), id="sum_min_count2"),
        pytest.param(lambda g: g.nunique(dropna=False), id="nunique_dropna_false"),
    ],
)
@pytest.mark.parametrize("alias", [None, "w"])
def test_count_arguments_of_computed_column(kind, reduce, alias, tmp_path):
    df = pd.DataFrame({"k": KEYS, "v": COLUMNS["float"], "w": VALUES * 100})
    ds = _make(kind, df, tmp_path)
    computed = ds.groupby("k")["v"] + 1
    if alias is not None:
        computed = computed.as_(alias)
    ds_result = get_series(reduce(computed))
    pd_result = reduce((df["v"] + 1).groupby(df["k"]))
    np.testing.assert_array_equal(ds_result.to_numpy(dtype=float), pd_result.to_numpy(dtype=float))
    assert list(ds_result.index) == list(pd_result.index)
    assert ds_result.name == get_series(computed.sum()).name

@pytest.fixture(params=["auto", "pandas"])
def engine(request):
    set_execution_engine(
        ExecutionEngine.PANDAS if request.param == "pandas" else ExecutionEngine.AUTO
    )
    yield
    set_execution_engine(ExecutionEngine.AUTO)


# A group with a missing value first (a), last (b), only missing values (c)
# and none (d)
EDGE_KEYS = ["a", "a", "a", "b", "b", "c", "c", "d", "d"]
EDGE_COLUMNS = {
    "float": [None, 1.0, 2.0, 3.0, None, None, None, 4.0, 5.0],
    "nullable_int": pd.array([None, 1, 2, 3, None, None, None, 4, 5], dtype="Int64"),
    "str": [None, "x", "y", "z", None, None, None, "u", "w"],
}


@pytest.mark.parametrize("column", list(EDGE_COLUMNS))
@pytest.mark.parametrize("method", ["first", "last"])
@pytest.mark.parametrize("skipna", [True, False])
def test_first_last_skipna(engine, kind, column, method, skipna, tmp_path):
    df = pd.DataFrame({"k": EDGE_KEYS, "v": EDGE_COLUMNS[column]})
    ds = _make(kind, df, tmp_path)
    assert_series_equal(
        get_series(getattr(ds.groupby("k")["v"], method)(skipna=skipna)),
        getattr(df.groupby("k")["v"], method)(skipna=skipna),
    )


def test_sum_numeric_only_on_string_column_raises(engine, kind, tmp_path):
    """A string column is not numeric: numeric_only=True raises, as in pandas.

    The reduction is lazy, so the error surfaces when the result is materialized.
    """
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": ["x", "y", "z"]})
    ds = _make(kind, df, tmp_path)
    with pytest.raises(TypeError) as pd_err:
        df.groupby("g")["v"].sum(numeric_only=True)
    with pytest.raises(TypeError) as ds_err:
        ds.groupby("g")["v"].sum(numeric_only=True).to_pandas()
    assert str(ds_err.value) == str(pd_err.value)

    with pytest.raises(TypeError) as agg_err:
        ds.groupby("g")["v"].agg("sum", numeric_only=True).to_pandas()
    assert str(agg_err.value) == str(pd_err.value)


def test_sum_numeric_only_on_numeric_column(engine, kind, tmp_path):
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    ds = _make(kind, df, tmp_path)
    assert_series_equal(
        get_series(ds.groupby("g")["v"].sum(numeric_only=True)),
        df.groupby("g")["v"].sum(numeric_only=True),
    )
