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
