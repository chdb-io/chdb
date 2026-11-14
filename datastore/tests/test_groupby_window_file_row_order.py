"""
groupby()[col].cumsum()/cummax()/cummin() over a file keep the file's row
order and row labels, as a Series and as an assigned column (#759). The SQL
window behind them returns its rows grouped by partition.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


KEYS = ["a", "b", "a", "b", "a", "b"]
FLOATS = pd.DataFrame({"k": KEYS, "v": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]})
INTS = pd.DataFrame({"k": KEYS, "v": [1, 10, 2, 20, 3, 30]})

METHODS = ["cumsum", "cummax", "cummin"]


@pytest.fixture(params=["parquet", "csv", "parquet_not_null"])
def source(tmp_path, request):
    """A frame and a file of it whose v can't hold pandas NA, so the
    window runs in SQL."""
    if request.param == "csv":
        path = str(tmp_path / "t.csv")
        FLOATS.to_csv(path, index=False)
        return FLOATS, path
    path = str(tmp_path / "t.parquet")
    if request.param == "parquet":
        FLOATS.to_parquet(path)
        return FLOATS, path
    schema = pa.schema(
        [pa.field("k", pa.string(), nullable=False), pa.field("v", pa.int64(), nullable=False)]
    )
    pq.write_table(pa.Table.from_pandas(INTS, schema=schema, preserve_index=False), path)
    return INTS, path


def _assign(frame, name, value):
    frame[name] = value
    return frame


@pytest.mark.parametrize("method", METHODS)
def test_series(source, method):
    df, path = source
    ds = DataStore.from_file(path)
    assert_series_equal(
        get_series(getattr(ds.groupby("k")["v"], method)()),
        getattr(df.groupby("k")["v"], method)(),
    )


@pytest.mark.parametrize("method", METHODS)
def test_assigned(source, method):
    df, path = source
    ds = DataStore.from_file(path)
    ds["c"] = getattr(ds.groupby("k")["v"], method)()
    pd_df = df.copy()
    pd_df["c"] = getattr(pd_df.groupby("k")["v"], method)()
    assert_datastore_equals_pandas(ds, pd_df, check_index=True)


CHAINS = [
    pytest.param(lambda d: d.assign(c=d.groupby("k")["v"].cumsum()), id="assign"),
    pytest.param(lambda d: _assign(d, "c", d.groupby("k")["v"].cumsum()).head(3), id="head"),
    pytest.param(
        lambda d: _assign(d, "c", d.groupby("k")["v"].cumsum())[["k", "c"]], id="projection"
    ),
    pytest.param(
        lambda d: (lambda x: _assign(x, "d", x["c"] * 2))(
            _assign(d, "c", d.groupby("k")["v"].cumsum())
        ),
        id="computed_from_window",
    ),
    pytest.param(
        lambda d: (lambda x: _assign(x, "c", x.groupby("k")["w"].cumsum()))(
            _assign(d, "w", d["v"] * 2)
        ),
        id="window_of_computed",
    ),
    pytest.param(lambda d: _assign(d, "r", d["v"].rank(method="first")), id="rank_first"),
]


@pytest.mark.parametrize("chain", CHAINS)
def test_chain_after_assignment(source, chain):
    df, path = source
    assert_datastore_equals_pandas(
        chain(DataStore.from_file(path)), chain(df.copy()), check_index=True
    )


@pytest.fixture(scope="module")
def large_file(tmp_path_factory):
    rng = np.random.default_rng(759)
    n = 200_000
    df = pd.DataFrame(
        {"k": rng.choice(list("abcdefg"), n), "v": rng.integers(0, 100, n).astype("float64")}
    )
    path = str(tmp_path_factory.mktemp("large") / "large.parquet")
    df.to_parquet(path, row_group_size=10_000)
    return df, path


def test_large_file(large_file):
    df, path = large_file
    ds = DataStore.from_file(path)
    assert_series_equal(get_series(ds.groupby("k")["v"].cumsum()), df.groupby("k")["v"].cumsum())

    ds["c"] = ds.groupby("k")["v"].cumsum()
    pd_df = df.copy()
    pd_df["c"] = pd_df.groupby("k")["v"].cumsum()
    assert_datastore_equals_pandas(ds, pd_df, check_index=True)


def test_large_file_filtered_values(large_file):
    df, path = large_file
    ds = DataStore.from_file(path)
    result = get_series(ds[ds["v"] > 30].groupby("k")["v"].cumsum())
    expected = df[df["v"] > 30].groupby("k")["v"].cumsum()
    # Labels are checked by the test below
    np.testing.assert_array_equal(result.to_numpy(), expected.to_numpy())


@pytest.mark.xfail(strict=True, reason="a filtered file source is relabelled 0..n-1 (#735)")
def test_large_file_filtered_labels(large_file):
    df, path = large_file
    ds = DataStore.from_file(path)
    result = get_series(ds[ds["v"] > 30].groupby("k")["v"].cumsum())
    expected = df[df["v"] > 30].groupby("k")["v"].cumsum()
    pd.testing.assert_index_equal(result.index, expected.index, exact=False)
