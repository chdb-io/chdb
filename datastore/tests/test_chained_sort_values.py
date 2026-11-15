"""
A sort_values() after another one keeps the earlier order for its ties when
both sorts end up in one SQL ORDER BY, as with a following head() (#760).
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


ISSUE_DF = pd.DataFrame({"w": [3, 1, 2, 3, 1, 2], "v": [1, 1, 2, 2, 1, 1], "x": list("abcdef")})

DF = pd.DataFrame(
    {
        "w": [3, 1, 2, 3, 1, 2, 2, 3, 1, 1, 3, 2],
        "v": [1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1],
        "u": [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        "x": list("abcdefghijkl"),
    }
)

STABLE = {"kind": "stable"}


def _source(kind, df, tmp_path):
    if kind == "dataframe":
        return DataStore(df.copy())
    path = str(tmp_path / f"t.{kind}")
    if kind == "parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path, index=False)
    return DataStore.from_file(path)


SOURCES = ["dataframe", "parquet", "csv"]


@pytest.mark.parametrize("source", SOURCES)
def test_issue_chain(source, tmp_path):
    def chain(d):
        return d.sort_values("w").sort_values("v", kind="stable").head(3)

    ds_result = chain(_source(source, ISSUE_DF, tmp_path))
    pd_result = chain(ISSUE_DF.copy())
    assert list(pd_result.index) == [1, 4, 5]
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=source == "dataframe")


CHAINS = [
    pytest.param(
        lambda d: d.sort_values("w", **STABLE).sort_values("v", **STABLE).head(5), id="stable"
    ),
    pytest.param(
        lambda d: d.sort_values("w", kind="mergesort").sort_values("v", kind="mergesort").head(5),
        id="mergesort",
    ),
    pytest.param(
        lambda d: d.sort_values("w", ascending=False, **STABLE)
        .sort_values("v", ascending=False, **STABLE)
        .head(5),
        id="descending",
    ),
    pytest.param(
        lambda d: d.sort_values(["w", "x"], ascending=[False, True], **STABLE)
        .sort_values(["v", "u"], ascending=[True, False], **STABLE)
        .head(6),
        id="multi_key",
    ),
    pytest.param(
        lambda d: d.sort_values(["v", "w"], **STABLE).sort_values("v", ascending=False, **STABLE).head(6),
        id="shared_key",
    ),
    pytest.param(
        lambda d: d.sort_values("x", ascending=False, **STABLE)
        .sort_values("w", **STABLE)
        .sort_values("v", **STABLE)
        .head(6),
        id="three_sorts",
    ),
    pytest.param(
        lambda d: d.sort_values("w", **STABLE)[["v", "x"]].sort_values("v", **STABLE).head(5),
        id="projection_between",
    ),
    pytest.param(
        lambda d: d.sort_values("w", **STABLE).sort_values("v", **STABLE).tail(4), id="tail"
    ),
    pytest.param(
        lambda d: d.sort_values("w", **STABLE).sort_values("v", **STABLE).iloc[2:7], id="iloc"
    ),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("chain", CHAINS)
def test_chain(source, chain, tmp_path):
    ds_result = chain(_source(source, DF, tmp_path))
    pd_result = chain(DF.copy())
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=source == "dataframe")


def test_filter_between_sorts():
    def chain(d):
        s = d.sort_values("w", **STABLE)
        return s[s["u"] == 1].sort_values("v", **STABLE).head(4)

    assert_datastore_equals_pandas(chain(DataStore(DF.copy())), chain(DF.copy()), check_index=True)


def test_to_sql_puts_later_sort_first(tmp_path):
    path = str(tmp_path / "t.parquet")
    ISSUE_DF.to_parquet(path)
    ds = DataStore.from_file(path).sort_values("w").sort_values("v", kind="stable").head(3)
    assert 'ORDER BY "v" ASC, "w" ASC LIMIT 3' in ds.to_sql()
