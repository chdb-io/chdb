"""
Chained head() / tail() / slices / offset() / limit() select the rows pandas
selects, with pandas' labels.

Each offset or limit applies to the previous result, as ``df.iloc[n:]`` and
``df.head(n)`` do, so the SQL a chain compiles to, its row labels and its
row count all come from one composed slice.

All tests use the mirror pattern and compare values, labels and length.
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas as _assert_values_equal


DATA = pd.DataFrame({"v": range(100, 112), "w": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]})


def assert_datastore_equals_pandas(ds_result, pd_result):
    _assert_values_equal(ds_result, pd_result)
    pd.testing.assert_index_equal(ds_result.index, pd_result.index, exact=False)
    assert len(ds_result) == len(pd_result)


@pytest.fixture(params=["dataframe", "parquet", "csv"])
def make_source(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(DATA.copy())
    path = str(tmp_path / f"data.{request.param}")
    if request.param == "parquet":
        DATA.to_parquet(path)
    else:
        DATA.to_csv(path, index=False)
    return lambda: DataStore.from_file(path)


CHAINS = {
    "offset_offset": (lambda d: d.offset(2).offset(3), lambda p: p.iloc[2:].iloc[3:]),
    "offset_offset_head": (
        lambda d: d.offset(2).offset(3).head(2),
        lambda p: p.iloc[2:].iloc[3:].head(2),
    ),
    "offset_limit_offset_limit": (
        lambda d: d.offset(2).limit(5).offset(1).limit(2),
        lambda p: p.iloc[2:].iloc[:5].iloc[1:].iloc[:2],
    ),
    "limit_offset": (lambda d: d.limit(5).offset(2), lambda p: p.iloc[:5].iloc[2:]),
    "limit_limit": (lambda d: d.limit(3).limit(5), lambda p: p.iloc[:3].iloc[:5]),
    "offset_past_end": (lambda d: d.offset(3).offset(20), lambda p: p.iloc[3:].iloc[20:]),
    "limit_offset_past_limit": (
        lambda d: d.limit(3).offset(5),
        lambda p: p.iloc[:3].iloc[5:],
    ),
    "head_tail": (lambda d: d.head(5).tail(2), lambda p: p.head(5).tail(2)),
    "tail_offset": (lambda d: d.tail(3).offset(1), lambda p: p.tail(3).iloc[1:]),
    "slice_slice": (lambda d: d[1:9][2:5], lambda p: p[1:9][2:5]),
    "slice_offset_limit": (lambda d: d[2:].offset(1).limit(3), lambda p: p[2:].iloc[1:].iloc[:3]),
    "empty_slice": (lambda d: d[5:3], lambda p: p[5:3]),
    "filter_offset_offset": (
        lambda d: d[d["v"] % 2 == 0].offset(1).offset(1),
        lambda p: p[p["v"] % 2 == 0].iloc[1:].iloc[1:],
    ),
    "offset_filter_offset": (
        lambda d: (lambda x: x[x["w"] > 2].offset(1))(d.offset(2)),
        lambda p: (lambda x: x[x["w"] > 2].iloc[1:])(p.iloc[2:]),
    ),
    "sort_offset_offset_head": (
        lambda d: d.sort_values("w", kind="stable").offset(1).offset(2).head(3),
        lambda p: p.sort_values("w", kind="stable").iloc[1:].iloc[2:].head(3),
    ),
}


@pytest.mark.parametrize("chain", list(CHAINS), ids=list(CHAINS))
def test_chained_slices(make_source, chain):
    ds_chain, pd_chain = CHAINS[chain]
    pd_result = pd_chain(DATA)

    ds_result = ds_chain(make_source())

    assert_datastore_equals_pandas(ds_result, pd_result)


def test_flat_sql_composes_offsets(make_source):
    sql = make_source().offset(2).offset(3).limit(4).to_sql()

    assert "LIMIT 4 OFFSET 5" in sql

