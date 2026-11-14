"""
drop_duplicates() runs where it appears in the chain (issue #753).

It reads the rows every earlier op leaves (filters, sorts, limits,
assignments, projections), and every later op reads its result. A string
subset is one column.

All tests use the mirror pattern on DataFrame, Parquet and CSV sources.
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


DATA = pd.DataFrame({"k": [1, 1, 2, 2, 3, 3, 4, 4], "w": [8, 7, 6, 5, 4, 3, 2, 1], "s": list("aabbccdd")})


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
    # From #753
    "tail_then_subset": lambda d: d.tail(4).drop_duplicates(subset=["k"]),
    "head_then_subset": lambda d: d.head(3).drop_duplicates(subset=["k"]),
    "sort_then_subset": lambda d: d.sort_values("w").drop_duplicates(subset=["k"]),
    "subset_then_filter": lambda d: (lambda x: x[x["w"] < 8])(d.drop_duplicates(subset=["k"])),
    "subset_then_groupby_sum": lambda d: d.drop_duplicates(subset=["k"]).groupby("k")["w"].sum(),
    "assign_then_all": lambda d: d.assign(w=0).drop_duplicates(),
    "string_subset": lambda d: d.drop_duplicates(subset="k"),
    # Earlier and later ops
    "assign_then_subset": lambda d: d.assign(w=d["w"] % 2).drop_duplicates(subset=["k", "w"]),
    "select_then_all": lambda d: d[["k", "s"]].drop_duplicates(),
    "all_then_select": lambda d: d.drop_duplicates()[["k"]],
    "filter_then_subset": lambda d: (lambda x: x[x["w"] < 8].drop_duplicates(subset=["k"]))(d),
    "slice_then_subset": lambda d: d[3:].drop_duplicates(subset=["k"]),
    "subset_then_head": lambda d: d.drop_duplicates(subset=["k"]).head(2),
    "subset_then_sort": lambda d: d.drop_duplicates(subset=["k"]).sort_values("w"),
    "subset_then_assign": lambda d: d.drop_duplicates(subset=["k"]).assign(z=lambda x: x["w"] * 2),
    "sort_head_then_all": lambda d: d.sort_values("w").head(5).drop_duplicates(subset=["s"]),
    "subset_twice": lambda d: d.drop_duplicates(subset=["k"]).drop_duplicates(subset=["s"]),
    "all_then_filter": lambda d: (lambda x: x[x["k"] > 1])(d[["k", "s"]].drop_duplicates()),
    "all_then_sort": lambda d: d[["k", "s"]].drop_duplicates().sort_values("k", ascending=False),
    "sort_then_all_then_sort": lambda d: (
        d[["k", "s"]].sort_values("s", ascending=False).drop_duplicates().sort_values("k")
    ),
}


@pytest.mark.parametrize("chain", list(CHAINS), ids=list(CHAINS))
def test_drop_duplicates_in_chain(make_source, chain):
    pd_result = CHAINS[chain](DATA)

    ds_result = CHAINS[chain](make_source())

    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)
