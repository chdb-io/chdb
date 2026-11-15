"""
to_sql(quote_char=...) quotes identifiers with the requested character also
when the SQL is rendered from the lazy-op plan (layered pipelines,
aggregations), not only for the flat query-builder SQL.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.expressions import col


@pytest.fixture
def ds(tmp_path):
    path = str(tmp_path / "t.parquet")
    pd.DataFrame({"g": ["x", "y", "x"], "a": [1, 2, 3], "b": [4, 5, 6]}).to_parquet(path)
    return DataStore.from_file(path)


CHAINS = [
    pytest.param(lambda d: (lambda x: x[x["c"] > 2])(d.assign(c=d["a"] * 2)), id="assign_then_filter"),
    pytest.param(lambda d: d.groupby("g").agg(t=col("a").sum()).sort_values("t"), id="agg_then_sort"),
    pytest.param(lambda d: d.groupby("g").agg(t=("a", "sum")).head(1), id="named_agg_then_head"),
    pytest.param(lambda d: d.select(col("a").as_("b"), col("b").as_("a")), id="select_alias_swap"),
    pytest.param(lambda d: d[d["a"] > 1].mask(d["a"] > 2, 0), id="filter_then_mask"),
    pytest.param(lambda d: d.head(2)[d["a"] > 1], id="limit_then_filter"),
]


@pytest.mark.parametrize("chain", CHAINS)
def test_requested_quote_char(ds, chain):
    pipeline = chain(ds)
    default = pipeline.to_sql()
    backticks = pipeline.to_sql(quote_char="`")
    assert '"' in default
    assert '"' not in backticks
    assert backticks == default.replace('"', "`")
