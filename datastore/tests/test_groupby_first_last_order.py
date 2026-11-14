"""
Grouped first()/last() read the first/last row in row order, also for a
computed column, on a DataFrame and on a Parquet file read in many row groups;
a grouped column under another name (as_()) aggregates its own values.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datastore import DataStore
from datastore.config import CompatMode, ExecutionEngine, set_compat_mode, set_execution_engine
from datastore.lazy_ops import LazyGroupByAgg
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series

ROWS = 60_000


def _frame():
    rng = np.random.default_rng(0)
    v = np.arange(ROWS, dtype="float64")
    v[:7] = np.nan  # the first row of every group is missing: first() skips it
    return pd.DataFrame({"g": np.arange(ROWS) % 7, "v": v, "w": rng.integers(0, 100, ROWS)})


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("first_last") / "rows.parquet")
    pq.write_table(pa.Table.from_pandas(_frame(), preserve_index=False), path, row_group_size=499)
    return path


@pytest.fixture(params=["dataframe", "parquet"])
def make_ds(request, parquet_path):
    if request.param == "dataframe":
        return lambda: DataStore(_frame())
    return lambda: DataStore.from_file(parquet_path)


SERIES = {
    "first": (lambda d: d.groupby("g")["v"].first(), lambda p: p.groupby("g")["v"].first()),
    "last": (lambda d: d.groupby("g")["v"].last(), lambda p: p.groupby("g")["v"].last()),
    "computed_first": (
        lambda d: (d.groupby("g")["v"] * 2).first(),
        lambda p: (p["v"] * 2).groupby(p["g"]).first(),
    ),
    "computed_last": (
        lambda d: (d.groupby("g")["w"] + 0).last(),
        lambda p: (p["w"] + 0).groupby(p["g"]).last(),
    ),
    "two_columns_first": (
        lambda d: (d.groupby("g")["v"] + d["w"]).first(),
        lambda p: (p["v"] + p["w"]).groupby(p["g"]).first(),
    ),
    "transform_first": (
        lambda d: d.groupby("g")["v"].transform("first"),
        lambda p: p.groupby("g")["v"].transform("first"),
    ),
    "assigned_last": (
        lambda d: d.assign(c=d["w"] * 3).groupby("g")["c"].last(),
        lambda p: p.assign(c=p["w"] * 3).groupby("g")["c"].last(),
    ),
}
FRAMES = {
    "named": lambda s: s.groupby("g").agg(f=("v", "first"), l=("w", "last")),
    "dict": lambda s: s.groupby("g").agg({"v": "first", "w": "last"}),
    "list": lambda s: s.groupby("g")["w"].agg(["first", "last"]),
    "frame_first": lambda s: s.groupby("g").first(),
    "frame_last": lambda s: s.groupby("g").last(),
}


@pytest.mark.parametrize("name", sorted(SERIES))
def test_series(make_ds, name):
    build_ds, build_pd = SERIES[name]
    assert_series_equal(get_series(build_ds(make_ds())), build_pd(_frame()))


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_frame(make_ds, name):
    assert_datastore_equals_pandas(FRAMES[name](make_ds()), FRAMES[name](_frame()))


@pytest.fixture
def aggregates(monkeypatch):
    """The SQL aggregates of every LazyGroupByAgg a grouped column runs."""
    seen = []
    original = LazyGroupByAgg.run_on

    def spy(self, datastore):
        seen.extend(self.expr_agg.values() if self.expr_agg else [])
        return original(self, datastore)

    monkeypatch.setattr(LazyGroupByAgg, "run_on", spy)
    return seen


@pytest.mark.parametrize("func", ["first", "last"])
def test_computed_column_aggregates_in_row_order(aggregates, func):
    d = DataStore(_frame())
    getattr(d.groupby("g")["v"] * 2, func)()._execute()
    names = {expr.name for expr in aggregates}
    assert names == {"argMin" if func == "first" else "argMax"}


@pytest.mark.parametrize("func", ["first", "last"])
def test_agg_mode_expression_in_row_order(func):
    d = DataStore(_frame())
    expr = getattr(d.groupby("g")["v"] * 2, func)()._expr
    assert expr.name == ("argMin" if func == "first" else "argMax")


@pytest.fixture
def performance_mode():
    set_compat_mode(CompatMode.PERFORMANCE)
    yield
    set_compat_mode(CompatMode.PANDAS)
    set_execution_engine(ExecutionEngine.AUTO)


def test_performance_mode_keeps_any(performance_mode):
    d = DataStore(_frame())
    assert (d.groupby("g")["v"] * 2).first()._expr.name == "any"
    assert (d.groupby("g")["v"] * 2).last()._expr.name == "anyLast"


ALIASES = {
    "existing_name_sum": (
        lambda d: d.groupby("g")["v"].as_("w").sum(),
        lambda p: p.groupby("g")["v"].sum().rename("w"),
    ),
    "new_name_sum": (
        lambda d: d.groupby("g")["v"].as_("z").sum(),
        lambda p: p.groupby("g")["v"].sum().rename("z"),
    ),
    "key_column_sum": (
        lambda d: d.groupby("w")["w"].as_("v").sum(),
        lambda p: p.groupby("w")["w"].sum().rename("v"),
    ),
    "mean": (
        lambda d: d.groupby("g")["w"].as_("v").mean(),
        lambda p: p.groupby("g")["w"].mean().rename("v"),
    ),
    "first": (
        lambda d: d.groupby("g")["w"].as_("v").first(),
        lambda p: p.groupby("g")["w"].first().rename("v"),
    ),
    "computed_last": (
        lambda d: (d.groupby("g")["w"] * 2).as_("z").last(),
        lambda p: (p["w"] * 2).groupby(p["g"]).last().rename("z"),
    ),
}


@pytest.mark.parametrize("name", sorted(ALIASES))
def test_aliased_column(make_ds, name):
    build_ds, build_pd = ALIASES[name]
    assert_series_equal(get_series(build_ds(make_ds())), build_pd(_frame()))


def test_aliased_column_as_index_false(make_ds):
    result = make_ds().groupby("g", as_index=False)["v"].as_("w").sum()
    expected = _frame().groupby("g", as_index=False)["v"].sum().rename(columns={"v": "w"})
    assert_datastore_equals_pandas(result, expected)
