"""
A window or other lazy value nested in a CASE, a where/mask, a condition or a
function argument keeps the rows it was computed over: assigned after a filter
it is aligned, not recomputed on the filtered rows. ``Expression.subtree()``
sees every child of every container node, so ``is_row_wise`` and
``columns_read()`` account for nested column expressions and lazy values.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.case_when import CaseWhenExpr
from datastore.column_expr import ColumnExpr
from datastore.conditions import BetweenCondition, InCondition
from datastore.expressions import Field, Literal, OpaqueValue
from datastore.functions import Function, OrderByItem, WindowFunction
from datastore.lazy_result import LazyCondition
from tests.test_utils import assert_datastore_equals_pandas


# Float data and literals: a CASE or if() that mixes ints with floats gets a
# different dtype from pandas, which is not what these tests are about.
def _frame():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "g": ["x", "y", "x", "y"]})


@pytest.fixture(params=["dataframe", "parquet"])
def make_ds(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(_frame())
    path = str(tmp_path / "frame.parquet")
    _frame().to_parquet(path)
    return lambda: DataStore.from_file(path)


WINDOWS = {
    "cumsum": lambda s: s["a"].cumsum(),
    "shift": lambda s: s["a"].shift(1),
    "rank": lambda s: s["a"].rank(method="first"),
}


def _ds_when(d, value):
    return d.when(d["a"] > 1, value).otherwise(0.0)


def _pd_when(p, value):
    return pd.Series(0.0, index=p.index).case_when([(p["a"] > 1, value)])


CHAINS = {
    "when_value": (lambda d, w: _ds_when(d, w(d)), lambda p, w: _pd_when(p, w(p))),
    "when_condition": (
        lambda d, w: d.when(w(d) > 2, 1.0).otherwise(0.0),
        lambda p, w: pd.Series(0.0, index=p.index).case_when([(w(p) > 2, 1.0)]),
    ),
    "case_when": (
        lambda d, w: d["a"].case_when([(d["a"] > 1, w(d))]),
        lambda p, w: p["a"].case_when([(p["a"] > 1, w(p))]),
    ),
    "where": (
        lambda d, w: d["a"].where(d["a"] > 1, w(d)),
        lambda p, w: p["a"].where(p["a"] > 1, w(p)),
    ),
    "mask": (
        lambda d, w: d["a"].mask(d["a"] > 1, w(d)),
        lambda p, w: p["a"].mask(p["a"] > 1, w(p)),
    ),
}


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_built_before_filter_assigned_after(make_ds, chain, window):
    build_ds, build_pd = CHAINS[chain]
    w = WINDOWS[window]

    d = make_ds()
    value = build_ds(d, w)
    filtered = d[d["a"] > 2]
    filtered["c"] = value

    p = _frame()
    pd_value = build_pd(p, w)
    pd_filtered = p[p["a"] > 2].copy()
    pd_filtered["c"] = pd_value

    assert_datastore_equals_pandas(filtered, pd_filtered)


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("chain", sorted(CHAINS))
def test_assigned_to_its_own_frame(chain, window):
    build_ds, build_pd = CHAINS[chain]
    w = WINDOWS[window]

    d = DataStore(_frame())
    d["c"] = build_ds(d, w)

    p = _frame()
    p["c"] = build_pd(p, w)

    assert_datastore_equals_pandas(d, p)


def test_macroscope_chain(make_ds):
    """CASE WHEN over d['a'].cumsum(), filtered, then assigned."""
    d = make_ds()
    case = d.when(d["a"] > 0, d["a"].cumsum()).otherwise(0.0)
    filtered = d[d["a"] > 1]
    filtered["c"] = case

    p = _frame()
    pd_case = pd.Series(0.0, index=p.index).case_when([(p["a"] > 0, p["a"].cumsum())])
    pd_filtered = p[p["a"] > 1].copy()
    pd_filtered["c"] = pd_case

    assert_datastore_equals_pandas(filtered, pd_filtered)


class TestContainersSeeEveryChild:
    """is_row_wise / columns_read for each node that keeps children in its own attributes."""

    @pytest.fixture
    def ds(self):
        return DataStore(_frame())

    @staticmethod
    def window(ds):
        expr = WindowFunction("sum", Field("a")).over(order_by=[Field("_row_id")])
        return ColumnExpr(expr=expr, datastore=ds)

    def test_case_value(self, ds):
        case = CaseWhenExpr([(ds["a"] > 1, self.window(ds))], 0, ds)
        assert not case.is_row_wise
        assert case.columns_read() == {"a"}

    def test_case_condition(self, ds):
        case = CaseWhenExpr([(self.window(ds) > 2, 1)], 0, ds)
        assert not case.is_row_wise
        assert case.columns_read() == {"a"}

    def test_case_default(self, ds):
        case = CaseWhenExpr([(ds["g"] == "x", 1)], self.window(ds), ds)
        assert not case.is_row_wise
        assert case.columns_read() == {"a", "g"}

    def test_case_lazy_condition(self, ds):
        case = CaseWhenExpr([(ds["g"].isin(["x"]), ds["a"] + 1)], 0, ds)
        assert isinstance(case._cases[0][0], LazyCondition)
        assert case.is_row_wise
        assert case.columns_read() == {"a", "g"}

    def test_case_value_computed_in_pandas(self, ds):
        pandas_value = ColumnExpr(source=ds["a"], method_name="cummax")
        case = CaseWhenExpr([(ds["a"] > 1, pandas_value)], 0, ds)
        assert not case.is_row_wise
        assert case.columns_read() is None
        assert case.contains_pandas_fallback()
        with pytest.raises(NotImplementedError):
            case.to_sql()

    def test_case_row_wise(self, ds):
        case = CaseWhenExpr([(ds["a"] > 1, ds["a"] * 2)], ds["g"], ds)
        assert case.is_row_wise
        assert case.columns_read() == {"a", "g"}
        assert not case.contains_pandas_fallback()

    def test_in_list(self):
        cond = InCondition(Field("a"), [1, 2, Field("b")])
        assert cond.is_row_wise
        assert cond.columns_read() == {"a", "b"}

    def test_in_subquery(self, ds):
        cond = InCondition(Field("a"), ds.select("a"))
        assert cond.is_row_wise
        assert cond.columns_read() is None
        assert any(isinstance(node, OpaqueValue) and node.subquery for node in cond.subtree())

    def test_between_bounds(self, ds):
        cond = BetweenCondition(Field("a"), Literal(0), self.window(ds)._expr)
        assert not cond.is_row_wise
        assert cond.columns_read() == {"a"}

    def test_function_arguments(self, ds):
        func = Function("coalesce", Field("b"), self.window(ds))
        assert not func.is_row_wise
        assert func.columns_read() == {"a", "b"}

    def test_function_keyword_arguments(self, ds):
        func = Function("plus", Field("b"), Literal(1), pandas_kwargs={"other": self.window(ds)})
        assert not func.is_row_wise
        assert func.columns_read() == {"a", "b"}

    def test_window_clause(self):
        func = WindowFunction("row_number").over(
            partition_by=[Field("g")], order_by=[OrderByItem(Field("b"), descending=True)]
        )
        assert not func.is_row_wise
        assert func.columns_read() == {"g", "b"}

    def test_lazy_value_without_expression_in_function(self, ds):
        pandas_value = ColumnExpr(source=ds["a"], method_name="cummax")
        with pytest.raises(TypeError, match="computed in pandas"):
            Function("coalesce", Field("a"), pandas_value)
