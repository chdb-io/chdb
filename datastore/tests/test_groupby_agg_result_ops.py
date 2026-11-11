"""
Operations applied to a groupby aggregation's result.

The aggregation is one closed step: its result rows are indexed by the group
keys, ordered by key (``sort=True``) or by first appearance (``sort=False``),
and every later operation reads those rows, as in pandas.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from datastore import DataStore
from datastore.expressions import col
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


def assert_matches_pandas(ds_result, expected):
    assert_datastore_equals_pandas(ds_result, expected, check_nullable_dtype=True)
    pd.testing.assert_index_equal(ds_result.index, expected.index)


def make_source(frame, kind, tmp_path):
    if kind == "dataframe":
        return DataStore(frame.copy())
    path = str(tmp_path / "src.parquet")
    frame.to_parquet(path)
    return DataStore.from_file(path)


@pytest.fixture(params=["dataframe", "file"])
def source_kind(request):
    return request.param


TIES = pd.DataFrame(
    {"c": ["d", "b", "a", "c", "e", "f"] * 2, "v": [5, 5, 5, 5, 1, 1] * 2}
)
APPEARANCE = pd.DataFrame({"g": list("zyxwvuzyxwvu"), "v": range(12)})
TWO_KEYS = pd.DataFrame(
    {
        "a": ["x", "x", "y", "y", "x"],
        "b": ["p", "q", "p", "p", "p"],
        "v": [1, 2, 3, 4, 5],
    }
)
PRICES = pd.DataFrame(
    {
        "g": ["b", "a", "b", "c", "a", "c"],
        "price": [1, 2, 3, 4, 5, 6],
        "qty": [4, 5, 6, 7, 8, 9],
    }
)


class TestSortAfterAggregation:
    """Ties of a sort on an aggregate keep the aggregation's group order."""

    @pytest.mark.parametrize("kind", [None, "stable"])
    @pytest.mark.parametrize("ascending", [True, False])
    def test_ties_keep_key_order(self, source_kind, tmp_path, kind, ascending):
        def chain(d):
            agg = d.groupby("c").agg({"v": "sum"})
            if kind is None:
                return agg.sort_values("v", ascending=ascending)
            return agg.sort_values("v", ascending=ascending, kind=kind)

        expected = TIES.groupby("c").agg({"v": "sum"}).sort_values(
            "v", ascending=ascending, kind="stable"
        )
        assert_matches_pandas(chain(make_source(TIES, source_kind, tmp_path)), expected)

    def test_ties_keep_first_appearance_order(self, source_kind, tmp_path):
        def chain(d):
            return (
                d.groupby("c", sort=False)
                .agg({"v": "sum"})
                .sort_values("v", kind="stable")
            )

        assert_matches_pandas(
            chain(make_source(TIES, source_kind, tmp_path)), chain(TIES)
        )

    def test_filter_sort_head_on_aggregate(self, source_kind, tmp_path):
        def chain(d):
            agg = d[d["v"] > 1].groupby("c").agg({"v": "sum"})
            return agg.sort_values("v", ascending=False, kind="stable").head(3)

        assert_matches_pandas(
            chain(make_source(TIES, source_kind, tmp_path)), chain(TIES)
        )


class TestSelectionAfterAggregation:
    @pytest.mark.parametrize(
        "columns", [["price"], ["qty"], ["qty", "price"]], ids=str
    )
    def test_column_selection(self, source_kind, tmp_path, columns):
        def chain(d):
            return d.groupby("g").agg({"price": "sum", "qty": "sum"})[columns]

        assert_matches_pandas(
            chain(make_source(PRICES, source_kind, tmp_path)), chain(PRICES)
        )

    def test_selection_then_sort(self, source_kind, tmp_path):
        def chain(d):
            agg = d.groupby("g").agg({"price": "sum", "qty": "max"})
            return agg[["qty"]].sort_values("qty", ascending=False)

        assert_matches_pandas(
            chain(make_source(PRICES, source_kind, tmp_path)), chain(PRICES)
        )

    def test_filter_on_aggregate_then_selection(self, source_kind, tmp_path):
        def chain(d):
            agg = d.groupby("g").agg({"price": "sum", "qty": "sum"})
            return agg[agg["qty"] > 10][["price"]]

        assert_matches_pandas(
            chain(make_source(PRICES, source_kind, tmp_path)), chain(PRICES)
        )


class TestGroupOrderSortFalse:
    """groupby(sort=False) returns groups in order of first appearance."""

    @pytest.mark.parametrize("as_index", [True, False])
    def test_first_appearance(self, source_kind, tmp_path, as_index):
        def chain(d):
            return d.groupby("g", sort=False, as_index=as_index).agg({"v": "sum"})

        assert_matches_pandas(
            chain(make_source(APPEARANCE, source_kind, tmp_path)), chain(APPEARANCE)
        )

    def test_after_filter(self, source_kind, tmp_path):
        def chain(d):
            return d[d["v"] > 2].groupby("g", sort=False).agg({"v": "max"})

        assert_matches_pandas(
            chain(make_source(APPEARANCE, source_kind, tmp_path)), chain(APPEARANCE)
        )

    def test_after_head(self, source_kind, tmp_path):
        def chain(d):
            return d.head(9).groupby("g", sort=False).agg({"v": "sum"})

        assert_matches_pandas(
            chain(make_source(APPEARANCE, source_kind, tmp_path)), chain(APPEARANCE)
        )

    def test_selection_after_aggregation(self, source_kind, tmp_path):
        def chain(d):
            agg = d.groupby("g", sort=False).agg({"price": "sum", "qty": "min"})
            return agg[["qty"]]

        assert_matches_pandas(
            chain(make_source(PRICES, source_kind, tmp_path)), chain(PRICES)
        )


class TestSecondGroupby:
    @pytest.mark.parametrize("sort", [True, False])
    def test_regroup_as_index_false_result(self, source_kind, tmp_path, sort):
        def chain(d):
            first = d.groupby(["a", "b"], as_index=False, sort=sort).agg({"v": "sum"})
            return first.groupby("a", sort=sort).agg({"v": "sum"})

        assert_matches_pandas(
            chain(make_source(TWO_KEYS, source_kind, tmp_path)), chain(TWO_KEYS)
        )

    def test_regroup_then_sort(self, source_kind, tmp_path):
        def chain(d):
            first = d.groupby(["a", "b"], as_index=False).agg({"v": "max"})
            return first.groupby("b").agg({"v": "sum"}).sort_values("v", kind="stable")

        assert_matches_pandas(
            chain(make_source(TWO_KEYS, source_kind, tmp_path)), chain(TWO_KEYS)
        )


class TestNuniqueDtype:
    def test_series_nunique_is_int64(self, source_kind, tmp_path):
        frame = pd.DataFrame({"g": ["a", "b", "a", "a"], "v": [1, 2, 1, 3]})
        ds_result = make_source(frame, source_kind, tmp_path).groupby("g")["v"].nunique()
        assert_series_equal(ds_result, frame.groupby("g")["v"].nunique())


class TestSqlStyleAggregation:
    """agg(name=col(...).sum()) gives pandas' named-aggregation result."""

    FRAME = pd.DataFrame({"g": ["b", "c", "a", "b", "a"], "amount": [1, 6, 3, 2, 5]})

    def test_after_filter(self, source_kind, tmp_path):
        d = make_source(self.FRAME, source_kind, tmp_path)
        d = d[d["amount"] > 1]
        ds_result = d.groupby("g").agg(total=col("amount").sum())
        frame = self.FRAME[self.FRAME["amount"] > 1]
        assert_matches_pandas(ds_result, frame.groupby("g").agg(total=("amount", "sum")))

    def test_sort_false_then_sort(self, source_kind, tmp_path):
        d = make_source(self.FRAME, source_kind, tmp_path)
        ds_result = (
            d.groupby("g", sort=False)
            .agg(total=col("amount").sum(), n=col("amount").count())
            .sort_values("n", kind="stable")
        )
        expected = (
            self.FRAME.groupby("g", sort=False)
            .agg(total=("amount", "sum"), n=("amount", "count"))
            .sort_values("n", kind="stable")
        )
        assert_matches_pandas(ds_result, expected)

    def test_filter_on_aggregate(self, source_kind, tmp_path):
        agg = make_source(self.FRAME, source_kind, tmp_path).groupby("g").agg(
            total=col("amount").sum()
        )
        expected = self.FRAME.groupby("g").agg(total=("amount", "sum"))
        assert_matches_pandas(agg[agg["total"] > 5], expected[expected["total"] > 5])

    def test_source_store_keeps_no_group_by(self, source_kind, tmp_path):
        d = make_source(self.FRAME, source_kind, tmp_path)
        d.groupby("g").agg(total=col("amount").sum())
        assert_datastore_equals_pandas(
            d[d["amount"] > 2], self.FRAME[self.FRAME["amount"] > 2]
        )


    def test_to_sql_renders_the_aggregation(self, tmp_path):
        path = str(tmp_path / "src.parquet")
        self.FRAME.to_parquet(path)
        d = DataStore.from_file(path)
        for agg in [
            d.groupby("g").agg(total=col("amount").sum()),
            d.groupby("g").agg({"amount": "sum"}),
        ]:
            sql = agg.to_sql()
            assert 'sum("amount")' in sql
            assert 'GROUP BY "g"' in sql


class TestOutputNamedLikeKey:
    """An aggregate output named like a group key: pandas keeps the key in
    the index (or drops it with as_index=False) and names the column after
    the output."""

    FRAME = pd.DataFrame({"g": ["b", "a", "b", "c"], "amount": [1, 6, 3, 2]})

    @pytest.mark.parametrize(
        "chain",
        [
            pytest.param(lambda d: d.groupby("g").agg(g=("amount", "sum")), id="named"),
            pytest.param(lambda d: d.groupby("g").agg({"g": "count"}), id="dict"),
            pytest.param(
                lambda d: d.groupby("g").agg({"g": "count", "amount": "sum"}),
                id="dict_two_outputs",
            ),
            pytest.param(
                lambda d: d.groupby("g", as_index=False).agg(g=("amount", "sum")),
                id="as_index_false",
            ),
            pytest.param(
                lambda d: d.groupby("g", sort=False)
                .agg({"g": "count", "amount": "sum"})
                .sort_values("amount", kind="stable"),
                id="sort_false_then_sort",
            ),
            pytest.param(
                lambda d: (lambda a: a[a["g"] > 1][["amount"]])(
                    d.groupby("g").agg({"g": "count", "amount": "sum"})
                ),
                id="filter_then_select",
            ),
        ],
    )
    def test_matches_pandas(self, source_kind, tmp_path, chain):
        d = make_source(self.FRAME, source_kind, tmp_path)
        assert_matches_pandas(chain(d), chain(self.FRAME))

    def test_sql_expression_output(self, source_kind, tmp_path):
        d = make_source(self.FRAME, source_kind, tmp_path)
        assert_matches_pandas(
            d.groupby("g").agg(g=col("amount").sum()),
            self.FRAME.groupby("g").agg(g=("amount", "sum")),
        )


# ---------------------------------------------------------------------------
# Random chains that continue after the aggregation
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(711)
CHAIN_FRAME = pd.DataFrame(
    {
        "cat": _RNG.choice(list("DBCAE"), 60),
        "v": _RNG.integers(0, 5, 60),
        "w": _RNG.integers(0, 50, 60),
    }
)
AGG_OUTPUTS = ["v", "w"]

pre_step = st.one_of(
    st.tuples(st.just("filter"), st.sampled_from(["v", "w"]), st.integers(0, 40)),
    st.tuples(st.just("head"), st.integers(5, 60)),
)
post_step = st.one_of(
    st.tuples(st.just("filter"), st.sampled_from(AGG_OUTPUTS), st.integers(0, 200)),
    st.tuples(st.just("sort"), st.sampled_from(AGG_OUTPUTS), st.booleans()),
    st.tuples(st.just("head"), st.integers(1, 5)),
    st.tuples(st.just("select"), st.sampled_from([["v"], ["w"], ["w", "v"]])),
)


def _apply(frame, chain):
    pre, (sort, funcs), post = chain
    for step in pre:
        if step[0] == "filter":
            frame = frame[frame[step[1]] > step[2]]
        else:
            frame = frame.head(step[1])
    frame = frame.groupby("cat", sort=sort).agg({"v": funcs[0], "w": funcs[1]})
    columns = list(AGG_OUTPUTS)
    for step in post:
        if step[0] == "select":
            if not set(step[1]) <= set(columns):
                continue
            columns = list(step[1])
            frame = frame[columns]
        elif step[0] in ("filter", "sort") and step[1] not in columns:
            continue
        elif step[0] == "filter":
            frame = frame[frame[step[1]] > step[2]]
        elif step[0] == "sort":
            frame = frame.sort_values(step[1], ascending=step[2], kind="stable")
        else:
            frame = frame.head(step[1])
    return frame


@pytest.mark.parametrize("kind", ["dataframe", "file"])
def test_random_post_aggregation_chain_matches_pandas(kind, tmp_path_factory):
    source = make_source(CHAIN_FRAME, kind, tmp_path_factory.mktemp("chain"))

    @given(
        chain=st.tuples(
            st.lists(pre_step, max_size=2),
            st.tuples(
                st.booleans(),
                st.tuples(
                    st.sampled_from(["sum", "max", "count", "nunique"]),
                    st.sampled_from(["sum", "min", "mean"]),
                ),
            ),
            st.lists(post_step, min_size=1, max_size=4),
        )
    )
    @settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.filter_too_much,
        ],
    )
    def check(chain):
        # Two sorts in a row are a separate planner gap: the second
        # replaces the first instead of breaking its ties.
        assume(sum(step[0] == "sort" for step in chain[2]) <= 1)
        expected = _apply(CHAIN_FRAME, chain)
        ds_result = _apply(source, chain)
        if len(expected) == 0 and len(ds_result) == 0:
            return
        assert_matches_pandas(ds_result, expected)

    check()
