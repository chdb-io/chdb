"""
Column assignment and repeated selection on top of an explicit projection.

Each chain runs unchanged on a pandas DataFrame and on a DataStore built from
a file source or a DataFrame source, and the full results are compared.
"""

import time

import pandas as pd
import pytest

from datastore import DataStore, F
from datastore.expressions import Field, resolve_computed_references
from datastore.functions import CastFunction

from tests.test_utils import assert_datastore_equals_pandas


pytestmark = pytest.mark.filterwarnings(
    "ignore:\\s*A value is trying to be set on a copy"
)

SOURCES = ["file", "dataframe"]


def _sales():
    return pd.DataFrame(
        {
            "region": ["a", "b", "a", "c", "b"],
            "price": [1, 2, 3, 4, 5],
            "qty": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture(params=SOURCES)
def make_ds(request, tmp_path):
    def make(df):
        if request.param == "file":
            path = tmp_path / "data.parquet"
            df.to_parquet(path)
            return DataStore.from_file(str(path))
        return DataStore(df.copy())

    return request.param, make


def _assert_chain(make_ds, chain, df=None):
    source, make = make_ds
    df = _sales() if df is None else df
    pd_result = chain(df.copy())
    ds_result = chain(make(df))
    # A file source renumbers rows after a filter; only the DataFrame
    # source keeps the pandas index.
    assert_datastore_equals_pandas(
        ds_result, pd_result, check_index=source == "dataframe"
    )


class TestOverwriteAfterProjection:
    """Issue #669: overwriting a selected column must replace it in place."""

    def test_issue_669_overwrite_selected_column(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d["price"] = d["price"] * 2
            return d

        _assert_chain(make_ds, chain)

    def test_issue_669_to_arrow_has_single_price_column(self, make_ds):
        _, make = make_ds
        ds = make(_sales())[["region", "price"]]
        ds["price"] = ds["price"] * 2

        table = ds.to_arrow()

        assert table.column_names == ["region", "price"]
        assert table.column("price").to_pylist() == [2, 4, 6, 8, 10]

    def test_overwrite_in_reordered_projection_then_new_column(self, make_ds):
        def chain(d):
            d = d[["price", "region"]]
            d["price"] = d["price"] * 2
            d["total"] = d["price"] + 1
            return d

        _assert_chain(make_ds, chain)

    def test_overwrite_then_sort(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d["price"] = d["price"] * -2
            return d.sort_values("price")

        _assert_chain(make_ds, chain)

    def test_overwrite_then_head(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d["price"] = d["price"] * 2
            return d.head(3)

        _assert_chain(make_ds, chain)

    def test_overwrite_then_filter_on_overwritten_column(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d["price"] = d["price"] * 2
            return d[d["price"] > 5]

        _assert_chain(make_ds, chain)

    def test_filter_then_overwrite(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d = d[d["price"] > 1]
            d["price"] = d["price"] * 2
            return d

        _assert_chain(make_ds, chain)

    def test_head_then_overwrite(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d = d.head(3)
            d["price"] = d["price"] * 2
            return d

        _assert_chain(make_ds, chain)

    def test_head_projection_overwrite_filter(self, make_ds):
        def chain(d):
            d = d.head(4)
            d = d[["region", "price"]]
            d["price"] = d["price"] * 2
            return d[d["price"] > 2]

        _assert_chain(make_ds, chain)

    def test_overwrite_head_then_filter(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d["price"] = d["price"] * 2
            d = d.head(4)
            return d[d["price"] > 2]

        _assert_chain(make_ds, chain)


class TestAssignAfterProjection:
    """``assign()`` must build on the current columns, not the source's."""

    def test_assign_overwrites_selected_column(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            return d.assign(price=d["price"] * 2)

        _assert_chain(make_ds, chain)

    def test_assign_overwrite_and_new_column(self, make_ds):
        def chain(d):
            d = d[["price", "qty"]]
            return d.assign(price=d["price"] * 2, w=d["qty"] - 1)

        _assert_chain(make_ds, chain)

    def test_assign_new_column_then_filter(self, make_ds):
        def chain(d):
            d = d[["region", "price"]]
            d = d.assign(n=d["price"] + 100)
            return d[d["price"] > 1]

        _assert_chain(make_ds, chain)

    def test_assign_after_drop(self, make_ds):
        def chain(d):
            d = d.drop(columns=["qty"])
            return d.assign(price=d["price"] * 2)

        _assert_chain(make_ds, chain)

    def test_assign_after_rename(self, make_ds):
        def chain(d):
            d = d.rename(columns={"qty": "q"})
            return d.assign(price=d["price"] * 2)

        _assert_chain(make_ds, chain)

    def test_assign_after_setitem(self, make_ds):
        def chain(d):
            d["x"] = d["price"] * 2
            return d.assign(y=d["qty"] + 1)

        _assert_chain(make_ds, chain)

    def test_assign_overwrites_two_columns_in_sequence(self, make_ds):
        def chain(d):
            d = d.assign(price=d["price"] * 2)
            return d.assign(qty=d["qty"] * 3)

        _assert_chain(make_ds, chain)

    def test_projection_of_assigned_column(self, make_ds):
        def chain(d):
            d = d.assign(z=d["price"] + 100)
            return d[["z", "region"]]

        _assert_chain(make_ds, chain)

    def test_sort_by_assigned_column_then_projection(self, make_ds):
        def chain(d):
            d = d.assign(x=d["price"] * -2)
            return d.sort_values("x")[["region"]]

        _assert_chain(make_ds, chain)

    def test_overwrite_assigned_column_after_projection(self, make_ds):
        def chain(d):
            d = d.assign(z=d["price"] + 100)
            d = d[["z", "region"]]
            d["z"] = d["z"] * 10
            return d

        _assert_chain(make_ds, chain)


class TestAssignAfterSetitem:
    """``assign()`` reads the columns as left by earlier ``ds[col] = ...``."""

    def test_assign_overwrites_column_set_by_setitem(self, make_ds):
        def chain(d):
            d["price"] = d["price"] * 2
            return d.assign(price=d["price"] + 1)

        _assert_chain(make_ds, chain)

    def test_assign_reads_column_overwritten_by_setitem(self, make_ds):
        def chain(d):
            d["price"] = d["price"] * 2
            return d.assign(t=d["price"] + 1)

        _assert_chain(make_ds, chain)

    def test_assign_reads_column_overwritten_after_projection(self, make_ds):
        def chain(d):
            d = d[["price", "qty"]]
            d["price"] = d["price"] * 10
            return d.assign(t=d["price"] + d["qty"])

        _assert_chain(make_ds, chain)

    def test_assign_reads_column_set_twice_by_setitem(self, make_ds):
        def chain(d):
            d["x"] = d["price"] * 2
            d["x"] = d["x"] + 1
            return d.assign(t=d["x"] * 10)

        _assert_chain(make_ds, chain)

    def test_setitem_reads_column_before_it_is_overwritten(self, make_ds):
        def chain(d):
            d["x"] = d["price"] * 2
            d["price"] = d["price"] + 100
            d["t"] = d["x"] + d["price"]
            return d

        _assert_chain(make_ds, chain)

    def test_setitem_flag_from_setitem_column_then_filter(self, make_ds):
        def chain(d):
            d["x"] = d["price"] * 2
            d["f"] = d["x"] > 5
            return d[d["qty"] > 15]

        _assert_chain(make_ds, chain)

    def test_assign_overwrite_read_by_same_call(self, make_ds):
        def chain(d):
            return d.assign(price=d["price"] * 2, z=d["price"] + 0)

        _assert_chain(make_ds, chain)

    def test_assign_overwrites_assigned_column_twice(self, make_ds):
        def chain(d):
            d = d.assign(a=d["price"] + 1)
            d = d.assign(a=d["a"] * 2)
            return d.assign(a=d["a"] - 1)

        _assert_chain(make_ds, chain)

    def test_assign_overwrites_then_groupby(self, make_ds):
        def chain(d):
            d = d.assign(price=d["price"] * 2)
            d = d.assign(price=d["price"] + 1)
            return d.groupby("region").agg({"price": "sum"})

        _, make = make_ds
        df = _sales()
        assert_datastore_equals_pandas(
            chain(make(df)), chain(df.copy()), check_index=True
        )


class TestOverwriteAfterFilter:
    """An overwrite must not change what an earlier filter or sort saw."""

    def test_filter_overwrite_filter(self, make_ds):
        def chain(d):
            d = d[d["price"] > 1]
            d["price"] = d["price"] * 3
            return d[d["price"] > 9]

        _assert_chain(make_ds, chain)

    def test_filter_projection_overwrite_filter(self, make_ds):
        def chain(d):
            d = d[d["price"] > 1][["region", "price"]]
            d["price"] = d["price"] * 3
            return d[d["price"] > 9]

        _assert_chain(make_ds, chain)

    def test_filter_head_overwrite_filter(self, make_ds):
        def chain(d):
            d = d[d["price"] > 1].head(4)
            d["price"] = d["price"] * 3
            return d[d["price"] > 9]

        _assert_chain(make_ds, chain)

    def test_filter_head_projection_overwrite_filter(self, make_ds):
        def chain(d):
            d = d[d["price"] > 1].head(4)[["region", "price"]]
            d["price"] = d["price"] * 3
            return d[d["price"] > 9]

        _assert_chain(make_ds, chain)

    def test_sort_then_overwrite_sort_column(self, make_ds):
        def chain(d):
            d = d.sort_values("qty", ascending=False)
            d["qty"] = d["qty"] * -1
            return d

        _assert_chain(make_ds, chain)


class TestRepeatedProjection:
    """Issue #670: selecting columns again must not duplicate them."""

    def test_issue_670_select_same_columns_twice_after_filter_and_head(
        self, make_ds
    ):
        df = pd.DataFrame(
            {"cat": ["A", "B", "C"], "sub": ["x", "y", "z"], "v": [5, 7, 9]}
        )

        def chain(d):
            d = d[d["v"] > 0]
            d = d.head(1)
            d = d[["cat", "v"]]
            d = d[["cat", "v"]]
            return d[d["v"] > 0]

        _assert_chain(make_ds, chain, df)

    def test_same_projection_twice(self, make_ds):
        _assert_chain(make_ds, lambda d: d[["region", "price"]][["region", "price"]])

    def test_narrowing_reordering_projection(self, make_ds):
        _assert_chain(
            make_ds, lambda d: d[["region", "price", "qty"]][["price", "region"]]
        )

    def test_head_then_narrowing_projection(self, make_ds):
        def chain(d):
            d = d.head(3)
            d = d[["region", "price"]]
            return d[["price"]]

        _assert_chain(make_ds, chain)

    def test_filter_head_projection_twice_filter(self, make_ds):
        def chain(d):
            d = d[d["price"] > 0]
            d = d.head(3)
            d = d[["region", "price"]]
            d = d[["region", "price"]]
            return d[d["price"] > 0]

        _assert_chain(make_ds, chain)

    def test_filter_head_narrowing_projection_filter(self, make_ds):
        def chain(d):
            d = d[d["price"] > 0]
            d = d.head(3)
            d = d[["region", "price"]]
            d = d[["price"]]
            return d[d["price"] > 1]

        _assert_chain(make_ds, chain)

    def test_head_projection_twice_filter(self, make_ds):
        def chain(d):
            d = d.head(3)
            d = d[["region", "price"]]
            d = d[["region", "price"]]
            return d[d["price"] > 1]

        _assert_chain(make_ds, chain)

    def test_filter_head_reordered_projections(self, make_ds):
        def chain(d):
            d = d[d["price"] > 1]
            d = d.head(2)
            d = d[["price", "region"]]
            return d[["region", "price"]]

        _assert_chain(make_ds, chain)

    def test_projection_head_reorder_filter(self, make_ds):
        def chain(d):
            d = d[["region", "price", "qty"]]
            d = d.head(3)
            d = d[["price", "region"]]
            return d[d["price"] > 1]

        _assert_chain(make_ds, chain)

    def test_head_filter_then_projection_without_filter_column(self, make_ds):
        def chain(d):
            d = d.head(3)
            d = d[d["price"] > 1]
            return d[["region"]]

        _assert_chain(make_ds, chain)


class TestComputedColumnsReadByGroupBy:
    """A GROUP BY layer inlines the computed columns it reads; the inlined
    expressions must keep their SQL meaning."""

    @staticmethod
    def _strings():
        return pd.DataFrame(
            {
                "region": ["a", "b", "a", "c"],
                "code": ["1", "2", "3", "4"],
                "tag": ["p", "q", "r", "s"],
            }
        )

    def test_string_concat_of_computed_column_then_groupby(self):
        def chain(d):
            d["label"] = d["tag"]
            d["key"] = d["label"] + d["code"]
            return d.groupby("region").agg({"key": "max"})

        df = self._strings()
        assert_datastore_equals_pandas(
            chain(DataStore(df.copy())), chain(df.copy()), check_index=True
        )

    def test_string_concat_of_computed_column_in_assign(self):
        def chain(d):
            d = d.assign(label=d["tag"])
            return d.assign(key=d["label"] + d["code"])

        df = self._strings()
        assert_datastore_equals_pandas(
            chain(DataStore(df.copy())), chain(df.copy()), check_index=True
        )

    def test_cast_of_computed_column_then_groupby(self, make_ds):
        source, make = make_ds
        df = self._strings()
        pd_result = df.copy()
        pd_result["copy"] = pd_result["code"]
        pd_result["n"] = pd_result["copy"].astype("int64")
        pd_result = pd_result.groupby("region").agg({"n": "sum"})

        ds = make(df)
        ds["copy"] = ds["code"]
        ds["n"] = F.cast(ds["copy"], "Int64")
        ds_result = ds.groupby("region").agg({"n": "sum"})

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_cast_of_computed_column_in_assign(self):
        df = self._strings()
        pd_result = df.assign(copy=df["code"])
        pd_result = pd_result.assign(n=pd_result["copy"].astype("int64"))

        ds = DataStore(df.copy())
        ds = ds.assign(copy=ds["code"])
        ds_result = ds.assign(n=F.cast(ds["copy"], "Int64"))

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_expanded_expression_keeps_subtype_and_state(self):
        computed = {"copy": Field("code")}
        cast = resolve_computed_references(
            F.cast(Field("copy"), "Int64", alias="n"), computed
        )
        assert isinstance(cast, CastFunction)
        assert cast.to_sql(with_alias=True) == 'CAST("code" AS Int64) AS "n"'

        concat = Field("copy") + Field("tag")
        concat._is_string_type = True
        resolved = resolve_computed_references(concat, computed)
        assert resolved.to_sql() == 'concat("code","tag")'


def _assign_twice(d):
    d = d.assign(w=d["price"])
    return d.assign(n=d["w"] * 2)


def _setitem_then_assign(d):
    d["w"] = d["price"] + 1
    return d.assign(n=d["w"] * 2)


def _assign_three_times(d):
    d = d.assign(w=d["price"])
    d = d.assign(n=d["w"] * 2)
    return d.assign(m=d["n"] + d["w"])


def _assign_twice_then_filter(d):
    d = _assign_twice(d)
    return d[d["n"] > 3]


def _head_then_filter(d):
    d = d.head(4)
    return d[d["qty"] > 15]


def _projection_then_filter(d):
    d = d[["region", "price"]]
    return d[d["price"] > 1]


def _sort_head_then_filter(d):
    d = d.sort_values("qty", ascending=False).head(3)
    return d[d["price"] > 2]


def _offset(d):
    return d[2:]


METADATA_CHAINS = [
    _assign_twice,
    _setitem_then_assign,
    _assign_three_times,
    _assign_twice_then_filter,
    _head_then_filter,
    _projection_then_filter,
    _sort_head_then_filter,
    _offset,
]


@pytest.mark.parametrize("chain", METADATA_CHAINS, ids=lambda c: c.__name__.strip("_"))
class TestMetadataOfLazyPipeline:
    """Row counts, schema probes and SQL aggregates that wrap the pipeline in
    their own query must see every lazy op, like execution does."""

    @staticmethod
    def _pair(make_ds, chain):
        source, make = make_ds
        df = _sales()
        return source, chain(make(df)), chain(df.copy())

    def test_len_shape_size_empty(self, make_ds, chain):
        _, ds, pdf = self._pair(make_ds, chain)
        assert len(ds) == len(pdf)
        assert ds.shape == pdf.shape
        assert ds.size == pdf.size
        assert ds.empty == pdf.empty

    def test_columns_and_dtypes(self, make_ds, chain):
        _, ds, pdf = self._pair(make_ds, chain)
        assert list(ds.columns) == list(pdf.columns)
        pd.testing.assert_series_equal(ds.dtypes, pdf.dtypes)

    def test_count_and_nunique(self, make_ds, chain):
        _, ds, pdf = self._pair(make_ds, chain)
        pd.testing.assert_series_equal(ds.count(), pdf.count(), check_dtype=False)
        pd.testing.assert_series_equal(
            ds.nunique(), pdf.nunique(), check_dtype=False
        )

    def test_describe(self, make_ds, chain):
        _, ds, pdf = self._pair(make_ds, chain)
        assert_datastore_equals_pandas(ds.describe(), pdf.describe())

    def test_sample_with_random_state(self, make_ds, chain):
        source, ds, pdf = self._pair(make_ds, chain)
        n = min(2, len(pdf))
        assert_datastore_equals_pandas(
            ds.sample(n=n, random_state=0),
            pdf.sample(n=n, random_state=0),
            check_index=source == "dataframe",
        )

    def test_sample_rows_come_from_result(self, make_ds, chain):
        _, ds, pdf = self._pair(make_ds, chain)
        n = min(2, len(pdf))
        sampled = ds.sample(n=n)
        assert list(sampled.columns) == list(pdf.columns)
        rows = {tuple(r) for r in pdf.itertuples(index=False)}
        picked = [tuple(r) for r in sampled.itertuples(index=False)]
        assert len(picked) == n
        assert set(picked) <= rows

    def test_memory_usage_columns(self, make_ds, chain):
        _, ds, pdf = self._pair(make_ds, chain)
        assert list(ds.memory_usage(index=False).index) == list(
            pdf.memory_usage(index=False).index
        )


def _dependency_chain(d, length, shape):
    d["c0"] = d["price"] * 2
    for i in range(1, length):
        prev = d[f"c{i - 1}"]
        d[f"c{i}"] = prev * 2 if shape == "linear" else prev + prev
    return d


class TestLongDependencyChains:
    """Each assignment reads the column assigned just before it. Inlining
    aliases that reference aliases in one SELECT grows the query tree
    quadratically (linear chain) or exponentially (doubling chain)."""

    @pytest.mark.parametrize("length", [30, 40])
    @pytest.mark.parametrize("shape", ["linear", "doubling"])
    def test_chain_matches_pandas(self, make_ds, shape, length):
        _assert_chain(make_ds, lambda d: _dependency_chain(d, length, shape))

    @pytest.mark.parametrize("shape", ["linear", "doubling"])
    def test_chain_then_filter(self, make_ds, shape):
        def chain(d):
            d = _dependency_chain(d, 30, shape)
            return d[d["c29"] > 2**30]

        _assert_chain(make_ds, chain)

    @pytest.mark.parametrize("shape", ["linear", "doubling"])
    def test_chain_then_groupby(self, make_ds, shape):
        def chain(d):
            d = _dependency_chain(d, 30, shape)
            return d.groupby("region").agg({"c29": "sum", "qty": "max"})

        _, make = make_ds
        df = _sales()
        assert_datastore_equals_pandas(
            chain(make(df)), chain(df.copy()), check_index=True
        )

    @pytest.mark.parametrize("shape", ["linear", "doubling"])
    def test_layer_computes_only_from_inner_columns(self, tmp_path, shape):
        from datastore.query_planner import QueryPlanner

        path = tmp_path / "data.parquet"
        _sales().to_parquet(path)
        ds = _dependency_chain(DataStore.from_file(str(path)), 40, shape)

        planner = QueryPlanner()
        exec_plan = planner.plan_segments(
            ds._lazy_ops, has_sql_source=True, schema=ds.schema()
        )
        for segment in exec_plan.segments:
            for layer in segment.plan.layers:
                computed = set()
                for op in layer:
                    assert not planner._columns_read_by(op) & computed
                    computed |= set(planner._computed_columns_of(op))

    def test_planning_is_bounded(self, tmp_path):
        from datastore.query_planner import MAX_SQL_LAYERS, QueryPlanner

        path = tmp_path / "data.parquet"
        _sales().to_parquet(path)
        ds = _dependency_chain(DataStore.from_file(str(path)), 300, "doubling")

        start = time.perf_counter()
        exec_plan = QueryPlanner().plan_segments(
            ds._lazy_ops, has_sql_source=True, schema=ds.schema()
        )
        assert time.perf_counter() - start < 5
        assert all(seg.is_sql() for seg in exec_plan.segments)
        assert all(
            len(seg.plan.layers) <= MAX_SQL_LAYERS for seg in exec_plan.segments
        )

