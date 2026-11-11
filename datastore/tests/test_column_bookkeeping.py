"""
Each lazy op reads the columns its input has at that point of the pipeline,
as pandas does. Every case runs on a DataFrame source and on a file source;
each class names the issue it covers.
"""

import itertools

import chdb
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas

_counter = itertools.count()


@pytest.fixture(params=["dataframe", "file"])
def source(request, tmp_path):
    """Build a DataStore over ``df`` from the parametrized kind of source."""

    def make(df):
        if request.param == "dataframe":
            return DataStore(df.copy())
        path = str(tmp_path / f"t{next(_counter)}.parquet")
        df.to_parquet(path)
        return DataStore.from_file(path)

    return make


def _groups():
    return pd.DataFrame({"g": ["a", "b", "a"], "v": [1, 2, 3], "w": [4, 5, 6]})


def _sales():
    return pd.DataFrame(
        {"region": ["n", "s", "n", "s"], "price": [1, 2, 4, 5], "qty": [4, 5, 6, 7]}
    )


class TestGroupByAfterAssign:
    """#709"""

    def test_sum_keeps_columns_not_assigned(self, source):
        df = _groups()
        ds = source(df)
        assert_datastore_equals_pandas(
            ds.assign(t=ds.v * 2).groupby("g").sum(),
            df.assign(t=df.v * 2).groupby("g").sum(),
            check_index=True,
        )

    def test_mean_after_setitem(self, source):
        df = _groups()
        ds = source(df)
        df["t"] = df["v"] * 2
        ds["t"] = ds["v"] * 2
        assert_datastore_equals_pandas(
            ds.groupby("g").mean(), df.groupby("g").mean(), check_index=True
        )

    def test_sum_after_selection_and_assign(self, source):
        df = _groups()
        ds = source(df)
        assert_datastore_equals_pandas(
            ds[["g", "w"]].assign(t=ds.v + ds.w).groupby("g").sum(),
            df[["g", "w"]].assign(t=df.v + df.w).groupby("g").sum(),
            check_index=True,
        )

    def test_sum_after_selection_then_assign_reading_the_selection(self, source):
        df = _groups()
        ds = source(df)
        d = ds[["g", "v"]]
        e = df[["g", "v"]]
        assert_datastore_equals_pandas(
            d.assign(t=d["v"] + 1).groupby("g").sum(),
            e.assign(t=e["v"] + 1).groupby("g").sum(),
            check_index=True,
        )

    def test_sum_after_overwrite(self, source):
        df = _groups()
        ds = source(df)
        assert_datastore_equals_pandas(
            ds.assign(v=ds.v * 10, t=ds.w - ds.v).groupby("g").sum(),
            df.assign(v=df.v * 10, t=df.w - df.v).groupby("g").sum(),
            check_index=True,
        )


class TestWhereMaskColumns:
    """#714"""

    @pytest.mark.parametrize("method", ["mask", "where"])
    def test_after_column_selection(self, source, method):
        df = _sales()
        pd_sel = df[["region", "price"]]
        ds_sel = source(df)[["region", "price"]]
        assert_datastore_equals_pandas(
            getattr(ds_sel, method)(ds_sel["price"] > 3, 0),
            getattr(pd_sel, method)(pd_sel["price"] > 3, 0),
        )

    @pytest.mark.parametrize("method", ["mask", "where"])
    def test_after_assign(self, source, method):
        df = _sales()
        ds = source(df)
        pd_new = df.assign(total=df.price * df.qty)
        ds_new = ds.assign(total=ds.price * ds.qty)
        assert_datastore_equals_pandas(
            getattr(ds_new, method)(ds_new["price"] > 3, 0),
            getattr(pd_new, method)(pd_new["price"] > 3, 0),
        )

    def test_after_setitem_overwrite(self, source):
        df = _sales()
        ds = source(df)
        ds["price"] = ds["price"] * 2
        df["price"] = df["price"] * 2
        assert_datastore_equals_pandas(
            ds.mask(ds["price"] > 5, 0), df.mask(df["price"] > 5, 0)
        )

    def test_after_drop(self, source):
        df = _sales()
        pd_drop = df.drop(columns=["region"])
        ds_drop = source(df).drop(columns=["region"])
        assert_datastore_equals_pandas(
            ds_drop.mask(ds_drop["qty"] > 5, -1),
            pd_drop.mask(pd_drop["qty"] > 5, -1),
        )


class TestToSqlRendersPipeline:
    """#715: the default to_sql() renders the same plan as execution."""

    @staticmethod
    def _run(sql):
        return chdb.query(sql, "DataFrame")

    @staticmethod
    def _assert_rows(got: pd.DataFrame, expected: pd.DataFrame):
        assert list(got.columns) == list(expected.columns)
        assert got.values.tolist() == expected.values.tolist()

    @pytest.fixture
    def path(self, tmp_path):
        p = str(tmp_path / "t.parquet")
        pd.DataFrame(
            {"region": ["n", "s", "n"], "price": [1, 2, 4], "qty": [4, 5, 6]}
        ).to_parquet(p)
        return p

    def test_repeated_assign_after_selection(self, path):
        df = pd.read_parquet(path)
        d = DataStore.from_file(path)[["region", "price"]]
        d = d.assign(price=d["price"] * 2)
        d = d.assign(price=d["price"] + 1)
        e = df[["region", "price"]]
        e = e.assign(price=e["price"] * 2)
        e = e.assign(price=e["price"] + 1)
        assert len(d) == 3
        assert_datastore_equals_pandas(d, e)
        self._assert_rows(self._run(d.to_sql()), e)
        assert d.to_sql().startswith(d.to_sql(execution_format=True) + " SETTINGS ")

    def test_setitem(self, path):
        df = pd.read_parquet(path)
        ds = DataStore.from_file(path)
        ds["price"] = ds["price"] * 2
        df["price"] = df["price"] * 2
        self._assert_rows(self._run(ds.to_sql()), df)

    def test_loc_filter(self, path):
        df = pd.read_parquet(path)
        ds = DataStore.from_file(path)
        ds = ds.loc[ds["price"] > 1]
        self._assert_rows(self._run(ds.to_sql()), df.loc[df["price"] > 1])

    def test_sort_head(self, path):
        df = pd.read_parquet(path)
        ds = DataStore.from_file(path).sort_values("price", ascending=False).head(2)
        self._assert_rows(
            self._run(ds.to_sql()), df.sort_values("price", ascending=False).head(2)
        )

class TestAssignReadsInputColumns:
    """#717: every assign() value is computed from the frame before the call."""

    def test_swap(self, source):
        df = pd.DataFrame({"region": ["n", "s"], "price": [1, 2], "qty": [10, 20]})
        d = source(df)
        assert_datastore_equals_pandas(
            d.assign(price=d["qty"], qty=d["price"]),
            df.assign(price=df["qty"], qty=df["price"]),
        )

    def test_rotate_with_expressions(self, source):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [100, 200, 300]})
        d = source(df)
        assert_datastore_equals_pandas(
            d.assign(a=d["b"] + 1, b=d["c"] * 2, c=d["a"] - d["b"]),
            df.assign(a=df["b"] + 1, b=df["c"] * 2, c=df["a"] - df["b"]),
        )

    def test_swap_then_filter_and_sort(self, source):
        df = pd.DataFrame({"k": [3, 1, 2], "x": [1, 2, 3], "y": [30, 10, 20]})
        d = source(df)
        r = d.assign(x=d["y"], y=d["x"])
        r = r[r["x"] > 10].sort_values("y")
        e = df.assign(x=df["y"], y=df["x"])
        e = e[e["x"] > 10].sort_values("y")
        assert_datastore_equals_pandas(r, e)

    def test_swap_in_callables(self, source):
        df = pd.DataFrame({"region": ["n", "s"], "price": [1, 2], "qty": [10, 20]})
        d = source(df)
        # pandas calls each callable on the frame with the earlier
        # assignments applied, so the second callable reads the new price.
        assert_datastore_equals_pandas(
            d.assign(price=lambda x: x["qty"], qty=lambda x: x["price"] * 3),
            df.assign(price=lambda x: x["qty"], qty=lambda x: x["price"] * 3),
        )
