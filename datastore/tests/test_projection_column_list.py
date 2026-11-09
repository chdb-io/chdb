"""
Column assignment and repeated selection on top of an explicit projection.

Each chain runs unchanged on a pandas DataFrame and on a DataStore built from
a file source or a DataFrame source, and the full results are compared.
"""

import pandas as pd
import pytest

from datastore import DataStore

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

    def test_overwrite_assigned_column_after_projection(self, make_ds):
        def chain(d):
            d = d.assign(z=d["price"] + 100)
            d = d[["z", "region"]]
            d["z"] = d["z"] * 10
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
