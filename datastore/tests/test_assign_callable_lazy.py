"""
assign() with callables stays lazy (issue #672).

pandas calls each assign() callable with the frame. DataStore calls it with
the lazy DataStore, so ``lambda d: d['price'] * 2`` becomes a SQL expression
on the original source instead of materializing the pipeline during assign().
"""

import chdb
import numpy as np
import pandas as pd
import pytest

import datastore.core as core
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_frame_equal,
    get_dataframe,
)


@pytest.fixture
def execute_calls(monkeypatch):
    calls = []
    original = core.DataStore._execute

    def counting_execute(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(core.DataStore, "_execute", counting_execute)
    return calls


@pytest.fixture
def sales_parquet(tmp_path):
    path = str(tmp_path / "sales.parquet")
    pd.DataFrame(
        {
            "region": ["a", "b", "c", "d"],
            "price": [1, 2, 3, 4],
            "qty": [10, 30, 20, 40],
        }
    ).to_parquet(path)
    return path


def _frame():
    return pd.DataFrame(
        {
            "region": ["a", "b", "c", "d"],
            "price": [1, 2, 3, 4],
            "qty": [10, 30, 20, 40],
        }
    )


class TestAssignCallableFileSource:
    def test_issue_chain_is_lazy_and_renders_source_sql(
        self, sales_parquet, execute_calls
    ):
        ds_result = DataStore.from_file(sales_parquet).assign(
            price=lambda d: d["price"] * 2
        )
        assert len(execute_calls) == 0

        sql = ds_result.to_sql(execution_format=True)
        assert len(execute_calls) == 0
        assert "FROM file(" in sql

        pd_result = pd.read_parquet(sales_parquet).assign(
            price=lambda d: d["price"] * 2
        )
        assert_datastore_equals_pandas(ds_result, pd_result)
        assert_datastore_equals_pandas(chdb.query(sql, "DataFrame"), pd_result)

    def test_later_callable_sees_earlier_assignment(
        self, sales_parquet, execute_calls
    ):
        def chain(df):
            return df.assign(
                double=lambda d: d["price"] * 2,
                total=lambda d: d["double"] * d["qty"],
            )

        ds_result = chain(DataStore.from_file(sales_parquet))
        assert len(execute_calls) == 0

        pd_result = chain(pd.read_parquet(sales_parquet))
        sql = ds_result.to_sql(execution_format=True)
        assert_datastore_equals_pandas(ds_result, pd_result)
        assert_datastore_equals_pandas(chdb.query(sql, "DataFrame"), pd_result)

    @pytest.mark.parametrize(
        "chain",
        [
            pytest.param(
                lambda df: df.assign(
                    price=lambda d: d["price"] * 2,
                    total=lambda d: d["price"] * d["qty"],
                ),
                id="overwrite_then_reference",
            ),
            pytest.param(
                lambda df: df.assign(
                    price=lambda d: d["price"] * 2,
                    n=np.array([5, 6, 7, 8]),
                    total=lambda d: d["price"] * d["qty"],
                ),
                id="overwrite_array_reference",
            ),
            pytest.param(
                lambda df: df.assign(price=lambda d: d["price"] * 2)
                .assign(n=np.array([5, 6, 7, 8]))
                .assign(total=lambda d: d["price"] * d["qty"]),
                id="chained_overwrite_array_reference",
            ),
        ],
    )
    def test_callable_sees_overwritten_column(
        self, sales_parquet, execute_calls, chain
    ):
        ds_result = chain(DataStore.from_file(sales_parquet))
        assert len(execute_calls) == 0
        assert_datastore_equals_pandas(ds_result, chain(pd.read_parquet(sales_parquet)))

    def test_callable_then_expression_keeps_kwarg_order(
        self, sales_parquet, execute_calls
    ):
        ds = DataStore.from_file(sales_parquet)
        ds_result = ds.assign(a=lambda d: d["price"] * 2, b=ds["qty"] + 1)
        assert len(execute_calls) == 0

        pdf = pd.read_parquet(sales_parquet)
        pd_result = pdf.assign(a=lambda d: d["price"] * 2, b=pdf["qty"] + 1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_after_groupby_agg(self, sales_parquet, execute_calls):
        def chain(df):
            return (
                df.groupby("region")
                .agg(sp=("price", "sum"), sq=("qty", "sum"))
                .assign(r=lambda x: x["sq"] / x["sp"])
            )

        ds_result = chain(DataStore.from_file(sales_parquet))
        assert len(execute_calls) == 0
        assert_datastore_equals_pandas(ds_result, chain(pd.read_parquet(sales_parquet)))


class TestAssignCallableDataFrameSource:
    @pytest.mark.parametrize(
        "make_value",
        [
            pytest.param(lambda: (lambda d: pd.Series([7, 9, 8, 6])), id="series"),
            pytest.param(lambda: (lambda d: np.array([7, 9, 8, 6])), id="ndarray"),
            pytest.param(
                lambda: (lambda d: d["price"].apply(lambda v: v * 3 + 1)),
                id="apply",
            ),
            pytest.param(
                lambda: (lambda d: d["region"] + "_" + d["qty"].astype(str)),
                id="concat_astype",
            ),
            pytest.param(lambda: (lambda d: d["qty"] - d["qty"].cumsum()), id="cumsum"),
            pytest.param(lambda: np.array([7, 9, 8, 6]), id="plain_ndarray"),
            pytest.param(lambda: 5, id="scalar"),
        ],
    )
    def test_non_sql_values_are_lazy_and_match_pandas(self, make_value, execute_calls):
        ds_result = DataStore(_frame()).assign(new=make_value())
        assert len(execute_calls) == 0

        pd_result = _frame().assign(new=make_value())
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_scalar_then_callable_referencing_it(self, execute_calls):
        def chain(df):
            return df.assign(s=5, t=lambda d: d["s"] + d["price"])

        ds_result = chain(DataStore(_frame()))
        assert len(execute_calls) == 0
        assert_datastore_equals_pandas(ds_result, chain(_frame()))

    def test_comparison_callable_is_bool(self):
        ds_result = DataStore(_frame()).assign(flag=lambda d: d["price"] > 2)
        pd_result = _frame().assign(flag=lambda d: d["price"] > 2)
        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_boolean_logic_with_missing_values(self, execute_calls):
        data = {
            "a": [True, True, False, False, None],
            "b": [True, False, True, False, True],
        }
        ds_result = DataStore(data).assign(
            c_and=lambda x: x["a"] & x["b"], c_or=lambda x: x["a"] | x["b"]
        )
        assert len(execute_calls) == 0

        pd_result = pd.DataFrame(data).assign(
            c_and=lambda x: x["a"] & x["b"], c_or=lambda x: x["a"] | x["b"]
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_numpy_scalar_operand(self):
        ds = DataStore(_frame())
        pdf = _frame()
        ds_result = ds.assign(share=ds["qty"] / ds["qty"].max())
        pd_result = pdf.assign(share=pdf["qty"] / pdf["qty"].max())
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_callable_is_lazy(self, execute_calls):
        ds_rows = DataStore(_frame()).loc[lambda d: d["qty"] > 15]
        ds_cols = DataStore(_frame()).loc[lambda d: d["qty"] > 15, ["region", "qty"]]
        assert isinstance(ds_rows, DataStore)
        assert isinstance(ds_cols, DataStore)
        assert len(execute_calls) == 0

        assert_datastore_equals_pandas(ds_rows, _frame().loc[lambda d: d["qty"] > 15])
        assert_datastore_equals_pandas(
            ds_cols, _frame().loc[lambda d: d["qty"] > 15, ["region", "qty"]]
        )

    def test_loc_single_column_returns_series(self, execute_calls):
        ds = DataStore(_frame())
        ds_by_callable = ds.loc[lambda d: d["qty"] > 15, "qty"]
        ds_by_condition = ds.loc[ds["qty"] > 15, "qty"]
        assert not isinstance(ds_by_callable, DataStore)
        assert not isinstance(ds_by_condition, DataStore)
        assert len(execute_calls) == 0

        pdf = _frame()
        assert_datastore_equals_pandas(ds_by_callable, pdf.loc[lambda d: d["qty"] > 15, "qty"])
        assert_datastore_equals_pandas(ds_by_condition, pdf.loc[pdf["qty"] > 15, "qty"])

    @pytest.mark.parametrize("use_callable", [True, False], ids=["callable", "expression"])
    @pytest.mark.parametrize("agg", ["named", "sum"])
    def test_assign_after_groupby_aggregation(self, use_callable, agg):
        data = {"g": list("xyxyz"), "a": [1, 2, 3, 4, 9], "b": [10.0, 20.0, 30.0, 40.0, 5.0]}

        def chain(df):
            if agg == "named":
                grouped = df.groupby("g").agg(sa=("a", "sum"), sb=("b", "sum"))
                left, right = "sa", "sb"
            else:
                grouped = df.groupby("g").sum()
                left, right = "a", "b"
            if use_callable:
                return grouped.assign(r=lambda x: x[left] / x[right])
            return grouped.assign(r=grouped[left] / grouped[right])

        assert_datastore_equals_pandas(chain(DataStore(data)), chain(pd.DataFrame(data)))

    def test_setitem_after_groupby_sum(self):
        data = {"g": list("xyxyz"), "a": [1, 2, 3, 4, 9], "b": [10.0, 20.0, 30.0, 40.0, 5.0]}
        ds = DataStore(data).groupby("g").sum()
        ds["r"] = ds["a"] * 2
        pdf = pd.DataFrame(data).groupby("g").sum()
        pdf["r"] = pdf["a"] * 2
        assert_datastore_equals_pandas(ds, pdf)

    @pytest.mark.parametrize(
        "data, func",
        [
            pytest.param(
                {"a": pd.array([1, None, 3, 4], dtype="Int64")},
                lambda x: x["a"] > 1,
                id="int64_gt",
            ),
            pytest.param(
                {"a": pd.array([1, None, 3, 4], dtype="Int64")},
                lambda x: x["a"] != 1,
                id="int64_ne",
            ),
            pytest.param(
                {"a": pd.array([True, None, False, True], dtype="boolean")},
                lambda x: x["a"] == True,  # noqa: E712
                id="boolean_eq",
            ),
            pytest.param(
                {"a": [1.0, np.nan, 3.0, 4.0]},
                lambda x: x["a"] > 1,
                id="float_nan_gt",
            ),
        ],
    )
    def test_comparison_keeps_pandas_missing_semantics(self, data, func, execute_calls):
        ds_result = DataStore(data).assign(c=func)
        assert len(execute_calls) == 0
        assert_frame_equal(get_dataframe(ds_result), pd.DataFrame(data).assign(c=func))

    @pytest.mark.parametrize(
        "func",
        [
            pytest.param(lambda x: x["t"] > np.datetime64("2023-01-01"), id="gt_datetime64"),
            pytest.param(lambda x: x["t"] + np.timedelta64(1, "D"), id="add_timedelta64"),
            pytest.param(lambda x: x["t"] - pd.Timestamp("2022-01-01"), id="sub_timestamp"),
        ],
    )
    def test_temporal_literals(self, func):
        data = {"t": pd.to_datetime(["2022-06-01", "2023-06-01", "2024-01-15"])}
        assert_datastore_equals_pandas(
            DataStore(data).assign(c=func), pd.DataFrame(data).assign(c=func)
        )

    @pytest.mark.parametrize(
        "func",
        [
            pytest.param(lambda x: x["f"] ^ (x["a"].cumsum() > 3), id="xor"),
            pytest.param(lambda x: x["f"] & (x["a"].cumsum() > 3), id="and"),
            pytest.param(lambda x: x["f"] | (x["a"].cumsum() > 3), id="or"),
            pytest.param(lambda x: x["a"] == x["a"].cummax(), id="eq"),
        ],
    )
    def test_operator_with_method_mode_operand(self, func):
        data = {"a": [1, 5, 2, 7], "f": [True, False, False, True]}
        assert_datastore_equals_pandas(
            DataStore(data).assign(c=func), pd.DataFrame(data).assign(c=func)
        )

    @pytest.mark.parametrize(
        "func",
        [
            pytest.param(lambda x: x.assign(z=x["price"] + 1)["z"], id="derived_column"),
            pytest.param(lambda x: x[x["qty"] > 15]["price"], id="filtered_rows"),
        ],
    )
    def test_column_of_another_datastore(self, func, execute_calls):
        ds_result = DataStore(_frame()).assign(c=func)
        assert len(execute_calls) == 0
        assert_datastore_equals_pandas(ds_result, _frame().assign(c=func))

    def test_callable_that_needs_a_pandas_dataframe(self, execute_calls):
        def row_total(d):
            return pd.DataFrame.sum(d[["price", "qty"]], axis=1)

        ds_result = DataStore(_frame()).assign(total=row_total)
        assert len(execute_calls) == 0
        assert_datastore_equals_pandas(ds_result, _frame().assign(total=row_total))


class TestColumnAssignmentSQL:
    def test_setitem_comparison_matches_pandas_dtype(self):
        ds = DataStore(_frame())
        ds["flag"] = ds["price"] > 2
        pdf = _frame()
        pdf["flag"] = pdf["price"] > 2
        assert_frame_equal(get_dataframe(ds), pdf)

    def test_setitem_nullable_comparison_keeps_na(self):
        data = {"a": pd.array([1, None, 3, 4], dtype="Int64")}
        ds = DataStore(data)
        ds["c"] = ds["a"] > 1
        pdf = pd.DataFrame(data)
        pdf["c"] = pdf["a"] > 1
        assert_frame_equal(get_dataframe(ds), pdf)

    def test_floordiv_and_pow_keep_int_dtype(self):
        ds = DataStore(_frame())
        pdf = _frame()
        ds_result = ds.assign(q=ds["qty"] // ds["price"], p=ds["price"] ** 2)
        pd_result = pdf.assign(q=pdf["qty"] // pdf["price"], p=pdf["price"] ** 2)
        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_assign_after_setitem_array_on_overwritten_column(self, sales_parquet):
        ds = DataStore.from_file(sales_parquet)
        ds = ds.assign(price=ds["price"] * 2)
        ds["n"] = np.array([5, 6, 7, 8])
        ds_result = ds.assign(total=ds["price"] * ds["qty"])

        pdf = pd.read_parquet(sales_parquet)
        pdf = pdf.assign(price=pdf["price"] * 2)
        pdf["n"] = np.array([5, 6, 7, 8])
        pd_result = pdf.assign(total=pdf["price"] * pdf["qty"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_selection_after_pandas_assignment_keeps_earlier_columns(self):
        def chain(df):
            df = df.assign(ratio=df["qty"] / df["price"])
            df = df.assign(keep=np.array([True, False, True, True]))
            return df[df["keep"]][["region", "ratio"]]

        assert_datastore_equals_pandas(chain(DataStore(_frame())), chain(_frame()))

    def test_assign_after_setitem_reads_columns_before_the_call(self):
        def chain(df):
            df["price"] = df["price"] * 2
            return df.assign(price=df["price"] + 1, half=df["price"] / 2)

        assert_datastore_equals_pandas(chain(DataStore(_frame())), chain(_frame()))

    def test_assign_after_setitem_on_copy(self):
        def chain(df):
            df = df.copy()
            df["x"] = df["price"] * 2
            return df.assign(y=df["qty"] + 1)

        assert_datastore_equals_pandas(chain(DataStore(_frame())), chain(_frame()))
