"""
Scenario 8: Multi-step Filter + Sort + Agg Interaction Verification (P1)

Verifies that complex multi-step operation chains produce correct results
when lazy ops interact — specifically:

1. Multi-filter + sort + dict agg
2. Filter + assign + groupby + agg
3. Dropna + assign + filter + sort + head
4. Lazy op merging correctness (WHERE/ORDER BY/LIMIT placement)
5. Mixed PANDAS_FILTER and SQL filter chains
"""

import pytest
import numpy as np
import pandas as pd
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
)
from tests.xfail_markers import chdb_category_to_object


# ============================================================================
# Shared test data
# ============================================================================


def make_sales_data():
    """Sales dataset for multi-step chain tests."""
    data = {
        "region": ["East", "West", "East", "West", "East", "West", "East", "West",
                    "North", "North", "South", "South"],
        "product": ["A", "A", "B", "B", "A", "B", "C", "C", "A", "B", "A", "C"],
        "revenue": [100, 200, 150, 300, 250, 50, 175, 225, 130, 310, 90, 400],
        "quantity": [10, 20, 15, 30, 25, 5, 17, 22, 13, 31, 9, 40],
        "discount": [0.1, 0.2, 0.0, 0.15, 0.05, 0.3, 0.1, 0.0, 0.2, 0.1, 0.25, 0.0],
    }
    return pd.DataFrame(data), DataStore(data)


def make_nullable_data():
    """Dataset with NaN values for dropna chain tests."""
    data = {
        "name": ["Alice", "Bob", None, "Dave", "Eve", None, "Grace", "Hank"],
        "score": [85.0, np.nan, 70.0, 90.0, np.nan, 60.0, 95.0, 80.0],
        "group": ["X", "X", "Y", "Y", "X", "Y", "X", "Y"],
        "rank": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    return pd.DataFrame(data), DataStore(data)


def make_numeric_data():
    """Larger numeric dataset for complex chains."""
    np.random.seed(42)
    n = 100
    data = {
        "x": np.random.randint(0, 50, n).tolist(),
        "y": np.random.randint(100, 200, n).tolist(),
        "z": np.random.choice(["alpha", "beta", "gamma"], n).tolist(),
        "w": np.random.uniform(0, 1, n).tolist(),
    }
    return pd.DataFrame(data), DataStore(data)


# ============================================================================
# Test Class 1: Multi-filter + Sort + Dict Agg
# ============================================================================


class TestMultiFilterSortAgg:
    """ds[cond1][cond2].sort_values([...]).agg({...})"""

    def test_two_filters_sort_dict_agg(self):
        """Two filters -> sort -> dict aggregation."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 100]
        pd_result = pd_result[pd_result["quantity"] > 10]
        pd_result = pd_result.sort_values("revenue")
        pd_result = pd_result.agg({"revenue": "sum", "quantity": "mean"})

        ds_result = ds_df[ds_df["revenue"] > 100]
        ds_result = ds_result[ds_result["quantity"] > 10]
        ds_result = ds_result.sort_values("revenue")
        ds_result = ds_result.agg({"revenue": "sum", "quantity": "mean"})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_filters_sort_multiple_columns_agg(self):
        """Three successive filters -> multi-column sort -> agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] >= 100]
        pd_result = pd_result[pd_result["discount"] < 0.25]
        pd_result = pd_result[pd_result["region"].isin(["East", "West"])]
        pd_result = pd_result.sort_values(["revenue", "quantity"], ascending=[True, False])
        pd_result = pd_result.agg({"revenue": "max", "quantity": "min", "discount": "mean"})

        ds_result = ds_df[ds_df["revenue"] >= 100]
        ds_result = ds_result[ds_result["discount"] < 0.25]
        ds_result = ds_result[ds_result["region"].isin(["East", "West"])]
        ds_result = ds_result.sort_values(["revenue", "quantity"], ascending=[True, False])
        ds_result = ds_result.agg({"revenue": "max", "quantity": "min", "discount": "mean"})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_agg_single_function(self):
        """Filter -> sort -> agg with single function string."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["region"] == "East"]
        pd_result = pd_result.sort_values("revenue", ascending=False)
        pd_result = pd_result[["revenue", "quantity"]].agg("sum")

        ds_result = ds_df[ds_df["region"] == "East"]
        ds_result = ds_result.sort_values("revenue", ascending=False)
        ds_result = ds_result[["revenue", "quantity"]].agg("sum")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_agg_multiple_funcs_per_column(self):
        """Filter -> sort -> agg with multiple functions per column."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["quantity"] > 10]
        pd_result = pd_result.sort_values("quantity")
        pd_result = pd_result.agg({"revenue": ["sum", "mean"], "quantity": ["min", "max"]})

        ds_result = ds_df[ds_df["quantity"] > 10]
        ds_result = ds_result.sort_values("quantity")
        ds_result = ds_result.agg({"revenue": ["sum", "mean"], "quantity": ["min", "max"]})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_descending_agg(self):
        """Filter -> descending sort -> agg verifies sort doesn't corrupt agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result.sort_values("revenue", ascending=False)
        pd_result = pd_result.agg({"revenue": "sum", "quantity": "sum"})

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result.sort_values("revenue", ascending=False)
        ds_result = ds_result.agg({"revenue": "sum", "quantity": "sum"})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_on_string_then_sort_then_agg(self):
        """String-based filter -> sort -> agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["product"] != "C"]
        pd_result = pd_result.sort_values("product")
        pd_result = pd_result.agg({"revenue": "mean", "discount": "sum"})

        ds_result = ds_df[ds_df["product"] != "C"]
        ds_result = ds_result.sort_values("product")
        ds_result = ds_result.agg({"revenue": "mean", "discount": "sum"})

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Test Class 2: Filter + Assign + GroupBy + Agg
# ============================================================================


class TestFilterAssignGroupbyAgg:
    """ds[cond].assign(new_col=expr).groupby('col').agg('sum')"""

    def test_filter_assign_lambda_groupby_sum(self):
        """Filter -> assign lambda col -> groupby -> sum."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 100]
        pd_result = pd_result.assign(net_revenue=lambda d: d["revenue"] * (1 - d["discount"]))
        pd_result = pd_result.groupby("region")["net_revenue"].sum()

        ds_result = ds_df[ds_df["revenue"] > 100]
        ds_result = ds_result.assign(net_revenue=lambda d: d["revenue"] * (1 - d["discount"]))
        ds_result = ds_result.groupby("region")["net_revenue"].sum()

        assert_series_equal(ds_result, pd_result, check_names=True)

    def test_filter_assign_scalar_groupby_mean(self):
        """Filter -> assign with scalar -> groupby -> mean."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["quantity"] >= 10]
        pd_result = pd_result.assign(bonus=10)
        pd_result = pd_result.groupby("product")["bonus"].mean()

        ds_result = ds_df[ds_df["quantity"] >= 10]
        ds_result = ds_result.assign(bonus=10)
        ds_result = ds_result.groupby("product")["bonus"].mean()

        assert_series_equal(ds_result, pd_result, check_names=True)

    def test_filter_assign_groupby_dict_agg(self):
        """Filter -> assign -> groupby -> dict agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["discount"] < 0.2]
        pd_result = pd_result.assign(total=lambda d: d["revenue"] * d["quantity"])
        pd_result = pd_result.groupby("region").agg({"total": "sum", "revenue": "mean"})

        ds_result = ds_df[ds_df["discount"] < 0.2]
        ds_result = ds_result.assign(total=lambda d: d["revenue"] * d["quantity"])
        ds_result = ds_result.groupby("region").agg({"total": "sum", "revenue": "mean"})

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_two_filters_assign_groupby_agg(self):
        """Two filters -> assign -> groupby -> agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result[pd_result["region"].isin(["East", "West"])]
        pd_result = pd_result.assign(adjusted=lambda d: d["revenue"] - d["discount"] * 100)
        pd_result = pd_result.groupby("product").agg({"adjusted": "sum", "quantity": "max"})

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result[ds_result["region"].isin(["East", "West"])]
        ds_result = ds_result.assign(adjusted=lambda d: d["revenue"] - d["discount"] * 100)
        ds_result = ds_result.groupby("product").agg({"adjusted": "sum", "quantity": "max"})

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_multiple_assign_groupby_sum(self):
        """Filter -> assign two cols -> groupby -> sum."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] >= 100]
        pd_result = pd_result.assign(
            rev_per_unit=lambda d: d["revenue"] / d["quantity"],
            high_value=lambda d: (d["revenue"] > 200).astype(int),
        )
        pd_result = pd_result.groupby("region")[["rev_per_unit", "high_value"]].sum()

        ds_result = ds_df[ds_df["revenue"] >= 100]
        ds_result = ds_result.assign(
            rev_per_unit=lambda d: d["revenue"] / d["quantity"],
            high_value=lambda d: (d["revenue"] > 200).astype(int),
        )
        ds_result = ds_result.groupby("region")[["rev_per_unit", "high_value"]].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_assign_groupby_count(self):
        """Filter -> assign -> groupby -> count (tests aggregation type variety)."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 100]
        pd_result = pd_result.assign(flag=1)
        pd_result = pd_result.groupby("product")["flag"].count()

        ds_result = ds_df[ds_df["revenue"] > 100]
        ds_result = ds_result.assign(flag=1)
        ds_result = ds_result.groupby("product")["flag"].count()

        assert_series_equal(ds_result, pd_result, check_names=True)


# ============================================================================
# Test Class 3: Dropna + Assign + Filter + Sort + Head
# ============================================================================


class TestDropnaAssignFilterSortHead:
    """ds.dropna().assign(new_col=expr)[cond].sort_values('col').head(10)"""

    def test_dropna_assign_filter_sort_head(self):
        """Dropna -> assign -> filter -> sort -> head."""
        pd_df, ds_df = make_nullable_data()

        pd_result = pd_df.dropna()
        pd_result = pd_result.assign(double_score=lambda d: d["score"] * 2)
        pd_result = pd_result[pd_result["score"] > 70]
        pd_result = pd_result.sort_values("score", ascending=False)
        pd_result = pd_result.head(3)

        ds_result = ds_df.dropna()
        ds_result = ds_result.assign(double_score=lambda d: d["score"] * 2)
        ds_result = ds_result[ds_result["score"] > 70]
        ds_result = ds_result.sort_values("score", ascending=False)
        ds_result = ds_result.head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset_assign_filter_sort(self):
        """Dropna on subset -> assign -> filter -> sort."""
        pd_df, ds_df = make_nullable_data()

        pd_result = pd_df.dropna(subset=["score"])
        pd_result = pd_result.assign(score_rank=lambda d: d["score"].rank())
        pd_result = pd_result[pd_result["score"] >= 70]
        pd_result = pd_result.sort_values("score_rank")

        ds_result = ds_df.dropna(subset=["score"])
        ds_result = ds_result.assign(score_rank=lambda d: d["score"].rank())
        ds_result = ds_result[ds_result["score"] >= 70]
        ds_result = ds_result.sort_values("score_rank")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_filter_sort_head_small_n(self):
        """Dropna -> filter -> sort -> head(2) - small head limit."""
        pd_df, ds_df = make_nullable_data()

        pd_result = pd_df.dropna()
        pd_result = pd_result[pd_result["group"] == "X"]
        pd_result = pd_result.sort_values("rank")
        pd_result = pd_result.head(2)

        ds_result = ds_df.dropna()
        ds_result = ds_result[ds_result["group"] == "X"]
        ds_result = ds_result.sort_values("rank")
        ds_result = ds_result.head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_assign_sort_head_all_rows(self):
        """Dropna -> assign -> sort -> head(100) (more than available rows)."""
        pd_df, ds_df = make_nullable_data()

        pd_result = pd_df.dropna()
        pd_result = pd_result.assign(adjusted=lambda d: d["score"] + d["rank"])
        pd_result = pd_result.sort_values("adjusted")
        pd_result = pd_result.head(100)

        ds_result = ds_df.dropna()
        ds_result = ds_result.assign(adjusted=lambda d: d["score"] + d["rank"])
        ds_result = ds_result.sort_values("adjusted")
        ds_result = ds_result.head(100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_multiple_assigns_filter_sort_head(self):
        """Dropna -> two assigns -> filter -> sort -> head."""
        pd_df, ds_df = make_nullable_data()

        pd_result = pd_df.dropna()
        pd_result = pd_result.assign(score_sq=lambda d: d["score"] ** 2)
        pd_result = pd_result.assign(combo=lambda d: d["score_sq"] / d["rank"])
        pd_result = pd_result[pd_result["combo"] > 100]
        pd_result = pd_result.sort_values("combo", ascending=False)
        pd_result = pd_result.head(5)

        ds_result = ds_df.dropna()
        ds_result = ds_result.assign(score_sq=lambda d: d["score"] ** 2)
        ds_result = ds_result.assign(combo=lambda d: d["score_sq"] / d["rank"])
        ds_result = ds_result[ds_result["combo"] > 100]
        ds_result = ds_result.sort_values("combo", ascending=False)
        ds_result = ds_result.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Test Class 4: SQL clause ordering (WHERE / ORDER BY / LIMIT)
# ============================================================================


class TestLazyOpMerging:
    """Verify correct SQL merging: WHERE, ORDER BY, LIMIT position."""

    def test_filter_sort_limit_order(self):
        """Filter -> sort -> head produces correct WHERE + ORDER BY + LIMIT."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df[pd_df["x"] > 20]
        pd_result = pd_result.sort_values(["y", "x"])
        pd_result = pd_result.head(10)

        ds_result = ds_df[ds_df["x"] > 20]
        ds_result = ds_result.sort_values(["y", "x"])
        ds_result = ds_result.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_merge_into_single_where(self):
        """Multiple filters should merge into a single WHERE with AND."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df[pd_df["x"] > 10]
        pd_result = pd_result[pd_result["x"] < 40]
        pd_result = pd_result[pd_result["y"] > 130]

        ds_result = ds_df[ds_df["x"] > 10]
        ds_result = ds_result[ds_result["x"] < 40]
        ds_result = ds_result[ds_result["y"] > 130]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_filter_sort_head(self):
        """Interleaved filter-sort-filter-sort-head chain."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df[pd_df["x"] > 10]
        pd_result = pd_result.sort_values("y")
        pd_result = pd_result[pd_result["y"] < 180]
        pd_result = pd_result.sort_values(["x", "y"], ascending=[False, True])
        pd_result = pd_result.head(15)

        ds_result = ds_df[ds_df["x"] > 10]
        ds_result = ds_result.sort_values("y")
        ds_result = ds_result[ds_result["y"] < 180]
        ds_result = ds_result.sort_values(["x", "y"], ascending=[False, True])
        ds_result = ds_result.head(15)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_after_sort_limits_correctly(self):
        """Sort -> head(5) gives top-5 sorted rows."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df.sort_values(["x", "y"], ascending=[False, True]).head(5)
        ds_result = ds_df.sort_values(["x", "y"], ascending=[False, True]).head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_filter_sort_head(self):
        """Column select -> filter -> sort -> head."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df[["x", "y", "z"]]
        pd_result = pd_result[pd_result["x"] > 25]
        pd_result = pd_result.sort_values("y", ascending=False)
        pd_result = pd_result.head(8)

        ds_result = ds_df[["x", "y", "z"]]
        ds_result = ds_result[ds_result["x"] > 25]
        ds_result = ds_result.sort_values("y", ascending=False)
        ds_result = ds_result.head(8)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_groupby_sort_head(self):
        """Filter -> groupby -> agg -> sort -> head."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result.groupby("region").agg({"revenue": "sum"}).reset_index()
        pd_result = pd_result.sort_values("revenue", ascending=False)
        pd_result = pd_result.head(3)

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result.groupby("region").agg({"revenue": "sum"}).reset_index()
        ds_result = ds_result.sort_values("revenue", ascending=False)
        ds_result = ds_result.head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Test Class 5: Mixed PANDAS_FILTER + SQL filter chains
# ============================================================================


class TestMixedFilterChains:
    """Verify PANDAS_FILTER and SQL filter interleaving."""

    def test_sql_filter_then_lambda_filter(self):
        """SQL-pushable filter followed by lambda (pandas) filter."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 100]
        pd_result = pd_result[pd_result.apply(lambda row: row["quantity"] > 10, axis=1)]

        ds_result = ds_df[ds_df["revenue"] > 100]
        ds_result = ds_result[ds_result.apply(lambda row: row["quantity"] > 10, axis=1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_lambda_filter_then_sql_filter(self):
        """Lambda (pandas) filter followed by SQL-pushable filter."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df.apply(lambda row: row["revenue"] > 100, axis=1)]
        pd_result = pd_result[pd_result["quantity"] >= 15]

        ds_result = ds_df[ds_df.apply(lambda row: row["revenue"] > 100, axis=1)]
        ds_result = ds_result[ds_result["quantity"] >= 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sql_filter_lambda_filter_sort_head(self):
        """SQL filter -> lambda filter -> sort -> head."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result[pd_result.apply(lambda r: len(r["product"]) == 1, axis=1)]
        pd_result = pd_result.sort_values("revenue", ascending=False)
        pd_result = pd_result.head(5)

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result[ds_result.apply(lambda r: len(r["product"]) == 1, axis=1)]
        ds_result = ds_result.sort_values("revenue", ascending=False)
        ds_result = ds_result.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_alternating_sql_pandas_filters(self):
        """SQL filter -> pandas filter -> SQL filter -> sort."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result[pd_result.apply(lambda r: r["product"] in ["A", "B"], axis=1)]
        pd_result = pd_result[pd_result["quantity"] > 10]
        pd_result = pd_result.sort_values("revenue")

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result[ds_result.apply(lambda r: r["product"] in ["A", "B"], axis=1)]
        ds_result = ds_result[ds_result["quantity"] > 10]
        ds_result = ds_result.sort_values("revenue")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_filter_assign_groupby(self):
        """SQL filter -> pandas filter -> assign -> groupby -> agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result[pd_result.apply(lambda r: r["discount"] < 0.2, axis=1)]
        pd_result = pd_result.assign(net=lambda d: d["revenue"] * (1 - d["discount"]))
        pd_result = pd_result.groupby("region")["net"].sum()

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result[ds_result.apply(lambda r: r["discount"] < 0.2, axis=1)]
        ds_result = ds_result.assign(net=lambda d: d["revenue"] * (1 - d["discount"]))
        ds_result = ds_result.groupby("region")["net"].sum()

        assert_series_equal(ds_result, pd_result, check_names=True)


# ============================================================================
# Test Class 6: Edge cases and complex combinations
# ============================================================================


class TestComplexChainEdgeCases:
    """Edge cases for multi-step operation chains."""

    def test_empty_result_after_filter_chain(self):
        """Multiple filters that produce empty result -> agg."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 9999]
        pd_result = pd_result.sort_values("revenue")
        pd_result = pd_result.agg({"revenue": "sum", "quantity": "sum"})

        ds_result = ds_df[ds_df["revenue"] > 9999]
        ds_result = ds_result.sort_values("revenue")
        ds_result = ds_result.agg({"revenue": "sum", "quantity": "sum"})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_after_filters_sort_head(self):
        """Filters produce single row -> sort -> head."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] == 400]
        pd_result = pd_result.sort_values("revenue")
        pd_result = pd_result.head(5)

        ds_result = ds_df[ds_df["revenue"] == 400]
        ds_result = ds_result.sort_values("revenue")
        ds_result = ds_result.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_filter_on_assigned_column(self):
        """Filter -> assign -> filter on the newly assigned column."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result.assign(efficiency=lambda d: d["revenue"] / d["quantity"])
        pd_result = pd_result[pd_result["efficiency"] > 10]
        pd_result = pd_result.sort_values("efficiency", ascending=False)

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result.assign(efficiency=lambda d: d["revenue"] / d["quantity"])
        ds_result = ds_result[ds_result["efficiency"] > 10]
        ds_result = ds_result.sort_values("efficiency", ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_category_to_object
    def test_assign_then_groupby_on_assigned_column(self):
        """Assign a category column, then groupby on it."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df.assign(
            rev_tier=lambda d: pd.cut(d["revenue"], bins=[0, 100, 200, 500], labels=["low", "mid", "high"])
        )
        pd_result = pd_result.groupby("rev_tier", observed=True)["quantity"].sum()

        ds_result = ds_df.assign(
            rev_tier=lambda d: pd.cut(d["revenue"], bins=[0, 100, 200, 500], labels=["low", "mid", "high"])
        )
        ds_result = ds_result.groupby("rev_tier", observed=True)["quantity"].sum()

        assert_series_equal(ds_result, pd_result, check_names=True, check_like=True)

    def test_full_chain_filter_dropna_assign_groupby_sort(self):
        """Full chain: filter -> dropna -> assign -> groupby -> agg -> sort."""
        pd_df, ds_df = make_nullable_data()

        pd_result = pd_df[pd_df["rank"] <= 7]
        pd_result = pd_result.dropna(subset=["name", "score"])
        pd_result = pd_result.assign(weighted=lambda d: d["score"] * d["rank"])
        pd_result = pd_result.groupby("group").agg({"weighted": "sum", "score": "mean"})
        pd_result = pd_result.sort_values("weighted", ascending=False)

        ds_result = ds_df[ds_df["rank"] <= 7]
        ds_result = ds_result.dropna(subset=["name", "score"])
        ds_result = ds_result.assign(weighted=lambda d: d["score"] * d["rank"])
        ds_result = ds_result.groupby("group").agg({"weighted": "sum", "score": "mean"})
        ds_result = ds_result.sort_values("weighted", ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_with_column_select_between_filters(self):
        """Filter -> select columns -> filter -> sort -> head."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df[pd_df["x"] > 10]
        pd_result = pd_result[["x", "y", "z"]]
        pd_result = pd_result[pd_result["y"] < 170]
        pd_result = pd_result.sort_values("x")
        pd_result = pd_result.head(10)

        ds_result = ds_df[ds_df["x"] > 10]
        ds_result = ds_result[["x", "y", "z"]]
        ds_result = ds_result[ds_result["y"] < 170]
        ds_result = ds_result.sort_values("x")
        ds_result = ds_result.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_sort_overrides_previous_sort(self):
        """Second sort_values should override the first sort order."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["revenue"] > 50]
        pd_result = pd_result.sort_values("revenue")
        pd_result = pd_result.sort_values("quantity", ascending=False)
        pd_result = pd_result.head(5)

        ds_result = ds_df[ds_df["revenue"] > 50]
        ds_result = ds_result.sort_values("revenue")
        ds_result = ds_result.sort_values("quantity", ascending=False)
        ds_result = ds_result.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_groupby_multiple_agg_reset_sort(self):
        """Filter -> groupby -> multi-column agg -> reset_index -> sort."""
        pd_df, ds_df = make_sales_data()

        pd_result = pd_df[pd_df["discount"] < 0.2]
        pd_result = pd_result.groupby("product").agg(
            {"revenue": "sum", "quantity": "mean", "discount": "max"}
        )
        pd_result = pd_result.reset_index()
        pd_result = pd_result.sort_values("revenue", ascending=False)

        ds_result = ds_df[ds_df["discount"] < 0.2]
        ds_result = ds_result.groupby("product").agg(
            {"revenue": "sum", "quantity": "mean", "discount": "max"}
        )
        ds_result = ds_result.reset_index()
        ds_result = ds_result.sort_values("revenue", ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all_strategy(self):
        """Dropna(how='all') -> filter -> sort."""
        data = {
            "a": [1, np.nan, 3, np.nan],
            "b": [np.nan, np.nan, 6, np.nan],
            "c": [7, 8, 9, np.nan],
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.dropna(how="all")
        pd_result = pd_result[pd_result["c"] > 7]
        pd_result = pd_result.sort_values("c")

        ds_result = ds_df.dropna(how="all")
        ds_result = ds_result[ds_result["c"] > 7]
        ds_result = ds_result.sort_values("c")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_long_chain_six_operations(self):
        """6-operation chain: filter -> assign -> filter -> sort -> head -> select cols."""
        pd_df, ds_df = make_numeric_data()

        pd_result = pd_df[pd_df["x"] > 10]
        pd_result = pd_result.assign(xy_ratio=lambda d: d["x"] / d["y"])
        pd_result = pd_result[pd_result["xy_ratio"] < 0.3]
        pd_result = pd_result.sort_values("xy_ratio")
        pd_result = pd_result.head(20)
        pd_result = pd_result[["x", "y", "xy_ratio"]]

        ds_result = ds_df[ds_df["x"] > 10]
        ds_result = ds_result.assign(xy_ratio=lambda d: d["x"] / d["y"])
        ds_result = ds_result[ds_result["xy_ratio"] < 0.3]
        ds_result = ds_result.sort_values("xy_ratio")
        ds_result = ds_result.head(20)
        ds_result = ds_result[["x", "y", "xy_ratio"]]

        assert_datastore_equals_pandas(ds_result, pd_result)
