"""
Exploratory Batch 96: Deep Operation Chains and SQL Merging

This batch focuses on testing complex operation chains (5+ operations)
and verifying SQL merging behavior:
1. Deep filter -> groupby -> agg -> sort -> head chains
2. Multiple consecutive filters (should merge into single WHERE)
3. Multiple consecutive assigns (should merge)
4. Mixed accessor chains
5. Nested operations with subqueries
6. Chain operations that force execution boundaries

Goal: Discover edge cases in lazy execution and SQL optimization.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_frame_equal,
    assert_series_equal,
    get_dataframe,
    get_series,
)


# =============================================================================
# 1. Deep Filter Chains
# =============================================================================


class TestDeepFilterChains:
    """Test multiple consecutive filter operations."""

    def test_two_consecutive_filters(self):
        """Two filters should produce same result as combined filter."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })

        # Two separate filters
        pd_result = pd_df[pd_df["a"] > 3]
        pd_result = pd_result[pd_result["b"] < 80]

        ds_result = ds_df[ds_df["a"] > 3]
        ds_result = ds_result[ds_result["b"] < 80]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_consecutive_filters(self):
        """Three filters should chain correctly."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "c": [True, False, True, False, True, False, True, False, True, False],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "c": [True, False, True, False, True, False, True, False, True, False],
        })

        pd_result = pd_df[pd_df["a"] > 2]
        pd_result = pd_result[pd_result["b"] == "a"]
        pd_result = pd_result[pd_result["c"] == True]

        ds_result = ds_df[ds_df["a"] > 2]
        ds_result = ds_result[ds_result["b"] == "a"]
        ds_result = ds_result[ds_result["c"] == True]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_filter_on_different_columns(self):
        """Filter on column A, then filter on column B."""
        pd_df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
            "z": ["p", "q", "r", "s", "t"],
        })
        ds_df = DataStore({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
            "z": ["p", "q", "r", "s", "t"],
        })

        pd_result = pd_df[pd_df["x"] > 2][pd_df["y"] < 4]
        ds_result = ds_df[ds_df["x"] > 2][ds_df["y"] < 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_resulting_in_empty_then_another_filter(self):
        """Filter to empty, then apply another filter."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        # First filter returns empty
        pd_result = pd_df[pd_df["a"] > 100]
        pd_result = pd_result[pd_result["b"] > 0]

        ds_result = ds_df[ds_df["a"] > 100]
        ds_result = ds_result[ds_result["b"] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0


# =============================================================================
# 2. Deep Assign Chains
# =============================================================================


class TestDeepAssignChains:
    """Test multiple consecutive assign operations."""

    @pytest.mark.xfail(reason="chDB pow() always returns Float64, need dtype correction layer")
    def test_multiple_new_columns(self):
        """Add multiple new columns in sequence."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(b=pd_df["a"] * 2)
        pd_result = pd_result.assign(c=pd_result["b"] + 1)
        pd_result = pd_result.assign(d=pd_result["c"] ** 2)

        ds_result = ds_df.assign(b=ds_df["a"] * 2)
        ds_result = ds_result.assign(c=ds_result["b"] + 1)
        ds_result = ds_result.assign(d=ds_result["c"] ** 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_overwrite_same_column_multiple_times(self):
        """Overwrite the same column multiple times."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(a=pd_df["a"] + 1)
        pd_result = pd_result.assign(a=pd_result["a"] * 2)
        pd_result = pd_result.assign(a=pd_result["a"] - 1)

        ds_result = ds_df.assign(a=ds_df["a"] + 1)
        ds_result = ds_result.assign(a=ds_result["a"] * 2)
        ds_result = ds_result.assign(a=ds_result["a"] - 1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_based_on_computed_column(self):
        """Assign new column based on previously computed column."""
        pd_df = pd.DataFrame({"x": [10, 20, 30]})
        ds_df = DataStore({"x": [10, 20, 30]})

        pd_result = pd_df.assign(
            y=pd_df["x"] * 2,
        ).assign(
            z=lambda df: df["y"] + df["x"],
        )

        ds_result = ds_df.assign(y=ds_df["x"] * 2)
        ds_result = ds_result.assign(z=ds_result["y"] + ds_result["x"])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 3. Filter + Groupby + Agg Chains
# =============================================================================


class TestFilterGroupbyAggChains:
    """Test filter -> groupby -> agg operation chains."""

    def test_filter_groupby_sum(self):
        """Filter then groupby with sum."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "A", "B", "A", "B"],
            "value": [10, 20, 30, 40, 50, 60],
        })
        ds_df = DataStore({
            "category": ["A", "B", "A", "B", "A", "B"],
            "value": [10, 20, 30, 40, 50, 60],
        })

        pd_result = pd_df[pd_df["value"] > 15].groupby("category")["value"].sum()
        ds_result = ds_df[ds_df["value"] > 15].groupby("category")["value"].sum()

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_filter_groupby_multiple_aggs(self):
        """Filter then groupby with multiple aggregations."""
        pd_df = pd.DataFrame({
            "cat": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        })
        ds_df = DataStore({
            "cat": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        })

        pd_filtered = pd_df[pd_df["val"] > 2]
        pd_result = pd_filtered.groupby("cat")["val"].agg(["sum", "mean", "count"])

        ds_filtered = ds_df[ds_df["val"] > 2]
        ds_result = ds_filtered.groupby("cat")["val"].agg(["sum", "mean", "count"])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_groupby_agg_then_filter(self):
        """Filter -> groupby -> agg -> filter on aggregated result."""
        pd_df = pd.DataFrame({
            "group": ["A", "A", "B", "B", "C", "C"],
            "amount": [100, 200, 50, 60, 300, 400],
        })
        ds_df = DataStore({
            "group": ["A", "A", "B", "B", "C", "C"],
            "amount": [100, 200, 50, 60, 300, 400],
        })

        # Filter -> groupby -> sum -> to_frame -> reset index -> filter
        pd_grouped = pd_df[pd_df["amount"] > 50].groupby("group")["amount"].sum()
        pd_result = pd_grouped[pd_grouped > 200].reset_index()

        ds_grouped = ds_df[ds_df["amount"] > 50].groupby("group")["amount"].sum()
        ds_result = ds_grouped[ds_grouped > 200].reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# 4. Sort + Head/Tail Chains
# =============================================================================


class TestSortHeadTailChains:
    """Test sort with head/tail combinations."""

    def test_sort_then_head(self):
        """Sort then head should return top N sorted rows."""
        pd_df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85, 92, 78, 95, 88],
        })
        ds_df = DataStore({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85, 92, 78, 95, 88],
        })

        pd_result = pd_df.sort_values("score", ascending=False).head(3)
        ds_result = ds_df.sort_values("score", ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_tail(self):
        """Sort then tail should return bottom N sorted rows."""
        pd_df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85, 92, 78, 95, 88],
        })
        ds_df = DataStore({
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "score": [85, 92, 78, 95, 88],
        })

        pd_result = pd_df.sort_values("score", ascending=True).tail(2)
        ds_result = ds_df.sort_values("score", ascending=True).tail(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_sort_head(self):
        """Filter -> sort -> head chain."""
        pd_df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        })
        ds_df = DataStore({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        })

        pd_result = pd_df[pd_df["x"] > 3].sort_values("y", ascending=False).head(3)
        ds_result = ds_df[ds_df["x"] > 3].sort_values("y", ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """head(0) should return empty DataFrame with same columns."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0
        assert list(ds_result.columns) == ["a", "b"]

    def test_head_larger_than_dataframe(self):
        """head(n) where n > row count should return all rows."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.head(100)
        ds_result = ds_df.head(100)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 3


# =============================================================================
# 5. Deep 5+ Operation Chains
# =============================================================================


class TestDeepOperationChains:
    """Test chains with 5 or more operations."""

    def test_five_operation_chain(self):
        """Filter -> assign -> filter -> sort -> head."""
        pd_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "value": [100, 200, 150, 300, 250, 180, 220, 280, 320, 160],
            "type": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        })
        ds_df = DataStore({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "value": [100, 200, 150, 300, 250, 180, 220, 280, 320, 160],
            "type": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        })

        # 5 operations: filter -> assign -> filter -> sort -> head
        pd_result = (
            pd_df[pd_df["value"] > 150]
            .assign(doubled=lambda df: df["value"] * 2)
        )
        pd_result = pd_result[pd_result["doubled"] > 400]
        pd_result = pd_result.sort_values("value", ascending=False).head(3)

        ds_result = ds_df[ds_df["value"] > 150]
        ds_result = ds_result.assign(doubled=ds_result["value"] * 2)
        ds_result = ds_result[ds_result["doubled"] > 400]
        ds_result = ds_result.sort_values("value", ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_six_operation_chain_with_groupby(self):
        """Filter -> assign -> groupby -> agg -> sort -> head."""
        pd_df = pd.DataFrame({
            "category": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "amount": [10, 20, 30, 40, 50, 60, 70, 80],
            "active": [True, True, False, True, True, False, True, True],
        })
        ds_df = DataStore({
            "category": ["X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "amount": [10, 20, 30, 40, 50, 60, 70, 80],
            "active": [True, True, False, True, True, False, True, True],
        })

        # Filter
        pd_result = pd_df[pd_df["active"] == True]
        ds_result = ds_df[ds_df["active"] == True]

        # Assign
        pd_result = pd_result.assign(bonus=pd_result["amount"] * 0.1)
        ds_result = ds_result.assign(bonus=ds_result["amount"] * 0.1)

        # Groupby + agg
        pd_result = pd_result.groupby("category").agg({"amount": "sum", "bonus": "sum"}).reset_index()
        ds_result = ds_result.groupby("category").agg({"amount": "sum", "bonus": "sum"}).reset_index()

        # Sort
        pd_result = pd_result.sort_values("amount", ascending=False)
        ds_result = ds_result.sort_values("amount", ascending=False)

        # Head
        pd_result = pd_result.head(2)
        ds_result = ds_result.head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_with_column_selection(self):
        """Filter -> select columns -> assign -> filter -> select columns."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [100, 200, 300, 400, 500],
        })

        pd_result = pd_df[pd_df["a"] > 1][["a", "b"]]
        pd_result = pd_result.assign(d=pd_result["a"] + pd_result["b"])
        pd_result = pd_result[pd_result["d"] > 30][["a", "d"]]

        ds_result = ds_df[ds_df["a"] > 1][["a", "b"]]
        ds_result = ds_result.assign(d=ds_result["a"] + ds_result["b"])
        ds_result = ds_result[ds_result["d"] > 30][["a", "d"]]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 6. String Accessor Chains
# =============================================================================


class TestStringAccessorChains:
    """Test chained string accessor operations."""

    def test_str_lower_then_contains(self):
        """str.lower() then str.contains()."""
        pd_df = pd.DataFrame({"name": ["Alice", "BOB", "Charlie", "DAVID"]})
        ds_df = DataStore({"name": ["Alice", "BOB", "Charlie", "DAVID"]})

        pd_result = pd_df[pd_df["name"].str.lower().str.contains("a")]
        ds_result = ds_df[ds_df["name"].str.lower().str.contains("a")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_strip_then_upper(self):
        """str.strip() then str.upper()."""
        pd_df = pd.DataFrame({"text": ["  hello  ", " world ", "  foo"]})
        ds_df = DataStore({"text": ["  hello  ", " world ", "  foo"]})

        pd_result = pd_df.assign(clean=pd_df["text"].str.strip().str.upper())
        ds_result = ds_df.assign(clean=ds_df["text"].str.strip().str.upper())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_chain(self):
        """Multiple str.replace() calls."""
        pd_df = pd.DataFrame({"s": ["a-b-c", "x-y-z", "1-2-3"]})
        ds_df = DataStore({"s": ["a-b-c", "x-y-z", "1-2-3"]})

        pd_result = pd_df.assign(
            clean=pd_df["s"].str.replace("-", "_").str.replace("_", ".")
        )
        ds_result = ds_df.assign(
            clean=ds_df["s"].str.replace("-", "_").str.replace("_", ".")
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 7. Mixed Operation Types
# =============================================================================


class TestMixedOperationTypes:
    """Test chains mixing different operation types."""

    def test_filter_assign_rename_select(self):
        """Filter -> assign -> rename -> select columns."""
        pd_df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
        })
        ds_df = DataStore({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50],
        })

        pd_result = (
            pd_df[pd_df["col1"] > 2]
            .assign(col3=lambda df: df["col1"] + df["col2"])
            .rename(columns={"col3": "total"})
            [["col1", "total"]]
        )

        ds_result = ds_df[ds_df["col1"] > 2]
        ds_result = ds_result.assign(col3=ds_result["col1"] + ds_result["col2"])
        ds_result = ds_result.rename(columns={"col3": "total"})
        ds_result = ds_result[["col1", "total"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_drop_filter_sort(self):
        """Assign -> drop -> filter -> sort."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [10, 20, 30, 40, 50],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [10, 20, 30, 40, 50],
        })

        pd_result = pd_df.assign(d=pd_df["a"] + pd_df["b"]).drop(columns=["c"])
        pd_result = pd_result[pd_result["d"] > 5].sort_values("a")

        ds_result = ds_df.assign(d=ds_df["a"] + ds_df["b"]).drop(columns=["c"])
        ds_result = ds_result[ds_result["d"] > 5].sort_values("a")

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 8. Edge Cases with Empty Results
# =============================================================================


class TestEmptyResultChains:
    """Test chains that produce empty results at various stages."""

    def test_filter_to_empty_then_groupby(self):
        """Filter to empty, then groupby."""
        pd_df = pd.DataFrame({
            "cat": ["A", "B", "C"],
            "val": [1, 2, 3],
        })
        ds_df = DataStore({
            "cat": ["A", "B", "C"],
            "val": [1, 2, 3],
        })

        pd_result = pd_df[pd_df["val"] > 100].groupby("cat")["val"].sum()
        ds_result = ds_df[ds_df["val"] > 100].groupby("cat")["val"].sum()

        assert len(pd_result) == 0
        assert len(ds_result) == 0

    def test_filter_to_empty_then_assign(self):
        """Filter to empty, then assign new column."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        pd_result = pd_df[pd_df["a"] > 100].assign(c=lambda df: df["a"] * 2)
        ds_result = ds_df[ds_df["a"] > 100].assign(c=ds_df["a"] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_filter_to_empty_then_sort(self):
        """Filter to empty, then sort."""
        pd_df = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
        ds_df = DataStore({"x": [1, 2, 3], "y": [3, 2, 1]})

        pd_result = pd_df[pd_df["x"] > 100].sort_values("y")
        ds_result = ds_df[ds_df["x"] > 100].sort_values("y")

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0


# =============================================================================
# 9. Contradictory/Redundant Operations
# =============================================================================


class TestRedundantOperations:
    """Test chains with redundant or contradictory operations."""

    def test_select_then_select_subset(self):
        """Select columns, then select subset of those columns."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12],
        })
        ds_df = DataStore({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
            "d": [10, 11, 12],
        })

        pd_result = pd_df[["a", "b", "c"]][["a", "b"]]
        ds_result = ds_df[["a", "b", "c"]][["a", "b"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_sort_different_column(self):
        """Sort by one column, then sort by another (second sort takes precedence)."""
        pd_df = pd.DataFrame({
            "x": [3, 1, 2, 5, 4],
            "y": [1, 2, 3, 4, 5],
        })
        ds_df = DataStore({
            "x": [3, 1, 2, 5, 4],
            "y": [1, 2, 3, 4, 5],
        })

        pd_result = pd_df.sort_values("x").sort_values("y", ascending=False)
        ds_result = ds_df.sort_values("x").sort_values("y", ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_then_head_smaller(self):
        """head(5) then head(3) should be same as head(3)."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        pd_result = pd_df.head(5).head(3)
        ds_result = ds_df.head(5).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 3


# =============================================================================
# 10. Arithmetic Operation Chains
# =============================================================================


class TestArithmeticChains:
    """Test chains of arithmetic operations on columns."""

    def test_multiple_arithmetic_ops(self):
        """Chain of arithmetic operations: (a + b) * c - d."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [2, 3, 4],
            "d": [5, 10, 15],
        })
        ds_df = DataStore({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [2, 3, 4],
            "d": [5, 10, 15],
        })

        pd_result = pd_df.assign(result=(pd_df["a"] + pd_df["b"]) * pd_df["c"] - pd_df["d"])
        ds_result = ds_df.assign(result=(ds_df["a"] + ds_df["b"]) * ds_df["c"] - ds_df["d"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.xfail(reason="chDB intDiv() returns Float64 when result fits in float, need dtype correction")
    def test_division_and_modulo(self):
        """Division and modulo operations."""
        pd_df = pd.DataFrame({
            "a": [10, 20, 30, 40],
            "b": [3, 4, 5, 6],
        })
        ds_df = DataStore({
            "a": [10, 20, 30, 40],
            "b": [3, 4, 5, 6],
        })

        pd_result = pd_df.assign(
            div=pd_df["a"] / pd_df["b"],
            floordiv=pd_df["a"] // pd_df["b"],
            mod=pd_df["a"] % pd_df["b"],
        )
        ds_result = ds_df.assign(
            div=ds_df["a"] / ds_df["b"],
            floordiv=ds_df["a"] // ds_df["b"],
            mod=ds_df["a"] % ds_df["b"],
        )

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-10)

    def test_comparison_chain(self):
        """Chain of comparison operations."""
        pd_df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
        })
        ds_df = DataStore({
            "x": [1, 2, 3, 4, 5],
            "y": [5, 4, 3, 2, 1],
        })

        # Filter with combined conditions
        pd_result = pd_df[(pd_df["x"] > 1) & (pd_df["y"] > 1) & (pd_df["x"] < pd_df["y"])]
        ds_result = ds_df[(ds_df["x"] > 1) & (ds_df["y"] > 1) & (ds_df["x"] < ds_df["y"])]

        assert_datastore_equals_pandas(ds_result, pd_result)
