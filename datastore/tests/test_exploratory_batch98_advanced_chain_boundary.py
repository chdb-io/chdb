"""
Exploratory Batch 98: Advanced Chain Boundaries and Edge Cases

This batch focuses on:
1. Complex multi-DataStore concat operations with lazy chains
2. Deep boolean condition combinations
3. Reset index with subsequent operations
4. Inplace operations in chains
5. Dynamic attribute access edge cases
6. GroupBy result post-processing chains
7. Copy semantics and mutation

Date: 2026-01-16
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
from tests.xfail_markers import (
    chdb_floordiv_returns_float,
    chdb_power_returns_float,
)


# =============================================================================
# 1. Multi-DataStore Concat with Lazy Chains
# =============================================================================


class TestConcatWithLazyChains:
    """Test concat operations combined with lazy chains."""

    def test_concat_two_datastores_basic(self):
        """Basic concat of two DataStores."""
        pd_df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pd_df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        ds_df1 = DataStore({"a": [1, 2], "b": [3, 4]})
        ds_df2 = DataStore({"a": [5, 6], "b": [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = pd.concat([get_dataframe(ds_df1), get_dataframe(ds_df2)], ignore_index=True)

        assert_frame_equal(ds_result, pd_result)

    def test_concat_filtered_datastores(self):
        """Concat DataStores after filtering each."""
        pd_df1 = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        pd_df2 = pd.DataFrame({"a": [4, 5, 6], "b": [40, 50, 60]})

        ds_df1 = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df2 = DataStore({"a": [4, 5, 6], "b": [40, 50, 60]})

        # Filter each before concat
        pd_filtered1 = pd_df1[pd_df1["a"] > 1]
        pd_filtered2 = pd_df2[pd_df2["b"] < 60]
        pd_result = pd.concat([pd_filtered1, pd_filtered2], ignore_index=True)

        ds_filtered1 = ds_df1[ds_df1["a"] > 1]
        ds_filtered2 = ds_df2[ds_df2["b"] < 60]
        ds_result = pd.concat([get_dataframe(ds_filtered1), get_dataframe(ds_filtered2)], ignore_index=True)

        assert_frame_equal(ds_result, pd_result)

    def test_filter_after_concat_wrap(self):
        """Filter a DataStore created from concat result."""
        pd_df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        pd_df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})

        ds_df1 = DataStore({"a": [1, 2], "b": [10, 20]})
        ds_df2 = DataStore({"a": [3, 4], "b": [30, 40]})

        # Concat then wrap in DataStore and filter
        pd_concat = pd.concat([pd_df1, pd_df2], ignore_index=True)
        pd_result = pd_concat[pd_concat["a"] > 2]

        ds_concat = pd.concat([get_dataframe(ds_df1), get_dataframe(ds_df2)], ignore_index=True)
        ds_wrapped = DataStore(ds_concat)
        ds_result = ds_wrapped[ds_wrapped["a"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_with_different_columns(self):
        """Concat DataStores with different columns (produces NaN)."""
        pd_df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pd_df2 = pd.DataFrame({"a": [5, 6], "c": [7, 8]})

        ds_df1 = DataStore({"a": [1, 2], "b": [3, 4]})
        ds_df2 = DataStore({"a": [5, 6], "c": [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = pd.concat([get_dataframe(ds_df1), get_dataframe(ds_df2)], ignore_index=True)

        assert_frame_equal(ds_result, pd_result)

    def test_concat_three_datastores_with_chains(self):
        """Concat three DataStores, each with different operations."""
        pd_df1 = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        pd_df2 = pd.DataFrame({"a": [4, 5, 6], "b": [40, 50, 60]})
        pd_df3 = pd.DataFrame({"a": [7, 8, 9], "b": [70, 80, 90]})

        ds_df1 = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df2 = DataStore({"a": [4, 5, 6], "b": [40, 50, 60]})
        ds_df3 = DataStore({"a": [7, 8, 9], "b": [70, 80, 90]})

        # Different operations on each
        pd_r1 = pd_df1[pd_df1["a"] > 1]
        pd_r2 = pd_df2.assign(c=pd_df2["a"] * 2)
        pd_r3 = pd_df3.head(2)
        pd_result = pd.concat([pd_r1, pd_r2[["a", "b"]], pd_r3], ignore_index=True)

        ds_r1 = ds_df1[ds_df1["a"] > 1]
        ds_r2 = ds_df2.assign(c=ds_df2["a"] * 2)
        ds_r3 = ds_df3.head(2)
        ds_result = pd.concat([get_dataframe(ds_r1), get_dataframe(ds_r2)[["a", "b"]], get_dataframe(ds_r3)], ignore_index=True)

        assert_frame_equal(ds_result, pd_result)


# =============================================================================
# 2. Deep Boolean Condition Combinations
# =============================================================================


class TestDeepBooleanConditions:
    """Test complex boolean condition combinations."""

    def test_three_and_conditions(self):
        """Three conditions combined with AND."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"]
        })

        pd_result = pd_df[(pd_df["a"] > 1) & (pd_df["b"] < 50) & (pd_df["c"] == "x")]
        ds_result = ds_df[(ds_df["a"] > 1) & (ds_df["b"] < 50) & (ds_df["c"] == "x")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_three_or_conditions(self):
        """Three conditions combined with OR."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"]
        })

        pd_result = pd_df[(pd_df["a"] == 1) | (pd_df["b"] == 30) | (pd_df["c"] == "y")]
        ds_result = ds_df[(ds_df["a"] == 1) | (ds_df["b"] == 30) | (ds_df["c"] == "y")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nested_and_or(self):
        """Nested AND and OR conditions."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

        # (a > 2 AND b < 40) OR (a == 1)
        pd_result = pd_df[((pd_df["a"] > 2) & (pd_df["b"] < 40)) | (pd_df["a"] == 1)]
        ds_result = ds_df[((ds_df["a"] > 2) & (ds_df["b"] < 40)) | (ds_df["a"] == 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_inverted_condition(self):
        """Inverted boolean condition."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [True, False, True, False, True]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [True, False, True, False, True]
        })

        # Filter by inverted bool column
        pd_result = pd_df[~pd_df["b"]]
        ds_result = ds_df[~ds_df["b"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_complex_comparison_chain(self):
        """Chain of comparisons: a > b > constant."""
        pd_df = pd.DataFrame({
            "a": [10, 20, 30, 40, 50],
            "b": [5, 25, 15, 35, 45]
        })
        ds_df = DataStore({
            "a": [10, 20, 30, 40, 50],
            "b": [5, 25, 15, 35, 45]
        })

        # a > b AND b > 10
        pd_result = pd_df[(pd_df["a"] > pd_df["b"]) & (pd_df["b"] > 10)]
        ds_result = ds_df[(ds_df["a"] > ds_df["b"]) & (ds_df["b"] > 10)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_combined_with_comparison(self):
        """isin() combined with comparison operators."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "z", "x", "y"]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "z", "x", "y"]
        })

        pd_result = pd_df[(pd_df["a"] > 2) & pd_df["b"].isin(["x", "y"])]
        ds_result = ds_df[(ds_df["a"] > 2) & ds_df["b"].isin(["x", "y"])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_combined_with_or(self):
        """between() combined with OR condition."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [10, 20, 30, 40, 50, 60, 70]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [10, 20, 30, 40, 50, 60, 70]
        })

        pd_result = pd_df[pd_df["a"].between(2, 4) | (pd_df["b"] > 60)]
        ds_result = ds_df[ds_df["a"].between(2, 4) | (ds_df["b"] > 60)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 3. Reset Index with Subsequent Operations
# =============================================================================


class TestResetIndexChains:
    """Test reset_index followed by various operations."""

    def test_reset_index_then_filter(self):
        """reset_index followed by filter."""
        pd_df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [10, 20, 30]},
            index=["x", "y", "z"]
        )
        ds_df = DataStore(
            pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=["x", "y", "z"])
        )

        pd_result = pd_df.reset_index(drop=True)
        pd_result = pd_result[pd_result["a"] > 1]

        ds_result = ds_df.reset_index(drop=True)
        ds_result = ds_result[ds_result["a"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_keep_column_then_groupby(self):
        """reset_index(drop=False) adds old index as column, then groupby."""
        pd_df = pd.DataFrame(
            {"a": [1, 2, 1, 2], "b": [10, 20, 30, 40]},
            index=["x", "y", "x", "y"]
        )
        ds_df = DataStore(
            pd.DataFrame({"a": [1, 2, 1, 2], "b": [10, 20, 30, 40]}, index=["x", "y", "x", "y"])
        )

        pd_result = pd_df.reset_index(drop=False)
        pd_result = pd_result.groupby("index")["b"].sum().reset_index()

        ds_result = ds_df.reset_index(drop=False)
        ds_result = ds_result.groupby("index")["b"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_reset_index_filter(self):
        """filter -> reset_index -> filter chain."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

        pd_result = pd_df[pd_df["a"] > 1]
        pd_result = pd_result.reset_index(drop=True)
        pd_result = pd_result[pd_result["b"] < 50]

        ds_result = ds_df[ds_df["a"] > 1]
        ds_result = ds_result.reset_index(drop=True)
        ds_result = ds_result[ds_result["b"] < 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_reset_index_assign(self):
        """groupby -> reset_index -> assign chain."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "val": [1, 2, 3, 4]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "val": [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby("cat")["val"].sum().reset_index()
        pd_result = pd_result.assign(doubled=pd_result["val"] * 2)

        ds_result = ds_df.groupby("cat")["val"].sum().reset_index()
        ds_result = ds_result.assign(doubled=ds_result["val"] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 4. GroupBy Result Post-Processing
# =============================================================================


class TestGroupByPostProcessing:
    """Test chains starting from groupby results."""

    def test_groupby_agg_filter(self):
        """groupby -> agg -> filter on aggregated value."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 2, 10, 20, 100, 200]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 2, 10, 20, 100, 200]
        })

        pd_agg = pd_df.groupby("cat")["val"].sum().reset_index()
        pd_result = pd_agg[pd_agg["val"] > 10]

        ds_agg = ds_df.groupby("cat")["val"].sum().reset_index()
        ds_result = ds_agg[ds_agg["val"] > 10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_sort_head(self):
        """groupby -> agg -> sort -> head."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 2, 10, 20, 5, 6]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "val": [1, 2, 10, 20, 5, 6]
        })

        pd_result = pd_df.groupby("cat")["val"].sum().reset_index()
        pd_result = pd_result.sort_values("val", ascending=False).head(2)

        ds_result = ds_df.groupby("cat")["val"].sum().reset_index()
        ds_result = ds_result.sort_values("val", ascending=False).head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_multi_agg_filter_assign(self):
        """groupby -> multi-agg -> filter -> assign."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "val1": [1, 2, 3, 4],
            "val2": [10, 20, 30, 40]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "val1": [1, 2, 3, 4],
            "val2": [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby("cat").agg({"val1": "sum", "val2": "mean"}).reset_index()
        pd_result = pd_result[pd_result["val1"] > 2]
        pd_result = pd_result.assign(ratio=pd_result["val2"] / pd_result["val1"])

        ds_result = ds_df.groupby("cat").agg({"val1": "sum", "val2": "mean"}).reset_index()
        ds_result = ds_result[ds_result["val1"] > 2]
        ds_result = ds_result.assign(ratio=ds_result["val2"] / ds_result["val1"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_size_filter(self):
        """groupby -> size -> filter on count."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "a", "b", "b", "c"]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "a", "b", "b", "c"]
        })

        pd_result = pd_df.groupby("cat").size().reset_index(name="count")
        pd_result = pd_result[pd_result["count"] > 1]

        ds_result = ds_df.groupby("cat").size().reset_index(name="count")
        ds_result = ds_result[ds_result["count"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 5. Copy Semantics and Mutation
# =============================================================================


class TestCopySemanticsAndMutation:
    """Test copy() behavior and mutation isolation."""

    def test_copy_then_modify(self):
        """Modify a copy should not affect original."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})

        pd_copy = pd_df.copy()
        pd_copy["c"] = [100, 200, 300]

        ds_copy = ds_df.copy()
        ds_copy["c"] = [100, 200, 300]

        # Original should be unchanged
        assert "c" not in list(pd_df.columns)
        assert "c" not in list(ds_df.columns)

        # Copy should have new column
        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_filter_result_is_independent(self):
        """Filter result modifications don't affect original."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})

        pd_filtered = pd_df[pd_df["a"] > 1].copy()
        pd_filtered["c"] = [100, 200]

        ds_filtered = ds_df[ds_df["a"] > 1].copy()
        ds_filtered["c"] = [100, 200]

        # Original should be unchanged
        assert "c" not in list(pd_df.columns)
        assert "c" not in list(ds_df.columns)

        # Filtered copy should have new column
        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_copy_deep_vs_shallow(self):
        """Test deep=True vs deep=False copy behavior."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})

        pd_deep = pd_df.copy(deep=True)
        ds_deep = ds_df.copy(deep=True)

        # Deep copies should be equal to original
        assert_datastore_equals_pandas(ds_deep, pd_deep)

    def test_chain_after_copy(self):
        """Operations on copy should work independently."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

        pd_copy = pd_df.copy()
        pd_result = pd_copy[pd_copy["a"] > 2].assign(c=pd_copy[pd_copy["a"] > 2]["b"] * 2)

        ds_copy = ds_df.copy()
        ds_filtered = ds_copy[ds_copy["a"] > 2]
        ds_result = ds_filtered.assign(c=ds_filtered["b"] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 6. Attribute Access Edge Cases
# =============================================================================


class TestAttributeAccessEdgeCases:
    """Test dynamic attribute access patterns."""

    def test_columns_attribute(self):
        """Access .columns attribute."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_cols = list(pd_df.columns)
        ds_cols = list(ds_df.columns)

        assert pd_cols == ds_cols

    def test_shape_attribute(self):
        """Access .shape attribute."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        assert pd_df.shape == ds_df.shape

    def test_dtypes_attribute(self):
        """Access .dtypes attribute."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

        assert_series_equal(ds_df.dtypes, pd_df.dtypes)

    def test_index_attribute(self):
        """Access .index attribute."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        ds_df = DataStore(pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"]))

        assert list(pd_df.index) == list(ds_df.index)

    def test_values_attribute(self):
        """Access .values attribute."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        np.testing.assert_array_equal(pd_df.values, ds_df.values)

    def test_column_access_via_dot(self):
        """Access column via bracket notation."""
        pd_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        ds_df = DataStore({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # Access column via bracket notation
        pd_col = pd_df["col1"]
        ds_col = ds_df["col1"]

        assert_series_equal(get_series(ds_col), pd_col)

    def test_empty_attribute(self):
        """Test .empty attribute."""
        pd_df_empty = pd.DataFrame()
        ds_df_empty = DataStore(pd.DataFrame())

        pd_df_nonempty = pd.DataFrame({"a": [1]})
        ds_df_nonempty = DataStore({"a": [1]})

        assert pd_df_empty.empty == ds_df_empty.empty
        assert pd_df_nonempty.empty == ds_df_nonempty.empty

    def test_ndim_attribute(self):
        """Test .ndim attribute."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        assert pd_df.ndim == ds_df.ndim


# =============================================================================
# 7. Arithmetic with Mixed Types
# =============================================================================


class TestMixedTypeArithmetic:
    """Test arithmetic operations with mixed types."""

    def test_int_plus_float_column(self):
        """Add int column to float column."""
        pd_df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [0.5, 1.5, 2.5]
        })
        ds_df = DataStore({
            "int_col": [1, 2, 3],
            "float_col": [0.5, 1.5, 2.5]
        })

        pd_result = pd_df.assign(sum_col=pd_df["int_col"] + pd_df["float_col"])
        ds_result = ds_df.assign(sum_col=ds_df["int_col"] + ds_df["float_col"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_divide_produces_float(self):
        """Integer division produces float result."""
        pd_df = pd.DataFrame({"a": [10, 20, 30], "b": [3, 4, 5]})
        ds_df = DataStore({"a": [10, 20, 30], "b": [3, 4, 5]})

        pd_result = pd_df.assign(div=pd_df["a"] / pd_df["b"])
        ds_result = ds_df.assign(div=ds_df["a"] / ds_df["b"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_floordiv_returns_float
    def test_floor_divide_preserves_int(self):
        """Floor division of integers stays integer."""
        pd_df = pd.DataFrame({"a": [10, 20, 30], "b": [3, 4, 5]})
        ds_df = DataStore({"a": [10, 20, 30], "b": [3, 4, 5]})

        pd_result = pd_df.assign(fdiv=pd_df["a"] // pd_df["b"])
        ds_result = ds_df.assign(fdiv=ds_df["a"] // ds_df["b"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_modulo_operation(self):
        """Modulo operation between columns."""
        pd_df = pd.DataFrame({"a": [10, 20, 30], "b": [3, 7, 8]})
        ds_df = DataStore({"a": [10, 20, 30], "b": [3, 7, 8]})

        pd_result = pd_df.assign(mod=pd_df["a"] % pd_df["b"])
        ds_result = ds_df.assign(mod=ds_df["a"] % ds_df["b"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_power_returns_float
    def test_power_operation(self):
        """Power operation between columns."""
        pd_df = pd.DataFrame({"base": [2, 3, 4], "exp": [2, 2, 3]})
        ds_df = DataStore({"base": [2, 3, 4], "exp": [2, 2, 3]})

        pd_result = pd_df.assign(power=pd_df["base"] ** pd_df["exp"])
        ds_result = ds_df.assign(power=ds_df["base"] ** ds_df["exp"])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 8. String Operations with Filters
# =============================================================================


class TestStringOperationsWithFilters:
    """Test string operations combined with filter chains."""

    def test_str_upper_then_filter(self):
        """String upper() then filter by original column."""
        pd_df = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "value": [1, 2, 3]
        })
        ds_df = DataStore({
            "name": ["alice", "bob", "charlie"],
            "value": [1, 2, 3]
        })

        pd_result = pd_df.assign(upper_name=pd_df["name"].str.upper())
        pd_result = pd_result[pd_result["value"] > 1]

        ds_result = ds_df.assign(upper_name=ds_df["name"].str.upper())
        ds_result = ds_result[ds_result["value"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_str_contains(self):
        """Filter using str.contains()."""
        pd_df = pd.DataFrame({
            "text": ["apple", "banana", "apricot", "cherry"],
            "value": [1, 2, 3, 4]
        })
        ds_df = DataStore({
            "text": ["apple", "banana", "apricot", "cherry"],
            "value": [1, 2, 3, 4]
        })

        pd_result = pd_df[pd_df["text"].str.contains("ap")]
        ds_result = ds_df[ds_df["text"].str.contains("ap")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_len_in_filter(self):
        """Use str.len() in filter condition."""
        pd_df = pd.DataFrame({
            "name": ["al", "bob", "charlie"],
            "value": [1, 2, 3]
        })
        ds_df = DataStore({
            "name": ["al", "bob", "charlie"],
            "value": [1, 2, 3]
        })

        pd_result = pd_df[pd_df["name"].str.len() > 2]
        ds_result = ds_df[ds_df["name"].str.len() > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_then_groupby(self):
        """String replace then groupby."""
        pd_df = pd.DataFrame({
            "cat": ["cat_a", "cat_a", "cat_b", "cat_b"],
            "value": [1, 2, 3, 4]
        })
        ds_df = DataStore({
            "cat": ["cat_a", "cat_a", "cat_b", "cat_b"],
            "value": [1, 2, 3, 4]
        })

        pd_df_copy = pd_df.copy()
        pd_df_copy["clean_cat"] = pd_df_copy["cat"].str.replace("cat_", "", regex=False)
        pd_result = pd_df_copy.groupby("clean_cat")["value"].sum().reset_index()

        ds_df_copy = ds_df.copy()
        ds_df_copy["clean_cat"] = ds_df_copy["cat"].str.replace("cat_", "", regex=False)
        ds_result = ds_df_copy.groupby("clean_cat")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 9. Edge Cases with Empty Results
# =============================================================================


class TestEmptyResultEdgeCases:
    """Test edge cases that produce empty results."""

    def test_filter_all_false(self):
        """Filter with condition that matches nothing."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})

        pd_result = pd_df[pd_df["a"] > 100]
        ds_result = ds_df[ds_df["a"] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_filter_then_groupby(self):
        """Empty result after filter, then groupby."""
        pd_df = pd.DataFrame({
            "cat": ["a", "b", "c"],
            "value": [1, 2, 3]
        })
        ds_df = DataStore({
            "cat": ["a", "b", "c"],
            "value": [1, 2, 3]
        })

        pd_result = pd_df[pd_df["value"] > 100]
        pd_grouped = pd_result.groupby("cat")["value"].sum().reset_index()

        ds_result = ds_df[ds_df["value"] > 100]
        ds_grouped = ds_result.groupby("cat")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_grouped, pd_grouped)

    def test_head_zero(self):
        """head(0) should return empty DataFrame."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_operations(self):
        """Operations on empty DataFrame."""
        pd_df = pd.DataFrame({"a": [], "b": []})
        ds_df = DataStore(pd.DataFrame({"a": [], "b": []}))

        # Various operations on empty DataFrame
        assert len(pd_df) == len(get_dataframe(ds_df))
        assert list(pd_df.columns) == list(ds_df.columns)


# =============================================================================
# 10. Chained Aggregations
# =============================================================================


class TestChainedAggregations:
    """Test aggregation results used in further aggregations."""

    def test_groupby_sum_then_sum(self):
        """Sum of sums: groupby sum then overall sum."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "value": [1, 2, 3, 4]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "value": [1, 2, 3, 4]
        })

        pd_grouped = pd_df.groupby("cat")["value"].sum().reset_index()
        pd_total = pd_grouped["value"].sum()

        ds_grouped = ds_df.groupby("cat")["value"].sum().reset_index()
        ds_total = ds_grouped["value"].sum()

        assert pd_total == ds_total

    def test_groupby_agg_then_describe(self):
        """groupby agg then describe on result."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "value": [1, 2, 10, 20, 100, 200]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "value": [1, 2, 10, 20, 100, 200]
        })

        pd_grouped = pd_df.groupby("cat")["value"].sum().reset_index()
        pd_desc = pd_grouped["value"].describe()

        ds_grouped = ds_df.groupby("cat")["value"].sum().reset_index()
        ds_desc = ds_grouped["value"].describe()

        assert_series_equal(get_series(ds_desc), pd_desc)

    def test_multi_level_groupby(self):
        """Two-level groupby with aggregation."""
        pd_df = pd.DataFrame({
            "cat1": ["a", "a", "a", "b", "b", "b"],
            "cat2": ["x", "x", "y", "x", "y", "y"],
            "value": [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            "cat1": ["a", "a", "a", "b", "b", "b"],
            "cat2": ["x", "x", "y", "x", "y", "y"],
            "value": [1, 2, 3, 4, 5, 6]
        })

        pd_result = pd_df.groupby(["cat1", "cat2"])["value"].sum().reset_index()
        ds_result = ds_df.groupby(["cat1", "cat2"])["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)
