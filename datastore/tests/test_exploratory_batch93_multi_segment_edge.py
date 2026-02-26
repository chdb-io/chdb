"""
Exploratory Batch 93: Multi-Segment Execution Edge Cases

This batch focuses on edge cases in the query planner's multi-segment execution:
1. Complex SQL-Pandas-SQL interleaving boundaries
2. dropna/fillna interaction with SQL pushdown
3. Assign operations followed by various filters
4. Type coercion through chain operations
5. Empty DataFrame handling through chain
6. Mixed-type columns and operations
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
# 1. Multi-Segment Execution Boundaries
# =============================================================================


class TestMultiSegmentBoundaries:
    """Test SQL-Pandas-SQL interleaving edge cases."""

    def test_filter_apply_filter_chain(self):
        """filter -> apply (breaks SQL) -> filter should work correctly."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "cat": ["x", "x", "y", "y", "z"]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "cat": ["x", "x", "y", "y", "z"]
        })

        # filter -> apply (custom function breaks SQL) -> filter
        pd_result = pd_df[pd_df["a"] > 1]
        pd_result = pd_result.apply(lambda x: x * 2 if x.name == "b" else x)
        pd_result = pd_result[pd_result["b"] > 50]

        ds_result = ds_df[ds_df["a"] > 1]
        ds_result = ds_result.apply(lambda x: x * 2 if x.name == "b" else x)
        ds_result = ds_result[ds_result["b"] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_assign_filter(self):
        """Multiple assign-filter pairs in sequence."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

        # First assign-filter
        pd_result = pd_df.assign(c=pd_df["a"] + pd_df["b"])
        pd_result = pd_result[pd_result["c"] > 20]

        ds_result = ds_df.assign(c=ds_df["a"] + ds_df["b"])
        ds_result = ds_result[ds_result["c"] > 20]

        assert_datastore_equals_pandas(ds_result, pd_result)

        # Continue with second assign-filter
        pd_result = pd_result.assign(d=pd_result["c"] * 2)
        pd_result = pd_result[pd_result["d"] > 100]

        ds_result = ds_result.assign(d=ds_result["c"] * 2)
        ds_result = ds_result[ds_result["d"] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_between_filters(self):
        """filter -> groupby.agg -> filter chain."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c"],
            "val": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b", "c"],
            "val": [10, 20, 30, 40, 50]
        })

        # Filter then groupby then filter
        pd_result = pd_df[pd_df["val"] > 15]
        pd_result = pd_result.groupby("cat")["val"].sum().reset_index()
        pd_result = pd_result[pd_result["val"] > 50]

        ds_result = ds_df[ds_df["val"] > 15]
        ds_result = ds_result.groupby("cat")["val"].sum().reset_index()
        ds_result = ds_result[ds_result["val"] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_head_filter_chain(self):
        """sort -> head -> filter requires new SQL segment."""
        pd_df = pd.DataFrame({
            "a": [5, 3, 1, 4, 2],
            "b": [50, 30, 10, 40, 20]
        })
        ds_df = DataStore({
            "a": [5, 3, 1, 4, 2],
            "b": [50, 30, 10, 40, 20]
        })

        # sort -> head -> filter (LIMIT before WHERE requires new layer)
        pd_result = pd_df.sort_values("a").head(4)
        pd_result = pd_result[pd_result["b"] > 20]

        ds_result = ds_df.sort_values("a").head(4)
        ds_result = ds_result[ds_result["b"] > 20]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_filter_slice_chain(self):
        """slice -> filter -> slice requires multiple layers."""
        pd_df = pd.DataFrame({
            "a": list(range(20)),
            "b": list(range(0, 200, 10))
        })
        ds_df = DataStore({
            "a": list(range(20)),
            "b": list(range(0, 200, 10))
        })

        # First slice -> filter -> second slice
        pd_result = pd_df.iloc[5:15]
        pd_result = pd_result[pd_result["b"] > 70]
        pd_result = pd_result.iloc[:3]

        ds_result = ds_df.iloc[5:15]
        ds_result = ds_result[ds_result["b"] > 70]
        ds_result = ds_result.iloc[:3]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 2. dropna/fillna Interaction with SQL Pushdown
# =============================================================================


class TestDropnaFillnaChains:
    """Test dropna and fillna in complex chains."""

    def test_fillna_then_filter(self):
        """fillna followed by filter using filled column."""
        pd_df = pd.DataFrame({
            "a": [1, 2, None, 4, None],
            "b": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, 2, None, 4, None],
            "b": [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.fillna({"a": 0})
        pd_result = pd_result[pd_result["a"] > 0]

        ds_result = ds_df.fillna({"a": 0})
        ds_result = ds_result[ds_result["a"] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_groupby(self):
        """dropna followed by groupby."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", None, "b", "b"],
            "val": [10, 20, 30, 40, None]
        })
        ds_df = DataStore({
            "cat": ["a", "a", None, "b", "b"],
            "val": [10, 20, 30, 40, None]
        })

        pd_result = pd_df.dropna()
        pd_result = pd_result.groupby("cat")["val"].sum().reset_index()

        ds_result = ds_df.dropna()
        ds_result = ds_result.groupby("cat")["val"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset_then_assign(self):
        """dropna with subset followed by assign."""
        pd_df = pd.DataFrame({
            "a": [1, None, 3, None, 5],
            "b": [10, 20, None, 40, 50],
            "c": [100, 200, 300, 400, 500]
        })
        ds_df = DataStore({
            "a": [1, None, 3, None, 5],
            "b": [10, 20, None, 40, 50],
            "c": [100, 200, 300, 400, 500]
        })

        # dropna only on column 'a', then assign using 'b'
        pd_result = pd_df.dropna(subset=["a"])
        pd_result = pd_result.assign(d=pd_result["b"] + pd_result["c"])

        ds_result = ds_df.dropna(subset=["a"])
        ds_result = ds_result.assign(d=ds_result["b"] + ds_result["c"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_fillna_filter_chain(self):
        """filter -> fillna -> filter chain."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [None, 20, None, 40, None]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [None, 20, None, 40, None]
        })

        pd_result = pd_df[pd_df["a"] > 1]
        pd_result = pd_result.fillna({"b": 0})
        pd_result = pd_result[pd_result["b"] > 0]

        ds_result = ds_df[ds_df["a"] > 1]
        ds_result = ds_result.fillna({"b": 0})
        ds_result = ds_result[ds_result["b"] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_ffill_then_filter(self):
        """fillna with method='ffill' then filter."""
        pd_df = pd.DataFrame({
            "a": [1, None, None, 4, None],
            "b": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, None, None, 4, None],
            "b": [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.ffill()
        pd_result = pd_result[pd_result["a"] > 2]

        ds_result = ds_df.ffill()
        ds_result = ds_result[ds_result["a"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 3. Type Coercion Through Chains
# =============================================================================


class TestTypeCoercionChains:
    """Test type coercion edge cases in chains."""

    def test_int_to_float_through_assign(self):
        """Integer column becoming float through assign with division."""
        pd_df = pd.DataFrame({"a": [10, 20, 30]})
        ds_df = DataStore({"a": [10, 20, 30]})

        pd_result = pd_df.assign(b=pd_df["a"] / 4)
        ds_result = ds_df.assign(b=ds_df["a"] / 4)

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Verify the type is float
        assert pd_result["b"].dtype == np.float64
        assert get_dataframe(ds_result)["b"].dtype == np.float64

    def test_float_to_int_with_astype(self):
        """Float to int conversion via astype in chain."""
        pd_df = pd.DataFrame({"a": [1.5, 2.5, 3.5]})
        ds_df = DataStore({"a": [1.5, 2.5, 3.5]})

        pd_result = pd_df.assign(b=pd_df["a"].astype(int))
        ds_result = ds_df.assign(b=ds_df["a"].astype(int))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_to_numeric_operations(self):
        """Operations that might trigger string-to-numeric issues."""
        pd_df = pd.DataFrame({
            "num": [1, 2, 3],
            "str_col": ["a", "b", "c"]
        })
        ds_df = DataStore({
            "num": [1, 2, 3],
            "str_col": ["a", "b", "c"]
        })

        # Filter on numeric, select string
        pd_result = pd_df[pd_df["num"] > 1][["str_col"]]
        ds_result = ds_df[ds_df["num"] > 1][["str_col"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_bool_column_arithmetic(self):
        """Boolean column in arithmetic operations."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "flag": [True, False, True, False, True]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "flag": [True, False, True, False, True]
        })

        # Multiply numeric by boolean
        pd_result = pd_df.assign(b=pd_df["a"] * pd_df["flag"])
        ds_result = ds_df.assign(b=ds_df["a"] * ds_df["flag"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nullable_int_operations(self):
        """Operations on nullable Int64 type."""
        pd_df = pd.DataFrame({"a": pd.array([1, 2, None, 4], dtype="Int64")})
        ds_df = DataStore({"a": pd.array([1, 2, None, 4], dtype="Int64")})

        # Sum should handle None/NA
        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert pd_result == ds_result


# =============================================================================
# 4. Empty DataFrame Handling Through Chains
# =============================================================================


class TestEmptyDataFrameChains:
    """Test empty DataFrame handling in chains."""

    def test_filter_to_empty_then_assign(self):
        """Filter resulting in empty df, then assign."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})

        # Filter to empty
        pd_result = pd_df[pd_df["a"] > 100]
        pd_result = pd_result.assign(c=pd_result["a"] * 2)

        ds_result = ds_df[ds_df["a"] > 100]
        ds_result = ds_result.assign(c=ds_result["a"] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_groupby(self):
        """Groupby on empty DataFrame."""
        pd_df = pd.DataFrame({"cat": [], "val": []})
        ds_df = DataStore({"cat": [], "val": []})

        pd_result = pd_df.groupby("cat")["val"].sum().reset_index()
        ds_result = ds_df.groupby("cat")["val"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_to_empty_then_dropna(self):
        """Filter to empty then dropna."""
        pd_df = pd.DataFrame({"a": [1, None, 3], "b": [10, 20, 30]})
        ds_df = DataStore({"a": [1, None, 3], "b": [10, 20, 30]})

        pd_result = pd_df[pd_df["a"] > 100].dropna()
        ds_result = ds_df[ds_df["a"] > 100].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_merge(self):
        """Merge resulting in empty DataFrame."""
        pd_df1 = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        pd_df2 = pd.DataFrame({"key": [3, 4], "val": [30, 40]})
        ds_df1 = DataStore({"key": [1, 2], "val": [10, 20]})
        ds_df2 = DataStore({"key": [3, 4], "val": [30, 40]})

        # Inner merge with no matches
        pd_result = pd_df1.merge(pd_df2, on="key", how="inner")
        ds_result = ds_df1.merge(ds_df2, on="key", how="inner")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_concat(self):
        """Concat with empty DataFrame."""
        pd_df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        pd_df2 = pd.DataFrame({"a": [], "b": []})
        ds_df1 = DataStore({"a": [1, 2], "b": [10, 20]})
        ds_df2 = DataStore({"a": [], "b": []})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        from datastore import concat as ds_concat
        ds_result = ds_concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 5. Mixed-Type Column Edge Cases
# =============================================================================


class TestMixedTypeEdgeCases:
    """Test edge cases with mixed types."""

    def test_object_column_comparison(self):
        """Comparison operations on object dtype columns."""
        pd_df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["a", "b", "c"]})
        ds_df = DataStore({"a": ["1", "2", "3"], "b": ["a", "b", "c"]})

        pd_result = pd_df[pd_df["a"] > "1"]
        ds_result = ds_df[ds_df["a"] > "1"]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_numeric_string_filter(self):
        """Filter with numeric condition on df with string column."""
        pd_df = pd.DataFrame({
            "num": [1, 2, 3],
            "str": ["a", "b", "c"],
            "mixed": [1, "two", 3.0]
        })
        ds_df = DataStore({
            "num": [1, 2, 3],
            "str": ["a", "b", "c"],
            "mixed": [1, "two", 3.0]
        })

        pd_result = pd_df[pd_df["num"] > 1]
        ds_result = ds_df[ds_df["num"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_different_type(self):
        """Assign column with different type than original."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.assign(a=["one", "two", "three"])
        ds_result = ds_df.assign(a=["one", "two", "three"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_with_different_type(self):
        """where() with replacement value of different type.
        
        Note: chDB uses nullable Int64 for integers with NULL, pandas uses float64.
        Values are correct, only dtype differs. This is a known chDB behavior.
        """
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        # Replace with None (different type)
        pd_result = pd_df.where(pd_df["a"] > 2)
        ds_result = ds_df.where(ds_df["a"] > 2)

        # Check values are correct (allow dtype difference - pandas uses float64 for NaN, chDB uses Int64)
        ds_df_result = get_dataframe(ds_result)
        assert list(ds_df_result["a"].dropna()) == [3, 4, 5]
        assert ds_df_result["a"].isna().sum() == 2


# =============================================================================
# 6. Complex Indexing Chains
# =============================================================================


class TestComplexIndexingChains:
    """Test complex indexing operations in chains."""

    def test_loc_after_filter(self):
        """loc access after filter."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        }, index=["r1", "r2", "r3", "r4", "r5"])
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        }, index=["r1", "r2", "r3", "r4", "r5"])

        # Filter preserves index, then loc should work
        pd_result = pd_df[pd_df["a"] > 2].loc["r4"]
        ds_result = ds_df[ds_df["a"] > 2].loc["r4"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_iloc_after_groupby(self):
        """iloc access after groupby.agg."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "val": [10, 20, 30, 40]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "val": [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby("cat")["val"].sum().reset_index().iloc[0]
        ds_result = ds_df.groupby("cat")["val"].sum().reset_index().iloc[0]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_multiple_column_selection_forms(self):
        """Different column selection syntaxes in chain."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [100, 200, 300]
        })
        ds_df = DataStore({
            "a": [1, 2, 3],
            "b": [10, 20, 30],
            "c": [100, 200, 300]
        })

        # Different selection forms
        pd_result = pd_df[["a", "b"]].loc[:, "a"]
        ds_result = ds_df[["a", "b"]].loc[:, "a"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_chained_boolean_indexing(self):
        """Multiple boolean indexing operations."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [50, 40, 30, 20, 10],
            "c": ["x", "y", "x", "y", "x"]
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [50, 40, 30, 20, 10],
            "c": ["x", "y", "x", "y", "x"]
        })

        # Multiple boolean conditions
        pd_result = pd_df[(pd_df["a"] > 1) & (pd_df["b"] < 45)]
        pd_result = pd_result[pd_result["c"] == "x"]

        ds_result = ds_df[(ds_df["a"] > 1) & (ds_df["b"] < 45)]
        ds_result = ds_result[ds_result["c"] == "x"]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 7. Aggregation Edge Cases
# =============================================================================


class TestAggregationEdgeCases:
    """Test edge cases in aggregation operations."""

    def test_agg_on_filtered_groupby(self):
        """Complex aggregation after groupby on filtered data."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b", "c", "c"],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df[pd_df["x"] > 1].groupby("cat").agg({"x": "sum", "y": "mean"}).reset_index()
        ds_result = ds_df[ds_df["x"] > 1].groupby("cat").agg({"x": "sum", "y": "mean"}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_aggregations_same_column(self):
        """Multiple aggregations on the same column."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "val": [10, 20, 30, 40]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "val": [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby("cat").agg(
            val_sum=("val", "sum"),
            val_mean=("val", "mean"),
            val_count=("val", "count")
        ).reset_index()

        ds_result = ds_df.groupby("cat").agg(
            val_sum=("val", "sum"),
            val_mean=("val", "mean"),
            val_count=("val", "count")
        ).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_with_all_null_group(self):
        """Aggregation where one group has all NULL values."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "val": [10.0, 20.0, None, None]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "val": [10.0, 20.0, None, None]
        })

        pd_result = pd_df.groupby("cat")["val"].sum().reset_index()
        ds_result = ds_df.groupby("cat")["val"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_min_max_on_strings(self):
        """min/max aggregation on string columns."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "name": ["alice", "bob", "charlie", "david"]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "name": ["alice", "bob", "charlie", "david"]
        })

        pd_result = pd_df.groupby("cat")["name"].min().reset_index()
        ds_result = ds_df.groupby("cat")["name"].min().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 8. Rename and Column Operations
# =============================================================================


class TestRenameColumnOps:
    """Test rename and column operations in chains."""

    def test_rename_then_filter(self):
        """Rename columns then filter using new name."""
        pd_df = pd.DataFrame({"old_name": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"old_name": [1, 2, 3, 4, 5]})

        pd_result = pd_df.rename(columns={"old_name": "new_name"})
        pd_result = pd_result[pd_result["new_name"] > 2]

        ds_result = ds_df.rename(columns={"old_name": "new_name"})
        ds_result = ds_result[ds_result["new_name"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_add_prefix_then_select(self):
        """add_prefix then select columns."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.add_prefix("col_")
        pd_result = pd_result[["col_a"]]

        ds_result = ds_df.add_prefix("col_")
        ds_result = ds_result[["col_a"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_columns_then_filter(self):
        """Drop columns then filter on remaining."""
        pd_df = pd.DataFrame({
            "keep": [1, 2, 3],
            "drop": [10, 20, 30],
            "filter_col": [100, 200, 300]
        })
        ds_df = DataStore({
            "keep": [1, 2, 3],
            "drop": [10, 20, 30],
            "filter_col": [100, 200, 300]
        })

        pd_result = pd_df.drop(columns=["drop"])
        pd_result = pd_result[pd_result["filter_col"] > 150]

        ds_result = ds_df.drop(columns=["drop"])
        ds_result = ds_result[ds_result["filter_col"] > 150]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 9. Sort and Order Edge Cases
# =============================================================================


class TestSortOrderEdgeCases:
    """Test sorting and ordering edge cases."""

    def test_sort_with_nulls(self):
        """Sort with NULL values."""
        pd_df = pd.DataFrame({"a": [3, None, 1, None, 2]})
        ds_df = DataStore({"a": [3, None, 1, None, 2]})

        pd_result = pd_df.sort_values("a")
        ds_result = ds_df.sort_values("a")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_multiple_columns_mixed_order(self):
        """Sort by multiple columns with mixed ascending/descending."""
        pd_df = pd.DataFrame({
            "a": [1, 1, 2, 2],
            "b": [4, 3, 2, 1]
        })
        ds_df = DataStore({
            "a": [1, 1, 2, 2],
            "b": [4, 3, 2, 1]
        })

        pd_result = pd_df.sort_values(["a", "b"], ascending=[True, False])
        ds_result = ds_df.sort_values(["a", "b"], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_head_then_sort(self):
        """sort -> head -> sort again."""
        pd_df = pd.DataFrame({
            "a": [5, 3, 1, 4, 2],
            "b": [10, 30, 50, 20, 40]
        })
        ds_df = DataStore({
            "a": [5, 3, 1, 4, 2],
            "b": [10, 30, 50, 20, 40]
        })

        pd_result = pd_df.sort_values("a").head(4).sort_values("b")
        ds_result = ds_df.sort_values("a").head(4).sort_values("b")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_nsmallest_chain(self):
        """nlargest followed by operations."""
        pd_df = pd.DataFrame({
            "a": [1, 5, 3, 4, 2],
            "b": [10, 50, 30, 40, 20]
        })
        ds_df = DataStore({
            "a": [1, 5, 3, 4, 2],
            "b": [10, 50, 30, 40, 20]
        })

        pd_result = pd_df.nlargest(3, "a")
        pd_result = pd_result[["b"]]

        ds_result = ds_df.nlargest(3, "a")
        ds_result = ds_result[["b"]]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 10. Edge Cases in Data Construction
# =============================================================================


class TestDataConstructionEdgeCases:
    """Test edge cases in DataStore construction."""

    def test_construct_from_series(self):
        """Construct DataStore from pandas Series."""
        pd_series = pd.Series([1, 2, 3], name="values")
        pd_df = pd_series.to_frame()

        ds_df = DataStore(pd_series)

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_construct_with_single_value_columns(self):
        """Construct with columns that have single values."""
        pd_df = pd.DataFrame({
            "single": [42, 42, 42],
            "varied": [1, 2, 3]
        })
        ds_df = DataStore({
            "single": [42, 42, 42],
            "varied": [1, 2, 3]
        })

        pd_result = pd_df[pd_df["single"] == 42]
        ds_result = ds_df[ds_df["single"] == 42]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_construct_with_unicode_column_names(self):
        """Construct with unicode column names."""
        pd_df = pd.DataFrame({"名前": [1, 2, 3], "valor": [10, 20, 30]})
        ds_df = DataStore({"名前": [1, 2, 3], "valor": [10, 20, 30]})

        pd_result = pd_df[pd_df["名前"] > 1]
        ds_result = ds_df[ds_df["名前"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_construct_with_space_in_column_name(self):
        """Construct with spaces in column names."""
        pd_df = pd.DataFrame({"column name": [1, 2, 3], "other col": [10, 20, 30]})
        ds_df = DataStore({"column name": [1, 2, 3], "other col": [10, 20, 30]})

        pd_result = pd_df[pd_df["column name"] > 1]
        ds_result = ds_df[ds_df["column name"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)
