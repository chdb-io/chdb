"""
Exploratory Batch 94: NULL Semantics in Joins, Aggregations, and Filters

This batch focuses on edge cases involving NULL/None/NaN handling:
1. NULL in join keys (SQL: NULL != NULL, pandas: None matches None in some cases)
2. NULL in aggregations (sum, mean, min, max behavior)
3. NULL comparison behavior in filters
4. GroupBy with NULL in group keys
5. NULL in isin() checks
6. Empty groups / all-NULL groups in aggregations

Discovered Bugs:
- String column min/max with None fails due to type coercion issue
- skipna=False parameter is not properly supported in aggregation functions
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
# 1. NULL in Aggregations
# =============================================================================


class TestNullInAggregations:
    """Test NULL handling in various aggregation functions."""

    def test_sum_with_single_null(self):
        """sum() should ignore NULL values and sum the rest."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert ds_result == pd_result

    def test_sum_all_null_returns_zero(self):
        """sum() of all-NULL column should return 0 (pandas behavior)."""
        pd_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        ds_df = DataStore({"a": [np.nan, np.nan, np.nan]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        # pandas returns 0.0 for sum of all NaN
        assert ds_result == pd_result
        assert ds_result == 0.0

    def test_mean_with_nulls(self):
        """mean() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        # mean of [1, 2, 4] = 7/3
        assert abs(ds_result - pd_result) < 1e-10

    def test_mean_all_null_returns_nan(self):
        """mean() of all-NULL column should return NaN."""
        pd_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        ds_df = DataStore({"a": [np.nan, np.nan, np.nan]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_min_with_nulls(self):
        """min() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [3.0, np.nan, 1.0, 5.0]})
        ds_df = DataStore({"a": [3.0, np.nan, 1.0, 5.0]})

        pd_result = pd_df["a"].min()
        ds_result = ds_df["a"].min()

        assert ds_result == pd_result
        assert ds_result == 1.0

    def test_max_with_nulls(self):
        """max() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [3.0, np.nan, 1.0, 5.0]})
        ds_df = DataStore({"a": [3.0, np.nan, 1.0, 5.0]})

        pd_result = pd_df["a"].max()
        ds_result = ds_df["a"].max()

        assert ds_result == pd_result
        assert ds_result == 5.0

    def test_count_excludes_nulls(self):
        """count() should not count NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].count()
        ds_result = ds_df["a"].count()

        assert ds_result == pd_result
        assert ds_result == 3

    def test_count_all_null(self):
        """count() of all-NULL column should return 0."""
        pd_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        ds_df = DataStore({"a": [np.nan, np.nan, np.nan]})

        pd_result = pd_df["a"].count()
        ds_result = ds_df["a"].count()

        assert ds_result == pd_result
        assert ds_result == 0


# =============================================================================
# 2. NULL in String Column Aggregations
# =============================================================================


class TestNullInStringAggregations:
    """Test NULL handling in string column aggregations (min/max/first/last)."""

    @pytest.mark.xfail(
        reason="BUG: String column min() with None causes TypeError in pandas due to type coercion issue"
    )
    def test_string_min_with_nulls(self):
        """min() on string column should ignore NULL values."""
        # Use nullable string dtype to avoid TypeError in pandas
        pd_df = pd.DataFrame({"a": pd.array(["b", None, "a", "c"], dtype="string")})
        ds_df = DataStore({"a": ["b", None, "a", "c"]})

        pd_result = pd_df["a"].min()
        ds_result = ds_df["a"].min()

        assert ds_result == pd_result
        assert ds_result == "a"

    @pytest.mark.xfail(
        reason="BUG: String column max() with None causes TypeError in pandas due to type coercion issue"
    )
    def test_string_max_with_nulls(self):
        """max() on string column should ignore NULL values."""
        # Use nullable string dtype to avoid TypeError in pandas
        pd_df = pd.DataFrame({"a": pd.array(["b", None, "a", "c"], dtype="string")})
        ds_df = DataStore({"a": ["b", None, "a", "c"]})

        pd_result = pd_df["a"].max()
        ds_result = ds_df["a"].max()

        assert ds_result == pd_result
        assert ds_result == "c"

    def test_string_all_null_min(self):
        """min() on all-NULL string column should return None/NaN."""
        pd_df = pd.DataFrame({"a": [None, None, None]})
        ds_df = DataStore({"a": [None, None, None]})

        pd_result = pd_df["a"].min()
        ds_result = ds_df["a"].min()

        assert pd.isna(pd_result)
        assert pd.isna(ds_result) or ds_result is None

    def test_string_min_no_nulls(self):
        """min() on string column without NULL values should work."""
        pd_df = pd.DataFrame({"a": ["b", "d", "a", "c"]})
        ds_df = DataStore({"a": ["b", "d", "a", "c"]})

        pd_result = pd_df["a"].min()
        ds_result = ds_df["a"].min()

        assert ds_result == pd_result
        assert ds_result == "a"

    def test_string_max_no_nulls(self):
        """max() on string column without NULL values should work."""
        pd_df = pd.DataFrame({"a": ["b", "d", "a", "c"]})
        ds_df = DataStore({"a": ["b", "d", "a", "c"]})

        pd_result = pd_df["a"].max()
        ds_result = ds_df["a"].max()

        assert ds_result == pd_result
        assert ds_result == "d"


# =============================================================================
# 3. GroupBy with NULL in Group Keys
# =============================================================================


class TestGroupByWithNullKeys:
    """Test groupby behavior when group keys contain NULL values."""

    def test_groupby_with_null_key_dropna_true(self):
        """groupby with dropna=True (default) should exclude NULL keys."""
        pd_df = pd.DataFrame({
            "group": ["A", "B", None, "A", "B", None],
            "value": [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            "group": ["A", "B", None, "A", "B", None],
            "value": [1, 2, 3, 4, 5, 6]
        })

        # dropna=True is default - NULL group should be excluded
        pd_result = pd_df.groupby("group", dropna=True)["value"].sum().reset_index()
        ds_result = ds_df.groupby("group", dropna=True)["value"].sum().reset_index()

        pd_result = pd_result.sort_values("group").reset_index(drop=True)
        ds_result = ds_result.sort_values("group").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_with_null_key_dropna_false(self):
        """groupby with dropna=False should include NULL as a group."""
        pd_df = pd.DataFrame({
            "group": ["A", "B", None, "A", "B", None],
            "value": [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            "group": ["A", "B", None, "A", "B", None],
            "value": [1, 2, 3, 4, 5, 6]
        })

        # dropna=False should include NULL as a group
        pd_result = pd_df.groupby("group", dropna=False)["value"].sum().reset_index()
        ds_result = ds_df.groupby("group", dropna=False)["value"].sum().reset_index()

        # Sort: None will be last in pandas
        pd_sorted = pd_result.sort_values("group", na_position="last").reset_index(drop=True)
        ds_sorted = ds_result.sort_values("group", na_position="last").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_sorted, pd_sorted)

    def test_groupby_all_null_key_with_dropna_false(self):
        """groupby on all-NULL key column with dropna=False."""
        pd_df = pd.DataFrame({
            "group": [None, None, None],
            "value": [1, 2, 3]
        })
        ds_df = DataStore({
            "group": [None, None, None],
            "value": [1, 2, 3]
        })

        pd_result = pd_df.groupby("group", dropna=False)["value"].sum().reset_index()
        ds_result = ds_df.groupby("group", dropna=False)["value"].sum().reset_index()

        # Should have one group with NULL key
        assert len(get_dataframe(ds_result)) == len(pd_result)

    def test_groupby_agg_with_null_values_in_agg_column(self):
        """groupby aggregation should handle NULL values in aggregated column."""
        pd_df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, np.nan]
        })
        ds_df = DataStore({
            "group": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, np.nan]
        })

        pd_result = pd_df.groupby("group")["value"].sum().reset_index()
        ds_result = ds_df.groupby("group")["value"].sum().reset_index()

        pd_result = pd_result.sort_values("group").reset_index(drop=True)
        ds_result = ds_result.sort_values("group").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 4. NULL in Filter Conditions
# =============================================================================


class TestNullInFilters:
    """Test NULL behavior in filter conditions."""

    def test_filter_equality_with_nan(self):
        """Equality filter should not match NaN (NaN != NaN in pandas)."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df[pd_df["a"] == 2.0]
        ds_result = ds_df[ds_df["a"] == 2.0]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 1

    def test_filter_greater_than_with_nan(self):
        """Comparison with NaN should return False (row excluded)."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df[pd_df["a"] > 1.5]
        ds_result = ds_df[ds_df["a"] > 1.5]

        assert_datastore_equals_pandas(ds_result, pd_result)
        # NaN row should be excluded
        assert len(get_dataframe(ds_result)) == 2

    def test_filter_isna(self):
        """isna() should correctly identify NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df[pd_df["a"].isna()]
        ds_result = ds_df[ds_df["a"].isna()]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 1

    def test_filter_notna(self):
        """notna() should correctly identify non-NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df[pd_df["a"].notna()]
        ds_result = ds_df[ds_df["a"].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 3

    def test_filter_string_equality_with_none(self):
        """String equality filter should not match None."""
        pd_df = pd.DataFrame({"a": ["x", "y", None, "z"]})
        ds_df = DataStore({"a": ["x", "y", None, "z"]})

        pd_result = pd_df[pd_df["a"] == "y"]
        ds_result = ds_df[ds_df["a"] == "y"]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 1


# =============================================================================
# 5. NULL in isin() Checks
# =============================================================================


class TestNullInIsin:
    """Test isin() behavior with NULL values."""

    def test_isin_with_values_no_null(self):
        """isin() should match values correctly."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df["a"].isin([2, 4])]
        ds_result = ds_df[ds_df["a"].isin([2, 4])]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 2

    def test_isin_column_has_nan(self):
        """isin() with column containing NaN, values list without NaN."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df[pd_df["a"].isin([2.0, 4.0])]
        ds_result = ds_df[ds_df["a"].isin([2.0, 4.0])]

        # NaN row should not match
        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 2

    def test_isin_string_with_none(self):
        """isin() with string column containing None."""
        pd_df = pd.DataFrame({"a": ["x", "y", None, "z"]})
        ds_df = DataStore({"a": ["x", "y", None, "z"]})

        pd_result = pd_df[pd_df["a"].isin(["x", "z"])]
        ds_result = ds_df[ds_df["a"].isin(["x", "z"])]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 2


# =============================================================================
# 6. NULL in Join Operations
# =============================================================================


class TestNullInJoins:
    """Test NULL handling in merge/join operations."""

    def test_inner_join_null_keys_excluded(self):
        """Inner join should not match NULL keys (SQL: NULL != NULL)."""
        pd_left = pd.DataFrame({
            "key": ["A", "B", None, "D"],
            "val_left": [1, 2, 3, 4]
        })
        pd_right = pd.DataFrame({
            "key": ["B", None, "D", "E"],
            "val_right": [10, 20, 30, 40]
        })

        ds_left = DataStore({
            "key": ["A", "B", None, "D"],
            "val_left": [1, 2, 3, 4]
        })
        ds_right = DataStore({
            "key": ["B", None, "D", "E"],
            "val_right": [10, 20, 30, 40]
        })

        pd_result = pd_left.merge(pd_right, on="key", how="inner")
        ds_result = ds_left.merge(ds_right, on="key", how="inner")

        pd_result = pd_result.sort_values("key").reset_index(drop=True)
        ds_result = ds_result.sort_values("key").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_left_join_preserves_left_nulls(self):
        """Left join should preserve left rows with NULL keys."""
        pd_left = pd.DataFrame({
            "key": ["A", None, "C"],
            "val_left": [1, 2, 3]
        })
        pd_right = pd.DataFrame({
            "key": ["A", "C"],
            "val_right": [10, 30]
        })

        ds_left = DataStore({
            "key": ["A", None, "C"],
            "val_left": [1, 2, 3]
        })
        ds_right = DataStore({
            "key": ["A", "C"],
            "val_right": [10, 30]
        })

        pd_result = pd_left.merge(pd_right, on="key", how="left")
        ds_result = ds_left.merge(ds_right, on="key", how="left")

        pd_result = pd_result.sort_values("key", na_position="last").reset_index(drop=True)
        ds_result = ds_result.sort_values("key", na_position="last").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 7. fillna and dropna
# =============================================================================


class TestFillnaDropna:
    """Test fillna and dropna operations."""

    def test_fillna_numeric(self):
        """fillna should replace NaN with specified value."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, np.nan]})

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_string(self):
        """fillna should replace None in string column."""
        pd_df = pd.DataFrame({"a": ["x", None, "z", None]})
        ds_df = DataStore({"a": ["x", None, "z", None]})

        pd_result = pd_df.fillna("missing")
        ds_result = ds_df.fillna("missing")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_any(self):
        """dropna(how='any') should drop rows with any NULL."""
        pd_df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", "y", None]
        })
        ds_df = DataStore({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", "y", None]
        })

        pd_result = pd_df.dropna(how="any")
        ds_result = ds_df.dropna(how="any")

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Only first row has no NULL
        assert len(get_dataframe(ds_result)) == 1

    def test_dropna_all(self):
        """dropna(how='all') should drop rows where all values are NULL."""
        pd_df = pd.DataFrame({
            "a": [1.0, np.nan, np.nan],
            "b": ["x", "y", None]
        })
        ds_df = DataStore({
            "a": [1.0, np.nan, np.nan],
            "b": ["x", "y", None]
        })

        pd_result = pd_df.dropna(how="all")
        ds_result = ds_df.dropna(how="all")

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Row 3 (NaN, None) has all NULL, so only 2 rows remain
        assert len(get_dataframe(ds_result)) == 2

    def test_dropna_subset(self):
        """dropna(subset=[col]) should only check specified columns."""
        pd_df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", "y", None]
        })
        ds_df = DataStore({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", "y", None]
        })

        pd_result = pd_df.dropna(subset=["a"])
        ds_result = ds_df.dropna(subset=["a"])

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Row 2 has NaN in 'a', so dropped. Rows 1 and 3 remain.
        assert len(get_dataframe(ds_result)) == 2


# =============================================================================
# 8. Edge Cases with Empty DataFrames
# =============================================================================


class TestEmptyDataFrameNullSemantics:
    """Test NULL semantics with empty DataFrames."""

    def test_sum_empty_dataframe(self):
        """sum() on empty DataFrame should return 0."""
        pd_df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        ds_df = DataStore({"a": []})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert ds_result == pd_result
        assert ds_result == 0.0

    def test_count_empty_dataframe(self):
        """count() on empty DataFrame should return 0."""
        pd_df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        ds_df = DataStore({"a": []})

        pd_result = pd_df["a"].count()
        ds_result = ds_df["a"].count()

        assert ds_result == pd_result
        assert ds_result == 0

    def test_mean_empty_dataframe(self):
        """mean() on empty DataFrame should return NaN."""
        pd_df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        ds_df = DataStore({"a": []})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_dropna_on_empty(self):
        """dropna() on empty DataFrame should return empty."""
        pd_df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        ds_df = DataStore({"a": []})

        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 0


# =============================================================================
# 9. Mixed NULL and Normal Operations Chain
# =============================================================================


class TestNullOperationChains:
    """Test chains of operations involving NULL values."""

    def test_filter_then_agg_with_nulls(self):
        """Filter then aggregate with NULL in data."""
        pd_df = pd.DataFrame({
            "cat": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, 4.0]
        })
        ds_df = DataStore({
            "cat": ["A", "A", "B", "B"],
            "value": [1.0, np.nan, 3.0, 4.0]
        })

        pd_result = pd_df[pd_df["cat"] == "A"]["value"].sum()
        ds_result = ds_df[ds_df["cat"] == "A"]["value"].sum()

        # A group has [1.0, NaN], sum should be 1.0
        assert ds_result == pd_result
        assert ds_result == 1.0

    def test_fillna_then_filter(self):
        """fillna then filter chain."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, np.nan]})

        pd_result = pd_df.fillna(0)[pd_df.fillna(0)["a"] > 0.5]
        ds_result = ds_df.fillna(0)[ds_df.fillna(0)["a"] > 0.5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_groupby(self):
        """dropna then groupby chain."""
        pd_df = pd.DataFrame({
            "cat": ["A", None, "B", "A"],
            "value": [1, 2, 3, 4]
        })
        ds_df = DataStore({
            "cat": ["A", None, "B", "A"],
            "value": [1, 2, 3, 4]
        })

        pd_result = pd_df.dropna().groupby("cat")["value"].sum().reset_index()
        ds_result = ds_df.dropna().groupby("cat")["value"].sum().reset_index()

        pd_result = pd_result.sort_values("cat").reset_index(drop=True)
        ds_result = ds_result.sort_values("cat").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 10. Additional NULL Edge Cases
# =============================================================================


class TestAdditionalNullEdgeCases:
    """Additional edge cases for NULL handling."""

    def test_std_with_nulls(self):
        """std() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].std()
        ds_result = ds_df["a"].std()

        assert abs(ds_result - pd_result) < 1e-10

    def test_var_with_nulls(self):
        """var() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].var()
        ds_result = ds_df["a"].var()

        assert abs(ds_result - pd_result) < 1e-10

    def test_median_with_nulls(self):
        """median() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 5.0, 6.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 5.0, 6.0]})

        pd_result = pd_df["a"].median()
        ds_result = ds_df["a"].median()

        # median of [1, 2, 5, 6] = 3.5
        assert ds_result == pd_result

    def test_multiple_null_columns_dropna(self):
        """dropna with multiple columns having NULL."""
        pd_df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, 2.0, 3.0, np.nan],
            "c": ["x", "y", "z", None]
        })
        ds_df = DataStore({
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, 2.0, 3.0, np.nan],
            "c": ["x", "y", "z", None]
        })

        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)
        # Only row 3 (3.0, 3.0, "z") has no NULL
        assert len(get_dataframe(ds_result)) == 1

    @pytest.mark.xfail(
        reason="BUG: DataStore does not properly support skipna=False parameter"
    )
    def test_sum_skipna_false(self):
        """sum(skipna=False) should return NaN if any NULL present."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].sum(skipna=False)
        ds_result = ds_df["a"].sum(skipna=False)

        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_prod_with_nulls(self):
        """prod() should ignore NULL values."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        ds_df = DataStore({"a": [1.0, 2.0, np.nan, 4.0]})

        pd_result = pd_df["a"].prod()
        ds_result = ds_df["a"].prod()

        # prod of [1, 2, 4] = 8
        assert ds_result == pd_result
        assert ds_result == 8.0
