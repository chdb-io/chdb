"""
Exploratory Test Batch 99: Cross-DataStore Operations and Accessor Edge Cases

Focus areas:
1. Cross-DataStore arithmetic and comparison operations
2. Complex str accessor chains with edge cases
3. dt accessor edge cases with timezone handling
4. MultiIndex operations
5. Memory-efficient operations with large chains
"""

import pytest
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
)


# =============================================================================
# 1. Cross-DataStore Operations
# =============================================================================


class TestCrossDataStoreOperations:
    """Test operations involving columns from different DataStores."""

    def test_cross_datastore_addition(self):
        """Add columns from different DataStores."""
        ds1 = DataStore({"a": [1, 2, 3], "b": [10, 20, 30]})
        ds2 = DataStore({"x": [100, 200, 300], "y": [1000, 2000, 3000]})

        pd1 = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        pd2 = pd.DataFrame({"x": [100, 200, 300], "y": [1000, 2000, 3000]})

        # Cross-datastore addition
        ds_result = ds1["a"] + ds2["x"]
        pd_result = pd1["a"] + pd2["x"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_cross_datastore_comparison(self):
        """Compare columns from different DataStores."""
        ds1 = DataStore({"a": [1, 5, 3]})
        ds2 = DataStore({"x": [2, 4, 3]})

        pd1 = pd.DataFrame({"a": [1, 5, 3]})
        pd2 = pd.DataFrame({"x": [2, 4, 3]})

        ds_result = ds1["a"] > ds2["x"]
        pd_result = pd1["a"] > pd2["x"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_cross_datastore_multiply(self):
        """Multiply columns from different DataStores."""
        ds1 = DataStore({"a": [2, 3, 4]})
        ds2 = DataStore({"x": [10, 20, 30]})

        pd1 = pd.DataFrame({"a": [2, 3, 4]})
        pd2 = pd.DataFrame({"x": [10, 20, 30]})

        ds_result = ds1["a"] * ds2["x"]
        pd_result = pd1["a"] * pd2["x"]

        assert_series_equal(get_series(ds_result), pd_result)

    @pytest.mark.xfail(reason="Cross-datastore operations with mismatched index: DataStore ignores index alignment")
    def test_cross_datastore_with_mismatched_index(self):
        """Cross-datastore operation with different indexes.
        
        BUG: DataStore does not perform pandas-style index alignment when
        doing operations between columns from different DataStores. Instead,
        it operates positionally. This is a deviation from pandas behavior.
        
        Expected: pandas aligns by index, resulting in NaN for non-matching indexes
        Actual: DataStore operates positionally, ignoring index
        """
        ds1 = DataStore(pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2]))
        ds2 = DataStore(pd.DataFrame({"x": [10, 20, 30]}, index=[1, 2, 3]))

        pd1 = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        pd2 = pd.DataFrame({"x": [10, 20, 30]}, index=[1, 2, 3])

        ds_result = ds1["a"] + ds2["x"]
        pd_result = pd1["a"] + pd2["x"]

        assert_series_equal(get_series(ds_result), pd_result)


# =============================================================================
# 2. String Accessor Edge Cases
# =============================================================================


class TestStrAccessorEdgeCases:
    """Test str accessor with edge cases."""

    def test_str_split_empty_string(self):
        """str.split on empty strings."""
        pd_df = pd.DataFrame({"text": ["a,b,c", "", "x,y", None, ""]})
        ds_df = DataStore({"text": ["a,b,c", "", "x,y", None, ""]})

        pd_result = pd_df["text"].str.split(",")
        ds_result = ds_df["text"].str.split(",")

        # Compare as lists since split returns lists
        ds_executed = get_series(ds_result)
        for i in range(len(pd_result)):
            pd_val = pd_result.iloc[i]
            ds_val = ds_executed.iloc[i]
            # Check for None/NaN using scalar check
            pd_is_na = pd_val is None or (isinstance(pd_val, float) and np.isnan(pd_val))
            ds_is_na = ds_val is None or (isinstance(ds_val, float) and np.isnan(ds_val))
            if pd_is_na:
                assert ds_is_na, f"Expected NaN at index {i}, got {ds_val}"
            else:
                assert list(pd_val) == list(ds_val), f"Mismatch at index {i}: {pd_val} vs {ds_val}"

    def test_str_replace_regex(self):
        """str.replace with regex pattern."""
        pd_df = pd.DataFrame({"text": ["abc123", "def456", "ghi789"]})
        ds_df = DataStore({"text": ["abc123", "def456", "ghi789"]})

        pd_result = pd_df["text"].str.replace(r"\d+", "NUM", regex=True)
        ds_result = ds_df["text"].str.replace(r"\d+", "NUM", regex=True)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_contains_case_insensitive(self):
        """str.contains with case insensitive matching."""
        pd_df = pd.DataFrame({"text": ["Hello", "WORLD", "hello", "world"]})
        ds_df = DataStore({"text": ["Hello", "WORLD", "hello", "world"]})

        pd_result = pd_df["text"].str.contains("hello", case=False)
        ds_result = ds_df["text"].str.contains("hello", case=False)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_strip_whitespace_variants(self):
        """str.strip with various whitespace."""
        pd_df = pd.DataFrame({"text": ["  hello  ", "\thello\t", "\nhello\n", "hello"]})
        ds_df = DataStore({"text": ["  hello  ", "\thello\t", "\nhello\n", "hello"]})

        pd_result = pd_df["text"].str.strip()
        ds_result = ds_df["text"].str.strip()

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_pad_with_fillchar(self):
        """str.pad with custom fill character."""
        pd_df = pd.DataFrame({"text": ["a", "bb", "ccc"]})
        ds_df = DataStore({"text": ["a", "bb", "ccc"]})

        pd_result = pd_df["text"].str.pad(5, fillchar="*")
        ds_result = ds_df["text"].str.pad(5, fillchar="*")

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_slice_negative_index(self):
        """str.slice with negative index."""
        pd_df = pd.DataFrame({"text": ["hello", "world", "test"]})
        ds_df = DataStore({"text": ["hello", "world", "test"]})

        pd_result = pd_df["text"].str.slice(-3)
        ds_result = ds_df["text"].str.slice(-3)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_chain_operations(self):
        """Chain multiple str operations."""
        pd_df = pd.DataFrame({"text": ["  HELLO  ", "  WORLD  ", "  TEST  "]})
        ds_df = DataStore({"text": ["  HELLO  ", "  WORLD  ", "  TEST  "]})

        pd_result = pd_df["text"].str.strip().str.lower().str.capitalize()
        ds_result = ds_df["text"].str.strip().str.lower().str.capitalize()

        assert_series_equal(get_series(ds_result), pd_result)


# =============================================================================
# 3. DateTime Accessor Edge Cases
# =============================================================================


class TestDtAccessorEdgeCases:
    """Test dt accessor with edge cases."""

    @pytest.mark.xfail(reason="dt.year with NaT: dtype mismatch (Int32 vs float64)")
    def test_dt_with_nat(self):
        """dt accessor with NaT values.
        
        BUG: DataStore returns Int32 dtype for dt.year when there are NaT values,
        while pandas returns float64 (to accommodate NaN values).
        
        Expected: float64 dtype (pandas behavior for nullable int)
        Actual: Int32 dtype from chDB
        """
        dates = pd.to_datetime(["2023-01-01", None, "2023-03-01", pd.NaT])
        pd_df = pd.DataFrame({"date": dates})
        ds_df = DataStore({"date": dates})

        pd_result = pd_df["date"].dt.year
        ds_result = ds_df["date"].dt.year

        assert_series_equal(get_series(ds_result), pd_result)

    def test_dt_dayofweek_boundary(self):
        """dt.dayofweek for week boundaries."""
        # Sunday (6) to Monday (0) transition
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-07", "2023-01-08"])
        pd_df = pd.DataFrame({"date": dates})
        ds_df = DataStore({"date": dates})

        pd_result = pd_df["date"].dt.dayofweek
        ds_result = ds_df["date"].dt.dayofweek

        assert_series_equal(get_series(ds_result), pd_result)

    def test_dt_is_month_end(self):
        """dt.is_month_end detection."""
        dates = pd.to_datetime(["2023-01-31", "2023-02-28", "2023-02-15", "2023-04-30"])
        pd_df = pd.DataFrame({"date": dates})
        ds_df = DataStore({"date": dates})

        pd_result = pd_df["date"].dt.is_month_end
        ds_result = ds_df["date"].dt.is_month_end

        assert_series_equal(get_series(ds_result), pd_result)

    def test_dt_quarter(self):
        """dt.quarter extraction."""
        dates = pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"])
        pd_df = pd.DataFrame({"date": dates})
        ds_df = DataStore({"date": dates})

        pd_result = pd_df["date"].dt.quarter
        ds_result = ds_df["date"].dt.quarter

        assert_series_equal(get_series(ds_result), pd_result)

    def test_dt_week_of_year(self):
        """dt.isocalendar().week extraction."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-07", "2023-12-31"])
        pd_df = pd.DataFrame({"date": dates})
        ds_df = DataStore({"date": dates})

        pd_result = pd_df["date"].dt.isocalendar().week
        ds_result = ds_df["date"].dt.isocalendar().week

        # Convert to same dtype for comparison
        assert_series_equal(get_series(ds_result).astype("int64"), pd_result.astype("int64"))


# =============================================================================
# 4. Numeric Edge Cases with Special Values
# =============================================================================


class TestNumericSpecialValues:
    """Test numeric operations with special values."""

    def test_division_by_zero(self):
        """Division resulting in inf."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0]})
        ds_df = DataStore({"a": [1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0]})

        pd_result = pd_df["a"] / pd_df["b"]
        ds_result = ds_df["a"] / ds_df["b"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_mod_with_zero(self):
        """Modulo with zero divisor."""
        pd_df = pd.DataFrame({"a": [10, 20, 30], "b": [3, 0, 7]})
        ds_df = DataStore({"a": [10, 20, 30], "b": [3, 0, 7]})

        pd_result = pd_df["a"] % pd_df["b"]
        ds_result = ds_df["a"] % ds_df["b"]

        # Handle inf/nan comparison
        ds_exec = get_series(ds_result)
        for i in range(len(pd_result)):
            if np.isnan(pd_result.iloc[i]) or np.isinf(pd_result.iloc[i]):
                assert np.isnan(ds_exec.iloc[i]) or np.isinf(ds_exec.iloc[i])
            else:
                assert pd_result.iloc[i] == ds_exec.iloc[i]

    def test_power_negative_base(self):
        """Power with negative base."""
        pd_df = pd.DataFrame({"a": [-2, -3, 2, 3]})
        ds_df = DataStore({"a": [-2, -3, 2, 3]})

        pd_result = pd_df["a"] ** 2
        ds_result = ds_df["a"] ** 2

        assert_series_equal(get_series(ds_result), pd_result)

    def test_sqrt_negative_values(self):
        """sqrt of negative values (should produce NaN)."""
        pd_df = pd.DataFrame({"a": [4.0, -4.0, 9.0, -9.0]})
        ds_df = DataStore({"a": [4.0, -4.0, 9.0, -9.0]})

        pd_result = np.sqrt(pd_df["a"])
        ds_result = np.sqrt(ds_df["a"]._execute())

        # Compare handling NaN
        assert_series_equal(ds_result, pd_result)


# =============================================================================
# 5. Boolean Operations Edge Cases
# =============================================================================


class TestBooleanEdgeCases:
    """Test boolean operations with edge cases."""

    def test_bool_and_with_null(self):
        """Boolean AND with null values."""
        pd_df = pd.DataFrame({"a": [True, False, True, None], "b": [True, True, False, True]})
        ds_df = DataStore({"a": [True, False, True, None], "b": [True, True, False, True]})

        # Use fillna to handle null before boolean operation
        pd_result = pd_df["a"].fillna(False) & pd_df["b"]
        ds_result = ds_df["a"].fillna(False) & ds_df["b"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_bool_or_with_null(self):
        """Boolean OR with null values."""
        pd_df = pd.DataFrame({"a": [True, False, None], "b": [False, False, True]})
        ds_df = DataStore({"a": [True, False, None], "b": [False, False, True]})

        pd_result = pd_df["a"].fillna(False) | pd_df["b"]
        ds_result = ds_df["a"].fillna(False) | ds_df["b"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_bool_xor(self):
        """Boolean XOR operation."""
        pd_df = pd.DataFrame({"a": [True, True, False, False], "b": [True, False, True, False]})
        ds_df = DataStore({"a": [True, True, False, False], "b": [True, False, True, False]})

        pd_result = pd_df["a"] ^ pd_df["b"]
        ds_result = ds_df["a"] ^ ds_df["b"]

        assert_series_equal(get_series(ds_result), pd_result)

    def test_bool_negation(self):
        """Boolean negation with ~."""
        pd_df = pd.DataFrame({"a": [True, False, True, False]})
        ds_df = DataStore({"a": [True, False, True, False]})

        pd_result = ~pd_df["a"]
        ds_result = ~ds_df["a"]

        assert_series_equal(get_series(ds_result), pd_result)


# =============================================================================
# 6. Index Operations Edge Cases
# =============================================================================


class TestIndexEdgeCases:
    """Test index-related operations."""

    def test_reset_index_with_drop(self):
        """reset_index with drop=True."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        ds_df = DataStore(pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"]))

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_then_reset(self):
        """set_index followed by reset_index."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ds_df = DataStore({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        pd_result = pd_df.set_index("b").reset_index()
        ds_result = ds_df.set_index("b").reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicate_index_handling(self):
        """Operations with duplicate index."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "x", "y"])
        ds_df = DataStore(pd.DataFrame({"a": [1, 2, 3]}, index=["x", "x", "y"]))

        pd_result = pd_df.reset_index()
        ds_result = ds_df.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 7. Aggregation Edge Cases
# =============================================================================


class TestAggregationEdgeCases:
    """Test aggregation with edge cases."""

    def test_sum_all_null_column(self):
        """sum of column with all null values."""
        pd_df = pd.DataFrame({"a": [None, None, None]}, dtype=float)
        ds_df = DataStore(pd.DataFrame({"a": [None, None, None]}, dtype=float))

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        # pandas returns 0 for sum of all-null
        assert pd_result == ds_result

    def test_mean_all_null_column(self):
        """mean of column with all null values."""
        pd_df = pd.DataFrame({"a": [None, None, None]}, dtype=float)
        ds_df = DataStore(pd.DataFrame({"a": [None, None, None]}, dtype=float))

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        # Both should return NaN
        assert pd.isna(pd_result) and pd.isna(ds_result)

    def test_std_single_value(self):
        """std of single value (should be NaN with ddof=1)."""
        pd_df = pd.DataFrame({"a": [5.0]})
        ds_df = DataStore({"a": [5.0]})

        pd_result = pd_df["a"].std()
        ds_result = ds_df["a"].std()

        # Both should be NaN for single value with ddof=1
        assert pd.isna(pd_result) and pd.isna(ds_result)

    def test_var_single_value(self):
        """var of single value."""
        pd_df = pd.DataFrame({"a": [5.0]})
        ds_df = DataStore({"a": [5.0]})

        pd_result = pd_df["a"].var()
        ds_result = ds_df["a"].var()

        assert pd.isna(pd_result) and pd.isna(ds_result)

    def test_count_with_nulls(self):
        """count should exclude nulls."""
        pd_df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        ds_df = DataStore({"a": [1, None, 3, None, 5]})

        pd_result = pd_df["a"].count()
        ds_result = ds_df["a"].count()

        assert pd_result == ds_result


# =============================================================================
# 8. GroupBy Edge Cases
# =============================================================================


class TestGroupByEdgeCases:
    """Test groupby with edge cases."""

    def test_groupby_single_group(self):
        """groupby where all rows belong to one group."""
        pd_df = pd.DataFrame({"cat": ["a", "a", "a"], "value": [1, 2, 3]})
        ds_df = DataStore({"cat": ["a", "a", "a"], "value": [1, 2, 3]})

        pd_result = pd_df.groupby("cat")["value"].sum().reset_index()
        ds_result = ds_df.groupby("cat")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_all_unique(self):
        """groupby where each row is its own group."""
        pd_df = pd.DataFrame({"cat": ["a", "b", "c"], "value": [1, 2, 3]})
        ds_df = DataStore({"cat": ["a", "b", "c"], "value": [1, 2, 3]})

        pd_result = pd_df.groupby("cat")["value"].sum().reset_index()
        ds_result = ds_df.groupby("cat")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_with_null_key(self):
        """groupby with null values in grouping column (default dropna=True)."""
        pd_df = pd.DataFrame({"cat": ["a", None, "a", None], "value": [1, 2, 3, 4]})
        ds_df = DataStore({"cat": ["a", None, "a", None], "value": [1, 2, 3, 4]})

        pd_result = pd_df.groupby("cat")["value"].sum().reset_index()
        ds_result = ds_df.groupby("cat")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_dict(self):
        """groupby.agg with dict specifying different funcs per column."""
        pd_df = pd.DataFrame({
            "cat": ["a", "a", "b", "b"],
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40]
        })
        ds_df = DataStore({
            "cat": ["a", "a", "b", "b"],
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40]
        })

        pd_result = pd_df.groupby("cat").agg({"x": "sum", "y": "mean"}).reset_index()
        ds_result = ds_df.groupby("cat").agg({"x": "sum", "y": "mean"}).reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 9. DataFrame Construction Edge Cases
# =============================================================================


class TestDataFrameConstruction:
    """Test DataFrame construction edge cases."""

    def test_from_empty_dict(self):
        """Create DataStore from empty dict."""
        pd_df = pd.DataFrame({})
        ds_df = DataStore({})

        assert len(get_dataframe(ds_df)) == len(pd_df)
        assert list(ds_df.columns) == list(pd_df.columns)

    def test_from_dict_with_list_values(self):
        """Create DataStore from dict with list values of different lengths should fail."""
        # This should raise an error for mismatched lengths
        with pytest.raises(Exception):
            DataStore({"a": [1, 2], "b": [1, 2, 3]})

    def test_from_single_column(self):
        """Create DataStore with single column."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_from_numpy_array(self):
        """Create DataStore from numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        pd_df = pd.DataFrame(arr, columns=["a", "b"])
        ds_df = DataStore(arr, columns=["a", "b"])

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# 10. Chained Method Edge Cases
# =============================================================================


class TestChainedMethodEdgeCases:
    """Test edge cases in method chaining."""

    def test_filter_head_tail(self):
        """Filter then head then operations."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        pd_result = pd_df[pd_df["a"] > 1].head(3)
        ds_result = ds_df[ds_df["a"] > 1].head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_filter(self):
        """Sort then filter."""
        pd_df = pd.DataFrame({"a": [3, 1, 4, 1, 5], "b": [10, 20, 30, 40, 50]})
        ds_df = DataStore({"a": [3, 1, 4, 1, 5], "b": [10, 20, 30, 40, 50]})

        pd_result = pd_df.sort_values("a")[pd_df["a"] > 2]
        ds_result = ds_df.sort_values("a")[ds_df["a"] > 2]

        # Note: pandas keeps original index after filter, compare values
        assert list(get_dataframe(ds_result)["a"]) == list(pd_result["a"])

    def test_multiple_column_assignments(self):
        """Multiple column assignments in sequence."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_df = pd_df.copy()
        pd_df["b"] = pd_df["a"] * 2
        pd_df["c"] = pd_df["b"] + 1

        ds_df["b"] = ds_df["a"] * 2
        ds_df["c"] = ds_df["b"] + 1

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_chain(self):
        """Use assign for chained column creation."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.assign(b=lambda x: x["a"] * 2, c=lambda x: x["a"] + 10)
        ds_result = ds_df.assign(b=lambda x: x["a"] * 2, c=lambda x: x["a"] + 10)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 11. Null Handling Edge Cases
# =============================================================================


class TestNullHandlingEdgeCases:
    """Test null handling in various operations."""

    def test_fillna_with_dict(self):
        """fillna with dict specifying different values per column."""
        pd_df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, None]})
        ds_df = DataStore({"a": [1, None, 3], "b": [None, 2, None]})

        pd_result = pd_df.fillna({"a": 0, "b": -1})
        ds_result = ds_df.fillna({"a": 0, "b": -1})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset(self):
        """dropna with subset parameter."""
        pd_df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3], "c": [1, 2, 3]})
        ds_df = DataStore({"a": [1, None, 3], "b": [None, 2, 3], "c": [1, 2, 3]})

        pd_result = pd_df.dropna(subset=["a"])
        ds_result = ds_df.dropna(subset=["a"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_on_mixed_types(self):
        """isna on column with mixed types."""
        pd_df = pd.DataFrame({"a": [1, None, "text", None, 3.14]})
        ds_df = DataStore({"a": [1, None, "text", None, 3.14]})

        pd_result = pd_df["a"].isna()
        ds_result = ds_df["a"].isna()

        assert_series_equal(get_series(ds_result), pd_result)


# =============================================================================
# 12. Type Conversion Edge Cases
# =============================================================================


class TestTypeConversionEdgeCases:
    """Test type conversion operations."""

    def test_astype_int_to_str(self):
        """Convert int column to string."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df["a"].astype(str)
        ds_result = ds_df["a"].astype(str)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_float_to_int(self):
        """Convert float column to int (truncation)."""
        pd_df = pd.DataFrame({"a": [1.1, 2.9, 3.5]})
        ds_df = DataStore({"a": [1.1, 2.9, 3.5]})

        pd_result = pd_df["a"].astype(int)
        ds_result = ds_df["a"].astype(int)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_str_to_float(self):
        """Convert string column to float."""
        pd_df = pd.DataFrame({"a": ["1.1", "2.2", "3.3"]})
        ds_df = DataStore({"a": ["1.1", "2.2", "3.3"]})

        pd_result = pd_df["a"].astype(float)
        ds_result = ds_df["a"].astype(float)

        assert_series_equal(get_series(ds_result), pd_result)
