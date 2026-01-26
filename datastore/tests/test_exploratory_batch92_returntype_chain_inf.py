"""
Exploratory Batch 92: Return Type Consistency, Chain Operations, and Inf/NaN Edge Cases

This batch focuses on:
1. Method return type consistency - ensure methods return exactly same types as pandas
2. Complex chain operations with type transitions (DataFrame -> Series -> scalar -> back)
3. Edge cases with inf, -inf, NaN in arithmetic operations
4. Comparison operators with special values
5. Operations that change behavior based on input shape/content
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
# 1. Return Type Consistency Tests
# =============================================================================


class TestReturnTypeConsistency:
    """Ensure DataStore methods return exactly the same types as pandas."""

    def test_tolist_returns_list(self):
        """tolist() must return Python list, not numpy array."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ds_df = DataStore({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        pd_result = pd_df["a"].tolist()
        ds_result = ds_df["a"].tolist()

        assert type(pd_result) == type(ds_result), f"Expected {type(pd_result)}, got {type(ds_result)}"
        assert isinstance(ds_result, list), "tolist() must return list"
        assert pd_result == ds_result

    def test_to_dict_returns_dict(self):
        """to_dict() must return Python dict."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.to_dict()
        ds_result = ds_df.to_dict()

        assert type(pd_result) == type(ds_result)
        assert isinstance(ds_result, dict)
        assert pd_result == ds_result

    def test_to_dict_orient_list(self):
        """to_dict(orient='list') must return dict of lists."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.to_dict(orient="list")
        ds_result = ds_df.to_dict(orient="list")

        assert type(pd_result) == type(ds_result)
        for key in pd_result:
            assert type(pd_result[key]) == type(ds_result[key])
            assert pd_result[key] == ds_result[key]

    def test_to_dict_orient_records(self):
        """to_dict(orient='records') must return list of dicts."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.to_dict(orient="records")
        ds_result = ds_df.to_dict(orient="records")

        assert type(pd_result) == type(ds_result)
        assert isinstance(ds_result, list)
        assert pd_result == ds_result

    def test_columns_returns_index(self):
        """columns attribute must return pandas Index."""
        pd_df = pd.DataFrame({"a": [1], "b": [2]})
        ds_df = DataStore({"a": [1], "b": [2]})

        pd_cols = pd_df.columns
        ds_cols = ds_df.columns

        assert type(pd_cols) == type(ds_cols), f"Expected {type(pd_cols)}, got {type(ds_cols)}"
        assert list(pd_cols) == list(ds_cols)

    def test_index_returns_index(self):
        """index attribute must return pandas Index."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_idx = pd_df.index
        ds_idx = ds_df.index

        # Note: type may differ (RangeIndex vs Int64Index), but values should match
        assert list(pd_idx) == list(ds_idx)

    def test_values_returns_ndarray(self):
        """values attribute must return numpy array."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_vals = pd_df.values
        ds_vals = ds_df.values

        assert type(pd_vals) == type(ds_vals)
        assert isinstance(ds_vals, np.ndarray)
        np.testing.assert_array_equal(pd_vals, ds_vals)

    def test_shape_returns_tuple(self):
        """shape attribute must return tuple."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        pd_shape = pd_df.shape
        ds_shape = ds_df.shape

        assert type(pd_shape) == type(ds_shape)
        assert pd_shape == ds_shape

    def test_dtypes_returns_series(self):
        """dtypes attribute must return Series."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        ds_df = DataStore({"a": [1, 2], "b": ["x", "y"]})

        pd_dtypes = pd_df.dtypes
        ds_dtypes = ds_df.dtypes

        assert type(pd_dtypes) == type(ds_dtypes)
        assert isinstance(ds_dtypes, pd.Series)


# =============================================================================
# 2. Chain Operation Type Transition Tests
# =============================================================================


class TestChainTypeTransitions:
    """Test operations that transition between DataFrame, Series, and scalar."""

    def test_df_to_series_to_scalar_mean(self):
        """DataFrame column selection -> Series -> scalar via mean."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        assert type(pd_result) == type(ds_result) or (
            isinstance(pd_result, (int, float, np.floating))
            and isinstance(ds_result, (int, float, np.floating))
        )
        np.testing.assert_almost_equal(float(pd_result), float(ds_result))

    def test_filter_agg_chain(self):
        """Filter -> agg produces correct type."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

        pd_result = pd_df[pd_df["a"] > 2]["b"].sum()
        ds_result = ds_df[ds_df["a"] > 2]["b"].sum()

        np.testing.assert_almost_equal(float(pd_result), float(ds_result))

    def test_groupby_agg_returns_dataframe(self):
        """GroupBy.agg() returns DataFrame."""
        pd_df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
        ds_df = DataStore({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})

        pd_result = pd_df.groupby("cat").agg({"val": "sum"})
        ds_result = ds_df.groupby("cat").agg({"val": "sum"})

        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_groupby_column_agg_returns_series(self):
        """GroupBy['col'].mean() returns Series."""
        pd_df = pd.DataFrame({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})
        ds_df = DataStore({"cat": ["A", "B", "A", "B"], "val": [1, 2, 3, 4]})

        pd_result = pd_df.groupby("cat")["val"].mean()
        ds_result = ds_df.groupby("cat")["val"].mean()

        assert_series_equal(ds_result, pd_result)

    def test_multiple_chain_operations(self):
        """Multiple chained operations maintain correct types."""
        pd_df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": ["x", "y", "x", "y", "x"]}
        )
        ds_df = DataStore(
            {"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": ["x", "y", "x", "y", "x"]}
        )

        # Chain: filter -> select columns -> sort -> head
        pd_result = pd_df[pd_df["a"] > 1][["a", "b"]].sort_values("b").head(2)
        ds_result = ds_df[ds_df["a"] > 1][["a", "b"]].sort_values("b").head(2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self):
        """Assign new column then filter."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4]})
        ds_df = DataStore({"a": [1, 2, 3, 4]})

        pd_result = pd_df.assign(b=pd_df["a"] * 2)[lambda x: x["b"] > 4]
        ds_result = ds_df.assign(b=ds_df["a"] * 2)[lambda x: x["b"] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 3. Inf/NaN Edge Cases
# =============================================================================


class TestInfNanEdgeCases:
    """Test arithmetic operations with inf, -inf, and NaN."""

    def test_addition_with_inf(self):
        """Addition with infinity."""
        pd_df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, np.nan]})
        ds_df = DataStore({"a": [1.0, np.inf, -np.inf, np.nan]})

        pd_result = pd_df["a"] + 1
        ds_result = ds_df["a"] + 1

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_multiplication_with_inf(self):
        """Multiplication with infinity."""
        pd_df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 0.0, np.nan]})
        ds_df = DataStore({"a": [1.0, np.inf, -np.inf, 0.0, np.nan]})

        pd_result = pd_df["a"] * 2
        ds_result = ds_df["a"] * 2

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_division_with_zero(self):
        """Division by zero produces inf/-inf."""
        pd_df = pd.DataFrame({"a": [1.0, -1.0, 0.0]})
        ds_df = DataStore({"a": [1.0, -1.0, 0.0]})

        # Suppress division by zero warning
        with np.errstate(divide="ignore", invalid="ignore"):
            pd_result = pd_df["a"] / 0.0
            ds_result = ds_df["a"] / 0.0

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_inf_comparison(self):
        """Comparison with infinity."""
        pd_df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, np.nan, 100.0]})
        ds_df = DataStore({"a": [1.0, np.inf, -np.inf, np.nan, 100.0]})

        pd_gt_inf = pd_df[pd_df["a"] > 50]
        ds_gt_inf = ds_df[ds_df["a"] > 50]

        assert_datastore_equals_pandas(ds_gt_inf, pd_gt_inf)

    def test_inf_in_mean(self):
        """Mean of column with infinity."""
        pd_df = pd.DataFrame({"a": [1.0, 2.0, np.inf]})
        ds_df = DataStore({"a": [1.0, 2.0, np.inf]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        assert np.isinf(pd_result)
        assert np.isinf(float(ds_result))

    def test_inf_in_sum(self):
        """Sum of column with infinity."""
        pd_df = pd.DataFrame({"a": [1.0, np.inf, -np.inf]})
        ds_df = DataStore({"a": [1.0, np.inf, -np.inf]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        # inf + (-inf) = nan
        assert np.isnan(pd_result)
        assert np.isnan(float(ds_result))

    def test_nan_propagation_in_chain(self):
        """NaN propagation through chain operations."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, 4.0]})

        pd_result = (pd_df["a"] + 1) * 2 - 3
        ds_result = (ds_df["a"] + 1) * 2 - 3

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_fillna_with_inf(self):
        """fillna() can use inf as fill value."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0]})

        pd_result = pd_df["a"].fillna(np.inf)
        ds_result = ds_df["a"].fillna(np.inf)

        assert_series_equal(ds_result, pd_result, check_dtype=False)


# =============================================================================
# 4. Comparison Operators with Special Values
# =============================================================================


class TestComparisonSpecialValues:
    """Test comparison operators with special values."""

    def test_nan_comparison_gt(self):
        """NaN > any number is False."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0]})

        pd_result = pd_df[pd_df["a"] > 0]
        ds_result = ds_df[ds_df["a"] > 0]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_comparison_eq(self):
        """NaN == NaN is False in comparisons."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0]})

        # NaN == NaN returns False
        pd_result = pd_df[pd_df["a"] == np.nan]
        ds_result = ds_df[ds_df["a"] == np.nan]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_for_nan_detection(self):
        """Use isna() to detect NaN values."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, None]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, None]})

        pd_result = pd_df[pd_df["a"].isna()]
        ds_result = ds_df[ds_df["a"].isna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isinf_equivalent(self):
        """Test detecting infinite values."""
        pd_df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, np.nan, 2.0]})
        ds_df = DataStore({"a": [1.0, np.inf, -np.inf, np.nan, 2.0]})

        # Filter infinite values
        pd_result = pd_df[np.isinf(pd_df["a"])]
        ds_result = ds_df[np.isinf(get_series(ds_df["a"]))]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 5. Operations with Shape-Dependent Behavior
# =============================================================================


class TestShapeDependentBehavior:
    """Test operations that behave differently based on input shape."""

    def test_single_element_squeeze(self):
        """1x1 DataFrame squeeze returns scalar."""
        pd_df = pd.DataFrame({"a": [42]})
        ds_df = DataStore({"a": [42]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # pandas returns numpy scalar for 1x1
        assert pd_result == ds_result

    def test_single_column_squeeze(self):
        """Single column DataFrame squeeze returns Series."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        assert_series_equal(ds_result, pd_result)

    def test_multi_column_squeeze(self):
        """Multi-column DataFrame squeeze returns DataFrame unchanged."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_mean(self):
        """Mean of empty DataFrame returns NaN."""
        pd_df = pd.DataFrame({"a": pd.array([], dtype="float64")})
        ds_df = DataStore({"a": []})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        # Empty mean is NaN
        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_empty_dataframe_sum(self):
        """Sum of empty DataFrame returns 0."""
        pd_df = pd.DataFrame({"a": pd.array([], dtype="float64")})
        ds_df = DataStore({"a": []})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert float(pd_result) == float(ds_result) == 0.0


# =============================================================================
# 6. Index Alignment in Operations
# =============================================================================


class TestIndexAlignment:
    """Test that index alignment works correctly."""

    def test_series_addition_with_matching_index(self):
        """Series addition with matching index."""
        pd_s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
        pd_s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])

        pd_df = pd.DataFrame({"x": pd_s1, "y": pd_s2})
        ds_df = DataStore({"x": pd_s1.values.tolist(), "y": pd_s2.values.tolist()})

        pd_result = pd_df["x"] + pd_df["y"]
        ds_result = ds_df["x"] + ds_df["y"]

        # Compare values (index might differ)
        np.testing.assert_array_equal(pd_result.values, get_series(ds_result).values)

    def test_filter_preserves_shape(self):
        """Filtering preserves expected number of rows."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df["a"] > 2]
        ds_result = ds_df[ds_df["a"] > 2]

        assert len(pd_result) == len(get_dataframe(ds_result))
        assert pd_result.shape == get_dataframe(ds_result).shape


# =============================================================================
# 7. Edge Cases in String Operations
# =============================================================================


class TestStringOperationEdgeCases:
    """Test string operations with edge cases."""

    def test_empty_string_operations(self):
        """Operations on empty strings."""
        pd_df = pd.DataFrame({"s": ["", "a", "", "bc"]})
        ds_df = DataStore({"s": ["", "a", "", "bc"]})

        pd_result = pd_df["s"].str.len()
        ds_result = ds_df["s"].str.len()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_whitespace_string_strip(self):
        """Strip whitespace from strings."""
        pd_df = pd.DataFrame({"s": ["  a  ", "b", "  c", "d  "]})
        ds_df = DataStore({"s": ["  a  ", "b", "  c", "d  "]})

        pd_result = pd_df["s"].str.strip()
        ds_result = ds_df["s"].str.strip()

        assert_series_equal(ds_result, pd_result)

    def test_string_contains_empty_pattern(self):
        """str.contains with empty pattern."""
        pd_df = pd.DataFrame({"s": ["abc", "def", ""]})
        ds_df = DataStore({"s": ["abc", "def", ""]})

        pd_result = pd_df["s"].str.contains("")
        ds_result = ds_df["s"].str.contains("")

        assert_series_equal(ds_result, pd_result, check_dtype=False)


# =============================================================================
# 8. Boolean Operations Edge Cases
# =============================================================================


class TestBooleanOperationEdgeCases:
    """Test boolean operations with edge cases."""

    def test_all_true(self):
        """all() on all-True column."""
        pd_df = pd.DataFrame({"b": [True, True, True]})
        ds_df = DataStore({"b": [True, True, True]})

        pd_result = pd_df["b"].all()
        ds_result = ds_df["b"].all()

        assert pd_result == ds_result == True

    def test_all_false(self):
        """all() on column with False."""
        pd_df = pd.DataFrame({"b": [True, False, True]})
        ds_df = DataStore({"b": [True, False, True]})

        pd_result = pd_df["b"].all()
        ds_result = ds_df["b"].all()

        assert pd_result == ds_result == False

    def test_any_true(self):
        """any() on column with True."""
        pd_df = pd.DataFrame({"b": [False, True, False]})
        ds_df = DataStore({"b": [False, True, False]})

        pd_result = pd_df["b"].any()
        ds_result = ds_df["b"].any()

        assert pd_result == ds_result == True

    def test_any_false(self):
        """any() on all-False column."""
        pd_df = pd.DataFrame({"b": [False, False, False]})
        ds_df = DataStore({"b": [False, False, False]})

        pd_result = pd_df["b"].any()
        ds_result = ds_df["b"].any()

        assert pd_result == ds_result == False

    def test_boolean_with_nan(self):
        """Boolean operations with NaN values."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0]})

        # Create boolean series (NaN becomes False in comparison)
        pd_bool = pd_df["a"] > 0
        ds_bool = ds_df["a"] > 0

        pd_result = pd_bool.all()
        ds_result = get_series(ds_bool).all()

        assert pd_result == ds_result


# =============================================================================
# 9. Cumulative Operations Edge Cases
# =============================================================================


class TestCumulativeEdgeCases:
    """Test cumulative operations with edge cases."""

    def test_cumsum_with_nan(self):
        """cumsum with NaN values."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, 4.0]})

        pd_result = pd_df["a"].cumsum()
        ds_result = ds_df["a"].cumsum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cumsum_skipna_false(self):
        """cumsum with skipna=False."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, 4.0]})

        pd_result = pd_df["a"].cumsum(skipna=False)
        ds_result = ds_df["a"].cumsum(skipna=False)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cumprod_with_zero(self):
        """cumprod with zero value."""
        pd_df = pd.DataFrame({"a": [1, 2, 0, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 0, 4, 5]})

        pd_result = pd_df["a"].cumprod()
        ds_result = ds_df["a"].cumprod()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_cummax_with_decreasing(self):
        """cummax on decreasing sequence."""
        pd_df = pd.DataFrame({"a": [5, 4, 3, 2, 1]})
        ds_df = DataStore({"a": [5, 4, 3, 2, 1]})

        pd_result = pd_df["a"].cummax()
        ds_result = ds_df["a"].cummax()

        assert_series_equal(ds_result, pd_result, check_dtype=False)


# =============================================================================
# 10. Mixed Type Edge Cases
# =============================================================================


class TestMixedTypeEdgeCases:
    """Test operations with mixed types."""

    def test_int_float_addition(self):
        """Addition of int and float columns."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})

        pd_result = pd_df["a"] + pd_df["b"]
        ds_result = ds_df["a"] + ds_df["b"]

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_int_string_concat_fails(self):
        """Cannot add int and string - should fail."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ds_df = DataStore({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        # In pandas, int + str raises TypeError
        with pytest.raises((TypeError, Exception)):
            _ = pd_df["a"] + pd_df["b"]

        with pytest.raises((TypeError, Exception)):
            _ = get_series(ds_df["a"] + ds_df["b"])

    def test_numeric_comparison_with_string(self):
        """Comparison between numeric and string columns."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        # Comparing int to string should either fail or return False
        # pandas behavior varies by version
        try:
            pd_result = pd_df["a"] == "1"
            ds_result = ds_df["a"] == "1"
            # If it doesn't raise, compare the results
            assert_series_equal(ds_result, pd_result, check_dtype=False)
        except TypeError:
            # Both should raise TypeError
            with pytest.raises(TypeError):
                _ = ds_df["a"] == "1"
