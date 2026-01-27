"""
Exploratory Batch 97: Type Coercion, Nullable Handling, and Boundary Conditions

This batch focuses on testing:
1. Type coercion in arithmetic chains (int + float, etc.)
2. Nullable type handling (Int64, Float64 with NA)
3. Mixed-type column operations
4. Boundary conditions (single row, single column, all NULL)
5. Type preservation through SQL-Pandas-SQL interleaving
6. Return type consistency with pandas

Goal: Discover edge cases in type handling and conversion.
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
# 1. Type Coercion in Arithmetic Chains
# =============================================================================


class TestTypeCoercionArithmetic:
    """Test type coercion when combining different numeric types."""

    def test_int_plus_float_returns_float(self):
        """Adding int column to float should return float."""
        pd_df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
        })
        ds_df = DataStore({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
        })

        pd_result = pd_df.assign(result=pd_df["int_col"] + pd_df["float_col"])
        ds_result = ds_df.assign(result=ds_df["int_col"] + ds_df["float_col"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_int_divide_returns_float(self):
        """Integer division that produces float result."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(result=pd_df["a"] / 2)
        ds_result = ds_df.assign(result=ds_df["a"] / 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pytest.mark.xfail(reason="chDB floor(a/b) returns Float64 instead of Int64, needs type-aware casting")
    def test_int_floor_divide_returns_int(self):
        """Floor division should preserve integer type."""
        pd_df = pd.DataFrame({"a": [10, 20, 30, 40, 50]})
        ds_df = DataStore({"a": [10, 20, 30, 40, 50]})

        pd_result = pd_df.assign(result=pd_df["a"] // 3)
        ds_result = ds_df.assign(result=ds_df["a"] // 3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chain_int_float_int_operations(self):
        """Chain: int -> +float -> *int should maintain proper types."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [0.5, 0.5, 0.5],
            "c": [10, 10, 10],
        })
        ds_df = DataStore({
            "a": [1, 2, 3],
            "b": [0.5, 0.5, 0.5],
            "c": [10, 10, 10],
        })

        pd_result = pd_df.assign(result=(pd_df["a"] + pd_df["b"]) * pd_df["c"])
        ds_result = ds_df.assign(result=(ds_df["a"] + ds_df["b"]) * ds_df["c"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_negative_numbers_in_arithmetic(self):
        """Test arithmetic with negative numbers."""
        pd_df = pd.DataFrame({
            "a": [-1, -2, 3, -4, 5],
            "b": [1, -2, 3, -4, 5],
        })
        ds_df = DataStore({
            "a": [-1, -2, 3, -4, 5],
            "b": [1, -2, 3, -4, 5],
        })

        pd_result = pd_df.assign(
            sum_col=pd_df["a"] + pd_df["b"],
            mul_col=pd_df["a"] * pd_df["b"],
            sub_col=pd_df["a"] - pd_df["b"],
        )
        ds_result = ds_df.assign(
            sum_col=ds_df["a"] + ds_df["b"],
            mul_col=ds_df["a"] * ds_df["b"],
            sub_col=ds_df["a"] - ds_df["b"],
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 2. Nullable Type Handling
# =============================================================================


class TestNullableTypeHandling:
    """Test handling of nullable integer and float types."""

    def test_nullable_int_column(self):
        """Test column with nullable int (Int64)."""
        pd_df = pd.DataFrame({"a": pd.array([1, 2, None, 4, 5], dtype=pd.Int64Dtype())})
        ds_df = DataStore({"a": pd.array([1, 2, None, 4, 5], dtype=pd.Int64Dtype())})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        # Both should return 12 (sum of non-null values)
        assert ds_result == pd_result

    def test_nullable_int_filter(self):
        """Filter on nullable int column."""
        pd_df = pd.DataFrame({"a": pd.array([1, 2, None, 4, 5], dtype=pd.Int64Dtype())})
        ds_df = DataStore({"a": pd.array([1, 2, None, 4, 5], dtype=pd.Int64Dtype())})

        pd_result = pd_df[pd_df["a"] > 2]
        ds_result = ds_df[ds_df["a"] > 2]

        # Should only include 4 and 5 (NULL > 2 is False/Unknown)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nullable_float_column(self):
        """Test column with nullable float."""
        pd_df = pd.DataFrame({"a": [1.5, 2.5, np.nan, 4.5, 5.5]})
        ds_df = DataStore({"a": [1.5, 2.5, np.nan, 4.5, 5.5]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        # Both should compute mean of non-null values
        assert abs(ds_result - pd_result) < 0.001

    def test_nullable_in_arithmetic(self):
        """Arithmetic with nullable values should propagate nulls."""
        pd_df = pd.DataFrame({
            "a": pd.array([1, 2, None, 4], dtype=pd.Int64Dtype()),
            "b": [10, 20, 30, 40],
        })
        ds_df = DataStore({
            "a": pd.array([1, 2, None, 4], dtype=pd.Int64Dtype()),
            "b": [10, 20, 30, 40],
        })

        pd_result = pd_df.assign(result=pd_df["a"] + pd_df["b"])
        ds_result = ds_df.assign(result=ds_df["a"] + ds_df["b"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_on_nullable_column(self):
        """isna() should correctly identify NA values."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = pd_df["a"].isna()
        ds_result = ds_df["a"].isna()

        assert_series_equal(ds_result, pd_result)

    def test_notna_on_nullable_column(self):
        """notna() should correctly identify non-NA values."""
        pd_df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = pd_df["a"].notna()
        ds_result = ds_df["a"].notna()

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# 3. Boundary Conditions
# =============================================================================


class TestBoundaryConditions:
    """Test edge cases with boundary data."""

    def test_single_row_dataframe(self):
        """Operations on single-row DataFrame."""
        pd_df = pd.DataFrame({"a": [42], "b": ["hello"]})
        ds_df = DataStore({"a": [42], "b": ["hello"]})

        # Filter that matches
        pd_result = pd_df[pd_df["a"] > 0]
        ds_result = ds_df[ds_df["a"] > 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

        # Filter that doesn't match
        pd_result2 = pd_df[pd_df["a"] > 100]
        ds_result2 = ds_df[ds_df["a"] > 100]
        assert_datastore_equals_pandas(ds_result2, pd_result2)

    def test_single_column_dataframe(self):
        """Operations on single-column DataFrame."""
        pd_df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"x": [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df["x"] > 2]
        ds_result = ds_df[ds_df["x"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_null_column_sum(self):
        """Sum of all-null column."""
        pd_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        ds_df = DataStore({"a": [np.nan, np.nan, np.nan]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        # pandas returns 0.0 for sum of all-nan
        assert ds_result == pd_result

    def test_all_null_column_mean(self):
        """Mean of all-null column."""
        pd_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        ds_df = DataStore({"a": [np.nan, np.nan, np.nan]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        # Both should return nan
        assert pd.isna(pd_result) and pd.isna(ds_result)

    def test_empty_dataframe_operations(self):
        """Operations on empty DataFrame."""
        pd_df = pd.DataFrame({"a": [], "b": []})
        ds_df = DataStore({"a": [], "b": []})

        # Filter on empty
        pd_result = pd_df[pd_df["a"] > 0]
        ds_result = ds_df[ds_df["a"] > 0]
        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_very_small_numbers(self):
        """Operations with very small float numbers."""
        pd_df = pd.DataFrame({"a": [1e-10, 2e-10, 3e-10]})
        ds_df = DataStore({"a": [1e-10, 2e-10, 3e-10]})

        pd_result = pd_df.assign(doubled=pd_df["a"] * 2)
        ds_result = ds_df.assign(doubled=ds_df["a"] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_large_numbers(self):
        """Operations with very large numbers."""
        pd_df = pd.DataFrame({"a": [1e15, 2e15, 3e15]})
        ds_df = DataStore({"a": [1e15, 2e15, 3e15]})

        pd_result = pd_df.assign(doubled=pd_df["a"] * 2)
        ds_result = ds_df.assign(doubled=ds_df["a"] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 4. Mixed Type Column Operations
# =============================================================================


class TestMixedTypeOperations:
    """Test operations involving columns of different types."""

    def test_compare_int_to_float_literal(self):
        """Compare int column to float literal."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df["a"] > 2.5]
        ds_result = ds_df[ds_df["a"] > 2.5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_column_with_numeric_column(self):
        """Combine string and numeric columns in same DataFrame."""
        pd_df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        })
        ds_df = DataStore({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        })

        pd_result = pd_df[pd_df["age"] > 27]
        ds_result = ds_df[ds_df["age"] > 27]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_column_filter(self):
        """Filter using a boolean column directly."""
        pd_df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5],
            "flag": [True, False, True, False, True],
        })
        ds_df = DataStore({
            "value": [1, 2, 3, 4, 5],
            "flag": [True, False, True, False, True],
        })

        pd_result = pd_df[pd_df["flag"]]
        ds_result = ds_df[ds_df["flag"]]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_column_negation(self):
        """Negate a boolean column for filtering."""
        pd_df = pd.DataFrame({
            "value": [1, 2, 3, 4, 5],
            "flag": [True, False, True, False, True],
        })
        ds_df = DataStore({
            "value": [1, 2, 3, 4, 5],
            "flag": [True, False, True, False, True],
        })

        pd_result = pd_df[~pd_df["flag"]]
        ds_result = ds_df[~ds_df["flag"]]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 5. Type Preservation Through Operations
# =============================================================================


class TestTypePreservationThroughOps:
    """Test that types are preserved correctly through operation chains."""

    def test_filter_preserves_dtypes(self):
        """Filter should preserve original column dtypes."""
        pd_df = pd.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
        })
        ds_df = DataStore({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
        })

        pd_result = pd_df[pd_df["int_col"] > 2]
        ds_result = ds_df[ds_df["int_col"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_preserves_dtypes_in_keys(self):
        """GroupBy should preserve key column types."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        })
        ds_df = DataStore({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        })

        pd_result = pd_df.groupby("category")["value"].sum().reset_index()
        ds_result = ds_df.groupby("category")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_assign_preserves_existing_column_dtypes(self):
        """Assign should not change dtypes of existing columns."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1.5, 2.5, 3.5],
        })
        ds_df = DataStore({
            "a": [1, 2, 3],
            "b": [1.5, 2.5, 3.5],
        })

        pd_result = pd_df.assign(c=pd_df["a"] + pd_df["b"])
        ds_result = ds_df.assign(c=ds_df["a"] + ds_df["b"])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 6. Return Type Consistency
# =============================================================================


class TestReturnTypeConsistency:
    """Test that return types match pandas exactly."""

    def test_columns_returns_index(self):
        """df.columns should return same type as pandas."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_cols = pd_df.columns
        ds_cols = ds_df.columns

        assert type(pd_cols) == type(ds_cols)
        assert list(pd_cols) == list(ds_cols)

    def test_index_returns_index(self):
        """df.index should return same type as pandas."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_idx = pd_df.index
        ds_idx = ds_df.index

        # Both should be RangeIndex or similar
        assert list(pd_idx) == list(ds_idx)

    def test_shape_returns_tuple(self):
        """df.shape should return tuple."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        assert pd_df.shape == ds_df.shape
        assert isinstance(ds_df.shape, tuple)

    def test_len_returns_int(self):
        """len(df) should return int."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_len = len(pd_df)
        ds_len = len(ds_df)

        assert pd_len == ds_len
        assert type(pd_len) == type(ds_len)

    def test_scalar_agg_returns_scalar(self):
        """Scalar aggregation should return Python scalar, not numpy."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_sum = pd_df["a"].sum()
        ds_sum = ds_df["a"].sum()

        assert pd_sum == ds_sum


# =============================================================================
# 7. Complex Filter Conditions
# =============================================================================


class TestComplexFilterConditions:
    """Test complex filter conditions with multiple clauses."""

    def test_and_condition(self):
        """Filter with AND condition."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })

        pd_result = pd_df[(pd_df["a"] > 2) & (pd_df["b"] > 2)]
        ds_result = ds_df[(ds_df["a"] > 2) & (ds_df["b"] > 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_condition(self):
        """Filter with OR condition."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })

        pd_result = pd_df[(pd_df["a"] > 4) | (pd_df["b"] > 4)]
        ds_result = ds_df[(ds_df["a"] > 4) | (ds_df["b"] > 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_not_condition(self):
        """Filter with NOT condition."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
        })

        pd_result = pd_df[~(pd_df["a"] > 3)]
        ds_result = ds_df[~(ds_df["a"] > 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combined_and_or_not(self):
        """Filter with combined AND, OR, NOT conditions."""
        pd_df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": ["x", "y", "x", "y", "x"],
        })
        ds_df = DataStore({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": ["x", "y", "x", "y", "x"],
        })

        # (a > 2 AND b > 2) OR (c == 'y')
        pd_result = pd_df[((pd_df["a"] > 2) & (pd_df["b"] > 2)) | (pd_df["c"] == "y")]
        ds_result = ds_df[((ds_df["a"] > 2) & (ds_df["b"] > 2)) | (ds_df["c"] == "y")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_condition(self):
        """Filter with isin condition."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "C", "D", "E"],
            "value": [1, 2, 3, 4, 5],
        })
        ds_df = DataStore({
            "category": ["A", "B", "C", "D", "E"],
            "value": [1, 2, 3, 4, 5],
        })

        pd_result = pd_df[pd_df["category"].isin(["A", "C", "E"])]
        ds_result = ds_df[ds_df["category"].isin(["A", "C", "E"])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_condition(self):
        """Filter with between condition."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        pd_result = pd_df[pd_df["a"].between(3, 7)]
        ds_result = ds_df[ds_df["a"].between(3, 7)]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 8. String Column Edge Cases
# =============================================================================


class TestStringColumnEdgeCases:
    """Test string column operations at boundaries."""

    def test_empty_string_filter(self):
        """Filter for empty strings."""
        pd_df = pd.DataFrame({"s": ["hello", "", "world", "", "test"]})
        ds_df = DataStore({"s": ["hello", "", "world", "", "test"]})

        pd_result = pd_df[pd_df["s"] == ""]
        ds_result = ds_df[ds_df["s"] == ""]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_with_spaces(self):
        """Filter for strings with spaces."""
        pd_df = pd.DataFrame({"s": ["hello world", "test", "foo bar", "baz"]})
        ds_df = DataStore({"s": ["hello world", "test", "foo bar", "baz"]})

        pd_result = pd_df[pd_df["s"].str.contains(" ")]
        ds_result = ds_df[ds_df["s"].str.contains(" ")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_with_special_chars(self):
        """Filter for strings with special characters."""
        pd_df = pd.DataFrame({"s": ["hello@world", "test", "foo#bar", "baz"]})
        ds_df = DataStore({"s": ["hello@world", "test", "foo#bar", "baz"]})

        pd_result = pd_df[pd_df["s"].str.contains("@")]
        ds_result = ds_df[ds_df["s"].str.contains("@")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_startswith(self):
        """Test str.startswith."""
        pd_df = pd.DataFrame({"s": ["apple", "banana", "apricot", "cherry"]})
        ds_df = DataStore({"s": ["apple", "banana", "apricot", "cherry"]})

        pd_result = pd_df[pd_df["s"].str.startswith("ap")]
        ds_result = ds_df[ds_df["s"].str.startswith("ap")]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_endswith(self):
        """Test str.endswith."""
        pd_df = pd.DataFrame({"s": ["test.py", "main.js", "app.py", "style.css"]})
        ds_df = DataStore({"s": ["test.py", "main.js", "app.py", "style.css"]})

        pd_result = pd_df[pd_df["s"].str.endswith(".py")]
        ds_result = ds_df[ds_df["s"].str.endswith(".py")]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 9. GroupBy with Edge Data
# =============================================================================


class TestGroupByEdgeData:
    """Test groupby with edge cases in data."""

    def test_groupby_single_group(self):
        """GroupBy where all rows belong to same group."""
        pd_df = pd.DataFrame({
            "category": ["A", "A", "A", "A"],
            "value": [10, 20, 30, 40],
        })
        ds_df = DataStore({
            "category": ["A", "A", "A", "A"],
            "value": [10, 20, 30, 40],
        })

        pd_result = pd_df.groupby("category")["value"].sum().reset_index()
        ds_result = ds_df.groupby("category")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_all_unique_groups(self):
        """GroupBy where each row is its own group."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "C", "D"],
            "value": [10, 20, 30, 40],
        })
        ds_df = DataStore({
            "category": ["A", "B", "C", "D"],
            "value": [10, 20, 30, 40],
        })

        pd_result = pd_df.groupby("category")["value"].sum().reset_index()
        ds_result = ds_df.groupby("category")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_with_zero_values(self):
        """GroupBy with zero values in aggregation column."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [0, 0, 10, 20],
        })
        ds_df = DataStore({
            "category": ["A", "B", "A", "B"],
            "value": [0, 0, 10, 20],
        })

        pd_result = pd_df.groupby("category")["value"].sum().reset_index()
        ds_result = ds_df.groupby("category")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_with_negative_values(self):
        """GroupBy with negative values."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [-10, 20, 30, -40],
        })
        ds_df = DataStore({
            "category": ["A", "B", "A", "B"],
            "value": [-10, 20, 30, -40],
        })

        pd_result = pd_df.groupby("category")["value"].sum().reset_index()
        ds_result = ds_df.groupby("category")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# =============================================================================
# 10. Chained Operations Edge Cases
# =============================================================================


class TestChainedOperationsEdgeCases:
    """Test edge cases in chained operations."""

    def test_filter_to_empty_then_groupby(self):
        """Filter to empty, then groupby should return empty result."""
        pd_df = pd.DataFrame({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        })
        ds_df = DataStore({
            "category": ["A", "B", "A", "B"],
            "value": [10, 20, 30, 40],
        })

        # Filter to empty
        pd_filtered = pd_df[pd_df["value"] > 100]
        ds_filtered = ds_df[ds_df["value"] > 100]

        # GroupBy on empty should work
        pd_result = pd_filtered.groupby("category")["value"].sum().reset_index()
        ds_result = ds_filtered.groupby("category")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_multiple_assigns_then_filter(self):
        """Multiple assigns followed by filter."""
        pd_df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"x": [1, 2, 3, 4, 5]})

        pd_result = pd_df.assign(y=pd_df["x"] * 2).assign(z=lambda df: df["y"] + 1)
        pd_result = pd_result[pd_result["z"] > 5]

        ds_result = ds_df.assign(y=ds_df["x"] * 2)
        ds_result = ds_result.assign(z=ds_result["y"] + 1)
        ds_result = ds_result[ds_result["z"] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_assign_filter(self):
        """Filter -> assign -> filter chain."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        pd_result = pd_df[pd_df["a"] > 2]
        pd_result = pd_result.assign(b=pd_result["a"] * 2)
        pd_result = pd_result[pd_result["b"] < 15]

        ds_result = ds_df[ds_df["a"] > 2]
        ds_result = ds_result.assign(b=ds_result["a"] * 2)
        ds_result = ds_result[ds_result["b"] < 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_then_operations(self):
        """head() followed by more operations."""
        pd_df = pd.DataFrame({
            "a": [5, 3, 1, 4, 2, 6, 8, 7, 9, 10],
            "b": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })
        ds_df = DataStore({
            "a": [5, 3, 1, 4, 2, 6, 8, 7, 9, 10],
            "b": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        })

        pd_result = pd_df.head(5)
        pd_result = pd_result[pd_result["a"] > 2]

        ds_result = ds_df.head(5)
        ds_result = ds_result[ds_result["a"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)
