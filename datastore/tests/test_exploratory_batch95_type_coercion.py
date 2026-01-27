"""
Exploratory Batch 95: Type Coercion Edge Cases

This batch focuses on edge cases involving type conversions and coercions:
1. Mixed numeric types (int + float)
2. String to numeric conversions
3. Nullable types (Int64, boolean)
4. Type preservation through operations
5. Explicit type conversion methods
6. Arithmetic between different types
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
# 1. Mixed Numeric Types
# =============================================================================


class TestMixedNumericTypes:
    """Test operations with mixed int/float types."""

    def test_int_float_column_addition(self):
        """Adding int and float columns should produce float result."""
        pd_df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5]
        })
        ds_df = DataStore({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5]
        })

        pd_result = pd_df["int_col"] + pd_df["float_col"]
        ds_result = ds_df["int_col"] + ds_df["float_col"]

        assert_series_equal(ds_result, pd_result)

    def test_int_division_produces_float(self):
        """Integer division should produce float result."""
        pd_df = pd.DataFrame({"a": [10, 20, 30]})
        ds_df = DataStore({"a": [10, 20, 30]})

        pd_result = pd_df["a"] / 4
        ds_result = ds_df["a"] / 4

        assert_series_equal(ds_result, pd_result)
        # Result should be float
        ds_values = get_series(ds_result)
        assert ds_values.dtype == np.float64

    def test_floor_division_preserves_int(self):
        """Floor division of ints should produce int result."""
        pd_df = pd.DataFrame({"a": [10, 20, 30]})
        ds_df = DataStore({"a": [10, 20, 30]})

        pd_result = pd_df["a"] // 4
        ds_result = ds_df["a"] // 4

        assert_series_equal(ds_result, pd_result)

    def test_mixed_int_sizes(self):
        """Operations between different int sizes."""
        pd_df = pd.DataFrame({
            "small": np.array([1, 2, 3], dtype=np.int8),
            "large": np.array([100, 200, 300], dtype=np.int64)
        })
        ds_df = DataStore({
            "small": np.array([1, 2, 3], dtype=np.int8),
            "large": np.array([100, 200, 300], dtype=np.int64)
        })

        pd_result = pd_df["small"] + pd_df["large"]
        ds_result = ds_df["small"] + ds_df["large"]

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# 2. Float Precision and Special Values
# =============================================================================


class TestFloatPrecision:
    """Test float precision and special values."""

    def test_very_small_floats(self):
        """Operations with very small float values."""
        pd_df = pd.DataFrame({"a": [1e-10, 2e-10, 3e-10]})
        ds_df = DataStore({"a": [1e-10, 2e-10, 3e-10]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert abs(ds_result - pd_result) < 1e-15

    def test_very_large_floats(self):
        """Operations with very large float values."""
        pd_df = pd.DataFrame({"a": [1e15, 2e15, 3e15]})
        ds_df = DataStore({"a": [1e15, 2e15, 3e15]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        # Allow for some floating point error
        assert abs(ds_result - pd_result) / pd_result < 1e-10

    def test_infinity_handling(self):
        """Operations involving infinity."""
        pd_df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0]})
        ds_df = DataStore({"a": [1.0, np.inf, -np.inf, 2.0]})

        # Count should not count inf
        pd_count = pd_df["a"].count()
        ds_count = ds_df["a"].count()
        assert ds_count == pd_count == 4  # inf is counted

        # Filter for finite values
        pd_finite = pd_df[pd_df["a"] != np.inf][pd_df["a"] != -np.inf]
        ds_finite = ds_df[ds_df["a"] != np.inf][ds_df["a"] != -np.inf]
        assert_datastore_equals_pandas(ds_finite, pd_finite)

    def test_negative_zero(self):
        """Handle negative zero properly."""
        pd_df = pd.DataFrame({"a": [0.0, -0.0, 1.0]})
        ds_df = DataStore({"a": [0.0, -0.0, 1.0]})

        pd_result = pd_df[pd_df["a"] == 0.0]
        ds_result = ds_df[ds_df["a"] == 0.0]

        # Both 0.0 and -0.0 should match
        assert len(get_dataframe(ds_result)) == len(pd_result) == 2


# =============================================================================
# 3. String Type Operations
# =============================================================================


class TestStringTypeOperations:
    """Test string type handling and conversions."""

    def test_string_concatenation(self):
        """Concatenating string columns."""
        pd_df = pd.DataFrame({
            "first": ["a", "b", "c"],
            "second": ["x", "y", "z"]
        })
        ds_df = DataStore({
            "first": ["a", "b", "c"],
            "second": ["x", "y", "z"]
        })

        pd_result = pd_df["first"] + pd_df["second"]
        ds_result = ds_df["first"] + ds_df["second"]

        assert_series_equal(ds_result, pd_result)

    def test_string_with_scalar(self):
        """String column concatenation with scalar."""
        pd_df = pd.DataFrame({"a": ["hello", "world"]})
        ds_df = DataStore({"a": ["hello", "world"]})

        pd_result = pd_df["a"] + "!"
        ds_result = ds_df["a"] + "!"

        assert_series_equal(ds_result, pd_result)

    def test_numeric_string_comparison(self):
        """Comparing numeric-like strings."""
        pd_df = pd.DataFrame({"a": ["1", "10", "2", "20"]})
        ds_df = DataStore({"a": ["1", "10", "2", "20"]})

        # String comparison (lexicographic, not numeric)
        pd_result = pd_df.sort_values("a")
        ds_result = ds_df.sort_values("a")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_string_handling(self):
        """Handle empty strings properly."""
        pd_df = pd.DataFrame({"a": ["hello", "", "world", ""]})
        ds_df = DataStore({"a": ["hello", "", "world", ""]})

        # Filter non-empty strings
        pd_result = pd_df[pd_df["a"] != ""]
        ds_result = ds_df[ds_df["a"] != ""]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 2


# =============================================================================
# 4. Boolean Operations
# =============================================================================


class TestBooleanOperations:
    """Test boolean type handling."""

    def test_boolean_column_and(self):
        """AND operation on boolean columns."""
        pd_df = pd.DataFrame({
            "a": [True, True, False, False],
            "b": [True, False, True, False]
        })
        ds_df = DataStore({
            "a": [True, True, False, False],
            "b": [True, False, True, False]
        })

        pd_result = pd_df["a"] & pd_df["b"]
        ds_result = ds_df["a"] & ds_df["b"]

        assert_series_equal(ds_result, pd_result)

    def test_boolean_column_or(self):
        """OR operation on boolean columns."""
        pd_df = pd.DataFrame({
            "a": [True, True, False, False],
            "b": [True, False, True, False]
        })
        ds_df = DataStore({
            "a": [True, True, False, False],
            "b": [True, False, True, False]
        })

        pd_result = pd_df["a"] | pd_df["b"]
        ds_result = ds_df["a"] | ds_df["b"]

        assert_series_equal(ds_result, pd_result)

    def test_boolean_negation(self):
        """NOT operation on boolean column."""
        pd_df = pd.DataFrame({"a": [True, False, True]})
        ds_df = DataStore({"a": [True, False, True]})

        pd_result = ~pd_df["a"]
        ds_result = ~ds_df["a"]

        assert_series_equal(ds_result, pd_result)

    def test_boolean_sum(self):
        """Sum of boolean column counts True values."""
        pd_df = pd.DataFrame({"a": [True, True, False, True]})
        ds_df = DataStore({"a": [True, True, False, True]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert ds_result == pd_result == 3


# =============================================================================
# 5. Type Conversion Methods
# =============================================================================


class TestTypeConversionMethods:
    """Test explicit type conversion methods."""

    def test_astype_int_to_float(self):
        """Convert int column to float."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df["a"].astype(float)
        ds_result = ds_df["a"].astype(float)

        assert_series_equal(ds_result, pd_result)
        assert get_series(ds_result).dtype == np.float64

    def test_astype_float_to_int(self):
        """Convert float column to int (truncates)."""
        pd_df = pd.DataFrame({"a": [1.7, 2.3, 3.9]})
        ds_df = DataStore({"a": [1.7, 2.3, 3.9]})

        pd_result = pd_df["a"].astype(int)
        ds_result = ds_df["a"].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_to_string(self):
        """Convert numeric column to string."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df["a"].astype(str)
        ds_result = ds_df["a"].astype(str)

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# 6. Assignment Type Coercion
# =============================================================================


class TestAssignmentTypeCoercion:
    """Test type coercion during column assignment."""

    def test_assign_int_to_float_column(self):
        """Assigning int value to existing float column."""
        pd_df = pd.DataFrame({"a": [1.5, 2.5, 3.5]})
        ds_df = DataStore({"a": [1.5, 2.5, 3.5]})

        pd_df["b"] = 10
        ds_df["b"] = 10

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_float_to_int_column(self):
        """Assigning float value to existing int column (creates new column)."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_df["b"] = 1.5
        ds_df["b"] = 1.5

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_assign_expression_result(self):
        """Assign result of arithmetic expression."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        pd_df["c"] = pd_df["a"] + pd_df["b"]
        ds_df["c"] = ds_df["a"] + ds_df["b"]

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# 7. Comparison Result Types
# =============================================================================


class TestComparisonResultTypes:
    """Test that comparisons return correct boolean types."""

    def test_numeric_comparison_returns_bool(self):
        """Numeric comparison should return boolean Series."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df["a"] > 3
        ds_result = ds_df["a"] > 3

        # Should be boolean type
        ds_values = get_series(ds_result)
        assert ds_values.dtype == bool or ds_values.dtype == np.bool_

        assert_series_equal(ds_result, pd_result)

    def test_string_comparison_returns_bool(self):
        """String comparison should return boolean Series."""
        pd_df = pd.DataFrame({"a": ["apple", "banana", "cherry"]})
        ds_df = DataStore({"a": ["apple", "banana", "cherry"]})

        pd_result = pd_df["a"] == "banana"
        ds_result = ds_df["a"] == "banana"

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# 8. Aggregation Result Types
# =============================================================================


class TestAggregationResultTypes:
    """Test that aggregations return correct types."""

    def test_sum_int_returns_int(self):
        """Sum of int column should return int."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert ds_result == pd_result
        assert isinstance(ds_result, (int, np.integer))

    def test_mean_int_returns_float(self):
        """Mean of int column should return float."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df["a"].mean()
        ds_result = ds_df["a"].mean()

        assert ds_result == pd_result
        assert isinstance(ds_result, (float, np.floating))

    def test_count_returns_int(self):
        """Count should always return int."""
        pd_df = pd.DataFrame({"a": [1.5, 2.5, 3.5]})
        ds_df = DataStore({"a": [1.5, 2.5, 3.5]})

        pd_result = pd_df["a"].count()
        ds_result = ds_df["a"].count()

        assert ds_result == pd_result
        assert isinstance(ds_result, (int, np.integer))


# =============================================================================
# 9. Mixed Type Filter Results
# =============================================================================


class TestMixedTypeFilters:
    """Test filtering with mixed types."""

    def test_filter_int_column_with_float_value(self):
        """Filter int column with float comparison value."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df["a"] > 2.5]
        ds_result = ds_df[ds_df["a"] > 2.5]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 3  # 3, 4, 5

    def test_filter_float_column_with_int_value(self):
        """Filter float column with int comparison value."""
        pd_df = pd.DataFrame({"a": [1.5, 2.5, 3.5, 4.5]})
        ds_df = DataStore({"a": [1.5, 2.5, 3.5, 4.5]})

        pd_result = pd_df[pd_df["a"] > 2]
        ds_result = ds_df[ds_df["a"] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(get_dataframe(ds_result)) == 3


# =============================================================================
# 10. Type Preservation Through Chains
# =============================================================================


class TestTypePreservationChains:
    """Test that types are preserved through operation chains."""

    def test_filter_preserves_column_types(self):
        """Filtering should preserve column types."""
        pd_df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        })
        ds_df = DataStore({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"]
        })

        pd_result = pd_df[pd_df["int_col"] > 1]
        ds_result = ds_df[ds_df["int_col"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_preserves_types(self):
        """GroupBy should preserve types in result."""
        pd_df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [1.5, 2.5, 3.5, 4.5]
        })
        ds_df = DataStore({
            "group": ["A", "A", "B", "B"],
            "value": [1.5, 2.5, 3.5, 4.5]
        })

        pd_result = pd_df.groupby("group")["value"].sum().reset_index()
        ds_result = ds_df.groupby("group")["value"].sum().reset_index()

        pd_result = pd_result.sort_values("group").reset_index(drop=True)
        ds_result = ds_result.sort_values("group").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter_type_preservation(self):
        """Assign followed by filter should preserve types."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4]})
        ds_df = DataStore({"a": [1, 2, 3, 4]})

        pd_result = pd_df.assign(b=pd_df["a"] * 2.0)[pd_df["a"] > 1]
        ds_result = ds_df.assign(b=ds_df["a"] * 2.0)[ds_df["a"] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# 11. Edge Cases
# =============================================================================


class TestTypeEdgeCases:
    """Edge cases in type handling."""

    def test_single_value_type(self):
        """Single value DataFrame preserves type."""
        pd_df = pd.DataFrame({"a": [42]})
        ds_df = DataStore({"a": [42]})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert ds_result == pd_result

    def test_empty_column_type(self):
        """Empty column should still have type."""
        pd_df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        ds_df = DataStore({"a": []})

        pd_result = pd_df["a"].sum()
        ds_result = ds_df["a"].sum()

        assert ds_result == pd_result == 0.0

    def test_modulo_type_preservation(self):
        """Modulo operation should preserve int type when applicable."""
        pd_df = pd.DataFrame({"a": [10, 20, 30]})
        ds_df = DataStore({"a": [10, 20, 30]})

        pd_result = pd_df["a"] % 7
        ds_result = ds_df["a"] % 7

        assert_series_equal(ds_result, pd_result)

    def test_power_operation_type(self):
        """Power operation with int base and exponent."""
        pd_df = pd.DataFrame({"a": [2, 3, 4]})
        ds_df = DataStore({"a": [2, 3, 4]})

        pd_result = pd_df["a"] ** 2
        ds_result = ds_df["a"] ** 2

        assert_series_equal(ds_result, pd_result)
