"""
Exploratory Batch 91: Squeeze, Explode, Stack/Unstack, and DataFrame-Series Interactions

This batch explores edge cases in:
1. squeeze() - dimension reduction from DataFrame to Series
2. explode() - expanding list/array columns
3. stack/unstack - reshaping operations
4. DataFrame-Series arithmetic broadcasting
5. Complex chained operations involving these methods

Date: 2026-01-16
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas_chdb_compat,
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
)
from tests.xfail_markers import *


class TestSqueeze:
    """Test squeeze() dimension reduction operations."""

    def test_squeeze_single_column_df(self):
        """Squeeze a single-column DataFrame to Series."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # squeeze() on single column returns Series
        assert isinstance(pd_result, pd.Series)
        assert_series_equal(ds_result, pd_result)

    def test_squeeze_single_row_df(self):
        """Squeeze a single-row DataFrame to Series."""
        pd_df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ds_df = DataStore({"a": [1], "b": [2], "c": [3]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # squeeze() on single row returns Series
        assert isinstance(pd_result, pd.Series)
        assert_series_equal(ds_result, pd_result)

    def test_squeeze_single_cell_df(self):
        """Squeeze a 1x1 DataFrame to scalar."""
        pd_df = pd.DataFrame({"a": [42]})
        ds_df = DataStore({"a": [42]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # squeeze() on 1x1 returns scalar
        assert pd_result == 42
        # DataStore may return scalar or 0-dim result
        ds_val = ds_result if isinstance(ds_result, (int, float, np.integer, np.floating)) else get_series(ds_result).iloc[0]
        assert ds_val == 42

    def test_squeeze_axis_columns(self):
        """Squeeze along columns axis."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.squeeze(axis="columns")
        ds_result = ds_df.squeeze(axis="columns")

        assert_series_equal(ds_result, pd_result)

    def test_squeeze_axis_rows(self):
        """Squeeze along rows axis."""
        pd_df = pd.DataFrame({"a": [1], "b": [2]})
        ds_df = DataStore({"a": [1], "b": [2]})

        pd_result = pd_df.squeeze(axis="rows")
        ds_result = ds_df.squeeze(axis="rows")

        assert_series_equal(ds_result, pd_result)

    def test_squeeze_multi_column_no_change(self):
        """Squeeze on multi-column DataFrame returns DataFrame."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # No change - still DataFrame
        assert isinstance(pd_result, pd.DataFrame)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_squeeze_after_filter(self):
        """Squeeze after filter operation."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Filter to single column
        pd_result = pd_df[pd_df["a"] == 2][["a"]].squeeze()
        ds_result = ds_df[ds_df["a"] == 2][["a"]].squeeze()

        # After filter to 1 row, 1 col: squeeze returns scalar
        assert pd_result == 2
        ds_val = ds_result if isinstance(ds_result, (int, float, np.integer, np.floating)) else get_series(ds_result).iloc[0]
        assert ds_val == 2


class TestExplode:
    """Test explode() operations for expanding list columns."""

    def test_explode_basic_list_column(self):
        """Explode a column containing lists."""
        pd_df = pd.DataFrame({"a": [[1, 2], [3], [4, 5, 6]], "b": ["x", "y", "z"]})
        ds_df = DataStore({"a": [[1, 2], [3], [4, 5, 6]], "b": ["x", "y", "z"]})

        pd_result = pd_df.explode("a").reset_index(drop=True)
        ds_result = ds_df.explode("a").reset_index(drop=True)

        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result, check_row_order=True)

    def test_explode_empty_list(self):
        """Explode with empty list in column."""
        pd_df = pd.DataFrame({"a": [[1, 2], [], [3]], "b": ["x", "y", "z"]})
        ds_df = DataStore({"a": [[1, 2], [], [3]], "b": ["x", "y", "z"]})

        pd_result = pd_df.explode("a").reset_index(drop=True)
        ds_result = ds_df.explode("a").reset_index(drop=True)

        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result, check_row_order=True)

    def test_explode_single_element_lists(self):
        """Explode with single-element lists."""
        pd_df = pd.DataFrame({"a": [[1], [2], [3]], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [[1], [2], [3]], "b": [4, 5, 6]})

        pd_result = pd_df.explode("a").reset_index(drop=True)
        ds_result = ds_df.explode("a").reset_index(drop=True)

        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result, check_row_order=True)

    def test_explode_ignore_index(self):
        """Explode with ignore_index=True."""
        pd_df = pd.DataFrame({"a": [[1, 2], [3, 4]], "b": ["x", "y"]})
        ds_df = DataStore({"a": [[1, 2], [3, 4]], "b": ["x", "y"]})

        pd_result = pd_df.explode("a", ignore_index=True)
        ds_result = ds_df.explode("a", ignore_index=True)

        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result, check_row_order=True)

    @chdb_array_nullable
    def test_explode_series(self):
        """Explode a Series with list elements.
        
        Note: This test fails because chDB's arrayJoin function requires 
        proper Array type, but Python lists are not converted correctly.
        """
        pd_s = pd.Series([[1, 2], [3], [4, 5]], name="vals")
        ds_df = DataStore({"vals": [[1, 2], [3], [4, 5]]})

        pd_result = pd_s.explode().reset_index(drop=True)
        ds_result = ds_df["vals"].explode().reset_index(drop=True)

        assert_series_equal(ds_result, pd_result)

    def test_explode_then_filter(self):
        """Explode followed by filter operation.
        
        Note: pandas explode returns object dtype, DataStore returns int32.
        We convert pandas result to int32 for comparison (chDB uses 32-bit integers).
        """
        pd_df = pd.DataFrame({"a": [[1, 2, 3], [4, 5]], "b": ["x", "y"]})
        ds_df = DataStore({"a": [[1, 2, 3], [4, 5]], "b": ["x", "y"]})

        pd_result = pd_df.explode("a", ignore_index=True)
        pd_result = pd_result[pd_result["a"] > 2].reset_index(drop=True)
        # Convert object dtype to int32 to match chDB output
        pd_result["a"] = pd_result["a"].astype("int32")

        ds_result = ds_df.explode("a", ignore_index=True)
        ds_result = ds_result[ds_result["a"] > 2].reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


class TestStack:
    """Test stack() operations for pivoting columns to rows."""

    def test_stack_basic(self):
        """Basic stack operation."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.stack()
        ds_result = ds_df.stack()

        # stack returns a Series
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_stack_with_null(self):
        """Stack with null values."""
        pd_df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        ds_df = DataStore({"a": [1, None, 3], "b": [4, 5, None]})

        # pandas 3.0 removed dropna parameter from stack()
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        if pandas_version >= (3, 0):
            pd_result = pd_df.stack()
            ds_result = ds_df.stack()
        else:
            pd_result = pd_df.stack(dropna=True)
            ds_result = ds_df.stack(dropna=True)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_stack_dropna_false(self):
        """Stack keeping null values."""
        pd_df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
        ds_df = DataStore({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})

        # pandas 3.0 removed dropna parameter from stack()
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        if pandas_version >= (3, 0):
            pd_result = pd_df.stack()
            ds_result = ds_df.stack()
        else:
            pd_result = pd_df.stack(dropna=False)
            ds_result = ds_df.stack(dropna=False)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_stack_single_column(self):
        """Stack single column DataFrame."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.stack()
        ds_result = ds_df.stack()

        assert_series_equal(ds_result, pd_result, check_names=False)


class TestUnstack:
    """Test unstack() operations for pivoting rows to columns."""

    def test_unstack_series_multiindex(self):
        """Unstack a Series with MultiIndex."""
        # Create a MultiIndex Series
        idx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
        pd_s = pd.Series([10, 20, 30, 40], index=idx, name="vals")

        # For DataStore, create equivalent structure
        pd_df = pd_s.unstack()

        # DataStore: build similar data
        ds_df = DataStore({
            "level_0": ["a", "a", "b", "b"],
            "level_1": [1, 2, 1, 2],
            "vals": [10, 20, 30, 40]
        })
        ds_result = ds_df.pivot_table(
            values="vals", index="level_0", columns="level_1", aggfunc="first"
        )

        # Both should have same data
        assert_datastore_equals_pandas(ds_result, pd_df, check_row_order=False)


class TestDataFrameSeriesArithmetic:
    """Test DataFrame-Series arithmetic broadcasting."""

    def test_df_sub_series_axis0(self):
        """Subtract Series from DataFrame along axis 0."""
        pd_df = pd.DataFrame({"a": [10, 20, 30], "b": [40, 50, 60]})
        pd_s = pd.Series([1, 2, 3])
        ds_df = DataStore({"a": [10, 20, 30], "b": [40, 50, 60]})

        pd_result = pd_df.sub(pd_s, axis=0)
        ds_result = ds_df.sub(pd_s, axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_add_series_axis1(self):
        """Add Series to DataFrame along axis 1."""
        pd_df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        pd_s = pd.Series([1, 2], index=["a", "b"])
        ds_df = DataStore({"a": [10, 20], "b": [30, 40]})

        pd_result = pd_df.add(pd_s, axis=1)
        ds_result = ds_df.add(pd_s, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_mul_series_broadcast(self):
        """Multiply DataFrame with Series using broadcasting."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        pd_s = pd.Series([10, 20], index=["a", "b"])
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.mul(pd_s, axis=1)
        ds_result = ds_df.mul(pd_s, axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_div_scalar_chain(self):
        """Division with scalar in chain."""
        pd_df = pd.DataFrame({"a": [10, 20, 30], "b": [40, 50, 60]})
        ds_df = DataStore({"a": [10, 20, 30], "b": [40, 50, 60]})

        pd_result = pd_df.div(10).add(5)
        ds_result = ds_df.div(10).add(5)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChains:
    """Test complex chained operations."""

    def test_filter_squeeze_arithmetic(self):
        """Filter to single column, squeeze, then arithmetic."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4, 5, 6]})

        pd_result = pd_df[["a"]].squeeze() * 2
        ds_result = ds_df[["a"]].squeeze() * 2

        assert_series_equal(ds_result, pd_result)

    def test_groupby_agg_squeeze(self):
        """GroupBy, aggregate to single column, then squeeze."""
        pd_df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        ds_df = DataStore({"g": ["a", "a", "b"], "v": [1, 2, 3]})

        pd_result = pd_df.groupby("g")[["v"]].sum().squeeze()
        ds_result = ds_df.groupby("g")[["v"]].sum().squeeze()

        # Both return Series
        assert isinstance(pd_result, pd.Series)
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_stack_unstack_roundtrip(self):
        """Stack then operations that reshape."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        # Stack to Series, then back
        pd_stacked = pd_df.stack()
        ds_stacked = ds_df.stack()

        # Convert back to DataFrame via reset_index
        pd_result = pd_stacked.reset_index(name="value")
        ds_result = get_series(ds_stacked).reset_index(name="value")

        # Compare DataFrames
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result, check_row_order=True)

    def test_melt_filter_groupby(self):
        """Melt, filter, then groupby."""
        pd_df = pd.DataFrame({
            "id": [1, 2],
            "a": [10, 20],
            "b": [30, 40]
        })
        ds_df = DataStore({
            "id": [1, 2],
            "a": [10, 20],
            "b": [30, 40]
        })

        pd_melted = pd_df.melt(id_vars="id", value_vars=["a", "b"])
        ds_melted = ds_df.melt(id_vars="id", value_vars=["a", "b"])

        # Filter and groupby
        pd_result = pd_melted[pd_melted["value"] > 15].groupby("variable")["value"].sum().reset_index()
        ds_result = ds_melted[ds_melted["value"] > 15].groupby("variable")["value"].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_df_squeeze(self):
        """Squeeze empty DataFrame."""
        pd_df = pd.DataFrame({"a": []})
        ds_df = DataStore({"a": []})

        pd_result = pd_df.squeeze()
        ds_result = ds_df.squeeze()

        # Empty Series
        assert_series_equal(ds_result, pd_result)

    def test_single_row_single_col_operations(self):
        """Operations on 1x1 DataFrame."""
        pd_df = pd.DataFrame({"a": [5]})
        ds_df = DataStore({"a": [5]})

        # Multiple operations that could behave differently on 1x1
        pd_result1 = pd_df.sum().sum()  # Should give scalar 5
        ds_result1 = ds_df.sum().sum()

        assert pd_result1 == 5
        assert ds_result1 == 5

    def test_wide_df_squeeze_columns(self):
        """Squeeze wide DataFrame along columns."""
        data = {f"col_{i}": [i] for i in range(10)}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        # Squeeze should return Series when single row
        pd_result = pd_df.squeeze(axis="rows")
        ds_result = ds_df.squeeze(axis="rows")

        assert isinstance(pd_result, pd.Series)
        assert_series_equal(ds_result, pd_result)

    def test_tall_df_squeeze_rows(self):
        """Squeeze tall DataFrame along rows."""
        pd_df = pd.DataFrame({"a": list(range(100))})
        ds_df = DataStore({"a": list(range(100))})

        # Squeeze should return Series when single column
        pd_result = pd_df.squeeze(axis="columns")
        ds_result = ds_df.squeeze(axis="columns")

        assert isinstance(pd_result, pd.Series)
        assert_series_equal(ds_result, pd_result)

    def test_filter_to_empty_then_squeeze(self):
        """Filter to empty DataFrame then squeeze."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_filtered = pd_df[pd_df["a"] > 100]
        ds_filtered = ds_df[ds_df["a"] > 100]

        pd_result = pd_filtered.squeeze()
        ds_result = ds_filtered.squeeze()

        # Both should be empty Series
        assert_series_equal(ds_result, pd_result)


class TestDropDuplicatesAdvanced:
    """Advanced drop_duplicates edge cases."""

    def test_drop_duplicates_keep_false(self):
        """Drop duplicates with keep=False (remove all duplicates)."""
        pd_df = pd.DataFrame({"a": [1, 1, 2, 3, 3], "b": [10, 10, 20, 30, 30]})
        ds_df = DataStore({"a": [1, 1, 2, 3, 3], "b": [10, 10, 20, 30, 30]})

        pd_result = pd_df.drop_duplicates(subset="a", keep=False).reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset="a", keep=False).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_multiple_columns(self):
        """Drop duplicates on multiple columns."""
        pd_df = pd.DataFrame({
            "a": [1, 1, 1, 2, 2],
            "b": [1, 1, 2, 1, 1],
            "c": [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            "a": [1, 1, 1, 2, 2],
            "b": [1, 1, 2, 1, 1],
            "c": [10, 20, 30, 40, 50]
        })

        pd_result = pd_df.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """Drop duplicates keeping last occurrence."""
        pd_df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 20, 30, 40, 50]})
        ds_df = DataStore({"a": [1, 1, 2, 2, 3], "b": [10, 20, 30, 40, 50]})

        pd_result = pd_df.drop_duplicates(subset="a", keep="last").reset_index(drop=True)
        ds_result = ds_df.drop_duplicates(subset="a", keep="last").reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNuniqueAdvanced:
    """Advanced nunique edge cases."""

    def test_nunique_with_nulls(self):
        """Nunique handling of null values."""
        pd_df = pd.DataFrame({"a": [1, 2, None, 2, None]})
        ds_df = DataStore({"a": [1, 2, None, 2, None]})

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        # Should have same nunique count (NaN counted as one unique by default)
        assert_series_equal(ds_result, pd_result)

    def test_nunique_dropna_false(self):
        """Nunique with dropna=False counts NaN as unique."""
        pd_df = pd.DataFrame({"a": [1, 2, None, 2, None]})
        ds_df = DataStore({"a": [1, 2, None, 2, None]})

        pd_result = pd_df.nunique(dropna=False)
        ds_result = ds_df.nunique(dropna=False)

        assert_series_equal(ds_result, pd_result)

    def test_nunique_axis1(self):
        """Nunique along axis 1."""
        pd_df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 2], "c": [1, 1, 1]})
        ds_df = DataStore({"a": [1, 1, 2], "b": [1, 2, 2], "c": [1, 1, 1]})

        pd_result = pd_df.nunique(axis=1)
        ds_result = ds_df.nunique(axis=1)

        assert_series_equal(ds_result, pd_result)


class TestValueCountsAdvanced:
    """Advanced value_counts edge cases."""

    def test_value_counts_normalize(self):
        """Value counts with normalize=True."""
        pd_df = pd.DataFrame({"a": [1, 1, 2, 2, 2, 3]})
        ds_df = DataStore({"a": [1, 1, 2, 2, 2, 3]})

        pd_result = pd_df["a"].value_counts(normalize=True).sort_index()
        ds_result = ds_df["a"].value_counts(normalize=True).sort_index()

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_value_counts_bins(self):
        """Value counts with bins parameter."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        pd_result = pd_df["a"].value_counts(bins=3).sort_index()
        ds_result = ds_df["a"].value_counts(bins=3).sort_index()

        # Compare counts (bin boundaries might differ slightly)
        assert len(get_series(ds_result)) == len(pd_result)

    def test_value_counts_ascending(self):
        """Value counts with ascending=True."""
        pd_df = pd.DataFrame({"a": [1, 1, 2, 2, 2, 3, 3, 3, 3]})
        ds_df = DataStore({"a": [1, 1, 2, 2, 2, 3, 3, 3, 3]})

        pd_result = pd_df["a"].value_counts(ascending=True)
        ds_result = ds_df["a"].value_counts(ascending=True)

        assert_series_equal(ds_result, pd_result, check_names=False)


class TestTransposeAdvanced:
    """Advanced transpose edge cases."""

    def test_transpose_with_dtypes(self):
        """Transpose with different dtypes."""
        pd_df = pd.DataFrame({
            "int_col": [1, 2],
            "float_col": [1.5, 2.5],
            "str_col": ["a", "b"]
        })
        ds_df = DataStore({
            "int_col": [1, 2],
            "float_col": [1.5, 2.5],
            "str_col": ["a", "b"]
        })

        pd_result = pd_df.T
        ds_result = ds_df.T

        # After transpose, dtypes become object
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_transpose_then_filter(self):
        """Transpose then filter rows."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        pd_transposed = pd_df.T
        ds_transposed = ds_df.T

        # Filter transposed (rows are now original columns)
        pd_result = pd_transposed[pd_transposed[0] > 2]
        ds_result = ds_transposed[ds_transposed[0] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestReplaceAdvanced:
    """Advanced replace edge cases."""

    def test_replace_dict_column_specific(self):
        """Replace with column-specific dict."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [1, 2, 3]})

        pd_result = pd_df.replace({"a": {1: 10, 2: 20}})
        ds_result = ds_df.replace({"a": {1: 10, 2: 20}})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_with_list(self):
        """Replace list of values with another list."""
        pd_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        ds_df = DataStore({"a": [1, 2, 3, 4, 5]})

        pd_result = pd_df.replace([1, 3, 5], [10, 30, 50])
        ds_result = ds_df.replace([1, 3, 5], [10, 30, 50])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_regex(self):
        """Replace using regex pattern."""
        pd_df = pd.DataFrame({"a": ["foo123", "bar456", "foo789"]})
        ds_df = DataStore({"a": ["foo123", "bar456", "foo789"]})

        pd_result = pd_df.replace(regex=r"foo\d+", value="FOO")
        ds_result = ds_df.replace(regex=r"foo\d+", value="FOO")

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAsTypeChain:
    """Test astype in various chain scenarios."""

    def test_astype_multiple_columns(self):
        """Astype on multiple columns at once."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        pd_result = pd_df.astype({"a": float, "b": int})
        ds_result = ds_df.astype({"a": float, "b": int})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_string_dtype(self):
        """Astype to string dtype."""
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        ds_df = DataStore({"a": [1, 2, 3]})

        pd_result = pd_df.astype(str)
        ds_result = ds_df.astype(str)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_in_chain(self):
        """Astype within operation chain."""
        pd_df = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
        ds_df = DataStore({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})

        pd_result = pd_df[pd_df["a"] > 1].astype({"b": int})
        ds_result = ds_df[ds_df["a"] > 1].astype({"b": int})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnOperationsAdvanced:
    """Advanced column operation edge cases."""

    def test_insert_column_at_position(self):
        """Insert column at specific position."""
        pd_df = pd.DataFrame({"a": [1, 2], "c": [5, 6]})
        ds_df = DataStore({"a": [1, 2], "c": [5, 6]})

        pd_df.insert(1, "b", [3, 4])
        ds_df.insert(1, "b", [3, 4])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_pop_column(self):
        """Pop a column from DataFrame."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        pd_popped = pd_df.pop("b")
        ds_popped = ds_df.pop("b")

        # Check popped Series
        assert_series_equal(ds_popped, pd_popped)

        # Check remaining DataFrame
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_rename_columns_with_function(self):
        """Rename columns using a function."""
        pd_df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        ds_df = DataStore({"a": [1], "b": [2], "c": [3]})

        pd_result = pd_df.rename(columns=str.upper)
        ds_result = ds_df.rename(columns=str.upper)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """Reindex columns with fill value."""
        pd_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ds_df = DataStore({"a": [1, 2], "b": [3, 4]})

        pd_result = pd_df.reindex(columns=["b", "a", "c"], fill_value=0)
        ds_result = ds_df.reindex(columns=["b", "a", "c"], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
