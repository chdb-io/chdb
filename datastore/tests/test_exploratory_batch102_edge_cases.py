"""
Exploratory tests for deeper edge cases:
- All-NaN column operations
- GroupBy with NaN in keys
- where()/mask() chaining
- apply() with various return types
- Mixed type arithmetic
- Single-row DataFrame operations
- Column operations after filter
- Aggregation on filtered data
- String column arithmetic (concatenation)
- pivot_table edge cases
- Multi-step lazy chain correctness

Mirror Code Pattern: every test compares DataStore vs pandas.
"""

import unittest

import numpy as np
import pandas as pd

from datastore import DataStore
from datastore.tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
    get_value,
)


class TestAllNaNColumn(unittest.TestCase):
    """Test operations on columns that are entirely NaN."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_sum_all_nan_column(self):
        pd_result = self.pd_df["b"].sum()
        ds_result = get_value(self.ds_df["b"].sum())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_mean_all_nan_column(self):
        pd_result = self.pd_df["b"].mean()
        ds_result = get_value(self.ds_df["b"].mean())
        # Both should be NaN
        assert np.isnan(pd_result) and np.isnan(ds_result), (
            f"Expected NaN, got pd={pd_result}, ds={ds_result}"
        )

    def test_count_all_nan_column(self):
        pd_result = self.pd_df["b"].count()
        ds_result = get_value(self.ds_df["b"].count())
        assert pd_result == ds_result == 0, (
            f"Expected 0, got pd={pd_result}, ds={ds_result}"
        )

    def test_dropna_all_nan_column(self):
        pd_result = self.pd_df.dropna(subset=["b"])
        ds_result = self.ds_df.dropna(subset=["b"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_all_nan_column(self):
        pd_result = self.pd_df.fillna(0)
        ds_result = self.ds_df.fillna(0)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSingleRowDataFrame(unittest.TestCase):
    """Test operations on a single-row DataFrame."""

    def setUp(self):
        self.data = {"a": [42], "b": [3.14], "c": ["hello"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_single_row_describe(self):
        pd_result = self.pd_df.describe()
        ds_result = self.ds_df.describe()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_single_row_groupby(self):
        pd_result = self.pd_df.groupby("c")["a"].sum().reset_index()
        ds_result = self.ds_df.groupby("c")["a"].sum().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_single_row_sort(self):
        pd_result = self.pd_df.sort_values("a")
        ds_result = self.ds_df.sort_values("a")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_match(self):
        pd_result = self.pd_df[self.pd_df["a"] > 0]
        ds_result = self.ds_df[self.ds_df["a"] > 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self):
        pd_result = self.pd_df[self.pd_df["a"] > 100]
        ds_result = self.ds_df[self.ds_df["a"] > 100]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_head_tail(self):
        pd_head = self.pd_df.head(1)
        ds_head = self.ds_df.head(1)
        assert_datastore_equals_pandas(ds_head, pd_head)

        pd_tail = self.pd_df.tail(1)
        ds_tail = self.ds_df.tail(1)
        assert_datastore_equals_pandas(ds_tail, pd_tail)


class TestGroupByWithNaN(unittest.TestCase):
    """Test groupby when keys contain NaN values."""

    def setUp(self):
        self.data = {
            "key": ["a", "b", np.nan, "a", np.nan],
            "val": [1, 2, 3, 4, 5],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_groupby_nan_key_default_dropna(self):
        """Default dropna=True should exclude NaN keys."""
        pd_result = self.pd_df.groupby("key")["val"].sum().reset_index()
        ds_result = self.ds_df.groupby("key")["val"].sum().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_nan_key_count(self):
        pd_result = self.pd_df.groupby("key")["val"].count().reset_index()
        ds_result = self.ds_df.groupby("key")["val"].count().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestWhereMaskChaining(unittest.TestCase):
    """Test where() and mask() operations."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_where_basic(self):
        pd_result = self.pd_df.where(self.pd_df > 2)
        ds_result = self.ds_df.where(self.ds_df > 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_with_other(self):
        pd_result = self.pd_df.where(self.pd_df > 2, other=-1)
        ds_result = self.ds_df.where(self.ds_df > 2, other=-1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mask_basic(self):
        pd_result = self.pd_df.mask(self.pd_df > 3)
        ds_result = self.ds_df.mask(self.ds_df > 3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mask_with_other(self):
        pd_result = self.pd_df.mask(self.pd_df > 3, other=0)
        ds_result = self.ds_df.mask(self.ds_df > 3, other=0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_series(self):
        pd_result = self.pd_df["a"].where(self.pd_df["a"] > 2)
        ds_result = self.ds_df["a"].where(self.ds_df["a"] > 2)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_mask_series(self):
        pd_result = self.pd_df["a"].mask(self.pd_df["a"] > 3, other=0)
        ds_result = self.ds_df["a"].mask(self.ds_df["a"] > 3, other=0)
        assert_series_equal(get_series(ds_result), pd_result)


class TestStringColumnOperations(unittest.TestCase):
    """Test string-specific operations."""

    def setUp(self):
        self.data = {
            "name": ["Alice", "Bob", "Charlie", "David"],
            "city": ["New York", "Los Angeles", "Chicago", "Houston"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_str_upper(self):
        pd_result = self.pd_df["name"].str.upper()
        ds_result = self.ds_df["name"].str.upper()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_lower(self):
        pd_result = self.pd_df["name"].str.lower()
        ds_result = self.ds_df["name"].str.lower()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_len(self):
        pd_result = self.pd_df["name"].str.len()
        ds_result = self.ds_df["name"].str.len()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_contains(self):
        pd_result = self.pd_df["name"].str.contains("li")
        ds_result = self.ds_df["name"].str.contains("li")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_startswith(self):
        pd_result = self.pd_df["name"].str.startswith("A")
        ds_result = self.ds_df["name"].str.startswith("A")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_str_slice(self):
        pd_result = self.pd_df["name"].str[:3]
        ds_result = self.ds_df["name"].str[:3]
        assert_series_equal(get_series(ds_result), pd_result)

    def test_filter_by_string_then_aggregate(self):
        pd_result = self.pd_df[self.pd_df["name"].str.startswith("C")]
        ds_result = self.ds_df[self.ds_df["name"].str.startswith("C")]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMixedTypeArithmetic(unittest.TestCase):
    """Test arithmetic between columns of different types."""

    def setUp(self):
        self.data = {
            "int_col": [1, 2, 3, 4],
            "float_col": [1.5, 2.5, 3.5, 4.5],
            "bool_col": [True, False, True, False],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_int_plus_float(self):
        pd_result = self.pd_df["int_col"] + self.pd_df["float_col"]
        ds_result = self.ds_df["int_col"] + self.ds_df["float_col"]
        assert_series_equal(get_series(ds_result), pd_result)

    def test_int_times_bool(self):
        pd_result = self.pd_df["int_col"] * self.pd_df["bool_col"]
        ds_result = self.ds_df["int_col"] * self.ds_df["bool_col"]
        assert_series_equal(get_series(ds_result), pd_result)

    def test_float_minus_int(self):
        pd_result = self.pd_df["float_col"] - self.pd_df["int_col"]
        ds_result = self.ds_df["float_col"] - self.ds_df["int_col"]
        assert_series_equal(get_series(ds_result), pd_result)

    def test_column_plus_scalar(self):
        pd_result = self.pd_df["int_col"] + 10
        ds_result = self.ds_df["int_col"] + 10
        assert_series_equal(get_series(ds_result), pd_result)

    def test_scalar_minus_column(self):
        pd_result = 10 - self.pd_df["int_col"]
        ds_result = 10 - self.ds_df["int_col"]
        assert_series_equal(get_series(ds_result), pd_result)


class TestColumnAfterFilter(unittest.TestCase):
    """Test accessing columns after filtering."""

    def setUp(self):
        self.data = {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [85, 92, 78, 95, 88],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_filter_then_column_access(self):
        pd_result = self.pd_df[self.pd_df["age"] > 30]["name"]
        ds_result = self.ds_df[self.ds_df["age"] > 30]["name"]
        assert_series_equal(get_series(ds_result), pd_result)

    def test_filter_then_multiple_columns(self):
        pd_result = self.pd_df[self.pd_df["age"] > 30][["name", "score"]]
        ds_result = self.ds_df[self.ds_df["age"] > 30][["name", "score"]]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_agg(self):
        pd_result = self.pd_df[self.pd_df["age"] > 30]["score"].mean()
        ds_result = get_value(self.ds_df[self.ds_df["age"] > 30]["score"].mean())
        assert abs(pd_result - ds_result) < 1e-5, (
            f"Expected {pd_result}, got {ds_result}"
        )

    def test_filter_then_sort_then_head(self):
        pd_result = (
            self.pd_df[self.pd_df["age"] > 25]
            .sort_values("score", ascending=False)
            .head(2)
        )
        ds_result = (
            self.ds_df[self.ds_df["age"] > 25]
            .sort_values("score", ascending=False)
            .head(2)
        )
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggOnFiltered(unittest.TestCase):
    """Test aggregation on filtered DataFrames."""

    def setUp(self):
        self.data = {
            "group": ["A", "A", "B", "B", "C", "C"],
            "val": [10, 20, 30, 40, 50, 60],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_filter_then_sum(self):
        pd_result = self.pd_df[self.pd_df["val"] > 20]["val"].sum()
        ds_result = get_value(self.ds_df[self.ds_df["val"] > 20]["val"].sum())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_filter_then_groupby_sum(self):
        pd_result = (
            self.pd_df[self.pd_df["val"] > 20]
            .groupby("group")["val"]
            .sum()
            .reset_index()
        )
        ds_result = (
            self.ds_df[self.ds_df["val"] > 20]
            .groupby("group")["val"]
            .sum()
            .reset_index()
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_then_groupby_mean(self):
        pd_result = (
            self.pd_df[self.pd_df["val"] > 15]
            .groupby("group")["val"]
            .mean()
            .reset_index()
        )
        ds_result = (
            self.ds_df[self.ds_df["val"] > 15]
            .groupby("group")["val"]
            .mean()
            .reset_index()
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestPivotTableEdgeCases(unittest.TestCase):
    """Test pivot_table with various configurations."""

    def setUp(self):
        self.data = {
            "name": ["Alice", "Alice", "Bob", "Bob"],
            "subject": ["math", "science", "math", "science"],
            "score": [90, 85, 80, 95],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_pivot_table_basic(self):
        pd_result = self.pd_df.pivot_table(
            values="score", index="name", columns="subject", aggfunc="mean"
        )
        ds_result = self.ds_df.pivot_table(
            values="score", index="name", columns="subject", aggfunc="mean"
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_pivot_table_sum(self):
        pd_result = self.pd_df.pivot_table(
            values="score", index="name", columns="subject", aggfunc="sum"
        )
        ds_result = self.ds_df.pivot_table(
            values="score", index="name", columns="subject", aggfunc="sum"
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestMultiStepLazyChain(unittest.TestCase):
    """Test multi-step chains to verify lazy execution correctness."""

    def setUp(self):
        self.data = {
            "dept": ["eng", "eng", "eng", "sales", "sales", "sales"],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            "salary": [100000, 120000, 90000, 80000, 95000, 85000],
            "years": [5, 8, 3, 6, 4, 7],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_filter_assign_sort_head(self):
        pd_result = (
            self.pd_df[self.pd_df["salary"] > 85000]
            .assign(sal_per_year=lambda df: df["salary"] / df["years"])
            .sort_values("sal_per_year", ascending=False)
            .head(3)
        )
        ds_result = (
            self.ds_df[self.ds_df["salary"] > 85000]
            .assign(sal_per_year=lambda df: df["salary"] / df["years"])
            .sort_values("sal_per_year", ascending=False)
            .head(3)
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_then_filter_result(self):
        pd_agg = self.pd_df.groupby("dept")["salary"].mean().reset_index()
        pd_result = pd_agg[pd_agg["salary"] > 90000]
        ds_agg = self.ds_df.groupby("dept")["salary"].mean().reset_index()
        ds_result = ds_agg[ds_agg["salary"] > 90000]
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_rename_filter_groupby_sort(self):
        pd_result = (
            self.pd_df.rename(columns={"dept": "department"})
            .query("salary > 85000")
            .groupby("department")["salary"]
            .mean()
            .reset_index()
            .sort_values("salary", ascending=False)
        )
        ds_result = (
            self.ds_df.rename(columns={"dept": "department"})
            .query("salary > 85000")
            .groupby("department")["salary"]
            .mean()
            .reset_index()
            .sort_values("salary", ascending=False)
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters(self):
        pd_result = self.pd_df[
            (self.pd_df["salary"] > 85000) & (self.pd_df["years"] > 3)
        ]
        ds_result = self.ds_df[
            (self.ds_df["salary"] > 85000) & (self.ds_df["years"] > 3)
        ]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullHandlingEdgeCases(unittest.TestCase):
    """Test null/NaN handling in various operations."""

    def setUp(self):
        self.data = {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [10.0, 20.0, np.nan, 40.0, np.nan],
            "group": ["x", "x", "y", "y", "y"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_isna(self):
        pd_result = self.pd_df["a"].isna()
        ds_result = self.ds_df["a"].isna()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_notna(self):
        pd_result = self.pd_df["a"].notna()
        ds_result = self.ds_df["a"].notna()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_isna_dataframe(self):
        pd_result = self.pd_df[["a", "b"]].isna()
        ds_result = self.ds_df[["a", "b"]].isna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_notna(self):
        pd_result = self.pd_df[self.pd_df["a"].notna()]
        ds_result = self.ds_df[self.ds_df["a"].notna()]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_with_nan_values(self):
        pd_result = self.pd_df.groupby("group")["a"].sum().reset_index()
        ds_result = self.ds_df.groupby("group")["a"].sum().reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_arithmetic_with_nan(self):
        pd_result = self.pd_df["a"] + self.pd_df["b"]
        ds_result = self.ds_df["a"] + self.ds_df["b"]
        assert_series_equal(get_series(ds_result), pd_result)

    def test_comparison_with_nan(self):
        pd_result = self.pd_df["a"] > 2
        ds_result = self.ds_df["a"] > 2
        assert_series_equal(get_series(ds_result), pd_result)


class TestIlocAccess(unittest.TestCase):
    """Test iloc-based integer indexing."""

    def setUp(self):
        self.data = {"a": [10, 20, 30, 40, 50], "b": ["x", "y", "z", "w", "v"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_iloc_single_row(self):
        pd_result = self.pd_df.iloc[0]
        ds_result = self.ds_df.iloc[0]
        # Single row returns a Series
        ds_series = get_series(ds_result)
        pd.testing.assert_series_equal(ds_series, pd_result)

    def test_iloc_slice(self):
        pd_result = self.pd_df.iloc[1:3]
        ds_result = self.ds_df.iloc[1:3]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_negative(self):
        pd_result = self.pd_df.iloc[-1]
        ds_result = self.ds_df.iloc[-1]
        ds_series = get_series(ds_result)
        pd.testing.assert_series_equal(ds_series, pd_result)

    def test_iloc_list(self):
        pd_result = self.pd_df.iloc[[0, 2, 4]]
        ds_result = self.ds_df.iloc[[0, 2, 4]]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestHeadTailEdgeCases(unittest.TestCase):
    """Test head() and tail() with edge cases."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_head_more_than_length(self):
        pd_result = self.pd_df.head(10)
        ds_result = self.ds_df.head(10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_more_than_length(self):
        pd_result = self.pd_df.tail(10)
        ds_result = self.ds_df.tail(10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        pd_result = self.pd_df.head(0)
        ds_result = self.ds_df.head(0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_zero(self):
        pd_result = self.pd_df.tail(0)
        ds_result = self.ds_df.tail(0)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnCreationAndDeletion(unittest.TestCase):
    """Test column assignment and deletion."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_new_column_from_expression(self):
        self.pd_df["c"] = self.pd_df["a"] + self.pd_df["b"]
        self.ds_df["c"] = self.ds_df["a"] + self.ds_df["b"]
        assert_datastore_equals_pandas(self.ds_df, self.pd_df)

    def test_new_column_scalar(self):
        self.pd_df["c"] = 100
        self.ds_df["c"] = 100
        assert_datastore_equals_pandas(self.ds_df, self.pd_df)

    def test_overwrite_column(self):
        self.pd_df["a"] = self.pd_df["a"] * 2
        self.ds_df["a"] = self.ds_df["a"] * 2
        assert_datastore_equals_pandas(self.ds_df, self.pd_df)

    def test_drop_column(self):
        pd_result = self.pd_df.drop(columns=["b"])
        ds_result = self.ds_df.drop(columns=["b"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_multiple_columns(self):
        data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.drop(columns=["a", "c"])
        ds_result = ds_df.drop(columns=["a", "c"])
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBetweenOperation(unittest.TestCase):
    """Test between() operation."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_between_inclusive_both(self):
        pd_result = self.pd_df[self.pd_df["a"].between(3, 7)]
        ds_result = self.ds_df[self.ds_df["a"].between(3, 7)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_inclusive_neither(self):
        pd_result = self.pd_df[self.pd_df["a"].between(3, 7, inclusive="neither")]
        ds_result = self.ds_df[self.ds_df["a"].between(3, 7, inclusive="neither")]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_inclusive_left(self):
        pd_result = self.pd_df[self.pd_df["a"].between(3, 7, inclusive="left")]
        ds_result = self.ds_df[self.ds_df["a"].between(3, 7, inclusive="left")]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_inclusive_right(self):
        pd_result = self.pd_df[self.pd_df["a"].between(3, 7, inclusive="right")]
        ds_result = self.ds_df[self.ds_df["a"].between(3, 7, inclusive="right")]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMinMaxWithMixedTypes(unittest.TestCase):
    """Test min/max aggregation with various column types."""

    def setUp(self):
        self.data = {
            "int_col": [3, 1, 4, 1, 5],
            "float_col": [2.7, 1.4, 3.1, 0.9, 2.6],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_min_int(self):
        pd_result = self.pd_df["int_col"].min()
        ds_result = get_value(self.ds_df["int_col"].min())
        assert pd_result == ds_result

    def test_max_int(self):
        pd_result = self.pd_df["int_col"].max()
        ds_result = get_value(self.ds_df["int_col"].max())
        assert pd_result == ds_result

    def test_min_float(self):
        pd_result = self.pd_df["float_col"].min()
        ds_result = get_value(self.ds_df["float_col"].min())
        assert abs(pd_result - ds_result) < 1e-10

    def test_max_float(self):
        pd_result = self.pd_df["float_col"].max()
        ds_result = get_value(self.ds_df["float_col"].max())
        assert abs(pd_result - ds_result) < 1e-10

    def test_idxmin(self):
        pd_result = self.pd_df["int_col"].idxmin()
        ds_result = get_value(self.ds_df["int_col"].idxmin())
        assert pd_result == ds_result

    def test_idxmax(self):
        pd_result = self.pd_df["int_col"].idxmax()
        ds_result = get_value(self.ds_df["int_col"].idxmax())
        assert pd_result == ds_result

    def test_dataframe_min(self):
        pd_result = self.pd_df.min()
        ds_result = self.ds_df.min()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_dataframe_max(self):
        pd_result = self.pd_df.max()
        ds_result = self.ds_df.max()
        assert_series_equal(get_series(ds_result), pd_result)


class TestApplyEdgeCases(unittest.TestCase):
    """Test apply() with various functions."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_apply_lambda_series(self):
        pd_result = self.pd_df["a"].apply(lambda x: x ** 2)
        ds_result = self.ds_df["a"].apply(lambda x: x ** 2)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_apply_lambda_dataframe_axis0(self):
        pd_result = self.pd_df.apply(lambda col: col.sum())
        ds_result = self.ds_df.apply(lambda col: col.sum())
        assert_series_equal(get_series(ds_result), pd_result)

    def test_apply_lambda_dataframe_axis1(self):
        pd_result = self.pd_df.apply(lambda row: row["a"] + row["b"], axis=1)
        ds_result = self.ds_df.apply(lambda row: row["a"] + row["b"], axis=1)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_apply_builtin(self):
        pd_result = self.pd_df["a"].apply(str)
        ds_result = self.ds_df["a"].apply(str)
        assert_series_equal(get_series(ds_result), pd_result)


class TestConcatEdgeCases(unittest.TestCase):
    """Test pd.concat equivalent operations."""

    def setUp(self):
        from datastore import concat as ds_concat
        self.ds_concat = ds_concat
        self.data1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        self.data2 = {"a": [7, 8, 9], "b": [10, 11, 12]}
        self.pd_df1 = pd.DataFrame(self.data1)
        self.pd_df2 = pd.DataFrame(self.data2)
        self.ds_df1 = DataStore(self.data1)
        self.ds_df2 = DataStore(self.data2)

    def test_concat_vertical(self):
        pd_result = pd.concat([self.pd_df1, self.pd_df2], ignore_index=True)
        ds_result = self.ds_concat([self.ds_df1, self.ds_df2], ignore_index=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_different_columns(self):
        data3 = {"a": [1, 2], "c": [7, 8]}
        pd_df3 = pd.DataFrame(data3)
        ds_df3 = DataStore(data3)
        pd_result = pd.concat([self.pd_df1, pd_df3], ignore_index=True)
        ds_result = self.ds_concat([self.ds_df1, ds_df3], ignore_index=True)
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == "__main__":
    unittest.main()
