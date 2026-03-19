"""
Exploratory tests for under-tested pandas operations:
- rename() with dict and function
- replace() with various patterns
- nlargest() / nsmallest() with ties
- rank() with all methods
- cumulative operations with NaN
- assign() with multiple columns
- select_dtypes() include/exclude
- duplicated() with subset and keep options
- value_counts() with normalize
- clip() edge cases
- diff() with various periods
- complex chained operations

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


class TestRenameOperations(unittest.TestCase):
    """Test rename() with various argument patterns."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_rename_columns_with_dict(self):
        pd_result = self.pd_df.rename(columns={"a": "x", "b": "y"})
        ds_result = self.ds_df.rename(columns={"a": "x", "b": "y"})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_columns_with_function(self):
        pd_result = self.pd_df.rename(columns=str.upper)
        ds_result = self.ds_df.rename(columns=str.upper)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_columns_with_lambda(self):
        pd_result = self.pd_df.rename(columns=lambda x: f"col_{x}")
        ds_result = self.ds_df.rename(columns=lambda x: f"col_{x}")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_partial_columns(self):
        pd_result = self.pd_df.rename(columns={"a": "alpha"})
        ds_result = self.ds_df.rename(columns={"a": "alpha"})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_nonexistent_column(self):
        pd_result = self.pd_df.rename(columns={"z": "zz"})
        ds_result = self.ds_df.rename(columns={"z": "zz"})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_does_not_mutate_original(self):
        _ = self.ds_df.rename(columns={"a": "x"})
        pd_result = self.pd_df
        ds_result = self.ds_df
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestReplaceOperations(unittest.TestCase):
    """Test replace() with various patterns."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 2, 1], "b": ["x", "y", "z", "x", "y"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_replace_scalar(self):
        pd_result = self.pd_df.replace(1, 100)
        ds_result = self.ds_df.replace(1, 100)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict_mapping(self):
        pd_result = self.pd_df.replace({"a": {1: 10, 2: 20}})
        ds_result = self.ds_df.replace({"a": {1: 10, 2: 20}})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_to_scalar(self):
        pd_result = self.pd_df.replace([1, 2], 0)
        ds_result = self.ds_df.replace([1, 2], 0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_to_list(self):
        pd_result = self.pd_df.replace([1, 2], [10, 20])
        ds_result = self.ds_df.replace([1, 2], [10, 20])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_string_values(self):
        pd_result = self.pd_df.replace("x", "replaced")
        ds_result = self.ds_df.replace("x", "replaced")
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNlargestNsmallest(unittest.TestCase):
    """Test nlargest() and nsmallest() on DataStore."""

    def setUp(self):
        self.data = {
            "val": [3, 1, 4, 1, 5, 9, 2, 6],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_nlargest_basic(self):
        pd_result = self.pd_df.nlargest(3, "val")
        ds_result = self.ds_df.nlargest(3, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_basic(self):
        pd_result = self.pd_df.nsmallest(3, "val")
        ds_result = self.ds_df.nsmallest(3, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_ties_keep_first(self):
        data = {"val": [3, 3, 3, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nlargest(2, "val", keep="first")
        ds_result = ds_df.nlargest(2, "val", keep="first")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_with_ties_keep_first(self):
        data = {"val": [1, 1, 1, 3, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nsmallest(2, "val", keep="first")
        ds_result = ds_df.nsmallest(2, "val", keep="first")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_ties_keep_all(self):
        data = {"val": [3, 3, 3, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nlargest(2, "val", keep="all")
        ds_result = ds_df.nlargest(2, "val", keep="all")
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRankMethods(unittest.TestCase):
    """Test rank() with all methods."""

    def setUp(self):
        self.data = {"val": [3.0, 1.0, 4.0, 1.0, 5.0]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_rank_average(self):
        pd_result = self.pd_df["val"].rank(method="average")
        ds_result = self.ds_df["val"].rank(method="average")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_rank_min(self):
        pd_result = self.pd_df["val"].rank(method="min")
        ds_result = self.ds_df["val"].rank(method="min")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_rank_max(self):
        pd_result = self.pd_df["val"].rank(method="max")
        ds_result = self.ds_df["val"].rank(method="max")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_rank_first(self):
        pd_result = self.pd_df["val"].rank(method="first")
        ds_result = self.ds_df["val"].rank(method="first")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_rank_dense(self):
        pd_result = self.pd_df["val"].rank(method="dense")
        ds_result = self.ds_df["val"].rank(method="dense")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_rank_ascending_false(self):
        pd_result = self.pd_df["val"].rank(ascending=False)
        ds_result = self.ds_df["val"].rank(ascending=False)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_rank_with_nan(self):
        data = {"val": [3.0, np.nan, 1.0, np.nan, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["val"].rank()
        ds_result = ds_df["val"].rank()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_dataframe_rank(self):
        data = {"a": [3, 1, 4], "b": [6, 5, 4]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.rank()
        ds_result = ds_df.rank()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCumulativeWithNaN(unittest.TestCase):
    """Test cumulative operations with NaN values."""

    def setUp(self):
        self.data = {"a": [1.0, np.nan, 3.0, 4.0], "b": [np.nan, 2.0, np.nan, 4.0]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_cumsum_with_nan(self):
        pd_result = self.pd_df.cumsum()
        ds_result = self.ds_df.cumsum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummin_with_nan(self):
        pd_result = self.pd_df.cummin()
        ds_result = self.ds_df.cummin()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cummax_with_nan(self):
        pd_result = self.pd_df.cummax()
        ds_result = self.ds_df.cummax()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cumsum_series_with_nan(self):
        pd_result = self.pd_df["a"].cumsum()
        ds_result = self.ds_df["a"].cumsum()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_cummin_series_with_nan(self):
        pd_result = self.pd_df["a"].cummin()
        ds_result = self.ds_df["a"].cummin()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_cummax_series_with_nan(self):
        pd_result = self.pd_df["a"].cummax()
        ds_result = self.ds_df["a"].cummax()
        assert_series_equal(get_series(ds_result), pd_result)


class TestAssignOperation(unittest.TestCase):
    """Test assign() for creating new columns."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_assign_scalar(self):
        pd_result = self.pd_df.assign(c=10)
        ds_result = self.ds_df.assign(c=10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_from_existing_column(self):
        pd_result = self.pd_df.assign(c=self.pd_df["a"] * 2)
        ds_result = self.ds_df.assign(c=self.ds_df["a"] * 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_lambda(self):
        pd_result = self.pd_df.assign(c=lambda df: df["a"] + df["b"])
        ds_result = self.ds_df.assign(c=lambda df: df["a"] + df["b"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_columns(self):
        pd_result = self.pd_df.assign(c=10, d=20)
        ds_result = self.ds_df.assign(c=10, d=20)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_does_not_mutate_original(self):
        _ = self.ds_df.assign(c=10)
        pd_result = self.pd_df
        ds_result = self.ds_df
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSelectDtypes(unittest.TestCase):
    """Test select_dtypes() with include/exclude."""

    def setUp(self):
        self.data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_select_dtypes_include_number(self):
        pd_result = self.pd_df.select_dtypes(include=["number"])
        ds_result = self.ds_df.select_dtypes(include=["number"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_object(self):
        pd_result = self.pd_df.select_dtypes(include=["object"])
        ds_result = self.ds_df.select_dtypes(include=["object"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_object(self):
        pd_result = self.pd_df.select_dtypes(exclude=["object"])
        ds_result = self.ds_df.select_dtypes(exclude=["object"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_int(self):
        pd_result = self.pd_df.select_dtypes(include=["int64"])
        ds_result = self.ds_df.select_dtypes(include=["int64"])
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDuplicatedOperations(unittest.TestCase):
    """Test duplicated() with subset and keep options."""

    def setUp(self):
        self.data = {
            "a": [1, 2, 2, 3, 3, 3],
            "b": ["x", "y", "y", "z", "z", "z"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_duplicated_default(self):
        pd_result = self.pd_df.duplicated()
        ds_result = self.ds_df.duplicated()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_duplicated_keep_last(self):
        pd_result = self.pd_df.duplicated(keep="last")
        ds_result = self.ds_df.duplicated(keep="last")
        assert_series_equal(get_series(ds_result), pd_result)

    def test_duplicated_keep_false(self):
        pd_result = self.pd_df.duplicated(keep=False)
        ds_result = self.ds_df.duplicated(keep=False)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_duplicated_subset(self):
        pd_result = self.pd_df.duplicated(subset=["a"])
        ds_result = self.ds_df.duplicated(subset=["a"])
        assert_series_equal(get_series(ds_result), pd_result)

    def test_drop_duplicates_keep_last(self):
        pd_result = self.pd_df.drop_duplicates(keep="last")
        ds_result = self.ds_df.drop_duplicates(keep="last")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset(self):
        pd_result = self.pd_df.drop_duplicates(subset=["a"])
        ds_result = self.ds_df.drop_duplicates(subset=["a"])
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestValueCountsNormalize(unittest.TestCase):
    """Test value_counts() with normalize parameter."""

    def setUp(self):
        self.data = {"a": [1, 2, 2, 3, 3, 3], "b": ["x", "x", "y", "y", "z", "z"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_series_value_counts_default(self):
        pd_result = self.pd_df["a"].value_counts()
        ds_result = self.ds_df["a"].value_counts()
        pd_series = get_series(ds_result)
        assert_series_equal(pd_series.sort_index(), pd_result.sort_index())

    def test_series_value_counts_normalize(self):
        pd_result = self.pd_df["a"].value_counts(normalize=True)
        ds_result = self.ds_df["a"].value_counts(normalize=True)
        pd_series = get_series(ds_result)
        assert_series_equal(pd_series.sort_index(), pd_result.sort_index())

    def test_series_value_counts_ascending(self):
        pd_result = self.pd_df["a"].value_counts(ascending=True)
        ds_result = self.ds_df["a"].value_counts(ascending=True)
        pd_series = get_series(ds_result)
        assert_series_equal(pd_series.sort_index(), pd_result.sort_index())

    def test_dataframe_value_counts(self):
        pd_result = self.pd_df.value_counts()
        ds_result = self.ds_df.value_counts()
        pd_series = get_series(ds_result)
        assert_series_equal(
            pd_series.sort_index(), pd_result.sort_index()
        )


class TestClipEdgeCases(unittest.TestCase):
    """Test clip() with various boundary combinations."""

    def setUp(self):
        self.data = {"a": [1, 5, 10, 15, 20], "b": [2.0, 4.0, 6.0, 8.0, 10.0]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_clip_both_bounds(self):
        pd_result = self.pd_df.clip(lower=5, upper=15)
        ds_result = self.ds_df.clip(lower=5, upper=15)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_lower_only(self):
        pd_result = self.pd_df.clip(lower=5)
        ds_result = self.ds_df.clip(lower=5)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_upper_only(self):
        pd_result = self.pd_df.clip(upper=15)
        ds_result = self.ds_df.clip(upper=15)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_series(self):
        pd_result = self.pd_df["a"].clip(lower=5, upper=15)
        ds_result = self.ds_df["a"].clip(lower=5, upper=15)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_clip_with_nan(self):
        data = {"a": [1.0, np.nan, 10.0, np.nan, 20.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.clip(lower=5, upper=15)
        ds_result = ds_df.clip(lower=5, upper=15)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDiffOperations(unittest.TestCase):
    """Test diff() with various periods."""

    def setUp(self):
        self.data = {"a": [1, 3, 6, 10, 15], "b": [2.0, 4.0, 8.0, 16.0, 32.0]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_diff_default(self):
        pd_result = self.pd_df.diff()
        ds_result = self.ds_df.diff()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_periods_2(self):
        pd_result = self.pd_df.diff(periods=2)
        ds_result = self.ds_df.diff(periods=2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_diff_series(self):
        pd_result = self.pd_df["a"].diff()
        ds_result = self.ds_df["a"].diff()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_diff_with_nan(self):
        data = {"a": [1.0, np.nan, 6.0, 10.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.diff()
        ds_result = ds_df.diff()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChainedOperations(unittest.TestCase):
    """Test complex chains of operations for correctness."""

    def setUp(self):
        self.data = {
            "name": ["Alice", "Bob", "Charlie", "Alice", "Bob", "Charlie"],
            "score": [85, 92, 78, 90, 88, 95],
            "subject": ["math", "math", "math", "science", "science", "science"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_filter_then_groupby_agg(self):
        pd_result = (
            self.pd_df[self.pd_df["score"] > 80]
            .groupby("name")["score"]
            .mean()
            .reset_index()
        )
        ds_result = (
            self.ds_df[self.ds_df["score"] > 80]
            .groupby("name")["score"]
            .mean()
            .reset_index()
        )
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_groupby_agg_then_sort(self):
        pd_result = (
            self.pd_df.groupby("name")["score"]
            .sum()
            .reset_index()
            .sort_values("score", ascending=False)
        )
        ds_result = (
            self.ds_df.groupby("name")["score"]
            .sum()
            .reset_index()
            .sort_values("score", ascending=False)
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self):
        pd_result = self.pd_df.assign(
            doubled=self.pd_df["score"] * 2
        )
        pd_result = pd_result[pd_result["doubled"] > 170]

        ds_result = self.ds_df.assign(
            doubled=self.ds_df["score"] * 2
        )
        ds_result = ds_result[ds_result["doubled"] > 170]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_groupby(self):
        pd_result = (
            self.pd_df.rename(columns={"name": "student", "score": "grade"})
            .groupby("student")["grade"]
            .mean()
            .reset_index()
        )
        ds_result = (
            self.ds_df.rename(columns={"name": "student", "score": "grade"})
            .groupby("student")["grade"]
            .mean()
            .reset_index()
        )
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_filter_then_nlargest(self):
        pd_result = self.pd_df[self.pd_df["subject"] == "math"].nlargest(
            2, "score"
        )
        ds_result = self.ds_df[self.ds_df["subject"] == "math"].nlargest(
            2, "score"
        )
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanIndexingComplex(unittest.TestCase):
    """Test complex boolean indexing with multiple conditions."""

    def setUp(self):
        self.data = {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "x", "y", "x"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_and_condition(self):
        pd_result = self.pd_df[(self.pd_df["a"] > 2) & (self.pd_df["b"] < 50)]
        ds_result = self.ds_df[(self.ds_df["a"] > 2) & (self.ds_df["b"] < 50)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_condition(self):
        pd_result = self.pd_df[(self.pd_df["a"] < 2) | (self.pd_df["a"] > 4)]
        ds_result = self.ds_df[(self.ds_df["a"] < 2) | (self.ds_df["a"] > 4)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_not_condition(self):
        pd_result = self.pd_df[~(self.pd_df["a"] > 3)]
        ds_result = self.ds_df[~(self.ds_df["a"] > 3)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combined_and_or(self):
        pd_result = self.pd_df[
            ((self.pd_df["a"] > 1) & (self.pd_df["b"] > 20))
            | (self.pd_df["c"] == "x")
        ]
        ds_result = self.ds_df[
            ((self.ds_df["a"] > 1) & (self.ds_df["b"] > 20))
            | (self.ds_df["c"] == "x")
        ]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_filter(self):
        pd_result = self.pd_df[self.pd_df["a"].isin([1, 3, 5])]
        ds_result = self.ds_df[self.ds_df["a"].isin([1, 3, 5])]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDescribeAndInfo(unittest.TestCase):
    """Test describe() with various options."""

    def setUp(self):
        self.data = {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "a", "b"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_describe_default(self):
        pd_result = self.pd_df.describe()
        ds_result = self.ds_df.describe()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_describe_include_all(self):
        pd_result = self.pd_df.describe(include="all")
        ds_result = self.ds_df.describe(include="all")
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_describe_percentiles(self):
        pd_result = self.pd_df.describe(percentiles=[0.1, 0.5, 0.9])
        ds_result = self.ds_df.describe(percentiles=[0.1, 0.5, 0.9])
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestPipeOperation(unittest.TestCase):
    """Test pipe() for functional chaining."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_pipe_with_function(self):
        def add_column(df):
            return df.assign(c=100)

        pd_result = self.pd_df.pipe(add_column)
        ds_result = self.ds_df.pipe(add_column)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_args(self):
        def filter_col(df, col, threshold):
            return df[df[col] > threshold]

        pd_result = self.pd_df.pipe(filter_col, "a", 3)
        ds_result = self.ds_df.pipe(filter_col, "a", 3)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMeltOperation(unittest.TestCase):
    """Test melt() / unpivot operation."""

    def setUp(self):
        self.data = {
            "id": [1, 2, 3],
            "math": [90, 80, 70],
            "science": [85, 95, 75],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_melt_basic(self):
        pd_result = self.pd_df.melt(id_vars=["id"], value_vars=["math", "science"])
        ds_result = self.ds_df.melt(id_vars=["id"], value_vars=["math", "science"])
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_melt_with_var_name(self):
        pd_result = self.pd_df.melt(
            id_vars=["id"],
            value_vars=["math", "science"],
            var_name="subject",
            value_name="score",
        )
        ds_result = self.ds_df.melt(
            id_vars=["id"],
            value_vars=["math", "science"],
            var_name="subject",
            value_name="score",
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestFillnaEdgeCases(unittest.TestCase):
    """Test fillna() with various strategies."""

    def setUp(self):
        self.data = {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, 2.0, np.nan, 4.0, np.nan],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_fillna_scalar(self):
        pd_result = self.pd_df.fillna(0)
        ds_result = self.ds_df.fillna(0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict(self):
        pd_result = self.pd_df.fillna({"a": -1, "b": -2})
        ds_result = self.ds_df.fillna({"a": -1, "b": -2})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_ffill(self):
        pd_result = self.pd_df.ffill()
        ds_result = self.ds_df.ffill()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_bfill(self):
        pd_result = self.pd_df.bfill()
        ds_result = self.ds_df.bfill()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDropnaEdgeCases(unittest.TestCase):
    """Test dropna() with various parameters."""

    def setUp(self):
        self.data = {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, 2.0, 3.0, np.nan],
            "c": [1.0, 2.0, 3.0, 4.0],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_dropna_any(self):
        pd_result = self.pd_df.dropna(how="any")
        ds_result = self.ds_df.dropna(how="any")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all(self):
        pd_result = self.pd_df.dropna(how="all")
        ds_result = self.ds_df.dropna(how="all")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset(self):
        pd_result = self.pd_df.dropna(subset=["a"])
        ds_result = self.ds_df.dropna(subset=["a"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_thresh(self):
        pd_result = self.pd_df.dropna(thresh=2)
        ds_result = self.ds_df.dropna(thresh=2)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAbsEdgeCases(unittest.TestCase):
    """Test abs() with various types."""

    def setUp(self):
        self.data = {"a": [-1, -2, 3, -4, 5], "b": [-1.5, 2.5, -3.5, 4.5, -5.5]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_abs_dataframe(self):
        pd_result = self.pd_df.abs()
        ds_result = self.ds_df.abs()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_abs_series(self):
        pd_result = self.pd_df["a"].abs()
        ds_result = self.ds_df["a"].abs()
        assert_series_equal(get_series(ds_result), pd_result)

    def test_abs_mixed_types(self):
        data = {"int_col": [-1, 2, -3], "float_col": [-1.1, 2.2, -3.3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.abs()
        ds_result = ds_df.abs()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAsTypeEdgeCases(unittest.TestCase):
    """Test astype() conversions."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [1.1, 2.2, 3.3]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_astype_int_to_float(self):
        pd_result = self.pd_df["a"].astype(float)
        ds_result = self.ds_df["a"].astype(float)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_float_to_int(self):
        pd_result = self.pd_df["b"].astype(int)
        ds_result = self.ds_df["b"].astype(int)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_to_string(self):
        pd_result = self.pd_df["a"].astype(str)
        ds_result = self.ds_df["a"].astype(str)
        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_dict(self):
        pd_result = self.pd_df.astype({"a": float, "b": int})
        ds_result = self.ds_df.astype({"a": float, "b": int})
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestQueryMethod(unittest.TestCase):
    """Test query() method with string expressions."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_query_simple(self):
        pd_result = self.pd_df.query("a > 3")
        ds_result = self.ds_df.query("a > 3")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_compound(self):
        pd_result = self.pd_df.query("a > 2 and b < 50")
        ds_result = self.ds_df.query("a > 2 and b < 50")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_or(self):
        pd_result = self.pd_df.query("a < 2 or a > 4")
        ds_result = self.ds_df.query("a < 2 or a > 4")
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByMultipleAggregations(unittest.TestCase):
    """Test groupby with multiple aggregation functions."""

    def setUp(self):
        self.data = {
            "group": ["A", "A", "B", "B", "C"],
            "val1": [10, 20, 30, 40, 50],
            "val2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_groupby_agg_dict(self):
        pd_result = self.pd_df.groupby("group").agg(
            {"val1": "sum", "val2": "mean"}
        ).reset_index()
        ds_result = self.ds_df.groupby("group").agg(
            {"val1": "sum", "val2": "mean"}
        ).reset_index()
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_groupby_multiple_columns(self):
        data = {
            "g1": ["A", "A", "B", "B"],
            "g2": ["x", "y", "x", "y"],
            "val": [1, 2, 3, 4],
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.groupby(["g1", "g2"])["val"].sum().reset_index()
        ds_result = ds_df.groupby(["g1", "g2"])["val"].sum().reset_index()
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_groupby_size(self):
        pd_result = self.pd_df.groupby("group").size().reset_index(name="count")
        ds_result = self.ds_df.groupby("group").size().reset_index(name="count")
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_groupby_count(self):
        pd_result = self.pd_df.groupby("group").count().reset_index()
        ds_result = self.ds_df.groupby("group").count().reset_index()
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_groupby_first_last(self):
        pd_first = self.pd_df.groupby("group").first().reset_index()
        ds_first = self.ds_df.groupby("group").first().reset_index()
        assert_datastore_equals_pandas(
            ds_first, pd_first, check_row_order=False
        )

        pd_last = self.pd_df.groupby("group").last().reset_index()
        ds_last = self.ds_df.groupby("group").last().reset_index()
        assert_datastore_equals_pandas(
            ds_last, pd_last, check_row_order=False
        )


class TestMergeOperations(unittest.TestCase):
    """Test merge/join operations."""

    def setUp(self):
        self.left_data = {"key": [1, 2, 3, 4], "val_left": [10, 20, 30, 40]}
        self.right_data = {"key": [2, 3, 4, 5], "val_right": [200, 300, 400, 500]}
        self.pd_left = pd.DataFrame(self.left_data)
        self.pd_right = pd.DataFrame(self.right_data)
        self.ds_left = DataStore(self.left_data)
        self.ds_right = DataStore(self.right_data)

    def test_merge_inner(self):
        pd_result = self.pd_left.merge(self.pd_right, on="key", how="inner")
        ds_result = self.ds_left.merge(self.ds_right, on="key", how="inner")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_left(self):
        pd_result = self.pd_left.merge(self.pd_right, on="key", how="left")
        ds_result = self.ds_left.merge(self.ds_right, on="key", how="left")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_right(self):
        pd_result = self.pd_left.merge(self.pd_right, on="key", how="right")
        ds_result = self.ds_left.merge(self.ds_right, on="key", how="right")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_outer(self):
        pd_result = self.pd_left.merge(self.pd_right, on="key", how="outer")
        ds_result = self.ds_left.merge(self.ds_right, on="key", how="outer")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestSortValuesEdgeCases(unittest.TestCase):
    """Test sort_values() edge cases."""

    def setUp(self):
        self.data = {
            "a": [3, 1, 2, 1, 3],
            "b": [10, 30, 20, 40, 50],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_sort_single_column(self):
        pd_result = self.pd_df.sort_values("a")
        ds_result = self.ds_df.sort_values("a")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_multiple_columns(self):
        pd_result = self.pd_df.sort_values(["a", "b"])
        ds_result = self.ds_df.sort_values(["a", "b"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending(self):
        pd_result = self.pd_df.sort_values("a", ascending=False)
        ds_result = self.ds_df.sort_values("a", ascending=False)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_mixed_ascending(self):
        pd_result = self.pd_df.sort_values(
            ["a", "b"], ascending=[True, False]
        )
        ds_result = self.ds_df.sort_values(
            ["a", "b"], ascending=[True, False]
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_with_nan(self):
        data = {"a": [3.0, np.nan, 1.0, np.nan, 2.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.sort_values("a")
        ds_result = ds_df.sort_values("a")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_na_position_first(self):
        data = {"a": [3.0, np.nan, 1.0, np.nan, 2.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.sort_values("a", na_position="first")
        ds_result = ds_df.sort_values("a", na_position="first")
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == "__main__":
    unittest.main()
