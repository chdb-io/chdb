"""
Tests for boundary/edge input SQL generation correctness.

Covers:
- isin([]) / notin([]) empty list
- isin([None, 1, 2]) / isin([np.nan, 1, 2]) with NULL values
- notin() NULL-safe semantics (pandas ~isin() compatibility)
- Single-row DataFrame groupby + agg
- Empty DataFrame count/shape/columns
- All-NULL column sum/mean/min/max
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.conditions import InCondition
from datastore.expressions import Field, Literal
from datastore.tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# SQL Generation Unit Tests
# =============================================================================


class TestInConditionSQLGeneration:
    """Unit tests for InCondition.to_sql() edge cases."""

    def test_isin_empty_list_sql(self):
        """isin([]) should generate '1=0' (always false)."""
        cond = InCondition(Field("id"), [], negate=False)
        assert cond.to_sql() == "1=0"

    def test_notin_empty_list_sql(self):
        """notin([]) should generate '1=1' (always true)."""
        cond = InCondition(Field("id"), [], negate=True)
        assert cond.to_sql() == "1=1"

    def test_isin_only_nan_sql(self):
        """isin([nan]) should generate 'col IS NULL'."""
        cond = InCondition(Field("id"), [np.nan], negate=False)
        assert cond.to_sql() == '"id" IS NULL'

    def test_notin_only_nan_sql(self):
        """notin([nan]) should generate 'col IS NOT NULL'."""
        cond = InCondition(Field("id"), [np.nan], negate=True)
        assert cond.to_sql() == '"id" IS NOT NULL'

    def test_isin_nan_with_values_sql(self):
        """isin([nan, 1, 2]) should generate '(col IS NULL OR col IN (1,2))'."""
        cond = InCondition(Field("id"), [np.nan, 1, 2], negate=False)
        sql = cond.to_sql()
        assert "IS NULL" in sql
        assert "IN (1,2)" in sql
        assert sql.startswith("(")

    def test_notin_nan_with_values_sql(self):
        """notin([nan, 1, 2]) should generate '(col IS NOT NULL AND col NOT IN (1,2))'."""
        cond = InCondition(Field("id"), [np.nan, 1, 2], negate=True)
        sql = cond.to_sql()
        assert "IS NOT NULL" in sql
        assert "NOT IN (1,2)" in sql
        assert "AND" in sql

    def test_isin_standard_values_sql(self):
        """isin([1, 2]) should generate standard 'col IN (1,2)'."""
        cond = InCondition(Field("id"), [1, 2], negate=False)
        assert cond.to_sql() == '"id" IN (1,2)'

    def test_notin_standard_values_null_safe_sql(self):
        """notin([1, 2]) should include NULL rows: '(col NOT IN (1,2) OR col IS NULL)'."""
        cond = InCondition(Field("id"), [1, 2], negate=True)
        sql = cond.to_sql()
        assert "NOT IN (1,2)" in sql
        assert "IS NULL" in sql
        assert "OR" in sql

    def test_isin_none_in_values_sql(self):
        """isin([None, 1]) should keep None as SQL NULL (matches pandas behavior)."""
        cond = InCondition(Field("id"), [None, 1], negate=False)
        sql = cond.to_sql()
        # None becomes SQL NULL in the IN list, which is standard behavior
        assert "IN (NULL,1)" in sql

    def test_literal_nan_to_sql(self):
        """Literal(NaN) should generate 'NULL' not 'nan'."""
        assert Literal(np.nan).to_sql() == "NULL"
        assert Literal(float("nan")).to_sql() == "NULL"

    def test_literal_none_to_sql(self):
        """Literal(None) should generate 'NULL'."""
        assert Literal(None).to_sql() == "NULL"

    def test_literal_normal_float_to_sql(self):
        """Literal with normal float values should work correctly."""
        assert Literal(3.14).to_sql() == "3.14"
        assert Literal(0.0).to_sql() == "0.0"
        assert Literal(-1.5).to_sql() == "-1.5"


# =============================================================================
# End-to-End isin/notin Tests (Mirror Pattern)
# =============================================================================


class TestIsinNotinBoundaryExecution:
    """End-to-end tests for isin/notin with boundary inputs."""

    @pytest.fixture
    def data_with_nulls(self):
        """DataFrame with NULL values for testing."""
        data = {"id": [1, 2, 3, None], "val": ["a", "b", "c", "d"]}
        return DataStore(data), pd.DataFrame(data)

    @pytest.fixture
    def data_no_nulls(self):
        """DataFrame without NULL values."""
        data = {"id": [1, 2, 3, 4], "val": ["a", "b", "c", "d"]}
        return DataStore(data), pd.DataFrame(data)

    def test_isin_empty_list(self, data_with_nulls):
        """isin([]) should return no rows."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[pd_df["id"].isin([])]
        ds_result = ds[ds["id"].isin([])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_empty_list(self, data_with_nulls):
        """notin([]) should return all rows including NULL rows."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[~pd_df["id"].isin([])]
        ds_result = ds[ds["id"].notin([])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_empty_list_no_nulls(self, data_no_nulls):
        """isin([]) should return no rows even without NULLs."""
        ds, pd_df = data_no_nulls
        pd_result = pd_df[pd_df["id"].isin([])]
        ds_result = ds[ds["id"].isin([])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_empty_list_no_nulls(self, data_no_nulls):
        """notin([]) should return all rows."""
        ds, pd_df = data_no_nulls
        pd_result = pd_df[~pd_df["id"].isin([])]
        ds_result = ds[ds["id"].notin([])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_nan_matches_null_rows(self, data_with_nulls):
        """isin([np.nan, 1, 2]) should match NULL rows (pandas behavior)."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[pd_df["id"].isin([np.nan, 1, 2])]
        ds_result = ds[ds["id"].isin([np.nan, 1, 2])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_only_nan(self, data_with_nulls):
        """isin([np.nan]) should match only NULL rows."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[pd_df["id"].isin([np.nan])]
        ds_result = ds[ds["id"].isin([np.nan])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_with_nan_excludes_null_rows(self, data_with_nulls):
        """notin([np.nan, 1]) should exclude NULL rows."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[~pd_df["id"].isin([np.nan, 1])]
        ds_result = ds[ds["id"].notin([np.nan, 1])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_without_nan_includes_null_rows(self, data_with_nulls):
        """notin([1]) should include NULL rows (pandas ~isin behavior)."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[~pd_df["id"].isin([1])]
        ds_result = ds[ds["id"].notin([1])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_none_does_not_match_nan(self, data_with_nulls):
        """isin([None, 1]) should NOT match NaN rows (None != NaN in pandas)."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[pd_df["id"].isin([None, 1])]
        ds_result = ds[ds["id"].isin([None, 1])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_with_only_nan(self, data_with_nulls):
        """notin([np.nan]) should return all non-NULL rows."""
        ds, pd_df = data_with_nulls
        pd_result = pd_df[~pd_df["id"].isin([np.nan])]
        ds_result = ds[ds["id"].notin([np.nan])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_string_values(self):
        """isin with string values should work correctly."""
        data = {"cat": ["a", "b", "c", None], "val": [1, 2, 3, 4]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["cat"].isin(["a", "b"])]
        ds_result = ds[ds["cat"].isin(["a", "b"])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_with_string_values_null_safe(self):
        """notin with strings should include NULL rows."""
        data = {"cat": ["a", "b", "c", None], "val": [1, 2, 3, 4]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[~pd_df["cat"].isin(["a"])]
        ds_result = ds[ds["cat"].notin(["a"])]
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Single-Row DataFrame GroupBy + Agg
# =============================================================================


class TestSingleRowGroupByAgg:
    """Test groupby + agg on single-row DataFrame (boundary: only one group)."""

    def test_single_row_groupby_sum(self):
        """Single-row groupby sum should work."""
        data = {"group": ["A"], "val": [42]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.groupby("group").agg({"val": "sum"})
        ds_result = ds.groupby("group").agg({"val": "sum"})
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_single_row_groupby_multiple_aggs(self):
        """Single-row groupby with multiple agg functions."""
        data = {"group": ["A"], "val": [42]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.groupby("group").agg({"val": ["sum", "mean", "count"]})
        ds_result = ds.groupby("group").agg({"val": ["sum", "mean", "count"]})
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_single_row_groupby_mean(self):
        """Single-row groupby mean should return the value itself."""
        data = {"group": ["X"], "val": [3.14]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.groupby("group").mean(numeric_only=True)
        ds_result = ds.groupby("group").mean(numeric_only=True)
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_single_row_groupby_count(self):
        """Single-row groupby count should return 1."""
        data = {"group": ["A"], "val": [10]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.groupby("group").count()
        ds_result = ds.groupby("group").count()
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )


# =============================================================================
# Empty DataFrame Operations
# =============================================================================


class TestEmptyDataFrameOperations:
    """Test operations on empty DataFrames."""

    @pytest.fixture
    def empty_df(self):
        """Empty DataFrame with typed columns."""
        data = {
            "a": pd.Series([], dtype="int64"),
            "b": pd.Series([], dtype="float64"),
            "c": pd.Series([], dtype="object"),
        }
        return DataStore(data), pd.DataFrame(data)

    def test_empty_df_shape(self, empty_df):
        """Empty DataFrame shape should be (0, n_cols)."""
        ds, pd_df = empty_df
        assert ds.shape == pd_df.shape == (0, 3)

    def test_empty_df_len(self, empty_df):
        """Empty DataFrame len should be 0."""
        ds, pd_df = empty_df
        assert len(ds) == len(pd_df) == 0

    def test_empty_df_columns(self, empty_df):
        """Empty DataFrame should preserve column names."""
        ds, pd_df = empty_df
        assert list(ds.columns) == list(pd_df.columns) == ["a", "b", "c"]

    def test_empty_df_count(self, empty_df):
        """Empty DataFrame count should return 0 for all columns."""
        ds, pd_df = empty_df
        pd_count = pd_df.count()
        ds_count = ds.count()
        pd.testing.assert_series_equal(
            ds_count, pd_count, check_dtype=False
        )

    def test_empty_df_groupby_sum(self, empty_df):
        """Empty DataFrame groupby sum should return empty result."""
        ds, pd_df = empty_df
        pd_result = pd_df.groupby("c").sum(numeric_only=True)
        ds_result = ds.groupby("c").sum(numeric_only=True)
        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_empty_df_filter_result(self):
        """Filtering to produce empty result should work."""
        data = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        # Filter that matches nothing
        pd_result = pd_df[pd_df["id"] > 100]
        ds_result = ds[ds["id"] > 100]
        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# All-NULL Column Aggregation
# =============================================================================


class TestAllNullColumnAggregation:
    """Test aggregation on columns with all NULL values."""

    @pytest.fixture
    def all_null_df(self):
        """DataFrame with an all-NULL numeric column."""
        data = {"val": [None, None, None]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data).astype("float64")
        return ds, pd_df

    def test_all_null_sum(self, all_null_df):
        """sum() on all-NULL column should return 0.0 (like pandas)."""
        ds, pd_df = all_null_df
        pd_sum = pd_df["val"].sum()
        ds_sum = float(ds["val"].sum())
        assert ds_sum == pd_sum == 0.0

    def test_all_null_mean(self, all_null_df):
        """mean() on all-NULL column should return NaN (like pandas)."""
        ds, pd_df = all_null_df
        pd_mean = pd_df["val"].mean()
        ds_mean = float(ds["val"].mean())
        assert np.isnan(ds_mean)
        assert np.isnan(pd_mean)

    def test_all_null_min(self, all_null_df):
        """min() on all-NULL column should return NaN (like pandas)."""
        ds, pd_df = all_null_df
        pd_min = pd_df["val"].min()
        ds_min = float(ds["val"].min())
        assert np.isnan(ds_min)
        assert np.isnan(pd_min)

    def test_all_null_max(self, all_null_df):
        """max() on all-NULL column should return NaN (like pandas)."""
        ds, pd_df = all_null_df
        pd_max = pd_df["val"].max()
        ds_max = float(ds["val"].max())
        assert np.isnan(ds_max)
        assert np.isnan(pd_max)

    def test_all_null_count(self, all_null_df):
        """count() on all-NULL column should return 0."""
        ds, pd_df = all_null_df
        pd_count = pd_df["val"].count()
        ds_count = int(ds["val"].count())
        assert ds_count == pd_count == 0

    def test_mixed_null_sum(self):
        """sum() with some NULLs should skip them (like pandas)."""
        data = {"val": [1.0, None, 3.0, None, 5.0]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_sum = pd_df["val"].sum()
        ds_sum = float(ds["val"].sum())
        assert ds_sum == pd_sum == 9.0

    def test_mixed_null_mean(self):
        """mean() with some NULLs should skip them (like pandas)."""
        data = {"val": [1.0, None, 3.0, None, 5.0]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_mean = pd_df["val"].mean()
        ds_mean = float(ds["val"].mean())
        assert abs(ds_mean - pd_mean) < 1e-10
        assert abs(ds_mean - 3.0) < 1e-10


# =============================================================================
# Additional Boundary Tests
# =============================================================================


class TestAdditionalBoundary:
    """Additional boundary tests for SQL generation correctness."""

    def test_isin_single_value(self):
        """isin with single value should work."""
        data = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["id"].isin([2])]
        ds_result = ds[ds["id"].isin([2])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_single_value(self):
        """notin with single value should work."""
        data = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[~pd_df["id"].isin([2])]
        ds_result = ds[ds["id"].notin([2])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_all_match(self):
        """isin where all rows match."""
        data = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["id"].isin([1, 2, 3])]
        ds_result = ds[ds["id"].isin([1, 2, 3])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notin_all_match(self):
        """notin where all values are excluded -> empty result."""
        data = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[~pd_df["id"].isin([1, 2, 3])]
        ds_result = ds[ds["id"].notin([1, 2, 3])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_duplicates_in_values(self):
        """isin with duplicate values in the list should work."""
        data = {"id": [1, 2, 3], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["id"].isin([1, 1, 2, 2])]
        ds_result = ds[ds["id"].isin([1, 1, 2, 2])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_isin_filter(self):
        """Chained isin filters should compose correctly."""
        data = {"id": [1, 2, 3, 4, 5], "cat": ["a", "b", "a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["id"].isin([1, 2, 3]) & pd_df["cat"].isin(["a"])]
        ds_result = ds[ds["id"].isin([1, 2, 3]) & ds["cat"].isin(["a"])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_float_nan_in_values(self):
        """isin with float('nan') should work same as np.nan."""
        data = {"id": [1.0, 2.0, None], "val": ["a", "b", "c"]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["id"].isin([float("nan"), 1.0])]
        ds_result = ds[ds["id"].isin([float("nan"), 1.0])]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_df_isin(self):
        """isin on empty DataFrame should return empty result."""
        data = {"id": pd.Series([], dtype="int64")}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df["id"].isin([1, 2])]
        ds_result = ds[ds["id"].isin([1, 2])]
        assert_datastore_equals_pandas(ds_result, pd_result)
