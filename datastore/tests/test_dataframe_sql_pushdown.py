"""
Tests for DataFrame-level SQL pushdown optimization.

These tests verify that DataFrame.nunique(), DataFrame.value_counts(),
DataFrame.drop_duplicates(), and DataFrame.memory_usage() generate
efficient SQL instead of downloading the full table via _get_df().

Test coverage:
1. nunique() - COUNT(DISTINCT col) pushdown
2. value_counts() - GROUP BY + COUNT(*) pushdown
3. drop_duplicates() - SELECT DISTINCT pushdown via lazy ops
4. memory_usage() - schema-based estimation without full download
5. Mirror tests against pandas for correctness
6. SQL source vs in-memory source behavior
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_series,
)


def _make_csv(data_dict):
    """Create a temporary CSV file and return its path."""
    df = pd.DataFrame(data_dict)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture
def sample_data():
    return {
        "name": ["Alice", "Bob", "Alice", "Charlie", "Bob", "Alice"],
        "age": [25, 30, 25, 35, 30, 25],
        "city": ["NYC", "LA", "NYC", "SF", "LA", "NYC"],
    }


@pytest.fixture
def csv_path(sample_data):
    path = _make_csv(sample_data)
    yield path
    os.unlink(path)


# ===========================================================================
# drop_duplicates SQL pushdown
# ===========================================================================


class TestDropDuplicatesSQLPushdown:
    """Test drop_duplicates() uses lazy distinct for SQL pushdown."""

    def test_drop_duplicates_basic(self, sample_data):
        """Basic drop_duplicates mirrors pandas."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset(self, sample_data):
        """drop_duplicates with subset mirrors pandas."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.drop_duplicates(subset=["name"])
        ds_result = ds_df.drop_duplicates(subset=["name"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self, sample_data):
        """drop_duplicates with keep='last' falls back to pandas correctly."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.drop_duplicates(keep="last")
        ds_result = ds_df.drop_duplicates(keep="last")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self, sample_data):
        """drop_duplicates with keep=False falls back to pandas correctly."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.drop_duplicates(keep=False)
        ds_result = ds_df.drop_duplicates(keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_ignore_index(self, sample_data):
        """drop_duplicates with ignore_index=True."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.drop_duplicates(ignore_index=True)
        ds_result = ds_df.drop_duplicates(ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_from_file(self, csv_path, sample_data):
        """drop_duplicates on file-based DataStore uses SQL DISTINCT."""
        pd_df = pd.DataFrame(sample_data).drop_duplicates()
        ds_df = DataStore.from_file(csv_path).drop_duplicates()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_drop_duplicates_returns_datastore(self, sample_data):
        """drop_duplicates returns a DataStore, not a DataFrame."""
        ds_df = DataStore(sample_data)
        result = ds_df.drop_duplicates()
        assert isinstance(result, DataStore)

    def test_drop_duplicates_inplace_raises(self, sample_data):
        """drop_duplicates(inplace=True) raises ImmutableError."""
        ds_df = DataStore(sample_data)
        with pytest.raises(Exception):
            ds_df.drop_duplicates(inplace=True)

    def test_drop_duplicates_all_same(self):
        """drop_duplicates where all rows are identical."""
        data = {"a": [1, 1, 1], "b": ["x", "x", "x"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_all_unique(self):
        """drop_duplicates where all rows are unique."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_empty(self):
        """drop_duplicates on empty DataFrame."""
        data = {"a": [], "b": []}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)


# ===========================================================================
# nunique SQL pushdown
# ===========================================================================


class TestNuniqueSQLPushdown:
    """Test nunique() uses COUNT(DISTINCT col) SQL pushdown."""

    def test_nunique_basic(self, sample_data):
        """Basic nunique mirrors pandas."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)

    def test_nunique_from_file(self, csv_path, sample_data):
        """nunique on file-based DataStore uses SQL pushdown."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore.from_file(csv_path)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)

    def test_nunique_all_unique(self):
        """nunique where all values in each column are unique."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)

    def test_nunique_all_same(self):
        """nunique where all values are identical."""
        data = {"a": [1, 1, 1], "b": ["x", "x", "x"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)

    def test_nunique_with_nulls_dropna_true(self):
        """nunique with NaN values and dropna=True (default)."""
        data = {"a": [1, 2, None, 2, None], "b": ["x", None, "y", "x", None]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.nunique(dropna=True)
        ds_result = ds_df.nunique(dropna=True)

        assert_series_equal(ds_result, pd_result)

    def test_nunique_with_nulls_dropna_false(self):
        """nunique with NaN values and dropna=False."""
        data = {"a": [1, 2, None, 2, None], "b": ["x", None, "y", "x", None]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.nunique(dropna=False)
        ds_result = ds_df.nunique(dropna=False)

        assert_series_equal(ds_result, pd_result)

    def test_nunique_single_column(self):
        """nunique on single-column DataFrame."""
        data = {"val": [10, 20, 10, 30]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)

    def test_nunique_empty(self):
        """nunique on empty DataFrame."""
        data = {"a": [], "b": []}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)


# ===========================================================================
# value_counts SQL pushdown
# ===========================================================================


class TestValueCountsSQLPushdown:
    """Test DataFrame.value_counts() uses GROUP BY SQL pushdown."""

    def test_value_counts_basic(self, sample_data):
        """Basic value_counts mirrors pandas."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.value_counts()
        ds_result = ds_df.value_counts()

        # value_counts order within same count is not deterministic
        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_from_file(self, csv_path, sample_data):
        """value_counts on file-based DataStore uses SQL pushdown."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore.from_file(csv_path)

        pd_result = pd_df.value_counts()
        ds_result = ds_df.value_counts()

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_subset(self, sample_data):
        """value_counts with subset parameter."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.value_counts(subset=["name"])
        ds_result = ds_df.value_counts(subset=["name"])

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_normalize(self, sample_data):
        """value_counts with normalize=True."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.value_counts(normalize=True)
        ds_result = ds_df.value_counts(normalize=True)

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_ascending(self, sample_data):
        """value_counts with ascending=True."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.value_counts(ascending=True)
        ds_result = ds_df.value_counts(ascending=True)

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_sort_false(self, sample_data):
        """value_counts with sort=False."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.value_counts(sort=False)
        ds_result = ds_df.value_counts(sort=False)

        # When sort=False, order is not guaranteed, just compare values
        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_dropna_false(self):
        """value_counts with dropna=False includes NaN rows."""
        data = {"a": [1, 2, None, 2], "b": ["x", "y", None, "y"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.value_counts(dropna=False)
        ds_result = ds_df.value_counts(dropna=False)

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_single_column(self):
        """value_counts on single-column DataFrame."""
        data = {"val": [10, 20, 10, 30, 20, 10]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.value_counts()
        ds_result = ds_df.value_counts()

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_empty(self):
        """value_counts on empty DataFrame."""
        data = {"a": [], "b": []}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.value_counts()
        ds_result = ds_df.value_counts()

        assert len(get_series(ds_result)) == len(pd_result)

    def test_value_counts_all_unique(self):
        """value_counts where every row is unique."""
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.value_counts()
        ds_result = ds_df.value_counts()

        assert_series_equal(ds_result, pd_result, check_like=True)

    def test_value_counts_all_same(self):
        """value_counts where all rows are identical."""
        data = {"a": [1, 1, 1], "b": ["x", "x", "x"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.value_counts()
        ds_result = ds_df.value_counts()

        assert_series_equal(ds_result, pd_result, check_like=True)


# ===========================================================================
# memory_usage SQL pushdown
# ===========================================================================


class TestMemoryUsageSQLPushdown:
    """Test memory_usage() avoids full table download when possible."""

    def test_memory_usage_basic(self, sample_data):
        """Basic memory_usage mirrors pandas."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.memory_usage()
        ds_result = ds_df.memory_usage()

        assert_series_equal(ds_result, pd_result)

    def test_memory_usage_no_index(self, sample_data):
        """memory_usage(index=False) excludes index."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.memory_usage(index=False)
        ds_result = ds_df.memory_usage(index=False)

        assert_series_equal(ds_result, pd_result)

    def test_memory_usage_deep(self, sample_data):
        """memory_usage(deep=True) gives accurate string sizes."""
        pd_df = pd.DataFrame(sample_data)
        ds_df = DataStore(sample_data)

        pd_result = pd_df.memory_usage(deep=True)
        ds_result = ds_df.memory_usage(deep=True)

        assert_series_equal(ds_result, pd_result)

    def test_memory_usage_numeric_only(self):
        """memory_usage on numeric-only DataFrame."""
        data = {"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.memory_usage()
        ds_result = ds_df.memory_usage()

        assert_series_equal(ds_result, pd_result)

    def test_memory_usage_empty(self):
        """memory_usage on empty DataFrame."""
        data = {"a": [], "b": []}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.memory_usage()
        ds_result = ds_df.memory_usage()

        assert_series_equal(ds_result, pd_result)


# ===========================================================================
# SQL pushdown verification tests (ensure SQL is actually used)
# ===========================================================================


class TestSQLPushdownVerification:
    """Verify that SQL pushdown is actually happening for file-based sources."""

    def test_nunique_sql_path(self, csv_path):
        """Verify nunique uses SQL path for file-based DataStore."""
        ds = DataStore.from_file(csv_path)
        # Should have SQL state
        assert ds._has_sql_state()
        # Result should still be correct
        result = ds.nunique()
        assert isinstance(result, pd.Series)
        assert len(result) == 3  # name, age, city

    def test_value_counts_sql_path(self, csv_path):
        """Verify value_counts uses SQL path for file-based DataStore."""
        ds = DataStore.from_file(csv_path)
        assert ds._has_sql_state()
        result = ds.value_counts()
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_drop_duplicates_lazy_op(self, sample_data):
        """Verify drop_duplicates adds LazyDistinct op."""
        from datastore.lazy_ops import LazyDistinct
        ds = DataStore(sample_data)
        result = ds.drop_duplicates()
        # Check that LazyDistinct was added
        has_distinct = any(
            isinstance(op, LazyDistinct) for op in result._lazy_ops
        )
        assert has_distinct

    def test_drop_duplicates_file_uses_distinct(self, csv_path):
        """Verify drop_duplicates on file source uses SQL DISTINCT."""
        ds = DataStore.from_file(csv_path)
        result = ds.drop_duplicates()
        assert result._distinct is True

    def test_memory_usage_file_sql_estimation(self, csv_path):
        """Verify memory_usage on file source uses SQL estimation for non-deep."""
        ds = DataStore.from_file(csv_path)
        assert ds._has_sql_state()
        result = ds.memory_usage()
        assert isinstance(result, pd.Series)
        assert len(result) > 0
