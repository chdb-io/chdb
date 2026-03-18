"""
Tests for drop_duplicates / value_counts / nunique SQL pushdown.

Verifies:
1. drop_duplicates(subset=...) generates LIMIT 1 BY in SQL (not just SELECT DISTINCT)
2. drop_duplicates() without subset generates SELECT DISTINCT
3. _can_sql_pushdown() helper correctly determines SQL eligibility
4. SQL log capture verifies actual SQL generation
5. Mirror pattern: DataStore results match pandas
"""

import logging
import pytest
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_frame_equal,
    assert_series_equal,
    get_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SQLLogCapture:
    """Capture DataStore SQL log messages for verification."""

    def __init__(self):
        self.messages = []
        self._handler = None
        self._logger = None

    def __enter__(self):
        self._logger = logging.getLogger("datastore")
        self._handler = logging.Handler()
        self._handler.emit = lambda record: self.messages.append(record.getMessage())
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *args):
        if self._logger and self._handler:
            self._logger.removeHandler(self._handler)

    def find(self, substring):
        """Return all messages containing *substring*."""
        return [m for m in self.messages if substring in m]

    def has(self, substring):
        """Check if any message contains *substring*."""
        return any(substring in m for m in self.messages)


def _make_parquet(df, tmp_path, name="data.parquet"):
    """Write a DataFrame to parquet and return the path string."""
    path = tmp_path / name
    df.to_parquet(str(path), index=False)
    return str(path)


# ===========================================================================
# drop_duplicates SQL generation tests
# ===========================================================================


class TestDropDuplicatesSQLGeneration:
    """Verify that drop_duplicates generates correct SQL."""

    def test_subset_generates_limit_by(self, tmp_path):
        """drop_duplicates(subset=['a']) should produce LIMIT 1 BY."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "y", "y", "z", "z"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        ds_dedup = ds.drop_duplicates(subset=["a"])
        sql = ds_dedup.to_sql()
        assert "LIMIT 1 BY" in sql
        assert '"a"' in sql
        assert "DISTINCT" not in sql

    def test_no_subset_generates_distinct(self, tmp_path):
        """drop_duplicates() without subset should produce SELECT DISTINCT."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        ds_dedup = ds.drop_duplicates()
        sql = ds_dedup.to_sql()
        assert "DISTINCT" in sql
        assert "LIMIT 1 BY" not in sql

    def test_multi_column_subset_limit_by(self, tmp_path):
        """drop_duplicates(subset=['a', 'b']) should list both in LIMIT 1 BY."""
        df = pd.DataFrame(
            {"a": [1, 1, 1], "b": ["x", "x", "y"], "c": [10, 20, 30]}
        )
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        ds_dedup = ds.drop_duplicates(subset=["a", "b"])
        sql = ds_dedup.to_sql()
        assert "LIMIT 1 BY" in sql
        assert '"a"' in sql
        assert '"b"' in sql

    def test_distinct_method_with_subset(self, tmp_path):
        """DataStore.distinct(subset=['a']) should also use LIMIT 1 BY."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        ds_dedup = ds.distinct(subset=["a"])
        sql = ds_dedup.to_sql()
        assert "LIMIT 1 BY" in sql
        assert "DISTINCT" not in sql


# ===========================================================================
# drop_duplicates correctness (Mirror Pattern)
# ===========================================================================


class TestDropDuplicatesCorrectness:
    """Mirror tests: DataStore results must match pandas."""

    def test_subset_single_column(self, tmp_path):
        """drop_duplicates(subset=['a']) keeps first per group."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "y", "y", "z", "z"]})
        path = _make_parquet(df, tmp_path)

        pd_result = df.drop_duplicates(subset=["a"])
        ds_result = DataStore.from_file(path).drop_duplicates(subset=["a"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subset_multi_column(self, tmp_path):
        """drop_duplicates(subset=['a', 'b']) with two-column key."""
        df = pd.DataFrame(
            {"a": [1, 1, 1, 2], "b": ["x", "x", "y", "x"], "c": [10, 20, 30, 40]}
        )
        path = _make_parquet(df, tmp_path)

        pd_result = df.drop_duplicates(subset=["a", "b"])
        ds_result = DataStore.from_file(path).drop_duplicates(subset=["a", "b"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_no_subset(self, tmp_path):
        """drop_duplicates() without subset removes full-row duplicates."""
        df = pd.DataFrame(
            {"a": [1, 1, 2, 2], "b": ["x", "x", "y", "z"]}
        )
        path = _make_parquet(df, tmp_path)

        pd_result = df.drop_duplicates()
        ds_result = DataStore.from_file(path).drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_subset_with_nulls(self, tmp_path):
        """drop_duplicates(subset=['a']) with NULL values."""
        df = pd.DataFrame(
            {"a": [1.0, np.nan, 1.0, np.nan, 2.0], "b": ["x", "y", "z", "w", "v"]}
        )
        path = _make_parquet(df, tmp_path)

        pd_result = df.drop_duplicates(subset=["a"])
        ds_result = DataStore.from_file(path).drop_duplicates(subset=["a"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_from_dataframe(self):
        """drop_duplicates on DataStore created from DataFrame."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "y", "y", "z", "z"]})

        pd_result = df.drop_duplicates(subset=["a"])
        ds_result = DataStore(df).drop_duplicates(subset=["a"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_unique(self, tmp_path):
        """drop_duplicates on data with no duplicates."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = _make_parquet(df, tmp_path)

        pd_result = df.drop_duplicates(subset=["a"])
        ds_result = DataStore.from_file(path).drop_duplicates(subset=["a"])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_all_same(self, tmp_path):
        """drop_duplicates on data where all rows have same subset values."""
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "y", "z"]})
        path = _make_parquet(df, tmp_path)

        pd_result = df.drop_duplicates(subset=["a"])
        ds_result = DataStore.from_file(path).drop_duplicates(subset=["a"])

        assert len(get_dataframe(ds_result)) == len(pd_result)
        assert len(get_dataframe(ds_result)) == 1

    def test_keep_last_falls_back(self):
        """keep='last' should fall back to pandas execution."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})

        pd_result = df.drop_duplicates(subset=["a"], keep="last")
        ds_result = DataStore(df).drop_duplicates(subset=["a"], keep="last")

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_keep_false_falls_back(self):
        """keep=False should fall back to pandas execution."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})

        pd_result = df.drop_duplicates(subset=["a"], keep=False)
        ds_result = DataStore(df).drop_duplicates(subset=["a"], keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


# ===========================================================================
# SQL log capture tests
# ===========================================================================


class TestSQLLogCapture:
    """Verify actual SQL execution via log capture."""

    def test_drop_duplicates_subset_sql_in_logs(self, tmp_path):
        """Executing drop_duplicates(subset=...) should log SQL with LIMIT 1 BY."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        with SQLLogCapture() as cap:
            result = ds.drop_duplicates(subset=["a"]).values.tolist()

        sql_msgs = cap.find("Executing SQL")
        assert any("LIMIT 1 BY" in m for m in sql_msgs), (
            f"Expected LIMIT 1 BY in SQL logs, got: {sql_msgs}"
        )

    def test_drop_duplicates_no_subset_sql_in_logs(self, tmp_path):
        """Executing drop_duplicates() should log SQL with DISTINCT."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        with SQLLogCapture() as cap:
            result = ds.drop_duplicates().values.tolist()

        sql_msgs = cap.find("Executing SQL")
        assert any("DISTINCT" in m for m in sql_msgs), (
            f"Expected DISTINCT in SQL logs, got: {sql_msgs}"
        )

    def test_nunique_sql_in_logs(self, tmp_path):
        """nunique() should log SQL with COUNT(DISTINCT ...)."""
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        with SQLLogCapture() as cap:
            result = ds.nunique()

        assert cap.has("COUNT(DISTINCT"), (
            f"Expected COUNT(DISTINCT in logs, got: {cap.messages}"
        )

    def test_value_counts_sql_in_logs(self, tmp_path):
        """value_counts() should log SQL with GROUP BY."""
        df = pd.DataFrame({"a": ["x", "x", "y", "z"]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)

        with SQLLogCapture() as cap:
            result = ds.value_counts()

        assert cap.has("GROUP BY"), (
            f"Expected GROUP BY in logs, got: {cap.messages}"
        )


# ===========================================================================
# _can_sql_pushdown() tests
# ===========================================================================


class TestCanSqlPushdown:
    """Test the _can_sql_pushdown() helper method."""

    def test_file_source_can_pushdown(self, tmp_path):
        """File-based DataStore should support SQL pushdown."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)
        assert ds._can_sql_pushdown() is True

    def test_dataframe_source_cannot_pushdown(self):
        """DataFrame-sourced DataStore without SQL state cannot pushdown."""
        ds = DataStore(pd.DataFrame({"a": [1, 2, 3]}))
        assert ds._can_sql_pushdown() is False

    def test_with_relational_ops_can_pushdown(self, tmp_path):
        """DataStore with only relational lazy ops can pushdown."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)
        ds_filtered = ds[ds["a"] > 2]
        assert ds_filtered._can_sql_pushdown() is True

    def test_with_non_sql_ops_cannot_pushdown(self, tmp_path):
        """DataStore with non-SQL lazy ops cannot pushdown."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = _make_parquet(df, tmp_path)
        ds = DataStore.from_file(path)
        ds_applied = ds.fillna(0)  # fillna adds a non-SQL lazy op
        assert ds_applied._can_sql_pushdown() is False
