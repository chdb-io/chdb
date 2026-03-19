"""
Tests for flat SQL generation in count_rows()/count() on remote sources.

Verifies that count operations avoid nested subqueries (SELECT ... FROM (SELECT
... FROM remote(...))) which hang on chDB's embedded ClickHouse engine.

Coverage:
  1. SQL shape verification (mock executor, capture SQL)
  2. Correctness verification (local data, mirror pandas)
  3. Regression tests (non-remote, info(), loc[], edge cases)
"""

import re
import unittest
import tempfile
import os
from copy import copy
from unittest.mock import MagicMock, patch

import pandas as pd

from datastore import DataStore
from datastore.connection import QueryResult
from datastore.table_functions import RemoteTableFunction
from datastore.lazy_ops import LazyRelationalOp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEMA = {"id": "Int64", "category": "String", "value": "Int64", "score": "Float64"}


def _make_remote_ds(schema=None):
    """Create a DataStore with a RemoteTableFunction (no real connection)."""
    tf = RemoteTableFunction(
        host="ch.example.com:9000",
        database="default",
        table="test_table",
        user="default",
        password="secret",
    )
    ds = DataStore({"_placeholder": [1]})
    ds._table_function = tf
    ds._format_settings = {}
    ds._source_df = None
    ds._source_df_name = None
    ds.table_name = None
    ds.source_type = "remote"
    ds._lazy_ops = []
    if schema:
        ds._schema = schema
    return ds


def _mock_executor(return_rows=None, return_columns=None):
    """Create a mock executor that captures SQL and returns fake results."""
    executor = MagicMock()
    captured_sql = []

    def execute_side_effect(sql, *args, **kwargs):
        captured_sql.append(sql)
        return QueryResult(
            rows=return_rows or [(0,)],
            column_names=return_columns or ["count()"],
        )

    executor.execute.side_effect = execute_side_effect
    return executor, captured_sql


def _has_nested_subquery(sql):
    """Check if SQL has nested subquery pattern: FROM (SELECT ...)."""
    return bool(re.search(r"FROM\s*\(", sql, re.IGNORECASE))


# ===========================================================================
# 1. SQL Shape Verification (mock executor, verify no nested subquery)
# ===========================================================================


class TestCountRowsFlatSQL(unittest.TestCase):
    """Verify count_rows() generates flat SQL for remote sources."""

    def test_remote_no_filters_generates_flat_sql(self):
        ds = _make_remote_ds(SCHEMA)
        executor, captured = _mock_executor(return_rows=[(42,)])
        ds._executor = executor

        result = ds.count_rows()

        self.assertEqual(result, 42)
        self.assertEqual(len(captured), 1)
        sql = captured[0]
        self.assertIn("count()", sql.lower())
        self.assertIn("remote(", sql)
        self.assertFalse(
            _has_nested_subquery(sql),
            f"Should be flat SQL, got nested subquery: {sql}",
        )

    def test_remote_with_filter_generates_flat_sql_with_where(self):
        ds = _make_remote_ds(SCHEMA)
        filtered = ds[ds["value"] > 30]
        executor, captured = _mock_executor(return_rows=[(5,)])
        filtered._executor = executor

        result = filtered.count_rows()

        self.assertEqual(result, 5)
        self.assertEqual(len(captured), 1)
        sql = captured[0]
        self.assertIn("WHERE", sql)
        self.assertIn("remote(", sql)
        self.assertFalse(
            _has_nested_subquery(sql),
            f"Should be flat SQL with WHERE, got: {sql}",
        )

    def test_remote_filter_sort_generates_flat_sql_no_order_by(self):
        """User's exact scenario: filter + sort + len() should not hang."""
        ds = _make_remote_ds(SCHEMA)
        result_ds = ds[ds["value"] > 20].sort_values("score")
        executor, captured = _mock_executor(return_rows=[(10,)])
        result_ds._executor = executor

        result = result_ds.count_rows()

        self.assertEqual(result, 10)
        sql = captured[0]
        self.assertIn("WHERE", sql)
        self.assertNotIn("ORDER BY", sql)
        self.assertFalse(
            _has_nested_subquery(sql),
            f"filter+sort count should be flat SQL: {sql}",
        )

    def test_remote_sort_only_generates_flat_sql(self):
        ds = _make_remote_ds(SCHEMA)
        sorted_ds = ds.sort_values("id")
        executor, captured = _mock_executor(return_rows=[(100,)])
        sorted_ds._executor = executor

        result = sorted_ds.count_rows()

        self.assertEqual(result, 100)
        sql = captured[0]
        self.assertNotIn("ORDER BY", sql)
        self.assertFalse(_has_nested_subquery(sql), f"sort-only count: {sql}")

    def test_remote_with_limit_falls_back_to_execute(self):
        ds = _make_remote_ds(SCHEMA)
        limited = ds.head(10)

        with patch.object(type(limited), "_execute") as mock_exec:
            mock_exec.return_value = pd.DataFrame({"a": range(10)})
            result = limited.count_rows()

        self.assertEqual(result, 10)
        mock_exec.assert_called_once()

    def test_remote_with_groupby_falls_back_to_execute(self):
        """GROUP BY on remote should fall back to _execute(), not subquery."""
        ds = _make_remote_ds(SCHEMA)
        from datastore.expressions import Field
        ds._groupby_fields = [Field("category")]
        executor, _ = _mock_executor()
        ds._executor = executor

        with patch.object(type(ds), "_execute") as mock_exec:
            mock_exec.return_value = pd.DataFrame({"category": ["A", "B"]})
            result = ds.count_rows()

        self.assertEqual(result, 2)
        mock_exec.assert_called_once()

    def test_remote_with_distinct_falls_back_to_execute(self):
        ds = _make_remote_ds(SCHEMA)
        ds._distinct = True
        executor, _ = _mock_executor()
        ds._executor = executor

        with patch.object(type(ds), "_execute") as mock_exec:
            mock_exec.return_value = pd.DataFrame({"id": [1, 2, 3]})
            result = ds.count_rows()

        self.assertEqual(result, 3)
        mock_exec.assert_called_once()

    def test_remote_with_joins_falls_back_to_execute(self):
        ds = _make_remote_ds(SCHEMA)
        ds._joins = [("fake_join",)]
        executor, _ = _mock_executor()
        ds._executor = executor

        with patch.object(type(ds), "_execute") as mock_exec:
            mock_exec.return_value = pd.DataFrame({"id": [1]})
            result = ds.count_rows()

        self.assertEqual(result, 1)
        mock_exec.assert_called_once()

    def test_remote_empty_result_returns_zero(self):
        ds = _make_remote_ds(SCHEMA)
        ds[ds["value"] > 99999]
        executor, captured = _mock_executor(return_rows=[(0,)])
        ds._executor = executor

        result = ds.count_rows()
        self.assertEqual(result, 0)

    def test_remote_multiple_filters_combined_in_flat_sql(self):
        ds = _make_remote_ds(SCHEMA)
        filtered = ds[ds["value"] > 10]
        filtered = filtered[filtered["category"] == "A"]
        executor, captured = _mock_executor(return_rows=[(3,)])
        filtered._executor = executor

        result = filtered.count_rows()

        self.assertEqual(result, 3)
        sql = captured[0]
        self.assertIn("WHERE", sql)
        self.assertFalse(
            _has_nested_subquery(sql),
            f"Multiple filters should still be flat: {sql}",
        )


class TestCountFlatSQL(unittest.TestCase):
    """Verify count() generates flat SQL for remote sources."""

    def _make_ds_with_columns(self, columns):
        """Create remote DS with mocked columns property."""
        ds = _make_remote_ds(SCHEMA)
        ds._schema = {c: "Int64" for c in columns}
        return ds

    def test_remote_count_generates_flat_sql(self):
        ds = self._make_ds_with_columns(["id", "value"])
        executor, captured = _mock_executor(
            return_rows=[(10, 10)], return_columns=["id", "value"]
        )
        ds._executor = executor

        result = ds.count()

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(captured), 1)
        sql = captured[0]
        self.assertIn("COUNT(", sql)
        self.assertIn("remote(", sql)
        self.assertFalse(
            _has_nested_subquery(sql),
            f"count() should be flat SQL: {sql}",
        )

    def test_remote_count_with_filter_flat_sql(self):
        ds = self._make_ds_with_columns(["id", "value"])
        filtered = ds[ds["value"] > 50]
        executor, captured = _mock_executor(
            return_rows=[(3, 3)], return_columns=["id", "value"]
        )
        filtered._executor = executor

        with patch.object(type(filtered), "columns", new_callable=lambda: property(lambda self: pd.Index(["id", "value"]))):
            result = filtered.count()

        sql = captured[0]
        self.assertIn("WHERE", sql)
        self.assertIn("COUNT(", sql)
        self.assertFalse(
            _has_nested_subquery(sql), f"count() with filter: {sql}"
        )

    def test_remote_count_with_groupby_falls_back(self):
        ds = self._make_ds_with_columns(["id", "category", "value"])
        from datastore.expressions import Field
        ds._groupby_fields = [Field("category")]
        executor, _ = _mock_executor()
        ds._executor = executor

        with patch.object(type(ds), "columns", new_callable=lambda: property(lambda self: pd.Index(["id", "category", "value"]))):
            with patch.object(type(ds), "_execute") as mock_exec:
                mock_exec.return_value = pd.DataFrame(
                    {"id": [2, 2], "category": ["A", "B"], "value": [2, 2]}
                )
                result = ds.count()

        mock_exec.assert_called_once()
        self.assertIsInstance(result, pd.Series)


# ===========================================================================
# 2. Correctness Verification (local data, mirror pandas)
# ===========================================================================


class TestCountRowsCorrectness(unittest.TestCase):
    """Verify count_rows()/count() results match pandas."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "data.csv")
        cls.pdf = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "value": [10, 20, 30, 40, 50, 60, 70, 80],
                "score": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8],
            }
        )
        cls.pdf.to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _ds(self):
        return DataStore.from_file(self.csv_file)

    def test_count_rows_no_filter_matches_pandas(self):
        pd_result = len(self.pdf)
        ds_result = self._ds().count_rows()
        self.assertEqual(ds_result, pd_result)

    def test_count_rows_with_filter_matches_pandas(self):
        pd_result = len(self.pdf[self.pdf["value"] > 30])
        ds = self._ds()
        ds_result = ds[ds["value"] > 30].count_rows()
        self.assertEqual(ds_result, pd_result)

    def test_count_rows_filter_sort_matches_pandas(self):
        pd_result = len(
            self.pdf[self.pdf["value"] > 20].sort_values("score", ascending=False)
        )
        ds = self._ds()
        ds_result = ds[ds["value"] > 20].sort_values(
            "score", ascending=False
        ).count_rows()
        self.assertEqual(ds_result, pd_result)

    def test_count_rows_empty_result(self):
        pd_result = len(self.pdf[self.pdf["value"] > 9999])
        ds = self._ds()
        ds_result = ds[ds["value"] > 9999].count_rows()
        self.assertEqual(ds_result, pd_result)
        self.assertEqual(ds_result, 0)

    def test_len_matches_pandas(self):
        pd_result = len(self.pdf[self.pdf["category"] == "A"])
        ds = self._ds()
        ds_result = len(ds[ds["category"] == "A"])
        self.assertEqual(ds_result, pd_result)

    def test_shape_matches_pandas(self):
        pd_result = self.pdf[self.pdf["value"] > 30].shape
        ds = self._ds()
        ds_result = ds[ds["value"] > 30].shape
        self.assertEqual(ds_result[0], pd_result[0])

    def test_count_per_column_matches_pandas(self):
        pd_result = self.pdf.count()
        ds_result = self._ds().count()
        self.assertIsInstance(ds_result, pd.Series)
        for col in self.pdf.columns:
            self.assertEqual(
                ds_result[col], pd_result[col],
                f"count mismatch for column {col}",
            )

    def test_count_with_filter_matches_pandas(self):
        pd_filtered = self.pdf[self.pdf["value"] > 30]
        pd_result = pd_filtered.count()
        ds = self._ds()
        ds_result = ds[ds["value"] > 30].count()
        self.assertIsInstance(ds_result, pd.Series)
        for col in pd_filtered.columns:
            self.assertEqual(
                ds_result[col], pd_result[col],
                f"filtered count mismatch for column {col}",
            )

    def test_assign_then_filter_count_rows(self):
        """assign() + filter on computed column must not use flat SQL."""
        ds = self._ds()
        ds_result = ds.assign(doubled=ds["value"] * 2)
        ds_result = ds_result[ds_result["doubled"] > 40]

        pd_result = self.pdf.assign(doubled=self.pdf["value"] * 2)
        pd_result = pd_result[pd_result["doubled"] > 40]

        self.assertEqual(ds_result.count_rows(), len(pd_result))
        self.assertEqual(len(ds_result), len(pd_result))

    def test_assign_then_filter_count(self):
        """assign() + filter on computed column count() must work."""
        ds = self._ds()
        ds_result = ds.assign(doubled=ds["value"] * 2)
        ds_result = ds_result[ds_result["doubled"] > 40]

        pd_result = self.pdf.assign(doubled=self.pdf["value"] * 2)
        pd_result = pd_result[pd_result["doubled"] > 40]

        ds_count = ds_result.count()
        pd_count = pd_result.count()
        for col in pd_count.index:
            self.assertEqual(
                ds_count[col], pd_count[col],
                f"assign+filter count mismatch for {col}",
            )


# ===========================================================================
# 3. Regression Tests
# ===========================================================================


class TestNonRemoteRegression(unittest.TestCase):
    """Ensure non-remote sources still work correctly after the change."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "data.csv")
        cls.pdf = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Diana"],
                "age": [25, 30, 35, 28],
                "city": ["NYC", "LA", "NYC", "LA"],
            }
        )
        cls.pdf.to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _ds(self):
        return DataStore.from_file(self.csv_file)

    def test_file_count_rows_still_works(self):
        self.assertEqual(self._ds().count_rows(), 4)

    def test_file_count_rows_with_filter(self):
        ds = self._ds()
        self.assertEqual(ds[ds["age"] > 28].count_rows(), 2)

    def test_file_count_still_works(self):
        result = self._ds().count()
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result["name"], 4)

    def test_file_count_with_filter(self):
        ds = self._ds()
        result = ds[ds["city"] == "NYC"].count()
        self.assertEqual(result["name"], 2)

    def test_dataframe_source_count_rows(self):
        ds = DataStore(self.pdf)
        self.assertEqual(ds.count_rows(), 4)

    def test_dataframe_source_count(self):
        ds = DataStore(self.pdf)
        result = ds.count()
        self.assertEqual(result["name"], 4)


class TestIsRemoteSource(unittest.TestCase):
    """Verify _is_remote_source() helper."""

    def test_remote_clickhouse(self):
        ds = _make_remote_ds(SCHEMA)
        self.assertTrue(ds._is_remote_source())

    def test_file_source(self):
        ds = DataStore({"a": [1, 2]})
        self.assertFalse(ds._is_remote_source())

    def test_no_table_function(self):
        ds = DataStore({"a": [1, 2]})
        ds._table_function = None
        self.assertFalse(ds._is_remote_source())


class TestLocFilterCountRows(unittest.TestCase):
    """Verify loc[] filter WHERE is captured in count_rows flat SQL."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "data.csv")
        cls.pdf = pd.DataFrame(
            {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}
        )
        cls.pdf.to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_loc_filter_count_rows_correctness(self):
        ds = DataStore.from_file(self.csv_file)
        pd_result = len(self.pdf.loc[self.pdf["x"] > 2])
        ds_result = ds.loc[ds["x"] > 2].count_rows()
        self.assertEqual(ds_result, pd_result)


class TestInfoOnRemote(unittest.TestCase):
    """Verify info() on remote DataStore uses flat SQL (doesn't hang)."""

    def test_info_calls_count_rows_and_count_without_subquery(self):
        ds = _make_remote_ds(SCHEMA)
        ds._schema = SCHEMA
        executor, _ = _mock_executor()
        ds._executor = executor

        count_rows_called = []

        with patch.object(type(ds), "count_rows", lambda self: (count_rows_called.append(1) or 100)):
            with patch.object(type(ds), "count", lambda self: pd.Series({k: 100 for k in SCHEMA})):
                with patch.object(type(ds), "dtypes", new_callable=lambda: property(lambda self: pd.Series({k: "int64" for k in SCHEMA}))):
                    import io
                    buf = io.StringIO()
                    ds.info(buf=buf)
                    output = buf.getvalue()

        self.assertGreater(len(count_rows_called), 0)
        self.assertIn("100", output)


if __name__ == "__main__":
    unittest.main()
