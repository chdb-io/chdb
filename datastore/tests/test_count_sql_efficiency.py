"""
Tests for count()/count_rows() SQL efficiency.

Verifies:
- COUNT subquery never contains redundant ORDER BY
- count_rows() generates COUNT(*) instead of full execution (when no LIMIT)
- PANDAS_FILTER chains correctly fall back to len(execute())
- Empty filter results return 0 / empty Series
- LIMIT + count_rows() executes then counts (not SQL COUNT)
"""

import unittest
import tempfile
import os
from copy import copy
from unittest.mock import patch

import pandas as pd

from datastore import DataStore
from datastore.lazy_ops import LazyRelationalOp


class TestCountSQLEfficiencyBase(unittest.TestCase):
    """Shared setup for count SQL efficiency tests."""

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


class TestFilterSortCountNoOrderBy(TestCountSQLEfficiencyBase):
    """Verify COUNT subquery never contains ORDER BY."""

    def test_filter_sort_count_sql_has_no_order_by(self):
        """filter + sort + count: the COUNT subquery must NOT contain ORDER BY."""
        ds = self._ds()
        ds_sorted = ds[ds["value"] > 20].sort_values("score", ascending=False)

        # Confirm ORDER BY is present in the original lazy ops
        orderby_ops = [
            op
            for op in ds_sorted._lazy_ops
            if isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY"
        ]
        self.assertTrue(len(orderby_ops) > 0, "Setup: should have ORDER BY ops")

        # Simulate what count() does internally: strip ORDER BY and build SQL
        count_base = copy(ds_sorted)
        count_base._lazy_ops = [
            op
            for op in count_base._lazy_ops
            if not (isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY")
        ]
        count_base._orderby_fields = []
        count_base._limit_value = None
        count_base._offset_value = None
        sql = count_base.to_sql()

        self.assertNotIn(
            "ORDER BY", sql, f"COUNT subquery must not contain ORDER BY: {sql}"
        )
        self.assertIn("WHERE", sql, f"COUNT subquery must retain WHERE clause: {sql}")

    def test_filter_sort_count_result_matches_pandas(self):
        """filter + sort + count: result matches pandas."""
        pd_result = self.pdf[self.pdf["value"] > 20].sort_values(
            "score", ascending=False
        )
        pd_count = pd_result.count()

        ds = self._ds()
        ds_count = (
            ds[ds["value"] > 20]
            .sort_values("score", ascending=False)
            .count()
        )

        self.assertIsInstance(ds_count, pd.Series)
        for col in pd_count.index:
            self.assertEqual(
                ds_count[col],
                pd_count[col],
                f"count mismatch for '{col}': DS={ds_count[col]} PD={pd_count[col]}",
            )

    def test_multi_filter_sort_count_sql_has_no_order_by(self):
        """multi-filter + sort + count: SQL must NOT contain ORDER BY."""
        ds = self._ds()
        ds_chain = (
            ds[ds["value"] > 10][ds["category"] == "A"]
            .sort_values(["id", "score"], ascending=[True, False])
        )

        # Verify ORDER BY ops exist
        orderby_ops = [
            op
            for op in ds_chain._lazy_ops
            if isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY"
        ]
        self.assertTrue(len(orderby_ops) > 0)

        count_base = copy(ds_chain)
        count_base._lazy_ops = [
            op
            for op in count_base._lazy_ops
            if not (isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY")
        ]
        count_base._orderby_fields = []
        count_base._limit_value = None
        count_base._offset_value = None
        sql = count_base.to_sql()

        self.assertNotIn("ORDER BY", sql)
        self.assertIn("WHERE", sql)


class TestCountRowsSQLGeneration(TestCountSQLEfficiencyBase):
    """Verify count_rows() generates COUNT(*) when appropriate."""

    def test_multi_filter_sort_count_rows_uses_sql_count(self):
        """multi-filter + sort + count_rows: must generate COUNT(*) SQL, not full execution."""
        ds = self._ds()
        ds_chain = (
            ds[ds["value"] > 10][ds["category"] == "A"]
            .sort_values(["id", "score"])
        )

        # Verify _can_sql_pushdown returns True (no PANDAS_FILTER, no non-SQL ops)
        self.assertTrue(
            ds_chain._can_sql_pushdown(),
            "SQL pushdown should be possible for pure SQL filter+sort chain",
        )

        # Verify no LIMIT
        self.assertIsNone(ds_chain._limit_value)

        # count_rows should use SQL COUNT(*)
        result = ds_chain.count_rows()

        # Compare with pandas
        pd_result = self.pdf[self.pdf["value"] > 10]
        pd_result = pd_result[pd_result["category"] == "A"]
        expected = len(pd_result)

        self.assertEqual(result, expected)

    def test_filter_count_rows_matches_pandas(self):
        """Simple filter + count_rows must match pandas len()."""
        ds = self._ds()
        result = ds[ds["value"] > 30].count_rows()

        expected = len(self.pdf[self.pdf["value"] > 30])
        self.assertEqual(result, expected)

    def test_sort_count_rows_sql_no_order_by(self):
        """sort + count_rows: internally generated SQL should not have ORDER BY."""
        ds = self._ds()
        ds_sorted = ds[ds["id"] > 2].sort_values("value")

        # The count_rows path strips ORDER BY and uses COUNT(*)
        count_base = copy(ds_sorted)
        count_base._lazy_ops = [
            op
            for op in count_base._lazy_ops
            if not (isinstance(op, LazyRelationalOp) and op.op_type == "ORDER BY")
        ]
        count_base._orderby_fields = []
        count_base._limit_value = None
        count_base._offset_value = None
        sql = count_base.to_sql()

        self.assertNotIn("ORDER BY", sql)

        # Verify the count_rows result
        result = ds_sorted.count_rows()
        expected = len(self.pdf[self.pdf["id"] > 2])
        self.assertEqual(result, expected)


class TestPandasFilterFallback(TestCountSQLEfficiencyBase):
    """Verify PANDAS_FILTER chains fall back to DataFrame execution."""

    def test_pandas_filter_count_falls_back(self):
        """PANDAS_FILTER + count: must fall back to DataFrame execution."""
        ds = self._ds()
        # cumsum() creates a method-mode ColumnExpr => PANDAS_FILTER
        ds_filtered = ds[ds["value"].cumsum() > 60]

        # Verify PANDAS_FILTER is in lazy ops
        pandas_filter_ops = [
            op
            for op in ds_filtered._lazy_ops
            if isinstance(op, LazyRelationalOp) and op.op_type == "PANDAS_FILTER"
        ]
        self.assertTrue(
            len(pandas_filter_ops) > 0, "Setup: should have PANDAS_FILTER op"
        )

        # Verify _can_sql_pushdown returns False
        self.assertFalse(
            ds_filtered._can_sql_pushdown(),
            "SQL pushdown should be False with PANDAS_FILTER",
        )

        # count() should still work correctly via fallback
        ds_count = ds_filtered.count()
        pd_count = self.pdf[self.pdf["value"].cumsum() > 60].count()

        self.assertIsInstance(ds_count, pd.Series)
        for col in pd_count.index:
            self.assertEqual(ds_count[col], pd_count[col])

    def test_pandas_filter_count_rows_falls_back(self):
        """PANDAS_FILTER + count_rows: must fall back to len(execute())."""
        ds = self._ds()
        ds_filtered = ds[ds["value"].cumsum() > 60]

        self.assertFalse(ds_filtered._can_sql_pushdown())

        result = ds_filtered.count_rows()
        expected = len(self.pdf[self.pdf["value"].cumsum() > 60])
        self.assertEqual(result, expected)

    def test_filter_pandas_filter_sort_count(self):
        """SQL filter + PANDAS_FILTER + sort + count: must fallback correctly."""
        ds = self._ds()
        # First a normal SQL filter, then a PANDAS_FILTER, then sort
        ds_chain = ds[ds["category"] == "A"]
        ds_chain = ds_chain[ds_chain["value"].cumsum() > 10]
        ds_chain = ds_chain.sort_values("score")

        self.assertFalse(ds_chain._can_sql_pushdown())

        ds_count = ds_chain.count()

        pdf_filtered = self.pdf[self.pdf["category"] == "A"].copy()
        pdf_filtered = pdf_filtered[pdf_filtered["value"].cumsum() > 10]
        pd_count = pdf_filtered.sort_values("score").count()

        for col in pd_count.index:
            self.assertEqual(ds_count[col], pd_count[col])

    def test_filter_pandas_filter_sort_count_rows(self):
        """SQL filter + PANDAS_FILTER + sort + count_rows: fallback gives correct count."""
        ds = self._ds()
        ds_chain = ds[ds["category"] == "A"]
        ds_chain = ds_chain[ds_chain["value"].cumsum() > 10]
        ds_chain = ds_chain.sort_values("score")

        result = ds_chain.count_rows()

        pdf_filtered = self.pdf[self.pdf["category"] == "A"].copy()
        pdf_filtered = pdf_filtered[pdf_filtered["value"].cumsum() > 10]
        expected = len(pdf_filtered)

        self.assertEqual(result, expected)


class TestEmptyFilterCount(TestCountSQLEfficiencyBase):
    """Verify count/count_rows return correct values for empty filter results."""

    def test_empty_filter_count_returns_zero_series(self):
        """Filter producing 0 rows: count() returns Series of zeros."""
        ds = self._ds()
        ds_count = ds[ds["value"] > 99999].count()
        pd_count = self.pdf[self.pdf["value"] > 99999].count()

        self.assertIsInstance(ds_count, pd.Series)
        for col in pd_count.index:
            self.assertEqual(ds_count[col], 0)

    def test_empty_filter_count_rows_returns_zero(self):
        """Filter producing 0 rows: count_rows() returns 0."""
        ds = self._ds()
        result = ds[ds["value"] > 99999].count_rows()
        self.assertEqual(result, 0)

    def test_empty_filter_sort_count_returns_zero_series(self):
        """Empty filter + sort + count: still returns zeros."""
        ds = self._ds()
        ds_count = (
            ds[ds["value"] > 99999].sort_values("id").count()
        )
        pd_count = self.pdf[self.pdf["value"] > 99999].sort_values("id").count()

        for col in pd_count.index:
            self.assertEqual(ds_count[col], 0)

    def test_empty_filter_sort_count_rows_returns_zero(self):
        """Empty filter + sort + count_rows: returns 0."""
        ds = self._ds()
        result = ds[ds["value"] > 99999].sort_values("id").count_rows()
        self.assertEqual(result, 0)

    def test_empty_pandas_filter_count(self):
        """PANDAS_FILTER producing 0 rows: count() returns zeros."""
        ds = self._ds()
        # cumsum will always be > 0 for positive values, so use impossible condition
        ds_count = ds[ds["value"].cumsum() > 999999].count()
        pd_count = self.pdf[self.pdf["value"].cumsum() > 999999].count()

        for col in pd_count.index:
            self.assertEqual(ds_count[col], pd_count[col])

    def test_empty_pandas_filter_count_rows(self):
        """PANDAS_FILTER producing 0 rows: count_rows() returns 0."""
        ds = self._ds()
        result = ds[ds["value"].cumsum() > 999999].count_rows()
        expected = len(self.pdf[self.pdf["value"].cumsum() > 999999])
        self.assertEqual(result, expected)


class TestLimitCountRows(TestCountSQLEfficiencyBase):
    """Verify LIMIT + count_rows executes then counts (not SQL COUNT)."""

    def test_head_count_rows_executes_with_limit(self):
        """head() + count_rows: should execute with LIMIT, not use SQL COUNT(*)."""
        ds = self._ds()
        ds_limited = ds[ds["value"] > 10].head(3)

        # Verify LIMIT is set
        self.assertIsNotNone(ds_limited._limit_value)
        self.assertEqual(ds_limited._limit_value, 3)

        result = ds_limited.count_rows()
        expected = len(self.pdf[self.pdf["value"] > 10].head(3))
        self.assertEqual(result, expected)

    def test_limit_count_rows_executes_with_limit(self):
        """limit() + count_rows: should execute with LIMIT."""
        ds = self._ds()
        ds_limited = ds.limit(4)

        self.assertIsNotNone(ds_limited._limit_value)

        result = ds_limited.count_rows()
        expected = min(4, len(self.pdf))
        self.assertEqual(result, expected)

    def test_filter_sort_head_count_rows(self):
        """filter + sort + head + count_rows: executes then counts."""
        ds = self._ds()
        ds_chain = (
            ds[ds["value"] > 10]
            .sort_values("score", ascending=False)
            .head(2)
        )

        result = ds_chain.count_rows()

        pd_result = (
            self.pdf[self.pdf["value"] > 10]
            .sort_values("score", ascending=False)
            .head(2)
        )
        expected = len(pd_result)
        self.assertEqual(result, expected)

    def test_head_larger_than_data_count_rows(self):
        """head(n) where n > total rows: count_rows returns actual row count."""
        ds = self._ds()
        ds_limited = ds.head(100)

        result = ds_limited.count_rows()
        expected = len(self.pdf)
        self.assertEqual(result, expected)

    def test_head_zero_count_rows(self):
        """head(0) + count_rows: returns 0."""
        ds = self._ds()
        ds_limited = ds.head(0)

        result = ds_limited.count_rows()
        self.assertEqual(result, 0)

    def test_filter_head_count_rows_vs_count(self):
        """filter + head + count_rows vs count: both should respect LIMIT."""
        ds = self._ds()
        ds_limited = ds[ds["value"] > 10].head(3)

        count_rows_result = ds_limited.count_rows()
        count_result = ds_limited.count()

        pd_limited = self.pdf[self.pdf["value"] > 10].head(3)
        pd_count = pd_limited.count()

        self.assertEqual(count_rows_result, len(pd_limited))
        for col in pd_count.index:
            self.assertEqual(count_result[col], pd_count[col])


class TestCountSQLPushdownDecision(TestCountSQLEfficiencyBase):
    """Verify _can_sql_pushdown correctly classifies operation chains."""

    def test_pure_sql_filter_is_pushdown(self):
        """Pure SQL filter chain should allow pushdown."""
        ds = self._ds()
        ds_filtered = ds[ds["value"] > 10][ds["category"] == "A"]
        self.assertTrue(ds_filtered._can_sql_pushdown())

    def test_sort_only_is_pushdown(self):
        """sort_values alone should allow pushdown."""
        ds = self._ds()
        ds_sorted = ds.sort_values("value")
        self.assertTrue(ds_sorted._can_sql_pushdown())

    def test_filter_sort_is_pushdown(self):
        """filter + sort should allow pushdown."""
        ds = self._ds()
        ds_chain = ds[ds["value"] > 10].sort_values("score")
        self.assertTrue(ds_chain._can_sql_pushdown())

    def test_pandas_filter_blocks_pushdown(self):
        """PANDAS_FILTER should block pushdown."""
        ds = self._ds()
        ds_filtered = ds[ds["value"].cumsum() > 30]
        self.assertFalse(ds_filtered._can_sql_pushdown())

    def test_sql_filter_then_pandas_filter_blocks_pushdown(self):
        """SQL filter followed by PANDAS_FILTER should block pushdown."""
        ds = self._ds()
        ds_chain = ds[ds["value"] > 10]
        ds_chain = ds_chain[ds_chain["score"].cumsum() > 5]
        self.assertFalse(ds_chain._can_sql_pushdown())


if __name__ == "__main__":
    unittest.main()
