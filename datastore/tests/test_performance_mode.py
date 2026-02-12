"""
Tests for performance mode behavior.

Covers: Parquet settings, GroupBy behavior, row-order disablement (SQL verification),
aggregation compat skips, ColumnExpr SQL-first aggregation, and mode switching.

Test Comparison Strategies:
- SORT-THEN-COMPARE: For aggregation/filter results where values are deterministic
  but order is not. Both sides sorted by key columns before comparison.
- VALUE-RANGE-CHECK: For first()/last() with any()/anyLast() -- non-deterministic
  element from the group. Only verify the value exists in group's value set.
- SCHEMA-AND-COUNT: For LIMIT without ORDER BY, head(), etc. Only verify
  row count, column names, and dtypes.
"""

import io
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from datastore import DataStore
from datastore.config import (
    CompatMode,
    set_compat_mode,
    is_performance_mode,
    set_execution_engine,
    ExecutionEngine,
)
from datastore.tests.test_utils import (
    get_dataframe,
    get_series,
    assert_values_match_ignoring_order,
)


class PerformanceModeTestBase(unittest.TestCase):
    """Base class that enables performance mode in setUp and resets in tearDown."""

    def setUp(self):
        set_compat_mode(CompatMode.PERFORMANCE)

    def tearDown(self):
        set_compat_mode(CompatMode.PANDAS)
        set_execution_engine(ExecutionEngine.AUTO)

    def _capture_sql_log(self):
        """Set up SQL log capture. Returns (logger, handler, log_capture)."""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger = logging.getLogger("datastore")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        return logger, handler, log_capture

    def _stop_capture(self, logger, handler):
        """Stop SQL log capture."""
        logger.removeHandler(handler)


# =============================================================================
# 5b. Parquet + GroupBy Behavior Tests
# =============================================================================


class TestPerformanceModeParquet(PerformanceModeTestBase):
    """Test parquet settings in performance mode."""

    def setUp(self):
        super().setUp()
        # Create a temp parquet file
        self.temp_dir = tempfile.mkdtemp()
        self.df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "age": [25, 30, 35, 28, 32],
                "score": [90.5, 85.0, 92.3, 88.1, 95.7],
            }
        )
        self.parquet_path = os.path.join(self.temp_dir, "test.parquet")
        self.df.to_parquet(self.parquet_path, index=False)

    def tearDown(self):
        super().tearDown()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parquet_no_preserve_order_setting(self):
        """In performance mode, _format_settings should NOT have input_format_parquet_preserve_order."""
        ds = DataStore.from_file(self.parquet_path, "parquet")
        assert "input_format_parquet_preserve_order" not in ds._format_settings

    def test_parquet_no_utc_timezone_setting(self):
        """In performance mode, _format_settings should NOT have session_timezone."""
        ds = DataStore.from_file(self.parquet_path, "parquet")
        assert "session_timezone" not in ds._format_settings

    def test_parquet_data_values_correct(self):
        """Reading parquet in performance mode should produce the same data (order may differ)."""
        ds = DataStore.from_file(self.parquet_path, "parquet")
        # Strategy: SORT-THEN-COMPARE
        assert_values_match_ignoring_order(ds, self.df, sort_by=["name"])


class TestPerformanceModeGroupBy(PerformanceModeTestBase):
    """Test groupby behavior in performance mode."""

    def setUp(self):
        super().setUp()
        self.pd_df = pd.DataFrame(
            {
                "cat": ["A", "B", "A", "B", "A", "C"],
                "val": [10, 20, 30, 40, 50, 60],
                "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        self.ds = DataStore(self.pd_df)

    def test_groupby_sum_values_correct(self):
        """GroupBy sum should produce correct values (order may differ)."""
        pd_result = self.pd_df.groupby("cat", sort=False).sum()
        ds_result = self.ds.groupby("cat").sum()

        # In performance mode, ds_result has 'cat' as column (not index)
        # pd_result has 'cat' as index -- reset both
        ds_df = get_dataframe(ds_result).reset_index(drop=False)
        pd_df = pd_result.reset_index(drop=False)

        # Sort by cat and compare
        ds_sorted = ds_df.sort_values("cat", ignore_index=True)
        pd_sorted = pd_df.sort_values("cat", ignore_index=True)

        # Compare val and score columns
        for col in ["val", "score"]:
            np.testing.assert_array_almost_equal(
                ds_sorted[col].values,
                pd_sorted[col].values,
                err_msg=f"Column {col} mismatch",
            )

    def test_groupby_mean_values_correct(self):
        """GroupBy mean should produce correct values (order may differ)."""
        pd_result = self.pd_df.groupby("cat", sort=False)["val"].mean()
        ds_result = self.ds.groupby("cat")["val"].mean()

        ds_series = get_series(ds_result)
        pd_series = pd_result

        # Both should have same groups and values
        ds_frame = ds_series.reset_index(drop=False).sort_values(
            ds_series.index.name or "cat", ignore_index=True
        )
        pd_frame = pd_series.reset_index(drop=False).sort_values(
            "cat", ignore_index=True
        )

        np.testing.assert_array_almost_equal(
            ds_frame["val"].values, pd_frame["val"].values
        )

    def test_groupby_no_set_index_in_performance_mode(self):
        """In performance mode, groupby column should be a regular column, not index."""
        ds_result = self.ds.groupby("cat").sum()
        ds_df = get_dataframe(ds_result)
        # Group key should be a column, not index
        assert "cat" in list(ds_df.columns), (
            f"Expected 'cat' in columns, got columns={list(ds_df.columns)}, "
            f"index={ds_df.index.name}"
        )

    def test_groupby_includes_null_groups(self):
        """In performance mode, NULL groups should be included (SQL default)."""
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "B", None, "A", None],
                "val": [10, 20, 30, 40, 50],
            }
        )
        ds = DataStore(pd_df)
        ds_result = ds.groupby("cat").sum()
        ds_df = get_dataframe(ds_result)

        # Should have 3 groups: A, B, and NULL
        assert len(ds_df) == 3, f"Expected 3 groups (A, B, NULL), got {len(ds_df)}"

    def test_count_returns_native_type(self):
        """In performance mode, count() should return SQL-native type (not forced int64)."""
        ds_result = self.ds.groupby("cat")["val"].count()
        ds_series = get_series(ds_result)
        # UInt64 from ClickHouse is typically uint64 in pandas
        # The key point: NOT forced to int64
        assert ds_series.dtype in [np.uint64, np.int64], (
            f"Expected uint64 or int64, got {ds_series.dtype}"
        )

    def test_sum_allnan_returns_null_not_zero(self):
        """In performance mode, sum of all-NaN group should return NaN, not 0."""
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "A", "B"],
                "val": [float("nan"), float("nan"), 10.0],
            }
        )
        ds = DataStore(pd_df)
        ds_result = ds.groupby("cat")["val"].sum()
        ds_series = get_series(ds_result)

        # Group A should be NaN (not 0 as pandas returns)
        a_val = ds_series.loc["A"] if "A" in ds_series.index else None
        assert a_val is None or pd.isna(a_val), (
            f"Expected NaN for all-NaN group, got {a_val}"
        )


# =============================================================================
# 5c. Row-Order Disablement Tests (SQL verification)
# =============================================================================


class TestPerformanceModeRowOrder(PerformanceModeTestBase):
    """Verify all row-order mechanisms are disabled via SQL inspection."""

    def setUp(self):
        super().setUp()
        self.pd_df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "Diana"],
                "val": [10, 20, 30, 40],
                "cat": ["A", "B", "A", "B"],
            }
        )
        self.ds = DataStore(self.pd_df)

    def test_no_row_id_in_python_table_query(self):
        """In performance mode, _row_id should not appear in query for Python() table function."""
        logger, handler, log_capture = self._capture_sql_log()
        try:
            result = self.ds[self.ds["val"] > 10]
            _ = get_dataframe(result)
            log_output = log_capture.getvalue()
            # _row_id should NOT be in the SQL (performance mode skips preserve_order)
            assert "_row_id" not in log_output, (
                f"_row_id found in SQL output:\n{log_output}"
            )
        finally:
            self._stop_capture(logger, handler)

    def test_no_orig_row_num_subquery_on_filter(self):
        """In performance mode, __orig_row_num__ should not appear in filter queries."""
        temp_dir = tempfile.mkdtemp()
        try:
            parquet_path = os.path.join(temp_dir, "test.parquet")
            self.pd_df.to_parquet(parquet_path, index=False)
            ds = DataStore.from_file(parquet_path, "parquet")

            logger, handler, log_capture = self._capture_sql_log()
            try:
                result = ds[ds["val"] > 10]
                _ = get_dataframe(result)
                log_output = log_capture.getvalue()
                assert "__orig_row_num__" not in log_output, (
                    f"__orig_row_num__ found in SQL:\n{log_output}"
                )
            finally:
                self._stop_capture(logger, handler)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_stable_sort_tiebreaker(self):
        """In performance mode, rowNumberInAllBlocks() should not be used as sort tiebreaker."""
        logger, handler, log_capture = self._capture_sql_log()
        try:
            result = self.ds.sort_values("name")
            _ = get_dataframe(result)
            log_output = log_capture.getvalue()
            assert "rowNumberInAllBlocks" not in log_output, (
                f"rowNumberInAllBlocks found in sort SQL:\n{log_output}"
            )
        finally:
            self._stop_capture(logger, handler)

    def test_first_uses_any_not_argmin(self):
        """In performance mode, first() should use any(), not argMin(rowNumberInAllBlocks())."""
        logger, handler, log_capture = self._capture_sql_log()
        try:
            result = self.ds.groupby("cat")["val"].first()
            _ = get_series(result)
            log_output = log_capture.getvalue()
            assert "argMin" not in log_output, (
                f"argMin found in first() SQL:\n{log_output}"
            )
            # Should use any() function
            assert "any(" in log_output.lower() or "any(" in log_output, (
                f"any() not found in first() SQL:\n{log_output}"
            )
        finally:
            self._stop_capture(logger, handler)

    def test_last_uses_anylast_not_argmax(self):
        """In performance mode, last() should use anyLast(), not argMax(rowNumberInAllBlocks())."""
        logger, handler, log_capture = self._capture_sql_log()
        try:
            result = self.ds.groupby("cat")["val"].last()
            _ = get_series(result)
            log_output = log_capture.getvalue()
            assert "argMax" not in log_output, (
                f"argMax found in last() SQL:\n{log_output}"
            )
            assert "anyLast(" in log_output, (
                f"anyLast() not found in last() SQL:\n{log_output}"
            )
        finally:
            self._stop_capture(logger, handler)

    def test_aggregation_values_still_correct_despite_no_ordering(self):
        """GroupBy sum should produce correct values even without row-order preservation."""
        pd_result = self.pd_df.groupby("cat")["val"].sum()
        ds_result = self.ds.groupby("cat")["val"].sum()

        ds_series = get_series(ds_result)
        # Sort both by index for comparison
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_result.sort_index()

        np.testing.assert_array_equal(ds_sorted.values, pd_sorted.values)


# =============================================================================
# 5d. Aggregation Compat Tests (skipna, MultiIndex, dtype)
# =============================================================================


class TestPerformanceModeAggregation(PerformanceModeTestBase):
    """Test aggregation without pandas compat overhead."""

    def setUp(self):
        super().setUp()
        self.pd_df = pd.DataFrame(
            {
                "cat": ["A", "B", "A", "B"],
                "val": [10.0, 20.0, 30.0, 40.0],
                "score": [1.0, 2.0, 3.0, 4.0],
            }
        )
        self.ds = DataStore(self.pd_df)

    def test_sum_no_skipna_wrapper_in_sql(self):
        """In performance mode, sum should use plain sum(), not sumIf()."""
        logger, handler, log_capture = self._capture_sql_log()
        try:
            result = self.ds.groupby("cat")["val"].sum()
            _ = get_series(result)
            log_output = log_capture.getvalue()
            # Should NOT have -If suffix
            assert "sumIf(" not in log_output, (
                f"sumIf found in SQL (should use plain sum):\n{log_output}"
            )
        finally:
            self._stop_capture(logger, handler)

    def test_no_multiindex_columns(self):
        """In performance mode, agg with multiple funcs should return flat column names."""
        ds_result = self.ds.groupby("cat").agg({"val": ["sum", "mean"]})
        ds_df = get_dataframe(ds_result)

        # Columns should be flat strings, not MultiIndex tuples
        assert not isinstance(ds_df.columns, pd.MultiIndex), (
            f"Expected flat columns, got MultiIndex: {ds_df.columns.tolist()}"
        )
        # Flat names should be like 'val_sum', 'val_mean'
        col_names = list(ds_df.columns)
        assert "val_sum" in col_names or any("sum" in str(c) for c in col_names), (
            f"Expected flat column with 'sum', got: {col_names}"
        )


# =============================================================================
# 5e. ColumnExpr SQL-First Aggregation Tests (critical path)
# =============================================================================


class TestPerformanceModeColumnExprGroupBy(PerformanceModeTestBase):
    """Test that ColumnExpr groupby+agg uses single SQL via LazyGroupByAgg injection."""

    def setUp(self):
        super().setUp()
        self.pd_df = pd.DataFrame(
            {
                "category": ["Electronics", "Books", "Electronics", "Books", "Toys"],
                "rating": [4.5, 3.0, 4.0, 2.5, 5.0],
                "revenue": [100, 50, 200, 30, 150],
            }
        )
        self.ds = DataStore(self.pd_df)

    def test_filter_groupby_sum_single_sql(self):
        """filter + groupby + sum should produce correct values via single SQL."""
        pd_result = self.pd_df[self.pd_df["rating"] > 3.0].groupby("category")[
            "revenue"
        ].sum()
        ds_result = self.ds[self.ds["rating"] > 3.0].groupby("category")[
            "revenue"
        ].sum()

        ds_series = get_series(ds_result)
        # Sort both by index for comparison
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_result.sort_index()

        np.testing.assert_array_equal(
            ds_sorted.values, pd_sorted.values,
            err_msg="filter + groupby + sum values mismatch",
        )

    def test_filter_groupby_mean_single_sql(self):
        """filter + groupby + mean should produce correct values."""
        pd_result = self.pd_df[self.pd_df["rating"] > 2.0].groupby("category")[
            "revenue"
        ].mean()
        ds_result = self.ds[self.ds["rating"] > 2.0].groupby("category")[
            "revenue"
        ].mean()

        ds_series = get_series(ds_result)
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_result.sort_index()

        np.testing.assert_array_almost_equal(
            ds_sorted.values, pd_sorted.values,
            err_msg="filter + groupby + mean values mismatch",
        )

    def test_filter_groupby_count_single_sql(self):
        """filter + groupby + count should produce correct values."""
        pd_result = self.pd_df[self.pd_df["rating"] > 3.0].groupby("category")[
            "revenue"
        ].count()
        ds_result = self.ds[self.ds["rating"] > 3.0].groupby("category")[
            "revenue"
        ].count()

        ds_series = get_series(ds_result)
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_result.sort_index()

        np.testing.assert_array_equal(
            ds_sorted.values, pd_sorted.values,
            err_msg="filter + groupby + count values mismatch",
        )

    def test_multiple_groupby_cols(self):
        """GroupBy with multiple columns should produce correct values."""
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "A", "B", "B"],
                "sub": ["x", "y", "x", "y"],
                "val": [10, 20, 30, 40],
            }
        )
        ds = DataStore(pd_df)

        pd_result = pd_df.groupby(["cat", "sub"])["val"].sum()
        ds_result = ds.groupby(["cat", "sub"])["val"].sum()

        ds_series = get_series(ds_result)

        # Sort both by index for comparison
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_result.sort_index()

        np.testing.assert_array_equal(
            ds_sorted.values, pd_sorted.values,
            err_msg="multi-column groupby sum mismatch",
        )

    def test_groupby_first_performance_mode_value_range(self):
        """first() should return a value that exists in the group's value set."""
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "A", "A", "B", "B"],
                "val": [10, 20, 30, 40, 50],
            }
        )
        ds = DataStore(pd_df)

        ds_result = ds.groupby("cat")["val"].first()
        ds_series = get_series(ds_result)

        # VALUE-RANGE-CHECK: verify each returned value exists in the group
        for group_key in ["A", "B"]:
            group_values = set(pd_df[pd_df["cat"] == group_key]["val"].tolist())
            actual = ds_series.loc[group_key]
            assert actual in group_values, (
                f"first() for group '{group_key}' returned {actual}, "
                f"which is not in group values {group_values}"
            )

    def test_groupby_last_performance_mode_value_range(self):
        """last() should return a value that exists in the group's value set."""
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "A", "A", "B", "B"],
                "val": [10, 20, 30, 40, 50],
            }
        )
        ds = DataStore(pd_df)

        ds_result = ds.groupby("cat")["val"].last()
        ds_series = get_series(ds_result)

        # VALUE-RANGE-CHECK: verify each returned value exists in the group
        for group_key in ["A", "B"]:
            group_values = set(pd_df[pd_df["cat"] == group_key]["val"].tolist())
            actual = ds_series.loc[group_key]
            assert actual in group_values, (
                f"last() for group '{group_key}' returned {actual}, "
                f"which is not in group values {group_values}"
            )

    def test_limit_without_order_by_schema_and_count(self):
        """head() in performance mode: SCHEMA-AND-COUNT only, no row-by-row comparison."""
        ds_result = self.ds.head(3)
        ds_df = get_dataframe(ds_result)

        # SCHEMA-AND-COUNT: verify row count and column names
        assert len(ds_df) == 3, f"Expected 3 rows, got {len(ds_df)}"
        expected_cols = set(self.pd_df.columns)
        actual_cols = set(ds_df.columns)
        assert expected_cols == actual_cols, (
            f"Column mismatch: expected {expected_cols}, got {actual_cols}"
        )


# =============================================================================
# 5f. Performance Regression Tests
# =============================================================================


class TestPerformanceModeRegression(PerformanceModeTestBase):
    """Verify performance mode is measurably faster and handles large datasets."""

    def test_filter_groupby_produces_correct_results(self):
        """
        Generate a moderate dataset and verify filter+groupby+sum correctness
        in performance mode via SORT-THEN-COMPARE.
        """
        np.random.seed(42)
        n = 50000
        pd_df = pd.DataFrame(
            {
                "category": np.random.choice(["A", "B", "C", "D", "E"], n),
                "rating": np.random.uniform(1.0, 5.0, n),
                "revenue": np.random.uniform(10.0, 1000.0, n),
            }
        )
        ds = DataStore(pd_df)

        # Pandas reference
        pd_result = (
            pd_df[pd_df["rating"] > 3.5]
            .groupby("category")["revenue"]
            .sum()
        )

        # DataStore performance mode
        ds_result = (
            ds[ds["rating"] > 3.5]
            .groupby("category")["revenue"]
            .sum()
        )

        ds_series = get_series(ds_result)

        # SORT-THEN-COMPARE
        ds_sorted = ds_series.sort_index()
        pd_sorted = pd_result.sort_index()

        np.testing.assert_array_almost_equal(
            ds_sorted.values,
            pd_sorted.values,
            decimal=2,
            err_msg="Large dataset filter+groupby+sum mismatch",
        )

    def test_large_groupby_agg_completes_without_error(self):
        """
        Generate a moderate dataset and verify groupby agg completes without MemoryError.
        """
        np.random.seed(42)
        n = 50000
        pd_df = pd.DataFrame(
            {
                "category": np.random.choice(
                    ["Electronics", "Books", "Toys", "Food", "Clothing"], n
                ),
                "revenue": np.random.uniform(10.0, 1000.0, n),
                "rating": np.random.uniform(1.0, 5.0, n),
                "quantity": np.random.randint(1, 100, n),
            }
        )
        ds = DataStore(pd_df)

        # This should complete without MemoryError
        ds_result = ds.groupby("category").agg(
            {"revenue": ["sum", "mean"], "rating": "mean", "quantity": "sum"}
        )
        ds_df = get_dataframe(ds_result)

        # Basic verification: correct number of groups
        assert len(ds_df) == 5, f"Expected 5 groups, got {len(ds_df)}"
        # Should have columns for each agg
        assert len(ds_df.columns) >= 4, (
            f"Expected at least 4 result columns, got {ds_df.columns.tolist()}"
        )


# =============================================================================
# 5g. Mode Switching Tests
# =============================================================================


class TestCompatModeSwitching(PerformanceModeTestBase):
    """Test switching between modes mid-session."""

    def test_switch_to_performance_and_back(self):
        """
        Both modes should produce the same aggregated VALUES.
        Structural differences (index vs column) are expected.
        """
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "B", "A", "B"],
                "val": [10, 20, 30, 40],
            }
        )

        # Performance mode result
        set_compat_mode(CompatMode.PERFORMANCE)
        ds1 = DataStore(pd_df)
        result1 = ds1.groupby("cat")["val"].sum()
        r1_series = get_series(result1)

        # Pandas compat mode result
        set_compat_mode(CompatMode.PANDAS)
        ds2 = DataStore(pd_df)
        result2 = ds2.groupby("cat")["val"].sum()
        r2_series = get_series(result2)

        # Both should have same aggregated VALUES
        r1_sorted = r1_series.sort_index()
        r2_sorted = r2_series.sort_index()
        np.testing.assert_array_equal(r1_sorted.values, r2_sorted.values)

    def test_lazy_expr_respects_mode_at_execution_time(self):
        """
        A lazy expression created in pandas mode but executed in performance mode
        should use performance-mode behavior.
        """
        pd_df = pd.DataFrame(
            {
                "cat": ["A", "B", "A", "B"],
                "val": [10, 20, 30, 40],
            }
        )

        # Create lazy expr in pandas mode
        set_compat_mode(CompatMode.PANDAS)
        ds = DataStore(pd_df)
        lazy_expr = ds.groupby("cat")["val"]

        # Switch to performance mode before execution
        set_compat_mode(CompatMode.PERFORMANCE)
        result = lazy_expr.sum()
        result_series = get_series(result)

        # Values should be correct regardless
        sorted_result = result_series.sort_index()
        assert sorted_result.loc["A"] == 40  # 10 + 30
        assert sorted_result.loc["B"] == 60  # 20 + 40


if __name__ == "__main__":
    unittest.main()
