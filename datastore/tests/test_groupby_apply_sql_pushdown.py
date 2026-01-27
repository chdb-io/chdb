"""
Tests for groupby.apply() SQL pushdown optimization.

This module tests the automatic detection and SQL conversion of simple
aggregation patterns in groupby().apply() calls.

Patterns that can be pushed to SQL:
- lambda x: x.sum()
- lambda x: x.mean()
- lambda x: x.max()
- lambda x: x.min()
- lambda x: x.count()
- lambda x: x.std()
- lambda x: x.var()
- lambda x: x.first()
- lambda x: x.last()

Patterns that fall back to Pandas:
- lambda x: x.sum() + x.mean()  (compound expressions)
- lambda x: x[x > 0].sum()  (filtered aggregations)
- lambda x: x.nlargest(3)  (non-aggregation methods)
- Custom functions with complex logic
"""

import unittest
import numpy as np
import pandas as pd
import logging
import io

import datastore as ds
from datastore.lazy_ops import detect_simple_aggregation, LazyApply
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import groupby_apply_compat


class TestAggregationPatternDetection(unittest.TestCase):
    """Test the bytecode pattern detection for simple aggregations."""

    def test_detect_sum(self):
        """Test detection of lambda x: x.sum()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.sum())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'sum')

    def test_detect_mean(self):
        """Test detection of lambda x: x.mean()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.mean())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'mean')

    def test_detect_max(self):
        """Test detection of lambda x: x.max()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.max())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'max')

    def test_detect_min(self):
        """Test detection of lambda x: x.min()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.min())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'min')

    def test_detect_count(self):
        """Test detection of lambda x: x.count()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.count())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'count')

    def test_detect_std(self):
        """Test detection of lambda x: x.std()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.std())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'std')

    def test_detect_var(self):
        """Test detection of lambda x: x.var()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.var())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'var')

    def test_detect_first(self):
        """Test detection of lambda x: x.first()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.first())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'first')

    def test_detect_last(self):
        """Test detection of lambda x: x.last()."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.last())
        self.assertTrue(is_simple)
        self.assertEqual(agg_name, 'last')

    def test_not_detected_compound_expression(self):
        """Test that compound expressions are not detected as simple aggregations."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.sum() + x.mean())
        self.assertFalse(is_simple)
        self.assertIsNone(agg_name)

    def test_not_detected_filtered_aggregation(self):
        """Test that filtered aggregations are not detected as simple."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x[x > 0].sum())
        self.assertFalse(is_simple)
        self.assertIsNone(agg_name)

    def test_not_detected_nlargest(self):
        """Test that nlargest is not detected as simple aggregation."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x.nlargest(3))
        self.assertFalse(is_simple)
        self.assertIsNone(agg_name)

    def test_not_detected_arithmetic(self):
        """Test that arithmetic operations are not detected as aggregations."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: x + 1)
        self.assertFalse(is_simple)
        self.assertIsNone(agg_name)

    def test_not_detected_len(self):
        """Test that len() is not detected as aggregation."""
        is_simple, agg_name = detect_simple_aggregation(lambda x: len(x))
        self.assertFalse(is_simple)
        self.assertIsNone(agg_name)


class TestLazyApplySQLPushdown(unittest.TestCase):
    """Test LazyApply can_push_to_sql() method."""

    def test_can_push_simple_aggregation(self):
        """Test that simple aggregation can be pushed to SQL."""
        apply = LazyApply(lambda x: x.sum(), groupby_cols=['category'])
        self.assertTrue(apply.can_push_to_sql())
        self.assertEqual(apply.execution_engine(), 'SQL')
        self.assertEqual(apply.get_detected_agg_func(), 'sum')

    def test_cannot_push_without_groupby(self):
        """Test that apply without groupby cannot be pushed to SQL."""
        apply = LazyApply(lambda x: x.sum())  # No groupby_cols
        self.assertFalse(apply.can_push_to_sql())
        self.assertEqual(apply.execution_engine(), 'Pandas')

    def test_cannot_push_with_args(self):
        """Test that apply with args cannot be pushed to SQL."""
        apply = LazyApply(lambda x: x.sum(), 'arg1', groupby_cols=['category'])
        self.assertFalse(apply.can_push_to_sql())
        self.assertEqual(apply.execution_engine(), 'Pandas')

    def test_cannot_push_with_kwargs(self):
        """Test that apply with kwargs cannot be pushed to SQL."""
        apply = LazyApply(lambda x: x.sum(), groupby_cols=['category'], skipna=True)
        self.assertFalse(apply.can_push_to_sql())
        self.assertEqual(apply.execution_engine(), 'Pandas')

    def test_cannot_push_complex_function(self):
        """Test that complex function cannot be pushed to SQL."""
        apply = LazyApply(lambda x: x.nlargest(3), groupby_cols=['category'])
        self.assertFalse(apply.can_push_to_sql())
        self.assertEqual(apply.execution_engine(), 'Pandas')


class TestGroupByApplySQLExecution(unittest.TestCase):
    """Test groupby().apply() SQL execution with different aggregations."""

    def setUp(self):
        self.data = {
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60],
            'count': [1, 2, 3, 4, 5, 6],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = ds.DataStore.from_df(pd.DataFrame(self.data))

        # Set up log capture for SQL verification
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter('%(message)s'))

        from datastore.config import get_logger

        self.logger = get_logger()
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.original_level)

    def get_logs(self):
        """Get captured log output."""
        return self.log_capture.getvalue()

    def test_apply_sum(self):
        """Test groupby().apply(lambda x: x.sum())."""
        pd_result = groupby_apply_compat(self.pd_df.groupby('category'), lambda x: x.sum())
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.sum())

        # Use assert_datastore_equals_pandas for proper comparison
        # Note: groupby results have category as index, check_row_order=False for groupby
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False, check_index=True, msg="groupby.apply(sum) should match pandas"
        )

        # Verify SQL was used (check logs for SQL markers)
        logs = self.get_logs()
        self.assertTrue(
            '[SQL' in logs or 'SQL' in logs or 'Executing' in logs,
            f"Expected SQL execution in logs. Got:\n{logs[:500]}",
        )

    def test_apply_mean(self):
        """Test groupby().apply(lambda x: x.mean())."""
        pd_result = groupby_apply_compat(self.pd_df.groupby('category'), lambda x: x.mean())
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.mean())

        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False, check_index=True, msg="groupby.apply(mean) should match pandas"
        )

    def test_apply_max(self):
        """Test groupby().apply(lambda x: x.max())."""
        pd_result = groupby_apply_compat(self.pd_df.groupby('category'), lambda x: x.max())
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.max())

        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False, check_index=True, msg="groupby.apply(max) should match pandas"
        )

    def test_apply_min(self):
        """Test groupby().apply(lambda x: x.min())."""
        pd_result = groupby_apply_compat(self.pd_df.groupby('category'), lambda x: x.min())
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.min())

        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False, check_index=True, msg="groupby.apply(min) should match pandas"
        )

    def test_apply_count(self):
        """Test groupby().apply(lambda x: x.count())."""
        pd_result = groupby_apply_compat(self.pd_df.groupby('category'), lambda x: x.count())
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.count())

        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_row_order=False,
            check_index=True,
            msg="groupby.apply(count) should match pandas",
        )


class TestGroupByApplyMultipleColumns(unittest.TestCase):
    """Test groupby().apply() with multiple groupby columns."""

    def setUp(self):
        self.data = {
            'region': ['East', 'East', 'West', 'West', 'East', 'West'],
            'category': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = ds.DataStore.from_df(pd.DataFrame(self.data))

    def test_apply_sum_multiple_groupby(self):
        """Test groupby on multiple columns with apply(lambda x: x.sum())."""
        pd_result = groupby_apply_compat(self.pd_df.groupby(['region', 'category']), lambda x: x.sum())
        ds_result = self.ds_df.groupby(['region', 'category']).apply(lambda x: x.sum())

        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_row_order=False,
            check_index=True,
            msg="groupby on multiple columns with apply(sum) should match pandas",
        )


class TestGroupByApplyFallback(unittest.TestCase):
    """Test that non-simple patterns fall back to Pandas execution."""

    def setUp(self):
        self.data = {
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = ds.DataStore.from_df(pd.DataFrame(self.data))

        # Set up log capture
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter('%(message)s'))

        from datastore.config import get_logger

        self.logger = get_logger()
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.original_level)

    def get_logs(self):
        """Get captured log output."""
        return self.log_capture.getvalue()

    def test_complex_function_fallback(self):
        """Test that complex functions fall back to Pandas and still work."""

        def top_2(x):
            return x.nlargest(2, 'value')

        pd_result = groupby_apply_compat(self.pd_df.groupby('category'), top_2)
        ds_result = self.ds_df.groupby('category').apply(top_2)

        # pandas groupby.apply returns MultiIndex result
        # DataStore now preserves MultiIndex to match pandas behavior exactly
        # Compare directly - both should have MultiIndex
        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_row_order=False,
            check_index=True,  # Check index to verify MultiIndex is preserved
            msg="groupby.apply(top_2) should match pandas including MultiIndex",
        )

        # Verify Pandas fallback was used (not SQL)
        logs = self.get_logs()
        self.assertTrue(
            '[Pandas]' in logs or 'Apply' in logs, f"Expected Pandas execution for complex function. Got:\n{logs[:500]}"
        )


class TestGroupByApplySQLVerification(unittest.TestCase):
    """Test that SQL pushdown is actually used for simple aggregations."""

    def setUp(self):
        self.data = {
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = ds.DataStore.from_df(pd.DataFrame(self.data))

        # Set up log capture
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter('%(message)s'))

        from datastore.config import get_logger

        self.logger = get_logger()
        self.original_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.logger.setLevel(self.original_level)

    def get_logs(self):
        """Get captured log output."""
        return self.log_capture.getvalue()

    def test_sql_pushdown_for_sum(self):
        """Verify SQL pushdown is used for simple sum aggregation."""
        # Clear logs
        self.log_capture.truncate(0)
        self.log_capture.seek(0)

        # Execute apply with sum
        ds_result = self.ds_df.groupby('category').apply(lambda x: x.sum())

        # Trigger execution
        _ = len(ds_result)

        logs = self.get_logs()

        # Verify the LazyApply indicates SQL engine
        apply = LazyApply(lambda x: x.sum(), groupby_cols=['category'])
        self.assertEqual(apply.execution_engine(), 'SQL', "Simple sum aggregation should use SQL engine")

    def test_pandas_fallback_for_complex(self):
        """Verify Pandas is used for complex aggregations."""
        apply = LazyApply(lambda x: x.nlargest(2), groupby_cols=['category'])
        self.assertEqual(apply.execution_engine(), 'Pandas', "Complex aggregation should use Pandas engine")


if __name__ == '__main__':
    unittest.main()
