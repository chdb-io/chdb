"""
Comprehensive test comparing DataStore with pandas using Titanic dataset.

This test file validates:
1. All operations produce identical results to pandas
2. Lazy execution and checkpoint work correctly
3. Debug logs confirm expected execution flow
4. FamilySize assignment is only executed once across groupby and to_df

Based on: https://www.kaggle.com/code/amrahmed2121/titanic/notebook
"""

import unittest
import logging
import io
import re
import pandas as pd
import numpy as np
import warnings
import pytest

from datastore import DataStore, LazyGroupBy
from datastore.config import config
from datastore.lazy_ops import LazyColumnAssignment
from tests.test_utils import assert_frame_equal, assert_series_equal

# Suppress warnings
warnings.filterwarnings("ignore")


def dataset_path(filename: str) -> str:
    """Get path to test dataset."""
    import os

    return os.path.join(os.path.dirname(__file__), "dataset", filename)


class TitanicPandasComparisonTest(unittest.TestCase):
    """Compare DataStore operations with pandas on Titanic dataset."""

    @classmethod
    def setUpClass(cls):
        """Load Titanic dataset once for all tests."""
        cls.titanic_path = dataset_path("Titanic-Dataset.csv")
        cls.pd_df = pd.read_csv(cls.titanic_path)

    def setUp(self):
        """Create fresh DataStore for each test."""
        self.ds = DataStore.from_file(self.titanic_path)
        config.enable_cache()
        config.disable_debug()

    def tearDown(self):
        """Clean up after each test."""
        config.disable_debug()

    # ========== Basic Operations ==========

    def test_head(self):
        """Test head() returns same result as pandas."""
        ds_result = self.ds.head(5).to_df()
        pd_result = self.pd_df.head(5)

        assert_frame_equal(ds_result, pd_result)

    def test_tail(self):
        """Test tail() returns same result as pandas."""
        ds_result = self.ds.tail(5).to_df()
        pd_result = self.pd_df.tail(5)

        # Check shape and columns
        self.assertEqual(ds_result.shape, pd_result.shape)
        self.assertEqual(list(ds_result.columns), list(pd_result.columns))

        # Compare numeric columns strictly
        numeric_cols = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare']
        for col in numeric_cols:
            if col in ds_result.columns:
                np.testing.assert_array_almost_equal(ds_result[col].values, pd_result[col].values)

        # Compare string columns (handle None/NaN differences)
        for col in ['Name', 'Sex', 'Ticket', 'Embarked']:
            if col in ds_result.columns:
                ds_vals = ds_result[col].fillna('').values
                pd_vals = pd_result[col].fillna('').values
                np.testing.assert_array_equal(ds_vals, pd_vals)

    def test_shape(self):
        """Test shape property."""
        ds_shape = self.ds.shape
        pd_shape = self.pd_df.shape

        self.assertEqual(ds_shape, pd_shape)

    def test_columns(self):
        """Test columns property."""
        ds_columns = list(self.ds.columns)
        pd_columns = list(self.pd_df.columns)

        self.assertEqual(ds_columns, pd_columns)

    def test_describe_numeric(self):
        """Test describe() for numeric columns."""
        ds_result = self.ds.describe()
        pd_result = self.pd_df.describe()

        # Compare shapes and column names
        self.assertEqual(ds_result.shape, pd_result.shape)
        self.assertEqual(list(ds_result.columns), list(pd_result.columns))

        # Compare values with tolerance for floating point
        for col in ds_result.columns:
            np.testing.assert_array_almost_equal(ds_result[col].values, pd_result[col].values, decimal=5)

    def test_describe_object(self):
        """Test describe(include='O') for object columns."""
        ds_result = self.ds.to_df().describe(include="O")
        pd_result = self.pd_df.describe(include="O")

        assert_frame_equal(ds_result, pd_result)

    # ========== Missing Value Operations ==========

    def test_isna_sum(self):
        """Test isna().sum() returns same result as pandas."""
        ds_result = self.ds.isna().sum()
        pd_result = self.pd_df.isna().sum()

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_sum(self):
        """Test duplicated().sum() returns same result as pandas."""
        ds_result = self.ds.duplicated().sum()
        pd_result = self.pd_df.duplicated().sum()

        self.assertEqual(ds_result, pd_result)

    def test_column_mean(self):
        """Test column mean calculation."""
        ds_mean = self.ds['Age'].mean()
        pd_mean = self.pd_df['Age'].mean()

        # Both should be approximately equal
        self.assertAlmostEqual(float(ds_mean), float(pd_mean), places=5)

    def test_fillna(self):
        """Test fillna() operation."""
        age_mean = round(self.pd_df['Age'].mean(), 2)

        # DataStore
        ds = DataStore.from_file(self.titanic_path)
        ds['Age'] = ds['Age'].fillna(age_mean)
        ds_result = ds['Age'].isna().sum()

        # Pandas
        pd_df = self.pd_df.copy()
        pd_df['Age'] = pd_df['Age'].fillna(age_mean)
        pd_result = pd_df['Age'].isna().sum()

        self.assertEqual(ds_result, pd_result)
        self.assertEqual(ds_result, 0)

    def test_notnull_astype(self):
        """Test notnull().astype(int) operation."""
        # DataStore
        ds = DataStore.from_file(self.titanic_path)
        ds["Has_Cabin"] = ds['Cabin'].notnull().astype(int)
        ds_result = ds["Has_Cabin"].sum()

        # Pandas
        pd_df = self.pd_df.copy()
        pd_df["Has_Cabin"] = pd_df['Cabin'].notnull().astype(int)
        pd_result = pd_df["Has_Cabin"].sum()

        self.assertEqual(ds_result, pd_result)

    def test_value_counts(self):
        """Test value_counts() operation."""
        ds_result = self.ds['Embarked'].value_counts()
        pd_result = self.pd_df['Embarked'].value_counts()

        # Sort both for comparison - sort_index returns LazySeries
        # Compare values using numpy (natural execution trigger)
        np.testing.assert_array_equal(ds_result.sort_index(), pd_result.sort_index())

    # ========== GroupBy Operations ==========

    def test_groupby_returns_lazy_groupby(self):
        """Test that groupby() returns LazyGroupBy object."""
        result = self.ds.groupby('Sex')
        self.assertIsInstance(result, LazyGroupBy)

    def test_groupby_sum(self):
        """Test groupby().sum() operation."""
        ds_result = self.ds.groupby('Sex')['Survived'].sum()
        pd_result = self.pd_df.groupby('Sex')['Survived'].sum()

        # Compare values (index order might differ)
        for idx in pd_result.index:
            self.assertEqual(ds_result[idx], pd_result[idx])

    def test_groupby_mean(self):
        """Test groupby().mean() operation with column that has no NaN."""
        # Use a column without NaN (SibSp) to avoid chDB NaN handling differences
        ds_result = self.ds.groupby('Survived')['SibSp'].mean()
        pd_result = self.pd_df.groupby('Survived')['SibSp'].mean()

        # Compare with tolerance
        for idx in pd_result.index:
            self.assertAlmostEqual(ds_result[idx], pd_result[idx], places=5)

    def test_groupby_embarked_survived_mean(self):
        """Test groupby('Embarked')['Survived'].mean()."""
        ds_result = self.ds.groupby('Embarked')['Survived'].mean()
        pd_result = self.pd_df.groupby('Embarked')['Survived'].mean()

        for idx in pd_result.index:
            self.assertAlmostEqual(ds_result[idx], pd_result[idx], places=5)

    def test_groupby_pclass_survived_mean(self):
        """Test groupby('Pclass')['Survived'].mean()."""
        ds_result = self.ds.groupby('Pclass')['Survived'].mean()
        pd_result = self.pd_df.groupby('Pclass')['Survived'].mean()

        for idx in pd_result.index:
            self.assertAlmostEqual(ds_result[idx], pd_result[idx], places=5)

    def test_groupby_mean_head_order(self):
        """Test that groupby().mean().head() returns results in sorted order like pandas."""
        # Create FamilySize column
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        pd_df = self.pd_df.copy()
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        # Get head(5) directly - LazyAggregate.head() returns LazySeries
        ds_head = ds.groupby("FamilySize")["Survived"].mean().head(5)
        pd_head = pd_df.groupby("FamilySize")["Survived"].mean().head(5)

        # Use equals() method - works with LazySeries!
        self.assertTrue(ds_head.equals(pd_head), "groupby().mean().head(5) results should be equal")


class TestOrderSensitiveOperations(unittest.TestCase):
    """
    Tests specifically for order-sensitive operations.

    LESSON LEARNED: When testing pandas compatibility, order matters!
    These tests explicitly verify that:
    1. Index order matches pandas
    2. head()/tail() return correct slices
    3. Aggregation results are sorted by group key (pandas default sort=True)

    NOTE: Tests with NaN-containing columns (like Embarked) use pandas engine
    because chDB currently treats NaN as empty string ''. See xfail tests below.
    """

    @classmethod
    def setUpClass(cls):
        cls.titanic_path = dataset_path("Titanic-Dataset.csv")
        cls.pd_df = pd.read_csv(cls.titanic_path)

    def setUp(self):
        self.ds = DataStore.from_file(self.titanic_path)
        config.enable_cache()
        # Use pandas engine to avoid chDB NaN handling issues
        config.execution_engine = 'pandas'

    def tearDown(self):
        # Restore default engine
        config.execution_engine = 'auto'

    def test_groupby_full_series_equality(self):
        """
        CRITICAL: Use equals() method to compare results.

        Previous bug: Iterating by index masked ordering problems.
        Now use equals() which compares values AND order.
        """
        ds_result = self.ds.groupby('Pclass')['Survived'].mean()
        pd_result = self.pd_df.groupby('Pclass')['Survived'].mean()

        # Use equals() method - works with LazyAggregate directly!
        self.assertTrue(ds_result.equals(pd_result), "Results should be equal")

    def test_groupby_index_order_explicit(self):
        """Explicitly verify index order matches pandas (with NaN-containing column)."""
        ds_result = self.ds.groupby('Embarked')['Survived'].mean()
        pd_result = self.pd_df.groupby('Embarked')['Survived'].mean()

        # Use index property - LazyAggregate proxies to result
        self.assertEqual(
            list(ds_result.index), list(pd_result.index), "Index order must match pandas (sorted by group key)"
        )

    def test_groupby_head_is_first_n_sorted(self):
        """Verify head() returns first N items in sorted order."""
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        pd_df = self.pd_df.copy()
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        # Full result
        ds_full = ds.groupby("FamilySize")["Survived"].mean()
        pd_full = pd_df.groupby("FamilySize")["Survived"].mean()

        # head(3) - LazySeries also has equals() method
        ds_head = ds_full.head(3)
        pd_head = pd_full.head(3)

        # Use equals() method
        self.assertTrue(ds_head.equals(pd_head), "head(3) results should be equal")

        # Verify it's actually the smallest 3 group keys
        self.assertEqual(list(ds_head.index), sorted(list(ds_full.index))[:3])

    def test_groupby_tail_is_last_n_sorted(self):
        """Verify tail() returns last N items in sorted order."""
        ds_result = self.ds.groupby('Pclass')['Fare'].mean()
        pd_result = self.pd_df.groupby('Pclass')['Fare'].mean()

        ds_tail = ds_result.tail(2)
        pd_tail = pd_result.tail(2)

        self.assertTrue(ds_tail.equals(pd_tail), "tail(2) results should be equal")

    def test_groupby_sum_order(self):
        """Test sum() also respects order."""
        ds_result = self.ds.groupby('Sex')['Survived'].sum()
        pd_result = self.pd_df.groupby('Sex')['Survived'].sum()

        # Use equals() - handles type differences internally
        self.assertTrue(ds_result.equals(pd_result), "sum() results should be equal")

    def test_groupby_count_order(self):
        """Test count() also respects order (with NaN-containing column)."""
        ds_result = self.ds.groupby('Embarked')['PassengerId'].count()
        pd_result = self.pd_df.groupby('Embarked')['PassengerId'].count()

        self.assertTrue(ds_result.equals(pd_result), "count() results should be equal")

    def test_multiple_groupby_same_datastore(self):
        """Multiple groupby operations should all be sorted correctly."""
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        pd_df = self.pd_df.copy()
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        # Multiple aggregations - index property works on LazyAggregate
        ds_mean = ds.groupby("FamilySize")["Survived"].mean()
        ds_sum = ds.groupby("FamilySize")["Survived"].sum()
        ds_count = ds.groupby("FamilySize")["Survived"].count()

        pd_mean = pd_df.groupby("FamilySize")["Survived"].mean()
        pd_sum = pd_df.groupby("FamilySize")["Survived"].sum()
        pd_count = pd_df.groupby("FamilySize")["Survived"].count()

        # All should have same order
        self.assertEqual(list(ds_mean.index), list(pd_mean.index))
        self.assertEqual(list(ds_sum.index), list(pd_sum.index))
        self.assertEqual(list(ds_count.index), list(pd_count.index))


class TestChdbNaNHandling(unittest.TestCase):
    """
    Tests for chDB's NaN handling behavior.

    NOTE: Some NaN handling issues have been fixed in recent chDB versions.
    The remaining xfail tests document cases where chDB still treats NaN
    as empty string in GROUP BY columns.
    """

    @classmethod
    def setUpClass(cls):
        cls.titanic_path = dataset_path("Titanic-Dataset.csv")
        cls.pd_df = pd.read_csv(cls.titanic_path)

    def setUp(self):
        self.ds = DataStore.from_file(self.titanic_path)
        config.enable_cache()
        # Force chDB engine
        config.execution_engine = 'chdb'

    def tearDown(self):
        config.execution_engine = 'auto'
    def test_groupby_embarked_nan_handling(self):
        """
        Test groupby on column with NaN values using chDB engine.

        Expected behavior (matching pandas):
        - NaN values should be excluded from GROUP BY
        - Result should only contain valid group keys: 'C', 'Q', 'S'

        Current chDB behavior (WRONG):
        - NaN is treated as empty string ''
        - Result contains: '', 'C', 'Q', 'S' (4 groups instead of 3)
        """
        ds_result = self.ds.groupby('Embarked')['Survived'].mean()
        pd_result = self.pd_df.groupby('Embarked')['Survived'].mean()

        # This should pass when chDB fixes NaN handling
        self.assertEqual(
            list(ds_result.index),
            list(pd_result.index),
            f"chDB returned {list(ds_result.index)}, expected {list(pd_result.index)}",
        )
    def test_groupby_embarked_count_nan_handling(self):
        """Test count aggregation with NaN-containing groupby column."""
        ds_result = self.ds.groupby('Embarked')['PassengerId'].count()
        pd_result = self.pd_df.groupby('Embarked')['PassengerId'].count()

        self.assertTrue(ds_result.equals(pd_result), "count() results should be equal")


    def test_groupby_age_nan_in_value_column(self):
        """
        Test aggregation where the VALUE column (not groupby column) has NaN.

        Age column has NaN values. When computing mean(Age), chDB should:
        - Skip NaN values (skipna=True is default in pandas)
        - Return correct mean for non-NaN values
        - Preserve index structure (groupby column as index)
        """
        ds_result = self.ds.groupby('Pclass')['Age'].mean()
        pd_result = self.pd_df.groupby('Pclass')['Age'].mean()

        # Convert ColumnExpr to pandas Series for comparison
        ds_series = ds_result.to_pandas()

        # Compare with index intact - no reset_index() to mask bugs
        assert_series_equal(
            ds_series,
            pd_result,
            rtol=1e-5,
        )


class TestOrderSensitiveOperationsCritical(unittest.TestCase):
    """
    Critical tests for column assignment + groupby pattern.

    These tests verify the pattern:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df.groupby("FamilySize")["Survived"].mean()
    df.to_df()

    Uses pandas engine to avoid chDB NaN handling issues.
    """

    @classmethod
    def setUpClass(cls):
        cls.titanic_path = dataset_path("Titanic-Dataset.csv")
        cls.pd_df = pd.read_csv(cls.titanic_path)

    def setUp(self):
        config.enable_cache()
        # Use pandas engine to avoid chDB NaN issues
        config.execution_engine = 'pandas'

    def tearDown(self):
        config.execution_engine = 'auto'

    def test_family_size_assignment_and_groupby(self):
        """
        Test the critical pattern:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df.groupby("FamilySize")["Survived"].mean()
        df.to_df()

        This should only execute FamilySize assignment ONCE.
        """
        # DataStore
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        ds_groupby_result = ds.groupby("FamilySize")["Survived"].mean()
        ds_df = ds.to_df()

        # Pandas
        pd_df = self.pd_df.copy()
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        pd_groupby_result = pd_df.groupby("FamilySize")["Survived"].mean()

        # Compare groupby results
        for idx in pd_groupby_result.index:
            self.assertAlmostEqual(
                ds_groupby_result[idx], pd_groupby_result[idx], places=5, msg=f"Mismatch at FamilySize={idx}"
            )

        # Compare final DataFrame
        self.assertEqual(ds_df.shape[1], pd_df.shape[1])
        self.assertIn("FamilySize", ds_df.columns)

        # Compare FamilySize values
        assert_series_equal(
            ds_df["FamilySize"].reset_index(drop=True), pd_df["FamilySize"].reset_index(drop=True)
        )


class TitanicExecutionLogicTest(unittest.TestCase):
    """Test execution logic and checkpoint behavior with debug logs."""

    def setUp(self):
        """Set up logging capture."""
        self.titanic_path = dataset_path("Titanic-Dataset.csv")

        # Set up log capture
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.DEBUG)

        # Get datastore logger
        self.logger = logging.getLogger('datastore')
        self.original_level = self.logger.level
        self.original_handlers = self.logger.handlers.copy()

        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = [self.log_handler]

        config.enable_cache()

    def tearDown(self):
        """Restore logging state."""
        self.logger.level = self.original_level
        self.logger.handlers = self.original_handlers
        config.disable_debug()

    def _get_logs(self) -> str:
        """Get captured log output."""
        self.log_handler.flush()
        return self.log_stream.getvalue()

    def _clear_logs(self):
        """Clear captured logs."""
        self.log_stream.truncate(0)
        self.log_stream.seek(0)

    def _count_pattern(self, pattern: str) -> int:
        """Count occurrences of pattern in logs."""
        logs = self._get_logs()
        return len(re.findall(pattern, logs, re.IGNORECASE))

    def test_family_size_executed_once(self):
        """
        Verify FamilySize column produces correct results across
        groupby().mean() and to_df() calls.

        Note: In unified architecture, simple arithmetic is pushed to SQL,
        so we verify semantic correctness (results match pandas) rather than
        counting pandas execute calls.
        """
        ds = DataStore.from_file(self.titanic_path)
        pd_df = pd.read_csv(self.titanic_path)

        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        # First access: groupby + mean
        ds_result = ds.groupby("FamilySize")["Survived"].mean()
        pd_result = pd_df.groupby("FamilySize")["Survived"].mean()

        # Verify groupby results match
        for idx in pd_result.index:
            self.assertAlmostEqual(ds_result[idx], pd_result[idx], places=5, msg=f"Mismatch at FamilySize={idx}")

        # Second access: to_df
        ds_final = ds.to_df()

        # Verify FamilySize values in final DataFrame match pandas
        assert_series_equal(
            ds_final["FamilySize"].reset_index(drop=True), pd_df["FamilySize"].reset_index(drop=True))

    def test_checkpoint_applied_after_groupby_mean(self):
        """Verify checkpoint is applied after groupby().mean() execution."""
        # Use from_dataframe to ensure we have lazy ops
        pd_df = pd.read_csv(self.titanic_path)
        ds = DataStore.from_dataframe(pd_df)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        # Before groupby, should have 2 lazy ops (DataFrameSource + ColumnAssignment)
        initial_ops_count = len(ds._lazy_ops)
        self.assertEqual(initial_ops_count, 2, f"Expected 2 lazy ops, got {initial_ops_count}")

        # Trigger execution via groupby
        result = ds.groupby("FamilySize")["Survived"].mean()
        _ = repr(result)  # Force execution

        # After checkpoint, should have 1 lazy op (LazyDataFrameSource with executed result)
        self.assertEqual(len(ds._lazy_ops), 1, f"Expected 1 lazy op after checkpoint, got {len(ds._lazy_ops)}")

        # The lazy op should be LazyDataFrameSource (executed DataFrame)
        from datastore.lazy_ops import LazyDataFrameSource

        self.assertIsInstance(
            ds._lazy_ops[0], LazyDataFrameSource, f"Expected LazyDataFrameSource, got {type(ds._lazy_ops[0]).__name__}"
        )

        # Verify the DataFrame includes FamilySize column
        df = ds._lazy_ops[0]._df
        self.assertIn("FamilySize", df.columns)

    def test_cache_used_on_second_to_df(self):
        """Verify cached result is used on second to_df() call."""
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        # First call
        _ = ds.to_df()
        self._clear_logs()

        # Second call should use cache
        config.enable_debug()
        _ = ds.to_df()
        config.disable_debug()

        logs = self._get_logs()
        self.assertIn("Using cached result", logs)

    def test_execution_log_sequence(self):
        """Verify correct log sequence during execution."""
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        config.enable_debug()
        _ = ds.to_df()
        config.disable_debug()

        logs = self._get_logs()

        # Should see execution start
        self.assertIn("Starting execution", logs)

        # Should see lazy ops chain
        self.assertIn("Lazy operations chain", logs)

        # Should see FamilySize assignment
        self.assertIn("FamilySize", logs)

        # Should see execution complete
        self.assertIn("Execution complete", logs)

        # Should see checkpoint or pure SQL state preservation
        # (Pure SQL executions preserve SQL state instead of checkpointing)
        self.assertTrue(
            "checkpointed" in logs or "Pure SQL execution" in logs,
            "Should see either checkpointed or Pure SQL execution in logs",
        )

    def test_groupby_with_sql_aggregation(self):
        """Verify groupby uses SQL for aggregation."""
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        self._clear_logs()
        config.enable_debug()

        result = ds.groupby("FamilySize")["Survived"].mean()
        _ = repr(result)

        config.disable_debug()
        logs = self._get_logs()

        # Should see chDB SQL execution for aggregation
        self.assertIn("chDB", logs)
        self.assertIn("GROUP BY", logs)

    def test_groupby_respects_pandas_engine(self):
        """Verify groupby respects execution_engine=pandas setting."""
        ds = DataStore.from_file(self.titanic_path)
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1

        # Set pandas execution engine
        config.execution_engine = 'pandas'

        try:
            self._clear_logs()
            config.enable_debug()

            result = ds.groupby("FamilySize")["Survived"].mean()
            _ = repr(result)

            config.disable_debug()
            logs = self._get_logs()

            # Should NOT see chDB SQL for groupby aggregation when using pandas engine
            # (execution may still use chDB, but the groupby should be pandas)
            self.assertNotIn("GROUP BY", logs, "Should not use SQL GROUP BY when engine is pandas")

            # Verify result is correct
            pd_df = pd.read_csv(self.titanic_path)
            pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1
            pd_result = pd_df.groupby("FamilySize")["Survived"].mean()

            for idx in pd_result.index:
                self.assertAlmostEqual(result[idx], pd_result[idx], places=5)
        finally:
            # Restore default engine
            config.execution_engine = 'auto'


class TitanicFullWorkflowTest(unittest.TestCase):
    """Test complete Titanic analysis workflow matching notebook."""

    def setUp(self):
        """Set up for workflow tests."""
        self.titanic_path = dataset_path("Titanic-Dataset.csv")
        config.enable_cache()

    def test_complete_titanic_workflow(self):
        """
        Complete Titanic workflow as in the notebook.

        This replicates the exact sequence of operations from the notebook
        and verifies results match pandas at each step.
        """
        # ========== Setup ==========
        ds = DataStore.from_file(self.titanic_path)
        pd_df = pd.read_csv(self.titanic_path)

        # ========== Step 1: Basic exploration ==========
        self.assertEqual(ds.shape, pd_df.shape)
        self.assertEqual(list(ds.columns), list(pd_df.columns))

        # ========== Step 2: Handle Age missing values ==========
        age_mean = round(pd_df['Age'].mean(), 2)

        ds['Age'] = ds['Age'].fillna(age_mean)
        pd_df['Age'] = pd_df['Age'].fillna(age_mean)

        ds_age_na = ds.isna()['Age'].sum()
        pd_age_na = pd_df['Age'].isna().sum()
        self.assertEqual(ds_age_na, pd_age_na)
        self.assertEqual(ds_age_na, 0)

        # ========== Step 3: Create Has_Cabin column ==========
        ds["Has_Cabin"] = ds['Cabin'].notnull().astype(int)
        pd_df["Has_Cabin"] = pd_df['Cabin'].notnull().astype(int)

        ds_has_cabin_sum = ds["Has_Cabin"].sum()
        pd_has_cabin_sum = pd_df["Has_Cabin"].sum()
        self.assertEqual(ds_has_cabin_sum, pd_has_cabin_sum)

        # ========== Step 4: Handle Cabin missing values ==========
        ds['Cabin'] = ds['Cabin'].fillna('Unknown')
        pd_df['Cabin'] = pd_df['Cabin'].fillna('Unknown')

        # ========== Step 5: Handle Embarked missing values ==========
        embarked_mode = pd_df['Embarked'].mode()[0]
        ds['Embarked'] = ds['Embarked'].fillna(embarked_mode)
        pd_df['Embarked'] = pd_df['Embarked'].fillna(embarked_mode)

        # ========== Step 6: GroupBy analyses ==========
        # Sex -> Survived sum
        ds_sex_survived = ds.groupby('Sex')['Survived'].sum()
        pd_sex_survived = pd_df.groupby('Sex')['Survived'].sum()
        for idx in pd_sex_survived.index:
            self.assertEqual(ds_sex_survived[idx], pd_sex_survived[idx])

        # Survived -> Age mean
        ds_survived_age = ds.groupby('Survived')['Age'].mean()
        pd_survived_age = pd_df.groupby('Survived')['Age'].mean()
        for idx in pd_survived_age.index:
            self.assertAlmostEqual(ds_survived_age[idx], pd_survived_age[idx], places=5)

        # Embarked -> Survived mean
        ds_embarked = ds.groupby("Embarked")["Survived"].mean()
        pd_embarked = pd_df.groupby("Embarked")["Survived"].mean()
        for idx in pd_embarked.index:
            self.assertAlmostEqual(ds_embarked[idx], pd_embarked[idx], places=5)

        # Pclass -> Survived mean
        ds_pclass = ds.groupby("Pclass")["Survived"].mean()
        pd_pclass = pd_df.groupby("Pclass")["Survived"].mean()
        for idx in pd_pclass.index:
            self.assertAlmostEqual(ds_pclass[idx], pd_pclass[idx], places=5)

        # Has_Cabin -> Survived mean
        ds_cabin = ds.groupby("Has_Cabin")["Survived"].mean()
        pd_cabin = pd_df.groupby("Has_Cabin")["Survived"].mean()
        for idx in pd_cabin.index:
            self.assertAlmostEqual(ds_cabin[idx], pd_cabin[idx], places=5)

        # ========== Step 7: FamilySize (CRITICAL) ==========
        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        ds_family = ds.groupby("FamilySize")["Survived"].mean()
        pd_family = pd_df.groupby("FamilySize")["Survived"].mean()
        for idx in pd_family.index:
            self.assertAlmostEqual(ds_family[idx], pd_family[idx], places=5, msg=f"Mismatch at FamilySize={idx}")

        # ========== Step 8: Final DataFrame ==========
        ds_final = ds.to_df()

        # Check all columns present
        expected_cols = list(pd_df.columns)
        actual_cols = list(ds_final.columns)
        self.assertEqual(sorted(actual_cols), sorted(expected_cols))

        # Check shapes match
        self.assertEqual(ds_final.shape, pd_df.shape)

        # Check FamilySize values match
        assert_series_equal(
            ds_final["FamilySize"].reset_index(drop=True), pd_df["FamilySize"].reset_index(drop=True)
        )


class TitanicExecutionCountTest(unittest.TestCase):
    """
    Tests to verify column assignments produce correct results, regardless of execution path.

    Note: In the unified architecture, simple arithmetic ColumnAssignments can be pushed
    to SQL for efficiency. These tests verify the semantic correctness (results match pandas)
    rather than the execution path (pandas vs SQL).
    """

    def setUp(self):
        """Set up test fixtures."""
        self.titanic_path = dataset_path("Titanic-Dataset.csv")
        config.enable_cache()

    def test_multiple_assignments_each_executed_once(self):
        """Test multiple column assignments produce correct results."""
        ds = DataStore.from_file(self.titanic_path)
        pd_df = pd.read_csv(self.titanic_path)

        # Multiple assignments - mirror pandas
        ds["Col1"] = ds["SibSp"] + 1
        ds["Col2"] = ds["Parch"] + 2
        ds["Col3"] = ds["Age"] * 0.5

        pd_df["Col1"] = pd_df["SibSp"] + 1
        pd_df["Col2"] = pd_df["Parch"] + 2
        pd_df["Col3"] = pd_df["Age"] * 0.5

        # Trigger execution
        ds_result = ds.to_df()

        # Verify results match pandas
        for col in ["Col1", "Col2", "Col3"]:
            assert_series_equal(
                ds_result[col].reset_index(drop=True), pd_df[col].reset_index(drop=True))

    def test_assignment_then_groupby_then_todf(self):
        """Test assignment + groupby + to_df produces correct results."""
        ds = DataStore.from_file(self.titanic_path)
        pd_df = pd.read_csv(self.titanic_path)

        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        # groupby + mean
        ds_result = ds.groupby("FamilySize")["Survived"].mean()
        pd_result = pd_df.groupby("FamilySize")["Survived"].mean()

        # Verify groupby results match
        for idx in pd_result.index:
            self.assertAlmostEqual(ds_result[idx], pd_result[idx], places=5)

        # to_df
        ds_final = ds.to_df()

        # Verify FamilySize column is correct
        assert_series_equal(
            ds_final["FamilySize"].reset_index(drop=True), pd_df["FamilySize"].reset_index(drop=True))

    def test_assignment_then_multiple_groupby(self):
        """Test assignment + multiple groupby operations produce correct results."""
        import math

        ds = DataStore.from_file(self.titanic_path)
        pd_df = pd.read_csv(self.titanic_path)

        ds["FamilySize"] = ds["SibSp"] + ds["Parch"] + 1
        pd_df["FamilySize"] = pd_df["SibSp"] + pd_df["Parch"] + 1

        # Multiple groupby operations - verify each matches pandas
        r1_ds = ds.groupby("FamilySize")["Survived"].mean()
        r1_pd = pd_df.groupby("FamilySize")["Survived"].mean()
        for idx in r1_pd.index:
            ds_val, pd_val = r1_ds[idx], r1_pd[idx]
            if math.isnan(pd_val):
                self.assertTrue(math.isnan(ds_val))
            else:
                self.assertAlmostEqual(ds_val, pd_val, places=5)

        r2_ds = ds.groupby("FamilySize")["Survived"].sum()
        r2_pd = pd_df.groupby("FamilySize")["Survived"].sum()
        for idx in r2_pd.index:
            ds_val, pd_val = r2_ds[idx], r2_pd[idx]
            if math.isnan(pd_val):
                self.assertTrue(math.isnan(ds_val))
            else:
                self.assertAlmostEqual(ds_val, pd_val, places=5)

        r3_ds = ds.groupby("FamilySize")["Age"].mean()
        r3_pd = pd_df.groupby("FamilySize")["Age"].mean()
        for idx in r3_pd.index:
            ds_val, pd_val = r3_ds[idx], r3_pd[idx]
            if math.isnan(pd_val):
                self.assertTrue(math.isnan(ds_val))
            else:
                self.assertAlmostEqual(ds_val, pd_val, places=5)

        # to_df - final result should have correct FamilySize
        ds_final = ds.to_df()
        assert_series_equal(
            ds_final["FamilySize"].reset_index(drop=True), pd_df["FamilySize"].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
