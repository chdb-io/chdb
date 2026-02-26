"""
Tests for groupby with column selection.

These tests verify that df[['col1', 'col2']].groupby('col1').agg() correctly
respects the column selection and only aggregates the selected columns.

This was a bug where column selection via df[['col1', 'col2']] was being ignored
during groupby aggregation, causing SQL to try to aggregate ALL columns from
the source (including string columns), which failed with:
  "Illegal type String of argument for aggregate function avg"

See: GitHub issue for reference
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestGroupByColumnSelection:
    """Tests for groupby with explicit column selection."""

    @pytest.fixture
    def mixed_type_data(self):
        """Create test data with mixed types (numeric and string columns)."""
        return {
            'Pclass': [1, 1, 2, 2, 3, 3],
            'Survived': [1, 0, 1, 0, 0, 0],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
            'Sex': ['female', 'male', 'male', 'male', 'female', 'male'],
            'Age': [30.0, 40.0, 25.0, 35.0, 20.0, 45.0],
        }

    def test_column_selection_with_groupby_mean(self, mixed_type_data):
        """
        Test that df[['col1', 'col2']].groupby('col1').mean() only aggregates selected columns.

        This was the original bug: without proper column selection propagation,
        mean() would try to aggregate ALL columns including strings, causing:
        "Illegal type String of argument for aggregate function avg"
        """
        pd_df = pd.DataFrame(mixed_type_data)
        ds = DataStore(mixed_type_data)

        # pandas operation
        pd_result = pd_df[['Pclass', 'Survived']].groupby(['Pclass']).mean()

        # DataStore operation (should not fail with string aggregation error)
        ds_result = ds[['Pclass', 'Survived']].groupby(['Pclass']).mean()

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_with_groupby_mean_as_index_false(self, mixed_type_data):
        """Test column selection with as_index=False."""
        pd_df = pd.DataFrame(mixed_type_data)
        ds = DataStore(mixed_type_data)

        # pandas operation
        pd_result = pd_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

        # DataStore operation
        ds_result = ds[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

        # Compare results (check_row_order=False because groupby order may differ)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_column_selection_with_groupby_sum(self, mixed_type_data):
        """Test that column selection works with sum() aggregation."""
        pd_df = pd.DataFrame(mixed_type_data)
        ds = DataStore(mixed_type_data)

        # pandas operation
        pd_result = pd_df[['Pclass', 'Age']].groupby(['Pclass']).sum()

        # DataStore operation
        ds_result = ds[['Pclass', 'Age']].groupby(['Pclass']).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_with_groupby_count(self, mixed_type_data):
        """Test that column selection works with count() aggregation."""
        pd_df = pd.DataFrame(mixed_type_data)
        ds = DataStore(mixed_type_data)

        # pandas operation
        pd_result = pd_df[['Pclass', 'Survived']].groupby(['Pclass']).count()

        # DataStore operation
        ds_result = ds[['Pclass', 'Survived']].groupby(['Pclass']).count()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_with_multiple_numeric_columns(self, mixed_type_data):
        """Test selecting multiple numeric columns for aggregation."""
        pd_df = pd.DataFrame(mixed_type_data)
        ds = DataStore(mixed_type_data)

        # Select multiple numeric columns
        pd_result = pd_df[['Pclass', 'Survived', 'Age']].groupby(['Pclass']).mean()
        ds_result = ds[['Pclass', 'Survived', 'Age']].groupby(['Pclass']).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_with_sort_values(self, mixed_type_data):
        """
        Test the full pattern from the original bug report:
        df[['col1', 'col2']].groupby(['col1'], as_index=False).mean().sort_values(by='col2', ascending=False)
        """
        pd_df = pd.DataFrame(mixed_type_data)
        ds = DataStore(mixed_type_data)

        # Original bug pattern
        pd_result = (
            pd_df[['Pclass', 'Survived']]
            .groupby(['Pclass'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        ds_result = (
            ds[['Pclass', 'Survived']]
            .groupby(['Pclass'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        # Note: row order for ties may differ, but values should match
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestGroupByColumnSelectionFromFile:
    """Tests for groupby with column selection when reading from file."""

    @pytest.fixture
    def csv_with_mixed_types(self, tmp_path):
        """Create a CSV file with mixed types (mimicking Titanic dataset)."""
        data = {
            'PassengerId': [1, 2, 3, 4, 5, 6],
            'Pclass': [1, 1, 2, 2, 3, 3],
            'Survived': [1, 0, 1, 0, 0, 0],
            'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
            'Sex': ['female', 'male', 'male', 'male', 'female', 'male'],
            'Age': [30.0, 40.0, 25.0, 35.0, 20.0, 45.0],
            'SibSp': [0, 1, 0, 1, 2, 0],
            'Parch': [0, 0, 0, 1, 1, 0],
            'Ticket': ['A123', 'B456', 'C789', 'D012', 'E345', 'F678'],
            'Fare': [100.0, 200.0, 50.0, 75.0, 25.0, 30.0],
            'Cabin': ['C1', 'C2', '', '', '', ''],
            'Embarked': ['S', 'C', 'S', 'Q', 'S', 'C'],
        }
        pd_df = pd.DataFrame(data)
        csv_path = tmp_path / "train.csv"
        pd_df.to_csv(csv_path, index=False)
        return csv_path, pd_df

    def test_file_based_column_selection_groupby_mean(self, csv_with_mixed_types):
        """
        Test the original bug scenario with file-based DataStore.

        Original error:
        Query execution failed: Code: 43. DB::Exception: Illegal type String of
        argument for aggregate function avg. (ILLEGAL_TYPE_OF_ARGUMENT)
        SQL: SELECT "Pclass", avg("PassengerId") AS "PassengerId", avg("Survived") AS "Survived",
             avg("Name") AS "Name", avg("Sex") AS "Sex", ...
        """
        csv_path, pd_df = csv_with_mixed_types

        ds = DataStore.from_file(str(csv_path))

        # pandas operation
        pd_result = pd_df[['Pclass', 'Survived']].groupby(['Pclass']).mean()

        # DataStore operation - this should NOT fail with string aggregation error
        ds_result = ds[['Pclass', 'Survived']].groupby(['Pclass']).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_file_based_full_pattern(self, csv_with_mixed_types):
        """
        Test the exact pattern from the bug report with file-based source.

        Note: as_index=False is not fully supported for file-based sources yet.
        The main fix here is that column selection is respected (no string aggregation error).
        We test without as_index=False to verify the core fix works.
        """
        csv_path, pd_df = csv_with_mixed_types

        ds = DataStore.from_file(str(csv_path))

        # Test without as_index=False (core fix verification)
        pd_result = (
            pd_df[['Pclass', 'Survived']]
            .groupby(['Pclass'])  # as_index=True (default)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        ds_result = (
            ds[['Pclass', 'Survived']]
            .groupby(['Pclass'])  # as_index=True (default)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_file_based_full_pattern_with_as_index_false(self, csv_with_mixed_types):
        """Test as_index=False with file-based source (known limitation)."""
        csv_path, pd_df = csv_with_mixed_types

        ds = DataStore.from_file(str(csv_path))

        pd_result = (
            pd_df[['Pclass', 'Survived']]
            .groupby(['Pclass'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        ds_result = (
            ds[['Pclass', 'Survived']]
            .groupby(['Pclass'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestSelectedColumnsPassthrough:
    """Tests to verify that selected_columns is correctly passed through the lazy op chain."""

    def test_selected_columns_extracted_from_lazy_ops(self):
        """Verify that groupby() correctly extracts selected_columns from prior LazyRelationalOp."""
        from datastore.lazy_ops import LazyRelationalOp

        data = {'a': [1, 1, 2], 'b': [10, 20, 30], 'c': ['x', 'y', 'z']}
        ds = DataStore(data)

        # Select specific columns
        selected = ds[['a', 'b']]

        # Verify LazyRelationalOp(SELECT) was created
        select_ops = [op for op in selected._lazy_ops if isinstance(op, LazyRelationalOp) and op.op_type == 'SELECT']
        assert len(select_ops) == 1
        assert len(select_ops[0].fields) == 2

        # Create groupby
        grouped = selected.groupby(['a'])

        # Verify selected_columns was extracted
        assert grouped._selected_columns is not None
        assert 'a' in grouped._selected_columns
        assert 'b' in grouped._selected_columns
        assert 'c' not in grouped._selected_columns

    def test_selected_columns_passed_to_lazy_groupby_agg(self):
        """Verify that selected_columns is passed to LazyGroupByAgg."""
        from datastore.lazy_ops import LazyGroupByAgg

        data = {'a': [1, 1, 2], 'b': [10, 20, 30], 'c': ['x', 'y', 'z']}
        ds = DataStore(data)

        # Apply the full pattern
        result = ds[['a', 'b']].groupby(['a']).mean()

        # Find LazyGroupByAgg in lazy ops
        groupby_agg_ops = [op for op in result._lazy_ops if isinstance(op, LazyGroupByAgg)]
        assert len(groupby_agg_ops) == 1

        # Verify selected_columns is set
        agg_op = groupby_agg_ops[0]
        assert agg_op.selected_columns is not None
        assert 'a' in agg_op.selected_columns
        assert 'b' in agg_op.selected_columns
        assert 'c' not in agg_op.selected_columns

    def test_groupby_cols_filtered_from_aggregation(self):
        """Verify that groupby columns are not aggregated (only non-groupby selected columns)."""
        data = {'a': [1, 1, 2], 'b': [10, 20, 30], 'c': ['x', 'y', 'z']}

        pd_df = pd.DataFrame(data)
        ds = DataStore(data)

        # When grouping by 'a' and selecting ['a', 'b'], only 'b' should be aggregated
        pd_result = pd_df[['a', 'b']].groupby(['a']).mean()
        ds_result = ds[['a', 'b']].groupby(['a']).mean()

        # Result should have 'a' as index and 'b' as the only data column
        assert_datastore_equals_pandas(ds_result, pd_result)

        # Verify 'b' is in the columns (not aggregating 'a')
        ds_columns = ds_result.columns.tolist()
        assert 'b' in ds_columns


class TestEdgeCases:
    """Edge cases for column selection with groupby."""

    def test_select_only_groupby_column(self):
        """Test selecting only the groupby column (no columns to aggregate)."""
        data = {'a': [1, 1, 2], 'b': [10, 20, 30]}

        pd_df = pd.DataFrame(data)
        ds = DataStore(data)

        # Select only the groupby column
        pd_result = pd_df[['a']].groupby(['a']).mean()
        ds_result = ds[['a']].groupby(['a']).mean()

        # Result should be empty (no columns to aggregate)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_all_columns_explicitly(self):
        """Test explicitly selecting all columns (should behave like no selection)."""
        data = {'a': [1, 1, 2], 'b': [10, 20, 30], 'c': [100, 200, 300]}

        pd_df = pd.DataFrame(data)
        ds = DataStore(data)

        # Select all columns explicitly
        pd_result = pd_df[['a', 'b', 'c']].groupby(['a']).mean()
        ds_result = ds[['a', 'b', 'c']].groupby(['a']).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_preserved_through_filter(self):
        """Test that column selection is preserved when combined with filter."""
        data = {'a': [1, 1, 2, 2, 3], 'b': [10, 20, 30, 40, 50], 'c': ['x', 'y', 'z', 'w', 'v']}

        pd_df = pd.DataFrame(data)
        ds = DataStore(data)

        # Filter then select then groupby
        pd_result = pd_df[pd_df['a'] > 1][['a', 'b']].groupby(['a']).mean()
        ds_result = ds[ds['a'] > 1][['a', 'b']].groupby(['a']).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_with_multiple_groupby_columns(self):
        """Test column selection with multiple groupby columns."""
        data = {
            'a': [1, 1, 2, 2],
            'b': ['x', 'y', 'x', 'y'],
            'c': [10, 20, 30, 40],
            'd': ['str1', 'str2', 'str3', 'str4'],
        }

        pd_df = pd.DataFrame(data)
        ds = DataStore(data)

        # Select columns for groupby with multiple keys
        pd_result = pd_df[['a', 'b', 'c']].groupby(['a', 'b']).mean()
        ds_result = ds[['a', 'b', 'c']].groupby(['a', 'b']).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)
