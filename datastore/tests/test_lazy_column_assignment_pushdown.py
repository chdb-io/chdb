"""
Unit tests for LazyColumnAssignment SQL Pushdown.

These tests verify the new subquery wrapping architecture for computed columns.
The key scenarios tested:

1. Simple computed column pushdown
2. Chained computed columns (auto subquery wrapping)
3. Column override (EXCEPT syntax for known columns)
4. Filter on computed column (subquery required)
5. Order by computed column
6. Combined operations (filter + sort + limit + computed)

Test Philosophy:
- Mirror code pattern: DataStore ops mirror pandas ops
- Complete output comparison: columns + values + order
- Verify SQL behavior matches pandas semantics
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestSimpleComputedColumn(unittest.TestCase):
    """Test basic computed column SQL pushdown."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        # Create test data
        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A'],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_simple_arithmetic_computed_column(self):
        """Simple computed column: ds['doubled'] = ds['value'] * 2"""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)

    def test_computed_column_with_multiple_columns(self):
        """Computed column using multiple source columns."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['combined'] = pd_df['id'] + pd_df['value']

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['combined'] = ds['id'] + ds['value']

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)

    def test_multiple_independent_computed_columns(self):
        """Multiple computed columns that don't reference each other."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df['halved'] = pd_df['value'] / 2

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds['halved'] = ds['value'] / 2

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)


class TestChainedComputedColumns(unittest.TestCase):
    """Test chained computed columns requiring subquery wrapping."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_chained_two_columns(self):
        """
        Chained columns: step2 references step1.
        ds['step1'] = ds['value'] * 2
        ds['step2'] = ds['step1'] + 10
        """
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['step1'] = pd_df['value'] * 2
        pd_df['step2'] = pd_df['step1'] + 10

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['step1'] = ds['value'] * 2
        ds['step2'] = ds['step1'] + 10

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)

    def test_chained_three_columns(self):
        """
        Three-level chain:
        ds['step1'] = ds['value'] * 2
        ds['step2'] = ds['step1'] + 10
        ds['step3'] = ds['step2'] * 2
        """
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['step1'] = pd_df['value'] * 2
        pd_df['step2'] = pd_df['step1'] + 10
        pd_df['step3'] = pd_df['step2'] * 2

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['step1'] = ds['value'] * 2
        ds['step2'] = ds['step1'] + 10
        ds['step3'] = ds['step2'] * 2

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)


class TestColumnOverride(unittest.TestCase):
    """Test column override scenarios using EXCEPT syntax."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_single_override(self):
        """Override an existing column once."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['value'] = pd_df['value'] * 2

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['value'] = ds['value'] * 2

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)

    def test_multiple_overrides(self):
        """Override the same column multiple times."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['value'] = pd_df['value'] * 2
        pd_df['value'] = pd_df['value'] + 10
        pd_df['value'] = pd_df['value'] * 3

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['value'] = ds['value'] * 2
        ds['value'] = ds['value'] + 10
        ds['value'] = ds['value'] * 3

        # Compare
        assert_datastore_equals_pandas(ds, pd_df)


class TestFilterOnComputedColumn(unittest.TestCase):
    """Test filtering on computed columns."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_filter_on_computed(self):
        """Filter on a computed column."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df[pd_df['doubled'] > 60]

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] > 60)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))

    def test_filter_on_chained_computed(self):
        """Filter on a chained computed column."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['step1'] = pd_df['value'] * 2
        pd_df['step2'] = pd_df['step1'] + 10
        pd_df = pd_df[pd_df['step2'] > 70]

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['step1'] = ds['value'] * 2
        ds['step2'] = ds['step1'] + 10
        ds = ds.filter(ds['step2'] > 70)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))


class TestSortOnComputedColumn(unittest.TestCase):
    """Test sorting on computed columns."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'value': [30, 10, 50, 20, 40],  # Not sorted
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_sort_on_computed_asc(self):
        """Sort ascending on computed column."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df.sort_values('doubled', ascending=True)

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.sort_values('doubled', ascending=True)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))

    def test_sort_on_computed_desc(self):
        """Sort descending on computed column."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df.sort_values('doubled', ascending=False)

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.sort_values('doubled', ascending=False)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))


class TestCombinedOperations(unittest.TestCase):
    """Test combinations of computed columns with filter, sort, limit."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        cls.data = {
            'id': list(range(1, 11)),
            'value': list(range(10, 110, 10)),  # 10, 20, ..., 100
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_computed_filter_sort_limit(self):
        """Computed column + filter + sort + limit."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df[pd_df['doubled'] > 60]
        pd_df = pd_df.sort_values('doubled', ascending=False)
        pd_df = pd_df.head(3)

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] > 60)
        ds = ds.sort_values('doubled', ascending=False)
        ds = ds.head(3)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))

    def test_filter_computed_filter_on_computed(self):
        """Filter -> computed -> filter on computed."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df = pd_df[pd_df['value'] > 30]
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df[pd_df['doubled'] > 100]

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds = ds.filter(ds['value'] > 30)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] > 100)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.tmpdir, 'test_data.csv')
        
        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    def test_empty_result_after_filter(self):
        """Filter that results in empty DataFrame."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df[pd_df['doubled'] > 1000]

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] > 1000)

        # Compare (both should be empty)
        ds_result = ds.to_df()
        self.assertEqual(len(ds_result), 0)
        self.assertEqual(len(pd_df), 0)

    def test_all_rows_match_filter(self):
        """Filter that matches all rows."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df[pd_df['doubled'] > 0]

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] > 0)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))

    def test_single_row_result(self):
        """Operation resulting in single row."""
        # Pandas reference
        pd_df = pd.DataFrame(self.data)
        pd_df['doubled'] = pd_df['value'] * 2
        pd_df = pd_df[pd_df['doubled'] == 60]

        # DataStore operations
        ds = DataStore.from_file(self.csv_file)
        ds['doubled'] = ds['value'] * 2
        ds = ds.filter(ds['doubled'] == 60)

        # Compare
        assert_datastore_equals_pandas(ds, pd_df.reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()

