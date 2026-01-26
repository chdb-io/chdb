"""
Test suite to verify DataStore aligns with pandas immutability behavior.

This test ensures that DataStore operations behave like pandas:
1. Most operations return NEW objects (not the same object)
2. Most operations do NOT modify the original object
3. Only explicit in-place operations (like __setitem__) modify the original

Run with: pytest tests/test_pandas_immutability_alignment.py -v
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from datastore import DataStore
from tests.test_utils import assert_frame_equal


class TestPandasImmutabilityAlignment(unittest.TestCase):
    """
    Verify DataStore matches pandas immutability semantics.

    In pandas, most operations return new objects and don't modify originals.
    DataStore should behave the same way.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        # Create test data
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'city': ['NYC', 'LA', 'NYC', 'LA', 'NYC'],
        }
        cls.test_df = pd.DataFrame(data)
        cls.test_df.to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def _get_fresh_ds(self):
        """Get a fresh DataStore instance."""
        return DataStore.from_file(self.csv_file)

    def _get_fresh_df(self):
        """Get a fresh pandas DataFrame."""
        return self.test_df.copy()

    # ========== Column Transformation Methods ==========

    def test_add_prefix_returns_new_object(self):
        """add_prefix should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.add_prefix('x_')
        self.assertIsNot(df, df_result, "Pandas: add_prefix should return new object")
        self.assertListEqual(
            list(df.columns), ['id', 'name', 'age', 'salary', 'city'], "Pandas: original should be unchanged"
        )

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.add_prefix('x_')
        self.assertIsNot(ds, ds_result, "DataStore: add_prefix should return new object")

    def test_add_suffix_returns_new_object(self):
        """add_suffix should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.add_suffix('_y')
        self.assertIsNot(df, df_result, "Pandas: add_suffix should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.add_suffix('_y')
        self.assertIsNot(ds, ds_result, "DataStore: add_suffix should return new object")

    def test_rename_returns_new_object(self):
        """rename should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.rename(columns={'name': 'full_name'})
        self.assertIsNot(df, df_result, "Pandas: rename should return new object")
        self.assertIn('name', df.columns, "Pandas: original should be unchanged")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.rename(columns={'name': 'full_name'})
        self.assertIsNot(ds, ds_result, "DataStore: rename should return new object")

    # ========== Filtering Methods ==========

    def test_filter_returns_new_object(self):
        """filter (column selection) should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.filter(items=['id', 'name'])
        self.assertIsNot(df, df_result, "Pandas: filter should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.filter(items=['id', 'name'])
        self.assertIsNot(ds, ds_result, "DataStore: filter should return new object")

    def test_boolean_indexing_returns_new_object(self):
        """Boolean indexing should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df[df['age'] > 30]
        self.assertIsNot(df, df_result, "Pandas: boolean indexing should return new object")
        self.assertEqual(len(df), 5, "Pandas: original should be unchanged")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds[ds['age'] > 30]
        self.assertIsNot(ds, ds_result, "DataStore: boolean indexing should return new object")

    def test_query_returns_new_object(self):
        """query should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.query('age > 30')
        self.assertIsNot(df, df_result, "Pandas: query should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.query('age > 30')
        self.assertIsNot(ds, ds_result, "DataStore: query should return new object")

    # ========== Selection Methods ==========

    def test_column_selection_returns_new_object(self):
        """Column selection df[['col1', 'col2']] should return new object."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df[['id', 'name']]
        self.assertIsNot(df, df_result, "Pandas: column selection should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds[['id', 'name']]
        self.assertIsNot(ds, ds_result, "DataStore: column selection should return new object")

    def test_head_returns_new_object(self):
        """head should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.head(3)
        self.assertIsNot(df, df_result, "Pandas: head should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.head(3)
        self.assertIsNot(ds, ds_result, "DataStore: head should return new object")

    def test_tail_returns_new_object(self):
        """tail should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.tail(3)
        self.assertIsNot(df, df_result, "Pandas: tail should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.tail(3)
        self.assertIsNot(ds, ds_result, "DataStore: tail should return new object")

    def test_slice_returns_new_object(self):
        """Slice notation df[:5] should return new object."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df[:3]
        self.assertIsNot(df, df_result, "Pandas: slice should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds[:3]
        self.assertIsNot(ds, ds_result, "DataStore: slice should return new object")

    # ========== Sorting Methods ==========

    def test_sort_values_returns_new_object(self):
        """sort_values should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.sort_values('age')
        self.assertIsNot(df, df_result, "Pandas: sort_values should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.sort_values('age')
        self.assertIsNot(ds, ds_result, "DataStore: sort_values should return new object")

    def test_sort_index_returns_new_object(self):
        """sort_index should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.sort_index()
        self.assertIsNot(df, df_result, "Pandas: sort_index should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.sort_index()
        self.assertIsNot(ds, ds_result, "DataStore: sort_index should return new object")

    # ========== Column Assignment ==========

    def test_assign_returns_new_object(self):
        """assign should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.assign(new_col=lambda x: x['age'] * 2)
        self.assertIsNot(df, df_result, "Pandas: assign should return new object")
        self.assertNotIn('new_col', df.columns, "Pandas: original should be unchanged")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.assign(new_col=lambda x: x['age'] * 2)
        self.assertIsNot(ds, ds_result, "DataStore: assign should return new object")

    # ========== Missing Value Methods ==========

    def test_dropna_returns_new_object(self):
        """dropna should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.dropna()
        self.assertIsNot(df, df_result, "Pandas: dropna should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.dropna()
        self.assertIsNot(ds, ds_result, "DataStore: dropna should return new object")

    def test_fillna_returns_new_object(self):
        """fillna should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.fillna(0)
        self.assertIsNot(df, df_result, "Pandas: fillna should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.fillna(0)
        self.assertIsNot(ds, ds_result, "DataStore: fillna should return new object")

    # ========== Duplicate Methods ==========

    def test_drop_duplicates_returns_new_object(self):
        """drop_duplicates should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.drop_duplicates()
        self.assertIsNot(df, df_result, "Pandas: drop_duplicates should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.drop_duplicates()
        self.assertIsNot(ds, ds_result, "DataStore: drop_duplicates should return new object")

    # ========== Column Drop Methods ==========

    def test_drop_returns_new_object(self):
        """drop should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.drop(columns=['age'])
        self.assertIsNot(df, df_result, "Pandas: drop should return new object")
        self.assertIn('age', df.columns, "Pandas: original should be unchanged")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.drop(columns=['age'])
        self.assertIsNot(ds, ds_result, "DataStore: drop should return new object")

    # ========== Type Conversion Methods ==========

    def test_astype_returns_new_object(self):
        """astype should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.astype({'age': float})
        self.assertIsNot(df, df_result, "Pandas: astype should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.astype({'age': float})
        self.assertIsNot(ds, ds_result, "DataStore: astype should return new object")

    # ========== Conditional Methods ==========

    def test_where_returns_new_object(self):
        """where (pandas-style with replacement) should return new object."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.where(df['age'] > 30, 0)
        self.assertIsNot(df, df_result, "Pandas: where should return new object")

        # DataStore behavior - ColumnExpr conditions are auto-executed
        ds = self._get_fresh_ds()
        ds_result = ds.where(ds['age'] > 30, 0)  # ds['age'] > 30 is ColumnExpr, auto-executed
        self.assertIsNot(ds, ds_result, "DataStore: where should return new object")

    def test_mask_returns_new_object(self):
        """mask should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.mask(df['age'] > 30, 0)
        self.assertIsNot(df, df_result, "Pandas: mask should return new object")

        # DataStore behavior - ColumnExpr conditions are auto-executed
        ds = self._get_fresh_ds()
        ds_result = ds.mask(ds['age'] > 30, 0)  # ds['age'] > 30 is ColumnExpr, auto-executed
        self.assertIsNot(ds, ds_result, "DataStore: mask should return new object")

    # ========== Copy Method ==========

    def test_copy_returns_new_object(self):
        """copy should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.copy()
        self.assertIsNot(df, df_result, "Pandas: copy should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.copy()
        self.assertIsNot(ds, ds_result, "DataStore: copy should return new object")

    # ========== Reset Index ==========

    def test_reset_index_returns_new_object(self):
        """reset_index should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.reset_index()
        self.assertIsNot(df, df_result, "Pandas: reset_index should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.reset_index()
        self.assertIsNot(ds, ds_result, "DataStore: reset_index should return new object")

    # ========== Sample Method ==========

    def test_sample_returns_new_object(self):
        """sample should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.sample(n=2, random_state=42)
        self.assertIsNot(df, df_result, "Pandas: sample should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.sample(n=2, random_state=42)
        self.assertIsNot(ds, ds_result, "DataStore: sample should return new object")

    # ========== Nlargest/Nsmallest ==========

    def test_nlargest_returns_new_object(self):
        """nlargest should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.nlargest(3, 'age')
        self.assertIsNot(df, df_result, "Pandas: nlargest should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.nlargest(3, 'age')
        self.assertIsNot(ds, ds_result, "DataStore: nlargest should return new object")

    def test_nsmallest_returns_new_object(self):
        """nsmallest should return new object (matches pandas)."""
        # Pandas behavior
        df = self._get_fresh_df()
        df_result = df.nsmallest(3, 'age')
        self.assertIsNot(df, df_result, "Pandas: nsmallest should return new object")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds_result = ds.nsmallest(3, 'age')
        self.assertIsNot(ds, ds_result, "DataStore: nsmallest should return new object")


class TestDataStoreSpecificImmutability(unittest.TestCase):
    """
    Test DataStore-specific methods for immutability.

    These are methods that don't exist in pandas but should still follow
    the immutable pattern.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        data = {
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50],
        }
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def _get_fresh_ds(self):
        """Get a fresh DataStore instance."""
        return DataStore.from_file(self.csv_file)

    def test_sql_returns_new_object(self):
        """sql() method should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.sql("value > 20")
        self.assertIsNot(ds, ds_result, "DataStore: sql should return new object")

    def test_select_returns_new_object(self):
        """select() method should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.select('id', 'value')
        self.assertIsNot(ds, ds_result, "DataStore: select should return new object")

    def test_filter_condition_returns_new_object(self):
        """filter() with condition should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.filter(ds.value > 20)
        self.assertIsNot(ds, ds_result, "DataStore: filter should return new object")

    def test_sort_returns_new_object(self):
        """sort() method should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.sort('value')
        self.assertIsNot(ds, ds_result, "DataStore: sort should return new object")

    def test_limit_returns_new_object(self):
        """limit() method should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.limit(3)
        self.assertIsNot(ds, ds_result, "DataStore: limit should return new object")

    def test_offset_returns_new_object(self):
        """offset() method should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.offset(2)
        self.assertIsNot(ds, ds_result, "DataStore: offset should return new object")

    def test_distinct_returns_new_object(self):
        """distinct() method should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.distinct()
        self.assertIsNot(ds, ds_result, "DataStore: distinct should return new object")

    def test_with_format_settings_returns_new_object(self):
        """with_format_settings() should return new object."""
        ds = self._get_fresh_ds()
        ds_result = ds.with_format_settings(input_format_csv_trim_whitespaces=1)
        self.assertIsNot(ds, ds_result, "DataStore: with_format_settings should return new object")


class TestOriginalUnchanged(unittest.TestCase):
    """
    Test that operations don't modify the original object.

    This is critical for safe method chaining and query reuse.
    Both pandas and DataStore should preserve original objects.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        cls.data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'value': [10, 20, 30, 40, 50],
        }
        pd.DataFrame(cls.data).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def _get_fresh_df(self):
        """Get a fresh pandas DataFrame."""
        return pd.DataFrame(self.data)

    def _get_fresh_ds(self):
        """Get a fresh DataStore."""
        return DataStore.from_file(self.csv_file)

    def test_filter_does_not_modify_original(self):
        """filter should not modify original (pandas uses query/boolean indexing)."""
        # Pandas behavior (using boolean indexing, similar to filter)
        df = self._get_fresh_df()
        original_len = len(df)
        filtered_df = df[df['value'] > 20]
        self.assertEqual(len(df), original_len, "Pandas: original should be unchanged")
        self.assertLess(len(filtered_df), original_len, "Pandas: filtered should be smaller")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_len = len(ds)
        filtered_ds = ds.filter(ds.value > 20)
        self.assertEqual(len(ds), original_len, "DataStore: original should be unchanged")
        self.assertLess(len(filtered_ds), original_len, "DataStore: filtered should be smaller")

    def test_sql_does_not_modify_original(self):
        """sql should not modify original DataStore (pandas uses query)."""
        # Pandas behavior (using query, similar to sql)
        df = self._get_fresh_df()
        df1 = df.query("value > 10")
        df2 = df.query("value > 20")
        df3 = df.query("value > 30")
        self.assertEqual(len(df1), 4, "Pandas: query result 1")
        self.assertEqual(len(df2), 3, "Pandas: query result 2")
        self.assertEqual(len(df3), 2, "Pandas: query result 3")
        self.assertEqual(len(df), 5, "Pandas: original should still be 5 rows")

        # DataStore behavior
        ds = self._get_fresh_ds()
        ds1 = ds.sql("value > 10")
        ds2 = ds.sql("value > 20")
        ds3 = ds.sql("value > 30")
        self.assertEqual(len(ds1), 4, "DataStore: sql result 1")
        self.assertEqual(len(ds2), 3, "DataStore: sql result 2")
        self.assertEqual(len(ds3), 2, "DataStore: sql result 3")
        self.assertEqual(len(ds), 5, "DataStore: original should still be 5 rows")

    def test_add_prefix_does_not_modify_original(self):
        """add_prefix should not modify original."""
        # Pandas behavior
        df = self._get_fresh_df()
        original_columns = list(df.columns)
        prefixed_df = df.add_prefix('x_')
        self.assertEqual(list(df.columns), original_columns, "Pandas: original columns unchanged")
        self.assertTrue(all(c.startswith('x_') for c in prefixed_df.columns), "Pandas: prefixed columns")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_columns = list(ds.columns)
        prefixed_ds = ds.add_prefix('x_')
        self.assertEqual(list(ds.columns), original_columns, "DataStore: original columns unchanged")
        self.assertTrue(all(c.startswith('x_') for c in prefixed_ds.columns), "DataStore: prefixed columns")

    def test_head_does_not_modify_original(self):
        """head should not modify original."""
        # Pandas behavior
        df = self._get_fresh_df()
        original_len = len(df)
        head_df = df.head(2)
        self.assertEqual(len(df), original_len, "Pandas: original should be unchanged")
        self.assertEqual(len(head_df), 2, "Pandas: head should return 2 rows")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_len = len(ds)
        head_ds = ds.head(2)
        self.assertEqual(len(ds), original_len, "DataStore: original should be unchanged")
        self.assertEqual(len(head_ds), 2, "DataStore: head should return 2 rows")

    def test_sort_values_does_not_modify_original(self):
        """sort_values should not modify original."""
        # Pandas behavior
        df = self._get_fresh_df()
        original_values = list(df['value'])
        sorted_df = df.sort_values('value', ascending=False)
        self.assertEqual(list(df['value']), original_values, "Pandas: original order unchanged")
        self.assertEqual(list(sorted_df['value']), [50, 40, 30, 20, 10], "Pandas: sorted descending")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_values = list(ds['value'].values)
        sorted_ds = ds.sort_values('value', ascending=False)
        self.assertEqual(list(ds['value'].values), original_values, "DataStore: original order unchanged")
        self.assertEqual(list(sorted_ds['value'].values), [50, 40, 30, 20, 10], "DataStore: sorted descending")

    def test_chain_does_not_modify_original(self):
        """Method chaining should not modify original."""
        # Pandas behavior
        df = self._get_fresh_df()
        original_df = df.copy()
        _ = df[df['value'] > 10].add_prefix('col_').sort_values('col_value', ascending=False).head(2)
        assert_frame_equal(df, original_df, obj="Pandas: original should be unchanged")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_ds = ds.copy()
        _ = ds.filter(ds.value > 10).add_prefix('col_').sort_values('col_value', ascending=False).head(2)
        np.testing.assert_array_equal(ds.values, original_ds.values, err_msg="DataStore: original should be unchanged")

    def test_query_reuse(self):
        """Base query should be reusable without modification."""
        # Pandas behavior
        df = self._get_fresh_df()
        base_df = df[df['value'] > 10]
        result1_df = base_df[base_df['value'] > 20]
        result2_df = base_df[base_df['value'] > 30]
        result3_df = base_df.head(2)
        self.assertEqual(len(result1_df), 3, "Pandas: result1")
        self.assertEqual(len(result2_df), 2, "Pandas: result2")
        self.assertEqual(len(result3_df), 2, "Pandas: result3")

        # DataStore behavior
        ds = self._get_fresh_ds()
        base_ds = ds.filter(ds.value > 10)
        result1_ds = base_ds.filter(ds.value > 20)
        result2_ds = base_ds.filter(ds.value > 30)
        result3_ds = base_ds.head(2)
        self.assertEqual(len(result1_ds), 3, "DataStore: result1")
        self.assertEqual(len(result2_ds), 2, "DataStore: result2")
        self.assertEqual(len(result3_ds), 2, "DataStore: result3")

    def test_drop_does_not_modify_original(self):
        """drop should not modify original."""
        # Pandas behavior
        df = self._get_fresh_df()
        original_columns = list(df.columns)
        dropped_df = df.drop(columns=['name'])
        self.assertEqual(list(df.columns), original_columns, "Pandas: original columns unchanged")
        self.assertNotIn('name', dropped_df.columns, "Pandas: column dropped in result")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_columns = list(ds.columns)
        dropped_ds = ds.drop(columns=['name'])
        self.assertEqual(list(ds.columns), original_columns, "DataStore: original columns unchanged")
        self.assertNotIn('name', dropped_ds.columns, "DataStore: column dropped in result")

    def test_rename_does_not_modify_original(self):
        """rename should not modify original."""
        # Pandas behavior
        df = self._get_fresh_df()
        original_columns = list(df.columns)
        renamed_df = df.rename(columns={'name': 'full_name'})
        self.assertEqual(list(df.columns), original_columns, "Pandas: original columns unchanged")
        self.assertIn('full_name', renamed_df.columns, "Pandas: column renamed in result")

        # DataStore behavior
        ds = self._get_fresh_ds()
        original_columns = list(ds.columns)
        renamed_ds = ds.rename(columns={'name': 'full_name'})
        self.assertEqual(list(ds.columns), original_columns, "DataStore: original columns unchanged")
        self.assertIn('full_name', renamed_ds.columns, "DataStore: column renamed in result")

    def test_dropna_does_not_modify_original(self):
        """dropna should not modify original."""
        # Create data with NaN
        df = pd.DataFrame({'a': [1, 2, None], 'b': [4, None, 6]})
        original_len = len(df)
        dropped_df = df.dropna()
        self.assertEqual(len(df), original_len, "Pandas: original length unchanged")
        self.assertLess(len(dropped_df), original_len, "Pandas: dropna removed rows")

        # DataStore (with NaN data)
        temp_file = os.path.join(self.temp_dir, "nan_test.csv")
        df.to_csv(temp_file, index=False)
        ds = DataStore.from_file(temp_file)
        original_len = len(ds)
        dropped_ds = ds.dropna()
        self.assertEqual(len(ds), original_len, "DataStore: original length unchanged")
        self.assertLess(len(dropped_ds), original_len, "DataStore: dropna removed rows")
        os.unlink(temp_file)

    def test_fillna_does_not_modify_original(self):
        """fillna should not modify original."""
        # Create data with NaN
        df = pd.DataFrame({'a': [1, 2, None], 'b': [4, None, 6]})
        filled_df = df.fillna(0)
        self.assertTrue(pd.isna(df['a'].iloc[2]), "Pandas: original NaN preserved")
        self.assertEqual(filled_df['a'].iloc[2], 0, "Pandas: NaN filled in result")

        # DataStore (with NaN data)
        temp_file = os.path.join(self.temp_dir, "nan_test2.csv")
        df.to_csv(temp_file, index=False)
        ds = DataStore.from_file(temp_file)
        filled_ds = ds.fillna(0)
        self.assertTrue(pd.isna(ds['a'].values[2]), "DataStore: original NaN preserved")
        self.assertEqual(filled_ds['a'].values[2], 0, "DataStore: NaN filled in result")
        os.unlink(temp_file)


class TestInPlaceOperations(unittest.TestCase):
    """
    Test operations that ARE expected to modify in-place (like pandas).

    In pandas, __setitem__ (df['col'] = ...) modifies the DataFrame.
    DataStore follows this same pattern for familiarity.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd.DataFrame(data).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_setitem_modifies_in_place_pandas(self):
        """Verify pandas __setitem__ modifies in place."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df_id = id(df)

        df['c'] = df['a'] * 2

        # Same object, modified in place
        self.assertEqual(id(df), df_id)
        self.assertIn('c', df.columns)

    def test_setitem_modifies_in_place_datastore(self):
        """DataStore __setitem__ should modify in place (like pandas)."""
        ds = DataStore.from_file(self.csv_file)
        ds_id = id(ds)

        ds['c'] = ds['a'] * 2

        # Same object, modified in place (lazy operation added)
        self.assertEqual(id(ds), ds_id)
        # Column will be available after execution
        self.assertIn('c', ds.columns)

    def test_delitem_modifies_in_place_pandas(self):
        """Verify pandas __delitem__ modifies in place."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df_id = id(df)

        del df['b']

        # Same object, modified in place
        self.assertEqual(id(df), df_id)
        self.assertNotIn('b', df.columns)

    def test_delitem_modifies_in_place_datastore(self):
        """DataStore __delitem__ should modify in place (like pandas)."""
        ds = DataStore.from_file(self.csv_file)
        ds_id = id(ds)

        del ds['b']

        # Same object, modified in place (lazy operation added)
        self.assertEqual(id(ds), ds_id)
        # Column will be removed after execution
        self.assertNotIn('b', ds.columns)


if __name__ == '__main__':
    unittest.main(verbosity=2)
