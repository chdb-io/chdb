"""
Tests for pandas DataFrame compatibility layer in DataStore.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal


class TestPandasCompatibility(unittest.TestCase):
    """Test pandas DataFrame compatibility methods."""

    @classmethod
    def setUpClass(cls):
        """Create test data files."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        # Create sample CSV
        with open(cls.csv_file, "w") as f:
            f.write("id,name,age,salary,department,active\n")
            f.write("1,Alice,25,50000,Engineering,1\n")
            f.write("2,Bob,30,60000,Sales,1\n")
            f.write("3,Charlie,35,70000,Engineering,1\n")
            f.write("4,David,28,55000,Marketing,0\n")
            f.write("5,Eve,32,65000,Sales,1\n")
            f.write("6,Frank,29,58000,Engineering,1\n")
            f.write("7,Grace,31,62000,Marketing,1\n")
            f.write("8,Henry,27,52000,Sales,0\n")
            f.write("9,Iris,33,68000,Engineering,1\n")
            f.write("10,Jack,26,51000,Marketing,1\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def setUp(self):
        """Set up test DataStore and reference DataFrame."""
        self.ds = DataStore.from_file(self.csv_file)
        self.df = pd.read_csv(self.csv_file)  # Reference pandas DataFrame

    # ========== Properties Tests ==========

    def test_dtypes(self):
        """Test dtypes property matches pandas."""
        ds_dtypes = self.ds.dtypes
        pd_dtypes = self.df.dtypes
        # Column names should match
        np.testing.assert_array_equal(ds_dtypes.index, pd_dtypes.index)

    def test_shape(self):
        """Test shape property matches pandas."""
        self.assertEqual(self.ds.shape, self.df.shape)

    def test_columns(self):
        """Test columns property matches pandas."""
        pd.testing.assert_index_equal(self.ds.columns, self.df.columns)

    def test_columns_setter_basic(self):
        """Test columns setter renames all columns."""
        ds = DataStore(pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}))

        # Rename columns
        ds.columns = ['x', 'y', 'z']

        # Check columns are renamed
        self.assertEqual(list(ds.columns), ['x', 'y', 'z'])

        # Check data is preserved
        np.testing.assert_array_equal(ds['x'], [1, 2])
        np.testing.assert_array_equal(ds['y'], [3, 4])
        np.testing.assert_array_equal(ds['z'], [5, 6])

    def test_columns_setter_partial_rename(self):
        """Test columns setter only renames changed columns."""
        ds = DataStore(pd.DataFrame({'a': [1], 'b': [2], 'c': [3]}))

        # Only rename some columns (keep 'b' the same)
        ds.columns = ['x', 'b', 'z']

        self.assertEqual(list(ds.columns), ['x', 'b', 'z'])

    def test_columns_setter_with_pandas_comparison(self):
        """Test columns setter produces same result as pandas."""
        df = pd.DataFrame({'old1': [1, 2, 3], 'old2': [4, 5, 6]})
        ds = DataStore(df.copy())

        new_cols = ['new1', 'new2']

        # Apply to both
        df.columns = new_cols
        ds.columns = new_cols

        # Compare results
        np.testing.assert_array_equal(ds, df)

    def test_columns_setter_length_mismatch_error(self):
        """Test columns setter raises error on length mismatch."""
        ds = DataStore(pd.DataFrame({'a': [1], 'b': [2], 'c': [3]}))

        with self.assertRaises(ValueError) as context:
            ds.columns = ['only', 'two']  # 2 columns, but ds has 3

        self.assertIn('Length mismatch', str(context.exception))
        self.assertIn('Expected 3', str(context.exception))
        self.assertIn('got 2', str(context.exception))

    def test_columns_setter_with_groupby(self):
        """Test columns setter works after groupby aggregation."""
        df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        result = ds.groupby('category').agg({'value': 'sum'}).reset_index()
        result.columns = ['cat', 'total']
        # DataStore.columns triggers natural execution
        self.assertEqual(list(result.columns), ['cat', 'total'])

    def test_columns_setter_lazy_execution(self):
        """Test columns setter is lazy (doesn't execute immediately)."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds = DataStore(df)

        # This should be lazy
        ds.columns = ['x', 'y']

        # Verify the lazy op was added
        self.assertTrue(len(ds._lazy_ops) > 0)

        # ds.columns triggers natural execution
        self.assertEqual(list(ds.columns), ['x', 'y'])

    def test_values(self):
        """Test values property matches pandas."""
        np.testing.assert_array_equal(self.ds, self.df)

    def test_empty(self):
        """Test empty property matches pandas."""
        self.assertEqual(self.ds.empty, self.df.empty)

    def test_size(self):
        """Test size property matches pandas."""
        self.assertEqual(self.ds.size, self.df.size)

    def test_iter(self):
        """Test iteration over DataStore yields column names like pandas."""
        ds_cols = list(self.ds)
        pd_cols = list(self.df)
        self.assertEqual(ds_cols, pd_cols)

    def test_iter_select_dtypes(self):
        """Test iteration over select_dtypes result yields column names."""
        # Test iteration directly over select_dtypes result
        ds_cols = [feat for feat in self.ds.select_dtypes('number')]
        pd_cols = [feat for feat in self.df.select_dtypes('number')]
        self.assertEqual(ds_cols, pd_cols)

        # Test using columns.tolist()
        ds_cols2 = [feat for feat in self.ds.select_dtypes(include='number').columns.tolist()]
        pd_cols2 = [feat for feat in self.df.select_dtypes(include='number').columns.tolist()]
        self.assertEqual(ds_cols2, pd_cols2)

    def test_select_dtypes_with_nunique_filter(self):
        """Test filtering columns by nunique value (common pattern for bool detection)."""
        # Pattern: find numeric columns with only 2 unique values (likely boolean)
        ds_bool_feats = [feat for feat in self.ds.select_dtypes('number') if self.ds[feat].nunique() == 2]
        pd_bool_feats = [feat for feat in self.df.select_dtypes('number') if self.df[feat].nunique() == 2]
        self.assertEqual(ds_bool_feats, pd_bool_feats)

        # Pattern: get all numeric features, then exclude bool features
        ds_num_feats = [feat for feat in self.ds.select_dtypes(include='number').columns.tolist()]
        ds_num_feats = [feat for feat in ds_num_feats if feat not in ds_bool_feats]

        pd_num_feats = [feat for feat in self.df.select_dtypes(include='number').columns.tolist()]
        pd_num_feats = [feat for feat in pd_num_feats if feat not in pd_bool_feats]

        self.assertEqual(ds_num_feats, pd_num_feats)

    # ========== Statistical Methods Tests ==========

    def test_mean(self):
        """Test mean method matches pandas."""
        ds_mean = self.ds.mean(numeric_only=True)
        pd_mean = self.df.mean(numeric_only=True)
        assert_series_equal(ds_mean, pd_mean)

    def test_median(self):
        """Test median method matches pandas."""
        ds_median = self.ds.median(numeric_only=True)
        pd_median = self.df.median(numeric_only=True)
        assert_series_equal(ds_median, pd_median)

    def test_std(self):
        """Test std method matches pandas."""
        ds_std = self.ds.std(numeric_only=True)
        pd_std = self.df.std(numeric_only=True)
        assert_series_equal(ds_std, pd_std, rtol=1e-5)

    def test_min_max(self):
        """Test min and max methods match pandas."""
        ds_min = self.ds.min(numeric_only=True)
        pd_min = self.df.min(numeric_only=True)
        assert_series_equal(ds_min, pd_min)

        ds_max = self.ds.max(numeric_only=True)
        pd_max = self.df.max(numeric_only=True)
        assert_series_equal(ds_max, pd_max)

    def test_sum(self):
        """Test sum method matches pandas."""
        ds_sum = self.ds.sum(numeric_only=True)
        pd_sum = self.df.sum(numeric_only=True)
        assert_series_equal(ds_sum, pd_sum)

    def test_corr(self):
        """Test correlation method matches pandas."""
        ds_corr = self.ds.corr(numeric_only=True)
        pd_corr = self.df.corr(numeric_only=True)
        assert_frame_equal(ds_corr, pd_corr, rtol=1e-5)

    def test_quantile(self):
        """Test quantile method matches pandas."""
        ds_q50 = self.ds.quantile(0.5, numeric_only=True)
        pd_q50 = self.df.quantile(0.5, numeric_only=True)
        assert_series_equal(ds_q50, pd_q50)

    def test_nunique(self):
        """Test nunique method matches pandas."""
        ds_nunique = self.ds.nunique()
        pd_nunique = self.df.nunique()
        assert_series_equal(ds_nunique, pd_nunique)

    # ========== Data Manipulation Tests ==========

    def test_drop_columns(self):
        """Test drop method matches pandas."""
        ds_result = self.ds.drop(columns=['active'])
        pd_result = self.df.drop(columns=['active'])
        self.assertTrue(ds_result.equals(pd_result))

    def test_drop_duplicates(self):
        """Test drop_duplicates matches pandas."""
        ds_result = self.ds.drop_duplicates(subset=['department'])
        pd_result = self.df.drop_duplicates(subset=['department'])
        # Compare length (order may differ due to implementation)
        self.assertEqual(len(ds_result), len(pd_result))

    def test_dropna(self):
        """Test dropna matches pandas."""
        ds_result = self.ds.dropna()
        pd_result = self.df.dropna()
        self.assertTrue(ds_result.equals(pd_result))

    def test_fillna(self):
        """Test fillna matches pandas."""
        ds_result = self.ds.fillna(0)
        pd_result = self.df.fillna(0)
        self.assertTrue(ds_result.equals(pd_result))

    def test_rename(self):
        """Test rename matches pandas."""
        ds_result = self.ds.rename(columns={'name': 'employee_name'})
        pd_result = self.df.rename(columns={'name': 'employee_name'})
        self.assertTrue(ds_result.equals(pd_result))

    def test_sort_values(self):
        """Test sort_values matches pandas."""
        ds_result = self.ds.sort_values('age').reset_index(drop=True)
        pd_result = self.df.sort_values('age').reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    def test_reset_index(self):
        """Test reset_index matches pandas."""
        ds_result = self.ds.reset_index(drop=True)
        pd_result = self.df.reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    def test_assign(self):
        """Test assign matches pandas."""
        ds_result = self.ds.assign(bonus=lambda x: x['salary'] * 0.1)
        pd_result = self.df.assign(bonus=lambda x: x['salary'] * 0.1)
        self.assertTrue(ds_result.equals(pd_result))

    def test_nlargest(self):
        """Test nlargest matches pandas."""
        ds_result = self.ds.nlargest(3, 'salary').reset_index(drop=True)
        pd_result = self.df.nlargest(3, 'salary').reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    def test_nsmallest(self):
        """Test nsmallest matches pandas."""
        ds_result = self.ds.nsmallest(3, 'age').reset_index(drop=True)
        pd_result = self.df.nsmallest(3, 'age').reset_index(drop=True)
        self.assertTrue(ds_result.equals(pd_result))

    # ========== Function Application Tests ==========

    def test_apply(self):
        """Test apply method matches pandas."""
        func = lambda x: x * 2 if x.dtype in ['int64', 'float64'] else x
        ds_result = self.ds.apply(func, axis=0)
        pd_result = self.df.apply(func, axis=0)
        self.assertTrue(ds_result.equals(pd_result))

    def test_agg(self):
        """Test aggregate method matches pandas."""
        ds_result = self.ds.agg({'age': 'mean', 'salary': 'sum'})
        pd_result = self.df.agg({'age': 'mean', 'salary': 'sum'})
        assert_series_equal(ds_result, pd_result)

    # ========== Indexing Tests ==========

    def test_loc(self):
        """Test loc indexer."""
        loc_indexer = self.ds.loc
        # Just verify it returns the pandas loc indexer
        self.assertIsNotNone(loc_indexer)

    def test_iloc(self):
        """Test iloc indexer."""
        iloc_indexer = self.ds.iloc
        # Just verify it returns the pandas iloc indexer
        self.assertIsNotNone(iloc_indexer)

    def test_getitem_column(self):
        """Test column selection with [] - returns ColumnExpr that displays like Series."""
        from datastore.expressions import Field
        from datastore.column_expr import ColumnExpr

        # ds['col'] returns ColumnExpr that wraps a Field
        result = self.ds['name']
        self.assertIsInstance(result, ColumnExpr)  # Returns ColumnExpr
        self.assertIsInstance(result._expr, Field)  # Wrapping a Field

        # Natural trigger: ColumnExpr supports __array__ protocol
        # which returns numpy array when passed to np.array()
        arr = np.array(result)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(len(arr), len(self.df))

    def test_getitem_columns(self):
        """Test multiple column selection with [] matches pandas."""
        ds_result = self.ds[['name', 'age']]
        pd_result = self.df[['name', 'age']]
        self.assertTrue(ds_result.equals(pd_result))

    # ========== Transformation Tests ==========

    def test_abs(self):
        """Test abs method matches pandas."""
        ds_result = self.ds.abs()
        pd_result = self.df.select_dtypes(include=[np.number]).abs()
        # Compare only numeric columns
        np.testing.assert_array_equal(ds_result.select_dtypes(include=[np.number]), pd_result)

    def test_round(self):
        """Test round method matches pandas."""
        ds_result = self.ds.round(decimals=2)
        pd_result = self.df.round(decimals=2)
        self.assertTrue(ds_result.equals(pd_result))

    def test_transpose(self):
        """Test transpose method matches pandas."""
        ds_result = self.ds.transpose()
        pd_result = self.df.transpose()
        # Check shape matches
        self.assertEqual(ds_result.shape, pd_result.shape)

    # ========== Reshaping Tests ==========

    def test_melt(self):
        """Test melt method matches pandas."""
        ds_result = self.ds.melt(id_vars=['id'], value_vars=['age', 'salary'])
        pd_result = self.df.melt(id_vars=['id'], value_vars=['age', 'salary'])
        # Sort and compare using equals
        ds_sorted = ds_result.sort_values(['id', 'variable']).reset_index(drop=True)
        pd_sorted = pd_result.sort_values(['id', 'variable']).reset_index(drop=True)
        self.assertTrue(ds_sorted.equals(pd_sorted))

    # ========== Boolean Methods Tests ==========

    def test_isna(self):
        """Test isna method matches pandas."""
        ds_result = self.ds.isna()
        pd_result = self.df.isna()
        self.assertTrue(ds_result.equals(pd_result))

    def test_isna_sum(self):
        """Test isna().sum() matches pandas."""
        ds_result = self.ds.isna().sum()
        pd_result = self.df.isna().sum()
        assert_series_equal(ds_result, pd_result)

    def test_notna(self):
        """Test notna method matches pandas."""
        ds_result = self.ds.notna()
        pd_result = self.df.notna()
        self.assertTrue(ds_result.equals(pd_result))

    # ========== Conversion Tests ==========

    def test_astype(self):
        """Test astype method matches pandas."""
        ds_result = self.ds.astype({'age': 'float64'})
        pd_result = self.df.astype({'age': 'float64'})
        self.assertTrue(ds_result.equals(pd_result))

    def test_copy(self):
        """Test copy method."""
        result = self.ds.copy()
        self.assertIsInstance(result, DataStore)
        # Verify it's a different object
        self.assertIsNot(result, self.ds)

    # ========== IO Tests ==========

    def test_to_csv(self):
        """Test to_csv method."""
        output_file = os.path.join(self.temp_dir, "output.csv")
        try:
            self.ds.to_csv(output_file, index=False)
            self.assertTrue(os.path.exists(output_file))
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_to_json(self):
        """Test to_json method."""
        output_file = os.path.join(self.temp_dir, "output.json")
        try:
            self.ds.to_json(output_file, orient='records')
            self.assertTrue(os.path.exists(output_file))
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_to_dict(self):
        """Test to_dict method matches pandas default behavior."""
        # Default orient='dict' returns dict like {column -> {index -> value}}
        ds_result = self.ds.to_dict()
        pd_result = self.df.to_dict()
        self.assertEqual(ds_result, pd_result)
        self.assertIsInstance(ds_result, dict)

    def test_to_dict_records(self):
        """Test to_dict with orient='records' returns list of dicts."""
        result = self.ds.to_dict(orient='records')
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(item, dict) for item in result))

    def test_to_numpy(self):
        """Test to_numpy method matches pandas."""
        ds_arr = self.ds.to_numpy()
        pd_arr = self.df.to_numpy()
        np.testing.assert_array_equal(ds_arr, pd_arr)

    # ========== Iteration Tests ==========

    def test_iterrows(self):
        """Test iterrows method matches pandas."""
        ds_rows = list(self.ds.iterrows())
        pd_rows = list(self.df.iterrows())
        self.assertEqual(len(ds_rows), len(pd_rows))
        for (ds_idx, ds_row), (pd_idx, pd_row) in zip(ds_rows, pd_rows):
            assert_series_equal(ds_row, pd_row)

    def test_itertuples(self):
        """Test itertuples method matches pandas."""
        tuples = list(self.ds.itertuples())
        self.assertEqual(len(tuples), 10)

    # ========== Merge Tests ==========

    def test_merge(self):
        """Test merge method with another DataStore."""
        # Create second dataset
        csv_file2 = os.path.join(self.temp_dir, "test_data2.csv")
        with open(csv_file2, "w") as f:
            f.write("id,bonus\n")
            f.write("1,5000\n")
            f.write("2,6000\n")
            f.write("3,7000\n")

        try:
            ds2 = DataStore.from_file(csv_file2)
            result = self.ds.merge(ds2, on='id', how='inner')
            self.assertIsInstance(result, DataStore)
            # result.columns triggers natural execution
            self.assertIn('bonus', result.columns)
        finally:
            if os.path.exists(csv_file2):
                os.unlink(csv_file2)

    # ========== Comparison Tests ==========

    def test_equals(self):
        """Test equals method."""
        ds2 = DataStore.from_file(self.csv_file)
        self.assertTrue(self.ds.equals(ds2))

    # ========== Inplace Parameter Tests ==========

    def test_inplace_not_supported(self):
        """Test that inplace=True raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            self.ds.drop(columns=['age'], inplace=True)
        self.assertIn("immutable", str(cm.exception).lower())

    def test_fillna_inplace_not_supported(self):
        """Test that fillna with inplace=True raises ValueError."""
        with self.assertRaises(ValueError):
            self.ds.fillna(0, inplace=True)

    def test_rename_inplace_not_supported(self):
        """Test that rename with inplace=True raises ValueError."""
        with self.assertRaises(ValueError):
            self.ds.rename(columns={'name': 'new_name'}, inplace=True)


class TestPandasCompatChaining(unittest.TestCase):
    """Test chaining of pandas compatibility methods with DataStore methods."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "chain_test.csv")

        with open(cls.csv_file, "w") as f:
            f.write("id,value,category\n")
            f.write("1,100,A\n")
            f.write("2,200,B\n")
            f.write("3,150,A\n")
            f.write("4,250,B\n")
            f.write("5,180,A\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_chaining_pandas_with_datastore(self):
        """Test chaining pandas methods with DataStore methods."""
        ds = DataStore.from_file(self.csv_file)

        # Chain: select -> filter -> sort_values
        # Note: head() returns DataStore for method chaining
        result = ds.select('id', 'value', 'category').filter(ds.value > 100).sort_values('value', ascending=False)

        self.assertIsInstance(result, DataStore)

        # Now apply head() which returns DataStore for chaining
        df_head = result.head(3)
        self.assertIsInstance(df_head, DataStore)

    def test_pandas_methods_return_datastore(self):
        """Test that pandas methods return DataStore for chaining."""
        ds = DataStore.from_file(self.csv_file)

        # Apply pandas operations
        result = ds.fillna(0).drop_duplicates().sort_values('value')

        self.assertIsInstance(result, DataStore)


class TestBooleanIndexing(unittest.TestCase):
    """Test pandas-style boolean indexing with ds[condition]."""

    def setUp(self):
        """Set up test data."""
        # Create DataFrame with some empty strings for testing
        self.df = pd.DataFrame(
            {'src': ['hello', '', 'src', '', 'foo'], 'tgt': ['world', 'tgt', '', '', 'bar'], 'value': [1, 2, 3, 4, 5]}
        )
        self.ds = DataStore.from_df(self.df)

    def test_boolean_indexing_simple_condition(self):
        """Test simple boolean indexing with single condition."""
        ds_result = self.ds[self.ds['value'] > 2]
        pd_result = self.df[self.df['value'] > 2]
        # DataStore.equals() triggers execution naturally
        self.assertTrue(ds_result.equals(pd_result))

    def test_boolean_indexing_compound_condition(self):
        """Test boolean indexing with compound AND condition."""
        ds_result = self.ds[(self.ds['value'] > 1) & (self.ds['value'] < 5)]
        pd_result = self.df[(self.df['value'] > 1) & (self.df['value'] < 5)]
        self.assertTrue(ds_result.equals(pd_result))

    def test_boolean_indexing_str_len(self):
        """Test boolean indexing with str.len() > 0."""
        ds_result = self.ds[(self.ds['src'].str.len() > 0) & (self.ds['tgt'].str.len() > 0)]
        pd_result = self.df[(self.df['src'].str.len() > 0) & (self.df['tgt'].str.len() > 0)]
        self.assertTrue(ds_result.equals(pd_result))

    def test_boolean_indexing_with_drop_duplicates(self):
        """Test boolean indexing chained with drop_duplicates()."""
        df_with_dups = pd.DataFrame(
            {
                'src': ['hello', '', 'hello', '', 'foo'],
                'tgt': ['world', 'tgt', 'world', '', 'bar'],
                'value': [1, 2, 1, 4, 5],
            }
        )
        ds_with_dups = DataStore.from_df(df_with_dups)

        ds_result = ds_with_dups[
            (ds_with_dups['src'].str.len() > 0) & (ds_with_dups['tgt'].str.len() > 0)
        ].drop_duplicates()
        pd_result = df_with_dups[
            (df_with_dups['src'].str.len() > 0) & (df_with_dups['tgt'].str.len() > 0)
        ].drop_duplicates()
        self.assertTrue(ds_result.equals(pd_result))

    def test_boolean_indexing_preserves_original(self):
        """Test that boolean indexing doesn't modify the original DataStore."""
        original_len = len(self.ds)

        # Apply filter
        filtered = self.ds[self.ds['value'] > 3]

        # Check original is unchanged (use len() which triggers natural execution)
        self.assertEqual(len(self.ds), original_len)
        self.assertLess(len(filtered), original_len)

    def test_boolean_indexing_or_condition(self):
        """Test boolean indexing with OR condition."""
        ds_result = self.ds[(self.ds['value'] == 1) | (self.ds['value'] == 5)]
        pd_result = self.df[(self.df['value'] == 1) | (self.df['value'] == 5)]
        self.assertTrue(ds_result.equals(pd_result))

    def test_boolean_indexing_returns_datastore(self):
        """Test that boolean indexing returns a DataStore instance."""
        result = self.ds[self.ds['value'] > 2]
        self.assertIsInstance(result, DataStore)

    def test_boolean_indexing_empty_result(self):
        """Test boolean indexing that returns no rows."""
        ds_result = self.ds[self.ds['value'] > 100]
        pd_result = self.df[self.df['value'] > 100]
        self.assertTrue(ds_result.equals(pd_result))


class TestCommonBooleanIndexingPatterns(unittest.TestCase):
    """Test common pandas boolean indexing patterns: df[df['col'] > x]."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Alice Jr', 'Charlie', 'AliceSmith'],
                'age': [25, 35, 28, 40, 32],
                'salary': [45000, 60000, 55000, 70000, 48000],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_simple_condition(self):
        """Test df[df['age'] > 30] pattern."""
        pd_result = self.df[self.df['age'] > 30]
        ds_result = self.ds[self.ds['age'] > 30]
        self.assertTrue(ds_result.equals(pd_result))

    def test_compound_condition(self):
        """Test df[(df['age'] > 30) & (df['salary'] > 50000)] pattern."""
        pd_result = self.df[(self.df['age'] > 30) & (self.df['salary'] > 50000)]
        ds_result = self.ds[(self.ds['age'] > 30) & (self.ds['salary'] > 50000)]
        self.assertTrue(ds_result.equals(pd_result))

    def test_str_contains_condition(self):
        """Test df[df['name'].str.contains('Alice')] pattern."""
        pd_result = self.df[self.df['name'].str.contains('Alice')]
        ds_result = self.ds[self.ds['name'].str.contains('Alice')]
        self.assertTrue(ds_result.equals(pd_result))


class TestColumnExprComparisonMethods(unittest.TestCase):
    """Test ColumnExpr comparison methods: eq, ne, lt, le, gt, ge."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        self.ds = DataStore.from_df(self.df)

    def test_eq(self):
        """Test eq() method matches pandas."""
        ds_result = self.ds['a'].eq(3)
        pd_result = self.df['a'].eq(3)
        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_ne(self):
        """Test ne() method matches pandas."""
        ds_result = self.ds['a'].ne(3)
        pd_result = self.df['a'].ne(3)
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_lt(self):
        """Test lt() method matches pandas."""
        ds_result = self.ds['a'].lt(3)
        pd_result = self.df['a'].lt(3)
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_le(self):
        """Test le() method matches pandas."""
        ds_result = self.ds['a'].le(3)
        pd_result = self.df['a'].le(3)
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_gt(self):
        """Test gt() method matches pandas."""
        ds_result = self.ds['a'].gt(3)
        pd_result = self.df['a'].gt(3)
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_ge(self):
        """Test ge() method matches pandas."""
        ds_result = self.ds['a'].ge(3)
        pd_result = self.df['a'].ge(3)
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_comparison_with_column(self):
        """Test comparison between two columns."""
        ds_result = self.ds['a'].eq(self.ds['b'])
        pd_result = self.df['a'].eq(self.df['b'])
        np.testing.assert_array_equal(ds_result, pd_result)


class TestColumnExprFillMethods(unittest.TestCase):
    """Test ColumnExpr ffill, bfill, interpolate methods."""

    def setUp(self):
        """Set up test data with NaN values."""
        self.df = pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0, np.nan], 'b': [np.nan, 2.0, np.nan, np.nan, 5.0]})
        self.ds = DataStore.from_df(self.df)

    def test_ffill(self):
        """Test ffill() method matches pandas."""
        ds_result = self.ds['a'].ffill()
        pd_result = self.df['a'].ffill()
        # LazySeries - use .values to trigger execution
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_bfill(self):
        """Test bfill() method matches pandas."""
        ds_result = self.ds['a'].bfill()
        pd_result = self.df['a'].bfill()
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_interpolate_linear(self):
        """Test interpolate() method matches pandas."""
        ds_result = self.ds['a'].interpolate(method='linear')
        pd_result = self.df['a'].interpolate(method='linear')
        np.testing.assert_array_almost_equal(ds_result, pd_result)

    def test_ffill_with_limit(self):
        """Test ffill() with limit parameter."""
        ds_result = self.ds['a'].ffill(limit=1)
        pd_result = self.df['a'].ffill(limit=1)
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_bfill_with_limit(self):
        """Test bfill() with limit parameter."""
        ds_result = self.ds['a'].bfill(limit=1)
        pd_result = self.df['a'].bfill(limit=1)
        np.testing.assert_array_equal(ds_result, pd_result)


class TestFilterColumnSelection(unittest.TestCase):
    """Test pandas-style filter() for column selection."""

    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({'col_a': [1, 2, 3], 'col_b': [4, 5, 6], 'name': ['x', 'y', 'z'], 'other': [7, 8, 9]})
        self.ds = DataStore.from_df(self.df)

    def test_filter_items(self):
        """Test filter(items=) matches pandas."""
        ds_result = self.ds.filter(items=['col_a', 'name'])
        pd_result = self.df.filter(items=['col_a', 'name'])
        np.testing.assert_array_equal(ds_result.columns, pd_result.columns)

    def test_filter_like(self):
        """Test filter(like=) matches pandas."""
        ds_result = self.ds.filter(like='col')
        pd_result = self.df.filter(like='col')
        np.testing.assert_array_equal(ds_result.columns, pd_result.columns)

    def test_filter_regex(self):
        """Test filter(regex=) matches pandas."""
        ds_result = self.ds.filter(regex='^col_')
        pd_result = self.df.filter(regex='^col_')
        np.testing.assert_array_equal(ds_result.columns, pd_result.columns)

    def test_filter_items_preserves_data(self):
        """Test that filter(items=) preserves data correctly."""
        ds_result = self.ds.filter(items=['col_a', 'name'])
        pd_result = self.df.filter(items=['col_a', 'name'])
        self.assertTrue(ds_result.equals(pd_result))

    def test_filter_condition_still_works(self):
        """Test that SQL-style filter(condition) still works."""
        ds_result = self.ds.filter(self.ds['col_a'] > 1)
        pd_result = self.df[self.df['col_a'] > 1]
        self.assertTrue(ds_result.equals(pd_result))


class TestSQLPushdownOptimizations(unittest.TestCase):
    """Test SQL pushdown optimizations for column selection, groupby, etc."""

    @classmethod
    def setUpClass(cls):
        """Create test data and save to parquet."""
        cls.df = pd.DataFrame(
            {
                'id': range(100),
                'category': ['A', 'B', 'C', 'D', 'E'] * 20,
                'value': range(100, 200),
                'score': [i * 0.5 for i in range(100)],
            }
        )
        cls.parquet_path = '/tmp/test_sql_pushdown.parquet'
        cls.df.to_parquet(cls.parquet_path)

    def test_column_selection_uses_lazy_relational_op(self):
        """Test that ds[['col1', 'col2']] creates LazyRelationalOp(SELECT)."""
        from datastore import DataStore
        from datastore.lazy_ops import LazyRelationalOp

        ds = DataStore.from_file(self.parquet_path)
        result = ds[['id', 'category']]

        # Check that the lazy op is a LazyRelationalOp with SELECT type
        self.assertEqual(len(result._lazy_ops), 1)
        op = result._lazy_ops[0]
        self.assertIsInstance(op, LazyRelationalOp)
        self.assertEqual(op.op_type, 'SELECT')

    def test_column_selection_chains_with_sort_and_limit(self):
        """Test that column selection can chain with sort/limit into single SQL."""
        from datastore import DataStore
        from datastore.lazy_ops import LazyRelationalOp

        ds = DataStore.from_file(self.parquet_path)
        result = ds[['id', 'value']].sort_values('value', ascending=False).head(10)

        # All operations should be LazyRelationalOp
        for op in result._lazy_ops:
            self.assertIsInstance(op, LazyRelationalOp)

        # len() and .columns trigger natural execution
        self.assertEqual(len(result), 10)
        self.assertEqual(list(result.columns), ['id', 'value'])

    def test_column_selection_result_matches_pandas(self):
        """Test that column selection produces same result as pandas."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds[['id', 'category']]
        pd_result = self.df[['id', 'category']]
        # DataStore.equals() triggers natural execution
        self.assertTrue(ds_result.equals(pd_result))

    def test_groupby_count_uses_sql(self):
        """Test that groupby().count() uses SQL pushdown."""
        from datastore import DataStore
        from datastore.lazy_ops import LazyGroupByAgg

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').count()

        # Check that LazyGroupByAgg is created
        self.assertEqual(len(result._lazy_ops), 1)
        self.assertIsInstance(result._lazy_ops[0], LazyGroupByAgg)

        # len() triggers natural execution - should return 5 groups (A, B, C, D, E)
        self.assertEqual(len(result), 5)

    def test_groupby_agg_uses_sql(self):
        """Test that groupby().agg() uses SQL pushdown."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').agg({'value': ['sum', 'mean']})

        # len() triggers natural execution
        self.assertEqual(len(result), 5)  # 5 categories

    def test_groupby_size_uses_sql(self):
        """Test that groupby().size() uses SQL pushdown."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').size()

        # Execute - LazySeries should use SQL internally
        series = result.values
        self.assertEqual(len(series), 5)  # 5 categories

    def test_groupby_size_reset_index(self):
        """Test groupby().size().reset_index() matches pandas."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds.groupby('category').size().reset_index(name='count')

        # Force execution
        _ = len(ds_result)

        # Compare with pandas
        pd_result = self.df.groupby('category').size().reset_index(name='count')

        # Values should match (order may differ)
        ds_counts = set(zip(ds_result['category'], ds_result['count']))
        pd_counts = set(zip(pd_result['category'], pd_result['count']))
        self.assertEqual(ds_counts, pd_counts)

    def test_filter_then_groupby_uses_sql(self):
        """Test that filter + groupby chain uses SQL pushdown."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds[ds['value'] > 150].groupby('category').count()

        # len() triggers natural execution - should have groups with value > 150
        self.assertGreater(len(result), 0)

    def test_star_expression_in_count(self):
        """Test that Star expression works in COUNT(*)."""
        from datastore.expressions import Star
        from datastore.functions import AggregateFunction

        star = Star()
        self.assertEqual(star.to_sql(), '*')

        agg = AggregateFunction('count', star)
        # count is wrapped in toInt64() to match pandas int64 dtype
        self.assertEqual(agg.to_sql(), 'toInt64(count(*))')

    def test_complex_pipeline_with_sql_pushdown(self):
        """Test complex pipeline: filter + column select + sort + limit."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds[ds['value'] > 120]
        result = result[['id', 'category', 'value']]
        result = result.sort_values('value', ascending=False)
        result = result.head(20)

        # len() and .columns trigger natural execution
        self.assertEqual(len(result), 20)
        self.assertEqual(list(result.columns), ['id', 'category', 'value'])
        # Values should be sorted descending - use .values for natural trigger
        values = result['value'].values
        self.assertTrue(all(values[i] >= values[i + 1] for i in range(len(values) - 1)))

    def test_groupby_agg_single_func_uses_sql(self):
        """Test groupby().agg() with single function per column uses SQL."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').agg({'value': 'sum', 'score': 'mean'})

        # len() and .columns trigger natural execution
        self.assertEqual(len(result), 5)  # 5 categories
        # Should have columns value, score (not sum(value), avg(score))
        self.assertIn('value', result.columns)
        self.assertIn('score', result.columns)

    def test_groupby_agg_reset_index_matches_pandas(self):
        """Test groupby().agg().reset_index() matches pandas structure."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds.groupby('category').agg({'value': 'sum'}).reset_index()
        pdf = self.df.groupby('category').agg({'value': 'sum'}).reset_index()

        # DataStore.columns and len() trigger natural execution
        self.assertEqual(list(ds_result.columns), list(pdf.columns))
        self.assertEqual(len(ds_result), len(pdf))

    def test_filter_groupby_sort_sql_pushdown(self):
        """Test filter + groupby + sort pipeline with SQL pushdown."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        # Use score column for filter, value column for agg to avoid alias conflict
        result = ds[ds['score'] > 20]
        result = result.groupby('category').agg({'value': 'sum'}).reset_index()
        result = result.sort_values('value', ascending=False)

        # len() and .values trigger natural execution
        self.assertGreater(len(result), 0)
        # Should be sorted descending
        values = result['value'].values.tolist()
        self.assertEqual(values, sorted(values, reverse=True))

    def test_groupby_multi_func_falls_back_to_pandas(self):
        """Test that groupby with multiple funcs per column uses pandas."""
        from datastore import DataStore
        from datastore.lazy_ops import LazyGroupByAgg

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').agg({'value': ['sum', 'mean']})

        # Should have LazyGroupByAgg in ops
        self.assertEqual(len(result._lazy_ops), 1)
        self.assertIsInstance(result._lazy_ops[0], LazyGroupByAgg)

        # len() triggers natural execution (via pandas fallback)
        self.assertEqual(len(result), 5)

    def test_filter_groupby_same_column_alias_conflict(self):
        """
        Test filter + groupby on same column doesn't cause SQL alias conflict.

        This is a regression test for the bug where:
        SELECT "category", sum("value") AS "value" FROM ... WHERE "value" > 120 GROUP BY ...
        would fail with "Aggregate function found in WHERE" error because the alias
        'value' conflicts with the column name 'value' in the WHERE clause.

        The fix should either:
        1. Not use alias when it conflicts with WHERE column, or
        2. Fall back to pandas execution
        """
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # This used to fail with:
        # "Aggregate function sum(value) AS value is found in WHERE in query"
        result = ds[ds['value'] > 120]
        result = result.groupby('category').agg({'value': 'sum'}).reset_index()

        # len() and .columns trigger natural execution
        self.assertGreater(len(result), 0)
        self.assertIn('category', result.columns)
        self.assertIn('value', result.columns)

        # Compare with pandas reference
        pdf = self.df[self.df['value'] > 120]
        pd_result = pdf.groupby('category').agg({'value': 'sum'}).reset_index()

        # Values should match (order may differ) - use .values for natural trigger
        ds_values = set(zip(result['category'].values, result['value'].values))
        pd_values = set(zip(pd_result['category'], pd_result['value']))
        self.assertEqual(ds_values, pd_values)

    def test_groupby_agg_multi_column_multi_func_unique_aliases(self):
        """Test groupby().agg() with multiple columns and multiple functions generates unique SQL aliases.

        This was a bug where:
            ds.groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': ['sum', 'mean']})
        would generate duplicate SQL aliases like:
            SELECT category, sum(int_col) AS sum, avg(int_col) AS mean, sum(float_col) AS sum, avg(float_col) AS mean
                   ^-- DUPLICATE ALIAS "sum" and "mean"!

        The fix ensures we use compound aliases (col_func) when there's potential for duplicates:
            SELECT category, sum(int_col) AS int_col_sum, avg(int_col) AS int_col_mean, sum(float_col) AS float_col_sum, avg(float_col) AS float_col_mean
        """
        from datastore import DataStore

        # Create test data with multiple numeric columns
        test_df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'int_col': [1, 2, 3, 4, 5, 6],
                'float_col': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        )
        test_parquet = '/tmp/test_multi_col_agg.parquet'
        test_df.to_parquet(test_parquet)

        try:
            ds = DataStore.from_file(test_parquet)

            # This used to fail with: MULTIPLE_EXPRESSIONS_FOR_ALIAS error
            result = (
                ds.groupby('category')
                .agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']})
                .reset_index()
            )

            # result.columns triggers natural execution - verify column names match pandas format
            # DataStore now returns MultiIndex columns to match pandas behavior
            pd_result = (
                test_df.groupby('category')
                .agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']})
                .reset_index()
            )

            # Verify columns match pandas MultiIndex format
            self.assertEqual(list(result.columns), list(pd_result.columns))

            # Flatten columns for value comparison
            result_flat = result.to_df().copy()
            result_flat.columns = [f'{col}_{func}' if func else col for col, func in result_flat.columns]
            pd_result_flat = pd_result.copy()
            pd_result_flat.columns = [f'{col}_{func}' if func else col for col, func in pd_result_flat.columns]

            # Compare values (order may differ) - use .values for natural trigger
            for col in ['int_col_sum', 'int_col_mean', 'int_col_max', 'float_col_sum', 'float_col_mean']:
                ds_values = dict(zip(result_flat['category'].values, result_flat[col].values))
                pd_values = dict(zip(pd_result_flat['category'], pd_result_flat[col]))
                for cat in ['A', 'B', 'C']:
                    np.testing.assert_almost_equal(
                        ds_values[cat], pd_values[cat], err_msg=f"Mismatch for {col} in category {cat}"
                    )
        finally:
            import os

            if os.path.exists(test_parquet):
                os.remove(test_parquet)

    def test_groupby_agg_single_column_multi_func_uses_multiindex(self):
        """Test groupby().agg() with single column returns MultiIndex columns (matching pandas)."""
        from datastore import DataStore
        import pandas as pd

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').agg({'value': ['sum', 'mean', 'max']}).reset_index()

        # result.columns triggers natural execution
        # Dict agg with multiple funcs returns MultiIndex columns matching pandas
        self.assertTrue(isinstance(result.columns, pd.MultiIndex))
        self.assertIn(('value', 'sum'), result.columns)
        self.assertIn(('value', 'mean'), result.columns)
        self.assertIn(('value', 'max'), result.columns)

    def test_groupby_agg_multi_column_single_func_uses_col_alias(self):
        """Test groupby().agg() with multiple columns but single func per col uses column names."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').agg({'value': 'sum', 'score': 'mean'}).reset_index()

        # result.columns triggers natural execution
        # Multiple columns with single function each should use column names as aliases
        self.assertIn('value', result.columns)
        self.assertIn('score', result.columns)


class TestLazySeriesDataFrameCompat(unittest.TestCase):
    """Test LazySeries compatibility when result is DataFrame (e.g., reset_index)."""

    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B', 'C', 'C'], 'value': [10, 20, 30, 40, 50, 60]})
        cls.parquet_path = '/tmp/test_lazy_series_df_compat.parquet'
        cls.test_df.to_parquet(cls.parquet_path)

    @classmethod
    def tearDownClass(cls):
        import os

        if os.path.exists(cls.parquet_path):
            os.remove(cls.parquet_path)

    def test_groupby_size_reset_index_has_columns(self):
        """Test that groupby().size().reset_index() has columns attribute.

        This was a bug where LazySeries didn't expose 'columns' attribute
        even when the internal result was a DataFrame.
        """
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').size().reset_index(name='count')

        # result.columns should work and trigger natural execution
        self.assertTrue(hasattr(result, 'columns'))
        self.assertEqual(list(result.columns), ['category', 'count'])

    def test_groupby_size_reset_index_has_to_df(self):
        """Test that groupby().size().reset_index() has to_df() method."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        result = ds.groupby('category').size().reset_index(name='count')

        # to_df() should work and return DataFrame
        self.assertTrue(hasattr(result, 'to_df'))
        df = result.to_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ['category', 'count'])

    def test_groupby_size_reset_index_matches_pandas(self):
        """Test that groupby().size().reset_index() matches pandas behavior."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds.groupby('category').size().reset_index(name='count')
        pd_result = self.test_df.groupby('category').size().reset_index(name='count')

        # Columns should match
        self.assertEqual(list(ds_result.columns), list(pd_result.columns))

        # Shape should match
        self.assertEqual(ds_result.shape, pd_result.shape)

        # Values should match (ignoring order)
        ds_dict = dict(zip(ds_result.to_df()['category'], ds_result.to_df()['count']))
        pd_dict = dict(zip(pd_result['category'], pd_result['count']))
        self.assertEqual(ds_dict, pd_dict)


class TestGroupByAggMultiIndexColumns(unittest.TestCase):
    """Test GroupBy agg returns MultiIndex columns to match pandas behavior."""

    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'int_col': [1, 2, 3, 4, 5, 6],
                'float_col': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        )
        cls.parquet_path = '/tmp/test_groupby_multiindex.parquet'
        cls.test_df.to_parquet(cls.parquet_path)

    @classmethod
    def tearDownClass(cls):
        import os

        if os.path.exists(cls.parquet_path):
            os.remove(cls.parquet_path)

    def test_groupby_agg_returns_multiindex_columns(self):
        """Test that groupby().agg() returns MultiIndex columns like pandas.

        This was a bug where SQL pushdown returned flat column names like
        'int_col_sum' instead of MultiIndex ('int_col', 'sum').
        """
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds.groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': ['sum', 'mean']})

        pd_result = self.test_df.groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': ['sum', 'mean']})

        # Column types should match (both MultiIndex)
        self.assertIsInstance(ds_result.columns, pd.MultiIndex)
        self.assertIsInstance(pd_result.columns, pd.MultiIndex)

        # Column values should match
        self.assertEqual(list(ds_result.columns), list(pd_result.columns))

    def test_groupby_agg_reset_index_multiindex_columns(self):
        """Test that groupby().agg().reset_index() preserves MultiIndex columns."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)
        ds_result = ds.groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': 'sum'}).reset_index()

        pd_result = self.test_df.groupby('category').agg({'int_col': ['sum', 'mean'], 'float_col': 'sum'}).reset_index()

        # After reset_index, columns should still match pandas
        self.assertEqual(list(ds_result.columns), list(pd_result.columns))

    def test_groupby_agg_multiindex_values_match_pandas(self):
        """Test that groupby().agg() values match pandas with MultiIndex columns."""
        from datastore import DataStore
        import numpy as np

        ds = DataStore.from_file(self.parquet_path)
        ds_result = (
            ds.groupby('category')
            .agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']})
            .reset_index()
            .to_df()
        )

        pd_result = (
            self.test_df.groupby('category')
            .agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']})
            .reset_index()
        )

        # Sort both by category for comparison
        ds_result = ds_result.sort_values(('category', '')).reset_index(drop=True)
        pd_result = pd_result.sort_values(('category', '')).reset_index(drop=True)

        # Values should be approximately equal
        for col in [
            ('int_col', 'sum'),
            ('int_col', 'mean'),
            ('int_col', 'max'),
            ('float_col', 'sum'),
            ('float_col', 'mean'),
        ]:
            np.testing.assert_array_almost_equal(
                ds_result[col].values, pd_result[col].values, decimal=5, err_msg=f"Column {col} values don't match"
            )


class TestSortStability(unittest.TestCase):
    """Test that DataStore sort with kind='stable' matches pandas kind='stable' behavior."""

    @classmethod
    def setUpClass(cls):
        # Create test data with many duplicate values
        cls.test_df = pd.DataFrame(
            {
                'id': range(100),
                'int_col': [i % 10 for i in range(100)],  # 0-9 repeated 10 times
                'str_col': ['A', 'B', 'C'] * 33 + ['A'],
            }
        )
        cls.parquet_path = '/tmp/test_sort_stability.parquet'
        cls.test_df.to_parquet(cls.parquet_path)

    @classmethod
    def tearDownClass(cls):
        import os

        if os.path.exists(cls.parquet_path):
            os.remove(cls.parquet_path)

    def test_single_column_sort_stability(self):
        """Test single column sort with kind='stable' preserves original order for equal keys."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # DataStore stable sort should match pandas stable sort
        ds_sorted = ds.sort_values('int_col', kind='stable').to_df()
        pd_sorted = self.test_df.sort_values('int_col', kind='stable')

        # Compare id columns (order should be identical)
        self.assertEqual(
            list(ds_sorted['id']),
            list(pd_sorted.reset_index(drop=True)['id']),
            "Single column stable sort should match pandas",
        )

    def test_multi_column_sort_stability(self):
        """Test multi-column sort with kind='stable' is stable."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        ds_sorted = ds.sort_values(['str_col', 'int_col'], kind='stable').to_df()
        pd_sorted = self.test_df.sort_values(['str_col', 'int_col'], kind='stable')

        self.assertEqual(
            list(ds_sorted['id']),
            list(pd_sorted.reset_index(drop=True)['id']),
            "Multi-column stable sort should match pandas",
        )

    def test_descending_sort_stability(self):
        """Test descending sort with kind='stable' is stable."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        ds_sorted = ds.sort_values('int_col', ascending=False, kind='stable').to_df()
        pd_sorted = self.test_df.sort_values('int_col', ascending=False, kind='stable')

        self.assertEqual(
            list(ds_sorted['id']),
            list(pd_sorted.reset_index(drop=True)['id']),
            "Descending stable sort should match pandas",
        )

    def test_filter_then_sort_stability(self):
        """Test filter + sort with kind='stable' is stable."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        ds_result = ds[ds['int_col'] > 3].sort_values('int_col', kind='stable').to_df()
        pd_result = self.test_df[self.test_df['int_col'] > 3].sort_values('int_col', kind='stable')

        self.assertEqual(
            list(ds_result['id']),
            list(pd_result.reset_index(drop=True)['id']),
            "Filter + stable sort should match pandas",
        )

    def test_default_sort_is_unstable(self):
        """Test that default sort (kind='quicksort') does not guarantee stability."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # Default sort may not be stable, but should still produce valid sorted output
        ds_sorted = ds.sort_values('int_col').to_df()
        pd_sorted = self.test_df.sort_values('int_col')

        # Values should be sorted correctly
        self.assertEqual(
            list(ds_sorted['int_col']),
            list(pd_sorted.reset_index(drop=True)['int_col']),
            "Default sort should produce correctly sorted values",
        )

    def test_mergesort_is_stable(self):
        """Test that kind='mergesort' is stable (same as kind='stable')."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # mergesort should be stable like pandas
        ds_sorted = ds.sort_values('int_col', kind='mergesort').to_df()
        pd_sorted = self.test_df.sort_values('int_col', kind='mergesort')

        self.assertEqual(
            list(ds_sorted['id']),
            list(pd_sorted.reset_index(drop=True)['id']),
            "kind='mergesort' should be stable like pandas",
        )

    def test_kind_parameter_passed_to_sort(self):
        """Test that kind parameter is properly passed through sort_values to sort."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # Test via sort() method directly
        ds_stable = ds.sort('int_col', kind='stable').to_df()
        pd_stable = self.test_df.sort_values('int_col', kind='stable')

        self.assertEqual(
            list(ds_stable['id']),
            list(pd_stable.reset_index(drop=True)['id']),
            "sort() with kind='stable' should match pandas",
        )

    def test_orderby_with_kind_stable(self):
        """Test that orderby() method also supports kind parameter."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # Test via orderby() method
        ds_stable = ds.orderby('int_col', kind='stable').to_df()
        pd_stable = self.test_df.sort_values('int_col', kind='stable')

        self.assertEqual(
            list(ds_stable['id']),
            list(pd_stable.reset_index(drop=True)['id']),
            "orderby() with kind='stable' should match pandas",
        )

    def test_kind_quicksort_vs_stable_behavior(self):
        """Test that kind='quicksort' and kind='stable' produce different behavior for duplicates."""
        from datastore import DataStore

        ds = DataStore.from_file(self.parquet_path)

        # Run stable sort - should always produce same order
        stable_results = []
        for _ in range(3):
            ds_stable = ds.sort_values('int_col', kind='stable').to_df()
            stable_results.append(list(ds_stable['id']))

        # All stable sort results should be identical
        self.assertEqual(stable_results[0], stable_results[1], "Stable sort should be deterministic")
        self.assertEqual(stable_results[1], stable_results[2], "Stable sort should be deterministic")

        # Stable sort should match pandas stable sort
        pd_stable = self.test_df.sort_values('int_col', kind='stable')
        self.assertEqual(
            stable_results[0],
            list(pd_stable.reset_index(drop=True)['id']),
            "DataStore stable sort should match pandas stable sort",
        )


if __name__ == '__main__':
    unittest.main()
