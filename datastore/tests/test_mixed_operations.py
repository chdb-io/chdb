"""
Tests for mixed SQL and pandas operations.

This module tests the critical behavior of mixing DataStore SQL operations
with pandas DataFrame operations in a chain.
"""

from tests.test_utils import get_dataframe
import unittest
import tempfile
import os

from datastore import DataStore


class TestMixedOperations(unittest.TestCase):
    """Test mixing SQL queries with pandas operations."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "test_data.csv")

        with open(cls.csv_file, "w") as f:
            f.write("id,name,age,salary\n")
            f.write("1,Alice,25,50000\n")
            f.write("2,Bob,30,60000\n")
            f.write("3,Charlie,35,70000\n")
            f.write("4,David,28,55000\n")
            f.write("5,Eve,32,65000\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_sql_then_pandas_prefix(self):
        """Test SQL operations followed by pandas add_prefix."""
        ds = DataStore.from_file(self.csv_file)

        # SQL operations
        ds_filtered = ds.select('*').filter(ds.age > 25)

        # Pandas operation
        ds_prefixed = ds_filtered.add_prefix('col_')

        # Get DataFrame
        df = ds_prefixed.to_df()

        # Verify columns have prefix
        self.assertIn('col_id', df.columns)
        self.assertIn('col_name', df.columns)
        self.assertIn('col_age', df.columns)
        self.assertNotIn('id', df.columns)

        # Verify data is filtered (age > 25)
        self.assertTrue(all(df['col_age'] > 25))

    def test_sql_then_pandas_suffix(self):
        """Test SQL operations followed by pandas add_suffix."""
        ds = DataStore.from_file(self.csv_file)

        # SQL filter and pandas suffix
        result = ds.filter(ds.salary > 55000).add_suffix('_info')
        df = result.to_df()

        # Verify columns have suffix
        self.assertIn('id_info', df.columns)
        self.assertIn('name_info', df.columns)
        self.assertNotIn('id', df.columns)

        # Verify data is filtered
        self.assertTrue(all(df['salary_info'] > 55000))

    def test_sql_then_pandas_rename(self):
        """Test SQL operations followed by pandas rename."""
        ds = DataStore.from_file(self.csv_file)

        result = (
            ds.select('id', 'name', 'age')
            .filter(ds.age < 35)
            .rename(columns={'id': 'employee_id', 'name': 'employee_name'})
        )

        df = result.to_df()

        # Verify renamed columns
        self.assertIn('employee_id', df.columns)
        self.assertIn('employee_name', df.columns)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('name', df.columns)

        # Verify filtering worked
        self.assertTrue(all(df['age'] < 35))

    def test_sql_then_pandas_drop(self):
        """Test SQL select followed by pandas drop."""
        ds = DataStore.from_file(self.csv_file)

        result = ds.select('*').filter(ds.age > 25).drop(columns=['salary'])

        df = result.to_df()

        # Verify column dropped
        self.assertNotIn('salary', df.columns)
        self.assertIn('id', df.columns)
        self.assertIn('name', df.columns)

        # Verify filtering
        self.assertTrue(all(df['age'] > 25))

    def test_pandas_then_pandas(self):
        """Test chaining multiple pandas operations."""
        ds = DataStore.from_file(self.csv_file)

        result = ds.add_prefix('p1_').add_suffix('_p2').rename(columns={'p1_id_p2': 'final_id'})

        df = result.to_df()

        # Verify all transformations applied
        self.assertIn('final_id', df.columns)
        self.assertIn('p1_name_p2', df.columns)
        self.assertNotIn('id', df.columns)
        self.assertNotIn('p1_id_p2', df.columns)

    def test_sql_pandas_sql_warning(self):
        """Test that pandas operations work correctly."""
        ds = DataStore.from_file(self.csv_file)

        # SQL operation
        ds1 = ds.select('*').filter(ds.age > 25)

        # Pandas operation
        ds2 = ds1.add_prefix('col_')

        # to_df() should return DataFrame with correct columns
        df = ds2.to_df()
        self.assertIn('col_id', df.columns)

    def test_executed_datastore_properties(self):
        """Test that executed DataStore properties work correctly."""
        ds = DataStore.from_file(self.csv_file)

        # Apply pandas operation
        ds_transformed = ds.add_prefix('x_')

        # These should all use the cached DataFrame
        shape = ds_transformed.shape
        cols = ds_transformed.columns
        dtypes = ds_transformed.dtypes

        # Verify they reflect the pandas transformation
        self.assertIn('x_id', cols)
        self.assertIn('x_name', cols)
        self.assertNotIn('id', cols)

    def test_multiple_to_df_calls(self):
        """Test that multiple to_df() calls on executed DataStore return same data."""
        ds = DataStore.from_file(self.csv_file)

        # Apply pandas transformation
        ds_renamed = ds.rename(columns={'id': 'ID', 'name': 'NAME'})

        # Multiple to_df() calls should return consistent results
        df1 = ds_renamed.to_df()
        df2 = ds_renamed.to_df()
        df3 = ds_renamed.to_df()

        # All should have renamed columns
        self.assertIn('ID', df1.columns)
        self.assertIn('ID', df2.columns)
        self.assertIn('ID', df3.columns)

        # Should be the same cached object
        self.assertIs(df1, df2)
        self.assertIs(df2, df3)

    def test_complex_mixed_chain(self):
        """Test complex chain with SQL, pandas, SQL-style, pandas operations."""
        ds = DataStore.from_file(self.csv_file)

        result = (
            ds.select('id', 'name', 'age', 'salary')  # SQL: SELECT
            .filter(ds.age > 25)  # SQL: WHERE
            .assign(bonus=lambda x: x['salary'] * 0.1)  # Pandas: add column
            .add_prefix('emp_')  # Pandas: rename
            .query('emp_salary > 55000')  # Pandas: filter
            .sort_values('emp_salary', ascending=False)  # Pandas: sort
            .head(3)
        )  # Pandas: limit

        # Verify final result has all transformations
        # Get DataFrame to access columns as Series
        result_df = get_dataframe(result)

        self.assertIn('emp_id', result_df.columns)
        self.assertIn('emp_name', result_df.columns)
        self.assertIn('emp_bonus', result_df.columns)
        self.assertNotIn('id', result_df.columns)

        # Verify filtering worked (age > 25 AND salary > 55000)
        self.assertTrue(all(result_df['emp_age'] > 25))
        self.assertTrue(all(result_df['emp_salary'] > 55000))

        # Verify only 3 rows
        self.assertEqual(len(result_df), 3)


class TestExecutionModel(unittest.TestCase):
    """Test the execution model for mixed operations."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, "exec_test.csv")

        with open(cls.csv_file, "w") as f:
            f.write("a,b,c\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            f.write("7,8,9\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if os.path.exists(cls.csv_file):
            os.unlink(cls.csv_file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_sql_operations_are_lazy(self):
        """Test that SQL-only operations are lazy (not executed until to_df)."""
        ds = DataStore.from_file(self.csv_file)

        # Only SQL operations - should record ops but not execute
        ds2 = ds.select('*').filter(ds.a > 1).limit(10)

        # Should have lazy ops recorded
        self.assertTrue(len(ds2._lazy_ops) > 0)

    def test_pandas_operations_execute(self):
        """Test that pandas operations execute and return correct data."""
        ds = DataStore.from_file(self.csv_file)

        # SQL operations
        ds1 = ds.select('*').filter(ds.a > 0)

        # Pandas operation
        ds2 = ds1.add_prefix('x_')

        # to_df should return correct data
        df = ds2.to_df()
        self.assertIn('x_a', df.columns)

    def test_chained_pandas_operations(self):
        """Test that chained pandas operations work correctly."""
        ds = DataStore.from_file(self.csv_file)

        # First pandas operation
        ds1 = ds.add_prefix('p1_')

        # Second pandas operation on ds1
        ds2 = ds1.add_suffix('_p2')

        # Verify result has correct columns
        df = ds2.to_df()
        self.assertIn('p1_a_p2', df.columns)


if __name__ == '__main__':
    unittest.main()
