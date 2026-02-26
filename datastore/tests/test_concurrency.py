"""
Tests for thread-safety and concurrent execution.

This module tests that DataStore can be safely used in concurrent scenarios
thanks to unique variable names for each instance.
"""

import unittest
import tempfile
import os
import concurrent.futures
import threading

from datastore import DataStore


class TestConcurrency(unittest.TestCase):
    """Test concurrent execution scenarios."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        cls.temp_dir = tempfile.mkdtemp()

        # Create multiple test files
        cls.files = []
        for i in range(5):
            csv_file = os.path.join(cls.temp_dir, f"data_{i}.csv")
            with open(csv_file, "w") as f:
                f.write("id,value,category\n")
                for j in range(10):
                    f.write(f"{j},{j * (i + 1)},cat_{j % 3}\n")
            cls.files.append(csv_file)

    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        for file in cls.files:
            if os.path.exists(file):
                os.unlink(file)
        if os.path.exists(cls.temp_dir):
            os.rmdir(cls.temp_dir)

    def test_unique_variable_names(self):
        """Test that each DataStore instance gets a unique variable name."""
        ds1 = DataStore.from_file(self.files[0])
        ds2 = DataStore.from_file(self.files[1])
        ds3 = DataStore.from_file(self.files[2])

        # Each should have a unique variable name
        self.assertNotEqual(ds1._df_var_name, ds2._df_var_name)
        self.assertNotEqual(ds2._df_var_name, ds3._df_var_name)
        self.assertNotEqual(ds1._df_var_name, ds3._df_var_name)

        # All should start with the prefix
        self.assertTrue(ds1._df_var_name.startswith('__ds_df_'))
        self.assertTrue(ds2._df_var_name.startswith('__ds_df_'))
        self.assertTrue(ds3._df_var_name.startswith('__ds_df_'))

    def test_variable_name_preserved_in_chain(self):
        """Test that variable name is preserved in lazy operation chains."""
        ds = DataStore.from_file(self.files[0])
        original_var_name = ds._df_var_name

        # SQL operations keep the variable name (returns new object, like pandas)
        ds1 = ds.select('*').filter(ds.value > 5)
        self.assertEqual(ds1._df_var_name, original_var_name)

        # Pandas operations also keep the variable name (returns new object)
        ds2 = ds1.add_prefix('x_')
        self.assertEqual(ds2._df_var_name, original_var_name)

        # All operations return NEW objects (matches pandas behavior)
        # In pandas: df.add_prefix('x_') returns a new DataFrame, not the same one
        self.assertIsNot(ds1, ds2)

        # TODO: Re-enable this test when chDB concurrency issues are resolved
        # def test_concurrent_queries(self):
        #     """Test concurrent execution of queries on different files."""
        #     def process_file(file_path):
        #         ds = DataStore.from_file(file_path)
        #         result = (ds
        #             .select('*')
        #             .filter(ds.value > 3)
        #             .add_prefix('col_')
        #             .filter(ds.col_value > 5)
        #             .to_df())
        #         return result

        #     # Execute concurrently
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #         futures = [executor.submit(process_file, f) for f in self.files]
        #         results = [f.result() for f in concurrent.futures.as_completed(futures)]

        #     # All should succeed
        #     self.assertEqual(len(results), 5)

        #     # All should have correct columns
        #     for result in results:
        #         self.assertIn('col_id', result.columns)
        #         self.assertIn('col_value', result.columns)
        #         self.assertNotIn('id', result.columns)

        # def test_concurrent_mixed_operations(self):
        #     """Test concurrent mixed SQL and pandas operations."""
        #     def complex_operation(file_path, multiplier):
        #         ds = DataStore.from_file(file_path)
        #         return (ds
        #             .filter(ds.value > 2)                     # SQL
        #             .assign(doubled=lambda x: x['value'] * multiplier)  # Pandas
        #             .filter(ds.doubled > 10)                  # SQL on DataFrame
        #             .sort_values('doubled', ascending=False)  # Pandas
        #             .head(3))

        #     # Execute with different parameters concurrently
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #         futures = [
        #             executor.submit(complex_operation, self.files[i], i + 1)
        #             for i in range(5)
        #         ]
        #         results = [f.result() for f in concurrent.futures.as_completed(futures)]

        #     # All should succeed
        #     self.assertEqual(len(results), 5)

        #     # All should have the 'doubled' column
        #     for result in results:
        #         self.assertIn('doubled', result.columns)

        # def test_no_variable_name_collision(self):
        #     """Test that concurrent operations don't collide in global namespace."""
        #     collision_detected = [False]
        #     lock = threading.Lock()

        #     def check_operation(file_path):
        #         ds = DataStore.from_file(file_path)

        #         # Execute
        #         ds_mat = ds.add_prefix('x_')

        #         # Check if variable exists in globals
        #         if ds_mat._df_var_name in globals():
        #             # Verify it's our DataFrame
        #             if globals()[ds_mat._df_var_name] is not ds_mat._cached_df:
        #                 with lock:
        #                     collision_detected[0] = True

        #         # Execute SQL on DataFrame
        #         ds_filtered = ds_mat.filter(ds.x_value > 5)
        #         return ds_filtered.to_df()

        # # Execute many operations concurrently
        # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        #     futures = []
        #     for _ in range(20):
        #         for file in self.files:
        #             futures.append(executor.submit(check_operation, file))

        #     results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # # No collisions should be detected
        # self.assertFalse(collision_detected[0], "Variable name collision detected!")

        # # All results should be valid
        # self.assertEqual(len(results), 100)  # 20 iterations * 5 files


class TestVariableNameGeneration(unittest.TestCase):
    """Test variable name generation."""

    def test_variable_name_format(self):
        """Test that variable names follow expected format."""
        temp_dir = tempfile.mkdtemp()
        csv_file = os.path.join(temp_dir, "test.csv")

        with open(csv_file, "w") as f:
            f.write("a,b\n1,2\n")

        try:
            ds = DataStore.from_file(csv_file)

            # Check format
            var_name = ds._df_var_name
            self.assertTrue(var_name.startswith('__ds_df_'))
            self.assertTrue(var_name.endswith('__'))

            # Check it's a valid Python identifier
            self.assertTrue(var_name.isidentifier())

            # Check length (UUID hex is 32 chars + prefix + suffix)
            self.assertGreater(len(var_name), 40)

        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def test_lazy_operations_preserve_variable_name(self):
        """Test that pandas-style operations preserve variable name but return new object.

        This matches pandas behavior where df.add_prefix('x_') returns a new DataFrame.
        """
        temp_dir = tempfile.mkdtemp()
        csv_file = os.path.join(temp_dir, "test.csv")

        with open(csv_file, "w") as f:
            f.write("a,b\n1,2\n3,4\n")

        try:
            ds = DataStore.from_file(csv_file)
            var1 = ds._df_var_name

            # add_prefix returns a new DataStore (matches pandas behavior)
            ds2 = ds.add_prefix('x_')
            var2 = ds2._df_var_name

            # Variable name is preserved in the new object
            self.assertEqual(var1, var2)
            # But returns new object (matches pandas: df.add_prefix() returns new DataFrame)
            self.assertIsNot(ds, ds2)

        finally:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)


if __name__ == '__main__':
    unittest.main()
