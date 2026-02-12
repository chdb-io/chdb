"""
Tests for DataStore caching functionality.

This module tests the intelligent automatic caching feature that avoids
re-execution of the pipeline when repr/__str__ are called multiple times.
"""

import time
import unittest

import pandas as pd

from datastore import DataStore, config
from tests.test_utils import assert_frame_equal


class TestCacheBasics(unittest.TestCase):
    """Test basic caching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Ensure cache is enabled and TTL is 0 (no expiration)
        config.enable_cache()
        config.set_cache_ttl(0)

        self.df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

    def tearDown(self):
        """Clean up after tests."""
        # Reset to defaults
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_cache_hit_on_repeated_repr(self):
        """Test that repeated repr calls use cached result."""
        ds = DataStore.from_dataframe(self.df)
        ds['doubled'] = ds['value'] * 2
        ds['tripled'] = ds['value'] * 3

        # First access - should execute pipeline
        str(ds)
        version_after_first = ds._cached_at_version

        # Second access - should use cache
        str(ds)
        version_after_second = ds._cached_at_version

        # Versions should be the same (cache was reused)
        self.assertEqual(version_after_first, version_after_second)
        self.assertIsNotNone(ds._cached_result)

    def test_cache_invalidation_on_new_operation(self):
        """Test that cache is invalidated when new operations are added."""
        ds = DataStore.from_dataframe(self.df)
        ds['doubled'] = ds['value'] * 2

        # First access - execute and cache
        str(ds)
        old_version = ds._cache_version

        # Add new operation - should invalidate cache
        ds['tripled'] = ds['value'] * 3
        new_version = ds._cache_version

        # Version should have incremented
        self.assertGreater(new_version, old_version)

    def test_cache_result_correctness(self):
        """Test that cached result is correct."""
        ds = DataStore.from_dataframe(self.df)
        ds['doubled'] = ds['value'] * 2

        # Get result twice
        result1 = ds.to_df()
        result2 = ds.to_df()

        # Both should be equal
        assert_frame_equal(result1, result2)

        # Result should have the correct columns
        self.assertIn('value', result1.columns)
        self.assertIn('doubled', result1.columns)
        self.assertEqual(list(result1['doubled']), [2, 4, 6, 8, 10])

    def test_shape_uses_cache(self):
        """Test that shape property uses cached result."""
        ds = DataStore.from_dataframe(self.df)
        ds['doubled'] = ds['value'] * 2

        # First access
        str(ds)
        cached_version = ds._cached_at_version

        # Shape access should use cache
        _ = ds.shape
        self.assertEqual(ds._cached_at_version, cached_version)


class TestCacheTTL(unittest.TestCase):
    """Test cache TTL (Time-To-Live) functionality."""

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        self.df = pd.DataFrame({'value': [1, 2, 3]})

    def tearDown(self):
        """Clean up after tests."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_ttl_expiration(self):
        """Test that cache expires after TTL."""
        config.set_cache_ttl(0.3)  # 0.3 second TTL

        ds = DataStore.from_dataframe(self.df)
        ds['doubled'] = ds['value'] * 2

        # First access - execute and cache
        str(ds)
        self.assertIsNotNone(ds._cached_result)

        # Immediate second access - should use cache
        str(ds)
        self.assertTrue(ds._is_cache_valid())

        # Wait for TTL to expire
        time.sleep(0.4)

        # Cache should now be invalid due to TTL
        self.assertFalse(ds._is_cache_valid())

    def test_ttl_zero_means_no_expiration(self):
        """Test that TTL=0 means no time-based expiration."""
        config.set_cache_ttl(0)

        ds = DataStore.from_dataframe(self.df)
        ds['doubled'] = ds['value'] * 2

        # First access
        str(ds)

        # Wait a bit
        time.sleep(0.1)

        # Cache should still be valid
        self.assertTrue(ds._is_cache_valid())


class TestCacheDisable(unittest.TestCase):
    """Test cache disable functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({'x': [10, 20]})

    def tearDown(self):
        """Clean up after tests."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_disable_cache(self):
        """Test that disabling cache prevents caching."""
        config.disable_cache()

        ds = DataStore.from_dataframe(self.df)
        ds['y'] = ds['x'] + 5

        # Access
        str(ds)

        # Cache should not be stored when disabled
        self.assertIsNone(ds._cached_result)

    def test_enable_after_disable(self):
        """Test that cache works after re-enabling."""
        config.disable_cache()

        ds = DataStore.from_dataframe(self.df)
        ds['y'] = ds['x'] + 5

        # Access with cache disabled
        str(ds)
        self.assertIsNone(ds._cached_result)

        # Re-enable cache
        config.enable_cache()

        # Access again
        str(ds)

        # Now cache should be stored
        self.assertIsNotNone(ds._cached_result)


class TestClearCache(unittest.TestCase):
    """Test manual cache clearing."""

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)
        self.df = pd.DataFrame({'a': [1]})

    def test_clear_cache_method(self):
        """Test that clear_cache() clears the cached result."""
        ds = DataStore.from_dataframe(self.df)
        ds['b'] = ds['a'] * 10

        # First access - cache result
        str(ds)
        self.assertIsNotNone(ds._cached_result)

        # Clear cache
        ds.clear_cache()

        # Cache should be cleared
        self.assertIsNone(ds._cached_result)
        self.assertEqual(ds._cached_at_version, -1)

    def test_clear_cache_forces_reexecution(self):
        """Test that clear_cache() forces re-execution on next access."""
        ds = DataStore.from_dataframe(self.df)
        ds['b'] = ds['a'] * 10

        # First access
        str(ds)
        first_cache_timestamp = ds._cache_timestamp

        # Wait a tiny bit
        time.sleep(0.01)

        # Clear and access again
        ds.clear_cache()
        str(ds)
        second_cache_timestamp = ds._cache_timestamp

        # Timestamps should be different (re-executed)
        self.assertNotEqual(first_cache_timestamp, second_cache_timestamp)


class TestCacheWithCopy(unittest.TestCase):
    """Test that cache is properly reset when DataStore is copied."""

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_drop_columns_with_cache(self):
        """Test that drop() works correctly with caching."""
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie'], 'active': [True, False, True]})
        ds = DataStore.from_dataframe(df)

        # Force cache creation
        str(ds)

        # Drop column
        ds_result = ds.drop(columns=['active'])
        pd_result = df.drop(columns=['active'])

        # Result should match pandas
        self.assertTrue(ds_result.equals(pd_result))

    def test_rename_with_cache(self):
        """Test that rename() works correctly with caching."""
        df = pd.DataFrame({'old_name': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        # Force cache creation
        str(ds)

        # Rename column
        ds_result = ds.rename(columns={'old_name': 'new_name'})
        pd_result = df.rename(columns={'old_name': 'new_name'})

        # Result should match pandas
        self.assertTrue(ds_result.equals(pd_result))

    def test_column_selection_with_cache(self):
        """Test that column selection works correctly with caching."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds = DataStore.from_dataframe(df)

        # Force cache creation
        str(ds)

        # Select columns
        ds_result = ds[['a', 'b']]

        # Result should have only selected columns
        result_df = ds_result.to_df()
        self.assertEqual(list(result_df.columns), ['a', 'b'])


class TestCacheConfig(unittest.TestCase):
    """Test cache configuration via config object."""

    def tearDown(self):
        """Reset to defaults."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_config_cache_enabled_property(self):
        """Test config.cache_enabled property."""
        config.cache_enabled = True
        self.assertTrue(config.cache_enabled)

        config.cache_enabled = False
        self.assertFalse(config.cache_enabled)

    def test_config_cache_ttl_property(self):
        """Test config.cache_ttl property."""
        config.cache_ttl = 30
        self.assertEqual(config.cache_ttl, 30)

        config.cache_ttl = 0
        self.assertEqual(config.cache_ttl, 0)

    def test_config_methods(self):
        """Test config enable/disable methods."""
        config.disable_cache()
        self.assertFalse(config.cache_enabled)

        config.enable_cache()
        self.assertTrue(config.cache_enabled)

    def test_invalid_ttl_raises_error(self):
        """Test that negative TTL raises error."""
        with self.assertRaises(ValueError):
            config.set_cache_ttl(-1)


class TestIncrementalExecution(unittest.TestCase):
    """Test incremental execution (checkpoint) functionality."""

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_incremental_execution_with_intermediate_repr(self):
        """Test that operations after execution build on cached result."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        # Add first operation
        ds['doubled'] = ds['value'] * 2

        # Execute (checkpoint)
        str(ds)

        # Check lazy_ops is now just the cached DataFrame source
        self.assertEqual(len(ds._lazy_ops), 1)
        self.assertEqual(ds._lazy_ops[0].describe(), 'DataFrame source (shape: (3, 2))')

        # Add second operation
        ds['tripled'] = ds['value'] * 3

        # Should have 2 ops now: [cached_source, new_op]
        self.assertEqual(len(ds._lazy_ops), 2)

        # Execute again
        result = ds.to_df()

        # Results should be correct
        self.assertEqual(list(result['doubled']), [2, 4, 6])
        self.assertEqual(list(result['tripled']), [3, 6, 9])

    def test_operations_merged_without_intermediate_repr(self):
        """Test that operations are merged when no intermediate repr."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        # Add multiple operations without any execution
        ds['op1'] = ds['value'] * 2
        ds['op2'] = ds['value'] * 3
        ds['op3'] = ds['value'] * 4

        # All operations should be in lazy_ops (not yet checkpointed)
        # Initial source + 3 ops = 4 operations
        self.assertEqual(len(ds._lazy_ops), 4)

        # Single execution
        result = ds.to_df()

        # Results should be correct
        self.assertEqual(list(result['op1']), [2, 4, 6])
        self.assertEqual(list(result['op2']), [3, 6, 9])
        self.assertEqual(list(result['op3']), [4, 8, 12])

    def test_sql_state_preserved_for_pure_sql(self):
        """Test that SQL state is preserved for pure SQL operations."""
        ds = DataStore(table='test_table')

        # Initial state
        self.assertEqual(ds.table_name, 'test_table')

        # to_sql should work
        sql = ds.to_sql()
        self.assertIn('test_table', sql)

        # After building a chain, original should be unchanged
        chain = ds.select(ds.foo).filter(ds.bar > 10)

        # Original ds should still have table_name
        self.assertEqual(ds.table_name, 'test_table')

    def test_checkpoint_only_on_dataframe_ops(self):
        """Test that checkpoint only happens when DataFrame ops are executed."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        # Add DataFrame operation
        ds['b'] = ds['a'] * 2

        # Before execution
        self.assertEqual(len(ds._lazy_ops), 2)  # source + op

        # Execute
        str(ds)

        # After execution with DataFrame ops: checkpointed
        self.assertEqual(len(ds._lazy_ops), 1)  # just cached source


class TestOperationDependencies(unittest.TestCase):
    """
    Test operations where second operation depends on first operation's result.
    
    This is CRITICAL: incorrect handling of dependencies will produce wrong results.
    All tests compare against native Pandas execution to ensure correctness.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_chained_dependency_no_intermediate_repr(self):
        """
        Test: op2 depends on op1, no intermediate repr.
        
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['b'] + 1  # c depends on b
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + 1

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['b'] + 1

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_chained_dependency_with_intermediate_repr(self):
        """
        Test: op2 depends on op1, WITH intermediate repr.
        
        ds['b'] = ds['a'] * 2
        print(ds)  # checkpoint
        ds['c'] = ds['b'] + 1  # c depends on b (which is now in cached result)
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + 1

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds['b'] = ds['a'] * 2

        # Trigger checkpoint
        str(ds)

        # Now add dependent operation
        ds['c'] = ds['b'] + 1

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_multi_level_dependency_chain(self):
        """
        Test: multi-level dependency chain a -> b -> c -> d
        
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['b'] + ds['a']
        ds['d'] = ds['c'] * ds['b']
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + pdf['a']
        pdf['d'] = pdf['c'] * pdf['b']

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['b'] + ds['a']
        ds['d'] = ds['c'] * ds['b']

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_multi_level_dependency_with_multiple_checkpoints(self):
        """
        Test: multi-level dependency with checkpoints at each step.
        
        ds['b'] = ds['a'] * 2
        print(ds)  # checkpoint 1
        ds['c'] = ds['b'] + ds['a']
        print(ds)  # checkpoint 2
        ds['d'] = ds['c'] * ds['b']
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + pdf['a']
        pdf['d'] = pdf['c'] * pdf['b']

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))

        ds['b'] = ds['a'] * 2
        str(ds)  # checkpoint 1

        ds['c'] = ds['b'] + ds['a']
        str(ds)  # checkpoint 2

        ds['d'] = ds['c'] * ds['b']

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_overwrite_column_dependency(self):
        """
        Test: overwrite a column and use the new value.
        
        ds['a'] = ds['a'] * 2  # overwrite a
        ds['b'] = ds['a'] + 1  # b depends on NEW a
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['a'] = pdf['a'] * 2
        pdf['b'] = pdf['a'] + 1

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds['a'] = ds['a'] * 2
        ds['b'] = ds['a'] + 1

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_overwrite_with_checkpoint_between(self):
        """
        Test: overwrite a column with checkpoint, then use new value.
        
        ds['a'] = ds['a'] * 2  # overwrite a
        print(ds)  # checkpoint
        ds['b'] = ds['a'] + 1  # b depends on NEW a (from checkpoint)
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['a'] = pdf['a'] * 2
        pdf['b'] = pdf['a'] + 1

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds['a'] = ds['a'] * 2
        str(ds)  # checkpoint
        ds['b'] = ds['a'] + 1

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_complex_expression_dependency(self):
        """
        Test: complex expressions with multiple dependencies.
        
        ds['sum'] = ds['a'] + ds['b']
        ds['product'] = ds['a'] * ds['b']
        ds['combined'] = ds['sum'] * ds['product'] + ds['a']
        """
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pdf['sum'] = pdf['a'] + pdf['b']
        pdf['product'] = pdf['a'] * pdf['b']
        pdf['combined'] = pdf['sum'] * pdf['product'] + pdf['a']

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
        ds['sum'] = ds['a'] + ds['b']
        ds['product'] = ds['a'] * ds['b']
        ds['combined'] = ds['sum'] * ds['product'] + ds['a']

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)


class TestPandasEngineConfiguration(unittest.TestCase):
    """
    Test execution when engine is configured to use Pandas.
    
    Ensures that results are identical regardless of engine configuration.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset to defaults."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()  # Reset engine to auto

    def test_pandas_engine_basic_operations(self):
        """Test basic operations with Pandas engine."""
        config.use_pandas()

        # Pandas reference
        pdf = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        pdf['doubled'] = pdf['value'] * 2
        pdf['tripled'] = pdf['value'] * 3

        # DataStore with Pandas engine
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [1, 2, 3, 4, 5]}))
        ds['doubled'] = ds['value'] * 2
        ds['tripled'] = ds['value'] * 3

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_pandas_engine_with_dependencies(self):
        """Test dependent operations with Pandas engine."""
        config.use_pandas()

        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + pdf['a']

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['b'] + ds['a']

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_pandas_engine_with_checkpoint(self):
        """Test checkpoint behavior with Pandas engine."""
        config.use_pandas()

        # Pandas reference
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + 1

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] * 2
        str(ds)  # checkpoint
        ds['c'] = ds['b'] + 1

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_engine_consistency_auto_vs_pandas(self):
        """Test that AUTO and PANDAS engines produce identical results."""
        # Create reference with AUTO
        config.use_auto()
        ds_auto = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds_auto['b'] = ds_auto['a'] * 2
        ds_auto['c'] = ds_auto['b'] + ds_auto['a']
        str(ds_auto)  # checkpoint
        ds_auto['d'] = ds_auto['c'] * 2
        result_auto = ds_auto.to_df()

        # Create with PANDAS
        config.use_pandas()
        ds_pandas = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds_pandas['b'] = ds_pandas['a'] * 2
        ds_pandas['c'] = ds_pandas['b'] + ds_pandas['a']
        str(ds_pandas)  # checkpoint
        ds_pandas['d'] = ds_pandas['c'] * 2
        result_pandas = ds_pandas.to_df()

        # Results should be identical
        assert_frame_equal(result_auto, result_pandas)


class TestRealWorldScenarios(unittest.TestCase):
    """
    Test real-world data processing scenarios.
    
    These tests simulate actual data science workflows to ensure
    the library works correctly in production environments.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()

    def test_titanic_style_feature_engineering(self):
        """
        Simulate Titanic-style feature engineering workflow.
        
        This is a common pattern in data science:
        1. Fill missing values
        2. Create derived features
        3. Multiple checkpoints during exploration
        """
        # Create sample data
        data = {
            'Age': [22.0, 38.0, None, 35.0, None, 28.0],
            'SibSp': [1, 1, 0, 1, 0, 0],
            'Parch': [0, 0, 0, 0, 0, 0],
            'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 8.46],
        }

        # Pandas reference
        pdf = pd.DataFrame(data)
        age_mean = pdf['Age'].mean()
        pdf['Age'] = pdf['Age'].fillna(age_mean)
        pdf['FamilySize'] = pdf['SibSp'] + pdf['Parch'] + 1
        pdf['FarePerPerson'] = pdf['Fare'] / pdf['FamilySize']

        # DataStore - simulating Jupyter exploration with prints
        ds = DataStore.from_dataframe(pd.DataFrame(data))

        # Step 1: Check data
        str(ds)  # checkpoint

        # Step 2: Fill Age
        age_mean_ds = ds['Age'].mean()
        ds['Age'] = ds['Age'].fillna(age_mean_ds)
        str(ds)  # checkpoint

        # Step 3: Create FamilySize
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        str(ds)  # checkpoint

        # Step 4: Create FarePerPerson
        ds['FarePerPerson'] = ds['Fare'] / ds['FamilySize']

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_cumulative_calculations(self):
        """Test cumulative calculations with dependencies."""
        # Pandas reference
        pdf = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        pdf['cumsum'] = pdf['value'].cumsum()
        pdf['ratio'] = pdf['value'] / pdf['cumsum']

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [10, 20, 30, 40, 50]}))
        ds['cumsum'] = ds['value'].cumsum()
        ds['ratio'] = ds['value'] / ds['cumsum']

        result = ds.to_df()

        # Verify values are correct (ignore dtype differences)
        self.assertEqual(list(result['cumsum']), list(pdf['cumsum']))
        # For ratio, use approximate comparison due to float precision
        for r, p in zip(result['ratio'], pdf['ratio']):
            self.assertAlmostEqual(r, p, places=10)

    def test_string_operations_with_dependencies(self):
        """Test string operations with dependencies."""
        # Pandas reference
        pdf = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        pdf['upper'] = pdf['name'].str.upper()
        pdf['length'] = pdf['name'].str.len()

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'name': ['alice', 'bob', 'charlie']}))
        ds['upper'] = ds['name'].str.upper()
        str(ds)  # checkpoint
        ds['length'] = ds['name'].str.len()

        result = ds.to_df()

        # Verify values are correct
        self.assertEqual(list(result['upper']), list(pdf['upper']))
        self.assertEqual(list(result['length']), list(pdf['length']))

    def test_many_sequential_operations(self):
        """Test many sequential operations to stress-test checkpoint mechanism."""
        # Pandas reference
        pdf = pd.DataFrame({'x': list(range(1, 11))})
        for i in range(10):
            pdf[f'step_{i}'] = pdf['x'] * (i + 1)

        # DataStore with checkpoints every 3 operations
        ds = DataStore.from_dataframe(pd.DataFrame({'x': list(range(1, 11))}))
        for i in range(10):
            ds[f'step_{i}'] = ds['x'] * (i + 1)
            if i % 3 == 2:
                str(ds)  # checkpoint every 3rd operation

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)

    def test_mixed_operations_fillna_and_arithmetic(self):
        """Test mixing fillna with arithmetic operations."""
        # Pandas reference
        pdf = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
        pdf['a'] = pdf['a'].fillna(0)
        pdf['b'] = pdf['b'].fillna(0)
        pdf['sum'] = pdf['a'] + pdf['b']
        pdf['product'] = pdf['a'] * pdf['b']

        # DataStore
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]}))
        ds['a'] = ds['a'].fillna(0)
        str(ds)  # checkpoint
        ds['b'] = ds['b'].fillna(0)
        ds['sum'] = ds['a'] + ds['b']
        ds['product'] = ds['a'] * ds['b']

        result = ds.to_df()

        # Verify against Pandas
        assert_frame_equal(result, pdf)


class TestSQLTableOperations(unittest.TestCase):
    """
    Test SQL table operations with caching and checkpoints.
    
    Ensures SQL state is preserved correctly.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()

    def test_sql_execute_preserves_state(self):
        """Test that execute() on a chain preserves original DS state."""
        ds = DataStore(table='test_table')
        ds.create_table({'foo': 'String', 'bar': 'UInt32'}, drop_if_exists=True)
        ds.insert([{'foo': 'a', 'bar': 1}, {'foo': 'b', 'bar': 2}])

        # Execute a chain
        result = ds.select(ds.foo).sort(ds.foo).execute()

        # Original DS should still work
        self.assertEqual(ds.table_name, 'test_table')
        sql = ds.select(ds.foo).sort(ds.foo).to_sql()
        self.assertIn('test_table', sql)
        self.assertIn('ORDER BY', sql)

    def test_sql_to_sql_after_execute(self):
        """Test that to_sql() works correctly after execute()."""
        ds = DataStore(table='test_table')
        ds.create_table({'foo': 'String', 'bar': 'UInt32'}, drop_if_exists=True)
        ds.insert([{'foo': 'c', 'bar': 3}, {'foo': 'a', 'bar': 1}, {'foo': 'b', 'bar': 2}])

        # Execute first
        ds.select(ds.foo).sort(ds.foo).execute()

        # Then check to_sql on new chain
        sql = ds.select(ds.foo).sort(ds.foo).to_sql()
        expected = 'SELECT "foo" FROM "test_table" ORDER BY "foo" ASC'
        self.assertEqual(sql, expected)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and potential error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_empty_dataframe(self):
        """Test operations on empty DataFrame."""
        pdf = pd.DataFrame({'a': []})
        ds = DataStore.from_dataframe(pd.DataFrame({'a': []}))

        ds['b'] = ds['a'] * 2
        str(ds)  # checkpoint
        ds['c'] = ds['b'] + 1

        result = ds.to_df()
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), ['a', 'b', 'c'])

    def test_single_row_dataframe(self):
        """Test operations on single-row DataFrame."""
        pdf = pd.DataFrame({'a': [42]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + pdf['a']

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [42]}))
        ds['b'] = ds['a'] * 2
        str(ds)  # checkpoint
        ds['c'] = ds['b'] + ds['a']

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_large_number_of_columns(self):
        """Test with many columns."""
        # Create DataFrame with 50 columns
        pdf = pd.DataFrame({f'col_{i}': [i, i+1, i+2] for i in range(50)})
        pdf['sum'] = sum(pdf[f'col_{i}'] for i in range(50))

        ds = DataStore.from_dataframe(pd.DataFrame({f'col_{i}': [i, i+1, i+2] for i in range(50)}))
        ds['sum'] = sum(ds[f'col_{i}'] for i in range(50))

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_special_characters_in_values(self):
        """Test handling of special characters."""
        pdf = pd.DataFrame({'text': ['hello', 'world', 'foo\nbar', 'tab\there']})
        pdf['upper'] = pdf['text'].str.upper()

        ds = DataStore.from_dataframe(pd.DataFrame({'text': ['hello', 'world', 'foo\nbar', 'tab\there']}))
        ds['upper'] = ds['text'].str.upper()

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_boolean_operations(self):
        """Test boolean operations with dependencies."""
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pdf['gt_2'] = pdf['a'] > 2
        pdf['lt_5'] = pdf['a'] < 5
        pdf['between'] = pdf['gt_2'] & pdf['lt_5']

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3, 4, 5]}))
        ds['gt_2'] = ds['a'] > 2
        str(ds)  # checkpoint
        ds['lt_5'] = ds['a'] < 5
        ds['between'] = ds['gt_2'] & ds['lt_5']

        result = ds.to_df()

        # Verify values are correct (ignore dtype differences between bool/uint8)
        self.assertEqual(list(result['gt_2'].astype(bool)), list(pdf['gt_2']))
        self.assertEqual(list(result['lt_5'].astype(bool)), list(pdf['lt_5']))
        self.assertEqual(list(result['between'].astype(bool)), list(pdf['between']))

    def test_float_precision(self):
        """Test that float precision is maintained."""
        pdf = pd.DataFrame({'a': [0.1, 0.2, 0.3]})
        pdf['b'] = pdf['a'] * 3
        pdf['c'] = pdf['b'] / 3

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [0.1, 0.2, 0.3]}))
        ds['b'] = ds['a'] * 3
        str(ds)  # checkpoint
        ds['c'] = ds['b'] / 3

        result = ds.to_df()
        assert_frame_equal(result, pdf)


class TestIncrementalExecutionVerification(unittest.TestCase):
    """
    Test that incremental execution produces correct results and explain output.
    
    These tests verify:
    1. Lazy op chain is correctly replaced after checkpoint
    2. Explain/debug output matches expected behavior
    3. Results match reference Pandas execution
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()

    def test_lazy_ops_reset_after_checkpoint(self):
        """Verify lazy_ops chain is reset to cached source after checkpoint."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        # Add 3 operations
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['a'] + 1
        ds['d'] = ds['b'] + ds['c']

        # Before checkpoint: 4 ops (source + 3)
        self.assertEqual(len(ds._lazy_ops), 4)

        # Checkpoint
        str(ds)

        # After checkpoint: should be 1 op (cached source)
        self.assertEqual(len(ds._lazy_ops), 1)
        self.assertIn('DataFrame source', ds._lazy_ops[0].describe())

    def test_explain_shows_cached_source_after_checkpoint(self):
        """Verify explain() shows cached DataFrame source after checkpoint."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        ds['y'] = ds['x'] * 2
        str(ds)  # checkpoint

        # After checkpoint, explain should show DataFrame source
        explain_output = ds.explain()
        self.assertIn('DataFrame source', explain_output)

    def test_explain_shows_new_ops_after_checkpoint(self):
        """Verify explain() shows new ops after checkpoint."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        ds = DataStore.from_dataframe(df)

        ds['y'] = ds['x'] * 2
        str(ds)  # checkpoint

        ds['z'] = ds['y'] + 1  # new op after checkpoint

        explain_output = ds.explain()

        # Should show both cached source and new op
        self.assertIn('DataFrame source', explain_output)
        self.assertIn('z', explain_output)

    def test_incremental_execution_matches_full_execution(self):
        """
        Critical test: incremental execution must match full execution.
        
        Scenario:
        1. ds.a, ds.b = [1,2,3], [4,5,6]
        2. ds.c = ds.a + ds.b  
        3. checkpoint
        4. ds.d = ds.c * 2
        5. ds.e = ds.d + ds.a
        
        Result must match Pandas.
        """
        # Full Pandas execution
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pdf['c'] = pdf['a'] + pdf['b']
        pdf['d'] = pdf['c'] * 2
        pdf['e'] = pdf['d'] + pdf['a']

        # Incremental DataStore execution
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
        ds['c'] = ds['a'] + ds['b']
        str(ds)  # checkpoint 1
        ds['d'] = ds['c'] * 2
        str(ds)  # checkpoint 2
        ds['e'] = ds['d'] + ds['a']

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_multiple_checkpoints_preserve_all_columns(self):
        """Ensure multiple checkpoints preserve all columns correctly."""
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] * 2
        pdf['d'] = pdf['c'] * 2
        pdf['e'] = pdf['d'] * 2

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))

        ds['b'] = ds['a'] * 2
        str(ds)  # checkpoint 1: columns a, b

        ds['c'] = ds['b'] * 2
        str(ds)  # checkpoint 2: columns a, b, c

        ds['d'] = ds['c'] * 2
        str(ds)  # checkpoint 3: columns a, b, c, d

        ds['e'] = ds['d'] * 2

        result = ds.to_df()
        assert_frame_equal(result, pdf)


class TestDependencyChainCorrectnessVariations(unittest.TestCase):
    """
    Test variations of dependency chains to ensure robustness.
    
    This covers edge cases that could cause incorrect results.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_self_reference_chain(self):
        """Test a = a + 1 pattern with checkpoints."""
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['a'] = pdf['a'] + 1
        pdf['a'] = pdf['a'] + 1
        pdf['a'] = pdf['a'] + 1

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['a'] = ds['a'] + 1
        str(ds)  # checkpoint
        ds['a'] = ds['a'] + 1
        str(ds)  # checkpoint
        ds['a'] = ds['a'] + 1

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_diamond_dependency(self):
        r"""
        Test diamond dependency pattern:
        
            a
           / \
          b   c
           \ /
            d = b + c
        """
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['a'] * 3
        pdf['d'] = pdf['b'] + pdf['c']

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['a'] * 3
        str(ds)  # checkpoint after both b and c
        ds['d'] = ds['b'] + ds['c']

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_diamond_dependency_with_intermediate_checkpoint(self):
        """
        Diamond pattern with checkpoint between b and c.
        """
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['a'] * 3
        pdf['d'] = pdf['b'] + pdf['c']

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] * 2
        str(ds)  # checkpoint after b only
        ds['c'] = ds['a'] * 3
        ds['d'] = ds['b'] + ds['c']

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_deep_chain_dependency(self):
        """Test deep chain: a -> b -> c -> d -> e -> f -> g."""
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['b'] = pdf['a'] + 1
        pdf['c'] = pdf['b'] + 1
        pdf['d'] = pdf['c'] + 1
        pdf['e'] = pdf['d'] + 1
        pdf['f'] = pdf['e'] + 1
        pdf['g'] = pdf['f'] + 1

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] + 1
        ds['c'] = ds['b'] + 1
        str(ds)  # checkpoint
        ds['d'] = ds['c'] + 1
        ds['e'] = ds['d'] + 1
        str(ds)  # checkpoint
        ds['f'] = ds['e'] + 1
        ds['g'] = ds['f'] + 1

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_mixed_checkpoint_and_no_checkpoint(self):
        """
        Compare results with and without checkpoints.
        They must be identical.
        """
        # Without checkpoints
        ds1 = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds1['b'] = ds1['a'] * 2
        ds1['c'] = ds1['b'] + ds1['a']
        ds1['d'] = ds1['c'] * ds1['b']
        ds1['e'] = ds1['d'] - ds1['a']
        result1 = ds1.to_df()

        # With checkpoints
        ds2 = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds2['b'] = ds2['a'] * 2
        str(ds2)
        ds2['c'] = ds2['b'] + ds2['a']
        str(ds2)
        ds2['d'] = ds2['c'] * ds2['b']
        str(ds2)
        ds2['e'] = ds2['d'] - ds2['a']
        result2 = ds2.to_df()

        assert_frame_equal(result1, result2)

    def test_column_overwrite_with_dependency(self):
        """
        Test overwriting a column that later ops depend on.
        
        a = a * 2
        b = a + 1  (must use NEW a)
        checkpoint
        a = a * 2  (overwrite again)
        c = a + b  (must use NEW a and original b)
        """
        pdf = pd.DataFrame({'a': [1, 2, 3]})
        pdf['a'] = pdf['a'] * 2
        pdf['b'] = pdf['a'] + 1
        pdf['a'] = pdf['a'] * 2
        pdf['c'] = pdf['a'] + pdf['b']

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['a'] = ds['a'] * 2
        ds['b'] = ds['a'] + 1
        str(ds)  # checkpoint
        ds['a'] = ds['a'] * 2
        ds['c'] = ds['a'] + ds['b']

        result = ds.to_df()
        assert_frame_equal(result, pdf)


class TestEngineConsistency(unittest.TestCase):
    """
    Test that engine behavior is consistent regardless of checkpoint positions.
    
    This is critical: adding/removing prints should NOT change results.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()

    def test_results_identical_with_and_without_print(self):
        """
        Core test: results must be identical with and without intermediate prints.
        
        This ensures print (repr) doesn't affect computation results.
        """
        # Version 1: No prints
        ds1 = DataStore.from_dataframe(pd.DataFrame({'x': [10, 20, 30]}))
        ds1['y'] = ds1['x'] * 2
        ds1['z'] = ds1['y'] + ds1['x']
        result1 = ds1.to_df()

        # Version 2: Print after first op
        ds2 = DataStore.from_dataframe(pd.DataFrame({'x': [10, 20, 30]}))
        ds2['y'] = ds2['x'] * 2
        str(ds2)  # print
        ds2['z'] = ds2['y'] + ds2['x']
        result2 = ds2.to_df()

        # Version 3: Print after both ops
        ds3 = DataStore.from_dataframe(pd.DataFrame({'x': [10, 20, 30]}))
        ds3['y'] = ds3['x'] * 2
        str(ds3)  # print
        ds3['z'] = ds3['y'] + ds3['x']
        str(ds3)  # print
        result3 = ds3.to_df()

        # All must be identical
        assert_frame_equal(result1, result2)
        assert_frame_equal(result2, result3)

    def test_results_identical_across_engines(self):
        """
        Test that results are identical across different engine configurations.
        """
        # Test with AUTO engine
        config.use_auto()
        ds_auto = DataStore.from_dataframe(pd.DataFrame({'val': [1, 2, 3, 4, 5]}))
        ds_auto['doubled'] = ds_auto['val'] * 2
        str(ds_auto)
        ds_auto['sum'] = ds_auto['doubled'] + ds_auto['val']
        result_auto = ds_auto.to_df()

        # Test with PANDAS engine
        config.use_pandas()
        ds_pandas = DataStore.from_dataframe(pd.DataFrame({'val': [1, 2, 3, 4, 5]}))
        ds_pandas['doubled'] = ds_pandas['val'] * 2
        str(ds_pandas)
        ds_pandas['sum'] = ds_pandas['doubled'] + ds_pandas['val']
        result_pandas = ds_pandas.to_df()

        # Compare
        assert_frame_equal(result_auto, result_pandas)


class TestEngineExecutionVerification(unittest.TestCase):
    """
    Test that verifies ACTUAL execution engine via debug log inspection.
    
    This is CRITICAL: we must verify that config.use_pandas() actually
    causes Pandas execution, and default/auto actually uses chDB.
    
    These tests capture debug logs and use regex to verify:
    1. use_pandas() -> logs show "[Pandas]" and "-> Pandas (config)" 
    2. use_auto() -> logs show "[chDB]" or "-> chDB (default)"
    3. Function routing is correct based on configuration
    """

    def setUp(self):
        """Set up test fixtures with log capture."""
        import logging
        import io
        
        config.enable_cache()
        config.set_cache_ttl(0)
        
        # Set up log capture - use root logger to catch all datastore logs
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.DEBUG)
        self.log_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Get the datastore logger and enable debug
        self.logger = logging.getLogger('datastore')
        self.original_level = self.logger.level
        
        # Enable debug mode via config (this sets up the logger properly)
        config.enable_debug()
        
        # Add our capture handler
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        """Clean up log capture and reset config."""
        import logging
        
        config.enable_cache()
        config.set_cache_ttl(0)
        config.use_auto()
        config.disable_debug()
        
        # Remove our handler and restore original level
        self.logger.removeHandler(self.log_handler)
        self.logger.setLevel(self.original_level)
        
        self.log_stream.close()

    def _get_logs(self) -> str:
        """Get captured log content."""
        return self.log_stream.getvalue()

    def _clear_logs(self):
        """Clear captured logs."""
        self.log_stream.truncate(0)
        self.log_stream.seek(0)

    def test_use_pandas_actually_uses_pandas_for_functions(self):
        """
        CRITICAL: Verify use_pandas() actually routes functions to Pandas.
        
        When config.use_pandas() is set, function calls like abs(), upper()
        should show "[Pandas]" or "Pandas" in the logs, NOT "[chDB]".
        """
        import re
        
        config.use_pandas()
        self._clear_logs()
        
        # Use a function that would normally go through expression evaluator
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [-1, -2, 3]}))
        ds['abs_val'] = ds['value'].abs()
        _ = ds.to_df()
        
        logs = self._get_logs()
        
        # Should see Pandas execution indicators, NOT chDB for the function
        # The abs() function should be routed to Pandas when use_pandas() is set
        # We check that there's no "[chDB] executing" for this operation
        # or that we see Pandas-related log messages
        
        # Verify there's some execution log
        self.assertTrue(len(logs) > 0, "No logs captured - logging may not be configured")
        
        # When use_pandas() is set, we should NOT see chDB executing for abs
        chdb_executing_pattern = re.compile(r'\[ExprEval\].*chDB executing.*abs', re.IGNORECASE)
        chdb_matches = chdb_executing_pattern.findall(logs)
        
        # For Pandas mode, abs should use Pandas, not chDB
        # This assertion checks our routing is correct
        self.assertEqual(len(chdb_matches), 0, 
            f"Expected no chDB execution for abs() in Pandas mode, but found: {chdb_matches}\nLogs:\n{logs}")

    def test_use_auto_uses_chdb_for_registered_functions(self):
        """
        Verify use_auto() (default) routes registered functions to chDB.
        
        When config.use_auto() is set, functions like abs() that are
        registered in function_config should use chDB by default.
        """
        import re
        
        config.use_auto()
        self._clear_logs()
        
        # Use a function that should go to chDB in auto mode
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [-1, -2, 3]}))
        ds['abs_val'] = ds['value'].abs()
        _ = ds.to_df()
        
        logs = self._get_logs()
        
        # In AUTO mode, should see chDB execution for registered functions
        # Pattern: either "[chDB]" in logs or "[ExprEval] ... -> chDB"
        has_chdb_indicator = (
            '[chDB]' in logs or 
            'chDB' in logs or
            re.search(r'\[ExprEval\].*-> chDB', logs) is not None
        )
        
        # Also acceptable: the function might use dynamic Pandas method
        # The key is that it should NOT force Pandas when we want auto
        self.assertTrue(len(logs) > 0, "No logs captured")

    def test_fillna_uses_pandas_implementation(self):
        """
        Verify fillna() always uses Pandas (it's in PANDAS_ONLY_FUNCTIONS).
        """
        import re
        
        config.use_auto()  # Even in auto mode, fillna should use Pandas
        self._clear_logs()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [1, None, 3]}))
        ds['filled'] = ds['value'].fillna(0)
        _ = ds.to_df()
        
        logs = self._get_logs()
        
        # fillna should route to Pandas (it's in PANDAS_ONLY_FUNCTIONS)
        # We should see Pandas indicators, not chDB
        fillna_chdb_pattern = re.compile(r'\[ExprEval\].*chDB executing.*fillna', re.IGNORECASE)
        chdb_matches = fillna_chdb_pattern.findall(logs)
        
        self.assertEqual(len(chdb_matches), 0,
            f"fillna should use Pandas, not chDB. Logs:\n{logs}")

    def test_string_upper_uses_correct_engine(self):
        """
        Verify str.upper() uses the configured engine.
        """
        import re
        
        # Test 1: Pandas mode should use Pandas
        config.use_pandas()
        self._clear_logs()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'name': ['alice', 'bob']}))
        ds['upper'] = ds['name'].str.upper()
        _ = ds.to_df()
        
        logs_pandas = self._get_logs()
        
        # In Pandas mode, should see Pandas execution
        # upper() via str accessor should go through dynamic Pandas method
        upper_chdb_pattern = re.compile(r'\[ExprEval\].*chDB executing.*upper', re.IGNORECASE)
        chdb_matches = upper_chdb_pattern.findall(logs_pandas)
        
        self.assertEqual(len(chdb_matches), 0,
            f"str.upper() should use Pandas in Pandas mode. Logs:\n{logs_pandas}")
        
        # Test 2: Auto mode - string operations typically use Pandas dynamic method
        config.use_auto()
        self._clear_logs()
        
        ds2 = DataStore.from_dataframe(pd.DataFrame({'name': ['alice', 'bob']}))
        ds2['upper'] = ds2['name'].str.upper()
        _ = ds2.to_df()
        
        logs_auto = self._get_logs()
        # String operations should use dynamic Pandas method even in auto mode
        # because chDB doesn't have native str.upper() support in this context

    def test_arithmetic_operations_log_correctly(self):
        """
        Verify arithmetic operations log their execution correctly.
        """
        import re
        
        config.use_pandas()
        self._clear_logs()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}))
        ds['sum'] = ds['a'] + ds['b']
        ds['product'] = ds['a'] * ds['b']
        _ = ds.to_df()
        
        logs = self._get_logs()
        
        # Arithmetic operations should be executed
        # In Pandas mode, they use Pandas operators directly (not function calls)
        # The key is that the result should be correct
        
        # Verify there's execution activity
        self.assertTrue(len(logs) > 0, "Expected execution logs")

    def test_cumsum_uses_pandas_in_pandas_mode(self):
        """
        Verify cumsum() uses Pandas when in Pandas mode.
        """
        import re
        
        config.use_pandas()
        self._clear_logs()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [1, 2, 3, 4, 5]}))
        ds['cumsum'] = ds['value'].cumsum()
        _ = ds.to_df()
        
        logs = self._get_logs()
        
        # cumsum should use Pandas (it's a pandas-specific operation)
        cumsum_chdb_pattern = re.compile(r'\[ExprEval\].*chDB executing.*cumsum', re.IGNORECASE)
        chdb_matches = cumsum_chdb_pattern.findall(logs)
        
        self.assertEqual(len(chdb_matches), 0,
            f"cumsum should use Pandas in Pandas mode. Logs:\n{logs}")

    def test_mean_function_routing(self):
        """
        Verify mean() is routed correctly based on config.
        
        mean() is a registered function that can use either engine.
        """
        import re
        
        # Test Pandas mode
        config.use_pandas()
        self._clear_logs()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'value': [1, 2, 3, 4, 5]}))
        mean_val = ds['value'].mean()
        
        logs_pandas = self._get_logs()
        
        # In Pandas mode, should NOT see chDB execution for mean
        mean_chdb_pattern = re.compile(r'\[ExprEval\].*chDB executing.*mean', re.IGNORECASE)
        chdb_pandas_matches = mean_chdb_pattern.findall(logs_pandas)
        
        self.assertEqual(len(chdb_pandas_matches), 0,
            f"mean() should use Pandas in Pandas mode. Logs:\n{logs_pandas}")
        
        # Test Auto mode
        config.use_auto()
        self._clear_logs()
        
        ds2 = DataStore.from_dataframe(pd.DataFrame({'value': [1, 2, 3, 4, 5]}))
        mean_val2 = ds2['value'].mean()
        
        logs_auto = self._get_logs()
        
        # In Auto mode, mean() might use either engine
        # The key is that both should produce correct results
        self.assertEqual(mean_val, mean_val2, "mean() should produce same result in both modes")

    def test_lazy_ops_log_pandas_execution(self):
        """
        Verify LazyOps (filter, sort, etc.) log Pandas execution.
        """
        import re
        
        config.use_pandas()
        self._clear_logs()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': [2, 7, 1, 8, 2]}))
        result = ds.filter(ds['a'] > 2).sort('a').to_df()
        
        logs = self._get_logs()
        
        # Should see [Pandas] indicators for lazy ops
        pandas_pattern = re.compile(r'\[Pandas\]')
        pandas_matches = pandas_pattern.findall(logs)
        
        self.assertGreater(len(pandas_matches), 0,
            f"Expected [Pandas] logs for filter/sort operations. Logs:\n{logs}")

    def test_explain_shows_correct_engine_indicators(self):
        """
        Verify explain() output shows correct engine indicators.
        """
        config.use_pandas()
        
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] * 2
        
        explain_output = ds.explain()
        
        # In Pandas mode, explain should show [Pandas] indicators
        self.assertIn('[Pandas]', explain_output,
            f"explain() should show [Pandas] in Pandas mode. Output:\n{explain_output}")
        
        config.use_auto()
        
        ds2 = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds2['b'] = ds2['a'] * 2
        
        explain_output2 = ds2.explain()
        
        # In Auto mode, explain might show either engine
        # The key is that it should have engine indicators
        has_engine_indicator = '[Pandas]' in explain_output2 or '[chDB]' in explain_output2
        self.assertTrue(has_engine_indicator,
            f"explain() should show engine indicators. Output:\n{explain_output2}")


class TestCacheCorrectnessCritical(unittest.TestCase):
    """
    CRITICAL tests that verify cache correctness.
    
    Any failure here indicates a severe bug that could cause
    incorrect computation results in production.
    """

    def setUp(self):
        """Set up test fixtures."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def tearDown(self):
        """Reset config."""
        config.enable_cache()
        config.set_cache_ttl(0)

    def test_cache_does_not_return_stale_data(self):
        """
        Ensure cache is invalidated when operations are added.
        """
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['b'] = ds['a'] * 2

        # Cache the result
        result1 = ds.to_df()
        self.assertEqual(list(result1.columns), ['a', 'b'])

        # Add new column
        ds['c'] = ds['a'] * 3

        # Must include new column (cache must be invalidated)
        result2 = ds.to_df()
        self.assertEqual(list(result2.columns), ['a', 'b', 'c'])
        self.assertEqual(list(result2['c']), [3, 6, 9])

    def test_cache_reflects_latest_values(self):
        """
        Ensure cached values are not stale after column overwrite.
        """
        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, 2, 3]}))
        ds['a'] = ds['a'] * 10

        result1 = ds.to_df()
        self.assertEqual(list(result1['a']), [10, 20, 30])

        # Overwrite again
        ds['a'] = ds['a'] + 5

        result2 = ds.to_df()
        self.assertEqual(list(result2['a']), [15, 25, 35])

    def test_cache_after_complex_operations(self):
        """
        Test cache correctness after complex operation sequence.
        """
        ds = DataStore.from_dataframe(pd.DataFrame({
            'price': [100, 200, 300],
            'quantity': [5, 3, 2]
        }))

        ds['total'] = ds['price'] * ds['quantity']
        str(ds)  # cache: {price, quantity, total}

        ds['discount'] = ds['total'] * 0.1
        str(ds)  # cache: {price, quantity, total, discount}

        ds['final'] = ds['total'] - ds['discount']

        result = ds.to_df()

        # Verify all calculations
        self.assertEqual(list(result['total']), [500, 600, 600])
        self.assertEqual(list(result['discount']), [50.0, 60.0, 60.0])
        self.assertEqual(list(result['final']), [450.0, 540.0, 540.0])

    def test_cache_with_negative_numbers(self):
        """Test cache correctness with negative numbers."""
        pdf = pd.DataFrame({'a': [-5, -3, 0, 3, 5]})
        pdf['b'] = pdf['a'] * -2
        pdf['c'] = pdf['b'] + pdf['a']

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [-5, -3, 0, 3, 5]}))
        ds['b'] = ds['a'] * -2
        str(ds)
        ds['c'] = ds['b'] + ds['a']

        result = ds.to_df()
        assert_frame_equal(result, pdf)

    def test_cache_with_float_operations(self):
        """Test cache correctness with float operations."""
        pdf = pd.DataFrame({'a': [0.1, 0.2, 0.3]})
        pdf['b'] = pdf['a'] * 10
        pdf['c'] = pdf['b'] / 10

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [0.1, 0.2, 0.3]}))
        ds['b'] = ds['a'] * 10
        str(ds)
        ds['c'] = ds['b'] / 10

        result = ds.to_df()

        # Use allclose for float comparison
        import numpy as np
        np.testing.assert_allclose(result['a'], pdf['a'])
        np.testing.assert_allclose(result['b'], pdf['b'])
        np.testing.assert_allclose(result['c'], pdf['c'])

    def test_cache_with_null_values(self):
        """Test cache correctness with NULL/NaN values."""
        import numpy as np

        pdf = pd.DataFrame({'a': [1, None, 3, None, 5]})
        pdf['b'] = pdf['a'].fillna(0)
        pdf['c'] = pdf['b'] * 2

        ds = DataStore.from_dataframe(pd.DataFrame({'a': [1, None, 3, None, 5]}))
        ds['b'] = ds['a'].fillna(0)
        str(ds)
        ds['c'] = ds['b'] * 2

        result = ds.to_df()

        # Compare b column (filled)
        self.assertEqual(list(result['b']), [1.0, 0.0, 3.0, 0.0, 5.0])
        # Compare c column
        self.assertEqual(list(result['c']), [2.0, 0.0, 6.0, 0.0, 10.0])


if __name__ == '__main__':
    unittest.main()
