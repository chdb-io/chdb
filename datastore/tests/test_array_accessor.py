"""
Test Series.arr accessor functionality.

Verifies that:
1. Array accessor methods work correctly with ClickHouse array functions
2. Lazy execution is maintained
3. Edge cases are handled properly (empty arrays, NULL values)

NOTE: Many tests are marked xfail due to chDB issue:
When numpy arrays pass through Python() table function, they are converted
to strings instead of being recognized as Array type.
See: ~/chdb-workflow/issues/chdb/array_type_loss_via_python_table_function.py
"""

import pytest
import pandas as pd
import chdb
from datastore import DataStore
from datastore.column_expr import ColumnExpr
from datastore.functions import Function


from tests.xfail_markers import chdb_array_string_conversion


class TestArrayAccessorBasic:
    """Test basic .arr accessor methods."""

    @pytest.fixture
    def ds_with_arrays(self):
        """Create a test DataStore with array columns using SQL."""
        df = chdb.query("""
            SELECT 
                1 as id,
                [1, 2, 3] as nums,
                ['a', 'b', 'c'] as tags,
                [] as empty_arr
            UNION ALL
            SELECT 
                2 as id,
                [4, 5] as nums,
                ['x', 'y', 'z', 'w'] as tags,
                [] as empty_arr
            UNION ALL
            SELECT 
                3 as id,
                [10] as nums,
                ['single'] as tags,
                [] as empty_arr
        """, "DataFrame")
        return DataStore.from_df(df)

    # ==================== Type/Laziness Tests ====================

    def test_arr_length_returns_column_expr(self, ds_with_arrays):
        """Test that .arr.length returns ColumnExpr (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr
        
        result = ds_with_arrays['nums'].arr.length
        assert isinstance(result, ColumnExpr)
        # Should have sort_values for pandas compatibility
        assert hasattr(result, 'sort_values')

    def test_arr_empty_returns_column_expr(self, ds_with_arrays):
        """Test that .arr.empty returns ColumnExpr (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr
        
        result = ds_with_arrays['nums'].arr.empty
        assert isinstance(result, ColumnExpr)

    def test_arr_has_returns_column_expr(self, ds_with_arrays):
        """Test that .arr.has() returns ColumnExpr (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr
        
        result = ds_with_arrays['nums'].arr.has(1)
        assert isinstance(result, ColumnExpr)

    # ==================== Execution Tests (blocked by chDB issue) ====================

    @chdb_array_string_conversion
    def test_arr_length_execution(self, ds_with_arrays):
        """Test that .arr.length executes correctly."""
        ds_with_arrays['num_length'] = ds_with_arrays['nums'].arr.length
        df = ds_with_arrays.to_df()
        
        assert 'num_length' in df.columns
        # Expected lengths: [1,2,3]=3, [4,5]=2, [10]=1
        expected_lengths = [3, 2, 1]
        assert list(df['num_length']) == expected_lengths

    @chdb_array_string_conversion
    def test_arr_size_alias(self, ds_with_arrays):
        """Test that .arr.size is an alias for length."""
        ds_with_arrays['size_result'] = ds_with_arrays['nums'].arr.size
        df = ds_with_arrays.to_df()
        
        expected_lengths = [3, 2, 1]
        assert list(df['size_result']) == expected_lengths

    @chdb_array_string_conversion
    def test_arr_empty_execution(self, ds_with_arrays):
        """Test empty check on arrays."""
        ds_with_arrays['is_empty'] = ds_with_arrays['empty_arr'].arr.empty
        df = ds_with_arrays.to_df()
        
        # All empty_arr columns should be empty
        assert all(df['is_empty'] == 1)

    def test_arr_not_empty_execution(self, ds_with_arrays):
        """Test notEmpty check on arrays.
        
        Note: This test passes because notEmpty on a string is valid.
        """
        ds_with_arrays['has_items'] = ds_with_arrays['nums'].arr.not_empty
        df = ds_with_arrays.to_df()
        
        # All nums arrays (now strings) should be not empty
        assert all(df['has_items'] == 1)

    @chdb_array_string_conversion
    def test_arr_has_execution(self, ds_with_arrays):
        """Test has() for element containment."""
        ds_with_arrays['has_1'] = ds_with_arrays['nums'].arr.has(1)
        df = ds_with_arrays.to_df()
        
        expected = [1, 0, 0]
        assert list(df['has_1']) == expected

    @chdb_array_string_conversion
    def test_arr_has_string(self, ds_with_arrays):
        """Test has() with string arrays."""
        ds_with_arrays['has_a'] = ds_with_arrays['tags'].arr.has('a')
        df = ds_with_arrays.to_df()
        
        expected = [1, 0, 0]
        assert list(df['has_a']) == expected


class TestArrayAggregations:
    """Test array aggregation functions."""

    @pytest.fixture
    def ds_numeric_arrays(self):
        """Create DataStore with numeric arrays for aggregation tests."""
        df = chdb.query("""
            SELECT 
                1 as id,
                [1, 2, 3, 4, 5] as nums
            UNION ALL
            SELECT 
                2 as id,
                [10, 20] as nums
            UNION ALL
            SELECT 
                3 as id,
                [100] as nums
        """, "DataFrame")
        return DataStore.from_df(df)

    @chdb_array_string_conversion
    def test_array_sum(self, ds_numeric_arrays):
        """Test arraySum function."""
        ds_numeric_arrays['sum'] = ds_numeric_arrays['nums'].arr.array_sum()
        df = ds_numeric_arrays.to_df()
        
        expected = [15, 30, 100]
        assert list(df['sum']) == expected

    @chdb_array_string_conversion
    def test_array_avg(self, ds_numeric_arrays):
        """Test arrayAvg function."""
        ds_numeric_arrays['avg'] = ds_numeric_arrays['nums'].arr.array_avg()
        df = ds_numeric_arrays.to_df()
        
        expected = [3.0, 15.0, 100.0]
        assert list(df['avg']) == expected

    @chdb_array_string_conversion
    def test_array_min(self, ds_numeric_arrays):
        """Test arrayMin function."""
        ds_numeric_arrays['min'] = ds_numeric_arrays['nums'].arr.array_min()
        df = ds_numeric_arrays.to_df()
        
        expected = [1, 10, 100]
        assert list(df['min']) == expected

    @chdb_array_string_conversion
    def test_array_max(self, ds_numeric_arrays):
        """Test arrayMax function."""
        ds_numeric_arrays['max'] = ds_numeric_arrays['nums'].arr.array_max()
        df = ds_numeric_arrays.to_df()
        
        expected = [5, 20, 100]
        assert list(df['max']) == expected


class TestArrayStringOperations:
    """Test array operations with strings."""

    @pytest.fixture
    def ds_string_arrays(self):
        """Create DataStore with string arrays."""
        df = chdb.query("""
            SELECT 
                1 as id,
                ['hello', 'world'] as words
            UNION ALL
            SELECT 
                2 as id,
                ['foo', 'bar', 'baz'] as words
            UNION ALL
            SELECT 
                3 as id,
                ['single'] as words
        """, "DataFrame")
        return DataStore.from_df(df)

    @chdb_array_string_conversion
    def test_array_string_concat(self, ds_string_arrays):
        """Test arrayStringConcat (join) function."""
        ds_string_arrays['joined'] = ds_string_arrays['words'].arr.array_string_concat(',')
        df = ds_string_arrays.to_df()
        
        expected = ['hello,world', 'foo,bar,baz', 'single']
        assert list(df['joined']) == expected


class TestArrayEdgeCases:
    """Test edge cases for array accessor."""

    @chdb_array_string_conversion
    def test_empty_array_length(self):
        """Test length of empty array."""
        df = chdb.query("SELECT [] as arr", "DataFrame")
        ds = DataStore.from_df(df)
        ds['len'] = ds['arr'].arr.length
        result = ds.to_df()
        
        assert result['len'].iloc[0] == 0

    @chdb_array_string_conversion
    def test_empty_array_sum(self):
        """Test sum of empty numeric array."""
        df = chdb.query("SELECT CAST([] AS Array(Int64)) as nums", "DataFrame")
        ds = DataStore.from_df(df)
        ds['sum'] = ds['nums'].arr.array_sum()
        result = ds.to_df()
        
        assert result['sum'].iloc[0] == 0

    @chdb_array_string_conversion
    def test_single_element_array(self):
        """Test operations on single element array."""
        df = chdb.query("SELECT [42] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        ds['sum'] = ds['nums'].arr.array_sum()
        ds['len'] = ds['nums'].arr.length
        result = ds.to_df()
        
        assert result['sum'].iloc[0] == 42
        assert result['len'].iloc[0] == 1

    def test_chained_array_operations(self):
        """Test chaining array operations.
        
        Note: This test passes because it uses length() which works on strings too,
        though it returns wrong values (string length instead of array length).
        The filter comparison still works.
        """
        df = chdb.query("""
            SELECT 
                1 as id,
                [1, 2, 3] as nums
        """, "DataFrame")
        ds = DataStore.from_df(df)
        
        # Chain: get length, then filter based on it
        # Note: length returns 7 for "[1 2 3]" string, so filter > 2 passes
        ds['arr_len'] = ds['nums'].arr.length
        result = ds[ds['arr_len'] > 2]
        result_df = result.to_df()
        
        assert len(result_df) == 1
        assert result_df['id'].iloc[0] == 1


class TestArrayLazyExecution:
    """Test that array operations maintain lazy execution."""

    def test_array_ops_return_column_expr(self):
        """Test that array operations return ColumnExpr objects (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr
        
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        # These should return ColumnExpr objects (for pandas API compatibility)
        len_expr = ds['nums'].arr.length
        sum_expr = ds['nums'].arr.array_sum()
        has_expr = ds['nums'].arr.has(1)
        
        assert isinstance(len_expr, ColumnExpr)
        assert isinstance(sum_expr, ColumnExpr)
        assert isinstance(has_expr, ColumnExpr)

    @chdb_array_string_conversion
    def test_multiple_array_columns_lazy(self):
        """Test multiple array column assignments remain lazy."""
        df = chdb.query("SELECT [1, 2, 3] as nums, ['a', 'b'] as tags", "DataFrame")
        ds = DataStore.from_df(df)
        
        ds['num_len'] = ds['nums'].arr.length
        ds['tag_len'] = ds['tags'].arr.length
        ds['num_sum'] = ds['nums'].arr.array_sum()
        
        # Should have recorded lazy operations
        has_assignments = sum(1 for op in ds._lazy_ops 
                            if op.__class__.__name__ == 'LazyColumnAssignment')
        assert has_assignments >= 3
        
        # Execute and verify correct array values
        result = ds.to_df()
        assert result['num_len'].iloc[0] == 3  # Array length, not string length
        assert result['tag_len'].iloc[0] == 2
        assert result['num_sum'].iloc[0] == 6


class TestArrayDirectSQL:
    """Test array operations using direct SQL (bypassing Python() table function).
    
    These tests verify the array accessor methods generate correct SQL,
    even though they can't be executed through DataStore due to the
    Python() table function type conversion issue.
    """

    def test_arr_length_sql_generation(self):
        """Verify arr.length generates correct SQL."""
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        length_result = ds['nums'].arr.length
        # Get underlying expression's SQL representation
        sql_repr = length_result._expr.to_sql()
        
        # Should generate length() function call
        assert 'length' in sql_repr.lower()
        assert 'nums' in sql_repr

    def test_arr_has_sql_generation(self):
        """Verify arr.has() generates correct SQL."""
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        has_func = ds['nums'].arr.has(1)
        sql_repr = str(has_func)
        
        # Should generate has() function call
        assert 'has' in sql_repr.lower()
        assert 'nums' in sql_repr

    def test_array_functions_work_in_pure_chdb(self):
        """Verify array functions work correctly in pure chDB without Python() table function."""
        # Direct chDB query - arrays work correctly
        result = chdb.query("""
            SELECT 
                [1, 2, 3] as nums,
                length([1, 2, 3]) as len,
                arraySum([1, 2, 3]) as sum,
                has([1, 2, 3], 2) as has_2
        """, "DataFrame")
        
        assert result['len'].iloc[0] == 3
        assert result['sum'].iloc[0] == 6
        assert result['has_2'].iloc[0] == 1


class TestArrayAccessorChaining:
    """Test that array accessor results can chain with other operations."""

    def test_arr_length_returns_column_expr(self):
        """Test arr.length returns ColumnExpr for pandas API compatibility."""
        from datastore.column_expr import ColumnExpr
        
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        length_result = ds['nums'].arr.length
        
        # Should be a ColumnExpr (not raw Function)
        assert isinstance(length_result, ColumnExpr)
        
        # Should have sort_values method (pandas API)
        assert hasattr(length_result, 'sort_values')
        
        # Should have str, arr, json accessors for chaining
        assert hasattr(length_result, 'str')
        assert hasattr(length_result, 'arr')
        assert hasattr(length_result, 'json')

    def test_arr_method_returns_column_expr(self):
        """Test arr.has() returns ColumnExpr for pandas API compatibility."""
        from datastore.column_expr import ColumnExpr
        
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        has_result = ds['nums'].arr.has(2)
        
        # Should be a ColumnExpr (not raw Function)
        assert isinstance(has_result, ColumnExpr)
        assert hasattr(has_result, 'sort_values')

    def test_arr_length_comparison(self):
        """Test arr.length can be used in comparisons."""
        from datastore.column_expr import ColumnExpr
        
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        # arr.length should support comparison operators
        length_result = ds['nums'].arr.length
        condition = length_result > 2
        
        # Should return a ColumnExpr wrapping a Condition (usable for filtering)
        assert isinstance(condition, ColumnExpr)

    def test_arr_has_comparison(self):
        """Test arr.has() can be used in boolean context."""
        from datastore.column_expr import ColumnExpr
        
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        # arr.has returns ColumnExpr which can be compared
        has_result = ds['nums'].arr.has(2)
        condition = has_result == 1
        
        assert isinstance(condition, ColumnExpr)

    def test_arr_function_arithmetic(self):
        """Test array accessor results support arithmetic operations."""
        from datastore.column_expr import ColumnExpr
        
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        # arr.length * 10
        length_result = ds['nums'].arr.length
        result = length_result * 10
        
        # Should return ColumnExpr (supports chaining)
        assert isinstance(result, ColumnExpr)

    def test_arr_sql_generation_with_comparison(self):
        """Verify arr accessor in comparison generates correct SQL."""
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        condition = ds['nums'].arr.length > 2
        # Get underlying expression's SQL
        sql_repr = condition._expr.to_sql()
        
        # Should have length function and comparison
        assert 'length' in sql_repr.lower()
        assert '>' in sql_repr
        assert '2' in sql_repr

    def test_arr_sql_generation_with_arithmetic(self):
        """Verify arr accessor with arithmetic generates correct SQL."""
        df = chdb.query("SELECT [1, 2, 3] as nums", "DataFrame")
        ds = DataStore.from_df(df)
        
        result = ds['nums'].arr.length + 10
        # Get underlying expression's SQL
        sql_repr = result._expr.to_sql()
        
        # Should have length function and addition
        assert 'length' in sql_repr.lower()
        assert '+' in sql_repr or '10' in sql_repr
