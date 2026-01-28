"""
Test Series.str accessor lazy execution behavior.

Verifies that:
1. SQL-based string methods remain lazy until execution
2. Executing methods (cat, extractall, get_dummies, partition, rpartition)
   explicitly execute and return proper results
3. Mixed usage patterns work correctly
"""

import pytest
import pandas as pd
from datastore import DataStore
from datastore.column_expr import ColumnExpr, ColumnExprStringAccessor


class TestStrAccessorLazy:
    """Test that .str accessor methods maintain lazy evaluation where appropriate."""

    @pytest.fixture
    def ds(self):
        """Create a test DataStore from DataFrame."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['John|Doe', 'Jane|Smith', 'Bob|Brown'],
                'text': ['hello world', 'foo bar baz', 'test string'],
                'numbers': ['abc123def456', 'xyz789', 'test123test456'],
            }
        )
        return DataStore.from_df(df)

    # ==================== SQL-Based Methods (Lazy) ====================

    def test_upper_is_lazy(self, ds):
        """Test that .str.upper() returns ColumnExpr (lazy)."""
        result = ds['name'].str.upper()
        assert isinstance(result, ColumnExpr)
        # Should not have executed yet - just built expression

    def test_lower_is_lazy(self, ds):
        """Test that .str.lower() returns ColumnExpr (lazy)."""
        result = ds['name'].str.lower()
        assert isinstance(result, ColumnExpr)

    def test_len_is_lazy(self, ds):
        """Test that .str.len() returns ColumnExpr (lazy)."""
        result = ds['name'].str.len()
        assert isinstance(result, ColumnExpr)

    def test_contains_is_lazy(self, ds):
        """Test that .str.contains() returns ColumnExpr (lazy)."""
        result = ds['name'].str.contains('John')
        assert isinstance(result, ColumnExpr)

    def test_replace_is_lazy(self, ds):
        """Test that .str.replace() returns ColumnExpr (lazy)."""
        result = ds['name'].str.replace('|', '-')
        assert isinstance(result, ColumnExpr)

    def test_split_is_lazy(self, ds):
        """Test that .str.split() returns ColumnExpr (lazy)."""
        result = ds['name'].str.split('|')
        assert isinstance(result, ColumnExpr)

    def test_strip_is_lazy(self, ds):
        """Test that .str.strip() returns ColumnExpr (lazy)."""
        result = ds['text'].str.strip()
        assert isinstance(result, ColumnExpr)

    def test_slice_is_lazy(self, ds):
        """Test that .str.slice() returns ColumnExpr (lazy)."""
        result = ds['text'].str.slice(0, 5)
        assert isinstance(result, ColumnExpr)

    def test_get_is_lazy(self, ds):
        """Test that .str.get() returns ColumnExpr (lazy)."""
        result = ds['text'].str.get(0)
        assert isinstance(result, ColumnExpr)

    def test_count_is_lazy(self, ds):
        """Test that .str.count() returns ColumnExpr (lazy)."""
        result = ds['text'].str.count('o')
        assert isinstance(result, ColumnExpr)

    # ==================== Lazy Execution Verification ====================

    def test_lazy_chain_no_execution(self, ds):
        """Test that chaining lazy str methods doesn't trigger execution."""
        # Chain multiple operations
        result = ds['name'].str.upper().str.lower().str.len()

        # All should be lazy - returning ColumnExpr
        assert isinstance(result, ColumnExpr)

        # No execution should have happened - verify by checking _lazy_ops
        # Original ds should still have minimal lazy ops
        assert len(ds._lazy_ops) <= 2  # Initial DataFrame source + maybe one more

    def test_lazy_column_assignment_with_str(self, ds):
        """Test that assigning str result to column is lazy."""
        # This should be recorded as lazy operation
        ds['upper_name'] = ds['name'].str.upper()

        # Should have recorded a lazy operation
        has_lazy_assignment = any(op.__class__.__name__ == 'LazyColumnAssignment' for op in ds._lazy_ops)
        assert has_lazy_assignment

    def test_lazy_str_executes_on_to_df(self, ds):
        """Test that lazy str operations execute when calling to_df()."""
        ds['upper_name'] = ds['name'].str.upper()

        # Execute
        df = ds.to_df()

        # Verify result
        assert 'upper_name' in df.columns

    # ==================== Executing Methods ====================

    def test_partition_executes(self, ds):
        """Test that .str.partition() executes and returns DataStore."""
        result = ds['name'].str.partition('|')

        # Should return a DataStore (executed)
        assert isinstance(result, DataStore)

        # Verify result structure
        df = result.to_df()
        assert df.shape[1] == 3  # Three columns: left, sep, right
        assert len(df) == 3  # Three rows

    def test_rpartition_executes(self, ds):
        """Test that .str.rpartition() executes and returns DataStore."""
        result = ds['name'].str.rpartition('|')

        # Should return a DataStore (executed)
        assert isinstance(result, DataStore)

        # Verify result
        df = result.to_df()
        assert df.shape[1] == 3

    def test_get_dummies_executes(self, ds):
        """Test that .str.get_dummies() executes and returns DataStore."""
        result = ds['name'].str.get_dummies('|')

        # Should return a DataStore (executed)
        assert isinstance(result, DataStore)

        # Verify result has dummy columns
        df = result.to_df()
        assert df.shape[1] > 0

    def test_extractall_executes(self, ds):
        """Test that .str.extractall() executes and returns DataStore."""
        result = ds['numbers'].str.extractall(r'(\d+)')

        # Should return a DataStore (executed)
        assert isinstance(result, DataStore)

        # Verify result
        df = result.to_df()
        assert len(df) > 0  # Should have extracted numbers

    def test_cat_executes_to_string(self, ds):
        """Test that .str.cat() executes and returns string."""
        result = ds['name'].str.cat(sep='-')

        # Should return a string (fully executed)
        assert isinstance(result, str)
        assert 'John|Doe' in result
        assert '-' in result

    def test_partition_expand_false_returns_series(self, ds):
        """Test that partition with expand=False returns Series."""
        result = ds['name'].str.partition('|', expand=False)

        # Should return a Series of tuples
        assert isinstance(result, pd.Series)
        assert all(isinstance(x, tuple) for x in result)

    # ==================== Mixed Usage ====================

    def test_lazy_after_executing_method(self, ds):
        """Test that lazy operations work after executing method."""
        # First execute with partition
        partitioned = ds['name'].str.partition('|')
        assert isinstance(partitioned, DataStore)

        # Get column names from the partitioned result
        df = partitioned.to_df()
        first_col = df.columns[0]

        # Create fresh DataStore from the DataFrame for lazy ops test
        partitioned2 = DataStore.from_df(df)
        partitioned2['first_upper'] = partitioned2[str(first_col)].str.upper()

        # Should have lazy assignment
        has_lazy = any(op.__class__.__name__ == 'LazyColumnAssignment' for op in partitioned2._lazy_ops)
        assert has_lazy

    def test_executing_method_preserves_parent(self, ds):
        """Test that executing method doesn't modify parent DataStore."""
        original_lazy_ops_count = len(ds._lazy_ops)

        # Call executing method
        result = ds['name'].str.partition('|')

        # Parent should be unchanged
        assert len(ds._lazy_ops) == original_lazy_ops_count

        # Can still do operations on parent
        ds['new_col'] = ds['text'].str.upper()
        assert len(ds._lazy_ops) == original_lazy_ops_count + 1

    # ==================== Correctness ====================

    def test_lazy_str_upper_correct_result(self, ds):
        """Test that lazy .str.upper() produces correct result."""
        ds['upper_name'] = ds['name'].str.upper()
        df = ds.order_by('id').to_df()

        expected = ['JOHN|DOE', 'JANE|SMITH', 'BOB|BROWN']
        assert list(df['upper_name']) == expected

    def test_lazy_str_len_correct_result(self, ds):
        """Test that lazy .str.len() produces correct result."""
        ds['name_len'] = ds['name'].str.len()
        df = ds.order_by('id').to_df()

        # Lengths: 'John|Doe'=8, 'Jane|Smith'=10, 'Bob|Brown'=9
        assert list(df['name_len']) == [8, 10, 9]

    def test_partition_correct_result(self, ds):
        """Test that .str.partition() produces correct result."""
        result = ds['name'].str.partition('|')
        df = result.to_df()

        assert list(df[0]) == ['John', 'Jane', 'Bob']
        assert list(df[1]) == ['|', '|', '|']
        assert list(df[2]) == ['Doe', 'Smith', 'Brown']

    def test_get_dummies_correct_result(self, ds):
        """Test that .str.get_dummies() produces correct result."""
        result = ds['name'].str.get_dummies('|')
        df = result.to_df()

        # Should have columns for each unique value
        assert 'John' in df.columns
        assert 'Doe' in df.columns
        assert 'Jane' in df.columns

    def test_extractall_correct_result(self, ds):
        """Test that .str.extractall() produces correct result with MultiIndex."""
        result = ds['numbers'].str.extractall(r'(\d+)')
        df = result.to_df()

        # extractall returns MultiIndex (original_index, match)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == [None, 'match']

        # First row 'abc123def456' should have matches '123', '456'
        first_row_matches = df.xs(0, level=0)[0].tolist()
        assert '123' in first_row_matches
        assert '456' in first_row_matches


class TestStrContainsNaParameter:
    """Test str.contains() with na parameter - pandas compatibility."""

    @pytest.fixture
    def ds_with_nulls(self):
        """Create a test DataStore with null values."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'name': ['Alice', None, 'Bob', 'Charlie'],
            }
        )
        return DataStore.from_df(df)

    def test_contains_with_na_false(self, ds_with_nulls):
        """Test that .str.contains() works with na=False parameter."""
        # This should not raise TypeError anymore
        result = ds_with_nulls['name'].str.contains('a', na=False)
        assert isinstance(result, ColumnExpr)

        # Execute and check result
        ds_with_nulls['has_a'] = result
        df = ds_with_nulls.to_df()

        # 'Alice' contains 'a' (lowercase match), None becomes False, 'Bob' no 'a', 'Charlie' has 'a'
        # Note: pandas str.contains is case-sensitive by default
        expected = [False, False, False, True]  # 'a' only in 'Charlie'
        assert list(df['has_a']) == expected

    def test_contains_with_na_true(self, ds_with_nulls):
        """Test that .str.contains() works with na=True parameter."""
        result = ds_with_nulls['name'].str.contains('a', na=True)
        assert isinstance(result, ColumnExpr)

        ds_with_nulls['has_a'] = result
        df = ds_with_nulls.to_df()

        # None becomes True with na=True
        expected = [False, True, False, True]
        assert list(df['has_a']) == expected

    def test_contains_with_case_insensitive(self, ds_with_nulls):
        """Test that .str.contains() works with case=False parameter."""
        result = ds_with_nulls['name'].str.contains('a', case=False, na=False)
        assert isinstance(result, ColumnExpr)

        ds_with_nulls['has_a'] = result
        df = ds_with_nulls.to_df()

        # Case insensitive: 'Alice' has 'A', 'Charlie' has 'a'
        expected = [True, False, False, True]
        assert list(df['has_a']) == expected

    def test_contains_comprehensive(self):
        """Comprehensive test for str.contains with various parameters."""
        df = pd.DataFrame({'name': ['Alice', None, 'Bob', 'Charlie']})

        # Test 1: na=False - NaN values become False
        ds1 = DataStore.from_df(df)
        ds1['has_a'] = ds1['name'].str.contains('a', na=False)
        df_result1 = ds1.to_df()
        assert list(df_result1['has_a']) == [False, False, False, True]

        # Test 2: na=True - NaN values become True
        ds2 = DataStore.from_df(df)
        ds2['has_a'] = ds2['name'].str.contains('a', na=True)
        df_result2 = ds2.to_df()
        assert list(df_result2['has_a']) == [False, True, False, True]

        # Test 3: case=False - Case insensitive matching
        ds3 = DataStore.from_df(df)
        ds3['has_a'] = ds3['name'].str.contains('A', case=False, na=False)
        df_result3 = ds3.to_df()
        assert list(df_result3['has_a']) == [True, False, False, True]

    def test_contains_with_na_chdb_native(self):
        """
        Test str.contains with na parameter behavior.

        Note: 'contains' is in PANDAS_ONLY_FUNCTIONS due to chDB's NaN handling issues,
        so this test uses pandas execution. The test verifies that na parameter works
        correctly with pandas fallback execution.
        """
        from datastore import function_config

        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['Alice', None, 'Bob'],
            }
        )
        ds = DataStore.from_df(df)

        # Note: contains is in PANDAS_ONLY_FUNCTIONS, so use_chdb won't override it
        # This test documents the limitation rather than testing chDB behavior
        try:
            function_config.use_chdb('contains')

            result = ds['name'].str.contains('a', na=False)
            ds['has_a'] = result
            df_result = ds.to_df()

            # With pandas execution: 'a' not in 'Alice'/'Bob' (case-sensitive), None->False
            # Note: Would fail with chDB which returns position values, not boolean
            expected = [False, False, False]
            assert list(df_result['has_a']) == expected
        finally:
            # Restore original setting
            function_config._function_engines.pop('contains', None)


class TestStrAccessorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.skip(reason="chDB crashes on empty DataFrame query - https://github.com/chdb-io/chdb/issues/452")
    def test_empty_dataframe(self):
        """Test str accessor on empty DataFrame."""
        df = pd.DataFrame({'name': pd.Series([], dtype=str)})
        ds = DataStore.from_df(df)

        # Lazy method should work
        result = ds['name'].str.upper()
        assert isinstance(result, ColumnExpr)

        # Executing method should work
        result = ds['name'].str.partition('|')
        assert isinstance(result, DataStore)
        assert len(result.to_df()) == 0

    def test_null_values(self):
        """Test str accessor with null values."""
        df = pd.DataFrame({'name': ['hello', None, 'world']})
        ds = DataStore.from_df(df)

        # Should handle nulls gracefully
        result = ds['name'].str.partition(' ')
        assert isinstance(result, DataStore)


class TestStrAccessorChaining:
    """Test that str accessor results can chain with other operations."""

    @pytest.fixture
    def ds(self):
        """Create a test DataStore."""
        df = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['alice', 'bob', 'charlie'],
                'text': ['hello world', 'foo bar', 'test string'],
            }
        )
        return DataStore.from_df(df)

    def test_str_upper_returns_column_expr_with_accessors(self, ds):
        """Test str.upper() returns ColumnExpr which has accessors."""
        result = ds['name'].str.upper()

        # Should be a ColumnExpr
        assert isinstance(result, ColumnExpr)

        # ColumnExpr should have str accessor for continued chaining
        assert hasattr(result, 'str')

    def test_str_upper_sort_values(self, ds):
        """Test str.upper().sort_values() - pandas API compatibility."""
        result = ds['name'].str.upper().sort_values()

        # Should return ColumnExpr
        assert isinstance(result, ColumnExpr)

        # Values should be sorted alphabetically (uppercase)
        expected = ['ALICE', 'BOB', 'CHARLIE']
        assert list(result.values) == expected

    def test_str_len_sort_values(self, ds):
        """Test str.len().sort_values() - pandas API compatibility."""
        result = ds['name'].str.len().sort_values()

        # Should return ColumnExpr
        assert isinstance(result, ColumnExpr)

        # Values should be sorted: bob(3), alice(5), charlie(7)
        expected = [3, 5, 7]
        assert list(result.values) == expected

    def test_str_chain_upper_then_len(self, ds):
        """Test chaining str.upper().str.len()."""
        result = ds['name'].str.upper().str.len()

        # Should return ColumnExpr (still lazy)
        assert isinstance(result, ColumnExpr)

        # Assign and execute
        ds['name_len'] = result
        df = ds.order_by('id').to_df()

        # Lengths: 'ALICE'=5, 'BOB'=3, 'CHARLIE'=7
        expected = [5, 3, 7]
        assert list(df['name_len']) == expected

    def test_str_chain_slice_then_upper(self, ds):
        """Test chaining str.slice().str.upper()."""
        result = ds['name'].str.slice(0, 3).str.upper()

        ds['prefix'] = result
        df = ds.order_by('id').to_df()

        # First 3 chars uppercased: 'ALI', 'BOB', 'CHA'
        expected = ['ALI', 'BOB', 'CHA']
        assert list(df['prefix']) == expected

    def test_str_chain_replace_then_len(self, ds):
        """Test chaining str.replace().str.len()."""
        result = ds['text'].str.replace(' ', '').str.len()

        ds['compact_len'] = result
        df = ds.order_by('id').to_df()

        # 'helloworld'=10, 'foobar'=6, 'teststring'=10
        expected = [10, 6, 10]
        assert list(df['compact_len']) == expected

    def test_str_len_comparison(self, ds):
        """Test str.len() can be used in comparisons for filtering."""
        # Filter names with length > 3
        result = ds[ds['name'].str.len() > 3]
        df = result.to_df()

        # 'alice'(5) and 'charlie'(7) should pass
        assert len(df) == 2
        names = set(df['name'])
        assert 'alice' in names
        assert 'charlie' in names
        assert 'bob' not in names

    def test_str_len_arithmetic(self, ds):
        """Test str.len() can be used in arithmetic operations."""
        result = ds['name'].str.len() * 10

        ds['name_len_x10'] = result
        df = ds.order_by('id').to_df()

        expected = [50, 30, 70]  # 5*10, 3*10, 7*10
        assert list(df['name_len_x10']) == expected

    def test_str_upper_then_contains(self, ds):
        """Test chaining str.upper() then str.contains()."""
        result = ds['name'].str.upper().str.contains('LI')

        ds['has_li'] = result
        df = ds.order_by('id').to_df()

        # 'ALICE' contains 'LI', 'BOB' doesn't, 'CHARLIE' contains 'LI'
        expected = [True, False, True]
        assert list(df['has_li']) == expected

    def test_str_triple_chain(self, ds):
        """Test three chained str operations."""
        result = ds['name'].str.upper().str.slice(0, 2).str.lower()

        ds['result'] = result
        df = ds.order_by('id').to_df()

        # 'alice' -> 'ALICE' -> 'AL' -> 'al'
        expected = ['al', 'bo', 'ch']
        assert list(df['result']) == expected

    def test_str_chain_sql_generation(self, ds):
        """Verify chained str accessor generates correct SQL."""
        result = ds['name'].str.upper().str.len()

        # Get the underlying expression and check SQL
        expr = result._expr if hasattr(result, '_expr') else result
        sql_repr = str(expr)

        # Should have both upper and length functions
        assert 'upper' in sql_repr.lower()
        assert 'length' in sql_repr.lower()

    def test_str_accessor_from_function_result(self, ds):
        """Test that Function results have str accessor."""
        from datastore.functions import Function
        from datastore.expressions import Expression

        # Create a Function manually
        upper_func = Function('upper', ds['name']._expr)

        # Function should have str accessor (inherited from Expression)
        assert hasattr(upper_func, 'str')

        # Should be able to chain .str.len() - returns another Expression (Function or ColumnExpr)
        len_func = upper_func.str.len()
        # The result is an Expression (could be Function, ColumnExpr, or another Expression subtype)
        assert isinstance(len_func, (Expression, ColumnExpr))


class TestStrContainsFilter:
    """Test that .str.contains() works correctly for filtering.

    This is a regression test for the bug where str.contains() only returned
    the first matching row because it was using position() instead of
    position() > 0 for boolean filtering.
    """

    @pytest.fixture
    def ds(self):
        """Create a test DataStore with names containing various characters."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'], 'value': [1, 2, 3, 4, 5]})
        return DataStore.from_df(df)

    def test_contains_filter_case_insensitive(self, ds):
        """Test str.contains() with case=False returns all matching rows."""
        # 'a' (case insensitive) should match Alice, Charlie, David, Eva
        result = ds[ds['name'].str.contains('a', case=False)]
        names = list(result['name'].values)
        assert len(names) == 4, f"Expected 4 matches, got {len(names)}: {names}"
        assert 'Alice' in names
        assert 'Charlie' in names
        assert 'David' in names
        assert 'Eva' in names

    def test_contains_filter_case_sensitive(self, ds):
        """Test str.contains() with case=True returns only exact matches."""
        # lowercase 'a' should only match Charlie, David, Eva (not Alice)
        result = ds[ds['name'].str.contains('a')]
        names = list(result['name'].values)
        assert 'Alice' not in names
        assert 'Charlie' in names
        assert 'David' in names
        # Note: Eva doesn't contain lowercase 'a'

    def test_contains_filter_uppercase(self, ds):
        """Test str.contains() with uppercase letter."""
        # 'A' should only match Alice
        result = ds[ds['name'].str.contains('A')]
        names = list(result['name'].values)
        assert len(names) == 1
        assert names[0] == 'Alice'

    def test_contains_filter_no_matches(self, ds):
        """Test str.contains() returns empty when no matches."""
        result = ds[ds['name'].str.contains('xyz')]
        assert len(result) == 0

    def test_contains_filter_all_match(self, ds):
        """Test str.contains() returns all rows when all match."""
        # Empty string matches everything
        # Or we use a common pattern
        df = pd.DataFrame({'name': ['test1', 'test2', 'test3'], 'value': [1, 2, 3]})
        ds = DataStore.from_df(df)
        result = ds[ds['name'].str.contains('test')]
        assert len(result) == 3

    def test_contains_mask_values(self, ds):
        """Test that str.contains() mask values are correct booleans."""
        mask = ds['name'].str.contains('a', case=False)
        values = list(mask.values)
        # Alice, Bob, Charlie, David, Eva
        # A: True, B: False, C: True, D: True, E: True
        assert values == [True, False, True, True, True]

    def test_contains_pandas_alignment(self, ds):
        """Test str.contains() matches pandas behavior exactly."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'], 'value': [1, 2, 3, 4, 5]})

        # Test case insensitive
        pd_result = df[df['name'].str.contains('a', case=False)]
        ds_result = ds[ds['name'].str.contains('a', case=False)]
        assert list(ds_result['name'].values) == list(pd_result['name'].values)

        # Test case sensitive
        pd_result2 = df[df['name'].str.contains('a')]
        ds_result2 = ds[ds['name'].str.contains('a')]
        assert list(ds_result2['name'].values) == list(pd_result2['name'].values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
