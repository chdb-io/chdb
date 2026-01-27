"""
Test Series.json accessor functionality.

Verifies that:
1. JSON accessor methods work correctly with ClickHouse JSON functions
2. Lazy execution is maintained
3. Nested path extraction works
4. Edge cases are handled properly (invalid JSON, missing keys, NULL values)
"""

import pytest
import pandas as pd
import chdb
from datastore import DataStore
from datastore.column_expr import ColumnExpr
from datastore.functions import Function


from tests.xfail_markers import chdb_array_nullable


class TestJsonAccessorBasic:
    """Test basic .json accessor methods."""

    @pytest.fixture
    def ds_with_json(self):
        """Create a test DataStore with JSON columns."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"name": "Alice", "age": 30, "active": true}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"name": "Bob", "age": 25, "active": false}' as data
            UNION ALL
            SELECT 
                3 as id,
                '{"name": "Charlie", "age": 35, "active": true}' as data
            ORDER BY id
        """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    # ==================== Type Tests ====================

    def test_json_extract_string_returns_column_expr(self, ds_with_json):
        """Test that .json.json_extract_string() returns ColumnExpr (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr

        result = ds_with_json['data'].json.json_extract_string('name')
        assert isinstance(result, ColumnExpr)
        # Should have sort_values for pandas compatibility
        assert hasattr(result, 'sort_values')

    def test_json_extract_int_returns_column_expr(self, ds_with_json):
        """Test that .json.json_extract_int() returns ColumnExpr (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr

        result = ds_with_json['data'].json.json_extract_int('age')
        assert isinstance(result, ColumnExpr)
        # Should have sort_values for pandas compatibility
        assert hasattr(result, 'sort_values')

    # ==================== Extract String Tests ====================

    def test_json_extract_string_execution(self, ds_with_json):
        """Test extracting string values from JSON."""
        ds_with_json['name'] = ds_with_json['data'].json.json_extract_string('name')
        df = ds_with_json.sort_values('id').to_df()

        assert 'name' in df.columns
        expected_names = ['Alice', 'Bob', 'Charlie']
        assert list(df['name']) == expected_names

    # ==================== Extract Int Tests ====================

    def test_json_extract_int_execution(self, ds_with_json):
        """Test extracting integer values from JSON."""
        ds_with_json['age'] = ds_with_json['data'].json.json_extract_int('age')
        df = ds_with_json.sort_values('id').to_df()

        # Note: ORDER BY id in fixture ensures consistent order
        expected_ages = [30, 25, 35]
        assert list(df['age']) == expected_ages

    # ==================== Extract Bool Tests ====================

    def test_json_extract_bool_execution(self, ds_with_json):
        """Test extracting boolean values from JSON."""
        ds_with_json['is_active'] = ds_with_json['data'].json.json_extract_bool('active')
        df = ds_with_json.sort_values('id').to_df()

        expected = [1, 0, 1]  # ClickHouse returns 1/0 for bool
        assert list(df['is_active']) == expected


class TestJsonNestedPaths:
    """Test nested path extraction from JSON."""

    @pytest.fixture
    def ds_nested_json(self):
        """Create DataStore with nested JSON."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}}}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"user": {"name": "Bob", "address": {"city": "LA", "zip": "90001"}}}' as data
            ORDER BY id
        """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    def test_nested_path_extraction(self, ds_nested_json):
        """Test extracting values from nested JSON paths."""
        ds_nested_json['username'] = ds_nested_json['data'].json.json_extract_string('user.name')
        # Sort by id for deterministic comparison (row order may vary with Python() table function)
        df = ds_nested_json.to_df().sort_values('id')

        expected = ['Alice', 'Bob']
        assert list(df['username']) == expected

    def test_deeply_nested_path(self, ds_nested_json):
        """Test extracting deeply nested values."""
        ds_nested_json['city'] = ds_nested_json['data'].json.json_extract_string('user.address.city')
        # Sort by id for deterministic comparison (row order may vary with Python() table function)
        df = ds_nested_json.to_df().sort_values('id')

        expected = ['NYC', 'LA']
        assert list(df['city']) == expected


class TestJsonExtractFloat:
    """Test float extraction from JSON."""

    @pytest.fixture
    def ds_float_json(self):
        """Create DataStore with float values in JSON."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"price": 19.99, "quantity": 5}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"price": 29.50, "quantity": 3}' as data
            ORDER BY id
        """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    def test_json_extract_float(self, ds_float_json):
        """Test extracting float values from JSON."""
        ds_float_json['price'] = ds_float_json['data'].json.json_extract_float('price')
        df = ds_float_json.to_df()

        expected = [19.99, 29.50]
        assert list(df['price']) == pytest.approx(expected)


class TestJsonExtractRaw:
    """Test raw JSON extraction."""

    @pytest.fixture
    def ds_complex_json(self):
        """Create DataStore with complex JSON containing nested objects."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"user": {"name": "Alice", "tags": ["admin", "user"]}}' as data
        """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    def test_json_extract_raw(self, ds_complex_json):
        """Test extracting raw JSON subtree."""
        ds_complex_json['user_raw'] = ds_complex_json['data'].json.json_extract_raw('user')
        df = ds_complex_json.to_df()

        # Should return the raw JSON string for the 'user' key
        result = df['user_raw'].iloc[0]
        assert 'name' in result
        assert 'Alice' in result


class TestJsonEdgeCases:
    """Test edge cases for JSON accessor."""

    def test_missing_key_returns_empty(self):
        """Test extracting non-existent key returns empty/default."""
        df = chdb.query(
            """
            SELECT '{"name": "Alice"}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)
        ds['missing'] = ds['data'].json.json_extract_string('nonexistent')
        result = ds.to_df()

        # ClickHouse returns empty string for missing keys
        assert result['missing'].iloc[0] == ''

    def test_missing_nested_key(self):
        """Test extracting non-existent nested key."""
        df = chdb.query(
            """
            SELECT '{"user": {"name": "Alice"}}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)
        ds['missing'] = ds['data'].json.json_extract_string('user.email')
        result = ds.to_df()

        assert result['missing'].iloc[0] == ''

    def test_type_mismatch_int_from_string(self):
        """Test extracting int from string value returns 0."""
        df = chdb.query(
            """
            SELECT '{"value": "not_a_number"}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)
        ds['num'] = ds['data'].json.json_extract_int('value')
        result = ds.to_df()

        # ClickHouse returns 0 for type mismatch
        assert result['num'].iloc[0] == 0

    def test_empty_json_object(self):
        """Test extracting from empty JSON object."""
        df = chdb.query("SELECT '{}' as data", "DataFrame")
        ds = DataStore.from_df(df)
        ds['missing'] = ds['data'].json.json_extract_string('key')
        result = ds.to_df()

        assert result['missing'].iloc[0] == ''

    def test_json_with_null_value(self):
        """Test extracting null value from JSON."""
        df = chdb.query(
            """
            SELECT '{"name": null}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)
        ds['name'] = ds['data'].json.json_extract_string('name')
        result = ds.to_df()

        # ClickHouse returns empty string for null
        assert result['name'].iloc[0] == ''


class TestJsonLazyExecution:
    """Test that JSON operations maintain lazy execution."""

    def test_json_ops_return_column_expr(self):
        """Test that JSON operations return ColumnExpr objects (pandas API compatible)."""
        from datastore.column_expr import ColumnExpr

        df = chdb.query("SELECT '{\"a\": 1, \"b\": \"test\"}' as data", "DataFrame")
        ds = DataStore.from_df(df)

        # These should return ColumnExpr objects (for pandas API compatibility like sort_values)
        str_expr = ds['data'].json.json_extract_string('b')
        int_expr = ds['data'].json.json_extract_int('a')

        assert isinstance(str_expr, ColumnExpr)
        assert isinstance(int_expr, ColumnExpr)

    def test_multiple_json_extractions_lazy(self):
        """Test multiple JSON extractions remain lazy."""
        df = chdb.query(
            """
            SELECT '{"name": "Alice", "age": 30}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)

        ds['name'] = ds['data'].json.json_extract_string('name')
        ds['age'] = ds['data'].json.json_extract_int('age')

        # Should have recorded lazy operations
        has_assignments = sum(1 for op in ds._lazy_ops if op.__class__.__name__ == 'LazyColumnAssignment')
        assert has_assignments >= 2

        # Execute and verify
        result = ds.to_df()
        assert result['name'].iloc[0] == 'Alice'
        assert result['age'].iloc[0] == 30

    def test_json_extraction_with_filter(self):
        """Test JSON extraction combined with filtering."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"score": 85}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"score": 45}' as data
            UNION ALL
            SELECT 
                3 as id,
                '{"score": 95}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)

        ds['score'] = ds['data'].json.json_extract_int('score')
        result = ds[ds['score'] > 50]
        result_df = result.to_df()

        # Should have 2 rows (scores 85 and 95)
        assert len(result_df) == 2
        assert set(result_df['id']) == {1, 3}


class TestJsonArrayExtraction:
    """Test extracting arrays from JSON."""

    @pytest.fixture
    def ds_json_arrays(self):
        """Create DataStore with JSON containing arrays."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"tags": ["python", "data", "ml"]}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"tags": ["java", "backend"]}' as data
            ORDER BY id
        """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    def test_json_extract_array_raw(self, ds_json_arrays):
        """Test extracting array as raw JSON.

        This uses pandas fallback because chDB doesn't support Array inside Nullable type
        when the source column comes from Python() table function.
        The fix detects this and falls back to JSON parsing in Python.
        """
        ds_json_arrays['tags_raw'] = ds_json_arrays['data'].json.json_extract_array_raw('tags')
        df = ds_json_arrays.to_df()

        # Result should be a list
        result = df['tags_raw'].iloc[0]
        assert isinstance(result, list)
        assert result == ['python', 'data', 'ml']

        # Check second row
        result2 = df['tags_raw'].iloc[1]
        assert isinstance(result2, list)
        assert result2 == ['java', 'backend']

    def test_json_extract_array_raw_nested_path(self, ds_json_arrays):
        """Test extracting array from nested JSON path using pandas fallback."""
        # Create data with nested structure
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"user": {"hobbies": ["reading", "coding"]}}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"user": {"hobbies": ["gaming"]}}' as data
            ORDER BY id
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)
        ds['hobbies'] = ds['data'].json.json_extract_array_raw('user.hobbies')
        result = ds.to_df()

        assert isinstance(result['hobbies'].iloc[0], list)
        assert result['hobbies'].iloc[0] == ['reading', 'coding']
        assert result['hobbies'].iloc[1] == ['gaming']

    def test_json_extract_array_raw_missing_key(self, ds_json_arrays):
        """Test extracting array from missing key returns None."""
        ds_json_arrays['missing'] = ds_json_arrays['data'].json.json_extract_array_raw('nonexistent')
        result = ds_json_arrays.to_df()

        assert result['missing'].iloc[0] is None
        assert result['missing'].iloc[1] is None

    def test_json_extract_array_raw_non_array_value(self, ds_json_arrays):
        """Test extracting array from non-array value returns None."""
        # Create data where 'tags' is a string, not an array
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"tags": "not_an_array"}' as data
        """,
            "DataFrame",
        )
        ds = DataStore.from_df(df)
        ds['tags'] = ds['data'].json.json_extract_array_raw('tags')
        result = ds.to_df()

        assert result['tags'].iloc[0] is None


class TestJsonDirectSQL:
    """Test JSON operations using direct SQL to verify correct behavior."""

    def test_json_functions_work_in_pure_chdb(self):
        """Verify JSON functions work correctly in pure chDB."""
        result = chdb.query(
            """
            SELECT 
                '{"name": "Alice", "age": 30}' as data,
                JSONExtractString('{"name": "Alice", "age": 30}', 'name') as name,
                JSONExtractInt('{"name": "Alice", "age": 30}', 'age') as age
        """,
            "DataFrame",
        )

        assert result['name'].iloc[0] == 'Alice'
        assert result['age'].iloc[0] == 30

    def test_json_sql_generation(self):
        """Verify json accessor generates correct SQL."""
        df = chdb.query("SELECT '{\"name\": \"test\"}' as data", "DataFrame")
        ds = DataStore.from_df(df)

        json_result = ds['data'].json.json_extract_string('name')
        # Get underlying expression's SQL representation
        sql_repr = json_result._expr.to_sql()

        # Should generate JSONExtractString function call
        assert 'JSONExtractString' in sql_repr
        assert 'data' in sql_repr
        assert 'name' in sql_repr

    def test_json_array_extraction_in_pure_chdb(self):
        """Verify JSONExtractArrayRaw works in pure chDB (not via Python() table function)."""
        result = chdb.query(
            """
            SELECT 
                JSONExtractArrayRaw('{"tags": ["a", "b", "c"]}', 'tags') as tags
        """,
            "DataFrame",
        )

        tags = result['tags'].iloc[0]
        # Should be a list/array with 3 elements
        assert len(tags) == 3


class TestJsonAccessorChaining:
    """Test that JSON accessor results can chain with other accessor operations."""

    @pytest.fixture
    def ds_with_json(self):
        """Create a test DataStore with JSON columns."""
        df = chdb.query(
            """
            SELECT 
                1 as id,
                '{"name": "alice", "age": 30}' as data
            UNION ALL
            SELECT 
                2 as id,
                '{"name": "bob", "age": 25}' as data
            ORDER BY id
        """,
            "DataFrame",
        )
        return DataStore.from_df(df)

    def test_json_accessor_returns_column_expr(self, ds_with_json):
        """Test json accessor methods return ColumnExpr for pandas API compatibility."""
        from datastore.column_expr import ColumnExpr

        result = ds_with_json['data'].json.json_extract_string('name')

        # Should return ColumnExpr, not Function
        assert isinstance(result, ColumnExpr)
        # Should have sort_values method
        assert hasattr(result, 'sort_values')

    def test_json_extract_string_sort_values(self, ds_with_json):
        """Test json.json_extract_string().sort_values() - pandas API compatibility."""
        result = ds_with_json['data'].json.json_extract_string('name').sort_values()

        # Should return ColumnExpr
        from datastore.column_expr import ColumnExpr

        assert isinstance(result, ColumnExpr)

        # Values should be sorted alphabetically
        expected = ['alice', 'bob']
        assert list(result.values) == expected

    def test_json_extract_int_sort_values(self, ds_with_json):
        """Test json.json_extract_int().sort_values() - pandas API compatibility."""
        result = ds_with_json['data'].json.json_extract_int('age').sort_values()

        # Values should be sorted numerically
        expected = [25, 30]
        assert list(result.values) == expected

    def test_json_sort_values_descending(self, ds_with_json):
        """Test json accessor with sort_values(ascending=False)."""
        result = ds_with_json['data'].json.json_extract_string('name').sort_values(ascending=False)

        expected = ['bob', 'alice']
        assert list(result.values) == expected

    def test_json_extract_string_chain_str_upper(self, ds_with_json):
        """Test json.json_extract_string().str.upper() chaining."""
        from datastore.column_expr import ColumnExpr

        # Extract JSON string, then chain .str accessor
        result = ds_with_json['data'].json.json_extract_string('name').str.upper()

        # Should return a ColumnExpr (lazy, supports pandas-like API)
        assert isinstance(result, ColumnExpr)

        # Assign and execute
        ds_with_json['upper_name'] = result
        df = ds_with_json.to_df()

        # Verify chained operations executed correctly
        expected = ['ALICE', 'BOB']
        assert list(df['upper_name']) == expected

    def test_json_extract_string_chain_str_len(self, ds_with_json):
        """Test json.json_extract_string().str.len() chaining."""
        from datastore.column_expr import ColumnExpr

        result = ds_with_json['data'].json.json_extract_string('name').str.len()

        assert isinstance(result, ColumnExpr)

        ds_with_json['name_len'] = result
        df = ds_with_json.to_df()

        # 'alice' = 5, 'bob' = 3
        expected = [5, 3]
        assert list(df['name_len']) == expected

    def test_json_extract_int_comparison(self, ds_with_json):
        """Test json.json_extract_int() can be used in comparisons."""
        # Extract int and use in filter
        age_expr = ds_with_json['data'].json.json_extract_int('age')
        result = ds_with_json[age_expr > 26]
        df = result.to_df()

        # Only id=1 (age=30) should pass filter
        assert len(df) == 1
        assert df['id'].iloc[0] == 1

    def test_json_extract_int_arithmetic(self, ds_with_json):
        """Test json.json_extract_int() can chain with arithmetic operations."""
        age_expr = ds_with_json['data'].json.json_extract_int('age')
        doubled = age_expr * 2

        ds_with_json['doubled_age'] = doubled
        df = ds_with_json.to_df()

        expected = [60, 50]  # 30*2, 25*2
        assert list(df['doubled_age']) == expected

    def test_json_chain_multiple_str_operations(self, ds_with_json):
        """Test multiple chained .str operations on JSON result."""
        result = ds_with_json['data'].json.json_extract_string('name').str.upper().str.slice(0, 2)  # First 2 chars

        ds_with_json['prefix'] = result
        df = ds_with_json.to_df()

        expected = ['AL', 'BO']
        assert list(df['prefix']) == expected

    def test_json_sql_generation_with_chain(self, ds_with_json):
        """Verify chained JSON + str accessor generates correct SQL."""
        result = ds_with_json['data'].json.json_extract_string('name').str.upper()

        # Get the underlying expression's SQL representation
        sql_repr = result._expr.to_sql()

        # Should have both JSONExtractString and upper in the SQL
        assert 'JSONExtractString' in sql_repr
        assert 'upper' in sql_repr.lower()
