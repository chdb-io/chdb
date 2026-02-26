"""
Regression tests for bugfixes.

These tests verify that the following issues are fixed:
1. assign() not supporting Function expressions
2. from_df() unable to use SQL functions
3. Nested JSON paths (user.name) not working
4. DataStore.run_sql() class method support
"""

import pandas as pd
import pytest
import tempfile
import os

from datastore import DataStore


class TestAssignWithFunctionExpressions:
    """Test that assign() correctly handles Function expressions."""

    def test_assign_with_str_upper(self, tmp_path):
        """assign() should support string function expressions."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,age\nAlice,30\nBob,25\n")

        ds = DataStore.uri(str(csv_path))
        result = ds.assign(name_upper=ds['name'].str.upper())

        assert 'name_upper' in result.columns
        assert result['name_upper'].tolist() == ['ALICE', 'BOB']
        # Original columns should still be present
        assert 'name' in result.columns
        assert 'age' in result.columns

    def test_assign_generates_correct_sql(self, tmp_path):
        """assign() should generate SQL with *, computed_col format."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,age\nAlice,30\n")

        ds = DataStore.uri(str(csv_path))
        result = ds.assign(name_upper=ds['name'].str.upper())

        sql = result.to_sql()
        assert '*, upper(' in sql.lower() or '*, UPPER(' in sql
        assert 'as "name_upper"' in sql.lower() or 'AS "name_upper"' in sql

    def test_assign_with_json_function(self, tmp_path):
        """assign() should support JSON function expressions."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame({'response': ['{"id": 1}', '{"id": 2}']})
        df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(str(parquet_path))
        result = ds.assign(user_id=ds['response'].json.json_extract_int('id'))

        assert 'user_id' in result.columns
        assert result['user_id'].tolist() == [1, 2]


class TestFromDfWithSQLFunctions:
    """Test that from_df() correctly supports SQL functions via Python() table function."""

    def test_from_df_with_str_upper(self):
        """from_df() DataStores should support string SQL functions."""
        pandas_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 25]})

        ds = DataStore.from_df(pandas_df)
        result = ds.assign(name_upper=ds['name'].str.upper())

        assert 'name_upper' in result.columns
        assert result['name_upper'].tolist() == ['ALICE', 'BOB']

    def test_from_df_with_json_accessor(self):
        """from_df() DataStores should support JSON SQL functions."""
        pandas_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'response': ['{"id": 1}', '{"id": 2}']})

        ds = DataStore.from_df(pandas_df)
        result = ds.select(
            'name',
            ds['response'].json.json_extract_int('id').as_('user_id'),
        )

        assert 'name' in result.columns
        assert 'user_id' in result.columns
        assert result['user_id'].tolist() == [1, 2]

    def test_from_df_creates_python_table_function_on_demand(self):
        """from_df() should create PythonTableFunction only when SQL functions are used."""
        pandas_df = pd.DataFrame({'name': ['Alice', 'Bob']})

        ds = DataStore.from_df(pandas_df)
        # Before using SQL functions, no PythonTableFunction
        assert ds._table_function is None
        assert ds._source_df is not None  # DataFrame is cached

        # After using SQL function, PythonTableFunction is created on-demand
        result = ds.assign(name_upper=ds['name'].str.upper())
        assert result._table_function is not None
        sql = result.to_sql()
        assert 'Python(' in sql
        assert '__datastore_df_' in sql


class TestNestedJSONPaths:
    """Test that nested JSON paths (user.name) work correctly."""

    def test_nested_json_extract_string(self, tmp_path):
        """json_extract_string() should support dot-separated nested paths."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame({'response': ['{"user": {"name": "Alice"}}', '{"user": {"name": "Bob"}}']})
        df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(str(parquet_path))
        result = ds.select(ds['response'].json.json_extract_string('user.name').as_('user_name'))

        assert result['user_name'].tolist() == ['Alice', 'Bob']

    def test_nested_json_extract_int(self, tmp_path):
        """json_extract_int() should support dot-separated nested paths."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame({'response': ['{"user": {"id": 1}}', '{"user": {"id": 2}}']})
        df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(str(parquet_path))
        result = ds.select(ds['response'].json.json_extract_int('user.id').as_('user_id'))

        assert result['user_id'].tolist() == [1, 2]

    def test_deep_nested_json_path(self, tmp_path):
        """json_extract_*() should support deeply nested paths."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame({'data': ['{"a": {"b": {"c": "value1"}}}', '{"a": {"b": {"c": "value2"}}}']})
        df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(str(parquet_path))
        result = ds.select(ds['data'].json.json_extract_string('a.b.c').as_('deep_value'))

        assert result['deep_value'].tolist() == ['value1', 'value2']

    def test_nested_json_generates_correct_sql(self, tmp_path):
        """Nested JSON paths should generate separate arguments in SQL."""
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame({'data': ['{}']})
        df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(str(parquet_path))
        result = ds.select(ds['data'].json.json_extract_string('user.name').as_('user_name'))
        sql = result.to_sql()

        # Should have separate arguments, not 'user.name' as single string
        assert "'user','name'" in sql.replace(' ', '') or "'user', 'name'" in sql


class TestDataStoreRunSql:
    """Test that DataStore.run_sql() works as a class method."""

    def test_run_sql_basic(self, tmp_path):
        """DataStore.run_sql() should execute raw SQL queries."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,age\nAlice,30\nBob,25\n")

        result = DataStore.run_sql(
            f"""
            SELECT name, age * 2 as doubled
            FROM file('{csv_path}', 'CSVWithNames')
        """
        )

        df = result
        assert 'name' in df.columns
        assert 'doubled' in df.columns
        assert df['doubled'].tolist() == [60, 50]

    def test_run_sql_with_filter(self, tmp_path):
        """DataStore.run_sql() should support WHERE clauses."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,age\nAlice,30\nBob,25\nCharlie,35\n")

        result = DataStore.run_sql(
            f"""
            SELECT name FROM file('{csv_path}', 'CSVWithNames')
            WHERE age > 26
        """
        )

        df = result
        assert set(df['name'].tolist()) == {'Alice', 'Charlie'}

    def test_run_sql_with_aggregation(self, tmp_path):
        """DataStore.run_sql() should support GROUP BY and aggregations."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("city,value\nA,10\nA,20\nB,30\n")

        result = DataStore.run_sql(
            f"""
            SELECT city, SUM(value) as total
            FROM file('{csv_path}', 'CSVWithNames')
            GROUP BY city
            ORDER BY city
        """
        )

        df = result
        assert df['city'].tolist() == ['A', 'B']
        assert df['total'].tolist() == [30, 30]


class TestCombinedFunctionality:
    """Test combined functionality of all fixes."""

    def test_from_df_assign_nested_json(self):
        """from_df() + assign() + nested JSON should work together."""
        pandas_df = pd.DataFrame(
            {'data': ['{"user": {"name": "Alice", "id": 1}}', '{"user": {"name": "Bob", "id": 2}}']}
        )

        ds = DataStore.from_df(pandas_df)
        result = ds.select(
            'data',
            ds['data'].json.json_extract_string('user.name').as_('user_name'),
            ds['data'].json.json_extract_int('user.id').as_('user_id'),
        )

        assert 'data' in result.columns
        assert 'user_name' in result.columns
        assert 'user_id' in result.columns
        assert result['user_name'].tolist() == ['Alice', 'Bob']
        assert result['user_id'].tolist() == [1, 2]

    def test_full_pipeline_from_df(self):
        """Full pipeline with from_df() should work correctly."""
        pandas_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [30, 25, 35]})

        ds = DataStore.from_df(pandas_df)
        result = ds.filter(ds['age'] > 26).assign(name_upper=ds['name'].str.upper())

        assert len(result) == 2  # Alice (30) and Charlie (35)
        assert 'name_upper' in result.columns
        assert set(result['name_upper'].tolist()) == {'ALICE', 'CHARLIE'}
