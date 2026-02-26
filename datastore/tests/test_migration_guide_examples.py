"""
Test examples from PANDAS_MIGRATION_GUIDE.md
Ensures that documented examples actually work.
"""

import pytest
import pandas as pandas_original
import numpy as np


@pytest.fixture
def employee_csv(tmp_path):
    """Create a sample employee CSV file."""
    csv_content = """name,age,salary,city,department
Alice,30,75000,NYC,Engineering
Bob,25,50000,LA,Marketing
Charlie,35,90000,NYC,Engineering
Diana,28,55000,Chicago,Sales
Eve,32,80000,NYC,Engineering
Frank,24,45000,LA,Marketing
Grace,29,60000,Chicago,Sales
Henry,38,100000,NYC,Engineering
Ivy,26,52000,LA,Marketing
Jack,31,70000,Chicago,Sales
"""
    csv_path = tmp_path / "employee_data.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def access_logs_csv(tmp_path):
    """Create a sample access logs CSV file."""
    csv_content = """url,client_ip,timestamp
https://example.com/products?utm_source=google,192.168.1.1,2024-01-15 10:30:00
https://shop.example.com/cart?id=123,10.0.0.5,2024-01-15 10:31:00
http://api.example.com/v1/users,172.16.0.1,2024-01-15 10:32:00
https://example.com/about,192.168.1.2,2024-01-15 10:33:00
https://blog.example.com/post/1?ref=twitter,10.0.0.6,2024-01-15 10:34:00
"""
    csv_path = tmp_path / "access_logs.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


class TestDataStoreUri:
    """Test DataStore.uri() - the unified data source interface."""

    def test_uri_local_csv(self, employee_csv):
        """Test DataStore.uri with local CSV file."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        result = ds.to_df()

        assert 'name' in result.columns
        assert len(result) == 10

    def test_uri_local_parquet(self, tmp_path):
        """Test DataStore.uri with local Parquet file."""
        from datastore import DataStore

        # Create parquet file
        pandas_df = pandas_original.DataFrame({'id': [1, 2, 3], 'value': [10.5, 20.5, 30.5]})
        parquet_path = str(tmp_path / "data.parquet")
        pandas_df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(parquet_path)
        result = ds.to_df()

        assert len(result) == 3
        assert 'id' in result.columns

    def test_uri_file_scheme(self, employee_csv):
        """Test DataStore.uri with file:// scheme."""
        from datastore import DataStore

        ds = DataStore.uri(f"file://{employee_csv}")
        result = ds.to_df()

        assert 'name' in result.columns
        assert len(result) == 10


class TestChainedOperations:
    """Test chained operations with DataStore.uri()."""

    def test_filter_groupby_sort_limit(self, employee_csv):
        """Test the main example from migration guide."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        result = (
            ds.filter(ds.age > 25)
            .filter(ds.salary > 50000)
            .groupby("city")
            .agg({"salary": ["mean", "sum", "count"]})
            .sort_values("salary_mean", ascending=False)  # Use flat name in SQL context
            .limit(10)
            .to_df()
        )

        # Verify result structure
        assert len(result) > 0
        assert result.index.name == 'city'  # With dict agg, city is index

    def test_sql_generation(self, employee_csv):
        """Test that SQL is generated correctly."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        query = ds.filter(ds.age > 25).groupby("city").agg({"salary": "mean"}).sort("salary").limit(5)

        sql = query.to_sql()

        assert 'SELECT' in sql
        assert 'ORDER BY' in sql or 'order by' in sql.lower()
        assert 'LIMIT' in sql or 'limit' in sql.lower()


class TestAccessorExamples:
    """Test accessor examples from the migration guide."""

    def test_url_accessor_domain(self, access_logs_csv):
        """Test URL accessor domain extraction with select()."""
        from datastore import DataStore

        ds = DataStore.uri(access_logs_csv)
        result = ds.select(
            'url',
            ds['url'].url.domain().as_('domain'),
        ).to_df()

        assert 'domain' in result.columns
        domains = result['domain'].tolist()
        assert len(domains) > 0
        assert all(d is not None and str(d) != '' for d in domains)

    def test_url_accessor_path(self, access_logs_csv):
        """Test URL accessor path extraction."""
        from datastore import DataStore

        ds = DataStore.uri(access_logs_csv)
        result = ds.select(
            'url',
            ds['url'].url.path().as_('path'),
        ).to_df()

        assert 'path' in result.columns

    def test_json_accessor_extract_string(self, tmp_path):
        """Test JSON accessor string extraction."""
        from datastore import DataStore

        pandas_df = pandas_original.DataFrame(
            {'response': ['{"name": "Alice", "id": 1}', '{"name": "Bob", "id": 2}', '{"error": "not_found"}']}
        )
        parquet_path = str(tmp_path / "json_data.parquet")
        pandas_df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(parquet_path)
        result = ds.select(
            'response',
            ds['response'].json.json_extract_string('name').as_('user_name'),
        ).to_df()

        assert 'user_name' in result.columns
        names = [str(n) for n in result['user_name'].tolist()]
        assert any('Alice' in n for n in names)

    def test_json_accessor_extract_int(self, tmp_path):
        """Test JSON accessor integer extraction."""
        from datastore import DataStore

        pandas_df = pandas_original.DataFrame({'response': ['{"name": "Alice", "id": 1}', '{"name": "Bob", "id": 2}']})
        parquet_path = str(tmp_path / "json_data.parquet")
        pandas_df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(parquet_path)
        result = ds.select(
            'response',
            ds['response'].json.json_extract_int('id').as_('user_id'),
        ).to_df()

        assert 'user_id' in result.columns
        ids = result['user_id'].tolist()
        assert 1 in ids or '1' in [str(i) for i in ids]

    def test_json_accessor_has(self, tmp_path):
        """Test JSON accessor has check."""
        from datastore import DataStore

        pandas_df = pandas_original.DataFrame({'response': ['{"name": "Alice"}', '{"error": "not_found"}']})
        parquet_path = str(tmp_path / "json_data.parquet")
        pandas_df.to_parquet(parquet_path, index=False)

        ds = DataStore.uri(parquet_path)
        result = ds.select(
            'response',
            ds['response'].json.json_has('error').as_('has_error'),
        ).to_df()

        assert 'has_error' in result.columns

    def test_string_accessor_upper(self, employee_csv):
        """Test string accessor upper()."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        result = ds.select(
            'name',
            ds['name'].str.upper().as_('name_upper'),
        ).to_df()

        assert 'name_upper' in result.columns
        assert all(str(n).isupper() for n in result['name_upper'].tolist())

    def test_string_accessor_len(self, employee_csv):
        """Test string accessor len()."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        result = ds.select(
            'name',
            ds['name'].str.len().as_('name_len'),
        ).to_df()

        assert 'name_len' in result.columns
        assert all(int(l) > 0 for l in result['name_len'].tolist())


class TestSQLStyle:
    """Test SQL style examples."""

    def test_fluent_sql_style(self, employee_csv):
        """Test fluent SQL style with DataStore.uri()."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        result = (
            ds.select("name", "city", "salary")
            .filter(ds.salary > 50000)
            .sort("salary", ascending=False)
            .limit(5)
            .to_df()
        )

        assert len(result) <= 5
        assert list(result.columns) == ['name', 'city', 'salary']

    def test_to_sql_method(self, employee_csv):
        """Test to_sql() method."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        query = ds.filter(ds.age > 30)

        sql = query.to_sql()

        assert 'SELECT' in sql
        assert 'WHERE' in sql
        assert '30' in sql

    def test_raw_sql(self, employee_csv):
        """Test DataStore().sql() for raw SQL execution."""
        from datastore import DataStore

        result = (
            DataStore()
            .sql(
                f"""
            SELECT city, AVG(salary) as avg_salary
            FROM file('{employee_csv}', 'CSVWithNames')
            WHERE age > 25
            GROUP BY city
            ORDER BY avg_salary DESC
        """
            )
            .to_df()
        )

        assert 'city' in result.columns
        assert 'avg_salary' in result.columns


class TestPerformanceTips:
    """Test performance optimization examples."""

    def test_select_specific_columns(self, employee_csv):
        """Test selecting specific columns."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        result = ds.select("name", "salary").to_df()

        assert list(result.columns) == ['name', 'salary']

    def test_limit_generates_sql_limit(self, employee_csv):
        """Test that limit() generates SQL LIMIT."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        query = ds.limit(5)

        sql = query.to_sql()

        assert 'LIMIT' in sql
        assert '5' in sql

    def test_early_filter(self, employee_csv):
        """Test early filtering generates WHERE clause."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        query = ds.filter(ds.age > 30)

        sql = query.to_sql()

        assert 'WHERE' in sql


class TestLazyExecution:
    """Test lazy execution behavior."""

    def test_operations_are_lazy(self, employee_csv):
        """Test that operations don't execute immediately."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)

        # These should not execute yet
        filtered = ds.filter(ds.age > 25)

        # Verify filtered is still a DataStore
        assert isinstance(filtered, DataStore)
        assert hasattr(filtered, 'to_sql')

        # Now trigger execution
        result = filtered.to_df()

        # Now it's a DataFrame
        assert isinstance(result, pandas_original.DataFrame)

    def test_chained_operations_compile_together(self, employee_csv):
        """Test that chained operations compile to SQL together."""
        from datastore import DataStore

        ds = DataStore.uri(employee_csv)
        query = ds.filter(ds.age > 25).groupby('city').agg({'salary': 'mean'}).sort('mean').limit(5)

        sql = query.to_sql()

        # All operations should be in the SQL
        assert 'WHERE' in sql or 'where' in sql.lower()
        assert 'ORDER BY' in sql or 'order by' in sql.lower()
        assert 'LIMIT' in sql or 'limit' in sql.lower()
