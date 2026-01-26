"""
Tests for DataStore.uri() factory method.

Tests the integration of URI parser with DataStore creation.
"""

import pytest
import os
from pathlib import Path
from datastore import DataStore


class TestFromURIFileSource:
    """Test DataStore.uri() with file sources."""

    def test_from_uri_local_csv(self):
        """Test creating DataStore from local CSV path."""
        # Use test dataset
        csv_path = "tests/dataset/users.csv"
        if os.path.exists(csv_path):
            ds = DataStore.uri(csv_path)
            assert ds is not None
            assert ds.source_type == "file"
            assert ds._table_function is not None

    def test_from_uri_file_scheme(self):
        """Test creating DataStore with file:// scheme."""
        csv_path = os.path.abspath("tests/dataset/users.csv")
        uri = f"file://{csv_path}"

        if os.path.exists(csv_path):
            ds = DataStore.uri(uri)
            assert ds is not None
            assert ds.source_type == "file"

    def test_from_uri_with_format_override(self):
        """Test URI with explicit format override."""
        csv_path = "tests/dataset/users.csv"
        if os.path.exists(csv_path):
            # Override format (even though .csv would auto-detect as CSV)
            ds = DataStore.uri(csv_path, format="CSV")
            assert ds is not None


class TestFromURIS3Source:
    """Test DataStore.uri() with S3 sources."""

    def test_from_uri_s3_basic(self):
        """Test creating DataStore from S3 URI."""
        uri = "s3://mybucket/data/file.parquet?nosign=true"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "s3"
        assert ds._table_function is not None

    def test_from_uri_s3_with_credentials(self):
        """Test S3 URI with credentials."""
        uri = "s3://mybucket/data.csv?access_key_id=TESTKEY&secret_access_key=TESTSECRET"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "s3"


class TestFromURIDatabaseSource:
    """Test DataStore.uri() with database sources."""

    def test_from_uri_mysql(self):
        """Test creating DataStore from MySQL URI."""
        uri = "mysql://root:password@localhost:3306/testdb/users"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "mysql"
        assert ds.table_name == "users"
        assert ds._table_function is not None

    def test_from_uri_postgresql(self):
        """Test creating DataStore from PostgreSQL URI."""
        uri = "postgresql://postgres:pass@localhost:5432/testdb/products"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "postgresql"
        assert ds.table_name == "products"

    def test_from_uri_postgres_alias(self):
        """Test postgres:// alias."""
        uri = "postgres://user:pass@host:5432/db/table"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "postgresql"

    def test_from_uri_mongodb(self):
        """Test creating DataStore from MongoDB URI."""
        uri = "mongodb://admin:pass@localhost:27017/testdb.users"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "mongodb"

    def test_from_uri_sqlite(self):
        """Test creating DataStore from SQLite URI."""
        uri = "sqlite:///path/to/test.db?table=users"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "sqlite"
        assert ds.table_name == "users"

    def test_from_uri_clickhouse(self):
        """Test creating DataStore from ClickHouse URI."""
        uri = "clickhouse://localhost:9000/default/events?user=default"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "clickhouse"
        assert ds.table_name == "events"


class TestFromURICloudStorage:
    """Test DataStore.uri() with cloud storage sources."""

    def test_from_uri_gcs(self):
        """Test creating DataStore from GCS URI."""
        uri = "gs://mybucket/path/data.parquet"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "gcs"

    def test_from_uri_azure(self):
        """Test creating DataStore from Azure URI."""
        uri = "az://mycontainer/path/data.csv?account_name=test"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "azureBlobStorage"

    def test_from_uri_hdfs(self):
        """Test creating DataStore from HDFS URI."""
        uri = "hdfs://namenode:9000/data/file.parquet"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "hdfs"


class TestFromURIHTTP:
    """Test DataStore.uri() with HTTP/HTTPS sources."""

    def test_from_uri_http(self):
        """Test creating DataStore from HTTP URI."""
        uri = "http://example.com/data/file.csv"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "url"

    def test_from_uri_https(self):
        """Test creating DataStore from HTTPS URI."""
        uri = "https://example.com/data/file.json"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "url"


class TestFromURIBigDataFormats:
    """Test DataStore.uri() with big data formats."""

    def test_from_uri_iceberg(self):
        """Test creating DataStore from Iceberg URI."""
        uri = "iceberg://catalog/namespace/table"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "iceberg"

    def test_from_uri_deltalake(self):
        """Test creating DataStore from Delta Lake URI."""
        uri = "deltalake:///path/to/delta/table"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "deltaLake"

    def test_from_uri_delta_alias(self):
        """Test delta:// alias."""
        uri = "delta:///path/to/table"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "deltaLake"

    def test_from_uri_hudi(self):
        """Test creating DataStore from Hudi URI."""
        uri = "hudi:///path/to/hudi/table"
        ds = DataStore.uri(uri)

        assert ds is not None
        assert ds.source_type == "hudi"


class TestFromURIParameterOverride:
    """Test parameter override in from_uri()."""

    def test_override_format(self):
        """Test overriding auto-detected format."""
        uri = "s3://bucket/file.csv"
        ds = DataStore.uri(uri, format="Parquet")

        # The format parameter should be passed through
        assert ds is not None

    def test_override_with_kwargs(self):
        """Test adding additional parameters."""
        uri = "s3://bucket/file.parquet"
        ds = DataStore.uri(uri, nosign=True, region="us-west-2")

        assert ds is not None


class TestFromURIErrorHandling:
    """Test error handling in from_uri()."""

    def test_empty_uri_error(self):
        """Test that empty URI raises error."""
        with pytest.raises(ValueError, match="URI cannot be empty"):
            DataStore.uri("")

    def test_unsupported_scheme_error(self):
        """Test that unsupported scheme raises error."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            DataStore.uri("ftp://example.com/file.csv")


class TestFromURIIntegrationWithExistingDataset:
    """Integration tests with actual test dataset if available."""

    def test_query_csv_with_uri(self):
        """Test querying CSV file using URI."""
        csv_path = "tests/dataset/users.csv"

        if os.path.exists(csv_path):
            ds = DataStore.uri(csv_path)
            ds.connect()

            # Try a simple query
            result = ds.select("*").limit(5).execute()
            assert result is not None
            assert len(result.rows) <= 5

    def test_query_with_filter(self):
        """Test querying with filter using URI-created DataStore."""
        csv_path = "tests/dataset/orders.csv"

        if os.path.exists(csv_path):
            ds = DataStore.uri(csv_path)
            ds.connect()

            # Query with limit
            result = ds.select("*").limit(10).execute()
            assert result is not None

    def test_file_uri_with_absolute_path(self):
        """Test file URI with absolute path."""
        csv_path = "tests/dataset/products.csv"

        if os.path.exists(csv_path):
            abs_path = os.path.abspath(csv_path)
            uri = f"file://{abs_path}"

            ds = DataStore.uri(uri)
            ds.connect()

            result = ds.select("*").limit(3).execute()
            assert result is not None


class TestFromURIMethodChaining:
    """Test method chaining with from_uri()."""

    def test_method_chaining(self):
        """Test that DataStore created via from_uri() supports method chaining."""
        csv_path = "tests/dataset/users.csv"

        if os.path.exists(csv_path):
            ds = DataStore.uri(csv_path)

            # Test method chaining
            query_ds = ds.select("*").limit(5)
            assert query_ds is not None
            assert query_ds != ds  # Should be immutable

    def test_complex_query_chain(self):
        """Test complex query chain with from_uri()."""
        csv_path = "tests/dataset/orders.csv"

        if os.path.exists(csv_path):
            ds = DataStore.uri(csv_path)
            ds.connect()

            # Build complex query
            query_ds = ds.select("*").limit(10).offset(5)

            # Should generate valid SQL
            sql = query_ds.to_sql()
            assert "LIMIT" in sql
            assert "OFFSET" in sql
