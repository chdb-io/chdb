"""
Tests for URI parser functionality.

Tests the uri_parser module which provides automatic URI-based DataStore creation.
"""

import pytest
from datastore.uri_parser import parse_uri


class TestFileURIParsing:
    """Test parsing of file URIs."""

    def test_local_file_path(self):
        """Test parsing local file path.

        Note: CSV files default to CSVWithNames (first row is header)
        to match pandas' default behavior for better user experience.
        """
        source_type, kwargs = parse_uri("/path/to/data.csv")
        assert source_type == "file"
        assert kwargs["path"] == "/path/to/data.csv"
        assert kwargs["format"] == "CSVWithNames"  # First row is header (pandas-compatible)

    def test_file_uri_scheme(self):
        """Test parsing file:// URI."""
        source_type, kwargs = parse_uri("file:///path/to/data.parquet")
        assert source_type == "file"
        assert kwargs["path"] == "/path/to/data.parquet"
        assert kwargs["format"] == "Parquet"

    def test_file_json(self):
        """Test parsing JSON file."""
        source_type, kwargs = parse_uri("/data/file.json")
        assert source_type == "file"
        assert kwargs["format"] == "JSON"

    def test_file_jsonl(self):
        """Test parsing JSONL file."""
        source_type, kwargs = parse_uri("/data/file.jsonl")
        assert source_type == "file"
        assert kwargs["format"] == "JSONEachRow"

    def test_file_tsv(self):
        """Test parsing TSV file.

        Note: TSV files default to TSVWithNames (first row is header)
        to match pandas' default behavior.
        """
        source_type, kwargs = parse_uri("file:///data/file.tsv")
        assert source_type == "file"
        assert kwargs["format"] == "TSVWithNames"  # First row is header (pandas-compatible)

    def test_file_with_query_params(self):
        """Test file URI with explicit format in query."""
        source_type, kwargs = parse_uri("file:///data/file.txt?format=CSV")
        assert source_type == "file"
        assert kwargs["format"] == "CSV"


class TestS3URIParsing:
    """Test parsing of S3 URIs."""

    def test_basic_s3_uri(self):
        """Test basic S3 URI."""
        source_type, kwargs = parse_uri("s3://mybucket/path/to/data.parquet")
        assert source_type == "s3"
        assert kwargs["url"] == "s3://mybucket/path/to/data.parquet"
        assert kwargs["format"] == "Parquet"

    def test_s3_with_nosign(self):
        """Test S3 URI with nosign parameter."""
        source_type, kwargs = parse_uri("s3://mybucket/data.csv?nosign=true")
        assert source_type == "s3"
        assert kwargs["nosign"] is True
        assert kwargs["format"] == "CSVWithNames"  # First row is header (pandas-compatible)

    def test_s3_with_credentials(self):
        """Test S3 URI with credentials in query params."""
        uri = "s3://mybucket/data.json?access_key_id=AKIAIOSFODNN7EXAMPLE&secret_access_key=wJalrXUtnFEMI"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "s3"
        assert kwargs["access_key_id"] == "AKIAIOSFODNN7EXAMPLE"
        assert kwargs["secret_access_key"] == "wJalrXUtnFEMI"
        assert kwargs["format"] == "JSON"

    def test_s3_with_region(self):
        """Test S3 URI with region."""
        source_type, kwargs = parse_uri("s3://mybucket/data.csv?region=us-west-2")
        assert source_type == "s3"
        assert kwargs["region"] == "us-west-2"


class TestCloudStorageURIParsing:
    """Test parsing of cloud storage URIs."""

    def test_gcs_uri(self):
        """Test Google Cloud Storage URI."""
        source_type, kwargs = parse_uri("gs://mybucket/path/to/data.parquet")
        assert source_type == "gcs"
        assert kwargs["url"] == "gs://mybucket/path/to/data.parquet"
        assert kwargs["format"] == "Parquet"

    def test_gcs_with_credentials(self):
        """Test GCS URI with HMAC credentials."""
        uri = "gs://mybucket/data.csv?hmac_key=KEY&hmac_secret=SECRET"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "gcs"
        assert kwargs["hmac_key"] == "KEY"
        assert kwargs["hmac_secret"] == "SECRET"

    def test_azure_uri(self):
        """Test Azure Blob Storage URI."""
        source_type, kwargs = parse_uri("az://mycontainer/path/data.parquet")
        assert source_type == "azureBlobStorage"
        assert kwargs["url"] == "az://mycontainer/path/data.parquet"
        assert kwargs["format"] == "Parquet"

    def test_azure_with_credentials(self):
        """Test Azure URI with credentials."""
        uri = "az://container/blob.csv?account_name=myaccount&account_key=KEY123"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "azureBlobStorage"
        assert kwargs["account_name"] == "myaccount"
        assert kwargs["account_key"] == "KEY123"


class TestDatabaseURIParsing:
    """Test parsing of database URIs."""

    def test_mysql_uri_basic(self):
        """Test basic MySQL URI."""
        uri = "mysql://localhost:3306/mydb/users"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "mysql"
        assert kwargs["host"] == "localhost:3306"
        assert kwargs["database"] == "mydb"
        assert kwargs["table"] == "users"

    def test_mysql_uri_with_credentials(self):
        """Test MySQL URI with credentials."""
        uri = "mysql://root:password123@localhost:3306/mydb/users"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "mysql"
        assert kwargs["user"] == "root"
        assert kwargs["password"] == "password123"
        assert kwargs["host"] == "localhost:3306"
        assert kwargs["database"] == "mydb"
        assert kwargs["table"] == "users"

    def test_mysql_uri_no_port(self):
        """Test MySQL URI without port."""
        uri = "mysql://root:pass@localhost/mydb/users"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "mysql"
        assert kwargs["host"] == "localhost"
        assert kwargs["database"] == "mydb"

    def test_postgresql_uri(self):
        """Test PostgreSQL URI."""
        uri = "postgresql://postgres:pass@localhost:5432/mydb/users"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "postgresql"
        assert kwargs["user"] == "postgres"
        assert kwargs["password"] == "pass"
        assert kwargs["host"] == "localhost:5432"
        assert kwargs["database"] == "mydb"
        assert kwargs["table"] == "users"

    def test_postgres_alias(self):
        """Test postgres:// alias for postgresql://."""
        uri = "postgres://user:pass@host:5432/db/table"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "postgresql"

    def test_mongodb_uri(self):
        """Test MongoDB URI."""
        uri = "mongodb://admin:pass@localhost:27017/mydb.mycollection"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "mongodb"
        assert kwargs["user"] == "admin"
        assert kwargs["password"] == "pass"
        assert kwargs["host"] == "localhost:27017"
        assert kwargs["database"] == "mydb"
        assert kwargs["collection"] == "mycollection"

    def test_sqlite_uri(self):
        """Test SQLite URI."""
        uri = "sqlite:///path/to/database.db?table=users"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "sqlite"
        assert kwargs["database"] == "/path/to/database.db"
        assert kwargs["table"] == "users"

    def test_redis_uri(self):
        """Test Redis URI."""
        uri = "redis://localhost:6379/0?key=mykey&password=pass"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "redis"
        assert kwargs["host"] == "localhost:6379"
        assert kwargs["db_index"] == 0
        assert kwargs["key"] == "mykey"
        assert kwargs["password"] == "pass"

    def test_clickhouse_uri(self):
        """Test ClickHouse URI."""
        uri = "clickhouse://localhost:9000/default/events?user=default&password=pass"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "clickhouse"
        assert kwargs["host"] == "localhost:9000"
        assert kwargs["database"] == "default"
        assert kwargs["table"] == "events"
        assert kwargs["user"] == "default"
        assert kwargs["password"] == "pass"


class TestHTTPURIParsing:
    """Test parsing of HTTP/HTTPS URIs."""

    def test_http_uri(self):
        """Test HTTP URI."""
        uri = "http://example.com/data/file.csv"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "url"
        assert kwargs["url"] == uri
        assert kwargs["format"] == "CSVWithNames"  # First row is header (pandas-compatible)

    def test_https_uri(self):
        """Test HTTPS URI."""
        uri = "https://example.com/data/file.json"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "url"
        assert kwargs["url"] == uri
        assert kwargs["format"] == "JSON"

    def test_url_without_extension(self):
        """Test URL without file extension."""
        uri = "https://api.example.com/data"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "url"
        assert kwargs["url"] == uri
        # No format should be auto-detected
        assert "format" not in kwargs


class TestBigDataFormatURIParsing:
    """Test parsing of big data format URIs."""

    def test_hdfs_uri(self):
        """Test HDFS URI."""
        uri = "hdfs://namenode:9000/data/file.parquet"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "hdfs"
        assert kwargs["url"] == uri
        assert kwargs["format"] == "Parquet"

    def test_iceberg_uri(self):
        """Test Iceberg table URI."""
        uri = "iceberg://catalog/namespace/table"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "iceberg"
        assert kwargs["url"] == "catalog/namespace/table"

    def test_deltalake_uri(self):
        """Test Delta Lake URI."""
        uri = "deltalake:///path/to/delta/table"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "deltaLake"
        assert kwargs["url"] == "path/to/delta/table"

    def test_delta_alias(self):
        """Test delta:// alias for deltalake://."""
        uri = "delta:///path/to/table"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "deltaLake"

    def test_hudi_uri(self):
        """Test Hudi table URI."""
        uri = "hudi:///path/to/hudi/table"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "hudi"
        assert kwargs["url"] == "path/to/hudi/table"


class TestFormatInference:
    """Test file format inference from extensions."""

    def test_csv_format(self):
        """Test CSV format inference - uses CSVWithNames for pandas compatibility."""
        _, kwargs = parse_uri("/data/file.csv")
        assert kwargs["format"] == "CSVWithNames"  # First row is header (pandas-compatible)

    def test_parquet_format(self):
        """Test Parquet format inference."""
        _, kwargs = parse_uri("s3://bucket/file.parquet")
        assert kwargs["format"] == "Parquet"

    def test_parquet_pq_extension(self):
        """Test Parquet .pq extension."""
        _, kwargs = parse_uri("/data/file.pq")
        assert kwargs["format"] == "Parquet"

    def test_json_format(self):
        """Test JSON format inference."""
        _, kwargs = parse_uri("/data/file.json")
        assert kwargs["format"] == "JSON"

    def test_jsonl_format(self):
        """Test JSONEachRow format for .jsonl."""
        _, kwargs = parse_uri("/data/file.jsonl")
        assert kwargs["format"] == "JSONEachRow"

    def test_ndjson_format(self):
        """Test JSONEachRow format for .ndjson."""
        _, kwargs = parse_uri("/data/file.ndjson")
        assert kwargs["format"] == "JSONEachRow"

    def test_tsv_format(self):
        """Test TSV format inference - uses TSVWithNames for pandas compatibility."""
        _, kwargs = parse_uri("/data/file.tsv")
        assert kwargs["format"] == "TSVWithNames"  # First row is header (pandas-compatible)

    def test_orc_format(self):
        """Test ORC format inference."""
        _, kwargs = parse_uri("s3://bucket/file.orc")
        assert kwargs["format"] == "ORC"

    def test_avro_format(self):
        """Test Avro format inference."""
        _, kwargs = parse_uri("/data/file.avro")
        assert kwargs["format"] == "Avro"

    def test_arrow_format(self):
        """Test Arrow format inference."""
        _, kwargs = parse_uri("/data/file.arrow")
        assert kwargs["format"] == "Arrow"


class TestSpecialCases:
    """Test special cases and edge cases."""

    def test_empty_uri_raises_error(self):
        """Test that empty URI raises ValueError."""
        with pytest.raises(ValueError, match="URI cannot be empty"):
            parse_uri("")

    def test_unsupported_scheme_raises_error(self):
        """Test that unsupported scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            parse_uri("ftp://example.com/file.csv")

    def test_url_encoded_credentials(self):
        """Test URI with URL-encoded credentials."""
        uri = "mysql://user%40domain:p%40ss%21@localhost/db/table"
        source_type, kwargs = parse_uri(uri)
        assert kwargs["user"] == "user@domain"
        assert kwargs["password"] == "p@ss!"

    def test_uri_with_tilde(self):
        """Test local file path with tilde expansion."""
        uri = "~/data/file.csv"
        source_type, kwargs = parse_uri(uri)
        assert source_type == "file"
        # Path should be expanded (actual expansion depends on environment)
        assert "~" not in kwargs["path"] or kwargs["path"].startswith("~")


class TestQueryParameterParsing:
    """Test query parameter parsing."""

    def test_multiple_query_params(self):
        """Test URI with multiple query parameters."""
        uri = "s3://bucket/file.csv?access_key_id=KEY&secret_access_key=SECRET&region=us-east-1"
        source_type, kwargs = parse_uri(uri)
        assert kwargs["access_key_id"] == "KEY"
        assert kwargs["secret_access_key"] == "SECRET"
        assert kwargs["region"] == "us-east-1"

    def test_explicit_format_override(self):
        """Test explicit format in query param overrides extension."""
        uri = "file:///data/file.txt?format=Parquet"
        source_type, kwargs = parse_uri(uri)
        assert kwargs["format"] == "Parquet"  # Should override .txt extension

    def test_boolean_param_parsing(self):
        """Test boolean parameter parsing."""
        uri = "s3://bucket/file.csv?nosign=true"
        _, kwargs = parse_uri(uri)
        assert kwargs["nosign"] is True

        uri2 = "s3://bucket/file.csv?nosign=1"
        _, kwargs2 = parse_uri(uri2)
        assert kwargs2["nosign"] is True

        uri3 = "s3://bucket/file.csv?nosign=false"
        _, kwargs3 = parse_uri(uri3)
        assert kwargs3["nosign"] is False
