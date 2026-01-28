"""
Tests for table functions integration.
"""

import sys
from pathlib import Path

# Add parent directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datastore import DataStore
from datastore.table_functions import (
    FileTableFunction,
    S3TableFunction,
    MySQLTableFunction,
    PostgreSQLTableFunction,
    MongoDBTableFunction,
    NumbersTableFunction,
    UrlTableFunction,
    AzureBlobStorageTableFunction,
    GcsTableFunction,
    RedisTableFunction,
    SQLiteTableFunction,
    RemoteTableFunction,
    IcebergTableFunction,
    DeltaLakeTableFunction,
    HudiTableFunction,
    GenerateRandomTableFunction,
    create_table_function,
)
from datastore.exceptions import DataStoreError, QueryError


class TestTableFunctionCreation:
    """Test table function factory."""

    def test_create_file_function(self):
        """Test creating file table function."""
        tf = create_table_function('file', path='data.csv', format='CSV')
        assert isinstance(tf, FileTableFunction)
        assert tf.can_read
        assert tf.can_write

    def test_create_s3_function(self):
        """Test creating S3 table function."""
        tf = create_table_function(
            's3', url='s3://bucket/data.parquet', access_key_id='KEY', secret_access_key='SECRET', format='Parquet'
        )
        assert isinstance(tf, S3TableFunction)
        assert tf.can_read
        assert tf.can_write

    def test_create_mysql_function(self):
        """Test creating MySQL table function."""
        tf = create_table_function(
            'mysql', host='localhost:3306', database='mydb', table='users', user='root', password='pass'
        )
        assert isinstance(tf, MySQLTableFunction)
        assert tf.can_read
        assert tf.can_write

    def test_create_mongodb_function(self):
        """Test creating MongoDB table function."""
        tf = create_table_function(
            'mongodb', host='localhost:27017', database='mydb', collection='users', user='admin', password='pass'
        )
        assert isinstance(tf, MongoDBTableFunction)
        assert tf.can_read
        assert not tf.can_write  # MongoDB is read-only

    def test_create_numbers_function(self):
        """Test creating numbers table function."""
        tf = create_table_function('numbers', count=100)
        assert isinstance(tf, NumbersTableFunction)
        assert tf.can_read
        assert not tf.can_write

    def test_unsupported_source_type(self):
        """Test error on unsupported source type."""
        with pytest.raises(DataStoreError) as exc_info:
            create_table_function('unsupported_type', param='value')
        assert 'Unsupported source type' in str(exc_info.value)


class TestTableFunctionSQL:
    """Test SQL generation for table functions."""

    def test_file_sql_basic(self):
        """Test basic file() SQL generation."""
        tf = FileTableFunction(path='data.csv', format='CSV')
        sql = tf.to_sql()
        assert sql == "file('data.csv', 'CSV')"

    def test_file_sql_with_structure(self):
        """Test file() with structure."""
        tf = FileTableFunction(path='data.csv', format='CSV', structure='id UInt32, name String')
        sql = tf.to_sql()
        assert sql == "file('data.csv', 'CSV', 'id UInt32, name String')"

    def test_s3_sql_with_credentials(self):
        """Test S3() with credentials."""
        tf = S3TableFunction(
            url='s3://bucket/data.parquet', access_key_id='KEY', secret_access_key='SECRET', format='Parquet'
        )
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.parquet', 'KEY', 'SECRET', 'Parquet')"

    def test_s3_sql_nosign(self):
        """Test S3() with NOSIGN."""
        tf = S3TableFunction(url='s3://bucket/data.csv', format='CSV', nosign=True)
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.csv', NOSIGN, 'CSV')"

    def test_mysql_sql(self):
        """Test mysql() SQL generation."""
        tf = MySQLTableFunction(host='localhost:3306', database='mydb', table='users', user='root', password='pass')
        sql = tf.to_sql()
        assert sql == "mysql('localhost:3306', 'mydb', 'users', 'root', 'pass')"

    def test_postgresql_sql(self):
        """Test postgresql() SQL generation."""
        tf = PostgreSQLTableFunction(
            host='localhost:5432', database='mydb', table='public.users', user='postgres', password='pass'
        )
        sql = tf.to_sql()
        assert sql == "postgresql('localhost:5432', 'mydb', 'public.users', 'postgres', 'pass')"

    def test_numbers_sql_count_only(self):
        """Test numbers() with count only."""
        tf = NumbersTableFunction(count=100)
        sql = tf.to_sql()
        assert sql == "numbers(100)"

    def test_numbers_sql_with_start(self):
        """Test numbers() with start and count."""
        tf = NumbersTableFunction(start=10, count=20)
        sql = tf.to_sql()
        assert sql == "numbers(10, 20)"

    def test_numbers_sql_with_step(self):
        """Test numbers() with start, count, and step."""
        tf = NumbersTableFunction(start=0, count=10, step=2)
        sql = tf.to_sql()
        assert sql == "numbers(0, 10, 2)"


class TestDataStoreWithTableFunctions:
    """Test DataStore integration with table functions."""

    def test_datastore_file(self):
        """Test DataStore with file source."""
        ds = DataStore("file", path="data.csv", format="CSV")
        assert ds._table_function is not None
        assert isinstance(ds._table_function, FileTableFunction)

        # Check SQL generation
        sql = ds.select("*").to_sql()
        assert "FROM file('data.csv', 'CSV')" in sql

    def test_datastore_s3(self):
        """Test DataStore with S3 source."""
        ds = DataStore(
            "s3", url="s3://bucket/data.parquet", access_key_id="KEY", secret_access_key="SECRET", format="Parquet"
        )
        assert ds._table_function is not None
        assert isinstance(ds._table_function, S3TableFunction)

        sql = ds.select("*").to_sql()
        assert "FROM s3(" in sql

    def test_datastore_mysql(self):
        """Test DataStore with MySQL source."""
        ds = DataStore("mysql", host="localhost:3306", database="mydb", table="users", user="root", password="pass")
        assert ds._table_function is not None

        sql = ds.select("*").filter(ds.age > 18).to_sql()
        assert "FROM mysql(" in sql
        assert "WHERE" in sql

    def test_datastore_numbers(self):
        """Test DataStore with numbers source."""
        ds = DataStore("numbers", count=100)
        assert ds._table_function is not None

        sql = ds.select("*").limit(10).to_sql()
        assert "FROM numbers(100)" in sql
        assert "LIMIT 10" in sql

    def test_datastore_with_format_settings(self):
        """Test DataStore with format settings."""
        ds = DataStore("file", path="data.csv", format="CSV")
        ds = ds.with_format_settings(format_csv_delimiter='|', input_format_csv_skip_first_lines=1)

        assert ds._format_settings['format_csv_delimiter'] == '|'
        assert ds._format_settings['input_format_csv_skip_first_lines'] == 1

        sql = ds.select("*").to_sql()
        assert "SETTINGS" in sql
        assert "format_csv_delimiter='|'" in sql

    def test_datastore_filter_and_select(self):
        """Test DataStore with filtering and selection."""
        ds = DataStore("file", path="sales.csv", format="CSV", structure="id UInt32, revenue Float64, date Date")

        sql = ds.select("id", "revenue").filter(ds.revenue > 1000).to_sql()
        assert "SELECT" in sql
        assert "id" in sql
        assert "revenue" in sql
        assert "FROM file(" in sql
        assert "WHERE" in sql


class TestReadOnlyTableFunctions:
    """Test read-only table function restrictions."""

    def test_mongodb_write_error(self):
        """Test that MongoDB table function rejects writes."""
        ds = DataStore(
            "mongodb", host="localhost:27017", database="mydb", collection="users", user="admin", password="pass"
        )

        # MongoDB is read-only, so insert should fail
        ds_insert = ds.insert_into("id", "name").insert_values((1, "Alice"))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()

        assert "does not support writing" in str(exc_info.value)

    def test_numbers_write_error(self):
        """Test that numbers table function rejects writes."""
        ds = DataStore("numbers", count=100)

        ds_insert = ds.insert_into("number").insert_values((42,))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()

        assert "does not support writing" in str(exc_info.value)


class TestTableFunctionSettings:
    """Test format settings with table functions."""

    def test_csv_settings(self):
        """Test CSV format settings."""
        tf = FileTableFunction(path="data.csv", format="CSV")
        tf.with_settings(format_csv_delimiter='|', input_format_csv_skip_first_lines=1)

        assert tf.settings['format_csv_delimiter'] == '|'
        assert tf.settings['input_format_csv_skip_first_lines'] == 1

    def test_parquet_settings(self):
        """Test Parquet format settings."""
        tf = S3TableFunction(url="s3://bucket/data.parquet", format="Parquet", nosign=True)
        tf.with_settings(input_format_parquet_filter_push_down=1, input_format_parquet_bloom_filter_push_down=1)

        assert tf.settings['input_format_parquet_filter_push_down'] == 1
        assert tf.settings['input_format_parquet_bloom_filter_push_down'] == 1


class TestComplexQueries:
    """Test complex queries with table functions."""

    def test_join_different_sources(self):
        """Test joining different data sources."""
        # File source
        file_ds = DataStore("file", path="sales.csv", format="CSV")

        # MySQL source
        mysql_ds = DataStore(
            "mysql", host="localhost:3306", database="mydb", table="customers", user="root", password="pass"
        )

        # Join them
        joined = mysql_ds.join(file_ds, left_on="id", right_on="customer_id").select("name", "product", "revenue")

        sql = joined.to_sql()
        assert "JOIN" in sql
        assert "mysql(" in sql

    def test_aggregation_with_table_function(self):
        """Test aggregation on table function."""
        ds = DataStore("file", path="sales.csv", format="CSV")

        from datastore.functions import Sum, Count

        sql = (
            ds.groupby("category")
            .select("category", Sum(ds.revenue).as_("total_revenue"), Count("*").as_("count"))
            .to_sql()
        )

        assert "GROUP BY" in sql
        assert "sum(" in sql.lower()
        assert "count(" in sql.lower()

    def test_subquery_with_table_function(self):
        """Test subquery with table function."""
        ds = DataStore("numbers", count=100)

        subquery = ds.select("*").filter(ds.number < 50).as_("sub")

        # Use subquery in another query would require more complex setup
        # For now, just test that it generates correct SQL
        sql = subquery.to_sql()
        assert "FROM numbers(100)" in sql
        assert "WHERE" in sql


class TestErrorCases:
    """Test error handling."""

    def test_file_missing_path(self):
        """Test error when path is missing for file()."""
        tf = FileTableFunction(format="CSV")

        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()

        assert "'path' parameter is required" in str(exc_info.value)

    def test_s3_missing_url(self):
        """Test error when URL is missing for s3()."""
        tf = S3TableFunction(format="CSV")

        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()

        assert "'url' or 'path' parameter is required" in str(exc_info.value)

    def test_mysql_missing_params(self):
        """Test error when required params are missing for mysql()."""
        tf = MySQLTableFunction(host="localhost:3306")

        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()

        assert "required for mysql()" in str(exc_info.value)

    def test_numbers_missing_count(self):
        """Test error when count is missing for numbers()."""
        tf = NumbersTableFunction()

        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()

        assert "'count' parameter is required" in str(exc_info.value)


class TestFormatParam:
    """Test _format_param method with different types."""

    def test_format_none(self):
        """Test formatting None."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param(None) == 'NULL'

    def test_format_bool_true(self):
        """Test formatting boolean True."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param(True) == '1'

    def test_format_bool_false(self):
        """Test formatting boolean False."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param(False) == '0'

    def test_format_string(self):
        """Test formatting string."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param('hello') == "'hello'"

    def test_format_string_with_quotes(self):
        """Test formatting string with single quotes (escaping)."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param("it's a test") == "'it''s a test'"

    def test_format_int(self):
        """Test formatting integer."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param(42) == '42'

    def test_format_float(self):
        """Test formatting float."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param(3.14) == '3.14'

    def test_format_other_type(self):
        """Test formatting other types (e.g., list)."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._format_param([1, 2, 3]) == "'[1, 2, 3]'"


class TestBuildSettingsClause:
    """Test _build_settings_clause method."""

    def test_empty_settings(self):
        """Test with no settings."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        assert tf._build_settings_clause() == ""

    def test_string_settings(self):
        """Test with string settings."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        tf.with_settings(format_csv_delimiter='|')
        assert tf._build_settings_clause() == " SETTINGS format_csv_delimiter='|'"

    def test_numeric_settings(self):
        """Test with numeric settings."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        tf.with_settings(input_format_csv_skip_first_lines=1)
        assert tf._build_settings_clause() == " SETTINGS input_format_csv_skip_first_lines=1"

    def test_mixed_settings(self):
        """Test with mixed string and numeric settings."""
        tf = FileTableFunction(path="test.csv", format="CSV")
        tf.with_settings(format_csv_delimiter='|', input_format_csv_skip_first_lines=1)
        clause = tf._build_settings_clause()
        assert "SETTINGS" in clause
        assert "format_csv_delimiter='|'" in clause
        assert "input_format_csv_skip_first_lines=1" in clause


class TestUrlTableFunction:
    """Test URL table function."""

    def test_url_basic(self):
        """Test basic URL table function."""
        tf = UrlTableFunction(url="https://example.com/data.csv", format="CSV")
        sql = tf.to_sql()
        assert sql == "url('https://example.com/data.csv', 'CSV')"

    def test_url_with_structure(self):
        """Test URL with structure."""
        tf = UrlTableFunction(
            url="https://example.com/data.json", format="JSONEachRow", structure="id UInt32, name String"
        )
        sql = tf.to_sql()
        assert sql == "url('https://example.com/data.json', 'JSONEachRow', 'id UInt32, name String')"

    def test_url_with_headers_list(self):
        """Test URL with headers as list."""
        tf = UrlTableFunction(
            url="https://example.com/data.csv", format="CSV", headers=["Authorization: Bearer token", "X-Custom: value"]
        )
        sql = tf.to_sql()
        assert "url('https://example.com/data.csv', 'CSV')" in sql
        assert "HEADERS('Authorization: Bearer token', 'X-Custom: value')" in sql

    def test_url_with_headers_string(self):
        """Test URL with headers as string."""
        tf = UrlTableFunction(url="https://example.com/data.csv", format="CSV", headers="Authorization: Bearer token")
        sql = tf.to_sql()
        assert "HEADERS('Authorization: Bearer token')" in sql

    def test_url_missing_url(self):
        """Test error when URL is missing."""
        tf = UrlTableFunction(format="CSV")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'url' parameter is required" in str(exc_info.value)

    def test_url_can_read_write(self):
        """Test URL can read and write."""
        tf = UrlTableFunction(url="https://example.com/data.csv", format="CSV")
        assert tf.can_read
        assert tf.can_write


class TestS3TableFunctionExtended:
    """Extended tests for S3 table function."""

    def test_s3_with_session_token(self):
        """Test S3 with session token."""
        tf = S3TableFunction(
            url="s3://bucket/data.csv",
            access_key_id="KEY",
            secret_access_key="SECRET",
            session_token="TOKEN",
            format="CSV",
        )
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.csv', 'KEY', 'SECRET', 'TOKEN', 'CSV')"

    def test_s3_with_structure(self):
        """Test S3 with structure."""
        tf = S3TableFunction(url="s3://bucket/data.csv", format="CSV", structure="id UInt32, name String", nosign=True)
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.csv', NOSIGN, 'CSV', 'id UInt32, name String')"

    def test_s3_with_compression(self):
        """Test S3 with compression."""
        tf = S3TableFunction(url="s3://bucket/data.csv.gz", format="CSV", compression="gzip", nosign=True)
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.csv.gz', NOSIGN, 'CSV', 'gzip')"

    def test_s3_with_path_param(self):
        """Test S3 with 'path' parameter instead of 'url'."""
        tf = S3TableFunction(path="s3://bucket/data.csv", format="CSV", nosign=True)
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.csv', NOSIGN, 'CSV')"

    def test_s3_with_structure_and_compression(self):
        """Test S3 with both structure and compression."""
        tf = S3TableFunction(
            url="s3://bucket/data.csv.gz",
            format="CSV",
            structure="id UInt32, name String",
            compression="gzip",
            nosign=True,
        )
        sql = tf.to_sql()
        assert sql == "s3('s3://bucket/data.csv.gz', NOSIGN, 'CSV', 'id UInt32, name String', 'gzip')"


class TestAzureBlobStorageTableFunction:
    """Test Azure Blob Storage table function."""

    def test_azure_basic(self):
        """Test basic Azure Blob Storage."""
        tf = AzureBlobStorageTableFunction(
            connection_string="DefaultEndpointsProtocol=https;...",
            container="mycontainer",
            path="data.csv",
            format="CSV",
        )
        sql = tf.to_sql()
        assert sql == "azureBlobStorage('DefaultEndpointsProtocol=https;...', 'mycontainer', 'data.csv', 'CSV')"

    def test_azure_with_structure(self):
        """Test Azure with structure."""
        tf = AzureBlobStorageTableFunction(
            connection_string="DefaultEndpointsProtocol=https;...",
            container="mycontainer",
            path="data.csv",
            format="CSV",
            structure="id UInt32, name String",
        )
        sql = tf.to_sql()
        assert "azureBlobStorage(" in sql
        assert "'id UInt32, name String'" in sql

    def test_azure_missing_connection_string(self):
        """Test error when connection_string is missing."""
        tf = AzureBlobStorageTableFunction(container="mycontainer", path="data.csv", format="CSV")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'connection_string' and 'container' are required" in str(exc_info.value)

    def test_azure_missing_container(self):
        """Test error when container is missing."""
        tf = AzureBlobStorageTableFunction(connection_string="conn_str", path="data.csv", format="CSV")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'connection_string' and 'container' are required" in str(exc_info.value)

    def test_azure_can_read_write(self):
        """Test Azure can read and write."""
        tf = AzureBlobStorageTableFunction(
            connection_string="conn_str", container="mycontainer", path="data.csv", format="CSV"
        )
        assert tf.can_read
        assert tf.can_write


class TestGcsTableFunction:
    """Test Google Cloud Storage table function."""

    def test_gcs_nosign(self):
        """Test GCS with NOSIGN."""
        tf = GcsTableFunction(url="https://storage.googleapis.com/bucket/data.csv", format="CSV", nosign=True)
        sql = tf.to_sql()
        assert sql == "gcs('https://storage.googleapis.com/bucket/data.csv', NOSIGN, 'CSV')"

    def test_gcs_with_hmac(self):
        """Test GCS with HMAC credentials."""
        tf = GcsTableFunction(
            url="https://storage.googleapis.com/bucket/data.csv",
            hmac_key="HMAC_KEY",
            hmac_secret="HMAC_SECRET",
            format="CSV",
        )
        sql = tf.to_sql()
        assert sql == "gcs('https://storage.googleapis.com/bucket/data.csv', 'HMAC_KEY', 'HMAC_SECRET', 'CSV')"

    def test_gcs_with_structure(self):
        """Test GCS with structure."""
        tf = GcsTableFunction(
            url="https://storage.googleapis.com/bucket/data.csv",
            format="CSV",
            structure="id UInt32, name String",
            nosign=True,
        )
        sql = tf.to_sql()
        assert "'id UInt32, name String'" in sql

    def test_gcs_with_path_param(self):
        """Test GCS with 'path' parameter."""
        tf = GcsTableFunction(path="https://storage.googleapis.com/bucket/data.csv", format="CSV", nosign=True)
        sql = tf.to_sql()
        assert "gcs('https://storage.googleapis.com/bucket/data.csv'" in sql

    def test_gcs_missing_url(self):
        """Test error when URL is missing."""
        tf = GcsTableFunction(format="CSV")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'url' or 'path' parameter is required" in str(exc_info.value)

    def test_gcs_can_read_write(self):
        """Test GCS can read and write."""
        tf = GcsTableFunction(url="https://storage.googleapis.com/bucket/data.csv", format="CSV", nosign=True)
        assert tf.can_read
        assert tf.can_write


class TestRedisTableFunction:
    """Test Redis table function."""

    def test_redis_basic(self):
        """Test basic Redis."""
        tf = RedisTableFunction(host="localhost:6379", key="mykey", structure="id UInt32, value String")
        sql = tf.to_sql()
        assert sql == "redis('localhost:6379', 'mykey', 'id UInt32, value String')"

    def test_redis_with_password(self):
        """Test Redis with password."""
        tf = RedisTableFunction(
            host="localhost:6379", key="mykey", structure="id UInt32, value String", password="mypassword"
        )
        sql = tf.to_sql()
        assert sql == "redis('localhost:6379', 'mykey', 'id UInt32, value String', 'mypassword', 0)"

    def test_redis_with_password_and_db(self):
        """Test Redis with password and database index."""
        tf = RedisTableFunction(
            host="localhost:6379", key="mykey", structure="id UInt32, value String", password="mypassword", db_index=2
        )
        sql = tf.to_sql()
        assert sql == "redis('localhost:6379', 'mykey', 'id UInt32, value String', 'mypassword', 2)"

    def test_redis_missing_params(self):
        """Test error when required params are missing."""
        tf = RedisTableFunction(host="localhost:6379")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'host', 'key', and 'structure' are required" in str(exc_info.value)

    def test_redis_can_read_write(self):
        """Test Redis can read and write."""
        tf = RedisTableFunction(host="localhost:6379", key="mykey", structure="id UInt32")
        assert tf.can_read
        assert tf.can_write


class TestSQLiteTableFunction:
    """Test SQLite table function."""

    def test_sqlite_basic(self):
        """Test basic SQLite."""
        tf = SQLiteTableFunction(database_path="/path/to/db.sqlite", table="users")
        sql = tf.to_sql()
        assert sql == "sqlite('/path/to/db.sqlite', 'users')"

    def test_sqlite_with_path_param(self):
        """Test SQLite with 'path' parameter."""
        tf = SQLiteTableFunction(path="/path/to/db.sqlite", table="users")
        sql = tf.to_sql()
        assert sql == "sqlite('/path/to/db.sqlite', 'users')"

    def test_sqlite_missing_params(self):
        """Test error when required params are missing."""
        tf = SQLiteTableFunction(database_path="/path/to/db.sqlite")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'database_path' and 'table' are required" in str(exc_info.value)

    def test_sqlite_can_read_only(self):
        """Test SQLite is read-only."""
        tf = SQLiteTableFunction(database_path="/path/to/db.sqlite", table="users")
        assert tf.can_read
        assert not tf.can_write


class TestRemoteTableFunction:
    """Test Remote/RemoteSecure table function."""

    def test_remote_basic(self):
        """Test basic remote."""
        tf = RemoteTableFunction(host="localhost:9000", database="mydb", table="users")
        sql = tf.to_sql()
        assert sql == "remote('localhost:9000', 'mydb', 'users', 'default', '')"

    def test_remote_with_credentials(self):
        """Test remote with credentials."""
        tf = RemoteTableFunction(host="localhost:9000", database="mydb", table="users", user="admin", password="secret")
        sql = tf.to_sql()
        assert sql == "remote('localhost:9000', 'mydb', 'users', 'admin', 'secret')"

    def test_remote_secure(self):
        """Test remoteSecure."""
        tf = RemoteTableFunction(host="localhost:9440", database="mydb", table="users", secure=True)
        sql = tf.to_sql()
        assert sql.startswith("remoteSecure(")

    def test_remote_missing_params(self):
        """Test error when required params are missing."""
        tf = RemoteTableFunction(host="localhost:9000")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'host' and 'table' are required" in str(exc_info.value)

    def test_remote_can_read_write(self):
        """Test remote can read and write."""
        tf = RemoteTableFunction(host="localhost:9000", table="users")
        assert tf.can_read
        assert tf.can_write


class TestIcebergTableFunction:
    """Test Iceberg table function."""

    def test_iceberg_basic(self):
        """Test basic Iceberg."""
        tf = IcebergTableFunction(url="s3://bucket/iceberg/table")
        sql = tf.to_sql()
        assert sql == "iceberg('s3://bucket/iceberg/table')"

    def test_iceberg_with_credentials(self):
        """Test Iceberg with credentials."""
        tf = IcebergTableFunction(url="s3://bucket/iceberg/table", access_key_id="KEY", secret_access_key="SECRET")
        sql = tf.to_sql()
        assert sql == "iceberg('s3://bucket/iceberg/table', 'KEY', 'SECRET')"

    def test_iceberg_with_format(self):
        """Test Iceberg with format."""
        tf = IcebergTableFunction(
            url="s3://bucket/iceberg/table", access_key_id="KEY", secret_access_key="SECRET", format="Parquet"
        )
        sql = tf.to_sql()
        assert sql == "iceberg('s3://bucket/iceberg/table', 'KEY', 'SECRET', 'Parquet')"

    def test_iceberg_with_structure(self):
        """Test Iceberg with structure."""
        tf = IcebergTableFunction(
            url="s3://bucket/iceberg/table",
            access_key_id="KEY",
            secret_access_key="SECRET",
            format="Parquet",
            structure="id UInt32, name String",
        )
        sql = tf.to_sql()
        assert "'id UInt32, name String'" in sql

    def test_iceberg_missing_url(self):
        """Test error when URL is missing."""
        tf = IcebergTableFunction()
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'url' or 'path' parameter is required" in str(exc_info.value)

    def test_iceberg_can_read_only(self):
        """Test Iceberg is read-only."""
        tf = IcebergTableFunction(url="s3://bucket/iceberg/table")
        assert tf.can_read
        assert not tf.can_write


class TestDeltaLakeTableFunction:
    """Test DeltaLake table function."""

    def test_deltalake_basic(self):
        """Test basic DeltaLake."""
        tf = DeltaLakeTableFunction(url="s3://bucket/delta/table")
        sql = tf.to_sql()
        assert sql == "deltaLake('s3://bucket/delta/table')"

    def test_deltalake_with_credentials(self):
        """Test DeltaLake with credentials."""
        tf = DeltaLakeTableFunction(url="s3://bucket/delta/table", access_key_id="KEY", secret_access_key="SECRET")
        sql = tf.to_sql()
        assert sql == "deltaLake('s3://bucket/delta/table', 'KEY', 'SECRET')"

    def test_deltalake_with_path_param(self):
        """Test DeltaLake with 'path' parameter."""
        tf = DeltaLakeTableFunction(path="s3://bucket/delta/table")
        sql = tf.to_sql()
        assert sql == "deltaLake('s3://bucket/delta/table')"

    def test_deltalake_missing_url(self):
        """Test error when URL is missing."""
        tf = DeltaLakeTableFunction()
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'url' or 'path' parameter is required" in str(exc_info.value)

    def test_deltalake_can_read_only(self):
        """Test DeltaLake is read-only."""
        tf = DeltaLakeTableFunction(url="s3://bucket/delta/table")
        assert tf.can_read
        assert not tf.can_write


class TestHudiTableFunction:
    """Test Hudi table function."""

    def test_hudi_basic(self):
        """Test basic Hudi."""
        tf = HudiTableFunction(url="s3://bucket/hudi/table")
        sql = tf.to_sql()
        assert sql == "hudi('s3://bucket/hudi/table')"

    def test_hudi_with_credentials(self):
        """Test Hudi with credentials."""
        tf = HudiTableFunction(url="s3://bucket/hudi/table", access_key_id="KEY", secret_access_key="SECRET")
        sql = tf.to_sql()
        assert sql == "hudi('s3://bucket/hudi/table', 'KEY', 'SECRET')"

    def test_hudi_with_path_param(self):
        """Test Hudi with 'path' parameter."""
        tf = HudiTableFunction(path="s3://bucket/hudi/table")
        sql = tf.to_sql()
        assert sql == "hudi('s3://bucket/hudi/table')"

    def test_hudi_missing_url(self):
        """Test error when URL is missing."""
        tf = HudiTableFunction()
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'url' or 'path' parameter is required" in str(exc_info.value)

    def test_hudi_can_read_only(self):
        """Test Hudi is read-only."""
        tf = HudiTableFunction(url="s3://bucket/hudi/table")
        assert tf.can_read
        assert not tf.can_write


class TestGenerateRandomTableFunction:
    """Test GenerateRandom table function."""

    def test_generaterandom_basic(self):
        """Test basic generateRandom."""
        tf = GenerateRandomTableFunction(structure="id UInt32, name String")
        sql = tf.to_sql()
        assert sql == "generateRandom('id UInt32, name String')"

    def test_generaterandom_with_seed(self):
        """Test generateRandom with seed."""
        tf = GenerateRandomTableFunction(structure="id UInt32, name String", random_seed=42)
        sql = tf.to_sql()
        assert sql == "generateRandom('id UInt32, name String', 42)"

    def test_generaterandom_with_max_string_length(self):
        """Test generateRandom with max_string_length."""
        tf = GenerateRandomTableFunction(structure="id UInt32, name String", random_seed=42, max_string_length=20)
        sql = tf.to_sql()
        assert sql == "generateRandom('id UInt32, name String', 42, 20)"

    def test_generaterandom_with_max_array_length(self):
        """Test generateRandom with all parameters."""
        tf = GenerateRandomTableFunction(
            structure="id UInt32, tags Array(String)", random_seed=42, max_string_length=20, max_array_length=5
        )
        sql = tf.to_sql()
        assert sql == "generateRandom('id UInt32, tags Array(String)', 42, 20, 5)"

    def test_generaterandom_missing_structure(self):
        """Test error when structure is missing."""
        tf = GenerateRandomTableFunction()
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'structure' parameter is required" in str(exc_info.value)

    def test_generaterandom_can_read_only(self):
        """Test generateRandom is read-only."""
        tf = GenerateRandomTableFunction(structure="id UInt32")
        assert tf.can_read
        assert not tf.can_write


class TestFileTableFunctionExtended:
    """Extended tests for File table function."""

    def test_file_with_compression(self):
        """Test file with compression."""
        tf = FileTableFunction(path="data.csv.gz", format="CSV", compression="gzip")
        sql = tf.to_sql()
        assert sql == "file('data.csv.gz', 'CSV', 'auto', 'gzip')"

    def test_file_with_structure_and_compression(self):
        """Test file with structure and compression."""
        tf = FileTableFunction(path="data.csv.gz", format="CSV", structure="id UInt32, name String", compression="gzip")
        sql = tf.to_sql()
        assert sql == "file('data.csv.gz', 'CSV', 'id UInt32, name String', 'gzip')"


class TestTableFunctionAliases:
    """Test table function factory with aliases."""

    def test_http_alias(self):
        """Test 'http' alias for url."""
        tf = create_table_function('http', url="http://example.com/data.csv", format="CSV")
        assert isinstance(tf, UrlTableFunction)

    def test_https_alias(self):
        """Test 'https' alias for url."""
        tf = create_table_function('https', url="https://example.com/data.csv", format="CSV")
        assert isinstance(tf, UrlTableFunction)

    def test_postgres_alias(self):
        """Test 'postgres' alias for postgresql."""
        tf = create_table_function('postgres', host="localhost:5432", database="mydb", table="users", user="postgres")
        assert isinstance(tf, PostgreSQLTableFunction)

    def test_mongo_alias(self):
        """Test 'mongo' alias for mongodb."""
        tf = create_table_function('mongo', host="localhost:27017", database="mydb", collection="users", user="admin")
        assert isinstance(tf, MongoDBTableFunction)

    def test_clickhouse_alias(self):
        """Test 'clickhouse' alias for remote."""
        tf = create_table_function('clickhouse', host="localhost:9000", table="users")
        assert isinstance(tf, RemoteTableFunction)

    def test_delta_alias(self):
        """Test 'delta' alias for deltalake."""
        tf = create_table_function('delta', url="s3://bucket/delta/table")
        assert isinstance(tf, DeltaLakeTableFunction)

    def test_azureblob_alias(self):
        """Test 'azureblob' alias for azure."""
        tf = create_table_function(
            'azureblob', connection_string="conn", container="cont", path="data.csv", format="CSV"
        )
        assert isinstance(tf, AzureBlobStorageTableFunction)


class TestMongoDBWithTable:
    """Test MongoDB with 'table' parameter instead of 'collection'."""

    def test_mongodb_with_table_param(self):
        """Test MongoDB with 'table' parameter."""
        tf = MongoDBTableFunction(
            host="localhost:27017",
            database="mydb",
            table="users",  # Using 'table' instead of 'collection'
            user="admin",
            password="pass",
        )
        sql = tf.to_sql()
        assert sql == "mongodb('localhost:27017', 'mydb', 'users', 'admin', 'pass')"

    def test_mongodb_with_structure(self):
        """Test MongoDB with structure."""
        tf = MongoDBTableFunction(
            host="localhost:27017",
            database="mydb",
            collection="users",
            user="admin",
            password="pass",
            structure="id UInt32, name String",
        )
        sql = tf.to_sql()
        assert "'id UInt32, name String'" in sql

    def test_mongodb_missing_collection(self):
        """Test error when both collection and table are missing."""
        tf = MongoDBTableFunction(host="localhost:27017", database="mydb", user="admin", password="pass")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "'host', 'database', 'collection', and 'user' are required" in str(exc_info.value)


class TestPostgreSQLMissingParams:
    """Test PostgreSQL error cases."""

    def test_postgresql_missing_params(self):
        """Test error when required params are missing."""
        tf = PostgreSQLTableFunction(host="localhost:5432", database="mydb")
        with pytest.raises(DataStoreError) as exc_info:
            tf.to_sql()
        assert "required for postgresql()" in str(exc_info.value)

    def test_postgresql_can_read_write(self):
        """Test PostgreSQL can read and write."""
        tf = PostgreSQLTableFunction(host="localhost:5432", database="mydb", table="users", user="postgres")
        assert tf.can_read
        assert tf.can_write


class TestTableFunctionWithSettings:
    """Test with_settings chaining."""

    def test_with_settings_returns_self(self):
        """Test that with_settings returns self for chaining."""
        tf = FileTableFunction(path="data.csv", format="CSV")
        result = tf.with_settings(format_csv_delimiter='|')
        assert result is tf

    def test_with_settings_multiple_calls(self):
        """Test multiple with_settings calls."""
        tf = FileTableFunction(path="data.csv", format="CSV")
        tf.with_settings(format_csv_delimiter='|')
        tf.with_settings(input_format_csv_skip_first_lines=1)

        assert tf.settings['format_csv_delimiter'] == '|'
        assert tf.settings['input_format_csv_skip_first_lines'] == 1


class TestReadOnlyTableFunctionsExtended:
    """Extended tests for read-only table functions."""

    def test_sqlite_write_error(self):
        """Test that SQLite table function rejects writes."""
        ds = DataStore("sqlite", database_path="/path/to/db.sqlite", table="users")
        ds_insert = ds.insert_into("id", "name").insert_values((1, "Alice"))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()
        assert "does not support writing" in str(exc_info.value)

    def test_iceberg_write_error(self):
        """Test that Iceberg table function rejects writes."""
        ds = DataStore("iceberg", url="s3://bucket/iceberg/table")
        ds_insert = ds.insert_into("id", "name").insert_values((1, "Alice"))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()
        assert "does not support writing" in str(exc_info.value)

    def test_deltalake_write_error(self):
        """Test that DeltaLake table function rejects writes."""
        ds = DataStore("delta", url="s3://bucket/delta/table")
        ds_insert = ds.insert_into("id", "name").insert_values((1, "Alice"))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()
        assert "does not support writing" in str(exc_info.value)

    def test_hudi_write_error(self):
        """Test that Hudi table function rejects writes."""
        ds = DataStore("hudi", url="s3://bucket/hudi/table")
        ds_insert = ds.insert_into("id", "name").insert_values((1, "Alice"))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()
        assert "does not support writing" in str(exc_info.value)

    def test_generaterandom_write_error(self):
        """Test that generateRandom table function rejects writes."""
        ds = DataStore("generaterandom", structure="id UInt32, name String")
        ds_insert = ds.insert_into("id", "name").insert_values((1, "Alice"))

        with pytest.raises(QueryError) as exc_info:
            ds_insert.to_sql()
        assert "does not support writing" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
