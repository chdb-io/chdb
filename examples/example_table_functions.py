"""
Examples demonstrating DataStore with various table functions.

This file shows how to use different data sources with chdb-ds.
"""

import sys
from pathlib import Path

# Add parent directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent))

from datastore import DataStore

# ====================
# File Operations
# ====================


def example_file_csv():
    """Read from local CSV file."""
    ds = DataStore("file", path="data/sales.csv", format="CSV")

    # Add CSV-specific settings
    ds = ds.with_format_settings(
        format_csv_delimiter=',', input_format_csv_skip_first_lines=1, input_format_csv_trim_whitespaces=1
    )

    # Query the file - simplified way
    records = ds.select("*").filter(ds.revenue > 1000).limit(10).to_dict()
    print(records)

    # Or get results as DataFrame
    # df = ds.select("*").filter(ds.revenue > 1000).limit(10).to_df()


def example_file_parquet():
    """Read from local Parquet file."""
    ds = DataStore("file", path="data/sales.parquet", format="Parquet")

    # Parquet-specific optimization settings
    ds = ds.with_format_settings(
        input_format_parquet_filter_push_down=1,
        input_format_parquet_bloom_filter_push_down=1,
        input_format_parquet_max_block_size=131072,
    )

    result = ds.select("product", "revenue").filter(ds.revenue > 5000).execute()
    print(f"SQL: {ds.to_sql()}")


def example_file_json():
    """Read from JSON file."""
    ds = DataStore("file", path="data/users.json", format="JSONEachRow")

    # JSON-specific settings
    ds = ds.with_format_settings(
        input_format_json_validate_types_from_metadata=1, input_format_json_ignore_unnecessary_fields=1
    )

    result = ds.select("name", "email").execute()


# ====================
# Cloud Storage
# ====================


def example_s3_public():
    """Read from public S3 bucket."""
    ds = DataStore(
        "s3",
        url="https://datasets-documentation.s3.eu-west-3.amazonaws.com/aapl_stock.csv",
        format="CSVWithNames",
        nosign=True,  # Public bucket
    )

    # Simplified way - directly get results as dict or DataFrame
    records = ds.select("*").limit(5).to_dict()
    print(records)

    # Or: df = ds.select("*").limit(5).to_df()


def example_s3_with_credentials():
    """Read from S3 with credentials."""
    ds = DataStore(
        "s3",
        path="s3://my-bucket/data/*.parquet",
        access_key_id="YOUR_ACCESS_KEY",
        secret_access_key="YOUR_SECRET_KEY",
        format="Parquet",
    )

    # Use glob pattern to read multiple files
    result = ds.select("*").filter(ds.date >= '2024-01-01').execute()


def example_azure_blob():
    """Read from Azure Blob Storage."""
    ds = DataStore(
        "azure",
        connection_string="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;",
        container="mycontainer",
        path="data/*.csv",
        format="CSV",
    )

    result = ds.select("*").execute()


def example_gcs():
    """Read from Google Cloud Storage."""
    ds = DataStore(
        "gcs",
        url="https://storage.googleapis.com/my-bucket/data.csv",
        hmac_key="YOUR_HMAC_KEY",
        hmac_secret="YOUR_HMAC_SECRET",
        format="CSV",
    )

    result = ds.select("*").execute()


# ====================
# Database Connections
# ====================


def example_mysql():
    """Query MySQL database."""
    ds = DataStore("mysql", host="localhost:3306", database="mydb", table="users", user="root", password="password")

    result = ds.select("id", "name", "email").filter(ds.active == 1).execute()


def example_postgresql():
    """Query PostgreSQL database."""
    ds = DataStore(
        "postgresql",
        host="localhost:5432",
        database="mydb",
        table="public.users",  # Can include schema
        user="postgres",
        password="password",
    )

    result = ds.select("*").filter(ds.age > 18).execute()


def example_mongodb():
    """Query MongoDB (read-only)."""
    ds = DataStore(
        "mongodb", host="localhost:27017", database="mydb", collection="users", user="admin", password="password"
    )

    # MongoDB table function is read-only
    result = ds.select("*").limit(100).execute()


def example_sqlite():
    """Query SQLite database (read-only)."""
    ds = DataStore("sqlite", database_path="/path/to/database.db", table="users")

    result = ds.select("*").filter(ds.age > 18).execute()


def example_redis():
    """Query Redis key-value store."""
    ds = DataStore(
        "redis",
        host="localhost:6379",
        key="key",  # Primary key column name
        structure="key String, value String, score UInt32",
        password="",
        db_index=0,
    )

    result = ds.select("*").execute()


# ====================
# Remote ClickHouse
# ====================


def example_remote_clickhouse():
    """Query remote ClickHouse server."""
    ds = DataStore(
        "clickhouse", host="clickhouse-server:9000", database="default", table="events", user="default", password=""
    )

    result = ds.select("*").filter(ds.timestamp >= '2024-01-01').execute()


def example_remote_secure():
    """Query remote ClickHouse with secure connection."""
    ds = DataStore(
        "remote",
        host="clickhouse-server:9440",
        database="default",
        table="events",
        user="default",
        password="",
        secure=True,  # Use remoteSecure
    )

    result = ds.select("*").execute()


# ====================
# Data Lake Formats
# ====================


def example_iceberg():
    """Query Iceberg table (read-only)."""
    ds = DataStore(
        "iceberg", url="s3://my-bucket/warehouse/my_table", access_key_id="YOUR_KEY", secret_access_key="YOUR_SECRET"
    )

    result = ds.select("*").filter(ds.year == 2024).execute()


def example_deltalake():
    """Query Delta Lake table (read-only)."""
    ds = DataStore(
        "delta", url="s3://my-bucket/delta/my_table", access_key_id="YOUR_KEY", secret_access_key="YOUR_SECRET"
    )

    result = ds.select("*").execute()


def example_hudi():
    """Query Hudi table (read-only)."""
    ds = DataStore(
        "hudi", url="s3://my-bucket/hudi/my_table", access_key_id="YOUR_KEY", secret_access_key="YOUR_SECRET"
    )

    result = ds.select("*").execute()


# ====================
# HTTP/URL Data
# ====================


def example_url():
    """Read from HTTP URL."""
    ds = DataStore("url", url="https://example.com/data.json", format="JSONEachRow")

    result = ds.select("*").limit(10).execute()


def example_url_with_headers():
    """Read from HTTP URL with custom headers."""
    ds = DataStore(
        "url",
        url="https://api.example.com/data",
        format="JSONEachRow",
        headers=["Authorization: Bearer YOUR_TOKEN", "Accept: application/json"],
    )

    result = ds.select("*").execute()


# ====================
# Data Generation
# ====================


def example_numbers():
    """Generate number sequence."""
    # Generate numbers 0-99
    ds = DataStore("numbers", count=100)
    result = ds.select("*").execute()

    # Generate numbers 10-19
    ds = DataStore("numbers", start=10, count=10)
    result = ds.select("*").execute()

    # Generate even numbers
    ds = DataStore("numbers", start=0, count=10, step=2)
    result = ds.select("*").execute()


def example_generate_random():
    """Generate random test data."""
    ds = DataStore(
        "generaterandom",
        structure="id UInt32, name String, value Float64, tags Array(String)",
        random_seed=42,
        max_string_length=20,
        max_array_length=5,
    )

    result = ds.select("*").limit(100).execute()


# ====================
# Multi-Source Queries
# ====================


def example_multi_source_join():
    """Join data from different sources."""
    # Local CSV file
    csv_data = DataStore("file", path="local_sales.csv", format="CSVWithNames")

    # Remote ClickHouse table
    ch_table = DataStore("clickhouse", host="localhost:9000", database="default", table="customer_info", user="default")

    # Join across data sources
    result = (
        ch_table.join(csv_data, left_on="customer_id", right_on="customer_id", how="inner")
        .select("customer_name", "product", "revenue")
        .filter(csv_data.purchase_date >= "2024-01-01")
        .execute()
    )

    print(f"SQL: {ch_table.to_sql()}")


def example_s3_and_mysql():
    """Query S3 data and join with MySQL."""
    # S3 data
    s3_data = DataStore(
        "s3", url="s3://my-bucket/events/*.parquet", access_key_id="KEY", secret_access_key="SECRET", format="Parquet"
    )

    # MySQL data
    mysql_data = DataStore("mysql", host="localhost:3306", database="mydb", table="users", user="root", password="pass")

    # Join and analyze
    result = (
        s3_data.join(mysql_data, left_on="user_id", right_on="id")
        .select("user_id", "event_type", "name", "email")
        .groupby("user_id", "name")
        .execute()
    )


# ====================
# Write Operations
# ====================


def example_write_to_s3():
    """Write query results to S3."""
    # Read from one source
    source = DataStore("file", path="input.csv", format="CSV")

    # Create S3 target
    target = DataStore(
        "s3", url="s3://my-bucket/output.parquet", access_key_id="KEY", secret_access_key="SECRET", format="Parquet"
    )

    # Note: Writing to table functions requires INSERT INTO TABLE FUNCTION syntax
    # This is handled automatically by the DataStore
    target.insert_into("col1", "col2").select_from(source.select("col1", "col2").filter(source.value > 100)).execute()


def example_write_to_file():
    """Write query results to local file."""
    # Read and transform data
    source = DataStore("file", path="input.csv", format="CSV")

    # Write to file
    target = DataStore("file", path="output.parquet", format="Parquet")

    target.insert_into("id", "name", "total").select_from(
        source.groupby("id", "name").select("id", "name", source.amount.sum().as_("total"))
    ).execute()


# ====================
# Advanced: Format Settings
# ====================


def example_csv_custom_settings():
    """CSV with custom delimiter and settings."""
    ds = DataStore("file", path="data.txt", format="CSV")

    ds = ds.with_format_settings(
        format_csv_delimiter='|',  # Pipe-delimited
        input_format_csv_skip_first_lines=2,  # Skip header rows
        input_format_csv_trim_whitespaces=1,  # Trim spaces
        input_format_csv_empty_as_default=1,  # Empty as default values
        format_csv_null_representation='NULL',  # NULL representation
    )

    result = ds.select("*").execute()
    print(f"SQL: {ds.to_sql()}")


def example_parquet_optimization():
    """Parquet with optimization settings."""
    ds = DataStore(
        "s3",
        url="s3://bucket/large_dataset/*.parquet",
        access_key_id="KEY",
        secret_access_key="SECRET",
        format="Parquet",
    )

    # Enable all optimization features
    ds = ds.with_format_settings(
        input_format_parquet_filter_push_down=1,
        input_format_parquet_bloom_filter_push_down=1,
        input_format_parquet_page_filter_push_down=1,
        input_format_parquet_use_offset_index=1,
        input_format_parquet_max_block_size=131072,
        input_format_parquet_enable_row_group_prefetch=1,
    )

    # These settings will be added to the query
    result = ds.select("*").filter(ds.date >= '2024-01-01').execute()


def example_json_flexible():
    """JSON with flexible type conversion."""
    ds = DataStore("file", path="data.json", format="JSONEachRow")

    # Allow flexible type reading
    ds = ds.with_format_settings(
        input_format_json_read_bools_as_numbers=1,
        input_format_json_read_numbers_as_strings=1,
        input_format_json_read_arrays_as_strings=1,
        input_format_json_ignore_unnecessary_fields=1,
        input_format_json_named_tuples_as_objects=1,
    )

    result = ds.select("*").execute()


if __name__ == "__main__":
    # Run examples
    from datastore.functions import Sum, Count, Avg

    print("=" * 60)
    print("File CSV Example")
    print("=" * 60)

    # Most examples require actual data sources
    # For demonstration, let's show the generated SQL

    ds = DataStore("file", path="data/sales.csv", format="CSV")
    ds = ds.with_format_settings(format_csv_delimiter=',', input_format_csv_skip_first_lines=1)
    print(ds.select("*").filter(ds.revenue > 1000).to_sql())
    print()

    print("=" * 60)
    print("S3 Example")
    print("=" * 60)
    ds = DataStore(
        "s3", url="s3://my-bucket/data.parquet", access_key_id="KEY", secret_access_key="SECRET", format="Parquet"
    )
    print(ds.select("product", "revenue").filter(ds.date >= '2024-01-01').to_sql())
    print()

    print("=" * 60)
    print("MySQL Example")
    print("=" * 60)
    ds = DataStore("mysql", host="localhost:3306", database="mydb", table="users", user="root", password="pass")
    print(ds.select("*").filter(ds.age > 18).to_sql())
    print()

    print("=" * 60)
    print("Multi-source Join Example")
    print("=" * 60)
    csv_data = DataStore("file", path="sales.csv", format="CSV")
    mysql_data = DataStore(
        "mysql", host="localhost:3306", database="mydb", table="customers", user="root", password="pass"
    )
    query = mysql_data.join(csv_data, left_on="id", right_on="customer_id").select("name", "product", "revenue")

    print(query.to_sql())
