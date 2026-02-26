"""
Examples demonstrating DataStore.uri() - the simplest way to create a DataStore.

This file shows how to use URIs to automatically create DataStore instances
without manually specifying source types and parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent))

from datastore import DataStore


# ====================
# Local File Examples
# ====================


def example_local_csv():
    """
    Read from local CSV file using simple file path.

    The format is automatically detected from the .csv extension.
    """
    print("\n=== Local CSV File ===")

    # Simple file path - format auto-detected
    ds = DataStore.uri("tests/dataset/users.csv")
    ds.connect()

    # Query the file
    result = ds.select("*").limit(5).execute()
    print(f"Found {result.row_count} rows")
    for row in result.rows[:3]:
        print(row)

    print(f"Generated SQL: {ds.select('*').limit(5).to_sql()}")


def example_local_file_with_file_scheme():
    """
    Read from local file using file:// URI scheme.
    """
    print("\n=== Local File with file:// Scheme ===")

    # Using file:// scheme
    import os

    csv_path = os.path.abspath("tests/dataset/products.csv")
    uri = f"file://{csv_path}"

    ds = DataStore.uri(uri)
    ds.connect()

    result = ds.select("*").limit(3).execute()
    print(f"Products: {result.row_count} rows")
    for row in result.rows:
        print(row)


def example_local_file_format_override():
    """
    Override the auto-detected format.
    """
    print("\n=== Local File with Format Override ===")

    # Override format if auto-detection doesn't work
    ds = DataStore.uri("tests/dataset/orders.csv", format="CSV")
    ds.connect()

    result = ds.select("*").limit(5).execute()
    print(f"Orders: {result.row_count} rows")


# ====================
# S3 Examples
# ====================


def example_s3_anonymous():
    """
    Read from S3 with anonymous access (no credentials).

    Note: This is a demo. The actual bucket/file may not exist.
    """
    print("\n=== S3 with Anonymous Access ===")

    # Simple S3 URI with nosign=true for anonymous access
    uri = "s3://mybucket/data/sales.parquet?nosign=true"

    ds = DataStore.uri(uri)
    print(f"Created DataStore for: {uri}")
    print(f"Source type: {ds.source_type}")
    print(f"SQL: {ds.select('*').limit(10).to_sql()}")


def example_s3_with_credentials():
    """
    Read from S3 with credentials in URI.

    Note: Never hardcode real credentials in code!
    Use environment variables or configuration files instead.
    """
    print("\n=== S3 with Credentials ===")

    # S3 URI with credentials (demo only - don't use real credentials!)
    uri = "s3://mybucket/data/events.parquet?access_key_id=YOUR_KEY&secret_access_key=YOUR_SECRET&region=us-west-2"

    ds = DataStore.uri(uri)
    print(f"Created DataStore for S3 with credentials")
    print(f"Source type: {ds.source_type}")


def example_s3_credentials_as_params():
    """
    Read from S3 with credentials passed as parameters.

    This is cleaner than putting credentials in the URI.
    """
    print("\n=== S3 with Credentials as Parameters ===")

    # Pass credentials as separate parameters (cleaner approach)
    uri = "s3://mybucket/data/events.parquet"

    ds = DataStore.uri(uri, access_key_id="YOUR_ACCESS_KEY", secret_access_key="YOUR_SECRET_KEY", region="us-west-2")

    print(f"Created DataStore for S3")
    print(f"Format auto-detected as: Parquet")


# ====================
# Database Examples
# ====================


def example_mysql():
    """
    Connect to MySQL database using URI.

    Format: mysql://user:password@host:port/database/table
    """
    print("\n=== MySQL Database ===")

    # MySQL connection string
    uri = "mysql://root:password@localhost:3306/mydb/users"

    ds = DataStore.uri(uri)
    print(f"Created MySQL DataStore")
    print(f"Table: {ds.table_name}")

    # Example query (won't execute without actual MySQL server)
    sql = ds.select("name", "email").filter(ds.age > 18).limit(10).to_sql()
    print(f"SQL: {sql}")


def example_postgresql():
    """
    Connect to PostgreSQL database using URI.

    Format: postgresql://user:password@host:port/database/table
    """
    print("\n=== PostgreSQL Database ===")

    # PostgreSQL connection string
    uri = "postgresql://postgres:pass@localhost:5432/mydb/products"

    ds = DataStore.uri(uri)
    print(f"Created PostgreSQL DataStore")
    print(f"Table: {ds.table_name}")

    # Example query
    sql = ds.select("product_name", "price").filter(ds.price > 100).to_sql()
    print(f"SQL: {sql}")


def example_mongodb():
    """
    Connect to MongoDB using URI.

    Format: mongodb://user:password@host:port/database.collection
    """
    print("\n=== MongoDB ===")

    # MongoDB connection string
    uri = "mongodb://admin:pass@localhost:27017/mydb.users"

    ds = DataStore.uri(uri)
    print(f"Created MongoDB DataStore")
    print(f"Source type: {ds.source_type}")


def example_sqlite():
    """
    Connect to SQLite database using URI.

    Format: sqlite:///path/to/database.db?table=tablename
    """
    print("\n=== SQLite Database ===")

    # SQLite connection string
    uri = "sqlite:///path/to/mydata.db?table=users"

    ds = DataStore.uri(uri)
    print(f"Created SQLite DataStore")
    print(f"Table: {ds.table_name}")


# ====================
# Cloud Storage Examples
# ====================


def example_gcs():
    """
    Read from Google Cloud Storage.

    Format: gs://bucket/path
    """
    print("\n=== Google Cloud Storage ===")

    # GCS URI
    uri = "gs://mybucket/data/events.parquet"

    ds = DataStore.uri(uri)
    print(f"Created GCS DataStore")
    print(f"Format: Parquet (auto-detected)")


def example_gcs_with_credentials():
    """
    Read from GCS with HMAC credentials.
    """
    print("\n=== GCS with HMAC Credentials ===")

    uri = "gs://mybucket/data/file.csv?hmac_key=YOUR_KEY&hmac_secret=YOUR_SECRET"

    ds = DataStore.uri(uri)
    print(f"Created GCS DataStore with credentials")


def example_azure():
    """
    Read from Azure Blob Storage.

    Format: az://container/blob
    """
    print("\n=== Azure Blob Storage ===")

    # Azure URI with credentials
    uri = "az://mycontainer/data/events.parquet?account_name=mystorageaccount&account_key=YOUR_KEY"

    ds = DataStore.uri(uri)
    print(f"Created Azure DataStore")


def example_hdfs():
    """
    Read from HDFS.

    Format: hdfs://namenode:port/path
    """
    print("\n=== HDFS ===")

    # HDFS URI
    uri = "hdfs://namenode:9000/data/events.parquet"

    ds = DataStore.uri(uri)
    print(f"Created HDFS DataStore")


# ====================
# HTTP/HTTPS Examples
# ====================


def example_http():
    """
    Read from HTTP/HTTPS URL.
    """
    print("\n=== HTTP/HTTPS URL ===")

    # HTTP URL
    uri = "https://raw.githubusercontent.com/example/data/master/data.csv"

    ds = DataStore.uri(uri)
    print(f"Created URL DataStore")
    print(f"Format: CSV (auto-detected from URL)")


# ====================
# Big Data Format Examples
# ====================


def example_iceberg():
    """
    Read from Apache Iceberg table.

    Format: iceberg://catalog/namespace/table
    """
    print("\n=== Apache Iceberg ===")

    uri = "iceberg://my_catalog/my_namespace/my_table"

    ds = DataStore.uri(uri)
    print(f"Created Iceberg DataStore")


def example_deltalake():
    """
    Read from Delta Lake table.

    Format: deltalake:///path/to/table or delta:///path/to/table
    """
    print("\n=== Delta Lake ===")

    # Delta Lake URI
    uri = "deltalake:///data/delta/events"

    ds = DataStore.uri(uri)
    print(f"Created Delta Lake DataStore")

    # Alternative: use 'delta://' alias
    uri2 = "delta:///data/delta/events"
    ds2 = DataStore.uri(uri2)
    print(f"Created Delta Lake DataStore (using delta:// alias)")


def example_hudi():
    """
    Read from Apache Hudi table.

    Format: hudi:///path/to/table
    """
    print("\n=== Apache Hudi ===")

    uri = "hudi:///data/hudi/events"

    ds = DataStore.uri(uri)
    print(f"Created Hudi DataStore")


# ====================
# Complex Query Examples
# ====================


def example_complex_query_with_uri():
    """
    Build complex queries using DataStore created from URI.
    """
    print("\n=== Complex Query with URI ===")

    # Create DataStore from local file
    ds = DataStore.uri("tests/dataset/orders.csv")
    ds.connect()

    # Build complex query with filtering, ordering, and limiting
    result = ds.select("*").limit(10).offset(5).execute()

    print(f"Query result: {result.row_count} rows")

    # Show generated SQL
    query = ds.select("*").limit(10).offset(5)
    print(f"Generated SQL:\n{query.to_sql()}")


def example_joins_with_uri():
    """
    Example showing joins between DataStores created from URIs.
    """
    print("\n=== Joins with URI DataStores ===")

    # Create two DataStores from local files
    users_ds = DataStore.uri("tests/dataset/users.csv")
    orders_ds = DataStore.uri("tests/dataset/orders.csv")

    users_ds.connect()

    # Note: This demonstrates the API; actual join execution depends on data
    print(f"Created DataStores for users and orders")
    print(f"Users SQL: {users_ds.select('*').limit(5).to_sql()}")
    print(f"Orders SQL: {orders_ds.select('*').limit(5).to_sql()}")


# ====================
# Comparison: Traditional vs URI
# ====================


def example_comparison():
    """
    Compare traditional DataStore creation vs URI-based creation.
    """
    print("\n=== Traditional vs URI Comparison ===")

    # Traditional way
    print("Traditional way:")
    ds_old = DataStore("file", path="tests/dataset/users.csv", format="CSV")
    ds_old.connect()
    result_old = ds_old.select("*").limit(3).execute()
    print(f"Rows: {result_old.row_count}")

    # URI way (simpler!)
    print("\nURI way:")
    ds_new = DataStore.uri("tests/dataset/users.csv")
    ds_new.connect()
    result_new = ds_new.select("*").limit(3).execute()
    print(f"Rows: {result_new.row_count}")

    print("\nBoth methods produce the same result!")


# ====================
# Main
# ====================


def main():
    """Run all examples."""
    print("=" * 70)
    print("DataStore.uri() Examples")
    print("=" * 70)

    # File examples
    try:
        example_local_csv()
    except Exception as e:
        print(f"Skipped: {e}")

    try:
        example_local_file_with_file_scheme()
    except Exception as e:
        print(f"Skipped: {e}")

    try:
        example_local_file_format_override()
    except Exception as e:
        print(f"Skipped: {e}")

    # S3 examples (demo only, won't execute without S3 access)
    example_s3_anonymous()
    example_s3_with_credentials()
    example_s3_credentials_as_params()

    # Database examples (demo only)
    example_mysql()
    example_postgresql()
    example_mongodb()
    example_sqlite()

    # Cloud storage examples (demo only)
    example_gcs()
    example_gcs_with_credentials()
    example_azure()
    example_hdfs()

    # HTTP examples
    example_http()

    # Big data format examples (demo only)
    example_iceberg()
    example_deltalake()
    example_hudi()

    # Complex queries
    try:
        example_complex_query_with_uri()
    except Exception as e:
        print(f"Skipped: {e}")

    try:
        example_joins_with_uri()
    except Exception as e:
        print(f"Skipped: {e}")

    # Comparison
    try:
        example_comparison()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
