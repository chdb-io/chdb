# DataStore Factory Methods

This document describes the convenient factory methods for creating DataStore instances from different data sources.

## Overview

DataStore provides two ways to create instances:

1. **Generic constructor**: `DataStore(source_type, **params)`
2. **Static factory methods**: `DataStore.from_xxx(...)` (recommended for IDE auto-completion)

## Factory Methods

### File Operations

#### `DataStore.from_file()`
Create DataStore from local files with automatic format detection.

```python
from datastore import DataStore

# Auto-detect format from extension
ds = DataStore.from_file("data.parquet")

# Explicit format
ds = DataStore.from_file("data.csv", format="CSV")

# With structure
ds = DataStore.from_file("data.csv", 
                         format="CSV",
                         structure="id UInt32, name String")

# Glob patterns
ds = DataStore.from_file("logs/*.csv", format="CSV")
```

**Signature**:
```python
@classmethod
def from_file(cls, path: str, format: str = None, structure: str = None, 
              compression: str = None, **kwargs) -> 'DataStore'
```

### Cloud Storage

#### `DataStore.from_s3()`
Create DataStore from Amazon S3 with automatic format detection.

```python
# Public bucket with auto-detection
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)

# With credentials
ds = DataStore.from_s3(
    "s3://bucket/data/*.parquet",
    access_key_id="YOUR_KEY",
    secret_access_key="YOUR_SECRET"
)

# Explicit format
ds = DataStore.from_s3(
    "s3://bucket/data.csv",
    access_key_id="KEY",
    secret_access_key="SECRET",
    format="CSV"
)
```

**Signature**:
```python
@classmethod
def from_s3(cls, url: str, access_key_id: str = None, 
            secret_access_key: str = None, format: str = None, 
            nosign: bool = False, **kwargs) -> 'DataStore'
```

#### `DataStore.from_hdfs()`
Create DataStore from HDFS with automatic format detection.

```python
# Auto-detect format
ds = DataStore.from_hdfs("hdfs://namenode:9000/data/*.parquet")

# Explicit format
ds = DataStore.from_hdfs(
    "hdfs://namenode:9000/data/*.csv",
    format="CSV"
)
```

**Signature**:
```python
@classmethod
def from_hdfs(cls, uri: str, format: str = None, 
              structure: str = None, **kwargs) -> 'DataStore'
```

### Database Connections

#### `DataStore.from_mysql()`
Create DataStore from MySQL database.

```python
ds = DataStore.from_mysql(
    host="localhost:3306",
    database="mydb",
    table="users",
    user="root",
    password="password"
)

# Query with chaining
result = ds.select("*").filter(ds.age > 18).execute()
```

**Signature**:
```python
@classmethod
def from_mysql(cls, host: str, database: str, table: str, 
               user: str, password: str = "", **kwargs) -> 'DataStore'
```

#### `DataStore.from_postgresql()`
Create DataStore from PostgreSQL database.

```python
ds = DataStore.from_postgresql(
    host="localhost:5432",
    database="mydb",
    table="public.users",  # Can include schema
    user="postgres",
    password="password"
)

result = ds.select("*").execute()
```

**Signature**:
```python
@classmethod
def from_postgresql(cls, host: str, database: str, table: str,
                    user: str, password: str = "", **kwargs) -> 'DataStore'
```

#### `DataStore.from_clickhouse()`
Create DataStore from remote ClickHouse server.

```python
# Regular connection
ds = DataStore.from_clickhouse(
    host="localhost:9000",
    database="default",
    table="events"
)

# Secure connection
ds = DataStore.from_clickhouse(
    host="server:9440",
    database="default",
    table="events",
    user="default",
    password="",
    secure=True  # Uses remoteSecure()
)

result = ds.select("*").filter(ds.date >= '2024-01-01').execute()
```

**Signature**:
```python
@classmethod
def from_clickhouse(cls, host: str, database: str, table: str,
                    user: str = "default", password: str = "", 
                    secure: bool = False, **kwargs) -> 'DataStore'
```

#### `DataStore.from_mongodb()`
Create DataStore from MongoDB (read-only).

```python
ds = DataStore.from_mongodb(
    host="localhost:27017",
    database="mydb",
    collection="users",
    user="admin",
    password="password"
)

# MongoDB is read-only
result = ds.select("*").limit(100).execute()
```

**Signature**:
```python
@classmethod
def from_mongodb(cls, host: str, database: str, collection: str,
                 user: str, password: str = "", **kwargs) -> 'DataStore'
```

#### `DataStore.from_sqlite()`
Create DataStore from SQLite database (read-only).

```python
ds = DataStore.from_sqlite(
    database_path="/path/to/database.db",
    table="users"
)

result = ds.select("*").filter(ds.age > 18).execute()
```

**Signature**:
```python
@classmethod
def from_sqlite(cls, database_path: str, table: str, **kwargs) -> 'DataStore'
```

### HTTP/URL Data

#### `DataStore.from_url()`
Create DataStore from HTTP/HTTPS URL.

```python
# Basic URL
ds = DataStore.from_url(
    url="https://example.com/data.json",
    format="JSONEachRow"
)

# With custom headers
ds = DataStore.from_url(
    url="https://api.example.com/data",
    format="JSONEachRow",
    headers=[
        "Authorization: Bearer TOKEN",
        "Accept: application/json"
    ]
)
```

**Signature**:
```python
@classmethod
def from_url(cls, url: str, format: str, structure: str = None, 
             headers: List[str] = None, **kwargs) -> 'DataStore'
```

### Data Lake Formats

#### `DataStore.from_iceberg()`
Create DataStore from Apache Iceberg table (read-only).

```python
ds = DataStore.from_iceberg(
    url="s3://warehouse/my_table",
    access_key_id="YOUR_KEY",
    secret_access_key="YOUR_SECRET"
)

result = ds.select("*").filter(ds.year == 2024).execute()
```

**Signature**:
```python
@classmethod
def from_iceberg(cls, url: str, access_key_id: str = None, 
                 secret_access_key: str = None, **kwargs) -> 'DataStore'
```

#### `DataStore.from_delta()`
Create DataStore from Delta Lake table (read-only).

```python
ds = DataStore.from_delta(
    url="s3://bucket/delta_table",
    access_key_id="YOUR_KEY",
    secret_access_key="YOUR_SECRET"
)

result = ds.select("*").execute()
```

**Signature**:
```python
@classmethod
def from_delta(cls, url: str, access_key_id: str = None,
               secret_access_key: str = None, **kwargs) -> 'DataStore'
```

### Data Generation

#### `DataStore.from_numbers()`
Create DataStore that generates number sequences.

```python
# Generate 0 to 99
ds = DataStore.from_numbers(100)

# Generate 10 to 19
ds = DataStore.from_numbers(10, start=10)

# Generate even numbers 0, 2, 4, ..., 18
ds = DataStore.from_numbers(10, start=0, step=2)

result = ds.select("*").execute()
```

**Signature**:
```python
@classmethod
def from_numbers(cls, count: int, start: int = None, 
                 step: int = None, **kwargs) -> 'DataStore'
```

### From pandas DataFrame

#### `DataStore.from_df()`
Create DataStore from an existing pandas DataFrame.

```python
import pandas as pd
from datastore import DataStore

# Create a DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

# Wrap with DataStore
ds = DataStore.from_df(df, name='users')

# Use DataStore features
result = ds.filter(ds.age > 26).to_df()

# Execute SQL on DataFrame
result = ds.sql('age > 28 ORDER BY name').to_df()

# Mix SQL and pandas operations
ds['age_group'] = ds.age // 10 * 10
result = ds.sql('age_group >= 30').to_df()
```

**Signature**:
```python
@classmethod
def from_df(cls, df, name: str = None) -> 'DataStore'
```

**Parameters**:
- `df`: pandas DataFrame to wrap
- `name`: Optional name for the data source (appears in explain output)

#### `DataStore.from_dataframe()`
Alias for `from_df()` with identical functionality.

```python
# Both are equivalent
ds = DataStore.from_df(df, name='users')
ds = DataStore.from_dataframe(df, name='users')
```

**Signature**:
```python
@classmethod
def from_dataframe(cls, df, name: str = None) -> 'DataStore'
```

## Comparison: Factory Methods vs Generic Constructor

### Using Factory Methods (Recommended)
```python
# ✓ IDE auto-completion shows available parameters
# ✓ Type hints for parameters
# ✓ Clear and discoverable API

ds = DataStore.from_s3(
    url="s3://bucket/data.parquet",
    access_key_id="KEY",
    secret_access_key="SECRET"
)
```

### Using Generic Constructor
```python
# ✗ Need to remember parameter names
# ✗ No IDE auto-completion for source-specific params

ds = DataStore(
    "s3",
    url="s3://bucket/data.parquet",
    access_key_id="KEY",
    secret_access_key="SECRET"
)
```

Both methods produce the same result, but factory methods provide better developer experience.

## Format Auto-Detection

The following table functions support automatic format detection from file extensions:

- **file()**: Local files
- **url()**: HTTP(S) URLs
- **s3()**: Amazon S3
- **azure()**: Azure Blob Storage
- **gcs()**: Google Cloud Storage
- **hdfs()**: HDFS

### Examples

```python
# All these will auto-detect format:
ds = DataStore.from_file("data.parquet")      # → Parquet
ds = DataStore.from_file("data.csv")          # → CSV
ds = DataStore.from_file("data.json")         # → JSON
ds = DataStore.from_s3("s3://bucket/file.orc", nosign=True)  # → ORC
ds = DataStore.from_hdfs("hdfs://nn:9000/data.arrow")        # → Arrow
```

## Complete Example

```python
from datastore import DataStore

# Read from S3 with auto-detection
s3_data = DataStore.from_s3(
    "s3://my-bucket/events/*.parquet",
    access_key_id="KEY",
    secret_access_key="SECRET"
)

# Read from MySQL
mysql_data = DataStore.from_mysql(
    host="localhost:3306",
    database="mydb",
    table="users",
    user="root",
    password="pass"
)

# Join different sources
result = s3_data.join(
    mysql_data,
    left_on="user_id",
    right_on="id"
).select(
    "user_id",
    "event_type",
    "name",
    "email"
).filter(
    s3_data.event_date >= '2024-01-01'
).execute()

# Generate test data
numbers = DataStore.from_numbers(1000)
test_data = numbers.select(
    numbers.number.as_("id"),
    (numbers.number * 10).as_("value")
).execute()
```

## Benefits

1. **Better IDE Support**: Auto-completion shows available parameters
2. **Type Safety**: Parameter types are documented and checked
3. **Discoverability**: Users can explore available data sources through `DataStore.from_*` methods
4. **Cleaner Code**: More readable and self-documenting
5. **Format Auto-Detection**: No need to specify format for common file types

## Migration Guide

### Old Style
```python
ds = DataStore("file", path="data.csv", format="CSV")
ds = DataStore("s3", url="s3://bucket/data.parquet", format="Parquet", nosign=True)
ds = DataStore("mysql", host="localhost:3306", database="db", table="users", ...)
```

### New Style (Recommended)
```python
ds = DataStore.from_file("data.csv")  # Format auto-detected
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)  # Format auto-detected
ds = DataStore.from_mysql("localhost:3306", "db", "users", ...)
```

Both styles are supported and produce identical results.


