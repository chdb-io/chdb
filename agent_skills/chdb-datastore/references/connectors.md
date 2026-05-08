# DataStore Connectors — All Data Sources

> Quick reference for connecting DataStore to any data source.
> After connecting, all sources share the same pandas API.

## Table of Contents

- [Local Files](#local-files)
- [Cloud Storage](#cloud-storage)
- [Databases](#databases)
- [Data Lakes](#data-lakes)
- [URI Shorthand](#uri-shorthand)
- [In-Memory Data](#in-memory-data)

---

## Local Files

```python
DataStore.from_file(path, format=None, structure=None, compression=None, **kwargs)
```

Format is auto-detected by extension: `.parquet`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.arrow`, `.orc`, `.avro`, `.xml`.

```python
from datastore import DataStore

ds = DataStore.from_file("sales.parquet")
ds = DataStore.from_file("data.csv")
ds = DataStore.from_file("events.jsonl")
ds = DataStore.from_file("logs/*.csv")              # glob pattern
ds = DataStore.from_file("data/2024-*/events.parquet")  # nested glob
ds = DataStore.from_file("data.csv.gz")             # compressed, auto-detected
ds = DataStore.from_file("data.tsv", format="TabSeparatedWithNames")  # explicit format
```

**Notes:**
- Glob patterns (`*`, `**`) work for querying multiple files at once
- Compression (`.gz`, `.zst`, `.bz2`, `.xz`, `.lz4`) is auto-detected from extension
- Use `structure` parameter to specify column types: `structure="id UInt64, name String"`

---

## Cloud Storage

### S3

```python
DataStore.from_s3(url, access_key_id=None, secret_access_key=None, format=None, nosign=False, **kwargs)
```

```python
# Public bucket (no auth)
ds = DataStore.from_s3("s3://public-data/dataset.parquet", nosign=True)

# Private bucket
ds = DataStore.from_s3("s3://my-bucket/data.parquet",
    access_key_id="AKIA...", secret_access_key="secret...")

# Glob pattern
ds = DataStore.from_s3("s3://bucket/logs/2024-*.parquet", nosign=True)
```

### GCS (Google Cloud Storage)

```python
DataStore.from_gcs(url, hmac_key=None, hmac_secret=None, format=None, nosign=False, **kwargs)
```

```python
ds = DataStore.from_gcs("gs://my-bucket/data.parquet", nosign=True)
ds = DataStore.from_gcs("gs://private/data.parquet", hmac_key="KEY", hmac_secret="SECRET")
```

### Azure Blob Storage

```python
DataStore.from_azure(connection_string, container, path="", format=None, **kwargs)
```

```python
ds = DataStore.from_azure(
    connection_string="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...",
    container="data", path="analytics/events.parquet")
```

### HDFS

```python
DataStore.from_hdfs(uri, format=None, structure=None, **kwargs)
```

```python
ds = DataStore.from_hdfs("hdfs://namenode:9000/warehouse/events/*.parquet")
```

### HTTP/HTTPS URL

```python
DataStore.from_url(url, format=None, structure=None, headers=None, **kwargs)
```

```python
ds = DataStore.from_url("https://example.com/data.csv")
```

---

## Databases

### MySQL

```python
DataStore.from_mysql(host, database=None, table=None, user=None, password="", port=None, **kwargs)
```

```python
ds = DataStore.from_mysql(
    host="db.example.com:3306", database="shop",
    table="orders", user="root", password="pass")
```

**Note:** Port must be included in `host` string (e.g., `"db:3306"`) or passed via `port` parameter.

### PostgreSQL

```python
DataStore.from_postgresql(host, database=None, table=None, user=None, password="", port=None, **kwargs)
```

```python
ds = DataStore.from_postgresql(
    host="pg:5432", database="analytics",
    table="events", user="user", password="pass")
```

### ClickHouse (Remote)

```python
DataStore.from_clickhouse(host, database=None, table=None, user="default", password="", secure=False, port=None, **kwargs)
```

```python
ds = DataStore.from_clickhouse(host="ch:9000", database="logs", table="access_log")
ds = DataStore.from_clickhouse(host="ch:9440", database="logs", table="hits",
    user="reader", password="pass", secure=True)
```

### MongoDB

```python
DataStore.from_mongodb(host, database, collection, user, password="", **kwargs)
```

```python
ds = DataStore.from_mongodb(
    host="mongo:27017", database="app",
    collection="users", user="user", password="pass")
```

### SQLite

```python
DataStore.from_sqlite(database_path, table, **kwargs)
```

```python
ds = DataStore.from_sqlite("/data/local.db", "users")
```

### Redis

```python
DataStore.from_redis(host, key, structure, password=None, db_index=0, **kwargs)
```

```python
ds = DataStore.from_redis("localhost:6379", key="mydata",
    structure="id UInt64, name String, value Float64")
```

---

## Data Lakes

### Apache Iceberg

```python
DataStore.from_iceberg(url, access_key_id=None, secret_access_key=None, **kwargs)
```

```python
ds = DataStore.from_iceberg("s3://warehouse/iceberg/events",
    access_key_id="KEY", secret_access_key="SECRET")
```

### Delta Lake

```python
DataStore.from_delta(url, access_key_id=None, secret_access_key=None, **kwargs)
```

```python
ds = DataStore.from_delta("s3://warehouse/delta/transactions",
    access_key_id="KEY", secret_access_key="SECRET")
```

### Apache Hudi

```python
DataStore.from_hudi(url, access_key_id=None, secret_access_key=None, **kwargs)
```

```python
ds = DataStore.from_hudi("s3://warehouse/hudi/logs",
    access_key_id="KEY", secret_access_key="SECRET")
```

---

## URI Shorthand

```python
DataStore.uri(uri_string, **kwargs)
```

Universal one-liner that auto-detects source type from the URI scheme:

| Scheme | Example |
|--------|---------|
| _(path)_ | `sales.parquet`, `/data/file.csv` |
| `file` | `file:///data/file.csv` |
| `s3`, `s3a`, `s3n` | `s3://bucket/key?nosign=true` |
| `gs`, `gcs` | `gs://bucket/path` |
| `az`, `azure`, `wasb` | `az://container/blob?account_name=X&account_key=Y` |
| `hdfs` | `hdfs://namenode:9000/path` |
| `http`, `https` | `https://example.com/data.json` |
| `mysql` | `mysql://user:pass@host:port/db/table` |
| `postgresql`, `postgres` | `postgresql://user:pass@host:port/db/table` |
| `clickhouse` | `clickhouse://host:port/db/table?user=X&password=Y` |
| `mongodb`, `mongo` | `mongodb://user:pass@host:port/db.collection` |
| `sqlite` | `sqlite:///path/to/db.db?table=name` |
| `redis` | `redis://host:port/db?key=mykey&password=pass` |
| `iceberg` | `iceberg://catalog/namespace/table` |
| `deltalake`, `delta` | `deltalake:///path/to/table` |
| `hudi` | `hudi:///path/to/table` |

```python
from datastore import DataStore

ds = DataStore.uri("s3://public-data/dataset.parquet?nosign=true")
ds = DataStore.uri("mysql://root:pass@localhost:3306/shop/orders")
ds = DataStore.uri("postgresql://analyst:pass@pg:5432/analytics/events")
ds = DataStore.uri("clickhouse://ch:9440/analytics/hits?user=reader&password=pass")
ds = DataStore.uri("mongodb://user:pass@mongo:27017/logs.app_events")
ds = DataStore.uri("sqlite:///data/local.db?table=users")
ds = DataStore.uri("deltalake:///data/delta/events")
```

---

## In-Memory Data

### From dict

```python
ds = DataStore({"name": ["Alice", "Bob"], "age": [25, 30]})
```

### From pandas DataFrame

```python
ds = DataStore(df)
ds = DataStore.from_df(df, name="my_data")
```

### Generated sequences

```python
ds = DataStore.from_numbers(100)                   # 0..99
ds = DataStore.from_numbers(10, start=5, step=2)   # 5, 7, 9, ...
```

### Random data (for testing)

```python
ds = DataStore.from_random(
    structure="id UInt64, name String, value Float64",
    random_seed=42, max_string_length=10)
```
