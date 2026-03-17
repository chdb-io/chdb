# ClickHouse Table Functions for chdb

> Table functions let you query external data sources directly in SQL.
> Use them with `chdb.query()` or inside a `Session`.

## Table of Contents

- [File Sources](#file-sources)
- [Cloud Storage](#cloud-storage)
- [Databases](#databases)
- [Data Lakes](#data-lakes)
- [Utility Functions](#utility-functions)

---

## File Sources

### file()

Query local files. Format is auto-detected from extension or specified explicitly.

```sql
SELECT * FROM file('data.parquet', Parquet)
SELECT * FROM file('data.csv', CSVWithNames)
SELECT * FROM file('events.jsonl', JSONEachRow)
SELECT * FROM file('logs/*.parquet', Parquet)              -- glob pattern
SELECT * FROM file('data/2024-*/events.csv', CSVWithNames) -- nested glob
```

**Parameters:** `file(path [, format [, structure [, compression]]])`

Supported formats: `Parquet`, `CSVWithNames`, `CSV`, `TSVWithNames`, `JSONEachRow`, `JSON`, `Arrow`, `ORC`, `Avro`, `XMLWithNames`.

Supported compression: auto-detected from extension (`.gz`, `.zst`, `.bz2`, `.xz`, `.lz4`).

---

## Cloud Storage

### s3()

```sql
-- Public (no auth)
SELECT * FROM s3('s3://bucket/path.parquet', NOSIGN)

-- With credentials
SELECT * FROM s3('s3://bucket/path.parquet', 'ACCESS_KEY', 'SECRET_KEY', 'Parquet')

-- Glob pattern
SELECT * FROM s3('s3://bucket/logs/2024-*.parquet', 'KEY', 'SECRET', 'Parquet')
```

**Parameters:** `s3(url [, NOSIGN | access_key, secret_key] [, format [, structure [, compression]]])`

### gcs()

```sql
SELECT * FROM gcs('gs://bucket/data.parquet', NOSIGN)
SELECT * FROM gcs('gs://bucket/data.parquet', 'HMAC_KEY', 'HMAC_SECRET', 'Parquet')
```

**Parameters:** Same as `s3()`.

### azureBlobStorage()

```sql
SELECT * FROM azureBlobStorage(
    'DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...',
    'container', 'path/data.parquet', 'Parquet')
```

**Parameters:** `azureBlobStorage(connection_string, container, path [, format [, structure [, compression]]])`

### hdfs()

```sql
SELECT * FROM hdfs('hdfs://namenode:9000/warehouse/data.parquet', 'Parquet')
SELECT * FROM hdfs('hdfs://namenode:9000/logs/*.parquet', 'Parquet')
```

**Parameters:** `hdfs(uri [, format [, structure [, compression]]])`

---

## Databases

### mysql()

```sql
SELECT * FROM mysql('host:3306', 'database', 'table', 'user', 'password')

-- With WHERE pushdown
SELECT * FROM mysql('db:3306', 'shop', 'orders', 'root', 'pass')
WHERE status = 'shipped' AND amount > 100
```

**Parameters:** `mysql(host:port, database, table, user, password)`

**Note:** Port is part of the host string (e.g., `'db:3306'`), not a separate parameter.

### postgresql()

```sql
SELECT * FROM postgresql('host:5432', 'database', 'table', 'user', 'password')

SELECT * FROM postgresql('pg:5432', 'analytics', 'events', 'analyst', 'pass')
ORDER BY created_at DESC LIMIT 100
```

**Parameters:** `postgresql(host:port, database, table, user, password)`

### remote() / remoteSecure()

Query a remote ClickHouse server:

```sql
SELECT * FROM remote('host:9000', 'database', 'table', 'user', 'password')
SELECT * FROM remoteSecure('host:9440', 'database', 'table', 'user', 'password')
```

**Parameters:** `remote(host:port, database, table [, user [, password]])`

### mongodb()

```sql
SELECT * FROM mongodb('host:27017', 'database', 'collection', 'user', 'password')
```

**Parameters:** `mongodb(host:port, database, collection, user, password)`

### sqlite()

```sql
SELECT * FROM sqlite('/path/to/database.db', 'table_name')
```

**Parameters:** `sqlite(database_path, table)`

---

## Data Lakes

### iceberg()

```sql
SELECT * FROM iceberg('s3://bucket/iceberg/table', 'ACCESS_KEY', 'SECRET_KEY')
SELECT * FROM iceberg('s3://bucket/iceberg/table', NOSIGN)
```

**Parameters:** `iceberg(url [, NOSIGN | access_key, secret_key] [, format])`

### deltaLake()

```sql
SELECT * FROM deltaLake('s3://bucket/delta/table', 'ACCESS_KEY', 'SECRET_KEY')
SELECT * FROM deltaLake('s3://bucket/delta/table', NOSIGN)
```

**Parameters:** `deltaLake(url [, NOSIGN | access_key, secret_key])`

**Note:** Function name is `deltaLake` (camelCase), not `deltalake`.

### hudi()

```sql
SELECT * FROM hudi('s3://bucket/hudi/table', 'ACCESS_KEY', 'SECRET_KEY')
SELECT * FROM hudi('s3://bucket/hudi/table', NOSIGN)
```

**Parameters:** `hudi(url [, NOSIGN | access_key, secret_key])`

---

## Utility Functions

### numbers()

Generate a sequence of numbers (useful for testing and date generation):

```sql
SELECT * FROM numbers(100)              -- 0 to 99
SELECT * FROM numbers(10, 100)          -- 10 to 109
SELECT toDate('2025-01-01') + number AS date FROM numbers(365)  -- date range
```

**Parameters:** `numbers([offset,] count)`

### Python()

Use a Python dict or DataFrame as a SQL table:

```python
import chdb

data = {"name": ["Alice", "Bob"], "score": [95, 87]}
chdb.query("SELECT * FROM Python(data) ORDER BY score DESC")

import pandas as pd
df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
chdb.query("SELECT * FROM Python(df) WHERE value > 15")
```

**Note:** The Python variable must be in scope when the query executes.

### url()

Query data from an HTTP/HTTPS URL:

```sql
SELECT * FROM url('https://example.com/data.csv', CSVWithNames)
SELECT * FROM url('https://api.example.com/data.json', JSONEachRow)
```

**Parameters:** `url(url, format [, structure])`
