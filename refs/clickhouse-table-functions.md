# ClickHouse Table Functions Reference

This document provides a comprehensive reference for ClickHouse table functions suitable for Python function wrapper implementation. External configuration dependencies (XML files, etc.) are excluded.

## Table of Contents
- [File Operations](#file-operations)
- [Cloud Storage](#cloud-storage)
- [Database Connections](#database-connections)
- [Data Lake Formats](#data-lake-formats)
- [Data Generation](#data-generation)
- [Utility Functions](#utility-functions)
- [Cluster Functions](#cluster-functions)

---

## File Operations

### file

**Purpose**: Provides a table-like interface to SELECT from and INSERT into local files.

**Parameters**:
- `path` (String): File path relative to `user_files_path`. Supports glob patterns:
  - `*` - matches any characters except `/`
  - `{option1,option2,...}` - substitutes strings
  - `{N..M}` - range of numbers
- `format` (String): File format (CSV, TSV, Parquet, JSON, etc.)
- `structure` (String): Table structure as `'column1 Type1, column2 Type2, ...'`
- `compression` (String, optional): Compression method (gzip, zstd, etc.)

**Read Example**:
```sql
SELECT * FROM file('test.csv', 'CSV', 'column1 UInt32, column2 UInt32, column3 UInt32')
LIMIT 2;
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION file('test.csv', 'CSV', 'column1 UInt32, column2 UInt32, column3 UInt32')
VALUES (1, 2, 3), (3, 2, 1);

-- With partitioning
INSERT INTO TABLE FUNCTION file('test_{_partition_id}.tsv', 'TSV', 'column1 UInt32, column2 UInt32, column3 UInt32')
PARTITION BY column3
VALUES (1, 2, 3), (3, 2, 1), (1, 3, 2);
```

**Capabilities**:
- Read: ✓
- Write: ✓

**Virtual Columns**:
- `_path` - File path (LowCardinality(String))
- `_file` - Filename (LowCardinality(String))
- `_size` - File size in bytes (Nullable(UInt64))
- `_time` - Last modified time (Nullable(DateTime))

---

### url

**Purpose**: Creates a table from an HTTP(S) URL with a given format and structure; supports writing via `INSERT INTO TABLE FUNCTION`.

**Parameters**:
- `url` (String): HTTP(S) URL to the data.
- `format` (String): Input/output data format (e.g., `CSV`, `JSONEachRow`).
- `structure` (String, optional): `'col1 Type1, col2 Type2, ...'`.
- `headers` (optional): pass as `HEADERS('K1: V1','K2: V2')`.

**Read Example**:
```sql
SELECT *
FROM url('https://example.com/data.json', 'JSONEachRow', 'a UInt32, b String')
LIMIT 5;
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION url('https://api/ingest', 'JSONEachRow')
SELECT * FROM t
SETTINGS output_format_json_named_tuples_as_objects = 1
HEADERS('Authorization: Bearer <token>', 'Accept: application/json');
```

**Capabilities**:
- Read: ✓
- Write: ✓

### format

**Purpose**: Parses data from arguments according to a specified input format.

**Parameters**:
- `format` (String): Input format name.
- `structure` (String, optional): Table structure `'col1 Type1, col2 Type2, ...'`.
- `data` (String): The data to parse.

**Read Examples**:
```sql
-- Without structure (inferred)
SELECT * FROM format('JSONEachRow', $$
{"a": "Hello", "b": 111}
{"a": "World", "b": 123}
$$);

-- With explicit structure
SELECT * FROM format('JSONEachRow', 'a String, b UInt32', $$
{"a": "Hello", "b": 111}
{"a": "World", "b": 123}
$$);
```

**Capabilities**:
- Read: ✓
- Write: ✗

### s3

**Purpose**: Provides table-like interface to select/insert files in Amazon S3 and Google Cloud Storage.

**Parameter Combinations**:

**Combination 1**: Basic with credentials
- `url` (String): S3 URL (e.g., `https://bucket.s3.region.amazonaws.com/path/file.csv`)
- `access_key_id` (String): AWS access key
- `secret_access_key` (String): AWS secret key
- `format` (String): Data format
- `structure` (String, optional): Table structure
- `compression` (String, optional): Compression method

**Combination 2**: Public bucket
- `url` (String): S3 URL
- `NOSIGN` (Keyword): Skip authentication
- `format` (String): Data format
- `structure` (String, optional): Table structure

**Combination 3**: With session token
- `url` (String): S3 URL
- `access_key_id` (String): AWS access key
- `secret_access_key` (String): AWS secret key
- `session_token` (String): AWS session token
- `format` (String): Data format
- `structure` (String, optional): Table structure

**Combination 4**: With role ARN (ClickHouse Cloud)
- `url` (String): S3 URL
- `format` (String): Data format
- `extra_credentials(role_arn = 'arn:aws:iam::...')`: Role-based access

**Read Example**:
```sql
-- Public bucket
SELECT * FROM s3(
    'https://datasets-documentation.s3.eu-west-3.amazonaws.com/aapl_stock.csv',
    NOSIGN,
    'CSVWithNames'
) LIMIT 5;

-- With credentials
SELECT * FROM s3(
    'https://bucket.s3.amazonaws.com/data/*.csv',
    'ACCESS_KEY',
    'SECRET_KEY',
    'CSV',
    'id UInt32, value String'
);

-- With glob patterns
SELECT * FROM s3(
    'https://bucket.s3.amazonaws.com/data/file_{1..3}.csv',
    'ACCESS_KEY',
    'SECRET_KEY',
    'CSV'
);

-- Archive extraction
SELECT * FROM s3(
    'https://bucket.s3.amazonaws.com/archive.zip :: *.csv',
    'ACCESS_KEY',
    'SECRET_KEY',
    'CSV'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION s3(
    'https://bucket.s3.amazonaws.com/output.csv',
    'ACCESS_KEY',
    'SECRET_KEY',
    'CSV',
    'a UInt32, b String'
) VALUES (1, 'test');

-- Partitioned write
INSERT INTO TABLE FUNCTION s3(
    'http://bucket.amazonaws.com/my_bucket_{_partition_id}/file.csv',
    'CSV',
    'a UInt32, b UInt32, c UInt32'
) PARTITION BY a VALUES (1, 2, 3), (10, 11, 12);
```

**Capabilities**:
- Read: ✓
- Write: ✓

**Virtual Columns**: Same as `file` function

**Supported Archive Formats**: ZIP, TAR, 7Z (7Z only from local filesystem)

---

### azureBlobStorage

**Purpose**: Provides table-like interface to select/insert files in Azure Blob Storage.

**Parameters**: Similar to s3
- `connection_string` or `storage_account_url` (String): Azure connection
- `container` (String): Container name
- `path` (String): Blob path with glob support
- `account_name` (String, optional): Storage account
- `account_key` (String, optional): Access key
- `format` (String): Data format
- `structure` (String, optional): Table structure
- `compression` (String, optional): Compression

**Read Example**:
```sql
SELECT * FROM azureBlobStorage(
    'DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;',
    'container',
    'data/*.csv',
    'CSV'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION azureBlobStorage(...) VALUES (...);
```

**Capabilities**:
- Read: ✓
- Write: ✓

---

### gcs

**Purpose**: Table-like interface for Google Cloud Storage. Uses GCS XML API.

**Parameters**:
- `NOSIGN` (flag, optional): use anonymous access (no signing)
- `url` (String): GCS URL in format `https://storage.googleapis.com/<bucket>/<path>`
- `hmac_key` (String): GCS HMAC key
- `hmac_secret` (String): GCS HMAC secret
- `format` (String): Data format
- `structure` (String, optional): Table structure
- `compression` (String, optional): Compression

**Read Example**:
```sql
SELECT * FROM gcs(
    'https://storage.googleapis.com/bucket/data.csv',
    'HMAC_KEY',
    'HMAC_SECRET',
    'CSV'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION gcs(...) VALUES (...);
```

**Capabilities**:
- Read: ✓
- Write: ✓

---

### hdfs

**Purpose**: Creates table from files in HDFS.

**Parameters**:
- `uri` (String): HDFS URI (e.g., `hdfs://namenode:port/path`)
- `format` (String): Data format
- `structure` (String, optional): Table structure

**Read Example**:
```sql
SELECT * FROM hdfs('hdfs://namenode:9000/data/*.csv', 'CSV');
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION hdfs(...) VALUES (...);
```

**Capabilities**:
- Read: ✓
- Write: ✓

---

## Database Connections

### mysql

**Purpose**: Allows SELECT and INSERT queries on remote MySQL servers.

**Parameters**:
- `host:port` (String): MySQL server address
- `database` (String): Database name
- `table` (String): Table name
- `user` (String): Username
- `password` (String): Password
- `replace_query` (Bool, optional): Replace INSERT with REPLACE
- `on_duplicate_clause` (String, optional): ON DUPLICATE KEY clause

**Read Example**:
```sql
SELECT * FROM mysql(
    'mysql_host:3306',
    'database',
    'table',
    'user',
    'password'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION mysql(
    'mysql_host:3306',
    'database',
    'table',
    'user',
    'password'
) VALUES (1, 'value');
```

**Capabilities**:
- Read: ✓
- Write: ✓

**Settings**:
- `external_table_functions_use_nulls` - Controls NULL handling (default: 1)

---

### postgresql

**Purpose**: Allows SELECT and INSERT queries on remote PostgreSQL servers.

**Parameters**:
- `host:port` (String): PostgreSQL server address
- `database` (String): Database name
- `table` or `schema.table` (String): Table name with optional schema
- `user` (String): Username
- `password` (String): Password

**Read Example**:
```sql
SELECT * FROM postgresql(
    'postgres_host:5432',
    'database',
    'table',
    'user',
    'password'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION postgresql(...) VALUES (...);
```

**Capabilities**:
- Read: ✓
- Write: ✓

---

### mongodb

**Purpose**: Allows `SELECT` queries on MongoDB collections (read-only).

**Parameters**:
- `host:port` (String): MongoDB server address
- `database` (String): Database name
- `collection` (String): Collection name
- `user` (String): Username
- `password` (String): Password
- `structure` (String, optional): `'col1 Type1, col2 Type2, ...'`
- `options` (String, optional): Connection/options string
- `oid_columns` (String, optional): OID mapping settings

**Read Example**:
```sql
SELECT * FROM mongodb(
  'mongo_host:27017',
  'database',
  'collection',
  'user',
  'password'
);
```

**Capabilities**:
- Read: ✓
- Write: ✗

### redis

**Purpose**: Integrates with a Redis key-value store. The primary key column must be present in the structure; equality filters on the key are efficient.

**Parameters**:
- `host:port` (String): Redis server address
- `key` (String): **Name of the primary-key column** in `structure`
- `structure` (String): `'key Type, v1 Type, ...'`
- `password` (String, optional): Redis password
- `db_index` (UInt32, optional): Database index (default: 0)

**Read Example**:
```sql
SELECT * FROM redis(
  'localhost:6379',
  'key',
  'key String, v1 String, v2 UInt32'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION redis(
  'localhost:6379',
  'key',
  'key String, v1 String, v2 UInt32'
)
SELECT key, v1, v2 FROM some_source;
```

**Capabilities**:
- Read: ✓
- Write: ✓

### sqlite

**Purpose**: Performs queries on SQLite database.

**Parameters**:
- `database_path` (String): Path to SQLite database file
- `table` (String): Table name

**Read Example**:
```sql
SELECT * FROM sqlite('/path/to/database.db', 'table_name');
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

### jdbc

**Purpose**: Returns table connected via JDBC driver.

**Parameters**:
- `connection_string` (String): JDBC connection string
- `schema` (String, optional): Schema name
- `table` (String): Table name

**Read Example**:
```sql
SELECT * FROM jdbc(
    'jdbc:postgresql://host:5432/database',
    'schema',
    'table'
);
```

**Capabilities**:
- Read: ✓
- Write: Limited (driver-dependent)

**Note**: Requires JDBC bridge configuration

---

### odbc

**Purpose**: Returns table connected via ODBC driver.

**Parameters**:
- `connection_string` (String): ODBC connection string or DSN
- `table` (String): Table name

**Read Example**:
```sql
SELECT * FROM odbc('DSN=mydsn', 'table_name');
```

**Capabilities**:
- Read: ✓
- Write: Limited (driver-dependent)

**Note**: Requires ODBC bridge configuration

---

### remote / remoteSecure

**Purpose**: Accesses remote ClickHouse servers on-the-fly without creating Distributed table.

**Parameter Combinations**:

**Combination 1**: Single server
- `host:port` (String): Remote server address
- `database` (String): Database name
- `table` (String): Table name
- `user` (String, optional): Username (default: 'default')
- `password` (String, optional): Password

**Combination 2**: Multiple addresses
- `addresses_expr` (String): Comma-separated or pattern-based addresses
- `database.table` or `database, table` (String): Database and table
- `user` (String, optional): Username
- `password` (String, optional): Password

**Combination 3**: With sharding key
- `addresses` (String): Server addresses
- `database` (String): Database name
- `table` (String): Table name
- `user` (String): Username
- `password` (String): Password
- `sharding_key` (Expression): Expression for sharding

**Read Example**:
```sql
-- Single server
SELECT * FROM remote('127.0.0.1:9000', 'default', 'table', 'user', 'password');

-- Multiple servers with pattern
SELECT * FROM remote('server{01..03}:9000', 'db', 'table');

-- Multiple explicit servers
SELECT * FROM remote('server1:9000,server2:9000', 'db', 'table');

-- Secure connection
SELECT * FROM remoteSecure('server:9440', 'db', 'table');
```

**Write Example**:
```sql
INSERT INTO FUNCTION remote('127.0.0.1:9000', 'db', 'table')
VALUES (1, 'test');
```

**Capabilities**:
- Read: ✓
- Write: ✓

---

## Data Lake Formats

### iceberg

**Purpose**: Provides read-only table-like interface to Apache Iceberg tables.

**Parameters**:
- `url` (String): Path to Iceberg table (S3, Azure, HDFS, or local)
- `access_key_id` (String, optional): Access key for cloud storage
- `secret_access_key` (String, optional): Secret key for cloud storage
- `format` (String, optional): File format (default: Parquet)
- `structure` (String, optional): Table structure

**Read Example**:
```sql
-- S3 Iceberg table
SELECT * FROM iceberg(
    's3://bucket/path/to/iceberg/table',
    'ACCESS_KEY',
    'SECRET_KEY'
);

-- Local Iceberg table
SELECT * FROM iceberg('/path/to/iceberg/table');

-- With catalog (ClickHouse 24.12+)
SELECT * FROM iceberg('catalog_name.schema.table');
```

**Capabilities**:
- Read: ✓
- Write: ✗ (Experimental in latest versions)

**Virtual Columns**: Same as `file` function

**Supported Catalogs**: REST, Glue, Hive Metastore, Polaris

---

### deltaLake

**Purpose**: Provides read-only table-like interface to Delta Lake tables.

**Parameters**:
- `url` (String): Path to Delta Lake table (S3, Azure, or local)
- `access_key_id` (String, optional): Access key
- `secret_access_key` (String, optional): Secret key
- `format` (String, optional): File format
- `structure` (String, optional): Table structure

**Read Example**:
```sql
SELECT * FROM deltaLake(
    's3://bucket/path/to/delta/table',
    'ACCESS_KEY',
    'SECRET_KEY'
);
```

**Capabilities**:
- Read: ✓
- Write: ✗ (Experimental in latest versions)

---

### hudi

**Purpose**: Provides read-only table-like interface to Apache Hudi tables.

**Parameters**:
- `url` (String): Path to Hudi table in S3
- `access_key_id` (String, optional): AWS access key
- `secret_access_key` (String, optional): AWS secret key
- `format` (String, optional): File format
- `structure` (String, optional): Table structure

**Read Example**:
```sql
SELECT * FROM hudi(
    's3://bucket/path/to/hudi/table',
    'ACCESS_KEY',
    'SECRET_KEY'
);
```

**Capabilities**:
- Read: ✓
- Write: ✗

**Virtual Columns**:
- `_path` - File path
- `_file` - Filename
- `_size` - File size (Nullable(UInt64))
- `_time` - Last modified time (Nullable(DateTime))
- `_etag` - File ETag

---

## Data Generation

### numbers

**Purpose**: Returns a table with a single `number` column containing integers.

**Parameter Combinations**:

**Combination 1**: Range from 0  
- `N` (UInt64): Generate numbers from 0 to N-1

**Combination 2**: Custom range  
- `N` (UInt64): Start number  
- `M` (UInt64): Count of numbers

**Combination 3**: Custom range with step  
- `N` (UInt64): Start number  
- `M` (UInt64): Count of numbers  
- `S` (UInt64): Step

**Read Example**:
```sql
-- 0 to 9
SELECT * FROM numbers(10);

-- 10 to 19
SELECT * FROM numbers(10, 10);

-- Even numbers 0..18
SELECT * FROM numbers(0, 20, 2);
```

**Capabilities**:
- Read: ✓
- Write: ✗

### zeros

**Purpose**: Generates table with zero values. Fastest method to generate many rows.

**Parameters**:
- `N` (UInt64): Number of rows

**Read Example**:
```sql
SELECT * FROM zeros(1000000);
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

### generate_series / generateSeries

**Purpose**: Returns table with single 'generate_series' column containing integers with custom step.

**Parameters**:
- `start` (UInt64): Start value
- `stop` (UInt64): End value (inclusive)
- `step` (UInt64, optional): Step size (default: 1)

**Read Example**:
```sql
-- 10, 13, 16, 19
SELECT * FROM generate_series(10, 20, 3);

-- 1 to 100
SELECT * FROM generate_series(1, 100);
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

### generateRandom

**Purpose**: Generates random data with given schema for testing.

**Parameters**:
- `structure` (String): Table structure with column types
- `random_seed` (UInt64, optional): Random seed for reproducibility
- `max_string_length` (UInt64, optional): Max string length (default: 10)
- `max_array_length` (UInt64, optional): Max array length (default: 10)

**Read Example**:
```sql
SELECT * FROM generateRandom(
    'id UInt32, name String, value Float64',
    42,
    20,
    5
) LIMIT 10;
```

**Capabilities**:
- Read: ✓
- Write: ✗

**Note**: Not all types are supported

---

## Utility Functions

### input

**Purpose**: Used **only** inside `INSERT … SELECT` to transform incoming rows to a target structure. Not a standalone readable source.

**Parameters**:
- `structure` (String): `'col1 Type1, col2 Type2, ...'`

**Usage Example**:
```sql
-- Pipe client data and reshape into target schema
INSERT INTO target_table
SELECT lower(col1), toUInt32(col3)
FROM input('col1 String, col2 Date, col3 Int32')
FORMAT CSV;
```

**Capabilities**:
- Read: — *(usable only within `INSERT … SELECT`)*
- Write: ✗

### values

**Purpose**: Creates temporary storage filled with specified values.

**Parameters**:
- `structure` (String): Table structure
- `values` (Literal): Inline values

**Read Example**:
```sql
SELECT * FROM values('id UInt32, name String', (1, 'Alice'), (2, 'Bob'));
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

### null

**Purpose**: Creates temporary table with Null table engine.

**Parameters**:
- `structure` (String): Table structure

**Read Example**:
```sql
SELECT * FROM null('column1 UInt32, column2 String');
-- Returns no rows
```

**Capabilities**:
- Read: ✓ (returns empty)
- Write: ✓ (discards all data)

**Use Case**: Testing and demonstrations

---

### view

**Purpose**: Turns subquery into table.

**Parameters**:
- `subquery` (SELECT): SELECT query to wrap

**Read Example**:
```sql
SELECT * FROM view(SELECT number * 2 AS doubled FROM numbers(10));
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

### merge

**Purpose**: Creates temporary Merge table from underlying tables.

**Parameters**:
- `database` (String): Database name
- `table_regexp` (String): Regular expression for table names

**Read Example**:
```sql
-- Query all tables matching pattern
SELECT * FROM merge('default', '^log_.*');
```

**Capabilities**:
- Read: ✓
- Write: ✗

**Virtual Column**:
- `_table` - Source table name

---

### loop

**Purpose**: Returns query results in infinite loop.

**Parameters**:
- `database` (String): Database name
- `table` (String): Table name

**Alternative**:
- `subquery` (SELECT): Query to loop

**Read Example**:
```sql
SELECT * FROM loop('default', 'table') LIMIT 100;

SELECT * FROM loop(SELECT * FROM numbers(10)) LIMIT 50;
```

**Capabilities**:
- Read: ✓ (infinite)
- Write: ✗

---

### fuzzQuery

**Purpose**: Perturbs query string with random variations for fuzzing.

**Parameters**:
- `query` (String): Original SQL query
- `seed` (UInt64, optional): Random seed

**Read Example**:
```sql
SELECT * FROM fuzzQuery('SELECT * FROM table WHERE id = 1');
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

### fuzzJSON

**Purpose**: Perturbs JSON string with random variations.

**Parameters**:
- `json` (String): Original JSON
- `seed` (UInt64, optional): Random seed

**Read Example**:
```sql
SELECT * FROM fuzzJSON('{"key": "value"}');
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

## Cluster Functions

### s3Cluster

**Purpose**: Processes S3 files in parallel across cluster nodes.

**Parameters**: Same as `s3` plus:
- `cluster_name` (String): Cluster name (first parameter)
- Then all s3 parameters follow

**Read Example**:
```sql
SELECT * FROM s3Cluster(
    'my_cluster',
    'https://bucket.s3.amazonaws.com/data/*.csv',
    'ACCESS_KEY',
    'SECRET_KEY',
    'CSV'
);
```

**Write Example**:
```sql
INSERT INTO TABLE FUNCTION s3Cluster(...) VALUES (...);
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✓ (to initiator node)

---

### fileCluster

**Purpose**: Processes local files in parallel across cluster nodes.

**Parameters**:
- `cluster_name` (String): Cluster name
- `path` (String): File path with globs
- `format` (String): Data format
- `structure` (String, optional): Table structure

**Read Example**:
```sql
SELECT * FROM fileCluster('my_cluster', '/data/*.csv', 'CSV');
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

### urlCluster

**Purpose**: Processes URLs in parallel across cluster nodes.

**Parameters**:
- `cluster_name` (String): Cluster name
- `url` (String): URL pattern
- `format` (String): Data format
- `structure` (String, optional): Table structure

**Read Example**:
```sql
SELECT * FROM urlCluster('my_cluster', 'http://example.com/data/*.csv', 'CSV');
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

### hdfsCluster

**Purpose**: Processes HDFS files in parallel across cluster.

**Parameters**:
- `cluster_name` (String): Cluster name
- `uri` (String): HDFS URI with pattern
- `format` (String): Data format
- `structure` (String, optional): Table structure

**Read Example**:
```sql
SELECT * FROM hdfsCluster('my_cluster', 'hdfs://namenode:9000/data/*.csv', 'CSV');
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

### icebergCluster

**Purpose**: Processes Iceberg tables in parallel across cluster.

**Parameters**:
- `cluster_name` (String): Cluster name
- Then all iceberg parameters follow

**Read Example**:
```sql
SELECT * FROM icebergCluster(
    'my_cluster',
    's3://bucket/iceberg/table',
    'ACCESS_KEY',
    'SECRET_KEY'
);
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

### deltaLakeCluster

**Purpose**: Processes Delta Lake tables in parallel across cluster.

**Parameters**:
- `cluster_name` (String): Cluster name
- Then all deltaLake parameters follow

**Read Example**:
```sql
SELECT * FROM deltaLakeCluster('my_cluster', 's3://bucket/delta/table', 'KEY', 'SECRET');
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

### hudiCluster

**Purpose**: Processes Hudi tables in parallel across cluster.

**Parameters**:
- `cluster_name` (String): Cluster name
- Then all hudi parameters follow

**Read Example**:
```sql
SELECT * FROM hudiCluster('my_cluster', 's3://bucket/hudi/table', 'KEY', 'SECRET');
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

### azureBlobStorageCluster

**Purpose**: Processes Azure Blob Storage in parallel across cluster.

**Parameters**:
- `cluster_name` (String): Cluster name
- Then all azureBlobStorage parameters follow

**Read Example**:
```sql
SELECT * FROM azureBlobStorageCluster('my_cluster', 'connection_string', 'container', 'path', 'CSV');
```

**Capabilities**:
- Read: ✓ (distributed)
- Write: ✗

---

## Special Table Functions (Introspection)

### mergeTreeIndex

**Purpose**: Represents contents of index and marks files of MergeTree tables for introspection.

**Parameters**:
- `database` (String): Database name
- `table` (String): Table name

**Read Example**:
```sql
SELECT * FROM mergeTreeIndex('default', 'my_table');
```

**Capabilities**:
- Read: ✓ (metadata only)
- Write: ✗

---

### dictionary

**Purpose**: Displays dictionary data as ClickHouse table.

**Parameters**:
- `dictionary_name` (String): Name of the dictionary

**Read Example**:
```sql
SELECT * FROM dictionary('my_dictionary');
```

**Capabilities**:
- Read: ✓
- Write: ✗

---

## Implementation Notes for Python Wrapper

### General Patterns

**Parameter Handling**:
- Most table functions follow pattern: `function_name(param1, param2, ...)`
- Optional parameters can be omitted from the end
- String parameters need proper escaping
- Glob patterns supported in path/URL parameters for many functions

**Format Support**:
Common formats across table functions:
- CSV, CSVWithNames, CSVWithNamesAndTypes
- TSV, TSVWithNames, TSVWithNamesAndTypes
- JSON, JSONEachRow, JSONCompactEachRow
- Parquet
- Arrow
- ORC
- Avro
- Native

**Structure Specification**:
- Format: `'column1 Type1, column2 Type2, ...'`
- Can be omitted if schema inference is available
- Types: UInt8/16/32/64, Int8/16/32/64, Float32/64, String, Date, DateTime, Array(T), Nullable(T), etc.

### Virtual Columns Pattern

Many file-based table functions provide virtual columns:
- `_path` - Full file path
- `_file` - Filename only
- `_size` - File size in bytes
- `_time` - Last modification time
- `_etag` - ETag (for cloud storage)

Access with: `SELECT _file, _size, * FROM table_function(...)`

### Glob Patterns

Supported in file/URL/cloud storage functions:
- `*` - Matches any characters except `/`
- `?` - Matches single character
- `{a,b,c}` - Matches any of the alternatives
- `{N..M}` - Matches range of numbers (e.g., `file_{1..10}.csv`)
- `**` - Matches any path recursively (in some contexts)

Example: `s3('bucket/data/year=202{1,2,3}/month=*/day=*/file_*.parquet')`

### Cluster Functions Pattern

All cluster functions follow the pattern:
```
<function_name>Cluster('cluster_name', <original_function_parameters>)
```

Benefits:
- Distributed query processing across cluster nodes
- Parallel file reading/processing
- Automatic workload distribution

Note: Writes go through initiator node, which may be a bottleneck

### Error Handling Considerations

**Common Issues**:
1. **Authentication**: Ensure credentials are valid
2. **Network**: Check connectivity to remote services
3. **Permissions**: Verify file/object access rights
4. **Format Mismatch**: Ensure format matches actual data
5. **Schema Changes**: Structure parameter may need updates

**Settings to Consider**:
- `external_table_functions_use_nulls` - NULL handling for external sources
- `s3_max_redirects` - For S3 redirects
- `max_http_get_redirects` - For URL redirects
- `input_format_allow_errors_num` - Error tolerance during parsing
- `input_format_allow_errors_ratio` - Error ratio tolerance

### Security Considerations

**Credentials Management**:
- Avoid hardcoding credentials in queries
- Use named collections or config files when possible
- Consider using IAM roles (AWS) or managed identities (Azure)
- Use `remoteSecure` instead of `remote` for encrypted connections

**Path Restrictions**:
- File paths are relative to `user_files_path` setting
- Cannot access files outside configured directories
- Glob patterns are restricted to prevent directory traversal

### Performance Optimization

**Parallel Processing**:
- Use cluster functions for distributed processing
- Leverage glob patterns to distribute files across nodes
- Consider partitioning for large datasets

**Format Selection**:
- Parquet/ORC: Best for columnar analytics
- Native: Fastest for ClickHouse-to-ClickHouse transfers
- CSV/TSV: Human-readable, slower parsing
- Arrow: Good for interoperability

**Schema Inference**:
- Explicitly specify structure when possible (faster)
- Schema inference requires reading sample data
- More reliable with consistent data

### Python Wrapper Design Suggestions

**Function Builder Pattern**:
```python
class TableFunction:
    def __init__(self, name):
        self.name = name
        self.params = []
    
    def add_param(self, value, is_string=True):
        if is_string:
            self.params.append(f"'{value}'")
        else:
            self.params.append(str(value))
        return self
    
    def build(self):
        return f"{self.name}({', '.join(self.params)})"
```

**Type-Safe Builders**:
```python
class S3TableFunction:
    def __init__(self, url: str):
        self.url = url
        self.credentials = None
        self.format = None
        self.structure = None
        
    def with_credentials(self, key: str, secret: str):
        self.credentials = (key, secret)
        return self
        
    def with_format(self, format: str):
        self.format = format
        return self
        
    def with_structure(self, structure: str):
        self.structure = structure
        return self
        
    def build(self) -> str:
        params = [f"'{self.url}'"]
        if self.credentials:
            params.extend([f"'{self.credentials[0]}'", 
                          f"'{self.credentials[1]}'"])
        if self.format:
            params.append(f"'{self.format}'")
        if self.structure:
            params.append(f"'{self.structure}'")
        return f"s3({', '.join(params)})"
```

### Common Use Cases

**ETL Pipelines**:
```sql
-- Load from S3, transform, write to ClickHouse
INSERT INTO target_table
SELECT 
    transform_column(col1) as new_col1,
    col2
FROM s3('s3://bucket/source/*.parquet', 'Parquet')
WHERE condition;
```

**Data Lake Queries**:
```sql
-- Federated query across Iceberg and ClickHouse
SELECT 
    i.product_id,
    i.product_name,
    c.total_sales
FROM iceberg('s3://lake/products') i
JOIN clickhouse_table c ON i.product_id = c.product_id;
```

**Testing and Development**:
```sql
-- Generate test data
INSERT INTO test_table
SELECT 
    number as id,
    concat('user_', toString(number)) as name,
    rand() % 100 as age
FROM numbers(1000000);
```

**Cross-Database Analytics**:
```sql
-- Compare data across MySQL and PostgreSQL
SELECT 
    'mysql' as source,
    count(*) as total
FROM mysql('host:3306', 'db', 'table', 'user', 'pass')
UNION ALL
SELECT 
    'postgresql' as source,
    count(*) as total
FROM postgresql('host:5432', 'db', 'table', 'user', 'pass');
```

### Version Compatibility Notes

- **Iceberg Catalog Support**: ClickHouse 24.12+ (REST, Glue, Polaris)
- **Delta Lake Write**: Experimental in 25.x
- **Iceberg Cluster Functions**: 24.11+
- **Hudi Support**: 23.x+
- **Native Parquet Reader**: 25.8+ (experimental)
- **generate_series**: Available in recent versions
- **Arrow Flight**: Recent versions

### Resource Limits

**Configuration Settings**:
- `max_execution_time` - Query timeout
- `max_memory_usage` - Memory limit per query
- `max_threads` - Thread limit
- `max_insert_block_size` - Insert block size
- `table_function_remote_max_addresses` - Limit for remote addresses

### Testing Recommendations

**Unit Testing**:
- Test with small datasets first
- Verify schema inference vs explicit structure
- Test error conditions (missing files, auth failures)
- Validate data type conversions

**Integration Testing**:
- Test with real cloud storage/databases
- Verify cluster function distribution
- Test with various file formats
- Validate virtual columns

**Performance Testing**:
- Benchmark different formats
- Test parallel vs sequential processing
- Measure cluster function speedup
- Profile memory usage

---

## Excluded Functions (Require External Configuration)

The following table functions require XML configuration files or external setup and are excluded from this reference:

- **executable** - Requires script configuration in XML
- **clusterAllReplicas** - Requires cluster configuration in remote_servers
- **ytsaurus** - Requires YTsaurus cluster configuration

For these functions, refer to ClickHouse documentation for XML configuration details.

---

## Additional Resources

- Official Documentation: https://clickhouse.com/docs/sql-reference/table-functions
- Format Reference: https://clickhouse.com/docs/interfaces/formats
- Data Type Reference: https://clickhouse.com/docs/sql-reference/data-types
- Settings Reference: https://clickhouse.com/docs/operations/settings

---

## Summary Table

| Function | Read | Write | Distributed | Primary Use Case |
|----------|------|-------|-------------|------------------|
| file | ✓ | ✓ | ✗ | Local file I/O |
| url | ✓ | ✓ | ✗ | HTTP(S) data sources |
| s3 | ✓ | ✓ | ✗ | S3/GCS storage |
| s3Cluster | ✓ | ✓ | ✓ | Distributed S3 processing |
| azureBlobStorage | ✓ | ✓ | ✗ | Azure storage |
| gcs | ✓ | ✓ | ✗ | Google Cloud Storage |
| hdfs | ✓ | ✓ | ✗ | HDFS storage |
| mysql | ✓ | ✓ | ✗ | MySQL integration |
| postgresql | ✓ | ✓ | ✗ | PostgreSQL integration |
| mongodb | ✓ | ✗ | ✗ | MongoDB queries |
| redis | ✓ | ✓ | ✗ | Redis integration |
| sqlite | ✓ | ✗ | ✗ | SQLite queries |
| jdbc | ✓ | Limited | ✗ | JDBC connections |
| odbc | ✓ | Limited | ✗ | ODBC connections |
| remote | ✓ | ✓ | ✗ | Remote ClickHouse |
| iceberg | ✓ | ✗ | ✗ | Iceberg tables |
| deltaLake | ✓ | ✗ | ✗ | Delta Lake tables |
| hudi | ✓ | ✗ | ✗ | Hudi tables |
| numbers | ✓ | ✗ | ✗ | Number generation |
| zeros | ✓ | ✗ | ✗ | Zero generation |
| generate_series | ✓ | ✗ | ✗ | Series generation |
| generateRandom | ✓ | ✗ | ✗ | Random test data |
| format | ✓ | ✗ | ✗ | Parse inline data |
| input | —* | ✗ | ✗ | Data conversion |
| values | ✓ | ✗ | ✗ | Inline values |
| null | ✓ | ✓ | ✗ | Testing/discard |
| view | ✓ | ✗ | ✗ | Subquery wrapper |
| merge | ✓ | ✗ | ✗ | Multi-table query |
| loop | ✓ | ✗ | ✗ | Infinite loops |
| dictionary | ✓ | ✗ | ✗ | Dictionary access |
| mergeTreeIndex | ✓ | ✗ | ✗ | Index introspection |
| mergeTreeProjection | ✓ | ✗ | ✗ | Projection introspection |

*Notes:*
* `input`: Read (`—*`) means usable only within `INSERT … SELECT` (not a standalone source).
* `iceberg` / `deltaLake`: Writes via table functions are experimental in 25.8+ and may require specific settings; treat as read-only unless verified on your version.


---

*Document Version: 1.0*  
*Last Updated: 2025-10-05*  
*ClickHouse Versions: 23.x - 25.x*