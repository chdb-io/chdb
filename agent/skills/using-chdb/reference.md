# chdb API Reference

## chdb.query()

```python
chdb.query(sql, output_format="CSV", path="", udf_path="", params=None)
```

| Param | Type | Description |
|-------|------|-------------|
| `sql` | str | ClickHouse SQL query |
| `output_format` | str | CSV, JSON, JSONEachRow, Arrow, Parquet, DataFrame, ArrowTable, Pretty, Debug |
| `path` | str | Database path (empty = in-memory) |
| `params` | dict | Named parameters `{name: value}`, referenced in SQL as `{name:Type}` |

Returns result with `.show()`, `.bytes()`, `.data()`, `.rows_read`, `.elapsed`, `.bytes_read`.

---

## DataStore

```python
from datastore import DataStore
# or: from chdb.datastore import DataStore
```

### Constructors

```python
DataStore(source=None, table=None, database=":memory:", connection=None, **kwargs)
```

| Source type | Usage |
|-------------|-------|
| dict | `DataStore({'col1': [1,2], 'col2': ['a','b']})` |
| pd.DataFrame | `DataStore(df)` |
| str (source type) | `DataStore("file", path="data.parquet")` |
| str (source type) | `DataStore("mysql", host="host:3306", database="db", table="t", user="u", password="p")` |

### Factory Methods — Files & Cloud Storage

| Method | Signature |
|--------|-----------|
| `from_file` | `(path, format=None, structure=None, compression=None, **kwargs)` |
| `from_s3` | `(url, access_key_id=None, secret_access_key=None, format=None, nosign=False, **kwargs)` |
| `from_gcs` | `(url, hmac_key=None, hmac_secret=None, format=None, nosign=False, **kwargs)` |
| `from_azure` | `(connection_string, container, path="", format=None, **kwargs)` |
| `from_hdfs` | `(uri, format=None, structure=None, **kwargs)` |
| `from_url` | `(url, format=None, structure=None, headers=None, **kwargs)` |

### Factory Methods — Databases

| Method | Signature |
|--------|-----------|
| `from_mysql` | `(host, database=None, table=None, user=None, password="", port=None, **kwargs)` |
| `from_postgresql` | `(host, database=None, table=None, user=None, password="", port=None, **kwargs)` |
| `from_clickhouse` | `(host, database=None, table=None, user="default", password="", secure=False, port=None, **kwargs)` |
| `from_mongodb` | `(host, database, collection, user, password="", **kwargs)` |
| `from_sqlite` | `(database_path, table, **kwargs)` |
| `from_redis` | `(host, key, structure, password=None, db_index=0, **kwargs)` |

### Factory Methods — Data Lakes

| Method | Signature |
|--------|-----------|
| `from_iceberg` | `(url, access_key_id=None, secret_access_key=None, **kwargs)` |
| `from_delta` | `(url, access_key_id=None, secret_access_key=None, **kwargs)` |
| `from_hudi` | `(url, access_key_id=None, secret_access_key=None, **kwargs)` |

### Factory Methods — Other

| Method | Signature |
|--------|-----------|
| `from_df` / `from_dataframe` | `(df, name=None)` |
| `from_numbers` | `(count, start=None, step=None, **kwargs)` |
| `from_random` | `(structure, random_seed=None, max_string_length=None, max_array_length=None, **kwargs)` |
| `run_sql` | `(query)` — run SQL and return DataStore |
| `uri` | `(uri_string, **kwargs)` — universal URI-based factory |

### URI Schemes

| Scheme | Example |
|--------|---------|
| (path) | `/data/file.csv`, `data.parquet` |
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

### Selection & Filtering

| Method | Description |
|--------|-------------|
| `ds['col']` | Single column → LazySeries |
| `ds[['c1', 'c2']]` | Multiple columns → DataStore |
| `ds[condition]` | Boolean filter → DataStore |
| `.select(*fields)` | SQL-style SELECT |
| `.filter(condition)` / `.where(condition)` | SQL-style WHERE |

### Sorting & Limiting

| Method | Description |
|--------|-------------|
| `.sort_values(by, ascending=True)` | Pandas-style sort |
| `.sort(*columns, ascending=True)` / `.orderby(...)` | SQL-style ORDER BY |
| `.limit(n)` | LIMIT |
| `.offset(n)` | OFFSET |
| `.head(n=5)` / `.tail(n=5)` | First/last N rows |

### GroupBy & Aggregation

| Method | Description |
|--------|-------------|
| `.groupby(*columns)` | → LazyGroupBy |
| `.agg(func=None, **kwargs)` | `'sum'`, `'mean'`, `'count'`, `'min'`, `'max'`, `'std'`, `'var'` |
| `.having(condition)` | HAVING clause |

### Joins

| Method | Description |
|--------|-------------|
| `.join(other, on=, how='inner', left_on=, right_on=, suffixes=)` | SQL JOIN (cross-source supported) |
| `.merge(other, on=, how='inner')` | Pandas-style merge |

`how` options: `'inner'`, `'left'`, `'right'`, `'outer'`, `'cross'`

### Mutation

| Method | Description |
|--------|-------------|
| `.assign(**kwargs)` | Add computed columns |
| `.with_column(name, expr)` | Add single column |
| `.drop(columns)` | Remove columns |
| `.rename(columns={})` | Rename columns |
| `.fillna(value)` | Fill NaN |
| `.dropna(subset=)` | Drop rows with NaN |
| `.distinct(subset=, keep='first')` | Deduplicate |

### String & DateTime Accessors

```python
ds['name'].str.upper()
ds['name'].str.contains('pattern')
ds['date'].dt.year
ds['date'].dt.month
```

### Inspection & Execution

| Property/Method | Description |
|----------------|-------------|
| `.columns` | Column names (triggers execution) |
| `.shape` | (rows, cols) tuple |
| `.dtypes` | Column types |
| `.head()` / `.tail()` | Preview rows |
| `.describe()` | Statistics |
| `.info()` | DataFrame info |
| `.to_sql()` | View generated SQL |
| `.explain()` | Execution plan |

Execution triggers naturally: `print()`, `len()`, `.columns`, `for row in ds`, `.equals()`.

### Writing Data

```python
target = DataStore("file", path="output.parquet", format="Parquet")
target.insert_into("col1", "col2").select_from(
    source.select("col1", "col2").filter(source['value'] > 100)
).execute()
```

---

## Session

```python
from chdb import session as chs
sess = chs.Session(path=":memory:")     # in-memory
sess = chs.Session(path="./mydb")       # persistent
```

| Method | Description |
|--------|-------------|
| `query(sql, fmt="CSV", params=None)` | Execute with state |
| `send_query(sql, format="CSV")` | Streaming (returns iterator) |
| `close()` | Close session |

---

## ClickHouse Table Functions (for raw SQL)

| Function | SQL Example |
|----------|-------------|
| `file()` | `SELECT * FROM file('data.csv', CSVWithNames)` |
| `s3()` | `SELECT * FROM s3('s3://bucket/key', 'KEY', 'SECRET', 'Parquet')` |
| `url()` | `SELECT * FROM url('https://example.com/data.json', JSONEachRow)` |
| `gcs()` | `SELECT * FROM gcs('gs://bucket/path', NOSIGN)` |
| `azureBlobStorage()` | `SELECT * FROM azureBlobStorage('conn_str', 'container', 'path', 'Format')` |
| `hdfs()` | `SELECT * FROM hdfs('hdfs://node:9000/path', 'Parquet')` |
| `mysql()` | `SELECT * FROM mysql('host:3306', 'db', 'table', 'user', 'pass')` |
| `postgresql()` | `SELECT * FROM postgresql('host:5432', 'db', 'table', 'user', 'pass')` |
| `remote()` / `remoteSecure()` | `SELECT * FROM remote('host:9000', 'db', 'table', 'user', 'pass')` |
| `mongodb()` | `SELECT * FROM mongodb('host:27017', 'db', 'collection', 'user', 'pass')` |
| `sqlite()` | `SELECT * FROM sqlite('/path/to/db.db', 'table')` |
| `iceberg()` | `SELECT * FROM iceberg('s3://bucket/iceberg/table', 'KEY', 'SECRET')` |
| `deltaLake()` | `SELECT * FROM deltaLake('s3://bucket/delta/table', 'KEY', 'SECRET')` |
| `hudi()` | `SELECT * FROM hudi('s3://bucket/hudi/table', 'KEY', 'SECRET')` |
| `numbers()` | `SELECT * FROM numbers(100)` |
| `Python()` | `SELECT * FROM Python(df)` |

## ClickHouse SQL Functions (commonly used)

| Category | Functions |
|----------|-----------|
| **Aggregate** | `count()`, `sum()`, `avg()`, `min()`, `max()`, `groupArray()`, `quantile(0.95)(col)`, `uniqExact()` |
| **String** | `lower()`, `upper()`, `trim()`, `splitByChar()`, `replaceAll()`, `like()`, `match()` (regex) |
| **Date** | `toDate()`, `toDateTime()`, `now()`, `today()`, `dateDiff()`, `formatDateTime()` |
| **Type** | `toInt32()`, `toFloat64()`, `toString()`, `CAST(x AS Type)` |
| **Conditional** | `if(cond, then, else)`, `multiIf()`, `CASE WHEN` |
| **Array** | `arrayJoin()`, `arrayMap()`, `arrayFilter()`, `length()` |
| **JSON** | `JSONExtract()`, `JSONExtractString()`, `simpleJSONExtractString()` |
| **Window** | `row_number()`, `rank()`, `lag()`, `lead()` over `OVER (PARTITION BY ... ORDER BY ...)` |

## DataStore Configuration

```python
from datastore import config

config.use_chdb()         # prefer chDB/SQL backend
config.use_pandas()       # prefer pandas backend
config.prefer_chdb()      # prefer chDB when possible
config.prefer_pandas()    # prefer pandas when possible
config.enable_debug()     # verbose logging
config.enable_profiling() # performance profiling
```
