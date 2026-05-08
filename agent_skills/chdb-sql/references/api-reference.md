# chdb SQL API Reference

> Complete signatures for the SQL-oriented chdb APIs.

## Table of Contents

- [chdb.query()](#chdbquery)
- [Session](#session)
- [Connection (DB-API 2.0)](#connection-db-api-20)
- [Output Formats](#output-formats)
- [Parametrized Queries](#parametrized-queries)
- [Streaming Queries](#streaming-queries)
- [Progress Callback](#progress-callback)
- [User-Defined Functions (UDF)](#user-defined-functions-udf)
- [AI-Assisted SQL](#ai-assisted-sql)

---

## chdb.query()

```python
chdb.query(sql, output_format="CSV", path="", udf_path="", params=None)
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `sql` | str | _(required)_ | ClickHouse SQL query |
| `output_format` | str | `"CSV"` | Output format (see [Output Formats](#output-formats)) |
| `path` | str | `""` | Database path (empty = in-memory, no state) |
| `udf_path` | str | `""` | Path for UDF scripts |
| `params` | dict | `None` | Named parameters (see [Parametrized Queries](#parametrized-queries)) |

**Returns:** Result object with:

| Property/Method | Description |
|-----------------|-------------|
| `.show()` | Print result to stdout |
| `.bytes()` | Raw bytes of the result |
| `.data()` | Result as string |
| `.rows_read` | Number of rows read |
| `.bytes_read` | Number of bytes read |
| `.elapsed` | Query execution time in seconds |

```python
import chdb

result = chdb.query("SELECT 1 + 1 AS answer")
result.show()       # prints: 2
print(result.data())  # "2\n"

df = chdb.query("SELECT * FROM numbers(10)", "DataFrame")
print(df)  # pandas DataFrame
```

---

## Session

```python
from chdb import session as chs

sess = chs.Session(path=":memory:")     # in-memory (no persistence)
sess = chs.Session(path="./mydb")       # persistent to disk
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `query()` | `(sql, fmt="CSV", params=None)` | Execute SQL with session state |
| `send_query()` | `(sql, format="CSV")` | Streaming query (returns iterator) |
| `close()` | `()` | Close session and release resources |

```python
from chdb import session as chs

sess = chs.Session("./analytics")

sess.query("CREATE TABLE t1 (id UInt64, name String) ENGINE = MergeTree() ORDER BY id")
sess.query("INSERT INTO t1 VALUES (1, 'Alice'), (2, 'Bob')")
result = sess.query("SELECT * FROM t1", "Pretty")
result.show()

sess.close()
```

**Key differences from `chdb.query()`:**
- Session maintains state: tables, databases, and settings persist across calls
- Persistent sessions (`path="./dir"`) survive process restarts
- In-memory sessions (`path=":memory:"`) are discarded on close

---

## Connection (DB-API 2.0)

```python
from chdb import dbapi

conn = dbapi.connect()    # or: dbapi.connect(path="./mydb")
```

| Method | Description |
|--------|-------------|
| `conn.cursor()` | Create a cursor |
| `cur.execute(sql)` | Execute SQL |
| `cur.execute(sql, params)` | Execute with parameters |
| `cur.fetchone()` | Fetch one row |
| `cur.fetchmany(size)` | Fetch `size` rows |
| `cur.fetchall()` | Fetch all rows |
| `cur.description` | Column metadata |
| `cur.close()` | Close cursor |
| `conn.close()` | Close connection |

```python
from chdb import dbapi

conn = dbapi.connect()
cur = conn.cursor()
cur.execute("SELECT number, number * 2 AS doubled FROM numbers(5)")
print(cur.fetchall())
# [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8)]
cur.close()
conn.close()
```

---

## Output Formats

| Format | Description | Use case |
|--------|-------------|----------|
| `"CSV"` | Comma-separated (default) | General export |
| `"CSVWithNames"` | CSV with header row | Spreadsheet import |
| `"JSON"` | JSON object with metadata | API responses |
| `"JSONEachRow"` | One JSON object per line | Streaming / NDJSON |
| `"DataFrame"` | pandas DataFrame | Python analysis |
| `"Arrow"` | Apache Arrow bytes | IPC format |
| `"ArrowTable"` | pyarrow.Table | Arrow ecosystem |
| `"Parquet"` | Parquet bytes | File export |
| `"Pretty"` | Formatted table | Terminal display |
| `"PrettyCompact"` | Compact table | Terminal display |
| `"TabSeparated"` | TSV | Tab-delimited export |
| `"Debug"` | Debug info | Troubleshooting |

```python
import chdb

chdb.query("SELECT 1", "Pretty").show()            # formatted table
df = chdb.query("SELECT * FROM numbers(5)", "DataFrame")  # pandas DataFrame
arrow = chdb.query("SELECT 1", "ArrowTable")        # pyarrow Table
```

---

## Parametrized Queries

Use `{name:Type}` placeholders in SQL, and pass values via `params`:

```python
import chdb

result = chdb.query(
    """
    SELECT toDate({start:String}) + number AS date, rand() % 1000 AS value
    FROM numbers({days:UInt64})
    """,
    "DataFrame",
    params={"start": "2025-01-01", "days": 30})
print(result)
```

Supported types: `String`, `UInt8`–`UInt64`, `Int8`–`Int64`, `Float32`, `Float64`, `Date`, `DateTime`.

---

## Streaming Queries

For large results, use `send_query` on a Session to get an iterator:

```python
from chdb import session as chs

sess = chs.Session()
iterator = sess.send_query("SELECT * FROM numbers(1000000)", format="CSV")
for chunk in iterator:
    print(chunk[:100])  # process each chunk
sess.close()
```

---

## Progress Callback

Monitor query progress:

```python
import chdb

def on_progress(progress):
    print(f"Rows: {progress.read_rows}, Bytes: {progress.read_bytes}")

chdb.query("SELECT * FROM numbers(10000000)", "CSV", progress_callback=on_progress)
```

---

## User-Defined Functions (UDF)

Register Python functions as SQL UDFs using the `@chdb_udf` decorator:

```python
from chdb.udf import chdb_udf

@chdb_udf()
def my_multiply(x, y):
    return x * y

import chdb
result = chdb.query("SELECT my_multiply(number, 10) FROM numbers(5)", "DataFrame")
print(result)
```

**Limitations:**
- UDFs execute in-process, not distributed
- Arguments and return values must be scalar types
- Performance may be lower than native ClickHouse functions for large datasets

---

## AI-Assisted SQL

Generate SQL queries from natural language:

```python
import chdb

sql = chdb.generate_sql("top 10 countries by revenue from orders.parquet")
print(sql)
# SELECT country, sum(revenue) AS total_revenue
# FROM file('orders.parquet', Parquet)
# GROUP BY country
# ORDER BY total_revenue DESC
# LIMIT 10

result = chdb.ask("What are the top products by sales?", data="sales.parquet")
print(result)
```

**Note:** These features require an LLM API key configured via environment variables.
