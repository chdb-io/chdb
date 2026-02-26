<div align="center">
   <a href="https://clickhouse.com/blog/chdb-joins-clickhouse-family">📢 chDB joins the ClickHouse family 🐍+🚀</a>
</div>
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/chdb-io/chdb/raw/main/docs/_static/snake-chdb-dark.png" height="130">
  <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/snake-chdb.png" height="130">
</picture>

[![Build X86](https://github.com/chdb-io/chdb/actions/workflows/build_linux_x86_wheels.yml/badge.svg?event=release)](https://github.com/chdb-io/chdb/actions/workflows/build_linux_x86_wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/chdb.svg)](https://pypi.org/project/chdb/)
[![Downloads](https://static.pepy.tech/badge/chdb)](https://pepy.tech/project/chdb)
[![Discord](https://img.shields.io/discord/1098133460310294528?logo=Discord)](https://discord.gg/D2Daa2fM5K)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/chdb_io)
</div>

# chDB


> chDB is an in-process SQL OLAP Engine powered by ClickHouse  [^1]
> For more details: [The birth of chDB](https://auxten.com/the-birth-of-chdb/)


## Features

* **🐼 Pandas-compatible DataStore API** - Use familiar pandas syntax with ClickHouse performance
* In-process SQL OLAP Engine, powered by ClickHouse
* No need to install ClickHouse
* Minimized data copy from C++ to Python with [python memoryview](https://docs.python.org/3/c-api/memoryview.html)
* Input&Output support Parquet, CSV, JSON, Arrow, ORC and 60+[more](https://clickhouse.com/docs/en/interfaces/formats) formats
* Support Python DB API 2.0

## Arch
<div align="center">
  <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/arch-chdb3.png" width="450">
</div>

## Installation
Currently, chDB supports Python 3.9+ on macOS and Linux (x86_64 and ARM64).
```bash
pip install chdb
```

<br>

---

## 🐼 DataStore: Pandas-Compatible API (Recommended)

DataStore provides a **familiar pandas-like API** with automatic SQL generation and ClickHouse performance. Write pandas code, get SQL performance - no learning curve required.

### Quick Start (30 seconds)

Just change your import - use the pandas API you already know:

```python
import datastore as pd  # That's it! Use pandas API as usual

# Create a DataFrame - works exactly like pandas
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['NYC', 'LA', 'NYC', 'LA']
})

# Filter with familiar pandas syntax
result = df[df['age'] > 26]
print(result)
#       name  age city
# 1      Bob   30   LA
# 2  Charlie   35  NYC
# 3    Diana   28   LA

# GroupBy works too
print(df.groupby('city')['age'].mean())
# city
# LA     29.0
# NYC    30.0
```

**✨ Zero code changes required.** All operations are lazy - they're recorded and compiled into optimized SQL, executed only when results are needed.

### Why DataStore?

| Feature | pandas | DataStore |
|---------|--------|-----------|
| API | ✅ Familiar | ✅ Same pandas API |
| Large datasets | ❌ Memory limited | ✅ SQL-optimized |
| Learning curve | ✅ Easy | ✅ None - same syntax |
| Performance | ❌ Single-threaded | ✅ ClickHouse engine |

### Architecture

<div align="center">
  <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/datastore_architecture.png" width="700">
</div>

DataStore uses **lazy evaluation** with **dual-engine execution**:
1. **Lazy Operation Chain**: Operations are recorded, not executed immediately
2. **Smart Engine Selection**: QueryPlanner routes each segment to optimal engine (chDB for SQL, Pandas for complex ops)
3. **Intermediate Caching**: Results cached at each step for fast iterative exploration

### Working with Files

```python
from datastore import DataStore

# Load any file format
ds = DataStore.from_file("data.parquet")  # or CSV, JSON, ORC...

# Explore your data
print(ds.head())       # Preview first 5 rows
print(ds.shape)        # (rows, columns)
print(ds.columns)      # Column names

# Build queries with method chaining
result = (ds
    .select("product", "revenue", "date")
    .filter(ds.revenue > 1000)
    .sort("revenue", ascending=False)
    .head(10))

print(result)
```

### Query Any Data Source

```python
from datastore import DataStore

# S3 (with anonymous access)
ds = DataStore.uri("s3://bucket/data.parquet?nosign=true")

# MySQL
ds = DataStore.uri("mysql://user:pass@localhost:3306/mydb/users")

# PostgreSQL
ds = DataStore.uri("postgresql://user:pass@localhost:5432/mydb/products")

# And more: SQLite, MongoDB, ClickHouse, HDFS, Azure, GCS...
```

### Pandas API Coverage

DataStore implements **comprehensive pandas compatibility**:

| Category | Coverage |
|----------|----------|
| DataFrame methods | 209 methods |
| Series.str accessor | 56 methods |
| Series.dt accessor | 42+ methods |
| ClickHouse SQL functions | 334 functions |

```python
# All these pandas methods work:
df.drop(columns=['unused'])
df.fillna(0)
df.assign(revenue=lambda x: x['price'] * x['quantity'])
df.sort_values('revenue', ascending=False)
df.groupby('category').agg({'revenue': 'sum', 'quantity': 'mean'})
df.merge(other_df, on='id')
df.pivot_table(values='sales', index='date', columns='product')
# ... and 200+ more
```

### String and DateTime Operations

```python
# String operations via .str accessor
ds['name'].str.upper()
ds['email'].str.contains('@gmail')
ds['text'].str.replace('old', 'new')

# DateTime operations via .dt accessor  
ds['date'].dt.year
ds['date'].dt.month
ds['timestamp'].dt.hour
```

### Documentation

- **[Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md)** - Full list of supported methods
- **[Function Reference](docs/FUNCTIONS.md)** - 334 ClickHouse SQL functions
- **[Migration Guide](docs/PANDAS_MIGRATION_GUIDE.md)** - Step-by-step guide for pandas users

---

<br>

## SQL API

For users who prefer SQL or need advanced ClickHouse features:

### Run in command line
> `python3 -m chdb SQL [OutputFormat]`
```bash
python3 -m chdb "SELECT 1,'abc'" Pretty
```

<br>

### Data Input
The following methods are available to access on-disk and in-memory data formats:

<details>
    <summary><h4>🗂️ Connection based API</h4></summary>

```python
import chdb

# Create a connection (in-memory by default)
conn = chdb.connect(":memory:")
# Or use file-based: conn = chdb.connect("test.db")

# Create a cursor
cur = conn.cursor()

# Execute queries
cur.execute("SELECT number, toString(number) as str FROM system.numbers LIMIT 3")

# Fetch data in different ways
print(cur.fetchone())    # Single row: (0, '0')
print(cur.fetchmany(2))  # Multiple rows: ((1, '1'), (2, '2'))

# Get column information
print(cur.column_names())  # ['number', 'str']
print(cur.column_types())  # ['UInt64', 'String']

# Use the cursor as an iterator
cur.execute("SELECT number FROM system.numbers LIMIT 3")
for row in cur:
    print(row)

# Always close resources when done
cur.close()
conn.close()
```

For more details, see [examples/connect.py](examples/connect.py).
</details>


<details>
    <summary><h4>🗂️ Query On File</h4> (Parquet, CSV, JSON, Arrow, ORC and 60+)</summary>

You can execute SQL and return desired format data.

```python
import chdb
res = chdb.query('select version()', 'Pretty'); print(res)
```

### Work with Parquet or CSV
```python
# See more data type format in tests/format_output.py
res = chdb.query('select * from file("data.parquet", Parquet)', 'JSON'); print(res)
res = chdb.query('select * from file("data.csv", CSV)', 'CSV');  print(res)
print(f"SQL read {res.rows_read()} rows, {res.bytes_read()} bytes, storage read {res.storage_rows_read()} rows, {res.storage_bytes_read()} bytes, elapsed {res.elapsed()} seconds")
```

### Parameterized queries
```python
import chdb

df = chdb.query(
    "SELECT toDate({base_date:String}) + number AS date "
    "FROM numbers({total_days:UInt64}) "
    "LIMIT {items_per_page:UInt64}",
    "DataFrame",
    params={"base_date": "2025-01-01", "total_days": 10, "items_per_page": 2},
)
print(df)
#         date
# 0 2025-01-01
# 1 2025-01-02
```

### Query progress (`progress=auto`)
```python
import chdb

# Connection API
conn = chdb.connect(":memory:?progress=auto")
conn.query("SELECT sum(number) FROM numbers_mt(1e10) GROUP BY number % 10 SETTINGS max_threads=4")
```

```python
import chdb

# One-shot API
res = chdb.query(
    "SELECT sum(number) FROM numbers_mt(1e10) GROUP BY number % 10 SETTINGS max_threads=4",
    options={"progress": "auto"},
)
```

`progress=auto` behavior:
- In terminal runs: show textual progress updates in the terminal.
- Jupyter/Marimo notebook: render progress bar in notebook output.

Other progress options:
- Progress bar:
  - `progress=tty`: write progress to terminal TTY.
  - `progress=err`: write progress to `stderr`.
  - `progress=off`: disable progress bar output.
- Progress table (terminal output):
  - `progress-table=tty`: write progress table to terminal TTY.
  - `progress-table=err`: write progress table to `stderr`.
  - `progress-table=off`: disable progress table output.

### Pandas dataframe output
```python
# See more in https://clickhouse.com/docs/en/interfaces/formats
chdb.query('select * from file("data.parquet", Parquet)', 'Dataframe')
```
</details>

<details>
    <summary><h4>🗂️ Query On Table</h4> (Pandas DataFrame, Parquet file/bytes, Arrow bytes) </summary>

### Query On Pandas DataFrame
```python
import chdb.dataframe as cdf
import pandas as pd
# Join 2 DataFrames
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': ["one", "two", "three"]})
df2 = pd.DataFrame({'c': [1, 2, 3], 'd': ["①", "②", "③"]})
ret_tbl = cdf.query(sql="select * from __tbl1__ t1 join __tbl2__ t2 on t1.a = t2.c",
                  tbl1=df1, tbl2=df2)
print(ret_tbl)
# Query on the DataFrame Table
print(ret_tbl.query('select b, sum(a) from __table__ group by b'))
# Pandas DataFrames are automatically registered as temporary tables in ClickHouse
chdb.query("SELECT * FROM Python(df1) t1 JOIN Python(df2) t2 ON t1.a = t2.c").show()
```
</details>

<details>
  <summary><h4>🗂️ Query with Stateful Session</h4></summary>

```python
from chdb import session as chs

## Create DB, Table, View in temp session, auto cleanup when session is deleted.
sess = chs.Session()
sess.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic")
sess.query("CREATE TABLE IF NOT EXISTS db_xxx.log_table_xxx (x String, y Int) ENGINE = Log;")
sess.query("INSERT INTO db_xxx.log_table_xxx VALUES ('a', 1), ('b', 3), ('c', 2), ('d', 5);")
sess.query(
    "CREATE VIEW db_xxx.view_xxx AS SELECT * FROM db_xxx.log_table_xxx LIMIT 4;"
)
print("Select from view:\n")
print(sess.query("SELECT * FROM db_xxx.view_xxx", "Pretty"))
```

see also: [test_stateful.py](tests/test_stateful.py).
</details>

<details>
    <summary><h4>🗂️ Query with Python DB-API 2.0</h4></summary>

```python
import chdb.dbapi as dbapi
print("chdb driver version: {0}".format(dbapi.get_client_info()))

conn1 = dbapi.connect()
cur1 = conn1.cursor()
cur1.execute('select version()')
print("description: ", cur1.description)
print("data: ", cur1.fetchone())
cur1.close()
conn1.close()
```
</details>


<details>
    <summary><h4>🗂️ Query with UDF (User Defined Functions)</h4></summary>

```python
from chdb.udf import chdb_udf
from chdb import query

@chdb_udf()
def sum_udf(lhs, rhs):
    return int(lhs) + int(rhs)

print(query("select sum_udf(12,22)"))
```

Some notes on chDB Python UDF(User Defined Function) decorator.
1. The function should be stateless. So, only UDFs are supported, not UDAFs(User Defined Aggregation Function).
2. Default return type is String. If you want to change the return type, you can pass in the return type as an argument.
    The return type should be one of the following: https://clickhouse.com/docs/en/sql-reference/data-types
3. The function should take in arguments of type String. As the input is TabSeparated, all arguments are strings.
4. The function will be called for each line of input. Something like this:
    ```
    def sum_udf(lhs, rhs):
        return int(lhs) + int(rhs)

    for line in sys.stdin:
        args = line.strip().split('\t')
        lhs = args[0]
        rhs = args[1]
        print(sum_udf(lhs, rhs))
        sys.stdout.flush()
    ```
5. The function should be pure python function. You SHOULD import all python modules used IN THE FUNCTION.
    ```
    def func_use_json(arg):
        import json
        ...
    ```
6. Python interpertor used is the same as the one used to run the script. Get from `sys.executable`

see also: [test_udf.py](tests/test_udf.py).
</details>


<details>
    <summary><h4>🗂️ Streaming Query</h4></summary>

Process large datasets with constant memory usage through chunked streaming.

```python
from chdb import session as chs

sess = chs.Session()

# Example 1: Basic example of using streaming query
rows_cnt = 0
with sess.send_query("SELECT * FROM numbers(200000)", "CSV") as stream_result:
    for chunk in stream_result:
        rows_cnt += chunk.rows_read()

print(rows_cnt) # 200000

# Example 2: Manual iteration with fetch()
rows_cnt = 0
stream_result = sess.send_query("SELECT * FROM numbers(200000)", "CSV")
while True:
    chunk = stream_result.fetch()
    if chunk is None:
        break
    rows_cnt += chunk.rows_read()

print(rows_cnt) # 200000
```

For more details, see [test_streaming_query.py](tests/test_streaming_query.py).
</details>


<details>
    <summary><h4>🗂️ Python Table Engine</h4></summary>

### Query on Pandas DataFrame

```python
import chdb
import pandas as pd
df = pd.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6],
        "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
    }
)

chdb.query("SELECT b, sum(a) FROM Python(df) GROUP BY b ORDER BY b").show()
```

### Query on Arrow Table

```python
import chdb
import pyarrow as pa
arrow_table = pa.table(
    {
        "a": [1, 2, 3, 4, 5, 6],
        "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
    }
)

chdb.query("SELECT b, sum(a) FROM Python(arrow_table) GROUP BY b ORDER BY b").show()
```

see also: [test_query_py.py](tests/test_query_py.py).
</details>

<details>
  <summary><h4>🧠 AI-assisted SQL generation</h4></summary>

chDB can translate natural language prompts into SQL. Configure the AI client through the connection (or session) string parameters:

- `ai_provider`: `openai` or `anthropic`. Defaults to OpenAI-compatible when `ai_base_url` is set, otherwise auto-detected.
- `ai_api_key`: API key; falls back to `AI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` env vars.
- `ai_base_url`: Custom base URL for OpenAI-compatible endpoints.
- `ai_model`: Model name (e.g., `gpt-4o-mini`, `claude-3-opus-20240229`).

```python
import chdb

# Use env OPENAI_API_KEY/AI_API_KEY/ANTHROPIC_API_KEY for credentials
conn = chdb.connect("file::memory:?ai_provider=openai&ai_model=gpt-4o-mini")
conn.query("CREATE TABLE nums (n UInt32) ENGINE = Memory")
conn.query("INSERT INTO nums VALUES (1), (2), (3)")

sql = conn.generate_sql("Select all rows from nums ordered by n desc")
print(sql)  # e.g., SELECT * FROM nums ORDER BY n DESC

# ask(): one-call generate + execute
print(conn.ask("List the numbers table", format="Pretty"))
```

</details>

For more examples, see [examples](examples) and [tests](tests).

<br>

## Demos and Examples

- [Project Documentation](https://clickhouse.com/docs/en/chdb) and [Usage Examples](https://clickhouse.com/docs/en/chdb/install/python)
- [Colab Notebooks](https://colab.research.google.com/drive/1-zKB6oKfXeptggXi0kUX87iR8ZTSr4P3?usp=sharing) and other [Script Examples](examples)

## Benchmark

- [ClickBench of embedded engines](https://benchmark.clickhouse.com/#eyJzeXN0ZW0iOnsiQXRoZW5hIChwYXJ0aXRpb25lZCkiOnRydWUsIkF0aGVuYSAoc2luZ2xlKSI6dHJ1ZSwiQXVyb3JhIGZvciBNeVNRTCI6dHJ1ZSwiQXVyb3JhIGZvciBQb3N0Z3JlU1FMIjp0cnVlLCJCeXRlSG91c2UiOnRydWUsImNoREIiOnRydWUsIkNpdHVzIjp0cnVlLCJjbGlja2hvdXNlLWxvY2FsIChwYXJ0aXRpb25lZCkiOnRydWUsImNsaWNraG91c2UtbG9jYWwgKHNpbmdsZSkiOnRydWUsIkNsaWNrSG91c2UiOnRydWUsIkNsaWNrSG91c2UgKHR1bmVkKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoenN0ZCkiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQiOnRydWUsIkNsaWNrSG91c2UgKHdlYikiOnRydWUsIkNyYXRlREIiOnRydWUsIkRhdGFiZW5kIjp0cnVlLCJEYXRhRnVzaW9uIChzaW5nbGUpIjp0cnVlLCJBcGFjaGUgRG9yaXMiOnRydWUsIkRydWlkIjp0cnVlLCJEdWNrREIgKFBhcnF1ZXQpIjp0cnVlLCJEdWNrREIiOnRydWUsIkVsYXN0aWNzZWFyY2giOnRydWUsIkVsYXN0aWNzZWFyY2ggKHR1bmVkKSI6ZmFsc2UsIkdyZWVucGx1bSI6dHJ1ZSwiSGVhdnlBSSI6dHJ1ZSwiSHlkcmEiOnRydWUsIkluZm9icmlnaHQiOnRydWUsIktpbmV0aWNhIjp0cnVlLCJNYXJpYURCIENvbHVtblN0b3JlIjp0cnVlLCJNYXJpYURCIjpmYWxzZSwiTW9uZXREQiI6dHJ1ZSwiTW9uZ29EQiI6dHJ1ZSwiTXlTUUwgKE15SVNBTSkiOnRydWUsIk15U1FMIjp0cnVlLCJQaW5vdCI6dHJ1ZSwiUG9zdGdyZVNRTCI6dHJ1ZSwiUG9zdGdyZVNRTCAodHVuZWQpIjpmYWxzZSwiUXVlc3REQiAocGFydGl0aW9uZWQpIjp0cnVlLCJRdWVzdERCIjp0cnVlLCJSZWRzaGlmdCI6dHJ1ZSwiU2VsZWN0REIiOnRydWUsIlNpbmdsZVN0b3JlIjp0cnVlLCJTbm93Zmxha2UiOnRydWUsIlNRTGl0ZSI6dHJ1ZSwiU3RhclJvY2tzIjp0cnVlLCJUaW1lc2NhbGVEQiAoY29tcHJlc3Npb24pIjp0cnVlLCJUaW1lc2NhbGVEQiI6dHJ1ZX0sInR5cGUiOnsic3RhdGVsZXNzIjpmYWxzZSwibWFuYWdlZCI6ZmFsc2UsIkphdmEiOmZhbHNlLCJjb2x1bW4tb3JpZW50ZWQiOmZhbHNlLCJDKysiOmZhbHNlLCJNeVNRTCBjb21wYXRpYmxlIjpmYWxzZSwicm93LW9yaWVudGVkIjpmYWxzZSwiQyI6ZmFsc2UsIlBvc3RncmVTUUwgY29tcGF0aWJsZSI6ZmFsc2UsIkNsaWNrSG91c2UgZGVyaXZhdGl2ZSI6ZmFsc2UsImVtYmVkZGVkIjp0cnVlLCJzZXJ2ZXJsZXNzIjpmYWxzZSwiUnVzdCI6ZmFsc2UsInNlYXJjaCI6ZmFsc2UsImRvY3VtZW50IjpmYWxzZSwidGltZS1zZXJpZXMiOmZhbHNlfSwibWFjaGluZSI6eyJzZXJ2ZXJsZXNzIjp0cnVlLCIxNmFjdSI6dHJ1ZSwiTCI6dHJ1ZSwiTSI6dHJ1ZSwiUyI6dHJ1ZSwiWFMiOnRydWUsImM2YS5tZXRhbCwgNTAwZ2IgZ3AyIjp0cnVlLCJjNmEuNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJjNS40eGxhcmdlLCA1MDBnYiBncDIiOnRydWUsIjE2IHRocmVhZHMiOnRydWUsIjIwIHRocmVhZHMiOnRydWUsIjI0IHRocmVhZHMiOnRydWUsIjI4IHRocmVhZHMiOnRydWUsIjMwIHRocmVhZHMiOnRydWUsIjQ4IHRocmVhZHMiOnRydWUsIjYwIHRocmVhZHMiOnRydWUsIm01ZC4yNHhsYXJnZSI6dHJ1ZSwiYzVuLjR4bGFyZ2UsIDIwMGdiIGdwMiI6dHJ1ZSwiYzZhLjR4bGFyZ2UsIDE1MDBnYiBncDIiOnRydWUsImRjMi44eGxhcmdlIjp0cnVlLCJyYTMuMTZ4bGFyZ2UiOnRydWUsInJhMy40eGxhcmdlIjp0cnVlLCJyYTMueGxwbHVzIjp0cnVlLCJTMjQiOnRydWUsIlMyIjp0cnVlLCIyWEwiOnRydWUsIjNYTCI6dHJ1ZSwiNFhMIjp0cnVlLCJYTCI6dHJ1ZX0sImNsdXN0ZXJfc2l6ZSI6eyIxIjp0cnVlLCIyIjp0cnVlLCI0Ijp0cnVlLCI4Ijp0cnVlLCIxNiI6dHJ1ZSwiMzIiOnRydWUsIjY0Ijp0cnVlLCIxMjgiOnRydWUsInNlcnZlcmxlc3MiOnRydWUsInVuZGVmaW5lZCI6dHJ1ZX0sIm1ldHJpYyI6ImhvdCIsInF1ZXJpZXMiOlt0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlXX0=)

- [chDB vs Pandas](https://colab.research.google.com/drive/1FogLujJ_-ds7RGurDrUnK-U0IW8a8Qd0)

- [Benchmark on DataFrame: chDB Pandas DuckDB Polars](https://benchmark.clickhouse.com/#eyJzeXN0ZW0iOnsiQWxsb3lEQiI6dHJ1ZSwiQWxsb3lEQiAodHVuZWQpIjp0cnVlLCJBdGhlbmEgKHBhcnRpdGlvbmVkKSI6dHJ1ZSwiQXRoZW5hIChzaW5nbGUpIjp0cnVlLCJBdXJvcmEgZm9yIE15U1FMIjp0cnVlLCJBdXJvcmEgZm9yIFBvc3RncmVTUUwiOnRydWUsIkJ5Q29uaXR5Ijp0cnVlLCJCeXRlSG91c2UiOnRydWUsImNoREIgKERhdGFGcmFtZSkiOnRydWUsImNoREIgKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiY2hEQiI6dHJ1ZSwiQ2l0dXMiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQgKGF3cykiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQgKGF6dXJlKSI6dHJ1ZSwiQ2xpY2tIb3VzZSBDbG91ZCAoZ2NwKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoZGF0YSBsYWtlLCBwYXJ0aXRpb25lZCkiOnRydWUsIkNsaWNrSG91c2UgKGRhdGEgbGFrZSwgc2luZ2xlKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoUGFycXVldCwgcGFydGl0aW9uZWQpIjp0cnVlLCJDbGlja0hvdXNlIChQYXJxdWV0LCBzaW5nbGUpIjp0cnVlLCJDbGlja0hvdXNlICh3ZWIpIjp0cnVlLCJDbGlja0hvdXNlIjp0cnVlLCJDbGlja0hvdXNlICh0dW5lZCkiOnRydWUsIkNsaWNrSG91c2UgKHR1bmVkLCBtZW1vcnkpIjp0cnVlLCJDbG91ZGJlcnJ5Ijp0cnVlLCJDcmF0ZURCIjp0cnVlLCJDcnVuY2h5IEJyaWRnZSBmb3IgQW5hbHl0aWNzIChQYXJxdWV0KSI6dHJ1ZSwiRGF0YWJlbmQiOnRydWUsIkRhdGFGdXNpb24gKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiRGF0YUZ1c2lvbiAoUGFycXVldCwgc2luZ2xlKSI6dHJ1ZSwiQXBhY2hlIERvcmlzIjp0cnVlLCJEcnVpZCI6dHJ1ZSwiRHVja0RCIChEYXRhRnJhbWUpIjp0cnVlLCJEdWNrREIgKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiRHVja0RCIjp0cnVlLCJFbGFzdGljc2VhcmNoIjp0cnVlLCJFbGFzdGljc2VhcmNoICh0dW5lZCkiOmZhbHNlLCJHbGFyZURCIjp0cnVlLCJHcmVlbnBsdW0iOnRydWUsIkhlYXZ5QUkiOnRydWUsIkh5ZHJhIjp0cnVlLCJJbmZvYnJpZ2h0Ijp0cnVlLCJLaW5ldGljYSI6dHJ1ZSwiTWFyaWFEQiBDb2x1bW5TdG9yZSI6dHJ1ZSwiTWFyaWFEQiI6ZmFsc2UsIk1vbmV0REIiOnRydWUsIk1vbmdvREIiOnRydWUsIk1vdGhlcmR1Y2siOnRydWUsIk15U1FMIChNeUlTQU0pIjp0cnVlLCJNeVNRTCI6dHJ1ZSwiT3hsYSI6dHJ1ZSwiUGFuZGFzIChEYXRhRnJhbWUpIjp0cnVlLCJQYXJhZGVEQiAoUGFycXVldCwgcGFydGl0aW9uZWQpIjp0cnVlLCJQYXJhZGVEQiAoUGFycXVldCwgc2luZ2xlKSI6dHJ1ZSwiUGlub3QiOnRydWUsIlBvbGFycyAoRGF0YUZyYW1lKSI6dHJ1ZSwiUG9zdGdyZVNRTCAodHVuZWQpIjpmYWxzZSwiUG9zdGdyZVNRTCI6dHJ1ZSwiUXVlc3REQiAocGFydGl0aW9uZWQpIjp0cnVlLCJRdWVzdERCIjp0cnVlLCJSZWRzaGlmdCI6dHJ1ZSwiU2luZ2xlU3RvcmUiOnRydWUsIlNub3dmbGFrZSI6dHJ1ZSwiU1FMaXRlIjp0cnVlLCJTdGFyUm9ja3MiOnRydWUsIlRhYmxlc3BhY2UiOnRydWUsIlRlbWJvIE9MQVAgKGNvbHVtbmFyKSI6dHJ1ZSwiVGltZXNjYWxlREIgKGNvbXByZXNzaW9uKSI6dHJ1ZSwiVGltZXNjYWxlREIiOnRydWUsIlVtYnJhIjp0cnVlfSwidHlwZSI6eyJDIjpmYWxzZSwiY29sdW1uLW9yaWVudGVkIjpmYWxzZSwiUG9zdGdyZVNRTCBjb21wYXRpYmxlIjpmYWxzZSwibWFuYWdlZCI6ZmFsc2UsImdjcCI6ZmFsc2UsInN0YXRlbGVzcyI6ZmFsc2UsIkphdmEiOmZhbHNlLCJDKysiOmZhbHNlLCJNeVNRTCBjb21wYXRpYmxlIjpmYWxzZSwicm93LW9yaWVudGVkIjpmYWxzZSwiQ2xpY2tIb3VzZSBkZXJpdmF0aXZlIjpmYWxzZSwiZW1iZWRkZWQiOmZhbHNlLCJzZXJ2ZXJsZXNzIjpmYWxzZSwiZGF0YWZyYW1lIjp0cnVlLCJhd3MiOmZhbHNlLCJhenVyZSI6ZmFsc2UsImFuYWx5dGljYWwiOmZhbHNlLCJSdXN0IjpmYWxzZSwic2VhcmNoIjpmYWxzZSwiZG9jdW1lbnQiOmZhbHNlLCJzb21ld2hhdCBQb3N0Z3JlU1FMIGNvbXBhdGlibGUiOmZhbHNlLCJ0aW1lLXNlcmllcyI6ZmFsc2V9LCJtYWNoaW5lIjp7IjE2IHZDUFUgMTI4R0IiOnRydWUsIjggdkNQVSA2NEdCIjp0cnVlLCJzZXJ2ZXJsZXNzIjp0cnVlLCIxNmFjdSI6dHJ1ZSwiYzZhLjR4bGFyZ2UsIDUwMGdiIGdwMiI6dHJ1ZSwiTCI6dHJ1ZSwiTSI6dHJ1ZSwiUyI6dHJ1ZSwiWFMiOnRydWUsImM2YS5tZXRhbCwgNTAwZ2IgZ3AyIjp0cnVlLCIxOTJHQiI6dHJ1ZSwiMjRHQiI6dHJ1ZSwiMzYwR0IiOnRydWUsIjQ4R0IiOnRydWUsIjcyMEdCIjp0cnVlLCI5NkdCIjp0cnVlLCJkZXYiOnRydWUsIjcwOEdCIjp0cnVlLCJjNW4uNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJBbmFseXRpY3MtMjU2R0IgKDY0IHZDb3JlcywgMjU2IEdCKSI6dHJ1ZSwiYzUuNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJjNmEuNHhsYXJnZSwgMTUwMGdiIGdwMiI6dHJ1ZSwiY2xvdWQiOnRydWUsImRjMi44eGxhcmdlIjp0cnVlLCJyYTMuMTZ4bGFyZ2UiOnRydWUsInJhMy40eGxhcmdlIjp0cnVlLCJyYTMueGxwbHVzIjp0cnVlLCJTMiI6dHJ1ZSwiUzI0Ijp0cnVlLCIyWEwiOnRydWUsIjNYTCI6dHJ1ZSwiNFhMIjp0cnVlLCJYTCI6dHJ1ZSwiTDEgLSAxNkNQVSAzMkdCIjp0cnVlLCJjNmEuNHhsYXJnZSwgNTAwZ2IgZ3AzIjp0cnVlfSwiY2x1c3Rlcl9zaXplIjp7IjEiOnRydWUsIjIiOnRydWUsIjQiOnRydWUsIjgiOnRydWUsIjE2Ijp0cnVlLCIzMiI6dHJ1ZSwiNjQiOnRydWUsIjEyOCI6dHJ1ZSwic2VydmVybGVzcyI6dHJ1ZX0sIm1ldHJpYyI6ImhvdCIsInF1ZXJpZXMiOlt0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlXX0=)


<div align="center">
    <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/df_bench.png" width="800">
</div>


## Documentation
- For chdb specific examples and documentation refer to [chDB docs](https://clickhouse.com/docs/en/chdb)
- For SQL syntax, please refer to [ClickHouse SQL Reference](https://clickhouse.com/docs/en/sql-reference/syntax)
- For DataStore API, see [Pandas Compatibility Guide](docs/PANDAS_COMPATIBILITY.md)


## AI Coding Agent Skill

chdb provides an [AI Skill](agent/skills/using-chdb/) that teaches AI coding agents (Cursor, Claude Code, etc.) chdb's multi-source data analytics API. Install it so your AI assistant can write correct chdb code out of the box:

```bash
curl -sL https://raw.githubusercontent.com/chdb-io/chdb/main/install_skill.sh | bash
```


## Events

- Demo chDB at [ClickHouse v23.7 livehouse!](https://t.co/todc13Kn19) and [Slides](https://docs.google.com/presentation/d/1ikqjOlimRa7QAg588TAB_Fna-Tad2WMg7_4AgnbQbFA/edit?usp=sharing)

## Contributing
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
There are something you can help:
- [ ] Help test and report bugs
- [ ] Help improve documentation
- [ ] Help improve code quality and performance

### Bindings

We welcome bindings for other languages, please refer to [bindings](bindings.md) for more details.

## Version Guide

Please refer to [VERSION-GUIDE.md](VERSION-GUIDE.md) for more details.

## Paper

- [ClickHouse - Lightning Fast Analytics for Everyone](https://www.vldb.org/pvldb/vol17/p3731-schulze.pdf)

## License
Apache 2.0, see [LICENSE](LICENSE.txt) for more information.

## Acknowledgments
chDB is mainly based on [ClickHouse](https://github.com/ClickHouse/ClickHouse) [^1]
for trade mark and other reasons, I named it chDB.

## Contact
- Discord: [https://discord.gg/D2Daa2fM5K](https://discord.gg/D2Daa2fM5K)
- Email: auxten@clickhouse.com
- Twitter: [@chdb](https://twitter.com/chdb_io)


<br>

[^1]: ClickHouse® is a trademark of ClickHouse Inc. All trademarks, service marks, and logos mentioned or depicted are the property of their respective owners. The use of any third-party trademarks, brand names, product names, and company names does not imply endorsement, affiliation, or association with the respective owners.
