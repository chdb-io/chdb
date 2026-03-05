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

# chdb-core

> chdb-core is the foundational engine of the [chDB](https://github.com/chdb-io/chdb) ecosystem — an in-process SQL OLAP Engine powered by ClickHouse [^1]

## Table of Contents

- [chDB Ecosystem](#chdb-ecosystem)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Demos and Examples](#demos-and-examples)
- [Benchmark](#benchmark)
- [Documentation](#documentation)
- [Contributing](#contributing)

---

## chDB Ecosystem

The chDB project is split into two packages:

| Package | Role | Install |
|---|---|---|
| **chdb-core** (this repo) | C++ engine + Session / Connection / DB-API interfaces | `pip install chdb-core` |
| [**chDB**](https://github.com/chdb-io/chdb) | Pandas-compatible DataStore API built on top of chdb-core | `pip install chdb` |

<div align="center">
<pre>
┌───────────────────────────────────────────┐
│          chDB (pip install chdb)          │
│  ┌─────────────────────────────────────┐  │
│  │  DataStore: pandas-like lazy API    │  │
│  │  QueryPlanner / dual-engine exec    │  │
│  └──────────────────┬──────────────────┘  │
│                     │                     │
│  ┌──────────────────▼──────────────────┐  │
│  │  chdb-core (pip install chdb-core)  │  │
│  │  C++ ClickHouse Engine              │  │
│  │  Session / Connection / DB-API      │  │
│  │  query() / UDF / Stream             │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
</pre>
</div>

**chdb-core** provides everything you need to run SQL queries with ClickHouse performance — no server required. If you want a higher-level pandas-compatible DataFrame API, install **chDB** instead.

---

## Features

* In-process SQL OLAP Engine, powered by ClickHouse
* No need to install ClickHouse
* Minimized data copy from C++ to Python with [python memoryview](https://docs.python.org/3/c-api/memoryview.html)
* Input & Output support Parquet, CSV, JSON, Arrow, ORC and [60+ more formats](https://clickhouse.com/docs/en/interfaces/formats)
* Session and Connection management with stateful queries
* Streaming query support for constant-memory processing
* Python DB-API 2.0 compliance
* User Defined Functions (UDF) support
* AI-assisted SQL generation

---

## Architecture

<div align="center">
  <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/arch-chdb3.png" width="450">
</div>

---

## Installation

Currently, chdb-core supports Python 3.9+ on macOS and Linux (x86_64 and ARM64).

```bash
pip install chdb-core
```

---

## Quick Start

```python
import chdb

result = chdb.query("SELECT version()", "Pretty")
print(result)
```

---

## API Reference

<details>
<summary><b>One-shot Query</b></summary>

The simplest way to run SQL — no session or connection needed:

```python
import chdb

# Basic query with CSV output (default)
result = chdb.query("SELECT 1, 'hello'")
print(result)

# Pandas DataFrame output
df = chdb.query("SELECT number, number * 2 AS double FROM numbers(10)", "DataFrame")
print(df)

# Parameterized queries
df = chdb.query(
    "SELECT toDate({base_date:String}) + number AS date "
    "FROM numbers({total_days:UInt64}) "
    "LIMIT {items_per_page:UInt64}",
    "DataFrame",
    params={"base_date": "2025-01-01", "total_days": 10, "items_per_page": 5},
)
print(df)
```

</details>

<details>
<summary><b>Query on Files (Parquet, CSV, JSON, Arrow, ORC and 60+)</b></summary>

```python
import chdb

res = chdb.query('SELECT * FROM file("data.parquet", Parquet)', "JSON")
print(res)

res = chdb.query('SELECT * FROM file("data.csv", CSV)', "CSV")
print(res)

# Query result statistics
print(f"SQL read {res.rows_read()} rows, {res.bytes_read()} bytes, "
      f"storage read {res.storage_rows_read()} rows, {res.storage_bytes_read()} bytes, "
      f"elapsed {res.elapsed()} seconds")

# Pandas DataFrame output
chdb.query('SELECT * FROM file("data.parquet", Parquet)', "Dataframe")
```

</details>

<details>
<summary><b>Connection API</b></summary>

Connection-based API for cursor-style interaction, supporting both in-memory and file-based databases:

```python
import chdb

conn = chdb.connect(":memory:")
cur = conn.cursor()

cur.execute("CREATE TABLE test (id UInt32, name String) ENGINE = Memory")
cur.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')")
cur.execute("SELECT * FROM test ORDER BY id")

print(cur.fetchone())      # (1, 'Alice')
print(cur.fetchmany(2))    # ((2, 'Bob'), (3, 'Charlie'))
print(cur.column_names())  # ['id', 'name']
print(cur.column_types())  # ['UInt32', 'String']

# Use the cursor as an iterator
cur.execute("SELECT number FROM system.numbers LIMIT 3")
for row in cur:
    print(row)

# Always close resources when done
cur.close()
conn.close()
```

For more details, see [examples/connect.py](examples/connect.py).

```python
# File-based persistent database
conn = chdb.connect("mydata.db")
conn.query("CREATE TABLE IF NOT EXISTS logs (ts DateTime, msg String) ENGINE = MergeTree ORDER BY ts")
conn.query("INSERT INTO logs VALUES (now(), 'started')")
result = conn.query("SELECT * FROM logs", "Pretty")
print(result)
conn.close()
```

</details>

<details>
<summary><b>Stateful Session</b></summary>

Sessions provide a higher-level API with automatic resource management:

```python
from chdb import session as chs

sess = chs.Session()
sess.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic")
sess.query("CREATE TABLE IF NOT EXISTS db_xxx.log_table (x String, y Int) ENGINE = Log")
sess.query("INSERT INTO db_xxx.log_table VALUES ('a', 1), ('b', 3), ('c', 2), ('d', 5)")
sess.query("CREATE VIEW db_xxx.view_xxx AS SELECT * FROM db_xxx.log_table LIMIT 4")

print(sess.query("SELECT * FROM db_xxx.view_xxx", "Pretty"))
```

see also: [test_stateful.py](tests/test_stateful.py).

</details>

<details>
<summary><b>Streaming Query</b></summary>

Process large datasets with constant memory usage through chunked streaming:

```python
from chdb import session as chs

sess = chs.Session()

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
<summary><b>Python DB-API 2.0</b></summary>

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
<summary><b>Query on Table (Pandas DataFrame, Parquet file/bytes, Arrow bytes)</b></summary>

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
<summary><b>Python Table Engine</b></summary>

#### Query on Pandas DataFrame

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

#### Query on Arrow Table

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
<summary><b>User Defined Functions (UDF)</b></summary>

```python
from chdb.udf import chdb_udf
from chdb import query

@chdb_udf()
def sum_udf(lhs, rhs):
    return int(lhs) + int(rhs)

print(query("SELECT sum_udf(12, 22)"))
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
<summary><b>Query Progress</b></summary>

```python
import chdb

# Auto-detect: TTY progress in terminal, progress bar in notebook
conn = chdb.connect(":memory:?progress=auto")
conn.query("SELECT sum(number) FROM numbers_mt(1e10) GROUP BY number % 10 SETTINGS max_threads=4")
```

Progress options: `progress=auto` | `progress=tty` | `progress=err` | `progress=off`

</details>

<details>
<summary><b>AI-assisted SQL Generation</b></summary>

```python
import chdb

conn = chdb.connect("file::memory:?ai_provider=openai&ai_model=gpt-4o-mini")
conn.query("CREATE TABLE nums (n UInt32) ENGINE = Memory")
conn.query("INSERT INTO nums VALUES (1), (2), (3)")

sql = conn.generate_sql("Select all rows from nums ordered by n desc")
print(sql)  # SELECT * FROM nums ORDER BY n DESC

print(conn.ask("List the numbers table", format="Pretty"))
```

</details>

<details>
<summary><b>Command Line</b></summary>

> `python3 -m chdb SQL [OutputFormat]`
```bash
python3 -m chdb "SELECT 1,'abc'" Pretty
```

For more examples, see [examples](examples) and [tests](tests).

</details>

---

## Demos and Examples

- [Project Documentation](https://clickhouse.com/docs/en/chdb) and [Usage Examples](https://clickhouse.com/docs/en/chdb/install/python)
- [Colab Notebooks](https://colab.research.google.com/drive/1-zKB6oKfXeptggXi0kUX87iR8ZTSr4P3?usp=sharing) and other [Script Examples](examples)

---

## Benchmark

- [ClickBench of embedded engines](https://benchmark.clickhouse.com/#eyJzeXN0ZW0iOnsiQXRoZW5hIChwYXJ0aXRpb25lZCkiOnRydWUsIkF0aGVuYSAoc2luZ2xlKSI6dHJ1ZSwiQXVyb3JhIGZvciBNeVNRTCI6dHJ1ZSwiQXVyb3JhIGZvciBQb3N0Z3JlU1FMIjp0cnVlLCJCeXRlSG91c2UiOnRydWUsImNoREIiOnRydWUsIkNpdHVzIjp0cnVlLCJjbGlja2hvdXNlLWxvY2FsIChwYXJ0aXRpb25lZCkiOnRydWUsImNsaWNraG91c2UtbG9jYWwgKHNpbmdsZSkiOnRydWUsIkNsaWNrSG91c2UiOnRydWUsIkNsaWNrSG91c2UgKHR1bmVkKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoenN0ZCkiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQiOnRydWUsIkNsaWNrSG91c2UgKHdlYikiOnRydWUsIkNyYXRlREIiOnRydWUsIkRhdGFiZW5kIjp0cnVlLCJEYXRhRnVzaW9uIChzaW5nbGUpIjp0cnVlLCJBcGFjaGUgRG9yaXMiOnRydWUsIkRydWlkIjp0cnVlLCJEdWNrREIgKFBhcnF1ZXQpIjp0cnVlLCJEdWNrREIiOnRydWUsIkVsYXN0aWNzZWFyY2giOnRydWUsIkVsYXN0aWNzZWFyY2ggKHR1bmVkKSI6ZmFsc2UsIkdyZWVucGx1bSI6dHJ1ZSwiSGVhdnlBSSI6dHJ1ZSwiSHlkcmEiOnRydWUsIkluZm9icmlnaHQiOnRydWUsIktpbmV0aWNhIjp0cnVlLCJNYXJpYURCIENvbHVtblN0b3JlIjp0cnVlLCJNYXJpYURCIjpmYWxzZSwiTW9uZXREQiI6dHJ1ZSwiTW9uZ29EQiI6dHJ1ZSwiTXlTUUwgKE15SVNBTSkiOnRydWUsIk15U1FMIjp0cnVlLCJQaW5vdCI6dHJ1ZSwiUG9zdGdyZVNRTCI6dHJ1ZSwiUG9zdGdyZVNRTCAodHVuZWQpIjpmYWxzZSwiUXVlc3REQiAocGFydGl0aW9uZWQpIjp0cnVlLCJRdWVzdERCIjp0cnVlLCJSZWRzaGlmdCI6dHJ1ZSwiU2VsZWN0REIiOnRydWUsIlNpbmdsZVN0b3JlIjp0cnVlLCJTbm93Zmxha2UiOnRydWUsIlNRTGl0ZSI6dHJ1ZSwiU3RhclJvY2tzIjp0cnVlLCJUaW1lc2NhbGVEQiAoY29tcHJlc3Npb24pIjp0cnVlLCJUaW1lc2NhbGVEQiI6dHJ1ZX0sInR5cGUiOnsic3RhdGVsZXNzIjpmYWxzZSwibWFuYWdlZCI6ZmFsc2UsIkphdmEiOmZhbHNlLCJjb2x1bW4tb3JpZW50ZWQiOmZhbHNlLCJDKysiOmZhbHNlLCJNeVNRTCBjb21wYXRpYmxlIjpmYWxzZSwicm93LW9yaWVudGVkIjpmYWxzZSwiQyI6ZmFsc2UsIlBvc3RncmVTUUwgY29tcGF0aWJsZSI6ZmFsc2UsIkNsaWNrSG91c2UgZGVyaXZhdGl2ZSI6ZmFsc2UsImVtYmVkZGVkIjp0cnVlLCJzZXJ2ZXJsZXNzIjpmYWxzZSwiUnVzdCI6ZmFsc2UsInNlYXJjaCI6ZmFsc2UsImRvY3VtZW50IjpmYWxzZSwidGltZS1zZXJpZXMiOmZhbHNlfSwibWFjaGluZSI6eyJzZXJ2ZXJsZXNzIjp0cnVlLCIxNmFjdSI6dHJ1ZSwiTCI6dHJ1ZSwiTSI6dHJ1ZSwiUyI6dHJ1ZSwiWFMiOnRydWUsImM2YS5tZXRhbCwgNTAwZ2IgZ3AyIjp0cnVlLCJjNmEuNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJjNS40eGxhcmdlLCA1MDBnYiBncDIiOnRydWUsIjE2IHRocmVhZHMiOnRydWUsIjIwIHRocmVhZHMiOnRydWUsIjI0IHRocmVhZHMiOnRydWUsIjI4IHRocmVhZHMiOnRydWUsIjMwIHRocmVhZHMiOnRydWUsIjQ4IHRocmVhZHMiOnRydWUsIjYwIHRocmVhZHMiOnRydWUsIm01ZC4yNHhsYXJnZSI6dHJ1ZSwiYzVuLjR4bGFyZ2UsIDIwMGdiIGdwMiI6dHJ1ZSwiYzZhLjR4bGFyZ2UsIDE1MDBnYiBncDIiOnRydWUsImRjMi44eGxhcmdlIjp0cnVlLCJyYTMuMTZ4bGFyZ2UiOnRydWUsInJhMy40eGxhcmdlIjp0cnVlLCJyYTMueGxwbHVzIjp0cnVlLCJTMjQiOnRydWUsIlMyIjp0cnVlLCIyWEwiOnRydWUsIjNYTCI6dHJ1ZSwiNFhMIjp0cnVlLCJYTCI6dHJ1ZX0sImNsdXN0ZXJfc2l6ZSI6eyIxIjp0cnVlLCIyIjp0cnVlLCI0Ijp0cnVlLCI4Ijp0cnVlLCIxNiI6dHJ1ZSwiMzIiOnRydWUsIjY0Ijp0cnVlLCIxMjgiOnRydWUsInNlcnZlcmxlc3MiOnRydWUsInVuZGVmaW5lZCI6dHJ1ZX0sIm1ldHJpYyI6ImhvdCIsInF1ZXJpZXMiOlt0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlXX0=)

- [chDB vs Pandas](https://colab.research.google.com/drive/1FogLujJ_-ds7RGurDrUnK-U0IW8a8Qd0)

- [Benchmark on DataFrame: chDB Pandas DuckDB Polars](https://benchmark.clickhouse.com/#eyJzeXN0ZW0iOnsiQWxsb3lEQiI6dHJ1ZSwiQWxsb3lEQiAodHVuZWQpIjp0cnVlLCJBdGhlbmEgKHBhcnRpdGlvbmVkKSI6dHJ1ZSwiQXRoZW5hIChzaW5nbGUpIjp0cnVlLCJBdXJvcmEgZm9yIE15U1FMIjp0cnVlLCJBdXJvcmEgZm9yIFBvc3RncmVTUUwiOnRydWUsIkJ5Q29uaXR5Ijp0cnVlLCJCeXRlSG91c2UiOnRydWUsImNoREIgKERhdGFGcmFtZSkiOnRydWUsImNoREIgKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiY2hEQiI6dHJ1ZSwiQ2l0dXMiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQgKGF3cykiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQgKGF6dXJlKSI6dHJ1ZSwiQ2xpY2tIb3VzZSBDbG91ZCAoZ2NwKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoZGF0YSBsYWtlLCBwYXJ0aXRpb25lZCkiOnRydWUsIkNsaWNrSG91c2UgKGRhdGEgbGFrZSwgc2luZ2xlKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoUGFycXVldCwgcGFydGl0aW9uZWQpIjp0cnVlLCJDbGlja0hvdXNlIChQYXJxdWV0LCBzaW5nbGUpIjp0cnVlLCJDbGlja0hvdXNlICh3ZWIpIjp0cnVlLCJDbGlja0hvdXNlIjp0cnVlLCJDbGlja0hvdXNlICh0dW5lZCkiOnRydWUsIkNsaWNrSG91c2UgKHR1bmVkLCBtZW1vcnkpIjp0cnVlLCJDbG91ZGJlcnJ5Ijp0cnVlLCJDcmF0ZURCIjp0cnVlLCJDcnVuY2h5IEJyaWRnZSBmb3IgQW5hbHl0aWNzIChQYXJxdWV0KSI6dHJ1ZSwiRGF0YWJlbmQiOnRydWUsIkRhdGFGdXNpb24gKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiRGF0YUZ1c2lvbiAoUGFycXVldCwgc2luZ2xlKSI6dHJ1ZSwiQXBhY2hlIERvcmlzIjp0cnVlLCJEcnVpZCI6dHJ1ZSwiRHVja0RCIChEYXRhRnJhbWUpIjp0cnVlLCJEdWNrREIgKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiRHVja0RCIjp0cnVlLCJFbGFzdGljc2VhcmNoIjp0cnVlLCJFbGFzdGljc2VhcmNoICh0dW5lZCkiOmZhbHNlLCJHbGFyZURCIjp0cnVlLCJHcmVlbnBsdW0iOnRydWUsIkhlYXZ5QUkiOnRydWUsIkh5ZHJhIjp0cnVlLCJJbmZvYnJpZ2h0Ijp0cnVlLCJLaW5ldGljYSI6dHJ1ZSwiTWFyaWFEQiBDb2x1bW5TdG9yZSI6dHJ1ZSwiTWFyaWFEQiI6ZmFsc2UsIk1vbmV0REIiOnRydWUsIk1vbmdvREIiOnRydWUsIk1vdGhlcmR1Y2siOnRydWUsIk15U1FMIChNeUlTQU0pIjp0cnVlLCJNeVNRTCI6dHJ1ZSwiT3hsYSI6dHJ1ZSwiUGFuZGFzIChEYXRhRnJhbWUpIjp0cnVlLCJQYXJhZGVEQiAoUGFycXVldCwgcGFydGl0aW9uZWQpIjp0cnVlLCJQYXJhZGVEQiAoUGFycXVldCwgc2luZ2xlKSI6dHJ1ZSwiUGlub3QiOnRydWUsIlBvbGFycyAoRGF0YUZyYW1lKSI6dHJ1ZSwiUG9zdGdyZVNRTCAodHVuZWQpIjpmYWxzZSwiUG9zdGdyZVNRTCI6dHJ1ZSwiUXVlc3REQiAocGFydGl0aW9uZWQpIjp0cnVlLCJRdWVzdERCIjp0cnVlLCJSZWRzaGlmdCI6dHJ1ZSwiU2luZ2xlU3RvcmUiOnRydWUsIlNub3dmbGFrZSI6dHJ1ZSwiU1FMaXRlIjp0cnVlLCJTdGFyUm9ja3MiOnRydWUsIlRhYmxlc3BhY2UiOnRydWUsIlRlbWJvIE9MQVAgKGNvbHVtbmFyKSI6dHJ1ZSwiVGltZXNjYWxlREIgKGNvbXByZXNzaW9uKSI6dHJ1ZSwiVGltZXNjYWxlREIiOnRydWUsIlVtYnJhIjp0cnVlfSwidHlwZSI6eyJDIjpmYWxzZSwiY29sdW1uLW9yaWVudGVkIjpmYWxzZSwiUG9zdGdyZVNRTCBjb21wYXRpYmxlIjpmYWxzZSwibWFuYWdlZCI6ZmFsc2UsImdjcCI6ZmFsc2UsInN0YXRlbGVzcyI6ZmFsc2UsIkphdmEiOmZhbHNlLCJDKysiOmZhbHNlLCJNeVNRTCBjb21wYXRpYmxlIjpmYWxzZSwicm93LW9yaWVudGVkIjpmYWxzZSwiQ2xpY2tIb3VzZSBkZXJpdmF0aXZlIjpmYWxzZSwiZW1iZWRkZWQiOmZhbHNlLCJzZXJ2ZXJsZXNzIjpmYWxzZSwiZGF0YWZyYW1lIjp0cnVlLCJhd3MiOmZhbHNlLCJhenVyZSI6ZmFsc2UsImFuYWx5dGljYWwiOmZhbHNlLCJSdXN0IjpmYWxzZSwic2VhcmNoIjpmYWxzZSwiZG9jdW1lbnQiOmZhbHNlLCJzb21ld2hhdCBQb3N0Z3JlU1FMIGNvbXBhdGlibGUiOmZhbHNlLCJ0aW1lLXNlcmllcyI6ZmFsc2V9LCJtYWNoaW5lIjp7IjE2IHZDUFUgMTI4R0IiOnRydWUsIjggdkNQVSA2NEdCIjp0cnVlLCJzZXJ2ZXJsZXNzIjp0cnVlLCIxNmFjdSI6dHJ1ZSwiYzZhLjR4bGFyZ2UsIDUwMGdiIGdwMiI6dHJ1ZSwiTCI6dHJ1ZSwiTSI6dHJ1ZSwiUyI6dHJ1ZSwiWFMiOnRydWUsImM2YS5tZXRhbCwgNTAwZ2IgZ3AyIjp0cnVlLCIxOTJHQiI6dHJ1ZSwiMjRHQiI6dHJ1ZSwiMzYwR0IiOnRydWUsIjQ4R0IiOnRydWUsIjcyMEdCIjp0cnVlLCI5NkdCIjp0cnVlLCJkZXYiOnRydWUsIjcwOEdCIjp0cnVlLCJjNW4uNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJBbmFseXRpY3MtMjU2R0IgKDY0IHZDb3JlcywgMjU2IEdCKSI6dHJ1ZSwiYzUuNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJjNmEuNHhsYXJnZSwgMTUwMGdiIGdwMiI6dHJ1ZSwiY2xvdWQiOnRydWUsImRjMi44eGxhcmdlIjp0cnVlLCJyYTMuMTZ4bGFyZ2UiOnRydWUsInJhMy40eGxhcmdlIjp0cnVlLCJyYTMueGxwbHVzIjp0cnVlLCJTMiI6dHJ1ZSwiUzI0Ijp0cnVlLCIyWEwiOnRydWUsIjNYTCI6dHJ1ZSwiNFhMIjp0cnVlLCJYTCI6dHJ1ZSwiTDEgLSAxNkNQVSAzMkdCIjp0cnVlLCJjNmEuNHhsYXJnZSwgNTAwZ2IgZ3AzIjp0cnVlfSwiY2x1c3Rlcl9zaXplIjp7IjEiOnRydWUsIjIiOnRydWUsIjQiOnRydWUsIjgiOnRydWUsIjE2Ijp0cnVlLCIzMiI6dHJ1ZSwiNjQiOnRydWUsIjEyOCI6dHJ1ZSwic2VydmVybGVzcyI6dHJ1ZX0sIm1ldHJpYyI6ImhvdCIsInF1ZXJpZXMiOlt0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlXX0=)


<div align="center">
    <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/df_bench.png" width="800">
</div>
---

## Documentation
- For chDB specific examples and documentation refer to [chDB docs](https://clickhouse.com/docs/en/chdb)
- For SQL syntax, please refer to [ClickHouse SQL Reference](https://clickhouse.com/docs/en/sql-reference/syntax)
- For pandas-like DataStore API, see [chDB](https://github.com/chdb-io/chdb)

---

## AI Coding Agent Skill

chDB provides an [AI Skill](agent/skills/using-chdb/) that teaches AI coding agents (Cursor, Claude Code, etc.) chDB's multi-source data analytics API. Install it so your AI assistant can write correct chDB code out of the box:

```bash
curl -sL https://raw.githubusercontent.com/chdb-io/chdb/main/install_skill.sh | bash
```

---

## Events

- Demo chDB at [ClickHouse v23.7 livehouse!](https://t.co/todc13Kn19) and [Slides](https://docs.google.com/presentation/d/1ikqjOlimRa7QAg588TAB_Fna-Tad2WMg7_4AgnbQbFA/edit?usp=sharing)

---

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

- [ ] Help test and report bugs
- [ ] Help improve documentation
- [ ] Help improve code quality and performance

### Bindings

We welcome bindings for other languages, please refer to [bindings](bindings.md) for more details.

---

## Version Guide

Please refer to [VERSION-GUIDE.md](VERSION-GUIDE.md) for more details.

---

## Paper

- [ClickHouse - Lightning Fast Analytics for Everyone](https://www.vldb.org/pvldb/vol17/p3731-schulze.pdf)

---

## License

Apache 2.0, see [LICENSE](LICENSE.txt) for more information.

---

## Acknowledgments

chDB is mainly based on [ClickHouse](https://github.com/ClickHouse/ClickHouse) [^1]
for trade mark and other reasons, I named it chDB.

---

## Contact

- Discord: [https://discord.gg/D2Daa2fM5K](https://discord.gg/D2Daa2fM5K)
- Email: auxten@clickhouse.com
- Twitter: [@chdb](https://twitter.com/chdb_io)

<br>

[^1]: ClickHouse® is a trademark of ClickHouse Inc. All trademarks, service marks, and logos mentioned or depicted are the property of their respective owners. The use of any third-party trademarks, brand names, product names, and company names does not imply endorsement, affiliation, or association with the respective owners.
