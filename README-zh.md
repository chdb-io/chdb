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

> chdb-core 是 [chDB](https://github.com/chdb-io/chdb) 生态的基础引擎 - 一个由 ClickHouse 驱动的进程内 SQL OLAP 引擎 [^1]

## 目录

- [chDB 生态](#chdb-生态)
- [特性](#特性)
- [架构](#架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [演示和示例](#演示和示例)
- [性能基准](#性能基准)
- [文档](#文档)
- [贡献](#贡献)

---

## chDB 生态

chDB 项目拆分为两个包：

| 包 | 定位 | 安装 |
|---|---|---|
| **chdb-core**（本仓库） | C++ 引擎 + Session / Connection / DB-API 接口 | `pip install chdb-core` |
| [**chDB**](https://github.com/chdb-io/chdb) | 基于 chdb-core 构建的 Pandas 兼容 DataStore API | `pip install chdb` |

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

**chdb-core** 提供了使用 ClickHouse 性能运行 SQL 查询所需的一切 - 无需安装服务器。如果需要更高层次的 Pandas 兼容 DataFrame API，请安装 **chDB**。

---

## 特性

* 由 ClickHouse 驱动的进程内 SQL OLAP 引擎
* 无需安装 ClickHouse
* 通过 [python memoryview](https://docs.python.org/3/c-api/memoryview.html) 最小化 C++ 到 Python 的数据拷贝
* 输入输出支持 Parquet、CSV、JSON、Arrow、ORC 等 [60+ 种格式](https://clickhouse.com/docs/en/interfaces/formats)
* Session 和 Connection 管理，支持有状态查询
* 流式查询支持，常量内存处理大数据集
* 兼容 Python DB-API 2.0
* 支持用户自定义函数（UDF）
* AI 辅助 SQL 生成

---

## 架构

<div align="center">
  <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/arch-chdb3.png" width="450">
</div>

---

## 安装

目前 chdb-core 支持 macOS 和 Linux（x86_64 及 ARM64）上的 Python 3.9+。

```bash
pip install chdb-core
```

---

## 快速开始

```python
import chdb

result = chdb.query("SELECT version()", "Pretty")
print(result)
```

---

## API 参考

<details>
<summary><b>一次性查询</b></summary>

最简单的 SQL 执行方式 - 无需 session 或 connection：

```python
import chdb

# 基本查询，默认 CSV 输出
result = chdb.query("SELECT 1, 'hello'")
print(result)

# Pandas DataFrame 输出
df = chdb.query("SELECT number, number * 2 AS double FROM numbers(10)", "DataFrame")
print(df)

# 参数化查询
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
<summary><b>查询文件（Parquet、CSV、JSON、Arrow、ORC 等 60+ 种格式）</b></summary>

```python
import chdb

res = chdb.query('SELECT * FROM file("data.parquet", Parquet)', "JSON")
print(res)

res = chdb.query('SELECT * FROM file("data.csv", CSV)', "CSV")
print(res)

# 查询结果统计
print(f"SQL read {res.rows_read()} rows, {res.bytes_read()} bytes, "
      f"storage read {res.storage_rows_read()} rows, {res.storage_bytes_read()} bytes, "
      f"elapsed {res.elapsed()} seconds")

# Pandas DataFrame 输出
chdb.query('SELECT * FROM file("data.parquet", Parquet)', "Dataframe")
```

</details>

<details>
<summary><b>Connection API</b></summary>

基于连接的 API，支持游标风格交互，同时支持内存数据库和基于文件的持久化数据库：

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

# 将游标用作迭代器
cur.execute("SELECT number FROM system.numbers LIMIT 3")
for row in cur:
    print(row)

# 使用完毕后关闭资源
cur.close()
conn.close()
```

更多详情见 [examples/connect.py](examples/connect.py)。

```python
# 基于文件的持久化数据库
conn = chdb.connect("mydata.db")
conn.query("CREATE TABLE IF NOT EXISTS logs (ts DateTime, msg String) ENGINE = MergeTree ORDER BY ts")
conn.query("INSERT INTO logs VALUES (now(), 'started')")
result = conn.query("SELECT * FROM logs", "Pretty")
print(result)
conn.close()
```

</details>

<details>
<summary><b>有状态 Session</b></summary>

Session 提供了更高层次的 API，支持自动资源管理：

```python
from chdb import session as chs

sess = chs.Session()
sess.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic")
sess.query("CREATE TABLE IF NOT EXISTS db_xxx.log_table (x String, y Int) ENGINE = Log")
sess.query("INSERT INTO db_xxx.log_table VALUES ('a', 1), ('b', 3), ('c', 2), ('d', 5)")
sess.query("CREATE VIEW db_xxx.view_xxx AS SELECT * FROM db_xxx.log_table LIMIT 4")

print(sess.query("SELECT * FROM db_xxx.view_xxx", "Pretty"))
```

另见: [test_stateful.py](tests/test_stateful.py)。

</details>

<details>
<summary><b>流式查询</b></summary>

通过分块流式处理大数据集，保持恒定内存使用：

```python
from chdb import session as chs

sess = chs.Session()

rows_cnt = 0
with sess.send_query("SELECT * FROM numbers(200000)", "CSV") as stream_result:
    for chunk in stream_result:
        rows_cnt += chunk.rows_read()

print(rows_cnt) # 200000

# 示例 2：使用 fetch() 手动迭代
rows_cnt = 0
stream_result = sess.send_query("SELECT * FROM numbers(200000)", "CSV")
while True:
    chunk = stream_result.fetch()
    if chunk is None:
        break
    rows_cnt += chunk.rows_read()

print(rows_cnt) # 200000
```

更多详情见 [test_streaming_query.py](tests/test_streaming_query.py)。

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
<summary><b>查询表（Pandas DataFrame、Parquet 文件/字节、Arrow 字节）</b></summary>

```python
import chdb.dataframe as cdf
import pandas as pd
# 关联两个 DataFrame
df1 = pd.DataFrame({'a': [1, 2, 3], 'b': ["one", "two", "three"]})
df2 = pd.DataFrame({'c': [1, 2, 3], 'd': ["①", "②", "③"]})
ret_tbl = cdf.query(sql="select * from __tbl1__ t1 join __tbl2__ t2 on t1.a = t2.c",
                  tbl1=df1, tbl2=df2)
print(ret_tbl)
# 在 DataFrame Table 上继续查询
print(ret_tbl.query('select b, sum(a) from __table__ group by b'))
# Pandas DataFrame 会自动注册为 ClickHouse 中的临时表
chdb.query("SELECT * FROM Python(df1) t1 JOIN Python(df2) t2 ON t1.a = t2.c").show()
```

</details>

<details>
<summary><b>Python Table Engine</b></summary>

#### 查询 Pandas DataFrame

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

#### 查询 Arrow Table

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

另见: [test_query_py.py](tests/test_query_py.py)。

</details>

<details>
<summary><b>用户自定义函数（UDF）</b></summary>

```python
from chdb.udf import chdb_udf
from chdb import query

@chdb_udf()
def sum_udf(lhs, rhs):
    return int(lhs) + int(rhs)

print(query("SELECT sum_udf(12, 22)"))
```

关于 chDB Python UDF（用户自定义函数）装饰器的一些说明：
1. 函数必须是无状态的。因此只支持 UDF，不支持 UDAF（用户自定义聚合函数）。
2. 默认返回类型为 String。如需更改返回类型，可以将返回类型作为参数传入。
    返回类型应为以下之一：https://clickhouse.com/docs/en/sql-reference/data-types
3. 函数参数类型为 String。由于输入是 TabSeparated 格式，所有参数都是字符串。
4. 函数会对每行输入调用一次。类似这样：
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
5. 函数必须是纯 Python 函数。所有用到的 Python 模块必须在函数内部导入。
    ```
    def func_use_json(arg):
        import json
        ...
    ```
6. 使用的 Python 解释器与运行脚本的解释器相同。通过 `sys.executable` 获取。

另见: [test_udf.py](tests/test_udf.py)。

</details>

<details>
<summary><b>查询进度</b></summary>

```python
import chdb

# 自动检测：终端中显示文本进度，Notebook 中显示进度条
conn = chdb.connect(":memory:?progress=auto")
conn.query("SELECT sum(number) FROM numbers_mt(1e10) GROUP BY number % 10 SETTINGS max_threads=4")
```

进度选项：`progress=auto` | `progress=tty` | `progress=err` | `progress=off`

</details>

<details>
<summary><b>AI 辅助 SQL 生成</b></summary>

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
<summary><b>命令行</b></summary>

> `python3 -m chdb SQL [OutputFormat]`
```bash
python3 -m chdb "SELECT 1,'abc'" Pretty
```

更多示例见 [examples](examples) 和 [tests](tests)。

</details>

---

## 演示和示例

- [项目文档](https://clickhouse.com/docs/en/chdb) 和 [使用示例](https://clickhouse.com/docs/en/chdb/install/python)
- [Colab Notebooks](https://colab.research.google.com/drive/1-zKB6oKfXeptggXi0kUX87iR8ZTSr4P3?usp=sharing) 和其他[脚本示例](examples)

---

## 性能基准

- [ClickBench of embedded engines](https://benchmark.clickhouse.com/#eyJzeXN0ZW0iOnsiQXRoZW5hIChwYXJ0aXRpb25lZCkiOnRydWUsIkF0aGVuYSAoc2luZ2xlKSI6dHJ1ZSwiQXVyb3JhIGZvciBNeVNRTCI6dHJ1ZSwiQXVyb3JhIGZvciBQb3N0Z3JlU1FMIjp0cnVlLCJCeXRlSG91c2UiOnRydWUsImNoREIiOnRydWUsIkNpdHVzIjp0cnVlLCJjbGlja2hvdXNlLWxvY2FsIChwYXJ0aXRpb25lZCkiOnRydWUsImNsaWNraG91c2UtbG9jYWwgKHNpbmdsZSkiOnRydWUsIkNsaWNrSG91c2UiOnRydWUsIkNsaWNrSG91c2UgKHR1bmVkKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoenN0ZCkiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQiOnRydWUsIkNsaWNrSG91c2UgKHdlYikiOnRydWUsIkNyYXRlREIiOnRydWUsIkRhdGFiZW5kIjp0cnVlLCJEYXRhRnVzaW9uIChzaW5nbGUpIjp0cnVlLCJBcGFjaGUgRG9yaXMiOnRydWUsIkRydWlkIjp0cnVlLCJEdWNrREIgKFBhcnF1ZXQpIjp0cnVlLCJEdWNrREIiOnRydWUsIkVsYXN0aWNzZWFyY2giOnRydWUsIkVsYXN0aWNzZWFyY2ggKHR1bmVkKSI6ZmFsc2UsIkdyZWVucGx1bSI6dHJ1ZSwiSGVhdnlBSSI6dHJ1ZSwiSHlkcmEiOnRydWUsIkluZm9icmlnaHQiOnRydWUsIktpbmV0aWNhIjp0cnVlLCJNYXJpYURCIENvbHVtblN0b3JlIjp0cnVlLCJNYXJpYURCIjpmYWxzZSwiTW9uZXREQiI6dHJ1ZSwiTW9uZ29EQiI6dHJ1ZSwiTXlTUUwgKE15SVNBTSkiOnRydWUsIk15U1FMIjp0cnVlLCJQaW5vdCI6dHJ1ZSwiUG9zdGdyZVNRTCI6dHJ1ZSwiUG9zdGdyZVNRTCAodHVuZWQpIjpmYWxzZSwiUXVlc3REQiAocGFydGl0aW9uZWQpIjp0cnVlLCJRdWVzdERCIjp0cnVlLCJSZWRzaGlmdCI6dHJ1ZSwiU2VsZWN0REIiOnRydWUsIlNpbmdsZVN0b3JlIjp0cnVlLCJTbm93Zmxha2UiOnRydWUsIlNRTGl0ZSI6dHJ1ZSwiU3RhclJvY2tzIjp0cnVlLCJUaW1lc2NhbGVEQiAoY29tcHJlc3Npb24pIjp0cnVlLCJUaW1lc2NhbGVEQiI6dHJ1ZX0sInR5cGUiOnsic3RhdGVsZXNzIjpmYWxzZSwibWFuYWdlZCI6ZmFsc2UsIkphdmEiOmZhbHNlLCJjb2x1bW4tb3JpZW50ZWQiOmZhbHNlLCJDKysiOmZhbHNlLCJNeVNRTCBjb21wYXRpYmxlIjpmYWxzZSwicm93LW9yaWVudGVkIjpmYWxzZSwiQyI6ZmFsc2UsIlBvc3RncmVTUUwgY29tcGF0aWJsZSI6ZmFsc2UsIkNsaWNrSG91c2UgZGVyaXZhdGl2ZSI6ZmFsc2UsImVtYmVkZGVkIjp0cnVlLCJzZXJ2ZXJsZXNzIjpmYWxzZSwiUnVzdCI6ZmFsc2UsInNlYXJjaCI6ZmFsc2UsImRvY3VtZW50IjpmYWxzZSwidGltZS1zZXJpZXMiOmZhbHNlfSwibWFjaGluZSI6eyJzZXJ2ZXJsZXNzIjp0cnVlLCIxNmFjdSI6dHJ1ZSwiTCI6dHJ1ZSwiTSI6dHJ1ZSwiUyI6dHJ1ZSwiWFMiOnRydWUsImM2YS5tZXRhbCwgNTAwZ2IgZ3AyIjp0cnVlLCJjNmEuNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJjNS40eGxhcmdlLCA1MDBnYiBncDIiOnRydWUsIjE2IHRocmVhZHMiOnRydWUsIjIwIHRocmVhZHMiOnRydWUsIjI0IHRocmVhZHMiOnRydWUsIjI4IHRocmVhZHMiOnRydWUsIjMwIHRocmVhZHMiOnRydWUsIjQ4IHRocmVhZHMiOnRydWUsIjYwIHRocmVhZHMiOnRydWUsIm01ZC4yNHhsYXJnZSI6dHJ1ZSwiYzVuLjR4bGFyZ2UsIDIwMGdiIGdwMiI6dHJ1ZSwiYzZhLjR4bGFyZ2UsIDE1MDBnYiBncDIiOnRydWUsImRjMi44eGxhcmdlIjp0cnVlLCJyYTMuMTZ4bGFyZ2UiOnRydWUsInJhMy40eGxhcmdlIjp0cnVlLCJyYTMueGxwbHVzIjp0cnVlLCJTMjQiOnRydWUsIlMyIjp0cnVlLCIyWEwiOnRydWUsIjNYTCI6dHJ1ZSwiNFhMIjp0cnVlLCJYTCI6dHJ1ZX0sImNsdXN0ZXJfc2l6ZSI6eyIxIjp0cnVlLCIyIjp0cnVlLCI0Ijp0cnVlLCI4Ijp0cnVlLCIxNiI6dHJ1ZSwiMzIiOnRydWUsIjY0Ijp0cnVlLCIxMjgiOnRydWUsInNlcnZlcmxlc3MiOnRydWUsInVuZGVmaW5lZCI6dHJ1ZX0sIm1ldHJpYyI6ImhvdCIsInF1ZXJpZXMiOlt0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlXX0=)

- [chDB vs Pandas](https://colab.research.google.com/drive/1FogLujJ_-ds7RGurDrUnK-U0IW8a8Qd0)

- [Benchmark on DataFrame: chDB Pandas DuckDB Polars](https://benchmark.clickhouse.com/#eyJzeXN0ZW0iOnsiQWxsb3lEQiI6dHJ1ZSwiQWxsb3lEQiAodHVuZWQpIjp0cnVlLCJBdGhlbmEgKHBhcnRpdGlvbmVkKSI6dHJ1ZSwiQXRoZW5hIChzaW5nbGUpIjp0cnVlLCJBdXJvcmEgZm9yIE15U1FMIjp0cnVlLCJBdXJvcmEgZm9yIFBvc3RncmVTUUwiOnRydWUsIkJ5Q29uaXR5Ijp0cnVlLCJCeXRlSG91c2UiOnRydWUsImNoREIgKERhdGFGcmFtZSkiOnRydWUsImNoREIgKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiY2hEQiI6dHJ1ZSwiQ2l0dXMiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQgKGF3cykiOnRydWUsIkNsaWNrSG91c2UgQ2xvdWQgKGF6dXJlKSI6dHJ1ZSwiQ2xpY2tIb3VzZSBDbG91ZCAoZ2NwKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoZGF0YSBsYWtlLCBwYXJ0aXRpb25lZCkiOnRydWUsIkNsaWNrSG91c2UgKGRhdGEgbGFrZSwgc2luZ2xlKSI6dHJ1ZSwiQ2xpY2tIb3VzZSAoUGFycXVldCwgcGFydGl0aW9uZWQpIjp0cnVlLCJDbGlja0hvdXNlIChQYXJxdWV0LCBzaW5nbGUpIjp0cnVlLCJDbGlja0hvdXNlICh3ZWIpIjp0cnVlLCJDbGlja0hvdXNlIjp0cnVlLCJDbGlja0hvdXNlICh0dW5lZCkiOnRydWUsIkNsaWNrSG91c2UgKHR1bmVkLCBtZW1vcnkpIjp0cnVlLCJDbG91ZGJlcnJ5Ijp0cnVlLCJDcmF0ZURCIjp0cnVlLCJDcnVuY2h5IEJyaWRnZSBmb3IgQW5hbHl0aWNzIChQYXJxdWV0KSI6dHJ1ZSwiRGF0YWJlbmQiOnRydWUsIkRhdGFGdXNpb24gKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiRGF0YUZ1c2lvbiAoUGFycXVldCwgc2luZ2xlKSI6dHJ1ZSwiQXBhY2hlIERvcmlzIjp0cnVlLCJEcnVpZCI6dHJ1ZSwiRHVja0RCIChEYXRhRnJhbWUpIjp0cnVlLCJEdWNrREIgKFBhcnF1ZXQsIHBhcnRpdGlvbmVkKSI6dHJ1ZSwiRHVja0RCIjp0cnVlLCJFbGFzdGljc2VhcmNoIjp0cnVlLCJFbGFzdGljc2VhcmNoICh0dW5lZCkiOmZhbHNlLCJHbGFyZURCIjp0cnVlLCJHcmVlbnBsdW0iOnRydWUsIkhlYXZ5QUkiOnRydWUsIkh5ZHJhIjp0cnVlLCJJbmZvYnJpZ2h0Ijp0cnVlLCJLaW5ldGljYSI6dHJ1ZSwiTWFyaWFEQiBDb2x1bW5TdG9yZSI6dHJ1ZSwiTWFyaWFEQiI6ZmFsc2UsIk1vbmV0REIiOnRydWUsIk1vbmdvREIiOnRydWUsIk1vdGhlcmR1Y2siOnRydWUsIk15U1FMIChNeUlTQU0pIjp0cnVlLCJNeVNRTCI6dHJ1ZSwiT3hsYSI6dHJ1ZSwiUGFuZGFzIChEYXRhRnJhbWUpIjp0cnVlLCJQYXJhZGVEQiAoUGFycXVldCwgcGFydGl0aW9uZWQpIjp0cnVlLCJQYXJhZGVEQiAoUGFycXVldCwgc2luZ2xlKSI6dHJ1ZSwiUGlub3QiOnRydWUsIlBvbGFycyAoRGF0YUZyYW1lKSI6dHJ1ZSwiUG9zdGdyZVNRTCAodHVuZWQpIjpmYWxzZSwiUG9zdGdyZVNRTCI6dHJ1ZSwiUXVlc3REQiAocGFydGl0aW9uZWQpIjp0cnVlLCJRdWVzdERCIjp0cnVlLCJSZWRzaGlmdCI6dHJ1ZSwiU2luZ2xlU3RvcmUiOnRydWUsIlNub3dmbGFrZSI6dHJ1ZSwiU1FMaXRlIjp0cnVlLCJTdGFyUm9ja3MiOnRydWUsIlRhYmxlc3BhY2UiOnRydWUsIlRlbWJvIE9MQVAgKGNvbHVtbmFyKSI6dHJ1ZSwiVGltZXNjYWxlREIgKGNvbXByZXNzaW9uKSI6dHJ1ZSwiVGltZXNjYWxlREIiOnRydWUsIlVtYnJhIjp0cnVlfSwidHlwZSI6eyJDIjpmYWxzZSwiY29sdW1uLW9yaWVudGVkIjpmYWxzZSwiUG9zdGdyZVNRTCBjb21wYXRpYmxlIjpmYWxzZSwibWFuYWdlZCI6ZmFsc2UsImdjcCI6ZmFsc2UsInN0YXRlbGVzcyI6ZmFsc2UsIkphdmEiOmZhbHNlLCJDKysiOmZhbHNlLCJNeVNRTCBjb21wYXRpYmxlIjpmYWxzZSwicm93LW9yaWVudGVkIjpmYWxzZSwiQ2xpY2tIb3VzZSBkZXJpdmF0aXZlIjpmYWxzZSwiZW1iZWRkZWQiOmZhbHNlLCJzZXJ2ZXJsZXNzIjpmYWxzZSwiZGF0YWZyYW1lIjp0cnVlLCJhd3MiOmZhbHNlLCJhenVyZSI6ZmFsc2UsImFuYWx5dGljYWwiOmZhbHNlLCJSdXN0IjpmYWxzZSwic2VhcmNoIjpmYWxzZSwiZG9jdW1lbnQiOmZhbHNlLCJzb21ld2hhdCBQb3N0Z3JlU1FMIGNvbXBhdGlibGUiOmZhbHNlLCJ0aW1lLXNlcmllcyI6ZmFsc2V9LCJtYWNoaW5lIjp7IjE2IHZDUFUgMTI4R0IiOnRydWUsIjggdkNQVSA2NEdCIjp0cnVlLCJzZXJ2ZXJsZXNzIjp0cnVlLCIxNmFjdSI6dHJ1ZSwiYzZhLjR4bGFyZ2UsIDUwMGdiIGdwMiI6dHJ1ZSwiTCI6dHJ1ZSwiTSI6dHJ1ZSwiUyI6dHJ1ZSwiWFMiOnRydWUsImM2YS5tZXRhbCwgNTAwZ2IgZ3AyIjp0cnVlLCIxOTJHQiI6dHJ1ZSwiMjRHQiI6dHJ1ZSwiMzYwR0IiOnRydWUsIjQ4R0IiOnRydWUsIjcyMEdCIjp0cnVlLCI5NkdCIjp0cnVlLCJkZXYiOnRydWUsIjcwOEdCIjp0cnVlLCJjNW4uNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJBbmFseXRpY3MtMjU2R0IgKDY0IHZDb3JlcywgMjU2IEdCKSI6dHJ1ZSwiYzUuNHhsYXJnZSwgNTAwZ2IgZ3AyIjp0cnVlLCJjNmEuNHhsYXJnZSwgMTUwMGdiIGdwMiI6dHJ1ZSwiY2xvdWQiOnRydWUsImRjMi44eGxhcmdlIjp0cnVlLCJyYTMuMTZ4bGFyZ2UiOnRydWUsInJhMy40eGxhcmdlIjp0cnVlLCJyYTMueGxwbHVzIjp0cnVlLCJTMiI6dHJ1ZSwiUzI0Ijp0cnVlLCIyWEwiOnRydWUsIjNYTCI6dHJ1ZSwiNFhMIjp0cnVlLCJYTCI6dHJ1ZSwiTDEgLSAxNkNQVSAzMkdCIjp0cnVlLCJjNmEuNHhsYXJnZSwgNTAwZ2IgZ3AzIjp0cnVlfSwiY2x1c3Rlcl9zaXplIjp7IjEiOnRydWUsIjIiOnRydWUsIjQiOnRydWUsIjgiOnRydWUsIjE2Ijp0cnVlLCIzMiI6dHJ1ZSwiNjQiOnRydWUsIjEyOCI6dHJ1ZSwic2VydmVybGVzcyI6dHJ1ZX0sIm1ldHJpYyI6ImhvdCIsInF1ZXJpZXMiOlt0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlLHRydWUsdHJ1ZSx0cnVlXX0=)


<div align="center">
    <img src="https://github.com/chdb-io/chdb/raw/main/docs/_static/df_bench.png" width="800">
</div>
---

## 文档
- chDB 相关示例和文档请参考 [chDB 文档](https://clickhouse.com/docs/en/chdb)
- SQL 语法请参考 [ClickHouse SQL 参考](https://clickhouse.com/docs/en/sql-reference/syntax)
- Pandas 风格的 DataStore API 请参考 [chDB](https://github.com/chdb-io/chdb)

---

## AI 编程助手技能

chDB 提供了 [AI Skill](agent/skills/using-chdb/)，可以教会 AI 编程助手（Cursor、Claude Code 等）使用 chDB 的多源数据分析 API。安装后，你的 AI 助手就能开箱即用地编写正确的 chDB 代码：

```bash
curl -sL https://raw.githubusercontent.com/chdb-io/chdb/main/install_skill.sh | bash
```

---

## 活动

- 在 [ClickHouse v23.7 livehouse!](https://t.co/todc13Kn19) 演示 chDB 以及 [幻灯片](https://docs.google.com/presentation/d/1ikqjOlimRa7QAg588TAB_Fna-Tad2WMg7_4AgnbQbFA/edit?usp=sharing)

---

## 贡献

贡献使得开源社区成为一个令人惊叹的学习、启发和创造的地方。我们非常感谢你的任何贡献。

- [ ] 帮助测试和报告 Bug
- [ ] 帮助改善文档
- [ ] 帮助提升代码质量和性能

### 语言绑定

我们欢迎其他语言的绑定，详情请参考 [bindings](bindings.md)。

---

## 版本指南

详情请参考 [VERSION-GUIDE.md](VERSION-GUIDE.md)。

---

## 论文

- [ClickHouse - Lightning Fast Analytics for Everyone](https://www.vldb.org/pvldb/vol17/p3731-schulze.pdf)

---

## 许可证

Apache 2.0，详情见 [LICENSE](LICENSE.txt)。

---

## 致谢

chDB 主要基于 [ClickHouse](https://github.com/ClickHouse/ClickHouse) [^1]
出于商标等原因，我将其命名为 chDB。

---

## 联系方式

- Discord: [https://discord.gg/D2Daa2fM5K](https://discord.gg/D2Daa2fM5K)
- Email: auxten@clickhouse.com
- Twitter: [@chdb](https://twitter.com/chdb_io)

<br>

[^1]: ClickHouse® 是 ClickHouse Inc. 的商标。所有提及或展示的商标、服务标志和徽标均为其各自所有者的财产。使用任何第三方商标、品牌名称、产品名称和公司名称并不意味着认可、关联或与各自所有者的联系。
