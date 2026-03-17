---
name: chdb-sql
description: >-
  In-process ClickHouse SQL engine for Python — run ClickHouse SQL queries
  directly on local files, remote databases, and cloud storage without a
  server. Use when the user wants to write SQL queries against Parquet/CSV/
  JSON files, use ClickHouse table functions (mysql(), s3(), postgresql(),
  iceberg(), deltaLake() etc.), build stateful analytical pipelines with
  Session, use parametrized queries, window functions, or other advanced
  ClickHouse SQL features. Also use when the user explicitly mentions
  chdb.query(), ClickHouse SQL syntax, or wants cross-source SQL joins.
  Do NOT use for pandas-style DataFrame operations — use chdb-datastore
  instead.
license: Apache-2.0
compatibility: Requires Python 3.9+, macOS or Linux. pip install chdb.
metadata:
  author: chdb-io
  version: "4.1"
  homepage: https://clickhouse.com/docs/chdb
---

# chdb SQL — ClickHouse in Your Python Process

Run ClickHouse SQL directly in Python — no server needed. Query local files, remote databases, and cloud storage with full ClickHouse SQL power.

```bash
pip install chdb
```

## Decision Tree: Pick the Right API

```
1. One-off query on files or databases → chdb.query()
2. Multi-step analysis with tables      → Session
3. DB-API 2.0 connection                → chdb.connect()
4. Pandas-style DataFrame operations    → Use chdb-datastore skill instead
```

## chdb.query() — One Line, Any Data

```python
import chdb

chdb.query("SELECT * FROM file('data.parquet', Parquet) WHERE price > 100 LIMIT 10")       # local files
chdb.query("SELECT * FROM mysql('db:3306', 'shop', 'orders', 'root', 'pass')")              # databases
chdb.query("SELECT * FROM s3('s3://bucket/data.parquet', NOSIGN) LIMIT 10")                 # cloud storage
chdb.query("SELECT * FROM deltaLake('s3://bucket/delta/table', NOSIGN) LIMIT 10")           # data lakes

# Cross-source join
chdb.query("""
    SELECT u.name, o.amount FROM mysql('db:3306', 'crm', 'users', 'root', 'pass') AS u
    JOIN file('orders.parquet', Parquet) AS o ON u.id = o.user_id ORDER BY o.amount DESC
""")

data = {"name": ["Alice", "Bob"], "score": [95, 87]}
chdb.query("SELECT * FROM Python(data) ORDER BY score DESC")                                # Python data
df = chdb.query("SELECT * FROM numbers(10)", "DataFrame")                                   # output formats
chdb.query("SELECT toDate({d:String}) + number FROM numbers({n:UInt64})",
    "DataFrame", params={"d": "2025-01-01", "n": 30})                                      # parametrized
```

Table functions → [table-functions.md](references/table-functions.md) | SQL functions → [sql-functions.md](references/sql-functions.md) | Full API → [api-reference.md](references/api-reference.md)

## Session — Stateful Analysis Pipelines

```python
from chdb import session as chs
sess = chs.Session("./analytics_db")   # persistent; Session() for in-memory

sess.query("CREATE TABLE users ENGINE=MergeTree() ORDER BY id AS SELECT * FROM mysql('db:3306','crm','users','root','pass')")
sess.query("CREATE TABLE events ENGINE=MergeTree() ORDER BY (ts,user_id) AS SELECT * FROM s3('s3://logs/events/*.parquet',NOSIGN)")
sess.query("""
    SELECT u.country, count() AS cnt, uniqExact(e.user_id) AS users
    FROM events e JOIN users u ON e.user_id = u.id
    WHERE e.ts >= today() - 7 GROUP BY u.country ORDER BY cnt DESC
""", "Pretty").show()
sess.close()
```

## Connection API (DB-API 2.0)

```python
from chdb import dbapi
conn = dbapi.connect()
cur = conn.cursor()
cur.execute("SELECT * FROM file('data.parquet', Parquet) WHERE value > 100")
print(cur.fetchall())
cur.close()
conn.close()
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: No module named 'chdb'` | `pip install chdb` |
| `DB::Exception: FILE_NOT_FOUND` | Check file path; use absolute path or verify cwd |
| `DB::Exception: Unknown table function` | Check function name spelling (e.g., `deltaLake` not `deltalake`) |
| Connection refused to remote DB | Check host:port format; ensure remote DB allows connections |
| Environment check | Run `python agent/skills/chdb-sql/scripts/verify_install.py` |

## References

- [API Reference](references/api-reference.md) — query/Session/connect signatures
- [Table Functions](references/table-functions.md) — All ClickHouse table functions
- [SQL Functions](references/sql-functions.md) — Commonly used SQL functions
- [Examples](examples/examples.md) — 9 runnable examples with expected output
- [Official Docs](https://clickhouse.com/docs/chdb)

> Note: This skill teaches how to *use* chdb SQL.
> For pandas-style operations, use the `chdb-datastore` skill.
> For contributing to chdb source code, see CLAUDE.md in the project root.
