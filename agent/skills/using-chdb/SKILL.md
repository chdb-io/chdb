---
name: using-chdb
description: Guide for using chdb, an in-process SQL OLAP engine powered by ClickHouse. Covers pandas-compatible DataStore API, 16+ data sources (MySQL, PostgreSQL, S3, ClickHouse, MongoDB, Iceberg, Delta Lake, etc.), 10+ file formats, and cross-source joins. Use when the user mentions chdb, DataStore, or wants to query files with SQL, join data across different databases and cloud storage, run ClickHouse queries in-process, or build serverless data pipelines without ETL.
---

# chdb — Pandas-Compatible Multi-Source Data Analytics

chdb is an in-process ClickHouse engine for Python. Write familiar pandas code, query 16+ data sources and 10+ file formats, join them freely — no server, no ETL, no data movement.

```bash
pip install chdb
```

## Why chdb

- **Drop-in pandas replacement**: `import datastore as pd` — same API, ClickHouse performance
- **16+ data sources as first-class citizens**: local files, S3, GCS, Azure, HDFS, MySQL, PostgreSQL, ClickHouse, MongoDB, SQLite, Redis, Iceberg, Delta Lake, Hudi, HTTP URLs
- **10+ file formats**: Parquet, CSV, TSV, JSON, JSONLines, Arrow, ORC, Avro, XML — auto-detected by extension
- **Cross-source joins**: join a MySQL table with an S3 Parquet file and a local CSV in one expression
- **Lazy evaluation**: operations compile to optimized SQL, execute only when results are needed

## DataStore: Pandas API on Any Data Source

### Connecting to data — always the same pattern

```python
from datastore import DataStore

# Local files (format auto-detected: .parquet, .csv, .json, .arrow, .orc, .avro, .tsv, .xml)
ds = DataStore.from_file("sales.parquet")
ds = DataStore.from_file("logs/*.csv")          # glob patterns

# Cloud storage
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)
ds = DataStore.from_s3("s3://private/data.parquet", access_key_id="KEY", secret_access_key="SECRET")
ds = DataStore.from_gcs("gs://bucket/data.parquet", nosign=True)
ds = DataStore.from_azure(connection_string="...", container="data", path="events.parquet")
ds = DataStore.from_hdfs("hdfs://namenode:9000/warehouse/*.parquet")
ds = DataStore.from_url("https://example.com/data.csv")

# Databases
ds = DataStore.from_mysql(host="db:3306", database="shop", table="orders", user="root", password="pass")
ds = DataStore.from_postgresql(host="pg:5432", database="analytics", table="events", user="user", password="pass")
ds = DataStore.from_clickhouse(host="ch:9000", database="logs", table="access_log")
ds = DataStore.from_mongodb(host="mongo:27017", database="app", collection="users", user="user", password="pass")
ds = DataStore.from_sqlite("/data/local.db", "users")

# Data lake formats
ds = DataStore.from_iceberg("s3://warehouse/iceberg/events", access_key_id="KEY", secret_access_key="SECRET")
ds = DataStore.from_delta("s3://warehouse/delta/transactions", access_key_id="KEY", secret_access_key="SECRET")
ds = DataStore.from_hudi("s3://warehouse/hudi/logs", access_key_id="KEY", secret_access_key="SECRET")

# URI shorthand — auto-detect source and format
ds = DataStore.uri("s3://bucket/data.parquet?nosign=true")
ds = DataStore.uri("mysql://root:pass@db:3306/shop/orders")
ds = DataStore.uri("postgresql://user:pass@pg:5432/analytics/events")
ds = DataStore.uri("clickhouse://ch:9440/logs/access_log?user=default")
ds = DataStore.uri("mongodb://user:pass@mongo:27017/app.users")
ds = DataStore.uri("deltalake:///data/delta/events")

# In-memory from dict or DataFrame
ds = DataStore({"name": ["Alice", "Bob"], "age": [25, 30]})
```

### Once connected, always the same pandas API

No matter where the data lives, the operations are identical:

```python
# Filter
result = ds[ds["age"] > 25]
result = ds[(ds["status"] == "active") & (ds["revenue"] > 1000)]

# Select columns
result = ds[["name", "city", "revenue"]]

# Sort
result = ds.sort_values("revenue", ascending=False)

# GroupBy + aggregation
result = ds.groupby("department")["salary"].mean()
result = ds.groupby(["region", "product"]).agg({"revenue": "sum", "quantity": "mean"})

# Add computed columns
result = ds.assign(profit=ds["revenue"] - ds["cost"], margin=lambda x: x["profit"] / x["revenue"])

# String and datetime accessors
ds["name"].str.upper()
ds["email"].str.contains("@gmail")
ds["order_date"].dt.year
ds["order_date"].dt.month

# Inspection
ds.columns        # column names
ds.shape           # (rows, cols)
ds.head(10)        # first 10 rows
ds.describe()      # statistics
ds.to_sql()        # view the generated SQL behind the scenes
```

### Cross-source joins — the killer feature

Join data across completely different sources with one expression:

```python
from datastore import DataStore

# Three different sources
customers = DataStore.from_mysql(host="db:3306", database="crm", table="customers", user="root", password="pass")
orders = DataStore.from_file("orders.parquet")
reviews = DataStore.from_s3("s3://feedback/reviews.parquet", nosign=True)

# Join them all with pandas syntax
result = (orders
    .join(customers, left_on="customer_id", right_on="id")
    .join(reviews, on="product_id")
    .groupby("country")
    .agg({"amount": "sum", "rating": "mean", "review_id": "count"})
    .sort_values("amount", ascending=False)
)
print(result)
```

### Writing data across sources

```python
source = DataStore.from_mysql(host="db:3306", database="shop", table="orders", user="root", password="pass")
target = DataStore("file", path="output/summary.parquet", format="Parquet")

target.insert_into("category", "total", "count").select_from(
    source.groupby("category").select("category", "sum(amount) AS total", "count() AS count")
).execute()
```

## Raw SQL: Direct ClickHouse Power

For complex analytics or when you prefer SQL:

```python
import chdb

# Query any file
chdb.query("SELECT * FROM file('data.parquet', Parquet) WHERE price > 100 LIMIT 10")

# Query databases directly
chdb.query("SELECT * FROM mysql('db:3306', 'shop', 'orders', 'root', 'pass') WHERE status = 'shipped'")
chdb.query("SELECT * FROM postgresql('pg:5432', 'analytics', 'events', 'user', 'pass') ORDER BY ts DESC LIMIT 100")

# Cross-source SQL join
chdb.query("""
    SELECT u.name, o.product, o.amount
    FROM mysql('db:3306', 'crm', 'users', 'root', 'pass') AS u
    JOIN file('orders.parquet', Parquet) AS o ON u.id = o.user_id
    WHERE o.amount > 100
    ORDER BY o.amount DESC
""")

# Data lake formats
chdb.query("SELECT * FROM deltaLake('s3://bucket/delta/table', NOSIGN) LIMIT 10")
chdb.query("SELECT * FROM iceberg('s3://bucket/iceberg/table', 'KEY', 'SECRET') LIMIT 10")

# Python dict/DataFrame as SQL table
data = {"name": ["Alice", "Bob"], "score": [95, 87]}
chdb.query("SELECT * FROM Python(data) ORDER BY score DESC")

# Output formats: CSV (default), JSON, DataFrame, Arrow, ArrowTable, Parquet, Pretty
df = chdb.query("SELECT * FROM numbers(10)", "DataFrame")

# Parametrized queries
chdb.query(
    "SELECT toDate({d:String}) + number AS date FROM numbers({n:UInt64})",
    "DataFrame",
    params={"d": "2025-01-01", "n": 30}
)
```

## Session: Stateful Pipelines

```python
from chdb import session as chs

sess = chs.Session("./analytics_db")   # persistent; use Session() for in-memory

# Step 1: Ingest from external sources into local tables
sess.query("""
    CREATE TABLE users ENGINE = MergeTree() ORDER BY id AS
    SELECT * FROM mysql('db:3306', 'crm', 'users', 'root', 'pass')
""")

# Step 2: Verify ingestion before proceeding
user_count = sess.query("SELECT count() FROM users", "DataFrame")
print(f"Ingested {user_count.iloc[0, 0]} users")

sess.query("""
    CREATE TABLE events ENGINE = MergeTree() ORDER BY (ts, user_id) AS
    SELECT * FROM s3('s3://logs/events/*.parquet', NOSIGN)
""")
event_count = sess.query("SELECT count() FROM events", "DataFrame")
print(f"Ingested {event_count.iloc[0, 0]} events")

# Step 3: Analyze locally — fast iterative queries
sess.query("""
    SELECT u.country, e.event_type, count() AS cnt, uniqExact(e.user_id) AS users
    FROM events e JOIN users u ON e.user_id = u.id
    WHERE e.ts >= today() - 7
    GROUP BY u.country, e.event_type
    ORDER BY cnt DESC
""", "Pretty").show()

sess.close()
```

## Quick Reference

- Official docs: https://clickhouse.com/docs/chdb
- API signatures and ClickHouse SQL functions: [reference.md](reference.md)
- 15 runnable examples (cross-source joins, data lakes, cloud storage, ETL pipelines): [examples.md](examples.md)
