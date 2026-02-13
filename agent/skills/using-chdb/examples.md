# chdb Multi-Source Data Analysis Examples

## 1. Query Any File with One Line

```python
import chdb

# Parquet
chdb.query("SELECT country, count() AS cnt FROM file('users.parquet', Parquet) GROUP BY country ORDER BY cnt DESC LIMIT 10", "Pretty").show()

# CSV
chdb.query("SELECT * FROM file('sales.csv', CSVWithNames) WHERE revenue > 10000 ORDER BY revenue DESC", "DataFrame")

# JSON
chdb.query("SELECT * FROM file('events.jsonl', JSONEachRow) WHERE event_type = 'purchase'")

# Glob patterns — query all files at once
chdb.query("SELECT * FROM file('logs/2024-*.parquet', Parquet) WHERE level = 'ERROR'")
```

## 2. DataStore: Pandas Workflow on Any Source

```python
from datastore import DataStore

# Local Parquet — feels like pandas
ds = DataStore.from_file("sales.parquet")
top_products = (ds[ds['revenue'] > 0]
    .groupby('product')
    .agg({'revenue': 'sum', 'quantity': 'sum'})
    .sort_values('revenue', ascending=False)
    .head(10))
print(top_products)

# MySQL — same pandas syntax, backed by SQL
ds = DataStore.from_mysql(
    host="db.example.com:3306", database="ecommerce",
    table="orders", user="analyst", password="pass"
)
monthly = ds.groupby(ds['order_date'].dt.month)['amount'].sum()
print(monthly)

# S3 Parquet — same API, cloud data
ds = DataStore.from_s3("s3://data-lake/clickstream/*.parquet", nosign=True)
active_users = ds[ds['event'] == 'login'].groupby('user_id')['ts'].count()
print(active_users.sort_values(ascending=False).head(20))
```

## 3. Cross-Source Join: MySQL + Local Parquet

```python
from datastore import DataStore

# Customer data in MySQL
customers = DataStore.from_mysql(
    host="db:3306", database="crm", table="customers",
    user="reader", password="pass"
)

# Order data in local Parquet
orders = DataStore.from_file("orders.parquet")

# Join and analyze — chdb handles the cross-source query
result = (customers
    .join(orders, left_on="id", right_on="customer_id", how="inner")
    .groupby("country")
    .agg({"amount": ["sum", "mean"], "order_id": "count"})
    .sort_values("sum", ascending=False)
)
print(result)

# View the generated SQL
print(result.to_sql())
```

## 4. Cross-Source Join: S3 + PostgreSQL

```python
from datastore import DataStore

# Event logs on S3
events = DataStore.from_s3(
    "s3://analytics/events/2024-*.parquet",
    access_key_id="AKIA...", secret_access_key="secret..."
)

# User profiles in PostgreSQL
profiles = DataStore.from_postgresql(
    host="pg.example.com:5432", database="users",
    table="profiles", user="analyst", password="pass"
)

# Combine cloud events with DB profiles
result = (events
    .join(profiles, left_on="user_id", right_on="id")
    .filter(events['event_type'] == 'purchase')
    .groupby(["country", "age_group"])
    .agg({"amount": "sum", "event_id": "count"})
    .sort_values("sum", ascending=False)
)
print(result)
```

## 5. Three-Way Join: File + Database + Cloud

```python
from datastore import DataStore

products = DataStore.from_file("products.csv")
orders = DataStore.from_mysql(
    host="db:3306", database="shop", table="orders",
    user="root", password="pass"
)
reviews = DataStore.from_s3("s3://feedback/reviews.parquet", nosign=True)

# Join all three sources
result = (orders
    .join(products, left_on="product_id", right_on="id")
    .join(reviews, left_on="product_id", right_on="product_id")
    .groupby("category")
    .agg({
        "amount": "sum",
        "rating": "mean",
        "review_id": "count"
    })
    .sort_values("sum", ascending=False)
)
print(result)
```

## 6. Data Lake Formats: Iceberg, Delta Lake, Hudi

```python
from datastore import DataStore

# Apache Iceberg on S3
ds = DataStore.from_iceberg(
    "s3://warehouse/iceberg/events",
    access_key_id="KEY", secret_access_key="SECRET"
)
print(ds.head(10))

# Delta Lake
ds = DataStore.from_delta(
    "s3://warehouse/delta/transactions",
    access_key_id="KEY", secret_access_key="SECRET"
)
summary = ds.groupby("category").agg({"amount": "sum"}).sort_values("sum", ascending=False)
print(summary)

# Hudi
ds = DataStore.from_hudi("s3://warehouse/hudi/logs", access_key_id="KEY", secret_access_key="SECRET")
errors = ds[ds['level'] == 'ERROR']
print(errors.head(20))

# Raw SQL also works
import chdb
chdb.query("SELECT * FROM deltaLake('s3://public-datasets/delta/hits/', NOSIGN) LIMIT 5", "Pretty").show()
```

## 7. URI-Based Access: One-Liner for Any Source

```python
from datastore import DataStore

# Local file
ds = DataStore.uri("sales.parquet")

# S3 (public)
ds = DataStore.uri("s3://public-data/dataset.parquet?nosign=true")

# MySQL
ds = DataStore.uri("mysql://root:pass@localhost:3306/shop/orders")

# PostgreSQL
ds = DataStore.uri("postgresql://analyst:pass@pg:5432/analytics/events")

# Remote ClickHouse
ds = DataStore.uri("clickhouse://ch.example.com:9440/analytics/hits?user=reader&password=pass")

# MongoDB
ds = DataStore.uri("mongodb://user:pass@mongo:27017/logs.app_events")

# SQLite
ds = DataStore.uri("sqlite:///data/local.db?table=users")

# Data lakes
ds = DataStore.uri("iceberg://my_catalog/my_namespace/my_table")
ds = DataStore.uri("deltalake:///data/delta/events")
ds = DataStore.uri("hudi:///data/hudi/events")

# After creating from any source, same pandas API
result = ds[ds['value'] > 100].groupby('category').sum().sort_values('value', ascending=False)
print(result)
```

## 8. Raw SQL Cross-Source Joins

```python
import chdb

# MySQL + Parquet join
chdb.query("""
    SELECT u.name, u.email, o.product, o.amount
    FROM mysql('db:3306', 'crm', 'users', 'root', 'pass') AS u
    JOIN file('orders.parquet', Parquet) AS o ON u.id = o.user_id
    WHERE o.amount > 100
    ORDER BY o.amount DESC
    LIMIT 20
""", "Pretty").show()

# S3 + PostgreSQL join
chdb.query("""
    SELECT e.event_type, p.country, count() AS cnt
    FROM s3('s3://bucket/events.parquet', 'KEY', 'SECRET', 'Parquet') AS e
    JOIN postgresql('pg:5432', 'users', 'profiles', 'user', 'pass') AS p ON e.user_id = p.id
    GROUP BY e.event_type, p.country
    ORDER BY cnt DESC
""", "DataFrame")

# ClickHouse + local CSV
chdb.query("""
    SELECT r.host, l.status_code, count() AS requests
    FROM remote('ch:9000', 'logs', 'access_log', 'default', '') AS r
    JOIN file('server_config.csv', CSVWithNames) AS l ON r.host = l.hostname
    GROUP BY r.host, l.status_code
    ORDER BY requests DESC
""")
```

## 9. Cloud Storage Variants

```python
from datastore import DataStore

# AWS S3 (private)
ds = DataStore.from_s3("s3://my-bucket/data.parquet", access_key_id="AKIA...", secret_access_key="secret...")

# AWS S3 (public, no signing)
ds = DataStore.from_s3("s3://public-data/dataset.parquet", nosign=True)

# Google Cloud Storage
ds = DataStore.from_gcs("gs://my-bucket/data.parquet", hmac_key="KEY", hmac_secret="SECRET")
ds = DataStore.from_gcs("gs://public-bucket/data.parquet", nosign=True)

# Azure Blob Storage
ds = DataStore.from_azure(
    connection_string="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...",
    container="data", path="analytics/events.parquet"
)

# HDFS
ds = DataStore.from_hdfs("hdfs://namenode:9000/warehouse/events/*.parquet")
```

## 10. Session: Build Analytical Tables from Multiple Sources

```python
from chdb import session as chs

sess = chs.Session("./analytics_db")

# Ingest from multiple sources into local tables
sess.query("""
    CREATE TABLE users ENGINE = MergeTree() ORDER BY id AS
    SELECT * FROM mysql('db:3306', 'crm', 'users', 'root', 'pass')
""")

sess.query("""
    CREATE TABLE events ENGINE = MergeTree() ORDER BY (ts, user_id) AS
    SELECT * FROM s3('s3://logs/events/*.parquet', NOSIGN)
""")

# Now analyze locally — super fast
sess.query("""
    SELECT
        u.country,
        e.event_type,
        count() AS cnt,
        uniqExact(e.user_id) AS unique_users
    FROM events e
    JOIN users u ON e.user_id = u.id
    WHERE e.ts >= today() - 7
    GROUP BY u.country, e.event_type
    ORDER BY cnt DESC
    LIMIT 20
""", "Pretty").show()

sess.close()
```

## 11. Python DataFrame as SQL Table

```python
import chdb
import pandas as pd

# Query any Python dict/DataFrame directly in SQL
scores = {"student": ["Alice", "Bob", "Carol"], "math": [95, 87, 92], "science": [88, 91, 85]}
chdb.query("SELECT student, math + science AS total FROM Python(scores) ORDER BY total DESC").show()

# Join Python data with external source
users_df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
chdb.query("""
    SELECT p.name, o.product, o.amount
    FROM Python(users_df) AS p
    JOIN file('orders.parquet', Parquet) AS o ON p.id = o.user_id
    ORDER BY o.amount DESC
""").show()
```

## 12. Parametrized Queries

```python
import chdb

result = chdb.query(
    """
    SELECT
        toDate({start:String}) + number AS date,
        rand() % 1000 AS value
    FROM numbers({days:UInt64})
    """,
    "DataFrame",
    params={"start": "2025-01-01", "days": 30}
)
print(result)
```

## 13. Writing Data Across Sources

```python
from datastore import DataStore

# Read from MySQL, transform, write to Parquet
source = DataStore.from_mysql(
    host="db:3306", database="shop", table="orders",
    user="root", password="pass"
)
target = DataStore("file", path="output/orders_summary.parquet", format="Parquet")

target.insert_into("category", "total_revenue", "order_count").select_from(
    source
        .groupby("category")
        .select("category", "sum(amount) AS total_revenue", "count() AS order_count")
        .filter(source['amount'] > 0)
).execute()

# Read from S3, filter, write to local file
source = DataStore.from_s3("s3://logs/events.parquet", nosign=True)
target = DataStore("file", path="filtered_events.parquet", format="Parquet")

target.insert_into("user_id", "event_type", "ts").select_from(
    source.select("user_id", "event_type", "ts").filter(source['event_type'] == 'error')
).execute()
```

## 14. Window Functions & Advanced Analytics

```python
import chdb

# Ranking within groups
chdb.query("""
    SELECT
        department,
        name,
        salary,
        rank() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,
        salary - avg(salary) OVER (PARTITION BY department) AS diff_from_avg
    FROM file('employees.parquet', Parquet)
    ORDER BY department, dept_rank
""", "Pretty").show()

# Running totals
chdb.query("""
    SELECT
        date,
        revenue,
        sum(revenue) OVER (ORDER BY date) AS cumulative_revenue,
        avg(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_avg
    FROM file('daily_sales.csv', CSVWithNames)
    ORDER BY date
""", "DataFrame")
```

## 15. Exploring Remote Database Schema

```python
from datastore import DataStore

# Connect and browse MySQL schema
mysql_ds = DataStore.from_mysql(
    host="db:3306", database="ecommerce",
    user="analyst", password="pass"
)

# Discover available tables and columns
print(mysql_ds.databases())
print(mysql_ds.tables("ecommerce"))

# Quick preview of a table
orders = DataStore.from_mysql(host="db:3306", database="ecommerce", table="orders", user="analyst", password="pass")
print(orders.columns)
print(orders.describe())
print(orders.head(5))
```
