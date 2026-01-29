# Pandas Migration Guide

Seamlessly migrate from Pandas to DataStore for 10-100x performance gains while keeping the familiar API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Unified Data Source Interface](#unified-data-source-interface)
3. [ClickHouse Extensions](#clickhouse-extensions)
4. [SQL and Method Chaining](#sql-and-method-chaining)
5. [Performance Tips](#performance-tips)
6. [FAQ](#faq)

---

## Quick Start

### Simplest Way: Change One Import

```python
#import pandas as pd
import datastore as pd  # Just change this line!

# Local file - familiar pandas API
df = pd.read_csv("employee_data.csv")

# Query and operations - pure pandas syntax
filtered = df[(df['age'] > 25) & (df['salary'] > 50000)]
grouped = filtered.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
sorted_df = grouped.sort_values('mean', ascending=False)
result = sorted_df.head(10)

print(result)  # Lazy execution triggered here!
```

### What Happens Behind the Scenes?

DataStore compiles your code into efficient SQL:

```sql
SELECT city, AVG(salary) AS mean, SUM(salary) AS sum, COUNT(salary) AS count
FROM file('employee_data.csv', 'CSVWithNames')
WHERE age > 25 AND salary > 50000
GROUP BY city ORDER BY mean DESC LIMIT 10
```

**One query, one execution** — instead of processing DataFrame step by step, a single optimized SQL is pushed down to chDB for execution.

---

## Unified Data Source Interface

`DataStore.uri()` is the unified entry point for all data sources, with automatic format and source type detection.

### Local Files

```python
from datastore import DataStore

# CSV (auto-detected)
ds = DataStore.uri("data.csv")
ds = DataStore.uri("/absolute/path/to/data.csv")
ds = DataStore.uri("file:///path/to/data.csv")

# Parquet (columnar storage, best performance)
ds = DataStore.uri("data.parquet")

# JSON
ds = DataStore.uri("data.json")

# More formats: ORC, Avro, Arrow...
ds = DataStore.uri("data.orc")
```

### Cloud Storage

```python
from datastore import DataStore

# S3 (anonymous access)
ds = DataStore.uri("s3://bucket/data.parquet?nosign=true")

# S3 (with credentials - via query parameters)
ds = DataStore.uri("s3://bucket/data.parquet?access_key_id=KEY&secret_access_key=SECRET")

# Google Cloud Storage
ds = DataStore.uri("gs://bucket/data.csv")

# Azure Blob
ds = DataStore.uri("az://container/data.parquet")

# HDFS
ds = DataStore.uri("hdfs://namenode:port/path/data.parquet")

# HTTP/HTTPS (query remote files directly)
ds = DataStore.uri("https://example.com/data.csv")
ds = DataStore.uri("https://raw.githubusercontent.com/user/repo/main/data.parquet")
```

### Databases

```python
from datastore import DataStore

# MySQL - bind to specific table
ds = DataStore.uri("mysql://user:pass@localhost:3306/mydb/users")

# PostgreSQL
ds = DataStore.uri("postgresql://user:pass@localhost:5432/mydb/products")

# SQLite
ds = DataStore.uri("sqlite:///path/to/db.db?table=users")

# ClickHouse
ds = DataStore.uri("clickhouse://host:9000/database/table")
```

**Connection-level mode** (explore databases dynamically):

```python
from datastore import DataStore

# Connect without specifying database/table
ds = DataStore.from_clickhouse(
    host="analytics.company.com:9440",
    user="analyst",
    password="secret",
    secure=True
)

# Discover metadata
ds.databases()                      # ['production', 'staging', 'ml']
ds.tables("production")             # ['users', 'orders', 'events']
ds.describe("production", "users")  # Table schema

# Execute SQL queries
ds.use("production")
result = ds.sql("SELECT * FROM users WHERE age > 25")

# Or select table for pandas-style operations
users = ds["production.users"]
result = users.filter(users['age'] > 25).head(10)
```

### Data Lakes

```python
from datastore import DataStore

# Delta Lake
ds = DataStore.uri("deltalake:///path/to/delta_table")

# Apache Iceberg
ds = DataStore.uri("iceberg://catalog/namespace/table")

# Apache Hudi
ds = DataStore.uri("hudi:///path/to/hudi_table")
```

---

## ClickHouse Extensions

DataStore provides powerful ClickHouse-specific functions through accessor patterns — capabilities that Pandas doesn't have.

### URL Parsing (`.url` accessor)

**Use case**: Analyze website access logs, extract domain in one line!

```python
from datastore import DataStore

ds = DataStore.uri("access_logs.csv")

# Extract URL components (Pandas would need regex or urlparse)
result = ds.select(
    'url',
    ds['url'].url.domain().as_('domain'),
    ds['url'].url.path().as_('path'),
    ds['url'].url.protocol().as_('protocol'),
    ds['url'].url.extract_url_parameter('utm_source').as_('utm_source'),
).to_df()

# Count visits by domain
result = (ds
    .select('url', ds['url'].url.domain().as_('domain'))
    .groupby('domain')
    .agg({'url': 'count'})
    .sort('count', ascending=False)
    .limit(20)
    .to_df()
)
```

### JSON Processing (`.json` accessor)

**Use case**: Extract fields directly from JSON string columns

```python
from datastore import DataStore

ds = DataStore.uri("api_logs.parquet")

# Extract JSON fields directly (Pandas would need json.loads + apply)
result = ds.select(
    'response',
    ds['response'].json.json_extract_string('name').as_('user_name'),
    ds['response'].json.json_extract_int('id').as_('user_id'),
    ds['response'].json.json_extract_bool('premium').as_('is_premium'),
    ds['response'].json.json_has('error').as_('has_error'),
).to_df()
```

### IP Address Processing (`.ip` accessor)

**Use case**: Analyze IP addresses in server logs

```python
from datastore import DataStore

ds = DataStore.uri("server_logs.parquet")

result = ds.select(
    'client_ip',
    ds['client_ip'].ip.to_ipv6().as_('ipv6'),
    ds['client_ip'].ip.is_ipv4().as_('is_v4'),
).to_df()
```

### String Enhancement (`.str` accessor)

```python
from datastore import DataStore

ds = DataStore.uri("products.csv")

result = ds.select(
    'name',
    ds['name'].str.upper().as_('name_upper'),
    ds['name'].str.len().as_('name_len'),
    ds['name'].str.reverse().as_('reversed'),      # ClickHouse extension
    ds['name'].str.crc32().as_('hash'),            # CRC32 hash
).to_df()
```

### DateTime Enhancement (`.dt` accessor)

```python
from datastore import DataStore

ds = DataStore.uri("events.parquet")

result = ds.select(
    'timestamp',
    ds['timestamp'].dt.year.as_('year'),
    ds['timestamp'].dt.month.as_('month'),
    ds['timestamp'].dt.quarter.as_('quarter'),
    ds['timestamp'].dt.to_start_of_month().as_('month_start'),
).to_df()
```

### Array Operations (`.arr` accessor)

```python
from datastore import DataStore

ds = DataStore.uri("articles.parquet")  # tags column is array type

result = ds.select(
    'tags',
    ds['tags'].arr.length().as_('tag_count'),
    ds['tags'].arr.has('python').as_('has_python'),
    ds['tags'].arr.element(0).as_('first_tag'),
    ds['tags'].arr.array_string_concat(', ').as_('tags_str'),
).to_df()
```

---

## SQL and Method Chaining

### Fluent SQL Style

```python
from datastore import DataStore

ds = DataStore.uri("sales.parquet")

result = (ds
    .select("product", "region", "amount")
    .filter(ds.amount > 1000)
    .filter(ds.region == "APAC")
    .groupby("product")
    .agg({"amount": ["sum", "mean", "count"]})
    .sort("sum", ascending=False)
    .limit(10)
    .to_df()
)
```

### View Generated SQL

```python
from datastore import DataStore

ds = DataStore.uri("data.csv")
query = ds.filter(ds.age > 30).groupby('city').agg({'salary': 'mean'})

print(query.to_sql())
```

Output:
```sql
SELECT city, AVG(salary) AS mean
FROM file('data.csv', 'CSVWithNames')
WHERE age > 30
GROUP BY city
```

### Execute Raw SQL

```python
from datastore import DataStore

result = DataStore().sql("""
    SELECT city, AVG(salary) as avg_salary
    FROM file('employees.csv', 'CSVWithNames')
    WHERE department = 'Engineering'
    GROUP BY city
    ORDER BY avg_salary DESC
""").to_df()
```

---

## Performance Tips

### 1. Use Parquet Instead of CSV

```python
from datastore import DataStore

# ❌ CSV is slow - full table scan
ds = DataStore.uri("data.csv")

# ✅ Parquet is 10-100x faster - columnar storage, reads only needed columns
ds = DataStore.uri("data.parquet")
```

### 2. Select Only Needed Columns

```python
from datastore import DataStore

ds = DataStore.uri("data.parquet")

# ❌ Reads all columns
result = ds.to_df()

# ✅ Reads only needed columns (column pruning)
result = ds.select("name", "age", "city").to_df()
```

### 3. Filter Early

```python
from datastore import DataStore

ds = DataStore.uri("huge_data.parquet")

# ✅ Filter conditions are automatically pushed down to file scan
result = (ds
    .filter(ds.status == 'active')
    .filter(ds.date >= '2024-01-01')
    .to_df()
)
```

### 4. Use LIMIT

```python
from datastore import DataStore

ds = DataStore.uri("data.parquet")

# Preview data
preview = ds.limit(100).to_df()
```

### 5. Leverage Lazy Execution

```python
from datastore import DataStore

ds = DataStore.uri("data.csv")

# All these operations are lazy - not executed immediately
query = (ds
    .filter(ds.age > 25)
    .groupby('city')
    .agg({'salary': 'mean'})
    .sort('mean')
)

# Only executes when result is needed (compiled into a single SQL)
result = query.to_df()  # ← triggers execution
```

---

## FAQ

### Q: How do I convert to a Pandas DataFrame?

```python
from datastore import DataStore

ds = DataStore.uri("data.parquet")

# Convert to Pandas anytime
pandas_df = ds.to_df()

# Then use Pandas-specific features
result = pandas_df.pivot_table(...)
```

### Q: What functions are supported?

DataStore implements 334 ClickHouse functions and a complete Pandas-compatible API:

- 📖 [Complete Function Reference](./FUNCTIONS.md)
- 🔧 [Pandas Compatibility Checklist](./PANDAS_COMPATIBILITY.md)

### Q: How do I handle large files?

```python
from datastore import DataStore

ds = DataStore.uri("huge_file.parquet")

# Only process rows that meet conditions - automatic streaming
result = (ds
    .filter(ds.date >= '2024-01-01')
    .groupby('category')
    .agg({'amount': 'sum'})
    .to_df()
)
```

---

## Next Steps

- 📖 [Complete API Documentation](./FUNCTIONS.md)
- 🔧 [Pandas Compatibility Checklist](./PANDAS_COMPATIBILITY.md)
- 🏗️ [Architecture Design](./ARCHITECTURE.md)
- 💡 [More Examples](../examples/)

**Questions or suggestions?** Feel free to open an Issue or PR!
