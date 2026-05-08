# chdb SQL Examples

> All examples are self-contained and runnable.
> Expected output is shown in comments.

## Table of Contents

1. [Query Any File](#1-query-any-file)
2. [Cross-Source SQL Joins](#2-cross-source-sql-joins)
3. [Session: Build Analytical Tables](#3-session-build-analytical-tables)
4. [Python Data as SQL Table](#4-python-data-as-sql-table)
5. [Parametrized Queries](#5-parametrized-queries)
6. [Window Functions](#6-window-functions)
7. [User-Defined Functions (UDF)](#7-user-defined-functions-udf)
8. [Streaming Large Results](#8-streaming-large-results)
9. [Common Errors & Fixes](#9-common-errors--fixes)

---

## 1. Query Any File

```python
import chdb

# Parquet
result = chdb.query("""
    SELECT country, count() AS cnt
    FROM file('users.parquet', Parquet)
    GROUP BY country
    ORDER BY cnt DESC
    LIMIT 10
""", "Pretty")
result.show()
# Expected: top 10 countries by user count, formatted table

# CSV
df = chdb.query("""
    SELECT * FROM file('sales.csv', CSVWithNames)
    WHERE revenue > 10000
    ORDER BY revenue DESC
""", "DataFrame")
print(df)
# Expected: pandas DataFrame with high-revenue rows

# JSON Lines
chdb.query("""
    SELECT * FROM file('events.jsonl', JSONEachRow)
    WHERE event_type = 'purchase'
""").show()

# Glob pattern — query all matching files
df = chdb.query("""
    SELECT level, count() AS cnt
    FROM file('logs/2024-*.parquet', Parquet)
    GROUP BY level
    ORDER BY cnt DESC
""", "DataFrame")
print(df)
# Expected:
#     level    cnt
# 0   INFO   45230
# 1   WARN    3210
# 2  ERROR     890
```

---

## 2. Cross-Source SQL Joins

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
df = chdb.query("""
    SELECT e.event_type, p.country, count() AS cnt
    FROM s3('s3://bucket/events.parquet', 'KEY', 'SECRET', 'Parquet') AS e
    JOIN postgresql('pg:5432', 'users', 'profiles', 'user', 'pass') AS p
        ON e.user_id = p.id
    GROUP BY e.event_type, p.country
    ORDER BY cnt DESC
""", "DataFrame")
print(df)

# ClickHouse + local CSV
chdb.query("""
    SELECT r.host, l.status_code, count() AS requests
    FROM remote('ch:9000', 'logs', 'access_log', 'default', '') AS r
    JOIN file('server_config.csv', CSVWithNames) AS l ON r.host = l.hostname
    GROUP BY r.host, l.status_code
    ORDER BY requests DESC
""").show()
```

---

## 3. Session: Build Analytical Tables

```python
from chdb import session as chs

sess = chs.Session("./analytics_db")

# Ingest from multiple external sources into local tables
sess.query("""
    CREATE TABLE users ENGINE = MergeTree() ORDER BY id AS
    SELECT * FROM mysql('db:3306', 'crm', 'users', 'root', 'pass')
""")

sess.query("""
    CREATE TABLE events ENGINE = MergeTree() ORDER BY (ts, user_id) AS
    SELECT * FROM s3('s3://logs/events/*.parquet', NOSIGN)
""")

# Analyze locally — fast iterative queries
result = sess.query("""
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
""", "Pretty")
result.show()
# Expected: formatted table with country, event_type, count, unique users

# Check table contents
sess.query("SELECT count() FROM users").show()
sess.query("SELECT count() FROM events").show()

sess.close()
```

---

## 4. Python Data as SQL Table

```python
import chdb
import pandas as pd

# Query a Python dict directly in SQL
scores = {"student": ["Alice", "Bob", "Carol"], "math": [95, 87, 92], "science": [88, 91, 85]}
chdb.query("SELECT student, math + science AS total FROM Python(scores) ORDER BY total DESC").show()
# Expected:
# Alice,183
# Carol,177
# Bob,178

# Query a pandas DataFrame in SQL
users_df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
chdb.query("""
    SELECT p.name, o.product, o.amount
    FROM Python(users_df) AS p
    JOIN file('orders.parquet', Parquet) AS o ON p.id = o.user_id
    ORDER BY o.amount DESC
""").show()

# Use Python data for parametrized lookups
allowed_ids = {"id": [1, 3, 5, 7, 9]}
df = chdb.query("""
    SELECT * FROM file('data.parquet', Parquet)
    WHERE id IN (SELECT id FROM Python(allowed_ids))
""", "DataFrame")
print(df)
```

---

## 5. Parametrized Queries

```python
import chdb

# Date range generation
result = chdb.query(
    """
    SELECT
        toDate({start:String}) + number AS date,
        rand() % 1000 AS value
    FROM numbers({days:UInt64})
    """,
    "DataFrame",
    params={"start": "2025-01-01", "days": 30})
print(result)
# Expected: DataFrame with 30 rows, date column from 2025-01-01 to 2025-01-30

# Filtering with parameters
result = chdb.query(
    """
    SELECT * FROM file('events.parquet', Parquet)
    WHERE event_type = {event:String}
      AND created_at >= {since:String}
    ORDER BY created_at DESC
    LIMIT {limit:UInt64}
    """,
    "DataFrame",
    params={"event": "purchase", "since": "2025-01-01", "limit": 100})
print(result)
```

---

## 6. Window Functions

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
# Expected: employees ranked within each department

# Running totals and moving averages
df = chdb.query("""
    SELECT
        date,
        revenue,
        sum(revenue) OVER (ORDER BY date) AS cumulative_revenue,
        avg(revenue) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg
    FROM file('daily_sales.csv', CSVWithNames)
    ORDER BY date
""", "DataFrame")
print(df)
# Expected: daily sales with cumulative and 7-day rolling average

# Top-N per group
df = chdb.query("""
    SELECT * FROM (
        SELECT
            category,
            product,
            sales,
            row_number() OVER (PARTITION BY category ORDER BY sales DESC) AS rn
        FROM file('products.parquet', Parquet)
    ) WHERE rn <= 3
    ORDER BY category, rn
""", "DataFrame")
print(df)
# Expected: top 3 products per category by sales
```

---

## 7. User-Defined Functions (UDF)

```python
from chdb.udf import chdb_udf
import chdb

@chdb_udf()
def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0 / 9.0

result = chdb.query("""
    SELECT
        city,
        temp_f,
        fahrenheit_to_celsius(temp_f) AS temp_c
    FROM file('weather.csv', CSVWithNames)
    ORDER BY temp_c DESC
    LIMIT 10
""", "DataFrame")
print(result)

@chdb_udf()
def classify_age(age):
    if age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"

chdb.query("""
    SELECT classify_age(age) AS group, count() AS cnt
    FROM file('users.parquet', Parquet)
    GROUP BY group
    ORDER BY cnt DESC
""", "Pretty").show()
```

---

## 8. Streaming Large Results

```python
from chdb import session as chs

sess = chs.Session()

# Stream results in chunks for memory efficiency
iterator = sess.send_query(
    "SELECT * FROM numbers(10000000)",
    format="CSV")

row_count = 0
for chunk in iterator:
    row_count += chunk.count(b'\n')

print(f"Total rows streamed: {row_count}")
# Expected: Total rows streamed: 10000000

sess.close()
```

---

## 9. Common Errors & Fixes

### File not found

```python
import chdb

# Error:
chdb.query("SELECT * FROM file('missing.parquet', Parquet)")
# → DB::Exception: FILE_NOT_FOUND

# Fix: verify the file path
import os
print(os.path.exists("missing.parquet"))  # → False
# Use absolute path or check current working directory
chdb.query("SELECT * FROM file('/absolute/path/to/data.parquet', Parquet)")
```

### Wrong table function name

```python
# Error: function name is case-sensitive for data lake functions
chdb.query("SELECT * FROM deltalake('s3://bucket/table', NOSIGN)")
# → DB::Exception: Unknown table function deltalake

# Fix: use camelCase
chdb.query("SELECT * FROM deltaLake('s3://bucket/table', NOSIGN)")
```

### Database connection refused

```python
# Error: missing port or wrong host format
chdb.query("SELECT * FROM mysql('db', 'shop', 'orders', 'root', 'pass')")
# → Connection refused

# Fix: include port in host string
chdb.query("SELECT * FROM mysql('db:3306', 'shop', 'orders', 'root', 'pass')")
```

### Wrong output format

```python
import chdb

# Error: format name is case-sensitive
df = chdb.query("SELECT 1", "dataframe")
# → might not return expected type

# Fix: use exact format name
df = chdb.query("SELECT 1", "DataFrame")  # capital D, capital F
```

### Debugging queries

```python
import chdb

# Use Pretty format to quickly inspect results
chdb.query("SELECT * FROM file('data.parquet', Parquet) LIMIT 5", "Pretty").show()

# Check column types
chdb.query("""
    SELECT name, toTypeName(name) AS name_type, toTypeName(value) AS value_type
    FROM file('data.parquet', Parquet)
    LIMIT 1
""", "Pretty").show()

# Explain query execution plan
chdb.query("EXPLAIN SELECT * FROM file('data.parquet', Parquet) WHERE x > 100").show()
```
