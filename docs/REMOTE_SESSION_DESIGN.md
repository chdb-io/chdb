# chdb DataStore Remote Connection Design

## Overview

Extend DataStore to support remote database connections with metadata discovery. No new classes or APIs - just make `database` and `table` parameters optional with smart error messages.

---

## User API

### 1. Create Connection-Level DataStore

```python
from datastore import DataStore

# ClickHouse (without specifying database/table)
ds = DataStore(
    source="clickhouse",
    host="analytics.company.com:9440",
    user="analyst",
    password="secret",
    secure=True
)

# MySQL
ds = DataStore(
    source="mysql",
    host="mysql.company.com:3306",
    user="root",
    password="secret"
)

# PostgreSQL
ds = DataStore(
    source="postgresql",
    host="pg.company.com:5432",
    user="postgres",
    password="secret"
)
```

**Using `from_xxx` factory methods:**

```python
# ClickHouse - connection level (no database/table)
ds = DataStore.from_clickhouse(
    host="analytics.company.com:9440",
    user="analyst",
    password="secret",
    secure=True
)

# MySQL - connection level
ds = DataStore.from_mysql(
    host="mysql.company.com:3306",
    user="root",
    password="secret"
)

# PostgreSQL - connection level
ds = DataStore.from_postgresql(
    host="pg.company.com:5432",
    user="postgres",
    password="secret"
)

# With database/table specified (existing behavior)
ds = DataStore.from_clickhouse(
    host="analytics.company.com:9440",
    database="production",
    table="users",
    user="analyst",
    password="secret"
)
```

**Password masking in display:**
```python
print(ds)
# DataStore(source='clickhouse', host='analytics.company.com:9440', 
#           user='analyst', password='***', secure=True)
```

---

### 2. Metadata Discovery

```python
# List databases
ds.databases()
# ['default', 'production', 'staging']

# List tables
ds.tables("production")
# ['users', 'orders', 'events']

# Get table schema
ds.describe("production", "users")
#   column      | type     | nullable
#   ------------|----------|----------
#   id          | UInt64   | NO
#   name        | String   | NO
#   email       | String   | YES
```

---

### 3. Select Database/Table

**Using `use()` to set defaults (mutable):**

```python
# use() sets default context (like specifying at init time)
ds.use("production")                     # set default database
ds.use("production", "users")            # set default database and table
ds.use("myschema", "production", "users") # set schema, database, and table

# After use(), shorter names work
ds.use("production")
result = ds.sql("SELECT * FROM users")  # uses production.users
```

**Using `table()` to create new DataStore (immutable):**

```python
# table() returns a NEW DataStore, original ds is unchanged
users = ds.table("production", "users")
# Or with dot notation:
users = ds.table("production.users")

# Original ds still has no table bound
ds.columns          # Error: no table specified
users.columns       # ['id', 'name', 'email', ...] - works

# Now table-level operations work on the new DataStore
users.columns       # ['id', 'name', 'email', ...]
users.head()        # first 5 rows
users.groupby('country').agg({'revenue': 'sum'})
```

**Why `table()` instead of `ds["..."]`?**

Using `ds["..."]` for table selection creates ambiguity with existing pandas-style operations:
- `ds["column"]` - select column (pandas behavior)
- `ds[condition]` - filter rows (pandas behavior)
- `ds["db.table"]` - select table (confusing!)

The explicit `table()` method eliminates this ambiguity:
```python
# Clear and unambiguous:
users = ds.table("production.users")         # select table
active = users[users['status'] == 'active']  # filter rows (pandas-style)
names = users['name']                        # select column (pandas-style)
```

**Key difference:**

| Operation | Mutates original? | Returns |
|-----------|-------------------|---------|
| `ds.use("db")` | ✅ Yes | `self` (for chaining) |
| `ds.table("db", "table")` | ❌ No | New DataStore |

---

### 4. Execute SQL Queries

```python
# Fully qualified table names work without use()
ds = DataStore(source="clickhouse", host="...", user="...", password="...")

result = ds.sql("""
    SELECT name, email 
    FROM production.users 
    WHERE created_at > '2024-01-01'
""")

# Cross-database query
result = ds.sql("""
    SELECT u.name, o.amount
    FROM production.users u
    JOIN staging.orders o ON u.id = o.user_id
""")

# Or set default database for shorter table names
ds.use("production")

result = ds.sql("""
    SELECT name, email 
    FROM users 
    WHERE created_at > '2024-01-01'
""")

# Result is a DataStore - continue chaining
filtered = result[result['amount'] > 100]
```

**Table name resolution:**

| SQL Input | Resolution |
|-----------|------------|
| `FROM production.users` | Uses specified database (no `use()` needed) |
| `FROM users` | Uses current database from `use()` |
| `FROM file('data.csv')` | Kept as-is (table function) |

---

### 5. Smart Error Messages

When table-level operations are called without specifying a table:

```python
ds = DataStore(source="clickhouse", host="...", user="...", password="...")

ds.columns
# DataStoreError: No table specified.
# Hint: Use ds.table("database", "table") to select a table, or specify table= when creating DataStore.

ds.groupby('name')
# DataStoreError: Cannot perform groupby - no table specified.
# Hint: Use ds.table("database", "table") to select a table first.

ds.table("users")  # without database
# DataStoreError: No database specified.
# Hint: Use ds.table("mydb", "users") or call ds.use("mydb") first.
```

---

### 6. Complete Workflow Example

```python
# Cell 1: Connect
from datastore import DataStore

ds = DataStore(
    source="clickhouse",
    host="analytics.company.com:9440",
    user="analyst",
    password="***",
    secure=True
)

# Cell 2: Explore
ds.databases()           # ['production', 'staging', 'ml']
ds.tables("production")  # ['users', 'orders', 'events']
ds.describe("production", "orders")

# Cell 3: Analyze with SQL
ds.use("production")

top_customers = ds.sql("""
    SELECT 
        u.name,
        SUM(o.amount) as total_spend
    FROM users u
    JOIN orders o ON u.id = o.user_id
    GROUP BY u.name
    ORDER BY total_spend DESC
    LIMIT 100
""")

# Cell 4: Continue with pandas-style API
vip = top_customers[top_customers['total_spend'] > 10000]
vip.sort_values('total_spend', ascending=False)

# Cell 5: Or work directly with a table
users = ds.table("users")  # uses default database from use()
active = users[users['status'] == 'active']  # pandas-style filter
active.groupby('country').count()
```

---

## Parameter Reference

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | ✅ | Data source type: `clickhouse`, `mysql`, `postgresql` |
| `host` | ✅ | Server address with port (e.g., `host:9000`) |
| `user` | ✅ | Username |
| `password` | ❌ | Password (optional for some sources) |
| `database` | ❌ | Default database. Can use `use()` or `table()` instead |
| `table` | ❌ | Table name. If omitted, creates connection-level DataStore |
| `secure` | ❌ | Use secure connection (ClickHouse only) |

---

## Supported Data Sources

| Source | `databases()` | `tables()` | `describe()` | `sql()` |
|--------|---------------|------------|--------------|---------|
| ClickHouse | ✅ | ✅ | ✅ | ✅ |
| MySQL | ✅ | ✅ | ✅ | ✅ |
| PostgreSQL | ✅ | ✅ | ✅ | ✅ |

---

## Key Design Decisions

1. **No new classes** - Everything through existing DataStore
2. **No caching** - Metadata queries always fetch fresh data
3. **Password masking** - Sensitive fields hidden in `repr()` / `str()` / notebook display
4. **Lazy execution** - `sql()` returns DataStore, execution deferred until needed
5. **Smart errors** - Guide users to correct usage with helpful hints
6. **Explicit table()** - Avoid ambiguity with pandas-style `ds["column"]` and `ds[condition]`
