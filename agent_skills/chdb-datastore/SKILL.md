---
name: chdb-datastore
description: >-
  Drop-in pandas replacement with ClickHouse performance. Use
  `import chdb.datastore as pd` (or `from datastore import DataStore`)
  and write standard pandas code — same API, 10-100x faster on large
  datasets. Supports 16+ data sources (MySQL, PostgreSQL, S3, MongoDB,
  ClickHouse, Iceberg, Delta Lake, etc.) and 10+ file formats (Parquet,
  CSV, JSON, Arrow, ORC, etc.) with cross-source joins. Use this skill
  when the user wants to analyze data with pandas-style syntax, speed
  up slow pandas code, query remote databases or cloud storage as
  DataFrames, or join data across different sources — even if they
  don't explicitly mention chdb or DataStore. Do NOT use for raw SQL
  queries, ClickHouse server administration, or non-Python languages.
license: Apache-2.0
compatibility: Requires Python 3.9+, macOS or Linux. pip install chdb.
metadata:
  author: chdb-io
  version: "4.1"
  homepage: https://clickhouse.com/docs/chdb
---

# chdb DataStore — It's Just Faster Pandas

## The Key Insight

```python
# Change this:
import pandas as pd
# To this:
import chdb.datastore as pd
# Everything else stays the same.
```

DataStore is a **lazy, ClickHouse-backed pandas replacement**. Your existing pandas code works unchanged — but operations compile to optimized SQL and execute only when results are needed (e.g., `print()`, `len()`, iteration).

```bash
pip install chdb
```

## Decision Tree: Pick the Right Approach

```
1. "I have a file/database and want to analyze it with pandas"
   → DataStore.from_file() / from_mysql() / from_s3() etc.
   → See references/connectors.md

2. "I need to join data from different sources"
   → Create DataStores from each source, use .join()
   → See examples/examples.md #3-5

3. "My pandas code is too slow"
   → import chdb.datastore as pd — change one line, keep the rest

4. "I need raw SQL queries"
   → Use the chdb-sql skill instead
```

## Connect to Any Data Source — One Pattern

```python
from datastore import DataStore

# Local file (auto-detects .parquet, .csv, .json, .arrow, .orc, .avro, .tsv, .xml)
ds = DataStore.from_file("sales.parquet")

# Database
ds = DataStore.from_mysql(host="db:3306", database="shop", table="orders", user="root", password="pass")

# Cloud storage
ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)

# URI shorthand — auto-detects source type
ds = DataStore.uri("mysql://root:pass@db:3306/shop/orders")
```

All 16+ sources and URI schemes → [connectors.md](references/connectors.md)

## After Connecting — Full Pandas API

```python
result = ds[ds["age"] > 25]                                          # filter
result = ds[["name", "city"]]                                        # select columns
result = ds.sort_values("revenue", ascending=False)                  # sort
result = ds.groupby("dept")["salary"].mean()                         # groupby
result = ds.assign(margin=lambda x: x["profit"] / x["revenue"])     # computed column
ds["name"].str.upper()                                               # string accessor
ds["date"].dt.year                                                   # datetime accessor
result = ds1.join(ds2, on="id")                                      # join
result = ds.head(10)                                                 # preview
print(ds.to_sql())                                                   # see generated SQL
```

209 DataFrame methods supported. Full API → [api-reference.md](references/api-reference.md)

## Cross-Source Join — The Killer Feature

```python
from datastore import DataStore

customers = DataStore.from_mysql(host="db:3306", database="crm", table="customers", user="root", password="pass")
orders = DataStore.from_file("orders.parquet")

result = (orders
    .join(customers, left_on="customer_id", right_on="id")
    .groupby("country")
    .agg({"amount": "sum", "rating": "mean"})
    .sort_values("sum", ascending=False))
print(result)
```

More join examples → [examples.md](examples/examples.md)

## Writing Data

```python
source = DataStore.from_mysql(host="db:3306", database="shop", table="orders", user="root", password="pass")
target = DataStore("file", path="summary.parquet", format="Parquet")

target.insert_into("category", "total", "count").select_from(
    source.groupby("category").select("category", "sum(amount) AS total", "count() AS count")
).execute()
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: No module named 'chdb'` | `pip install chdb` |
| `ImportError: cannot import 'DataStore'` | Use `from datastore import DataStore` or `from chdb.datastore import DataStore` |
| Database connection timeout | Include port in host: `host="db:3306"` not `host="db"` |
| Join returns empty result | Check key types match (both int or both string); use `.to_sql()` to inspect |
| Unexpected results | Call `ds.to_sql()` to see the generated SQL and debug |
| Environment check | Run `python agent/skills/chdb-datastore/scripts/verify_install.py` |

## References

- [API Reference](references/api-reference.md) — Full DataStore method signatures
- [Connectors](references/connectors.md) — All 16+ data source connection methods
- [Examples](examples/examples.md) — 10+ runnable examples with expected output
- [Verify Install](scripts/verify_install.py) — Environment verification script
- [Official Docs](https://clickhouse.com/docs/chdb)

> Note: This skill teaches how to *use* chdb DataStore.
> For raw SQL queries, use the `chdb-sql` skill.
> For contributing to chdb source code, see CLAUDE.md in the project root.
