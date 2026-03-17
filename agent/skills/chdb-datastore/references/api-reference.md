# DataStore API Reference

> Complete method signatures for the DataStore class.
> DataStore provides a pandas-compatible API backed by ClickHouse.

## Table of Contents

- [Import & Construction](#import--construction)
- [Selection & Filtering](#selection--filtering)
- [Sorting & Limiting](#sorting--limiting)
- [GroupBy & Aggregation](#groupby--aggregation)
- [Joins](#joins)
- [Mutation](#mutation)
- [String Accessor (.str)](#string-accessor-str)
- [DateTime Accessor (.dt)](#datetime-accessor-dt)
- [Inspection & Execution Triggers](#inspection--execution-triggers)
- [Writing Data](#writing-data)
- [Configuration](#configuration)

---

## Import & Construction

```python
from datastore import DataStore
# or: from chdb.datastore import DataStore
# or: import chdb.datastore as pd  (drop-in replacement)
```

### Constructor

```python
DataStore(source=None, table=None, database=":memory:", connection=None, **kwargs)
```

| Source type | Usage |
|-------------|-------|
| dict | `DataStore({'col1': [1, 2], 'col2': ['a', 'b']})` |
| pd.DataFrame | `DataStore(df)` |
| str (source type) | `DataStore("file", path="data.parquet")` |
| str (source type) | `DataStore("mysql", host="host:3306", database="db", table="t", user="u", password="p")` |

### Factory Methods

See [connectors.md](connectors.md) for all factory methods (`from_file`, `from_mysql`, `from_s3`, `uri`, etc.).

---

## Selection & Filtering

| Expression | Returns | Description |
|------------|---------|-------------|
| `ds['col']` | LazySeries | Single column |
| `ds[['c1', 'c2']]` | DataStore | Multiple columns |
| `ds[condition]` | DataStore | Boolean filter (e.g., `ds[ds['age'] > 25]`) |
| `.select(*fields)` | DataStore | SQL-style SELECT with expressions |
| `.filter(condition)` | DataStore | SQL-style WHERE clause |
| `.where(condition)` | DataStore | Alias for `.filter()` |

```python
result = ds[ds["age"] > 25]
result = ds[(ds["status"] == "active") & (ds["revenue"] > 1000)]
result = ds[["name", "city", "revenue"]]
result = ds.select("name", "revenue * 1.1 AS adjusted_revenue")
result = ds.filter(ds["country"] == "US")
```

---

## Sorting & Limiting

| Method | Description |
|--------|-------------|
| `.sort_values(by, ascending=True)` | Pandas-style sort (by can be str or list) |
| `.sort(*columns, ascending=True)` | SQL-style ORDER BY |
| `.orderby(*columns, ascending=True)` | Alias for `.sort()` |
| `.limit(n)` | LIMIT n rows |
| `.offset(n)` | Skip first n rows |
| `.head(n=5)` | First n rows |
| `.tail(n=5)` | Last n rows |

```python
result = ds.sort_values("revenue", ascending=False)
result = ds.sort_values(["country", "city"])
result = ds.head(10)
result = ds.limit(100).offset(50)
```

---

## GroupBy & Aggregation

```python
grouped = ds.groupby(*columns)          # returns LazyGroupBy
grouped = ds.groupby("dept")
grouped = ds.groupby(["region", "product"])
```

| Method | Description |
|--------|-------------|
| `.agg(func=None, **kwargs)` | Aggregate with named functions |
| `.sum()`, `.mean()`, `.count()`, `.min()`, `.max()` | Single aggregation |
| `.std()`, `.var()` | Standard deviation / variance |
| `.having(condition)` | HAVING clause (after aggregation) |

```python
result = ds.groupby("dept")["salary"].mean()
result = ds.groupby("dept").agg({"salary": "mean", "bonus": "sum"})
result = ds.groupby(["region", "product"]).agg(
    total_revenue=("revenue", "sum"),
    avg_quantity=("quantity", "mean"))
```

---

## Joins

```python
.join(other, on=None, how='inner', left_on=None, right_on=None, suffixes=('_x', '_y'))
.merge(other, on=None, how='inner')
```

| `how` | Description |
|-------|-------------|
| `'inner'` | Only matching rows (default) |
| `'left'` | All left rows + matching right |
| `'right'` | All right rows + matching left |
| `'outer'` | All rows from both sides |
| `'cross'` | Cartesian product |

```python
result = orders.join(customers, left_on="customer_id", right_on="id")
result = orders.join(customers, on="customer_id", how="left")
result = ds1.merge(ds2, on="key", how="outer")
```

**Cross-source joins work transparently** — join a MySQL table with a Parquet file:

```python
mysql_ds = DataStore.from_mysql(host="db:3306", database="crm", table="users", user="root", password="pass")
parquet_ds = DataStore.from_file("orders.parquet")
result = mysql_ds.join(parquet_ds, left_on="id", right_on="user_id")
```

---

## Mutation

| Method | Description |
|--------|-------------|
| `.assign(**kwargs)` | Add computed columns |
| `.with_column(name, expr)` | Add single column |
| `.drop(columns)` | Remove columns (str or list) |
| `.rename(columns={})` | Rename columns via mapping |
| `.fillna(value)` | Fill NaN/NULL values |
| `.dropna(subset=None)` | Drop rows with NaN/NULL |
| `.distinct(subset=None, keep='first')` | Deduplicate rows |

```python
result = ds.assign(
    profit=ds["revenue"] - ds["cost"],
    margin=lambda x: x["profit"] / x["revenue"])
result = ds.drop("temp_column")
result = ds.rename(columns={"old_name": "new_name"})
result = ds.fillna(0)
result = ds.dropna(subset=["email", "phone"])
result = ds.distinct(subset=["user_id"], keep="first")
```

---

## String Accessor (.str)

Access via `ds['column'].str.*`. 56 methods available, including:

| Method | Description |
|--------|-------------|
| `.str.upper()`, `.str.lower()` | Case conversion |
| `.str.strip()`, `.str.lstrip()`, `.str.rstrip()` | Whitespace trimming |
| `.str.contains(pattern)` | Substring/regex match → boolean |
| `.str.startswith(prefix)`, `.str.endswith(suffix)` | Prefix/suffix check |
| `.str.replace(old, new)` | String replacement |
| `.str.split(sep)` | Split into parts |
| `.str.len()` | String length |
| `.str.slice(start, stop)` | Substring extraction |
| `.str.cat(sep=None)` | Concatenation |
| `.str.extract(pattern)` | Regex group extraction |
| `.str.pad(width)`, `.str.zfill(width)` | Padding |
| `.str.match(pattern)` | Full regex match |

```python
ds["name"].str.upper()
ds["email"].str.contains("@gmail")
ds["code"].str.slice(0, 3)
```

---

## DateTime Accessor (.dt)

Access via `ds['column'].dt.*`. 42+ methods available, including:

| Property/Method | Description |
|-----------------|-------------|
| `.dt.year`, `.dt.month`, `.dt.day` | Date components |
| `.dt.hour`, `.dt.minute`, `.dt.second` | Time components |
| `.dt.dayofweek`, `.dt.dayofyear` | Day ordinals |
| `.dt.quarter` | Quarter (1-4) |
| `.dt.date`, `.dt.time` | Date/time part |
| `.dt.strftime(format)` | Format as string |
| `.dt.floor(freq)`, `.dt.ceil(freq)` | Round to frequency |
| `.dt.tz_localize(tz)`, `.dt.tz_convert(tz)` | Timezone handling |
| `.dt.normalize()` | Reset time to midnight |

```python
ds["order_date"].dt.year
ds["order_date"].dt.month
ds["timestamp"].dt.hour
ds["created_at"].dt.strftime("%Y-%m-%d")
```

---

## Inspection & Execution Triggers

These properties/methods **trigger execution** of the lazy query:

| Property/Method | Returns | Description |
|-----------------|---------|-------------|
| `.columns` | list | Column names |
| `.shape` | (rows, cols) | Dimensions |
| `.dtypes` | dict | Column types |
| `.head(n=5)` | DataStore | First n rows |
| `.tail(n=5)` | DataStore | Last n rows |
| `.describe()` | DataStore | Summary statistics |
| `.info()` | None | Print DataFrame info |
| `print(ds)` | — | Display results |
| `len(ds)` | int | Row count |
| `for row in ds` | — | Iterate rows |
| `.equals(other)` | bool | Compare DataStores |

These methods **do not trigger execution**:

| Method | Returns | Description |
|--------|---------|-------------|
| `.to_sql()` | str | View the generated SQL |
| `.explain()` | str | Execution plan |

```python
print(ds.columns)          # → ['name', 'age', 'city']
print(ds.shape)            # → (1000, 3)
print(ds.to_sql())         # → SELECT ... FROM ... WHERE ...
print(ds.describe())       # → statistics table
```

---

## Writing Data

Use the `insert_into` / `select_from` pattern:

```python
source = DataStore.from_mysql(host="db:3306", database="shop", table="orders", user="root", password="pass")
target = DataStore("file", path="output.parquet", format="Parquet")

target.insert_into("col1", "col2").select_from(
    source.select("col1", "col2").filter(source['value'] > 100)
).execute()
```

---

## Configuration

```python
from datastore import config

config.use_chdb()           # force chDB/SQL backend
config.use_pandas()         # force pandas backend
config.prefer_chdb()        # prefer chDB when possible, fallback to pandas
config.prefer_pandas()      # prefer pandas when possible, fallback to chDB
config.enable_debug()       # verbose logging (shows generated SQL)
config.enable_profiling()   # performance profiling
```
