# ClickHouse SQL Functions Quick Reference

> Commonly used SQL functions available in chdb.
> For the full list, see [ClickHouse documentation](https://clickhouse.com/docs/en/sql-reference/functions).

## Table of Contents

- [Aggregate Functions](#aggregate-functions)
- [String Functions](#string-functions)
- [Date & Time Functions](#date--time-functions)
- [Type Conversion](#type-conversion)
- [Conditional Functions](#conditional-functions)
- [Array Functions](#array-functions)
- [JSON Functions](#json-functions)
- [Window Functions](#window-functions)

---

## Aggregate Functions

| Function | Description | Example |
|----------|-------------|---------|
| `count()` | Row count | `SELECT count() FROM t` |
| `count(col)` | Non-null count | `SELECT count(email) FROM users` |
| `sum(col)` | Sum | `SELECT sum(amount) FROM orders` |
| `avg(col)` | Average | `SELECT avg(salary) FROM employees` |
| `min(col)`, `max(col)` | Min/Max | `SELECT min(price), max(price) FROM products` |
| `uniqExact(col)` | Exact distinct count | `SELECT uniqExact(user_id) FROM events` |
| `uniq(col)` | Approximate distinct count (faster) | `SELECT uniq(user_id) FROM events` |
| `groupArray(col)` | Collect values into array | `SELECT dept, groupArray(name) FROM emp GROUP BY dept` |
| `quantile(level)(col)` | Quantile | `SELECT quantile(0.95)(latency) FROM requests` |
| `quantiles(0.5, 0.9, 0.99)(col)` | Multiple quantiles | `SELECT quantiles(0.5, 0.9, 0.99)(duration)` |
| `median(col)` | Median (= quantile(0.5)) | `SELECT median(age) FROM users` |
| `stddevPop(col)` | Population std dev | `SELECT stddevPop(value) FROM measurements` |
| `varPop(col)` | Population variance | `SELECT varPop(value) FROM measurements` |
| `argMax(col, val)` | Value of col at max val | `SELECT argMax(name, score) FROM students` |
| `argMin(col, val)` | Value of col at min val | `SELECT argMin(name, score) FROM students` |
| `topK(N)(col)` | Most frequent N values | `SELECT topK(10)(search_term) FROM queries` |

---

## String Functions

| Function | Description | Example |
|----------|-------------|---------|
| `lower(s)` | Lowercase | `SELECT lower('Hello')` → `'hello'` |
| `upper(s)` | Uppercase | `SELECT upper('Hello')` → `'HELLO'` |
| `trim(s)` | Remove whitespace | `SELECT trim('  hi  ')` → `'hi'` |
| `length(s)` | String length | `SELECT length('hello')` → `5` |
| `substring(s, offset, length)` | Extract substring | `SELECT substring('hello', 1, 3)` → `'hel'` |
| `concat(a, b, ...)` | Concatenate | `SELECT concat(first, ' ', last)` |
| `like(s, pattern)` | LIKE match | `WHERE like(email, '%@gmail.com')` |
| `match(s, pattern)` | Regex match | `WHERE match(url, '^https?://')` |
| `extract(s, pattern)` | Regex extract | `SELECT extract(url, '://([^/]+)')` |
| `replaceAll(s, from, to)` | Replace all occurrences | `SELECT replaceAll(text, '\n', ' ')` |
| `replaceOne(s, from, to)` | Replace first occurrence | `SELECT replaceOne(s, 'old', 'new')` |
| `splitByChar(sep, s)` | Split string to array | `SELECT splitByChar(',', 'a,b,c')` |
| `splitByString(sep, s)` | Split by substring | `SELECT splitByString('::', path)` |
| `format(template, ...)` | Format string | `SELECT format('{} - {}', name, dept)` |
| `reverse(s)` | Reverse string | `SELECT reverse('hello')` → `'olleh'` |
| `base64Encode(s)` | Base64 encode | `SELECT base64Encode('hello')` |
| `base64Decode(s)` | Base64 decode | `SELECT base64Decode(encoded)` |

---

## Date & Time Functions

| Function | Description | Example |
|----------|-------------|---------|
| `today()` | Current date | `WHERE date = today()` |
| `now()` | Current datetime | `SELECT now()` |
| `toDate(x)` | Convert to Date | `SELECT toDate('2025-01-15')` |
| `toDateTime(x)` | Convert to DateTime | `SELECT toDateTime('2025-01-15 10:30:00')` |
| `toYear(d)` | Extract year | `SELECT toYear(order_date)` |
| `toMonth(d)` | Extract month | `SELECT toMonth(order_date)` |
| `toDayOfWeek(d)` | Day of week (1=Mon) | `SELECT toDayOfWeek(date)` |
| `toDayOfYear(d)` | Day of year | `SELECT toDayOfYear(date)` |
| `toHour(dt)` | Extract hour | `SELECT toHour(timestamp)` |
| `toMinute(dt)` | Extract minute | `SELECT toMinute(timestamp)` |
| `dateDiff(unit, d1, d2)` | Date difference | `SELECT dateDiff('day', start, end)` |
| `dateAdd(unit, n, d)` | Add to date | `SELECT dateAdd('month', 1, today())` |
| `dateSub(unit, n, d)` | Subtract from date | `SELECT dateSub('day', 7, today())` |
| `formatDateTime(dt, fmt)` | Format datetime | `SELECT formatDateTime(now(), '%Y-%m-%d %H:%M')` |
| `toStartOfMonth(d)` | First day of month | `SELECT toStartOfMonth(date)` |
| `toStartOfWeek(d)` | First day of week | `SELECT toStartOfWeek(date)` |
| `toStartOfHour(dt)` | Truncate to hour | `SELECT toStartOfHour(timestamp)` |
| `toMonday(d)` | Previous Monday | `SELECT toMonday(date)` |

**Date units for dateDiff/dateAdd/dateSub:** `'second'`, `'minute'`, `'hour'`, `'day'`, `'week'`, `'month'`, `'quarter'`, `'year'`.

---

## Type Conversion

| Function | Description | Example |
|----------|-------------|---------|
| `toInt32(x)` | Convert to Int32 | `SELECT toInt32('42')` |
| `toUInt64(x)` | Convert to UInt64 | `SELECT toUInt64(id)` |
| `toFloat64(x)` | Convert to Float64 | `SELECT toFloat64('3.14')` |
| `toString(x)` | Convert to String | `SELECT toString(123)` |
| `CAST(x AS Type)` | SQL-style cast | `SELECT CAST(price AS Decimal(10,2))` |
| `toFixedString(s, n)` | Fixed-length string | `SELECT toFixedString(code, 3)` |
| `toDecimal64(x, s)` | Decimal with scale | `SELECT toDecimal64(price, 2)` |
| `parseDateTimeBestEffort(s)` | Smart datetime parse | `SELECT parseDateTimeBestEffort('Jan 15 2025')` |
| `toTypeName(x)` | Get type name | `SELECT toTypeName(column)` |

---

## Conditional Functions

| Function | Description | Example |
|----------|-------------|---------|
| `if(cond, then, else)` | Ternary | `SELECT if(age >= 18, 'adult', 'minor')` |
| `multiIf(c1,v1, c2,v2, ..., default)` | Multi-branch | `SELECT multiIf(x>100,'high', x>50,'mid', 'low')` |
| `CASE WHEN ... THEN ... END` | SQL CASE | `CASE WHEN status=1 THEN 'active' ELSE 'inactive' END` |
| `coalesce(a, b, ...)` | First non-null | `SELECT coalesce(nickname, name, 'Unknown')` |
| `nullIf(a, b)` | NULL if a=b | `SELECT nullIf(value, 0)` |
| `ifNull(x, alt)` | Replace NULL | `SELECT ifNull(email, 'no-email')` |
| `isNull(x)` | Check NULL | `WHERE isNull(deleted_at)` |
| `isNotNull(x)` | Check not NULL | `WHERE isNotNull(email)` |

---

## Array Functions

| Function | Description | Example |
|----------|-------------|---------|
| `arrayJoin(arr)` | Expand array to rows | `SELECT arrayJoin([1, 2, 3])` |
| `length(arr)` | Array length | `SELECT length(tags)` |
| `arrayMap(f, arr)` | Transform elements | `SELECT arrayMap(x -> x * 2, [1, 2, 3])` |
| `arrayFilter(f, arr)` | Filter elements | `SELECT arrayFilter(x -> x > 1, [1, 2, 3])` |
| `arrayExists(f, arr)` | Any element matches | `WHERE arrayExists(x -> x = 'admin', roles)` |
| `arrayAll(f, arr)` | All elements match | `WHERE arrayAll(x -> x > 0, scores)` |
| `arraySort(arr)` | Sort array | `SELECT arraySort([3, 1, 2])` → `[1, 2, 3]` |
| `arrayDistinct(arr)` | Unique elements | `SELECT arrayDistinct(tags)` |
| `arrayConcat(a, b)` | Merge arrays | `SELECT arrayConcat([1, 2], [3, 4])` |
| `has(arr, elem)` | Contains element | `WHERE has(tags, 'important')` |
| `indexOf(arr, elem)` | Find element index | `SELECT indexOf(arr, 'target')` |
| `arraySlice(arr, offset, length)` | Sub-array | `SELECT arraySlice(arr, 1, 3)` |

---

## JSON Functions

| Function | Description | Example |
|----------|-------------|---------|
| `JSONExtract(json, key, Type)` | Extract typed value | `SELECT JSONExtract(data, 'age', 'Int32')` |
| `JSONExtractString(json, key)` | Extract as string | `SELECT JSONExtractString(data, 'name')` |
| `JSONExtractInt(json, key)` | Extract as integer | `SELECT JSONExtractInt(data, 'count')` |
| `JSONExtractFloat(json, key)` | Extract as float | `SELECT JSONExtractFloat(data, 'price')` |
| `JSONExtractBool(json, key)` | Extract as boolean | `SELECT JSONExtractBool(data, 'active')` |
| `JSONExtractArrayRaw(json, key)` | Extract array as strings | `SELECT JSONExtractArrayRaw(data, 'tags')` |
| `simpleJSONExtractString(json, key)` | Fast string extract (flat JSON) | `SELECT simpleJSONExtractString(log, 'level')` |
| `JSONHas(json, key)` | Key exists | `WHERE JSONHas(data, 'email')` |
| `JSONLength(json, key)` | Array/object length | `SELECT JSONLength(data, 'items')` |
| `JSONType(json, key)` | Value type | `SELECT JSONType(data, 'value')` |

**Nested access:** Use path syntax: `JSONExtractString(data, 'user', 'address', 'city')`

---

## Window Functions

Window functions compute values across a set of rows related to the current row.

### Syntax

```sql
function() OVER (
    [PARTITION BY col1, col2, ...]
    [ORDER BY col1 [ASC|DESC], ...]
    [ROWS|RANGE BETWEEN ... AND ...]
)
```

### Ranking Functions

| Function | Description |
|----------|-------------|
| `row_number()` | Sequential number (no ties) |
| `rank()` | Rank with gaps for ties |
| `dense_rank()` | Rank without gaps |
| `ntile(n)` | Distribute into n buckets |

```sql
SELECT name, dept, salary,
    row_number() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn,
    rank() OVER (ORDER BY salary DESC) AS overall_rank
FROM employees
```

### Value Functions

| Function | Description |
|----------|-------------|
| `lag(col, offset, default)` | Previous row value |
| `lead(col, offset, default)` | Next row value |
| `first_value(col)` | First value in window |
| `last_value(col)` | Last value in window |

```sql
SELECT date, revenue,
    lag(revenue, 1, 0) OVER (ORDER BY date) AS prev_revenue,
    revenue - lag(revenue, 1, 0) OVER (ORDER BY date) AS daily_change
FROM daily_sales
```

### Aggregate as Window

```sql
SELECT date, revenue,
    sum(revenue) OVER (ORDER BY date) AS cumulative,
    avg(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d
FROM daily_sales
```
