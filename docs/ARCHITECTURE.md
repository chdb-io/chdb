# DataStore Architecture & Design Principles

This document describes the core architecture, design principles, and development philosophy of DataStore. It is intended for contributors and developers who want to understand the internals.

## Core Vision

> **Users write familiar pandas-style code, the backend automatically selects the optimal execution engine.**

DataStore bridges the gap between pandas' intuitive API and SQL's powerful query optimization. Users get the best of both worlds without changing their coding habits.

### Architecture Overview

![DataStore Architecture](_static/datastore_architecture.png)

The diagram above shows the four-layer architecture:
1. **User API**: pandas-like interface that returns lazy objects
2. **Lazy Operation Chain**: Operations are recorded, not executed
3. **Execution Trigger & Planning**: Natural triggers invoke QueryPlanner to create execution segments
4. **Segmented Execution**: Each segment routes to optimal engine (chDB or Pandas), with intermediate results cached

## Philosophy

### Respect Pandas Experience

We deeply respect the pandas ecosystem and the expertise data scientists have built over years. Pandas has established itself as the de facto standard for data manipulation in Python, and millions of developers are fluent in its API.

**Our goal is NOT to replace pandas**, but to provide a performance-optimized alternative that honors pandas conventions. When a user writes `df.groupby('x').mean()`, they should get the same result whether using pandas or DataStore—just faster for large datasets.

### Lazy Execution for Performance

Traditional pandas executes operations eagerly—every operation immediately processes data. DataStore takes a different approach:

```python
# Pandas: Each line executes immediately
df = pd.read_csv("huge.csv")           # Load entire file
df = df[df['age'] > 25]                # Filter (creates new DataFrame)
df = df.groupby('city')['salary']      # Prepare groupby
result = df.mean()                     # Execute groupby

# DataStore: Operations are lazy until needed
ds = DataStore.from_file("huge.csv")   # Just records the source
ds = ds.filter(ds.age > 25)            # Records filter condition
ds = ds.groupby('city')['salary']      # Records groupby
result = ds.mean()                     # Still lazy!
print(result)                          # NOW executes - as optimized SQL:
                                       # SELECT city, avg(salary) FROM file
                                       # WHERE age > 25 GROUP BY city
```

**Why this matters:**
- Cross-row operations (filter, groupby, aggregations) compile to SQL
- chDB/ClickHouse's columnar engine optimizes these operations
- Only the final result is materialized, not intermediate DataFrames

### Cache for Exploratory Analysis

Data exploration is iterative. You query, inspect, adjust, repeat. DataStore caches intermediate results:

```python
ds = DataStore.from_file("data.csv")

# First query - executes SQL
ds.filter(ds.age > 25).head()          # SQL executed, result cached

# Same filter - uses cache
ds.filter(ds.age > 25).describe()      # No SQL, uses cached result
ds.filter(ds.age > 25)['salary'].mean() # No SQL, uses cached result
```

### Pragmatic Compatibility (Not 100%)

We do NOT guarantee 100% pandas syntax compatibility. That's not our goal and would be impractical.

Instead, we take a **pragmatic approach**:

1. **Extensive Testing**: We run compatibility tests using `import datastore as pd` against real-world pandas code (including Kaggle notebooks) to identify gaps.

2. **Cover Common Patterns**: We prioritize implementing the pandas operations that appear most frequently in data analysis workflows.

3. **Minimal Migration**: Our goal is that users can migrate existing code with **minimal changes**—ideally just changing the import statement.

4. **Document Differences**: When behavior differs from pandas, we document it clearly.

```python
# Ideal migration path
- import pandas as pd
+ import datastore as pd

# Most code should work unchanged
df = pd.read_csv("data.csv")
df = df[df['value'] > 0]
result = df.groupby('category').agg({'value': 'sum'})
```

### Two API Styles

Not everyone loves pandas syntax. DataStore offers two equivalent approaches:

**Pandas-style** (familiar to pandas users):
```python
import datastore as pd

df = pd.read_csv("data.csv")
result = df[df['age'] > 25].groupby('city')['salary'].mean()
```

**Fluent SQL-style** (explicit and Pythonic):
```python
from datastore import DataStore

ds = DataStore.from_file("data.csv")
result = (ds
    .filter(ds.age > 25)
    .select('city', 'salary')
    .groupby('city')
    .agg({'salary': 'mean'})
    .orderby('salary', ascending=False)
    .to_df())
```

**Both styles should compile to the same optimized SQL** - API style must not determine the execution engine. Choose based on your preference:
- **Pandas-style**: Minimal migration, familiar syntax
- **Fluent-style**: Explicit intent, better readability for complex queries, IDE auto-completion

> **Note**: Full SQL compilation for pandas-style groupby is still in development. This is a known limitation we're actively working to resolve.

### Why chDB/ClickHouse?

ClickHouse is a modern columnar database optimized for analytical queries:

- **Columnar Storage**: Only reads columns you need
- **Vectorized Execution**: Processes data in batches
- **Compression**: Efficient storage and I/O
- **SQL Optimization**: Query planner optimizes your operations
- **100+ File Formats**: Parquet, CSV, JSON, ORC, Avro, and [80+ more](https://clickhouse.com/docs/interfaces/formats)
- **20+ Data Sources**: S3, MySQL, PostgreSQL, MongoDB, Iceberg, Delta Lake, etc.
- **334 Built-in Functions**: String, datetime, geo, URL, IP, JSON, array, and more

By compiling pandas operations to ClickHouse SQL, DataStore gives you these optimizations for free.

### Zero-Copy Data Exchange

A key capability that makes DataStore fast is chDB's **zero-copy** data exchange with pandas:

```
pandas DataFrame ←────── native ──────→ chDB (ClickHouse)
                     (zero-copy)
```

When you call `ds.to_df()`, chDB returns data directly as pandas DataFrame without copying memory. Similarly, when using `DataStore.from_df(df)`, the DataFrame is passed to chDB's `Python()` table function which can query it directly without serialization overhead.

This is critical for exploratory workflows where you frequently switch between pandas operations and SQL queries:

```python
ds = DataStore.from_file("data.csv")
df = ds.filter(ds.age > 25).to_df()    # Zero-copy: chDB → pandas
df['new_col'] = df['a'] * 2            # Pure pandas operation
ds2 = DataStore.from_df(df)            # Zero-copy: pandas → chDB
result = ds2.filter(ds2.new_col > 100).to_df()  # Back to chDB
```

### Comparison with Alternatives

| Aspect | DataStore | Polars | DuckDB |
|--------|-----------|--------|--------|
| **Goal** | pandas API + ClickHouse power | New high-perf API | SQL-first analytics |
| **Learning Curve** | Low (know pandas? you're set) | Medium (new API + LazyFrame) | Medium (SQL-first) |
| **Ecosystem** | ClickHouse (100+ formats, 20+ sources) | ~10 formats, ~5 sources | ~15 formats, ~10 sources |
| **Zero-Copy pandas** | ✅ pandas ↔ chDB (native) | ❌ (copy required) | ✅ via Arrow |
| **Lazy Execution** | ✅ Automatic | ⚠️ Manual (explicit LazyFrame) | ✅ Automatic |
| **SQL Support** | ✅ Full ClickHouse SQL | ⚠️ Limited (SQLContext) | ✅ Full |
| **Unique Strength** | pandas comfort + ClickHouse scale | Rust-based, memory efficient | Embedded SQL engine |

## Design Principles

### 1. Fully Lazy Execution Architecture

**Every method that returns DataFrame or Series should return a Lazy object.**

```python
# All these return lazy objects - no execution happens yet
ds['column']                    # → ColumnExpr (lazy)
ds['column'].str.upper()        # → ColumnExpr (lazy)
ds['column'].mean()             # → LazyAggregate (lazy)
ds['column'].head(5)            # → LazySeries (lazy)
ds.groupby('x').size()          # → LazySeries (lazy)
ds['a'] > 5                     # → ColumnExpr wrapping Condition (lazy)
```

**Why?**
- Defers execution until results are truly needed
- Preserves the ability to choose the optimal execution engine at execution time
- Enables query optimization by analyzing the full operation chain
- At execution time, the config system determines whether to use pandas or chDB `ExecutionEngine`

### 2. Natural Execution Triggers (No Explicit `_execute()`)

**Explicitly calling `_execute()` in code and tests is forbidden.**

Execution should be triggered through natural interactions:

| Trigger | Description |
|---------|-------------|
| `.values` | Access underlying numpy array |
| `.index` | Access index |
| `repr()` / `__repr__` | Display in notebook/REPL |
| `__iter__` | Iteration |
| `len()` | Get length |
| `print()` | Print output |
| `to_df()` / `to_pandas()` | Explicit conversion |

```python
# ✅ Good: Natural triggers
result = ds['age'].mean()
print(result)                           # Triggers execution via __repr__
np.testing.assert_array_equal(result.values, expected)  # Triggers via .values

# ❌ Bad: Explicit execution
result = ds['age'].mean()
result._execute()                       # FORBIDDEN
```

**Testing Convention:**
```python
# ✅ Use numpy testing with natural triggers
np.testing.assert_array_equal(result.values, expected.values)

# ❌ Avoid pandas testing that may not recognize lazy objects
pd.testing.assert_series_equal(result, expected)  # May fail with lazy objects
```

### 3. Architecture Simplicity & Elegance First

**Backward compatibility is NOT a priority. Architectural elegance IS.**

- First priority: Clean, simple, elegant architecture
- Avoid duplicate definitions (e.g., `value_counts` should have ONE implementation)
- Single responsibility for each class
- Clear naming conventions (e.g., `LazySeries` not `LazySeriesMethod`)

**Example: Unified Naming**
```python
# ✅ Good: Clear, unified naming
class LazySeries:
    """Wraps any Series method call for lazy evaluation."""
    pass

# ❌ Bad: Confusing, redundant naming
class LazySeriesMethod:  # Redundant "Method"
class LazySlice:         # Split responsibility - merged into LazySeries
```

### 4. Unified Architecture, Avoid Fragmentation

**Do not create split class hierarchies for different execution engines.**

- `ColumnExpr` uniformly wraps ALL expression types (including `Condition`)
- `LazySeries` handles all deferred Series method executions
- No separate `BoolColumnExpr` - comparisons return `ColumnExpr` wrapping `Condition`

```python
# ✅ Good: Unified approach
ds['age'] > 25                    # Returns ColumnExpr(Condition)
(ds['age'] > 25).value_counts()   # Works! ColumnExpr handles it

# ❌ Bad: Fragmented approach (rejected design)
ds['age'] > 25                    # Returns BoolColumnExpr (separate class)
# Now need to duplicate methods in BoolColumnExpr...
```

**Why unified?**
- Less code duplication
- Consistent behavior
- Easier to maintain
- Single code path to debug

### 5. Cherish Test Failures, Don't Just Fix Tests

**Collect and analyze errors during execution. Test failures are valuable signals.**

When a test fails:
1. **First**: Analyze if it's a library issue that should be fixed
2. **Ask**: Is this a feature we should implement but haven't?
3. **Never**: Modify tests just to make them pass

```python
# Test fails: ds['col'].str.contains('x', regex=True) doesn't work

# ❌ Bad response: Comment out the test or change expected behavior
# def test_contains():
#     pass  # TODO: fix later

# ✅ Good response: Analyze the root cause
# - Is regex=True not implemented? → Implement it
# - Is the SQL generation wrong? → Fix the SQL builder
# - Is this a chDB limitation? → Document it and provide workaround
```

## Core Classes

### Expression Hierarchy

```
Expression (base)
├── Field                    # Column reference: ds['column']
├── Literal                  # Constant value: 42, 'hello'
├── Function                 # SQL function: upper(x), sum(x)
│   └── AggregateFunction    # Aggregate: sum, avg, count
├── ArithmeticExpression     # Math: a + b, a * b
├── Condition                # Boolean: a > b, a == b
│   ├── BinaryCondition      # Two operands: a > b
│   ├── CompoundCondition    # Combined: (a > b) & (c < d)
│   └── UnaryCondition       # Single: IS NULL
├── DateTimePropertyExpr     # dt accessor: ds['date'].dt.year
└── DateTimeMethodExpr       # dt method: ds['date'].dt.strftime('%Y')
```

### Lazy Evaluation Classes

```
Lazy Objects
├── ColumnExpr               # Wraps Expression, provides pandas-like API
├── LazySeries               # Deferred Series method execution
├── LazyAggregate            # Deferred aggregate (mean, sum, etc.)
├── LazyCondition            # Dual SQL/pandas condition
└── LazyGroupBy              # Deferred groupby operations
```

### Accessor Classes

```
Accessors (via .str, .dt, .arr, etc.)
├── StringAccessor           # String functions
├── DateTimeAccessor         # DateTime functions
├── ArrayAccessor            # Array functions (ClickHouse-specific)
├── JsonAccessor             # JSON functions (ClickHouse-specific)
├── UrlAccessor              # URL functions (ClickHouse-specific)
├── IpAccessor               # IP functions (ClickHouse-specific)
└── GeoAccessor              # Geo functions (ClickHouse-specific)
```

## Execution Flow

```
User Code                    Lazy Building                 Execution
─────────────────────────────────────────────────────────────────────
ds['age']                    → ColumnExpr                  
  .filter(ds.age > 25)       → DataStore (SQL WHERE)      
  .str.upper()               → ColumnExpr                  
  .groupby('dept')           → LazyGroupBy                 
  .mean()                    → LazySeries                  
  .values                    ──────────────────────────────→ Execute!
                                                              ↓
                                                           Config check
                                                              ↓
                                                     ┌───────┴───────┐
                                                     │               │
                                                   chDB           pandas
                                                  (SQL)         (in-memory)
```

## Configuration System

The `config` module controls execution behavior:

```python
from datastore import config

# Set default execution engine
config.default_engine = ExecutionEngine.CHDB  # or ExecutionEngine.PANDAS

# Configure per-function engine
config.function_config.use_pandas('strftime')  # Use pandas for strftime
config.function_config.use_chdb('sum')         # Use chDB for sum

# Profiling
config.profiling_enabled = True
```

## Mixed Execution Model

DataStore supports arbitrary mixing of SQL and pandas operations:

```
SQL ops (lazy)    →    Pandas op (triggers)    →    SQL on DataFrame    →    Result
     ↓                        ↓                           ↓
 Build query            Execute SQL                Use chDB Python()
                        Cache result               table function
```

```python
result = (ds
    .filter(ds.price > 100)              # SQL (lazy)
    .add_prefix('sales_')                # Pandas (executes SQL, caches)
    .filter(ds.sales_revenue > 1000)     # SQL on cached DataFrame!
    .fillna(0)                           # Pandas on cached
    .to_df())                            # Return result
```

## Adding New Features

### Adding a New String Method

1. **Register in `function_definitions.py`:**
```python
@register_function(
    name='my_method',
    clickhouse_name='myClickHouseFunc',
    func_type=FunctionType.SCALAR,
    category=FunctionCategory.STRING,
    doc='Description of what it does.',
)
def _build_my_method(expr, arg1, alias=None):
    from .functions import Function
    from .expressions import Literal
    return Function('myClickHouseFunc', expr, Literal(arg1), alias=alias)
```

2. **Methods are auto-injected into `StringAccessor`** via the registry.

3. **If it needs execution** (returns DataFrame/changes structure), implement in `ColumnExprStringAccessor`:
```python
def my_method(self, ...):
    series = self._execute_series()
    result = series.str.my_method(...)
    return DataStore.from_df(result)
```

### Adding a New Lazy Operation

1. **Return `LazySeries` from the method:**
```python
def my_operation(self):
    return LazySeries(
        datastore=self._datastore,
        method_name='my_operation',
        method_args=(),
        method_kwargs={},
        source_expr=self._expr
    )
```

2. **Ensure `LazySeries` knows how to execute it** (usually automatic via pandas delegation).

## Testing Guidelines

### Do's
```python
# ✅ Test lazy behavior
result = ds['col'].mean()
assert isinstance(result, LazyAggregate)

# ✅ Use natural triggers for execution
np.testing.assert_array_equal(result.values, expected_values)

# ✅ Test both execution engines when relevant
with use_pandas():
    pandas_result = ds['col'].mean().values
with use_chdb():
    chdb_result = ds['col'].mean().values
np.testing.assert_allclose(pandas_result, chdb_result)
```

### Don'ts
```python
# ❌ Don't call _execute() directly
result._execute()

# ❌ Don't modify tests to pass without understanding why they fail
# (commented out test)

# ❌ Don't assume execution engine
# Tests should work with both pandas and chDB when possible
```

## Error Analysis Protocol

When encountering an error:

1. **Categorize**: Is this a user error, library bug, or missing feature?

2. **Analyze**: 
   - Check if similar operations work
   - Review the execution path
   - Identify where it diverges from expected behavior

3. **Document**: If it's a limitation, document it clearly

4. **Implement**: If it's a missing feature we should have, implement it

5. **Never**: Just suppress or work around without understanding

## File Organization

```
datastore/
├── __init__.py              # Public API exports
├── core.py                  # DataStore class, main entry point
├── column_expr.py           # ColumnExpr and related classes
├── expressions.py           # Expression base classes
├── conditions.py            # Condition classes
├── functions.py             # Function classes
├── function_definitions.py  # Function registry definitions
├── function_registry.py     # Function registration system
├── lazy_result.py           # LazySeries, LazyCondition, etc.
├── lazy_ops.py              # Lazy operation utilities
├── groupby.py               # LazyGroupBy
├── config.py                # Configuration system
├── executor.py              # Execution engine
├── expression_evaluator.py  # Expression evaluation
├── pandas_compat.py         # PandasCompatMixin
├── pandas_api.py            # Module-level pandas functions
├── connection.py            # chDB connection
└── accessors/               # Accessor classes
    ├── string.py
    ├── datetime.py
    ├── array.py
    ├── json.py
    ├── url.py
    ├── ip.py
    └── geo.py
```

## Summary

| Principle | Description |
|-----------|-------------|
| **Lazy First** | Every DataFrame/Series-returning method returns a lazy object |
| **Natural Triggers** | Execution via `.values`, `repr()`, etc. - never explicit `_execute()` |
| **Elegance > Compatibility** | Clean architecture over backward compatibility |
| **Unified Design** | Single class hierarchy, no fragmentation by engine |
| **Cherish Failures** | Analyze test failures deeply, don't just fix tests |

The goal is simple: **pandas API comfort with SQL performance, achieved through elegant lazy evaluation.**

