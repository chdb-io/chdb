# DataStore explain() Method

The `explain()` method provides visibility into how DataStore will execute a chain of mixed SQL and Pandas operations, without actually executing or modifying any data.

## Overview

When you chain SQL operations (like `select()`, `filter()`) with Pandas operations (like `add_prefix()`, `rename()`), DataStore intelligently decides when to execute SQL queries and when to operate on cached DataFrames. The `explain()` method shows you this execution plan.

## Features

- ✅ **Non-Destructive**: Never executes queries or modifies data
- 📊 **Information Dense**: Shows operation type, sequence, and execution points
- 🔍 **Detailed**: Includes SQL queries, DataFrame shapes, and operation metadata
- 🎯 **Clear Phases**: Separates lazy SQL, execution, and DataFrame operations

## Usage

```python
from datastore import DataStore

ds = DataStore.from_file("data.csv")

result = (
    ds.select('*')
    .filter(ds.age > 25)
    .add_prefix('p1_')
    .filter(ds.p1_salary > 55000)
)

# Show execution plan
result.explain()

# Show detailed execution plan with metadata
result.explain(verbose=True)
```

## Output Format

The `explain()` method outputs three phases:

### Phase 1: SQL Query Building (Lazy)

Shows SQL operations that build up a query without executing:

```
Phase 1: SQL Query Building (Lazy)
────────────────────────────────────────────────────────────────────────────────
 [1] 🔍 SQL: SELECT *
 [2] 🔍 SQL: WHERE "age" > 25
```

**Characteristics:**
- No data movement
- Lightweight operation
- Best place for filtering and aggregation
- Operations are combined into a single SQL query

### Phase 2: Execution Point

Shows the operation that triggers SQL execution:

```
Phase 2: Execution Point
────────────────────────────────────────────────────────────────────────────────
 [3] 🔄 add_prefix('p1_')
     └─> Executes SQL query and caches result as DataFrame
```

**What Happens:**
1. All Phase 1 SQL operations are combined into a single query
2. The query is executed against the data source
3. Results are cached as a pandas DataFrame in memory
4. The Pandas operation is applied to the cached DataFrame

**Common Triggers:**
- Column manipulation: `add_prefix()`, `add_suffix()`, `rename()`
- Data transformation: `fillna()`, `replace()`, `drop()`
- Statistical operations: `describe()`, `corr()`
- Any pandas-specific operation

### Phase 3: Operations on Executed DataFrame

Shows operations that work on the cached DataFrame:

```
Phase 3: Operations on Executed DataFrame
────────────────────────────────────────────────────────────────────────────────
 [4] 🔍 SQL on DataFrame: WHERE "p1_salary" > 55000
 [5] 🐼 PANDAS: rename(columns={'p1_id': 'final_id'})
 [6] 🔍 SQL on DataFrame: WHERE "final_id" > 2
```

**Characteristics:**
- All operations work on in-memory DataFrame
- SQL operations use chDB to query the DataFrame
- Pandas operations work directly on DataFrame
- No database queries are executed

## Verbose Mode

Use `explain(verbose=True)` to see additional details:

```
Phase 1: SQL Query Building (Lazy)
────────────────────────────────────────────────────────────────────────────────
 [1] 🔍 SQL: SELECT *
     └─ lazy: True
 [2] 🔍 SQL: WHERE "age" > 25
     └─ lazy: True

Phase 2: Execution Point
────────────────────────────────────────────────────────────────────────────────
 [3] 🔄 add_prefix('p1_')
     └─> Executes SQL query and caches result as DataFrame
         • shape: (100, 5)
         • triggers_execution: True

Phase 3: Operations on Executed DataFrame
────────────────────────────────────────────────────────────────────────────────
 [4] 🔍 SQL on DataFrame: WHERE "p1_salary" > 55000
     └─ on_dataframe: True
 [5] 🐼 PANDAS: rename(columns={'p1_id': 'final_id'})
     └─ shape: (45, 5)
     └─ on_cached_df: True
```

## Interpreting the Output

### Icons

- 🔍 **SQL Operation**: SQL query building or execution
- 🔄 **Execution**: Operation that triggers query execution
- 🐼 **Pandas Operation**: Pure pandas DataFrame operation
- ✅ **Final State**: Shows whether result is cached or needs execution
- 📊 **SQL Query**: Shows pending SQL query to be executed

### Final State

The output ends with the final state:

```
────────────────────────────────────────────────────────────────────────────────
Final State: ✅ Executed DataFrame (cached)
             └─> No database query will be executed
             └─> Shape: (45, 5)
```

Or for unexecuted queries:

```
────────────────────────────────────────────────────────────────────────────────
Final State: 📊 SQL Query (lazy, not yet executed)
             └─> Will execute when .execute() or .to_df() is called

────────────────────────────────────────────────────────────────────────────────
Generated SQL Query:
────────────────────────────────────────────────────────────────────────────────

SELECT * FROM file('data.csv') WHERE "age" > 25 AND "salary" > 60000
```

## Best Practices

### 1. Filter Early (Phase 1)

✅ **Good**: Filter before execution
```python
result = (
    ds.filter(ds.age > 30)      # Phase 1: Filter 1000 rows → 100 rows
    .add_prefix('emp_')         # Phase 2: Execute 100 rows
)
```

❌ **Bad**: Filter after execution
```python
result = (
    ds.add_prefix('emp_')       # Phase 2: Execute 1000 rows
    .filter(ds.emp_age > 30)    # Phase 3: Filter DataFrame
)
```

### 2. Use explain() to Verify Performance

Always check your execution plan before running on large datasets:

```python
# Check the plan
result.explain()

# If satisfied, execute
df = result.to_df()
```

### 3. Understand Execution Triggers

These operations trigger execution:
- Column operations: `add_prefix()`, `add_suffix()`, `rename()`
- Reshaping: `pivot()`, `melt()`, `stack()`, `unstack()`
- Data cleaning: `fillna()`, `dropna()`, `replace()`
- Statistical: `describe()`, `corr()`, `cov()`

These do NOT trigger execution (Phase 1):
- `select()`, `filter()`, `groupby()`, `orderby()`, `limit()`
- Aggregate functions: `Count()`, `Sum()`, `Avg()`
- Joins and subqueries

### 4. Mix Operations Strategically

Combine SQL efficiency with pandas flexibility:

```python
result = (
    # Phase 1: Efficient SQL filtering
    ds.select('*')
    .filter(ds.date >= '2024-01-01')
    .filter(ds.status == 'active')
    
    # Phase 2: Pandas for complex transformations
    .assign(age_group=lambda x: pd.cut(x['age'], bins=[0, 30, 50, 100]))
    
    # Phase 3: More SQL filtering on transformed data
    .filter(ds.age_group == 'Mid')
)

result.explain()  # Verify the execution plan
```

## Examples

See `examples/example_explain.py` for comprehensive examples including:
- Pure SQL queries
- Mixed SQL and Pandas operations
- Pandas-first operations
- Performance comparisons
- Verbose mode usage

## Technical Details

### Operation Tracking

DataStore tracks operations internally to build the execution plan:
- Each method call records its type, description, and metadata
- The execution point is detected when pandas operations are called
- The execution plan is built from the operation history

### Non-Execution Guarantee

The `explain()` method:
- Never calls `execute()` or `to_df()` on unexecuted queries
- Does not modify any data
- Does not create side effects
- Safe to call multiple times

### Thread Safety

Operation tracking is instance-specific:
- Each DataStore instance has its own operation history
- Copying a DataStore preserves operation history
- Safe for concurrent use across threads

## See Also

- [Pandas Compatibility](PANDAS_COMPATIBILITY.md) - Execution model and pandas API coverage
- [Factory Methods](FACTORY_METHODS.md) - Creating DataStore from various sources
- [Profiling Guide](PROFILING.md) - Performance analysis and profiling
- [Function Reference](FUNCTIONS.md) - Complete list of available functions
- [Architecture & Design](ARCHITECTURE.md) - Core design principles and lazy execution model

