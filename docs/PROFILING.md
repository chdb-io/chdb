# DataStore Profiling Guide

DataStore includes built-in profiling capabilities to help you analyze and optimize query execution performance.

## Overview

The profiling system tracks execution timing across various operations:
- SQL query execution
- Cache operations (check, hit, miss)
- DataFrame operations
- Expression evaluation

## Quick Start

```python
from datastore import DataStore, enable_profiling, disable_profiling, get_profiler

# Enable profiling before your operations
enable_profiling()

# Perform DataStore operations
ds = DataStore.from_file("data.csv")
result = (ds
    .filter(ds.age > 25)
    .groupby("department")
    .agg({"salary": "mean"})
    .to_df())

# Get the profiler and view results
profiler = get_profiler()
profiler.report()

# Disable profiling when done
disable_profiling()
```

## API Reference

### Global Functions

#### `enable_profiling()`
Enable profiling for all subsequent DataStore operations.

```python
from datastore import enable_profiling
enable_profiling()
```

#### `disable_profiling()`
Disable profiling and stop recording.

```python
from datastore import disable_profiling
disable_profiling()
```

#### `is_profiling_enabled()`
Check if profiling is currently enabled.

```python
from datastore import is_profiling_enabled
if is_profiling_enabled():
    print("Profiling is active")
```

#### `get_profiler()`
Get the current profiler instance. Returns a no-op profiler if profiling is disabled.

```python
from datastore import get_profiler
profiler = get_profiler()
```

### Profiler Class

#### `profiler.report()`
Print a detailed timing report to stdout.

```python
profiler = get_profiler()
profiler.report()
```

Example output:
```
=== DataStore Profiling Report ===
Total execution time: 1.234s

Operation Breakdown:
  SQL Execution:     0.856s (69.4%)
  Cache Check:       0.012s (1.0%)
  DataFrame Ops:     0.366s (29.6%)

Step Details:
  [1] filter(age > 25)           0.002s
  [2] groupby(department)        0.001s
  [3] SQL: SELECT ...            0.856s
  [4] agg({'salary': 'mean'})    0.375s
```

#### `profiler.get_total_time()`
Get total execution time in seconds.

```python
total = profiler.get_total_time()
print(f"Total time: {total:.3f}s")
```

#### `profiler.get_steps()`
Get list of all recorded profiling steps.

```python
steps = profiler.get_steps()
for step in steps:
    print(f"{step.name}: {step.duration:.3f}s")
```

#### `profiler.clear()`
Clear all recorded profiling data.

```python
profiler.clear()
```

## Configuration via DataStoreConfig

You can also control profiling through the configuration object:

```python
from datastore import config

# Enable profiling
config.profiling_enabled = True

# Check status
print(config.profiling_enabled)  # True

# Disable profiling
config.profiling_enabled = False
```

## Use Cases

### 1. Identifying Slow Operations

```python
enable_profiling()

ds = DataStore.from_file("large_data.parquet")
result = (ds
    .filter(ds.date >= '2024-01-01')
    .groupby(['category', 'region'])
    .agg({'amount': ['sum', 'mean', 'count']})
    .sort_values('amount_sum', ascending=False)
    .head(100)
    .to_df())

profiler = get_profiler()
profiler.report()

# Identify which operation takes the most time
steps = profiler.get_steps()
slowest = max(steps, key=lambda s: s.duration)
print(f"Slowest operation: {slowest.name} ({slowest.duration:.3f}s)")
```

### 2. Comparing Execution Strategies

```python
from datastore import enable_profiling, disable_profiling, get_profiler

def profile_strategy(name, operation):
    """Profile a strategy and return timing."""
    enable_profiling()
    result = operation()
    profiler = get_profiler()
    total = profiler.get_total_time()
    disable_profiling()
    print(f"{name}: {total:.3f}s")
    return total

# Strategy 1: Filter then aggregate
time1 = profile_strategy("Filter-first", lambda: (
    ds.filter(ds.amount > 1000)
      .groupby('category')
      .agg({'amount': 'sum'})
      .to_df()
))

# Strategy 2: Aggregate then filter
time2 = profile_strategy("Aggregate-first", lambda: (
    ds.groupby('category')
      .agg({'amount': 'sum'})
      .filter(ds.amount_sum > 10000)
      .to_df()
))

print(f"Speedup: {time2/time1:.2f}x")
```

### 3. Monitoring Cache Effectiveness

```python
enable_profiling()

ds = DataStore.from_file("data.csv")

# First access - cold cache
df1 = ds.filter(ds.age > 25).to_df()

# Second access - should hit cache
df2 = ds.filter(ds.age > 25).to_df()

profiler = get_profiler()
profiler.report()

# Look for cache hit/miss in the report
```

## Integration with explain()

Profiling works well with the `explain()` method for comprehensive analysis:

```python
ds = DataStore.from_file("data.csv")

query = (ds
    .filter(ds.age > 25)
    .add_prefix('emp_')
    .filter(ds.emp_salary > 50000))

# First, understand the execution plan
query.explain(verbose=True)

# Then, profile actual execution
enable_profiling()
result = query.to_df()
get_profiler().report()
disable_profiling()
```

## Best Practices

### 1. Profile in Development
Enable profiling during development to catch performance issues early:

```python
import os

if os.environ.get('DATASTORE_PROFILE', '').lower() == 'true':
    enable_profiling()
```

### 2. Clear Between Tests
When profiling multiple operations, clear the profiler between tests:

```python
profiler = get_profiler()
profiler.clear()
# ... new operations ...
profiler.report()
```

### 3. Disable in Production
Profiling adds overhead. Disable it in production:

```python
disable_profiling()
```

### 4. Use with Logging
Combine with logging for persistent records:

```python
import logging

enable_profiling()
# ... operations ...
profiler = get_profiler()

logging.info(f"Query completed in {profiler.get_total_time():.3f}s")
for step in profiler.get_steps():
    logging.debug(f"  {step.name}: {step.duration:.3f}s")
```

## ProfileStep Class

Each profiling step contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Operation name/description |
| `duration` | float | Time in seconds |
| `category` | str | Category (sql, cache, dataframe, etc.) |
| `metadata` | dict | Additional operation-specific data |

## See Also

- [Explain Method](EXPLAIN_METHOD.md) - Understanding execution plans
- [Pandas Compatibility](PANDAS_COMPATIBILITY.md) - Execution model documentation
- [Architecture & Design](ARCHITECTURE.md) - Core design principles
- [examples/example_profiling.py](../examples/example_profiling.py) - Example usage

