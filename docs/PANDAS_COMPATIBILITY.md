# Pandas DataFrame Compatibility

DataStore provides comprehensive pandas DataFrame API compatibility, allowing you to use familiar pandas methods directly on DataStore objects while maintaining the benefits of SQL-based query optimization.

## Our Approach

**We don't guarantee 100% pandas compatibility—we optimize for practical migration.**

Our compatibility strategy:

1. **Real-World Testing**: We test against actual pandas code from Kaggle notebooks and common data analysis patterns using `import datastore as pd`.

2. **Prioritize Common Operations**: We implement the pandas operations that appear most frequently in real workflows.

3. **Minimal Code Changes**: The goal is that most existing pandas code works with just an import change.

4. **Document Differences**: When behavior differs from pandas, we clearly document it.

```python
# Typical migration
- import pandas as pd
+ import datastore as pd

# Most pandas code works unchanged
df = pd.read_csv("data.csv")
result = df[df['age'] > 25].groupby('city')['salary'].mean()
```

**Alternative: Fluent SQL-style API**

If you prefer more explicit, SQL-like syntax over pandas conventions:

```python
from datastore import DataStore

ds = DataStore.from_file("data.csv")
result = (ds
    .filter(ds.age > 25)
    .select('city', 'salary')
    .groupby('city')
    .agg({'salary': 'mean'})
    .to_df())
```

Both styles produce identical results. Choose based on your preference.

## read_csv() Compatibility

DataStore provides a pandas-compatible `read_csv()` function that automatically chooses the optimal execution engine:

### Default Behavior (Matches pandas)

```python
import datastore as ds

# These work exactly like pandas
df = ds.read_csv("data.csv")                    # First row is header
df = ds.read_csv("data.csv", header=0)          # Explicit: first row is header
df = ds.read_csv("data.csv", header=None)       # No header, auto-generate column names
df = ds.read_csv("data.csv", sep=";")           # Semicolon delimiter
df = ds.read_csv("data.csv", sep="\t")          # Tab delimiter (uses TSV format)
df = ds.read_csv("data.csv", nrows=100)         # Read first 100 rows
df = ds.read_csv("data.csv", compression='gzip') # Compressed CSV
```

### Parameters Handled by chDB SQL Engine

These parameters are translated to ClickHouse settings for optimal performance:

| Parameter | ClickHouse Setting | Notes |
|-----------|-------------------|-------|
| `sep=','` | Default CSV format | Comma delimiter |
| `sep='\t'` | `TSVWithNames` format | Tab delimiter uses native TSV |
| `header=None` | `CSV` format | No header row |
| `skiprows=N` | `input_format_csv_skip_first_lines` | Skip initial rows |
| `nrows=N` | `LIMIT N` | Read first N rows |
| `compression` | File function parameter | gzip, zstd, etc. |

### Parameters That Fall Back to pandas

For full compatibility, these parameters automatically use pandas' `read_csv()`:

- **Column customization**: `names`, `usecols`, `index_col`
- **Type conversion**: `dtype`, `converters`
- **Date parsing**: `parse_dates`, `date_parser`, `date_format`
- **Custom delimiters**: Any delimiter other than `,` or `\t`
- **Complex features**: `skipfooter`, `comment`, `thousands`, `chunksize`

```python
# These automatically use pandas for full compatibility
df = ds.read_csv("data.csv", usecols=['name', 'age'])    # Column selection
df = ds.read_csv("data.csv", dtype={'age': int})        # Type conversion
df = ds.read_csv("data.csv", parse_dates=['date_col'])  # Date parsing
df = ds.read_csv("data.csv", header=None, names=['a', 'b', 'c'])  # Custom names
```

### Boolean Value Handling

By default, ClickHouse recognizes `true`/`false` (case-insensitive). For custom boolean strings:

```python
# Custom boolean values (uses pandas fallback for full compatibility)
df = ds.read_csv("data.csv", 
                 true_values=['yes', 'Yes', 'TRUE'],
                 false_values=['no', 'No', 'FALSE'])
```

### Best Practices

1. **Use standard CSV format**: Files with comma delimiters and first-row headers work best with chDB engine
2. **Prefer chDB-supported parameters**: `nrows`, `compression` for performance
3. **Fall back to pandas when needed**: Complex parsing requirements are handled automatically

## Implementation Statistics

### Pandas API Coverage

| Category | Pandas Total | Implemented | Notes |
|----------|--------------|-------------|-------|
| **DataFrame methods** | 209 | 209 | All pandas DataFrame methods |
| **Series methods** | 210 | (via delegation) | Delegated to pandas |
| **Series.str accessor** | 56 | 56 | All pandas str methods |
| **Series.dt accessor** | 42 | 42+ | All pandas + ClickHouse extras |

### ClickHouse-Specific Accessors

| Accessor | Methods | Description |
|----------|---------|-------------|
| **Series.arr accessor** | 37 | Array functions (ClickHouse-specific) |
| **Series.json accessor** | 13 | JSON functions |
| **Series.url accessor** | 15 | URL parsing functions |
| **Series.ip accessor** | 9 | IP address functions |
| **Series.geo accessor** | 14 | Geo/distance functions |

> Note: DataStore implements all pandas DataFrame API methods and all pandas Series.str/dt accessor methods. Additionally, it provides ClickHouse-specific accessors for array, JSON, URL, IP, and geo operations.

### ClickHouse Functions

| Engine | Functions |
|--------|-----------|
| ClickHouse (ch_functions.json) | 1,475 |
| **Implemented in DataStore** | **334** |
| Implementation Rate | 22.6% |

## Overview

- **209 Methods**: **100%** coverage of pandas DataFrame API
- **Seamless Integration**: Mix SQL-style queries with pandas transformations
- **Automatic Wrapping**: DataFrame/Series results automatically wrapped as DataStore
- **Immutable**: All operations return new instances (no `inplace=True`)
- **Smart Execution**: SQL operations build queries, pandas operations execute and cache results
- **Correct Chaining**: Handles mixed SQL→pandas→pandas chains correctly

## Quick Start

```python
from datastore import DataStore

ds = DataStore.from_file("data.csv")

# Use any pandas method
df = ds.drop(columns=['unused'])
      .fillna(0)
      .assign(revenue=lambda x: x['price'] * x['quantity'])
      .sort_values('revenue', ascending=False)
      .head(10)

# Mix SQL and pandas
result = (ds
    .select('*')
    .filter(ds.price > 100)              # SQL-style
    .assign(margin=lambda x: x['profit'] / x['revenue'])  # pandas-style
    .query('margin > 0.2')               # SQL-style
    .groupby('category').agg({'revenue': 'sum'}))  # pandas-style
```

## Feature Checklist

### ✅ Attributes and Properties
- [x] `df.index` - Row labels
- [x] `df.columns` - Column labels
- [x] `df.dtypes` - Data types
- [x] `df.values` - NumPy array representation
- [x] `df.shape` - Dimensions (rows, cols)
- [x] `df.size` - Total elements
- [x] `df.ndim` - Number of dimensions
- [x] `df.empty` - Empty check
- [x] `df.T` - Transpose
- [x] `df.axes` - Axis labels

### ✅ Indexing and Selection
- [x] `df.loc[...]` - Label-based indexing
- [x] `df.iloc[...]` - Integer-based indexing
- [x] `df.at[...]` - Fast scalar access
- [x] `df.iat[...]` - Fast integer scalar access
- [x] `df['col']` - Column selection
- [x] `df[['col1', 'col2']]` - Multiple columns
- [x] `df.head(n)` - First n rows
- [x] `df.tail(n)` - Last n rows
- [x] `df.sample(n)` - Random sample
- [x] `df.select_dtypes()` - Select by dtype
- [x] `df.query()` - Query by expression
- [x] `df.where()` - Conditional replacement
- [x] `df.mask()` - Inverse where
- [x] `df.isin()` - Value membership
- [x] `df.get()` - Safe column access
- [x] `df.xs()` - Cross-section
- [x] `df.pop()` - Remove and return column
- [x] `df.insert()` - Insert column

### ✅ Statistical Methods
- [x] `df.describe()` - Summary statistics
- [x] `df.mean()` - Mean values
- [x] `df.median()` - Median values
- [x] `df.mode()` - Mode values
- [x] `df.std()` - Standard deviation
- [x] `df.var()` - Variance
- [x] `df.min()` / `df.max()` - Min/Max values
- [x] `df.sum()` - Sum
- [x] `df.prod()` - Product
- [x] `df.count()` - Non-null counts
- [x] `df.nunique()` - Unique counts
- [x] `df.value_counts()` - Value frequencies
- [x] `df.quantile()` - Quantiles
- [x] `df.corr()` - Correlation matrix
- [x] `df.cov()` - Covariance matrix
- [x] `df.corrwith()` - Pairwise correlation
- [x] `df.rank()` - Rank values
- [x] `df.abs()` - Absolute values
- [x] `df.round()` - Round values
- [x] `df.clip()` - Clip values
- [x] `df.cumsum()` - Cumulative sum
- [x] `df.cumprod()` - Cumulative product
- [x] `df.cummin()` - Cumulative min
- [x] `df.cummax()` - Cumulative max
- [x] `df.diff()` - Difference
- [x] `df.pct_change()` - Percent change
- [x] `df.skew()` - Skewness
- [x] `df.kurt()` - Kurtosis
- [x] `df.sem()` - Standard error
- [x] `df.all()` / `df.any()` - Boolean aggregation
- [x] `df.idxmin()` / `df.idxmax()` - Index of min/max
- [x] `df.eval()` - Expression evaluation

### ✅ Data Manipulation
- [x] `df.drop()` - Drop rows/columns
- [x] `df.drop_duplicates()` - Remove duplicates
- [x] `df.duplicated()` - Mark duplicates
- [x] `df.dropna()` - Remove missing
- [x] `df.fillna()` - Fill missing
- [x] `df.ffill()` / `df.bfill()` - Forward/backward fill
- [x] `df.interpolate()` - Interpolate values
- [x] `df.replace()` - Replace values
- [x] `df.rename()` - Rename labels
- [x] `df.rename_axis()` - Rename axis
- [x] `df.assign()` - Add columns
- [x] `df.astype()` - Convert types
- [x] `df.convert_dtypes()` - Infer types
- [x] `df.copy()` - Copy data

### ✅ Sorting and Ranking
- [x] `df.sort_values()` - Sort by values
- [x] `df.sort_index()` - Sort by index
- [x] `df.nlargest()` - N largest values
- [x] `df.nsmallest()` - N smallest values

### ✅ Reindexing
- [x] `df.reset_index()` - Reset index
- [x] `df.set_index()` - Set index
- [x] `df.reindex()` - Conform to new index
- [x] `df.reindex_like()` - Match another's index
- [x] `df.add_prefix()` - Add prefix to labels
- [x] `df.add_suffix()` - Add suffix to labels
- [x] `df.align()` - Align two objects
- [x] `df.set_axis()` - Set axis labels
- [x] `df.take()` - Select by positions
- [x] `df.truncate()` - Truncate by range

### ✅ Reshaping
- [x] `df.pivot()` - Pivot table
- [x] `df.pivot_table()` - Pivot with aggregation
- [x] `df.melt()` - Unpivot
- [x] `df.stack()` - Stack columns to index
- [x] `df.unstack()` - Unstack index to columns
- [x] `df.transpose()` / `df.T` - Transpose
- [x] `df.explode()` - Explode lists to rows
- [x] `df.squeeze()` - Reduce dimensions
- [x] `df.droplevel()` - Drop index level
- [x] `df.swaplevel()` - Swap index levels
- [x] `df.swapaxes()` - Swap axes
- [x] `df.reorder_levels()` - Reorder levels

### ✅ Combining / Joining / Merging
- [x] `df.append()` - Append rows
- [x] `df.merge()` - SQL-style merge
- [x] `df.join()` - Join on index
- [x] `df.concat()` - Concatenate
- [x] `df.compare()` - Compare differences
- [x] `df.update()` - Update values
- [x] `df.combine()` - Combine with function
- [x] `df.combine_first()` - Combine with priority

### ✅ Binary Operators
- [x] `df.add()` / `df.radd()` - Addition
- [x] `df.sub()` / `df.rsub()` - Subtraction
- [x] `df.mul()` / `df.rmul()` - Multiplication
- [x] `df.div()` / `df.rdiv()` - Division
- [x] `df.truediv()` / `df.rtruediv()` - True division
- [x] `df.floordiv()` / `df.rfloordiv()` - Floor division
- [x] `df.mod()` / `df.rmod()` - Modulo
- [x] `df.pow()` / `df.rpow()` - Power
- [x] `df.dot()` - Matrix multiplication

### ✅ Comparison Operators
- [x] `df.eq()` - Equal
- [x] `df.ne()` - Not equal
- [x] `df.lt()` - Less than
- [x] `df.le()` - Less than or equal
- [x] `df.gt()` - Greater than
- [x] `df.ge()` - Greater than or equal

### ✅ Function Application
- [x] `df.apply()` - Apply function
- [x] `df.applymap()` - Apply element-wise
- [x] `df.map()` - Apply element-wise (alias)
- [x] `df.agg()` / `df.aggregate()` - Aggregate
- [x] `df.transform()` - Transform
- [x] `df.pipe()` - Pipe functions
- [x] `df.groupby()` - Group by (returns GroupBy)

### ✅ Time Series
- [x] `df.rolling()` - Rolling window
- [x] `df.expanding()` - Expanding window
- [x] `df.ewm()` - Exponentially weighted
- [x] `df.resample()` - Resample time series
- [x] `df.shift()` - Shift values
- [x] `df.asfreq()` - Convert frequency
- [x] `df.asof()` - Latest value as of time
- [x] `df.at_time()` - Select at time
- [x] `df.between_time()` - Select time range
- [x] `df.first()` / `df.last()` - First/last periods
- [x] `df.first_valid_index()` - First valid index
- [x] `df.last_valid_index()` - Last valid index
- [x] `df.to_period()` - Convert to period
- [x] `df.to_timestamp()` - Convert to timestamp
- [x] `df.tz_convert()` - Convert timezone
- [x] `df.tz_localize()` - Localize timezone

### ✅ Missing Data
- [x] `df.isna()` / `df.isnull()` - Detect missing
- [x] `df.notna()` / `df.notnull()` - Detect non-missing
- [x] `df.dropna()` - Drop missing
- [x] `df.fillna()` - Fill missing
- [x] `df.ffill()` - Forward fill
- [x] `df.bfill()` - Backward fill
- [x] `df.backfill()` - Backward fill (alias)
- [x] `df.pad()` - Forward fill (alias)
- [x] `df.interpolate()` - Interpolate
- [x] `df.replace()` - Replace values

### ✅ Export / IO
- [x] `df.to_csv()` - Export to CSV
- [x] `df.to_json()` - Export to JSON
- [x] `df.to_excel()` - Export to Excel
- [x] `df.to_parquet()` - Export to Parquet
- [x] `df.to_feather()` - Export to Feather
- [x] `df.to_hdf()` - Export to HDF5
- [x] `df.to_sql()` - Export to SQL database
- [x] `df.to_stata()` - Export to Stata
- [x] `df.to_pickle()` - Pickle to file
- [x] `df.to_html()` - Render as HTML
- [x] `df.to_latex()` - Render as LaTeX
- [x] `df.to_markdown()` - Render as Markdown
- [x] `df.to_string()` - Render as string
- [x] `df.to_dict()` - Convert to dictionary
- [x] `df.to_records()` - Convert to records
- [x] `df.to_numpy()` - Convert to NumPy
- [x] `df.to_clipboard()` - Copy to clipboard
- [x] `df.to_xarray()` - Convert to xarray
- [x] `df.to_orc()` - Export to ORC
- [x] `df.to_gbq()` - Export to BigQuery

### ✅ Iteration
- [x] `df.items()` - Iterate (column, Series) pairs
- [x] `df.iterrows()` - Iterate (index, Series) pairs
- [x] `df.itertuples()` - Iterate as namedtuples

### ✅ Plotting
- [x] `df.plot` - Plotting accessor
- [x] `df.plot.*` - Various plot types
- [x] `df.hist()` - Histogram
- [x] `df.boxplot()` - Box plot

### ✅ Accessors
- [x] `df.str` - String accessor (for Series)
- [x] `df.dt` - Datetime accessor
- [x] `df.sparse` - Sparse accessor
- [x] `df.style` - Styling accessor

## Series.str Accessor

The `.str` accessor provides all 56 pandas Series.str methods. Methods are implemented in two ways:

### Lazy Methods (SQL-based, 51 methods)

These methods return `ColumnExpr` and remain lazy until execution:

```python
# All these are lazy - no execution until to_df()
ds['name'].str.upper()           # → ColumnExpr (lazy)
ds['name'].str.lower()           # → ColumnExpr (lazy)
ds['name'].str.len()             # → ColumnExpr (lazy)
ds['name'].str.contains('test')  # → ColumnExpr (lazy)
ds['name'].str.replace('a', 'b') # → ColumnExpr (lazy)

# Assign to column (still lazy)
ds['upper_name'] = ds['name'].str.upper()

# Execute when needed
df = ds.to_df()  # ← SQL executes here
```

**Lazy methods include**: `upper`, `lower`, `len`, `strip`, `lstrip`, `rstrip`, `contains`, `startswith`, `endswith`, `replace`, `split`, `rsplit`, `slice`, `pad`, `center`, `ljust`, `rjust`, `zfill`, `repeat`, `find`, `rfind`, `index`, `rindex`, `match`, `fullmatch`, `extract`, `encode`, `decode`, `capitalize`, `title`, `swapcase`, `casefold`, `normalize`, `isalnum`, `isalpha`, `isdigit`, `isspace`, `islower`, `isupper`, `istitle`, `isnumeric`, `isdecimal`, `wrap`, `get`, `count`, `join`, `slice_replace`, `translate`, `removeprefix`, `removesuffix`

### Executing Methods (5 methods)

These methods **must execute** because they change the return structure:

| Method | Return Type | Why Execution Required |
|--------|-------------|------------------------------|
| `partition(sep)` | `DataStore` (3 columns) | Returns DataFrame with 3 columns (left, sep, right) - cannot be represented as single SQL expression |
| `rpartition(sep)` | `DataStore` (3 columns) | Same as partition, splits from right |
| `get_dummies(sep)` | `DataStore` (N columns) | Creates dynamic number of columns based on unique values - column count unknown at query time |
| `extractall(pat)` | `DataStore` | Returns MultiIndex DataFrame with all regex matches - row count changes |
| `cat(sep)` | `str` | **Aggregates** all strings into single value - reduces N rows to 1 scalar |

```python
# These execute immediately and return results
ds['name'].str.partition('|')      # → DataStore (3 columns)
ds['name'].str.get_dummies('|')    # → DataStore (N dummy columns)
ds['name'].str.extractall(r'\d+')  # → DataStore (all matches)
ds['name'].str.cat(sep='-')        # → str ("John-Jane-Bob")
```

### Why `cat()` Requires Execution

`cat()` is fundamentally different from other string methods:

```python
# Other methods: row-wise transformation (N rows → N rows)
ds['name'].str.upper()  # ["john", "jane"] → ["JOHN", "JANE"]

# cat(): aggregation (N rows → 1 scalar)  
ds['name'].str.cat(sep='-')  # ["john", "jane"] → "john-jane"
```

`cat()` performs an **aggregation** operation that:
1. Reads all values in the column
2. Concatenates them with a separator
3. Returns a single string

This cannot be expressed as a per-row SQL expression. It requires executing the data first, then calling pandas' `str.cat()` method.

### ✅ Comparison
- [x] `df.equals()` - Test equality
- [x] `df.compare()` - Show differences

### ✅ Miscellaneous
- [x] `df.info()` - Print summary
- [x] `df.memory_usage()` - Memory usage
- [x] `df.copy()` - Copy DataFrame

## Key Differences from Pandas

### 1. Row Ordering Behavior

DataStore has different row ordering guarantees depending on the data source:

**For DataFrame sources (in-memory):** Row order IS preserved. DataStore uses chDB's built-in `_row_id` virtual column (available in chDB v4.0.0b5+) to maintain original row order and pandas index:

```python
# ✅ Row order is preserved for DataFrame sources
df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
ds = DataStore(df)
result = ds[ds['value'] > 10]
# Result preserves original row order and pandas index
assert list(result.index) == [1, 2]  # Original indices preserved
```

**For file sources (CSV, Parquet):** Row order is NOT guaranteed unless you explicitly specify ORDER BY:

```python
# ❌ Order may vary between executions for file sources
ds = DataStore.from_file("data.csv")
result = ds.filter(ds.value > 50).to_df()
# Row order is NOT guaranteed to match the original file order

# ✅ Explicitly specify ORDER BY for deterministic order
result = ds.filter(ds.value > 50).order_by('id').to_df()
# Rows are ordered by 'id' column
```

**Why the difference?** For DataFrame sources, chDB provides a deterministic `_row_id` virtual column that represents the 0-based row position from the original DataFrame. For file sources, ClickHouse may return rows in any order for better performance (standard SQL behavior).

**Impact on comparisons**:
- For DataFrame sources: Results should match pandas behavior including row order
- For file sources: Sort both DataFrames first or use set-based comparisons
- Use `df.sort_values('col').reset_index(drop=True)` before `pd.testing.assert_frame_equal()` for file sources

### 2. Immutability
DataStore operations are immutable - `inplace=True` is not supported:

```python
# ❌ Not supported
df.drop(columns=['col'], inplace=True)

# ✅ Correct usage
df = df.drop(columns=['col'])
```

### 3. Return Types
DataStore uses lazy evaluation for optimal performance:

```python
# DataFrame methods return DataStore
result = ds.drop(columns=['col'])  # Returns DataStore
df = result.to_df()  # Get underlying DataFrame

# Column access returns ColumnExpr (lazy)
col = ds['column']  # Returns ColumnExpr (displays like Series)
pd_series = col.to_pandas()  # Convert to pd.Series when needed

# Aggregations return LazyAggregate (lazy)
mean_result = ds['age'].mean()  # Returns LazyAggregate
print(mean_result)  # Triggers execution, displays value

# Convert to pandas types
df = ds.to_df()  # DataStore → pd.DataFrame
series = ds['col'].to_pandas()  # ColumnExpr → pd.Series
```

### 4. Comparing Results with pandas

DataStore uses a **Lazy Execution** model, so column operations return `ColumnExpr` or `LazySeries` objects instead of `pd.Series`. When comparing DataStore results with pandas, you need to convert to pandas first:

```python
import pandas as pd
import datastore as ds

df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
ds_df = ds.DataFrame(df)

pd_col = df['a']
ds_col = ds_df['a']

# ❌ Wrong: pandas.equals() doesn't recognize DataStore objects
pd_col.equals(ds_col)  # Returns False (pandas limitation)

# ✅ Correct: Use to_pandas() to convert first
pd_col.equals(ds_col.to_pandas())  # Returns True

# ✅ Correct: Use DataStore's equals() method (works both ways)
ds_col.equals(pd_col)  # Returns True

# ✅ For testing: Use pandas testing utilities
pd.testing.assert_series_equal(pd_col, ds_col.to_pandas())

# ✅ For values only: Use numpy
import numpy as np
np.array_equal(pd_col.values, ds_col.values)  # Returns True
```

**Why does `pd_col.equals(ds_col)` return False?**

pandas `Series.equals()` checks `isinstance(other, pd.Series)` first. Since DataStore's `ColumnExpr` is not a `pd.Series` subclass, it immediately returns `False` - this is a pandas design limitation.

**Recommended patterns:**

| Use Case | Recommended Method |
|----------|-------------------|
| Compare values | `ds_col.equals(pd_col)` or `pd_col.equals(ds_col.to_pandas())` |
| Test assertions | `pd.testing.assert_series_equal(pd_result, ds_result.to_pandas())` |
| Values only | `np.array_equal(pd_col.values, ds_col.values)` |
| Float comparison | `np.allclose(pd_col.values, ds_col.values)` |

### 5. Series Handling and LazySeries
Operations that return Series in pandas return lazy objects in DataStore:

```python
# Returns ColumnExpr (lazy), displays like pandas Series
series = ds['column']  
print(type(series))  # <class 'datastore.column_expr.ColumnExpr'>

# Convert to pandas when needed
pd_series = series.to_pandas()
print(type(pd_series))  # <class 'pandas.core.series.Series'>

# Multiple columns return DataStore
datastore = ds[['col1', 'col2']]
print(type(datastore))  # <class 'datastore.core.DataStore'>

# Method calls on columns return LazySeries
head_result = ds['column'].head(5)
print(type(head_result))  # <class 'datastore.lazy_result.LazySeries'>

# LazySeries executes on access (values, index, repr, etc.)
values = head_result.values  # Triggers execution
```

**Why Lazy?** This enables SQL query optimization and deferred execution.

### 6. The `to_pandas()` Method

Both DataStore and ColumnExpr provide `to_pandas()` for explicit conversion:

```python
# DataStore to DataFrame
ds = DataStore.from_file("data.csv")
df = ds.to_pandas()  # Returns pd.DataFrame (alias for to_df())

# ColumnExpr to Series
col = ds['age']
series = col.to_pandas()  # Returns pd.Series

# Useful for library interoperability
import seaborn as sns
sns.histplot(ds['age'].to_pandas())
```

### 6. Method Naming
The INSERT VALUES method has been renamed to avoid conflicts:

```python
# Old (conflicts with df.values property)
ds.insert_into('id', 'name').values(1, 'Alice')

# New (recommended)
ds.insert_into('id', 'name').insert_values(1, 'Alice')
```

## Execution Model

DataStore implements a sophisticated **Mixed Execution Engine** that enables **arbitrary mixing** of SQL and pandas operations. 

### Key Innovation: SQL on DataFrames

After execution, SQL-style operations use **chDB's `Python()` table function** to execute SQL directly on cached DataFrames, enabling true mixed execution.

### Three-Stage Execution

**Stage 1: SQL Query Building (Lazy)**
```python
ds = DataStore.from_file("data.csv")
ds1 = ds.select('*')                    # Builds: SELECT *
ds2 = ds1.filter(ds.age > 25)           # Adds: WHERE age > 25
# ds2._executed = False (no execution yet)
```

**Stage 2: Execution (First pandas Operation)**
```python
ds3 = ds2.add_prefix('emp_')            # ← Executes SQL here!
# ds3._executed = True
# ds3._cached_df = DataFrame with filtered data and prefixed columns
```

**Stage 3: SQL on DataFrame (chDB Magic)**
```python
ds4 = ds3.filter(ds.emp_age > 30)       # SQL on DataFrame!
# Internally: SELECT * FROM Python(__datastore_cached_df__) WHERE emp_age > 30
# ds4._executed = True (result cached)
```

### Arbitrary Mixing Examples

**Example 1: SQL → Pandas → SQL → Pandas**
```python
result = (ds
    .filter(ds.age > 25)                      # SQL query building
    .add_prefix('emp_')                       # Pandas (executes)
    .filter(ds.emp_salary > 55000)            # SQL on DataFrame!
    .fillna(0))                               # Pandas on DataFrame
```

**Example 2: Pandas → SQL → Pandas → SQL**
```python
result = (ds
    .rename(columns={'id': 'ID'})             # Pandas (executes)
    .filter(ds.ID > 5)                        # SQL on DataFrame
    .sort_values('salary')                    # Pandas
    .select('ID', 'name', 'salary'))          # SQL on DataFrame again!
```

**Example 3: Complex Mixed Chain**
```python
result = (ds
    .select('*')                              # SQL 1
    .filter(ds.status == 'active')            # SQL 2
    .assign(revenue=lambda x: x['price'] * x['qty'])  # Pandas (executes)
    .filter(ds.revenue > 1000)                # SQL 3 on DataFrame
    .add_prefix('sales_')                     # Pandas
    .query('sales_revenue > 5000')            # Pandas
    .select('sales_id', 'sales_customer', 'sales_revenue'))  # SQL 4 on DataFrame
```

**For detailed execution plan visualization, see [Explain Method](EXPLAIN_METHOD.md)**

## Performance Tips

### 1. Use SQL for Filtering
Let the query engine do heavy filtering before pandas operations:

```python
# ✅ Efficient
result = (ds
    .select('*')
    .filter(ds.date >= '2024-01-01')  # SQL filter
    .filter(ds.amount > 1000)         # SQL filter
    .assign(margin=lambda x: x['profit'] / x['revenue'])  # Pandas transform
    .groupby('category').agg({'revenue': 'sum'}))  # Pandas aggregation

# ❌ Less efficient
result = (ds
    .to_df()  # Load all data
    .query('date >= "2024-01-01" and amount > 1000'))  # Filter in memory
```

### 2. Understand Execution
Once executed (pandas operation applied), all subsequent operations use cached data:

```python
ds = DataStore.from_file("big_data.csv")

# SQL operations - build query (lazy)
ds_filtered = ds.select('*').filter(ds.value > 0)  # No execution yet

# First pandas operation - executes
ds_prefixed = ds_filtered.add_prefix('col_')  # ← Query executes here!

# All subsequent operations use cached DataFrame
mean = ds_prefixed.mean()       # Uses cache, no SQL
std = ds_prefixed.std()         # Uses cache, no SQL
df = ds_prefixed.to_df()        # Returns cache, no SQL
```

### 3. Optimal Workflow Pattern

**Best Practice**: Filter in SQL, transform in pandas

```python
# ✅ Optimal: SQL filtering → Pandas transformation
result = (ds
    .select('*')
    .filter(ds.date >= '2024-01-01')    # SQL: Filters billions of rows
    .filter(ds.amount > 1000)           # SQL: More filtering
    # ↑ Query built but not executed yet
    
    .add_prefix('col_')                 # ← Executes SQL here, caches result
    .fillna(0)                          # Pandas: Works on cached result
    .assign(margin=lambda x: x['col_profit'] / x['col_revenue']))  # Pandas
```

### 4. Chain Operations
Chain multiple operations for better readability and potential optimization:

```python
result = (ds
    .drop(columns=['unused1', 'unused2'])
    .fillna(0)
    .assign(
        revenue=lambda x: x['price'] * x['quantity'],
        margin=lambda x: x['profit'] / x['revenue']
    )
    .query('margin > 0.2')
    .sort_values('revenue', ascending=False)
    .head(100))
```

## Examples

### Example 1: Data Cleaning
```python
cleaned = (ds
    .drop(columns=['temp_col'])
    .dropna(subset=['important_col'])
    .drop_duplicates()
    .fillna({'numeric_col': 0, 'string_col': 'unknown'})
    .astype({'id': 'int64', 'amount': 'float64'}))
```

### Example 2: Feature Engineering
```python
featured = ds.assign(
    revenue=lambda x: x['price'] * x['quantity'],
    profit=lambda x: x['revenue'] - x['cost'],
    margin=lambda x: x['profit'] / x['revenue'],
    high_value=lambda x: x['revenue'] > 1000
)
```

### Example 3: Time Series Analysis
```python
ts_result = (ds
    .set_index('date')
    .sort_index()
    .asfreq('D')
    .fillna(method='ffill')
    .rolling(window=7).mean()
    .shift(1))
```

### Example 4: Binary Operations
```python
# Calculate year-over-year growth
growth = (current_year
    .set_index('product')
    .sub(last_year.set_index('product'))
    .div(last_year.set_index('product'))
    .mul(100))
```

### Example 5: Conditional Operations
```python
# Complex filtering and transformation
result = (ds
    .query('age > 18 and income > 50000')
    .assign(
        segment=lambda x: pd.cut(x['income'], 
                                  bins=[0, 75000, 150000, float('inf')],
                                  labels=['Low', 'Medium', 'High'])
    )
    .where(lambda x: x['score'] > 0, 0)
    .groupby('segment')
    .agg({'income': 'mean', 'score': 'sum'}))
```

### Example 6: Mixing SQL and Pandas
```python
# Optimal workflow
result = (ds
    # Use SQL for heavy filtering
    .select('customer_id', 'order_date', 'amount', 'product_category')
    .filter(ds.order_date >= '2024-01-01')
    .filter(ds.order_date < '2024-02-01')
    .filter(ds.amount > 0)
    
    # Use pandas for complex transformations
    .assign(
        month=lambda x: pd.to_datetime(x['order_date']).dt.month,
        is_high_value=lambda x: x['amount'] > x['amount'].quantile(0.75)
    )
    .groupby(['customer_id', 'month'])
    .agg({
        'amount': ['sum', 'mean', 'count'],
        'is_high_value': 'sum'
    })
    .reset_index()
    
    # Export
    .to_parquet('customer_monthly_summary.parquet'))
```

## Limitations

### Not Implemented
- `inplace=True` parameter (DataStore is immutable)
- Some deprecated pandas methods
- Methods that don't make sense for DataStore (e.g., `from_dict`, `from_records` as instance methods)

### Partial Support
- `df.groupby()` - Returns pandas GroupBy object, not DataStore
- Class methods - Return pandas objects, not DataStore

## Pandas Version Compatibility

DataStore requires **pandas >= 2.1.0** and **Python >= 3.9**.

### Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| `DataFrame.map()` | ✅ Available | Recommended over `applymap()` |
| `groupby.apply(include_groups=...)` | ✅ Available | Control group column inclusion |
| Nullable type handling | ✅ Full support | Best dtype preservation |
| `first()`/`last()` deprecation | ✅ FutureWarning | Warns about offset string usage |

### Recommendations

1. **For production**: Pin specific pandas version to ensure consistent behavior
2. **For best performance**: Use pandas 2.2+ for latest optimizations

## Getting Help

- **Documentation**: See [DataStore README](../README.md)
- **Examples**: Check [examples/example_pandas_extended.py](../examples/example_pandas_extended.py)
- **Pandas Docs**: https://pandas.pydata.org/docs/reference/frame.html

## Summary

DataStore provides **comprehensive pandas DataFrame API** compatibility with seamless integration:

- ✅ **209** pandas DataFrame methods implemented
- ✅ **56** pandas `.str` accessor methods (all pandas str methods covered)
- ✅ **42+** pandas `.dt` accessor methods (plus ClickHouse datetime extras)
- ✅ **334 ClickHouse functions** mapped to Pandas-like API
- ✅ **ClickHouse-specific accessors**: `.arr` (37 methods), `.json`, `.url`, `.ip`, `.geo`
- ✅ Mix SQL queries with pandas transformations
- ✅ Automatic DataFrame/Series wrapping
- ✅ Performance optimization through caching
- ✅ Immutable, thread-safe operations

Use DataStore when you need the power of pandas with the performance of SQL query optimization!
