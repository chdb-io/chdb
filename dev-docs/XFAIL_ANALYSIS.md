# xfail Marker Analysis

> Generated: 2026-01-14
> 
> This document categorizes all active xfail markers in `tests/xfail_markers.py`.

---

## Overview

| Category | Marker Count | Test Cases | Status |
|----------|--------------|------------|--------|
| **chDB Engine Limitations** | 24 | 54 | Cannot fix at DataStore layer |
| **DataStore Bug** | 0 | 0 | All fixed |
| **DataStore Limitations** | 1 | 1 | Can be implemented |
| **Design Decisions** | 1 | 2 | Intentional |
| **Deprecated Features** | 1 | 1 | pandas evolution |
| **Fixed (no-op)** | 14+ | 15+ | Kept for import compatibility |
| **Total** | **27 active** | **58 + 15** | |

**Test Impact**: ~73 test cases marked (58 active xfail + 15 no-op), distributed across 32 test files.

---

## 1. chDB Engine Limitations (chdb_*) - Cannot Fix at DataStore Layer

These are limitations of the chDB/ClickHouse engine itself that DataStore cannot work around.

### Type Support (4)

| Marker | Reason | Notes |
|--------|--------|-------|
| `chdb_category_type` | chDB does not support CATEGORY numpy type | Read-only access works |
| `chdb_timedelta_type` | chDB does not support TIMEDELTA numpy type | Read-only access works |
| `chdb_array_nullable` | Array type cannot be inside Nullable | Affects JSON-related functions |
| `chdb_array_string_conversion` | numpy array is converted to string in SQL | Affects array accessor |

### Missing Functions (4)

| Marker | Reason | pandas Equivalent |
|--------|--------|-------------------|
| `chdb_no_product_function` | Does not support `product()` aggregate function | `df.prod()` |
| `chdb_no_normalize_utf8` | No `normalizeUTF8NFD` function | `str.normalize()` |
| `chdb_no_quantile_array` | `quantile` does not support array parameter | `quantile([0.25, 0.75])` |
| `chdb_median_in_where` | Aggregate functions in WHERE clause need subquery | `df[df['x'] > df['x'].median()]` |

### String/Unicode (2)

| Marker | Reason |
|--------|--------|
| `chdb_unicode_filter` | Unicode strings have encoding issues in SQL filters |
| `chdb_strip_whitespace` | `str.strip()` cannot handle all whitespace types |

### DateTime (5)

| Marker | Reason | strict |
|--------|--------|--------|
| `chdb_datetime_range_comparison` | Python() table function adds local timezone offset to dates, causing date range comparison deviation | True |
| `chdb_datetime_extraction_conflict` | Multiple dt extractions cause column name conflicts | True |
| `chdb_dt_month_type` | `dt.month` returns inconsistent types between SQL and DataFrame | True |
| `chdb_no_day_month_name` | `day_name()`/`month_name()` not implemented in SQL mapping | True |
| `chdb_strftime_format_difference` | `strftime('%M')` returns month name instead of minute number | True |

> **Note**: `chdb_datetime_timezone` (dt.year and other date extractions) was fixed in chDB 4.0.0b3.

### SQL Behavior (3)

| Marker | Reason |
|--------|--------|
| `chdb_duplicate_column_rename` | SQL automatically renames duplicate column names |
| `chdb_case_bool_conversion` | CASE WHEN cannot convert between Bool and Int64/String |
| `chdb_alias_shadows_column_in_where` | SELECT alias may shadow original column name in complex groupby chains |

### String Method Limitations (3)

| Marker | Reason | pandas Method |
|--------|--------|---------------|
| `chdb_pad_no_side_param` | `str.pad()` only supports left padding, no `side` parameter | `str.pad(side='right')` |
| `chdb_center_implementation` | `str.center()` implementation uses rightPad instead of proper centering | `str.center()` |
| `chdb_startswith_no_tuple` | `startswith/endswith` does not support tuple parameter | `str.startswith(('a', 'b'))` |

### dtype Differences (3)

> **Note**: In these cases **values are correct**, only the data type differs from pandas. The types returned by DataStore may be semantically more correct.

| Marker | Reason | DataStore Returns | pandas Returns |
|--------|--------|-------------------|----------------|
| `chdb_nat_returns_nullable_int` | NaT handling | Nullable Int32 | float64 |
| `chdb_replace_none_dtype` | `replace(None)` | Nullable Int64 | object |
| `chdb_mask_dtype_nullable` | `mask/where` on int | Nullable Int64 | float64 |

### chDB Bug (0)

> **Note**: `chdb_python_table_noncontiguous_index` has been fixed in chDB 4.0.0b6, see Fixed Markers section.

---

## 2. DataStore Bug (bug_*) - Should Be Fixed

These are DataStore bugs that should be fixed to match pandas behavior.

| Marker | Reason | Status |
|--------|--------|--------|
| ~~`bug_extractall_multiindex`~~ | `extractall` returns MultiIndex DataFrame | Fixed (2026-01-14) |

> **Note**: `bug_extractall_multiindex` has been fixed, MultiIndex is now correctly preserved via `DataStore.from_df()`.

---

## 3. DataStore Limitations (limit_*) - Unimplemented Features

These are features not yet implemented in DataStore.

| Marker | Reason | Priority | Workaround |
|--------|--------|----------|------------|
| `limit_str_join_array` | `str.join()` requires Array type column | Low | Use pandas fallback |

> **Note**: `limit_datastore_index_setter` and `limit_groupby_series_param` have been fixed, see Fixed Markers section.

---

## 4. Design Decisions (design_*) - Intentional Behavior Differences

These are conscious design decisions, not bugs to be fixed.

| Marker | Reason | Explanation |
|--------|--------|-------------|
| `design_datetime_fillna_nat` | datetime `where/mask` uses NaT instead of 0/-1 | pandas uses 0/-1 replacement, DataStore uses NaT which is semantically clearer |

---

## 5. Deprecated Features (deprecated_*)

Features deprecated by pandas.

| Marker | Reason | pandas Version |
|--------|--------|----------------|
| `deprecated_fillna_downcast` | `fillna(downcast=...)` parameter is deprecated | pandas 2.x |

---

## 6. Pandas Version Compatibility (pandas_version_*)

> **Note**: These are `skipif` markers, not `xfail`. Used to handle API differences between different pandas versions.

| Marker | Condition | Explanation |
|--------|-----------|-------------|
| `pandas_version_no_dataframe_map` | pandas < 2.1 | `DataFrame.map()` added in 2.1+ |
| `pandas_version_no_include_groups` | pandas < 2.1 | `groupby.apply(include_groups=...)` added in 2.1+ |
| `pandas_version_nullable_int_dtype` | pandas < 2.1 | Nullable Int64 handling improved in 2.1+ |
| `pandas_version_nullable_bool_sql` | pandas < 2.1 | Nullable bool SQL handling differences |

---

## Fix Priority Recommendations

### High Priority
None (all high priority bugs have been fixed)

### Medium Priority
None (all medium priority have been fixed)

### Low Priority (consider pandas fallback)
1. **DateTime related** (`chdb_datetime_*`): Most problematic area, can add fallback
2. **String methods** (`chdb_pad_*`, `chdb_center_*`): Less common use cases

---

## Fixed Markers (Reference)

The following markers have been fixed and are kept as no-op functions in `xfail_markers.py` for import compatibility:

- `chdb_nullable_int64_comparison` - Fixed in chDB 4.0.0b3
- `chdb_null_in_groupby` - dropna parameter implemented
- `chdb_nan_sum_behavior` - fillna(0) workaround
- `chdb_string_plus_operator` - Auto-converted to concat()
- `chdb_datetime_timezone` - dt.year/month/day extraction fixed in chDB 4.0.0b3
- `bug_groupby_first_last` - chDB any()/anyLast() now preserves order
- `bug_groupby_index` - groupby now correctly preserves index
- `bug_index_not_preserved` - lazy execution now preserves index info
- `bug_extractall_multiindex` - MultiIndex correctly preserved via DataStore.from_df() (2026-01-14)
- `limit_datastore_index_setter` - index property setter implemented (2026-01-14)
- `limit_groupby_series_param` - groupby now supports ColumnExpr/LazySeries parameter (2026-01-14)
- `limit_callable_index` - callable as index now supported
- `limit_query_variable_scope` - query() @variable now supported
- `limit_loc_conditional_assignment` - loc conditional assignment now supported
- `limit_where_condition` - where() condition now supported
- `design_unstack_column_expr` - unstack() implemented
- `chdb_python_table_rownumber_nondeterministic` - Solved with _row_id virtual column
- `limit_datastore_no_invert` - `__invert__` method added to PandasCompatMixin
- `chdb_python_table_noncontiguous_index` - Fixed in chDB 4.0.0b6, non-contiguous index (e.g., df[::2]) now works correctly (2026-01-15)
