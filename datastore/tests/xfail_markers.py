"""
Centralized xfail markers for DataStore tests.

This module provides a single source of truth for all expected test failures
due to known limitations in chDB, DataStore, or design differences from pandas.

Usage:
    from tests.xfail_markers import chdb_category_type, bug_groupby_first_last

    @chdb_category_type
    def test_something():
        ...

Marker Naming Conventions:
    - chdb_*     : chDB/ClickHouse engine limitations (cannot fix in DataStore)
    - bug_*      : DataStore bugs to be fixed (should match pandas behavior)
    - limit_*    : DataStore limitations (features not yet implemented)
    - design_*   : Intentional behavioral differences from pandas
    - deprecated_*: Deprecated pandas features

When a bug is fixed or limitation is resolved, remove the marker from this file
and update all tests that use it.
"""

from typing import List

import pytest
import pandas as pd


# =============================================================================
# chDB Engine Limitations (chdb_*)
# Cannot be fixed in DataStore - inherent to chDB/ClickHouse
# =============================================================================

# Type Support
# NOTE: Timedelta works for read-only access but fails during SQL execution
# FIXED (chDB v4.x): Categorical type now works in SQL operations
def chdb_category_type(func):
    """FIXED (chDB v4.x): Categorical type now works in SQL operations."""
    return func


# Dtype Difference: chDB converts categorical to object after SQL execution
# Values are CORRECT, only dtype differs from pandas (category -> object)
chdb_category_to_object = pytest.mark.xfail(
    reason="chDB converts categorical dtype to object after SQL execution - VALUES ARE CORRECT, dtype differs",
    strict=True,
)

# FIXED (chDB v4.x): Timedelta type now works in SQL operations
def chdb_timedelta_type(func):
    """FIXED (chDB v4.x): Timedelta type now works in SQL operations."""
    return func


# DataStore limitation: timedelta arithmetic SQL generation
# NOTE: This is different from chdb_timedelta_type which was about INPUT support.
# This marker is for cases where DataStore tries to generate SQL for timedelta arithmetic
# (e.g., date + timedelta) and fails because pd.Timedelta is rendered as string.
datastore_timedelta_arithmetic = pytest.mark.xfail(
    reason="DataStore limitation: timedelta arithmetic in SQL generation renders Timedelta as string instead of INTERVAL",
    strict=True,
)

# FIXED (2026-01-07): json_extract_array_raw now falls back to pandas
# The fix detects Array-returning JSON functions and uses Python JSON parsing
# instead of SQL execution to avoid the "Array cannot be inside Nullable" error.
# Keeping the marker for other potential uses of Array types via Python() table function.
chdb_array_nullable = pytest.mark.xfail(
    reason="chDB: Array type cannot be inside Nullable type",
    strict=True,
)

# NOTE: numpy arrays work for read-only access but may have issues in SQL operations
chdb_array_string_conversion = pytest.mark.xfail(
    reason="chDB converts numpy arrays to strings via Python() table function",
    strict=True,
)

# NOTE: bytes/blob columns work for read-only access but are converted to strings in SQL operations
chdb_blob_to_string_conversion = pytest.mark.xfail(
    reason="chDB converts bytes/blob data to strings during SQL execution (filter, sort, head/tail, select)",
    strict=True,
)

# FIXED: chDB now handles Nullable Int64 comparison correctly (resolved in recent chDB version)
# chdb_nullable_int64_comparison = pytest.mark.xfail(
#     reason="chDB does not handle Nullable Int64 comparison correctly - returns raw bytes",
#     strict=True,
# )

# Function Limitations
chdb_no_product_function = pytest.mark.xfail(
    reason="chDB does not support product() aggregate function",
    strict=True,
)

chdb_no_normalize_utf8 = pytest.mark.xfail(
    reason="chDB: normalizeUTF8NFD function does not exist",
    strict=True,
)

chdb_no_quantile_array = pytest.mark.xfail(
    reason="chDB does not support quantile with array parameter",
    strict=True,
)

chdb_median_in_where = pytest.mark.xfail(
    reason="Aggregate function median() in WHERE clause requires subquery - not supported",
    strict=True,
)

# NULL/NaN Handling
# NOTE: chdb_null_in_groupby REMOVED - Fixed by implementing dropna parameter support
# in groupby operations. DataStore now properly excludes NULL groups by default (dropna=True)
# and includes them when dropna=False, matching pandas behavior.
# See: tracking/discoveries/2026-01-06_groupby_dropna_alignment_research.md

# NOTE: chdb_nan_sum_behavior REMOVED
# Fixed in column_expr.py _execute_groupby_aggregation() by adding fillna(0)
# for sum aggregation results to match pandas behavior.
# chdb_nan_sum_behavior = pytest.mark.xfail(
#     reason="chDB returns NA for sum of all-NaN, pandas returns 0 (SQL standard behavior, may add workaround in DataStore)",
#     strict=True,
# )

# NOTE: chdb_null_comparison_semantics and chdb_null_string_comparison REMOVED
# Fixed in conditions.py using ifNull() wrapping for pandas NULL semantics

# String/Unicode
chdb_unicode_filter = pytest.mark.xfail(
    reason="Unicode string equality in SQL filter has encoding issues",
    strict=True,
)

chdb_strip_whitespace = pytest.mark.xfail(
    reason="str.strip() doesn't handle all whitespace types correctly in chDB",
    strict=True,
)

# FIXED: String concatenation now auto-converts '+' to concat() in ArithmeticExpression
# chdb_string_plus_operator = pytest.mark.xfail(
#     reason="chDB/ClickHouse does not support '+' operator for string concatenation, must use concat() function",
#     strict=True,
# )


# Datetime
# FIXED (2026-01-14): dt.year/month/day extraction now works correctly in chDB 4.0.0b3
# The original issue was year extraction at timezone boundaries, which is now fixed.
# chdb_datetime_timezone = pytest.mark.xfail(
#     reason="chDB adds timezone to datetime, causing boundary comparison differences",
#     strict=False,  # behavior varies by Python/chDB version
# )
def chdb_datetime_timezone(func):
    """FIXED (chDB 4.0.0b3): Datetime extraction (dt.year etc) now works correctly."""
    return func


# NOTE: Date range comparison still has issues due to Python() table function timezone offset
# When pandas DataFrame dates are loaded via Python() table function, local timezone offset is added.
# Workaround: use toTimezone(date, 'UTC') in SQL comparisons
chdb_datetime_range_comparison = pytest.mark.xfail(
    reason="chDB Python() table function adds local timezone offset to dates, "
    "causing date range boundary comparisons to be off by timezone offset (e.g., +8 hours for UTC+8)",
    strict=False,  # Pass in UTC (CI), fail in non-UTC timezones (e.g., UTC+8)
)


# FIXED (2026-01-14): Test was using shared DataFrame reference, causing column conflicts
# Fix: Use DataStore(pd_df.copy()) to avoid shared modification
def chdb_datetime_extraction_conflict(func):
    """FIXED: Issue was test code not using .copy(), not chDB limitation."""
    return func


# FIXED (2026-01-14): Test was using shared DataFrame reference, causing type conflicts
# Fix: Use DataStore(pd_df.copy()) to avoid shared modification
def chdb_dt_month_type(func):
    """FIXED: Issue was test code not using .copy(), not chDB type mismatch."""
    return func


# SQL
chdb_duplicate_column_rename = pytest.mark.xfail(
    reason="SQL automatically renames duplicate columns - known limitation",
    strict=True,
)

chdb_case_bool_conversion = pytest.mark.xfail(
    reason="SQL CASE WHEN cannot convert Bool to Int64/String",
    strict=True,
)

# Dtype Differences - values are CORRECT, only dtype differs from pandas
# These are acceptable differences where DataStore returns more semantically correct types
chdb_nat_returns_nullable_int = pytest.mark.xfail(
    reason="chDB datetime accessor with NaT returns nullable Int (Int32), pandas returns float64 - VALUES ARE CORRECT",
    strict=True,
)

chdb_replace_none_dtype = pytest.mark.xfail(
    reason="chDB replace with None returns nullable Int64, pandas returns object dtype - VALUES ARE CORRECT",
    strict=True,
)

chdb_mask_dtype_nullable = pytest.mark.xfail(
    reason="chDB mask/where on int returns nullable Int64, pandas returns float64 (due to NaN) - VALUES ARE CORRECT",
    strict=True,
)

chdb_floordiv_returns_float = pytest.mark.xfail(
    reason="chDB floor division (//) returns float64, pandas returns int64 when both operands are int - VALUES ARE CORRECT",
    strict=True,
)

chdb_power_returns_float = pytest.mark.xfail(
    reason="chDB power (**) returns float64, pandas returns int64 when both operands are int - VALUES ARE CORRECT",
    strict=True,
)


# =============================================================================
# DataStore Bugs (bug_*)
# Should be fixed to match pandas behavior
# =============================================================================

# FIXED: chDB any()/anyLast() now correctly returns row-order based first/last (2026-01-06)
# bug_groupby_first_last = pytest.mark.xfail(
#     reason="""DataStore first()/last() not pandas-compatible.
#     Uses chDB any()/anyLast() which don't guarantee row-order.
#     Fix: use argMin/argMax with row_number or pandas fallback.""",
#     strict=False,  # behavior varies by chDB version
# )

# FIXED: DataStore groupby aggregation now preserves index correctly (2026-01-06)
# bug_groupby_index = pytest.mark.xfail(
#     reason="""DataStore groupby aggregation doesn't preserve index correctly.
#     Pandas groupby().agg() returns Series with groupby column as index.
#     DataStore should match this behavior.""",
#     strict=False,  # behavior varies by chDB version
# )

# FIXED (2026-01-06): Index info is now preserved through lazy SQL execution
# The fix tracks index info in _index_info during _ensure_sql_source() and
# restores the index after SQL execution in _execute().
# bug_index_not_preserved = pytest.mark.xfail(
#     reason="Index info not preserved through lazy SQL execution",
#     strict=True,
# )


# No-op decorator for import compatibility
def bug_index_not_preserved(func):
    """FIXED: Index info is now preserved through lazy SQL execution."""
    return func


# FIXED 2026-01-14: MultiIndex is now preserved in DataStore.from_df()
# bug_extractall_multiindex was an xfail marker for a bug that has been fixed
def bug_extractall_multiindex(func):
    """FIXED: MultiIndex is now preserved through DataStore.from_df() for extractall."""
    return func


# FIXED: None comparison now matches pandas semantics
# bug_null_string_comparison = pytest.mark.xfail(
#     reason="BUG: ds[ds['col'] != None] returns 0 rows, should return non-None rows. "
#     "Fix: convert != None to IS NOT NULL in DataStore layer",
#     strict=True,
# )

# FIXED: where() with computed column now works correctly
# bug_where_computed_column = pytest.mark.xfail(
#     reason="BUG: where() with lazy assigned column fails with 'Unknown expression identifier'. "
#     "Fix: resolve computed columns before where execution",
#     strict=True,
# )
# Use a no-op decorator since the bug is fixed
bug_where_computed_column = lambda f: f


# =============================================================================
# DataStore Limitations (limit_*)
# Features not yet implemented
# =============================================================================


# FIXED 2026-01-07: Callable as index now supported via __getitem__ callable handling
# limit_callable_index was an xfail marker for a limitation that has been fixed
def limit_callable_index(func):
    """No-op decorator for previously failing test that is now fixed."""
    return func


# FIXED 2026-01-07: query() @variable scope now works via level parameter
# limit_query_variable_scope was an xfail marker for a bug that has been fixed
def limit_query_variable_scope(func):
    """No-op decorator for previously failing test that is now fixed."""
    return func


# FIXED 2026-01-07: loc conditional assignment with ColumnExpr now works
# Added DataStoreLocIndexer wrapper in pandas_compat.py
def limit_loc_conditional_assignment(func):
    """No-op decorator for previously failing test that is now fixed."""
    return func


# FIXED 2026-01-06: where() with DataStore condition now works
# limit_where_condition was an xfail marker for a bug that has been fixed
def limit_where_condition(func):
    """No-op decorator - bug has been fixed."""
    return func


# NOTE: unstack() has been implemented on ColumnExpr
# These decorators are now no-op for backward compatibility

limit_str_join_array = pytest.mark.xfail(
    reason="str.join() requires Array type column, not string column",
    strict=True,
)

# =============================================================================
# Design Differences (design_*)
# Intentional behavioral differences from pandas
# These are conscious decisions, not bugs to be fixed.
# =============================================================================

design_datetime_fillna_nat = pytest.mark.xfail(
    reason="Design decision: Pandas where/mask replaces datetime with 0/-1, DataStore uses NaT (semantically clearer)",
    strict=True,
)


# FIXED: unstack() is now implemented on ColumnExpr
def design_unstack_column_expr(func):
    """No-op decorator - unstack() has been implemented."""
    return func


# Alias for backward compatibility
limit_unstack_column_expr = design_unstack_column_expr

# FIXED: DataStore now restricts column access after select() to match pandas behavior
# design_sql_select_column_access = pytest.mark.xfail(
#     reason="Design difference: SQL pushdown allows accessing original columns after select(), pandas restricts to selected columns only",
#     strict=True,
# )

# FIXED: SQL builder now properly layers computed columns between LIMIT and WHERE
# limit_sql_column_dependency_after_limit = pytest.mark.xfail(
#     reason="SQL pushdown limitation: FILTER referencing computed column created after LIMIT requires complex subquery nesting not yet implemented",
#     strict=True,
# )


# =============================================================================
# Deprecated Features (deprecated_*)
# Deprecated pandas features
# =============================================================================

deprecated_fillna_downcast = pytest.mark.xfail(
    reason="fillna downcast parameter is deprecated in pandas 2.x",
    strict=True,
)


# =============================================================================
# Legacy Aliases (for backward compatibility during migration)
# TODO: Remove after updating all test files
# =============================================================================

# bug_* aliases
# datastore_groupby_first_last_order = bug_groupby_first_last  # FIXED
# datastore_groupby_index_preservation = bug_groupby_index  # FIXED
# lazy_index_not_preserved = bug_index_not_preserved  # FIXED - now a no-op
lazy_index_not_preserved = bug_index_not_preserved  # Now a no-op function
lazy_extractall_multiindex = bug_extractall_multiindex

# limit_* aliases
datastore_callable_index = limit_callable_index
datastore_query_variable_scope = limit_query_variable_scope
datastore_loc_conditional_assignment = limit_loc_conditional_assignment
datastore_where_condition = limit_where_condition
datastore_unstack_column_expr = design_unstack_column_expr  # FIXED: unstack() now implemented
datastore_str_join_array = limit_str_join_array

# deprecated_* aliases
pandas_deprecated_fillna_downcast = deprecated_fillna_downcast


# =============================================================================
# Marker Registry
#
# This registry provides metadata for tracking and reporting.
# Format: marker_name -> (category, issue_url, notes)
# =============================================================================

MARKER_REGISTRY = {
    # =========================================================================
    # chDB Engine Limitations (cannot fix in DataStore)
    # =========================================================================
    # Type Support
    # NOTE: Timedelta works for read-only access but fails during SQL execution
    "chdb_category_type": ("fixed", "chDB v4.x", "CATEGORY type now works in SQL operations - FIXED"),
    "chdb_category_to_object": ("chdb", None, "chDB converts categorical to object dtype after SQL (values correct)"),
    "chdb_timedelta_type": ("fixed", "chDB v4.x", "TIMEDELTA type now works in SQL operations - FIXED"),
    "chdb_array_nullable": ("chdb", None, "Array cannot be inside Nullable type"),
    "chdb_array_string_conversion": ("chdb", None, "numpy arrays may be converted to strings in SQL operations"),
    # Functions
    "chdb_no_product_function": ("chdb", None, "product() aggregate not available"),
    "chdb_no_normalize_utf8": ("chdb", None, "normalizeUTF8NFD function not available"),
    "chdb_no_quantile_array": ("chdb", None, "quantile with array parameter not supported"),
    "chdb_median_in_where": ("chdb", None, "Aggregate in WHERE requires subquery"),
    # NULL/NaN
    # "chdb_nan_sum_behavior": ("chdb", None, "Sum of all-NaN returns NA (SQL standard)"),  # FIXED
    # String/Unicode
    "chdb_unicode_filter": ("chdb", None, "Unicode in SQL filter has encoding issues"),
    "chdb_strip_whitespace": ("chdb", None, "strip() doesn't handle all whitespace types"),
    # Datetime
    "chdb_datetime_timezone": ("fixed", "2026-01-14", "dt.year/month/day extraction - FIXED in chDB 4.0.0b3"),
    "chdb_datetime_range_comparison": ("chdb", None, "Python() table function adds local timezone offset to dates"),
    "chdb_datetime_extraction_conflict": ("chdb", None, "Multiple datetime extractions cause column name conflict"),
    "chdb_dt_month_type": ("chdb", None, "dt.month type inconsistency between SQL and DataFrame"),
    # SQL Behavior
    "chdb_duplicate_column_rename": ("chdb", None, "SQL auto-renames duplicate columns"),
    "chdb_case_bool_conversion": ("chdb", None, "CASE WHEN cannot mix Bool with other types"),
    # Dtype Differences (values correct, only dtype differs)
    "chdb_nat_returns_nullable_int": ("chdb", None, "dt accessor with NaT returns Nullable Int (values correct)"),
    "chdb_replace_none_dtype": ("chdb", None, "replace with None returns Nullable Int (values correct)"),
    "chdb_mask_dtype_nullable": ("chdb", None, "mask/where returns Nullable Int64 (values correct)"),
    # =========================================================================
    # DataStore Bugs (should be fixed)
    # =========================================================================
    "bug_index_not_preserved": ("fixed", None, "Index info lost through lazy SQL execution - FIXED"),
    "bug_extractall_multiindex": (
        "fixed",
        "2026-01-14",
        "MultiIndex lost in extractall - FIXED via DataStore.from_df()",
    ),
    # =========================================================================
    # DataStore Limitations (not yet implemented)
    # =========================================================================
    "limit_callable_index": ("fixed", "2026-01-07", "Callable as index - FIXED"),
    "limit_query_variable_scope": ("fixed", "2026-01-07", "query() @variable scope - FIXED via level parameter"),
    # FIXED: "limit_loc_conditional_assignment" - no longer needed,
    "limit_where_condition": ("fixed", "2026-01-06", "where() with DataStore condition - FIXED"),
    "limit_str_join_array": ("limit", None, "str.join() needs Array type column"),
    # =========================================================================
    # Design Decisions (intentional differences)
    # =========================================================================
    "design_datetime_fillna_nat": ("design", None, "Use NaT instead of 0/-1 for datetime where/mask"),
    "design_unstack_column_expr": ("fixed", None, "unstack() now implemented on ColumnExpr"),
    # =========================================================================
    # Deprecated Features (pandas deprecated)
    # =========================================================================
    "deprecated_fillna_downcast": ("deprecated", None, "fillna downcast deprecated in pandas 2.x"),
    # =========================================================================
    # FIXED (kept for reference)
    # =========================================================================
    # "chdb_nullable_int64_comparison": FIXED in chDB 4.0.0b3
    # "chdb_null_in_groupby": FIXED by dropna parameter implementation
    # "chdb_empty_df_str_dtype": FIXED in core.py empty DataFrame handling
    # "chdb_integer_column_names": FIXED in connection.py via string conversion
    # "bug_groupby_first_last": FIXED - chDB any()/anyLast() now row-order based
    # "bug_groupby_index": FIXED - groupby now preserves index correctly
    # "bug_setitem_computed_column_groupby": FIXED - setitem updates _computed_columns
    # "bug_groupby_column_selection_extra_columns": FIXED - column selection filters correctly
    # =========================================================================
    # FIXED 2026-01-07
    # =========================================================================
    "chdb_alias_shadows_column_in_where": ("fixed", "2026-01-07", "Alias no longer shadows column in WHERE - FIXED"),
    "limit_datastore_no_invert": ("fixed", "2026-01-07", "__invert__ (~) operator - FIXED"),
}


def get_markers_by_category(category: str) -> List[str]:
    """Get all marker names in a specific category."""
    return [name for name, (cat, _, _) in MARKER_REGISTRY.items() if cat == category]


def get_all_categories() -> List[str]:
    """Get all unique categories."""
    return list(set(cat for cat, _, _ in MARKER_REGISTRY.values()))


# =============================================================================
# FIXED Bug markers - kept for import compatibility
# =============================================================================


# FIXED (2026-01-06): setitem now correctly updates _computed_columns
def bug_setitem_computed_column_groupby(func):
    """FIXED: ds['col'] = expr now correctly populates _computed_columns."""
    return func


# =============================================================================
# FIXED markers - kept as no-op functions for import compatibility
# =============================================================================


# FIXED (2026-01-06): Empty DataFrame now executes SQL to get correct dtypes
def chdb_empty_df_str_dtype(func):
    """FIXED: Empty DataFrame str accessor now returns correct dtype."""
    return func


# FIXED (2026-01-06): Integer column names now work via string conversion in connection.py
def chdb_integer_column_names(func):
    """FIXED: Integer column names now work via string conversion."""
    return func


# =============================================================================
# Bug: groupby column selection includes extra columns - FIXED (2026-01-06)
# =============================================================================

# FIXED: groupby column selection now correctly filters columns
# bug_groupby_column_selection_extra_columns = pytest.mark.xfail(
#     reason="Bug: After assign() + groupby(), selecting specific columns [['a', 'b']] includes extra columns in result",
#     strict=True,
# )


# =============================================================================
# FIXED (2026-01-14): day_name/month_name now implemented via dateName() function
# =============================================================================


# No-op decorator for import compatibility
def chdb_no_day_month_name(func):
    """FIXED: day_name/month_name now implemented via dateName() SQL function."""
    return func


# =============================================================================
# FIXED (2026-01-14): strftime now uses pandas fallback for correct format codes
# =============================================================================


# No-op decorator for import compatibility
def chdb_strftime_format_difference(func):
    """FIXED: strftime now uses pandas fallback to ensure correct format code behavior."""
    return func


# =============================================================================
# chDB limitation: str.pad doesn't support 'side' parameter
# =============================================================================

chdb_pad_no_side_param = pytest.mark.xfail(
    reason="chDB limitation: str.pad() only supports left padding, 'side' parameter not implemented",
    strict=True,
)


# =============================================================================
# chDB limitation: str.center implementation differs
# =============================================================================

chdb_center_implementation = pytest.mark.xfail(
    reason="chDB limitation: str.center() implementation uses rightPad instead of proper centering",
    strict=True,
)


# =============================================================================
# chDB limitation: startswith/endswith don't support tuple argument
# =============================================================================

chdb_startswith_no_tuple = pytest.mark.xfail(
    reason="chDB limitation: startswith/endswith don't support tuple of prefixes/suffixes",
    strict=True,
)


# =============================================================================
# FIXED 2026-01-14: index property setter now implemented
# Added index.setter to PandasCompatMixin in pandas_compat.py
# =============================================================================


def limit_datastore_index_setter(func):
    """No-op decorator for previously failing test that is now fixed."""
    return func


# FIXED 2026-01-14: groupby now supports ColumnExpr/LazySeries as parameter
# Modified groupby() method in core.py to auto-assign expressions to temp columns


def limit_groupby_series_param(func):
    """No-op decorator for previously failing test that is now fixed."""
    return func


# NOTE: Simple alias cases work but complex chains with groupby still have issues
chdb_alias_shadows_column_in_where = pytest.mark.xfail(
    reason="chDB: In complex chains with groupby, SELECT alias may still shadow original column"
)


# FIXED 2026-01-14: __invert__ (~) operator for entire DataFrame now implemented
# Added __invert__ method to PandasCompatMixin in pandas_compat.py
def limit_datastore_no_invert(func):
    """No-op decorator for previously failing test that is now fixed."""
    return func


# =============================================================================
# FIXED (chDB v4.0.0b6): Python() table function non-contiguous index bug
# See: https://github.com/chdb-io/chdb/issues/478
# Previously, when a DataFrame had non-contiguous index (e.g., after slicing with step),
# chDB returned incorrect data from the original DataFrame instead of the sliced data.
# This is now fixed in chDB 4.0.0b6.
# =============================================================================


def chdb_python_table_noncontiguous_index(func):
    """FIXED (chDB v4.0.0b6): Non-contiguous index now correctly handled."""
    return func


# =============================================================================
# FIXED (chDB v4.0.0b5): rowNumberInAllBlocks() non-deterministic with Python() table function
# See: https://github.com/chdb-io/chdb/issues/469
# Fixed by using _row_id virtual column instead of rowNumberInAllBlocks()
# _row_id is a built-in deterministic virtual column in chDB v4.0.0b5+ that
# provides the 0-based row number from the original DataFrame.
# =============================================================================


def chdb_python_table_rownumber_nondeterministic(func):
    """FIXED (chDB v4.0.0b5): _row_id virtual column is now deterministic."""
    return func


# =============================================================================
# Legacy Pandas Version Markers (kept as no-ops for import compatibility)
# Since pandas >= 2.1.0 is now required, these are no longer needed
# =============================================================================


def pandas_version_no_dataframe_map(func):
    """No-op: pandas >= 2.1.0 now required, DataFrame.map() always available."""
    return func


def pandas_version_no_include_groups(func):
    """No-op: pandas >= 2.1.0 now required, include_groups always available."""
    return func


def pandas_version_first_last_offset_warning(func):
    """No-op: pandas >= 2.1.0 now required."""
    return func


def pandas_version_nullable_int_dtype(func):
    """No-op: pandas >= 2.1.0 now required."""
    return func


def pandas_version_nullable_bool_sql(func):
    """No-op: pandas >= 2.1.0 now required."""
    return func


def pandas_version_cut_array_protocol(func):
    """No-op: pandas >= 2.1.0 now required."""
    return func


def skip_if_old_pandas(reason: str = "Requires pandas 2.1+"):
    """
    No-op decorator: pandas >= 2.1.0 now required.

    Kept for import compatibility with existing tests.
    """

    def decorator(func):
        return func

    return decorator


def groupby_apply_compat(grouped, func, **kwargs):
    """
    Call groupby.apply() with include_groups=False.

    Usage:
        pd_result = groupby_apply_compat(df.groupby('category'), lambda x: x.sum())
    """
    return grouped.apply(func, include_groups=False, **kwargs)
