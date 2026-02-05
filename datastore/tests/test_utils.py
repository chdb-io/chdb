"""
Test utility functions for DataStore vs Pandas comparison.

These utilities follow the Mirror Code Pattern and Complete Output Comparison
principles defined in .cursor/rules/chdb-ds.mdc
"""

import numpy as np
import pandas as pd
from typing import Any


# =============================================================================
# Unified comparison functions - wrappers with check_names=True by default
# =============================================================================


def assert_frame_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    check_dtype: bool = True,
    check_index_type: str = "equiv",
    check_column_type: str = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
    by_blocks: bool = False,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_like: bool = False,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "DataFrame",
) -> None:
    """
    Wrapper for pd.testing.assert_frame_equal with check_names=True by default.

    This ensures column names and index names are checked by default.
    All parameters match pd.testing.assert_frame_equal signature.
    """
    pd.testing.assert_frame_equal(
        left,
        right,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=check_frame_type,
        check_names=check_names,
        by_blocks=by_blocks,
        check_exact=check_exact,
        check_datetimelike_compat=check_datetimelike_compat,
        check_categorical=check_categorical,
        check_like=check_like,
        check_freq=check_freq,
        check_flags=check_flags,
        rtol=rtol,
        atol=atol,
        obj=obj,
    )


def assert_series_equal(
    left,
    right: pd.Series,
    check_dtype: bool = True,
    check_index_type: str = "equiv",
    check_series_type: bool = True,
    check_names: bool = True,
    check_exact: bool = False,
    check_datetimelike_compat: bool = False,
    check_categorical: bool = True,
    check_category_order: bool = True,
    check_freq: bool = True,
    check_flags: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    obj: str = "Series",
    check_index: bool = True,
    check_like: bool = False,
) -> None:
    """
    Wrapper for pd.testing.assert_series_equal with check_names=True by default.

    This ensures Series names and index names are checked by default.
    All parameters match pd.testing.assert_series_equal signature.

    Supports DataStore lazy objects (ColumnExpr, LazyCondition, etc.) as left argument.
    These are automatically executed to get the underlying Series.
    """
    # Auto-convert DataStore lazy objects to Series
    if not isinstance(left, pd.Series):
        left = get_series(left)

    # Suppress FutureWarning about None vs nan mismatch in pandas equality testing
    # This occurs when comparing Series with mixed null representations (None vs np.nan)
    # Both represent missing values and should be treated as equal for our tests
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mismatched null-like values.*found', category=FutureWarning)
        pd.testing.assert_series_equal(
            left,
            right,
            check_dtype=check_dtype,
            check_index_type=check_index_type,
            check_series_type=check_series_type,
            check_names=check_names,
            check_exact=check_exact,
            check_datetimelike_compat=check_datetimelike_compat,
            check_categorical=check_categorical,
            check_category_order=check_category_order,
            check_freq=check_freq,
            check_flags=check_flags,
            rtol=rtol,
            atol=atol,
            obj=obj,
            check_index=check_index,
            check_like=check_like,
        )


# =============================================================================
# chDB dtype handling (see test_chdb_dtype_differences.py)
# =============================================================================
# FIXED in recent chDB versions:
# - float64 with NaN: now correctly preserves float64
# - Integer columns with None: now correctly preserves original dtype
# - datetime64[ns]: now correctly preserves naive datetime (no timezone added)
#
# The _normalize_chdb_dtypes() function below is kept for backward compatibility
# but should no longer be necessary with current chDB versions.
# =============================================================================


def _normalize_chdb_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is only used for tests that we already know are chDB issues.
    Use this function means there is a known issue with chDB!!

    Normalize chDB output dtypes to standard pandas dtypes.

    Converts:
    - Float64 (nullable) → float64
    - Int64 (nullable) → int64 or float64 (if has NA)
    - Timezone-aware datetime → timezone-naive datetime

    This allows for value comparison when dtype differences are expected.

    Args:
        df: DataFrame potentially containing chDB nullable dtypes

    Returns:
        DataFrame with standard pandas dtypes
    """
    result = df.copy()
    for col in result.columns:
        dtype_str = str(result[col].dtype)

        # Handle nullable float (Float64 → float64)
        if dtype_str == "Float64":
            result[col] = result[col].astype("float64")

        # Handle nullable int (Int64 → int64 or float64 if has NA)
        elif dtype_str == "Int64":
            if result[col].isna().any():
                result[col] = result[col].astype("float64")
            else:
                result[col] = result[col].astype("int64")

        # Handle timezone-aware datetime → naive
        elif hasattr(result[col].dtype, "tz") and result[col].dtype.tz is not None:
            result[col] = result[col].dt.tz_localize(None)

    return result


def _are_dtypes_nullable_equivalent(dtype1, dtype2) -> bool:
    """
    Check if two dtypes are equivalent when ignoring nullable vs non-nullable distinction.

    This handles the common case where chDB/SQL execution returns numpy dtypes (int64, float64, bool)
    but pandas uses nullable dtypes (Int64, Float64, boolean).

    Equivalent pairs:
    - int64 ↔ Int64 (nullable integer)
    - int32 ↔ Int32
    - int16 ↔ Int16
    - int8 ↔ Int8
    - uint64 ↔ UInt64
    - uint32 ↔ UInt32
    - uint16 ↔ UInt16
    - uint8 ↔ UInt8
    - float64 ↔ Float64 (nullable float)
    - float32 ↔ Float32
    - bool ↔ boolean (nullable boolean)
    - uint64 ↔ int64 (count operations return uint64 in chDB)
    - uint8 ↔ bool (chDB string operations return uint8 for boolean results)
    """
    str1, str2 = str(dtype1).lower(), str(dtype2).lower()

    # Exact match
    if str1 == str2:
        return True

    # Nullable integer equivalents (Int64 vs int64, etc.)
    nullable_int_pairs = {
        ('int64', 'int64'),  # Int64 lowercase == int64
        ('int32', 'int32'),
        ('int16', 'int16'),
        ('int8', 'int8'),
        ('uint64', 'uint64'),
        ('uint32', 'uint32'),
        ('uint16', 'uint16'),
        ('uint8', 'uint8'),
    }

    # Check if one is nullable (capitalized) and other is numpy dtype
    # pandas nullable dtypes: Int64, Int32, Float64, etc. (capitalized)
    # numpy dtypes: int64, int32, float64, etc. (lowercase)
    dtype1_str = str(dtype1)
    dtype2_str = str(dtype2)

    # Int64 vs int64, Float64 vs float64, etc.
    if dtype1_str.lower() == dtype2_str.lower():
        return True

    # bool vs boolean
    if {str1, str2} == {'bool', 'boolean'}:
        return True

    # uint64 vs int64 (chDB count operations return uint64)
    if {str1, str2} == {'uint64', 'int64'}:
        return True

    # uint8 vs bool (chDB string operations like startswith/endswith return uint8)
    if {str1, str2} == {'uint8', 'bool'}:
        return True

    if {str1, str2} == {'str', 'object'}:
        return True

    # datetime64[ns] vs datetime64[us] (pandas 3.0 uses microseconds by default)
    if str1.startswith('datetime64') and str2.startswith('datetime64'):
        return True

    # timedelta64 with different resolutions
    if str1.startswith('timedelta64') and str2.startswith('timedelta64'):
        return True

    # int32 vs int64, uint32 vs uint64 (common in SQL vs pandas)
    if str1.replace('32', '64') == str2 or str2.replace('32', '64') == str1:
        return True

    # period[M] vs int32/int64 (dt.month returns different types)
    if 'period' in str1 or 'period' in str2:
        other = str2 if 'period' in str1 else str1
        if other in ('int32', 'int64', 'uint32', 'uint64'):
            return True

    return False


def assert_datastore_equals_pandas(
    ds_result,
    pd_result: pd.DataFrame,
    check_column_order: bool = True,
    check_row_order: bool = True,  # Default True: chDB v4.0.0b5 _row_id provides deterministic order
    check_nullable_dtype: bool = False,
    check_index: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
) -> None:
    """
    Complete comparison between DataStore result and pandas DataFrame.

    Compares: column names (with order), data values, row order, and dtypes.

    Args:
        ds_result: DataStore result (DataStore, LazySeries, or similar)
        pd_result: Expected pandas DataFrame or Series
        check_column_order: If True, column order must match exactly
        check_row_order: If True (default), row order must match exactly.
                        Since chDB v4.0.0b5, row order is deterministic for DataFrame sources
                        using the built-in _row_id virtual column.
                        Set to False only for operations with inherently undefined order:
                        - groupby() without subsequent sort
                        - merge/join (SQL join order is implementation-dependent)
                        - value_counts() (tie order is undefined)
                        - drop_duplicates(keep=False)
        check_nullable_dtype: If True, treat nullable and non-nullable dtypes as different
                             (e.g., Int64 != int64). If False, treat them as equivalent.
                             Common equivalents: int64↔Int64, float64↔Float64, bool↔boolean, uint64↔int64
        check_index: If True, verify index matches (DataStore typically doesn't preserve index)
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison
        msg: Additional message to include in assertion errors

    Raises:
        AssertionError: If any comparison fails

    Example:
        # pandas operations
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        pd_result = pd_df[pd_df['age'] > 20].sort_values('name')

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        ds_result = ds_df[ds_df['age'] > 20].sort_values('name')

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

        # Allow nullable dtype differences (Int64 vs int64, etc.)
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)
    """
    prefix = f"{msg}: " if msg else ""

    # Handle Series comparison
    if isinstance(pd_result, pd.Series):
        _assert_series_equals(ds_result, pd_result, check_row_order, check_nullable_dtype, rtol, atol, prefix)
        return

    # Get DataStore columns
    ds_columns = list(ds_result.columns)
    pd_columns = list(pd_result.columns)

    # 1. Compare column names
    if check_column_order:
        assert ds_columns == pd_columns, (
            f"{prefix}Column names or order don't match.\n"
            f"DataStore columns: {ds_columns}\n"
            f"Pandas columns:    {pd_columns}"
        )
    else:
        assert set(ds_columns) == set(pd_columns), (
            f"{prefix}Column names don't match.\n"
            f"DataStore columns: {set(ds_columns)}\n"
            f"Pandas columns:    {set(pd_columns)}"
        )

    # 2. Compare row count
    ds_len = len(ds_result)
    pd_len = len(pd_result)
    assert ds_len == pd_len, (
        f"{prefix}Row count doesn't match.\n" f"DataStore: {ds_len} rows\n" f"Pandas:    {pd_len} rows"
    )

    # 3. Compare data values for each column
    columns_to_check = pd_columns if check_column_order else sorted(pd_columns)

    for col in columns_to_check:
        ds_values = np.asarray(ds_result[col].values)
        pd_values = np.asarray(pd_result[col].values)

        if not check_row_order:
            # Sort both for comparison when order doesn't matter
            # Handle mixed types that can't be directly sorted
            try:
                ds_values = np.sort(ds_values)
                pd_values = np.sort(pd_values)
            except TypeError:
                # Mixed types (e.g., float and str) - convert to string for comparison
                ds_values = np.sort(np.array([str(x) if pd.notna(x) else '' for x in ds_values]))
                pd_values = np.sort(np.array([str(x) if pd.notna(x) else '' for x in pd_values]))

        _assert_array_equal(ds_values, pd_values, f"{prefix}Column '{col}' values don't match", rtol, atol)

    # 4. Check dtypes (always enforced)
    for col in columns_to_check:
        ds_col = ds_result[col]
        pd_col = pd_result[col]
        # Handle duplicate column names - when df[col] returns DataFrame instead of Series
        if hasattr(ds_col, 'dtype'):
            ds_dtype = ds_col.dtype
        elif hasattr(ds_col, 'dtypes'):
            # Multiple columns with same name - use dtypes (Series of dtypes)
            ds_dtype = ds_col.dtypes
        else:
            continue  # Skip dtype check for unsupported types

        if hasattr(pd_col, 'dtype'):
            pd_dtype = pd_col.dtype
        elif hasattr(pd_col, 'dtypes'):
            pd_dtype = pd_col.dtypes
        else:
            continue

        # For Series of dtypes, compare element-wise
        if hasattr(ds_dtype, '__iter__') and hasattr(pd_dtype, '__iter__'):
            for ds_d, pd_d in zip(ds_dtype, pd_dtype):
                if ds_d != pd_d:
                    # Allow bool vs uint8 mismatch (common in chDB)
                    if {str(ds_d), str(pd_d)} == {'bool', 'uint8'}:
                        continue
                    # Allow nullable dtype equivalents if check_nullable_dtype=False
                    if not check_nullable_dtype and _are_dtypes_nullable_equivalent(ds_d, pd_d):
                        continue
                    assert False, (
                        f"{prefix}Column '{col}' dtype doesn't match.\n"
                        f"DataStore dtype: {ds_dtype}\n"
                        f"Pandas dtype:    {pd_dtype}"
                    )
        else:
            if ds_dtype != pd_dtype:
                # Allow bool vs uint8 mismatch (common in chDB)
                if {str(ds_dtype), str(pd_dtype)} == {'bool', 'uint8'}:
                    continue
                # Allow nullable dtype equivalents if check_nullable_dtype=False
                if not check_nullable_dtype and _are_dtypes_nullable_equivalent(ds_dtype, pd_dtype):
                    continue
                assert False, (
                    f"{prefix}Column '{col}' dtype doesn't match.\n"
                    f"DataStore dtype: {ds_dtype}\n"
                    f"Pandas dtype:    {pd_dtype}"
                )


def _assert_series_equals(
    ds_result,
    pd_result: pd.Series,
    check_order: bool,
    check_nullable_dtype: bool,
    rtol: float,
    atol: float,
    prefix: str,
) -> None:
    """Helper to compare DataStore Series-like result with pandas Series."""
    ds_values = np.asarray(ds_result.values)
    pd_values = np.asarray(pd_result.values)

    # Compare length
    assert len(ds_values) == len(pd_values), (
        f"{prefix}Series length doesn't match.\n" f"DataStore: {len(ds_values)}\n" f"Pandas:    {len(pd_values)}"
    )

    if not check_order:
        # Use pandas Series sorting to properly handle NA values
        # np.sort fails on arrays with pd.NA ("boolean value of NA is ambiguous")
        try:
            ds_values = pd.Series(ds_values).sort_values(na_position='last').values
            pd_values = pd.Series(pd_values).sort_values(na_position='last').values
        except TypeError:
            # Fallback for mixed types that can't be sorted
            ds_values = np.array([str(x) if pd.notna(x) else '' for x in ds_values])
            pd_values = np.array([str(x) if pd.notna(x) else '' for x in pd_values])
            ds_values = np.sort(ds_values)
            pd_values = np.sort(pd_values)

    _assert_array_equal(ds_values, pd_values, f"{prefix}Series values don't match", rtol, atol)

    # Always check dtype
    ds_dtype = ds_result.dtype
    pd_dtype = pd_result.dtype
    if ds_dtype != pd_dtype:
        # Allow nullable dtype equivalents if check_nullable_dtype=False
        if not check_nullable_dtype and _are_dtypes_nullable_equivalent(ds_dtype, pd_dtype):
            pass  # Equivalent, skip assertion
        else:
            assert False, (
                f"{prefix}Series dtype doesn't match.\n" f"DataStore dtype: {ds_dtype}\n" f"Pandas dtype:    {pd_dtype}"
            )


def _assert_array_equal(
    ds_values: np.ndarray,
    pd_values: np.ndarray,
    err_msg: str,
    rtol: float,
    atol: float,
) -> None:
    """Helper to compare two numpy arrays with proper handling of different dtypes."""
    # Check for floating point types - use allclose for tolerance
    if np.issubdtype(ds_values.dtype, np.floating) or np.issubdtype(pd_values.dtype, np.floating):
        # Handle NaN values
        ds_nan_mask = pd.isna(ds_values)
        pd_nan_mask = pd.isna(pd_values)

        np.testing.assert_array_equal(ds_nan_mask, pd_nan_mask, err_msg=f"{err_msg} (NaN positions differ)")

        # Compare non-NaN values with tolerance
        if not np.all(ds_nan_mask):
            np.testing.assert_allclose(
                ds_values[~ds_nan_mask].astype(float),
                pd_values[~pd_nan_mask].astype(float),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
    else:
        # For non-float types, use exact comparison
        # Handle object dtype (strings, mixed types)
        if ds_values.dtype == object or pd_values.dtype == object:
            # Convert to string for comparison to handle None/NaN consistently
            ds_str = np.array([str(x) if pd.notna(x) else None for x in ds_values])
            pd_str = np.array([str(x) if pd.notna(x) else None for x in pd_values])
            np.testing.assert_array_equal(ds_str, pd_str, err_msg=err_msg)
        else:
            np.testing.assert_array_equal(ds_values, pd_values, err_msg=err_msg)


def get_dataframe(ds_result) -> pd.DataFrame:
    """
    Get a pandas DataFrame from a DataStore result using Duck Typing.

    Triggers execution implicitly by accessing standard properties.
    Handles various input types: DataStore, DataFrame, Series, numpy array.

    Args:
        ds_result: DataStore result (DataStore, DataFrame, Series, numpy array, or similar)

    Returns:
        pandas DataFrame

    Example:
        df = get_dataframe(ds_result)
    """
    # If already a pandas DataFrame, return it directly
    if isinstance(ds_result, pd.DataFrame):
        return ds_result

    # If pandas Series, convert to DataFrame
    if isinstance(ds_result, pd.Series):
        return ds_result.to_frame()

    # If numpy array, wrap in DataFrame
    if isinstance(ds_result, np.ndarray):
        if ds_result.ndim == 1:
            return pd.DataFrame(ds_result)
        return pd.DataFrame(ds_result)

    # If None, raise error
    if ds_result is None:
        raise ValueError("Cannot convert None to DataFrame")

    # For DataStore and similar objects, try accessing _get_df() first
    try:
        df = ds_result._get_df()
        if isinstance(df, pd.DataFrame):
            return df
        if isinstance(df, pd.Series):
            return df.to_frame()
        # Recursively handle nested DataStore
        return get_dataframe(df)
    except (AttributeError, TypeError):
        pass

    # Fall back to building DataFrame from columns
    try:
        columns = list(ds_result.columns)
        data = {}
        for col in columns:
            data[col] = np.asarray(ds_result[col].values)
        return pd.DataFrame(data, columns=columns)
    except (AttributeError, TypeError):
        pass

    # Last resort: try to convert using pd.DataFrame constructor
    return pd.DataFrame(ds_result)


def get_series(ds_result) -> pd.Series:
    """
    Get a pandas Series from a DataStore Series-like result using Duck Typing.

    Triggers execution implicitly by accessing standard properties.
    Preserves index information when possible.

    Args:
        ds_result: DataStore Series-like result (ColumnExpr, pandas Series, numpy array,
                   DataStore, etc.)

    Returns:
        pandas Series

    Example:
        series = get_series(ds_result)
    """
    # If already a pandas Series, return it directly
    if isinstance(ds_result, pd.Series):
        return ds_result

    # If numpy array, wrap in Series
    if isinstance(ds_result, np.ndarray):
        return pd.Series(ds_result)

    # If pandas DataFrame, return first column as Series
    if isinstance(ds_result, pd.DataFrame):
        if len(ds_result.columns) == 1:
            return ds_result.iloc[:, 0]
        raise ValueError(f"Cannot convert DataFrame with {len(ds_result.columns)} columns to Series")

    # For DataStore and ColumnExpr, use len() to trigger execution, then get the result
    # We try different approaches to preserve as much information (especially index) as possible
    try:
        # Trigger execution by accessing len
        _ = len(ds_result)

        # Try _get_df() first (DataStore) - preserves index for groupby results
        try:
            df = ds_result._get_df()
            if isinstance(df, pd.Series):
                return df
            if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
                return df.iloc[:, 0]
            # For groupby results that return DataStore wrapping Series
            return df
        except AttributeError:
            pass

        # Try _execute() for ColumnExpr - preserves index information
        try:
            executed = ds_result._execute()
            if isinstance(executed, pd.Series):
                return executed
            if isinstance(executed, pd.DataFrame) and len(executed.columns) == 1:
                return executed.iloc[:, 0]
            if isinstance(executed, np.ndarray):
                name = getattr(ds_result, 'name', None)
                return pd.Series(executed, name=name)
            return pd.Series(executed)
        except AttributeError:
            pass
    except (TypeError, AttributeError):
        pass

    # Fallback: access .values to trigger execution
    # This works for ColumnExpr objects but may lose index information
    try:
        values = np.asarray(ds_result.values)
        name = getattr(ds_result, 'name', None)
        return pd.Series(values, name=name)
    except AttributeError:
        pass

    # Last resort: try to convert to Series directly
    return pd.Series(ds_result)


def assert_datastore_equals_pandas_chdb_compat(
    ds_result,
    pd_result: pd.DataFrame,
    check_column_order: bool = True,
    check_row_order: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
) -> None:
    """
    Compare DataStore result with pandas, tolerating chDB dtype differences.

    This is a convenience wrapper around assert_datastore_equals_pandas that:
    1. Normalizes chDB nullable dtypes (Float64 → float64, etc.)

    Use this for tests affected by chDB's dtype behavior.
    See test_chdb_dtype_differences.py for documentation of these differences.

    Uses Duck Typing principle: triggers execution implicitly via standard properties.
    NEVER uses hasattr checks or explicit _execute() calls.

    Args:
        ds_result: DataStore result (DataStore, LazySeries, or similar)
        pd_result: Expected pandas DataFrame or Series
        check_column_order: If True, column order must match exactly
        check_row_order: If True, row order must match exactly
        rtol: Relative tolerance for float comparison
        atol: Absolute tolerance for float comparison
        msg: Additional message to include in assertion errors

    Example:
        # When dtype differences are acceptable
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)
    """
    # Duck Typing: use get_dataframe() helper to trigger execution implicitly
    # This avoids hasattr checks and works uniformly for all DataStore types
    ds_df = get_dataframe(ds_result)

    # Normalize chDB dtypes
    ds_df_normalized = _normalize_chdb_dtypes(ds_df)

    # Compare with dtype checking disabled
    assert_datastore_equals_pandas(
        ds_df_normalized,
        pd_result,
        check_column_order=check_column_order,
        check_row_order=check_row_order,
        rtol=rtol,
        atol=atol,
        msg=msg,
    )


def normalize_dataframe_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame for comparison, handling chDB dtype differences.

    Public wrapper for _normalize_chdb_dtypes.

    Args:
        df: DataFrame to normalize

    Returns:
        Normalized DataFrame with standard pandas dtypes
    """
    return _normalize_chdb_dtypes(df)


def get_value(ds_result):
    """
    Get the executed value from a DataStore result using Duck Typing.

    Unlike get_series(), this function returns the result as-is (scalar, Series, or DataFrame)
    without forcing conversion to Series.

    Args:
        ds_result: DataStore result (ColumnExpr, DataStore, scalar, etc.)

    Returns:
        The executed result (may be scalar, Series, DataFrame, or numpy array)
    """
    # If already a primitive type, return as-is
    if isinstance(ds_result, (int, float, str, bool, type(None))):
        return ds_result

    # If already pandas type, return as-is
    if isinstance(ds_result, (pd.Series, pd.DataFrame)):
        return ds_result

    # If numpy array, return as-is
    if isinstance(ds_result, np.ndarray):
        return ds_result

    # Try _execute() for lazy objects
    try:
        return ds_result._execute()
    except AttributeError:
        pass

    # Try _get_df() for DataStore
    try:
        return ds_result._get_df()
    except AttributeError:
        pass

    # Return as-is if nothing else works
    return ds_result
