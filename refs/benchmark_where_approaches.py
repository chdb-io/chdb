#!/usr/bin/env python3
"""
Benchmark different approaches for where/mask operations.

Approaches:
1. Pure Pandas (baseline)
2. Current DataStore (Pandas fallback for type safety)
3. SQL CASE WHEN + Python post-processing (experimental)
4. Pure SQL CASE WHEN (no type preservation - for speed reference only)
"""

import time
import pandas as pd
import numpy as np
import chdb
from typing import Callable, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datastore import DataStore

# Global connection for reuse
_conn = None


def get_conn():
    """Get or create a reusable chDB connection."""
    global _conn
    if _conn is None:
        _conn = chdb.connect(":memory:")
    return _conn


def query_with_conn(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute query using persistent connection.

    Note: chDB's Python() table function requires DataFrame to be
    in local scope when conn.query() is called.
    """
    conn = get_conn()
    return conn.query(sql, 'DataFrame')


def create_test_data(n_rows: int) -> pd.DataFrame:
    """Create test DataFrame with mixed types."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            'int_col': np.random.randint(0, 100, n_rows),
            'float_col': np.random.random(n_rows) * 100,
            'str_col': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'bool_col': np.random.choice([True, False], n_rows),
        }
    )


def benchmark(func: Callable, name: str, iterations: int = 5) -> Dict[str, Any]:
    """Run benchmark and return timing stats."""
    times = []
    result = None
    for i in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        'name': name,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'result': result,
    }


# ============================================================================
# Approach 1: Pure Pandas (baseline)
# ============================================================================
def approach_pandas(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """Pure pandas where operation."""
    return df.where(df['int_col'] > 50, other)


# ============================================================================
# Approach 2: Current DataStore (uses Pandas fallback for type safety)
# ============================================================================
def approach_datastore_current(ds: DataStore, other=0) -> pd.DataFrame:
    """Current DataStore implementation - falls back to Pandas for type safety."""
    result = ds.where(ds['int_col'] > 50, other)
    return result._execute()


# ============================================================================
# Approach 3: SQL CASE WHEN + Python post-processing
# ============================================================================
def get_sql_other_for_dtype(dtype, other) -> str:
    """Get SQL-compatible 'other' value based on column dtype."""
    if pd.api.types.is_bool_dtype(dtype):
        if isinstance(other, bool):
            return 'true' if other else 'false'
        else:
            return 'false'  # Default for bool columns
    elif pd.api.types.is_integer_dtype(dtype):
        return str(int(other) if isinstance(other, (int, float, bool)) else 0)
    elif pd.api.types.is_float_dtype(dtype):
        return str(float(other) if isinstance(other, (int, float, bool)) else 0.0)
    elif pd.api.types.is_string_dtype(dtype) or dtype == object:
        if isinstance(other, str):
            return f"'{other}'"
        else:
            return f"'{other}'"  # Convert to string
    else:
        return str(other)


def approach_sql_with_postprocess(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """
    Execute SQL CASE WHEN, then fix types in Python.

    Strategy:
    1. Run SQL CASE WHEN (fast, but loses type info)
    2. Get the condition mask
    3. For columns where type differs, manually restore correct types
    """
    # Step 1: Execute SQL CASE WHEN for all columns
    columns = df.columns.tolist()
    case_whens = []
    for col in columns:
        # SQL CASE WHEN: keep original if condition, else use type-compatible 'other'
        other_sql = get_sql_other_for_dtype(df[col].dtype, other)
        case_whens.append(f'CASE WHEN "int_col" > 50 THEN "{col}" ELSE {other_sql} END AS "{col}"')

    sql = f"SELECT {', '.join(case_whens)} FROM Python(df)"
    conn = get_conn()
    sql_result = conn.query(sql, 'DataFrame')

    # Step 2: Get condition mask (need to evaluate condition)
    mask = df['int_col'] > 50

    # Step 3: Post-process to restore correct types for mixed-type columns
    # For rows where mask is False, we need 'other' with its original type
    # For rows where mask is True, we need original value with its original type
    result = pd.DataFrame(index=sql_result.index)
    for col in columns:
        col_dtype = df[col].dtype

        # Check if this column needs type restoration
        if col_dtype == bool and not isinstance(other, bool):
            # Bool column with non-bool other -> need object dtype with mixed types
            result[col] = sql_result[col].copy()
            # Where mask is True, restore original bool values
            result.loc[mask, col] = df.loc[mask, col]
            # Where mask is False, use other with correct type
            result.loc[~mask, col] = other
        else:
            # No type mixing needed, use SQL result directly
            result[col] = sql_result[col]

    return result


# ============================================================================
# Approach 4: Pure SQL CASE WHEN (no type preservation - speed reference)
# ============================================================================
def approach_pure_sql(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """
    Pure SQL CASE WHEN - fastest but doesn't preserve types.
    This is for reference only - results won't match pandas exactly.
    """
    columns = df.columns.tolist()
    case_whens = []
    for col in columns:
        other_sql = get_sql_other_for_dtype(df[col].dtype, other)
        case_whens.append(f'CASE WHEN "int_col" > 50 THEN "{col}" ELSE {other_sql} END AS "{col}"')

    sql = f"SELECT {', '.join(case_whens)} FROM Python(df)"
    conn = get_conn()
    return conn.query(sql, 'DataFrame')


# ============================================================================
# Approach 5: Optimized SQL + minimal post-processing
# ============================================================================
def approach_sql_optimized_postprocess(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """
    Optimized approach:
    - Use SQL for columns that don't need type restoration
    - Only do Python post-processing for columns that need mixed types
    """
    columns = df.columns.tolist()

    # Identify which columns need type restoration
    cols_need_restore = []
    cols_sql_only = []
    for col in columns:
        col_dtype = df[col].dtype
        if col_dtype == bool and not isinstance(other, bool):
            cols_need_restore.append(col)
        else:
            cols_sql_only.append(col)

    # Build SQL - for cols that need restore, we still run SQL but will override
    case_whens = []
    for col in columns:
        other_sql = get_sql_other_for_dtype(df[col].dtype, other)
        case_whens.append(f'CASE WHEN "int_col" > 50 THEN "{col}" ELSE {other_sql} END AS "{col}"')

    sql = f"SELECT {', '.join(case_whens)} FROM Python(df)"
    conn = get_conn()
    sql_result = conn.query(sql, 'DataFrame')

    # If no columns need restoration, return SQL result directly
    if not cols_need_restore:
        return sql_result

    # Only compute mask if we need post-processing
    mask = df['int_col'] > 50

    # Post-process only columns that need it
    for col in cols_need_restore:
        # Create object dtype column with correct mixed types
        new_col = np.empty(len(df), dtype=object)
        new_col[mask] = df.loc[mask, col].values
        new_col[~mask] = other
        sql_result[col] = new_col

    return sql_result


# ============================================================================
# Approach 6: SQL generates single column, replace into DataFrame
# ============================================================================
def approach_sql_single_column(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """
    Use SQL to generate each column individually, then replace into DataFrame.
    This reduces serialization overhead by only returning the computed column.
    """
    result = df.copy()
    conn = get_conn()

    for col in df.columns:
        col_dtype = df[col].dtype
        other_sql = get_sql_other_for_dtype(col_dtype, other)

        # Generate single column via SQL
        sql = f'SELECT CASE WHEN "int_col" > 50 THEN "{col}" ELSE {other_sql} END AS "{col}" FROM Python(df)'
        col_result = conn.query(sql, 'DataFrame')

        # Check if this column needs type restoration (bool col with non-bool other)
        if col_dtype == bool and not isinstance(other, bool):
            # Need to create object dtype with mixed types
            mask = df['int_col'] > 50
            new_col = np.empty(len(df), dtype=object)
            new_col[mask] = df.loc[mask, col].values
            new_col[~mask] = other
            result[col] = new_col
        else:
            result[col] = col_result[col].values

    return result


# ============================================================================
# Approach 7: SQL generates only the mask, then use pandas where (WITH ORDER BY)
# ============================================================================
def approach_sql_mask_only(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """
    Use SQL only to evaluate the condition (generate mask), then use pandas.where().
    This tests if SQL condition evaluation is faster than pandas.

    NOTE: Python() table function does NOT preserve row order!
    We must add row index to DataFrame BEFORE SQL and ORDER BY to maintain correct alignment.
    rowNumberInAllBlocks() doesn't work - must use pre-added index.
    """
    conn = get_conn()

    # Must add index to DataFrame before SQL (rowNumberInAllBlocks doesn't preserve original order)
    df_with_idx = df.copy()
    df_with_idx['__row_idx__'] = range(len(df))

    sql = '''
    SELECT "int_col" > 50 AS mask
    FROM Python(df_with_idx)
    ORDER BY __row_idx__
    '''
    mask_result = conn.query(sql, 'DataFrame')
    mask = mask_result['mask'].astype(bool)

    # Use pandas where with the SQL-computed mask
    return df.where(mask, other)


# ============================================================================
# Approach 8: Only compute condition-dependent columns via SQL
# ============================================================================
def approach_sql_affected_columns_only(df: pd.DataFrame, other=0) -> pd.DataFrame:
    """
    Only use SQL for columns that will actually be affected by the condition.
    For 'where' with condition on int_col > 50:
    - Rows where int_col <= 50 get replaced with 'other'
    - We can compute which rows need replacement, then do targeted updates
    """
    result = df.copy()
    conn = get_conn()

    # Get the mask from pandas (faster than SQL for this simple comparison)
    mask = df['int_col'] > 50

    # For columns that need type mixing (bool col + non-bool other), handle specially
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == bool and not isinstance(other, bool):
            # Need object dtype for mixed types
            new_col = np.empty(len(df), dtype=object)
            new_col[mask] = df.loc[mask, col].values
            new_col[~mask] = other
            result[col] = new_col
        else:
            # Just use pandas assignment for non-mixed-type columns
            result.loc[~mask, col] = other

    return result


# ============================================================================
# Verification helper
# ============================================================================
def verify_results(pandas_result: pd.DataFrame, other_result: pd.DataFrame, name: str) -> bool:
    """Verify that results match pandas (both values and types)."""
    try:
        # Check shape
        if pandas_result.shape != other_result.shape:
            print(f"  {name}: Shape mismatch {pandas_result.shape} vs {other_result.shape}")
            return False

        # Check each column
        all_match = True
        for col in pandas_result.columns:
            pd_col = pandas_result[col]
            other_col = other_result[col]

            # Check values
            values_match = True
            for i in range(len(pd_col)):
                if pd_col.iloc[i] != other_col.iloc[i]:
                    values_match = False
                    break

            # Check types of each value (important for object dtype)
            types_match = True
            if pd_col.dtype == object:
                for i in range(len(pd_col)):
                    if type(pd_col.iloc[i]) != type(other_col.iloc[i]):
                        types_match = False
                        break

            if not values_match or not types_match:
                all_match = False
                print(f"  {name} - {col}: values_match={values_match}, types_match={types_match}")

        return all_match
    except Exception as e:
        print(f"  {name}: Error during verification: {e}")
        return False


# ============================================================================
# Main benchmark
# ============================================================================
def warmup_chdb():
    """Warmup chdb connection to eliminate connection setup overhead."""
    dummy_df = pd.DataFrame({'a': [1, 2, 3]})
    conn = get_conn()
    for _ in range(3):
        conn.query("SELECT * FROM Python(dummy_df)", 'DataFrame')
    print("chDB connection warmed up.")


def run_benchmark(n_rows: int, other=0):
    """Run all benchmarks for given row count."""
    print(f"\n{'='*70}")
    print(f"Benchmark: {n_rows:,} rows, other={other} (type: {type(other).__name__})")
    print('=' * 70)

    # Create test data
    df = create_test_data(n_rows)
    ds = DataStore(df)

    # Warmup: run each approach once before timing to eliminate cold start effects
    _ = approach_pandas(df, other)
    _ = approach_datastore_current(ds, other)
    _ = approach_sql_with_postprocess(df, other)
    _ = approach_pure_sql(df, other)
    _ = approach_sql_optimized_postprocess(df, other)
    _ = approach_sql_single_column(df, other)
    _ = approach_sql_mask_only(df, other)
    _ = approach_sql_affected_columns_only(df, other)

    # Run benchmarks
    results = []

    # 1. Pure Pandas (baseline)
    r1 = benchmark(lambda: approach_pandas(df, other), "1. Pure Pandas")
    results.append(r1)
    pandas_result = r1['result']

    # 2. Current DataStore
    r2 = benchmark(lambda: approach_datastore_current(ds, other), "2. DataStore (current)")
    results.append(r2)

    # 3. SQL + post-processing
    r3 = benchmark(lambda: approach_sql_with_postprocess(df, other), "3. SQL + post-process")
    results.append(r3)

    # 4. Pure SQL (no type preservation)
    r4 = benchmark(lambda: approach_pure_sql(df, other), "4. Pure SQL (no types)")
    results.append(r4)

    # 5. Optimized SQL + minimal post-processing
    r5 = benchmark(lambda: approach_sql_optimized_postprocess(df, other), "5. SQL + optimized post")
    results.append(r5)

    # 6. SQL single column replacement
    r6 = benchmark(lambda: approach_sql_single_column(df, other), "6. SQL single column")
    results.append(r6)

    # 7. SQL mask only + pandas where
    r7 = benchmark(lambda: approach_sql_mask_only(df, other), "7. SQL mask + pd.where")
    results.append(r7)

    # 8. Pure pandas with targeted updates (baseline for comparison)
    r8 = benchmark(lambda: approach_sql_affected_columns_only(df, other), "8. Pandas targeted")
    results.append(r8)

    # Print timing results
    print("\nTiming Results:")
    print("-" * 70)
    print(f"{'Approach':<30} {'Mean (ms)':>12} {'Std':>10} {'vs Pandas':>12}")
    print("-" * 70)

    baseline = results[0]['mean_ms']
    for r in results:
        ratio = r['mean_ms'] / baseline
        ratio_str = f"{ratio:.2f}x" if ratio >= 1 else f"{1/ratio:.2f}x faster"
        print(f"{r['name']:<30} {r['mean_ms']:>10.2f}ms {r['std_ms']:>8.2f}ms {ratio_str:>12}")

    # Verify correctness
    print("\nCorrectness Verification:")
    print("-" * 70)
    for r in results[1:]:  # Skip pandas baseline
        is_correct = verify_results(pandas_result, r['result'], r['name'])
        status = "✓ MATCH" if is_correct else "✗ MISMATCH"
        print(f"  {r['name']}: {status}")

    return results


def main():
    """Run benchmarks for different data sizes and 'other' values."""

    # Warmup chdb connection first
    warmup_chdb()

    # Test with different row counts
    row_counts = [100_000, 500_000, 1_000_000]

    # Test with different 'other' values
    # Note: other=False with mixed column types causes chDB Variant type errors
    # so we only test other=0 for now
    other_values = [
        0,  # int - causes type mixing with bool columns
        # False,  # bool - causes Variant type errors with mixed columns
    ]

    all_results = {}

    for other in other_values:
        for n_rows in row_counts:
            key = f"{n_rows}_{other}"
            all_results[key] = run_benchmark(n_rows, other)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        """
Key Findings:
- Pure SQL is fastest but loses type information (bool -> int conversion)
- SQL + post-processing can restore types but adds overhead
- The question is: can we beat pure Pandas with SQL + post-processing?

Type Preservation:
- For other=0 (int), bool columns become object dtype with mixed int/bool
- For other=False (bool), types should match if all columns are bool
"""
    )


if __name__ == "__main__":
    main()
