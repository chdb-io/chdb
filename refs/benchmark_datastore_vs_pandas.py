#!/usr/bin/env python3
"""
Benchmark: DataStore (chDB lazy mode) vs Pure Pandas

This benchmark compares the performance of:
1. Pure Pandas operations (eager evaluation - each step executes immediately)
2. DataStore lazy execution (multiple operations merged into single SQL via chDB)

Key insight: DataStore's lazy execution can merge multiple operations (filter, sort,
groupby, etc.) into a single SQL query, which is especially advantageous for:
- Complex multi-step pipelines
- Chained filter operations
- Filter + GroupBy + Sort patterns

Operations tested:
- Single operations (filter, sort, groupby)
- Multi-step pipelines (where DataStore shines)
"""

import time
import tempfile
import os
import json
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict

# Import DataStore
from datastore import DataStore
from datastore.config import (
    enable_profiling,
    disable_profiling,
    get_profiler,
    reset_profiler,
)


def get_git_info() -> Dict[str, str]:
    """Get current git branch, commit hash, and other info."""
    git_info = {
        'branch': 'unknown',
        'commit_hash': 'unknown',
        'commit_short': 'unknown',
        'commit_message': 'unknown',
        'is_dirty': False,
    }
    try:
        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()

        # Get full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
            git_info['commit_short'] = result.stdout.strip()[:7]

        # Get commit message (first line)
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%s'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['commit_message'] = result.stdout.strip()[:80]

        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['is_dirty'] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Warning: Could not get git info: {e}")

    return git_info


def create_benchmark_output_dir(benchmark_name: str, base_dir: str = None) -> tuple:
    """
    Create a timestamped output directory for benchmark results.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'datastore_vs_pandas')
        base_dir: Base directory for all benchmark results

    Returns:
        tuple: (output_dir_path, metadata_dict)
    """
    if base_dir is None:
        # Default to refs/benchmark_results/ relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, 'benchmark_results', benchmark_name)

    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Get timestamp and git info
    timestamp = datetime.now()
    git_info = get_git_info()

    # Create directory name: YYYYMMDD_HHMMSS_branch_commit
    branch_safe = git_info['branch'].replace('/', '_').replace('\\', '_')[:20]
    dirty_marker = '_dirty' if git_info['is_dirty'] else ''
    dir_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{branch_safe}_{git_info['commit_short']}{dirty_marker}"

    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare metadata
    metadata = {
        'timestamp': timestamp.isoformat(),
        'timestamp_unix': timestamp.timestamp(),
        'git_branch': git_info['branch'],
        'git_commit_hash': git_info['commit_hash'],
        'git_commit_short': git_info['commit_short'],
        'git_commit_message': git_info['commit_message'],
        'git_is_dirty': git_info['is_dirty'],
        'output_dir': output_dir,
    }

    return output_dir, metadata


@dataclass
class BenchmarkResult:
    operation: str
    data_size: int
    pandas_time: float
    datastore_time: float
    profile_data: Optional[Dict[str, float]] = field(default_factory=dict)  # Profiling breakdown

    @property
    def fastest(self) -> str:
        times = {'Pandas': self.pandas_time, 'DataStore': self.datastore_time}
        winner = min(times, key=times.get)
        winner_time = times[winner]
        loser_time = max(times.values())
        speedup = loser_time / winner_time if winner_time > 0 else float('inf')
        return f"{winner} ({speedup:.2f}x)"

    @property
    def speedup(self) -> float:
        """Speedup of DataStore over Pandas (>1 means DataStore is faster)"""
        return self.pandas_time / self.datastore_time if self.datastore_time > 0 else float('inf')


def generate_test_data(n_rows: int) -> pd.DataFrame:
    """Generate test DataFrame with various column types."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            'id': range(n_rows),
            'int_col': np.random.randint(0, 1000, n_rows),
            'float_col': np.random.uniform(0, 1000, n_rows),
            'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'category': np.random.choice(
                ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10'], n_rows
            ),
            'bool_col': np.random.choice([True, False], n_rows),
            'date_col': pd.date_range('2020-01-01', periods=n_rows, freq='s')[:n_rows],
        }
    )

    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize MultiIndex columns to flat column names."""
    if isinstance(df.columns, pd.MultiIndex):
        # Convert MultiIndex to flat names like 'col_agg'
        df = df.copy()
        df.columns = ['_'.join(filter(None, map(str, col))).strip('_') for col in df.columns]
    return df


def _to_dataframe(result) -> pd.DataFrame:
    """Convert various result types to DataFrame."""
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.to_frame()
    # Handle LazySeries and other lazy types
    if hasattr(result, 'to_df'):
        return result.to_df()
    if hasattr(result, 'values') and hasattr(result, 'name'):
        # Series-like object
        return pd.DataFrame({result.name or 'value': result.values})
    raise TypeError(f"Cannot convert {type(result)} to DataFrame")


def _looks_like_datetime_with_zeros(col: pd.Series) -> bool:
    """Check if an object column contains datetime values mixed with zeros.

    This happens when pandas where() replaces datetime values with 0,
    creating a mixed object column.
    """
    if col.dtype != 'object':
        return False
    # Sample a few values to check
    sample = col.dropna().head(10)
    has_datetime = False
    has_zero = False
    for val in sample:
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            has_datetime = True
        elif val == 0:
            has_zero = True
    return has_datetime or has_zero


def _normalize_datetime_column(col: pd.Series) -> pd.Series:
    """Normalize a datetime column by removing timezone info.

    Handles:
    - datetime64[ns, tz] -> datetime64[ns] (remove timezone)
    - object columns with mixed tz-aware timestamps -> tz-naive
    - Preserves NaT and None values
    """
    if isinstance(col.dtype, pd.DatetimeTZDtype):
        # Direct tz-aware column
        return col.dt.tz_convert('UTC').dt.tz_localize(None)
    elif col.dtype == 'object':
        # Mixed object column - convert each value individually
        def normalize_val(val):
            if pd.isna(val) or val is None:
                return pd.NaT
            if isinstance(val, pd.Timestamp):
                if val.tz is not None:
                    return val.tz_convert('UTC').tz_localize(None)
                return val
            if isinstance(val, np.datetime64):
                return pd.Timestamp(val)
            # Non-datetime value (e.g., 0 from where())
            return pd.NaT

        return col.apply(normalize_val)
    else:
        # Already tz-naive datetime or other type
        return col


def verify_results(pandas_result, datastore_result, op_name: str, ignore_row_order: bool = False) -> tuple:
    """
    Verify that Pandas and DataStore results are consistent.

    Args:
        pandas_result: Result from Pandas operation
        datastore_result: Result from DataStore operation
        op_name: Name of the operation (for error messages)
        ignore_row_order: If True, sort both DataFrames before comparison (for Sort operations)

    Returns:
        tuple: (status: str, message: str)
            status: 'match', 'mismatch', 'design_diff', 'bug'
    """
    try:
        # Convert to DataFrame
        try:
            pandas_df = _to_dataframe(pandas_result)
        except Exception as e:
            return 'bug', f"Cannot convert Pandas result: {e}"

        try:
            datastore_df = _to_dataframe(datastore_result)
        except Exception as e:
            return 'bug', f"Cannot convert DataStore result (should return DataFrame): {e}"

        # Normalize column names (handle MultiIndex)
        pandas_df = _normalize_columns(pandas_df)
        datastore_df = _normalize_columns(datastore_df)

        # Check shape
        if pandas_df.shape != datastore_df.shape:
            return 'mismatch', f"Shape mismatch: Pandas={pandas_df.shape}, DataStore={datastore_df.shape}"

        # Check columns
        pandas_cols = sorted(pandas_df.columns.tolist())
        datastore_cols = sorted(datastore_df.columns.tolist())
        if pandas_cols != datastore_cols:
            return 'design_diff', f"Columns differ: Pandas={pandas_cols}, DataStore={datastore_cols}"

        # Reorder columns to match
        datastore_df = datastore_df[pandas_df.columns]

        # Reset index
        pandas_df = pandas_df.reset_index(drop=True)
        datastore_df = datastore_df.reset_index(drop=True)

        # For operations with Sort, compare as sets (ignore row order due to sort stability)
        # Check if this is a Sort operation
        is_sort_op = 'Sort' in op_name or 'sort' in op_name.lower()

        if is_sort_op or ignore_row_order:
            # Sort both DataFrames by all columns for comparison
            sort_cols = list(pandas_df.columns)
            try:
                pandas_sorted = pandas_df.sort_values(by=sort_cols, ignore_index=True)
                datastore_sorted = datastore_df.sort_values(by=sort_cols, ignore_index=True)
            except Exception:
                # If sorting fails, fall back to row-by-row comparison
                pandas_sorted = pandas_df
                datastore_sorted = datastore_df
        else:
            pandas_sorted = pandas_df
            datastore_sorted = datastore_df

        # Compare values column by column
        for col in pandas_sorted.columns:
            pandas_col = pandas_sorted[col]
            datastore_col = datastore_sorted[col]

            # Handle numeric columns with tolerance
            if pd.api.types.is_numeric_dtype(pandas_col) and pd.api.types.is_numeric_dtype(datastore_col):
                # Convert to float for comparison
                pd_vals = pd.to_numeric(pandas_col, errors='coerce').fillna(0).values
                ds_vals = pd.to_numeric(datastore_col, errors='coerce').fillna(0).values

                if not np.allclose(pd_vals, ds_vals, rtol=1e-5, atol=1e-8, equal_nan=True):
                    diff_mask = ~np.isclose(pd_vals, ds_vals, rtol=1e-5, atol=1e-8)
                    if diff_mask.any():
                        idx = np.where(diff_mask)[0][0]
                        return (
                            'mismatch',
                            f"Column '{col}' value mismatch at row {idx}: "
                            f"Pandas={pandas_col.iloc[idx]}, DataStore={datastore_col.iloc[idx]}",
                        )
            elif (
                pd.api.types.is_datetime64_any_dtype(pandas_col)
                or pd.api.types.is_datetime64_any_dtype(datastore_col)
                or (pandas_col.dtype == 'object' and _looks_like_datetime_with_zeros(pandas_col))
            ):
                # Handle datetime columns - compare as timestamps
                # Remove timezone info for comparison (chDB may return tz-aware, pandas tz-naive)
                # Also handle mixed object columns (e.g., from where() with 0 as replacement)

                # Normalize datastore column - handle both tz-aware dtype and mixed object columns
                ds_col_normalized = _normalize_datetime_column(datastore_col)
                pd_col_normalized = _normalize_datetime_column(pandas_col)

                # Now convert both to datetime
                pd_ts = pd.to_datetime(pd_col_normalized, errors='coerce')
                ds_ts = pd.to_datetime(ds_col_normalized, errors='coerce')

                # Check if values are equal (both NaT or same timestamp)
                match = (pd_ts.isna() & ds_ts.isna()) | (pd_ts == ds_ts)
                if not match.all():
                    idx = (~match).idxmax()
                    # This might be a design difference (e.g., where() with 0 vs NaT)
                    return (
                        'design_diff',
                        f"Column '{col}' datetime handling differs at row {idx}: "
                        f"Pandas={pandas_col.iloc[idx]}, DataStore={datastore_col.iloc[idx]}",
                    )
            else:
                # Non-numeric: string comparison
                pd_str = pandas_col.astype(str)
                ds_str = datastore_col.astype(str)
                if not pd_str.equals(ds_str):
                    diff_mask = pd_str != ds_str
                    if diff_mask.any():
                        idx = diff_mask.idxmax()
                        return (
                            'mismatch',
                            f"Column '{col}' value mismatch at row {idx}: "
                            f"Pandas={pandas_col.iloc[idx]}, DataStore={datastore_col.iloc[idx]}",
                        )

        return 'match', "Results match"

    except Exception as e:
        import traceback

        return 'bug', f"Verification error: {e}\n{traceback.format_exc()}"


def time_operation(func: Callable, n_runs: int = 5, collect_profile: bool = False) -> tuple:
    """
    Time an operation, return average time in milliseconds.

    Args:
        func: Function to benchmark
        n_runs: Number of runs
        collect_profile: If True, collect profiling data on the last run

    Returns:
        tuple: (avg_time_ms, profile_summary_dict or None)
    """
    from datastore.lazy_result import LazySeries

    times = []
    profile_summary = None

    for i in range(n_runs):
        # Collect profile on last run only
        is_last_run = (i == n_runs - 1) and collect_profile

        if is_last_run:
            reset_profiler()
            enable_profiling()

        start = time.perf_counter()
        result = func()
        # Force evaluation for lazy results
        if isinstance(result, pd.DataFrame):
            _ = len(result)
        elif isinstance(result, LazySeries):
            # LazySeries - force materialization via .values
            _ = len(result.values)
        elif hasattr(result, 'to_df'):
            # DataStore - force materialization
            _ = len(result.to_df())
        elif hasattr(result, '__len__'):
            # Other lazy objects with __len__
            _ = len(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        if is_last_run:
            profiler = get_profiler()
            if profiler and profiler.steps:
                profile_summary = profiler.summary()
            disable_profiling()

    # Remove outliers and return average
    times.sort()
    if len(times) > 2:
        times = times[1:-1]  # Remove min and max
    avg_time = sum(times) / len(times)

    return avg_time, profile_summary


class Benchmark:
    """
    Benchmark class for comparing Pandas vs DataStore performance.

    IMPORTANT: For fair comparison, BOTH Pandas and DataStore start from the same
    parquet file. Each operation reads fresh from disk to ensure fair comparison:
    - Pandas: pd.read_parquet(path) then operations
    - DataStore: DataStore.from_file(path) then operations

    This is the typical DataStore use case (lazy SQL execution on file sources).
    """

    def __init__(self, parquet_path: str, reuse_connection: bool = True):
        self.parquet_path = parquet_path
        self.reuse_connection = reuse_connection

        # Get row count from parquet file
        self.n_rows = len(pd.read_parquet(parquet_path))

        # Pre-connect DataStore to avoid connection overhead in tight loops
        self._ds_template = DataStore.from_file(self.parquet_path)
        self._ds_template.connect()

        # Store the executor for reuse
        self._shared_executor = self._ds_template._executor if reuse_connection else None

        # Warm up chDB session with a simple query to avoid cold start overhead
        if self._shared_executor:
            _ = self._ds_template.head(1).to_df()

    def _fresh_ds(self) -> DataStore:
        """
        Create a fresh DataStore from the same file source.

        If reuse_connection is True, shares the executor to avoid connection overhead.
        """
        ds = DataStore.from_file(self.parquet_path)
        if self.reuse_connection and self._shared_executor:
            # Reuse the shared executor to avoid connection overhead
            ds._executor = self._shared_executor
        return ds

    # ==================== Filter Operations ====================

    def pandas_filter_single(self) -> pd.DataFrame:
        # Fair comparison: read from file like DataStore does
        df = pd.read_parquet(self.parquet_path)
        return df[df['int_col'] > 500]

    def datastore_filter_single(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds[ds['int_col'] > 500].to_df()

    def pandas_filter_multiple(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df[(df['int_col'] > 300) & (df['int_col'] < 700) & (df['str_col'] == 'A')]

    def datastore_filter_multiple(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds[(ds['int_col'] > 300) & (ds['int_col'] < 700) & (ds['str_col'] == 'A')].to_df()

    # ==================== Sort Operations ====================

    def pandas_sort_single(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.sort_values('int_col', kind='stable')

    def datastore_sort_single(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.sort_values('int_col').to_df()

    def pandas_sort_multiple(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.sort_values(['str_col', 'int_col'], ascending=[True, False], kind='stable')

    def datastore_sort_multiple(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.sort_values(['str_col', 'int_col'], ascending=[True, False]).to_df()

    # ==================== GroupBy Operations ====================

    def pandas_groupby_count(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.groupby('str_col').size().reset_index(name='count')

    def datastore_groupby_count(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.groupby('str_col').size().reset_index(name='count')

    def pandas_groupby_agg(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return (
            df.groupby('category').agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']}).reset_index()
        )

    def datastore_groupby_agg(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return (
            ds.groupby('category').agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']}).reset_index()
        )

    # ==================== Head/Limit Operations ====================

    def pandas_head(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        return df.head(1000)

    def datastore_head(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        return ds.head(1000).to_df()

    # ==================== Combined Operations ====================

    def pandas_combined(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result.head(100)

    def datastore_combined(self) -> pd.DataFrame:
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        # Mirror pandas: use kind='stable' for deterministic ordering
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result.head(100).to_df()

    # ==================== Multi-Step Operations (DataStore Advantage) ====================
    # DataStore merges multiple operations into single SQL query

    def pandas_multi_filter(self) -> pd.DataFrame:
        """Multiple filter operations - Pandas executes each step separately."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result[result['int_col'] < 800]
        result = result[result['str_col'].isin(['A', 'B', 'C'])]
        result = result[result['float_col'] > 100]
        return result

    def datastore_multi_filter(self) -> pd.DataFrame:
        """DataStore merges all filters into single SQL WHERE clause."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result[result['int_col'] < 800]
        result = result[result['str_col'].isin(['A', 'B', 'C'])]
        result = result[result['float_col'] > 100]
        return result.to_df()

    def pandas_filter_select_sort(self) -> pd.DataFrame:
        """Filter -> Select -> Sort: Pandas processes step by step."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result

    def datastore_filter_select_sort(self) -> pd.DataFrame:
        """DataStore merges filter + select + sort into single optimized SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.to_df()

    def pandas_filter_groupby_sort(self) -> pd.DataFrame:
        """Filter -> GroupBy -> Sort: Analytics pattern."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
        result.columns = ['category', 'int_sum', 'float_avg']
        result = result.sort_values('int_sum', ascending=False, kind='stable')
        return result

    def datastore_filter_groupby_sort(self) -> pd.DataFrame:
        """DataStore: filter before groupby is pushed down to SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
        result.columns = ['category', 'int_sum', 'float_avg']
        return result.sort_values('int_sum', ascending=False)

    def pandas_complex_pipeline(self) -> pd.DataFrame:
        """Complex multi-step pipeline (simulates real usage)."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result.copy()
        result['computed'] = result['int_col'] * 2 + result['float_col']
        result = result[result['computed'] > 500]
        result = result[['id', 'int_col', 'str_col', 'computed']]
        result = result.sort_values('computed', ascending=False, kind='stable')
        return result.head(500)

    def datastore_complex_pipeline(self) -> pd.DataFrame:
        """DataStore handles complex pipeline with lazy execution."""
        result = self._fresh_ds()
        result = result[result['int_col'] > 100]
        result['computed'] = result['int_col'] * 2 + result['float_col']
        result = result[result['computed'] > 500]
        result = result[['id', 'int_col', 'str_col', 'computed']]
        result = result.sort_values('computed', ascending=False)
        return result.head(500).to_df()

    def pandas_chain_5_filters(self) -> pd.DataFrame:
        """5 sequential filter operations - each creates intermediate DataFrame."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['float_col'] < 950]
        result = result[result['str_col'] != 'E']
        return result

    def datastore_chain_5_filters(self) -> pd.DataFrame:
        """DataStore merges 5 filters into single SQL WHERE with AND."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['float_col'] < 950]
        result = result[result['str_col'] != 'E']
        return result.to_df()

    # ==================== Ultra-Complex Pipeline (DataStore Advantage) ====================

    def pandas_ultra_complex(self) -> pd.DataFrame:
        """Ultra-complex pipeline with 10+ operations."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result[result['bool_col'] == True]  # noqa: E712
        result = result[['id', 'int_col', 'float_col', 'str_col', 'category']]
        result = result.sort_values(['category', 'int_col'], ascending=[True, False], kind='stable')
        return result.head(1000)

    def datastore_ultra_complex(self) -> pd.DataFrame:
        """DataStore merges all operations into single optimized SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result[result['bool_col'] == True]  # noqa: E712
        result = result[['id', 'int_col', 'float_col', 'str_col', 'category']]
        # Mirror pandas: use kind='stable' for deterministic ordering
        result = result.sort_values(['category', 'int_col'], ascending=[True, False], kind='stable')
        return result.head(1000).to_df()

    # ==================== Pandas-Style Lazy API (Now Optimized!) ====================
    # After optimization, pandas-style API also uses lazy SQL execution

    def pandas_lazy_filter_sort_limit(self) -> pd.DataFrame:
        """Filter + Sort + Limit: Pandas baseline."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 200]
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result.head(100)

    def datastore_lazy_filter_sort_limit(self) -> pd.DataFrame:
        """DataStore pandas-style: filter+sort_values+head merged into single SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 200]
        # Mirror pandas: use kind='stable' for deterministic ordering
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result.head(100).to_df()

    def pandas_lazy_multi_filter_sort_limit(self) -> pd.DataFrame:
        """Multi-filter + Sort + Limit: Pandas baseline."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result.head(500)

    def datastore_lazy_multi_filter_sort_limit(self) -> pd.DataFrame:
        """DataStore pandas-style: all filters+sort_values+head merged into single SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['str_col'].isin(['A', 'B', 'C', 'D'])]
        # Mirror pandas: use kind='stable' for deterministic ordering
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result.head(500).to_df()

    def pandas_lazy_select_filter_sort(self) -> pd.DataFrame:
        """Select + Filter + Sort: Pandas baseline."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col']]
        result = result.sort_values('int_col', ascending=False, kind='stable')
        return result

    def datastore_lazy_select_filter_sort(self) -> pd.DataFrame:
        """DataStore pandas-style: filter+column_select+sort_values merged into SQL."""
        ds = self._fresh_ds()
        result = ds[ds['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.to_df()

    # ==================== Where (Conditional Replace) ====================
    # Pandas-style where: keep values where condition is True, replace with other where False
    # This is NOT SQL WHERE (row filter), but element-wise conditional replacement
    # DataStore must execute DataFrame to perform this operation

    def pandas_where_replace(self) -> pd.DataFrame:
        """Pandas where: conditional value replacement (not row filter)."""
        df = pd.read_parquet(self.parquet_path)
        # Keep values where int_col > 500, replace others with 0
        return df.where(df['int_col'] > 500, 0)

    def datastore_where_replace(self) -> pd.DataFrame:
        """DataStore where: triggers execution, then pandas where."""
        ds = self._fresh_ds()
        # This triggers execution because pandas where needs element-wise ops
        return ds.where(ds['int_col'] > 500, 0).to_df()

    def pandas_where_filter_then_replace(self) -> pd.DataFrame:
        """Filter first, then conditional replace."""
        df = pd.read_parquet(self.parquet_path)
        result = df[df['str_col'].isin(['A', 'B', 'C'])]
        return result.where(result['int_col'] > 500, 0)

    def datastore_where_filter_then_replace(self) -> pd.DataFrame:
        """DataStore: SQL filter pushdown, then pandas where."""
        ds = self._fresh_ds()
        result = ds[ds['str_col'].isin(['A', 'B', 'C'])]
        # Filter is pushed to SQL, where triggers execution
        return result.where(result['int_col'] > 500, 0).to_df()

    def pandas_mask_replace(self) -> pd.DataFrame:
        """Pandas mask: opposite of where (replace where condition is True)."""
        df = pd.read_parquet(self.parquet_path)
        # Replace values where int_col > 500 with -1
        return df.mask(df['int_col'] > 500, -1)

    def datastore_mask_replace(self) -> pd.DataFrame:
        """DataStore mask: triggers execution, then pandas mask."""
        ds = self._fresh_ds()
        return ds.mask(ds['int_col'] > 500, -1).to_df()


def run_benchmarks(
    data_sizes: List[int], temp_dir: str, n_runs: int = 5, collect_profiles: bool = False, verify: bool = False
) -> List[BenchmarkResult]:
    """Run all benchmarks for different data sizes."""
    results = []

    operations = [
        # Single operations (pandas-style API)
        ('Filter (single)', 'pandas_filter_single', 'datastore_filter_single'),
        ('Filter (multiple AND)', 'pandas_filter_multiple', 'datastore_filter_multiple'),
        ('Sort (single)', 'pandas_sort_single', 'datastore_sort_single'),
        ('Sort (multiple)', 'pandas_sort_multiple', 'datastore_sort_multiple'),
        ('GroupBy count', 'pandas_groupby_count', 'datastore_groupby_count'),
        ('GroupBy agg', 'pandas_groupby_agg', 'datastore_groupby_agg'),
        ('Head/Limit', 'pandas_head', 'datastore_head'),
        ('Combined ops', 'pandas_combined', 'datastore_combined'),
        # Multi-step operations (pandas-style API)
        ('Multi-filter (4x)', 'pandas_multi_filter', 'datastore_multi_filter'),
        ('Filter+Select+Sort', 'pandas_filter_select_sort', 'datastore_filter_select_sort'),
        ('Filter+GroupBy+Sort', 'pandas_filter_groupby_sort', 'datastore_filter_groupby_sort'),
        ('Complex pipeline', 'pandas_complex_pipeline', 'datastore_complex_pipeline'),
        ('Chain 5 filters', 'pandas_chain_5_filters', 'datastore_chain_5_filters'),
        ('Ultra-complex (10+ ops)', 'pandas_ultra_complex', 'datastore_ultra_complex'),
        # Pandas-style lazy API (now optimized - sort_values/head use lazy SQL)
        ('Pandas-style: Filter+Sort+Head', 'pandas_lazy_filter_sort_limit', 'datastore_lazy_filter_sort_limit'),
        (
            'Pandas-style: MultiFilter+Sort+Head',
            'pandas_lazy_multi_filter_sort_limit',
            'datastore_lazy_multi_filter_sort_limit',
        ),
        ('Pandas-style: Select+Filter+Sort', 'pandas_lazy_select_filter_sort', 'datastore_lazy_select_filter_sort'),
        # Where/Mask operations (element-wise conditional replace, NOT row filter)
        # These trigger immediate execution as they require pandas element-wise ops
        ('Where (value replace)', 'pandas_where_replace', 'datastore_where_replace'),
        ('Filter+Where', 'pandas_where_filter_then_replace', 'datastore_where_filter_then_replace'),
        ('Mask (value replace)', 'pandas_mask_replace', 'datastore_mask_replace'),
    ]

    for size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Generating {size:,} rows of test data...")
        df = generate_test_data(size)

        # Save to parquet file - both Pandas and DataStore will read from this
        parquet_path = os.path.join(temp_dir, f"test_data_{size}.parquet")
        df.to_parquet(parquet_path)
        print(f"Saved to {parquet_path}")
        del df  # Free memory - benchmark will read fresh from parquet

        benchmark = Benchmark(parquet_path)

        print(f"Running benchmarks (n_runs={n_runs})...")

        for op_name, pandas_method, datastore_method in operations:
            pandas_func = getattr(benchmark, pandas_method)
            datastore_func = getattr(benchmark, datastore_method)

            # Warm up and optionally verify results
            try:
                pandas_result = pandas_func()
                datastore_result = datastore_func()

                # Verify results consistency
                if verify:
                    status, message = verify_results(pandas_result, datastore_result, op_name)
                    if status == 'match':
                        print(f"  ✓  {op_name}: Results match")
                    elif status == 'design_diff':
                        print(f"  ⚠️  {op_name}: Design difference - {message}")
                    elif status == 'bug':
                        print(f"  ❌ {op_name}: BUG DETECTED - {message}")
                    else:  # mismatch
                        print(f"  ✗  {op_name}: Mismatch - {message}")

            except Exception as e:
                print(f"  Skipping {op_name}: {e}")
                continue

            # Benchmark
            pandas_time, _ = time_operation(pandas_func, n_runs, collect_profile=False)
            datastore_time, profile_data = time_operation(datastore_func, n_runs, collect_profile=collect_profiles)

            result = BenchmarkResult(
                operation=op_name,
                data_size=size,
                pandas_time=pandas_time,
                datastore_time=datastore_time,
                profile_data=profile_data or {},
            )
            results.append(result)

            print(
                f"  {op_name:25s}: Pandas={pandas_time:8.2f}ms, DataStore={datastore_time:8.2f}ms -> {result.fastest}"
            )

    return results


def analyze_profile_bottlenecks(results: List[BenchmarkResult]):
    """Analyze profiling data to identify performance bottlenecks."""
    print("\n" + "=" * 100)
    print("PROFILING ANALYSIS: Performance Bottlenecks")
    print("=" * 100)

    # Aggregate profile data across all results
    step_totals = defaultdict(float)
    step_counts = defaultdict(int)

    for r in results:
        if r.profile_data:
            for step, duration in r.profile_data.items():
                # Normalize step names (remove nested prefixes for aggregation)
                simple_name = step.split('.')[-1] if '.' in step else step
                step_totals[simple_name] += duration
                step_counts[simple_name] += 1

    if not step_totals:
        print("\nNo profiling data collected. Run with --profile flag.")
        return

    # Calculate averages and sort by total time
    step_avgs = {}
    for step in step_totals:
        step_avgs[step] = step_totals[step] / step_counts[step]

    sorted_steps = sorted(step_totals.items(), key=lambda x: x[1], reverse=True)
    total_time = sum(step_totals.values())

    print("\n1. AGGREGATE TIME BY STEP (across all operations)")
    print("-" * 70)
    print(f"{'Step':<40} {'Total (ms)':>12} {'Avg (ms)':>12} {'% of Total':>10}")
    print("-" * 70)

    for step, total in sorted_steps[:15]:  # Top 15 steps
        avg = step_avgs[step]
        pct = (total / total_time * 100) if total_time > 0 else 0
        print(f"{step:<40} {total:>12.2f} {avg:>12.2f} {pct:>9.1f}%")

    # Identify bottlenecks by operation
    print("\n2. TOP BOTTLENECKS BY OPERATION")
    print("-" * 70)

    for r in results:
        if r.profile_data and r.datastore_time > 50:  # Only show slow operations
            print(f"\n{r.operation} ({r.data_size:,} rows) - Total: {r.datastore_time:.2f}ms")
            sorted_profile = sorted(r.profile_data.items(), key=lambda x: x[1], reverse=True)
            for step, duration in sorted_profile[:5]:  # Top 5 steps per operation
                pct = (duration / r.datastore_time * 100) if r.datastore_time > 0 else 0
                print(f"  {step:<50} {duration:>8.2f}ms ({pct:>5.1f}%)")

    # Identify key bottleneck categories
    print("\n3. BOTTLENECK SUMMARY")
    print("-" * 70)

    categories = {
        'Connection': ['Connection'],
        'SQL Build': ['SQL Build', 'Query Planning'],
        'SQL Execution': ['SQL Execution', 'chDB Query', 'chDB DataFrame Query'],
        'DataFrame Ops': ['DataFrame Operations', 'LazyRelationalOp', 'LazyColumnAssignment', 'LazySQLQuery'],
        'Cache': ['Cache Check', 'Cache Write'],
        'Result Conversion': ['Result to DataFrame'],
    }

    category_totals = {}
    for cat, keywords in categories.items():
        cat_total = sum(step_totals[s] for s in step_totals if any(kw in s for kw in keywords))
        category_totals[cat] = cat_total

    sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    total_categorized = sum(category_totals.values())

    print(f"\n{'Category':<25} {'Total Time (ms)':>15} {'% of Total':>12}")
    print("-" * 55)
    for cat, total in sorted_cats:
        pct = (total / total_categorized * 100) if total_categorized > 0 else 0
        print(f"{cat:<25} {total:>15.2f} {pct:>11.1f}%")

    # Recommendations
    print("\n4. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 70)

    if category_totals.get('Connection', 0) > total_categorized * 0.3:
        print("⚠️  CONNECTION overhead is high (>30%). Consider:")
        print("   - Reusing connections across operations")
        print("   - Using connection pooling")

    if category_totals.get('SQL Execution', 0) > total_categorized * 0.5:
        print("⚠️  SQL EXECUTION is the primary bottleneck (>50%). Consider:")
        print("   - Query optimization (indexes, better predicates)")
        print("   - Data format optimization (columnar vs row-based)")

    if category_totals.get('DataFrame Ops', 0) > total_categorized * 0.3:
        print("⚠️  DATAFRAME OPERATIONS overhead is high (>30%). Consider:")
        print("   - Pushing more operations to SQL layer")
        print("   - Reducing Python-side data manipulation")

    if category_totals.get('Result Conversion', 0) > total_categorized * 0.2:
        print("⚠️  RESULT CONVERSION overhead is high (>20%). Consider:")
        print("   - Reducing result set size")
        print("   - Using more efficient data formats")


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Group by data size
    sizes = sorted(set(r.data_size for r in results))
    operations = sorted(set(r.operation for r in results), key=lambda x: [r.operation for r in results].index(x))

    # Print header
    print(f"\n{'Operation':<25}", end='')
    for size in sizes:
        print(f" | {size:>12,} rows", end='')
    print()
    print("-" * 25, end='')
    for _ in sizes:
        print("-" + "-" * 17, end='')
    print()

    # Print results
    for op in operations:
        print(f"{op:<25}", end='')
        for size in sizes:
            matching = [r for r in results if r.operation == op and r.data_size == size]
            if matching:
                r = matching[0]
                times = {'Pandas': r.pandas_time, 'DataStore': r.datastore_time}
                winner = min(times, key=times.get)
                winner_time = times[winner]
                loser_time = max(times.values())
                ratio = loser_time / winner_time if winner_time > 0 else float('inf')
                print(f" | {winner:>8} {ratio:>5.1f}x ", end='')
            else:
                print(f" | {'N/A':>14}", end='')
        print()

    # Print detailed times
    print("\n" + "=" * 100)
    print("DETAILED TIMES (milliseconds)")
    print("=" * 100)

    print(f"\n{'Operation':<25} | {'Size':>10} | {'Pandas':>10} | {'DataStore':>10} | {'Winner':>16}")
    print("-" * 85)

    for r in results:
        print(
            f"{r.operation:<25} | {r.data_size:>10,} | {r.pandas_time:>10.2f} | {r.datastore_time:>10.2f} | {r.fastest:>16}"
        )


def save_benchmark_results(
    results: List[BenchmarkResult], output_dir: str, metadata: Dict, data_sizes: List[int], n_runs: int
):
    """Save benchmark results to CSV and metadata to JSON."""
    # Convert results to serializable format
    results_data = []
    for r in results:
        results_data.append(
            {
                'operation': r.operation,
                'data_size': r.data_size,
                'pandas_time_ms': r.pandas_time,
                'datastore_time_ms': r.datastore_time,
                'speedup': r.speedup,
                'fastest': r.fastest,
            }
        )

    # Save CSV (flat format for easy analysis)
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    df = pd.DataFrame(results_data)
    df['timestamp'] = metadata['timestamp']
    df['git_branch'] = metadata['git_branch']
    df['git_commit'] = metadata['git_commit_short']
    df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # Save metadata separately
    meta_path = os.path.join(output_dir, 'metadata.json')
    full_metadata = {
        **metadata,
        'config': {
            'data_sizes': data_sizes,
            'n_runs': n_runs,
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    return csv_path


def plot_benchmark_results(
    results: List[BenchmarkResult], output_dir: str = None, output_prefix: str = 'benchmark_pandas_datastore'
):
    """Generate benchmark visualization plot."""
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12

    # Convert results to DataFrame
    data = []
    for r in results:
        times = {'Pandas': r.pandas_time, 'DataStore': r.datastore_time}
        winner = min(times, key=times.get)
        data.append(
            {
                'op': r.operation,
                'size': r.data_size,
                'pandas': r.pandas_time,
                'datastore': r.datastore_time,
                'winner': winner,
                'speedup': r.speedup,
            }
        )

    df = pd.DataFrame(data)

    # Define colors
    colors = {'DataStore': '#5B8FF9', 'Pandas': '#5AD8A6'}  # Blue  # Teal

    # Get unique sizes and operations
    sizes = sorted(df['size'].unique())
    size_labels = [f'{s//1000}K' if s < 1000000 else f'{s//1000000}M' for s in sizes]
    all_ops = list(df['op'].unique())

    # Calculate wins for each size
    wins_by_size = {}
    for size in sizes:
        df_size = df[df['size'] == size]
        wins_by_size[size] = {
            'Pandas': len(df_size[df_size['winner'] == 'Pandas']),
            'DataStore': len(df_size[df_size['winner'] == 'DataStore']),
        }

    n_ops = len(all_ops)
    n_sizes = len(sizes)

    # Group spacing parameters
    width = 0.35
    gap_between_sizes = 0.2
    gap_between_ops = 1.0
    size_group_width = 2 * width
    total_size_group_width = n_sizes * size_group_width + (n_sizes - 1) * gap_between_sizes

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Calculate positions
    x_positions = np.arange(n_ops) * (total_size_group_width + gap_between_ops)

    # Store positions for labels
    all_positions = []
    all_labels = []

    # Plot bars for all operations
    for op_idx, op in enumerate(all_ops):
        df_op = df[df['op'] == op]

        for size_idx, (size, size_label) in enumerate(zip(sizes, size_labels)):
            df_size = df_op[df_op['size'] == size]

            if len(df_size) == 0:
                continue

            row = df_size.iloc[0]

            # Calculate x position
            base_x = x_positions[op_idx] + size_idx * (size_group_width + gap_between_sizes)

            # Plot bars - Pandas, DataStore order
            pandas_bar = ax.bar(
                base_x, row['pandas'], width, color=colors['Pandas'], alpha=0.75, edgecolor='black', linewidth=0.5
            )
            datastore_bar = ax.bar(
                base_x + width,
                row['datastore'],
                width,
                color=colors['DataStore'],
                alpha=0.75,
                edgecolor='black',
                linewidth=0.5,
            )

            # Highlight winner
            if row['winner'] == 'Pandas':
                pandas_bar[0].set_alpha(1.0)
                pandas_bar[0].set_linewidth(2.0)
            elif row['winner'] == 'DataStore':
                datastore_bar[0].set_alpha(1.0)
                datastore_bar[0].set_linewidth(2.0)

            # Store position
            center_x = base_x + width * 0.5
            all_positions.append(center_x)
            all_labels.append(size_label)

    # Formatting
    ax.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

    # Build table data
    table_data = [['Wins'] + size_labels]
    for engine in ['DataStore', 'Pandas']:
        row = [engine] + [str(wins_by_size[size][engine]) for size in sizes]
        table_data.append(row)

    # Table colors
    table_colors = [
        ['white'] * (n_sizes + 1),
        [colors['DataStore']] + ['white'] * n_sizes,
        [colors['Pandas']] + ['white'] * n_sizes,
    ]

    # Add table
    table = ax.table(
        cellText=table_data, cellLoc='center', loc='upper left', bbox=[0.02, 0.78, 0.12, 0.18], cellColours=table_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header row
    for i in range(n_sizes + 1):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=8)
        cell.set_facecolor('#E8E8E8')

    # Style color column
    for i in range(1, 3):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold', fontsize=8, color='black')
        cell.set_alpha(0.8)

    # Table borders
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1.0)
        cell.set_edgecolor('black')

    # Title
    ax.set_title(
        'DataFrame Performance Benchmark: Pandas vs DataStore (chDB lazy mode)',
        fontweight='bold',
        fontsize=12,
        pad=15,
    )

    # Two-level x-axis labels
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=7, color='gray')
    ax.tick_params(axis='x', which='major', length=3)

    # Operation names on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    op_centers = x_positions + total_size_group_width / 2
    ax2.set_xticks(op_centers)
    ax2.set_xticklabels([op.replace(' ', '\n') for op in all_ops], fontsize=7, fontweight='bold')
    ax2.tick_params(axis='x', which='major', length=0)
    ax2.spines['top'].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 10))

    plt.tight_layout()

    # Determine output path
    if output_dir:
        pdf_path = os.path.join(output_dir, f'{output_prefix}.pdf')
    else:
        pdf_path = f'{output_prefix}.pdf'

    # Save figure (PDF only)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {pdf_path}")

    # Print wins summary
    print("\nWins Summary by Data Size:")
    print("=" * 50)
    for size, label in zip(sizes, size_labels):
        wins = wins_by_size[size]
        print(f"{label:>5}: DataStore={wins['DataStore']}, Pandas={wins['Pandas']}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark DataStore vs Pandas')
    parser.add_argument('--profile', action='store_true', help='Collect profiling data')
    parser.add_argument(
        '--sizes', type=str, default='100000,1000000', help='Comma-separated data sizes (default: 100000,1000000)'
    )
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per operation')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results to file')
    parser.add_argument('--verify', action='store_true', help='Verify result consistency between Pandas and DataStore')
    parser.add_argument('--output-dir', type=str, help='Custom output directory for results')
    args = parser.parse_args()

    print("=" * 60)
    print("Pandas vs DataStore (chDB Lazy Mode) Benchmark")
    print("=" * 60)

    # Create output directory for results
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        git_info = get_git_info()
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'timestamp_unix': datetime.now().timestamp(),
            'git_branch': git_info['branch'],
            'git_commit_hash': git_info['commit_hash'],
            'git_commit_short': git_info['commit_short'],
            'git_commit_message': git_info['commit_message'],
            'git_is_dirty': git_info['is_dirty'],
            'output_dir': output_dir,
        }
    else:
        output_dir, metadata = create_benchmark_output_dir('datastore_vs_pandas')

    print(f"Output directory: {output_dir}")
    print(f"Git: {metadata.get('git_branch', 'unknown')}@{metadata.get('git_commit_short', 'unknown')}")
    if metadata.get('git_is_dirty'):
        print("Warning: Working directory has uncommitted changes")

    # Create temporary directory for parquet files
    temp_dir = tempfile.mkdtemp(prefix='datastore_benchmark_')
    print(f"Using temp directory: {temp_dir}")

    try:
        # Parse data sizes
        data_sizes = [int(s.strip()) for s in args.sizes.split(',')]
        print(f"Data sizes: {data_sizes}")
        print(f"Profiling: {'enabled' if args.profile else 'disabled'}")
        print(f"Verification: {'enabled' if args.verify else 'disabled'}")

        # Run benchmarks
        results = run_benchmarks(
            data_sizes, temp_dir, n_runs=args.runs, collect_profiles=args.profile, verify=args.verify
        )

        # Print summary
        print_summary(results)

        # Print profiling analysis if collected
        if args.profile:
            analyze_profile_bottlenecks(results)

        # Recommendations
        print("\n" + "=" * 100)
        print("ANALYSIS: DataStore Lazy Execution Advantages")
        print("=" * 100)

        # Analyze results - count wins for each engine
        def get_winner(r):
            times = {'Pandas': r.pandas_time, 'DataStore': r.datastore_time}
            return min(times, key=times.get)

        pandas_wins = sum(1 for r in results if get_winner(r) == 'Pandas')
        datastore_wins = sum(1 for r in results if get_winner(r) == 'DataStore')

        print(f"\nOverall wins: Pandas={pandas_wins}/{len(results)}, DataStore={datastore_wins}/{len(results)}")

        # Group by size
        for size in sorted(set(r.data_size for r in results)):
            size_results = [r for r in results if r.data_size == size]
            pandas_better = sum(1 for r in size_results if get_winner(r) == 'Pandas')
            datastore_better = sum(1 for r in size_results if get_winner(r) == 'DataStore')
            print(f"\n  {size:>10,} rows: Pandas={pandas_better}, DataStore={datastore_better}")

        # Find operations where DataStore excels
        print("\n" + "-" * 60)
        print("Operations where DataStore excels (multi-step SQL merging):")
        print("-" * 60)

        multi_step_ops = [
            'Multi-filter (4x)',
            'Filter+Select+Sort',
            'Filter+GroupBy+Sort',
            'Complex pipeline',
            'Chain 5 filters',
            'Ultra-complex (10+ ops)',
            'Pandas-style: Filter+Sort+Head',
            'Pandas-style: MultiFilter+Sort+Head',
            'Pandas-style: Select+Filter+Sort',
        ]

        for op in multi_step_ops:
            op_results = [r for r in results if r.operation == op]
            if op_results:
                avg_speedup = sum(r.speedup for r in op_results) / len(op_results)
                if avg_speedup > 1:
                    print(f"  {op:<25}: DataStore is {avg_speedup:.2f}x faster on average")
                else:
                    print(f"  {op:<25}: Pandas is {1/avg_speedup:.2f}x faster on average")

        # Save results to output directory
        if not args.no_save:
            save_benchmark_results(results, output_dir, metadata, data_sizes, args.runs)

        # Generate plot (unless --no-plot)
        if not args.no_plot:
            plot_benchmark_results(results, output_dir=output_dir, output_prefix='benchmark_pandas_datastore')

        print(f"\n{'=' * 60}")
        print(f"Results saved to: {output_dir}")
        print(f"{'=' * 60}")

    finally:
        # Cleanup temporary files
        import shutil

        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"\nWarning: Could not clean up temp directory: {e}")


if __name__ == '__main__':
    main()
