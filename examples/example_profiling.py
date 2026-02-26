#!/usr/bin/env python
"""
Example: Execution Profiling for DataStore

This example demonstrates how to use the profiling feature to analyze
execution performance and identify bottlenecks.

The profiler tracks:
- Cache check time
- Query planning time
- SQL build time
- SQL execution time
- DataFrame operations time (with per-operation breakdown)
- Cache write time
"""

import logging
import pandas as pd
from datastore import (
    DataStore,
    enable_profiling,
    disable_profiling,
    set_log_level,
)


def example_basic_profiling():
    """Basic profiling example with DataFrame operations."""
    print("=" * 70)
    print("Example 1: Basic Profiling with DataFrame Operations")
    print("=" * 70)

    # Enable profiling - output goes to logger at INFO level
    enable_profiling()
    set_log_level(logging.INFO)

    # Create sample data
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'] * 200,
        'value': range(1000),
        'category': ['A', 'B', 'C', 'A', 'B'] * 200
    })

    # Build pipeline with multiple operations
    ds = DataStore(df)
    ds = ds.filter(ds['value'] > 100)
    ds['doubled'] = ds['value'] * 2
    ds = ds.filter(ds['doubled'] > 300)

    # Execute - profiling report will be printed
    result = ds.to_df()
    print(f"\nResult: {len(result)} rows\n")


def example_sql_profiling():
    """Profiling example with SQL source."""
    print("=" * 70)
    print("Example 2: Profiling with SQL Source (File)")
    print("=" * 70)

    enable_profiling()
    set_log_level(logging.INFO)

    # Read from file (SQL execution)
    ds = DataStore.from_file('sales_data.parquet')
    ds = ds.filter(ds['quantity'] > 5)
    ds = ds.head(100)

    # Execute - profiling shows SQL build and execution time
    result = ds.to_df()
    print(f"\nResult: {len(result)} rows, {len(result.columns)} columns\n")


def example_mixed_sql_pandas():
    """Profiling example with mixed SQL and Pandas operations."""
    print("=" * 70)
    print("Example 3: Mixed SQL and Pandas Operations")
    print("=" * 70)

    enable_profiling()
    set_log_level(logging.INFO)

    # Create sample data
    df = pd.DataFrame({
        'id': range(500),
        'value': [x % 100 for x in range(500)],
        'name': ['Item_' + str(x % 50) for x in range(500)]
    })

    ds = DataStore(df)
    ds = ds.filter(ds['value'] > 20)        # Pandas filter
    ds['computed'] = ds['value'] * 2         # Pandas column assignment
    ds = ds.sql('computed > 100 LIMIT 10')   # SQL on DataFrame

    # Execute - shows breakdown of each operation type
    result = ds.to_df()
    print(f"\nResult:\n{result}\n")


def example_profiling_disabled():
    """Show that profiling can be disabled."""
    print("=" * 70)
    print("Example 4: Profiling Disabled (No Overhead)")
    print("=" * 70)

    disable_profiling()
    set_log_level(logging.WARNING)  # Reset to default

    df = pd.DataFrame({
        'a': range(100),
        'b': ['x', 'y'] * 50
    })

    ds = DataStore(df)
    ds = ds.filter(ds['a'] > 50)
    result = ds.to_df()

    print(f"Result: {len(result)} rows")
    print("(No profiling output when disabled)")


def example_programmatic_access():
    """Access profiling data programmatically."""
    print("\n" + "=" * 70)
    print("Example 5: Programmatic Access to Profiling Data")
    print("=" * 70)

    from datastore.config import is_profiling_enabled

    enable_profiling()
    set_log_level(logging.WARNING)  # Suppress auto-report

    df = pd.DataFrame({
        'x': range(1000),
        'y': range(1000)
    })

    ds = DataStore(df)
    ds = ds.filter(ds['x'] > 500)
    ds['z'] = ds['x'] + ds['y']
    result = ds.to_df()

    # Get the profiler after execution
    from datastore.config import get_profiler
    profiler = get_profiler()

    if profiler and profiler.steps:
        # Get summary as dict
        summary = profiler.summary()
        print("\nProfiling Summary (programmatic access):")
        for step_name, duration_ms in summary.items():
            print(f"  {step_name}: {duration_ms:.2f}ms")

        # Custom report with threshold
        print("\n" + profiler.report(min_duration_ms=0.5))


if __name__ == '__main__':
    example_basic_profiling()
    print()
    example_sql_profiling()
    print()
    example_mixed_sql_pandas()
    print()
    example_profiling_disabled()
    example_programmatic_access()

