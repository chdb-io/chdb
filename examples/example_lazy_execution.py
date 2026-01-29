"""
Examples demonstrating lazy execution in DataStore.

After implementing the lazy execution architecture, these examples will work.
"""

from datastore import DataStore as ds

# =============================================================================
# Example 1: Basic Lazy Column Assignment
# =============================================================================


def example_1_basic_lazy():
    """Demonstrate basic lazy column assignment."""
    print("=" * 80)
    print("Example 1: Basic Lazy Column Assignment")
    print("=" * 80)

    # Load data (lazy - no execution yet)
    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")
    print("✓ Data source created (not loaded yet)")

    # Column assignment (lazy - no execution yet)
    nat["n_nationkey"] = nat["n_nationkey"] - 1
    print("✓ Column assignment recorded (not executed yet)")

    # Show explain (no execution)
    print("\nExecution plan:")
    nat.explain()

    # Trigger execution by printing
    print("\nTriggering execution by print():")
    print(nat.head())  # NOW it executes


# =============================================================================
# Example 2: Multiple Lazy Operations
# =============================================================================


def example_2_multiple_operations():
    """Demonstrate multiple lazy operations."""
    print("\n" + "=" * 80)
    print("Example 2: Multiple Lazy Operations")
    print("=" * 80)

    # Load data
    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")

    # Multiple operations (all lazy)
    nat["n_nationkey_minus_1"] = nat["n_nationkey"] - 1
    nat["n_nationkey_squared"] = nat["n_nationkey"] ** 2
    nat["n_nationkey_plus_region"] = nat["n_nationkey"] + nat["n_regionkey"]

    print("✓ Three column assignments recorded")

    # Show execution plan
    nat.explain()

    # Execute and show
    print("\nResult:")
    print(nat)


# =============================================================================
# Example 3: Mix SQL and Lazy Operations
# =============================================================================


def example_3_mixed_operations():
    """Demonstrate mixing SQL operations and lazy pandas operations."""
    print("\n" + "=" * 80)
    print("Example 3: Mix SQL and Lazy Operations")
    print("=" * 80)

    # Load and filter (SQL - lazy)
    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")
    nat = nat.filter(nat["n_regionkey"] < 3)

    print("✓ SQL filter recorded")

    # Add computed columns (Pandas - lazy)
    nat["key_category"] = nat["n_nationkey"] // 10
    nat["is_low_key"] = nat["n_nationkey"] < 10

    print("✓ Pandas operations recorded")

    # Select specific columns (SQL - lazy)
    nat = nat.select("n_name", "n_nationkey", "key_category", "is_low_key")

    print("✓ SQL select recorded")

    # Show execution plan
    nat.explain()

    # Execute
    print("\nResult:")
    print(nat)


# =============================================================================
# Example 4: Complex Expression
# =============================================================================


def example_4_complex_expression():
    """Demonstrate complex expressions in column assignment."""
    print("\n" + "=" * 80)
    print("Example 4: Complex Expression")
    print("=" * 80)

    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")

    # Complex expression
    nat["complex_value"] = (nat["n_nationkey"] * 2 + nat["n_regionkey"]) ** 2 - 100

    print("✓ Complex expression recorded")

    # Show plan
    nat.explain()

    # Execute
    print("\nResult:")
    print(nat[["n_name", "n_nationkey", "n_regionkey", "complex_value"]])


# =============================================================================
# Example 5: Lazy Operations with Join
# =============================================================================


def example_5_join_with_lazy():
    """Demonstrate lazy operations with joins."""
    print("\n" + "=" * 80)
    print("Example 5: Lazy Operations with Join")
    print("=" * 80)

    # Two data sources
    nation = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")
    region = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/region.parquet")

    # Add computed columns (lazy)
    nation["nation_id_doubled"] = nation["n_nationkey"] * 2
    region["region_id_doubled"] = region["r_regionkey"] * 2

    print("✓ Column operations recorded")

    # Join (lazy)
    result = nation.join(region, on=nation["n_regionkey"] == region["r_regionkey"], how="inner")

    print("✓ Join recorded")

    # Select columns (lazy)
    result = result.select("n_name", "r_name", "nation_id_doubled", "region_id_doubled")

    print("✓ Select recorded")

    # Show plan
    result.explain()

    # Execute
    print("\nResult:")
    print(result)


# =============================================================================
# Example 6: Chained Operations
# =============================================================================


def example_6_chained_operations():
    """Demonstrate method chaining with lazy execution."""
    print("\n" + "=" * 80)
    print("Example 6: Chained Operations")
    print("=" * 80)

    # Everything is lazy until the end
    result = (
        ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")
        .filter(ds["n_regionkey"] < 3)
        .select("n_name", "n_nationkey", "n_regionkey")
        .sort("n_nationkey", ascending=False)
        .limit(5)
    )

    # Add computed column (still lazy)
    result["nation_score"] = result["n_nationkey"] * 10 + result["n_regionkey"]

    print("✓ All operations recorded")

    # Show plan
    result.explain()

    # Execute
    print("\nResult:")
    print(result)


# =============================================================================
# Example 7: Column Selection with Assignment
# =============================================================================


def example_7_column_selection():
    """Demonstrate column selection behavior."""
    print("\n" + "=" * 80)
    print("Example 7: Column Selection")
    print("=" * 80)

    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")

    # Add new columns
    nat["key_plus_1"] = nat["n_nationkey"] + 1
    nat["key_plus_2"] = nat["n_nationkey"] + 2
    nat["key_plus_3"] = nat["n_nationkey"] + 3

    # Select specific columns (including computed ones)
    nat = nat[["n_name", "key_plus_1", "key_plus_3"]]

    print("✓ Operations recorded")

    # Show plan
    nat.explain()

    # Execute
    print("\nResult:")
    print(nat)


# =============================================================================
# Example 8: Explicit Execution
# =============================================================================


def example_8_explicit_execute():
    """Demonstrate explicit execution with to_df()."""
    print("\n" + "=" * 80)
    print("Example 8: Explicit Execution")
    print("=" * 80)

    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")

    # Operations
    nat["computed"] = nat["n_nationkey"] * 2
    nat = nat.filter(nat["n_regionkey"] < 2)

    print("✓ Operations recorded (not executed)")

    # Explicit execution
    print("\nCalling to_df() to execute...")
    df = nat.to_df()

    print(f"✓ Executed to pandas DataFrame: {df.shape}")
    print(df.head())

    # Now df is a regular pandas DataFrame
    print(f"\nType: {type(df)}")
    print(f"Columns: {list(df.columns)}")


# =============================================================================
# Example 9: Performance Comparison
# =============================================================================


def example_9_performance():
    """Compare lazy vs eager execution performance."""
    import time

    print("\n" + "=" * 80)
    print("Example 9: Performance Comparison")
    print("=" * 80)

    # Lazy approach
    start = time.time()
    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")
    nat["col1"] = nat["n_nationkey"] * 2
    nat["col2"] = nat["n_nationkey"] * 3
    nat["col3"] = nat["n_nationkey"] * 4
    nat["col4"] = nat["n_nationkey"] * 5
    lazy_record_time = time.time() - start

    print(f"✓ Recording 4 operations took: {lazy_record_time:.4f}s")

    start = time.time()
    df = nat.to_df()
    lazy_exec_time = time.time() - start

    print(f"✓ Executing all operations took: {lazy_exec_time:.4f}s")
    print(f"  Total: {lazy_record_time + lazy_exec_time:.4f}s")


# =============================================================================
# Example 10: Debugging with explain()
# =============================================================================


def example_10_debugging():
    """Demonstrate debugging with explain()."""
    print("\n" + "=" * 80)
    print("Example 10: Debugging with explain()")
    print("=" * 80)

    # Build a complex query
    nat = ds.uri("https://shell.duckdb.org/data/tpch/0_01/parquet/nation.parquet")

    # SQL operations
    nat = nat.filter(nat["n_regionkey"] < 3)
    nat = nat.select("n_name", "n_nationkey", "n_regionkey")

    # Pandas operations
    nat["doubled"] = nat["n_nationkey"] * 2
    nat["squared"] = nat["n_nationkey"] ** 2

    # More SQL operations
    nat = nat.sort("n_nationkey")
    nat = nat.limit(10)

    # Show execution plan WITHOUT executing
    print("\nExecution plan:")
    nat.explain(verbose=True)

    print("\nNote: explain() shows the plan but doesn't execute it")
    print("To execute, call print(nat) or nat.to_df()")


# =============================================================================
# Run all examples
# =============================================================================

if __name__ == "__main__":
    # Run examples
    examples = [
        example_1_basic_lazy,
        example_2_multiple_operations,
        example_3_mixed_operations,
        example_4_complex_expression,
        example_5_join_with_lazy,
        example_6_chained_operations,
        example_7_column_selection,
        example_8_explicit_execute,
        example_9_performance,
        example_10_debugging,
    ]

    for example in examples:
        try:
            example()
            print("\n✅ Example completed successfully\n")
        except Exception as e:
            print(f"\n❌ Example failed: {e}\n")
            import traceback

            traceback.print_exc()
