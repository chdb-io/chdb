#!/usr/bin/env python3
"""
Example demonstrating the explain() method in DataStore.

The explain() method shows the execution plan for mixed SQL and Pandas operations,
making it clear which operations are lazy SQL queries, which trigger execution,
and which operate on cached DataFrames.
"""

import tempfile
import os
from datastore import DataStore


def create_sample_data():
    """Create sample data for demonstration."""
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "employees.csv")
    
    with open(csv_file, "w") as f:
        f.write("id,name,age,salary,department\n")
        f.write("1,Alice,28,65000,Engineering\n")
        f.write("2,Bob,32,70000,Sales\n")
        f.write("3,Charlie,26,55000,Engineering\n")
        f.write("4,David,35,80000,Marketing\n")
        f.write("5,Eve,31,72000,Sales\n")
        f.write("6,Frank,29,62000,Engineering\n")
    
    return csv_file, temp_dir


def main():
    csv_file, temp_dir = create_sample_data()
    
    try:
        print("=" * 80)
        print("DataStore explain() Method Demo")
        print("=" * 80)
        print("\nThe explain() method helps you understand what DataStore will do")
        print("when you execute your query chain, without actually executing it.")
        print("=" * 80)
        
        # ========== Example 1: Pure SQL Query ==========
        print("\n" + "=" * 80)
        print("Example 1: Pure SQL Query (Lazy)")
        print("=" * 80)
        print("\nCode:")
        print("  ds.select('*').filter(ds.age > 25).filter(ds.salary > 60000)")
        
        ds = DataStore.from_file(csv_file)
        result = ds.select('*').filter(ds.age > 25).filter(ds.salary > 60000)
        
        result.explain()
        
        # ========== Example 2: Mixed Operations (User's Example) ==========
        print("\n" + "=" * 80)
        print("Example 2: Mixed SQL and Pandas Operations")
        print("=" * 80)
        print("\nCode:")
        print("  result = (")
        print("      ds.select('*')                              # SQL 1")
        print("      .filter(ds.age > 25)                        # SQL 2")
        print("      .add_prefix('p1_')                          # Pandas 1: executes")
        print("      .filter(ds.p1_salary > 55000)               # SQL 3: on cached df")
        print("      .rename(columns={'p1_id': 'final_id'})      # Pandas 2: on cached df")
        print("      .filter(ds.final_id > 2)                    # SQL 4: on cached df")
        print("      .add_suffix('_end')                         # Pandas 3: on cached df")
        print("  )")
        
        ds = DataStore.from_file(csv_file)
        result = (
            ds.select('*')                              # SQL 1
            .filter(ds.age > 25)                        # SQL 2
            .add_prefix('p1_')                          # Pandas 1: executes
            .filter(ds.p1_salary > 55000)               # SQL 3: on cached df
            .rename(columns={'p1_id': 'final_id'})      # Pandas 2: on cached df
            .filter(ds.final_id > 2)                    # SQL 4: on cached df
            .add_suffix('_end')                         # Pandas 3: on cached df
        )
        
        result.explain()
        
        # ========== Example 3: Verbose Mode ==========
        print("\n" + "=" * 80)
        print("Example 3: Verbose Mode (More Details)")
        print("=" * 80)
        print("\nSame query as Example 2, but with verbose=True:")
        
        result.explain(verbose=True)
        
        # ========== Example 4: Pandas-First Operations ==========
        print("\n" + "=" * 80)
        print("Example 4: Pandas-First Operations")
        print("=" * 80)
        print("\nCode:")
        print("  ds.select('*').add_prefix('emp_').filter(ds.emp_age > 28)")
        
        ds = DataStore.from_file(csv_file)
        result = ds.select('*').add_prefix('emp_').filter(ds.emp_age > 28)
        
        result.explain()
        
        # ========== Example 5: Comparing Approaches ==========
        print("\n" + "=" * 80)
        print("Example 5: Comparing Different Approaches")
        print("=" * 80)
        
        ds = DataStore.from_file(csv_file)
        
        print("\nApproach A: Filter first (more efficient)")
        print("─" * 40)
        approach_a = (
            ds.select('*')
            .filter(ds.age > 30)              # Filter early (on source data)
            .add_prefix('emp_')               # Then execute
        )
        approach_a.explain()
        
        print("\nApproach B: Execute first (less efficient)")
        print("─" * 40)
        approach_b = (
            ds.select('*')
            .add_prefix('emp_')               # Execute all data
            .filter(ds.emp_age > 30)          # Then filter (on DataFrame)
        )
        approach_b.explain()
        
        # ========== Key Insights ==========
        print("\n" + "=" * 80)
        print("Key Insights from explain()")
        print("=" * 80)
        print("""
📊 Phase 1: SQL Query Building (Lazy)
   • SQL operations build up a query without executing
   • Very efficient - no data movement
   • Best place to do filtering, aggregation

🔄 Phase 2: Execution Point
   • First pandas operation triggers execution
   • SQL query is executed and result is cached
   • After this, all operations work on in-memory DataFrame

🐼 Phase 3: Operations on Executed DataFrame
   • SQL operations use chDB to query the cached DataFrame
   • Pandas operations work directly on the DataFrame
   • All operations are now in-memory

💡 Best Practices:
   1. Do heavy filtering in Phase 1 (SQL) when possible
   2. Use pandas operations when you need functionality SQL doesn't have
   3. After execution, both SQL and pandas operations work on cache
   4. Use explain() to verify your query will execute efficiently
        """)
        
        print("=" * 80)
        print("explain() guarantees NO execution or data modification!")
        print("It's safe to call anytime to understand what will happen.")
        print("=" * 80)
        
    finally:
        # Cleanup
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    main()

