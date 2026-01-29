"""
Example demonstrating the Mixed Execution Engine in DataStore.

This example showcases how DataStore enables arbitrary mixing of SQL-style
operations with pandas DataFrame operations, leveraging chDB's ability to
execute SQL on pandas DataFrames.
"""

from datastore import DataStore
import tempfile
import os


def create_sample_data():
    """Create sample data for demonstration."""
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "employees.csv")
    
    with open(csv_file, "w") as f:
        f.write("id,name,age,salary,department,hire_date,active\n")
        f.write("1,Alice,28,65000,Engineering,2020-01-15,1\n")
        f.write("2,Bob,32,70000,Sales,2019-03-20,1\n")
        f.write("3,Charlie,29,62000,Engineering,2021-06-10,1\n")
        f.write("4,David,35,80000,Marketing,2018-09-05,1\n")
        f.write("5,Eve,31,72000,Sales,2020-11-12,1\n")
        f.write("6,Frank,27,58000,Engineering,2022-02-28,0\n")
        f.write("7,Grace,33,75000,Marketing,2019-07-19,1\n")
        f.write("8,Henry,30,68000,Sales,2021-04-03,1\n")
        f.write("9,Iris,34,78000,Engineering,2018-12-14,1\n")
        f.write("10,Jack,26,55000,Marketing,2022-08-25,1\n")
    
    return csv_file, temp_dir


def main():
    csv_file, temp_dir = create_sample_data()
    
    try:
        print("=" * 80)
        print("DataStore Mixed Execution Engine Demo")
        print("=" * 80)
        print("\nThis demo shows how you can mix SQL and pandas operations")
        print("in ANY ORDER and it just works!")
        print("=" * 80)
        
        ds = DataStore.from_file(csv_file)
        
        # ========== Example 1: SQL → Pandas → SQL ==========
        print("\n" + "=" * 80)
        print("Example 1: SQL → Pandas → SQL")
        print("=" * 80)
        
        print("\nCode:")
        print("  ds.filter(ds.age > 28)")
        print("    .add_prefix('emp_')")
        print("    .filter(ds.emp_salary > 65000)")
        
        result1 = (ds
            .filter(ds.age > 28)                          # SQL: WHERE age > 28
            .add_prefix('emp_')                           # Pandas: execute and rename
            .filter(ds.emp_salary > 65000))               # SQL on DataFrame!
        
        print(f"\nExecuted: {result1._executed}")
        print(f"Result:\n{result1.to_df()}")
        
        # ========== Example 2: Pandas → SQL → Pandas ==========
        print("\n" + "=" * 80)
        print("Example 2: Pandas → SQL → Pandas")
        print("=" * 80)
        
        print("\nCode:")
        print("  ds.rename(columns={'id': 'ID', 'name': 'NAME'})")
        print("    .filter(ds.ID > 5)")
        print("    .sort_values('salary', ascending=False)")
        
        result2 = (ds
            .rename(columns={'id': 'ID', 'name': 'NAME'})  # Pandas: execute
            .filter(ds.ID > 5)                            # SQL on DataFrame
            .sort_values('salary', ascending=False))      # Pandas on DataFrame
        
        print(f"\nResult:\n{result2.to_df()}")
        
        # ========== Example 3: Complex Alternating Pattern ==========
        print("\n" + "=" * 80)
        print("Example 3: SQL → Pandas → SQL → Pandas → SQL → Pandas")
        print("=" * 80)
        
        print("\nCode:")
        print("  ds.select('*')")
        print("    .filter(ds.active == 1)")
        print("    .assign(bonus=lambda x: x['salary'] * 0.1)")
        print("    .filter(ds.bonus > 6000)")
        print("    .add_prefix('final_')")
        print("    .select('final_id', 'final_name', 'final_salary', 'final_bonus')")
        
        result3 = (ds
            .select('*')                                  # SQL 1
            .filter(ds.active == 1)                       # SQL 2
            .assign(bonus=lambda x: x['salary'] * 0.1)   # Pandas (executes)
            .filter(ds.bonus > 6000)                      # SQL 3 on DataFrame!
            .add_prefix('final_')                         # Pandas
            .select('final_id', 'final_name', 'final_salary', 'final_bonus'))  # SQL 4 on DataFrame!
        
        print(f"\nResult:\n{result3.to_df()}")
        
        # ========== Example 4: Feature Engineering Pipeline ==========
        print("\n" + "=" * 80)
        print("Example 4: Real-World Feature Engineering Pipeline")
        print("=" * 80)
        
        result4 = (ds
            # Step 1: SQL filtering (efficient on large data)
            .select('*')
            .filter(ds.active == 1)
            .filter(ds.salary > 60000)
            
            # Step 2: Feature engineering (pandas)
            .assign(
                salary_k=lambda x: x['salary'] / 1000,
                age_group=lambda x: pd.cut(x['age'], 
                                          bins=[0, 30, 35, 100],
                                          labels=['Junior', 'Mid', 'Senior'])
            )
            
            # Step 3: More SQL filtering on engineered features
            .filter(ds.salary_k > 65)
            
            # Step 4: Data cleaning (pandas)
            .fillna(0)
            .drop_duplicates()
            
            # Step 5: Final SQL selection
            .select('id', 'name', 'age_group', 'salary_k')
            
            # Step 6: Final pandas transformation
            .add_prefix('employee_')
            .sort_values('employee_salary_k', ascending=False))
        
        print("\nFinal Result:")
        print(result4.to_df())
        
        # ========== Example 5: Conditional Operations ==========
        print("\n" + "=" * 80)
        print("Example 5: Mixing SQL Filters with Pandas Conditionals")
        print("=" * 80)
        
        result5 = (ds
            .filter(ds.age > 27)                          # SQL
            .assign(performance=lambda x: x['salary'] / x['age'])  # Pandas
            .filter(ds.performance > 2000)                # SQL on new column!
            .where(lambda x: x['performance'] < 2500, None)  # Pandas where
            .dropna())                                    # Pandas
        
        print(f"\nEmployees with high performance (2000-2500):")
        print(result5[['name', 'age', 'salary', 'performance']].to_df())
        
        # ========== Example 6: Verification ==========
        print("\n" + "=" * 80)
        print("Example 6: Execution Model Verification")
        print("=" * 80)
        
        # Track execution state
        ds1 = ds.select('*')
        print(f"\n1. After select(): executed={ds1._executed}")
        
        ds2 = ds1.filter(ds.age > 30)
        print(f"2. After SQL filter(): executed={ds2._executed}")
        
        ds3 = ds2.add_prefix('x_')
        print(f"3. After add_prefix(): executed={ds3._executed} ← Executes!")
        
        ds4 = ds3.filter(ds.x_age > 32)
        print(f"4. After another filter(): executed={ds4._executed}")
        print(f"   (SQL executed on DataFrame using chDB!)")
        
        ds5 = ds4.fillna(0)
        print(f"5. After fillna(): executed={ds5._executed}")
        
        # ========== Summary ==========
        print("\n" + "=" * 80)
        print("Summary: Mixed Execution Engine Benefits")
        print("=" * 80)
        print("\n✅ Advantages:")
        print("  1. Mix SQL and pandas operations in ANY order")
        print("  2. SQL filters large datasets efficiently")
        print("  3. Pandas enables complex transformations")
        print("  4. SQL on DataFrames (via chDB) enables SQL after pandas")
        print("  5. Natural, intuitive API - no mental mode switching")
        print("  6. Optimal performance through lazy SQL + cached DataFrames")
        
        print("\n📊 How it works:")
        print("  • SQL operations before execution: Build query (lazy)")
        print("  • First pandas operation: Execute SQL, cache result (eager)")
        print("  • SQL operations after execution: Use chDB on DataFrame")
        print("  • Pandas operations after execution: Work on cache")
        
        print("\n🚀 Result: The most flexible data manipulation framework!")
        print("=" * 80)
        
    finally:
        # Cleanup
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    import pandas as pd
    main()

