"""
Example demonstrating pandas DataFrame compatibility layer in DataStore.

This example shows how DataStore now supports the full pandas DataFrame API,
allowing you to use familiar pandas methods while leveraging DataStore's
query-building capabilities.
"""

from datastore import DataStore
import tempfile
import os


def create_sample_data():
    """Create temporary CSV files with sample data."""
    temp_dir = tempfile.mkdtemp()
    
    # Sales data
    sales_file = os.path.join(temp_dir, "sales.csv")
    with open(sales_file, "w") as f:
        f.write("id,product,category,price,quantity,date,region,sales_rep\n")
        f.write("1,Laptop,Electronics,1200,5,2024-01-15,North,Alice\n")
        f.write("2,Mouse,Electronics,25,50,2024-01-16,South,Bob\n")
        f.write("3,Keyboard,Electronics,75,30,2024-01-17,East,Charlie\n")
        f.write("4,Monitor,Electronics,300,15,2024-01-18,West,Alice\n")
        f.write("5,Desk,Furniture,450,10,2024-01-19,North,Bob\n")
        f.write("6,Chair,Furniture,200,25,2024-01-20,South,Charlie\n")
        f.write("7,Notebook,Office,5,200,2024-01-21,East,Alice\n")
        f.write("8,Pen,Office,2,500,2024-01-22,West,Bob\n")
        f.write("9,Stapler,Office,15,100,2024-01-23,North,Charlie\n")
        f.write("10,Lamp,Furniture,60,40,2024-01-24,South,Alice\n")
    
    # Employees data
    employees_file = os.path.join(temp_dir, "employees.csv")
    with open(employees_file, "w") as f:
        f.write("name,employee_id,department,hire_date\n")
        f.write("Alice,101,Sales,2020-01-15\n")
        f.write("Bob,102,Sales,2019-03-20\n")
        f.write("Charlie,103,Marketing,2021-06-10\n")
    
    return sales_file, employees_file, temp_dir


def main():
    # Create sample data
    sales_file, employees_file, temp_dir = create_sample_data()
    
    try:
        print("=" * 80)
        print("DataStore Pandas DataFrame Compatibility Layer Demo")
        print("=" * 80)
        
        # Create DataStore from file
        ds = DataStore.from_file(sales_file)
        
        # ========== Section 1: Properties ==========
        print("\n" + "=" * 80)
        print("SECTION 1: DataFrame Properties")
        print("=" * 80)
        
        print(f"\nShape: {ds.shape}")
        print(f"Size: {ds.size}")
        print(f"Columns: {list(ds.columns)}")
        print(f"Data types:\n{ds.dtypes}")
        print(f"Empty: {ds.empty}")
        
        # ========== Section 2: Statistical Methods ==========
        print("\n" + "=" * 80)
        print("SECTION 2: Statistical Methods")
        print("=" * 80)
        
        print("\nMean values (numeric columns):")
        print(ds.mean(numeric_only=True))
        
        print("\nMedian values (numeric columns):")
        print(ds.median(numeric_only=True))
        
        print("\nStandard deviation:")
        print(ds.std(numeric_only=True))
        
        print("\nMin and Max:")
        print(f"Min:\n{ds.min(numeric_only=True)}")
        print(f"Max:\n{ds.max(numeric_only=True)}")
        
        print("\nQuantiles (25%, 50%, 75%):")
        print(ds.quantile([0.25, 0.5, 0.75], numeric_only=True))
        
        print("\nCorrelation matrix:")
        print(ds.corr(numeric_only=True))
        
        # ========== Section 3: Data Manipulation ==========
        print("\n" + "=" * 80)
        print("SECTION 3: Data Manipulation")
        print("=" * 80)
        
        # Drop columns
        print("\nDrop 'date' column:")
        ds_no_date = ds.drop(columns=['date'])
        print(f"Columns after drop: {list(ds_no_date.columns)}")
        
        # Rename columns
        print("\nRename columns:")
        ds_renamed = ds.rename(columns={'price': 'unit_price', 'quantity': 'qty'})
        print(f"Columns after rename: {list(ds_renamed.columns)}")
        
        # Sort values
        print("\nSort by price (descending):")
        ds_sorted = ds.sort_values('price', ascending=False)
        print(ds_sorted.head(3))
        
        # Add new column
        print("\nAdd 'revenue' column (price * quantity):")
        ds_with_revenue = ds.assign(revenue=lambda x: x['price'] * x['quantity'])
        print(ds_with_revenue[['product', 'price', 'quantity', 'revenue']].head())
        
        # Get top N
        print("\nTop 3 most expensive products:")
        print(ds.nlargest(3, 'price')[['product', 'price']])
        
        print("\nTop 3 cheapest products:")
        print(ds.nsmallest(3, 'price')[['product', 'price']])
        
        # Drop duplicates
        print("\nUnique categories:")
        unique_categories = ds[['category']].drop_duplicates()
        print(unique_categories)
        
        # ========== Section 4: Filtering and Selection ==========
        print("\n" + "=" * 80)
        print("SECTION 4: Filtering and Selection")
        print("=" * 80)
        
        # Column selection
        print("\nSelect specific columns:")
        print(ds[['product', 'price', 'quantity']].head(3))
        
        # Filtering (using DataStore filter, then pandas operations)
        print("\nFilter expensive items (price > 100) and get stats:")
        expensive = ds.filter(ds.price > 100)
        print(f"Count: {len(expensive._get_df())}")
        print(f"Mean price: ${expensive.mean(numeric_only=True)['price']:.2f}")
        
        # ========== Section 5: Aggregation and Grouping ==========
        print("\n" + "=" * 80)
        print("SECTION 5: Aggregation and Grouping")
        print("=" * 80)
        
        # Using pandas agg
        print("\nAggregate functions on numeric columns:")
        agg_result = ds.agg({
            'price': ['mean', 'min', 'max'],
            'quantity': ['sum', 'mean']
        })
        print(agg_result)
        
        # ========== Section 6: Reshaping ==========
        print("\n" + "=" * 80)
        print("SECTION 6: Reshaping")
        print("=" * 80)
        
        # Melt
        print("\nMelt data (unpivot):")
        melted = ds.melt(
            id_vars=['product', 'category'],
            value_vars=['price', 'quantity'],
            var_name='metric',
            value_name='value'
        )
        print(melted.head(6))
        
        # Pivot table
        print("\nPivot table - average price by category and region:")
        pivot = ds.pivot_table(
            values='price',
            index='category',
            columns='region',
            aggfunc='mean'
        )
        print(pivot)
        
        # ========== Section 7: Function Application ==========
        print("\n" + "=" * 80)
        print("SECTION 7: Function Application")
        print("=" * 80)
        
        # Apply function
        print("\nApply function - calculate 10% discount:")
        ds_discount = ds.assign(
            discounted_price=lambda x: x['price'] * 0.9
        )
        print(ds_discount[['product', 'price', 'discounted_price']].head())
        
        # Transform
        print("\nTransform - normalize prices by category:")
        ds_normalized = ds.copy()
        # Note: transform works on pandas DataFrame
        
        # ========== Section 8: Merging DataStores ==========
        print("\n" + "=" * 80)
        print("SECTION 8: Merging DataStores")
        print("=" * 80)
        
        # Create employee DataStore
        employees = DataStore.from_file(employees_file)
        
        print("\nOriginal sales data:")
        print(ds[['product', 'sales_rep']].head(3))
        
        print("\nEmployee data:")
        print(employees._get_df())
        
        # Merge with employee data
        print("\nMerge sales with employee info:")
        merged = ds.merge(
            employees,
            left_on='sales_rep',
            right_on='name',
            how='left'
        )
        print(merged[['product', 'sales_rep', 'employee_id', 'department']].head())
        
        # ========== Section 9: Chaining Operations ==========
        print("\n" + "=" * 80)
        print("SECTION 9: Chaining Operations")
        print("=" * 80)
        
        print("\nComplex chaining: filter -> add column -> sort -> select top 3")
        result = (ds
                  .filter(ds.category == 'Electronics')
                  .assign(revenue=lambda x: x['price'] * x['quantity'])
                  .sort_values('revenue', ascending=False)
                  .nlargest(3, 'revenue'))
        
        print(result[['product', 'price', 'quantity', 'revenue']])
        
        # ========== Section 10: Data Export ==========
        print("\n" + "=" * 80)
        print("SECTION 10: Data Export")
        print("=" * 80)
        
        # Export to CSV
        output_csv = os.path.join(temp_dir, "output.csv")
        ds.to_csv(output_csv, index=False)
        print(f"\nExported to CSV: {output_csv}")
        
        # Export to JSON
        output_json = os.path.join(temp_dir, "output.json")
        ds.to_json(output_json, orient='records')
        print(f"Exported to JSON: {output_json}")
        
        # Get as numpy array
        arr = ds.to_numpy()
        print(f"\nAs NumPy array shape: {arr.shape}")
        
        # ========== Section 11: Iteration ==========
        print("\n" + "=" * 80)
        print("SECTION 11: Iteration")
        print("=" * 80)
        
        print("\nIterate over rows (first 3):")
        for idx, row in enumerate(ds.iterrows()):
            if idx >= 3:
                break
            row_idx, row_data = row
            print(f"Row {row_idx}: {row_data['product']} - ${row_data['price']}")
        
        print("\nIterate as named tuples (first 3):")
        for idx, row in enumerate(ds.itertuples()):
            if idx >= 3:
                break
            print(f"  {row.product}: ${row.price}")
        
        # ========== Section 12: Missing Data Handling ==========
        print("\n" + "=" * 80)
        print("SECTION 12: Missing Data Handling")
        print("=" * 80)
        
        # Check for missing values
        print("\nCheck for missing values:")
        print(ds.isna().sum())
        
        # Fill missing values (if any)
        ds_filled = ds.fillna(0)
        print("Missing values filled with 0")
        
        # Drop rows with missing values (if any)
        ds_no_na = ds.dropna()
        print(f"Shape after dropna: {ds_no_na.shape}")
        
        # ========== Section 13: Type Conversion ==========
        print("\n" + "=" * 80)
        print("SECTION 13: Type Conversion")
        print("=" * 80)
        
        print("\nOriginal dtypes:")
        print(ds.dtypes)
        
        print("\nConvert 'quantity' to float:")
        ds_converted = ds.astype({'quantity': 'float64'})
        print(f"New dtype for quantity: {ds_converted.dtypes['quantity']}")
        
        # ========== Section 14: Combining DataStore and Pandas Methods ==========
        print("\n" + "=" * 80)
        print("SECTION 14: Combining DataStore SQL and Pandas Operations")
        print("=" * 80)
        
        print("\nMix DataStore query building with pandas operations:")
        
        # Step 1: Use DataStore to filter and select
        ds_filtered = (ds
                      .select('product', 'category', 'price', 'quantity', 'region')
                      .filter(ds.price > 50))
        
        print(f"\nAfter DataStore filter (price > 50): {ds_filtered.shape[0]} rows")
        
        # Step 2: Apply pandas operations
        result = (ds_filtered
                  .assign(revenue=lambda x: x['price'] * x['quantity'])
                  .sort_values('revenue', ascending=False)
                  .groupby('category')
                  .agg({
                      'revenue': 'sum',
                      'quantity': 'sum',
                      'product': 'count'
                  })
                  .rename(columns={'product': 'product_count'}))
        
        print("\nRevenue by category:")
        print(result)
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. DataStore supports 180+ pandas DataFrame methods")
        print("2. All DataFrame methods that return DataFrames now return DataStore")
        print("3. You can chain DataStore SQL operations with pandas operations")
        print("4. Internal DataFrame is cached for performance")
        print("5. DataStore remains immutable (inplace=True not supported)")
        print("\nFor complete feature checklist, see: docs/PANDAS_COMPATIBILITY.md")
        print("=" * 80)
        
    finally:
        # Cleanup
        for file in [sales_file, employees_file]:
            if os.path.exists(file):
                os.unlink(file)
        
        for filename in ['output.csv', 'output.json']:
            filepath = os.path.join(temp_dir, filename)
            if os.path.exists(filepath):
                os.unlink(filepath)
        
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    main()


