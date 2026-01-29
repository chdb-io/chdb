"""
Example demonstrating pandas-style convenience methods in DataStore.

This example shows how to use describe(), head(), tail(), and other
pandas-like methods with DataStore.
"""

from datastore import DataStore
import tempfile
import os


def create_sample_data():
    """Create a temporary CSV file with sample data."""
    temp_dir = tempfile.mkdtemp()
    csv_file = os.path.join(temp_dir, "sales_data.csv")
    
    with open(csv_file, "w") as f:
        f.write("id,product,category,price,quantity,date,region\n")
        f.write("1,Laptop,Electronics,1200,5,2024-01-15,North\n")
        f.write("2,Mouse,Electronics,25,50,2024-01-16,South\n")
        f.write("3,Keyboard,Electronics,75,30,2024-01-17,East\n")
        f.write("4,Monitor,Electronics,300,15,2024-01-18,West\n")
        f.write("5,Desk,Furniture,450,10,2024-01-19,North\n")
        f.write("6,Chair,Furniture,200,25,2024-01-20,South\n")
        f.write("7,Notebook,Office,5,200,2024-01-21,East\n")
        f.write("8,Pen,Office,2,500,2024-01-22,West\n")
        f.write("9,Stapler,Office,15,100,2024-01-23,North\n")
        f.write("10,Lamp,Furniture,60,40,2024-01-24,South\n")
    
    return csv_file, temp_dir


def main():
    # Create sample data
    csv_file, temp_dir = create_sample_data()
    
    try:
        print("=" * 70)
        print("DataStore Pandas-Style Convenience Methods Demo")
        print("=" * 70)
        
        # Create DataStore from file
        ds = DataStore.from_file(csv_file)
        
        # Example 1: describe() / desc()
        print("\n1. DESCRIBE - Statistical Summary")
        print("-" * 70)
        stats = ds.select("*").describe()
        print(stats)
        
        # Short version
        print("\nUsing short version desc():")
        stats_short = ds.select("*").desc()
        print(stats_short)
        
        # Example 2: head()
        print("\n2. HEAD - First N Rows")
        print("-" * 70)
        print("First 5 rows (default):")
        print(ds.head())
        
        print("\nFirst 3 rows:")
        print(ds.select("*").head(3))
        
        # Example 3: tail()
        print("\n3. TAIL - Last N Rows")
        print("-" * 70)
        print("Last 3 rows:")
        print(ds.select("*").tail(3))
        
        # Example 4: shape
        print("\n4. SHAPE - Dataset Dimensions")
        print("-" * 70)
        rows, cols = ds.select("*").shape
        print(f"Dataset shape: {rows} rows × {cols} columns")
        
        # Example 5: columns
        print("\n5. COLUMNS - Column Names")
        print("-" * 70)
        cols = ds.select("*").columns
        print(f"Columns: {list(cols)}")
        
        # Example 6: count()
        print("\n6. COUNT - Non-null Values Per Column")
        print("-" * 70)
        counts = ds.select("*").count()
        print(counts)
        
        # Example 7: sample()
        print("\n7. SAMPLE - Random Sample")
        print("-" * 70)
        print("Random 3 rows:")
        print(ds.sample(n=3, random_state=42))
        
        # Example 8: Chaining with filters
        print("\n8. CHAINING - Combine with Filters")
        print("-" * 70)
        print("Electronics products (first 3):")
        electronics = (
            ds.select("product", "category", "price")
            .filter(ds.category == "Electronics")
            .head(3)
        )
        print(electronics)
        
        # Example 9: Statistics with filters
        print("\n9. FILTERED STATISTICS")
        print("-" * 70)
        print("Statistics for high-value items (price > 100):")
        high_value_stats = (
            ds.select("*")
            .filter(ds.price > 100)
            .describe()
        )
        print(high_value_stats)
        
        # Example 10: info()
        print("\n10. INFO - Dataset Summary")
        print("-" * 70)
        ds.select("*").info()
        
        # Example 11: Custom percentiles in describe
        print("\n11. CUSTOM PERCENTILES")
        print("-" * 70)
        custom_stats = ds.select("*").describe(percentiles=[0.1, 0.5, 0.9])
        print(custom_stats)
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
    finally:
        # Cleanup
        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


if __name__ == "__main__":
    main()

