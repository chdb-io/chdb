"""
Pandas Common Operations - Based on Kaggle Popular Notebooks

This script consolidates common Pandas operation patterns from the most popular
data science notebooks on Kaggle, including:

1. Data Loading & Initial Exploration
2. Data Cleaning & Preprocessing
3. Feature Engineering
4. Data Aggregation & Grouping
5. Data Merging & Joining
6. Time Series Operations
7. Advanced Indexing & Selection
8. Data Quality Checks
9. Performance Optimization
10. Data Export

Data Source: Simulated E-commerce Sales Dataset
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

print("=" * 80)
print("PANDAS COMMON OPERATIONS - KAGGLE BEST PRACTICES")
print("=" * 80)

# ============================================================================
# 1. CREATE TEST DATASET
# ============================================================================
print("\n" + "=" * 80)
print("1. CREATING TEST DATASET")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Generate sales data
n_records = 1000

data = {
    'order_id': [f'ORD{str(i).zfill(5)}' for i in range(1, n_records + 1)],
    'order_date': np.random.choice(dates, n_records),
    'customer_id': np.random.randint(1001, 1201, n_records),
    'customer_name': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'], n_records),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports'], n_records),
    'product_name': np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E'], n_records),
    'quantity': np.random.randint(1, 10, n_records),
    'unit_price': np.random.uniform(10, 500, n_records),
    'discount_pct': np.random.choice([0, 5, 10, 15, 20], n_records),
    'shipping_cost': np.random.uniform(0, 50, n_records),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
    'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer', None], n_records),
    'customer_age': np.random.randint(18, 70, n_records),
    'is_member': np.random.choice([True, False], n_records),
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values
df.loc[np.random.choice(df.index, 50, replace=False), 'discount_pct'] = np.nan
df.loc[np.random.choice(df.index, 30, replace=False), 'customer_age'] = np.nan

print(f"\nDataset created successfully with {len(df)} records")
print("\nFirst 10 rows:")
print(df.head(10))

# ============================================================================
# 2. DATA LOADING & INITIAL EXPLORATION
# ============================================================================
print("\n" + "=" * 80)
print("2. DATA LOADING & INITIAL EXPLORATION")
print("=" * 80)

# 2.1 View basic information
print("\n" + "-" * 50)
print("2.1 Basic Dataset Information:")
print("-" * 50)
df.info()

# 2.2 View data shape
print(f"\nDataset shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 2.3 Statistical summary for numeric columns
print("\n" + "-" * 50)
print("2.3 Statistical Summary for Numeric Columns:")
print("-" * 50)
print(df.describe())

# 2.4 Statistical summary for categorical columns
print("\n" + "-" * 50)
print("2.4 Statistical Summary for Categorical Columns:")
print("-" * 50)
print(df.describe(include=['object', 'bool']))

# 2.5 Check missing values
print("\n" + "-" * 50)
print("2.5 Missing Values Statistics:")
print("-" * 50)
missing_data = pd.DataFrame(
    {'Missing_Count': df.isnull().sum(), 'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)}
)
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data)

# 2.6 View unique values
print("\n" + "-" * 50)
print("2.6 Unique Values Count per Column:")
print("-" * 50)
print(df.nunique().sort_values(ascending=False))

# 2.7 View data types
print("\n" + "-" * 50)
print("2.7 Data Types:")
print("-" * 50)
print(df.dtypes)

# 2.8 Random sampling
print("\n" + "-" * 50)
print("2.8 Random Sample of 5 Records:")
print("-" * 50)
print(df.sample(5))

# ============================================================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("3. DATA CLEANING & PREPROCESSING")
print("=" * 80)

# 3.1 Handle missing values - Filling
print("\n" + "-" * 50)
print("3.1 Handling Missing Values:")
print("-" * 50)
df_clean = df.copy()

# Fill numeric missing values with median
df_clean['customer_age'].fillna(df_clean['customer_age'].median(), inplace=True)

# Fill categorical missing values with mode
df_clean['discount_pct'].fillna(df_clean['discount_pct'].mode()[0], inplace=True)

# Fill with specific value
df_clean['payment_method'].fillna('Unknown', inplace=True)

print("Missing values after handling:")
print(df_clean.isnull().sum())

# 3.2 Remove duplicates
print("\n" + "-" * 50)
print("3.2 Removing Duplicates:")
print("-" * 50)
print(f"Before removal: {len(df_clean)} rows")
df_clean.drop_duplicates(inplace=True)
print(f"After removal: {len(df_clean)} rows")

# 3.3 Data type conversion
print("\n" + "-" * 50)
print("3.3 Data Type Conversion:")
print("-" * 50)
# Ensure date column is datetime type
df_clean['order_date'] = pd.to_datetime(df_clean['order_date'])

# Convert to category type to save memory
categorical_cols = ['product_category', 'region', 'payment_method', 'customer_name']
for col in categorical_cols:
    df_clean[col] = df_clean[col].astype('category')

print("Data types after conversion:")
print(df_clean.dtypes)

# 3.4 String processing
print("\n" + "-" * 50)
print("3.4 String Processing:")
print("-" * 50)
# Uniform case
df_clean['product_name'] = df_clean['product_name'].str.upper()

# Remove whitespace
df_clean['customer_name'] = df_clean['customer_name'].str.strip()

print("String processing example:")
print(df_clean[['product_name', 'customer_name']].head())

# 3.5 Conditional replacement
print("\n" + "-" * 50)
print("3.5 Conditional Replacement:")
print("-" * 50)
# Use loc for conditional replacement
df_clean.loc[df_clean['quantity'] > 7, 'quantity'] = 7

# Use replace
df_clean['region'] = df_clean['region'].replace({'North': 'N', 'South': 'S', 'East': 'E', 'West': 'W'})

print("Region distribution after replacement:")
print(df_clean['region'].value_counts())

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 80)
print("4. FEATURE ENGINEERING")
print("=" * 80)

# 4.1 Create new features - Numerical calculations
print("\n" + "-" * 50)
print("4.1 Creating New Features - Numerical Calculations:")
print("-" * 50)
df_clean['total_price'] = df_clean['quantity'] * df_clean['unit_price']
df_clean['discount_amount'] = df_clean['total_price'] * (df_clean['discount_pct'] / 100)
df_clean['final_amount'] = df_clean['total_price'] - df_clean['discount_amount'] + df_clean['shipping_cost']

print("New price-related features:")
print(df_clean[['total_price', 'discount_amount', 'final_amount']].head())

# 4.2 Extract features from dates
print("\n" + "-" * 50)
print("4.2 Extracting Features from Dates:")
print("-" * 50)
df_clean['year'] = df_clean['order_date'].dt.year
df_clean['month'] = df_clean['order_date'].dt.month
df_clean['day'] = df_clean['order_date'].dt.day
df_clean['dayofweek'] = df_clean['order_date'].dt.dayofweek
df_clean['quarter'] = df_clean['order_date'].dt.quarter
df_clean['week'] = df_clean['order_date'].dt.isocalendar().week
df_clean['is_weekend'] = df_clean['dayofweek'].isin([5, 6]).astype(int)

print("Features extracted from dates:")
print(df_clean[['order_date', 'year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']].head())

# 4.3 Binning operations
print("\n" + "-" * 50)
print("4.3 Binning Operations:")
print("-" * 50)
# Age grouping
df_clean['age_group'] = pd.cut(
    df_clean['customer_age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+']
)

# Price grouping
df_clean['price_category'] = pd.qcut(df_clean['unit_price'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

print("Features after binning:")
print(df_clean[['customer_age', 'age_group', 'unit_price', 'price_category']].head())

# 4.4 Conditional feature creation
print("\n" + "-" * 50)
print("4.4 Conditional Feature Creation:")
print("-" * 50)
# Using np.where
df_clean['high_value_order'] = np.where(df_clean['final_amount'] > 500, 1, 0)

# Using apply and lambda
df_clean['season'] = df_clean['month'].apply(
    lambda x: 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall' if x in [9, 10, 11] else 'Winter'
)

print("Conditional features:")
print(df_clean[['final_amount', 'high_value_order', 'month', 'season']].head())

# 4.5 Encoding categorical variables
print("\n" + "-" * 50)
print("4.5 Encoding Categorical Variables:")
print("-" * 50)
# One-Hot encoding
region_dummies = pd.get_dummies(df_clean['region'], prefix='region')
df_encoded = pd.concat([df_clean, region_dummies], axis=1)

# Label encoding
le = LabelEncoder()
df_clean['product_category_encoded'] = le.fit_transform(df_clean['product_category'])

print("Encoded features example:")
print(df_encoded[['region', 'region_E', 'region_N', 'region_S', 'region_W']].head())

# ============================================================================
# 5. DATA AGGREGATION & GROUPING
# ============================================================================
print("\n" + "=" * 80)
print("5. DATA AGGREGATION & GROUPING")
print("=" * 80)

# 5.1 Basic grouping and aggregation
print("\n" + "-" * 50)
print("5.1 Basic Grouping and Aggregation:")
print("-" * 50)
category_stats = (
    df_clean.groupby('product_category')
    .agg({'final_amount': ['sum', 'mean', 'median', 'std'], 'quantity': ['sum', 'mean'], 'order_id': 'count'})
    .round(2)
)

category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
print("Aggregated statistics by product category:")
print(category_stats)

# 5.2 Multi-column grouping
print("\n" + "-" * 50)
print("5.2 Multi-column Grouping:")
print("-" * 50)
region_category = (
    df_clean.groupby(['region', 'product_category'])['final_amount'].agg(['sum', 'mean', 'count']).round(2)
)

print("Sales statistics by region and category:")
print(region_category.head(10))

# 5.3 Custom aggregation functions
print("\n" + "-" * 50)
print("5.3 Custom Aggregation Functions:")
print("-" * 50)


def price_range(x):
    return x.max() - x.min()


custom_agg = (
    df_clean.groupby('product_category')['unit_price']
    .agg([('Min', 'min'), ('Max', 'max'), ('Range', price_range), ('Std', 'std')])
    .round(2)
)

print("Custom aggregation results:")
print(custom_agg)

# 5.4 Transform - Maintain original data shape
print("\n" + "-" * 50)
print("5.4 Transform - Maintain Original Shape:")
print("-" * 50)
df_clean['category_avg_price'] = df_clean.groupby('product_category')['final_amount'].transform('mean')
df_clean['price_vs_category_avg'] = df_clean['final_amount'] / df_clean['category_avg_price']

print("Transform example - Comparison with category average price:")
print(df_clean[['product_category', 'final_amount', 'category_avg_price', 'price_vs_category_avg']].head())

# 5.5 Ranking and cumulative calculations
print("\n" + "-" * 50)
print("5.5 Ranking and Cumulative Calculations:")
print("-" * 50)
df_clean['amount_rank'] = df_clean.groupby('product_category')['final_amount'].rank(ascending=False)
df_clean['cumulative_amount'] = df_clean.groupby('product_category')['final_amount'].cumsum()

print("Ranking and cumulative calculations:")
print(df_clean[['product_category', 'final_amount', 'amount_rank', 'cumulative_amount']].head(10))

# ============================================================================
# 6. DATA PIVOTING & CROSSTAB
# ============================================================================
print("\n" + "=" * 80)
print("6. DATA PIVOTING & CROSSTAB")
print("=" * 80)

# 6.1 Pivot table
print("\n" + "-" * 50)
print("6.1 Pivot Table:")
print("-" * 50)
pivot_table = pd.pivot_table(
    df_clean, values='final_amount', index='product_category', columns='region', aggfunc='sum', fill_value=0
).round(2)

print("Pivot table - Sales by region and category:")
print(pivot_table)

# 6.2 Pivot table with multiple aggregations
print("\n" + "-" * 50)
print("6.2 Pivot Table with Multiple Aggregations:")
print("-" * 50)
pivot_multi = pd.pivot_table(
    df_clean,
    values='final_amount',
    index='product_category',
    columns='is_member',
    aggfunc=['sum', 'mean', 'count'],
    fill_value=0,
).round(2)

print("Multi-aggregation pivot table:")
print(pivot_multi.head())

# 6.3 Crosstab
print("\n" + "-" * 50)
print("6.3 Crosstab:")
print("-" * 50)
crosstab = pd.crosstab(
    df_clean['product_category'],
    df_clean['region'],
    values=df_clean['final_amount'],
    aggfunc='sum',
    margins=True,  # Add totals
    margins_name='Total',
).round(2)

print("Crosstab with totals:")
print(crosstab)

# 6.4 Normalized crosstab
print("\n" + "-" * 50)
print("6.4 Normalized Crosstab:")
print("-" * 50)
crosstab_normalized = pd.crosstab(
    df_clean['product_category'], df_clean['region'], normalize='columns'  # Normalize by columns
).round(3)

print("Normalized crosstab (column percentages):")
print(crosstab_normalized)

# ============================================================================
# 7. DATA MERGING & JOINING
# ============================================================================
print("\n" + "=" * 80)
print("7. DATA MERGING & JOINING")
print("=" * 80)

# Create customer information table
customers = pd.DataFrame(
    {
        'customer_id': range(1001, 1201),
        'customer_segment': np.random.choice(['Premium', 'Regular', 'Basic'], 200),
        'registration_date': pd.date_range(start='2020-01-01', periods=200, freq='D'),
        'lifetime_value': np.random.uniform(100, 5000, 200),
    }
)

# Create product information table
products = pd.DataFrame(
    {
        'product_name': ['PRODUCT_A', 'PRODUCT_B', 'PRODUCT_C', 'PRODUCT_D', 'PRODUCT_E'],
        'product_cost': [50, 100, 150, 200, 250],
        'supplier': ['Supplier_1', 'Supplier_2', 'Supplier_1', 'Supplier_3', 'Supplier_2'],
    }
)

print("\nCustomer information table:")
print(customers.head())
print("\nProduct information table:")
print(products)

# 7.1 Inner Join
print("\n" + "-" * 50)
print("7.1 Inner Join:")
print("-" * 50)
df_with_customers = df_clean.merge(customers, on='customer_id', how='inner')

print(f"Original data: {len(df_clean)} rows")
print(f"After join: {len(df_with_customers)} rows")
print(df_with_customers[['customer_id', 'customer_segment', 'lifetime_value']].head())

# 7.2 Left Join
print("\n" + "-" * 50)
print("7.2 Left Join:")
print("-" * 50)
df_with_products = df_clean.merge(products, on='product_name', how='left')

print("Data after left join:")
print(df_with_products[['product_name', 'product_cost', 'supplier']].head())

# 7.3 Multi-key join
print("\n" + "-" * 50)
print("7.3 Multi-key Join:")
print("-" * 50)
df_sample1 = df_clean[['customer_id', 'product_category', 'final_amount']].head(100)
df_sample2 = df_clean[['customer_id', 'product_category', 'quantity']].head(100)

merged_multi = df_sample1.merge(df_sample2, on=['customer_id', 'product_category'], how='inner')

print("Multi-key join result:")
print(merged_multi.head())

# 7.4 Concat - Vertical concatenation
print("\n" + "-" * 50)
print("7.4 Concat - Vertical Concatenation:")
print("-" * 50)
df1 = df_clean.head(100)
df2 = df_clean.tail(100)

df_concat = pd.concat([df1, df2], ignore_index=True)
print(f"Concat result: {len(df_concat)} rows")

# ============================================================================
# 8. TIME SERIES OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("8. TIME SERIES OPERATIONS")
print("=" * 80)

# 8.1 Set date as index
print("\n" + "-" * 50)
print("8.1 Set Date as Index:")
print("-" * 50)
df_ts = df_clean.set_index('order_date').sort_index()

print("Time series data:")
print(df_ts.head())

# 8.2 Resampling
print("\n" + "-" * 50)
print("8.2 Resampling:")
print("-" * 50)
monthly_sales = df_ts['final_amount'].resample('M').agg(['sum', 'mean', 'count']).round(2)
monthly_sales.columns = ['Total_Sales', 'Avg_Order_Amount', 'Order_Count']

print("Monthly sales statistics:")
print(monthly_sales.head())

# 8.3 Rolling window calculations
print("\n" + "-" * 50)
print("8.3 Rolling Window Calculations:")
print("-" * 50)
df_ts['rolling_7day_avg'] = df_ts['final_amount'].rolling(window=7).mean()
df_ts['rolling_30day_sum'] = df_ts['final_amount'].rolling(window=30).sum()

print("Rolling window calculations:")
print(df_ts[['final_amount', 'rolling_7day_avg', 'rolling_30day_sum']].tail(10))

# 8.4 Shift operations
print("\n" + "-" * 50)
print("8.4 Shift Operations:")
print("-" * 50)
df_ts['prev_day_amount'] = df_ts['final_amount'].shift(1)
df_ts['next_day_amount'] = df_ts['final_amount'].shift(-1)
df_ts['day_over_day_change'] = df_ts['final_amount'] - df_ts['prev_day_amount']

print("Shift operations:")
print(df_ts[['final_amount', 'prev_day_amount', 'next_day_amount', 'day_over_day_change']].tail(10))

# 8.5 Time difference calculations
print("\n" + "-" * 50)
print("8.5 Time Difference Calculations:")
print("-" * 50)
df_clean_sorted = df_clean.sort_values('order_date')
df_clean_sorted['days_since_first_order'] = (
    df_clean_sorted['order_date'] - df_clean_sorted['order_date'].min()
).dt.days

print("Time difference calculations:")
print(df_clean_sorted[['order_date', 'days_since_first_order']].head())

# ============================================================================
# 9. ADVANCED INDEXING & SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("9. ADVANCED INDEXING & SELECTION")
print("=" * 80)

# 9.1 Conditional filtering
print("\n" + "-" * 50)
print("9.1 Conditional Filtering:")
print("-" * 50)
# Single condition
high_value = df_clean[df_clean['final_amount'] > 1000]
print(f"High-value orders: {len(high_value)}")

# Multiple conditions (AND)
premium_electronics = df_clean[(df_clean['product_category'] == 'Electronics') & (df_clean['final_amount'] > 500)]
print(f"High-value electronics orders: {len(premium_electronics)}")

# Multiple conditions (OR)
weekend_or_member = df_clean[(df_clean['is_weekend'] == 1) | (df_clean['is_member'] == True)]
print(f"Weekend or member orders: {len(weekend_or_member)}")

# 9.2 isin method
print("\n" + "-" * 50)
print("9.2 isin Method:")
print("-" * 50)
selected_categories = df_clean[df_clean['product_category'].isin(['Electronics', 'Books'])]
print(f"Selected category orders: {len(selected_categories)}")

# Reverse selection
not_selected = df_clean[~df_clean['product_category'].isin(['Electronics', 'Books'])]
print(f"Non-selected category orders: {len(not_selected)}")

# 9.3 query method
print("\n" + "-" * 50)
print("9.3 query Method:")
print("-" * 50)
result = df_clean.query('final_amount > 500 and is_member == True')
print(f"Query result: {len(result)} records")

# Using variables
threshold = 300
result2 = df_clean.query('final_amount > @threshold')
print(f"Orders with amount > {threshold}: {len(result2)} records")

# 9.4 loc and iloc
print("\n" + "-" * 50)
print("9.4 loc and iloc:")
print("-" * 50)
# loc - label-based
subset1 = df_clean.loc[:10, ['order_id', 'final_amount', 'product_category']]
print("loc selection:")
print(subset1.head())

# iloc - position-based
subset2 = df_clean.iloc[:5, [0, -3, -2]]
print("\niloc selection:")
print(subset2)

# 9.5 nlargest and nsmallest
print("\n" + "-" * 50)
print("9.5 nlargest and nsmallest:")
print("-" * 50)
top_10_orders = df_clean.nlargest(10, 'final_amount')
print("Top 10 orders by sales amount:")
print(top_10_orders[['order_id', 'final_amount', 'product_category']])

bottom_5_orders = df_clean.nsmallest(5, 'final_amount')
print("\nBottom 5 orders by sales amount:")
print(bottom_5_orders[['order_id', 'final_amount', 'product_category']])

# ============================================================================
# 10. DATA QUALITY CHECKS & VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("10. DATA QUALITY CHECKS & VALIDATION")
print("=" * 80)

# 10.1 Check duplicates
print("\n" + "-" * 50)
print("10.1 Duplicate Values Check:")
print("-" * 50)
print(f"Duplicate order IDs: {df_clean['order_id'].duplicated().sum()}")
print(f"Completely duplicate rows: {df_clean.duplicated().sum()}")

# 10.2 Numeric range check
print("\n" + "-" * 50)
print("10.2 Numeric Range Check:")
print("-" * 50)
print(f"Negative values: {(df_clean['quantity'] < 0).sum()}")
print(f"Abnormally large quantities: {(df_clean['quantity'] > 100).sum()}")
print(f"Zero prices: {(df_clean['unit_price'] == 0).sum()}")

# 10.3 Check outliers using describe
print("\n" + "-" * 50)
print("10.3 Statistical Summary for Outlier Detection:")
print("-" * 50)
print(df_clean[['quantity', 'unit_price', 'final_amount']].describe())

# 10.4 Correlation analysis
print("\n" + "-" * 50)
print("10.4 Correlation Analysis:")
print("-" * 50)
correlation_matrix = df_clean[['quantity', 'unit_price', 'discount_pct', 'final_amount']].corr()
print("Correlation matrix:")
print(correlation_matrix.round(2))

# 10.5 Data consistency check
print("\n" + "-" * 50)
print("10.5 Data Consistency Check:")
print("-" * 50)
df_clean['calculated_total'] = df_clean['quantity'] * df_clean['unit_price']
df_clean['amount_diff'] = abs(df_clean['total_price'] - df_clean['calculated_total'])

print(f"Records with inconsistent amounts: {(df_clean['amount_diff'] > 0.01).sum()}")

# ============================================================================
# 11. PERFORMANCE OPTIMIZATION
# ============================================================================
print("\n" + "=" * 80)
print("11. PERFORMANCE OPTIMIZATION")
print("=" * 80)

# 11.1 Memory optimization
print("\n" + "-" * 50)
print("11.1 Memory Optimization:")
print("-" * 50)


def reduce_mem_usage(df):
    """Reduce DataFrame memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Original memory usage: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        # Skip object, datetime, and category types
        if col_type == object or str(col_type).startswith('datetime') or str(col_type) == 'category':
            continue

        if str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif str(col_type)[:5] == 'float':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Optimized memory usage: {end_mem:.2f} MB')
    print(f'Reduced by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df


# Apply memory optimization
df_optimized = reduce_mem_usage(df_clean.copy())

# 11.2 Vectorization vs Loop
print("\n" + "-" * 50)
print("11.2 Vectorization vs Loop Performance:")
print("-" * 50)

# Using loop (slow)
start = time.time()
result_loop = []
for val in df_clean['final_amount']:
    result_loop.append(val * 1.1)
loop_time = time.time() - start

# Using vectorization (fast)
start = time.time()
result_vectorized = df_clean['final_amount'] * 1.1
vectorized_time = time.time() - start

print(f"Loop time: {loop_time:.4f} seconds")
print(f"Vectorization time: {vectorized_time:.4f} seconds")
print(f"Speed improvement: {loop_time/vectorized_time:.1f}x")

# 11.3 Chunked reading for large files
print("\n" + "-" * 50)
print("11.3 Chunked Reading Example:")
print("-" * 50)
print("Example code for reading large CSV files in chunks:")
print(
    """
chunk_size = 1000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = chunk[chunk['final_amount'] > 100]
    chunks.append(processed_chunk)
df_large = pd.concat(chunks, ignore_index=True)
"""
)

# ============================================================================
# 12. DATA EXPORT
# ============================================================================
print("\n" + "=" * 80)
print("12. DATA EXPORT")
print("=" * 80)

# 12.1 Export to CSV
print("\n" + "-" * 50)
print("12.1 Export to CSV:")
print("-" * 50)
df_clean.to_csv('cleaned_sales_data.csv', index=False, encoding='utf-8')
print("Exported to CSV")

# 12.2 Export to Excel
print("\n" + "-" * 50)
print("12.2 Export to Excel (Multiple Sheets):")
print("-" * 50)
try:
    with pd.ExcelWriter('sales_analysis.xlsx', engine='openpyxl') as writer:
        df_clean.to_excel(writer, sheet_name='Raw_Data', index=False)
        category_stats.to_excel(writer, sheet_name='Category_Stats')
        monthly_sales.to_excel(writer, sheet_name='Monthly_Sales')
    print("Exported to Excel with multiple sheets")
except ImportError:
    print("Skipped: openpyxl not installed (pip install openpyxl)")

# 12.3 Export to Parquet (more efficient)
print("\n" + "-" * 50)
print("12.3 Export to Parquet:")
print("-" * 50)
try:
    df_clean.to_parquet('sales_data.parquet', index=False, compression='snappy')
    print("Exported to Parquet")
except ImportError:
    print("Skipped: pyarrow not installed (pip install pyarrow)")

# 12.4 Export to JSON
print("\n" + "-" * 50)
print("12.4 Export to JSON:")
print("-" * 50)
df_clean.head(100).to_json('sample_sales.json', orient='records', indent=2)
print("Exported to JSON")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    """
This script covers the most popular Pandas operation patterns from Kaggle, including:

CORE SKILLS:
1. Data Loading & Exploration - info(), describe(), head(), tail(), sample()
2. Data Cleaning - fillna(), dropna(), drop_duplicates(), replace()
3. Feature Engineering - Creating new features, binning, encoding
4. Data Aggregation - groupby(), agg(), transform()
5. Data Pivoting - pivot_table(), crosstab()
6. Data Merging - merge(), concat(), join()
7. Time Series - resample(), rolling(), shift()
8. Advanced Indexing - loc, iloc, query(), isin()
9. Data Quality Checks - Duplicates, outliers, consistency
10. Performance Optimization - Memory reduction, vectorization

BEST PRACTICES:
- Use vectorization instead of loops
- Properly utilize data types to save memory
- Chain operations for better code readability
- Regularly check data quality
- Use appropriate file formats (Parquet > CSV)

These operation patterns are derived from analyzing the most popular Kaggle notebooks:
- Titanic Survival Prediction
- House Price Prediction
- Credit Card Fraud Detection
- Customer Segmentation Clustering
- Time Series Forecasting

Mastering these skills will help you process data more efficiently in data science 
and machine learning projects!
"""
)

print("\n" + "=" * 80)
print("SCRIPT EXECUTION COMPLETED")
print("=" * 80)
