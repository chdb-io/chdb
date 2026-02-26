# Test Dataset for Comprehensive JOIN Tests

This directory contains test datasets used for validating DataStore JOIN operations across multiple data sources.

## Dataset Files

### users.csv
User information with the following columns:
- `user_id`: Unique identifier
- `name`: User's full name
- `email`: Email address
- `age`: User's age
- `country`: Country of residence
- `registration_date`: Registration date

**Sample Data:** 10 users from various countries (USA, UK, Canada, Australia)

### orders.csv
Order records with the following columns:
- `order_id`: Unique order identifier
- `user_id`: Reference to user
- `product_id`: Reference to product
- `quantity`: Number of items ordered
- `order_date`: Date of order
- `amount`: Total order amount

**Sample Data:** 15 orders across different users and products

### products.csv
Product catalog with the following columns:
- `product_id`: Unique product identifier
- `product_name`: Product name
- `category_id`: Reference to category
- `price`: Product price
- `stock`: Available inventory
- `supplier`: Supplier name

**Sample Data:** 10 products across different categories

### categories.csv
Product categories with the following columns:
- `category_id`: Unique category identifier
- `category_name`: Category name
- `description`: Category description

**Sample Data:** 5 categories (Electronics, Cables, Furniture, Software, Books)

## Relationships

```
users (user_id) ─┐
                 ├──→ orders (user_id, product_id) ──→ products (product_id) ──→ categories (category_id)
                 │
                 └──→ Multiple orders per user
```

## Test Coverage

The comprehensive JOIN tests (`test_comprehensive_joins.py`) validate:

1. **Basic 2-Way JOIN**: Users ⟗ Orders
2. **3-Way JOIN**: Users ⟗ Orders ⟗ Products
3. **4-Way JOIN**: Users ⟗ Orders ⟗ Products ⟗ Categories
4. **LEFT JOIN with Filters**: Selective data retrieval
5. **JOIN with Aggregation**: GROUP BY, COUNT, SUM, AVG, MAX, MIN
6. **JOIN with Complex Conditions**: Multiple WHERE clauses
7. **JOIN with ORDER BY and LIMIT**: Sorted and paginated results
8. **Table Function Integration**: File, Numbers generators
9. **Cross-DataSource Operations**: Mixed source types

## Usage

These datasets are automatically loaded by the test suite. To use them manually:

```python
from datastore import DataStore

# Load data with column names
users = DataStore("file", path="tests/dataset/users.csv", format="CSVWithNames")
orders = DataStore("file", path="tests/dataset/orders.csv", format="CSVWithNames")

# Perform JOIN
result = (
    users
    .join(orders, left_on="user_id", right_on="user_id")
    .select("name", "email", "order_id", "amount")
    .connect()
    .execute()
)

print(result.to_dict())
```

## Data Format

All CSV files use:
- **Format**: `CSVWithNames` (header row with column names)
- **Delimiter**: Comma (`,`)
- **Encoding**: UTF-8
- **Quote Character**: Double quotes (`"`) for strings containing special characters

## Extending the Datasets

To add more test data:

1. Maintain referential integrity (user_id, product_id, category_id relationships)
2. Use consistent data types across related columns
3. Include edge cases (NULL values, empty strings, boundary values)
4. Update this README with new column descriptions

