"""
chdb API Example - Demonstrating the Connection and Cursor interfaces
This example shows common usage patterns with explanatory comments
"""

import chdb
from datetime import datetime, date

# 1. Creating a connection
# The connect function supports several connection string formats:
# - ":memory:" (in-memory database, default)
# - "test.db" or "file:test.db" (relative path)
# - "/path/to/test.db" or "file:/path/to/test.db" (absolute path)
# - Connection strings can include query parameters: "file:test.db?param1=value1"

# Create an in-memory connection
conn = chdb.connect(":memory:")
# Alternative: conn = chdb.connect("test.db")  # file-based connection

# 2. Working with cursors
# Cursors allow executing SQL queries and fetching results
cursor = conn.cursor()

# 3. Execute a simple query with multiple data types
cursor.execute("""
    SELECT 
        42 as int_val,
        3.14 as float_val,
        'hello' as str_val,
        true as bool_val,
        NULL as null_val,
        toDateTime('2024-03-20 15:30:00') as datetime_val,
        toDate('2024-03-20') as date_val
""")

# 4. Fetch methods
# 4.1. fetchone() - returns a single row as a tuple, or None if no more rows
row = cursor.fetchone()
print("Single row:", row)
# Output example: (42, 3.14, 'hello', True, None, datetime(2024,3,20,15,30), date(2024,3,20))

# 4.2. Execute a query with multiple rows
cursor.execute("""
    SELECT number, toString(number) as str_val 
    FROM system.numbers 
    LIMIT 5
""")

# fetchmany(size) - returns up to 'size' rows as a tuple of tuples
batch = cursor.fetchmany(2)
print("First 2 rows:", batch)
# Output example: ((0, '0'), (1, '1'))

# fetchall() - returns all remaining rows as a tuple of tuples
remaining = cursor.fetchall()
print("Remaining rows:", remaining)
# Output example: ((2, '2'), (3, '3'), (4, '4'))

# 5. Cursor as an iterator
cursor.execute("SELECT number FROM system.numbers LIMIT 3")
print("Iterating through cursor:")
for row in cursor:
    print(f"  Row: {row}")
# Output:
#   Row: (0,)
#   Row: (1,)
#   Row: (2,)

# 6. Get metadata about the result set
cursor.execute("""
    SELECT 
        42 as int_col,
        'hello' as str_col, 
        now() as time_col
""")

# 6.1. column_names() - get a list of column names
col_names = cursor.column_names()
print("Column names:", col_names)
# Output example: ['int_col', 'str_col', 'time_col']

# 6.2. column_types() - get a list of column types
col_types = cursor.column_types()
print("Column types:", col_types)
# Output example: ['UInt8', 'String', 'DateTime']

# 6.3. description - DB-API 2.0 compatible column descriptions
# Format: list of 7-item tuples (name, type_code, display_size, internal_size, precision, scale, null_ok)
# Only name and type_code are populated in this implementation
desc = cursor.description
print("Description:", desc)
# Output example: [('int_col', 'UInt8', None, None, None, None, None), ...]

# 7. Using the Connection.query() method for direct queries
# This method returns results directly without needing to create a cursor

# 7.1. Default format is CSV
csv_result = conn.query("SELECT 1 as val, 'test' as name")
print("CSV query result:", csv_result)
# Returns data in CSV format as a string

# 7.2. Using Arrow format to get a PyArrow table
# Note: requires pyarrow to be installed
arrow_result = conn.query("SELECT 1 as val, 'test' as name", format="Arrow") 
print("Arrow query result type:", type(arrow_result))
# Returns a PyArrow Table object

# 7.3. Using DataFrame format to get a pandas DataFrame
# Note: requires both pyarrow and pandas to be installed
df_result = conn.query("SELECT 1 as val, 'test' as name", format="dataframe")
print("DataFrame query result:\n", df_result)
# Returns a pandas DataFrame

# 8. Error handling
try:
    cursor.execute("SELECT non_existent_column")
except Exception as e:
    print("SQL Error:", e)

# 9. Always close resources when done
cursor.close()
conn.close()
