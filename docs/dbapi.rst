DB-API 2.0 Interface
====================

chDB provides a Python DB-API 2.0 compatible interface for database connectivity, allowing you to use chDB with tools and frameworks that expect standard database interfaces.

Overview
--------

The chDB DB-API 2.0 interface includes:

- **Connections**: Database connection management with connection strings
- **Cursors**: Query execution and result retrieval 
- **Type System**: DB-API 2.0 compliant type constants and converters
- **Error Handling**: Standard database exception hierarchy
- **Thread Safety**: Level 1 thread safety (threads may share modules but not connections)

API Reference
-------------

**Core Functions**

.. autofunction:: chdb.dbapi.connect

.. autofunction:: chdb.dbapi.get_client_info
   :no-index:

**Type Constructors**

.. autofunction:: chdb.dbapi.Binary
   :no-index:

**Connection Class**

.. autoclass:: chdb.dbapi.connections.Connection
   :members:
   :show-inheritance:

**Cursor Class**  

.. autoclass:: chdb.dbapi.cursors.Cursor
   :members:
   :show-inheritance:

**Error Classes**

.. automodule:: chdb.dbapi.err
   :members:
   :show-inheritance:

Basic Usage
-----------

**Simple Query Example**

.. code-block:: python

   import chdb.dbapi as dbapi
   
   print("chdb driver version: {0}".format(dbapi.get_client_info()))
   
   # Create connection and cursor
   conn = dbapi.connect()
   cur = conn.cursor()
   
   # Execute query
   cur.execute('SELECT version()')
   print("description:", cur.description)
   print("data:", cur.fetchone())
   
   # Clean up
   cur.close()
   conn.close()

**Working with Data**

.. code-block:: python

   import chdb.dbapi as dbapi
   
   conn = dbapi.connect()
   cur = conn.cursor()
   
   # Create table
   cur.execute("""
       CREATE TABLE employees (
           id UInt32,
           name String,
           department String,
           salary Decimal(10,2)
       ) ENGINE = Memory
   """)
   
   # Insert data
   cur.execute("""
       INSERT INTO employees VALUES 
       (1, 'Alice', 'Engineering', 75000.00),
       (2, 'Bob', 'Marketing', 65000.00),
       (3, 'Charlie', 'Engineering', 80000.00)
   """)
   
   # Query data
   cur.execute("SELECT * FROM employees WHERE department = 'Engineering'")
   
   # Fetch results
   print("Column names:", [desc[0] for desc in cur.description])
   for row in cur.fetchall():
       print(row)
   
   conn.close()

Connection Management
---------------------

**Connection Strings**

.. code-block:: python

   import chdb.dbapi as dbapi
   
   # In-memory database (default)
   conn1 = dbapi.connect()
   
   # Persistent database file
   conn2 = dbapi.connect("./my_database.chdb")
   
   # Connection with parameters
   conn3 = dbapi.connect("./my_database.chdb?log-level=debug&verbose")
   
   # Read-only connection
   conn4 = dbapi.connect("./my_database.chdb?mode=ro")

**Connection Context Manager**

.. code-block:: python

   # Automatic connection cleanup
   with dbapi.connect("test.chdb") as conn:
       cur = conn.cursor()
       cur.execute("SELECT count() FROM numbers(1000)")
       result = cur.fetchone()
       print(f"Count: {result[0]}")
       cur.close()
   # Connection automatically closed

Cursor Operations
-----------------

**Fetching Results**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   cur.execute("SELECT number, number * 2 as doubled FROM numbers(5)")
   
   # Fetch one row at a time
   print("First row:", cur.fetchone())    # (0, 0)
   
   # Fetch multiple rows
   print("Next 2 rows:", cur.fetchmany(2))  # ((1, 2), (2, 4))
   
   # Fetch all remaining rows
   print("Remaining rows:", cur.fetchall())  # ((3, 6), (4, 8))
   
   conn.close()

**Cursor as Iterator**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   cur.execute("SELECT number, toString(number) FROM numbers(3)")
   
   # Iterate over results
   for row in cur:
       print(f"Number: {row[0]}, String: {row[1]}")
   
   conn.close()

**Column Metadata**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   cur.execute("SELECT 1 as id, 'Alice' as name, 25.5 as score")
   
   # Get column information
   print("Column descriptions:")
   for desc in cur.description:
       name, type_code, display_size, internal_size, precision, scale, null_ok = desc
       print(f"  {name}: type={type_code}, nullable={null_ok}")
   
   # Get column names and types
   print("Column names:", cur.column_names())
   print("Column types:", cur.column_types())
   
   conn.close()

Advanced Usage
--------------

**Parameterized Queries**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   # Note: chDB uses format-style parameters
   query = "SELECT number FROM numbers(%s) WHERE number > %s"
   cur.execute(query, (10, 5))
   
   results = cur.fetchall()
   print(f"Found {len(results)} rows")
   
   conn.close()

**Working with Large Datasets**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   # Execute query on large dataset
   cur.execute("SELECT number, number * number FROM numbers(100000)")
   
   # Process in batches to avoid memory issues
   batch_size = 1000
   processed = 0
   
   while True:
       rows = cur.fetchmany(batch_size)
       if not rows:
           break
       
       # Process batch
       for row in rows:
           # Your processing logic here
           processed += 1
       
       print(f"Processed {processed} rows...")
   
   print(f"Total processed: {processed}")
   conn.close()

**File Data Processing**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   # Query CSV file directly
   cur.execute("""
       SELECT 
           column1,
           column2,
           count(*) as count
       FROM file('data.csv', 'CSV')
       GROUP BY column1, column2
       ORDER BY count DESC
   """)
   
   print("Top combinations:")
   for row in cur.fetchmany(10):
       print(f"  {row[0]}, {row[1]}: {row[2]}")
   
   conn.close()

Type System
-----------

**DB-API 2.0 Type Constants**

.. code-block:: python

   import chdb.dbapi as dbapi
   
   conn = dbapi.connect()
   cur = conn.cursor()
   
   cur.execute("SELECT 'hello' as text, 42 as number, now() as timestamp")
   
   # Check column types using DB-API 2.0 constants
   for desc in cur.description:
       col_name, type_code = desc[0], desc[1]
       
       if type_code == dbapi.STRING:
           print(f"{col_name} is a string type")
       elif type_code == dbapi.NUMBER:
           print(f"{col_name} is a number type") 
       elif type_code == dbapi.TIMESTAMP:
           print(f"{col_name} is a timestamp type")
   
   conn.close()

**Binary Data Handling**

.. code-block:: python

   conn = dbapi.connect()
   cur = conn.cursor()
   
   # Handle binary data
   binary_data = dbapi.Binary(b"Hello, World!")
   
   # In a real scenario, you might store binary data in a blob field
   print(f"Binary data: {binary_data}")
   print(f"Type: {type(binary_data)}")
   
   conn.close()

Error Handling
--------------

**Database Exceptions**

.. code-block:: python

   import chdb.dbapi as dbapi
   
   conn = dbapi.connect()
   cur = conn.cursor()
   
   try:
       # This will cause an error
       cur.execute("SELECT * FROM non_existent_table")
   except dbapi.Error as e:
       print(f"Database error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")
   finally:
       cur.close()
       conn.close()

**Robust Query Execution**

.. code-block:: python

   def safe_execute(cursor, query, params=None):
       """Execute query with comprehensive error handling"""
       try:
           if params:
               cursor.execute(query, params)
           else:
               cursor.execute(query)
           return True, None
       except dbapi.Error as e:
           return False, f"Database error: {e}"
       except Exception as e:
           return False, f"Unexpected error: {e}"
   
   # Example usage
   conn = dbapi.connect()
   cur = conn.cursor()
   
   success, error = safe_execute(cur, "SELECT count() FROM numbers(100)")
   if success:
       result = cur.fetchone()
       print(f"Count: {result[0]}")
   else:
       print(f"Query failed: {error}")
   
   conn.close()

Integration Examples
--------------------

**With Pandas**

.. code-block:: python

   import pandas as pd
   import chdb.dbapi as dbapi
   
   conn = dbapi.connect()
   
   # Execute query and convert to DataFrame
   query = """
       SELECT 
           number as id,
           number * 2 as value,
           number % 3 as category
       FROM numbers(10)
   """
   
   # Note: Direct pandas.read_sql might not work, use manual conversion
   cur = conn.cursor()
   cur.execute(query)
   
   # Get column names
   columns = [desc[0] for desc in cur.description]
   
   # Fetch all data
   data = cur.fetchall()
   
   # Create DataFrame
   df = pd.DataFrame(data, columns=columns)
   print(df)
   
   cur.close()
   conn.close()

**Custom Data Processing**

.. code-block:: python

   class DataProcessor:
       def __init__(self, connection_string=None):
           self.conn = dbapi.connect(connection_string)
           self.cur = self.conn.cursor()
       
       def __enter__(self):
           return self
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           self.close()
       
       def execute_analysis(self, query):
           """Execute analytical query and return structured results"""
           self.cur.execute(query)
           
           # Get metadata
           columns = [desc[0] for desc in self.cur.description]
           
           # Get data
           rows = self.cur.fetchall()
           
           return {
               'columns': columns,
               'data': rows,
               'row_count': len(rows)
           }
       
       def close(self):
           if hasattr(self, 'cur') and self.cur:
               self.cur.close()
           if hasattr(self, 'conn') and self.conn:
               self.conn.close()
   
   # Usage
   with DataProcessor() as processor:
       results = processor.execute_analysis("""
           SELECT 
               toYYYYMM(today() - number) as month,
               number as days_ago,
               number * 100 as metric
           FROM numbers(12)
       """)
       
       print(f"Analysis complete: {results['row_count']} rows")
       for row in results['data'][:3]:
           print(f"  Month: {row[0]}, Days ago: {row[1]}, Metric: {row[2]}")

Best Practices
--------------

1. **Connection Management**: Always close connections and cursors when done
2. **Context Managers**: Use ``with`` statements for automatic cleanup
3. **Batch Processing**: Use ``fetchmany()`` for large result sets
4. **Error Handling**: Wrap database operations in try-except blocks
5. **Parameter Binding**: Use parameterized queries when possible
6. **Memory Management**: Avoid ``fetchall()`` for very large datasets

.. note::
   - chDB's DB-API 2.0 interface is compatible with most Python database tools
   - The interface provides Level 1 thread safety (threads may share modules but not connections)
   - Connection strings support the same parameters as chDB sessions
   - All standard DB-API 2.0 exceptions are supported

.. warning::
   - Always close cursors and connections to avoid resource leaks
   - Large result sets should be processed in batches
   - Parameter binding syntax follows format style: ``%s``

Module Constants
----------------

**API Level and Threading**

.. autodata:: chdb.dbapi.apilevel
.. autodata:: chdb.dbapi.threadsafety  
.. autodata:: chdb.dbapi.paramstyle

**Type Constants**

.. autodata:: chdb.dbapi.STRING
.. autodata:: chdb.dbapi.BINARY
.. autodata:: chdb.dbapi.NUMBER
.. autodata:: chdb.dbapi.DATE
.. autodata:: chdb.dbapi.TIME
.. autodata:: chdb.dbapi.TIMESTAMP
.. autodata:: chdb.dbapi.DATETIME
.. autodata:: chdb.dbapi.ROWID

See Also
--------

- :doc:`session` - For stateful database operations
- :doc:`examples` - More usage examples and patterns
- :doc:`api` - Complete API reference
- `Python DB-API 2.0 Specification <https://peps.python.org/pep-0249/>`_ - Official DB-API standard