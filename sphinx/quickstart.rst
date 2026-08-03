Quick Start
===========

This guide will get you up and running with chDB in minutes.

Your First Query
----------------

Let's start with the simplest example:

.. code-block:: python

   import chdb
   
   # Your first query
   result = chdb.query("SELECT 1 as id, 'Hello World' as message", "CSV")
   print(result)

Output:

.. code-block:: text

   1,Hello World

Connection-Based API (Recommended)
----------------------------------

For better performance and more features, use the connection-based API:

.. code-block:: python

   import chdb
   
   # Create a connection
   conn = chdb.connect(":memory:")
   cur = conn.cursor()
   
   # Execute queries
   cur.execute("SELECT number, toString(number) as str FROM system.numbers LIMIT 3")
   
   # Fetch results
   for row in cur:
       print(row)
   
   # Clean up
   conn.close()

Output Formats
--------------

chDB supports multiple output formats for different use cases:

**CSV (Default)**

.. code-block:: python

   result = chdb.query("SELECT 1, 'test'", "CSV")
   print(result)  # CSV string

**DataFrame (Pandas)**

.. code-block:: python

   import chdb
   
   df = chdb.query("SELECT number, number*2 as doubled FROM numbers(5)", "DataFrame")
   print(type(df))  # <class 'pandas.core.frame.DataFrame'>
   print(df.head())

**Arrow Table**

.. code-block:: python

   table = chdb.query("SELECT number FROM numbers(1000)", "ArrowTable")
   print(type(table))  # <class 'pyarrow.lib.Table'>
   print(f"Rows: {len(table)}")

**Pretty Format**

.. code-block:: python

   result = chdb.query("""
       SELECT 
           'Alice' as name, 25 as age 
       UNION ALL 
       SELECT 'Bob', 30
   """, "Pretty")
   print(result)

Working with Files
------------------

chDB can query 70+ file formats directly:

**CSV Files**

.. code-block:: python

   # Query a local CSV file
   result = chdb.query("""
       SELECT count(*), avg(column_name)
       FROM file('data.csv', 'CSV')
   """)

**JSON Files**

.. code-block:: python

   # Query JSON data
   result = chdb.query("""
       SELECT * FROM file('data.json', 'JSONEachRow')
       WHERE field > 100
       LIMIT 10
   """)

**Parquet Files**

.. code-block:: python

   # Efficient querying of Parquet files
   result = chdb.query("""
       SELECT department, sum(salary) as total_salary
       FROM file('employees.parquet', 'Parquet')
       GROUP BY department
       ORDER BY total_salary DESC
   """)

DataFrame Integration
---------------------

Query pandas DataFrames directly:

.. code-block:: python

   import pandas as pd
   import chdb
   
   # Create sample DataFrame
   df = pd.DataFrame({
       'product': ['A', 'B', 'C', 'A', 'B'],
       'sales': [100, 200, 150, 300, 250],
       'region': ['North', 'South', 'North', 'South', 'North']
   })
   
   # Query the DataFrame using chDB
   result = chdb.query("""
       SELECT 
           product,
           region,
           sum(sales) as total_sales,
           avg(sales) as avg_sales
       FROM Python(df)
       GROUP BY product, region
       ORDER BY total_sales DESC
   """, "DataFrame")
   
   print(result)

Memory vs Persistent Storage
----------------------------

**In-Memory (Default)**

Perfect for data analysis and temporary operations:

.. code-block:: python

   # All data stays in memory
   result = chdb.query("""
       SELECT number, number^2 as squared
       FROM numbers(1000000)
       WHERE number % 1000 = 0
   """)

**Persistent Storage**

For data that needs to persist between sessions:

.. code-block:: python

   # Create a persistent database
   conn = chdb.connect("my_database.chdb")
   cur = conn.cursor()
   
   # Create and populate table
   cur.execute("""
       CREATE TABLE IF NOT EXISTS users (
           id UInt32,
           name String,
           email String
       ) ENGINE = MergeTree() ORDER BY id
   """)
   
   cur.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')")
   cur.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@example.com')")
   
   # Query the persistent data
   cur.execute("SELECT * FROM users ORDER BY id")
   for row in cur:
       print(row)
   
   conn.close()

Performance Tips
----------------

**Use Connection Objects for Multiple Queries**

.. code-block:: python

   # More efficient for multiple queries
   conn = chdb.connect()
   cur = conn.cursor()
   
   for i in range(100):
       cur.execute(f"SELECT {i} as iteration")
       result = cur.fetchone()
   
   conn.close()

Error Handling
--------------

Handle errors gracefully:

.. code-block:: python

   import chdb
   
   try:
       result = chdb.query("SELECT invalid_column FROM non_existent_table")
   except chdb.ChdbError as e:
       print(f"Query error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Next Steps
----------

Now that you're familiar with the basics:

- Explore the :doc:`examples` for more advanced use cases
- Check out :doc:`udf` for custom functions
- Learn about :doc:`session` for stateful operations
- Review the :doc:`api` reference for complete functionality
