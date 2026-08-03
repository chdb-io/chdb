Session Management
==================

chDB provides stateful session management for maintaining state across multiple queries, creating temporary tables, views, and executing complex workflows.

Overview
--------

The Session class allows you to:

- Maintain state between queries (tables, views, databases persist within the session)
- Use temporary or persistent storage
- Stream large query results efficiently
- Execute complex SQL workflows with multiple steps

Session API
-----------

For the complete Session API reference, see :doc:`api`.

Basic Usage
-----------

**Creating and Using Sessions**

.. code-block:: python

   from chdb import session as chs
   
   # Create a temporary session (auto-cleanup)
   sess = chs.Session()
   
   # Execute queries with persistent state
   sess.query("CREATE DATABASE IF NOT EXISTS db_xxx ENGINE = Atomic")
   sess.query("CREATE TABLE IF NOT EXISTS db_xxx.log_table_xxx (x String, y Int) ENGINE = Log;")
   sess.query("INSERT INTO db_xxx.log_table_xxx VALUES ('a', 1), ('b', 3), ('c', 2), ('d', 5);")
   sess.query("CREATE VIEW db_xxx.view_xxx AS SELECT * FROM db_xxx.log_table_xxx LIMIT 4;")
   
   print("Select from view:")
   print(sess.query("SELECT * FROM db_xxx.view_xxx", "Pretty"))
   
   # Session automatically cleaned up when object is destroyed

**Session with File-based Storage**

.. code-block:: python

   # Create persistent session with file storage
   sess = chs.Session("my_database.db")
   
   # Create persistent tables
   sess.query("""
       CREATE TABLE users (
           id UInt32,
           name String,
           created_date Date
       ) ENGINE = MergeTree() ORDER BY id
   """)
   
   # Insert data
   sess.query("INSERT INTO users VALUES (1, 'Alice', '2024-01-01'), (2, 'Bob', '2024-01-02')")
   
   # Query data
   result = sess.query("SELECT * FROM users ORDER BY id", "JSONEachRow")
   print(result)
   
   # Close session (data persists in file)
   sess.close()

Connection String Support
-------------------------

Sessions support flexible connection strings for configuration:

.. code-block:: python

   # In-memory database
   sess = chs.Session(":memory:")
   
   # File-based with relative path
   sess = chs.Session("test.db")
   
   # Absolute path
   sess = chs.Session("/path/to/database.db")
   
   # URI format with parameters
   sess = chs.Session("file:test.db?param1=value1&param2=value2")
   
   # Read-only mode
   sess = chs.Session("test.db?mode=ro")
   
   # With verbose logging
   sess = chs.Session("test.db?verbose&log-level=debug")

Streaming Queries
-----------------

For processing large datasets efficiently, use streaming queries that don't load all results into memory:

**Basic Streaming Example**

.. code-block:: python

   from chdb import session as chs
   
   sess = chs.Session()
   
   # Stream large result set
   rows_cnt = 0
   with sess.send_query("SELECT * FROM numbers(200000)", "CSV") as stream_result:
       for chunk in stream_result:
           rows_cnt += chunk.rows_read()
   
   print(f"Processed {rows_cnt} rows")  # 200000

**Manual Streaming Control**

.. code-block:: python

   # Manual iteration with fetch()
   rows_cnt = 0
   stream_result = sess.send_query("SELECT * FROM numbers(200000)", "CSV")
   
   while True:
       chunk = stream_result.fetch()
       if chunk is None:
           break
       rows_cnt += chunk.rows_read()
   
   print(f"Processed {rows_cnt} rows")  # 200000

**Early Termination**

.. code-block:: python

   # Early cancellation example
   rows_cnt = 0
   stream_result = sess.send_query("SELECT * FROM numbers(200000)", "CSV")
   
   while True:
       chunk = stream_result.fetch()
       if chunk is None:
           break
       
       # Process some data then terminate early
       if rows_cnt > 0:
           stream_result.close()  # Important: close to free resources
           break
       
       rows_cnt += chunk.rows_read()
   
   print(f"Early termination after {rows_cnt} rows")

**PyArrow Integration**

.. code-block:: python

   import pyarrow as pa
   
   # Stream results in Arrow format
   stream_result = sess.send_query("SELECT * FROM numbers(100000)", "Arrow")
   
   # Create RecordBatchReader with custom batch size
   batch_reader = stream_result.record_batch(rows_per_batch=10000)
   
   # Use with external libraries (example: Delta Lake)
   # from deltalake import write_deltalake
   # write_deltalake("./my_delta_table", data=batch_reader, mode="overwrite")
   
   # Process batches manually
   for batch in batch_reader:
       print(f"Batch shape: {batch.num_rows} rows, {batch.num_columns} columns")
   
   stream_result.close()

Context Manager Support
-----------------------

Sessions support context manager protocol for automatic cleanup:

.. code-block:: python

   # Automatic cleanup with context manager
   with chs.Session("temp_session.db") as sess:
       sess.query("CREATE TABLE test (id Int32, name String)")
       sess.query("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
       
       result = sess.query("SELECT * FROM test", "Pretty")
       print(result)
       
   # Session automatically closed and resources cleaned up

Advanced Usage
--------------

**Complex Analytical Workflows**

.. code-block:: python

   sess = chs.Session("analytics.db")
   
   # Create analytical tables
   sess.query("""
       CREATE TABLE sales (
           date Date,
           product String,
           revenue Decimal(10,2),
           quantity UInt32
       ) ENGINE = MergeTree() ORDER BY date
   """)
   
   # Load data (example with CSV file)
   sess.query("""
       INSERT INTO sales 
       SELECT * FROM file('sales_data.csv', 'CSV', 
                         'date Date, product String, revenue Decimal(10,2), quantity UInt32')
   """)
   
   # Create analytical views
   sess.query("""
       CREATE VIEW monthly_sales AS
       SELECT 
           toYYYYMM(date) as month,
           product,
           sum(revenue) as total_revenue,
           sum(quantity) as total_quantity
       FROM sales
       GROUP BY month, product
   """)
   
   # Run analysis
   result = sess.query("""
       SELECT 
           month,
           product,
           total_revenue,
           total_revenue / lag(total_revenue) OVER (PARTITION BY product ORDER BY month) - 1 as growth_rate
       FROM monthly_sales
       ORDER BY month DESC, total_revenue DESC
   """, "JSONEachRow")
   
   print("Monthly growth analysis:")
   print(result)

**Working with Multiple Databases**

.. code-block:: python

   sess = chs.Session("multi_db.chdb")
   
   # Create multiple databases
   sess.query("CREATE DATABASE sales ENGINE = Atomic")
   sess.query("CREATE DATABASE analytics ENGINE = Atomic")
   
   # Create tables in different databases
   sess.query("""
       CREATE TABLE sales.transactions (
           id UInt32,
           customer_id UInt32,
           amount Decimal(10,2),
           timestamp DateTime
       ) ENGINE = MergeTree() ORDER BY timestamp
   """)
   
   sess.query("""
       CREATE TABLE analytics.daily_summary AS
       SELECT 
           toDate(timestamp) as date,
           count(*) as transaction_count,
           sum(amount) as total_amount
       FROM sales.transactions
       GROUP BY date
   """)

Error Handling
--------------

**Robust Session Management**

.. code-block:: python

   import chdb
   
   def safe_session_query(session, sql, fmt="CSV"):
       """Execute session query with proper error handling"""
       try:
           result = session.query(sql, fmt)
           return result, None
       except Exception as e:
           return None, str(e)
   
   # Example usage
   try:
       sess = chs.Session("test.db")
       
       # Test table creation
       result, error = safe_session_query(sess, 
           "CREATE TABLE test (id Int32, name String)")
       
       if error:
           print(f"Table creation failed: {error}")
       else:
           print("Table created successfully")
       
       # Test data insertion
       result, error = safe_session_query(sess,
           "INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
       
       if error:
           print(f"Data insertion failed: {error}")
       else:
           # Query data
           result, error = safe_session_query(sess, 
               "SELECT * FROM test", "Pretty")
           if not error:
               print("Query results:")
               print(result)
       
   finally:
       if 'sess' in locals():
           sess.close()

Best Practices
--------------

1. **Resource Management**: Always close sessions when done, or use context managers
2. **Memory Usage**: Use streaming queries for large datasets
3. **Persistence**: Choose appropriate storage (memory vs file) based on needs
4. **Error Handling**: Wrap session operations in try-catch blocks
5. **Connection Strings**: Use connection string parameters for configuration

.. note::
   - Only one session can be active at a time per process
   - Creating a new session will automatically close any existing session
   - Temporary sessions are automatically cleaned up when the session object is destroyed
   - File-based sessions persist data across Python interpreter restarts

.. warning::
   - Always call :meth:`StreamingResult.close()` when terminating streaming queries early
   - Large result sets should use streaming queries to avoid memory issues
   - Session state is not shared between different Python processes

See Also
--------

- :doc:`api` - Complete API reference including Session class and DB-API interface
- :doc:`examples` - More comprehensive usage examples