Troubleshooting
===============

Installation Issues
-------------------

Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~~

chDB currently supports Python 3.8+ on **macOS and Linux** (x86_64 and ARM64). Windows support is not available yet.

**Supported Platforms:**

- macOS (x86_64 and ARM64)
- Linux (x86_64 and ARM64)
- Python 3.8, 3.9, 3.10, 3.11, 3.12+

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: bash

   ImportError: No module named '_chdb'
   ImportError: No module named 'chdb'

**Solution**: Ensure chDB is properly installed:

.. code-block:: bash

   pip uninstall chdb
   pip install chdb

**Alternative installation methods:**

.. code-block:: bash

   # Force reinstall
   pip install --force-reinstall chdb
   
   # Install specific version
   pip install chdb==3.6.0

**Check installation:**

.. code-block:: python

   import chdb
   print(f"chDB version: {chdb.__version__}")
   print(f"Engine version: {chdb.engine_version}")

Python Version Issues
~~~~~~~~~~~~~~~~~~~~~

If you get Python version compatibility errors:

.. code-block:: text

   ERROR: chdb requires Python >=3.8

**Solution**: Upgrade your Python version:

.. code-block:: bash

   # Check current Python version
   python --version
   
   # Use Python 3.8+ explicitly
   python3.8 -m pip install chdb

Query Execution Issues
----------------------

Memory Issues
~~~~~~~~~~~~~

If you encounter memory-related errors:

.. code-block:: text

   Memory limit exceeded
   Out of memory while executing query

**Solutions**:

1. **Use Streaming Queries for Large Datasets**

.. code-block:: python

   from chdb import session as chs
   
   sess = chs.Session()
   
   # Process large datasets with streaming
   rows_cnt = 0
   with sess.send_query("SELECT * FROM numbers(1000000)", "CSV") as stream_result:
       for chunk in stream_result:
           # Process chunk by chunk to avoid memory issues
           rows_cnt += chunk.rows_read()
   
   print(f"Processed {rows_cnt} rows")

2. **Use File-based Sessions for Persistence**

.. code-block:: python

   # Use persistent storage to reduce memory usage
   sess = chs.Session("large_dataset.chdb")  # File-based storage
   
   # Instead of in-memory
   # sess = chs.Session()  # In-memory storage

3. **Process Data in Smaller Batches**

.. code-block:: python

   import chdb
   
   # Good: Process in batches
   for i in range(0, 1000000, 10000):
       result = chdb.query(f"SELECT * FROM numbers({i}, 10000)")
       # Process batch
   
   # Avoid: Loading entire dataset at once
   # result = chdb.query("SELECT * FROM numbers(1000000)")

4. **Use Column Selection**

.. code-block:: python

   # Good: Select only needed columns
   result = chdb.query("SELECT id, name FROM large_table WHERE id > 100")
   
   # Avoid: Select all columns
   # result = chdb.query("SELECT * FROM large_table WHERE id > 100")

File Access Issues
~~~~~~~~~~~~~~~~~~

If you encounter file access errors:

.. code-block:: text

   Permission denied: Cannot read file
   File not found: /path/to/file.csv
   Cannot determine file format

**Solutions**:

1. **Check File Permissions and Path**

.. code-block:: python

   import os
   import chdb
   
   # Check if file exists
   file_path = "data.csv"
   if not os.path.exists(file_path):
       print(f"File does not exist: {file_path}")
   
   # Use absolute path
   abs_path = os.path.abspath(file_path)
   result = chdb.query(f"SELECT * FROM file('{abs_path}', 'CSV')")

2. **Supported File Formats**

.. code-block:: python

   # chDB supports 60+ formats including:
   result = chdb.query("SELECT * FROM file('data.parquet', 'Parquet')")
   result = chdb.query("SELECT * FROM file('data.csv', 'CSV')")
   result = chdb.query("SELECT * FROM file('data.json', 'JSONEachRow')")
   result = chdb.query("SELECT * FROM file('data.orc', 'ORC')")

3. **File Format Detection Issues**

.. code-block:: python

   # Explicitly specify format and schema if auto-detection fails
   result = chdb.query("""
       SELECT * FROM file('data.csv', 'CSV', 
                         'id UInt32, name String, age UInt8')
   """)

4. **Working with Remote Files**

.. code-block:: python

   # HTTP/HTTPS files
   result = chdb.query("""
       SELECT * FROM url('https://example.com/data.csv', 'CSV')
   """)

Connection and Session Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Session Already Exists Error**

.. code-block:: text

   Session already exists

**Solution**: Only one session can be active at a time per process:

.. code-block:: python

   from chdb import session as chs
   
   # Close existing session before creating new one
   if 'sess' in locals():
       sess.close()
   
   sess = chs.Session()

**DB-API Connection Issues**

.. code-block:: python

   import chdb.dbapi as dbapi
   
   # Always close connections properly
   conn = dbapi.connect()
   try:
       cur = conn.cursor()
       cur.execute("SELECT 1")
       result = cur.fetchone()
   finally:
       cur.close()
       conn.close()
   
   # Or use context manager for automatic cleanup
   with dbapi.connect() as conn:
       cur = conn.cursor()
       cur.execute("SELECT 1")
       result = cur.fetchone()
       cur.close()

Performance Issues
~~~~~~~~~~~~~~~~~~

If queries are running slowly:

**Solutions**:

1. **Use Efficient Query Patterns**

.. code-block:: python

   # Good: Select specific columns
   result = chdb.query("SELECT id, name FROM users WHERE id > 100")
   
   # Good: Use LIMIT for exploration
   result = chdb.query("SELECT * FROM large_table LIMIT 100")
   
   # Avoid: Select all columns from large tables
   # result = chdb.query("SELECT * FROM users WHERE id > 100")

2. **Optimize Data Formats**

.. code-block:: python

   # Parquet is usually faster than CSV for analytical queries
   result = chdb.query("SELECT * FROM file('data.parquet', 'Parquet')")
   
   # For repeated queries, consider using session with persistent storage
   from chdb import session as chs
   sess = chs.Session("analytics.chdb")
   
   # Load data once
   sess.query("CREATE TABLE users AS SELECT * FROM file('users.parquet', 'Parquet')")
   
   # Query multiple times efficiently
   result1 = sess.query("SELECT COUNT(*) FROM users WHERE age > 25")
   result2 = sess.query("SELECT AVG(age) FROM users GROUP BY department")

3. **Use Column-based Operations**

.. code-block:: python

   # Good: Use aggregations and grouping
   result = chdb.query("""
       SELECT department, COUNT(*), AVG(salary)
       FROM employees 
       GROUP BY department
       ORDER BY AVG(salary) DESC
   """)
   
   # Good: Use window functions for analytics
   result = chdb.query("""
       SELECT name, salary, 
              rank() OVER (PARTITION BY department ORDER BY salary DESC) as rank
       FROM employees
   """)

DataFrame Integration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pandas DataFrame Problems**

.. code-block:: python

   import chdb.dataframe as cdf
   import pandas as pd
   
   # Ensure DataFrames have proper column types
   df = pd.DataFrame({
       'id': [1, 2, 3],
       'name': ['Alice', 'Bob', 'Charlie'],
       'score': [85.5, 92.0, 88.5]
   })
   
   # Use chDB dataframe query
   result = cdf.query("SELECT name, score FROM __tbl__ WHERE score > 85", tbl=df)
   
   # Or use Python table engine
   result = chdb.query("SELECT name, AVG(score) FROM Python(df) GROUP BY name")

**Arrow Integration Issues**

.. code-block:: python

   import pyarrow as pa
   import chdb
   
   # Create Arrow table with proper types
   arrow_table = pa.table({
       'id': pa.array([1, 2, 3], type=pa.int64()),
       'name': pa.array(['Alice', 'Bob', 'Charlie'], type=pa.string()),
       'score': pa.array([85.5, 92.0, 88.5], type=pa.float64())
   })
   
   # Query Arrow table
   result = chdb.query("SELECT * FROM Python(arrow_table) WHERE score > 85")

UDF (User Defined Functions) Issues
------------------------------------

**UDF Import/Registration Problems**

.. code-block:: python

   from chdb.udf import chdb_udf
   from chdb import query
   
   # Ensure UDF is stateless and uses proper imports
   @chdb_udf()
   def clean_text(text):
       # Import modules inside the function
       import re
       return re.sub(r'[^\w\s]', '', text.lower())
   
   # Test UDF
   result = query("SELECT clean_text('Hello, World!') as cleaned")
   print(result)

**UDF Type Issues**

.. code-block:: python

   # Specify return type if not String
   @chdb_udf(return_type="UInt64")
   def calculate_sum(a, b):
       return int(a) + int(b)
   
   # All input arguments are strings, convert as needed
   @chdb_udf()
   def process_json(json_str):
       import json
       try:
           data = json.loads(json_str)
           return str(data.get('value', 0))
       except:
           return '0'

Streaming Query Issues
----------------------

**Resource Not Released**

.. code-block:: python

   from chdb import session as chs
   
   sess = chs.Session()
   
   # Always close streaming results if not fully consumed
   stream_result = sess.send_query("SELECT * FROM numbers(1000000)", "CSV")
   try:
       for i, chunk in enumerate(stream_result):
           if i >= 5:  # Early termination
               break
           # Process chunk
   finally:
       stream_result.close()  # Important: release resources
   
   # Or use with statement for automatic cleanup
   with sess.send_query("SELECT * FROM numbers(1000000)", "CSV") as stream_result:
       for chunk in stream_result:
           # Process chunk
           pass
   # Automatically closed

**Arrow RecordBatch Issues**

.. code-block:: python

   import pyarrow as pa
   
   # Ensure proper batch size for memory management
   stream_result = sess.send_query("SELECT * FROM numbers(100000)", "Arrow")
   
   # Use appropriate batch size
   batch_reader = stream_result.record_batch(rows_per_batch=10000)
   
   for batch in batch_reader:
       print(f"Processing batch: {batch.num_rows} rows")
       # Process batch
   
   stream_result.close()

Debug and Diagnostics
---------------------

**Enable Verbose Logging**

.. code-block:: python

   import chdb
   
   # Enable detailed output for debugging
   result = chdb.query("SELECT 1", "Pretty")
   
   # Check query performance metrics
   print(f"Rows read: {result.rows_read()}")
   print(f"Bytes read: {result.bytes_read()}")
   print(f"Elapsed time: {result.elapsed()} seconds")

**Session with Debug Parameters**

.. code-block:: python

   from chdb import session as chs
   
   # Create session with debug logging
   sess = chs.Session("debug.chdb?log-level=debug&verbose")
   
   result = sess.query("SELECT version()", "Pretty")
   print(result)

**Command Line Debug Mode**

.. code-block:: bash

   # Run chDB from command line with debug output
   python3 -m chdb "SELECT version()" Pretty
   python3 -m chdb "SELECT count() FROM numbers(100)" JSON

Getting Help
------------

If you need additional help:

1. Check the `GitHub Issues <https://github.com/chdb-io/chdb/issues>`_
2. Read the `ClickHouse Documentation <https://clickhouse.com/docs>`_
3. Join the `Discord Community <https://discord.gg/D2Daa2fM5K>`_
4. Check the `Project Documentation <https://clickhouse.com/docs/en/chdb>`_

Error Reporting
---------------

When reporting errors, please include:

1. chDB version: ``print(chdb.__version__)``
2. Python version: ``print(sys.version)``
3. Operating system
4. Complete error traceback
5. Minimal example that reproduces the issue

.. code-block:: python

   import chdb
   import sys
   
   print(f"chDB version: {chdb.__version__}")
   print(f"Python version: {sys.version}")
   print(f"Engine version: {chdb.engine_version}")

Common Error Messages
---------------------

**"Session already exists"**
Only one session can be active per process. Close existing sessions before creating new ones.

**"Memory limit exceeded"**  
Use streaming queries, file-based sessions, or process data in smaller batches.

**"File not found"**
Check file paths, use absolute paths, and ensure file exists and is readable.

**"Cannot determine file format"**
Explicitly specify file format and schema in your queries.

**"Import Error: No module named '_chdb'"**
Reinstall chDB: ``pip uninstall chdb && pip install chdb``

**"Python version not supported"**
chDB requires Python 3.8+. Upgrade your Python installation.

Frequently Asked Questions
--------------------------

**Q: What platforms does chDB support?**

A: chDB supports Python 3.8+ on macOS and Linux (x86_64 and ARM64). Windows support is not available yet.

**Q: Can chDB work with large datasets?**

A: Yes, chDB can handle large datasets efficiently. Use streaming queries, file-based sessions, and persistent storage for very large datasets.

**Q: Can I use chDB in production?**

A: Yes, chDB is production-ready and part of the ClickHouse family. Test thoroughly in your specific environment and follow best practices for resource management.

**Q: How does chDB compare to SQLite?**

A: chDB is optimized for analytical workloads (OLAP) while SQLite is better for transactional workloads (OLTP). chDB offers better performance for complex analytical queries, aggregations, and data processing tasks.

**Q: What file formats does chDB support?**

A: chDB supports 70+ formats including Parquet, CSV, JSON, Arrow, ORC, and many more. See the `ClickHouse formats documentation <https://clickhouse.com/docs/en/interfaces/formats>`_ for the complete list.

**Q: Can I query Pandas DataFrames directly?**

A: Yes, chDB provides multiple ways to query Pandas DataFrames:
   - ``chdb.dataframe.query()`` function  
   - ``Python(df)`` table engine
   - DataFrame-to-Parquet conversion

**Q: How do I optimize query performance?**

A: Use column selection instead of ``SELECT *``, leverage Parquet format for better performance, use persistent sessions for repeated queries, and consider using streaming for large datasets.

**Q: Can I use external Python libraries in UDFs?**

A: Yes, but you must import all required modules inside the UDF function. UDFs should be stateless and pure Python functions.

