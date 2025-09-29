API Reference
=============

Core Query Functions
--------------------

.. automodule:: chdb
   :members: query, sql, to_df, to_arrowTable
   :show-inheritance:

Connection and Session Management
---------------------------------

**Session Functions**

.. autofunction:: chdb.connect

.. automodule:: chdb.session
   :members: Session
   :show-inheritance:

**State Management**

.. automodule:: chdb.state
   :members: connect
   :show-inheritance:
   :exclude-members: __init__

.. automodule:: chdb.state.sqlitelike
   :members: to_arrowTable, to_df, Connection, Cursor
   :show-inheritance:
   :exclude-members: __init__

DataFrame Integration
---------------------

.. automodule:: chdb.dataframe
   :members:
   :show-inheritance:

.. autoclass:: chdb.dataframe.Table
   :show-inheritance:
   :no-index:

Database API (DBAPI) 2.0 Interface
-----------------------------------

chDB provides a Python DB-API 2.0 compatible interface for database connectivity, allowing you to use chDB with tools and frameworks that expect standard database interfaces.

The chDB DB-API 2.0 interface includes:

- **Connections**: Database connection management with connection strings
- **Cursors**: Query execution and result retrieval 
- **Type System**: DB-API 2.0 compliant type constants and converters
- **Error Handling**: Standard database exception hierarchy
- **Thread Safety**: Level 1 thread safety (threads may share modules but not connections)

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

**Module Constants**

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

**Usage Examples**

Basic Query Example:

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

Working with Data:

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

Connection Management:

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
   
   # Automatic connection cleanup
   with dbapi.connect("test.chdb") as conn:
       cur = conn.cursor()
       cur.execute("SELECT count() FROM numbers(1000)")
       result = cur.fetchone()
       print(f"Count: {result[0]}")
       cur.close()

**Best Practices**

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

User-Defined Functions (UDF)
-----------------------------

.. automodule:: chdb.udf
   :members:
   :show-inheritance:

Utilities
---------

.. automodule:: chdb.utils
   :members:
   :show-inheritance:

Abstract Base Classes
---------------------

.. automodule:: chdb.rwabc
   :members: PyReader, PyWriter
   :show-inheritance:
   :exclude-members: __init__

Exception Handling
------------------

.. autoclass:: chdb.ChdbError
   :show-inheritance:

Version Information
-------------------

.. autodata:: chdb.chdb_version
.. autodata:: chdb.engine_version
.. autodata:: chdb.__version__
