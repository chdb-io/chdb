DB-API 2.0 Interface
====================

chDB provides a Python DB-API 2.0 compatible interface for database connectivity.

Connection Management
---------------------

.. automodule:: chdb.dbapi.connections
   :show-inheritance:

Cursor Operations
-----------------

.. automodule:: chdb.dbapi.cursors
   :show-inheritance:

Error Handling
--------------

.. automodule:: chdb.dbapi.err
   :show-inheritance:

Type Converters
---------------

.. automodule:: chdb.dbapi.converters
   :show-inheritance:

Time Utilities
--------------

.. automodule:: chdb.dbapi.times
   :show-inheritance:

Basic Usage Example
-------------------

.. code-block:: python

   import chdb.dbapi as dbapi
   
   # Create connection
   conn = dbapi.connect()
   cursor = conn.cursor()
   
   # Execute query
   cursor.execute("SELECT number FROM numbers(5)")
   
   # Fetch results
   results = cursor.fetchall()
   for row in results:
       print(row)
   
   # Close connection
   cursor.close()
   conn.close()

Connection String Examples
--------------------------

.. code-block:: python

   import chdb.dbapi as dbapi
   
   # In-memory database
   conn = dbapi.connect()
   
   # Persistent database
   conn = dbapi.connect("./my_database")
   
   # With additional parameters
   conn = dbapi.connect("./my_database?log-level=debug")