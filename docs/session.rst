Session Management
==================

chDB provides session management for maintaining state across multiple queries.

Session State
-------------

.. automodule:: chdb.session.state
   :show-inheritance:

Session Interface
-----------------

.. automodule:: chdb.session
   :show-inheritance:

Basic Usage
-----------

.. code-block:: python

   import chdb.session as session
   
   # Create a new session
   sess = session.Session()
   
   # Execute queries with persistent state
   sess.sql("CREATE TABLE temp_table (id Int32, name String)")
   sess.sql("INSERT INTO temp_table VALUES (1, 'Alice'), (2, 'Bob')")
   result = sess.sql("SELECT * FROM temp_table")
   print(result)

Session Configuration
---------------------

Sessions can be configured with various parameters:

.. code-block:: python

   import chdb.session as session
   
   # Create session with custom settings
   sess = session.Session(
       path="./session_data",
       settings={
           "max_memory_usage": "1G",
           "log_level": "debug"
       }
   )

Session Persistence
-------------------

Sessions can persist data across Python interpreter restarts:

.. code-block:: python

   import chdb.session as session
   
   # Create persistent session
   sess = session.Session(path="./persistent_session")
   
   # Create table (persisted to disk)
   sess.sql("CREATE TABLE users (id Int32, name String) ENGINE = MergeTree() ORDER BY id")
   
   # Data will be available in future sessions using the same path