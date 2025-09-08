Utilities
=========

chDB provides various utility functions and classes for enhanced functionality.

Type Utilities
--------------

.. automodule:: chdb.utils.types
   :show-inheritance:

Tracing Utilities
-----------------

.. automodule:: chdb.utils.trace
   :show-inheritance:

Utils Module
------------

.. automodule:: chdb.utils
   :show-inheritance:

Type Conversion Examples
------------------------

Working with different data types:

.. code-block:: python

   import chdb
   from chdb.utils import types
   
   # Convert Python types to ClickHouse types
   ch_type = types.python_to_clickhouse(int)
   print(ch_type)  # Int64
   
   # Convert ClickHouse types to Python types
   py_type = types.clickhouse_to_python("String")
   print(py_type)  # str

Tracing and Debugging
---------------------

Enable tracing for debugging:

.. code-block:: python

   import chdb
   from chdb.utils import trace
   
   # Enable query tracing
   trace.enable_trace()
   
   # Run query with tracing
   result = chdb.query("SELECT count() FROM numbers(1000)")
   
   # View trace information
   trace_info = trace.get_trace()
   print(trace_info)

Performance Monitoring
----------------------

Monitor query performance:

.. code-block:: python

   import chdb
   import time
   
   # Time query execution
   start_time = time.time()
   result = chdb.query("SELECT count() FROM numbers(1000000)")
   end_time = time.time()
   
   print(f"Query took {end_time - start_time:.4f} seconds")

Memory Usage
------------

Monitor memory usage:

.. code-block:: python

   import chdb
   import psutil
   import os
   
   # Get current process
   process = psutil.Process(os.getpid())
   
   # Monitor memory before query
   memory_before = process.memory_info().rss / 1024 / 1024  # MB
   
   # Execute query
   result = chdb.query("SELECT * FROM numbers(1000000)")
   
   # Monitor memory after query
   memory_after = process.memory_info().rss / 1024 / 1024  # MB
   
   print(f"Memory usage: {memory_after - memory_before:.2f} MB")