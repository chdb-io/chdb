Troubleshooting
===============

Common Issues
-------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: bash

   ImportError: No module named '_chdb'

**Solution**: Ensure chDB is properly installed:

.. code-block:: bash

   pip uninstall chdb
   pip install chdb

Memory Issues
~~~~~~~~~~~~~

If you encounter memory-related errors:

.. code-block:: text

   Memory limit exceeded

**Solutions**:

1. Reduce query complexity
2. Process data in smaller chunks
3. Increase available memory
4. Use persistent storage instead of in-memory

.. code-block:: python

   # Use persistent storage to reduce memory usage
   result = chdb.query("SELECT * FROM large_table", path="./data")

File Access Issues
~~~~~~~~~~~~~~~~~~

If you encounter file access errors:

.. code-block:: text

   Permission denied: Cannot read file

**Solutions**:

1. Check file permissions
2. Ensure file exists
3. Use absolute paths

.. code-block:: python

   import os
   
   # Use absolute path
   file_path = os.path.abspath("data.csv")
   result = chdb.query(f"SELECT * FROM file('{file_path}', 'CSV')")

Performance Issues
~~~~~~~~~~~~~~~~~~

If queries are running slowly:

**Solutions**:

1. Use appropriate indexes
2. Optimize query structure
3. Use column selection instead of SELECT *
4. Consider data partitioning

.. code-block:: python

   # Good: Select specific columns
   result = chdb.query("SELECT id, name FROM users WHERE id > 100")
   
   # Avoid: Select all columns
   # result = chdb.query("SELECT * FROM users WHERE id > 100")

Debug Mode
----------

Enable debug mode for more detailed error messages:

.. code-block:: python

   import chdb
   
   # Enable debug mode
   result = chdb.query("SELECT 1", "Debug")

Getting Help
------------

If you need additional help:

1. Check the `GitHub Issues <https://github.com/chdb-io/chdb/issues>`_
2. Read the `ClickHouse Documentation <https://clickhouse.com/docs>`_
3. Join the community discussions

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

Frequently Asked Questions
--------------------------

**Q: Can chDB work with large datasets?**

A: Yes, chDB can handle large datasets efficiently. Use persistent storage for very large datasets to reduce memory usage.

**Q: Is chDB thread-safe?**

A: chDB connections should not be shared across threads. Create separate connections for each thread.

**Q: Can I use chDB in production?**

A: Yes, chDB is production-ready. However, test thoroughly in your specific environment.

**Q: How does chDB compare to SQLite?**

A: chDB is optimized for analytical workloads (OLAP) while SQLite is better for transactional workloads (OLTP). chDB offers better performance for complex analytical queries.