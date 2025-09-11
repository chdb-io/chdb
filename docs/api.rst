API Reference
=============

Core Query Functions
--------------------

.. automodule:: chdb
   :members: query, sql, to_df, to_arrowTable
   :show-inheritance:

Connection and Session
----------------------

.. autofunction:: chdb.connect

.. seealso:: :doc:`session` - Complete session management documentation

State Management
----------------

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

Database API (DBAPI) Support
-----------------------------

.. seealso:: :doc:`dbapi` - Complete DB-API 2.0 interface documentation

User-Defined Functions (UDF)
-----------------------------

.. seealso:: :doc:`udf` - Complete UDF documentation and examples

Exception Handling
------------------

.. autoclass:: chdb.ChdbError
   :show-inheritance:

Abstract Base Classes
---------------------

.. automodule:: chdb.rwabc
   :members: PyReader, PyWriter
   :show-inheritance:
   :exclude-members: __init__

Utilities
---------

.. seealso:: :doc:`utils` - Complete utilities documentation and examples

Version Information
-------------------

.. autodata:: chdb.chdb_version
.. autodata:: chdb.engine_version
.. autodata:: chdb.__version__
