API Reference
=============

Core Functions
--------------

.. automodule:: chdb
   :members: query, sql, to_df, to_arrowTable, chdb_version, engine_version
   :show-inheritance:

Exception Classes
-----------------

.. autoclass:: chdb.ChdbError
   :show-inheritance:

PyReader Class
--------------

.. autoclass:: chdb.PyReader
   :show-inheritance:

Connection Functions
--------------------

.. autofunction:: chdb.connect

Version Information
-------------------

.. autodata:: chdb.chdb_version
.. autodata:: chdb.engine_version
.. autodata:: chdb.__version__