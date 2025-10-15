"""User-defined functions module for chDB.

This module provides functionality for creating and managing user-defined functions (UDFs)
in chDB. It allows you to extend chDB's capabilities by writing custom Python functions
that can be called from SQL queries.
"""

from .udf import chdb_udf, generate_udf

__all__ = ["chdb_udf", "generate_udf"]
