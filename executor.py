"""
Query executor for DataStore.

Provides a high-level interface for query execution, delegating all chDB
operations to the centralized Connection class.

This is a facade that:
- Manages connection lifecycle
- Provides convenience methods for common operations
- Ensures connection is established before queries
"""

import pandas as pd
from typing import Optional
from .connection import Connection, QueryResult
from .exceptions import ExecutionError
from .config import get_logger


class Executor:
    """
    High-level query executor for DataStore.

    Delegates ALL chDB operations to Connection for centralized:
    - Logging
    - Error handling
    - Query execution
    """

    def __init__(self, connection: Optional[Connection] = None):
        """
        Initialize executor.

        Args:
            connection: chdb Connection object (creates default if None)
        """
        self.connection = connection
        self._owns_connection = connection is None
        self._logger = get_logger()

        if self._owns_connection:
            self.connection = Connection(":memory:")

    def _ensure_connected(self):
        """Ensure connection is established."""
        if self.connection is None:
            raise ExecutionError("No connection available")
        if self.connection._conn is None:
            self.connection.connect()

    def execute(self, sql: str) -> QueryResult:
        """
        Execute a SQL query via Connection.

        Args:
            sql: SQL query string

        Returns:
            QueryResult with data and metadata
        """
        self._ensure_connected()
        return self.connection.execute(sql)

    def query_dataframe(self, sql: str, df: pd.DataFrame, df_name: str = '__df__') -> pd.DataFrame:
        """
        Execute a SQL query on a DataFrame using chDB's Python() table function.

        Args:
            sql: SQL query string. Reference DataFrame using Python(df_name) or df_name.
            df: The DataFrame to query
            df_name: Name for the DataFrame in the query (default: '__df__')

        Returns:
            Result DataFrame
        """
        self._ensure_connected()
        return self.connection.query_df(sql, df, df_name)

    def execute_expression(self, expr_sql: str, df: pd.DataFrame, result_column: str = '__result__') -> pd.Series:
        """
        Evaluate a SQL expression on a DataFrame and return the result as a Series.

        Args:
            expr_sql: SQL expression to evaluate (e.g., "CAST(value AS Float64)")
            df: DataFrame to operate on
            result_column: Name for the result column (default: '__result__')

        Returns:
            Result Series with the original DataFrame's index
        """
        self._ensure_connected()
        return self.connection.eval_expression(expr_sql, df, result_column)

    def close(self):
        """Close the connection if we own it."""
        if self._owns_connection and self.connection:
            self.connection.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Global executor instance for convenience
# This allows lazy_ops to use a shared executor without creating new connections
_global_executor: Optional[Executor] = None


def get_executor() -> Executor:
    """
    Get the global Executor instance.

    Creates one if it doesn't exist.

    Returns:
        Global Executor instance
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = Executor()
    return _global_executor


def reset_executor():
    """Reset the global executor (useful for testing)."""
    global _global_executor
    if _global_executor is not None:
        _global_executor.close()
        _global_executor = None
