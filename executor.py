"""
Query executor for DataStore.
"""

from typing import Optional, Any
from .connection import Connection, QueryResult
from .exceptions import ExecutionError


class Executor:
    """
    Executes DataStore queries using chdb.
    """

    def __init__(self, connection: Optional[Connection] = None):
        """
        Initialize executor.

        Args:
            connection: chdb Connection object (creates default if None)
        """
        self.connection = connection
        self._owns_connection = connection is None

        if self._owns_connection:
            # Create default in-memory connection
            self.connection = Connection(":memory:")

    def execute(self, sql: str) -> QueryResult:
        """
        Execute a SQL query.

        Args:
            sql: SQL query string

        Returns:
            QueryResult with data and metadata
        """
        if self.connection is None:
            raise ExecutionError("No connection available")

        # Ensure connected
        if self.connection._conn is None:
            self.connection.connect()

        return self.connection.execute(sql)

    def close(self):
        """Close the connection if we own it."""
        if self._owns_connection and self.connection:
            self.connection.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
