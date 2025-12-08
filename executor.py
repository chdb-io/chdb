"""
Query executor for DataStore.
"""

from typing import Optional, Any
from .connection import Connection, QueryResult
from .exceptions import ExecutionError
from .config import get_logger


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
        self._logger = get_logger()

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

        # Log SQL execution at DEBUG level
        self._logger.debug("=" * 60)
        self._logger.debug("Executing SQL:")
        self._logger.debug("-" * 60)
        for line in sql.split('\n'):
            self._logger.debug("  %s", line)
        self._logger.debug("=" * 60)

        result = self.connection.execute(sql)

        # Log result info
        self._logger.debug("Query returned %d rows", len(result.rows) if result.rows else 0)

        return result

    def close(self):
        """Close the connection if we own it."""
        if self._owns_connection and self.connection:
            self.connection.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
