"""
Connection management for DataStore using chdb (ClickHouse)
"""

from typing import Any, Optional, Dict, List, Tuple, TYPE_CHECKING
import chdb

from .exceptions import ConnectionError, ExecutionError

if TYPE_CHECKING:
    from chdb.state.sqlitelike import Cursor


class Connection:
    """
    Wrapper around chdb connection.

    chdb provides an embedded ClickHouse engine for Python.
    """

    def __init__(self, database: str = ":memory:", **kwargs):
        """
        Initialize chdb connection.

        Args:
            database: Database path (":memory:" for in-memory, or file path)
            **kwargs: Additional connection parameters
        """
        self.database = database
        self.connection_params = kwargs
        self._conn = None
        self._cursor = None

    def connect(self) -> 'Connection':
        """Establish connection to chdb."""
        try:
            self._conn = chdb.connect(self.database, **self.connection_params)
            return self
        except Exception as e:
            raise ConnectionError(f"Failed to connect to chdb: {e}")

    def cursor(self):
        """Get a cursor for executing queries."""
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        try:
            self._cursor = self._conn.cursor()
            return self._cursor
        except Exception as e:
            raise ConnectionError(f"Failed to create cursor: {e}")

    def execute(self, sql: str) -> 'QueryResult':
        """
        Execute a SQL query and return results.

        Args:
            sql: SQL query string

        Returns:
            QueryResult object with data and metadata
        """
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        try:
            cursor = self.cursor()
            cursor.execute(sql)

            # Fetch all results
            rows = cursor.fetchall()

            # Get metadata
            column_names = cursor.column_names() if hasattr(cursor, 'column_names') else []
            column_types = cursor.column_types() if hasattr(cursor, 'column_types') else []

            return QueryResult(
                rows=rows, column_names=column_names, column_types=column_types, row_count=len(rows) if rows else 0
            )
        except Exception as e:
            raise ExecutionError(f"Query execution failed: {e}\nSQL: {sql}")

    def close(self):
        """Close cursor and connection."""
        if self._cursor:
            try:
                self._cursor.close()
            except:
                pass
            self._cursor = None

        if self._conn:
            try:
                self._conn.close()
            except:
                pass
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class QueryResult:
    """
    Wrapper for query results.

    Provides convenient access to query results with metadata.
    """

    def __init__(self, rows: List[Tuple], column_names: List[str], column_types: List[str], row_count: int):
        """
        Initialize query result.

        Args:
            rows: List of result rows (tuples)
            column_names: List of column names
            column_types: List of column types
            row_count: Number of rows
        """
        self.rows = rows
        self.column_names = column_names
        self.column_types = column_types
        self.row_count = row_count

    def fetchone(self) -> Optional[Tuple]:
        """Get the first row."""
        return self.rows[0] if self.rows else None

    def fetchall(self) -> List[Tuple]:
        """Get all rows."""
        return self.rows

    def fetchmany(self, size: int) -> List[Tuple]:
        """Get first 'size' rows."""
        return self.rows[:size]

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert results to list of dictionaries.

        Returns:
            List of dicts where keys are column names
        """
        if not self.rows:
            return []

        return [dict(zip(self.column_names, row)) for row in self.rows]

    def __iter__(self):
        """Iterate over rows."""
        return iter(self.rows)

    def __len__(self):
        """Number of rows."""
        return self.row_count

    def __repr__(self):
        return f"QueryResult(rows={self.row_count}, columns={len(self.column_names)})"
