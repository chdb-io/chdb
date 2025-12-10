"""
Connection management for DataStore using chdb (ClickHouse)
"""

from typing import Any, Optional, Dict, List, Tuple, TYPE_CHECKING
import chdb

from .exceptions import ConnectionError, ExecutionError
from .config import get_logger

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

    def execute(self, sql: str, output_format: str = "Dataframe") -> 'QueryResult':
        """
        Execute a SQL query and return results.

        Args:
            sql: SQL query string
            output_format: Output format for chdb query (default: "Dataframe")

        Returns:
            QueryResult object with data and metadata
        """
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        # Debug logging before chdb execution
        logger = get_logger()
        logger.debug("=" * 70)
        logger.debug("[chdb] Executing query via Connection.execute()")
        logger.debug("-" * 70)
        logger.debug("[chdb] Database: %s", self.database)
        logger.debug("[chdb] Output format: %s", output_format)
        logger.debug("[chdb] SQL:")
        for line in sql.split('\n'):
            logger.debug("  %s", line)
        logger.debug("=" * 70)

        try:
            # Use chdb's query method with DataFrame format by default
            result = self._conn.query(sql, output_format)

            return QueryResult(data=result, output_format=output_format)
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
    Supports both DataFrame and legacy row-based formats.
    """

    def __init__(
        self,
        data: Any = None,
        output_format: str = "Dataframe",
        rows: List[Tuple] = None,
        column_names: List[str] = None,
        column_types: List[str] = None,
        row_count: int = None,
    ):
        """
        Initialize query result.

        Args:
            data: Query result data (DataFrame or other format from chdb)
            output_format: Format of the data ("Dataframe" or other)
            rows: (Legacy) List of result rows (tuples)
            column_names: (Legacy) List of column names
            column_types: (Legacy) List of column types
            row_count: (Legacy) Number of rows
        """
        self.output_format = output_format
        self._data = data

        # Legacy support: if rows/column_names provided directly
        if rows is not None:
            self._rows = rows
            self._column_names = column_names or []
            self._column_types = column_types or []
            self._row_count = row_count if row_count is not None else len(rows) if rows else 0
        else:
            # Initialize from data
            self._rows = None
            self._column_names = None
            self._column_types = None
            self._row_count = None

    @property
    def rows(self) -> List[Tuple]:
        """Get all rows as list of tuples (lazy conversion from DataFrame)."""
        if self._rows is None and self._data is not None:
            # Convert DataFrame to rows
            try:
                import pandas as pd

                if isinstance(self._data, pd.DataFrame):
                    self._rows = [tuple(row) for row in self._data.itertuples(index=False, name=None)]
                else:
                    # Fallback for other formats
                    self._rows = []
            except:
                self._rows = []
        return self._rows if self._rows is not None else []

    @property
    def column_names(self) -> List[str]:
        """Get column names (lazy extraction from DataFrame)."""
        if self._column_names is None and self._data is not None:
            try:
                import pandas as pd

                if isinstance(self._data, pd.DataFrame):
                    self._column_names = list(self._data.columns)
                else:
                    self._column_names = []
            except:
                self._column_names = []
        return self._column_names if self._column_names is not None else []

    @property
    def column_types(self) -> List[str]:
        """Get column types (lazy extraction from DataFrame)."""
        if self._column_types is None and self._data is not None:
            try:
                import pandas as pd

                if isinstance(self._data, pd.DataFrame):
                    self._column_types = [str(dtype) for dtype in self._data.dtypes]
                else:
                    self._column_types = []
            except:
                self._column_types = []
        return self._column_types if self._column_types is not None else []

    @property
    def row_count(self) -> int:
        """Get number of rows (lazy calculation from DataFrame)."""
        if self._row_count is None and self._data is not None:
            try:
                import pandas as pd

                if isinstance(self._data, pd.DataFrame):
                    self._row_count = len(self._data)
                else:
                    self._row_count = 0
            except:
                self._row_count = 0
        return self._row_count if self._row_count is not None else 0

    def fetchone(self) -> Optional[Tuple]:
        """Get the first row."""
        rows = self.rows
        return rows[0] if rows else None

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
        try:
            import pandas as pd

            if isinstance(self._data, pd.DataFrame):
                return self._data.to_dict('records')
        except:
            pass

        # Fallback to legacy method
        rows = self.rows
        if not rows:
            return []
        return [dict(zip(self.column_names, row)) for row in rows]

    def to_df(self):
        """
        Return the result as a pandas DataFrame.

        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd

            if isinstance(self._data, pd.DataFrame):
                return self._data
            else:
                # Convert rows to DataFrame
                if self.rows:
                    return pd.DataFrame(self.rows, columns=self.column_names)
                else:
                    return pd.DataFrame()
        except ImportError:
            raise ImportError("pandas is required for to_df(). Install it with: pip install pandas")

    def __iter__(self):
        """Iterate over rows."""
        return iter(self.rows)

    def __len__(self):
        """Number of rows."""
        return self.row_count

    def __repr__(self):
        return f"QueryResult(rows={self.row_count}, columns={len(self.column_names)})"
