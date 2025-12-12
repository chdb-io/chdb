"""
Connection management for DataStore using chdb (ClickHouse)

This module centralizes ALL chDB query execution with unified logging.

Two execution modes:
1. Connection-based queries: conn.query() for database/file operations
2. DataFrame queries: chdb.query() with Python() table function

All chDB execution should go through this module for:
- Unified logging format
- Centralized error handling
- Future extensibility (caching, metrics, etc.)
"""

from typing import Any, Optional, Dict, List, Tuple
import chdb
import pandas as pd

from .exceptions import ConnectionError, ExecutionError
from .config import get_logger


class Connection:
    """
    Wrapper around chdb connection.

    chdb provides an embedded ClickHouse engine for Python.

    This class centralizes ALL chDB query execution for:
    - Consistent logging
    - Unified error handling
    - Single point of control for query execution
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
        self._logger = get_logger()

    def connect(self) -> 'Connection':
        """Establish connection to chdb."""
        try:
            self._conn = chdb.connect(self.database, **self.connection_params)
            self._logger.debug("[chDB] Connected to database: %s", self.database)
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
        Execute a SQL query via connection and return results.

        This is for database/file queries. For DataFrame queries, use query_df().

        Args:
            sql: SQL query string
            output_format: Output format for chdb query (default: "Dataframe")

        Returns:
            QueryResult object with data and metadata
        """
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        self._log_query(sql, "Connection", output_format)

        try:
            result = self._conn.query(sql, output_format)
            self._log_result(result, output_format)
            return QueryResult(data=result, output_format=output_format)
        except Exception as e:
            self._logger.error("[chDB] Query failed: %s", e)
            raise ExecutionError(f"Query execution failed: {e}\nSQL: {sql}")

    def query_df(self, sql: str, df: pd.DataFrame, df_name: str = '__df__') -> pd.DataFrame:
        """
        Execute a SQL query on a DataFrame using chDB's Python() table function.

        This enables SQL operations on in-memory DataFrames.
        Uses chdb.query() directly (not connection-based) as required by Python().

        Args:
            sql: SQL query string. Use Python(df_name) or df_name to reference the DataFrame.
            df: The DataFrame to query
            df_name: Name for the DataFrame in the query (default: '__df__')

        Returns:
            Result DataFrame
        """
        # Auto-wrap table reference if not already wrapped
        processed_sql = sql
        if f'Python({df_name})' not in sql:
            processed_sql = sql.replace(df_name, f'Python({df_name})')

        self._log_query(processed_sql, "DataFrame")

        try:
            # Execute with DataFrame in local scope
            result = self._execute_df_query(processed_sql, df, df_name)
            self._log_result(result)
            return result
        except Exception as e:
            self._logger.error("[chDB] DataFrame query failed: %s", e)
            raise ExecutionError(f"Failed to execute SQL on DataFrame: {e}")

    def _execute_df_query(self, sql: str, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Internal: execute SQL with DataFrame in local scope.

        chDB's Python() table function requires the DataFrame to be
        accessible in the local scope where chdb.query() is called.
        """
        __df__ = df  # noqa: F841 - Required for chdb.query to access via Python(__df__)
        if df_name != '__df__':
            exec(f"{df_name} = df")
        return chdb.query(sql, 'DataFrame')

    def eval_expression(self, expr_sql: str, df: pd.DataFrame, result_column: str = '__result__') -> pd.Series:
        """
        Evaluate a SQL expression on a DataFrame and return the result as a Series.

        Useful for column assignments like: ds['new_col'] = ds['value'].cast('Float64')

        Args:
            expr_sql: SQL expression to evaluate (e.g., "CAST(value AS Float64)")
            df: DataFrame to operate on
            result_column: Name for the result column (default: '__result__')

        Returns:
            Result Series with the original DataFrame's index
        """
        query = f"SELECT {expr_sql} AS {result_column} FROM Python(__df__)"

        self._log_query(query, "Expression")

        __df__ = df  # noqa: F841
        try:
            result_df = chdb.query(query, 'DataFrame')
            result_series = result_df[result_column]
            result_series.index = df.index
            self._logger.debug("[chDB] Expression result: %d values", len(result_series))
            return result_series
        except Exception as e:
            self._logger.error("[chDB] Expression evaluation failed: %s", e)
            raise ExecutionError(f"Failed to evaluate expression '{expr_sql}': {e}")

    def _log_query(self, sql: str, query_type: str = "Query", output_format: str = None):
        """Unified query logging."""
        self._logger.debug("=" * 70)
        self._logger.debug("[chDB] %s execution", query_type)
        self._logger.debug("-" * 70)
        if output_format:
            self._logger.debug("[chDB] Output format: %s", output_format)
        self._logger.debug("[chDB] SQL:")
        for line in sql.split('\n'):
            self._logger.debug("  %s", line)
        self._logger.debug("=" * 70)

    def _log_result(self, result, output_format: str = None):
        """Unified result logging."""
        if isinstance(result, pd.DataFrame):
            self._logger.debug("[chDB] Result: %d rows x %d cols", len(result), len(result.columns))
        elif hasattr(result, '__len__'):
            self._logger.debug("[chDB] Result: %d rows", len(result))
        else:
            self._logger.debug("[chDB] Query completed")

    def close(self):
        """Close cursor and connection."""
        if self._cursor:
            try:
                self._cursor.close()
            except Exception:
                pass
            self._cursor = None

        if self._conn:
            try:
                self._conn.close()
            except Exception:
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
            except Exception:
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
            except Exception:
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
            except Exception:
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
            except Exception:
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
        except Exception:
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
