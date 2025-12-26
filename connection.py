"""
Connection management for DataStore using chdb (ClickHouse)

This module centralizes ALL chDB query execution with unified logging.

All queries (database/file operations and DataFrame queries) use
conn.query() for consistent connection-based execution.

All chDB execution should go through this module for:
- Unified logging format
- Centralized error handling
- Better performance (reusing connection)
- Future extensibility (caching, metrics, etc.)
"""

from typing import Any, Optional, Dict, List, Tuple
import time
import chdb
import pandas as pd
import numpy as np

from .exceptions import ConnectionError, ExecutionError
from .config import get_logger, get_profiler


def _convert_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert nullable pandas dtypes to non-nullable equivalents for chDB compatibility.

    On Python 3.8, chDB has issues reading from DataFrames with nullable dtypes
    (Int64, Float64, etc.) which can cause uninitialized memory reads.

    This function converts:
    - Float64 -> float64 (NaN preserved as np.nan)
    - Int64 -> int64 (NA values become 0, use with caution)
    - Other nullable types -> their non-nullable equivalents

    Args:
        df: DataFrame that may contain nullable dtypes

    Returns:
        DataFrame with non-nullable dtypes
    """
    import sys

    # Only apply fix for Python 3.8.x
    if sys.version_info[:2] != (3, 8):
        return df

    result = df
    needs_copy = False

    for col in df.columns:
        dtype = df[col].dtype
        dtype_str = str(dtype)

        # Check for nullable extension types
        if dtype_str == 'Float64':
            if not needs_copy:
                result = df.copy()
                needs_copy = True
            # Convert to float64, preserving NaN
            result[col] = df[col].astype('float64')
        elif dtype_str == 'Float32':
            if not needs_copy:
                result = df.copy()
                needs_copy = True
            result[col] = df[col].astype('float32')
        elif dtype_str in ('Int64', 'Int32', 'Int16', 'Int8', 'UInt64', 'UInt32', 'UInt16', 'UInt8'):
            if not needs_copy:
                result = df.copy()
                needs_copy = True
            # For nullable integers with NA, convert to float to preserve NaN
            if df[col].isna().any():
                result[col] = df[col].astype('float64')
            else:
                result[col] = df[col].astype(dtype_str.lower())
        elif dtype_str == 'boolean':
            if not needs_copy:
                result = df.copy()
                needs_copy = True
            if df[col].isna().any():
                result[col] = df[col].astype('object')
            else:
                result[col] = df[col].astype('bool')
        elif dtype_str == 'string':
            if not needs_copy:
                result = df.copy()
                needs_copy = True
            result[col] = df[col].astype('object')

    return result


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
            start_time = time.perf_counter()
            result = self._conn.query(sql, output_format)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self._log_result(result, output_format)
            self._logger.debug("[chDB] Query time: %.2fms", elapsed_ms)

            # Add profiling info if profiler is active
            profiler = get_profiler()
            if profiler:
                sql_preview = sql[:50] + "..." if len(sql) > 50 else sql
                with profiler.step("chDB Query", sql=sql_preview, time_ms=f"{elapsed_ms:.2f}"):
                    pass  # Already executed, just record timing

            return QueryResult(data=result, output_format=output_format)
        except Exception as e:
            self._logger.error("[chDB] Query failed: %s", e)
            raise ExecutionError(f"Query execution failed: {e}\nSQL: {sql}")

    def query_df(
        self,
        sql: str,
        df: pd.DataFrame,
        df_name: str = '__df__',
        preserve_order: bool = True,
    ) -> pd.DataFrame:
        """
        Execute a SQL query on a DataFrame using chDB's Python() table function.

        This enables SQL operations on in-memory DataFrames.
        Uses conn.query() for better performance with connection reuse.

        IMPORTANT: chDB's Python() table function does NOT guarantee row order by default.
        Set preserve_order=True (default) to maintain original DataFrame row order.
        This adds a hidden index column and ORDER BY clause with minimal overhead (~2-5%).

        Args:
            sql: SQL query string. Use Python(df_name) or df_name to reference the DataFrame.
            df: The DataFrame to query
            df_name: Name for the DataFrame in the query (default: '__df__')
            preserve_order: If True, preserve original row order (default: True)

        Returns:
            Result DataFrame with original row order preserved (if preserve_order=True)
        """
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        # Prepare DataFrame with row index if order preservation is needed
        # When preserve_order is explicitly True, always preserve order
        # (only skip for GROUP BY or explicit ORDER BY where order semantics change)
        row_idx_col = '__row_idx__'
        df_to_use = df
        sql_upper = sql.upper().strip()

        # Determine how to handle row order preservation
        # __row_idx__ is used to prevent chDB from shuffling original row order
        has_order_by = self._has_outer_order_by(sql_upper)
        has_group_by = 'GROUP BY' in sql_upper
        is_aggregate = self._is_aggregate_query(sql_upper)

        # Determine row index handling strategy
        needs_row_idx = False  # Whether to add __row_idx__ column
        add_order_by_row_idx = False  # Whether to add ORDER BY __row_idx__ (no existing ORDER BY)
        add_row_idx_as_tiebreaker = False  # Whether to append __row_idx__ to existing ORDER BY

        if not preserve_order:
            pass  # All False
        elif has_group_by or is_aggregate:
            # GROUP BY/aggregate: rows don't correspond to original rows
            # __row_idx__ is meaningless after aggregation
            pass  # All False
        elif has_order_by:
            # Has explicit ORDER BY: add __row_idx__ as tie-breaker for stable sort
            # This ensures rows with equal sort keys maintain original relative order
            needs_row_idx = True
            add_row_idx_as_tiebreaker = True
        elif self._is_select_star_query(sql_upper) or '__ROW_IDX__' in sql_upper:
            # SELECT * or query already has __row_idx__: preserve original order
            needs_row_idx = True
            add_order_by_row_idx = True
        # else: other queries without ORDER BY - no preservation needed

        if needs_row_idx:
            # Add row position column (0, 1, 2, ...) to preserve original row order
            # This is independent of DataFrame's index - it's purely for chDB row ordering
            df_to_use = df.copy()
            df_to_use[row_idx_col] = range(len(df))

        # Auto-wrap table reference if not already wrapped
        processed_sql = sql
        if f'Python({df_name})' not in sql:
            processed_sql = sql.replace(df_name, f'Python({df_name})')

        # Handle ORDER BY modifications
        if add_order_by_row_idx:
            # No existing ORDER BY: wrap query and add ORDER BY __row_idx__
            processed_sql = self._add_order_by(processed_sql, row_idx_col)
        elif add_row_idx_as_tiebreaker:
            # Has existing ORDER BY: append __row_idx__ as tie-breaker for stable sort
            processed_sql = self._append_tiebreaker_to_orderby(processed_sql, row_idx_col)

        self._log_query(processed_sql, "DataFrame")

        try:
            start_time = time.perf_counter()
            # Execute with DataFrame in local scope
            result = self._execute_df_query(processed_sql, df_to_use, df_name)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Restore original pandas index from __row_idx__ (row positions)
            # Map row positions back to original index values
            if needs_row_idx and row_idx_col in result.columns:
                # __row_idx__ contains row positions (0, 1, 2, ...)
                # Look up original index values from these positions
                row_positions = result[row_idx_col].values
                result = result.drop(columns=[row_idx_col])
                result.index = df.index[row_positions]
                result.index.name = df.index.name  # Preserve original index name

            self._log_result(result)
            self._logger.debug(
                "[chDB] DataFrame query time: %.2fms (rows_in=%d, rows_out=%d)", elapsed_ms, len(df), len(result)
            )

            # Add profiling info if profiler is active
            profiler = get_profiler()
            if profiler:
                sql_preview = processed_sql[:50] + "..." if len(processed_sql) > 50 else processed_sql
                with profiler.step(
                    "chDB DataFrame Query",
                    sql=sql_preview,
                    rows_in=len(df),
                    rows_out=len(result),
                    time_ms=f"{elapsed_ms:.2f}",
                ):
                    pass  # Already executed, just record timing

            return result
        except Exception as e:
            self._logger.error("[chDB] DataFrame query failed: %s", e)
            raise ExecutionError(f"Failed to execute SQL on DataFrame: {e}")

    def _should_preserve_order(self, sql: str) -> bool:
        """
        Determine if the query needs order preservation.

        Returns False for:
        - Queries with explicit ORDER BY (user controls order)
        - Queries with GROUP BY (grouped results have their own order)
        - Queries that select specific columns (not SELECT *)
        - Aggregate-only queries (no row correspondence)

        Order preservation only makes sense for SELECT * queries from the table,
        where we want to maintain the original DataFrame row order.
        """
        sql_upper = sql.upper().strip()

        # Skip if query already has ORDER BY at the outer level
        if self._has_outer_order_by(sql_upper):
            return False

        # Skip for GROUP BY queries (aggregation changes row semantics)
        if 'GROUP BY' in sql_upper:
            return False

        # Skip for explicit column selection (not SELECT *)
        # User queries that select specific columns are intentional about output
        if not self._is_select_star_query(sql_upper):
            return False

        return True

    def _is_select_star_query(self, sql_upper: str) -> bool:
        """
        Check if the query is a simple SELECT * FROM table type query.

        This includes:
        - SELECT * FROM table
        - SELECT *, expr AS alias FROM table
        - SELECT * FROM table WHERE ...

        But not:
        - SELECT col1, col2 FROM table
        - SELECT col AS alias FROM table
        """
        # Find SELECT ... FROM pattern
        select_pos = sql_upper.find('SELECT')
        from_pos = sql_upper.find('FROM')

        if select_pos == -1 or from_pos == -1 or select_pos > from_pos:
            return False

        # Extract the SELECT clause
        select_clause = sql_upper[select_pos + 6 : from_pos].strip()

        # Check if it starts with * or contains *,
        # This covers: SELECT *, SELECT *, col, SELECT *
        if select_clause.startswith('*'):
            return True

        return False

    def _has_outer_order_by(self, sql_upper: str) -> bool:
        """
        Check if SQL has ORDER BY at the outer query level (not in subquery).
        """
        # Simple approach: find ORDER BY and check if it's balanced with parentheses
        order_pos = sql_upper.rfind('ORDER BY')
        if order_pos == -1:
            return False

        # Count parentheses after ORDER BY position
        # If we're inside a subquery, there will be more closing than opening parens
        after_order = sql_upper[order_pos:]
        open_count = after_order.count('(')
        close_count = after_order.count(')')

        # If balanced or more opens than closes, ORDER BY is at outer level
        return open_count >= close_count

    def _is_aggregate_query(self, sql_upper: str) -> bool:
        """
        Check if the query is an aggregate query (has aggregate functions but no GROUP BY).

        Aggregate queries return a single row or reduced result set,
        so row order preservation doesn't make sense.
        """
        # Common aggregate functions
        aggregate_functions = [
            'COUNT(',
            'SUM(',
            'AVG(',
            'MIN(',
            'MAX(',
            'STDDEV(',
            'STDDEVPOP(',
            'STDDEVSAMP(',
            'VAR(',
            'VARPOP(',
            'VARSAMP(',
            'CORR(',
            'COVAR_POP(',
            'COVAR_SAMP(',
            'ANY(',
            'ANYIF(',
            'UNIQ(',
            'UNIQEXACT(',
            'QUANTILE(',
            'QUANTILES(',
            'MEDIAN(',
            'MEDIANEXACT(',
        ]

        # Check if any aggregate function is present
        has_aggregate = any(func in sql_upper for func in aggregate_functions)

        # It's an aggregate query if it has aggregate functions but no GROUP BY
        # (GROUP BY is already handled separately)
        return has_aggregate

    def _add_order_by(self, sql: str, order_col: str) -> str:
        """
        Add ORDER BY clause to preserve row order.

        For queries with LIMIT/OFFSET at the outer level, ORDER BY must be applied
        BEFORE them. For nested subqueries with LIMIT/OFFSET inside, we wrap the
        whole query and add ORDER BY outside.
        """
        sql_upper = sql.upper()

        # Check if LIMIT/OFFSET exists at the OUTER level (not inside a subquery)
        limit_pos = self._find_outer_clause_position(sql_upper, 'LIMIT')
        offset_pos = self._find_outer_clause_position(sql_upper, 'OFFSET')

        if limit_pos != -1 or offset_pos != -1:
            # LIMIT/OFFSET at outer level - insert ORDER BY before it
            clause_start = len(sql)
            if limit_pos != -1:
                clause_start = min(clause_start, limit_pos)
            if offset_pos != -1:
                clause_start = min(clause_start, offset_pos)

            base_query = sql[:clause_start].strip()
            limit_offset_clause = sql[clause_start:].strip()
            return f"{base_query} ORDER BY \"{order_col}\" {limit_offset_clause}"
        else:
            # No LIMIT/OFFSET at outer level: wrap in subquery
            return f"SELECT * FROM ({sql}) ORDER BY \"{order_col}\""

    def _find_outer_clause_position(self, sql_upper: str, clause: str) -> int:
        """
        Find the position of a clause (LIMIT/OFFSET) that is at the outer query level.

        Returns -1 if the clause doesn't exist or is only inside subqueries.
        """
        # Find all occurrences of the clause
        search_term = f' {clause} '
        pos = 0
        while True:
            found = sql_upper.find(search_term, pos)
            if found == -1:
                # Also check for clause at end without trailing space
                if sql_upper.endswith(f' {clause}'):
                    found = len(sql_upper) - len(clause) - 1
                else:
                    return -1
            else:
                pos = found + 1

            # Check if this occurrence is at outer level by counting parens before it
            prefix = sql_upper[:found]
            open_parens = prefix.count('(')
            close_parens = prefix.count(')')

            # If balanced, clause is at outer level
            if open_parens == close_parens:
                return found

        return -1

    def _append_tiebreaker_to_orderby(self, sql: str, tiebreaker_col: str) -> str:
        """
        Append a tie-breaker column to existing ORDER BY clause for stable sort.

        When there's an explicit ORDER BY, we append __row_idx__ as a secondary
        sort key to maintain original relative order for rows with equal sort keys.
        This implements stable sort behavior like pandas sort_values(kind='stable').

        Example:
            Input:  SELECT * FROM t ORDER BY name DESC
            Output: SELECT * FROM t ORDER BY name DESC, __row_idx__ ASC
        """
        sql_upper = sql.upper()

        # Find the last ORDER BY position (to handle subqueries)
        order_by_pos = sql_upper.rfind('ORDER BY')
        if order_by_pos == -1:
            # No ORDER BY found, shouldn't happen but handle gracefully
            return sql

        # Find where ORDER BY clause ends (LIMIT, OFFSET, or end of string)
        order_by_end = len(sql)
        for keyword in [' LIMIT ', ' OFFSET ', ';']:
            pos = sql_upper.find(keyword, order_by_pos)
            if pos != -1 and pos < order_by_end:
                order_by_end = pos

        # Insert tie-breaker before the end of ORDER BY clause
        return sql[:order_by_end] + f', "{tiebreaker_col}" ASC' + sql[order_by_end:]

    def _execute_df_query(self, sql: str, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Internal: execute SQL with DataFrame in local scope.

        chDB's Python() table function requires the DataFrame to be
        accessible in the local scope where conn.query() is called.
        """
        # Convert nullable dtypes to non-nullable for Python 3.8 compatibility
        df_converted = _convert_nullable_dtypes(df)
        __df__ = df_converted  # noqa: F841 - Required for conn.query to access via Python(__df__)
        if df_name != '__df__':
            exec(f"{df_name} = df_converted")
        return self._conn.query(sql, 'DataFrame')

    def eval_expression(self, expr_sql: str, df: pd.DataFrame, result_column: str = '__result__') -> pd.Series:
        """
        Evaluate a SQL expression on a DataFrame and return the result as a Series.

        Useful for column assignments like: ds['new_col'] = ds['value'].cast('Float64')

        IMPORTANT: This method preserves row order by adding an index column and
        ORDER BY clause to ensure results align with the original DataFrame.
        For aggregate expressions, ORDER BY is skipped as they return single values.

        Args:
            expr_sql: SQL expression to evaluate (e.g., "CAST(value AS Float64)")
            df: DataFrame to operate on
            result_column: Name for the result column (default: '__result__')

        Returns:
            Result Series with the original DataFrame's index
        """
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        # Check if expression contains aggregate functions
        is_aggregate = self._is_aggregate_expression(expr_sql)

        # Convert nullable dtypes to non-nullable for Python 3.8 compatibility
        df_converted = _convert_nullable_dtypes(df)

        if is_aggregate:
            # Aggregate expressions return single value, no ORDER BY needed
            __df__ = df_converted  # noqa: F841
            query = f"SELECT {expr_sql} AS {result_column} FROM Python(__df__)"
        else:
            # Add row index to preserve order for row-level expressions
            row_idx_col = '__row_idx__'
            __df__ = df_converted.copy()  # noqa: F841
            __df__[row_idx_col] = range(len(df))
            query = f"SELECT {expr_sql} AS {result_column} FROM Python(__df__) ORDER BY {row_idx_col}"

        self._log_query(query, "Expression")

        try:
            start_time = time.perf_counter()
            result_df = self._conn.query(query, 'DataFrame')
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            result_series = result_df[result_column]
            if not is_aggregate:
                result_series.index = df.index
            self._logger.debug("[chDB] Expression result: %d values, time: %.2fms", len(result_series), elapsed_ms)

            # Add profiling info if profiler is active
            profiler = get_profiler()
            if profiler:
                expr_preview = expr_sql[:40] + "..." if len(expr_sql) > 40 else expr_sql
                with profiler.step("chDB Expression", expr=expr_preview, rows=len(df), time_ms=f"{elapsed_ms:.2f}"):
                    pass  # Already executed, just record timing

            return result_series
        except Exception as e:
            self._logger.error("[chDB] Expression evaluation failed: %s", e)
            raise ExecutionError(f"Failed to evaluate expression '{expr_sql}': {e}")

    def _is_aggregate_expression(self, expr_sql: str) -> bool:
        """
        Check if SQL expression contains aggregate functions.

        Aggregate functions return a single value and cannot be used with ORDER BY
        on non-aggregated columns.

        Args:
            expr_sql: SQL expression to check

        Returns:
            True if expression contains aggregate functions
        """
        import re

        # Common aggregate function patterns (case-insensitive)
        aggregate_patterns = [
            r'\bavg\s*\(',
            r'\bsum\s*\(',
            r'\bcount\s*\(',
            r'\bmin\s*\(',
            r'\bmax\s*\(',
            r'\bmedian\s*\(',
            r'\bstddev\w*\s*\(',
            r'\bvar\w*\s*\(',
            r'\bany\s*\(',
            r'\ball\s*\(',
            r'\bargMin\s*\(',
            r'\bargMax\s*\(',
            r'\buniq\w*\s*\(',
            r'\bgroupArray\s*\(',
            r'\bgroupUniqArray\s*\(',
            r'\bquantile\w*\s*\(',
        ]

        expr_lower = expr_sql.lower()
        for pattern in aggregate_patterns:
            if re.search(pattern, expr_lower, re.IGNORECASE):
                return True
        return False

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
        if isinstance(self._data, pd.DataFrame):
            return self._data
        else:
            # Convert rows to DataFrame
            if self.rows:
                return pd.DataFrame(self.rows, columns=self.column_names)
            else:
                return pd.DataFrame()

    def __iter__(self):
        """Iterate over rows."""
        return iter(self.rows)

    def __len__(self):
        """Number of rows."""
        return self.row_count

    def __repr__(self):
        return f"QueryResult(rows={self.row_count}, columns={len(self.column_names)})"
