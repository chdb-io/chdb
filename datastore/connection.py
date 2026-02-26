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
    Legacy function for Python 3.8 compatibility (no longer needed).

    Previously converted nullable pandas dtypes to non-nullable equivalents.
    Now returns the DataFrame unchanged since Python 3.9+ handles nullable dtypes correctly.

    Args:
        df: DataFrame that may contain nullable dtypes

    Returns:
        DataFrame unchanged
    """
    return df


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
        This uses chDB's built-in _row_id virtual column for ORDER BY with minimal overhead.

        Note: _row_id is a built-in virtual column in chDB v4.0.0b5+ that provides
        the 0-based row number from the original DataFrame, enabling deterministic
        row order preservation without manually adding an index column.

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

        # Handle integer column names (chDB Python() can't handle them)
        # Convert to strings for SQL execution, track for restoration
        int_col_map = {}  # str -> original (int or other non-string)
        df_for_sql = df
        if any(not isinstance(c, str) for c in df.columns):
            df_for_sql = df.copy()
            new_cols = []
            for c in df.columns:
                if isinstance(c, int):
                    str_col = str(c)
                    int_col_map[str_col] = c
                    new_cols.append(str_col)
                elif isinstance(c, str):
                    new_cols.append(c)
                else:
                    # Other non-string types - convert to string
                    str_col = str(c)
                    int_col_map[str_col] = c
                    new_cols.append(str_col)
            df_for_sql.columns = new_cols
            # Update SQL to use string column names
            for str_col, orig_col in int_col_map.items():
                # Replace numeric column references in SQL with quoted strings
                # e.g., "0" stays as "0", but any unquoted references need quoting
                sql = sql.replace(f'"{orig_col}"', f'"{str_col}"')

        # Row order preservation using chDB's built-in _row_id virtual column
        # _row_id provides the 0-based row number from the original DataFrame
        # No need to manually add an index column - _row_id is deterministic
        #
        # IMPORTANT: chDB's Python() table function reads DataFrame data using the
        # pandas index values as row positions. For DataFrames with non-contiguous
        # indices (e.g., after slicing with step), we must reset the index to ensure
        # chdb reads the correct data. The original index is stored and restored after.
        # Handle non-contiguous index for chDB compatibility
        df_to_use, original_index, original_index_name = self._prepare_df_for_chdb(df_for_sql)

        sql_upper = sql.upper().strip()

        # Determine how to handle row order preservation
        # _row_id is used to preserve original row order in Python() table function
        has_order_by = self._has_outer_order_by(sql_upper)
        has_group_by = 'GROUP BY' in sql_upper
        is_aggregate = self._is_aggregate_query(sql_upper)

        # Determine row order handling strategy using _row_id
        add_order_by_row_id = False  # Whether to add ORDER BY _row_id (no existing ORDER BY)
        add_row_id_as_tiebreaker = False  # Whether to append _row_id to existing ORDER BY
        need_row_id_in_result = False  # Whether we need _row_id in result for index restoration

        if not preserve_order:
            pass  # All False
        elif has_group_by or is_aggregate:
            # GROUP BY/aggregate: rows don't correspond to original rows
            # _row_id is meaningless after aggregation
            pass  # All False
        elif has_order_by:
            # Has explicit ORDER BY: add _row_id as tie-breaker for stable sort
            # This ensures rows with equal sort keys maintain original relative order
            add_row_id_as_tiebreaker = True
            need_row_id_in_result = True
        else:
            # No explicit ORDER BY and no GROUP BY/aggregate: preserve original row order
            # This applies to all queries including column selection (SELECT col1, col2)
            # chDB doesn't guarantee row order, so we must add _row_id ordering
            add_order_by_row_id = True
            need_row_id_in_result = True

        # Auto-wrap table reference if not already wrapped
        processed_sql = sql
        if f'Python({df_name})' not in sql:
            processed_sql = sql.replace(df_name, f'Python({df_name})')

        # Replace rowNumberInAllBlocks() with _row_id for Python() table function
        # rowNumberInAllBlocks() is non-deterministic with Python() table function
        # but _row_id is a built-in deterministic virtual column in chDB v4.0.0b5+
        processed_sql = processed_sql.replace('rowNumberInAllBlocks()', '_row_id')

        # Handle ORDER BY modifications using _row_id
        if add_order_by_row_id:
            # No existing ORDER BY: add ORDER BY _row_id to preserve row order
            processed_sql = self._add_order_by_row_id(processed_sql)
        elif add_row_id_as_tiebreaker:
            # Has existing ORDER BY: append _row_id as tie-breaker for stable sort
            processed_sql = self._append_row_id_tiebreaker(processed_sql)

        # Add _row_id to SELECT if we need it for index restoration
        if need_row_id_in_result:
            processed_sql = self._add_row_id_to_select(processed_sql)

        self._log_query(processed_sql, "DataFrame")

        try:
            start_time = time.perf_counter()
            # Execute with DataFrame in local scope
            result = self._execute_df_query(processed_sql, df_to_use, df_name)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Restore original pandas index from _row_id (row positions)
            # Map row positions back to original index values
            # Also handle auto-renamed columns like _row_id_1, _row_id_2 from nested subqueries
            row_id_cols = [
                c for c in result.columns if c == '_row_id' or (isinstance(c, str) and c.startswith('_row_id_'))
            ]
            if need_row_id_in_result and '_row_id' in result.columns:
                # _row_id contains row positions (0, 1, 2, ...)
                # Look up original index values from these positions
                row_positions = result['_row_id'].values
                result = result.drop(columns=row_id_cols)

                # If we reset the index earlier, use original_index for restoration
                if original_index is not None:
                    result.index = original_index[row_positions]
                    result.index.name = original_index_name
                else:
                    result.index = df.index[row_positions]
                    result.index.name = df.index.name  # Preserve original index name
            elif row_id_cols:
                # Remove any _row_id columns that weren't used for index restoration
                result = result.drop(columns=row_id_cols)
                # Restore original index if we reset it earlier
                if original_index is not None and len(result) == len(original_index):
                    result.index = original_index
                    result.index.name = original_index_name

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

            # Restore original integer column names if we converted them
            if int_col_map:
                new_cols = []
                for c in result.columns:
                    if c in int_col_map:
                        new_cols.append(int_col_map[c])
                    else:
                        new_cols.append(c)
                result.columns = new_cols

            return result
        except Exception as e:
            self._logger.error("[chDB] DataFrame query failed: %s", e)
            raise ExecutionError(f"Failed to execute SQL on DataFrame: {e}")

    def _prepare_df_for_chdb(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Index], Optional[str]]:
        """
        Prepare DataFrame for chDB execution by handling non-contiguous index.

        chDB's Python() table function reads DataFrame data using pandas index
        values as row positions. For DataFrames with non-contiguous indices
        (e.g., after slicing with step like df[::2]), we must reset the index
        to ensure chDB reads the correct data.

        Args:
            df: DataFrame to prepare

        Returns:
            Tuple of (df_to_use, original_index, original_index_name):
            - df_to_use: DataFrame with contiguous 0..n-1 index (or original if already contiguous)
            - original_index: Original index if reset was needed, None otherwise
            - original_index_name: Original index name if reset was needed, None otherwise
        """
        if len(df) == 0:
            return df, None, None

        # Check if index is contiguous (0, 1, 2, ..., n-1)
        # pd.RangeIndex.equals() correctly handles value comparison even for
        # non-RangeIndex types (e.g., Int64Index with values 0,1,2 equals RangeIndex(3))
        if df.index.equals(pd.RangeIndex(len(df))):
            return df, None, None

        # Non-contiguous index - need to reset for chDB compatibility
        original_index = df.index.copy()
        original_index_name = df.index.name
        df_to_use = df.reset_index(drop=True)
        return df_to_use, original_index, original_index_name

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
        # Common aggregate functions (standard SQL and ClickHouse-specific)
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
            # ClickHouse-specific aggregate functions
            'TOPK(',
            'TOPKWEIGHTED(',
            'GROUPARRAY(',
            'GROUPARRAYINSERTAT(',
            'GROUPUNIQARRAY(',
            'GROUPBITAND(',
            'GROUPBITOR(',
            'GROUPBITXOR(',
            'ARGMIN(',
            'ARGMAX(',
            'FIRST_VALUE(',
            'LAST_VALUE(',
            'ENTROPY(',
            'SIMPLELINEARREGRESSION(',
            'STOCHASTICLINEARREGRESSION(',
        ]

        # Check if any aggregate function is present
        has_aggregate = any(func in sql_upper for func in aggregate_functions)

        # It's an aggregate query if it has aggregate functions but no GROUP BY
        # (GROUP BY is already handled separately)
        return has_aggregate

    def _add_order_by(self, sql: str, order_col: str) -> str:
        """
        Add ORDER BY clause to preserve row order.

        This method modifies the SQL to:
        1. Add the order column to the SELECT clause (if not already present)
        2. Add ORDER BY clause at the appropriate position

        For queries with LIMIT/OFFSET at the outer level, ORDER BY must be applied
        BEFORE them.
        """
        sql_upper = sql.upper()

        # Check if the order column is already in the SELECT clause
        # or if it's a SELECT * query (which includes all columns)
        is_select_star = self._is_select_star_query(sql_upper)
        has_order_col = (
            f'"{order_col.upper()}"' in sql_upper or f'"{order_col}"' in sql or order_col.upper() in sql_upper
        )

        # If not SELECT * and order_col not in select, we need to add it
        modified_sql = sql
        if not is_select_star and not has_order_col:
            # Find the outer FROM clause (not inside quotes or subqueries)
            from_pos = self._find_outer_from_clause(sql)
            if from_pos != -1:
                # Insert order_col before FROM
                modified_sql = sql[:from_pos] + f', "{order_col}" ' + sql[from_pos:]
                sql_upper = modified_sql.upper()

        # Check if LIMIT/OFFSET exists at the OUTER level (not inside a subquery)
        limit_pos = self._find_outer_clause_position(sql_upper, 'LIMIT')
        offset_pos = self._find_outer_clause_position(sql_upper, 'OFFSET')

        if limit_pos != -1 or offset_pos != -1:
            # LIMIT/OFFSET at outer level - insert ORDER BY before it
            clause_start = len(modified_sql)
            if limit_pos != -1:
                clause_start = min(clause_start, limit_pos)
            if offset_pos != -1:
                clause_start = min(clause_start, offset_pos)

            base_query = modified_sql[:clause_start].strip()
            limit_offset_clause = modified_sql[clause_start:].strip()
            return f"{base_query} ORDER BY \"{order_col}\" ASC {limit_offset_clause}"
        else:
            # No LIMIT/OFFSET at outer level: add ORDER BY at the end
            return f"{modified_sql} ORDER BY \"{order_col}\" ASC"

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

    def _find_outer_from_clause(self, sql: str) -> int:
        """
        Find the position of the outer FROM clause in a SQL query.

        This method correctly handles:
        - Column names that contain SQL keywords (e.g., "from", "select")
        - Quoted identifiers
        - Subqueries

        Returns:
            Position of FROM in the SQL string, or -1 if not found.
        """
        sql_upper = sql.upper()
        pos = 0

        while pos < len(sql_upper):
            # Find the next occurrence of FROM
            from_pos = sql_upper.find('FROM', pos)
            if from_pos == -1:
                return -1

            # Check if FROM is at the start (no character before it)
            # or preceded by a space/newline/tab (not inside a quoted identifier)
            if from_pos > 0:
                char_before = sql[from_pos - 1]
                if char_before not in ' \t\n\r,(':
                    # FROM is part of a word/identifier, skip it
                    pos = from_pos + 4
                    continue

            # Check if FROM is followed by a space or is at the end
            if from_pos + 4 < len(sql):
                char_after = sql[from_pos + 4]
                if char_after not in ' \t\n\r':
                    # FROM is part of a word/identifier, skip it
                    pos = from_pos + 4
                    continue

            # Check if we're inside quotes by counting quotes before this position
            prefix = sql[:from_pos]
            # Count unescaped double quotes
            double_quotes = prefix.count('"')
            single_quotes = prefix.count("'")

            # If odd number of quotes, we're inside a quoted string
            if double_quotes % 2 == 1 or single_quotes % 2 == 1:
                pos = from_pos + 4
                continue

            # Check if we're inside a subquery by counting parentheses
            open_parens = prefix.count('(')
            close_parens = prefix.count(')')

            # If parentheses are balanced, FROM is at the outer level
            if open_parens == close_parens:
                return from_pos

            # FROM is inside a subquery, continue searching
            pos = from_pos + 4

        return -1

    def _append_tiebreaker_to_orderby(self, sql: str, tiebreaker_col: str) -> str:
        """
        Append a tie-breaker column to existing ORDER BY clause for stable sort.

        Append a tie-breaker column to existing ORDER BY clause for stable sort.

        When there's an explicit ORDER BY, we append the tie-breaker column as a secondary
        sort key to maintain original relative order for rows with equal sort keys.
        This implements stable sort behavior like pandas sort_values(kind='stable').

        Example:
            Input:  SELECT * FROM t ORDER BY name DESC
            Output: SELECT * FROM t ORDER BY name DESC, __tiebreaker__ ASC
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

    def _add_order_by_row_id(self, sql: str) -> str:
        """
        Add ORDER BY _row_id clause to preserve original row order.

        Uses chDB's built-in _row_id virtual column which provides the 0-based
        row number from the original DataFrame.

        For queries with LIMIT/OFFSET at the outer level, ORDER BY must be applied
        BEFORE them.

        Example:
            Input:  SELECT id, value FROM Python(df) WHERE value > 10
            Output: SELECT id, value FROM Python(df) WHERE value > 10 ORDER BY _row_id
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
            return f"{base_query} ORDER BY _row_id {limit_offset_clause}"
        else:
            # No LIMIT/OFFSET at outer level: add ORDER BY at the end
            return f"{sql} ORDER BY _row_id"

    def _append_row_id_tiebreaker(self, sql: str) -> str:
        """
        Append _row_id as tie-breaker to existing ORDER BY clause for stable sort.

        When there's an explicit ORDER BY, we append _row_id as a secondary
        sort key to maintain original relative order for rows with equal sort keys.
        This implements stable sort behavior like pandas sort_values(kind='stable').

        Example:
            Input:  SELECT * FROM Python(df) ORDER BY name DESC
            Output: SELECT * FROM Python(df) ORDER BY name DESC, _row_id ASC
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

        # Insert _row_id tie-breaker before the end of ORDER BY clause
        return sql[:order_by_end] + ', _row_id ASC' + sql[order_by_end:]

    def _add_row_id_to_select(self, sql: str) -> str:
        """
        Add _row_id to SELECT clauses for index restoration after query.

        For queries with subqueries, _row_id must be added to ALL SELECT clauses
        because _row_id is a virtual column that is NOT included in SELECT *.
        The innermost query (with Python()) needs _row_id explicitly selected,
        and each outer query needs it too for propagation.

        Example:
            Input:  SELECT id, value FROM Python(df)
            Output: SELECT id, value, _row_id FROM Python(df)

            Input:  SELECT * FROM Python(df)
            Output: SELECT *, _row_id FROM Python(df)

            Input:  SELECT * FROM (SELECT * FROM Python(df)) AS sub
            Output: SELECT *, _row_id FROM (SELECT *, _row_id FROM Python(df)) AS sub
        """
        # For queries with subqueries, we need to add _row_id to ALL SELECT clauses
        # because _row_id is a virtual column not included in SELECT *
        return self._add_row_id_to_all_selects(sql)

    def _add_row_id_to_all_selects(self, sql: str) -> str:
        """
        Add _row_id to all SELECT clauses in the SQL.

        This is necessary because _row_id is a virtual column that's only
        available on Python() table function, and SELECT * doesn't include it.
        We need to explicitly select it at every level for it to be available
        in outer queries.
        """
        # Find all FROM positions that are at valid SQL boundaries (not inside quoted strings)
        # and add _row_id before each one
        result = []
        i = 0
        sql_len = len(sql)

        while i < sql_len:
            # Track if we're inside a quoted string
            if sql[i] == '"':
                # Find closing quote
                end = sql.find('"', i + 1)
                if end == -1:
                    result.append(sql[i:])
                    break
                result.append(sql[i : end + 1])
                i = end + 1
                continue
            elif sql[i] == "'":
                # Find closing quote
                end = sql.find("'", i + 1)
                if end == -1:
                    result.append(sql[i:])
                    break
                result.append(sql[i : end + 1])
                i = end + 1
                continue

            # Check if we're at a FROM keyword (case-insensitive)
            upper_remaining = sql[i:].upper()
            if upper_remaining.startswith('FROM') and (i == 0 or not sql[i - 1].isalnum() and sql[i - 1] != '_'):
                # Check if FROM is followed by a non-alphanumeric character (word boundary)
                after_from = i + 4
                if after_from >= sql_len or (not sql[after_from].isalnum() and sql[after_from] != '_'):
                    # This is a valid FROM keyword
                    # Check the preceding content for _row_id
                    preceding = ''.join(result)

                    # Find the last SELECT before this FROM
                    select_pos = preceding.upper().rfind('SELECT')
                    if select_pos != -1:
                        # Check if _row_id is already in this SELECT clause
                        select_clause = preceding[select_pos:]
                        if '_row_id' not in select_clause.lower():
                            # Add _row_id before FROM
                            # Remove trailing whitespace and add _row_id
                            while result and result[-1].isspace():
                                result.pop()
                            result.append(', _row_id ')

                    result.append(sql[i : i + 4])  # Add FROM
                    i = i + 4
                    continue

            result.append(sql[i])
            i += 1

        return ''.join(result)

    def _execute_df_query(self, sql: str, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """
        Internal: execute SQL with DataFrame in local scope.

        chDB's Python() table function requires the DataFrame to be
        accessible in the local scope where conn.query() is called.
        """
        df_converted = _convert_nullable_dtypes(df)
        __df__ = df_converted  # noqa: F841 - Required for conn.query to access via Python(__df__)
        if df_name != '__df__':
            exec(f"{df_name} = df_converted")
        return self._conn.query(sql, 'DataFrame')

    def eval_expression(self, expr_sql: str, df: pd.DataFrame, result_column: str = '__result__') -> pd.Series:
        """
        Evaluate a SQL expression on a DataFrame and return the result as a Series.

        Useful for column assignments like: ds['new_col'] = ds['value'].cast('Float64')

        IMPORTANT: This method preserves row order using chDB's built-in _row_id
        virtual column (available in chDB v4.0.0b5+) to ensure results align with
        the original DataFrame.
        For aggregate expressions, ORDER BY is skipped as they return single values.
        For row-expanding expressions (arrayJoin/explode), the result may have more
        rows than the input and index alignment is skipped.

        Args:
            expr_sql: SQL expression to evaluate (e.g., "CAST(value AS Float64)")
            df: DataFrame to operate on
            result_column: Name for the result column (default: '__result__')

        Returns:
            Result Series with the original DataFrame's index (when possible)
        """
        if self._conn is None:
            raise ConnectionError("Not connected. Call connect() first.")

        # Check if expression contains aggregate functions
        is_aggregate = self._is_aggregate_expression(expr_sql)

        # Check if expression contains row-expanding functions (arrayJoin/explode)
        # These functions can change the number of rows, so we can't preserve index
        is_row_expanding = self._is_row_expanding_expression(expr_sql)

        df_converted = _convert_nullable_dtypes(df)

        # Handle non-contiguous index for chDB compatibility
        # Skip for aggregate expressions as they don't need index restoration
        if is_aggregate:
            df_to_use, original_index, original_index_name = df_converted, None, None
        else:
            df_to_use, original_index, original_index_name = self._prepare_df_for_chdb(df_converted)

        if is_aggregate:
            # Aggregate expressions return single value, no ORDER BY needed
            __df__ = df_to_use  # noqa: F841
            query = f"SELECT {expr_sql} AS {result_column} FROM Python(__df__)"
        else:
            # Use _row_id to preserve order for row-level expressions
            # _row_id is a built-in virtual column in chDB that provides the 0-based
            # row number from the original DataFrame
            # IMPORTANT: Include _row_id in SELECT and use it to restore index
            # (don't assume ORDER BY _row_id returns exact 0,1,2... order due to
            # potential parallel execution in chDB)
            __df__ = df_to_use  # noqa: F841
            # Replace rowNumberInAllBlocks() with _row_id for window function ordering
            # This ensures window functions use the original DataFrame row order
            expr_sql_fixed = expr_sql.replace('rowNumberInAllBlocks()', '_row_id')
            query = f"SELECT {expr_sql_fixed} AS {result_column}, _row_id FROM Python(__df__) ORDER BY _row_id"

        self._log_query(query, "Expression")

        try:
            start_time = time.perf_counter()
            result_df = self._conn.query(query, 'DataFrame')
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            result_series = result_df[result_column]

            # Restore index using _row_id values from result
            # This correctly handles cases where parallel execution in chDB
            # causes non-deterministic _row_id assignment
            if not is_aggregate and not is_row_expanding and len(result_series) == len(df):
                if '_row_id' in result_df.columns:
                    row_positions = result_df['_row_id'].values
                    if original_index is not None:
                        # Restore original non-contiguous index
                        result_series.index = original_index[row_positions]
                        result_series.index.name = original_index_name
                    else:
                        # Use row positions to look up original index values
                        result_series.index = df.index[row_positions]
                        result_series.index.name = df.index.name
                else:
                    # Fallback: direct assignment (for aggregate or older chDB)
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

    def _is_row_expanding_expression(self, expr_sql: str) -> bool:
        """
        Check if SQL expression contains row-expanding functions.

        Row-expanding functions like arrayJoin can produce more rows than the input,
        so we cannot align the result index with the original DataFrame.

        Args:
            expr_sql: SQL expression to check

        Returns:
            True if expression contains row-expanding functions
        """
        import re

        # Row-expanding function patterns (case-insensitive)
        row_expanding_patterns = [
            r'\barrayJoin\s*\(',
        ]

        expr_lower = expr_sql.lower()
        for pattern in row_expanding_patterns:
            if re.search(pattern, expr_lower, re.IGNORECASE):
                return True
        return False

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
        # Use local references to avoid race conditions during GC
        cursor = self._cursor
        conn = self._conn

        self._cursor = None
        self._conn = None

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass

        if conn:
            try:
                conn.close()
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        # Avoid calling close() during GC if already closed
        # This prevents issues when GC runs during chdb internal operations
        # (e.g., when pandas validates hashable types during DataFrame creation)
        try:
            self.close()
        except Exception:
            pass


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
        """Get all rows as list of tuples (lazy conversion from DataFrame/Series)."""
        if self._rows is None and self._data is not None:
            try:
                if isinstance(self._data, pd.DataFrame):
                    self._rows = [tuple(row) for row in self._data.itertuples(index=False, name=None)]
                elif isinstance(self._data, pd.Series):
                    # Series: each value becomes a single-element tuple
                    self._rows = [(v,) for v in self._data.values]
                else:
                    # Fallback for other formats
                    self._rows = []
            except Exception:
                self._rows = []
        return self._rows if self._rows is not None else []

    @property
    def column_names(self) -> List[str]:
        """Get column names (lazy extraction from DataFrame/Series)."""
        if self._column_names is None and self._data is not None:
            try:
                if isinstance(self._data, pd.DataFrame):
                    self._column_names = list(self._data.columns)
                elif isinstance(self._data, pd.Series):
                    # Series has a single column (its name or default)
                    self._column_names = [self._data.name or 'value']
                else:
                    self._column_names = []
            except Exception:
                self._column_names = []
        return self._column_names if self._column_names is not None else []

    @property
    def column_types(self) -> List[str]:
        """Get column types (lazy extraction from DataFrame/Series)."""
        if self._column_types is None and self._data is not None:
            try:
                if isinstance(self._data, pd.DataFrame):
                    self._column_types = [str(dtype) for dtype in self._data.dtypes]
                elif isinstance(self._data, pd.Series):
                    self._column_types = [str(self._data.dtype)]
                else:
                    self._column_types = []
            except Exception:
                self._column_types = []
        return self._column_types if self._column_types is not None else []

    @property
    def row_count(self) -> int:
        """Get number of rows (lazy calculation from DataFrame/Series)."""
        if self._row_count is None and self._data is not None:
            try:
                if isinstance(self._data, (pd.DataFrame, pd.Series)):
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

    def to_dict(self, orient: str = 'dict', *, into=dict, index: bool = True):
        """
        Convert results to a dictionary.

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
            Determines the type of the values of the dictionary.
            See pandas DataFrame.to_dict() for details.

        into : class, default dict
            The collections.abc.MutableMapping subclass used for all Mappings.

        index : bool, default True
            Whether to include the index item.

        Returns:
            dict, list or collections.abc.MutableMapping
        """
        try:
            if isinstance(self._data, pd.DataFrame):
                return self._data.to_dict(orient=orient, into=into, index=index)
            elif isinstance(self._data, pd.Series):
                return self._data.to_dict(into=into)
        except Exception:
            pass

        # Fallback to legacy method - only supports records format
        if orient != 'records':
            raise ValueError(f"orient='{orient}' not supported in fallback mode, only 'records' is supported")
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
        elif isinstance(self._data, pd.Series):
            return self._data.to_frame()
        else:
            # Convert rows to DataFrame
            if self.rows:
                return pd.DataFrame(self.rows, columns=self.column_names)
            else:
                return pd.DataFrame()

    def to_series(self):
        """
        Return the result as a pandas Series.

        Returns:
            pandas Series
        """
        if isinstance(self._data, pd.Series):
            return self._data
        elif isinstance(self._data, pd.DataFrame):
            if len(self._data.columns) == 1:
                return self._data.iloc[:, 0]
            else:
                raise ValueError("Cannot convert multi-column DataFrame to Series")
        else:
            # Convert rows to Series
            if self.rows:
                values = [row[0] for row in self.rows]
                name = self.column_names[0] if self.column_names else None
                return pd.Series(values, name=name)
            else:
                return pd.Series()

    def __iter__(self):
        """Iterate over rows."""
        return iter(self.rows)

    def __len__(self):
        """Number of rows."""
        return self.row_count

    def __repr__(self):
        """Return pandas-like string representation."""
        if isinstance(self._data, (pd.DataFrame, pd.Series)):
            # Delegate to pandas repr (respects display options, truncates automatically)
            return repr(self._data)
        else:
            return f"QueryResult(rows={self.row_count}, columns={len(self.column_names)})"

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        if isinstance(self._data, (pd.DataFrame, pd.Series)):
            # Delegate to pandas HTML repr
            return self._data._repr_html_()
        else:
            return f"<p>QueryResult(rows={self.row_count}, columns={len(self.column_names)})</p>"
