"""
Core DataStore class - main entry point for data operations
"""

import time
import pandas as pd
from typing import Any, Optional, List, Dict, Union, TYPE_CHECKING
from copy import copy

if TYPE_CHECKING:
    from .column_expr import ColumnExpr
    from .groupby import LazyGroupBy
    from .case_when import CaseWhenBuilder
    from .query_planner import ExecutionSegment, QueryPlan

from .expressions import Field, Expression, Literal, Star
from .conditions import Condition
from .utils import immutable, ignore_copy, format_identifier, normalize_ascending, map_agg_func
from .exceptions import QueryError, ConnectionError, ExecutionError
from .connection import Connection, QueryResult
from .executor import Executor
from .table_functions import create_table_function, TableFunction
from .uri_parser import parse_uri
from .pandas_compat import PandasCompatMixin
from .lazy_ops import LazyOp, LazyRelationalOp
from .sql_executor import (
    ExtractedClauses,
    extract_clauses_from_ops,
    apply_alias_renames_to_orderby,
    build_groupby_select_fields,
    SQLExecutionEngine,
    SQLBuildResult,
)
from .config import (
    get_logger,
    config as _global_config,
    DataStoreConfig,
    is_cache_enabled,
    get_cache_ttl,
    get_profiler,
    is_profiling_enabled,
)

__all__ = ['DataStore']

# Sentinel value to distinguish "no argument passed" from "None passed explicitly"
_MISSING = object()


class DataStore(PandasCompatMixin):
    """
    DataStore - Pandas-like data manipulation with SQL generation.

    Example:
        >>> ds = DataStore("file", path="data.parquet")
        >>> ds.connect()
        >>> result = ds.select("name", "age").filter(ds.age > 18).execute()

    Logging Configuration:
        >>> import logging
        >>> DataStore.config.log_level = logging.DEBUG  # Enable debug logging
        >>> DataStore.config.enable_debug()  # Or use convenience method
    """

    # Class-level config for all DataStore instances
    config: DataStoreConfig = _global_config

    def __init__(
        self,
        source: Union[str, pd.DataFrame] = None,
        table: str = None,
        database: str = ":memory:",
        connection: Connection = None,
        **kwargs,
    ):
        """
        Initialize DataStore.

        Args:
            source: Data source - can be:
                   - pandas DataFrame: wrap directly for manipulation
                   - str: source type ('file', 's3', 'mysql', 'clickhouse', etc.)
                   - None: create empty DataStore
            table: Table name (for regular tables or remote ClickHouse)
            database: Database path (":memory:" for in-memory, or file path)
            connection: Existing Connection object (creates new if None)
            **kwargs: Additional parameters (path, url, format, host, etc.)

        Examples:
            >>> # From pandas DataFrame (most convenient)
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> ds = DataStore(df)

            >>> # Local file (auto-detect format)
            >>> ds = DataStore("file", path="data.parquet")

            >>> # Local file (explicit format)
            >>> ds = DataStore("file", path="data.csv", format="CSV")

            >>> # S3 data (auto-detect format)
            >>> ds = DataStore("s3", url="s3://bucket/data.parquet", nosign=True)

            >>> # S3 data (with credentials and explicit format)
            >>> ds = DataStore("s3", url="s3://bucket/data.parquet",
            ...                access_key_id="KEY", secret_access_key="SECRET",
            ...                format="Parquet")

            >>> # MySQL database
            >>> ds = DataStore("mysql", host="localhost:3306",
            ...                database="mydb", table="users",
            ...                user="root", password="pass")

            >>> # Regular ClickHouse table (no table function)
            >>> ds = DataStore(table="my_table")
        """
        # Handle backward compatibility: source_type keyword argument
        if 'source_type' in kwargs:
            if source is None:
                source = kwargs.pop('source_type')
            else:
                # If both provided, just remove from kwargs to avoid passing to connection
                kwargs.pop('source_type')

        # Handle DataFrame input directly
        if isinstance(source, pd.DataFrame):
            # Initialize with DataFrame - delegate to _init_from_dataframe
            self._init_from_dataframe(source, database, connection, **kwargs)
            return

        self.source_type = source or 'chdb'
        self.table_name = table
        self.database = database
        self.connection_params = kwargs

        # Table function support
        self._table_function: Optional[TableFunction] = None
        self._format_settings: Dict[str, Any] = {}

        # Source DataFrame (for from_df() - enables on-demand PythonTableFunction creation)
        self._source_df = None
        self._source_df_name: Optional[str] = None

        # Create table function if source is specified
        if source and source.lower() != 'chdb':
            try:
                # For database sources with explicit table, pass table name
                if table and source.lower() in [
                    'clickhouse',
                    'remote',
                    'mysql',
                    'postgresql',
                    'postgres',
                    'mongodb',
                    'mongo',
                    'sqlite',
                ]:
                    kwargs['table'] = table

                # For database sources, also pass database if provided
                if (
                    database
                    and database != ":memory:"
                    and source.lower()
                    in ['clickhouse', 'remote', 'mysql', 'postgresql', 'postgres', 'mongodb', 'mongo', 'sqlite']
                ):
                    kwargs['database'] = database

                self._table_function = create_table_function(source, **kwargs)
            except Exception:
                # If table function creation fails, it might be a regular table
                # We'll treat it as a regular table and table_function remains None
                pass

        # Query state
        self._select_fields: List[Expression] = []
        self._select_star: bool = False  # True when SELECT * is used (possibly with additional computed columns)
        self._where_condition: Optional[Condition] = None
        self._joins: List[tuple] = []  # [(table/datastore, join_type, on_condition), ...]
        self._groupby_fields: List[Expression] = []
        self._having_condition: Optional[Condition] = None
        self._orderby_fields: List[tuple] = []  # [(field, ascending), ...]
        self._orderby_kind: str = 'quicksort'  # Sort algorithm (matches pandas default)
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._distinct: bool = False

        # INSERT/UPDATE/DELETE state
        self._insert_columns: List[str] = []
        self._insert_values: List[List[Any]] = []
        self._insert_select: Optional['DataStore'] = None
        self._update_fields: List[tuple] = []  # [(field, value), ...]
        self._delete_flag: bool = False

        # Subquery support
        self._alias: Optional[str] = None
        self._is_subquery: bool = False

        # Connection and execution
        self._connection: Optional[Connection] = connection
        self._executor: Optional[Executor] = None
        self._schema: Optional[Dict[str, str]] = None

        # Configuration
        self.quote_char = '"'

        # Lazy execution support
        self._lazy_ops: List[LazyOp] = []  # Lazy operation chain

        # Generate unique variable name for chDB Python() table function
        # This ensures thread-safety and concurrent execution support
        import uuid

        self._df_var_name: str = f"__ds_df_{uuid.uuid4().hex}__"

        # Operation tracking for explain()
        self._operation_history: List[Dict[str, Any]] = []
        self._original_source_desc: Optional[str] = None  # Preserve original data source for explain()

        # Logger instance
        self._logger = get_logger()

        # Cache state for executed results
        # This implements intelligent automatic caching to avoid re-execution
        # when repr/__str__ are called multiple times
        self._cached_result: Optional[pd.DataFrame] = None
        self._cache_version: int = 0  # Incremented when operations are added
        self._cached_at_version: int = -1  # Version when cache was created
        self._cache_timestamp: Optional[float] = None  # For TTL support

    def _init_from_dataframe(
        self,
        df: pd.DataFrame,
        database: str = ":memory:",
        connection: Connection = None,
        **kwargs,
    ):
        """
        Initialize DataStore from a pandas DataFrame.

        This is called internally when DataStore(df) is used.
        """
        from .lazy_ops import LazyDataFrameSource

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        # Basic setup
        self.source_type = 'dataframe'
        self.table_name = None
        self.database = database
        self.connection_params = kwargs

        # Table function support
        self._table_function: Optional[TableFunction] = None
        self._format_settings: Dict[str, Any] = {}

        # Source DataFrame (for from_df() - enables on-demand PythonTableFunction creation)
        self._source_df = None
        self._source_df_name: Optional[str] = None

        # Query state - all cleared for DataFrame source
        self._select_fields: List[Expression] = []
        self._select_star: bool = False
        self._where_condition: Optional[Condition] = None
        self._joins: List[tuple] = []
        self._groupby_fields: List[Expression] = []
        self._having_condition: Optional[Condition] = None
        self._orderby_fields: List[tuple] = []
        self._orderby_kind: str = 'quicksort'
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._distinct: bool = False

        # INSERT/UPDATE/DELETE state
        self._insert_columns: List[str] = []
        self._insert_values: List[List[Any]] = []
        self._insert_select: Optional['DataStore'] = None
        self._update_fields: List[tuple] = []
        self._delete_flag: bool = False

        # Subquery support
        self._alias: Optional[str] = None
        self._is_subquery: bool = False

        # Connection and execution
        self._connection: Optional[Connection] = connection
        self._executor: Optional[Executor] = None

        # Build schema from DataFrame dtypes
        self._schema: Optional[Dict[str, str]] = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Configuration
        self.quote_char = '"'

        # Add the DataFrame as a lazy source
        self._lazy_ops: List[LazyOp] = [LazyDataFrameSource(df)]

        # Generate unique variable name
        import uuid

        self._df_var_name: str = f"__ds_df_{uuid.uuid4().hex}__"

        # Set source description for explain()
        name = kwargs.get('name')
        shape_str = f"{df.shape[0]} rows x {df.shape[1]} cols"
        self._original_source_desc: Optional[str] = f"DataFrame({name or 'unnamed'}, {shape_str})"

        # Operation tracking
        self._operation_history: List[Dict[str, Any]] = []

        # Logger instance
        self._logger = get_logger()

        # Cache state
        self._cached_result: Optional[pd.DataFrame] = None
        self._cache_version: int = 0
        self._cached_at_version: int = -1
        self._cache_timestamp: Optional[float] = None

    # ========== Operation Tracking for explain() ==========

    def _track_operation(self, op_type: str, description: str, details: Dict[str, Any] = None):
        """
        Track an operation for explain() output.

        Args:
            op_type: Type of operation ('sql', 'pandas', 'execute')
            description: Human-readable description
            details: Additional details about the operation
        """
        operation = {
            'type': op_type,
            'description': description,
            'details': details or {},
            'is_on_dataframe': op_type == 'pandas',
            'executed_at_call': False,
        }
        self._operation_history.append(operation)

    def _get_data_source_description(self):
        """Get a description of the data source."""
        # Return cached description if available
        if hasattr(self, '_original_source_desc') and self._original_source_desc:
            return self._original_source_desc

        # Generate and cache description
        desc = None
        if self._table_function:
            # Show table function
            sql = self._table_function.to_sql(quote_char=self.quote_char)
            if len(sql) > 100:
                sql = sql[:97] + "..."
            desc = f"Data Source: {sql}"
        elif self.table_name:
            # Show table name
            desc = f"Data Source: Table '{self.table_name}'"

        # Cache for future use (survives execution)
        if desc and (not hasattr(self, '_original_source_desc') or not self._original_source_desc):
            self._original_source_desc = desc

        return desc

    def _analyze_execution_phases(self):
        """Analyze operation history and group operations into execution phases."""
        if not self._operation_history:
            return [], None, []

        # Find the execution point
        mat_idx = next((i for i, op in enumerate(self._operation_history) if op['type'] == 'execute'), None)

        if mat_idx is not None:
            # Explicit execution operation present
            return (
                self._operation_history[:mat_idx],
                self._operation_history[mat_idx],
                self._operation_history[mat_idx + 1 :],
            )

        # No explicit execution - split by is_on_dataframe flag
        lazy = [op for op in self._operation_history if not op.get('is_on_dataframe', False)]
        executed = [op for op in self._operation_history if op.get('is_on_dataframe', False)]

        # If there are only executed ops, the first one becomes the implicit execution point
        if executed and not lazy and executed[0]['type'] in ['pandas', 'execute']:
            return [], executed[0], executed[1:]

        # If both exist, the first executed operation is the implicit execution point
        if lazy and executed:
            return lazy, executed[0], executed[1:]

        return lazy, None, executed

    def _render_operations(self, operations, start_num=1, verbose=False):
        """Render a list of operations."""
        lines = []
        for i, op in enumerate(operations, start_num):
            icon = {'sql': '🔍', 'pandas': '🐼', 'execute': '🔄'}.get(op['type'], '📝')
            desc = (
                f"SQL on DataFrame: {op['description']}"
                if op['type'] == 'sql' and op.get('is_on_dataframe')
                else f"{op['type'].upper()}: {op['description']}"
            )
            lines.append(f" [{i}] {icon} {desc}")
            if verbose and op.get('details'):
                for k, v in op['details'].items():
                    lines.append(f"     └─ {k}: {v}")
        return lines

    def explain(self, verbose: bool = False) -> str:
        """
        Generate and display the execution plan in original operation order.

        This method shows operations in the exact order they were defined,
        which is critical because order affects execution results.

        Args:
            verbose: If True, show additional details like full SQL queries

        Returns:
            String representation of the execution plan

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> ds = ds.select('name', 'age').filter(ds.age > 25)
            >>> ds['computed'] = ds['age'] * 2
            >>> ds = ds.filter(ds['age'] < 50)  # Order matters!
            >>> print(ds.explain())
        """
        # Ensure data source description is cached before analysis
        if not hasattr(self, '_original_source_desc') or not self._original_source_desc:
            self._get_data_source_description()

        lines = []
        lines.append("=" * 80)
        lines.append("Execution Plan (in execution order)")
        lines.append("=" * 80)

        counter = 0

        # ========== Data Source ==========
        data_source_desc = self._get_data_source_description()
        if data_source_desc:
            counter += 1
            lines.append(f"\n [{counter}] 📊 {data_source_desc}")

        # ========== Operations in Original Order ==========
        # _lazy_ops contains ALL operations (SQL snapshots + lazy ops) in order
        if self._lazy_ops:
            # Find the first non-SQL operation (same logic as _execute)
            first_df_op_idx = None
            for i, op in enumerate(self._lazy_ops):
                if not isinstance(op, LazyRelationalOp):
                    first_df_op_idx = i
                    break

            lines.append("\nOperations:")
            lines.append("─" * 80)

            # Show execution engine info
            if first_df_op_idx is not None:
                lines.append(f"    ️  Phase 1 (Initial SQL): Operations 1-{first_df_op_idx}")
                lines.append(
                    f"    ️  Phase 2 (DataFrame Operations): Operations {first_df_op_idx + 1}-{len(self._lazy_ops)}"
                )
                lines.append("    ️  Note: Each operation shows its execution engine [chDB] or [Pandas]")
                lines.append("")
            else:
                lines.append("    ️  All operations will execute via SQL Engine")
                lines.append("")

            for i, op in enumerate(self._lazy_ops):
                counter += 1
                # Determine which engine will execute this operation
                is_sql_phase = (first_df_op_idx is None) or (i < first_df_op_idx)

                if isinstance(op, LazyRelationalOp):
                    # LazyRelationalOp engine depends on which phase it's in
                    if is_sql_phase:
                        # SQL engine will execute this
                        lines.append(f" [{counter}] 🚀 [chDB] {op.describe()}")
                    else:
                        # Pandas engine will execute this - use pandas terminology
                        lines.append(f" [{counter}] 🐼 [Pandas] {op.describe_pandas()}")
                else:
                    # For other ops, ask the op itself which engine it will use
                    engine = op.execution_engine()
                    if engine == 'chDB':
                        lines.append(f" [{counter}] 🚀 [chDB] {op.describe()}")
                    else:
                        lines.append(f" [{counter}] 🐼 [Pandas] {op.describe()}")

        # ========== Legacy operation history (for pandas compat operations) ==========
        history_lazy_ops, mat_op, history_executed_ops = self._analyze_execution_phases()

        if mat_op:
            lines.append("\nExecution Point:")
            lines.append("─" * 80)
            counter += 1
            lines.append(f" [{counter}] 🔄 {mat_op['description']}")
            lines.append("     └─> Executes SQL query and caches result as DataFrame")
            if verbose and mat_op.get('details'):
                for k, v in mat_op['details'].items():
                    lines.append(f"         • {k}: {v}")

        if history_executed_ops:
            lines.append("\nPost-Execution Operations:")
            lines.append("─" * 80)
            lines.extend(self._render_operations(history_executed_ops, counter + 1, verbose))
            counter += len(history_executed_ops)

        # ========== Final State ==========
        lines.append("\n" + "─" * 80)
        if self._lazy_ops or self._has_sql_state():
            lines.append("Final State: 📊 Pending (lazy, not yet executed)")
            lines.append("             └─> Will execute when print(), .to_df(), .execute() is called")
        else:
            lines.append("Final State: 📊 No operations recorded")
            lines.append("             └─> Start by loading data or defining operations")

        # ========== Generated SQL Query ==========
        if self._has_sql_state():
            lines.append("\n" + "─" * 80)
            lines.append("Generated SQL Query:")
            lines.append("─" * 80)
            try:
                # Use execution_format=True to show the actual SQL that will be executed
                sql = self.to_sql(execution_format=True)
                if verbose or len(sql) < 500:
                    lines.append(f"\n{sql}\n")
                else:
                    lines.append(f"\n{sql[:500]}...")
                    lines.append("[Query truncated. Use explain(verbose=True) for full query]\n")
            except Exception as e:
                lines.append(f"\nError generating SQL: {e}\n")

        lines.append("=" * 80)

        output = "\n".join(lines)
        print(output)
        return output

    def _is_sql_query(self) -> bool:
        """Check if this DataStore represents a SQL query."""
        return (
            self._select_fields
            or self._where_condition
            or self._joins
            or self._groupby_fields
            or self._having_condition
            or self._orderby_fields
            or self._limit_value is not None
            or self._offset_value is not None
            or self._distinct
            or self._table_function is not None
            or self.table_name is not None
        )

    def _has_sql_state(self) -> bool:
        """Check if DataStore has any SQL query state."""
        return bool(
            self._select_fields
            or self._where_condition
            or self._joins
            or self._groupby_fields
            or self._having_condition
            or self._orderby_fields
            or self._table_function
            or self.table_name
        )

    def _is_cache_valid(self) -> bool:
        """
        Check if the cached result is still valid.

        Returns:
            True if cache is valid and can be used, False otherwise.
        """
        # Check if caching is enabled globally
        if not is_cache_enabled():
            return False

        # Check if we have a cached result
        if self._cached_result is None:
            return False

        # Check version match (cache invalidated if operations were added)
        if self._cached_at_version != self._cache_version:
            self._logger.debug(
                "Cache invalid: version mismatch (cached=%d, current=%d)", self._cached_at_version, self._cache_version
            )
            return False

        # Check TTL if configured
        ttl = get_cache_ttl()
        if ttl > 0 and self._cache_timestamp is not None:
            age = time.time() - self._cache_timestamp
            if age > ttl:
                self._logger.debug("Cache invalid: TTL expired (age=%.2fs, ttl=%.2fs)", age, ttl)
                return False

        return True

    def _invalidate_cache(self):
        """
        Invalidate the cached result.

        Called when new operations are added to the pipeline.
        """
        self._cache_version += 1
        self._logger.debug("Cache invalidated: version now %d", self._cache_version)

    def clear_cache(self):
        """
        Manually clear the cached result.

        This forces re-execution on the next access.

        Example:
            >>> ds = DataStore.from_dataframe(df)
            >>> ds = ds.filter(ds['value'] > 10)
            >>> print(ds)  # Executes and caches
            >>> ds.clear_cache()  # Clear cache
            >>> print(ds)  # Re-executes
        """
        self._cached_result = None
        self._cached_at_version = -1
        self._cache_timestamp = None
        self._logger.debug("Cache manually cleared")

    def _add_lazy_op(self, op: LazyOp):
        """
        Add a lazy operation to the pipeline and invalidate cache.

        This is the central method for adding lazy operations. It ensures
        that the cache is properly invalidated when the pipeline changes.

        Args:
            op: The lazy operation to add
        """
        self._lazy_ops.append(op)
        self._invalidate_cache()

    def _execute(self):
        """
        Execute all lazy operations into a DataFrame.

        Execute all pending operations and return a DataFrame.

        This is the core of lazy execution. When caching is enabled (default),
        results are cached and reused for repeated accesses (e.g., multiple repr calls).
        Cache is automatically invalidated when new operations are added.

        Execution strategy (segmented execution):
        1. Check if valid cache exists (return cached result if so)
        2. Use QueryPlanner.plan_segments() to create a segmented execution plan
        3. Execute each segment:
           - SQL segments: via chDB (from source or Python() table function)
           - Pandas segments: execute operations on DataFrame in order
        4. Cache the result if caching is enabled

        Returns:
            pandas DataFrame with all operations applied
        """
        from .query_planner import QueryPlanner

        # Create profiler for this execution
        profiler = get_profiler()

        with profiler.step("Total Execution"):
            # Check cache first
            with profiler.step("Cache Check"):
                if self._is_cache_valid():
                    self._logger.debug("Using cached result (version=%d)", self._cached_at_version)
                    if is_profiling_enabled():
                        profiler.log_report()
                    return self._cached_result

            self._logger.debug("=" * 70)
            self._logger.debug("Starting execution (version=%d)", self._cache_version)
            self._logger.debug("=" * 70)

            # Log all lazy operations
            if self._lazy_ops:
                self._logger.debug("Lazy operations chain (%d operations):", len(self._lazy_ops))
                for i, op in enumerate(self._lazy_ops):
                    self._logger.debug("  [%d] %s", i + 1, op.describe())
            else:
                self._logger.debug("No lazy operations recorded")

            # Segmented planning phase
            with profiler.step("Query Planning", ops_count=len(self._lazy_ops)):
                planner = QueryPlanner()
                has_sql_source = bool(self._table_function or self.table_name)
                schema = self.schema() if has_sql_source else self._schema
                exec_plan = planner.plan_segments(self._lazy_ops, has_sql_source, schema=schema)

                self._logger.debug(exec_plan.describe())

            # Execute segments
            df = pd.DataFrame()
            has_executed_pandas = False

            for seg_idx, segment in enumerate(exec_plan.segments):
                seg_num = seg_idx + 1
                self._logger.debug("-" * 70)
                self._logger.debug("Segment %d/%d: %s", seg_num, len(exec_plan.segments), segment.describe())
                self._logger.debug("-" * 70)

                if segment.is_sql():
                    with profiler.step(f"SQL Segment {seg_num}", ops=len(segment.ops)):
                        df = self._execute_sql_segment(segment, df, schema, profiler)
                else:
                    with profiler.step(f"Pandas Segment {seg_num}", ops=len(segment.ops)):
                        df = self._execute_pandas_segment(segment, df, profiler)
                        has_executed_pandas = True

            self._logger.debug("=" * 70)
            self._logger.debug("Execution complete. Final DataFrame shape: %s", df.shape)
            self._logger.debug("=" * 70)

            # Cache the result if caching is enabled
            with profiler.step("Cache Write"):
                if is_cache_enabled():
                    self._cached_result = df
                    self._cache_timestamp = time.time()

                    # Checkpoint if we executed any Pandas operations or multiple SQL segments
                    # This enables incremental execution for future operations
                    needs_checkpoint = has_executed_pandas or exec_plan.sql_segment_count() > 1

                    if needs_checkpoint:
                        from .lazy_ops import LazyDataFrameSource

                        # Replace lazy_ops with a single DataFrame source
                        self._lazy_ops = [LazyDataFrameSource(df)]

                        # Clear SQL state since we're now working from a DataFrame
                        self._table_function = None
                        self.table_name = None
                        self._select_fields = []
                        self._where_condition = None
                        self._joins = []
                        self._groupby_fields = []
                        self._having_condition = None
                        self._orderby_fields = []
                        self._limit_value = None
                        self._offset_value = None
                        self._distinct = False

                        self._logger.debug("Pipeline checkpointed: lazy_ops replaced with DataFrame source")
                    else:
                        self._logger.debug("Pure SQL execution: SQL state preserved")

                    self._cache_version = 0
                    self._cached_at_version = 0

        if is_profiling_enabled():
            profiler.log_report()

        return df

    def _execute_sql_segment(
        self,
        segment: 'ExecutionSegment',
        df: pd.DataFrame,
        schema: Dict[str, str],
        profiler,
    ) -> pd.DataFrame:
        """
        Execute a SQL segment.

        For the first segment (from original data source), uses direct SQL execution.
        For subsequent segments, uses Python() table function to query the DataFrame.

        Args:
            segment: The SQL segment to execute
            df: Current DataFrame (used for subsequent segments)
            schema: Column schema
            profiler: Profiler for timing

        Returns:
            Result DataFrame
        """
        from .query_planner import ExecutionSegment

        if segment.is_first_segment and (self._table_function or self.table_name):
            # First segment: execute from original data source
            if self._executor is None:
                with profiler.step("Connection"):
                    self._logger.debug("Connecting to data source...")
                    self.connect()

            # Build SQL from segment's plan
            plan = segment.plan
            if plan is None:
                # Fallback: create a plan from segment ops
                from .query_planner import QueryPlan

                plan = QueryPlan(has_sql_source=True)
                plan.sql_ops = segment.ops.copy()

            sql_engine = SQLExecutionEngine(self)
            build_result = sql_engine.build_sql_from_plan(plan, schema)
            sql = build_result.sql

            # Append format settings if present (e.g., input_format_parquet_preserve_order)
            if self._format_settings:
                settings_parts = []
                for key, value in self._format_settings.items():
                    if isinstance(value, str):
                        settings_parts.append(f"{key}='{value}'")
                    else:
                        settings_parts.append(f"{key}={value}")
                sql = f"{sql} SETTINGS {', '.join(settings_parts)}"

            self._logger.debug("  Executing SQL: %s", sql[:200] + "..." if len(sql) > 200 else sql)

            with profiler.step("SQL Execution"):
                result = self._executor.execute(sql)

            with profiler.step("Result to DataFrame"):
                df = result.to_df()
                df = self._postprocess_sql_result(df, plan)

            self._logger.debug("  SQL returned DataFrame with shape: %s", df.shape)

        else:
            # Subsequent segment: execute on DataFrame via Python() table function
            if df.empty and not (self._table_function or self.table_name):
                self._logger.debug("  No data to execute SQL on")
                return df

            # If we have a source but no DataFrame yet, load raw data first
            if df.empty and (self._table_function or self.table_name):
                if self._executor is None:
                    self.connect()
                # Load raw data without any filters
                if self._table_function:
                    table_sql = self._table_function.to_sql()
                else:
                    table_sql = f'{self.quote_char}{self.table_name}{self.quote_char}'
                load_sql = f"SELECT * FROM {table_sql}"
                # Append format settings if present
                if self._format_settings:
                    settings_parts = []
                    for key, value in self._format_settings.items():
                        if isinstance(value, str):
                            settings_parts.append(f"{key}='{value}'")
                        else:
                            settings_parts.append(f"{key}={value}")
                    load_sql = f"{load_sql} SETTINGS {', '.join(settings_parts)}"
                result = self._executor.execute(load_sql)
                df = result.to_df()
                self._logger.debug("  Loaded raw data from source: %s rows", len(df))

            plan = segment.plan
            if plan is None:
                from .query_planner import QueryPlan

                plan = QueryPlan(has_sql_source=True)
                plan.sql_ops = segment.ops.copy()

            sql_engine = SQLExecutionEngine(self)
            df = sql_engine.execute_sql_on_dataframe(df, plan, schema)

            self._logger.debug("  SQL on DataFrame returned shape: %s", df.shape)

        return df

    def _execute_pandas_segment(
        self,
        segment: 'ExecutionSegment',
        df: pd.DataFrame,
        profiler,
    ) -> pd.DataFrame:
        """
        Execute a Pandas segment.

        Args:
            segment: The Pandas segment to execute
            df: Current DataFrame
            profiler: Profiler for timing

        Returns:
            Result DataFrame
        """
        # If first operation and we have SQL source but no DataFrame, load raw data
        if df.empty and (self._table_function or self.table_name):
            if self._executor is None:
                self.connect()
            # Load raw data - include JOINs if present
            if self._table_function:
                table_sql = self._table_function.to_sql()
            else:
                table_sql = f'{self.quote_char}{self.table_name}{self.quote_char}'

            # Build FROM clause with optional alias for joins
            if self._joins:
                alias = self._get_table_alias()
                from_clause = f"{table_sql} AS {self.quote_char}{alias}{self.quote_char}"
            else:
                from_clause = table_sql

            load_sql = f"SELECT * FROM {from_clause}"

            # Add JOIN clauses if present
            if self._joins:
                sql_engine = SQLExecutionEngine(self)
                for other_ds, join_type, join_condition in self._joins:
                    join_clause = sql_engine._build_join_clause(other_ds, join_type, join_condition)
                    load_sql = f"{load_sql} {join_clause}"

            # Append format settings if present
            if self._format_settings:
                settings_parts = []
                for key, value in self._format_settings.items():
                    if isinstance(value, str):
                        settings_parts.append(f"{key}='{value}'")
                    else:
                        settings_parts.append(f"{key}={value}")
                load_sql = f"{load_sql} SETTINGS {', '.join(settings_parts)}"
            self._logger.debug("  Load SQL for Pandas: %s", load_sql)
            result = self._executor.execute(load_sql)
            df = result.to_df()
            self._logger.debug("  Loaded raw data from source for Pandas: %s rows", len(df))

        for i, op in enumerate(segment.ops, 1):
            self._logger.debug("  [%d/%d] Executing: %s", i, len(segment.ops), op.describe())
            op_name = op.__class__.__name__
            with profiler.step(op_name):
                df = op.execute(df, self)

        return df

    def _postprocess_sql_result(self, df: pd.DataFrame, plan: 'QueryPlan') -> pd.DataFrame:
        """
        Post-process SQL result: handle alias renames, groupby index, etc.

        Args:
            df: Result DataFrame from SQL execution
            plan: QueryPlan with metadata

        Returns:
            Post-processed DataFrame
        """
        # Rename temp aliases back to original names
        if plan.alias_renames:
            rename_back = {temp: orig for temp, orig in plan.alias_renames.items() if temp in df.columns}
            if rename_back:
                df = df.rename(columns=rename_back)
                self._logger.debug("  Renamed temp aliases: %s", rename_back)

        # For GroupBy SQL pushdown: set group keys as index
        if plan.groupby_agg and plan.groupby_agg.groupby_cols:
            groupby_cols = plan.groupby_agg.groupby_cols
            if all(col in df.columns for col in groupby_cols):
                df = df.set_index(groupby_cols)
                self._logger.debug("  Set groupby columns as index: %s", groupby_cols)

            # Convert flat column names to MultiIndex for pandas compatibility
            if plan.groupby_agg.agg_dict:
                col_rename_map = {}
                for col, funcs in plan.groupby_agg.agg_dict.items():
                    if isinstance(funcs, str):
                        funcs = [funcs]
                    for func in funcs:
                        flat_name = f"{col}_{func}"
                        if flat_name in df.columns:
                            col_rename_map[flat_name] = (col, func)

                if col_rename_map:
                    new_columns = []
                    for c in df.columns:
                        if c in col_rename_map:
                            new_columns.append(col_rename_map[c])
                        else:
                            new_columns.append((c, ''))
                    df.columns = pd.MultiIndex.from_tuples(new_columns)
                    self._logger.debug("  Converted flat columns to MultiIndex")

        return df

    def _build_execution_sql(self) -> Optional[str]:
        """
        Build SQL query that will be executed for lazy operations.

        This method generates the exact SQL that _execute() would run,
        ensuring that explain() and to_sql() show consistent SQL with actual execution.

        Returns:
            SQL query string if there's a SQL source and lazy operations,
            None if no SQL source or no operations.
        """
        # Only generate SQL if we have a SQL source
        if not (self._table_function or self.table_name):
            return None

        # No lazy ops means use traditional SQL generation
        if not self._lazy_ops:
            return self._generate_select_sql(self.quote_char)

        from .query_planner import QueryPlanner

        # Use QueryPlanner to analyze lazy operations (same logic as _execute)
        planner = QueryPlanner()
        schema = self._schema or {}
        plan = planner.plan(self._lazy_ops, has_sql_source=True, schema=schema)

        # Use SQLExecutionEngine to build SQL from plan
        sql_engine = SQLExecutionEngine(self)
        result = sql_engine.build_sql_from_plan(plan, schema)
        return result.sql

    def _get_table_source(self) -> str:
        """Get the table source for SQL generation."""
        if self._table_function:
            return self._table_function.to_sql()
        elif self.table_name:
            return format_identifier(self.table_name, self.quote_char)
        return ""

    def _build_sql_from_state(
        self,
        select_fields,
        where_conditions,
        orderby_fields,
        limit_value,
        offset_value,
        joins=None,
        distinct=False,
        groupby_fields=None,
        having_condition=None,
    ):
        """Build SQL query from given state (not from instance variables)."""
        from .utils import build_orderby_clause, is_stable_sort

        # Check if we need a subquery for stable sort with WHERE
        # When both WHERE and stable ORDER BY exist, rowNumberInAllBlocks() would give
        # post-filter row numbers, not original row numbers. We need a subquery to preserve
        # the original row order.
        needs_stable_sort = orderby_fields and is_stable_sort(self._orderby_kind)
        needs_subquery_for_stable = needs_stable_sort and where_conditions and not groupby_fields and not joins

        if needs_subquery_for_stable:
            return self._build_sql_with_stable_sort_subquery(
                select_fields,
                where_conditions,
                orderby_fields,
                limit_value,
                offset_value,
                distinct,
            )

        parts = []

        # SELECT (with optional DISTINCT)
        distinct_keyword = 'DISTINCT ' if distinct else ''
        if select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            # Check if we need to prepend '*' (SELECT *, computed_col)
            if self._select_star:
                parts.append(f"SELECT {distinct_keyword}*, {fields_sql}")
            else:
                parts.append(f"SELECT {distinct_keyword}{fields_sql}")
        else:
            parts.append(f"SELECT {distinct_keyword}*")

        # FROM (with alias if joins present)
        if self._table_function:
            # Handle table function objects
            if hasattr(self._table_function, 'to_sql'):
                table_sql = self._table_function.to_sql()
            else:
                table_sql = str(self._table_function)
            # Add alias when joins are present (required by ClickHouse for disambiguation)
            if joins:
                alias = self._get_table_alias()
                parts.append(f"FROM {table_sql} AS {format_identifier(alias, self.quote_char)}")
            else:
                parts.append(f"FROM {table_sql}")
        elif self.table_name:
            parts.append(f"FROM {self.quote_char}{self.table_name}{self.quote_char}")

        # JOIN clauses
        if joins:
            for other_ds, join_type, join_condition in joins:
                # Generate JOIN clause
                join_keyword = join_type.value if join_type.value else ''
                if join_keyword:
                    join_clause = f"{join_keyword} JOIN"
                else:
                    join_clause = "JOIN"

                # Handle subquery joins
                if isinstance(other_ds, DataStore) and other_ds._is_subquery:
                    other_table = other_ds.to_sql(quote_char=self.quote_char, as_subquery=True)
                elif isinstance(other_ds, DataStore) and other_ds._table_function:
                    # Use table function for the joined table with alias
                    table_func_sql = other_ds._table_function.to_sql(quote_char=self.quote_char)
                    alias = other_ds._get_table_alias()
                    other_table = f"{table_func_sql} AS {format_identifier(alias, self.quote_char)}"
                else:
                    other_table = format_identifier(other_ds.table_name, self.quote_char)

                # Handle USING vs ON syntax
                if isinstance(join_condition, tuple) and join_condition[0] == 'USING':
                    # USING (col1, col2, ...) syntax
                    columns = join_condition[1]
                    using_cols = ', '.join(format_identifier(c, self.quote_char) for c in columns)
                    parts.append(f"{join_clause} {other_table} USING ({using_cols})")
                else:
                    # ON condition syntax
                    condition_sql = join_condition.to_sql(quote_char=self.quote_char)
                    parts.append(f"{join_clause} {other_table} ON {condition_sql}")

        # WHERE
        if where_conditions:
            combined = where_conditions[0]
            for cond in where_conditions[1:]:
                combined = combined & cond
            parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # GROUP BY
        if groupby_fields:
            groupby_sql = ', '.join(f.to_sql(quote_char=self.quote_char) for f in groupby_fields)
            parts.append(f"GROUP BY {groupby_sql}")

        # HAVING
        if having_condition:
            having_sql = having_condition.to_sql(quote_char=self.quote_char)
            parts.append(f"HAVING {having_sql}")

        # ORDER BY (stable sort if kind='stable' or 'mergesort', matching pandas behavior)
        if orderby_fields:
            # Check for special row order marker (must be a string, not a Field object)
            first_field = orderby_fields[0][0]
            if len(orderby_fields) == 1 and isinstance(first_field, str) and first_field == '__rowNumberInAllBlocks__':
                # Special case: use rowNumberInAllBlocks() for row order preservation
                parts.append("ORDER BY rowNumberInAllBlocks()")
            else:
                orderby_sql = build_orderby_clause(
                    orderby_fields, self.quote_char, stable=is_stable_sort(self._orderby_kind)
                )
                parts.append(f"ORDER BY {orderby_sql}")

        # LIMIT
        if limit_value is not None:
            parts.append(f"LIMIT {limit_value}")

        # OFFSET
        if offset_value is not None:
            parts.append(f"OFFSET {offset_value}")

        return ' '.join(parts)

    def _build_sql_with_stable_sort_subquery(
        self,
        select_fields,
        where_conditions,
        orderby_fields,
        limit_value,
        offset_value,
        distinct=False,
    ):
        """
        Build SQL with subquery for stable sort when WHERE is present.

        When we have both WHERE and stable ORDER BY, rowNumberInAllBlocks() would give
        post-filter row numbers. To preserve original row order as tie-breaker, we use:

        SELECT * EXCEPT(__orig_row_num__) FROM (
            SELECT *, rowNumberInAllBlocks() AS __orig_row_num__
            FROM source
            WHERE conditions
        ) ORDER BY col1 ASC/DESC, __orig_row_num__ ASC
        LIMIT N

        This ensures __orig_row_num__ is calculated from the source data before filtering.
        """
        from .utils import build_orderby_clause

        # Build the inner query: SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM source
        if self._table_function:
            if hasattr(self._table_function, 'to_sql'):
                table_sql = self._table_function.to_sql()
            else:
                table_sql = str(self._table_function)
        elif self.table_name:
            table_sql = f"{self.quote_char}{self.table_name}{self.quote_char}"
        else:
            table_sql = "source"

        inner_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_sql}"

        # Build middle query: SELECT columns FROM (inner) WHERE conditions
        middle_parts = []
        distinct_keyword = 'DISTINCT ' if distinct else ''

        if select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            if self._select_star:
                middle_parts.append(f"SELECT {distinct_keyword}*, {fields_sql}, __orig_row_num__")
            else:
                middle_parts.append(f"SELECT {distinct_keyword}{fields_sql}, __orig_row_num__")
        else:
            middle_parts.append(f"SELECT {distinct_keyword}*, __orig_row_num__")

        middle_parts.append(f"FROM ({inner_sql}) AS __subq_with_rownum__")

        # WHERE
        if where_conditions:
            combined = where_conditions[0]
            for cond in where_conditions[1:]:
                combined = combined & cond
            middle_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        middle_sql = ' '.join(middle_parts)

        # Build outer query: SELECT * EXCEPT(__orig_row_num__) FROM (middle) ORDER BY ... LIMIT
        outer_parts = []
        outer_parts.append("SELECT * EXCEPT(__orig_row_num__)")
        outer_parts.append(f"FROM ({middle_sql}) AS __subq_for_stable_sort__")

        # ORDER BY with __orig_row_num__ as tie-breaker instead of rowNumberInAllBlocks()
        if orderby_fields:
            orderby_sql = build_orderby_clause(
                orderby_fields, self.quote_char, stable=False  # Don't add rowNumberInAllBlocks()
            )
            outer_parts.append(f"ORDER BY {orderby_sql}, __orig_row_num__ ASC")

        # LIMIT
        if limit_value is not None:
            outer_parts.append(f"LIMIT {limit_value}")

        # OFFSET
        if offset_value is not None:
            outer_parts.append(f"OFFSET {offset_value}")

        return ' '.join(outer_parts)

    # ========== Static Factory Methods for Data Sources ==========

    @classmethod
    def from_file(
        cls, path: str, format: str = None, structure: str = None, compression: str = None, **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from local file.

        Args:
            path: File path (supports glob patterns)
            format: File format (optional, auto-detected from extension).
                    For CSV files, defaults to 'CSVWithNames' (first row is header),
                    matching pandas' default behavior.
            structure: Optional table structure
            compression: Optional compression method
            **kwargs: Additional connection parameters

        Example:
            >>> ds = DataStore.from_file("data.parquet")
            >>> ds = DataStore.from_file("data.csv")  # Uses CSVWithNames (header row)
            >>> ds = DataStore.from_file("data.csv", format="CSV")  # No header row
        """
        # Auto-infer format from file extension if not specified
        if format is None:
            from .uri_parser import _infer_format_from_path

            format = _infer_format_from_path(path)

        ds = cls("file", path=path, format=format, structure=structure, compression=compression, **kwargs)

        # Use UTC timezone to ensure datetime values match pandas (which uses naive UTC)
        # New versions of chDB may apply system timezone, causing value shifts
        ds._format_settings['session_timezone'] = 'UTC'

        # For Parquet files, enable row order preservation to match pandas behavior
        # chDB may read row groups in parallel which can reorder rows
        if format and format.lower() == 'parquet':
            ds._format_settings['input_format_parquet_preserve_order'] = 1

        return ds

    @classmethod
    def from_s3(
        cls,
        url: str,
        access_key_id: str = None,
        secret_access_key: str = None,
        format: str = None,
        nosign: bool = False,
        **kwargs,
    ) -> 'DataStore':
        """
        Create DataStore from S3.

        Args:
            url: S3 URL
            access_key_id: AWS access key (optional if nosign=True)
            secret_access_key: AWS secret key (optional if nosign=True)
            format: Data format (optional, auto-detected)
            nosign: Use anonymous access
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)
            >>> ds = DataStore.from_s3("s3://bucket/data.csv",
            ...                        access_key_id="KEY",
            ...                        secret_access_key="SECRET")
        """
        return cls(
            "s3",
            url=url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            format=format,
            nosign=nosign,
            **kwargs,
        )

    @classmethod
    def from_hdfs(cls, uri: str, format: str = None, structure: str = None, **kwargs) -> 'DataStore':
        """
        Create DataStore from HDFS.

        Args:
            uri: HDFS URI (e.g., 'hdfs://namenode:9000/path')
            format: Data format (optional, auto-detected)
            structure: Optional table structure
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_hdfs("hdfs://namenode:9000/data/*.parquet")
        """
        return cls("hdfs", uri=uri, format=format, structure=structure, **kwargs)

    @classmethod
    def from_mysql(cls, host: str, database: str, table: str, user: str, password: str = "", **kwargs) -> 'DataStore':
        """
        Create DataStore from MySQL database.

        Args:
            host: MySQL server address (host:port)
            database: Database name
            table: Table name
            user: Username
            password: Password
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_mysql("localhost:3306", "mydb", "users",
            ...                           user="root", password="pass")
        """
        return cls("mysql", host=host, database=database, table=table, user=user, password=password, **kwargs)

    @classmethod
    def from_postgresql(
        cls, host: str, database: str, table: str, user: str, password: str = "", **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from PostgreSQL database.

        Args:
            host: PostgreSQL server address (host:port)
            database: Database name
            table: Table name (can include schema like 'schema.table')
            user: Username
            password: Password
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_postgresql("localhost:5432", "mydb", "users",
            ...                                user="postgres", password="pass")
        """
        return cls("postgresql", host=host, database=database, table=table, user=user, password=password, **kwargs)

    @classmethod
    def from_clickhouse(
        cls,
        host: str,
        database: str,
        table: str,
        user: str = "default",
        password: str = "",
        secure: bool = False,
        **kwargs,
    ) -> 'DataStore':
        """
        Create DataStore from remote ClickHouse server.

        Args:
            host: ClickHouse server address (host:port)
            database: Database name
            table: Table name
            user: Username (default: 'default')
            password: Password
            secure: Use secure connection (remoteSecure)
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_clickhouse("localhost:9000", "default", "events")
            >>> ds_secure = DataStore.from_clickhouse("server:9440", "default", "events",
            ...                                       secure=True)
        """
        return cls(
            "clickhouse",
            host=host,
            database=database,
            table=table,
            user=user,
            password=password,
            secure=secure,
            **kwargs,
        )

    @classmethod
    def from_mongodb(
        cls, host: str, database: str, collection: str, user: str, password: str = "", **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from MongoDB (read-only).

        Args:
            host: MongoDB server address (host:port)
            database: Database name
            collection: Collection name
            user: Username
            password: Password
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_mongodb("localhost:27017", "mydb", "users",
            ...                             user="admin", password="pass")
        """
        return cls(
            "mongodb", host=host, database=database, collection=collection, user=user, password=password, **kwargs
        )

    @classmethod
    def from_url(
        cls, url: str, format: str = None, structure: str = None, headers: List[str] = None, **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from HTTP/HTTPS URL.

        Args:
            url: HTTP(S) URL to the data
            format: Data format (optional, auto-detected from URL if not provided)
            structure: Optional table structure
            headers: Optional HTTP headers
            **kwargs: Additional parameters

        Example:
            >>> # Auto-detect format from URL
            >>> ds = DataStore.from_url("https://example.com/data.json")
            >>>
            >>> # Explicit format
            >>> ds = DataStore.from_url("https://example.com/data.json",
            ...                         format="JSONEachRow")
        """
        return cls("url", url=url, format=format, structure=structure, headers=headers, **kwargs)

    @classmethod
    def from_sqlite(cls, database_path: str, table: str, **kwargs) -> 'DataStore':
        """
        Create DataStore from SQLite database (read-only).

        Args:
            database_path: Path to SQLite database file
            table: Table name
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_sqlite("/path/to/database.db", "users")
        """
        return cls("sqlite", database_path=database_path, table=table, **kwargs)

    @classmethod
    def from_iceberg(cls, url: str, access_key_id: str = None, secret_access_key: str = None, **kwargs) -> 'DataStore':
        """
        Create DataStore from Apache Iceberg table (read-only).

        Args:
            url: Path to Iceberg table
            access_key_id: Access key for cloud storage
            secret_access_key: Secret key for cloud storage
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_iceberg("s3://warehouse/my_table",
            ...                             access_key_id="KEY",
            ...                             secret_access_key="SECRET")
        """
        return cls("iceberg", url=url, access_key_id=access_key_id, secret_access_key=secret_access_key, **kwargs)

    @classmethod
    def from_delta(cls, url: str, access_key_id: str = None, secret_access_key: str = None, **kwargs) -> 'DataStore':
        """
        Create DataStore from Delta Lake table (read-only).

        Args:
            url: Path to Delta Lake table
            access_key_id: Access key for cloud storage
            secret_access_key: Secret key for cloud storage
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_delta("s3://bucket/delta_table",
            ...                           access_key_id="KEY",
            ...                           secret_access_key="SECRET")
        """
        return cls("delta", url=url, access_key_id=access_key_id, secret_access_key=secret_access_key, **kwargs)

    @classmethod
    def from_numbers(cls, count: int, start: int = None, step: int = None, **kwargs) -> 'DataStore':
        """
        Create DataStore that generates number sequence.

        Args:
            count: Number of values to generate
            start: Start number (optional)
            step: Step size (optional)
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_numbers(100)  # 0 to 99
            >>> ds = DataStore.from_numbers(10, start=10)  # 10 to 19
            >>> ds = DataStore.from_numbers(10, start=0, step=2)  # Even numbers
        """
        return cls("numbers", count=count, start=start, step=step, **kwargs)

    @classmethod
    def from_azure(
        cls, connection_string: str, container: str, path: str = "", format: str = None, **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from Azure Blob Storage.

        Args:
            connection_string: Azure connection string
            container: Container name
            path: Blob path (supports glob patterns)
            format: Data format (optional, auto-detected)
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_azure(
            ...     connection_string="DefaultEndpointsProtocol=https;...",
            ...     container="mycontainer",
            ...     path="data/*.parquet"
            ... )
        """
        return cls(
            "azure", connection_string=connection_string, container=container, path=path, format=format, **kwargs
        )

    @classmethod
    def from_gcs(
        cls, url: str, hmac_key: str = None, hmac_secret: str = None, format: str = None, nosign: bool = False, **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from Google Cloud Storage.

        Args:
            url: GCS URL (https://storage.googleapis.com/bucket/path)
            hmac_key: GCS HMAC key (optional if nosign)
            hmac_secret: GCS HMAC secret (optional if nosign)
            format: Data format (optional, auto-detected)
            nosign: Use anonymous access
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_gcs(
            ...     "gs://bucket/data.parquet",
            ...     hmac_key="KEY",
            ...     hmac_secret="SECRET"
            ... )
        """
        return cls("gcs", url=url, hmac_key=hmac_key, hmac_secret=hmac_secret, format=format, nosign=nosign, **kwargs)

    @classmethod
    def from_redis(
        cls, host: str, key: str, structure: str, password: str = None, db_index: int = 0, **kwargs
    ) -> 'DataStore':
        """
        Create DataStore from Redis key-value store.

        Args:
            host: Redis server address (host:port)
            key: Name of the primary-key column in structure
            structure: Table structure 'key Type, v1 Type, ...'
            password: Redis password (optional)
            db_index: Database index (default: 0)
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_redis(
            ...     host="localhost:6379",
            ...     key="key",
            ...     structure="key String, value String, score UInt32"
            ... )
        """
        return cls("redis", host=host, key=key, structure=structure, password=password, db_index=db_index, **kwargs)

    @classmethod
    def from_hudi(cls, url: str, access_key_id: str = None, secret_access_key: str = None, **kwargs) -> 'DataStore':
        """
        Create DataStore from Apache Hudi table (read-only).

        Args:
            url: Path to Hudi table in S3
            access_key_id: AWS access key
            secret_access_key: AWS secret key
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_hudi(
            ...     "s3://bucket/hudi_table",
            ...     access_key_id="KEY",
            ...     secret_access_key="SECRET"
            ... )
        """
        return cls("hudi", url=url, access_key_id=access_key_id, secret_access_key=secret_access_key, **kwargs)

    @classmethod
    def from_random(
        cls,
        structure: str,
        random_seed: int = None,
        max_string_length: int = None,
        max_array_length: int = None,
        **kwargs,
    ) -> 'DataStore':
        """
        Create DataStore that generates random data for testing.

        Args:
            structure: Table structure with column types
            random_seed: Random seed for reproducibility (optional)
            max_string_length: Max string length (optional)
            max_array_length: Max array length (optional)
            **kwargs: Additional parameters

        Example:
            >>> ds = DataStore.from_random(
            ...     structure="id UInt32, name String, value Float64",
            ...     random_seed=42
            ... )
        """
        return cls(
            "generaterandom",
            structure=structure,
            random_seed=random_seed,
            max_string_length=max_string_length,
            max_array_length=max_array_length,
            **kwargs,
        )

    @classmethod
    def from_df(cls, df, name: str = None) -> 'DataStore':
        """
        Create DataStore from a pandas DataFrame.

        This allows you to use DataStore's query building and lazy execution
        features on an existing DataFrame.

        Args:
            df: pandas DataFrame to wrap
            name: Optional name for the data source (used in explain output)

        Returns:
            DataStore wrapping the DataFrame

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> ds = DataStore.from_df(df)
            >>> ds.filter(ds.age > 26).to_df()
               name  age
            1   Bob   30

            >>> # With SQL operations
            >>> ds = DataStore.from_df(df, name='users')
            >>> ds.sql("SELECT * FROM __df__ WHERE age > 26").to_df()
        """
        from .lazy_ops import LazyDataFrameSource

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        from .lazy_ops import LazyDataFrameSource

        # Create a new DataStore with DataFrame source
        new_ds = cls()

        # Cache the DataFrame for later SQL access if needed
        # PythonTableFunction will be created on-demand when SQL functions are used
        new_ds._source_df = df
        new_ds._source_df_name = name

        # No table function yet - will be created on-demand
        new_ds._table_function = None
        new_ds.table_name = None

        # Clear other state
        new_ds._select_fields = []
        new_ds._select_star = False
        new_ds._where_condition = None
        new_ds._joins = []
        new_ds._groupby_fields = []
        new_ds._having_condition = None
        new_ds._orderby_fields = []
        new_ds._limit_value = None
        new_ds._offset_value = None
        new_ds._distinct = False

        # Add the DataFrame as a lazy source for pandas-style operations
        new_ds._lazy_ops = [LazyDataFrameSource(df)]

        # Set source description for explain()
        shape_str = f"{df.shape[0]} rows x {df.shape[1]} cols"
        new_ds._original_source_desc = f"DataFrame({name or 'unnamed'}, {shape_str})"

        # Build schema from DataFrame dtypes
        new_ds._schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return new_ds

    @classmethod
    def from_dataframe(cls, df, name: str = None) -> 'DataStore':
        """
        Create DataStore from a pandas DataFrame.

        Alias for `from_df()`. See `from_df()` for full documentation.

        Args:
            df: pandas DataFrame to wrap
            name: Optional name for the data source (used in explain output)

        Returns:
            DataStore wrapping the DataFrame

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
            >>> ds = DataStore.from_dataframe(df)
            >>> ds.filter(ds.age > 26).to_df()
        """
        return cls.from_df(df, name=name)

    @classmethod
    def uri(cls, uri: str, **kwargs) -> 'DataStore':
        """
        Create DataStore from a URI string with automatic type inference.

        This is the simplest way to create a DataStore - just provide a URI
        and the source type and parameters will be automatically inferred.

        Args:
            uri: Connection URI string
            **kwargs: Additional parameters to override auto-detected values

        Supported URI formats:
            - Local files: "file:///path/to/data.csv" or "/path/to/data.csv"
            - S3: "s3://bucket/key?access_key_id=KEY&secret_access_key=SECRET"
            - Google Cloud Storage: "gs://bucket/path?hmac_key=KEY&hmac_secret=SECRET"
            - Azure Blob Storage: "az://container/blob?account_name=NAME&account_key=KEY"
            - HDFS: "hdfs://namenode:port/path"
            - HTTP/HTTPS: "https://example.com/data.json"
            - MySQL: "mysql://user:pass@host:port/database/table"
            - PostgreSQL: "postgresql://user:pass@host:port/database/table"
            - MongoDB: "mongodb://user:pass@host:port/database.collection"
            - SQLite: "sqlite:///path/to/db.db?table=tablename"
            - Redis: "redis://host:port/db?key=mykey&password=pass"
            - ClickHouse: "clickhouse://host:port/database/table?user=USER&password=PASS"
            - Iceberg: "iceberg://catalog/namespace/table"
            - Delta Lake: "deltalake:///path/to/table"
            - Hudi: "hudi:///path/to/table"

        Examples:
            >>> # Simple local file
            >>> ds = DataStore.uri("/path/to/data.csv")
            >>> ds.connect()
            >>> result = ds.select("*").execute()

            >>> # S3 with auto-detection
            >>> ds = DataStore.uri("s3://mybucket/data.parquet?nosign=true")
            >>> result = ds.select("*").execute()

            >>> # MySQL with full connection string
            >>> ds = DataStore.uri("mysql://root:pass@localhost:3306/mydb/users")
            >>> result = ds.select("name", "email").filter(ds.age > 18).execute()

            >>> # PostgreSQL
            >>> ds = DataStore.uri("postgresql://user:pass@localhost:5432/mydb/products")
            >>> result = ds.select("*").limit(10).execute()

            >>> # Override auto-detected parameters
            >>> ds = DataStore.uri("s3://bucket/data.csv", format="CSV")
        """
        # Parse the URI to get source_type and connection parameters
        source_type, parsed_kwargs = parse_uri(uri)

        # Merge parsed kwargs with user-provided kwargs (user kwargs take precedence)
        final_kwargs = {**parsed_kwargs, **kwargs}

        # Extract table name if present (for database sources)
        table = final_kwargs.pop('table', None)

        # Create and return DataStore
        return cls(source=source_type, table=table, **final_kwargs)

    # ========== Data Source Operations ==========

    def with_format_settings(self, **settings) -> 'DataStore':
        """
        Add format-specific settings for table functions.

        Args:
            **settings: Format settings (e.g., format_csv_delimiter='|',
                       input_format_parquet_filter_push_down=1, etc.)

        Example:
            >>> ds = DataStore("file", path="data.csv", format="CSV")
            >>> ds.with_format_settings(
            ...     format_csv_delimiter='|',
            ...     input_format_csv_skip_first_lines=1,
            ...     input_format_csv_trim_whitespaces=1
            ... )

        Returns:
            New DataStore with settings applied (immutable)
        """
        from copy import copy, deepcopy

        # Create a copy (immutable pattern)
        new_ds = copy(self)
        new_ds._format_settings.update(settings)
        if new_ds._table_function:
            # Deep copy table_function to avoid modifying shared state
            new_ds._table_function = deepcopy(self._table_function)
            new_ds._table_function.with_settings(**settings)
        return new_ds

    def connect(self, test_connection: bool = True) -> 'DataStore':
        """
        Connect to the data source using chdb.

        Args:
            test_connection: If True, verify data source accessibility by
                executing a test query (SELECT 1). Default is True.

        Returns:
            self for chaining

        Raises:
            ConnectionError: If connection fails or data source is not accessible
        """
        self._logger.debug("Connecting to data source (database=%s)...", self.database)

        if self._connection is None:
            # When using table functions, don't pass table function params to connection
            # Only pass database parameter
            if self._table_function is not None:
                self._connection = Connection(self.database)
                self._logger.debug("Created connection for table function")
            else:
                self._connection = Connection(self.database, **self.connection_params)
                self._logger.debug("Created connection with params: %s", self.connection_params)

        try:
            self._connection.connect()
            self._executor = Executor(self._connection)
            self._logger.debug("Connection established successfully")

            # Try to get schema if table exists
            if self.table_name:
                self._discover_schema()
                self._logger.debug("Schema discovered: %s", self._schema)

            # Test data source accessibility for table functions
            if self._table_function is not None and test_connection:
                self._test_data_source()

            return self
        except Exception as e:
            self._logger.error("Connection failed: %s", e)
            raise ConnectionError(f"Failed to connect: {e}")

    def _test_data_source(self) -> None:
        """
        Test if the data source is accessible by executing a simple query.

        For table functions (file, S3, MySQL, etc.), this executes
        SELECT 1 FROM <table_function> LIMIT 1 to verify accessibility.

        Raises:
            ConnectionError: If data source is not accessible
        """
        if self._table_function is None or self._executor is None:
            return

        try:
            table_source = self._table_function.to_sql()
            test_sql = f"SELECT 1 FROM {table_source} LIMIT 1"
            self._logger.debug("Testing data source accessibility: %s", test_sql)
            self._executor.execute(test_sql)
            self._logger.debug("Data source is accessible")
        except Exception as e:
            self._logger.error("Data source not accessible: %s", e)
            raise ConnectionError(f"Data source not accessible: {e}")

    def schema(self) -> dict:
        """
        Get the schema of the data source.

        For table functions, this executes DESCRIBE to discover column types.
        Requires connect() to be called first.

        Returns:
            Dict mapping column names to their types

        Example:
            >>> ds = DataStore.from_file('data.csv')
            >>> ds.connect()
            >>> ds.schema()
            {'id': 'Int64', 'name': 'String', 'age': 'Int64'}
        """
        if self._schema:
            return self._schema

        if self._executor is None:
            self.connect(test_connection=False)

        if self._table_function is not None:
            self._discover_table_function_schema()
        elif self.table_name:
            self._discover_schema()

        return self._schema

    def _discover_table_function_schema(self) -> None:
        """
        Discover schema for table functions and verify data source accessibility.

        For table functions (file, S3, MySQL, etc.), this executes
        DESCRIBE <table_function> to both verify accessibility and get schema.

        Raises:
            ConnectionError: If data source is not accessible
        """
        if self._table_function is None or self._executor is None:
            return

        try:
            # Use DESCRIBE to both test connectivity and get schema
            table_source = self._table_function.to_sql()
            describe_sql = f"DESCRIBE {table_source}"

            # Apply format settings if present (important for custom delimiters, etc.)
            if self._format_settings:
                settings_parts = []
                for key, value in self._format_settings.items():
                    if isinstance(value, str):
                        settings_parts.append(f"{key}='{value}'")
                    else:
                        settings_parts.append(f"{key}={value}")
                if settings_parts:
                    describe_sql += " SETTINGS " + ", ".join(settings_parts)

            self._logger.debug("Discovering schema for table function: %s", describe_sql)
            result = self._executor.execute(describe_sql)

            # Build schema dictionary from DESCRIBE result
            self._schema = {}
            for row in result.rows:
                # ClickHouse DESCRIBE returns: (name, type, default_type, default_expression, comment, ...)
                col_name = row[0]
                col_type = row[1]
                self._schema[col_name] = col_type

            self._logger.debug("Data source is accessible, schema discovered: %s", self._schema)
        except Exception as e:
            self._logger.error("Data source not accessible: %s", e)
            raise ConnectionError(f"Data source not accessible: {e}")

    def _get_table_alias(self) -> str:
        """
        Get a short alias for table functions.

        For file table functions, extracts filename without extension.
        For other table functions, uses table name or a generic name.
        """
        if self._table_function and hasattr(self._table_function, 'params'):
            # Try to get a meaningful alias from path
            path = self._table_function.params.get('path')
            if path:
                import os

                # Extract filename without extension
                basename = os.path.basename(path)
                name_without_ext = os.path.splitext(basename)[0]
                return name_without_ext

            # For other table functions, try to use table name
            table = self._table_function.params.get('table')
            if table:
                return table

            # For numbers or other generators
            if hasattr(self._table_function, '__class__'):
                class_name = self._table_function.__class__.__name__.replace('TableFunction', '').lower()
                return class_name

        # Fallback to table name or generic
        return self.table_name if self.table_name else 'tbl'

    def _ensure_sql_source(self) -> bool:
        """
        Ensure we have a SQL source (table function or table name) for SQL operations.

        If this DataStore was created from a DataFrame via from_df() and doesn't
        have a table function yet, creates a PythonTableFunction on-demand.

        Returns:
            True if SQL source is available, False otherwise
        """
        # Already have SQL source
        if self._table_function or self.table_name:
            return True

        # Check if we have a cached DataFrame from from_df()
        if self._source_df is not None:
            from .table_functions import PythonTableFunction
            from .lazy_ops import LazyDataFrameSource

            # Create PythonTableFunction on-demand
            self._table_function = PythonTableFunction(df=self._source_df, name=self._source_df_name)
            # Remove LazyDataFrameSource from lazy_ops (will use SQL instead)
            # but keep other ops like WHERE, ORDER BY, etc.
            self._lazy_ops = [op for op in self._lazy_ops if not isinstance(op, LazyDataFrameSource)]
            return True

        return False

    def _discover_schema(self):
        """Discover table schema from chdb."""
        if not self._executor or not self.table_name:
            return

        try:
            # Query system tables for schema info
            sql = f"DESCRIBE TABLE {format_identifier(self.table_name, self.quote_char)}"
            result = self._executor.execute(sql)

            # Build schema dictionary
            self._schema = {}
            for row in result.rows:
                # ClickHouse DESCRIBE returns: (name, type, default_type, default_expression, comment, ...)
                col_name = row[0]
                col_type = row[1]
                self._schema[col_name] = col_type
        except Exception:
            # Table might not exist yet, that's ok
            self._schema = {}

    def _get_all_column_names(self) -> List[str]:
        """
        Get all column names for the current data source.

        Used for LazyWhere/LazyMask SQL pushdown (CASE WHEN generation).
        Tries multiple sources:
        1. Cached schema (_schema with real types)
        2. DESCRIBE query on table function
        3. LIMIT 0 query to get column names (without types)

        Returns:
            List of column names, or empty list if unavailable
        """
        # Try cached schema first (only if it has real types)
        if self._schema and not all(v == 'Unknown' for v in self._schema.values()):
            return list(self._schema.keys())

        # Try to discover schema
        if self._executor is None:
            self.connect()

        # For table functions - use DESCRIBE to get types
        if self._table_function:
            try:
                table_source = self._table_function.to_sql()
                describe_sql = f"DESCRIBE {table_source}"
                result = self._executor.execute(describe_sql)
                # DESCRIBE returns: (name, type, default_type, default_expression, ...)
                self._schema = {}
                columns = []
                for row in result.rows:
                    col_name = row[0]
                    col_type = row[1] if len(row) > 1 else 'Unknown'
                    columns.append(col_name)
                    self._schema[col_name] = col_type
                self._logger.debug("Schema discovered via DESCRIBE: %s", self._schema)
                return columns
            except Exception as e:
                self._logger.debug("DESCRIBE failed: %s, trying LIMIT 0", e)

        # For regular tables - use DESCRIBE TABLE
        elif self.table_name:
            try:
                describe_sql = f"DESCRIBE TABLE {format_identifier(self.table_name, self.quote_char)}"
                result = self._executor.execute(describe_sql)
                self._schema = {}
                columns = []
                for row in result.rows:
                    col_name = row[0]
                    col_type = row[1] if len(row) > 1 else 'Unknown'
                    columns.append(col_name)
                    self._schema[col_name] = col_type
                return columns
            except Exception as e:
                self._logger.debug("DESCRIBE TABLE failed: %s", e)

        # Fallback: LIMIT 0 query (no type info)
        try:
            if self._table_function:
                table_source = self._table_function.to_sql()
            elif self.table_name:
                table_source = format_identifier(self.table_name, self.quote_char)
            else:
                return []

            sql = f"SELECT * FROM {table_source} LIMIT 0"
            result = self._executor.execute(sql)
            df = result.to_df()
            columns = list(df.columns)
            # Cache for future use (without type info)
            self._schema = {col: 'Unknown' for col in columns}
            return columns
        except Exception as e:
            self._logger.debug("LIMIT 0 query failed: %s", e)
            return []

    def execute(self) -> QueryResult:
        """
        Execute the query and return results.

        Returns:
            QueryResult object with data and metadata
        """
        self._logger.debug("=" * 70)
        self._logger.debug("DataStore.execute() called")
        self._logger.debug("=" * 70)

        # Ensure we're connected
        if self._executor is None:
            self._logger.debug("Not connected, establishing connection...")
            self.connect()

        # Generate SQL
        sql = self.to_sql()
        self._logger.debug("Generated SQL for execution")

        try:
            result = self._executor.execute(sql)
            self._logger.debug("Query executed successfully")
            return result
        except Exception as e:
            self._logger.error("Query execution failed: %s", e)
            raise ExecutionError(f"Query execution failed: {e}")

    def exec(self) -> QueryResult:
        """
        Execute the query and return results. Alias for execute().

        Returns:
            QueryResult object with data and metadata
        """
        return self.execute()

    def to_df(self):
        """
        Execute all operations and return pandas DataFrame.

        This triggers execution of all lazy operations.

        If the DataStore has been executed (pandas operations applied), returns
        the cached DataFrame. Otherwise, executes the SQL query and lazy operations.

        Returns:
            pandas DataFrame

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> df = ds.select("*").filter(ds.age > 18).to_df()

            >>> # After lazy operations, uses cached result
            >>> ds["new_col"] = ds["age"] * 2
            >>> df2 = ds.to_df()  # Executes SQL + lazy ops
        """
        self._logger.debug("to_df() called - triggering execution")
        return self._execute()

    def to_pandas(self) -> pd.DataFrame:
        """
        Execute all operations and return pandas DataFrame.

        Alias for to_df() to provide API consistency with Polars, Dask, and other
        DataFrame libraries that use to_pandas() for conversion.

        Returns:
            pandas DataFrame

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> df = ds.select("*").filter(ds.age > 18).to_pandas()
        """
        return self.to_df()

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Execute the query and return results as a list of dictionaries.
        Convenience method that combines execute() and to_dict().

        If the DataStore has been executed (pandas operations applied), converts
        the cached DataFrame to dict. Otherwise, executes the SQL query.

        Returns:
            List of dicts where keys are column names

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> records = ds.select("*").filter(ds.age > 18).to_dict()
        """
        # Execute and convert to dict
        return self._execute().to_dict('records')

    def _wrap_result_fallback(self, result_df):
        """
        Fallback method to wrap a DataFrame result into a new DataStore.
        This is used when _wrap_result is not available (shouldn't happen normally).

        Args:
            result_df: pandas DataFrame to wrap

        Returns:
            DataStore with the result DataFrame
        """
        # Create a new DataStore and set up lazy op to return this DataFrame
        from .lazy_ops import LazyDataFrameSource

        new_ds = copy(self)
        # Clear SQL state and lazy ops, replace with DataFrame source
        new_ds._select_fields = []
        new_ds._where_condition = None
        new_ds._joins = []
        new_ds._groupby_fields = []
        new_ds._having_condition = None
        new_ds._orderby_fields = []
        new_ds._limit_value = None
        new_ds._offset_value = None
        new_ds._distinct = False
        new_ds._lazy_ops = [LazyDataFrameSource(result_df)]

        # Reset cache state for the new DataStore
        new_ds._cached_result = None
        new_ds._cache_version = 0
        new_ds._cached_at_version = -1
        new_ds._cache_timestamp = None
        new_ds._table_function = None
        new_ds.table_name = None
        return new_ds

    def describe(self, percentiles=None, include=None, exclude=None):
        """
        Generate descriptive statistics of the data.
        Convenience method that combines execute(), to_df(), and describe().

        Works correctly with both SQL queries and executed DataFrames.

        Args:
            percentiles: List of percentiles to include (default: [.25, .5, .75])
            include: Data types to include ('all', None, or list of dtypes)
            exclude: Data types to exclude (None or list of dtypes)

        Returns:
            DataStore with descriptive statistics

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> stats = ds.select("*").describe()
            >>> # With custom percentiles
            >>> stats = ds.describe(percentiles=[.1, .5, .9])
        """
        # Use pandas compat layer if available, otherwise fallback
        if hasattr(self, '_get_df'):
            df = self._get_df()
        else:
            df = self.to_df()
        result_df = df.describe(percentiles=percentiles, include=include, exclude=exclude)

        # Wrap result in DataStore
        if hasattr(self, '_wrap_result'):
            return self._wrap_result(result_df, 'describe()')
        else:
            return self._wrap_result_fallback(result_df)

    def desc(self, percentiles=None, include=None, exclude=None):
        """
        Shortcut for describe().
        Generate descriptive statistics of the data.

        Args:
            percentiles: List of percentiles to include (default: [.25, .5, .75])
            include: Data types to include ('all', None, or list of dtypes)
            exclude: Data types to exclude (None or list of dtypes)

        Returns:
            DataStore with descriptive statistics

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> stats = ds.select("*").desc()
        """
        return self.describe(percentiles=percentiles, include=include, exclude=exclude)

    def head(self, n: int = 5):
        """
        Return the first n rows of the query result.
        Convenience method that applies limit and returns as DataStore.

        This method uses lazy execution - the LIMIT is added to the SQL query
        and only executed when the result is executed (e.g., via to_df() or print).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            DataStore with first n rows (lazy - not yet executed)

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> first_rows = ds.select("*").head()  # Lazy
            >>> first_10 = ds.head(10).to_df()      # Executes here
        """
        # Use limit() which adds a lazy LazyRelationalOp
        # This allows head() to be chained with other operations
        # and merged into a single SQL query
        return self.limit(n)

    def tail(self, n: int = 5):
        """
        Return the last n rows of the query result.
        Convenience method that returns as DataStore with reversed order.

        Works correctly with both SQL queries and executed DataFrames.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            DataStore with last n rows

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> last_rows = ds.select("*").tail()
            >>> last_10 = ds.select("*").tail(10)
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, '_get_df'):
            df = self._get_df()
        else:
            df = self.to_df()
        result_df = df.tail(n)

        # Wrap result in DataStore
        if hasattr(self, '_wrap_result'):
            return self._wrap_result(result_df, f'tail({n})')
        else:
            return self._wrap_result_fallback(result_df)

    def sample(self, n: int = None, frac: float = None, random_state: int = None):
        """
        Return a random sample of rows from the query result.

        Works correctly with both SQL queries and executed DataFrames.

        Args:
            n: Number of rows to return (mutually exclusive with frac)
            frac: Fraction of rows to return (mutually exclusive with n)
            random_state: Random seed for reproducibility

        Returns:
            DataStore with sampled rows

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> sample_10 = ds.select("*").sample(n=10)
            >>> sample_20_percent = ds.select("*").sample(frac=0.2)
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, '_get_df'):
            df = self._get_df()
        else:
            df = self.to_df()
        result_df = df.sample(n=n, frac=frac, random_state=random_state)

        # Wrap result in DataStore
        sample_desc = f'sample(n={n})' if n is not None else f'sample(frac={frac})'
        if hasattr(self, '_wrap_result'):
            return self._wrap_result(result_df, sample_desc)
        else:
            return self._wrap_result_fallback(result_df)

    @property
    def shape(self):
        """
        Return the shape (rows, columns) of the query result.

        Works correctly with both SQL queries and executed DataFrames.

        Returns:
            Tuple of (rows, columns)

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> rows, cols = ds.select("*").shape
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, '_get_df'):
            df = self._get_df()
        else:
            df = self.to_df()
        return df.shape

    @property
    def columns(self):
        """
        Return the column names of the query result.

        Works correctly with both SQL queries and executed DataFrames.

        Returns:
            pandas Index of column names

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> cols = ds.select("*").columns
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, '_get_df'):
            df = self._get_df()
        else:
            df = self.to_df()
        return df.columns

    @columns.setter
    def columns(self, new_columns):
        """
        Rename all columns by providing a new list of column names.

        This is equivalent to pandas DataFrame.columns setter.
        Internally uses rename() to create a lazy rename operation.

        Args:
            new_columns: List of new column names (must match current column count)

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> ds.columns = ['new_col1', 'new_col2', 'new_col3']

        Raises:
            ValueError: If number of new columns doesn't match current columns
        """
        from .lazy_ops import LazyRenameColumns

        current_columns = list(self.columns)
        new_columns = list(new_columns)

        if len(new_columns) != len(current_columns):
            raise ValueError(f"Length mismatch: Expected {len(current_columns)} columns, " f"got {len(new_columns)}")

        # Build rename mapping
        rename_mapping = {
            old: new for old, new in zip(current_columns, new_columns) if old != new  # Only rename if different
        }

        if rename_mapping:
            self._add_lazy_op(LazyRenameColumns(rename_mapping))

    def count(self):
        """
        Count non-null values for each column in the query result.

        This method uses SQL COUNT(column) to efficiently count non-null values
        without executing the entire DataFrame, making it suitable for large
        datasets. The COUNT is pushed down to chDB for optimal performance.

        Returns:
            pandas Series with counts per column

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> counts = ds.select("*").count()
            >>> print(counts['name'])  # Non-null count for 'name' column

        Note:
            For total row count (like SQL COUNT(*)), use count_rows() instead.
            Falls back to DataFrame execution if non-SQL operations are pending.
        """
        # Check if there are any non-SQL operations in the lazy ops
        has_non_sql_ops = False
        if self._lazy_ops:
            from .lazy_ops import LazyRelationalOp

            for op in self._lazy_ops:
                if not isinstance(op, LazyRelationalOp):
                    has_non_sql_ops = True
                    break

        if has_non_sql_ops:
            # Fall back to execution for non-SQL operations
            self._logger.debug("count() falling back to execution due to non-SQL operations")
            if hasattr(self, '_get_df'):
                df = self._get_df()
            else:
                df = self.to_df()
            return df.count()

        # Ensure we're connected
        if self._executor is None:
            self.connect()

        # Step 1: Get column names efficiently using LIMIT 1
        # Note: LIMIT 0 doesn't return column info in chDB, so we use LIMIT 1
        from copy import copy

        schema_ds = copy(self)
        schema_ds._limit_value = 1
        schema_ds._offset_value = None
        schema_sql = schema_ds.to_sql()
        self._logger.debug("count() getting column names with: %s", schema_sql)

        try:
            schema_result = self._executor.execute(schema_sql)
            column_names = schema_result.column_names

            if not column_names:
                # No columns found, return empty Series
                return pd.Series(dtype='int64')

            # Step 2: Build COUNT query for each column
            # SELECT COUNT(col1) AS col1, COUNT(col2) AS col2, ... FROM (subquery)
            count_exprs = []
            for col_name in column_names:
                # Use format_identifier for proper quoting
                quoted_col = format_identifier(col_name, self.quote_char)
                count_exprs.append(f"COUNT({quoted_col}) AS {quoted_col}")

            count_select = ", ".join(count_exprs)

            # Build subquery from current state (without LIMIT 0)
            count_ds = copy(self)
            count_ds._limit_value = None
            count_ds._offset_value = None
            subquery_sql = count_ds.to_sql()

            count_sql = f"SELECT {count_select} FROM ({subquery_sql})"
            self._logger.debug("count() executing: %s", count_sql)

            result = self._executor.execute(count_sql)

            if result.rows and result.rows[0]:
                # Build Series from result
                counts = {col: int(val) for col, val in zip(column_names, result.rows[0])}
                return pd.Series(counts)
            else:
                # Return Series with zeros
                return pd.Series({col: 0 for col in column_names})

        except Exception as e:
            # Fall back to execution on any error
            self._logger.debug("count() falling back to execution due to error: %s", e)
            if hasattr(self, '_get_df'):
                df = self._get_df()
            else:
                df = self.to_df()
            return df.count()

    def count_rows(self) -> int:
        """
        Count total number of rows using SQL COUNT(*).

        This method efficiently executes COUNT(*) without executing the DataFrame,
        making it suitable for large datasets. Unlike count() which returns per-column
        non-null counts, this returns the total row count.

        If LIMIT is applied (e.g., via head() or limit()), executes the query
        with LIMIT and returns the actual row count.

        Returns:
            int: Total number of rows

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> total = ds.select("*").filter(ds.age > 18).count_rows()
            >>> print(f"Found {total} rows")
            >>> limited = ds.head(10).count_rows()  # Executes with LIMIT, returns actual count

        Note:
            This is more efficient than len() for large datasets as it uses SQL COUNT(*)
            instead of executing the entire DataFrame.
        """
        from .functions import Count

        # If we have lazy operations that can't be expressed in SQL, fall back to execution
        # Check if there are any non-SQL operations in the lazy ops
        has_non_sql_ops = False
        if self._lazy_ops:
            from .lazy_ops import LazyRelationalOp

            for op in self._lazy_ops:
                if not isinstance(op, LazyRelationalOp):
                    has_non_sql_ops = True
                    break

        if has_non_sql_ops:
            # Fall back to execution for non-SQL operations
            self._logger.debug("count_rows() falling back to execution due to non-SQL operations")
            return len(self._execute())

        # If LIMIT is applied, execute with LIMIT and return actual count
        # This is more accurate than COUNT(*) without LIMIT
        if self._limit_value is not None:
            self._logger.debug("count_rows() executing due to LIMIT")
            return len(self._execute())

        # Build a COUNT(*) query
        # Create a copy of the current DataStore to modify for counting
        from copy import copy

        count_ds = copy(self)

        # Replace SELECT fields with COUNT(*)
        count_ds._select_fields = [Count('*')]
        count_ds._select_star = False  # Must clear _select_star to avoid "SELECT *, COUNT(*)"

        # Clear ORDER BY (not needed for COUNT)
        count_ds._orderby_fields = []

        # Clear OFFSET (not applicable for COUNT without LIMIT)
        count_ds._offset_value = None

        # Clear DISTINCT (COUNT(*) counts all rows)
        count_ds._distinct = False

        # Ensure we're connected
        if count_ds._executor is None:
            count_ds.connect()

        # Execute the COUNT query
        sql = count_ds.to_sql()
        self._logger.debug("count_rows() executing SQL: %s", sql)
        result = count_ds._executor.execute(sql)

        # Extract the count value from result
        if result.rows:
            return int(result.rows[0][0])
        return 0

    def info(self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None):
        """
        Print concise summary of the query result.

        Works correctly with both SQL queries and executed DataFrames.

        Args:
            verbose: Whether to print full summary
            buf: Buffer to write to
            max_cols: Maximum number of columns to show
            memory_usage: Whether to show memory usage
            show_counts: Whether to show non-null counts

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> ds.select("*").info()
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, '_get_df'):
            df = self._get_df()
        else:
            df = self.to_df()
        return df.info(verbose=verbose, buf=buf, max_cols=max_cols, memory_usage=memory_usage, show_counts=show_counts)

    def create_table(self, schema: Dict[str, str], engine: str = "Memory", drop_if_exists: bool = False) -> 'DataStore':
        """
        Create a table in chdb.

        Args:
            schema: Dictionary of column_name -> column_type
            engine: ClickHouse table engine (default: Memory for in-memory)
            drop_if_exists: If True, drop the table first if it exists (useful for tests)

        Example:
            >>> ds = DataStore(table="users")
            >>> ds.connect()
            >>> ds.create_table({"id": "UInt64", "name": "String", "age": "UInt8"})

        Returns:
            self for chaining
        """
        if not self.table_name:
            raise ValueError("Table name required to create table")

        if self._executor is None:
            self.connect()

        # Drop table first if requested
        if drop_if_exists:
            drop_sql = f"DROP TABLE IF EXISTS {format_identifier(self.table_name, self.quote_char)}"
            self._executor.execute(drop_sql)

        # Build CREATE TABLE statement
        columns = ", ".join([f"{format_identifier(name, self.quote_char)} {dtype}" for name, dtype in schema.items()])

        table_ident = format_identifier(self.table_name, self.quote_char)
        sql = f"CREATE TABLE IF NOT EXISTS {table_ident} ({columns}) ENGINE = {engine}"

        self._executor.execute(sql)
        self._schema = schema

        return self

    def insert(self, data: List[Dict[str, Any]] = None, **columns) -> 'DataStore':
        """
        Insert data into the table (executes immediately).

        Args:
            data: List of dictionaries with column_name -> value
            **columns: Alternative way to specify columns (for single row)

        Example:
            >>> ds.insert([{"id": 1, "name": "Alice", "age": 25}])
            >>> ds.insert(id=1, name="Alice", age=25)

        Returns:
            self for chaining
        """
        if not self.table_name:
            raise ValueError("Table name required to insert data")

        # Handle single row via keyword arguments
        if columns and not data:
            data = [columns]

        if not data:
            return self

        if self._executor is None:
            self.connect()

        # Get column names from first row
        columns = list(data[0].keys())
        columns_sql = ", ".join([format_identifier(col, self.quote_char) for col in columns])

        # Build values
        values_list = []
        for row in data:
            values = []
            for col in columns:
                val = row.get(col)
                if val is None:
                    values.append("NULL")
                elif isinstance(val, str):
                    # Escape single quotes
                    escaped = val.replace("'", "''")
                    values.append(f"'{escaped}'")
                elif isinstance(val, bool):
                    values.append("1" if val else "0")
                else:
                    values.append(str(val))
            values_list.append(f"({', '.join(values)})")

        values_sql = ", ".join(values_list)
        sql = f"INSERT INTO {format_identifier(self.table_name, self.quote_char)} ({columns_sql}) VALUES {values_sql}"

        self._executor.execute(sql)

        return self

    def close(self):
        """Close the connection."""
        if self._executor:
            self._executor.close()
            self._executor = None
        if self._connection:
            self._connection.close()
            self._connection = None

    # ========== INSERT/UPDATE/DELETE Query Building ==========

    @immutable
    def insert_into(self, *columns: str) -> 'DataStore':
        """
        Start building an INSERT query (ClickHouse style).

        Args:
            *columns: Column names to insert into

        Example:
            >>> ds.insert_into('id', 'name', 'age').values(1, 'Alice', 25)
            >>> ds.insert_into('id', 'name').select_from(other_ds.select('id', 'name'))

        Returns:
            DataStore with INSERT query state
        """
        self._insert_columns = list(columns)

    @immutable
    def insert_values(self, *rows) -> 'DataStore':
        """
        Add VALUES clause to INSERT query.

        Args:
            *rows: Each row can be a tuple/list or individual values

        Example:
            >>> ds.insert_into('id', 'name').insert_values((1, 'Alice'), (2, 'Bob'))
            >>> ds.insert_into('id', 'name').insert_values(1, 'Alice').insert_values(2, 'Bob')

        Returns:
            DataStore for chaining
        """
        if not self._insert_columns:
            raise QueryError("Must call insert_into() before insert_values()")

        # Handle different input formats
        if len(rows) == 1 and isinstance(rows[0], (list, tuple)):
            # Single row as tuple: values((1, 'Alice'))
            self._insert_values.append(list(rows[0]))
        elif len(rows) > 1 and all(isinstance(r, (list, tuple)) for r in rows):
            # Multiple rows: values((1, 'Alice'), (2, 'Bob'))
            for row in rows:
                self._insert_values.append(list(row))
        else:
            # Individual values: insert_values(1, 'Alice')
            self._insert_values.append(list(rows))

    @immutable
    def select_from(self, subquery: 'DataStore') -> 'DataStore':
        """
        Add SELECT subquery to INSERT query (INSERT INTO ... SELECT ...).

        Args:
            subquery: DataStore representing the SELECT query

        Example:
            >>> ds.insert_into('id', 'name').select_from(
            ...     other_ds.select('user_id', 'user_name').filter(other_ds.active == True)
            ... )

        Returns:
            DataStore for chaining
        """
        if not self._insert_columns:
            raise QueryError("Must call insert_into() before select_from()")

        self._insert_select = subquery

    @immutable
    def update_set(self, **fields) -> 'DataStore':
        """
        Build an UPDATE query (ClickHouse style: ALTER TABLE ... UPDATE ...).

        Args:
            **fields: Field-value pairs to update

        Example:
            >>> ds.update_set(age=26, city='NYC').filter(ds.id == 1)
            >>> # Generates: ALTER TABLE table UPDATE age=26, city='NYC' WHERE id=1

        Returns:
            DataStore for chaining
        """
        for field_name, value in fields.items():
            self._update_fields.append((field_name, value))

    @immutable
    def delete_rows(self) -> 'DataStore':
        """
        Build a DELETE query (ClickHouse style: ALTER TABLE ... DELETE).

        Example:
            >>> ds.delete_rows().filter(ds.age < 18)
            >>> # Generates: ALTER TABLE table DELETE WHERE age < 18

        Returns:
            DataStore for chaining
        """
        self._delete_flag = True

    # ========== Query Building Methods ==========

    @immutable
    def select(self, *fields: Union[str, Expression]) -> 'DataStore':
        """
        Select specific columns.

        If DataStore is executed (pandas operations applied), selects columns
        from cached DataFrame. Otherwise, builds SQL SELECT clause.

        Args:
            *fields: Column names (strings) or Expression objects

        Example:
            >>> ds.select("name", "age")
            >>> ds.select(ds.name, ds.age + 1)
            >>> ds.select("*", ds.name.str.upper().as_("name_upper"))  # All columns + computed
        """
        from .column_expr import ColumnExpr
        from .functions import Function

        # Check if any field is a SQL expression (Function, Expression, ColumnExpr with expression)
        has_sql_expr = any(
            isinstance(f, (Function, Expression))
            or (isinstance(f, ColumnExpr) and isinstance(f._expr, (Function, Expression)))
            for f in fields
            if not isinstance(f, str)
        )

        if has_sql_expr:
            # Ensure we have a SQL source (create PythonTableFunction if needed)
            self._ensure_sql_source()

        # Track operation
        field_names = ', '.join([str(f) for f in fields]) if fields else '*'

        # Build SQL SELECT (lazy)
        self._track_operation('sql', f"SELECT {field_names}", {'lazy': True})

        # Record in lazy ops for correct execution order in explain()
        # Store fields for DataFrame execution
        self._add_lazy_op(LazyRelationalOp('SELECT', field_names, fields=list(fields)))

        for field in fields:
            if isinstance(field, str):
                # Special case: "*" means SELECT * (all existing columns)
                if field == "*":
                    self._select_star = True
                    # Clear existing fields - will render as * in SQL
                    self._select_fields = []
                    continue  # Continue processing other fields (e.g., computed columns)
                # Don't add table prefix for string fields - user's explicit choice
                field = Field(field)
            # Handle ColumnExpr - unwrap to get the underlying expression
            elif isinstance(field, ColumnExpr):
                field = field._expr

            self._select_fields.append(field)

        return self

    @immutable
    def filter(
        self, condition: Union[Condition, str, 'ColumnExpr', None] = None, items=None, like=None, regex=None, axis=None
    ) -> 'DataStore':
        """
        Filter rows or columns.

        This method supports two modes:
        1. SQL-style row filtering (when condition is provided)
        2. Pandas-style column selection (when items/like/regex is provided)

        Args:
            condition: Condition object, ColumnExpr (boolean expression), or SQL string
                       for row filtering
            items: List of column names to select (pandas-style)
            like: Keep columns where column name contains this string (pandas-style)
            regex: Keep columns where column name matches this regex (pandas-style)
            axis: Axis to filter on (0 for rows, 1 for columns). Default None.
                  For pandas compatibility, axis=1 uses items/like/regex.

        Example:
            >>> # SQL-style row filtering
            >>> ds.filter(ds.age > 18)
            >>> ds.filter((ds.age > 18) & (ds.city == 'NYC'))
            >>> ds.filter(ds['email'].isnull())

            >>> # Pandas-style column selection
            >>> ds.filter(items=['a', 'b'])
            >>> ds.filter(like='name')
            >>> ds.filter(regex='^col_')
        """
        import re
        from .column_expr import ColumnExpr
        from .conditions import BinaryCondition
        from .expressions import Literal

        # Check if pandas-style column filtering is requested
        if items is not None or like is not None or regex is not None:
            # Pandas-style: select columns by name
            df = self._execute()
            cols = df.columns.tolist()

            if items is not None:
                # Select specific columns
                selected = [c for c in items if c in cols]
            elif like is not None:
                # Select columns containing substring
                selected = [c for c in cols if like in c]
            elif regex is not None:
                # Select columns matching regex
                pattern = re.compile(regex)
                selected = [c for c in cols if pattern.search(c)]
            else:
                selected = cols

            # Return DataStore with selected columns
            return self[selected] if selected else self

        # SQL-style row filtering
        if condition is None:
            raise ValueError(
                "Must provide either 'condition' for row filtering or 'items'/'like'/'regex' for column selection"
            )

        # Handle LazyCondition (e.g., from isin(), between()) - extract underlying Condition
        from .lazy_result import LazyCondition

        if isinstance(condition, LazyCondition):
            # Extract the underlying Condition for SQL generation
            condition = condition.condition

        # Unified handling: extract Condition from ColumnExpr._expr
        # Since comparison operators now return ColumnExpr wrapping Condition,
        # we just need to check if _expr is already a Condition.
        if isinstance(condition, ColumnExpr):
            if isinstance(condition._expr, Condition):
                # ColumnExpr wrapping Condition (e.g., ds['col'] > 5)
                condition = condition._expr
            else:
                # Non-condition ColumnExpr (e.g., boolean function result)
                # Convert to truthy check: expr = 1
                condition = BinaryCondition('=', condition._expr, Literal(1))

        # Convert condition to string for tracking
        if isinstance(condition, str):
            condition_str = condition
        else:
            condition_str = condition.to_sql(quote_char=self.quote_char)

        # Build SQL WHERE clause (lazy)
        self._track_operation('sql', f"WHERE {condition_str}", {'lazy': True})

        # Record in lazy ops for correct execution order in explain()
        # Store condition object for DataFrame execution
        self._add_lazy_op(LazyRelationalOp('WHERE', condition_str, condition=condition))

        if isinstance(condition, str):
            # TODO: Parse string conditions
            raise NotImplementedError("String conditions not yet implemented")

        if self._where_condition is None:
            self._where_condition = condition
        else:
            # Combine with existing condition using AND
            self._where_condition = self._where_condition & condition

        return self

    def where(self, condition: Union[Condition, str], other=_MISSING, **kwargs) -> 'DataStore':
        """
        Filter rows or replace values conditionally.

        This method handles both:
        1. SQL-style WHERE clause: When condition is a simple Condition/str AND no other args
        2. Pandas-style where: Conditional replacement (default behavior for ColumnExpr)

        Args:
            condition: Condition object, SQL string, or pandas condition
            other: Value to replace where condition is False (pandas-style)
                   Default is NaN (pandas behavior), use _MISSING sentinel to detect
            **kwargs: Additional pandas where() arguments

        Example:
            >>> # SQL-style (simple Condition, no other args)
            >>> ds.filter(ds.age > 18)  # Prefer explicit filter() for clarity
            >>>
            >>> # Pandas-style (value replacement)
            >>> ds.where(ds['age'] > 18, 0)  # Replace False values with 0
            >>> ds.where(ds['age'] > 18)     # Replace False values with NaN
        """
        from .column_expr import ColumnExpr

        # Pandas-style where:
        # - ColumnExpr condition (always pandas-style)
        # - Explicit other value provided
        # - Any kwargs provided
        is_pandas_style = isinstance(condition, ColumnExpr) or other is not _MISSING or kwargs

        if is_pandas_style:
            # Convert _MISSING to None for pandas compatibility
            actual_other = None if other is _MISSING else other
            # Pandas-style where() - delegate to mixin
            if hasattr(super(), 'where'):
                return super().where(condition, other=actual_other, **kwargs)
            # Fallback if no mixin
            raise NotImplementedError("Pandas-style where() requires pandas compatibility layer")

        # SQL-style filter (simple Condition or string, no other args)
        return self.filter(condition)

    def when(self, condition: Any, value: Any) -> 'CaseWhenBuilder':
        """
        Start building a CASE WHEN expression.

        This method provides a fluent API for creating conditional expressions
        similar to SQL CASE WHEN:

            ds['grade'] = ds.when(ds['score'] >= 90, 'A') \\
                            .when(ds['score'] >= 80, 'B') \\
                            .when(ds['score'] >= 60, 'C') \\
                            .otherwise('F')

        This generates SQL:
            CASE WHEN score >= 90 THEN 'A'
                 WHEN score >= 80 THEN 'B'
                 WHEN score >= 60 THEN 'C'
                 ELSE 'F'
            END

        And can also execute via pandas using np.select().

        Args:
            condition: Boolean condition (e.g., ds['score'] >= 90)
            value: Value to use when condition is True

        Returns:
            CaseWhenBuilder for chaining .when() and .otherwise()

        Note:
            You MUST call .otherwise() to complete the expression.

        Example:
            >>> # Simple grade classification
            >>> ds['grade'] = ds.when(ds['score'] >= 90, 'A') \\
            ...                 .when(ds['score'] >= 80, 'B') \\
            ...                 .otherwise('C')

            >>> # Numeric transformation
            >>> ds['adjusted'] = ds.when(ds['value'] < 0, 0) \\
            ...                    .when(ds['value'] > 100, 100) \\
            ...                    .otherwise(ds['value'])

            >>> # Multiple conditions with expressions
            >>> ds['category'] = ds.when(ds['age'] < 18, 'minor') \\
            ...                    .when(ds['age'] < 65, 'adult') \\
            ...                    .otherwise('senior')
        """
        from .case_when import CaseWhenBuilder

        builder = CaseWhenBuilder(self)
        return builder.when(condition, value)

    @classmethod
    def run_sql(cls, query: str) -> 'DataStore':
        """
        Execute a raw SQL query directly against chDB.

        This is a convenience class method for running SQL queries without
        needing an existing DataStore instance.

        Args:
            query: Full SQL query string

        Returns:
            DataStore with the query result

        Example:
            >>> result = DataStore.run_sql('''
            ...     SELECT city, AVG(salary) as avg_salary
            ...     FROM file('employees.csv', 'CSVWithNames')
            ...     GROUP BY city
            ...     ORDER BY avg_salary DESC
            ... ''')
            >>> print(result.to_df())
        """
        from .lazy_ops import LazySQLQuery

        new_ds = cls()
        # Add the SQL query as a lazy operation
        lazy_op = LazySQLQuery(query, df_alias=None, is_raw_query=True)
        new_ds._lazy_ops = [lazy_op]
        return new_ds

    def sql(self, query: str) -> 'DataStore':
        """
        Execute a SQL query on the current DataFrame using chDB's SQL engine.

        This enables true SQL-Pandas-SQL interleaving within the lazy pipeline.

        Supports two syntaxes:
        1. **Short form** (auto-adds SELECT * FROM __df__):
           - Condition only: `ds.sql("value > 100")`
           - With ORDER BY: `ds.sql("value > 100 ORDER BY id")`
           - Clauses only: `ds.sql("ORDER BY id LIMIT 5")`

        2. **Full SQL form** (when query contains SELECT/FROM/GROUP BY):
           - `ds.sql("SELECT id, SUM(value) FROM __df__ GROUP BY id")`

        Args:
            query: SQL query or condition. Examples:
                   - "value > 100" → becomes SELECT * FROM __df__ WHERE value > 100
                   - "ORDER BY id LIMIT 5" → becomes SELECT * FROM __df__ ORDER BY id LIMIT 5
                   - "SELECT id, name FROM __df__ WHERE age > 20" → used as-is

        Returns:
            A new DataStore with the SQL query result.

        Example:
            >>> ds = DataStore.from_file('users.csv')
            >>> ds = ds.filter(ds.age > 20)
            >>> ds['doubled'] = ds['age'] * 2
            >>>
            >>> # Short form - just the condition! (auto-adds SELECT * FROM __df__ WHERE)
            >>> ds = ds.sql("doubled > 50 ORDER BY age DESC LIMIT 10")
            >>>
            >>> # Continue with pandas operations
            >>> ds = ds.add_prefix('result_')

        Advanced Examples:
            >>> # SQL aggregation (full form needed for GROUP BY)
            >>> ds = ds.sql('''
            ...     SELECT category, COUNT(*) as cnt, SUM(value) as total
            ...     FROM __df__
            ...     GROUP BY category
            ... ''')

            >>> # SQL window functions (full form)
            >>> ds = ds.sql('''
            ...     SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY value DESC) as rank
            ...     FROM __df__
            ... ''')

            >>> # Simple filter and sort (short form)
            >>> ds = ds.sql("age > 18 AND status = 'active' ORDER BY created_at DESC")

        Note:
            - The query is executed via chDB's Python() table function
            - This forces execution of all pending operations before executing the SQL
            - The result is a new DataFrame that can be used for subsequent operations
        """
        from copy import copy

        from .lazy_ops import LazySQLQuery

        # Create a copy of this DataStore (immutable pattern)
        new_ds = copy(self)

        # Record the SQL query operation on the copy
        lazy_op = LazySQLQuery(query, df_alias='__df__')
        new_ds._add_lazy_op(lazy_op)

        return new_ds

    @immutable
    def join(
        self, other: 'DataStore', on=None, how: str = 'inner', left_on: str = None, right_on: str = None
    ) -> 'DataStore':
        """
        Join with another DataStore.

        Args:
            other: Another DataStore to join with
            on: Join condition - can be:
                - Condition object (e.g., ds1.id == ds2.user_id) -> generates ON clause
                - str (e.g., "user_id") -> generates USING (user_id) clause
                - list of str (e.g., ["user_id", "country"]) -> generates USING (user_id, country)
            how: Join type ('inner', 'left', 'right', 'outer', 'cross')
            left_on: Column name from left table (alternative to on)
            right_on: Column name from right table (alternative to on)

        Example:
            >>> ds1.join(ds2, on=ds1.id == ds2.user_id)  # ON clause
            >>> ds1.join(ds2, on="user_id")              # USING clause (simpler!)
            >>> ds1.join(ds2, on=["user_id", "country"]) # USING with multiple columns
            >>> ds1.join(ds2, left_on='id', right_on='user_id', how='left')
        """
        from .enums import JoinType
        from .lazy_ops import LazyJoin

        # Convert how string to JoinType
        join_type_map = {
            'inner': JoinType.inner,
            'left': JoinType.left,
            'right': JoinType.right,
            'outer': JoinType.outer,
            'full': JoinType.full_outer,
            'cross': JoinType.cross,
        }

        if how.lower() not in join_type_map:
            raise QueryError(f"Invalid join type: {how}")

        join_type = join_type_map[how.lower()]

        # Determine join column(s) for pandas merge
        pandas_on = None
        pandas_left_on = left_on
        pandas_right_on = right_on

        # Build join condition
        if on is not None:
            if isinstance(on, str):
                # String -> USING (column) syntax for SQL, on= for pandas
                join_condition = ('USING', [on])
                pandas_on = on
            elif isinstance(on, (list, tuple)) and all(isinstance(c, str) for c in on):
                # List of strings -> USING (col1, col2, ...) syntax
                join_condition = ('USING', list(on))
                pandas_on = list(on)
            else:
                # Condition object -> ON clause (SQL only, need to extract columns for pandas)
                join_condition = on
                # For Condition objects, we can't easily extract columns, so use SQL path
        elif left_on and right_on:
            # Create condition from column names
            # Use table alias for table functions
            left_table = self._get_table_alias() if self._table_function else self.table_name
            right_table = other._get_table_alias() if other._table_function else other.table_name

            left_field = Field(left_on, table=left_table)
            right_field = Field(right_on, table=right_table)
            join_condition = left_field == right_field
        else:
            raise QueryError("Either 'on' or both 'left_on' and 'right_on' must be specified")

        # Determine execution path based on source type
        has_sql_source = bool(self._table_function or self.table_name)
        other_has_sql_source = bool(other._table_function or other.table_name)

        if has_sql_source and other_has_sql_source:
            # Both sides have SQL sources - use SQL JOIN only
            self._joins.append((other, join_type, join_condition))
        else:
            # At least one side is DataFrame-only - use LazyJoin for pandas merge
            self._add_lazy_op(
                LazyJoin(
                    right=other,
                    on=pandas_on,
                    how=how.lower(),
                    left_on=pandas_left_on,
                    right_on=pandas_right_on,
                )
            )
        # Note: Don't return self - @immutable decorator will return the copy

    @immutable
    def union(self, other: 'DataStore', all: bool = False) -> 'DataStore':
        """
        Union with another DataStore (vertical concatenation).

        This is equivalent to SQL UNION (or UNION ALL if all=True).

        Args:
            other: Another DataStore to union with
            all: If True, keep all rows (UNION ALL). If False, remove duplicates (UNION).

        Example:
            >>> ds1.union(ds2)  # Removes duplicates (like SQL UNION)
            >>> ds1.union(ds2, all=True)  # Keeps all rows (like SQL UNION ALL)
        """
        from .lazy_ops import LazyUnion

        # Add LazyUnion operation - pass the DataStore/DataFrame directly
        # Execution is deferred until needed
        self._add_lazy_op(LazyUnion(other=other, all=all))
        # Note: Don't return self - @immutable decorator will return the copy

    @immutable
    def with_column(self, name: str, expr: Union[Expression, Any]) -> 'DataStore':
        """
        Add a new column with the given name and expression.

        This is similar to polars' with_column() and pandas' assign().

        Args:
            name: Name of the new column
            expr: Expression for computing the column values

        Example:
            >>> ds.with_column('total', ds['price'] * ds['quantity'])
            >>> ds.with_column('doubled', ds['value'] * 2)
        """
        from .lazy_ops import LazyColumnAssignment

        self._add_lazy_op(LazyColumnAssignment(name, expr))
        # Note: Don't return self - @immutable decorator will return the copy

    def groupby(self, *fields: Union[str, Expression, List], sort: bool = True, **kwargs) -> 'LazyGroupBy':
        """
        Group by columns.

        Returns a LazyGroupBy object that references the ORIGINAL DataStore.
        This matches pandas semantics where groupby() returns a GroupBy object,
        not a copy of the DataFrame.

        When aggregation methods are called on the GroupBy object, they execute
        and checkpoint the ORIGINAL DataStore. This ensures that subsequent calls
        to df.to_df() use the cached result without re-execution.

        Args:
            *fields: Column names (strings), Expression objects, or a list of column names.
                     Supports both pandas-style `groupby(["a", "b"])` and
                     `groupby("a", "b")` syntax.
            sort: Sort group keys (default: True, matching pandas behavior).
                  When True, the result is sorted by group keys in ascending order.
            **kwargs: Additional arguments (for pandas compatibility, currently ignored).

        Returns:
            LazyGroupBy: GroupBy wrapper referencing this DataStore

        Example:
            >>> ds.groupby("category")  # Returns LazyGroupBy, sorted by category
            >>> ds.groupby("category", sort=False)  # Unsorted (order not guaranteed)
            >>> ds.groupby(["a", "b"])  # pandas-style list argument
            >>> ds.groupby("a", "b")    # Also supported
            >>> ds.groupby("category")["sales"].mean()  # Executes ds, returns Series
            >>> ds.to_df()  # Uses cached result (no re-computation!)
        """
        from .groupby import LazyGroupBy

        groupby_fields = []
        for field in fields:
            # Handle list argument (pandas-style): groupby(["a", "b"])
            if isinstance(field, (list, tuple)):
                for f in field:
                    if isinstance(f, str):
                        groupby_fields.append(Field(f))
                    else:
                        groupby_fields.append(f)
            elif isinstance(field, str):
                # Don't add table prefix for string fields
                groupby_fields.append(Field(field))
            else:
                groupby_fields.append(field)

        # Return a GroupBy wrapper that references self (not a copy!)
        return LazyGroupBy(self, groupby_fields, sort=sort)

    @immutable
    def agg(self, func=None, axis=0, *args, **kwargs) -> 'DataStore':
        """
        Aggregate using one or more operations.

        Supports two modes:
        1. SQL-style aggregation with keyword arguments (typically used with groupby):
           >>> ds.groupby("region").agg(
           ...     total_revenue=col("revenue").sum(),
           ...     avg_quantity=col("quantity").mean(),
           ...     order_count=col("order_id").count()
           ... )

        2. Pandas-style aggregation with dict (executes DataFrame first):
           >>> ds.agg({'amount': 'sum', 'count': 'count'})
           >>> ds.agg({'amount': ['sum', 'mean', 'max']})

        When using SQL-style aggregation:
        - Keyword argument names become output column aliases
        - Values should be aggregate expressions (e.g., col("x").sum())
        - Groupby columns are automatically included in SELECT

        Args:
            func: For pandas-style, aggregation function(s)
            axis: For pandas-style, axis to aggregate (default: 0)
            **kwargs: For SQL-style, alias=aggregate_expression pairs

        Returns:
            DataStore with aggregation applied
        """
        from .column_expr import ColumnExpr
        from .functions import AggregateFunction
        from .lazy_ops import LazySQLQuery

        # Check if we have SQL-style keyword arguments with expressions
        has_sql_agg = any(isinstance(v, (Expression, ColumnExpr, AggregateFunction)) for v in kwargs.values())

        if has_sql_agg or (func is None and kwargs and not args):
            # SQL-style aggregation: agg(alias=col("x").sum(), ...)
            # Build list of select fields: groupby fields + aggregate expressions
            select_fields = []

            # First, add groupby fields
            for gf in self._groupby_fields:
                select_fields.append(gf)
                if gf not in self._select_fields:
                    self._select_fields.append(gf)

            # Add aggregate expressions with aliases
            for alias, expr in kwargs.items():
                if isinstance(expr, ColumnExpr):
                    # Unwrap ColumnExpr to get underlying expression
                    expr = expr._expr
                if isinstance(expr, Expression):
                    # Set alias on the expression
                    expr_with_alias = copy(expr)
                    expr_with_alias.alias = alias
                    select_fields.append(expr_with_alias)
                    self._select_fields.append(expr_with_alias)
                else:
                    # Provide helpful error message for common mistakes
                    import numpy as np

                    if isinstance(expr, (int, float, np.integer, np.floating)):
                        raise QueryError(
                            f"Invalid aggregate expression for '{alias}': got scalar value {type(expr).__name__}. "
                            f"Did you mean to use col('{alias}').mean() or col('{alias}').sum() instead of ds['{alias}'].mean()? "
                            f"Use col() from datastore.expressions for SQL aggregations in agg()."
                        )
                    elif isinstance(expr, pd.Series):
                        raise QueryError(
                            f"Invalid aggregate expression for '{alias}': got pandas Series. "
                            f"Did you mean to use col('{alias}').mean() instead of ds.groupby(...)['col'].mean()? "
                            f"Use col() from datastore.expressions for SQL aggregations in agg()."
                        )
                    else:
                        raise QueryError(
                            f"Invalid aggregate expression for '{alias}': "
                            f"expected Expression, got {type(expr).__name__}. "
                            f"Use col() from datastore.expressions for SQL aggregations."
                        )

            # Build complete SQL query with GROUP BY
            select_parts = []
            for f in select_fields:
                if hasattr(f, 'alias') and f.alias:
                    select_parts.append(f'{f.to_sql()} AS "{f.alias}"')
                else:
                    select_parts.append(f.to_sql())

            select_clause = ', '.join(select_parts)

            # Build GROUP BY clause
            groupby_parts = [gf.to_sql() for gf in self._groupby_fields]
            groupby_clause = ', '.join(groupby_parts) if groupby_parts else ''

            # Build full SQL
            if groupby_clause:
                full_sql = f'SELECT {select_clause} FROM __df__ GROUP BY {groupby_clause}'
            else:
                full_sql = f'SELECT {select_clause} FROM __df__'

            # Use LazySQLQuery for proper execution
            self._add_lazy_op(LazySQLQuery(full_sql))

            return self
        else:
            # Pandas-style aggregation: agg({'col': 'func'}) or agg('sum')
            # Delegate to parent class (PandasCompatMixin)
            return super().agg(func, axis, *args, **kwargs)

    @immutable
    def assign(self, **kwargs) -> 'DataStore':
        """
        Assign new columns to a DataStore.

        Supports three modes:
        1. SQL-style aggregation when used with groupby (aggregate expressions):
           >>> ds.groupby("region").assign(
           ...     total_revenue=col("revenue").sum(),
           ...     avg_quantity=col("quantity").mean(),
           ...     order_count=col("order_id").count()
           ... )

        2. SQL expressions (Function, Expression, ColumnExpr):
           >>> ds.assign(domain=ds['url'].url.domain())
           >>> ds.assign(upper_name=ds['name'].str.upper())

        3. Standard pandas-style assignment (non-aggregate expressions or no groupby):
           >>> ds.assign(new_col=lambda x: x['old_col'] * 2)
           >>> ds.assign(doubled=ds['amount'] * 2)

        When used with groupby and aggregate expressions:
        - Acts like agg(): keyword argument names become output column aliases
        - Values should be aggregate expressions (e.g., col("x").sum())
        - Groupby columns are automatically included in SELECT

        Args:
            **kwargs: alias=expression pairs

        Returns:
            DataStore with new columns assigned
        """
        from .column_expr import ColumnExpr
        from .functions import AggregateFunction, Function

        # Check if we have groupby and aggregate expressions
        has_groupby = len(self._groupby_fields) > 0
        has_agg_expr = any(
            isinstance(v, (AggregateFunction,))
            or (isinstance(v, (Expression, ColumnExpr)) and self._is_aggregate_expr(v))
            for v in kwargs.values()
        )

        if has_groupby and has_agg_expr:
            # Delegate to agg() for groupby + aggregate expressions
            return self.agg(**kwargs)

        # Check if any value is a SQL expression (Function, Expression, ColumnExpr)
        # These need to be executed via SQL, not pandas
        has_sql_expr = any(isinstance(v, (Function, Expression, ColumnExpr)) for v in kwargs.values())

        if has_sql_expr:
            # Ensure we have a SQL source (create PythonTableFunction if needed)
            self._ensure_sql_source()

            # Use select() to add computed columns via SQL
            # First, get all existing columns
            select_items = ['*']

            # Add new computed columns with aliases
            for alias, expr in kwargs.items():
                if isinstance(expr, ColumnExpr):
                    expr = expr._expr
                if isinstance(expr, Expression):
                    # Set alias on the expression
                    expr_with_alias = expr.as_(alias)
                    select_items.append(expr_with_alias)
                else:
                    # Non-expression value, fall back to pandas
                    # This shouldn't happen often, but handle it
                    select_items.append(expr)

            return self.select(*select_items)
        else:
            # Standard pandas-style assignment
            return super().assign(**kwargs)

    def _is_aggregate_expr(self, expr) -> bool:
        """Check if an expression is an aggregate expression."""
        from .column_expr import ColumnExpr
        from .functions import AggregateFunction

        if isinstance(expr, ColumnExpr):
            expr = expr._expr

        if isinstance(expr, AggregateFunction):
            return True

        # Check if expression has aggregate function in its tree
        if hasattr(expr, '_func_name'):
            # Check registry for aggregate function
            from .function_registry import get_function_registry

            registry = get_function_registry()
            func_def = registry.get(expr._func_name)
            if func_def and func_def.func_type.name == 'AGGREGATE':
                return True

        return False

    @immutable
    def sort(
        self, *fields: Union[str, Expression], ascending: Union[bool, List[bool]] = True, kind: str = 'quicksort'
    ) -> 'DataStore':
        """
        Sort results (ORDER BY clause).

        Args:
            *fields: Column names (strings) or Expression objects.
                     Also accepts a single list/tuple of field names for pandas compatibility.
            ascending: Sort direction (default: True). Can be a single bool for all columns,
                      or a list of bools matching the number of fields.
            kind: Sort algorithm - 'quicksort' (default, unstable), 'stable', or 'mergesort' (stable)
                  Matches pandas sort_values kind parameter behavior.

        Example:
            >>> ds.sort("name")
            >>> ds.sort("price", ascending=False)
            >>> ds.sort("name", kind='stable')  # Stable sort
            >>> ds.sort(ds.date, ds.amount, ascending=False)
            >>> ds.sort("category", "price", ascending=[True, False])  # Multi-column different directions
            >>> ds.sort(['name', 'age'])  # Also accepts list (pandas-style)
        """
        from .column_expr import ColumnExpr

        # Handle case where a single list/tuple is passed (pandas-style sort(['a', 'b']))
        if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
            fields = tuple(fields[0])

        # Helper to get SQL-safe field representation
        def field_to_sql(f):
            if isinstance(f, str):
                return f
            if isinstance(f, ColumnExpr):
                return f._expr.to_sql() if hasattr(f._expr, 'to_sql') else str(f._expr)
            if hasattr(f, 'to_sql'):
                return f.to_sql()
            return str(f)

        # Normalize ascending to a list
        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(fields)
        else:
            ascending_list = list(ascending)
            if len(ascending_list) != len(fields):
                raise ValueError(f"Length of ascending ({len(ascending_list)}) != length of fields ({len(fields)})")

        # Build description for explain()
        parts = []
        for f, asc in zip(fields, ascending_list):
            direction = 'ASC' if asc else 'DESC'
            parts.append(f"{field_to_sql(f)} {direction}")
        description = ', '.join(parts)

        # Record in lazy ops for correct execution order in explain()
        # Store fields, ascending (as list), and kind for DataFrame execution
        self._add_lazy_op(
            LazyRelationalOp('ORDER BY', description, fields=list(fields), ascending=ascending_list, kind=kind)
        )

        for field, asc in zip(fields, ascending_list):
            if isinstance(field, str):
                # Don't add table prefix for string fields
                field = Field(field)
            elif isinstance(field, ColumnExpr):
                # Extract underlying expression from ColumnExpr
                field = field._expr
            elif not isinstance(field, Expression):
                # Convert other types to Field
                field = Field(str(field))
            self._orderby_fields.append((field, asc))

        # Store kind for SQL building
        self._orderby_kind = kind

        return self

    @immutable
    def orderby(self, *fields: Union[str, Expression], ascending: bool = True, kind: str = 'quicksort') -> 'DataStore':
        """
        Sort results (ORDER BY clause). Alias for sort().

        Args:
            *fields: Column names (strings) or Expression objects
            ascending: Sort direction (default: True)
            kind: Sort algorithm - 'quicksort' (default, unstable), 'stable', or 'mergesort' (stable)

        Example:
            >>> ds.orderby("name")
            >>> ds.orderby("price", ascending=False)
            >>> ds.orderby("name", kind='stable')  # Stable sort
            >>> ds.orderby(ds.date, ds.amount, ascending=False)
        """
        return self.sort(*fields, ascending=ascending, kind=kind)

    # Alias: order_by -> orderby
    order_by = orderby

    @immutable
    def limit(self, n: int) -> 'DataStore':
        """Limit number of results."""
        # Record in lazy ops for correct execution order in explain()
        # Store limit_value for DataFrame execution
        self._add_lazy_op(LazyRelationalOp('LIMIT', str(n), limit_value=n))
        self._limit_value = n
        return self

    @immutable
    def offset(self, n: int) -> 'DataStore':
        """Skip first n results."""
        # Record in lazy ops for correct execution order in explain()
        # Store offset_value for DataFrame execution
        self._add_lazy_op(LazyRelationalOp('OFFSET', str(n), offset_value=n))
        self._offset_value = n

    @immutable
    def distinct(self, subset=None, keep='first') -> 'DataStore':
        """
        Remove duplicate rows from the DataStore.

        For SQL sources, this adds DISTINCT to the query.
        For DataFrame sources, this uses drop_duplicates().

        Args:
            subset: Column label or sequence of labels to consider for duplicates.
                    If None, use all columns.
            keep: Which duplicates to keep ('first', 'last', False to drop all).

        Example:
            >>> ds.select("city").distinct()
            >>> ds.distinct(subset=['name', 'age'])
        """
        from .lazy_ops import LazyDistinct

        # Set SQL flag for SQL execution path
        self._distinct = True
        # Add lazy op for DataFrame execution path
        self._add_lazy_op(LazyDistinct(subset=subset, keep=keep))

    @immutable
    def having(self, condition: Union[Condition, str]) -> 'DataStore':
        """
        Add HAVING clause for filtering aggregated results.

        Args:
            condition: Condition object or SQL string

        Example:
            >>> ds.groupby("city").having(Count("*") > 10)
        """
        if isinstance(condition, str):
            raise NotImplementedError("String conditions not yet implemented")

        if self._having_condition is None:
            self._having_condition = condition
        else:
            # Combine with existing condition using AND
            self._having_condition = self._having_condition & condition

    @immutable
    def as_(self, alias: str) -> 'DataStore':
        """
        Set an alias for this DataStore (for use as subquery).

        Args:
            alias: Alias name

        Example:
            >>> subquery = ds.select('id', 'name').as_('sub')
            >>> main_ds.select('*').from_subquery(subquery)

        Returns:
            DataStore for chaining
        """
        self._alias = alias
        self._is_subquery = True

    def __getitem__(self, key: Union[int, slice, str, List[str]]) -> 'DataStore':
        """
        Support various indexing operations for lazy evaluation.

        - str: Return ColumnExpr that shows actual values when displayed
        - list: Record column selection operation (lazy, returns copy to avoid modifying original)
        - slice: LIMIT/OFFSET (lazy SQL operation, modifies self)
        - Condition/ColumnExpr: Boolean indexing (filter rows, returns copy)

        Examples:
            >>> ds[:10]          # LIMIT 10
            >>> ds[10:]          # OFFSET 10
            >>> ds[10:20]        # LIMIT 10 OFFSET 10
            >>> ds['column']     # Returns ColumnExpr (displays like pandas Series)
            >>> ds['column'] - 1 # Returns ColumnExpr with computed values
            >>> ds[['col1', 'col2']]  # Select multiple columns (lazy, returns copy)
            >>> ds[ds['age'] > 18]    # Boolean indexing (filter, returns copy)
            >>> ds[(ds['a'] > 0) & (ds['b'] > 0)]  # Compound condition
        """
        from .column_expr import ColumnExpr
        from .conditions import Condition
        from copy import copy

        if isinstance(key, str):
            # Return ColumnExpr that wraps a Field and can execute
            # This allows pandas-like behavior: ds['col'] shows actual values
            # but ds['col'] > 18 still returns Condition for filtering
            return ColumnExpr(Field(key), self)

        elif isinstance(key, (list, pd.Index)):
            # Multi-column selection: use LazyRelationalOp(SELECT) for SQL pushdown
            # Convert pandas Index to list if needed
            if isinstance(key, pd.Index):
                key = key.tolist()
            # Create a copy to avoid modifying the original DataStore's _lazy_ops
            # This fixes the bug where df[['col1', 'col2']].head() would modify df
            result = copy(self) if getattr(self, 'is_immutable', True) else self
            if result is not self:
                # Reset cache state for the new copy
                result._cached_result = None
                result._cache_version = 0
                result._cached_at_version = -1
                result._cache_timestamp = None
            # Use LazyRelationalOp(SELECT) so column selection can be pushed to SQL
            # This allows ds[['col1', 'col2']].sort_values('col1').head(10) to be
            # fully executed as SQL: SELECT col1, col2 FROM ... ORDER BY col1 LIMIT 10
            fields = [Field(col) for col in key]
            result._add_lazy_op(
                LazyRelationalOp(op_type='SELECT', description=f"Select columns: {', '.join(key)}", fields=fields)
            )
            return result

        elif isinstance(key, slice):
            # LIMIT/OFFSET - this is a SQL operation
            # Create a copy to avoid modifying the original DataStore (pandas-like behavior)
            start, stop, step = key.start, key.stop, key.step

            if step is not None:
                raise ValueError("Step not supported in slice notation")

            result = copy(self) if getattr(self, 'is_immutable', True) else self
            if result is not self:
                # Reset cache state for the new copy
                result._cached_result = None
                result._cache_version = 0
                result._cached_at_version = -1
                result._cache_timestamp = None

            if stop is not None:
                if start is not None:
                    # ds[start:stop] -> LIMIT (stop-start) OFFSET start
                    limit_val = stop - start if stop > start else stop
                    result._offset_value = start
                    result._limit_value = limit_val
                    result._add_lazy_op(LazyRelationalOp('OFFSET', f'OFFSET {start}', offset_value=start))
                    result._add_lazy_op(LazyRelationalOp('LIMIT', f'LIMIT {limit_val}', limit_value=limit_val))
                else:
                    # ds[:stop] -> LIMIT stop
                    result._limit_value = stop
                    result._add_lazy_op(LazyRelationalOp('LIMIT', f'LIMIT {stop}', limit_value=stop))
            elif start is not None:
                # ds[start:] -> OFFSET start
                result._offset_value = start
                result._add_lazy_op(LazyRelationalOp('OFFSET', f'OFFSET {start}', offset_value=start))
            return result

        elif isinstance(key, (Condition, ColumnExpr)):
            # Boolean indexing: filter rows like pandas df[condition]
            # Create a copy to avoid modifying the original DataStore
            result = copy(self) if getattr(self, 'is_immutable', True) else self
            if result is not self:
                # Reset cache state for the new copy
                result._cached_result = None
                result._cache_version = 0
                result._cached_at_version = -1
                result._cache_timestamp = None
            return result.filter(key)

        else:
            # Check for LazyCondition (from isin/between) - needs late import
            from .lazy_result import LazyCondition

            if isinstance(key, LazyCondition):
                # Boolean indexing with LazyCondition
                result = copy(self) if getattr(self, 'is_immutable', True) else self
                if result is not self:
                    result._cached_result = None
                    result._cache_version = 0
                    result._cached_at_version = -1
                    result._cache_timestamp = None
                return result.filter(key)

            raise TypeError(
                f"DataStore indices must be slices, strings, lists, or conditions, not {type(key).__name__}"
            )

    def __iter__(self):
        """
        Iterate over column names.

        Matches pandas DataFrame behavior where iterating yields column names.

        Example:
            >>> for col in ds:
            ...     print(col)
        """
        return iter(self.columns)

    # ========== SQL Generation ==========

    def to_sql(self, quote_char: str = None, as_subquery: bool = False, execution_format: bool = False) -> str:
        """
        Generate SQL query.

        Args:
            quote_char: Quote character for identifiers
            as_subquery: Whether to format as subquery with parentheses
            execution_format: If True, returns the exact SQL that _execute() runs
                             (including row order preservation logic).
                             If False (default), returns clean semantic SQL.

        Returns:
            SQL query string
        """
        if quote_char is None:
            quote_char = self.quote_char

        # Handle different query types
        if self._delete_flag:
            sql = self._generate_delete_sql(quote_char)
        elif self._update_fields:
            sql = self._generate_update_sql(quote_char)
        elif self._insert_columns:
            sql = self._generate_insert_sql(quote_char)
        elif execution_format and self._lazy_ops and (self._table_function or self.table_name):
            # Use unified SQL generation for lazy operations
            # This returns the exact SQL that _execute() would run
            sql = self._build_execution_sql()
            if sql is None:
                sql = self._generate_select_sql(quote_char)
        else:
            sql = self._generate_select_sql(quote_char)

        # Add subquery formatting
        if as_subquery or self._is_subquery:
            sql = f"({sql})"
            if self._alias:
                sql = f"{sql} AS {format_identifier(self._alias, quote_char)}"

        return sql

    def _generate_select_sql(self, quote_char: str) -> str:
        """Generate SELECT SQL."""
        parts = []

        # SELECT clause
        if self._select_fields:
            fields_sql = ', '.join(
                field.to_sql(quote_char=quote_char, with_alias=True) for field in self._select_fields
            )
            # If _select_star is True, prepend '*' to include all existing columns
            if self._select_star:
                fields_sql = f"*, {fields_sql}"
        else:
            fields_sql = '*'

        distinct_keyword = 'DISTINCT ' if self._distinct else ''
        parts.append(f"SELECT {distinct_keyword}{fields_sql}")

        # FROM clause
        if self._table_function:
            # Use table function instead of table name
            table_func_sql = self._table_function.to_sql(quote_char=quote_char)
            # Add alias for table function (required by ClickHouse for JOINs)
            alias = self._get_table_alias()
            parts.append(f"FROM {table_func_sql} AS {format_identifier(alias, quote_char)}")
        elif self.table_name:
            parts.append(f"FROM {format_identifier(self.table_name, quote_char)}")

        # JOIN clauses
        if self._joins:
            for other_ds, join_type, join_condition in self._joins:
                # Generate JOIN clause
                join_keyword = join_type.value if join_type.value else ''
                if join_keyword:
                    join_clause = f"{join_keyword} JOIN"
                else:
                    join_clause = "JOIN"

                # Handle subquery joins
                if isinstance(other_ds, DataStore) and other_ds._is_subquery:
                    other_table = other_ds.to_sql(quote_char=quote_char, as_subquery=True)
                elif isinstance(other_ds, DataStore) and other_ds._table_function:
                    # Use table function for the joined table with alias
                    table_func_sql = other_ds._table_function.to_sql(quote_char=quote_char)
                    alias = other_ds._get_table_alias()
                    other_table = f"{table_func_sql} AS {format_identifier(alias, quote_char)}"
                else:
                    other_table = format_identifier(other_ds.table_name, quote_char)

                # Handle USING vs ON syntax
                if isinstance(join_condition, tuple) and join_condition[0] == 'USING':
                    # USING (col1, col2, ...) syntax
                    columns = join_condition[1]
                    using_cols = ', '.join(format_identifier(c, quote_char) for c in columns)
                    parts.append(f"{join_clause} {other_table} USING ({using_cols})")
                else:
                    # ON condition syntax
                    condition_sql = join_condition.to_sql(quote_char=quote_char)
                    parts.append(f"{join_clause} {other_table} ON {condition_sql}")

        # WHERE clause
        if self._where_condition:
            where_sql = self._where_condition.to_sql(quote_char=quote_char)
            parts.append(f"WHERE {where_sql}")

        # GROUP BY clause
        if self._groupby_fields:
            groupby_sql = ', '.join(field.to_sql(quote_char=quote_char) for field in self._groupby_fields)
            parts.append(f"GROUP BY {groupby_sql}")

        # HAVING clause
        if self._having_condition:
            having_sql = self._having_condition.to_sql(quote_char=quote_char)
            parts.append(f"HAVING {having_sql}")

        # ORDER BY clause
        if self._orderby_fields:
            orderby_parts = []
            for field, ascending in self._orderby_fields:
                field_sql = field.to_sql(quote_char=quote_char)
                direction = 'ASC' if ascending else 'DESC'
                orderby_parts.append(f"{field_sql} {direction}")
            parts.append(f"ORDER BY {', '.join(orderby_parts)}")

        # LIMIT clause
        if self._limit_value is not None:
            parts.append(f"LIMIT {self._limit_value}")

        # OFFSET clause
        if self._offset_value is not None:
            parts.append(f"OFFSET {self._offset_value}")

        # Add format settings if present
        if self._format_settings:
            settings_parts = []
            for key, value in self._format_settings.items():
                if isinstance(value, str):
                    settings_parts.append(f"{key}='{value}'")
                else:
                    settings_parts.append(f"{key}={value}")
            parts.append(f"SETTINGS {', '.join(settings_parts)}")

        return ' '.join(parts)

    def _generate_insert_sql(self, quote_char: str) -> str:
        """Generate INSERT SQL (ClickHouse style)."""
        # Determine target (table function or table name)
        if self._table_function:
            if not self._table_function.can_write:
                raise QueryError(
                    f"Table function '{self.source_type}' does not support writing. "
                    f"Read-only table functions: mongodb, sqlite, iceberg, deltaLake, hudi, numbers, generateRandom"
                )
            target = f"TABLE FUNCTION {self._table_function.to_sql(quote_char=quote_char)}"
        elif self.table_name:
            target = format_identifier(self.table_name, quote_char)
        else:
            raise QueryError("Table name or table function required for INSERT")

        parts = [f"INSERT INTO {target}"]

        # Columns
        if self._insert_columns:
            columns_sql = ', '.join(format_identifier(col, quote_char) for col in self._insert_columns)
            parts.append(f"({columns_sql})")

        # VALUES or SELECT
        if self._insert_select is not None:
            # INSERT INTO ... SELECT ...
            select_sql = self._insert_select.to_sql(quote_char=quote_char)
            parts.append(select_sql)
        elif self._insert_values:
            # INSERT INTO ... VALUES ...
            values_parts = []
            for row in self._insert_values:
                row_values = []
                for value in row:
                    if value is None:
                        row_values.append('NULL')
                    elif isinstance(value, bool):
                        row_values.append('1' if value else '0')
                    elif isinstance(value, str):
                        escaped = value.replace("'", "''")
                        row_values.append(f"'{escaped}'")
                    elif isinstance(value, Expression):
                        row_values.append(value.to_sql(quote_char=quote_char))
                    else:
                        row_values.append(str(value))
                values_parts.append(f"({', '.join(row_values)})")
            parts.append(f"VALUES {', '.join(values_parts)}")
        else:
            raise QueryError("INSERT query requires either VALUES or SELECT")

        return ' '.join(parts)

    def _generate_update_sql(self, quote_char: str) -> str:
        """Generate UPDATE SQL (ClickHouse style: ALTER TABLE ... UPDATE ...)."""
        if not self.table_name:
            raise QueryError("Table name required for UPDATE")

        if not self._update_fields:
            raise QueryError("UPDATE query requires at least one field to update")

        parts = [f"ALTER TABLE {format_identifier(self.table_name, quote_char)}"]

        # UPDATE clause
        update_parts = []
        for field_name, value in self._update_fields:
            field_sql = format_identifier(field_name, quote_char)

            if value is None:
                value_sql = 'NULL'
            elif isinstance(value, bool):
                value_sql = '1' if value else '0'
            elif isinstance(value, str):
                escaped = value.replace("'", "''")
                value_sql = f"'{escaped}'"
            elif isinstance(value, Expression):
                value_sql = value.to_sql(quote_char=quote_char)
            else:
                value_sql = str(value)

            update_parts.append(f"{field_sql}={value_sql}")

        parts.append(f"UPDATE {', '.join(update_parts)}")

        # WHERE clause
        if self._where_condition:
            where_sql = self._where_condition.to_sql(quote_char=quote_char)
            parts.append(f"WHERE {where_sql}")

        return ' '.join(parts)

    def _generate_delete_sql(self, quote_char: str) -> str:
        """Generate DELETE SQL (ClickHouse style: ALTER TABLE ... DELETE WHERE ...)."""
        if not self.table_name:
            raise QueryError("Table name required for DELETE")

        parts = [f"ALTER TABLE {format_identifier(self.table_name, quote_char)}"]
        parts.append("DELETE")

        # WHERE clause (required for ClickHouse DELETE)
        if self._where_condition:
            where_sql = self._where_condition.to_sql(quote_char=quote_char)
            parts.append(f"WHERE {where_sql}")
        else:
            raise QueryError("ClickHouse DELETE requires WHERE clause. Use WHERE 1=1 to delete all rows.")

        return ' '.join(parts)

    # ========== Dynamic Field Access ==========

    # List of special attribute names that should not be treated as column names
    _RESERVED_ATTRS = frozenset(
        {
            'is_immutable',
            'is_mutable',
            'is_copy',
            'is_view',
            'config',
            'index',
            'columns',
            'values',
            'dtypes',
            'shape',
            'size',
            'ndim',
            'empty',
            'T',
            'axes',
            'to_pandas',  # Prevent being treated as column name
            'to_df',
        }
    )

    @ignore_copy
    def __getattr__(self, name: str):
        """
        Support dynamic field access: ds.column_name

        Returns a ColumnExpr that displays actual values like pandas.

        Example:
            >>> ds.age        # Shows actual values (like pandas Series)
            >>> ds.age > 18   # Returns Condition for filtering
            >>> ds.age - 10   # Returns ColumnExpr with computed values

        Note:
            Dynamic field access does NOT add table prefix by default.
            For JOIN conditions where disambiguation is needed, use left_on/right_on
            parameters or explicitly create Field with table: Field('col', table='t').
        """
        from .column_expr import ColumnExpr

        # Avoid infinite recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Don't treat reserved/special attributes as column names
        if name in self._RESERVED_ATTRS:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Return ColumnExpr that wraps a Field and can execute
        return ColumnExpr(Field(name), self)

    # ========== Copy Support ==========

    def __copy__(self) -> 'DataStore':
        """Create a shallow copy (note: less important now without strict immutability)."""
        new_ds = type(self).__new__(type(self))
        new_ds.__dict__.update(self.__dict__)

        # Copy mutable collections
        new_ds._select_fields = self._select_fields.copy()
        new_ds._joins = self._joins.copy()
        new_ds._groupby_fields = self._groupby_fields.copy()
        new_ds._orderby_fields = self._orderby_fields.copy()
        new_ds._insert_columns = self._insert_columns.copy()
        new_ds._insert_values = self._insert_values.copy()
        new_ds._update_fields = self._update_fields.copy()
        new_ds._format_settings = self._format_settings.copy()
        new_ds._lazy_ops = self._lazy_ops.copy()  # Copy lazy operations

        # Copy operation history
        if hasattr(self, '_operation_history'):
            new_ds._operation_history = self._operation_history.copy()

        # Share connection, executor, and table_function (not deep copied)

        return new_ds

    # ========== Built-in Methods ==========

    def __len__(self) -> int:
        """
        Return the number of rows in the DataStore.

        This method efficiently uses SQL COUNT(*) when possible, falling back to
        execution only when necessary (e.g., after pandas operations).

        This enables using len(ds) on DataStore objects.

        Example:
            >>> ds = DataStore.from_file("large_data.parquet")
            >>> row_count = len(ds.filter(ds.age > 18))  # Uses SQL COUNT(*)
        """
        # Use SQL-based count_rows() for efficiency
        return self.count_rows()

    def __array__(self, dtype=None, copy=None):
        """
        NumPy array interface for compatibility with numpy functions.

        This allows DataStore to be used directly with numpy functions like:
        - np.array(ds)
        - np.mean(ds)
        - np.sum(ds)

        Args:
            dtype: Optional dtype for the resulting array
            copy: If True, ensure the returned array is a copy (numpy 2.0+)

        Returns:
            numpy array representation of the DataFrame

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> arr = np.array(ds)  # Convert entire DataFrame to array
            >>> mean = np.mean(ds)  # Compute mean across all values
        """
        import numpy as np

        df = self._execute()
        if isinstance(df, pd.DataFrame):
            # Use to_numpy() to handle categorical/extension dtypes
            arr = df.to_numpy()
        elif isinstance(df, pd.Series):
            arr = df.to_numpy()
        elif df is None:
            arr = np.array([])
        else:
            arr = np.array(df)

        if dtype is not None:
            arr = arr.astype(dtype)
        # Handle copy parameter for numpy 2.0+ compatibility
        if copy:
            arr = np.array(arr, copy=True)
        return arr

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """
        DataFrame Interchange Protocol implementation.

        This enables DataStore to be used directly with libraries that support
        the DataFrame Interchange Protocol (e.g., seaborn, plotly, altair).

        Instead of:
            seaborn.countplot(x="col", data=ds.to_df())

        You can now write:
            seaborn.countplot(x="col", data=ds)

        Args:
            nan_as_null: Whether to convert NaN values to null (default: False)
            allow_copy: Whether to allow copying the data (default: True)

        Returns:
            DataFrame interchange object from the underlying pandas DataFrame

        Example:
            >>> import seaborn as sns
            >>> ds = DataStore.from_file("data.csv")
            >>> sns.countplot(x="Survived", hue="Sex", data=ds)  # Works directly!
        """
        df = self._execute()
        return df.__dataframe__(nan_as_null=nan_as_null, allow_copy=allow_copy)

    # ========== String Representation ==========

    def __str__(self) -> str:
        """
        Return string representation - triggers execution.

        This is called by print().
        If we have any operations (SQL or lazy), executes and shows the DataFrame.
        """
        # If we have operations, execute and show the DataFrame
        if self._has_sql_state() or self._lazy_ops:
            try:
                df = self._execute()
                return str(df)
            except Exception as e:
                return f"DataStore (execution failed: {e})"

        # Fallback: show basic info
        return f"DataStore(source_type={self.source_type!r}, table={self.table_name!r})"

    def __repr__(self) -> str:
        """
        Return repr representation - triggers execution.

        This is called in Jupyter/REPL when displaying the object.
        If we have operations, executes and shows the DataFrame.
        """
        # If we have operations, execute and show the DataFrame
        if self._has_sql_state() or self._lazy_ops:
            try:
                df = self._execute()
                return repr(df)
            except Exception as e:
                # If execution fails, show error info
                return f"DataStore(execution failed: {e})"

        # Fallback: show basic info
        parts = [f"DataStore(source_type={self.source_type!r}"]
        if self.table_name:
            parts.append(f", table={self.table_name!r}")
        if self._table_function:
            parts.append(", table_function=True")
        parts.append(")")
        return "".join(parts)

    def _repr_html_(self) -> str:
        """
        Return HTML representation for Jupyter - triggers execution.

        This method is automatically called by Jupyter when displaying the object.
        """
        # If we have operations, execute and show the DataFrame
        if self._has_sql_state() or self._lazy_ops:
            try:
                df = self._execute()
                return df._repr_html_()
            except Exception as e:
                # If execution fails, show error in HTML
                return f"<div><strong>DataStore</strong> (execution failed: {e})</div>"

        # Fallback: show basic info in HTML
        html = "<div><strong>DataStore</strong><br>"
        html += f"Source type: {self.source_type}<br>"
        if self.table_name:
            html += f"Table: {self.table_name}<br>"
        if self._table_function:
            html += "Using table function<br>"
        html += "</div>"
        return html
