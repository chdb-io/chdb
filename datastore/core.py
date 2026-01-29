"""
Core DataStore class - main entry point for data operations
"""

import time
import pandas as pd
import numpy as np
from typing import Any, Optional, List, Dict, Union, TYPE_CHECKING
from copy import copy

if TYPE_CHECKING:
    from .column_expr import ColumnExpr
    from .groupby import LazyGroupBy
    from .case_when import CaseWhenBuilder
    from .query_planner import ExecutionSegment, QueryPlan

from .expressions import Field, Expression, Literal, Star
from .conditions import Condition
from .utils import (
    immutable,
    ignore_copy,
    format_identifier,
    normalize_ascending,
    map_agg_func,
)
from .exceptions import (
    QueryError,
    ConnectionError,
    ExecutionError,
    UnsupportedOperationError,
)
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
from .adapters import get_adapter, SourceAdapter

__all__ = ["DataStore"]

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

    # Sensitive fields to mask in repr/str output
    _SENSITIVE_FIELDS = frozenset(
        {
            "password",
            "secret",
            "token",
            "key",
            "api_key",
            "secret_key",
            "access_key",
            "secret_access_key",
        }
    )
    _MASKED_VALUE = "***"

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
        if "source_type" in kwargs:
            if source is None:
                source = kwargs.pop("source_type")
            else:
                # If both provided, just remove from kwargs to avoid passing to connection
                kwargs.pop("source_type")

        # Extract DataFrame constructor kwargs (index, columns, dtype) before other processing
        # These should be passed to pd.DataFrame() when creating from dict/list/etc
        df_constructor_kwargs = {}
        for key in ["index", "columns", "dtype"]:
            if key in kwargs:
                df_constructor_kwargs[key] = kwargs.pop(key)

        # Handle columns-only construction: DataStore(columns=['A', 'B', 'C'])
        # This creates an empty DataFrame with specified columns
        if source is None and "columns" in df_constructor_kwargs:
            df = pd.DataFrame(**df_constructor_kwargs)
            self._init_from_dataframe(df, database, connection, **kwargs)
            return

        # Handle pandas Series input - convert to DataFrame
        if isinstance(source, pd.Series):
            df = source.to_frame()
            if df_constructor_kwargs:
                # Apply any index override
                if "index" in df_constructor_kwargs:
                    df.index = df_constructor_kwargs["index"]
            self._init_from_dataframe(df, database, connection, **kwargs)
            return

        # Handle DataFrame input directly
        if isinstance(source, pd.DataFrame):
            # Initialize with DataFrame - delegate to _init_from_dataframe
            self._init_from_dataframe(source, database, connection, **kwargs)
            return

        # Handle dict input - convert to DataFrame first
        if isinstance(source, dict):
            df = pd.DataFrame(source, **df_constructor_kwargs)
            self._init_from_dataframe(df, database, connection, **kwargs)
            return

        # Handle list/tuple input (list of lists or list of dicts)
        if isinstance(source, (list, tuple)):
            df = pd.DataFrame(source, **df_constructor_kwargs)
            self._init_from_dataframe(df, database, connection, **kwargs)
            return

        # Handle numpy array input
        if isinstance(source, np.ndarray):
            df = pd.DataFrame(source, **df_constructor_kwargs)
            self._init_from_dataframe(df, database, connection, **kwargs)
            return

        # Handle file path input directly (convenience feature)
        # Detect if source looks like a file path and convert to proper format
        import os

        if isinstance(source, str) and source not in (
            "chdb",
            "file",
            "s3",
            "gcs",
            "http",
            "https",
            "mysql",
            "postgresql",
            "postgres",
            "clickhouse",
            "remote",
            "mongodb",
            "mongo",
            "sqlite",
            "redis",
            "azure",
            "azureblob",
            "hdfs",
            "iceberg",
            "hudi",
            "delta",
            "deltalake",
            "numbers",
            "generaterandom",
            "python",
            "url",
            "remotesecure",
        ):
            # Check if it looks like a file path (has extension or path separators)
            if (
                "/" in source
                or "\\" in source
                or source.endswith(
                    (
                        ".parquet",
                        ".csv",
                        ".tsv",
                        ".json",
                        ".jsonl",
                        ".arrow",
                        ".feather",
                        ".orc",
                        ".avro",
                    )
                )
            ):
                # Auto-convert file path to proper format
                kwargs["path"] = source
                source = "file"

        self.source_type = source or "chdb"
        self.table_name = table
        self.connection_params = kwargs

        # Table function support
        self._table_function: Optional[TableFunction] = None
        self._format_settings: Dict[str, Any] = {}

        # Source DataFrame (for from_df() - enables on-demand PythonTableFunction creation)
        self._source_df = None
        self._source_df_name: Optional[str] = None

        # Determine if this is a table function source (external data sources)
        # These sources use table functions and should always use :memory: for chDB
        table_function_sources = [
            "file",
            "s3",
            "gcs",
            "http",
            "https",
            "url",
            "mysql",
            "postgresql",
            "postgres",
            "clickhouse",
            "remote",
            "remotesecure",
            "mongodb",
            "mongo",
            "sqlite",
            "redis",
            "azure",
            "azureblob",
            "hdfs",
            "iceberg",
            "hudi",
            "delta",
            "deltalake",
            "numbers",
            "generaterandom",
            "python",
        ]
        is_table_function_source = source and source.lower() in table_function_sources

        # For table function sources, the 'database' parameter refers to the REMOTE
        # database name (e.g., MySQL database, ClickHouse database), NOT the local
        # chDB database path. chDB should always use :memory: for these sources.
        if is_table_function_source:
            self.database = ":memory:"
        else:
            self.database = database

        # Create table function if source is specified
        if source and source.lower() != "chdb":
            try:
                # For database sources with explicit table, pass table name
                if table and source.lower() in [
                    "clickhouse",
                    "remote",
                    "mysql",
                    "postgresql",
                    "postgres",
                    "mongodb",
                    "mongo",
                    "sqlite",
                ]:
                    kwargs["table"] = table

                # For database sources, pass the database name to the table function
                # (this is the REMOTE database, not the local chDB path)
                if (
                    database
                    and database != ":memory:"
                    and source.lower()
                    in [
                        "clickhouse",
                        "remote",
                        "mysql",
                        "postgresql",
                        "postgres",
                        "mongodb",
                        "mongo",
                        "sqlite",
                    ]
                ):
                    kwargs["database"] = database

                self._table_function = create_table_function(source, **kwargs)
            except Exception:
                # If table function creation fails, it might be a regular table
                # We'll treat it as a regular table and table_function remains None
                pass

        # Query state
        self._select_fields: List[Expression] = []
        self._select_star: bool = (
            False  # True when SELECT * is used (possibly with additional computed columns)
        )
        self._where_condition: Optional[Condition] = None
        self._joins: List[tuple] = (
            []
        )  # [(table/datastore, join_type, on_condition), ...]
        self._groupby_fields: List[Expression] = []
        self._having_condition: Optional[Condition] = None
        self._orderby_fields: List[tuple] = []  # [(field, ascending), ...]
        self._orderby_kind: str = "quicksort"  # Sort algorithm (matches pandas default)
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._distinct: bool = False

        # INSERT/UPDATE/DELETE state
        self._insert_columns: List[str] = []
        self._insert_values: List[List[Any]] = []
        self._insert_select: Optional["DataStore"] = None
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
        self._original_source_desc: Optional[str] = (
            None  # Preserve original data source for explain()
        )

        # Logger instance
        self._logger = get_logger()

        # Cache state for executed results
        # This implements intelligent automatic caching to avoid re-execution
        # when repr/__str__ are called multiple times
        self._cached_result: Optional[pd.DataFrame] = None
        self._cache_version: int = 0  # Incremented when operations are added
        self._cached_at_version: int = -1  # Version when cache was created
        self._cache_timestamp: Optional[float] = None  # For TTL support

        # Computed columns tracking for chained assign support
        # Maps column alias -> original Expression (before any aliasing)
        # This allows ds['computed_col'] to return the full expression
        # instead of just Field('computed_col')
        self._computed_columns: Dict[str, Expression] = {}

        # Index preservation for lazy SQL execution
        # Tracks index info when DataFrame has a custom index (from set_index())
        # Format: {'name': index_name, 'names': [names_for_multiindex]} or None
        self._index_info: Optional[Dict[str, Any]] = None

        # Remote connection mode support
        # 'connection' = only host/user/password, no db/table
        # 'database' = has database but no table
        # 'table' = has both (normal mode, existing behavior)
        self._connection_mode: str = "table"  # default

        # Default context for queries (set via use())
        self._default_schema: Optional[str] = None
        self._default_database: Optional[str] = None
        self._default_table: Optional[str] = None

        # Remote connection parameters (for metadata operations)
        self._remote_params: Dict[str, Any] = {}

        # Detect connection mode for remote database sources
        remote_db_sources = [
            "clickhouse",
            "remote",
            "remotesecure",
            "mysql",
            "postgresql",
            "postgres",
        ]
        if source and source.lower() in remote_db_sources:
            # Store remote params for metadata operations
            self._remote_params = {
                "host": kwargs.get("host"),
                "user": kwargs.get("user"),
                "password": kwargs.get("password", ""),
                "secure": kwargs.get("secure", False),
            }

            # Determine connection mode
            has_database = database and database != ":memory:"
            has_table = table is not None

            if not has_database and not has_table:
                self._connection_mode = "connection"
            elif has_database and not has_table:
                self._connection_mode = "database"
                self._default_database = database
            else:
                self._connection_mode = "table"

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
        self.source_type = "dataframe"
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
        self._orderby_kind: str = "quicksort"
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None
        self._distinct: bool = False

        # INSERT/UPDATE/DELETE state
        self._insert_columns: List[str] = []
        self._insert_values: List[List[Any]] = []
        self._insert_select: Optional["DataStore"] = None
        self._update_fields: List[tuple] = []
        self._delete_flag: bool = False

        # Subquery support
        self._alias: Optional[str] = None
        self._is_subquery: bool = False

        # Connection and execution
        self._connection: Optional[Connection] = connection
        self._executor: Optional[Executor] = None

        # Build schema from DataFrame dtypes
        self._schema: Optional[Dict[str, str]] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }

        # Configuration
        self.quote_char = '"'

        # Add the DataFrame as a lazy source
        self._lazy_ops: List[LazyOp] = [LazyDataFrameSource(df)]

        # Generate unique variable name
        import uuid

        self._df_var_name: str = f"__ds_df_{uuid.uuid4().hex}__"

        # Set source description for explain()
        name = kwargs.get("name")
        shape_str = f"{df.shape[0]} rows x {df.shape[1]} cols"
        self._original_source_desc: Optional[str] = (
            f"DataFrame({name or 'unnamed'}, {shape_str})"
        )

        # Operation tracking
        self._operation_history: List[Dict[str, Any]] = []

        # Logger instance
        self._logger = get_logger()

        # Cache state
        self._cached_result: Optional[pd.DataFrame] = None
        self._cache_version: int = 0
        self._cached_at_version: int = -1
        self._cache_timestamp: Optional[float] = None

        # Computed columns tracking for chained assign support
        self._computed_columns: Dict[str, Expression] = {}

        # Index preservation for lazy SQL execution
        # Tracks index info when DataFrame has a custom index (from set_index())
        self._index_info: Optional[Dict[str, Any]] = None

        # Remote connection mode support (not applicable for DataFrame sources, but initialized for consistency)
        self._connection_mode: str = "table"
        self._default_schema: Optional[str] = None
        self._default_database: Optional[str] = None
        self._default_table: Optional[str] = None
        self._remote_params: Dict[str, Any] = {}

    # ========== Operation Tracking for explain() ==========

    def _track_operation(
        self, op_type: str, description: str, details: Dict[str, Any] = None
    ):
        """
        Track an operation for explain() output.

        Args:
            op_type: Type of operation ('sql', 'pandas', 'execute')
            description: Human-readable description
            details: Additional details about the operation
        """
        operation = {
            "type": op_type,
            "description": description,
            "details": details or {},
            "is_on_dataframe": op_type == "pandas",
            "executed_at_call": False,
        }
        self._operation_history.append(operation)

    def _get_data_source_description(self):
        """Get a description of the data source."""
        # Return cached description if available
        if hasattr(self, "_original_source_desc") and self._original_source_desc:
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
        if desc and (
            not hasattr(self, "_original_source_desc") or not self._original_source_desc
        ):
            self._original_source_desc = desc

        return desc

    def _analyze_execution_phases(self):
        """Analyze operation history and group operations into execution phases."""
        if not self._operation_history:
            return [], None, []

        # Find the execution point
        mat_idx = next(
            (
                i
                for i, op in enumerate(self._operation_history)
                if op["type"] == "execute"
            ),
            None,
        )

        if mat_idx is not None:
            # Explicit execution operation present
            return (
                self._operation_history[:mat_idx],
                self._operation_history[mat_idx],
                self._operation_history[mat_idx + 1 :],
            )

        # No explicit execution - split by is_on_dataframe flag
        lazy = [
            op for op in self._operation_history if not op.get("is_on_dataframe", False)
        ]
        executed = [
            op for op in self._operation_history if op.get("is_on_dataframe", False)
        ]

        # If there are only executed ops, the first one becomes the implicit execution point
        if executed and not lazy and executed[0]["type"] in ["pandas", "execute"]:
            return [], executed[0], executed[1:]

        # If both exist, the first executed operation is the implicit execution point
        if lazy and executed:
            return lazy, executed[0], executed[1:]

        return lazy, None, executed

    def _render_operations(self, operations, start_num=1, verbose=False):
        """Render a list of operations."""
        lines = []
        for i, op in enumerate(operations, start_num):
            icon = {"sql": "🔍", "pandas": "🐼", "execute": "🔄"}.get(op["type"], "📝")
            desc = (
                f"SQL on DataFrame: {op['description']}"
                if op["type"] == "sql" and op.get("is_on_dataframe")
                else f"{op['type'].upper()}: {op['description']}"
            )
            lines.append(f" [{i}] {icon} {desc}")
            if verbose and op.get("details"):
                for k, v in op["details"].items():
                    lines.append(f"     └─ {k}: {v}")
        return lines

    def explain(self, verbose: bool = False) -> str:
        """
        Generate and display the execution plan in original operation order.

        This method shows operations in the exact order they were defined,
        which is critical because order affects execution results.

        Uses plan_segments() for accurate SQL-Pandas-SQL interleaving analysis.

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
        from .query_planner import QueryPlanner

        # Ensure data source description is cached before analysis
        if not hasattr(self, "_original_source_desc") or not self._original_source_desc:
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
        # Use plan_segments() for accurate SQL-Pandas-SQL interleaving analysis
        if self._lazy_ops:
            planner = QueryPlanner()
            has_sql_source = bool(
                self._table_function or self.table_name or self._source_df is not None
            )
            schema = self.schema() if has_sql_source else self._schema
            exec_plan = planner.plan_segments(
                self._lazy_ops, has_sql_source, schema=schema
            )

            lines.append("\nOperations:")
            lines.append("─" * 80)

            # Build segment info for display
            num_segments = len(exec_plan.segments)
            if num_segments > 0:
                # Show segment summary
                # Note: Operation numbering starts at 2 because [1] is the data source
                segment_summaries = []
                op_idx = 0
                for seg_idx, segment in enumerate(exec_plan.segments):
                    # +2 because: +1 for 1-based indexing, +1 for data source being [1]
                    start_op = op_idx + 2
                    end_op = op_idx + len(segment.ops) + 1
                    engine = "chDB" if segment.is_sql() else "Pandas"
                    source_info = (
                        "(from source)"
                        if segment.is_first_segment
                        else "(on DataFrame)"
                    )
                    if segment.ops:
                        segment_summaries.append(
                            f"    ️  Segment {seg_idx + 1} [{engine}] {source_info}: Operations {start_op}-{end_op}"
                        )
                    op_idx += len(segment.ops)

                for summary in segment_summaries:
                    lines.append(summary)
                lines.append(
                    "    ️  Note: SQL operations after Pandas ops use Python() table function"
                )
                lines.append("")

            # Build op -> segment mapping for accurate engine display
            op_to_segment = {}  # op index -> (segment_idx, is_sql)
            op_idx = 0
            for seg_idx, segment in enumerate(exec_plan.segments):
                for _ in segment.ops:
                    op_to_segment[op_idx] = (
                        seg_idx,
                        segment.is_sql(),
                        segment.is_first_segment,
                    )
                    op_idx += 1

            for i, op in enumerate(self._lazy_ops):
                counter += 1
                # Determine which engine will execute this operation based on segment
                seg_idx, is_sql, is_first = op_to_segment.get(i, (0, False, False))

                if is_sql:
                    # SQL engine will execute this
                    if isinstance(op, LazyRelationalOp):
                        lines.append(f" [{counter}] 🚀 [chDB] {op.describe()}")
                    else:
                        engine = op.execution_engine()
                        if engine == "chDB":
                            lines.append(f" [{counter}] 🚀 [chDB] {op.describe()}")
                        else:
                            # Even in SQL segment, some ops use Pandas
                            lines.append(f" [{counter}] 🐼 [Pandas] {op.describe()}")
                else:
                    # Pandas segment
                    if isinstance(op, LazyRelationalOp):
                        lines.append(f" [{counter}] 🐼 [Pandas] {op.describe_pandas()}")
                    else:
                        engine = op.execution_engine()
                        if engine == "chDB":
                            lines.append(f" [{counter}] 🚀 [chDB] {op.describe()}")
                        else:
                            lines.append(f" [{counter}] 🐼 [Pandas] {op.describe()}")

        # ========== Legacy operation history (for pandas compat operations) ==========
        history_lazy_ops, mat_op, history_executed_ops = (
            self._analyze_execution_phases()
        )

        if mat_op:
            lines.append("\nExecution Point:")
            lines.append("─" * 80)
            counter += 1
            lines.append(f" [{counter}] 🔄 {mat_op['description']}")
            lines.append("     └─> Executes SQL query and caches result as DataFrame")
            if verbose and mat_op.get("details"):
                for k, v in mat_op["details"].items():
                    lines.append(f"         • {k}: {v}")

        if history_executed_ops:
            lines.append("\nPost-Execution Operations:")
            lines.append("─" * 80)
            lines.extend(
                self._render_operations(history_executed_ops, counter + 1, verbose)
            )
            counter += len(history_executed_ops)

        # ========== Final State ==========
        lines.append("\n" + "─" * 80)
        if self._lazy_ops or self._has_sql_state():
            lines.append("Final State: 📊 Pending (lazy, not yet executed)")
            lines.append(
                "             └─> Will execute when print(), .to_df(), .execute() is called"
            )
        else:
            lines.append("Final State: 📊 No operations recorded")
            lines.append(
                "             └─> Start by loading data or defining operations"
            )

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
                    lines.append(
                        "[Query truncated. Use explain(verbose=True) for full query]\n"
                    )
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

    def _get_accessible_columns(self) -> Optional[set]:
        """
        Get the set of columns that are accessible for column access operations.

        After select() with specific columns (not *), only those columns and
        any computed columns added afterwards should be accessible. This matches
        pandas behavior where selecting columns restricts what can be referenced.

        This method also tracks column name transformations through lazy ops
        like add_prefix, add_suffix, and rename.

        Returns:
            Set of accessible column names, or None if all columns are accessible
            (e.g., SELECT * or no select operation)
        """
        # Check if there's a SELECT operation with specific fields (not *)
        if self._select_star:
            return None  # All columns accessible

        if not self._select_fields:
            return None  # No restriction

        # If there are joins, we can't restrict columns since joins add new columns
        if hasattr(self, "_joins") and self._joins:
            return None

        # Build list of accessible columns from select fields (order matters for transforms)
        accessible = []
        for f in self._select_fields:
            if isinstance(f, str):
                accessible.append(f)
            elif hasattr(f, "alias") and f.alias:
                # Expression with alias takes priority (e.g., computed columns, as_())
                accessible.append(f.alias)
            elif hasattr(f, "name"):
                # Field or similar with name attribute
                name = f.name.strip('"') if isinstance(f.name, str) else str(f.name)
                accessible.append(name)

        # Apply column transformations and add computed columns from lazy ops
        from .lazy_ops import LazyColumnAssignment, LazyJoin

        for op in self._lazy_ops:
            if isinstance(op, LazyJoin):
                # Joins add columns from other tables - we can't restrict access
                # since we don't know what columns the join will add
                return None
            elif isinstance(op, LazyColumnAssignment):
                # Add new computed column
                accessible.append(op.column)
            elif hasattr(op, "transform_columns"):
                # Apply column name transformation (add_prefix, add_suffix, rename)
                accessible = op.transform_columns(accessible)

        # Add any columns from _computed_columns tracking
        if hasattr(self, "_computed_columns"):
            for col in self._computed_columns.keys():
                if col not in accessible:
                    accessible.append(col)

        return set(accessible) if accessible else None

    def _is_column_accessible(self, column_name: str) -> bool:
        """
        Check if a column is accessible for column operations.

        This enforces pandas-like behavior where select() restricts accessible columns.

        Args:
            column_name: Name of the column to check

        Returns:
            True if the column is accessible, False otherwise
        """
        accessible = self._get_accessible_columns()
        if accessible is None:
            return True  # No restriction
        return column_name in accessible

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
                "Cache invalid: version mismatch (cached=%d, current=%d)",
                self._cached_at_version,
                self._cache_version,
            )
            return False

        # Check TTL if configured
        ttl = get_cache_ttl()
        if ttl > 0 and self._cache_timestamp is not None:
            age = time.time() - self._cache_timestamp
            if age > ttl:
                self._logger.debug(
                    "Cache invalid: TTL expired (age=%.2fs, ttl=%.2fs)", age, ttl
                )
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
                    self._logger.debug(
                        "Using cached result (version=%d)", self._cached_at_version
                    )
                    if is_profiling_enabled():
                        profiler.log_report()
                    return self._cached_result

            self._logger.debug("=" * 70)
            self._logger.debug("Starting execution (version=%d)", self._cache_version)
            self._logger.debug("=" * 70)

            # Log all lazy operations
            if self._lazy_ops:
                self._logger.debug(
                    "Lazy operations chain (%d operations):", len(self._lazy_ops)
                )
                for i, op in enumerate(self._lazy_ops):
                    self._logger.debug("  [%d] %s", i + 1, op.describe())
            else:
                self._logger.debug("No lazy operations recorded")

            # Segmented planning phase
            with profiler.step("Query Planning", ops_count=len(self._lazy_ops)):
                planner = QueryPlanner()
                has_sql_source = bool(
                    self._table_function
                    or self.table_name
                    or self._source_df is not None
                )
                schema = self.schema() if has_sql_source else self._schema
                exec_plan = planner.plan_segments(
                    self._lazy_ops, has_sql_source, schema=schema
                )

                self._logger.debug(exec_plan.describe())

            # Execute segments
            df = pd.DataFrame()
            has_executed_pandas = False

            for seg_idx, segment in enumerate(exec_plan.segments):
                seg_num = seg_idx + 1
                self._logger.debug("-" * 70)
                self._logger.debug(
                    "Segment %d/%d: %s",
                    seg_num,
                    len(exec_plan.segments),
                    segment.describe(),
                )
                self._logger.debug("-" * 70)

                if segment.is_sql():
                    with profiler.step(f"SQL Segment {seg_num}", ops=len(segment.ops)):
                        df = self._execute_sql_segment(segment, df, schema, profiler)
                else:
                    with profiler.step(
                        f"Pandas Segment {seg_num}", ops=len(segment.ops)
                    ):
                        df = self._execute_pandas_segment(segment, df, profiler)
                        has_executed_pandas = True

            self._logger.debug("=" * 70)
            self._logger.debug(
                "Execution complete. Final DataFrame shape: %s", df.shape
            )
            self._logger.debug("=" * 70)

            # Cache the result if caching is enabled
            with profiler.step("Cache Write"):
                if is_cache_enabled():
                    self._cached_result = df
                    self._cache_timestamp = time.time()

                    # Checkpoint if we executed any Pandas operations or multiple SQL segments
                    # This enables incremental execution for future operations
                    needs_checkpoint = (
                        has_executed_pandas or exec_plan.sql_segment_count() > 1
                    )

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

                        self._logger.debug(
                            "Pipeline checkpointed: lazy_ops replaced with DataFrame source"
                        )
                    else:
                        self._logger.debug("Pure SQL execution: SQL state preserved")

                    self._cache_version = 0
                    self._cached_at_version = 0

        if is_profiling_enabled():
            profiler.log_report()

        # Restore index if we tracked index info during _ensure_sql_source
        # Skip if DataFrame already has the correct index (e.g., from set_index(drop=False))
        if self._index_info is not None and not df.empty:
            if self._index_info.get("is_multiindex"):
                # Restore MultiIndex
                index_names = self._index_info["names"]
                # Check if DataFrame already has this MultiIndex
                current_index_names = (
                    list(df.index.names) if isinstance(df.index, pd.MultiIndex) else []
                )
                if current_index_names != index_names:
                    # Check if all index columns exist in the result
                    existing_index_cols = [n for n in index_names if n in df.columns]
                    if existing_index_cols:
                        df = df.set_index(existing_index_cols)
            elif self._index_info.get("is_noncontiguous"):
                # Non-contiguous index was reset for SQL execution
                # The result already has correct row order via _row_id handling in query_df
                # Just ensure the DataFrame has a clean RangeIndex (which it should have)
                if not df.index.equals(pd.RangeIndex(len(df))):
                    df = df.reset_index(drop=True)
            else:
                # Restore single index
                index_name = self._index_info.get("name")
                # Check if DataFrame already has this index
                if index_name is not None and df.index.name != index_name:
                    if index_name in df.columns:
                        df = df.set_index(index_name)

        return df

    def _execute_sql_segment(
        self,
        segment: "ExecutionSegment",
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

        # Ensure SQL source is available (creates PythonTableFunction on-demand for from_df())
        if segment.is_first_segment:
            self._ensure_sql_source()

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

            self._logger.debug(
                "  Executing SQL: %s", sql[:200] + "..." if len(sql) > 200 else sql
            )

            # For PythonTableFunction, use query_df to ensure row order preservation
            # (chDB's parallel execution can cause non-deterministic row ordering)
            # EXCEPTION: If SQL already contains row order handling (e.g., __orig_row_num__
            # from nested subqueries), skip query_df's additional _row_id processing
            from .table_functions import PythonTableFunction

            sql_has_row_order_handling = (
                "__orig_row_num__" in sql or "ORDER BY _row_id" in sql
            )

            if (
                isinstance(self._table_function, PythonTableFunction)
                and not sql_has_row_order_handling
            ):
                with profiler.step("SQL Execution (query_df)"):
                    # Use executor.query_dataframe which calls connection.query_df
                    # with preserve_order=True for deterministic row ordering
                    from .executor import get_executor

                    executor = get_executor()
                    df = executor.query_dataframe(
                        sql, self._table_function._df, "__df__", preserve_order=True
                    )
                    df = self._postprocess_sql_result(df, plan)
            else:
                # For non-Python table functions (file, URL, etc.), or SQL that already
                # has row order handling, use direct execution
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
                # Even for empty DataFrames, apply column selection if present
                # This ensures df[['a', 'b']] on empty df returns correct columns
                plan = segment.plan
                if plan is None:
                    from .query_planner import QueryPlan

                    plan = QueryPlan(has_sql_source=True)
                    plan.sql_ops = segment.ops.copy()
                # Check if there's a SELECT operation for column selection
                select_cols = None
                computed_cols = []
                for op in plan.sql_ops or []:
                    if hasattr(op, "op_type") and op.op_type == "SELECT":
                        if hasattr(op, "fields") and op.fields:
                            select_cols = []
                            has_star = False
                            for f in op.fields:
                                if isinstance(f, str):
                                    col = f.strip('"')
                                    if col == "*":
                                        has_star = True
                                    else:
                                        select_cols.append(col)
                                elif hasattr(f, "alias") and f.alias:
                                    select_cols.append(f.alias)
                                elif hasattr(f, "name"):
                                    select_cols.append(f.name.strip('"'))
                                # Skip fields without identifiable names
                            # If '*' was in fields, expand it to all existing columns
                            if has_star:
                                select_cols = list(df.columns) + [
                                    c for c in select_cols if c not in df.columns
                                ]
                    # Also check LazyColumnAssignment for computed columns
                    elif hasattr(op, "column") and hasattr(op, "can_push_to_sql"):
                        # LazyColumnAssignment
                        if op.can_push_to_sql():
                            computed_cols.append(op.column)

                # Check if SELECT has any new expressions (not just column references)
                # e.g., assign() creates SELECT *, expression AS alias
                has_new_expressions = False
                for op in plan.sql_ops or []:
                    if hasattr(op, "op_type") and op.op_type == "SELECT":
                        if hasattr(op, "fields") and op.fields:
                            for f in op.fields:
                                if not isinstance(f, str):
                                    # Non-string field = expression (need SQL execution)
                                    has_new_expressions = True
                                    break

                # If we have computed columns or new expressions, execute SQL for correct dtypes
                # chDB can correctly infer output types even for empty DataFrames
                if computed_cols or has_new_expressions:
                    self._logger.debug(
                        "  Empty df has computed columns/expressions, executing SQL for correct dtypes"
                    )
                    # Fall through to execute SQL below
                else:
                    # No computed columns, just apply column selection if present
                    if select_cols:
                        existing_cols = [c for c in select_cols if c in df.columns]
                        if existing_cols:
                            df = df[existing_cols]
                        self._logger.debug(
                            "  Applied column selection to empty df: %s", select_cols
                        )
                    return df

            # If we have a source but no DataFrame yet, load raw data first
            if df.empty and (self._table_function or self.table_name):
                if self._executor is None:
                    self.connect()
                # Load raw data without any filters
                if self._table_function:
                    table_sql = self._table_function.to_sql()
                else:
                    table_sql = f"{self.quote_char}{self.table_name}{self.quote_char}"
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
        segment: "ExecutionSegment",
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
                table_sql = f"{self.quote_char}{self.table_name}{self.quote_char}"

            # Build FROM clause with optional alias for joins
            if self._joins:
                alias = self._get_table_alias()
                from_clause = (
                    f"{table_sql} AS {self.quote_char}{alias}{self.quote_char}"
                )
            else:
                from_clause = table_sql

            load_sql = f"SELECT * FROM {from_clause}"

            # Add JOIN clauses if present
            if self._joins:
                sql_engine = SQLExecutionEngine(self)
                for other_ds, join_type, join_condition in self._joins:
                    join_clause = sql_engine._build_join_clause(
                        other_ds, join_type, join_condition
                    )
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
            self._logger.debug(
                "  Loaded raw data from source for Pandas: %s rows", len(df)
            )

        for i, op in enumerate(segment.ops, 1):
            self._logger.debug(
                "  [%d/%d] Executing: %s", i, len(segment.ops), op.describe()
            )
            op_name = op.__class__.__name__
            with profiler.step(op_name):
                df = op.execute(df, self)

        return df

    def _postprocess_sql_result(
        self, df: pd.DataFrame, plan: "QueryPlan"
    ) -> pd.DataFrame:
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
            rename_back = {
                temp: orig
                for temp, orig in plan.alias_renames.items()
                if temp in df.columns
            }
            if rename_back:
                df = df.rename(columns=rename_back)
                self._logger.debug("  Renamed temp aliases: %s", rename_back)

        # For GroupBy SQL pushdown: set group keys as index
        if plan.groupby_agg and plan.groupby_agg.groupby_cols:
            groupby_cols = plan.groupby_agg.groupby_cols
            # Check as_index: only set index if as_index=True (default)
            as_index = getattr(plan.groupby_agg, "as_index", True)
            if as_index and all(col in df.columns for col in groupby_cols):
                df = df.set_index(groupby_cols)
                self._logger.debug("  Set groupby columns as index: %s", groupby_cols)

            # Convert flat column names to MultiIndex for pandas compatibility
            # Skip for single_column_agg which should return flat column names
            if plan.groupby_agg.agg_dict:
                is_single_col_agg = getattr(
                    plan.groupby_agg, "single_column_agg", False
                )
                if not is_single_col_agg:
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
                                new_columns.append((c, ""))
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
        needs_subquery_for_stable = (
            needs_stable_sort and where_conditions and not groupby_fields and not joins
        )

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
        distinct_keyword = "DISTINCT " if distinct else ""
        if select_fields:
            fields_sql = ", ".join(
                f.to_sql(quote_char=self.quote_char, with_alias=True)
                for f in select_fields
            )
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
            if hasattr(self._table_function, "to_sql"):
                table_sql = self._table_function.to_sql()
            else:
                table_sql = str(self._table_function)
            # Add alias when joins are present (required by ClickHouse for disambiguation)
            if joins:
                alias = self._get_table_alias()
                parts.append(
                    f"FROM {table_sql} AS {format_identifier(alias, self.quote_char)}"
                )
            else:
                parts.append(f"FROM {table_sql}")
        elif self.table_name:
            parts.append(f"FROM {self.quote_char}{self.table_name}{self.quote_char}")

        # JOIN clauses
        if joins:
            for other_ds, join_type, join_condition in joins:
                # Generate JOIN clause
                join_keyword = join_type.value if join_type.value else ""
                if join_keyword:
                    join_clause = f"{join_keyword} JOIN"
                else:
                    join_clause = "JOIN"

                # Handle subquery joins
                if isinstance(other_ds, DataStore) and other_ds._is_subquery:
                    other_table = other_ds.to_sql(
                        quote_char=self.quote_char, as_subquery=True
                    )
                elif isinstance(other_ds, DataStore) and other_ds._table_function:
                    # Use table function for the joined table with alias
                    table_func_sql = other_ds._table_function.to_sql(
                        quote_char=self.quote_char
                    )
                    alias = other_ds._get_table_alias()
                    other_table = f"{table_func_sql} AS {format_identifier(alias, self.quote_char)}"
                else:
                    other_table = format_identifier(
                        other_ds.table_name, self.quote_char
                    )

                # Handle USING vs ON syntax
                if isinstance(join_condition, tuple) and join_condition[0] == "USING":
                    # USING (col1, col2, ...) syntax
                    columns = join_condition[1]
                    using_cols = ", ".join(
                        format_identifier(c, self.quote_char) for c in columns
                    )
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
            groupby_sql = ", ".join(
                f.to_sql(quote_char=self.quote_char) for f in groupby_fields
            )
            parts.append(f"GROUP BY {groupby_sql}")

        # HAVING
        if having_condition:
            having_sql = having_condition.to_sql(quote_char=self.quote_char)
            parts.append(f"HAVING {having_sql}")

        # ORDER BY (stable sort if kind='stable' or 'mergesort', matching pandas behavior)
        if orderby_fields:
            # Check for special row order marker (must be a string, not a Field object)
            first_field = orderby_fields[0][0]
            if (
                len(orderby_fields) == 1
                and isinstance(first_field, str)
                and first_field == "__rowNumberInAllBlocks__"
            ):
                # Special case: use rowNumberInAllBlocks() for row order preservation
                parts.append("ORDER BY rowNumberInAllBlocks()")
            else:
                orderby_sql = build_orderby_clause(
                    orderby_fields,
                    self.quote_char,
                    stable=is_stable_sort(self._orderby_kind),
                )
                parts.append(f"ORDER BY {orderby_sql}")

        # LIMIT
        if limit_value is not None:
            parts.append(f"LIMIT {limit_value}")

        # OFFSET
        if offset_value is not None:
            parts.append(f"OFFSET {offset_value}")

        return " ".join(parts)

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
            if hasattr(self._table_function, "to_sql"):
                table_sql = self._table_function.to_sql()
            else:
                table_sql = str(self._table_function)
        elif self.table_name:
            table_sql = f"{self.quote_char}{self.table_name}{self.quote_char}"
        else:
            table_sql = "source"

        inner_sql = (
            f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_sql}"
        )

        # Build middle query: SELECT columns FROM (inner) WHERE conditions
        middle_parts = []
        distinct_keyword = "DISTINCT " if distinct else ""

        if select_fields:
            fields_sql = ", ".join(
                f.to_sql(quote_char=self.quote_char, with_alias=True)
                for f in select_fields
            )
            if self._select_star:
                middle_parts.append(
                    f"SELECT {distinct_keyword}*, {fields_sql}, __orig_row_num__"
                )
            else:
                middle_parts.append(
                    f"SELECT {distinct_keyword}{fields_sql}, __orig_row_num__"
                )
        else:
            middle_parts.append(f"SELECT {distinct_keyword}*, __orig_row_num__")

        middle_parts.append(f"FROM ({inner_sql}) AS __subq_with_rownum__")

        # WHERE
        if where_conditions:
            combined = where_conditions[0]
            for cond in where_conditions[1:]:
                combined = combined & cond
            middle_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        middle_sql = " ".join(middle_parts)

        # Build outer query: SELECT * EXCEPT(__orig_row_num__) FROM (middle) ORDER BY ... LIMIT
        outer_parts = []
        outer_parts.append("SELECT * EXCEPT(__orig_row_num__)")
        outer_parts.append(f"FROM ({middle_sql}) AS __subq_for_stable_sort__")

        # ORDER BY with __orig_row_num__ as tie-breaker instead of rowNumberInAllBlocks()
        if orderby_fields:
            orderby_sql = build_orderby_clause(
                orderby_fields,
                self.quote_char,
                stable=False,  # Don't add rowNumberInAllBlocks()
            )
            outer_parts.append(f"ORDER BY {orderby_sql}, __orig_row_num__ ASC")

        # LIMIT
        if limit_value is not None:
            outer_parts.append(f"LIMIT {limit_value}")

        # OFFSET
        if offset_value is not None:
            outer_parts.append(f"OFFSET {offset_value}")

        return " ".join(outer_parts)

    # ========== Static Factory Methods for Data Sources ==========

    @classmethod
    def from_file(
        cls,
        path: str,
        format: str = None,
        structure: str = None,
        compression: str = None,
        **kwargs,
    ) -> "DataStore":
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

        ds = cls(
            "file",
            path=path,
            format=format,
            structure=structure,
            compression=compression,
            **kwargs,
        )

        # Use UTC timezone to ensure datetime values match pandas (which uses naive UTC)
        # New versions of chDB may apply system timezone, causing value shifts
        ds._format_settings["session_timezone"] = "UTC"

        # For Parquet files, enable row order preservation to match pandas behavior
        # chDB may read row groups in parallel which can reorder rows
        if format and format.lower() == "parquet":
            ds._format_settings["input_format_parquet_preserve_order"] = 1

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
    ) -> "DataStore":
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
    def from_hdfs(
        cls, uri: str, format: str = None, structure: str = None, **kwargs
    ) -> "DataStore":
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
    def from_mysql(
        cls,
        host: str,
        database: str = None,
        table: str = None,
        user: str = None,
        password: str = "",
        **kwargs,
    ) -> "DataStore":
        """
        Create DataStore from MySQL database.

        Args:
            host: MySQL server address (host:port)
            database: Database name (optional - can be set later with use())
            table: Table name (optional - can be set later with ds["db.table"])
            user: Username
            password: Password
            **kwargs: Additional parameters

        Examples:
            >>> # Connection level - browse metadata
            >>> ds = DataStore.from_mysql("localhost:3306", user="root", password="pass")
            >>> ds.databases()  # ['db1', 'db2', ...]
            >>> ds.tables("mydb")  # ['users', 'orders', ...]

            >>> # Table level - ready for queries
            >>> ds = DataStore.from_mysql("localhost:3306", "mydb", "users",
            ...                           user="root", password="pass")
        """
        return cls(
            "mysql",
            host=host,
            database=database,
            table=table,
            user=user,
            password=password,
            **kwargs,
        )

    @classmethod
    def from_postgresql(
        cls,
        host: str,
        database: str = None,
        table: str = None,
        user: str = None,
        password: str = "",
        **kwargs,
    ) -> "DataStore":
        """
        Create DataStore from PostgreSQL database.

        Args:
            host: PostgreSQL server address (host:port)
            database: Database name (optional - can be set later with use())
            table: Table name (can include schema like 'schema.table', optional)
            user: Username
            password: Password
            **kwargs: Additional parameters

        Examples:
            >>> # Connection level - browse metadata
            >>> ds = DataStore.from_postgresql("localhost:5432", user="postgres", password="pass")
            >>> ds.databases()  # ['db1', 'db2', ...]
            >>> ds.tables("mydb")  # ['users', 'orders', ...]

            >>> # Table level - ready for queries
            >>> ds = DataStore.from_postgresql("localhost:5432", "mydb", "users",
            ...                                user="postgres", password="pass")
        """
        return cls(
            "postgresql",
            host=host,
            database=database,
            table=table,
            user=user,
            password=password,
            **kwargs,
        )

    @classmethod
    def from_clickhouse(
        cls,
        host: str,
        database: str = None,
        table: str = None,
        user: str = "default",
        password: str = "",
        secure: bool = False,
        **kwargs,
    ) -> "DataStore":
        """
        Create DataStore from remote ClickHouse server.

        Args:
            host: ClickHouse server address (host:port)
            database: Database name (optional - can be set later with use())
            table: Table name (optional - can be set later with ds["db.table"])
            user: Username (default: 'default')
            password: Password
            secure: Use secure connection (remoteSecure)
            **kwargs: Additional parameters

        Examples:
            >>> # Connection level - browse metadata
            >>> ds = DataStore.from_clickhouse("localhost:9000", user="default", password="")
            >>> ds.databases()  # ['default', 'system', ...]
            >>> ds.tables("default")  # ['events', 'users', ...]

            >>> # Table level - ready for queries
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
        cls,
        host: str,
        database: str,
        collection: str,
        user: str,
        password: str = "",
        **kwargs,
    ) -> "DataStore":
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
            "mongodb",
            host=host,
            database=database,
            collection=collection,
            user=user,
            password=password,
            **kwargs,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        format: str = None,
        structure: str = None,
        headers: List[str] = None,
        **kwargs,
    ) -> "DataStore":
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
        return cls(
            "url",
            url=url,
            format=format,
            structure=structure,
            headers=headers,
            **kwargs,
        )

    @classmethod
    def from_sqlite(cls, database_path: str, table: str, **kwargs) -> "DataStore":
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
    def from_iceberg(
        cls,
        url: str,
        access_key_id: str = None,
        secret_access_key: str = None,
        **kwargs,
    ) -> "DataStore":
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
        return cls(
            "iceberg",
            url=url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            **kwargs,
        )

    @classmethod
    def from_delta(
        cls,
        url: str,
        access_key_id: str = None,
        secret_access_key: str = None,
        **kwargs,
    ) -> "DataStore":
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
        return cls(
            "delta",
            url=url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            **kwargs,
        )

    @classmethod
    def from_numbers(
        cls, count: int, start: int = None, step: int = None, **kwargs
    ) -> "DataStore":
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
        cls,
        connection_string: str,
        container: str,
        path: str = "",
        format: str = None,
        **kwargs,
    ) -> "DataStore":
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
            "azure",
            connection_string=connection_string,
            container=container,
            path=path,
            format=format,
            **kwargs,
        )

    @classmethod
    def from_gcs(
        cls,
        url: str,
        hmac_key: str = None,
        hmac_secret: str = None,
        format: str = None,
        nosign: bool = False,
        **kwargs,
    ) -> "DataStore":
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
        return cls(
            "gcs",
            url=url,
            hmac_key=hmac_key,
            hmac_secret=hmac_secret,
            format=format,
            nosign=nosign,
            **kwargs,
        )

    @classmethod
    def from_redis(
        cls,
        host: str,
        key: str,
        structure: str,
        password: str = None,
        db_index: int = 0,
        **kwargs,
    ) -> "DataStore":
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
        return cls(
            "redis",
            host=host,
            key=key,
            structure=structure,
            password=password,
            db_index=db_index,
            **kwargs,
        )

    @classmethod
    def from_hudi(
        cls,
        url: str,
        access_key_id: str = None,
        secret_access_key: str = None,
        **kwargs,
    ) -> "DataStore":
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
        return cls(
            "hudi",
            url=url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            **kwargs,
        )

    @classmethod
    def from_random(
        cls,
        structure: str,
        random_seed: int = None,
        max_string_length: int = None,
        max_array_length: int = None,
        **kwargs,
    ) -> "DataStore":
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
    def from_df(cls, df, name: str = None) -> "DataStore":
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
        import numpy as np
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
    def from_dataframe(cls, df, name: str = None) -> "DataStore":
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
        import numpy as np
                    >>> df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
                    >>> ds = DataStore.from_dataframe(df)
                    >>> ds.filter(ds.age > 26).to_df()
        """
        return cls.from_df(df, name=name)

    @classmethod
    def uri(cls, uri: str, **kwargs) -> "DataStore":
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
        table = final_kwargs.pop("table", None)

        # Create and return DataStore
        return cls(source=source_type, table=table, **final_kwargs)

    # ========== Data Source Operations ==========

    def with_format_settings(self, **settings) -> "DataStore":
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

    def connect(self, test_connection: bool = True) -> "DataStore":
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
                self._logger.debug(
                    "Created connection with params: %s", self.connection_params
                )

        try:
            self._connection.connect()
            self._executor = Executor(self._connection)
            self._logger.debug("Connection established successfully")

            # Try to get schema if table exists (but not for table functions)
            # For table functions, schema is discovered via _discover_table_function_schema()
            if self._table_function is None and self.table_name:
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

            self._logger.debug(
                "Discovering schema for table function: %s", describe_sql
            )
            result = self._executor.execute(describe_sql)

            # Build schema dictionary from DESCRIBE result
            self._schema = {}
            for row in result.rows:
                # ClickHouse DESCRIBE returns: (name, type, default_type, default_expression, comment, ...)
                col_name = row[0]
                col_type = row[1]
                self._schema[col_name] = col_type

            self._logger.debug(
                "Data source is accessible, schema discovered: %s", self._schema
            )
        except Exception as e:
            self._logger.error("Data source not accessible: %s", e)
            raise ConnectionError(f"Data source not accessible: {e}")

    def _get_table_alias(self) -> str:
        """
        Get a short alias for table functions.

        For file table functions, extracts filename without extension.
        For other table functions, uses table name or a generic name.
        """
        if self._table_function and hasattr(self._table_function, "params"):
            # Try to get a meaningful alias from path
            path = self._table_function.params.get("path")
            if path:
                import os

                # Extract filename without extension
                basename = os.path.basename(path)
                name_without_ext = os.path.splitext(basename)[0]
                return name_without_ext

            # For other table functions, try to use table name
            table = self._table_function.params.get("table")
            if table:
                return table

            # For numbers or other generators
            if hasattr(self._table_function, "__class__"):
                class_name = self._table_function.__class__.__name__.replace(
                    "TableFunction", ""
                ).lower()
                return class_name

        # Fallback to table name or generic
        return self.table_name if self.table_name else "tbl"

    def _resolve_expr_dependencies(
        self, expr: "Expression", _visited: set = None
    ) -> "Expression":
        """
        Recursively resolve computed column references in an expression.

        When a computed column (created via assign()) is referenced in another
        expression, this method expands it to the original expression.

        This enables chained assigns like:
            ds = ds.assign(years_since_signup=2024 - ds['signup_year'])
            ds = ds.assign(purchase_per_year=ds['amount'] / ds['years_since_signup'])

        Without this resolution, the second assign would reference 'years_since_signup'
        as a simple Field, which doesn't exist in the source data.

        Args:
            expr: Expression to resolve
            _visited: Internal set to track visited column names and prevent infinite recursion
                     when a computed column references itself (e.g., a = a + 1, then a = a * 2)

        Returns:
            Expression with all computed column references expanded
        """
        from .expressions import Field, ArithmeticExpression, Literal
        from .functions import Function
        from copy import copy

        if expr is None:
            return None

        # Initialize visited set on first call
        if _visited is None:
            _visited = set()

        # If it's a Field that references a computed column, expand it
        if isinstance(expr, Field):
            col_name = expr.name.strip("\"'")  # Remove quotes if present
            if (
                hasattr(self, "_computed_columns")
                and col_name in self._computed_columns
            ):
                # Check for circular reference to prevent infinite recursion
                # This happens when overwriting a column: a = a + 1, then a = a * 2
                # In this case, we should NOT expand the reference - just use the Field as-is
                if col_name in _visited:
                    return expr
                # Mark this column as being visited
                _visited.add(col_name)
                try:
                    # Recursively resolve the computed column's expression
                    resolved = self._resolve_expr_dependencies(
                        self._computed_columns[col_name], _visited
                    )
                    # Preserve any alias from the original Field
                    if expr.alias:
                        resolved = copy(resolved)
                        resolved.alias = expr.alias
                    return resolved
                finally:
                    # Remove from visited after processing (allow re-visiting in different branches)
                    _visited.discard(col_name)
            return expr

        # For ArithmeticExpression, recursively resolve left and right
        if isinstance(expr, ArithmeticExpression):
            resolved_left = self._resolve_expr_dependencies(expr.left, _visited)
            resolved_right = self._resolve_expr_dependencies(expr.right, _visited)
            # Only create new expression if something changed
            if resolved_left is expr.left and resolved_right is expr.right:
                return expr
            result = ArithmeticExpression(
                expr.operator, resolved_left, resolved_right, expr.alias
            )
            return result

        # For Function, recursively resolve all arguments
        if isinstance(expr, Function):
            resolved_args = [
                self._resolve_expr_dependencies(arg, _visited) for arg in expr.args
            ]
            # Only create new function if something changed
            if all(r is o for r, o in zip(resolved_args, expr.args)):
                return expr
            result = Function(
                expr.name,
                *resolved_args,
                alias=expr.alias,
                pandas_name=getattr(expr, "pandas_name", None),
                pandas_kwargs=getattr(expr, "pandas_kwargs", {}),
            )
            return result

        # For BinaryCondition (e.g., x > 5, a = b), resolve both sides
        from .conditions import BinaryCondition, CompoundCondition

        if isinstance(expr, BinaryCondition):
            resolved_left = self._resolve_expr_dependencies(expr.left, _visited)
            resolved_right = self._resolve_expr_dependencies(expr.right, _visited)
            if resolved_left is expr.left and resolved_right is expr.right:
                return expr
            result = BinaryCondition(
                expr.operator, resolved_left, resolved_right, expr.alias
            )
            return result

        # For CompoundCondition (AND, OR), resolve both sides
        if isinstance(expr, CompoundCondition):
            resolved_left = self._resolve_expr_dependencies(expr.left, _visited)
            resolved_right = self._resolve_expr_dependencies(expr.right, _visited)
            if resolved_left is expr.left and resolved_right is expr.right:
                return expr
            result = CompoundCondition(expr.operator, resolved_left, resolved_right)
            return result

        # For Literal and other types, return as-is
        return expr

    def _ensure_sql_source(self) -> bool:
        """
        Ensure we have a SQL source (table function or table name) for SQL operations.

        If this DataStore was created from a DataFrame via from_df() and doesn't
        have a table function yet, creates a PythonTableFunction on-demand.

        Index preservation: If the source DataFrame has a custom index (from set_index()),
        the index is reset and stored as a regular column. The index info is tracked in
        _index_info so it can be restored after SQL execution.

        Returns:
            True if SQL source is available, False otherwise
        """
        # Already have SQL source
        if self._table_function or self.table_name:
            return True

        # Check if we have a cached DataFrame from from_df()
        if self._source_df is not None:
            from .table_functions import PythonTableFunction
            from .lazy_ops import (
                LazyDataFrameSource,
                LazyRelationalOp,
                LazyColumnSelection,
            )

            # Use the unified column tracking method to get current columns
            # This handles SELECT, drop, rename, and other column-changing operations
            current_cols = self._get_current_columns()
            source_cols = list(self._source_df.columns)

            # Check if columns have changed from the source
            df_to_use = self._source_df
            columns_changed = (
                set(current_cols) != set(source_cols) or current_cols != source_cols
            )

            if columns_changed and current_cols:
                # Apply column selection to the DataFrame
                existing_cols = [c for c in current_cols if c in df_to_use.columns]
                if existing_cols:
                    df_to_use = df_to_use[existing_cols]
                    self._source_df = df_to_use

            # Preserve index info if DataFrame has a custom index
            # A custom index is one with a name (from set_index()) or a MultiIndex
            has_custom_index = (
                df_to_use.index.name is not None
                or isinstance(df_to_use.index, pd.MultiIndex)
                or (
                    hasattr(df_to_use.index, "names")
                    and any(n is not None for n in df_to_use.index.names)
                )
            )

            # Also check for non-contiguous index (e.g., after dropna, filter, etc.)
            # chDB's Python() table function requires contiguous 0..n-1 index for correct data access
            has_noncontiguous_index = len(df_to_use) > 0 and not df_to_use.index.equals(
                pd.RangeIndex(len(df_to_use))
            )

            if has_custom_index:
                # Store index info for restoration after SQL execution
                if isinstance(df_to_use.index, pd.MultiIndex):
                    self._index_info = {
                        "names": list(df_to_use.index.names),
                        "is_multiindex": True,
                    }
                else:
                    self._index_info = {
                        "name": df_to_use.index.name,
                        "is_multiindex": False,
                    }
                # Reset index to include it as a column for SQL processing
                df_to_use = df_to_use.reset_index()
            elif has_noncontiguous_index:
                # Non-contiguous index without a name (e.g., from dropna/filter)
                # Store original index for potential restoration
                self._index_info = {
                    "original_index": df_to_use.index.copy(),
                    "is_noncontiguous": True,
                }
                # Reset index to contiguous 0..n-1 for chDB compatibility
                df_to_use = df_to_use.reset_index(drop=True)

            # Create PythonTableFunction on-demand with the (possibly filtered) DataFrame
            self._table_function = PythonTableFunction(
                df=df_to_use, name=self._source_df_name
            )

            # Remove lazy ops that have been "baked" into the DataFrame:
            # - LazyDataFrameSource (will use SQL instead)
            # - SELECT ops that we've already applied to the DataFrame
            # - LazyColumnSelection that we've already applied
            self._lazy_ops = [
                op
                for op in self._lazy_ops
                if not isinstance(op, LazyDataFrameSource)
                and not (
                    isinstance(op, LazyRelationalOp)
                    and op.op_type == "SELECT"
                    and columns_changed
                )
                and not (isinstance(op, LazyColumnSelection) and columns_changed)
            ]
            return True

        return False

    def _discover_schema(self):
        """Discover table schema from chdb."""
        if not self._executor or not self.table_name:
            return

        try:
            # Query system tables for schema info
            sql = (
                f"DESCRIBE TABLE {format_identifier(self.table_name, self.quote_char)}"
            )
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

    def _get_current_columns(self) -> List[str]:
        """
        Get the current column list after applying lazy operations.

        This method tracks how columns change through the lazy op chain
        WITHOUT executing the operations. Used by _ensure_sql_source() and
        other methods that need to know the current column state.

        Returns:
            List of column names after all lazy ops would be applied
        """
        # Start with source DataFrame columns
        if self._source_df is not None:
            columns = list(self._source_df.columns)
        elif self._schema:
            columns = list(self._schema.keys())
        else:
            columns = []

        # Apply column transformations from lazy ops
        for op in self._lazy_ops:
            columns = op.transform_columns(columns)

        return columns

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
        if self._schema and not all(v == "Unknown" for v in self._schema.values()):
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
                    col_type = row[1] if len(row) > 1 else "Unknown"
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
                    col_type = row[1] if len(row) > 1 else "Unknown"
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
            self._schema = {col: "Unknown" for col in columns}
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

    def to_dict(self, orient: str = "dict", *, into=dict, index: bool = True):
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters.

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'tight' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
              'index_names' -> [index.names], 'column_names' -> [column.names]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

        into : class, default dict
            The collections.abc.MutableMapping subclass used for all Mappings
            in the return value.

        index : bool, default True
            Whether to include the index item (and index_names item if `orient`
            is 'tight') in the returned dictionary. Can only be ``False``
            when `orient` is 'split' or 'tight'.

        Returns
        -------
        dict, list or collections.abc.MutableMapping
            Return a collections.abc.MutableMapping object representing the
            DataFrame. The resulting transformation depends on the `orient`
            parameter.

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> records = ds.select("*").filter(ds.age > 18).to_dict(orient='records')
        """
        return self._execute().to_dict(orient=orient, into=into, index=index)

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

    def describe(self, *args, percentiles=None, include=None, exclude=None, **kwargs):
        """
        Describe data - works for both local statistics and remote table schema.

        **For local DataFrames (table mode):**
        Generate descriptive statistics (count, mean, std, min, max, percentiles).

        **For remote connections (connection/database mode):**
        Get remote table schema (columns and types) when database/table args provided.

        Args:
            For local stats:
                percentiles: List of percentiles to include (default: [.25, .5, .75])
                include: Data types to include ('all', None, or list of dtypes)
                exclude: Data types to exclude (None or list of dtypes)
            For remote schema:
                database (str): Database name
                table (str): Table name

        Returns:
            DataStore with descriptive statistics (local) or DataFrame with schema (remote)

        Example (local):
            >>> ds = DataStore.from_file("data.csv")
            >>> stats = ds.describe()

        Example (remote):
            >>> ds = DataStore.from_clickhouse(host="localhost:9000", ...)
            >>> schema = ds.describe("mydb", "users")
        """
        # Check if called with database/table args for remote schema
        if (
            args
            and self._connection_mode in ("connection", "database")
            and self._remote_params
        ):
            database = args[0] if len(args) > 0 else kwargs.get("database")
            table = args[1] if len(args) > 1 else kwargs.get("table")
            return self._remote_describe(database, table)

        # Local DataFrame statistics
        # Use pandas compat layer if available, otherwise fallback
        if hasattr(self, "_get_df"):
            df = self._get_df()
        else:
            df = self.to_df()
        result_df = df.describe(
            percentiles=percentiles, include=include, exclude=exclude
        )

        # Wrap result in DataStore
        if hasattr(self, "_wrap_result"):
            return self._wrap_result(result_df, "describe()")
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

        If n is positive, returns the first n rows using SQL LIMIT.
        If n is negative, returns all rows except the last |n| rows,
        matching pandas behavior.

        This method uses lazy execution - the operation is only executed
        when the result is accessed (e.g., via to_df() or print).

        Args:
            n: Number of rows to return (default: 5).
               If negative, returns all rows except the last |n| rows.

        Returns:
            DataStore with the selected rows (lazy - not yet executed)

        Example:
            >>> ds = DataStore({'a': [1, 2, 3, 4, 5]})
            >>> ds.head(3)   # Returns first 3 rows
            >>> ds.head(-2)  # Returns all except last 2 rows (first 3 rows)
        """
        if n >= 0:
            # Positive n: use SQL LIMIT (lazy)
            return self.limit(n)
        else:
            # Negative n: need to exclude last |n| rows
            # Must execute to know total count, then use pandas
            if hasattr(self, "_get_df"):
                df = self._get_df()
            else:
                df = self.to_df()
            result_df = df.head(n)

            if hasattr(self, "_wrap_result"):
                return self._wrap_result(result_df, f"head({n})")
            else:
                return self._wrap_result_fallback(result_df)

    def tail(self, n: int = 5):
        """
        Return the last n rows of the query result.

        If n is positive, returns the last n rows.
        If n is negative, returns all rows except the first |n| rows,
        matching pandas behavior.

        Args:
            n: Number of rows to return (default: 5).
               If negative, returns all rows except the first |n| rows.

        Returns:
            DataStore with the selected rows

        Example:
            >>> ds = DataStore({'a': [1, 2, 3, 4, 5]})
            >>> ds.tail(3)   # Returns last 3 rows
            >>> ds.tail(-2)  # Returns all except first 2 rows (last 3 rows)
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, "_get_df"):
            df = self._get_df()
        else:
            df = self.to_df()
        result_df = df.tail(n)

        # Wrap result in DataStore
        if hasattr(self, "_wrap_result"):
            return self._wrap_result(result_df, f"tail({n})")
        else:
            return self._wrap_result_fallback(result_df)

    def sample(
        self,
        n: int = None,
        frac: float = None,
        replace: bool = False,
        weights=None,
        random_state: int = None,
        axis=None,
        ignore_index: bool = False,
    ):
        """
        Return a random sample of rows from the query result.

        Works correctly with both SQL queries and executed DataFrames.

        Args:
            n: Number of rows to return (mutually exclusive with frac)
            frac: Fraction of rows to return (mutually exclusive with n)
            replace: Allow or disallow sampling with replacement
            weights: Weight values for sampling probability
            random_state: Random seed for reproducibility
            axis: Axis to sample (0 or 'index')
            ignore_index: If True, reset index in result

        Returns:
            DataStore with sampled rows

        Example:
            >>> ds = DataStore.from_file("data.csv")
            >>> sample_10 = ds.select("*").sample(n=10)
            >>> sample_20_percent = ds.select("*").sample(frac=0.2)
            >>> sample_replace = ds.sample(n=150, replace=True)
        """
        # Use _get_df if available (handles caching properly)
        if hasattr(self, "_get_df"):
            df = self._get_df()
        else:
            df = self.to_df()
        result_df = df.sample(
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
            axis=axis,
            ignore_index=ignore_index,
        )

        # Wrap result in DataStore
        sample_desc = f"sample(n={n})" if n is not None else f"sample(frac={frac})"
        if hasattr(self, "_wrap_result"):
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
        if hasattr(self, "_get_df"):
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
        if hasattr(self, "_get_df"):
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
            raise ValueError(
                f"Length mismatch: Expected {len(current_columns)} columns, "
                f"got {len(new_columns)}"
            )

        # Build rename mapping
        rename_mapping = {
            old: new
            for old, new in zip(current_columns, new_columns)
            if old != new  # Only rename if different
        }

        if rename_mapping:
            self._add_lazy_op(LazyRenameColumns(rename_mapping))
            # Invalidate cache since we've added a new operation
            self._cached_result = None
            self._cached_at_version = -1

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
            self._logger.debug(
                "count() falling back to execution due to non-SQL operations"
            )
            if hasattr(self, "_get_df"):
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
                return pd.Series(dtype="int64")

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
                counts = {
                    col: int(val) for col, val in zip(column_names, result.rows[0])
                }
                return pd.Series(counts)
            else:
                # Return Series with zeros
                return pd.Series({col: 0 for col in column_names})

        except Exception as e:
            # Fall back to execution on any error
            self._logger.debug("count() falling back to execution due to error: %s", e)
            if hasattr(self, "_get_df"):
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
            self._logger.debug(
                "count_rows() falling back to execution due to non-SQL operations"
            )
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
        count_ds._select_fields = [Count("*")]
        count_ds._select_star = (
            False  # Must clear _select_star to avoid "SELECT *, COUNT(*)"
        )

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

    def info(
        self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None
    ):
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
        if hasattr(self, "_get_df"):
            df = self._get_df()
        else:
            df = self.to_df()
        return df.info(
            verbose=verbose,
            buf=buf,
            max_cols=max_cols,
            memory_usage=memory_usage,
            show_counts=show_counts,
        )

    def create_table(
        self,
        schema: Dict[str, str],
        engine: str = "Memory",
        drop_if_exists: bool = False,
    ) -> "DataStore":
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
        columns = ", ".join(
            [
                f"{format_identifier(name, self.quote_char)} {dtype}"
                for name, dtype in schema.items()
            ]
        )

        table_ident = format_identifier(self.table_name, self.quote_char)
        sql = f"CREATE TABLE IF NOT EXISTS {table_ident} ({columns}) ENGINE = {engine}"

        self._executor.execute(sql)
        self._schema = schema

        return self

    def insert(
        self,
        loc_or_data=None,
        column=None,
        value=None,
        allow_duplicates=False,
        **columns,
    ) -> None:
        """
        Insert column or rows into the DataStore.

        This method supports two modes:

        1. Pandas-compatible mode (insert column at position):
           >>> ds.insert(loc, column, value, allow_duplicates=False)
           Args:
               loc: Insertion index for the new column
               column: Label of the inserted column
               value: Scalar, Series, or array-like
               allow_duplicates: Allow duplicate column names
           Returns: None (modifies DataStore in place, like pandas)

        2. SQL row insertion mode (insert rows into table):
           >>> ds.insert([{"id": 1, "name": "Alice"}])
           >>> ds.insert(id=1, name="Alice")
           Args:
               data: List of dictionaries with column_name -> value
               **columns: Alternative way to specify columns (for single row)
           Returns: self for chaining

        The mode is auto-detected based on the first argument type:
        - int -> pandas mode
        - list/dict/None with kwargs -> SQL mode
        """
        # Detect pandas mode: first arg is int (loc parameter)
        if isinstance(loc_or_data, int):
            # Pandas-compatible column insertion (inplace operation like pandas)
            loc = loc_or_data
            if column is None:
                raise TypeError("insert() missing required argument: 'column'")
            if value is None:
                raise TypeError("insert() missing required argument: 'value'")

            # Execute any pending lazy operations first
            df = self._get_df().copy()
            df.insert(loc, column, value, allow_duplicates=allow_duplicates)

            # Update internal state to reflect the new column (inplace modification)
            from .lazy_ops import LazyDataFrameSource

            self._source_df = df
            self._cached_result = df
            self._cache_valid = True
            # Add LazyDataFrameSource so count_rows() knows to use execution
            self._lazy_ops = [LazyDataFrameSource(df)]
            self._computed_columns = set()
            # Handle duplicate column names - use iloc for unique column access
            self._schema = {}
            for i, col in enumerate(df.columns):
                col_data = df.iloc[:, i]
                self._schema[col] = str(col_data.dtype)

            # Return None to match pandas behavior
            return None

        # SQL row insertion mode
        data = loc_or_data
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
        col_names = list(data[0].keys())
        columns_sql = ", ".join(
            [format_identifier(col, self.quote_char) for col in col_names]
        )

        # Build values
        values_list = []
        for row in data:
            values = []
            for col in col_names:
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
        """
        Close the connection and reset global state.

        This method closes:
        1. The DataStore's own executor and connection
        2. The global executor (to ensure clean state for next DataStore)
        """
        if self._executor:
            self._executor.close()
            self._executor = None
        if self._connection:
            self._connection.close()
            self._connection = None

        # Also reset the global executor to ensure clean state
        # This prevents state leakage between different DataStore instances
        from .executor import reset_executor

        reset_executor()

    # ========== INSERT/UPDATE/DELETE Query Building ==========

    @immutable
    def insert_into(self, *columns: str) -> "DataStore":
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
    def insert_values(self, *rows) -> "DataStore":
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
    def select_from(self, subquery: "DataStore") -> "DataStore":
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
    def update_set(self, **fields) -> "DataStore":
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
    def delete_rows(self) -> "DataStore":
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
    def select(self, *fields: Union[str, Expression]) -> "DataStore":
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
            or (
                isinstance(f, ColumnExpr)
                and isinstance(f._expr, (Function, Expression))
            )
            for f in fields
            if not isinstance(f, str)
        )

        if has_sql_expr:
            # Ensure we have a SQL source (create PythonTableFunction if needed)
            self._ensure_sql_source()

        # Track operation
        field_names = ", ".join([str(f) for f in fields]) if fields else "*"

        # Build SQL SELECT (lazy)
        self._track_operation("sql", f"SELECT {field_names}", {"lazy": True})

        # Record in lazy ops for correct execution order in explain()
        # Store fields for DataFrame execution
        self._add_lazy_op(LazyRelationalOp("SELECT", field_names, fields=list(fields)))

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
        self,
        condition: Union[Condition, str, "ColumnExpr", None] = None,
        items=None,
        like=None,
        regex=None,
        axis=None,
    ) -> "DataStore":
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
            # Check if this is a method-mode ColumnExpr (e.g., cumsum() > 6, rank() > 3)
            # These cannot be converted to SQL and need pandas-based filtering
            if condition._exec_mode == "method" or condition._expr is None:
                # For method-mode ColumnExpr, we need to execute it and use pandas filtering
                # Add as a pandas filter operation
                self._track_operation(
                    "pandas", f"filter(method_mode_condition)", {"lazy": True}
                )
                self._add_lazy_op(
                    LazyRelationalOp(
                        "PANDAS_FILTER", "method_mode", condition=condition
                    )
                )
                return self

            if isinstance(condition._expr, Condition):
                # ColumnExpr wrapping Condition (e.g., ds['col'] > 5)
                condition = condition._expr
            else:
                # Check if this is a computed column that stores a Condition
                # e.g., ds.assign(is_high=ds['x'] > 100) then ds[ds['is_high']]
                if isinstance(condition._expr, Field):
                    col_name = condition._expr.name.strip("\"'")
                    if (
                        hasattr(self, "_computed_columns")
                        and col_name in self._computed_columns
                    ):
                        computed_expr = self._computed_columns[col_name]
                        if isinstance(computed_expr, Condition):
                            # Resolve any dependencies in the condition and use it directly
                            condition = self._resolve_expr_dependencies(computed_expr)
                        else:
                            # Non-condition computed column, convert to truthy check
                            resolved = self._resolve_expr_dependencies(computed_expr)
                            condition = BinaryCondition("=", resolved, Literal(1))
                    else:
                        # Regular column, convert to truthy check: expr = 1
                        condition = BinaryCondition("=", condition._expr, Literal(1))
                else:
                    # Non-condition ColumnExpr (e.g., boolean function result)
                    # Convert to truthy check: expr = 1
                    condition = BinaryCondition("=", condition._expr, Literal(1))

        # Convert condition to string for tracking
        if isinstance(condition, str):
            condition_str = condition
        else:
            condition_str = condition.to_sql(quote_char=self.quote_char)

        # Build SQL WHERE clause (lazy)
        self._track_operation("sql", f"WHERE {condition_str}", {"lazy": True})

        # Record in lazy ops for correct execution order in explain()
        # Store condition object for DataFrame execution
        self._add_lazy_op(LazyRelationalOp("WHERE", condition_str, condition=condition))

        if isinstance(condition, str):
            # TODO: Parse string conditions
            raise UnsupportedOperationError(
                operation="string condition",
                reason="string-based filter conditions are not yet implemented in SQL mode",
                suggestion="Use boolean expressions: ds[ds['col'] > 5] instead of ds['col > 5']",
            )

        if self._where_condition is None:
            self._where_condition = condition
        else:
            # Combine with existing condition using AND
            self._where_condition = self._where_condition & condition

        return self

    def where(
        self, condition: Union[Condition, str], other=_MISSING, **kwargs
    ) -> "DataStore":
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
        # - DataStore condition (boolean DataFrame, e.g., ds_df > 2)
        # - Explicit other value provided
        # - Any kwargs provided
        # Note: DataStore boolean condition must use pandas-style where (CASE WHEN semantics),
        # not SQL WHERE (row filtering)
        is_pandas_style = (
            isinstance(condition, ColumnExpr)
            or isinstance(condition, DataStore)
            or other is not _MISSING
            or kwargs
        )

        if is_pandas_style:
            # Convert _MISSING to None for pandas compatibility
            actual_other = None if other is _MISSING else other
            # Pandas-style where() - delegate to mixin
            if hasattr(super(), "where"):
                return super().where(condition, other=actual_other, **kwargs)
            # Fallback if no mixin
            raise UnsupportedOperationError(
                operation="where(condition)",
                reason="pandas-style where() with positional arguments requires pandas compatibility layer",
                suggestion="Use ds.where(cond, other) or ds[cond] for filtering",
            )

        # SQL-style filter (simple Condition or string, no other args)
        return self.filter(condition)

    def when(self, condition: Any, value: Any) -> "CaseWhenBuilder":
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
    def run_sql(cls, query: str) -> "DataStore":
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

    def sql(self, query: str) -> "DataStore":
        """
        Execute a SQL query - works for both local DataFrames and remote databases.

        **For local DataFrames (table mode):**
        Executes SQL on the current DataFrame using chDB's SQL engine.
        Supports short form (auto-adds SELECT * FROM __df__) and full SQL form.

        **For remote connections (connection/database mode):**
        Executes SQL against the remote database. Table names in FROM/JOIN
        clauses are automatically resolved to table functions.

        Args:
            query: SQL query or condition. Examples:
                   Local: "value > 100", "SELECT * FROM __df__ WHERE age > 20"
                   Remote: "SELECT * FROM users WHERE age > 25"

        Returns:
            A new DataStore with the SQL query result.

        Example (local):
            >>> ds = DataStore.from_file('users.csv')
            >>> ds = ds.sql("age > 20 ORDER BY name")

        Example (remote):
            >>> ds = DataStore.from_clickhouse(host="localhost:9000", ...)
            >>> ds.use("mydb")
            >>> result = ds.sql("SELECT * FROM users WHERE age > 25")
        """
        # Check if this is a remote connection without bound data
        if self._connection_mode in ("connection", "database") and self._remote_params:
            return self._remote_sql(query)

        # Local DataFrame SQL execution
        from copy import copy

        from .lazy_ops import LazySQLQuery

        # Create a copy of this DataStore (immutable pattern)
        new_ds = copy(self)

        # Record the SQL query operation on the copy
        lazy_op = LazySQLQuery(query, df_alias="__df__")
        new_ds._add_lazy_op(lazy_op)

        return new_ds

    @immutable
    def join(
        self,
        other: "DataStore",
        on=None,
        how: str = "inner",
        left_on: str = None,
        right_on: str = None,
    ) -> "DataStore":
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
            "inner": JoinType.inner,
            "left": JoinType.left,
            "right": JoinType.right,
            "outer": JoinType.outer,
            "full": JoinType.full_outer,
            "cross": JoinType.cross,
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
                join_condition = ("USING", [on])
                pandas_on = on
            elif isinstance(on, (list, tuple)) and all(isinstance(c, str) for c in on):
                # List of strings -> USING (col1, col2, ...) syntax
                join_condition = ("USING", list(on))
                pandas_on = list(on)
            else:
                # Condition object -> ON clause (SQL only, need to extract columns for pandas)
                join_condition = on
                # For Condition objects, we can't easily extract columns, so use SQL path
        elif left_on and right_on:
            # Create condition from column names
            # Use table alias for table functions
            left_table = (
                self._get_table_alias() if self._table_function else self.table_name
            )
            right_table = (
                other._get_table_alias() if other._table_function else other.table_name
            )

            left_field = Field(left_on, table=left_table)
            right_field = Field(right_on, table=right_table)
            join_condition = left_field == right_field
        else:
            raise QueryError(
                "Either 'on' or both 'left_on' and 'right_on' must be specified"
            )

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
    def union(self, other: "DataStore", all: bool = False) -> "DataStore":
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
    def with_column(self, name: str, expr: Union[Expression, Any]) -> "DataStore":
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

    def groupby(
        self,
        *fields: Union[str, Expression, List],
        sort: bool = True,
        as_index: bool = True,
        dropna: bool = True,
        **kwargs,
    ) -> "LazyGroupBy":
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
            as_index: If True (default), group keys become the index of the result.
                      If False, group keys are returned as columns in the result.
            dropna: If True (default), exclude NA/null values in keys.
                    If False, NA values are also grouped.
            **kwargs: Additional arguments (for pandas compatibility, currently ignored).

        Returns:
            LazyGroupBy: GroupBy wrapper referencing this DataStore

        Example:
            >>> ds.groupby("category")  # Returns LazyGroupBy, sorted by category
            >>> ds.groupby("category", sort=False)  # Unsorted (order not guaranteed)
            >>> ds.groupby("category", as_index=False)  # Group key as column
            >>> ds.groupby(["a", "b"])  # pandas-style list argument
            >>> ds.groupby("a", "b")    # Also supported
            >>> ds.groupby("category")["sales"].mean()  # Executes ds, returns Series
            >>> ds.to_df()  # Uses cached result (no re-computation!)
        """
        from .groupby import LazyGroupBy
        from .column_expr import ColumnExpr
        from .lazy_result import LazySeries
        from .lazy_ops import LazyColumnAssignment
        from copy import copy
        import uuid

        # Track the datastore to use - may need to create a copy if we have derived columns
        target_ds = self
        groupby_fields = []
        temp_column_counter = 0

        def process_field(f) -> Field:
            """Process a single field, handling ColumnExpr/LazySeries specially."""
            nonlocal target_ds, temp_column_counter

            if isinstance(f, str):
                return Field(f)
            elif isinstance(f, ColumnExpr):
                # ColumnExpr like ds['date'].dt.year
                # Check if it's a simple Field reference - use directly
                if isinstance(f._expr, Field):
                    return f._expr
                # Complex expression - assign to temp column
                temp_name = (
                    f"__groupby_temp_{temp_column_counter}_{uuid.uuid4().hex[:8]}"
                )
                temp_column_counter += 1
                # Create a copy if we haven't already
                if target_ds is self:
                    target_ds = copy(self)
                target_ds._add_lazy_op(LazyColumnAssignment(temp_name, f._expr))
                return Field(temp_name)
            elif isinstance(f, LazySeries):
                # LazySeries - similar handling
                temp_name = (
                    f"__groupby_temp_{temp_column_counter}_{uuid.uuid4().hex[:8]}"
                )
                temp_column_counter += 1
                # Create a copy if we haven't already
                if target_ds is self:
                    target_ds = copy(self)
                target_ds._add_lazy_op(LazyColumnAssignment(temp_name, f))
                return Field(temp_name)
            else:
                return f

        for field in fields:
            # Handle list argument (pandas-style): groupby(["a", "b"])
            if isinstance(field, (list, tuple)):
                for f in field:
                    groupby_fields.append(process_field(f))
            else:
                groupby_fields.append(process_field(field))

        # Extract selected columns from prior LazyRelationalOp(SELECT) operations
        # This ensures that df[['col1', 'col2']].groupby('col1').mean() only aggregates
        # the explicitly selected columns, not all columns from the source
        selected_columns = None
        for op in reversed(target_ds._lazy_ops):
            if (
                isinstance(op, LazyRelationalOp)
                and op.op_type == "SELECT"
                and op.fields
            ):
                # Found a SELECT operation - extract column names
                selected_columns = []
                for f in op.fields:
                    if isinstance(f, str) and f != "*":
                        selected_columns.append(f)
                    elif isinstance(f, Field):
                        selected_columns.append(f.name)
                # If SELECT * or no explicit columns, don't override
                if not selected_columns:
                    selected_columns = None
                break

        # Return a GroupBy wrapper that references target_ds
        return LazyGroupBy(
            target_ds,
            groupby_fields,
            sort=sort,
            as_index=as_index,
            dropna=dropna,
            selected_columns=selected_columns,
        )

    @immutable
    def agg(self, func=None, axis=0, *args, **kwargs) -> "DataStore":
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
        has_sql_agg = any(
            isinstance(v, (Expression, ColumnExpr, AggregateFunction))
            for v in kwargs.values()
        )

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
                if hasattr(f, "alias") and f.alias:
                    select_parts.append(f'{f.to_sql()} AS "{f.alias}"')
                else:
                    select_parts.append(f.to_sql())

            select_clause = ", ".join(select_parts)

            # Build GROUP BY clause
            groupby_parts = [gf.to_sql() for gf in self._groupby_fields]
            groupby_clause = ", ".join(groupby_parts) if groupby_parts else ""

            # Build full SQL
            if groupby_clause:
                full_sql = (
                    f"SELECT {select_clause} FROM __df__ GROUP BY {groupby_clause}"
                )
            else:
                full_sql = f"SELECT {select_clause} FROM __df__"

            # Use LazySQLQuery for proper execution
            self._add_lazy_op(LazySQLQuery(full_sql))

            return self
        else:
            # Pandas-style aggregation: agg({'col': 'func'}) or agg('sum')
            # Delegate to parent class (PandasCompatMixin)
            return super().agg(func, axis, *args, **kwargs)

    @immutable
    def assign(self, **kwargs) -> "DataStore":
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

        4. Mixed mode (SQL expressions + lambda/scalar values):
           >>> ds.assign(D=ds['A'] + ds['B'], E=lambda x: x['C'] * 2)

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

        # Separate SQL expressions from pandas expressions (lambda, scalar, etc.)
        sql_kwargs = {}
        pandas_kwargs = {}

        for alias, expr in kwargs.items():
            if isinstance(expr, ColumnExpr):
                # Use canonical classification methods for clear separation
                if expr.is_pandas_only():
                    # Executor mode (transform), method mode without expr - must use Pandas
                    pandas_kwargs[alias] = expr._execute()
                elif expr.is_sql_compatible():
                    # Has valid expression tree - can convert to SQL
                    sql_kwargs[alias] = expr
                else:
                    # Fallback: execute and use result
                    pandas_kwargs[alias] = expr._execute()
            elif isinstance(expr, (Function, Expression)):
                sql_kwargs[alias] = expr
            else:
                # Lambda functions, scalars, Series, etc. - handle via pandas
                pandas_kwargs[alias] = expr

        # Start with self
        result = self

        # Ensure _computed_columns is a fresh copy (not shared due to shallow copy from @immutable)
        if not hasattr(result, "_computed_columns") or result._computed_columns is None:
            result._computed_columns = {}
        else:
            result._computed_columns = result._computed_columns.copy()

        # Process SQL expressions first (if any)
        if sql_kwargs:
            # Ensure we have a SQL source (create PythonTableFunction if needed)
            result._ensure_sql_source()

            # Check if there's an existing column selection in lazy ops
            # If the last SELECT op has specific columns (not just '*'), preserve them
            existing_select_fields = None
            for op in reversed(result._lazy_ops):
                if hasattr(op, "op_type") and op.op_type == "SELECT":
                    if hasattr(op, "fields") and op.fields:
                        # Check if this SELECT has specific columns vs '*'
                        # A SELECT with just '*' has no fields, or has '*' string in fields
                        has_star = any(
                            f == "*" or (isinstance(f, str) and f.strip() == "*")
                            for f in op.fields
                        )
                        if not has_star:
                            # We have specific columns selected, preserve them
                            existing_select_fields = op.fields
                    break

            # Use existing column selection if available, otherwise use '*'
            if existing_select_fields:
                # Preserve the existing column selection and add new columns
                select_items = list(existing_select_fields)
            else:
                select_items = ["*"]

            # Get list of aliases being assigned
            aliases_to_assign = set(sql_kwargs.keys())

            # Get existing columns to check for overwrites
            existing_columns = list(result._get_all_column_names())
            existing_columns_set = set(existing_columns)

            # Check if any alias overwrites an existing column
            columns_to_overwrite = aliases_to_assign & existing_columns_set

            # New columns that don't exist yet
            new_columns = aliases_to_assign - existing_columns_set

            # Build expression map for all assigned columns first
            expr_map = {}
            for alias, expr in sql_kwargs.items():
                original_expr = expr
                if isinstance(expr, ColumnExpr):
                    expr = expr._expr
                if isinstance(expr, Expression):
                    # Resolve any computed column dependencies in the expression
                    # This enables chained assigns where later assigns reference earlier ones
                    # e.g., ds.assign(a=...).assign(b=ds['a'] * 2) - 'a' needs to be expanded
                    resolved_expr = result._resolve_expr_dependencies(expr)
                    # Record the resolved expression (alias -> expression without alias)
                    result._computed_columns[alias] = resolved_expr
                    # Set alias on the expression
                    expr_map[alias] = resolved_expr.as_(alias)

            if columns_to_overwrite:
                # When overwriting, expand '*' and replace columns in-place to preserve order
                if select_items == ["*"]:
                    # Build select items preserving order, replacing overwrites in-place
                    new_select_items = []
                    for col in existing_columns:
                        if col in columns_to_overwrite:
                            # Replace with new expression in-place
                            new_select_items.append(expr_map[col])
                        else:
                            new_select_items.append(col)
                    select_items = new_select_items
                else:
                    # Replace overwritten columns in existing selection
                    new_select_items = []
                    for item in select_items:
                        col_name = (
                            item
                            if isinstance(item, str)
                            else getattr(item, "_alias", None)
                        )
                        if col_name in columns_to_overwrite:
                            new_select_items.append(expr_map[col_name])
                        else:
                            new_select_items.append(item)
                    select_items = new_select_items

                # Add new (non-overwrite) columns at the end
                for alias in sql_kwargs.keys():
                    if alias in new_columns:
                        select_items.append(expr_map[alias])
            else:
                # No overwrites - keep '*' and just append new columns
                for alias, expr_with_alias in expr_map.items():
                    select_items.append(expr_with_alias)

            result = result.select(*select_items)

        # Process pandas expressions (if any)
        if pandas_kwargs:
            # Use pandas-style assignment for lambda/scalar values
            # Need to execute SQL first to get DataFrame, then apply pandas assign
            df = result._execute()
            # Apply pandas assign
            new_df = df.assign(**pandas_kwargs)
            # Wrap back into DataStore
            result = result._wrap_result(new_df)

        return result

    def _is_aggregate_expr(self, expr) -> bool:
        """Check if an expression is an aggregate expression."""
        from .column_expr import ColumnExpr
        from .functions import AggregateFunction

        if isinstance(expr, ColumnExpr):
            expr = expr._expr

        if isinstance(expr, AggregateFunction):
            return True

        # Check if expression has aggregate function in its tree
        if hasattr(expr, "_func_name"):
            # Check registry for aggregate function
            from .function_registry import get_function_registry

            registry = get_function_registry()
            func_def = registry.get(expr._func_name)
            if func_def and func_def.func_type.name == "AGGREGATE":
                return True

        return False

    @immutable
    def sort(
        self,
        *fields: Union[str, Expression],
        ascending: Union[bool, List[bool]] = True,
        kind: str = "quicksort",
    ) -> "DataStore":
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
                return f._expr.to_sql() if hasattr(f._expr, "to_sql") else str(f._expr)
            if hasattr(f, "to_sql"):
                return f.to_sql()
            return str(f)

        # Normalize ascending to a list
        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(fields)
        else:
            ascending_list = list(ascending)
            if len(ascending_list) != len(fields):
                raise ValueError(
                    f"Length of ascending ({len(ascending_list)}) != length of fields ({len(fields)})"
                )

        # Build description for explain()
        parts = []
        for f, asc in zip(fields, ascending_list):
            direction = "ASC" if asc else "DESC"
            parts.append(f"{field_to_sql(f)} {direction}")
        description = ", ".join(parts)

        # Record in lazy ops for correct execution order in explain()
        # Store fields, ascending (as list), and kind for DataFrame execution
        self._add_lazy_op(
            LazyRelationalOp(
                "ORDER BY",
                description,
                fields=list(fields),
                ascending=ascending_list,
                kind=kind,
            )
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
    def orderby(
        self,
        *fields: Union[str, Expression],
        ascending: bool = True,
        kind: str = "quicksort",
    ) -> "DataStore":
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
    def limit(self, n: int) -> "DataStore":
        """Limit number of results."""
        # Record in lazy ops for correct execution order in explain()
        # Store limit_value for DataFrame execution
        self._add_lazy_op(LazyRelationalOp("LIMIT", str(n), limit_value=n))
        self._limit_value = n
        return self

    @immutable
    def offset(self, n: int) -> "DataStore":
        """Skip first n results."""
        # Record in lazy ops for correct execution order in explain()
        # Store offset_value for DataFrame execution
        self._add_lazy_op(LazyRelationalOp("OFFSET", str(n), offset_value=n))
        self._offset_value = n

    @immutable
    def distinct(self, subset=None, keep="first") -> "DataStore":
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
    def having(self, condition: Union[Condition, str]) -> "DataStore":
        """
        Add HAVING clause for filtering aggregated results.

        Args:
            condition: Condition object or SQL string

        Example:
            >>> ds.groupby("city").having(Count("*") > 10)
        """
        if isinstance(condition, str):
            raise UnsupportedOperationError(
                operation="string condition",
                reason="string-based filter conditions are not yet implemented in SQL mode",
                suggestion="Use boolean expressions: ds[ds['col'] > 5] instead of ds['col > 5']",
            )

        if self._having_condition is None:
            self._having_condition = condition
        else:
            # Combine with existing condition using AND
            self._having_condition = self._having_condition & condition

    @immutable
    def as_(self, alias: str) -> "DataStore":
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

    def use(self, *args) -> "DataStore":
        """
        Set default schema, database, and/or table context.

        This sets defaults for subsequent operations like query() and ds["table"].
        Mutates the current DataStore and returns self for chaining.

        Args:
            With 1 arg: use(database) - set default database
            With 2 args: use(database, table) - set database and table
            With 3 args: use(schema, database, table) - set all three

        Returns:
            self for method chaining

        Examples:
            >>> ds = DataStore.from_clickhouse(host="localhost:9000", user="default", password="")
            >>> ds.use("production")
            >>> ds.query("SELECT * FROM users")  # uses production.users

            >>> ds.use("production", "users")  # set both
            >>> ds.columns  # now works

        Raises:
            ValueError: If wrong number of arguments provided
        """
        if len(args) == 0:
            raise ValueError("use() requires at least 1 argument")
        elif len(args) == 1:
            self._default_database = args[0]
        elif len(args) == 2:
            self._default_database = args[0]
            self._default_table = args[1]
            # Update connection mode if we now have table
            if self._connection_mode == "connection":
                self._connection_mode = "database"
            if args[1]:
                self._connection_mode = "table"
        elif len(args) == 3:
            self._default_schema = args[0]
            self._default_database = args[1]
            self._default_table = args[2]
            if args[2]:
                self._connection_mode = "table"
        else:
            raise ValueError(
                "use() takes 1, 2, or 3 arguments (database), (database, table), or (schema, database, table)"
            )

        return self

    def databases(self) -> List[str]:
        """
        List all databases on the remote server.

        Returns:
            List of database names

        Raises:
            DataStoreError: If connection info is missing or source doesn't support metadata

        Example:
            >>> ds = DataStore.from_clickhouse(host="localhost:9000", user="default", password="")
            >>> ds.databases()
            ['default', 'system', 'test_db']
        """
        adapter = self._get_adapter()
        sql = adapter.list_databases_sql()
        self._logger.debug(f"Listing databases: {sql}")
        result = self._execute_metadata_query(sql)
        return result["name"].tolist()

    def tables(self, database: str = None) -> List[str]:
        """
        List all tables in the specified database.

        Args:
            database: Database name. Uses default database if not specified.

        Returns:
            List of table names

        Raises:
            DataStoreError: If database is not specified and no default is set

        Example:
            >>> ds = DataStore.from_clickhouse(host="localhost:9000", user="default", password="")
            >>> ds.tables("system")
            ['tables', 'databases', 'columns', ...]

            >>> ds.use("test_db")
            >>> ds.tables()  # uses default database
            ['users', 'orders']
        """
        adapter = self._get_adapter()

        # Use provided database or default
        database = database or self._default_database
        if database is None:
            from .exceptions import DataStoreError

            raise DataStoreError(
                "No database specified.\n"
                'Hint: Provide database argument: ds.tables("mydb"), '
                'or set default: ds.use("mydb")'
            )

        sql = adapter.list_tables_sql(database)
        self._logger.debug(f"Listing tables in {database}: {sql}")
        result = self._execute_metadata_query(sql)
        return result["name"].tolist()

    def _remote_describe(self, database: str = None, table: str = None) -> pd.DataFrame:
        """
        Get a remote table's schema (internal method).
        """
        adapter = self._get_adapter()

        database = database or self._default_database
        table = table or self._default_table

        if database is None:
            from .exceptions import DataStoreError

            raise DataStoreError(
                "No database specified.\n"
                'Hint: ds.describe("mydb", "mytable") or set default with ds.use("mydb")'
            )
        if table is None:
            from .exceptions import DataStoreError

            raise DataStoreError(
                "No table specified.\n"
                'Hint: ds.describe("mydb", "mytable") or set default with ds.use("mydb", "mytable")'
            )

        sql = adapter.describe_table_sql(database, table)
        self._logger.debug(f"Getting schema for {database}.{table}: {sql}")
        return self._execute_metadata_query(sql)

    def _remote_sql(self, query: str) -> "DataStore":
        """
        Execute SQL query against the remote database (internal method).

        Table names in FROM/JOIN clauses are automatically resolved to table functions.
        """
        self._require_connection_params()

        # Rewrite table references to table functions
        rewritten_sql = self._rewrite_table_references(query)
        self._logger.debug(f"Original SQL: {query}")
        self._logger.debug(f"Rewritten SQL: {rewritten_sql}")

        # Execute and wrap in DataStore
        import chdb

        result_df = chdb.query(rewritten_sql, output_format="DataFrame")

        # Return a new DataStore wrapping the result
        return DataStore(result_df)

    def _rewrite_table_references(self, sql: str) -> str:
        """
        Rewrite table references in SQL to use table functions.

        Uses regex to find FROM/JOIN table references and replaces them
        with appropriate table function calls.

        Args:
            sql: Original SQL query

        Returns:
            SQL with table references replaced by table functions
        """
        import re

        adapter = self._get_adapter()

        # Pattern to match table references after FROM/JOIN
        # Matches: FROM table, FROM db.table, FROM schema.db.table
        # Excludes: FROM func(...), FROM (subquery), FROM "quoted"
        pattern = r'\b(FROM|JOIN)\s+(?![\w]+\s*\()(?!\()(?!")([a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*){0,2})(\s+(?:AS\s+)?([a-zA-Z_][\w]*))?'

        def replace_table_ref(match):
            keyword = match.group(1)  # FROM or JOIN
            table_ref = match.group(2)  # table reference
            alias_part = match.group(3) or ""  # optional alias (e.g., " AS u" or " u")

            parts = table_ref.split(".")

            if len(parts) == 1:
                # Just table name - use default database
                table = parts[0]
                database = self._default_database
                if database is None:
                    from .exceptions import DataStoreError

                    raise DataStoreError(
                        f"No database specified for table '{table}' in query.\n"
                        f'Hint: Use fully qualified name (db.table) or call ds.use("mydb") first.'
                    )
            elif len(parts) == 2:
                database, table = parts
            elif len(parts) == 3:
                # schema.db.table - combine schema.table for some DBs
                schema, database, table = parts
                table = f"{schema}.{table}"
            else:
                # Don't rewrite if we can't parse
                return match.group(0)

            table_func = adapter.build_table_function(database, table)
            return f"{keyword} {table_func}{alias_part}"

        return re.sub(pattern, replace_table_ref, sql, flags=re.IGNORECASE)

    def __getitem__(
        self, key: Union[int, slice, str, List[str]]
    ) -> Union["DataStore", "ColumnExpr", Any]:
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

        For remote connections in connection/database mode:
            >>> ds['mydb.users']  # Returns new DataStore bound to mydb.users table
            >>> ds['users']       # Returns new DataStore bound to users table (uses default db)
        """
        from .column_expr import ColumnExpr
        from .conditions import Condition
        from copy import copy

        # Handle table selection in connection/database mode
        if isinstance(key, str) and self._connection_mode in ("connection", "database"):
            # Check if this looks like a table reference (contains dot or we're in connection mode)
            if "." in key or self._connection_mode == "connection":
                return self._select_table(key)

        if isinstance(key, str):
            # Check if column is accessible (respects select() restrictions)
            # This matches pandas behavior: after select(['col1', 'col2']),
            # only col1 and col2 can be accessed.
            if not self._is_column_accessible(key):
                accessible = self._get_accessible_columns()
                raise KeyError(
                    f"Column '{key}' is not accessible. "
                    f"After select(), only these columns are available: {sorted(accessible) if accessible else '(none)'}. "
                    f"Either include '{key}' in your select() or access it before calling select()."
                )

            # Return ColumnExpr that wraps a Field and can execute
            # This allows pandas-like behavior: ds['col'] shows actual values
            # but ds['col'] > 18 still returns Condition for filtering
            #
            # NOTE: We always return Field(key) here, NOT the computed expression.
            # The computed expression is only resolved when:
            # 1. Building new expressions that reference this column (in _resolve_expr_dependencies)
            # 2. When generating SQL (in extract_clauses_from_ops)
            #
            # If we returned the computed expression here, it would break cases like:
            #   result = ds.assign(name_upper=ds['name'].str.upper())
            #   result['name_upper'].tolist()  # Would try to evaluate upper("name") on result
            #   # But result's DataFrame only has 'name_upper', not 'name'!
            return ColumnExpr(Field(key), self)

        elif isinstance(key, tuple):
            # Tuple indexing: support for MultiIndex columns
            # In pandas, df[('level1', 'level2')] accesses a column with MultiIndex name
            # For DataStore, we delegate to pandas for MultiIndex column access
            df = self._execute()
            if key in df.columns:
                # Return the column as a Series wrapped in ColumnExpr-like behavior
                return df[key]
            else:
                raise KeyError(f"Column {key} not found in DataFrame")

        elif isinstance(key, (list, pd.Index)):
            # Multi-column selection or boolean mask indexing
            # Convert pandas Index to list if needed
            if isinstance(key, pd.Index):
                key = key.tolist()

            # Check if this is a boolean list (for row filtering, not column selection)
            if key and all(isinstance(x, (bool, np.bool_)) for x in key):
                # Boolean mask indexing: df[[True, False, True, ...]]
                from .lazy_ops import LazyBooleanMask

                result = copy(self) if getattr(self, "is_immutable", True) else self
                if result is not self:
                    result._cached_result = None
                    result._cache_version = 0
                    result._cached_at_version = -1
                    result._cache_timestamp = None
                result._add_lazy_op(LazyBooleanMask(key))
                return result

            # Check for empty list - return DataFrame with no columns
            if len(key) == 0:
                result = copy(self) if getattr(self, "is_immutable", True) else self
                if result is not self:
                    result._cached_result = None
                    result._cache_version = 0
                    result._cached_at_version = -1
                    result._cache_timestamp = None
                result._select_star = False
                result._add_lazy_op(
                    LazyRelationalOp(
                        op_type="SELECT", description="Select no columns", fields=[]
                    )
                )
                return result

            # Multi-column selection: use LazyRelationalOp(SELECT) for SQL pushdown
            # Create a copy to avoid modifying the original DataStore's _lazy_ops
            # This fixes the bug where df[['col1', 'col2']].head() would modify df
            result = copy(self) if getattr(self, "is_immutable", True) else self
            if result is not self:
                # Reset cache state for the new copy
                result._cached_result = None
                result._cache_version = 0
                result._cached_at_version = -1
                result._cache_timestamp = None
            # When explicitly selecting columns, reset _select_star since we no longer want all columns
            result._select_star = False
            # Use LazyRelationalOp(SELECT) so column selection can be pushed to SQL
            # This allows ds[['col1', 'col2']].sort_values('col1').head(10) to be
            # fully executed as SQL: SELECT col1, col2 FROM ... ORDER BY col1 LIMIT 10
            # Convert column names to strings for Field creation (handles int column names)
            fields = [Field(str(col) if isinstance(col, int) else col) for col in key]
            col_names = [str(col) if isinstance(col, int) else col for col in key]
            result._add_lazy_op(
                LazyRelationalOp(
                    op_type="SELECT",
                    description=f"Select columns: {', '.join(col_names)}",
                    fields=fields,
                )
            )
            return result

        elif isinstance(key, slice):
            # LIMIT/OFFSET - this is a SQL operation
            # Create a copy to avoid modifying the original DataStore (pandas-like behavior)
            start, stop, step = key.start, key.stop, key.step

            result = copy(self) if getattr(self, "is_immutable", True) else self
            if result is not self:
                # Reset cache state for the new copy
                result._cached_result = None
                result._cache_version = 0
                result._cached_at_version = -1
                result._cache_timestamp = None

            # Handle step slicing (e.g., ds[::2], ds[1::2], ds[::-1])
            if step is not None:
                from .lazy_ops import LazySliceStep

                result._add_lazy_op(LazySliceStep(start=start, stop=stop, step=step))
                return result

            # Handle negative indices - need to use pandas fallback as we don't know row count
            if (start is not None and start < 0) or (stop is not None and stop < 0):
                from .lazy_ops import LazySliceStep

                result._add_lazy_op(LazySliceStep(start=start, stop=stop, step=None))
                return result

            if stop is not None:
                if start is not None:
                    # ds[start:stop] -> LIMIT (stop-start) OFFSET start
                    limit_val = stop - start if stop > start else stop
                    result._offset_value = start
                    result._limit_value = limit_val
                    result._add_lazy_op(
                        LazyRelationalOp(
                            "OFFSET", f"OFFSET {start}", offset_value=start
                        )
                    )
                    result._add_lazy_op(
                        LazyRelationalOp(
                            "LIMIT", f"LIMIT {limit_val}", limit_value=limit_val
                        )
                    )
                else:
                    # ds[:stop] -> LIMIT stop
                    result._limit_value = stop
                    result._add_lazy_op(
                        LazyRelationalOp("LIMIT", f"LIMIT {stop}", limit_value=stop)
                    )
            elif start is not None:
                # ds[start:] -> OFFSET start
                result._offset_value = start
                result._add_lazy_op(
                    LazyRelationalOp("OFFSET", f"OFFSET {start}", offset_value=start)
                )
            return result

        elif isinstance(key, (Condition, ColumnExpr)):
            # Boolean indexing: filter rows like pandas df[condition]
            # Create a copy to avoid modifying the original DataStore
            result = copy(self) if getattr(self, "is_immutable", True) else self
            if result is not self:
                # Reset cache state for the new copy
                result._cached_result = None
                result._cache_version = 0
                result._cached_at_version = -1
                result._cache_timestamp = None
            return result.filter(key)

        elif isinstance(key, pd.Series):
            # Boolean Series indexing: df[pd.Series([True, False, True])]
            if key.dtype == bool or key.dtype == "boolean":
                from .lazy_ops import LazyBooleanMask

                result = copy(self) if getattr(self, "is_immutable", True) else self
                if result is not self:
                    result._cached_result = None
                    result._cache_version = 0
                    result._cached_at_version = -1
                    result._cache_timestamp = None
                result._add_lazy_op(LazyBooleanMask(key.tolist()))
                return result
            else:
                raise TypeError(
                    f"Boolean Series indexing requires boolean dtype, got {key.dtype}"
                )

        elif isinstance(key, np.ndarray):
            # Boolean ndarray indexing: df[np.array([True, False, True])]
            if key.dtype == bool or key.dtype == np.bool_:
                from .lazy_ops import LazyBooleanMask

                result = copy(self) if getattr(self, "is_immutable", True) else self
                if result is not self:
                    result._cached_result = None
                    result._cache_version = 0
                    result._cached_at_version = -1
                    result._cache_timestamp = None
                result._add_lazy_op(LazyBooleanMask(key.tolist()))
                return result
            else:
                raise TypeError(
                    f"Boolean ndarray indexing requires boolean dtype, got {key.dtype}"
                )

        elif isinstance(key, int):
            # Integer column name access: df[0] when column name is 0
            return ColumnExpr(Field(str(key)), self)

        elif callable(key):
            # Callable indexing: df[lambda x: x['a'] > 2]
            # Call the callable with self, then apply the result as indexing
            condition = key(self)
            # Recursively handle the result (could be Condition, ColumnExpr, list, etc.)
            return self[condition]

        else:
            # Check for LazyCondition (from isin/between) - needs late import
            from .lazy_result import LazyCondition

            if isinstance(key, LazyCondition):
                # Boolean indexing with LazyCondition
                result = copy(self) if getattr(self, "is_immutable", True) else self
                if result is not self:
                    result._cached_result = None
                    result._cache_version = 0
                    result._cached_at_version = -1
                    result._cache_timestamp = None
                return result.filter(key)

            raise TypeError(
                f"DataStore indices must be slices, strings, lists, or conditions, not {type(key).__name__}"
            )

    def _select_table(self, key: str) -> "DataStore":
        """
        Select a database/table and return a NEW DataStore (immutable operation).

        Args:
            key: Table reference in format "table", "db.table", or "schema.db.table"

        Returns:
            New DataStore bound to the specified table

        Raises:
            DataStoreError: If database cannot be determined
        """
        from .exceptions import DataStoreError

        parts = key.split(".")

        schema = None
        database = None
        table = None

        if len(parts) == 1:
            # Just table name - need default database
            table = parts[0]
            database = self._default_database
            schema = self._default_schema
            if database is None:
                raise DataStoreError(
                    f"No database specified for table '{table}'.\n"
                    f'Hint: Use ds["{table}"] after ds.use("mydb"), or use ds["mydb.{table}"] format.'
                )
        elif len(parts) == 2:
            database, table = parts
            schema = self._default_schema
        elif len(parts) == 3:
            schema, database, table = parts
        else:
            raise DataStoreError(
                f"Invalid table reference: '{key}'. Expected 'table', 'db.table', or 'schema.db.table'."
            )

        # Create new DataStore with table bound
        # Copy remote params and set database/table
        new_ds = DataStore(
            source=self.source_type,
            table=table,
            database=database,
            host=self._remote_params.get("host"),
            user=self._remote_params.get("user"),
            password=self._remote_params.get("password", ""),
            secure=self._remote_params.get("secure", False),
        )

        # Copy schema if present
        if schema:
            new_ds._default_schema = schema

        return new_ds

    def _require_connection_params(self):
        """
        Raise error if connection parameters are missing.

        Raises:
            DataStoreError: If host is not specified
        """
        if not self._remote_params.get("host"):
            from .exceptions import DataStoreError

            raise DataStoreError(
                "Cannot perform metadata operation - no connection info.\n"
                'Hint: DataStore(source="clickhouse", host="...", user="...", password="...")'
            )

    def _get_adapter(self) -> "SourceAdapter":
        """
        Get the appropriate adapter for the current source type.

        Returns:
            SourceAdapter instance for metadata operations

        Raises:
            DataStoreError: If source type doesn't support metadata operations
        """
        self._require_connection_params()
        return get_adapter(
            self.source_type,
            host=self._remote_params.get("host"),
            user=self._remote_params.get("user"),
            password=self._remote_params.get("password", ""),
            secure=self._remote_params.get("secure", False),
        )

    def _execute_metadata_query(self, sql: str) -> pd.DataFrame:
        """
        Execute a metadata query and return DataFrame.

        Uses chdb to execute the query with table functions.

        Args:
            sql: SQL query string (typically using table functions)

        Returns:
            pandas DataFrame with query results
        """
        import chdb

        result = chdb.query(sql, output_format="DataFrame")
        return result

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

    def to_sql(
        self,
        quote_char: str = None,
        as_subquery: bool = False,
        execution_format: bool = False,
    ) -> str:
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
        elif (
            execution_format
            and self._lazy_ops
            and (self._table_function or self.table_name)
        ):
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
            fields_sql = ", ".join(
                field.to_sql(quote_char=quote_char, with_alias=True)
                for field in self._select_fields
            )
            # If _select_star is True, prepend '*' to include all existing columns
            if self._select_star:
                fields_sql = f"*, {fields_sql}"
        else:
            fields_sql = "*"

        distinct_keyword = "DISTINCT " if self._distinct else ""
        parts.append(f"SELECT {distinct_keyword}{fields_sql}")

        # FROM clause
        if self._table_function:
            # Use table function instead of table name
            table_func_sql = self._table_function.to_sql(quote_char=quote_char)
            # Add alias for table function (required by ClickHouse for JOINs)
            alias = self._get_table_alias()
            parts.append(
                f"FROM {table_func_sql} AS {format_identifier(alias, quote_char)}"
            )
        elif self.table_name:
            parts.append(f"FROM {format_identifier(self.table_name, quote_char)}")

        # JOIN clauses
        if self._joins:
            for other_ds, join_type, join_condition in self._joins:
                # Generate JOIN clause
                join_keyword = join_type.value if join_type.value else ""
                if join_keyword:
                    join_clause = f"{join_keyword} JOIN"
                else:
                    join_clause = "JOIN"

                # Handle subquery joins
                if isinstance(other_ds, DataStore) and other_ds._is_subquery:
                    other_table = other_ds.to_sql(
                        quote_char=quote_char, as_subquery=True
                    )
                elif isinstance(other_ds, DataStore) and other_ds._table_function:
                    # Use table function for the joined table with alias
                    table_func_sql = other_ds._table_function.to_sql(
                        quote_char=quote_char
                    )
                    alias = other_ds._get_table_alias()
                    other_table = (
                        f"{table_func_sql} AS {format_identifier(alias, quote_char)}"
                    )
                else:
                    other_table = format_identifier(other_ds.table_name, quote_char)

                # Handle USING vs ON syntax
                if isinstance(join_condition, tuple) and join_condition[0] == "USING":
                    # USING (col1, col2, ...) syntax
                    columns = join_condition[1]
                    using_cols = ", ".join(
                        format_identifier(c, quote_char) for c in columns
                    )
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
            groupby_sql = ", ".join(
                field.to_sql(quote_char=quote_char) for field in self._groupby_fields
            )
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
                direction = "ASC" if ascending else "DESC"
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

        return " ".join(parts)

    def _generate_insert_sql(self, quote_char: str) -> str:
        """Generate INSERT SQL (ClickHouse style)."""
        # Determine target (table function or table name)
        if self._table_function:
            if not self._table_function.can_write:
                raise QueryError(
                    f"Table function '{self.source_type}' does not support writing. "
                    f"Read-only table functions: mongodb, sqlite, iceberg, deltaLake, hudi, numbers, generateRandom"
                )
            target = (
                f"TABLE FUNCTION {self._table_function.to_sql(quote_char=quote_char)}"
            )
        elif self.table_name:
            target = format_identifier(self.table_name, quote_char)
        else:
            raise QueryError("Table name or table function required for INSERT")

        parts = [f"INSERT INTO {target}"]

        # Columns
        if self._insert_columns:
            columns_sql = ", ".join(
                format_identifier(col, quote_char) for col in self._insert_columns
            )
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
                        row_values.append("NULL")
                    elif isinstance(value, bool):
                        row_values.append("1" if value else "0")
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

        return " ".join(parts)

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
                value_sql = "NULL"
            elif isinstance(value, bool):
                value_sql = "1" if value else "0"
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

        return " ".join(parts)

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
            raise QueryError(
                "ClickHouse DELETE requires WHERE clause. Use WHERE 1=1 to delete all rows."
            )

        return " ".join(parts)

    # ========== Dynamic Field Access ==========

    # List of special attribute names that should not be treated as column names
    _RESERVED_ATTRS = frozenset(
        {
            "is_immutable",
            "is_mutable",
            "is_copy",
            "is_view",
            "config",
            "index",
            "columns",
            "values",
            "dtypes",
            "shape",
            "size",
            "ndim",
            "empty",
            "T",
            "axes",
            "to_pandas",  # Prevent being treated as column name
            "to_df",
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
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Don't treat reserved/special attributes as column names
        if name in self._RESERVED_ATTRS:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Return ColumnExpr that wraps a Field and can execute
        return ColumnExpr(Field(name), self)

    # ========== Copy Support ==========

    def __copy__(self) -> "DataStore":
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
        if hasattr(self, "_operation_history"):
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
        For connection-level DataStore, shows connection info with masked password.
        """
        # Connection-level mode - show connection info
        if (
            self._connection_mode in ("connection", "database")
            and not self._has_sql_state()
            and not self._lazy_ops
        ):
            return self._connection_repr()

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
        import html

        # Connection-level mode - show connection info
        if (
            self._connection_mode in ("connection", "database")
            and not self._has_sql_state()
            and not self._lazy_ops
        ):
            return f"<pre>{html.escape(self._connection_repr())}</pre>"

        # If we have operations, execute and show the DataFrame
        if self._has_sql_state() or self._lazy_ops:
            try:
                df = self._execute()
                return df._repr_html_()
            except Exception as e:
                # If execution fails, show error in HTML
                return f"<div><strong>DataStore</strong> (execution failed: {html.escape(str(e))})</div>"

        # Fallback: show basic info in HTML
        info_html = "<div><strong>DataStore</strong><br>"
        info_html += f"Source type: {html.escape(self.source_type or '')}<br>"
        if self.table_name:
            info_html += f"Table: {html.escape(self.table_name)}<br>"
        if self._table_function:
            info_html += "Using table function<br>"
        info_html += "</div>"
        return info_html

    def _connection_repr(self) -> str:
        """
        Generate repr string for connection-level DataStore with masked sensitive fields.

        Returns:
            String like: DataStore(source='clickhouse', host='localhost:9000', user='default', password='***')
        """
        parts = [f"DataStore(source={self.source_type!r}"]

        # Add remote params with masking
        for key, value in self._remote_params.items():
            if value is not None:
                # Mask sensitive fields
                if key.lower() in self._SENSITIVE_FIELDS:
                    parts.append(f", {key}={self._MASKED_VALUE!r}")
                else:
                    parts.append(f", {key}={value!r}")

        # Show default database if set
        if self._default_database:
            parts.append(f", database={self._default_database!r}")

        # Show default table if set
        if self._default_table:
            parts.append(f", table={self._default_table!r}")

        parts.append(")")
        return "".join(parts)
