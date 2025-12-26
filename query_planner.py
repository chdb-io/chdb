"""
Query Planner for DataStore.

This module provides query planning and optimization for the DataStore execution pipeline.
It analyzes LazyOp chains and determines the optimal execution strategy, separating
operations that can be pushed to SQL from those that require Pandas execution.

Design Principles:
- API style does not determine execution engine
- Execution engine selection happens at execution time, not at API call time
- SQL pushdown is an optimization, not a requirement

Key Classes:
- QueryPlan: Intermediate representation of the execution plan
- QueryPlanner: Analyzes LazyOp chains and produces QueryPlans

SQL building is handled by SQLExecutionEngine in sql_executor.py.
"""

from typing import List, Optional, Set, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import re

from .lazy_ops import (
    LazyOp,
    LazyRelationalOp,
    LazyGroupByAgg,
    LazyFilter,
    LazyTransform,
    LazyApply,
    LazyColumnAssignment,
    LazyColumnSelection,
    LazySQLQuery,
    LazyWhere,
    LazyMask,
)
from .expressions import Field, Expression
from .config import get_logger
import pandas as pd

if TYPE_CHECKING:
    from .core import DataStore

# Import CaseWhenExpr from sql_executor (canonical location)
# This re-export maintains backward compatibility for existing imports
from .sql_executor import CaseWhenExpr, WhereMaskCaseExpr


@dataclass
class QueryPlan:
    """
    Intermediate representation of an execution plan.

    This separates operations into SQL-pushable and Pandas-only phases,
    and captures optimization opportunities like GroupBy SQL pushdown.

    Attributes:
        sql_ops: Operations that can be executed via SQL
        df_ops: Operations that require Pandas execution
        groupby_agg: Optional GroupBy aggregation that can be pushed to SQL
        where_ops: List of LazyWhere/LazyMask operations that can be pushed to SQL
        layers: Nested subquery layers (for LIMIT-before-WHERE patterns)
        has_sql_source: Whether there's a SQL-compatible data source
        first_df_op_idx: Index of first DataFrame-only operation in original chain
        where_columns: Column names referenced in WHERE conditions (for conflict detection)
        alias_renames: Dict mapping temp alias -> original alias (for conflict resolution)
    """

    sql_ops: List[LazyOp] = field(default_factory=list)
    df_ops: List[LazyOp] = field(default_factory=list)
    groupby_agg: Optional[LazyGroupByAgg] = None
    where_ops: List[LazyOp] = field(default_factory=list)  # LazyWhere/LazyMask for SQL CASE WHEN
    layers: List[List[LazyOp]] = field(default_factory=list)
    has_sql_source: bool = False
    first_df_op_idx: Optional[int] = None
    where_columns: Set[str] = field(default_factory=set)
    alias_renames: Dict[str, str] = field(default_factory=dict)  # temp_alias -> original_alias

    def has_two_phases(self) -> bool:
        """Check if execution requires both SQL and DataFrame phases."""
        return self.has_sql_source and self.first_df_op_idx is not None

    def needs_nested_subqueries(self) -> bool:
        """Check if SQL needs nested subqueries (multiple layers)."""
        return len(self.layers) > 1

    def describe(self) -> str:
        """Return a human-readable description of the plan."""
        lines = []
        lines.append(f"QueryPlan:")
        lines.append(f"  SQL source: {self.has_sql_source}")
        lines.append(f"  SQL ops: {len(self.sql_ops)}")
        lines.append(f"  DataFrame ops: {len(self.df_ops)}")
        lines.append(f"  GroupBy pushdown: {self.groupby_agg is not None}")
        lines.append(f"  Nested layers: {len(self.layers)}")
        return "\n".join(lines)


@dataclass
class ExecutionSegment:
    """
    A segment of operations that can be executed together with the same engine.

    This enables true SQL-Pandas-SQL interleaving:
    - SQL segments are executed via chDB (either from source or via Python() table function)
    - Pandas segments are executed in-memory on DataFrames

    Attributes:
        segment_type: 'sql' or 'pandas'
        ops: List of operations in this segment
        plan: QueryPlan for SQL segments (includes layers, groupby_agg, where_ops)
        is_first_segment: Whether this is the first segment (uses original data source)
    """

    segment_type: str  # 'sql' or 'pandas'
    ops: List[LazyOp] = field(default_factory=list)
    plan: Optional[QueryPlan] = None  # For SQL segments
    is_first_segment: bool = False

    def is_sql(self) -> bool:
        """Check if this is a SQL segment."""
        return self.segment_type == 'sql'

    def is_pandas(self) -> bool:
        """Check if this is a Pandas segment."""
        return self.segment_type == 'pandas'

    def describe(self) -> str:
        """Return a human-readable description of this segment."""
        engine = "chDB" if self.is_sql() else "Pandas"
        source = " (from source)" if self.is_first_segment else " (on DataFrame)"
        return f"{engine}{source}: {len(self.ops)} ops"


@dataclass
class ExecutionPlan:
    """
    Complete execution plan with multiple segments for SQL-Pandas interleaving.

    This replaces the single-boundary QueryPlan approach with a multi-segment
    approach that maximizes SQL pushdown opportunities.

    Example:
        ops: [filter1, select, apply, filter2, transform, filter3]
        segments:
          1. SQL (from source): [filter1, select]
          2. Pandas: [apply]
          3. SQL (on DataFrame): [filter2]
          4. Pandas: [transform]
          5. SQL (on DataFrame): [filter3]

    Attributes:
        segments: Ordered list of execution segments
        has_sql_source: Whether original data source supports SQL
    """

    segments: List[ExecutionSegment] = field(default_factory=list)
    has_sql_source: bool = False

    def describe(self) -> str:
        """Return a human-readable description of the execution plan."""
        lines = ["ExecutionPlan:"]
        lines.append(f"  SQL source: {self.has_sql_source}")
        lines.append(f"  Segments: {len(self.segments)}")
        for i, seg in enumerate(self.segments, 1):
            lines.append(f"    [{i}] {seg.describe()}")
        return "\n".join(lines)

    def total_ops(self) -> int:
        """Return total number of operations across all segments."""
        return sum(len(seg.ops) for seg in self.segments)

    def sql_segment_count(self) -> int:
        """Return number of SQL segments."""
        return sum(1 for seg in self.segments if seg.is_sql())

    def pandas_segment_count(self) -> int:
        """Return number of Pandas segments."""
        return sum(1 for seg in self.segments if seg.is_pandas())


class QueryPlanner:
    """
    Query planner that analyzes LazyOp chains and produces execution plans.

    The planner determines:
    1. Which operations can be pushed to SQL
    2. Which operations require Pandas execution
    3. How to structure nested subqueries for complex patterns
    4. Optimization opportunities like GroupBy SQL pushdown

    Example:
        >>> planner = QueryPlanner()
        >>> plan = planner.plan(lazy_ops, has_sql_source=True)
        >>> if plan.has_two_phases():
        >>>     # Execute SQL phase first, then DataFrame phase
    """

    def __init__(self):
        self._logger = get_logger()

    def plan(self, lazy_ops: List[LazyOp], has_sql_source: bool, schema: Dict[str, str] = None) -> QueryPlan:
        """
        Analyze LazyOp chain and produce an execution plan.

        Args:
            lazy_ops: List of lazy operations to analyze
            has_sql_source: Whether there's a SQL-compatible data source
                           (table_function or table_name)
            schema: Optional dict mapping column names to types (for type-aware SQL pushdown)

        Returns:
            QueryPlan with optimized execution strategy
        """
        plan = QueryPlan(has_sql_source=has_sql_source)

        if not lazy_ops:
            return plan

        # Collect WHERE column names for alias conflict detection
        plan.where_columns = self._collect_where_columns(lazy_ops)

        # Find the SQL/DataFrame boundary
        plan.first_df_op_idx, plan.groupby_agg, plan.where_ops, plan.alias_renames = self._find_sql_boundary(
            lazy_ops, has_sql_source, plan.where_columns, schema
        )

        # Split operations
        if plan.first_df_op_idx is not None:
            plan.sql_ops = lazy_ops[: plan.first_df_op_idx]
            plan.df_ops = lazy_ops[plan.first_df_op_idx :]
        else:
            plan.sql_ops = lazy_ops.copy()
            plan.df_ops = []

        # Remove GroupByAgg from sql_ops if it's being pushed separately
        if plan.groupby_agg:
            plan.sql_ops = [op for op in plan.sql_ops if not isinstance(op, LazyGroupByAgg)]

        # Remove LazyWhere/LazyMask from sql_ops if they're being pushed separately
        if plan.where_ops:
            plan.sql_ops = [op for op in plan.sql_ops if not isinstance(op, (LazyWhere, LazyMask))]

        # Build layers for nested subqueries (LIMIT-before-WHERE patterns)
        if has_sql_source:
            plan.layers = self._build_layers(plan.sql_ops)

        self._logger.debug(
            "Query plan: %d SQL ops, %d DataFrame ops, %d where ops",
            len(plan.sql_ops),
            len(plan.df_ops),
            len(plan.where_ops),
        )

        return plan

    def _collect_where_columns(self, ops: List[LazyOp]) -> Set[str]:
        """
        Extract column names from WHERE conditions.

        This is used to detect alias conflicts when pushing GroupBy to SQL.

        Args:
            ops: List of lazy operations

        Returns:
            Set of column names referenced in WHERE conditions
        """
        where_columns = set()

        for op in ops:
            if isinstance(op, LazyRelationalOp) and op.op_type == 'WHERE' and op.condition:
                try:
                    cond_sql = op.condition.to_sql(quote_char='"')
                    # Extract quoted column names
                    where_columns.update(re.findall(r'"(\w+)"', cond_sql))
                except Exception:
                    pass

        return where_columns

    def _find_sql_boundary(
        self, ops: List[LazyOp], has_sql_source: bool, where_columns: Set[str], schema: Dict[str, str] = None
    ) -> Tuple[Optional[int], Optional[LazyGroupByAgg], List[LazyOp], Dict[str, str]]:
        """
        Find the first operation that cannot be pushed to SQL.

        Also detects GroupByAgg and LazyWhere/LazyMask that can be pushed to SQL.

        Args:
            ops: List of lazy operations
            has_sql_source: Whether there's a SQL-compatible data source
            where_columns: Column names in WHERE conditions (for conflict detection)
            schema: Optional dict mapping column names to types (for type-aware SQL pushdown)

        Returns:
            Tuple of (first_df_op_idx, groupby_agg_op, where_ops, alias_renames)
        """
        first_df_op_idx = None
        groupby_agg_op = None
        where_ops = []  # LazyWhere/LazyMask that can be pushed to SQL
        alias_renames = {}  # temp_alias -> original_alias for conflict resolution

        for i, op in enumerate(ops):
            if isinstance(op, LazyRelationalOp):
                # Relational ops can be pushed to SQL
                continue

            elif isinstance(op, LazyGroupByAgg) and groupby_agg_op is None:
                # Check if GroupByAgg can be pushed to SQL
                can_push = has_sql_source
                if can_push and op.agg_dict:
                    # Check for alias conflict with WHERE columns
                    # ClickHouse has a quirk: if SELECT has `agg(col) AS col` where `col` is
                    # also referenced in WHERE, ClickHouse will incorrectly try to use the
                    # aggregate function in the WHERE clause, causing ILLEGAL_AGGREGATION error.
                    # Example that fails:
                    #   SELECT category, sum(int_col) AS int_col FROM t WHERE int_col > 200 GROUP BY category
                    #
                    # OPTIMIZATION: Instead of falling back to pandas, we use temporary aliases
                    # for conflicting columns and rename them back after SQL execution.
                    agg_aliases = self._get_agg_aliases(op, schema)
                    conflict_aliases = agg_aliases & where_columns
                    if conflict_aliases:
                        # Record the aliases that need renaming
                        for alias in conflict_aliases:
                            temp_alias = f"__agg_{alias}__"
                            alias_renames[temp_alias] = alias
                        self._logger.debug(
                            "  [GroupBy] Using temp aliases for conflict resolution: %s",
                            alias_renames,
                        )

                if can_push:
                    groupby_agg_op = op
                    continue  # Include in SQL, continue looking
                else:
                    # Cannot push to SQL - this operation breaks the SQL chain
                    first_df_op_idx = i
                    break

            elif isinstance(op, (LazyWhere, LazyMask)):
                # Check if LazyWhere/LazyMask can be pushed to SQL
                # Pass schema for type-aware checking (auto fallback for incompatible types)
                if has_sql_source and op.can_push_to_sql(schema):
                    where_ops.append(op)
                    self._logger.debug("  [Where/Mask] Can push to SQL: CASE WHEN")
                    continue  # Include in SQL, continue looking
                else:
                    # Cannot push to SQL - breaks the chain
                    # (either function_config or type incompatibility)
                    self._logger.debug("  [Where/Mask] Falling back to Pandas (type incompatibility or config)")
                    first_df_op_idx = i
                    break

            elif isinstance(op, LazySQLQuery):
                # LazySQLQuery is a SQL operation but executes via Python() table function
                # It cannot be pushed further, so it marks a boundary
                first_df_op_idx = i
                break

            # Any other operation (LazyColumnAssignment, LazyGroupByFilter, etc.)
            # breaks the SQL chain
            first_df_op_idx = i
            break

        return first_df_op_idx, groupby_agg_op, where_ops, alias_renames

    def _get_agg_aliases(self, op: LazyGroupByAgg, schema: Dict[str, str] = None) -> Set[str]:
        """
        Get alias names that would be created by a GroupByAgg operation.

        Args:
            op: LazyGroupByAgg operation
            schema: Column schema (used for count() to get non-groupby columns)

        Returns:
            Set of alias names
        """
        agg_aliases = set()

        if op.agg_dict:
            for col, funcs in op.agg_dict.items():
                if isinstance(funcs, (list, tuple)):
                    agg_aliases.update(funcs)  # Function names as aliases
                else:
                    agg_aliases.add(col)  # Column name as alias for single func
        elif op.agg_func == 'count' and schema:
            # count() creates aliases for all non-groupby columns
            non_groupby_cols = [c for c in schema.keys() if c not in op.groupby_cols]
            agg_aliases.update(non_groupby_cols)

        return agg_aliases

    def _build_layers(self, ops: List[LazyOp]) -> List[List[LazyOp]]:
        """
        Build nested subquery layers for complex SQL patterns.

        Layer boundaries are created when:
        - WHERE follows LIMIT/OFFSET (pandas: slice then filter)
        - ORDER BY follows LIMIT/OFFSET (pandas: slice then sort)

        This preserves pandas-like execution order semantics in SQL.

        Args:
            ops: List of SQL-pushable operations

        Returns:
            List of operation lists, innermost first
        """
        layers = []
        current_layer = []
        pending_limit_offset = False

        for op in ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type in ('WHERE', 'ORDER BY') and pending_limit_offset:
                    # WHERE or ORDER BY after LIMIT/OFFSET - start new layer
                    if current_layer:
                        layers.append(current_layer)
                    current_layer = [op]
                    pending_limit_offset = False
                else:
                    current_layer.append(op)
                    if op.op_type in ('LIMIT', 'OFFSET'):
                        pending_limit_offset = True

        if current_layer:
            layers.append(current_layer)

        return layers

    def can_push_all_to_sql(self, plan: QueryPlan) -> bool:
        """
        Check if all operations can be executed via SQL.

        Args:
            plan: Query plan to check

        Returns:
            True if no DataFrame operations are needed
        """
        return len(plan.df_ops) == 0 and plan.has_sql_source

    def plan_segments(
        self, lazy_ops: List[LazyOp], has_sql_source: bool, schema: Dict[str, str] = None
    ) -> ExecutionPlan:
        """
        Analyze LazyOp chain and produce a segmented execution plan.

        This method splits the operation chain into alternating SQL and Pandas
        segments, maximizing SQL pushdown opportunities even when Pandas-only
        operations are interspersed.

        Example:
            ops: [filter1, select, apply, filter2, transform, filter3]
            Result:
              Segment 1 (SQL from source): [filter1, select]
              Segment 2 (Pandas): [apply]
              Segment 3 (SQL on DataFrame): [filter2]
              Segment 4 (Pandas): [transform]
              Segment 5 (SQL on DataFrame): [filter3]

        Args:
            lazy_ops: List of lazy operations to analyze
            has_sql_source: Whether there's a SQL-compatible data source
            schema: Optional dict mapping column names to types

        Returns:
            ExecutionPlan with multiple segments
        """
        exec_plan = ExecutionPlan(has_sql_source=has_sql_source)

        if not lazy_ops:
            # Even with no operations, if there's a SQL source we need a segment
            # to read the data (SELECT *)
            if has_sql_source:
                segment = ExecutionSegment(
                    segment_type='sql',
                    ops=[],
                    is_first_segment=True,
                )
                # Create an empty QueryPlan for SELECT *
                plan = QueryPlan(has_sql_source=True)
                segment.plan = plan
                exec_plan.segments.append(segment)
            return exec_plan

        # Track effective schema through operations (column selections reduce schema)
        effective_schema = dict(schema) if schema else {}

        # Classify each operation as SQL-pushable or Pandas-only
        op_types = []  # List of ('sql', op) or ('pandas', op)
        for op in lazy_ops:
            # Update effective schema based on column selection
            # LazyColumnSelection: df[["col1", "col2"]]
            # LazyRelationalOp SELECT: also used for column selection
            if isinstance(op, LazyColumnSelection) and effective_schema:
                effective_schema = {col: effective_schema[col] for col in op.columns if col in effective_schema}
                self._logger.debug("  [Schema] After LazyColumnSelection: %s", list(effective_schema.keys()))
            elif isinstance(op, LazyRelationalOp) and op.op_type == 'SELECT' and effective_schema:
                # LazyRelationalOp SELECT stores columns in fields (as Field objects or strings)
                if hasattr(op, 'fields') and op.fields:
                    # Extract column names from fields
                    selected_cols = []
                    for f in op.fields:
                        if isinstance(f, str):
                            selected_cols.append(f)
                        elif isinstance(f, Field):
                            # Field.name is the column name (may have quotes)
                            col_name = f.name.strip('"\'')
                            selected_cols.append(col_name)
                    if selected_cols:
                        effective_schema = {
                            col: effective_schema[col] for col in selected_cols if col in effective_schema
                        }
                        self._logger.debug("  [Schema] After SELECT: %s", list(effective_schema.keys()))

            if self._can_push_op_to_sql(op, effective_schema):
                op_types.append(('sql', op))
            else:
                op_types.append(('pandas', op))

        # Group consecutive operations of the same type into segments
        current_type = None
        current_ops = []
        is_first = True

        for op_type, op in op_types:
            if op_type != current_type:
                # Save previous segment
                if current_ops:
                    segment = self._create_segment(current_type, current_ops, is_first, has_sql_source, schema)
                    exec_plan.segments.append(segment)
                    is_first = False
                # Start new segment
                current_type = op_type
                current_ops = [op]
            else:
                current_ops.append(op)

        # Save last segment
        if current_ops:
            segment = self._create_segment(current_type, current_ops, is_first, has_sql_source, schema)
            exec_plan.segments.append(segment)

        self._logger.debug(
            "Execution plan: %d segments (%d SQL, %d Pandas)",
            len(exec_plan.segments),
            exec_plan.sql_segment_count(),
            exec_plan.pandas_segment_count(),
        )

        return exec_plan

    def _can_push_op_to_sql(self, op: LazyOp, schema: Dict[str, str] = None) -> bool:
        """
        Check if a single operation can be pushed to SQL.

        Args:
            op: The operation to check
            schema: Column schema for type-aware checking

        Returns:
            True if the operation can be executed via SQL
        """
        if isinstance(op, LazyRelationalOp):
            # Relational ops (WHERE, SELECT, ORDER BY, LIMIT, OFFSET) can be pushed
            return True

        if isinstance(op, LazyGroupByAgg):
            # GroupBy aggregation can be pushed
            # Note: alias conflict handling is done at segment planning time
            return True

        if isinstance(op, (LazyWhere, LazyMask)):
            # Check type compatibility for SQL pushdown
            return op.can_push_to_sql(schema)

        if isinstance(op, LazySQLQuery):
            # LazySQLQuery has its own execution method and can_push_to_sql() returns False
            # because it already contains a complete SQL query
            # It should NOT be grouped with other SQL ops - treat as Pandas segment
            # so it gets executed via its own execute() method
            return False

        # All other ops (LazyColumnAssignment, LazyFilter, LazyTransform, LazyApply, etc.)
        # require Pandas execution
        return False

    def _create_segment(
        self,
        segment_type: str,
        ops: List[LazyOp],
        is_first: bool,
        has_sql_source: bool,
        schema: Dict[str, str] = None,
    ) -> ExecutionSegment:
        """
        Create an ExecutionSegment from a list of operations.

        For SQL segments, also creates a QueryPlan for SQL building.

        Args:
            segment_type: 'sql' or 'pandas'
            ops: List of operations in this segment
            is_first: Whether this is the first segment
            has_sql_source: Whether original source supports SQL
            schema: Column schema

        Returns:
            ExecutionSegment
        """
        segment = ExecutionSegment(
            segment_type=segment_type,
            ops=ops.copy(),
            is_first_segment=is_first,
        )

        if segment_type == 'sql':
            # Create a QueryPlan for this SQL segment
            # The segment can use SQL if:
            # - It's the first segment and has_sql_source is True, OR
            # - It's a subsequent segment (will use Python() table function)
            can_use_sql = is_first and has_sql_source or not is_first

            if can_use_sql:
                plan = QueryPlan(has_sql_source=True)
                plan.sql_ops = ops.copy()

                # Collect WHERE columns first for alias conflict detection
                plan.where_columns = self._collect_where_columns(ops)

                # Handle special ops
                for op in ops:
                    if isinstance(op, LazyGroupByAgg):
                        plan.groupby_agg = op
                        # Check for alias conflicts with WHERE columns
                        agg_aliases = self._get_agg_aliases(op, schema)
                        conflict_aliases = agg_aliases & plan.where_columns
                        if conflict_aliases:
                            for alias in conflict_aliases:
                                temp_alias = f"__agg_{alias}__"
                                plan.alias_renames[temp_alias] = alias
                            self._logger.debug(
                                "  [GroupBy] Using temp aliases for conflict resolution: %s",
                                plan.alias_renames,
                            )
                    elif isinstance(op, (LazyWhere, LazyMask)):
                        plan.where_ops.append(op)

                # Remove special ops from sql_ops (they're handled separately)
                plan.sql_ops = [op for op in plan.sql_ops if not isinstance(op, (LazyGroupByAgg, LazyWhere, LazyMask))]

                # Build layers for nested subqueries
                plan.layers = self._build_layers(plan.sql_ops)

                segment.plan = plan

        return segment
