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
    LazyDataFrameSource,
    LazyDistinct,
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
    Intermediate representation of one SQL segment's execution plan.

    Produced by :meth:`QueryPlanner.plan_segments` (one ``QueryPlan`` per SQL
    ``ExecutionSegment``). The SQL build pipeline (``build_sql_from_plan``
    in :mod:`sql_executor`) consumes a ``QueryPlan`` and produces SQL by
    chaining :meth:`SQLExecutionEngine._build_layer` over the plan's
    ``layers``.

    Data flow:

    1. The planner segments lazy ops into SQL / Pandas / SQL / ... segments.
    2. For each SQL segment, ``_create_segment`` builds a ``QueryPlan``
       whose ``layers`` contain the FULL op list (including
       ``LazyGroupByAgg``, ``LazyWhere``, ``LazyMask``, pushable
       ``LazyApply``) per nested-subquery layer.
    3. ``build_sql_from_plan`` runs ``_build_layered_sql`` which calls
       ``_build_layer`` per layer; each layer dispatches by op type.
    4. Any temp aliases the SQL builders introduce
       (``temp_alias -> original_alias``) are recorded in
       ``alias_renames``. After execution the post-processing step in
       ``core.py`` reads ``alias_renames`` and renames the result
       DataFrame columns back to the user-visible names.

    Attributes:
        sql_ops: Relational + column-assignment ops in the segment, with
            special ops (``LazyGroupByAgg``, ``LazyWhere``, ``LazyMask``,
            pushable ``LazyApply``) excluded. Kept for legacy helpers
            (``_build_sql_with_builder``, ``_check_limit_before_where``)
            that only know how to iterate WHERE/ORDER BY/LIMIT/OFFSET/
            SELECT/ColumnAssignment.
        df_ops: Operations that require Pandas execution (only used by the
            removed ``plan()`` method; segmented planning splits these
            into separate Pandas ``ExecutionSegment``s instead).
        groupby_agg: Convenience pointer to the (single) ``LazyGroupByAgg``
            in ``layers`` (or one synthesized from a pushable
            ``LazyApply``). Read by post-execution code for set_index,
            MultiIndex conversion, dtype corrections.
        where_ops: Convenience list of ``LazyWhere``/``LazyMask`` ops in
            ``layers``. Read by ``_build_sql_for_dataframe`` for CASE
            WHEN temp alias mapping.
        layers: Canonical input for SQL building. Each entry is a layer's
            op list; layer 0 reads from the table source, layers 1+ read
            from ``(inner_sql) AS __subqN__``.
        has_sql_source: Whether the segment has a SQL-compatible source.
        first_df_op_idx: Vestigial - was set by the removed ``plan()``
            method. Always ``None`` from segmented planning.
        where_columns: Source-column names referenced in pre-aggregation
            WHEREs. Drives the planner's source-col / agg-alias conflict
            detection (which pre-populates ``alias_renames``).
        alias_renames: Map ``temp_alias -> original_alias``. Written
            during SQL building, consumed at post-execution to rename
            result DataFrame columns back to user-visible names.
    """

    sql_ops: List[LazyOp] = field(default_factory=list)
    df_ops: List[LazyOp] = field(default_factory=list)
    groupby_agg: Optional[LazyGroupByAgg] = None
    where_ops: List[LazyOp] = field(default_factory=list)
    layers: List[List[LazyOp]] = field(default_factory=list)
    has_sql_source: bool = False
    first_df_op_idx: Optional[int] = None
    where_columns: Set[str] = field(default_factory=set)
    alias_renames: Dict[str, str] = field(default_factory=dict)

    def has_two_phases(self) -> bool:
        """Check if execution requires both SQL and DataFrame phases."""
        return self.has_sql_source and self.first_df_op_idx is not None

    def needs_nested_subqueries(self) -> bool:
        """Check if SQL needs nested subqueries (multiple layers)."""
        return len(self.layers) > 1

    def describe(self) -> str:
        """Return a human-readable description of the plan."""
        lines = []
        lines.append("QueryPlan:")
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
        return self.segment_type == "sql"

    def is_pandas(self) -> bool:
        """Check if this is a Pandas segment."""
        return self.segment_type == "pandas"

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

    # NOTE: the legacy ``plan()`` method (which produced a single QueryPlan
    # from a flat lazy_ops list) was removed - call sites (``_execute``,
    # ``_build_execution_sql`` in core.py, and the test suite) now use
    # ``plan_segments`` and take the plan from the first SQL segment when
    # they need a single-plan view.

    def _collect_where_columns(self, ops: List[LazyOp]) -> Set[str]:
        """
        Extract source-column names referenced in pre-aggregation WHERE clauses.

        Convenience wrapper around
        :meth:`_collect_where_columns_per_segment` that returns the source-col
        set for the chain *up to* (and used as input by) the first
        ``LazyGroupByAgg``. The result is consumed by alias-conflict detection:
        an aggregate output alias that collides with a source column referenced
        in a pre-agg WHERE has to be renamed to a temp alias so chDB doesn't
        mis-bind the aggregate inside the WHERE.

        Post-aggregation WHEREs are intentionally excluded - they operate on
        the *output* of the aggregation (e.g. ``count > 5000``), so treating
        them as source-col conflicts would force a useless rename and break
        the outer reference.

        Args:
            ops: List of lazy operations

        Returns:
            Set of column names referenced in pre-aggregation WHERE conditions.
        """
        # Plans currently carry at most one LazyGroupByAgg per QueryPlan, so
        # the per-segment helper collapses to "ops before the (only) agg".
        return self._collect_where_columns_per_segment(ops, segment_end=None)

    def _collect_where_columns_per_segment(
        self,
        ops: List[LazyOp],
        segment_end: Optional[int] = None,
    ) -> Set[str]:
        """
        Position-aware collector: source columns referenced in WHEREs that
        come strictly before the chain's first ``LazyGroupByAgg`` (or before
        ``segment_end`` if supplied).

        This makes the conflict-detection semantics explicit: WHEREs after
        the boundary act on aggregation output, not source rows, and must not
        contribute source-col aliases.

        Args:
            ops: List of lazy operations
            segment_end: Optional explicit upper bound (exclusive); when None,
                stops at the first encountered ``LazyGroupByAgg``.

        Returns:
            Set of column names from WHERE conditions within the segment.
        """
        where_columns: Set[str] = set()
        for idx, op in enumerate(ops):
            if segment_end is not None and idx >= segment_end:
                break
            if segment_end is None and isinstance(op, LazyGroupByAgg):
                break
            if (
                isinstance(op, LazyRelationalOp)
                and op.op_type == "WHERE"
                and op.condition
            ):
                try:
                    cond_sql = op.condition.to_sql(quote_char='"')
                    where_columns.update(re.findall(r'"(\w+)"', cond_sql))
                except Exception:
                    pass
        return where_columns

    def _get_agg_aliases(
        self, op: LazyGroupByAgg, schema: Dict[str, str] = None
    ) -> Set[str]:
        """
        Get alias names that would be created by a GroupByAgg operation.

        Args:
            op: LazyGroupByAgg operation
            schema: Column schema (used for count() to get non-groupby columns)

        Returns:
            Set of alias names
        """
        agg_aliases = set()

        if op.agg_dict is not None and isinstance(op.agg_dict, dict):
            for col, funcs in op.agg_dict.items():
                if isinstance(funcs, (list, tuple)):
                    agg_aliases.update(funcs)  # Function names as aliases
                else:
                    agg_aliases.add(col)  # Column name as alias for single func
        elif op.agg_func == "count" and schema:
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
        - OFFSET follows LIMIT (pandas: chained slices like [10:50][5:20])
        - LIMIT follows LIMIT (pandas: chained limits like [:50][:20])
        - WHERE follows computed column that came after LIMIT (dependency)
        - LazyGroupByAgg follows LIMIT/OFFSET (pandas: ``head(N).groupby()``
          aggregates over the first N rows; emitting LIMIT in the same SQL
          layer as GROUP BY would otherwise mean "first N groups" instead).
        - A pure-projection SELECT (e.g. ``df[['a','b']]``) follows a
          WHERE in the same layer and the WHERE references columns that
          would be dropped by the projection. This split keeps the
          WHERE's columns alive in the inner subquery before the outer
          layer projects them away.

        This preserves pandas-like execution order semantics in SQL.

        Args:
            ops: List of SQL-pushable operations

        Returns:
            List of operation lists, innermost first
        """
        layers = []
        current_layer = []
        seen_limit = False  # Have we seen a LIMIT in current layer?
        seen_offset = False  # Have we seen an OFFSET in current layer?
        pending_column_assignments = []  # Column assignments after LIMIT/OFFSET
        computed_columns_after_limit = set()  # Computed columns added after LIMIT

        def _is_pure_projection_select(op):
            """Pure projection SELECT = ``SELECT col1, col2, ...`` with no
            ``*`` marker (pandas ``df[[cols]]``). Assignment-style SELECTs
            (``df.assign(c=...)``) carry a ``*`` and stay in the same
            layer so the assignments survive."""
            return (
                isinstance(op, LazyRelationalOp)
                and op.op_type == "SELECT"
                and op.fields
                and not any(isinstance(f, str) and f == "*" for f in op.fields)
            )

        def _projection_field_names(op):
            names = set()
            for f in op.fields or []:
                name = None
                if isinstance(f, str):
                    name = f
                elif hasattr(f, "name"):
                    name = getattr(f, "name", None)
                if isinstance(name, str):
                    names.add(name.strip('"`'))
            return names

        def _where_refs_columns_not_in(op_set, layer):
            """Does any WHERE in ``layer`` reference a column outside
            ``op_set`` (e.g. one created by an assignment-style
            SELECT/LazyColumnAssignment in the same layer)? If yes, the
            outer projection would drop that column before the WHERE
            could bind to it, so we must split into a new layer that
            wraps the inner WHERE."""
            wheres = [
                prev
                for prev in layer
                if isinstance(prev, LazyRelationalOp)
                and prev.op_type == "WHERE"
                and prev.condition is not None
            ]
            if not wheres:
                return False
            for w in wheres:
                refs = self._extract_referenced_columns(w.condition)
                if refs - op_set:
                    return True
            return False

        def _start_new_layer_with(op):
            """Push current_layer, drop any pending assignments into it
            first, then begin a fresh layer containing ``op``."""
            nonlocal current_layer, seen_limit, seen_offset
            nonlocal computed_columns_after_limit, pending_column_assignments
            if pending_column_assignments:
                current_layer.extend(pending_column_assignments)
                pending_column_assignments = []
            layers.append(current_layer)
            current_layer = [op]
            seen_limit = isinstance(op, LazyRelationalOp) and op.op_type == "LIMIT"
            seen_offset = (
                isinstance(op, LazyRelationalOp) and op.op_type == "OFFSET"
            )
            computed_columns_after_limit = set()

        for op in ops:
            if isinstance(op, LazyRelationalOp):
                needs_new_layer = False

                if op.op_type == "WHERE" and (seen_limit or seen_offset):
                    # WHERE after LIMIT/OFFSET - start new layer
                    needs_new_layer = True
                elif op.op_type == "ORDER BY" and (seen_limit or seen_offset):
                    # ORDER BY after LIMIT/OFFSET - start new layer
                    needs_new_layer = True
                elif op.op_type == "OFFSET" and seen_limit:
                    # OFFSET after LIMIT means chained slices like [10:50][5:20]
                    # The second slice operates on the result of the first
                    needs_new_layer = True
                elif op.op_type == "LIMIT" and seen_limit:
                    # Second LIMIT means chained limits like [:50][:20]
                    # The second limit operates on the result of the first
                    needs_new_layer = True
                elif _is_pure_projection_select(op):
                    proj_fields = _projection_field_names(op)
                    if _where_refs_columns_not_in(proj_fields, current_layer):
                        # Pure projection that drops a column the layer's
                        # WHERE depends on (e.g. ``assign(x=...)[where x>0][[id]]``).
                        # Split into a new layer so the outer SELECT
                        # projects after the inner WHERE has run.
                        needs_new_layer = True

                if needs_new_layer and current_layer:
                    _start_new_layer_with(op)
                else:
                    # Add pending column assignments to current layer first
                    if pending_column_assignments:
                        current_layer.extend(pending_column_assignments)
                        pending_column_assignments = []

                    current_layer.append(op)
                    if op.op_type == "LIMIT":
                        seen_limit = True
                    elif op.op_type == "OFFSET":
                        seen_offset = True

            elif isinstance(op, LazyColumnAssignment):
                # Column assignment after LIMIT/OFFSET needs special handling
                if seen_limit or seen_offset:
                    # Track the column for dependency checking
                    computed_columns_after_limit.add(op.column)
                    # Queue the assignment to be added before any dependent WHERE/ORDER BY
                    pending_column_assignments.append(op)
                else:
                    # No LIMIT/OFFSET yet, add directly to current layer
                    current_layer.append(op)
            elif isinstance(op, LazyGroupByAgg) and (seen_limit or seen_offset):
                # ``head(N).groupby(...).agg(...)`` semantics: pandas
                # aggregates over the first N rows. If LIMIT and GROUP BY
                # live in the same SQL layer we end up emitting
                # ``... GROUP BY ... LIMIT N``, which limits the number of
                # GROUPS instead of the number of input rows. Split into a
                # new layer so the LIMIT applies to the inner subquery
                # (i.e. ``SELECT ... FROM (SELECT ... LIMIT N) GROUP BY``).
                _start_new_layer_with(op)
            else:
                # Other operations - add pending column assignments first, then this op
                if pending_column_assignments:
                    current_layer.extend(pending_column_assignments)
                    pending_column_assignments = []
                current_layer.append(op)

        # Add any remaining pending column assignments
        if pending_column_assignments:
            current_layer.extend(pending_column_assignments)

        if current_layer:
            layers.append(current_layer)

        return layers

    def _extract_referenced_columns(self, expr) -> set:
        """
        Extract all column names referenced in an expression.

        This is used to detect dependencies between column assignments.
        If assignment A references column X, and assignment B targets column X,
        then A depends on B and they cannot be in the same SQL segment.

        ClickHouse has a quirk where alias references in the same SELECT can cause
        unexpected results when multiple columns reference each other.

        Args:
            expr: Expression to analyze (can be Expression, ColumnExpr, or other)

        Returns:
            Set of column names referenced in the expression
        """
        from .expressions import ArithmeticExpression
        from .column_expr import ColumnExpr
        from .functions import Function

        columns = set()

        def _extract(e):
            if e is None:
                return

            if isinstance(e, ColumnExpr):
                # Unwrap ColumnExpr and extract from inner expression
                if e._expr is not None:
                    _extract(e._expr)
                if e._source is not None:
                    _extract(e._source)

            elif isinstance(e, Field):
                # Field directly references a column
                columns.add(e.name)

            elif isinstance(e, ArithmeticExpression):
                # Binary expression - extract from both sides
                _extract(e.left)
                _extract(e.right)

            elif isinstance(e, Function):
                # Function call - extract from arguments
                for arg in e.args:
                    _extract(arg)

            elif hasattr(e, "expr") and e.expr is not None:
                # Wrapper types with an 'expr' attribute
                _extract(e.expr)

        _extract(expr)
        return columns

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
        self,
        lazy_ops: List[LazyOp],
        has_sql_source: bool,
        schema: Dict[str, str] = None,
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
                    segment_type="sql",
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
                effective_schema = {
                    col: effective_schema[col]
                    for col in op.columns
                    if col in effective_schema
                }
                self._logger.debug(
                    "  [Schema] After LazyColumnSelection: %s",
                    list(effective_schema.keys()),
                )
            elif (
                isinstance(op, LazyRelationalOp)
                and op.op_type == "SELECT"
                and effective_schema
            ):
                # LazyRelationalOp SELECT stores columns in fields (as Field objects or strings)
                if hasattr(op, "fields") and op.fields:
                    # Extract column names from fields
                    selected_cols = []
                    for f in op.fields:
                        if isinstance(f, str):
                            selected_cols.append(f)
                        elif isinstance(f, Field):
                            # Field.name is the column name (may have quotes)
                            col_name = f.name.strip("\"'")
                            selected_cols.append(col_name)
                    if selected_cols:
                        effective_schema = {
                            col: effective_schema[col]
                            for col in selected_cols
                            if col in effective_schema
                        }
                        self._logger.debug(
                            "  [Schema] After SELECT: %s", list(effective_schema.keys())
                        )

            # Pass preceding and following operations for context-aware decisions
            op_idx = lazy_ops.index(op) if op in lazy_ops else -1
            preceding_ops = lazy_ops[:op_idx] if op_idx >= 0 else []
            following_ops = lazy_ops[op_idx + 1 :] if op_idx >= 0 else []
            if self._can_push_op_to_sql(
                op, effective_schema, preceding_ops, following_ops
            ):
                op_types.append(("sql", op))
            else:
                op_types.append(("pandas", op))

        # Group consecutive operations of the same type into segments
        # Special case: when multiple LazyColumnAssignment ops target the same column,
        # they must be executed in separate SQL segments to maintain correct execution order.
        # Each assignment must see the result of the previous one.
        current_type = None
        current_ops = []
        is_first = True
        # Track columns assigned in current SQL segment for conflict detection
        assigned_columns_in_segment = set()

        for op_type, op in op_types:
            # Check if this is a column assignment that conflicts with previous ones in segment
            needs_new_segment = False
            if op_type == "sql" and isinstance(op, LazyColumnAssignment):
                if op.column in assigned_columns_in_segment:
                    # Same column being assigned again - need a new segment
                    needs_new_segment = True
                    self._logger.debug(
                        "  [Segment Split] Column '%s' already assigned in current segment, splitting",
                        op.column,
                    )
                else:
                    # Check if this assignment references any column that was assigned
                    # in the current segment (cross-column dependency)
                    # ClickHouse has a quirk where alias references in the same SELECT
                    # can cause unexpected results when columns reference each other
                    referenced_cols = self._extract_referenced_columns(op.expr)
                    dependency_cols = referenced_cols & assigned_columns_in_segment
                    if dependency_cols:
                        # This assignment references a column that was just assigned
                        # Split to ensure correct execution order
                        needs_new_segment = True
                        self._logger.debug(
                            "  [Segment Split] Column '%s' references recently assigned columns %s, splitting",
                            op.column,
                            dependency_cols,
                        )

            if op_type != current_type or needs_new_segment:
                # Save previous segment
                if current_ops:
                    segment = self._create_segment(
                        current_type, current_ops, is_first, has_sql_source, schema
                    )
                    exec_plan.segments.append(segment)
                    is_first = False
                # Start new segment
                current_type = op_type
                current_ops = [op]
                # Reset column tracking for new segment
                assigned_columns_in_segment = set()
                if isinstance(op, LazyColumnAssignment):
                    assigned_columns_in_segment.add(op.column)
            else:
                current_ops.append(op)
                if isinstance(op, LazyColumnAssignment):
                    assigned_columns_in_segment.add(op.column)

        # Save last segment
        if current_ops:
            segment = self._create_segment(
                current_type, current_ops, is_first, has_sql_source, schema
            )
            exec_plan.segments.append(segment)

        self._logger.debug(
            "Execution plan: %d segments (%d SQL, %d Pandas)",
            len(exec_plan.segments),
            exec_plan.sql_segment_count(),
            exec_plan.pandas_segment_count(),
        )

        return exec_plan

    def _can_push_op_to_sql(
        self,
        op: LazyOp,
        schema: Dict[str, str] = None,
        preceding_ops: List[LazyOp] = None,
        following_ops: List[LazyOp] = None,
    ) -> bool:
        """
        Check if a single operation can be pushed to SQL.

        Args:
            op: The operation to check
            schema: Column schema for type-aware checking
            preceding_ops: List of operations that come before this one (for context-aware decisions)
            following_ops: List of operations that come after this one (for ORDER BY cost awareness)

        Returns:
            True if the operation can be executed via SQL
        """
        if isinstance(op, LazyDataFrameSource):
            # LazyDataFrameSource is handled specially in segment creation
            # It should be skipped (not classified as SQL or Pandas)
            # Return True here to keep the SQL chain continuous only if SQL source exists
            # But this depends on context, so we let plan_segments handle it
            # For now, return False to let it fall into Pandas segment by default
            # The actual SQL execution will use _ensure_sql_source() to create table function
            return False

        if isinstance(op, LazyRelationalOp):
            # Most relational ops (WHERE, SELECT, ORDER BY, LIMIT, OFFSET) can be pushed
            # But PANDAS_FILTER is for method-mode ColumnExpr that cannot be converted to SQL
            if op.op_type == "PANDAS_FILTER":
                return False

            if op.op_type == "ORDER BY":
                # Cost-aware ORDER BY optimization: only push ORDER BY to SQL when
                # LIMIT follows in the operation chain. Unbounded ORDER BY forces
                # remote servers to sort the entire dataset before returning results,
                # which is very expensive for large remote tables.
                # When no LIMIT follows, the sort happens in pandas after data fetch.
                following = following_ops or []

                # Check if ORDER BY precedes a GROUP BY with non-order-sensitive
                # aggregation (sum, mean, count, etc.) - the sort is semantically
                # meaningless and should be dropped entirely from SQL
                for f_op in following:
                    if isinstance(f_op, LazyGroupByAgg):
                        if f_op.agg_func not in ("first", "last"):
                            self._logger.debug(
                                "  [ORDER BY] Skipping push to SQL: meaningless before GROUP BY %s()",
                                f_op.agg_func,
                            )
                            return False
                        break  # first/last handled separately

                # ORDER BY after GROUP BY operates on aggregated (small) result set,
                # so SQL sort is cheap - always push to SQL
                has_preceding_groupby = any(
                    isinstance(p_op, LazyGroupByAgg)
                    for p_op in (preceding_ops or [])
                )
                if has_preceding_groupby:
                    return True

                has_limit = any(
                    isinstance(f_op, LazyRelationalOp) and f_op.op_type == "LIMIT"
                    for f_op in following
                )
                if not has_limit:
                    self._logger.debug(
                        "  [ORDER BY] Skipping push to SQL: no LIMIT follows (unbounded sort)"
                    )
                return has_limit

            return True

        if isinstance(op, LazyGroupByAgg):
            # GroupBy aggregation can be pushed if it doesn't use named_agg
            # named_agg (pandas named aggregation syntax) requires Pandas execution
            if not op.can_push_to_sql():
                return False

            # Special case: first() and last() cannot be pushed to SQL when preceded
            # by ORDER BY, because SQL's GROUP BY executes BEFORE ORDER BY, making
            # the sort order meaningless for SQL aggregation functions like any().
            # In pandas, sort_values().groupby().first() returns the first row per
            # group based on the sort order, but SQL cannot express this semantics.
            if op.agg_func in ("first", "last") and preceding_ops:
                has_preceding_orderby = any(
                    isinstance(prev_op, LazyRelationalOp)
                    and prev_op.op_type == "ORDER BY"
                    for prev_op in preceding_ops
                )
                if has_preceding_orderby:
                    self._logger.debug(
                        "  [GroupBy] %s() cannot be pushed to SQL after ORDER BY (requires pandas execution)",
                        op.agg_func,
                    )
                    return False

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

        if isinstance(op, LazyColumnAssignment):
            # Check if the column assignment has an SQL-convertible expression
            # Pass existing columns to prevent pushing when column already exists
            existing_columns = list(schema.keys()) if schema else None
            return op.can_push_to_sql(existing_columns)

        if isinstance(op, LazyApply):
            # LazyApply can be pushed to SQL if it's a simple aggregation pattern
            # e.g., groupby('category').apply(lambda x: x.sum())
            return op.can_push_to_sql()

        if isinstance(op, LazyDistinct):
            # LazyDistinct can be pushed to SQL when executing from a table/file source
            # (first segment). For subsequent segments (on DataFrame), keep as pandas
            # to preserve DataFrame index information.
            # Check if there's a LazyDataFrameSource in preceding ops - if so, this is
            # a DataFrame context where pandas execution is needed for index preservation.
            if preceding_ops and any(
                isinstance(p, LazyDataFrameSource) for p in preceding_ops
            ):
                return False
            return True

        # All other ops (LazyFilter, LazyTransform, etc.)
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

        if segment_type == "sql":
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
                    elif isinstance(op, LazyApply) and op.can_push_to_sql():
                        # Convert LazyApply with simple aggregation to LazyGroupByAgg
                        agg_func = op.get_detected_agg_func()
                        plan.groupby_agg = LazyGroupByAgg(
                            groupby_cols=op.groupby_cols,
                            agg_func=agg_func,
                            sort=True,  # Default pandas behavior
                            as_index=True,
                            dropna=True,
                        )
                        self._logger.debug(
                            "  [Apply] Converted to GroupByAgg in segment: %s -> %s",
                            op.describe(),
                            agg_func,
                        )
                    elif isinstance(op, (LazyWhere, LazyMask)):
                        plan.where_ops.append(op)

                # Build layers from the FULL op list. We keep LazyGroupByAgg,
                # LazyWhere, LazyMask, and pushable LazyApply IN the layer
                # contents so that the per-layer SQL builder
                # (``_build_layer_sql``) can dispatch by op type and naturally
                # emit GROUP BY / CASE WHEN / etc. in the correct subquery
                # layer.
                #
                # When LazyApply is converted to a synthetic LazyGroupByAgg
                # (see above) the layer is patched to use the synthetic op
                # rather than the original LazyApply so SQL building sees a
                # uniform representation.
                raw_layers = self._build_layers(plan.sql_ops)

                synthetic_groupby_agg = (
                    plan.groupby_agg
                    if any(
                        isinstance(op, LazyApply) and op.can_push_to_sql()
                        for op in plan.sql_ops
                    )
                    else None
                )
                if synthetic_groupby_agg is not None:
                    # Replace the originating LazyApply in layers with the
                    # synthesized LazyGroupByAgg so the layer ops list contains
                    # the canonical operation to dispatch on.
                    patched_layers = []
                    replaced = False
                    for layer_ops in raw_layers:
                        new_layer = []
                        for op in layer_ops:
                            if (
                                not replaced
                                and isinstance(op, LazyApply)
                                and op.can_push_to_sql()
                            ):
                                new_layer.append(synthetic_groupby_agg)
                                replaced = True
                            else:
                                new_layer.append(op)
                        patched_layers.append(new_layer)
                    raw_layers = patched_layers

                plan.layers = raw_layers

                # ``plan.sql_ops`` still excludes the special ops because some
                # legacy paths (e.g. ``_build_sql_with_builder``,
                # ``_check_limit_before_where``) iterate plan.sql_ops without
                # the dispatch logic. Keeping that contract avoids touching
                # those paths in this phase.
                plan.sql_ops = [
                    op
                    for op in plan.sql_ops
                    if not isinstance(op, (LazyGroupByAgg, LazyWhere, LazyMask))
                    and not (isinstance(op, LazyApply) and op.can_push_to_sql())
                ]

                segment.plan = plan

        return segment
