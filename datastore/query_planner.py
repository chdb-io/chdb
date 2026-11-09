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
from .expressions import Field, Expression, is_star_item, projection_name
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
       ``alias_renames``. Each aggregation layer renames its temp aliases
       back in SQL, so the SQL result already has the user-visible names.

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
        alias_renames: Map ``temp_alias -> original_alias``. Filled during
            SQL building with every temp alias used inside an aggregation
            layer.
    """

    sql_ops: List[LazyOp] = field(default_factory=list)
    df_ops: List[LazyOp] = field(default_factory=list)
    groupby_agg: Optional[LazyGroupByAgg] = None
    where_ops: List[LazyOp] = field(default_factory=list)
    layers: List[List[LazyOp]] = field(default_factory=list)
    has_sql_source: bool = False
    first_df_op_idx: Optional[int] = None
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


# Most subquery layers one SQL query nests. ClickHouse parses about ten
# levels per layer against max_parser_depth (1000 by default), and analysis
# time grows faster than linearly with depth, so longer pipelines continue
# in a new SQL segment that runs on the previous segment's result.
MAX_SQL_LAYERS = 16


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
        - A pure-projection SELECT (e.g. ``df[['a','b']]``) drops a
          column computed in the same layer that a WHERE, ORDER BY or
          another computed expression of the layer reads. This split keeps
          the column alive in the inner subquery before the outer layer
          projects it away.
        - An op computes a column that an earlier op in the layer reads
          (WHERE, ORDER BY, computed expression) or computes. Within one
          SELECT, ClickHouse binds every reference to a name to that
          SELECT's alias, so the earlier op would see the new value.
        - A column assignment (``LazyColumnAssignment`` or an
          assignment-style ``SELECT *, expr``) follows a LazyGroupByAgg:
          it computes on the aggregated rows, while in the GROUP BY layer
          its fields would be taken as the aggregation targets.
        - An op computes an expression that reads a column computed in the
          same layer. ClickHouse inlines an alias wherever it is referenced,
          so chains of aliases referencing aliases grow the query tree
          without bound (exponentially when a column is read twice); in a
          new layer the reference is to a real column of the inner layer.

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
        seen_groupby = False  # Has the current layer a LazyGroupByAgg?

        def _is_pure_projection_select(op):
            """Pure projection SELECT = ``SELECT col1, col2, ...`` with no
            ``*`` marker (pandas ``df[[cols]]``). Assignment-style SELECTs
            (``df.assign(c=...)``) carry a ``*`` and stay in the same
            layer so the assignments survive."""
            return (
                isinstance(op, LazyRelationalOp)
                and op.op_type == "SELECT"
                and op.fields
                and not any(is_star_item(f) for f in op.fields)
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

        def _computed_and_read(layer):
            """Columns the layer's ops compute, and columns they read."""
            computed = set()
            read = set()
            for prev in layer:
                computed.update(self._computed_columns_of(prev))
                read |= self._columns_read_by(prev)
            return computed, read

        def _drops_used_computed_column(op_set, layer):
            """Does the layer compute a column outside ``op_set`` that one
            of its ops reads? If yes, the outer projection would drop that
            column before the reader could bind to it, so we must split
            into a new layer. Source columns stay visible to the
            projecting layer."""
            computed, read = _computed_and_read(layer)
            return bool((read & computed) - op_set)

        def _append(op):
            nonlocal seen_limit, seen_offset, seen_groupby
            current_layer.append(op)
            if isinstance(op, LazyGroupByAgg):
                seen_groupby = True
            if isinstance(op, LazyRelationalOp):
                if op.op_type == "LIMIT":
                    seen_limit = True
                elif op.op_type == "OFFSET":
                    seen_offset = True

        def _open_layer_with(op):
            nonlocal current_layer, seen_limit, seen_offset, seen_groupby
            nonlocal computed_columns_after_limit
            layers.append(current_layer)
            current_layer = []
            seen_limit = seen_offset = seen_groupby = False
            computed_columns_after_limit = set()
            _append(op)

        def _shadows_layer_column(op, layer):
            new_columns = self._computed_columns_of(op).keys()
            if not new_columns:
                return False
            computed, read = _computed_and_read(layer)
            if new_columns & (computed | read):
                return True
            return bool(self._columns_read_by(op) & computed)

        def _place(op):
            """Append ``op``, or open a new layer for it when it would
            shadow a column the layer already uses or compute from a column
            the layer computes. Returns whether a layer was opened."""
            if current_layer and _shadows_layer_column(op, current_layer):
                _open_layer_with(op)
                return True
            _append(op)
            return False

        def _flush_pending():
            nonlocal pending_column_assignments
            pending = pending_column_assignments
            pending_column_assignments = []
            opened = False
            for assignment in pending:
                opened = _place(assignment) or opened
            return opened

        def _start_new_layer_with(op):
            """Drop any pending assignments into current_layer, then begin a
            fresh layer containing ``op``. If placing the assignments already
            opened a fresh layer (no LIMIT/OFFSET/WHERE yet), ``op`` joins it."""
            if _flush_pending():
                _place(op)
            else:
                _open_layer_with(op)

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
                elif (
                    seen_groupby
                    and op.op_type == "SELECT"
                    and (
                        not _is_pure_projection_select(op)
                        or self._computed_columns_of(op)
                    )
                ):
                    # Assignment after an aggregation computes on its result,
                    # also when it overwrites a column in an explicit projection
                    needs_new_layer = True
                elif _is_pure_projection_select(op):
                    proj_fields = _projection_field_names(op)
                    if _drops_used_computed_column(proj_fields, current_layer):
                        # Pure projection that drops a column the layer
                        # still reads (e.g. ``assign(x=...)[where x>0][[id]]``).
                        # Split into a new layer so the outer SELECT
                        # projects after the inner layer has used it.
                        needs_new_layer = True

                if needs_new_layer and current_layer:
                    _start_new_layer_with(op)
                else:
                    # Add pending column assignments to current layer first
                    _flush_pending()
                    _place(op)

            elif isinstance(op, LazyColumnAssignment):
                # Column assignment after LIMIT/OFFSET needs special handling
                if seen_groupby:
                    # Assignment after an aggregation computes on its result
                    _start_new_layer_with(op)
                elif seen_limit or seen_offset:
                    # Track the column for dependency checking
                    computed_columns_after_limit.add(op.column)
                    # Queue the assignment to be added before any dependent WHERE/ORDER BY
                    pending_column_assignments.append(op)
                else:
                    # No LIMIT/OFFSET yet, add directly to current layer
                    _place(op)
            elif isinstance(op, LazyGroupByAgg) and (seen_limit or seen_offset):
                # ``head(N).groupby(...).agg(...)`` semantics: pandas
                # aggregates over the first N rows. If LIMIT and GROUP BY
                # live in the same SQL layer we end up emitting
                # ``... GROUP BY ... LIMIT N``, which limits the number of
                # GROUPS instead of the number of input rows. Split into a
                # new layer so the LIMIT applies to the inner subquery
                # (i.e. ``SELECT ... FROM (SELECT ... LIMIT N) GROUP BY``).
                _start_new_layer_with(op)
            elif isinstance(op, LazyGroupByAgg) and any(
                isinstance(prev, (LazyWhere, LazyMask)) for prev in current_layer
            ):
                # ``mask(cond, other).groupby(...).agg(...)`` (or ``where``)
                # semantics: the mask must run BEFORE the GROUP BY so the
                # aggregation sees masked values. In the same SQL layer,
                # the wrapper-layer CASE-WHEN temp-alias machinery
                # (``__tmp_<col>__``) collides with the agg's own select
                # list (the inner SELECT only emits aggregations, not
                # ``__tmp_*__``). Split into a new layer so the mask runs
                # in the inner subquery and the GROUP BY runs on the
                # masked output:
                # ``SELECT ... agg() FROM (SELECT CASE WHEN ...) GROUP BY``.
                _start_new_layer_with(op)
            else:
                # Other operations - add pending column assignments first, then this op
                _flush_pending()
                _place(op)

        # Add any remaining pending column assignments
        _flush_pending()

        if current_layer:
            layers.append(current_layer)

        return layers

    def _extract_referenced_columns(self, expr) -> set:
        """
        Extract all column names referenced in an expression.

        Used to keep an op that overwrites a column out of the SQL layer of
        an earlier op that reads it, and to keep a projection from dropping
        a computed column its layer still reads.

        Args:
            expr: Expression to analyze (can be Expression, ColumnExpr, or other)

        Returns:
            Set of column names referenced in the expression
        """
        from .expressions import ArithmeticExpression
        from .column_expr import ColumnExpr
        from .conditions import Condition
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

            elif isinstance(e, Condition):
                # Comparison, AND/OR/NOT, IS NULL, IN, BETWEEN, LIKE operands
                for operand in ("left", "right", "condition", "expression", "lower", "upper"):
                    _extract(getattr(e, operand, None))

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
        local_source: bool = False,
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
                op,
                effective_schema,
                preceding_ops,
                following_ops,
                local_source=local_source,
            ):
                op_types.append(("sql", op))
            else:
                op_types.append(("pandas", op))

        # Group consecutive operations of the same type into segments.
        # Ordering between column assignments inside a SQL segment is kept
        # by _build_layers, which gives an overwrite its own subquery layer.
        current_type = None
        current_ops = []
        is_first = True

        def save_segment(segment_type, segment_ops):
            nonlocal is_first
            chunks = (
                self._split_by_layer_depth(segment_ops)
                if segment_type == "sql"
                else [segment_ops]
            )
            for chunk in chunks:
                exec_plan.segments.append(
                    self._create_segment(
                        segment_type, chunk, is_first, has_sql_source, schema
                    )
                )
                is_first = False

        for op_type, op in op_types:
            if op_type != current_type:
                # Save previous segment
                if current_ops:
                    save_segment(current_type, current_ops)
                # Start new segment
                current_type = op_type
                current_ops = [op]
            else:
                current_ops.append(op)

        # Save last segment
        if current_ops:
            save_segment(current_type, current_ops)

        self._logger.debug(
            "Execution plan: %d segments (%d SQL, %d Pandas)",
            len(exec_plan.segments),
            exec_plan.sql_segment_count(),
            exec_plan.pandas_segment_count(),
        )

        return exec_plan

    def _split_by_layer_depth(self, ops: List[LazyOp]) -> List[List[LazyOp]]:
        """Split SQL ops into consecutive runs of at most MAX_SQL_LAYERS
        subquery layers each, in layer order."""
        layers = self._build_layers(ops)
        if len(layers) <= MAX_SQL_LAYERS:
            return [ops]
        return [
            [op for layer in layers[i : i + MAX_SQL_LAYERS] for op in layer]
            for i in range(0, len(layers), MAX_SQL_LAYERS)
        ]

    @staticmethod
    def _computed_columns_of(op: LazyOp) -> Dict[str, Any]:
        """Columns an op computes in SQL, mapped to their expressions.

        Covers ``ds[col] = expr`` (``LazyColumnAssignment``) and the aliased
        expressions of a SELECT (``assign()`` and similar).
        """
        if isinstance(op, LazyColumnAssignment):
            return {op.column: op.expr}
        if isinstance(op, LazyRelationalOp) and op.op_type == "SELECT" and op.fields:
            return {
                f.alias: f
                for f in op.fields
                if not isinstance(f, str) and getattr(f, "alias", None)
            }
        return {}

    def _columns_read_by(self, op: LazyOp) -> Set[str]:
        """Columns an op's condition, sort keys or computed expressions read.

        Plain columns of a projection are not included: the projection is
        rebound in place when one of them is overwritten.
        """
        if isinstance(op, LazyRelationalOp):
            if op.op_type == "WHERE":
                return self._extract_referenced_columns(op.condition)
            if op.op_type == "ORDER BY":
                return {
                    projection_name(f) for f in op.fields or [] if projection_name(f)
                }
        if isinstance(op, (LazyWhere, LazyMask)):
            return self._extract_referenced_columns(op.condition)
        columns = set()
        for expr in self._computed_columns_of(op).values():
            columns |= self._extract_referenced_columns(expr)
        return columns

    def _can_push_op_to_sql(
        self,
        op: LazyOp,
        schema: Dict[str, str] = None,
        preceding_ops: List[LazyOp] = None,
        following_ops: List[LazyOp] = None,
        local_source: bool = False,
    ) -> bool:
        """
        Check if a single operation can be pushed to SQL.

        Args:
            op: The operation to check
            schema: Column schema for type-aware checking
            preceding_ops: List of operations that come before this one (for context-aware decisions)
            following_ops: List of operations that come after this one (for ORDER BY cost awareness)
            local_source: True when the SQL source is an in-process PythonTableFunction
                (from_df). For local sources there is no network/serialization cost,
                so unbounded ORDER BY can be pushed down profitably.

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
                #
                # Exception: in-process PythonTableFunction sources (from_df) have
                # no network/serialization cost, and chDB's parallel ORDER BY is
                # consistently faster than pandas single-threaded sort_values on
                # large frames, so unbounded ORDER BY is pushed down for them.
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
                if has_limit:
                    return True
                if local_source:
                    self._logger.debug(
                        "  [ORDER BY] Pushing to SQL: local PythonTableFunction source"
                    )
                    return True
                self._logger.debug(
                    "  [ORDER BY] Skipping push to SQL: no LIMIT follows (unbounded sort)"
                )
                return False

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

                # Handle special ops
                for op in ops:
                    if isinstance(op, LazyGroupByAgg):
                        plan.groupby_agg = op
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
