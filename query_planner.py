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
- SQLBuilder: Generates SQL from QueryPlans (for SQL-pushable operations)
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
    LazySQLQuery,
    LazyWhere,
    LazyMask,
)
from .expressions import Field, Expression
from .config import get_logger
import pandas as pd

if TYPE_CHECKING:
    from .core import DataStore


class CaseWhenExpr(Expression):
    """
    SQL CASE WHEN expression for LazyWhere/LazyMask pushdown.

    Generates: CASE WHEN cond THEN col ELSE other END AS col

    For mask (opposite of where):
    CASE WHEN NOT(cond) THEN col ELSE other END AS col

    Type handling:
    - For numeric columns: uses the literal other value
    - For string columns with numeric other: uses Variant type to preserve mixed types
    - NULL is used as a safe fallback for type mismatches

    The Variant type approach allows SQL to return mixed types (like pandas object dtype),
    ensuring type consistency between SQL and pandas execution paths.
    """

    def __init__(self, column: str, where_ops: List, quote_char: str = '"', col_type: str = None, alias: str = None):
        """
        Args:
            column: Column name to transform
            where_ops: List of LazyWhere/LazyMask operations
            quote_char: Quote character for identifiers
            col_type: Column type (for type-aware other value formatting)
            alias: Optional output alias (defaults to column name)
        """
        self.column = column
        self.where_ops = where_ops
        self.quote_char = quote_char
        self.col_type = col_type or 'Unknown'
        self.alias = alias or column  # Default to column name if not specified

    def _is_numeric_type(self) -> bool:
        """Check if column is a numeric type."""
        col_type_lower = self.col_type.lower()
        numeric_types = ('int', 'float', 'double', 'decimal', 'uint', 'number')
        return any(t in col_type_lower for t in numeric_types)

    def _is_string_type(self) -> bool:
        """Check if column is a string type."""
        col_type_lower = self.col_type.lower()
        string_types = ('string', 'fixedstring', 'enum', 'uuid')
        return any(t in col_type_lower for t in string_types)

    def _is_date_type(self) -> bool:
        """Check if column is a date/datetime type."""
        col_type_lower = self.col_type.lower()
        date_types = ('date', 'datetime', 'datetime64')
        return any(t in col_type_lower for t in date_types)

    def _is_bool_type(self) -> bool:
        """Check if column is a boolean type."""
        col_type_lower = self.col_type.lower()
        return 'bool' in col_type_lower

    def _needs_variant_type(self, other) -> bool:
        """
        Check if we need to use Variant type to preserve mixed types.

        This is needed when:
        - Column is string type AND
        - other is numeric (int/float, not None/NaN)

        Without Variant, ClickHouse would convert numeric to string,
        but pandas keeps mixed types in object dtype.
        """
        if other is None or (isinstance(other, float) and pd.isna(other)):
            return False
        return self._is_string_type() and isinstance(other, (int, float))

    def _get_variant_type(self, other) -> str:
        """Get the Variant type string for mixed type scenarios."""
        if isinstance(other, float):
            return "Variant(String, Float64)"
        else:
            return "Variant(String, Int64)"

    def _format_other_value(self, other, use_variant: bool = False) -> str:
        """
        Format the 'other' value for SQL, considering column type.

        Returns SQL-safe representation of the value.
        For incompatible types (e.g., numeric 'other' for Date columns), returns NULL.

        Args:
            other: The replacement value
            use_variant: If True, wrap in Variant type cast
        """
        # None or NaN -> NULL
        if other is None or (isinstance(other, float) and pd.isna(other)):
            return "NULL"

        # Date/DateTime columns: numeric 'other' is incompatible, use NULL
        # This handles cases like df.where(cond, 0) where df has date columns
        if self._is_date_type() and isinstance(other, (int, float)):
            return "NULL"

        # Boolean columns: preserve boolean semantics
        if self._is_bool_type() and isinstance(other, (int, float)):
            # 0 -> false, non-0 -> true, but safest is NULL for type safety
            if other == 0:
                return "false"
            elif other == 1:
                return "true"
            else:
                return "NULL"

        # String value
        if isinstance(other, str):
            base_val = f"'{other}'"
            if use_variant:
                return f"{base_val}::{self._get_variant_type(other)}"
            return base_val

        # Numeric value
        if isinstance(other, (int, float)):
            base_val = str(other)
            if use_variant:
                # Use Variant to preserve numeric type for string columns
                return f"{base_val}::{self._get_variant_type(other)}"
            elif self._is_string_type():
                # Legacy behavior: convert to string if not using Variant
                return f"'{other}'"
            else:
                return base_val

        # Boolean
        if isinstance(other, bool):
            return "1" if other else "0"

        # Default: try string conversion
        return f"'{other}'"

    def to_sql(self, quote_char: str = None, **kwargs) -> str:
        """Generate SQL CASE WHEN expression."""
        from .column_expr import ColumnExpr
        from .conditions import Condition

        qc = quote_char or self.quote_char
        col_quoted = f"{qc}{self.column}{qc}"

        # Check if any operation needs Variant type
        use_variant = any(self._needs_variant_type(op.other) for op in self.where_ops)

        # Start with the column itself, optionally cast to Variant
        if use_variant:
            # Get the Variant type from the first numeric other
            variant_type = None
            for op in self.where_ops:
                if self._needs_variant_type(op.other):
                    variant_type = self._get_variant_type(op.other)
                    break
            current_expr = f"{col_quoted}::{variant_type}"
        else:
            current_expr = col_quoted

        # Apply each where/mask operation in order
        for op in self.where_ops:
            # Get condition SQL
            cond = op.condition
            if isinstance(cond, ColumnExpr):
                cond = cond._expr if hasattr(cond, '_expr') else cond

            if isinstance(cond, Condition):
                cond_sql = cond.to_sql(quote_char=qc)
            else:
                raise ValueError(f"Cannot convert condition to SQL: {type(cond)}")

            # For mask, invert the condition
            if op._is_mask:
                cond_sql = f"NOT ({cond_sql})"

            # Format other value with type awareness
            other_sql = self._format_other_value(op.other, use_variant=use_variant)

            # Build CASE WHEN
            current_expr = f"CASE WHEN {cond_sql} THEN {current_expr} ELSE {other_sql} END"

        # Use alias (may be different from column name to avoid ClickHouse conflicts)
        alias_quoted = f"{qc}{self.alias}{qc}"
        return f"{current_expr} AS {alias_quoted}"

    def __repr__(self) -> str:
        return f"CaseWhenExpr({self.column}, {len(self.where_ops)} ops, type={self.col_type})"


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
                    agg_aliases = self._get_agg_aliases(op)
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

    def _get_agg_aliases(self, op: LazyGroupByAgg) -> Set[str]:
        """
        Get alias names that would be created by a GroupByAgg operation.

        Args:
            op: LazyGroupByAgg operation

        Returns:
            Set of alias names
        """
        agg_aliases = set()

        if not op.agg_dict:
            return agg_aliases

        for col, funcs in op.agg_dict.items():
            if isinstance(funcs, (list, tuple)):
                agg_aliases.update(funcs)  # Function names as aliases
            else:
                agg_aliases.add(col)  # Column name as alias for single func

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


class SQLBuilder:
    """
    SQL builder that generates SQL strings from QueryPlans.

    This class handles:
    - Building SELECT/WHERE/ORDER BY/LIMIT/OFFSET clauses
    - Nested subqueries for complex patterns
    - GroupBy SQL pushdown with aggregations
    - Proper quoting of identifiers
    """

    def __init__(self, quote_char: str = '"'):
        self.quote_char = quote_char
        self._logger = get_logger()

    def build_sql(
        self,
        plan: QueryPlan,
        table_source: str,
        select_fields: List[Expression] = None,
        groupby_fields: List[Expression] = None,
        having_condition: Any = None,
        joins: List[Tuple] = None,
        distinct: bool = False,
        all_columns: List[str] = None,
    ) -> str:
        """
        Build SQL query from a QueryPlan.

        Args:
            plan: Query plan with SQL operations
            table_source: SQL table source (table name or table function)
            select_fields: Fields to select (from DataStore state)
            groupby_fields: GroupBy fields (from DataStore state)
            having_condition: HAVING condition (from DataStore state)
            joins: JOIN clauses (from DataStore state)
            distinct: Whether to use DISTINCT
            all_columns: All column names (needed for LazyWhere CASE WHEN generation)

        Returns:
            SQL query string
        """
        if plan.needs_nested_subqueries():
            return self._build_nested_sql(
                plan, table_source, select_fields, groupby_fields, having_condition, joins, distinct, all_columns
            )
        else:
            return self._build_simple_sql(
                plan, table_source, select_fields, groupby_fields, having_condition, joins, distinct, all_columns
            )

    def _build_simple_sql(
        self,
        plan: QueryPlan,
        table_source: str,
        select_fields: List[Expression],
        groupby_fields: List[Expression],
        having_condition: Any,
        joins: List[Tuple],
        distinct: bool,
        all_columns: List[str] = None,
    ) -> str:
        """Build a simple (non-nested) SQL query."""
        # Extract clauses from operations
        where_conditions = []
        orderby_fields = []
        limit_value = None
        offset_value = None

        ops = plan.layers[0] if plan.layers else []

        for op in ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'WHERE' and op.condition is not None:
                    where_conditions.append(op.condition)
                elif op.op_type == 'ORDER BY' and op.fields:
                    # Later ORDER BY replaces earlier ones (pandas semantics)
                    orderby_fields = []
                    for f in op.fields:
                        if isinstance(f, str):
                            orderby_fields.append((Field(f), op.ascending))
                        else:
                            orderby_fields.append((f, op.ascending))
                elif op.op_type == 'LIMIT':
                    limit_value = op.limit_value
                elif op.op_type == 'OFFSET':
                    offset_value = op.offset_value

        # Handle GroupBy pushdown
        final_select_fields = select_fields or []
        final_groupby_fields = groupby_fields or []

        if plan.groupby_agg:
            final_groupby_fields, final_select_fields = self._build_groupby_select(
                plan.groupby_agg, final_groupby_fields, final_select_fields
            )

        # Handle LazyWhere/LazyMask pushdown (CASE WHEN)
        if plan.where_ops and all_columns:
            final_select_fields = self._build_where_select(plan.where_ops, all_columns)

        # Build SQL using the existing method pattern
        # This delegates to DataStore._build_sql_from_state (we return the parts)
        return self._assemble_sql(
            table_source,
            final_select_fields,
            where_conditions,
            orderby_fields,
            limit_value,
            offset_value,
            joins,
            distinct,
            final_groupby_fields,
            having_condition,
        )

    def _build_nested_sql(
        self,
        plan: QueryPlan,
        table_source: str,
        select_fields: List[Expression],
        groupby_fields: List[Expression],
        having_condition: Any,
        joins: List[Tuple],
        distinct: bool,
        all_columns: List[str] = None,
    ) -> str:
        """Build nested subquery SQL for complex patterns."""
        # Build innermost query (layer 0)
        inner_where = []
        inner_orderby = []
        inner_orderby_kind = 'quicksort'
        inner_limit = None
        inner_offset = None

        for op in plan.layers[0]:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'WHERE' and op.condition is not None:
                    inner_where.append(op.condition)
                elif op.op_type == 'ORDER BY' and op.fields:
                    inner_orderby = []
                    inner_orderby_kind = getattr(op, 'kind', 'quicksort')
                    for f in op.fields:
                        if isinstance(f, str):
                            inner_orderby.append((Field(f), op.ascending))
                        else:
                            inner_orderby.append((f, op.ascending))
                elif op.op_type == 'LIMIT':
                    inner_limit = op.limit_value
                elif op.op_type == 'OFFSET':
                    inner_offset = op.offset_value

        # Handle LazyWhere/LazyMask pushdown (CASE WHEN)
        final_select_fields = select_fields
        if plan.where_ops and all_columns:
            final_select_fields = self._build_where_select(plan.where_ops, all_columns)

        sql = self._assemble_sql(
            table_source,
            final_select_fields,
            inner_where,
            inner_orderby,
            inner_limit,
            inner_offset,
            joins,
            distinct,
            groupby_fields,
            having_condition,
            inner_orderby_kind,
        )

        # Wrap with outer layers
        for layer_idx, layer_ops in enumerate(plan.layers[1:], 1):
            sql = self._wrap_with_layer(sql, layer_ops, layer_idx)

        return sql

    def _wrap_with_layer(self, inner_sql: str, layer_ops: List[LazyOp], layer_idx: int) -> str:
        """Wrap an inner SQL query with an outer layer."""
        layer_where = []
        layer_orderby = []
        layer_orderby_kind = 'quicksort'
        layer_limit = None
        layer_offset = None

        for op in layer_ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'WHERE' and op.condition is not None:
                    layer_where.append(op.condition)
                elif op.op_type == 'ORDER BY' and op.fields:
                    layer_orderby = []
                    layer_orderby_kind = getattr(op, 'kind', 'quicksort')
                    for f in op.fields:
                        if isinstance(f, str):
                            layer_orderby.append((Field(f), op.ascending))
                        else:
                            layer_orderby.append((f, op.ascending))
                elif op.op_type == 'LIMIT':
                    layer_limit = op.limit_value
                elif op.op_type == 'OFFSET':
                    layer_offset = op.offset_value

        # Build outer query
        outer_parts = ["SELECT *"]
        outer_parts.append(f"FROM ({inner_sql}) AS __subq{layer_idx}__")

        if layer_where:
            combined = layer_where[0]
            for cond in layer_where[1:]:
                combined = combined & cond
            outer_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        if layer_orderby:
            from .utils import build_orderby_clause, is_stable_sort

            orderby_sql = build_orderby_clause(
                layer_orderby, self.quote_char, stable=is_stable_sort(layer_orderby_kind)
            )
            outer_parts.append(f"ORDER BY {orderby_sql}")

        if layer_limit is not None:
            outer_parts.append(f"LIMIT {layer_limit}")

        if layer_offset is not None:
            outer_parts.append(f"OFFSET {layer_offset}")

        return ' '.join(outer_parts)

    def _build_groupby_select(
        self, groupby_agg: LazyGroupByAgg, groupby_fields: List[Expression], select_fields: List[Expression]
    ) -> Tuple[List[Expression], List[Expression]]:
        """
        Build SELECT and GROUP BY fields for GroupBy pushdown.

        Returns:
            Tuple of (groupby_fields, select_fields)
        """
        from .functions import AggregateFunction
        from .expressions import Star

        # Build GROUP BY fields
        final_groupby = [Field(col) for col in groupby_agg.groupby_cols]

        # Build SELECT fields with aggregations
        final_select = list(final_groupby)  # Include group keys

        if groupby_agg.agg_dict:
            # Pandas-style: agg({'col': 'func'}) or agg({'col': ['func1', 'func2']})
            has_multi_col = len(groupby_agg.agg_dict) > 1
            has_any_multi_func = any(isinstance(f, (list, tuple)) for f in groupby_agg.agg_dict.values())

            # Check for function name conflicts across columns
            all_funcs = []
            for col, funcs in groupby_agg.agg_dict.items():
                if isinstance(funcs, str):
                    all_funcs.append(funcs)
                else:
                    all_funcs.extend(funcs)
            has_func_conflict = len(all_funcs) != len(set(all_funcs))

            use_compound_alias = has_multi_col and (has_any_multi_func or has_func_conflict)

            for col, funcs in groupby_agg.agg_dict.items():
                is_multi_func = isinstance(funcs, (list, tuple))
                if isinstance(funcs, str):
                    funcs = [funcs]

                for func in funcs:
                    sql_func = self._map_agg_func(func)

                    if use_compound_alias:
                        alias = f"{col}_{func}"
                    elif is_multi_func:
                        alias = func
                    else:
                        alias = col

                    agg_expr = AggregateFunction(sql_func, Field(col), alias=alias)
                    final_select.append(agg_expr)

        elif groupby_agg.agg_func:
            # Single function for all columns
            func = groupby_agg.agg_func
            sql_func = self._map_agg_func(func)

            if func in ('count', 'size'):
                final_select.append(AggregateFunction(sql_func, Star()))

        return final_groupby, final_select

    def _map_agg_func(self, func: str) -> str:
        """Map pandas aggregation function name to SQL."""
        sql_func_map = {
            'sum': 'sum',
            'mean': 'avg',
            'avg': 'avg',
            'count': 'count',
            'min': 'min',
            'max': 'max',
            'std': 'stddevPop',
            'var': 'varPop',
            'first': 'any',
            'last': 'anyLast',
            'size': 'count',
        }
        return sql_func_map.get(func, func)

    def _build_where_select(
        self, where_ops: List[LazyOp], all_columns: List[str], schema: Dict[str, str] = None
    ) -> List['CaseWhenExpr']:
        """
        Build SELECT fields with CASE WHEN for LazyWhere/LazyMask operations.

        For each column, generates:
        - where: CASE WHEN cond THEN col ELSE other END AS col
        - mask:  CASE WHEN NOT(cond) THEN col ELSE other END AS col

        Multiple where_ops are chained: the output of one becomes input to next.

        Args:
            where_ops: List of LazyWhere/LazyMask operations
            all_columns: List of all column names
            schema: Optional dict mapping column names to types

        Returns:
            List of CaseWhenExpr for each column
        """
        schema = schema or {}

        # Build CASE WHEN expressions for each column
        result = []
        for col in all_columns:
            col_type = schema.get(col, 'Unknown')
            case_expr = CaseWhenExpr(col, where_ops, self.quote_char, col_type)
            result.append(case_expr)
        return result

    def _assemble_sql(
        self,
        table_source: str,
        select_fields: List[Expression],
        where_conditions: List,
        orderby_fields: List[Tuple],
        limit_value: Optional[int],
        offset_value: Optional[int],
        joins: List[Tuple],
        distinct: bool,
        groupby_fields: List[Expression],
        having_condition: Any,
        orderby_kind: str = 'quicksort',
    ) -> str:
        """
        Assemble SQL string from components.

        This is a simplified version that handles basic SQL assembly.
        Complex cases may still use DataStore._build_sql_from_state.
        """
        parts = []

        # SELECT clause
        distinct_str = "DISTINCT " if distinct else ""
        if select_fields:
            select_str = ', '.join(f.to_sql(quote_char=self.quote_char) for f in select_fields)
            parts.append(f"SELECT {distinct_str}{select_str}")
        else:
            parts.append(f"SELECT {distinct_str}*")

        # FROM clause
        parts.append(f"FROM {table_source}")

        # JOIN clauses
        if joins:
            for join_table, join_type, on_condition in joins:
                join_sql = self._build_join_clause(join_table, join_type, on_condition)
                parts.append(join_sql)

        # WHERE clause
        if where_conditions:
            combined = where_conditions[0]
            for cond in where_conditions[1:]:
                combined = combined & cond
            parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # GROUP BY clause
        if groupby_fields:
            groupby_str = ', '.join(f.to_sql(quote_char=self.quote_char) for f in groupby_fields)
            parts.append(f"GROUP BY {groupby_str}")

        # HAVING clause
        if having_condition:
            parts.append(f"HAVING {having_condition.to_sql(quote_char=self.quote_char)}")

        # ORDER BY clause (stable sort if kind='stable' or 'mergesort', matching pandas)
        if orderby_fields:
            from .utils import build_orderby_clause, is_stable_sort

            orderby_sql = build_orderby_clause(orderby_fields, self.quote_char, stable=is_stable_sort(orderby_kind))
            parts.append(f"ORDER BY {orderby_sql}")

        # LIMIT clause
        if limit_value is not None:
            parts.append(f"LIMIT {limit_value}")

        # OFFSET clause
        if offset_value is not None:
            parts.append(f"OFFSET {offset_value}")

        return '\n'.join(parts)

    def _build_join_clause(self, join_table, join_type: str, on_condition) -> str:
        """Build a JOIN clause string."""
        # Handle DataStore as join table
        if hasattr(join_table, 'to_sql'):
            table_sql = f"({join_table.to_sql()}) AS {join_table._alias or '__join__'}"
        else:
            table_sql = f"{self.quote_char}{join_table}{self.quote_char}"

        if on_condition:
            on_sql = on_condition.to_sql(quote_char=self.quote_char)
            return f"{join_type} JOIN {table_sql} ON {on_sql}"
        else:
            return f"{join_type} JOIN {table_sql}"
