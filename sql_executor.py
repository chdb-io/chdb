"""
SQL Execution Engine for DataStore.

This module encapsulates the SQL building and execution logic, extracting common
patterns from core.py to reduce code duplication and improve maintainability.

Key Classes:
- ExtractedClauses: Dataclass holding SQL clauses extracted from LazyOps
- SQLExecutionEngine: Main class for executing lazy operations via SQL

Design Goals:
1. Eliminate repeated LazyRelationalOp parsing code (6+ occurrences in core.py)
2. Centralize ascending_list normalization
3. Provide a clean interface for SQL building and execution
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
import pandas as pd

from .expressions import Field, Expression, Star
from .lazy_ops import LazyOp, LazyRelationalOp, LazyGroupByAgg
from .utils import (
    normalize_ascending,
    map_agg_func,
    build_orderby_clause,
    is_stable_sort,
    format_identifier,
)
from .config import get_logger

if TYPE_CHECKING:
    from .core import DataStore
    from .query_planner import QueryPlan


@dataclass
class ExtractedClauses:
    """
    Holds SQL clauses extracted from a list of LazyRelationalOps.

    This eliminates the repeated pattern of iterating through ops and
    checking op_type == 'WHERE', 'ORDER BY', 'LIMIT', 'OFFSET'.
    """

    where_conditions: List[Any] = field(default_factory=list)
    orderby_fields: List[Tuple[Any, bool]] = field(default_factory=list)
    orderby_kind: str = 'quicksort'
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None
    select_fields: List[Expression] = field(default_factory=list)

    def has_orderby(self) -> bool:
        """Check if ORDER BY is present."""
        return len(self.orderby_fields) > 0

    def has_where(self) -> bool:
        """Check if WHERE is present."""
        return len(self.where_conditions) > 0

    def needs_row_order(self, has_groupby: bool = False) -> bool:
        """
        Check if we need rowNumberInAllBlocks() for row order preservation.

        chDB may return rows in different order due to parallel processing.
        We need ORDER BY rowNumberInAllBlocks() when:
        - No explicit ORDER BY is specified
        - No GROUP BY aggregation (which has its own ordering)
        """
        return not self.has_orderby() and not has_groupby


def extract_clauses_from_ops(ops: List[LazyOp], quote_char: str = '"') -> ExtractedClauses:
    """
    Extract SQL clauses from a list of LazyOps.

    This is the unified replacement for the repeated pattern:
        for op in ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'WHERE' and op.condition is not None:
                    ...
                elif op.op_type == 'ORDER BY' and op.fields:
                    ...
                elif op.op_type == 'LIMIT':
                    ...
                elif op.op_type == 'OFFSET':
                    ...

    Args:
        ops: List of LazyOps to process
        quote_char: Quote character for identifiers

    Returns:
        ExtractedClauses with all SQL components
    """
    result = ExtractedClauses()

    for op in ops:
        if not isinstance(op, LazyRelationalOp):
            continue

        if op.op_type == 'WHERE' and op.condition is not None:
            result.where_conditions.append(op.condition)

        elif op.op_type == 'ORDER BY' and op.fields:
            # Later ORDER BY replaces earlier ones (pandas semantics)
            result.orderby_fields = []
            result.orderby_kind = getattr(op, 'kind', 'quicksort')

            ascending_list = normalize_ascending(op.ascending, len(op.fields))
            for i, f in enumerate(op.fields):
                asc = ascending_list[i] if i < len(ascending_list) else True
                if isinstance(f, str):
                    result.orderby_fields.append((Field(f), asc))
                else:
                    result.orderby_fields.append((f, asc))

        elif op.op_type == 'LIMIT':
            result.limit_value = op.limit_value

        elif op.op_type == 'OFFSET':
            result.offset_value = op.offset_value

        elif op.op_type == 'SELECT' and op.fields:
            for f in op.fields:
                if isinstance(f, str):
                    if f != '*':
                        result.select_fields.append(Field(f))
                else:
                    result.select_fields.append(f)

    return result


def apply_alias_renames_to_orderby(
    orderby_fields: List[Tuple[Any, bool]], alias_renames: Dict[str, str]
) -> List[Tuple[Any, bool]]:
    """
    Handle ORDER BY with alias conflicts.

    If ORDER BY references a column that has a temp alias (due to GroupBy conflict),
    we need to use the temp alias in ORDER BY.

    Args:
        orderby_fields: List of (field, ascending) tuples
        alias_renames: Dict mapping temp_alias -> original_alias

    Returns:
        Updated orderby_fields with temp aliases applied
    """
    if not alias_renames or not orderby_fields:
        return orderby_fields

    # Build reverse mapping: original_alias -> temp_alias
    reverse_renames = {orig: temp for temp, orig in alias_renames.items()}

    new_orderby = []
    for field_obj, asc in orderby_fields:
        field_name = field_obj.name if isinstance(field_obj, Field) else str(field_obj)
        if field_name in reverse_renames:
            # Replace with temp alias
            new_orderby.append((Field(reverse_renames[field_name]), asc))
        else:
            new_orderby.append((field_obj, asc))

    return new_orderby


def build_groupby_select_fields(
    groupby_agg: LazyGroupByAgg, alias_renames: Dict[str, str] = None
) -> Tuple[List[Field], List[Expression]]:
    """
    Build GROUP BY and SELECT fields from a LazyGroupByAgg operation.

    This centralizes the aggregation SQL building logic.

    Args:
        groupby_agg: LazyGroupByAgg operation
        alias_renames: Dict for alias conflict resolution

    Returns:
        Tuple of (groupby_fields, select_fields)
    """
    from .functions import AggregateFunction

    alias_renames = alias_renames or {}

    # Build GROUP BY fields
    groupby_fields = [Field(col) for col in groupby_agg.groupby_cols]

    # Build SELECT fields with aggregations
    select_fields = list(groupby_fields)  # Include group keys

    if groupby_agg.agg_dict:
        # Pandas-style: agg({'col': 'func'}) or agg({'col': ['func1', 'func2']})
        # Determine if we need compound aliases (col_func) to avoid duplicates
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
                sql_func = map_agg_func(func)

                # Alias strategy
                if use_compound_alias:
                    alias = f"{col}_{func}"
                elif is_multi_func:
                    alias = func
                else:
                    alias = col

                # Check if this alias conflicts with WHERE columns
                temp_alias = f"__agg_{alias}__"
                if temp_alias in alias_renames:
                    alias = temp_alias

                agg_expr = AggregateFunction(sql_func, Field(col), alias=alias)
                select_fields.append(agg_expr)

    elif groupby_agg.agg_func:
        # Single function for all numeric columns
        func = groupby_agg.agg_func
        sql_func = map_agg_func(func)

        if func in ('count', 'size'):
            # COUNT(*) for counting rows
            select_fields.append(AggregateFunction(sql_func, Star()))

    return groupby_fields, select_fields


@dataclass
class SQLBuildResult:
    """
    Result of SQL building from a QueryPlan.

    Encapsulates both the SQL string and metadata needed for post-processing
    (like column renames and index setting for GroupBy).
    """

    sql: str
    alias_renames: Dict[str, str] = field(default_factory=dict)
    groupby_agg: Optional[LazyGroupByAgg] = None


class SQLExecutionEngine:
    """
    Engine for building and executing SQL from lazy operations.

    This class encapsulates the complex SQL building logic that was previously
    duplicated in _execute() and _build_execution_sql() methods.
    """

    def __init__(self, datastore: 'DataStore'):
        self.ds = datastore
        self.quote_char = datastore.quote_char
        self._logger = get_logger()

    def get_table_source(self) -> str:
        """Get the SQL table source (table function or table name)."""
        if self.ds._table_function:
            return self.ds._table_function.to_sql()
        elif self.ds.table_name:
            return format_identifier(self.ds.table_name, self.quote_char)
        return ""

    def build_simple_sql(
        self,
        clauses: ExtractedClauses,
        select_fields: List[Expression] = None,
        groupby_fields: List[Expression] = None,
        having_condition: Any = None,
        distinct: bool = False,
    ) -> str:
        """
        Build a simple (non-nested) SQL query.

        Args:
            clauses: Extracted SQL clauses
            select_fields: Fields to select
            groupby_fields: GROUP BY fields
            having_condition: HAVING condition
            distinct: Whether to use DISTINCT

        Returns:
            SQL query string
        """
        parts = []

        # SELECT (with optional DISTINCT)
        distinct_keyword = 'DISTINCT ' if distinct else ''
        select_fields = select_fields or clauses.select_fields

        if select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            if self.ds._select_star:
                parts.append(f"SELECT {distinct_keyword}*, {fields_sql}")
            else:
                parts.append(f"SELECT {distinct_keyword}{fields_sql}")
        else:
            parts.append(f"SELECT {distinct_keyword}*")

        # FROM
        table_source = self.get_table_source()
        if table_source:
            # Add alias when joins are present
            if self.ds._joins:
                alias = self.ds._get_table_alias()
                parts.append(f"FROM {table_source} AS {format_identifier(alias, self.quote_char)}")
            else:
                parts.append(f"FROM {table_source}")

        # JOIN clauses
        if self.ds._joins:
            for other_ds, join_type, join_condition in self.ds._joins:
                parts.append(self._build_join_clause(other_ds, join_type, join_condition))

        # WHERE
        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # GROUP BY
        if groupby_fields:
            groupby_sql = ', '.join(f.to_sql(quote_char=self.quote_char) for f in groupby_fields)
            parts.append(f"GROUP BY {groupby_sql}")

        # HAVING
        if having_condition:
            parts.append(f"HAVING {having_condition.to_sql(quote_char=self.quote_char)}")

        # ORDER BY
        if clauses.orderby_fields:
            orderby_sql = build_orderby_clause(
                clauses.orderby_fields, self.quote_char, stable=is_stable_sort(clauses.orderby_kind)
            )
            parts.append(f"ORDER BY {orderby_sql}")

        # LIMIT
        if clauses.limit_value is not None:
            parts.append(f"LIMIT {clauses.limit_value}")

        # OFFSET
        if clauses.offset_value is not None:
            parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(parts)

    def build_nested_sql(self, layers: List[List[LazyOp]], base_clauses: ExtractedClauses = None) -> str:
        """
        Build nested subquery SQL for complex patterns.

        Multiple layers are needed when:
        - WHERE follows LIMIT/OFFSET (pandas: slice then filter)
        - ORDER BY follows LIMIT/OFFSET (pandas: slice then sort)

        Args:
            layers: List of operation lists, innermost first
            base_clauses: Clauses for the innermost query

        Returns:
            SQL query string with nested subqueries
        """
        if not layers:
            return ""

        # Build innermost query (layer 0)
        inner_clauses = extract_clauses_from_ops(layers[0], self.quote_char)

        sql = self.build_simple_sql(
            inner_clauses,
            select_fields=inner_clauses.select_fields or None,
            groupby_fields=self.ds._groupby_fields or None,
            having_condition=self.ds._having_condition,
            distinct=self.ds._distinct,
        )

        # Wrap with outer layers (layer 1, 2, ...)
        for layer_idx, layer_ops in enumerate(layers[1:], 1):
            layer_clauses = extract_clauses_from_ops(layer_ops, self.quote_char)
            sql = self._wrap_with_layer(sql, layer_clauses, layer_idx)

        return sql

    def _wrap_with_layer(self, inner_sql: str, clauses: ExtractedClauses, layer_idx: int) -> str:
        """Wrap an inner SQL query with an outer layer."""
        outer_parts = ["SELECT *"]
        outer_parts.append(f"FROM ({inner_sql}) AS __subq{layer_idx}__")

        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            outer_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        if clauses.orderby_fields:
            orderby_sql = build_orderby_clause(
                clauses.orderby_fields, self.quote_char, stable=is_stable_sort(clauses.orderby_kind)
            )
            outer_parts.append(f"ORDER BY {orderby_sql}")

        if clauses.limit_value is not None:
            outer_parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            outer_parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(outer_parts)

    def _build_join_clause(self, other_ds: 'DataStore', join_type, join_condition) -> str:
        """Build a JOIN clause."""
        from .enums import JoinType

        # Generate JOIN keyword
        join_keyword = join_type.value if hasattr(join_type, 'value') and join_type.value else ''
        if join_keyword:
            join_clause = f"{join_keyword} JOIN"
        else:
            join_clause = "JOIN"

        # Handle subquery joins
        if hasattr(other_ds, '_is_subquery') and other_ds._is_subquery:
            other_table = other_ds.to_sql(quote_char=self.quote_char, as_subquery=True)
        elif hasattr(other_ds, '_table_function') and other_ds._table_function:
            table_func_sql = other_ds._table_function.to_sql(quote_char=self.quote_char)
            alias = other_ds._get_table_alias()
            other_table = f"{table_func_sql} AS {format_identifier(alias, self.quote_char)}"
        else:
            other_table = format_identifier(other_ds.table_name, self.quote_char)

        # Handle USING vs ON syntax
        if isinstance(join_condition, tuple) and join_condition[0] == 'USING':
            columns = join_condition[1]
            using_cols = ', '.join(format_identifier(c, self.quote_char) for c in columns)
            return f"{join_clause} {other_table} USING ({using_cols})"
        else:
            condition_sql = join_condition.to_sql(quote_char=self.quote_char)
            return f"{join_clause} {other_table} ON {condition_sql}"

    def build_sql_with_row_order_subquery(
        self, clauses: ExtractedClauses, select_fields: List[Expression] = None, groupby_fields: List[Expression] = None
    ) -> str:
        """
        Build SQL with subquery for row order preservation.

        When we have WHERE conditions and need to preserve row order,
        rowNumberInAllBlocks() must be computed BEFORE filtering.

        Args:
            clauses: Extracted SQL clauses
            select_fields: Fields to select
            groupby_fields: GROUP BY fields

        Returns:
            SQL query string with row order subquery
        """
        table_source = self.get_table_source()

        # Build base query with rowNumberInAllBlocks()
        base_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_source}"

        # Build outer query with WHERE and ORDER BY __orig_row_num__
        sql_parts = []
        if select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            sql_parts.append(f"SELECT {fields_sql}")
        else:
            sql_parts.append("SELECT * EXCEPT(__orig_row_num__)")

        sql_parts.append(f"FROM ({base_sql}) AS __row_num_subq__")

        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            sql_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        if groupby_fields:
            groupby_sql = ', '.join(f.to_sql(quote_char=self.quote_char) for f in groupby_fields)
            sql_parts.append(f"GROUP BY {groupby_sql}")

        sql_parts.append("ORDER BY __orig_row_num__")

        if clauses.limit_value is not None:
            sql_parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            sql_parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(sql_parts)

    def build_sql_with_case_when_subquery(
        self,
        clauses: ExtractedClauses,
        case_when_select: List[Expression],
        temp_alias_columns: List[Tuple[str, str]],
    ) -> str:
        """
        Build SQL with CASE WHEN subquery for LazyWhere/LazyMask.

        ClickHouse has alias conflict quirks, so we use TWO-LAYER subquery:
        1. Inner: WHERE conditions (filters data)
        2. Middle: CASE WHEN with temp aliases
        3. Outer: Rename temp aliases back to original

        Args:
            clauses: Extracted SQL clauses
            case_when_select: CASE WHEN expressions with temp aliases
            temp_alias_columns: List of (temp_alias, original_col) tuples

        Returns:
            SQL query string
        """
        table_source = self.get_table_source()

        # Build SQL based on whether we need WHERE subquery
        if clauses.where_conditions:
            # Step 1: Build inner query with WHERE (no CASE WHEN)
            inner_sql = f"SELECT * FROM {table_source}"

            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            inner_sql += f" WHERE {combined.to_sql(quote_char=self.quote_char)}"

            # Step 2: Build middle query with CASE WHEN using temp aliases
            middle_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in case_when_select)
            middle_sql = f"SELECT {middle_select} FROM ({inner_sql}) AS __filter_subq__"
        else:
            # No WHERE - just CASE WHEN on source table
            inner_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in case_when_select)
            middle_sql = f"SELECT {inner_select} FROM {table_source}"

        # Step 3: Build outer query to rename temp aliases back
        outer_select = ', '.join(
            f'{self.quote_char}{temp}{self.quote_char} AS {self.quote_char}{orig}{self.quote_char}'
            for temp, orig in temp_alias_columns
        )

        sql_parts = [f"SELECT {outer_select}"]
        sql_parts.append(f"FROM ({middle_sql}) AS __case_subq__")

        # Add ORDER BY, LIMIT, OFFSET
        if clauses.orderby_fields:
            orderby_sql = build_orderby_clause(
                clauses.orderby_fields, self.quote_char, stable=is_stable_sort(clauses.orderby_kind)
            )
            sql_parts.append(f"ORDER BY {orderby_sql}")
        elif clauses.needs_row_order():
            sql_parts.append("ORDER BY rowNumberInAllBlocks()")

        if clauses.limit_value is not None:
            sql_parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            sql_parts.append(f"OFFSET {clauses.offset_value}")

        return '\n'.join(sql_parts)

    def build_sql_with_stable_sort_subquery(
        self,
        clauses: ExtractedClauses,
        select_fields: List[Expression] = None,
        distinct: bool = False,
    ) -> str:
        """
        Build SQL with subquery for stable sort when WHERE is present.

        When we have both WHERE and stable ORDER BY, rowNumberInAllBlocks() would give
        post-filter row numbers. To preserve original row order as tie-breaker, we use:

        SELECT * EXCEPT(__orig_row_num__) FROM (
            SELECT *, rowNumberInAllBlocks() AS __orig_row_num__
            FROM source
        ) WHERE conditions
        ORDER BY col1 ASC/DESC, __orig_row_num__ ASC
        LIMIT N

        Args:
            clauses: Extracted SQL clauses
            select_fields: Fields to select
            distinct: Whether to use DISTINCT

        Returns:
            SQL query string
        """
        table_source = self.get_table_source()

        # Build the inner query
        inner_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_source}"

        # Build middle query with columns and WHERE
        middle_parts = []
        distinct_keyword = 'DISTINCT ' if distinct else ''

        if select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            if self.ds._select_star:
                middle_parts.append(f"SELECT {distinct_keyword}*, {fields_sql}, __orig_row_num__")
            else:
                middle_parts.append(f"SELECT {distinct_keyword}{fields_sql}, __orig_row_num__")
        else:
            middle_parts.append(f"SELECT {distinct_keyword}*, __orig_row_num__")

        middle_parts.append(f"FROM ({inner_sql}) AS __subq_with_rownum__")

        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            middle_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        middle_sql = ' '.join(middle_parts)

        # Build outer query with ORDER BY using __orig_row_num__ as tie-breaker
        outer_parts = []
        outer_parts.append("SELECT * EXCEPT(__orig_row_num__)")
        outer_parts.append(f"FROM ({middle_sql}) AS __subq_for_stable_sort__")

        if clauses.orderby_fields:
            orderby_sql = build_orderby_clause(
                clauses.orderby_fields, self.quote_char, stable=False  # Don't add rowNumberInAllBlocks()
            )
            outer_parts.append(f"ORDER BY {orderby_sql}, __orig_row_num__ ASC")

        if clauses.limit_value is not None:
            outer_parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            outer_parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(outer_parts)

    def build_sql_from_plan(self, plan: 'QueryPlan', schema: Dict[str, str] = None) -> SQLBuildResult:
        """
        Build SQL from a QueryPlan.

        This is the unified entry point for SQL building, used by both
        _execute() and _build_execution_sql() to eliminate code duplication.

        Args:
            plan: QueryPlan from QueryPlanner
            schema: Column schema for type-aware SQL building

        Returns:
            SQLBuildResult containing the SQL string and metadata
        """
        from .query_planner import CaseWhenExpr

        schema = schema or {}
        layers = plan.layers
        groupby_agg_op = plan.groupby_agg
        where_ops = plan.where_ops
        alias_renames = plan.alias_renames

        # Collect SELECT fields from SQL operations
        sql_select_fields = []
        for op in plan.sql_ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'SELECT' and op.fields:
                    for f in op.fields:
                        if isinstance(f, str):
                            if f != '*':
                                sql_select_fields.append(Field(f))
                        else:
                            sql_select_fields.append(f)

        # Handle simple vs nested query
        if len(layers) <= 1:
            sql = self._build_simple_query_from_plan(
                layers[0] if layers else [],
                sql_select_fields,
                groupby_agg_op,
                where_ops,
                alias_renames,
                schema,
            )
        else:
            sql = self._build_nested_query_from_plan(layers, sql_select_fields)

        return SQLBuildResult(
            sql=sql,
            alias_renames=alias_renames,
            groupby_agg=groupby_agg_op,
        )

    def _build_simple_query_from_plan(
        self,
        layer_ops: List[LazyOp],
        sql_select_fields: List[Expression],
        groupby_agg_op: Optional[LazyGroupByAgg],
        where_ops: List[Any],
        alias_renames: Dict[str, str],
        schema: Dict[str, str],
    ) -> str:
        """
        Build a simple (non-nested) SQL query from plan components.
        """
        from .query_planner import CaseWhenExpr

        # Extract clauses
        clauses = extract_clauses_from_ops(layer_ops, self.quote_char)

        # Apply alias renames to ORDER BY
        clauses.orderby_fields = apply_alias_renames_to_orderby(clauses.orderby_fields, alias_renames)

        # Handle LazyWhere/LazyMask SQL pushdown (CASE WHEN)
        where_needs_subquery = False
        where_needs_temp_alias = False
        where_temp_alias_columns = []

        if where_ops:
            all_columns = self.ds._get_all_column_names()
            if all_columns:
                where_needs_temp_alias = True
                for col in all_columns:
                    temp_alias = f"__tmp_{col}__"
                    where_temp_alias_columns.append((temp_alias, col))

                sql_select_fields = [
                    CaseWhenExpr(
                        col,
                        where_ops,
                        self.quote_char,
                        schema.get(col, 'Unknown'),
                        alias=f"__tmp_{col}__",
                    )
                    for col in all_columns
                ]

                if clauses.where_conditions:
                    where_needs_subquery = True

        # Handle LazyGroupByAgg SQL pushdown
        groupby_fields_for_sql = self.ds._groupby_fields
        select_fields_for_sql = sql_select_fields

        if groupby_agg_op:
            groupby_fields_for_sql, select_fields_for_sql = build_groupby_select_fields(groupby_agg_op, alias_renames)
            if groupby_agg_op.sort and not clauses.orderby_fields:
                clauses.orderby_fields = [(Field(col), True) for col in groupby_agg_op.groupby_cols]

        # Determine if row order preservation is needed
        needs_row_order = clauses.needs_row_order(has_groupby=groupby_agg_op is not None)

        # Build SQL based on complexity
        if where_needs_subquery:
            return self.build_sql_with_case_when_subquery(clauses, select_fields_for_sql, where_temp_alias_columns)
        elif where_needs_temp_alias and where_temp_alias_columns:
            return self._build_temp_alias_query(
                clauses, select_fields_for_sql, where_temp_alias_columns, needs_row_order
            )
        elif needs_row_order and clauses.where_conditions and not self.ds._joins:
            # Only use row order subquery when there are no JOINs
            # JOINs require the full _build_sql_from_state path
            return self.build_sql_with_row_order_subquery(clauses, select_fields_for_sql, groupby_fields_for_sql)
        else:
            # Simple case - use _build_sql_from_state
            effective_orderby = clauses.orderby_fields
            if needs_row_order and not clauses.orderby_fields:
                effective_orderby = [('__rowNumberInAllBlocks__', True)]

            return self.ds._build_sql_from_state(
                select_fields_for_sql,
                clauses.where_conditions,
                effective_orderby,
                clauses.limit_value,
                clauses.offset_value,
                joins=self.ds._joins,
                distinct=self.ds._distinct,
                groupby_fields=groupby_fields_for_sql,
                having_condition=self.ds._having_condition,
            )

    def _build_temp_alias_query(
        self,
        clauses: ExtractedClauses,
        select_fields: List[Expression],
        temp_alias_columns: List[Tuple[str, str]],
        needs_row_order: bool,
    ) -> str:
        """Build SQL with temp alias wrapping (no WHERE subquery needed)."""
        # Build inner query with CASE WHEN using temp aliases
        inner_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in select_fields)
        inner_sql = self.ds._build_sql_from_state(
            [],
            clauses.where_conditions,
            [],
            None,
            None,
            joins=self.ds._joins,
            distinct=False,
            groupby_fields=[],
            having_condition=None,
        )

        if "SELECT *" in inner_sql:
            inner_sql = inner_sql.replace("SELECT *", f"SELECT {inner_select}", 1)
        else:
            table_source = self.get_table_source()
            inner_sql = f"SELECT {inner_select} FROM {table_source}"

        # Build outer query to rename temp aliases back
        outer_select = ', '.join(
            f'{self.quote_char}{temp}{self.quote_char} AS {self.quote_char}{orig}{self.quote_char}'
            for temp, orig in temp_alias_columns
        )

        sql_parts = [f"SELECT {outer_select}"]
        sql_parts.append(f"FROM ({inner_sql}) AS __case_subq__")

        if clauses.orderby_fields:
            orderby_sql = build_orderby_clause(
                clauses.orderby_fields, self.quote_char, stable=is_stable_sort(clauses.orderby_kind)
            )
            sql_parts.append(f"ORDER BY {orderby_sql}")
        elif needs_row_order:
            sql_parts.append("ORDER BY rowNumberInAllBlocks()")

        if clauses.limit_value is not None:
            sql_parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            sql_parts.append(f"OFFSET {clauses.offset_value}")

        return '\n'.join(sql_parts)

    def _build_nested_query_from_plan(self, layers: List[List[LazyOp]], sql_select_fields: List[Expression]) -> str:
        """Build nested subquery SQL for complex patterns."""
        # Build innermost query (layer 0)
        inner_clauses = extract_clauses_from_ops(layers[0], self.quote_char)

        sql = self.ds._build_sql_from_state(
            sql_select_fields,
            inner_clauses.where_conditions,
            inner_clauses.orderby_fields,
            inner_clauses.limit_value,
            inner_clauses.offset_value,
            joins=self.ds._joins,
            distinct=self.ds._distinct,
            groupby_fields=self.ds._groupby_fields,
            having_condition=self.ds._having_condition,
        )

        # Wrap with outer layers
        for layer_idx, layer_ops in enumerate(layers[1:], 1):
            layer_clauses = extract_clauses_from_ops(layer_ops, self.quote_char)
            sql = self._wrap_with_layer(sql, layer_clauses, layer_idx)

        return sql
