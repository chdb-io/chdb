"""
SQL Execution Engine for DataStore.

This module encapsulates the SQL building and execution logic, extracting common
patterns from core.py to reduce code duplication and improve maintainability.

Key Classes:
- WhereMaskCaseExpr: SQL CASE WHEN expression for LazyWhere/LazyMask pushdown
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
from .lazy_ops import LazyOp, LazyRelationalOp, LazyGroupByAgg, LazyColumnAssignment, LazyJoin, LazyWhere, LazyMask
from .sql_builder import SQLBuilder
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


class WhereMaskCaseExpr(Expression):
    """
    SQL CASE WHEN expression for LazyWhere/LazyMask pushdown.

    This is different from case_when.CaseWhenExpr which is for the general
    CASE WHEN API (ds.when(...).otherwise(...)). This class specifically
    handles the SQL pushdown for pandas-style df.where() and df.mask().

    Generates: CASE WHEN cond THEN col ELSE other END AS col

    For mask (opposite of where):
    CASE WHEN NOT(cond) THEN col ELSE other END AS col

    Type handling:
    - For numeric columns: uses the literal other value
    - For string columns with numeric other: uses Variant type to preserve mixed types
    - NULL is used as a safe fallback for type mismatches
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
        self.alias = alias or column

    def _is_numeric_type(self) -> bool:
        """Check if column is a numeric type."""
        col_type_lower = self.col_type.lower()
        numeric_types = ('int', 'float', 'double', 'decimal', 'uint', 'number')
        return any(t in col_type_lower for t in numeric_types)

    def _is_string_type(self) -> bool:
        """Check if column is a string type."""
        col_type_lower = self.col_type.lower()
        # Include pandas 'object' dtype which is typically used for strings
        string_types = ('string', 'fixedstring', 'enum', 'uuid', 'object')
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
        """Check if we need to use Variant type to preserve mixed types."""
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
        """Format the 'other' value for SQL, considering column type."""
        # None or NaN -> NULL
        if other is None or (isinstance(other, float) and pd.isna(other)):
            return "NULL"

        # Date/DateTime columns: numeric 'other' is incompatible, use NULL
        if self._is_date_type() and isinstance(other, (int, float)):
            return "NULL"

        # Boolean columns: preserve boolean semantics
        if self._is_bool_type() and isinstance(other, (int, float)):
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
                return f"{base_val}::{self._get_variant_type(other)}"
            elif self._is_string_type():
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

        # Use alias
        alias_quoted = f"{qc}{self.alias}{qc}"
        return f"{current_expr} AS {alias_quoted}"

    def __repr__(self) -> str:
        return f"WhereMaskCaseExpr({self.column}, {len(self.where_ops)} ops, type={self.col_type})"


# Backward compatibility alias
CaseWhenExpr = WhereMaskCaseExpr


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
    empty_column_select: bool = False

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
        # Handle LazyColumnAssignment - add computed column to SELECT
        if isinstance(op, LazyColumnAssignment):
            if op.can_push_to_sql():
                expr = op.get_sql_expression()
                result.select_fields.append(expr)
            continue

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

        elif op.op_type == 'SELECT':
            # Handle empty column selection (fields=[]) - return empty DataFrame
            if op.fields is not None and len(op.fields) == 0:
                result.select_fields = []  # Empty select
                result.empty_column_select = True
                continue

            if not op.fields:
                continue
            # Check if this SELECT includes '*' (add computed columns mode)
            # or is explicit column selection (replace mode)
            has_star = any(f == '*' for f in op.fields if isinstance(f, str))

            if has_star:
                # Add computed columns mode: append non-* fields to existing select
                for f in op.fields:
                    if isinstance(f, str):
                        if f != '*':
                            result.select_fields.append(Field(f))
                    else:
                        result.select_fields.append(f)
            else:
                # Explicit column selection: need to check if any selected column
                # references a previously computed column (aliased expression)
                # Build a lookup for computed columns by their alias
                computed_by_alias = {}
                for prev_field in result.select_fields:
                    alias = getattr(prev_field, 'alias', None)
                    if alias:
                        computed_by_alias[alias] = prev_field

                # Build final select in the order specified by the new SELECT
                new_select_fields = []
                for f in op.fields:
                    if isinstance(f, str):
                        col_name = f
                    elif isinstance(f, Field):
                        col_name = f.name
                    else:
                        col_name = None

                    if col_name and col_name in computed_by_alias:
                        # This column references a computed column - use the expression
                        new_select_fields.append(computed_by_alias[col_name])
                    elif isinstance(f, str):
                        new_select_fields.append(Field(f))
                    else:
                        new_select_fields.append(f)

                result.select_fields = new_select_fields

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
    groupby_agg: LazyGroupByAgg,
    alias_renames: Dict[str, str] = None,
    all_columns: List[str] = None,
    computed_columns: Dict[str, Expression] = None,
) -> Tuple[List[Field], List[Expression]]:
    """
    Build GROUP BY and SELECT fields from a LazyGroupByAgg operation.

    This centralizes the aggregation SQL building logic.

    Args:
        groupby_agg: LazyGroupByAgg operation
        alias_renames: Dict for alias conflict resolution
        all_columns: All column names (needed for count() to generate COUNT(col) per column)
        computed_columns: Dict mapping computed column names to their expressions
                          (for expanding assign() columns in aggregations)

    Returns:
        Tuple of (groupby_fields, select_fields)
    """
    from .functions import AggregateFunction

    alias_renames = alias_renames or {}
    computed_columns = computed_columns or {}

    def resolve_column(col_name: str) -> Expression:
        """Resolve column name - return expression if computed column, else Field."""
        if col_name in computed_columns:
            return computed_columns[col_name]
        return Field(col_name)

    # Build GROUP BY fields
    # Use resolve_column to expand computed columns in groupby keys
    groupby_fields = []
    for col in groupby_agg.groupby_cols:
        col_expr = resolve_column(col)
        # If computed column, alias it with the column name for SELECT
        if col in computed_columns:
            col_expr = col_expr.as_(col)
        groupby_fields.append(col_expr)

    # Build SELECT fields with aggregations
    select_fields = list(groupby_fields)  # Include group keys

    if groupby_agg.named_agg:
        # Pandas named aggregation: agg(alias=('col', 'func'))
        # Convert to SQL: SELECT func(col) AS alias ...
        for alias, (col, func) in groupby_agg.named_agg.items():
            sql_func = map_agg_func(func)

            # Check if this alias conflicts with WHERE columns
            temp_alias = f"__agg_{alias}__"
            if temp_alias in alias_renames:
                final_alias = temp_alias
            else:
                final_alias = alias

            # Resolve column - may be a computed column from assign()
            col_expr = resolve_column(col)
            agg_expr = AggregateFunction(sql_func, col_expr, alias=final_alias)
            select_fields.append(agg_expr)

    elif groupby_agg.agg_dict is not None and isinstance(groupby_agg.agg_dict, dict):
        # Pandas-style: agg({'col': 'func'}) or agg({'col': ['func1', 'func2']})
        # Determine if we need compound aliases (col_func) to avoid duplicates
        has_multi_col = len(groupby_agg.agg_dict) > 1
        has_any_multi_func = any(isinstance(f, (list, tuple)) for f in groupby_agg.agg_dict.values())
        is_single_col_agg = getattr(groupby_agg, 'single_column_agg', False)

        # Check for function name conflicts across columns
        all_funcs = []
        for col, funcs in groupby_agg.agg_dict.items():
            if isinstance(funcs, str):
                all_funcs.append(funcs)
            else:
                all_funcs.extend(funcs)
        has_func_conflict = len(all_funcs) != len(set(all_funcs))

        # Alias strategy:
        # - single_column_agg (ColumnExpr.agg(['funcs'])): use function names only (pandas returns flat columns)
        # - agg({col: [funcs]}): use compound alias col_func (pandas returns MultiIndex)
        # - agg({col: func}): use column names only (pandas returns flat columns)
        if is_single_col_agg:
            # Single column agg: pandas returns flat column names with just function names
            use_compound_alias = False
            use_func_only_alias = True
        elif has_any_multi_func or (has_multi_col and has_func_conflict):
            # Multi-func or conflicts: use compound alias for MultiIndex
            use_compound_alias = True
            use_func_only_alias = False
        else:
            # Single func per column: use column names
            use_compound_alias = False
            use_func_only_alias = False

        for col, funcs in groupby_agg.agg_dict.items():
            if isinstance(funcs, str):
                funcs = [funcs]

            for func in funcs:
                sql_func = map_agg_func(func)

                # Alias strategy based on context
                if use_compound_alias:
                    alias = f"{col}_{func}"
                elif use_func_only_alias:
                    alias = func
                else:
                    alias = col

                # Check if this alias conflicts with WHERE columns
                temp_alias = f"__agg_{alias}__"
                if temp_alias in alias_renames:
                    alias = temp_alias

                # Resolve column - may be a computed column from assign()
                col_expr = resolve_column(col)
                agg_expr = AggregateFunction(sql_func, col_expr, alias=alias)
                select_fields.append(agg_expr)

    elif groupby_agg.agg_func:
        # Single function for all numeric columns
        func = groupby_agg.agg_func
        sql_func = map_agg_func(func)

        if func == 'size':
            # size() counts ALL rows including NULL -> COUNT(*)
            select_fields.append(AggregateFunction(sql_func, Star()))
        elif all_columns:
            # Apply aggregation to non-groupby columns (or only selected columns)
            # This handles sum, mean, count, min, max, std, var, first, last, etc.
            # If selected_columns is set, only aggregate those columns (excluding groupby keys)
            if groupby_agg.selected_columns:
                # Filter out groupby columns from selected_columns
                cols_to_agg = [c for c in groupby_agg.selected_columns if c not in groupby_agg.groupby_cols]
            else:
                cols_to_agg = [c for c in all_columns if c not in groupby_agg.groupby_cols]
            for col in cols_to_agg:
                # Check if this alias conflicts with WHERE columns
                temp_alias = f"__agg_{col}__"
                if temp_alias in alias_renames:
                    alias = temp_alias
                else:
                    alias = col
                # Resolve column - may be a computed column from assign()
                col_expr = resolve_column(col)
                agg_expr = AggregateFunction(sql_func, col_expr, alias=alias)
                select_fields.append(agg_expr)
        else:
            # Fallback: if we don't know columns, use COUNT(*) for count, otherwise skip
            if func == 'count':
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

    def _wrap_with_layer(
        self,
        inner_sql: str,
        clauses: ExtractedClauses,
        layer_idx: int,
        preserved_orderby: List[Tuple[Any, bool]] = None,
        preserved_orderby_kind: str = None,
    ) -> str:
        """Wrap an inner SQL query with an outer layer.

        Args:
            inner_sql: The inner SQL query to wrap
            clauses: Clauses to apply in this layer
            layer_idx: Index of this layer (for alias naming)
            preserved_orderby: ORDER BY from inner layers to preserve result ordering
            preserved_orderby_kind: Sort kind (for stable sort detection)
        """
        outer_parts = ["SELECT *"]
        outer_parts.append(f"FROM ({inner_sql}) AS __subq{layer_idx}__")

        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            outer_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # Use layer's ORDER BY if present, otherwise use preserved ORDER BY from inner layers
        # This ensures patterns like (ORDER BY + LIMIT) + WHERE maintain consistent row ordering
        orderby_to_use = clauses.orderby_fields if clauses.orderby_fields else preserved_orderby
        orderby_kind_to_use = clauses.orderby_kind if clauses.orderby_fields else preserved_orderby_kind

        if orderby_to_use:
            orderby_sql = build_orderby_clause(
                orderby_to_use, self.quote_char, stable=is_stable_sort(orderby_kind_to_use)
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
        self,
        clauses: ExtractedClauses,
        select_fields: List[Expression] = None,
        groupby_fields: List[Expression] = None,
        include_star: bool = False,
    ) -> str:
        """
        Build SQL with subquery for row order preservation.

        When we have WHERE conditions and need to preserve row order,
        rowNumberInAllBlocks() must be computed BEFORE filtering.

        Args:
            clauses: Extracted SQL clauses
            select_fields: Fields to select
            groupby_fields: GROUP BY fields
            include_star: If True and select_fields present, use SELECT *, fields

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
            if include_star:
                # For computed columns, need to include all original columns plus computed
                sql_parts.append(f"SELECT * EXCEPT(__orig_row_num__), {fields_sql}")
            else:
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

        ClickHouse has alias conflict quirks, so we use multi-layer subquery:
        1. Innermost: Capture original row order BEFORE filtering
        2. Inner: WHERE conditions (filters data)
        3. Middle: CASE WHEN with temp aliases
        4. Outer: Rename temp aliases back to original, ORDER BY original row number

        This ensures row order is preserved even when WHERE filters rows,
        because rowNumberInAllBlocks() is captured before filtering.

        Args:
            clauses: Extracted SQL clauses
            case_when_select: CASE WHEN expressions with temp aliases
            temp_alias_columns: List of (temp_alias, original_col) tuples

        Returns:
            SQL query string
        """
        table_source = self.get_table_source()
        needs_row_order = clauses.needs_row_order()

        # Build SQL based on whether we need WHERE subquery
        if clauses.where_conditions:
            if needs_row_order:
                # Step 0: Capture original row number BEFORE filtering
                # This is critical for stable row order - rowNumberInAllBlocks() must be
                # called on the source table, not after WHERE filtering
                innermost_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_source}"

                # Step 1: Build query with WHERE (no CASE WHEN yet)
                combined = clauses.where_conditions[0]
                for cond in clauses.where_conditions[1:]:
                    combined = combined & cond
                inner_sql = f"SELECT * FROM ({innermost_sql}) AS __rownum_subq__ WHERE {combined.to_sql(quote_char=self.quote_char)}"

                # Step 2: Build middle query with CASE WHEN using temp aliases
                # Need to include __orig_row_num__ in the select
                middle_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in case_when_select)
                middle_sql = f"SELECT {middle_select}, __orig_row_num__ FROM ({inner_sql}) AS __filter_subq__"
            else:
                # No row order needed - simpler structure
                inner_sql = f"SELECT * FROM {table_source}"
                combined = clauses.where_conditions[0]
                for cond in clauses.where_conditions[1:]:
                    combined = combined & cond
                inner_sql += f" WHERE {combined.to_sql(quote_char=self.quote_char)}"

                middle_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in case_when_select)
                middle_sql = f"SELECT {middle_select} FROM ({inner_sql}) AS __filter_subq__"
        else:
            # No WHERE - just CASE WHEN on source table
            if needs_row_order:
                # Still need to capture row order
                innermost_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_source}"
                inner_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in case_when_select)
                middle_sql = f"SELECT {inner_select}, __orig_row_num__ FROM ({innermost_sql}) AS __rownum_subq__"
            else:
                inner_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in case_when_select)
                middle_sql = f"SELECT {inner_select} FROM {table_source}"

        # Step 3: Build outer query to rename temp aliases back
        outer_select = ', '.join(
            f'{self.quote_char}{temp}{self.quote_char} AS {self.quote_char}{orig}{self.quote_char}'
            for temp, orig in temp_alias_columns
        )

        if needs_row_order:
            # Use SELECT ... EXCEPT to exclude __orig_row_num__ from final output
            sql_parts = [f"SELECT {outer_select}"]
            sql_parts.append(f"FROM ({middle_sql}) AS __case_subq__")

            # Order by original row number (captured before WHERE)
            if clauses.orderby_fields:
                orderby_sql = build_orderby_clause(clauses.orderby_fields, self.quote_char, stable=False)
                sql_parts.append(f"ORDER BY {orderby_sql}, __orig_row_num__ ASC")
            else:
                sql_parts.append("ORDER BY __orig_row_num__ ASC")
        else:
            sql_parts = [f"SELECT {outer_select}"]
            sql_parts.append(f"FROM ({middle_sql}) AS __case_subq__")

            if clauses.orderby_fields:
                orderby_sql = build_orderby_clause(
                    clauses.orderby_fields, self.quote_char, stable=is_stable_sort(clauses.orderby_kind)
                )
                sql_parts.append(f"ORDER BY {orderby_sql}")

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
        schema = schema or {}
        layers = plan.layers
        groupby_agg_op = plan.groupby_agg
        where_ops = plan.where_ops
        alias_renames = plan.alias_renames

        # Collect LazyColumnAssignments for potential SQLBuilder handling
        column_assignments = [
            op for op in plan.sql_ops if isinstance(op, LazyColumnAssignment) and op.can_push_to_sql()
        ]

        # Check if we need SQLBuilder for complex column assignment scenarios
        # SQLBuilder is needed when:
        # 1. Multiple assignments to the same column (override scenario)
        # 2. Computed column is referenced in subsequent operations (WHERE, ORDER BY)
        # Skip SQLBuilder if there is a groupby operation (SQLBuilder does not support GROUP BY yet)
        if (
            column_assignments
            and not groupby_agg_op
            and self._needs_sql_builder(plan.sql_ops, column_assignments, schema)
        ):
            sql = self._build_sql_with_builder(plan, schema)
            return SQLBuildResult(
                sql=sql,
                alias_renames=alias_renames,
                groupby_agg=groupby_agg_op,
            )

        # Collect SELECT fields from SQL operations (original logic)
        sql_select_fields = []
        has_column_assignments = False
        for op in plan.sql_ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'SELECT' and op.fields:
                    for f in op.fields:
                        if isinstance(f, str):
                            if f != '*':
                                sql_select_fields.append(Field(f))
                        else:
                            sql_select_fields.append(f)
            # Handle LazyColumnAssignment - add computed column expression
            elif isinstance(op, LazyColumnAssignment) and op.can_push_to_sql():
                sql_select_fields.append(op.get_sql_expression())
                has_column_assignments = True

        # Handle simple vs nested query
        if len(layers) <= 1:
            sql = self._build_simple_query_from_plan(
                layers[0] if layers else [],
                sql_select_fields,
                groupby_agg_op,
                where_ops,
                alias_renames,
                schema,
                has_column_assignments,
            )
        else:
            sql = self._build_nested_query_from_plan(layers, sql_select_fields, has_column_assignments)

        return SQLBuildResult(
            sql=sql,
            alias_renames=alias_renames,
            groupby_agg=groupby_agg_op,
        )

    def _needs_sql_builder(
        self,
        sql_ops: List[LazyOp],
        column_assignments: List[LazyColumnAssignment],
        schema: Dict[str, str],
    ) -> bool:
        """
        Check if we need to use SQLBuilder for complex column assignment scenarios.

        SQLBuilder is needed when:
        1. Multiple assignments to the same column name (override scenario)
        2. A WHERE/ORDER BY references a computed column
        3. A computed column expression references another computed column
        4. Single assignment overrides an existing column (needs EXCEPT)
        5. Column selection with computed columns (needs subquery)

        SQLBuilder is NOT used when:
        - There are JOIN operations (SQLBuilder doesn't handle JOINs yet)

        Args:
            sql_ops: All SQL operations in the plan
            column_assignments: LazyColumnAssignment operations
            schema: Column schema

        Returns:
            True if SQLBuilder should be used
        """
        if not column_assignments:
            return False

        # Don't use SQLBuilder if there are JOIN operations (not supported yet)
        if self.ds._joins:
            return False

        # Don't use SQLBuilder if there are GROUP BY operations (not supported yet)
        if self.ds._groupby_fields:
            return False

        # Don't use SQLBuilder if DISTINCT is used (not supported yet)
        if self.ds._distinct:
            return False

        # Get existing column names from schema
        existing_columns = set(schema.keys()) if schema else set()

        # Check for duplicate column names (override scenario)
        assigned_columns = set()
        for op in column_assignments:
            if op.column in assigned_columns:
                return True
            assigned_columns.add(op.column)

        # Check if any assignment overrides an existing column (needs EXCEPT)
        if assigned_columns & existing_columns:
            return True

        # Check if any computed column is referenced in WHERE, ORDER BY, or where()/mask() conditions
        for op in sql_ops:
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'WHERE' and op.condition:
                    referenced = self._extract_column_references(op.condition)
                    if referenced & assigned_columns:
                        return True
                elif op.op_type == 'ORDER BY' and op.fields:
                    for f in op.fields:
                        if isinstance(f, Field):
                            col_name = f.name.strip('"\'')
                            if col_name in assigned_columns:
                                return True
                        elif isinstance(f, str):
                            if f.strip('"\'') in assigned_columns:
                                return True
            # Check LazyWhere/LazyMask conditions for computed column references
            elif isinstance(op, (LazyWhere, LazyMask)):
                referenced = self._extract_column_references(op.condition)
                if referenced & assigned_columns:
                    return True

        # Check if any computed column references another computed column
        for op in column_assignments:
            expr_refs = self._extract_column_references(op.expr)
            # Remove self-reference (for override detection)
            other_computed = assigned_columns - {op.column}
            if expr_refs & other_computed:
                return True

        # Check if there's a column selection (SELECT) with computed columns
        # This requires a subquery to first materialize computed columns
        for op in sql_ops:
            if isinstance(op, LazyRelationalOp) and op.op_type == 'SELECT':
                if op.fields:
                    # Has explicit column selection with computed columns - need subquery
                    return True

        return False

    def _extract_column_references(self, expr) -> set:
        """
        Extract column names referenced in an expression.

        Args:
            expr: Expression to analyze

        Returns:
            Set of column names
        """
        columns = set()

        if expr is None:
            return columns

        if isinstance(expr, Field):
            name = expr.name.strip('"\'')
            columns.add(name)

        elif hasattr(expr, 'left') and hasattr(expr, 'right'):
            columns.update(self._extract_column_references(expr.left))
            columns.update(self._extract_column_references(expr.right))

        elif hasattr(expr, 'condition'):
            columns.update(self._extract_column_references(expr.condition))

        elif hasattr(expr, 'expression'):
            columns.update(self._extract_column_references(expr.expression))

        elif hasattr(expr, 'args'):
            for arg in expr.args:
                columns.update(self._extract_column_references(arg))

        elif hasattr(expr, '_expr'):
            # ColumnExpr
            columns.update(self._extract_column_references(expr._expr))

        elif hasattr(expr, 'nodes'):
            # Expression with nodes() method
            for node in expr.nodes():
                if isinstance(node, Field):
                    name = node.name.strip('"\'')
                    columns.add(name)

        return columns

    def _build_sql_with_builder(self, plan: 'QueryPlan', schema: Dict[str, str]) -> str:
        """
        Build SQL using SQLBuilder for complex column assignment scenarios.

        SQLBuilder handles:
        - Automatic subquery wrapping when computed columns are referenced
        - Column override with EXCEPT syntax
        - Proper ordering of operations

        Args:
            plan: QueryPlan
            schema: Column schema

        Returns:
            SQL string
        """
        table_source = self.get_table_source()
        known_columns = list(schema.keys()) if schema else None

        builder = SQLBuilder(table_source, known_columns)

        # Determine if row order preservation is needed
        has_explicit_orderby = False

        # Process operations in order
        for op in plan.sql_ops:
            if isinstance(op, LazyColumnAssignment) and op.can_push_to_sql():
                # Get the expression from ColumnAssignment
                expr = op.get_sql_expression()
                # Remove alias from expression (SQLBuilder will add it)
                if hasattr(expr, 'alias'):
                    expr.alias = None
                builder.add_computed_column(op.column, expr)

            elif isinstance(op, LazyRelationalOp):
                if op.op_type == 'WHERE' and op.condition:
                    builder.add_filter(op.condition)
                elif op.op_type == 'ORDER BY' and op.fields:
                    has_explicit_orderby = True
                    orderby_fields = []
                    ascending_list = normalize_ascending(
                        op.ascending if hasattr(op, 'ascending') else True,
                        len(op.fields),
                    )
                    for f, asc in zip(op.fields, ascending_list):
                        if isinstance(f, str):
                            orderby_fields.append((Field(f), asc))
                        else:
                            orderby_fields.append((f, asc))
                    builder.add_orderby(orderby_fields)
                elif op.op_type == 'LIMIT' and hasattr(op, 'limit_value'):
                    builder.add_limit(op.limit_value)
                elif op.op_type == 'OFFSET' and hasattr(op, 'offset_value'):
                    builder.add_offset(op.offset_value)
                elif op.op_type == 'SELECT' and op.fields:
                    # Column selection
                    columns = []
                    for f in op.fields:
                        if isinstance(f, str):
                            if f != '*':
                                columns.append(f)
                        elif isinstance(f, Field):
                            columns.append(f.name.strip('"\''))
                    if columns:
                        builder.select_columns(columns)

        # Set row order preservation if no explicit ORDER BY
        if not has_explicit_orderby:
            builder.set_preserve_row_order(True)

        sql = builder.build(self.quote_char)

        # Add timezone settings
        sql = self._append_settings(sql)

        return sql

    def _append_settings(self, sql: str) -> str:
        """Append timezone settings to SQL if needed."""
        # Check if SETTINGS already present
        if 'SETTINGS' not in sql:
            sql = f"{sql} SETTINGS session_timezone='UTC'"
        return sql

    def _build_simple_query_from_plan(
        self,
        layer_ops: List[LazyOp],
        sql_select_fields: List[Expression],
        groupby_agg_op: Optional[LazyGroupByAgg],
        where_ops: List[Any],
        alias_renames: Dict[str, str],
        schema: Dict[str, str],
        has_column_assignments: bool = False,
    ) -> str:
        """
        Build a simple (non-nested) SQL query from plan components.
        """
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
            # Get columns for aggregation - prioritize explicit column selection from
            # df[['col1', 'col2']].groupby(...) over all source columns
            # This ensures that df[['Pclass', 'Survived']].groupby('Pclass').mean()
            # only aggregates 'Survived', not all columns from the source file
            if clauses.select_fields:
                # Use explicitly selected columns (from LazyRelationalOp(SELECT))
                all_cols = []
                for f in clauses.select_fields:
                    if isinstance(f, Field):
                        all_cols.append(f.name)
                    elif hasattr(f, 'alias') and f.alias:
                        all_cols.append(f.alias)
                    else:
                        all_cols.append(str(f))
            elif sql_select_fields:
                # Use columns from passed-in select fields
                all_cols = []
                for f in sql_select_fields:
                    if isinstance(f, Field):
                        all_cols.append(f.name)
                    elif hasattr(f, 'alias') and f.alias:
                        all_cols.append(f.alias)
                    else:
                        all_cols.append(str(f))
            else:
                # Fall back to all source columns
                all_cols = self.ds._get_all_column_names() if hasattr(self.ds, '_get_all_column_names') else None
            # Get computed columns for expanding assign() columns in aggregations
            computed_cols = getattr(self.ds, '_computed_columns', None) or {}
            groupby_fields_for_sql, select_fields_for_sql = build_groupby_select_fields(
                groupby_agg_op, alias_renames, all_columns=all_cols, computed_columns=computed_cols
            )
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
            return self.build_sql_with_row_order_subquery(
                clauses, select_fields_for_sql, groupby_fields_for_sql, include_star=has_column_assignments
            )
        else:
            # Simple case - use assemble_sql
            effective_orderby = clauses.orderby_fields
            if needs_row_order and not clauses.orderby_fields:
                effective_orderby = [('__rowNumberInAllBlocks__', True)]

            return self.assemble_sql(
                select_fields_for_sql,
                clauses.where_conditions,
                effective_orderby,
                clauses.limit_value,
                clauses.offset_value,
                joins=self.ds._joins,
                distinct=self.ds._distinct,
                groupby_fields=groupby_fields_for_sql,
                having_condition=self.ds._having_condition,
                include_star=has_column_assignments if has_column_assignments else None,
            )

    def _build_temp_alias_query(
        self,
        clauses: ExtractedClauses,
        select_fields: List[Expression],
        temp_alias_columns: List[Tuple[str, str]],
        needs_row_order: bool,
    ) -> str:
        """Build SQL with temp alias wrapping (no WHERE subquery needed).

        When row order preservation is needed, we must capture rowNumberInAllBlocks()
        at the source level, even if there are no WHERE conditions.
        """
        table_source = self.get_table_source()
        inner_select = ', '.join(f.to_sql(quote_char=self.quote_char) for f in select_fields)

        if needs_row_order:
            if clauses.where_conditions:
                # Capture row number before WHERE filtering
                rownum_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_source}"

                # Build WHERE on top of row number subquery
                combined = clauses.where_conditions[0]
                for cond in clauses.where_conditions[1:]:
                    combined = combined & cond

                inner_sql = f"SELECT {inner_select}, __orig_row_num__ FROM ({rownum_sql}) AS __rownum_subq__ WHERE {combined.to_sql(quote_char=self.quote_char)}"
            else:
                # No WHERE but still need row order - simpler structure
                rownum_sql = f"SELECT *, rowNumberInAllBlocks() AS __orig_row_num__ FROM {table_source}"
                inner_sql = f"SELECT {inner_select}, __orig_row_num__ FROM ({rownum_sql}) AS __rownum_subq__"
        else:
            # No row order needed - use original simple structure
            inner_sql = self.assemble_sql(
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
                inner_sql = f"SELECT {inner_select} FROM {table_source}"

        # Build outer query to rename temp aliases back
        outer_select = ', '.join(
            f'{self.quote_char}{temp}{self.quote_char} AS {self.quote_char}{orig}{self.quote_char}'
            for temp, orig in temp_alias_columns
        )

        sql_parts = [f"SELECT {outer_select}"]
        sql_parts.append(f"FROM ({inner_sql}) AS __case_subq__")

        if clauses.orderby_fields:
            if needs_row_order:
                # Use __orig_row_num__ as tie-breaker (no need for rowNumberInAllBlocks())
                orderby_sql = build_orderby_clause(clauses.orderby_fields, self.quote_char, stable=False)
                sql_parts.append(f"ORDER BY {orderby_sql}, __orig_row_num__ ASC")
            else:
                # Respect user's stable sort setting
                orderby_sql = build_orderby_clause(
                    clauses.orderby_fields, self.quote_char, stable=is_stable_sort(clauses.orderby_kind)
                )
                sql_parts.append(f"ORDER BY {orderby_sql}")
        elif needs_row_order:
            sql_parts.append("ORDER BY __orig_row_num__ ASC")

        if clauses.limit_value is not None:
            sql_parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            sql_parts.append(f"OFFSET {clauses.offset_value}")

        return '\n'.join(sql_parts)

    def _build_nested_query_from_plan(
        self, layers: List[List[LazyOp]], sql_select_fields: List[Expression], has_column_assignments: bool = False
    ) -> str:
        """Build nested subquery SQL for complex patterns."""
        # Build innermost query (layer 0)
        inner_clauses = extract_clauses_from_ops(layers[0], self.quote_char)

        sql = self.assemble_sql(
            sql_select_fields,
            inner_clauses.where_conditions,
            inner_clauses.orderby_fields,
            inner_clauses.limit_value,
            inner_clauses.offset_value,
            joins=self.ds._joins,
            distinct=self.ds._distinct,
            groupby_fields=self.ds._groupby_fields,
            having_condition=self.ds._having_condition,
            include_star=has_column_assignments if has_column_assignments else None,
        )

        # Track ORDER BY from inner layers to preserve sort order in final result
        # When we have patterns like ORDER BY + LIMIT + WHERE, the inner ORDER BY
        # must be applied to the final result to maintain consistent row ordering
        preserved_orderby = inner_clauses.orderby_fields
        preserved_orderby_kind = inner_clauses.orderby_kind

        # Wrap with outer layers
        for layer_idx, layer_ops in enumerate(layers[1:], 1):
            layer_clauses = extract_clauses_from_ops(layer_ops, self.quote_char)

            # If this layer has ORDER BY, use it; otherwise preserve from inner
            if layer_clauses.orderby_fields:
                preserved_orderby = layer_clauses.orderby_fields
                preserved_orderby_kind = layer_clauses.orderby_kind

            sql = self._wrap_with_layer(
                sql,
                layer_clauses,
                layer_idx,
                preserved_orderby=preserved_orderby,
                preserved_orderby_kind=preserved_orderby_kind,
            )

        return sql

    def assemble_sql(
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
        include_star=None,
    ) -> str:
        """
        Assemble SQL query from given components.

        This is the core SQL assembly method, moved from DataStore._build_sql_from_state
        to centralize SQL building logic.

        Args:
            select_fields: Fields to select
            where_conditions: List of WHERE conditions
            orderby_fields: List of (field, ascending) tuples
            limit_value: LIMIT value
            offset_value: OFFSET value
            joins: List of JOIN tuples
            distinct: Whether to use DISTINCT
            groupby_fields: GROUP BY fields
            having_condition: HAVING condition
            include_star: If True, use SELECT *, computed_cols. If None, use self.ds._select_star

        Returns:
            SQL query string
        """
        # Check if we need a subquery for stable sort with WHERE
        # When both WHERE and stable ORDER BY exist, rowNumberInAllBlocks() would give
        # post-filter row numbers, not original row numbers. We need a subquery to preserve
        # the original row order.
        needs_stable_sort = orderby_fields and is_stable_sort(self.ds._orderby_kind)
        needs_subquery_for_stable = needs_stable_sort and where_conditions and not groupby_fields and not joins

        if needs_subquery_for_stable:
            # Use stable sort subquery
            clauses = ExtractedClauses(
                where_conditions=where_conditions,
                orderby_fields=orderby_fields,
                orderby_kind=self.ds._orderby_kind,
                limit_value=limit_value,
                offset_value=offset_value,
            )
            return self.build_sql_with_stable_sort_subquery(clauses, select_fields, distinct)

        parts = []

        # SELECT (with optional DISTINCT)
        distinct_keyword = 'DISTINCT ' if distinct else ''
        if select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            # Check if we need to prepend '*' (SELECT *, computed_col)
            # IMPORTANT: Don't use SELECT * with GROUP BY - only groupby columns and aggregates are valid
            # Use include_star if explicitly set, otherwise fall back to self.ds._select_star
            use_star = include_star if include_star is not None else self.ds._select_star
            if use_star and not groupby_fields:
                parts.append(f"SELECT {distinct_keyword}*, {fields_sql}")
            else:
                parts.append(f"SELECT {distinct_keyword}{fields_sql}")
        else:
            parts.append(f"SELECT {distinct_keyword}*")

        # FROM (with alias if joins present)
        if self.ds._table_function:
            # Handle table function objects
            if hasattr(self.ds._table_function, 'to_sql'):
                table_sql = self.ds._table_function.to_sql()
            else:
                table_sql = str(self.ds._table_function)
            # Add alias when joins are present (required by ClickHouse for disambiguation)
            if joins:
                alias = self.ds._get_table_alias()
                parts.append(f"FROM {table_sql} AS {format_identifier(alias, self.quote_char)}")
            else:
                parts.append(f"FROM {table_sql}")
        elif self.ds.table_name:
            parts.append(f"FROM {self.quote_char}{self.ds.table_name}{self.quote_char}")

        # JOIN clauses
        if joins:
            for other_ds, join_type, join_condition in joins:
                parts.append(self._build_join_clause(other_ds, join_type, join_condition))

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
                    orderby_fields, self.quote_char, stable=is_stable_sort(self.ds._orderby_kind)
                )
                parts.append(f"ORDER BY {orderby_sql}")

        # LIMIT
        if limit_value is not None:
            parts.append(f"LIMIT {limit_value}")

        # OFFSET
        if offset_value is not None:
            parts.append(f"OFFSET {offset_value}")

        return ' '.join(parts)

    def execute_sql_on_dataframe(
        self,
        df: pd.DataFrame,
        plan: 'QueryPlan',
        schema: Dict[str, str] = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL segment on an existing DataFrame using chDB's Python() table function.

        This enables SQL execution on intermediate DataFrames in the pipeline,
        supporting true SQL-Pandas-SQL interleaving.

        Args:
            df: Input DataFrame to query
            plan: QueryPlan for this SQL segment
            schema: Column schema for type-aware SQL generation

        Returns:
            Result DataFrame after SQL execution
        """
        from .executor import get_executor

        schema = schema or {}

        # Build SQL query using Python() table function
        sql = self._build_sql_for_dataframe(df, plan, schema)

        self._logger.debug("  [SQL on DataFrame] Executing: %s", sql[:200] + "..." if len(sql) > 200 else sql)

        # Execute via chDB using Python() table function
        executor = get_executor()
        result_df = executor.query_dataframe(sql, df, '__df__')

        # Handle empty column selection: return DataFrame with 0 columns but correct row count
        if getattr(plan, '_empty_column_select', False):
            # Remove _row_id and return empty column DataFrame
            result_df = result_df.drop(columns=['_row_id'], errors='ignore')
            # Return DataFrame with correct index but no columns
            return result_df[[]]

        # Handle GroupBy SQL pushdown: set group keys as index
        # Exception: when as_index=False, keep columns as regular columns (matching Pandas behavior)
        if plan.groupby_agg and plan.groupby_agg.groupby_cols:
            groupby_cols = plan.groupby_agg.groupby_cols
            # Don't set index if as_index=False (user wants group keys as columns)
            as_index = getattr(plan.groupby_agg, 'as_index', True)
            if as_index:
                if all(col in result_df.columns for col in groupby_cols):
                    result_df = result_df.set_index(groupby_cols)
                    self._logger.debug("  Set groupby columns as index: %s", groupby_cols)

        # Handle alias renames
        if plan.alias_renames:
            rename_back = {temp: orig for temp, orig in plan.alias_renames.items() if temp in result_df.columns}
            if rename_back:
                result_df = result_df.rename(columns=rename_back)
                self._logger.debug("  Renamed temp aliases: %s", rename_back)

        # Convert flat column names to MultiIndex for pandas compatibility
        # (when using agg_dict with multiple functions per column)
        # Skip for single_column_agg which should return flat column names
        if plan.groupby_agg and plan.groupby_agg.agg_dict is not None and isinstance(plan.groupby_agg.agg_dict, dict):
            is_single_col_agg = getattr(plan.groupby_agg, 'single_column_agg', False)
            if not is_single_col_agg:
                col_rename_map = {}
                for col, funcs in plan.groupby_agg.agg_dict.items():
                    if isinstance(funcs, str):
                        funcs = [funcs]
                    for func in funcs:
                        flat_name = f"{col}_{func}"
                        if flat_name in result_df.columns:
                            col_rename_map[flat_name] = (col, func)

                if col_rename_map:
                    new_columns = []
                    for c in result_df.columns:
                        if c in col_rename_map:
                            new_columns.append(col_rename_map[c])
                        else:
                            new_columns.append((c, ''))
                    result_df.columns = pd.MultiIndex.from_tuples(new_columns)
                    self._logger.debug("  Converted flat columns to MultiIndex")

        # Apply dtype corrections for SQL results (e.g., abs() on signed integers)
        result_df = self._apply_sql_dtype_corrections(result_df, df, plan)

        return result_df

    def _apply_sql_dtype_corrections(
        self,
        result_df: pd.DataFrame,
        input_df: pd.DataFrame,
        plan: 'QueryPlan',
    ) -> pd.DataFrame:
        """
        Apply dtype corrections for SQL results where chDB returns different types than pandas.

        Uses the centralized DtypeCorrectionRegistry to apply corrections based on
        configurable rules. This handles various dtype mismatches:
        - abs(): unsigned → signed for signed int input
        - sign(): int8 → preserve input type
        - pow(): float64 → int for integer input
        - arithmetic ops: type width preservation
        - And more...

        Args:
            result_df: Result DataFrame from SQL execution
            input_df: Input DataFrame (for dtype lookup)
            plan: QueryPlan with operation metadata

        Returns:
            DataFrame with corrected dtypes
        """
        from .dtype_correction import dtype_registry

        # Extract operations that may need correction from the plan
        operations = dtype_registry.extract_operations_from_plan(plan, input_df)

        if not operations:
            return result_df

        # Apply corrections using the registry
        return dtype_registry.apply_corrections_to_dataframe(result_df, input_df, operations)

    def _build_sql_for_dataframe(
        self,
        df: pd.DataFrame,
        plan: 'QueryPlan',
        schema: Dict[str, str] = None,
    ) -> str:
        """
        Build SQL query for executing on a DataFrame via Python() table function.

        Row order preservation is handled by connection.query_df with preserve_order=True,
        so this method builds simple SQL without explicit row ordering logic.

        Handles nested LIMIT-WHERE patterns: When we have patterns like
        [:50][>60][:10][>75], we need nested subqueries to maintain Pandas semantics.

        Args:
            df: Input DataFrame (used for schema info)
            plan: QueryPlan for this SQL segment
            schema: Column schema

        Returns:
            SQL query string using Python() table function
        """
        schema = schema or {}

        # Check if we have multiple layers (nested LIMIT-WHERE patterns)
        if plan.layers and len(plan.layers) > 1:
            # Build nested subqueries from layers
            return self._build_nested_sql_for_dataframe(plan.layers)

        # Extract clauses for simple/single layer case
        clauses = extract_clauses_from_ops(plan.sql_ops, self.quote_char)

        # Apply alias renames to ORDER BY
        if plan.alias_renames and clauses.orderby_fields:
            clauses.orderby_fields = apply_alias_renames_to_orderby(clauses.orderby_fields, plan.alias_renames)

        # Handle GroupBy and CASE WHEN ops (these apply to final result)
        groupby_fields = []
        select_fields = []

        if plan.groupby_agg:
            # Pass all_columns for count() to generate COUNT(col) per column
            df_columns = list(df.columns)
            # Get computed columns for expanding assign() columns in aggregations
            computed_cols = getattr(self.ds, '_computed_columns', None) or {}
            groupby_fields, select_fields = build_groupby_select_fields(
                plan.groupby_agg, plan.alias_renames, all_columns=df_columns, computed_columns=computed_cols
            )

        # Track temp alias mapping for where_ops (CASE WHEN)
        # chDB has alias conflict quirks: if a CASE WHEN uses "col" as alias and
        # later expressions reference "col", they use the aliased value instead of original.
        # Solution: use temp aliases (__tmp_col__) then rename back with outer subquery.
        where_temp_alias_map = {}  # temp_alias -> original_col

        if plan.where_ops:
            all_columns = list(df.columns)
            for col in all_columns:
                col_type = schema.get(col, str(df[col].dtype))
                temp_alias = f"__tmp_{col}__"
                case_expr = CaseWhenExpr(col, plan.where_ops, self.quote_char, col_type, alias=temp_alias)
                select_fields.append(case_expr)
                where_temp_alias_map[temp_alias] = col

        # Determine SELECT clause
        # Priority: empty column select -> CASE WHEN/GroupBy fields -> explicit column selection -> *
        if clauses.empty_column_select:
            # Empty column selection: df[[]] - return DataFrame with 0 columns
            # We use _row_id for row tracking (built-in virtual column in chDB v4.0.0b5+)
            select_sql = '_row_id'
            # Mark that we want to return empty columns (handled in post-processing)
            plan._empty_column_select = True
        elif select_fields:
            select_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            # No need to add _row_id here - connection.query_df handles it automatically
        elif clauses.select_fields:
            # Check if SELECT * was specified (need all original columns + computed columns)
            # But only add '*' if clauses.select_fields doesn't already contain explicit column
            # selections (Field objects). If there are Field objects, they represent the columns
            # from a previous column selection operation and '*' should not be added.
            # Note: Fields with alias (e.g., from LazyColumnAssignment) are computed columns,
            # not explicit column selections. Only count non-aliased Fields as explicit.
            has_explicit_columns = any(isinstance(f, Field) and not f.alias for f in clauses.select_fields)
            # Also check if we have LazyColumnAssignment - these need * plus computed columns
            # BUT only if there's no explicit column selection (SELECT col1, col2, ...)
            column_assignments = [
                op for op in plan.sql_ops if isinstance(op, LazyColumnAssignment) and op.can_push_to_sql()
            ]
            has_column_assignments = bool(column_assignments)

            # Check for column overrides (assigned column already exists in DataFrame)
            df_columns_ordered = list(df.columns)
            df_columns_set = set(df_columns_ordered)
            override_columns = {op.column for op in column_assignments if op.column in df_columns_set}

            if (self.ds._select_star and not has_explicit_columns) or (
                has_column_assignments and not has_explicit_columns
            ):
                if override_columns:
                    # Build explicit column list to preserve column order when overriding
                    # For each column: if it's being overridden, use the computed expression
                    # Otherwise, just select the original column
                    assignment_map = {op.column: op for op in column_assignments if op.column in override_columns}
                    select_parts = []
                    for col in df_columns_ordered:
                        if col in assignment_map:
                            # Use the computed expression for overridden column
                            op = assignment_map[col]
                            expr = op.get_sql_expression()
                            select_parts.append(expr.to_sql(quote_char=self.quote_char, with_alias=True))
                        else:
                            select_parts.append(f'{self.quote_char}{col}{self.quote_char}')
                    # Add new columns (not overrides)
                    new_columns = [op for op in column_assignments if op.column not in override_columns]
                    for op in new_columns:
                        expr = op.get_sql_expression()
                        select_parts.append(expr.to_sql(quote_char=self.quote_char, with_alias=True))
                    select_sql = ', '.join(select_parts)
                else:
                    fields_sql = ', '.join(
                        f.to_sql(quote_char=self.quote_char, with_alias=True) for f in clauses.select_fields
                    )
                    select_sql = '*, ' + fields_sql
            else:
                fields_sql = ', '.join(
                    f.to_sql(quote_char=self.quote_char, with_alias=True) for f in clauses.select_fields
                )
                select_sql = fields_sql
        else:
            select_sql = '*'

        # Standard case: Build simple SQL - row order is preserved by executor
        from_sql = "__df__"

        # If we have temp aliases from where_ops, wrap in subquery to rename back
        if where_temp_alias_map:
            # Inner query with temp aliases
            inner_parts = [f"SELECT {select_sql}", f"FROM {from_sql}"]

            # WHERE clause
            if clauses.where_conditions:
                combined = clauses.where_conditions[0]
                for cond in clauses.where_conditions[1:]:
                    combined = combined & cond
                inner_parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

            inner_sql = ' '.join(inner_parts)

            # Outer query to rename temp aliases back to original names
            outer_select = ', '.join(
                f'{self.quote_char}{temp}{self.quote_char} AS {self.quote_char}{orig}{self.quote_char}'
                for temp, orig in where_temp_alias_map.items()
            )
            # _row_id is added automatically by connection.query_df
            parts = [f"SELECT {outer_select}", f"FROM ({inner_sql}) AS __case_subq__"]
        else:
            parts = [f"SELECT {select_sql}", f"FROM {from_sql}"]

        # WHERE clause
        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # GROUP BY clause
        if groupby_fields:
            groupby_sql = ', '.join(f.to_sql(quote_char=self.quote_char) for f in groupby_fields)
            parts.append(f"GROUP BY {groupby_sql}")

            # Handle dropna for groupby: add WHERE ... IS NOT NULL for groupby columns
            # when dropna=True (pandas default). This must be added BEFORE GROUP BY.
            if plan.groupby_agg and getattr(plan.groupby_agg, 'dropna', True):
                # Build IS NOT NULL conditions for groupby columns
                dropna_conditions = []
                for col in plan.groupby_agg.groupby_cols:
                    dropna_conditions.append(f'"{col}" IS NOT NULL')
                if dropna_conditions:
                    dropna_filter = ' AND '.join(dropna_conditions)
                    # Find WHERE clause index in parts and add to it, or insert new WHERE
                    where_idx = None
                    for i, part in enumerate(parts):
                        if part.startswith('WHERE '):
                            where_idx = i
                            break
                    if where_idx is not None:
                        # Append to existing WHERE
                        parts[where_idx] = parts[where_idx] + f' AND {dropna_filter}'
                    else:
                        # Find GROUP BY and insert WHERE before it
                        groupby_idx = None
                        for i, part in enumerate(parts):
                            if part.startswith('GROUP BY'):
                                groupby_idx = i
                                break
                        if groupby_idx is not None:
                            parts.insert(groupby_idx, f'WHERE {dropna_filter}')

        # ORDER BY clause
        # Use chDB's built-in _row_id virtual column (available in v4.0.0b5+) to preserve row order
        # This is explicitly added here rather than relying solely on connection.query_df
        # to ensure consistent behavior across all platforms
        if clauses.orderby_fields:
            # Build ORDER BY without stable sort modifier
            orderby_sql = build_orderby_clause(clauses.orderby_fields, self.quote_char, stable=False)
            parts.append(f"ORDER BY {orderby_sql}")
        elif plan.groupby_agg and plan.groupby_agg.sort:
            # GroupBy with sort=True (default): order by group keys
            orderby_cols = [(Field(col), True) for col in plan.groupby_agg.groupby_cols]
            orderby_sql = build_orderby_clause(orderby_cols, self.quote_char, stable=False)
            parts.append(f"ORDER BY {orderby_sql}")
        elif not plan.groupby_agg:
            # No explicit ORDER BY and no GROUP BY: preserve original row order using _row_id
            # This ensures filter operations maintain pandas-like row order semantics
            parts.append("ORDER BY _row_id")

        # LIMIT clause
        if clauses.limit_value is not None:
            parts.append(f"LIMIT {clauses.limit_value}")

        # OFFSET clause
        if clauses.offset_value is not None:
            parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(parts)

    def _check_limit_before_where(self, ops: List[LazyOp]) -> bool:
        """
        Check if LIMIT appears before WHERE in the ops list.

        This is important for Pandas semantics: df[:n][cond] means
        "take first n rows, then filter" which differs from SQL's
        "filter, then take first n rows".

        Args:
            ops: List of lazy operations

        Returns:
            True if LIMIT appears before any WHERE in the list
        """
        from .lazy_ops import LazyRelationalOp

        limit_idx = None
        where_idx = None

        for i, op in enumerate(ops):
            if isinstance(op, LazyRelationalOp):
                if op.op_type == 'LIMIT' and limit_idx is None:
                    limit_idx = i
                elif op.op_type == 'WHERE' and where_idx is None:
                    where_idx = i

        # LIMIT before WHERE if both exist and LIMIT comes first
        if limit_idx is not None and where_idx is not None:
            return limit_idx < where_idx

        return False

    def _build_nested_sql_for_dataframe(self, layers: List[List[LazyOp]]) -> str:
        """
        Build nested subquery SQL for DataFrame execution with multiple layers.

        Each layer becomes a subquery wrapping the previous one:
        Layer 0: SELECT * FROM __df__ WHERE value > 20 ORDER BY a DESC LIMIT 30
        Layer 1: SELECT * FROM (layer0) WHERE value > 60 ORDER BY a DESC LIMIT 10
        Layer 2: SELECT * FROM (layer1) WHERE value > 75 ORDER BY a DESC

        ORDER BY from inner layers is preserved in outer layers to maintain consistent
        row ordering after filters. This ensures patterns like
        df.sort_values('a').head(7)[df['a'] > 4] return rows in the sorted order.

        Args:
            layers: List of operation layers from QueryPlan

        Returns:
            Nested SQL query string
        """
        # Build innermost query from layer 0
        inner_clauses = extract_clauses_from_ops(layers[0], self.quote_char)
        # add_row_order=True ensures LIMIT/OFFSET with no explicit ORDER BY gets ORDER BY _row_id
        sql = self._assemble_simple_sql("__df__", inner_clauses, add_row_order=True)

        # Track ORDER BY from inner layer to preserve in outer layers
        preserved_orderby = inner_clauses.orderby_fields

        # Wrap with outer layers
        for layer_idx, layer_ops in enumerate(layers[1:], 1):
            layer_clauses = extract_clauses_from_ops(layer_ops, self.quote_char)
            subq_alias = f"__subq{layer_idx}__"

            # If this layer has ORDER BY, use it; otherwise preserve from inner layers
            if layer_clauses.orderby_fields:
                preserved_orderby = layer_clauses.orderby_fields

            # Each layer also needs deterministic ordering for LIMIT/OFFSET
            sql = self._assemble_simple_sql(
                f"({sql}) AS {subq_alias}", layer_clauses, add_row_order=True, preserved_orderby=preserved_orderby
            )

        return sql

    def _assemble_simple_sql(
        self,
        from_source: str,
        clauses: ExtractedClauses,
        add_row_order: bool = False,
        preserved_orderby: List[Tuple[Any, bool]] = None,
    ) -> str:
        """
        Assemble a simple SQL query from a source and clauses.

        Args:
            from_source: The FROM clause source (table name or subquery)
            clauses: Extracted SQL clauses
            add_row_order: If True and there's LIMIT/OFFSET without explicit ORDER BY,
                           add ORDER BY _row_id to preserve pandas-like row order
            preserved_orderby: ORDER BY fields from inner layers to preserve sort order

        Returns:
            SQL query string
        """
        # Build SELECT clause - include computed columns if present
        if clauses.select_fields:
            fields_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in clauses.select_fields)
            parts = [f"SELECT *, {fields_sql}"]
        else:
            parts = ["SELECT *"]
        parts.append(f"FROM {from_source}")

        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # Add ORDER BY - either explicit, preserved from inner, or _row_id for row order preservation
        # Priority: explicit ORDER BY > preserved ORDER BY > _row_id fallback
        orderby_to_use = clauses.orderby_fields if clauses.orderby_fields else preserved_orderby

        if orderby_to_use:
            orderby_sql = build_orderby_clause(orderby_to_use, self.quote_char, stable=False)
            parts.append(f"ORDER BY {orderby_sql}")
        elif add_row_order:
            # Add ORDER BY _row_id to preserve pandas-like row order
            # This ensures filter operations maintain original row order
            # _row_id is a built-in virtual column in chDB v4.0.0b5+
            parts.append("ORDER BY _row_id")

        if clauses.limit_value is not None:
            parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(parts)
