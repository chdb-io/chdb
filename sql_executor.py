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
    groupby_agg: LazyGroupByAgg,
    alias_renames: Dict[str, str] = None,
    all_columns: List[str] = None,
) -> Tuple[List[Field], List[Expression]]:
    """
    Build GROUP BY and SELECT fields from a LazyGroupByAgg operation.

    This centralizes the aggregation SQL building logic.

    Args:
        groupby_agg: LazyGroupByAgg operation
        alias_renames: Dict for alias conflict resolution
        all_columns: All column names (needed for count() to generate COUNT(col) per column)

    Returns:
        Tuple of (groupby_fields, select_fields)
    """
    from .functions import AggregateFunction

    alias_renames = alias_renames or {}

    # Build GROUP BY fields
    groupby_fields = [Field(col) for col in groupby_agg.groupby_cols]

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

            agg_expr = AggregateFunction(sql_func, Field(col), alias=final_alias)
            select_fields.append(agg_expr)

    elif groupby_agg.agg_dict:
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

        if func == 'size':
            # size() counts ALL rows including NULL -> COUNT(*)
            select_fields.append(AggregateFunction(sql_func, Star()))
        elif func == 'count':
            # count() counts NON-NULL values per column -> COUNT(column)
            # This matches pandas behavior: df.groupby('x').count() excludes NaN
            if all_columns:
                # Get non-groupby columns
                non_groupby_cols = [c for c in all_columns if c not in groupby_agg.groupby_cols]
                for col in non_groupby_cols:
                    # Check if this alias conflicts with WHERE columns
                    # If so, use a temp alias to avoid "Aggregate function found in WHERE" error
                    temp_alias = f"__agg_{col}__"
                    if temp_alias in alias_renames:
                        alias = temp_alias
                    else:
                        alias = col
                    agg_expr = AggregateFunction(sql_func, Field(col), alias=alias)
                    select_fields.append(agg_expr)
            else:
                # Fallback: use COUNT(*) if we don't know the columns
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
            # Get all columns for count() to generate COUNT(col) per column
            all_cols = self.ds._get_all_column_names() if hasattr(self.ds, '_get_all_column_names') else None
            groupby_fields_for_sql, select_fields_for_sql = build_groupby_select_fields(
                groupby_agg_op, alias_renames, all_columns=all_cols
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
            return self.build_sql_with_row_order_subquery(clauses, select_fields_for_sql, groupby_fields_for_sql)
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

    def _build_nested_query_from_plan(self, layers: List[List[LazyOp]], sql_select_fields: List[Expression]) -> str:
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
        )

        # Wrap with outer layers
        for layer_idx, layer_ops in enumerate(layers[1:], 1):
            layer_clauses = extract_clauses_from_ops(layer_ops, self.quote_char)
            sql = self._wrap_with_layer(sql, layer_clauses, layer_idx)

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
            if self.ds._select_star:
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

        # Handle GroupBy SQL pushdown: set group keys as index
        # Exception: when using named_agg, keep columns as regular columns
        # (matching Pandas behavior where reset_index() was called)
        if plan.groupby_agg and plan.groupby_agg.groupby_cols:
            groupby_cols = plan.groupby_agg.groupby_cols
            # Don't set index for named_agg - it keeps columns as regular columns
            if plan.groupby_agg.named_agg is None:
                if all(col in result_df.columns for col in groupby_cols):
                    result_df = result_df.set_index(groupby_cols)
                    self._logger.debug("  Set groupby columns as index: %s", groupby_cols)

        # Handle alias renames
        if plan.alias_renames:
            rename_back = {temp: orig for temp, orig in plan.alias_renames.items() if temp in result_df.columns}
            if rename_back:
                result_df = result_df.rename(columns=rename_back)
                self._logger.debug("  Renamed temp aliases: %s", rename_back)

        return result_df

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
            df_columns = [c for c in df.columns if c != '__row_idx__']
            groupby_fields, select_fields = build_groupby_select_fields(
                plan.groupby_agg, plan.alias_renames, all_columns=df_columns
            )

        if plan.where_ops:
            all_columns = list(df.columns)
            for col in all_columns:
                if col == '__row_idx__':
                    continue
                col_type = schema.get(col, str(df[col].dtype))
                case_expr = CaseWhenExpr(col, plan.where_ops, self.quote_char, col_type)
                select_fields.append(case_expr)

        # Determine SELECT clause
        # Priority: CASE WHEN/GroupBy fields -> explicit column selection -> *
        if select_fields:
            select_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in select_fields)
            if not groupby_fields:
                select_sql += ', "__row_idx__"'
        elif clauses.select_fields:
            select_sql = ', '.join(f.to_sql(quote_char=self.quote_char, with_alias=True) for f in clauses.select_fields)
        else:
            select_sql = '*'

        # Standard case: Build simple SQL - row order is preserved by executor
        from_sql = "__df__"
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

        # ORDER BY clause
        # For SQL on DataFrame, we need a different approach for stable sort:
        # 1. Add __row_idx__ column (handled by query_df)
        # 2. Use __row_idx__ as tie-breaker instead of rowNumberInAllBlocks()
        if clauses.orderby_fields:
            # Build ORDER BY without stable sort modifier
            orderby_sql = build_orderby_clause(clauses.orderby_fields, self.quote_char, stable=False)
            parts.append(f"ORDER BY {orderby_sql}")
        elif plan.groupby_agg and plan.groupby_agg.sort:
            # GroupBy with sort=True (default): order by group keys
            orderby_cols = [(Field(col), True) for col in plan.groupby_agg.groupby_cols]
            orderby_sql = build_orderby_clause(orderby_cols, self.quote_char, stable=False)
            parts.append(f"ORDER BY {orderby_sql}")

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
        Layer 0: SELECT * FROM __df__ WHERE value > 20 ORDER BY __row_idx__ LIMIT 30
        Layer 1: SELECT * FROM (layer0) WHERE value > 60 ORDER BY __row_idx__ LIMIT 10
        Layer 2: SELECT * FROM (layer1) WHERE value > 75

        ORDER BY __row_idx__ is automatically added when LIMIT/OFFSET is present
        without explicit ORDER BY to ensure deterministic results matching pandas semantics.

        Args:
            layers: List of operation layers from QueryPlan

        Returns:
            Nested SQL query string
        """
        # Build innermost query from layer 0
        inner_clauses = extract_clauses_from_ops(layers[0], self.quote_char)
        # add_row_order=True ensures LIMIT/OFFSET with no explicit ORDER BY gets ORDER BY __row_idx__
        sql = self._assemble_simple_sql("__df__", inner_clauses, add_row_order=True)

        # Wrap with outer layers
        for layer_idx, layer_ops in enumerate(layers[1:], 1):
            layer_clauses = extract_clauses_from_ops(layer_ops, self.quote_char)
            subq_alias = f"__subq{layer_idx}__"
            # Each layer also needs deterministic ordering for LIMIT/OFFSET
            sql = self._assemble_simple_sql(f"({sql}) AS {subq_alias}", layer_clauses, add_row_order=True)

        return sql

    def _assemble_simple_sql(self, from_source: str, clauses: ExtractedClauses, add_row_order: bool = False) -> str:
        """
        Assemble a simple SQL query from a source and clauses.

        Args:
            from_source: The FROM clause source (table name or subquery)
            clauses: Extracted SQL clauses
            add_row_order: If True and there's LIMIT/OFFSET without explicit ORDER BY,
                           add ORDER BY __row_idx__ to preserve pandas-like row order

        Returns:
            SQL query string
        """
        parts = ["SELECT *"]
        parts.append(f"FROM {from_source}")

        if clauses.where_conditions:
            combined = clauses.where_conditions[0]
            for cond in clauses.where_conditions[1:]:
                combined = combined & cond
            parts.append(f"WHERE {combined.to_sql(quote_char=self.quote_char)}")

        # Add ORDER BY - either explicit or for LIMIT/OFFSET determinism
        if clauses.orderby_fields:
            orderby_sql = build_orderby_clause(clauses.orderby_fields, self.quote_char, stable=False)
            parts.append(f"ORDER BY {orderby_sql}")
        elif add_row_order and (clauses.limit_value is not None or clauses.offset_value is not None):
            # Add ORDER BY __row_idx__ for LIMIT/OFFSET without explicit ORDER BY
            # This preserves pandas-like row order (original DataFrame row positions)
            parts.append("ORDER BY __row_idx__")

        if clauses.limit_value is not None:
            parts.append(f"LIMIT {clauses.limit_value}")

        if clauses.offset_value is not None:
            parts.append(f"OFFSET {clauses.offset_value}")

        return ' '.join(parts)
