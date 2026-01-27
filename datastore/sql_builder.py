"""
SQL Builder for LazyColumnAssignment SQL Pushdown.

This module provides a smart SQL builder that handles computed column pushdown
using subquery wrapping strategy. It supports:

1. Unknown schema scenarios (uses SELECT *)
2. Column override (uses EXCEPT syntax)
3. Computed column references (auto-wraps subqueries)
4. Column selection (explicit column list in outer layer)

Design Philosophy:
- Trust ClickHouse optimizer to handle nested subqueries efficiently
- Use SELECT * to preserve all original columns when schema is unknown
- Use SELECT * EXCEPT(col) for column override scenarios
- Wrap subquery whenever a computed column is referenced

Example SQL Generation:

    # Simple computed column
    ds['c'] = ds['a'] + ds['b']
    -> SELECT *, ("a" + "b") AS "c" FROM source

    # Computed column + filter referencing it
    ds['c'] = ds['a'] + ds['b']
    ds = ds.filter(ds['c'] > 100)
    -> SELECT * FROM (
           SELECT *, ("a" + "b") AS "c" FROM source
       ) AS __subq1__ WHERE "c" > 100

    # Column override (when schema is known)
    ds['value'] = ds['value'] * 2
    -> SELECT * EXCEPT("value"), ("value" * 2) AS "value" FROM source
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Set, Any
from copy import copy

from .expressions import Expression, Field
from .conditions import Condition, CompoundCondition, BinaryCondition, NotCondition
from .conditions import InCondition, BetweenCondition, UnaryCondition


@dataclass
class SQLLayer:
    """
    Represents a single SQL subquery layer.

    SQL Generation Strategy:
    - Uses SELECT * to preserve all original columns (unknown schema support)
    - When overriding columns with known_column_order, uses explicit column list
      to preserve column order (replacing overridden columns in-place)
    - Falls back to EXCEPT syntax only when column order is unknown
    - Supports explicit column selection for final output

    Attributes:
        source: Data source - table function string or inner SQLLayer
        computed_columns: List of (column_name, expression) tuples
        except_columns: Columns to exclude from SELECT * (for override)
        explicit_columns: If set, use explicit column list instead of SELECT *
        known_column_order: Original column order for preserving positions
        where_conditions: WHERE conditions
        orderby_fields: ORDER BY fields with ascending flags
        limit_value: LIMIT value
        offset_value: OFFSET value
        groupby_fields: GROUP BY fields
    """

    # Data source: table function/name OR inner SQLLayer
    source: Union[str, "SQLLayer"]

    # Computed columns: (column_name, expression)
    computed_columns: List[Tuple[str, Expression]] = field(default_factory=list)

    # Columns to exclude from SELECT * (for column override)
    except_columns: Set[str] = field(default_factory=set)

    # Explicit column selection (if set, don't use SELECT *)
    explicit_columns: Optional[List[str]] = None

    # Original column order for preserving positions during override
    known_column_order: Optional[List[str]] = None

    # SQL clauses
    where_conditions: List[Condition] = field(default_factory=list)
    orderby_fields: List[Tuple[Field, bool]] = field(default_factory=list)
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None
    groupby_fields: List[Field] = field(default_factory=list)

    # Row order preservation flag
    preserve_row_order: bool = False

    def get_computed_column_names(self) -> Set[str]:
        """Get the set of computed column names in this layer."""
        return {name for name, _ in self.computed_columns}

    def to_sql(self, quote_char: str = '"', alias_counter: int = 0) -> str:
        """
        Render this layer as a SQL string.

        Args:
            quote_char: Quote character for identifiers
            alias_counter: Counter for generating unique subquery aliases

        Returns:
            SQL query string
        """
        parts = []

        # === SELECT clause ===
        select_items = self._build_select_items(quote_char)
        parts.append(f"SELECT {', '.join(select_items)}")

        # === FROM clause ===
        from_clause = self._build_from_clause(quote_char, alias_counter)
        parts.append(from_clause)

        # === WHERE clause ===
        if self.where_conditions:
            where_clause = self._build_where_clause(quote_char)
            parts.append(where_clause)

        # === GROUP BY clause ===
        if self.groupby_fields:
            groupby_sql = ", ".join(
                f.to_sql(quote_char=quote_char) for f in self.groupby_fields
            )
            parts.append(f"GROUP BY {groupby_sql}")

        # === ORDER BY clause ===
        if self.orderby_fields:
            orderby_clause = self._build_orderby_clause(quote_char)
            parts.append(orderby_clause)
        elif self.preserve_row_order and not self.groupby_fields:
            # Preserve original row order using rowNumberInAllBlocks()
            parts.append("ORDER BY rowNumberInAllBlocks()")

        # === LIMIT / OFFSET ===
        if self.limit_value is not None:
            parts.append(f"LIMIT {self.limit_value}")
        if self.offset_value is not None:
            parts.append(f"OFFSET {self.offset_value}")

        return " ".join(parts)

    def _build_select_items(self, quote_char: str) -> List[str]:
        """Build SELECT clause items."""
        select_items = []

        if self.explicit_columns is not None:
            # Explicit column selection mode
            for col in self.explicit_columns:
                select_items.append(f"{quote_char}{col}{quote_char}")
        else:
            # Build computed column lookup for in-place replacement
            computed_by_name = {name: expr for name, expr in self.computed_columns}

            if self.except_columns and self.known_column_order:
                # Use explicit column list to preserve order
                # Replace overridden columns with their computed expressions in-place
                for col in self.known_column_order:
                    if col in computed_by_name:
                        # This column is being overridden - use the expression
                        expr = computed_by_name[col]
                        expr_sql = expr.to_sql(quote_char=quote_char)
                        select_items.append(
                            f"({expr_sql}) AS {quote_char}{col}{quote_char}"
                        )
                    else:
                        # Original column - keep as-is
                        select_items.append(f"{quote_char}{col}{quote_char}")

                # Add any NEW computed columns (not in original order) at the end
                for name, expr in self.computed_columns:
                    if name not in self.known_column_order:
                        expr_sql = expr.to_sql(quote_char=quote_char)
                        select_items.append(
                            f"({expr_sql}) AS {quote_char}{name}{quote_char}"
                        )

                return select_items
            elif self.except_columns:
                # Fallback: use EXCEPT when column order is unknown
                # Note: This may change column order
                except_list = ", ".join(
                    f"{quote_char}{c}{quote_char}" for c in sorted(self.except_columns)
                )
                select_items.append(f"* EXCEPT({except_list})")
            else:
                select_items.append("*")

        # Append computed columns
        for name, expr in self.computed_columns:
            expr_sql = expr.to_sql(quote_char=quote_char)
            select_items.append(f"({expr_sql}) AS {quote_char}{name}{quote_char}")

        return select_items

    def _build_from_clause(self, quote_char: str, alias_counter: int) -> str:
        """Build FROM clause."""
        if isinstance(self.source, SQLLayer):
            inner_sql = self.source.to_sql(quote_char, alias_counter)
            alias = f"__subq{alias_counter + 1}__"
            return f"FROM ({inner_sql}) AS {alias}"
        else:
            return f"FROM {self.source}"

    def _build_where_clause(self, quote_char: str) -> str:
        """Build WHERE clause by combining all conditions with AND."""
        combined = self.where_conditions[0]
        for cond in self.where_conditions[1:]:
            combined = combined & cond
        return f"WHERE {combined.to_sql(quote_char=quote_char)}"

    def _build_orderby_clause(self, quote_char: str) -> str:
        """Build ORDER BY clause."""
        orderby_items = []
        for f, asc in self.orderby_fields:
            f_sql = f.to_sql(quote_char=quote_char)
            orderby_items.append(f"{f_sql} {'ASC' if asc else 'DESC'}")
        return f"ORDER BY {', '.join(orderby_items)}"

    def copy(self) -> "SQLLayer":
        """Create a shallow copy of this layer."""
        return SQLLayer(
            source=self.source,
            computed_columns=list(self.computed_columns),
            except_columns=set(self.except_columns),
            explicit_columns=(
                list(self.explicit_columns) if self.explicit_columns else None
            ),
            known_column_order=(
                list(self.known_column_order) if self.known_column_order else None
            ),
            where_conditions=list(self.where_conditions),
            orderby_fields=list(self.orderby_fields),
            limit_value=self.limit_value,
            offset_value=self.offset_value,
            groupby_fields=list(self.groupby_fields),
            preserve_row_order=self.preserve_row_order,
        )


class SQLBuilder:
    """
    Smart SQL builder that handles computed column pushdown.

    Core Logic:
    - Uses SELECT * by default to preserve all columns (unknown schema support)
    - New columns: appended to SELECT
    - Column override: uses EXCEPT to exclude original column
    - Computed column reference: wraps current layer as subquery

    The key insight is that whenever an operation (WHERE, ORDER BY) references
    a computed column from the current layer, we need to wrap that layer as a
    subquery first. This is because SQL doesn't allow referencing SELECT aliases
    in the same level's WHERE/ORDER BY clauses.

    Example:
        builder = SQLBuilder("file('data.csv', 'CSVWithNames')")
        builder.add_computed_column('c', expr_a_plus_b)
        builder.add_filter(condition_c_gt_100)  # auto-wraps subquery
        sql = builder.build()
        # SELECT * FROM (SELECT *, (a+b) AS c FROM ...) AS __subq1__ WHERE c > 100
    """

    def __init__(self, base_source: str, known_columns: List[str] = None):
        """
        Initialize the SQL builder.

        Args:
            base_source: Base data source (table function or table name SQL)
            known_columns: Known column names (optional, for optimization)
        """
        self._alias_counter = 0
        self._known_columns = set(known_columns) if known_columns else set()
        self.current_layer = SQLLayer(source=base_source)
        # Store column order for preserving positions
        self._known_column_order = list(known_columns) if known_columns else None
        self.current_layer.known_column_order = self._known_column_order
        self._preserve_row_order = False

    def set_preserve_row_order(self, value: bool) -> "SQLBuilder":
        """Set whether to preserve row order in the final output."""
        self._preserve_row_order = value
        return self

    def add_computed_column(self, name: str, expr: Expression) -> "SQLBuilder":
        """
        Add a computed column.

        Handles five scenarios:
        1. New column (no reference to computed): directly append
        2. Override known original column: use EXCEPT
        3. Override same-layer computed column: wrap subquery first
        4. Expression references current-layer computed column: wrap first
        5. Column name is referenced in existing WHERE conditions: wrap first
           (ClickHouse quirk: SELECT alias shadows original column in WHERE)

        Args:
            name: Column name
            expr: Expression for computing the column

        Returns:
            self for chaining
        """
        current_computed = self.current_layer.get_computed_column_names()

        # Case 3: Override same-layer computed column -> need to wrap
        if name in current_computed:
            self._wrap_current_layer()
            # After wrapping, the computed column becomes an "existing" column
            # Need EXCEPT to override it
            self.current_layer.except_columns.add(name)
        else:
            # Case 4: Expression references current-layer computed column -> wrap first
            referenced = self._extract_referenced_columns(expr)
            if referenced & current_computed:
                self._wrap_current_layer()
                # After wrapping, referenced columns are now "existing" columns
                # No EXCEPT needed since we're adding a new column

        # Case 5: Column name is referenced in existing WHERE conditions
        # This handles ClickHouse quirk where SELECT alias shadows original column in WHERE
        # Example: SELECT value*2 AS value WHERE value > 15
        #   ClickHouse incorrectly uses computed value*2 in WHERE instead of original value
        #   Fix: wrap to apply WHERE on original column first, then compute in outer query
        columns_in_where = self._get_columns_in_current_where()
        if name in columns_in_where:
            self._wrap_current_layer()

        # Case 2: Override known original column -> use EXCEPT
        # Check after potential wrap, as known_columns may have changed
        if name in self._known_columns:
            self.current_layer.except_columns.add(name)

        # Case 1: New column -> just append
        # (If schema unknown and column might exist, chDB will auto-rename)

        self.current_layer.computed_columns.append((name, expr))
        return self

    def _get_columns_in_current_where(self) -> Set[str]:
        """
        Get column names referenced in current layer's WHERE conditions.

        Returns:
            Set of column names referenced in WHERE conditions
        """
        columns = set()
        for condition in self.current_layer.where_conditions:
            columns.update(self._extract_referenced_columns(condition))
        return columns

    def add_filter(self, condition: Condition) -> "SQLBuilder":
        """
        Add a filter condition.

        If the condition references any computed column in the current layer,
        we need to wrap the current layer as a subquery first.

        Args:
            condition: Filter condition

        Returns:
            self for chaining
        """
        referenced = self._extract_referenced_columns(condition)
        current_computed = self.current_layer.get_computed_column_names()

        if referenced & current_computed:
            # References computed column -> need to wrap
            self._wrap_current_layer()

        self.current_layer.where_conditions.append(condition)
        return self

    def add_orderby(self, fields: List[Tuple[Field, bool]]) -> "SQLBuilder":
        """
        Add ORDER BY clause.

        If any sort field references a computed column, wraps first.

        Args:
            fields: List of (field, ascending) tuples

        Returns:
            self for chaining
        """
        referenced = set()
        for f, _ in fields:
            referenced.update(self._extract_referenced_columns(f))

        current_computed = self.current_layer.get_computed_column_names()

        if referenced & current_computed:
            self._wrap_current_layer()

        self.current_layer.orderby_fields = fields
        return self

    def add_limit(self, value: int) -> "SQLBuilder":
        """Add LIMIT clause."""
        self.current_layer.limit_value = value
        return self

    def add_offset(self, value: int) -> "SQLBuilder":
        """Add OFFSET clause."""
        self.current_layer.offset_value = value
        return self

    def add_groupby(self, fields: List[Field]) -> "SQLBuilder":
        """Add GROUP BY clause."""
        self.current_layer.groupby_fields = fields
        return self

    def select_columns(self, columns: List[str]) -> "SQLBuilder":
        """
        Set explicit column selection for the output.

        If there are computed columns, wraps first to ensure they are materialized.

        Args:
            columns: List of column names to select

        Returns:
            self for chaining
        """
        if self.current_layer.computed_columns:
            self._wrap_current_layer()

        # Set explicit column selection
        self.current_layer.explicit_columns = columns
        return self

    def _wrap_current_layer(self) -> None:
        """
        Wrap the current layer as a subquery, creating a new outer layer.

        After wrapping:
        - Current layer's computed columns become output columns of the subquery
        - New layer uses SELECT * to inherit all columns
        - Known columns are updated to include computed columns
        """
        self._alias_counter += 1

        # Update known columns: add current layer's computed columns
        new_known = self._known_columns | self.current_layer.get_computed_column_names()
        self._known_columns = new_known
        # Update column order - add new computed columns to the end
        if self._known_column_order is not None:
            new_cols = [
                c
                for c in self.current_layer.get_computed_column_names()
                if c not in self._known_column_order
            ]
            self._known_column_order = self._known_column_order + new_cols

        # Create new outer layer with current layer as source
        self.current_layer = SQLLayer(source=self.current_layer)

    def build(self, quote_char: str = '"') -> str:
        """
        Build the final SQL string.

        Args:
            quote_char: Quote character for identifiers

        Returns:
            SQL query string
        """
        # Apply row order preservation to final layer
        self.current_layer.preserve_row_order = self._preserve_row_order
        return self.current_layer.to_sql(quote_char, self._alias_counter)

    def _extract_referenced_columns(self, expr: Any) -> Set[str]:
        """
        Extract column names referenced in an expression.

        Recursively traverses the expression tree to find all Field references.

        Args:
            expr: Expression to analyze

        Returns:
            Set of column names referenced
        """
        columns = set()

        if expr is None:
            return columns

        if isinstance(expr, Field):
            # Field.name may have quotes, strip them
            name = expr.name.strip('"').strip("'")
            columns.add(name)

        elif isinstance(expr, BinaryCondition):
            columns.update(self._extract_referenced_columns(expr.left))
            columns.update(self._extract_referenced_columns(expr.right))

        elif isinstance(expr, CompoundCondition):
            columns.update(self._extract_referenced_columns(expr.left))
            columns.update(self._extract_referenced_columns(expr.right))

        elif isinstance(expr, NotCondition):
            columns.update(self._extract_referenced_columns(expr.condition))

        elif isinstance(expr, InCondition):
            columns.update(self._extract_referenced_columns(expr.expression))

        elif isinstance(expr, BetweenCondition):
            columns.update(self._extract_referenced_columns(expr.expression))
            columns.update(self._extract_referenced_columns(expr.lower))
            columns.update(self._extract_referenced_columns(expr.upper))

        elif isinstance(expr, UnaryCondition):
            columns.update(self._extract_referenced_columns(expr.expression))

        elif hasattr(expr, "expression"):
            # Generic expression wrapper
            columns.update(self._extract_referenced_columns(expr.expression))

        elif hasattr(expr, "left") and hasattr(expr, "right"):
            # Arithmetic expression or similar
            columns.update(self._extract_referenced_columns(expr.left))
            columns.update(self._extract_referenced_columns(expr.right))

        elif hasattr(expr, "nodes"):
            # Expression with nodes() method (like ArithmeticExpression)
            for node in expr.nodes():
                if isinstance(node, Field):
                    name = node.name.strip('"').strip("'")
                    columns.add(name)

        elif hasattr(expr, "args"):
            # Function with args
            for arg in expr.args:
                columns.update(self._extract_referenced_columns(arg))

        return columns

    def get_known_columns(self) -> Set[str]:
        """Get the current set of known columns (including computed ones)."""
        return self._known_columns | self.current_layer.get_computed_column_names()

    def get_computed_columns(self) -> Set[str]:
        """Get computed column names from the current layer."""
        return self.current_layer.get_computed_column_names()
