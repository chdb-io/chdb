"""
Utility functions and decorators for DataStore
"""

from typing import TypeVar, Callable, List, Tuple, Any, Optional, Dict, TYPE_CHECKING
from copy import copy
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .lazy_ops import LazyGroupByAgg

__all__ = [
    'immutable',
    'ignore_copy',
    'build_orderby_clause',
    'STABLE_SORT_TIEBREAKER',
    'STABLE_SORT_KINDS',
    'is_stable_sort',
    'ClauseExtractorResult',
    'extract_clauses_from_ops',
    'SQLBuildContext',
]

T = TypeVar('T')


def immutable(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that makes builder methods immutable.

    Each decorated method will:
    1. Create a shallow copy of self
    2. Execute the method on the copy
    3. Return the copy

    This ensures that the original object remains unchanged,
    enabling safe method chaining and thread-safe operations.

    Example:
        >>> class Builder:
        ...     def __init__(self):
        ...         self._items = []
        ...         self.is_immutable = True
        ...
        ...     @immutable
        ...     def add(self, item):
        ...         self._items.append(item)
        ...
        ...     def __copy__(self):
        ...         new = type(self).__new__(type(self))
        ...         new.__dict__.update(self.__dict__)
        ...         new._items = self._items.copy()
        ...         return new
        >>>
        >>> b1 = Builder()
        >>> b2 = b1.add(1)
        >>> b1._items  # Original unchanged
        []
        >>> b2._items  # Copy has the item
        [1]
    """

    def wrapper(self, *args, **kwargs):
        # Check if immutability is enabled (default: True)
        if getattr(self, 'is_immutable', True):
            self_copy = copy(self)
        else:
            self_copy = self

        # Execute method on copy
        result = func(self_copy, *args, **kwargs)

        # Return copy if method returns None, otherwise return result
        return self_copy if result is None else result

    return wrapper


def ignore_copy(func: Callable) -> Callable:
    """
    Decorator for __getattr__ to prevent infinite recursion during copy operations.

    When using copy() or deepcopy() on objects with __getattr__, Python looks for
    special methods like __copy__, __deepcopy__, __getstate__, etc. If __getattr__
    tries to handle these, it can cause infinite recursion.

    This decorator makes __getattr__ raise AttributeError for these special methods,
    allowing copy to work correctly.

    Example:
        >>> class DynamicObject:
        ...     @ignore_copy
        ...     def __getattr__(self, name):
        ...         # Dynamically create attributes
        ...         return f"dynamic_{name}"
        >>>
        >>> obj = DynamicObject()
        >>> obj.foo  # Works
        'dynamic_foo'
        >>> copy(obj)  # Also works, won't cause infinite recursion
        <DynamicObject object at 0x...>
    """

    def wrapper(self, name):
        # Special methods that copy looks for
        if name in [
            '__copy__',
            '__deepcopy__',
            '__getstate__',
            '__setstate__',
            '__getnewargs__',
            '__reduce__',
            '__reduce_ex__',
        ]:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return func(self, name)

    return wrapper


def format_identifier(name: str, quote_char: str = '"') -> str:
    """
    Format an identifier (table/column name) with quotes.

    Args:
        name: The identifier name
        quote_char: Quote character to use (default: ")

    Returns:
        Quoted identifier

    Example:
        >>> format_identifier("my_table")
        '"my_table"'
        >>> format_identifier("table", "`")
        '`table`'
    """
    if quote_char:
        return f"{quote_char}{name}{quote_char}"
    return name


# ========== Sort Stability Constants and Functions ==========
# These ensure DataStore sort matches pandas stable sort (kind='stable') behavior

# Tie-breaker for stable sort: rowNumberInAllBlocks() preserves original row order
# when sort keys have duplicate values. Performance overhead is minimal (~5%).
STABLE_SORT_TIEBREAKER = "rowNumberInAllBlocks() ASC"

# Sort kinds that guarantee stability (matching pandas behavior)
STABLE_SORT_KINDS = ('stable', 'mergesort')


def is_stable_sort(kind: str) -> bool:
    """Check if the given sort kind guarantees stability."""
    return kind in STABLE_SORT_KINDS


def build_orderby_clause(orderby_fields: list, quote_char: str = '"', stable: bool = False) -> str:
    """
    Build ORDER BY clause with optional stable sort tie-breaker.

    This ensures consistent ordering when sort keys have duplicate values,
    matching pandas sort_values(kind='stable') behavior.

    Args:
        orderby_fields: List of (field, ascending) tuples
        quote_char: Quote character for field names
        stable: If True, add tie-breaker for stable sort (default: False, matching pandas)

    Returns:
        ORDER BY clause string (without 'ORDER BY' prefix)

    Example:
        >>> build_orderby_clause([('name', True), ('age', False)])
        '"name" ASC, "age" DESC'
        >>> build_orderby_clause([('name', True)], stable=True)
        '"name" ASC, rowNumberInAllBlocks() ASC'
    """
    if not orderby_fields:
        return ""

    order_parts = []
    for field, asc in orderby_fields:
        direction = 'ASC' if asc else 'DESC'
        if hasattr(field, 'to_sql'):
            field_sql = field.to_sql(quote_char=quote_char)
        else:
            field_sql = f'{quote_char}{field}{quote_char}' if quote_char else str(field)
        order_parts.append(f"{field_sql} {direction}")

    # Add tie-breaker for stable sort (always ASC to preserve original row order)
    if stable:
        order_parts.append(STABLE_SORT_TIEBREAKER)

    return ', '.join(order_parts)


def format_alias(sql: str, alias: str = None, quote_char: str = '"', use_as: bool = True) -> str:
    """
    Format SQL with an optional alias.

    Args:
        sql: The SQL expression
        alias: Optional alias name
        quote_char: Quote character for alias
        use_as: Whether to use AS keyword

    Returns:
        SQL with alias if provided, otherwise original SQL

    Example:
        >>> format_alias("COUNT(*)", "total")
        'COUNT(*) AS "total"'
        >>> format_alias("name", "full_name", use_as=False)
        'name "full_name"'
    """
    if not alias:
        return sql

    as_keyword = ' AS ' if use_as else ' '
    quoted_alias = format_identifier(alias, quote_char)
    return f"{sql}{as_keyword}{quoted_alias}"


def normalize_ascending(ascending, field_count: int) -> list:
    """
    Normalize ascending parameter to a list matching field count.

    Args:
        ascending: bool or list of bools
        field_count: Number of fields

    Returns:
        List of bools with length matching field_count

    Example:
        >>> normalize_ascending(True, 3)
        [True, True, True]
        >>> normalize_ascending([True, False], 2)
        [True, False]
    """
    if isinstance(ascending, bool):
        return [ascending] * field_count
    return list(ascending)


# SQL aggregation function mapping (pandas -> SQL)
SQL_AGG_FUNC_MAP = {
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


def map_agg_func(func: str) -> str:
    """
    Map pandas aggregation function name to SQL equivalent.

    Args:
        func: Pandas function name (e.g., 'mean', 'sum')

    Returns:
        SQL function name (e.g., 'avg', 'sum')

    Example:
        >>> map_agg_func('mean')
        'avg'
        >>> map_agg_func('std')
        'stddevPop'
    """
    return SQL_AGG_FUNC_MAP.get(func, func)


# ========== SQL Build Context ==========


@dataclass
class ClauseExtractorResult:
    """Result of extracting SQL clauses from LazyOp list."""

    where_conditions: List[Any] = field(default_factory=list)
    orderby_fields: List[Tuple[Any, bool]] = field(default_factory=list)
    orderby_kind: str = 'quicksort'
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None


def extract_clauses_from_ops(ops: List[Any], quote_char: str) -> ClauseExtractorResult:
    """
    Extract WHERE, ORDER BY, LIMIT, OFFSET clauses from a list of LazyRelationalOp.

    This centralizes the clause extraction logic that was duplicated in
    _execute() and _build_execution_sql().

    Args:
        ops: List of lazy operations to extract clauses from
        quote_char: Quote character for identifiers

    Returns:
        ClauseExtractorResult with extracted clauses
    """
    from .lazy_ops import LazyRelationalOp
    from .expressions import Field

    result = ClauseExtractorResult()
    for op in ops:
        if isinstance(op, LazyRelationalOp):
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
    return result


@dataclass
class SQLBuildContext:
    """
    Context object containing all state needed for SQL query building.

    This encapsulates the intermediate state during SQL construction,
    eliminating the need for multiple local variables and reducing
    code duplication between _execute() and _build_execution_sql().
    """

    # Core SQL components
    select_fields: List[Any] = field(default_factory=list)
    where_conditions: List[Any] = field(default_factory=list)
    orderby_fields: List[Tuple[Any, bool]] = field(default_factory=list)
    orderby_kind: str = 'quicksort'
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None

    # GROUP BY components
    groupby_fields: List[Any] = field(default_factory=list)
    groupby_agg: Optional['LazyGroupByAgg'] = None

    # WHERE/CASE WHEN special handling
    where_needs_subquery: bool = False
    where_needs_temp_alias: bool = False
    where_temp_alias_columns: List[Tuple[str, str]] = field(default_factory=list)

    # Alias renames for conflict resolution
    alias_renames: Dict[str, str] = field(default_factory=dict)

    # Row order preservation
    needs_row_order: bool = False

    # Multi-layer query state
    layers: List[List[Any]] = field(default_factory=list)

    # Table metadata (for SQL building)
    quote_char: str = '"'

    def is_simple_query(self) -> bool:
        """Check if this is a simple (non-nested) query."""
        return len(self.layers) <= 1

    def has_where_ops(self) -> bool:
        """Check if there are WHERE/CASE WHEN operations requiring special handling."""
        return self.where_needs_subquery or self.where_needs_temp_alias
