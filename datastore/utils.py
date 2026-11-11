"""
Utility functions and decorators for DataStore.

This module contains general-purpose utilities that are not specific to any
particular subsystem. SQL-specific utilities are in sql_executor.py.
"""

from typing import TypeVar, Callable
from contextlib import contextmanager
from copy import copy
import contextvars

__all__ = [
    'immutable',
    'ignore_copy',
    'format_identifier',
    'format_alias',
    'normalize_ascending',
    'build_orderby_clause',
    'is_stable_sort',
    'map_agg_func',
    'STABLE_SORT_TIEBREAKER',
    'STABLE_SORT_KINDS',
    'SQL_AGG_FUNC_MAP',
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
    Format an identifier (table/column name) with quotes, properly escaping special characters.

    Args:
        name: The identifier name
        quote_char: Quote character to use (default: ")

    Returns:
        Quoted identifier with special characters escaped

    Example:
        >>> format_identifier("my_table")
        '"my_table"'
        >>> format_identifier("table", "`")
        '`table`'
        >>> format_identifier('col"quote')
        '"col""quote"'
        >>> format_identifier('col\\\\backslash')
        '"col\\\\backslash"'
    """
    if quote_char:
        # Escape backslashes first (must be done before other escapes)
        escaped_name = name.replace('\\', '\\\\')
        # Escape the quote character by doubling it (SQL standard)
        escaped_name = escaped_name.replace(quote_char, quote_char + quote_char)
        return f"{quote_char}{escaped_name}{quote_char}"
    return name


# ========== Sort Stability Constants and Functions ==========
# These ensure DataStore sort matches pandas stable sort (kind='stable') behavior

# Position of a row in a source without its own row positions (database
# tables, generated rows): the order the rows reach the query.
DEFAULT_ROW_POSITION = "rowNumberInAllBlocks()"

# Tie-breaker for stable sort to preserve original row order when sort keys have duplicates.
STABLE_SORT_TIEBREAKER = f"{DEFAULT_ROW_POSITION} ASC"


class RowPositionScope:
    """
    The SQL for a row's position in the source of the query being built:
    ``_row_id`` for a DataFrame source, the file reader's position for a
    file (see ``TableFunction.row_position_source``), ``DEFAULT_ROW_POSITION``
    otherwise. ``uses`` counts how often the built SQL read it.
    """

    def __init__(self, sql: str):
        self.sql = sql
        self.uses = 0

    @property
    def used(self) -> bool:
        return self.uses > 0


_row_position_scope = contextvars.ContextVar("row_position_scope", default=None)


def row_position_sql() -> str:
    """SQL for a row's position in its source, in the query being built."""
    scope = _row_position_scope.get()
    if scope is None:
        return DEFAULT_ROW_POSITION
    scope.uses += 1
    return scope.sql


def current_row_position_scope() -> "RowPositionScope":
    """The row position scope SQL is being built in, or None."""
    return _row_position_scope.get()


@contextmanager
def row_position_scope(sql: str):
    """Build SQL in which a row's source position is ``sql``."""
    scope = RowPositionScope(sql)
    token = _row_position_scope.set(scope)
    try:
        yield scope
    finally:
        _row_position_scope.reset(token)


def dataframe_sql(expr, quote_char: str = '"') -> str:
    """
    SQL for ``expr`` evaluated on a DataFrame through Python(), where a
    row's source position is the DataFrame's virtual ``_row_id`` column.
    """
    with row_position_scope("_row_id"):
        return expr.to_sql(quote_char=quote_char)


def compose_slice(offset, limit, op_type: str, n: int):
    """
    ``(offset, limit)`` of the rows an ``op_type`` (``"OFFSET"`` or
    ``"LIMIT"``) of ``n`` keeps from the rows ``OFFSET offset LIMIT limit``
    selects. Each applies to the previous result, as ``df.iloc[n:]`` and
    ``df.head(n)`` do, so any chain of offsets and limits is one slice.
    ``None`` means no offset or no limit.
    """
    if op_type == "OFFSET":
        offset = (offset or 0) + n
        if limit is not None:
            limit = max(limit - n, 0)
    elif op_type == "LIMIT":
        limit = n if limit is None else min(limit, n)
    return offset, limit


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
    # In performance mode, skip tiebreaker for throughput
    from .config import is_performance_mode

    if stable and not is_performance_mode():
        order_parts.append(f"{row_position_sql()} ASC")

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
    'std': 'stddevSamp',
    'var': 'varSamp',
    'first': 'any',
    'last': 'anyLast',
    'nunique': 'uniqExact',
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
