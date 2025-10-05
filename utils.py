"""
Utility functions and decorators for DataStore
"""

from typing import TypeVar, Callable
from copy import copy

__all__ = ['immutable', 'ignore_copy']

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

