"""
StringAccessor - String functions via .str accessor.

Provides ClickHouse string functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from .base import BaseAccessor
from ..exceptions import UnsupportedOperationError
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..expressions import Expression


class StringAccessor(BaseAccessor):
    """
    Accessor for string functions via .str property.

    Maps to ClickHouse string functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['name'].str.upper()           # upper(name)
        >>> ds['name'].str.length()          # length(name)
        >>> ds['name'].str.substring(1, 5)   # substring(name, 1, 5)
        >>> ds['name'].str.replace('a', 'b') # replace(name, 'a', 'b')
        >>> ds['text'].str.split().str[0]    # Get first element after split

    ClickHouse String Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/string-functions
    """

    def __getitem__(self, index):
        """
        Get element at index from array result or slice a string.

        For integer index:
            Maps to arrayElement(arr, index). Note: ClickHouse uses 1-based indexing,
            but this method accepts 0-based indexing for pandas compatibility.

        For slice:
            Maps to substring(s, start+1, length) in ClickHouse.
            Supports str[:n], str[n:], str[n:m] syntax like pandas.

        Args:
            index: 0-based index or slice object

        Example:
            >>> ds['text'].str.split().str[0]  # Get first word
            >>> ds['text'].str.split().str[-1] # Get last word
            >>> ds['name'].str[:3]  # Get first 3 characters
            >>> ds['name'].str[2:]  # Skip first 2 characters
            >>> ds['name'].str[1:4]  # Characters 1-3 (0-indexed)
        """
        from ..functions import Function
        from ..expressions import Literal

        # Handle slice: str[:3], str[2:], str[1:4]
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop
            step = index.step

            if step is not None and step != 1:
                # Slices with step are not directly supported in SQL
                # Fall back to pandas execution
                raise UnsupportedOperationError(
                    operation="str[::step]",
                    reason="string slicing with step is not supported in SQL mode",
                    suggestion="Use pandas directly: series.str[::step] or convert to pandas first",
                )

            if start < 0:
                # Negative start - need to compute from end
                # Use substring with negative offset (ClickHouse supports this)
                if stop is None:
                    # str[-n:] - last n characters
                    return Function('substring', self._expr, Literal(start))
                else:
                    # Complex case with negative start and positive stop
                    raise UnsupportedOperationError(
                        operation="str[-n:m]",
                        reason="string slicing with negative start and positive stop not supported in SQL",
                        suggestion="Use pandas directly: series.str[-n:m]",
                    )

            # ClickHouse substring is 1-based
            ch_start = start + 1

            if stop is None:
                # str[n:] - from position n to end
                return Function('substring', self._expr, Literal(ch_start))
            else:
                # str[n:m] or str[:m] - from position n to m
                length = stop - start
                if length <= 0:
                    # Empty result
                    return Literal('')
                return Function('substring', self._expr, Literal(ch_start), Literal(length))

        # Handle integer index: array element access
        if index >= 0:
            ch_index = index + 1
        else:
            ch_index = index

        return Function('arrayElement', self._expr, Literal(ch_index))

    def get(self, i: int, alias=None):
        """
        Get character at index position.

        Maps to substring(s, i+1, 1) in ClickHouse.

        Args:
            i: 0-based character index (negative indices work from the end)
            alias: Optional alias for the result

        Example:
            >>> ds['name'].str.get(0)   # Get first character
            >>> ds['name'].str.get(-1)  # Get last character
        """
        from ..function_registry import FunctionRegistry

        spec = FunctionRegistry.get('str_get')
        return spec.sql_builder(self._expr, i, alias=alias)

    def count(self, pattern: str, alias=None):
        """Count occurrences of pattern in string. Alias for str_count."""
        from ..function_registry import FunctionRegistry

        spec = FunctionRegistry.get('str_count')
        return spec.sql_builder(self._expr, pattern, alias=alias)

    def join(self, sep: str = '', alias=None):
        """Join elements of array column with separator. Alias for str_join."""
        from ..function_registry import FunctionRegistry

        spec = FunctionRegistry.get('str_join')
        return spec.sql_builder(self._expr, sep, alias=alias)


# =============================================================================
# INJECT STRING METHODS FROM REGISTRY
# =============================================================================


def _inject_string_accessor_methods():
    """Inject string methods from registry into StringAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.STRING):
        # Skip if already exists
        if hasattr(StringAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(StringAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(StringAccessor, alias_name):
                setattr(StringAccessor, alias_name, getattr(StringAccessor, spec.name))


# Perform injection when module is loaded
_inject_string_accessor_methods()
