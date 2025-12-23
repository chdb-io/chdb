"""
StringAccessor - String functions via .str accessor.

Provides ClickHouse string functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from .base import BaseAccessor
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

    def __getitem__(self, index: int):
        """
        Get element at index from array result (e.g., after str.split()).

        Maps to arrayElement(arr, index). Note: ClickHouse uses 1-based indexing,
        but this method accepts 0-based indexing for pandas compatibility.

        Args:
            index: 0-based index (will be converted to 1-based for ClickHouse)

        Example:
            >>> ds['text'].str.split().str[0]  # Get first word
            >>> ds['text'].str.split().str[-1] # Get last word
        """
        from ..functions import Function
        from ..expressions import Literal

        # Convert 0-based to 1-based indexing for positive indices
        # Negative indices work the same in ClickHouse (-1 is last element)
        if index >= 0:
            ch_index = index + 1
        else:
            ch_index = index

        return Function('arrayElement', self._expr, Literal(ch_index))

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
