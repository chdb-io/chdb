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

    ClickHouse String Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/string-functions
    """

    # Methods will be injected from registry below
    pass


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
