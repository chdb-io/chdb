"""
ArrayAccessor - Array functions via .arr accessor.

Provides ClickHouse array functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class ArrayAccessor(BaseAccessor):
    """
    Accessor for array functions via .arr property.

    Maps to ClickHouse array functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['tags'].arr.length()           # length(tags)
        >>> ds['nums'].arr.sum()              # arraySum(nums)
        >>> ds['items'].arr.join(',')         # arrayStringConcat(items, ',')
        >>> ds['arr'].arr.first()             # arrayElement(arr, 1)
        >>> ds['arr'].arr.has('value')        # has(arr, 'value')

    ClickHouse Array Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/array-functions
    """

    # Common properties for convenience
    @property
    def length(self) -> 'Function':
        """Get array length. Maps to length(arr)."""
        return self._create_function('length')

    @property
    def size(self) -> 'Function':
        """Alias for length."""
        return self.length

    @property
    def empty(self) -> 'Function':
        """Check if array is empty. Maps to empty(arr)."""
        return self._create_function('empty')

    @property
    def not_empty(self) -> 'Function':
        """Check if array is not empty. Maps to notEmpty(arr)."""
        return self._create_function('notEmpty')

    # Standard methods will be injected from registry below


# =============================================================================
# INJECT ARRAY METHODS FROM REGISTRY
# =============================================================================


def _inject_array_accessor_methods():
    """Inject array methods from registry into ArrayAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.ARRAY):
        # Skip if already exists (properties defined above)
        if hasattr(ArrayAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(ArrayAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(ArrayAccessor, alias_name):
                setattr(ArrayAccessor, alias_name, getattr(ArrayAccessor, spec.name))


# Perform injection when module is loaded
_inject_array_accessor_methods()
