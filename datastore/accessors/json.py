"""
JsonAccessor - JSON functions via .json accessor.

Provides ClickHouse JSON functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class JsonAccessor(BaseAccessor):
    """
    Accessor for JSON functions via .json property.

    Maps to ClickHouse JSON functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['data'].json.get_string('name')     # JSONExtractString(data, 'name')
        >>> ds['data'].json.get_int('age')         # JSONExtractInt(data, 'age')
        >>> ds['data'].json.is_valid()             # isValidJSON(data)
        >>> ds['data'].json.type()                 # JSONType(data)

    ClickHouse JSON Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/json-functions
    """

    # Standard methods will be injected from registry below
    pass


# =============================================================================
# INJECT JSON METHODS FROM REGISTRY
# =============================================================================


def _inject_json_accessor_methods():
    """Inject JSON methods from registry into JsonAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.JSON):
        # Skip if already exists
        if hasattr(JsonAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(JsonAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(JsonAccessor, alias_name):
                setattr(JsonAccessor, alias_name, getattr(JsonAccessor, spec.name))


# Perform injection when module is loaded
_inject_json_accessor_methods()
