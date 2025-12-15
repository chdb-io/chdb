"""
UrlAccessor - URL functions via .url accessor.

Provides ClickHouse URL functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class UrlAccessor(BaseAccessor):
    """
    Accessor for URL functions via .url property.

    Maps to ClickHouse URL functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['link'].url.domain()              # domain(link)
        >>> ds['link'].url.path()                # path(link)
        >>> ds['link'].url.protocol()            # protocol(link)
        >>> ds['link'].url.query_string()        # queryString(link)

    ClickHouse URL Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/url-functions
    """

    # Standard methods will be injected from registry below
    pass


# =============================================================================
# INJECT URL METHODS FROM REGISTRY
# =============================================================================


def _inject_url_accessor_methods():
    """Inject URL methods from registry into UrlAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.URL):
        # Skip if already exists
        if hasattr(UrlAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(UrlAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(UrlAccessor, alias_name):
                setattr(UrlAccessor, alias_name, getattr(UrlAccessor, spec.name))


# Perform injection when module is loaded
_inject_url_accessor_methods()
