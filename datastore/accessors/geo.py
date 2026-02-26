"""
GeoAccessor - Geo/distance functions via .geo accessor.

Provides ClickHouse geo and distance functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class GeoAccessor(BaseAccessor):
    """
    Accessor for geo and distance functions via .geo property.

    Maps to ClickHouse geo functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['point'].geo.latitude()              # tupleElement(point, 1)
        >>> ds['point'].geo.longitude()             # tupleElement(point, 2)
        >>> ds.apply(F.geo_distance(lat1, lon1, lat2, lon2))  # geoDistance(...)
        >>> ds.apply(F.great_circle_distance(...))   # greatCircleDistance(...)

    ClickHouse Geo Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/geo/
    """

    # Standard methods will be injected from registry below
    pass


# =============================================================================
# INJECT GEO METHODS FROM REGISTRY
# =============================================================================


def _inject_geo_accessor_methods():
    """Inject geo methods from registry into GeoAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.GEO):
        # Skip if already exists
        if hasattr(GeoAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(GeoAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(GeoAccessor, alias_name):
                setattr(GeoAccessor, alias_name, getattr(GeoAccessor, spec.name))


# Perform injection when module is loaded
_inject_geo_accessor_methods()
