"""
IpAccessor - IP address functions via .ip accessor.

Provides ClickHouse IP functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class IpAccessor(BaseAccessor):
    """
    Accessor for IP address functions via .ip property.

    Maps to ClickHouse IP functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['ip'].ip.to_ipv4()                # toIPv4(ip)
        >>> ds['ip'].ip.to_ipv6()                # toIPv6(ip)
        >>> ds['ip'].ip.ipv4_to_num()            # IPv4NumToString(ip)
        >>> ds['ip'].ip.is_ipv4()                # isIPv4String(ip)

    ClickHouse IP Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/ip-address-functions
    """

    # Standard methods will be injected from registry below
    pass


# =============================================================================
# INJECT IP METHODS FROM REGISTRY
# =============================================================================


def _inject_ip_accessor_methods():
    """Inject IP methods from registry into IpAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.IP):
        # Skip if already exists
        if hasattr(IpAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(IpAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(IpAccessor, alias_name):
                setattr(IpAccessor, alias_name, getattr(IpAccessor, spec.name))


# Perform injection when module is loaded
_inject_ip_accessor_methods()
