"""
Accessor classes for Expression - providing Pandas-like .str, .dt, .arr, .json accessors.

These accessors wrap ClickHouse SQL functions in a Pandas-compatible API,
allowing users to write code like:

    ds['name'].str.upper()
    ds['date'].dt.year
    ds['tags'].arr.length()
    ds['data'].json.get_string('$.name')
    ds['link'].url.domain()
    ds['ip_addr'].ip.to_ipv4()
    ds['point'].geo.distance(other_point)

Each accessor method returns a Function expression that generates the
appropriate ClickHouse SQL when executed.
"""

from .base import BaseAccessor
from .string import StringAccessor
from .datetime import DateTimeAccessor
from .array import ArrayAccessor
from .json import JsonAccessor
from .url import UrlAccessor
from .ip import IpAccessor
from .geo import GeoAccessor

__all__ = [
    'BaseAccessor',
    'StringAccessor',
    'DateTimeAccessor',
    'ArrayAccessor',
    'JsonAccessor',
    'UrlAccessor',
    'IpAccessor',
    'GeoAccessor',
]
