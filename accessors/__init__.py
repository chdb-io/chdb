"""
Accessor classes for Expression - providing Pandas-like .str, .dt, .arr, .json accessors.

These accessors wrap ClickHouse SQL functions in a Pandas-compatible API,
allowing users to write code like:

    ds['name'].str.upper()
    ds['date'].dt.year
    ds['tags'].arr.length()
    ds['data'].json.get_string('$.name')

Each accessor method returns a Function expression that generates the
appropriate ClickHouse SQL when executed.
"""

from .base import BaseAccessor
from .string import StringAccessor
from .datetime import DateTimeAccessor

# Optional accessors (can be added later)
# from .array import ArrayAccessor
# from .json import JsonAccessor

__all__ = [
    'BaseAccessor',
    'StringAccessor',
    'DateTimeAccessor',
    # 'ArrayAccessor',
    # 'JsonAccessor',
]
