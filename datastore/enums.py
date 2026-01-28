"""
Enumerations for DataStore
"""

from enum import Enum

__all__ = ['JoinType']


class JoinType(Enum):
    """JOIN types supported by DataStore"""

    inner = "INNER"
    left = "LEFT"
    right = "RIGHT"
    outer = "FULL OUTER"
    left_outer = "LEFT OUTER"
    right_outer = "RIGHT OUTER"
    full_outer = "FULL OUTER"
    cross = "CROSS"

    def __str__(self):
        return self.value
