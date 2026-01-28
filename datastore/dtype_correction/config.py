"""
Configuration for dtype correction system.

Controls which levels of dtype correction are applied.
"""

from enum import IntEnum
from typing import Optional


class CorrectionLevel(IntEnum):
    """Dtype correction levels - higher value means more corrections."""

    NONE = 0  # No corrections
    CRITICAL = 1  # Only critical (sign changes like abs)
    HIGH = 2  # Critical + high priority (sign, pow, floordiv)
    MEDIUM = 3  # All above + arithmetic type preservation
    ALL = 4  # All corrections including precision


class DtypeCorrectionConfig:
    """
    Configuration for dtype correction behavior.

    Example:
        >>> from datastore.dtype_correction import dtype_correction_config
        >>>
        >>> # Set correction level
        >>> dtype_correction_config.set_level(CorrectionLevel.HIGH)
        >>>
        >>> # Check current level
        >>> dtype_correction_config.level  # CorrectionLevel.HIGH
        >>>
        >>> # Disable all corrections
        >>> dtype_correction_config.set_level(CorrectionLevel.NONE)
    """

    def __init__(self):
        self._level: CorrectionLevel = CorrectionLevel.HIGH  # Default: critical + high

    @property
    def level(self) -> CorrectionLevel:
        """Get current correction level."""
        return self._level

    def set_level(self, level: CorrectionLevel) -> None:
        """
        Set the dtype correction level.

        Args:
            level: The correction level to apply
        """
        self._level = level

    def should_apply(self, priority) -> bool:
        """
        Check if a correction with given priority should be applied.

        Args:
            priority: The priority of the correction rule (CorrectionPriority enum)

        Returns:
            True if the correction should be applied
        """
        from .rules import CorrectionPriority

        priority_to_level = {
            CorrectionPriority.CRITICAL: CorrectionLevel.CRITICAL,
            CorrectionPriority.HIGH: CorrectionLevel.HIGH,
            CorrectionPriority.MEDIUM: CorrectionLevel.MEDIUM,
            CorrectionPriority.LOW: CorrectionLevel.ALL,
        }
        required_level = priority_to_level.get(priority, CorrectionLevel.ALL)
        return self._level >= required_level

    def reset(self) -> None:
        """Reset to default configuration."""
        self._level = CorrectionLevel.HIGH


# Global configuration instance
dtype_correction_config = DtypeCorrectionConfig()
