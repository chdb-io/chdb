"""
Dtype Correction System for chDB-DataStore.

This module provides a unified system for correcting dtype mismatches
between chDB SQL execution results and pandas expected behavior.

Usage:
    from datastore.dtype_correction import dtype_registry, CorrectionPriority

    # Apply corrections to a result DataFrame
    corrected_df = dtype_registry.apply_corrections(
        result_df, input_df, column_operations
    )

    # Check if correction is needed
    if dtype_registry.should_correct('abs', 'int64', 'uint64'):
        target = dtype_registry.get_target_dtype('abs', 'int64')
"""

from .registry import DtypeCorrectionRegistry, dtype_registry
from .rules import (
    CorrectionPriority,
    DtypeCorrectionRule,
    SignedAbsRule,
    SignPreserveRule,
    ArithmeticPreserveRule,
)
from .config import dtype_correction_config

__all__ = [
    'DtypeCorrectionRegistry',
    'dtype_registry',
    'CorrectionPriority',
    'DtypeCorrectionRule',
    'SignedAbsRule',
    'SignPreserveRule',
    'ArithmeticPreserveRule',
    'dtype_correction_config',
]
