"""
Dtype correction rules for chDB-DataStore.

Each rule defines how to correct dtype mismatches for specific functions/operations.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional, Set, Dict, Tuple
import pandas as pd
import numpy as np


class CorrectionPriority(IntEnum):
    """Priority levels for dtype corrections."""

    CRITICAL = 1  # Sign changes - must fix (e.g., abs)
    HIGH = 2  # Family changes - should fix (e.g., sign, pow)
    MEDIUM = 3  # Type width changes - nice to fix (e.g., add, sub, mul)
    LOW = 4  # Precision changes - optional (e.g., float32 → float64)


class DtypeCorrectionRule(ABC):
    """
    Abstract base class for dtype correction rules.

    Each rule defines how to correct dtype mismatches for specific operations.
    """

    # Function names this rule applies to (lowercase)
    func_names: Set[str] = set()

    # Priority level of this rule
    priority: CorrectionPriority = CorrectionPriority.MEDIUM

    @abstractmethod
    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """
        Check if correction should be applied.

        Args:
            input_dtype: The dtype of the input column (e.g., 'int64')
            output_dtype: The dtype of the output after chDB execution (e.g., 'uint64')

        Returns:
            True if correction should be applied
        """
        pass

    @abstractmethod
    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """
        Get the target dtype that the output should be corrected to.

        Args:
            input_dtype: The dtype of the input column

        Returns:
            The target dtype string, or None if no correction needed
        """
        pass

    def apply(self, series: pd.Series, input_dtype: str) -> pd.Series:
        """
        Apply the dtype correction to a Series.

        Args:
            series: The result Series from chDB
            input_dtype: The dtype of the input column

        Returns:
            Series with corrected dtype
        """
        target_dtype = self.get_target_dtype(input_dtype)
        if target_dtype is None:
            return series

        try:
            return series.astype(target_dtype)
        except (ValueError, TypeError, OverflowError):
            # If conversion fails, return original
            return series


class SignedAbsRule(DtypeCorrectionRule):
    """
    Correction rule for abs() on signed integers.

    chDB returns unsigned types for abs() on signed integers,
    but pandas returns the same signed type.

    Example:
        abs(int64) → chDB returns uint64, pandas returns int64
    """

    func_names = {"abs"}
    priority = CorrectionPriority.CRITICAL

    # Mapping: (input_dtype, output_dtype) → target_dtype
    _UNSIGNED_TO_SIGNED: Dict[Tuple[str, str], str] = {
        ("int8", "uint8"): "int8",
        ("int16", "uint16"): "int16",
        ("int32", "uint32"): "int32",
        ("int64", "uint64"): "int64",
    }

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if input was signed int and output is unsigned."""
        return (input_dtype, output_dtype) in self._UNSIGNED_TO_SIGNED

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """Return the signed equivalent of the input dtype."""
        if input_dtype.startswith("int") and not input_dtype.startswith("uint"):
            return input_dtype
        return None


class SignPreserveRule(DtypeCorrectionRule):
    """
    Correction rule for sign() function.

    chDB always returns Int8 for sign(), but pandas/numpy preserves
    the input type (or returns float for float input).

    Example:
        sign(int64) → chDB returns int8, numpy returns int64
        sign(float64) → chDB returns int8, numpy returns float64
    """

    func_names = {"sign"}
    priority = CorrectionPriority.HIGH

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if output is int8 but input was different."""
        return output_dtype == "int8" and input_dtype != "int8"

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """Preserve the input dtype for sign()."""
        # For integer input, preserve the integer type
        if "int" in input_dtype:
            return input_dtype
        # For float input, preserve float type
        if "float" in input_dtype:
            return input_dtype
        return None


class ArithmeticPreserveRule(DtypeCorrectionRule):
    """
    Correction rule for arithmetic operations (add, sub, mul, mod).

    chDB may widen or narrow types for overflow protection,
    but pandas preserves the input type.

    Example:
        int8 + 2 → chDB returns int16, pandas returns int8
        mod(int64, 2) → chDB returns int16, pandas returns int64
    """

    func_names = {"add", "sub", "mul", "mod", "modulo"}
    priority = CorrectionPriority.MEDIUM

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if dtype width changed but family stayed the same."""
        # Only correct within same family (int/uint/float)
        input_family = self._get_dtype_family(input_dtype)
        output_family = self._get_dtype_family(output_dtype)

        if input_family != output_family:
            return False

        return input_dtype != output_dtype

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """Preserve the input dtype."""
        return input_dtype

    @staticmethod
    def _get_dtype_family(dtype: str) -> str:
        """Get the family of a dtype (int, uint, or float)."""
        dtype = dtype.lower()
        if dtype.startswith("uint"):
            return "uint"
        elif dtype.startswith("int"):
            return "int"
        elif dtype.startswith("float"):
            return "float"
        return "other"


class PowPreserveRule(DtypeCorrectionRule):
    """
    Correction rule for pow() function.

    chDB always returns Float64 for pow(), but pandas preserves
    integer type for integer ** integer operations.

    Note: This is a complex case because:
    - int ** int with small result → should be int
    - int ** float → should be float
    - Any ** negative → should be float

    For simplicity, we only correct int ** positive_int_literal cases.
    """

    func_names = {"pow", "power"}
    priority = CorrectionPriority.HIGH

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if input was int and output is float64."""
        return "int" in input_dtype and output_dtype == "float64"

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """
        Return int64 for integer pow results.

        Note: We use int64 as the safest integer type to avoid overflow.
        """
        if "int" in input_dtype:
            return "int64"
        return None

    def apply(self, series: pd.Series, input_dtype: str) -> pd.Series:
        """
        Apply correction only if values can be safely represented as int.
        """
        if "int" not in input_dtype:
            return series

        # Check if all values are actually integers (no decimal part)
        if series.isna().any():
            return series

        try:
            # Check if values are whole numbers
            if not np.allclose(series, series.astype("int64")):
                return series  # Keep as float if fractional
            return series.astype("int64")
        except (ValueError, TypeError, OverflowError):
            return series


class FloordivPreserveRule(DtypeCorrectionRule):
    """
    Correction rule for floor division (//).

    chDB's intDiv returns int, but pandas // preserves float for float input.

    Example:
        float64 // 2 → chDB returns int64, pandas returns float64
    """

    func_names = {"floordiv", "intdiv"}
    priority = CorrectionPriority.HIGH

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if input was float but output is int."""
        return "float" in input_dtype and "int" in output_dtype

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """Preserve float type for floordiv."""
        if "float" in input_dtype:
            return input_dtype
        return None


class Float32PreserveRule(DtypeCorrectionRule):
    """
    Correction rule for float32 precision preservation.

    chDB often promotes float32 to float64, but pandas preserves float32.
    This is a LOW priority rule as float64 is generally more precise.
    """

    func_names = {"add", "sub", "mul", "div", "mod", "pow", "sqrt", "exp", "log"}
    priority = CorrectionPriority.LOW

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if input was float32 but output is float64."""
        return input_dtype == "float32" and output_dtype == "float64"

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """Return float32 if input was float32."""
        if input_dtype == "float32":
            return "float32"
        return None


class ClipNullablePreserveRule(DtypeCorrectionRule):
    """
    Correction rule for clip() on nullable integer columns.

    chDB's greatest/least functions return float64 for nullable integer columns,
    but pandas preserves the original nullable dtype (Int64, Int32, etc.).

    Example:
        clip(Int64) → chDB returns float64, pandas returns Int64
    """

    func_names = {"greatest", "least", "clip"}  # clip uses greatest/least internally
    priority = CorrectionPriority.HIGH

    def should_correct(self, input_dtype: str, output_dtype: str) -> bool:
        """Check if input was nullable int and output is float64."""
        # Nullable int dtypes start with capital letter: Int64, Int32, Int16, Int8
        is_nullable_int = input_dtype.startswith("Int") and input_dtype[3:].isdigit()
        return is_nullable_int and output_dtype == "float64"

    def get_target_dtype(self, input_dtype: str) -> Optional[str]:
        """Return the original nullable dtype."""
        if input_dtype.startswith("Int") and input_dtype[3:].isdigit():
            return input_dtype
        return None

    def apply(self, series: pd.Series, input_dtype: str) -> pd.Series:
        """
        Apply correction: convert float64 back to nullable Int.

        Need special handling because we need to use pd.array with nullable dtype.
        """
        target_dtype = self.get_target_dtype(input_dtype)
        if target_dtype is None:
            return series

        try:
            # Use convert_dtypes() or direct astype to nullable int
            return series.astype(target_dtype)
        except (ValueError, TypeError, OverflowError):
            # If conversion fails, return original
            return series


# All available rules
ALL_RULES = [
    SignedAbsRule(),
    SignPreserveRule(),
    ArithmeticPreserveRule(),
    PowPreserveRule(),
    FloordivPreserveRule(),
    Float32PreserveRule(),
    ClipNullablePreserveRule(),
]
