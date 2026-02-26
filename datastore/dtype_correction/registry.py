"""
Dtype correction registry - central management of all dtype correction rules.
"""

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import pandas as pd
import logging

from .rules import (
    DtypeCorrectionRule,
    CorrectionPriority,
    ALL_RULES,
)
from .config import dtype_correction_config

if TYPE_CHECKING:
    from ..lazy_ops import LazyColumnAssignment

logger = logging.getLogger('datastore')


class DtypeCorrectionRegistry:
    """
    Registry for dtype correction rules.

    Provides centralized management of dtype corrections for chDB-pandas compatibility.

    Example:
        >>> from datastore.dtype_correction import dtype_registry
        >>>
        >>> # Check if correction needed
        >>> if dtype_registry.should_correct('abs', 'int64', 'uint64'):
        ...     target = dtype_registry.get_target_dtype('abs', 'int64')
        ...     print(f"Should convert to {target}")
        >>>
        >>> # Apply corrections to a column
        >>> corrected = dtype_registry.apply_correction('abs', series, 'int64')
    """

    def __init__(self):
        # Map from function name to list of applicable rules (sorted by priority)
        self._rules: Dict[str, List[DtypeCorrectionRule]] = {}
        # Register all built-in rules
        for rule in ALL_RULES:
            self.register(rule)

    def register(self, rule: DtypeCorrectionRule) -> None:
        """
        Register a correction rule.

        Args:
            rule: The rule to register
        """
        for func_name in rule.func_names:
            if func_name not in self._rules:
                self._rules[func_name] = []
            self._rules[func_name].append(rule)
            # Sort by priority (lower priority value = higher precedence)
            self._rules[func_name].sort(key=lambda r: r.priority)

    def get_rules(self, func_name: str) -> List[DtypeCorrectionRule]:
        """
        Get all rules applicable to a function.

        Args:
            func_name: Function name (lowercase)

        Returns:
            List of applicable rules sorted by priority
        """
        return self._rules.get(func_name.lower(), [])

    def should_correct(self, func_name: str, input_dtype: str, output_dtype: str) -> bool:
        """
        Check if dtype correction should be applied.

        Args:
            func_name: The function name (e.g., 'abs', 'sign')
            input_dtype: The dtype of the input column
            output_dtype: The dtype of the output after chDB execution

        Returns:
            True if correction should be applied
        """
        rules = self.get_rules(func_name)
        for rule in rules:
            if dtype_correction_config.should_apply(rule.priority):
                if rule.should_correct(input_dtype, output_dtype):
                    return True
        return False

    def get_target_dtype(self, func_name: str, input_dtype: str) -> Optional[str]:
        """
        Get the target dtype for correction.

        Args:
            func_name: The function name
            input_dtype: The dtype of the input column

        Returns:
            The target dtype, or None if no correction needed
        """
        rules = self.get_rules(func_name)
        for rule in rules:
            if dtype_correction_config.should_apply(rule.priority):
                target = rule.get_target_dtype(input_dtype)
                if target is not None:
                    return target
        return None

    def apply_correction(
        self,
        func_name: str,
        series: pd.Series,
        input_dtype: str,
    ) -> pd.Series:
        """
        Apply dtype correction to a Series.

        Args:
            func_name: The function name
            series: The result Series from chDB
            input_dtype: The dtype of the input column

        Returns:
            Series with corrected dtype
        """
        output_dtype = str(series.dtype)
        rules = self.get_rules(func_name)

        for rule in rules:
            if not dtype_correction_config.should_apply(rule.priority):
                continue

            if rule.should_correct(input_dtype, output_dtype):
                logger.debug(
                    "[DtypeCorrection] Applying %s for %s: %s -> %s",
                    rule.__class__.__name__,
                    func_name,
                    output_dtype,
                    rule.get_target_dtype(input_dtype),
                )
                return rule.apply(series, input_dtype)

        return series

    def apply_corrections_to_dataframe(
        self,
        result_df: pd.DataFrame,
        input_df: pd.DataFrame,
        column_operations: Dict[str, Tuple[str, str]],
    ) -> pd.DataFrame:
        """
        Apply dtype corrections to a DataFrame based on column operations.

        Args:
            result_df: Result DataFrame from SQL execution
            input_df: Input DataFrame (for dtype lookup)
            column_operations: Dict mapping column names to (func_name, source_column)

        Returns:
            DataFrame with corrected dtypes
        """
        for col_name, (func_name, source_col) in column_operations.items():
            if col_name not in result_df.columns:
                continue
            if source_col not in input_df.columns:
                continue

            input_dtype = str(input_df[source_col].dtype)
            result_df[col_name] = self.apply_correction(func_name, result_df[col_name], input_dtype)

        return result_df

    def extract_operations_from_plan(
        self,
        plan,
        input_df: pd.DataFrame,
    ) -> Dict[str, Tuple[str, str]]:
        """
        Extract column operations from a QueryPlan.

        Scans LazyColumnAssignment operations to identify functions
        that may need dtype correction.

        Args:
            plan: QueryPlan object
            input_df: Input DataFrame

        Returns:
            Dict mapping target_column to (func_name, source_column)
        """
        from ..lazy_ops import LazyColumnAssignment
        from ..functions import Function
        from ..column_expr import ColumnExpr
        from ..expressions import Field

        operations = {}

        # Get column assignments from plan
        column_assignments = [
            op for op in plan.sql_ops if isinstance(op, LazyColumnAssignment) and op.can_push_to_sql()
        ]

        for op in column_assignments:
            target_col = op.column
            expr = op.expr

            # Extract function name and source column
            func_name = None
            source_col = None

            if isinstance(expr, ColumnExpr) and expr._expr is not None:
                inner_expr = expr._expr
                if isinstance(inner_expr, Function):
                    func_name = inner_expr.name.lower()
                    if inner_expr.args and isinstance(inner_expr.args[0], Field):
                        source_col = inner_expr.args[0].name
            elif isinstance(expr, Function):
                func_name = expr.name.lower()
                if expr.args and isinstance(expr.args[0], Field):
                    source_col = expr.args[0].name

            if func_name and source_col:
                # Check if this function has any correction rules
                if func_name in self._rules:
                    operations[target_col] = (func_name, source_col)

        return operations


# Global registry instance
dtype_registry = DtypeCorrectionRegistry()
