"""
Unified Expression Evaluator for DataStore.

This module provides a centralized expression evaluation system that respects
the function_config settings. It is used by both ColumnExpr and LazyOp classes
to ensure consistent behavior across all execution paths.

Key Features:
- Unified evaluation logic for all expression types
- Respects function_config for Pandas vs chDB execution
- Dynamic Pandas method invocation fallback
- Centralized chDB execution via Executor
"""

from typing import Any, Union, TYPE_CHECKING
import pandas as pd

from .expressions import Expression, Field, Literal, ArithmeticExpression
from .config import get_logger

if TYPE_CHECKING:
    from .core import DataStore


class ExpressionEvaluator:
    """
    Unified expression evaluator that respects function_config settings.

    This class centralizes all expression evaluation logic, ensuring that:
    1. function_config is always checked for execution engine preference
    2. Pandas-only functions always use Pandas
    3. Overlapping functions respect user configuration
    4. Unknown functions try Pandas first, then fall back to chDB

    Usage:
        evaluator = ExpressionEvaluator(df, context)
        result = evaluator.evaluate(expr)

    Example:
        >>> df = pd.DataFrame({'value': [1, 2, 3]})
        >>> ds = DataStore.from_dataframe(df)
        >>> evaluator = ExpressionEvaluator(df, ds)
        >>>
        >>> # Evaluate a Field
        >>> result = evaluator.evaluate(Field('value'))
        >>> # Evaluate an arithmetic expression
        >>> result = evaluator.evaluate(ArithmeticExpression('+', Field('value'), Literal(1)))
        >>> # Evaluate a Function (respects function_config)
        >>> result = evaluator.evaluate(Function('upper', Field('name')))
    """

    def __init__(self, df: pd.DataFrame, context: 'DataStore'):
        """
        Initialize the evaluator.

        Args:
            df: The DataFrame to evaluate expressions against
            context: The DataStore context (for accessing configuration)
        """
        self.df = df
        self.context = context
        self._logger = get_logger()

    def evaluate(self, expr: Any) -> Union[pd.Series, Any]:
        """
        Recursively evaluate an expression.

        Supports:
        - ColumnExpr: unwrap and evaluate the underlying expression
        - Field references: df['col']
        - Literals: 1, "hello"
        - Arithmetic: df['a'] + 1
        - Functions: df['col'].str.upper(), respects function_config
        - Series: direct pandas Series (pass-through)
        - Scalars: direct values (pass-through)

        Args:
            expr: The expression to evaluate

        Returns:
            The evaluated result (typically a pd.Series or scalar)
        """
        from .functions import Function, CastFunction
        from .function_executor import function_config
        from .column_expr import ColumnExpr

        # Handle ColumnExpr - unwrap to get the underlying expression
        if isinstance(expr, ColumnExpr):
            return self.evaluate(expr._expr)

        if isinstance(expr, Field):
            # Column reference
            if expr.name in self.df.columns:
                return self.df[expr.name]
            else:
                raise KeyError(f"Column '{expr.name}' not found in DataFrame")

        elif isinstance(expr, Literal):
            # Literal value
            return expr.value

        elif isinstance(expr, ArithmeticExpression):
            # Arithmetic operation: recursively evaluate
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            return self._apply_operator(left, expr.operator, right)

        elif isinstance(expr, CastFunction):
            # Special handling for CAST - always use chDB
            return self._evaluate_via_chdb(expr)

        elif isinstance(expr, Function):
            # Function call - check execution config
            # Use pandas_name if available (for functions where SQL name differs from user-facing name)
            # e.g., contains uses SQL 'position' but should be checked as 'contains' for execution engine
            func_name = (getattr(expr, 'pandas_name', None) or expr.name).lower()

            # Priority order:
            # 1. Pandas-only functions or explicitly configured to use Pandas
            # 2. Has registered Pandas implementation but not configured -> use chDB (default)
            # 3. Try dynamic Pandas method (for unregistered functions)
            # 4. Fallback to chDB

            if function_config.should_use_pandas(func_name):
                # Pandas-only or explicitly configured for Pandas
                self._logger.debug("[ExprEval] Function '%s' -> Pandas (config)", func_name)
                return self._evaluate_function_via_pandas(expr)
            elif function_config.has_pandas_implementation(func_name):
                # Has registered implementation - use chDB by default unless configured
                self._logger.debug("[ExprEval] Function '%s' -> chDB (default)", func_name)
                return self._evaluate_via_chdb(expr)
            else:
                # Unknown function - try Pandas dynamic first, then chDB
                return self._evaluate_unknown_function(expr)

        elif isinstance(expr, pd.Series):
            # Direct pandas Series - pass through
            return expr

        elif isinstance(expr, Expression):
            # Other Expression types - try chDB
            return self._evaluate_via_chdb(expr)

        else:
            # Scalar value - pass through
            return expr

    def _apply_operator(self, left: Any, operator: str, right: Any) -> Any:
        """Apply an arithmetic operator to two operands."""
        if operator == '+':
            return left + right
        elif operator == '-':
            return left - right
        elif operator == '*':
            return left * right
        elif operator == '/':
            return left / right
        elif operator == '//':
            return left // right
        elif operator == '%':
            return left % right
        elif operator == '**':
            return left**right
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def _evaluate_function_via_pandas(self, expr) -> Any:
        """
        Evaluate a Function expression using Pandas implementation.

        Supports three modes:
        1. Registered implementation: use function_config._pandas_implementations
        2. Dynamic method: try to call method on Series (e.g., s.str.title(), s.abs())
        3. Fallback to chDB if nothing works

        Args:
            expr: Function expression

        Returns:
            Result of the Pandas operation (typically a Series)
        """
        from .function_executor import function_config

        # Use pandas_name if available (for functions where SQL name differs from user-facing name)
        func_name = (getattr(expr, 'pandas_name', None) or expr.name).lower()
        pandas_impl = function_config.get_pandas_implementation(func_name)

        # Evaluate first argument (the column/expression)
        if not expr.args:
            return self._evaluate_via_chdb(expr)

        first_arg = self.evaluate(expr.args[0])
        other_args = [self.evaluate(arg) for arg in expr.args[1:]]
        
        # Get pandas_kwargs if available (for functions with extra parameters like contains)
        pandas_kwargs = getattr(expr, 'pandas_kwargs', {}) or {}

        # Mode 1: Try registered implementation first
        if pandas_impl is not None:
            try:
                self._logger.debug("[ExprEval] Using registered Pandas impl for '%s'", func_name)
                # If we have pandas_kwargs, use them instead of positional args
                if pandas_kwargs:
                    return pandas_impl(first_arg, **pandas_kwargs)
                else:
                    return pandas_impl(first_arg, *other_args)
            except Exception as e:
                self._logger.debug("[ExprEval] Registered impl for '%s' failed: %s", func_name, e)

        # Mode 2: Try dynamic Pandas method invocation
        result = self._try_dynamic_pandas_method(func_name, first_arg, other_args)
        if result is not None:
            return result

        # Mode 3: Fallback to chDB
        self._logger.debug("[ExprEval] No Pandas method found for '%s', falling back to chDB", func_name)
        return self._evaluate_via_chdb(expr)

    def _try_dynamic_pandas_method(self, func_name: str, series: pd.Series, args: list) -> Any:
        """
        Try to dynamically call a Pandas method on a Series.

        Attempts to find and call the method in this order:
        1. Direct Series method: series.func_name(*args)
        2. Series.str accessor: series.str.func_name(*args)
        3. Series.dt accessor: series.dt.func_name(*args)

        Args:
            func_name: Function name to look for
            series: Pandas Series to operate on
            args: Additional arguments

        Returns:
            Result Series if successful, None if method not found
        """
        if not isinstance(series, pd.Series):
            return None

        # Try direct Series method
        if hasattr(series, func_name):
            method = getattr(series, func_name)
            if callable(method):
                try:
                    self._logger.debug("[ExprEval] Dynamic call: series.%s(*args)", func_name)
                    return method(*args) if args else method()
                except Exception as e:
                    self._logger.debug("[ExprEval] series.%s failed: %s", func_name, e)

        # Try Series.str accessor
        if hasattr(series, 'str') and hasattr(series.str, func_name):
            method = getattr(series.str, func_name)
            if callable(method):
                try:
                    self._logger.debug("[ExprEval] Dynamic call: series.str.%s(*args)", func_name)
                    return method(*args) if args else method()
                except Exception as e:
                    self._logger.debug("[ExprEval] series.str.%s failed: %s", func_name, e)

        # Try Series.dt accessor
        if hasattr(series, 'dt') and hasattr(series.dt, func_name):
            attr = getattr(series.dt, func_name)
            try:
                if callable(attr):
                    self._logger.debug("[ExprEval] Dynamic call: series.dt.%s(*args)", func_name)
                    return attr(*args) if args else attr()
                else:
                    # It's a property
                    self._logger.debug("[ExprEval] Dynamic access: series.dt.%s", func_name)
                    return attr
            except Exception as e:
                self._logger.debug("[ExprEval] series.dt.%s failed: %s", func_name, e)

        return None

    def _evaluate_unknown_function(self, expr) -> Any:
        """
        Evaluate an unknown function - try Pandas dynamic method first, then chDB.

        For functions not registered in function_config, we try:
        1. Dynamic Pandas method invocation
        2. Fallback to chDB (which may have the function)

        Args:
            expr: Function expression

        Returns:
            Result of the function evaluation
        """
        func_name = expr.name.lower()

        # Evaluate first argument
        if not expr.args:
            return self._evaluate_via_chdb(expr)

        first_arg = self.evaluate(expr.args[0])
        other_args = [self.evaluate(arg) for arg in expr.args[1:]]

        # Try dynamic Pandas method first
        result = self._try_dynamic_pandas_method(func_name, first_arg, other_args)
        if result is not None:
            self._logger.debug("[ExprEval] Dynamic method '%s' succeeded", func_name)
            return result

        # Fallback to chDB
        self._logger.debug("[ExprEval] No dynamic method for '%s', trying chDB", func_name)
        return self._evaluate_via_chdb(expr)

    def _evaluate_via_chdb(self, expr) -> pd.Series:
        """
        Evaluate the expression using chDB's Python() table function.

        This executes the expression as SQL on the DataFrame via the centralized Executor.

        Args:
            expr: Expression to evaluate

        Returns:
            Result Series from chDB execution
        """
        from .executor import get_executor

        # Build the SQL expression
        sql_expr = expr.to_sql(quote_char='"')
        self._logger.debug("[ExprEval] chDB executing: %s", sql_expr)

        # Use centralized executor
        executor = get_executor()
        return executor.execute_expression(sql_expr, self.df)


# Convenience function for quick evaluation
def evaluate_expression(expr: Any, df: pd.DataFrame, context: 'DataStore' = None) -> Any:
    """
    Convenience function to evaluate an expression.

    Args:
        expr: The expression to evaluate
        df: The DataFrame to evaluate against
        context: Optional DataStore context

    Returns:
        The evaluated result
    """
    evaluator = ExpressionEvaluator(df, context)
    return evaluator.evaluate(expr)
