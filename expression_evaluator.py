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

from typing import Any, Optional, Union, TYPE_CHECKING
import pandas as pd
import numpy as np

from .expressions import Expression, Field, Literal, ArithmeticExpression
from .conditions import (
    Condition,
    BinaryCondition,
    CompoundCondition,
    NotCondition,
    UnaryCondition,
    InCondition,
    BetweenCondition,
    LikeCondition,
)
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
        from .lazy_result import LazySeries

        # Helper to check if something is a lazy object
        def _is_lazy(obj):
            return isinstance(obj, ColumnExpr) or isinstance(obj, LazySeries) or hasattr(obj, '_execute')

        # Handle ColumnExpr - check execution mode
        if isinstance(expr, ColumnExpr):
            if expr._exec_mode == 'expr' and expr._expr is not None:
                # Expression mode - unwrap and evaluate
                return self.evaluate(expr._expr)
            elif expr._exec_mode == 'method' and expr._source is not None:
                # Method mode - evaluate source and call method
                source_series = self.evaluate(expr._source)
                if source_series is None:
                    return None
                # Execute arguments
                args = tuple(self.evaluate(arg) if _is_lazy(arg) else arg for arg in expr._method_args)
                kwargs = {k: self.evaluate(v) if _is_lazy(v) else v for k, v in expr._method_kwargs.items()}
                # Handle chained function calls (e.g., diff().abs() creates _chain_abs)
                method_name = expr._method_name
                if method_name.startswith('_chain_'):
                    method_name = method_name[7:]  # Remove '_chain_' prefix
                    # Remove 'alias' from kwargs if present (SQL-specific parameter)
                    kwargs = {k: v for k, v in kwargs.items() if k != 'alias'}
                # Call the method
                if hasattr(source_series, method_name):
                    method = getattr(source_series, method_name)
                    return method(*args, **kwargs)
                # Try numpy function as fallback for _chain_ methods
                if expr._method_name.startswith('_chain_'):
                    import numpy as np

                    if hasattr(np, method_name):
                        np_func = getattr(np, method_name)
                        return np_func(source_series, *args)
                return source_series
            elif expr._exec_mode == 'agg' and expr._source is not None:
                # Aggregation mode - evaluate source and apply aggregation
                source_series = self.evaluate(expr._source)
                if source_series is None:
                    return None
                agg_method = getattr(source_series, expr._pandas_agg_func)
                return agg_method()
            elif expr._exec_mode == 'executor' and expr._executor is not None:
                # Executor mode - check for circular reference to avoid recursion
                # If the ColumnExpr's datastore is the same as our context,
                # we're in a circular execution scenario (e.g., transform inside assign)
                # In this case, we need to use the current df instead of calling ds._execute()
                if expr._datastore is self.context and expr._groupby_fields:
                    # This is a groupby transform/agg being evaluated during the same
                    # DataStore's execution - use the current df
                    from .expressions import Field as ExprField

                    # Get groupby columns
                    groupby_cols = []
                    for gf in expr._groupby_fields:
                        if isinstance(gf, ExprField):
                            groupby_cols.append(gf.name)
                        else:
                            groupby_cols.append(str(gf))

                    # Get the column name
                    col_name = None
                    if isinstance(expr._expr, ExprField):
                        col_name = expr._expr.name
                    elif expr._expr is not None:
                        col_name = str(expr._expr)

                    # Use the current df from the evaluator
                    if col_name and col_name in self.df.columns:
                        # Re-execute the transform using current df
                        # The original executor captures the transform function
                        # but we can't access it directly, so we need to call the executor
                        # with a workaround - set up a temporary source df
                        try:
                            return self.df.groupby(groupby_cols)[col_name].transform(
                                expr._transform_func, *expr._transform_args, **expr._transform_kwargs
                            )
                        except AttributeError:
                            # If transform params not available, fall back to executor
                            return expr._executor()
                else:
                    # Normal executor call
                    return expr._executor()
            else:
                # Fallback - try to execute directly
                return expr._execute()

        # Handle LazySeries - evaluate the underlying column expr first
        # to avoid circular execution, then apply the method
        if isinstance(expr, LazySeries):
            # Evaluate the source ColumnExpr using this evaluator's df
            # (avoids circular execution when inside DataStore._execute)
            source_series = self.evaluate(expr._column_expr)

            # Helper to execute lazy arguments
            def _execute_arg(arg):
                if _is_lazy(arg):
                    result = self.evaluate(arg)
                    # If it's a single-value Series, extract scalar for fillna-like operations
                    if isinstance(result, pd.Series) and len(result) == 1:
                        return result.iloc[0]
                    return result
                return arg

            # Execute any lazy objects in args or kwargs
            args = tuple(_execute_arg(arg) for arg in expr._args)
            kwargs = {k: _execute_arg(v) for k, v in expr._kwargs.items()}

            # Handle special _dt_* methods for datetime operations (pandas fallback)
            if expr._method_name.startswith('_dt_'):
                dt_attr = expr._method_name[4:]  # Remove '_dt_' prefix
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(source_series):
                    if source_series.dtype == 'object' or pd.api.types.is_string_dtype(source_series):
                        try:
                            source_series = pd.to_datetime(source_series, errors='coerce')
                        except Exception:
                            pass
                # Access .dt accessor
                dt_accessor = source_series.dt
                attr = getattr(dt_accessor, dt_attr)
                if callable(attr):
                    return attr(*args, **kwargs)
                else:
                    return attr

            # Apply the method on the series
            method = getattr(source_series, expr._method_name)
            return method(*args, **kwargs)

        if isinstance(expr, Field):
            # Column reference - handle both string and integer column names
            col_name = expr.name
            if col_name in self.df.columns:
                return self.df[col_name]
            # Try converting to int if the column name looks like a number
            # pandas allows integer column names: df = pd.DataFrame({0: [1, 2, 3]})
            if isinstance(col_name, str) and col_name.isdigit():
                int_name = int(col_name)
                if int_name in self.df.columns:
                    return self.df[int_name]
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

        # Handle DateTimePropertyExpr and DateTimeMethodExpr
        # Engine selection happens here at execution time based on function_config
        elif hasattr(expr, 'property_name') and hasattr(expr, 'source_expr'):
            # DateTimePropertyExpr
            return self._evaluate_datetime_property(expr)

        elif hasattr(expr, 'method_name') and hasattr(expr, 'source_expr') and hasattr(expr, 'args'):
            # DateTimeMethodExpr
            return self._evaluate_datetime_method(expr)

        elif hasattr(expr, 'component') and hasattr(expr, 'source_expr'):
            # IsoCalendarComponentExpr
            return self._evaluate_isocalendar_component(expr)

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

        elif isinstance(expr, BinaryCondition):
            # Binary comparison condition -> boolean Series
            return self._evaluate_binary_condition(expr)

        elif isinstance(expr, CompoundCondition):
            # Compound condition (AND, OR, XOR) -> boolean Series
            return self._evaluate_compound_condition(expr)

        elif isinstance(expr, NotCondition):
            # NOT condition -> boolean Series
            return self._evaluate_not_condition(expr)

        elif isinstance(expr, UnaryCondition):
            # Unary condition (IS NULL, IS NOT NULL) -> boolean Series
            return self._evaluate_unary_condition(expr)

        elif isinstance(expr, InCondition):
            # IN condition -> boolean Series
            return self._evaluate_in_condition(expr)

        elif isinstance(expr, BetweenCondition):
            # BETWEEN condition -> boolean Series
            return self._evaluate_between_condition(expr)

        elif isinstance(expr, LikeCondition):
            # LIKE condition -> boolean Series
            return self._evaluate_like_condition(expr)

        # Handle LazyCondition (from isin, between, etc.)
        from .lazy_result import LazyCondition

        if isinstance(expr, LazyCondition):
            # Evaluate the underlying condition using current df
            # Don't call _execute() as it would trigger DataStore._execute() recursively
            return self.evaluate(expr._condition)

        # Handle CaseWhenExpr - CASE WHEN expression
        from .case_when import CaseWhenExpr

        if isinstance(expr, CaseWhenExpr):
            # Evaluate via np.select
            return expr.evaluate(self.df)

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

    # ========== Condition Evaluation Methods ==========

    def _evaluate_binary_condition(self, cond: BinaryCondition) -> pd.Series:
        """Evaluate a binary comparison condition to a boolean Series."""
        left = self.evaluate(cond.left)
        right = self.evaluate(cond.right)

        op = cond.operator
        if op == '=':
            return left == right
        elif op in ('!=', '<>'):
            return left != right
        elif op == '>':
            return left > right
        elif op == '>=':
            return left >= right
        elif op == '<':
            return left < right
        elif op == '<=':
            return left <= right
        elif op == 'IS':
            # IS NULL handling
            if right is None:
                return left.isna() if isinstance(left, pd.Series) else pd.Series([left is None])
            return left == right
        else:
            raise ValueError(f"Unknown comparison operator: {op}")

    def _evaluate_compound_condition(self, cond: CompoundCondition) -> pd.Series:
        """Evaluate a compound condition (AND, OR, XOR) to a boolean Series."""
        left = self.evaluate(cond.left)
        right = self.evaluate(cond.right)

        op = cond.operator
        if op == 'AND':
            return left & right
        elif op == 'OR':
            return left | right
        elif op == 'XOR':
            return left ^ right
        else:
            raise ValueError(f"Unknown logical operator: {op}")

    def _evaluate_not_condition(self, cond: NotCondition) -> pd.Series:
        """Evaluate a NOT condition to a boolean Series."""
        inner = self.evaluate(cond.condition)
        return ~inner

    def _evaluate_unary_condition(self, cond: UnaryCondition) -> pd.Series:
        """Evaluate a unary condition (IS NULL, IS NOT NULL) to a boolean Series."""
        expr_val = self.evaluate(cond.expression)

        op = cond.operator
        if op == 'IS NULL':
            return expr_val.isna() if isinstance(expr_val, pd.Series) else pd.Series([expr_val is None])
        elif op == 'IS NOT NULL':
            return expr_val.notna() if isinstance(expr_val, pd.Series) else pd.Series([expr_val is not None])
        else:
            raise ValueError(f"Unknown unary operator: {op}")

    def _evaluate_in_condition(self, cond: InCondition) -> pd.Series:
        """Evaluate an IN condition to a boolean Series."""
        expr_val = self.evaluate(cond.expression)

        # Handle subquery values
        if hasattr(cond.values, 'to_sql') and hasattr(cond.values, 'table_name'):
            # Subquery - execute and get values
            values = cond.values._execute().iloc[:, 0].tolist()
        else:
            values = cond.values

        result = expr_val.isin(values)
        return ~result if cond.negate else result

    def _evaluate_between_condition(self, cond: BetweenCondition) -> pd.Series:
        """Evaluate a BETWEEN condition to a boolean Series."""
        expr_val = self.evaluate(cond.expression)
        lower = self.evaluate(cond.lower)
        upper = self.evaluate(cond.upper)

        # Handle different inclusive modes
        inclusive = getattr(cond, 'inclusive', 'both')
        if inclusive == 'both':
            return (expr_val >= lower) & (expr_val <= upper)
        elif inclusive == 'neither':
            return (expr_val > lower) & (expr_val < upper)
        elif inclusive == 'left':
            return (expr_val >= lower) & (expr_val < upper)
        elif inclusive == 'right':
            return (expr_val > lower) & (expr_val <= upper)
        else:
            raise ValueError(f"Invalid inclusive value: {inclusive}")

    def _evaluate_like_condition(self, cond: LikeCondition) -> pd.Series:
        """Evaluate a LIKE condition to a boolean Series."""
        import re

        expr_val = self.evaluate(cond.expression)

        # Convert SQL LIKE pattern to regex
        pattern = cond.pattern
        # Escape regex special chars except % and _
        pattern = re.escape(pattern)
        # Convert SQL wildcards to regex
        pattern = pattern.replace(r'\%', '.*').replace(r'\_', '.')
        # Anchor the pattern
        pattern = f'^{pattern}$'

        flags = 0 if cond.case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        result = expr_val.astype(str).str.match(pattern, case=cond.case_sensitive)
        return ~result if cond.negate else result

    def _get_source_column_name(self, source_expr, source_series: pd.Series) -> Optional[str]:
        """Get column name from source expression or series."""
        from .expressions import Field

        # Try to get name from Field expression
        if isinstance(source_expr, Field):
            return source_expr.name

        # Fall back to series name
        if hasattr(source_series, 'name') and source_series.name:
            return source_series.name

        return None

    def _evaluate_datetime_property(self, expr) -> pd.Series:
        """
        Evaluate a DateTimePropertyExpr using function_config to select engine.

        At execution time, checks function_config.should_use_pandas(property_name)
        to determine whether to use pandas .dt accessor or chDB SQL function.
        """
        from .function_executor import function_config

        property_name = expr.property_name

        # Check if this property should use pandas
        if function_config.should_use_pandas(property_name):
            self._logger.debug("[ExprEval] DateTime property '%s' -> Pandas", property_name)
            return self._evaluate_datetime_property_pandas(expr)
        else:
            self._logger.debug("[ExprEval] DateTime property '%s' -> chDB", property_name)
            return self._evaluate_datetime_property_chdb(expr)

    def _evaluate_datetime_property_pandas(self, expr) -> pd.Series:
        """Evaluate datetime property using pandas .dt accessor."""
        source_series = self.evaluate(expr.source_expr)

        # Get original column name for result Series
        original_name = self._get_source_column_name(expr.source_expr, source_series)

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(source_series):
            if source_series.dtype == 'object' or pd.api.types.is_string_dtype(source_series):
                try:
                    source_series = pd.to_datetime(source_series, errors='coerce')
                except Exception:
                    pass

        # Access .dt accessor
        dt_accessor = source_series.dt
        result = getattr(dt_accessor, expr.property_name)

        # Preserve original column name (pandas behavior)
        if original_name and result.name != original_name:
            result = result.rename(original_name)

        return result

    def _evaluate_datetime_property_chdb(self, expr) -> pd.Series:
        """Evaluate datetime property using chDB SQL function."""
        from .expressions import DateTimePropertyExpr

        # Get original column name for result Series
        source_series = self.evaluate(expr.source_expr)
        original_name = self._get_source_column_name(expr.source_expr, source_series)

        # Check if the source series is timezone-aware
        # If so, use pandas to avoid complex timezone conversion issues in chDB
        if hasattr(source_series, 'dt') and source_series.dt.tz is not None:
            self._logger.debug("[ExprEval] Source has timezone info, using pandas for '%s'", expr.property_name)
            return self._evaluate_datetime_property_pandas(expr)

        source_sql = expr.source_expr.to_sql(quote_char='"')

        ch_func = DateTimePropertyExpr.CHDB_FUNCTION_MAP.get(expr.property_name)
        if not ch_func:
            # No chDB mapping, fall back to pandas
            self._logger.debug("[ExprEval] No chDB mapping for '%s', using pandas", expr.property_name)
            return self._evaluate_datetime_property_pandas(expr)

        # Build the SQL expression with proper datetime handling
        # Use toTimezone to convert to UTC to avoid timezone issues when chDB
        # interprets the datetime in local timezone (e.g., Asia/Singapore)
        # This ensures dt.year/month/day/hour etc. return values matching pandas
        # First parse the datetime (handles both string and DateTime types), then convert to UTC
        sql_expr = f"{ch_func}(toTimezone(parseDateTimeBestEffort(toString({source_sql})), 'UTC'))"

        # Handle dayofweek adjustment (chDB is 1-7 Monday, pandas is 0-6 Monday)
        if expr.property_name in ('dayofweek', 'weekday'):
            sql_expr = f"({sql_expr} - 1)"

        # Execute via chDB
        result = self._execute_sql_expression(sql_expr)

        # Align with pandas: preserve column name and convert dtype to int32
        if original_name:
            result = result.rename(original_name)

        # Convert dtype to match pandas (int32 for year/month/day/hour/minute/second)
        # Use nullable Int32 to handle NaT values (which become NA in result)
        if expr.property_name in (
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'dayofweek',
            'weekday',
            'dayofyear',
            'quarter',
            'week',
            'weekofyear',
        ):
            # Use nullable Int32 dtype to handle NA values from NaT
            try:
                result = result.astype('int32')
            except (ValueError, TypeError):
                # If conversion fails (e.g., NA values), use nullable Int32
                result = result.astype('Int32')

        return result

    def _evaluate_datetime_method(self, expr) -> pd.Series:
        """
        Evaluate a DateTimeMethodExpr using function_config to select engine.
        """
        from .function_executor import function_config

        method_name = expr.method_name

        # Check if this method should use pandas
        if function_config.should_use_pandas(method_name):
            self._logger.debug("[ExprEval] DateTime method '%s' -> Pandas", method_name)
            return self._evaluate_datetime_method_pandas(expr)
        else:
            self._logger.debug("[ExprEval] DateTime method '%s' -> chDB", method_name)
            return self._evaluate_datetime_method_chdb(expr)

    def _evaluate_datetime_method_pandas(self, expr) -> pd.Series:
        """Evaluate datetime method using pandas .dt accessor."""
        source_series = self.evaluate(expr.source_expr)

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(source_series):
            if source_series.dtype == 'object' or pd.api.types.is_string_dtype(source_series):
                try:
                    source_series = pd.to_datetime(source_series, errors='coerce')
                except Exception:
                    pass

        # Get method from .dt accessor
        dt_accessor = source_series.dt

        # Handle special method names that differ from pandas
        method_name = expr.method_name
        if method_name == 'floor_dt':
            method_name = 'floor'
        elif method_name == 'ceil_dt':
            method_name = 'ceil'
        elif method_name == 'round_dt':
            method_name = 'round'

        method = getattr(dt_accessor, method_name)
        return method(*expr.args, **expr.kwargs)

    def _evaluate_datetime_method_chdb(self, expr) -> pd.Series:
        """Evaluate datetime method using chDB SQL function."""
        from .expressions import DateTimeMethodExpr

        ch_func = DateTimeMethodExpr.CHDB_FUNCTION_MAP.get(expr.method_name)
        if not ch_func:
            # No chDB mapping, fall back to pandas
            self._logger.debug("[ExprEval] No chDB mapping for method '%s', using pandas", expr.method_name)
            return self._evaluate_datetime_method_pandas(expr)

        source_sql = expr.source_expr.to_sql(quote_char='"')

        # For floor/ceil/round/normalize, need to handle timezone properly
        # Use toTimezone(..., 'UTC') to ensure correct results matching pandas
        if expr.method_name in ('floor_dt', 'ceil_dt', 'round_dt', 'normalize'):
            # Build a modified expression with UTC timezone handling
            # First convert source to UTC to avoid local timezone issues
            utc_source = f"toTimezone(parseDateTimeBestEffort(toString({source_sql})), 'UTC')"
            sql_expr = self._build_datetime_method_sql(expr, utc_source)
            result = self._execute_sql_expression(sql_expr)

            # Post-process result to match pandas dtype
            # 1. Remove UTC timezone (convert to naive datetime)
            # 2. Ensure datetime64[ns] resolution
            if hasattr(result.dt, 'tz') and result.dt.tz is not None:
                result = result.dt.tz_localize(None)
            # Convert to datetime64[ns] to match pandas
            if result.dtype == 'datetime64[s]' or str(result.dtype).startswith('datetime64[s'):
                result = result.astype('datetime64[ns]')
            return result

        # Build SQL expression for other methods
        if expr.method_name == 'strftime' and expr.args:
            fmt = expr.args[0]
            sql_expr = f"{ch_func}(parseDateTimeBestEffort(toString({source_sql})), '{fmt}')"
        else:
            sql_expr = f"{ch_func}(parseDateTimeBestEffort(toString({source_sql})))"

        return self._execute_sql_expression(sql_expr)

    def _build_datetime_method_sql(self, expr, source_sql: str) -> str:
        """Build SQL for datetime methods (floor/ceil/round/normalize) with proper source."""
        from .expressions import DateTimeMethodExpr

        method_name = expr.method_name

        if method_name == 'normalize':
            return f"toStartOfDay({source_sql})"

        freq = expr.args[0] if expr.args else 'D'
        floor_func = DateTimeMethodExpr.FREQ_TO_FLOOR_FUNC.get(freq, 'toStartOfDay')

        if method_name == 'floor_dt':
            return f"{floor_func}({source_sql})"

        add_info = DateTimeMethodExpr.FREQ_TO_ADD_FUNC.get(freq, ('addDays', 1, 43200))
        add_func, amount, half_seconds = add_info
        floor_expr = f"{floor_func}({source_sql})"
        ceil_expr = f"{add_func}({floor_expr}, {amount})"

        if method_name == 'ceil_dt':
            # ceil = if(floor(x) == x, x, floor(x) + 1 unit)
            return f"if({floor_expr} = {source_sql}, {source_sql}, {ceil_expr})"

        if method_name == 'round_dt':
            # round = if(diff >= half_interval, ceil, floor)
            diff_seconds = f"dateDiff('second', {floor_expr}, {source_sql})"
            return f"if({diff_seconds} >= {half_seconds}, {ceil_expr}, {floor_expr})"

        return f"{floor_func}({source_sql})"

    def _evaluate_isocalendar_component(self, expr) -> pd.Series:
        """Evaluate an IsoCalendarComponentExpr - returns ISO calendar component."""
        from .function_executor import function_config

        component = expr.component

        # Check if this should use pandas
        if function_config.should_use_pandas('isocalendar'):
            self._logger.debug("[ExprEval] isocalendar.%s -> Pandas", component)
            return self._evaluate_isocalendar_component_pandas(expr)
        else:
            self._logger.debug("[ExprEval] isocalendar.%s -> chDB", component)
            return self._evaluate_isocalendar_component_chdb(expr)

    def _evaluate_isocalendar_component_pandas(self, expr) -> pd.Series:
        """Evaluate isocalendar component using pandas."""
        source_series = self.evaluate(expr.source_expr)

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(source_series):
            if source_series.dtype == 'object' or pd.api.types.is_string_dtype(source_series):
                try:
                    source_series = pd.to_datetime(source_series, errors='coerce')
                except Exception:
                    pass

        # Get isocalendar DataFrame
        isocal = source_series.dt.isocalendar()
        return isocal[expr.component]

    def _evaluate_isocalendar_component_chdb(self, expr) -> pd.Series:
        """Evaluate isocalendar component using chDB SQL function."""
        from .expressions import IsoCalendarComponentExpr

        source_sql = expr.source_expr.to_sql(quote_char='"')

        ch_func = IsoCalendarComponentExpr.CHDB_FUNCTION_MAP.get(expr.component)
        if not ch_func:
            # No chDB mapping, fall back to pandas
            self._logger.debug("[ExprEval] No chDB mapping for isocalendar '%s', using pandas", expr.component)
            return self._evaluate_isocalendar_component_pandas(expr)

        # Build SQL expression
        # Use parseDateTimeBestEffort to handle various datetime formats
        if expr.component == 'day':
            # toDayOfWeek needs mode=0 for Monday=1 (ISO standard: 1-7)
            sql_expr = f"toUInt32({ch_func}(parseDateTimeBestEffort(toString({source_sql})), 0))"
        else:
            sql_expr = f"toUInt32({ch_func}(parseDateTimeBestEffort(toString({source_sql}))))"

        return self._execute_sql_expression(sql_expr)

    def _execute_sql_expression(self, sql_expr: str) -> pd.Series:
        """Execute a SQL expression against the current DataFrame using chDB."""
        from .executor import get_executor

        executor = get_executor()
        return executor.execute_expression(sql_expr, self.df)

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

        # Get original column's null mask for restoration
        # (chDB converts NULL to empty string, need to restore)
        original_name = self._extract_column_name(expr)
        original_null_mask = None
        if original_name and original_name in self.df.columns:
            original_null_mask = self.df[original_name].isna()

        # Use centralized executor
        executor = get_executor()
        result = executor.execute_expression(sql_expr, self.df)

        # Workaround for chDB NULL handling
        # NOTE: chDB 4.0.0b3 has improved NULL handling (see GitHub issue #447),
        # but some operations still need this workaround for pandas compatibility.
        # Skip NULL restoration for functions that SHOULD preserve/report NULL values:
        # - isNull/isNotNull: These functions detect NULL, so their result should not be NaN
        # - ifNull: This function fills NULL, so its result should not have NaN at NULL positions
        # - toBool(isNull/isNotNull/ifNull): Wrapper for bool dtype compatibility
        should_restore_nulls = not self._is_null_handling_function(expr)

        if should_restore_nulls and original_null_mask is not None and original_null_mask.any():
            result = self._restore_nulls(result, original_null_mask)

        # Preserve original column name for accessor operations
        # For functions like upper(name), the series name should be 'name', not '__result__'
        if original_name and hasattr(result, 'name'):
            result = result.rename(original_name)

        # Apply dtype corrections for functions where chDB returns different types than pandas
        # (e.g., abs() on signed integers returns unsigned in chDB but signed in pandas)
        result = self._apply_dtype_correction(expr, result, original_name)

        return result

    def _is_null_handling_function(self, expr) -> bool:
        """
        Check if expression is a null-handling function that should NOT have
        its NULL values restored.

        This includes:
        - isNull/isNotNull: Detect NULL values, return boolean
        - ifNull: Fill NULL values
        - toBool(isNull/isNotNull/ifNull): Wrapper for bool dtype compatibility

        Args:
            expr: Expression to check

        Returns:
            True if this is a null-handling function
        """
        null_handling_funcs = {'isnull', 'isnotnull', 'ifnull'}
        func_name = getattr(expr, 'name', '').lower()

        # Direct null-handling function
        if func_name in null_handling_funcs:
            return True

        # toBool wrapper around null-handling function
        if func_name == 'tobool':
            args = getattr(expr, 'args', [])
            if args:
                inner_func_name = getattr(args[0], 'name', '').lower()
                if inner_func_name in null_handling_funcs:
                    return True

        return False

    def _restore_nulls(self, result: pd.Series, null_mask: pd.Series) -> pd.Series:
        """
        Restore NULL values in result based on original null mask.

        Workaround for chDB issue #447 where NULL becomes empty string.

        Args:
            result: Result series from chDB
            null_mask: Boolean series indicating original NULL positions

        Returns:
            Series with NaN restored at original NULL positions
        """
        if len(result) != len(null_mask):
            self._logger.debug("[ExprEval] Length mismatch, skipping NULL restoration")
            return result

        # Make a copy to avoid modifying the original
        result = result.copy()

        # For string columns, empty strings at NULL positions should become NaN
        if result.dtype == 'object':
            # Set NaN where original was null
            result.loc[null_mask] = None
        elif pd.api.types.is_numeric_dtype(result.dtype):
            # For numeric columns, use np.nan
            result = result.astype(float)
            result.loc[null_mask] = np.nan

        return result

    def _extract_column_name(self, expr) -> str:
        """
        Extract the original column name from an expression.

        For Field expressions, returns the field name.
        For Function expressions, recursively looks for the first Field argument.

        Args:
            expr: Expression to extract column name from

        Returns:
            Column name if found, None otherwise
        """
        from .functions import Function

        if isinstance(expr, Field):
            return expr.name
        elif isinstance(expr, Function):
            # Look for Field in arguments
            for arg in expr.args:
                if isinstance(arg, Field):
                    return arg.name
                elif isinstance(arg, Expression):
                    name = self._extract_column_name(arg)
                    if name:
                        return name
        return None

    def _apply_dtype_correction(self, expr, result: pd.Series, original_col_name: Optional[str]) -> pd.Series:
        """
        Apply dtype corrections for functions where chDB returns different types than pandas.

        Uses the centralized DtypeCorrectionRegistry to apply corrections based on
        configurable rules. This handles various dtype mismatches:
        - abs(): unsigned → signed for signed int input
        - sign(): int8 → preserve input type
        - pow(): float64 → int for integer input
        - arithmetic ops: type width preservation
        - And more...

        Args:
            expr: The expression that was evaluated
            result: The result Series from chDB
            original_col_name: The original column name (for dtype lookup)

        Returns:
            Series with corrected dtype if needed
        """
        from .functions import Function
        from .dtype_correction import dtype_registry

        # Only process Function expressions
        if not isinstance(expr, Function):
            return result

        func_name = expr.name.lower()

        # Get the input column's dtype
        if not original_col_name or original_col_name not in self.df.columns:
            return result

        input_dtype = str(self.df[original_col_name].dtype)

        # Apply correction using the registry
        return dtype_registry.apply_correction(func_name, result, input_dtype)


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
