"""
CASE WHEN expression builder for DataStore.

Provides a fluent API for building SQL CASE WHEN expressions:

    ds['grade'] = ds.when(ds['score'] >= 90, 'A') \\
                    .when(ds['score'] >= 80, 'B') \\
                    .when(ds['score'] >= 60, 'C') \\
                    .otherwise('F')

This compiles to SQL:
    CASE WHEN score >= 90 THEN 'A'
         WHEN score >= 80 THEN 'B'
         WHEN score >= 60 THEN 'C'
         ELSE 'F'
    END

Execution Engine:
- Default: SQL (chDB) for better performance
- Can be configured via function_config.use_pandas('when') to use pandas (np.select)
"""

from typing import Any, List, Tuple, Optional, TYPE_CHECKING, Union
import numpy as np
import pandas as pd

from .expressions import Expression, Literal
from .conditions import Condition
from .utils import format_identifier
from .config import get_logger

if TYPE_CHECKING:
    from .core import DataStore
    from .column_expr import ColumnExpr

__all__ = ['CaseWhenBuilder', 'CaseWhenExpr']

_logger = get_logger()


class CaseWhenExpr(Expression):
    """
    SQL CASE WHEN expression that can be assigned to a column.

    This is the final expression returned by CaseWhenBuilder.otherwise().
    It can:
    1. Generate SQL: CASE WHEN cond1 THEN val1 WHEN cond2 THEN val2 ... ELSE default END
    2. Execute via pandas: np.select(conditions, choices, default)

    Example:
        >>> expr = CaseWhenExpr(
        ...     cases=[(ds['score'] >= 90, 'A'), (ds['score'] >= 80, 'B')],
        ...     default='F',
        ...     datastore=ds
        ... )
        >>> ds['grade'] = expr
    """

    def __init__(
        self,
        cases: List[Tuple[Any, Any]],
        default: Any,
        datastore: 'DataStore',
        alias: Optional[str] = None,
    ):
        """
        Initialize a CaseWhenExpr.

        Args:
            cases: List of (condition, value) tuples
            default: Default value when no condition matches (ELSE clause)
            datastore: Reference to the DataStore for context
            alias: Optional output column alias
        """
        super().__init__(alias=alias)
        self._cases = cases
        self._default = default
        self._datastore = datastore

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """
        Generate SQL CASE WHEN expression.

        Returns:
            SQL string like: CASE WHEN score >= 90 THEN 'A' WHEN score >= 80 THEN 'B' ELSE 'F' END
        """
        parts = ["CASE"]

        for condition, value in self._cases:
            cond_sql = self._condition_to_sql(condition, quote_char)
            val_sql = self._value_to_sql(value)
            parts.append(f"WHEN {cond_sql} THEN {val_sql}")

        parts.append(f"ELSE {self._value_to_sql(self._default)}")
        parts.append("END")

        result = " ".join(parts)

        if self.alias:
            result = f"{result} AS {format_identifier(self.alias, quote_char)}"

        return result

    def _condition_to_sql(self, condition: Any, quote_char: str) -> str:
        """Convert a condition to SQL."""
        from .column_expr import ColumnExpr

        if isinstance(condition, Condition):
            return condition.to_sql(quote_char=quote_char)
        elif isinstance(condition, ColumnExpr):
            # ColumnExpr wrapping a condition
            if condition._expr is not None:
                if isinstance(condition._expr, Condition):
                    return condition._expr.to_sql(quote_char=quote_char)
                return condition._expr.to_sql(quote_char=quote_char)
            raise ValueError(f"ColumnExpr does not contain a valid condition: {condition}")
        elif hasattr(condition, 'to_sql'):
            return condition.to_sql(quote_char=quote_char)
        else:
            raise ValueError(f"Cannot convert {type(condition)} to SQL condition")

    def _value_to_sql(self, value: Any) -> str:
        """Convert a value to SQL literal."""
        from .column_expr import ColumnExpr

        if value is None:
            return "NULL"
        elif isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, (int, float)):
            if pd.isna(value):
                return "NULL"
            return str(value)
        elif isinstance(value, Expression):
            return value.to_sql(quote_char='"')
        elif isinstance(value, ColumnExpr):
            if value._expr is not None:
                return value._expr.to_sql(quote_char='"')
            return f'"{value._column_name}"'
        else:
            # Try to convert to string
            return f"'{str(value)}'"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """
        Execute the CASE WHEN expression using the configured engine.

        Engine selection is based on function_config settings:
        - Default: SQL (chDB) for better performance
        - Can use pandas (np.select) if configured via function_config.use_pandas('when')

        Args:
            df: DataFrame to evaluate against

        Returns:
            pd.Series with the computed values
        """
        from .function_executor import function_config

        if function_config.should_use_pandas('when'):
            _logger.debug("[CaseWhenExpr] Using Pandas engine (np.select)")
            return self._evaluate_via_pandas(df)
        else:
            _logger.debug("[CaseWhenExpr] Using SQL engine (chDB)")
            return self._evaluate_via_chdb(df)

    def _evaluate_via_chdb(self, df: pd.DataFrame) -> pd.Series:
        """
        Execute CASE WHEN via chDB SQL engine.

        Uses the SQL representation of the expression and executes it via chDB's
        Python() table function for potentially better performance on large datasets.

        Args:
            df: DataFrame to evaluate against

        Returns:
            pd.Series with the computed values
        """
        from .executor import get_executor

        # Handle empty DataFrame - chDB may not handle this well
        if len(df) == 0:
            _logger.debug("[CaseWhenExpr] Empty DataFrame, falling back to Pandas")
            return self._evaluate_via_pandas(df)

        # Generate SQL for the CASE WHEN expression
        sql_expr = self.to_sql(quote_char='"')
        _logger.debug("[CaseWhenExpr] SQL: %s", sql_expr)

        try:
            # Execute via chDB
            executor = get_executor()
            result = executor.execute_expression(sql_expr, df)

            # Ensure result has the correct index
            if isinstance(result, pd.Series):
                result.index = df.index

            return result
        except Exception as e:
            # Fallback to Pandas if chDB execution fails
            # (e.g., type incompatibility issues)
            _logger.debug("[CaseWhenExpr] chDB execution failed, falling back to Pandas: %s", e)
            return self._evaluate_via_pandas(df)

    def _evaluate_via_pandas(self, df: pd.DataFrame) -> pd.Series:
        """
        Execute CASE WHEN via pandas using np.select.

        This is the pandas fallback execution path.

        Args:
            df: DataFrame to evaluate against

        Returns:
            pd.Series with the computed values
        """
        from .expression_evaluator import ExpressionEvaluator
        from .column_expr import ColumnExpr

        # Build conditions list for np.select
        conditions = []
        choices = []

        evaluator = ExpressionEvaluator(df, self._datastore)

        def _evaluate_value(val):
            """Evaluate a value that could be Expression, ColumnExpr, or literal."""
            if isinstance(val, ColumnExpr):
                # ColumnExpr - evaluate using evaluator
                result = evaluator.evaluate(val)
                if isinstance(result, pd.Series):
                    return result.values
                return result
            elif isinstance(val, Expression):
                result = evaluator.evaluate(val)
                if isinstance(result, pd.Series):
                    return result.values
                return result
            else:
                # Literal value
                return val

        for condition, value in self._cases:
            # Evaluate condition to boolean Series
            cond_result = evaluator.evaluate(condition)
            if isinstance(cond_result, pd.Series):
                conditions.append(cond_result.values)
            else:
                conditions.append(cond_result)

            # Evaluate value (could be Expression, ColumnExpr, or literal)
            choices.append(_evaluate_value(value))

        # Evaluate default
        default_val = _evaluate_value(self._default)

        # Use np.select for efficient conditional selection
        result = np.select(conditions, choices, default=default_val)

        return pd.Series(result, index=df.index)

    def execution_engine(self) -> str:
        """
        Return which engine will execute this expression.

        Returns:
            'chDB' if using SQL engine (default)
            'Pandas' if configured to use pandas via function_config
        """
        from .function_executor import function_config

        if function_config.should_use_pandas('when'):
            return 'Pandas'
        return 'chDB'

    def __repr__(self) -> str:
        cases_repr = ", ".join(f"({c}, {v})" for c, v in self._cases)
        return f"CaseWhenExpr(cases=[{cases_repr}], default={self._default})"


class CaseWhenBuilder:
    """
    Builder for SQL CASE WHEN expressions.

    Provides a fluent API for building conditional expressions:

        ds['grade'] = ds.when(ds['score'] >= 90, 'A') \\
                        .when(ds['score'] >= 80, 'B') \\
                        .when(ds['score'] >= 60, 'C') \\
                        .otherwise('F')

    The builder collects (condition, value) pairs until .otherwise() is called,
    which returns the final CaseWhenExpr that can be assigned to a column.

    Note: .otherwise() MUST be called to complete the expression.
    """

    def __init__(self, datastore: 'DataStore'):
        """
        Initialize the builder.

        Args:
            datastore: Reference to the DataStore for context
        """
        self._datastore = datastore
        self._cases: List[Tuple[Any, Any]] = []

    def when(self, condition: Any, value: Any) -> 'CaseWhenBuilder':
        """
        Add a WHEN condition.

        Args:
            condition: Boolean condition (e.g., ds['score'] >= 90)
            value: Value to use when condition is True

        Returns:
            self for method chaining

        Example:
            >>> builder.when(ds['score'] >= 90, 'A')
            ...        .when(ds['score'] >= 80, 'B')
        """
        self._cases.append((condition, value))
        return self

    def otherwise(self, default: Any) -> CaseWhenExpr:
        """
        Set the ELSE value and return the final expression.

        This method MUST be called to complete the CASE WHEN expression.

        Args:
            default: Default value when no condition matches

        Returns:
            CaseWhenExpr that can be assigned to a column

        Example:
            >>> expr = ds.when(ds['score'] >= 90, 'A') \\
            ...          .when(ds['score'] >= 80, 'B') \\
            ...          .otherwise('F')
            >>> ds['grade'] = expr
        """
        if not self._cases:
            raise ValueError("CaseWhenBuilder requires at least one .when() condition before .otherwise()")

        return CaseWhenExpr(
            cases=self._cases.copy(),
            default=default,
            datastore=self._datastore,
        )

    def __repr__(self) -> str:
        cases_repr = ", ".join(f"({c}, {v})" for c, v in self._cases)
        return f"CaseWhenBuilder(cases=[{cases_repr}])"

