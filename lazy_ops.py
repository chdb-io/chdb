"""
Lazy operation system for DataStore.

This module implements a lazy evaluation system where operations are recorded
and only executed when materialization is triggered (e.g., print, to_df()).
"""

from typing import Any, Dict, List, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import pandas as pd

from .expressions import Expression, Field, Literal, ArithmeticExpression
from .config import get_logger

if TYPE_CHECKING:
    from .core import DataStore


class LazyOp(ABC):
    """
    Base class for lazy operations.

    Each operation knows how to:
    1. Describe itself (for explain())
    2. Execute itself on a DataFrame
    3. Optimize itself (future)
    """

    def __init__(self):
        self._logger = get_logger()

    @abstractmethod
    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """
        Execute this operation on a DataFrame.

        Args:
            df: Input DataFrame
            context: DataStore instance (for accessing other state)

        Returns:
            Modified DataFrame
        """
        pass

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this operation."""
        pass

    def can_push_to_sql(self) -> bool:
        """Whether this operation can be pushed down to SQL layer."""
        return False

    def _log_execute(self, op_name: str, details: str = None):
        """Log operation execution at DEBUG level."""
        if details:
            self._logger.debug("[LazyOp] Executing %s: %s", op_name, details)
        else:
            self._logger.debug("[LazyOp] Executing %s", op_name)


class LazyColumnAssignment(LazyOp):
    """
    Lazy column assignment: df[col] = expr

    Example:
        nat["n_nationkey"] = nat["n_nationkey"] - 1
    """

    def __init__(self, column: str, expr: Union[Expression, Any]):
        super().__init__()
        self.column = column
        self.expr = expr

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute column assignment."""
        self._log_execute("ColumnAssignment", f"column='{self.column}'")
        df = df.copy()  # Don't modify input

        # Evaluate expression
        value = self._evaluate_expr(self.expr, df, context)

        # Assign
        df[self.column] = value
        self._logger.debug("[LazyOp]   -> DataFrame shape after assignment: %s", df.shape)
        return df

    def _evaluate_expr(self, expr, df: pd.DataFrame, context: 'DataStore'):
        """
        Recursively evaluate an expression.

        Supports:
        - Field references: nat["col"]
        - Literals: 1, "hello"
        - Arithmetic: nat["a"] + 1
        - Series: direct pandas Series
        """
        if isinstance(expr, Field):
            # Column reference
            return df[expr.name]

        elif isinstance(expr, Literal):
            # Literal value
            return expr.value

        elif isinstance(expr, ArithmeticExpression):
            # Arithmetic operation: recursively evaluate
            left = self._evaluate_expr(expr.left, df, context)
            right = self._evaluate_expr(expr.right, df, context)

            # Apply operator
            if expr.operator == '+':
                return left + right
            elif expr.operator == '-':
                return left - right
            elif expr.operator == '*':
                return left * right
            elif expr.operator == '/':
                return left / right
            elif expr.operator == '//':
                return left // right
            elif expr.operator == '%':
                return left % right
            elif expr.operator == '**':
                return left**right
            else:
                raise ValueError(f"Unknown operator: {expr.operator}")

        elif isinstance(expr, pd.Series):
            # Direct pandas Series
            return expr

        else:
            # Scalar value
            return expr

    def describe(self) -> str:
        if isinstance(self.expr, Expression):
            expr_str = self.expr.to_sql(quote_char='"')
        else:
            expr_str = repr(self.expr)
        return f"Assign column '{self.column}' = {expr_str}"


class LazyColumnSelection(LazyOp):
    """
    Lazy column selection: df[["col1", "col2"]]

    Example:
        nat[["n_name", "n_nationkey"]]
    """

    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("ColumnSelection", f"columns={self.columns}")
        result = df[self.columns]
        self._logger.debug("[LazyOp]   -> DataFrame shape after selection: %s", result.shape)
        return result

    def describe(self) -> str:
        return f"Select columns: {', '.join(self.columns)}"

    def can_push_to_sql(self) -> bool:
        # Could be pushed to SQL SELECT
        return True


class LazyDropColumns(LazyOp):
    """Drop columns: df.drop(columns=[...])"""

    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("DropColumns", f"columns={self.columns}")
        result = df.drop(columns=self.columns)
        self._logger.debug("[LazyOp]   -> DataFrame shape after drop: %s", result.shape)
        return result

    def describe(self) -> str:
        return f"Drop columns: {', '.join(self.columns)}"


class LazyRenameColumns(LazyOp):
    """Rename columns: df.rename(columns={...})"""

    def __init__(self, mapping: Dict[str, str]):
        super().__init__()
        self.mapping = mapping

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("RenameColumns", f"mapping={self.mapping}")
        result = df.rename(columns=self.mapping)
        self._logger.debug("[LazyOp]   -> New columns: %s", list(result.columns))
        return result

    def describe(self) -> str:
        renames = ', '.join(f"{old}→{new}" for old, new in self.mapping.items())
        return f"Rename columns: {renames}"


class LazyAddPrefix(LazyOp):
    """Add prefix to all columns."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AddPrefix", f"prefix='{self.prefix}'")
        result = df.add_prefix(self.prefix)
        self._logger.debug("[LazyOp]   -> New columns: %s", list(result.columns))
        return result

    def describe(self) -> str:
        return f"Add prefix '{self.prefix}' to all columns"


class LazyAddSuffix(LazyOp):
    """Add suffix to all columns."""

    def __init__(self, suffix: str):
        super().__init__()
        self.suffix = suffix

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AddSuffix", f"suffix='{self.suffix}'")
        result = df.add_suffix(self.suffix)
        self._logger.debug("[LazyOp]   -> New columns: %s", list(result.columns))
        return result

    def describe(self) -> str:
        return f"Add suffix '{self.suffix}' to all columns"


class LazyFillNA(LazyOp):
    """Fill NA values."""

    def __init__(self, value, method=None):
        super().__init__()
        self.value = value
        self.method = method

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("FillNA", f"value={self.value}, method={self.method}")
        result = df.fillna(value=self.value, method=self.method)
        return result

    def describe(self) -> str:
        if self.method:
            return f"Fill NA with method '{self.method}'"
        return f"Fill NA with value {self.value}"


class LazyDropNA(LazyOp):
    """Drop rows with NA values."""

    def __init__(self, how='any', subset=None):
        super().__init__()
        self.how = how
        self.subset = subset

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("DropNA", f"how='{self.how}', subset={self.subset}")
        rows_before = len(df)
        result = df.dropna(how=self.how, subset=self.subset)
        self._logger.debug(
            "[LazyOp]   -> Dropped %d rows (from %d to %d)", rows_before - len(result), rows_before, len(result)
        )
        return result

    def describe(self) -> str:
        desc = f"Drop NA (how='{self.how}')"
        if self.subset:
            desc += f" on columns: {', '.join(self.subset)}"
        return desc


class LazyAsType(LazyOp):
    """Cast column types."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AsType", f"dtype={self.dtype}")
        result = df.astype(self.dtype)
        self._logger.debug("[LazyOp]   -> New dtypes: %s", dict(result.dtypes))
        return result

    def describe(self) -> str:
        return f"Cast types: {self.dtype}"


class LazySQLSnapshot(LazyOp):
    """
    A snapshot of SQL state at a point in time.

    This stores both the SQL description and the original condition/parameters,
    so it can be executed either as SQL or on a DataFrame.

    Example:
        users = ds.from_file('users.csv')
        users = users.select('name', 'age')  # Records LazySQLSnapshot
        users['doubled'] = users['age'] * 2  # Records LazyColumnAssignment
        users = users.filter(users['age'] > 25)  # Records another LazySQLSnapshot
    """

    def __init__(
        self,
        op_type: str,
        description: str,
        condition=None,
        fields=None,
        ascending=True,
        limit_value=None,
        offset_value=None,
    ):
        """
        Args:
            op_type: Type of SQL operation (e.g., 'SELECT', 'WHERE', 'ORDER BY', 'LIMIT', 'OFFSET')
            description: Human-readable description
            condition: Original condition object for WHERE operations
            fields: List of fields for SELECT/ORDER BY operations
            ascending: Sort direction for ORDER BY
            limit_value: Value for LIMIT
            offset_value: Value for OFFSET
        """
        super().__init__()
        self.op_type = op_type
        self.description = description
        self.condition = condition
        self.fields = fields
        self.ascending = ascending
        self.limit_value = limit_value
        self.offset_value = offset_value

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute this filter/transform operation on a DataFrame using pandas."""
        self._log_execute(f"Pandas {self.op_type}", self.description)
        rows_before = len(df)

        if self.op_type == 'SELECT' and self.fields:
            # Select specific columns
            cols = [f if isinstance(f, str) else f.name for f in self.fields]
            existing_cols = [c for c in cols if c in df.columns]
            # Log condition (originally SQL-like, but executed via pandas)
            try:
                fields_sql = ', '.join(
                    f.to_sql(quote_char='"') if hasattr(f, 'to_sql') else f'"{f}"' for f in self.fields
                )
                self._logger.debug("[Pandas]   -> df[%s]", existing_cols)
            except Exception:
                pass
            if existing_cols:
                result = df[existing_cols]
                self._logger.debug("[Pandas]   -> Selected columns: %s", existing_cols)
                return result
            return df
        elif self.op_type == 'FILTER' and self.condition is not None:
            # Log the condition (executed via pandas boolean mask)
            try:
                condition_sql = self.condition.to_sql(quote_char='"')
                self._logger.debug("[Pandas]   -> Condition: %s", condition_sql)
            except Exception:
                pass
            # Apply filter condition on DataFrame using pandas
            result = self._apply_condition(df, self.condition, context)
            self._logger.debug("[Pandas]   -> df[mask]: %d -> %d rows", rows_before, len(result))
            return result
        elif self.op_type == 'ORDER BY' and self.fields:
            # Sort DataFrame
            cols = [f if isinstance(f, str) else f.name for f in self.fields]
            existing_cols = [c for c in cols if c in df.columns]
            # Log sort info
            try:
                direction = 'ascending' if self.ascending else 'descending'
                self._logger.debug("[Pandas]   -> df.sort_values(by=%s, ascending=%s)", existing_cols, self.ascending)
            except Exception:
                pass
            if existing_cols:
                result = df.sort_values(by=existing_cols, ascending=self.ascending)
                self._logger.debug("[Pandas]   -> Sorted by: %s (%s)", existing_cols, direction)
                return result
            return df
        elif self.op_type == 'LIMIT' and self.limit_value is not None:
            self._logger.debug("[Pandas]   -> df.head(%d)", self.limit_value)
            result = df.head(self.limit_value)
            self._logger.debug("[Pandas]   -> Limited to %d rows", self.limit_value)
            return result
        elif self.op_type == 'OFFSET' and self.offset_value is not None:
            self._logger.debug("[Pandas]   -> df.iloc[%d:]", self.offset_value)
            result = df.iloc[self.offset_value :]
            self._logger.debug("[Pandas]   -> Offset by %d rows", self.offset_value)
            return result
        return df

    def _apply_condition(self, df: pd.DataFrame, condition, context: 'DataStore') -> pd.DataFrame:
        """Apply a condition to filter a DataFrame."""
        from .conditions import Condition, CompoundCondition, NotCondition
        from .expressions import Expression, Field

        if isinstance(condition, CompoundCondition):
            if condition.operator == 'AND':
                # Apply both conditions
                result = self._apply_condition(df, condition.left, context)
                return self._apply_condition(result, condition.right, context)
            elif condition.operator == 'OR':
                # Get masks for both and combine with OR
                left_mask = self._get_condition_mask(df, condition.left, context)
                right_mask = self._get_condition_mask(df, condition.right, context)
                return df[left_mask | right_mask]
        elif isinstance(condition, NotCondition):
            mask = self._get_condition_mask(df, condition.condition, context)
            return df[~mask]
        elif isinstance(condition, Condition):
            mask = self._get_condition_mask(df, condition, context)
            return df[mask]
        return df

    def _get_condition_mask(self, df: pd.DataFrame, condition, context: 'DataStore'):
        """Get a boolean mask for a condition."""
        from .conditions import Condition, BinaryCondition, CompoundCondition, NotCondition
        from .conditions import InCondition, BetweenCondition, LikeCondition, UnaryCondition

        if not isinstance(condition, Condition):
            return pd.Series([True] * len(df), index=df.index)

        if isinstance(condition, CompoundCondition):
            left_mask = self._get_condition_mask(df, condition.left, context)
            right_mask = self._get_condition_mask(df, condition.right, context)
            if condition.operator == 'AND':
                return left_mask & right_mask
            elif condition.operator == 'OR':
                return left_mask | right_mask
            return left_mask

        if isinstance(condition, NotCondition):
            return ~self._get_condition_mask(df, condition.condition, context)

        if isinstance(condition, BinaryCondition):
            left = self._evaluate_operand(df, condition.left, context)
            right = self._evaluate_operand(df, condition.right, context)

            op = condition.operator
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
            elif op == 'LIKE':
                # Convert SQL LIKE to regex
                pattern = str(right).replace('%', '.*').replace('_', '.')
                return left.astype(str).str.match(f'^{pattern}$', case=True)
            elif op == 'IN':
                return left.isin(right if isinstance(right, (list, tuple)) else [right])
            elif op == 'IS':
                # Handle IS NULL
                if right is None or str(right).upper() == 'NULL':
                    return left.isna()
                return left == right
            return pd.Series([True] * len(df), index=df.index)

        if isinstance(condition, InCondition):
            left = self._evaluate_operand(df, condition.field, context)
            values = condition.values
            if condition.negated:
                return ~left.isin(values)
            return left.isin(values)

        if isinstance(condition, BetweenCondition):
            field = self._evaluate_operand(df, condition.field, context)
            low = self._evaluate_operand(df, condition.low, context)
            high = self._evaluate_operand(df, condition.high, context)
            if condition.negated:
                return (field < low) | (field > high)
            return (field >= low) & (field <= high)

        if isinstance(condition, UnaryCondition):
            field = self._evaluate_operand(df, condition.field, context)
            if condition.operator == 'IS NULL':
                return field.isna()
            elif condition.operator == 'IS NOT NULL':
                return field.notna()

        return pd.Series([True] * len(df), index=df.index)

    def _evaluate_operand(self, df: pd.DataFrame, operand, context: 'DataStore'):
        """Evaluate an operand (field, expression, or literal value)."""
        from .expressions import Expression, Field, ArithmeticExpression, Literal

        if isinstance(operand, Literal):
            # Return the actual value from Literal
            return operand.value
        elif isinstance(operand, Field):
            if operand.name in df.columns:
                return df[operand.name]
            return operand.name  # Return as literal if column doesn't exist
        elif isinstance(operand, ArithmeticExpression):
            left = self._evaluate_operand(df, operand.left, context)
            right = self._evaluate_operand(df, operand.right, context)
            op = operand.operator
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left / right
            elif op == '//':
                return left // right
            elif op == '**':
                return left**right
            elif op == '%':
                return left % right
        elif isinstance(operand, Expression):
            # Handle other expression types - try to get value if available
            if hasattr(operand, 'value'):
                return operand.value
            return operand
        return operand  # Return literal value as-is

    def describe(self) -> str:
        return f"{self.op_type}: {self.description}"

    def can_push_to_sql(self) -> bool:
        return True


class LazyDataFrameSource(LazyOp):
    """
    A lazy operation that provides a DataFrame as the data source.

    This is used when we need to wrap an existing DataFrame into the lazy pipeline,
    for example after executing SQL on a materialized DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Return the stored DataFrame, ignoring the input."""
        self._log_execute("DataFrameSource", f"shape={self._df.shape}")
        return self._df

    def describe(self) -> str:
        return f"DataFrame source (shape: {self._df.shape})"


# Add more operations as needed...
