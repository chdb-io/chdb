"""
Lazy operation system for DataStore.

This module implements a lazy evaluation system where operations are recorded
and only executed when materialization is triggered (e.g., print, to_df()).
"""

from typing import Any, Dict, List, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
import pandas as pd

from .expressions import Expression, Field, Literal, ArithmeticExpression

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


class LazyColumnAssignment(LazyOp):
    """
    Lazy column assignment: df[col] = expr

    Example:
        nat["n_nationkey"] = nat["n_nationkey"] - 1
    """

    def __init__(self, column: str, expr: Union[Expression, Any]):
        self.column = column
        self.expr = expr

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute column assignment."""
        df = df.copy()  # Don't modify input

        # Evaluate expression
        value = self._evaluate_expr(self.expr, df, context)

        # Assign
        df[self.column] = value
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
        self.columns = columns

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df[self.columns]

    def describe(self) -> str:
        return f"Select columns: {', '.join(self.columns)}"

    def can_push_to_sql(self) -> bool:
        # Could be pushed to SQL SELECT
        return True


class LazyDropColumns(LazyOp):
    """Drop columns: df.drop(columns=[...])"""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.drop(columns=self.columns)

    def describe(self) -> str:
        return f"Drop columns: {', '.join(self.columns)}"


class LazyRenameColumns(LazyOp):
    """Rename columns: df.rename(columns={...})"""

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.rename(columns=self.mapping)

    def describe(self) -> str:
        renames = ', '.join(f"{old}→{new}" for old, new in self.mapping.items())
        return f"Rename columns: {renames}"


class LazyAddPrefix(LazyOp):
    """Add prefix to all columns."""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.add_prefix(self.prefix)

    def describe(self) -> str:
        return f"Add prefix '{self.prefix}' to all columns"


class LazyAddSuffix(LazyOp):
    """Add suffix to all columns."""

    def __init__(self, suffix: str):
        self.suffix = suffix

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.add_suffix(self.suffix)

    def describe(self) -> str:
        return f"Add suffix '{self.suffix}' to all columns"


class LazyFillNA(LazyOp):
    """Fill NA values."""

    def __init__(self, value, method=None):
        self.value = value
        self.method = method

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.fillna(value=self.value, method=self.method)

    def describe(self) -> str:
        if self.method:
            return f"Fill NA with method '{self.method}'"
        return f"Fill NA with value {self.value}"


class LazyDropNA(LazyOp):
    """Drop rows with NA values."""

    def __init__(self, how='any', subset=None):
        self.how = how
        self.subset = subset

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.dropna(how=self.how, subset=self.subset)

    def describe(self) -> str:
        desc = f"Drop NA (how='{self.how}')"
        if self.subset:
            desc += f" on columns: {', '.join(self.subset)}"
        return desc


class LazyAsType(LazyOp):
    """Cast column types."""

    def __init__(self, dtype):
        self.dtype = dtype

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        return df.astype(self.dtype)

    def describe(self) -> str:
        return f"Cast types: {self.dtype}"


class LazySQLSnapshot(LazyOp):
    """
    A snapshot of SQL state at a point in time.

    This is used to record the order of SQL operations relative to other lazy ops.
    It doesn't execute anything - the actual SQL execution happens in _materialize().

    Example:
        users = ds.from_file('users.csv')
        users = users.select('name', 'age')  # Records LazySQLSnapshot
        users['doubled'] = users['age'] * 2  # Records LazyColumnAssignment
        users = users.filter(users['age'] > 25)  # Records another LazySQLSnapshot
    """

    def __init__(self, op_type: str, description: str, sql_fragment: str = None):
        """
        Args:
            op_type: Type of SQL operation (e.g., 'select', 'filter', 'join', 'sort')
            description: Human-readable description
            sql_fragment: Optional SQL fragment for verbose output
        """
        self.op_type = op_type
        self.description = description
        self.sql_fragment = sql_fragment

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """SQL operations don't execute here - they're handled by _materialize()."""
        return df

    def describe(self) -> str:
        return f"SQL {self.op_type}: {self.description}"

    def can_push_to_sql(self) -> bool:
        return True


class LazyDataFrameSource(LazyOp):
    """
    A lazy operation that provides a DataFrame as the data source.

    This is used when we need to wrap an existing DataFrame into the lazy pipeline,
    for example after executing SQL on a materialized DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Return the stored DataFrame, ignoring the input."""
        return self._df

    def describe(self) -> str:
        return f"DataFrame source (shape: {self._df.shape})"


# Add more operations as needed...
