"""
Lazy operation system for DataStore.

This module implements a lazy evaluation system where operations are recorded
and only executed when execution is triggered (e.g., print, to_df()).
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
    3. Transform column list (for column tracking)
    4. Optimize itself (future)
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

    def execution_engine(self) -> str:
        """
        Return which engine will execute this operation.

        Returns:
            'chDB' - will use chDB SQL engine
            'Pandas' - will use Pandas operations
            'mixed' - uses both (e.g., column assignment with SQL function)

        Note:
            This method enables explain() to accurately report execution engines.
            Subclasses should override this if they can use chDB.
        """
        return 'Pandas'  # Default: most LazyOps use Pandas

    def transform_columns(self, columns: List[str]) -> List[str]:
        """
        Transform a column list based on this operation's effect.

        This allows tracking column changes through the lazy op chain without
        executing the operations. Used by _ensure_sql_source() and other
        methods that need to know the current column state.

        Args:
            columns: Current list of column names

        Returns:
            New list of column names after this operation would be applied.
            Default implementation returns columns unchanged.
        """
        return columns

    def _log_execute(self, op_name: str, details: str = None, prefix: str = "Pandas"):
        """Log operation execution at DEBUG level with indentation."""
        if details:
            self._logger.debug("    [%s] Executing %s: %s", prefix, op_name, details)
        else:
            self._logger.debug("    [%s] Executing %s", prefix, op_name)


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

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Add the assigned column if it doesn't exist."""
        if self.column not in columns:
            return columns + [self.column]
        return columns

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute column assignment using unified ExpressionEvaluator."""
        from .expression_evaluator import ExpressionEvaluator

        self._log_execute("ColumnAssignment", f"column='{self.column}'")
        df = df.copy()  # Don't modify input

        # Use unified expression evaluator (respects function_config)
        evaluator = ExpressionEvaluator(df, context)
        value = evaluator.evaluate(self.expr)

        # Assign
        df[self.column] = value
        self._logger.debug("      -> DataFrame shape after assignment: %s", df.shape)
        return df

    def describe(self) -> str:
        from .column_expr import ColumnExpr
        from .lazy_result import LazySeries

        if isinstance(self.expr, ColumnExpr):
            # Handle different ColumnExpr modes
            if self.expr._expr is not None:
                # Expression mode - use SQL representation
                expr_str = self.expr._expr.to_sql(quote_char='"')
            elif self.expr._method_name is not None:
                # Method mode - show method name
                expr_str = f"<ColumnExpr: {self.expr._method_name}()>"
            elif self.expr._agg_func_name is not None:
                # Aggregation mode - show aggregation function
                expr_str = f"<ColumnExpr: {self.expr._agg_func_name}()>"
            elif self.expr._executor is not None:
                # Executor mode
                expr_str = "<ColumnExpr: executor>"
            else:
                expr_str = "<ColumnExpr>"
        elif isinstance(self.expr, Expression):
            expr_str = self.expr.to_sql(quote_char='"')
        elif isinstance(self.expr, LazySeries):
            # Avoid triggering execution - use descriptive string
            expr_str = f"<LazySeries: {self.expr._method_name}>"
        elif isinstance(self.expr, pd.Series):
            # Avoid printing entire Series content
            expr_str = f"<Series: {self.expr.dtype}, len={len(self.expr)}>"
        elif isinstance(self.expr, pd.DataFrame):
            # Avoid printing entire DataFrame content
            expr_str = f"<DataFrame: {self.expr.shape}>"
        else:
            # For other types, limit repr length to avoid verbose output
            raw_repr = repr(self.expr)
            if len(raw_repr) > 80:
                expr_str = raw_repr[:77] + "..."
            else:
                expr_str = raw_repr
        return f"Assign column '{self.column}' = {expr_str}"

    def execution_engine(self) -> str:
        """
        Determine which engine will execute this assignment.

        Returns 'chDB' if the expression uses SQL functions (CastFunction, etc.),
        'Pandas' for simple arithmetic/field access.
        """
        return self._determine_engine(self.expr)

    def _determine_engine(self, expr) -> str:
        """Recursively determine which engine an expression will use."""
        from .functions import Function, CastFunction
        from .function_executor import function_config
        from .column_expr import ColumnExpr
        from .expressions import DateTimePropertyExpr, DateTimeMethodExpr
        from .case_when import CaseWhenExpr

        # Handle ColumnExpr - unwrap
        if isinstance(expr, ColumnExpr):
            return self._determine_engine(expr._expr)

        if isinstance(expr, CastFunction):
            # CastFunction always uses chDB
            return 'chDB'

        elif isinstance(expr, CaseWhenExpr):
            # CaseWhenExpr - delegate to its own execution_engine method
            return expr.execution_engine()

        elif isinstance(expr, DateTimePropertyExpr):
            # DateTime property - check function_config for engine selection
            if function_config.should_use_pandas(expr.property_name):
                return 'Pandas'
            else:
                return 'chDB'

        elif isinstance(expr, DateTimeMethodExpr):
            # DateTime method - check function_config for engine selection
            if function_config.should_use_pandas(expr.method_name):
                return 'Pandas'
            else:
                return 'chDB'

        elif isinstance(expr, Function):
            # Check function config
            # Use pandas_name if available (for functions where SQL name differs from user-facing name)
            func_name = (getattr(expr, 'pandas_name', None) or expr.name).lower()
            if function_config.should_use_pandas(func_name) and function_config.has_pandas_implementation(func_name):
                return 'Pandas'
            else:
                return 'chDB'

        elif isinstance(expr, ArithmeticExpression):
            # Check both sides
            left_engine = self._determine_engine(expr.left)
            right_engine = self._determine_engine(expr.right)
            if left_engine == 'chDB' or right_engine == 'chDB':
                return 'chDB'  # If any part uses chDB, the whole thing does
            return 'Pandas'

        elif isinstance(expr, (Field, Literal)):
            return 'Pandas'

        else:
            # Scalar value, Series, etc.
            return 'Pandas'

    def can_push_to_sql(self, existing_columns: list = None, computed_columns: set = None) -> bool:
        """
        Check if this column assignment can be pushed to SQL as a SELECT expression.

        The key insight is that SQL pushdown for column assignments requires
        proper subquery wrapping when:
        1. The expression references another computed column (requires subquery)
        2. The column name already exists (use EXCEPT to override)
        3. The expression has pandas-specific kwargs (e.g., na=False for str.contains)

        This method only checks if the expression ITSELF is SQL-compatible.
        The SQLBuilder will handle subquery wrapping when needed.

        Args:
            existing_columns: Original columns from the data source (for EXCEPT detection)
            computed_columns: Columns computed in the current SQL layer (for subquery detection)

        Returns:
            True if the expression can be converted to SQL
        """
        from .column_expr import ColumnExpr
        from .functions import Function

        # Check for pandas-specific kwargs that can't be properly expressed in SQL
        # (e.g., str.contains(na=False) - SQL NULL semantics differ from pandas)
        if self._has_pandas_only_kwargs(self.expr):
            return False

        # Check if expr is a ColumnExpr
        if isinstance(self.expr, ColumnExpr):
            # First check if expression contains pandas-only operations
            # (e.g., ** operator which causes dtype mismatch in SQL)
            if self._is_pandas_only_function(self.expr):
                return False
            # Use canonical method to check SQL compatibility
            return self.expr.is_sql_compatible()

        # Check if expr is directly an Expression
        if isinstance(self.expr, Expression):
            # Check if it's a Function that's pandas-only
            if self._is_pandas_only_function(self.expr):
                return False
            # CaseWhenExpr may have type mismatch issues in ClickHouse multiIf
            # When THEN and ELSE return different types (Int vs Float), SQL fails
            # Be conservative and don't push CaseWhenExpr to SQL
            from .case_when import CaseWhenExpr

            if isinstance(self.expr, CaseWhenExpr):
                return False
            return True

        # Other types (scalar values, pandas Series, etc.) can't be pushed to SQL
        return False

    def _is_pandas_only_function(self, expr) -> bool:
        """
        Check if expression contains pandas-only functions that cannot be pushed to SQL.

        Uses the global function_config to determine pandas-only functions,
        ensuring consistency with the rest of the system.
        """
        from .functions import Function
        from .function_executor import function_config
        from .expressions import ArithmeticExpression
        from .column_expr import ColumnExpr

        # Check ColumnExpr FIRST (before checking left/right attributes)
        if isinstance(expr, ColumnExpr):
            if expr._expr is not None:
                return self._is_pandas_only_function(expr._expr)
            return False

        if isinstance(expr, Function):
            func_name = expr.name.lower()
            # Use global config to check if function is pandas-only
            # All pandas-only functions are centralized in function_executor.PANDAS_ONLY_FUNCTIONS
            if function_config.is_pandas_only(func_name):
                return True
            # Recursively check function arguments
            for arg in expr.args:
                if self._is_pandas_only_function(arg):
                    return True

        elif isinstance(expr, ArithmeticExpression):
            # Certain arithmetic operators cause dtype differences in SQL
            # ** (power), // (floor div), % (modulo) return different types in ClickHouse
            if expr.operator in {'**', '//', '%'}:
                return True
            # Recursively check operands
            if self._is_pandas_only_function(expr.left):
                return True
            if self._is_pandas_only_function(expr.right):
                return True

        elif hasattr(expr, 'left') and hasattr(expr, 'right'):
            # Other binary expressions (BinaryCondition)
            left = expr.left
            right = expr.right
            if not callable(left) and self._is_pandas_only_function(left):
                return True
            if not callable(right) and self._is_pandas_only_function(right):
                return True

        return False

    def _has_pandas_only_kwargs(self, expr) -> bool:
        """
        Recursively check if expression tree contains pandas-only kwargs.

        These kwargs (like na=False for str.contains) have semantics that
        can't be properly expressed in SQL, so we must fall back to pandas.
        """
        from .functions import Function
        from .column_expr import ColumnExpr

        # Check ColumnExpr FIRST (before checking left/right attributes)
        # because ColumnExpr has left/right as methods, not expressions
        if isinstance(expr, ColumnExpr):
            if expr._expr is not None:
                return self._has_pandas_only_kwargs(expr._expr)
            return False

        if isinstance(expr, Function):
            # Check pandas_kwargs for special parameters
            if expr.pandas_kwargs:
                # 'na' parameter especially affects NULL handling
                # SQL NULLs behave differently than pandas NA handling
                if 'na' in expr.pandas_kwargs and expr.pandas_kwargs['na'] is not None:
                    return True
            # Recursively check function arguments
            for arg in expr.args:
                if self._has_pandas_only_kwargs(arg):
                    return True

        elif hasattr(expr, 'left') and hasattr(expr, 'right'):
            # Binary expressions (ArithmeticExpression, BinaryCondition)
            # Make sure left/right are not methods
            left = expr.left
            right = expr.right
            if not callable(left) and self._has_pandas_only_kwargs(left):
                return True
            if not callable(right) and self._has_pandas_only_kwargs(right):
                return True

        return False

    def get_sql_expression(self) -> Expression:
        """
        Get the SQL expression for this column assignment.

        Returns an Expression with the column name as alias, suitable for
        use in SELECT clause: SELECT *, (expr) AS "column"

        For comparison conditions (BinaryCondition, CompoundCondition), wraps with
        NullSafeCondition to match pandas NULL semantics:
        - NaN > NaN returns False (not NULL)
        - 'str' != NULL returns True (not NULL)

        Raises:
            ValueError: If expression is not SQL-convertible
        """
        from copy import copy
        from .column_expr import ColumnExpr
        from .conditions import BinaryCondition, CompoundCondition, NullSafeCondition

        if isinstance(self.expr, ColumnExpr):
            if self.expr._expr is None:
                raise ValueError(f"ColumnExpr has no SQL expression: {self.expr}")
            # Copy the expression to avoid modifying the original
            expr_copy = copy(self.expr._expr)
            # Wrap comparison conditions for pandas NULL semantics
            if isinstance(expr_copy, (BinaryCondition, CompoundCondition)):
                expr_copy = NullSafeCondition(expr_copy)
            expr_copy.alias = self.column
            return expr_copy

        if isinstance(self.expr, Expression):
            # Copy the expression to avoid modifying the original
            expr_copy = copy(self.expr)
            # Wrap comparison conditions for pandas NULL semantics
            if isinstance(expr_copy, (BinaryCondition, CompoundCondition)):
                expr_copy = NullSafeCondition(expr_copy)
            expr_copy.alias = self.column
            return expr_copy

        raise ValueError(f"Expression is not SQL-convertible: {type(self.expr)}")


class LazyColumnSelection(LazyOp):
    """
    Lazy column selection: df[["col1", "col2"]]

    Example:
        nat[["n_name", "n_nationkey"]]
    """

    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Return only the selected columns (preserving order)."""
        return [c for c in self.columns if c in columns or c in self.columns]

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("ColumnSelection", f"columns={self.columns}")
        result = df[self.columns]
        self._logger.debug("      -> DataFrame shape after selection: %s", result.shape)
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

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Remove the dropped columns."""
        return [c for c in columns if c not in self.columns]

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("DropColumns", f"columns={self.columns}")
        result = df.drop(columns=self.columns)
        self._logger.debug("      -> DataFrame shape after drop: %s", result.shape)
        return result

    def describe(self) -> str:
        return f"Drop columns: {', '.join(self.columns)}"


class LazyRenameColumns(LazyOp):
    """Rename columns: df.rename(columns={...})

    Handles both regular columns and MultiIndex columns.
    For MultiIndex columns, pd.rename() doesn't work with tuple->string mappings,
    so we use direct column assignment instead.
    """

    def __init__(self, mapping: Dict[str, str]):
        super().__init__()
        self.mapping = mapping

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Apply the rename mapping to column names."""
        return [self.mapping.get(c, c) for c in columns]

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("RenameColumns", f"mapping={self.mapping}")

        # Check if we have MultiIndex columns that need renaming to flat strings
        # This happens when user does: ds.columns = ['flat_name1', 'flat_name2', ...]
        # after a groupby().agg() which creates MultiIndex columns
        has_multiindex_to_flat = any(
            isinstance(old, tuple) and isinstance(new, str) for old, new in self.mapping.items()
        )

        if has_multiindex_to_flat:
            # For MultiIndex -> flat string renaming, we need direct assignment
            # because pd.rename() doesn't handle tuple->string mapping
            new_columns = []
            for col in df.columns:
                if col in self.mapping:
                    new_columns.append(self.mapping[col])
                else:
                    new_columns.append(col)
            result = df.copy()
            result.columns = new_columns
        else:
            # Standard rename for regular columns
            result = df.rename(columns=self.mapping)

        self._logger.debug("      -> New columns: %s", list(result.columns))
        return result

    def describe(self) -> str:
        renames = ', '.join(f"{old}→{new}" for old, new in self.mapping.items())
        return f"Rename columns: {renames}"


class LazyAddPrefix(LazyOp):
    """Add prefix to all columns."""

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Add prefix to all column names."""
        return [f"{self.prefix}{c}" for c in columns]

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AddPrefix", f"prefix='{self.prefix}'")
        result = df.add_prefix(self.prefix)
        self._logger.debug("      -> New columns: %s", list(result.columns))
        return result

    def describe(self) -> str:
        return f"Add prefix '{self.prefix}' to all columns"


class LazyAddSuffix(LazyOp):
    """Add suffix to all columns."""

    def __init__(self, suffix: str):
        super().__init__()
        self.suffix = suffix

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Add suffix to all column names."""
        return [f"{c}{self.suffix}" for c in columns]

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AddSuffix", f"suffix='{self.suffix}'")
        result = df.add_suffix(self.suffix)
        self._logger.debug("      -> New columns: %s", list(result.columns))
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
            "[Pandas]   -> Dropped %d rows (from %d to %d)", rows_before - len(result), rows_before, len(result)
        )
        return result

    def describe(self) -> str:
        desc = f"Drop NA (how='{self.how}')"
        if self.subset:
            desc += f" on columns: {', '.join(self.subset)}"
        return desc


class LazyDistinct(LazyOp):
    """Drop duplicate rows: df.drop_duplicates()"""

    def __init__(self, subset=None, keep='first'):
        super().__init__()
        self.subset = subset
        self.keep = keep

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("Distinct", f"subset={self.subset}, keep='{self.keep}'")
        rows_before = len(df)
        result = df.drop_duplicates(subset=self.subset, keep=self.keep)
        self._logger.debug(
            "    [Pandas] -> Dropped %d duplicate rows (from %d to %d)",
            rows_before - len(result),
            rows_before,
            len(result),
        )
        return result

    def describe(self) -> str:
        if self.subset:
            return f"Distinct on columns: {', '.join(self.subset)}"
        return "Distinct (all columns)"

    def can_push_to_sql(self) -> bool:
        # DISTINCT can be pushed to SQL
        return True


class LazyAsType(LazyOp):
    """Cast column types."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AsType", f"dtype={self.dtype}")
        result = df.astype(self.dtype)
        self._logger.debug("      -> New dtypes: %s", dict(result.dtypes))
        return result

    def describe(self) -> str:
        return f"Cast types: {self.dtype}"


class LazyRelationalOp(LazyOp):
    """
    A relational operation that can be executed via SQL or pandas.

    This stores both the SQL description and the original condition/parameters,
    so it can be executed either as SQL (compiled into query) or on a DataFrame
    (using pandas operations).

    When operations are purely SQL-based, they are compiled into a SQL query.
    When mixed with pandas operations, later relational ops execute on DataFrames.

    Example:
        users = ds.from_file('users.csv')
        users = users.select('name', 'age')  # Records LazyRelationalOp
        users['doubled'] = users['age'] * 2  # Records LazyColumnAssignment
        users = users.filter(users['age'] > 25)  # Records another LazyRelationalOp
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
        kind='quicksort',
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
            kind: Sort algorithm for ORDER BY ('quicksort', 'stable', 'mergesort', etc.)
        """
        super().__init__()
        self.op_type = op_type
        self.description = description
        self.condition = condition
        self.fields = fields
        self.ascending = ascending
        self.limit_value = limit_value
        self.offset_value = offset_value
        self.kind = kind

    def transform_columns(self, columns: List[str]) -> List[str]:
        """Transform columns based on the operation type."""
        if self.op_type == 'SELECT' and self.fields:
            # SELECT operation: return only selected columns
            selected = []
            for f in self.fields:
                if isinstance(f, str):
                    if f != '*':
                        selected.append(f)
                elif hasattr(f, 'alias') and f.alias:
                    # Expression with alias takes priority (e.g., as_())
                    selected.append(f.alias)
                elif hasattr(f, 'name'):
                    # Field object - strip quotes from name
                    name = f.name.strip('"') if isinstance(f.name, str) else str(f.name)
                    selected.append(name)
            # If we got specific columns, return them; otherwise return all
            return selected if selected else columns
        # Other operations (WHERE, ORDER BY, LIMIT, OFFSET) don't change columns
        return columns

    # Map SQL op_type to pandas terminology for logging
    _PANDAS_OP_NAMES = {
        'WHERE': 'filter',
        'SELECT': 'select',
        'ORDER BY': 'sort_values',
        'LIMIT': 'head',
        'OFFSET': 'iloc',
    }

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute this filter/transform operation on a DataFrame using pandas."""
        pandas_op = self._PANDAS_OP_NAMES.get(self.op_type, self.op_type)
        self._log_execute(f"Pandas {pandas_op}", self.description)
        rows_before = len(df)

        if self.op_type == 'SELECT' and self.fields:
            # Select specific columns
            cols = [f if isinstance(f, str) else f.name for f in self.fields]
            existing_cols = [c for c in cols if c in df.columns]
            # Log column selection
            self._logger.debug("      -> df[%s]", existing_cols)
            if existing_cols:
                result = df[existing_cols]
                self._logger.debug("      -> Selected columns: %s", existing_cols)
                return result
            return df
        elif self.op_type == 'WHERE' and self.condition is not None:
            # Log the condition (executed via pandas boolean mask)
            try:
                condition_sql = self.condition.to_sql(quote_char='"')
                self._logger.debug("      -> Condition: %s", condition_sql)
            except Exception:
                pass
            # Apply filter condition on DataFrame using pandas
            result = self._apply_condition(df, self.condition, context)
            self._logger.debug("      -> df[mask]: %d -> %d rows", rows_before, len(result))
            return result
        elif self.op_type == 'PANDAS_FILTER' and self.condition is not None:
            # Method-mode ColumnExpr condition (e.g., cumsum() > 6, rank() > 3)
            # Execute the condition to get boolean Series
            from .column_expr import ColumnExpr

            if isinstance(self.condition, ColumnExpr):
                # Set datastore for proper execution context
                if self.condition._datastore is None:
                    self.condition._datastore = context
                # Execute to get boolean mask
                bool_series = self.condition._execute()
                # Filter DataFrame using the boolean mask
                # Align index if necessary
                if len(bool_series) == len(df):
                    result = df[bool_series.values]
                else:
                    # Try to align by index
                    result = df.loc[bool_series.index[bool_series]]
                self._logger.debug("      -> Pandas filter (method-mode): %d -> %d rows", rows_before, len(result))
                return result
            return df
        elif self.op_type == 'ORDER BY' and self.fields:
            # Sort DataFrame
            cols = [f if isinstance(f, str) else f.name for f in self.fields]
            existing_cols = [c for c in cols if c in df.columns]
            # Log sort info
            try:
                direction = 'ascending' if self.ascending else 'descending'
                self._logger.debug(
                    "      -> df.sort_values(by=%s, ascending=%s, kind=%s)", existing_cols, self.ascending, self.kind
                )
            except Exception:
                pass
            if existing_cols:
                result = df.sort_values(by=existing_cols, ascending=self.ascending, kind=self.kind)
                self._logger.debug("      -> Sorted by: %s (%s, kind=%s)", existing_cols, direction, self.kind)
                return result
            return df
        elif self.op_type == 'LIMIT' and self.limit_value is not None:
            self._logger.debug("      -> df.head(%d)", self.limit_value)
            result = df.head(self.limit_value)
            self._logger.debug("      -> Limited to %d rows", self.limit_value)
            return result
        elif self.op_type == 'OFFSET' and self.offset_value is not None:
            self._logger.debug("      -> df.iloc[%d:]", self.offset_value)
            result = df.iloc[self.offset_value :]
            self._logger.debug("      -> Offset by %d rows", self.offset_value)
            return result
        return df

    def _apply_condition(self, df: pd.DataFrame, condition, context: 'DataStore') -> pd.DataFrame:
        """Apply a condition to filter a DataFrame."""
        from .conditions import Condition, CompoundCondition, NotCondition

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
        from .conditions import InCondition, BetweenCondition, UnaryCondition

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
            left = self._evaluate_operand(df, condition.expression, context)
            values = condition.values
            if condition.negate:
                return ~left.isin(values)
            return left.isin(values)

        if isinstance(condition, BetweenCondition):
            field = self._evaluate_operand(df, condition.expression, context)
            low = self._evaluate_operand(df, condition.lower, context)
            high = self._evaluate_operand(df, condition.upper, context)
            # BetweenCondition doesn't have negate attribute, always inclusive
            return (field >= low) & (field <= high)

        if isinstance(condition, UnaryCondition):
            field = self._evaluate_operand(df, condition.expression, context)
            if condition.operator == 'IS NULL':
                return field.isna()
            elif condition.operator == 'IS NOT NULL':
                return field.notna()

        return pd.Series([True] * len(df), index=df.index)

    def _evaluate_operand(self, df: pd.DataFrame, operand, context: 'DataStore'):
        """
        Evaluate an operand using unified ExpressionEvaluator.

        This delegates to ExpressionEvaluator which respects function_config.
        """
        from .expression_evaluator import ExpressionEvaluator

        evaluator = ExpressionEvaluator(df, context)
        return evaluator.evaluate(operand)

    def describe(self) -> str:
        return f"{self.op_type}: {self.description}"

    def describe_pandas(self) -> str:
        """Describe using pandas terminology instead of SQL."""
        pandas_op = self._PANDAS_OP_NAMES.get(self.op_type, self.op_type)
        return f"{pandas_op}: {self.description}"

    def can_push_to_sql(self) -> bool:
        return True


class LazyGroupByAgg(LazyOp):
    """
    Lazy groupby aggregation operation.

    This allows groupby operations to remain lazy until execution is triggered.
    Returns a DataFrame with group keys as index (or columns if as_index=False).

    Example:
        ds.groupby('category').mean()  # Returns DataStore with LazyGroupByAgg
        ds.groupby('category').agg({'value': 'sum'})
        ds.groupby('category').agg(total=('value', 'sum'))  # Named aggregation
    """

    def __init__(
        self,
        groupby_cols: List[str],
        agg_func: str = None,
        agg_dict: dict = None,
        named_agg: dict = None,
        sort: bool = True,
        as_index: bool = True,
        dropna: bool = True,
        single_column_agg: bool = False,
        selected_columns: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            groupby_cols: Column names to group by
            agg_func: Aggregation function name ('mean', 'sum', etc.) for all columns
            agg_dict: Dict mapping columns to aggregation functions (for pandas-style agg)
            named_agg: Dict of named aggregations {alias: (col, func)} (pandas named agg syntax)
            sort: Sort group keys (default: True, matching pandas behavior).
                  When True, the result is sorted by group keys in ascending order.
            as_index: If True (default), group keys become the index.
                      If False, group keys are returned as columns.
            dropna: If True (default), exclude NA/null values in keys.
                    If False, NA values are also grouped. Matches pandas default.
            single_column_agg: If True, this agg was called on a single column via
                               ColumnExpr.agg(['funcs']), which in pandas returns flat column names
            selected_columns: List of column names to aggregate (None = all columns).
            **kwargs: Additional arguments passed to aggregation function
        """
        super().__init__()
        self.groupby_cols = groupby_cols
        self.agg_func = agg_func
        self.agg_dict = agg_dict
        self.named_agg = named_agg
        self.sort = sort
        self.as_index = as_index
        self.dropna = dropna
        self.single_column_agg = single_column_agg
        self.selected_columns = selected_columns
        self.kwargs = kwargs

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute groupby aggregation on DataFrame."""
        self._log_execute(
            "GroupByAgg",
            f"groupby={self.groupby_cols}, func={self.agg_func or self.agg_dict or self.named_agg}, sort={self.sort}, as_index={self.as_index}, dropna={self.dropna}, selected_columns={self.selected_columns}",
        )

        # If selected_columns is specified, filter the DataFrame to only include
        # groupby columns + selected columns before aggregation
        if self.selected_columns:
            cols_to_keep = self.groupby_cols + self.selected_columns
            df = df[cols_to_keep]

        # Pass sort, as_index, and dropna parameters to pandas groupby
        grouped = df.groupby(self.groupby_cols, sort=self.sort, as_index=self.as_index, dropna=self.dropna)

        if self.named_agg is not None:
            # Pandas named aggregation: agg(alias=('col', 'func'))
            # Pass the named_agg dict as **kwargs to grouped.agg()
            result = grouped.agg(**self.named_agg)
            # For named_agg with as_index=True, reset index to make groupby columns regular columns
            # This matches pandas behavior where named aggregation returns flat columns
            if self.as_index:
                result = result.reset_index()
        elif self.agg_dict is not None:
            # Pandas-style: agg({'col': 'func'})
            result = grouped.agg(self.agg_dict, **self.kwargs)
        elif self.agg_func == 'size':
            # size() returns Series, convert to DataFrame
            result = grouped.size().to_frame('size')
            if not self.as_index:
                result = result.reset_index()
        else:
            # Single function for all columns
            agg_method = getattr(grouped, self.agg_func)
            result = agg_method(**self.kwargs)

        self._logger.debug("      -> GroupBy result shape: %s", result.shape)
        return result

    def describe(self) -> str:
        if self.named_agg:
            func_str = str(self.named_agg)
        elif self.agg_dict:
            func_str = str(self.agg_dict)
        else:
            func_str = self.agg_func
        return f"GroupBy({self.groupby_cols}).{func_str}()"

    def can_push_to_sql(self) -> bool:
        """
        Check if this GroupByAgg can be pushed to SQL.

        Most aggregation modes can be pushed to SQL, except:
        - agg_dict is a list (e.g., ['sum', 'mean', 'count']) which needs pandas
        """
        # List format agg (e.g., agg(['sum', 'mean'])) needs pandas
        if self.agg_dict is not None and isinstance(self.agg_dict, list):
            return False
        return True

    def execution_engine(self) -> str:
        """Return which engine this operation should use."""
        return 'SQL'


class LazyDataFrameSource(LazyOp):
    """
    A lazy operation that provides a DataFrame as the data source.

    This is used when we need to wrap an existing DataFrame into the lazy pipeline,
    for example after executing SQL on a executed DataFrame.
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


class LazySQLQuery(LazyOp):
    """
    Execute a SQL query on the current DataFrame using chDB's Python() table function.

    This enables true SQL-Pandas-SQL interleaving within the lazy pipeline.

    Supports three modes:
    1. Raw SQL (is_raw_query=True):
       - DataStore.run_sql("SELECT * FROM file('data.csv', 'CSVWithNames')")
       - Executes directly against chDB without a DataFrame

    2. Short form (auto-adds SELECT * FROM __df__):
       - ds.sql("doubled > 100")  -> SELECT * FROM __df__ WHERE doubled > 100
       - ds.sql("doubled > 100 ORDER BY id")  -> SELECT * FROM __df__ WHERE doubled > 100 ORDER BY id
       - ds.sql("ORDER BY id LIMIT 5")  -> SELECT * FROM __df__ ORDER BY id LIMIT 5

    3. Full SQL form (when query contains SELECT/FROM/GROUP BY):
       - ds.sql("SELECT id, SUM(value) FROM __df__ GROUP BY id")

    Example:
        ds = DataStore.from_file('users.csv')
        ds = ds.filter(ds.age > 20)
        ds['doubled'] = ds['age'] * 2
        ds = ds.sql("doubled > 50 ORDER BY age DESC LIMIT 10")  # Short form!
        ds = ds.add_prefix('result_')
    """

    def __init__(self, query: str, df_alias: str = '__df__', is_raw_query: bool = False):
        """
        Args:
            query: SQL query or condition. Can be:
                   - Full SQL: "SELECT * FROM __df__ WHERE x > 10"
                   - Short form: "x > 10" (auto-adds SELECT * FROM __df__ WHERE)
                   - Clauses only: "ORDER BY id LIMIT 5" (auto-adds SELECT * FROM __df__)
                   - Raw SQL: "SELECT * FROM file('data.csv', 'CSVWithNames')" (when is_raw_query=True)
            df_alias: Alias for the DataFrame in the query (default: '__df__')
            is_raw_query: If True, execute the query directly without a DataFrame
        """
        super().__init__()
        self.original_query = query.strip()
        self.df_alias = df_alias
        self.is_raw_query = is_raw_query

        # Process the query to determine if it needs boilerplate (only for non-raw queries)
        if is_raw_query:
            self.query = self.original_query
        else:
            self.query = self._process_query(self.original_query)

    def _process_query(self, query: str) -> str:
        """
        Process the query to add SELECT * FROM __df__ if needed.

        Rules:
        1. If query contains SELECT or FROM, use as-is (full SQL)
        2. If query starts with WHERE, ORDER BY, LIMIT, OFFSET, add SELECT * FROM __df__
        3. Otherwise, treat as WHERE condition and add SELECT * FROM __df__ WHERE
        """
        query_upper = query.upper().strip()

        # Check if it's already a full SQL statement
        if query_upper.startswith('SELECT') or 'FROM' in query_upper:
            return query

        # Check if it starts with a clause (WHERE, ORDER BY, LIMIT, OFFSET, GROUP BY, HAVING)
        clause_starters = ('WHERE', 'ORDER BY', 'LIMIT', 'OFFSET', 'GROUP BY', 'HAVING')
        for clause in clause_starters:
            if query_upper.startswith(clause):
                return f"SELECT * FROM __df__ {query}"

        # Otherwise, treat as a WHERE condition
        return f"SELECT * FROM __df__ WHERE {query}"

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute the SQL query using chDB via centralized Executor."""
        from .executor import get_executor

        if self.is_raw_query:
            # Raw query: execute directly without a DataFrame
            self._log_execute("Raw SQL Query", "", prefix="chDB")
            self._logger.debug("    [chDB] Raw query: %s", self.query)

            executor = get_executor()
            # Execute raw SQL directly
            result = executor.execute(self.query)
            # Convert QueryResult to DataFrame
            if hasattr(result, 'to_df'):
                return result.to_df()
            if isinstance(result, pd.DataFrame):
                return result
            return pd.DataFrame(result) if result is not None else pd.DataFrame()
        else:
            # Query on existing DataFrame
            self._log_execute("SQL Query", f"rows={len(df)}", prefix="chDB")

            self._logger.debug("    [chDB] Original input: %s", self.original_query)
            self._logger.debug("    [chDB] Expanded query: %s", self.query)

            # Use centralized executor
            executor = get_executor()
            result = executor.query_dataframe(self.query, df, '__df__')
            return result

    def describe(self) -> str:
        # Show original query for brevity, but indicate if it was expanded
        if self.original_query != self.query:
            display = self.original_query if len(self.original_query) <= 50 else self.original_query[:47] + '...'
            return f"SQL: {display} (expanded)"
        else:
            display = self.query if len(self.query) <= 60 else self.query[:57] + '...'
            return f"SQL Query: {display}"

    def can_push_to_sql(self) -> bool:
        # This is already a SQL operation, but executes via Python() table function
        return False

    def execution_engine(self) -> str:
        """LazySQLQuery always uses chDB."""
        return 'chDB'


class LazyFilter(LazyOp):
    """
    Lazy groupby filter operation.

    Filters groups based on a callable that takes a group DataFrame and returns bool.
    This operation requires groupby_cols and cannot be pushed to SQL.

    Note: This is different from pandas DataFrame.filter() which is for column selection.
    For row filtering, use boolean indexing via DataStore.filter(condition).

    Example:
        ds.groupby('category').filter(lambda x: x['value'].mean() > 35)
    """

    def __init__(self, func, groupby_cols: List[str]):
        """
        Args:
            func: Callable that takes a group DataFrame and returns True/False
            groupby_cols: Column names to group by (required)
        """
        super().__init__()
        self.func = func
        self.groupby_cols = groupby_cols

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute groupby filter on DataFrame."""
        self._log_execute("Filter", f"groupby={self.groupby_cols}")
        rows_before = len(df)
        # Note: groupby_dropna should be passed from LazyGroupBy but is not available here
        # For filter operations on groups, we use the default dropna=True behavior
        result = df.groupby(self.groupby_cols, dropna=True).filter(self.func)
        self._logger.debug("      -> Filtered: %d -> %d rows", rows_before, len(result))
        return result

    def describe(self) -> str:
        func_name = getattr(self.func, '__name__', '<lambda>')
        return f"GroupBy({self.groupby_cols}).filter({func_name})"

    def can_push_to_sql(self) -> bool:
        # Python callable cannot be pushed to SQL
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyTransform(LazyOp):
    """
    Lazy transform operation with optional groupby support.

    When groupby_cols is provided, applies transform within each group.
    When groupby_cols is None, applies transform to entire DataFrame.

    Example:
        # With groupby
        ds.groupby('category').transform(lambda x: x / x.sum())

        # Without groupby
        ds.transform(lambda x: x * 2)  # Future use
    """

    def __init__(self, func, *args, groupby_cols: List[str] = None, columns: List[str] = None, **kwargs):
        """
        Args:
            func: Function to apply (callable or string like 'mean', 'sum')
            groupby_cols: Optional column names to group by
            columns: Specific columns to transform (None = all applicable)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        """
        super().__init__()
        self.func = func
        self.groupby_cols = groupby_cols
        self.columns = columns
        self.args = args
        self.kwargs = kwargs

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute transform on DataFrame."""
        func_desc = self.func if isinstance(self.func, str) else getattr(self.func, '__name__', '<callable>')

        if self.groupby_cols:
            self._log_execute("Transform", f"groupby={self.groupby_cols}, func={func_desc}")
            # Note: groupby_dropna should be passed from LazyGroupBy but is not available here
            # For transform operations, we use the default dropna=True behavior
            grouped = df.groupby(self.groupby_cols, dropna=True)

            if self.columns:
                result = df.copy()
                for col in self.columns:
                    if col in df.columns:
                        result[col] = grouped[col].transform(self.func, *self.args, **self.kwargs)
            else:
                result = grouped.transform(self.func, *self.args, **self.kwargs)
        else:
            self._log_execute("Transform", f"func={func_desc}")
            if self.columns:
                result = df.copy()
                for col in self.columns:
                    if col in df.columns:
                        result[col] = df[col].transform(self.func, *self.args, **self.kwargs)
            else:
                result = df.transform(self.func, *self.args, **self.kwargs)

        self._logger.debug("      -> Transform result shape: %s", result.shape)
        return result

    def describe(self) -> str:
        func_name = self.func if isinstance(self.func, str) else getattr(self.func, '__name__', '<callable>')
        cols_desc = f"[{', '.join(self.columns)}]" if self.columns else "all"
        if self.groupby_cols:
            return f"GroupBy({self.groupby_cols}).transform({func_name}, columns={cols_desc})"
        return f"transform({func_name}, columns={cols_desc})"

    def can_push_to_sql(self) -> bool:
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyApply(LazyOp):
    """
    Lazy apply operation with optional groupby support.

    When groupby_cols is provided, applies function to each group.
    When groupby_cols is None, applies function to entire DataFrame.

    Example:
        # With groupby
        ds.groupby('category').apply(lambda x: x.nlargest(3, 'value'))

        # Without groupby
        ds.apply(lambda df: df.describe())  # Future use
    """

    def __init__(self, func, *args, groupby_cols: List[str] = None, **kwargs):
        """
        Args:
            func: Function to apply
            groupby_cols: Optional column names to group by
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        """
        super().__init__()
        self.func = func
        self.groupby_cols = groupby_cols
        self.args = args
        self.kwargs = kwargs

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute apply on DataFrame."""
        func_name = getattr(self.func, '__name__', '<callable>')

        if self.groupby_cols:
            self._log_execute("Apply", f"groupby={self.groupby_cols}, func={func_name}")
            # Note: groupby_dropna should be passed from LazyGroupBy but is not available here
            # For apply operations, we use the default dropna=True behavior
            result = df.groupby(self.groupby_cols, dropna=True).apply(
                self.func, *self.args, include_groups=False, **self.kwargs
            )
            # Reset index if result has MultiIndex from groupby
            if isinstance(result.index, pd.MultiIndex):
                result = result.reset_index(drop=True)
        else:
            self._log_execute("Apply", f"func={func_name}")
            result = df.apply(self.func, *self.args, **self.kwargs)
            if isinstance(result, pd.Series):
                result = result.to_frame()

        self._logger.debug("      -> Apply result shape: %s", result.shape)
        return result

    def describe(self) -> str:
        func_name = getattr(self.func, '__name__', '<callable>')
        if self.groupby_cols:
            return f"GroupBy({self.groupby_cols}).apply({func_name})"
        return f"apply({func_name})"

    def can_push_to_sql(self) -> bool:
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


# Aliases for backward compatibility
LazyGroupByFilter = LazyFilter
LazyGroupByTransform = LazyTransform
LazyGroupByApply = LazyApply


class LazyNth(LazyOp):
    """
    Lazy groupby nth operation.

    Returns the nth row from each group. Supports negative indexing.

    Example:
        ds.groupby('category').nth(0)   # First row from each group
        ds.groupby('category').nth(1)   # Second row from each group
        ds.groupby('category').nth(-1)  # Last row from each group
        ds.groupby('category').nth([0, 2])  # First and third rows from each group
    """

    def __init__(self, n: Union[int, List[int]], groupby_cols: List[str], dropna: str = None):
        """
        Args:
            n: Integer or list of integers indicating which row(s) to select from each group.
               Negative values select from the end of each group.
            groupby_cols: Column names to group by
            dropna: Optional, how to handle NA values. Can be 'any', 'all', or None.
        """
        super().__init__()
        self.n = n
        self.groupby_cols = groupby_cols
        self.dropna = dropna

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute groupby nth on DataFrame."""
        n_desc = self.n if isinstance(self.n, int) else list(self.n)
        self._log_execute("Nth", f"groupby={self.groupby_cols}, n={n_desc}")

        # Note: groupby_dropna should be passed from LazyGroupBy but is not available here
        # For nth operations, we use the default dropna=True behavior for groupby
        grouped = df.groupby(self.groupby_cols, sort=False, dropna=True)

        # pandas nth() behavior:
        # - Returns rows with their original index preserved
        # - Supports negative indexing
        # - Supports list of indices
        if self.dropna is not None:
            result = grouped.nth(self.n, dropna=self.dropna)
        else:
            result = grouped.nth(self.n)

        self._logger.debug("      -> Nth result shape: %s", result.shape)
        return result

    def describe(self) -> str:
        n_desc = self.n if isinstance(self.n, int) else list(self.n)
        return f"GroupBy({self.groupby_cols}).nth({n_desc})"

    def can_push_to_sql(self) -> bool:
        # nth requires row ordering within groups, complex to push to SQL
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyHead(LazyOp):
    """
    Lazy groupby head operation.

    Returns the first n rows from each group, preserving original index.

    Example:
        ds.groupby('category').head(2)   # First 2 rows from each group
        ds.groupby('category').head()    # First 5 rows from each group (default)
    """

    def __init__(self, n: int, groupby_cols: List[str]):
        """
        Args:
            n: Number of rows to return from each group (default: 5)
            groupby_cols: Column names to group by
        """
        super().__init__()
        self.n = n
        self.groupby_cols = groupby_cols

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute groupby head on DataFrame."""
        self._log_execute("Head", f"groupby={self.groupby_cols}, n={self.n}")

        # Note: groupby_dropna should be passed from LazyGroupBy but is not available here
        # For nth operations, we use the default dropna=True behavior for groupby
        grouped = df.groupby(self.groupby_cols, sort=False, dropna=True)

        # pandas head() returns first n rows from each group with original index preserved
        result = grouped.head(self.n)

        self._logger.debug("      -> Head result shape: %s", result.shape)
        return result

    def describe(self) -> str:
        return f"GroupBy({self.groupby_cols}).head({self.n})"

    def can_push_to_sql(self) -> bool:
        # head requires row ordering within groups, complex to push to SQL
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyTail(LazyOp):
    """
    Lazy groupby tail operation.

    Returns the last n rows from each group, preserving original index.

    Example:
        ds.groupby('category').tail(2)   # Last 2 rows from each group
        ds.groupby('category').tail()    # Last 5 rows from each group (default)
    """

    def __init__(self, n: int, groupby_cols: List[str]):
        """
        Args:
            n: Number of rows to return from each group (default: 5)
            groupby_cols: Column names to group by
        """
        super().__init__()
        self.n = n
        self.groupby_cols = groupby_cols

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute groupby tail on DataFrame."""
        self._log_execute("Tail", f"groupby={self.groupby_cols}, n={self.n}")

        # Note: groupby_dropna should be passed from LazyGroupBy but is not available here
        # For nth operations, we use the default dropna=True behavior for groupby
        grouped = df.groupby(self.groupby_cols, sort=False, dropna=True)

        # pandas tail() returns last n rows from each group with original index preserved
        result = grouped.tail(self.n)

        self._logger.debug("      -> Tail result shape: %s", result.shape)
        return result

    def describe(self) -> str:
        return f"GroupBy({self.groupby_cols}).tail({self.n})"

    def can_push_to_sql(self) -> bool:
        # tail requires row ordering within groups, complex to push to SQL
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyWhere(LazyOp):
    """
    Lazy where operation: keep values where condition is True, replace with other where False.

    This is the pandas-style where (element-wise conditional replacement),
    NOT SQL WHERE (row filtering).

    SQL pushdown is possible when:
    1. Condition is SQL-compatible (Condition object)
    2. other is a scalar value
    3. All column names are known

    SQL equivalent:
        SELECT CASE WHEN cond THEN col ELSE other END AS col, ... FROM table

    Example:
        ds.where(ds['value'] > 100, 0)  # Keep where value > 100, else 0
    """

    def __init__(self, condition, other, columns: List[str] = None):
        """
        Args:
            condition: Condition object or ColumnExpr containing condition
            other: Value to use where condition is False
            columns: Column names for SQL pushdown (optional, inferred at execution)
        """
        super().__init__()
        self.condition = condition
        self.other = other
        self.columns = columns
        self._is_mask = False  # Subclass sets this to True

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute where operation on DataFrame."""
        from .expression_evaluator import ExpressionEvaluator
        from .column_expr import ColumnExpr
        from .conditions import Condition

        op_name = "Mask" if self._is_mask else "Where"
        self._log_execute(op_name, f"other={self.other}")

        # Evaluate condition to boolean Series
        cond = self.condition
        if isinstance(cond, ColumnExpr):
            cond_expr = cond._expr
            evaluator = ExpressionEvaluator(df, context)
            bool_series = evaluator.evaluate(cond_expr)
        elif isinstance(cond, Condition):
            evaluator = ExpressionEvaluator(df, context)
            bool_series = evaluator.evaluate(cond)
        elif isinstance(cond, pd.Series):
            bool_series = cond
        else:
            # Try to use as-is (numpy array, etc.)
            bool_series = cond

        # Apply where or mask
        if self._is_mask:
            result = df.mask(bool_series, self.other)
        else:
            result = df.where(bool_series, self.other)

        self._logger.debug("      -> %s result shape: %s", op_name, result.shape)
        return result

    def describe(self) -> str:
        from .column_expr import ColumnExpr
        from .conditions import Condition

        op_name = "mask" if self._is_mask else "where"

        # Describe condition
        cond = self.condition
        if isinstance(cond, ColumnExpr):
            cond_str = str(cond._expr) if hasattr(cond, '_expr') else str(cond)
        elif isinstance(cond, Condition):
            try:
                cond_str = cond.to_sql(quote_char='"')
            except Exception:
                cond_str = str(cond)
        else:
            cond_str = str(type(cond).__name__)

        return f"{op_name}({cond_str}, other={self.other})"

    def can_push_to_sql(self, schema: Dict[str, str] = None) -> bool:
        """
        Check if this where can be pushed to SQL.

        SQL pushdown requires:
        1. function_config allows chDB for 'where'/'mask'
        2. Condition must be SQL-compatible
        3. other must be a scalar
        4. Type compatibility between other and columns (no NO_COMMON_TYPE errors)

        Args:
            schema: Optional dict mapping column names to types for type-aware checking
        """
        from .column_expr import ColumnExpr
        from .conditions import Condition
        from .function_executor import function_config

        # Check function_config - respect user's engine preference
        func_name = 'mask' if self._is_mask else 'where'
        if not function_config.should_use_chdb(func_name):
            return False

        # Check if condition is SQL-compatible
        cond = self.condition
        if isinstance(cond, ColumnExpr):
            cond = cond._expr if hasattr(cond, '_expr') else None

        if not isinstance(cond, Condition):
            return False

        # Check if other is a scalar (not DataFrame/Series)
        if isinstance(self.other, (pd.DataFrame, pd.Series)):
            return False

        # Type compatibility check when schema is available
        if schema and not self._is_type_compatible_with_schema(schema):
            return False

        # Scalar other - can push to SQL
        return True

    def _is_type_compatible_with_schema(self, schema: Dict[str, str]) -> bool:
        """
        Check if 'other' type is compatible with all columns in the schema.

        ClickHouse SQL has strict type requirements. These combinations cause issues:
        1. String 'other' + numeric column (Int64/Float64) -> NO_COMMON_TYPE error
        2. Float 'other' + Int column (without explicit cast) -> NO_COMMON_TYPE error
        3. Int 'other' + String column -> handled by Variant type (OK)
        4. Numeric 'other' + Bool column -> ALWAYS fall back to Pandas
           - SQL CASE WHEN converts 0 to false, 1 to true (type change)
           - Pandas preserves the actual int value with object dtype
           - Type correctness is more important than performance

        Args:
            schema: Dict mapping column names to types

        Returns:
            True if type combination is SQL-compatible, False if should fall back to Pandas
        """
        other = self.other

        # None/NaN are always compatible (become NULL)
        if other is None or (isinstance(other, float) and pd.isna(other)):
            return True

        # Detect column types
        has_string_col = False
        has_int_col = False
        has_float_col = False
        has_bool_col = False

        for col_type in schema.values():
            col_type_lower = col_type.lower()
            if any(t in col_type_lower for t in ('string', 'fixedstring', 'enum', 'uuid')):
                has_string_col = True
            elif any(t in col_type_lower for t in ('float', 'double', 'decimal')):
                has_float_col = True
            elif any(t in col_type_lower for t in ('int', 'uint')):
                has_int_col = True
            elif 'bool' in col_type_lower:
                has_bool_col = True

        # Case 1: String 'other' - incompatible with numeric columns
        if isinstance(other, str):
            if has_int_col or has_float_col:
                return False  # Would cause NO_COMMON_TYPE

        # Case 2: Float 'other' + Int column - ClickHouse doesn't auto-convert
        if isinstance(other, float) and has_int_col:
            return False  # Would cause NO_COMMON_TYPE

        # Case 3: Int 'other' + String column - handled by Variant type
        # The SQL query MUST include ORDER BY rowNumberInAllBlocks() to preserve row order
        # when using Variant type. This is handled in _build_sql_from_state.

        # Case 4: Numeric 'other' + Bool column -> ALWAYS fall back to Pandas
        # SQL CASE WHEN converts numeric values to bool (0->false, 1->true),
        # which changes both the value and the dtype.
        # Pandas keeps the actual int value and uses object dtype for mixed types.
        # Type correctness is more important than performance.
        if has_bool_col and isinstance(other, (int, float)) and not isinstance(other, bool):
            return False  # Always fall back to Pandas for bool columns with numeric other

        return True

    def get_sql_case_when(self, columns: List[str], quote_char: str = '"') -> Dict[str, str]:
        """
        Generate SQL CASE WHEN expressions for each column.

        Args:
            columns: List of column names to transform
            quote_char: Quote character for identifiers

        Returns:
            Dict mapping column names to CASE WHEN expressions
        """
        from .column_expr import ColumnExpr
        from .conditions import Condition

        # Get condition SQL
        cond = self.condition
        if isinstance(cond, ColumnExpr):
            cond = cond._expr if hasattr(cond, '_expr') else cond

        if isinstance(cond, Condition):
            cond_sql = cond.to_sql(quote_char=quote_char)
        else:
            raise ValueError("Cannot convert condition to SQL")

        # For mask, invert the condition
        if self._is_mask:
            cond_sql = f"NOT ({cond_sql})"

        # Format other value
        if isinstance(self.other, str):
            other_sql = f"'{self.other}'"
        elif self.other is None or (isinstance(self.other, float) and pd.isna(self.other)):
            other_sql = "NULL"
        else:
            other_sql = str(self.other)

        # Generate CASE WHEN for each column
        result = {}
        for col in columns:
            col_quoted = f"{quote_char}{col}{quote_char}"
            result[col] = f"CASE WHEN {cond_sql} THEN {col_quoted} ELSE {other_sql} END"

        return result

    def execution_engine(self) -> str:
        """Return execution engine based on pushdown capability."""
        return 'chDB' if self.can_push_to_sql() else 'Pandas'


class LazyMask(LazyWhere):
    """
    Lazy mask operation: replace values where condition is True with other.

    This is the opposite of where:
    - where: keep where True, replace where False
    - mask: replace where True, keep where False

    Example:
        ds.mask(ds['value'] > 100, -1)  # Replace where value > 100 with -1
    """

    def __init__(self, condition, other, columns: List[str] = None):
        super().__init__(condition, other, columns)
        self._is_mask = True


class LazyJoin(LazyOp):
    """
    Lazy join operation for DataFrame execution.

    Performs pandas merge on two DataFrames.

    Example:
        ds1.join(ds2, on='user_id', how='inner')
    """

    def __init__(self, right, on=None, how='inner', left_on=None, right_on=None):
        """
        Args:
            right: Right DataFrame or DataStore to join with (lazy - executed when needed)
            on: Column name(s) to join on
            how: Join type ('inner', 'left', 'right', 'outer')
            left_on: Column name in left DataFrame
            right_on: Column name in right DataFrame
        """
        super().__init__()
        self.right = right  # Can be DataFrame or DataStore
        self.on = on
        self.how = how
        self.left_on = left_on
        self.right_on = right_on

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute pandas merge."""
        self._log_execute("Join", f"how='{self.how}', on={self.on}")
        left_rows = len(df)

        # Lazily execute the right side if it's a DataStore
        if hasattr(self.right, '_execute'):
            right_df = self.right._execute()
        else:
            right_df = self.right

        right_rows = len(right_df)

        result = pd.merge(
            df,
            right_df,
            on=self.on,
            how=self.how,
            left_on=self.left_on,
            right_on=self.right_on,
        )

        self._logger.debug("    [Pandas] -> Joined: left(%d) x right(%d) = %d rows", left_rows, right_rows, len(result))
        return result

    def describe(self) -> str:
        on_desc = self.on if self.on else f"left_on={self.left_on}, right_on={self.right_on}"
        return f"Join (how='{self.how}', on={on_desc})"

    def can_push_to_sql(self) -> bool:
        # Join can be pushed to SQL when both sides have SQL sources
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyUnion(LazyOp):
    """
    Lazy union operation (vertical concatenation).

    Concatenates two DataFrames vertically.

    Example:
        ds1.union(ds2)
    """

    def __init__(self, other, all: bool = False):
        """
        Args:
            other: DataFrame or DataStore to union with (lazy - executed when needed)
            all: If True, keep all rows (UNION ALL). If False, remove duplicates (UNION).
        """
        super().__init__()
        self.other = other  # Can be DataFrame or DataStore
        self.all = all

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute pandas concat."""
        self._log_execute("Union", f"all={self.all}")
        left_rows = len(df)

        # Lazily execute the other side if it's a DataStore
        if hasattr(self.other, '_execute'):
            other_df = self.other._execute()
        else:
            other_df = self.other

        right_rows = len(other_df)

        result = pd.concat([df, other_df], ignore_index=True)

        if not self.all:
            # UNION (without ALL) removes duplicates
            result = result.drop_duplicates()

        self._logger.debug("    [Pandas] -> Union: %d + %d = %d rows", left_rows, right_rows, len(result))
        return result

    def describe(self) -> str:
        return f"Union (all={self.all})"

    def can_push_to_sql(self) -> bool:
        return False

    def execution_engine(self) -> str:
        return 'Pandas'


class LazyBooleanMask(LazyOp):
    """
    Lazy boolean mask filtering operation.

    Implements pandas-style boolean mask indexing: df[[True, False, True, ...]]
    This filters rows based on a list of boolean values.

    Examples:
        ds[[True, True, False]]  # Keep first two rows, drop third
        ds[[False, True, True, False]]  # Keep rows at indices 1 and 2
    """

    def __init__(self, mask: list):
        """
        Args:
            mask: List of boolean values. Length must match DataFrame row count.
        """
        super().__init__()
        self.mask = mask

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute boolean mask filtering using pandas loc."""
        self._log_execute("BooleanMask", f"mask length={len(self.mask)}")
        rows_before = len(df)

        # Use pandas boolean indexing (preserves original index like pandas)
        result = df[self.mask]

        self._logger.debug("      -> df[mask]: %d -> %d rows", rows_before, len(result))
        return result

    def describe(self) -> str:
        mask_len = len(self.mask)
        true_count = sum(self.mask)
        return f"Boolean mask filter ({true_count}/{mask_len} rows selected)"

    def can_push_to_sql(self) -> bool:
        """Boolean mask cannot be pushed to SQL."""
        return False


class LazySliceStep(LazyOp):
    """
    Lazy slice operation with step support.

    Implements pandas-style slice with step: df[start:stop:step]
    This is implemented via SQL using ROW_NUMBER() for efficiency.

    Examples:
        ds[::2]       # Every 2nd row (step=2, start=0)
        ds[1::2]      # Every 2nd row starting at index 1
        ds[::3]       # Every 3rd row
        ds[1:8:2]     # Rows 1,3,5,7 (from 1 to 8, step 2)
        ds[::-1]      # Reverse order (all rows, reversed)
        ds[8:2:-2]    # Rows 8,6,4 (reverse with step)
    """

    def __init__(self, start: int = None, stop: int = None, step: int = 1):
        """
        Args:
            start: Start index (0-based, inclusive). None means 0 for positive step, -1 for negative.
            stop: Stop index (0-based, exclusive). None means end for positive step, beginning for negative.
            step: Step size. Positive for forward, negative for reverse.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        """Execute slice with step using pandas iloc."""
        self._log_execute("SliceStep", f"start={self.start}, stop={self.stop}, step={self.step}")
        rows_before = len(df)

        # Use pandas iloc for slicing with step
        result = df.iloc[self.start : self.stop : self.step]

        self._logger.debug(
            "      -> df.iloc[%s:%s:%s]: %d -> %d rows", self.start, self.stop, self.step, rows_before, len(result)
        )
        return result

    def describe(self) -> str:
        start_str = "" if self.start is None else str(self.start)
        stop_str = "" if self.stop is None else str(self.stop)
        step_str = "" if self.step is None or self.step == 1 else str(self.step)

        if step_str:
            return f"Slice [{start_str}:{stop_str}:{step_str}]"
        elif start_str or stop_str:
            return f"Slice [{start_str}:{stop_str}]"
        else:
            return "Slice [:]"

    def can_push_to_sql(self) -> bool:
        """
        Slice with step can be pushed to SQL using ROW_NUMBER().

        For positive step: WHERE (row_num - start) % step = 0 AND row_num >= start AND row_num < stop
        For negative step: More complex, requires reversing and then applying step
        """
        # Only push simple positive step to SQL for now
        # Negative step (reverse) is more complex and pandas handles it well
        return self.step is not None and self.step > 0

    def execution_engine(self) -> str:
        if self.can_push_to_sql():
            return 'chDB'
        return 'Pandas'

    def to_sql_condition(self, row_num_col: str = '__row_num__') -> str:
        """
        Generate SQL WHERE condition for step slicing.

        Args:
            row_num_col: Name of the row number column (0-based)

        Returns:
            SQL WHERE condition string
        """
        conditions = []
        step = self.step if self.step else 1

        if step > 0:
            # Forward slicing
            start = self.start if self.start is not None else 0

            # Rows where (row_num - start) % step = 0
            if step > 1:
                conditions.append(f"({row_num_col} - {start}) % {step} = 0")

            # row_num >= start
            if start > 0:
                conditions.append(f"{row_num_col} >= {start}")

            # row_num < stop
            if self.stop is not None:
                conditions.append(f"{row_num_col} < {self.stop}")

        else:
            # Negative step - handled by pandas execution
            # This method shouldn't be called for negative step
            pass

        return " AND ".join(conditions) if conditions else "1=1"
