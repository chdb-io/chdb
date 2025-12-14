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

    def _log_execute(self, op_name: str, details: str = None, prefix: str = "Pandas"):
        """Log operation execution at DEBUG level."""
        if details:
            self._logger.debug("[%s] Executing %s: %s", prefix, op_name, details)
        else:
            self._logger.debug("[%s] Executing %s", prefix, op_name)


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
        self._logger.debug("[Pandas]   -> DataFrame shape after assignment: %s", df.shape)
        return df

    def _evaluate_expr(self, expr, df: pd.DataFrame, context: 'DataStore'):
        """
        Recursively evaluate an expression.

        Supports:
        - ColumnExpr: unwrap and evaluate the underlying expression
        - Field references: nat["col"]
        - Literals: 1, "hello"
        - Arithmetic: nat["a"] + 1
        - Functions: nat["col"].str.upper(), nat["value"].cast("Float64")
        - Series: direct pandas Series
        """
        from .functions import Function, CastFunction
        from .function_executor import function_config
        from .column_expr import ColumnExpr

        # Handle ColumnExpr - unwrap to get the underlying expression
        if isinstance(expr, ColumnExpr):
            return self._evaluate_expr(expr._expr, df, context)

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

        elif isinstance(expr, CastFunction):
            # Special handling for CAST - use chDB via Python() table function
            return self._evaluate_function_via_chdb(expr, df, context)

        elif isinstance(expr, Function):
            # Function call - check execution config
            func_name = expr.name.lower()

            # Priority order:
            # 1. Pandas-only functions or explicitly configured to use Pandas
            # 2. Has registered Pandas implementation
            # 3. Try dynamic Pandas method (for unregistered functions)
            # 4. Fallback to chDB

            if function_config.should_use_pandas(func_name):
                # Pandas-only or configured for Pandas
                return self._evaluate_function_via_pandas(expr, df, context)
            elif function_config.has_pandas_implementation(func_name):
                # Has registered implementation - prefer chDB by default unless configured
                return self._evaluate_function_via_chdb(expr, df, context)
            else:
                # Unknown function - try Pandas dynamic first, then chDB
                return self._evaluate_unknown_function(expr, df, context)

        elif isinstance(expr, pd.Series):
            # Direct pandas Series
            return expr

        else:
            # Scalar value
            return expr

    def _evaluate_function_via_pandas(self, expr, df: pd.DataFrame, context: 'DataStore'):
        """
        Evaluate a Function expression using Pandas implementation.

        Supports three modes:
        1. Registered implementation: use function_config._pandas_implementations
        2. Dynamic method: try to call method on Series (e.g., s.str.title(), s.abs())
        3. Fallback to chDB if nothing works

        Args:
            expr: Function expression
            df: DataFrame to operate on
            context: DataStore context

        Returns:
            Result of the Pandas operation (typically a Series)
        """
        from .function_executor import function_config

        func_name = expr.name.lower()
        pandas_impl = function_config.get_pandas_implementation(func_name)

        # Evaluate first argument (the column/expression)
        if not expr.args:
            return self._evaluate_function_via_chdb(expr, df, context)

        first_arg = self._evaluate_expr(expr.args[0], df, context)
        other_args = [self._evaluate_expr(arg, df, context) for arg in expr.args[1:]]

        # Mode 1: Try registered implementation first
        if pandas_impl is not None:
            try:
                return pandas_impl(first_arg, *other_args)
            except Exception as e:
                self._logger.debug("[Pandas] Registered impl for %s failed: %s", func_name, e)

        # Mode 2: Try dynamic Pandas method invocation
        result = self._try_dynamic_pandas_method(func_name, first_arg, other_args)
        if result is not None:
            return result

        # Mode 3: Fallback to chDB
        self._logger.debug("[Pandas] No Pandas method found for %s, falling back to chDB", func_name)
        return self._evaluate_function_via_chdb(expr, df, context)

    def _try_dynamic_pandas_method(self, func_name: str, series: pd.Series, args: list):
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
                    self._logger.debug("[Pandas] Dynamic call: series.%s(*args)", func_name)
                    return method(*args) if args else method()
                except Exception as e:
                    self._logger.debug("[Pandas] series.%s failed: %s", func_name, e)

        # Try Series.str accessor
        if hasattr(series, 'str') and hasattr(series.str, func_name):
            method = getattr(series.str, func_name)
            if callable(method):
                try:
                    self._logger.debug("[Pandas] Dynamic call: series.str.%s(*args)", func_name)
                    return method(*args) if args else method()
                except Exception as e:
                    self._logger.debug("[Pandas] series.str.%s failed: %s", func_name, e)

        # Try Series.dt accessor
        if hasattr(series, 'dt') and hasattr(series.dt, func_name):
            attr = getattr(series.dt, func_name)
            try:
                if callable(attr):
                    self._logger.debug("[Pandas] Dynamic call: series.dt.%s(*args)", func_name)
                    return attr(*args) if args else attr()
                else:
                    # It's a property
                    self._logger.debug("[Pandas] Dynamic access: series.dt.%s", func_name)
                    return attr
            except Exception as e:
                self._logger.debug("[Pandas] series.dt.%s failed: %s", func_name, e)

        return None

    def _evaluate_unknown_function(self, expr, df: pd.DataFrame, context: 'DataStore'):
        """
        Evaluate an unknown function - try Pandas dynamic method first, then chDB.

        For functions not registered in function_config, we try:
        1. Dynamic Pandas method invocation
        2. Fallback to chDB (which may have the function)

        Args:
            expr: Function expression
            df: DataFrame to operate on
            context: DataStore context

        Returns:
            Result of the function evaluation
        """
        func_name = expr.name.lower()

        # Evaluate first argument
        if not expr.args:
            return self._evaluate_function_via_chdb(expr, df, context)

        first_arg = self._evaluate_expr(expr.args[0], df, context)
        other_args = [self._evaluate_expr(arg, df, context) for arg in expr.args[1:]]

        # Try dynamic Pandas method first
        result = self._try_dynamic_pandas_method(func_name, first_arg, other_args)
        if result is not None:
            self._logger.debug("[Pandas] Dynamic method '%s' succeeded", func_name)
            return result

        # Fallback to chDB
        self._logger.debug("[Pandas] No dynamic method for '%s', trying chDB", func_name)
        return self._evaluate_function_via_chdb(expr, df, context)

    def _evaluate_function_via_chdb(self, expr, df: pd.DataFrame, context: 'DataStore'):
        """
        Evaluate a Function expression using chDB's Python() table function.

        This executes the function as SQL on the DataFrame via the centralized Executor.

        Args:
            expr: Function expression
            df: DataFrame to operate on
            context: DataStore context

        Returns:
            Result Series from chDB execution
        """
        from .executor import get_executor

        # Build the SQL expression
        sql_expr = expr.to_sql(quote_char='"')

        # Use centralized executor
        executor = get_executor()
        return executor.execute_expression(sql_expr, df)

    def describe(self) -> str:
        from .column_expr import ColumnExpr

        if isinstance(self.expr, ColumnExpr):
            # Use underlying expression's SQL, don't call repr (would materialize)
            expr_str = self.expr._expr.to_sql(quote_char='"')
        elif isinstance(self.expr, Expression):
            expr_str = self.expr.to_sql(quote_char='"')
        else:
            expr_str = repr(self.expr)
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

        # Handle ColumnExpr - unwrap
        if isinstance(expr, ColumnExpr):
            return self._determine_engine(expr._expr)

        if isinstance(expr, CastFunction):
            # CastFunction always uses chDB
            return 'chDB'

        elif isinstance(expr, Function):
            # Check function config
            func_name = expr.name.lower()
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
        self._logger.debug("[Pandas]   -> DataFrame shape after selection: %s", result.shape)
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
        self._logger.debug("[Pandas]   -> DataFrame shape after drop: %s", result.shape)
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
        self._logger.debug("[Pandas]   -> New columns: %s", list(result.columns))
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
        self._logger.debug("[Pandas]   -> New columns: %s", list(result.columns))
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
        self._logger.debug("[Pandas]   -> New columns: %s", list(result.columns))
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


class LazyAsType(LazyOp):
    """Cast column types."""

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def execute(self, df: pd.DataFrame, context: 'DataStore') -> pd.DataFrame:
        self._log_execute("AsType", f"dtype={self.dtype}")
        result = df.astype(self.dtype)
        self._logger.debug("[Pandas]   -> New dtypes: %s", dict(result.dtypes))
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
            self._logger.debug("[Pandas]   -> df[%s]", existing_cols)
            if existing_cols:
                result = df[existing_cols]
                self._logger.debug("[Pandas]   -> Selected columns: %s", existing_cols)
                return result
            return df
        elif self.op_type == 'WHERE' and self.condition is not None:
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
        from .expressions import Field, ArithmeticExpression, Literal
        from .functions import Function, CastFunction
        from .column_expr import ColumnExpr

        # Handle ColumnExpr - unwrap to get the underlying expression
        if isinstance(operand, ColumnExpr):
            return self._evaluate_operand(df, operand._expr, context)

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
        elif isinstance(operand, (Function, CastFunction)):
            # Handle Function expressions - evaluate via chDB or Pandas
            return self._evaluate_function(operand, df, context)
        elif isinstance(operand, Expression):
            # Handle other expression types - try to get value if available
            if hasattr(operand, 'value'):
                return operand.value
            return operand
        return operand  # Return literal value as-is

    def _evaluate_function(self, expr, df: pd.DataFrame, context: 'DataStore'):
        """
        Evaluate a Function expression.

        Uses chDB via Python() table function or Pandas implementation
        based on function_config.
        """
        from .functions import CastFunction
        from .function_executor import function_config

        func_name = expr.name.lower()

        # For CastFunction or if no Pandas implementation, use chDB
        if isinstance(expr, CastFunction) or not function_config.has_pandas_implementation(func_name):
            return self._evaluate_function_via_chdb(expr, df)

        # Check config for execution preference
        if function_config.should_use_pandas(func_name):
            pandas_impl = function_config.get_pandas_implementation(func_name)
            if pandas_impl and expr.args:
                try:
                    first_arg = self._evaluate_operand(df, expr.args[0], context)
                    other_args = [self._evaluate_operand(df, arg, context) for arg in expr.args[1:]]
                    return pandas_impl(first_arg, *other_args)
                except Exception as e:
                    self._logger.debug("[Pandas] Function %s failed: %s, falling back to chDB", func_name, e)

        # Fallback to chDB
        return self._evaluate_function_via_chdb(expr, df)

    def _evaluate_function_via_chdb(self, expr, df: pd.DataFrame):
        """Execute function via chDB's Python() table function using centralized Executor."""
        from .executor import get_executor

        sql_expr = expr.to_sql(quote_char='"')

        # Use centralized executor
        executor = get_executor()
        return executor.execute_expression(sql_expr, df)

    def describe(self) -> str:
        return f"{self.op_type}: {self.description}"

    def describe_pandas(self) -> str:
        """Describe using pandas terminology instead of SQL."""
        pandas_op = self._PANDAS_OP_NAMES.get(self.op_type, self.op_type)
        return f"{pandas_op}: {self.description}"

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


class LazySQLQuery(LazyOp):
    """
    Execute a SQL query on the current DataFrame using chDB's Python() table function.

    This enables true SQL-Pandas-SQL interleaving within the lazy pipeline.

    Supports two syntaxes:
    1. Short form (auto-adds SELECT * FROM __df__):
       - ds.sql("doubled > 100")  -> SELECT * FROM __df__ WHERE doubled > 100
       - ds.sql("doubled > 100 ORDER BY id")  -> SELECT * FROM __df__ WHERE doubled > 100 ORDER BY id
       - ds.sql("ORDER BY id LIMIT 5")  -> SELECT * FROM __df__ ORDER BY id LIMIT 5

    2. Full SQL form (when query contains SELECT/FROM/GROUP BY):
       - ds.sql("SELECT id, SUM(value) FROM __df__ GROUP BY id")

    Example:
        ds = DataStore.from_file('users.csv')
        ds = ds.filter(ds.age > 20)
        ds['doubled'] = ds['age'] * 2
        ds = ds.sql("doubled > 50 ORDER BY age DESC LIMIT 10")  # Short form!
        ds = ds.add_prefix('result_')
    """

    def __init__(self, query: str, df_alias: str = '__df__'):
        """
        Args:
            query: SQL query or condition. Can be:
                   - Full SQL: "SELECT * FROM __df__ WHERE x > 10"
                   - Short form: "x > 10" (auto-adds SELECT * FROM __df__ WHERE)
                   - Clauses only: "ORDER BY id LIMIT 5" (auto-adds SELECT * FROM __df__)
            df_alias: Alias for the DataFrame in the query (default: '__df__')
        """
        super().__init__()
        self.original_query = query.strip()
        self.df_alias = df_alias

        # Process the query to determine if it needs boilerplate
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
        """Execute the SQL query on the input DataFrame using chDB via centralized Executor."""
        from .executor import get_executor

        self._log_execute("SQL Query", f"rows={len(df)}", prefix="chDB")

        self._logger.debug("[chDB] Original input: %s", self.original_query)
        self._logger.debug("[chDB] Expanded query: %s", self.query)

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


# Add more operations as needed...
