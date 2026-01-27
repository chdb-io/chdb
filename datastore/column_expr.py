"""
ColumnExpr - A column expression that can execute when displayed.

This provides pandas-like behavior where accessing a column or performing
operations on it shows actual values when displayed, while still supporting
lazy expression building for filters and assignments.

Uses composition (not inheritance) to wrap Expression and return ColumnExpr
for all operations. This ensures pandas-like behavior is preserved.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Iterator, List

import pandas as pd

from .expressions import Expression, Field, ArithmeticExpression, Literal, Node
from .lazy_result import LazySeries, LazyCondition
from .utils import immutable
from .exceptions import ImmutableError


def _parse_pandas_version():
    """Parse pandas version to a tuple for comparison."""
    try:
        parts = pd.__version__.split(".")[:2]
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0)


# Check if pandas version supports limit_area parameter (added in pandas 2.1.0)
_PANDAS_HAS_LIMIT_AREA = _parse_pandas_version() >= (2, 1)

if TYPE_CHECKING:
    from .core import DataStore
    from .conditions import Condition, BinaryCondition


# Sentinel value for detecting if argument was passed
_MISSING = object()


class ColumnExpr:
    """
    A unified column expression that supports lazy evaluation in multiple modes.

    This is the single type for all column-level operations in DataStore,
    providing a pandas Series-like interface with lazy execution.

    Execution Modes:
    1. Expression mode: Wraps an Expression (Field, ArithmeticExpression, Function)
    2. Method mode: Wraps a method call on another ColumnExpr (e.g., value_counts)
    3. Aggregation mode: Wraps an aggregation operation (e.g., mean, sum)
    4. Executor mode: Uses a custom callable for complex operations

    When displayed (via __repr__, __str__, or IPython), it executes the
    expression and shows actual values like a pandas Series.

    When used in operations:
    - Arithmetic (+, -, *, /) returns another ColumnExpr
    - Comparisons (>, <, ==, etc.) return Conditions (for filtering)
    - Methods like .str.upper() return ColumnExpr wrapping Functions

    This provides a pandas-like experience while preserving lazy evaluation.

    Example:
        >>> ds = DataStore.from_file('users.csv')
        >>> ds['age']  # Returns ColumnExpr, displays actual values
        0    28
        1    31
        2    29
        Name: age, dtype: int64

        >>> ds['age'] - 10  # Returns ColumnExpr, displays computed values
        0    18
        1    21
        2    19
        Name: age, dtype: int64

        >>> ds['age'].value_counts()  # Returns ColumnExpr (method mode)
        28    1
        31    1
        29    1
        Name: age, dtype: int64

        >>> ds['age'].mean()  # Returns ColumnExpr (aggregation mode)
        29.333...

        >>> ds['age'] > 25  # Returns Condition (for filtering)
        BinaryCondition('"age" > 25')
    """

    def __init__(
        self,
        expr: Expression = None,
        datastore: "DataStore" = None,
        alias: Optional[str] = None,
        groupby_fields: Optional[List] = None,
        groupby_as_index: bool = True,
        groupby_sort: bool = True,
        groupby_dropna: bool = True,
        # Method mode parameters
        source: "ColumnExpr" = None,
        method_name: str = None,
        method_args: tuple = None,
        method_kwargs: dict = None,
        # Aggregation mode parameters
        agg_func_name: str = None,
        pandas_agg_func: str = None,
        skipna: bool = True,
        # Executor mode parameters (DEPRECATED - use op mode instead)
        executor: Any = None,
        # Operation descriptor mode parameters (preferred over executor)
        op_type: str = None,  # 'accessor' | 'method' | 'groupby_transform' | 'groupby_agg'
        op_source: "ColumnExpr" = None,  # source expression to evaluate first
        op_accessor: str = None,  # 'str' | 'dt' | 'json' (for accessor ops)
        op_method: str = None,  # method name to call
        op_args: tuple = None,  # positional arguments
        op_kwargs: dict = None,  # keyword arguments
        op_groupby_cols: list = None,  # groupby columns (for groupby ops)
    ):
        """
        Initialize ColumnExpr with expression and DataStore reference.

        Supports multiple modes:
        1. Expression mode: expr + datastore
        2. Method mode: source + method_name (+ args/kwargs)
        3. Aggregation mode: source + agg_func_name
        4. Operation descriptor mode: op_type + op_source + op_method (PREFERRED)
        5. Executor mode: executor + datastore (DEPRECATED - use op mode instead)

        Args:
            expr: The underlying expression (Field, ArithmeticExpression, Function, etc.)
            datastore: Reference to the DataStore for execution
            alias: Optional alias for the expression
            groupby_fields: Optional groupby fields from LazyGroupBy
            groupby_as_index: If True (default), groupby keys become the index.
                              If False, groupby keys are returned as columns.
            groupby_sort: If True (default), sort by groupby keys.
            groupby_dropna: If True (default), exclude NA/null values in keys.
            source: Source ColumnExpr for method/aggregation mode
            method_name: Method name to call on source (method mode)
            method_args: Positional arguments for method call
            method_kwargs: Keyword arguments for method call
            agg_func_name: SQL aggregate function name (aggregation mode)
            pandas_agg_func: Pandas aggregation method name
            skipna: Whether to skip NaN values in aggregation
            executor: Custom callable for complex operations (DEPRECATED)
            op_type: Operation type for descriptor mode ('accessor', 'method', etc.)
            op_source: Source ColumnExpr to evaluate first (op mode)
            op_accessor: Accessor name ('str', 'dt', 'json') for accessor ops
            op_method: Method name to call on source/accessor
            op_args: Positional arguments for the method
            op_kwargs: Keyword arguments for the method
            op_groupby_cols: Groupby column names (for groupby ops)
        """
        # Expression mode fields
        self._expr = expr
        self._alias = alias
        self._groupby_fields = groupby_fields or []
        self._groupby_as_index = groupby_as_index
        self._groupby_sort = groupby_sort
        self._groupby_dropna = groupby_dropna

        # Method/Aggregation mode fields
        self._source = source
        self._method_name = method_name
        self._method_args = method_args or ()
        self._method_kwargs = method_kwargs or {}

        # Aggregation mode fields
        self._agg_func_name = agg_func_name
        self._pandas_agg_func = pandas_agg_func or agg_func_name
        self._skipna = skipna

        # Executor mode (DEPRECATED - use op mode instead)
        self._executor = executor

        # Operation descriptor mode fields (PREFERRED)
        self._op_type = op_type
        self._op_source = op_source
        self._op_accessor = op_accessor
        self._op_method = op_method
        self._op_args = op_args or ()
        self._op_kwargs = op_kwargs or {}
        self._op_groupby_cols = op_groupby_cols or []

        # DataStore reference (from different sources)
        if datastore is not None:
            self._datastore = datastore
        elif source is not None:
            self._datastore = source._datastore
        else:
            self._datastore = None

        # Inherit groupby_fields from source if not provided
        if (
            not self._groupby_fields
            and source is not None
            and hasattr(source, "_groupby_fields")
        ):
            self._groupby_fields = (
                source._groupby_fields.copy() if source._groupby_fields else []
            )

        # Cache for executed result
        self._cached_result = None

        # Determine execution mode
        # Priority: op mode > executor mode (op is the preferred new pattern)
        if op_type is not None:
            self._exec_mode = "op"
        elif executor is not None:
            self._exec_mode = "executor"
        elif agg_func_name is not None:
            self._exec_mode = "agg"
            # Create AggregateFunction for SQL building (needed by agg() method)
            if source is not None and source._expr is not None:
                from .functions import AggregateFunction

                self._expr = AggregateFunction(agg_func_name, source._expr)
        elif method_name is not None:
            self._exec_mode = "method"
        else:
            self._exec_mode = "expr"

    @property
    def expr(self) -> Expression:
        """Get the underlying expression."""
        return self._expr

    @property
    def datastore(self) -> "DataStore":
        """Get the DataStore reference."""
        return self._datastore

    @property
    def alias(self) -> Optional[str]:
        """Get the alias."""
        return self._alias

    @property
    def columns(self):
        """
        Get columns of the underlying result if it's a DataFrame.

        This property is primarily useful when as_index=False in groupby,
        which returns a DataFrame instead of a Series.

        Returns:
            pd.Index: Column names if result is DataFrame
            None: If result is not a DataFrame

        Example:
            >>> ds.groupby('category', as_index=False)['value'].sum().columns
            Index(['category', 'value'], dtype='object')
        """
        result = self._execute()
        if isinstance(result, pd.DataFrame):
            return result.columns
        return None

    # ========== Factory Methods ==========

    def _derive_expr(
        self, expr: Expression, alias: Optional[str] = None
    ) -> "ColumnExpr":
        """
        Create a new ColumnExpr derived from this one, preserving groupby context.

        This is the preferred way to create new ColumnExpr instances when
        performing operations that produce a new expression. It ensures
        groupby_fields, groupby_as_index, and groupby_sort are properly
        propagated.

        Use this instead of direct ColumnExpr(expr, self._datastore) calls
        to avoid losing groupby context parameters.

        Args:
            expr: The new expression (Field, ArithmeticExpression, Function, etc.)
            alias: Optional alias for the expression

        Returns:
            ColumnExpr with groupby context preserved from self

        Example:
            # Instead of:
            return self._derive_expr(new_expr)

            # Use:
            return self._derive_expr(new_expr)
        """
        return ColumnExpr(
            expr=expr,
            datastore=self._datastore,
            alias=alias,
            groupby_fields=(
                self._groupby_fields.copy() if self._groupby_fields else None
            ),
            groupby_as_index=self._groupby_as_index,
            groupby_sort=self._groupby_sort,
            groupby_dropna=self._groupby_dropna,
        )

    # ========== Operation Descriptor Factory Methods ==========
    # These create ColumnExprs using the 'op' mode which is safe from recursion.
    # Use these instead of creating executors that call _execute().

    @classmethod
    def from_accessor_op(
        cls,
        source: "ColumnExpr",
        accessor: str,
        method: str,
        args: tuple = (),
        kwargs: dict = None,
    ) -> "ColumnExpr":
        """
        Create a ColumnExpr for an accessor operation (e.g., str.extract, dt.year).

        This is the preferred way to create lazy accessor operations. The evaluator
        can safely execute this by:
        1. Evaluating the source expression using the current df
        2. Getting the accessor (e.g., series.str)
        3. Calling the method with args/kwargs

        This avoids the recursion problem of executor closures that call _execute().

        Args:
            source: The source ColumnExpr to evaluate first
            accessor: The accessor name ('str', 'dt', 'json')
            method: The method name to call on the accessor
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method

        Returns:
            ColumnExpr in 'op' mode

        Example:
            >>> # Instead of:
            >>> def executor():
            >>>     series = col_expr._execute()  # DANGEROUS - causes recursion!
            >>>     return series.str.extract(pat)
            >>> return ColumnExpr(executor=executor, datastore=ds)
            >>>
            >>> # Use:
            >>> return ColumnExpr.from_accessor_op(col_expr, 'str', 'extract', (pat,), {'expand': False})
        """
        return cls(
            datastore=source._datastore,
            op_type="accessor",
            op_source=source,
            op_accessor=accessor,
            op_method=method,
            op_args=args,
            op_kwargs=kwargs or {},
        )

    @classmethod
    def from_method_op(
        cls,
        source: "ColumnExpr",
        method: str,
        args: tuple = (),
        kwargs: dict = None,
    ) -> "ColumnExpr":
        """
        Create a ColumnExpr for a method operation (e.g., fillna, replace).

        Similar to from_accessor_op but for direct Series methods.

        Args:
            source: The source ColumnExpr to evaluate first
            method: The method name to call on the source series
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method

        Returns:
            ColumnExpr in 'op' mode
        """
        return cls(
            datastore=source._datastore,
            op_type="method",
            op_source=source,
            op_method=method,
            op_args=args,
            op_kwargs=kwargs or {},
        )

    @classmethod
    def from_groupby_transform_op(
        cls,
        source: "ColumnExpr",
        groupby_cols: list,
        func: str,
        args: tuple = (),
        kwargs: dict = None,
    ) -> "ColumnExpr":
        """
        Create a ColumnExpr for a groupby transform operation.

        Args:
            source: The source ColumnExpr (column to transform)
            groupby_cols: List of column names to group by
            func: Transform function name or callable
            args: Positional arguments for the transform
            kwargs: Keyword arguments for the transform

        Returns:
            ColumnExpr in 'op' mode
        """
        result = cls(
            datastore=source._datastore,
            op_type="groupby_transform",
            op_source=source,
            op_method=func if isinstance(func, str) else None,
            op_args=args,
            op_kwargs=kwargs or {},
            op_groupby_cols=groupby_cols,
        )
        # Store the function if it's a callable
        if not isinstance(func, str):
            result._op_func = func
        return result

    def _create_window_method(
        self, method_name: str, sql_func: str, **method_kwargs
    ) -> "ColumnExpr":
        """
        Create a ColumnExpr for a window method (cumsum, cummax, cummin, rank, shift, diff).

        When _groupby_fields is set AND data source is SQL-based (has _table_function),
        this creates a SQL window function expression:
            SUM(col) OVER (PARTITION BY groupby_col ORDER BY rowNumberInAllBlocks() ROWS UNBOUNDED PRECEDING)

        The ORDER BY rowNumberInAllBlocks() is crucial to maintain original row order,
        which matches pandas behavior where cumulative operations follow the DataFrame's
        original row order within each group.

        Without _groupby_fields or without a SQL data source, it falls back to pandas.

        Args:
            method_name: pandas method name ('cumsum', 'cummax', etc.)
            sql_func: SQL aggregate function name ('sum', 'max', 'min', etc.)
            **method_kwargs: Additional keyword arguments for pandas method

        Returns:
            ColumnExpr with SQL window function or pandas method fallback
        """
        from .functions import WindowFunction, Function
        from .expressions import Field

        # If no groupby context, fall back to pandas method mode
        if not self._groupby_fields:
            return ColumnExpr(
                source=self,
                method_name=method_name,
                method_kwargs=method_kwargs,
            )

        # Get column field from expression
        col_field = self._expr
        if col_field is None:
            # Fallback to pandas if no expression
            return ColumnExpr(
                source=self,
                method_name=method_name,
                method_kwargs=method_kwargs,
            )

        # Check if data source is SQL-based (has _table_function)
        # Window functions via Python() table function have row order issues,
        # so only use SQL for native SQL sources (parquet files, etc.)
        has_sql_source = (
            self._datastore._table_function is not None if self._datastore else False
        )
        if not has_sql_source:
            # Fall back to pandas for DataFrame-based sources
            # This avoids row order issues with Python() table function
            return ColumnExpr(
                source=self,
                method_name=method_name,
                method_kwargs=method_kwargs,
            )

        # Extract partition by columns from groupby_fields
        partition_by = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                partition_by.append(gf.name)
            else:
                partition_by.append(str(gf))

        # Create SQL window function with ORDER BY rowNumberInAllBlocks() to preserve row order
        # Format: SUM(col) OVER (PARTITION BY groupby_col ORDER BY rowNumberInAllBlocks() ROWS UNBOUNDED PRECEDING)
        # The ORDER BY ensures cumulative operations follow the original row order within each group
        row_num_func = Function("rowNumberInAllBlocks")
        window_func = WindowFunction(sql_func, col_field).over(
            partition_by=partition_by,
            order_by=[row_num_func],
            frame="ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        )

        # Return a new ColumnExpr with the window function expression
        # Preserve groupby context for potential chained operations
        result = ColumnExpr(
            expr=window_func,
            datastore=self._datastore,
            groupby_fields=(
                self._groupby_fields.copy() if self._groupby_fields else None
            ),
            groupby_as_index=self._groupby_as_index,
            groupby_sort=self._groupby_sort,
            groupby_dropna=self._groupby_dropna,
        )
        # Store fallback info for pandas execution if SQL fails
        result._window_method_fallback = {
            "method_name": method_name,
            "method_kwargs": method_kwargs,
            "source": self,
        }
        # Mark this expression as needing row order preservation in final SQL
        result._needs_row_order_preservation = True
        return result

    # ========== Classification Methods ==========

    # DateTime methods that should always use pandas execution
    # These have incompatible behavior between chDB SQL and pandas
    _PANDAS_ONLY_DT_METHODS = frozenset(
        {
            "strftime",  # Format code differences (%M) + timezone issues
            "tz_localize",  # Timezone handling
            "tz_convert",  # Timezone handling
            "to_pydatetime",  # Python object conversion
            "to_pytimedelta",  # Python object conversion
        }
    )

    def is_pandas_only(self) -> bool:
        """
        Check if this expression can ONLY be executed via Pandas.

        Returns True for:
        - executor mode (e.g., groupby().transform() results)
        - method mode without SQL-compatible expression
        - expressions that reference pandas-only operations
        - DateTimeMethodExpr with methods known to have incompatible SQL behavior

        This is the canonical method to determine execution path.
        Use this instead of checking _executor/_expr directly.
        """
        # Executor mode always requires Pandas execution
        if self._exec_mode == "executor":
            return True

        # Method mode without underlying expression needs Pandas
        if self._exec_mode == "method" and self._expr is None:
            return True

        # Check if expression contains pandas-only datetime methods
        if self._expr is not None:
            from .expressions import DateTimeMethodExpr

            if isinstance(self._expr, DateTimeMethodExpr):
                if self._expr.method_name in self._PANDAS_ONLY_DT_METHODS:
                    return True

        return False

    def is_sql_compatible(self) -> bool:
        """
        Check if this expression can be converted to SQL.

        Returns True only if:
        - Has a valid _expr (Expression tree)
        - Is NOT in pandas-only mode
        - Is NOT in 'op' mode (operation descriptors use Pandas execution)

        This is the canonical method to determine if SQL pushdown is possible.
        """
        # Operation descriptor mode always uses Pandas execution
        # (accessor ops, groupby transforms, etc. cannot be SQL-pushed)
        if self._exec_mode == "op":
            return False
        # Must have an expression tree AND not be pandas-only
        return self._expr is not None and not self.is_pandas_only()

    # ========== Execution ==========

    def _execute(self):
        """
        Execute this expression and return the result.

        Handles different execution modes:
        - expr: Execute expression tree
        - method: Call method on source result
        - agg: Execute aggregation (scalar or Series with groupby)
        - op: Execute operation descriptor (PREFERRED - safe from recursion)
        - executor: Call custom executor (DEPRECATED)

        Returns:
            pd.Series, scalar, or other result depending on operation
        """
        # Return cached result if available
        if self._cached_result is not None:
            return self._cached_result

        if self._exec_mode == "op":
            self._cached_result = self._execute_op()
        elif self._exec_mode == "executor":
            self._cached_result = self._execute_executor()
        elif self._exec_mode == "agg":
            self._cached_result = self._execute_aggregation()
        elif self._exec_mode == "method":
            self._cached_result = self._execute_method()
        else:
            self._cached_result = self._execute_expression()

        return self._cached_result

    def _execute_expression(self) -> pd.Series:
        """Execute expression mode - evaluate the expression tree."""
        from .expression_evaluator import ExpressionEvaluator
        from .conditions import Condition

        # Get the executed DataFrame from the DataStore
        df = self._datastore._execute()

        # Use unified expression evaluator (respects function_config)
        evaluator = ExpressionEvaluator(df, self._datastore)
        result = evaluator.evaluate(self._expr)

        # Ensure we return a Series with proper index
        if isinstance(result, pd.Series):
            # Apply alias as series name if set, or explicitly None
            # This ensures we don't keep evaluator's default name (like __result__)
            if self._alias is not None:
                result = result.copy()
                result.name = self._alias
            elif isinstance(result.name, str) and result.name.startswith("__"):
                # Clear internal names like __result__ when no explicit alias
                result = result.copy()
                result.name = None
            return result
        elif isinstance(self._expr, Literal):
            # Literal value - expand to Series
            return pd.Series([result] * len(df), index=df.index)
        elif isinstance(self._expr, Condition) and isinstance(result, bool):
            # Boolean condition returning scalar (e.g., 1=1 for None comparison)
            # Expand to Series to match pandas behavior
            return pd.Series([result] * len(df), index=df.index, dtype=bool)
        else:
            # Scalar result - return as is (will be wrapped if needed)
            return result

    def _execute_method(self):
        """Execute method mode - call method on source result."""
        from .expressions import Field

        # Execute any ColumnExpr in args or kwargs
        args = tuple(self._execute_if_needed(arg) for arg in self._method_args)
        kwargs = {k: self._execute_if_needed(v) for k, v in self._method_kwargs.items()}

        # Handle groupby + agg scenario: df.groupby('col')['value'].agg(['sum', 'mean'])
        if (
            self._source is not None
            and hasattr(self._source, "_groupby_fields")
            and self._source._groupby_fields
            and self._method_name in ("agg", "aggregate")
        ):
            groupby_col_names = []
            for gf in self._source._groupby_fields:
                if isinstance(gf, Field):
                    groupby_col_names.append(gf.name)
                else:
                    groupby_col_names.append(str(gf))

            col_name = None
            if self._source._expr is not None and isinstance(self._source._expr, Field):
                col_name = self._source._expr.name
            elif self._source._expr is not None:
                col_name = str(self._source._expr)

            df = self._datastore._execute()
            agg_kwargs = {k: v for k, v in kwargs.items() if k != "axis"}

            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_col_names)[col_name]
                return grouped.agg(*args, **agg_kwargs)
            else:
                series = self._source._execute()
                method = getattr(series, self._method_name)
                return method(*args, **agg_kwargs)

        # Handle groupby + window function scenario: df.groupby('col')['value'].cumsum(), rank(), etc.
        # These methods need to operate within each group, not globally
        _GROUPBY_WINDOW_METHODS = {
            "cumsum",
            "cummax",
            "cummin",
            "cumprod",
            "rank",
            "diff",
            "shift",
            "pct_change",
        }
        if (
            self._source is not None
            and hasattr(self._source, "_groupby_fields")
            and self._source._groupby_fields
            and self._method_name in _GROUPBY_WINDOW_METHODS
        ):
            groupby_col_names = []
            for gf in self._source._groupby_fields:
                if isinstance(gf, Field):
                    groupby_col_names.append(gf.name)
                else:
                    groupby_col_names.append(str(gf))

            col_name = None
            if self._source._expr is not None and isinstance(self._source._expr, Field):
                col_name = self._source._expr.name
            elif self._source._expr is not None:
                col_name = str(self._source._expr)

            df = self._datastore._execute()
            # Different methods accept different parameters - filter appropriately
            # rank() in groupby doesn't accept 'numeric_only'
            if self._method_name == "rank":
                method_kwargs = {
                    k: v for k, v in kwargs.items() if k not in ("axis", "numeric_only")
                }
            else:
                method_kwargs = {k: v for k, v in kwargs.items() if k not in ("axis",)}

            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_col_names)[col_name]
                method = getattr(grouped, self._method_name)
                return method(**method_kwargs)
            else:
                # Fallback: execute source and apply method
                series = self._source._execute()
                method = getattr(series, self._method_name)
                return method(**method_kwargs)

        # Handle groupby + apply scenario: df.groupby('col')['value'].apply(func)
        # apply() with groupby should pass each group (as Series) to the function,
        # not individual elements
        if (
            self._source is not None
            and hasattr(self._source, "_groupby_fields")
            and self._source._groupby_fields
            and self._method_name == "apply"
        ):
            groupby_col_names = []
            for gf in self._source._groupby_fields:
                if isinstance(gf, Field):
                    groupby_col_names.append(gf.name)
                else:
                    groupby_col_names.append(str(gf))

            col_name = None
            if self._source._expr is not None and isinstance(self._source._expr, Field):
                col_name = self._source._expr.name
            elif self._source._expr is not None:
                col_name = str(self._source._expr)

            df = self._datastore._execute()
            # Get groupby sort and dropna settings from source
            sort = getattr(self._source, "_groupby_sort", True)
            dropna = getattr(self._source, "_groupby_dropna", True)

            # Filter kwargs for apply - remove convert_dtype and args as pandas SeriesGroupBy.apply
            # doesn't accept them (they're only for Series.apply)
            apply_kwargs = {
                k: v for k, v in kwargs.items() if k not in ("convert_dtype", "args")
            }

            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_col_names, sort=sort, dropna=dropna)[
                    col_name
                ]
                # args[0] is the function
                return grouped.apply(args[0], **apply_kwargs)
            else:
                # Fallback for computed columns - use pandas directly
                series = self._source._execute()
                return series.apply(*args, **kwargs)

        # Handle value_counts with SQL pushdown
        if self._method_name == "value_counts":
            return self._execute_value_counts_sql(args, kwargs)

        # Execute source
        if self._source is not None:
            series = self._source._execute()
        else:
            series = None

        # Handle special _dt_* methods for datetime operations
        if self._method_name and self._method_name.startswith("_dt_"):
            dt_attr = self._method_name[4:]
            if not pd.api.types.is_datetime64_any_dtype(series):
                if series.dtype == "object" or pd.api.types.is_string_dtype(series):
                    try:
                        series = pd.to_datetime(series, errors="coerce")
                    except Exception:
                        pass
            dt_accessor = series.dt
            attr = getattr(dt_accessor, dt_attr)
            if callable(attr):
                return attr(*args, **kwargs)
            return attr

        # Handle chained function calls from method-mode ColumnExpr
        # These are created when calling e.g., ds['a'].fillna(0).abs()
        # where fillna returns method-mode ColumnExpr and abs() chains on it
        if self._method_name and self._method_name.startswith("_chain_"):
            func_name = self._method_name[7:]  # Remove '_chain_' prefix
            # Remove 'alias' from kwargs if present (SQL-specific parameter)
            kwargs_clean = {k: v for k, v in kwargs.items() if k != "alias"}
            if hasattr(series, func_name):
                method = getattr(series, func_name)
                return method(*args, **kwargs_clean)
            else:
                # Try numpy function as fallback
                import numpy as np

                if hasattr(np, func_name):
                    np_func = getattr(np, func_name)
                    return np_func(series, *args)
                raise AttributeError(
                    f"'{type(series).__name__}' has no attribute '{func_name}'"
                )

        # Execute the method
        if series is None:
            return None

        if not hasattr(series, self._method_name):
            return series

        method = getattr(series, self._method_name)
        return method(*args, **kwargs)

    def _execute_value_counts_sql(self, args: tuple, kwargs: dict) -> pd.Series:
        """
        Execute value_counts with SQL pushdown when possible.

        SQL equivalent:
            SELECT col, COUNT(*) AS count
            FROM table
            [WHERE col IS NOT NULL]  -- when dropna=True
            GROUP BY col

        Post-processing ensures pandas-compatible tie-breaking:
        - When sort=True: sort by count, ties broken by first-appearance order
        - When sort=False: order by first-appearance

        For normalize=True:
            SELECT col, CAST(COUNT(*) AS Float64) / total AS proportion
            ...

        Args:
            args: Positional arguments (unused for value_counts)
            kwargs: Keyword arguments (normalize, sort, ascending, bins, dropna)

        Returns:
            pd.Series with counts/proportions, indexed by unique values
        """
        from .config import get_execution_engine, ExecutionEngine
        from .executor import get_executor
        from .expressions import Field

        # Extract parameters with defaults matching pandas
        normalize = kwargs.get("normalize", False)
        sort = kwargs.get("sort", True)
        ascending = kwargs.get("ascending", False)
        bins = kwargs.get("bins", None)
        dropna = kwargs.get("dropna", True)

        # bins parameter requires special handling - fall back to pandas
        # because SQL doesn't have a direct equivalent to pandas cut/qcut
        if bins is not None:
            # Fall back to pandas for bins
            series = (
                self._source._execute()
                if self._source is not None
                else self._execute_expression()
            )
            return series.value_counts(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )

        # Check execution engine
        engine = get_execution_engine()

        # Use pandas for PANDAS engine
        if engine == ExecutionEngine.PANDAS:
            series = (
                self._source._execute()
                if self._source is not None
                else self._execute_expression()
            )
            return series.value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            )

        # Get the DataFrame from datastore
        df = self._datastore._execute()

        # Get column name from expression
        col_name = None
        if self._source is not None:
            if hasattr(self._source, "_expr") and self._source._expr is not None:
                if isinstance(self._source._expr, Field):
                    col_name = self._source._expr.name
                else:
                    # Could be an expression - get SQL representation
                    col_name = self._source._expr.to_sql(quote_char='"')

        # If we can't determine the column, fall back to pandas
        if col_name is None or (
            not isinstance(self._source._expr, Field) and col_name not in df.columns
        ):
            series = (
                self._source._execute()
                if self._source is not None
                else self._execute_expression()
            )
            return series.value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            )

        # Build SQL query for value_counts
        try:
            executor = get_executor()

            # Determine if we're dealing with a simple column or expression
            if isinstance(self._source._expr, Field):
                col_sql = f'"{col_name}"'
                is_simple_column = True
            else:
                col_sql = col_name  # Already a SQL expression
                is_simple_column = False

            # Build WHERE clause for dropna
            where_clause = ""
            if dropna:
                where_clause = f" WHERE {col_sql} IS NOT NULL"

            # Build SELECT and GROUP BY (no ORDER BY - we'll sort in Python for pandas compatibility)
            if normalize:
                # For normalize, we need to calculate proportion
                # Use subquery to get total count
                total_sql = f"(SELECT COUNT(*) FROM __df__{where_clause})"
                select_sql = f"{col_sql} AS __value__, CAST(COUNT(*) AS Float64) / {total_sql} AS __count__"
            else:
                select_sql = f"{col_sql} AS __value__, toInt64(COUNT(*)) AS __count__"

            # Build complete SQL (no ORDER BY - sorting done in Python)
            sql = f"SELECT {select_sql} FROM __df__{where_clause} GROUP BY {col_sql}"

            # Execute SQL
            result_df = executor.query_dataframe(sql, df, preserve_order=False)

            # Convert to Series with proper index
            if len(result_df) == 0:
                # Empty result - return empty Series matching pandas behavior
                result = pd.Series(
                    [], dtype="int64" if not normalize else "float64", name="count"
                )
                result.index.name = col_name if is_simple_column else None
                return result

            # Get first-appearance order from original data for pandas-compatible tie-breaking
            # This ensures that when counts are tied, values appear in their first-occurrence order
            if is_simple_column and col_name in df.columns:
                original_series = df[col_name]
                if dropna:
                    original_series = original_series.dropna()
                # Get first appearance position for each unique value
                first_appearance = {}
                for idx, val in enumerate(original_series):
                    if val not in first_appearance:
                        first_appearance[val] = idx
            else:
                # For expressions, we can't easily get first appearance, use index order
                first_appearance = {
                    val: idx for idx, val in enumerate(result_df["__value__"])
                }

            # Create result Series
            result = result_df.set_index("__value__")["__count__"]

            # Sort according to pandas behavior
            if sort:
                # pandas sorts by count (desc by default), ties broken by first-appearance order
                # Create a DataFrame for sorting with count and first_appearance
                sort_df = pd.DataFrame(
                    {
                        "value": result.index,
                        "count": result.values,
                        "first_pos": [
                            first_appearance.get(v, float("inf")) for v in result.index
                        ],
                    }
                )
                # Sort by count (ascending or descending), then by first_pos (always ascending for ties)
                sort_df = sort_df.sort_values(
                    by=["count", "first_pos"], ascending=[ascending, True]
                )
                # Rebuild result with correct order
                result = pd.Series(
                    sort_df["count"].values,
                    index=sort_df["value"].values,
                    name="count" if not normalize else "proportion",
                )
            else:
                # sort=False: pandas returns in first-appearance order
                sort_df = pd.DataFrame(
                    {
                        "value": result.index,
                        "count": result.values,
                        "first_pos": [
                            first_appearance.get(v, float("inf")) for v in result.index
                        ],
                    }
                )
                sort_df = sort_df.sort_values(by="first_pos", ascending=True)
                result = pd.Series(
                    sort_df["count"].values,
                    index=sort_df["value"].values,
                    name="count" if not normalize else "proportion",
                )

            result.index.name = col_name if is_simple_column else None

            return result

        except Exception as e:
            # Log and fall back to pandas on any SQL error
            from .config import get_logger

            logger = get_logger()
            logger.debug(
                f"value_counts SQL pushdown failed, falling back to pandas: {e}"
            )

            series = (
                self._source._execute()
                if self._source is not None
                else self._execute_expression()
            )
            return series.value_counts(
                normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
            )

    def _execute_aggregation(self):
        """Execute aggregation mode - compute aggregate with optional groupby."""
        from .config import get_execution_engine, ExecutionEngine
        from .expressions import Field

        # Check if we have groupby fields
        if self._groupby_fields:
            return self._execute_groupby_aggregation()
        else:
            # No groupby - compute scalar value
            series = (
                self._source._execute()
                if self._source is not None
                else self._execute_expression()
            )
            agg_method = getattr(series, self._pandas_agg_func)
            return agg_method(
                **{
                    k: v
                    for k, v in self._method_kwargs.items()
                    if k not in ("axis", "numeric_only")
                }
            )

    def _execute_groupby_aggregation(self):
        """Execute aggregation with GROUP BY."""
        from .config import get_execution_engine, ExecutionEngine
        from .expressions import Field
        from .executor import get_executor

        # Get groupby column names
        groupby_col_names = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_col_names.append(gf.name)
            else:
                groupby_col_names.append(str(gf))

        # Temporarily clear groupby fields on datastore to prevent SQL generation issues
        datastore = self._datastore
        original_groupby = (
            datastore._groupby_fields.copy() if datastore._groupby_fields else []
        )
        datastore._groupby_fields = []

        # Get as_index and sort parameters
        as_index = getattr(self, "_groupby_as_index", True)
        sort = getattr(self, "_groupby_sort", True)

        try:
            df = datastore._execute()

            # Check execution engine configuration
            engine = get_execution_engine()
            col_name = self._get_column_name()

            if engine == ExecutionEngine.PANDAS:
                # Use pure pandas groupby with as_index, sort, and dropna parameters
                dropna = getattr(self, "_groupby_dropna", True)
                grouped = df.groupby(
                    groupby_col_names, sort=sort, as_index=as_index, dropna=dropna
                )
                agg_method = getattr(grouped[col_name], self._pandas_agg_func)
                result = agg_method()
                if as_index:
                    # Returns Series with groupby keys as index
                    result.name = col_name
                else:
                    # Returns DataFrame with groupby keys as columns
                    if isinstance(result, pd.Series):
                        result = result.reset_index()
                        result.columns = list(groupby_col_names) + [col_name]
                return result
            else:
                # Use chDB for CHDB or AUTO
                col_expr_sql = (
                    self._source._expr.to_sql(quote_char='"')
                    if self._source and self._source._expr
                    else '"' + col_name + '"'
                )
                groupby_sql = ", ".join(f'"{name}"' for name in groupby_col_names)

                # Determine null check function based on column type
                # isNaN() only works for numeric types, use isNull() for strings/other types
                col_dtype = df[col_name].dtype if col_name in df.columns else None
                is_numeric = col_dtype is not None and pd.api.types.is_numeric_dtype(
                    col_dtype
                )
                null_check_fn = "isNaN" if is_numeric else "isNull"

                # Special handling for first/last: use argMin/argMax with rowNumberInAllBlocks()
                # to preserve row order semantics (any/anyLast don't guarantee order)
                # NOTE: For Python() table function, connection.py replaces rowNumberInAllBlocks()
                # with _row_id (deterministic) at execution time (chDB v4.0.0b5+)
                # For file sources (Parquet, CSV), rowNumberInAllBlocks() works correctly as-is
                if self._agg_func_name == "any":
                    # first() -> argMin(value, rowNumberInAllBlocks())
                    if self._skipna:
                        agg_sql = f"argMinIf({col_expr_sql}, rowNumberInAllBlocks(), NOT {null_check_fn}({col_expr_sql}))"
                    else:
                        agg_sql = f"argMin({col_expr_sql}, rowNumberInAllBlocks())"
                elif self._agg_func_name == "anyLast":
                    # last() -> argMax(value, rowNumberInAllBlocks())
                    if self._skipna:
                        agg_sql = f"argMaxIf({col_expr_sql}, rowNumberInAllBlocks(), NOT {null_check_fn}({col_expr_sql}))"
                    else:
                        agg_sql = f"argMax({col_expr_sql}, rowNumberInAllBlocks())"
                elif self._skipna:
                    agg_sql = f"{self._agg_func_name}If({col_expr_sql}, NOT {null_check_fn}({col_expr_sql}))"
                else:
                    agg_sql = f"{self._agg_func_name}({col_expr_sql})"

                # Wrap count in toInt64() to match pandas dtype (int64 instead of uint64)
                if self._agg_func_name == "count":
                    agg_sql = f"toInt64({agg_sql})"

                # Add ORDER BY if sort=True (pandas default behavior)
                order_by = f" ORDER BY {groupby_sql}" if sort else ""

                # Add WHERE clause for dropna=True (filter out NULL groups)
                dropna = getattr(self, "_groupby_dropna", True)
                where_clause = ""
                if dropna:
                    null_filter_conditions = [
                        f'"{name}" IS NOT NULL' for name in groupby_col_names
                    ]
                    where_clause = " WHERE " + " AND ".join(null_filter_conditions)

                sql = f'SELECT {groupby_sql}, {agg_sql} AS "{col_name}" FROM __df__{where_clause} GROUP BY {groupby_sql}{order_by}'

                executor = get_executor()
                result_df = executor.query_dataframe(sql, df)

                # Fix for sum of all-NaN: SQL returns NULL, pandas returns 0
                # When using sumIf with skipna=True and all values are NaN,
                # SQL returns NULL. Convert to 0 to match pandas behavior.
                if self._agg_func_name == "sum" and self._skipna:
                    result_df[col_name] = result_df[col_name].fillna(0)

                if as_index:
                    # Return Series with groupby keys as index
                    if len(groupby_col_names) == 1:
                        result_series = result_df.set_index(groupby_col_names[0])[
                            col_name
                        ]
                    else:
                        result_series = result_df.set_index(groupby_col_names)[col_name]
                    result_series.name = col_name
                    return result_series
                else:
                    # Return DataFrame with groupby keys as columns
                    return result_df
        finally:
            # Restore original groupby fields
            datastore._groupby_fields = original_groupby

    def _execute_op(self):
        """
        Execute operation descriptor mode - the preferred pattern for lazy operations.

        This evaluates the source expression first, then applies the operation.
        When called directly (not through ExpressionEvaluator), it's safe because
        we evaluate the source first, avoiding recursion.

        Returns:
            The result of the operation (usually pd.Series or pd.DataFrame)
        """
        op_type = self._op_type
        op_source = self._op_source
        op_method = self._op_method
        op_args = self._op_args
        op_kwargs = self._op_kwargs

        # First, evaluate the source expression
        source_result = op_source._execute() if op_source is not None else None

        if source_result is None:
            return None

        if op_type == "accessor":
            # Accessor operation (e.g., str.extract, dt.year, json._extract_array)
            accessor_name = self._op_accessor

            # Special handling for JSON array extraction
            if accessor_name == "json" and op_method == "_extract_array":
                json_path = op_args[0] if op_args else ""
                return _extract_json_array(source_result, json_path)

            accessor = getattr(source_result, accessor_name)
            method = getattr(accessor, op_method)
            return method(*op_args, **op_kwargs)

        elif op_type == "method":
            # Direct method operation (e.g., fillna, replace)
            method = getattr(source_result, op_method)
            return method(*op_args, **op_kwargs)

        elif op_type == "groupby_transform":
            # Groupby transform operation
            groupby_cols = self._op_groupby_cols
            func = getattr(self, "_op_func", None) or op_method
            # Use the datastore's df for groupby context
            df = self._datastore._execute()
            col_name = None
            if op_source is not None and hasattr(op_source, "_expr"):
                from .expressions import Field as ExprField

                if isinstance(op_source._expr, ExprField):
                    col_name = op_source._expr.name
            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_cols)[col_name]
                return grouped.transform(func, *op_args, **op_kwargs)
            return source_result

        elif op_type == "groupby_agg":
            # Groupby aggregation operation
            groupby_cols = self._op_groupby_cols
            func = getattr(self, "_op_func", None) or op_method
            df = self._datastore._execute()
            col_name = None
            if op_source is not None and hasattr(op_source, "_expr"):
                from .expressions import Field as ExprField

                if isinstance(op_source._expr, ExprField):
                    col_name = op_source._expr.name
            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_cols)[col_name]
                agg_method = getattr(grouped, func) if isinstance(func, str) else func
                if callable(agg_method):
                    return agg_method(*op_args, **op_kwargs)
                return agg_method
            return source_result

        elif op_type == "groupby_nth":
            # Groupby nth operation
            groupby_cols = self._op_groupby_cols
            n = op_args[0] if op_args else 0
            df = self._datastore._execute()
            col_name = None
            if op_source is not None and hasattr(op_source, "_expr"):
                from .expressions import Field as ExprField

                if isinstance(op_source._expr, ExprField):
                    col_name = op_source._expr.name
            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_cols, sort=False)[col_name]
                return grouped.nth(n, **op_kwargs)
            return source_result

        else:
            raise ValueError(f"Unknown op_type: {op_type}")

    def _execute_executor(self):
        """Execute executor mode - call the custom executor (DEPRECATED)."""
        return self._executor()

    def _execute_if_needed(self, value: Any) -> Any:
        """Execute ColumnExpr arguments if needed."""
        if isinstance(value, ColumnExpr):
            result = value._execute()
            if isinstance(result, pd.Series) and len(result) == 1:
                return result.iloc[0]
            return result
        return value

    def to_pandas(self) -> pd.Series:
        """
        Execute this expression and return as pandas Series.

        This provides API consistency with Polars, Dask, and other
        DataFrame libraries that use to_pandas() for conversion.

        Returns:
            pd.Series: The executed expression as a pandas Series

        Example:
            >>> ds['age'].to_pandas()
            0    28
            1    31
            2    29
            Name: age, dtype: int64

            >>> (ds['age'] * 2).to_pandas()
            0    56
            1    62
            2    58
            Name: age, dtype: int64
        """
        return self._execute()

    def _evaluate_via_chdb(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate the expression using chDB's Python() table function.

        Note: This is kept for backward compatibility but the preferred path
        is now through ExpressionEvaluator which respects function_config.
        """
        from .executor import get_executor

        executor = get_executor()
        sql_expr = self._expr.to_sql(quote_char='"')
        return executor.execute_expression(sql_expr, df)

    def _aggregate_with_groupby(
        self, agg_func_name: str, pandas_agg_func: str = None, skipna: bool = True
    ) -> pd.Series:
        """
        Execute aggregation with GROUP BY and return a pandas Series.

        When the DataStore has _groupby_fields set, this method executes
        a GROUP BY aggregation query and returns a Series with the groupby
        column(s) as index.

        Respects the execution engine configuration:
        - PANDAS: Use pure pandas groupby
        - CHDB/AUTO: Use chDB for SQL execution

        Args:
            agg_func_name: SQL aggregate function name (e.g., 'avg', 'sum')
            pandas_agg_func: Pandas aggregation method name (e.g., 'mean', 'sum').
                            If None, uses agg_func_name.
            skipna: Whether to skip NaN values (default True, matching pandas behavior)

        Returns:
            pd.Series: Aggregated values with groupby column(s) as index
        """
        from .config import get_execution_engine, ExecutionEngine

        pandas_agg_func = pandas_agg_func or agg_func_name

        # Get the groupby fields from the datastore and clear them before execution
        # This prevents _execute() from generating SQL with GROUP BY
        groupby_fields = self._datastore._groupby_fields.copy()
        self._datastore._groupby_fields = []

        # Get the executed DataFrame from the DataStore (without groupby applied)
        df = self._datastore._execute()

        # Get groupby column names
        groupby_col_names = []
        for gf in groupby_fields:
            if isinstance(gf, Field):
                groupby_col_names.append(gf.name)
            else:
                groupby_col_names.append(gf.to_sql(quote_char='"'))

        # Check execution engine configuration
        engine = get_execution_engine()

        if engine == ExecutionEngine.PANDAS:
            # Use pure pandas groupby
            col_name = self._get_column_name()
            grouped = df.groupby(groupby_col_names, sort=True)
            agg_method = getattr(grouped[col_name], pandas_agg_func)
            result_series = agg_method()
            result_series.name = col_name
        else:
            # Use chDB for CHDB or AUTO
            from .executor import get_executor

            # Get the column name for the aggregation
            col_expr_sql = self._expr.to_sql(quote_char='"')

            # Build SQL query with GROUP BY
            groupby_sql = ", ".join(f'"{name}"' for name in groupby_col_names)

            # Use -If suffix to skip NaN values (matching pandas skipna=True behavior)
            # In chDB, pandas NaN is recognized as isNaN(), not isNull()
            if skipna:
                # Use aggregate function with If suffix to skip NaN values
                agg_sql = (
                    f"{agg_func_name}If({col_expr_sql}, NOT isNaN({col_expr_sql}))"
                )
            else:
                agg_sql = f"{agg_func_name}({col_expr_sql})"

            # Wrap count in toInt64() to match pandas dtype (int64 instead of uint64)
            if agg_func_name == "count":
                agg_sql = f"toInt64({agg_sql})"

            sql = f"SELECT {groupby_sql}, {agg_sql} AS __agg_result__ FROM __df__ GROUP BY {groupby_sql}"

            # Execute via chDB
            executor = get_executor()
            result_df = executor.query_dataframe(sql, df)

            # Convert result to Series with proper index
            if len(groupby_col_names) == 1:
                # Single groupby column - use it as index
                result_series = result_df.set_index(groupby_col_names[0])[
                    "__agg_result__"
                ]
                result_series.name = self._get_column_name()
            else:
                # Multiple groupby columns - use MultiIndex
                result_series = result_df.set_index(groupby_col_names)["__agg_result__"]
                result_series.name = self._get_column_name()

            # Sort by index to match pandas default behavior (sort=True)
            result_series = result_series.sort_index()

        return result_series

    def _get_column_name(self) -> str:
        """Get the column name for this expression."""
        if self._alias:
            return self._alias
        # For method/agg mode, try to get from source first
        if self._source is not None:
            return self._source._get_column_name()
        if isinstance(self._expr, Field):
            return self._expr.name
        if self._expr is None:
            return "unknown"
        return self._expr.to_sql(quote_char='"')

    # ========== Value Comparison Methods ==========

    def _compare_values(self, other, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Compare executed values with another Series/array.

        Args:
            other: Series, list, or array-like to compare with
            rtol: Relative tolerance for float comparison
            atol: Absolute tolerance for float comparison

        Returns:
            bool: True if values are equivalent
        """
        import numpy as np

        self_series = self._execute()

        # Convert other to Series if needed
        if isinstance(other, pd.DataFrame):
            if len(other.columns) == 1:
                other = other.iloc[:, 0]
            else:
                return False
        elif not isinstance(other, pd.Series):
            other = pd.Series(other)

        # Check lengths
        if len(self_series) != len(other):
            return False

        # Reset index for comparison
        s1 = self_series.reset_index(drop=True)
        s2 = other.reset_index(drop=True)

        # Check null positions
        null1 = s1.isna()
        null2 = s2.isna()
        if not null1.equals(null2):
            return False

        # If all null, they're equal
        if null1.all():
            return True

        # Get non-null values
        vals1 = s1[~null1]
        vals2 = s2[~null2]

        # Handle numeric comparison with tolerance
        if np.issubdtype(s1.dtype, np.number) and np.issubdtype(s2.dtype, np.number):
            return np.allclose(
                vals1.values, vals2.values, rtol=rtol, atol=atol, equal_nan=True
            )

        # Handle datetime comparison
        if np.issubdtype(s1.dtype, np.datetime64) or np.issubdtype(
            s2.dtype, np.datetime64
        ):
            try:
                dt1 = pd.to_datetime(vals1, errors="coerce")
                dt2 = pd.to_datetime(vals2, errors="coerce")
                return dt1.equals(dt2)
            except Exception:
                pass

        # String/object comparison
        return list(vals1) == list(vals2)

    def equals(
        self, other, check_names: bool = False, rtol: float = 1e-5, atol: float = 1e-8
    ) -> bool:
        """
        Test whether ColumnExpr contains the same elements as another object.

        This follows pandas Series.equals() semantics: both values AND index must match.
        Use values_equal() if you only want to compare values by position.

        Args:
            other: Series, ColumnExpr, list, or array-like to compare with
            check_names: Whether to check Series names match (default False)
            rtol: Relative tolerance for float comparison (only used with values_equal)
            atol: Absolute tolerance for float comparison (only used with values_equal)

        Returns:
            bool: True if both values and index are equivalent

        Examples:
            >>> ds['name'].str.upper().equals(pd_df['name'].str.upper())
            True
        """
        import numpy as np

        self_series = self._execute()

        # Unwrap ColumnExpr or lazy objects
        if isinstance(other, ColumnExpr):
            other = other._execute()
        elif hasattr(other, "_execute"):
            other = other._execute()

        # Convert to Series if needed
        if isinstance(other, pd.DataFrame):
            if len(other.columns) == 1:
                other = other.iloc[:, 0]
            else:
                return False
        elif not isinstance(other, pd.Series):
            other = pd.Series(other)

        # Use pandas equals() which considers both values and index
        result = self_series.equals(other)

        # If pandas equals returns True and check_names is required, verify names
        if result and check_names:
            if self_series.name != other.name:
                return False

        return result

    def values_equal(self, other, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """
        Test whether ColumnExpr contains the same values as another object (ignoring index).

        This compares values by position only, ignoring index labels.
        Use equals() if you want to check both values and index.

        Args:
            other: Series, ColumnExpr, list, or array-like to compare with
            rtol: Relative tolerance for float comparison
            atol: Absolute tolerance for float comparison

        Returns:
            bool: True if values are equivalent by position

        Examples:
            >>> ds['name'].str.upper().values_equal(pd_df['name'].str.upper())
            True
        """
        # Unwrap ColumnExpr or lazy objects
        if isinstance(other, ColumnExpr):
            other = other._execute()
        elif hasattr(other, "_execute"):
            other = other._execute()

        return self._compare_values(other, rtol=rtol, atol=atol)

    # ========== Display Methods ==========

    def __repr__(self) -> str:
        """Return a representation that shows actual values."""
        try:
            series = self._execute()
            return repr(series)
        except Exception as e:
            return f"ColumnExpr({self._expr!r}) [Error: {e}]"

    def __str__(self) -> str:
        """Return string representation showing actual values."""
        try:
            series = self._execute()
            return str(series)
        except Exception:
            return self._expr.to_sql()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            series = self._execute()
            if hasattr(series, "_repr_html_"):
                return series._repr_html_()
            return f"<pre>{repr(series)}</pre>"
        except Exception as e:
            return f"<pre>ColumnExpr({self._expr.to_sql()}) [Error: {e}]</pre>"

    # ========== Numeric Conversion Methods ==========

    def __float__(self) -> float:
        """Convert to float (executes the expression)."""
        result = self._execute()
        if isinstance(result, pd.Series):
            if len(result) == 1:
                return float(result.iloc[0])
            raise TypeError(f"Cannot convert Series of length {len(result)} to float")
        return float(result)

    def __int__(self) -> int:
        """Convert to int (executes the expression)."""
        result = self._execute()
        if isinstance(result, pd.Series):
            if len(result) == 1:
                return int(result.iloc[0])
            raise TypeError(f"Cannot convert Series of length {len(result)} to int")
        return int(result)

    def __round__(self, ndigits: int = None):
        """Round the result (executes the expression)."""
        result = self._execute()
        if isinstance(result, pd.Series):
            if len(result) == 1:
                return round(result.iloc[0], ndigits)
            return result.round(ndigits)
        return round(result, ndigits)

    # ========== Expression Interface (delegation) ==========

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for the underlying expression."""
        return self._expr.to_sql(quote_char=quote_char, **kwargs)

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield from self._expr.nodes()

    # ========== Comparison Operators (Return ColumnExpr wrapping Condition) ==========
    #
    # Design Note: Comparison operators return ColumnExpr (not a separate BoolColumnExpr).
    # The underlying Expression is a Condition, which:
    # 1. Can be executed to boolean Series via ExpressionEvaluator
    # 2. Can be used in filter() for SQL WHERE clause generation
    # 3. Supports value_counts(), sum(), mean() etc. (inherited from ColumnExpr)
    #
    # This unified design allows:
    # - (ds['col'] > 5).value_counts()  # Works
    # - ds.filter(ds['col'] > 5)        # Works
    # - (ds['a'] > 0) & (ds['b'] < 10)  # Works

    def __eq__(self, other: Any):
        """
        Compare ColumnExpr with another value.

        If other is a pandas Series/DataFrame (already executed data),
        performs value comparison and returns bool.

        For aggregation mode (scalar result), directly execute and compare.

        Otherwise, returns a ColumnExpr wrapping a Condition that can both:
        - Execute to a boolean Series (for value_counts(), sum(), etc.)
        - Be used as a filter condition in ds.filter()

        Examples:
            # Value comparison with executed data (returns bool)
            >>> ds['name'].str.upper() == pd_df['name'].str.upper()  # True/False

            # Aggregation comparison (returns bool)
            >>> ds['col'].nunique() == 2  # True/False

            # ColumnExpr wrapping Condition (supports both operations)
            >>> bool_col = ds['age'] == 30  # Returns ColumnExpr
            >>> bool_col.value_counts()  # Counts True/False
            >>> ds.filter(bool_col)  # Filters rows
        """
        from .conditions import BinaryCondition

        # Value comparison only with pandas Series/DataFrame (executed data)
        if isinstance(other, pd.Series):
            return self._compare_values(other)
        if isinstance(other, pd.DataFrame):
            return self._compare_values(other)

        # For aggregation mode or method mode, use pandas comparison on executed result
        # This handles cases like: groupby_result == 20, cumsum_result == 10
        if self._exec_mode in ("agg", "method", "executor"):
            result = self._execute()
            # If result is scalar, return boolean comparison directly
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result == other
            # If result is a single-element Series, compare the scalar
            if isinstance(result, pd.Series) and len(result) == 1:
                return result.iloc[0] == other
            # For multi-element Series, return method-mode ColumnExpr for boolean indexing
            return ColumnExpr(source=self, method_name="__eq__", method_args=(other,))

        # For expression mode, check if _expr exists
        if self._exec_mode == "expr" and self._expr is None:
            # Fallback: execute and compare
            result = self._execute()
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result == other

        # NOTE: Do NOT add cross-DataStore handling for __eq__ here!
        # __eq__ is commonly used for join conditions (e.g., ds1.id == ds2.user_id)
        # which need to remain as BinaryCondition for SQL generation.
        # Cross-DataStore value comparison for == should use .eq() method instead.

        # Return ColumnExpr wrapping Condition for lazy boolean column
        # pandas semantics: col == None returns False for ALL rows
        # This is element-wise comparison with Python singleton None,
        # NOT a check for NA values (use .isna() for that)
        if other is None:
            from .expressions import Literal

            condition = BinaryCondition("=", Literal(0), Literal(1))  # Always False
        else:
            condition = BinaryCondition("=", self._expr, Expression.wrap(other))
        return self._derive_expr(condition)

    def __ne__(self, other: Any) -> "ColumnExpr":
        # For aggregation/method mode, use pandas comparison on executed result
        if self._exec_mode in ("agg", "method", "executor"):
            result = self._execute()
            # If result is scalar, return boolean comparison directly
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result != other
            # If result is a single-element Series, compare the scalar
            if isinstance(result, pd.Series) and len(result) == 1:
                return result.iloc[0] != other
            # For multi-element Series, return method-mode ColumnExpr
            return ColumnExpr(source=self, method_name="__ne__", method_args=(other,))

        # NOTE: Do NOT add cross-DataStore handling for __ne__ here!
        # Similar to __eq__, != may be used in join conditions.
        # Use .ne() method for cross-DataStore value comparison.

        from .conditions import BinaryCondition
        from .expressions import Literal

        # pandas semantics: col != None returns True for ALL rows
        # This is element-wise comparison with Python singleton None,
        # NOT a check for non-NA values (use .notna() for that)
        if other is None:
            condition = BinaryCondition("=", Literal(1), Literal(1))  # Always True
        else:
            condition = BinaryCondition("!=", self._expr, Expression.wrap(other))
        return self._derive_expr(condition)

    def __gt__(self, other: Any) -> "ColumnExpr":
        # For aggregation/method mode, use pandas comparison on executed result
        if self._exec_mode in ("agg", "method", "executor"):
            result = self._execute()
            # For scalar results, return bool directly
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result > other
            # For Series results, return method-mode ColumnExpr for boolean indexing
            return ColumnExpr(source=self, method_name="__gt__", method_args=(other,))

        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op(">", other, is_comparison=True)

        from .conditions import BinaryCondition

        condition = BinaryCondition(">", self._expr, Expression.wrap(other))
        return self._derive_expr(condition)

    def __ge__(self, other: Any) -> "ColumnExpr":
        # For aggregation/method mode, use pandas comparison on executed result
        if self._exec_mode in ("agg", "method", "executor"):
            result = self._execute()
            # For scalar results, return bool directly
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result >= other
            # For Series results, return method-mode ColumnExpr for boolean indexing
            return ColumnExpr(source=self, method_name="__ge__", method_args=(other,))

        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op(">=", other, is_comparison=True)

        from .conditions import BinaryCondition

        condition = BinaryCondition(">=", self._expr, Expression.wrap(other))
        return self._derive_expr(condition)

    def __lt__(self, other: Any) -> "ColumnExpr":
        # For aggregation/method mode, use pandas comparison on executed result
        if self._exec_mode in ("agg", "method", "executor"):
            result = self._execute()
            # For scalar results, return bool directly
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result < other
            # For Series results, return method-mode ColumnExpr for boolean indexing
            return ColumnExpr(source=self, method_name="__lt__", method_args=(other,))

        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("<", other, is_comparison=True)

        from .conditions import BinaryCondition

        condition = BinaryCondition("<", self._expr, Expression.wrap(other))
        return self._derive_expr(condition)

    def __le__(self, other: Any) -> "ColumnExpr":
        # For aggregation/method mode, use pandas comparison on executed result
        if self._exec_mode in ("agg", "method", "executor"):
            result = self._execute()
            # For scalar results, return bool directly
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                return result <= other
            # For Series results, return method-mode ColumnExpr for boolean indexing
            return ColumnExpr(source=self, method_name="__le__", method_args=(other,))

        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("<=", other, is_comparison=True)

        from .conditions import BinaryCondition

        condition = BinaryCondition("<=", self._expr, Expression.wrap(other))
        return self._derive_expr(condition)

    # ========== Pandas-style Comparison Methods ==========

    def eq(self, other: Any) -> "ColumnExpr":
        """
        Element-wise equality comparison, returns boolean Series (lazy).

        Unlike __eq__ which returns a ColumnExpr for filtering,
        this method returns a lazy wrapper that produces a pandas boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series indicating equality

        Example:
            >>> ds['value'].eq(5)
            0    False
            1     True
            2    False
            dtype: bool
        """
        return ColumnExpr(source=self, method_name="eq", method_args=(other,))

    def ne(self, other: Any) -> "ColumnExpr":
        """
        Element-wise not-equal comparison, returns boolean Series (lazy).

        Args:
            other: Value or Series to compare with

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series indicating inequality

        Example:
            >>> ds['value'].ne(5)
        """
        return ColumnExpr(source=self, method_name="ne", method_args=(other,))

    def lt(self, other: Any) -> "ColumnExpr":
        """
        Element-wise less-than comparison, returns boolean Series (lazy).

        Args:
            other: Value or Series to compare with

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series

        Example:
            >>> ds['value'].lt(5)
        """
        return ColumnExpr(source=self, method_name="lt", method_args=(other,))

    def le(self, other: Any) -> "ColumnExpr":
        """
        Element-wise less-than-or-equal comparison, returns boolean Series (lazy).

        Args:
            other: Value or Series to compare with

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series

        Example:
            >>> ds['value'].le(5)
        """
        return ColumnExpr(source=self, method_name="le", method_args=(other,))

    def gt(self, other: Any) -> "ColumnExpr":
        """
        Element-wise greater-than comparison, returns boolean Series (lazy).

        Args:
            other: Value or Series to compare with

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series

        Example:
            >>> ds['value'].gt(5)
        """
        return ColumnExpr(source=self, method_name="gt", method_args=(other,))

    def ge(self, other: Any) -> "ColumnExpr":
        """
        Element-wise greater-than-or-equal comparison, returns boolean Series (lazy).

        Args:
            other: Value or Series to compare with

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series

        Example:
            >>> ds['value'].ge(5)
        """
        return ColumnExpr(source=self, method_name="ge", method_args=(other,))

    # ========== Logical Operators (Return ColumnExpr wrapping Condition) ==========
    #
    # These operators combine boolean expressions and return ColumnExpr.
    # If the underlying _expr is already a Condition, use it directly;
    # otherwise convert to a truthy check (expr = 1).

    def _to_condition(self):
        """Convert this ColumnExpr to a Condition for logical operations."""
        from .conditions import Condition, BinaryCondition
        from .expressions import Literal

        if isinstance(self._expr, Condition):
            return self._expr
        else:
            # Non-condition expression, convert to truthy check: expr = 1
            return BinaryCondition("=", self._expr, Literal(1))

    def __and__(self, other: Any) -> "ColumnExpr":
        """
        Combine with AND operator.

        Returns a ColumnExpr wrapping CompoundCondition('AND', ...).
        This allows chaining: (ds['a'] > 0) & (ds['b'] < 10)

        Example:
            >>> ds.filter((ds['age'] > 18) & (ds['status'] == 'active'))
            >>> ((ds['a'] > 0) & (ds['b'] < 10)).value_counts()
        """
        from .conditions import CompoundCondition, Condition

        # For method mode (e.g., from groupby comparison), use pandas & on executed results
        import numpy as np
        import pandas as pd

        # Handle numpy arrays and pandas Series by deferring to method mode
        if isinstance(other, (np.ndarray, pd.Series)):
            return ColumnExpr(source=self, method_name="__and__", method_args=(other,))

        # For executor mode (e.g., cross-DataStore comparison result), use pandas & on executed results
        if self._exec_mode in ("method", "executor") or (
            isinstance(other, ColumnExpr) and other._exec_mode in ("method", "executor")
        ):
            return ColumnExpr(source=self, method_name="__and__", method_args=(other,))

        self_cond = self._to_condition()

        # Handle other operand
        if isinstance(other, ColumnExpr):
            other_cond = other._to_condition()
        elif isinstance(other, LazyCondition):
            other_cond = other.condition
        elif isinstance(other, Condition):
            other_cond = other
        else:
            raise TypeError(f"Cannot AND ColumnExpr with {type(other).__name__}")

        return self._derive_expr(CompoundCondition("AND", self_cond, other_cond))

    def __rand__(self, other: Any) -> "ColumnExpr":
        """Right AND operator."""
        from .conditions import CompoundCondition, Condition
        import numpy as np
        import pandas as pd

        # Handle numpy arrays and pandas Series by deferring to method mode
        if isinstance(other, (np.ndarray, pd.Series)):
            return ColumnExpr(source=self, method_name="__rand__", method_args=(other,))

        self_cond = self._to_condition()

        if isinstance(other, LazyCondition):
            return self._derive_expr(
                CompoundCondition("AND", other.condition, self_cond)
            )
        elif isinstance(other, Condition):
            return self._derive_expr(CompoundCondition("AND", other, self_cond))
        else:
            raise TypeError(f"Cannot AND {type(other).__name__} with ColumnExpr")

    def __or__(self, other: Any) -> "ColumnExpr":
        """
        Combine with OR operator.

        Returns a ColumnExpr wrapping CompoundCondition('OR', ...).

        Example:
            >>> ds.filter((ds['age'] < 18) | (ds['age'] > 65))
        """
        from .conditions import CompoundCondition, Condition

        import numpy as np
        import pandas as pd

        # Handle numpy arrays and pandas Series by deferring to method mode
        if isinstance(other, (np.ndarray, pd.Series)):
            return ColumnExpr(source=self, method_name="__or__", method_args=(other,))

        # For executor/method mode (e.g., cross-DataStore comparison result), use pandas | on executed results
        if self._exec_mode in ("method", "executor") or (
            isinstance(other, ColumnExpr) and other._exec_mode in ("method", "executor")
        ):
            return ColumnExpr(source=self, method_name="__or__", method_args=(other,))

        self_cond = self._to_condition()

        # Handle other operand
        if isinstance(other, ColumnExpr):
            other_cond = other._to_condition()
        elif isinstance(other, LazyCondition):
            other_cond = other.condition
        elif isinstance(other, Condition):
            other_cond = other
        else:
            raise TypeError(f"Cannot OR ColumnExpr with {type(other).__name__}")

        return self._derive_expr(CompoundCondition("OR", self_cond, other_cond))

    def __ror__(self, other: Any) -> "ColumnExpr":
        """Right OR operator."""
        from .conditions import CompoundCondition, Condition
        import numpy as np
        import pandas as pd

        # Handle numpy arrays and pandas Series by deferring to method mode
        if isinstance(other, (np.ndarray, pd.Series)):
            return ColumnExpr(source=self, method_name="__ror__", method_args=(other,))

        self_cond = self._to_condition()

        if isinstance(other, LazyCondition):
            return self._derive_expr(
                CompoundCondition("OR", other.condition, self_cond)
            )
        elif isinstance(other, Condition):
            return self._derive_expr(CompoundCondition("OR", other, self_cond))
        else:
            raise TypeError(f"Cannot OR {type(other).__name__} with ColumnExpr")

    def __xor__(self, other: Any) -> "ColumnExpr":
        """
        Combine with XOR operator.

        Returns a ColumnExpr wrapping CompoundCondition('XOR', ...).
        """
        from .conditions import CompoundCondition, Condition
        import numpy as np
        import pandas as pd

        # Handle numpy arrays and pandas Series by deferring to method mode
        if isinstance(other, (np.ndarray, pd.Series)):
            return ColumnExpr(source=self, method_name="__xor__", method_args=(other,))

        self_cond = self._to_condition()

        if isinstance(other, ColumnExpr):
            other_cond = other._to_condition()
        elif isinstance(other, Condition):
            other_cond = other
        else:
            raise TypeError(f"Cannot XOR ColumnExpr with {type(other).__name__}")

        return self._derive_expr(CompoundCondition("XOR", self_cond, other_cond))

    def __invert__(self) -> "ColumnExpr":
        """
        Negate with NOT operator.

        Returns a ColumnExpr wrapping NotCondition.

        Example:
            >>> ds.filter(~(ds['age'] > 18))  # age <= 18
            >>> (~(ds['col'] > 5)).value_counts()
        """
        from .conditions import NotCondition

        self_cond = self._to_condition()
        return self._derive_expr(NotCondition(self_cond))

    # ========== Arithmetic Operators (Return ColumnExpr) ==========

    def __add__(self, other: Any) -> "ColumnExpr":
        # For non-expression modes, use method call mode
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__add__", method_args=(other,))

        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("+", other)

        # Check if this is string concatenation
        other_is_string = isinstance(other, str)
        other_is_string_col = (
            isinstance(other, ColumnExpr) and other._is_string_column()
        )
        self_is_string = self._is_string_column()

        new_expr = ArithmeticExpression("+", self._expr, Expression.wrap(other))

        # Mark expression as string type if either operand is a string
        if self_is_string or other_is_string or other_is_string_col:
            self._mark_string_expression(new_expr)

        return self._derive_expr(new_expr)

    def __radd__(self, other: Any) -> "ColumnExpr":
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__radd__", method_args=(other,))

        # Check if this is string concatenation
        other_is_string = isinstance(other, str)
        self_is_string = self._is_string_column()

        new_expr = ArithmeticExpression("+", Expression.wrap(other), self._expr)

        # Mark expression as string type if either operand is a string
        if self_is_string or other_is_string:
            self._mark_string_expression(new_expr)

        return self._derive_expr(new_expr)

    def _is_method_mode_columnexpr(self, value: Any) -> bool:
        """Check if value is a ColumnExpr in method mode (with _expr=None)."""
        return isinstance(value, ColumnExpr) and (
            value._exec_mode != "expr" or value._expr is None
        )

    def _is_cross_datastore(self, other: Any) -> bool:
        """
        Check if other is a ColumnExpr from a different DataStore.

        This is used to detect cross-DataStore operations that need
        special handling (row alignment via _row_id).

        Args:
            other: The other operand in an arithmetic/comparison operation

        Returns:
            True if other is a ColumnExpr from a different DataStore
        """
        if not isinstance(other, ColumnExpr):
            return False
        if other._datastore is None or self._datastore is None:
            return False
        return other._datastore is not self._datastore

    def _create_cross_datastore_op(
        self, op: str, other: "ColumnExpr", is_comparison: bool = False
    ) -> "ColumnExpr":
        """
        Create a ColumnExpr for cross-DataStore operations.

        Uses _row_id for row alignment between two DataStores.
        The operation is executed by materializing both sides and
        aligning by position (similar to pandas index alignment).

        The execution path is controlled by config.cross_datastore_engine:
        - 'auto': Execute each side with its optimal engine, then combine with pandas
        - 'pandas': Use pure pandas for entire operation (fastest for simple ops)
        - 'chdb': Force chDB execution for each side (not recommended)

        Args:
            op: The operator ('+', '-', '*', '/', '>', '<', '==', etc.)
            other: The ColumnExpr from another DataStore
            is_comparison: If True, the result is a boolean Series

        Returns:
            ColumnExpr in executor mode that performs the aligned operation
        """
        from .config import get_cross_datastore_engine, ExecutionEngine

        # Capture references for the closure
        self_expr = self
        other_expr = other
        cross_engine = get_cross_datastore_engine()

        def executor():
            import pandas as pd

            def get_series(expr: "ColumnExpr") -> pd.Series:
                """Get Series from ColumnExpr, using pandas path if configured and possible."""
                if cross_engine == ExecutionEngine.PANDAS:
                    # Try pure pandas path for simple Field expressions
                    if isinstance(expr._expr, Field) and expr._datastore is not None:
                        col_name = expr._expr.name
                        df = expr._datastore._get_df()
                        if col_name in df.columns:
                            return df[col_name]
                # Fall back to execute (may use SQL for complex expressions)
                return expr._execute()

            # Get Series based on configured engine
            s1 = get_series(self_expr)
            s2 = get_series(other_expr)

            # Ensure both are Series
            if not isinstance(s1, pd.Series):
                s1 = pd.Series([s1])
            if not isinstance(s2, pd.Series):
                s2 = pd.Series([s2])

            # Align by position using reset_index only if necessary
            # Skip reset_index if both Series have compatible indices (same length RangeIndex)
            # This optimization avoids expensive index operations when not needed
            s1_idx = s1.index
            s2_idx = s2.index
            need_reset = True

            if len(s1) == len(s2):
                # Check if both have default RangeIndex (0, 1, 2, ...)
                if (
                    isinstance(s1_idx, pd.RangeIndex)
                    and isinstance(s2_idx, pd.RangeIndex)
                    and s1_idx.start == 0
                    and s2_idx.start == 0
                    and s1_idx.step == 1
                    and s2_idx.step == 1
                ):
                    need_reset = False
                # Or if indices are already equal
                elif s1_idx.equals(s2_idx):
                    need_reset = False

            if need_reset:
                s1_aligned = s1.reset_index(drop=True)
                s2_aligned = s2.reset_index(drop=True)
            else:
                s1_aligned = s1
                s2_aligned = s2

            # Perform the operation
            op_map = {
                "+": "__add__",
                "-": "__sub__",
                "*": "__mul__",
                "/": "__truediv__",
                "//": "__floordiv__",
                "%": "__mod__",
                "**": "__pow__",
                ">": "__gt__",
                ">=": "__ge__",
                "<": "__lt__",
                "<=": "__le__",
                "==": "__eq__",
                "!=": "__ne__",
            }
            method_name = op_map.get(op, f"__{op}__")
            result = getattr(s1_aligned, method_name)(s2_aligned)

            # Restore original index from self (left operand)
            if len(result) == len(s1):
                result.index = s1.index

            return result

        return ColumnExpr(executor=executor, datastore=self._datastore)

    def _is_string_column(self) -> bool:
        """
        Check if this ColumnExpr represents a string column based on DataStore schema.

        Returns True if the column type is a string type (object, string, str, etc.)
        """
        if self._datastore is None:
            return False

        # Get schema from DataStore
        schema = getattr(self._datastore, "_schema", None)
        if not schema:
            return False

        # Get column name from expression
        if isinstance(self._expr, Field):
            col_name = self._expr.name
            if col_name in schema:
                col_type = str(schema[col_name]).lower()
                # Check for string types (pandas 'object', 'string', ClickHouse 'String', etc.)
                string_indicators = (
                    "object",
                    "string",
                    "str",
                    "fixedstring",
                    "enum",
                    "uuid",
                )
                return any(s in col_type for s in string_indicators)

        return False

    def _mark_string_expression(self, expr: Expression) -> Expression:
        """
        Mark an expression as involving string types.

        This helps ArithmeticExpression.to_sql() detect that '+' should become concat().
        """
        expr._is_string_type = True
        return expr

    def __sub__(self, other: Any) -> "ColumnExpr":
        # If self or other is method mode, use pandas arithmetic
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(source=self, method_name="__sub__", method_args=(other,))
        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("-", other)
        new_expr = ArithmeticExpression("-", self._expr, Expression.wrap(other))
        return self._derive_expr(new_expr)

    def __rsub__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(source=self, method_name="__rsub__", method_args=(other,))
        new_expr = ArithmeticExpression("-", Expression.wrap(other), self._expr)
        return self._derive_expr(new_expr)

    def __mul__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(source=self, method_name="__mul__", method_args=(other,))
        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("*", other)
        new_expr = ArithmeticExpression("*", self._expr, Expression.wrap(other))
        return self._derive_expr(new_expr)

    def __rmul__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(source=self, method_name="__rmul__", method_args=(other,))
        new_expr = ArithmeticExpression("*", Expression.wrap(other), self._expr)
        return self._derive_expr(new_expr)

    def __truediv__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(
                source=self, method_name="__truediv__", method_args=(other,)
            )
        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("/", other)
        new_expr = ArithmeticExpression("/", self._expr, Expression.wrap(other))
        return self._derive_expr(new_expr)

    def __rtruediv__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(
                source=self, method_name="__rtruediv__", method_args=(other,)
            )
        new_expr = ArithmeticExpression("/", Expression.wrap(other), self._expr)
        return self._derive_expr(new_expr)

    def __floordiv__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(
                source=self, method_name="__floordiv__", method_args=(other,)
            )
        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("//", other)
        new_expr = ArithmeticExpression("//", self._expr, Expression.wrap(other))
        return self._derive_expr(new_expr)

    def __rfloordiv__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(
                source=self, method_name="__rfloordiv__", method_args=(other,)
            )
        new_expr = ArithmeticExpression("//", Expression.wrap(other), self._expr)
        return self._derive_expr(new_expr)

    def __mod__(self, other: Any) -> "ColumnExpr":
        if (
            self._exec_mode != "expr"
            or self._expr is None
            or self._is_method_mode_columnexpr(other)
        ):
            return ColumnExpr(source=self, method_name="__mod__", method_args=(other,))
        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("%", other)
        new_expr = ArithmeticExpression("%", self._expr, Expression.wrap(other))
        return self._derive_expr(new_expr)

    def __rmod__(self, other: Any) -> "ColumnExpr":
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__rmod__", method_args=(other,))
        new_expr = ArithmeticExpression("%", Expression.wrap(other), self._expr)
        return self._derive_expr(new_expr)

    def __pow__(self, other: Any) -> "ColumnExpr":
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__pow__", method_args=(other,))
        # Cross-DataStore operation: align by _row_id (position)
        if self._is_cross_datastore(other):
            return self._create_cross_datastore_op("**", other)
        new_expr = ArithmeticExpression("**", self._expr, Expression.wrap(other))
        return self._derive_expr(new_expr)

    def __rpow__(self, other: Any) -> "ColumnExpr":
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__rpow__", method_args=(other,))
        new_expr = ArithmeticExpression("**", Expression.wrap(other), self._expr)
        return self._derive_expr(new_expr)

    def __neg__(self) -> "ColumnExpr":
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__neg__")
        new_expr = ArithmeticExpression("-", Literal(0), self._expr)
        return self._derive_expr(new_expr)

    def __pos__(self) -> "ColumnExpr":
        """
        Unary positive operator (+column).

        Returns the column unchanged (identity operation), same as pandas Series behavior.
        """
        if self._exec_mode != "expr" or self._expr is None:
            return ColumnExpr(source=self, method_name="__pos__")
        # Unary + is identity - return the same expression
        return self._derive_expr(self._expr)

    def __round__(self, ndigits: Optional[int] = None) -> "ColumnExpr":
        """
        Support Python's built-in round() function.

        Args:
            ndigits: Number of decimal places to round to (default 0)

        Returns:
            ColumnExpr: SQL expression for round(column, ndigits)

        Example:
            >>> round(ds['value'].mean(), 2)
        """
        from .functions import Function

        if ndigits is None:
            return self._derive_expr(Function("round", self._expr))
        return self._derive_expr(Function("round", self._expr, Literal(ndigits)))

    # ========== Accessor Properties ==========

    @property
    def str(self) -> "ColumnExprStringAccessor":
        """Accessor for string functions."""
        return ColumnExprStringAccessor(self)

    @property
    def dt(self) -> "ColumnExprDateTimeAccessor":
        """Accessor for date/time functions."""
        return ColumnExprDateTimeAccessor(self)

    @property
    def json(self) -> "ColumnExprJsonAccessor":
        """Accessor for JSON functions."""
        return ColumnExprJsonAccessor(self)

    @property
    def arr(self) -> "ColumnExprArrayAccessor":
        """Accessor for array functions."""
        return ColumnExprArrayAccessor(self)

    @property
    def url(self) -> "ColumnExprUrlAccessor":
        """Accessor for URL functions."""
        return ColumnExprUrlAccessor(self)

    @property
    def ip(self) -> "ColumnExprIpAccessor":
        """Accessor for IP address functions."""
        return ColumnExprIpAccessor(self)

    @property
    def geo(self) -> "ColumnExprGeoAccessor":
        """Accessor for geo/distance functions."""
        return ColumnExprGeoAccessor(self)

    @property
    def iloc(self) -> "ColumnExprILocIndexer":
        """
        Integer-location based indexing for selection by position.

        Executes the column and returns an indexer that supports integer-based
        selection like pandas Series.iloc.

        Examples:
            >>> ds['col'].iloc[0]      # Get first element
            >>> ds['col'].iloc[-1]     # Get last element
            >>> ds['col'].iloc[1:5]    # Get elements 1-4
            >>> ds['col'].iloc[[0, 2]] # Get elements at positions 0 and 2

        Returns:
            ColumnExprILocIndexer: Indexer for integer-location based selection
        """
        return ColumnExprILocIndexer(self)

    @property
    def loc(self) -> "ColumnExprLocIndexer":
        """
        Label-based indexing for selection by label.

        Executes the column and returns an indexer that supports label-based
        selection like pandas Series.loc.

        Examples:
            >>> ds['col'].loc[0]           # Get element with label 0
            >>> ds['col'].loc['key']       # Get element with label 'key'
            >>> ds['col'].loc[0:5]         # Get elements with labels 0-5

        Returns:
            ColumnExprLocIndexer: Indexer for label-based selection
        """
        return ColumnExprLocIndexer(self)

    @property
    def plot(self):
        """
        Accessor for pandas plotting functions.

        Executes the column and returns the pandas Series plot accessor.
        Supports all pandas Series plotting methods like .plot(), .plot.bar(),
        .plot.hist(), etc.

        Example:
            >>> ds['age'].plot(kind='hist')
            >>> ds['value'].plot.line()
            >>> ds['category'].plot(title='My Plot', figsize=(10, 6))

        Returns:
            pandas.plotting.PlotAccessor: The plot accessor from the executed Series
        """
        return self._execute().plot

    @property
    def cat(self):
        """
        Accessor for categorical data methods.

        Executes the column and returns the pandas Series categorical accessor.
        Only works if the underlying data is categorical.

        Example:
            >>> ds['category'].cat.categories
            Index(['A', 'B', 'C'], dtype='object')
            >>> ds['category'].cat.codes
            0    0
            1    1
            2    2
            dtype: int8
            >>> ds['category'].cat.ordered
            False

        Returns:
            pandas.core.arrays.categorical.CategoricalAccessor: The cat accessor

        Raises:
            AttributeError: If the data is not categorical
        """
        return self._execute().cat

    @property
    def sparse(self):
        """
        Accessor for sparse data methods.

        Executes the column and returns the pandas Series sparse accessor.
        Only works if the underlying data is sparse.

        Example:
            >>> sparse_series = pd.arrays.SparseArray([0, 0, 1, 0, 2])
            >>> ds['sparse_col'].sparse.density
            0.4
            >>> ds['sparse_col'].sparse.fill_value
            0

        Returns:
            pandas.core.arrays.sparse.accessor.SparseAccessor: The sparse accessor

        Raises:
            AttributeError: If the data is not sparse
        """
        return self._execute().sparse

    # ========== Condition Methods (for filtering) ==========

    def isnull(self) -> "ColumnExpr":
        """
        Detect missing values (NULL/NaN), returns a boolean-like ColumnExpr.

        Uses isNull function. The execution engine (chDB or Pandas) is determined
        by function_config - by default uses Pandas for proper NaN handling.

        This can be used:
        1. For filtering: ds.filter(ds['email'].isnull())
        2. For value computation: ds['value'].isnull()
        3. For column assignment: ds['is_null'] = ds['value'].isnull()

        Example:
            >>> ds.filter(ds['email'].isnull())
            >>> ds['value'].isnull()  # Returns boolean Series when executed

        Returns:
            ColumnExpr: Expression that evaluates to True (NULL/NaN) or False
        """
        from .functions import Function

        # Wrap with toBool() to return bool dtype instead of uint8
        # This ensures pandas compatibility (pandas isna() returns bool)
        return self._derive_expr(Function("toBool", Function("isNull", self._expr)))

    def notnull(self) -> "ColumnExpr":
        """
        Detect non-missing values (not NULL/NaN), returns a boolean-like ColumnExpr.

        Uses isNotNull function. The execution engine (chDB or Pandas) is determined
        by function_config - by default uses Pandas for proper NaN handling.

        This can be used:
        1. For filtering: ds.filter(ds['email'].notnull())
        2. For value computation: ds['value'].notnull()
        3. For column assignment: ds['is_not_null'] = ds['value'].notnull()

        Example:
            >>> ds.filter(ds['email'].notnull())
            >>> ds['value'].notnull()  # Returns boolean Series when executed

        Returns:
            ColumnExpr: Expression that evaluates to True (not NULL/NaN) or False
        """
        from .functions import Function

        # Wrap with toBool() to return bool dtype instead of uint8
        # This ensures pandas compatibility (pandas notna() returns bool)
        return self._derive_expr(Function("toBool", Function("isNotNull", self._expr)))

    # Aliases for pandas compatibility
    def isna(self) -> "ColumnExpr":
        """Alias for isnull() - pandas compatibility."""
        return self.isnull()

    def notna(self) -> "ColumnExpr":
        """Alias for notnull() - pandas compatibility."""
        return self.notnull()

    def isnull_condition(self) -> "Condition":
        """Create IS NULL condition for filtering (SQL style)."""
        return self._expr.isnull()

    def notnull_condition(self) -> "Condition":
        """Create IS NOT NULL condition for filtering (SQL style)."""
        return self._expr.notnull()

    def isin(self, values) -> "LazyCondition":
        """
        Check if values are contained in a list.

        Returns a LazyCondition that can be used both for:
        1. SQL-style filtering: ds.filter(ds['col'].isin([1, 2]))
        2. Pandas-style boolean Series: ds['col'].isin([1, 2]).to_pandas()

        Args:
            values: List of values to check membership against

        Returns:
            LazyCondition: Condition wrapper supporting both SQL and pandas modes

        Example:
            >>> ds['category'].isin(['A', 'B'])  # Returns LazyCondition
            >>> ds.filter(ds['category'].isin(['A', 'B']))  # SQL-style
            >>> ds['category'].isin(['A', 'B']).to_pandas()  # Boolean Series
        """
        return LazyCondition(self._expr.isin(values), self._datastore)

    def notin(self, values) -> "LazyCondition":
        """Create NOT IN condition."""
        return LazyCondition(self._expr.notin(values), self._datastore)

    def between(self, left, right, inclusive: str = "both") -> "LazyCondition":
        """
        Check if values are between lower and upper bounds.

        Returns a LazyCondition that can be used both for:
        1. SQL-style filtering: ds.filter(ds['col'].between(10, 30))
        2. Pandas-style boolean Series: ds['col'].between(10, 30).to_pandas()

        Args:
            left: Lower bound
            right: Upper bound
            inclusive: Include boundaries. Valid options are:
                'both' (default): Include both boundaries
                'neither': Exclude both boundaries
                'left': Include left boundary only
                'right': Include right boundary only

        Returns:
            LazyCondition: Condition wrapper supporting both SQL and pandas modes

        Example:
            >>> ds['value'].between(10, 30)  # Returns LazyCondition
            >>> ds.filter(ds['value'].between(10, 30))  # SQL-style
            >>> ds['value'].between(10, 30).to_pandas()  # Boolean Series
            >>> ds['value'].between(10, 30, inclusive='neither')  # 10 < x < 30
        """
        return LazyCondition(
            self._expr.between(left, right, inclusive=inclusive), self._datastore
        )

    def like(self, pattern: str) -> "Condition":
        """Create LIKE condition."""
        return self._expr.like(pattern)

    def notlike(self, pattern: str) -> "Condition":
        """Create NOT LIKE condition."""
        return self._expr.notlike(pattern)

    def ilike(self, pattern: str) -> "Condition":
        """Create ILIKE condition (case-insensitive)."""
        return self._expr.ilike(pattern)

    # ========== Alias Support ==========

    @immutable
    def as_(self, alias: str) -> "ColumnExpr":
        """Set an alias for this expression."""
        new_expr = self._expr.as_(alias)
        return self._derive_expr(new_expr, alias=alias)

    # ========== Pandas Series Methods (for compatibility) ==========

    @property
    def values(self) -> Any:
        """Return underlying numpy array (property for pandas compatibility)."""
        import numpy as np

        result = self._execute()
        # Handle case where _execute() returns numpy array directly (e.g., unique())
        if isinstance(result, np.ndarray):
            return result
        return result.values

    @property
    def name(self) -> Optional[str]:
        """Return the name of the column (triggers execution if not a simple Field)."""
        if isinstance(self._expr, Field):
            return self._expr.name
        # For complex expressions, execute and get the result's name
        result = self._execute()
        if hasattr(result, "name"):
            return result.name
        return None

    @property
    def dtype(self):
        """
        Return the dtype of the executed column.

        Example:
            >>> ds['age'].dtype
            dtype('int64')
        """
        result = self._execute()
        if hasattr(result, "dtype"):
            return result.dtype
        elif hasattr(result, "dtypes"):
            # For DataFrame results (e.g., when selecting duplicate column names)
            return result.dtypes
        else:
            # Fallback for scalar results
            import numpy as np

            return np.array([result]).dtype

    @property
    def dtypes(self):
        """
        Return the dtype of the column (alias for dtype).

        For Series, dtypes is the same as dtype.
        """
        return self.dtype

    @property
    def shape(self) -> tuple:
        """
        Return shape of the executed column.

        Example:
            >>> ds['age'].shape
            (5,)
        """
        return self._execute().shape

    @property
    def ndim(self) -> int:
        """
        Return number of dimensions (always 1 for Series).

        Example:
            >>> ds['age'].ndim
            1
        """
        return 1

    @property
    def index(self):
        """
        Return the index of the executed column.

        Example:
            >>> ds['age'].index
            RangeIndex(start=0, stop=5, step=1)
        """
        return self._execute().index

    @property
    def empty(self) -> bool:
        """
        Return True if the column has no elements.

        Example:
            >>> ds['age'].empty
            False
        """
        return len(self) == 0

    @property
    def T(self):
        """
        Return the transpose (same as values for Series).

        Example:
            >>> ds['age'].T
            array([28, 31, 29, 45, 22])
        """
        return self._execute().T

    @property
    def axes(self) -> list:
        """
        Return a list of the row axis labels.

        Example:
            >>> ds['age'].axes
            [RangeIndex(start=0, stop=5, step=1)]
        """
        return self._execute().axes

    @property
    def nbytes(self) -> int:
        """
        Return the number of bytes in the underlying data.

        Example:
            >>> ds['age'].nbytes
            40
        """
        return self._execute().nbytes

    @property
    def hasnans(self) -> bool:
        """
        Return True if there are any NaN values.

        Example:
            >>> ds['age'].hasnans
            False
        """
        return self._execute().hasnans

    @property
    def is_unique(self) -> bool:
        """
        Return True if values are unique.

        Example:
            >>> ds['age'].is_unique
            True
        """
        return self._execute().is_unique

    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return True if values are monotonic increasing.

        Example:
            >>> ds['value'].is_monotonic_increasing
            True
        """
        return self._execute().is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return True if values are monotonic decreasing.

        Example:
            >>> ds['value'].is_monotonic_decreasing
            False
        """
        return self._execute().is_monotonic_decreasing

    @property
    def array(self):
        """
        Return the ExtensionArray of the data.

        Example:
            >>> ds['age'].array
            <IntegerArray>
            [28, 31, 29, 45, 22]
            Length: 5, dtype: Int64
        """
        return self._execute().array

    def __len__(self) -> int:
        """Return length of the executed series."""
        return len(self._execute())

    def __iter__(self):
        """Iterate over executed values."""
        return iter(self._execute())

    def __getitem__(self, key):
        """
        Support indexing/subscripting like pandas Series.

        This allows patterns like:
        - df['col'].mode()[0]  # Integer index
        - result[result > 50]  # Boolean indexing with ColumnExpr

        Args:
            key: Index, slice, array of indices, or ColumnExpr for boolean indexing

        Returns:
            Single value, Series, or ColumnExpr depending on key type and context
        """
        from .conditions import Condition

        # For ColumnExpr keys (boolean indexing), return method-mode ColumnExpr
        # to preserve index alignment for chained operations
        if isinstance(key, ColumnExpr):
            # Return method-mode ColumnExpr so chained indexing works correctly
            # e.g., ds_grouped[cond1][cond2] where cond2 has different length
            return ColumnExpr(
                source=self, method_name="__getitem__", method_args=(key,)
            )

        # For Condition keys, execute condition and use as boolean mask
        if isinstance(key, Condition):
            series = self._execute()
            key = key.evaluate(pd.DataFrame({series.name or "_": series}))
            return series[key]

        # For scalar/slice keys, directly execute and index
        return self._execute()[key]

    def tolist(self) -> list:
        """Convert to Python list."""
        return self._execute().tolist()

    def to_list(self) -> list:
        """
        Convert to Python list (alias for tolist).

        Example:
            >>> ds['age'].to_list()
            [28, 31, 29, 45, 22]
        """
        return self.tolist()

    def to_numpy(self):
        """Convert to numpy array."""
        return self._execute().to_numpy()

    def to_dict(self, into=dict):
        """
        Convert Series to dict.

        Args:
            into: The collections.abc.Mapping subclass to use as the return
                object. Default is dict.

        Returns:
            dict: Dict with index as keys and values as values.

        Example:
            >>> ds['age'].to_dict()
            {0: 28, 1: 31, 2: 29, 3: 45, 4: 22}
        """
        return self._execute().to_dict(into=into)

    def to_frame(self, name=_MISSING):
        """
        Convert Series to DataFrame.

        Args:
            name: The passed name should substitute for the series name (if it has one).
                  If not provided, uses the series name.

        Returns:
            DataStore: DataStore wrapping single-column DataFrame

        Example:
            >>> ds['age'].to_frame()
               age
            0   28
            1   31
            ...
        """
        from .core import DataStore

        series = self._execute()
        # Only pass name if explicitly provided (including None)
        if name is _MISSING:
            df = series.to_frame()
        else:
            df = series.to_frame(name=name)
        return DataStore.from_df(df)

    def reset_index(
        self,
        level=None,
        *,
        drop=False,
        name=None,
        inplace=False,
        allow_duplicates=False,
    ):
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column,
        or when the index is meaningless and needs to be reset to the
        default integer index.

        Args:
            level: Only remove the given levels from the index. Removes all levels by default.
            drop: Do not try to insert index into dataframe columns.
                  This resets the index to the default integer index.
            name: The name to use for the column containing the original Series values.
                  Uses the Series name by default. This parameter is ignored when drop=True.
            inplace: Not supported (ColumnExpr is immutable)
            allow_duplicates: Allow duplicate column labels if inserting index into columns.

        Returns:
            DataFrame or Series: When drop is False (the default), a DataFrame is returned.
                The newly created columns will come first in the DataFrame, followed by the
                original Series values. When drop is True, a Series is returned.

        Example:
            >>> ds = DataStore({'category': ['A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40]})
            >>> agg_result = ds.groupby('category')['value'].sum()
            >>> agg_result.reset_index(name='total')
              category  total
            0        A     40
            1        B     60

            >>> agg_result.reset_index(drop=True)
            0    40
            1    60
            Name: value, dtype: int64
        """
        if inplace:
            raise ImmutableError("ColumnExpr")

        from .core import DataStore

        # Execute to get the actual Series
        series = self._execute()

        # Use pandas reset_index
        # Build kwargs carefully to match pandas signature
        kwargs = {"drop": drop, "allow_duplicates": allow_duplicates}
        if level is not None:
            kwargs["level"] = level
        # 'name' parameter is only valid when drop=False
        if not drop and name is not None:
            kwargs["name"] = name

        result = series.reset_index(**kwargs)

        # Return appropriate type based on result
        if isinstance(result, pd.DataFrame):
            return DataStore.from_df(result)
        else:
            # drop=True returns a Series
            return ColumnExpr(
                source=self,
                method_name="reset_index",
                method_kwargs=kwargs,
            )

    def unstack(self, level=-1, fill_value=None):
        """
        Pivot level(s) of the (necessarily hierarchical) index labels.

        Returns a DataFrame having a new level of column labels whose inner-most level
        consists of the pivoted index labels.

        If the index is not a MultiIndex, an error is raised.

        Args:
            level: Level(s) of the index to unstack, can pass level name.
                   Default is -1 (last level).
            fill_value: Replace NaN with this value if the unstack produces missing values.

        Returns:
            DataStore: Unstacked DataFrame wrapped in DataStore.

        Example:
            >>> ds = DataStore({'a': ['A', 'A', 'B', 'B'], 'b': ['X', 'Y', 'X', 'Y'], 'v': [1, 2, 3, 4]})
            >>> grouped = ds.groupby(['a', 'b'])['v'].sum()
            >>> grouped
            a  b
            A  X    1
               Y    2
            B  X    3
               Y    4
            Name: v, dtype: int64
            >>> grouped.unstack()
            b  X  Y
            a
            A  1  2
            B  3  4
        """
        from .core import DataStore

        # Execute to get the actual Series with MultiIndex
        series = self._execute()

        # Call pandas unstack
        result = series.unstack(level=level, fill_value=fill_value)

        # Return DataStore wrapping the result DataFrame
        return DataStore.from_df(result)

    def copy(self, deep=True) -> "ColumnExpr":
        """
        Make a copy of this ColumnExpr's data (lazy).

        Args:
            deep: Make a deep copy (default True)

        Returns:
            ColumnExpr: Lazy wrapper returning a copy of the data

        Example:
            >>> s = ds['age'].copy()
            >>> s.values  # Triggers execution
            array([28, 31, 29, 45, 22])
        """
        return ColumnExpr(
            source=self, method_name="copy", method_kwargs=dict(deep=deep)
        )

    def describe(self, percentiles=None, include=None, exclude=None) -> "ColumnExpr":
        """
        Generate descriptive statistics.

        Args:
            percentiles: The percentiles to include in the output.
            include: A white list of data types to include.
            exclude: A black list of data types to exclude.

        Returns:
            ColumnExpr: Lazy wrapper returning summary statistics Series

        Example:
            >>> ds['age'].describe()
            count     5.000000
            mean     31.000000
            std       8.602325
            min      22.000000
            25%      28.000000
            50%      29.000000
            75%      31.000000
            max      45.000000
            Name: age, dtype: float64
        """
        return ColumnExpr(
            source=self,
            method_name="describe",
            method_kwargs=dict(
                percentiles=percentiles, include=include, exclude=exclude
            ),
        )

    def info(
        self, verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None
    ):
        """
        Print a concise summary of a Series.

        Args:
            verbose: Whether to print the full summary.
            buf: Where to send the output (default: sys.stdout).
            max_cols: Not used for Series.
            memory_usage: Specifies whether memory usage should be displayed.
            show_counts: Whether to show non-null counts.

        Returns:
            None: Prints to stdout (or buf)

        Example:
            >>> ds['age'].info()
            <class 'pandas.core.series.Series'>
            RangeIndex: 5 entries, 0 to 4
            Series name: age
            Non-Null Count  Dtype
            --------------  -----
            5 non-null      int64
            dtypes: int64(1)
            memory usage: 168.0 bytes
        """
        return self._execute().info(
            verbose=verbose,
            buf=buf,
            max_cols=max_cols,
            memory_usage=memory_usage,
            show_counts=show_counts,
        )

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        ignore_index=False,
    ) -> "ColumnExpr":
        """
        Return a random sample of items (lazy).

        Args:
            n: Number of items to return.
            frac: Fraction of items to return.
            replace: Allow or disallow sampling with replacement.
            weights: Weight values for sampling probability.
            random_state: Seed for random number generator.
            axis: Axis to sample (0 or 'index').
            ignore_index: If True, reset index in result.

        Returns:
            ColumnExpr: Lazy wrapper returning random sample Series

        Example:
            >>> ds['age'].sample(3, random_state=42)
            2    29
            4    22
            0    28
            Name: age, dtype: int64
        """
        return ColumnExpr(
            source=self,
            method_name="sample",
            method_kwargs=dict(
                n=n,
                frac=frac,
                replace=replace,
                weights=weights,
                random_state=random_state,
                axis=axis,
                ignore_index=ignore_index,
            ),
        )

    def nlargest(self, n=5, keep="first") -> "ColumnExpr":
        """
        Return the largest n elements (lazy).

        Args:
            n: Number of largest elements to return.
            keep: How to handle duplicate values ('first', 'last', 'all').

        Returns:
            ColumnExpr: Lazy wrapper returning n largest values Series

        Example:
            >>> ds['salary'].nlargest(3)
            3    120000.0
            1     75000.0
            2     60000.0
            Name: salary, dtype: float64
        """
        return ColumnExpr(
            source=self, method_name="nlargest", method_kwargs=dict(n=n, keep=keep)
        )

    def nsmallest(self, n=5, keep="first") -> "ColumnExpr":
        """
        Return the smallest n elements (lazy).

        Args:
            n: Number of smallest elements to return.
            keep: How to handle duplicate values ('first', 'last', 'all').

        Returns:
            ColumnExpr: Lazy wrapper returning n smallest values Series

        Example:
            >>> ds['salary'].nsmallest(3)
            4    45000.0
            0    50000.0
            2    60000.0
            Name: salary, dtype: float64
        """
        return ColumnExpr(
            source=self, method_name="nsmallest", method_kwargs=dict(n=n, keep=keep)
        )

    def drop_duplicates(
        self, keep="first", inplace=False, ignore_index=False
    ) -> "ColumnExpr":
        """
        Return Series with duplicate values removed (lazy).

        Args:
            keep: Which duplicates to keep ('first', 'last', False to drop all).
            inplace: Not used (always returns new Series).
            ignore_index: If True, reset index in result.

        Returns:
            ColumnExpr: Lazy wrapper returning Series with duplicates removed

        Example:
            >>> ds['department'].drop_duplicates()
            0             HR
            1    Engineering
            3     Management
            Name: department, dtype: object
        """
        return ColumnExpr(
            source=self,
            method_name="drop_duplicates",
            method_kwargs=dict(keep=keep, ignore_index=ignore_index),
        )

    def duplicated(self, keep="first") -> "ColumnExpr":
        """
        Indicate duplicate values (lazy).

        Args:
            keep: How to mark duplicates ('first', 'last', False).

        Returns:
            ColumnExpr: Lazy wrapper returning boolean Series indicating duplicates

        Example:
            >>> ds['department'].duplicated()
            0    False
            1    False
            2     True
            3    False
            4     True
            Name: department, dtype: bool
        """
        return ColumnExpr(
            source=self, method_name="duplicated", method_kwargs=dict(keep=keep)
        )

    def hist(self, bins=10, **kwargs):
        """
        Draw histogram of the column values.

        Args:
            bins: Number of histogram bins.
            **kwargs: Additional arguments passed to matplotlib.

        Returns:
            matplotlib.axes.Axes: The histogram plot

        Example:
            >>> ds['age'].hist(bins=5)
        """
        return self._execute().hist(bins=bins, **kwargs)

    def agg(self, func=None, axis=0, *args, **kwargs):
        """
        Aggregate using one or more operations (lazy).

        When called on a ColumnExpr with groupby context (e.g., df.groupby('city')['salary'].agg(...)),
        returns a DataStore to preserve SQL semantics and enable chaining with sort_values(), head(), etc.

        Args:
            func: Function to use for aggregating (str, list, or dict).
            axis: Axis to aggregate (0 for index).
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            DataStore: When called with groupby context - preserves SQL semantics
            LazySeries: When called without groupby context - lazy wrapper

        Example:
            >>> ds['age'].agg('mean')
            31.0
            >>> ds['age'].agg(['sum', 'mean', 'std'])
            sum     155.000000
            mean     31.000000
            std       8.602325
            dtype: float64
            >>> # With groupby - returns DataStore for SQL compilation
            >>> ds.groupby('city')['salary'].agg(['mean', 'sum', 'count'])
        """
        # When called with groupby context, return DataStore to preserve SQL semantics
        if hasattr(self, "_groupby_fields") and self._groupby_fields:
            from .lazy_ops import LazyGroupByAgg
            from copy import copy
            from .expressions import Field as ExprField

            # Get the column name being aggregated
            col_name = None
            if isinstance(self._expr, ExprField):
                col_name = self._expr.name
            else:
                col_name = str(self._expr)

            # Get groupby column names
            groupby_cols = []
            for gf in self._groupby_fields:
                if isinstance(gf, ExprField):
                    groupby_cols.append(gf.name)
                else:
                    groupby_cols.append(str(gf))

            # Build agg_dict: map column to function(s)
            # func can be str, list, or dict
            if isinstance(func, str):
                agg_dict = {col_name: func}
            elif isinstance(func, (list, tuple)):
                agg_dict = {col_name: list(func)}
            elif isinstance(func, dict):
                # Already a dict, use as-is
                agg_dict = func
            else:
                # Fallback to ColumnExpr for unknown func types
                return ColumnExpr(
                    source=self,
                    method_name="agg",
                    method_args=(func,),
                    method_kwargs=dict(axis=axis, **kwargs),
                )

            # Create a shallow copy of the datastore
            new_ds = copy(self._datastore)

            # Add the lazy groupby aggregation operation
            # Mark as single_column_agg=True when called via ColumnExpr.agg(['funcs'])
            # This affects column naming: pandas returns flat names for single column agg
            is_single_col_agg = isinstance(func, (list, tuple))
            new_ds._add_lazy_op(
                LazyGroupByAgg(
                    groupby_cols=groupby_cols,
                    agg_dict=agg_dict,
                    single_column_agg=is_single_col_agg,
                    **kwargs,
                )
            )

            return new_ds

        # No groupby context - return ColumnExpr for lazy execution
        return ColumnExpr(
            source=self,
            method_name="agg",
            method_args=(func,),
            method_kwargs=dict(axis=axis, **kwargs),
        )

    def aggregate(self, func=None, axis=0, *args, **kwargs):
        """
        Aggregate using one or more operations (alias for agg).

        Args:
            func: Function to use for aggregating.
            axis: Axis to aggregate.
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Scalar, Series or DataFrame depending on func

        Example:
            >>> ds['age'].aggregate(['sum', 'mean'])
            sum     155.0
            mean     31.0
            dtype: float64
        """
        return self.agg(func=func, axis=axis, *args, **kwargs)

    def transform(self, func, *args, **kwargs):
        """
        Apply a function to the column, preserving the same index (lazy).

        When used with groupby (i.e., when _groupby_fields is set), applies
        the function within each group, returning a Series with the same
        index as the original.

        Args:
            func: Function to apply. Can be:
                - A callable that takes a Series and returns a Series/scalar
                - A string function name like 'mean', 'sum', etc.
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            LazySeries: Lazy wrapper returning transformed Series when executed

        Example:
            >>> # Without groupby - apply to entire column
            >>> ds['value'].transform(lambda x: x * 2)

            >>> # With groupby - normalize within groups
            >>> ds.groupby('category')['value'].transform(lambda x: x / x.sum())
        """
        # Check if we have groupby context
        if hasattr(self, "_groupby_fields") and self._groupby_fields:
            # Groupby transform - use operation descriptor mode (safe from recursion)
            from .expressions import Field as ExprField

            # Get groupby column names
            groupby_cols = []
            for gf in self._groupby_fields:
                if isinstance(gf, ExprField):
                    groupby_cols.append(gf.name)
                else:
                    groupby_cols.append(str(gf))

            # Use operation descriptor mode (preferred - safe from recursion)
            result = ColumnExpr.from_groupby_transform_op(
                source=self,
                groupby_cols=groupby_cols,
                func=func,
                args=args,
                kwargs=kwargs,
            )
            # Copy groupby fields for context
            result._groupby_fields = self._groupby_fields
            result._expr = self._expr  # Copy the expression (column reference)
            return result
        else:
            # Regular transform without groupby
            return ColumnExpr(
                source=self,
                method_name="transform",
                method_args=(func,) + args,
                method_kwargs=kwargs,
            )

    def where(
        self, cond, other=pd.NA, inplace=False, axis=None, level=None
    ) -> "ColumnExpr":
        """
        Replace values where the condition is False (lazy).

        Args:
            cond: Where True, keep the original value. Where False, replace.
            other: Replacement value where cond is False.
            inplace: Not used (always returns new Series).
            axis: Alignment axis.
            level: Alignment level.

        Returns:
            ColumnExpr: Lazy wrapper returning Series with replaced values

        Example:
            >>> ds['age'].where(ds['age'] > 25, 0)
            0    28
            1    31
            2    29
            3    45
            4     0
            Name: age, dtype: int64
        """
        return ColumnExpr(
            source=self,
            method_name="where",
            method_args=(cond,),
            method_kwargs=dict(other=other, axis=axis, level=level),
        )

    def argsort(
        self, axis=0, kind="quicksort", order=None, stable=None
    ) -> "ColumnExpr":
        """
        Return the indices that would sort the Series (lazy).

        Args:
            axis: Axis to sort (0 for index).
            kind: Sorting algorithm ('quicksort', 'mergesort', 'heapsort', 'stable').
            order: Not used for Series.
            stable: If True, use stable sorting.

        Returns:
            ColumnExpr: Lazy wrapper returning integer indices Series

        Example:
            >>> ds['age'].argsort()
            0    4
            1    0
            2    2
            3    1
            4    3
            dtype: int64
        """
        return ColumnExpr(
            source=self,
            method_name="argsort",
            method_kwargs=dict(axis=axis, kind=kind, order=order),
        )

    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
        ignore_index=False,
        key=None,
    ) -> "ColumnExpr":
        """
        Sort Series by index labels (lazy).

        Args:
            axis: Axis to sort.
            level: If not None, sort on values in specified index level.
            ascending: Sort ascending vs descending.
            inplace: Not used.
            kind: Sorting algorithm.
            na_position: Where to put NaN values ('first' or 'last').
            sort_remaining: If True, sort remaining levels.
            ignore_index: If True, reset index in result.
            key: Function to transform index before sorting.

        Returns:
            ColumnExpr: Lazy wrapper returning sorted Series

        Example:
            >>> ds['age'].sort_index(ascending=False)
        """
        return ColumnExpr(
            source=self,
            method_name="sort_index",
            method_kwargs=dict(
                axis=axis,
                level=level,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
                ignore_index=ignore_index,
                key=key,
            ),
        )

    def __array__(self, dtype=None, copy=None):
        """
        NumPy array interface for compatibility with numpy functions.

        This allows ColumnExpr to be used directly with numpy functions like:
        - np.allclose(ds['a'], ds['b'])
        - np.mean(ds['a'])
        - np.array(ds['a'])

        Args:
            dtype: Optional dtype for the resulting array
            copy: If True, ensure the returned array is a copy (numpy 2.0+)

        Returns:
            numpy array representation of the column
        """
        import numpy as np

        result = self._execute()
        if isinstance(result, pd.Series):
            # Use to_numpy() instead of .values to handle categorical/extension dtypes
            arr = result.to_numpy()
        elif isinstance(result, pd.DataFrame):
            arr = result.to_numpy()
        elif result is None:
            arr = np.array([])
        else:
            # Scalar result - wrap in array
            arr = np.array([result])

        if dtype is not None:
            arr = arr.astype(dtype)
        # Handle copy parameter for numpy 2.0+ compatibility
        if copy:
            arr = np.array(arr, copy=True)
        return arr

    # ========== Aggregate Methods ==========
    # These methods return ColumnExpr for SQL when called with default args,
    # or execute and compute when called with pandas/numpy-style args.

    def _create_agg_expr(
        self,
        agg_func_name: str,
        pandas_agg_func: str,
        skipna: bool = True,
        method_kwargs: dict = None,
    ):
        """
        Create an aggregation ColumnExpr, preserving groupby parameters from source.

        When called without groupby context (no _groupby_fields), this method
        executes immediately and returns the scalar result to match pandas behavior.
        With groupby context, it returns a lazy ColumnExpr.

        This helper ensures groupby_fields, groupby_as_index, and groupby_sort
        are properly propagated to aggregation expressions.

        Args:
            agg_func_name: SQL aggregate function name
            pandas_agg_func: Pandas aggregation method name
            skipna: Whether to skip NaN values
            method_kwargs: Additional method arguments

        Returns:
            scalar: When no groupby context (matches pandas Series.agg() behavior)
            ColumnExpr: When groupby context exists (lazy evaluation for SQL)
        """
        # When there's no groupby, execute immediately and return scalar
        # This matches pandas behavior: Series.mean() returns a scalar
        if not self._groupby_fields:
            series = self._execute()
            agg_method = getattr(series, pandas_agg_func)
            # Filter out axis and numeric_only for methods that don't accept them
            filtered_kwargs = {
                k: v
                for k, v in (method_kwargs or {}).items()
                if k not in ("axis", "numeric_only")
            }
            return agg_method(**filtered_kwargs)

        # With groupby, return lazy ColumnExpr for SQL generation
        return ColumnExpr(
            source=self,
            agg_func_name=agg_func_name,
            pandas_agg_func=pandas_agg_func,
            skipna=skipna,
            method_kwargs=method_kwargs or {},
            # Inherit groupby parameters from source
            groupby_fields=(
                self._groupby_fields.copy() if self._groupby_fields else None
            ),
            groupby_as_index=self._groupby_as_index,
            groupby_sort=self._groupby_sort,
            groupby_dropna=self._groupby_dropna,
        )

    def mean(
        self, axis=None, skipna=True, numeric_only=False, **kwargs
    ) -> "ColumnExpr":
        """
        Compute mean of the column.

        Returns a ColumnExpr (aggregation mode) that:
        - Displays the result when shown in notebook/REPL
        - Can be used in agg() for SQL building
        - Returns Series with groupby, scalar without

        Args:
            axis: Axis for computation (pandas/numpy compatibility)
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display

        Example:
            >>> ds['value'].mean()  # Displays scalar when shown
            31.0
            >>> ds.groupby('category')['value'].mean()  # Displays Series
            category
            A    25.5
            B    30.0
            Name: value, dtype: float64
            >>> ds.groupby('x').agg(avg=ds['value'].mean())  # Uses in SQL
        """
        return self._create_agg_expr(
            agg_func_name="avg",
            pandas_agg_func="mean",
            skipna=skipna,
            method_kwargs=dict(axis=axis, numeric_only=numeric_only, **kwargs),
        )

    def mean_sql(self):
        """
        Return AVG() SQL expression for use in select().

        Unlike mean(), this returns a ColumnExpr for building SQL queries.
        Note: SQL avg() may handle NaN differently than pandas.

        Returns:
            ColumnExpr: SQL expression for AVG(column)

        Example:
            >>> ds.select(ds['value'].mean_sql().as_('avg_value'))
        """
        from .functions import AggregateFunction

        return self._derive_expr(AggregateFunction("avg", self._expr))

    def sum(
        self, axis=None, skipna=True, numeric_only=False, min_count=0, **kwargs
    ) -> "ColumnExpr":
        """
        Compute sum of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation (pandas/numpy compatibility)
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            min_count: Minimum count of valid values
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        return self._create_agg_expr(
            agg_func_name="sum",
            pandas_agg_func="sum",
            skipna=skipna,
            method_kwargs=dict(
                axis=axis, numeric_only=numeric_only, min_count=min_count, **kwargs
            ),
        )

    def sum_sql(self):
        """Return SUM() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return self._derive_expr(AggregateFunction("sum", self._expr))

    def std(
        self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs
    ) -> "ColumnExpr":
        """
        Compute standard deviation of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            ddof: Delta degrees of freedom (1=sample, 0=population)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        # Use sample std by default (ddof=1)
        func_name = "stddevSamp" if ddof == 1 else "stddevPop"
        return self._create_agg_expr(
            agg_func_name=func_name,
            pandas_agg_func="std",
            skipna=skipna,
            method_kwargs=dict(
                axis=axis, ddof=ddof, numeric_only=numeric_only, **kwargs
            ),
        )

    def std_sql(self, sample=True):
        """Return stddevSamp() or stddevPop() SQL expression for use in select()."""
        from .functions import AggregateFunction

        func_name = "stddevSamp" if sample else "stddevPop"
        return self._derive_expr(AggregateFunction(func_name, self._expr))

    def var(
        self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs
    ) -> "ColumnExpr":
        """
        Compute variance of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            ddof: Delta degrees of freedom (1=sample, 0=population)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        # Use sample var by default (ddof=1)
        func_name = "varSamp" if ddof == 1 else "varPop"
        return self._create_agg_expr(
            agg_func_name=func_name,
            pandas_agg_func="var",
            skipna=skipna,
            method_kwargs=dict(
                axis=axis, ddof=ddof, numeric_only=numeric_only, **kwargs
            ),
        )

    def var_sql(self, sample=True):
        """Return varSamp() or varPop() SQL expression for use in select()."""
        from .functions import AggregateFunction

        func_name = "varSamp" if sample else "varPop"
        return self._derive_expr(AggregateFunction(func_name, self._expr))

    def min(self, axis=None, skipna=True, numeric_only=False, **kwargs) -> "ColumnExpr":
        """
        Compute minimum of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        return self._create_agg_expr(
            agg_func_name="min",
            pandas_agg_func="min",
            skipna=skipna,
            method_kwargs=dict(axis=axis, numeric_only=numeric_only, **kwargs),
        )

    def min_sql(self):
        """Return MIN() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return self._derive_expr(AggregateFunction("min", self._expr))

    def max(self, axis=None, skipna=True, numeric_only=False, **kwargs) -> "ColumnExpr":
        """
        Compute maximum of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        return self._create_agg_expr(
            agg_func_name="max",
            pandas_agg_func="max",
            skipna=skipna,
            method_kwargs=dict(axis=axis, numeric_only=numeric_only, **kwargs),
        )

    def max_sql(self):
        """Return MAX() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return self._derive_expr(AggregateFunction("max", self._expr))

    def first(self, **kwargs) -> "ColumnExpr":
        """
        Return the first element in each group (for groupby) or the first element (for Series).

        When used with groupby (i.e., when _groupby_fields is set), returns a Series
        with the first value from each group.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display

        Example:
            >>> ds.groupby('category')['value'].first()  # First value in each group
            category
            A    10
            B    30
            Name: value, dtype: int64
        """
        return self._create_agg_expr(
            agg_func_name="any",  # ClickHouse any() returns first encountered value
            pandas_agg_func="first",
            skipna=True,
            method_kwargs=kwargs,
        )

    def last(self, **kwargs) -> "ColumnExpr":
        """
        Return the last element in each group (for groupby) or the last element (for Series).

        When used with groupby (i.e., when _groupby_fields is set), returns a Series
        with the last value from each group.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display

        Example:
            >>> ds.groupby('category')['value'].last()  # Last value in each group
            category
            A    20
            B    40
            Name: value, dtype: int64
        """
        return self._create_agg_expr(
            agg_func_name="anyLast",  # ClickHouse anyLast() returns last encountered value
            pandas_agg_func="last",
            skipna=True,
            method_kwargs=kwargs,
        )

    def quantile(self, q=0.5, interpolation="linear", **kwargs) -> "ColumnExpr":
        """
        Compute quantile of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.
        When used with groupby (i.e., when _groupby_fields is set), returns
        a Series with the quantile value for each group.

        Args:
            q: Quantile(s) to compute. Must be between 0 and 1.
               Can be a float or list of floats. Default is 0.5 (median).
            interpolation: Method for interpolation when the desired quantile
                          lies between two data points. Options:
                          'linear', 'lower', 'higher', 'midpoint', 'nearest'.
                          Default is 'linear'.
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display

        Example:
            >>> ds['value'].quantile(0.5)  # Displays scalar when shown
            25.0
            >>> ds.groupby('category')['value'].quantile(0.5)  # Displays Series
            category
            A    20.0
            B    50.0
            Name: value, dtype: float64
        """
        return self._create_agg_expr(
            agg_func_name=f"quantile({q})",
            pandas_agg_func="quantile",
            skipna=False,  # quantile ignores NaN by default, no If suffix needed
            method_kwargs=dict(q=q, interpolation=interpolation, **kwargs),
        )

    def nth(self, n, dropna=None) -> "ColumnExpr":
        """
        Return the nth value from each group.

        When used with groupby (i.e., when _groupby_fields is set), returns the
        nth value from each group. Supports negative indexing.

        Args:
            n: Integer or list of integers. Position(s) to select. Negative
               values count from the end of each group.
            dropna: Optional, how to handle NA values. Can be 'any', 'all', or None.

        Returns:
            ColumnExpr: Lazy result that executes on display

        Example:
            >>> ds.groupby('category')['value'].nth(0)   # First value in each group
            >>> ds.groupby('category')['value'].nth(1)   # Second value in each group
            >>> ds.groupby('category')['value'].nth(-1)  # Last value in each group
        """
        # Check if we have groupby context
        if hasattr(self, "_groupby_fields") and self._groupby_fields:
            from .expressions import Field as ExprField

            # Get groupby column names
            groupby_cols = []
            for gf in self._groupby_fields:
                if isinstance(gf, ExprField):
                    groupby_cols.append(gf.name)
                else:
                    groupby_cols.append(str(gf))

            # Use operation descriptor mode (safe from recursion)
            result = ColumnExpr(
                datastore=self._datastore,
                op_type="groupby_nth",
                op_source=self,
                op_method="nth",
                op_args=(n,),
                op_kwargs={"dropna": dropna} if dropna is not None else {},
                op_groupby_cols=groupby_cols,
            )
            # Copy groupby fields for context
            result._groupby_fields = self._groupby_fields
            result._expr = self._expr
            return result
        else:
            # Without groupby context, nth doesn't make sense
            raise AttributeError(
                "nth() requires groupby context. Use ds.groupby('col')['value'].nth(n)"
            )

    def count(self) -> "ColumnExpr":
        """
        Count non-NA values in the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        return self._create_agg_expr(
            agg_func_name="count",
            pandas_agg_func="count",
            skipna=True,
        )

    def count_sql(self):
        """Return COUNT() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return self._derive_expr(AggregateFunction("count", self._expr))

    def median(
        self, axis=None, skipna=True, numeric_only=False, **kwargs
    ) -> "ColumnExpr":
        """
        Compute median of the column.

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy aggregate that executes on display
        """
        return self._create_agg_expr(
            agg_func_name="median",
            pandas_agg_func="median",
            skipna=skipna,
            method_kwargs=dict(axis=axis, numeric_only=numeric_only, **kwargs),
        )

    def median_sql(self):
        """Return median() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return self._derive_expr(AggregateFunction("median", self._expr))

    def prod(
        self, axis=None, skipna=True, numeric_only=False, min_count=0, **kwargs
    ) -> "ColumnExpr":
        """
        Compute product of values (lazy).

        Returns a ColumnExpr (aggregation mode) that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values
            numeric_only: Include only numeric columns
            min_count: Minimum count of valid values
            **kwargs: Additional pandas arguments

        Returns:
            ColumnExpr: Lazy wrapper returning product of values
        """
        return self._create_agg_expr(
            agg_func_name="prod",
            pandas_agg_func="prod",
            skipna=skipna,
            method_kwargs=dict(
                axis=axis, numeric_only=numeric_only, min_count=min_count, **kwargs
            ),
        )

    def cumsum(
        self, axis=None, dtype=None, out=None, *, skipna=True, **kwargs
    ) -> "ColumnExpr":
        """
        Compute cumulative sum of the column (lazy).

        When called on a groupby result (e.g., ds.groupby('cat')['val'].cumsum()),
        this generates SQL window function: SUM(val) OVER (PARTITION BY cat).

        Args:
            axis: NumPy axis parameter
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            ColumnExpr: Lazy wrapper returning cumulative sum Series
        """
        return self._create_window_method(
            "cumsum", "sum", axis=axis, skipna=skipna, **kwargs
        )

    def cumprod(
        self, axis=None, dtype=None, out=None, *, skipna=True, **kwargs
    ) -> "ColumnExpr":
        """
        Compute cumulative product of the column (lazy).

        Note: cumprod is not easily expressed in SQL, so it falls back to pandas execution.

        Args:
            axis: NumPy axis parameter
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            ColumnExpr: Lazy wrapper returning cumulative product Series
        """
        # cumprod doesn't have a direct SQL equivalent, use pandas fallback
        return ColumnExpr(
            source=self,
            method_name="cumprod",
            method_kwargs=dict(axis=axis, skipna=skipna, **kwargs),
        )

    def cummax(self, axis=None, skipna=True, **kwargs) -> "ColumnExpr":
        """
        Compute cumulative maximum of the column (lazy).

        When called on a groupby result (e.g., ds.groupby('cat')['val'].cummax()),
        this generates SQL window function: MAX(val) OVER (PARTITION BY cat).

        Args:
            axis: Axis parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            ColumnExpr: Lazy wrapper returning cumulative maximum Series
        """
        return self._create_window_method(
            "cummax", "max", axis=axis, skipna=skipna, **kwargs
        )

    def cummin(self, axis=None, skipna=True, **kwargs) -> "ColumnExpr":
        """
        Compute cumulative minimum of the column (lazy).

        When called on a groupby result (e.g., ds.groupby('cat')['val'].cummin()),
        this generates SQL window function: MIN(val) OVER (PARTITION BY cat).

        Args:
            axis: Axis parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            ColumnExpr: Lazy wrapper returning cumulative minimum Series
        """
        return self._create_window_method(
            "cummin", "min", axis=axis, skipna=skipna, **kwargs
        )

    # ========== Window / Rolling Methods ==========

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        closed=None,
        step=None,
        method="single",
    ):
        """
        Provide rolling window calculations.

        Executes the column and returns a pandas Rolling object.

        Args:
            window: Size of the moving window
            min_periods: Minimum observations in window required to have a value
            center: Set labels at center of window
            win_type: Window type for weighted calculations
            on: Column label to use for index
            closed: Endpoints inclusion for interval
            step: Step size between windows (pandas >= 1.5)
            method: Execution method ('single' or 'table')

        Returns:
            Rolling: pandas Rolling object for chaining aggregations

        Example:
            >>> ds['value'].rolling(3).mean()
            >>> ds['value'].rolling(5, min_periods=1).sum()
            >>> ds['value'].rolling(window=3, center=True).std()
        """
        series = self._execute()
        # Build kwargs, excluding None values and deprecated params
        kwargs = {
            "window": window,
            "min_periods": min_periods,
            "center": center,
            "win_type": win_type,
            "on": on,
            "closed": closed,
            "method": method,
        }
        # step parameter added in pandas 1.5
        if step is not None:
            kwargs["step"] = step
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return series.rolling(**kwargs)

    def expanding(self, min_periods=1, method="single"):
        """
        Provide expanding window calculations.

        Executes the column and returns a pandas Expanding object.

        Args:
            min_periods: Minimum observations in window required to have a value
            method: Execution method ('single' or 'table')

        Returns:
            Expanding: pandas Expanding object for chaining aggregations

        Example:
            >>> ds['value'].expanding().mean()
            >>> ds['value'].expanding(min_periods=3).sum()
        """
        series = self._execute()
        return series.expanding(min_periods=min_periods, method=method)

    def ewm(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        times=None,
        method="single",
    ):
        """
        Provide exponentially weighted (EW) calculations.

        Executes the column and returns a pandas ExponentialMovingWindow object.

        Args:
            com: Decay in terms of center of mass
            span: Decay in terms of span
            halflife: Decay in terms of half-life
            alpha: Smoothing factor directly
            min_periods: Minimum observations required
            adjust: Adjust weights for imbalance
            ignore_na: Ignore NA values in weights
            times: Times for observations (for halflife)
            method: Execution method ('single' or 'table')

        Returns:
            ExponentialMovingWindow: pandas EWM object for chaining

        Example:
            >>> ds['value'].ewm(span=10).mean()
            >>> ds['value'].ewm(alpha=0.5).std()
        """
        series = self._execute()
        kwargs = {
            "com": com,
            "span": span,
            "halflife": halflife,
            "alpha": alpha,
            "min_periods": min_periods,
            "adjust": adjust,
            "ignore_na": ignore_na,
            "times": times,
            "method": method,
        }
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return series.ewm(**kwargs)

    def shift(self, periods=1, freq=None, axis=0, fill_value=None) -> "ColumnExpr":
        """
        Shift values by desired number of periods (lazy).

        When possible, uses SQL LAG()/LEAD() window functions for better performance.
        Uses lagInFrame for positive periods (shift down / previous values) and
        leadInFrame for negative periods (shift up / next values).

        Falls back to pandas for:
        - freq parameter (time series offset)
        - fill_value parameter (SQL returns NULL, pandas supports custom fill)
        - Groupby context (has special handling in _execute)
        - String/object columns (toFloat64 would corrupt data)

        Args:
            periods: Number of periods to shift (positive = shift down)
            freq: Offset to use from tseries
            axis: Axis to shift on
            fill_value: Value to use for filling new missing values

        Returns:
            ColumnExpr: Lazy wrapper returning shifted Series

        Example:
            >>> ds['value'].shift(1)  # Previous value (uses SQL LAG)
            >>> ds['value'].shift(-1)  # Next value (uses SQL LEAD)
            >>> ds['value'].shift(1, fill_value=0)  # Uses pandas (fill_value)
        """
        from .functions import WindowFunction, Function
        from .expressions import Literal, Field

        # Special case: shift(0) is a no-op, return self to preserve dtype
        if periods == 0 and freq is None:
            return self

        # Check if we can use SQL window functions
        # Requirements:
        # 1. No freq parameter (time series offset not supported in SQL)
        # 2. No fill_value (SQL LAG/LEAD returns NULL, pandas supports custom fill)
        # 3. We have a valid expression (Field) to shift
        # 4. Not in groupby context (groupby shift has special handling in _execute)
        # 5. Must be numeric column (string columns don't work with toFloat64)
        can_use_sql = (
            freq is None
            and fill_value is None
            and self._expr is not None
            and not self._groupby_fields  # Groupby context uses different path
        )

        # Check if the column is numeric (skip SQL for string columns)
        if can_use_sql and self._datastore is not None:
            try:
                # Try to get the column dtype from the datastore
                col_name = None
                if isinstance(self._expr, Field):
                    col_name = self._expr.name
                if col_name:
                    # Use _get_df() to get the DataFrame
                    df = self._datastore._get_df()
                    if df is not None and col_name in df.columns:
                        col_dtype = df[col_name].dtype
                        # Only use SQL for numeric columns
                        if col_dtype == "object" or pd.api.types.is_string_dtype(
                            col_dtype
                        ):
                            can_use_sql = False
            except Exception:
                # If we can't determine dtype, fall back to pandas
                pass

        if can_use_sql:
            # Build the window function
            # lagInFrame for positive periods (previous values)
            # leadInFrame for negative periods (next values)
            # Wrap column with toNullable to get proper NULL for missing rows
            col_nullable = Function("toNullable", self._expr)

            if periods >= 0:
                # shift(1) -> lagInFrame(col, 1) = previous value
                window_func = WindowFunction(
                    "lagInFrame", col_nullable, Literal(periods)
                )
            else:
                # shift(-1) -> leadInFrame(col, 1) = next value
                window_func = WindowFunction(
                    "leadInFrame", col_nullable, Literal(-periods)
                )

            # Add OVER clause with ORDER BY _row_id to preserve original row order
            # _row_id is a built-in virtual column in chDB v4.0.0b5+ for Python() table function
            # Frame must span entire partition to access all rows
            row_idx_field = Field("_row_id")
            window_func = window_func.over(
                order_by=[row_idx_field],
                frame="ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
            )

            # Wrap with toFloat64 to match pandas behavior
            # pandas shift() returns float64 because NaN is only supported in float
            float_wrapped = Function("toFloat64", window_func)

            # Preserve the original column name for the result Series
            col_name = None
            if isinstance(self._expr, Field):
                col_name = self._expr.name

            # Return new ColumnExpr with the window function expression
            return ColumnExpr(
                expr=float_wrapped,
                datastore=self._datastore,
                alias=col_name,
                groupby_fields=(
                    self._groupby_fields.copy() if self._groupby_fields else None
                ),
                groupby_as_index=self._groupby_as_index,
                groupby_sort=self._groupby_sort,
            )

        # Fall back to pandas for unsupported cases
        return ColumnExpr(
            source=self,
            method_name="shift",
            method_kwargs=dict(
                periods=periods, freq=freq, axis=axis, fill_value=fill_value
            ),
        )

    def diff(self, periods=1) -> "ColumnExpr":
        """
        First discrete difference of element (lazy).

        When possible, uses SQL window functions for better performance:
        diff(n) = value - lagInFrame(value, n)  for positive periods
        diff(-n) = value - leadInFrame(value, n)  for negative periods

        Falls back to pandas for:
        - Groupby context (has special handling in _execute)
        - No valid expression

        Args:
            periods: Periods to shift for calculating difference

        Returns:
            ColumnExpr: Lazy wrapper returning first differences Series

        Example:
            >>> ds['value'].diff()  # Difference from previous (uses SQL)
            >>> ds['value'].diff(2)  # Difference from 2 periods ago
        """
        from .functions import WindowFunction, Function
        from .expressions import Literal, Field

        # Check if we can use SQL window functions
        # Requirements:
        # 1. We have a valid expression (Field) to diff
        # 2. Not in groupby context (groupby diff has special handling in _execute)
        can_use_sql = (
            self._expr is not None and not self._groupby_fields
        )  # Groupby context uses different path

        if can_use_sql:
            # Build: value - lagInFrame(toNullable(value), periods)
            # Wrap column with toNullable to get proper NULL for first rows
            col_nullable = Function("toNullable", self._expr)

            # Create lag/lead window function based on periods sign
            if periods >= 0:
                # diff(1) -> value - lagInFrame(col, 1)
                lag_func = WindowFunction("lagInFrame", col_nullable, Literal(periods))
            else:
                # diff(-1) -> value - leadInFrame(col, 1)
                lag_func = WindowFunction(
                    "leadInFrame", col_nullable, Literal(-periods)
                )

            # Add OVER clause with ORDER BY _row_id to preserve original row order
            # _row_id is a built-in virtual column in chDB v4.0.0b5+ for Python() table function
            row_idx_field = Field("_row_id")
            lag_func = lag_func.over(
                order_by=[row_idx_field],
                frame="ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
            )

            # diff = current - lag
            # Result will be NULL for first `periods` rows (since lag is NULL)
            diff_expr = Function("minus", self._expr, lag_func)

            # Wrap with toFloat64 to match pandas behavior
            # pandas diff() returns float64 because NaN is only supported in float
            float_wrapped = Function("toFloat64", diff_expr)

            # Preserve the original column name for the result Series
            col_name = None
            if isinstance(self._expr, Field):
                col_name = self._expr.name

            # Return new ColumnExpr with the diff expression
            return ColumnExpr(
                expr=float_wrapped,
                datastore=self._datastore,
                alias=col_name,
                groupby_fields=(
                    self._groupby_fields.copy() if self._groupby_fields else None
                ),
                groupby_as_index=self._groupby_as_index,
                groupby_sort=self._groupby_sort,
            )

        # Fall back to pandas for unsupported cases
        return ColumnExpr(
            source=self, method_name="diff", method_kwargs=dict(periods=periods)
        )

    def pct_change(
        self, periods=1, fill_method=None, limit=None, freq=None, **kwargs
    ) -> "ColumnExpr":
        """
        Percentage change between current and prior element (lazy).

        Args:
            periods: Periods to shift for calculation
            fill_method: Method to use for filling NA (deprecated)
            limit: Number of NA values to fill (deprecated)
            freq: Increment for time series
            **kwargs: Additional arguments

        Returns:
            ColumnExpr: Lazy wrapper returning percentage change Series

        Example:
            >>> ds['value'].pct_change()  # % change from previous
            >>> ds['value'].pct_change(periods=2)  # % change from 2 ago
        """
        # Handle deprecated parameters - pass only valid ones
        pct_kwargs = {"periods": periods}
        if freq is not None:
            pct_kwargs["freq"] = freq
        pct_kwargs.update(kwargs)
        return ColumnExpr(
            source=self, method_name="pct_change", method_kwargs=pct_kwargs
        )

    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=False,
        na_option="keep",
        ascending=True,
        pct=False,
    ) -> "ColumnExpr":
        """
        Compute numerical data ranks along axis (lazy).

        When possible, uses SQL window functions for better performance:
        - method='first' -> SQL ROW_NUMBER()
        - method='min' -> SQL RANK()
        - method='dense' -> SQL DENSE_RANK()

        Other methods ('average', 'max') and special options (na_option, pct)
        fall back to pandas execution.

        Args:
            axis: Axis for ranking
            method: How to rank equal values ('average', 'min', 'max', 'first', 'dense')
            numeric_only: Include only numeric columns
            na_option: How to handle NA values ('keep', 'top', 'bottom')
            ascending: Ascending or descending
            pct: Return ranks as percentile

        Returns:
            ColumnExpr: Lazy wrapper returning ranks Series

        Example:
            >>> ds['score'].rank()
            >>> ds['score'].rank(method='dense', ascending=False)
            >>> ds['score'].rank(method='first')  # Uses SQL ROW_NUMBER()
        """
        from .functions import WindowFunction, Function

        # Mapping from pandas method to SQL window function
        SQL_RANK_METHODS = {
            "first": "row_number",  # ROW_NUMBER() - unique rank by order of appearance
            "min": "rank",  # RANK() - same rank for ties, skip next
            "dense": "dense_rank",  # DENSE_RANK() - same rank for ties, no skip
        }

        # Check if we can use SQL window functions
        # Requirements:
        # 1. method is SQL-compatible ('first', 'min', 'dense')
        # 2. na_option is 'keep' (SQL NULL handling matches this)
        # 3. pct is False (SQL doesn't directly support percentile rank)
        # 4. We have a valid expression (Field) to order by
        # 5. Not in groupby context (groupby rank has special handling)
        can_use_sql = (
            method in SQL_RANK_METHODS
            and na_option == "keep"
            and not pct
            and self._expr is not None
            and not self._groupby_fields  # Groupby context uses different path
        )

        if can_use_sql:
            # Create SQL window function
            sql_func_name = SQL_RANK_METHODS[method]

            # Build ORDER BY clause
            order_by_expr = self._expr
            if not ascending:
                # For descending, append DESC to the order
                if isinstance(order_by_expr, Field):
                    order_by = f"{order_by_expr.name} DESC"
                else:
                    # For complex expressions, convert to SQL and append DESC
                    order_by = f"{order_by_expr.to_sql()} DESC"
            else:
                order_by = order_by_expr

            # Create the window function: e.g., row_number() OVER (ORDER BY col)
            window_func = WindowFunction(sql_func_name).over(order_by=order_by)

            # Wrap with toFloat64 to match pandas return type (float64)
            # pandas rank() always returns float64, SQL window functions return integers
            float_wrapped = Function("toFloat64", window_func)

            # Preserve the original column name for the result Series
            # Only set alias for simple Field expressions, not complex expressions
            # This matches pandas behavior where (df['a'] + df['b']).rank() has name=None
            col_name = None
            if isinstance(self._expr, Field):
                col_name = self._expr.name

            # Return new ColumnExpr with the window function expression
            return ColumnExpr(
                expr=float_wrapped,
                datastore=self._datastore,
                alias=col_name,  # Preserve original column name (None for complex expr)
                groupby_fields=(
                    self._groupby_fields.copy() if self._groupby_fields else None
                ),
                groupby_as_index=self._groupby_as_index,
                groupby_sort=self._groupby_sort,
            )

        # Fall back to pandas for unsupported methods or options
        return ColumnExpr(
            source=self,
            method_name="rank",
            method_kwargs=dict(
                axis=axis,
                method=method,
                numeric_only=numeric_only,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            ),
        )

    def mode(self, dropna: bool = True) -> "ColumnExpr":
        """
        Return the mode(s) of the column.

        The mode is the value that appears most frequently.

        Args:
            dropna: Don't consider NaN/NaT values (default True)

        Returns:
            ColumnExpr: Lazy wrapper returning Series containing the mode value(s)

        Example:
            >>> ds['category'].mode()
            0    A
            dtype: object

            >>> ds['category'].mode()[0]  # Get first mode value
            'A'
        """
        return ColumnExpr(
            source=self, method_name="mode", method_kwargs=dict(dropna=dropna)
        )

    def argmin(self, axis=None, out=None, *, skipna=True, **kwargs):
        """
        Return the index of the minimum value.

        Args:
            axis: NumPy axis parameter
            out: NumPy out parameter (not supported)
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            int: Index of the minimum value
        """
        return self._execute().argmin(axis=axis, skipna=skipna, **kwargs)

    def argmax(self, axis=None, out=None, *, skipna=True, **kwargs):
        """
        Return the index of the maximum value.

        Args:
            axis: NumPy axis parameter
            out: NumPy out parameter (not supported)
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            int: Index of the maximum value
        """
        return self._execute().argmax(axis=axis, skipna=skipna, **kwargs)

    def idxmin(self, axis=0, skipna=True, numeric_only=False):
        """
        Return the row label of the minimum value.

        Args:
            axis: Not used for Series, kept for pandas compatibility
            skipna: Whether to skip NA values (default True)
            numeric_only: Not used for Series

        Returns:
            The index label of the minimum value
        """
        return self._execute().idxmin(skipna=skipna)

    def idxmax(self, axis=0, skipna=True, numeric_only=False):
        """
        Return the row label of the maximum value.

        Args:
            axis: Not used for Series, kept for pandas compatibility
            skipna: Whether to skip NA values (default True)
            numeric_only: Not used for Series

        Returns:
            The index label of the maximum value
        """
        return self._execute().idxmax(skipna=skipna)

    def any(self, axis=None, out=None, keepdims=False, *, skipna=True, **kwargs):
        """
        Return whether any element is True.

        Args:
            axis: NumPy axis parameter
            out: NumPy out parameter (not supported)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            bool: True if any element is True
        """
        return self._execute().any(axis=axis, skipna=skipna, **kwargs)

    def all(self, axis=None, out=None, keepdims=False, *, skipna=True, **kwargs):
        """
        Return whether all elements are True.

        Args:
            axis: NumPy axis parameter
            out: NumPy out parameter (not supported)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            bool: True if all elements are True
        """
        return self._execute().all(axis=axis, skipna=skipna, **kwargs)

    def sem(self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs):
        """
        Compute standard error of the mean.

        Args:
            axis: Not used for Series, kept for pandas compatibility
            skipna: Whether to skip NA values (default True)
            ddof: Delta degrees of freedom (default 1)
            numeric_only: Not used for Series

        Returns:
            Scalar: Standard error of the mean
        """
        return self._execute().sem(skipna=skipna, ddof=ddof)

    def skew(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute unbiased skewness.

        Args:
            axis: Not used for Series, kept for pandas compatibility
            skipna: Whether to skip NA values (default True)
            numeric_only: Not used for Series

        Returns:
            Scalar: Unbiased skewness
        """
        return self._execute().skew(skipna=skipna)

    def kurt(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute unbiased kurtosis.

        Args:
            axis: Not used for Series, kept for pandas compatibility
            skipna: Whether to skip NA values (default True)
            numeric_only: Not used for Series

        Returns:
            Scalar: Unbiased kurtosis
        """
        return self._execute().kurt(skipna=skipna)

    def kurtosis(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute unbiased kurtosis (alias for kurt).

        Args:
            axis: Not used for Series, kept for pandas compatibility
            skipna: Whether to skip NA values (default True)
            numeric_only: Not used for Series

        Returns:
            Scalar: Unbiased kurtosis
        """
        return self.kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def first_valid_index(self):
        """
        Return the index of the first non-NA value.

        Returns:
            Index label or None if all values are NA
        """
        return self._execute().first_valid_index()

    def last_valid_index(self):
        """
        Return the index of the last non-NA value.

        Returns:
            Index label or None if all values are NA
        """
        return self._execute().last_valid_index()

    # ========== Pandas Series Methods ==========

    def apply(self, func, convert_dtype=True, args=(), **kwargs) -> "ColumnExpr":
        """
        Apply a function to each element of the column (lazy).

        Note: Functions cannot be translated to SQL, so this always uses pandas.

        Args:
            func: Function to apply to each element
            convert_dtype: Try to find better dtype for results (default True)
            args: Positional arguments to pass to func
            **kwargs: Additional keyword arguments to pass to func

        Returns:
            ColumnExpr: Lazy wrapper returning Series with the function applied

        Example:
            >>> ds = DataStore.from_file('data.csv')
            >>> ds['name'].apply(str.upper)
            0    ALICE
            1      BOB
            2    CAROL
            Name: name, dtype: object

            >>> ds['age'].apply(lambda x: x * 2)
            0    56
            1    62
            2    58
            Name: age, dtype: int64
        """
        return ColumnExpr(
            source=self,
            method_name="apply",
            method_args=(func,),
            method_kwargs=dict(convert_dtype=convert_dtype, args=args, **kwargs),
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> "ColumnExpr":
        """
        Return a Series containing counts of unique values (lazy).

        Returns a ColumnExpr that executes only when displayed
        or explicitly converted. This enables:
        - Delayed execution until result is needed
        - Chaining with other operations (e.g., .head(), .plot.pie())
        - Future: SQL-based execution via GROUP BY COUNT

        Args:
            normalize: If True, return relative frequencies instead of counts
            sort: Sort by frequencies (default True)
            ascending: Sort in ascending order (default False)
            bins: Group values into half-open bins (for numeric data)
            dropna: Don't include counts of NaN (default True)

        Returns:
            ColumnExpr: Lazy wrapper that executes on display

        Example:
            >>> ds = DataStore.from_file('data.csv')
            >>> ds['category'].value_counts()  # Lazy - not executed yet
            A    150
            B    100
            C     50
            Name: category, dtype: int64

            >>> ds['category'].value_counts().head(2)  # Still lazy
            A    150
            B    100
            Name: category, dtype: int64

            >>> (ds['col'] > 0.5).value_counts().plot.pie()  # Works!
        """
        return ColumnExpr(
            source=self,
            method_name="value_counts",
            method_kwargs=dict(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            ),
        )

    def unique(self) -> "ColumnExpr":
        """
        Return unique values of the column (lazy).

        Returns a ColumnExpr that executes only when displayed.

        Returns:
            ColumnExpr: Lazy wrapper returning unique values

        Example:
            >>> ds['category'].unique()  # Lazy
            array(['A', 'B', 'C'], dtype=object)
        """
        return ColumnExpr(source=self, method_name="unique")

    def nunique(self, dropna: bool = True) -> "ColumnExpr":
        """
        Return number of unique values (lazy).

        Returns a ColumnExpr that executes only when displayed.
        When used with groupby, returns a Series with unique counts per group.

        Args:
            dropna: Do not include NaN in the count (default True)

        Returns:
            ColumnExpr: Lazy wrapper returning count of unique values

        Example:
            >>> ds['category'].nunique()  # Lazy scalar
            3
            >>> ds.groupby('group')['value'].nunique()  # Series per group
        """
        # Use uniqExact for counting distinct values in ClickHouse
        # uniqExact already excludes NULL by default, which matches pandas dropna=True behavior
        return self._create_agg_expr(
            agg_func_name="uniqExact",
            pandas_agg_func="nunique",
            skipna=dropna,
            method_kwargs=dict(dropna=dropna),
        )

    def factorize(self, sort: bool = False, use_na_sentinel: bool = True):
        """
        Encode the object as an enumerated type or categorical variable.

        This method is useful for obtaining a numeric representation of an
        array when all that matters is identifying distinct values.

        Args:
            sort: Sort unique values and shuffle codes to maintain the
                  relationship. (default False)
            use_na_sentinel: If True, the sentinel -1 will be used for NaN values.
                            If False, NaN values will be encoded as non-negative
                            integers and will not drop the NaN from the uniques.
                            (default True)

        Returns:
            tuple[np.ndarray, np.ndarray]: (codes, uniques)
                - codes: An integer ndarray that is an indexer into uniques.
                - uniques: The unique valid values.

        Example:
            >>> ds['category'].factorize()
            (array([0, 1, 2, 0, 1]), array(['a', 'b', 'c'], dtype=object))
        """
        series = self._execute()
        return series.factorize(sort=sort, use_na_sentinel=use_na_sentinel)

    def searchsorted(self, value, side: str = "left", sorter=None):
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted Series such that, if the corresponding
        elements in value were inserted before the indices, the order of
        the Series would be preserved.

        Args:
            value: Values to insert into the Series.
            side: If 'left', the index of the first suitable location found
                  is given. If 'right', return the last such index.
                  (default 'left')
            sorter: Optional array of integer indices that sort the Series
                   into ascending order.

        Returns:
            int or ndarray: Insertion point(s)

        Example:
            >>> ds['value'].searchsorted(4)
            2
        """
        series = self._execute()
        return series.searchsorted(value, side=side, sorter=sorter)

    def get(self, key, default=None):
        """
        Get value at key from Series result, with optional default.

        This method provides pandas Series.get() behavior for ColumnExpr,
        enabling key-based access to aggregated results.

        Args:
            key: Label of the value to get
            default: Default value if key is not found (default None)

        Returns:
            Value at key, or default if key not found

        Example:
            >>> result = ds.groupby('category')['value'].sum()
            >>> result.get('A', 0)  # Get sum for category 'A', default 0
            150
        """
        # Execute to get the actual Series
        series = self._execute()

        # If result is a Series, use its .get() method
        if isinstance(series, pd.Series):
            return series.get(key, default)

        # For scalar results or other types, can't use .get()
        return default

    def map(self, arg, na_action=None) -> "ColumnExpr":
        """
        Map values of Series according to input mapping or function (lazy).

        Args:
            arg: Mapping correspondence (dict, Series, or function)
            na_action: If 'ignore', propagate NaN values without passing to mapping

        Returns:
            ColumnExpr: Lazy wrapper returning Series with mapped values

        Example:
            >>> ds['grade'].map({'A': 4.0, 'B': 3.0, 'C': 2.0})
            0    4.0
            1    3.0
            2    2.0
            Name: grade, dtype: float64
        """
        return ColumnExpr(
            source=self,
            method_name="map",
            method_args=(arg,),
            method_kwargs=dict(na_action=na_action),
        )

    def fillna(
        self, value=None, method=None, axis=None, inplace=False, limit=None
    ) -> "ColumnExpr":
        """
        Fill NA/NaN values using pandas (lazy).

        Args:
            value: Value to use to fill holes (can be ColumnExpr or scalar)
            method: Method to use for filling holes ('ffill', 'bfill')
            axis: Axis along which to fill (0 or 'index')
            inplace: Not supported, always returns new Series
            limit: Maximum number of consecutive NaN values to fill

        Returns:
            ColumnExpr: Lazy wrapper returning Series with NA values filled

        Example:
            >>> ds['value'].fillna(0)
            >>> ds['Cabin'].fillna('Unknown')
        """
        if inplace:
            raise ImmutableError("ColumnExpr")

        return ColumnExpr(
            source=self,
            method_name="fillna",
            method_kwargs=dict(value=value, method=method, axis=axis, limit=limit),
        )

    def fillna_sql(self, value):
        """
        Return ifNull() SQL expression for use in SQL queries.

        Note: SQL ifNull() only handles SQL NULL, not pandas NaN.
        For pandas compatibility, use fillna() instead.

        Args:
            value: Value to use to fill NULL values

        Returns:
            ColumnExpr: SQL expression for ifNull(column, value)
        """
        from .functions import Function
        from .expressions import Literal

        if isinstance(value, ColumnExpr):
            fill_value = value._expr
        elif isinstance(value, Expression):
            fill_value = value
        else:
            fill_value = Literal(value)

        return self._derive_expr(Function("ifNull", self._expr, fill_value))

    def _contains_aggregate(self, expr) -> bool:
        """Check if an expression contains aggregate functions."""
        from .functions import Function, AggregateFunction

        if isinstance(expr, AggregateFunction):
            return True

        if isinstance(expr, Function):
            # Check function name for common aggregates
            agg_names = {
                "avg",
                "sum",
                "count",
                "min",
                "max",
                "median",
                "stddevSamp",
                "stddevPop",
                "varSamp",
                "varPop",
                "any",
                "argMin",
                "argMax",
                "groupArray",
            }
            if expr.name.lower() in agg_names:
                return True
            # Recursively check arguments
            for arg in expr.args:
                if self._contains_aggregate(arg):
                    return True

        # Check other expression types that might contain nested expressions
        if hasattr(expr, "args"):
            for arg in expr.args:
                if self._contains_aggregate(arg):
                    return True

        return False

    def dropna(self) -> "ColumnExpr":
        """
        Return Series with missing values removed (lazy).

        Returns:
            ColumnExpr: Lazy wrapper returning Series with NA values removed

        Example:
            >>> ds['value'].dropna()
        """
        return ColumnExpr(source=self, method_name="dropna")

    def ffill(
        self, axis=None, inplace=False, limit=None, limit_area=None
    ) -> "ColumnExpr":
        """
        Fill NA/NaN values by propagating the last valid observation forward (lazy).

        Args:
            axis: Not used, for pandas compatibility
            inplace: Not supported (ColumnExpr is immutable)
            limit: Maximum number of consecutive NaN values to forward fill
            limit_area: Restrict filling to 'inside' or 'outside' values (pandas >= 2.1.0)

        Returns:
            ColumnExpr: Lazy wrapper returning forward-filled Series

        Example:
            >>> ds['value'].ffill()
        """
        if inplace:
            raise ImmutableError("ColumnExpr")

        # Build kwargs, excluding limit_area if pandas doesn't support it
        ffill_kwargs = {"axis": axis, "limit": limit}
        if _PANDAS_HAS_LIMIT_AREA and limit_area is not None:
            ffill_kwargs["limit_area"] = limit_area
        return ColumnExpr(source=self, method_name="ffill", method_kwargs=ffill_kwargs)

    def bfill(
        self, axis=None, inplace=False, limit=None, limit_area=None
    ) -> "ColumnExpr":
        """
        Fill NA/NaN values by propagating the next valid observation backward (lazy).

        Args:
            axis: Not used, for pandas compatibility
            inplace: Not supported (ColumnExpr is immutable)
            limit: Maximum number of consecutive NaN values to backward fill
            limit_area: Restrict filling to 'inside' or 'outside' values (pandas >= 2.1.0)

        Returns:
            ColumnExpr: Lazy wrapper returning backward-filled Series

        Example:
            >>> ds['value'].bfill()
        """
        if inplace:
            raise ImmutableError("ColumnExpr")

        # Build kwargs, excluding limit_area if pandas doesn't support it
        bfill_kwargs = {"axis": axis, "limit": limit}
        if _PANDAS_HAS_LIMIT_AREA and limit_area is not None:
            bfill_kwargs["limit_area"] = limit_area
        return ColumnExpr(source=self, method_name="bfill", method_kwargs=bfill_kwargs)

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction=None,
        limit_area=None,
        **kwargs,
    ) -> "ColumnExpr":
        """
        Fill NaN values using an interpolation method (lazy).

        Args:
            method: Interpolation method ('linear', 'index', 'pad', etc.)
            axis: Axis to interpolate along
            limit: Maximum number of consecutive NaNs to fill
            inplace: Not supported (ColumnExpr is immutable)
            limit_direction: Direction to fill ('forward', 'backward', 'both')
            limit_area: Restrict filling to 'inside' or 'outside' values (pandas >= 2.1.0)
            **kwargs: Additional arguments passed to pandas interpolate

        Returns:
            ColumnExpr: Lazy wrapper returning interpolated Series

        Example:
            >>> ds['value'].interpolate()
            >>> ds['value'].interpolate(method='polynomial', order=2)
        """
        if inplace:
            raise ImmutableError("ColumnExpr")

        # Build kwargs, excluding limit_area if pandas doesn't support it
        interp_kwargs = {
            "method": method,
            "axis": axis,
            "limit": limit,
            "limit_direction": limit_direction,
        }
        if _PANDAS_HAS_LIMIT_AREA and limit_area is not None:
            interp_kwargs["limit_area"] = limit_area
        interp_kwargs.update(kwargs)
        return ColumnExpr(
            source=self, method_name="interpolate", method_kwargs=interp_kwargs
        )

    def astype(self, dtype, copy=True, errors="raise") -> "ColumnExpr":
        """
        Cast to a specified dtype (lazy).

        Args:
            dtype: Data type to cast to
            copy: Return a copy (default True)
            errors: Control raising of exceptions ('raise' or 'ignore')

        Returns:
            ColumnExpr: Lazy wrapper returning Series with new dtype

        Example:
            >>> ds['age'].astype(float)
        """
        return ColumnExpr(
            source=self,
            method_name="astype",
            method_args=(dtype,),
            method_kwargs=dict(copy=copy, errors=errors),
        )

    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
        key=None,
    ) -> "ColumnExpr":
        """
        Sort by the values (lazy).

        Args:
            axis: Axis to sort along
            ascending: Sort ascending vs. descending
            inplace: Not supported
            kind: Sort algorithm
            na_position: Position of NaN values ('first' or 'last')
            ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1
            key: Apply the key function to values before sorting

        Returns:
            ColumnExpr: Lazy wrapper returning sorted Series
        """
        if inplace:
            raise ImmutableError("ColumnExpr")
        return ColumnExpr(
            source=self,
            method_name="sort_values",
            method_kwargs=dict(
                axis=axis,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            ),
        )

    def head(self, n: int = 5) -> "ColumnExpr":
        """
        Return the first n elements (lazy).

        The result is not executed until displayed or explicitly converted.
        This allows for SQL LIMIT optimization and consistent lazy behavior.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            ColumnExpr: Lazy wrapper that executes on display

        Example:
            >>> ds['age'].head(5)  # Lazy, no execution yet
            >>> print(ds['age'].head(5))  # Triggers execution
        """
        return ColumnExpr(source=self, method_name="head", method_args=(n,))

    def tail(self, n: int = 5) -> "ColumnExpr":
        """
        Return the last n elements (lazy).

        The result is not executed until displayed or explicitly converted.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            ColumnExpr: Lazy wrapper that executes on display

        Example:
            >>> ds['age'].tail(5)  # Lazy, no execution yet
            >>> print(ds['age'].tail(5))  # Triggers execution
        """
        return ColumnExpr(source=self, method_name="tail", method_args=(n,))

    # ========== Dynamic Method Delegation ==========

    def __getattr__(self, name: str):
        """
        Dynamic attribute access for all Expression methods.

        Delegates to self._expr and wraps Expression results in ColumnExpr.
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Try to get from underlying expression
        try:
            attr = getattr(self._expr, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        if callable(attr):
            # It's a method - wrap the result in ColumnExpr
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Wrap result in ColumnExpr if it's an Expression
                if isinstance(result, Expression):
                    return self._derive_expr(result)
                return result

            return wrapper
        else:
            # It's a property - wrap if Expression
            if isinstance(attr, Expression):
                return self._derive_expr(attr)
            return attr


class ColumnExprStringAccessor:
    """String accessor for ColumnExpr that returns ColumnExpr for each method."""

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.str

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        base_method = getattr(self._base_accessor, name)

        def wrapper(*args, **kwargs):
            result = base_method(*args, **kwargs)
            return ColumnExpr(result, self._column_expr._datastore)

        return wrapper

    def split(self, pat=None, *, n=-1, expand=False, regex=None):
        """
        Split strings around given separator/delimiter.

        When expand=True, returns a DataFrame with each split element as a column.
        When expand=False, returns a Series of arrays (uses SQL).

        Args:
            pat: String or regex to split on. Default is whitespace.
            n: Limit number of splits. -1 (default) for all.
            expand: If True, return DataFrame (requires pandas fallback).
                   If False, return Series of arrays.
            regex: If True, interpret pat as regex.

        Returns:
            ColumnExpr (wrapping DataFrame if expand=True, or Series of arrays)

        Example:
            >>> ds['text'].str.split('-')  # Returns Series of arrays
            >>> ds['text'].str.split('-', expand=True)  # Returns DataFrame
        """
        from .function_executor import function_config
        from .config import get_logger

        _logger = get_logger()

        # Check if pandas fallback is needed based on parameters
        if function_config.needs_accessor_fallback("str.split", expand=expand):
            reason = function_config.get_accessor_fallback_reason(
                "str.split", expand=expand
            )
            _logger.debug("[Accessor] %s", reason)

            n_val = n if n != -1 else None
            # Use operation descriptor mode (safe from recursion)
            return ColumnExpr.from_accessor_op(
                source=self._column_expr,
                accessor="str",
                method="split",
                args=(),
                kwargs={"pat": pat, "n": n_val, "expand": True, "regex": regex},
            )
        else:
            _logger.debug("[Accessor] str.split -> chDB SQL")
            # Use SQL-based split (returns array)
            result = self._base_accessor.split(pat, alias=None)
            return ColumnExpr(result, self._column_expr._datastore)

    def extract(self, pat, flags=0, expand=True):
        """
        Extract capture groups from regex pattern.

        For each subject string, extract groups from the first match of regex pat.

        Args:
            pat: Regular expression pattern with capturing groups.
            flags: Flags from the re module (ignored, for pandas compat).
            expand: If True, return DataFrame with one column per group.
                   If False, return Series.

        Returns:
            DataFrame or Series depending on expand parameter.

        Example:
            >>> ds['text'].str.extract(r'([a-z]+)_(\\d+)')  # Returns DataFrame
        """
        from .function_executor import function_config
        from .config import get_logger

        _logger = get_logger()
        reason = function_config.get_accessor_fallback_reason("str.extract")
        _logger.debug("[Accessor] %s", reason)

        # Use operation descriptor mode (safe from recursion)
        return ColumnExpr.from_accessor_op(
            source=self._column_expr,
            accessor="str",
            method="extract",
            args=(pat,),
            kwargs={"flags": flags, "expand": expand},
        )

    def extractall(self, pat, flags=0):
        """
        Extract capture groups from regex pattern for all matches.

        For each subject string, extract groups from all matches of regex pat.

        Args:
            pat: Regular expression pattern with capturing groups.
            flags: Flags from the re module (ignored, for pandas compat).

        Returns:
            DataFrame with MultiIndex (original index + match number).

        Example:
            >>> ds['text'].str.extractall(r'([a-z])(\\d)')
        """
        from .function_executor import function_config
        from .config import get_logger

        _logger = get_logger()
        reason = function_config.get_accessor_fallback_reason("str.extractall")
        _logger.debug("[Accessor] %s", reason)

        # Use operation descriptor mode (safe from recursion)
        return ColumnExpr.from_accessor_op(
            source=self._column_expr,
            accessor="str",
            method="extractall",
            args=(pat,),
            kwargs={"flags": flags},
        )

    def wrap(self, width, **kwargs):
        """
        Wrap long strings to be formatted in paragraphs.

        Args:
            width: Maximum line width.
            **kwargs: Additional arguments passed to textwrap.TextWrapper.

        Returns:
            Series with wrapped strings.

        Example:
            >>> ds['text'].str.wrap(10)
        """
        from .function_executor import function_config
        from .config import get_logger

        _logger = get_logger()
        reason = function_config.get_accessor_fallback_reason("str.wrap")
        _logger.debug("[Accessor] %s", reason)

        # Use operation descriptor mode (safe from recursion)
        return ColumnExpr.from_accessor_op(
            source=self._column_expr,
            accessor="str",
            method="wrap",
            args=(width,),
            kwargs=kwargs,
        )

    def __getitem__(self, index: int):
        """
        Get element at index from array result (e.g., after str.split()).

        This delegates to the base accessor's __getitem__ and wraps the result
        in a ColumnExpr for lazy evaluation.

        Args:
            index: 0-based index

        Example:
            >>> ds['text'].str.split().str[0]  # Get first word
        """
        result = self._base_accessor[index]
        return ColumnExpr(result, self._column_expr._datastore)

    def _execute_series(self):
        """Execute the column as a Pandas Series."""
        from .expressions import Field
        from .executor import get_executor

        ds = self._column_expr._datastore
        col_expr = self._column_expr._expr

        # Get the full DataFrame
        df = ds.to_df()

        # Get column name from expression
        # For Field expressions, use .name directly to get unquoted column name
        # str(col_expr) returns SQL representation with quotes (e.g., '"name"')
        # which won't match df.columns (e.g., 'name')
        if isinstance(col_expr, Field):
            col_name = col_expr.name
        else:
            col_name = str(col_expr)

        # Try to find the column in the DataFrame
        if col_name in df.columns:
            return df[col_name]

        # If not found directly, the expression might be complex
        # Use executor.execute_expression which preserves row order
        # using _row_id virtual column and ORDER BY clause
        executor = get_executor()
        sql_expr = col_expr.to_sql(quote_char='"')
        return executor.execute_expression(sql_expr, df)

    def cat(self, others=None, sep=None, na_rep=None, join="left"):
        """
        Concatenate strings in the Series/Index with given separator.

        Returns:
            Series with concatenated strings
        """
        series = self._execute_series()
        return series.str.cat(others=others, sep=sep, na_rep=na_rep, join=join)

    def extractall(self, pat, flags=0):
        """
        Extract all matches of pattern from each string.

        Returns:
            DataStore wrapping the MultiIndex DataFrame result
        """
        from .core import DataStore

        series = self._execute_series()
        result = series.str.extractall(pat, flags=flags)
        return DataStore.from_df(result)

    def get_dummies(self, sep="|"):
        """
        Return DataFrame of dummy/indicator variables for Series.

        Returns:
            DataStore wrapping the dummy DataFrame
        """
        from .core import DataStore

        series = self._execute_series()
        result = series.str.get_dummies(sep=sep)
        return DataStore.from_df(result)

    def partition(self, sep=" ", expand=True):
        """
        Split the string at the first occurrence of sep.

        Returns:
            DataStore wrapping the 3-column DataFrame (if expand=True)
            or Series of tuples (if expand=False)
        """
        from .core import DataStore

        series = self._execute_series()
        result = series.str.partition(sep=sep, expand=expand)
        if expand:
            return DataStore.from_df(result)
        return result

    def rpartition(self, sep=" ", expand=True):
        """
        Split the string at the last occurrence of sep.

        Returns:
            DataStore wrapping the 3-column DataFrame (if expand=True)
            or Series of tuples (if expand=False)
        """
        from .core import DataStore

        series = self._execute_series()
        result = series.str.rpartition(sep=sep, expand=expand)
        if expand:
            return DataStore.from_df(result)
        return result

    def __repr__(self) -> str:
        return f"ColumnExprStringAccessor({self._column_expr._expr!r})"


class ColumnExprJsonAccessor:
    """JSON accessor for ColumnExpr that returns ColumnExpr for each method.

    This wraps the base JsonAccessor and ensures all results are wrapped in
    ColumnExpr, maintaining pandas-like API compatibility (e.g., sort_values()).

    Some methods that return Array types (like json_extract_array_raw) fall back
    to pandas execution because chDB doesn't support Array inside Nullable type
    when using Python() table function.

    Example:
        >>> ds['data'].json.json_extract_string('name')  # Returns ColumnExpr
        >>> ds['data'].json.json_extract_string('name').sort_values()  # Works!
        >>> ds['data'].json.json_extract_string('name').str.upper()  # Chaining works!
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.json

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        from .function_executor import function_config
        from .config import get_logger

        _logger = get_logger()

        base_attr = getattr(self._base_accessor, name)

        # If it's a method, wrap the result
        if callable(base_attr):

            def wrapper(*args, **kwargs):
                # Check if this JSON method needs pandas fallback
                accessor_method = f"json.{name}"
                if function_config.needs_accessor_fallback(accessor_method):
                    reason = function_config.get_accessor_fallback_reason(
                        accessor_method
                    )
                    _logger.debug("[Accessor] %s", reason)

                    # Get the JSON path
                    json_path = args[0] if args else kwargs.get("path", "")

                    # Use operation descriptor mode (safe from recursion)
                    # Note: JSON extraction uses a special 'json_extract' method
                    # that the evaluator handles via _extract_json_array
                    result = ColumnExpr.from_accessor_op(
                        source=self._column_expr,
                        accessor="json",
                        method="_extract_array",  # Special method for JSON array extraction
                        args=(json_path,),
                        kwargs={},
                    )
                    return result
                else:
                    _logger.debug("[Accessor] json.%s -> chDB SQL", name)
                    result = base_attr(*args, **kwargs)
                    return ColumnExpr(result, self._column_expr._datastore)

            return wrapper
        else:
            # It's a property (like .length on some accessors)
            return ColumnExpr(base_attr, self._column_expr._datastore)

    def __repr__(self) -> str:
        return f"ColumnExprJsonAccessor({self._column_expr._expr!r})"


def _extract_json_array(series, json_path):
    """Extract array from JSON column using pure Python.

    This function is used by json.json_extract_array_raw fallback to pandas.
    """
    import json as json_module

    def extract_array(json_str):
        if json_str is None or (isinstance(json_str, float) and json_str != json_str):
            return None
        try:
            data = json_module.loads(json_str)
            # Navigate the path
            if json_path:
                for key in json_path.split("."):
                    if isinstance(data, dict):
                        data = data.get(key)
                    else:
                        return None
                    if data is None:
                        return None
            # Return as list (array)
            if isinstance(data, list):
                return data
            return None
        except (json_module.JSONDecodeError, TypeError):
            return None

    return series.apply(extract_array)


class ColumnExprArrayAccessor:
    """Array accessor for ColumnExpr that returns ColumnExpr for each method.

    This wraps the base ArrayAccessor and ensures all results are wrapped in
    ColumnExpr, maintaining pandas-like API compatibility (e.g., sort_values()).

    Example:
        >>> ds['nums'].arr.length  # Returns ColumnExpr
        >>> ds['nums'].arr.length.sort_values()  # Works!
        >>> ds['nums'].arr.has(5)  # Returns ColumnExpr
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.arr

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        base_attr = getattr(self._base_accessor, name)

        # If it's a method, wrap the result
        if callable(base_attr):

            def wrapper(*args, **kwargs):
                result = base_attr(*args, **kwargs)
                return ColumnExpr(result, self._column_expr._datastore)

            return wrapper
        else:
            # It's a property (like .length, .size, .empty, .not_empty)
            return ColumnExpr(base_attr, self._column_expr._datastore)

    def __repr__(self) -> str:
        return f"ColumnExprArrayAccessor({self._column_expr._expr!r})"


class ColumnExprUrlAccessor:
    """URL accessor for ColumnExpr that returns ColumnExpr for each method.

    This wraps the base UrlAccessor and ensures all results are wrapped in
    ColumnExpr, maintaining pandas-like API compatibility (e.g., sort_values()).

    Example:
        >>> ds['url'].url.domain()  # Returns ColumnExpr
        >>> ds['url'].url.domain().sort_values()  # Works!
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.url

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        base_attr = getattr(self._base_accessor, name)

        if callable(base_attr):

            def wrapper(*args, **kwargs):
                result = base_attr(*args, **kwargs)
                return ColumnExpr(result, self._column_expr._datastore)

            return wrapper
        else:
            return ColumnExpr(base_attr, self._column_expr._datastore)

    def __repr__(self) -> str:
        return f"ColumnExprUrlAccessor({self._column_expr._expr!r})"


class ColumnExprIpAccessor:
    """IP accessor for ColumnExpr that returns ColumnExpr for each method.

    This wraps the base IpAccessor and ensures all results are wrapped in
    ColumnExpr, maintaining pandas-like API compatibility (e.g., sort_values()).

    Example:
        >>> ds['ip'].ip.to_ipv4()  # Returns ColumnExpr
        >>> ds['ip'].ip.is_ipv4_string()  # Returns ColumnExpr
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.ip

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        base_attr = getattr(self._base_accessor, name)

        if callable(base_attr):

            def wrapper(*args, **kwargs):
                result = base_attr(*args, **kwargs)
                return ColumnExpr(result, self._column_expr._datastore)

            return wrapper
        else:
            return ColumnExpr(base_attr, self._column_expr._datastore)

    def __repr__(self) -> str:
        return f"ColumnExprIpAccessor({self._column_expr._expr!r})"


class ColumnExprGeoAccessor:
    """Geo accessor for ColumnExpr that returns ColumnExpr for each method.

    This wraps the base GeoAccessor and ensures all results are wrapped in
    ColumnExpr, maintaining pandas-like API compatibility (e.g., sort_values()).

    Example:
        >>> ds['vec'].geo.l2_norm()  # Returns ColumnExpr
        >>> ds['vec'].geo.l2_normalize()  # Returns ColumnExpr
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.geo

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        base_attr = getattr(self._base_accessor, name)

        if callable(base_attr):

            def wrapper(*args, **kwargs):
                result = base_attr(*args, **kwargs)
                return ColumnExpr(result, self._column_expr._datastore)

            return wrapper
        else:
            return ColumnExpr(base_attr, self._column_expr._datastore)

    def __repr__(self) -> str:
        return f"ColumnExprGeoAccessor({self._column_expr._expr!r})"


class ColumnExprDateTimeAccessor:
    """
    DateTime accessor for ColumnExpr that defers engine selection to execution time.

    All properties return ColumnExpr wrapping DateTimePropertyExpr, which is evaluated
    by ExpressionEvaluator at execution time. The evaluator checks function_config
    to determine whether to use chDB SQL functions or pandas .dt accessor.

    This design:
    1. Defers engine selection to execution time (not definition time)
    2. Supports function-level engine configuration via function_config
    3. Allows the same expression tree to be evaluated differently based on config

    Example:
        >>> ds['date'].dt.year  # Returns ColumnExpr with DateTimePropertyExpr
        >>> # At execution time:
        >>> # - If function_config.should_use_pandas('year'): s.dt.year
        >>> # - Otherwise: toYear(date_col) via chDB
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr

    def _make_property_expr(self, property_name: str) -> ColumnExpr:
        """Create a ColumnExpr wrapping DateTimePropertyExpr for lazy evaluation."""
        from .expressions import DateTimePropertyExpr

        expr = DateTimePropertyExpr(self._column_expr._expr, property_name)
        return ColumnExpr(expr, self._column_expr._datastore)

    def _make_method_expr(self, method_name: str, *args, **kwargs) -> ColumnExpr:
        """Create a ColumnExpr wrapping DateTimeMethodExpr for lazy evaluation."""
        from .expressions import DateTimeMethodExpr

        expr = DateTimeMethodExpr(self._column_expr._expr, method_name, args, kwargs)
        return ColumnExpr(expr, self._column_expr._datastore)

    @property
    def year(self) -> ColumnExpr:
        """Extract year from date/datetime. Engine selected at execution time."""
        return self._make_property_expr("year")

    @property
    def month(self) -> ColumnExpr:
        """Extract month from date/datetime (1-12). Engine selected at execution time."""
        return self._make_property_expr("month")

    @property
    def day(self) -> ColumnExpr:
        """Extract day of month from date/datetime (1-31). Engine selected at execution time."""
        return self._make_property_expr("day")

    @property
    def hour(self) -> ColumnExpr:
        """Extract hour from datetime (0-23). Engine selected at execution time."""
        return self._make_property_expr("hour")

    @property
    def minute(self) -> ColumnExpr:
        """Extract minute from datetime (0-59). Engine selected at execution time."""
        return self._make_property_expr("minute")

    @property
    def second(self) -> ColumnExpr:
        """Extract second from datetime (0-59). Engine selected at execution time."""
        return self._make_property_expr("second")

    @property
    def microsecond(self) -> ColumnExpr:
        """Extract microsecond from datetime. Always uses pandas (no chDB equivalent)."""
        return self._make_property_expr("microsecond")

    @property
    def nanosecond(self) -> ColumnExpr:
        """Extract nanosecond from datetime. Always uses pandas (no chDB equivalent)."""
        return self._make_property_expr("nanosecond")

    @property
    def dayofweek(self) -> ColumnExpr:
        """Return day of the week (Monday=0). Engine selected at execution time."""
        return self._make_property_expr("dayofweek")

    @property
    def weekday(self) -> ColumnExpr:
        """Alias for dayofweek."""
        return self._make_property_expr("weekday")

    @property
    def day_of_week(self) -> ColumnExpr:
        """Alias for dayofweek."""
        return self.dayofweek

    @property
    def dayofyear(self) -> ColumnExpr:
        """Return day of the year (1-366). Engine selected at execution time."""
        return self._make_property_expr("dayofyear")

    @property
    def day_of_year(self) -> ColumnExpr:
        """Alias for dayofyear."""
        return self.dayofyear

    @property
    def quarter(self) -> ColumnExpr:
        """Return quarter of the year (1-4). Engine selected at execution time."""
        return self._make_property_expr("quarter")

    @property
    def week(self) -> ColumnExpr:
        """Return ISO week number. Engine selected at execution time."""
        return self._make_property_expr("week")

    @property
    def weekofyear(self) -> ColumnExpr:
        """Alias for week."""
        return self._make_property_expr("weekofyear")

    @property
    def date(self) -> ColumnExpr:
        """Return date part. Engine selected at execution time."""
        return self._make_property_expr("date")

    @property
    def is_month_start(self) -> ColumnExpr:
        """Indicate whether the date is the first day of a month. Always uses pandas."""
        return self._make_property_expr("is_month_start")

    @property
    def is_month_end(self) -> ColumnExpr:
        """Indicate whether the date is the last day of a month. Always uses pandas."""
        return self._make_property_expr("is_month_end")

    @property
    def is_quarter_start(self) -> ColumnExpr:
        """Indicate whether the date is the first day of a quarter. Always uses pandas."""
        return self._make_property_expr("is_quarter_start")

    @property
    def is_quarter_end(self) -> ColumnExpr:
        """Indicate whether the date is the last day of a quarter. Always uses pandas."""
        return self._make_property_expr("is_quarter_end")

    @property
    def is_year_start(self) -> ColumnExpr:
        """Indicate whether the date is the first day of a year. Always uses pandas."""
        return self._make_property_expr("is_year_start")

    @property
    def is_year_end(self) -> ColumnExpr:
        """Indicate whether the date is the last day of a year. Always uses pandas."""
        return self._make_property_expr("is_year_end")

    @property
    def is_leap_year(self) -> ColumnExpr:
        """Indicate whether the date is in a leap year. Always uses pandas."""
        return self._make_property_expr("is_leap_year")

    @property
    def days_in_month(self) -> ColumnExpr:
        """Return number of days in month. Always uses pandas."""
        return self._make_property_expr("days_in_month")

    def strftime(self, fmt: str) -> ColumnExpr:
        """Format datetime as string. Engine selected at execution time."""
        return self._make_method_expr("strftime", fmt)

    def floor(self, freq: str) -> ColumnExpr:
        """Floor datetime to frequency. Always uses pandas."""
        return self._make_method_expr("floor_dt", freq)

    def ceil(self, freq: str) -> ColumnExpr:
        """Ceil datetime to frequency. Always uses pandas."""
        return self._make_method_expr("ceil_dt", freq)

    def round(self, freq: str) -> ColumnExpr:
        """Round datetime to frequency. Always uses pandas."""
        return self._make_method_expr("round_dt", freq)

    def tz_localize(self, tz) -> ColumnExpr:
        """Localize to timezone. Always uses pandas."""
        return self._make_method_expr("tz_localize", tz)

    def tz_convert(self, tz) -> ColumnExpr:
        """Convert to timezone. Always uses pandas."""
        return self._make_method_expr("tz_convert", tz)

    def to_period(self, freq) -> ColumnExpr:
        """Convert to PeriodIndex with specified frequency. Always uses pandas."""
        return self._make_method_expr("to_period", freq)

    def to_timestamp(self, freq=None, how="start") -> ColumnExpr:
        """Convert to DatetimeIndex from PeriodIndex. Always uses pandas."""
        return self._make_method_expr("to_timestamp", freq, how=how)

    def normalize(self) -> ColumnExpr:
        """Normalize times to midnight. Always uses pandas."""
        return self._make_method_expr("normalize")

    @property
    def time(self) -> ColumnExpr:
        """Return the time part of the datetime. Always uses pandas."""
        return self._make_property_expr("time")

    @property
    def daysinmonth(self) -> ColumnExpr:
        """Alias for days_in_month (pandas compatibility)."""
        return self.days_in_month

    @property
    def freq(self):
        """Return the frequency object if set, otherwise None. Executes immediately."""
        # freq returns scalar, not Series - execute immediately
        series = self._column_expr._execute()
        return series.dt.freq

    @property
    def tz(self):
        """Return the timezone. Executes immediately."""
        # tz returns scalar, not Series - execute immediately
        series = self._column_expr._execute()
        return series.dt.tz

    @property
    def unit(self):
        """Return the unit of the datetime. Executes immediately."""
        # unit returns scalar, not Series - execute immediately
        series = self._column_expr._execute()
        return series.dt.unit

    @property
    def timetz(self) -> ColumnExpr:
        """Return the time with timezone. Always uses pandas."""
        return self._make_property_expr("timetz")

    def as_unit(self, unit: str) -> ColumnExpr:
        """Convert to the specified unit. Always uses pandas."""
        return self._make_method_expr("as_unit", unit)

    def day_name(self, locale: str = None) -> ColumnExpr:
        """Return the day names (Monday, Tuesday, etc.). Always uses pandas."""
        if locale is not None:
            return self._make_method_expr("day_name", locale=locale)
        return self._make_method_expr("day_name")

    def month_name(self, locale: str = None) -> ColumnExpr:
        """Return the month names (January, February, etc.). Always uses pandas."""
        if locale is not None:
            return self._make_method_expr("month_name", locale=locale)
        return self._make_method_expr("month_name")

    def to_pydatetime(self) -> ColumnExpr:
        """Return an ndarray of python datetime objects. Always uses pandas."""
        return self._make_method_expr("to_pydatetime")

    def isocalendar(self) -> "ColumnExprIsoCalendarResult":
        """
        Return a DataFrame with ISO year, week number, and weekday.

        Returns:
            ColumnExprIsoCalendarResult with .year, .week, .day properties
        """
        return ColumnExprIsoCalendarResult(self._column_expr)

    def __repr__(self) -> str:
        return f"ColumnExprDateTimeAccessor({self._column_expr._expr!r})"


class ColumnExprIsoCalendarResult:
    """
    Result of dt.isocalendar() - provides access to ISO calendar components.

    Supports:
        - .year: ISO year
        - .week: ISO week number (1-53)
        - .day: ISO day of week (1=Monday, 7=Sunday)
        - Direct iteration/execution returns DataFrame with all three columns
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr

    def _make_isocalendar_component(self, component: str) -> ColumnExpr:
        """Create a ColumnExpr for an isocalendar component."""
        from .expressions import IsoCalendarComponentExpr

        expr = IsoCalendarComponentExpr(self._column_expr._expr, component)
        return ColumnExpr(expr, self._column_expr._datastore)

    @property
    def year(self) -> ColumnExpr:
        """ISO year."""
        return self._make_isocalendar_component("year")

    @property
    def week(self) -> ColumnExpr:
        """ISO week number (1-53)."""
        return self._make_isocalendar_component("week")

    @property
    def day(self) -> ColumnExpr:
        """ISO day of week (1=Monday, 7=Sunday)."""
        return self._make_isocalendar_component("day")

    @property
    def columns(self):
        """Return column names."""
        return ["year", "week", "day"]

    @property
    def values(self):
        """Return as numpy array."""
        return self._to_dataframe().values

    def __len__(self):
        """Trigger execution and return length."""
        return len(self._column_expr)

    def _to_dataframe(self):
        """Convert to DataFrame with year, week, day columns."""
        import pandas as pd

        return pd.DataFrame(
            {"year": self.year.values, "week": self.week.values, "day": self.day.values}
        )

    def __getitem__(self, key):
        """Allow column access like result['week']."""
        if key == "year":
            return self.year
        elif key == "week":
            return self.week
        elif key == "day":
            return self.day
        else:
            raise KeyError(f"Unknown column: {key}")

    def __repr__(self) -> str:
        return f"ColumnExprIsoCalendarResult({self._column_expr._expr!r})"


class ColumnExprILocIndexer:
    """
    Integer-location based indexer for ColumnExpr.

    Executes the underlying ColumnExpr and provides integer-location based
    selection like pandas Series.iloc. Supports:
    - Single integer: iloc[0] → scalar
    - Negative integer: iloc[-1] → scalar (last element)
    - Slice: iloc[0:3] → Series
    - List of integers: iloc[[0, 2]] → Series
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr

    def __getitem__(self, key):
        """
        Get elements by integer position.

        Args:
            key: Integer, slice, or list of integers

        Returns:
            Scalar value or pandas Series depending on key type
        """
        # Execute the column to get the actual Series
        series = self._column_expr._execute()

        # If result is a scalar, can only access index 0 or -1
        if not isinstance(series, (pd.Series, pd.DataFrame)):
            if isinstance(key, int) and key in (0, -1):
                return series
            raise IndexError(f"Cannot use iloc[{key}] on scalar result")

        # Delegate to pandas iloc
        return series.iloc[key]

    def __repr__(self) -> str:
        return f"ColumnExprILocIndexer({self._column_expr._expr!r})"


class ColumnExprLocIndexer:
    """
    Label-based indexer for ColumnExpr.

    Executes the underlying ColumnExpr and provides label-based selection
    like pandas Series.loc. Supports:
    - Single label: loc[0] → scalar (if 0 is in index)
    - Slice: loc['a':'c'] → Series
    - List of labels: loc[['a', 'b']] → Series
    """

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr

    def __getitem__(self, key):
        """
        Get elements by label.

        Args:
            key: Label, slice, or list of labels

        Returns:
            Scalar value or pandas Series depending on key type
        """
        # Execute the column to get the actual Series
        series = self._column_expr._execute()

        # If result is a scalar, can only access if key matches
        if not isinstance(series, (pd.Series, pd.DataFrame)):
            # For scalar results, we can't really do label-based access
            # but we can return the value for key=0 for compatibility
            if key == 0:
                return series
            raise KeyError(f"Cannot use loc[{key}] on scalar result")

        # Delegate to pandas loc
        return series.loc[key]

    def __repr__(self) -> str:
        return f"ColumnExprLocIndexer({self._column_expr._expr!r})"


# NOTE: BoolColumnExpr has been removed in favor of unified ColumnExpr.
# Comparison operators now return ColumnExpr wrapping Condition objects.
# This simplifies the architecture while maintaining full functionality:
# - (ds['col'] > 5).value_counts()  # Works
# - ds.filter(ds['col'] > 5)        # Works
# - (ds['a'] > 0) & (ds['b'] < 10)  # Works
# See ARCHITECTURE_PROPOSAL.md for design rationale.

# NOTE: LazyAggregate has been removed and merged into ColumnExpr.
# Aggregation methods (mean, sum, etc.) now return ColumnExpr in 'agg' mode.
# This simplifies the type system: users always get ColumnExpr from column operations.


# =============================================================================
# INJECT FUNCTION METHODS FROM REGISTRY
# =============================================================================
# This enables methods like abs(), round(), upper(), lower() to be called
# directly on ColumnExpr, supporting chaining even in method mode.
#
# Example: ds['a'].fillna(0).abs().round()
#          ^^^^^^^^^^^^^^^
#          Returns method-mode ColumnExpr with _expr=None
#                          ^^^^^^^^^^^^^^
#                          Now works! Uses injected method.


def _inject_column_expr_methods():
    """Inject function methods from registry into ColumnExpr class."""
    from .function_mixin import inject_methods_to_column_expr

    # Import function_definitions to ensure all functions are registered
    from . import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    # Inject all registered functions to ColumnExpr
    inject_methods_to_column_expr(ColumnExpr)


# Perform injection when module is loaded
_inject_column_expr_methods()
