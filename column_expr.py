"""
ColumnExpr - A column expression that can materialize when displayed.

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
from .utils import immutable

if TYPE_CHECKING:
    from .core import DataStore
    from .conditions import Condition, BinaryCondition
    from .lazy_result import LazySlice


class ColumnExpr:
    """
    A column expression that wraps an underlying Expression and can materialize.

    When displayed (via __repr__, __str__, or IPython), it materializes the
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

        >>> ds['age'] > 25  # Returns Condition (for filtering)
        BinaryCondition('"age" > 25')
    """

    def __init__(
        self,
        expr: Expression,
        datastore: 'DataStore',
        alias: Optional[str] = None,
        groupby_fields: Optional[List] = None,
    ):
        """
        Initialize ColumnExpr with expression and DataStore reference.

        Args:
            expr: The underlying expression (Field, ArithmeticExpression, Function, etc.)
            datastore: Reference to the DataStore for materialization
            alias: Optional alias for the expression
            groupby_fields: Optional groupby fields from LazyGroupBy (to avoid polluting DataStore state)
        """
        self._expr = expr
        self._datastore = datastore
        self._alias = alias
        self._groupby_fields = groupby_fields or []

    @property
    def expr(self) -> Expression:
        """Get the underlying expression."""
        return self._expr

    @property
    def datastore(self) -> 'DataStore':
        """Get the DataStore reference."""
        return self._datastore

    @property
    def alias(self) -> Optional[str]:
        """Get the alias."""
        return self._alias

    # ========== Materialization ==========

    def _materialize(self) -> pd.Series:
        """
        Materialize this expression and return a pandas Series.

        This executes the expression against the DataStore's data using the
        unified ExpressionEvaluator, which respects function_config settings.
        """
        from .expression_evaluator import ExpressionEvaluator

        # Get the materialized DataFrame from the DataStore
        df = self._datastore._materialize()

        # Use unified expression evaluator (respects function_config)
        evaluator = ExpressionEvaluator(df, self._datastore)
        result = evaluator.evaluate(self._expr)

        # Ensure we return a Series with proper index
        if isinstance(result, pd.Series):
            return result
        elif isinstance(self._expr, Literal):
            # Literal value - expand to Series
            return pd.Series([result] * len(df), index=df.index)
        else:
            # Scalar result - return as is (will be wrapped if needed)
            return result

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

        # Get the groupby fields from the datastore and clear them before materialization
        # This prevents _materialize() from generating SQL with GROUP BY
        groupby_fields = self._datastore._groupby_fields.copy()
        self._datastore._groupby_fields = []

        # Get the materialized DataFrame from the DataStore (without groupby applied)
        df = self._datastore._materialize()

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
            groupby_sql = ', '.join(f'"{name}"' for name in groupby_col_names)

            # Use -If suffix to skip NaN values (matching pandas skipna=True behavior)
            # In chDB, pandas NaN is recognized as isNaN(), not isNull()
            if skipna:
                # Use aggregate function with If suffix to skip NaN values
                agg_sql = f'{agg_func_name}If({col_expr_sql}, NOT isNaN({col_expr_sql}))'
            else:
                agg_sql = f'{agg_func_name}({col_expr_sql})'

            sql = f'SELECT {groupby_sql}, {agg_sql} AS __agg_result__ FROM __df__ GROUP BY {groupby_sql}'

            # Execute via chDB
            executor = get_executor()
            result_df = executor.query_dataframe(sql, df)

            # Convert result to Series with proper index
            if len(groupby_col_names) == 1:
                # Single groupby column - use it as index
                result_series = result_df.set_index(groupby_col_names[0])['__agg_result__']
                result_series.name = self._get_column_name()
            else:
                # Multiple groupby columns - use MultiIndex
                result_series = result_df.set_index(groupby_col_names)['__agg_result__']
                result_series.name = self._get_column_name()

            # Sort by index to match pandas default behavior (sort=True)
            result_series = result_series.sort_index()

        return result_series

    def _get_column_name(self) -> str:
        """Get the column name for this expression."""
        if self._alias:
            return self._alias
        if isinstance(self._expr, Field):
            return self._expr.name
        return self._expr.to_sql(quote_char='"')

    # ========== Display Methods ==========

    def __repr__(self) -> str:
        """Return a representation that shows actual values."""
        try:
            series = self._materialize()
            return repr(series)
        except Exception as e:
            return f"ColumnExpr({self._expr!r}) [Error: {e}]"

    def __str__(self) -> str:
        """Return string representation showing actual values."""
        try:
            series = self._materialize()
            return str(series)
        except Exception:
            return self._expr.to_sql()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            series = self._materialize()
            if hasattr(series, '_repr_html_'):
                return series._repr_html_()
            return f"<pre>{repr(series)}</pre>"
        except Exception as e:
            return f"<pre>ColumnExpr({self._expr.to_sql()}) [Error: {e}]</pre>"

    # ========== Expression Interface (delegation) ==========

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for the underlying expression."""
        return self._expr.to_sql(quote_char=quote_char, **kwargs)

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield from self._expr.nodes()

    # ========== Comparison Operators (Return Conditions for filtering) ==========

    def __eq__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('=', self._expr, Expression.wrap(other))

    def __ne__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('!=', self._expr, Expression.wrap(other))

    def __gt__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('>', self._expr, Expression.wrap(other))

    def __ge__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('>=', self._expr, Expression.wrap(other))

    def __lt__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('<', self._expr, Expression.wrap(other))

    def __le__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('<=', self._expr, Expression.wrap(other))

    # ========== Pandas-style Comparison Methods ==========

    def eq(self, other: Any) -> pd.Series:
        """
        Element-wise equality comparison, returns boolean Series.

        Unlike __eq__ which returns a Condition for filtering,
        this method materializes and returns a pandas boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            pd.Series: Boolean Series indicating equality

        Example:
            >>> ds['value'].eq(5)
            0    False
            1     True
            2    False
            dtype: bool
        """
        series = self._materialize()
        if isinstance(other, ColumnExpr):
            other = other._materialize()
        return series.eq(other)

    def ne(self, other: Any) -> pd.Series:
        """
        Element-wise not-equal comparison, returns boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            pd.Series: Boolean Series indicating inequality

        Example:
            >>> ds['value'].ne(5)
        """
        series = self._materialize()
        if isinstance(other, ColumnExpr):
            other = other._materialize()
        return series.ne(other)

    def lt(self, other: Any) -> pd.Series:
        """
        Element-wise less-than comparison, returns boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            pd.Series: Boolean Series

        Example:
            >>> ds['value'].lt(5)
        """
        series = self._materialize()
        if isinstance(other, ColumnExpr):
            other = other._materialize()
        return series.lt(other)

    def le(self, other: Any) -> pd.Series:
        """
        Element-wise less-than-or-equal comparison, returns boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            pd.Series: Boolean Series

        Example:
            >>> ds['value'].le(5)
        """
        series = self._materialize()
        if isinstance(other, ColumnExpr):
            other = other._materialize()
        return series.le(other)

    def gt(self, other: Any) -> pd.Series:
        """
        Element-wise greater-than comparison, returns boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            pd.Series: Boolean Series

        Example:
            >>> ds['value'].gt(5)
        """
        series = self._materialize()
        if isinstance(other, ColumnExpr):
            other = other._materialize()
        return series.gt(other)

    def ge(self, other: Any) -> pd.Series:
        """
        Element-wise greater-than-or-equal comparison, returns boolean Series.

        Args:
            other: Value or Series to compare with

        Returns:
            pd.Series: Boolean Series

        Example:
            >>> ds['value'].ge(5)
        """
        series = self._materialize()
        if isinstance(other, ColumnExpr):
            other = other._materialize()
        return series.ge(other)

    # ========== Logical Operators (for combining boolean ColumnExpr with Conditions) ==========

    def __and__(self, other: Any) -> 'Condition':
        """
        Combine with AND operator.

        Allows combining boolean ColumnExpr (like isnull()) with Conditions.
        Converts self to a condition (expr = 1) before combining.

        Example:
            >>> ds.filter(ds['email'].isnull() & (ds['status'] == 'active'))
        """
        from .conditions import CompoundCondition, BinaryCondition
        from .expressions import Literal

        # Convert self to condition: self._expr = 1
        self_cond = BinaryCondition('=', self._expr, Literal(1))

        # Handle other operand
        if isinstance(other, ColumnExpr):
            other_cond = BinaryCondition('=', other._expr, Literal(1))
        else:
            other_cond = other

        return CompoundCondition('AND', self_cond, other_cond)

    def __rand__(self, other: Any) -> 'Condition':
        """Right AND operator."""
        from .conditions import CompoundCondition, BinaryCondition
        from .expressions import Literal

        self_cond = BinaryCondition('=', self._expr, Literal(1))
        return CompoundCondition('AND', other, self_cond)

    def __or__(self, other: Any) -> 'Condition':
        """
        Combine with OR operator.

        Allows combining boolean ColumnExpr (like isnull()) with Conditions.
        Converts self to a condition (expr = 1) before combining.

        Example:
            >>> ds.filter(ds['email'].isnull() | ds['phone'].isnull())
        """
        from .conditions import CompoundCondition, BinaryCondition
        from .expressions import Literal

        # Convert self to condition: self._expr = 1
        self_cond = BinaryCondition('=', self._expr, Literal(1))

        # Handle other operand
        if isinstance(other, ColumnExpr):
            other_cond = BinaryCondition('=', other._expr, Literal(1))
        else:
            other_cond = other

        return CompoundCondition('OR', self_cond, other_cond)

    def __ror__(self, other: Any) -> 'Condition':
        """Right OR operator."""
        from .conditions import CompoundCondition, BinaryCondition
        from .expressions import Literal

        self_cond = BinaryCondition('=', self._expr, Literal(1))
        return CompoundCondition('OR', other, self_cond)

    def __invert__(self) -> 'Condition':
        """
        Negate with NOT operator.

        For boolean ColumnExpr (like isnull()), returns NOT(expr = 1).

        Example:
            >>> ds.filter(~ds['email'].isnull())  # Equivalent to notnull()
        """
        from .conditions import NotCondition, BinaryCondition
        from .expressions import Literal

        self_cond = BinaryCondition('=', self._expr, Literal(1))
        return NotCondition(self_cond)

    # ========== Arithmetic Operators (Return ColumnExpr) ==========

    def __add__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('+', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __radd__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('+', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __sub__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('-', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __rsub__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('-', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __mul__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('*', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __rmul__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('*', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __truediv__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('/', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __rtruediv__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('/', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __floordiv__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('//', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __rfloordiv__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('//', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __mod__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('%', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __rmod__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('%', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __pow__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('**', self._expr, Expression.wrap(other))
        return ColumnExpr(new_expr, self._datastore)

    def __rpow__(self, other: Any) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('**', Expression.wrap(other), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __neg__(self) -> 'ColumnExpr':
        new_expr = ArithmeticExpression('-', Literal(0), self._expr)
        return ColumnExpr(new_expr, self._datastore)

    def __round__(self, ndigits: Optional[int] = None) -> 'ColumnExpr':
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
            return ColumnExpr(Function('round', self._expr), self._datastore)
        return ColumnExpr(Function('round', self._expr, Literal(ndigits)), self._datastore)

    # ========== Accessor Properties ==========

    @property
    def str(self) -> 'ColumnExprStringAccessor':
        """Accessor for string functions."""
        return ColumnExprStringAccessor(self)

    @property
    def dt(self) -> 'ColumnExprDateTimeAccessor':
        """Accessor for date/time functions."""
        return ColumnExprDateTimeAccessor(self)

    # ========== Condition Methods (for filtering) ==========

    def isnull(self) -> 'ColumnExpr':
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
            >>> ds['value'].isnull()  # Returns boolean Series when materialized

        Returns:
            ColumnExpr: Expression that evaluates to True (NULL/NaN) or False
        """
        from .functions import Function

        return ColumnExpr(Function('isNull', self._expr), self._datastore)

    def notnull(self) -> 'ColumnExpr':
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
            >>> ds['value'].notnull()  # Returns boolean Series when materialized

        Returns:
            ColumnExpr: Expression that evaluates to True (not NULL/NaN) or False
        """
        from .functions import Function

        return ColumnExpr(Function('isNotNull', self._expr), self._datastore)

    # Aliases for pandas compatibility
    def isna(self) -> 'ColumnExpr':
        """Alias for isnull() - pandas compatibility."""
        return self.isnull()

    def notna(self) -> 'ColumnExpr':
        """Alias for notnull() - pandas compatibility."""
        return self.notnull()

    def isnull_condition(self) -> 'Condition':
        """Create IS NULL condition for filtering (SQL style)."""
        return self._expr.isnull()

    def notnull_condition(self) -> 'Condition':
        """Create IS NOT NULL condition for filtering (SQL style)."""
        return self._expr.notnull()

    def isin(self, values) -> 'Condition':
        """Create IN condition."""
        return self._expr.isin(values)

    def notin(self, values) -> 'Condition':
        """Create NOT IN condition."""
        return self._expr.notin(values)

    def between(self, lower, upper) -> 'Condition':
        """Create BETWEEN condition."""
        return self._expr.between(lower, upper)

    def like(self, pattern: str) -> 'Condition':
        """Create LIKE condition."""
        return self._expr.like(pattern)

    def notlike(self, pattern: str) -> 'Condition':
        """Create NOT LIKE condition."""
        return self._expr.notlike(pattern)

    def ilike(self, pattern: str) -> 'Condition':
        """Create ILIKE condition (case-insensitive)."""
        return self._expr.ilike(pattern)

    # ========== Alias Support ==========

    @immutable
    def as_(self, alias: str) -> 'ColumnExpr':
        """Set an alias for this expression."""
        new_expr = self._expr.as_(alias)
        return ColumnExpr(new_expr, self._datastore, alias=alias)

    # ========== Pandas Series Methods (for compatibility) ==========

    @property
    def values(self) -> Any:
        """Return underlying numpy array (property for pandas compatibility)."""
        return self._materialize().values

    @property
    def name(self) -> Optional[str]:
        """Return the name of the column if it's a simple Field."""
        if isinstance(self._expr, Field):
            return self._expr.name
        return None

    def __len__(self) -> int:
        """Return length of the materialized series."""
        return len(self._materialize())

    def __iter__(self):
        """Iterate over materialized values."""
        return iter(self._materialize())

    def __getitem__(self, key):
        """
        Support indexing/subscripting like pandas Series.

        This allows patterns like: df['col'].mode()[0]

        Args:
            key: Index, slice, or array of indices

        Returns:
            Single value or Series depending on key type
        """
        return self._materialize()[key]

    def tolist(self) -> list:
        """Convert to Python list."""
        return self._materialize().tolist()

    def to_numpy(self):
        """Convert to numpy array."""
        return self._materialize().to_numpy()

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
        arr = self._materialize().values
        if dtype is not None:
            arr = arr.astype(dtype)
        # Handle copy parameter for numpy 2.0+ compatibility
        if copy:
            import numpy as np

            arr = np.array(arr, copy=True)
        return arr

    # ========== Aggregate Methods ==========
    # These methods return ColumnExpr for SQL when called with default args,
    # or materialize and compute when called with pandas/numpy-style args.

    def mean(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute mean of the column.

        Returns a LazyAggregate that:
        - Displays the result when shown in notebook/REPL
        - Can be used in agg() for SQL building
        - Returns Series with groupby, scalar without

        Args:
            axis: Axis for computation (pandas/numpy compatibility)
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display

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
        return LazyAggregate(self, 'avg', 'mean', skipna=skipna, axis=axis, numeric_only=numeric_only, **kwargs)

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

        return ColumnExpr(AggregateFunction('avg', self._expr), self._datastore)

    def sum(self, axis=None, skipna=True, numeric_only=False, min_count=0, **kwargs):
        """
        Compute sum of the column.

        Returns a LazyAggregate that executes on display.

        Args:
            axis: Axis for computation (pandas/numpy compatibility)
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            min_count: Minimum count of valid values
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        return LazyAggregate(
            self, 'sum', 'sum', skipna=skipna, axis=axis, numeric_only=numeric_only, min_count=min_count, **kwargs
        )

    def sum_sql(self):
        """Return SUM() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return ColumnExpr(AggregateFunction('sum', self._expr), self._datastore)

    def std(self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs):
        """
        Compute standard deviation of the column.

        Returns a LazyAggregate that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            ddof: Delta degrees of freedom (1=sample, 0=population)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        # Use sample std by default (ddof=1)
        func_name = 'stddevSamp' if ddof == 1 else 'stddevPop'
        return LazyAggregate(
            self, func_name, 'std', skipna=skipna, axis=axis, ddof=ddof, numeric_only=numeric_only, **kwargs
        )

    def std_sql(self, sample=True):
        """Return stddevSamp() or stddevPop() SQL expression for use in select()."""
        from .functions import AggregateFunction

        func_name = 'stddevSamp' if sample else 'stddevPop'
        return ColumnExpr(AggregateFunction(func_name, self._expr), self._datastore)

    def var(self, axis=None, skipna=True, ddof=1, numeric_only=False, **kwargs):
        """
        Compute variance of the column.

        Returns a LazyAggregate that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            ddof: Delta degrees of freedom (1=sample, 0=population)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        # Use sample var by default (ddof=1)
        func_name = 'varSamp' if ddof == 1 else 'varPop'
        return LazyAggregate(
            self, func_name, 'var', skipna=skipna, axis=axis, ddof=ddof, numeric_only=numeric_only, **kwargs
        )

    def var_sql(self, sample=True):
        """Return varSamp() or varPop() SQL expression for use in select()."""
        from .functions import AggregateFunction

        func_name = 'varSamp' if sample else 'varPop'
        return ColumnExpr(AggregateFunction(func_name, self._expr), self._datastore)

    def min(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute minimum of the column.

        Returns a LazyAggregate that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        return LazyAggregate(self, 'min', 'min', skipna=skipna, axis=axis, numeric_only=numeric_only, **kwargs)

    def min_sql(self):
        """Return MIN() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return ColumnExpr(AggregateFunction('min', self._expr), self._datastore)

    def max(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute maximum of the column.

        Returns a LazyAggregate that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        return LazyAggregate(self, 'max', 'max', skipna=skipna, axis=axis, numeric_only=numeric_only, **kwargs)

    def max_sql(self):
        """Return MAX() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return ColumnExpr(AggregateFunction('max', self._expr), self._datastore)

    def count(self):
        """
        Count non-NA values in the column.

        Returns a LazyAggregate that executes on display.

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        return LazyAggregate(self, 'count', 'count', skipna=True)

    def count_sql(self):
        """Return COUNT() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return ColumnExpr(AggregateFunction('count', self._expr), self._datastore)

    def median(self, axis=None, skipna=True, numeric_only=False, **kwargs):
        """
        Compute median of the column.

        Returns a LazyAggregate that executes on display.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values (default True)
            numeric_only: Include only numeric columns
            **kwargs: Additional pandas arguments

        Returns:
            LazyAggregate: Lazy aggregate that executes on display
        """
        return LazyAggregate(self, 'median', 'median', skipna=skipna, axis=axis, numeric_only=numeric_only, **kwargs)

    def median_sql(self):
        """Return median() SQL expression for use in select()."""
        from .functions import AggregateFunction

        return ColumnExpr(AggregateFunction('median', self._expr), self._datastore)

    def prod(self, axis=None, skipna=True, numeric_only=False, min_count=0, **kwargs):
        """
        Compute product of values.

        Note: No direct SQL equivalent, always materializes.

        Args:
            axis: Axis for computation
            skipna: Whether to skip NA values
            numeric_only: Include only numeric columns
            min_count: Minimum count of valid values
            **kwargs: Additional pandas arguments

        Returns:
            scalar: Product of values
        """
        series = self._materialize()
        return series.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    def cumsum(self, axis=None, dtype=None, out=None, *, skipna=True, **kwargs):
        """
        Compute cumulative sum of the column.

        Args:
            axis: NumPy axis parameter
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            Series: Cumulative sum
        """
        series = self._materialize()
        return series.cumsum(axis=axis, skipna=skipna, **kwargs)

    def cumprod(self, axis=None, dtype=None, out=None, *, skipna=True, **kwargs):
        """
        Compute cumulative product of the column.

        Args:
            axis: NumPy axis parameter
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            Series: Cumulative product
        """
        series = self._materialize()
        return series.cumprod(axis=axis, skipna=skipna, **kwargs)

    def median(self, axis=None, out=None, overwrite_input=False, keepdims=False, *, skipna=True, **kwargs):
        """
        Compute the median of the column.

        Args:
            axis: NumPy axis parameter (enables np.median() compatibility)
            out: NumPy out parameter (not supported)
            overwrite_input: NumPy parameter (ignored)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            float: The computed median
        """
        series = self._materialize()
        return series.median(axis=axis, skipna=skipna, **kwargs)

    def mode(self, dropna: bool = True):
        """
        Return the mode(s) of the column.

        The mode is the value that appears most frequently.
        Materializes the column and computes mode using pandas.

        Args:
            dropna: Don't consider NaN/NaT values (default True)

        Returns:
            pd.Series: Series containing the mode value(s)

        Example:
            >>> ds['category'].mode()
            0    A
            dtype: object

            >>> ds['category'].mode()[0]  # Get first mode value
            'A'
        """
        series = self._materialize()
        return series.mode(dropna=dropna)

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
        series = self._materialize()
        return series.argmin(axis=axis, skipna=skipna, **kwargs)

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
        series = self._materialize()
        return series.argmax(axis=axis, skipna=skipna, **kwargs)

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
        series = self._materialize()
        return series.any(axis=axis, skipna=skipna, **kwargs)

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
        series = self._materialize()
        return series.all(axis=axis, skipna=skipna, **kwargs)

    # ========== Pandas Series Methods ==========

    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        """
        Apply a function to each element of the column.

        This method materializes the column and applies the function element-wise,
        similar to pandas Series.apply().

        Args:
            func: Function to apply to each element
            convert_dtype: Try to find better dtype for results (default True)
            args: Positional arguments to pass to func
            **kwargs: Additional keyword arguments to pass to func

        Returns:
            pd.Series: Series with the function applied

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
        series = self._materialize()
        return series.apply(func, convert_dtype=convert_dtype, args=args, **kwargs)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ):
        """
        Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.

        Args:
            normalize: If True, return relative frequencies instead of counts
            sort: Sort by frequencies (default True)
            ascending: Sort in ascending order (default False)
            bins: Group values into half-open bins (for numeric data)
            dropna: Don't include counts of NaN (default True)

        Returns:
            pd.Series: Series with value counts

        Example:
            >>> ds = DataStore.from_file('data.csv')
            >>> ds['category'].value_counts()
            A    150
            B    100
            C     50
            Name: category, dtype: int64

            >>> ds['category'].value_counts(normalize=True)
            A    0.50
            B    0.33
            C    0.17
            Name: category, dtype: float64
        """
        series = self._materialize()
        return series.value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
        )

    def unique(self):
        """
        Return unique values of the column.

        Returns:
            numpy.ndarray: Unique values in order of appearance

        Example:
            >>> ds['category'].unique()
            array(['A', 'B', 'C'], dtype=object)
        """
        series = self._materialize()
        return series.unique()

    def nunique(self, dropna: bool = True):
        """
        Return number of unique values.

        Args:
            dropna: Don't include NaN in the count (default True)

        Returns:
            int: Number of unique values

        Example:
            >>> ds['category'].nunique()
            3
        """
        series = self._materialize()
        return series.nunique(dropna=dropna)

    def map(self, arg, na_action=None):
        """
        Map values of Series according to input mapping or function.

        Args:
            arg: Mapping correspondence (dict, Series, or function)
            na_action: If 'ignore', propagate NaN values without passing to mapping

        Returns:
            pd.Series: Series with mapped values

        Example:
            >>> ds['grade'].map({'A': 4.0, 'B': 3.0, 'C': 2.0})
            0    4.0
            1    3.0
            2    2.0
            Name: grade, dtype: float64
        """
        series = self._materialize()
        return series.map(arg, na_action=na_action)

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None):
        """
        Fill NA/NaN values using pandas.

        Materializes the column and uses pandas fillna() for proper NaN handling.
        This ensures compatibility with pandas behavior for both numeric and string columns.

        Args:
            value: Value to use to fill holes
            method: Method to use for filling holes ('ffill', 'bfill')
            axis: Axis along which to fill (0 or 'index')
            inplace: Not supported, always returns new Series
            limit: Maximum number of consecutive NaN values to fill

        Returns:
            pd.Series: Series with NA values filled

        Example:
            >>> ds['value'].fillna(0)
            >>> ds['Cabin'].fillna('Unknown')
        """
        if inplace:
            raise ValueError("ColumnExpr is immutable, inplace=True is not supported")

        # Handle ColumnExpr, LazyAggregate, or other values - materialize them first
        fill_value = value
        if isinstance(value, LazyAggregate):
            # Execute LazyAggregate to get the scalar value
            result = value._execute()
            if isinstance(result, pd.Series):
                fill_value = result.iloc[0] if len(result) == 1 else result
            else:
                fill_value = result
        elif isinstance(value, ColumnExpr):
            # Materialize to get actual value(s)
            materialized = value._materialize()
            # Handle scalar results (e.g., from round(agg_func(...)))
            if not isinstance(materialized, pd.Series):
                fill_value = materialized
            # If it's a single value (aggregate), extract scalar
            elif len(materialized) == 1:
                fill_value = materialized.iloc[0]
            else:
                fill_value = materialized

        # Always use pandas fillna for proper NaN handling
        series = self._materialize()
        return series.fillna(value=fill_value, method=method, axis=axis, limit=limit)

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

        return ColumnExpr(Function('ifNull', self._expr, fill_value), self._datastore)

    def _contains_aggregate(self, expr) -> bool:
        """Check if an expression contains aggregate functions."""
        from .functions import Function, AggregateFunction

        if isinstance(expr, AggregateFunction):
            return True

        if isinstance(expr, Function):
            # Check function name for common aggregates
            agg_names = {
                'avg',
                'sum',
                'count',
                'min',
                'max',
                'median',
                'stddevSamp',
                'stddevPop',
                'varSamp',
                'varPop',
                'any',
                'argMin',
                'argMax',
                'groupArray',
            }
            if expr.name.lower() in agg_names:
                return True
            # Recursively check arguments
            for arg in expr.args:
                if self._contains_aggregate(arg):
                    return True

        # Check other expression types that might contain nested expressions
        if hasattr(expr, 'args'):
            for arg in expr.args:
                if self._contains_aggregate(arg):
                    return True

        return False

    def dropna(self):
        """
        Return Series with missing values removed.

        Returns:
            pd.Series: Series with NA values removed

        Example:
            >>> ds['value'].dropna()
        """
        series = self._materialize()
        return series.dropna()

    def ffill(self, axis=None, inplace=False, limit=None, limit_area=None):
        """
        Fill NA/NaN values by propagating the last valid observation forward.

        Args:
            axis: Not used, for pandas compatibility
            inplace: Not supported (ColumnExpr is immutable)
            limit: Maximum number of consecutive NaN values to forward fill
            limit_area: Restrict filling to 'inside' or 'outside' values

        Returns:
            pd.Series: Series with forward-filled values

        Example:
            >>> ds['value'].ffill()
        """
        if inplace:
            raise ValueError("ColumnExpr is immutable, inplace=True is not supported")
        series = self._materialize()
        return series.ffill(axis=axis, limit=limit, limit_area=limit_area)

    def bfill(self, axis=None, inplace=False, limit=None, limit_area=None):
        """
        Fill NA/NaN values by propagating the next valid observation backward.

        Args:
            axis: Not used, for pandas compatibility
            inplace: Not supported (ColumnExpr is immutable)
            limit: Maximum number of consecutive NaN values to backward fill
            limit_area: Restrict filling to 'inside' or 'outside' values

        Returns:
            pd.Series: Series with backward-filled values

        Example:
            >>> ds['value'].bfill()
        """
        if inplace:
            raise ValueError("ColumnExpr is immutable, inplace=True is not supported")
        series = self._materialize()
        return series.bfill(axis=axis, limit=limit, limit_area=limit_area)

    def interpolate(
        self, method='linear', axis=0, limit=None, inplace=False, limit_direction=None, limit_area=None, **kwargs
    ):
        """
        Fill NaN values using an interpolation method.

        Args:
            method: Interpolation method ('linear', 'index', 'pad', etc.)
            axis: Axis to interpolate along
            limit: Maximum number of consecutive NaNs to fill
            inplace: Not supported (ColumnExpr is immutable)
            limit_direction: Direction to fill ('forward', 'backward', 'both')
            limit_area: Restrict filling to 'inside' or 'outside' values
            **kwargs: Additional arguments passed to pandas interpolate

        Returns:
            pd.Series: Series with interpolated values

        Example:
            >>> ds['value'].interpolate()
            >>> ds['value'].interpolate(method='polynomial', order=2)
        """
        if inplace:
            raise ValueError("ColumnExpr is immutable, inplace=True is not supported")
        series = self._materialize()
        return series.interpolate(
            method=method, axis=axis, limit=limit, limit_direction=limit_direction, limit_area=limit_area, **kwargs
        )

    def astype(self, dtype, copy=True, errors='raise'):
        """
        Cast to a specified dtype.

        Args:
            dtype: Data type to cast to
            copy: Return a copy (default True)
            errors: Control raising of exceptions ('raise' or 'ignore')

        Returns:
            pd.Series: Series with new dtype

        Example:
            >>> ds['age'].astype(float)
        """
        series = self._materialize()
        return series.astype(dtype, copy=copy, errors=errors)

    def sort_values(
        self, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None
    ):
        """
        Sort by the values.

        Args:
            axis: Axis to sort along
            ascending: Sort ascending vs. descending
            inplace: Not supported
            kind: Sort algorithm
            na_position: Position of NaN values ('first' or 'last')
            ignore_index: If True, the resulting axis will be labeled 0, 1, …, n - 1
            key: Apply the key function to values before sorting

        Returns:
            pd.Series: Sorted Series
        """
        if inplace:
            raise ValueError("ColumnExpr is immutable, inplace=True is not supported")
        series = self._materialize()
        return series.sort_values(
            axis=axis, ascending=ascending, kind=kind, na_position=na_position, ignore_index=ignore_index, key=key
        )

    def head(self, n: int = 5) -> 'LazySlice':
        """
        Return the first n elements (lazy).

        The result is not materialized until displayed or explicitly converted.
        This allows for SQL LIMIT optimization and consistent lazy behavior.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            LazySlice: Lazy wrapper that materializes on display

        Example:
            >>> ds['age'].head(5)  # Lazy, no execution yet
            >>> print(ds['age'].head(5))  # Triggers execution
        """
        from .lazy_result import LazySlice

        return LazySlice(self, 'head', n)

    def tail(self, n: int = 5) -> 'LazySlice':
        """
        Return the last n elements (lazy).

        The result is not materialized until displayed or explicitly converted.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            LazySlice: Lazy wrapper that materializes on display

        Example:
            >>> ds['age'].tail(5)  # Lazy, no execution yet
            >>> print(ds['age'].tail(5))  # Triggers execution
        """
        from .lazy_result import LazySlice

        return LazySlice(self, 'tail', n)

    # ========== Dynamic Method Delegation ==========

    def __getattr__(self, name: str):
        """
        Dynamic attribute access for all Expression methods.

        Delegates to self._expr and wraps Expression results in ColumnExpr.
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Try to get from underlying expression
        try:
            attr = getattr(self._expr, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if callable(attr):
            # It's a method - wrap the result in ColumnExpr
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # Wrap result in ColumnExpr if it's an Expression
                if isinstance(result, Expression):
                    return ColumnExpr(result, self._datastore)
                return result

            return wrapper
        else:
            # It's a property - wrap if Expression
            if isinstance(attr, Expression):
                return ColumnExpr(attr, self._datastore)
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

    def _materialize_series(self):
        """Materialize the column as a Pandas Series."""
        ds = self._column_expr._datastore
        col_expr = self._column_expr._expr

        # Get the full DataFrame
        df = ds.to_df()

        # Get column name from expression
        col_name = str(col_expr)

        # Try to find the column in the DataFrame
        if col_name in df.columns:
            return df[col_name]

        # If not found directly, the expression might be complex
        # In that case, execute a select with this column
        result = ds.select(col_expr).to_df()
        return result.iloc[:, 0]

    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        """
        Concatenate strings in the Series/Index with given separator.

        Returns:
            Series with concatenated strings
        """
        series = self._materialize_series()
        return series.str.cat(others=others, sep=sep, na_rep=na_rep, join=join)

    def extractall(self, pat, flags=0):
        """
        Extract all matches of pattern from each string.

        Returns:
            DataStore wrapping the MultiIndex DataFrame result
        """
        from .core import DataStore

        series = self._materialize_series()
        result = series.str.extractall(pat, flags=flags)
        # Reset index to make it a regular DataFrame
        result = result.reset_index()
        return DataStore.from_df(result)

    def get_dummies(self, sep='|'):
        """
        Return DataFrame of dummy/indicator variables for Series.

        Returns:
            DataStore wrapping the dummy DataFrame
        """
        from .core import DataStore

        series = self._materialize_series()
        result = series.str.get_dummies(sep=sep)
        return DataStore.from_df(result)

    def partition(self, sep=' ', expand=True):
        """
        Split the string at the first occurrence of sep.

        Returns:
            DataStore wrapping the 3-column DataFrame (if expand=True)
            or Series of tuples (if expand=False)
        """
        from .core import DataStore

        series = self._materialize_series()
        result = series.str.partition(sep=sep, expand=expand)
        if expand:
            return DataStore.from_df(result)
        return result

    def rpartition(self, sep=' ', expand=True):
        """
        Split the string at the last occurrence of sep.

        Returns:
            DataStore wrapping the 3-column DataFrame (if expand=True)
            or Series of tuples (if expand=False)
        """
        from .core import DataStore

        series = self._materialize_series()
        result = series.str.rpartition(sep=sep, expand=expand)
        if expand:
            return DataStore.from_df(result)
        return result

    def __repr__(self) -> str:
        return f"ColumnExprStringAccessor({self._column_expr._expr!r})"


class ColumnExprDateTimeAccessor:
    """DateTime accessor for ColumnExpr that returns ColumnExpr for each method."""

    def __init__(self, column_expr: ColumnExpr):
        self._column_expr = column_expr
        self._base_accessor = column_expr._expr.dt

    def __getattr__(self, name: str):
        """Delegate to base accessor and wrap result in ColumnExpr."""
        base_method = getattr(self._base_accessor, name)

        # Check if it's a property (like .year, .month)
        if not callable(base_method):
            return ColumnExpr(base_method, self._column_expr._datastore)

        def wrapper(*args, **kwargs):
            result = base_method(*args, **kwargs)
            return ColumnExpr(result, self._column_expr._datastore)

        return wrapper

    def __repr__(self) -> str:
        return f"ColumnExprDateTimeAccessor({self._column_expr._expr!r})"


class LazyAggregate:
    """
    A lazy aggregate expression that executes only when displayed.

    This class allows aggregate methods like mean(), sum() to return an object
    that:
    1. In notebooks/REPL: Displays the computed result when __repr__ is called
    2. In agg(): Can be recognized as an Expression and used for SQL building
    3. With groupby: Returns a Series with group keys as index
    4. Without groupby: Returns a scalar value

    This design allows both:
    - ds.groupby('x')['col'].mean()  -> displays Series
    - ds.groupby('x').agg(avg=ds['col'].mean())  -> uses Expression for SQL
    """

    def __init__(
        self, column_expr: ColumnExpr, agg_func_name: str, pandas_agg_func: str = None, skipna: bool = True, **kwargs
    ):
        """
        Initialize a lazy aggregate.

        Args:
            column_expr: The ColumnExpr being aggregated
            agg_func_name: SQL aggregate function name (e.g., 'avg', 'sum')
            pandas_agg_func: Pandas aggregation method name (e.g., 'mean', 'sum')
            skipna: Whether to skip NaN values
            **kwargs: Additional arguments for the aggregation
        """
        self._column_expr = column_expr
        self._agg_func_name = agg_func_name
        self._pandas_agg_func = pandas_agg_func or agg_func_name
        self._skipna = skipna
        self._kwargs = kwargs

        # Capture groupby fields from ColumnExpr (if any) to avoid polluting DataStore state
        self._groupby_fields = column_expr._groupby_fields.copy() if column_expr._groupby_fields else []

        # Create the underlying AggregateFunction expression for SQL building
        from .functions import AggregateFunction

        self._expr = AggregateFunction(agg_func_name, column_expr._expr)

    @property
    def _datastore(self):
        """Get the DataStore reference."""
        return self._column_expr._datastore

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for the aggregate expression."""
        return self._expr.to_sql(quote_char=quote_char, **kwargs)

    def as_(self, alias: str) -> 'LazyAggregate':
        """Set an alias for this aggregate expression."""
        self._expr.alias = alias
        return self

    @property
    def alias(self):
        """Get the alias."""
        return self._expr.alias

    @alias.setter
    def alias(self, value):
        """Set the alias."""
        self._expr.alias = value

    def _execute(self):
        """
        Execute the aggregation and return the result.

        Returns:
            pd.Series if groupby is present, scalar otherwise
        """
        # Cache the result to avoid re-execution
        if hasattr(self, '_cached_result'):
            return self._cached_result

        datastore = self._column_expr._datastore

        # Check if we have groupby fields (from LazyGroupBy, stored in self._groupby_fields)
        if self._groupby_fields:
            # Set groupby fields on datastore temporarily for _aggregate_with_groupby
            # This will be cleared by _aggregate_with_groupby after materialization
            datastore._groupby_fields = self._groupby_fields.copy()
            result = self._column_expr._aggregate_with_groupby(
                self._agg_func_name, self._pandas_agg_func, skipna=self._skipna
            )
        else:
            # No groupby - compute scalar value
            series = self._column_expr._materialize()
            agg_method = getattr(series, self._pandas_agg_func)
            result = agg_method(**self._kwargs)

        self._cached_result = result
        return result

    # Proxy attributes to the executed result (for Series-like behavior)
    @property
    def index(self):
        """Get the index of the result (for Series)."""
        result = self._execute()
        if hasattr(result, 'index'):
            return result.index
        raise AttributeError("Scalar result has no 'index' attribute")

    @property
    def values(self):
        """Get the values of the result."""
        result = self._execute()
        if hasattr(result, 'values'):
            return result.values
        return result

    @property
    def name(self):
        """Get the name of the result Series."""
        result = self._execute()
        if hasattr(result, 'name'):
            return result.name
        return None

    @property
    def dtype(self):
        """Get the dtype of the result."""
        result = self._execute()
        if hasattr(result, 'dtype'):
            return result.dtype
        import numpy as np

        return np.dtype(type(result))

    def tolist(self):
        """Convert to list."""
        result = self._execute()
        if hasattr(result, 'tolist'):
            return result.tolist()
        return [result]

    def __len__(self):
        """Get length of the result."""
        result = self._execute()
        if hasattr(result, '__len__'):
            return len(result)
        return 1

    def __iter__(self):
        """Iterate over the result."""
        result = self._execute()
        if hasattr(result, '__iter__'):
            return iter(result)
        return iter([result])

    def __getitem__(self, key):
        """Support indexing/subscripting like pandas Series."""
        result = self._execute()
        if hasattr(result, '__getitem__'):
            return result[key]
        raise TypeError(f"'{type(result).__name__}' object is not subscriptable")

    def __repr__(self) -> str:
        """Display the computed result when shown in notebook/REPL."""
        try:
            result = self._execute()
            return repr(result)
        except Exception as e:
            return f"LazyAggregate({self._agg_func_name}({self._column_expr._expr!r})) [Error: {e}]"

    def __str__(self) -> str:
        """String representation showing the result."""
        try:
            result = self._execute()
            return str(result)
        except Exception:
            return f"{self._agg_func_name}({self._column_expr._expr.to_sql()})"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            result = self._execute()
            if hasattr(result, '_repr_html_'):
                return result._repr_html_()
            return f"<pre>{repr(result)}</pre>"
        except Exception as e:
            return f"<pre>LazyAggregate({self._agg_func_name}(...)) [Error: {e}]</pre>"

    # Support numeric operations on the result
    def __float__(self):
        """Convert to float (executes the aggregation)."""
        result = self._execute()
        return float(result) if not isinstance(result, pd.Series) else float(result.iloc[0])

    def __int__(self):
        """Convert to int (executes the aggregation)."""
        result = self._execute()
        return int(result) if not isinstance(result, pd.Series) else int(result.iloc[0])

    def __array__(self, dtype=None):
        """
        Support numpy array protocol.

        This allows numpy ufuncs to work with LazyAggregate.
        """
        import numpy as np

        result = self._execute()
        if isinstance(result, pd.Series):
            arr = result.values
        else:
            arr = np.array([result])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Support numpy ufuncs on LazyAggregate.

        Converts to scalar before applying the ufunc.
        """
        import numpy as np

        # Convert LazyAggregate inputs to actual values
        converted_inputs = []
        for inp in inputs:
            if isinstance(inp, LazyAggregate):
                result = inp._execute()
                if isinstance(result, pd.Series):
                    converted_inputs.append(result.values)
                else:
                    converted_inputs.append(result)
            else:
                converted_inputs.append(inp)
        return getattr(ufunc, method)(*converted_inputs, **kwargs)

    # Comparison operators (execute and compare)
    def __eq__(self, other):
        result = self._execute()
        return result == other

    def __ne__(self, other):
        result = self._execute()
        return result != other

    def __lt__(self, other):
        result = self._execute()
        return result < other

    def __le__(self, other):
        result = self._execute()
        return result <= other

    def __gt__(self, other):
        result = self._execute()
        return result > other

    def __ge__(self, other):
        result = self._execute()
        return result >= other

    # ========== Pandas-compatible comparison methods ==========

    def equals(self, other) -> bool:
        """
        Test whether two objects contain the same elements.

        This method allows you to compare a LazyAggregate with a pd.Series
        or another LazyAggregate and returns a single boolean.

        Args:
            other: The other object to compare with (Series, LazyAggregate, etc.)

        Returns:
            bool: True if objects are equal, False otherwise

        Example:
            >>> ds_result = ds.groupby('col')['val'].mean()
            >>> pd_result = df.groupby('col')['val'].mean()
            >>> ds_result.equals(pd_result)  # True if equal
        """
        result = self._execute()
        if hasattr(other, '_execute'):
            other = other._execute()
        elif hasattr(other, '_get_result'):
            other = other._get_result()

        if hasattr(result, 'equals'):
            return result.equals(other)
        return result == other

    def eq(self, other):
        """
        Element-wise equality comparison.

        Returns a boolean Series showing element-wise equality.
        """
        result = self._execute()
        if hasattr(other, '_execute'):
            other = other._execute()
        elif hasattr(other, '_get_result'):
            other = other._get_result()

        if hasattr(result, 'eq'):
            return result.eq(other)
        return result == other

    def compare(self, other, **kwargs):
        """
        Compare to another Series and show differences.

        Returns a DataFrame with differences between the two objects.
        """
        result = self._execute()
        if hasattr(other, '_execute'):
            other = other._execute()
        elif hasattr(other, '_get_result'):
            other = other._get_result()

        if hasattr(result, 'compare'):
            return result.compare(other, **kwargs)
        raise TypeError(f"Cannot compare {type(result)} with compare()")

    def to_series(self) -> pd.Series:
        """
        Execute and return the result as a pandas Series.

        This is useful when you need an actual pd.Series for compatibility
        with pandas testing functions or other pandas-specific operations.

        Returns:
            pd.Series: The executed result

        Example:
            >>> ds_result = ds.groupby('col')['val'].mean()
            >>> pd.testing.assert_series_equal(ds_result.to_series(), expected)
        """
        result = self._execute()
        if isinstance(result, pd.Series):
            return result
        # Scalar - wrap in Series
        return pd.Series([result])

    # Arithmetic operators (execute and compute)
    def __add__(self, other):
        result = self._execute()
        return result + other

    def __radd__(self, other):
        result = self._execute()
        return other + result

    def __sub__(self, other):
        result = self._execute()
        return result - other

    def __rsub__(self, other):
        result = self._execute()
        return other - result

    def __mul__(self, other):
        result = self._execute()
        return result * other

    def __rmul__(self, other):
        result = self._execute()
        return other * result

    def __truediv__(self, other):
        result = self._execute()
        return result / other

    def __rtruediv__(self, other):
        result = self._execute()
        return other / result

    def __round__(self, ndigits: int = None) -> 'ColumnExpr':
        """
        Support Python's built-in round() function.

        Returns a ColumnExpr wrapping round(agg_func(...), ndigits) for SQL.

        Args:
            ndigits: Number of decimal places to round to (default 0)

        Returns:
            ColumnExpr: Expression for round(aggregate, ndigits)
        """
        from .functions import Function

        if ndigits is None:
            round_expr = Function('round', self._expr)
        else:
            from .expressions import Literal

            round_expr = Function('round', self._expr, Literal(ndigits))

        return ColumnExpr(round_expr, self._column_expr._datastore)

    def head(self, n: int = 5) -> 'LazySlice':
        """
        Return the first n elements of the aggregated result (lazy).

        The result is not materialized until displayed or explicitly converted.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            LazySlice: Lazy wrapper that materializes on display

        Example:
            >>> ds.groupby('category')['value'].mean().head(5)  # Lazy
            >>> print(ds.groupby('category')['value'].mean().head(5))  # Executes
        """
        from .lazy_result import LazySlice

        return LazySlice(self, 'head', n)

    def tail(self, n: int = 5) -> 'LazySlice':
        """
        Return the last n elements of the aggregated result (lazy).

        The result is not materialized until displayed or explicitly converted.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            LazySlice: Lazy wrapper that materializes on display

        Example:
            >>> ds.groupby('category')['value'].mean().tail(5)  # Lazy
            >>> print(ds.groupby('category')['value'].mean().tail(5))  # Executes
        """
        from .lazy_result import LazySlice

        return LazySlice(self, 'tail', n)
