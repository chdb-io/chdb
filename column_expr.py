"""
ColumnExpr - A column expression that can materialize when displayed.

This provides pandas-like behavior where accessing a column or performing
operations on it shows actual values when displayed, while still supporting
lazy expression building for filters and assignments.
"""

from typing import Any, Optional, TYPE_CHECKING, Iterator

import pandas as pd

from .expressions import Expression, Field, ArithmeticExpression, Literal, Node
from .utils import immutable

if TYPE_CHECKING:
    from .core import DataStore
    from .conditions import Condition, BinaryCondition


class ColumnExpr(Expression):
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

    def __init__(self, expr: Expression, datastore: 'DataStore', alias: Optional[str] = None):
        """
        Initialize ColumnExpr with expression and DataStore reference.

        Args:
            expr: The underlying expression (Field, ArithmeticExpression, Function, etc.)
            datastore: Reference to the DataStore for materialization
            alias: Optional alias for the expression
        """
        super().__init__(alias)
        self._expr = expr
        self._datastore = datastore

    @property
    def expr(self) -> Expression:
        """Get the underlying expression."""
        return self._expr

    @property
    def datastore(self) -> 'DataStore':
        """Get the DataStore reference."""
        return self._datastore

    # ========== Materialization ==========

    def _materialize(self) -> pd.Series:
        """
        Materialize this expression and return a pandas Series.

        This executes the expression against the DataStore's data.
        """
        from .functions import Function, CastFunction

        # Get the materialized DataFrame from the DataStore
        df = self._datastore._materialize()

        # Handle different expression types
        if isinstance(self._expr, Field):
            # Simple column access
            col_name = self._expr.name
            if col_name in df.columns:
                return df[col_name]
            else:
                raise KeyError(f"Column '{col_name}' not found in DataFrame")

        elif isinstance(self._expr, ArithmeticExpression):
            # Evaluate arithmetic expression via chDB
            return self._evaluate_via_chdb(df)

        elif isinstance(self._expr, (Function, CastFunction)):
            # Evaluate function via chDB
            return self._evaluate_via_chdb(df)

        elif isinstance(self._expr, Literal):
            # Return scalar as Series
            return pd.Series([self._expr.value] * len(df), index=df.index)

        else:
            # Try to evaluate via chDB for other expression types
            return self._evaluate_via_chdb(df)

    def _evaluate_via_chdb(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate the expression using chDB's Python() table function."""
        from .executor import get_executor

        executor = get_executor()
        sql_expr = self._expr.to_sql(quote_char='"')
        return executor.execute_expression(sql_expr, df)

    # ========== Display Methods ==========

    def __repr__(self) -> str:
        """
        Return a representation that shows actual values.

        In IPython/Jupyter, this is what gets displayed.
        """
        try:
            series = self._materialize()
            return repr(series)
        except Exception as e:
            # Fallback to expression representation if materialization fails
            return f"ColumnExpr({self._expr!r}) [Error: {e}]"

    def __str__(self) -> str:
        """Return string representation showing actual values."""
        try:
            series = self._materialize()
            return str(series)
        except Exception:
            # Fallback to expression SQL
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

    # ========== Expression Interface ==========

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for the underlying expression."""
        return self._expr.to_sql(quote_char=quote_char, **kwargs)

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
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

    # ========== Accessor Properties ==========

    @property
    def str(self) -> 'ColumnExprStringAccessor':
        """
        Accessor for string functions.

        Returns a ColumnExprStringAccessor that wraps string operations
        and returns ColumnExpr for each method call.
        """
        return ColumnExprStringAccessor(self)

    @property
    def dt(self) -> 'ColumnExprDateTimeAccessor':
        """
        Accessor for date/time functions.

        Returns a ColumnExprDateTimeAccessor that wraps datetime operations
        and returns ColumnExpr for each method call.
        """
        return ColumnExprDateTimeAccessor(self)

    # ========== Condition Methods (for filtering) ==========

    def isnull(self) -> 'Condition':
        """Create IS NULL condition."""
        return self._expr.isnull()

    def notnull(self) -> 'Condition':
        """Create IS NOT NULL condition."""
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

    # ========== Math Functions ==========

    def abs(self, alias: str = None) -> 'ColumnExpr':
        """Absolute value."""
        new_expr = self._expr.abs(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def round(self, precision: int = 0, alias: str = None) -> 'ColumnExpr':
        """Round to N decimal places."""
        new_expr = self._expr.round(precision, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def floor(self, alias: str = None) -> 'ColumnExpr':
        """Round down to nearest integer."""
        new_expr = self._expr.floor(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def ceil(self, alias: str = None) -> 'ColumnExpr':
        """Round up to nearest integer."""
        new_expr = self._expr.ceil(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def ceiling(self, alias: str = None) -> 'ColumnExpr':
        """Alias for ceil()."""
        return self.ceil(alias=alias)

    def sqrt(self, alias: str = None) -> 'ColumnExpr':
        """Square root."""
        new_expr = self._expr.sqrt(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def exp(self, alias: str = None) -> 'ColumnExpr':
        """Exponential (e^x)."""
        new_expr = self._expr.exp(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def log(self, base: float = None, alias: str = None) -> 'ColumnExpr':
        """Logarithm."""
        new_expr = self._expr.log(base, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def log10(self, alias: str = None) -> 'ColumnExpr':
        """Base-10 logarithm."""
        new_expr = self._expr.log10(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def log2(self, alias: str = None) -> 'ColumnExpr':
        """Base-2 logarithm."""
        new_expr = self._expr.log2(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def sin(self, alias: str = None) -> 'ColumnExpr':
        """Sine."""
        new_expr = self._expr.sin(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def cos(self, alias: str = None) -> 'ColumnExpr':
        """Cosine."""
        new_expr = self._expr.cos(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def tan(self, alias: str = None) -> 'ColumnExpr':
        """Tangent."""
        new_expr = self._expr.tan(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def power(self, exponent, alias: str = None) -> 'ColumnExpr':
        """Raise to power."""
        new_expr = self._expr.power(exponent, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def sign(self, alias: str = None) -> 'ColumnExpr':
        """Sign of number (-1, 0, or 1)."""
        new_expr = self._expr.sign(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    # ========== Type Conversion ==========

    def cast(self, target_type: str, alias: str = None) -> 'ColumnExpr':
        """Cast to specified type."""
        new_expr = self._expr.cast(target_type, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def to_string(self, alias: str = None) -> 'ColumnExpr':
        """Convert to String type."""
        new_expr = self._expr.to_string(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def to_int(self, bits: int = 64, alias: str = None) -> 'ColumnExpr':
        """Convert to integer type."""
        new_expr = self._expr.to_int(bits, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def to_float(self, bits: int = 64, alias: str = None) -> 'ColumnExpr':
        """Convert to float type."""
        new_expr = self._expr.to_float(bits, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def to_date(self, alias: str = None) -> 'ColumnExpr':
        """Convert to Date type."""
        new_expr = self._expr.to_date(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def to_datetime(self, timezone: str = None, alias: str = None) -> 'ColumnExpr':
        """Convert to DateTime type."""
        new_expr = self._expr.to_datetime(timezone, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    # ========== Conditional Functions ==========

    def if_null(self, default, alias: str = None) -> 'ColumnExpr':
        """Return default value if expression is NULL."""
        new_expr = self._expr.if_null(default, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def coalesce(self, *alternatives, alias: str = None) -> 'ColumnExpr':
        """Return first non-NULL value."""
        new_expr = self._expr.coalesce(*alternatives, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def null_if(self, value, alias: str = None) -> 'ColumnExpr':
        """Return NULL if expression equals value."""
        new_expr = self._expr.null_if(value, alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    # ========== Aggregate Functions ==========
    # These return scalar values when materialized

    def sum(self, alias: str = None) -> 'ColumnExpr':
        """Sum aggregate."""
        new_expr = self._expr.sum(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def avg(self, alias: str = None) -> 'ColumnExpr':
        """Average aggregate."""
        new_expr = self._expr.avg(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def mean(self, alias: str = None) -> 'ColumnExpr':
        """Alias for avg()."""
        return self.avg(alias=alias)

    def count(self, alias: str = None) -> 'ColumnExpr':
        """Count aggregate."""
        new_expr = self._expr.count(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def max(self, alias: str = None) -> 'ColumnExpr':
        """Maximum aggregate."""
        new_expr = self._expr.max(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    def min(self, alias: str = None) -> 'ColumnExpr':
        """Minimum aggregate."""
        new_expr = self._expr.min(alias=alias)
        return ColumnExpr(new_expr, self._datastore)

    # ========== Alias Support ==========

    @immutable
    def as_(self, alias: str) -> 'ColumnExpr':
        """Set an alias for this expression."""
        new_expr = self._expr.as_(alias)
        return ColumnExpr(new_expr, self._datastore, alias=alias)

    # ========== Pandas Series Methods (for compatibility) ==========

    def values(self) -> Any:
        """Return underlying numpy array."""
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

    def tolist(self) -> list:
        """Convert to Python list."""
        return self._materialize().tolist()

    def to_numpy(self):
        """Convert to numpy array."""
        return self._materialize().to_numpy()


class ColumnExprStringAccessor:
    """
    String accessor for ColumnExpr that returns ColumnExpr for each method.
    """

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

    def __repr__(self) -> str:
        return f"ColumnExprStringAccessor({self._column_expr._expr!r})"


class ColumnExprDateTimeAccessor:
    """
    DateTime accessor for ColumnExpr that returns ColumnExpr for each method.
    """

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
