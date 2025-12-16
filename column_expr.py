"""
ColumnExpr - A column expression that can materialize when displayed.

This provides pandas-like behavior where accessing a column or performing
operations on it shows actual values when displayed, while still supporting
lazy expression building for filters and assignments.

Uses composition (not inheritance) to wrap Expression and return ColumnExpr
for all operations. This ensures pandas-like behavior is preserved.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING, Iterator

import pandas as pd

from .expressions import Expression, Field, ArithmeticExpression, Literal, Node
from .utils import immutable

if TYPE_CHECKING:
    from .core import DataStore
    from .conditions import Condition, BinaryCondition


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

    def __init__(self, expr: Expression, datastore: 'DataStore', alias: Optional[str] = None):
        """
        Initialize ColumnExpr with expression and DataStore reference.

        Args:
            expr: The underlying expression (Field, ArithmeticExpression, Function, etc.)
            datastore: Reference to the DataStore for materialization
            alias: Optional alias for the expression
        """
        self._expr = expr
        self._datastore = datastore
        self._alias = alias

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
        """Accessor for string functions."""
        return ColumnExprStringAccessor(self)

    @property
    def dt(self) -> 'ColumnExprDateTimeAccessor':
        """Accessor for date/time functions."""
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

    # ========== NumPy-Compatible Statistical Methods ==========
    # These methods accept NumPy-style parameters (axis, dtype, out, keepdims)
    # to enable direct usage with np.mean(), np.sum(), np.std(), etc.

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, skipna=True, **kwargs):
        """
        Compute the mean of the column.

        When called without NumPy-style parameters, returns a SQL expression.
        When called with NumPy-style parameters (by np.mean()), materializes and computes.

        Args:
            axis: NumPy axis parameter (ignored for 1D data, enables np.mean() compatibility)
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported, for signature compatibility)
            keepdims: NumPy keepdims parameter (ignored for 1D data)
            skipna: Whether to skip NA values (pandas style)
            **kwargs: Additional arguments

        Returns:
            float: The computed mean value
        """
        # Materialize and compute using pandas Series
        series = self._materialize()
        return series.mean(axis=axis, skipna=skipna, **kwargs)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, *, skipna=True, min_count=0, **kwargs):
        """
        Compute the sum of the column.

        Args:
            axis: NumPy axis parameter (enables np.sum() compatibility)
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            min_count: Minimum number of non-NA values required
            **kwargs: Additional arguments

        Returns:
            float: The computed sum value
        """
        series = self._materialize()
        return series.sum(axis=axis, skipna=skipna, min_count=min_count, **kwargs)

    def std(self, axis=None, dtype=None, out=None, ddof=1, keepdims=False, *, skipna=True, **kwargs):
        """
        Compute the standard deviation of the column.

        Args:
            axis: NumPy axis parameter (enables np.std() compatibility)
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            ddof: Delta degrees of freedom (default 1 for sample std)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            float: The computed standard deviation
        """
        series = self._materialize()
        return series.std(axis=axis, ddof=ddof, skipna=skipna, **kwargs)

    def var(self, axis=None, dtype=None, out=None, ddof=1, keepdims=False, *, skipna=True, **kwargs):
        """
        Compute the variance of the column.

        Args:
            axis: NumPy axis parameter (enables np.var() compatibility)
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            ddof: Delta degrees of freedom (default 1 for sample variance)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            float: The computed variance
        """
        series = self._materialize()
        return series.var(axis=axis, ddof=ddof, skipna=skipna, **kwargs)

    def min(self, axis=None, out=None, keepdims=False, *, skipna=True, **kwargs):
        """
        Compute the minimum value of the column.

        Args:
            axis: NumPy axis parameter (enables np.min() compatibility)
            out: NumPy out parameter (not supported)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            The minimum value
        """
        series = self._materialize()
        return series.min(axis=axis, skipna=skipna, **kwargs)

    def max(self, axis=None, out=None, keepdims=False, *, skipna=True, **kwargs):
        """
        Compute the maximum value of the column.

        Args:
            axis: NumPy axis parameter (enables np.max() compatibility)
            out: NumPy out parameter (not supported)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            **kwargs: Additional arguments

        Returns:
            The maximum value
        """
        series = self._materialize()
        return series.max(axis=axis, skipna=skipna, **kwargs)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False, *, skipna=True, min_count=0, **kwargs):
        """
        Compute the product of the column values.

        Args:
            axis: NumPy axis parameter (enables np.prod() compatibility)
            dtype: NumPy dtype parameter
            out: NumPy out parameter (not supported)
            keepdims: NumPy keepdims parameter
            skipna: Whether to skip NA values
            min_count: Minimum number of non-NA values required
            **kwargs: Additional arguments

        Returns:
            float: The computed product
        """
        series = self._materialize()
        return series.prod(axis=axis, skipna=skipna, min_count=min_count, **kwargs)

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
        Fill NA/NaN values.

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
        """
        if inplace:
            raise ValueError("ColumnExpr is immutable, inplace=True is not supported")
        series = self._materialize()
        return series.fillna(value=value, method=method, axis=axis, limit=limit)

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

    def head(self, n: int = 5):
        """
        Return the first n elements.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            pd.Series: First n elements
        """
        series = self._materialize()
        return series.head(n)

    def tail(self, n: int = 5):
        """
        Return the last n elements.

        Args:
            n: Number of elements to return (default 5)

        Returns:
            pd.Series: Last n elements
        """
        series = self._materialize()
        return series.tail(n)

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
