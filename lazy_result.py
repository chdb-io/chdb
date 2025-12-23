"""
Lazy Result Classes for DataStore.

This module provides lazy wrappers for results that should not be materialized
immediately. These classes maintain the lazy evaluation chain and only execute
when display or explicit conversion is triggered.

Key classes:
- LazySeries: Wraps any Series method call for lazy evaluation
- LazySeries: Wraps any Series method call for lazy evaluation (including head/tail)

Design Principle:
    All operations on ColumnExpr should return lazy objects that:
    1. Build an expression tree (don't execute immediately)
    2. Execute only when displayed or explicitly converted
    3. Can choose execution engine (Pandas vs SQL) at execution time
    4. Cache results to avoid re-execution
"""

from typing import TYPE_CHECKING, Union, Optional, Any, Callable
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .column_expr import ColumnExpr, LazyAggregate
    from .core import DataStore


class LazySeries:
    """
    A lazy wrapper for Series method calls that executes only when displayed.

    This class enables delayed execution of pandas Series methods on ColumnExpr.
    The method is not executed until the result needs to be displayed or
    explicitly converted to a pandas object.

    Key features:
    - Delayed execution: method runs only when result is needed
    - Result caching: avoids re-execution on repeated access
    - Execution engine selection: can choose Pandas or SQL (extensible)
    - Series interface: exposes common Series properties for chaining

    Example:
        >>> ds['category'].value_counts()  # Returns LazySeries, not pd.Series
        LazySeries(value_counts)

        >>> print(ds['category'].value_counts())  # Triggers execution
        A    150
        B    100
        C     50
        Name: category, dtype: int64

        >>> # Can chain with other lazy operations
        >>> ds['category'].value_counts().head(3)  # Still lazy

    Attributes:
        _column_expr: The source ColumnExpr
        _method_name: Name of the Series method to call
        _args: Positional arguments for the method
        _kwargs: Keyword arguments for the method
    """

    def __init__(
        self,
        column_expr: 'ColumnExpr',
        method_name: str,
        *args,
        **kwargs,
    ):
        """
        Initialize a LazySeries.

        Args:
            column_expr: The source ColumnExpr
            method_name: Name of the Series method to call (e.g., 'value_counts', 'unique')
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
        """
        self._column_expr = column_expr
        self._method_name = method_name
        self._args = args
        self._kwargs = kwargs
        self._cached_result = None

    @property
    def _datastore(self) -> Optional['DataStore']:
        """Get the DataStore reference."""
        return self._column_expr._datastore

    def _execute(self) -> Union[pd.Series, pd.DataFrame, np.ndarray, Any]:
        """
        Execute the method and return the result.

        This is where execution engine selection can be implemented.
        Currently uses Pandas, but can be extended to use SQL for
        operations like value_counts (GROUP BY COUNT).

        Returns:
            The result of the method call (typically pd.Series)
        """
        if self._cached_result is not None:
            return self._cached_result

        # Materialize the source - support ColumnExpr, LazyAggregate, and LazySeries
        from .column_expr import LazyAggregate

        if isinstance(self._column_expr, LazyAggregate):
            series = self._column_expr._execute()
        elif isinstance(self._column_expr, LazySeries):
            series = self._column_expr._execute()
        else:
            series = self._column_expr._materialize()

        # Materialize any ColumnExpr in args or kwargs
        args = tuple(self._materialize_if_needed(arg) for arg in self._args)
        kwargs = {k: self._materialize_if_needed(v) for k, v in self._kwargs.items()}

        # Execute the method - handle case where method doesn't exist (e.g., head() on scalar)
        if not hasattr(series, self._method_name):
            # Method doesn't exist on this type (e.g., calling head() on a scalar)
            # Return the series as-is
            self._cached_result = series
            return self._cached_result

        method = getattr(series, self._method_name)
        self._cached_result = method(*args, **kwargs)

        return self._cached_result

    def _materialize_if_needed(self, value: Any) -> Any:
        """Materialize ColumnExpr, LazySeries, or LazyAggregate arguments."""
        from .column_expr import ColumnExpr, LazyAggregate

        if isinstance(value, LazyAggregate):
            # Execute LazyAggregate to get the result
            result = value._execute()
            # If it's a single-value Series, extract scalar for fillna-like operations
            if isinstance(result, pd.Series) and len(result) == 1:
                return result.iloc[0]
            return result
        if isinstance(value, ColumnExpr):
            materialized = value._materialize()
            # If it's a single-value Series, extract scalar for fillna-like operations
            if isinstance(materialized, pd.Series) and len(materialized) == 1:
                return materialized.iloc[0]
            return materialized
        if isinstance(value, LazySeries):
            return value._execute()
        return value

    # ========== Display Methods ==========

    def __repr__(self) -> str:
        """Display the result when shown in notebook/REPL."""
        try:
            result = self._execute()
            return repr(result)
        except Exception as e:
            return f"LazySeries({self._method_name}) [Error: {e}]"

    def __str__(self) -> str:
        """String representation showing the result."""
        try:
            result = self._execute()
            return str(result)
        except Exception:
            return f"LazySeries({self._method_name})"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            result = self._execute()
            if hasattr(result, '_repr_html_'):
                return result._repr_html_()
            return f"<pre>{repr(result)}</pre>"
        except Exception as e:
            return f"<pre>LazySeries({self._method_name}) [Error: {e}]</pre>"

    # ========== Chaining Methods ==========

    def head(self, n: int = 5) -> 'LazySeries':
        """Return the first n elements (lazy)."""
        return LazySeries(self, 'head', n)

    def tail(self, n: int = 5) -> 'LazySeries':
        """Return the last n elements (lazy)."""
        return LazySeries(self, 'tail', n)

    # ========== Conversion Methods ==========

    def to_pandas(self) -> Union[pd.Series, pd.DataFrame]:
        """Execute and return as pandas object."""
        return self._execute()

    def to_numpy(self) -> np.ndarray:
        """Execute and return as numpy array."""
        result = self._execute()
        if hasattr(result, 'to_numpy'):
            return result.to_numpy()
        return np.array(result)

    def to_list(self) -> list:
        """Execute and return as list."""
        result = self._execute()
        if hasattr(result, 'tolist'):
            return result.tolist()
        return list(result)

    def tolist(self) -> list:
        """Alias for to_list()."""
        return self.to_list()

    # ========== Numeric Operations ==========

    def __len__(self) -> int:
        """Return length of result."""
        result = self._execute()
        return len(result)

    def __iter__(self):
        """Iterate over result."""
        result = self._execute()
        return iter(result)

    def __getitem__(self, key):
        """Index into result."""
        result = self._execute()
        return result[key]

    def __array__(self, dtype=None, copy=None):
        """Support numpy array protocol."""
        result = self._execute()
        if hasattr(result, 'to_numpy'):
            arr = result.to_numpy()
        else:
            arr = np.array(result)
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            arr = np.array(arr, copy=True)
        return arr

    # ========== Property Proxies ==========

    @property
    def values(self) -> np.ndarray:
        """Return values as numpy array."""
        result = self._execute()
        if hasattr(result, 'values'):
            return result.values
        return np.array(result)

    @property
    def index(self):
        """Return index of result."""
        result = self._execute()
        if hasattr(result, 'index'):
            return result.index
        return None

    @property
    def dtype(self):
        """Return dtype of result."""
        result = self._execute()
        if hasattr(result, 'dtype'):
            return result.dtype
        return type(result)

    @property
    def name(self):
        """Return name of result."""
        result = self._execute()
        if hasattr(result, 'name'):
            return result.name
        return None

    @property
    def shape(self):
        """Return shape of result."""
        result = self._execute()
        if hasattr(result, 'shape'):
            return result.shape
        return (len(result),) if hasattr(result, '__len__') else ()

    # ========== Plotting Support ==========

    @property
    def plot(self):
        """Access plot accessor for visualization."""
        result = self._execute()
        if hasattr(result, 'plot'):
            return result.plot
        raise AttributeError(f"'{type(result).__name__}' has no 'plot' attribute")

    # ========== Comparison Operators ==========

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

    # ========== Series Method Proxies ==========
    # These methods execute and return the result (or chain lazily where possible)

    def sort_index(self, **kwargs):
        """Sort by index."""
        result = self._execute()
        if hasattr(result, 'sort_index'):
            return result.sort_index(**kwargs)
        return result

    def sort_values(self, **kwargs):
        """Sort by values."""
        result = self._execute()
        if hasattr(result, 'sort_values'):
            return result.sort_values(**kwargs)
        return result

    def reset_index(self, **kwargs):
        """Reset index."""
        result = self._execute()
        if hasattr(result, 'reset_index'):
            return result.reset_index(**kwargs)
        return result

    def astype(self, dtype):
        """Cast to dtype."""
        result = self._execute()
        if hasattr(result, 'astype'):
            return result.astype(dtype)
        return result

    def copy(self):
        """Return copy of result."""
        result = self._execute()
        if hasattr(result, 'copy'):
            return result.copy()
        return result

    def equals(self, other):
        """Test equality with another object."""
        result = self._execute()
        if hasattr(result, 'equals'):
            if hasattr(other, '_execute'):
                other = other._execute()
            return result.equals(other)
        return result == other

    # ========== Arithmetic Operators (lazy - return LazySeries) ==========

    def __add__(self, other):
        """Addition (lazy)."""
        return LazySeries(self, '__add__', other)

    def __radd__(self, other):
        """Right addition (lazy)."""
        return LazySeries(self, '__radd__', other)

    def __sub__(self, other):
        """Subtraction (lazy)."""
        return LazySeries(self, '__sub__', other)

    def __rsub__(self, other):
        """Right subtraction (lazy)."""
        return LazySeries(self, '__rsub__', other)

    def __mul__(self, other):
        """Multiplication (lazy)."""
        return LazySeries(self, '__mul__', other)

    def __rmul__(self, other):
        """Right multiplication (lazy)."""
        return LazySeries(self, '__rmul__', other)

    def __truediv__(self, other):
        """Division (lazy)."""
        return LazySeries(self, '__truediv__', other)

    def __rtruediv__(self, other):
        """Right division (lazy)."""
        return LazySeries(self, '__rtruediv__', other)

    def __floordiv__(self, other):
        """Floor division (lazy)."""
        return LazySeries(self, '__floordiv__', other)

    def __rfloordiv__(self, other):
        """Right floor division (lazy)."""
        return LazySeries(self, '__rfloordiv__', other)

    def __mod__(self, other):
        """Modulo (lazy)."""
        return LazySeries(self, '__mod__', other)

    def __rmod__(self, other):
        """Right modulo (lazy)."""
        return LazySeries(self, '__rmod__', other)

    def __pow__(self, other):
        """Power (lazy)."""
        return LazySeries(self, '__pow__', other)

    def __rpow__(self, other):
        """Right power (lazy)."""
        return LazySeries(self, '__rpow__', other)

    def __neg__(self):
        """Negation (lazy)."""
        return LazySeries(self, '__neg__')

    def __pos__(self):
        """Positive (lazy)."""
        return LazySeries(self, '__pos__')

    def __abs__(self):
        """Absolute value (lazy)."""
        return LazySeries(self, '__abs__')

    # ========== Numeric Conversions (trigger execution) ==========

    def __float__(self) -> float:
        """Convert to float."""
        result = self._execute()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 1:
                return float(result.iloc[0])
            raise ValueError("Cannot convert multi-element result to float")
        return float(result)

    def __int__(self) -> int:
        """Convert to int."""
        result = self._execute()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 1:
                return int(result.iloc[0])
            raise ValueError("Cannot convert multi-element result to int")
        return int(result)

    def __bool__(self) -> bool:
        """Convert to bool."""
        result = self._execute()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 0:
                return False
            if len(result) == 1:
                return bool(result.iloc[0])
            raise ValueError("Truth value of multi-element result is ambiguous")
        return bool(result)

    # ========== Aggregation Methods (trigger execution, return scalar) ==========

    def sum(self, *args, **kwargs):
        """Compute sum of the result."""
        return self._execute().sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        """Compute mean of the result."""
        return self._execute().mean(*args, **kwargs)

    def min(self, *args, **kwargs):
        """Compute min of the result."""
        return self._execute().min(*args, **kwargs)

    def max(self, *args, **kwargs):
        """Compute max of the result."""
        return self._execute().max(*args, **kwargs)

    def std(self, *args, **kwargs):
        """Compute standard deviation of the result."""
        return self._execute().std(*args, **kwargs)

    def var(self, *args, **kwargs):
        """Compute variance of the result."""
        return self._execute().var(*args, **kwargs)

    def median(self, *args, **kwargs):
        """Compute median of the result."""
        return self._execute().median(*args, **kwargs)

    def count(self, *args, **kwargs):
        """Count non-NA values."""
        return self._execute().count(*args, **kwargs)
