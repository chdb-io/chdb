"""
Lazy Result Classes for DataStore.

This module provides lazy wrappers for results that should not be executed
immediately. These classes maintain the lazy evaluation chain and only execute
when display or explicit conversion is triggered.

Key classes:
- LazySeries: Wraps any Series method call for lazy evaluation
- LazyCondition: Wraps Condition for dual SQL/pandas mode

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
    from .column_expr import ColumnExpr
    from .core import DataStore
    from .expressions import Field
    from .conditions import Condition


class LazyCondition:
    """
    A lazy wrapper for Condition objects that supports both SQL and pandas modes.

    This class enables conditions like isin() and between() to be used both for:
    1. SQL-style filtering (preserves SQL generation capability)
    2. Pandas-style boolean Series (when to_pandas() is called)

    Key features:
    - Delegates to_sql() to underlying Condition (for SQL generation)
    - Provides to_pandas() for converting to boolean Series
    - Supports boolean operations (&, |, ~) for combining conditions
    - Can be used with DataStore.filter() for SQL or DataFrame filtering

    Example:
        >>> cond = ds['category'].isin(['A', 'B'])  # Returns LazyCondition
        >>> ds.filter(cond)  # Works for SQL generation
        >>> cond.to_pandas()  # Returns boolean Series
        >>> ds[cond.to_pandas()]  # Pandas-style boolean indexing
    """

    def __init__(self, condition: 'Condition', datastore: 'DataStore'):
        """
        Initialize a LazyCondition.

        Args:
            condition: The underlying Condition object
            datastore: Reference to the DataStore for execution
        """
        self._condition = condition
        self._datastore = datastore
        self._cached_result = None

    @property
    def condition(self) -> 'Condition':
        """Get the underlying Condition object."""
        return self._condition

    def _execute(self) -> pd.Series:
        """Execute and return boolean Series."""
        if self._cached_result is not None:
            return self._cached_result

        from .expression_evaluator import ExpressionEvaluator

        df = self._datastore._execute()
        evaluator = ExpressionEvaluator(df, self._datastore)
        self._cached_result = evaluator.evaluate(self._condition)
        return self._cached_result

    # ========== SQL Delegation ==========

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for this condition (delegates to underlying Condition)."""
        return self._condition.to_sql(quote_char=quote_char, **kwargs)

    # ========== Pandas Conversion ==========

    def to_pandas(self) -> pd.Series:
        """Execute and return as boolean pandas Series."""
        return self._execute()

    # ========== Display Methods ==========

    def __repr__(self) -> str:
        """Display the result when shown in notebook/REPL."""
        try:
            result = self._execute()
            return repr(result)
        except Exception as e:
            return f"LazyCondition({self._condition}) [Error: {e}]"

    def __str__(self) -> str:
        """String representation showing the result."""
        try:
            result = self._execute()
            return str(result)
        except Exception:
            return f"LazyCondition({self._condition})"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            result = self._execute()
            if hasattr(result, '_repr_html_'):
                return result._repr_html_()
            return f"<pre>{repr(result)}</pre>"
        except Exception as e:
            return f"<pre>LazyCondition({self._condition}) [Error: {e}]</pre>"

    # ========== Boolean Operations ==========

    def __and__(self, other: 'LazyCondition') -> 'LazyCondition':
        """Combine conditions with AND."""
        if isinstance(other, LazyCondition):
            return LazyCondition(self._condition & other._condition, self._datastore)
        return LazyCondition(self._condition & other, self._datastore)

    def __or__(self, other: 'LazyCondition') -> 'LazyCondition':
        """Combine conditions with OR."""
        if isinstance(other, LazyCondition):
            return LazyCondition(self._condition | other._condition, self._datastore)
        return LazyCondition(self._condition | other, self._datastore)

    def __invert__(self) -> 'LazyCondition':
        """Negate condition with NOT."""
        return LazyCondition(~self._condition, self._datastore)

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

    # ========== Array Protocol ==========

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
        column_expr: 'ColumnExpr' = None,
        method_name: str = None,
        *args,
        executor: Callable[[], Any] = None,
        datastore: 'DataStore' = None,
        **kwargs,
    ):
        """
        Initialize a LazySeries.

        Supports two modes:
        1. Method mode: column_expr + method_name (traditional)
           >>> LazySeries(column_expr, 'value_counts')

        2. Executor mode: executor callable (for complex operations like groupby.size())
           >>> LazySeries(executor=lambda: df.groupby(cols).size(), datastore=ds)

        Args:
            column_expr: The source ColumnExpr (method mode)
            method_name: Name of the Series method to call (method mode)
            *args: Positional arguments for the method
            executor: Callable that returns the result when executed (executor mode)
            datastore: DataStore reference for executor mode
            **kwargs: Keyword arguments for the method
        """
        self._column_expr = column_expr
        self._method_name = method_name
        self._args = args
        self._executor = executor
        self._explicit_datastore = datastore
        self._kwargs = kwargs
        self._cached_result = None

    @property
    def _datastore(self) -> Optional['DataStore']:
        """Get the DataStore reference."""
        if self._explicit_datastore is not None:
            return self._explicit_datastore
        if self._column_expr is not None:
            return self._column_expr._datastore
        return None

    def _execute(self) -> Union[pd.Series, pd.DataFrame, np.ndarray, Any]:
        """
        Execute the method and return the result.

        This is where execution engine selection can be implemented.
        Currently uses Pandas, but can be extended to use SQL for
        operations like value_counts (GROUP BY COUNT).

        Returns:
            The result of the method call (typically pd.Series or pd.DataFrame)
        """
        if self._cached_result is not None:
            return self._cached_result

        # Executor mode: directly call the executor callable
        if self._executor is not None:
            self._cached_result = self._executor()
            return self._cached_result

        # Method mode: execute column_expr and call method
        from .column_expr import ColumnExpr
        from .expressions import Field

        # Execute any ColumnExpr in args or kwargs
        args = tuple(self._execute_if_needed(arg) for arg in self._args)
        kwargs = {k: self._execute_if_needed(v) for k, v in self._kwargs.items()}

        # Handle groupby + agg scenario: df.groupby('col')['value'].agg(['sum', 'mean'])
        # When _column_expr has _groupby_fields, we need to do groupby first, then agg
        if (
            isinstance(self._column_expr, ColumnExpr)
            and hasattr(self._column_expr, '_groupby_fields')
            and self._column_expr._groupby_fields
            and self._method_name in ('agg', 'aggregate')
        ):
            # Get groupby column names
            groupby_col_names = []
            for gf in self._column_expr._groupby_fields:
                if isinstance(gf, Field):
                    groupby_col_names.append(gf.name)
                else:
                    groupby_col_names.append(str(gf))

            # Get the column name being aggregated
            col_name = None
            if isinstance(self._column_expr._expr, Field):
                col_name = self._column_expr._expr.name
            else:
                col_name = str(self._column_expr._expr)

            # Execute the DataFrame
            df = self._column_expr._datastore._execute()

            # Filter out 'axis' from kwargs - pandas groupby agg doesn't accept axis
            agg_kwargs = {k: v for k, v in kwargs.items() if k != 'axis'}

            # Perform groupby and agg
            if col_name and col_name in df.columns:
                grouped = df.groupby(groupby_col_names)[col_name]
                self._cached_result = grouped.agg(*args, **agg_kwargs)
            else:
                # Fallback: just execute the column and agg without groupby
                series = self._column_expr._execute()
                method = getattr(series, self._method_name)
                self._cached_result = method(*args, **agg_kwargs)

            return self._cached_result

        # Execute the column expression - handles ColumnExpr, LazySeries, and any object with _execute
        series = self._column_expr._execute()

        # Handle special _dt_* methods for datetime operations (pandas fallback)
        if self._method_name.startswith('_dt_'):
            dt_attr = self._method_name[4:]  # Remove '_dt_' prefix
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(series):
                if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
                    try:
                        series = pd.to_datetime(series, errors='coerce')
                    except Exception:
                        pass
            # Access .dt accessor
            dt_accessor = series.dt
            attr = getattr(dt_accessor, dt_attr)
            if callable(attr):
                self._cached_result = attr(*args, **kwargs)
            else:
                self._cached_result = attr
            return self._cached_result

        # Execute the method - handle case where method doesn't exist (e.g., head() on scalar)
        if not hasattr(series, self._method_name):
            # Method doesn't exist on this type (e.g., calling head() on a scalar)
            # Return the series as-is
            self._cached_result = series
            return self._cached_result

        method = getattr(series, self._method_name)
        self._cached_result = method(*args, **kwargs)

        return self._cached_result

    def _execute_if_needed(self, value: Any) -> Any:
        """Execute ColumnExpr, LazySeries arguments if needed."""
        from .column_expr import ColumnExpr

        if isinstance(value, ColumnExpr):
            result = value._execute()
            # If it's a single-value Series, extract scalar for fillna-like operations
            if isinstance(result, pd.Series) and len(result) == 1:
                return result.iloc[0]
            return result
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
            name = self._method_name or 'executor'
            return f"LazySeries({name}) [Error: {e}]"

    def __str__(self) -> str:
        """String representation showing the result."""
        try:
            result = self._execute()
            return str(result)
        except Exception:
            name = self._method_name or 'executor'
            return f"LazySeries({name})"

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

    @property
    def columns(self):
        """Return columns of result (if DataFrame)."""
        result = self._execute()
        if hasattr(result, 'columns'):
            return result.columns
        raise AttributeError(f"'{type(result).__name__}' has no 'columns' attribute")

    def to_df(self) -> pd.DataFrame:
        """
        Convert to DataFrame.

        If the result is a Series, converts to single-column DataFrame.
        If the result is already a DataFrame, returns it directly.
        """
        result = self._execute()
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, pd.Series):
            return result.to_frame()
        else:
            return pd.DataFrame({'value': [result]})

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
        """Element-wise equality (lazy)."""
        return LazySeries(self, '__eq__', other)

    def __ne__(self, other):
        """Element-wise not-equal (lazy)."""
        return LazySeries(self, '__ne__', other)

    def __lt__(self, other):
        """Element-wise less-than (lazy)."""
        return LazySeries(self, '__lt__', other)

    def __le__(self, other):
        """Element-wise less-than-or-equal (lazy)."""
        return LazySeries(self, '__le__', other)

    def __gt__(self, other):
        """Element-wise greater-than (lazy)."""
        return LazySeries(self, '__gt__', other)

    def __ge__(self, other):
        """Element-wise greater-than-or-equal (lazy)."""
        return LazySeries(self, '__ge__', other)

    # ========== Chaining Methods (lazy - return LazySeries) ==========
    # These methods return LazySeries to maintain lazy evaluation chain

    def sort_index(self, **kwargs):
        """Sort by index (lazy)."""
        return LazySeries(self, 'sort_index', **kwargs)

    def sort_values(self, **kwargs):
        """Sort by values (lazy)."""
        return LazySeries(self, 'sort_values', **kwargs)

    def reset_index(self, **kwargs):
        """Reset index (lazy)."""
        return LazySeries(self, 'reset_index', **kwargs)

    def astype(self, dtype):
        """Cast to dtype (lazy)."""
        return LazySeries(self, 'astype', dtype)

    def copy(self, deep=True):
        """Return copy of result (lazy)."""
        return LazySeries(self, 'copy', deep=deep)

    def dropna(self, **kwargs):
        """Remove missing values (lazy)."""
        return LazySeries(self, 'dropna', **kwargs)

    def fillna(self, value=None, **kwargs):
        """Fill missing values (lazy)."""
        return LazySeries(self, 'fillna', value, **kwargs)

    def drop_duplicates(self, **kwargs):
        """Remove duplicate values (lazy)."""
        return LazySeries(self, 'drop_duplicates', **kwargs)

    def clip(self, lower=None, upper=None, **kwargs):
        """Clip values at thresholds (lazy)."""
        return LazySeries(self, 'clip', lower=lower, upper=upper, **kwargs)

    def abs(self):
        """Return absolute value (lazy)."""
        return LazySeries(self, 'abs')

    def round(self, decimals=0):
        """Round to given number of decimals (lazy)."""
        return LazySeries(self, 'round', decimals)

    def replace(self, to_replace=None, value=None, **kwargs):
        """Replace values (lazy)."""
        return LazySeries(self, 'replace', to_replace=to_replace, value=value, **kwargs)

    def where(self, cond, other=None, **kwargs):
        """Replace values where condition is False (lazy)."""
        return LazySeries(self, 'where', cond, other=other, **kwargs)

    def mask(self, cond, other=None, **kwargs):
        """Replace values where condition is True (lazy)."""
        return LazySeries(self, 'mask', cond, other=other, **kwargs)

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

    # ========== Aggregation Methods (lazy) ==========

    def sum(self, *args, **kwargs):
        """Compute sum of the result (lazy)."""
        return LazySeries(self, 'sum', *args, **kwargs)

    def mean(self, *args, **kwargs):
        """Compute mean of the result (lazy)."""
        return LazySeries(self, 'mean', *args, **kwargs)

    def min(self, *args, **kwargs):
        """Compute min of the result (lazy)."""
        return LazySeries(self, 'min', *args, **kwargs)

    def max(self, *args, **kwargs):
        """Compute max of the result (lazy)."""
        return LazySeries(self, 'max', *args, **kwargs)

    def std(self, *args, **kwargs):
        """Compute standard deviation of the result (lazy)."""
        return LazySeries(self, 'std', *args, **kwargs)

    def var(self, *args, **kwargs):
        """Compute variance of the result (lazy)."""
        return LazySeries(self, 'var', *args, **kwargs)

    def median(self, *args, **kwargs):
        """Compute median of the result (lazy)."""
        return LazySeries(self, 'median', *args, **kwargs)

    def count(self, *args, **kwargs):
        """Count non-NA values (lazy)."""
        return LazySeries(self, 'count', *args, **kwargs)
