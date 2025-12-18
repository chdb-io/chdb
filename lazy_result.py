"""
Lazy Result Classes for DataStore.

This module provides lazy wrappers for results that should not be materialized
immediately. These classes maintain the lazy evaluation chain and only execute
when display or explicit conversion is triggered.

Key classes:
- LazySlice: Wraps head()/tail() operations for lazy evaluation
"""

from typing import TYPE_CHECKING, Union, Optional, Any
import pandas as pd

if TYPE_CHECKING:
    from .column_expr import ColumnExpr, LazyAggregate
    from .core import DataStore


class LazySlice:
    """
    A lazy slice of a Series/DataFrame that executes only when displayed.

    Wraps head()/tail() operations to maintain lazy evaluation chain.
    This allows SQL LIMIT optimization and consistent lazy behavior.

    Example:
        >>> ds['age'].head(5)  # Returns LazySlice, not pd.Series
        LazySlice(ColumnExpr("age"), head=5)

        >>> print(ds['age'].head(5))  # Triggers materialization
        0    28
        1    31
        2    29
        3    45
        4    22
        Name: age, dtype: int64

        >>> # Chainable
        >>> ds['age'].head(10).tail(3)  # Still lazy

    Attributes:
        _source: The source object (ColumnExpr, LazyAggregate, LazySlice, etc.)
        _slice_type: Type of slice ('head' or 'tail')
        _n: Number of elements to return
    """

    def __init__(
        self,
        source: Union['ColumnExpr', 'LazyAggregate', 'LazySlice', pd.Series, pd.DataFrame],
        slice_type: str,
        n: int = 5,
    ):
        """
        Initialize a LazySlice.

        Args:
            source: The source object to slice
            slice_type: Type of slice ('head' or 'tail')
            n: Number of elements to return (default 5)
        """
        if slice_type not in ('head', 'tail'):
            raise ValueError(f"slice_type must be 'head' or 'tail', got '{slice_type}'")

        self._source = source
        self._slice_type = slice_type
        self._n = n
        self._cached_result = None

    @property
    def _datastore(self) -> Optional['DataStore']:
        """Get the DataStore reference if available."""
        if hasattr(self._source, '_datastore'):
            return self._source._datastore
        return None

    def _get_result(self):
        """
        Execute and cache the result.

        Returns:
            pd.Series or pd.DataFrame: The sliced result
        """
        if self._cached_result is not None:
            return self._cached_result

        # Get the source result
        source_result = self._get_source_result()

        # Apply slice
        if self._slice_type == 'head':
            if hasattr(source_result, 'head'):
                self._cached_result = source_result.head(self._n)
            else:
                # Scalar or non-sliceable - return as is
                self._cached_result = source_result
        elif self._slice_type == 'tail':
            if hasattr(source_result, 'tail'):
                self._cached_result = source_result.tail(self._n)
            else:
                self._cached_result = source_result

        return self._cached_result

    def _get_source_result(self):
        """Get the materialized result from the source."""
        if hasattr(self._source, '_get_result'):
            # Another LazySlice or similar
            return self._source._get_result()
        elif hasattr(self._source, '_materialize'):
            # ColumnExpr
            return self._source._materialize()
        elif hasattr(self._source, '_execute'):
            # LazyAggregate
            return self._source._execute()
        elif isinstance(self._source, (pd.Series, pd.DataFrame)):
            return self._source
        else:
            # Unknown type - try to use directly
            return self._source

    # ========== Display Methods (trigger execution) ==========

    def __repr__(self) -> str:
        """Return representation showing actual values."""
        try:
            result = self._get_result()
            return repr(result)
        except Exception as e:
            return f"LazySlice({self._slice_type}={self._n}) [Error: {e}]"

    def __str__(self) -> str:
        """Return string representation showing actual values."""
        try:
            result = self._get_result()
            return str(result)
        except Exception:
            return f"LazySlice({self._slice_type}={self._n})"

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            result = self._get_result()
            if hasattr(result, '_repr_html_'):
                return result._repr_html_()
            return f"<pre>{repr(result)}</pre>"
        except Exception as e:
            return f"<pre>LazySlice({self._slice_type}={self._n}) [Error: {e}]</pre>"

    # ========== Chainable Methods (return new LazySlice) ==========

    def head(self, n: int = 5) -> 'LazySlice':
        """
        Return the first n elements (lazy).

        Can be chained: ds['col'].head(10).head(5)

        Args:
            n: Number of elements to return (default 5)

        Returns:
            LazySlice: New lazy wrapper
        """
        return LazySlice(self, 'head', n)

    def tail(self, n: int = 5) -> 'LazySlice':
        """
        Return the last n elements (lazy).

        Can be chained: ds['col'].head(10).tail(5)

        Args:
            n: Number of elements to return (default 5)

        Returns:
            LazySlice: New lazy wrapper
        """
        return LazySlice(self, 'tail', n)

    # ========== Terminal Methods (trigger execution) ==========

    def to_pandas(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Explicitly materialize to pandas Series/DataFrame.

        Returns:
            pd.Series or pd.DataFrame: The materialized result
        """
        return self._get_result()

    def to_series(self) -> pd.Series:
        """
        Explicitly materialize to pandas Series.

        Returns:
            pd.Series: The materialized result
        """
        result = self._get_result()
        if isinstance(result, pd.DataFrame):
            if len(result.columns) == 1:
                return result.iloc[:, 0]
            raise ValueError("Cannot convert multi-column DataFrame to Series")
        return result

    def tolist(self) -> list:
        """
        Convert to list.

        Returns:
            list: The values as a list
        """
        result = self._get_result()
        if hasattr(result, 'tolist'):
            return result.tolist()
        return list(result)

    def to_numpy(self):
        """
        Convert to numpy array.

        Returns:
            numpy.ndarray: The values as a numpy array
        """
        result = self._get_result()
        if hasattr(result, 'to_numpy'):
            return result.to_numpy()
        import numpy as np

        return np.array(result)

    # ========== Series-like Properties (trigger execution) ==========

    @property
    def values(self):
        """Get the values of the result."""
        result = self._get_result()
        if hasattr(result, 'values'):
            return result.values
        return result

    @property
    def index(self):
        """Get the index of the result."""
        result = self._get_result()
        if hasattr(result, 'index'):
            return result.index
        raise AttributeError("Result has no 'index' attribute")

    @property
    def dtype(self):
        """Get the dtype of the result."""
        result = self._get_result()
        if hasattr(result, 'dtype'):
            return result.dtype
        import numpy as np

        return np.dtype(type(result))

    @property
    def name(self):
        """Get the name of the result Series."""
        result = self._get_result()
        if hasattr(result, 'name'):
            return result.name
        return None

    @property
    def shape(self):
        """Get the shape of the result."""
        result = self._get_result()
        if hasattr(result, 'shape'):
            return result.shape
        return (1,)

    # ========== Iteration & Indexing (trigger execution) ==========

    def __len__(self) -> int:
        """Get length of the result."""
        result = self._get_result()
        if hasattr(result, '__len__'):
            return len(result)
        return 1

    def __iter__(self):
        """Iterate over the result."""
        result = self._get_result()
        if hasattr(result, '__iter__'):
            return iter(result)
        return iter([result])

    def __getitem__(self, key):
        """Support indexing/subscripting."""
        result = self._get_result()
        return result[key]

    def __contains__(self, item) -> bool:
        """Support 'in' operator."""
        result = self._get_result()
        return item in result

    # ========== Numeric Conversions (trigger execution) ==========

    def __float__(self) -> float:
        """Convert to float."""
        result = self._get_result()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 1:
                return float(result.iloc[0])
            raise ValueError("Cannot convert multi-element result to float")
        return float(result)

    def __int__(self) -> int:
        """Convert to int."""
        result = self._get_result()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 1:
                return int(result.iloc[0])
            raise ValueError("Cannot convert multi-element result to int")
        return int(result)

    def __bool__(self) -> bool:
        """Convert to bool."""
        result = self._get_result()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            if len(result) == 0:
                return False
            if len(result) == 1:
                return bool(result.iloc[0])
            raise ValueError("Truth value of multi-element result is ambiguous")
        return bool(result)

    def __array__(self, dtype=None):
        """Support numpy array protocol."""
        import numpy as np

        result = self._get_result()
        if isinstance(result, (pd.Series, pd.DataFrame)):
            arr = result.values
        else:
            arr = np.array([result])
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    # ========== Comparison Operators (trigger execution) ==========

    def __eq__(self, other):
        """Equal comparison."""
        return self._get_result() == other

    def __ne__(self, other):
        """Not equal comparison."""
        return self._get_result() != other

    def __lt__(self, other):
        """Less than comparison."""
        return self._get_result() < other

    def __le__(self, other):
        """Less than or equal comparison."""
        return self._get_result() <= other

    def __gt__(self, other):
        """Greater than comparison."""
        return self._get_result() > other

    def __ge__(self, other):
        """Greater than or equal comparison."""
        return self._get_result() >= other

    # ========== Arithmetic Operators (trigger execution) ==========

    def __add__(self, other):
        """Addition."""
        return self._get_result() + other

    def __radd__(self, other):
        """Right addition."""
        return other + self._get_result()

    def __sub__(self, other):
        """Subtraction."""
        return self._get_result() - other

    def __rsub__(self, other):
        """Right subtraction."""
        return other - self._get_result()

    def __mul__(self, other):
        """Multiplication."""
        return self._get_result() * other

    def __rmul__(self, other):
        """Right multiplication."""
        return other * self._get_result()

    def __truediv__(self, other):
        """Division."""
        return self._get_result() / other

    def __rtruediv__(self, other):
        """Right division."""
        return other / self._get_result()

    def __floordiv__(self, other):
        """Floor division."""
        return self._get_result() // other

    def __rfloordiv__(self, other):
        """Right floor division."""
        return other // self._get_result()

    def __mod__(self, other):
        """Modulo."""
        return self._get_result() % other

    def __rmod__(self, other):
        """Right modulo."""
        return other % self._get_result()

    def __pow__(self, other):
        """Power."""
        return self._get_result() ** other

    def __rpow__(self, other):
        """Right power."""
        return other ** self._get_result()

    def __neg__(self):
        """Negation."""
        return -self._get_result()

    def __pos__(self):
        """Positive."""
        return +self._get_result()

    def __abs__(self):
        """Absolute value."""
        return abs(self._get_result())

    # ========== Aggregation Methods (trigger execution, return scalar) ==========

    def sum(self, *args, **kwargs):
        """Compute sum of the sliced result."""
        return self._get_result().sum(*args, **kwargs)

    def mean(self, *args, **kwargs):
        """Compute mean of the sliced result."""
        return self._get_result().mean(*args, **kwargs)

    def min(self, *args, **kwargs):
        """Compute min of the sliced result."""
        return self._get_result().min(*args, **kwargs)

    def max(self, *args, **kwargs):
        """Compute max of the sliced result."""
        return self._get_result().max(*args, **kwargs)

    def std(self, *args, **kwargs):
        """Compute standard deviation of the sliced result."""
        return self._get_result().std(*args, **kwargs)

    def var(self, *args, **kwargs):
        """Compute variance of the sliced result."""
        return self._get_result().var(*args, **kwargs)

    def median(self, *args, **kwargs):
        """Compute median of the sliced result."""
        return self._get_result().median(*args, **kwargs)

    def count(self, *args, **kwargs):
        """Count non-NA values."""
        return self._get_result().count(*args, **kwargs)
