"""
Pandas DataFrame compatibility layer for DataStore.

This module provides a lightweight wrapper around pandas DataFrame methods,
allowing DataStore to support the full pandas DataFrame API while maintaining
its query-building capabilities.

Key Design Principles:
1. Methods that return DataFrame in pandas return DataStore here
2. Internal DataFrame is cached to avoid redundant queries
3. Function signatures match pandas as closely as possible
4. Lightweight wrapper - delegate to pandas for actual implementation
5. Materialized DataStores (post-pandas operations) only use cached DataFrame

Execution Model:
- SQL operations: Build query state, don't execute
- First pandas operation: Execute SQL query, cache DataFrame, mark as materialized
- SQL operations after materialization: Use chDB's Python() table function to run SQL on DataFrame
- Subsequent operations: Work on cached DataFrame
- to_df(): Return cached DataFrame if materialized, otherwise execute SQL
"""

from typing import Optional
from copy import copy
import pandas as pd


def execute_sql_on_dataframe(df: pd.DataFrame, sql: str, df_var_name: str = '__datastore_df__') -> pd.DataFrame:
    """
    Execute SQL query on a pandas DataFrame using chDB.

    Args:
        df: pandas DataFrame to query
        sql: SQL query string (should reference the DataFrame as Python(df_var_name))
        df_var_name: Variable name to use for the DataFrame in SQL

    Returns:
        Result as pandas DataFrame

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> sql = "SELECT a, b FROM Python(__datastore_df__) WHERE a > 1"
        >>> result = execute_sql_on_dataframe(df, sql)
    """
    try:
        import chdb
    except ImportError:
        raise ImportError("chdb is required for SQL operations on materialized DataFrames")

    # Execute SQL with the DataFrame in local scope
    result = chdb.query(sql, 'DataFrame')

    return result


class PandasCompatMixin:
    """
    Mixin class that adds pandas DataFrame compatibility to DataStore.

    This mixin provides methods that mirror the pandas DataFrame API.
    Methods are organized into categories matching pandas documentation.
    """

    def _get_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get the internal DataFrame - triggers execution.

        This method is now a simple wrapper around _materialize() from core.py.

        Args:
            force_refresh: Ignored (no caching)

        Returns:
            pandas DataFrame
        """
        # Use _materialize() from core.py - always executes fresh
        return self._materialize()

    def _wrap_result(self, result, operation_name: str = None):
        """
        Wrap a pandas DataFrame result back into a DataStore, or return Series as-is.

        Design decision: Series are returned directly without wrapping to maintain
        pandas semantics and user expectations (e.g., df['column'] returns Series).

        Args:
            result: Result from pandas operation (DataFrame, Series, or other)
            operation_name: Name of the operation for tracking (optional)

        Returns:
            - DataStore if result is DataFrame
            - Series if result is Series (returned as-is)
            - Other types returned as-is
        """
        from .lazy_ops import LazyDataFrameSource

        if isinstance(result, pd.Series):
            # Return Series directly without wrapping
            # This maintains pandas semantics: df['column'] returns Series
            return result
        elif isinstance(result, pd.DataFrame):
            # Create a new DataStore with the DataFrame as source
            new_ds = copy(self)

            # Track the operation
            if operation_name:
                new_ds._track_operation('pandas', operation_name, {'shape': result.shape})

            # Generate new unique variable name for the new DataStore
            import uuid

            new_ds._df_var_name = f"__ds_df_{uuid.uuid4().hex}__"

            # Clear SQL query state and set DataFrame as the source
            new_ds._select_fields = []
            new_ds._where_condition = None
            new_ds._joins = []
            new_ds._groupby_fields = []
            new_ds._having_condition = None
            new_ds._orderby_fields = []
            new_ds._limit_value = None
            new_ds._offset_value = None
            new_ds._distinct = False
            new_ds._table_function = None
            new_ds.table_name = None

            # Set the DataFrame as the data source via lazy op
            new_ds._lazy_ops = [LazyDataFrameSource(result)]

            return new_ds
        return result

    # ========== Attributes and Properties ==========

    @property
    def dtypes(self):
        """Return the dtypes in the DataFrame."""
        return self._get_df().dtypes

    @property
    def values(self):
        """
        Return a Numpy representation of the DataFrame.

        Note: DataStore's INSERT VALUES method has been renamed to insert_values()
        to avoid conflict with this pandas-compatible property.
        """
        return self._get_df().values

    @property
    def axes(self):
        """Return a list representing the axes of the DataFrame."""
        return self._get_df().axes

    @property
    def ndim(self):
        """Return the number of dimensions."""
        return self._get_df().ndim

    @property
    def size(self):
        """Return the number of elements in the DataFrame."""
        return self._get_df().size

    @property
    def empty(self):
        """Indicator whether DataFrame is empty."""
        return self._get_df().empty

    @property
    def T(self):
        """Transpose index and columns."""
        return self._wrap_result(self._get_df().T)

    # ========== Indexing and Selection ==========

    @property
    def loc(self):
        """Access a group of rows and columns by label(s)."""
        return self._get_df().loc

    @property
    def iloc(self):
        """Access a group of rows and columns by integer position(s)."""
        return self._get_df().iloc

    @property
    def at(self):
        """Access a single value for a row/column label pair."""
        return self._get_df().at

    @property
    def iat(self):
        """Access a single value for a row/column pair by integer position."""
        return self._get_df().iat

    def __getitem__(self, key):
        """
        Get item - delegates to core.py for lazy behavior.

        Examples:
            >>> ds['column_name']  # Returns Field for expression building
            >>> ds[['col1', 'col2']]  # Lazy column selection
            >>> ds[ds.age > 18]  # For pandas compatibility when needed
        """
        # Delegate to parent class (DataStore in core.py)
        # This will handle lazy Field return and column selection
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """
        Lazy column assignment - does NOT execute immediately.

        This method records the operation and marks the cache as invalid.
        Actual execution happens when the DataStore is materialized.

        Args:
            key: Column name (string) to set/update
            value: Value to assign (can be scalar, Series, Expression, or array-like)

        Examples:
            >>> ds['new_column'] = 10  # Records operation (lazy)
            >>> ds['col'] = ds['other_column'] * 2  # Records expression (lazy)
            >>> ds['col'] = ds['col'] - 1  # Records operation (lazy)
            >>> print(ds)  # NOW it executes

        Note:
            This operation modifies the DataStore in-place (not immutable).
            The operation is recorded and will be executed during materialization.
        """
        from .lazy_ops import LazyColumnAssignment

        if not isinstance(key, str):
            raise TypeError(f"Column assignment requires string key, got {type(key)}")

        # Record the lazy operation
        self._lazy_ops.append(LazyColumnAssignment(key, value))

    def select_dtypes(self, include=None, exclude=None):
        """Return subset of columns based on column dtypes."""
        return self._wrap_result(self._get_df().select_dtypes(include=include, exclude=exclude))

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert column into DataFrame at specified location."""
        df = self._get_df().copy()
        df.insert(loc, column, value, allow_duplicates=allow_duplicates)
        return self._wrap_result(df)

    def pop(self, item):
        """Return item and drop from DataFrame."""
        df = self._get_df().copy()
        result = df.pop(item)
        # Return both the popped series (as DataStore) and modified df (as DataStore)
        return self._wrap_result(result)

    def xs(self, key, axis=0, level=None, drop_level=True):
        """Return cross-section from the DataFrame."""
        return self._wrap_result(self._get_df().xs(key, axis=axis, level=level, drop_level=drop_level))

    def get(self, key, default=None):
        """Get item from object for given key."""
        result = self._get_df().get(key, default=default)
        if result is not None:
            return self._wrap_result(result)
        return default

    def isin(self, values):
        """Whether each element is contained in values."""
        return self._wrap_result(self._get_df().isin(values))

    def where(self, cond, other=None, *, inplace=False, axis=None, level=None):
        """Replace values where condition is False."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().where(cond, other=other, axis=axis, level=level))

    def mask(self, cond, other=None, *, inplace=False, axis=None, level=None):
        """Replace values where condition is True."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().mask(cond, other=other, axis=axis, level=level))

    def query(self, expr, *, inplace=False, **kwargs):
        """Query the DataFrame with a boolean expression."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().query(expr, **kwargs))

    # ========== Statistical Methods ==========

    def abs(self):
        """Return a DataStore with absolute numeric value of each element."""
        return self._wrap_result(self._get_df().select_dtypes(include='number').abs())

    def all(self, axis=0, bool_only=None, skipna=True, **kwargs):
        """Return whether all elements are True over requested axis."""
        return self._get_df().all(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def any(self, axis=0, bool_only=None, skipna=True, **kwargs):
        """Return whether any element is True over requested axis."""
        return self._get_df().any(axis=axis, bool_only=bool_only, skipna=skipna, **kwargs)

    def clip(self, lower=None, upper=None, *, axis=None, inplace=False, **kwargs):
        """Trim values at input threshold(s)."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().clip(lower=lower, upper=upper, axis=axis, **kwargs))

    def corr(self, method='pearson', min_periods=1, numeric_only=True):
        """Compute pairwise correlation of columns."""
        return self._get_df().corr(method=method, min_periods=min_periods, numeric_only=numeric_only)

    def corrwith(self, other, axis=0, drop=False, method='pearson', numeric_only=True):
        """Compute pairwise correlation."""
        return self._get_df().corrwith(other, axis=axis, drop=drop, method=method, numeric_only=numeric_only)

    def cov(self, min_periods=None, ddof=1, numeric_only=True):
        """Compute pairwise covariance of columns."""
        return self._get_df().cov(min_periods=min_periods, ddof=ddof, numeric_only=numeric_only)

    def cummax(self, axis=0, skipna=True, **kwargs):
        """Return cumulative maximum over requested axis."""
        return self._wrap_result(self._get_df().cummax(axis=axis, skipna=skipna, **kwargs))

    def cummin(self, axis=0, skipna=True, **kwargs):
        """Return cumulative minimum over requested axis."""
        return self._wrap_result(self._get_df().cummin(axis=axis, skipna=skipna, **kwargs))

    def cumprod(self, axis=0, skipna=True, **kwargs):
        """Return cumulative product over requested axis."""
        return self._wrap_result(self._get_df().cumprod(axis=axis, skipna=skipna, **kwargs))

    def cumsum(self, axis=0, skipna=True, **kwargs):
        """Return cumulative sum over requested axis."""
        return self._wrap_result(self._get_df().cumsum(axis=axis, skipna=skipna, **kwargs))

    def diff(self, periods=1, axis=0):
        """First discrete difference of element."""
        return self._wrap_result(self._get_df().diff(periods=periods, axis=axis))

    def kurt(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return unbiased kurtosis over requested axis."""
        return self._get_df().kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def kurtosis(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return unbiased kurtosis over requested axis (alias for kurt)."""
        return self.kurt(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def mad(self, axis=0, skipna=True, **kwargs):
        """Return mean absolute deviation over requested axis."""
        return self._get_df().mad(axis=axis, skipna=skipna, **kwargs)

    def max(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return maximum over requested axis."""
        return self._get_df().max(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def mean(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return mean over requested axis."""
        return self._get_df().mean(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def median(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return median over requested axis."""
        return self._get_df().median(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def min(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return minimum over requested axis."""
        return self._get_df().min(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def mode(self, axis=0, numeric_only=True, dropna=True):
        """Get the mode(s) of each element along selected axis."""
        return self._wrap_result(self._get_df().mode(axis=axis, numeric_only=numeric_only, dropna=dropna))

    def pct_change(self, periods=1, fill_method='pad', limit=None, freq=None, **kwargs):
        """Percentage change between current and prior element."""
        return self._wrap_result(
            self._get_df().pct_change(periods=periods, fill_method=fill_method, limit=limit, freq=freq, **kwargs)
        )

    def prod(self, axis=0, skipna=True, numeric_only=True, min_count=0, **kwargs):
        """Return product over requested axis."""
        return self._get_df().prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    def product(self, axis=0, skipna=True, numeric_only=True, min_count=0, **kwargs):
        """Return product over requested axis (alias for prod)."""
        return self.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation='linear', method='single'):
        """Return values at given quantile over requested axis."""
        return self._get_df().quantile(
            q=q, axis=axis, numeric_only=numeric_only, interpolation=interpolation, method=method
        )

    def rank(self, axis=0, method='average', numeric_only=True, na_option='keep', ascending=True, pct=False):
        """Compute numerical data ranks along axis."""
        return self._wrap_result(
            self._get_df().rank(
                axis=axis, method=method, numeric_only=numeric_only, na_option=na_option, ascending=ascending, pct=pct
            )
        )

    def round(self, decimals=0, *args, **kwargs):
        """Round DataFrame to variable number of decimal places."""
        return self._wrap_result(self._get_df().round(decimals=decimals, *args, **kwargs))

    def sem(self, axis=0, skipna=True, ddof=1, numeric_only=True, **kwargs):
        """Return standard error of the mean over requested axis."""
        return self._get_df().sem(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def skew(self, axis=0, skipna=True, numeric_only=True, **kwargs):
        """Return unbiased skew over requested axis."""
        return self._get_df().skew(axis=axis, skipna=skipna, numeric_only=numeric_only, **kwargs)

    def std(self, axis=0, skipna=True, ddof=1, numeric_only=True, **kwargs):
        """Return standard deviation over requested axis."""
        return self._get_df().std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def sum(self, axis=0, skipna=True, numeric_only=True, min_count=0, **kwargs):
        """Return sum over requested axis."""
        return self._get_df().sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count, **kwargs)

    def var(self, axis=0, skipna=True, ddof=1, numeric_only=True, **kwargs):
        """Return variance over requested axis."""
        return self._get_df().var(axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only, **kwargs)

    def nunique(self, axis=0, dropna=True):
        """Count number of distinct elements in specified axis."""
        return self._get_df().nunique(axis=axis, dropna=dropna)

    def value_counts(self, subset=None, normalize=False, sort=True, ascending=False, dropna=True):
        """Return counts of unique rows."""
        return self._get_df().value_counts(
            subset=subset, normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
        )

    # ========== Data Manipulation ==========

    def drop(self, labels=None, *, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        """Drop specified labels from rows or columns."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, errors=errors)
        )

    def drop_duplicates(self, subset=None, keep='first', inplace=False, ignore_index=False):
        """Return DataFrame with duplicate rows removed."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index))

    def dropna(self, *, axis=0, how=None, thresh=None, subset=None, inplace=False):
        """Remove missing values."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        # Build kwargs to avoid passing both how and thresh
        kwargs = {'axis': axis, 'subset': subset}
        if thresh is not None:
            kwargs['thresh'] = thresh
        else:
            # Default how='any' if thresh is not set
            kwargs['how'] = how if how is not None else 'any'
        return self._wrap_result(self._get_df().dropna(**kwargs))

    def duplicated(self, subset=None, keep='first'):
        """Return boolean Series denoting duplicate rows."""
        return self._get_df().duplicated(subset=subset, keep=keep)

    def fillna(self, value=None, *, method=None, axis=None, inplace=False, limit=None, downcast=None):
        """Fill NA/NaN values."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().fillna(value=value, method=method, axis=axis, limit=limit, downcast=downcast)
        )

    def replace(self, to_replace=None, value=None, *, inplace=False, limit=None, regex=False, method=None):
        """Replace values given in to_replace with value."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().replace(to_replace=to_replace, value=value, limit=limit, regex=regex, method=method)
        )

    def interpolate(
        self,
        method='linear',
        *,
        axis=0,
        limit=None,
        inplace=False,
        limit_direction=None,
        limit_area=None,
        downcast=None,
        **kwargs,
    ):
        """Fill NaN values using interpolation method."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().interpolate(
                method=method,
                axis=axis,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
                downcast=downcast,
                **kwargs,
            )
        )

    def rename(
        self, mapper=None, *, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore'
    ):
        """Rename columns or index labels."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")

        # Build operation description
        op_desc = "rename("
        if columns:
            op_desc += f"columns={columns}"
        elif mapper:
            op_desc += f"mapper={mapper}"
        if index:
            op_desc += f", index={index}" if "=" in op_desc else f"index={index}"
        op_desc += ")"

        return self._wrap_result(
            self._get_df().rename(
                mapper=mapper, index=index, columns=columns, axis=axis, copy=copy, level=level, errors=errors
            ),
            op_desc,
        )

    def rename_axis(self, mapper=None, *, index=None, columns=None, axis=None, copy=True, inplace=False):
        """Set the name of the axis."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().rename_axis(mapper=mapper, index=index, columns=columns, axis=axis, copy=copy)
        )

    def reset_index(
        self, level=None, *, drop=False, inplace=False, col_level=0, col_fill='', allow_duplicates=False, names=None
    ):
        """Reset the index."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().reset_index(
                level=level,
                drop=drop,
                col_level=col_level,
                col_fill=col_fill,
                allow_duplicates=allow_duplicates,
                names=names,
            )
        )

    def set_index(self, keys, *, drop=True, append=False, inplace=False, verify_integrity=False):
        """Set the DataFrame index using existing columns."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().set_index(keys, drop=drop, append=append, verify_integrity=verify_integrity)
        )

    def sort_index(
        self,
        *,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind='quicksort',
        na_position='last',
        sort_remaining=True,
        ignore_index=False,
        key=None,
    ):
        """Sort object by labels along an axis."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().sort_index(
                axis=axis,
                level=level,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
                ignore_index=ignore_index,
                key=key,
            )
        )

    def sort_values(
        self,
        by,
        *,
        axis=0,
        ascending=True,
        inplace=False,
        kind='quicksort',
        na_position='last',
        ignore_index=False,
        key=None,
    ):
        """Sort by values along an axis."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(
            self._get_df().sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
        )

    def nlargest(self, n, columns, keep='first'):
        """Return first n rows ordered by columns in descending order."""
        return self._wrap_result(self._get_df().nlargest(n=n, columns=columns, keep=keep))

    def nsmallest(self, n, columns, keep='first'):
        """Return first n rows ordered by columns in ascending order."""
        return self._wrap_result(self._get_df().nsmallest(n=n, columns=columns, keep=keep))

    def assign(self, **kwargs):
        """Assign new columns to DataFrame."""
        return self._wrap_result(self._get_df().assign(**kwargs))

    # ========== Function Application ==========

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
        """Apply a function along an axis of the DataFrame."""
        result = self._get_df().apply(func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs)
        return self._wrap_result(result)

    def applymap(self, func, na_action=None, **kwargs):
        """Apply a function to a DataFrame elementwise."""
        return self._wrap_result(self._get_df().applymap(func, na_action=na_action, **kwargs))

    def map(self, func, na_action=None, **kwargs):
        """Apply a function to a DataFrame elementwise (alias for applymap)."""
        return self.applymap(func, na_action=na_action, **kwargs)

    def pipe(self, func, *args, **kwargs):
        """Apply chainable functions that expect Series or DataFrames."""
        result = self._get_df().pipe(func, *args, **kwargs)
        return self._wrap_result(result)

    def transform(self, func, axis=0, *args, **kwargs):
        """Call function on self producing a DataFrame with same axis shape."""
        result = self._get_df().transform(func, axis=axis, *args, **kwargs)
        return self._wrap_result(result)

    def agg(self, func=None, axis=0, *args, **kwargs):
        """Aggregate using one or more operations."""
        result = self._get_df().agg(func, axis=axis, *args, **kwargs)
        return self._wrap_result(result)

    def aggregate(self, func=None, axis=0, *args, **kwargs):
        """Aggregate using one or more operations (alias for agg)."""
        return self.agg(func, axis=axis, *args, **kwargs)

    # ========== Reshaping and Pivoting ==========

    def pivot(self, *, columns, index=None, values=None):
        """Return reshaped DataFrame organized by given index / column values."""
        return self._wrap_result(self._get_df().pivot(columns=columns, index=index, values=values))

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc='mean',
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name='All',
        observed=False,
        sort=True,
    ):
        """Create a spreadsheet-style pivot table."""
        return self._wrap_result(
            self._get_df().pivot_table(
                values=values,
                index=index,
                columns=columns,
                aggfunc=aggfunc,
                fill_value=fill_value,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
            )
        )

    def stack(self, level=-1, dropna=True):
        """Stack prescribed level(s) from columns to index."""
        return self._wrap_result(self._get_df().stack(level=level, dropna=dropna))

    def unstack(self, level=-1, fill_value=None):
        """Pivot level(s) of the index labels."""
        return self._wrap_result(self._get_df().unstack(level=level, fill_value=fill_value))

    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name='value', col_level=None, ignore_index=True):
        """Unpivot DataFrame from wide to long format."""
        return self._wrap_result(
            self._get_df().melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name,
                col_level=col_level,
                ignore_index=ignore_index,
            )
        )

    def explode(self, column, ignore_index=False):
        """Transform each element of a list-like to a row."""
        return self._wrap_result(self._get_df().explode(column=column, ignore_index=ignore_index))

    def transpose(self, *args, copy=False):
        """Transpose index and columns."""
        return self._wrap_result(self._get_df().transpose(*args, copy=copy))

    # ========== Combining / Joining / Merging ==========

    def append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        """Append rows of other to the end of caller."""
        return self._wrap_result(
            self._get_df().append(other, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)
        )

    def merge(
        self,
        right,
        how='inner',
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=('_x', '_y'),
        copy=True,
        indicator=False,
        validate=None,
    ):
        """Merge DataFrame objects."""
        # Convert right to DataFrame if it's a DataStore
        if hasattr(right, '_get_df'):
            right = right._get_df()
        return self._wrap_result(
            self._get_df().merge(
                right,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                sort=sort,
                suffixes=suffixes,
                copy=copy,
                indicator=indicator,
                validate=validate,
            )
        )

    def concat(
        self,
        objs,
        axis=0,
        join='outer',
        ignore_index=False,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        sort=False,
        copy=True,
    ):
        """Concatenate DataStore/DataFrame objects."""
        # Convert DataStore objects to DataFrames
        dfs = []
        for obj in objs:
            if hasattr(obj, '_get_df'):
                dfs.append(obj._get_df())
            else:
                dfs.append(obj)

        result = pd.concat(
            dfs,
            axis=axis,
            join=join,
            ignore_index=ignore_index,
            keys=keys,
            levels=levels,
            names=names,
            verify_integrity=verify_integrity,
            sort=sort,
            copy=copy,
        )
        return self._wrap_result(result)

    # ========== Time Series Methods ==========

    def resample(
        self,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention='start',
        kind=None,
        on=None,
        level=None,
        origin='start_day',
        offset=None,
        group_keys=False,
    ):
        """Resample time-series data."""
        return self._get_df().resample(
            rule,
            axis=axis,
            closed=closed,
            label=label,
            convention=convention,
            kind=kind,
            on=on,
            level=level,
            origin=origin,
            offset=offset,
            group_keys=group_keys,
        )

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        step=None,
        method='single',
    ):
        """Provide rolling window calculations."""
        return self._get_df().rolling(
            window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
            step=step,
            method=method,
        )

    def expanding(self, min_periods=1, axis=0, method='single'):
        """Provide expanding window calculations."""
        return self._get_df().expanding(min_periods=min_periods, axis=axis, method=method)

    def ewm(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        axis=0,
        times=None,
        method='single',
    ):
        """Provide exponentially weighted window calculations."""
        return self._get_df().ewm(
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
            method=method,
        )

    # ========== String Methods ==========

    @property
    def str(self):
        """Accessor object for string methods."""
        return self._get_df().str

    # ========== Datetime Methods ==========

    @property
    def dt(self):
        """Accessor object for datetime-like properties."""
        return self._get_df().dt

    # ========== Sparse Methods ==========

    @property
    def sparse(self):
        """Accessor object for sparse methods."""
        return self._get_df().sparse

    # ========== Plotting ==========

    @property
    def plot(self):
        """DataFrame plotting accessor and method."""
        return self._get_df().plot

    def hist(
        self,
        column=None,
        by=None,
        grid=True,
        xlabelsize=None,
        xrot=None,
        ylabelsize=None,
        yrot=None,
        ax=None,
        sharex=False,
        sharey=False,
        figsize=None,
        layout=None,
        bins=10,
        **kwargs,
    ):
        """Make a histogram of the DataFrame's columns."""
        return self._get_df().hist(
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            **kwargs,
        )

    def boxplot(
        self,
        column=None,
        by=None,
        ax=None,
        fontsize=None,
        rot=0,
        grid=True,
        figsize=None,
        layout=None,
        return_type=None,
        **kwargs,
    ):
        """Make a box plot from DataFrame columns."""
        return self._get_df().boxplot(
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            figsize=figsize,
            layout=layout,
            return_type=return_type,
            **kwargs,
        )

    # ========== IO Methods ==========

    def to_csv(
        self,
        path_or_buf=None,
        sep=',',
        na_rep='',
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode='w',
        encoding=None,
        compression='infer',
        quoting=None,
        quotechar='"',
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal='.',
        errors='strict',
        storage_options=None,
    ):
        """Write object to comma-separated values file."""
        return self._get_df().to_csv(
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            mode=mode,
            encoding=encoding,
            compression=compression,
            quoting=quoting,
            quotechar=quotechar,
            lineterminator=lineterminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
            errors=errors,
            storage_options=storage_options,
        )

    def to_excel(
        self,
        excel_writer,
        sheet_name='Sheet1',
        na_rep='',
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        startrow=0,
        startcol=0,
        engine=None,
        merge_cells=True,
        inf_rep='inf',
        freeze_panes=None,
        storage_options=None,
    ):
        """Write object to Excel sheet."""
        return self._get_df().to_excel(
            excel_writer,
            sheet_name=sheet_name,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            startrow=startrow,
            startcol=startcol,
            engine=engine,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
            freeze_panes=freeze_panes,
            storage_options=storage_options,
        )

    def to_json(
        self,
        path_or_buf=None,
        orient=None,
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit='ms',
        default_handler=None,
        lines=False,
        compression='infer',
        index=None,
        indent=None,
        storage_options=None,
        mode='w',
    ):
        """Convert object to JSON string."""
        # Default index based on orient
        if index is None:
            index = True if orient in [None, 'split', 'table', 'index', 'columns'] else False

        # Build kwargs for to_json, only include index for compatible orients
        kwargs = {
            'path_or_buf': path_or_buf,
            'orient': orient,
            'date_format': date_format,
            'double_precision': double_precision,
            'force_ascii': force_ascii,
            'date_unit': date_unit,
            'default_handler': default_handler,
            'lines': lines,
            'compression': compression,
            'indent': indent,
            'storage_options': storage_options,
            'mode': mode,
        }

        # Only pass index parameter for orients that support it
        if orient in ['split', 'table']:
            kwargs['index'] = index

        return self._get_df().to_json(**kwargs)

    def to_html(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep='NaN',
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal='.',
        bold_rows=True,
        classes=None,
        escape=True,
        notebook=False,
        border=None,
        table_id=None,
        render_links=False,
        encoding=None,
    ):
        """Render DataFrame as HTML table."""
        return self._get_df().to_html(
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            bold_rows=bold_rows,
            classes=classes,
            escape=escape,
            notebook=notebook,
            border=border,
            table_id=table_id,
            render_links=render_links,
            encoding=encoding,
        )

    def to_parquet(
        self,
        path=None,
        engine='auto',
        compression='snappy',
        index=None,
        partition_cols=None,
        storage_options=None,
        **kwargs,
    ):
        """Write DataFrame to parquet format."""
        return self._get_df().to_parquet(
            path=path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            storage_options=storage_options,
            **kwargs,
        )

    def to_feather(self, path, **kwargs):
        """Write DataFrame to feather format."""
        return self._get_df().to_feather(path, **kwargs)

    def to_sql(
        self,
        name,
        con,
        schema=None,
        if_exists='fail',
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):
        """Write records stored in DataFrame to SQL database."""
        return self._get_df().to_sql(
            name,
            con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    def to_pickle(self, path, compression='infer', protocol=5, storage_options=None):
        """Pickle (serialize) object to file."""
        return self._get_df().to_pickle(
            path, compression=compression, protocol=protocol, storage_options=storage_options
        )

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        """Copy object to system clipboard."""
        return self._get_df().to_clipboard(excel=excel, sep=sep, **kwargs)

    def to_markdown(self, buf=None, mode='wt', index=True, storage_options=None, **kwargs):
        """Print DataFrame in Markdown-friendly format."""
        return self._get_df().to_markdown(buf=buf, mode=mode, index=index, storage_options=storage_options, **kwargs)

    def to_latex(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep='NaN',
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        bold_rows=False,
        column_format=None,
        longtable=None,
        escape=None,
        encoding=None,
        decimal='.',
        multicolumn=None,
        multicolumn_format=None,
        multirow=None,
        caption=None,
        label=None,
        position=None,
    ):
        """Render object to LaTeX table."""
        return self._get_df().to_latex(
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            bold_rows=bold_rows,
            column_format=column_format,
            longtable=longtable,
            escape=escape,
            encoding=encoding,
            decimal=decimal,
            multicolumn=multicolumn,
            multicolumn_format=multicolumn_format,
            multirow=multirow,
            caption=caption,
            label=label,
            position=position,
        )

    def to_records(self, index=True, column_dtypes=None, index_dtypes=None):
        """Convert DataFrame to NumPy record array."""
        return self._get_df().to_records(index=index, column_dtypes=column_dtypes, index_dtypes=index_dtypes)

    def to_string(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep='NaN',
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        min_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal='.',
        line_width=None,
        max_colwidth=None,
        encoding=None,
    ):
        """Render DataFrame to console-friendly tabular output."""
        return self._get_df().to_string(
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            line_width=line_width,
            max_colwidth=max_colwidth,
            encoding=encoding,
        )

    def to_numpy(self, dtype=None, copy=False, na_value=None):
        """Convert DataFrame to NumPy array."""
        return self._get_df().to_numpy(dtype=dtype, copy=copy, na_value=na_value)

    # ========== Memory Methods ==========

    def memory_usage(self, index=True, deep=False):
        """Return memory usage of each column in bytes."""
        return self._get_df().memory_usage(index=index, deep=deep)

    # ========== Boolean Methods ==========

    def isna(self):
        """Detect missing values."""
        return self._wrap_result(self._get_df().isna())

    def isnull(self):
        """Detect missing values (alias for isna)."""
        return self.isna()

    def notna(self):
        """Detect existing (non-missing) values."""
        return self._wrap_result(self._get_df().notna())

    def notnull(self):
        """Detect existing (non-missing) values (alias for notna)."""
        return self.notna()

    # ========== Conversion Methods ==========

    def astype(self, dtype, copy=True, errors='raise'):
        """Cast object to specified dtype."""
        return self._wrap_result(self._get_df().astype(dtype=dtype, copy=copy, errors=errors))

    def convert_dtypes(
        self,
        infer_objects=True,
        convert_string=True,
        convert_integer=True,
        convert_boolean=True,
        convert_floating=True,
        dtype_backend='numpy_nullable',
    ):
        """Convert columns to best possible dtypes."""
        return self._wrap_result(
            self._get_df().convert_dtypes(
                infer_objects=infer_objects,
                convert_string=convert_string,
                convert_integer=convert_integer,
                convert_boolean=convert_boolean,
                convert_floating=convert_floating,
                dtype_backend=dtype_backend,
            )
        )

    def infer_objects(self, copy=True):
        """Attempt to infer better dtypes for object columns."""
        return self._wrap_result(self._get_df().infer_objects(copy=copy))

    def copy(self, deep=True):
        """Make a copy of this object's data."""
        return self._wrap_result(self._get_df().copy(deep=deep))

    def __copy__(self):
        """Shallow copy."""
        return super().__copy__()

    def __deepcopy__(self, memo):
        """Deep copy."""
        return copy(self)

    # ========== Iteration Methods ==========

    def items(self):
        """Iterate over (column name, Series) pairs."""
        return self._get_df().items()

    def iteritems(self):
        """Iterate over (column name, Series) pairs (alias for items)."""
        return self.items()

    def iterrows(self):
        """Iterate over DataFrame rows as (index, Series) pairs."""
        return self._get_df().iterrows()

    def itertuples(self, index=True, name='Pandas'):
        """Iterate over DataFrame rows as namedtuples."""
        return self._get_df().itertuples(index=index, name=name)

    # ========== Style Methods ==========

    @property
    def style(self):
        """Returns a Styler object."""
        return self._get_df().style

    # ========== Comparison Methods ==========

    def equals(self, other):
        """Test whether two objects contain same elements."""
        if hasattr(other, '_get_df'):
            other = other._get_df()
        return self._get_df().equals(other)

    def compare(self, other, align_axis=1, keep_shape=False, keep_equal=False, result_names=('self', 'other')):
        """Compare to another DataFrame and show differences."""
        if hasattr(other, '_get_df'):
            other = other._get_df()
        return self._wrap_result(
            self._get_df().compare(
                other, align_axis=align_axis, keep_shape=keep_shape, keep_equal=keep_equal, result_names=result_names
            )
        )

    # ========== Binary Operator Functions ==========

    def add(self, other, axis='columns', level=None, fill_value=None):
        """Addition of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().add(other, axis=axis, level=level, fill_value=fill_value))

    def sub(self, other, axis='columns', level=None, fill_value=None):
        """Subtraction of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().sub(other, axis=axis, level=level, fill_value=fill_value))

    def mul(self, other, axis='columns', level=None, fill_value=None):
        """Multiplication of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().mul(other, axis=axis, level=level, fill_value=fill_value))

    def div(self, other, axis='columns', level=None, fill_value=None):
        """Floating division of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().div(other, axis=axis, level=level, fill_value=fill_value))

    def truediv(self, other, axis='columns', level=None, fill_value=None):
        """Floating division of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().truediv(other, axis=axis, level=level, fill_value=fill_value))

    def floordiv(self, other, axis='columns', level=None, fill_value=None):
        """Integer division of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().floordiv(other, axis=axis, level=level, fill_value=fill_value))

    def mod(self, other, axis='columns', level=None, fill_value=None):
        """Modulo of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().mod(other, axis=axis, level=level, fill_value=fill_value))

    def pow(self, other, axis='columns', level=None, fill_value=None):
        """Exponential power of DataFrame and other element-wise."""
        return self._wrap_result(self._get_df().pow(other, axis=axis, level=level, fill_value=fill_value))

    def dot(self, other):
        """Matrix multiplication with DataFrame or Series."""
        return self._wrap_result(self._get_df().dot(other))

    def radd(self, other, axis='columns', level=None, fill_value=None):
        """Reverse addition of DataFrame and other."""
        return self._wrap_result(self._get_df().radd(other, axis=axis, level=level, fill_value=fill_value))

    def rsub(self, other, axis='columns', level=None, fill_value=None):
        """Reverse subtraction of DataFrame and other."""
        return self._wrap_result(self._get_df().rsub(other, axis=axis, level=level, fill_value=fill_value))

    def rmul(self, other, axis='columns', level=None, fill_value=None):
        """Reverse multiplication of DataFrame and other."""
        return self._wrap_result(self._get_df().rmul(other, axis=axis, level=level, fill_value=fill_value))

    def rdiv(self, other, axis='columns', level=None, fill_value=None):
        """Reverse floating division of DataFrame and other."""
        return self._wrap_result(self._get_df().rdiv(other, axis=axis, level=level, fill_value=fill_value))

    def rtruediv(self, other, axis='columns', level=None, fill_value=None):
        """Reverse floating division of DataFrame and other."""
        return self._wrap_result(self._get_df().rtruediv(other, axis=axis, level=level, fill_value=fill_value))

    def rfloordiv(self, other, axis='columns', level=None, fill_value=None):
        """Reverse integer division of DataFrame and other."""
        return self._wrap_result(self._get_df().rfloordiv(other, axis=axis, level=level, fill_value=fill_value))

    def rmod(self, other, axis='columns', level=None, fill_value=None):
        """Reverse modulo of DataFrame and other."""
        return self._wrap_result(self._get_df().rmod(other, axis=axis, level=level, fill_value=fill_value))

    def rpow(self, other, axis='columns', level=None, fill_value=None):
        """Reverse exponential power of DataFrame and other."""
        return self._wrap_result(self._get_df().rpow(other, axis=axis, level=level, fill_value=fill_value))

    def lt(self, other, axis='columns', level=None):
        """Less than comparison of DataFrame and other."""
        return self._wrap_result(self._get_df().lt(other, axis=axis, level=level))

    def gt(self, other, axis='columns', level=None):
        """Greater than comparison of DataFrame and other."""
        return self._wrap_result(self._get_df().gt(other, axis=axis, level=level))

    def le(self, other, axis='columns', level=None):
        """Less than or equal comparison of DataFrame and other."""
        return self._wrap_result(self._get_df().le(other, axis=axis, level=level))

    def ge(self, other, axis='columns', level=None):
        """Greater than or equal comparison of DataFrame and other."""
        return self._wrap_result(self._get_df().ge(other, axis=axis, level=level))

    def ne(self, other, axis='columns', level=None):
        """Not equal comparison of DataFrame and other."""
        return self._wrap_result(self._get_df().ne(other, axis=axis, level=level))

    def eq(self, other, axis='columns', level=None):
        """Equal comparison of DataFrame and other."""
        return self._wrap_result(self._get_df().eq(other, axis=axis, level=level))

    def combine(self, other, func, fill_value=None, overwrite=True):
        """Perform column-wise combine with another DataFrame."""
        return self._wrap_result(self._get_df().combine(other, func, fill_value=fill_value, overwrite=overwrite))

    def combine_first(self, other):
        """Update null elements with value from other DataFrame."""
        return self._wrap_result(self._get_df().combine_first(other))

    # ========== Additional Reindexing Methods ==========

    def add_prefix(self, prefix, axis=None):
        """Prefix labels with string prefix (lazy operation)."""
        from .lazy_ops import LazyAddPrefix

        self._lazy_ops.append(LazyAddPrefix(prefix))
        return self

    def add_suffix(self, suffix, axis=None):
        """Suffix labels with string suffix (lazy operation)."""
        from .lazy_ops import LazyAddSuffix

        self._lazy_ops.append(LazyAddSuffix(suffix))
        return self

    def align(self, other, **kwargs):
        """Align two objects on their axes."""
        result = self._get_df().align(other, **kwargs)
        if isinstance(result, tuple):
            return tuple(self._wrap_result(r) for r in result)
        return self._wrap_result(result)

    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=None,
        limit=None,
        tolerance=None,
    ):
        """Conform DataFrame to new index."""
        return self._wrap_result(
            self._get_df().reindex(
                labels=labels,
                index=index,
                columns=columns,
                axis=axis,
                method=method,
                copy=copy,
                level=level,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )
        )

    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        """Return object with matching indices."""
        return self._wrap_result(
            self._get_df().reindex_like(other, method=method, copy=copy, limit=limit, tolerance=tolerance)
        )

    def set_axis(self, labels, *, axis=0, copy=True):
        """Assign desired index to given axis."""
        return self._wrap_result(self._get_df().set_axis(labels, axis=axis, copy=copy))

    def take(self, indices, axis=0, **kwargs):
        """Return elements along given axis."""
        return self._wrap_result(self._get_df().take(indices, axis=axis, **kwargs))

    def truncate(self, before=None, after=None, axis=None, copy=True):
        """Truncate before and after some index values."""
        return self._wrap_result(self._get_df().truncate(before=before, after=after, axis=axis, copy=copy))

    def first(self, offset):
        """Select first periods of time series data based on date offset."""
        return self._wrap_result(self._get_df().first(offset))

    def last(self, offset):
        """Select final periods of time series data based on date offset."""
        return self._wrap_result(self._get_df().last(offset))

    def at_time(self, time, asof=False, axis=None):
        """Select values at particular time of day."""
        return self._wrap_result(self._get_df().at_time(time, asof=asof, axis=axis))

    def between_time(self, start_time, end_time, inclusive='both', axis=None):
        """Select values between particular times of day."""
        return self._wrap_result(self._get_df().between_time(start_time, end_time, inclusive=inclusive, axis=axis))

    # ========== Missing Data Methods ==========

    def backfill(self, *, axis=None, inplace=False, limit=None, downcast=None):
        """Backward fill missing values (alias for bfill)."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().backfill(axis=axis, limit=limit, downcast=downcast))

    def bfill(self, *, axis=None, inplace=False, limit=None, downcast=None):
        """Backward fill missing values."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().bfill(axis=axis, limit=limit, downcast=downcast))

    def ffill(self, *, axis=None, inplace=False, limit=None, downcast=None):
        """Forward fill missing values."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().ffill(axis=axis, limit=limit, downcast=downcast))

    def pad(self, *, axis=None, inplace=False, limit=None, downcast=None):
        """Forward fill missing values (alias for ffill)."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().pad(axis=axis, limit=limit, downcast=downcast))

    # ========== Additional Reshaping Methods ==========

    def droplevel(self, level, axis=0):
        """Return DataFrame with requested index/column level(s) removed."""
        return self._wrap_result(self._get_df().droplevel(level, axis=axis))

    def swaplevel(self, i=-2, j=-1, axis=0):
        """Swap levels i and j in a MultiIndex."""
        return self._wrap_result(self._get_df().swaplevel(i, j, axis=axis))

    def swapaxes(self, axis1, axis2, copy=True):
        """Interchange axes and swap values axes appropriately."""
        return self._wrap_result(self._get_df().swapaxes(axis1, axis2, copy=copy))

    def reorder_levels(self, order, axis=0):
        """Rearrange index levels using input order."""
        return self._wrap_result(self._get_df().reorder_levels(order, axis=axis))

    def squeeze(self, axis=None):
        """Squeeze 1-dimensional axis objects into scalars."""
        result = self._get_df().squeeze(axis=axis)
        # Only wrap if still a DataFrame/Series
        if isinstance(result, (pd.DataFrame, pd.Series)):
            return self._wrap_result(result)
        return result

    def to_xarray(self):
        """Return an xarray object from the pandas object."""
        return self._get_df().to_xarray()

    # ========== Time Series Methods ==========

    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        """Convert time series to specified frequency."""
        return self._wrap_result(
            self._get_df().asfreq(freq, method=method, how=how, normalize=normalize, fill_value=fill_value)
        )

    def asof(self, where, subset=None):
        """Return last row(s) without NaN before where."""
        return self._wrap_result(self._get_df().asof(where, subset=subset))

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift index by desired number of periods."""
        return self._wrap_result(self._get_df().shift(periods=periods, freq=freq, axis=axis, fill_value=fill_value))

    def first_valid_index(self):
        """Return index of first non-NA value."""
        return self._get_df().first_valid_index()

    def last_valid_index(self):
        """Return index of last non-NA value."""
        return self._get_df().last_valid_index()

    def to_period(self, freq=None, axis=0, copy=True):
        """Convert DataFrame to PeriodIndex."""
        return self._wrap_result(self._get_df().to_period(freq=freq, axis=axis, copy=copy))

    def to_timestamp(self, freq=None, how='start', axis=0, copy=True):
        """Cast to DatetimeIndex of timestamps."""
        return self._wrap_result(self._get_df().to_timestamp(freq=freq, how=how, axis=axis, copy=copy))

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        """Convert tz-aware axis to target time zone."""
        return self._wrap_result(self._get_df().tz_convert(tz, axis=axis, level=level, copy=copy))

    def tz_localize(self, tz, axis=0, level=None, copy=True, ambiguous='raise', nonexistent='raise'):
        """Localize tz-naive index to target time zone."""
        return self._wrap_result(
            self._get_df().tz_localize(
                tz, axis=axis, level=level, copy=copy, ambiguous=ambiguous, nonexistent=nonexistent
            )
        )

    # ========== Additional Combining Methods ==========

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='', sort=False, validate=None):
        """Join columns with other DataFrame."""
        if hasattr(other, '_get_df'):
            other = other._get_df()
        return self._wrap_result(
            self._get_df().join(other, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort, validate=validate)
        )

    def update(self, other, join='left', overwrite=True, filter_func=None, errors='ignore'):
        """Modify in place using non-NA values from another DataFrame."""
        # Since DataStore is immutable, we need to create a copy
        df = self._get_df().copy()
        df.update(other, join=join, overwrite=overwrite, filter_func=filter_func, errors=errors)
        return self._wrap_result(df)

    # ========== Additional Statistical Methods ==========

    def eval(self, expr, *, inplace=False, **kwargs):
        """Evaluate a string describing operations on DataFrame columns."""
        if inplace:
            raise ValueError("DataStore is immutable, inplace=True is not supported")
        return self._wrap_result(self._get_df().eval(expr, **kwargs))

    def idxmax(self, axis=0, skipna=True, numeric_only=False):
        """Return index of first occurrence of maximum over requested axis."""
        return self._get_df().idxmax(axis=axis, skipna=skipna, numeric_only=numeric_only)

    def idxmin(self, axis=0, skipna=True, numeric_only=False):
        """Return index of first occurrence of minimum over requested axis."""
        return self._get_df().idxmin(axis=axis, skipna=skipna, numeric_only=numeric_only)

    # ========== Additional IO Methods ==========

    def to_hdf(
        self,
        path_or_buf,
        *,
        key,
        mode='a',
        complevel=None,
        complib=None,
        append=False,
        format=None,
        index=True,
        min_itemsize=None,
        nan_rep=None,
        dropna=None,
        data_columns=None,
        errors='strict',
        encoding='UTF-8',
    ):
        """Write to HDF5 file."""
        return self._get_df().to_hdf(
            path_or_buf,
            key=key,
            mode=mode,
            complevel=complevel,
            complib=complib,
            append=append,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )

    def to_stata(
        self,
        path,
        *,
        convert_dates=None,
        write_index=True,
        byteorder=None,
        time_stamp=None,
        data_label=None,
        variable_labels=None,
        version=114,
        convert_strl=None,
        compression='infer',
        storage_options=None,
        **kwargs,
    ):
        """Export DataFrame to Stata dta format."""
        return self._get_df().to_stata(
            path,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            version=version,
            convert_strl=convert_strl,
            compression=compression,
            storage_options=storage_options,
            **kwargs,
        )

    def to_gbq(
        self,
        destination_table,
        *,
        project_id=None,
        chunksize=None,
        reauth=False,
        if_exists='fail',
        auth_local_webserver=True,
        table_schema=None,
        location=None,
        progress_bar=True,
        credentials=None,
    ):
        """Write DataFrame to Google BigQuery table."""
        return self._get_df().to_gbq(
            destination_table,
            project_id=project_id,
            chunksize=chunksize,
            reauth=reauth,
            if_exists=if_exists,
            auth_local_webserver=auth_local_webserver,
            table_schema=table_schema,
            location=location,
            progress_bar=progress_bar,
            credentials=credentials,
        )

    def to_orc(self, path=None, *, engine='pyarrow', index=None, engine_kwargs=None):
        """Write DataFrame to ORC format."""
        return self._get_df().to_orc(path=path, engine=engine, index=index, engine_kwargs=engine_kwargs)

    # ========== Class Methods ==========

    @classmethod
    def from_dict(cls, data, orient='columns', dtype=None, columns=None):
        """Construct DataFrame from dict of array-like or dicts."""
        df = pd.DataFrame.from_dict(data, orient=orient, dtype=dtype, columns=columns)
        # This is a class method, so we need to create a DataStore differently
        # For now, return the DataFrame wrapped in a basic way
        return df

    @classmethod
    def from_records(cls, data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None):
        """Convert structured or record ndarray to DataFrame."""
        df = pd.DataFrame.from_records(
            data, index=index, exclude=exclude, columns=columns, coerce_float=coerce_float, nrows=nrows
        )
        return df
