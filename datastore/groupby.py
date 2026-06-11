"""
LazyGroupBy - Lazy GroupBy wrapper for DataStore.

This implements pandas-like groupby semantics where:
- df.groupby('col') returns a GroupBy object (not a copy of DataFrame)
- The GroupBy object references the original DataFrame
- Aggregation operations execute and checkpoint the ORIGINAL DataFrame

This design ensures that:
1. When groupby().agg() executes, the original df is checkpointed
2. Subsequent calls to df.to_df() use the cached result
3. No redundant computation occurs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Any

import pandas as pd

from .expressions import Expression, Field

if TYPE_CHECKING:
    from .core import DataStore
    from .column_expr import ColumnExpr
    from .lazy_result import LazySeries


class LazyGroupBy:
    """
    A GroupBy wrapper for DataStore that references the original DataStore.

    Similar to pandas DataFrameGroupBy, this object:
    - Holds a reference to the original DataStore (not a copy)
    - Stores the groupby fields
    - Provides aggregation methods that operate on the original DataStore

    Example:
        >>> df = DataStore.from_dataframe(pd.DataFrame(...))
        >>> df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        >>>
        >>> grp = df.groupby('FamilySize')  # Returns LazyGroupBy
        >>> result = grp['Survived'].mean()  # Executes ORIGINAL df
        >>>
        >>> df.to_df()  # Uses cached result (no re-execution!)
    """

    def __init__(
        self,
        datastore: 'DataStore',
        groupby_fields: List[Expression],
        sort: bool = True,
        as_index: bool = True,
        dropna: bool = True,
        selected_columns: List[str] = None,
    ):
        """
        Initialize LazyGroupBy.

        Args:
            datastore: Reference to the ORIGINAL DataStore (not a copy)
            groupby_fields: List of Field expressions to group by
            sort: Sort group keys (default: True, matching pandas behavior).
                  When True, the result is sorted by group keys in ascending order.
            as_index: If True (default), group keys become the index.
                      If False, group keys are returned as columns.
            dropna: If True (default), exclude NA/null values in keys.
                    If False, NA values are also grouped. Matches pandas default.
            selected_columns: List of column names to aggregate (None = all columns).
                              When set via groupby()[['col1', 'col2']], only these columns
                              are included in aggregation results.
        """
        self._datastore = datastore
        self._groupby_fields = groupby_fields.copy()  # Copy the list, not the DataStore
        self._sort = sort
        self._as_index = as_index
        self._dropna = dropna
        self._selected_columns = selected_columns

    @property
    def datastore(self) -> 'DataStore':
        """Get the original DataStore reference."""
        return self._datastore

    @property
    def groupby_fields(self) -> List[Expression]:
        """Get the groupby fields."""
        return self._groupby_fields

    @property
    def ngroups(self) -> int:
        """
        Return the number of groups.

        Returns:
            int: Number of unique groups based on groupby keys.

        Example:
            >>> df.groupby('category').ngroups
            3
        """
        return self._pandas_groupby().ngroups

    def _groupby_col_names(self) -> List[str]:
        """Resolve groupby Expression list to plain column-name strings."""
        cols: List[str] = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                cols.append(gf.name)
            else:
                cols.append(str(gf))
        return cols

    def _pandas_groupby(self):
        """Materialize underlying DataStore and return a pandas GroupBy object.

        Shared helper used by iteration, get_group(), groups, indices, ngroups,
        len(), and ``in``. Respects ``sort``, ``dropna``, and ``selected_columns``
        so behaviour matches pandas DataFrameGroupBy exactly.

        Note: this triggers execution of the underlying DataStore. Each call
        re-executes; callers in tight loops should hoist the result.
        """
        groupby_cols = self._groupby_col_names()
        df = self._datastore._get_df()
        # When grouping by a single column, pass the column name as a scalar to
        # pandas (not a one-element list). This makes pandas yield scalar group
        # keys (e.g. ``'A'``) instead of one-tuple keys (e.g. ``('A',)``),
        # matching the dominant pandas convention ``df.groupby('col')`` and
        # what users hit when chaining ``ds.groupby('col')`` from DataStore.
        by = groupby_cols[0] if len(groupby_cols) == 1 else groupby_cols
        pandas_gb = df.groupby(by, sort=self._sort, dropna=self._dropna)
        if self._selected_columns is not None:
            # Mirrors df.groupby(...)[ [cols] ] semantics: yielded sub-DataFrames
            # only contain the selected columns, not the groupby key columns.
            return pandas_gb[self._selected_columns]
        return pandas_gb

    def __iter__(self):
        """Iterate over ``(group_key, sub_DataFrame)`` pairs - pandas semantics.

        Mirrors :py:meth:`pandas.core.groupby.DataFrameGroupBy.__iter__`:

        - Single-column groupby: yields ``(key, sub_df)`` where ``key`` is a scalar.
        - Multi-column groupby:  yields ``((k1, k2, ...), sub_df)`` where the key
          is a tuple of values matching the groupby columns.
        - ``sub_df`` is a real ``pd.DataFrame`` containing the rows for that
          group, preserving the original index of the source DataFrame.
        - If ``selected_columns`` was set via ``gb[['c1', 'c2']]``, the yielded
          sub-DataFrames only contain those columns.

        Example:
            >>> for (date, code), group in ds.groupby(['date', 'code']):
            ...     print(date, code, len(group))
        """
        return iter(self._pandas_groupby())

    def __len__(self) -> int:
        """Return number of groups (matches pandas: ``len(gb) == gb.ngroups``)."""
        return self.ngroups

    def __contains__(self, key) -> bool:
        """Return whether ``key`` is one of the group labels.

        Matches pandas ``key in gb`` semantics: for multi-column groupby the
        key should be a tuple of values matching the groupby columns.
        """
        return key in self._pandas_groupby().groups

    @property
    def groups(self) -> dict:
        """Mapping from group label to the index labels in that group.

        Mirrors :py:attr:`pandas.core.groupby.DataFrameGroupBy.groups`:
        returns ``dict`` of ``{group_key: Index(...)}`` where each value is the
        labels (not positions) of the rows in that group, using the original
        DataFrame's index.

        Example:
            >>> ds.groupby('category').groups
            {'A': Index([0, 2, 4]), 'B': Index([1, 3, 5])}
        """
        return self._pandas_groupby().groups

    @property
    def indices(self) -> dict:
        """Mapping from group label to the positional locations in that group.

        Mirrors :py:attr:`pandas.core.groupby.DataFrameGroupBy.indices`:
        returns ``dict`` of ``{group_key: np.ndarray(positions)}`` where each
        value gives integer row positions (0-based, after any reset) of the
        rows in that group.

        Example:
            >>> ds.groupby('category').indices
            {'A': array([0, 2, 4]), 'B': array([1, 3, 5])}
        """
        return self._pandas_groupby().indices

    def get_group(self, name, obj=None) -> pd.DataFrame:
        """Return the rows in the ``name`` group as a DataFrame.

        Mirrors :py:meth:`pandas.core.groupby.DataFrameGroupBy.get_group`.

        Args:
            name: Group key. For single-column groupby this is a scalar matching
                  the groupby column's value. For multi-column groupby it must
                  be a tuple of values matching the groupby columns in order.
            obj: Retained for pandas API compatibility (deprecated/removed in
                 recent pandas versions); ignored here.

        Returns:
            pd.DataFrame: Sub-DataFrame for the requested group, preserving the
            original index. If ``selected_columns`` was set via
            ``gb[['c1', 'c2']]``, only those columns are included.

        Raises:
            KeyError: If ``name`` is not a valid group label.

        Example:
            >>> ds.groupby('category').get_group('A')
            >>> ds.groupby(['date', 'code']).get_group(('2026-05-23', '000001'))
        """
        del obj  # unused, accepted for pandas API parity
        return self._pandas_groupby().get_group(name)

    def __getitem__(self, key: Union[str, List[str]]) -> 'ColumnExpr':
        """
        Access a column or columns for aggregation.

        Returns a ColumnExpr that references the ORIGINAL DataStore
        with the groupby fields set.

        Args:
            key: Column name (str) or list of column names

        Returns:
            ColumnExpr for single column, or LazyGroupBy for multiple
        """
        from .column_expr import ColumnExpr

        if isinstance(key, str):
            # Single column - return ColumnExpr with groupby fields attached
            # This avoids polluting _datastore._groupby_fields which would affect to_df()
            # Also pass as_index, sort, and dropna parameters for proper aggregation behavior
            field = Field(key)
            return ColumnExpr(
                field,
                self._datastore,
                groupby_fields=self._groupby_fields.copy(),
                groupby_as_index=self._as_index,
                groupby_sort=self._sort,
                groupby_dropna=self._dropna,
            )

        elif isinstance(key, list):
            # Multiple columns - return new LazyGroupBy with selected_columns tracked
            return LazyGroupBy(
                self._datastore,
                self._groupby_fields,
                sort=self._sort,
                as_index=self._as_index,
                dropna=self._dropna,
                selected_columns=key,
            )

        else:
            if isinstance(key, int):
                raise TypeError(
                    f"LazyGroupBy does not support integer indexing (got {key!r}). "
                    "Use ``for key, group in groupby:`` to iterate over groups, "
                    "or ``groupby.get_group(key)`` to access a specific group."
                )
            raise TypeError(
                "LazyGroupBy column selection expects a str or list of str, "
                f"got {type(key).__name__}"
            )

    def __getattr__(self, name: str):
        """
        Support attribute access for column names.

        Example:
            >>> grp.Survived.mean()  # Same as grp['Survived'].mean()
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        return self[name]

    def __setitem__(self, key, value):
        """
        Support column assignment after groupby.

        This delegates to the underlying datastore's __setitem__ for backward
        compatibility. Note: in pandas, you cannot assign to a GroupBy object.

        Example:
            >>> grp = ds.groupby('category')
            >>> grp['new_col'] = grp['value'] * 2  # Assigns to underlying ds
        """
        self._datastore[key] = value

    # ========== Aggregation Methods ==========

    def agg(self, func=None, **kwargs):
        """
        Aggregate using one or more operations.

        Supports three modes:
        1. Pandas-style aggregation with dict:
           >>> df.groupby('dept').agg({'salary': 'mean', 'age': 'max'})

        2. Pandas named aggregation (tuple syntax):
           >>> df.groupby('dept').agg(avg_salary=('salary', 'mean'), max_age=('age', 'max'))

        3. SQL-style aggregation with keyword arguments:
           >>> from datastore import col
           >>> df.groupby('dept').agg(avg_salary=col('salary').mean())

        Args:
            func: Aggregation function(s) - dict or string
            **kwargs: Named aggregate expressions or named tuples

        Returns:
            DataStore with lazy aggregation
        """
        from .column_expr import ColumnExpr
        from .exceptions import QueryError
        from .functions import AggregateFunction
        from .pandas_col_compat import (
            PandasFallbackExpr,
            is_pandas_col_expression,
            translate_pandas_expression,
        )

        # Translate ``pd.col(...).agg_fn()`` values into chdb-ds
        # AggregateFunction nodes; otherwise the has_sql_agg / has_named_agg
        # checks below miss them and the kwargs are silently dropped.
        if kwargs:
            kwargs = {
                alias: (
                    translate_pandas_expression(value)
                    if is_pandas_col_expression(value)
                    else value
                )
                for alias, value in kwargs.items()
            }

        # Reject untranslatable pd.col aggregates up front. Without this,
        # PandasFallbackExpr fails every isinstance check below and the
        # kwargs are silently dropped into _create_lazy_agg_datastore,
        # producing an *empty* DataFrame (Columns: []) — worse than crashing.
        # pandas itself does not support agg(kw=pd.col(...).mean()) either,
        # so falling back to pandas here is not an option. Mirrors the same
        # check in DataStore.agg for parity.
        for alias, value in kwargs.items():
            if isinstance(value, PandasFallbackExpr):
                raise QueryError(
                    f"Invalid aggregate expression for '{alias}': "
                    f"{value.original!r} contains operations that cannot be "
                    f"pushed to SQL (e.g. .astype, numpy ufunc, .apply). "
                    f"SQL aggregations require pushable expressions; rewrite "
                    f"using col(...) arithmetic / accessors only, or "
                    f"precompute the column via assign() first."
                )

        # Check if we have SQL-style keyword arguments with expressions
        has_sql_agg = any(isinstance(v, (Expression, ColumnExpr, AggregateFunction)) for v in kwargs.values())

        if has_sql_agg:
            # SQL-style aggregation: agg(alias=col("x").sum(), ...)
            # Set groupby fields on original datastore and delegate
            self._datastore._groupby_fields = self._groupby_fields.copy()
            return self._datastore.agg(func, **kwargs)

        # Check for pandas named aggregation: agg(alias=('col', 'func'))
        # This is when kwargs contains tuples of (column, aggfunc)
        has_named_agg = kwargs and all(
            isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], str)
            for v in kwargs.values()
        )

        if has_named_agg:
            # Convert named aggregation to pandas format
            # pandas expects: agg(**kwargs) where kwargs={'alias': ('col', 'func')}
            return self._create_lazy_agg_datastore(agg_dict=None, named_agg=kwargs)
        elif isinstance(func, str):
            # Single function name: agg('sum'), agg('mean'), etc.
            # Pass as agg_func (single function for all numeric columns)
            return self._create_lazy_agg_datastore(agg_func=func, **kwargs)
        else:
            # Pandas-style aggregation: agg({'col': 'func'}) or agg({'col': ['func1', 'func2']})
            # Return lazy DataStore with LazyGroupByAgg operation
            return self._create_lazy_agg_datastore(agg_dict=func, **kwargs)

    def mean(self, numeric_only: bool = False) -> 'DataStore':
        """Compute mean of groups. Returns lazy DataStore."""
        return self._apply_agg('mean', numeric_only=numeric_only)

    def sum(self, numeric_only: bool = False) -> 'DataStore':
        """Compute sum of groups. Returns lazy DataStore."""
        return self._apply_agg('sum', numeric_only=numeric_only)

    def count(self) -> 'DataStore':
        """Compute count of groups. Returns lazy DataStore."""
        return self._apply_agg('count')

    def size(self) -> 'LazySeries':
        """
        Compute group sizes (number of rows in each group).

        Unlike count(), size() includes NaN values and returns a Series
        (matching pandas behavior).

        When possible, this uses SQL pushdown for better performance:
        SELECT groupby_col, COUNT(*) FROM ... GROUP BY groupby_col

        Returns:
            LazySeries: Lazy wrapper that returns pd.Series when executed.

        Example:
            >>> ds.groupby('department').size()
            department
            Engineering    10
            Sales           5
            Marketing       3
            dtype: int64
        """
        from .lazy_result import LazySeries

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Use operation descriptor mode (safe from recursion)
        return LazySeries.from_op(
            datastore=self._datastore,
            op_type='groupby_size',
            op_groupby_cols=groupby_cols,
            op_sort=self._sort,
            op_dropna=self._dropna,
        )

    def min(self, numeric_only: bool = False) -> 'DataStore':
        """Compute min of groups. Returns lazy DataStore."""
        return self._apply_agg('min', numeric_only=numeric_only)

    def max(self, numeric_only: bool = False) -> 'DataStore':
        """Compute max of groups. Returns lazy DataStore."""
        return self._apply_agg('max', numeric_only=numeric_only)

    def std(self, numeric_only: bool = False) -> 'DataStore':
        """Compute standard deviation of groups. Returns lazy DataStore."""
        return self._apply_agg('std', numeric_only=numeric_only)

    def var(self, numeric_only: bool = False) -> 'DataStore':
        """Compute variance of groups. Returns lazy DataStore."""
        return self._apply_agg('var', numeric_only=numeric_only)

    def first(self) -> 'DataStore':
        """Return first value in each group. Returns lazy DataStore."""
        return self._apply_agg('first')

    def last(self) -> 'DataStore':
        """Return last value in each group. Returns lazy DataStore."""
        return self._apply_agg('last')

    def cumcount(self, ascending: bool = True) -> 'LazySeries':
        """
        Number each item in each group from 0 to the length of that group - 1.

        Essentially this is equivalent to:
            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))

        Args:
            ascending: If True (default), count from 0 to len(group)-1.
                      If False, count from len(group)-1 to 0.

        Returns:
            LazySeries: Series with index matching the original DataFrame.

        Example:
            >>> df.groupby('category').cumcount()
            0    0
            1    1
            2    0
            3    2
            4    1
            5    2
            dtype: int64
        """
        from .lazy_result import LazySeries

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Use operation descriptor mode (safe from recursion)
        return LazySeries.from_op(
            datastore=self._datastore,
            op_type='groupby_cumcount',
            op_groupby_cols=groupby_cols,
            op_sort=self._sort,
            op_dropna=self._dropna,
            op_kwargs={'ascending': ascending},
        )

    def pipe(self, func, *args, **kwargs):
        """
        Apply a function to this GroupBy object.

        Pipe enables method chaining by passing the GroupBy object as the
        first argument to func.

        Args:
            func: Function to apply to this GroupBy object.
                  Should accept a GroupBy object as its first argument.
            *args: Positional arguments passed to func.
            **kwargs: Keyword arguments passed to func.

        Returns:
            The return value of func.

        Example:
            >>> def summary(grp):
            ...     return grp.agg({'value': ['mean', 'sum']})
            >>> df.groupby('category').pipe(summary)
        """
        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Execute underlying datastore and create pandas GroupBy
        df = self._datastore._get_df()
        pandas_grp = df.groupby(groupby_cols, sort=self._sort, dropna=self._dropna)

        # Apply the function to the pandas GroupBy
        return func(pandas_grp, *args, **kwargs)

    def nth(self, n: Union[int, List[int]], dropna: str = None) -> 'DataStore':
        """
        Return the nth row from each group.

        Takes a scalar or list of scalars representing row positions.
        Supports negative indexing.

        Args:
            n: Integer or list of integers. A single value or list of values
               indicating which row(s) to select. Negative values count from
               the end of each group.
            dropna: Optional, how to handle NA values. Can be 'any', 'all', or None.
                    - 'any': if any NA values are present, skip that row
                    - 'all': if all values are NA, skip that row
                    - None (default): no special NA handling

        Returns:
            DataStore: Lazy DataStore with nth operation

        Example:
            >>> ds.groupby('category').nth(0)   # First row from each group
            >>> ds.groupby('category').nth(1)   # Second row from each group
            >>> ds.groupby('category').nth(-1)  # Last row from each group
            >>> ds.groupby('category').nth([0, 2])  # First and third rows
        """
        from .lazy_ops import LazyNth
        from copy import copy

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Create a shallow copy of the datastore
        new_ds = copy(self._datastore)

        # Add the lazy nth operation
        new_ds._add_lazy_op(LazyNth(n=n, groupby_cols=groupby_cols, dropna=dropna))

        return new_ds

    def head(self, n: int = 5) -> 'DataStore':
        """
        Return first n rows of each group.

        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index preserved rather than
        a new DataFrame.

        Args:
            n: Number of rows to return from each group. Default is 5.

        Returns:
            DataStore: Lazy DataStore with head operation

        Example:
            >>> ds.groupby('category').head(2)  # First 2 rows from each group
            >>> ds.groupby('category').head()   # First 5 rows from each group
        """
        from .lazy_ops import LazyHead
        from copy import copy

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Create a shallow copy of the datastore
        new_ds = copy(self._datastore)

        # Add the lazy head operation
        new_ds._add_lazy_op(LazyHead(n=n, groupby_cols=groupby_cols))

        return new_ds

    def tail(self, n: int = 5) -> 'DataStore':
        """
        Return last n rows of each group.

        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index preserved rather than
        a new DataFrame.

        Args:
            n: Number of rows to return from each group. Default is 5.

        Returns:
            DataStore: Lazy DataStore with tail operation

        Example:
            >>> ds.groupby('category').tail(2)  # Last 2 rows from each group
            >>> ds.groupby('category').tail()   # Last 5 rows from each group
        """
        from .lazy_ops import LazyTail
        from copy import copy

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Create a shallow copy of the datastore
        new_ds = copy(self._datastore)

        # Add the lazy tail operation
        new_ds._add_lazy_op(LazyTail(n=n, groupby_cols=groupby_cols))

        return new_ds

    def _apply_agg(self, func_name: str, **kwargs) -> 'DataStore':
        """
        Apply aggregation function to all columns.

        Returns a lazy DataStore with the aggregation operation.
        """
        return self._create_lazy_agg_datastore(agg_func=func_name, **kwargs)

    def _create_lazy_agg_datastore(
        self, agg_func: str = None, agg_dict: dict = None, named_agg: dict = None, **kwargs
    ) -> 'DataStore':
        """
        Create a new DataStore with lazy groupby aggregation.

        Args:
            agg_func: Aggregation function name ('mean', 'sum', etc.)
            agg_dict: Dict mapping columns to aggregation functions
            named_agg: Dict of named aggregations {alias: (col, func)}
            **kwargs: Additional arguments for the aggregation function

        Returns:
            DataStore: New DataStore with lazy groupby aggregation
        """
        from .lazy_ops import LazyGroupByAgg
        from copy import copy

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Create a shallow copy of the datastore
        new_ds = copy(self._datastore)

        # Add the lazy groupby aggregation operation
        # Pass sort, as_index, and selected_columns parameters from LazyGroupBy to LazyGroupByAgg
        new_ds._add_lazy_op(
            LazyGroupByAgg(
                groupby_cols=groupby_cols,
                agg_func=agg_func,
                agg_dict=agg_dict,
                named_agg=named_agg,
                sort=self._sort,
                as_index=self._as_index,
                dropna=self._dropna,
                selected_columns=self._selected_columns,
                **kwargs,
            )
        )

        return new_ds

    # ========== SQL Building Methods ==========
    # These methods support SQL building patterns like:
    #   ds.select(...).groupby("col").to_sql()
    #
    # IMPORTANT: For SQL building operations, we create a copy of the datastore
    # to avoid modifying the original. This maintains immutability for SQL building
    # while still allowing efficient execution for data operations.

    def _get_sql_copy(self) -> 'DataStore':
        """
        Get a copy of the datastore with groupby fields set for SQL building.

        This preserves immutability for SQL building operations.
        """
        from copy import copy

        ds_copy = copy(self._datastore)
        ds_copy._groupby_fields = self._groupby_fields.copy()
        # Note: SQL copy preserves sort for ORDER BY but dropna doesn't apply to SQL building
        return ds_copy

    def to_sql(self, **kwargs) -> str:
        """
        Generate SQL query with GROUP BY.

        Uses a copy to avoid modifying the original datastore.
        """
        return self._get_sql_copy().to_sql(**kwargs)

    def having(self, condition) -> 'LazyGroupBy':
        """
        Add HAVING clause.

        Creates a copy with groupby fields and applies having.
        """
        ds_copy = self._get_sql_copy()
        result_ds = ds_copy.having(condition)
        return LazyGroupBy(
            result_ds, self._groupby_fields, sort=self._sort, as_index=self._as_index, dropna=self._dropna
        )

    def select(self, *fields) -> 'DataStore':
        """
        Select fields after groupby - uses copy to preserve immutability.
        """
        ds_copy = self._get_sql_copy()
        return ds_copy.select(*fields)

    def sort(self, *fields, ascending: bool = True) -> 'LazyGroupBy':
        """
        Sort results after groupby.
        """
        ds_copy = self._get_sql_copy()
        result_ds = ds_copy.sort(*fields, ascending=ascending)
        return LazyGroupBy(
            result_ds, self._groupby_fields, sort=self._sort, as_index=self._as_index, dropna=self._dropna
        )

    def orderby(self, *fields, ascending: bool = True) -> 'LazyGroupBy':
        """Alias for sort()."""
        return self.sort(*fields, ascending=ascending)

    def limit(self, n: int) -> 'LazyGroupBy':
        """Limit results after groupby."""
        ds_copy = self._get_sql_copy()
        result_ds = ds_copy.limit(n)
        return LazyGroupBy(
            result_ds, self._groupby_fields, sort=self._sort, as_index=self._as_index, dropna=self._dropna
        )

    def offset(self, n: int) -> 'LazyGroupBy':
        """Offset results after groupby."""
        ds_copy = self._get_sql_copy()
        result_ds = ds_copy.offset(n)
        return LazyGroupBy(
            result_ds, self._groupby_fields, sort=self._sort, as_index=self._as_index, dropna=self._dropna
        )

    def transform(self, func, *args, **kwargs) -> 'DataStore':
        """
        Apply a function to each group producing same-shaped output (pandas-style).

        The result has the same index as the original DataFrame, with each value
        transformed according to its group.

        This is commonly used for operations like:
        - Normalizing within groups: lambda x: x / x.sum()
        - Centering: lambda x: x - x.mean()
        - Z-score: lambda x: (x - x.mean()) / x.std()

        Args:
            func: Function to apply to each group. Can be:
                - A callable that takes a Series and returns a Series/scalar
                - A string function name like 'mean', 'sum', etc.
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            DataStore: Lazy DataStore with transform operation added to chain

        Example:
            >>> ds.groupby('category')['value'].transform(lambda x: x / x.sum())
            >>> ds.groupby('category')['value'].transform('mean')
        """
        from .lazy_ops import LazyTransform
        from copy import copy

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Create a shallow copy of the datastore
        new_ds = copy(self._datastore)

        # Add the lazy transform operation (func first, groupby_cols optional)
        new_ds._add_lazy_op(LazyTransform(func, *args, groupby_cols=groupby_cols, **kwargs))

        return new_ds

    def filter(self, func) -> 'DataStore':
        """
        Filter groups based on a function (pandas-style).

        This method supports two modes:
        1. Pandas-style: callable that takes a DataFrame and returns bool
           >>> df.groupby('category').filter(lambda x: x['value'].mean() > 35)

        2. SQL-style: Condition object (delegates to having())
           >>> df.groupby('category').filter(col('value').mean() > 35)

        Args:
            func: Either:
                - A callable that takes a group DataFrame and returns True/False
                - A Condition object for SQL-style filtering

        Returns:
            DataStore: Lazy DataStore with filter operation added to chain

        Example:
            >>> # Keep groups where mean value > 35
            >>> ds.groupby('category').filter(lambda x: x['value'].mean() > 35)
        """
        from .lazy_ops import LazyFilter
        from .conditions import Condition
        from copy import copy

        # SQL-style: delegate to having()
        if isinstance(func, Condition):
            return self.having(func)

        # Pandas-style: callable that returns bool - add to lazy chain
        if callable(func):
            # Get groupby column names
            groupby_cols = []
            for gf in self._groupby_fields:
                if isinstance(gf, Field):
                    groupby_cols.append(gf.name)
                else:
                    groupby_cols.append(str(gf))

            # Create a shallow copy of the datastore
            new_ds = copy(self._datastore)

            # Add the lazy filter operation
            new_ds._add_lazy_op(LazyFilter(func, groupby_cols))

            return new_ds

        raise TypeError(f"filter() argument must be callable or Condition, got {type(func).__name__}")

    def apply(self, func, *args, **kwargs) -> 'DataStore':
        """
        Apply a function to each group and combine results (pandas-style).

        This method applies an arbitrary function to each group. The function
        should take a DataFrame (the group) and return a DataFrame, Series, or scalar.

        Args:
            func: Function to apply to each group
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            DataStore: Lazy DataStore with apply operation added to chain

        Example:
            >>> ds.groupby('category').apply(lambda x: x.nlargest(3, 'value'))
        """
        from .lazy_ops import LazyApply
        from copy import copy

        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Create a shallow copy of the datastore
        new_ds = copy(self._datastore)

        # Add the lazy apply operation (func first, groupby_cols optional)
        new_ds._add_lazy_op(LazyApply(func, *args, groupby_cols=groupby_cols, **kwargs))

        return new_ds

    def execute(self):
        """
        Execute the query with GROUP BY.

        For execution, we use a COPY to preserve immutability of the original.
        """
        ds_copy = self._get_sql_copy()
        return ds_copy.execute()

    def to_df(self) -> pd.DataFrame:
        """
        Execute the grouped data.

        For execution, we set groupby fields on the ORIGINAL datastore
        so that the checkpoint is properly applied.
        """
        self._datastore._groupby_fields = self._groupby_fields.copy()
        return self._datastore.to_df()

    def as_(self, alias: str) -> 'DataStore':
        """
        Set an alias for this grouped query (for use as subquery).

        Returns a DataStore with the alias set.
        """
        ds_copy = self._get_sql_copy()
        return ds_copy.as_(alias)

    def __repr__(self) -> str:
        """String representation."""
        fields = [f.name if isinstance(f, Field) else str(f) for f in self._groupby_fields]
        return f"LazyGroupBy(fields={fields})"

    def __str__(self) -> str:
        return self.__repr__()
