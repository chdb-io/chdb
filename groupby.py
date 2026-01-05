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
        self, datastore: 'DataStore', groupby_fields: List[Expression], sort: bool = True, as_index: bool = True
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
        """
        self._datastore = datastore
        self._groupby_fields = groupby_fields.copy()  # Copy the list, not the DataStore
        self._sort = sort
        self._as_index = as_index

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
        # Get groupby column names
        groupby_cols = []
        for gf in self._groupby_fields:
            if isinstance(gf, Field):
                groupby_cols.append(gf.name)
            else:
                groupby_cols.append(str(gf))

        # Execute the underlying datastore and get unique count
        df = self._datastore._get_df()
        return df.groupby(groupby_cols, sort=self._sort).ngroups

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
            # Also pass as_index and sort parameters for proper aggregation behavior
            field = Field(key)
            return ColumnExpr(
                field,
                self._datastore,
                groupby_fields=self._groupby_fields.copy(),
                groupby_as_index=self._as_index,
                groupby_sort=self._sort,
            )

        elif isinstance(key, list):
            # Multiple columns - return new GroupBy with column selection
            return self

        else:
            raise TypeError(f"Expected str or list, got {type(key).__name__}")

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
        from .functions import AggregateFunction

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
        else:
            # Pandas-style aggregation: agg({'col': 'func'})
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

        # Capture datastore, cols, and sort for closure
        ds = self._datastore
        cols = groupby_cols
        sort = self._sort

        def executor():
            # Check if we can use SQL pushdown
            if ds._table_function or ds.table_name:
                # Use SQL GROUP BY for performance
                # Build: SELECT col, COUNT(*) FROM ... GROUP BY col
                from .expressions import Star
                from .functions import AggregateFunction

                # Ensure connection exists
                if ds._executor is None:
                    ds.connect()

                # Build SELECT fields: groupby cols + COUNT(*)
                select_parts = [f'"{col}"' for col in cols]
                select_parts.append('COUNT(*) AS "size"')
                select_sql = ', '.join(select_parts)

                # Build GROUP BY
                groupby_sql = ', '.join(f'"{col}"' for col in cols)

                # Get base table SQL
                if ds._table_function:
                    table_sql = ds._table_function.to_sql()
                else:
                    table_sql = f'"{ds.table_name}"'

                # Build WHERE clause from lazy ops if any
                where_sql = ''
                from .lazy_ops import LazyRelationalOp

                where_conditions = []
                for op in ds._lazy_ops:
                    if isinstance(op, LazyRelationalOp) and op.op_type == 'WHERE' and op.condition:
                        where_conditions.append(op.condition.to_sql(quote_char='"'))
                if where_conditions:
                    where_sql = ' WHERE ' + ' AND '.join(where_conditions)

                # Add ORDER BY for sorted groupby (pandas default: sort=True)
                orderby_sql = ''
                if sort:
                    orderby_sql = ' ORDER BY ' + ', '.join(f'"{col}"' for col in cols)

                sql = f'SELECT {select_sql} FROM {table_sql}{where_sql} GROUP BY {groupby_sql}{orderby_sql}'
                result_df = ds._executor.execute(sql).to_df()

                # Convert to Series with groupby col as index
                if len(cols) == 1:
                    return result_df.set_index(cols[0])['size']
                else:
                    return result_df.set_index(cols)['size']
            else:
                # Fall back to pandas
                df = ds._execute()
                return df.groupby(cols).size()

        return LazySeries(executor=executor, datastore=ds)

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

        # Capture datastore, cols, and ascending for closure
        ds = self._datastore
        cols = groupby_cols
        asc = ascending

        def executor():
            # Execute underlying datastore and use pandas groupby.cumcount
            df = ds._get_df()
            return df.groupby(cols).cumcount(ascending=asc)

        return LazySeries(executor=executor, datastore=ds)

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
        pandas_grp = df.groupby(groupby_cols, sort=self._sort)

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
        # Pass sort and as_index parameters from LazyGroupBy to LazyGroupByAgg
        new_ds._add_lazy_op(
            LazyGroupByAgg(
                groupby_cols=groupby_cols,
                agg_func=agg_func,
                agg_dict=agg_dict,
                named_agg=named_agg,
                sort=self._sort,
                as_index=self._as_index,
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
        return LazyGroupBy(result_ds, self._groupby_fields)

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
        return LazyGroupBy(result_ds, self._groupby_fields)

    def orderby(self, *fields, ascending: bool = True) -> 'LazyGroupBy':
        """Alias for sort()."""
        return self.sort(*fields, ascending=ascending)

    def limit(self, n: int) -> 'LazyGroupBy':
        """Limit results after groupby."""
        ds_copy = self._get_sql_copy()
        result_ds = ds_copy.limit(n)
        return LazyGroupBy(result_ds, self._groupby_fields)

    def offset(self, n: int) -> 'LazyGroupBy':
        """Offset results after groupby."""
        ds_copy = self._get_sql_copy()
        result_ds = ds_copy.offset(n)
        return LazyGroupBy(result_ds, self._groupby_fields)

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
