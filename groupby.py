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

    def __init__(self, datastore: 'DataStore', groupby_fields: List[Expression]):
        """
        Initialize LazyGroupBy.

        Args:
            datastore: Reference to the ORIGINAL DataStore (not a copy)
            groupby_fields: List of Field expressions to group by
        """
        self._datastore = datastore
        self._groupby_fields = groupby_fields.copy()  # Copy the list, not the DataStore

    @property
    def datastore(self) -> 'DataStore':
        """Get the original DataStore reference."""
        return self._datastore

    @property
    def groupby_fields(self) -> List[Expression]:
        """Get the groupby fields."""
        return self._groupby_fields

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
            field = Field(key)
            return ColumnExpr(field, self._datastore, groupby_fields=self._groupby_fields.copy())

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

        Supports two modes:
        1. Pandas-style aggregation with dict:
           >>> df.groupby('dept').agg({'salary': 'mean', 'age': 'max'})

        2. SQL-style aggregation with keyword arguments:
           >>> from datastore import col
           >>> df.groupby('dept').agg(avg_salary=col('salary').mean())

        Args:
            func: Aggregation function(s) - dict or string
            **kwargs: Named aggregate expressions

        Returns:
            pd.DataFrame for pandas-style, DataStore for SQL-style
        """
        from .column_expr import ColumnExpr, LazyAggregate
        from .functions import AggregateFunction

        # Check if we have SQL-style keyword arguments with expressions
        has_sql_agg = any(
            isinstance(v, (Expression, ColumnExpr, AggregateFunction, LazyAggregate)) for v in kwargs.values()
        )

        if has_sql_agg:
            # SQL-style aggregation: agg(alias=col("x").sum(), ...)
            # Set groupby fields on original datastore and delegate
            self._datastore._groupby_fields = self._groupby_fields.copy()
            return self._datastore.agg(func, **kwargs)
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

        # Capture datastore and cols for closure
        ds = self._datastore
        cols = groupby_cols

        def executor():
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

    def _apply_agg(self, func_name: str, **kwargs) -> 'DataStore':
        """
        Apply aggregation function to all columns.

        Returns a lazy DataStore with the aggregation operation.
        """
        return self._create_lazy_agg_datastore(agg_func=func_name, **kwargs)

    def _create_lazy_agg_datastore(self, agg_func: str = None, agg_dict: dict = None, **kwargs) -> 'DataStore':
        """
        Create a new DataStore with lazy groupby aggregation.

        Args:
            agg_func: Aggregation function name ('mean', 'sum', etc.)
            agg_dict: Dict mapping columns to aggregation functions
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
        new_ds._add_lazy_op(LazyGroupByAgg(groupby_cols=groupby_cols, agg_func=agg_func, agg_dict=agg_dict, **kwargs))

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

    def filter(self, condition) -> 'LazyGroupBy':
        """Filter after groupby (applies HAVING in SQL context)."""
        return self.having(condition)

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
