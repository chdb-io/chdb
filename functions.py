"""
Function system for DataStore
"""

from typing import Any, Optional, Sequence
from copy import copy

from .expressions import Expression
from .utils import format_alias
from .exceptions import ValidationError

__all__ = [
    'Function',
    'AggregateFunction',
    'WindowFunction',
    'CustomFunction',
    'CastFunction',
    'F',  # Function namespace
    'Sum',
    'Count',
    'Avg',
    'Min',
    'Max',
    'Upper',
    'Lower',
    'Concat',
]


class Function(Expression):
    """
    Base class for SQL functions.

    Example:
        >>> Function('UPPER', Field('name'))
        >>> Function('CONCAT', Literal('Hello'), Literal(' World'))
    """

    is_aggregate = False

    def __init__(self, name: str, *args: Expression, alias: Optional[str] = None, pandas_name: Optional[str] = None, pandas_kwargs: Optional[dict] = None):
        super().__init__(alias)
        self.name = name  # Keep original case for database compatibility
        self.args = [self.wrap(arg) for arg in args]
        # pandas_name is used for execution engine decisions when the SQL function
        # name differs from the user-facing/pandas function name
        self.pandas_name = pandas_name
        # pandas_kwargs stores extra keyword arguments for pandas implementation
        # (e.g., case, flags, na, regex for str.contains)
        self.pandas_kwargs = pandas_kwargs or {}

    def get_special_params_sql(self, **kwargs) -> str:
        """
        Override this to add special parameters after function arguments.
        E.g., CAST(x AS INT) - the "AS INT" part.
        """
        return ""

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for function call."""
        # Generate arguments SQL
        args_sql = ','.join(arg.to_sql(quote_char=quote_char, **kwargs) for arg in self.args)

        # Get special parameters
        special_params = self.get_special_params_sql(quote_char=quote_char, **kwargs)

        # Build function call
        if special_params:
            sql = f"{self.name}({args_sql} {special_params})"
        else:
            sql = f"{self.name}({args_sql})"

        # Add alias if present and requested
        if kwargs.get('with_alias', False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return Function(self.name, *[copy(arg) for arg in self.args], alias=self.alias, pandas_name=self.pandas_name, pandas_kwargs=self.pandas_kwargs.copy() if self.pandas_kwargs else None)


class AggregateFunction(Function):
    """
    Base class for aggregate functions (SUM, COUNT, AVG, etc.).
    These are used in GROUP BY queries.
    """

    is_aggregate = True


class WindowFunction(Function):
    """
    Window function with OVER clause support.

    Window functions operate over a set of rows (window) and return a value
    for each row. They require an OVER clause to define the window.

    Example:
        >>> # Row number within partition
        >>> WindowFunction('row_number').over(partition_by='category', order_by='value DESC')
        >>> # SQL: row_number() OVER (PARTITION BY "category" ORDER BY value DESC)

        >>> # Lead/Lag functions
        >>> WindowFunction('leadInFrame', Field('value'), Literal(1)).over(
        ...     partition_by='user_id',
        ...     order_by='timestamp'
        ... )
        >>> # SQL: leadInFrame("value", 1) OVER (PARTITION BY "user_id" ORDER BY timestamp)

    ClickHouse Window Functions:
        - row_number, rank, dense_rank, ntile
        - lead, lag (leadInFrame, lagInFrame in ClickHouse)
        - first_value, last_value, nth_value
        - sum, avg, count, min, max (as window functions)
    """

    is_window = True
    is_aggregate = False  # Window functions are not aggregates in the traditional sense

    def __init__(self, name: str, *args: Expression, alias: Optional[str] = None):
        super().__init__(name, *args, alias=alias)
        self._partition_by: list = []
        self._order_by: list = []
        self._frame: Optional[str] = None

    def over(
        self,
        partition_by=None,
        order_by=None,
        frame: Optional[str] = None,
    ) -> 'WindowFunction':
        """
        Add OVER clause to the window function.

        Args:
            partition_by: Column(s) to partition by. Can be:
                - str: Single column name
                - Expression: Single expression
                - list: Multiple columns/expressions
            order_by: Column(s) to order by. Can be:
                - str: Single column with optional DESC/ASC (e.g., 'value DESC')
                - Expression: Single expression
                - list: Multiple columns/expressions
            frame: Window frame specification, e.g.:
                - 'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'
                - 'ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING'
                - 'RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING'

        Returns:
            New WindowFunction with OVER clause configured

        Example:
            >>> F.row_number().over(
            ...     partition_by='department',
            ...     order_by='salary DESC'
            ... )
            >>> # SQL: row_number() OVER (PARTITION BY "department" ORDER BY salary DESC)

            >>> F.sum('amount').over(
            ...     partition_by=['region', 'year'],
            ...     order_by='month',
            ...     frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'
            ... )
        """
        # Create a copy to maintain immutability
        result = copy(self)
        result._partition_by = []
        result._order_by = []
        result._frame = None

        # Handle partition_by
        if partition_by is not None:
            if isinstance(partition_by, (list, tuple)):
                result._partition_by = list(partition_by)
            else:
                result._partition_by = [partition_by]

        # Handle order_by
        if order_by is not None:
            if isinstance(order_by, (list, tuple)):
                result._order_by = list(order_by)
            else:
                result._order_by = [order_by]

        # Handle frame
        if frame is not None:
            result._frame = frame

        return result

    def _format_over_item(self, item, quote_char: str) -> str:
        """Format a single partition_by or order_by item."""
        if hasattr(item, 'to_sql'):
            return item.to_sql(quote_char=quote_char)
        # String: could be 'column' or 'column DESC'
        item_str = str(item)
        parts = item_str.split()
        if len(parts) == 1:
            # Just column name, quote it
            return f'{quote_char}{item_str}{quote_char}'
        elif len(parts) == 2 and parts[1].upper() in ('ASC', 'DESC'):
            # Column with direction
            return f'{quote_char}{parts[0]}{quote_char} {parts[1].upper()}'
        else:
            # Complex expression, use as-is
            return item_str

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for window function with OVER clause."""
        # Build base function call (without alias for now)
        kwargs_no_alias = {**kwargs, 'with_alias': False}
        base_sql = super().to_sql(quote_char=quote_char, **kwargs_no_alias)

        # If no OVER clause parts, return just the function
        if not self._partition_by and not self._order_by and not self._frame:
            # Add alias if present and requested
            if kwargs.get('with_alias', False) and self.alias:
                return format_alias(base_sql, self.alias, quote_char)
            return base_sql

        # Build OVER clause
        over_parts = []

        if self._partition_by:
            partition_items = [self._format_over_item(item, quote_char) for item in self._partition_by]
            over_parts.append(f"PARTITION BY {', '.join(partition_items)}")

        if self._order_by:
            order_items = [self._format_over_item(item, quote_char) for item in self._order_by]
            over_parts.append(f"ORDER BY {', '.join(order_items)}")

        if self._frame:
            over_parts.append(self._frame)

        over_clause = ' '.join(over_parts)
        result = f"{base_sql} OVER ({over_clause})"

        # Add alias if present and requested
        if kwargs.get('with_alias', False) and self.alias:
            return format_alias(result, self.alias, quote_char)

        return result

    def __copy__(self):
        result = WindowFunction(self.name, *[copy(arg) for arg in self.args], alias=self.alias)
        result._partition_by = list(self._partition_by)
        result._order_by = list(self._order_by)
        result._frame = self._frame
        return result

    def __repr__(self) -> str:
        parts = [f"WindowFunction('{self.name}'"]
        if self.args:
            parts.append(f", args={self.args}")
        if self._partition_by:
            parts.append(f", partition_by={self._partition_by}")
        if self._order_by:
            parts.append(f", order_by={self._order_by}")
        if self._frame:
            parts.append(f", frame='{self._frame}'")
        if self.alias:
            parts.append(f", alias='{self.alias}'")
        parts.append(")")
        return ''.join(parts)


class CustomFunction:
    """
    Factory for creating custom functions easily.

    Example:
        >>> DateDiff = CustomFunction('DATE_DIFF', ['interval', 'start', 'end'])
        >>> DateDiff('day', Field('created_at'), Field('updated_at'))
    """

    def __init__(self, name: str, params: Optional[Sequence[str]] = None):
        self.name = name
        self.params = params

    def __call__(self, *args, **kwargs) -> Function:
        """Make the factory callable."""
        # Validate argument count if params specified
        if self.params and len(args) != len(self.params):
            raise ValidationError(
                f"Function {self.name} requires {len(self.params)} arguments "
                f"({', '.join(self.params)}), but {len(args)} were provided"
            )

        return Function(self.name, *args, alias=kwargs.get('alias'))


# ========== Common Aggregate Functions ==========


class Sum(AggregateFunction):
    """SUM aggregate function."""

    def __init__(self, expr: Expression, alias: Optional[str] = None):
        super().__init__('SUM', expr, alias=alias)


class Count(AggregateFunction):
    """COUNT aggregate function."""

    def __init__(self, expr: Any = '*', alias: Optional[str] = None):
        # Check if it's the string '*' (COUNT(*))
        if isinstance(expr, str) and expr == '*':
            # COUNT(*) - use a special marker
            from .expressions import Literal

            expr = Literal('*')
        super().__init__('COUNT', expr, alias=alias)

    def __copy__(self):
        """Preserve Count type when copying (e.g., for .as_() method)."""
        # Reconstruct with the original argument
        from .expressions import Literal

        if len(self.args) == 1 and isinstance(self.args[0], Literal) and self.args[0].value == '*':
            return Count('*', alias=self.alias)
        return Count(copy(self.args[0]), alias=self.alias)

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Special handling for COUNT(*)."""
        # For COUNT(*), don't quote the *
        if len(self.args) == 1:
            from .expressions import Literal

            if isinstance(self.args[0], Literal) and self.args[0].value == '*':
                sql = "COUNT(*)"
                if kwargs.get('with_alias', False) and self.alias:
                    return format_alias(sql, self.alias, quote_char)
                return sql

        return super().to_sql(quote_char=quote_char, **kwargs)


class Avg(AggregateFunction):
    """AVG aggregate function."""

    def __init__(self, expr: Expression, alias: Optional[str] = None):
        super().__init__('AVG', expr, alias=alias)


class Min(AggregateFunction):
    """MIN aggregate function."""

    def __init__(self, expr: Expression, alias: Optional[str] = None):
        super().__init__('MIN', expr, alias=alias)


class Max(AggregateFunction):
    """MAX aggregate function."""

    def __init__(self, expr: Expression, alias: Optional[str] = None):
        super().__init__('MAX', expr, alias=alias)


# ========== Common String Functions ==========


class Upper(Function):
    """UPPER string function."""

    def __init__(self, expr: Expression, alias: Optional[str] = None):
        super().__init__('UPPER', expr, alias=alias)


class Lower(Function):
    """LOWER string function."""

    def __init__(self, expr: Expression, alias: Optional[str] = None):
        super().__init__('LOWER', expr, alias=alias)


class Concat(Function):
    """CONCAT string function."""

    def __init__(self, *exprs: Expression, alias: Optional[str] = None):
        super().__init__('CONCAT', *exprs, alias=alias)


# ========== Cast Function ==========


class CastFunction(Function):
    """
    CAST function for type conversion.

    Example:
        >>> CastFunction(Field('value'), 'Float64')
        >>> # SQL: CAST("value" AS Float64)
    """

    def __init__(self, expr: Expression, target_type: str, alias: Optional[str] = None):
        super().__init__('CAST', expr, alias=alias)
        self.target_type = target_type

    def get_special_params_sql(self, **kwargs) -> str:
        """Add AS type after the expression."""
        return f"AS {self.target_type}"

    def __copy__(self):
        return CastFunction(copy(self.args[0]), self.target_type, self.alias)


# ========== Function Namespace F ==========


class F:
    """
    ClickHouse function namespace for explicit function calls.

    Provides a clean namespace for calling ClickHouse functions explicitly,
    similar to pyspark.sql.functions or SQLAlchemy func.

    All function methods are dynamically injected from the FunctionRegistry
    at module load time. This eliminates code duplication and ensures
    consistency with Expression methods.

    Example:
        >>> from datastore import F
        >>>
        >>> # In select statements
        >>> ds.select(
        ...     F.upper(ds.name).as_('upper_name'),
        ...     F.year(ds.date).as_('year'),
        ...     F.round(ds.price, 2).as_('rounded_price')
        ... )
        >>>
        >>> # Conditional functions
        >>> F.if_(ds.age > 18, 'adult', 'minor')
        >>> F.coalesce(ds.value, 0)
        >>>
        >>> # Aggregates
        >>> F.sum(ds.amount)
        >>> F.count()
        >>>
        >>> # Window functions with OVER clause
        >>> F.row_number().over(partition_by='category', order_by='value DESC')
        >>> F.lead(ds.value, 1).over(partition_by='user_id', order_by='timestamp')
    """

    # ========== Special Methods ==========
    # Methods that need special handling are defined here.
    # Standard functions are injected from the registry below.

    @staticmethod
    def cast(expr, target_type: str, alias: str = None) -> CastFunction:
        """Cast to type. Maps to CAST(x AS type)."""
        from .expressions import Expression

        return CastFunction(Expression.wrap(expr), target_type, alias=alias)

    @staticmethod
    def json_extract(json, path: str, type_name: str = None, alias: str = None) -> Function:
        """
        Extract value from JSON.

        Args:
            json: JSON string or column
            path: JSONPath or key
            type_name: Optional return type (e.g., 'String', 'Int64')

        Maps to JSONExtract(json, path, type) or JSONExtractRaw(json, path)
        """
        from .expressions import Expression, Literal

        if type_name:
            return Function('JSONExtract', Expression.wrap(json), Literal(path), Literal(type_name), alias=alias)
        return Function('JSONExtractRaw', Expression.wrap(json), Literal(path), alias=alias)


# =============================================================================
# INJECT F CLASS METHODS FROM REGISTRY
# =============================================================================


def _inject_f_class_methods():
    """Inject static methods from registry into F class."""
    from .function_registry import FunctionRegistry
    from . import function_definitions  # noqa: F401 - ensures registration

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.all_specs():
        # Skip if method already exists (don't override special implementations)
        if hasattr(F, spec.name):
            continue

        # Create static method that wraps the sql_builder
        def make_static_method(func_spec):
            def static_method(*args, alias=None, **kwargs):
                from .expressions import Expression

                # Wrap the first argument if needed
                if args and not isinstance(args[0], Expression):
                    args = (Expression.wrap(args[0]),) + args[1:]
                return func_spec.sql_builder(*args, alias=alias, **kwargs)

            static_method.__name__ = func_spec.name
            static_method.__doc__ = func_spec.doc
            return staticmethod(static_method)

        # Set the static method on F class
        setattr(F, spec.name, make_static_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(F, alias_name):
                setattr(F, alias_name, getattr(F, spec.name))


# Perform injection when module is loaded
_inject_f_class_methods()
