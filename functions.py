"""
Function system for DataStore
"""

from typing import Any, Optional, List, Sequence
from copy import copy

from .expressions import Expression
from .utils import format_alias
from .exceptions import ValidationError

__all__ = [
    'Function',
    'AggregateFunction',
    'CustomFunction',
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

    def __init__(self, name: str, *args: Expression, alias: Optional[str] = None):
        super().__init__(alias)
        self.name = name  # Keep original case for database compatibility
        self.args = [self.wrap(arg) for arg in args]

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
        return Function(self.name, *[copy(arg) for arg in self.args], alias=self.alias)


class AggregateFunction(Function):
    """
    Base class for aggregate functions (SUM, COUNT, AVG, etc.).
    These are used in GROUP BY queries.
    """

    is_aggregate = True


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
