"""
Base Accessor class for Expression accessors.

Provides common functionality for all accessor types (string, datetime, array, json).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..expressions import Expression
    from ..functions import Function


class BaseAccessor:
    """
    Base class for all Expression accessors.

    Accessor classes provide a namespace for related functions, similar to
    Pandas' .str, .dt accessors. Each accessor wraps an Expression and
    provides methods that return new Function expressions.

    Example:
        >>> # StringAccessor provides string functions
        >>> ds['name'].str.upper()  # Returns Function('upper', Field('name'))
        >>>
        >>> # DateTimeAccessor provides date/time functions
        >>> ds['date'].dt.year  # Returns Function('toYear', Field('date'))
    """

    def __init__(self, expr: 'Expression'):
        """
        Initialize accessor with the expression to operate on.

        Args:
            expr: The Expression this accessor wraps
        """
        self._expr = expr

    @property
    def expr(self) -> 'Expression':
        """Get the wrapped expression."""
        return self._expr

    def _create_function(self, name: str, *args, **kwargs) -> 'Function':
        """
        Create a Function with this accessor's expression as first argument.

        Args:
            name: ClickHouse function name
            *args: Additional arguments
            **kwargs: Keyword arguments (e.g., alias)

        Returns:
            Function expression
        """
        from ..functions import Function
        from ..expressions import Literal

        # Wrap non-Expression arguments as Literals
        wrapped_args = []
        for arg in args:
            if hasattr(arg, 'to_sql'):
                wrapped_args.append(arg)
            else:
                wrapped_args.append(Literal(arg))

        return Function(name, self._expr, *wrapped_args, alias=kwargs.get('alias'))

    def _create_function_no_expr(self, name: str, *args, **kwargs) -> 'Function':
        """
        Create a Function without prepending this accessor's expression.
        Useful for functions where the expression is not the first argument.

        Args:
            name: ClickHouse function name
            *args: All arguments including expression
            **kwargs: Keyword arguments (e.g., alias)

        Returns:
            Function expression
        """
        from ..functions import Function
        from ..expressions import Literal

        wrapped_args = []
        for arg in args:
            if hasattr(arg, 'to_sql'):
                wrapped_args.append(arg)
            else:
                wrapped_args.append(Literal(arg))

        return Function(name, *wrapped_args, alias=kwargs.get('alias'))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._expr!r})"
