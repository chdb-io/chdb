"""
Expression system for DataStore - inspired by pypika but independent implementation
"""

from typing import Any, Optional, Union, Iterator, List, Type, TYPE_CHECKING
from copy import copy

from .utils import immutable, ignore_copy, format_identifier, format_alias
from .exceptions import ValidationError

if TYPE_CHECKING:
    from .conditions import BinaryCondition, Condition

__all__ = ['Node', 'Expression', 'Field', 'Literal', 'ArithmeticExpression']


class Node:
    """
    Base class for all expression nodes.
    Provides tree traversal capabilities.
    """

    def nodes(self) -> Iterator['Node']:
        """Iterate over all nodes in the expression tree."""
        yield self

    def find(self, node_type: Type['Node']) -> List['Node']:
        """Find all nodes of a specific type."""
        return [node for node in self.nodes() if isinstance(node, node_type)]


class Expression(Node):
    """
    Base class for all expressions.

    Expressions can be:
    - Fields (columns)
    - Literals (constants)
    - Functions
    - Arithmetic operations
    - Conditions
    """

    def __init__(self, alias: Optional[str] = None):
        self.alias = alias

    @immutable
    def as_(self, alias: str) -> 'Expression':
        """Set an alias for this expression."""
        self.alias = alias

    @staticmethod
    def wrap(value: Any) -> 'Expression':
        """
        Intelligently wrap a value as an Expression.

        - Expression -> return as-is
        - None -> Literal(None)
        - list/tuple -> handle specially
        - other -> Literal(value)
        """
        if isinstance(value, Expression):
            return value
        return Literal(value)

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Convert expression to SQL string."""
        raise NotImplementedError(f"{type(self).__name__} must implement to_sql()")

    # ========== Comparison Operators ==========

    def __eq__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('=', self, self.wrap(other))

    def __ne__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('!=', self, self.wrap(other))

    def __gt__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('>', self, self.wrap(other))

    def __ge__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('>=', self, self.wrap(other))

    def __lt__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('<', self, self.wrap(other))

    def __le__(self, other: Any) -> 'BinaryCondition':
        from .conditions import BinaryCondition

        return BinaryCondition('<=', self, self.wrap(other))

    # ========== Advanced Condition Methods ==========

    def isnull(self) -> 'Condition':
        """
        Create IS NULL condition.

        Example:
            >>> Field('name').isnull()
            >>> # Generates: "name" IS NULL
        """
        from .conditions import UnaryCondition

        return UnaryCondition('IS NULL', self)

    def notnull(self) -> 'Condition':
        """
        Create IS NOT NULL condition.

        Example:
            >>> Field('name').notnull()
            >>> # Generates: "name" IS NOT NULL
        """
        from .conditions import UnaryCondition

        return UnaryCondition('IS NOT NULL', self)

    def isin(self, values) -> 'Condition':
        """
        Create IN condition.

        Args:
            values: List of values or subquery

        Example:
            >>> Field('id').isin([1, 2, 3])
            >>> # Generates: "id" IN (1,2,3)
        """
        from .conditions import InCondition

        return InCondition(self, values, negate=False)

    def notin(self, values) -> 'Condition':
        """
        Create NOT IN condition.

        Args:
            values: List of values or subquery

        Example:
            >>> Field('id').notin([1, 2, 3])
            >>> # Generates: "id" NOT IN (1,2,3)
        """
        from .conditions import InCondition

        return InCondition(self, values, negate=True)

    def between(self, lower, upper) -> 'Condition':
        """
        Create BETWEEN condition.

        Args:
            lower: Lower bound
            upper: Upper bound

        Example:
            >>> Field('age').between(18, 65)
            >>> # Generates: "age" BETWEEN 18 AND 65
        """
        from .conditions import BetweenCondition

        return BetweenCondition(self, self.wrap(lower), self.wrap(upper))

    def like(self, pattern: str) -> 'Condition':
        """
        Create LIKE condition.

        Args:
            pattern: SQL LIKE pattern (% for wildcard)

        Example:
            >>> Field('name').like('John%')
            >>> # Generates: "name" LIKE 'John%'
        """
        from .conditions import LikeCondition

        return LikeCondition(self, pattern, negate=False, case_sensitive=True)

    def notlike(self, pattern: str) -> 'Condition':
        """
        Create NOT LIKE condition.

        Args:
            pattern: SQL LIKE pattern

        Example:
            >>> Field('name').notlike('John%')
            >>> # Generates: "name" NOT LIKE 'John%'
        """
        from .conditions import LikeCondition

        return LikeCondition(self, pattern, negate=True, case_sensitive=True)

    def ilike(self, pattern: str) -> 'Condition':
        """
        Create ILIKE condition (case-insensitive).

        Args:
            pattern: SQL LIKE pattern

        Example:
            >>> Field('name').ilike('john%')
            >>> # Generates: "name" ILIKE 'john%'
        """
        from .conditions import LikeCondition

        return LikeCondition(self, pattern, negate=False, case_sensitive=False)

    # ========== Arithmetic Operators ==========

    def __add__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('+', self, self.wrap(other))

    def __sub__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('-', self, self.wrap(other))

    def __mul__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('*', self, self.wrap(other))

    def __truediv__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('/', self, self.wrap(other))

    def __mod__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('%', self, self.wrap(other))

    # ========== Reverse Arithmetic Operators ==========

    def __radd__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('+', self.wrap(other), self)

    def __rsub__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('-', self.wrap(other), self)

    def __rmul__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('*', self.wrap(other), self)

    def __rtruediv__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('/', self.wrap(other), self)

    # ========== Unary Operators ==========

    def __neg__(self) -> 'ArithmeticExpression':
        return ArithmeticExpression('-', Literal(0), self)

    # ========== String/Utility Methods ==========

    def __str__(self) -> str:
        return self.to_sql()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_sql()!r})"


class Field(Expression):
    """
    Represents a field (column) in a data source.

    Example:
        >>> Field('name')
        >>> Field('age', table='customers')
    """

    def __init__(self, name: str, table: Optional[str] = None, alias: Optional[str] = None):
        super().__init__(alias)
        self.name = name
        self.table = table

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for field."""
        # Format: [table.]field [AS alias]
        parts = []

        if self.table:
            parts.append(format_identifier(self.table, quote_char))

        parts.append(format_identifier(self.name, quote_char))
        field_sql = '.'.join(parts)

        # Add alias if present and requested
        if kwargs.get('with_alias', False) and self.alias:
            return format_alias(field_sql, self.alias, quote_char)

        return field_sql

    def __copy__(self):
        return Field(self.name, self.table, self.alias)


class Literal(Expression):
    """
    Represents a literal value (constant).

    Example:
        >>> Literal(42)
        >>> Literal("hello")
        >>> Literal(None)
    """

    def __init__(self, value: Any, alias: Optional[str] = None):
        super().__init__(alias)
        self.value = value

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for literal."""
        # Convert Python value to SQL literal
        if self.value is None:
            sql = 'NULL'
        elif isinstance(self.value, bool):
            sql = 'TRUE' if self.value else 'FALSE'
        elif isinstance(self.value, (int, float)):
            sql = str(self.value)
        elif isinstance(self.value, str):
            # Escape single quotes
            escaped = self.value.replace("'", "''")
            sql = f"'{escaped}'"
        else:
            # Fallback: convert to string
            sql = f"'{str(self.value)}'"

        # Add alias if present and requested
        if kwargs.get('with_alias', False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return Literal(self.value, self.alias)


class ArithmeticExpression(Expression):
    """
    Represents an arithmetic operation (e.g., a + b, x * 2).

    Example:
        >>> ArithmeticExpression('+', Field('a'), Literal(1))
        >>> Field('price') * Literal(1.1)  # 10% increase
    """

    OPERATORS = {'+', '-', '*', '/', '%', '**'}

    def __init__(self, operator: str, left: Expression, right: Expression, alias: Optional[str] = None):
        super().__init__(alias)

        if operator not in self.OPERATORS:
            raise ValidationError(f"Invalid operator: {operator}")

        self.operator = operator
        self.left = left
        self.right = right

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.left.nodes()
        yield from self.right.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for arithmetic expression."""
        left_sql = self.left.to_sql(quote_char=quote_char, **kwargs)
        right_sql = self.right.to_sql(quote_char=quote_char, **kwargs)

        # Use ** for power in Python, but POW() in SQL
        if self.operator == '**':
            sql = f"POW({left_sql},{right_sql})"
        else:
            sql = f"({left_sql}{self.operator}{right_sql})"

        # Add alias if present and requested
        if kwargs.get('with_alias', False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return ArithmeticExpression(self.operator, copy(self.left), copy(self.right), self.alias)
