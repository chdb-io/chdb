"""
Expression system for DataStore - inspired by pypika but independent implementation
"""

from typing import Any, Optional, Iterator, List, Type, TYPE_CHECKING
from copy import copy

from .utils import immutable, format_identifier, format_alias
from .exceptions import ValidationError

if TYPE_CHECKING:
    from .conditions import BinaryCondition, Condition
    from .accessors.string import StringAccessor
    from .accessors.datetime import DateTimeAccessor

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

    def __floordiv__(self, other: Any) -> 'ArithmeticExpression':
        # Floor division: a // b  == intDiv(a, b) in ClickHouse
        return ArithmeticExpression('//', self, self.wrap(other))

    def __mod__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('%', self, self.wrap(other))

    def __pow__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('**', self, self.wrap(other))

    # ========== Reverse Arithmetic Operators ==========

    def __radd__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('+', self.wrap(other), self)

    def __rsub__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('-', self.wrap(other), self)

    def __rmul__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('*', self.wrap(other), self)

    def __rtruediv__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('/', self.wrap(other), self)

    def __rfloordiv__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('//', self.wrap(other), self)

    def __rpow__(self, other: Any) -> 'ArithmeticExpression':
        return ArithmeticExpression('**', self.wrap(other), self)

    # ========== Unary Operators ==========

    def __neg__(self) -> 'ArithmeticExpression':
        return ArithmeticExpression('-', Literal(0), self)

    # ========== String/Utility Methods ==========

    def __str__(self) -> str:
        return self.to_sql()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_sql()!r})"

    # ========== Accessor Properties ==========

    @property
    def str(self) -> 'StringAccessor':
        """
        Accessor for string functions.

        Provides a Pandas-like interface to ClickHouse string functions.

        Example:
            >>> ds['name'].str.upper()          # upper(name)
            >>> ds['name'].str.length()         # length(name)
            >>> ds['name'].str.substring(1, 5)  # substring(name, 1, 5)
            >>> ds['text'].str.replace('a', 'b') # replace(text, 'a', 'b')

        Returns:
            StringAccessor with string function methods
        """
        from .accessors.string import StringAccessor

        return StringAccessor(self)

    @property
    def dt(self) -> 'DateTimeAccessor':
        """
        Accessor for date/time functions.

        Provides a Pandas-like interface to ClickHouse date/time functions.

        Example:
            >>> ds['date'].dt.year              # toYear(date)
            >>> ds['date'].dt.month             # toMonth(date)
            >>> ds['ts'].dt.hour                # toHour(ts)
            >>> ds['date'].dt.day_of_week       # toDayOfWeek(date)
            >>> ds['date'].dt.add_days(7)       # addDays(date, 7)

        Returns:
            DateTimeAccessor with date/time function methods
        """
        from .accessors.datetime import DateTimeAccessor

        return DateTimeAccessor(self)

    # ========== Math Functions (Direct Methods) ==========

    def abs(self, alias: str = None) -> 'Expression':
        """
        Absolute value.

        Maps to ClickHouse: abs(x)

        Returns:
            Function expression for abs(self)

        Example:
            >>> ds['value'].abs()
            >>> # SQL: abs("value")
        """
        from .functions import Function

        return Function('abs', self, alias=alias)

    def round(self, precision: int = 0, alias: str = None) -> 'Expression':
        """
        Round to N decimal places.

        Maps to ClickHouse: round(x, N)

        Args:
            precision: Number of decimal places (default: 0)

        Returns:
            Function expression for round(self, precision)

        Example:
            >>> ds['price'].round(2)
            >>> # SQL: round("price", 2)
        """
        from .functions import Function

        return Function('round', self, Literal(precision), alias=alias)

    def floor(self, alias: str = None) -> 'Expression':
        """
        Round down to nearest integer.

        Maps to ClickHouse: floor(x)

        Returns:
            Function expression for floor(self)
        """
        from .functions import Function

        return Function('floor', self, alias=alias)

    def ceil(self, alias: str = None) -> 'Expression':
        """
        Round up to nearest integer.

        Maps to ClickHouse: ceiling(x)

        Returns:
            Function expression for ceiling(self)
        """
        from .functions import Function

        return Function('ceiling', self, alias=alias)

    def ceiling(self, alias: str = None) -> 'Expression':
        """Alias for ceil(). Round up to nearest integer."""
        return self.ceil(alias=alias)

    def sqrt(self, alias: str = None) -> 'Expression':
        """
        Square root.

        Maps to ClickHouse: sqrt(x)

        Returns:
            Function expression for sqrt(self)
        """
        from .functions import Function

        return Function('sqrt', self, alias=alias)

    def exp(self, alias: str = None) -> 'Expression':
        """
        Exponential (e^x).

        Maps to ClickHouse: exp(x)

        Returns:
            Function expression for exp(self)
        """
        from .functions import Function

        return Function('exp', self, alias=alias)

    def log(self, base: float = None, alias: str = None) -> 'Expression':
        """
        Logarithm. Natural log if no base specified.

        Maps to ClickHouse: log(x) or log(base, x)

        Args:
            base: Log base (optional, natural log if not specified)

        Returns:
            Function expression for log(self) or log(base, self)
        """
        from .functions import Function

        if base is not None:
            return Function('log', Literal(base), self, alias=alias)
        return Function('log', self, alias=alias)

    def log10(self, alias: str = None) -> 'Expression':
        """
        Base-10 logarithm.

        Maps to ClickHouse: log10(x)

        Returns:
            Function expression for log10(self)
        """
        from .functions import Function

        return Function('log10', self, alias=alias)

    def log2(self, alias: str = None) -> 'Expression':
        """
        Base-2 logarithm.

        Maps to ClickHouse: log2(x)

        Returns:
            Function expression for log2(self)
        """
        from .functions import Function

        return Function('log2', self, alias=alias)

    def sin(self, alias: str = None) -> 'Expression':
        """Sine. Maps to ClickHouse: sin(x)"""
        from .functions import Function

        return Function('sin', self, alias=alias)

    def cos(self, alias: str = None) -> 'Expression':
        """Cosine. Maps to ClickHouse: cos(x)"""
        from .functions import Function

        return Function('cos', self, alias=alias)

    def tan(self, alias: str = None) -> 'Expression':
        """Tangent. Maps to ClickHouse: tan(x)"""
        from .functions import Function

        return Function('tan', self, alias=alias)

    def asin(self, alias: str = None) -> 'Expression':
        """Arc sine. Maps to ClickHouse: asin(x)"""
        from .functions import Function

        return Function('asin', self, alias=alias)

    def acos(self, alias: str = None) -> 'Expression':
        """Arc cosine. Maps to ClickHouse: acos(x)"""
        from .functions import Function

        return Function('acos', self, alias=alias)

    def atan(self, alias: str = None) -> 'Expression':
        """Arc tangent. Maps to ClickHouse: atan(x)"""
        from .functions import Function

        return Function('atan', self, alias=alias)

    def power(self, exponent, alias: str = None) -> 'Expression':
        """
        Raise to power.

        Maps to ClickHouse: pow(base, exponent)

        Args:
            exponent: The exponent value

        Returns:
            Function expression for pow(self, exponent)
        """
        from .functions import Function

        return Function('pow', self, self.wrap(exponent), alias=alias)

    def sign(self, alias: str = None) -> 'Expression':
        """
        Sign of number (-1, 0, or 1).

        Maps to ClickHouse: sign(x)

        Returns:
            Function expression for sign(self)
        """
        from .functions import Function

        return Function('sign', self, alias=alias)

    # ========== Aggregate Functions ==========

    def sum(self, alias: str = None) -> 'Expression':
        """
        Sum aggregate.

        Maps to ClickHouse: sum(x)

        Returns:
            AggregateFunction expression for sum(self)

        Example:
            >>> ds.groupby('category').select(ds['price'].sum())
            >>> # SQL: sum("price")
        """
        from .functions import AggregateFunction

        return AggregateFunction('sum', self, alias=alias)

    def avg(self, alias: str = None) -> 'Expression':
        """
        Average aggregate.

        Maps to ClickHouse: avg(x)

        Returns:
            AggregateFunction expression for avg(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('avg', self, alias=alias)

    def mean(self, alias: str = None) -> 'Expression':
        """Alias for avg(). Average aggregate."""
        return self.avg(alias=alias)

    def count(self, alias: str = None) -> 'Expression':
        """
        Count aggregate.

        Maps to ClickHouse: count(x)

        Returns:
            AggregateFunction expression for count(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('count', self, alias=alias)

    def max(self, alias: str = None) -> 'Expression':
        """
        Maximum aggregate.

        Maps to ClickHouse: max(x)

        Returns:
            AggregateFunction expression for max(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('max', self, alias=alias)

    def min(self, alias: str = None) -> 'Expression':
        """
        Minimum aggregate.

        Maps to ClickHouse: min(x)

        Returns:
            AggregateFunction expression for min(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('min', self, alias=alias)

    def count_distinct(self, alias: str = None) -> 'Expression':
        """
        Count distinct values.

        Maps to ClickHouse: uniq(x)

        Returns:
            AggregateFunction expression for uniq(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('uniq', self, alias=alias)

    def uniq(self, alias: str = None) -> 'Expression':
        """Alias for count_distinct(). Count unique values."""
        return self.count_distinct(alias=alias)

    def stddev(self, alias: str = None) -> 'Expression':
        """
        Standard deviation (population).

        Maps to ClickHouse: stddevPop(x)

        Returns:
            AggregateFunction expression for stddevPop(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('stddevPop', self, alias=alias)

    def stddev_samp(self, alias: str = None) -> 'Expression':
        """
        Standard deviation (sample).

        Maps to ClickHouse: stddevSamp(x)

        Returns:
            AggregateFunction expression for stddevSamp(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('stddevSamp', self, alias=alias)

    def variance(self, alias: str = None) -> 'Expression':
        """
        Variance (population).

        Maps to ClickHouse: varPop(x)

        Returns:
            AggregateFunction expression for varPop(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('varPop', self, alias=alias)

    def var_samp(self, alias: str = None) -> 'Expression':
        """
        Variance (sample).

        Maps to ClickHouse: varSamp(x)

        Returns:
            AggregateFunction expression for varSamp(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('varSamp', self, alias=alias)

    def median(self, alias: str = None) -> 'Expression':
        """
        Median (50th percentile).

        Maps to ClickHouse: median(x)

        Returns:
            AggregateFunction expression for median(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('median', self, alias=alias)

    def quantile(self, level: float, alias: str = None) -> 'Expression':
        """
        Quantile at specified level.

        Maps to ClickHouse: quantile(level)(x)

        Args:
            level: Quantile level (0.0 to 1.0)

        Returns:
            Function expression for quantile
        """
        from .functions import Function

        # ClickHouse uses quantile(level)(column) syntax
        return Function(f'quantile({level})', self, alias=alias)

    def group_array(self, alias: str = None) -> 'Expression':
        """
        Collect values into array.

        Maps to ClickHouse: groupArray(x)

        Returns:
            AggregateFunction expression for groupArray(self)
        """
        from .functions import AggregateFunction

        return AggregateFunction('groupArray', self, alias=alias)

    # ========== Type Conversion ==========

    def cast(self, target_type: str, alias: str = None) -> 'Expression':
        """
        Cast to specified type.

        Maps to ClickHouse: CAST(x AS type)

        Args:
            target_type: Target ClickHouse type (e.g., 'Int64', 'String', 'Float64')

        Returns:
            CastFunction expression

        Example:
            >>> ds['value'].cast('Float64')
            >>> # SQL: CAST("value" AS Float64)
        """
        from .functions import CastFunction

        return CastFunction(self, target_type, alias=alias)

    def to_string(self, alias: str = None) -> 'Expression':
        """
        Convert to String type.

        Maps to ClickHouse: toString(x)

        Returns:
            Function expression for toString(self)
        """
        from .functions import Function

        return Function('toString', self, alias=alias)

    def to_int(self, bits: int = 64, alias: str = None) -> 'Expression':
        """
        Convert to integer type.

        Maps to ClickHouse: toInt32(x), toInt64(x), etc.

        Args:
            bits: Integer bit width (8, 16, 32, 64, 128, 256)

        Returns:
            Function expression for toIntN(self)
        """
        from .functions import Function

        return Function(f'toInt{bits}', self, alias=alias)

    def to_float(self, bits: int = 64, alias: str = None) -> 'Expression':
        """
        Convert to float type.

        Maps to ClickHouse: toFloat32(x), toFloat64(x)

        Args:
            bits: Float bit width (32, 64)

        Returns:
            Function expression for toFloatN(self)
        """
        from .functions import Function

        return Function(f'toFloat{bits}', self, alias=alias)

    def to_date(self, alias: str = None) -> 'Expression':
        """
        Convert to Date type.

        Maps to ClickHouse: toDate(x)

        Returns:
            Function expression for toDate(self)
        """
        from .functions import Function

        return Function('toDate', self, alias=alias)

    def to_datetime(self, timezone: str = None, alias: str = None) -> 'Expression':
        """
        Convert to DateTime type.

        Maps to ClickHouse: toDateTime(x) or toDateTime(x, timezone)

        Args:
            timezone: Optional timezone string

        Returns:
            Function expression for toDateTime(self)
        """
        from .functions import Function

        if timezone:
            return Function('toDateTime', self, Literal(timezone), alias=alias)
        return Function('toDateTime', self, alias=alias)

    # ========== Conditional Functions ==========

    def if_null(self, default, alias: str = None) -> 'Expression':
        """
        Return default value if expression is NULL.

        Maps to ClickHouse: ifNull(x, default)

        Args:
            default: Default value to use if NULL

        Returns:
            Function expression for ifNull(self, default)

        Example:
            >>> ds['value'].if_null(0)
            >>> # SQL: ifNull("value", 0)
        """
        from .functions import Function

        return Function('ifNull', self, self.wrap(default), alias=alias)

    def coalesce(self, *alternatives, alias: str = None) -> 'Expression':
        """
        Return first non-NULL value.

        Maps to ClickHouse: coalesce(x, ...)

        Args:
            *alternatives: Alternative values to try

        Returns:
            Function expression for coalesce(self, alternatives...)

        Example:
            >>> ds['value'].coalesce(ds['backup'], 0)
            >>> # SQL: coalesce("value", "backup", 0)
        """
        from .functions import Function

        all_args = [self] + [self.wrap(a) for a in alternatives]
        return Function('coalesce', *all_args, alias=alias)

    def null_if(self, value, alias: str = None) -> 'Expression':
        """
        Return NULL if expression equals value.

        Maps to ClickHouse: nullIf(x, value)

        Args:
            value: Value to compare against

        Returns:
            Function expression for nullIf(self, value)
        """
        from .functions import Function

        return Function('nullIf', self, self.wrap(value), alias=alias)


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

    OPERATORS = {'+', '-', '*', '/', '//', '%', '**'}

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

        # Handle operators that need special SQL translation
        if self.operator == '**':
            # Python ** -> SQL POW()
            sql = f"POW({left_sql},{right_sql})"
        elif self.operator == '//':
            # Python // (floor division) -> SQL floor(a/b)
            # NOTE: We use floor(a/b) instead of intDiv(a,b) because:
            # - Python // is floor division (rounds toward negative infinity)
            # - intDiv is truncation (rounds toward zero)
            # For negative numbers: -10 // 3 = -4 (Python), intDiv(-10,3) = -3 (ClickHouse)
            sql = f"floor({left_sql}/{right_sql})"
        else:
            sql = f"({left_sql}{self.operator}{right_sql})"

        # Add alias if present and requested
        if kwargs.get('with_alias', False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return ArithmeticExpression(self.operator, copy(self.left), copy(self.right), self.alias)
