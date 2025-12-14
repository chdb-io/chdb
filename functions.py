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
    """

    # ========== String Functions ==========

    @staticmethod
    def upper(expr, alias: str = None) -> Function:
        """Convert to uppercase. Maps to upper(x)."""
        from .expressions import Expression

        return Function('upper', Expression.wrap(expr), alias=alias)

    @staticmethod
    def lower(expr, alias: str = None) -> Function:
        """Convert to lowercase. Maps to lower(x)."""
        from .expressions import Expression

        return Function('lower', Expression.wrap(expr), alias=alias)

    @staticmethod
    def length(expr, alias: str = None) -> Function:
        """String length in bytes. Maps to length(x)."""
        from .expressions import Expression

        return Function('length', Expression.wrap(expr), alias=alias)

    @staticmethod
    def char_length(expr, alias: str = None) -> Function:
        """String length in characters. Maps to char_length(x)."""
        from .expressions import Expression

        return Function('char_length', Expression.wrap(expr), alias=alias)

    @staticmethod
    def substring(expr, offset: int, length: int = None, alias: str = None) -> Function:
        """Extract substring. Maps to substring(s, offset, length)."""
        from .expressions import Expression, Literal

        if length is not None:
            return Function('substring', Expression.wrap(expr), Literal(offset), Literal(length), alias=alias)
        return Function('substring', Expression.wrap(expr), Literal(offset), alias=alias)

    @staticmethod
    def concat(*args, alias: str = None) -> Function:
        """Concatenate strings. Maps to concat(...)."""
        from .expressions import Expression

        return Function('concat', *[Expression.wrap(a) for a in args], alias=alias)

    @staticmethod
    def replace(expr, pattern: str, replacement: str, alias: str = None) -> Function:
        """Replace occurrences. Maps to replace(s, from, to)."""
        from .expressions import Expression, Literal

        return Function('replace', Expression.wrap(expr), Literal(pattern), Literal(replacement), alias=alias)

    @staticmethod
    def trim(expr, alias: str = None) -> Function:
        """Trim whitespace. Maps to trim(x)."""
        from .expressions import Expression

        return Function('trim', Expression.wrap(expr), alias=alias)

    @staticmethod
    def reverse(expr, alias: str = None) -> Function:
        """Reverse string. Maps to reverse(x)."""
        from .expressions import Expression

        return Function('reverse', Expression.wrap(expr), alias=alias)

    # ========== Date/Time Functions ==========

    @staticmethod
    def year(expr, alias: str = None) -> Function:
        """Extract year. Maps to toYear(x)."""
        from .expressions import Expression

        return Function('toYear', Expression.wrap(expr), alias=alias)

    @staticmethod
    def month(expr, alias: str = None) -> Function:
        """Extract month. Maps to toMonth(x)."""
        from .expressions import Expression

        return Function('toMonth', Expression.wrap(expr), alias=alias)

    @staticmethod
    def day(expr, alias: str = None) -> Function:
        """Extract day of month. Maps to toDayOfMonth(x)."""
        from .expressions import Expression

        return Function('toDayOfMonth', Expression.wrap(expr), alias=alias)

    @staticmethod
    def hour(expr, alias: str = None) -> Function:
        """Extract hour. Maps to toHour(x)."""
        from .expressions import Expression

        return Function('toHour', Expression.wrap(expr), alias=alias)

    @staticmethod
    def minute(expr, alias: str = None) -> Function:
        """Extract minute. Maps to toMinute(x)."""
        from .expressions import Expression

        return Function('toMinute', Expression.wrap(expr), alias=alias)

    @staticmethod
    def second(expr, alias: str = None) -> Function:
        """Extract second. Maps to toSecond(x)."""
        from .expressions import Expression

        return Function('toSecond', Expression.wrap(expr), alias=alias)

    @staticmethod
    def day_of_week(expr, alias: str = None) -> Function:
        """Day of week (1=Monday). Maps to toDayOfWeek(x)."""
        from .expressions import Expression

        return Function('toDayOfWeek', Expression.wrap(expr), alias=alias)

    @staticmethod
    def day_of_year(expr, alias: str = None) -> Function:
        """Day of year. Maps to toDayOfYear(x)."""
        from .expressions import Expression

        return Function('toDayOfYear', Expression.wrap(expr), alias=alias)

    @staticmethod
    def quarter(expr, alias: str = None) -> Function:
        """Quarter (1-4). Maps to toQuarter(x)."""
        from .expressions import Expression

        return Function('toQuarter', Expression.wrap(expr), alias=alias)

    @staticmethod
    def date_diff(unit: str, start, end, alias: str = None) -> Function:
        """Date difference. Maps to dateDiff(unit, start, end)."""
        from .expressions import Expression, Literal

        return Function('dateDiff', Literal(unit), Expression.wrap(start), Expression.wrap(end), alias=alias)

    @staticmethod
    def date_add(unit: str, interval: int, date, alias: str = None) -> Function:
        """Add interval to date. Maps to date_add(unit, interval, date)."""
        from .expressions import Expression, Literal

        return Function('date_add', Literal(unit), Literal(interval), Expression.wrap(date), alias=alias)

    @staticmethod
    def date_trunc(unit: str, date, alias: str = None) -> Function:
        """Truncate date to unit. Maps to date_trunc(unit, date)."""
        from .expressions import Expression, Literal

        return Function('date_trunc', Literal(unit), Expression.wrap(date), alias=alias)

    @staticmethod
    def now(alias: str = None) -> Function:
        """Current datetime. Maps to now()."""
        return Function('now', alias=alias)

    @staticmethod
    def today(alias: str = None) -> Function:
        """Current date. Maps to today()."""
        return Function('today', alias=alias)

    @staticmethod
    def yesterday(alias: str = None) -> Function:
        """Yesterday's date. Maps to yesterday()."""
        return Function('yesterday', alias=alias)

    # ========== Math Functions ==========

    @staticmethod
    def abs(expr, alias: str = None) -> Function:
        """Absolute value. Maps to abs(x)."""
        from .expressions import Expression

        return Function('abs', Expression.wrap(expr), alias=alias)

    @staticmethod
    def round(expr, precision: int = 0, alias: str = None) -> Function:
        """Round to N decimal places. Maps to round(x, N)."""
        from .expressions import Expression, Literal

        return Function('round', Expression.wrap(expr), Literal(precision), alias=alias)

    @staticmethod
    def floor(expr, alias: str = None) -> Function:
        """Round down. Maps to floor(x)."""
        from .expressions import Expression

        return Function('floor', Expression.wrap(expr), alias=alias)

    @staticmethod
    def ceil(expr, alias: str = None) -> Function:
        """Round up. Maps to ceiling(x)."""
        from .expressions import Expression

        return Function('ceiling', Expression.wrap(expr), alias=alias)

    @staticmethod
    def ceiling(expr, alias: str = None) -> Function:
        """Round up. Maps to ceiling(x)."""
        return F.ceil(expr, alias=alias)

    @staticmethod
    def sqrt(expr, alias: str = None) -> Function:
        """Square root. Maps to sqrt(x)."""
        from .expressions import Expression

        return Function('sqrt', Expression.wrap(expr), alias=alias)

    @staticmethod
    def pow(base, exponent, alias: str = None) -> Function:
        """Power. Maps to pow(base, exponent)."""
        from .expressions import Expression

        return Function('pow', Expression.wrap(base), Expression.wrap(exponent), alias=alias)

    @staticmethod
    def power(base, exponent, alias: str = None) -> Function:
        """Alias for pow()."""
        return F.pow(base, exponent, alias=alias)

    @staticmethod
    def exp(expr, alias: str = None) -> Function:
        """Exponential (e^x). Maps to exp(x)."""
        from .expressions import Expression

        return Function('exp', Expression.wrap(expr), alias=alias)

    @staticmethod
    def log(expr, base: float = None, alias: str = None) -> Function:
        """Logarithm. Maps to log(x) or log(base, x)."""
        from .expressions import Expression, Literal

        if base is not None:
            return Function('log', Literal(base), Expression.wrap(expr), alias=alias)
        return Function('log', Expression.wrap(expr), alias=alias)

    @staticmethod
    def log10(expr, alias: str = None) -> Function:
        """Base-10 logarithm. Maps to log10(x)."""
        from .expressions import Expression

        return Function('log10', Expression.wrap(expr), alias=alias)

    @staticmethod
    def log2(expr, alias: str = None) -> Function:
        """Base-2 logarithm. Maps to log2(x)."""
        from .expressions import Expression

        return Function('log2', Expression.wrap(expr), alias=alias)

    @staticmethod
    def sin(expr, alias: str = None) -> Function:
        """Sine. Maps to sin(x)."""
        from .expressions import Expression

        return Function('sin', Expression.wrap(expr), alias=alias)

    @staticmethod
    def cos(expr, alias: str = None) -> Function:
        """Cosine. Maps to cos(x)."""
        from .expressions import Expression

        return Function('cos', Expression.wrap(expr), alias=alias)

    @staticmethod
    def tan(expr, alias: str = None) -> Function:
        """Tangent. Maps to tan(x)."""
        from .expressions import Expression

        return Function('tan', Expression.wrap(expr), alias=alias)

    @staticmethod
    def sign(expr, alias: str = None) -> Function:
        """Sign (-1, 0, 1). Maps to sign(x)."""
        from .expressions import Expression

        return Function('sign', Expression.wrap(expr), alias=alias)

    @staticmethod
    def mod(a, b, alias: str = None) -> Function:
        """Modulo. Maps to modulo(a, b)."""
        from .expressions import Expression

        return Function('modulo', Expression.wrap(a), Expression.wrap(b), alias=alias)

    # ========== Conditional Functions ==========

    @staticmethod
    def if_(condition, then_value, else_value, alias: str = None) -> Function:
        """
        Conditional expression.

        Maps to ClickHouse: if(condition, then, else)

        Args:
            condition: Boolean condition
            then_value: Value if condition is true
            else_value: Value if condition is false

        Example:
            >>> F.if_(ds.age > 18, 'adult', 'minor')
            >>> # SQL: if("age" > 18, 'adult', 'minor')
        """
        from .expressions import Expression

        return Function(
            'if', Expression.wrap(condition), Expression.wrap(then_value), Expression.wrap(else_value), alias=alias
        )

    @staticmethod
    def coalesce(*args, alias: str = None) -> Function:
        """
        Return first non-NULL value.

        Maps to ClickHouse: coalesce(...)

        Example:
            >>> F.coalesce(ds.value, ds.backup, 0)
        """
        from .expressions import Expression

        return Function('coalesce', *[Expression.wrap(a) for a in args], alias=alias)

    @staticmethod
    def null_if(expr, value, alias: str = None) -> Function:
        """Return NULL if expr equals value. Maps to nullIf(x, value)."""
        from .expressions import Expression

        return Function('nullIf', Expression.wrap(expr), Expression.wrap(value), alias=alias)

    @staticmethod
    def nullIf(expr, value, alias: str = None) -> Function:
        """Alias for null_if()."""
        return F.null_if(expr, value, alias=alias)

    @staticmethod
    def if_null(expr, default, alias: str = None) -> Function:
        """Return default if expr is NULL. Maps to ifNull(x, default)."""
        from .expressions import Expression

        return Function('ifNull', Expression.wrap(expr), Expression.wrap(default), alias=alias)

    @staticmethod
    def ifNull(expr, default, alias: str = None) -> Function:
        """Alias for if_null()."""
        return F.if_null(expr, default, alias=alias)

    @staticmethod
    def multiIf(*args, alias: str = None) -> Function:
        """
        Multiple conditions (CASE WHEN equivalent).

        Maps to ClickHouse: multiIf(cond1, then1, cond2, then2, ..., else)

        Example:
            >>> F.multiIf(
            ...     ds.age < 13, 'child',
            ...     ds.age < 20, 'teen',
            ...     ds.age < 65, 'adult',
            ...     'senior'
            ... )
        """
        from .expressions import Expression

        return Function('multiIf', *[Expression.wrap(a) for a in args], alias=alias)

    # ========== Aggregate Functions ==========

    @staticmethod
    def sum(expr, alias: str = None) -> AggregateFunction:
        """Sum aggregate. Maps to sum(x)."""
        from .expressions import Expression

        return AggregateFunction('sum', Expression.wrap(expr), alias=alias)

    @staticmethod
    def avg(expr, alias: str = None) -> AggregateFunction:
        """Average aggregate. Maps to avg(x)."""
        from .expressions import Expression

        return AggregateFunction('avg', Expression.wrap(expr), alias=alias)

    @staticmethod
    def count(expr='*', alias: str = None) -> AggregateFunction:
        """Count aggregate. Maps to count(x) or count(*)."""
        from .expressions import Expression, Literal

        if expr == '*':
            return AggregateFunction('count', Literal('*'), alias=alias)
        return AggregateFunction('count', Expression.wrap(expr), alias=alias)

    @staticmethod
    def max(expr, alias: str = None) -> AggregateFunction:
        """Maximum aggregate. Maps to max(x)."""
        from .expressions import Expression

        return AggregateFunction('max', Expression.wrap(expr), alias=alias)

    @staticmethod
    def min(expr, alias: str = None) -> AggregateFunction:
        """Minimum aggregate. Maps to min(x)."""
        from .expressions import Expression

        return AggregateFunction('min', Expression.wrap(expr), alias=alias)

    @staticmethod
    def uniq(expr, alias: str = None) -> AggregateFunction:
        """Count distinct. Maps to uniq(x)."""
        from .expressions import Expression

        return AggregateFunction('uniq', Expression.wrap(expr), alias=alias)

    @staticmethod
    def count_distinct(expr, alias: str = None) -> AggregateFunction:
        """Alias for uniq(). Count distinct values."""
        return F.uniq(expr, alias=alias)

    @staticmethod
    def group_array(expr, alias: str = None) -> AggregateFunction:
        """Collect values into array. Maps to groupArray(x)."""
        from .expressions import Expression

        return AggregateFunction('groupArray', Expression.wrap(expr), alias=alias)

    @staticmethod
    def group_uniq_array(expr, alias: str = None) -> AggregateFunction:
        """Collect unique values into array. Maps to groupUniqArray(x)."""
        from .expressions import Expression

        return AggregateFunction('groupUniqArray', Expression.wrap(expr), alias=alias)

    @staticmethod
    def any(expr, alias: str = None) -> AggregateFunction:
        """Any value. Maps to any(x)."""
        from .expressions import Expression

        return AggregateFunction('any', Expression.wrap(expr), alias=alias)

    @staticmethod
    def any_last(expr, alias: str = None) -> AggregateFunction:
        """Last encountered value. Maps to anyLast(x)."""
        from .expressions import Expression

        return AggregateFunction('anyLast', Expression.wrap(expr), alias=alias)

    @staticmethod
    def stddev_pop(expr, alias: str = None) -> AggregateFunction:
        """Population standard deviation. Maps to stddevPop(x)."""
        from .expressions import Expression

        return AggregateFunction('stddevPop', Expression.wrap(expr), alias=alias)

    @staticmethod
    def stddev_samp(expr, alias: str = None) -> AggregateFunction:
        """Sample standard deviation. Maps to stddevSamp(x)."""
        from .expressions import Expression

        return AggregateFunction('stddevSamp', Expression.wrap(expr), alias=alias)

    @staticmethod
    def var_pop(expr, alias: str = None) -> AggregateFunction:
        """Population variance. Maps to varPop(x)."""
        from .expressions import Expression

        return AggregateFunction('varPop', Expression.wrap(expr), alias=alias)

    @staticmethod
    def var_samp(expr, alias: str = None) -> AggregateFunction:
        """Sample variance. Maps to varSamp(x)."""
        from .expressions import Expression

        return AggregateFunction('varSamp', Expression.wrap(expr), alias=alias)

    @staticmethod
    def median(expr, alias: str = None) -> AggregateFunction:
        """Median (50th percentile). Maps to median(x)."""
        from .expressions import Expression

        return AggregateFunction('median', Expression.wrap(expr), alias=alias)

    # ========== Type Conversion ==========

    @staticmethod
    def cast(expr, target_type: str, alias: str = None) -> CastFunction:
        """Cast to type. Maps to CAST(x AS type)."""
        from .expressions import Expression

        return CastFunction(Expression.wrap(expr), target_type, alias=alias)

    @staticmethod
    def to_string(expr, alias: str = None) -> Function:
        """Convert to string. Maps to toString(x)."""
        from .expressions import Expression

        return Function('toString', Expression.wrap(expr), alias=alias)

    @staticmethod
    def toString(expr, alias: str = None) -> Function:
        """Alias for to_string()."""
        return F.to_string(expr, alias=alias)

    @staticmethod
    def to_int8(expr, alias: str = None) -> Function:
        """Convert to Int8. Maps to toInt8(x)."""
        from .expressions import Expression

        return Function('toInt8', Expression.wrap(expr), alias=alias)

    @staticmethod
    def to_int16(expr, alias: str = None) -> Function:
        """Convert to Int16. Maps to toInt16(x)."""
        from .expressions import Expression

        return Function('toInt16', Expression.wrap(expr), alias=alias)

    @staticmethod
    def to_int32(expr, alias: str = None) -> Function:
        """Convert to Int32. Maps to toInt32(x)."""
        from .expressions import Expression

        return Function('toInt32', Expression.wrap(expr), alias=alias)

    @staticmethod
    def to_int64(expr, alias: str = None) -> Function:
        """Convert to Int64. Maps to toInt64(x)."""
        from .expressions import Expression

        return Function('toInt64', Expression.wrap(expr), alias=alias)

    @staticmethod
    def toInt64(expr, alias: str = None) -> Function:
        """Alias for to_int64()."""
        return F.to_int64(expr, alias=alias)

    @staticmethod
    def to_float32(expr, alias: str = None) -> Function:
        """Convert to Float32. Maps to toFloat32(x)."""
        from .expressions import Expression

        return Function('toFloat32', Expression.wrap(expr), alias=alias)

    @staticmethod
    def to_float64(expr, alias: str = None) -> Function:
        """Convert to Float64. Maps to toFloat64(x)."""
        from .expressions import Expression

        return Function('toFloat64', Expression.wrap(expr), alias=alias)

    @staticmethod
    def toFloat64(expr, alias: str = None) -> Function:
        """Alias for to_float64()."""
        return F.to_float64(expr, alias=alias)

    @staticmethod
    def to_date(expr, alias: str = None) -> Function:
        """Convert to Date. Maps to toDate(x)."""
        from .expressions import Expression

        return Function('toDate', Expression.wrap(expr), alias=alias)

    @staticmethod
    def toDate(expr, alias: str = None) -> Function:
        """Alias for to_date()."""
        return F.to_date(expr, alias=alias)

    @staticmethod
    def to_datetime(expr, timezone: str = None, alias: str = None) -> Function:
        """Convert to DateTime. Maps to toDateTime(x) or toDateTime(x, tz)."""
        from .expressions import Expression, Literal

        if timezone:
            return Function('toDateTime', Expression.wrap(expr), Literal(timezone), alias=alias)
        return Function('toDateTime', Expression.wrap(expr), alias=alias)

    @staticmethod
    def toDateTime(expr, timezone: str = None, alias: str = None) -> Function:
        """Alias for to_datetime()."""
        return F.to_datetime(expr, timezone, alias=alias)

    # ========== Array Functions ==========

    @staticmethod
    def array(*args, alias: str = None) -> Function:
        """Create array. Maps to array(...)."""
        from .expressions import Expression

        return Function('array', *[Expression.wrap(a) for a in args], alias=alias)

    @staticmethod
    def array_length(expr, alias: str = None) -> Function:
        """Array length. Maps to length(arr)."""
        from .expressions import Expression

        return Function('length', Expression.wrap(expr), alias=alias)

    @staticmethod
    def array_join(expr, alias: str = None) -> Function:
        """Expand array to rows. Maps to arrayJoin(arr)."""
        from .expressions import Expression

        return Function('arrayJoin', Expression.wrap(expr), alias=alias)

    @staticmethod
    def has(arr, elem, alias: str = None) -> Function:
        """Check if array contains element. Maps to has(arr, elem)."""
        from .expressions import Expression

        return Function('has', Expression.wrap(arr), Expression.wrap(elem), alias=alias)

    # ========== JSON Functions ==========

    @staticmethod
    def json_extract_string(json, path: str, alias: str = None) -> Function:
        """Extract string from JSON. Maps to JSONExtractString(json, path)."""
        from .expressions import Expression, Literal

        return Function('JSONExtractString', Expression.wrap(json), Literal(path), alias=alias)

    @staticmethod
    def json_extract_int(json, path: str, alias: str = None) -> Function:
        """Extract integer from JSON. Maps to JSONExtractInt(json, path)."""
        from .expressions import Expression, Literal

        return Function('JSONExtractInt', Expression.wrap(json), Literal(path), alias=alias)

    @staticmethod
    def json_extract_float(json, path: str, alias: str = None) -> Function:
        """Extract float from JSON. Maps to JSONExtractFloat(json, path)."""
        from .expressions import Expression, Literal

        return Function('JSONExtractFloat', Expression.wrap(json), Literal(path), alias=alias)

    @staticmethod
    def json_extract_bool(json, path: str, alias: str = None) -> Function:
        """Extract boolean from JSON. Maps to JSONExtractBool(json, path)."""
        from .expressions import Expression, Literal

        return Function('JSONExtractBool', Expression.wrap(json), Literal(path), alias=alias)

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

    # ========== Hash Functions ==========

    @staticmethod
    def md5(expr, alias: str = None) -> Function:
        """MD5 hash. Maps to MD5(x)."""
        from .expressions import Expression

        return Function('MD5', Expression.wrap(expr), alias=alias)

    @staticmethod
    def sha256(expr, alias: str = None) -> Function:
        """SHA256 hash. Maps to SHA256(x)."""
        from .expressions import Expression

        return Function('SHA256', Expression.wrap(expr), alias=alias)

    @staticmethod
    def city_hash64(expr, alias: str = None) -> Function:
        """CityHash64 (fast non-crypto hash). Maps to cityHash64(x)."""
        from .expressions import Expression

        return Function('cityHash64', Expression.wrap(expr), alias=alias)

    @staticmethod
    def sip_hash64(expr, alias: str = None) -> Function:
        """SipHash64. Maps to sipHash64(x)."""
        from .expressions import Expression

        return Function('sipHash64', Expression.wrap(expr), alias=alias)
