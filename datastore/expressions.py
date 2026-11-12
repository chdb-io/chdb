"""
Expression system for DataStore - inspired by pypika but independent implementation
"""

import math
import numpy as np
from typing import Any, Optional, Iterator, List, Type, TYPE_CHECKING
from copy import copy

from .utils import immutable, format_identifier, format_alias
from .exceptions import ValidationError

if TYPE_CHECKING:
    from .conditions import BinaryCondition, Condition
    from .accessors.string import StringAccessor
    from .accessors.datetime import DateTimeAccessor
    from .accessors.array import ArrayAccessor
    from .accessors.json import JsonAccessor
    from .accessors.url import UrlAccessor
    from .accessors.ip import IpAccessor
    from .accessors.geo import GeoAccessor

__all__ = [
    "Node",
    "Expression",
    "Field",
    "Literal",
    "ArithmeticExpression",
    "col",
    "DateTimePropertyExpr",
    "DateTimeMethodExpr",
]


class Node:
    """
    Base class for all expression nodes.
    Provides tree traversal capabilities.
    """

    def nodes(self) -> Iterator["Node"]:
        """Iterate over all nodes in the expression tree."""
        yield self

    def find(self, node_type: Type["Node"]) -> List["Node"]:
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

    # Attributes that refer to the context an expression was built in (e.g.
    # its DataStore) rather than holding children of the expression.
    _context_attributes: frozenset = frozenset()

    def __init__(self, alias: Optional[str] = None):
        self.alias = alias

    @property
    def is_row_wise(self) -> bool:
        """
        Whether each output row is computed from the same input row only: no
        window (``OVER``) or aggregate function anywhere in the tree, and no
        lazy value that may depend on the rows it is computed over (see
        ``OpaqueValue``). A value that isn't row-wise depends on which rows
        it is computed over.
        """
        return not any(
            getattr(node, "is_window", False)
            or getattr(node, "is_aggregate", False)
            or (isinstance(node, OpaqueValue) and not node.subquery)
            for node in self.subtree()
        )

    def subtree(self) -> Iterator["Expression"]:
        """
        Every node of this tree, from every attribute that holds children:
        function arguments and keyword arguments, window specifications, CASE
        branches, IN lists, ... (``nodes()`` doesn't descend into all of
        them). A column expression or lazy condition contributes its
        expression; any other lazy value is an ``OpaqueValue`` leaf.
        """
        seen, stack = set(), [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            if isinstance(node, OpaqueValue):
                continue
            for name, value in vars(node).items():
                if name not in node._context_attributes:
                    stack.extend(_child_nodes(value))

    def columns_read(self) -> Optional[set]:
        """
        Names of the columns this expression reads, from every node of its
        tree (``subtree()``). ``None`` when a part can't be inspected: SQL
        text, or a lazy value with no expression (``OpaqueValue``).
        """
        names = set()
        for node in self.subtree():
            if isinstance(node, (SQLText, OpaqueValue)):
                return None
            if isinstance(node, Field) and node.name not in ROW_POSITION_COLUMNS:
                names.add(node.name)
        return names

    def with_attrs(self, **attrs: Any) -> "Expression":
        """Return a shallow copy of this node with ``attrs`` replaced.

        The copy keeps the node's class and every other attribute (a cast's
        target type, a string-concatenation marker, a window's OVER clause),
        which rebuilding the node from base constructor arguments loses.
        """
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.__dict__.update(attrs)
        return clone

    @immutable
    def as_(self, alias: str) -> "Expression":
        """Set an alias for this expression."""
        self.alias = alias

    @staticmethod
    def wrap(value: Any) -> "Expression":
        """
        Intelligently wrap a value as an Expression.

        - Expression -> return as-is
        - ColumnExpr -> extract underlying expression
        - pd.col(...) (pandas 3.x) -> translate to chdb-ds expression
        - None -> Literal(None)
        - list/tuple -> handle specially
        - other -> Literal(value)
        """
        if isinstance(value, Expression):
            return value

        # pandas 3.x ``pd.col(...)`` interop: translate so users can mix
        # ``ds['a'] + pd.col('b')`` etc. Untranslatable chains (``.astype``,
        # ``np.log``, ...) come back as PandasFallbackExpr — there is no
        # safe way to combine those with a chdb-ds expression tree (the
        # combined node would have to be evaluated entirely in pandas, and
        # we'd lose the SQL pushdown on the chdb-ds side). Raise a clear
        # error pointing at the workaround instead of silently emitting a
        # broken SQL fragment from str(value).
        from .pandas_col_compat import (
            is_pandas_col_expression,
            translate_pandas_expression,
            PandasFallbackExpr,
        )

        if is_pandas_col_expression(value):
            translated = translate_pandas_expression(value)
            if isinstance(translated, Expression):
                return translated
            if isinstance(translated, PandasFallbackExpr):
                raise TypeError(
                    "Cannot combine a chdb-ds expression with an "
                    "untranslatable pd.col(...) chain (e.g. .astype, "
                    "numpy ufunc, .apply). Pre-materialize the chain "
                    "first, then combine: "
                    "``ds = ds.assign(tmp=pd.col('x').astype(int)); "
                    "ds['y'] + ds['tmp']``."
                )

        # Handle ColumnExpr (uses composition, not inheritance)
        # Import here to avoid circular imports
        from .column_expr import ColumnExpr

        if isinstance(value, ColumnExpr):
            if value._expr is None:
                raise TypeError(
                    "This column value is computed in pandas and has no SQL expression; "
                    "it can't be part of a SQL expression"
                )
            return value._expr

        return Literal(value)

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Convert expression to SQL string."""
        raise NotImplementedError(f"{type(self).__name__} must implement to_sql()")

    # ========== Comparison Operators ==========

    def __eq__(self, other: Any) -> "Condition":
        from .conditions import BinaryCondition

        # pandas semantics: col == None returns False for ALL rows
        # This is element-wise comparison with Python singleton None,
        # NOT a check for NA values (use .isna() for that)
        if other is None:
            return BinaryCondition("=", Literal(0), Literal(1))  # Always False

        return BinaryCondition("=", self, self.wrap(other))

    def __ne__(self, other: Any) -> "Condition":
        from .conditions import BinaryCondition

        # pandas semantics: col != None returns True for ALL rows
        # This is element-wise comparison with Python singleton None,
        # NOT a check for non-NA values (use .notna() for that)
        if other is None:
            return BinaryCondition("=", Literal(1), Literal(1))  # Always True

        return BinaryCondition("!=", self, self.wrap(other))

    def __gt__(self, other: Any) -> "BinaryCondition":
        from .conditions import BinaryCondition

        return BinaryCondition(">", self, self.wrap(other))

    def __ge__(self, other: Any) -> "BinaryCondition":
        from .conditions import BinaryCondition

        return BinaryCondition(">=", self, self.wrap(other))

    def __lt__(self, other: Any) -> "BinaryCondition":
        from .conditions import BinaryCondition

        return BinaryCondition("<", self, self.wrap(other))

    def __le__(self, other: Any) -> "BinaryCondition":
        from .conditions import BinaryCondition

        return BinaryCondition("<=", self, self.wrap(other))

    # ========== Advanced Condition Methods ==========

    def isnull(self) -> "Condition":
        """
        Create IS NULL condition.

        Example:
            >>> Field('name').isnull()
            >>> # Generates: "name" IS NULL
        """
        from .conditions import UnaryCondition

        return UnaryCondition("IS NULL", self)

    def notnull(self) -> "Condition":
        """
        Create IS NOT NULL condition.

        Example:
            >>> Field('name').notnull()
            >>> # Generates: "name" IS NOT NULL
        """
        from .conditions import UnaryCondition

        return UnaryCondition("IS NOT NULL", self)

    def isin(self, values) -> "Condition":
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

    def notin(self, values) -> "Condition":
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

    def between(self, lower, upper, inclusive: str = "both") -> "Condition":
        """
        Create BETWEEN condition.

        Args:
            lower: Lower bound
            upper: Upper bound
            inclusive: Include boundaries. Valid options are:
                'both' (default): Include both boundaries
                'neither': Exclude both boundaries
                'left': Include left boundary only
                'right': Include right boundary only

        Example:
            >>> Field('age').between(18, 65)
            >>> # Generates: "age" BETWEEN 18 AND 65
            >>> Field('age').between(18, 65, inclusive='neither')
            >>> # Generates: "age" > 18 AND "age" < 65
        """
        from .conditions import BetweenCondition

        return BetweenCondition(
            self, self.wrap(lower), self.wrap(upper), inclusive=inclusive
        )

    def like(self, pattern: str) -> "Condition":
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

    def notlike(self, pattern: str) -> "Condition":
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

    def ilike(self, pattern: str) -> "Condition":
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

    def __add__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("+", self, self.wrap(other))

    def __sub__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("-", self, self.wrap(other))

    def __mul__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("*", self, self.wrap(other))

    def __truediv__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("/", self, self.wrap(other))

    def __floordiv__(self, other: Any) -> "ArithmeticExpression":
        # Floor division: a // b  == intDiv(a, b) in ClickHouse
        return ArithmeticExpression("//", self, self.wrap(other))

    def __mod__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("%", self, self.wrap(other))

    def __pow__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("**", self, self.wrap(other))

    # ========== Reverse Arithmetic Operators ==========

    def __radd__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("+", self.wrap(other), self)

    def __rsub__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("-", self.wrap(other), self)

    def __rmul__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("*", self.wrap(other), self)

    def __rtruediv__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("/", self.wrap(other), self)

    def __rfloordiv__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("//", self.wrap(other), self)

    def __rpow__(self, other: Any) -> "ArithmeticExpression":
        return ArithmeticExpression("**", self.wrap(other), self)

    # ========== Unary Operators ==========

    def __neg__(self) -> "ArithmeticExpression":
        return ArithmeticExpression("-", Literal(0), self)

    # ========== String/Utility Methods ==========

    def __str__(self) -> str:
        return self.to_sql()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_sql()!r})"

    # ========== Accessor Properties ==========

    @property
    def str(self) -> "StringAccessor":
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
    def dt(self) -> "DateTimeAccessor":
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

    @property
    def arr(self) -> "ArrayAccessor":
        """
        Accessor for array functions.

        Provides a Pandas-like interface to ClickHouse array functions.

        Example:
            >>> ds['tags'].arr.length           # length(tags)
            >>> ds['nums'].arr.array_sum()      # arraySum(nums)
            >>> ds['arr'].arr.array_first()     # arrayElement(arr, 1)
            >>> ds['arr'].arr.has('value')      # has(arr, 'value')

        Returns:
            ArrayAccessor with array function methods
        """
        from .accessors.array import ArrayAccessor

        return ArrayAccessor(self)

    @property
    def json(self) -> "JsonAccessor":
        """
        Accessor for JSON functions.

        Provides a Pandas-like interface to ClickHouse JSON functions.

        Example:
            >>> ds['data'].json.json_extract_string('name')  # JSONExtractString(data, 'name')
            >>> ds['data'].json.json_extract_int('age')      # JSONExtractInt(data, 'age')
            >>> ds['data'].json.is_valid_json()              # isValidJSON(data)

        Returns:
            JsonAccessor with JSON function methods
        """
        from .accessors.json import JsonAccessor

        return JsonAccessor(self)

    @property
    def url(self) -> "UrlAccessor":
        """
        Accessor for URL functions.

        Provides a Pandas-like interface to ClickHouse URL functions.

        Example:
            >>> ds['link'].url.domain()                      # domain(link)
            >>> ds['link'].url.url_path()                    # path(link)
            >>> ds['link'].url.extract_url_parameter('id')   # extractURLParameter(link, 'id')

        Returns:
            UrlAccessor with URL function methods
        """
        from .accessors.url import UrlAccessor

        return UrlAccessor(self)

    @property
    def ip(self) -> "IpAccessor":
        """
        Accessor for IP address functions.

        Provides a Pandas-like interface to ClickHouse IP functions.

        Example:
            >>> ds['ip'].ip.to_ipv4()           # toIPv4(ip)
            >>> ds['ip'].ip.is_ipv4_string()    # isIPv4String(ip)
            >>> ds['ip'].ip.ipv4_to_ipv6()      # IPv4ToIPv6(ip)

        Returns:
            IpAccessor with IP function methods
        """
        from .accessors.ip import IpAccessor

        return IpAccessor(self)

    @property
    def geo(self) -> "GeoAccessor":
        """
        Accessor for geo/distance functions.

        Provides a Pandas-like interface to ClickHouse geo functions.

        Example:
            >>> ds['vec'].geo.l2_norm()                    # L2Norm(vec)
            >>> ds['vec'].geo.l2_normalize()               # L2Normalize(vec)

        Returns:
            GeoAccessor with geo function methods
        """
        from .accessors.geo import GeoAccessor

        return GeoAccessor(self)

    # ========== Function Methods ==========
    # NOTE: Function methods (abs, round, sum, avg, etc.) are now dynamically
    # injected from the FunctionRegistry at the end of this module.
    # This eliminates code duplication - each function is defined once in
    # function_definitions.py and automatically available here.
    #
    # Methods like cast(), to_int(), to_float() that have special logic are
    # defined below. Standard functions are injected via the registry.

    def cast(self, target_type: str, alias: str = None) -> "Expression":
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

    def to_int(self, bits: int = 64, alias: str = None) -> "Expression":
        """
        Convert to integer type.

        Maps to ClickHouse: toInt32(x), toInt64(x), etc.

        Args:
            bits: Integer bit width (8, 16, 32, 64, 128, 256)

        Returns:
            Function expression for toIntN(self)
        """
        from .functions import Function

        return Function(f"toInt{bits}", self, alias=alias)

    def to_float(self, bits: int = 64, alias: str = None) -> "Expression":
        """
        Convert to float type.

        Maps to ClickHouse: toFloat32(x), toFloat64(x)

        Args:
            bits: Float bit width (32, 64)

        Returns:
            Function expression for toFloatN(self)
        """
        from .functions import Function

        return Function(f"toFloat{bits}", self, alias=alias)

    def quantile(self, level: float, alias: str = None) -> "Expression":
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
        return Function(f"quantile({level})", self, alias=alias)

    # Opt out of numpy ufunc dispatch entirely. Without this, numpy falls
    # back to calling same-named methods (Field.log injected by
    # FunctionRegistry, Field.sqrt, etc.), which silently pushes
    # ``np.log(pd.col('x'))`` to a ClickHouse log() call instead of letting
    # pandas own the semantics. Setting it to ``None`` makes numpy raise
    # TypeError; the pd.col translator catches it and wraps the original
    # expression in PandasFallbackExpr so the pandas segment evaluates it.
    __array_ufunc__ = None

class Field(Expression):
    """
    Represents a field (column) in a data source.

    Example:
        >>> Field('name')
        >>> Field('age', table='customers')
    """

    def __init__(
        self, name: str, table: Optional[str] = None, alias: Optional[str] = None
    ):
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
        field_sql = ".".join(parts)

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(field_sql, self.alias, quote_char)

        return field_sql

    def __copy__(self):
        return Field(self.name, self.table, self.alias)


# chDB's virtual row-position column, which windows order by to keep the
# source's row order; it is not a column of the data.
ROW_POSITION_COLUMNS = frozenset({"_row_id"})


def _child_nodes(value) -> Iterator[Expression]:
    """
    The expression nodes held in one attribute of an expression node, through
    nested lists, tuples, sets and dicts. A column expression or lazy
    condition contributes its expression and any other lazy value an
    ``OpaqueValue``; plain data (literals, arrays, pandas objects) holds none.
    """
    from .column_expr import ColumnExpr
    from .core import DataStore
    from .lazy_ops import LazyOp
    from .lazy_result import LazyCondition
    from .pandas_col_compat import PandasFallbackExpr

    if isinstance(value, Expression):
        yield value
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _child_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _child_nodes(item)
    elif isinstance(value, ColumnExpr) and value._exec_mode == "expr" and value._expr is not None:
        yield value._expr
    elif isinstance(value, LazyCondition):
        yield from _child_nodes(value.condition)
    elif isinstance(value, (DataStore, LazyOp, PandasFallbackExpr)) or hasattr(value, "_execute"):
        yield OpaqueValue(value)


class OpaqueValue(Expression):
    """
    A child of an expression that is not an expression: a lazy value with no
    expression behind it (a pandas method result, a lazy Series, a pd.col
    chain left to pandas, a lazy op) or a subquery. The columns it reads are
    not known. Only a subquery has SQL and is row-wise (computed once for
    every row); any other one is evaluated in pandas.
    """

    def __init__(self, value: Any):
        from .core import DataStore

        super().__init__()
        self.value = value
        self.subquery = isinstance(value, DataStore)

    def __repr__(self):
        return f"OpaqueValue({type(self.value).__name__})"


class SQLText(Expression):
    """Already-rendered SQL; the columns it reads are not known."""

    def __init__(self, sql: str, alias: Optional[str] = None):
        super().__init__(alias)
        self.sql = sql

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        return self.sql


class Star(Expression):
    """
    Represents * in SQL (e.g., SELECT *, COUNT(*)).

    Example:
        >>> Star()  # generates: *
        >>> AggregateFunction('count', Star())  # generates: count(*)
    """

    def __init__(self, alias: Optional[str] = None):
        super().__init__(alias)

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for star (*)."""
        return "*"

    def __repr__(self):
        return "Star()"


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
        # Handle ColumnExpr - delegate to its to_sql() to avoid infinite recursion
        # (ColumnExpr.__str__ executes, which could call to_sql again)
        from .column_expr import ColumnExpr

        if isinstance(self.value, ColumnExpr):
            return self.value.to_sql(quote_char=quote_char, **kwargs)

        # Handle Expression - delegate to its to_sql()
        if isinstance(self.value, Expression):
            return self.value.to_sql(quote_char=quote_char, **kwargs)

        value = self.value
        if isinstance(value, (np.number, np.bool_)) and not isinstance(
            value, np.timedelta64
        ):
            value = value.item()
        elif isinstance(value, np.datetime64):
            import pandas as pd

            value = None if np.isnat(value) else pd.Timestamp(value)

        # Convert Python value to SQL literal
        if value is None:
            sql = "NULL"
        elif isinstance(value, bool):
            sql = "TRUE" if value else "FALSE"
        elif isinstance(value, float) and math.isnan(value):
            sql = "NULL"
        elif isinstance(value, (int, float)):
            sql = str(value)
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            sql = f"'{escaped}'"
        else:
            # Handle special types
            import datetime
            import uuid

            if isinstance(value, datetime.datetime):
                # Format datetime as ISO string; pd.Timestamp also carries
                # nanoseconds
                text = value.strftime("%Y-%m-%d %H:%M:%S")
                nanosecond = getattr(value, "nanosecond", 0)
                if nanosecond:
                    text += f".{value.microsecond:06d}{nanosecond:03d}"
                elif value.microsecond:
                    text += f".{value.microsecond:06d}"
                sql = f"'{text}'"
            elif isinstance(value, datetime.date):
                # Format date as ISO string
                sql = f"'{value.strftime('%Y-%m-%d')}'"
            elif isinstance(value, datetime.time):
                # Format time as ISO string
                sql = f"'{value.strftime('%H:%M:%S')}'"
            elif isinstance(value, uuid.UUID):
                # Format UUID as string
                sql = f"'{str(value)}'"
            else:
                # Fallback: convert to string (use repr to avoid triggering __str__ side effects)
                sql = f"'{repr(value)}'"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
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

    OPERATORS = {"+", "-", "*", "/", "//", "%", "**"}

    def __init__(
        self,
        operator: str,
        left: Expression,
        right: Expression,
        alias: Optional[str] = None,
    ):
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
        if self.operator == "**":
            # Python ** -> SQL POW()
            sql = f"POW({left_sql},{right_sql})"
        elif self.operator == "//":
            # Python // (floor division) -> SQL floor(a/b)
            # NOTE: We use floor(a/b) instead of intDiv(a,b) because:
            # - Python // is floor division (rounds toward negative infinity)
            # - intDiv is truncation (rounds toward zero)
            # For negative numbers: -10 // 3 = -4 (Python), intDiv(-10,3) = -3 (ClickHouse)
            sql = f"floor({left_sql}/{right_sql})"
        elif self.operator == "+" and self._involves_string_operand():
            # String concatenation: chDB/ClickHouse doesn't support '+' for strings
            # Must use concat() function instead
            sql = f"concat({left_sql},{right_sql})"
        else:
            sql = f"({left_sql}{self.operator}{right_sql})"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def _involves_string_operand(self) -> bool:
        """
        Check if this expression involves string operands.

        Returns True if:
        - This expression itself is marked as string type (set by ColumnExpr.__add__)
        - Either operand is a string Literal
        - Either operand is marked as a string type expression

        This enables automatic conversion of '+' to concat() for string concatenation.
        """
        # First check if this expression itself is marked as string type
        if getattr(self, "_is_string_type", False):
            return True

        def is_string_expr(expr) -> bool:
            # Check if it's a string literal
            if isinstance(expr, Literal):
                return isinstance(expr.value, str)
            # Check if expression has been marked as string type
            if getattr(expr, "_is_string_type", False):
                return True
            # Recursively check nested ArithmeticExpression (for chained concatenation)
            if isinstance(expr, ArithmeticExpression):
                return expr._involves_string_operand()
            return False

        return is_string_expr(self.left) or is_string_expr(self.right)

    def __copy__(self):
        result = ArithmeticExpression(
            self.operator, copy(self.left), copy(self.right), self.alias
        )
        # Preserve string type marker for string concatenation support
        if getattr(self, "_is_string_type", False):
            result._is_string_type = True
        return result


def col(name: str) -> Field:
    """
    Create a column reference for use in expressions.

    This is the preferred way to reference columns in aggregations
    and other operations, especially with groupby().agg().

    Args:
        name: Column name

    Returns:
        Field expression that can be used with aggregation methods

    Example:
        >>> from datastore import col
        >>>
        >>> # Use in groupby aggregation
        >>> ds.groupby("region").agg(
        ...     total_revenue=col("revenue").sum(),
        ...     avg_quantity=col("quantity").mean(),
        ...     order_count=col("order_id").count()
        ... )
        >>>
        >>> # Use in filters
        >>> ds.filter(col("age") > 18)
        >>>
        >>> # Use in select with expressions
        >>> ds.select(col("price") * col("quantity"))
    """
    return Field(name)


def is_star_item(item: Any) -> bool:
    """Whether ``item`` is the ``"*"`` marker of a SELECT field list.

    ``Field.__eq__`` builds a condition, so ``field == "*"`` is truthy and
    must never be used to detect the marker.
    """
    return isinstance(item, str) and item.strip() == "*"


def projection_name(item: Any) -> Optional[str]:
    """Output column name of a SELECT field list item, or None if unnamed."""
    if isinstance(item, str):
        return item.strip('"')
    alias = getattr(item, "alias", None)
    if alias:
        return alias
    if isinstance(item, Field):
        return item.name.strip('"')
    return None


def bind_projection_item(items: List[Any], item: Any) -> None:
    """Bind ``item`` into an ordered SELECT list.

    An item whose output name is already in the list replaces that entry in
    place (pandas ``df[col] = ...`` keeps the column position); otherwise it is
    appended.
    """
    name = projection_name(item)
    if name is not None:
        for i, existing in enumerate(items):
            if projection_name(existing) == name:
                items[i] = item
                return
    items.append(item)


def resolve_computed_references(
    expr: Optional["Expression"], computed: dict, _visited: set = None
) -> Optional["Expression"]:
    """
    Recursively expand references to computed columns in an expression.

    Args:
        expr: Expression to resolve
        computed: Column name -> expression defining it
        _visited: Names being expanded; a reference to one of them (e.g.
            ``a = a + 1``) is kept as a plain column reference.

    Returns:
        Expression with computed column references expanded
    """
    return _resolve_computed(
        expr, computed, set() if _visited is None else _visited, {}
    )


def _resolve_computed(expr, computed: dict, _visited: set, memo: dict):
    """
    Memoized step of :func:`resolve_computed_references`: a subexpression
    shared by several references (e.g. ``c2 = c1 + c1``) is resolved once
    per set of names being expanded, so a chain of computed columns is
    resolved in linear rather than exponential time.
    """
    if expr is None:
        return None
    key = (id(expr), frozenset(_visited))
    hit = memo.get(key)
    if hit is not None and hit[0] is expr:
        return hit[1]
    resolved = _resolve_computed_node(expr, computed, _visited, memo)
    memo[key] = (expr, resolved)
    return resolved


def _resolve_computed_node(expr, computed: dict, _visited: set, memo: dict):
    from .functions import Function

    # If it's a Field that references a computed column, expand it
    if isinstance(expr, Field):
        col_name = expr.name.strip("\"'")  # Remove quotes if present
        if col_name in computed:
            # Check for circular reference to prevent infinite recursion
            # This happens when overwriting a column: a = a + 1, then a = a * 2
            # In this case, we should NOT expand the reference - just use the Field as-is
            if col_name in _visited:
                return expr
            # Mark this column as being visited
            _visited.add(col_name)
            try:
                # Recursively resolve the computed column's expression
                resolved = _resolve_computed(
                    computed[col_name], computed, _visited, memo
                )
                # Preserve any alias from the original Field
                if expr.alias:
                    resolved = resolved.with_attrs(alias=expr.alias)
                return resolved
            finally:
                # Remove from visited after processing (allow re-visiting in different branches)
                _visited.discard(col_name)
        return expr

    # For ArithmeticExpression, recursively resolve left and right
    if isinstance(expr, ArithmeticExpression):
        resolved_left = _resolve_computed(expr.left, computed, _visited, memo)
        resolved_right = _resolve_computed(expr.right, computed, _visited, memo)
        # Only create new expression if something changed
        if resolved_left is expr.left and resolved_right is expr.right:
            return expr
        return expr.with_attrs(left=resolved_left, right=resolved_right)

    # For Function, recursively resolve all arguments
    if isinstance(expr, Function):
        resolved_args = [
            _resolve_computed(arg, computed, _visited, memo) for arg in expr.args
        ]
        # Only create new function if something changed
        if all(r is o for r, o in zip(resolved_args, expr.args)):
            return expr
        return expr.with_attrs(args=resolved_args)

    # For BinaryCondition (e.g., x > 5, a = b), resolve both sides
    from .conditions import BinaryCondition, CompoundCondition, NullSafeCondition

    if isinstance(expr, BinaryCondition):
        resolved_left = _resolve_computed(expr.left, computed, _visited, memo)
        resolved_right = _resolve_computed(expr.right, computed, _visited, memo)
        if resolved_left is expr.left and resolved_right is expr.right:
            return expr
        return expr.with_attrs(left=resolved_left, right=resolved_right)

    # For CompoundCondition (AND, OR), resolve both sides
    if isinstance(expr, CompoundCondition):
        resolved_left = _resolve_computed(expr.left, computed, _visited, memo)
        resolved_right = _resolve_computed(expr.right, computed, _visited, memo)
        if resolved_left is expr.left and resolved_right is expr.right:
            return expr
        return expr.with_attrs(left=resolved_left, right=resolved_right)

    # Conditions wrapping one or more expressions (an IN subquery is not
    # descended into: it has its own column scope)
    from .conditions import (
        BetweenCondition,
        InCondition,
        LikeCondition,
        NotCondition,
        UnaryCondition,
    )

    children = None
    if isinstance(expr, (NotCondition, NullSafeCondition)):
        children = ("condition",)
    elif isinstance(expr, (UnaryCondition, InCondition, LikeCondition)):
        children = ("expression",)
    elif isinstance(expr, BetweenCondition):
        children = ("expression", "lower", "upper")
    if children:
        resolved = {
            name: _resolve_computed(getattr(expr, name), computed, _visited, memo)
            for name in children
        }
        if all(resolved[name] is getattr(expr, name) for name in children):
            return expr
        return expr.with_attrs(**resolved)

    # For Literal and other types, return as-is
    return expr


class DateTimePropertyExpr(Expression):
    """
    Expression representing a datetime accessor property (.dt.year, .dt.month, etc.)

    This is a lazy expression that defers execution to ExpressionEvaluator,
    which decides at runtime whether to use chDB SQL functions or pandas .dt accessor
    based on function_config settings.

    Example:
        >>> expr = DateTimePropertyExpr(Field('date_col'), 'year')
        >>> # At execution time, evaluator checks function_config.should_use_pandas('year')
        >>> # If pandas: s.dt.year
        >>> # If chDB: toYear(date_col)
    """

    # Mapping from property name to chDB function name
    CHDB_FUNCTION_MAP = {
        "year": "toYear",
        "month": "toMonth",
        "day": "toDayOfMonth",
        "hour": "toHour",
        "minute": "toMinute",
        "second": "toSecond",
        "dayofweek": "toDayOfWeek",
        "weekday": "toDayOfWeek",
        "dayofyear": "toDayOfYear",
        "week": "toISOWeek",
        "weekofyear": "toISOWeek",
        "quarter": "toQuarter",
        "date": "toDate",
    }

    def __init__(
        self, source_expr: Expression, property_name: str, alias: Optional[str] = None
    ):
        """
        Args:
            source_expr: The datetime column expression
            property_name: The property name (e.g., 'year', 'month', 'dayofweek')
            alias: Optional alias for the result
        """
        super().__init__(alias)
        self.source_expr = source_expr
        self.property_name = property_name

    # Datetime properties that should be cast to Int32 to match pandas dtype
    _INT32_PROPERTIES = frozenset(
        {
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dayofweek",
            "weekday",
            "dayofyear",
            "quarter",
            "week",
            "weekofyear",
        }
    )

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL using chDB function."""
        ch_func = self.CHDB_FUNCTION_MAP.get(self.property_name)
        if not ch_func:
            raise ValueError(
                f"No chDB function mapping for datetime property: {self.property_name}"
            )

        source_sql = self.source_expr.to_sql(quote_char=quote_char, **kwargs)

        # Handle dayofweek adjustment (chDB is 1-7 Monday, pandas is 0-6 Monday)
        if self.property_name in ("dayofweek", "weekday"):
            result = f"({ch_func}({source_sql}) - 1)"
        else:
            result = f"{ch_func}({source_sql})"

        # Cast to Int32 to match pandas dtype for datetime properties
        # chDB datetime functions return various integer types (UInt8, UInt16, etc.)
        # but pandas returns int32 for all datetime properties
        if self.property_name in self._INT32_PROPERTIES:
            result = f"toInt32({result})"

        if self.alias:
            return f"{result} AS {format_identifier(self.alias, quote_char)}"
        return result

    def nodes(self) -> Iterator["Node"]:
        yield self
        yield from self.source_expr.nodes()


class DateTimeMethodExpr(Expression):
    """
    Expression representing a datetime accessor method call (.dt.strftime(), .dt.floor(), etc.)

    Similar to DateTimePropertyExpr but for methods that take arguments.
    """

    # Mapping from method name to chDB function name
    CHDB_FUNCTION_MAP = {
        "strftime": "formatDateTime",
        "floor_dt": "FLOOR_DT",  # Special marker for floor datetime
        "ceil_dt": "CEIL_DT",  # Special marker for ceil datetime
        "round_dt": "ROUND_DT",  # Special marker for round datetime
        "normalize": "toStartOfDay",  # Normalize = start of day
        "day_name": "dateName",  # dateName('weekday', date) -> 'Monday', etc.
        "month_name": "dateName",  # dateName('month', date) -> 'January', etc.
    }

    # Mapping from pandas frequency to ClickHouse floor function
    FREQ_TO_FLOOR_FUNC = {
        "h": "toStartOfHour",
        "H": "toStartOfHour",
        "min": "toStartOfMinute",
        "T": "toStartOfMinute",
        "s": "toStartOfSecond",
        "S": "toStartOfSecond",
        "d": "toStartOfDay",
        "D": "toStartOfDay",
    }

    # Mapping from pandas frequency to ClickHouse add function and interval
    FREQ_TO_ADD_FUNC = {
        "h": ("addHours", 1, 1800),  # (add_func, amount, half_interval_seconds)
        "H": ("addHours", 1, 1800),
        "min": ("addMinutes", 1, 30),
        "T": ("addMinutes", 1, 30),
        "s": ("addSeconds", 1, 0.5),
        "S": ("addSeconds", 1, 0.5),
        "d": ("addDays", 1, 43200),  # 12 hours
        "D": ("addDays", 1, 43200),
    }

    # Mapping from Python strftime format codes to ClickHouse formatDateTime codes
    # Python: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    # ClickHouse: https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions#formatDateTime
    STRFTIME_FORMAT_MAP = {
        "%M": "%i",  # Python %M = minute (00-59), ClickHouse %i = minute
        # ClickHouse %M = full month name (January, etc.) - different meaning!
    }

    @classmethod
    def _convert_strftime_format(cls, fmt: str) -> str:
        """Convert Python strftime format to ClickHouse formatDateTime format."""
        result = fmt
        for py_code, ch_code in cls.STRFTIME_FORMAT_MAP.items():
            result = result.replace(py_code, ch_code)
        return result

    def __init__(
        self,
        source_expr: Expression,
        method_name: str,
        args: tuple = (),
        kwargs: dict = None,
        alias: Optional[str] = None,
    ):
        super().__init__(alias)
        self.source_expr = source_expr
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs or {}

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL using chDB function."""
        ch_func = self.CHDB_FUNCTION_MAP.get(self.method_name)
        if not ch_func:
            raise ValueError(
                f"No chDB function mapping for datetime method: {self.method_name}"
            )

        source_sql = self.source_expr.to_sql(quote_char=quote_char, **kwargs)

        if self.method_name == "strftime" and self.args:
            fmt = self._convert_strftime_format(self.args[0])
            result = f"{ch_func}({source_sql}, '{fmt}')"
        elif self.method_name == "floor_dt":
            result = self._build_floor_sql(source_sql)
        elif self.method_name == "ceil_dt":
            result = self._build_ceil_sql(source_sql)
        elif self.method_name == "round_dt":
            result = self._build_round_sql(source_sql)
        elif self.method_name == "normalize":
            result = f"{ch_func}({source_sql})"
        elif self.method_name == "day_name":
            # dateName('weekday', date) returns 'Monday', 'Tuesday', etc.
            result = f"{ch_func}('weekday', {source_sql})"
        elif self.method_name == "month_name":
            # dateName('month', date) returns 'January', 'February', etc.
            result = f"{ch_func}('month', {source_sql})"
        else:
            result = f"{ch_func}({source_sql})"

        if self.alias:
            return f"{result} AS {format_identifier(self.alias, quote_char)}"
        return result

    def _build_floor_sql(self, source_sql: str) -> str:
        """Build SQL for floor operation."""
        freq = self.args[0] if self.args else "D"
        floor_func = self.FREQ_TO_FLOOR_FUNC.get(freq, "toStartOfDay")
        return f"{floor_func}({source_sql})"

    def _build_ceil_sql(self, source_sql: str) -> str:
        """Build SQL for ceil operation.

        ceil = if(floor(x) == x, x, floor(x) + 1 unit)
        """
        freq = self.args[0] if self.args else "D"
        floor_func = self.FREQ_TO_FLOOR_FUNC.get(freq, "toStartOfDay")
        add_info = self.FREQ_TO_ADD_FUNC.get(freq, ("addDays", 1, 43200))
        add_func, amount, _ = add_info

        floor_expr = f"{floor_func}({source_sql})"
        ceil_expr = f"{add_func}({floor_expr}, {amount})"
        # If already at boundary, return as-is; otherwise ceil
        return f"if({floor_expr} = {source_sql}, {source_sql}, {ceil_expr})"

    def _build_round_sql(self, source_sql: str) -> str:
        """Build SQL for round operation.

        round = if(time_to_next_boundary <= half_interval, ceil, floor)
        For simplicity, we use: if the fractional part >= 0.5, ceil; else floor
        """
        freq = self.args[0] if self.args else "D"
        floor_func = self.FREQ_TO_FLOOR_FUNC.get(freq, "toStartOfDay")
        add_info = self.FREQ_TO_ADD_FUNC.get(freq, ("addDays", 1, 43200))
        add_func, amount, half_seconds = add_info

        floor_expr = f"{floor_func}({source_sql})"
        ceil_expr = f"{add_func}({floor_expr}, {amount})"

        # Calculate seconds from floor to original
        diff_seconds = f"dateDiff('second', {floor_expr}, {source_sql})"

        # If diff >= half_interval, round up (ceil), else round down (floor)
        return f"if({diff_seconds} >= {half_seconds}, {ceil_expr}, {floor_expr})"

    def nodes(self) -> Iterator["Node"]:
        yield self
        yield from self.source_expr.nodes()


class IsoCalendarComponentExpr(Expression):
    """
    Expression representing an isocalendar component (.dt.isocalendar().year/week/day).

    This is a lazy expression that evaluates to the ISO calendar component
    of a datetime column.
    """

    # Mapping from component name to chDB function
    CHDB_FUNCTION_MAP = {
        "year": "toISOYear",
        "week": "toISOWeek",
        "day": "toDayOfWeek",  # with mode 1 for Monday=1
    }

    @staticmethod
    def cast_function(component: str) -> str:
        """Return the chDB cast that gives a component pandas' isocalendar() dtype.

        pandas 3.1 made ``year`` Int32 instead of UInt32, so that years before
        year 1 are representable (pandas-dev/pandas#66851); ``week`` and ``day``
        stay UInt32.
        """
        if component == "year":
            from .pandas_col_compat import _parse_pandas_version

            if _parse_pandas_version() >= (3, 1):
                return "toInt32"
        return "toUInt32"

    def __init__(
        self, source_expr: Expression, component: str, alias: Optional[str] = None
    ):
        """
        Args:
            source_expr: The datetime column expression
            component: 'year', 'week', or 'day'
            alias: Optional alias for the result
        """
        super().__init__(alias)
        self.source_expr = source_expr
        self.component = component

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL using chDB function."""
        ch_func = self.CHDB_FUNCTION_MAP.get(self.component)
        if not ch_func:
            raise ValueError(f"Unknown isocalendar component: {self.component}")

        source_sql = self.source_expr.to_sql(quote_char=quote_char, **kwargs)

        # toDayOfWeek needs mode=0 for Monday=1 (ISO standard: 1-7)
        # mode=1 would return Monday=0 (0-6)
        if self.component == "day":
            result = f"{ch_func}({source_sql}, 0)"
        else:
            result = f"{ch_func}({source_sql})"

        # Cast to match pandas isocalendar() dtype
        result = f"{self.cast_function(self.component)}({result})"

        if self.alias:
            return f"{result} AS {format_identifier(self.alias, quote_char)}"
        return result

    def nodes(self) -> Iterator["Node"]:
        yield self
        yield from self.source_expr.nodes()


# =============================================================================
# INJECT FUNCTION METHODS FROM REGISTRY
# =============================================================================
# Function methods (abs, round, sum, avg, upper, lower, etc.) are now
# dynamically injected from the FunctionRegistry.
#
# This approach:
# - Eliminates code duplication (each function defined once)
# - Ensures consistency across Expression, F, Accessor, and ColumnExpr
# - Makes it easy to add new functions
# - Enables future SQL/Pandas dual engine support


def _inject_expression_methods():
    """Inject function methods from registry into Expression class."""
    from .function_registry import FunctionRegistry
    from . import function_definitions  # noqa: F401 - ensures registration

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.all_specs():
        # Skip accessor-only functions (like dt.year, dt.month)
        if spec.accessor_only:
            continue

        # Skip if method already exists (don't override special implementations)
        if hasattr(Expression, spec.name):
            continue

        # Create method that delegates to the registry's sql_builder
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        # Set the method on Expression class
        setattr(Expression, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(Expression, alias_name):
                setattr(Expression, alias_name, getattr(Expression, spec.name))


# Perform injection when module is loaded
_inject_expression_methods()
