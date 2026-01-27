"""
Condition system for DataStore (WHERE clause expressions)
"""

from typing import Optional, List, Iterator
from copy import copy

from .expressions import Expression, Node
from .utils import format_alias
from .exceptions import ValidationError

__all__ = [
    "Condition",
    "BinaryCondition",
    "CompoundCondition",
    "UnaryCondition",
    "InCondition",
    "BetweenCondition",
    "LikeCondition",
]


class Condition(Expression):
    """Base class for all conditions (used in WHERE, HAVING clauses)."""

    def _convert_to_condition(self, other):
        """Convert ColumnExpr to condition if needed."""
        from .column_expr import ColumnExpr
        from .expressions import Literal

        if isinstance(other, ColumnExpr):
            # Convert boolean ColumnExpr (like isNull()) to condition: expr = 1
            return BinaryCondition("=", other._expr, Literal(1))
        return other

    def __and__(self, other: "Condition") -> "CompoundCondition":
        """Combine conditions with AND."""
        return CompoundCondition("AND", self, self._convert_to_condition(other))

    def __or__(self, other: "Condition") -> "CompoundCondition":
        """Combine conditions with OR."""
        return CompoundCondition("OR", self, self._convert_to_condition(other))

    def __xor__(self, other: "Condition") -> "CompoundCondition":
        """Combine conditions with XOR."""
        return CompoundCondition("XOR", self, self._convert_to_condition(other))

    def __invert__(self) -> "NotCondition":
        """Negate condition with NOT."""
        return NotCondition(self)

    @staticmethod
    def all(conditions: List["Condition"]) -> Optional["Condition"]:
        """Combine multiple conditions with AND."""
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]

        result = conditions[0]
        for cond in conditions[1:]:
            result = result & cond
        return result

    @staticmethod
    def any(conditions: List["Condition"]) -> Optional["Condition"]:
        """Combine multiple conditions with OR."""
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]

        result = conditions[0]
        for cond in conditions[1:]:
            result = result | cond
        return result


class BinaryCondition(Condition):
    """
    Binary comparison condition (e.g., a = b, x > 5).

    Example:
        >>> BinaryCondition('=', Field('age'), Literal(18))
        >>> Field('price') > 100  # Uses operator overloading
    """

    OPERATORS = {"=", "!=", "<>", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "IS"}

    def __init__(
        self,
        operator: str,
        left: Expression,
        right: Expression,
        alias: Optional[str] = None,
    ):
        super().__init__(alias)

        if operator.upper() not in self.OPERATORS:
            raise ValidationError(f"Invalid comparison operator: {operator}")

        self.operator = operator.upper()
        self.left = left
        self.right = right

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.left.nodes()
        yield from self.right.nodes()

    # Operators that need NULL-safe wrapping for pandas compatibility
    # Used by NullSafeCondition wrapper class
    # pandas follows IEEE 754: comparisons with NaN/NULL return bool, not NULL
    # - For =, >, >=, <, <=: NULL comparison returns False in pandas
    # - For !=, <>: NULL comparison returns True in pandas
    _NULL_TO_FALSE_OPS = {"=", ">", ">=", "<", "<="}
    _NULL_TO_TRUE_OPS = {"!=", "<>"}

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for binary condition."""
        left_sql = self.left.to_sql(quote_char=quote_char, **kwargs)
        right_sql = self.right.to_sql(quote_char=quote_char, **kwargs)

        sql = f"{left_sql} {self.operator} {right_sql}"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return BinaryCondition(
            self.operator, copy(self.left), copy(self.right), self.alias
        )


class NullSafeCondition(Condition):
    """
    Wrapper that applies pandas-style NULL semantics to comparisons.

    This wraps a BinaryCondition or CompoundCondition and generates SQL that
    matches pandas' NULL/NaN comparison behavior using ifNull().

    pandas follows IEEE 754:
    - =, >, >=, <, <=: comparisons with NULL return False
    - !=, <>: comparisons with NULL return True

    Example:
        >>> NullSafeCondition(Field('a') > Field('b'))
        >>> # Generates: ifNull("a" > "b", 0)
    """

    def __init__(self, condition: Condition, alias: Optional[str] = None):
        super().__init__(alias)
        self.condition = condition

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.condition.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL with ifNull wrapping for pandas NULL semantics."""
        # Get the inner condition SQL
        inner_sql = self.condition.to_sql(quote_char=quote_char, **kwargs)

        # Determine the default value based on the operator
        default_value = self._get_null_default(self.condition)

        if default_value is not None:
            sql = f"ifNull({inner_sql}, {default_value})"
        else:
            sql = inner_sql

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def _get_null_default(self, condition: Condition) -> Optional[int]:
        """
        Get the default value for NULL based on the condition type.

        Returns:
            0 for =, >, >=, <, <= (NULL -> False)
            1 for !=, <> (NULL -> True)
            None if no wrapping needed
        """
        if isinstance(condition, BinaryCondition):
            if condition.operator in BinaryCondition._NULL_TO_FALSE_OPS:
                return 0
            elif condition.operator in BinaryCondition._NULL_TO_TRUE_OPS:
                return 1
        elif isinstance(condition, CompoundCondition):
            # For compound conditions (AND, OR), apply recursively
            # The result of AND/OR with NULL should be 0 (False) in pandas semantics
            return 0
        return None

    def __copy__(self):
        return NullSafeCondition(copy(self.condition), self.alias)


class CompoundCondition(Condition):
    """
    Compound condition combining multiple conditions with AND/OR.

    Example:
        >>> CompoundCondition('AND', cond1, cond2)
        >>> (Field('age') > 18) & (Field('city') == 'NYC')
    """

    OPERATORS = {"AND", "OR", "XOR"}

    def __init__(
        self,
        operator: str,
        left: Condition,
        right: Condition,
        alias: Optional[str] = None,
    ):
        super().__init__(alias)

        if operator.upper() not in self.OPERATORS:
            raise ValidationError(f"Invalid logical operator: {operator}")

        self.operator = operator.upper()
        self.left = left
        self.right = right

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.left.nodes()
        yield from self.right.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for compound condition."""
        left_sql = self.left.to_sql(quote_char=quote_char, **kwargs)
        right_sql = self.right.to_sql(quote_char=quote_char, **kwargs)

        # Wrap in parentheses for clarity
        sql = f"({left_sql} {self.operator} {right_sql})"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return CompoundCondition(
            self.operator, copy(self.left), copy(self.right), self.alias
        )


class NotCondition(Condition):
    """
    NOT condition (negation).

    Example:
        >>> NotCondition(Field('active') == True)
        >>> ~(Field('age') > 18)
    """

    def __init__(self, condition: Condition, alias: Optional[str] = None):
        super().__init__(alias)
        self.condition = condition

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.condition.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for NOT condition with pandas-style NULL semantics.

        In pandas, comparisons with NaN return False, and NOT False = True.
        In SQL, comparisons with NULL return NULL, and NOT NULL = NULL.

        To match pandas behavior, we need:
        - NOT (ifNull(condition, 0)) - treats NULL as 0 (False), so NOT gives 1 (True)
        - OR: ifNull(NOT (condition), 1) - if NOT returns NULL, default to 1 (True)

        We use the second form as it's cleaner for nested conditions.
        """
        cond_sql = self.condition.to_sql(quote_char=quote_char, **kwargs)
        # Wrap with ifNull to handle NULL comparisons like pandas
        # NOT NULL -> NULL in SQL, but should be True in pandas
        sql = f"ifNull(NOT ({cond_sql}), 1)"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return NotCondition(copy(self.condition), self.alias)


class UnaryCondition(Condition):
    """
    Unary condition (e.g., IS NULL, IS NOT NULL).

    Example:
        >>> Field('name').isnull()
        >>> # Generates: "name" IS NULL
    """

    OPERATORS = {"IS NULL", "IS NOT NULL"}

    def __init__(
        self, operator: str, expression: Expression, alias: Optional[str] = None
    ):
        super().__init__(alias)

        if operator.upper() not in self.OPERATORS:
            raise ValidationError(f"Invalid unary operator: {operator}")

        self.operator = operator.upper()
        self.expression = expression

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.expression.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for unary condition."""
        expr_sql = self.expression.to_sql(quote_char=quote_char, **kwargs)
        sql = f"{expr_sql} {self.operator}"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return UnaryCondition(self.operator, copy(self.expression), self.alias)


class InCondition(Condition):
    """
    IN condition (e.g., field IN (1, 2, 3)).

    Example:
        >>> Field('id').isin([1, 2, 3])
        >>> # Generates: "id" IN (1,2,3)
    """

    def __init__(
        self,
        expression: Expression,
        values,
        negate: bool = False,
        alias: Optional[str] = None,
    ):
        super().__init__(alias)
        self.expression = expression
        # Check if values is a subquery (DataStore) - don't convert to list
        if hasattr(values, "to_sql") and hasattr(values, "table_name"):
            self.values = values  # Subquery
        else:
            self.values = values if isinstance(values, (list, tuple)) else [values]
        self.negate = negate

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.expression.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for IN condition."""
        from .expressions import Literal

        expr_sql = self.expression.to_sql(quote_char=quote_char, **kwargs)

        # Check if values is a subquery (DataStore object)
        # We need to avoid circular import, so check for method existence
        if hasattr(self.values, "to_sql") and hasattr(self.values, "table_name"):
            # This is a DataStore subquery
            values_sql = self.values.to_sql(quote_char=quote_char)
            operator = "NOT IN" if self.negate else "IN"
            sql = f"{expr_sql} {operator} ({values_sql})"
        else:
            # Convert values to SQL
            value_sqls = []
            for val in self.values:
                if isinstance(val, Expression):
                    value_sqls.append(val.to_sql(quote_char=quote_char, **kwargs))
                else:
                    value_sqls.append(
                        Literal(val).to_sql(quote_char=quote_char, **kwargs)
                    )

            values_sql = ",".join(value_sqls)
            operator = "NOT IN" if self.negate else "IN"
            sql = f"{expr_sql} {operator} ({values_sql})"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return InCondition(copy(self.expression), self.values, self.negate, self.alias)


class BetweenCondition(Condition):
    """
    BETWEEN condition (e.g., field BETWEEN 1 AND 10).

    Example:
        >>> Field('age').between(18, 65)
        >>> # Generates: "age" BETWEEN 18 AND 65
        >>> Field('age').between(18, 65, inclusive='neither')
        >>> # Generates: "age" > 18 AND "age" < 65
    """

    def __init__(
        self,
        expression: Expression,
        lower: Expression,
        upper: Expression,
        inclusive: str = "both",
        alias: Optional[str] = None,
    ):
        super().__init__(alias)
        self.expression = expression
        self.lower = lower
        self.upper = upper
        self.inclusive = inclusive

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.expression.nodes()
        yield from self.lower.nodes()
        yield from self.upper.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for BETWEEN condition."""
        expr_sql = self.expression.to_sql(quote_char=quote_char, **kwargs)
        lower_sql = self.lower.to_sql(quote_char=quote_char, **kwargs)
        upper_sql = self.upper.to_sql(quote_char=quote_char, **kwargs)

        # Handle different inclusive modes
        if self.inclusive == "both":
            sql = f"{expr_sql} BETWEEN {lower_sql} AND {upper_sql}"
        elif self.inclusive == "neither":
            sql = f"({expr_sql} > {lower_sql} AND {expr_sql} < {upper_sql})"
        elif self.inclusive == "left":
            sql = f"({expr_sql} >= {lower_sql} AND {expr_sql} < {upper_sql})"
        elif self.inclusive == "right":
            sql = f"({expr_sql} > {lower_sql} AND {expr_sql} <= {upper_sql})"
        else:
            raise ValueError(
                f"Invalid inclusive value: {self.inclusive}. "
                f"Valid options are: 'both', 'neither', 'left', 'right'"
            )

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return BetweenCondition(
            copy(self.expression),
            copy(self.lower),
            copy(self.upper),
            self.inclusive,
            self.alias,
        )


class LikeCondition(Condition):
    """
    LIKE condition for pattern matching.

    Example:
        >>> Field('name').like('John%')
        >>> # Generates: "name" LIKE 'John%'
    """

    def __init__(
        self,
        expression: Expression,
        pattern: str,
        negate: bool = False,
        case_sensitive: bool = True,
        alias: Optional[str] = None,
    ):
        super().__init__(alias)
        self.expression = expression
        self.pattern = pattern
        self.negate = negate
        self.case_sensitive = case_sensitive

    def nodes(self) -> Iterator[Node]:
        """Traverse expression tree."""
        yield self
        yield from self.expression.nodes()

    def to_sql(self, quote_char: str = '"', **kwargs) -> str:
        """Generate SQL for LIKE condition."""
        from .expressions import Literal

        expr_sql = self.expression.to_sql(quote_char=quote_char, **kwargs)
        pattern_sql = Literal(self.pattern).to_sql(quote_char=quote_char, **kwargs)

        # Determine operator
        if self.case_sensitive:
            operator = "NOT LIKE" if self.negate else "LIKE"
        else:
            operator = "NOT ILIKE" if self.negate else "ILIKE"

        sql = f"{expr_sql} {operator} {pattern_sql}"

        # Add alias if present and requested
        if kwargs.get("with_alias", False) and self.alias:
            return format_alias(sql, self.alias, quote_char)

        return sql

    def __copy__(self):
        return LikeCondition(
            copy(self.expression),
            self.pattern,
            self.negate,
            self.case_sensitive,
            self.alias,
        )
