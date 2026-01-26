"""
Function Registry - Single Source of Truth for all function definitions.

This module provides a centralized registry for all functions (scalar, aggregate, window).
Each function is defined once and automatically generates methods for:
- Expression class (instance methods)
- F class (static methods)
- Accessor classes (DateTimeAccessor, StringAccessor, etc.)
- ColumnExpr class (wrapper methods)

Benefits:
- No duplicate logic across files
- Easy to add new functions
- Support for function aliases
- Type classification (scalar, aggregate, window)
- Category classification (datetime, string, math, etc.)
- Future: configurable SQL/Pandas dual engine support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .functions import Function

__all__ = [
    'FunctionType',
    'FunctionCategory',
    'FunctionSpec',
    'FunctionRegistry',
    'register_function',
]


class FunctionType(Enum):
    """
    Function type classification.

    - SCALAR: Regular functions that operate row-by-row (e.g., upper, toDate)
    - AGGREGATE: Functions that combine multiple rows (e.g., sum, avg, count)
    - WINDOW: Functions that operate over a window/partition (e.g., row_number, rank)
    - TABLE: Table-valued functions (e.g., file, url, s3)
    """

    SCALAR = auto()
    AGGREGATE = auto()
    WINDOW = auto()
    TABLE = auto()


class FunctionCategory(Enum):
    """
    Function category for organization and accessor routing.

    Functions in specific categories can be accessed via accessors:
    - STRING -> .str accessor
    - DATETIME -> .dt accessor
    - ARRAY -> .arr accessor (future)
    """

    STRING = "string"
    DATETIME = "datetime"
    ARRAY = "array"
    MATH = "math"
    TYPE_CONVERSION = "type"
    CONDITIONAL = "conditional"
    HASH = "hash"
    JSON = "json"
    URL = "url"
    IP = "ip"
    UUID = "uuid"
    GEO = "geo"
    ENCODING = "encoding"
    AGGREGATE = "aggregate"
    WINDOW = "window"
    OTHER = "other"


@dataclass
class FunctionSpec:
    """
    Function specification - complete definition of a function.

    This is the Single Source of Truth for each function. All method
    generation and routing is based on this specification.

    Attributes:
        name: Primary function name in snake_case (e.g., 'to_datetime')
        clickhouse_name: ClickHouse function name (e.g., 'toDateTime')
        func_type: Type classification (SCALAR, AGGREGATE, WINDOW, TABLE)
        category: Category for accessor routing
        aliases: Alternative names for this function
        sql_builder: Callable that builds the SQL expression
        pandas_impl: Optional callable for pandas-based execution
        doc: Documentation string
        signature: Parameter signature as {name: (type, default)}
        min_args: Minimum number of arguments (excluding self/expr)
        max_args: Maximum number of arguments (-1 for unlimited)
        accessor_only: If True, only available via accessor (not on Expression)
        supports_over: For window functions, whether OVER clause is required
    """

    # Basic info
    name: str
    clickhouse_name: str

    # Classification
    func_type: FunctionType = FunctionType.SCALAR
    category: FunctionCategory = FunctionCategory.OTHER

    # Aliases
    aliases: List[str] = field(default_factory=list)

    # Implementation
    sql_builder: Optional[Callable] = None
    pandas_impl: Optional[Callable] = None

    # Metadata
    doc: str = ""
    signature: Dict[str, Tuple[type, Any]] = field(default_factory=dict)
    min_args: int = 0
    max_args: int = -1  # -1 means unlimited

    # Behavior flags
    accessor_only: bool = False  # Only available via accessor, not Expression
    supports_over: bool = False  # Window function requires OVER clause

    @property
    def is_aggregate(self) -> bool:
        """Check if this is an aggregate function."""
        return self.func_type == FunctionType.AGGREGATE

    @property
    def is_window(self) -> bool:
        """Check if this is a window function."""
        return self.func_type == FunctionType.WINDOW

    @property
    def is_scalar(self) -> bool:
        """Check if this is a scalar function."""
        return self.func_type == FunctionType.SCALAR

    @property
    def all_names(self) -> Set[str]:
        """Get all names including aliases."""
        return {self.name} | set(self.aliases)

    def build(self, *args, **kwargs) -> 'Function':
        """
        Build a function expression using the sql_builder.

        Args:
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments (e.g., alias)

        Returns:
            Function expression
        """
        if self.sql_builder is None:
            raise ValueError(f"Function '{self.name}' has no sql_builder defined")
        return self.sql_builder(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FunctionSpec(name='{self.name}', "
            f"ch='{self.clickhouse_name}', "
            f"type={self.func_type.name}, "
            f"category={self.category.name})"
        )


class FunctionRegistry:
    """
    Global function registry - singleton pattern.

    All function definitions are stored here and can be accessed by name or alias.

    Usage:
        # Register a function
        @FunctionRegistry.register(
            name='to_datetime',
            clickhouse_name='toDateTime',
            func_type=FunctionType.SCALAR,
            category=FunctionCategory.DATETIME,
            aliases=['toDateTime', 'as_datetime'],
        )
        def build_to_datetime(expr, timezone=None, alias=None):
            from .functions import Function
            from .expressions import Literal
            if timezone:
                return Function('toDateTime', expr, Literal(timezone), alias=alias)
            return Function('toDateTime', expr, alias=alias)

        # Get a function spec
        spec = FunctionRegistry.get('to_datetime')
        spec = FunctionRegistry.get('toDateTime')  # Same result via alias

        # Get functions by category
        datetime_funcs = FunctionRegistry.get_by_category(FunctionCategory.DATETIME)
    """

    _functions: Dict[str, FunctionSpec] = {}
    _alias_map: Dict[str, str] = {}  # alias -> canonical name
    _by_category: Dict[FunctionCategory, List[str]] = {}
    _by_type: Dict[FunctionType, List[str]] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        clickhouse_name: str,
        func_type: FunctionType = FunctionType.SCALAR,
        category: FunctionCategory = FunctionCategory.OTHER,
        aliases: Optional[List[str]] = None,
        doc: str = "",
        signature: Optional[Dict[str, Tuple[type, Any]]] = None,
        pandas_impl: Optional[Callable] = None,
        min_args: int = 0,
        max_args: int = -1,
        accessor_only: bool = False,
        supports_over: bool = False,
    ) -> Callable:
        """
        Decorator to register a function builder.

        The decorated function becomes the sql_builder for the FunctionSpec.

        Args:
            name: Primary function name (snake_case)
            clickhouse_name: ClickHouse function name
            func_type: Function type (SCALAR, AGGREGATE, WINDOW, TABLE)
            category: Function category for accessor routing
            aliases: List of alternative names
            doc: Documentation string
            signature: Parameter signature {name: (type, default)}
            pandas_impl: Optional pandas implementation
            min_args: Minimum arguments (default 0)
            max_args: Maximum arguments (-1 = unlimited)
            accessor_only: Only available via accessor
            supports_over: Window function requires OVER

        Returns:
            Decorator function

        Example:
            @FunctionRegistry.register(
                name='upper',
                clickhouse_name='upper',
                category=FunctionCategory.STRING,
                aliases=['uppercase', 'ucase'],
            )
            def build_upper(expr, alias=None):
                return Function('upper', expr, alias=alias)
        """

        def decorator(sql_builder: Callable) -> Callable:
            spec = FunctionSpec(
                name=name,
                clickhouse_name=clickhouse_name,
                func_type=func_type,
                category=category,
                aliases=aliases or [],
                sql_builder=sql_builder,
                pandas_impl=pandas_impl,
                doc=doc or sql_builder.__doc__ or "",
                signature=signature or {},
                min_args=min_args,
                max_args=max_args,
                accessor_only=accessor_only,
                supports_over=supports_over,
            )

            cls._register_spec(spec)
            return sql_builder

        return decorator

    @classmethod
    def register_spec(cls, spec: FunctionSpec) -> None:
        """
        Register a FunctionSpec directly (without decorator).

        Useful for programmatic registration.
        """
        cls._register_spec(spec)

    @classmethod
    def _register_spec(cls, spec: FunctionSpec) -> None:
        """Internal: register a spec and update all indexes."""
        # Register primary name
        cls._functions[spec.name] = spec

        # Register all aliases
        for alias in spec.aliases:
            if alias != spec.name:  # Don't map name to itself
                cls._alias_map[alias] = spec.name

        # Index by category
        if spec.category not in cls._by_category:
            cls._by_category[spec.category] = []
        if spec.name not in cls._by_category[spec.category]:
            cls._by_category[spec.category].append(spec.name)

        # Index by type
        if spec.func_type not in cls._by_type:
            cls._by_type[spec.func_type] = []
        if spec.name not in cls._by_type[spec.func_type]:
            cls._by_type[spec.func_type].append(spec.name)

    @classmethod
    def get(cls, name: str) -> Optional[FunctionSpec]:
        """
        Get function spec by name or alias.

        Args:
            name: Function name or alias

        Returns:
            FunctionSpec or None if not found
        """
        # Try direct lookup first
        if name in cls._functions:
            return cls._functions[name]

        # Try alias lookup
        if name in cls._alias_map:
            canonical = cls._alias_map[name]
            return cls._functions.get(canonical)

        return None

    @classmethod
    def get_or_raise(cls, name: str) -> FunctionSpec:
        """
        Get function spec by name or alias, raising if not found.

        Args:
            name: Function name or alias

        Returns:
            FunctionSpec

        Raises:
            KeyError: If function not found
        """
        spec = cls.get(name)
        if spec is None:
            raise KeyError(f"Function '{name}' not found in registry")
        return spec

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a function is registered."""
        return name in cls._functions or name in cls._alias_map

    @classmethod
    def get_by_category(cls, category: FunctionCategory) -> List[FunctionSpec]:
        """Get all functions in a category."""
        names = cls._by_category.get(category, [])
        return [cls._functions[n] for n in names]

    @classmethod
    def get_by_type(cls, func_type: FunctionType) -> List[FunctionSpec]:
        """Get all functions of a type."""
        names = cls._by_type.get(func_type, [])
        return [cls._functions[n] for n in names]

    @classmethod
    def get_aggregates(cls) -> List[FunctionSpec]:
        """Get all aggregate functions."""
        return cls.get_by_type(FunctionType.AGGREGATE)

    @classmethod
    def get_window_functions(cls) -> List[FunctionSpec]:
        """Get all window functions."""
        return cls.get_by_type(FunctionType.WINDOW)

    @classmethod
    def get_scalars(cls) -> List[FunctionSpec]:
        """Get all scalar functions."""
        return cls.get_by_type(FunctionType.SCALAR)

    @classmethod
    def all_names(cls) -> Set[str]:
        """Get all registered function names (including aliases)."""
        return set(cls._functions.keys()) | set(cls._alias_map.keys())

    @classmethod
    def all_specs(cls) -> List[FunctionSpec]:
        """Get all registered function specs."""
        return list(cls._functions.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered functions. Mainly for testing."""
        cls._functions.clear()
        cls._alias_map.clear()
        cls._by_category.clear()
        cls._by_type.clear()
        cls._initialized = False

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_functions': len(cls._functions),
            'total_aliases': len(cls._alias_map),
            'by_type': {t.name: len(names) for t, names in cls._by_type.items()},
            'by_category': {c.name: len(names) for c, names in cls._by_category.items()},
        }


# Convenience alias for the decorator
register_function = FunctionRegistry.register
