"""
Function Method Generator - Automatically inject methods from FunctionRegistry.

This module provides utilities to dynamically generate methods for:
- Expression class (instance methods like expr.upper())
- F class (static methods like F.upper(expr))
- Accessor classes (methods like expr.str.upper())
- ColumnExpr class (wrapper methods)

This eliminates code duplication - functions are defined once in function_definitions.py
and automatically made available through all interfaces.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, Callable, List, Optional, Type

from .function_registry import (
    FunctionRegistry,
    FunctionSpec,
    FunctionCategory,
)

if TYPE_CHECKING:
    from .expressions import Expression

__all__ = [
    'generate_expression_method',
    'generate_static_method',
    'generate_accessor_method',
    'generate_column_expr_method',
    'inject_methods_to_expression',
    'inject_methods_to_f_class',
    'inject_methods_to_accessor',
    'inject_methods_to_column_expr',
]


def _get_signature_params(spec: FunctionSpec) -> str:
    """Extract parameter names from the sql_builder signature."""
    if spec.sql_builder is None:
        return ''

    sig = inspect.signature(spec.sql_builder)
    params = list(sig.parameters.keys())
    # Remove 'expr' or first positional arg for instance methods
    # Keep all params for static methods
    return params


def generate_expression_method(spec: FunctionSpec) -> Callable:
    """
    Generate an instance method for Expression class.

    The generated method will call the sql_builder with self as the first argument.

    Example:
        # For spec with name='upper', generates:
        def upper(self, alias=None):
            return _build_upper(self, alias=alias)
    """
    builder = spec.sql_builder
    if builder is None:
        raise ValueError(f"Function '{spec.name}' has no sql_builder")

    # Get the signature to understand parameters
    sig = inspect.signature(builder)
    params = list(sig.parameters.keys())

    # Check if first param is 'expr' (most functions) or no expr (like row_number)
    needs_expr = len(params) > 0 and params[0] in ('expr', 'self', 'json', 'arr', 'a', 'base', 'condition')

    if needs_expr:
        # Instance method: self is passed as first arg
        @wraps(builder)
        def method(self, *args, **kwargs):
            return builder(self, *args, **kwargs)

    else:
        # No expr needed (e.g., row_number, now, today)
        @wraps(builder)
        def method(self, *args, **kwargs):
            return builder(*args, **kwargs)

    method.__name__ = spec.name
    method.__doc__ = spec.doc
    return method


def generate_static_method(spec: FunctionSpec) -> Callable:
    """
    Generate a static method for F class.

    The generated method will wrap the first argument as Expression and call sql_builder.

    Example:
        # For spec with name='upper', generates:
        @staticmethod
        def upper(expr, alias=None):
            from .expressions import Expression
            return _build_upper(Expression.wrap(expr), alias=alias)
    """
    builder = spec.sql_builder
    if builder is None:
        raise ValueError(f"Function '{spec.name}' has no sql_builder")

    # Get the signature
    sig = inspect.signature(builder)
    params = list(sig.parameters.keys())

    # Check if needs expr wrapping
    needs_expr = len(params) > 0 and params[0] in ('expr', 'json', 'arr', 'a', 'base', 'condition')

    if needs_expr:

        @wraps(builder)
        def static_method(expr, *args, **kwargs):
            from .expressions import Expression

            return builder(Expression.wrap(expr), *args, **kwargs)

    else:
        # No expr needed (e.g., row_number, now, today)
        @wraps(builder)
        def static_method(*args, **kwargs):
            return builder(*args, **kwargs)

    static_method.__name__ = spec.name
    static_method.__doc__ = spec.doc
    return staticmethod(static_method)


def generate_accessor_method(spec: FunctionSpec) -> Callable:
    """
    Generate a method for Accessor class (StringAccessor, DateTimeAccessor, etc.).

    The generated method will use self._expr as the expression argument.

    Example:
        # For spec with name='upper', generates:
        def upper(self, alias=None):
            return _build_upper(self._expr, alias=alias)
    """
    builder = spec.sql_builder
    if builder is None:
        raise ValueError(f"Function '{spec.name}' has no sql_builder")

    @wraps(builder)
    def accessor_method(self, *args, **kwargs):
        return builder(self._expr, *args, **kwargs)

    accessor_method.__name__ = spec.name
    accessor_method.__doc__ = spec.doc
    return accessor_method


def generate_column_expr_method(spec: FunctionSpec) -> Callable:
    """
    Generate a method for ColumnExpr class.

    The generated method wraps the result in ColumnExpr for chaining.

    Supports two modes:
    1. Expression mode (_expr is not None): Apply function to expression directly
    2. Method mode (_expr is None): Create a chained method-mode ColumnExpr

    Example:
        # For spec with name='upper', generates:
        def upper(self, alias=None):
            if self._expr is not None:
                new_expr = self._expr.upper(alias=alias)
                return ColumnExpr(new_expr, self._datastore)
            else:
                # Method mode: chain as another method call
                return ColumnExpr(source=self, method_name='upper', ...)
    """
    builder = spec.sql_builder
    func_name = spec.name
    if builder is None:
        raise ValueError(f"Function '{func_name}' has no sql_builder")

    @wraps(builder)
    def column_expr_method(self, *args, **kwargs):
        from .column_expr import ColumnExpr

        if self._expr is not None:
            # Expression mode: apply builder to expression
            result = builder(self._expr, *args, **kwargs)
            return ColumnExpr(result, self._datastore)
        else:
            # Method mode: chain as another method call on top of current source
            # This enables chaining like: ds['a'].fillna(0).abs()
            # Where fillna returns method-mode ColumnExpr, and abs() chains on it
            return ColumnExpr(
                source=self,
                method_name=f'_chain_{func_name}',  # Prefix to distinguish from pandas methods
                method_args=args,
                method_kwargs=kwargs,
            )

    column_expr_method.__name__ = func_name
    column_expr_method.__doc__ = spec.doc
    return column_expr_method


def inject_methods_to_expression(
    target_class: Type['Expression'],
    categories: Optional[List[FunctionCategory]] = None,
    exclude_accessor_only: bool = True,
) -> None:
    """
    Inject function methods to Expression class.

    Args:
        target_class: The Expression class to inject methods into
        categories: Optional list of categories to include (None = all)
        exclude_accessor_only: If True, skip functions marked as accessor_only
    """
    for spec in FunctionRegistry.all_specs():
        # Filter by category if specified
        if categories and spec.category not in categories:
            continue

        # Skip accessor-only functions
        if exclude_accessor_only and spec.accessor_only:
            continue

        # Skip if already exists (don't override existing implementations)
        if hasattr(target_class, spec.name):
            continue

        # Generate and set the method
        method = generate_expression_method(spec)
        setattr(target_class, spec.name, method)

        # Also set aliases
        for alias in spec.aliases:
            if not hasattr(target_class, alias):
                setattr(target_class, alias, method)


def inject_methods_to_f_class(
    target_class: Type,
    categories: Optional[List[FunctionCategory]] = None,
) -> None:
    """
    Inject static methods to F class.

    Args:
        target_class: The F class to inject methods into
        categories: Optional list of categories to include (None = all)
    """
    for spec in FunctionRegistry.all_specs():
        # Filter by category if specified
        if categories and spec.category not in categories:
            continue

        # Skip if already exists
        if hasattr(target_class, spec.name):
            continue

        # Generate and set the static method
        static_method = generate_static_method(spec)
        setattr(target_class, spec.name, static_method)

        # Also set aliases
        for alias in spec.aliases:
            if not hasattr(target_class, alias):
                setattr(target_class, alias, static_method)


def inject_methods_to_accessor(
    target_class: Type,
    category: FunctionCategory,
) -> None:
    """
    Inject methods to an Accessor class.

    Only injects functions matching the specified category.

    Args:
        target_class: The Accessor class (StringAccessor, DateTimeAccessor, etc.)
        category: The category to filter by
    """
    for spec in FunctionRegistry.get_by_category(category):
        # Skip if already exists
        if hasattr(target_class, spec.name):
            continue

        # Generate and set the method
        method = generate_accessor_method(spec)
        setattr(target_class, spec.name, method)

        # Also set aliases
        for alias in spec.aliases:
            if not hasattr(target_class, alias):
                setattr(target_class, alias, method)


def inject_methods_to_column_expr(
    target_class: Type,
    categories: Optional[List[FunctionCategory]] = None,
    exclude_accessor_only: bool = True,
) -> None:
    """
    Inject methods to ColumnExpr class.

    Args:
        target_class: The ColumnExpr class
        categories: Optional list of categories to include (None = all)
        exclude_accessor_only: If True, skip functions marked as accessor_only
    """
    for spec in FunctionRegistry.all_specs():
        # Filter by category if specified
        if categories and spec.category not in categories:
            continue

        # Skip accessor-only functions
        if exclude_accessor_only and spec.accessor_only:
            continue

        # Skip if already exists
        if hasattr(target_class, spec.name):
            continue

        # Generate and set the method
        method = generate_column_expr_method(spec)
        setattr(target_class, spec.name, method)

        # Also set aliases
        for alias in spec.aliases:
            if not hasattr(target_class, alias):
                setattr(target_class, alias, method)


# =============================================================================
# CLASS DECORATORS - Alternative way to inject methods
# =============================================================================


def with_registry_methods(
    categories: Optional[List[FunctionCategory]] = None,
    method_type: str = 'instance',
):
    """
    Class decorator to inject methods from FunctionRegistry.

    Args:
        categories: Categories to include (None = all)
        method_type: 'instance', 'static', 'accessor', or 'column_expr'

    Example:
        @with_registry_methods(categories=[FunctionCategory.STRING])
        class StringAccessor(BaseAccessor):
            pass
    """

    def decorator(cls):
        if method_type == 'instance':
            inject_methods_to_expression(cls, categories)
        elif method_type == 'static':
            inject_methods_to_f_class(cls, categories)
        elif method_type == 'accessor':
            if categories and len(categories) == 1:
                inject_methods_to_accessor(cls, categories[0])
            else:
                for cat in categories or list(FunctionCategory):
                    inject_methods_to_accessor(cls, cat)
        elif method_type == 'column_expr':
            inject_methods_to_column_expr(cls, categories)
        return cls

    return decorator


# =============================================================================
# INITIALIZATION HELPER
# =============================================================================


def initialize_all_methods():
    """
    Initialize all methods from the registry into their respective classes.

    This should be called after all modules are loaded to avoid circular imports.
    Call this in __init__.py or when first accessing the classes.
    """
    # Import here to avoid circular imports
    from . import function_definitions  # noqa: F401 - ensures functions are registered

    # Note: Actual injection is done lazily or explicitly
    # This function just ensures the definitions are loaded
    function_definitions.ensure_functions_registered()
