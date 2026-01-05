"""
Exception classes for DataStore
"""

__all__ = [
    'DataStoreError',
    'ConnectionError',
    'SchemaError',
    'QueryError',
    'ExecutionError',
    'ValidationError',
    'UnsupportedOperationError',
    'ImmutableError',
    'ColumnNotFoundError',
]


class DataStoreError(Exception):
    """Base exception for all DataStore errors."""

    pass


class ConnectionError(DataStoreError):
    """Raised when connection to data source fails."""

    pass


class SchemaError(DataStoreError):
    """Raised when schema-related operations fail."""

    pass


class QueryError(DataStoreError):
    """Raised when query construction fails."""

    pass


class ExecutionError(DataStoreError):
    """Raised when query execution fails."""

    pass


class ValidationError(DataStoreError):
    """Raised when data validation fails."""

    pass


class UnsupportedOperationError(DataStoreError):
    """Raised when an operation is not supported by DataStore.

    Provides clear error messages with operation name, reason, and optional suggestions.

    Example:
        raise UnsupportedOperationError(
            operation="str.split(expand=True)",
            reason="SQL does not support splitting strings into multiple columns",
            suggestion="Use pandas directly: df['col'].str.split(expand=True)"
        )
    """

    def __init__(self, operation: str, reason: str, suggestion: str = None):
        self.operation = operation
        self.reason = reason
        self.suggestion = suggestion

        msg = f"Operation '{operation}' is not supported: {reason}"
        if suggestion:
            msg += f"\nSuggestion: {suggestion}"
        super().__init__(msg)


class ImmutableError(DataStoreError, ValueError):
    """Raised when attempting to modify an immutable DataStore or ColumnExpr.

    DataStore and ColumnExpr are designed to be immutable. Operations that would
    modify them in-place should return a new instance instead.

    Example:
        raise ImmutableError(
            object_type="DataStore",
            operation="sort_values",
            suggestion="result = ds.sort_values('col') instead of ds.sort_values('col', inplace=True)"
        )
    """

    def __init__(self, object_type: str, operation: str = None, suggestion: str = None):
        self.object_type = object_type
        self.operation = operation
        self.suggestion = suggestion

        msg = f"{object_type} is immutable"
        if operation:
            msg += f", inplace modification via '{operation}' is not supported"
        else:
            msg += ", inplace=True is not supported"

        if suggestion:
            msg += f". {suggestion}"
        else:
            msg += ". All operations return a new instance."

        super().__init__(msg)


class ColumnNotFoundError(DataStoreError):
    """Raised when a referenced column does not exist in the DataFrame.

    Provides the column name and optionally lists available columns.

    Example:
        raise ColumnNotFoundError(
            column="nonexistent_col",
            available_columns=["a", "b", "c"]
        )
    """

    def __init__(self, column: str, available_columns: list = None):
        self.column = column
        self.available_columns = available_columns

        msg = f"Column '{column}' not found"
        if available_columns:
            if len(available_columns) <= 10:
                cols_str = ", ".join(repr(c) for c in available_columns)
                msg += f". Available columns: [{cols_str}]"
            else:
                cols_str = ", ".join(repr(c) for c in available_columns[:10])
                msg += f". Available columns (first 10 of {len(available_columns)}): [{cols_str}, ...]"
        super().__init__(msg)
