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

