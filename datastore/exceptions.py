"""
Exception classes for DataStore
"""

import re

__all__ = [
    "DataStoreError",
    "ConnectionError",
    "SchemaError",
    "QueryError",
    "ExecutionError",
    "ValidationError",
    "UnsupportedOperationError",
    "ImmutableError",
    "ColumnNotFoundError",
    "translate_remote_error",
    "_extract_clickhouse_error_code",
    "_extract_host_from_error",
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



def _extract_clickhouse_error_code(error_str: str) -> str:
    """Extract the primary ClickHouse error code tag from an error message.

    ClickHouse errors contain tags like (AUTHENTICATION_FAILED), (UNKNOWN_DATABASE), etc.
    For nested errors (e.g. NO_REMOTE_SHARD_AVAILABLE wrapping DNS_ERROR),
    returns the innermost/root-cause code.

    Returns empty string if no code found.
    """
    codes = re.findall(r'\(([A-Z_]+)\)', error_str)
    if not codes:
        return ""
    # Priority order: return the most specific (innermost) error
    priority = [
        "AUTHENTICATION_FAILED",
        "UNKNOWN_USER",
        "UNKNOWN_DATABASE",
        "UNKNOWN_TABLE",
        "ACCESS_DENIED",
        "DNS_ERROR",
        "NETWORK_ERROR",
        "SOCKET_TIMEOUT",
        "SYNTAX_ERROR",
    ]
    for code in priority:
        if code in codes:
            return code
    return codes[0]


def _extract_host_from_error(error_str: str) -> str:
    """Try to extract the host from a ClickHouse error message."""
    # Pattern: "Not found address of host: xxx: (xxx:port)"
    m = re.search(r'Not found address of host:\s*([^:]+):', error_str)
    if m:
        return m.group(1).strip()
    # Pattern: "Connection refused (host:port)"
    m = re.search(r'Connection refused \(([^)]+)\)', error_str)
    if m:
        return m.group(1).strip()
    return ""


def translate_remote_error(error: Exception, context: dict = None) -> str:
    """Translate a raw chdb/ClickHouse error into a user-friendly message.

    Args:
        error: The original exception
        context: Optional dict with keys like 'host', 'user', 'database', 'table'

    Returns:
        A user-friendly error message string with actionable suggestions.
    """
    error_str = str(error)
    ctx = context or {}
    host = ctx.get("host", "")
    user = ctx.get("user", "")
    database = ctx.get("database", "")
    table = ctx.get("table", "")

    code = _extract_clickhouse_error_code(error_str)

    # --- Authentication errors ---
    if code in ("AUTHENTICATION_FAILED", "UNKNOWN_USER"):
        msg = "Authentication failed"
        if user:
            msg += f" for user '{user}'"
        if host:
            msg += f" on '{host}'"
        msg += (
            ". Please check your username and password."
            "\nHint: Verify credentials with: "
            "clickhouse-client --host <host> --port <port> --user <user> --password <password>"
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- Unknown database ---
    if code == "UNKNOWN_DATABASE":
        db_name = database
        m = re.search(r'Database\s+(\S+)\s+does(?:n.t| not)\s+exist', error_str, re.IGNORECASE)
        if m:
            db_name = m.group(1).strip("`'\"")
        msg = f"Database '{db_name}' does not exist"
        if host:
            msg += f" on '{host}'"
        msg += "."
        # Hint about listing databases
        msg += (
            "\nHint: Use ds.databases() to list available databases, "
            "or check the database name for typos."
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- Unknown table ---
    if code == "UNKNOWN_TABLE":
        tbl_name = table
        m = re.search(r'Table\s+(\S+)\s+does(?:n.t| not)\s+exist', error_str, re.IGNORECASE)
        if m:
            tbl_name = m.group(1).strip("`'\"")
        msg = f"Table '{tbl_name}' does not exist"
        if database:
            msg += f" in database '{database}'"
        if host:
            msg += f" on '{host}'"
        msg += "."
        msg += (
            "\nHint: Use ds.tables('database_name') to list available tables, "
            "or check the table name for typos."
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- Access denied / permissions ---
    if code == "ACCESS_DENIED" or "access denied" in error_str.lower() or "not enough privileges" in error_str.lower():
        msg = "Access denied"
        if user:
            msg += f" for user '{user}'"
        if host:
            msg += f" on '{host}'"
        msg += "."
        msg += (
            "\nHint: Check that the user has the required privileges "
            "(SELECT, INSERT, etc.) on the target database/table. "
            "Use GRANT statements or contact your database administrator."
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- DNS resolution failure ---
    if code == "DNS_ERROR" or "not found address of host" in error_str.lower():
        resolved_host = _extract_host_from_error(error_str) or host
        msg = f"Cannot resolve hostname '{resolved_host}'."
        msg += (
            "\nHint: Check the host address for typos. "
            "Expected format: 'hostname:port' (e.g., 'localhost:9000' or 'my-server.example.com:9440')."
            "\nAlso verify DNS resolution and network connectivity."
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- Connection refused ---
    if code == "NETWORK_ERROR" or "connection refused" in error_str.lower():
        resolved_host = _extract_host_from_error(error_str) or host
        msg = f"Connection refused to '{resolved_host}'."
        msg += (
            "\nHint: Check that:"
            "\n  1. The ClickHouse server is running"
            "\n  2. The host and port are correct (default native port: 9000, secure: 9440)"
            "\n  3. No firewall is blocking the connection"
            "\n  4. The server is accepting remote connections"
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- Socket timeout ---
    if code == "SOCKET_TIMEOUT" or "timed out" in error_str.lower() or "timeout" in error_str.lower():
        msg = "Connection timed out"
        if host:
            msg += f" to '{host}'"
        msg += "."
        msg += (
            "\nHint: Check network connectivity and firewall settings. "
            "The server may be overloaded or unreachable. "
            "Consider increasing the connection timeout if the server is slow to respond."
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- SQL syntax error ---
    if code == "SYNTAX_ERROR":
        msg = "SQL syntax error in your query."
        # Try to extract position info
        m = re.search(r'failed at position (\d+)', error_str)
        if m:
            msg += f" Error at position {m.group(1)}."
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- No remote shard available (wrapper for connection errors) ---
    if code == "NO_REMOTE_SHARD_AVAILABLE" or code == "ALL_CONNECTION_TRIES_FAILED":
        # Try to find the root cause in nested errors
        if "DNS_ERROR" in error_str or "not found address of host" in error_str.lower():
            return translate_remote_error(error, {**ctx, "_code_override": "DNS_ERROR"})
        if "NETWORK_ERROR" in error_str or "connection refused" in error_str.lower():
            return translate_remote_error(error, {**ctx, "_code_override": "NETWORK_ERROR"})
        if "SOCKET_TIMEOUT" in error_str or "timed out" in error_str.lower():
            return translate_remote_error(error, {**ctx, "_code_override": "SOCKET_TIMEOUT"})
        if "AUTHENTICATION_FAILED" in error_str:
            return translate_remote_error(error, {**ctx, "_code_override": "AUTHENTICATION_FAILED"})

        # Generic connection failure
        msg = "Failed to connect to remote server"
        if host:
            msg += f" '{host}'"
        msg += ". All connection attempts failed."
        msg += (
            "\nHint: Check that the server is running and accessible, "
            "and that the host:port is correct."
        )
        msg += f"\n\nOriginal error: {error_str}"
        return msg

    # --- Fallback: return original error unchanged ---
    return error_str
