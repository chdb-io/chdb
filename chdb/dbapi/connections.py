from . import err
from .cursors import Cursor
from . import converters
from ..state import sqlitelike as chdb_stateful

DEBUG = False
VERBOSE = False


class Connection(object):
    """DB-API 2.0 compliant connection to chDB database.

    This class provides a standard DB-API interface for connecting to and interacting
    with chDB databases. It supports both in-memory and file-based databases.

    The connection manages the underlying chDB engine and provides methods for
    executing queries, managing transactions (no-op for ClickHouse), and creating cursors.

    Args:
        path (str, optional): Database file path. If None, uses in-memory database.
                              Can be a file path like 'database.db' or None for ':memory:'

    Attributes:
        encoding (str): Character encoding for queries, defaults to 'utf8'
        open (bool): True if connection is open, False if closed

    Examples:
        >>> # In-memory database
        >>> conn = Connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT 1")
        >>> result = cursor.fetchall()
        >>> conn.close()

        >>> # File-based database
        >>> conn = Connection('mydata.db')
        >>> with conn.cursor() as cur:
        ...     cur.execute("CREATE TABLE users (id INT, name STRING)")
        ...     cur.execute("INSERT INTO users VALUES (1, 'Alice')")
        >>> conn.close()

        >>> # Context manager usage
        >>> with Connection() as cur:
        ...     cur.execute("SELECT version()")
        ...     version = cur.fetchone()

    Note:
        ClickHouse does not support traditional transactions, so commit() and rollback()
        operations are no-ops but provided for DB-API compliance.
    """

    def __init__(self, path=None):
        """Initialize a new database connection.

        Args:
            path (str, optional): Database file path. None for in-memory database.

        Raises:
            err.Error: If connection cannot be established
        """
        self._closed = False
        self.encoding = "utf8"
        self._affected_rows = 0
        self._resp = None

        # Initialize sqlitelike connection
        connection_string = ":memory:" if path is None else f"file:{path}"
        self._conn = chdb_stateful.Connection(connection_string)

        # Test connection with a simple query
        cursor = self._conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()

    def close(self):
        """Close the database connection.

        Closes the underlying chDB connection and marks this connection as closed.
        Subsequent operations on this connection will raise an Error.

        Raises:
            err.Error: If connection is already closed
        """
        if self._closed:
            raise err.Error("Already closed")
        self._closed = True
        self._conn.close()

    @property
    def open(self):
        """Check if the connection is open.

        Returns:
            bool: True if connection is open, False if closed
        """
        return not self._closed

    def commit(self):
        """Commit the current transaction.

        Note:
            This is a no-op for chDB/ClickHouse as it doesn't support traditional
            transactions. Provided for DB-API 2.0 compliance.
        """
        # No-op for ClickHouse
        pass

    def rollback(self):
        """Roll back the current transaction.

        Note:
            This is a no-op for chDB/ClickHouse as it doesn't support traditional
            transactions. Provided for DB-API 2.0 compliance.
        """
        # No-op for ClickHouse
        pass

    def cursor(self, cursor=None):
        """Create a new cursor for executing queries.

        Args:
            cursor: Ignored, provided for compatibility

        Returns:
            Cursor: New cursor object for this connection

        Raises:
            err.Error: If connection is closed

        Example:
            >>> conn = Connection()
            >>> cur = conn.cursor()
            >>> cur.execute("SELECT 1")
            >>> result = cur.fetchone()
        """
        if self._closed:
            raise err.Error("Connection closed")
        if cursor:
            return Cursor(self)
        return Cursor(self)

    def query(self, sql, fmt="CSV"):
        """Execute a SQL query directly and return raw results.

        This method bypasses the cursor interface and executes queries directly.
        For standard DB-API usage, prefer using cursor() method.

        Args:
            sql (str or bytes): SQL query to execute
            fmt (str, optional): Output format. Defaults to "CSV".
                Supported formats include "CSV", "JSON", "Arrow", "Parquet", etc.

        Returns:
            Query result in the specified format

        Raises:
            err.InterfaceError: If connection is closed or query fails

        Example:
            >>> conn = Connection()
            >>> result = conn.query("SELECT 1, 'hello'", "CSV")
            >>> print(result)
            "1,hello\\n"
        """
        if self._closed:
            raise err.InterfaceError("Connection closed")

        if isinstance(sql, str):
            sql = sql.encode(self.encoding, "surrogateescape")

        try:
            result = self._conn.query(sql.decode(), fmt)
            self._resp = result
            return result
        except Exception as error:
            raise err.InterfaceError(f"Query error: {error}")

    def escape(self, obj, mapping=None):
        """Escape a value for safe inclusion in SQL queries.

        Args:
            obj: Value to escape (string, bytes, number, etc.)
            mapping: Optional character mapping for escaping

        Returns:
            Escaped version of the input suitable for SQL queries

        Example:
            >>> conn = Connection()
            >>> safe_value = conn.escape("O'Reilly")
            >>> query = f"SELECT * FROM users WHERE name = {safe_value}"
        """
        return converters.escape_item(obj, mapping)

    def escape_string(self, s):
        """Escape a string value for SQL queries.

        Args:
            s (str): String to escape

        Returns:
            str: Escaped string safe for SQL inclusion
        """
        return converters.escape_string(s)

    def _quote_bytes(self, s):
        """Quote and escape bytes data for SQL queries.

        Args:
            s (bytes): Bytes data to quote

        Returns:
            str: Quoted and escaped bytes representation
        """
        return converters.escape_bytes(s)

    def __enter__(self):
        """Enter context manager and return a cursor.

        Returns:
            Cursor: New cursor for this connection

        Example:
            >>> with Connection() as cur:
            ...     cur.execute("SELECT 1")
            ...     result = cur.fetchone()
        """
        return self.cursor()

    def __exit__(self, exc, value, traceback):
        """Exit context manager with proper cleanup.

        Commits on successful exit, rolls back on exception, and always closes connection.

        Args:
            exc: Exception type (if any)
            value: Exception value (if any)
            traceback: Exception traceback (if any)
        """
        if exc:
            self.rollback()
        else:
            self.commit()
        self.close()

    @property
    def resp(self):
        """Get the last query response.

        Returns:
            The raw response from the last query() call

        Note:
            This property is updated each time query() is called directly.
            It does not reflect queries executed through cursors.
        """
        return self._resp
