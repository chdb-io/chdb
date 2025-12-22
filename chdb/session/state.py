import warnings

import chdb
from ..state import sqlitelike as chdb_stateful
from ..state.sqlitelike import StreamingResult


class Session:
    """
    Session will keep the state of query.
    If path is None, it will create a temporary directory and use it as the database path
    and the temporary directory will be removed when the session is closed.
    You can also pass in a path to create a database at that path where will keep your data.

    You can also use a connection string to pass in the path and other parameters.
    Examples:
        - ":memory:" (for in-memory database)
        - "test.db" (for relative path)
        - "file:test.db" (same as above)
        - "/path/to/test.db" (for absolute path)
        - "file:/path/to/test.db" (same as above)
        - "file:test.db?param1=value1&param2=value2" (for relative path with query params)
        - "file::memory:?verbose&log-level=test" (for in-memory database with query params)
        - "///path/to/test.db?param1=value1&param2=value2" (for absolute path)

    Connection string args handling:
        Connection string can contain query params like "file:test.db?param1=value1&param2=value2"
        "param1=value1" will be passed to ClickHouse engine as start up args.

        For more details, see `clickhouse local --help --verbose`
        Some special args handling:
        - "mode=ro" would be "--readonly=1" for clickhouse (read-only mode)

    Important:
        - Multiple sessions can coexist. Each session has its own connection and database context.
        - Sessions are thread-safe: Multiple threads can safely use the same session concurrently.
        - Internal mutexes protect concurrent access to the underlying connection and client.
        - For optimal performance in multi-threaded scenarios, consider creating a separate session for each thread
          to avoid lock contention, though sharing a session across threads is safe.
    """

    def __init__(self, path=None):
        self._conn = None
        if path is None:
            self._path = ":memory:"
        else:
            self._path = path
        if chdb.g_udf_path != "":
            self._udf_path = chdb.g_udf_path
            # add udf_path to conn_str here.
            # - the `user_scripts_path` will be the value of `udf_path`
            # - the `user_defined_executable_functions_config` will be `user_scripts_path/*.xml`
            # Both of them will be added to the conn_str in the Connection class
            if "?" in self._path:
                self._conn_str = f"{self._path}&udf_path={self._udf_path}"
            else:
                self._conn_str = f"{self._path}?udf_path={self._udf_path}"
        else:
            self._udf_path = ""
            self._conn_str = f"{self._path}"
        self._conn = chdb_stateful.Connection(self._conn_str)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close the session and cleanup resources.

        This method closes the underlying connection and resets the global session state.
        After calling this method, the session becomes invalid and cannot be used for
        further queries.

        .. note::
            This method is automatically called when the session is used as a context manager
            or when the session object is destroyed.

        .. warning::
            Any attempt to use the session after calling close() will result in an error.

        Examples:
            >>> session = Session("test.db")
            >>> session.query("SELECT 1")
            >>> session.close()  # Explicitly close the session
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def cleanup(self):
        """Cleanup session resources with exception handling.

        This method attempts to close the session while suppressing any exceptions
        that might occur during the cleanup process. It's particularly useful in
        error handling scenarios or when you need to ensure cleanup happens regardless
        of the session state.

        .. note::
            This method will never raise an exception, making it safe to call in
            finally blocks or destructors.

        .. seealso::
            :meth:`close` - For explicit session closing with error propagation

        Examples:
            >>> session = Session("test.db")
            >>> try:
            ...     session.query("INVALID SQL")
            ... finally:
            ...     session.cleanup()  # Safe cleanup regardless of errors
        """
        try:
            self.close()
        except:  # noqa
            pass

    def query(self, sql, fmt="CSV", udf_path="", params=None):
        """Execute a SQL query and return the results.

        This method executes a SQL query against the session's database and returns
        the results in the specified format. The method supports various output formats
        and maintains session state between queries.

        Args:
            sql (str): SQL query string to execute
            fmt (str, optional): Output format for results. Defaults to "CSV".
                Available formats include:

                - "CSV" - Comma-separated values
                - "JSON" - JSON format
                - "TabSeparated" - Tab-separated values
                - "Pretty" - Pretty-printed table format
                - "JSONCompact" - Compact JSON format
                - "Arrow" - Apache Arrow format
                - "Parquet" - Parquet format
                - "DataFrame" - Pandas DataFrame
                - "ArrowTable" - PyArrow Table

            udf_path (str, optional): Path to user-defined functions. Defaults to "".
                If not specified, uses the UDF path from session initialization.
            params (dict, optional): Named parameters for ``{name:Type}`` placeholders.
                Values must be compatible with the declared ClickHouse type; otherwise
                query execution raises a RuntimeError.

        Returns:
            Query results in the specified format. The exact return type depends on
            the format parameter:

            - String formats (CSV, JSON, etc.) return str
            - Binary formats (Arrow, Parquet) return bytes

        Raises:
            RuntimeError: If the session is closed or invalid
            ValueError: If the SQL query is malformed

        .. note::
            The "Debug" format is not supported and will be automatically converted
            to "CSV" with a warning. For debugging, use connection string parameters
            instead.

        .. warning::
            This method executes the query synchronously and loads all results into
            memory. For large result sets, consider using :meth:`send_query` for
            streaming results.

        Examples:
            >>> session = Session("test.db")
            >>>
            >>> # Basic query with default CSV format
            >>> result = session.query("SELECT 1 as number")
            >>> print(result)
            number
            1

            >>> # Query with JSON format
            >>> result = session.query("SELECT 1 as number", fmt="JSON")
            >>> print(result)
            {"number": "1"}

            >>> # Complex query with table creation
            >>> session.query("CREATE TABLE test (id INT, name String)")
            >>> session.query("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
            >>> result = session.query("SELECT * FROM test ORDER BY id")
            >>> print(result)
            id,name
            1,Alice
            2,Bob

        .. seealso::
            :meth:`send_query` - For streaming query execution
            :attr:`sql` - Alias for this method
        """
        if fmt == "Debug":
            warnings.warn(
                """Debug format is not supported in Session.query
Please try use parameters in connection string instead:
Eg: conn = connect(f"db_path?verbose&log-level=test")"""
            )
            fmt = "CSV"
        return self._conn.query(sql, fmt, params=params)

    # alias sql = query
    sql = query

    def generate_sql(self, prompt: str) -> str:
        """Generate SQL text from a natural language prompt using the configured AI provider."""
        if self._conn is None:
            raise RuntimeError("Session is closed.")
        return self._conn.generate_sql(prompt)

    def ask(self, prompt: str, **kwargs):
        """Generate SQL from a prompt, execute it, and return the results.

        All keyword arguments are forwarded to the underlying :meth:`query`.
        """
        if self._conn is None:
            raise RuntimeError("Session is closed.")
        return self._conn.ask(prompt, **kwargs)

    def send_query(self, sql, fmt="CSV", params=None) -> StreamingResult:
        """Execute a SQL query and return a streaming result iterator.

        This method executes a SQL query against the session's database and returns
        a streaming result object that allows you to iterate over the results without
        loading everything into memory at once. This is particularly useful for large
        result sets.

        Args:
            sql (str): SQL query string to execute
            fmt (str, optional): Output format for results. Defaults to "CSV".
                Available formats include:

                - "CSV" - Comma-separated values
                - "JSON" - JSON format
                - "TabSeparated" - Tab-separated values
                - "JSONCompact" - Compact JSON format
                - "Arrow" - Apache Arrow format
                - "Parquet" - Parquet format
                - "DataFrame" - Pandas DataFrame
                - "ArrowTable" - PyArrow Table
            params (dict, optional): Named parameters for ``{name:Type}`` placeholders.
                Type mismatches or missing required parameters propagate as RuntimeError
                when fetching from the stream.

        Returns:
            StreamingResult: A streaming result iterator that yields query results
            incrementally. The iterator can be used in for loops or converted to
            other data structures.

        Raises:
            RuntimeError: If the session is closed or invalid
            ValueError: If the SQL query is malformed

        .. note::
            The "Debug" format is not supported and will be automatically converted
            to "CSV" with a warning. For debugging, use connection string parameters
            instead.

        .. warning::
            The returned StreamingResult object should be consumed promptly or stored
            appropriately, as it maintains a connection to the database.

        Examples:
            >>> session = Session("test.db")
            >>> session.query("CREATE TABLE big_table (id INT, data String)")
            >>>
            >>> # Insert large dataset
            >>> for i in range(1000):
            ...     session.query(f"INSERT INTO big_table VALUES ({i}, 'data_{i}')")
            >>>
            >>> # Stream results to avoid memory issues
            >>> streaming_result = session.send_query("SELECT * FROM big_table ORDER BY id")
            >>> for chunk in streaming_result:
            ...     print(f"Processing chunk: {len(chunk)} bytes")
            ...     # Process chunk without loading entire result set

            >>> # Using with context manager
            >>> with session.send_query("SELECT COUNT(*) FROM big_table") as stream:
            ...     for result in stream:
            ...         print(f"Count result: {result}")

        .. seealso::
            :meth:`query` - For non-streaming query execution
            :class:`chdb.state.sqlitelike.StreamingResult` - Streaming result iterator
        """
        if fmt == "Debug":
            warnings.warn(
                """Debug format is not supported in Session.query
Please try use parameters in connection string instead:
Eg: conn = connect(f"db_path?verbose&log-level=test")"""
            )
            fmt = "CSV"
        return self._conn.send_query(sql, fmt, params=params)
