from . import err
import re

# Regular expression for :meth:`Cursor.executemany`.
# executemany only supports simple bulk insert.
# You can use it to load large dataset.
RE_INSERT_VALUES = re.compile(
    r"\s*((?:INSERT|REPLACE)\b.+\bVALUES?\s*)"
    + r"(\(\s*(?:%s|%\(.+\)s|\?)\s*(?:,\s*(?:%s|%\(.+\)s|\?)\s*)*\))"
    + r"(\s*(?:ON DUPLICATE.*)?);?\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


class Cursor(object):
    """DB-API 2.0 cursor for executing queries and fetching results.

    The cursor provides methods for executing SQL statements, managing query results,
    and navigating through result sets. It supports parameter binding, bulk operations,
    and follows DB-API 2.0 specifications.

    Do not create Cursor instances directly. Use Connection.cursor() instead.

    Attributes:
        description (tuple): Column metadata for the last query result
        rowcount (int): Number of rows affected by the last query (-1 if unknown)
        arraysize (int): Default number of rows to fetch at once (default: 1)
        lastrowid: ID of the last inserted row (if applicable)
        max_stmt_length (int): Maximum statement size for executemany() (default: 1024000)

    Examples:
        >>> conn = Connection()
        >>> cur = conn.cursor()
        >>> cur.execute("SELECT 1 as id, 'test' as name")
        >>> result = cur.fetchone()
        >>> print(result)  # (1, 'test')
        >>> cur.close()

    Note:
        See `DB-API 2.0 Cursor Objects <https://www.python.org/dev/peps/pep-0249/#cursor-objects>`_
        for complete specification details.
    """

    #: Max statement size which :meth:`executemany` generates.
    #:
    #: Default value is 1024000.
    max_stmt_length = 1024000

    def __init__(self, connection):
        """Initialize cursor for the given connection.

        Args:
            connection (Connection): Database connection to use
        """
        self.connection = connection
        self._cursor = connection._conn.cursor()
        self.description = None
        self.rowcount = -1
        self.arraysize = 1
        self.lastrowid = None
        self._executed = None

    def __enter__(self):
        """Enter context manager and return self.

        Returns:
            Cursor: This cursor instance
        """
        return self

    def __exit__(self, *exc_info):
        """Exit context manager and close cursor.

        Args:
            *exc_info: Exception information (ignored)
        """
        del exc_info
        self.close()

    def __iter__(self):
        """Make cursor iterable over result rows.

        Returns:
            iterator: Iterator yielding rows until None is returned

        Example:
            >>> cur.execute("SELECT id FROM users")
            >>> for row in cur:
            ...     print(row[0])
        """
        return iter(self.fetchone, None)

    def callproc(self, procname, args=()):
        """Execute a stored procedure (placeholder implementation).

        Args:
            procname (str): Name of stored procedure to execute
            args (sequence): Parameters to pass to the procedure

        Returns:
            sequence: The original args parameter (unmodified)

        Note:
            chDB/ClickHouse does not support stored procedures in the traditional sense.
            This method is provided for DB-API 2.0 compliance but does not perform
            any actual operation. Use execute() for all SQL operations.

        Compatibility Warning:
            This is a placeholder implementation. Traditional stored procedure
            features like OUT/INOUT parameters, multiple result sets, and server
            variables are not supported by the underlying ClickHouse engine.
        """

        return args

    def close(self):
        """Close the cursor and free associated resources.

        After closing, the cursor becomes unusable and any operation will raise an exception.
        Closing a cursor exhausts all remaining data and releases the underlying cursor.
        """
        self._cursor.close()

    def _get_db(self):
        """Internal method to get the database connection.

        Returns:
            Connection: The database connection

        Raises:
            ProgrammingError: If cursor is closed
        """
        if not self.connection:
            raise err.ProgrammingError("Cursor closed")
        return self.connection

    def _escape_args(self, args, conn):
        """Internal method to escape query arguments.

        Args:
            args (tuple/list/dict): Arguments to escape
            conn (Connection): Database connection for escaping

        Returns:
            Escaped arguments in the same structure as input
        """
        if isinstance(args, (tuple, list)):
            return tuple(conn.escape(arg) for arg in args)
        elif isinstance(args, dict):
            return {key: conn.escape(val) for (key, val) in args.items()}
        else:
            # If it's not a dictionary let's try escaping it anyway.
            # Worst case it will throw a Value error
            return conn.escape(args)

    def _format_query(self, query, args, conn):
        """Format SQL query by substituting parameter placeholders.

        This internal method handles parameter binding for both question mark (?) and
        format (%s) style placeholders, with proper escaping for SQL injection prevention.

        Args:
            query (str): SQL query with parameter placeholders
            args (tuple/list): Parameter values to substitute
            conn (Connection): Database connection for escaping values

        Returns:
            str: SQL query with parameters substituted and properly escaped

        Note:
            This is an internal method. Use execute() or mogrify() instead.
        """
        if args is None or ('?' not in query and '%' not in query):
            return query

        escaped_args = self._escape_args(args, conn)
        if not isinstance(escaped_args, (tuple, list)):
            escaped_args = (escaped_args,)

        result = []
        arg_index = 0
        max_args = len(escaped_args)
        i = 0
        query_len = len(query)
        in_string = False
        quote_char = None

        while i < query_len:
            char = query[i]
            if not in_string:
                if char in ("'", '"'):
                    in_string = True
                    quote_char = char
                elif arg_index < max_args:
                    if char == '?':
                        result.append(str(escaped_args[arg_index]))
                        arg_index += 1
                        i += 1
                        continue
                    elif char == '%' and i + 1 < query_len and query[i + 1] == 's':
                        result.append(str(escaped_args[arg_index]))
                        arg_index += 1
                        i += 2
                        continue
            elif char == quote_char and (i == 0 or query[i - 1] != '\\'):
                in_string = False
                quote_char = None

            result.append(char)
            i += 1

        return ''.join(result)

    def mogrify(self, query, args=None):
        """Return the exact query string that would be sent to the database.

        This method shows the final SQL query after parameter substitution,
        which is useful for debugging and logging purposes.

        Args:
            query (str): SQL query with parameter placeholders
            args (tuple/list/dict, optional): Parameters to substitute

        Returns:
            str: The final SQL query string with parameters substituted

        Example:
            >>> cur.mogrify("SELECT * FROM users WHERE id = ?", (123,))
            "SELECT * FROM users WHERE id = 123"

        Note:
            This method follows the extension to DB-API 2.0 used by Psycopg.
        """
        conn = self._get_db()
        return self._format_query(query, args, conn)

    def execute(self, query, args=None):
        """Execute a SQL query with optional parameter binding.

        This method executes a single SQL statement with optional parameter substitution.
        It supports multiple parameter placeholder styles for flexibility.

        Args:
            query (str): SQL query to execute
            args (tuple/list/dict, optional): Parameters to bind to placeholders

        Returns:
            int: Number of affected rows (-1 if unknown)

        Parameter Styles:
            - Question mark style: "SELECT * FROM users WHERE id = ?"
            - Named style: "SELECT * FROM users WHERE name = %(name)s"
            - Format style: "SELECT * FROM users WHERE age = %s" (legacy)

        Examples:
            >>> # Question mark parameters
            >>> cur.execute("SELECT * FROM users WHERE id = ? AND age > ?", (123, 18))
            >>>
            >>> # Named parameters
            >>> cur.execute("SELECT * FROM users WHERE name = %(name)s", {'name': 'Alice'})
            >>>
            >>> # No parameters
            >>> cur.execute("SELECT COUNT(*) FROM users")

        Raises:
            ProgrammingError: If cursor is closed or query is malformed
            InterfaceError: If database error occurs during execution
        """
        query = self._format_query(query, args, self.connection)
        self._cursor.execute(query)

        # Get description from column names and types
        if hasattr(self._cursor, "_column_names") and self._cursor._column_names:
            self.description = [
                (name, type_info, None, None, None, None, None)
                for name, type_info in zip(
                    self._cursor._column_names, self._cursor._column_types
                )
            ]
            self.rowcount = (
                len(self._cursor._current_table) if self._cursor._current_table else -1
            )
        else:
            self.description = None
            self.rowcount = -1

        self._executed = query
        return self.rowcount

    def executemany(self, query, args):
        """Execute a query multiple times with different parameter sets.

        This method efficiently executes the same SQL query multiple times with
        different parameter values. It's particularly useful for bulk INSERT operations.

        Args:
            query (str): SQL query to execute multiple times
            args (sequence): Sequence of parameter tuples/dicts/lists for each execution

        Returns:
            int: Total number of affected rows across all executions

        Examples:
            >>> # Bulk insert with question mark parameters
            >>> users_data = [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
            >>> cur.executemany("INSERT INTO users VALUES (?, ?)", users_data)
            >>>
            >>> # Bulk insert with named parameters
            >>> users_data = [
            ...     {'id': 1, 'name': 'Alice'},
            ...     {'id': 2, 'name': 'Bob'}
            ... ]
            >>> cur.executemany(
            ...     "INSERT INTO users VALUES (%(id)s, %(name)s)",
            ...     users_data
            ... )

        Note:
            This method improves performance for multiple-row INSERT and UPDATE operations
            by optimizing the query execution process.
        """
        if not args:
            return 0

        m = RE_INSERT_VALUES.match(query)
        if m:
            q_prefix = m.group(1) % ()
            q_values = m.group(2).rstrip()
            q_postfix = m.group(3) or ""
            assert q_values[0] == "(" and q_values[-1] == ")"
            return self._do_execute_many(
                q_prefix,
                q_values,
                q_postfix,
                args,
                self.max_stmt_length,
                self._get_db().encoding,
            )

        self.rowcount = sum(self.execute(query, arg) for arg in args)
        return self.rowcount

    def _find_placeholder_positions(self, query):
        positions = []
        i = 0
        query_len = len(query)
        in_string = False
        quote_char = None

        while i < query_len:
            char = query[i]
            if not in_string:
                if char in ("'", '"'):
                    in_string = True
                    quote_char = char
                elif char == '?':
                    positions.append((i, 1))  # (position, length)
                elif char == '%' and i + 1 < query_len and query[i + 1] == 's':
                    positions.append((i, 2))
                    i += 1
            elif char == quote_char and (i == 0 or query[i - 1] != '\\'):
                in_string = False
                quote_char = None
            i += 1

        return positions

    def _do_execute_many(
        self, prefix, values, postfix, args, max_stmt_length, encoding
    ):
        conn = self._get_db()
        if isinstance(prefix, str):
            prefix = prefix.encode(encoding)
        if isinstance(postfix, str):
            postfix = postfix.encode(encoding)

        # Pre-compute placeholder positions
        placeholder_positions = self._find_placeholder_positions(values)

        sql = prefix
        args = iter(args)

        if not placeholder_positions:
            values_bytes = values.encode(encoding, "surrogateescape") if isinstance(values, str) else values
            sql += values_bytes
            rows = 0
            for _ in args:
                if len(sql) + len(values_bytes) + len(postfix) + 2 > max_stmt_length:
                    rows += self.execute(sql + postfix)
                    sql = prefix + values_bytes
                else:
                    sql += ",".encode(encoding)
                    sql += values_bytes
            rows += self.execute(sql + postfix)
            self.rowcount = rows
            return rows

        template_parts = []
        last_pos = 0
        for pos, length in placeholder_positions:
            template_parts.append(values[last_pos:pos])
            last_pos = pos + length
        template_parts.append(values[last_pos:])

        def format_values_fast(escaped_arg):
            if len(escaped_arg) != len(placeholder_positions):
                return values
            result = template_parts[0]
            for i, val in enumerate(escaped_arg):
                result += str(val) + template_parts[i + 1]
            return result

        def format_values_with_positions(arg):
            escaped_arg = self._escape_args(arg, conn)
            if not isinstance(escaped_arg, (tuple, list)):
                escaped_arg = (escaped_arg,)
            return format_values_fast(escaped_arg)

        v = format_values_with_positions(next(args))
        if isinstance(v, str):
            v = v.encode(encoding, "surrogateescape")
        sql += v
        rows = 0

        for arg in args:
            v = format_values_with_positions(arg)
            if isinstance(v, str):
                v = v.encode(encoding, "surrogateescape")
            if len(sql) + len(v) + len(postfix) + 2 > max_stmt_length:  # +2 for comma
                rows += self.execute(sql + postfix)
                sql = prefix + v
            else:
                sql += ",".encode(encoding)
                sql += v
        rows += self.execute(sql + postfix)
        self.rowcount = rows
        return rows

    def _check_executed(self):
        """Internal method to verify that execute() has been called.

        Raises:
            ProgrammingError: If no query has been executed yet
        """
        if not self._executed:
            raise err.ProgrammingError("execute() first")

    def fetchone(self):
        """Fetch the next row from the query result.

        Returns:
            tuple or None: Next row as a tuple, or None if no more rows available

        Raises:
            ProgrammingError: If execute() has not been called first

        Example:
            >>> cursor.execute("SELECT id, name FROM users LIMIT 3")
            >>> row = cursor.fetchone()
            >>> print(row)  # (1, 'Alice')
            >>> row = cursor.fetchone()
            >>> print(row)  # (2, 'Bob')
        """
        if not self._executed:
            raise err.ProgrammingError("execute() first")
        return self._cursor.fetchone()

    def fetchmany(self, size=1):
        """Fetch multiple rows from the query result.

        Args:
            size (int, optional): Number of rows to fetch. Defaults to 1.
                                 If not specified, uses cursor.arraysize.

        Returns:
            list: List of tuples representing the fetched rows

        Raises:
            ProgrammingError: If execute() has not been called first

        Example:
            >>> cursor.execute("SELECT id, name FROM users")
            >>> rows = cursor.fetchmany(3)
            >>> print(rows)  # [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie')]
        """
        if not self._executed:
            raise err.ProgrammingError("execute() first")
        return self._cursor.fetchmany(size)

    def fetchall(self):
        """Fetch all remaining rows from the query result.

        Returns:
            list: List of tuples representing all remaining rows

        Raises:
            ProgrammingError: If execute() has not been called first

        Warning:
            This method can consume large amounts of memory for big result sets.
            Consider using fetchmany() for large datasets.

        Example:
            >>> cursor.execute("SELECT id, name FROM users")
            >>> all_rows = cursor.fetchall()
            >>> print(len(all_rows))  # Number of total rows
        """
        if not self._executed:
            raise err.ProgrammingError("execute() first")
        return self._cursor.fetchall()

    def nextset(self):
        """Move to the next result set (not supported).

        Returns:
            None: Always returns None as multiple result sets are not supported

        Note:
            chDB/ClickHouse does not support multiple result sets from a single query.
            This method is provided for DB-API 2.0 compliance but always returns None.
        """
        # Not support for now
        return None

    def setinputsizes(self, *args):
        """Set input sizes for parameters (no-op implementation).

        Args:
            *args: Parameter size specifications (ignored)

        Note:
            This method does nothing but is required by DB-API 2.0 specification.
            chDB automatically handles parameter sizing internally.
        """

    def setoutputsizes(self, *args):
        """Set output column sizes (no-op implementation).

        Args:
            *args: Column size specifications (ignored)

        Note:
            This method does nothing but is required by DB-API 2.0 specification.
            chDB automatically handles output sizing internally.
        """
