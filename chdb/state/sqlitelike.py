from typing import Optional, Any
from chdb import _chdb

# try import pyarrow if failed, raise ImportError with suggestion
try:
    import pyarrow as pa  # noqa
except ImportError as e:
    print(f"ImportError: {e}")
    print('Please install pyarrow via "pip install pyarrow"')
    raise ImportError("Failed to import pyarrow") from None


_arrow_format = set({"arrowtable"})
_process_result_format_funs = {
    "arrowtable": lambda x: to_arrowTable(x),
}


# return pyarrow table
def to_arrowTable(res):
    """Convert query result to PyArrow Table.

    This function converts chdb query results to a PyArrow Table format,
    which provides efficient columnar data access and interoperability
    with other data processing libraries.

    Args:
        res: Query result object from chdb containing Arrow format data

    Returns:
        pyarrow.Table: PyArrow Table containing the query results

    Raises:
        ImportError: If pyarrow or pandas packages are not installed

    .. note::
        This function requires both pyarrow and pandas to be installed.
        Install them with: ``pip install pyarrow pandas``

    .. warning::
        Empty results return an empty PyArrow Table with no schema.

    Examples:
        >>> import chdb
        >>> result = chdb.query("SELECT 1 as num, 'hello' as text", "Arrow")
        >>> table = to_arrowTable(result)
        >>> print(table.schema)
        num: int64
        text: string
        >>> print(table.to_pandas())
           num   text
        0    1  hello
    """
    # try import pyarrow and pandas, if failed, raise ImportError with suggestion
    try:
        import pyarrow as pa  # noqa
        import pandas as pd  # noqa
    except ImportError as e:
        print(f"ImportError: {e}")
        print('Please install pyarrow and pandas via "pip install pyarrow pandas"')
        raise ImportError("Failed to import pyarrow or pandas") from None
    if len(res) == 0:
        return pa.Table.from_batches([], schema=pa.schema([]))

    memview = res.get_memview()
    return pa.RecordBatchFileReader(memview.view()).read_all()


class StreamingResult:
    def __init__(self, c_result, conn, result_func, supports_record_batch, is_dataframe):
        self._result = c_result
        self._result_func = result_func
        self._conn = conn
        self._exhausted = False
        self._supports_record_batch = supports_record_batch
        self._is_dataframe = is_dataframe

    def fetch(self):
        """Fetch the next chunk of streaming results.

        This method retrieves the next available chunk of data from the streaming
        query result. It automatically handles exhaustion detection and applies
        the configured result transformation function.

        Returns:
            The next chunk of results in the format specified during query execution,
            or None if no more data is available

        Raises:
            RuntimeError: If the streaming query encounters an error

        .. note::
            Once the stream is exhausted (returns None), subsequent calls will
            continue to return None.

        .. warning::
            This method should be called sequentially. Concurrent calls may
            result in undefined behavior.

        Examples:
            >>> conn = Connection(":memory:")
            >>> stream = conn.send_query("SELECT number FROM numbers(100)")
            >>> chunk = stream.fetch()
            >>> while chunk is not None:
            ...     print(f"Got chunk with {len(chunk)} bytes")
            ...     chunk = stream.fetch()
        """
        if self._exhausted:
            return None

        try:
            if self._is_dataframe:
                result = self._conn.streaming_fetch_df(self._result)
                if (result is None or result.empty):
                    self._exhausted = True
                    return None
            else:
                result = self._conn.streaming_fetch_result(self._result)
                if (result is None or result.rows_read() == 0):
                    self._exhausted = True
                    return None
            return self._result_func(result)
        except Exception as e:
            self._exhausted = True
            raise RuntimeError(f"Streaming query failed: {str(e)}") from e

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration

        chunk = self.fetch()
        if chunk is None:
            self._exhausted = True
            raise StopIteration

        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cancel()

    def close(self):
        """Close the streaming result and cleanup resources.

        This method is an alias for :meth:`cancel` and provides a more
        intuitive interface for resource cleanup. It cancels the streaming
        query and marks the result as exhausted.

        .. seealso::
            :meth:`cancel` - The underlying cancellation method

        Examples:
            >>> stream = conn.send_query("SELECT * FROM large_table")
            >>> # Process some data
            >>> chunk = stream.fetch()
            >>> # Close when done
            >>> stream.close()
        """
        self.cancel()

    def cancel(self):
        """Cancel the streaming query and cleanup resources.

        This method cancels the streaming query on the server side and marks
        the StreamingResult as exhausted. After calling this method, no more
        data can be fetched from this result.

        Raises:
            RuntimeError: If cancellation fails on the server side

        .. note::
            This method is idempotent - calling it multiple times is safe
            and will not cause errors.

        .. warning::
            Once cancelled, the streaming result cannot be resumed or reset.
            You must create a new query to get fresh results.

        Examples:
            >>> stream = conn.send_query("SELECT * FROM huge_table")
            >>> # Process first few chunks
            >>> for i, chunk in enumerate(stream):
            ...     if i >= 5:  # Stop after 5 chunks
            ...         stream.cancel()
            ...         break
            ...     process_chunk(chunk)
        """
        if not self._exhausted:
            self._exhausted = True
            try:
                self._conn.streaming_cancel_query(self._result)
            except Exception as e:
                raise RuntimeError(f"Failed to cancel streaming query: {str(e)}") from e

    def record_batch(self, rows_per_batch: int = 1000000) -> pa.RecordBatchReader:
        """
        Create a PyArrow RecordBatchReader from this StreamingResult.

        This method requires that the StreamingResult was created with arrow format.
        It wraps the streaming result with ChdbRecordBatchReader to provide efficient
        batching with configurable batch sizes.

        Args:
            rows_per_batch (int): Number of rows per batch. Defaults to 1000000.

        Returns:
            pa.RecordBatchReader: PyArrow RecordBatchReader for efficient streaming

        Raises:
            ValueError: If the StreamingResult was not created with arrow format
        """
        if not self._supports_record_batch:
            raise ValueError(
                "record_batch() can only be used with arrow format. "
                "Please use format='Arrow' when calling send_query."
            )

        chdb_reader = ChdbRecordBatchReader(self, rows_per_batch)
        return pa.RecordBatchReader.from_batches(chdb_reader.schema(), chdb_reader)


class ChdbRecordBatchReader:
    """
    A PyArrow RecordBatchReader wrapper for chdb StreamingResult.

    This class provides an efficient way to read large result sets as PyArrow RecordBatches
    with configurable batch sizes to optimize memory usage and performance.
    """

    def __init__(self, chdb_stream_result, batch_size_rows):
        self._stream_result = chdb_stream_result
        self._schema = None
        self._closed = False
        self._pending_batches = []
        self._accumulator = []
        self._batch_size_rows = batch_size_rows
        self._current_rows = 0
        self._first_batch = None
        self._first_batch_consumed = True
        self._schema = self.schema()

    def schema(self):
        if self._schema is None:
            # Get the first chunk to determine schema
            chunk = self._stream_result.fetch()
            if chunk is not None:
                arrow_bytes = chunk.bytes()
                reader = pa.RecordBatchFileReader(arrow_bytes)
                self._schema = reader.schema

                table = reader.read_all()
                if table.num_rows > 0:
                    batches = table.to_batches()
                    self._first_batch = batches[0]
                    if len(batches) > 1:
                        self._pending_batches = batches[1:]
                    self._first_batch_consumed = False
                else:
                    self._first_batch = None
                    self._first_batch_consumed = True
            else:
                self._schema = pa.schema([])
                self._first_batch = None
                self._first_batch_consumed = True
                self._closed = True
        return self._schema

    def read_next_batch(self):
        if self._accumulator:
            result = self._accumulator.pop(0)
            return result

        if self._closed:
            raise StopIteration

        while True:
            batch = None

            # 1. Return the first batch if not consumed yet
            if not self._first_batch_consumed:
                self._first_batch_consumed = True
                batch = self._first_batch

            # 2. Check pending batches from current chunk
            elif self._pending_batches:
                batch = self._pending_batches.pop(0)

            # 3. Fetch new chunk from chdb stream
            else:
                chunk = self._stream_result.fetch()
                if chunk is None:
                    # No more data - return accumulated batches if any
                    break

                arrow_bytes = chunk.bytes()
                if not arrow_bytes:
                    continue

                reader = pa.RecordBatchFileReader(arrow_bytes)
                table = reader.read_all()

                if table.num_rows > 0:
                    batches = table.to_batches()
                    batch = batches[0]
                    if len(batches) > 1:
                        self._pending_batches = batches[1:]
                else:
                    continue

            # Process the batch if we got one
            if batch is not None:
                self._accumulator.append(batch)
                self._current_rows += batch.num_rows

                # If accumulated enough rows, return combined batch
                if self._current_rows >= self._batch_size_rows:
                    if len(self._accumulator) == 1:
                        result = self._accumulator.pop(0)
                    else:
                        if hasattr(pa, 'concat_batches'):
                            result = pa.concat_batches(self._accumulator)
                            self._accumulator = []
                        else:
                            result = self._accumulator.pop(0)

                    self._current_rows = 0
                    return result

        # End of stream - return any accumulated batches
        if self._accumulator:
            if len(self._accumulator) == 1:
                result = self._accumulator.pop(0)
            else:
                if hasattr(pa, 'concat_batches'):
                    result = pa.concat_batches(self._accumulator)
                    self._accumulator = []
                else:
                    result = self._accumulator.pop(0)

            self._current_rows = 0
            self._closed = True
            return result

        # No more data
        self._closed = True
        raise StopIteration

    def close(self):
        if not self._closed:
            self._stream_result.close()
            self._closed = True

    def __iter__(self):
        return self

    def __next__(self):
        return self.read_next_batch()


class Connection:
    def __init__(self, connection_string: str):
        # print("Connection", connection_string)
        self._cursor: Optional[Cursor] = None
        self._conn = _chdb.connect(connection_string)

    def cursor(self) -> "Cursor":
        """Create a cursor object for executing queries.

        This method creates a database cursor that provides the standard
        DB-API 2.0 interface for executing queries and fetching results.
        The cursor allows for fine-grained control over query execution
        and result retrieval.

        Returns:
            Cursor: A cursor object for database operations

        .. note::
            Creating a new cursor will replace any existing cursor associated
            with this connection. Only one cursor per connection is supported.

        Examples:
            >>> conn = connect(":memory:")
            >>> cursor = conn.cursor()
            >>> cursor.execute("CREATE TABLE test (id INT, name String)")
            >>> cursor.execute("INSERT INTO test VALUES (1, 'Alice')")
            >>> cursor.execute("SELECT * FROM test")
            >>> rows = cursor.fetchall()
            >>> print(rows)
            ((1, 'Alice'),)

        .. seealso::
            :class:`Cursor` - Database cursor implementation
        """
        self._cursor = Cursor(self._conn)
        return self._cursor

    def query(self, query: str, format: str = "CSV", params=None) -> Any:
        """Execute a SQL query and return the complete results.

        This method executes a SQL query synchronously and returns the complete
        result set. It supports various output formats and automatically applies
        format-specific post-processing.

        Args:
            query (str): SQL query string to execute
            format (str, optional): Output format for results. Defaults to "CSV".
                Supported formats:

                - "CSV" - Comma-separated values (string)
                - "JSON" - JSON format (string)
                - "Arrow" - Apache Arrow format (bytes)
                - "Dataframe" - Pandas DataFrame (requires pandas)
                - "Arrowtable" - PyArrow Table (requires pyarrow)

        Returns:
            Query results in the specified format. Type depends on format:

            - String formats return str
            - Arrow format returns bytes
            - dataframe format returns pandas.DataFrame
            - arrowtable format returns pyarrow.Table

        Raises:
            RuntimeError: If query execution fails
            ImportError: If required packages for format are not installed

        .. warning::
            This method loads the entire result set into memory. For large
            results, consider using :meth:`send_query` for streaming.

        Examples:
            >>> conn = connect(":memory:")
            >>>
            >>> # Basic CSV query
            >>> result = conn.query("SELECT 1 as num, 'hello' as text")
            >>> print(result)
            num,text
            1,hello

            >>> # DataFrame format
            >>> df = conn.query("SELECT number FROM numbers(5)", "dataframe")
            >>> print(df)
               number
            0       0
            1       1
            2       2
            3       3
            4       4

        .. seealso::
            :meth:`send_query` - For streaming query execution
        """
        lower_output_format = format.lower()
        result_func = _process_result_format_funs.get(lower_output_format, lambda x: x)
        if lower_output_format in _arrow_format:
            format = "Arrow"

        if lower_output_format == "dataframe":
            result = self._conn.query_df(query, params=params or {})
        else:
            result = self._conn.query(query, format, params=params or {})
        return result_func(result)

    def generate_sql(self, prompt: str) -> str:
        """Generate SQL text from a natural language prompt using the configured AI provider."""
        if not hasattr(self._conn, "generate_sql"):
            raise RuntimeError("AI SQL generation is not available in this build.")
        return self._conn.generate_sql(prompt)

    def ask(self, prompt: str, **kwargs) -> Any:
        """Generate SQL from a prompt, execute it, and return the results.

        This convenience method first calls :meth:`generate_sql` to translate
        a natural language prompt into SQL, then executes the generated SQL via
        :meth:`query`, forwarding any keyword arguments to :meth:`query`.

        Args:
            prompt (str): Natural language description of the desired query.
            **kwargs: Additional keyword arguments forwarded to :meth:`query`
                (for example ``format`` or ``params``). If omitted, defaults
                from :meth:`query` are used.

        Returns:
            Query results in the requested format (CSV by default).

        Raises:
            RuntimeError: If SQL generation is unavailable or query execution fails.
        """
        generated_sql = self.generate_sql(prompt)
        return self.query(generated_sql, **kwargs)

    def send_query(self, query: str, format: str = "CSV", params=None) -> StreamingResult:
        """Execute a SQL query and return a streaming result iterator.

        This method executes a SQL query and returns a StreamingResult object
        that allows you to iterate over the results without loading everything
        into memory at once. This is ideal for processing large result sets.

        Args:
            query (str): SQL query string to execute
            format (str, optional): Output format for results. Defaults to "CSV".
                Supported formats:

                - "CSV" - Comma-separated values
                - "JSON" - JSON format
                - "Arrow" - Apache Arrow format (enables record_batch() method)
                - "dataframe" - Pandas DataFrame chunks
                - "arrowtable" - PyArrow Table chunks

        Returns:
            StreamingResult: A streaming iterator for query results that supports:

            - Iterator protocol (for loops)
            - Context manager protocol (with statements)
            - Manual fetching with fetch() method
            - PyArrow RecordBatch streaming (Arrow format only)

        Raises:
            RuntimeError: If query execution fails
            ImportError: If required packages for format are not installed

        .. note::
            Only the "Arrow" format supports the record_batch() method on the
            returned StreamingResult.

        Examples:
            >>> conn = connect(":memory:")
            >>>
            >>> # Basic streaming
            >>> stream = conn.send_query("SELECT number FROM numbers(1000)")
            >>> for chunk in stream:
            ...     print(f"Processing chunk: {len(chunk)} bytes")

            >>> # Using context manager for cleanup
            >>> with conn.send_query("SELECT * FROM large_table") as stream:
            ...     chunk = stream.fetch()
            ...     while chunk:
            ...         process_data(chunk)
            ...         chunk = stream.fetch()

            >>> # Arrow format with RecordBatch streaming
            >>> stream = conn.send_query("SELECT * FROM data", "Arrow")
            >>> reader = stream.record_batch(rows_per_batch=10000)
            >>> for batch in reader:
            ...     print(f"Batch shape: {batch.num_rows} x {batch.num_columns}")

        .. seealso::
            :meth:`query` - For non-streaming query execution
            :class:`StreamingResult` - Streaming result iterator
        """
        lower_output_format = format.lower()
        supports_record_batch = lower_output_format == "arrow"
        result_func = _process_result_format_funs.get(lower_output_format, lambda x: x)
        if lower_output_format in _arrow_format:
            format = "Arrow"

        c_stream_result = self._conn.send_query(query, format, params=params or {})
        is_dataframe = lower_output_format == "dataframe"
        return StreamingResult(c_stream_result, self._conn, result_func, supports_record_batch, is_dataframe)

    def __enter__(self):
        """Enter the context manager and return the connection.

        Returns:
            Connection: The connection object itself
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the connection.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            False to propagate any exception that occurred
        """
        self.close()
        return False

    def close(self) -> None:
        """Close the connection and cleanup resources.

        This method closes the database connection and cleans up any associated
        resources including active cursors. After calling this method, the
        connection becomes invalid and cannot be used for further operations.

        .. note::
            This method is idempotent - calling it multiple times is safe.

        .. warning::
            Any ongoing streaming queries will be cancelled when the connection
            is closed. Ensure all important data is processed before closing.

        Examples:
            >>> conn = connect("test.db")
            >>> # Use connection for queries
            >>> conn.query("CREATE TABLE test (id INT)")
            >>> # Close when done
            >>> conn.close()

            >>> # Using with context manager (automatic cleanup)
            >>> with connect("test.db") as conn:
            ...     conn.query("SELECT 1")
            ...     # Connection automatically closed
        """
        # print("close")
        if self._cursor:
            self._cursor.close()
        self._conn.close()


class Cursor:
    def __init__(self, connection):
        self._conn = connection
        self._cursor = self._conn.cursor()
        self._current_table: Optional[pa.Table] = None
        self._current_row: int = 0

    def execute(self, query: str) -> None:
        """Execute a SQL query and prepare results for fetching.

        This method executes a SQL query and prepares the results for retrieval
        using the fetch methods. It handles the parsing of result data and
        automatic type conversion for ClickHouse data types.

        Args:
            query (str): SQL query string to execute

        Raises:
            Exception: If query execution fails or result parsing fails

        .. note::
            This method follows DB-API 2.0 specifications for cursor.execute().
            After execution, use fetchone(), fetchmany(), or fetchall() to
            retrieve results.

        .. note::
            The method automatically converts ClickHouse data types to appropriate
            Python types:

            - Int/UInt types → int
            - Float types → float
            - String/FixedString → str
            - DateTime → datetime.datetime
            - Date → datetime.date
            - Bool → bool

        Examples:
            >>> cursor = conn.cursor()
            >>>
            >>> # Execute DDL
            >>> cursor.execute("CREATE TABLE test (id INT, name String)")
            >>>
            >>> # Execute DML
            >>> cursor.execute("INSERT INTO test VALUES (1, 'Alice')")
            >>>
            >>> # Execute SELECT and fetch results
            >>> cursor.execute("SELECT * FROM test")
            >>> rows = cursor.fetchall()
            >>> print(rows)
            ((1, 'Alice'),)

        .. seealso::
            :meth:`fetchone` - Fetch single row
            :meth:`fetchmany` - Fetch multiple rows
            :meth:`fetchall` - Fetch all remaining rows
        """
        self._cursor.execute(query)
        result_mv = self._cursor.get_memview()
        if self._cursor.has_error():
            raise Exception(self._cursor.error_message())
        if self._cursor.data_size() == 0:
            self._current_table = None
            self._current_row = 0
            self._column_names = []
            self._column_types = []
            return

        # Parse JSON data
        json_data = result_mv.tobytes().decode("utf-8")
        import json

        try:
            # First line contains column names
            # Second line contains column types
            # Following lines contain data
            lines = json_data.strip().split("\n")
            if len(lines) < 2:
                self._current_table = None
                self._current_row = 0
                self._column_names = []
                self._column_types = []
                return

            self._column_names = json.loads(lines[0])
            self._column_types = json.loads(lines[1])

            # Convert data rows
            rows = []
            for line in lines[2:]:
                if not line.strip():
                    continue
                row_data = json.loads(line)
                converted_row = []
                for val, type_info in zip(row_data, self._column_types):
                    # Handle NULL values first
                    if val is None:
                        converted_row.append(None)
                        continue

                    # Basic type conversion
                    try:
                        if type_info.startswith("Int") or type_info.startswith("UInt"):
                            converted_row.append(int(val))
                        elif type_info.startswith("Float"):
                            converted_row.append(float(val))
                        elif type_info == "Bool":
                            converted_row.append(bool(val))
                        elif type_info == "String" or type_info == "FixedString":
                            converted_row.append(str(val))
                        elif type_info.startswith("DateTime"):
                            from datetime import datetime

                            # Check if the value is numeric (timestamp)
                            val_str = str(val)
                            if val_str.replace(".", "").isdigit():
                                converted_row.append(datetime.fromtimestamp(float(val)))
                            else:
                                # Handle datetime string formats
                                if "." in val_str:  # Has microseconds
                                    converted_row.append(
                                        datetime.strptime(
                                            val_str, "%Y-%m-%d %H:%M:%S.%f"
                                        )
                                    )
                                else:  # No microseconds
                                    converted_row.append(
                                        datetime.strptime(val_str, "%Y-%m-%d %H:%M:%S")
                                    )
                        elif type_info.startswith("Date"):
                            from datetime import date, datetime

                            # Check if the value is numeric (days since epoch)
                            val_str = str(val)
                            if val_str.isdigit():
                                converted_row.append(
                                    date.fromtimestamp(float(val) * 86400)
                                )
                            else:
                                # Handle date string format
                                converted_row.append(
                                    datetime.strptime(val_str, "%Y-%m-%d").date()
                                )
                        else:
                            # For unsupported types, keep as string
                            converted_row.append(str(val))
                    except (ValueError, TypeError):
                        # If conversion fails, keep original value as string
                        converted_row.append(str(val))
                rows.append(tuple(converted_row))

            self._current_table = rows
            self._current_row = 0

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON data: {e}")

    def commit(self) -> None:
        """Commit any pending transaction.

        This method commits any pending database transaction. In ClickHouse,
        most operations are auto-committed, but this method is provided for
        DB-API 2.0 compatibility.

        .. note::
            ClickHouse typically auto-commits operations, so explicit commits
            are usually not necessary. This method is provided for compatibility
            with standard DB-API 2.0 workflow.

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("INSERT INTO test VALUES (1, 'data')")
            >>> cursor.commit()
        """
        self._cursor.commit()

    def fetchone(self) -> Optional[tuple]:
        """Fetch the next row from the query result.

        This method retrieves the next available row from the current query
        result set. It returns a tuple containing the column values with
        appropriate Python type conversion applied.

        Returns:
            Optional[tuple]: Next row as a tuple of column values, or None
            if no more rows are available

        .. note::
            This method follows DB-API 2.0 specifications. Column values are
            automatically converted to appropriate Python types based on
            ClickHouse column types.

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT id, name FROM users")
            >>> row = cursor.fetchone()
            >>> while row is not None:
            ...     user_id, user_name = row
            ...     print(f"User {user_id}: {user_name}")
            ...     row = cursor.fetchone()

        .. seealso::
            :meth:`fetchmany` - Fetch multiple rows
            :meth:`fetchall` - Fetch all remaining rows
        """
        if not self._current_table or self._current_row >= len(self._current_table):
            return None

        # Now self._current_table is a list of row tuples
        row = self._current_table[self._current_row]
        self._current_row += 1
        return row

    def fetchmany(self, size: int = 1) -> tuple:
        """Fetch multiple rows from the query result.

        This method retrieves up to 'size' rows from the current query result
        set. It returns a tuple of row tuples, with each row containing column
        values with appropriate Python type conversion.

        Args:
            size (int, optional): Maximum number of rows to fetch. Defaults to 1.

        Returns:
            tuple: Tuple containing up to 'size' row tuples. May contain fewer
            rows if the result set is exhausted.

        .. note::
            This method follows DB-API 2.0 specifications. It will return fewer
            than 'size' rows if the result set is exhausted.

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT * FROM large_table")
            >>>
            >>> # Process results in batches
            >>> while True:
            ...     batch = cursor.fetchmany(100)  # Fetch 100 rows at a time
            ...     if not batch:
            ...         break
            ...     process_batch(batch)

        .. seealso::
            :meth:`fetchone` - Fetch single row
            :meth:`fetchall` - Fetch all remaining rows
        """
        if not self._current_table:
            return tuple()

        rows = []
        for _ in range(size):
            if (row := self.fetchone()) is None:
                break
            rows.append(row)
        return tuple(rows)

    def fetchall(self) -> tuple:
        """Fetch all remaining rows from the query result.

        This method retrieves all remaining rows from the current query result
        set starting from the current cursor position. It returns a tuple of
        row tuples with appropriate Python type conversion applied.

        Returns:
            tuple: Tuple containing all remaining row tuples from the result set.
            Returns empty tuple if no rows are available.

        .. warning::
            This method loads all remaining rows into memory at once. For large
            result sets, consider using :meth:`fetchmany` to process results
            in batches.

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT id, name FROM users")
            >>> all_users = cursor.fetchall()
            >>> for user_id, user_name in all_users:
            ...     print(f"User {user_id}: {user_name}")

        .. seealso::
            :meth:`fetchone` - Fetch single row
            :meth:`fetchmany` - Fetch multiple rows in batches
        """
        if not self._current_table:
            return tuple()

        remaining_rows = []
        while (row := self.fetchone()) is not None:
            remaining_rows.append(row)
        return tuple(remaining_rows)

    def close(self) -> None:
        """Close the cursor and cleanup resources.

        This method closes the cursor and cleans up any associated resources.
        After calling this method, the cursor becomes invalid and cannot be
        used for further operations.

        .. note::
            This method is idempotent - calling it multiple times is safe.
            The cursor is also automatically closed when the connection is closed.

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT 1")
            >>> result = cursor.fetchone()
            >>> cursor.close()  # Cleanup cursor resources
        """
        self._cursor.close()

    def __iter__(self):
        return self

    def __next__(self) -> tuple:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def column_names(self) -> list:
        """Return a list of column names from the last executed query.

        This method returns the column names from the most recently executed
        SELECT query. The names are returned in the same order as they appear
        in the result set.

        Returns:
            list: List of column name strings, or empty list if no query
            has been executed or the query returned no columns

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT id, name, email FROM users LIMIT 1")
            >>> print(cursor.column_names())
            ['id', 'name', 'email']

        .. seealso::
            :meth:`column_types` - Get column type information
            :attr:`description` - DB-API 2.0 column description
        """
        return self._column_names if hasattr(self, "_column_names") else []

    def column_types(self) -> list:
        """Return a list of column types from the last executed query.

        This method returns the ClickHouse column type names from the most
        recently executed SELECT query. The types are returned in the same
        order as they appear in the result set.

        Returns:
            list: List of ClickHouse type name strings, or empty list if no
            query has been executed or the query returned no columns

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT toInt32(1), toString('hello')")
            >>> print(cursor.column_types())
            ['Int32', 'String']

        .. seealso::
            :meth:`column_names` - Get column name information
            :attr:`description` - DB-API 2.0 column description
        """
        return self._column_types if hasattr(self, "_column_types") else []

    @property
    def description(self) -> list:
        """Return column description as per DB-API 2.0 specification.

        This property returns a list of 7-item tuples describing each column
        in the result set of the last executed SELECT query. Each tuple contains:
        (name, type_code, display_size, internal_size, precision, scale, null_ok)

        Currently, only name and type_code are provided, with other fields set to None.

        Returns:
            list: List of 7-tuples describing each column, or empty list if no
            SELECT query has been executed

        .. note::
            This follows the DB-API 2.0 specification for cursor.description.
            Only the first two elements (name and type_code) contain meaningful
            data in this implementation.

        Examples:
            >>> cursor = conn.cursor()
            >>> cursor.execute("SELECT id, name FROM users LIMIT 1")
            >>> for desc in cursor.description:
            ...     print(f"Column: {desc[0]}, Type: {desc[1]}")
            Column: id, Type: Int32
            Column: name, Type: String

        .. seealso::
            :meth:`column_names` - Get just column names
            :meth:`column_types` - Get just column types
        """
        if not hasattr(self, "_column_names") or not self._column_names:
            return []

        return [
            (name, type_info, None, None, None, None, None)
            for name, type_info in zip(self._column_names, self._column_types)
        ]


def connect(connection_string: str = ":memory:") -> Connection:
    """Create a connection to chDB background server.

    This function establishes a connection to the chDB (ClickHouse) database engine.
    Only one open connection is allowed per process. Multiple calls with the same
    connection string will return the same connection object.

    Args:
        connection_string (str, optional): Database connection string. Defaults to ":memory:".
            Supported connection string formats:

            **Basic formats:**

            - ":memory:" - In-memory database (default)
            - "test.db" - Relative path database file
            - "file:test.db" - Same as relative path
            - "/path/to/test.db" - Absolute path database file
            - "file:/path/to/test.db" - Same as absolute path

            **With query parameters:**

            - "file:test.db?param1=value1&param2=value2" - Relative path with params
            - "file::memory:?verbose&log-level=test" - In-memory with params
            - "///path/to/test.db?param1=value1&param2=value2" - Absolute path with params

            **Query parameter handling:**

            Query parameters are passed to ClickHouse engine as startup arguments.
            Special parameter handling:

            - "mode=ro" becomes "--readonly=1" (read-only mode)
            - "verbose" enables verbose logging
            - "log-level=test" sets logging level

            For complete parameter list, see ``clickhouse local --help --verbose``

    Returns:
        Connection: Database connection object that supports:

        - Creating cursors with :meth:`Connection.cursor`
        - Direct queries with :meth:`Connection.query`
        - Streaming queries with :meth:`Connection.send_query`
        - Context manager protocol for automatic cleanup

    Raises:
        RuntimeError: If connection to database fails

    .. warning::
        Only one connection per process is supported. Creating a new connection
        will close any existing connection.

    Examples:
        >>> # In-memory database
        >>> conn = connect()
        >>> conn = connect(":memory:")
        >>>
        >>> # File-based database
        >>> conn = connect("my_data.db")
        >>> conn = connect("/path/to/data.db")
        >>>
        >>> # With parameters
        >>> conn = connect("data.db?mode=ro")  # Read-only mode
        >>> conn = connect(":memory:?verbose&log-level=debug")  # Debug logging
        >>>
        >>> # Using context manager for automatic cleanup
        >>> with connect("data.db") as conn:
        ...     result = conn.query("SELECT 1")
        ...     print(result)
        >>> # Connection automatically closed

    .. seealso::
        :class:`Connection` - Database connection class
        :class:`Cursor` - Database cursor for DB-API 2.0 operations
    """
    return Connection(connection_string)
