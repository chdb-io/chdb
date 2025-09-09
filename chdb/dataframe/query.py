import os
import tempfile
from io import BytesIO
import re
import pandas as pd
import pyarrow as pa
from chdb import query as chdb_query


class Table:
    """Wrapper for multiple data formats enabling SQL queries on DataFrames, Parquet files, and Arrow tables.

    The Table class provides a unified interface for querying different data formats using SQL.
    It supports pandas DataFrames, Parquet files (both on disk and in memory), and PyArrow Tables.
    All data is internally converted to Parquet format for efficient querying with chDB.

    Args:
        parquet_path (str, optional): Path to an existing Parquet file
        temp_parquet_path (str, optional): Path to a temporary Parquet file
        parquet_memoryview (memoryview, optional): Parquet data in memory as memoryview
        dataframe (pd.DataFrame, optional): pandas DataFrame to wrap
        arrow_table (pa.Table, optional): PyArrow Table to wrap
        use_memfd (bool, optional): Use memfd_create for temporary files (Linux only). Defaults to False.

    Examples:
        >>> # Create from pandas DataFrame
        >>> import pandas as pd
        >>> df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        >>> table = Table(dataframe=df)
        >>> result = table.query("SELECT * FROM __table__ WHERE id > 1")

        >>> # Create from Parquet file
        >>> table = Table(parquet_path="data.parquet")
        >>> result = table.query("SELECT COUNT(*) FROM __table__")

        >>> # Multi-table queries
        >>> table1 = Table(dataframe=df1)
        >>> table2 = Table(dataframe=df2)
        >>> result = Table.queryStatic(
        ...     "SELECT * FROM __table1__ JOIN __table2__ ON __table1__.id = __table2__.id",
        ...     table1=table1, table2=table2
        ... )
    """

    def __init__(
        self,
        parquet_path: str = None,
        temp_parquet_path: str = None,
        parquet_memoryview: memoryview = None,
        dataframe: pd.DataFrame = None,
        arrow_table: pa.Table = None,
        use_memfd: bool = False,
    ):
        """Initialize a Table object with one of the supported data formats.

        Only one data source should be provided. The Table will wrap the provided data
        and enable SQL querying capabilities.

        Args:
            parquet_path (str, optional): Path to existing Parquet file
            temp_parquet_path (str, optional): Path to temporary Parquet file
            parquet_memoryview (memoryview, optional): Parquet data in memory
            dataframe (pd.DataFrame, optional): pandas DataFrame to wrap
            arrow_table (pa.Table, optional): PyArrow Table to wrap
            use_memfd (bool, optional): Use memory-based file descriptors on Linux
        """
        self._parquet_path = parquet_path
        self._temp_parquet_path = temp_parquet_path
        self._parquet_memoryview = parquet_memoryview
        self._dataframe = dataframe
        self._arrow_table = arrow_table
        self.use_memfd = use_memfd
        self._rows_read = 0
        self._bytes_read = 0
        self._elapsed = 0

    def __del__(self):
        if self._temp_parquet_path is not None:
            try:
                os.remove(self._temp_parquet_path)
            except OSError:
                pass

    def rows_read(self):
        """Get the number of rows read from the last query operation.

        Returns:
            int: Number of rows processed in the last query
        """
        return self._rows_read

    def bytes_read(self):
        """Get the number of bytes read from the last query operation.

        Returns:
            int: Number of bytes processed in the last query
        """
        return self._bytes_read

    def elapsed(self):
        """Get the elapsed time for the last query operation.

        Returns:
            float: Query execution time
        """
        return self._elapsed

    def to_pandas(self) -> pd.DataFrame:
        """Convert the Table data to a pandas DataFrame.

        This method handles conversion from various internal formats (Parquet files,
        memory buffers, Arrow tables) to a unified pandas DataFrame representation.

        Returns:
            pd.DataFrame: The table data as a pandas DataFrame

        Raises:
            ValueError: If no data source is available in the Table object

        Example:
            >>> table = Table(dataframe=df)
            >>> result_table = table.query("SELECT * FROM __table__ LIMIT 5")
            >>> df_result = result_table.to_pandas()
            >>> print(df_result)
        """
        if self._dataframe is None:
            if self._arrow_table is not None:
                return self._arrow_table.to_pandas()
            elif self._parquet_memoryview is not None:
                # wrap bytes to ReadBuffer
                pq_reader = BytesIO(self._parquet_memoryview.tobytes())
                return pandas_read_parquet(pq_reader)
            elif self._parquet_path is not None:
                return pandas_read_parquet(self._parquet_path)
            elif self._temp_parquet_path is not None:
                return pandas_read_parquet(self._temp_parquet_path)
            else:
                raise ValueError("No data buffer in Table object")
        return self._dataframe

    def flush_to_disk(self):
        """Flush in-memory data to disk as a temporary Parquet file.

        This method converts in-memory data (DataFrame, Arrow table, or memory buffer)
        to a temporary Parquet file on disk. This can be useful for memory management
        or when working with large datasets.

        The method does nothing if data is already stored on disk.

        Raises:
            ValueError: If the Table object contains no data to flush

        Example:
            >>> table = Table(dataframe=large_df)
            >>> table.flush_to_disk()  # Frees memory, keeps data accessible
        """
        if self._parquet_path is not None or self._temp_parquet_path is not None:
            return

        if self._dataframe is not None:
            self._df_to_disk(self._dataframe)
            self._dataframe = None
        elif self._arrow_table is not None:
            self._arrow_table_to_disk(self._arrow_table)
            self._arrow_table = None
        elif self._parquet_memoryview is not None:
            self._memoryview_to_disk(self._parquet_memoryview)
            self._parquet_memoryview = None
        else:
            raise ValueError("No data in Table object")

    def _df_to_disk(self, df):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp)
            self._temp_parquet_path = tmp.name

    def _arrow_table_to_disk(self, arrow_table):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            pa.parquet.write_table(arrow_table, tmp.name)
            self._temp_parquet_path = tmp.name

    def _memoryview_to_disk(self, memoryview):
        # copy memoryview to temp file
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp.write(memoryview.tobytes())
            self._temp_parquet_path = tmp.name

    def __repr__(self):
        return repr(self.to_pandas())

    def __str__(self):
        return str(self.to_pandas())

    def query(self, sql: str, **kwargs) -> "Table":
        """Execute SQL query on the current Table and return a new Table with results.

        This method allows you to run SQL queries on the table data using chDB.
        The table is referenced as '__table__' in the SQL statement.

        Args:
            sql (str): SQL query string. Must reference the table as '__table__'
            **kwargs: Additional arguments passed to the chDB query engine

        Returns:
            Table: New Table object containing the query results

        Raises:
            ValueError: If SQL doesn't contain '__table__' reference or if Table is not initialized

        Examples:
            >>> table = Table(dataframe=df)
            >>> # Filter rows
            >>> result = table.query("SELECT * FROM __table__ WHERE age > 25")
            >>>
            >>> # Aggregate data
            >>> summary = table.query("SELECT COUNT(*), AVG(salary) FROM __table__")
            >>>
            >>> # Complex operations
            >>> processed = table.query(
            ...     "SELECT name, age * 2 as double_age FROM __table__ ORDER BY age DESC"
            ... )
        """
        self._validate_sql(sql)

        if (
            self._parquet_path is not None
        ):  # if we have parquet file path, run chdb query on it directly is faster
            return self._query_on_path(self._parquet_path, sql, **kwargs)
        elif self._temp_parquet_path is not None:
            return self._query_on_path(self._temp_parquet_path, sql, **kwargs)
        elif self._parquet_memoryview is not None:
            return self.queryParquetBuffer(sql, **kwargs)
        elif self._dataframe is not None:
            return self.queryDF(sql, **kwargs)
        elif self._arrow_table is not None:
            return self.queryArrowTable(sql, **kwargs)
        else:
            raise ValueError("Table object is not initialized correctly")

    # alias sql = query
    sql = query

    def show(self):
        """Display the Table data by printing the pandas DataFrame representation.

        This is a convenience method for quickly viewing the table contents.
        Equivalent to print(table.to_pandas()).

        Example:
            >>> table = Table(dataframe=df)
            >>> table.show()
               id    name
            0   1   Alice
            1   2     Bob
        """
        print(self.to_pandas())

    def _query_on_path(self, path, sql, **kwargs):
        new_sql = sql.replace("__table__", f'file("{path}", Parquet)')
        res = chdb_query(new_sql, "Parquet", **kwargs)
        tbl = Table(parquet_memoryview=res.get_memview())
        tbl._rows_read = res.rows_read()
        tbl._bytes_read = res.bytes_read()
        tbl._elapsed = res.elapsed()
        return tbl

    def _validate_sql(self, sql):
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")

    def queryParquetBuffer(self, sql: str, **kwargs) -> "Table":
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")
        if self._parquet_memoryview is None:
            raise ValueError("Parquet buffer is None")

        temp_path = None
        parquet_fd = -1
        if self.use_memfd:
            parquet_fd = memfd_create("parquet_buffer")
        # if memfd_create failed, use tempfile to create a file descriptor for the memoryview
        if parquet_fd == -1:
            parquet_fd, temp_path = tempfile.mkstemp()
        ffd = os.fdopen(parquet_fd, "wb")
        ffd.write(self._parquet_memoryview.tobytes())
        ffd.flush()
        ret = self._run_on_temp(parquet_fd, temp_path, sql=sql, fmt="Parquet", **kwargs)
        ffd.close()
        if temp_path is not None:
            os.remove(temp_path)
        return ret

    def queryArrowTable(self, sql: str, **kwargs) -> "Table":
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")
        if self._arrow_table is None:
            raise ValueError("Arrow table is None")

        temp_path = None
        arrow_fd = -1
        if self.use_memfd:
            arrow_fd = memfd_create("arrow_buffer")
        if arrow_fd == -1:
            arrow_fd, temp_path = tempfile.mkstemp()
        ffd = os.fdopen(arrow_fd, "wb")
        with pa.RecordBatchFileWriter(ffd, self._arrow_table.schema) as writer:
            writer.write_table(self._arrow_table)
        ffd.flush()
        ret = self._run_on_temp(arrow_fd, temp_path, sql=sql, fmt="Arrow", **kwargs)
        ffd.close()
        if temp_path is not None:
            os.remove(temp_path)
        return ret

    def queryDF(self, sql: str, **kwargs) -> "Table":
        if "__table__" not in sql:
            raise ValueError("SQL should always contain `FROM __table__`")
        if self._dataframe is None:
            raise ValueError("Dataframe is None")

        temp_path = None
        parquet_fd = -1
        if self.use_memfd:
            parquet_fd = memfd_create()
        if parquet_fd == -1:
            parquet_fd, temp_path = tempfile.mkstemp()
        ffd = os.fdopen(parquet_fd, "wb")
        self._dataframe.to_parquet(ffd, engine="pyarrow", compression=None)
        ffd.flush()
        ret = self._run_on_temp(parquet_fd, temp_path, sql=sql, fmt="Parquet", **kwargs)
        ffd.close()
        if temp_path is not None:
            os.remove(temp_path)
        return ret

    @staticmethod
    def queryStatic(sql: str, **kwargs) -> "Table":
        """Execute SQL query across multiple Table objects.

        This static method enables complex queries involving multiple tables by referencing
        them as '__tablename__' in the SQL and passing them as keyword arguments.

        Args:
            sql (str): SQL query with table references as '__name__' patterns
            **kwargs: Table objects referenced in the SQL, where key matches the table name
                     Can also include pandas DataFrames, which will be auto-converted to Tables

        Returns:
            Table: New Table object containing the query results

        Raises:
            ValueError: If referenced table names are missing from kwargs or have invalid types

        Examples:
            >>> users = Table(dataframe=users_df)
            >>> orders = Table(dataframe=orders_df)
            >>>
            >>> # Join two tables
            >>> result = Table.queryStatic(
            ...     "SELECT u.name, COUNT(o.id) as order_count "
            ...     "FROM __users__ u LEFT JOIN __orders__ o ON u.id = o.user_id "
            ...     "GROUP BY u.name",
            ...     users=users, orders=orders
            ... )
            >>>
            >>> # Works with pandas DataFrames directly
            >>> result = Table.queryStatic(
            ...     "SELECT * FROM __df1__ UNION ALL SELECT * FROM __df2__",
            ...     df1=dataframe1, df2=dataframe2
            ... )
            >>>
            >>> # Complex multi-table operations
            >>> analytics = Table.queryStatic(
            ...     "SELECT p.category, AVG(o.amount) as avg_order "
            ...     "FROM __products__ p "
            ...     "JOIN __order_items__ oi ON p.id = oi.product_id "
            ...     "JOIN __orders__ o ON oi.order_id = o.id "
            ...     "GROUP BY p.category ORDER BY avg_order DESC",
            ...     products=products_table,
            ...     order_items=order_items_table,
            ...     orders=orders_table
            ... )
        """
        ansiTablePattern = re.compile(r"__([a-zA-Z][a-zA-Z0-9_]*)__")
        temp_paths = []
        ffds = []

        def replace_table_name(match):
            tableName = match.group(1)
            if tableName not in kwargs:
                raise ValueError(f"Table {tableName} should be passed as a parameter")

            tbl = kwargs[tableName]
            # if tbl is DataFrame, convert it to Table
            if isinstance(tbl, pd.DataFrame):
                tbl = Table(dataframe=tbl)
            elif not isinstance(tbl, Table):
                raise ValueError(
                    f"Table {tableName} should be an instance of Table or DataFrame")

            if tbl._parquet_path is not None:
                return f'file("{tbl._parquet_path}", Parquet)'

            if tbl._temp_parquet_path is not None:
                return f'file("{tbl._temp_parquet_path}", Parquet)'

            temp_path = None
            data_fd = -1

            if tbl.use_memfd:
                data_fd = memfd_create()

            if data_fd == -1:
                data_fd, temp_path = tempfile.mkstemp()
                temp_paths.append(temp_path)

            ffd = os.fdopen(data_fd, "wb")
            ffds.append(ffd)

            if tbl._parquet_memoryview is not None:
                ffd.write(tbl._parquet_memoryview.tobytes())
                ffd.flush()
                os.lseek(data_fd, 0, os.SEEK_SET)
                return f'file("/dev/fd/{data_fd}", Parquet)'

            if tbl._dataframe is not None:
                ffd.write(tbl._dataframe.to_parquet(engine="pyarrow", compression=None))
                ffd.flush()
                os.lseek(data_fd, 0, os.SEEK_SET)
                return f'file("/dev/fd/{data_fd}", Parquet)'

            if tbl._arrow_table is not None:
                with pa.RecordBatchFileWriter(ffd, tbl._arrow_table.schema) as writer:
                    writer.write_table(tbl._arrow_table)
                ffd.flush()
                os.lseek(data_fd, 0, os.SEEK_SET)
                return f'file("/dev/fd/{data_fd}", Arrow)'

            raise ValueError(f"Table {tableName} is not initialized correctly")

        sql = ansiTablePattern.sub(replace_table_name, sql)
        res = chdb_query(sql, "Parquet")

        for fd in ffds:
            fd.close()

        for tmp_path in temp_paths:
            os.remove(tmp_path)

        tbl = Table(parquet_memoryview=res.get_memview())
        tbl._rows_read = res.rows_read()
        tbl._bytes_read = res.bytes_read()
        tbl._elapsed = res.elapsed()
        return tbl

    def _run_on_temp(
        self,
        fd: int,
        temp_path: str = None,
        sql: str = None,
        fmt: str = "Parquet",
        **kwargs,
    ) -> "Table":
        # replace "__table__" with file("temp_path", Parquet) or file("/dev/fd/{parquet_fd}", Parquet)
        if temp_path is not None:
            new_sql = sql.replace("__table__", f'file("{temp_path}", {fmt})')
        else:
            os.lseek(fd, 0, os.SEEK_SET)
            new_sql = sql.replace("__table__", f'file("/dev/fd/{fd}", {fmt})')
        res = chdb_query(new_sql, "Parquet", **kwargs)
        tbl = Table(parquet_memoryview=res.get_memview())
        tbl._rows_read = res.rows_read()
        tbl._bytes_read = res.bytes_read()
        tbl._elapsed = res.elapsed()
        return tbl


def pandas_read_parquet(path) -> pd.DataFrame:
    """Read a Parquet file into a pandas DataFrame.

    This is a convenience wrapper around pandas.read_parquet() for consistency
    with the chdb.dataframe module interface.

    Args:
        path: File path or file-like object to read from

    Returns:
        pd.DataFrame: The loaded DataFrame
    """
    return pd.read_parquet(path)


def memfd_create(name: str = None) -> int:
    """Create an in-memory file descriptor using memfd_create system call.

    This function attempts to use the Linux-specific memfd_create(2) system call
    to create a file descriptor that refers to an anonymous memory-backed file.
    This provides better performance for temporary data operations.

    Args:
        name (str, optional): Name for the memory file (for debugging). Defaults to None.

    Returns:
        int: File descriptor on success, -1 on failure or if not supported

    Note:
        This function only works on Linux 3.17 or newer with glibc 2.27 or newer.
        On other systems or if the call fails, it returns -1 and callers should
        fall back to regular temporary files.

    Example:
        >>> fd = memfd_create("temp_data")
        >>> if fd != -1:
        ...     # Use memory-based file descriptor
        ...     with os.fdopen(fd, 'wb') as f:
        ...         f.write(data)
        ... else:
        ...     # Fall back to regular temp file
        ...     fd, path = tempfile.mkstemp()
    """
    if hasattr(os, "memfd_create"):
        try:
            fd = os.memfd_create(name, flags=os.MFD_CLOEXEC)
            return fd
        except:  # noqa
            return -1
    return -1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SQL on parquet file")
    parser.add_argument("parquet_path", type=str, help="path to parquet file")
    parser.add_argument("sql", type=str, help="SQL to run")
    parser.add_argument(
        "--use-memfd",
        action="store_true",
        help="use memfd_create to create file descriptor",
    )
    args = parser.parse_args()

    table = Table(parquet_path=args.parquet_path, use_memfd=args.use_memfd)
    print(table.query(args.sql))
