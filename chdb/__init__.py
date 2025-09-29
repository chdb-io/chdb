import sys
import os
import threading


class ChdbError(Exception):
    """Base exception class for chDB-related errors.

    This exception is raised when chDB query execution fails or encounters
    an error. It inherits from the standard Python Exception class and
    provides error information from the underlying ClickHouse engine.

    The exception message typically contains detailed error information
    from ClickHouse, including syntax errors, type mismatches, missing
    tables/columns, and other query execution issues.

    Attributes:
        args: Tuple containing the error message and any additional arguments

    Examples:
        >>> try:
        ...     result = chdb.query("SELECT * FROM non_existent_table")
        ... except chdb.ChdbError as e:
        ...     print(f"Query failed: {e}")
        Query failed: Table 'non_existent_table' doesn't exist

        >>> try:
        ...     result = chdb.query("SELECT invalid_syntax FROM")
        ... except chdb.ChdbError as e:
        ...     print(f"Syntax error: {e}")
        Syntax error: Syntax error near 'FROM'

    Note:
        This exception is automatically raised by chdb.query() and related
        functions when the underlying ClickHouse engine reports an error.
        You should catch this exception when handling potentially failing
        queries to provide appropriate error handling in your application.
    """


_arrow_format = set({"arrowtable"})
_process_result_format_funs = {
    "arrowtable": lambda x: to_arrowTable(x),
}

# If any UDF is defined, the path of the UDF will be set to this variable
# and the path will be deleted when the process exits
# UDF config path will be f"{g_udf_path}/udf_config.xml"
# UDF script path will be f"{g_udf_path}/{func_name}.py"
g_udf_path = ""

__version__ = "3.6.0"
if sys.version_info[:2] >= (3, 7):
    # get the path of the current file
    current_path = os.path.dirname(os.path.abspath(__file__))
    # change the current working directory to the path of the current file
    # and import _chdb then change the working directory back
    cwd = os.getcwd()
    os.chdir(current_path)
    from . import _chdb  # noqa

    os.chdir(cwd)
    conn = _chdb.connect()
    engine_version = str(conn.query("SELECT version();", "CSV").bytes())[3:-4]
    conn.close()
else:
    raise NotImplementedError("Python 3.6 or lower version is not supported")

chdb_version = tuple(__version__.split('.'))


# return pyarrow table
def to_arrowTable(res):
    """Convert query result to PyArrow Table.

    Converts a chDB query result to a PyArrow Table for efficient columnar data processing.
    Returns an empty table if the result is empty.

    Args:
        res: chDB query result object containing binary Arrow data

    Returns:
        pa.Table: PyArrow Table containing the query results

    Raises:
        ImportError: If pyarrow or pandas are not installed

    Example:
        >>> result = chdb.query("SELECT 1 as id, 'hello' as msg", "Arrow")
        >>> table = chdb.to_arrowTable(result)
        >>> print(table.to_pandas())
           id    msg
        0   1  hello
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
    return pa.RecordBatchFileReader(res.bytes()).read_all()


# global connection lock, for multi-threading use of legacy chdb.query()
g_conn_lock = threading.Lock()


# wrap _chdb functions
def query(sql, output_format="CSV", path="", udf_path=""):
    """Execute SQL query using chDB engine.

    This is the main query function that executes SQL statements using the embedded
    ClickHouse engine. Supports various output formats and can work with in-memory
    or file-based databases.

    Args:
        sql (str): SQL query string to execute
        output_format (str, optional): Output format for results. Defaults to "CSV".
            Supported formats include:

            - "CSV" - Comma-separated values
            - "JSON" - JSON format
            - "Arrow" - Apache Arrow format
            - "Parquet" - Parquet format
            - "DataFrame" - Pandas DataFrame
            - "ArrowTable" - PyArrow Table
            - "Debug" - Enable verbose logging

        path (str, optional): Database file path. Defaults to "" (in-memory database).
            Can be a file path or ":memory:" for in-memory database.
        udf_path (str, optional): Path to User-Defined Functions directory. Defaults to "".

    Returns:
        Query result in the specified format:

        - str: For text formats like CSV, JSON
        - pd.DataFrame: When output_format is "DataFrame" or "dataframe"
        - pa.Table: When output_format is "ArrowTable" or "arrowtable"
        - chdb result object: For other formats

    Raises:
        ChdbError: If the SQL query execution fails
        ImportError: If required dependencies are missing for DataFrame/Arrow formats

    Examples:
        >>> # Basic CSV query
        >>> result = chdb.query("SELECT 1, 'hello'")
        >>> print(result)
        "1,hello"

        >>> # Query with DataFrame output
        >>> df = chdb.query("SELECT 1 as id, 'hello' as msg", "DataFrame")
        >>> print(df)
           id    msg
        0   1  hello

        >>> # Query with file-based database
        >>> result = chdb.query("CREATE TABLE test (id INT)", path="mydb.chdb")

        >>> # Query with UDF
        >>> result = chdb.query("SELECT my_udf('test')", udf_path="/path/to/udfs")
    """
    global g_udf_path
    if udf_path != "":
        g_udf_path = udf_path
    conn_str = ""
    if path == "":
        conn_str = ":memory:"
    else:
        conn_str = f"{path}"
    if g_udf_path != "":
        if "?" in conn_str:
            conn_str = f"{conn_str}&udf_path={g_udf_path}"
        else:
            conn_str = f"{conn_str}?udf_path={g_udf_path}"
    if output_format == "Debug":
        output_format = "CSV"
        if "?" in conn_str:
            conn_str = f"{conn_str}&verbose&log-level=test"
        else:
            conn_str = f"{conn_str}?verbose&log-level=test"

    lower_output_format = output_format.lower()
    result_func = _process_result_format_funs.get(lower_output_format, lambda x: x)
    if lower_output_format in _arrow_format:
        output_format = "Arrow"

    with g_conn_lock:
        conn = _chdb.connect(conn_str)
        res = conn.query(sql, output_format)
        if res.has_error():
            conn.close()
            raise ChdbError(res.error_message())
        conn.close()
    return result_func(res)


# alias for query
sql = query

PyReader = _chdb.PyReader

from . import dbapi, session, udf, utils  # noqa: E402
from .state import connect  # noqa: E402

__all__ = [
    "_chdb",
    "PyReader",
    "ChdbError",
    "query",
    "sql",
    "chdb_version",
    "engine_version",
    "to_df",
    "to_arrowTable",
    "dbapi",
    "session",
    "udf",
    "utils",
    "connect",
]
