"""
DataStore - A Pandas-like Data Manipulation Framework
======================================================

DataStore provides a high-level API for data manipulation with automatic
query generation and execution capabilities.

Key Features:
- Fluent API similar to Pandas/Polars
- Automatic SQL generation
- Multiple data source support (File, S3, MySQL, PostgreSQL, etc.)
- ClickHouse table functions integration
- Immutable operations for thread safety

Example:
    >>> from chdb_ds import DataStore
    >>>
    >>> # Simplest way: Use URI with automatic type inference
    >>> ds = DataStore.uri("/path/to/data.csv")
    >>> ds.connect()
    >>> result = ds.select("name", "age").filter(ds.age > 18).execute()
    >>>
    >>> # S3 with URI
    >>> ds = DataStore.uri("s3://bucket/data.parquet?nosign=true")
    >>> result = ds.select("*").execute()
    >>>
    >>> # MySQL with URI
    >>> ds = DataStore.uri("mysql://root:pass@localhost:3306/mydb/users")
    >>> result = ds.select("*").filter(ds.age > 18).execute()
    >>>
    >>> # Traditional way: Query a local file (auto-detect format)
    >>> ds = DataStore("file", path="data.parquet")
    >>> ds.connect()
    >>> result = ds.select("name", "age").filter(ds.age > 18).execute()
    >>>
    >>> # Traditional way: Query a local file (explicit format)
    >>> ds = DataStore("file", path="data.csv", format="CSV")
    >>> result = ds.select("name", "age").filter(ds.age > 18).execute()
    >>>
    >>> # Traditional way: Query S3 data (auto-detect format)
    >>> ds = DataStore("s3", url="s3://bucket/data.parquet", nosign=True)
    >>> result = ds.select("*").execute()
    >>>
    >>> # Traditional way: Query S3 data (with credentials and explicit format)
    >>> ds = DataStore("s3", url="s3://bucket/data.parquet",
    ...                access_key_id="KEY", secret_access_key="SECRET",
    ...                format="Parquet")
    >>> result = ds.select("*").execute()
    >>>
    >>> # Traditional way: Query MySQL
    >>> ds = DataStore("mysql", host="localhost:3306",
    ...                database="mydb", table="users",
    ...                user="root", password="pass")
    >>> result = ds.select("*").filter(ds.age > 18).execute()

Core Classes:
- DataStore: Main entry point for data operations
- Expression: Base class for all expressions
- Function: SQL function wrapper
- Connection: Database connection abstraction
- TableFunction: ClickHouse table function wrappers
"""

from .core import DataStore
from .expressions import Expression, Field, Literal, col
from .column_expr import ColumnExpr
from .lazy_result import LazySlice
from .functions import (
    Function,
    AggregateFunction,
    WindowFunction,
    CustomFunction,
    CastFunction,
    F,  # Function namespace for explicit function calls
    # Common functions
    Sum,
    Count,
    Avg,
    Min,
    Max,
    Upper,
    Lower,
    Concat,
)

# Function Registry - Single Source of Truth for function definitions
from .function_registry import (
    FunctionRegistry,
    FunctionType,
    FunctionCategory,
    FunctionSpec,
    register_function,
)

# Import function definitions to register all functions
from . import function_definitions as _function_definitions

_function_definitions.ensure_functions_registered()
from .function_executor import (  # noqa: E402
    FunctionExecutorConfig,
    ExecutionEngine,
    function_config,
    use_chdb,
    use_pandas,
    prefer_chdb,
    prefer_pandas,
    reset_function_config,
)
from .accessors import (  # noqa: E402
    StringAccessor,
    DateTimeAccessor,
    ArrayAccessor,
    JsonAccessor,
    UrlAccessor,
    IpAccessor,
    GeoAccessor,
)
from .conditions import Condition, BinaryCondition  # noqa: E402
from .connection import Connection, QueryResult  # noqa: E402
from .executor import Executor, get_executor, reset_executor  # noqa: E402
from .exceptions import (  # noqa: E402
    DataStoreError,
    ConnectionError,
    SchemaError,
    QueryError,
    ExecutionError,
)
from .enums import JoinType  # noqa: E402
from .config import (  # noqa: E402
    config,
    set_log_level,
    set_log_format,
    enable_debug,
    disable_debug,
    get_logger,
)
from .table_functions import (  # noqa: E402
    TableFunction,
    create_table_function,
    FileTableFunction,
    UrlTableFunction,
    S3TableFunction,
    AzureBlobStorageTableFunction,
    GcsTableFunction,
    HdfsTableFunction,
    MySQLTableFunction,
    PostgreSQLTableFunction,
    MongoDBTableFunction,
    RedisTableFunction,
    SQLiteTableFunction,
    RemoteTableFunction,
    IcebergTableFunction,
    DeltaLakeTableFunction,
    HudiTableFunction,
    NumbersTableFunction,
    GenerateRandomTableFunction,
)

# ========== Pandas-Compatible Top-Level Functions ==========
# These functions provide pandas-like API for reading data


def read_csv(filepath_or_buffer, sep=',', **kwargs) -> 'DataStore':
    """
    Read a comma-separated values (CSV) file into DataStore.

    This function provides full pandas.read_csv() compatibility by using pandas
    internally. For high-performance reading of simple CSV files, consider
    using DataStore.from_file() directly.

    Args:
        filepath_or_buffer: Path to the CSV file, URL, or file-like object
        sep: Delimiter to use (default ',')
        **kwargs: All pandas.read_csv() arguments are supported, including:
            - header: Row number to use as column names (default 'infer')
            - names: List of column names to use
            - index_col: Column(s) to use as row labels
            - usecols: Return a subset of columns
            - dtype: Data type for columns
            - skiprows: Number of rows to skip at the beginning
            - nrows: Number of rows to read
            - na_values: Additional strings to recognize as NA/NaN
            - parse_dates: Columns to parse as dates
            - encoding: Encoding to use for reading (default 'utf-8')
            - compression: Compression type ('infer', 'gzip', 'bz2', etc.)
            - quotechar: Character used to denote quoted strings
            - escapechar: Character used to escape other characters
            - comment: Character indicating comment lines
            - thousands: Thousands separator
            - decimal: Character for decimal point

    Returns:
        DataStore: A DataStore object containing the CSV data

    Example:
        >>> from datastore import read_csv
        >>> df = read_csv("data.csv")
        >>> df.head()

        >>> # With options (full pandas compatibility)
        >>> df = read_csv("data.csv", sep=";", skiprows=1, encoding='latin-1')
        >>> df = read_csv("data.csv", usecols=['name', 'age'], dtype={'age': int})
        >>> df = read_csv("data.csv", parse_dates=['date_col'])
    """
    import pandas as pd

    pandas_df = pd.read_csv(filepath_or_buffer, sep=sep, **kwargs)
    return DataStore.from_df(pandas_df)


def read_parquet(path, columns=None, **kwargs) -> 'DataStore':
    """
    Read a Parquet file into DataStore.

    This function provides pandas.read_parquet() compatibility.

    Args:
        path: Path to the Parquet file, URL, or file-like object
        columns: List of column names to read (None reads all columns)
        **kwargs: Additional pandas.read_parquet() arguments:
            - engine: Parquet library to use ('auto', 'pyarrow', 'fastparquet')
            - use_nullable_dtypes: Use nullable dtypes for pandas 1.0+
            - filters: List of filters for row group filtering

    Returns:
        DataStore: A DataStore object containing the Parquet data

    Example:
        >>> from datastore import read_parquet
        >>> df = read_parquet("data.parquet")
        >>> df = read_parquet("data.parquet", columns=['name', 'age'])
    """
    import pandas as pd

    pandas_df = pd.read_parquet(path, columns=columns, **kwargs)
    return DataStore.from_df(pandas_df)


def read_json(path_or_buf, orient=None, lines=False, **kwargs) -> 'DataStore':
    """
    Read a JSON file into DataStore.

    This function provides full pandas.read_json() compatibility.

    Args:
        path_or_buf: Path to the JSON file, URL, or file-like object
        orient: Expected JSON string format. Compatible values are:
            - 'split': dict like {index -> [index], columns -> [columns], data -> [values]}
            - 'records': list like [{column -> value}, ... , {column -> value}]
            - 'index': dict like {index -> {column -> value}}
            - 'columns': dict like {column -> {index -> value}}
            - 'values': just the values array
        lines: Read the file as JSON Lines (one JSON object per line)
        **kwargs: Additional pandas.read_json() arguments:
            - typ: Type of object to recover ('frame' or 'series')
            - dtype: Data types for columns
            - convert_axes: Try to convert axes to proper dtypes
            - convert_dates: Parse date columns
            - precise_float: Use higher precision float parsing
            - encoding: Encoding for reading
            - compression: Compression type

    Returns:
        DataStore: A DataStore object containing the JSON data

    Example:
        >>> from datastore import read_json
        >>> df = read_json("data.json")
        >>> df = read_json("data.json", orient='records')

        >>> # JSON Lines format
        >>> df = read_json("data.jsonl", lines=True)
    """
    import pandas as pd

    pandas_df = pd.read_json(path_or_buf, orient=orient, lines=lines, **kwargs)
    return DataStore.from_df(pandas_df)


def read_excel(io, sheet_name=0, **kwargs) -> 'DataStore':
    """
    Read an Excel file into DataStore.

    Note: This reads the Excel file via pandas and wraps it in DataStore.

    Args:
        io: Path to the Excel file
        sheet_name: Sheet name or index to read (default 0)
        **kwargs: Additional arguments passed to pandas.read_excel()

    Returns:
        DataStore: A DataStore object containing the Excel data

    Example:
        >>> from datastore import read_excel
        >>> df = read_excel("data.xlsx")
        >>> df = read_excel("data.xlsx", sheet_name="Sheet2")
    """
    import pandas as pd

    pandas_df = pd.read_excel(io, sheet_name=sheet_name, **kwargs)
    return DataStore.from_df(pandas_df)


def read_sql(sql, con, **kwargs) -> 'DataStore':
    """
    Read SQL query into DataStore.

    Note: This executes the SQL via pandas and wraps it in DataStore.

    Args:
        sql: SQL query string or table name
        con: Database connection (SQLAlchemy engine, connection string, etc.)
        **kwargs: Additional arguments passed to pandas.read_sql()

    Returns:
        DataStore: A DataStore object containing the SQL result

    Example:
        >>> from datastore import read_sql
        >>> df = read_sql("SELECT * FROM users", engine)
    """
    import pandas as pd

    pandas_df = pd.read_sql(sql, con, **kwargs)
    return DataStore.from_df(pandas_df)


def read_table(filepath_or_buffer, sep='\t', **kwargs) -> 'DataStore':
    """
    Read general delimited file into DataStore.

    This is similar to read_csv but with tab ('\\t') as the default delimiter.
    Uses pandas.read_table internally for full compatibility.

    Args:
        filepath_or_buffer: Path to the file or file-like object
        sep: Delimiter to use (default '\\t' for tab)
        **kwargs: Additional arguments passed to pandas.read_table()
            - header: Row number to use as column names
            - names: List of column names to use
            - usecols: Return a subset of columns
            - dtype: Data type for columns
            - skiprows: Number of rows to skip
            - nrows: Number of rows to read
            - na_values: Additional strings to recognize as NA/NaN
            - encoding: Encoding to use for reading

    Returns:
        DataStore: A DataStore object containing the data

    Example:
        >>> from datastore import read_table
        >>> df = read_table("data.tsv")  # Tab-separated (default)
        >>> df = read_table("data.txt", sep="|")  # Pipe-separated
    """
    import pandas as pd

    pandas_df = pd.read_table(filepath_or_buffer, sep=sep, **kwargs)
    return DataStore.from_df(pandas_df)


def read_feather(path, **kwargs) -> 'DataStore':
    """
    Read a Feather file into DataStore.

    Note: This reads via pandas and wraps in DataStore.

    Args:
        path: Path to the Feather file
        **kwargs: Additional arguments passed to pandas.read_feather()

    Returns:
        DataStore: A DataStore object containing the Feather data

    Example:
        >>> from datastore import read_feather
        >>> df = read_feather("data.feather")
    """
    import pandas as pd

    pandas_df = pd.read_feather(path, **kwargs)
    return DataStore.from_df(pandas_df)


def read_orc(path, columns=None, **kwargs) -> 'DataStore':
    """
    Read an ORC file into DataStore.

    This function provides pandas.read_orc() compatibility.

    Args:
        path: Path to the ORC file
        columns: List of column names to read (None reads all columns)
        **kwargs: Additional pandas.read_orc() arguments

    Returns:
        DataStore: A DataStore object containing the ORC data

    Example:
        >>> from datastore import read_orc
        >>> df = read_orc("data.orc")
        >>> df = read_orc("data.orc", columns=['name', 'age'])
    """
    import pandas as pd

    pandas_df = pd.read_orc(path, columns=columns, **kwargs)
    return DataStore.from_df(pandas_df)


def read_pickle(filepath_or_buffer, **kwargs) -> 'DataStore':
    """
    Read a pickled pandas DataFrame into DataStore.

    Note: This reads via pandas and wraps in DataStore.

    Args:
        filepath_or_buffer: Path to the pickle file
        **kwargs: Additional arguments passed to pandas.read_pickle()

    Returns:
        DataStore: A DataStore object containing the pickled data

    Example:
        >>> from datastore import read_pickle
        >>> df = read_pickle("data.pkl")
    """
    import pandas as pd

    pandas_df = pd.read_pickle(filepath_or_buffer, **kwargs)
    return DataStore.from_df(pandas_df)


def concat(objs, axis=0, join='outer', ignore_index=False, keys=None, **kwargs) -> 'DataStore':
    """
    Concatenate DataStore/DataFrame objects along a particular axis.

    Args:
        objs: Sequence of DataStore or DataFrame objects to concatenate
        axis: The axis to concatenate along (default 0)
        join: How to handle indexes on other axis ('outer' or 'inner')
        ignore_index: If True, do not use index values along concatenation axis
        keys: Sequence to use as keys for hierarchical index
        **kwargs: Additional arguments passed to pandas.concat()

    Returns:
        DataStore: A DataStore object containing the concatenated data

    Example:
        >>> from datastore import concat
        >>> result = concat([df1, df2, df3])
        >>> result = concat([df1, df2], axis=1)
    """
    import pandas as pd

    # Convert DataStore objects to DataFrames
    dfs = []
    for obj in objs:
        if hasattr(obj, 'to_df'):
            dfs.append(obj.to_df())
        else:
            dfs.append(obj)

    result = pd.concat(dfs, axis=axis, join=join, ignore_index=ignore_index, keys=keys, **kwargs)
    return DataStore.from_df(result)


def merge(
    left,
    right,
    how='inner',
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    suffixes=('_x', '_y'),
    **kwargs,
) -> 'DataStore':
    """
    Merge DataStore/DataFrame objects with a database-style join.

    Args:
        left: Left DataStore or DataFrame
        right: Right DataStore or DataFrame
        how: Type of merge ('left', 'right', 'outer', 'inner', 'cross')
        on: Column or index level names to join on
        left_on: Column(s) from left to use as keys
        right_on: Column(s) from right to use as keys
        left_index: Use index from left as join key
        right_index: Use index from right as join key
        suffixes: Suffix to apply to overlapping columns
        **kwargs: Additional arguments passed to pandas.merge()

    Returns:
        DataStore: A DataStore object containing the merged data

    Example:
        >>> from datastore import merge
        >>> result = merge(df1, df2, on='id')
        >>> result = merge(df1, df2, left_on='user_id', right_on='id')
    """
    import pandas as pd

    # Convert to DataFrames
    left_df = left.to_df() if hasattr(left, 'to_df') else left
    right_df = right.to_df() if hasattr(right, 'to_df') else right

    result = pd.merge(
        left_df,
        right_df,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        suffixes=suffixes,
        **kwargs,
    )
    return DataStore.from_df(result)


__version__ = "0.1.0"
__author__ = "DataStore Contributors"

__all__ = [
    # Core
    'DataStore',
    # Pandas-Compatible IO Functions
    'read_csv',
    'read_parquet',
    'read_json',
    'read_excel',
    'read_sql',
    'read_table',
    'read_feather',
    'read_orc',
    'read_pickle',
    # Pandas-Compatible Data Manipulation Functions
    'concat',
    'merge',
    # Expressions
    'Expression',
    'Field',
    'Literal',
    'ColumnExpr',
    'LazySlice',
    'col',
    # Functions
    'Function',
    'AggregateFunction',
    'WindowFunction',
    'CustomFunction',
    'CastFunction',
    'F',  # Function namespace
    'Sum',
    'Count',
    'Avg',
    'Min',
    'Max',
    'Upper',
    'Lower',
    'Concat',
    # Function Registry
    'FunctionRegistry',
    'FunctionType',
    'FunctionCategory',
    'FunctionSpec',
    'register_function',
    # Accessors (for advanced use)
    'StringAccessor',
    'DateTimeAccessor',
    'ArrayAccessor',
    'JsonAccessor',
    'UrlAccessor',
    'IpAccessor',
    'GeoAccessor',
    # Function Executor Config
    'FunctionExecutorConfig',
    'ExecutionEngine',
    'function_config',
    'use_chdb',
    'use_pandas',
    'prefer_chdb',
    'prefer_pandas',
    'reset_function_config',
    # Conditions
    'Condition',
    'BinaryCondition',
    # Enums
    'JoinType',
    # Connection and Execution
    'Connection',
    'QueryResult',
    'Executor',
    'get_executor',
    'reset_executor',
    # Exceptions
    'DataStoreError',
    'ConnectionError',
    'SchemaError',
    'QueryError',
    'ExecutionError',
    # Configuration
    'config',
    'set_log_level',
    'set_log_format',
    'enable_debug',
    'disable_debug',
    'get_logger',
    # Table Functions
    'TableFunction',
    'create_table_function',
    'FileTableFunction',
    'UrlTableFunction',
    'S3TableFunction',
    'AzureBlobStorageTableFunction',
    'GcsTableFunction',
    'HdfsTableFunction',
    'MySQLTableFunction',
    'PostgreSQLTableFunction',
    'MongoDBTableFunction',
    'RedisTableFunction',
    'SQLiteTableFunction',
    'RemoteTableFunction',
    'IcebergTableFunction',
    'DeltaLakeTableFunction',
    'HudiTableFunction',
    'NumbersTableFunction',
    'GenerateRandomTableFunction',
]
