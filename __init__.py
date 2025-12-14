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
from .function_executor import (
    FunctionExecutorConfig,
    ExecutionEngine,
    function_config,
    use_chdb,
    use_pandas,
    prefer_chdb,
    prefer_pandas,
    reset_function_config,
)
from .accessors import (
    StringAccessor,
    DateTimeAccessor,
)
from .conditions import Condition, BinaryCondition
from .connection import Connection, QueryResult
from .executor import Executor, get_executor, reset_executor
from .exceptions import (
    DataStoreError,
    ConnectionError,
    SchemaError,
    QueryError,
    ExecutionError,
)
from .enums import JoinType
from .config import (
    config,
    set_log_level,
    set_log_format,
    enable_debug,
    disable_debug,
    get_logger,
)
from .table_functions import (
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

__version__ = "0.1.0"
__author__ = "DataStore Contributors"

__all__ = [
    # Core
    'DataStore',
    # Expressions
    'Expression',
    'Field',
    'Literal',
    'ColumnExpr',
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
