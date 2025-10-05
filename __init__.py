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
    >>> # Query a local file
    >>> ds = DataStore("file", path="data.csv", format="CSV")
    >>> ds.connect()
    >>> result = ds.select("name", "age").filter(ds.age > 18).execute()
    >>>
    >>> # Query S3 data
    >>> ds = DataStore("s3", path="s3://bucket/data.parquet",
    ...                access_key_id="KEY", secret_access_key="SECRET",
    ...                format="Parquet")
    >>> result = ds.select("*").execute()
    >>>
    >>> # Query MySQL
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
from .expressions import Expression, Field, Literal
from .functions import (
    Function,
    AggregateFunction,
    CustomFunction,
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
from .conditions import Condition, BinaryCondition
from .connection import Connection, QueryResult
from .executor import Executor
from .exceptions import (
    DataStoreError,
    ConnectionError,
    SchemaError,
    QueryError,
    ExecutionError,
)
from .enums import JoinType
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
    # Functions
    'Function',
    'AggregateFunction',
    'CustomFunction',
    'Sum',
    'Count',
    'Avg',
    'Min',
    'Max',
    'Upper',
    'Lower',
    'Concat',
    # Conditions
    'Condition',
    'BinaryCondition',
    # Enums
    'JoinType',
    # Connection and Execution
    'Connection',
    'QueryResult',
    'Executor',
    # Exceptions
    'DataStoreError',
    'ConnectionError',
    'SchemaError',
    'QueryError',
    'ExecutionError',
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
