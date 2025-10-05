"""
ClickHouse Table Functions - Unified interface for different data sources.

This module provides Python wrappers for ClickHouse table functions,
supporting various data sources with their specific capabilities and settings.
"""

from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from .exceptions import DataStoreError


class TableFunction(ABC):
    """
    Base class for ClickHouse table functions.

    Each table function wraps a ClickHouse table function like file(), s3(), etc.
    """

    def __init__(self, **kwargs):
        """Initialize table function with parameters."""
        self.params = kwargs
        self.settings: Dict[str, Any] = {}

    @property
    @abstractmethod
    def can_read(self) -> bool:
        """Whether this table function supports reading."""
        pass

    @property
    @abstractmethod
    def can_write(self) -> bool:
        """Whether this table function supports writing."""
        pass

    @abstractmethod
    def to_sql(self, quote_char: str = '"') -> str:
        """Generate SQL for the table function."""
        pass

    def with_settings(self, **settings) -> 'TableFunction':
        """
        Add format-specific settings to the table function.

        Args:
            **settings: Format-specific settings (e.g., format_csv_delimiter='|')

        Returns:
            self for chaining
        """
        self.settings.update(settings)
        return self

    def _format_param(self, value: Any) -> str:
        """Format parameter value for SQL."""
        if value is None:
            return 'NULL'
        elif isinstance(value, bool):
            return '1' if value else '0'
        elif isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return f"'{str(value)}'"

    def _build_settings_clause(self) -> str:
        """Build SETTINGS clause if settings are present."""
        if not self.settings:
            return ""

        settings_parts = []
        for key, value in self.settings.items():
            if isinstance(value, str):
                settings_parts.append(f"{key}='{value}'")
            else:
                settings_parts.append(f"{key}={value}")

        return " SETTINGS " + ", ".join(settings_parts)


class FileTableFunction(TableFunction):
    """
    Wrapper for file() table function.

    Supports reading from and writing to local files.

    Parameters:
        path: File path relative to user_files_path (supports glob patterns)
        format: File format (optional, auto-detected if not provided)
        structure: Optional table structure 'column1 Type1, column2 Type2, ...'
        compression: Optional compression method (gzip, zstd, etc.)

    Example:
        >>> tf = FileTableFunction(path="data.csv", format="CSV",
        ...                        structure="id UInt32, name String")
        >>> tf.to_sql()
        "file('data.csv', 'CSV', 'id UInt32, name String')"

        >>> # Format auto-detection
        >>> tf = FileTableFunction(path="data.parquet")
        >>> tf.to_sql()
        "file('data.parquet')"
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        path = self.params.get('path')
        format_name = self.params.get('format')
        structure = self.params.get('structure')
        compression = self.params.get('compression')

        if not path:
            raise DataStoreError("'path' parameter is required for file()")

        sql_params = [self._format_param(path)]

        # Format is optional - chdb can auto-detect from file extension
        if format_name:
            sql_params.append(self._format_param(format_name))

            if structure:
                sql_params.append(self._format_param(structure))

            if compression:
                sql_params.append(self._format_param(compression))

        return f"file({', '.join(sql_params)})"


class UrlTableFunction(TableFunction):
    """
    Wrapper for url() table function.

    Supports reading from and writing to HTTP(S) URLs.

    Parameters:
        url: HTTP(S) URL to the data
        format: Input/output data format (optional, auto-detected if not provided)
        structure: Optional table structure
        headers: Optional HTTP headers as list of 'Key: Value' strings

    Example:
        >>> # Auto-detection from URL
        >>> tf = UrlTableFunction(url="https://example.com/data.json")
        >>> tf.to_sql()
        "url('https://example.com/data.json')"

        >>> # Explicit format
        >>> tf = UrlTableFunction(url="https://example.com/data.json",
        ...                       format="JSONEachRow")
        >>> tf.to_sql()
        "url('https://example.com/data.json', 'JSONEachRow')"
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        url = self.params.get('url')
        format_name = self.params.get('format')
        structure = self.params.get('structure')
        headers = self.params.get('headers')

        if not url:
            raise DataStoreError("'url' parameter is required for url()")

        sql_params = [self._format_param(url)]

        # Format is optional - chdb can auto-detect
        if format_name:
            sql_params.append(self._format_param(format_name))

            if structure:
                sql_params.append(self._format_param(structure))

        sql = f"url({', '.join(sql_params)})"

        # Add headers if provided
        if headers:
            if isinstance(headers, list):
                headers_sql = ', '.join([self._format_param(h) for h in headers])
            else:
                headers_sql = self._format_param(headers)
            sql += f" HEADERS({headers_sql})"

        return sql


class S3TableFunction(TableFunction):
    """
    Wrapper for s3() table function.

    Supports reading from and writing to Amazon S3 and Google Cloud Storage.

    Parameters:
        url: S3 URL (e.g., 'https://bucket.s3.region.amazonaws.com/path/file.csv')
        access_key_id: AWS access key (optional if NOSIGN)
        secret_access_key: AWS secret key (optional if NOSIGN)
        session_token: AWS session token (optional)
        format: Data format (optional, auto-detected if not provided)
        structure: Optional table structure
        compression: Optional compression method
        nosign: Use anonymous access (default: False)

    Example:
        >>> # Public bucket with auto-detection
        >>> tf = S3TableFunction(url="s3://bucket/data.parquet", nosign=True)

        >>> # With credentials and explicit format
        >>> tf = S3TableFunction(url="s3://bucket/data.csv",
        ...                      access_key_id="KEY",
        ...                      secret_access_key="SECRET",
        ...                      format="CSV")
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        url = self.params.get('url') or self.params.get('path')
        access_key = self.params.get('access_key_id')
        secret_key = self.params.get('secret_access_key')
        session_token = self.params.get('session_token')
        format_name = self.params.get('format')
        structure = self.params.get('structure')
        compression = self.params.get('compression')
        nosign = self.params.get('nosign', False)

        if not url:
            raise DataStoreError("'url' or 'path' parameter is required for s3()")

        sql_params = [self._format_param(url)]

        # Handle authentication
        if nosign:
            sql_params.append('NOSIGN')
        elif access_key and secret_key:
            sql_params.append(self._format_param(access_key))
            sql_params.append(self._format_param(secret_key))
            if session_token:
                sql_params.append(self._format_param(session_token))

        # Format is optional - chdb can auto-detect
        if format_name:
            sql_params.append(self._format_param(format_name))

            # Optional structure
            if structure:
                sql_params.append(self._format_param(structure))

            # Optional compression
            if compression:
                sql_params.append(self._format_param(compression))

        return f"s3({', '.join(sql_params)})"


class AzureBlobStorageTableFunction(TableFunction):
    """
    Wrapper for azureBlobStorage() table function.

    Supports reading from and writing to Azure Blob Storage.

    Parameters:
        connection_string: Azure connection string
        container: Container name
        path: Blob path (supports glob patterns)
        account_name: Storage account name (optional)
        account_key: Access key (optional)
        format: Data format (optional, auto-detected if not provided)
        structure: Optional table structure
        compression: Optional compression method
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        connection_string = self.params.get('connection_string')
        container = self.params.get('container')
        path = self.params.get('path', '')
        format_name = self.params.get('format')
        structure = self.params.get('structure')
        compression = self.params.get('compression')

        if not connection_string or not container:
            raise DataStoreError("'connection_string' and 'container' are required for azureBlobStorage()")

        sql_params = [
            self._format_param(connection_string),
            self._format_param(container),
            self._format_param(path),
        ]

        # Format is optional - chdb can auto-detect
        if format_name:
            sql_params.append(self._format_param(format_name))

            if structure:
                sql_params.append(self._format_param(structure))

            if compression:
                sql_params.append(self._format_param(compression))

        return f"azureBlobStorage({', '.join(sql_params)})"


class GcsTableFunction(TableFunction):
    """
    Wrapper for gcs() table function.

    Supports reading from and writing to Google Cloud Storage.

    Parameters:
        url: GCS URL (https://storage.googleapis.com/bucket/path)
        hmac_key: GCS HMAC key (optional if nosign)
        hmac_secret: GCS HMAC secret (optional if nosign)
        format: Data format (optional, auto-detected if not provided)
        structure: Optional table structure
        compression: Optional compression
        nosign: Use anonymous access
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        url = self.params.get('url') or self.params.get('path')
        hmac_key = self.params.get('hmac_key')
        hmac_secret = self.params.get('hmac_secret')
        format_name = self.params.get('format')
        structure = self.params.get('structure')
        compression = self.params.get('compression')
        nosign = self.params.get('nosign', False)

        if not url:
            raise DataStoreError("'url' or 'path' parameter is required for gcs()")

        sql_params = [self._format_param(url)]

        if nosign:
            sql_params.append('NOSIGN')
        elif hmac_key and hmac_secret:
            sql_params.append(self._format_param(hmac_key))
            sql_params.append(self._format_param(hmac_secret))

        # Format is optional - chdb can auto-detect
        if format_name:
            sql_params.append(self._format_param(format_name))

            if structure:
                sql_params.append(self._format_param(structure))

            if compression:
                sql_params.append(self._format_param(compression))

        return f"gcs({', '.join(sql_params)})"


class HdfsTableFunction(TableFunction):
    """
    Wrapper for hdfs() table function.

    Creates table from files in HDFS.

    Parameters:
        uri: HDFS URI (e.g., 'hdfs://namenode:port/path')
        format: Data format (optional, auto-detected if not provided)
        structure: Optional table structure
        compression: Optional compression method
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        uri = self.params.get('uri') or self.params.get('path')
        format_name = self.params.get('format')
        structure = self.params.get('structure')
        compression = self.params.get('compression')

        if not uri:
            raise DataStoreError("'uri' or 'path' parameter is required for hdfs()")

        sql_params = [self._format_param(uri)]

        # Format is optional - chdb can auto-detect
        if format_name:
            sql_params.append(self._format_param(format_name))

            if structure:
                sql_params.append(self._format_param(structure))

            if compression:
                sql_params.append(self._format_param(compression))

        return f"hdfs({', '.join(sql_params)})"


class MySQLTableFunction(TableFunction):
    """
    Wrapper for mysql() table function.

    Supports SELECT and INSERT queries on remote MySQL servers.

    Parameters:
        host: MySQL server address (host:port)
        database: Database name
        table: Table name
        user: Username
        password: Password
        replace_query: Replace INSERT with REPLACE (optional)
        on_duplicate_clause: ON DUPLICATE KEY clause (optional)
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        host = self.params.get('host')
        database = self.params.get('database')
        table = self.params.get('table')
        user = self.params.get('user')
        password = self.params.get('password', '')

        if not all([host, database, table, user]):
            raise DataStoreError("'host', 'database', 'table', and 'user' are required for mysql()")

        sql_params = [
            self._format_param(host),
            self._format_param(database),
            self._format_param(table),
            self._format_param(user),
            self._format_param(password),
        ]

        return f"mysql({', '.join(sql_params)})"


class PostgreSQLTableFunction(TableFunction):
    """
    Wrapper for postgresql() table function.

    Supports SELECT and INSERT queries on remote PostgreSQL servers.

    Parameters:
        host: PostgreSQL server address (host:port)
        database: Database name
        table: Table name (can include schema as 'schema.table')
        user: Username
        password: Password
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        host = self.params.get('host')
        database = self.params.get('database')
        table = self.params.get('table')
        user = self.params.get('user')
        password = self.params.get('password', '')

        if not all([host, database, table, user]):
            raise DataStoreError("'host', 'database', 'table', and 'user' are required for postgresql()")

        sql_params = [
            self._format_param(host),
            self._format_param(database),
            self._format_param(table),
            self._format_param(user),
            self._format_param(password),
        ]

        return f"postgresql({', '.join(sql_params)})"


class MongoDBTableFunction(TableFunction):
    """
    Wrapper for mongodb() table function.

    Supports SELECT queries on MongoDB collections (read-only).

    Parameters:
        host: MongoDB server address (host:port)
        database: Database name
        collection: Collection name
        user: Username
        password: Password
        structure: Optional table structure
        options: Optional connection options
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False  # MongoDB table function is read-only

    def to_sql(self, quote_char: str = '"') -> str:
        host = self.params.get('host')
        database = self.params.get('database')
        collection = self.params.get('collection') or self.params.get('table')
        user = self.params.get('user')
        password = self.params.get('password', '')
        structure = self.params.get('structure')

        if not all([host, database, collection, user]):
            raise DataStoreError("'host', 'database', 'collection', and 'user' are required for mongodb()")

        sql_params = [
            self._format_param(host),
            self._format_param(database),
            self._format_param(collection),
            self._format_param(user),
            self._format_param(password),
        ]

        if structure:
            sql_params.append(self._format_param(structure))

        return f"mongodb({', '.join(sql_params)})"


class RedisTableFunction(TableFunction):
    """
    Wrapper for redis() table function.

    Integrates with Redis key-value store.

    Parameters:
        host: Redis server address (host:port)
        key: Name of the primary-key column in structure
        structure: Table structure 'key Type, v1 Type, ...'
        password: Redis password (optional)
        db_index: Database index (optional, default: 0)
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        host = self.params.get('host')
        key = self.params.get('key')
        structure = self.params.get('structure')
        password = self.params.get('password')
        db_index = self.params.get('db_index', 0)

        if not all([host, key, structure]):
            raise DataStoreError("'host', 'key', and 'structure' are required for redis()")

        sql_params = [self._format_param(host), self._format_param(key), self._format_param(structure)]

        if password:
            sql_params.append(self._format_param(password))
            sql_params.append(str(db_index))

        return f"redis({', '.join(sql_params)})"


class SQLiteTableFunction(TableFunction):
    """
    Wrapper for sqlite() table function.

    Performs queries on SQLite database (read-only).

    Parameters:
        database_path: Path to SQLite database file
        table: Table name
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False  # SQLite table function is read-only

    def to_sql(self, quote_char: str = '"') -> str:
        database_path = self.params.get('database_path') or self.params.get('path')
        table = self.params.get('table')

        if not all([database_path, table]):
            raise DataStoreError("'database_path' and 'table' are required for sqlite()")

        sql_params = [self._format_param(database_path), self._format_param(table)]

        return f"sqlite({', '.join(sql_params)})"


class RemoteTableFunction(TableFunction):
    """
    Wrapper for remote() / remoteSecure() table functions.

    Accesses remote ClickHouse servers without creating Distributed table.

    Parameters:
        host: Remote server address (host:port) or addresses pattern
        database: Database name
        table: Table name
        user: Username (optional, default: 'default')
        password: Password (optional)
        secure: Use remoteSecure for encrypted connections (default: False)
        sharding_key: Expression for sharding (optional)
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return True

    def to_sql(self, quote_char: str = '"') -> str:
        host = self.params.get('host')
        database = self.params.get('database', 'default')
        table = self.params.get('table')
        user = self.params.get('user', 'default')
        password = self.params.get('password', '')
        secure = self.params.get('secure', False)

        if not all([host, table]):
            raise DataStoreError("'host' and 'table' are required for remote()")

        func_name = 'remoteSecure' if secure else 'remote'

        sql_params = [
            self._format_param(host),
            self._format_param(database),
            self._format_param(table),
            self._format_param(user),
            self._format_param(password),
        ]

        return f"{func_name}({', '.join(sql_params)})"


class IcebergTableFunction(TableFunction):
    """
    Wrapper for iceberg() table function.

    Provides read-only interface to Apache Iceberg tables.

    Parameters:
        url: Path to Iceberg table (S3, Azure, HDFS, or local)
        access_key_id: Access key for cloud storage (optional)
        secret_access_key: Secret key for cloud storage (optional)
        format: File format (optional, default: Parquet)
        structure: Optional table structure
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False  # Iceberg is read-only (experimental write in 25.x)

    def to_sql(self, quote_char: str = '"') -> str:
        url = self.params.get('url') or self.params.get('path')
        access_key = self.params.get('access_key_id')
        secret_key = self.params.get('secret_access_key')
        format_name = self.params.get('format')
        structure = self.params.get('structure')

        if not url:
            raise DataStoreError("'url' or 'path' parameter is required for iceberg()")

        sql_params = [self._format_param(url)]

        if access_key and secret_key:
            sql_params.append(self._format_param(access_key))
            sql_params.append(self._format_param(secret_key))

        if format_name:
            sql_params.append(self._format_param(format_name))

        if structure:
            sql_params.append(self._format_param(structure))

        return f"iceberg({', '.join(sql_params)})"


class DeltaLakeTableFunction(TableFunction):
    """
    Wrapper for deltaLake() table function.

    Provides read-only interface to Delta Lake tables.

    Parameters:
        url: Path to Delta Lake table (S3, Azure, or local)
        access_key_id: Access key (optional)
        secret_access_key: Secret key (optional)
        format: File format (optional)
        structure: Optional table structure
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False  # DeltaLake is read-only (experimental write in 25.x)

    def to_sql(self, quote_char: str = '"') -> str:
        url = self.params.get('url') or self.params.get('path')
        access_key = self.params.get('access_key_id')
        secret_key = self.params.get('secret_access_key')

        if not url:
            raise DataStoreError("'url' or 'path' parameter is required for deltaLake()")

        sql_params = [self._format_param(url)]

        if access_key and secret_key:
            sql_params.append(self._format_param(access_key))
            sql_params.append(self._format_param(secret_key))

        return f"deltaLake({', '.join(sql_params)})"


class HudiTableFunction(TableFunction):
    """
    Wrapper for hudi() table function.

    Provides read-only interface to Apache Hudi tables.

    Parameters:
        url: Path to Hudi table in S3
        access_key_id: AWS access key (optional)
        secret_access_key: AWS secret key (optional)
        format: File format (optional)
        structure: Optional table structure
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False  # Hudi is read-only

    def to_sql(self, quote_char: str = '"') -> str:
        url = self.params.get('url') or self.params.get('path')
        access_key = self.params.get('access_key_id')
        secret_key = self.params.get('secret_access_key')

        if not url:
            raise DataStoreError("'url' or 'path' parameter is required for hudi()")

        sql_params = [self._format_param(url)]

        if access_key and secret_key:
            sql_params.append(self._format_param(access_key))
            sql_params.append(self._format_param(secret_key))

        return f"hudi({', '.join(sql_params)})"


class NumbersTableFunction(TableFunction):
    """
    Wrapper for numbers() table function.

    Returns a table with single 'number' column containing integers.

    Parameters:
        start: Start number (optional, if only one param provided, it's count from 0)
        count: Number of values to generate
        step: Step size (optional, default: 1)

    Example:
        >>> # 0 to 9
        >>> tf = NumbersTableFunction(count=10)

        >>> # 10 to 19
        >>> tf = NumbersTableFunction(start=10, count=10)

        >>> # Even numbers 0, 2, 4, ..., 18
        >>> tf = NumbersTableFunction(start=0, count=10, step=2)
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False

    def to_sql(self, quote_char: str = '"') -> str:
        start = self.params.get('start')
        count = self.params.get('count')
        step = self.params.get('step')

        # Handle different parameter combinations
        sql_params = []

        if start is not None and count is not None:
            sql_params.append(str(start))
            sql_params.append(str(count))
            if step is not None:
                sql_params.append(str(step))
        elif count is not None:
            # Only count provided - range from 0
            sql_params.append(str(count))
        else:
            raise DataStoreError("'count' parameter is required for numbers()")

        return f"numbers({', '.join(sql_params)})"


class GenerateRandomTableFunction(TableFunction):
    """
    Wrapper for generateRandom() table function.

    Generates random data with given schema for testing.

    Parameters:
        structure: Table structure with column types
        random_seed: Random seed for reproducibility (optional)
        max_string_length: Max string length (optional, default: 10)
        max_array_length: Max array length (optional, default: 10)
    """

    @property
    def can_read(self) -> bool:
        return True

    @property
    def can_write(self) -> bool:
        return False

    def to_sql(self, quote_char: str = '"') -> str:
        structure = self.params.get('structure')
        random_seed = self.params.get('random_seed')
        max_string_length = self.params.get('max_string_length')
        max_array_length = self.params.get('max_array_length')

        if not structure:
            raise DataStoreError("'structure' parameter is required for generateRandom()")

        sql_params = [self._format_param(structure)]

        if random_seed is not None:
            sql_params.append(str(random_seed))

        if max_string_length is not None:
            sql_params.append(str(max_string_length))

        if max_array_length is not None:
            sql_params.append(str(max_array_length))

        return f"generateRandom({', '.join(sql_params)})"


# Map source_type to TableFunction class
TABLE_FUNCTION_MAP = {
    'file': FileTableFunction,
    'url': UrlTableFunction,
    'http': UrlTableFunction,  # Alias for url
    'https': UrlTableFunction,  # Alias for url
    's3': S3TableFunction,
    'azure': AzureBlobStorageTableFunction,
    'azureblob': AzureBlobStorageTableFunction,
    'gcs': GcsTableFunction,
    'hdfs': HdfsTableFunction,
    'mysql': MySQLTableFunction,
    'postgresql': PostgreSQLTableFunction,
    'postgres': PostgreSQLTableFunction,  # Alias
    'mongodb': MongoDBTableFunction,
    'mongo': MongoDBTableFunction,  # Alias
    'redis': RedisTableFunction,
    'sqlite': SQLiteTableFunction,
    'remote': RemoteTableFunction,
    'remotesecure': RemoteTableFunction,  # Alias with secure=True
    'clickhouse': RemoteTableFunction,  # Alias for remote
    'iceberg': IcebergTableFunction,
    'deltalake': DeltaLakeTableFunction,
    'delta': DeltaLakeTableFunction,  # Alias
    'hudi': HudiTableFunction,
    'numbers': NumbersTableFunction,
    'generaterandom': GenerateRandomTableFunction,
}


def create_table_function(source_type: str, **params) -> TableFunction:
    """
    Factory function to create appropriate TableFunction based on source_type.

    Args:
        source_type: Type of data source (file, s3, mysql, etc.)
        **params: Parameters specific to the table function

    Returns:
        TableFunction instance

    Raises:
        DataStoreError: If source_type is not supported

    Example:
        >>> tf = create_table_function('file', path='data.csv', format='CSV')
        >>> tf.to_sql()
        "file('data.csv', 'CSV')"
    """
    source_type_lower = source_type.lower()

    if source_type_lower not in TABLE_FUNCTION_MAP:
        raise DataStoreError(
            f"Unsupported source type: {source_type}. "
            f"Supported types: {', '.join(sorted(TABLE_FUNCTION_MAP.keys()))}"
        )

    table_function_class = TABLE_FUNCTION_MAP[source_type_lower]

    # Handle remoteSecure - set secure=True automatically
    if source_type_lower == 'remotesecure':
        params.setdefault('secure', True)

    return table_function_class(**params)
