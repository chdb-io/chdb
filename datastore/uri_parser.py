"""
URI parser for DataStore - provides automatic URI-based DataStore creation.

Supports various URI formats:
- file:///path/to/data.csv
- /path/to/data.csv (local file path)
- s3://bucket/key
- gs://bucket/key (Google Cloud Storage)
- az://container/blob (Azure Blob Storage)
- hdfs://namenode:port/path
- http://example.com/data.csv
- https://example.com/data.csv
- mysql://user:pass@host:port/database/table
- postgresql://user:pass@host:port/database/table
- mongodb://user:pass@host:port/database.collection
- sqlite:///path/to/database.db?table=tablename
- redis://host:port?key=mykey
- iceberg://catalog/namespace/table
- deltalake://path/to/table
- hudi://path/to/table
"""

from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote
import os


def parse_uri(uri: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a URI and return source_type and kwargs for DataStore.

    Args:
        uri: URI string (e.g., "s3://bucket/key", "mysql://user:pass@host/db/table")

    Returns:
        Tuple of (source_type, kwargs) where kwargs contains the connection parameters

    Examples:
        >>> source_type, kwargs = parse_uri("file:///data/test.csv")
        >>> print(source_type)  # "file"
        >>> print(kwargs)  # {"path": "/data/test.csv", "format": "CSV"}

        >>> source_type, kwargs = parse_uri("s3://mybucket/data.parquet")
        >>> print(source_type)  # "s3"
        >>> print(kwargs)  # {"url": "s3://mybucket/data.parquet", "format": "Parquet"}

        >>> source_type, kwargs = parse_uri("mysql://root:pass@localhost:3306/mydb/users")
        >>> print(source_type)  # "mysql"
        >>> print(kwargs)  # {"host": "localhost:3306", "database": "mydb", "table": "users", ...}
    """
    if not uri:
        raise ValueError("URI cannot be empty")

    # Parse the URI
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower() if parsed.scheme else None

    # If no scheme, treat as local file path
    if not scheme or (scheme and len(scheme) == 1):  # Windows drive letter like C:
        return _parse_file_path(uri)

    # Route to appropriate parser based on scheme
    if scheme == "file":
        return _parse_file_uri(parsed)
    elif scheme in ["s3", "s3a", "s3n"]:
        return _parse_s3_uri(parsed)
    elif scheme in ["gs", "gcs"]:
        return _parse_gcs_uri(parsed)
    elif scheme in ["az", "azure", "wasb", "wasbs"]:
        return _parse_azure_uri(parsed)
    elif scheme == "hdfs":
        return _parse_hdfs_uri(parsed)
    elif scheme in ["http", "https"]:
        return _parse_url_uri(parsed)
    elif scheme == "mysql":
        return _parse_mysql_uri(parsed)
    elif scheme in ["postgresql", "postgres"]:
        return _parse_postgresql_uri(parsed)
    elif scheme in ["mongodb", "mongo"]:
        return _parse_mongodb_uri(parsed)
    elif scheme == "sqlite":
        return _parse_sqlite_uri(parsed)
    elif scheme == "redis":
        return _parse_redis_uri(parsed)
    elif scheme == "clickhouse":
        return _parse_clickhouse_uri(parsed)
    elif scheme == "iceberg":
        return _parse_iceberg_uri(parsed)
    elif scheme in ["deltalake", "delta"]:
        return _parse_deltalake_uri(parsed)
    elif scheme == "hudi":
        return _parse_hudi_uri(parsed)
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme}")


def _infer_format_from_path(path: str) -> Optional[str]:
    """
    Infer file format from file extension.

    Note: For CSV files, we use 'CSVWithNames' which expects the first row
    to be column headers. This matches pandas' default behavior and is more
    user-friendly than 'CSV' which uses auto-generated column names (c1, c2, ...).
    """
    if not path:
        return None

    ext_map = {
        '.csv': 'CSVWithNames',  # First row is header (pandas-compatible)
        '.tsv': 'TSVWithNames',  # First row is header (pandas-compatible)
        '.parquet': 'Parquet',
        '.pq': 'Parquet',
        '.json': 'JSON',
        '.jsonl': 'JSONEachRow',
        '.ndjson': 'JSONEachRow',
        '.xml': 'XML',
        '.arrow': 'Arrow',
        '.orc': 'ORC',
        '.avro': 'Avro',
    }

    # Get extension
    _, ext = os.path.splitext(path.lower())
    return ext_map.get(ext)


def _parse_file_path(path: str) -> Tuple[str, Dict[str, Any]]:
    """Parse a local file path."""
    # Normalize path
    path = os.path.expanduser(path)

    kwargs = {
        'path': path,
    }

    # Infer format from extension
    format_type = _infer_format_from_path(path)
    if format_type:
        kwargs['format'] = format_type

    return 'file', kwargs


def _parse_file_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """Parse file:// URI."""
    # file:///path/to/file -> /path/to/file
    path = unquote(parsed.path)

    # Handle query parameters
    query_params = parse_qs(parsed.query)
    kwargs = {
        'path': path,
    }

    # Check for explicit format in query params
    if 'format' in query_params:
        kwargs['format'] = query_params['format'][0]
    else:
        # Infer format from extension
        format_type = _infer_format_from_path(path)
        if format_type:
            kwargs['format'] = format_type

    return 'file', kwargs


def _parse_s3_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """Parse s3:// URI."""
    # s3://bucket/key/to/file.parquet
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    # Reconstruct full S3 URL
    url = f"s3://{bucket}/{key}"

    kwargs = {
        'url': url,
    }

    # Parse query parameters for credentials and options
    query_params = parse_qs(parsed.query)

    if 'access_key_id' in query_params:
        kwargs['access_key_id'] = query_params['access_key_id'][0]
    if 'secret_access_key' in query_params:
        kwargs['secret_access_key'] = query_params['secret_access_key'][0]
    if 'session_token' in query_params:
        kwargs['session_token'] = query_params['session_token'][0]
    if 'region' in query_params:
        kwargs['region'] = query_params['region'][0]
    if 'nosign' in query_params:
        kwargs['nosign'] = query_params['nosign'][0].lower() in ['true', '1', 'yes']
    if 'format' in query_params:
        kwargs['format'] = query_params['format'][0]

    # Infer format from key if not specified
    if 'format' not in kwargs:
        format_type = _infer_format_from_path(key)
        if format_type:
            kwargs['format'] = format_type

    return 's3', kwargs


def _parse_gcs_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """Parse gs:// URI for Google Cloud Storage."""
    # gs://bucket/path/to/file
    bucket = parsed.netloc
    path = parsed.path.lstrip('/')

    url = f"gs://{bucket}/{path}"

    kwargs = {
        'url': url,
    }

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    if 'hmac_key' in query_params:
        kwargs['hmac_key'] = query_params['hmac_key'][0]
    if 'hmac_secret' in query_params:
        kwargs['hmac_secret'] = query_params['hmac_secret'][0]
    if 'format' in query_params:
        kwargs['format'] = query_params['format'][0]

    # Infer format
    if 'format' not in kwargs:
        format_type = _infer_format_from_path(path)
        if format_type:
            kwargs['format'] = format_type

    return 'gcs', kwargs


def _parse_azure_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """Parse az:// URI for Azure Blob Storage."""
    # az://container/blob/path
    container = parsed.netloc
    blob = parsed.path.lstrip('/')

    url = f"az://{container}/{blob}"

    kwargs = {
        'url': url,
    }

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    if 'account_name' in query_params:
        kwargs['account_name'] = query_params['account_name'][0]
    if 'account_key' in query_params:
        kwargs['account_key'] = query_params['account_key'][0]
    if 'connection_string' in query_params:
        kwargs['connection_string'] = query_params['connection_string'][0]
    if 'format' in query_params:
        kwargs['format'] = query_params['format'][0]

    # Infer format
    if 'format' not in kwargs:
        format_type = _infer_format_from_path(blob)
        if format_type:
            kwargs['format'] = format_type

    return 'azureBlobStorage', kwargs


def _parse_hdfs_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """Parse hdfs:// URI."""
    # hdfs://namenode:port/path/to/file
    url = f"hdfs://{parsed.netloc}{parsed.path}"

    kwargs = {
        'url': url,
    }

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    if 'format' in query_params:
        kwargs['format'] = query_params['format'][0]

    # Infer format
    if 'format' not in kwargs:
        format_type = _infer_format_from_path(parsed.path)
        if format_type:
            kwargs['format'] = format_type

    return 'hdfs', kwargs


def _parse_url_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """Parse http:// or https:// URI."""
    # Reconstruct full URL
    url = parsed.geturl()

    kwargs = {
        'url': url,
    }

    # Try to infer format from path
    format_type = _infer_format_from_path(parsed.path)
    if format_type:
        kwargs['format'] = format_type

    return 'url', kwargs


def _parse_mysql_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse mysql:// URI.

    Format: mysql://user:password@host:port/database/table?option=value
    """
    kwargs = {}

    # User and password
    if parsed.username:
        kwargs['user'] = unquote(parsed.username)
    if parsed.password:
        kwargs['password'] = unquote(parsed.password)

    # Host and port
    if parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        kwargs['host'] = host

    # Parse path for database and table
    # Format: /database/table
    path_parts = [p for p in parsed.path.split('/') if p]
    if len(path_parts) >= 1:
        kwargs['database'] = path_parts[0]
    if len(path_parts) >= 2:
        kwargs['table'] = path_parts[1]

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    for key, values in query_params.items():
        if key not in kwargs:  # Don't override path-based params
            kwargs[key] = values[0] if len(values) == 1 else values

    return 'mysql', kwargs


def _parse_postgresql_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse postgresql:// URI.

    Format: postgresql://user:password@host:port/database/table?option=value
    """
    kwargs = {}

    # User and password
    if parsed.username:
        kwargs['user'] = unquote(parsed.username)
    if parsed.password:
        kwargs['password'] = unquote(parsed.password)

    # Host and port
    if parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        kwargs['host'] = host

    # Parse path for database and table
    path_parts = [p for p in parsed.path.split('/') if p]
    if len(path_parts) >= 1:
        kwargs['database'] = path_parts[0]
    if len(path_parts) >= 2:
        kwargs['table'] = path_parts[1]

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    for key, values in query_params.items():
        if key not in kwargs:
            kwargs[key] = values[0] if len(values) == 1 else values

    return 'postgresql', kwargs


def _parse_mongodb_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse mongodb:// URI.

    Format: mongodb://user:password@host:port/database.collection?option=value
    """
    kwargs = {}

    # User and password
    if parsed.username:
        kwargs['user'] = unquote(parsed.username)
    if parsed.password:
        kwargs['password'] = unquote(parsed.password)

    # Host and port
    if parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        kwargs['host'] = host

    # Parse path for database.collection
    # Format: /database.collection
    path = parsed.path.lstrip('/')
    if '.' in path:
        database, collection = path.split('.', 1)
        kwargs['database'] = database
        kwargs['collection'] = collection
    elif path:
        kwargs['database'] = path

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    for key, values in query_params.items():
        if key not in kwargs:
            kwargs[key] = values[0] if len(values) == 1 else values

    return 'mongodb', kwargs


def _parse_sqlite_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse sqlite:// URI.

    Format: sqlite:///path/to/database.db?table=tablename
    """
    # sqlite:///path/to/db.db -> /path/to/db.db
    path = unquote(parsed.path)

    kwargs = {
        'database': path,
    }

    # Parse query parameters for table name
    query_params = parse_qs(parsed.query)
    if 'table' in query_params:
        kwargs['table'] = query_params['table'][0]

    return 'sqlite', kwargs


def _parse_redis_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse redis:// URI.

    Format: redis://host:port/db?key=mykey
    """
    kwargs = {}

    # Host and port
    if parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        kwargs['host'] = host

    # Parse path for database number
    path = parsed.path.lstrip('/')
    if path:
        try:
            kwargs['db_index'] = int(path)
        except ValueError:
            pass

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    if 'key' in query_params:
        kwargs['key'] = query_params['key'][0]
    if 'password' in query_params:
        kwargs['password'] = query_params['password'][0]

    return 'redis', kwargs


def _parse_clickhouse_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse clickhouse:// URI.

    Format: clickhouse://host:port/database/table
    """
    kwargs = {}

    # Host and port
    if parsed.hostname:
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        kwargs['host'] = host

    # Parse path for database and table
    path_parts = [p for p in parsed.path.split('/') if p]
    if len(path_parts) >= 1:
        kwargs['database'] = path_parts[0]
    if len(path_parts) >= 2:
        kwargs['table'] = path_parts[1]

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    if 'user' in query_params:
        kwargs['user'] = query_params['user'][0]
    if 'password' in query_params:
        kwargs['password'] = query_params['password'][0]

    return 'clickhouse', kwargs


def _parse_iceberg_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse iceberg:// URI.

    Format: iceberg://catalog/namespace/table
    """
    # Combine netloc (catalog) and path (namespace/table)
    catalog = parsed.netloc
    path = parsed.path.lstrip('/')

    # Build full path
    if catalog and path:
        full_path = f"{catalog}/{path}"
    elif catalog:
        full_path = catalog
    else:
        full_path = path

    kwargs = {}
    if full_path:
        kwargs['url'] = full_path

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    for key, values in query_params.items():
        kwargs[key] = values[0] if len(values) == 1 else values

    return 'iceberg', kwargs


def _parse_deltalake_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse deltalake:// URI.

    Format: deltalake:///path/to/table or deltalake://s3/bucket/path
    """
    path = parsed.path.lstrip('/')

    kwargs = {
        'url': path if path else parsed.netloc,
    }

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    for key, values in query_params.items():
        kwargs[key] = values[0] if len(values) == 1 else values

    return 'deltaLake', kwargs


def _parse_hudi_uri(parsed) -> Tuple[str, Dict[str, Any]]:
    """
    Parse hudi:// URI.

    Format: hudi:///path/to/table
    """
    path = parsed.path.lstrip('/')

    kwargs = {
        'url': path if path else parsed.netloc,
    }

    # Parse query parameters
    query_params = parse_qs(parsed.query)
    for key, values in query_params.items():
        kwargs[key] = values[0] if len(values) == 1 else values

    return 'hudi', kwargs
