"""
Source adapters for remote database metadata operations.

Each adapter provides database-specific SQL generation for:
- Listing databases
- Listing tables  
- Describing table schema
- Building table function SQL
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class SourceAdapter(ABC):
    """
    Base adapter for remote database metadata operations.
    
    Subclasses implement database-specific SQL generation.
    """
    
    def __init__(self, host: str, user: str, password: str = '', **kwargs):
        """
        Initialize adapter with connection parameters.
        
        Args:
            host: Server address (host:port)
            user: Username
            password: Password
            **kwargs: Additional parameters (e.g., secure for ClickHouse)
        """
        self.host = host
        self.user = user
        self.password = password
        self.kwargs = kwargs
    
    @abstractmethod
    def get_table_function_name(self) -> str:
        """Return the chdb table function name (e.g., 'remote', 'mysql')."""
        pass
    
    @abstractmethod  
    def list_databases_sql(self) -> str:
        """Return SQL to list all databases."""
        pass
    
    @abstractmethod
    def list_tables_sql(self, database: str) -> str:
        """Return SQL to list tables in a database."""
        pass
    
    @abstractmethod
    def describe_table_sql(self, database: str, table: str) -> str:
        """Return SQL to describe table schema."""
        pass
    
    def build_table_function(self, database: str, table: str) -> str:
        """
        Build table function SQL for accessing a specific table.
        
        Args:
            database: Database name
            table: Table name
            
        Returns:
            SQL table function call (e.g., "mysql('host', 'db', 'table', 'user', 'pass')")
        """
        pass
    
    def _escape_sql_string(self, value: str) -> str:
        """Escape single quotes in SQL string values."""
        if value is None:
            return ''
        return value.replace("'", "''")


class ClickHouseAdapter(SourceAdapter):
    """
    Adapter for remote ClickHouse servers.
    
    Uses remote() or remoteSecure() table functions.
    """
    
    def __init__(self, host: str, user: str = 'default', password: str = '', 
                 secure: bool = False, **kwargs):
        super().__init__(host, user, password, **kwargs)
        self.secure = secure
    
    def get_table_function_name(self) -> str:
        return 'remoteSecure' if self.secure else 'remote'
    
    def list_databases_sql(self) -> str:
        func = self.get_table_function_name()
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        return f"SELECT name FROM {func}('{host}', 'system', 'databases', '{user}', '{password}')"
    
    def list_tables_sql(self, database: str) -> str:
        func = self.get_table_function_name()
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        return (
            f"SELECT name FROM {func}('{host}', 'system', 'tables', '{user}', '{password}') "
            f"WHERE database = '{database}'"
        )
    
    def describe_table_sql(self, database: str, table: str) -> str:
        func = self.get_table_function_name()
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        table = self._escape_sql_string(table)
        return (
            f"SELECT name, type, default_kind, default_expression, comment "
            f"FROM {func}('{host}', 'system', 'columns', '{user}', '{password}') "
            f"WHERE database = '{database}' AND table = '{table}' "
            f"ORDER BY position"
        )
    
    def build_table_function(self, database: str, table: str) -> str:
        func = self.get_table_function_name()
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        table = self._escape_sql_string(table)
        return f"{func}('{host}', '{database}', '{table}', '{user}', '{password}')"


class MySQLAdapter(SourceAdapter):
    """
    Adapter for MySQL servers.
    
    Uses mysql() table function.
    """
    
    def get_table_function_name(self) -> str:
        return 'mysql'
    
    def list_databases_sql(self) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        return (
            f"SELECT schema_name AS name "
            f"FROM mysql('{host}', 'information_schema', 'schemata', '{user}', '{password}')"
        )
    
    def list_tables_sql(self, database: str) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        return (
            f"SELECT table_name AS name "
            f"FROM mysql('{host}', 'information_schema', 'tables', '{user}', '{password}') "
            f"WHERE table_schema = '{database}' AND table_type = 'BASE TABLE'"
        )
    
    def describe_table_sql(self, database: str, table: str) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        table = self._escape_sql_string(table)
        return (
            f"SELECT column_name AS name, data_type AS type, "
            f"is_nullable, column_default AS default_expression, column_comment AS comment "
            f"FROM mysql('{host}', 'information_schema', 'columns', '{user}', '{password}') "
            f"WHERE table_schema = '{database}' AND table_name = '{table}' "
            f"ORDER BY ordinal_position"
        )
    
    def build_table_function(self, database: str, table: str) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        table = self._escape_sql_string(table)
        return f"mysql('{host}', '{database}', '{table}', '{user}', '{password}')"


class PostgreSQLAdapter(SourceAdapter):
    """
    Adapter for PostgreSQL servers.
    
    Uses postgresql() table function.
    """
    
    def get_table_function_name(self) -> str:
        return 'postgresql'
    
    def list_databases_sql(self) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        # Query pg_database through postgres database
        return (
            f"SELECT datname AS name "
            f"FROM postgresql('{host}', 'postgres', 'pg_database', '{user}', '{password}') "
            f"WHERE datistemplate = false"
        )
    
    def list_tables_sql(self, database: str) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        # Query information_schema.tables through the target database
        return (
            f"SELECT table_name AS name "
            f"FROM postgresql('{host}', '{database}', 'information_schema.tables', '{user}', '{password}') "
            f"WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
    
    def describe_table_sql(self, database: str, table: str) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        table = self._escape_sql_string(table)
        return (
            f"SELECT column_name AS name, data_type AS type, "
            f"is_nullable, column_default AS default_expression "
            f"FROM postgresql('{host}', '{database}', 'information_schema.columns', '{user}', '{password}') "
            f"WHERE table_name = '{table}' AND table_schema = 'public' "
            f"ORDER BY ordinal_position"
        )
    
    def build_table_function(self, database: str, table: str) -> str:
        host = self._escape_sql_string(self.host)
        user = self._escape_sql_string(self.user)
        password = self._escape_sql_string(self.password)
        database = self._escape_sql_string(database)
        table = self._escape_sql_string(table)
        return f"postgresql('{host}', '{database}', '{table}', '{user}', '{password}')"


# Adapter registry
ADAPTER_MAP = {
    'clickhouse': ClickHouseAdapter,
    'remote': ClickHouseAdapter,
    'remotesecure': ClickHouseAdapter,
    'mysql': MySQLAdapter,
    'postgresql': PostgreSQLAdapter,
    'postgres': PostgreSQLAdapter,
}


def get_adapter(source_type: str, **params) -> SourceAdapter:
    """
    Get the appropriate adapter for a source type.
    
    Args:
        source_type: Database source type
        **params: Connection parameters (host, user, password, etc.)
        
    Returns:
        SourceAdapter instance
        
    Raises:
        DataStoreError: If source type is not supported for metadata operations
    """
    from .exceptions import DataStoreError
    
    source_lower = source_type.lower()
    if source_lower not in ADAPTER_MAP:
        raise DataStoreError(
            f"Metadata discovery is not supported for source type: {source_type}.\n"
            f"Supported types: {', '.join(sorted(set(ADAPTER_MAP.keys())))}"
        )
    
    adapter_cls = ADAPTER_MAP[source_lower]
    
    # Handle remotesecure alias
    if source_lower == 'remotesecure':
        params.setdefault('secure', True)
    
    return adapter_cls(**params)
