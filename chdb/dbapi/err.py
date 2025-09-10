"""Exception classes for chdb database operations.

This module provides a complete hierarchy of exception classes for handling
database-related errors in chdb, following the Python Database API Specification v2.0.

The exception hierarchy is structured as follows::

    StandardError
    ├── Warning
    └── Error
        ├── InterfaceError
        └── DatabaseError
            ├── DataError
            ├── OperationalError
            ├── IntegrityError
            ├── InternalError
            ├── ProgrammingError
            └── NotSupportedError

Each exception class represents a specific category of database errors:

- **Warning**: Non-fatal warnings during database operations
- **InterfaceError**: Problems with the database interface itself
- **DatabaseError**: Base class for all database-related errors
- **DataError**: Problems with data processing (invalid values, type errors)
- **OperationalError**: Database operational issues (connectivity, resources)
- **IntegrityError**: Constraint violations (foreign keys, uniqueness)
- **InternalError**: Database internal errors and corruption
- **ProgrammingError**: SQL syntax errors and API misuse
- **NotSupportedError**: Unsupported features or operations

.. note::
    These exception classes are compliant with Python DB API 2.0 specification
    and provide consistent error handling across different database operations.

.. seealso::
    - `Python Database API Specification v2.0 <https://peps.python.org/pep-0249/>`_
    - :mod:`chdb.dbapi.connections` - Database connection management
    - :mod:`chdb.dbapi.cursors` - Database cursor operations

Examples:
    >>> try:
    ...     cursor.execute("SELECT * FROM nonexistent_table")
    ... except ProgrammingError as e:
    ...     print(f"SQL Error: {e}")
    ...
    SQL Error: Table 'nonexistent_table' doesn't exist

    >>> try:
    ...     cursor.execute("INSERT INTO users (id) VALUES (1), (1)")
    ... except IntegrityError as e:
    ...     print(f"Constraint violation: {e}")
    ...
    Constraint violation: Duplicate entry '1' for key 'PRIMARY'
"""


class StandardError(Exception):
    """Exception related to operation with chdb.

    This is the base class for all chdb-related exceptions. It inherits from
    Python's built-in Exception class and serves as the root of the exception
    hierarchy for database operations.

    .. note::
        This exception class follows the Python DB API 2.0 specification
        for database exception handling.
    """


class Warning(StandardError):
    """Exception raised for important warnings like data truncations while inserting, etc.

    This exception is raised when the database operation completes but with
    important warnings that should be brought to the attention of the application.
    Common scenarios include:

    - Data truncation during insertion
    - Precision loss in numeric conversions
    - Character set conversion warnings

    .. note::
        This follows the Python DB API 2.0 specification for warning exceptions.
    """


class Error(StandardError):
    """Exception that is the base class of all other error exceptions (not Warning).

    This is the base class for all error exceptions in chdb, excluding warnings.
    It serves as the parent class for all database error conditions that prevent
    successful completion of operations.

    .. note::
        This exception hierarchy follows the Python DB API 2.0 specification.

    .. seealso::
        :class:`Warning` - For non-fatal warnings that don't prevent operation completion
    """


class InterfaceError(Error):
    """Exception raised for errors that are related to the database interface rather than the database itself.

    This exception is raised when there are problems with the database interface
    implementation, such as:

    - Invalid connection parameters
    - API misuse (calling methods on closed connections)
    - Interface-level protocol errors
    - Module import or initialization failures

    :raises InterfaceError: When database interface encounters errors unrelated to database operations

    .. note::
        These errors are typically programming errors or configuration issues
        that can be resolved by fixing the client code or configuration.
    """


class DatabaseError(Error):
    """Exception raised for errors that are related to the database.

    This is the base class for all database-related errors. It encompasses
    all errors that occur during database operations and are related to the
    database itself rather than the interface.

    Common scenarios include:

    - SQL execution errors
    - Database connectivity issues
    - Transaction-related problems
    - Database-specific constraints violations

    .. note::
        This serves as the parent class for more specific database error types
        such as :class:`DataError`, :class:`OperationalError`, etc.
    """


class DataError(DatabaseError):
    """Exception raised for errors that are due to problems with the processed data.

    This exception is raised when database operations fail due to issues with
    the data being processed, such as:

    - Division by zero operations
    - Numeric values out of range
    - Invalid date/time values
    - String truncation errors
    - Type conversion failures
    - Invalid data format for column type

    :raises DataError: When data validation or processing fails

    Examples:
        >>> # Division by zero in SQL
        >>> cursor.execute("SELECT 1/0")
        DataError: Division by zero

        >>> # Invalid date format
        >>> cursor.execute("INSERT INTO table VALUES ('invalid-date')")
        DataError: Invalid date format
    """


class OperationalError(DatabaseError):
    """Exception raised for errors that are related to the database's operation.

    This exception is raised for errors that occur during database operation
    and are not necessarily under the control of the programmer, including:

    - Unexpected disconnection from database
    - Database server not found or unreachable
    - Transaction processing failures
    - Memory allocation errors during processing
    - Disk space or resource exhaustion
    - Database server internal errors
    - Authentication or authorization failures

    :raises OperationalError: When database operations fail due to operational issues

    .. note::
        These errors are typically transient and may be resolved by retrying
        the operation or addressing system-level issues.

    .. warning::
        Some operational errors may indicate serious system problems that
        require administrative intervention.
    """


class IntegrityError(DatabaseError):
    """Exception raised when the relational integrity of the database is affected.

    This exception is raised when database operations violate integrity constraints,
    including:

    - Foreign key constraint violations
    - Primary key or unique constraint violations (duplicate keys)
    - Check constraint violations
    - NOT NULL constraint violations
    - Referential integrity violations

    :raises IntegrityError: When database integrity constraints are violated

    Examples:
        >>> # Duplicate primary key
        >>> cursor.execute("INSERT INTO users (id, name) VALUES (1, 'John')")
        >>> cursor.execute("INSERT INTO users (id, name) VALUES (1, 'Jane')")
        IntegrityError: Duplicate entry '1' for key 'PRIMARY'

        >>> # Foreign key violation
        >>> cursor.execute("INSERT INTO orders (user_id) VALUES (999)")
        IntegrityError: Cannot add or update a child row: foreign key constraint fails
    """


class InternalError(DatabaseError):
    """Exception raised when the database encounters an internal error.

    This exception is raised when the database system encounters internal
    errors that are not caused by the application, such as:

    - Invalid cursor state (cursor is not valid anymore)
    - Transaction state inconsistencies (transaction is out of sync)
    - Database corruption issues
    - Internal data structure corruption
    - System-level database errors

    :raises InternalError: When database encounters internal inconsistencies

    .. warning::
        Internal errors may indicate serious database problems that require
        database administrator attention. These errors are typically not
        recoverable through application-level retry logic.

    .. note::
        These errors are generally outside the control of the application
        and may require database restart or repair operations.
    """


class ProgrammingError(DatabaseError):
    """Exception raised for programming errors in database operations.

    This exception is raised when there are programming errors in the
    application's database usage, including:

    - Table or column not found
    - Table or index already exists when creating
    - SQL syntax errors in statements
    - Wrong number of parameters specified in prepared statements
    - Invalid SQL operations (e.g., DROP on non-existent objects)
    - Incorrect usage of database API methods

    :raises ProgrammingError: When SQL statements or API usage contains errors

    Examples:
        >>> # Table not found
        >>> cursor.execute("SELECT * FROM nonexistent_table")
        ProgrammingError: Table 'nonexistent_table' doesn't exist

        >>> # SQL syntax error
        >>> cursor.execute("SELCT * FROM users")
        ProgrammingError: You have an error in your SQL syntax

        >>> # Wrong parameter count
        >>> cursor.execute("INSERT INTO users (name, age) VALUES (%s)", ('John',))
        ProgrammingError: Column count doesn't match value count
    """


class NotSupportedError(DatabaseError):
    """Exception raised when a method or database API is not supported.

    This exception is raised when the application attempts to use database
    features or API methods that are not supported by the current database
    configuration or version, such as:

    - Requesting rollback() on connections without transaction support
    - Using advanced SQL features not supported by the database version
    - Calling methods not implemented by the current driver
    - Attempting to use disabled database features

    :raises NotSupportedError: When unsupported database features are accessed

    Examples:
        >>> # Transaction rollback on non-transactional connection
        >>> connection.rollback()
        NotSupportedError: Transactions are not supported

        >>> # Using unsupported SQL syntax
        >>> cursor.execute("SELECT * FROM table WITH (NOLOCK)")
        NotSupportedError: WITH clause not supported in this database version

    .. note::
        Check database documentation and driver capabilities to avoid
        these errors. Consider graceful fallbacks where possible.
    """
