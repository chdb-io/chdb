from .constants import FIELD_TYPE
from . import connections as _orig_conn
from .. import chdb_version

if len(chdb_version) > 3 and chdb_version[3] is not None:
    VERSION_STRING = "%s.%s.%s_%s" % chdb_version
else:
    VERSION_STRING = "%s.%s.%s" % chdb_version[:3]

threadsafety = 1
apilevel = "2.0"
paramstyle = "format"


class DBAPISet(frozenset):
    """Extended frozenset for DB-API 2.0 type comparison.

    This class extends frozenset to support DB-API 2.0 type comparison semantics.
    It allows for flexible type checking where individual items can be compared
    against the set using both equality and inequality operators.

    This is used for type constants like STRING, BINARY, NUMBER, etc. to enable
    comparisons like "field_type == STRING" where field_type is a single type value.

    Examples:
        >>> string_types = DBAPISet([FIELD_TYPE.STRING, FIELD_TYPE.VAR_STRING])
        >>> FIELD_TYPE.STRING == string_types  # Returns True
        >>> FIELD_TYPE.INT != string_types     # Returns True
        >>> FIELD_TYPE.BLOB in string_types    # Returns False
    """

    def __ne__(self, other):
        """Check inequality with flexible type comparison.

        Args:
            other: Value to compare against this set

        Returns:
            bool: True if other is not in this set (for non-set types) or
                  True if sets are not equal (for set types)
        """
        if isinstance(other, set):
            return frozenset.__ne__(self, other)
        else:
            return other not in self

    def __eq__(self, other):
        """Check equality with flexible type comparison.

        Args:
            other: Value to compare against this set

        Returns:
            bool: True if other is in this set (for non-set types) or
                  True if sets are equal (for set types)
        """
        if isinstance(other, frozenset):
            return frozenset.__eq__(self, other)
        else:
            return other in self

    def __hash__(self):
        """Return hash value for the set.

        Returns:
            int: Hash value of the underlying frozenset
        """
        return frozenset.__hash__(self)


# TODO it's in pep249 find out meaning and usage of this
# https://www.python.org/dev/peps/pep-0249/#string
STRING = DBAPISet([FIELD_TYPE.ENUM, FIELD_TYPE.STRING,
                   FIELD_TYPE.VAR_STRING])
BINARY = DBAPISet([FIELD_TYPE.BLOB, FIELD_TYPE.LONG_BLOB,
                   FIELD_TYPE.MEDIUM_BLOB, FIELD_TYPE.TINY_BLOB])
NUMBER = DBAPISet([FIELD_TYPE.DECIMAL, FIELD_TYPE.DOUBLE, FIELD_TYPE.FLOAT,
                   FIELD_TYPE.INT24, FIELD_TYPE.LONG, FIELD_TYPE.LONGLONG,
                   FIELD_TYPE.TINY, FIELD_TYPE.YEAR])
DATE = DBAPISet([FIELD_TYPE.DATE, FIELD_TYPE.NEWDATE])
TIME = DBAPISet([FIELD_TYPE.TIME])
TIMESTAMP = DBAPISet([FIELD_TYPE.TIMESTAMP, FIELD_TYPE.DATETIME])
DATETIME = TIMESTAMP
ROWID = DBAPISet()


def Binary(x):
    """Return x as a binary type.

    This function converts the input to bytes type for use with binary
    database fields, following the DB-API 2.0 specification.

    Args:
        x: Input data to convert to binary

    Returns:
        bytes: The input converted to bytes
    """
    return bytes(x)


def Connect(*args, **kwargs):
    """
    Connect to the database; see connections.Connection.__init__() for
    more information.
    """
    from .connections import Connection
    return Connection(*args, **kwargs)


if _orig_conn.Connection.__init__.__doc__ is not None:
    Connect.__doc__ = _orig_conn.Connection.__init__.__doc__
del _orig_conn


def get_client_info():
    """Get client version information.

    Returns the chDB client version as a string for MySQLdb compatibility.

    Returns:
        str: Version string in format 'major.minor.patch'
    """
    version = chdb_version
    if len(chdb_version) > 3 and chdb_version[3] is None:
        version = chdb_version[:3]
    return '.'.join(map(str, version))


connect = Connection = Connect

NULL = "NULL"

__version__ = get_client_info()
