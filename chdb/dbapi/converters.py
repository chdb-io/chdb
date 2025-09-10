"""Type conversion and escaping functions for chDB database operations.

This module provides comprehensive type conversion between Python types and SQL values,
including proper escaping for SQL injection prevention and bidirectional conversion
between database column values and Python objects.

The module handles:
- Escaping Python values for safe SQL inclusion
- Converting database column data to appropriate Python types
- Time/date/datetime conversions with proper formatting
- Collection types (lists, tuples, sets, dicts) handling
- Special value handling (None, boolean, numeric types)
"""

import datetime
from decimal import Decimal
from .err import DataError
import re
import time


def escape_item(val, mapping=None):
    """Escape a single value for safe SQL inclusion.

    This function takes a Python value and converts it to a properly escaped
    SQL representation using the appropriate encoder for the value's type.

    Args:
        val: Python value to escape (any supported type)
        mapping (dict, optional): Custom encoder mapping. Uses default encoders if None.

    Returns:
        str: SQL-safe string representation of the value

    Raises:
        TypeError: If no encoder is found for the value type

    Examples:
        >>> escape_item("O'Reilly")
        "'O''Reilly'"
        >>> escape_item(42)
        "42"
        >>> escape_item(None)
        "NULL"
        >>> escape_item(True)
        "1"
    """
    if mapping is None:
        mapping = encoders
    encoder = mapping.get(type(val))

    # Fallback to default when no encoder found
    if not encoder:
        try:
            encoder = mapping[str]
        except KeyError:
            raise TypeError("no default type converter defined")

    val = encoder(val, mapping)
    return val


def escape_dict(val, mapping=None):
    """Escape all values in a dictionary.

    Args:
        val (dict): Dictionary with values to escape
        mapping (dict, optional): Custom encoder mapping

    Returns:
        dict: Dictionary with all values properly escaped for SQL

    Example:
        >>> escape_dict({'name': "O'Reilly", 'age': 30})
        {'name': "'O''Reilly'", 'age': '30'}
    """
    n = {}
    for k, v in val.items():
        quoted = escape_item(v, mapping)
        n[k] = quoted
    return n


def escape_sequence(val, mapping=None):
    """Escape a sequence (list, tuple, etc.) for SQL VALUES clause.

    Args:
        val (sequence): Sequence of values to escape
        mapping (dict, optional): Custom encoder mapping

    Returns:
        str: SQL VALUES clause representation like '(val1, val2, val3)'

    Example:
        >>> escape_sequence([1, "hello", None])
        "(1, 'hello', NULL)"
    """
    n = []
    for item in val:
        quoted = escape_item(item, mapping)
        n.append(quoted)
    return "(" + ",".join(n) + ")"


def escape_set(val, mapping=None):
    """Escape a set for SQL representation.

    Args:
        val (set): Set of values to escape
        mapping (dict, optional): Custom encoder mapping

    Returns:
        str: Comma-separated escaped values

    Example:
        >>> escape_set({1, 2, 3})
        "1,2,3"
    """
    return ','.join([escape_item(x, mapping) for x in val])


def escape_bool(value, mapping=None):
    """Escape boolean value for SQL.

    Args:
        value (bool): Boolean value to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: "1" for True, "0" for False

    Example:
        >>> escape_bool(True)
        "1"
        >>> escape_bool(False)
        "0"
    """
    return str(int(value))


def escape_object(value, mapping=None):
    """Generic object escaper using string conversion.

    Args:
        value: Object to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: String representation of the object
    """
    return str(value)


def escape_int(value, mapping=None):
    """Escape integer value for SQL.

    Args:
        value (int): Integer to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: String representation of the integer
    """
    return str(value)


def escape_float(value, mapping=None):
    """Escape float value for SQL with precision control.

    Args:
        value (float): Float to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: String representation with up to 15 significant digits
    """
    return '%.15g' % value


_escape_table = [chr(x) for x in range(128)]
_escape_table[ord("'")] = u"''"
_escape_table[ord("\\")] = "\\\\"


def _escape_unicode(value, mapping=None):
    """Escape Unicode string by replacing special characters.

    This function escapes single quotes and backslashes in Unicode strings
    to prevent SQL injection attacks.

    Args:
        value (str): Unicode string to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: Escaped Unicode string (without surrounding quotes)

    Note:
        This function does not add surrounding quotes. Use escape_unicode()
        for complete string escaping with quotes.
    """
    return value.translate(_escape_table)


escape_string = _escape_unicode

# On Python ~3.5, str.decode('ascii', 'surrogateescape') is slow.
# (fixed in Python 3.6, http://bugs.python.org/issue24870)
# Workaround is str.decode('latin1') then translate 0x80-0xff into 0udc80-0udcff.
# We can escape special chars and surrogateescape at once.
_escape_bytes_table = _escape_table + [chr(i) for i in range(0xdc80, 0xdd00)]


def escape_bytes(value, mapping=None):
    """Escape bytes value for SQL with proper encoding handling.

    Args:
        value (bytes): Bytes to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: Escaped bytes as quoted SQL string

    Example:
        >>> escape_bytes(b"hello'world")
        "'hello''world'"
    """
    return "'%s'" % value.decode('latin1').translate(_escape_bytes_table)


def escape_unicode(value, mapping=None):
    """Escape Unicode string for SQL with surrounding quotes.

    Args:
        value (str): Unicode string to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: Properly escaped and quoted SQL string

    Example:
        >>> escape_unicode("O'Reilly")
        "'O''Reilly'"
    """
    return u"'%s'" % _escape_unicode(value)


def escape_str(value, mapping=None):
    """Escape string value for SQL.

    Args:
        value: Value to convert to string and escape
        mapping (dict, optional): Custom encoder mapping

    Returns:
        str: Escaped and quoted SQL string
    """
    return "'%s'" % escape_string(str(value), mapping)


def escape_None(value, mapping=None):
    """Escape None value for SQL.

    Args:
        value: None value (ignored)
        mapping: Unused, for interface compatibility

    Returns:
        str: SQL NULL literal
    """
    return 'NULL'


def escape_timedelta(obj, mapping=None):
    """Escape timedelta object for SQL TIME format.

    Args:
        obj (datetime.timedelta): Timedelta to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: SQL TIME format string like 'HH:MM:SS' or 'HH:MM:SS.microseconds'

    Example:
        >>> td = datetime.timedelta(hours=2, minutes=30, seconds=45, microseconds=123456)
        >>> escape_timedelta(td)
        "'02:30:45.123456'"
    """
    seconds = int(obj.seconds) % 60
    minutes = int(obj.seconds // 60) % 60
    hours = int(obj.seconds // 3600) % 24 + int(obj.days) * 24
    if obj.microseconds:
        fmt = "'{0:02d}:{1:02d}:{2:02d}.{3:06d}'"
    else:
        fmt = "'{0:02d}:{1:02d}:{2:02d}'"
    return fmt.format(hours, minutes, seconds, obj.microseconds)


def escape_time(obj, mapping=None):
    """Escape time object for SQL.

    Args:
        obj (datetime.time): Time to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: SQL time string in ISO format with microseconds

    Example:
        >>> t = datetime.time(14, 30, 45, 123456)
        >>> escape_time(t)
        "'14:30:45.123456'"
    """
    return "'{}'".format(obj.isoformat(timespec='microseconds'))


def escape_datetime(obj, mapping=None):
    """Escape datetime object for SQL DATETIME format.

    Args:
        obj (datetime.datetime): Datetime to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: SQL datetime string in ISO format with space separator and microseconds

    Example:
        >>> dt = datetime.datetime(2023, 12, 25, 14, 30, 45, 123456)
        >>> escape_datetime(dt)
        "'2023-12-25 14:30:45.123456'"
    """
    return "'{}'".format(obj.isoformat(sep=' ', timespec='microseconds'))


def escape_date(obj, mapping=None):
    """Escape date object for SQL DATE format.

    Args:
        obj (datetime.date): Date to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: SQL date string in ISO format

    Example:
        >>> d = datetime.date(2023, 12, 25)
        >>> escape_date(d)
        "'2023-12-25'"
    """
    return "'{}'".format(obj.isoformat())


def escape_struct_time(obj, mapping=None):
    """Escape struct_time object for SQL by converting to datetime.

    Args:
        obj (time.struct_time): Struct time to escape
        mapping: Unused, for interface compatibility

    Returns:
        str: SQL datetime string converted from struct_time
    """
    return escape_datetime(datetime.datetime(*obj[:6]))


def _convert_second_fraction(s):
    if not s:
        return 0
    # Pad zeros to ensure the fraction length in microseconds
    s = s.ljust(6, '0')
    return int(s[:6])


def convert_datetime(obj):
    """Convert SQL DATETIME or TIMESTAMP string to datetime object.

    Parses a SQL datetime string and returns a corresponding Python datetime object.
    Handles both string and bytes input.

    Args:
        obj (str or bytes): SQL datetime string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        datetime.datetime: Parsed datetime object

    Raises:
        DataError: If the datetime string format is invalid

    Examples:
        >>> convert_datetime('2007-02-25 23:06:20')
        datetime.datetime(2007, 2, 25, 23, 6, 20)
        >>> convert_datetime(b'2023-12-25 14:30:45')
        datetime.datetime(2023, 12, 25, 14, 30, 45)
    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')

    try:
        time_obj = datetime.datetime.strptime(obj, '%Y-%m-%d %H:%M:%S')
        return time_obj
    except Exception as err:
        raise DataError("Not valid datetime struct: %s" % err)


TIMEDELTA_RE = re.compile(r"(-)?(\d{1,3}):(\d{1,2}):(\d{1,2})(?:.(\d{1,6}))?")


def convert_timedelta(obj):
    """Convert SQL TIME string to timedelta object.

    Parses a SQL TIME string (which can represent time intervals) and returns
    a corresponding Python timedelta object. Supports negative intervals.

    Args:
        obj (str or bytes): SQL TIME string in format '[+|-]HH:MM:SS[.microseconds]'

    Returns:
        datetime.timedelta: Parsed timedelta object
        str: Original string if parsing fails (for compatibility)

    Raises:
        DataError: If the time string format is invalid

    Examples:
        >>> convert_timedelta('25:06:17')
        datetime.timedelta(seconds=90377)
        >>> convert_timedelta('-25:06:17')
        datetime.timedelta(days=-2, seconds=83223)
        >>> convert_timedelta('12:30:45.123456')
        datetime.timedelta(seconds=45045, microseconds=123456)

    Note:
        This function expects TIME format as HH:MM:SS, not DD HH:MM:SS.
        Negative times are supported with leading minus sign.
    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')

    m = TIMEDELTA_RE.match(obj)
    if not m:
        return obj

    try:
        groups = list(m.groups())
        groups[-1] = _convert_second_fraction(groups[-1])
        negate = -1 if groups[0] else 1
        hours, minutes, seconds, microseconds = groups[1:]

        tdelta = datetime.timedelta(
            hours=int(hours),
            minutes=int(minutes),
            seconds=int(seconds),
            microseconds=int(microseconds)
        ) * negate
        return tdelta
    except ValueError as err:
        raise DataError("Not valid time or timedelta struct: %s" % err)


def convert_time(obj):
    """Convert SQL TIME string to time object.

    Parses a SQL TIME string and returns a corresponding Python time object.
    Falls back to timedelta conversion for time intervals.

    Args:
        obj (str or bytes): SQL TIME string in format 'HH:MM:SS'

    Returns:
        datetime.time: Parsed time object for regular times
        datetime.timedelta: Parsed timedelta for time intervals

    Examples:
        >>> convert_time('15:06:17')
        datetime.time(15, 6, 17)
        >>> convert_time('25:06:17')  # Falls back to timedelta
        datetime.timedelta(seconds=90377)
    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')

    try:
        time_obj = datetime.datetime.strptime(obj, '%H:%M:%S')
        return time_obj.time()
    except Exception:
        return convert_timedelta(obj)


def convert_date(obj):
    """Convert SQL DATE string to date object.

    Parses a SQL DATE string and returns a corresponding Python date object.

    Args:
        obj (str or bytes): SQL DATE string in format 'YYYY-MM-DD'

    Returns:
        datetime.date: Parsed date object

    Raises:
        DataError: If the date string format is invalid

    Examples:
        >>> convert_date('2007-02-26')
        datetime.date(2007, 2, 26)
        >>> convert_date(b'2023-12-25')
        datetime.date(2023, 12, 25)
    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')
    try:
        time_obj = datetime.datetime.strptime(obj, '%Y-%m-%d')
        return time_obj.date()
    except Exception as err:
        raise DataError("Not valid date struct: %s" % err)


def convert_set(s):
    """Convert comma-separated string to Python set.

    Args:
        s (str or bytes): Comma-separated values

    Returns:
        set: Set of string values split by comma

    Example:
        >>> convert_set("apple,banana,cherry")
        {'apple', 'banana', 'cherry'}
        >>> convert_set(b"1,2,3")
        {b'1', b'2', b'3'}
    """
    if isinstance(s, (bytes, bytearray)):
        return set(s.split(b","))
    return set(s.split(","))


def convert_characters(connection, data):
    """Convert character data based on connection encoding settings.

    Args:
        connection: Database connection object
        data (bytes): Raw character data from database

    Returns:
        str or bytes: Decoded string if unicode enabled, otherwise raw bytes
    """
    if connection.use_unicode:
        data = data.decode("utf8")
    return data


def convert_column_data(column_type, column_data):
    """Convert database column data to appropriate Python type.

    This function automatically converts database column values to the most
    appropriate Python type based on the column's SQL type.

    Args:
        column_type (str): SQL column type name (e.g., 'time', 'date', 'datetime')
        column_data: Raw column value from database

    Returns:
        Converted Python object appropriate for the column type, or original data if no conversion needed

    Example:
        >>> convert_column_data('date', '2023-12-25')
        datetime.date(2023, 12, 25)
        >>> convert_column_data('time', '14:30:45')
        datetime.time(14, 30, 45)
        >>> convert_column_data('varchar', 'hello')
        'hello'
    """
    data = column_data

    # Null
    if data is None:
        return data

    if not isinstance(column_type, str):
        return data

    column_type = column_type.lower().strip()
    if column_type == 'time':
        data = convert_time(column_data)
    elif column_type == 'date':
        data = convert_date(column_data)
    elif column_type == 'datetime':
        data = convert_datetime(column_data)

    return data


encoders = {
    bool: escape_bool,
    int: escape_int,
    float: escape_float,
    str: escape_unicode,
    tuple: escape_sequence,
    list: escape_sequence,
    set: escape_sequence,
    frozenset: escape_sequence,
    dict: escape_dict,
    type(None): escape_None,
    datetime.date: escape_date,
    datetime.datetime: escape_datetime,
    datetime.timedelta: escape_timedelta,
    datetime.time: escape_time,
    time.struct_time: escape_struct_time,
    Decimal: escape_object,
}
