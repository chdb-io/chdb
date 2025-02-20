import datetime
from decimal import Decimal
from .err import DataError
import re
import time


def escape_item(val, mapping=None):
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
    n = {}
    for k, v in val.items():
        quoted = escape_item(v, mapping)
        n[k] = quoted
    return n


def escape_sequence(val, mapping=None):
    n = []
    for item in val:
        quoted = escape_item(item, mapping)
        n.append(quoted)
    return "(" + ",".join(n) + ")"


def escape_set(val, mapping=None):
    return ','.join([escape_item(x, mapping) for x in val])


def escape_bool(value, mapping=None):
    return str(int(value))


def escape_object(value, mapping=None):
    return str(value)


def escape_int(value, mapping=None):
    return str(value)


def escape_float(value, mapping=None):
    return '%.15g' % value


_escape_table = [chr(x) for x in range(128)]
_escape_table[ord("'")] = u"''"
_escape_table[ord("\\")] = "\\\\"


def _escape_unicode(value, mapping=None):
    """escapes *value* with adding single quote.

    Value should be unicode
    """
    return value.translate(_escape_table)


escape_string = _escape_unicode

# On Python ~3.5, str.decode('ascii', 'surrogateescape') is slow.
# (fixed in Python 3.6, http://bugs.python.org/issue24870)
# Workaround is str.decode('latin1') then translate 0x80-0xff into 0udc80-0udcff.
# We can escape special chars and surrogateescape at once.
_escape_bytes_table = _escape_table + [chr(i) for i in range(0xdc80, 0xdd00)]


def escape_bytes(value, mapping=None):
    return "'%s'" % value.decode('latin1').translate(_escape_bytes_table)


def escape_unicode(value, mapping=None):
    return u"'%s'" % _escape_unicode(value)


def escape_str(value, mapping=None):
    return "'%s'" % escape_string(str(value), mapping)


def escape_None(value, mapping=None):
    return 'NULL'


def escape_timedelta(obj, mapping=None):
    seconds = int(obj.seconds) % 60
    minutes = int(obj.seconds // 60) % 60
    hours = int(obj.seconds // 3600) % 24 + int(obj.days) * 24
    if obj.microseconds:
        fmt = "'{0:02d}:{1:02d}:{2:02d}.{3:06d}'"
    else:
        fmt = "'{0:02d}:{1:02d}:{2:02d}'"
    return fmt.format(hours, minutes, seconds, obj.microseconds)


def escape_time(obj, mapping=None):
    return "'{}'".format(obj.isoformat(timespec='microseconds'))


def escape_datetime(obj, mapping=None):
    return "'{}'".format(obj.isoformat(sep=' ', timespec='microseconds'))
    # if obj.microsecond:
    #    fmt = "'{0.year:04}-{0.month:02}-{0.day:02} {0.hour:02}:{0.minute:02}:{0.second:02}.{0.microsecond:06}'"
    # else:
    #    fmt = "'{0.year:04}-{0.month:02}-{0.day:02} {0.hour:02}:{0.minute:02}:{0.second:02}'"
    # return fmt.format(obj)


def escape_date(obj, mapping=None):
    return "'{}'".format(obj.isoformat())


def escape_struct_time(obj, mapping=None):
    return escape_datetime(datetime.datetime(*obj[:6]))


def _convert_second_fraction(s):
    if not s:
        return 0
    # Pad zeros to ensure the fraction length in microseconds
    s = s.ljust(6, '0')
    return int(s[:6])


def convert_datetime(obj):
    """Returns a DATETIME or TIMESTAMP column value as a datetime object:

      >>> datetime_or_None('2007-02-25 23:06:20')
      datetime.datetime(2007, 2, 25, 23, 6, 20)

    Illegal values are raise DataError

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
    """Returns a TIME column as a timedelta object:

      >>> timedelta_or_None('25:06:17')
      datetime.timedelta(1, 3977)
      >>> timedelta_or_None('-25:06:17')
      datetime.timedelta(-2, 83177)

    Illegal values are returned as None:

      >>> timedelta_or_None('random crap') is None
      True

    Note that MySQL always returns TIME columns as (+|-)HH:MM:SS, but
    can accept values as (+|-)DD HH:MM:SS. The latter format will not
    be parsed correctly by this function.
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
    """Returns a TIME column as a time object:

      >>> time_or_None('15:06:17')
      datetime.time(15, 6, 17)

    Illegal values are returned DataError:

    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')

    try:
        time_obj = datetime.datetime.strptime(obj, '%H:%M:%S')
        return time_obj.time()
    except Exception:
        return convert_timedelta(obj)


def convert_date(obj):
    """Returns a DATE column as a date object:

      >>> date_or_None('2007-02-26')
      datetime.date(2007, 2, 26)

    Illegal values are returned as None:

      >>> date_or_None('2007-02-31') is None
      True
      >>> date_or_None('0000-00-00') is None
      True

    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')
    try:
        time_obj = datetime.datetime.strptime(obj, '%Y-%m-%d')
        return time_obj.date()
    except Exception as err:
        raise DataError("Not valid date struct: %s" % err)


def convert_set(s):
    if isinstance(s, (bytes, bytearray)):
        return set(s.split(b","))
    return set(s.split(","))


def convert_characters(connection, data):
    if connection.use_unicode:
        data = data.decode("utf8")
    return data


def convert_column_data(column_type, column_data):
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
