"""Date and time type constructors for chdb database operations.

This module provides Python DB API 2.0 compliant date and time constructors
for converting between Unix timestamps and Python datetime objects. These
functions are essential for handling temporal data in database operations.

The module provides the following type aliases and constructor functions:

Type Aliases:
    - **Date**: Alias for :class:`datetime.date`
    - **Time**: Alias for :class:`datetime.time`
    - **TimeDelta**: Alias for :class:`datetime.timedelta`
    - **Timestamp**: Alias for :class:`datetime.datetime`

Constructor Functions:
    - :func:`DateFromTicks`: Convert Unix timestamp to date object
    - :func:`TimeFromTicks`: Convert Unix timestamp to time object
    - :func:`TimestampFromTicks`: Convert Unix timestamp to datetime object

These constructors are particularly useful when working with database systems
that store temporal data as Unix timestamps or when interfacing with APIs
that use timestamp representations.

.. note::
    All constructor functions use the local timezone for timestamp conversion.
    For UTC conversions, consider using :func:`datetime.datetime.utcfromtimestamp`.

.. seealso::
    - :mod:`datetime` - Core datetime functionality
    - :mod:`time` - Time access and conversions
    - `Python DB API Specification v2.0 <https://peps.python.org/pep-0249/>`_

Examples:
    >>> import time
    >>> timestamp = time.time()  # Current Unix timestamp
    >>>
    >>> # Convert to different temporal types
    >>> date_obj = DateFromTicks(timestamp)
    >>> time_obj = TimeFromTicks(timestamp)
    >>> datetime_obj = TimestampFromTicks(timestamp)
    >>>
    >>> print(f"Date: {date_obj}")
    >>> print(f"Time: {time_obj}")
    >>> print(f"DateTime: {datetime_obj}")
    Date: 2023-12-25
    Time: 14:30:45
    DateTime: 2023-12-25 14:30:45
"""

from time import localtime
from datetime import date, datetime, time, timedelta


Date = date
"""Type alias for :class:`datetime.date`.

This provides a standard DB API 2.0 compliant alias for the date class,
allowing portable code across different database drivers.

:type: type[datetime.date]
"""

Time = time
"""Type alias for :class:`datetime.time`.

This provides a standard DB API 2.0 compliant alias for the time class,
allowing portable code across different database drivers.

:type: type[datetime.time]
"""

TimeDelta = timedelta
"""Type alias for :class:`datetime.timedelta`.

This provides a standard DB API 2.0 compliant alias for the timedelta class,
useful for representing time intervals and durations in database operations.

:type: type[datetime.timedelta]
"""

Timestamp = datetime
"""Type alias for :class:`datetime.datetime`.

This provides a standard DB API 2.0 compliant alias for the datetime class,
allowing portable code across different database drivers.

:type: type[datetime.datetime]
"""


def DateFromTicks(ticks):
    """Convert a Unix timestamp to a date object.

    This function takes a Unix timestamp (seconds since epoch) and converts
    it to a Python date object representing the date in local time.

    Args:
        ticks (float): Unix timestamp (seconds since January 1, 1970, 00:00 UTC)

    Returns:
        datetime.date: Date object representing the local date for the given timestamp

    Note:
        The conversion uses the local timezone. The time component is ignored,
        only the date portion is extracted.

    Examples:
        >>> # Convert current timestamp to date
        >>> import time
        >>> current_time = time.time()
        >>> today = DateFromTicks(current_time)
        >>> print(today)
        2023-12-25

        >>> # Convert specific timestamp
        >>> timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        >>> date_obj = DateFromTicks(timestamp)
        >>> print(date_obj)
        2021-12-31  # Local time (assuming UTC-5 timezone)
    """
    return date(*localtime(ticks)[:3])


def TimeFromTicks(ticks):
    """Convert a Unix timestamp to a time object.

    This function takes a Unix timestamp (seconds since epoch) and converts
    it to a Python time object representing the time of day in local time.

    Args:
        ticks (float): Unix timestamp (seconds since January 1, 1970, 00:00 UTC)

    Returns:
        datetime.time: Time object representing the local time for the given timestamp

    Note:
        The conversion uses the local timezone. The date component is ignored,
        only the time-of-day portion is extracted.

    Examples:
        >>> # Convert timestamp to time
        >>> timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        >>> time_obj = TimeFromTicks(timestamp)
        >>> print(time_obj)
        19:00:00  # Local time (assuming UTC-5 timezone)

        >>> # Extract current time
        >>> import time
        >>> current_time = time.time()
        >>> now_time = TimeFromTicks(current_time)
        >>> print(f"Current time: {now_time}")
    """
    return time(*localtime(ticks)[3:6])


def TimestampFromTicks(ticks):
    """Convert a Unix timestamp to a datetime object.

    This function takes a Unix timestamp (seconds since epoch) and converts
    it to a Python datetime object representing the complete date and time
    in local time.

    Args:
        ticks (float): Unix timestamp (seconds since January 1, 1970, 00:00 UTC)

    Returns:
        datetime.datetime: DateTime object representing the local datetime for the given timestamp

    Note:
        The conversion uses the local timezone. This provides both date and time
        components, making it the most complete temporal conversion function.

    Examples:
        >>> # Convert timestamp to datetime
        >>> timestamp = 1640995200.0  # 2022-01-01 00:00:00 UTC
        >>> dt_obj = TimestampFromTicks(timestamp)
        >>> print(dt_obj)
        2021-12-31 19:00:00  # Local time (assuming UTC-5 timezone)

        >>> # Convert current timestamp
        >>> import time
        >>> current_time = time.time()
        >>> now_dt = TimestampFromTicks(current_time)
        >>> print(f"Current datetime: {now_dt}")
        Current datetime: 2023-12-25 14:30:45

        >>> # Use in database operations
        >>> cursor.execute("INSERT INTO events (created_at) VALUES (%s)",
        ...               (TimestampFromTicks(time.time()),))
    """
    return datetime(*localtime(ticks)[:6])
