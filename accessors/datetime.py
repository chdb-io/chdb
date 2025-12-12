"""
DateTimeAccessor - Date/Time functions via .dt accessor.

Provides ClickHouse date and time functions in a Pandas-like API.
Maps to ClickHouse date functions: https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..expressions import Expression
    from ..functions import Function


class DateTimeAccessor(BaseAccessor):
    """
    Accessor for date/time functions via .dt property.

    Maps to ClickHouse date/time functions with a Pandas-like interface.

    Example:
        >>> ds['date'].dt.year           # toYear(date)
        >>> ds['date'].dt.month          # toMonth(date)
        >>> ds['ts'].dt.hour             # toHour(ts)
        >>> ds['date'].dt.day_of_week    # toDayOfWeek(date)

    ClickHouse Date Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions
    """

    # ========== Date Part Extraction (Properties) ==========

    @property
    def year(self) -> 'Function':
        """
        Extract year from date/datetime.

        Maps to ClickHouse: toYear(date)

        Returns:
            Function expression for toYear(expr)

        Example:
            >>> ds['date'].dt.year
            >>> # SQL: toYear("date")
        """
        return self._create_function('toYear')

    @property
    def month(self) -> 'Function':
        """
        Extract month from date/datetime (1-12).

        Maps to ClickHouse: toMonth(date)

        Returns:
            Function expression for toMonth(expr)

        Example:
            >>> ds['date'].dt.month
            >>> # SQL: toMonth("date")
        """
        return self._create_function('toMonth')

    @property
    def day(self) -> 'Function':
        """
        Extract day of month from date/datetime (1-31).

        Maps to ClickHouse: toDayOfMonth(date)

        Returns:
            Function expression for toDayOfMonth(expr)

        Example:
            >>> ds['date'].dt.day
            >>> # SQL: toDayOfMonth("date")
        """
        return self._create_function('toDayOfMonth')

    @property
    def hour(self) -> 'Function':
        """
        Extract hour from datetime (0-23).

        Maps to ClickHouse: toHour(datetime)

        Returns:
            Function expression for toHour(expr)

        Example:
            >>> ds['timestamp'].dt.hour
            >>> # SQL: toHour("timestamp")
        """
        return self._create_function('toHour')

    @property
    def minute(self) -> 'Function':
        """
        Extract minute from datetime (0-59).

        Maps to ClickHouse: toMinute(datetime)

        Returns:
            Function expression for toMinute(expr)

        Example:
            >>> ds['timestamp'].dt.minute
            >>> # SQL: toMinute("timestamp")
        """
        return self._create_function('toMinute')

    @property
    def second(self) -> 'Function':
        """
        Extract second from datetime (0-59).

        Maps to ClickHouse: toSecond(datetime)

        Returns:
            Function expression for toSecond(expr)

        Example:
            >>> ds['timestamp'].dt.second
            >>> # SQL: toSecond("timestamp")
        """
        return self._create_function('toSecond')

    @property
    def millisecond(self) -> 'Function':
        """
        Extract millisecond from DateTime64.

        Maps to ClickHouse: toMillisecond(datetime64)

        Returns:
            Function expression for toMillisecond(expr)
        """
        return self._create_function('toMillisecond')

    @property
    def microsecond(self) -> 'Function':
        """
        Extract microsecond from DateTime64.

        Maps to ClickHouse: toMicrosecond(datetime64)

        Returns:
            Function expression for toMicrosecond(expr)
        """
        return self._create_function('toMicrosecond')

    @property
    def quarter(self) -> 'Function':
        """
        Extract quarter from date/datetime (1-4).

        Maps to ClickHouse: toQuarter(date)

        Returns:
            Function expression for toQuarter(expr)

        Example:
            >>> ds['date'].dt.quarter
            >>> # SQL: toQuarter("date")
        """
        return self._create_function('toQuarter')

    @property
    def day_of_week(self) -> 'Function':
        """
        Extract day of week (1=Monday, 7=Sunday).

        Maps to ClickHouse: toDayOfWeek(date)

        Returns:
            Function expression for toDayOfWeek(expr)

        Example:
            >>> ds['date'].dt.day_of_week
            >>> # SQL: toDayOfWeek("date")
        """
        return self._create_function('toDayOfWeek')

    @property
    def dayofweek(self) -> 'Function':
        """Alias for day_of_week. Extract day of week (1=Monday, 7=Sunday)."""
        return self.day_of_week

    @property
    def day_of_year(self) -> 'Function':
        """
        Extract day of year (1-366).

        Maps to ClickHouse: toDayOfYear(date)

        Returns:
            Function expression for toDayOfYear(expr)

        Example:
            >>> ds['date'].dt.day_of_year
            >>> # SQL: toDayOfYear("date")
        """
        return self._create_function('toDayOfYear')

    @property
    def dayofyear(self) -> 'Function':
        """Alias for day_of_year. Extract day of year (1-366)."""
        return self.day_of_year

    @property
    def week(self) -> 'Function':
        """
        Extract week of year (ISO 8601).

        Maps to ClickHouse: toISOWeek(date)

        Returns:
            Function expression for toISOWeek(expr)

        Example:
            >>> ds['date'].dt.week
            >>> # SQL: toISOWeek("date")
        """
        return self._create_function('toISOWeek')

    @property
    def weekofyear(self) -> 'Function':
        """Alias for week. Extract week of year (ISO 8601)."""
        return self.week

    @property
    def iso_year(self) -> 'Function':
        """
        Extract ISO year (may differ from calendar year at year boundaries).

        Maps to ClickHouse: toISOYear(date)

        Returns:
            Function expression for toISOYear(expr)
        """
        return self._create_function('toISOYear')

    # ========== Date Conversion Methods ==========

    def to_date(self, alias: str = None) -> 'Function':
        """
        Convert to Date type (strips time part).

        Maps to ClickHouse: toDate(datetime)

        Returns:
            Function expression for toDate(expr)

        Example:
            >>> ds['timestamp'].dt.to_date()
            >>> # SQL: toDate("timestamp")
        """
        return self._create_function('toDate', alias=alias)

    def to_datetime(self, timezone: str = None, alias: str = None) -> 'Function':
        """
        Convert to DateTime type.

        Maps to ClickHouse: toDateTime(date) or toDateTime(date, timezone)

        Args:
            timezone: Optional timezone string (e.g., 'UTC', 'America/New_York')

        Returns:
            Function expression for toDateTime(expr)

        Example:
            >>> ds['date'].dt.to_datetime()
            >>> ds['date'].dt.to_datetime('UTC')
            >>> # SQL: toDateTime("date") or toDateTime("date", 'UTC')
        """
        if timezone:
            return self._create_function('toDateTime', timezone, alias=alias)
        return self._create_function('toDateTime', alias=alias)

    def to_start_of_day(self, alias: str = None) -> 'Function':
        """
        Round down to start of day.

        Maps to ClickHouse: toStartOfDay(datetime)

        Returns:
            Function expression for toStartOfDay(expr)

        Example:
            >>> ds['timestamp'].dt.to_start_of_day()
            >>> # SQL: toStartOfDay("timestamp")
        """
        return self._create_function('toStartOfDay', alias=alias)

    def to_start_of_week(self, mode: int = 0, alias: str = None) -> 'Function':
        """
        Round down to start of week.

        Maps to ClickHouse: toStartOfWeek(date, mode)

        Args:
            mode: 0 = week starts Sunday, 1 = week starts Monday (default: 0)

        Returns:
            Function expression for toStartOfWeek(expr, mode)

        Example:
            >>> ds['date'].dt.to_start_of_week()
            >>> ds['date'].dt.to_start_of_week(1)  # Monday start
            >>> # SQL: toStartOfWeek("date", 0)
        """
        return self._create_function('toStartOfWeek', mode, alias=alias)

    def to_start_of_month(self, alias: str = None) -> 'Function':
        """
        Round down to start of month.

        Maps to ClickHouse: toStartOfMonth(date)

        Returns:
            Function expression for toStartOfMonth(expr)

        Example:
            >>> ds['date'].dt.to_start_of_month()
            >>> # SQL: toStartOfMonth("date")
        """
        return self._create_function('toStartOfMonth', alias=alias)

    def to_start_of_quarter(self, alias: str = None) -> 'Function':
        """
        Round down to start of quarter.

        Maps to ClickHouse: toStartOfQuarter(date)

        Returns:
            Function expression for toStartOfQuarter(expr)
        """
        return self._create_function('toStartOfQuarter', alias=alias)

    def to_start_of_year(self, alias: str = None) -> 'Function':
        """
        Round down to start of year.

        Maps to ClickHouse: toStartOfYear(date)

        Returns:
            Function expression for toStartOfYear(expr)
        """
        return self._create_function('toStartOfYear', alias=alias)

    def to_start_of_hour(self, alias: str = None) -> 'Function':
        """
        Round down to start of hour.

        Maps to ClickHouse: toStartOfHour(datetime)

        Returns:
            Function expression for toStartOfHour(expr)
        """
        return self._create_function('toStartOfHour', alias=alias)

    def to_start_of_minute(self, alias: str = None) -> 'Function':
        """
        Round down to start of minute.

        Maps to ClickHouse: toStartOfMinute(datetime)

        Returns:
            Function expression for toStartOfMinute(expr)
        """
        return self._create_function('toStartOfMinute', alias=alias)

    def to_start_of_second(self, alias: str = None) -> 'Function':
        """
        Round down to start of second.

        Maps to ClickHouse: toStartOfSecond(datetime64)

        Returns:
            Function expression for toStartOfSecond(expr)
        """
        return self._create_function('toStartOfSecond', alias=alias)

    # ========== Date Arithmetic Methods ==========

    def add_years(self, n: int, alias: str = None) -> 'Function':
        """
        Add years to date/datetime.

        Maps to ClickHouse: addYears(date, n)

        Args:
            n: Number of years to add (can be negative)

        Returns:
            Function expression for addYears(expr, n)

        Example:
            >>> ds['date'].dt.add_years(1)
            >>> # SQL: addYears("date", 1)
        """
        return self._create_function('addYears', n, alias=alias)

    def add_months(self, n: int, alias: str = None) -> 'Function':
        """
        Add months to date/datetime.

        Maps to ClickHouse: addMonths(date, n)

        Args:
            n: Number of months to add (can be negative)

        Returns:
            Function expression for addMonths(expr, n)

        Example:
            >>> ds['date'].dt.add_months(3)
            >>> # SQL: addMonths("date", 3)
        """
        return self._create_function('addMonths', n, alias=alias)

    def add_weeks(self, n: int, alias: str = None) -> 'Function':
        """
        Add weeks to date/datetime.

        Maps to ClickHouse: addWeeks(date, n)

        Args:
            n: Number of weeks to add (can be negative)

        Returns:
            Function expression for addWeeks(expr, n)
        """
        return self._create_function('addWeeks', n, alias=alias)

    def add_days(self, n: int, alias: str = None) -> 'Function':
        """
        Add days to date/datetime.

        Maps to ClickHouse: addDays(date, n)

        Args:
            n: Number of days to add (can be negative)

        Returns:
            Function expression for addDays(expr, n)

        Example:
            >>> ds['date'].dt.add_days(7)
            >>> # SQL: addDays("date", 7)
        """
        return self._create_function('addDays', n, alias=alias)

    def add_hours(self, n: int, alias: str = None) -> 'Function':
        """
        Add hours to datetime.

        Maps to ClickHouse: addHours(datetime, n)

        Args:
            n: Number of hours to add (can be negative)

        Returns:
            Function expression for addHours(expr, n)
        """
        return self._create_function('addHours', n, alias=alias)

    def add_minutes(self, n: int, alias: str = None) -> 'Function':
        """
        Add minutes to datetime.

        Maps to ClickHouse: addMinutes(datetime, n)

        Args:
            n: Number of minutes to add (can be negative)

        Returns:
            Function expression for addMinutes(expr, n)
        """
        return self._create_function('addMinutes', n, alias=alias)

    def add_seconds(self, n: int, alias: str = None) -> 'Function':
        """
        Add seconds to datetime.

        Maps to ClickHouse: addSeconds(datetime, n)

        Args:
            n: Number of seconds to add (can be negative)

        Returns:
            Function expression for addSeconds(expr, n)
        """
        return self._create_function('addSeconds', n, alias=alias)

    def sub_years(self, n: int, alias: str = None) -> 'Function':
        """Subtract years from date/datetime. Maps to subtractYears()."""
        return self._create_function('subtractYears', n, alias=alias)

    def sub_months(self, n: int, alias: str = None) -> 'Function':
        """Subtract months from date/datetime. Maps to subtractMonths()."""
        return self._create_function('subtractMonths', n, alias=alias)

    def sub_weeks(self, n: int, alias: str = None) -> 'Function':
        """Subtract weeks from date/datetime. Maps to subtractWeeks()."""
        return self._create_function('subtractWeeks', n, alias=alias)

    def sub_days(self, n: int, alias: str = None) -> 'Function':
        """Subtract days from date/datetime. Maps to subtractDays()."""
        return self._create_function('subtractDays', n, alias=alias)

    def sub_hours(self, n: int, alias: str = None) -> 'Function':
        """Subtract hours from datetime. Maps to subtractHours()."""
        return self._create_function('subtractHours', n, alias=alias)

    def sub_minutes(self, n: int, alias: str = None) -> 'Function':
        """Subtract minutes from datetime. Maps to subtractMinutes()."""
        return self._create_function('subtractMinutes', n, alias=alias)

    def sub_seconds(self, n: int, alias: str = None) -> 'Function':
        """Subtract seconds from datetime. Maps to subtractSeconds()."""
        return self._create_function('subtractSeconds', n, alias=alias)

    # ========== Date Difference Methods ==========

    def diff(self, other: 'Expression', unit: str = 'day', alias: str = None) -> 'Function':
        """
        Calculate difference between two dates in specified unit.

        Maps to ClickHouse: dateDiff(unit, startdate, enddate)

        Args:
            other: The other date expression to compare with
            unit: Unit for difference ('year', 'quarter', 'month', 'week',
                  'day', 'hour', 'minute', 'second')

        Returns:
            Function expression for dateDiff(unit, self, other)

        Example:
            >>> ds['start_date'].dt.diff(ds['end_date'], 'day')
            >>> # SQL: dateDiff('day', "start_date", "end_date")
        """
        from ..functions import Function
        from ..expressions import Literal

        other_expr = other if hasattr(other, 'to_sql') else Literal(other)
        return Function('dateDiff', Literal(unit), self._expr, other_expr, alias=alias)

    def days_diff(self, other: 'Expression', alias: str = None) -> 'Function':
        """
        Calculate difference in days between two dates.

        Args:
            other: The other date expression

        Returns:
            Function expression for dateDiff('day', self, other)

        Example:
            >>> ds['start'].dt.days_diff(ds['end'])
        """
        return self.diff(other, 'day', alias=alias)

    def months_diff(self, other: 'Expression', alias: str = None) -> 'Function':
        """Calculate difference in months between two dates."""
        return self.diff(other, 'month', alias=alias)

    def years_diff(self, other: 'Expression', alias: str = None) -> 'Function':
        """Calculate difference in years between two dates."""
        return self.diff(other, 'year', alias=alias)

    def hours_diff(self, other: 'Expression', alias: str = None) -> 'Function':
        """Calculate difference in hours between two datetimes."""
        return self.diff(other, 'hour', alias=alias)

    def minutes_diff(self, other: 'Expression', alias: str = None) -> 'Function':
        """Calculate difference in minutes between two datetimes."""
        return self.diff(other, 'minute', alias=alias)

    def seconds_diff(self, other: 'Expression', alias: str = None) -> 'Function':
        """Calculate difference in seconds between two datetimes."""
        return self.diff(other, 'second', alias=alias)

    # ========== Date Truncation ==========

    def date_trunc(self, unit: str, alias: str = None) -> 'Function':
        """
        Truncate date/datetime to specified unit.

        Maps to ClickHouse: date_trunc(unit, date)

        Args:
            unit: Unit to truncate to ('year', 'quarter', 'month', 'week',
                  'day', 'hour', 'minute', 'second')

        Returns:
            Function expression for date_trunc(unit, expr)

        Example:
            >>> ds['timestamp'].dt.date_trunc('month')
            >>> # SQL: date_trunc('month', "timestamp")
        """
        from ..functions import Function
        from ..expressions import Literal

        return Function('date_trunc', Literal(unit), self._expr, alias=alias)

    def truncate(self, unit: str, alias: str = None) -> 'Function':
        """Alias for date_trunc(). Truncate to specified unit."""
        return self.date_trunc(unit, alias=alias)

    # ========== Formatting ==========

    def format(self, format_string: str, alias: str = None) -> 'Function':
        """
        Format date/datetime as string.

        Maps to ClickHouse: formatDateTime(datetime, format)

        Args:
            format_string: Format string (e.g., '%Y-%m-%d', '%H:%M:%S')

        Returns:
            Function expression for formatDateTime(expr, format)

        Example:
            >>> ds['date'].dt.format('%Y-%m-%d')
            >>> # SQL: formatDateTime("date", '%Y-%m-%d')
        """
        return self._create_function('formatDateTime', format_string, alias=alias)

    def strftime(self, format_string: str, alias: str = None) -> 'Function':
        """Alias for format(). Format date/datetime as string."""
        return self.format(format_string, alias=alias)

    # ========== Timezone Operations ==========

    def to_timezone(self, timezone: str, alias: str = None) -> 'Function':
        """
        Convert datetime to different timezone.

        Maps to ClickHouse: toTimezone(datetime, timezone)

        Args:
            timezone: Target timezone (e.g., 'UTC', 'America/New_York')

        Returns:
            Function expression for toTimezone(expr, timezone)

        Example:
            >>> ds['timestamp'].dt.to_timezone('UTC')
            >>> # SQL: toTimezone("timestamp", 'UTC')
        """
        return self._create_function('toTimezone', timezone, alias=alias)

    @property
    def timezone(self) -> 'Function':
        """
        Get timezone of datetime value.

        Maps to ClickHouse: timezone(datetime)

        Returns:
            Function expression for timezone(expr)
        """
        return self._create_function('timezone')

    # ========== Unix Timestamp Conversion ==========

    def to_unix_timestamp(self, alias: str = None) -> 'Function':
        """
        Convert to Unix timestamp (seconds since epoch).

        Maps to ClickHouse: toUnixTimestamp(datetime)

        Returns:
            Function expression for toUnixTimestamp(expr)

        Example:
            >>> ds['timestamp'].dt.to_unix_timestamp()
            >>> # SQL: toUnixTimestamp("timestamp")
        """
        return self._create_function('toUnixTimestamp', alias=alias)

    # ========== Utility Properties ==========

    @property
    def date(self) -> 'Function':
        """Extract date part from datetime. Alias for to_date()."""
        return self._create_function('toDate')

    @property
    def time(self) -> 'Function':
        """
        Extract time part from datetime as string.

        Maps to ClickHouse: formatDateTime(dt, '%H:%M:%S')

        Returns:
            Function expression for time extraction
        """
        return self._create_function('toTime')

    @property
    def is_weekend(self) -> 'Function':
        """
        Check if date is weekend (Saturday or Sunday).

        Maps to ClickHouse: toDayOfWeek(date) >= 6

        Returns:
            Condition expression
        """
        from ..expressions import Literal

        return self._create_function('toDayOfWeek') >= Literal(6)

    @property
    def is_weekday(self) -> 'Function':
        """
        Check if date is weekday (Monday-Friday).

        Maps to ClickHouse: toDayOfWeek(date) < 6

        Returns:
            Condition expression
        """
        from ..expressions import Literal

        return self._create_function('toDayOfWeek') < Literal(6)

    @property
    def is_leap_year(self) -> 'Function':
        """
        Check if year is leap year.

        Maps to ClickHouse: toDayOfYear(last day of year) = 366

        Returns:
            Function expression for leap year check
        """
        # Use modulo check: (year % 4 = 0) AND ((year % 100 != 0) OR (year % 400 = 0))
        return self._create_function('isLeapYear')

    @property
    def days_in_month(self) -> 'Function':
        """
        Get number of days in the month.

        Maps to ClickHouse: toDayOfMonth(toLastDayOfMonth(date))

        Returns:
            Function expression for days in month
        """
        from ..functions import Function

        last_day = Function('toLastDayOfMonth', self._expr)
        return Function('toDayOfMonth', last_day)
