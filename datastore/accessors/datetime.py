"""
DateTimeAccessor - Date/Time functions via .dt accessor.

Provides ClickHouse date and time functions in a Pandas-like API.
All methods are dynamically injected from the FunctionRegistry.
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class DateTimeAccessor(BaseAccessor):
    """
    Accessor for date/time functions via .dt property.

    Maps to ClickHouse date/time functions with a Pandas-like interface.
    Methods are automatically injected from FunctionRegistry.

    Example:
        >>> ds['date'].dt.year           # toYear(date)
        >>> ds['date'].dt.month          # toMonth(date)
        >>> ds['ts'].dt.hour             # toHour(ts)
        >>> ds['date'].dt.day_of_week    # toDayOfWeek(date)

    ClickHouse Date Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions
    """

    # Properties for common date parts (these must be properties, not methods)
    @property
    def year(self) -> 'Function':
        """Extract year from date/datetime. Maps to toYear(x)."""
        return self._create_function('toYear')

    @property
    def month(self) -> 'Function':
        """Extract month from date/datetime (1-12). Maps to toMonth(x)."""
        return self._create_function('toMonth')

    @property
    def day(self) -> 'Function':
        """Extract day of month from date/datetime (1-31). Maps to toDayOfMonth(x)."""
        return self._create_function('toDayOfMonth')

    @property
    def hour(self) -> 'Function':
        """Extract hour from datetime (0-23). Maps to toHour(x)."""
        return self._create_function('toHour')

    @property
    def minute(self) -> 'Function':
        """Extract minute from datetime (0-59). Maps to toMinute(x)."""
        return self._create_function('toMinute')

    @property
    def second(self) -> 'Function':
        """Extract second from datetime (0-59). Maps to toSecond(x)."""
        return self._create_function('toSecond')

    @property
    def day_of_week(self) -> 'Function':
        """Day of week (1=Monday, 7=Sunday). Maps to toDayOfWeek(x)."""
        return self._create_function('toDayOfWeek')

    @property
    def dayofweek(self) -> 'Function':
        """Alias for day_of_week."""
        return self.day_of_week

    @property
    def day_of_year(self) -> 'Function':
        """Day of year (1-365/366). Maps to toDayOfYear(x)."""
        return self._create_function('toDayOfYear')

    @property
    def dayofyear(self) -> 'Function':
        """Alias for day_of_year."""
        return self.day_of_year

    @property
    def quarter(self) -> 'Function':
        """Quarter (1-4). Maps to toQuarter(x)."""
        return self._create_function('toQuarter')

    @property
    def week(self) -> 'Function':
        """Week number. Maps to toWeek(x)."""
        return self._create_function('toWeek')

    @property
    def weekofyear(self) -> 'Function':
        """Alias for week."""
        return self.week

    @property
    def millisecond(self) -> 'Function':
        """Extract millisecond (0-999). Maps to toMillisecond(x)."""
        return self._create_function('toMillisecond')

    @property
    def daysinmonth(self) -> 'Function':
        """Alias for days_in_month (pandas compatibility)."""
        return self._create_function('toDaysInMonth')

    def isocalendar(self):
        """
        Return a DataFrame with ISO year, week number, and weekday.

        Returns a DataFrame with columns 'year', 'week', and 'day'.
        ISO week date: Monday is day 1, weeks start on Monday.

        Returns:
            IsoCalendarResult: Object with .year, .week, .day properties
        """
        return IsoCalendarResult(self._expr)

    def day_name(self, locale: str = None) -> 'Function':
        """
        Return the day names (Monday, Tuesday, etc.).

        Args:
            locale: Locale for day names (ignored, always English)

        Returns:
            Series with day names
        """
        # ClickHouse: dateName('weekday', date) returns 'Monday', 'Tuesday', etc.
        return self._create_function('dateName', 'weekday')

    def month_name(self, locale: str = None) -> 'Function':
        """
        Return the month names (January, February, etc.).

        Args:
            locale: Locale for month names (ignored, always English)

        Returns:
            Series with month names
        """
        # ClickHouse: dateName('month', date) returns 'January', 'February', etc.
        return self._create_function('dateName', 'month')

    # Standard methods will be injected from registry below


class IsoCalendarResult:
    """
    Result of dt.isocalendar() - provides access to ISO calendar components.

    Supports:
        - .year: ISO year
        - .week: ISO week number (1-53)
        - .day: ISO day of week (1=Monday, 7=Sunday)
        - Direct iteration/execution returns DataFrame with all three columns
    """

    def __init__(self, expr):
        self._expr = expr

    @property
    def year(self):
        """ISO year. Maps to toISOYear(x)."""
        from ..functions import Function

        return Function('toISOYear', self._expr)

    @property
    def week(self):
        """ISO week number (1-53). Maps to toISOWeek(x)."""
        from ..functions import Function

        return Function('toISOWeek', self._expr)

    @property
    def day(self):
        """ISO day of week (1=Monday, 7=Sunday). Maps to toDayOfWeek(x, 1)."""
        from ..functions import Function
        from ..expressions import Literal

        # toDayOfWeek(date, mode) where mode=1 means Monday=1
        return Function('toDayOfWeek', self._expr, Literal(1))

    def __repr__(self):
        return f"IsoCalendarResult({self._expr})"

    def __len__(self):
        """Trigger execution and return length."""
        return len(self._to_datastore())

    @property
    def values(self):
        """Return as numpy array."""
        return self._to_datastore().values

    @property
    def columns(self):
        """Return column names."""
        return ['year', 'week', 'day']

    def _to_datastore(self):
        """Convert to DataStore with year, week, day columns."""
        from ..core import DataStore

        # Get the source DataStore from the expression
        source_ds = self._expr._datastore

        # Create new DataStore with isocalendar columns
        return source_ds.assign(year=self.year, week=self.week, day=self.day)[['year', 'week', 'day']]

    def __getitem__(self, key):
        """Allow column access like result['week']."""
        if key == 'year':
            return self.year
        elif key == 'week':
            return self.week
        elif key == 'day':
            return self.day
        else:
            raise KeyError(f"Unknown column: {key}")


# =============================================================================
# INJECT DATETIME METHODS FROM REGISTRY
# =============================================================================


def _inject_datetime_accessor_methods():
    """Inject datetime methods from registry into DateTimeAccessor class."""
    from ..function_registry import FunctionRegistry, FunctionCategory
    from .. import function_definitions  # noqa: F401

    function_definitions.ensure_functions_registered()

    for spec in FunctionRegistry.get_by_category(FunctionCategory.DATETIME):
        # Skip if already exists (properties defined above)
        if hasattr(DateTimeAccessor, spec.name):
            continue

        # Create method that calls the registry's sql_builder with self._expr
        def make_method(func_spec):
            def method(self, *args, alias=None, **kwargs):
                return func_spec.sql_builder(self._expr, *args, alias=alias, **kwargs)

            method.__name__ = func_spec.name
            method.__doc__ = func_spec.doc
            return method

        setattr(DateTimeAccessor, spec.name, make_method(spec))

        # Also set aliases
        for alias_name in spec.aliases:
            if not hasattr(DateTimeAccessor, alias_name):
                setattr(DateTimeAccessor, alias_name, getattr(DateTimeAccessor, spec.name))


# Perform injection when module is loaded
_inject_datetime_accessor_methods()
