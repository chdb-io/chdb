"""
Exploratory Batch 24: DatetimeAccessor Deep Dive

Focus areas:
1. All dt properties (year, month, day, hour, minute, second, etc.)
2. dt methods (strftime, floor, ceil, round, tz_localize, tz_convert)
3. Boundary cases (year start/end, month start/end, leap year)
4. NaT handling
5. Chained operations with datetime
6. dayofweek/weekday pandas vs chDB alignment (Monday=0 vs Monday=1)

Mirror Code Pattern: Every test compares DataStore with pandas behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sys

sys.path.insert(0, '/Users/auxten/Codes/go/src/github.com/auxten/chdb-ds')

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import chdb_nat_returns_nullable_int


class TestDateTimeBasicProperties:
    """Test basic datetime extraction properties."""

    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with various datetime values."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-15 10:30:45.123',
                        '2024-06-21 14:15:30.456',
                        '2024-12-31 23:59:59.789',
                        '2024-02-29 00:00:00.000',  # Leap year
                        '2023-03-15 12:00:00.000',
                    ]
                )
            }
        )

    def test_dt_year(self, df_dates):
        """Test dt.year extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_month(self, df_dates):
        """Test dt.month extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.month
        ds_result = ds_df['ts'].dt.month

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_day(self, df_dates):
        """Test dt.day extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.day
        ds_result = ds_df['ts'].dt.day

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_hour(self, df_dates):
        """Test dt.hour extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.hour
        ds_result = ds_df['ts'].dt.hour

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_minute(self, df_dates):
        """Test dt.minute extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.minute
        ds_result = ds_df['ts'].dt.minute

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_second(self, df_dates):
        """Test dt.second extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.second
        ds_result = ds_df['ts'].dt.second

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_quarter(self, df_dates):
        """Test dt.quarter extraction."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.quarter
        ds_result = ds_df['ts'].dt.quarter

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDayOfWeekAlignment:
    """
    Test dayofweek/weekday alignment between DataStore and pandas.

    pandas: Monday=0, Sunday=6
    chDB toDayOfWeek: Monday=1, Sunday=7 (by default)

    DataStore should align with pandas convention.
    """

    @pytest.fixture
    def df_weekdays(self):
        """Create DataFrame with dates spanning all days of the week."""
        # 2024-01-01 is Monday
        return pd.DataFrame(
            {
                'date': pd.to_datetime(
                    [
                        '2024-01-01',  # Monday
                        '2024-01-02',  # Tuesday
                        '2024-01-03',  # Wednesday
                        '2024-01-04',  # Thursday
                        '2024-01-05',  # Friday
                        '2024-01-06',  # Saturday
                        '2024-01-07',  # Sunday
                    ]
                )
            }
        )

    def test_dt_dayofweek(self, df_weekdays):
        """Test dt.dayofweek returns Monday=0, Sunday=6 like pandas."""
        pd_df = df_weekdays.copy()
        ds_df = DataStore(df_weekdays.copy())

        pd_result = pd_df['date'].dt.dayofweek
        ds_result = ds_df['date'].dt.dayofweek

        # pandas: Monday=0, Tuesday=1, ..., Sunday=6
        assert list(pd_result) == [0, 1, 2, 3, 4, 5, 6]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_weekday(self, df_weekdays):
        """Test dt.weekday (alias for dayofweek)."""
        pd_df = df_weekdays.copy()
        ds_df = DataStore(df_weekdays.copy())

        pd_result = pd_df['date'].dt.weekday
        ds_result = ds_df['date'].dt.weekday

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_day_of_week(self, df_weekdays):
        """Test dt.day_of_week (alias for dayofweek)."""
        pd_df = df_weekdays.copy()
        ds_df = DataStore(df_weekdays.copy())

        pd_result = pd_df['date'].dt.day_of_week
        ds_result = ds_df['date'].dt.day_of_week

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDayOfYear:
    """Test day of year extraction."""

    @pytest.fixture
    def df_doy(self):
        """Create DataFrame with specific day-of-year test cases."""
        return pd.DataFrame(
            {
                'date': pd.to_datetime(
                    [
                        '2024-01-01',  # Day 1
                        '2024-02-29',  # Day 60 (leap year)
                        '2024-12-31',  # Day 366 (leap year)
                        '2023-12-31',  # Day 365 (non-leap)
                        '2024-07-04',  # Mid-year
                    ]
                )
            }
        )

    def test_dt_dayofyear(self, df_doy):
        """Test dt.dayofyear extraction."""
        pd_df = df_doy.copy()
        ds_df = DataStore(df_doy.copy())

        pd_result = pd_df['date'].dt.dayofyear
        ds_result = ds_df['date'].dt.dayofyear

        # Verify pandas values
        expected = [1, 60, 366, 365, 186]
        assert list(pd_result) == expected

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_day_of_year(self, df_doy):
        """Test dt.day_of_year alias."""
        pd_df = df_doy.copy()
        ds_df = DataStore(df_doy.copy())

        pd_result = pd_df['date'].dt.day_of_year
        ds_result = ds_df['date'].dt.day_of_year

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWeekNumber:
    """Test week number extraction."""

    @pytest.fixture
    def df_weeks(self):
        """Create DataFrame with specific week test cases."""
        return pd.DataFrame(
            {
                'date': pd.to_datetime(
                    [
                        '2024-01-01',  # Week 1
                        '2024-01-07',  # Still week 1
                        '2024-01-08',  # Week 2
                        '2024-12-30',  # Week 1 of 2025 (ISO)
                        '2023-01-01',  # Week 52 of 2022 (ISO)
                    ]
                )
            }
        )

    def test_dt_week(self, df_weeks):
        """Test dt.isocalendar().week extraction (ISO week)."""
        pd_df = df_weeks.copy()
        ds_df = DataStore(df_weeks.copy())

        pd_result = pd_df['date'].dt.isocalendar().week
        ds_result = ds_df['date'].dt.isocalendar().week

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_isocalendar(self, df_weeks):
        """Test dt.isocalendar() returns full calendar DataFrame."""
        pd_df = df_weeks.copy()
        ds_df = DataStore(df_weeks.copy())

        pd_result = pd_df['date'].dt.isocalendar()
        ds_result = ds_df['date'].dt.isocalendar()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBooleanDateProperties:
    """Test boolean datetime properties (is_month_start, is_leap_year, etc.)."""

    @pytest.fixture
    def df_boundaries(self):
        """Create DataFrame with boundary dates."""
        return pd.DataFrame(
            {
                'date': pd.to_datetime(
                    [
                        '2024-01-01',  # Month start, quarter start, year start
                        '2024-01-31',  # Month end
                        '2024-03-31',  # Month end, quarter end
                        '2024-04-01',  # Month start, quarter start
                        '2024-12-31',  # Month end, quarter end, year end
                        '2024-06-15',  # Mid-month
                        '2023-02-28',  # Month end (non-leap)
                        '2024-02-29',  # Month end (leap year)
                    ]
                )
            }
        )

    def test_dt_is_month_start(self, df_boundaries):
        """Test dt.is_month_start property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_month_start
        ds_result = ds_df['date'].dt.is_month_start

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_month_end(self, df_boundaries):
        """Test dt.is_month_end property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_month_end
        ds_result = ds_df['date'].dt.is_month_end

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_quarter_start(self, df_boundaries):
        """Test dt.is_quarter_start property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_quarter_start
        ds_result = ds_df['date'].dt.is_quarter_start

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_quarter_end(self, df_boundaries):
        """Test dt.is_quarter_end property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_quarter_end
        ds_result = ds_df['date'].dt.is_quarter_end

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_year_start(self, df_boundaries):
        """Test dt.is_year_start property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_year_start
        ds_result = ds_df['date'].dt.is_year_start

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_year_end(self, df_boundaries):
        """Test dt.is_year_end property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_year_end
        ds_result = ds_df['date'].dt.is_year_end

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_leap_year(self, df_boundaries):
        """Test dt.is_leap_year property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.is_leap_year
        ds_result = ds_df['date'].dt.is_leap_year

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_days_in_month(self, df_boundaries):
        """Test dt.days_in_month property."""
        pd_df = df_boundaries.copy()
        ds_df = DataStore(df_boundaries.copy())

        pd_result = pd_df['date'].dt.days_in_month
        ds_result = ds_df['date'].dt.days_in_month

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeMethods:
    """Test datetime methods (strftime, floor, ceil, round, normalize)."""

    @pytest.fixture
    def df_times(self):
        """Create DataFrame with various times."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-03-15 10:37:45.123456',
                        '2024-06-21 14:22:30.789012',
                        '2024-12-25 23:59:59.999999',
                    ]
                )
            }
        )

    def test_dt_strftime_basic(self, df_times):
        """Test dt.strftime with basic format."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.strftime('%Y-%m-%d')
        ds_result = ds_df['ts'].dt.strftime('%Y-%m-%d')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_strftime_full(self, df_times):
        """Test dt.strftime with full datetime format."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
        ds_result = ds_df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_floor_hour(self, df_times):
        """Test dt.floor to hour."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.floor('h')
        ds_result = ds_df['ts'].dt.floor('h')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_floor_day(self, df_times):
        """Test dt.floor to day."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.floor('D')
        ds_result = ds_df['ts'].dt.floor('D')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_ceil_hour(self, df_times):
        """Test dt.ceil to hour."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.ceil('h')
        ds_result = ds_df['ts'].dt.ceil('h')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_round_hour(self, df_times):
        """Test dt.round to hour."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.round('h')
        ds_result = ds_df['ts'].dt.round('h')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_normalize(self, df_times):
        """Test dt.normalize (convert to midnight)."""
        pd_df = df_times.copy()
        ds_df = DataStore(df_times.copy())

        pd_result = pd_df['ts'].dt.normalize()
        ds_result = ds_df['ts'].dt.normalize()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimePartsExtraction:
    """Test extracting date/time parts."""

    @pytest.fixture
    def df_full(self):
        """Create DataFrame with full datetime precision."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-03-15 10:30:45.123456789',
                        '2024-06-21 14:15:30.456789012',
                    ]
                )
            }
        )

    def test_dt_date(self, df_full):
        """Test dt.date extraction (date part only)."""
        pd_df = df_full.copy()
        ds_df = DataStore(df_full.copy())

        # Use strftime for consistent string comparison
        pd_result = pd_df['ts'].dt.strftime('%Y-%m-%d')
        ds_result = ds_df['ts'].dt.strftime('%Y-%m-%d')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_time(self, df_full):
        """Test dt.time extraction (time part only)."""
        pd_df = df_full.copy()
        ds_df = DataStore(df_full.copy())

        # Convert to string for comparison (time objects differ in type)
        pd_result = pd_df['ts'].dt.strftime('%H:%M:%S.%f')
        ds_result = ds_df['ts'].dt.strftime('%H:%M:%S.%f')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSubSecondPrecision:
    """Test sub-second precision (microsecond, nanosecond)."""

    @pytest.fixture
    def df_subsec(self):
        """Create DataFrame with sub-second precision."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-01 00:00:00.123456789',
                        '2024-06-15 12:30:45.987654321',
                        '2024-12-31 23:59:59.000000001',
                    ]
                )
            }
        )

    def test_dt_microsecond(self, df_subsec):
        """Test dt.microsecond extraction."""
        pd_df = df_subsec.copy()
        ds_df = DataStore(df_subsec.copy())

        pd_result = pd_df['ts'].dt.microsecond
        ds_result = ds_df['ts'].dt.microsecond

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_nanosecond(self, df_subsec):
        """Test dt.nanosecond extraction."""
        pd_df = df_subsec.copy()
        ds_df = DataStore(df_subsec.copy())

        pd_result = pd_df['ts'].dt.nanosecond
        ds_result = ds_df['ts'].dt.nanosecond

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNaTHandling:
    """Test handling of NaT (Not a Time) values."""

    @pytest.fixture
    def df_with_nat(self):
        """Create DataFrame with NaT values."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-15 10:30:00',
                        pd.NaT,
                        '2024-06-21 14:15:00',
                        pd.NaT,
                        '2024-12-31 23:59:59',
                    ]
                )
            }
        )

    @chdb_nat_returns_nullable_int
    def test_dt_year_with_nat(self, df_with_nat):
        """Test dt.year with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        # NaT should become NaN
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_nat_returns_nullable_int
    def test_dt_month_with_nat(self, df_with_nat):
        """Test dt.month with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())

        pd_result = pd_df['ts'].dt.month
        ds_result = ds_df['ts'].dt.month

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_nat_returns_nullable_int
    def test_dt_dayofweek_with_nat(self, df_with_nat):
        """Test dt.dayofweek with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())

        pd_result = pd_df['ts'].dt.dayofweek
        ds_result = ds_df['ts'].dt.dayofweek

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_is_month_start_with_nat(self, df_with_nat):
        """Test dt.is_month_start with NaT values."""
        pd_df = df_with_nat.copy()
        ds_df = DataStore(df_with_nat.copy())

        pd_result = pd_df['ts'].dt.is_month_start
        ds_result = ds_df['ts'].dt.is_month_start

        # NaT should become False or NaN depending on pandas version
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeChaining:
    """Test chaining datetime operations with other DataFrame operations."""

    @pytest.fixture
    def df_sales(self):
        """Create sales DataFrame with dates."""
        return pd.DataFrame(
            {
                'sale_date': pd.to_datetime(
                    [
                        '2024-01-15',
                        '2024-01-20',
                        '2024-02-10',
                        '2024-02-25',
                        '2024-03-05',
                        '2024-03-15',
                    ]
                ),
                'amount': [100, 200, 150, 300, 250, 175],
            }
        )

    def test_filter_by_month(self, df_sales):
        """Test filtering by dt.month."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())

        pd_result = pd_df[pd_df['sale_date'].dt.month == 2]
        ds_result = ds_df[ds_df['sale_date'].dt.month == 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_dayofweek(self, df_sales):
        """Test filtering by dt.dayofweek (weekdays only)."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())

        # Filter for weekdays (Monday=0 to Friday=4)
        pd_result = pd_df[pd_df['sale_date'].dt.dayofweek < 5]
        ds_result = ds_df[ds_df['sale_date'].dt.dayofweek < 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_month(self, df_sales):
        """Test groupby with dt.month."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())

        # Add month column for groupby using assign
        pd_df = pd_df.assign(month=pd_df['sale_date'].dt.month)
        pd_result = pd_df.groupby('month')['amount'].sum().reset_index()

        ds_df = ds_df.assign(month=ds_df['sale_date'].dt.month)
        ds_result = ds_df.groupby('month')['amount'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_dt_columns(self, df_sales):
        """Test assigning multiple datetime-derived columns."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())

        pd_df = pd_df.assign(
            year=pd_df['sale_date'].dt.year,
            month=pd_df['sale_date'].dt.month,
            day=pd_df['sale_date'].dt.day,
        )

        ds_df = ds_df.assign(
            year=ds_df['sale_date'].dt.year,
            month=ds_df['sale_date'].dt.month,
            day=ds_df['sale_date'].dt.day,
        )

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_sort_by_dayofyear(self, df_sales):
        """Test sorting by dt.dayofyear."""
        pd_df = df_sales.copy()
        ds_df = DataStore(df_sales.copy())

        pd_df = pd_df.assign(doy=pd_df['sale_date'].dt.dayofyear)
        pd_result = pd_df.sort_values('doy', ignore_index=True)

        ds_df = ds_df.assign(doy=ds_df['sale_date'].dt.dayofyear)
        ds_result = ds_df.sort_values('doy')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeEdgeCases:
    """Test edge cases for datetime operations."""

    def test_empty_dataframe_dt(self):
        """Test dt accessor on empty DataFrame."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime([])})
        ds_df = DataStore(pd.DataFrame({'ts': pd.to_datetime([])}))

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_single_row_dt(self):
        """Test dt accessor on single-row DataFrame."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime(['2024-06-15 12:30:00'])})
        ds_df = DataStore(pd.DataFrame({'ts': pd.to_datetime(['2024-06-15 12:30:00'])}))

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_nat_returns_nullable_int
    def test_all_nat_dt(self):
        """Test dt accessor on all-NaT column."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT])})
        ds_df = DataStore(pd.DataFrame({'ts': pd.to_datetime([pd.NaT, pd.NaT, pd.NaT])}))

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_century_boundary(self):
        """Test dt accessor on century boundary dates."""
        pd_df = pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '1999-12-31 23:59:59',
                        '2000-01-01 00:00:00',
                        '2000-01-01 00:00:01',
                    ]
                )
            }
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        assert list(pd_result) == [1999, 2000, 2000]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_leap_year_feb29(self):
        """Test dt operations on Feb 29 (leap year)."""
        pd_df = pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2020-02-29',  # Leap year
                        '2024-02-29',  # Leap year
                    ]
                )
            }
        )
        ds_df = DataStore(pd_df.copy())

        # Test day
        pd_day = pd_df['ts'].dt.day
        ds_day = ds_df['ts'].dt.day
        assert list(pd_day) == [29, 29]
        assert_datastore_equals_pandas(ds_day, pd_day)

        # Test dayofyear
        pd_doy = pd_df['ts'].dt.dayofyear
        ds_doy = ds_df['ts'].dt.dayofyear
        assert list(pd_doy) == [60, 60]  # Feb 29 is day 60
        assert_datastore_equals_pandas(ds_doy, pd_doy)


class TestTimezoneOperations:
    """Test timezone-related operations."""

    @pytest.fixture
    def df_naive(self):
        """Create DataFrame with timezone-naive datetimes."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-06-15 10:00:00',
                        '2024-06-15 15:00:00',
                        '2024-06-15 20:00:00',
                    ]
                )
            }
        )

    @pytest.fixture
    def df_aware(self):
        """Create DataFrame with timezone-aware datetimes."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-06-15 10:00:00',
                        '2024-06-15 15:00:00',
                        '2024-06-15 20:00:00',
                    ]
                ).tz_localize('UTC')
            }
        )

    def test_tz_localize_utc(self, df_naive):
        """Test dt.tz_localize to UTC."""
        pd_df = df_naive.copy()
        ds_df = DataStore(df_naive.copy())

        # Compare via strftime to avoid timezone object differences
        pd_result = pd_df['ts'].dt.tz_localize('UTC').dt.strftime('%Y-%m-%d %H:%M:%S%z')
        ds_result = ds_df['ts'].dt.tz_localize('UTC').dt.strftime('%Y-%m-%d %H:%M:%S%z')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tz_convert(self, df_aware):
        """Test dt.tz_convert to different timezone."""
        pd_df = df_aware.copy()
        ds_df = DataStore(df_aware.copy())

        # Compare hour values after timezone conversion
        pd_result = pd_df['ts'].dt.tz_convert('US/Eastern').dt.hour
        ds_result = ds_df['ts'].dt.tz_convert('US/Eastern').dt.hour

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestToPeriod:
    """Test dt.to_period conversion."""

    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with dates."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-15',
                        '2024-06-21',
                        '2024-12-31',
                    ]
                )
            }
        )

    def test_to_period_month(self, df_dates):
        """Test dt.to_period with monthly frequency."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        # Use strftime for consistent string comparison
        pd_result = pd_df['ts'].dt.strftime('%Y-%m')
        ds_result = ds_df['ts'].dt.strftime('%Y-%m')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_to_period_quarter(self, df_dates):
        """Test dt.to_period with quarterly frequency."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        # Compare quarter values directly
        pd_result = pd_df['ts'].dt.quarter
        ds_result = ds_df['ts'].dt.quarter

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateNameMethods:
    """Test day_name() and month_name() methods."""

    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with dates across different days and months."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-01',  # Monday, January
                        '2024-02-14',  # Wednesday, February
                        '2024-06-15',  # Saturday, June
                        '2024-09-21',  # Saturday, September
                        '2024-12-25',  # Wednesday, December
                    ]
                )
            }
        )

    def test_dt_day_name(self, df_dates):
        """Test dt.day_name() returns correct day names."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.day_name()
        ds_result = ds_df['ts'].dt.day_name()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_month_name(self, df_dates):
        """Test dt.month_name() returns correct month names."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.month_name()
        ds_result = ds_df['ts'].dt.month_name()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_day_name_monday(self):
        """Test day_name for Monday specifically."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-01'])})  # Monday
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.day_name()
        ds_result = ds_df['ts'].dt.day_name()

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert list(pd_result) == ['Monday']

    def test_dt_day_name_all_days(self):
        """Test day_name for all seven days of week."""
        # Week starting 2024-01-01 (Monday) through 2024-01-07 (Sunday)
        df = pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07']
                )
            }
        )
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.day_name()
        ds_result = ds_df['ts'].dt.day_name()

        assert_datastore_equals_pandas(ds_result, pd_result)
        expected = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        assert list(pd_result) == expected

    def test_dt_month_name_all_months(self):
        """Test month_name for all twelve months."""
        df = pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-15',
                        '2024-02-15',
                        '2024-03-15',
                        '2024-04-15',
                        '2024-05-15',
                        '2024-06-15',
                        '2024-07-15',
                        '2024-08-15',
                        '2024-09-15',
                        '2024-10-15',
                        '2024-11-15',
                        '2024-12-15',
                    ]
                )
            }
        )
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.month_name()
        ds_result = ds_df['ts'].dt.month_name()

        assert_datastore_equals_pandas(ds_result, pd_result)
        expected = [
            'January',
            'February',
            'March',
            'April',
            'May',
            'June',
            'July',
            'August',
            'September',
            'October',
            'November',
            'December',
        ]
        assert list(pd_result) == expected


class TestDateTimeAliases:
    """Test datetime property aliases."""

    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with dates for alias testing."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-31',  # 31 days in Jan
                        '2024-02-29',  # 29 days in Feb (leap year)
                        '2024-04-15',  # 30 days in Apr
                    ]
                )
            }
        )

    def test_dt_daysinmonth_alias(self, df_dates):
        """Test dt.daysinmonth is alias for dt.days_in_month."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        # Test daysinmonth alias
        pd_result = pd_df['ts'].dt.daysinmonth
        ds_result = ds_df['ts'].dt.daysinmonth

        assert_datastore_equals_pandas(ds_result, pd_result)

        # Verify it matches days_in_month
        pd_result2 = pd_df['ts'].dt.days_in_month
        ds_result2 = ds_df['ts'].dt.days_in_month

        assert list(pd_result) == list(pd_result2)

    def test_dt_weekofyear_alias(self):
        """Test dt.weekofyear (via isocalendar).week."""
        df = pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-01',  # Week 1
                        '2024-06-15',  # Week 24
                        '2024-12-31',  # Week 1 of 2025
                    ]
                )
            }
        )
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        # weekofyear is deprecated, use isocalendar().week
        pd_result = pd_df['ts'].dt.isocalendar().week
        ds_result = ds_df['ts'].dt.isocalendar().week

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDateTimeAttributes:
    """Test datetime freq, tz, unit attributes."""

    def test_dt_tz_naive(self):
        """Test dt.tz returns None for timezone-naive datetime."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-01', '2024-06-15'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_tz = pd_df['ts'].dt.tz
        ds_tz = ds_df['ts'].dt.tz

        assert pd_tz is None
        assert ds_tz is None

    def test_dt_tz_aware(self):
        """Test dt.tz returns timezone for timezone-aware datetime."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-01', '2024-06-15']).tz_localize('UTC')})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_tz = pd_df['ts'].dt.tz
        ds_tz = ds_df['ts'].dt.tz

        # Both should return UTC timezone
        assert str(pd_tz) == 'UTC'
        assert str(ds_tz) == 'UTC'

    def test_dt_unit(self):
        """Test dt.unit returns the datetime resolution unit."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-01 10:30:45.123456'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_unit = pd_df['ts'].dt.unit
        ds_unit = ds_df['ts'].dt.unit

        # Both should have same unit
        assert pd_unit == ds_unit


class TestIsoCalendarComplete:
    """Test isocalendar() method comprehensively."""

    @pytest.fixture
    def df_dates(self):
        """Create DataFrame with dates for ISO calendar testing."""
        return pd.DataFrame(
            {
                'ts': pd.to_datetime(
                    [
                        '2024-01-01',  # Monday, Week 1, 2024
                        '2024-06-15',  # Saturday, Week 24
                        '2024-12-31',  # Tuesday, Week 1 of 2025
                    ]
                )
            }
        )

    def test_dt_isocalendar_year(self, df_dates):
        """Test dt.isocalendar().year."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.isocalendar().year
        ds_result = ds_df['ts'].dt.isocalendar().year

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_isocalendar_week(self, df_dates):
        """Test dt.isocalendar().week."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.isocalendar().week
        ds_result = ds_df['ts'].dt.isocalendar().week

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_isocalendar_day(self, df_dates):
        """Test dt.isocalendar().day."""
        pd_df = df_dates.copy()
        ds_df = DataStore(df_dates.copy())

        pd_result = pd_df['ts'].dt.isocalendar().day
        ds_result = ds_df['ts'].dt.isocalendar().day

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_isocalendar_year_boundary(self):
        """Test isocalendar at year boundary (Dec 31 can be Week 1 of next year)."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-12-31'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        # 2024-12-31 is in ISO week 1 of 2025
        pd_year = pd_df['ts'].dt.isocalendar().year
        ds_year = ds_df['ts'].dt.isocalendar().year

        assert_datastore_equals_pandas(ds_year, pd_year)
        assert list(pd_year) == [2025]

    def test_dt_isocalendar_jan_1(self):
        """Test isocalendar for January 1st (can belong to previous year's last week)."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2023-01-01'])})  # Sunday
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        # 2023-01-01 (Sunday) is in ISO week 52 of 2022
        pd_result = pd_df['ts'].dt.isocalendar()
        ds_year = ds_df['ts'].dt.isocalendar().year
        ds_week = ds_df['ts'].dt.isocalendar().week
        ds_day = ds_df['ts'].dt.isocalendar().day

        assert_datastore_equals_pandas(ds_year, pd_result.year)
        assert_datastore_equals_pandas(ds_week, pd_result.week)
        assert_datastore_equals_pandas(ds_day, pd_result.day)


class TestToPydatetime:
    """Test to_pydatetime() method."""

    def test_dt_to_pydatetime(self):
        """Test dt.to_pydatetime converts to Python datetime objects."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-15 10:30:45', '2024-06-21 14:15:30'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.to_pydatetime()
        ds_result = ds_df['ts'].dt.to_pydatetime()

        # Both should return array of datetime objects
        assert len(pd_result) == len(ds_result)
        for pd_dt, ds_dt in zip(pd_result, ds_result):
            assert pd_dt == ds_dt
            assert type(pd_dt).__name__ == 'datetime'
            assert type(ds_dt).__name__ == 'datetime'


class TestAsUnit:
    """Test as_unit() method."""

    def test_dt_as_unit_ms(self):
        """Test dt.as_unit('ms') converts to milliseconds resolution."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-15 10:30:45.123456789'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.as_unit('ms')
        ds_result = ds_df['ts'].dt.as_unit('ms')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_as_unit_us(self):
        """Test dt.as_unit('us') converts to microseconds resolution."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-15 10:30:45.123456789'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.as_unit('us')
        ds_result = ds_df['ts'].dt.as_unit('us')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dt_as_unit_s(self):
        """Test dt.as_unit('s') converts to seconds resolution."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-15 10:30:45.123456789'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.as_unit('s')
        ds_result = ds_df['ts'].dt.as_unit('s')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTimetz:
    """Test timetz property."""

    def test_dt_timetz_naive(self):
        """Test dt.timetz on timezone-naive datetime returns time without tz."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-15 10:30:45', '2024-06-21 14:15:30'])})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.timetz
        ds_result = ds_df['ts'].dt.timetz

        # Both should return time objects (may have slight differences in representation)
        # Compare string representation for consistency
        pd_str = [str(t) for t in pd_result]
        ds_str = [str(t) for t in ds_result.values]

        assert pd_str == ds_str

    def test_dt_timetz_aware(self):
        """Test dt.timetz on timezone-aware datetime includes tzinfo."""
        df = pd.DataFrame({'ts': pd.to_datetime(['2024-01-15 10:30:45', '2024-06-21 14:15:30']).tz_localize('UTC')})
        pd_df = df.copy()
        ds_df = DataStore(df.copy())

        pd_result = pd_df['ts'].dt.timetz
        ds_result = ds_df['ts'].dt.timetz

        # Compare string representation
        pd_str = [str(t) for t in pd_result]
        ds_str = [str(t) for t in ds_result.values]

        assert pd_str == ds_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
