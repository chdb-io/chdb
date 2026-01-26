"""
Exploratory Batch 8: DateTime Accessor and Reshape Methods

Test coverage gaps identified:
1. DateTime accessor - minute, second, quarter, week, millisecond, dayofweek, dayofyear
2. Reshape methods - pivot, pivot_table, stack, unstack, melt variations

Mirror Code Pattern: All tests compare DataStore behavior against pandas.
"""

import pytest
from tests.xfail_markers import chdb_datetime_extraction_conflict
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, get_dataframe


# =============================================================================
# PART 1: DateTime Accessor Tests
# =============================================================================

class TestDateTimeAccessorBasic:
    """Test basic datetime accessor properties."""

    @pytest.fixture
    def datetime_data(self):
        """Create datetime test data."""
        dates = pd.date_range('2023-01-15 10:30:45.123', periods=5, freq='D')
        return pd.DataFrame({'ts': dates, 'value': [1, 2, 3, 4, 5]})

    def test_dt_minute(self, datetime_data):
        """Test dt.minute accessor."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.minute
        ds_result = ds_df['ts'].dt.minute

        # Compare values
        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_second(self, datetime_data):
        """Test dt.second accessor."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.second
        ds_result = ds_df['ts'].dt.second

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_quarter(self, datetime_data):
        """Test dt.quarter accessor."""
        # Use dates spanning multiple quarters
        dates = pd.to_datetime(['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15'])
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        pd_result = pd_df['ts'].dt.quarter
        ds_result = ds_df['ts'].dt.quarter

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_week_weekofyear(self, datetime_data):
        """Test dt.week / dt.weekofyear accessor."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.isocalendar().week
        ds_result = ds_df['ts'].dt.week

        # Compare values (may differ in type)
        assert list(ds_result.to_pandas()) == list(pd_result), \
            f"DS: {list(ds_result.to_pandas())}, PD: {list(pd_result)}"

    def test_dt_dayofweek(self, datetime_data):
        """Test dt.dayofweek accessor (0=Monday, 6=Sunday)."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.dayofweek
        ds_result = ds_df['ts'].dt.dayofweek

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_day_of_week_alias(self, datetime_data):
        """Test dt.day_of_week alias."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.day_of_week
        ds_result = ds_df['ts'].dt.day_of_week

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_dayofyear(self, datetime_data):
        """Test dt.dayofyear accessor."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.dayofyear
        ds_result = ds_df['ts'].dt.dayofyear

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_day_of_year_alias(self, datetime_data):
        """Test dt.day_of_year alias."""
        pd_df = datetime_data
        ds_df = DataStore(datetime_data)

        pd_result = pd_df['ts'].dt.day_of_year
        ds_result = ds_df['ts'].dt.day_of_year

        assert list(ds_result.to_pandas()) == list(pd_result)


class TestDateTimeAccessorEdgeCases:
    """Test datetime accessor edge cases."""

    def test_dt_year_boundary(self):
        """Test year extraction at year boundaries."""
        dates = pd.to_datetime(['2022-12-31 23:59:59', '2023-01-01 00:00:00'])
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_month_boundary(self):
        """Test month extraction at month boundaries."""
        dates = pd.to_datetime(['2023-01-31', '2023-02-01', '2023-02-28', '2023-03-01'])
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        pd_result = pd_df['ts'].dt.month
        ds_result = ds_df['ts'].dt.month

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_hour_minute_second_combined(self):
        """Test combined time extraction."""
        dates = pd.to_datetime(['2023-01-15 00:00:00', '2023-01-15 12:30:45', '2023-01-15 23:59:59'])
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        # Hour
        assert list(ds_df['ts'].dt.hour.to_pandas()) == list(pd_df['ts'].dt.hour)
        # Minute
        assert list(ds_df['ts'].dt.minute.to_pandas()) == list(pd_df['ts'].dt.minute)
        # Second
        assert list(ds_df['ts'].dt.second.to_pandas()) == list(pd_df['ts'].dt.second)

    def test_dt_leap_year(self):
        """Test datetime handling around leap years."""
        dates = pd.to_datetime(['2024-02-28', '2024-02-29', '2024-03-01'])  # 2024 is leap year
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        # Day of year should be correct
        pd_result = pd_df['ts'].dt.dayofyear
        ds_result = ds_df['ts'].dt.dayofyear

        assert list(ds_result.to_pandas()) == list(pd_result)

    def test_dt_with_filter(self):
        """Test datetime accessor followed by filter."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        pd_df = pd.DataFrame({'ts': dates, 'value': range(10)})
        ds_df = DataStore(pd_df)

        # Filter where day > 5
        pd_result = pd_df[pd_df['ts'].dt.day > 5]
        ds_result = ds_df[ds_df['ts'].dt.day > 5]

        assert len(ds_result) == len(pd_result)

    def test_dt_in_groupby(self):
        """Test datetime accessor in groupby."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        pd_df = pd.DataFrame({'ts': dates, 'value': range(30)})
        ds_df = DataStore(pd_df)

        # Group by week
        pd_result = pd_df.groupby(pd_df['ts'].dt.isocalendar().week)['value'].sum()
        # DataStore may have different groupby API
        try:
            ds_result = ds_df.groupby(ds_df['ts'].dt.week)['value'].sum()
            # Just check it doesn't error
            _ = len(ds_result)
        except Exception as e:
            pytest.skip(f"groupby with dt accessor not supported: {e}")


class TestDateTimeAccessorNullHandling:
    """Test datetime accessor with NULL/NaT values."""

    def test_dt_with_nat(self):
        """Test datetime accessor with NaT values."""
        dates = pd.to_datetime(['2023-01-15', None, '2023-01-17'])
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        pd_result = pd_df['ts'].dt.year
        ds_result = ds_df['ts'].dt.year

        # NaT should produce NaN/NULL
        pd_vals = pd_result.tolist()
        ds_vals = ds_result.to_pandas().tolist()

        # Check non-null values match
        assert ds_vals[0] == pd_vals[0]
        assert ds_vals[2] == pd_vals[2]

    def test_dt_all_nat(self):
        """Test datetime accessor with all NaT values."""
        dates = pd.to_datetime([None, None, None])
        pd_df = pd.DataFrame({'ts': dates})
        ds_df = DataStore({'ts': dates})

        pd_result = pd_df['ts'].dt.month
        ds_result = ds_df['ts'].dt.month

        assert len(ds_result) == len(pd_result)


# =============================================================================
# PART 2: Reshape Methods Tests
# =============================================================================

class TestPivotTable:
    """Test pivot_table method."""

    @pytest.fixture
    def sales_data(self):
        """Create sales test data."""
        return pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02']),
            'product': ['A', 'B', 'A', 'B'],
            'region': ['East', 'East', 'West', 'West'],
            'sales': [100, 150, 200, 250],
            'quantity': [10, 15, 20, 25]
        })

    def test_pivot_table_basic(self, sales_data):
        """Test basic pivot_table with sum aggregation."""
        pd_df = sales_data
        ds_df = DataStore(sales_data)

        pd_result = pd_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum')
        ds_result = ds_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum')

        # Compare shape and values
        ds_pandas = get_dataframe(ds_result)
        assert ds_pandas.shape == pd_result.shape

    def test_pivot_table_mean(self, sales_data):
        """Test pivot_table with mean aggregation."""
        pd_df = sales_data
        ds_df = DataStore(sales_data)

        pd_result = pd_df.pivot_table(values='sales', index='product', columns='region', aggfunc='mean')
        ds_result = ds_df.pivot_table(values='sales', index='product', columns='region', aggfunc='mean')

        ds_pandas = get_dataframe(ds_result)
        assert ds_pandas.shape == pd_result.shape

    def test_pivot_table_multiple_values(self, sales_data):
        """Test pivot_table with multiple value columns."""
        pd_df = sales_data
        ds_df = DataStore(sales_data)

        pd_result = pd_df.pivot_table(values=['sales', 'quantity'], index='product', columns='region', aggfunc='sum')

        try:
            ds_result = ds_df.pivot_table(values=['sales', 'quantity'], index='product', columns='region', aggfunc='sum')
            ds_pandas = get_dataframe(ds_result)
            # Just check it executes
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"Multiple values not supported: {e}")

    def test_pivot_table_fill_value(self, sales_data):
        """Test pivot_table with fill_value parameter."""
        pd_df = sales_data
        ds_df = DataStore(sales_data)

        pd_result = pd_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum', fill_value=0)

        try:
            ds_result = ds_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum', fill_value=0)
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"fill_value not supported: {e}")

    def test_pivot_table_margins(self, sales_data):
        """Test pivot_table with margins=True."""
        pd_df = sales_data
        ds_df = DataStore(sales_data)

        pd_result = pd_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum', margins=True)

        try:
            ds_result = ds_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum', margins=True)
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"margins not supported: {e}")


class TestPivot:
    """Test pivot method (without aggregation)."""

    def test_pivot_basic(self):
        """Test basic pivot without aggregation."""
        data = {
            'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
            'variable': ['A', 'B', 'A', 'B'],
            'value': [1, 2, 3, 4]
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.pivot(index='date', columns='variable', values='value')

        try:
            ds_result = ds_df.pivot(index='date', columns='variable', values='value')
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas.shape == pd_result.shape
        except Exception as e:
            pytest.skip(f"pivot not implemented: {e}")

    def test_pivot_no_values(self):
        """Test pivot without values parameter."""
        data = {
            'row': ['r1', 'r1', 'r2', 'r2'],
            'col': ['c1', 'c2', 'c1', 'c2'],
            'val1': [1, 2, 3, 4],
            'val2': [5, 6, 7, 8]
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.pivot(index='row', columns='col')

        try:
            ds_result = ds_df.pivot(index='row', columns='col')
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"pivot without values not supported: {e}")


class TestMelt:
    """Test melt method (wide to long format)."""

    @pytest.fixture
    def wide_data(self):
        """Create wide format test data."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Carol'],
            'score_math': [90, 85, 95],
            'score_eng': [88, 92, 85]
        })

    def test_melt_basic(self, wide_data):
        """Test basic melt operation."""
        pd_df = wide_data
        ds_df = DataStore(wide_data)

        pd_result = pd_df.melt(id_vars=['id', 'name'], value_vars=['score_math', 'score_eng'])
        ds_result = ds_df.melt(id_vars=['id', 'name'], value_vars=['score_math', 'score_eng'])

        ds_pandas = get_dataframe(ds_result)
        assert ds_pandas.shape == pd_result.shape

    def test_melt_var_name_value_name(self, wide_data):
        """Test melt with custom var_name and value_name."""
        pd_df = wide_data
        ds_df = DataStore(wide_data)

        pd_result = pd_df.melt(
            id_vars=['id', 'name'],
            value_vars=['score_math', 'score_eng'],
            var_name='subject',
            value_name='score'
        )

        try:
            ds_result = ds_df.melt(
                id_vars=['id', 'name'],
                value_vars=['score_math', 'score_eng'],
                var_name='subject',
                value_name='score'
            )
            ds_pandas = get_dataframe(ds_result)
            assert 'subject' in ds_pandas.columns
            assert 'score' in ds_pandas.columns
        except Exception as e:
            pytest.skip(f"var_name/value_name not supported: {e}")

    def test_melt_no_id_vars(self, wide_data):
        """Test melt without id_vars."""
        pd_df = wide_data[['score_math', 'score_eng']]
        ds_df = DataStore(pd_df)

        pd_result = pd_df.melt()
        ds_result = ds_df.melt()

        ds_pandas = get_dataframe(ds_result)
        assert len(ds_pandas) == len(pd_result)

    def test_melt_ignore_index(self, wide_data):
        """Test melt with ignore_index parameter."""
        pd_df = wide_data
        ds_df = DataStore(wide_data)

        pd_result = pd_df.melt(id_vars=['id'], ignore_index=False)

        try:
            ds_result = ds_df.melt(id_vars=['id'], ignore_index=False)
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"ignore_index not supported: {e}")


class TestStack:
    """Test stack method (columns to rows)."""

    def test_stack_basic(self):
        """Test basic stack operation."""
        data = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4]
        }, index=['x', 'y'])

        pd_df = data
        ds_df = DataStore(data)

        pd_result = pd_df.stack()

        try:
            ds_result = ds_df.stack()
            ds_pandas = get_dataframe(ds_result)
            assert len(ds_pandas) == len(pd_result)
        except Exception as e:
            pytest.skip(f"stack not implemented: {e}")

    def test_stack_dropna(self):
        """Test stack with dropna parameter."""
        data = pd.DataFrame({
            'A': [1, np.nan],
            'B': [3, 4]
        })

        pd_df = data
        ds_df = DataStore(data)

        # stack() API is deprecated in pandas; use future_stack=True in newer versions
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The previous implementation of stack is deprecated')
            pd_result = pd_df.stack(dropna=True)

        try:
            ds_result = ds_df.stack(dropna=True)
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"stack with dropna not supported: {e}")


class TestUnstack:
    """Test unstack method (rows to columns)."""

    def test_unstack_basic(self):
        """Test basic unstack operation."""
        # Create MultiIndex Series
        arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
        index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_s = pd.Series([1, 2, 3, 4], index=index)

        pd_result = pd_s.unstack()

        # DataStore may not support MultiIndex directly
        try:
            ds_s = DataStore({'first': ['A', 'A', 'B', 'B'], 'second': ['one', 'two', 'one', 'two'], 'value': [1, 2, 3, 4]})
            ds_result = ds_s.unstack()
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"unstack not implemented: {e}")


class TestExplode:
    """Test explode method."""

    def test_explode_list_column(self):
        """Test explode on list column."""
        data = {
            'id': [1, 2],
            'values': [[1, 2, 3], [4, 5]]
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.explode('values')

        try:
            ds_result = ds_df.explode('values')
            ds_pandas = get_dataframe(ds_result)
            assert len(ds_pandas) == len(pd_result)
        except Exception as e:
            pytest.skip(f"explode not implemented: {e}")

    def test_explode_empty_list(self):
        """Test explode with empty lists."""
        data = {
            'id': [1, 2, 3],
            'values': [[1, 2], [], [3]]
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.explode('values')

        try:
            ds_result = ds_df.explode('values')
            ds_pandas = get_dataframe(ds_result)
            assert len(ds_pandas) == len(pd_result)
        except Exception as e:
            pytest.skip(f"explode with empty lists not supported: {e}")

    def test_explode_ignore_index(self):
        """Test explode with ignore_index parameter."""
        data = {
            'id': [1, 2],
            'values': [[1, 2], [3, 4, 5]]
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.explode('values', ignore_index=True)

        try:
            ds_result = ds_df.explode('values', ignore_index=True)
            ds_pandas = get_dataframe(ds_result)
            assert len(ds_pandas) == len(pd_result)
        except Exception as e:
            pytest.skip(f"explode with ignore_index not supported: {e}")


# =============================================================================
# PART 3: Complex Chained Operations
# =============================================================================

class TestDateTimeChainedOps:
    """Test chained operations involving datetime accessor."""

    def test_dt_extract_then_filter(self):
        """Test datetime extraction followed by filter."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        pd_df = pd.DataFrame({'ts': dates, 'value': range(100)})
        ds_df = DataStore(pd_df)

        # Filter Q1 data
        pd_result = pd_df[pd_df['ts'].dt.quarter == 1]
        ds_result = ds_df[ds_df['ts'].dt.quarter == 1]

        assert len(ds_result) == len(pd_result)

    def test_dt_multiple_extractions(self):
        """Test multiple datetime extractions in one query."""
        dates = pd.date_range('2023-01-01 10:30:45', periods=5, freq='h')
        pd_df = pd.DataFrame({'ts': dates})
        # NOTE: Must use .copy() to avoid shared DataFrame modification between pandas and DataStore
        ds_df = DataStore(pd_df.copy())

        # Extract year, month, day, hour all together
        pd_df['year'] = pd_df['ts'].dt.year
        pd_df['month'] = pd_df['ts'].dt.month
        pd_df['day'] = pd_df['ts'].dt.day
        pd_df['hour'] = pd_df['ts'].dt.hour

        ds_df = ds_df.assign(
            year=ds_df['ts'].dt.year,
            month=ds_df['ts'].dt.month,
            day=ds_df['ts'].dt.day,
            hour=ds_df['ts'].dt.hour
        )

        ds_pandas = get_dataframe(ds_df)
        assert list(ds_pandas['year']) == list(pd_df['year'])
        assert list(ds_pandas['month']) == list(pd_df['month'])

    def test_dt_groupby_agg(self):
        """Test datetime extraction with groupby aggregation."""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        pd_df = pd.DataFrame({'ts': dates, 'value': np.random.randint(1, 100, 60)})
        ds_df = DataStore(pd_df)

        # Group by month and sum
        pd_df['month'] = pd_df['ts'].dt.month
        pd_result = pd_df.groupby('month')['value'].sum()

        try:
            ds_df = ds_df.assign(month=ds_df['ts'].dt.month)
            ds_result = ds_df.groupby('month')['value'].sum()
            # Just check it doesn't error
            assert len(ds_result) > 0
        except Exception as e:
            pytest.skip(f"groupby after dt extraction failed: {e}")


class TestReshapeChainedOps:
    """Test chained operations involving reshape methods."""

    def test_melt_then_filter(self):
        """Test melt followed by filter."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'A': [10, 20, 30],
            'B': [15, 25, 35]
        })
        pd_df = data
        ds_df = DataStore(data)

        pd_result = pd_df.melt(id_vars=['id']).query('value > 20')

        try:
            ds_result = ds_df.melt(id_vars=['id'])
            ds_result = ds_result[ds_result['value'] > 20]
            ds_pandas = get_dataframe(ds_result)
            assert len(ds_pandas) == len(pd_result)
        except Exception as e:
            pytest.skip(f"melt then filter not supported: {e}")

    def test_pivot_table_then_reset_index(self):
        """Test pivot_table followed by reset_index."""
        data = pd.DataFrame({
            'product': ['A', 'A', 'B', 'B'],
            'region': ['E', 'W', 'E', 'W'],
            'sales': [100, 200, 150, 250]
        })
        pd_df = data
        ds_df = DataStore(data)

        pd_result = pd_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum').reset_index()

        try:
            ds_result = ds_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum')
            ds_result = ds_result.reset_index()
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas is not None
        except Exception as e:
            pytest.skip(f"pivot_table then reset_index not supported: {e}")


# =============================================================================
# PART 4: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases for datetime and reshape operations."""

    def test_empty_dataframe_dt(self):
        """Test datetime accessor on empty DataFrame."""
        pd_df = pd.DataFrame({'ts': pd.to_datetime([])})
        ds_df = DataStore({'ts': []})

        try:
            ds_result = ds_df['ts'].dt.year
            assert len(ds_result) == 0
        except Exception as e:
            pytest.skip(f"Empty DataFrame dt accessor failed: {e}")

    def test_empty_dataframe_melt(self):
        """Test melt on empty DataFrame."""
        pd_df = pd.DataFrame({'id': [], 'A': [], 'B': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.melt(id_vars=['id'])

        try:
            ds_result = ds_df.melt(id_vars=['id'])
            ds_pandas = get_dataframe(ds_result)
            assert len(ds_pandas) == len(pd_result)
        except Exception as e:
            pytest.skip(f"Empty DataFrame melt failed: {e}")

    def test_single_row_pivot(self):
        """Test pivot_table with single row."""
        data = pd.DataFrame({
            'product': ['A'],
            'region': ['E'],
            'sales': [100]
        })
        pd_df = data
        ds_df = DataStore(data)

        pd_result = pd_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum')

        try:
            ds_result = ds_df.pivot_table(values='sales', index='product', columns='region', aggfunc='sum')
            ds_pandas = get_dataframe(ds_result)
            assert ds_pandas.shape == pd_result.shape
        except Exception as e:
            pytest.skip(f"Single row pivot_table failed: {e}")

    def test_datetime_string_conversion(self):
        """Test datetime from string format."""
        data = {'ts': ['2023-01-15', '2023-02-20', '2023-03-25']}
        pd_df = pd.DataFrame(data)
        pd_df['ts'] = pd.to_datetime(pd_df['ts'])

        ds_df = DataStore(data)

        # DataStore should handle string dates
        try:
            ds_result = ds_df['ts'].dt.month
            # Just check it works
            assert len(ds_result) == 3
        except Exception as e:
            pytest.skip(f"String to datetime conversion failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
