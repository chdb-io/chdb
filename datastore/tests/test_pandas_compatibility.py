"""
Pandas Compatibility Tests
==========================

Test suite to verify datastore's compatibility with pandas API.
Each test compares the result of an operation in both pandas and datastore.

Uses the `==` operator for comparison, which is implemented in DataStore,
ColumnExpr, and LazyAggregate to handle smart comparison with pandas objects.
"""

import pytest
import pandas as pd
import numpy as np
import datastore as ds
from tests.test_utils import (
    assert_datastore_equals_pandas_chdb_compat,
    assert_frame_equal,
    assert_series_equal,
    get_series,
)


# =============================================================================
# Test Data Fixture
# =============================================================================


@pytest.fixture
def test_data():
    """Comprehensive test dataset."""
    return {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None, 'Grace'],
        'age': [25, 30, 35, None, 28, 32, 45, 27],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Sydney', 'Toronto', None],
        'salary': [50000, 60000, None, 75000, 55000, 80000, 70000, 52000],
        'department': ['HR', 'IT', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
        'hire_date': [
            '2020-01-15',
            '2019-06-20',
            '2018-03-10',
            '2021-09-05',
            '2020-11-12',
            '2019-02-28',
            '2017-08-15',
            '2021-04-30',
        ],
        'performance_score': [8.5, 7.2, 9.1, 6.8, 8.0, 9.5, 7.5, 8.3],
    }


@pytest.fixture
def pd_df(test_data):
    """Pandas DataFrame with test data."""
    return pd.DataFrame(test_data)


@pytest.fixture
def ds_df(test_data):
    """DataStore DataFrame with test data."""
    # Use DataStore.from_df() to create a DataStore object explicitly
    # ds.DataFrame is now pd.DataFrame for monkey-patching compatibility
    return ds.DataStore.from_df(pd.DataFrame(test_data))


# =============================================================================
# DataFrame Creation Tests
# =============================================================================


class TestDataFrameCreation:
    """Test DataFrame creation operations."""

    def test_create_from_dict(self):
        """Create DataFrame from dict."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd_result = pd.DataFrame(data)
        ds_result = ds.DataStore.from_df(pd.DataFrame(data))
        assert ds_result == pd_result

    def test_create_with_index(self):
        """Create DataFrame with index."""
        data = {'a': [1, 2, 3]}
        index = ['x', 'y', 'z']
        pd_result = pd.DataFrame(data, index=index)
        ds_result = ds.DataStore.from_df(pd.DataFrame(data, index=index))
        assert ds_result == pd_result


# =============================================================================
# Data Selection Tests
# =============================================================================


class TestDataSelection:
    """Test data selection operations."""

    def test_select_single_column(self, pd_df, ds_df):
        """Select single column."""
        pd_result = pd_df['name']
        ds_result = ds_df['name']
        assert ds_result == pd_result

    def test_select_multiple_columns(self, pd_df, ds_df):
        """Select multiple columns."""
        pd_result = pd_df[['name', 'age']]
        ds_result = ds_df[['name', 'age']]
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_select_rows_by_slice(self, pd_df, ds_df):
        """Select rows by slice."""
        pd_result = pd_df[:3]
        ds_result = ds_df[:3]
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_select_with_boolean_indexing(self, pd_df, ds_df):
        """Select with boolean indexing."""
        pd_result = pd_df[pd_df['age'] > 30]
        ds_result = ds_df[ds_df['age'] > 30]
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_select_with_loc(self, pd_df, ds_df):
        """Select with loc."""
        pd_result = pd_df.loc[0:2, ['name', 'age']]
        ds_result = ds_df.loc[0:2, ['name', 'age']]
        # loc returns pandas DataFrame directly
        assert_frame_equal(ds_result, pd_result)

    def test_select_with_iloc(self, pd_df, ds_df):
        """Select with iloc."""
        pd_result = pd_df.iloc[0:3, 0:2]
        ds_result = ds_df.iloc[0:3, 0:2]
        # iloc returns pandas DataFrame directly
        assert_frame_equal(ds_result, pd_result)


# =============================================================================
# Data Cleaning Tests
# =============================================================================


class TestDataCleaning:
    """Test data cleaning operations."""

    def test_dropna(self, pd_df, ds_df):
        """Drop NA values."""
        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()
        assert ds_result == pd_result

    def test_fillna(self, pd_df, ds_df):
        """Fill NA values."""
        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)
        assert ds_result == pd_result

    def test_drop_duplicates(self, pd_df, ds_df):
        """Drop duplicates."""
        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()
        assert ds_result == pd_result

    def test_replace_values(self, pd_df, ds_df):
        """Replace values."""
        pd_result = pd_df.replace('HR', 'Human Resources')
        ds_result = ds_df.replace('HR', 'Human Resources')
        assert ds_result == pd_result

    def test_drop_column(self, pd_df, ds_df):
        """Drop column."""
        pd_result = pd_df.drop('salary', axis=1)
        ds_result = ds_df.drop('salary', axis=1)
        assert ds_result == pd_result


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test statistical operations."""

    def test_mean(self, pd_df, ds_df):
        """Compute mean."""
        pd_result = pd_df['age'].mean()
        ds_result = ds_df['age'].mean()
        assert ds_result == pd_result

    def test_sum(self, pd_df, ds_df):
        """Compute sum."""
        pd_result = pd_df['salary'].sum()
        ds_result = ds_df['salary'].sum()
        assert ds_result == pd_result

    def test_median(self, pd_df, ds_df):
        """Compute median."""
        pd_result = pd_df['age'].median()
        ds_result = ds_df['age'].median()
        assert ds_result == pd_result

    def test_std(self, pd_df, ds_df):
        """Compute std."""
        pd_result = pd_df['age'].std()
        ds_result = ds_df['age'].std()
        assert ds_result == pd_result

    def test_describe(self, pd_df, ds_df):
        """Describe DataFrame."""
        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        assert ds_result == pd_result

    def test_value_counts(self, pd_df, ds_df):
        """Value counts."""
        pd_result = pd_df['department'].value_counts()
        ds_result = ds_df['department'].value_counts()
        # value_counts returns LazySeries, sort_index also returns LazySeries
        # Compare values using numpy (natural execution trigger)
        np.testing.assert_array_equal(ds_result.sort_index(), pd_result.sort_index())

    def test_correlation(self, pd_df, ds_df):
        """Correlation."""
        pd_result = pd_df[['age', 'salary']].corr()
        ds_result = ds_df[['age', 'salary']].corr()
        # corr returns pandas DataFrame
        assert_frame_equal(ds_result, pd_result)


# =============================================================================
# Data Transformation Tests
# =============================================================================


class TestDataTransformation:
    """Test data transformation operations."""

    def test_apply_function(self, pd_df, ds_df):
        """Apply function to column."""
        pd_result = pd_df['age'].apply(lambda x: x * 2 if pd.notna(x) else x)
        ds_result = ds_df['age'].apply(lambda x: x * 2 if pd.notna(x) else x)
        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_map_values(self, pd_df, ds_df):
        """Map values."""
        mapping = {'HR': 1, 'IT': 2, 'Finance': 3}
        pd_result = pd_df['department'].map(mapping)
        ds_result = ds_df['department'].map(mapping)
        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_rename_columns(self, pd_df, ds_df):
        """Rename columns."""
        pd_result = pd_df.rename(columns={'name': 'employee_name'})
        ds_result = ds_df.rename(columns={'name': 'employee_name'})
        assert ds_result == pd_result

    def test_astype(self, pd_df, ds_df):
        """Convert type (astype)."""
        pd_result = pd_df['age'].astype(str)
        ds_result = ds_df['age'].astype(str)
        # LazySeries implements __array__, numpy can accept it directly
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_add_new_column(self, pd_df, ds_df):
        """Add new column."""
        pd_df = pd_df.copy()
        pd_df['bonus'] = pd_df['salary'] * 0.1

        ds_df = ds_df.copy()
        ds_df['bonus'] = ds_df['salary'] * 0.1

        assert ds_df == pd_df


# =============================================================================
# Sorting Tests
# =============================================================================


class TestSorting:
    """Test sorting operations."""

    def test_sort_by_single_column(self, pd_df, ds_df):
        """Sort by single column."""
        pd_result = pd_df.sort_values('age')
        ds_result = ds_df.sort_values('age')
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_sort_by_multiple_columns(self, pd_df, ds_df):
        """Sort by multiple columns."""
        pd_result = pd_df.sort_values(['department', 'age'])
        ds_result = ds_df.sort_values(['department', 'age'])
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_sort_descending(self, pd_df, ds_df):
        """Sort descending."""
        pd_result = pd_df.sort_values('salary', ascending=False)
        ds_result = ds_df.sort_values('salary', ascending=False)
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_sort_index(self, pd_df, ds_df):
        """Sort index."""
        pd_result = pd_df.sort_index()
        ds_result = ds_df.sort_index()
        assert ds_result == pd_result


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregation:
    """Test aggregation operations.

    Uses natural execution triggers (np.testing.assert_array_equal)
    instead of explicit .to_df() calls.
    """

    def test_groupby_single_aggregation(self, pd_df, ds_df):
        """GroupBy with single aggregation."""
        pd_result = pd_df.groupby('department')['salary'].mean()
        ds_result = ds_df.groupby('department')['salary'].mean()
        # Natural trigger via comparison
        assert ds_result == pd_result

    def test_groupby_multiple_aggregations(self, pd_df, ds_df):
        """GroupBy with multiple aggregations - returns lazy DataStore."""
        pd_result = pd_df.groupby('department').agg({'salary': 'mean', 'age': 'max'})
        ds_result = ds_df.groupby('department').agg({'salary': 'mean', 'age': 'max'})
        # GroupBy order is not guaranteed, use check_row_order=False
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result, check_row_order=False)

    def test_groupby_sum(self, pd_df, ds_df):
        """GroupBy with sum."""
        pd_result = pd_df.groupby('department')['salary'].sum()
        ds_result = ds_df.groupby('department')['salary'].sum()
        # Natural trigger via comparison
        assert ds_result == pd_result

    def test_groupby_count(self, pd_df, ds_df):
        """GroupBy with size - returns LazySeries (pd.Series compatible)."""
        pd_result = pd_df.groupby('department').size()
        ds_result = ds_df.groupby('department').size()
        # Natural trigger: np.testing with __array__ protocol
        # Both return Series with group keys as index
        np.testing.assert_array_equal(ds_result, pd_result)


# =============================================================================
# String Operations Tests
# =============================================================================


class TestStringOperations:
    """Test string operations."""

    def test_str_upper(self, pd_df, ds_df):
        """String upper - with NULL handling workaround."""
        pd_result = pd_df['name'].str.upper()
        ds_result = ds_df['name'].str.upper()
        assert ds_result == pd_result

    def test_str_lower(self, pd_df, ds_df):
        """String lower - with NULL handling workaround."""
        pd_result = pd_df['city'].str.lower()
        ds_result = ds_df['city'].str.lower()
        assert ds_result == pd_result

    def test_str_contains(self, pd_df, ds_df):
        """String contains."""
        pd_result = pd_df['name'].str.contains('a', na=False)
        ds_result = ds_df['name'].str.contains('a', na=False)
        assert ds_result == pd_result

    def test_str_len(self, pd_df, ds_df):
        """String length - with NULL handling workaround."""
        pd_result = pd_df['name'].str.len()
        ds_result = ds_df['name'].str.len()
        assert ds_result == pd_result

    def test_str_replace(self, pd_df, ds_df):
        """String replace - with NULL handling workaround."""
        pd_result = pd_df['city'].str.replace('York', 'Amsterdam')
        ds_result = ds_df['city'].str.replace('York', 'Amsterdam')
        assert ds_result == pd_result


# =============================================================================
# Merging Tests
# =============================================================================


class TestMerging:
    """Test merge/join operations."""

    def test_concat(self):
        """Concat DataFrames."""
        df1_pd = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2_pd = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        pd_result = pd.concat([df1_pd, df2_pd])

        df1_ds = ds.DataStore.from_df(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
        df2_ds = ds.DataStore.from_df(pd.DataFrame({'a': [5, 6], 'b': [7, 8]}))
        ds_result = ds.concat([df1_ds, df2_ds])

        assert ds_result == pd_result

    def test_merge(self):
        """Merge DataFrames."""
        df1_pd = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        df2_pd = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})
        pd_result = pd.merge(df1_pd, df2_pd, on='key', how='inner')

        df1_ds = ds.DataStore.from_df(pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]}))
        df2_ds = ds.DataStore.from_df(pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]}))
        ds_result = ds.merge(df1_ds, df2_ds, on='key', how='inner')

        assert ds_result == pd_result


# =============================================================================
# DateTime Operations Tests
# =============================================================================


class TestDateTimeOperations:
    """Test datetime operations."""

    def test_to_datetime(self, pd_df, ds_df):
        """Convert to datetime."""
        pd_result = pd.to_datetime(pd_df['hire_date'])
        ds_result = ds.to_datetime(ds_df['hire_date'])
        # to_datetime may return DatetimeIndex or Series - compare values
        np.testing.assert_array_equal(
            np.array(ds_result, dtype='datetime64[ns]'), np.array(pd_result, dtype='datetime64[ns]')
        )

    def test_dt_year(self, pd_df, ds_df):
        """Extract year from date - auto-converts string to datetime."""
        # Pandas needs explicit conversion, datastore does it automatically
        pd_result = pd.to_datetime(pd_df['hire_date']).dt.year
        ds_result = ds_df['hire_date'].dt.year
        # Use np.testing for lazy results
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_dt_month(self, pd_df, ds_df):
        """Extract month from date - auto-converts string to datetime."""
        pd_result = pd.to_datetime(pd_df['hire_date']).dt.month
        ds_result = ds_df['hire_date'].dt.month
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_dt_strftime(self, pd_df, ds_df):
        """Date formatting - auto-converts string to datetime."""
        pd_result = pd.to_datetime(pd_df['hire_date']).dt.strftime('%Y-%m')
        ds_result = ds_df['hire_date'].dt.strftime('%Y-%m')
        np.testing.assert_array_equal(ds_result, pd_result)


class TestDateTimeEngineSwitch:
    """Test datetime accessor engine switching based on config."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with datetime column."""
        return pd.DataFrame({'date': pd.to_datetime(['2023-01-15', '2023-06-20', '2023-12-25']), 'value': [10, 20, 30]})

    def test_dt_year_chdb_engine(self, sample_df):
        """Test .dt.year uses chDB when engine is set to chdb."""
        from datastore.config import get_execution_engine, set_execution_engine
        from datastore.column_expr import ColumnExpr

        # Save original setting
        original_engine = get_execution_engine()

        try:
            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))

            # Get the lazy result
            result = ds_df['date'].dt.year

            # Should return ColumnExpr (wrapping chDB Function)
            assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result)}"

            # Verify the expression contains toYear
            sql = result._expr.to_sql()
            assert 'toYear' in sql, f"Expected toYear in SQL, got: {sql}"

            # Verify values are correct
            expected = sample_df['date'].dt.year
            np.testing.assert_array_equal(result, expected)
        finally:
            # Restore original setting
            set_execution_engine(original_engine)

    def test_dt_year_pandas_engine(self, sample_df):
        """Test .dt.year uses pandas when engine is set to pandas."""
        from datastore.config import get_execution_engine, set_execution_engine
        from datastore.column_expr import ColumnExpr

        # Save original setting
        original_engine = get_execution_engine()

        try:
            set_execution_engine('pandas')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))

            # Get the lazy result
            result = ds_df['date'].dt.year

            # Should return ColumnExpr (unified lazy architecture)
            assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result)}"

            # Verify underlying expression is DateTimePropertyExpr
            from datastore.expressions import DateTimePropertyExpr

            assert isinstance(
                result._expr, DateTimePropertyExpr
            ), f"Expected DateTimePropertyExpr, got {type(result._expr)}"
            assert result._expr.property_name == 'year'

            # Verify values are correct (execution happens via pandas)
            expected = sample_df['date'].dt.year
            np.testing.assert_array_equal(result, expected)
        finally:
            # Restore original setting
            set_execution_engine(original_engine)

    def test_dt_month_engine_switch(self, sample_df):
        """Test .dt.month works correctly with both engines."""
        from datastore.config import get_execution_engine, set_execution_engine

        original_engine = get_execution_engine()
        expected = sample_df['date'].dt.month

        try:
            # Test chDB engine
            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            result_chdb = ds_df['date'].dt.month
            np.testing.assert_array_equal(result_chdb, expected)

            # Test pandas engine
            set_execution_engine('pandas')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            result_pandas = ds_df['date'].dt.month
            np.testing.assert_array_equal(result_pandas, expected)

            # Both should produce same results
            np.testing.assert_array_equal(result_chdb, result_pandas)
        finally:
            set_execution_engine(original_engine)

    def test_dt_dayofweek_engine_switch(self, sample_df):
        """Test .dt.dayofweek works correctly with both engines."""
        from datastore.config import get_execution_engine, set_execution_engine

        original_engine = get_execution_engine()
        expected = sample_df['date'].dt.dayofweek

        try:
            # Test chDB engine
            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            result_chdb = ds_df['date'].dt.dayofweek
            np.testing.assert_array_equal(result_chdb, expected)

            # Test pandas engine
            set_execution_engine('pandas')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            result_pandas = ds_df['date'].dt.dayofweek
            np.testing.assert_array_equal(result_pandas, expected)
        finally:
            set_execution_engine(original_engine)

    def test_dt_strftime_engine_switch(self, sample_df):
        """Test .dt.strftime works correctly with both engines."""
        from datastore.config import get_execution_engine, set_execution_engine

        original_engine = get_execution_engine()
        expected = sample_df['date'].dt.strftime('%Y-%m')

        try:
            # Test chDB engine
            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            result_chdb = ds_df['date'].dt.strftime('%Y-%m')
            np.testing.assert_array_equal(result_chdb, expected)

            # Test pandas engine
            set_execution_engine('pandas')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            result_pandas = ds_df['date'].dt.strftime('%Y-%m')
            np.testing.assert_array_equal(result_pandas, expected)
        finally:
            set_execution_engine(original_engine)

    def test_dt_explain_shows_engine(self, sample_df):
        """Test explain() shows the correct execution plan for dt operations."""
        from datastore.config import get_execution_engine, set_execution_engine

        original_engine = get_execution_engine()

        try:
            # Test chDB engine
            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            ds_df['year'] = ds_df['date'].dt.year

            # Get explain output
            explain_output = ds_df.explain()

            # Should show toYear in the plan (the lazy expression)
            assert 'toYear' in explain_output, f"Expected 'toYear' in explain output:\n{explain_output}"

            # Test pandas engine - now also shows toYear (unified DateTimePropertyExpr)
            # but execution engine is determined at runtime by function_config
            set_execution_engine('pandas')
            ds_df2 = ds.DataStore.from_df(pd.DataFrame(sample_df))
            ds_df2['year'] = ds_df2['date'].dt.year

            explain_output2 = ds_df2.explain()

            # In new architecture, explain shows the expression (toYear) but execution
            # engine selection happens at runtime. Check that the plan is present.
            assert (
                'toYear' in explain_output2 or 'year' in explain_output2
            ), f"Expected datetime property in explain output:\n{explain_output2}"
        finally:
            set_execution_engine(original_engine)

    def test_dt_chdb_explain_regex(self, sample_df):
        """Test chDB engine explain output matches expected patterns using regex."""
        import re
        from datastore.config import get_execution_engine, set_execution_engine

        original_engine = get_execution_engine()

        try:
            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            ds_df['year'] = ds_df['date'].dt.year
            ds_df['month'] = ds_df['date'].dt.month

            explain_output = ds_df.explain()

            # Verify chDB function patterns in explain output
            # Pattern: [chDB] Assign column 'year' = toYear(...)
            year_pattern = r"\[chDB\].*Assign column 'year'.*toYear"
            assert re.search(
                year_pattern, explain_output
            ), f"Expected pattern '{year_pattern}' not found in explain:\n{explain_output}"

            month_pattern = r"\[chDB\].*Assign column 'month'.*toMonth"
            assert re.search(
                month_pattern, explain_output
            ), f"Expected pattern '{month_pattern}' not found in explain:\n{explain_output}"

            # In new architecture, type conversion happens at execution time in ExpressionEvaluator
            # The explain output shows the simplified expression (toYear, toMonth)

        finally:
            set_execution_engine(original_engine)

    def test_dt_pandas_explain_regex(self, sample_df):
        """Test pandas engine explain output matches expected patterns using regex."""
        import re
        from datastore.config import get_execution_engine, set_execution_engine

        original_engine = get_execution_engine()

        try:
            set_execution_engine('pandas')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            ds_df['year'] = ds_df['date'].dt.year
            ds_df['month'] = ds_df['date'].dt.month

            explain_output = ds_df.explain()

            # In new unified architecture, explain shows toYear/toMonth but engine is [Pandas]
            # Engine selection happens at execution time based on function_config
            year_pattern = r"\[Pandas\].*Assign column 'year'.*toYear"
            assert re.search(
                year_pattern, explain_output
            ), f"Expected pattern '{year_pattern}' not found in explain:\n{explain_output}"

            # In new architecture, month also shows toMonth (unified expression)
            month_pattern = r"\[Pandas\].*Assign column 'month'.*toMonth"
            assert re.search(
                month_pattern, explain_output
            ), f"Expected pattern '{month_pattern}' not found in explain:\n{explain_output}"

            # In new unified architecture, expressions like toYear/toMonth are shown
            # but the [Pandas] tag indicates execution will use pandas .dt accessor

        finally:
            set_execution_engine(original_engine)

    def test_dt_chdb_debug_log_regex(self, sample_df, caplog):
        """Test chDB engine debug logs match expected SQL patterns using regex."""
        import re
        import logging
        from datastore.config import get_execution_engine, set_execution_engine, set_log_level

        original_engine = get_execution_engine()

        try:
            # Enable debug logging
            set_log_level(logging.DEBUG)

            set_execution_engine('chdb')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            ds_df['year'] = ds_df['date'].dt.year

            # Execute to trigger debug logs
            with caplog.at_level(logging.DEBUG, logger='datastore'):
                _ = ds_df.to_df()

            log_text = caplog.text

            # In new unified architecture, dt.year is compiled to toInt32(toYear(...)) and pushed to SQL
            # as a column assignment: SELECT *, toInt32(toYear("date")) AS "year" FROM ...
            # The toInt32 wrapper ensures the result matches pandas dtype (int32)
            sql_pattern = r'toInt32\(toYear\("date"\)\)\s+AS\s+"year"'
            assert re.search(
                sql_pattern, log_text
            ), f"Expected SQL pattern '{sql_pattern}' not found in debug log:\n{log_text}"

            # Verify chDB execution (DataFrame execution or Result marker)
            chdb_marker = r"\[chDB\]"
            assert re.search(
                chdb_marker, log_text
            ), f"Expected chDB marker '{chdb_marker}' not found in debug log:\n{log_text}"

        finally:
            set_execution_engine(original_engine)
            set_log_level(logging.WARNING)

    def test_dt_pandas_debug_log_regex(self, sample_df, caplog):
        """Test execution logs contain expected patterns for dt.year operation."""
        import re
        import logging
        from datastore.config import get_execution_engine, set_execution_engine, set_log_level

        original_engine = get_execution_engine()

        try:
            # Enable debug logging
            set_log_level(logging.DEBUG)

            # Note: In new unified architecture, even with execution_engine='pandas',
            # SQL-compatible operations like toYear() can still be pushed to SQL
            # for efficiency. The execution_engine setting primarily affects
            # function-level execution choice, not SQL pushdown optimization.
            set_execution_engine('pandas')
            ds_df = ds.DataStore.from_df(pd.DataFrame(sample_df))
            ds_df['year'] = ds_df['date'].dt.year

            # Clear previous logs
            caplog.clear()

            # Execute to trigger debug logs
            with caplog.at_level(logging.DEBUG, logger='datastore'):
                _ = ds_df.to_df()

            log_text = caplog.text

            # In unified architecture, dt.year is compiled to toYear and pushed to SQL
            # Verify the column assignment is in the execution plan
            assign_pattern = r"Assign column 'year'.*toYear"
            assert re.search(
                assign_pattern, log_text
            ), f"Expected assignment pattern '{assign_pattern}' not found in debug log:\n{log_text}"

            # Verify execution completed successfully
            assert "Execution complete" in log_text

        finally:
            set_execution_engine(original_engine)
            set_log_level(logging.WARNING)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_dataframe(self):
        """Test empty DataFrame operations."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = ds.DataStore.from_df(pd.DataFrame({'a': [], 'b': []}))
        assert ds_df == pd_df

    def test_single_row(self):
        """Test single row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = ds.DataStore.from_df(pd.DataFrame({'a': [1], 'b': [2]}))
        assert ds_df == pd_df

    def test_single_column(self):
        """Test single column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = ds.DataStore.from_df(pd.DataFrame({'a': [1, 2, 3]}))
        assert ds_df == pd_df

    def test_all_null_column(self):
        """Test column with all nulls."""
        pd_df = pd.DataFrame({'a': [None, None, None]})
        ds_df = ds.DataStore.from_df(pd.DataFrame({'a': [None, None, None]}))
        # Compare shapes at least
        assert ds_df.shape == pd_df.shape

    def test_mixed_types(self):
        """Test DataFrame with mixed types."""
        data = {
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
        }
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd.DataFrame(data))
        assert ds_df == pd_df

    def test_chained_operations(self, pd_df, ds_df):
        """Test chained operations."""
        pd_result = pd_df[pd_df['age'] > 25][['name', 'age']].sort_values('age')
        ds_result = ds_df[ds_df['age'] > 25][['name', 'age']].sort_values('age')
        assert_datastore_equals_pandas_chdb_compat(ds_result, pd_result)

    def test_pop_removes_column(self, pd_df, ds_df):
        """Test that pop() removes column and returns it.

        This is a regression test for data leakage bug where pop() didn't
        remove the column from DataStore, causing target leakage in ML workflows.
        """
        # Test with pandas
        pd_copy = pd_df.copy()
        pd_popped = pd_copy.pop('salary')

        # Test with datastore
        ds_copy = ds_df.copy()
        ds_popped = ds_copy.pop('salary')

        # Verify column is removed
        assert 'salary' not in pd_copy.columns
        assert 'salary' not in ds_copy.columns

        # Verify popped values match
        assert_series_equal(get_series(ds_popped).reset_index(drop=True), pd_popped.reset_index(drop=True))

    def test_pop_ml_workflow_pattern(self):
        """Test the typical ML workflow: X = df.copy(); y = X.pop(target)."""
        data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [10, 20, 30, 40, 50], 'target': [0, 1, 0, 1, 0]}
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataStore.from_df(pd.DataFrame(data))

        # Pandas workflow
        X_pd = pd_df.copy()
        y_pd = X_pd.pop('target')

        # DataStore workflow
        X_ds = ds_df.copy()
        y_ds = X_ds.pop('target')

        # Verify features don't contain target
        assert 'target' not in X_pd.columns
        assert 'target' not in X_ds.columns

        # Verify feature count matches
        assert len(X_pd.columns) == len(X_ds.columns) == 2

        # Verify target values match
        assert list(y_ds) == list(y_pd)

    def test_delitem_removes_column(self, pd_df, ds_df):
        """Test that del ds[column] removes the column in-place."""
        # Test with pandas
        pd_copy = pd_df.copy()
        del pd_copy['salary']

        # Test with datastore
        ds_copy = ds_df.copy()
        del ds_copy['salary']

        # Verify column is removed
        assert 'salary' not in pd_copy.columns
        assert 'salary' not in ds_copy.columns

        # Verify other columns remain
        assert len(pd_copy.columns) == len(ds_copy.columns)

    def test_update_modifies_inplace(self):
        """Test that update() modifies the DataStore in-place."""
        # Create DataFrames with some NA values
        data1 = {'a': [1, 2, 3], 'b': [None, None, None]}
        data2 = {'b': [10, 20, 30]}

        pd_df = pd.DataFrame(data1)
        ds_df = ds.DataStore.from_df(pd.DataFrame(data1))

        update_df = pd.DataFrame(data2)

        # Apply update
        pd_df.update(update_df)
        ds_df.update(update_df)

        # Verify values are updated - use get_series to preserve Series name
        assert_series_equal(
            get_series(ds_df['b']).reset_index(drop=True),
            pd_df['b'].reset_index(drop=True),
        )
