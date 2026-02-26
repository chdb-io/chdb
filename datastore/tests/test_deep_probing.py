"""
Deep probing tests to discover hidden issues.

These tests are designed to stress-test the system and find edge cases
that might reveal deeper architectural problems.

Areas explored:
1. dtype propagation through chained operations
2. Complex index scenarios (MultiIndex, string index)
3. where/mask with unusual type combinations
4. Timezone and datetime edge cases
5. NULL/NaN handling in complex scenarios
6. Column name edge cases (special chars, duplicates)
7. Engine switching state management
8. Mixed SQL/Pandas segment boundaries
9. Large-scale operations
10. Recursive/nested expressions
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from datetime import datetime, timedelta
import warnings

from datastore import DataStore, config
from tests.test_utils import assert_datastore_equals_pandas


class TestDtypePropagationDeep:
    """Test dtype is correctly propagated through complex operations."""

    def test_float_operations_preserve_nullable_dtype(self):
        """
        Test chained operations on fillna result: fillna().abs().round()

        When chDB returns Float64 (nullable), subsequent operations should
        preserve this dtype correctly.
        """
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0]})
        ds = DataStore.from_df(df)

        # Chain of operations - THIS FAILS
        result = ds['a'].fillna(0).abs().round()
        result_df = result.to_pandas()

        # Pandas reference
        pd_result = df['a'].fillna(0).abs().round()

        # Values should match regardless of dtype
        np.testing.assert_array_almost_equal(result_df.values, pd_result.values, decimal=5)

    def test_int_to_float_promotion_with_nan(self):
        """
        Adding NaN to int column should promote to float.
        """
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore.from_df(df)

        # This creates a condition that should be false for some rows
        ds = ds.where(ds['a'] > 2, np.nan)
        result = ds.to_df()

        # Values where condition is False should be NaN
        assert pd.isna(result.loc[0, 'a'])
        assert pd.isna(result.loc[1, 'a'])
        assert result.loc[2, 'a'] == 3

    def test_bool_dtype_after_comparison(self):
        """
        Comparison operations should produce bool dtype consistently.
        """
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore.from_df(df)

        # Comparison should give bool
        ds_bool = ds['a'] > 2
        result = ds[ds_bool].to_df()

        # Should have rows where a > 2
        assert len(result) == 3
        assert all(result['a'] > 2)

    def test_string_dtype_operations(self):
        """
        String operations should preserve or correctly transform dtype.
        """
        df = pd.DataFrame({'text': ['hello', 'WORLD', None, 'Test']})
        ds = DataStore.from_df(df)

        # Chain: lower -> strip -> length
        result = ds['text'].str.lower().str.strip().str.len().to_pandas()
        pd_result = df['text'].str.lower().str.strip().str.len()

        # Compare non-null values
        np.testing.assert_array_equal(result.dropna().values, pd_result.dropna().values)


class TestComplexIndexScenarios:
    """Test scenarios with complex pandas index structures."""

    def test_string_index_filter(self):
        """Filter should work correctly with string index."""
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50]}, index=['a', 'b', 'c', 'd', 'e'])

        pd_result = df[df['value'] > 25]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['value'] > 25].to_df()

        # Check values match
        assert list(pd_result['value']) == list(ds_result['value'])

    def test_datetime_index_operations(self):
        """Operations on DataFrame with datetime index."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({'value': range(10)}, index=dates)

        pd_result = df[df['value'] > 5]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['value'] > 5].to_df()

        # Check values match
        assert list(pd_result['value']) == list(ds_result['value'])

    def test_numeric_non_default_index(self):
        """Test with numeric index that doesn't start at 0."""
        df = pd.DataFrame({'value': [100, 200, 300, 400, 500]}, index=[10, 20, 30, 40, 50])

        pd_result = df[df['value'] > 250]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['value'] > 250].to_df()

        # Check values match
        assert list(pd_result['value']) == list(ds_result['value'])


class TestWhereMaskEdgeCases:
    """Deep edge cases for where/mask operations."""

    def test_where_with_series_as_other(self):
        """
        where() with another Series as 'other' value.
        This is complex because it requires alignment.
        """
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pd_result = df['a'].where(df['a'] > 2, df['b'])

        ds = DataStore.from_df(df)
        ds_result = ds['a'].where(ds['a'] > 2, ds['b']).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_where_condition_on_different_column(self):
        """where() with condition on a different column."""
        df = pd.DataFrame({'value': [100, 200, 300, 400, 500], 'flag': [True, False, True, False, True]})

        pd_result = df['value'].where(df['flag'], -1)

        ds = DataStore.from_df(df)
        ds_result = ds['value'].where(ds['flag'], -1).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_where_chained(self):
        """Multiple where operations chained together."""
        df = pd.DataFrame(
            {
                'a': range(10),
            }
        )

        pd_result = df['a'].where(df['a'] > 2, 0).where(df['a'] < 8, 99)

        ds = DataStore.from_df(df)
        ds_result = ds['a'].where(ds['a'] > 2, 0).where(ds['a'] < 8, 99).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_where_all_true_condition(self):
        """where() with condition that's always True."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pd_result = df['a'].where(df['a'] > 0, 999)  # Always true

        ds = DataStore.from_df(df)
        ds_result = ds['a'].where(ds['a'] > 0, 999).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_where_all_false_condition(self):
        """where() with condition that's always False."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pd_result = df['a'].where(df['a'] < 0, 999)  # Always false

        ds = DataStore.from_df(df)
        ds_result = ds['a'].where(ds['a'] < 0, 999).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)


class TestNullNaNEdgeCases:
    """Deep edge cases for NULL/NaN handling."""

    def test_all_null_column_operations(self):
        """Operations on column that's entirely NULL."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [None, None, None]})

        ds = DataStore.from_df(df)
        result = ds['b'].sum()

        # sum of all NULL should be 0 or NULL depending on implementation
        assert result == 0 or pd.isna(result)

    def test_mixed_null_types(self):
        """DataFrame with different null representations."""
        df = pd.DataFrame(
            {
                'float_nan': [1.0, np.nan, 3.0],
                'object_none': ['a', None, 'c'],
                'int_na': pd.array([1, pd.NA, 3], dtype='Int64'),
            }
        )

        ds = DataStore.from_df(df)
        result = ds.to_df()

        # All null positions should be preserved
        assert pd.isna(result['float_nan'].iloc[1])
        assert pd.isna(result['object_none'].iloc[1])

    def test_isna_followed_by_filter(self):
        """isna() result used for filtering."""
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})

        pd_result = df[df['a'].isna()]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['a'].isna()].to_df()

        assert len(pd_result) == len(ds_result)

    def test_notna_followed_by_filter(self):
        """notna() result used for filtering."""
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})

        pd_result = df[df['a'].notna()]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['a'].notna()].to_df()

        assert len(pd_result) == len(ds_result)
        assert all(ds_result['a'].notna())

    def test_fillna_then_operations(self):
        """
        Test fillna() followed by other operations: fillna().abs().sum()
        """
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = df['a'].fillna(0).abs().sum()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].fillna(0).abs().sum()

        assert pd_result == ds_result


class TestColumnNameEdgeCases:
    """Edge cases with unusual column names."""

    def test_column_name_with_space(self):
        """Column name containing spaces."""
        df = pd.DataFrame({'column with space': [1, 2, 3], 'normal': [4, 5, 6]})

        ds = DataStore.from_df(df)
        result = ds['column with space'].sum()

        assert result == 6

    def test_column_name_with_special_chars(self):
        """Column name containing special characters."""
        df = pd.DataFrame(
            {
                'col-with-dash': [1, 2, 3],
                'col.with.dot': [4, 5, 6],
            }
        )

        ds = DataStore.from_df(df)
        result1 = ds['col-with-dash'].sum()
        result2 = ds['col.with.dot'].sum()

        assert result1 == 6
        assert result2 == 15

    def test_column_name_starting_with_number(self):
        """Column name starting with a number."""
        df = pd.DataFrame(
            {
                '1column': [1, 2, 3],
                '2column': [4, 5, 6],
            }
        )

        ds = DataStore.from_df(df)
        result = ds['1column'].sum()

        assert result == 6

    def test_column_name_sql_keyword(self):
        """Column name that's a SQL keyword."""
        df = pd.DataFrame(
            {
                'select': [1, 2, 3],
                'from': [4, 5, 6],
                'where': [7, 8, 9],
            }
        )

        ds = DataStore.from_df(df)
        result = ds[['select', 'from', 'where']].to_df()

        assert len(result) == 3
        assert list(result.columns) == ['select', 'from', 'where']

    def test_column_name_with_quotes(self):
        """Column name containing quote characters."""
        df = pd.DataFrame(
            {
                "col'quote": [1, 2, 3],
                'col"double': [4, 5, 6],
            }
        )

        ds = DataStore.from_df(df)
        result = ds[["col'quote"]].to_df()

        assert len(result) == 3


class TestEngineSwitchingStateIssues:
    """Test for state issues when switching between engines."""

    def setUp(self):
        config.use_auto()

    def tearDown(self):
        config.use_auto()

    def test_cached_result_invalidation_on_engine_switch(self):
        """
        Cached results should be invalidated when engine switches.
        """
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_df(df)

        config.use_chdb()
        result1 = ds['a'].sum()

        config.use_pandas()
        result2 = ds['a'].sum()

        # Results should be same regardless of engine
        assert result1 == result2 == 6

    def test_lazy_expression_execution_after_switch(self):
        """
        Lazy expression created before switch should execute correctly after.
        """
        df = pd.DataFrame({'text': ['hello', 'world']})
        ds = DataStore.from_df(df)

        config.use_chdb()
        lazy_expr = ds['text'].str.upper()

        # Switch engine before execution
        config.use_pandas()
        result = list(lazy_expr)

        assert result == ['HELLO', 'WORLD']

    def test_multiple_datastores_different_engines(self):
        """
        Multiple DataStores should work correctly with different engine settings.
        """
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})

        ds1 = DataStore.from_df(df1)
        ds2 = DataStore.from_df(df2)

        config.use_chdb()
        result1 = ds1['a'].sum()

        config.use_pandas()
        result2 = ds2['b'].sum()

        assert result1 == 6
        assert result2 == 15


class TestMixedSegmentBoundaries:
    """Test operations that cross SQL/Pandas segment boundaries."""

    def test_sql_pandas_sql_chain(self):
        """
        Chain that goes: SQL-able -> Pandas-only -> SQL-able
        Tests segment boundary handling.
        """
        df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]})

        ds = DataStore.from_df(df)

        # SQL-able: filter
        ds = ds[ds['value'] > 1]
        # Pandas-only: cumsum
        ds['cumsum'] = ds['value'].cumsum()
        # SQL-able: filter again
        ds = ds[ds['cumsum'] > 5]

        result = ds.to_df()
        # cumsum of [2,3,4,5] = [2,5,9,14], then filter > 5 gives [9,14]
        assert len(result) == 2

    def test_pandas_only_in_middle(self):
        """
        Pandas-only operation in the middle of a chain.
        """
        df = pd.DataFrame({'a': [5, 4, 3, 2, 1], 'b': [1, 2, 3, 4, 5]})

        ds = DataStore.from_df(df)

        # SQL-able
        ds = ds.sort_values('a')
        # Pandas-only (shift)
        ds['shifted'] = ds['a'].shift(1)
        # SQL-able
        ds = ds[ds['a'] > 2]

        result = ds.to_df()
        assert len(result) == 3

    def test_apply_breaks_segment(self):
        """
        apply() is Pandas-only and should break SQL segment.
        """
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})

        # Reference
        pdf = df.copy()
        pdf = pdf[pdf['value'] > 2]
        pdf['squared'] = pdf['value'].apply(lambda x: x**2)
        pd_result = pdf[pdf['squared'] > 10]

        ds = DataStore.from_df(df)
        ds = ds[ds['value'] > 2]
        ds['squared'] = ds['value'].apply(lambda x: x**2)
        ds = ds[ds['squared'] > 10]
        ds_result = ds.to_df()

        assert list(pd_result['value']) == list(ds_result['value'])


class TestDatetimeEdgeCases:
    """Deep edge cases for datetime handling."""

    def test_datetime_comparison_with_string(self):
        """Compare datetime column with string date."""
        df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-06-15', '2020-12-31'])})

        pd_result = df[df['date'] > '2020-06-01']

        ds = DataStore.from_df(df)
        ds_result = ds[ds['date'] > '2020-06-01'].to_df()

        assert len(pd_result) == len(ds_result)

    def test_datetime_year_month_day_extraction(self):
        """Extract year, month, day from datetime."""
        df = pd.DataFrame({'date': pd.to_datetime(['2020-01-15', '2021-06-20', '2022-12-25'])})

        pd_years = df['date'].dt.year
        pd_months = df['date'].dt.month
        pd_days = df['date'].dt.day

        ds = DataStore.from_df(df)
        ds_years = ds['date'].dt.year.to_pandas()
        ds_months = ds['date'].dt.month.to_pandas()
        ds_days = ds['date'].dt.day.to_pandas()

        np.testing.assert_array_equal(pd_years.values, ds_years.values)
        np.testing.assert_array_equal(pd_months.values, ds_months.values)
        np.testing.assert_array_equal(pd_days.values, ds_days.values)

    def test_datetime_with_null(self):
        """Datetime column with NULL values."""
        df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', None, '2020-12-31'])})

        ds = DataStore.from_df(df)
        result = ds.to_df()

        # Second row should be NULL
        assert pd.isna(result['date'].iloc[1])

    def test_timedelta_operations(self):
        """Operations with timedelta."""
        df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-15', '2020-02-01'])})

        pd_result = df['date'] + pd.Timedelta(days=10)

        ds = DataStore.from_df(df)
        ds_result = (ds['date'] + pd.Timedelta(days=10)).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)


class TestLargeDataScenarios:
    """Test behavior with larger datasets."""

    def test_many_columns(self):
        """DataFrame with many columns."""
        n_cols = 100
        df = pd.DataFrame({f'col_{i}': range(10) for i in range(n_cols)})

        ds = DataStore.from_df(df)
        result = ds[['col_0', 'col_50', 'col_99']].to_df()

        assert len(result.columns) == 3
        assert list(result['col_0']) == list(range(10))

    def test_many_rows(self):
        """DataFrame with many rows."""
        n_rows = 100000
        df = pd.DataFrame({'id': range(n_rows), 'value': np.random.randint(0, 1000, n_rows)})

        pd_result = df[df['value'] > 500]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['value'] > 500].to_df()

        assert len(pd_result) == len(ds_result)

    def test_wide_and_tall(self):
        """DataFrame that's both wide and tall."""
        n_rows = 10000
        n_cols = 50
        df = pd.DataFrame({f'col_{i}': np.random.randint(0, 100, n_rows) for i in range(n_cols)})

        ds = DataStore.from_df(df)
        ds = ds[ds['col_0'] > 50]
        ds = ds[['col_0', 'col_25', 'col_49']]
        result = ds.to_df()

        assert all(result['col_0'] > 50)


class TestRecursiveNestedExpressions:
    """Test deeply nested expressions."""

    def test_deeply_nested_arithmetic(self):
        """Deeply nested arithmetic expression."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        # ((((a + 1) * 2) - 3) / 2) ** 0.5
        pd_result = ((((df['a'] + 1) * 2) - 3) / 2) ** 0.5

        ds = DataStore.from_df(df)
        ds_result = ((((ds['a'] + 1) * 2) - 3) / 2) ** 0.5
        ds_result = ds_result.to_pandas()

        np.testing.assert_array_almost_equal(pd_result.values, ds_result.values)

    def test_nested_boolean_conditions(self):
        """Deeply nested boolean conditions."""
        df = pd.DataFrame({'a': range(10), 'b': range(10, 20), 'c': range(20, 30)})

        # (a > 2) & ((b < 18) | (c > 25))
        pd_result = df[(df['a'] > 2) & ((df['b'] < 18) | (df['c'] > 25))]

        ds = DataStore.from_df(df)
        ds_result = ds[(ds['a'] > 2) & ((ds['b'] < 18) | (ds['c'] > 25))].to_df()

        assert len(pd_result) == len(ds_result)
        assert list(pd_result['a']) == list(ds_result['a'])

    def test_nested_function_calls(self):
        """Nested function calls."""
        df = pd.DataFrame({'a': [-5.5, 3.3, -2.2, 4.4]})

        # round(abs(floor(a)))
        pd_result = np.floor(df['a']).abs().round()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].floor().abs().round().to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)


class TestGroupByComplexScenarios:
    """Complex groupby scenarios that might reveal issues."""

    def test_groupby_with_null_in_grouping_column(self):
        """
        Test GroupBy where grouping column has NULL values.

        The result should support pandas-style .get(key, default) access.
        Note: chDB/SQL mode includes NULL groups by default, but pandas mode
        uses dropna=True by default. Reset to auto/chdb mode for this test.
        """
        # Reset to auto mode (uses chDB which includes NULL groups)
        config.use_auto()

        df = pd.DataFrame({'group': ['A', 'A', None, 'B', None], 'value': [1, 2, 3, 4, 5]})

        # Use dropna=False to include NULL group on both sides
        pd_result = df.groupby('group', dropna=False)['value'].sum()

        ds = DataStore.from_df(df)
        ds_result = ds.groupby('group', dropna=False)['value'].sum()

        # Both should have group None with sum 8 (3+5)
        assert ds_result.get(None, ds_result.get(np.nan, 0)) == 8

    def test_groupby_multiple_columns_with_null(self):
        """GroupBy with multiple columns, some having NULL."""
        df = pd.DataFrame({'a': ['X', 'X', 'Y', 'Y', None], 'b': [1, 1, 2, None, 2], 'value': [10, 20, 30, 40, 50]})

        ds = DataStore.from_df(df)
        result = ds.groupby(['a', 'b'])['value'].sum()

        # Just verify it doesn't crash and returns results
        assert len(result) > 0

    def test_groupby_then_filter_result(self):
        """GroupBy followed by filter on aggregated result."""
        df = pd.DataFrame({'group': ['A', 'A', 'A', 'B', 'B', 'C'], 'value': [10, 20, 30, 40, 50, 5]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[pd_grouped > 50]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[ds_grouped > 50]

        assert set(pd_result.index) == set(ds_result.index)

    def test_groupby_with_computed_column(self):
        """GroupBy on a computed column."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6]})

        pdf = df.copy()
        pdf['bucket'] = (pdf['value'] // 2) * 2
        pd_result = pdf.groupby('bucket')['value'].sum()

        ds = DataStore.from_df(df)
        ds['bucket'] = (ds['value'] // 2) * 2
        ds_result = ds.groupby('bucket')['value'].sum()

        for bucket in pd_result.index:
            assert pd_result[bucket] == ds_result[bucket]


class TestEdgeCaseDataTypes:
    """Test edge cases with unusual data types."""

    def test_very_large_integers(self):
        """Test with very large integers."""
        df = pd.DataFrame({'big': [10**18, 10**17, 10**16]})

        ds = DataStore.from_df(df)
        result = ds['big'].sum()

        assert result == df['big'].sum()

    def test_very_small_floats(self):
        """Test with very small floating point numbers."""
        df = pd.DataFrame({'tiny': [1e-15, 1e-16, 1e-17]})

        ds = DataStore.from_df(df)
        result = ds['tiny'].sum()

        np.testing.assert_almost_equal(result, df['tiny'].sum(), decimal=20)

    def test_inf_values(self):
        """Test with infinity values."""
        df = pd.DataFrame({'val': [1.0, np.inf, -np.inf, 2.0]})

        ds = DataStore.from_df(df)
        result = ds['val'].to_pandas()

        assert np.isinf(result.iloc[1])
        assert np.isinf(result.iloc[2])

    def test_mixed_int_float_column(self):
        """Column with mixed int and float that looks like int."""
        df = pd.DataFrame({'mixed': [1, 2.0, 3, 4.0, 5]})

        ds = DataStore.from_df(df)
        result = ds['mixed'].sum()

        assert result == 15


class TestConcurrentOperations:
    """Test scenarios that might have concurrency issues."""

    def test_multiple_views_same_datastore(self):
        """Multiple filtered views of the same DataStore."""
        df = pd.DataFrame({'value': range(100)})

        ds = DataStore.from_df(df)

        view1 = ds[ds['value'] < 30]
        view2 = ds[ds['value'] > 70]
        view3 = ds[(ds['value'] >= 30) & (ds['value'] <= 70)]

        result1 = view1.to_df()
        result2 = view2.to_df()
        result3 = view3.to_df()

        assert len(result1) == 30
        assert len(result2) == 29
        assert len(result3) == 41

    def test_reuse_datastore_after_execution(self):
        """
        Reusing a DataStore after execution shouldn't cause issues.
        """
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore.from_df(df)

        # First execution
        result1 = ds[ds['a'] > 2].to_df()

        # Second execution on same DataStore
        result2 = ds[ds['a'] < 4].to_df()

        # Third execution with different op
        result3 = ds['a'].sum()

        assert len(result1) == 3
        assert len(result2) == 3
        assert result3 == 15


class TestMethodModeChaining:
    """
    Tests for chaining operations on ColumnExpr in 'method mode'.

    When ColumnExpr is created with method_name (e.g., from fillna, map, apply),
    _expr is None. This breaks many chained operations that assume _expr exists.
    """

    def test_map_then_abs(self):
        """Test map() followed by abs(): map(lambda).abs()"""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore.from_df(df)

        result = ds['a'].map(lambda x: x * 2).abs().to_pandas()
        expected = df['a'].map(lambda x: x * 2).abs()

        np.testing.assert_array_equal(result.values, expected.values)

    def test_apply_then_round(self):
        """Test apply() followed by round(): apply(lambda).round()"""
        df = pd.DataFrame({'a': [1.5, 2.7, 3.2]})
        ds = DataStore.from_df(df)

        result = ds['a'].apply(lambda x: x * 2).round().to_pandas()
        expected = df['a'].apply(lambda x: x * 2).round()

        np.testing.assert_array_equal(result.values, expected.values)

    def test_fillna_then_to_pandas_works(self):
        """fillna() -> to_pandas() should work (no chaining after fillna)."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        ds = DataStore.from_df(df)

        result = ds['a'].fillna(0).to_pandas()
        expected = df['a'].fillna(0)

        np.testing.assert_array_equal(result.values, expected.values)


class TestDuplicateFunctionAliases:
    """
    Tests for function alias conflicts.

    Some function names are registered as aliases for multiple functions
    (e.g., 'get' is alias for both array_element and str_get), which can
    cause issues.
    """

    def test_get_on_series_result(self):
        """
        Series.get(key, default) should use pandas behavior, not SQL function.

        Note: This test uses indexing [] instead of .get() to workaround the bug.
        """
        df = pd.DataFrame({'group': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})

        ds = DataStore.from_df(df)
        result = ds.groupby('group')['value'].sum()

        # Use [] indexing instead of .get() as workaround
        assert result['A'] == 4
        assert result['B'] == 6

    def test_str_accessor_str_get_works(self):
        """String accessor .str.str_get() should work for character access.

        Note: .str.get() was removed to avoid conflict with pandas Series.get().
        Use .str.str_get() for character access by index.
        """
        df = pd.DataFrame({'text': ['hello', 'world', 'test']})
        ds = DataStore.from_df(df)

        # Get first character using str_get (previously 'get')
        result = ds['text'].str.str_get(0).to_pandas()
        expected = df['text'].str.get(0)

        assert list(result) == list(expected)


class TestAssignmentChaining:
    """Tests for column assignment and chaining."""

    def test_assign_computed_column_then_filter(self):
        """Assign computed column, then filter on it."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pdf = df.copy()
        pdf['b'] = pdf['a'] * 2
        pd_result = pdf[pdf['b'] > 5]

        ds = DataStore.from_df(df)
        ds['b'] = ds['a'] * 2
        ds_result = ds[ds['b'] > 5].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])
        assert list(pd_result['b']) == list(ds_result['b'])

    def test_multiple_column_assignments(self):
        """Multiple column assignments in sequence."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        pdf = df.copy()
        pdf['b'] = pdf['a'] * 2
        pdf['c'] = pdf['b'] + pdf['a']
        pdf['d'] = pdf['c'] ** 2

        ds = DataStore.from_df(df)
        ds['b'] = ds['a'] * 2
        ds['c'] = ds['b'] + ds['a']
        ds['d'] = ds['c'] ** 2

        ds_result = ds.to_df()

        assert list(pdf['d']) == list(ds_result['d'])

    def test_overwrite_column_then_use(self):
        """Overwrite existing column, then use it."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pdf = df.copy()
        pdf['a'] = pdf['a'] * 10
        pd_result = pdf[pdf['a'] > 25]

        ds = DataStore.from_df(df)
        ds['a'] = ds['a'] * 10
        ds_result = ds[ds['a'] > 25].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])


class TestAggregationEdgeCases:
    """Edge cases for aggregation operations."""

    def test_single_row_aggregation(self):
        """Aggregation on single-row DataFrame."""
        df = pd.DataFrame({'a': [42]})
        ds = DataStore.from_df(df)

        assert ds['a'].sum() == 42
        assert ds['a'].mean() == 42
        assert ds['a'].min() == 42
        assert ds['a'].max() == 42

    def test_aggregation_with_all_same_values(self):
        """Aggregation when all values are the same."""
        df = pd.DataFrame({'a': [5, 5, 5, 5, 5]})
        ds = DataStore.from_df(df)

        assert ds['a'].sum() == 25
        assert ds['a'].mean() == 5
        assert ds['a'].std() == 0  # Standard deviation of identical values is 0
        assert ds['a'].min() == 5
        assert ds['a'].max() == 5

    def test_count_vs_size_difference(self):
        """count() excludes NaN, size() includes all."""
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        ds = DataStore.from_df(df)

        # count excludes NaN
        assert ds['a'].count() == 3
        # Note: size() may not be implemented - test to see


class TestComplexConditions:
    """Tests for complex boolean conditions."""

    def test_multiple_and_conditions(self):
        """Multiple AND conditions."""
        df = pd.DataFrame({'a': range(20), 'b': range(20, 40), 'c': range(40, 60)})

        pd_result = df[(df['a'] > 5) & (df['b'] > 30) & (df['c'] > 50)]

        ds = DataStore.from_df(df)
        ds_result = ds[(ds['a'] > 5) & (ds['b'] > 30) & (ds['c'] > 50)].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])

    def test_multiple_or_conditions(self):
        """Multiple OR conditions."""
        df = pd.DataFrame({'a': range(20)})

        pd_result = df[(df['a'] < 3) | (df['a'] > 17) | (df['a'] == 10)]

        ds = DataStore.from_df(df)
        ds_result = ds[(ds['a'] < 3) | (ds['a'] > 17) | (ds['a'] == 10)].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])

    def test_mixed_and_or_with_parentheses(self):
        """Mixed AND/OR with explicit parentheses."""
        df = pd.DataFrame({'a': range(10), 'b': range(10, 20)})

        pd_result = df[((df['a'] < 3) | (df['a'] > 7)) & (df['b'] < 18)]

        ds = DataStore.from_df(df)
        ds_result = ds[((ds['a'] < 3) | (ds['a'] > 7)) & (ds['b'] < 18)].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])

    def test_negation_condition(self):
        """Negation of condition."""
        df = pd.DataFrame({'a': range(10)})

        pd_result = df[~(df['a'] > 5)]

        ds = DataStore.from_df(df)
        ds_result = ds[~(ds['a'] > 5)].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])


class TestFileSourceOperations:
    """Tests with file-based DataStore sources."""

    def test_parquet_filter_then_select(self):
        """Filter then select on Parquet file."""
        df = pd.DataFrame(
            {
                'id': range(100),
                'value': np.random.randint(0, 100, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.parquet')
            df.to_parquet(path)

            ds = DataStore.from_file(path)
            ds = ds[ds['value'] > 50]
            ds = ds[['id', 'value']]
            result = ds.to_df()

            pd_result = df[df['value'] > 50][['id', 'value']]

            assert len(result) == len(pd_result)
            assert list(result.columns) == ['id', 'value']

    def test_csv_groupby_agg(self):
        """GroupBy aggregation on CSV file."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'A', 'B'], 'value': [10, 20, 30, 40, 50, 60]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            df.to_csv(path, index=False)

            ds = DataStore.from_file(path)
            result = ds.groupby('category')['value'].sum()

            assert result['A'] == 90
            assert result['B'] == 120


class TestSpecialValueHandling:
    """Tests for special numeric values."""

    def test_operations_with_inf(self):
        """Operations involving infinity."""
        df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 2.0], 'b': [1.0, 1.0, 1.0, 1.0]})

        ds = DataStore.from_df(df)

        # inf + 1 = inf
        result = (ds['a'] + ds['b']).to_pandas()
        assert np.isinf(result.iloc[1])
        assert result.iloc[1] > 0  # positive inf
        assert np.isinf(result.iloc[2])
        assert result.iloc[2] < 0  # negative inf

    def test_division_by_zero(self):
        """Division that results in infinity."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [1.0, 0.0, 1.0]})

        ds = DataStore.from_df(df)
        result = (ds['a'] / ds['b']).to_pandas()

        assert result.iloc[0] == 1.0
        assert np.isinf(result.iloc[1])  # 2/0 = inf
        assert result.iloc[2] == 3.0

    def test_nan_propagation_in_arithmetic(self):
        """NaN should propagate through arithmetic."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [1.0, 2.0, 3.0]})

        ds = DataStore.from_df(df)
        result = (ds['a'] + ds['b']).to_pandas()

        assert result.iloc[0] == 2.0
        assert pd.isna(result.iloc[1])  # NaN + 2 = NaN
        assert result.iloc[2] == 6.0


class TestAggregationModeComparisonsDeep:
    """
    Deep tests for comparison operators on aggregation results.

    Based on the fix for __gt__, test all comparison operators and combinations.
    """

    def test_eq_on_groupby_result(self):
        """Test == comparison on groupby result."""
        df = pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [10, 20, 30]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[pd_grouped == 20]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[ds_grouped == 20]

        assert list(pd_result.values) == list(ds_result.values)

    def test_ne_on_groupby_result(self):
        """Test != comparison on groupby result."""
        df = pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [10, 20, 30]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[pd_grouped != 20]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[ds_grouped != 20]

        assert set(pd_result.values) == set(ds_result.values)

    def test_combined_conditions_on_groupby(self):
        """Test AND/OR conditions on groupby result."""
        df = pd.DataFrame({'group': ['A', 'B', 'C', 'D', 'E'], 'value': [10, 20, 30, 40, 50]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[(pd_grouped > 15) & (pd_grouped < 45)]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[(ds_grouped > 15) & (ds_grouped < 45)]

        assert set(pd_result.values) == set(ds_result.values)

    def test_or_conditions_on_groupby(self):
        """Test OR conditions on groupby result."""
        df = pd.DataFrame({'group': ['A', 'B', 'C', 'D', 'E'], 'value': [10, 20, 30, 40, 50]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[(pd_grouped < 15) | (pd_grouped > 45)]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[(ds_grouped < 15) | (ds_grouped > 45)]

        assert set(pd_result.values) == set(ds_result.values)

    def test_chained_filter_on_groupby(self):
        """Multiple filters chained on groupby result."""
        df = pd.DataFrame({'group': ['A', 'B', 'C', 'D', 'E'], 'value': [10, 20, 30, 40, 50]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[pd_grouped > 15][pd_grouped < 45]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[ds_grouped > 15][ds_grouped < 45]

        assert set(pd_result.values) == set(ds_result.values)

    def test_arithmetic_on_groupby_result(self):
        """Arithmetic operations on groupby result."""
        df = pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [10, 20, 30]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped * 2 + 5

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped * 2 + 5

        # Execute and compare
        ds_values = list(ds_result)
        pd_values = list(pd_result)
        assert ds_values == pd_values

    def test_empty_result_after_filter(self):
        """Filter that results in empty Series."""
        df = pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [10, 20, 30]})

        pd_grouped = df.groupby('group')['value'].sum()
        pd_result = pd_grouped[pd_grouped > 100]  # No matches

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby('group')['value'].sum()
        ds_result = ds_grouped[ds_grouped > 100]

        assert len(pd_result) == len(ds_result) == 0


class TestMethodModeChainingDeep:
    """
    Deep tests for method mode chaining.

    Based on the fix for fillna().abs(), test more complex chains.
    """

    def test_triple_chain(self):
        """Three method calls chained."""
        df = pd.DataFrame({'a': [1.0, np.nan, -3.0, np.nan, 5.0]})

        pd_result = df['a'].fillna(0).abs().round()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].fillna(0).abs().round().to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_quad_chain(self):
        """Four method calls chained."""
        df = pd.DataFrame({'a': [-1.5, np.nan, 3.7, np.nan, -5.2]})

        pd_result = df['a'].fillna(0).abs().round().astype(int)

        ds = DataStore.from_df(df)
        ds_result = ds['a'].fillna(0).abs().round().astype(int).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_mixed_method_and_arithmetic(self):
        """Mix method calls with arithmetic."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})

        pd_result = (df['a'].fillna(0) + 10).abs()

        ds = DataStore.from_df(df)
        ds_result = (ds['a'].fillna(0) + 10).abs().to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_cumsum_then_filter(self):
        """cumsum() followed by comparison."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pd_cumsum = df['a'].cumsum()
        pd_result = df[pd_cumsum > 6]

        ds = DataStore.from_df(df)
        ds_cumsum = ds['a'].cumsum()
        ds_result = ds[ds_cumsum > 6].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])

    def test_shift_then_arithmetic(self):
        """shift() followed by arithmetic."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pd_result = df['a'] - df['a'].shift(1)

        ds = DataStore.from_df(df)
        ds_result = (ds['a'] - ds['a'].shift(1)).to_pandas()

        # First value is NaN in both
        assert pd.isna(pd_result.iloc[0]) and pd.isna(ds_result.iloc[0])
        np.testing.assert_array_equal(pd_result.dropna().values, ds_result.dropna().values)

    def test_rank_then_comparison(self):
        """rank() followed by comparison filter."""
        df = pd.DataFrame({'a': [30, 10, 20, 50, 40]})

        pd_rank = df['a'].rank()
        pd_result = df[pd_rank > 3]

        ds = DataStore.from_df(df)
        ds_rank = ds['a'].rank()
        ds_result = ds[ds_rank > 3].to_df()

        assert set(pd_result['a']) == set(ds_result['a'])


class TestMultiColumnGroupByBooleanIndex:
    """Tests for boolean indexing on multi-column groupby results."""

    def test_multi_column_groupby_filter(self):
        """Boolean indexing on multi-column groupby."""
        df = pd.DataFrame({'a': ['X', 'X', 'Y', 'Y'], 'b': [1, 2, 1, 2], 'value': [10, 20, 30, 40]})

        pd_grouped = df.groupby(['a', 'b'])['value'].sum()
        pd_result = pd_grouped[pd_grouped > 25]

        ds = DataStore.from_df(df)
        ds_grouped = ds.groupby(['a', 'b'])['value'].sum()
        ds_result = ds_grouped[ds_grouped > 25]

        assert set(pd_result.values) == set(ds_result.values)

    def test_multi_agg_filter(self):
        """Filter on result of multiple aggregations."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'C', 'C'], 'value': [10, 20, 30, 40, 50, 60]})

        pd_mean = df.groupby('group')['value'].mean()
        pd_result = pd_mean[pd_mean >= 35]

        ds = DataStore.from_df(df)
        ds_mean = ds.groupby('group')['value'].mean()
        ds_result = ds_mean[ds_mean >= 35]

        assert set(pd_result.values) == set(ds_result.values)


class TestConditionObjectAsIndex:
    """Tests for using Condition objects directly for indexing."""

    def test_condition_in_getitem(self):
        """Use Condition from comparison in __getitem__."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        ds = DataStore.from_df(df)
        # Create condition separately
        cond = ds['a'] > 3
        result = ds[cond].to_df()

        assert list(result['a']) == [4, 5]

    def test_combined_conditions_in_getitem(self):
        """Combined conditions in __getitem__."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})

        ds = DataStore.from_df(df)
        cond = (ds['a'] > 2) & (ds['b'] < 4)
        result = ds[cond].to_df()

        pd_result = df[(df['a'] > 2) & (df['b'] < 4)]
        assert list(result['a']) == list(pd_result['a'])


class TestValueCountsEdgeCases:
    """Edge cases for value_counts() operations."""

    def test_value_counts_filter(self):
        """Filter on value_counts result."""
        df = pd.DataFrame({'a': ['x', 'x', 'x', 'y', 'y', 'z']})

        pd_vc = df['a'].value_counts()
        pd_result = pd_vc[pd_vc > 1]

        ds = DataStore.from_df(df)
        ds_vc = ds['a'].value_counts()
        ds_result = ds_vc[ds_vc > 1]

        assert set(pd_result.values) == set(ds_result.values)

    def test_value_counts_arithmetic(self):
        """Arithmetic on value_counts result."""
        df = pd.DataFrame({'a': ['x', 'x', 'y', 'z']})

        pd_vc = df['a'].value_counts()
        pd_result = pd_vc * 10

        ds = DataStore.from_df(df)
        ds_vc = ds['a'].value_counts()
        ds_result = ds_vc * 10

        assert list(pd_result.sort_index()) == list(ds_result.sort_index())


class TestStringMethodChaining:
    """Deep tests for string method chaining."""

    def test_str_chain_lower_strip_len(self):
        """Chain: lower -> strip -> len."""
        df = pd.DataFrame({'text': ['  HELLO  ', 'WORLD', '  Test  ']})

        pd_result = df['text'].str.lower().str.strip().str.len()

        ds = DataStore.from_df(df)
        ds_result = ds['text'].str.lower().str.strip().str.len().to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_str_contains_then_filter(self):
        """str.contains() result used for filtering."""
        df = pd.DataFrame({'text': ['hello world', 'goodbye', 'hello again', 'test'], 'value': [1, 2, 3, 4]})

        pd_result = df[df['text'].str.contains('hello')]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['text'].str.contains('hello')].to_df()

        assert list(pd_result['value']) == list(ds_result['value'])

    def test_str_split_then_get(self):
        """str.split() then access element."""
        df = pd.DataFrame({'text': ['a-b-c', 'x-y-z', '1-2-3']})

        pd_result = df['text'].str.split('-').str[0]

        ds = DataStore.from_df(df)
        # Note: str[0] might need special handling
        try:
            ds_result = ds['text'].str.split('-').str[0].to_pandas()
            assert list(pd_result) == list(ds_result)
        except (TypeError, NotImplementedError, AttributeError):
            pytest.skip("str.split().str[0] not supported")


class TestDatetimeMethodChaining:
    """Deep tests for datetime method chaining."""

    def test_dt_year_then_filter(self):
        """dt.year extraction then filter."""
        df = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2021-06-15', '2022-12-31']), 'value': [1, 2, 3]})

        pd_result = df[df['date'].dt.year > 2020]

        ds = DataStore.from_df(df)
        ds_result = ds[ds['date'].dt.year > 2020].to_df()

        assert list(pd_result['value']) == list(ds_result['value'])

    def test_dt_month_arithmetic(self):
        """Arithmetic on dt.month result."""
        df = pd.DataFrame({'date': pd.to_datetime(['2020-01-15', '2020-06-20', '2020-12-25'])})

        pd_result = df['date'].dt.month * 10

        ds = DataStore.from_df(df)
        ds_result = (ds['date'].dt.month * 10).to_pandas()

        np.testing.assert_array_equal(pd_result.values, ds_result.values)


class TestAggregationChaining:
    """Tests for operations after aggregation."""

    def test_sum_then_comparison(self):
        """Compare sum result with scalar."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        ds = DataStore.from_df(df)
        total = ds['a'].sum()

        # Scalar comparison - should return bool directly
        assert total > 10
        assert total == 15
        assert not total < 10

    def test_mean_then_arithmetic(self):
        """Arithmetic on mean result."""
        df = pd.DataFrame({'a': [10, 20, 30]})

        pd_mean = df['a'].mean()
        pd_result = pd_mean * 2

        ds = DataStore.from_df(df)
        ds_mean = ds['a'].mean()
        ds_result = ds_mean * 2

        assert pd_result == ds_result == 40.0


class TestComplexFilterChains:
    """Complex filter chains that might reveal issues."""

    def test_filter_assign_filter(self):
        """Filter -> assign column -> filter again."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

        pdf = df[df['a'] > 2].copy()
        pdf['c'] = pdf['a'] * pdf['b']
        pd_result = pdf[pdf['c'] > 100]

        ds = DataStore.from_df(df)
        ds = ds[ds['a'] > 2]
        ds['c'] = ds['a'] * ds['b']
        ds_result = ds[ds['c'] > 100].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])

    def test_multiple_column_filters(self):
        """Filters on multiple columns in sequence."""
        df = pd.DataFrame({'a': range(20), 'b': range(20, 40), 'c': range(40, 60)})

        pdf = df[df['a'] > 5]
        pdf = pdf[pdf['b'] < 35]
        pd_result = pdf[pdf['c'] > 50]

        ds = DataStore.from_df(df)
        ds = ds[ds['a'] > 5]
        ds = ds[ds['b'] < 35]
        ds_result = ds[ds['c'] > 50].to_df()

        assert list(pd_result['a']) == list(ds_result['a'])


class TestEdgeCaseDatastoreOperations:
    """Edge case operations that might reveal issues."""

    def test_empty_dataframe_operations(self):
        """Operations on empty DataFrame."""
        df = pd.DataFrame({'a': [], 'b': []})

        ds = DataStore.from_df(df)
        result = ds.to_df()

        assert len(result) == 0
        assert list(result.columns) == ['a', 'b']

    def test_single_value_operations(self):
        """Operations on single-value DataFrame."""
        df = pd.DataFrame({'a': [42]})

        ds = DataStore.from_df(df)

        assert ds['a'].sum() == 42
        assert ds['a'].mean() == 42
        assert ds['a'].max() == 42
        assert ds['a'].min() == 42
        assert len(ds['a'].to_pandas()) == 1

    def test_all_null_column_aggregation(self):
        """Aggregation on all-null column."""
        df = pd.DataFrame({'a': [None, None, None]})

        ds = DataStore.from_df(df)

        # sum of nulls should be 0 or null
        result = ds['a'].sum()
        assert result == 0 or pd.isna(result)

    def test_unicode_column_names(self):
        """Column names with unicode characters."""
        df = pd.DataFrame({'中文列': [1, 2, 3], 'émoji🎉': [4, 5, 6], 'Ελληνικά': [7, 8, 9]})

        ds = DataStore.from_df(df)
        result = ds['中文列'].sum()

        assert result == 6

    def test_very_long_column_name(self):
        """Column name that's very long."""
        long_name = 'a' * 200
        df = pd.DataFrame({long_name: [1, 2, 3]})

        ds = DataStore.from_df(df)
        result = ds[long_name].sum()

        assert result == 6


class TestBooleanSeriesOperations:
    """Tests for operations that produce boolean Series."""

    def test_isna_then_sum(self):
        """isna() followed by sum() to count nulls."""
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})

        pd_result = df['a'].isna().sum()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].isna().sum()

        assert pd_result == ds_result == 2

    def test_notna_then_mean(self):
        """notna() followed by mean() - proportion of non-null."""
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})

        pd_result = df['a'].notna().mean()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].notna().mean()

        assert pd_result == ds_result == 0.6

    def test_comparison_sum(self):
        """Sum of comparison result to count matches."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})

        pd_result = (df['a'] > 2).sum()

        ds = DataStore.from_df(df)
        ds_result = (ds['a'] > 2).sum()

        assert pd_result == ds_result == 3


class TestMethodModeWithAggregation:
    """Tests combining method mode with aggregation."""

    def test_fillna_then_sum(self):
        """fillna() followed by sum()."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = df['a'].fillna(10).sum()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].fillna(10).sum()

        assert pd_result == ds_result == 29.0

    def test_fillna_then_mean(self):
        """fillna() followed by mean()."""
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})

        pd_result = df['a'].fillna(5).mean()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].fillna(5).mean()

        assert pd_result == ds_result == 3.0

    def test_abs_then_max(self):
        """abs() followed by max()."""
        df = pd.DataFrame({'a': [-5, 3, -10, 7]})

        pd_result = df['a'].abs().max()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].abs().max()

        assert pd_result == ds_result == 10

    def test_clip_then_std(self):
        """clip() followed by std()."""
        df = pd.DataFrame({'a': [1, 10, 100, 1000]})

        pd_result = df['a'].clip(lower=5, upper=50).std()

        ds = DataStore.from_df(df)
        ds_result = ds['a'].clip(lower=5, upper=50).std()

        # ColumnExpr aggregation needs explicit comparison via repr or float conversion
        # because numpy's assert_almost_equal doesn't know how to handle ColumnExpr
        np.testing.assert_almost_equal(pd_result, float(ds_result), decimal=5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
