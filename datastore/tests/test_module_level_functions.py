"""
Tests for pandas-compatible module-level functions in datastore.

These tests verify that datastore module-level functions work correctly
and produce results compatible with pandas.
"""

from tests.test_utils import assert_frame_equal, assert_series_equal, get_dataframe
import numpy as np
import pandas as pd
import pytest

import datastore as ds


class TestMissingValueFunctions:
    """Test isna, isnull, notna, notnull functions."""

    def test_isna_scalar_none(self):
        """isna should return True for None."""
        assert ds.isna(None) is True
        assert pd.isna(None) == ds.isna(None)

    def test_isna_scalar_nan(self):
        """isna should return True for NaN."""
        assert ds.isna(float('nan')) is True
        assert pd.isna(float('nan')) == ds.isna(float('nan'))

    def test_isna_scalar_value(self):
        """isna should return False for valid values."""
        assert ds.isna(1) is False
        assert ds.isna('hello') is False
        assert ds.isna(0) is False

    def test_isna_array(self):
        """isna should work on arrays."""
        arr = [1, None, 3, float('nan')]
        result = ds.isna(arr)
        expected = pd.isna(arr)
        np.testing.assert_array_equal(result, expected)

    def test_isnull_is_alias(self):
        """isnull should be alias for isna."""
        assert ds.isnull(None) == ds.isna(None)
        assert ds.isnull(1) == ds.isna(1)

    def test_notna_scalar(self):
        """notna should return inverse of isna."""
        assert ds.notna(1) is True
        assert ds.notna(None) is False
        assert ds.notna(float('nan')) is False

    def test_notna_array(self):
        """notna should work on arrays."""
        arr = [1, None, 3, float('nan')]
        result = ds.notna(arr)
        expected = pd.notna(arr)
        np.testing.assert_array_equal(result, expected)

    def test_notnull_is_alias(self):
        """notnull should be alias for notna."""
        assert ds.notnull(1) == ds.notna(1)
        assert ds.notnull(None) == ds.notna(None)


class TestTypeConversionFunctions:
    """Test to_datetime, to_numeric, to_timedelta functions."""

    def test_to_datetime_string(self):
        """to_datetime should parse date string."""
        result = ds.to_datetime('2024-01-15')
        expected = pd.to_datetime('2024-01-15')
        assert result == expected

    def test_to_datetime_list(self):
        """to_datetime should parse list of strings."""
        dates = ['2024-01-15', '2024-02-20', '2024-03-25']
        result = ds.to_datetime(dates)
        expected = pd.to_datetime(dates)
        pd.testing.assert_index_equal(result, expected)

    def test_to_datetime_with_format(self):
        """to_datetime should respect format parameter."""
        result = ds.to_datetime('15/01/2024', format='%d/%m/%Y')
        expected = pd.to_datetime('15/01/2024', format='%d/%m/%Y')
        assert result == expected

    def test_to_datetime_errors_coerce(self):
        """to_datetime should coerce errors when specified."""
        result = ds.to_datetime('not a date', errors='coerce')
        assert pd.isna(result)

    def test_to_numeric_string(self):
        """to_numeric should convert string to number."""
        assert ds.to_numeric('1.5') == 1.5
        assert ds.to_numeric('42') == 42

    def test_to_numeric_list(self):
        """to_numeric should convert list of strings."""
        result = ds.to_numeric(['1', '2', '3'])
        expected = pd.to_numeric(['1', '2', '3'])
        np.testing.assert_array_equal(result, expected)

    def test_to_numeric_errors_coerce(self):
        """to_numeric should coerce errors when specified."""
        result = ds.to_numeric('not a number', errors='coerce')
        assert np.isnan(result)

    def test_to_timedelta_string(self):
        """to_timedelta should parse timedelta string."""
        result = ds.to_timedelta('1 days')
        expected = pd.to_timedelta('1 days')
        assert result == expected

    def test_to_timedelta_with_unit(self):
        """to_timedelta should respect unit parameter."""
        result = ds.to_timedelta(1, unit='h')
        expected = pd.to_timedelta(1, unit='h')
        assert result == expected


class TestDateRangeFunctions:
    """Test date_range, bdate_range, period_range, timedelta_range, interval_range."""

    def test_date_range_basic(self):
        """date_range should create DatetimeIndex."""
        result = ds.date_range('2024-01-01', periods=5, freq='D')
        expected = pd.date_range('2024-01-01', periods=5, freq='D')
        pd.testing.assert_index_equal(result, expected)

    def test_date_range_start_end(self):
        """date_range should work with start and end."""
        result = ds.date_range(start='2024-01-01', end='2024-01-05')
        expected = pd.date_range(start='2024-01-01', end='2024-01-05')
        pd.testing.assert_index_equal(result, expected)

    def test_date_range_with_tz(self):
        """date_range should respect timezone."""
        result = ds.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
        expected = pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
        pd.testing.assert_index_equal(result, expected)

    def test_bdate_range_basic(self):
        """bdate_range should create business day DatetimeIndex."""
        result = ds.bdate_range('2024-01-01', periods=5)
        expected = pd.bdate_range('2024-01-01', periods=5)
        pd.testing.assert_index_equal(result, expected)

    def test_period_range_basic(self):
        """period_range should create PeriodIndex."""
        result = ds.period_range('2024-01', periods=3, freq='M')
        expected = pd.period_range('2024-01', periods=3, freq='M')
        pd.testing.assert_index_equal(result, expected)

    def test_timedelta_range_basic(self):
        """timedelta_range should create TimedeltaIndex."""
        result = ds.timedelta_range('1 day', periods=3, freq='12h')
        expected = pd.timedelta_range('1 day', periods=3, freq='12h')
        pd.testing.assert_index_equal(result, expected)

    def test_interval_range_basic(self):
        """interval_range should create IntervalIndex."""
        result = ds.interval_range(start=0, end=5)
        expected = pd.interval_range(start=0, end=5)
        pd.testing.assert_index_equal(result, expected)


class TestBinningFunctions:
    """Test cut, qcut functions."""

    def test_cut_basic(self):
        """cut should bin values into intervals."""
        data = [1, 7, 5, 4, 6, 3]
        result = ds.cut(data, 3)
        expected = pd.cut(data, 3)
        pd.testing.assert_extension_array_equal(result, expected)

    def test_cut_with_labels(self):
        """cut should use custom labels."""
        data = [1, 7, 5, 4, 6, 3]
        labels = ['low', 'medium', 'high']
        result = ds.cut(data, 3, labels=labels)
        expected = pd.cut(data, 3, labels=labels)
        pd.testing.assert_extension_array_equal(result, expected)

    def test_cut_with_bins_list(self):
        """cut should work with explicit bin edges."""
        data = [1, 7, 5, 4, 6, 3]
        bins = [0, 3, 5, 10]
        result = ds.cut(data, bins)
        expected = pd.cut(data, bins)
        pd.testing.assert_extension_array_equal(result, expected)

    def test_qcut_basic(self):
        """qcut should create quantile-based bins."""
        data = list(range(10))
        result = ds.qcut(data, 4)
        expected = pd.qcut(data, 4)
        pd.testing.assert_extension_array_equal(result, expected)

    def test_qcut_with_labels(self):
        """qcut should use custom labels."""
        data = list(range(10))
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        result = ds.qcut(data, 4, labels=labels)
        expected = pd.qcut(data, 4, labels=labels)
        pd.testing.assert_extension_array_equal(result, expected)


class TestCategorizationFunctions:
    """Test get_dummies, factorize, unique, value_counts functions."""

    def test_get_dummies_list(self):
        """get_dummies should convert list to dummy variables."""
        data = ['a', 'b', 'a', 'c']
        result = ds.get_dummies(data)
        expected = pd.get_dummies(data)
        # Result is DataStore, convert to DataFrame for comparison
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_get_dummies_with_prefix(self):
        """get_dummies should use custom prefix."""
        data = ['a', 'b', 'a']
        result = ds.get_dummies(data, prefix='cat')
        expected = pd.get_dummies(data, prefix='cat')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_get_dummies_drop_first(self):
        """get_dummies should drop first category when specified."""
        data = ['a', 'b', 'a', 'c']
        result = ds.get_dummies(data, drop_first=True)
        expected = pd.get_dummies(data, drop_first=True)
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_get_dummies_datastore_with_columns(self):
        """get_dummies should work with DataStore input and columns parameter."""
        df = pd.DataFrame({
            'A': ['a', 'b', 'a'],
            'B': ['x', 'y', 'x'],
            'C': [1, 2, 3]
        })
        ds_df = ds.DataStore(df)

        # This was a bug - DataStore input caused ValueError
        result = ds.get_dummies(ds_df, columns=['A', 'B'])
        expected = pd.get_dummies(df, columns=['A', 'B'])

        assert isinstance(result, ds.DataStore)
        result_df = result.to_df()
        assert_frame_equal(result_df, expected)

    def test_get_dummies_datastore_no_columns(self):
        """get_dummies should work with DataStore input without columns parameter."""
        df = pd.DataFrame({
            'A': ['a', 'b', 'a'],
            'C': [1, 2, 3]
        })
        ds_df = ds.DataStore(df)

        result = ds.get_dummies(ds_df)
        expected = pd.get_dummies(df)

        assert isinstance(result, ds.DataStore)
        result_df = result.to_df()
        assert_frame_equal(result_df, expected)

    def test_factorize_basic(self):
        """factorize should encode values."""
        data = ['b', 'a', 'c', 'b']
        # pandas 3.0 requires Series/Index instead of list for factorize
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        data_series = pd.Series(data)
        codes, uniques = ds.factorize(data_series)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='factorize with argument that is not')
            expected_codes, expected_uniques = pd.factorize(data_series)
        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(uniques, expected_uniques)

    def test_factorize_sorted(self):
        """factorize should sort when specified."""
        data = ['b', 'a', 'c', 'b']
        # pandas 3.0 requires Series/Index instead of list for factorize
        data_series = pd.Series(data)
        codes, uniques = ds.factorize(data_series, sort=True)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='factorize with argument that is not')
            expected_codes, expected_uniques = pd.factorize(data_series, sort=True)
        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(uniques, expected_uniques)

    def test_unique_basic(self):
        """unique should return unique values."""
        data = np.array([1, 2, 2, 3, 1])
        result = ds.unique(data)
        expected = pd.unique(data)
        np.testing.assert_array_equal(result, expected)

    def test_value_counts_basic(self):
        """value_counts should count values."""
        data = pd.Series(['a', 'b', 'a', 'a'])
        result = ds.value_counts(data)
        # pandas 3.0 removed pd.value_counts(), use Series.value_counts() instead
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='pandas.value_counts is deprecated')
            if pandas_version >= (3, 0):
                expected = data.value_counts()
            else:
                expected = pd.value_counts(data)
        assert_series_equal(result, expected)


class TestReshapingFunctions:
    """Test melt, pivot, pivot_table, crosstab, wide_to_long functions."""

    def test_melt_basic(self):
        """melt should unpivot DataFrame."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = ds.melt(df, id_vars=['A'], value_vars=['B', 'C'])
        expected = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_pivot_basic(self):
        """pivot should reshape DataFrame."""
        df = pd.DataFrame({'foo': ['one', 'one', 'two', 'two'], 'bar': ['A', 'B', 'A', 'B'], 'baz': [1, 2, 3, 4]})
        result = ds.pivot(df, columns='bar', index='foo', values='baz')
        expected = pd.pivot(df, columns='bar', index='foo', values='baz')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_pivot_table_basic(self):
        """pivot_table should create pivot table."""
        df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'], 'B': ['one', 'two', 'one', 'two'], 'C': [1, 2, 3, 4]})
        result = ds.pivot_table(df, values='C', index='A', columns='B', aggfunc='sum')
        expected = pd.pivot_table(df, values='C', index='A', columns='B', aggfunc='sum')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_crosstab_basic(self):
        """crosstab should compute cross tabulation."""
        a = ['foo', 'foo', 'bar', 'bar']
        b = ['one', 'two', 'one', 'two']
        result = ds.crosstab(a, b)
        expected = pd.crosstab(a, b)
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_wide_to_long_basic(self):
        """wide_to_long should unpivot wide format."""
        df = pd.DataFrame({'A1970': [1, 2], 'A1980': [3, 4], 'B1970': [5, 6], 'B1980': [7, 8], 'id': [0, 1]})
        result = ds.wide_to_long(df, stubnames=['A', 'B'], i='id', j='year')
        expected = pd.wide_to_long(df, stubnames=['A', 'B'], i='id', j='year')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)


class TestMergeFunctions:
    """Test merge, merge_asof, merge_ordered functions."""

    def test_merge_basic(self):
        """merge should join DataFrames."""
        left = pd.DataFrame({'key': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        right = pd.DataFrame({'key': ['a', 'b', 'd'], 'other': [4, 5, 6]})
        result = ds.merge(left, right, on='key')
        expected = pd.merge(left, right, on='key')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_merge_asof_basic(self):
        """merge_asof should merge by nearest key."""
        left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
        right = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]})
        result = ds.merge_asof(left, right, on='a')
        expected = pd.merge_asof(left, right, on='a')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_merge_ordered_basic(self):
        """merge_ordered should merge with ordered fills."""
        left = pd.DataFrame({'key': ['a', 'c', 'e'], 'lvalue': [1, 2, 3]})
        right = pd.DataFrame({'key': ['b', 'c', 'd'], 'rvalue': [4, 5, 6]})
        result = ds.merge_ordered(left, right, on='key')
        expected = pd.merge_ordered(left, right, on='key')
        result = get_dataframe(result)
        assert_frame_equal(result, expected)


class TestConcatFunction:
    """Test concat function."""

    def test_concat_basic(self):
        """concat should concatenate DataFrames."""
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        result = ds.concat([df1, df2])
        expected = pd.concat([df1, df2])
        result = get_dataframe(result)
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_concat_axis1(self):
        """concat should work along axis 1."""
        df1 = pd.DataFrame({'A': [1, 2]})
        df2 = pd.DataFrame({'B': [3, 4]})
        result = ds.concat([df1, df2], axis=1)
        expected = pd.concat([df1, df2], axis=1)
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_concat_ignore_index(self):
        """concat should ignore index when specified."""
        df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({'A': [3, 4]}, index=[0, 1])
        result = ds.concat([df1, df2], ignore_index=True)
        expected = pd.concat([df1, df2], ignore_index=True)
        result = get_dataframe(result)
        assert_frame_equal(result, expected)


class TestUtilityFunctions:
    """Test infer_freq, json_normalize, array functions."""

    def test_infer_freq_daily(self):
        """infer_freq should detect daily frequency."""
        idx = pd.date_range('2024-01-01', periods=5, freq='D')
        result = ds.infer_freq(idx)
        expected = pd.infer_freq(idx)
        assert result == expected

    def test_infer_freq_hourly(self):
        """infer_freq should detect hourly frequency."""
        idx = pd.date_range('2024-01-01', periods=5, freq='h')
        result = ds.infer_freq(idx)
        expected = pd.infer_freq(idx)
        assert result == expected

    def test_json_normalize_basic(self):
        """json_normalize should flatten JSON."""
        data = [
            {'id': 1, 'name': {'first': 'Alice', 'last': 'Smith'}},
            {'id': 2, 'name': {'first': 'Bob', 'last': 'Jones'}},
        ]
        result = ds.json_normalize(data)
        expected = pd.json_normalize(data)
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_array_basic(self):
        """array should create ExtensionArray."""
        result = ds.array([1, 2, 3])
        expected = pd.array([1, 2, 3])
        pd.testing.assert_extension_array_equal(result, expected)


class TestConfigurationFunctions:
    """Test set_option, get_option, reset_option, describe_option, option_context."""

    def test_get_option(self):
        """get_option should return option value."""
        result = ds.get_option('display.max_rows')
        expected = pd.get_option('display.max_rows')
        assert result == expected

    def test_set_and_reset_option(self):
        """set_option and reset_option should work correctly."""
        original = ds.get_option('display.max_rows')
        ds.set_option('display.max_rows', 100)
        assert ds.get_option('display.max_rows') == 100
        ds.reset_option('display.max_rows')
        assert ds.get_option('display.max_rows') == original

    def test_option_context(self):
        """option_context should temporarily change options."""
        original = ds.get_option('display.max_rows')
        with ds.option_context('display.max_rows', 5):
            assert ds.get_option('display.max_rows') == 5
        assert ds.get_option('display.max_rows') == original

    def test_describe_option(self):
        """describe_option should return description."""
        result = ds.describe_option('display.max_rows', _print_desc=False)
        expected = pd.describe_option('display.max_rows', _print_desc=False)
        assert result == expected


class TestIOFunctions:
    """Test read_* functions with simple data."""

    def test_read_fwf(self, tmp_path):
        """read_fwf should read fixed-width file."""
        content = "name      age\nAlice     30 \nBob       25 \n"
        file_path = tmp_path / "data.txt"
        file_path.write_text(content)

        result = ds.read_fwf(file_path, widths=[10, 3])
        expected = pd.read_fwf(file_path, widths=[10, 3])
        result = get_dataframe(result)
        assert_frame_equal(result, expected)

    def test_read_xml(self, tmp_path):
        """read_xml should read XML file."""
        content = """<?xml version='1.0'?>
<data>
  <row><name>Alice</name><age>30</age></row>
  <row><name>Bob</name><age>25</age></row>
</data>"""
        file_path = tmp_path / "data.xml"
        file_path.write_text(content)

        try:
            result = ds.read_xml(file_path, parser='etree')
            expected = pd.read_xml(file_path, parser='etree')
            result = get_dataframe(result)
            assert_frame_equal(result, expected)
        except ImportError:
            pytest.skip("XML parser not available")


class TestDataFrameSeriesCreation:
    """Test DataFrame and Series creation functions."""

    def test_dataframe_from_dict(self):
        """DataFrame should create DataStore from dict."""
        df = ds.DataStore.from_df(pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}))
        assert hasattr(df, 'to_df')
        result = df.to_df()
        assert list(result.columns) == ['A', 'B']
        assert len(result) == 3

    def test_dataframe_from_list(self):
        """DataFrame should create DataStore from list of lists."""
        df = ds.DataStore.from_df(pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A', 'B']))
        assert hasattr(df, 'to_df')
        result = df.to_df()
        assert list(result.columns) == ['A', 'B']
        assert len(result) == 3

    def test_dataframe_with_index(self):
        """DataFrame should support custom index."""
        df = ds.DataStore.from_df(pd.DataFrame({'A': [1, 2]}, index=['x', 'y']))
        result = df.to_df()
        assert list(result.index) == ['x', 'y']

    def test_series_from_list(self):
        """Series should create pandas Series from list."""
        s = ds.Series([1, 2, 3], name='values')
        assert isinstance(s, pd.Series)
        assert s.name == 'values'
        assert len(s) == 3

    def test_series_from_dict(self):
        """Series should create pandas Series from dict."""
        s = ds.Series({'a': 1, 'b': 2, 'c': 3})
        assert isinstance(s, pd.Series)
        assert s['a'] == 1
        assert s['b'] == 2

    def test_dataframe_matches_pandas(self):
        """DataFrame output should match pandas DataFrame."""
        data = {'A': [1, 2, 3], 'B': ['x', 'y', 'z']}
        ds_df = ds.DataStore.from_df(pd.DataFrame(data))
        pd_df = pd.DataFrame(data)
        np.testing.assert_array_equal(ds_df, pd_df)


class TestDataStoreIntegration:
    """Test that module-level functions work with DataStore objects."""

    def test_merge_with_datastore(self):
        """merge should work with DataStore objects."""
        left = ds.DataStore.from_df(pd.DataFrame({'key': ['a', 'b'], 'val': [1, 2]}))
        right = ds.DataStore.from_df(pd.DataFrame({'key': ['a', 'b'], 'other': [3, 4]}))
        result = ds.merge(left, right, on='key')
        assert hasattr(result, 'to_df')
        assert list(result.to_df().columns) == ['key', 'val', 'other']

    def test_concat_with_datastore(self):
        """concat should work with DataStore objects."""
        ds1 = ds.DataStore.from_df(pd.DataFrame({'A': [1, 2]}))
        ds2 = ds.DataStore.from_df(pd.DataFrame({'A': [3, 4]}))
        result = ds.concat([ds1, ds2], ignore_index=True)
        assert hasattr(result, 'to_df')
        assert len(result.to_df()) == 4

    def test_melt_with_datastore(self):
        """melt should work with DataStore objects."""
        df = ds.DataStore.from_df(pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}))
        result = ds.melt(df, id_vars=['A'], value_vars=['B', 'C'])
        assert hasattr(result, 'to_df')
        assert 'variable' in result.to_df().columns
        assert 'value' in result.to_df().columns

    def test_pivot_table_with_datastore(self):
        """pivot_table should work with DataStore objects."""
        df = ds.DataStore.from_df(
            pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'], 'B': ['one', 'two', 'one', 'two'], 'C': [1, 2, 3, 4]})
        )
        result = ds.pivot_table(df, values='C', index='A', columns='B', aggfunc='sum')
        assert hasattr(result, 'to_df')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
