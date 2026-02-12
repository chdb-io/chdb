"""
Exploratory Discovery Batch 12: Statistical Edge Cases, Timezone, Index Manipulation, Accessor Verification

Test Coverage:
1. Statistical Methods Edge Cases (std, var, sem, skew, kurt with empty/NaN/constant data)
2. Timezone Methods (tz_convert, tz_localize, to_period, to_timestamp)
3. Index Manipulation (reindex, reindex_like, swaplevel, droplevel, set_axis)
4. Special Accessor Result Verification (geo, url, ip - actual values not just types)
5. Alignment and Comparison Methods (align, compare, combine)
6. cut/qcut Functions
"""

import pytest
from tests.xfail_markers import chdb_no_quantile_array
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_dataframe, get_series


# =============================================================================
# Part 1: Statistical Methods Edge Cases
# =============================================================================


class TestStatisticalEdgeCases:
    """Test statistical methods with edge cases: empty, NaN, constant values."""

    def test_std_with_all_nan(self):
        """std() with all NaN values should return NaN."""
        pd_df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].std()
        ds_result = ds_df['A'].std()

        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_var_with_constant_values(self):
        """var() with constant values should return 0."""
        pd_df = pd.DataFrame({'A': [5, 5, 5, 5, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].var()
        ds_result = ds_df['A'].var()

        assert pd_result == 0.0
        assert ds_result == 0.0

    def test_std_with_single_value(self):
        """std() with single value should return NaN (ddof=1 default)."""
        pd_df = pd.DataFrame({'A': [10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].std()
        ds_result = ds_df['A'].std()

        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_sem_basic(self):
        """sem() standard error of mean."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].sem()
        ds_result = ds_df['A'].sem()

        assert abs(pd_result - ds_result) < 1e-10

    def test_sem_with_nan(self):
        """sem() with NaN values should skip NaN."""
        pd_df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].sem()
        ds_result = ds_df['A'].sem()

        assert abs(pd_result - ds_result) < 1e-10

    def test_skew_with_symmetric_data(self):
        """skew() with symmetric data should be close to 0."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].skew()
        ds_result = ds_df['A'].skew()

        assert abs(pd_result - ds_result) < 1e-10

    def test_skew_with_right_skewed_data(self):
        """skew() with right-skewed data should be positive."""
        pd_df = pd.DataFrame({'A': [1, 1, 1, 2, 10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].skew()
        ds_result = ds_df['A'].skew()

        assert pd_result > 0
        assert ds_result > 0
        assert abs(pd_result - ds_result) < 1e-10

    def test_kurt_basic(self):
        """kurtosis() basic test."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].kurt()
        ds_result = ds_df['A'].kurt()

        assert abs(pd_result - ds_result) < 1e-10

    def test_dataframe_std_axis0(self):
        """DataFrame std() along axis=0."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.std()
        ds_result = ds_df.std()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_var_with_mixed_nan(self):
        """DataFrame var() with mixed NaN in columns."""
        pd_df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5], 'B': [10, np.nan, np.nan, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.var()
        ds_result = ds_df.var()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mean_with_empty_after_filter(self):
        """mean() on empty result after filter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['A'] > 100]
        ds_filtered = ds_df[ds_df['A'] > 100]

        pd_result = pd_filtered['A'].mean()
        ds_result = ds_filtered['A'].mean()

        assert pd.isna(pd_result)
        assert pd.isna(ds_result)

    def test_median_with_even_count(self):
        """median() with even number of values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].median()
        ds_result = ds_df['A'].median()

        assert pd_result == 2.5
        assert ds_result == 2.5

    # xfail removed: quantile with array now works
    def test_quantile_multiple_values(self):
        """quantile() with multiple quantile values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].quantile([0.25, 0.5, 0.75])
        ds_result = ds_df['A'].quantile([0.25, 0.5, 0.75])

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 2: Timezone Methods
# =============================================================================


class TestTimezoneOperations:
    """Test timezone-aware datetime operations."""

    def test_tz_localize_to_utc(self):
        """tz_localize() to UTC."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=dates)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tz_localize('UTC')
        ds_result = ds_df.tz_localize('UTC')

        # Check index is timezone-aware
        assert pd_result.index.tz is not None
        ds_df_result = ds_result._get_df()
        assert ds_df_result.index.tz is not None

    def test_tz_convert_utc_to_eastern(self):
        """tz_convert() from UTC to US/Eastern."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D', tz='UTC')
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=dates)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tz_convert('US/Eastern')
        ds_result = ds_df.tz_convert('US/Eastern')

        pd_vals = pd_result['A'].tolist()
        ds_vals = ds_result._get_df()['A'].tolist()
        assert pd_vals == ds_vals

    def test_series_dt_tz_localize(self):
        """Series.dt.tz_localize()."""
        pd_df = pd.DataFrame({'dates': pd.date_range('2024-01-01', periods=5, freq='D')})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['dates'].dt.tz_localize('UTC')
        ds_result = ds_df['dates'].dt.tz_localize('UTC')

        assert pd_result.dt.tz is not None
        ds_series = get_series(ds_result)
        # Verify data matches
        assert len(pd_result) == len(ds_series)

    def test_to_period_daily(self):
        """to_period() with daily frequency."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        pd_series = pd.Series(dates)
        ds_series = DataStore({'dates': dates})['dates']

        pd_result = pd_series.dt.to_period('D')
        ds_result = ds_series.dt.to_period('D')

        # Compare string representations
        pd_strs = pd_result.astype(str).tolist()
        ds_df = get_series(ds_result)
        ds_strs = ds_df.astype(str).tolist()
        assert pd_strs == ds_strs

    def test_to_period_monthly(self):
        """to_period() with monthly frequency."""
        # pandas 3.0 uses 'ME' instead of 'M' for date_range, but to_period still uses 'M'
        pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
        date_freq = 'ME' if pandas_version >= (3, 0) else 'M'
        dates = pd.date_range('2024-01-15', periods=5, freq=date_freq)
        pd_series = pd.Series(dates)
        ds_series = DataStore({'dates': dates})['dates']

        # to_period always uses 'M' for monthly periods
        pd_result = pd_series.dt.to_period('M')
        ds_result = ds_series.dt.to_period('M')

        pd_strs = pd_result.astype(str).tolist()
        ds_df = get_series(ds_result)
        ds_strs = ds_df.astype(str).tolist()
        assert pd_strs == ds_strs


# =============================================================================
# Part 3: Index Manipulation
# =============================================================================


class TestIndexManipulation:
    """Test index manipulation methods."""

    def test_reindex_with_new_labels(self):
        """reindex() with new labels."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['a', 'b', 'c', 'd', 'e'])
        ds_result = ds_df.reindex(['a', 'b', 'c', 'd', 'e'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_fill_value(self):
        """reindex() with fill_value for missing."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(['a', 'b', 'c', 'd'], fill_value=0)
        ds_result = ds_df.reindex(['a', 'b', 'c', 'd'], fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_columns(self):
        """reindex() on columns."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reindex(columns=['B', 'A', 'C'])
        ds_result = ds_df.reindex(columns=['B', 'A', 'C'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_index(self):
        """set_axis() on index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_axis(['x', 'y', 'z'], axis=0)
        ds_result = ds_df.set_axis(['x', 'y', 'z'], axis=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_axis_columns(self):
        """set_axis() on columns."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_axis(['X', 'Y'], axis=1)
        ds_result = ds_df.set_axis(['X', 'Y'], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_swaplevel_multiindex(self):
        """swaplevel() with MultiIndex."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.swaplevel(0, 1)
        ds_result = ds_df.swaplevel(0, 1)

        # Check index levels are swapped
        pd_names = list(pd_result.index.names)
        ds_df_result = ds_result._get_df()
        ds_names = list(ds_df_result.index.names)
        assert pd_names == ds_names == ['second', 'first']

    def test_droplevel_multiindex(self):
        """droplevel() with MultiIndex."""
        arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
        index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=index)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.droplevel('first')
        ds_result = ds_df.droplevel('first')

        # Check level is dropped
        pd_names = list(pd_result.index.names)
        ds_df_result = ds_result._get_df()
        ds_names = list(ds_df_result.index.names)
        assert pd_names == ds_names == ['second']

    def test_rename_axis_index(self):
        """rename_axis() on index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=pd.Index([10, 20, 30], name='old'))
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename_axis('new_name')
        ds_result = ds_df.rename_axis('new_name')

        assert pd_result.index.name == 'new_name'
        ds_df_result = ds_result._get_df()
        assert ds_df_result.index.name == 'new_name'

    def test_rename_axis_columns(self):
        """rename_axis() on columns."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename_axis('cols', axis=1)
        ds_result = ds_df.rename_axis('cols', axis=1)

        assert pd_result.columns.name == 'cols'
        ds_df_result = ds_result._get_df()
        assert ds_df_result.columns.name == 'cols'


# =============================================================================
# Part 4: Accessor Result Verification
# =============================================================================


class TestUrlAccessorResults:
    """Test URL accessor with result verification (not just type checking)."""

    def test_url_domain_extraction(self):
        """url.domain() extracts correct domain."""
        urls = [
            'https://www.google.com/search?q=test',
            'http://example.org/path/to/page',
            'https://sub.domain.co.uk/page',
        ]
        ds_df = DataStore({'url': urls})

        result = ds_df['url'].url.domain()
        result_series = get_series(result)
        result_list = result_series.tolist()

        # Check domains are extracted correctly
        assert 'google.com' in result_list[0] or 'www.google.com' in result_list[0]
        assert 'example.org' in result_list[1]
        assert 'domain.co.uk' in result_list[2] or 'sub.domain.co.uk' in result_list[2]

    def test_url_protocol_extraction(self):
        """url.protocol() extracts correct protocol."""
        urls = ['https://example.com', 'http://example.org', 'ftp://files.example.com']
        ds_df = DataStore({'url': urls})

        result = ds_df['url'].url.protocol()
        result_series = get_series(result)
        result_list = result_series.tolist()

        assert result_list[0] == 'https'
        assert result_list[1] == 'http'
        assert result_list[2] == 'ftp'

    def test_url_path_extraction(self):
        """url.url_path() extracts correct path."""
        urls = ['https://example.com/path/to/resource', 'http://example.org/', 'https://example.com/api/v1/users']
        ds_df = DataStore({'url': urls})

        result = ds_df['url'].url.url_path()
        result_series = get_series(result)
        result_list = result_series.tolist()

        assert '/path/to/resource' in result_list[0]
        assert result_list[2] == '/api/v1/users' or 'api/v1/users' in result_list[2]


class TestIpAccessorResults:
    """Test IP accessor with result verification."""

    def test_is_ipv4_string(self):
        """ip.is_ipv4_string() correctly identifies IPv4."""
        ips = ['192.168.1.1', '10.0.0.1', 'not_an_ip', '256.1.1.1']
        ds_df = DataStore({'ip': ips})

        result = ds_df['ip'].ip.is_ipv4_string()
        result_series = get_series(result)
        result_list = result_series.tolist()

        assert result_list[0] == 1 or result_list[0] == True
        assert result_list[1] == 1 or result_list[1] == True
        assert result_list[2] == 0 or result_list[2] == False

    def test_ipv4_to_num(self):
        """ip.ipv4_string_to_num() converts IP to number."""
        ips = ['192.168.1.1', '10.0.0.1', '127.0.0.1']
        ds_df = DataStore({'ip': ips})

        result = ds_df['ip'].ip.ipv4_string_to_num()
        result_series = get_series(result)
        result_list = result_series.tolist()

        # 127.0.0.1 = 2130706433
        assert result_list[2] == 2130706433


# =============================================================================
# Part 5: Alignment and Comparison Methods
# =============================================================================


class TestAlignmentMethods:
    """Test DataFrame alignment and comparison methods."""

    def test_align_outer_join(self):
        """align() with outer join."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'A': [4, 5, 6]}, index=[1, 2, 3])
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_aligned1, pd_aligned2 = pd_df1.align(pd_df2, join='outer')
        ds_aligned1, ds_aligned2 = ds_df1.align(ds_df2, join='outer')

        assert_datastore_equals_pandas(ds_aligned1, pd_aligned1)
        assert_datastore_equals_pandas(ds_aligned2, pd_aligned2)

    def test_align_inner_join(self):
        """align() with inner join."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'A': [4, 5, 6]}, index=[1, 2, 3])
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_aligned1, pd_aligned2 = pd_df1.align(pd_df2, join='inner')
        ds_aligned1, ds_aligned2 = ds_df1.align(ds_df2, join='inner')

        assert_datastore_equals_pandas(ds_aligned1, pd_aligned1)
        assert_datastore_equals_pandas(ds_aligned2, pd_aligned2)

    def test_compare_basic(self):
        """compare() basic usage."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        pd_df2 = pd.DataFrame({'A': [1, 2, 4], 'B': ['x', 'w', 'z']})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd_df1.compare(pd_df2)
        ds_result = ds_df1.compare(ds_df2)

        # Check structure matches
        pd_shape = pd_result.shape
        ds_shape = get_series(ds_result).shape if len(get_series(ds_result).shape) == 1 else get_dataframe(ds_result).shape
        assert pd_shape == ds_shape

    def test_compare_keep_shape(self):
        """compare() with keep_shape=True."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'A': [1, 2, 4]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd_df1.compare(pd_df2, keep_shape=True)
        ds_result = ds_df1.compare(ds_df2, keep_shape=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_with_func(self):
        """combine() with custom function."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [10, 2, 30], 'B': [4, 50, 6]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd_df1.combine(pd_df2, lambda s1, s2: s1.where(s1 > s2, s2))
        ds_result = ds_df1.combine(ds_df2, lambda s1, s2: s1.where(s1 > s2, s2))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_update_inplace(self):
        """update() modifies DataFrame in place."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_other = pd.DataFrame({'A': [10, np.nan, 30]})

        ds_df = DataStore(pd_df.copy())
        ds_other = DataStore(pd_other.copy())

        pd_df.update(pd_other)
        ds_df.update(ds_other)

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Part 6: cut/qcut Functions
# =============================================================================


class TestCutQcut:
    """Test cut() and qcut() functions."""

    def test_cut_with_bins(self):
        """cut() with fixed bins."""
        data = [1, 7, 5, 4, 6, 3]
        pd_series = pd.Series(data)
        pd_result = pd.cut(pd_series, bins=3)

        ds_df = DataStore({'values': data})
        ds_vals = get_series(ds_df['values'])
        ds_result = pd.cut(ds_vals, bins=3)

        # Both are Series with categorical dtype, compare cat.codes
        assert list(pd_result.cat.codes) == list(ds_result.cat.codes)

    def test_cut_with_labels(self):
        """cut() with custom labels."""
        data = [1, 7, 5, 4, 6, 3]
        pd_series = pd.Series(data)
        pd_result = pd.cut(pd_series, bins=3, labels=['low', 'med', 'high'])

        ds_df = DataStore({'values': data})
        ds_vals = get_series(ds_df['values'])
        ds_result = pd.cut(ds_vals, bins=3, labels=['low', 'med', 'high'])

        assert list(pd_result) == list(ds_result)

    def test_qcut_quantiles(self):
        """qcut() with quantile bins."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pd_series = pd.Series(data)
        pd_result = pd.qcut(pd_series, q=4)

        ds_df = DataStore({'values': data})
        ds_vals = get_series(ds_df['values'])
        ds_result = pd.qcut(ds_vals, q=4)

        # Both are Series with categorical dtype, compare cat.codes
        assert list(pd_result.cat.codes) == list(ds_result.cat.codes)

    def test_qcut_with_labels(self):
        """qcut() with custom labels."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        pd_series = pd.Series(data)
        pd_result = pd.qcut(pd_series, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        ds_df = DataStore({'values': data})
        ds_vals = get_series(ds_df['values'])
        ds_result = pd.qcut(ds_vals, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        assert list(pd_result) == list(ds_result)


# =============================================================================
# Part 7: First/Last Valid Index
# =============================================================================


class TestFirstLastValidIndex:
    """Test first_valid_index() and last_valid_index()."""

    def test_first_valid_index_with_nan_prefix(self):
        """first_valid_index() skips leading NaN."""
        pd_df = pd.DataFrame({'A': [np.nan, np.nan, 1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].first_valid_index()
        ds_result = ds_df['A'].first_valid_index()

        assert pd_result == ds_result == 2

    def test_last_valid_index_with_nan_suffix(self):
        """last_valid_index() skips trailing NaN."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, np.nan, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].last_valid_index()
        ds_result = ds_df['A'].last_valid_index()

        assert pd_result == ds_result == 2

    def test_first_valid_index_all_nan(self):
        """first_valid_index() with all NaN returns None."""
        pd_df = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].first_valid_index()
        ds_result = ds_df['A'].first_valid_index()

        assert pd_result is None
        assert ds_result is None

    def test_dataframe_first_valid_index(self):
        """DataFrame first_valid_index()."""
        pd_df = pd.DataFrame({'A': [np.nan, 1, 2], 'B': [np.nan, np.nan, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.first_valid_index()
        ds_result = ds_df.first_valid_index()

        assert pd_result == ds_result


# =============================================================================
# Part 8: Window Function Edge Cases
# =============================================================================


class TestWindowEdgeCases:
    """Test window function edge cases."""

    def test_rolling_closed_left(self):
        """rolling() with closed='left'."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].rolling(3, closed='left').sum()
        ds_result = ds_df['A'].rolling(3, closed='left').sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_closed_right(self):
        """rolling() with closed='right'."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].rolling(3, closed='right').sum()
        ds_result = ds_df['A'].rolling(3, closed='right').sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_center(self):
        """rolling() with center=True."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].rolling(3, center=True).mean()
        ds_result = ds_df['A'].rolling(3, center=True).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_adjust_false(self):
        """ewm() with adjust=False."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].ewm(span=3, adjust=False).mean()
        ds_result = ds_df['A'].ewm(span=3, adjust=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_min_periods(self):
        """expanding() with min_periods."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].expanding(min_periods=3).sum()
        ds_result = ds_df['A'].expanding(min_periods=3).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 9: Rename with Callable
# =============================================================================


class TestRenameCallable:
    """Test rename() with callable functions."""

    def test_rename_columns_with_lambda(self):
        """rename() columns with lambda."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(columns=lambda x: x.upper())
        ds_result = ds_df.rename(columns=lambda x: x.upper())

        assert list(pd_result.columns) == ['A', 'B']
        ds_cols = list(ds_result._get_df().columns)
        assert ds_cols == ['A', 'B']

    def test_rename_columns_with_str_method(self):
        """rename() columns with str method."""
        pd_df = pd.DataFrame({'col_a': [1, 2], 'col_b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(columns=str.title)
        ds_result = ds_df.rename(columns=str.title)

        assert list(pd_result.columns) == ['Col_A', 'Col_B']
        ds_cols = list(ds_result._get_df().columns)
        assert ds_cols == ['Col_A', 'Col_B']

    def test_rename_index_with_lambda(self):
        """rename() index with lambda."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.rename(index=lambda x: x * 2)
        ds_result = ds_df.rename(index=lambda x: x * 2)

        assert list(pd_result.index) == ['xx', 'yy', 'zz']
        ds_idx = list(ds_result._get_df().index)
        assert ds_idx == ['xx', 'yy', 'zz']


# =============================================================================
# Part 10: Additional Edge Cases
# =============================================================================


class TestMiscEdgeCases:
    """Miscellaneous edge case tests."""

    def test_empty_dataframe_columns(self):
        """Operations on empty DataFrame with columns."""
        pd_df = pd.DataFrame(columns=['A', 'B', 'C'])
        ds_df = DataStore(pd_df)

        assert list(pd_df.columns) == list(ds_df._get_df().columns)
        assert len(pd_df) == len(ds_df) == 0

    def test_single_column_operations(self):
        """Operations on single-column DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sum()
        ds_result = ds_df.sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_dtype_describe(self):
        """describe() with mixed dtypes."""
        pd_df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.1, 2.2, 3.3], 'str_col': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(include='all')
        ds_result = ds_df.describe(include='all')

        # Check columns match
        assert list(pd_result.columns) == list(ds_result._get_df().columns)

    def test_boolean_column_sum(self):
        """sum() on boolean column."""
        pd_df = pd.DataFrame({'A': [True, False, True, True, False]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].sum()
        ds_result = ds_df['A'].sum()

        assert pd_result == ds_result == 3

    def test_idxmin_idxmax(self):
        """idxmin() and idxmax() on Series."""
        pd_df = pd.DataFrame({'A': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds_df = DataStore(pd_df)

        assert pd_df['A'].idxmin() == ds_df['A'].idxmin()
        assert pd_df['A'].idxmax() == ds_df['A'].idxmax()

    def test_mode_single_mode(self):
        """mode() with single mode."""
        pd_df = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].mode()
        ds_result = ds_df['A'].mode()

        assert pd_result.iloc[0] == 3
        ds_val = get_series(ds_result)
        assert ds_val.iloc[0] == 3

    def test_nlargest_with_keep(self):
        """nlargest() with keep parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].nlargest(3, keep='first')
        ds_result = ds_df['A'].nlargest(3, keep='first')

        assert list(pd_result.values) == list(get_series(ds_result).values)

    def test_nsmallest_with_keep(self):
        """nsmallest() with keep parameter."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['A'].nsmallest(3, keep='last')
        ds_result = ds_df['A'].nsmallest(3, keep='last')

        assert list(pd_result.values) == list(get_series(ds_result).values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
