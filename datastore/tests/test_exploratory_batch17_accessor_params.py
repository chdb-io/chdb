"""
Exploratory Batch 17: URL/IP/Geo Accessors and Parameter Edge Cases

Test Coverage:
1. URL Accessor Methods - domain, path, protocol, query_string, fragment
2. IP Accessor Methods - to_ipv4, to_ipv6, is_ipv4_string, is_ipv6_string
3. Geo Accessor Methods - Basic geo functions where applicable
4. fillna() Method Parameter Combinations
5. join() Parameter Edge Cases
6. dropna() Parameter Variations
7. sort_values() Edge Cases
8. value_counts() Parameter Combinations

NOTE: Some tests may use xfail markers for known chDB limitations.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from datastore.column_expr import ColumnExpr
from tests.test_utils import get_series, get_dataframe, assert_frame_equal, assert_datastore_equals_pandas


# =============================================================================
# URL ACCESSOR TESTS
# =============================================================================


class TestUrlAccessor:
    """Test .url accessor methods for URL parsing."""

    @pytest.fixture
    def ds_with_urls(self):
        """Create DataStore with URL data."""
        data = {
            'url': [
                'https://www.example.com/path/to/page?query=value#section',
                'http://subdomain.test.org:8080/api/v1',
                'ftp://files.server.net/downloads/file.zip',
                'https://example.com',
                '',  # empty URL
            ]
        }
        return DataStore(data)

    def test_url_domain_returns_column_expr(self, ds_with_urls):
        """Test that .url.domain returns ColumnExpr."""
        result = ds_with_urls['url'].url.domain()
        assert isinstance(result, ColumnExpr)

    def test_url_domain_execution(self, ds_with_urls):
        """Test domain extraction."""
        ds_with_urls['extracted_domain'] = ds_with_urls['url'].url.domain()

        assert 'extracted_domain' in ds_with_urls.columns
        # Should extract domain from URL
        domains = list(ds_with_urls['extracted_domain'])
        assert 'www.example.com' in domains[0] or 'example.com' in domains[0]

    def test_url_domain_without_www(self, ds_with_urls):
        """Test domain extraction without www prefix."""
        ds_with_urls['domain_no_www'] = ds_with_urls['url'].url.domain_without_www()

        assert 'domain_no_www' in ds_with_urls.columns
        # First URL should have www stripped
        assert ds_with_urls['domain_no_www'].iloc[0] == 'example.com'

    def test_url_protocol(self, ds_with_urls):
        """Test protocol/scheme extraction."""
        ds_with_urls['proto'] = ds_with_urls['url'].url.protocol()

        protocols = list(ds_with_urls['proto'])
        assert protocols[0] == 'https'
        assert protocols[1] == 'http'
        assert protocols[2] == 'ftp'

    def test_url_path(self, ds_with_urls):
        """Test path extraction."""
        ds_with_urls['url_path'] = ds_with_urls['url'].url.url_path()

        paths = list(ds_with_urls['url_path'])
        assert '/path/to/page' in paths[0]
        assert '/api/v1' in paths[1]

    def test_url_query_string(self, ds_with_urls):
        """Test query string extraction."""
        ds_with_urls['qs'] = ds_with_urls['url'].url.query_string()

        # First URL has query=value
        assert 'query=value' in ds_with_urls['qs'].iloc[0]

    def test_url_fragment(self, ds_with_urls):
        """Test fragment/anchor extraction."""
        ds_with_urls['frag'] = ds_with_urls['url'].url.fragment()

        # First URL has #section
        assert ds_with_urls['frag'].iloc[0] == 'section'

    def test_url_top_level_domain(self, ds_with_urls):
        """Test top-level domain extraction."""
        ds_with_urls['tld'] = ds_with_urls['url'].url.top_level_domain()

        tlds = list(ds_with_urls['tld'])
        assert tlds[0] == 'com'
        assert tlds[1] == 'org'
        assert tlds[2] == 'net'

    def test_url_empty_string_handling(self, ds_with_urls):
        """Test URL methods on empty strings."""
        ds_with_urls['domain'] = ds_with_urls['url'].url.domain()

        # Empty URL should return empty domain
        assert ds_with_urls['domain'].iloc[4] == ''


# =============================================================================
# IP ACCESSOR TESTS
# =============================================================================


class TestIpAccessor:
    """Test .ip accessor methods for IP address operations."""

    @pytest.fixture
    def ds_with_ips(self):
        """Create DataStore with IP address data."""
        data = {
            'ip_str': [
                '192.168.1.1',
                '10.0.0.1',
                '8.8.8.8',
                '255.255.255.0',
                '0.0.0.0',
            ],
            'ipv6_str': [
                '::1',
                '2001:db8::1',
                'fe80::1',
                '::ffff:192.168.1.1',
                '::',
            ],
        }
        return DataStore(data)

    def test_ip_is_ipv4_string_returns_column_expr(self, ds_with_ips):
        """Test that .ip.is_ipv4_string returns ColumnExpr."""
        result = ds_with_ips['ip_str'].ip.is_ipv4_string()
        assert isinstance(result, ColumnExpr)

    def test_ip_is_ipv4_string_execution(self, ds_with_ips):
        """Test IPv4 string validation."""
        ds_with_ips['is_ipv4'] = ds_with_ips['ip_str'].ip.is_ipv4_string()

        # All ip_str values should be valid IPv4
        assert all(ds_with_ips['is_ipv4'] == 1)

    def test_ip_is_ipv6_string_execution(self, ds_with_ips):
        """Test IPv6 string validation."""
        ds_with_ips['is_ipv6'] = ds_with_ips['ipv6_str'].ip.is_ipv6_string()

        # All ipv6_str values should be valid IPv6
        assert all(ds_with_ips['is_ipv6'] == 1)

    def test_ip_is_ipv4_on_ipv6_false(self, ds_with_ips):
        """Test that IPv6 strings are not valid IPv4."""
        ds_with_ips['ipv6_as_ipv4'] = ds_with_ips['ipv6_str'].ip.is_ipv4_string()

        # IPv6 strings should not be valid IPv4
        assert all(ds_with_ips['ipv6_as_ipv4'] == 0)

    def test_ip_to_ipv4(self, ds_with_ips):
        """Test string to IPv4 conversion."""
        ds_with_ips['ipv4_type'] = ds_with_ips['ip_str'].ip.to_ipv4()

        # Should successfully convert
        assert 'ipv4_type' in ds_with_ips.columns

    def test_ip_to_ipv6(self, ds_with_ips):
        """Test string to IPv6 conversion."""
        ds_with_ips['ipv6_type'] = ds_with_ips['ipv6_str'].ip.to_ipv6()

        # Should successfully convert
        assert 'ipv6_type' in ds_with_ips.columns

    def test_ip_ipv4_string_to_num(self, ds_with_ips):
        """Test IPv4 string to numeric conversion."""
        ds_with_ips['ip_num'] = ds_with_ips['ip_str'].ip.ipv4_string_to_num()

        # 192.168.1.1 = 192*256^3 + 168*256^2 + 1*256 + 1
        expected_first = 192 * (256**3) + 168 * (256**2) + 1 * 256 + 1
        assert ds_with_ips['ip_num'].iloc[0] == expected_first


# =============================================================================
# FILLNA PARAMETER COMBINATIONS
# =============================================================================


class TestFillnaParameters:
    """Test fillna() method with various parameter combinations."""

    @pytest.fixture
    def ds_with_nulls(self):
        """Create DataStore with NULL values."""
        data = {
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [np.nan, 2.0, np.nan, 4.0, np.nan],
            'c': ['x', None, 'z', None, 'w'],
        }
        return DataStore(data)

    def test_fillna_scalar_value(self, ds_with_nulls):
        """Test fillna with scalar value."""
        pd_df = pd.DataFrame(
            {
                'a': [1.0, np.nan, 3.0, np.nan, 5.0],
                'b': [np.nan, 2.0, np.nan, 4.0, np.nan],
            }
        )

        ds_result = ds_with_nulls[['a', 'b']].fillna(0)
        pd_result = pd_df.fillna(0)

        assert_frame_equal(get_dataframe(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_fillna_dict_value(self, ds_with_nulls):
        """Test fillna with dict specifying values per column."""
        pd_df = pd.DataFrame(
            {
                'a': [1.0, np.nan, 3.0, np.nan, 5.0],
                'b': [np.nan, 2.0, np.nan, 4.0, np.nan],
            }
        )

        ds_result = ds_with_nulls[['a', 'b']].fillna({'a': 100, 'b': 200})
        pd_result = pd_df.fillna({'a': 100, 'b': 200})

        assert_frame_equal(get_dataframe(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_fillna_column_specific(self, ds_with_nulls):
        """Test fillna on a specific column."""
        pd_df = pd.DataFrame(
            {
                'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )

        ds_col = ds_with_nulls['a'].fillna(-1)
        pd_col = pd_df['a'].fillna(-1)

        # Compare values
        np.testing.assert_array_equal(get_series(ds_col).values, pd_col.values)

    def test_fillna_string_column(self, ds_with_nulls):
        """Test fillna on string column."""
        ds_with_nulls['c_filled'] = ds_with_nulls['c'].fillna('MISSING')

        # Check that nulls are replaced
        assert 'MISSING' in list(ds_with_nulls['c_filled'])

    def test_fillna_preserves_non_null(self, ds_with_nulls):
        """Test that fillna preserves non-null values."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})

        ds_result = ds_with_nulls['a'].fillna(999)
        pd_result = pd_df['a'].fillna(999)

        np.testing.assert_array_equal(get_series(ds_result).values, pd_result.values)


# =============================================================================
# JOIN PARAMETER EDGE CASES
# =============================================================================


class TestJoinParameters:
    """Test join() method with various parameter combinations."""

    @pytest.fixture
    def ds_left(self):
        """Left DataFrame for join tests."""
        return DataStore(
            {
                'key': ['a', 'b', 'c', 'd'],
                'val_left': [1, 2, 3, 4],
            }
        )

    @pytest.fixture
    def ds_right(self):
        """Right DataFrame for join tests."""
        return DataStore(
            {
                'key': ['b', 'c', 'd', 'e'],
                'val_right': [20, 30, 40, 50],
            }
        )

    @pytest.fixture
    def pd_left(self):
        """Pandas left DataFrame."""
        return pd.DataFrame(
            {
                'key': ['a', 'b', 'c', 'd'],
                'val_left': [1, 2, 3, 4],
            }
        )

    @pytest.fixture
    def pd_right(self):
        """Pandas right DataFrame."""
        return pd.DataFrame(
            {
                'key': ['b', 'c', 'd', 'e'],
                'val_right': [20, 30, 40, 50],
            }
        )

    def test_join_inner(self, ds_left, ds_right, pd_left, pd_right):
        """Test inner join."""
        ds_result = ds_left.merge(ds_right, on='key', how='inner')
        pd_result = pd_left.merge(pd_right, on='key', how='inner')

        assert len(ds_result) == len(pd_result)
        assert set(ds_result['key']) == set(pd_result['key'])

    def test_join_left(self, ds_left, ds_right, pd_left, pd_right):
        """Test left join."""
        ds_result = ds_left.merge(ds_right, on='key', how='left')
        pd_result = pd_left.merge(pd_right, on='key', how='left')

        assert len(ds_result) == len(pd_result)

    def test_join_right(self, ds_left, ds_right, pd_left, pd_right):
        """Test right join."""
        ds_result = ds_left.merge(ds_right, on='key', how='right')
        pd_result = pd_left.merge(pd_right, on='key', how='right')

        assert len(ds_result) == len(pd_result)

    def test_join_outer(self, ds_left, ds_right, pd_left, pd_right):
        """Test outer join."""
        ds_result = ds_left.merge(ds_right, on='key', how='outer')
        pd_result = pd_left.merge(pd_right, on='key', how='outer')

        assert len(ds_result) == len(pd_result)

    def test_join_left_on_right_on(self):
        """Test join with different column names."""
        ds_left = DataStore(
            {
                'left_key': ['a', 'b', 'c'],
                'val1': [1, 2, 3],
            }
        )
        ds_right = DataStore(
            {
                'right_key': ['b', 'c', 'd'],
                'val2': [20, 30, 40],
            }
        )

        pd_left = pd.DataFrame(
            {
                'left_key': ['a', 'b', 'c'],
                'val1': [1, 2, 3],
            }
        )
        pd_right = pd.DataFrame(
            {
                'right_key': ['b', 'c', 'd'],
                'val2': [20, 30, 40],
            }
        )

        ds_result = ds_left.merge(ds_right, left_on='left_key', right_on='right_key', how='inner')
        pd_result = pd_left.merge(pd_right, left_on='left_key', right_on='right_key', how='inner')

        assert len(ds_result) == len(pd_result)

    def test_join_empty_result(self):
        """Test join that produces empty result."""
        ds_left = DataStore(
            {
                'key': ['a', 'b'],
                'val': [1, 2],
            }
        )
        ds_right = DataStore(
            {
                'key': ['x', 'y'],
                'val2': [10, 20],
            }
        )

        ds_result = ds_left.merge(ds_right, on='key', how='inner')

        assert len(ds_result) == 0

    def test_join_multiple_keys(self):
        """Test join on multiple columns."""
        ds_left = DataStore(
            {
                'k1': ['a', 'a', 'b'],
                'k2': [1, 2, 1],
                'val': [10, 20, 30],
            }
        )
        ds_right = DataStore(
            {
                'k1': ['a', 'b', 'b'],
                'k2': [1, 1, 2],
                'val2': [100, 200, 300],
            }
        )

        pd_left = pd.DataFrame(
            {
                'k1': ['a', 'a', 'b'],
                'k2': [1, 2, 1],
                'val': [10, 20, 30],
            }
        )
        pd_right = pd.DataFrame(
            {
                'k1': ['a', 'b', 'b'],
                'k2': [1, 1, 2],
                'val2': [100, 200, 300],
            }
        )

        ds_result = ds_left.merge(ds_right, on=['k1', 'k2'], how='inner')
        pd_result = pd_left.merge(pd_right, on=['k1', 'k2'], how='inner')

        assert len(ds_result) == len(pd_result)


# =============================================================================
# DROPNA PARAMETER VARIATIONS
# =============================================================================


class TestDropnaParameters:
    """Test dropna() method with various parameter combinations."""

    @pytest.fixture
    def ds_with_nulls(self):
        """Create DataStore with NULL values in various positions."""
        data = {
            'a': [1.0, np.nan, 3.0, np.nan, 5.0],
            'b': [np.nan, 2.0, 3.0, 4.0, np.nan],
            'c': [1.0, 2.0, np.nan, 4.0, 5.0],
        }
        return DataStore(data)

    @pytest.fixture
    def pd_with_nulls(self):
        """Pandas DataFrame with same NULL values."""
        return pd.DataFrame(
            {
                'a': [1.0, np.nan, 3.0, np.nan, 5.0],
                'b': [np.nan, 2.0, 3.0, 4.0, np.nan],
                'c': [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )

    def test_dropna_default(self, ds_with_nulls, pd_with_nulls):
        """Test dropna with default parameters (drop any row with NaN)."""
        ds_result = ds_with_nulls.dropna()
        pd_result = pd_with_nulls.dropna()

        assert len(ds_result) == len(pd_result)

    def test_dropna_how_all(self, ds_with_nulls, pd_with_nulls):
        """Test dropna with how='all' (drop row only if all values are NaN)."""
        # Add a row with all NaN
        ds_all_nan = DataStore(
            {
                'a': [1.0, np.nan, np.nan],
                'b': [2.0, np.nan, np.nan],
                'c': [3.0, np.nan, np.nan],
            }
        )
        pd_all_nan = pd.DataFrame(
            {
                'a': [1.0, np.nan, np.nan],
                'b': [2.0, np.nan, np.nan],
                'c': [3.0, np.nan, np.nan],
            }
        )

        ds_result = ds_all_nan.dropna(how='all')
        pd_result = pd_all_nan.dropna(how='all')

        # Should keep row 0 and possibly row 1 (not all NaN in row 1 above)
        assert len(ds_result) == len(pd_result)

    def test_dropna_subset(self, ds_with_nulls, pd_with_nulls):
        """Test dropna with subset parameter (only check specific columns)."""
        ds_result = ds_with_nulls.dropna(subset=['a'])
        pd_result = pd_with_nulls.dropna(subset=['a'])

        # Should drop rows where 'a' is NaN
        assert len(ds_result) == len(pd_result)
        assert ds_result['a'].isna().sum() == 0

    def test_dropna_thresh(self, ds_with_nulls, pd_with_nulls):
        """Test dropna with thresh parameter (minimum non-NA values required)."""
        ds_result = ds_with_nulls.dropna(thresh=2)
        pd_result = pd_with_nulls.dropna(thresh=2)

        # Should keep rows with at least 2 non-NA values
        assert len(ds_result) == len(pd_result)


# =============================================================================
# SORT_VALUES EDGE CASES
# =============================================================================


class TestSortValuesEdgeCases:
    """Test sort_values() edge cases."""

    def test_sort_values_ascending_false(self):
        """Test descending sort."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds = DataStore({'a': [3, 1, 4, 1, 5, 9, 2, 6]})

        ds_result = ds.sort_values('a', ascending=False)
        pd_result = pd_df.sort_values('a', ascending=False)

        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_sort_values_with_nan(self):
        """Test sorting with NaN values."""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 1.0, np.nan, 5.0]})
        ds = DataStore({'a': [3.0, np.nan, 1.0, np.nan, 5.0]})

        ds_result = ds.sort_values('a')
        pd_result = pd_df.sort_values('a')

        # NaN should be at the end by default
        assert len(ds_result) == len(pd_result)

    def test_sort_values_na_position_first(self):
        """Test sorting with na_position='first'."""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 1.0, np.nan, 5.0]})
        ds = DataStore({'a': [3.0, np.nan, 1.0, np.nan, 5.0]})

        ds_result = ds.sort_values('a', na_position='first')
        pd_result = pd_df.sort_values('a', na_position='first')

        # NaN should be at the beginning
        assert len(ds_result) == len(pd_result)

    def test_sort_values_multiple_columns(self):
        """Test sorting by multiple columns."""
        pd_df = pd.DataFrame(
            {
                'a': [1, 1, 2, 2],
                'b': [4, 3, 2, 1],
            }
        )
        ds = DataStore(
            {
                'a': [1, 1, 2, 2],
                'b': [4, 3, 2, 1],
            }
        )

        ds_result = ds.sort_values(['a', 'b'])
        pd_result = pd_df.sort_values(['a', 'b'])

        assert_frame_equal(get_dataframe(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_sort_values_mixed_ascending(self):
        """Test sorting with mixed ascending per column."""
        pd_df = pd.DataFrame(
            {
                'a': [1, 1, 2, 2],
                'b': [4, 3, 2, 1],
            }
        )
        ds = DataStore(
            {
                'a': [1, 1, 2, 2],
                'b': [4, 3, 2, 1],
            }
        )

        ds_result = ds.sort_values(['a', 'b'], ascending=[True, False])
        pd_result = pd_df.sort_values(['a', 'b'], ascending=[True, False])

        assert_frame_equal(get_dataframe(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True))


# =============================================================================
# VALUE_COUNTS PARAMETER COMBINATIONS
# =============================================================================


class TestValueCountsParameters:
    """Test value_counts() method with various parameter combinations."""

    @pytest.fixture
    def ds_with_data(self):
        """Create DataStore with categorical-like data."""
        return DataStore(
            {
                'category': ['a', 'b', 'a', 'c', 'b', 'a', 'a', 'c', 'b', 'a'],
                'value': [1, 2, 1, 3, 2, 1, 1, 3, 2, 1],
            }
        )

    @pytest.fixture
    def pd_with_data(self):
        """Pandas DataFrame with same data."""
        return pd.DataFrame(
            {
                'category': ['a', 'b', 'a', 'c', 'b', 'a', 'a', 'c', 'b', 'a'],
                'value': [1, 2, 1, 3, 2, 1, 1, 3, 2, 1],
            }
        )

    def test_value_counts_default(self, ds_with_data, pd_with_data):
        """Test value_counts with default parameters."""
        ds_result = ds_with_data['category'].value_counts()
        pd_result = pd_with_data['category'].value_counts()

        # Compare counts (order may differ)
        assert get_series(ds_result).sum() == pd_result.sum()

    def test_value_counts_normalize(self, ds_with_data, pd_with_data):
        """Test value_counts with normalize=True."""
        ds_result = ds_with_data['category'].value_counts(normalize=True)
        pd_result = pd_with_data['category'].value_counts(normalize=True)

        # Normalized values should sum to 1
        assert abs(get_series(ds_result).sum() - 1.0) < 0.01

    def test_value_counts_ascending(self, ds_with_data, pd_with_data):
        """Test value_counts with ascending=True."""
        ds_result = ds_with_data['category'].value_counts(ascending=True)
        pd_result = pd_with_data['category'].value_counts(ascending=True)

        # Should be sorted ascending by count
        assert list(get_series(ds_result).values) == sorted(get_series(ds_result).values)

    def test_value_counts_sort_false(self, ds_with_data, pd_with_data):
        """Test value_counts with sort=False."""
        ds_result = ds_with_data['category'].value_counts(sort=False)
        pd_result = pd_with_data['category'].value_counts(sort=False)

        # Should have same total count
        assert get_series(ds_result).sum() == pd_result.sum()

    def test_value_counts_dropna_false(self):
        """Test value_counts with dropna=False."""
        ds = DataStore(
            {
                'col': ['a', 'b', None, 'a', None, 'b', 'a'],
            }
        )
        pd_df = pd.DataFrame(
            {
                'col': ['a', 'b', None, 'a', None, 'b', 'a'],
            }
        )

        ds_result = ds['col'].value_counts(dropna=False)
        pd_result = pd_df['col'].value_counts(dropna=False)

        # Should include None/NaN in counts
        assert get_series(ds_result).sum() == pd_result.sum()


# =============================================================================
# ASTYPE EDGE CASES
# =============================================================================


class TestAstypeEdgeCases:
    """Test astype() edge cases."""

    def test_astype_int_to_float(self):
        """Test converting int to float."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore({'a': [1, 2, 3]})

        ds_result = ds.astype({'a': 'float64'})
        pd_result = pd_df.astype({'a': 'float64'})

        assert ds_result['a'].dtype == pd_result['a'].dtype

    def test_astype_float_to_int(self):
        """Test converting float to int (truncation)."""
        pd_df = pd.DataFrame({'a': [1.1, 2.9, 3.5]})
        ds = DataStore({'a': [1.1, 2.9, 3.5]})

        ds_result = ds.astype({'a': 'int64'})
        pd_result = pd_df.astype({'a': 'int64'})

        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_astype_to_string(self):
        """Test converting numeric to string."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore({'a': [1, 2, 3]})

        ds_result = ds.astype({'a': 'str'})
        pd_result = pd_df.astype({'a': 'str'})

        # Should be string type
        assert ds_result['a'].dtype == object or ds_result['a'].dtype.name == 'string'

    def test_astype_string_to_int(self):
        """Test converting string to int."""
        pd_df = pd.DataFrame({'a': ['1', '2', '3']})
        ds = DataStore({'a': ['1', '2', '3']})

        ds_result = ds.astype({'a': 'int64'})
        pd_result = pd_df.astype({'a': 'int64'})

        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)


# =============================================================================
# SAMPLE PARAMETER VARIATIONS
# =============================================================================


class TestSampleParameters:
    """Test sample() method with various parameter combinations."""

    @pytest.fixture
    def ds_large(self):
        """Create a larger DataStore for sampling tests."""
        return DataStore(
            {
                'id': list(range(100)),
                'val': list(range(100)),
            }
        )

    def test_sample_n(self, ds_large):
        """Test sampling n rows."""
        ds_result = ds_large.sample(n=10, random_state=42)

        assert len(ds_result) == 10

    def test_sample_frac(self, ds_large):
        """Test sampling fraction of rows."""
        ds_result = ds_large.sample(frac=0.1, random_state=42)

        assert len(ds_result) == 10

    def test_sample_replace(self, ds_large):
        """Test sampling with replacement."""
        ds_result = ds_large.sample(n=150, replace=True, random_state=42)

        assert len(ds_result) == 150  # More than original size is possible with replacement

    def test_sample_deterministic(self, ds_large):
        """Test that same random_state gives same result."""
        df1 = ds_large.sample(n=10, random_state=42)
        df2 = ds_large.sample(n=10, random_state=42)

        # Compare as DataFrames
        assert_datastore_equals_pandas(df1, get_dataframe(df2))


# =============================================================================
# RANK PARAMETER VARIATIONS
# =============================================================================


class TestRankParameters:
    """Test rank() method with various parameter combinations."""

    @pytest.fixture
    def ds_with_ties(self):
        """Create DataStore with tied values."""
        return DataStore(
            {
                'val': [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            }
        )

    @pytest.fixture
    def pd_with_ties(self):
        """Pandas DataFrame with same data."""
        return pd.DataFrame(
            {
                'val': [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            }
        )

    def test_rank_average(self, ds_with_ties, pd_with_ties):
        """Test rank with method='average' (default)."""
        ds_result = ds_with_ties['val'].rank(method='average')
        pd_result = pd_with_ties['val'].rank(method='average')

        np.testing.assert_array_almost_equal(get_series(ds_result).values, pd_result.values)

    def test_rank_min(self, ds_with_ties, pd_with_ties):
        """Test rank with method='min'."""
        ds_result = ds_with_ties['val'].rank(method='min')
        pd_result = pd_with_ties['val'].rank(method='min')

        np.testing.assert_array_equal(get_series(ds_result).values, pd_result.values)

    def test_rank_max(self, ds_with_ties, pd_with_ties):
        """Test rank with method='max'."""
        ds_result = ds_with_ties['val'].rank(method='max')
        pd_result = pd_with_ties['val'].rank(method='max')

        np.testing.assert_array_equal(get_series(ds_result).values, pd_result.values)

    def test_rank_dense(self, ds_with_ties, pd_with_ties):
        """Test rank with method='dense'."""
        ds_result = ds_with_ties['val'].rank(method='dense')
        pd_result = pd_with_ties['val'].rank(method='dense')

        np.testing.assert_array_equal(get_series(ds_result).values, pd_result.values)

    def test_rank_ascending_false(self, ds_with_ties, pd_with_ties):
        """Test rank with ascending=False."""
        ds_result = ds_with_ties['val'].rank(ascending=False)
        pd_result = pd_with_ties['val'].rank(ascending=False)

        np.testing.assert_array_almost_equal(get_series(ds_result).values, pd_result.values)

    def test_rank_pct(self, ds_with_ties, pd_with_ties):
        """Test rank with pct=True (percentage ranks)."""
        ds_result = ds_with_ties['val'].rank(pct=True)
        pd_result = pd_with_ties['val'].rank(pct=True)

        np.testing.assert_array_almost_equal(get_series(ds_result).values, pd_result.values)


# =============================================================================
# NLARGEST/NSMALLEST EDGE CASES
# =============================================================================


class TestNlargestNsmallestEdgeCases:
    """Test nlargest() and nsmallest() edge cases."""

    def test_nlargest_basic(self):
        """Test nlargest with basic data."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds = DataStore({'a': [3, 1, 4, 1, 5, 9, 2, 6]})

        ds_result = ds.nlargest(3, 'a')
        pd_result = pd_df.nlargest(3, 'a')

        assert len(ds_result) == 3
        assert set(ds_result['a']) == set(pd_result['a'])

    def test_nsmallest_basic(self):
        """Test nsmallest with basic data."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9, 2, 6]})
        ds = DataStore({'a': [3, 1, 4, 1, 5, 9, 2, 6]})

        ds_result = ds.nsmallest(3, 'a')
        pd_result = pd_df.nsmallest(3, 'a')

        assert len(ds_result) == 3
        assert set(ds_result['a']) == set(pd_result['a'])

    def test_nlargest_with_ties(self):
        """Test nlargest when there are tied values."""
        pd_df = pd.DataFrame({'a': [5, 5, 5, 3, 3, 1]})
        ds = DataStore({'a': [5, 5, 5, 3, 3, 1]})

        ds_result = ds.nlargest(3, 'a')
        pd_result = pd_df.nlargest(3, 'a')

        # Should get the top 3 (all 5s)
        assert len(ds_result) == len(pd_result)

    def test_nlargest_n_greater_than_length(self):
        """Test nlargest when n > length of DataFrame."""
        pd_df = pd.DataFrame({'a': [3, 1, 4]})
        ds = DataStore({'a': [3, 1, 4]})

        ds_result = ds.nlargest(10, 'a')
        pd_result = pd_df.nlargest(10, 'a')

        # Should return all rows
        assert len(ds_result) == len(pd_result) == 3


# =============================================================================
# CLIP EDGE CASES
# =============================================================================


class TestClipEdgeCases:
    """Test clip() edge cases."""

    def test_clip_both_bounds(self):
        """Test clip with both lower and upper bounds."""
        pd_df = pd.DataFrame({'a': [-5, 0, 5, 10, 15]})
        ds = DataStore({'a': [-5, 0, 5, 10, 15]})

        ds_result = ds.clip(lower=0, upper=10)
        pd_result = pd_df.clip(lower=0, upper=10)

        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_clip_lower_only(self):
        """Test clip with only lower bound."""
        pd_df = pd.DataFrame({'a': [-5, 0, 5, 10, 15]})
        ds = DataStore({'a': [-5, 0, 5, 10, 15]})

        ds_result = ds.clip(lower=0)
        pd_result = pd_df.clip(lower=0)

        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_clip_upper_only(self):
        """Test clip with only upper bound."""
        pd_df = pd.DataFrame({'a': [-5, 0, 5, 10, 15]})
        ds = DataStore({'a': [-5, 0, 5, 10, 15]})

        ds_result = ds.clip(upper=10)
        pd_result = pd_df.clip(upper=10)

        np.testing.assert_array_equal(ds_result['a'].values, pd_result['a'].values)

    def test_clip_column_expr(self):
        """Test clip on ColumnExpr (Series)."""
        pd_df = pd.DataFrame({'a': [-5, 0, 5, 10, 15]})
        ds = DataStore({'a': [-5, 0, 5, 10, 15]})

        ds_result = ds['a'].clip(lower=0, upper=10)
        pd_result = pd_df['a'].clip(lower=0, upper=10)

        np.testing.assert_array_equal(get_series(ds_result).values, pd_result.values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
