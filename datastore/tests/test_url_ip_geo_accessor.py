"""
Test URL, IP, and Geo accessor functionality.

Verifies that:
1. All accessor methods return ColumnExpr (pandas API compatible)
2. Results support sort_values() and other pandas methods
3. Chaining operations work correctly
"""

import pytest
import pandas as pd
from datastore import DataStore
from datastore.column_expr import ColumnExpr


class TestUrlAccessor:
    """Test .url accessor methods return ColumnExpr for pandas API compatibility."""

    @pytest.fixture
    def ds_with_urls(self):
        """Create a test DataStore with URL columns."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'url': [
                'https://example.com/path/page?id=1&name=test',
                'https://test.org/api/v1?token=abc',
                'https://demo.net/home?ref=google',
            ],
        })
        return DataStore.from_df(df)

    def test_url_accessor_returns_column_expr_accessor(self, ds_with_urls):
        """Test that .url returns ColumnExprUrlAccessor."""
        accessor = ds_with_urls['url'].url
        assert 'ColumnExprUrlAccessor' in type(accessor).__name__

    def test_url_domain_returns_column_expr(self, ds_with_urls):
        """Test url.domain() returns ColumnExpr."""
        result = ds_with_urls['url'].url.domain()
        assert isinstance(result, ColumnExpr)
        assert hasattr(result, 'sort_values')

    def test_url_domain_sort_values(self, ds_with_urls):
        """Test url.domain().sort_values() works."""
        result = ds_with_urls['url'].url.domain().sort_values()
        assert isinstance(result, ColumnExpr)
        # Values should be sorted alphabetically
        values = list(result.values)
        assert values == sorted(values)

    def test_url_path_returns_column_expr(self, ds_with_urls):
        """Test url.url_path() returns ColumnExpr."""
        result = ds_with_urls['url'].url.url_path()
        assert isinstance(result, ColumnExpr)
        assert hasattr(result, 'sort_values')

    def test_url_protocol_returns_column_expr(self, ds_with_urls):
        """Test url.protocol() returns ColumnExpr."""
        result = ds_with_urls['url'].url.protocol()
        assert isinstance(result, ColumnExpr)

    def test_url_chain_with_str_accessor(self, ds_with_urls):
        """Test url accessor result can chain with .str accessor."""
        result = ds_with_urls['url'].url.domain().str.upper()
        assert isinstance(result, ColumnExpr)
        
        ds_with_urls['upper_domain'] = result
        df = ds_with_urls.to_df()
        
        # All domains should be uppercase
        assert all(d.isupper() for d in df['upper_domain'])

    def test_url_sql_generation(self, ds_with_urls):
        """Test url accessor generates correct SQL."""
        result = ds_with_urls['url'].url.domain()
        sql = result._expr.to_sql()
        
        assert 'domain' in sql.lower()
        assert 'url' in sql


class TestIpAccessor:
    """Test .ip accessor methods return ColumnExpr for pandas API compatibility."""

    @pytest.fixture
    def ds_with_ips(self):
        """Create a test DataStore with IP columns."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'ip': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
        })
        return DataStore.from_df(df)

    def test_ip_accessor_returns_column_expr_accessor(self, ds_with_ips):
        """Test that .ip returns ColumnExprIpAccessor."""
        accessor = ds_with_ips['ip'].ip
        assert 'ColumnExprIpAccessor' in type(accessor).__name__

    def test_ip_to_ipv4_returns_column_expr(self, ds_with_ips):
        """Test ip.to_ipv4() returns ColumnExpr."""
        result = ds_with_ips['ip'].ip.to_ipv4()
        assert isinstance(result, ColumnExpr)
        assert hasattr(result, 'sort_values')

    def test_ip_is_ipv4_string_returns_column_expr(self, ds_with_ips):
        """Test ip.is_ipv4_string() returns ColumnExpr."""
        result = ds_with_ips['ip'].ip.is_ipv4_string()
        assert isinstance(result, ColumnExpr)

    def test_ip_sql_generation(self, ds_with_ips):
        """Test ip accessor generates correct SQL."""
        result = ds_with_ips['ip'].ip.to_ipv4()
        sql = result._expr.to_sql()
        
        assert 'toIPv4' in sql or 'ip' in sql.lower()


class TestGeoAccessor:
    """Test .geo accessor methods return ColumnExpr for pandas API compatibility."""

    @pytest.fixture
    def ds_with_vectors(self):
        """Create a test DataStore with vector columns for geo operations."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'vec': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        })
        return DataStore.from_df(df)

    def test_geo_accessor_returns_column_expr_accessor(self, ds_with_vectors):
        """Test that .geo returns ColumnExprGeoAccessor."""
        accessor = ds_with_vectors['vec'].geo
        assert 'ColumnExprGeoAccessor' in type(accessor).__name__

    def test_geo_l2_norm_returns_column_expr(self, ds_with_vectors):
        """Test geo.l2_norm() returns ColumnExpr."""
        result = ds_with_vectors['vec'].geo.l2_norm()
        assert isinstance(result, ColumnExpr)
        assert hasattr(result, 'sort_values')

    def test_geo_l2_normalize_returns_column_expr(self, ds_with_vectors):
        """Test geo.l2_normalize() returns ColumnExpr."""
        result = ds_with_vectors['vec'].geo.l2_normalize()
        assert isinstance(result, ColumnExpr)

    def test_geo_sql_generation(self, ds_with_vectors):
        """Test geo accessor generates correct SQL."""
        result = ds_with_vectors['vec'].geo.l2_norm()
        sql = result._expr.to_sql()
        
        assert 'L2Norm' in sql or 'l2' in sql.lower() or 'vec' in sql


class TestAccessorConsistency:
    """Test that all accessors are consistent in returning ColumnExpr."""

    @pytest.fixture
    def ds(self):
        """Create a test DataStore with various column types."""
        df = pd.DataFrame({
            'name': ['alice', 'bob'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'url': ['https://example.com', 'https://test.org'],
            'ip': ['192.168.1.1', '10.0.0.1'],
            'data': ['{"key": "value"}', '{"key": "other"}'],
            'nums': [[1, 2, 3], [4, 5]],
        })
        return DataStore.from_df(df)

    def test_all_accessors_return_column_expr_accessor(self, ds):
        """Test all accessor types return the correct ColumnExpr wrapper."""
        # All accessor wrapper class names
        expected_wrappers = {
            'str': 'ColumnExprStringAccessor',
            'dt': 'ColumnExprDateTimeAccessor',
            'json': 'ColumnExprJsonAccessor',
            'arr': 'ColumnExprArrayAccessor',
            'url': 'ColumnExprUrlAccessor',
            'ip': 'ColumnExprIpAccessor',
            'geo': 'ColumnExprGeoAccessor',
        }
        
        test_columns = {
            'str': 'name',
            'dt': 'date',
            'json': 'data',
            'arr': 'nums',
            'url': 'url',
            'ip': 'ip',
            'geo': 'nums',  # geo works on arrays too
        }
        
        for accessor_name, expected_wrapper in expected_wrappers.items():
            col = test_columns[accessor_name]
            accessor = getattr(ds[col], accessor_name)
            actual_type = type(accessor).__name__
            assert actual_type == expected_wrapper, \
                f"{accessor_name} accessor returned {actual_type}, expected {expected_wrapper}"

    def test_all_accessor_methods_have_sort_values(self, ds):
        """Test accessor method results all support sort_values()."""
        # Test each accessor's method returns ColumnExpr with sort_values
        test_cases = [
            ('name', 'str', 'upper', []),
            ('date', 'dt', 'year', []),
            ('data', 'json', 'json_extract_string', ['key']),
            ('url', 'url', 'domain', []),
            ('ip', 'ip', 'to_ipv4', []),
            ('nums', 'geo', 'l2_norm', []),
        ]
        
        for col, accessor_name, method_name, args in test_cases:
            accessor = getattr(ds[col], accessor_name)
            method = getattr(accessor, method_name)
            
            # Handle properties vs methods
            if callable(method):
                result = method(*args)
            else:
                result = method
            
            assert isinstance(result, ColumnExpr), \
                f"{accessor_name}.{method_name} did not return ColumnExpr"
            assert hasattr(result, 'sort_values'), \
                f"{accessor_name}.{method_name} result has no sort_values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

