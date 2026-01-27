"""
chDB Limitations Tracker Tests

This module provides automated tracking of known chDB limitations.
When a limitation is resolved in a new chDB version, the corresponding
test will pass unexpectedly (xpassed), indicating the limitation can be
removed from documentation.

Run with: pytest tests/test_chdb_limitations_tracker.py -v

To check which limitations might be resolved:
    pytest tests/test_chdb_limitations_tracker.py -v --tb=no | grep xpassed
"""

import numpy as np
import pandas as pd
import pytest

from tests.xfail_markers import (
    chdb_array_nullable,
    chdb_no_normalize_utf8,
    chdb_no_quantile_array,
    chdb_datetime_timezone,
)

# Import DataStore - adjust path as needed
import sys
sys.path.insert(0, '/Users/auxten/Codes/go/src/github.com/auxten/chdb-ds')
from datastore import DataStore


class TestTypeSupportResolved:
    """Tests for chDB type support - previously limited, now resolved."""

    def test_categorical_type_support(self):
        """Categorical type is now supported (resolved in chDB 4.0.0b3)."""
        df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'c', 'a'])})
        ds = DataStore(df)
        result = ds._get_df()
        # Categorical is now supported - converted to string internally
        assert len(result) == 4

    def test_timedelta_type_support(self):
        """Timedelta type is now supported (resolved in chDB 4.0.0b3)."""
        df = pd.DataFrame({'td': pd.to_timedelta(['1 days', '2 days', '3 days'])})
        ds = DataStore(df)
        result = ds._get_df()
        # Timedelta is now supported
        assert len(result) == 3


class TestTypeSupportLimitations:
    """Tests for remaining chDB type support limitations."""

    def test_nullable_int64_comparison(self):
        """Check if chDB handles Nullable Int64 comparison correctly."""
        df = pd.DataFrame({'val': pd.array([1, pd.NA, 3], dtype='Int64')})
        ds = DataStore(df)
        # Try filtering - this used to return raw bytes
        result = ds[ds['val'] > 1]._get_df()
        assert len(result) == 1
        assert result['val'].iloc[0] == 3

    @chdb_array_nullable
    def test_array_in_nullable(self):
        """Check if chDB allows Array inside Nullable type."""
        df = pd.DataFrame({'text': ['hello world', 'foo bar', None]})
        ds = DataStore(df)
        # str.findall returns array - used to fail with Array in Nullable error
        result = ds['text'].str.findall(r'\w+')._get_df()
        assert len(result) == 3


class TestNullHandlingResolved:
    """Tests for NULL handling - previously limited, now resolved."""

    def test_nan_sum_behavior(self):
        """Sum of all-NaN now correctly handled (resolved in chDB 4.0.0b3)."""
        df = pd.DataFrame({'val': [np.nan, np.nan, np.nan]})
        ds = DataStore(df)
        
        pd_result = df['val'].sum()  # pandas returns 0.0
        ds_result = ds['val'].sum()
        
        # Both return 0.0 now
        assert ds_result == pd_result


class TestNullHandlingLimitations:
    """Tests for remaining NULL handling limitations."""
    def test_groupby_null_handling(self):
        """Check if DataStore excludes NULL from groupby like pandas (dropna=True default)."""
        df = pd.DataFrame({'group': ['A', 'B', None, 'A'], 'val': [1, 2, 3, 4]})
        ds = DataStore(df)
        
        pd_result = df.groupby('group', dropna=True)['val'].sum()
        ds_result = ds.groupby('group')['val'].sum()
        
        # pandas with dropna=True has 2 groups
        # DataStore now correctly excludes NULL by default (fixed dropna support)
        assert len(ds_result) == len(pd_result)


class TestFunctionAvailabilityResolved:
    """Tests for function availability - previously limited, now resolved."""

    def test_product_function(self):
        """product() is now available (resolved in chDB 4.0.0b3)."""
        df = pd.DataFrame({'val': [1, 2, 3, 4]})
        ds = DataStore(df)
        
        pd_result = df['val'].prod()  # 24
        ds_result = ds['val'].prod()
        
        assert ds_result == pd_result


class TestFunctionAvailabilityLimitations:
    """Tests for remaining function availability limitations."""

    @chdb_no_normalize_utf8
    def test_normalize_utf8(self):
        """Check if chDB now has normalizeUTF8NFD function."""
        df = pd.DataFrame({'text': ['cafe\u0301', 'naive\u0308']})  # cafe with accent
        ds = DataStore(df)
        
        result = ds['text'].str.normalize('NFD')._get_df()
        # If this passes, normalize is now supported
        assert len(result) == 2

    # xfail removed: quantile with array now works
    def test_quantile_array_param(self):
        """Check if chDB now supports quantile with array parameter."""
        df = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds = DataStore(df)
        
        pd_result = df['val'].quantile([0.25, 0.5, 0.75])
        ds_result = ds['val'].quantile([0.25, 0.5, 0.75])
        
        # If this passes, array quantile is now supported
        assert len(ds_result) == 3


class TestStringHandlingResolved:
    """Tests for string handling - previously limited, now resolved."""

    def test_unicode_filter(self):
        """Unicode string filtering now works (resolved in chDB 4.0.0b3)."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
        ds = DataStore(df)
        
        result = ds[ds['name'] == 'Alice']._get_df()
        assert len(result) == 1


class TestStringHandlingLimitations:
    """Tests for remaining string handling limitations."""

    def test_strip_all_whitespace(self):
        """Check if chDB str.strip() handles all whitespace types."""
        df = pd.DataFrame({'text': ['  hello  ', '\thello\t', '\nhello\n']})
        ds = DataStore(df)
        
        pd_result = df['text'].str.strip()
        ds_result = ds['text'].str.strip()
        
        # Check tabs and newlines are stripped
        pd_list = pd_result.tolist()
        ds_list = list(ds_result)
        for pd_val, ds_val in zip(pd_list, ds_list):
            assert pd_val == ds_val


class TestDatetimeLimitations:
    """Tests for datetime handling limitations."""

    @chdb_datetime_timezone
    def test_datetime_timezone_boundary(self):
        """Check if chDB handles datetime timezone boundaries correctly."""
        # Create datetime at year boundary in UTC
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2024-01-01 00:30:00', '2023-12-31 23:30:00'])
        })
        ds = DataStore(df)
        
        pd_year = df['ts'].dt.year.tolist()
        ds_year = list(ds['ts'].dt.year.values)
        
        # Should be [2024, 2023], not affected by timezone
        assert pd_year == ds_year


class TestLimitationsSummary:
    """Summary test that reports limitation status."""

    def test_print_limitation_summary(self):
        """Print summary of tested limitations (always passes)."""
        from tests.xfail_markers import MARKER_REGISTRY, get_markers_by_category
        
        categories = {
            'chdb_engine': 'chDB Engine Limitations',
            'datastore': 'DataStore Limitations',
            'index': 'Index Preservation',
            'design': 'Design Differences',
            'deprecated': 'Deprecated Features',
        }
        
        print("\n" + "=" * 60)
        print("chDB LIMITATIONS STATUS")
        print("=" * 60)
        
        # Report resolved limitations
        resolved = [
            'chdb_category_type',
            'chdb_timedelta_type', 
            'chdb_nan_sum_behavior',
            'chdb_no_product_function',
            'chdb_unicode_filter',
        ]
        print("\nRESOLVED LIMITATIONS (chDB 4.0.0b3):")
        for marker in resolved:
            print(f"  - {marker}")
        
        print("\nREMAINING LIMITATIONS:")
        for cat_key, cat_name in categories.items():
            markers = get_markers_by_category(cat_key)
            remaining = [m for m in markers if m not in resolved]
            if remaining:
                print(f"\n{cat_name}:")
                for marker in remaining:
                    _, issue_url, notes = MARKER_REGISTRY.get(marker, (None, None, 'No notes'))
                    print(f"  - {marker}: {notes}")
        
        print("\n" + "=" * 60)
        print("Run with --tb=no | grep xpassed to find newly resolved limitations")
        print("=" * 60 + "\n")
        
        # This test always passes - it's just for reporting
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
