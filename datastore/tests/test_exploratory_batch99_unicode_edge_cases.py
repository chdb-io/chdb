"""
Exploratory tests for Unicode and string edge cases.

Discovered issues:
1. str.len() counts bytes instead of characters for Unicode (chDB behavior)
2. str.upper() may add null bytes to Unicode strings (chDB behavior)
3. str.strip() fails with Unicode + whitespace (chDB trimBoth bug)

These tests use mirror pattern to compare DataStore vs pandas behavior.
"""

import sys
import os

# Add datastore to path for test imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.tests.test_utils import (
    assert_series_equal,
    assert_frame_equal,
    assert_datastore_equals_pandas,
)


class TestUnicodeStringOperations:
    """Test string operations with Unicode characters."""

    def test_str_len_ascii(self):
        """str.len() should match pandas for ASCII strings."""
        pd_df = pd.DataFrame({
            'text': ['hello', 'world', 'test', '']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    @pytest.mark.xfail(reason="chDB counts bytes not characters for Unicode")
    def test_str_len_unicode_chinese(self):
        """str.len() should count characters, not bytes, for Chinese."""
        pd_df = pd.DataFrame({
            'text': ['世界', '你好', '中文测试']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        # pandas counts characters: 2, 2, 4
        # chDB counts bytes (UTF-8): 6, 6, 12 (3 bytes per Chinese char)
        assert_series_equal(ds_result, pd_result, check_dtype=False)

    @pytest.mark.xfail(reason="chDB counts bytes not characters for emoji")
    def test_str_len_emoji(self):
        """str.len() should count characters for emoji."""
        pd_df = pd.DataFrame({
            'text': ['🎉', '🎉🎉', 'a🎉b']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        # pandas counts characters: 1, 2, 3
        # chDB counts bytes: 4, 8, 6 (4 bytes per emoji in UTF-8)
        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_str_upper_ascii(self):
        """str.upper() should match pandas for ASCII."""
        pd_df = pd.DataFrame({
            'text': ['hello', 'WORLD', 'MiXeD']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.upper()
        ds_result = ds_df['text'].str.upper()

        assert_series_equal(ds_result, pd_result)

    @pytest.mark.xfail(reason="chDB may add null bytes to Unicode in upper()")
    def test_str_upper_unicode(self):
        """str.upper() should not add null bytes to Unicode."""
        pd_df = pd.DataFrame({
            'text': ['世界', '你好']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.upper()
        ds_result = ds_df['text'].str.upper()

        # Check no null bytes are added
        ds_values = list(ds_result)
        for val in ds_values:
            assert '\x00' not in val, f"Null byte found in '{repr(val)}'"

        assert_series_equal(ds_result, pd_result)

    def test_str_lower_ascii(self):
        """str.lower() should match pandas for ASCII."""
        pd_df = pd.DataFrame({
            'text': ['HELLO', 'World', 'TEST']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.lower()
        ds_result = ds_df['text'].str.lower()

        assert_series_equal(ds_result, pd_result)

    def test_str_contains_ascii(self):
        """str.contains() should match pandas for ASCII patterns."""
        pd_df = pd.DataFrame({
            'text': ['hello world', 'test case', 'no match']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.contains('ell')
        ds_result = ds_df['text'].str.contains('ell')

        assert_series_equal(ds_result, pd_result)

    @pytest.mark.xfail(reason="chDB trimBoth bug with Unicode via Python() table function")
    def test_str_strip_with_unicode(self):
        """str.strip() should handle Unicode whitespace.

        This fails due to a chDB bug: trimBoth() on Unicode strings via Python()
        table function doesn't correctly trim right-side whitespace and adds
        null bytes to the result.
        """
        pd_df = pd.DataFrame({
            'text': ['  hello  ', '\t世界\t', '  test']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.strip()
        ds_result = ds_df['text'].str.strip()

        assert_series_equal(ds_result, pd_result)

    def test_str_strip_ascii_only(self):
        """str.strip() should work correctly for ASCII strings."""
        pd_df = pd.DataFrame({
            'text': ['  hello  ', '\tworld\t', '  test  ']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.strip()
        ds_result = ds_df['text'].str.strip()

        assert_series_equal(ds_result, pd_result)

    def test_str_with_none_values(self):
        """String operations should handle None values like pandas."""
        pd_df = pd.DataFrame({
            'text': ['hello', None, 'world', None]
        })
        ds_df = DataStore(pd_df)

        # Test len with None
        pd_len = pd_df['text'].str.len()
        ds_len = ds_df['text'].str.len()
        assert_series_equal(ds_len, pd_len, check_dtype=False)

        # Test upper with None
        pd_upper = pd_df['text'].str.upper()
        ds_upper = ds_df['text'].str.upper()
        assert_series_equal(ds_upper, pd_upper)


class TestMixedTypeOperations:
    """Test operations with mixed data types."""

    def test_filter_then_get_columns(self):
        """Filter should produce DataStore that supports column access."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore(pd_df.copy())

        # Filter
        pd_filtered = pd_df[pd_df['a'] > 2]
        ds_filtered = ds_df[ds_df['a'] > 2]

        # Use assert_datastore_equals_pandas for complete comparison
        # including values, not just shape
        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_empty_result_operations(self):
        """Operations on empty filtered results should work like pandas."""
        pd_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        ds_df = DataStore(pd_df)

        # Filter to empty
        pd_empty = pd_df[pd_df['x'] > 100]
        ds_empty = ds_df[ds_df['x'] > 100]

        # Use assert_datastore_equals_pandas for complete comparison
        assert_datastore_equals_pandas(ds_empty, pd_empty)

    def test_chained_filter_operations(self):
        """Chained filters should work correctly."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': list('aabbccddee')
        })
        ds_df = DataStore(pd_df)

        # Chain: filter -> filter -> filter
        pd_result = pd_df[pd_df['a'] > 2][pd_df['a'] < 8][pd_df['b'].isin(['b', 'c', 'd'])]
        ds_result = ds_df[ds_df['a'] > 2][ds_df['a'] < 8][ds_df['b'].isin(['b', 'c', 'd'])]

        # Use assert_datastore_equals_pandas for complete comparison
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByEdgeCases:
    """Test GroupBy edge cases."""

    def test_groupby_sum_all_nan(self):
        """GroupBy sum of all-NaN column should return 0 like pandas."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [np.nan, np.nan, np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        # pandas returns 0.0 for sum of all-NaN
        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_groupby_mean_all_nan(self):
        """GroupBy mean of all-NaN column should return NaN like pandas."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b'],
            'val': [np.nan, np.nan, np.nan, np.nan]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].mean()
        ds_result = ds_df.groupby('key')['val'].mean()

        # pandas returns NaN for mean of all-NaN
        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_groupby_single_group(self):
        """GroupBy with only one group."""
        pd_df = pd.DataFrame({
            'key': ['a', 'a', 'a'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_groupby_all_unique_keys(self):
        """GroupBy where each key appears only once."""
        pd_df = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val': [1, 2, 3]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('key')['val'].sum()
        ds_result = ds_df.groupby('key')['val'].sum()

        assert_series_equal(ds_result, pd_result, check_dtype=False)


class TestIlocEdgeCases:
    """Test iloc edge cases."""

    def test_iloc_negative_single(self):
        """iloc with negative single index."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[-1]
        ds_result = ds_df.iloc[-1]

        # Both should return Series
        assert_series_equal(ds_result, pd_result)

    def test_iloc_negative_slice(self):
        """iloc with negative slice should return comparable result."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': ['x', 'y', 'z', 'w', 'v']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[-2:]
        ds_result = ds_df.iloc[-2:]

        # Use assert_datastore_equals_pandas for complete comparison
        # Duck typing: access columns property to trigger execution
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWhereAndMask:
    """Test where() and mask() operations."""

    def test_where_with_scalar(self):
        """where() with scalar replacement."""
        pd_df = pd.DataFrame({
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_mask = pd_df['val'] > 2
        ds_mask = ds_df['val'] > 2

        pd_result = pd_df['val'].where(pd_mask, 0)
        ds_result = ds_df['val'].where(ds_mask, 0)

        assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_mask_with_scalar(self):
        """mask() with scalar replacement."""
        pd_df = pd.DataFrame({
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore(pd_df)

        pd_mask = pd_df['val'] > 2
        ds_mask = ds_df['val'] > 2

        pd_result = pd_df['val'].mask(pd_mask, 999)
        ds_result = ds_df['val'].mask(ds_mask, 999)

        assert_series_equal(ds_result, pd_result, check_dtype=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
