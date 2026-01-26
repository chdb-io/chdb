"""
Comprehensive tests for replace functionality.

Tests both Series.replace (value replacement) and str.replace (substring replacement).
"""

import pandas as pd
import pytest

from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import chdb_replace_none_dtype


class TestSeriesReplaceSingleValue:
    """Test Series.replace with single value."""

    def test_replace_single_numeric(self):
        """Replace single numeric value."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace(1, 100)
        ds_result = ds_df['a'].replace(1, 100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_single_string(self):
        """Replace single string value."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': ['hello', 'world', 'test']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace('hello', 'hi')
        ds_result = ds_df['a'].replace('hello', 'hi')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_single_no_match(self):
        """Replace with no matching values."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace(100, 999)
        ds_result = ds_df['a'].replace(100, 999)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_single_float(self):
        """Replace single float value."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace(2.0, 200.0)
        ds_result = ds_df['a'].replace(2.0, 200.0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSeriesReplaceDict:
    """Test Series.replace with dict argument."""

    def test_replace_dict_numeric(self):
        """Replace multiple values using dict - numeric."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace({1: 100, 2: 200})
        ds_result = ds_df['a'].replace({1: 100, 2: 200})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict_string(self):
        """Replace multiple values using dict - string."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': ['hello', 'world', 'test', 'hello']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace({'hello': 'hi', 'world': 'earth'})
        ds_result = ds_df['a'].replace({'hello': 'hi', 'world': 'earth'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict_empty(self):
        """Replace with empty dict - should return unchanged."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace({})
        ds_result = ds_df['a'].replace({})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict_many_values(self):
        """Replace many values using dict."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df)

        replace_dict = {1: 10, 3: 30, 5: 50, 7: 70, 9: 90}
        pd_result = pd_df['a'].replace(replace_dict)
        ds_result = ds_df['a'].replace(replace_dict)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict_partial_match(self):
        """Replace dict where not all keys are present."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace({1: 100, 99: 999})  # 99 doesn't exist
        ds_result = ds_df['a'].replace({1: 100, 99: 999})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSeriesReplaceList:
    """Test Series.replace with list arguments."""

    def test_replace_list_numeric(self):
        """Replace multiple values using lists - numeric."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace([1, 2], [100, 200])
        ds_result = ds_df['a'].replace([1, 2], [100, 200])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_string(self):
        """Replace multiple values using lists - string."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': ['hello', 'world', 'test']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace(['hello', 'world'], ['hi', 'earth'])
        ds_result = ds_df['a'].replace(['hello', 'world'], ['hi', 'earth'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_empty(self):
        """Replace with empty lists - should return unchanged."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace([], [])
        ds_result = ds_df['a'].replace([], [])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSeriesReplaceEdgeCases:
    """Test edge cases for Series.replace."""

    @chdb_replace_none_dtype
    def test_replace_with_none_value(self):
        """Replace value with None/null."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace(1, None)
        ds_result = ds_df['a'].replace(1, None)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_all_values(self):
        """Replace all values in series."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 1, 1, 1, 1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace(1, 100)
        ds_result = ds_df['a'].replace(1, 100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_preserves_unchanged(self):
        """Values not in replace should be unchanged."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].replace({1: 100})
        ds_result = ds_df['a'].replace({1: 100})

        # Check that 2, 3, 4, 5 are unchanged
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStrReplaceSubstring:
    """Test str.replace for substring replacement."""

    def test_str_replace_basic(self):
        """Basic substring replacement."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'s': ['hello world', 'test world', 'foo bar']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['s'].str.replace('world', 'earth')
        ds_result = ds_df['s'].str.replace('world', 'earth')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_no_match(self):
        """Substring replacement with no matches."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'s': ['hello', 'test', 'foo']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['s'].str.replace('xyz', 'abc')
        ds_result = ds_df['s'].str.replace('xyz', 'abc')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_multiple_occurrences(self):
        """Replace multiple occurrences of substring."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'s': ['aaa', 'abab', 'axa']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['s'].str.replace('a', 'X')
        ds_result = ds_df['s'].str.replace('a', 'X')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_with_regex(self):
        """Substring replacement with regex."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'s': ['hello world', 'test world', 'foo bar']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['s'].str.replace('w.*d', 'X', regex=True)
        ds_result = ds_df['s'].str.replace('w.*d', 'X', regex=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_regex_groups(self):
        """Substring replacement with regex groups."""
        from datastore import DataStore

        pd_df = pd.DataFrame({'s': ['abc123', 'def456', 'ghi789']})
        ds_df = DataStore(pd_df)

        # Replace digits with X
        pd_result = pd_df['s'].str.replace('[0-9]+', 'X', regex=True)
        ds_result = ds_df['s'].str.replace('[0-9]+', 'X', regex=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestReplaceMirrorPattern:
    """Mirror pattern tests - comprehensive comparison with pandas."""

    def test_numeric_series_replace_dict_mirror(self):
        """Mirror test: numeric series with dict replace."""
        from datastore import DataStore

        data = {'col': [1, 2, 3, 4, 5, 1, 2, 3]}

        pd_df = pd.DataFrame(data)
        ds_df = DataStore(pd_df)

        pd_result = pd_df['col'].replace({1: 100, 3: 300})
        ds_result = ds_df['col'].replace({1: 100, 3: 300})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_string_series_replace_dict_mirror(self):
        """Mirror test: string series with dict replace."""
        from datastore import DataStore

        data = {'col': ['a', 'b', 'c', 'a', 'b', 'c']}

        pd_df = pd.DataFrame(data)
        ds_df = DataStore(pd_df)

        pd_result = pd_df['col'].replace({'a': 'A', 'b': 'B'})
        ds_result = ds_df['col'].replace({'a': 'A', 'b': 'B'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_float_series_replace_dict_mirror(self):
        """Mirror test: float series with dict replace."""
        from datastore import DataStore

        data = {'col': [1.0, 2.0, 3.0, 1.0, 2.0]}

        pd_df = pd.DataFrame(data)
        ds_df = DataStore(pd_df)

        pd_result = pd_df['col'].replace({1.0: 10.0, 2.0: 20.0})
        ds_result = ds_df['col'].replace({1.0: 10.0, 2.0: 20.0})

        assert_datastore_equals_pandas(ds_result, pd_result)
