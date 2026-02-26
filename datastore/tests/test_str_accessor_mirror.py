"""
Test Series.str accessor methods with Mirror Code Pattern.

Tests for str.cat(), str.get_dummies(), str.partition(), str.rpartition()
to ensure DataStore behavior matches pandas exactly.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore


class TestStrCatMirror:
    """Test str.cat() with Mirror Code Pattern."""

    def test_cat_basic(self):
        """Test basic str.cat() with separator."""
        # pandas
        pd_series = pd.Series(['a', 'b', 'c'])
        pd_result = pd_series.str.cat(sep='-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['a', 'b', 'c']}))
        ds_result = ds['col'].str.cat(sep='-')

        # Compare
        assert ds_result == pd_result
        assert isinstance(ds_result, str)

    def test_cat_no_separator(self):
        """Test str.cat() without separator."""
        # pandas
        pd_series = pd.Series(['hello', 'world'])
        pd_result = pd_series.str.cat()

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello', 'world']}))
        ds_result = ds['col'].str.cat()

        # Compare
        assert ds_result == pd_result

    def test_cat_with_others(self):
        """Test str.cat() with other Series."""
        # pandas - use named series to match DataStore behavior
        pd_series1 = pd.Series(['a', 'b', 'c'], name='col1')
        pd_series2 = pd.Series(['x', 'y', 'z'], name='col2')
        pd_result = pd_series1.str.cat(pd_series2, sep='-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['x', 'y', 'z']}))
        ds_series2 = ds['col2']._execute()  # Need pandas Series for others
        ds_result = ds['col1'].str.cat(ds_series2, sep='-')

        # Compare - result should be a Series
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_cat_with_na_rep(self):
        """Test str.cat() with NA replacement."""
        # pandas
        pd_series = pd.Series(['a', None, 'c'])
        pd_result = pd_series.str.cat(sep='-', na_rep='NULL')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['a', None, 'c']}))
        ds_result = ds['col'].str.cat(sep='-', na_rep='NULL')

        # Compare
        assert ds_result == pd_result


class TestStrGetDummiesMirror:
    """Test str.get_dummies() with Mirror Code Pattern."""

    def test_get_dummies_basic(self):
        """Test basic str.get_dummies() with separator."""
        # pandas
        pd_series = pd.Series(['a|b', 'b|c', 'a|c'])
        pd_result = pd_series.str.get_dummies('|')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['a|b', 'b|c', 'a|c']}))
        ds_result = ds['col'].str.get_dummies('|')

        # Compare - DataStore returns DataStore, need to convert
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_get_dummies_single_value(self):
        """Test str.get_dummies() with single values (no separator)."""
        # pandas
        pd_series = pd.Series(['a', 'b', 'a'])
        pd_result = pd_series.str.get_dummies('|')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['a', 'b', 'a']}))
        ds_result = ds['col'].str.get_dummies('|')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_get_dummies_multiple_dummies(self):
        """Test str.get_dummies() with multiple values per row."""
        # pandas
        pd_series = pd.Series(['a|b|c', 'a', 'b|c'])
        pd_result = pd_series.str.get_dummies('|')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['a|b|c', 'a', 'b|c']}))
        ds_result = ds['col'].str.get_dummies('|')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)


class TestStrPartitionMirror:
    """Test str.partition() with Mirror Code Pattern."""

    def test_partition_basic(self):
        """Test basic str.partition()."""
        # pandas
        pd_series = pd.Series(['hello-world', 'foo-bar', 'test-string'])
        pd_result = pd_series.str.partition('-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello-world', 'foo-bar', 'test-string']}))
        ds_result = ds['col'].str.partition('-')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_partition_default_sep(self):
        """Test str.partition() with default separator (space)."""
        # pandas
        pd_series = pd.Series(['hello world', 'foo bar', 'one two'])
        pd_result = pd_series.str.partition()  # Default sep is ' '

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello world', 'foo bar', 'one two']}))
        ds_result = ds['col'].str.partition()

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_partition_not_found(self):
        """Test str.partition() when separator is not found."""
        # pandas
        pd_series = pd.Series(['hello', 'world', 'test'])
        pd_result = pd_series.str.partition('-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello', 'world', 'test']}))
        ds_result = ds['col'].str.partition('-')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_partition_expand_false(self):
        """Test str.partition() with expand=False."""
        # pandas - use named series to match DataStore behavior
        pd_series = pd.Series(['hello-world', 'foo-bar'], name='col')
        pd_result = pd_series.str.partition('-', expand=False)

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello-world', 'foo-bar']}))
        ds_result = ds['col'].str.partition('-', expand=False)

        # Compare - should return Series of tuples
        pd.testing.assert_series_equal(ds_result, pd_result)


class TestStrRpartitionMirror:
    """Test str.rpartition() with Mirror Code Pattern."""

    def test_rpartition_basic(self):
        """Test basic str.rpartition()."""
        # pandas
        pd_series = pd.Series(['hello-world-test', 'foo-bar-baz', 'a-b-c'])
        pd_result = pd_series.str.rpartition('-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello-world-test', 'foo-bar-baz', 'a-b-c']}))
        ds_result = ds['col'].str.rpartition('-')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_rpartition_vs_partition(self):
        """Test rpartition splits on last occurrence vs partition on first."""
        test_data = ['a-b-c']
        
        # pandas
        pd_series = pd.Series(test_data)
        pd_partition = pd_series.str.partition('-')
        pd_rpartition = pd_series.str.rpartition('-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': test_data}))
        ds_partition = ds['col'].str.partition('-').to_df()
        ds_rpartition = ds['col'].str.rpartition('-').to_df()

        # Compare
        pd.testing.assert_frame_equal(ds_partition, pd_partition)
        pd.testing.assert_frame_equal(ds_rpartition, pd_rpartition)
        
        # Verify difference
        # partition('a-b-c', '-') -> 'a', '-', 'b-c'
        # rpartition('a-b-c', '-') -> 'a-b', '-', 'c'
        assert list(ds_partition.iloc[0]) == ['a', '-', 'b-c']
        assert list(ds_rpartition.iloc[0]) == ['a-b', '-', 'c']

    def test_rpartition_default_sep(self):
        """Test str.rpartition() with default separator (space)."""
        # pandas
        pd_series = pd.Series(['hello world test', 'foo bar baz'])
        pd_result = pd_series.str.rpartition()  # Default sep is ' '

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello world test', 'foo bar baz']}))
        ds_result = ds['col'].str.rpartition()

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_rpartition_expand_false(self):
        """Test str.rpartition() with expand=False."""
        # pandas - use named series to match DataStore behavior
        pd_series = pd.Series(['hello-world-test', 'foo-bar'], name='col')
        pd_result = pd_series.str.rpartition('-', expand=False)

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello-world-test', 'foo-bar']}))
        ds_result = ds['col'].str.rpartition('-', expand=False)

        # Compare - should return Series of tuples
        pd.testing.assert_series_equal(ds_result, pd_result)


class TestStrAccessorEdgeCases:
    """Test edge cases for str accessor methods."""

    def test_partition_with_multi_char_sep(self):
        """Test partition with multi-character separator."""
        # pandas
        pd_series = pd.Series(['hello::world', 'foo::bar'])
        pd_result = pd_series.str.partition('::')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['hello::world', 'foo::bar']}))
        ds_result = ds['col'].str.partition('::')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_get_dummies_with_space_sep(self):
        """Test get_dummies with space separator."""
        # pandas
        pd_series = pd.Series(['a b c', 'a', 'b c'])
        pd_result = pd_series.str.get_dummies(' ')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['a b c', 'a', 'b c']}))
        ds_result = ds['col'].str.get_dummies(' ')

        # Compare
        ds_df = ds_result.to_df()
        pd.testing.assert_frame_equal(ds_df, pd_result)

    def test_cat_empty_strings(self):
        """Test str.cat() with empty strings."""
        # pandas
        pd_series = pd.Series(['', 'a', ''])
        pd_result = pd_series.str.cat(sep='-')

        # DataStore (mirror)
        ds = DataStore.from_df(pd.DataFrame({'col': ['', 'a', '']}))
        ds_result = ds['col'].str.cat(sep='-')

        # Compare
        assert ds_result == pd_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
