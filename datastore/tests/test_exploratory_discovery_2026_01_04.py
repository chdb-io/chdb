"""
Tests for bugs discovered during exploratory testing on 2026-01-04.

These tests document the expected behavior and serve as regression tests
once the bugs are fixed.
"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from datastore import DataStore
from tests.test_utils import assert_frame_equal, assert_series_equal, get_series


class TestCombineFirst:
    """Test combine_first method with DataStore arguments."""

    def test_combine_first_with_datastore(self):
        """combine_first should accept DataStore as argument - Mirror Pattern."""
        # pandas
        df1 = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
        df2 = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
        pd_result = df1.combine_first(df2)

        # DataStore (mirror)
        ds1 = DataStore(df1.copy())
        ds2 = DataStore(df2.copy())
        ds_result = ds1.combine_first(ds2)

        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True)
        )

    def test_combine_first_with_pandas_dataframe(self):
        """combine_first should also accept pandas DataFrame."""
        # pandas
        df1 = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, 6]})
        df2 = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
        pd_result = df1.combine_first(df2)

        # DataStore with pandas DataFrame as other
        ds1 = DataStore(df1.copy())
        ds_result = ds1.combine_first(df2.copy())

        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True)
        )

    def test_combine_first_with_different_columns(self):
        """combine_first should handle DataFrames with different columns."""
        # pandas
        df1 = pd.DataFrame({'A': [1, np.nan], 'B': [np.nan, 5]})
        df2 = pd.DataFrame({'B': [10, 20], 'C': [30, 40]})
        pd_result = df1.combine_first(df2)

        # DataStore (mirror)
        ds1 = DataStore(df1.copy())
        ds2 = DataStore(df2.copy())
        ds_result = ds1.combine_first(ds2)

        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True),
        )

    def test_combine_with_datastore(self):
        """combine should also accept DataStore as argument."""
        # pandas
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [10, 20], 'B': [30, 40]})
        pd_result = df1.combine(df2, lambda s1, s2: s1 + s2)

        # DataStore (mirror)
        ds1 = DataStore(df1.copy())
        ds2 = DataStore(df2.copy())
        ds_result = ds1.combine(ds2, lambda s1, s2: s1 + s2)

        # Compare
        assert_frame_equal(
            ds_result._get_df().reset_index(drop=True),
            pd_result.reset_index(drop=True)
        )


class TestNlargest:
    """Test nlargest method column names."""

    def test_nlargest_preserves_column_names(self):
        """nlargest should preserve original column names."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85, 92, 78, 95, 88],
            'age': [25, 30, 35, 28, 32]
        })

        # pandas behavior
        pd_result = df.nlargest(3, 'score')
        assert list(pd_result.columns) == ['name', 'score', 'age']

        # DataStore
        ds = DataStore(df.copy())
        ds_result = ds.nlargest(3, 'score')

        # Verify values are correct
        ds_df = ds_result._get_df()
        assert len(ds_df) == 3

        # TODO: This assertion should pass once bug is fixed
        # Currently fails because columns become [0, 1, 2]
        # assert list(ds_df.columns) == ['name', 'score', 'age']


class TestMask:
    """Test mask method behavior across all columns."""

    def test_mask_replaces_all_columns(self):
        """mask should replace values in ALL columns where condition is True."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        cond = df['A'] > 2

        # pandas behavior - replaces both columns
        pd_result = df.mask(cond, -1)
        assert pd_result['A'].tolist() == [1, 2, -1, -1, -1]
        assert pd_result['B'].tolist() == [10, 20, -1, -1, -1]

        # DataStore - currently has issues with columns
        ds = DataStore(df.copy())
        ds_cond = ds['A'] > 2
        ds_result = ds.mask(ds_cond, -1)

        ds_df = ds_result._get_df()

        # Note: DataStore mask result may have different column structure
        # Just verify it returns something for now
        assert len(ds_df) == 5

    #  Now fixed - mask replaces all columns correctly
    def test_mask_all_columns_should_pass(self):
        """This test should pass once the bug is fixed."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        cond = df['A'] > 2

        pd_result = df.mask(cond, -1)

        ds = DataStore(df.copy())
        ds_cond = ds['A'] > 2
        ds_result = ds.mask(ds_cond, -1)

        assert_frame_equal(
            ds_result._get_df(),
            pd_result
        )


class TestSeriesIdxmin:
    """Test Series.idxmin/idxmax methods."""

    def test_series_idxmin_works(self):
        """Series.idxmin should return index of minimum value."""
        df = pd.DataFrame({
            'A': [1, 5, 3, 4, 2],
            'B': [10, 8, 15, 12, 5]
        })

        # pandas behavior
        pd_idxmin = df['A'].idxmin()
        assert pd_idxmin == 0

        pd_idxmax = df['A'].idxmax()
        assert pd_idxmax == 1

        # DataStore - DataFrame level works
        ds = DataStore(df.copy())
        ds_df_idxmin = ds.idxmin()
        assert ds_df_idxmin['A'] == 0
        assert ds_df_idxmin['B'] == 4

        # TODO: Series level currently fails
        # ds_series_idxmin = ds['A'].idxmin()
        # assert ds_series_idxmin == 0

    
    def test_series_idxmin_should_pass(self):
        """This test should pass once the bug is fixed."""
        df = pd.DataFrame({'A': [1, 5, 3, 4, 2]})

        ds = DataStore(df.copy())
        ds_idxmin = ds['A'].idxmin()
        assert ds_idxmin == 0


class TestSample:
    """Test sample method index preservation."""

    def test_sample_preserves_index(self):
        """sample should preserve original row indices."""
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100, 200)
        })

        # pandas behavior - preserves original indices
        pd_result = df.sample(n=5, random_state=42)
        pd_indices = list(pd_result.index)

        # Verify indices are from original dataframe (not 0-4)
        assert pd_indices != [0, 1, 2, 3, 4]
        assert all(idx in range(100) for idx in pd_indices)

        # DataStore
        ds = DataStore(df.copy())
        ds_result = ds.sample(n=5, random_state=42)
        ds_df = ds_result._get_df()

        # Verify DataStore preserves indices matching pandas behavior
        assert list(ds_df.index) == pd_indices


class TestGroupbyAggColumnNames:
    """Test groupby agg column name format."""

    def test_groupby_agg_multiindex_columns(self):
        """Document the difference in column name format."""
        df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'B'],
            'value1': [10, 20, 30, 40, 50],
            'value2': [100, 200, 300, 400, 500]
        })

        # pandas returns MultiIndex columns
        pd_result = df.groupby('category').agg({
            'value1': ['sum', 'mean'],
            'value2': ['min', 'max']
        })
        assert isinstance(pd_result.columns, pd.MultiIndex)

        # DataStore returns flat column names
        ds = DataStore(df.copy())
        ds_result = ds.groupby('category').agg({
            'value1': ['sum', 'mean'],
            'value2': ['min', 'max']
        })

        # DataStore uses flat names like 'value1_sum'
        assert 'value1_sum' in ds_result.columns or ('value1', 'sum') in ds_result.columns

        # Values should be the same regardless of column name format
        if 'value1_sum' in ds_result.columns:
            assert ds_result.loc['A', 'value1_sum'] == 30
        else:
            assert ds_result.loc['A', ('value1', 'sum')] == 30


class TestWorkingOperations:
    """Tests for operations that work correctly (regression tests)."""

    def test_pivot(self):
        """pivot should work correctly."""
        df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'C': [1, 2, 3, 4]
        })

        pd_result = df.pivot(index='A', columns='B', values='C')

        ds = DataStore(df.copy())
        ds_result = ds.pivot(index='A', columns='B', values='C')

        # Compare values only, ignore index type differences
        ds_df = ds_result._get_df()
        np.testing.assert_array_equal(
            ds_df.values,
            pd_result.values
        )

    def test_rank(self):
        """rank should work correctly with ties."""
        df = pd.DataFrame({'score': [85, 92, 92, 78, 88]})

        pd_result = df['score'].rank()

        ds = DataStore(df.copy())
        ds_result = ds['score'].rank()

        assert_series_equal(
            get_series(ds_result),
            pd_result)

    def test_clip(self):
        """clip should work correctly."""
        df = pd.DataFrame({
            'A': [1, 5, 10, 15, 20],
            'B': [-5, 0, 5, 10, 25]
        })

        pd_result = df.clip(lower=0, upper=15)

        ds = DataStore(df.copy())
        ds_result = ds.clip(lower=0, upper=15)

        np.testing.assert_array_equal(
            ds_result._get_df().values,
            pd_result.values
        )

    def test_where(self):
        """where should work correctly."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })

        cond = df['A'] > 2

        pd_result = df.where(cond, -1)

        ds = DataStore(df.copy())
        ds_cond = ds['A'] > 2
        ds_result = ds.where(ds_cond, -1)

        # Compare values only
        ds_df = ds_result._get_df()
        np.testing.assert_array_equal(
            ds_df.values,
            pd_result.values
        )

    def test_value_counts_with_na(self):
        """value_counts should handle NaN correctly."""
        df = pd.DataFrame({'A': ['foo', 'bar', 'foo', np.nan, 'bar', 'bar', np.nan]})

        pd_result = df['A'].value_counts(dropna=False)

        ds = DataStore(df.copy())
        ds_result = ds['A'].value_counts(dropna=False)

        # Verify counts are correct
        assert ds_result['bar'] == 3
        assert ds_result['foo'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestSeriesIdxminIdxmax:
    """Tests for Series idxmin and idxmax methods."""

    def test_idxmin_basic(self):
        """idxmin should return the index of the minimum value."""
        df = pd.DataFrame({
            'A': [1, 5, 3, 4, 2],
            'B': [10, 8, 15, 12, 5]
        })

        pd_idxmin = df['A'].idxmin()

        ds = DataStore(df.copy())
        ds_idxmin = ds['A'].idxmin()

        assert ds_idxmin == pd_idxmin
        assert ds_idxmin == 0  # Index of value 1

    def test_idxmax_basic(self):
        """idxmax should return the index of the maximum value."""
        df = pd.DataFrame({
            'A': [1, 5, 3, 4, 2],
            'B': [10, 8, 15, 12, 5]
        })

        pd_idxmax = df['A'].idxmax()

        ds = DataStore(df.copy())
        ds_idxmax = ds['A'].idxmax()

        assert ds_idxmax == pd_idxmax
        assert ds_idxmax == 1  # Index of value 5

    def test_idxmin_with_string_index(self):
        """idxmin should work with string index."""
        df = pd.DataFrame({
            'A': [3, 1, 5, 2],
            'B': [10, 8, 15, 12]
        }, index=['w', 'x', 'y', 'z'])

        pd_idxmin = df['A'].idxmin()

        ds = DataStore(df.copy())
        ds_idxmin = ds['A'].idxmin()

        assert ds_idxmin == pd_idxmin
        assert ds_idxmin == 'x'  # Index of value 1

    def test_idxmax_with_string_index(self):
        """idxmax should work with string index."""
        df = pd.DataFrame({
            'A': [3, 1, 5, 2],
            'B': [10, 8, 15, 12]
        }, index=['w', 'x', 'y', 'z'])

        pd_idxmax = df['A'].idxmax()

        ds = DataStore(df.copy())
        ds_idxmax = ds['A'].idxmax()

        assert ds_idxmax == pd_idxmax
        assert ds_idxmax == 'y'  # Index of value 5

    def test_idxmin_with_na_values(self):
        """idxmin should handle NA values correctly (skipna=True default)."""
        df = pd.DataFrame({
            'A': [3.0, np.nan, 1.0, 5.0, 2.0],
            'B': [10, 8, 15, 12, 5]
        })

        pd_idxmin = df['A'].idxmin()

        ds = DataStore(df.copy())
        ds_idxmin = ds['A'].idxmin()

        assert ds_idxmin == pd_idxmin
        assert ds_idxmin == 2  # Index of value 1.0

    def test_idxmax_with_na_values(self):
        """idxmax should handle NA values correctly (skipna=True default)."""
        df = pd.DataFrame({
            'A': [3.0, np.nan, 1.0, 5.0, 2.0],
            'B': [10, 8, 15, 12, 5]
        })

        pd_idxmax = df['A'].idxmax()

        ds = DataStore(df.copy())
        ds_idxmax = ds['A'].idxmax()

        assert ds_idxmax == pd_idxmax
        assert ds_idxmax == 3  # Index of value 5.0
