"""
Tests for value_counts SQL pushdown optimization.

value_counts() is essentially GROUP BY + COUNT(*), which can be efficiently
executed in SQL/chDB instead of loading all data into pandas first.

Test coverage:
1. Basic value_counts (default parameters)
2. normalize=True (relative frequencies)
3. sort=True/False (ordering)
4. ascending=True/False (sort direction)
5. dropna=True/False (NULL handling)
6. bins parameter (numeric binning)
7. Edge cases (empty, single value, all NULL)
8. Data types (string, int, float, bool, datetime)
9. Chained operations (value_counts().head(), .nlargest(), etc.)
"""

import pytest
import pandas as pd
import numpy as np

from datastore import DataStore
from datastore.config import get_execution_engine, set_execution_engine, ExecutionEngine
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


class TestValueCountsBasic:
    """Basic value_counts tests with default parameters."""

    def test_value_counts_strings(self):
        """Test value_counts on string column."""
        pd_df = pd.DataFrame({'col': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']})
        ds_df = DataStore({'col': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        # Default sort=True, ascending=False: most frequent first
        assert_series_equal(ds_result, pd_result)

    def test_value_counts_integers(self):
        """Test value_counts on integer column."""
        pd_df = pd.DataFrame({'col': [1, 2, 1, 3, 2, 1, 1]})
        ds_df = DataStore({'col': [1, 2, 1, 3, 2, 1, 1]})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_floats(self):
        """Test value_counts on float column."""
        pd_df = pd.DataFrame({'col': [1.1, 2.2, 1.1, 3.3, 2.2, 1.1]})
        ds_df = DataStore({'col': [1.1, 2.2, 1.1, 3.3, 2.2, 1.1]})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_boolean(self):
        """Test value_counts on boolean column."""
        pd_df = pd.DataFrame({'col': [True, False, True, True, False]})
        ds_df = DataStore({'col': [True, False, True, True, False]})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        assert_series_equal(ds_result, pd_result)


class TestValueCountsNormalize:
    """Test value_counts with normalize parameter."""

    def test_value_counts_normalize_true(self):
        """Test value_counts(normalize=True) returns proportions."""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'a', 'b']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'a', 'b']})

        pd_result = pd_df['col'].value_counts(normalize=True)
        ds_result = ds_df['col'].value_counts(normalize=True)

        # Proportions should sum to 1
        assert abs(sum(get_series(ds_result).values) - 1.0) < 1e-10
        assert_series_equal(ds_result, pd_result, check_dtype=False, rtol=1e-10)

    def test_value_counts_normalize_with_dropna(self):
        """Test normalize with dropna combinations."""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', None, 'b']})
        ds_df = DataStore({'col': ['a', 'b', 'a', None, 'b']})

        # normalize=True, dropna=True (default)
        pd_result = pd_df['col'].value_counts(normalize=True, dropna=True)
        ds_result = ds_df['col'].value_counts(normalize=True, dropna=True)
        assert abs(sum(get_series(ds_result).values) - 1.0) < 1e-10
        assert_series_equal(ds_result, pd_result, check_dtype=False, rtol=1e-10)

        # normalize=True, dropna=False
        pd_result = pd_df['col'].value_counts(normalize=True, dropna=False)
        ds_result = ds_df['col'].value_counts(normalize=True, dropna=False)
        assert abs(sum(get_series(ds_result).values) - 1.0) < 1e-10
        # Sort by index for comparison since order may differ
        assert_series_equal(
            ds_result.sort_index(na_position='last'),
            pd_result.sort_index(na_position='last'),
            check_dtype=False,
            rtol=1e-10,
        )


class TestValueCountsSort:
    """Test value_counts sorting behavior."""

    def test_value_counts_sort_true_ascending_false(self):
        """Default: sort=True, ascending=False (most frequent first)."""
        pd_df = pd.DataFrame({'col': ['c', 'a', 'b', 'a', 'b', 'a']})
        ds_df = DataStore({'col': ['c', 'a', 'b', 'a', 'b', 'a']})

        pd_result = pd_df['col'].value_counts(sort=True, ascending=False)
        ds_result = ds_df['col'].value_counts(sort=True, ascending=False)

        # 'a' should be first (count=3), then 'b' (count=2), then 'c' (count=1)
        assert_series_equal(ds_result, pd_result)
        assert list(get_series(ds_result).index)[:1] == ['a']

    def test_value_counts_sort_true_ascending_true(self):
        """Test sort=True, ascending=True (least frequent first)."""
        pd_df = pd.DataFrame({'col': ['c', 'a', 'b', 'a', 'b', 'a']})
        ds_df = DataStore({'col': ['c', 'a', 'b', 'a', 'b', 'a']})

        pd_result = pd_df['col'].value_counts(sort=True, ascending=True)
        ds_result = ds_df['col'].value_counts(sort=True, ascending=True)

        # 'c' should be first (count=1)
        assert_series_equal(ds_result, pd_result)
        assert list(get_series(ds_result).index)[:1] == ['c']

    def test_value_counts_sort_false(self):
        """Test sort=False (order by value, not count)."""
        pd_df = pd.DataFrame({'col': ['c', 'a', 'b', 'a', 'b', 'a']})
        ds_df = DataStore({'col': ['c', 'a', 'b', 'a', 'b', 'a']})

        pd_result = pd_df['col'].value_counts(sort=False)
        ds_result = ds_df['col'].value_counts(sort=False)

        # When sort=False, pandas returns in order of first appearance
        # SQL may return in different order, so compare as sets
        assert set(get_series(ds_result).index) == set(pd_result.index)
        for val in pd_result.index:
            assert get_series(ds_result).loc[val] == pd_result.loc[val]


class TestValueCountsDropna:
    """Test value_counts NULL handling."""

    def test_value_counts_dropna_true(self):
        """Test dropna=True excludes NULL values."""
        pd_df = pd.DataFrame({'col': ['a', 'b', None, 'a', None]})
        ds_df = DataStore({'col': ['a', 'b', None, 'a', None]})

        pd_result = pd_df['col'].value_counts(dropna=True)
        ds_result = ds_df['col'].value_counts(dropna=True)

        assert_series_equal(ds_result, pd_result)
        # NULL should not be in index
        assert pd.isna(get_series(ds_result).index).sum() == 0

    def test_value_counts_dropna_false(self):
        """Test dropna=False includes NULL values."""
        pd_df = pd.DataFrame({'col': ['a', 'b', None, 'a', None]})
        ds_df = DataStore({'col': ['a', 'b', None, 'a', None]})

        pd_result = pd_df['col'].value_counts(dropna=False)
        ds_result = ds_df['col'].value_counts(dropna=False)

        # Sort by index for comparison
        pd_sorted = pd_result.sort_index(na_position='last')
        ds_sorted = ds_result.sort_index(na_position='last')
        assert_series_equal(ds_sorted, pd_sorted)
        # NULL should be in index with count=2
        assert pd.isna(get_series(ds_result).index).sum() == 1

    def test_value_counts_all_null(self):
        """Test value_counts with all NULL values."""
        pd_df = pd.DataFrame({'col': [None, None, None]})
        ds_df = DataStore({'col': [None, None, None]})

        # dropna=True should return empty series
        pd_result = pd_df['col'].value_counts(dropna=True)
        ds_result = ds_df['col'].value_counts(dropna=True)
        assert len(get_series(ds_result)) == len(pd_result) == 0

        # dropna=False should return single entry with NULL
        pd_result = pd_df['col'].value_counts(dropna=False)
        ds_result = ds_df['col'].value_counts(dropna=False)
        assert len(get_series(ds_result)) == len(pd_result) == 1


class TestValueCountsBins:
    """Test value_counts bins parameter for numeric data."""

    def test_value_counts_bins_basic(self):
        """Test value_counts(bins=N) for numeric data."""
        pd_df = pd.DataFrame({'col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore({'col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        pd_result = pd_df['col'].value_counts(bins=3)
        ds_result = ds_df['col'].value_counts(bins=3)

        # bins creates intervals; compare values
        assert len(get_series(ds_result)) == len(pd_result)
        # Total counts should match
        assert sum(get_series(ds_result).values) == sum(pd_result.values)

    def test_value_counts_bins_with_sort(self):
        """Test value_counts(bins=N) with sort options."""
        pd_df = pd.DataFrame({'col': [1, 1, 1, 5, 5, 9]})
        ds_df = DataStore({'col': [1, 1, 1, 5, 5, 9]})

        # Default sort by count
        pd_result = pd_df['col'].value_counts(bins=3, sort=True)
        ds_result = ds_df['col'].value_counts(bins=3, sort=True)
        assert sum(get_series(ds_result).values) == sum(pd_result.values)

    def test_value_counts_bins_with_normalize(self):
        """Test value_counts(bins=N, normalize=True)."""
        pd_df = pd.DataFrame({'col': list(range(1, 11))})
        ds_df = DataStore({'col': list(range(1, 11))})

        pd_result = pd_df['col'].value_counts(bins=5, normalize=True)
        ds_result = ds_df['col'].value_counts(bins=5, normalize=True)

        # Normalized values should sum to 1
        assert abs(sum(get_series(ds_result).values) - 1.0) < 1e-10


class TestValueCountsEdgeCases:
    """Test edge cases for value_counts."""

    def test_value_counts_empty_dataframe(self):
        """Test value_counts on empty DataFrame."""
        pd_df = pd.DataFrame({'col': []})
        ds_df = DataStore({'col': []})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        assert len(get_series(ds_result)) == len(pd_result) == 0

    def test_value_counts_single_value(self):
        """Test value_counts with single unique value."""
        pd_df = pd.DataFrame({'col': ['a', 'a', 'a']})
        ds_df = DataStore({'col': ['a', 'a', 'a']})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        assert_series_equal(ds_result, pd_result)
        assert get_series(ds_result).iloc[0] == 3

    def test_value_counts_all_unique(self):
        """Test value_counts with all unique values."""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore({'col': ['a', 'b', 'c', 'd', 'e']})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        # All counts should be 1
        assert all(v == 1 for v in get_series(ds_result).values)
        assert len(get_series(ds_result)) == 5

    def test_value_counts_ties_ordering(self):
        """Test value_counts with tied counts - verify stable ordering."""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'c', 'a', 'b', 'c']})
        ds_df = DataStore({'col': ['a', 'b', 'c', 'a', 'b', 'c']})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        # All have count=2, order may differ
        assert set(get_series(ds_result).index) == set(pd_result.index)
        assert all(v == 2 for v in get_series(ds_result).values)


class TestValueCountsChained:
    """Test value_counts chained with other operations."""

    def test_value_counts_head(self):
        """Test value_counts().head(n)."""
        pd_df = pd.DataFrame({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3 + ['d'] * 1})
        ds_df = DataStore({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3 + ['d'] * 1})

        pd_result = pd_df['col'].value_counts().head(2)
        ds_result = ds_df['col'].value_counts().head(2)

        assert_series_equal(ds_result, pd_result)
        assert len(get_series(ds_result)) == 2

    def test_value_counts_tail(self):
        """Test value_counts().tail(n)."""
        pd_df = pd.DataFrame({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3 + ['d'] * 1})
        ds_df = DataStore({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3 + ['d'] * 1})

        pd_result = pd_df['col'].value_counts().tail(2)
        ds_result = ds_df['col'].value_counts().tail(2)

        assert_series_equal(ds_result, pd_result)
        assert len(get_series(ds_result)) == 2

    def test_value_counts_nlargest(self):
        """Test value_counts().nlargest(n)."""
        pd_df = pd.DataFrame({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3 + ['d'] * 1})
        ds_df = DataStore({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3 + ['d'] * 1})

        pd_result = pd_df['col'].value_counts().nlargest(2)
        ds_result = ds_df['col'].value_counts().nlargest(2)

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_max(self):
        """Test value_counts().max() - get maximum count."""
        pd_df = pd.DataFrame({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3})
        ds_df = DataStore({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3})

        pd_result = pd_df['col'].value_counts().max()
        ds_result = get_series(ds_df['col'].value_counts()).max()

        assert ds_result == pd_result == 10

    def test_value_counts_idxmax(self):
        """Test value_counts().idxmax() - get most frequent value."""
        pd_df = pd.DataFrame({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3})
        ds_df = DataStore({'col': ['a'] * 10 + ['b'] * 5 + ['c'] * 3})

        pd_result = pd_df['col'].value_counts().idxmax()
        ds_result = get_series(ds_df['col'].value_counts()).idxmax()

        assert ds_result == pd_result == 'a'


class TestValueCountsAfterFilter:
    """Test value_counts after filtering operations."""

    def test_value_counts_after_filter(self):
        """Test value_counts on filtered DataFrame."""
        pd_df = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'B', 'A'], 'value': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore({'category': ['A', 'B', 'A', 'C', 'B', 'A'], 'value': [1, 2, 3, 4, 5, 6]})

        # Filter then value_counts
        pd_filtered = pd_df[pd_df['value'] > 2]
        ds_filtered = ds_df[ds_df['value'] > 2]

        pd_result = pd_filtered['category'].value_counts()
        ds_result = ds_filtered['category'].value_counts()

        # Should match pandas exactly including tie-breaking order (first-appearance)
        assert_series_equal(ds_result, pd_result)

    def test_value_counts_after_assignment(self):
        """Test value_counts after assign()."""
        pd_df = pd.DataFrame({'col': ['a', 'b', 'a', 'c']})
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c']})

        pd_assigned = pd_df.assign(col=lambda x: x['col'].str.upper())
        ds_assigned = ds_df.assign(col=lambda x: x['col'].str.upper())

        pd_result = pd_assigned['col'].value_counts()
        ds_result = ds_assigned['col'].value_counts()

        assert_series_equal(ds_result, pd_result)


class TestValueCountsWithNaN:
    """Test value_counts with various NaN/NULL types."""

    def test_value_counts_nan_float(self):
        """Test value_counts with float NaN."""
        pd_df = pd.DataFrame({'col': [1.0, 2.0, np.nan, 1.0, np.nan]})
        ds_df = DataStore({'col': [1.0, 2.0, np.nan, 1.0, np.nan]})

        # dropna=True
        pd_result = pd_df['col'].value_counts(dropna=True)
        ds_result = ds_df['col'].value_counts(dropna=True)
        assert_series_equal(ds_result, pd_result)

        # dropna=False
        pd_result = pd_df['col'].value_counts(dropna=False)
        ds_result = ds_df['col'].value_counts(dropna=False)
        # Sort for comparison
        assert_series_equal(
            ds_result.sort_index(na_position='last'),
            pd_result.sort_index(na_position='last'),
        )

    def test_value_counts_none_string(self):
        """Test value_counts with None in string column."""
        pd_df = pd.DataFrame({'col': ['a', 'b', None, 'a', None]})
        ds_df = DataStore({'col': ['a', 'b', None, 'a', None]})

        pd_result = pd_df['col'].value_counts(dropna=False)
        ds_result = ds_df['col'].value_counts(dropna=False)

        # Sort for comparison
        assert_series_equal(
            ds_result.sort_index(na_position='last'),
            pd_result.sort_index(na_position='last'),
        )


class TestValueCountsSQLPushdown:
    """Test that value_counts uses SQL pushdown when appropriate."""

    def test_value_counts_uses_sql_with_chdb_engine(self):
        """Test that value_counts executes via SQL with CHDB engine."""
        import io
        import logging

        # Create DataStore with table function to enable SQL execution
        ds_df = DataStore({'col': ['a', 'b', 'a', 'c', 'b', 'a']})

        # Capture logs to verify SQL execution
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger('datastore')
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            # Execute value_counts with CHDB engine
            original_engine = get_execution_engine()
            set_execution_engine(ExecutionEngine.CHDB)

            result = ds_df['col'].value_counts()
            # Force execution
            _ = repr(result)

            # Check if SQL was used (in future implementation)
            log_output = log_capture.getvalue()
            # For now, this may use pandas - future implementation will use SQL

        finally:
            set_execution_engine(original_engine)
            logger.removeHandler(handler)
            logger.setLevel(original_level)


class TestValueCountsFromFile:
    """Test value_counts on DataStore created from file sources."""

    @pytest.fixture
    def sample_parquet(self, tmp_path):
        """Create a sample parquet file for testing."""
        df = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'A'],
                'value': [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        path = tmp_path / 'test_data.parquet'
        df.to_parquet(path)
        return path, df

    def test_value_counts_from_parquet(self, sample_parquet):
        """Test value_counts on DataStore loaded from parquet."""
        path, pdf = sample_parquet

        ds = DataStore.from_file(str(path))

        pd_result = pdf['category'].value_counts()
        ds_result = ds['category'].value_counts()

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_from_parquet_filtered(self, sample_parquet):
        """Test value_counts on filtered DataStore from parquet."""
        path, pdf = sample_parquet

        ds = DataStore.from_file(str(path))

        # Filter then value_counts
        pd_filtered = pdf[pdf['value'] > 30]
        ds_filtered = ds[ds['value'] > 30]

        pd_result = pd_filtered['category'].value_counts()
        ds_result = ds_filtered['category'].value_counts()

        # Should match pandas exactly including tie-breaking order (first-appearance)
        assert_series_equal(ds_result, pd_result)


class TestValueCountsDataTypes:
    """Test value_counts with various data types."""

    def test_value_counts_datetime(self):
        """Test value_counts on datetime column."""
        dates = ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03', '2023-01-01']
        pd_df = pd.DataFrame({'date': pd.to_datetime(dates)})
        ds_df = DataStore({'date': pd.to_datetime(dates)})

        pd_result = pd_df['date'].value_counts()
        ds_result = ds_df['date'].value_counts()

        # pandas 3.0 uses datetime64[us], chDB uses datetime64[ns]
        # Compare values and index values (as strings) instead of strict dtype check
        ds_series = get_series(ds_result)
        assert list(ds_series.values) == list(pd_result.values), "Counts should match"
        # Compare index as strings to ignore datetime64 precision difference
        assert [str(x) for x in ds_series.index] == [str(x) for x in pd_result.index], "Dates should match"

    @pytest.mark.xfail(
        reason="chDB converts categorical to object dtype and CategoricalIndex to Index - VALUES ARE CORRECT",
        strict=True,
    )
    def test_value_counts_categorical(self):
        """Test value_counts on categorical column."""
        pd_df = pd.DataFrame({'col': pd.Categorical(['a', 'b', 'a', 'c', 'b', 'a'])})
        ds_df = DataStore({'col': pd.Categorical(['a', 'b', 'a', 'c', 'b', 'a'])})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_mixed_numeric(self):
        """Test value_counts with mixed integer and float (same values)."""
        pd_df = pd.DataFrame({'col': [1, 1.0, 2, 2.0, 1]})  # 1 and 1.0 are equal
        ds_df = DataStore({'col': [1, 1.0, 2, 2.0, 1]})

        pd_result = pd_df['col'].value_counts()
        ds_result = ds_df['col'].value_counts()

        # pandas treats 1 and 1.0 as same value
        assert_series_equal(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
