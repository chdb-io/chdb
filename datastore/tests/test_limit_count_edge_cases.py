"""
Tests for limit() and count_rows() edge cases.

This module tests various scenarios including:
- Data size less than limit
- Data size equal to limit
- Data size greater than limit
- Limit of 0
- Limit with filter
- Chained limits
- Count with limit
- Offset and limit combinations
"""

import unittest
import tempfile
import os
import pandas as pd

from datastore import DataStore
from tests.test_utils import assert_frame_equal


class TestLimitCountEdgeCases(unittest.TestCase):
    """Test edge cases for limit() and count_rows()."""

    @classmethod
    def setUpClass(cls):
        """Create test data files with known row counts."""
        cls.temp_dir = tempfile.mkdtemp()

        # Small dataset (4 rows)
        cls.small_csv = os.path.join(cls.temp_dir, "small.csv")
        pd.DataFrame(
            {
                'id': [1, 2, 3, 4],
                'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                'age': [25, 30, 35, 28],
                'score': [85.5, 90.0, 78.5, 92.0],
            }
        ).to_csv(cls.small_csv, index=False)

        # Medium dataset (10 rows)
        cls.medium_csv = os.path.join(cls.temp_dir, "medium.csv")
        pd.DataFrame({'id': list(range(1, 11)), 'value': [i * 10 for i in range(1, 11)]}).to_csv(
            cls.medium_csv, index=False
        )

        # Single row dataset
        cls.single_csv = os.path.join(cls.temp_dir, "single.csv")
        pd.DataFrame({'id': [1], 'name': ['Only']}).to_csv(cls.single_csv, index=False)

        # Empty dataset - use parquet which handles empty data correctly
        cls.empty_parquet = os.path.join(cls.temp_dir, "empty.parquet")
        pd.DataFrame(columns=['id', 'name']).to_parquet(cls.empty_parquet)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # ==================== Limit Less Than Data Size ====================

    def test_limit_less_than_data_size(self):
        """Test limit when n < total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df['id']), [1, 2])

    def test_head_less_than_data_size(self):
        """Test head when n < total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df['name']), ['Alice', 'Bob'])

    # ==================== Limit Equal To Data Size ====================

    def test_limit_equal_to_data_size(self):
        """Test limit when n == total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(4)
        self.assertEqual(len(result), 4)

    def test_head_equal_to_data_size(self):
        """Test head when n == total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(4)
        self.assertEqual(len(result), 4)

    # ==================== Limit Greater Than Data Size ====================

    def test_limit_greater_than_data_size(self):
        """Test limit when n > total rows - should return all rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(100)
        self.assertEqual(len(result), 4)  # Only 4 rows exist

    def test_head_greater_than_data_size(self):
        """Test head when n > total rows - should return all rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(100)
        self.assertEqual(len(result), 4)  # Only 4 rows exist

    def test_limit_much_greater_than_data_size(self):
        """Test limit when n >> total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(1000000)
        self.assertEqual(len(result), 4)

    # ==================== Limit of Zero ====================

    def test_limit_zero(self):
        """Test limit(0) returns no rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(0)
        self.assertEqual(len(result), 0)
        df = result.to_df()
        self.assertEqual(len(df), 0)

    def test_head_zero(self):
        """Test head(0) returns no rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(0)
        self.assertEqual(len(result), 0)

    # ==================== Limit of One ====================

    def test_limit_one(self):
        """Test limit(1) returns exactly one row."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.limit(1)
        self.assertEqual(len(result), 1)
        df = result.to_df()
        self.assertEqual(df['id'].iloc[0], 1)

    def test_head_one(self):
        """Test head(1) returns exactly one row."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.head(1)
        self.assertEqual(len(result), 1)

    # ==================== Single Row Dataset ====================

    def test_limit_on_single_row_less(self):
        """Test limit(0) on single row dataset."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.limit(0)
        self.assertEqual(len(result), 0)

    def test_limit_on_single_row_equal(self):
        """Test limit(1) on single row dataset."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.limit(1)
        self.assertEqual(len(result), 1)

    def test_limit_on_single_row_greater(self):
        """Test limit(10) on single row dataset."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.limit(10)
        self.assertEqual(len(result), 1)

    def test_head_on_single_row(self):
        """Test head() on single row dataset with default n=5."""
        ds = DataStore.from_file(self.single_csv)
        result = ds.head()  # default n=5
        self.assertEqual(len(result), 1)

    # ==================== Empty Dataset ====================

    def test_limit_on_empty_dataset(self):
        """Test limit on empty dataset."""
        ds = DataStore.from_file(self.empty_parquet)
        result = ds.limit(10)
        self.assertEqual(len(result), 0)

    def test_head_on_empty_dataset(self):
        """Test head on empty dataset."""
        ds = DataStore.from_file(self.empty_parquet)
        result = ds.head(10)
        self.assertEqual(len(result), 0)

    def test_count_rows_on_empty_dataset(self):
        """Test count_rows on empty dataset."""
        ds = DataStore.from_file(self.empty_parquet)
        self.assertEqual(ds.count_rows(), 0)

    # ==================== Limit With Filter ====================

    def test_limit_with_filter_less_than_filtered_rows(self):
        """Test limit < filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches: Bob(30), Charlie(35), Diana(28) = 3 rows
        result = ds.filter(ds.age > 25).limit(2)
        self.assertEqual(len(result), 2)

    def test_limit_with_filter_equal_to_filtered_rows(self):
        """Test limit == filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches: Bob(30), Charlie(35), Diana(28) = 3 rows
        result = ds.filter(ds.age > 25).limit(3)
        self.assertEqual(len(result), 3)

    def test_limit_with_filter_greater_than_filtered_rows(self):
        """Test limit > filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches: Bob(30), Charlie(35), Diana(28) = 3 rows
        result = ds.filter(ds.age > 25).limit(100)
        self.assertEqual(len(result), 3)

    def test_head_with_filter_greater_than_filtered_rows(self):
        """Test head > filtered row count."""
        ds = DataStore.from_file(self.small_csv)
        # age > 30 matches only Charlie(35) = 1 row
        result = ds.filter(ds.age > 30).head(10)
        self.assertEqual(len(result), 1)

    def test_limit_with_filter_returns_no_rows(self):
        """Test limit when filter matches no rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.filter(ds.age > 100).limit(10)
        self.assertEqual(len(result), 0)

    # ==================== Chained Limits ====================
    # Note: In SQL semantics, later LIMIT overwrites earlier LIMIT.
    # The final LIMIT in the query determines the result.

    def test_chained_limits_decreasing(self):
        """Test limit(10).limit(5) - later limit overwrites (SQL semantics)."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.limit(10).limit(5)
        self.assertEqual(len(result), 5)

    def test_chained_limits_increasing(self):
        """Test limit(5).limit(10) - pandas semantics: chained limits.

        In pandas semantics, df[:5][:10] means:
        1. Take first 5 rows
        2. From those 5, take first 10 -> min(5, 10) = 5

        DataStore follows pandas semantics for consistency with user expectations.
        """
        ds = DataStore.from_file(self.medium_csv)
        result = ds.limit(5).limit(10)
        # Pandas semantics: chained limits, result is min(5, 10) = 5
        self.assertEqual(len(result), 5)

    def test_chained_head(self):
        """Test head(5).head(3) - later head overwrites (SQL semantics)."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.head(5).head(3)
        self.assertEqual(len(result), 3)

    def test_chained_head_increasing(self):
        """Test head(3).head(5) - pandas semantics: chained head.

        In pandas semantics, df.head(3).head(5) means:
        1. Take first 3 rows
        2. From those 3, take first 5 -> min(3, 5) = 3
        """
        ds = DataStore.from_file(self.medium_csv)
        result = ds.head(3).head(5)
        # Pandas semantics: chained head, result is min(3, 5) = 3
        self.assertEqual(len(result), 3)

    # ==================== Offset and Limit Combinations ====================

    def test_offset_and_limit(self):
        """Test offset + limit combination."""
        ds = DataStore.from_file(self.medium_csv)
        result = ds.offset(3).limit(4)
        self.assertEqual(len(result), 4)
        df = result.to_df()
        # Should skip first 3, get rows 4-7 (id: 4, 5, 6, 7)
        self.assertEqual(list(df['id']), [4, 5, 6, 7])

    def test_offset_greater_than_data_size(self):
        """Test offset > total rows."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.offset(100).limit(10)
        self.assertEqual(len(result), 0)

    def test_offset_with_limit_exceeding_remaining(self):
        """Test offset + limit when limit > remaining rows."""
        ds = DataStore.from_file(self.medium_csv)
        # 10 rows, offset 8 leaves 2 rows
        result = ds.offset(8).limit(100)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df['id']), [9, 10])

    # ==================== Count Rows with Limit ====================

    def test_count_rows_with_limit(self):
        """Test count_rows() respects limit."""
        ds = DataStore.from_file(self.medium_csv)
        self.assertEqual(ds.limit(5).count_rows(), 5)

    def test_count_rows_with_limit_greater_than_data(self):
        """Test count_rows() with limit > data size."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(ds.limit(100).count_rows(), 4)

    def test_count_rows_with_limit_zero(self):
        """Test count_rows() with limit(0)."""
        ds = DataStore.from_file(self.medium_csv)
        self.assertEqual(ds.limit(0).count_rows(), 0)

    def test_len_with_limit(self):
        """Test len() respects limit."""
        ds = DataStore.from_file(self.medium_csv)
        self.assertEqual(len(ds.limit(3)), 3)
        self.assertEqual(len(ds.head(7)), 7)

    def test_len_with_limit_greater_than_data(self):
        """Test len() with limit > data size."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(len(ds.limit(1000)), 4)
        self.assertEqual(len(ds.head(1000)), 4)

    # ==================== Count Rows with Filter and Limit ====================

    def test_count_rows_filter_then_limit(self):
        """Test count_rows() with filter + limit."""
        ds = DataStore.from_file(self.small_csv)
        # age > 25 matches 3 rows, limit to 2
        result = ds.filter(ds.age > 25).limit(2)
        self.assertEqual(result.count_rows(), 2)

    def test_count_rows_filter_reduces_below_limit(self):
        """Test count_rows() when filter result < limit."""
        ds = DataStore.from_file(self.small_csv)
        # age > 32 matches only Charlie(35) = 1 row
        result = ds.filter(ds.age > 32).limit(100)
        self.assertEqual(result.count_rows(), 1)

    # ==================== Sort with Limit ====================

    def test_sort_then_limit(self):
        """Test sort + limit returns correct top N."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.sort_values('age', ascending=False).head(2)
        df = result.to_df()
        # Should get Charlie(35) and Bob(30)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(df['name']), ['Charlie', 'Bob'])

    def test_sort_then_limit_greater_than_data(self):
        """Test sort + limit > data size."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.sort_values('age', ascending=True).head(100)
        self.assertEqual(len(result), 4)

    # ==================== Count Without Limit (baseline) ====================

    def test_count_rows_without_limit(self):
        """Test count_rows() without limit - baseline."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(ds.count_rows(), 4)

    def test_count_rows_with_filter_no_limit(self):
        """Test count_rows() with filter but no limit."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(ds.filter(ds.age > 25).count_rows(), 3)

    def test_len_without_limit(self):
        """Test len() without limit - baseline."""
        ds = DataStore.from_file(self.small_csv)
        self.assertEqual(len(ds), 4)

    # ==================== Complex Combinations ====================

    def test_filter_sort_limit_count(self):
        """Test filter + sort + limit + count."""
        ds = DataStore.from_file(self.small_csv)
        result = ds.filter(ds.age > 25).sort_values('score', ascending=False).head(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        # age > 25: Bob(90), Charlie(78.5), Diana(92)
        # sorted by score desc: Diana(92), Bob(90), Charlie(78.5)
        # top 2: Diana, Bob
        self.assertEqual(list(df['name']), ['Diana', 'Bob'])

    def test_select_filter_limit(self):
        """Test select + filter + limit."""
        ds = DataStore.from_file(self.small_csv)
        result = ds[['name', 'age']][ds['age'] > 25].head(2)
        self.assertEqual(len(result), 2)
        df = result.to_df()
        self.assertEqual(list(df.columns), ['name', 'age'])


class TestLimitCountWithDataStore(unittest.TestCase):
    """Test limit and count with DataStore created from DataFrame."""

    def test_from_dataframe_limit(self):
        """Test limit on DataStore created from DataFrame."""
        df = pd.DataFrame({'x': range(100)})
        ds = DataStore.from_dataframe(df)
        result = ds.limit(10)
        self.assertEqual(len(result), 10)

    def test_from_dataframe_head_greater_than_data(self):
        """Test head > data size on DataFrame-backed DataStore."""
        df = pd.DataFrame({'x': range(5)})
        ds = DataStore.from_dataframe(df)
        result = ds.head(100)
        self.assertEqual(len(result), 5)

    def test_from_dataframe_count_rows(self):
        """Test count_rows on DataFrame-backed DataStore."""
        df = pd.DataFrame({'x': range(50)})
        ds = DataStore.from_dataframe(df)
        self.assertEqual(ds.count_rows(), 50)

    def test_from_dataframe_count_rows_with_limit(self):
        """Test count_rows with limit on DataFrame-backed DataStore."""
        df = pd.DataFrame({'x': range(50)})
        ds = DataStore.from_dataframe(df)
        self.assertEqual(ds.limit(10).count_rows(), 10)


class TestSliceStyleLimit(unittest.TestCase):
    """Test slice-style limit operations like ds[0:5], ds[:10], ds[5:]."""

    def setUp(self):
        """Create test DataStore with 20 rows."""
        self.df = pd.DataFrame(
            {
                'id': list(range(20)),
                'value': [i * 10 for i in range(20)],
                'name': [f'item_{i}' for i in range(20)],
            }
        )
        self.ds = DataStore.from_dataframe(self.df)

    # ==================== Basic Slice Tests ====================

    def test_slice_stop_only(self):
        """Test ds[:5] returns first 5 rows."""
        result = self.ds[:5]
        self.assertEqual(len(result), 5)
        df = result.to_df()
        self.assertEqual(list(df['id']), [0, 1, 2, 3, 4])

    def test_slice_start_only(self):
        """Test ds[5:] returns rows starting from index 5."""
        result = self.ds[5:]
        self.assertEqual(len(result), 15)
        df = result.to_df()
        self.assertEqual(list(df['id']), list(range(5, 20)))

    def test_slice_start_and_stop(self):
        """Test ds[5:10] returns rows 5-9."""
        result = self.ds[5:10]
        self.assertEqual(len(result), 5)
        df = result.to_df()
        self.assertEqual(list(df['id']), [5, 6, 7, 8, 9])

    def test_slice_full_range(self):
        """Test ds[:] returns all rows."""
        result = self.ds[:]
        self.assertEqual(len(result), 20)

    # ==================== Edge Cases ====================

    def test_slice_zero_stop(self):
        """Test ds[:0] returns empty result."""
        result = self.ds[:0]
        self.assertEqual(len(result), 0)

    def test_slice_stop_greater_than_data(self):
        """Test ds[:100] returns all rows when stop > data size."""
        result = self.ds[:100]
        self.assertEqual(len(result), 20)

    def test_slice_start_greater_than_data(self):
        """Test ds[100:] returns empty when start > data size."""
        result = self.ds[100:]
        self.assertEqual(len(result), 0)

    def test_slice_start_equals_stop(self):
        """Test ds[5:5] behavior - current implementation uses stop as limit."""
        # Note: When start == stop, current implementation uses stop as limit
        # ds[5:5] becomes LIMIT 5 OFFSET 5, not LIMIT 0 OFFSET 5
        # This matches the code: limit_val = stop - start if stop > start else stop
        result = self.ds[5:5]
        # With 20 rows, OFFSET 5 LIMIT 5 gives rows 5-9
        self.assertEqual(len(result), 5)
        df = result.to_df()
        self.assertEqual(list(df['id']), [5, 6, 7, 8, 9])

    def test_slice_single_element(self):
        """Test ds[5:6] returns single row."""
        result = self.ds[5:6]
        self.assertEqual(len(result), 1)
        df = result.to_df()
        self.assertEqual(df['id'].iloc[0], 5)

    def test_slice_with_step(self):
        """Test that step in slice is supported via LazySliceStep."""
        # Step slicing is now supported using pandas iloc
        result = self.ds[::2]  # Every 2nd row
        df = result.to_df()
        # Should get every other row from original 20 rows
        self.assertEqual(len(df), 10)
        # Should get rows at indices 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
        self.assertEqual(list(df['id']), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])

    # ==================== Chained Slice Tests ====================

    def test_slice_then_select(self):
        """Test slicing then column selection."""
        result = self.ds[:5][['id', 'name']]
        self.assertEqual(len(result), 5)
        df = result.to_df()
        self.assertEqual(list(df.columns), ['id', 'name'])

    def test_select_then_slice(self):
        """Test column selection then slicing."""
        result = self.ds[['id', 'value']][:5]
        self.assertEqual(len(result), 5)
        df = result.to_df()
        self.assertEqual(list(df.columns), ['id', 'value'])

    def test_slice_then_filter(self):
        """Test slicing then filtering."""
        result = self.ds[:10]
        result = result[result['value'] > 50]
        df = result.to_df()
        # Values 0-90, keep > 50: 60, 70, 80, 90 (indices 6,7,8,9)
        self.assertEqual(len(df), 4)
        for v in df['value']:
            self.assertGreater(v, 50)

    def test_filter_then_slice(self):
        """Test filtering then slicing."""
        filtered = self.ds[self.ds['value'] >= 100]  # 10 rows: 100, 110, ..., 190
        result = filtered[:3]
        self.assertEqual(len(result), 3)
        df = result.to_df()
        self.assertEqual(list(df['value']), [100, 110, 120])

    def test_chained_slices(self):
        """Test chained slices."""
        result = self.ds[:15][5:10]
        self.assertEqual(len(result), 5)
        df = result.to_df()
        self.assertEqual(list(df['id']), [5, 6, 7, 8, 9])

    # ==================== Comparison with head/tail ====================

    def test_slice_matches_head(self):
        """Test ds[:n] matches ds.head(n)."""
        slice_result = self.ds[:7].to_df()
        head_result = self.ds.head(7).to_df()
        assert_frame_equal(slice_result, head_result)

    def test_slice_matches_limit(self):
        """Test ds[:n] matches ds.limit(n)."""
        slice_result = self.ds[:8].to_df()
        limit_result = self.ds.limit(8).to_df()
        assert_frame_equal(slice_result, limit_result)

    # ==================== Count with Slice ====================

    def test_count_rows_with_slice(self):
        """Test count_rows after slicing."""
        self.assertEqual(self.ds[:10].count_rows(), 10)
        self.assertEqual(self.ds[5:15].count_rows(), 10)
        self.assertEqual(self.ds[10:].count_rows(), 10)

    # ==================== Immutability Tests ====================

    def test_slice_does_not_modify_original(self):
        """Test that slicing creates a new DataStore."""
        original_len = len(self.ds)
        _ = self.ds[:5]
        self.assertEqual(len(self.ds), original_len)

    def test_chained_slice_immutability(self):
        """Test that chained operations don't modify original."""
        ds1 = self.ds[:15]
        ds2 = ds1[:10]
        ds3 = ds2[:5]

        self.assertEqual(len(self.ds), 20)
        self.assertEqual(len(ds1), 15)
        self.assertEqual(len(ds2), 10)
        self.assertEqual(len(ds3), 5)


class TestSliceStyleLimitWithFiles(unittest.TestCase):
    """Test slice-style limit with file-backed DataStore."""

    @classmethod
    def setUpClass(cls):
        """Create test data files."""
        import tempfile

        cls.temp_dir = tempfile.mkdtemp()

        # Create CSV with 50 rows
        cls.csv_file = os.path.join(cls.temp_dir, "data.csv")
        pd.DataFrame(
            {
                'id': list(range(50)),
                'value': [i * 2 for i in range(50)],
            }
        ).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_file_slice_stop_only(self):
        """Test slice on file-backed DataStore."""
        ds = DataStore.from_file(self.csv_file)
        result = ds[:10]
        self.assertEqual(len(result), 10)
        df = result.to_df()
        self.assertEqual(list(df['id']), list(range(10)))

    def test_file_slice_start_and_stop(self):
        """Test slice with offset on file-backed DataStore."""
        ds = DataStore.from_file(self.csv_file)
        result = ds[20:30]
        self.assertEqual(len(result), 10)
        df = result.to_df()
        self.assertEqual(list(df['id']), list(range(20, 30)))

    def test_file_filter_then_slice(self):
        """Test filter then slice on file-backed DataStore."""
        ds = DataStore.from_file(self.csv_file)
        # First filter, then slice - this is the correct order for SQL
        filtered = ds[ds['value'] > 20]
        result = filtered[:10]
        df = result.to_df()
        # Values > 20: 22, 24, 26, 28, 30, 32, 34, 36, 38, 40 (first 10)
        self.assertEqual(len(df), 10)
        # All values should be > 20
        for v in df['value']:
            self.assertGreater(v, 20)

    def test_file_slice_then_filter(self):
        """Test slice then filter on file-backed DataStore.

        This tests pandas-like behavior where:
        - ds[:20] limits to first 20 rows first
        - Then filter is applied on those 20 rows
        - Result should be 9 rows (values 22, 24, 26, 28, 30, 32, 34, 36, 38)
        """
        ds = DataStore.from_file(self.csv_file)
        sliced = ds[:20]
        result = sliced[sliced['value'] > 20]
        df = result.to_df()
        # First 20 rows have values 0,2,4,...,38
        # Values > 20: 22, 24, 26, 28, 30, 32, 34, 36, 38 (9 values)
        self.assertEqual(len(df), 9)
        self.assertEqual(list(df['value']), [22, 24, 26, 28, 30, 32, 34, 36, 38])


class TestMultiLevelLimitFilter(unittest.TestCase):
    """Test multi-level LIMIT-FILTER-LIMIT-FILTER patterns.

    These tests verify that DataStore correctly handles complex chains of
    LIMIT and FILTER operations, matching pandas behavior exactly.
    """

    @classmethod
    def setUpClass(cls):
        """Create temp file with 100 rows for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, 'data.csv')
        pd.DataFrame(
            {
                'id': list(range(100)),
                'value': [i * 2 for i in range(100)],  # 0, 2, 4, ..., 198
            }
        ).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_limit_filter_limit_filter_basic(self):
        """Test: [:50][>60][:10][>75] - Basic 2-level nesting."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:50]
        p = p[p['value'] > 60]
        p = p[:10]
        p = p[p['value'] > 75]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:50]
        ds = ds[ds['value'] > 60]
        ds = ds[:10]
        ds = ds[ds['value'] > 75]
        d = ds.to_df()

        self.assertEqual(list(p['value']), list(d['value']))
        self.assertEqual(list(p['value']), [76, 78, 80])

    def test_limit_filter_limit_filter_limit(self):
        """Test: [:80][>20][:30][<100][:5] - 2-level with trailing limit."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:80]
        p = p[p['value'] > 20]
        p = p[:30]
        p = p[p['value'] < 100]
        p = p[:5]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:80]
        ds = ds[ds['value'] > 20]
        ds = ds[:30]
        ds = ds[ds['value'] < 100]
        ds = ds[:5]
        d = ds.to_df()

        self.assertEqual(list(p['value']), list(d['value']))
        self.assertEqual(list(p['value']), [22, 24, 26, 28, 30])

    def test_three_level_nesting(self):
        """Test: [:70][>10][:50][<140][:30][>40] - 3-level nesting."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:70]
        p = p[p['value'] > 10]
        p = p[:50]
        p = p[p['value'] < 140]
        p = p[:30]
        p = p[p['value'] > 40]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:70]
        ds = ds[ds['value'] > 10]
        ds = ds[:50]
        ds = ds[ds['value'] < 140]
        ds = ds[:30]
        ds = ds[ds['value'] > 40]
        d = ds.to_df()

        expected = [42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_three_level_empty_result(self):
        """Test: [:60][>10][:40][<120][:20][>50] - 3-level with empty result."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:60]
        p = p[p['value'] > 10]
        p = p[:40]
        p = p[p['value'] < 120]
        p = p[:20]
        p = p[p['value'] > 50]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:60]
        ds = ds[ds['value'] > 10]
        ds = ds[:40]
        ds = ds[ds['value'] < 120]
        ds = ds[:20]
        ds = ds[ds['value'] > 50]
        d = ds.to_df()

        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), 0)

    def test_multiple_consecutive_filters(self):
        """Test: [:50][>20][<80][:10][>30] - Multiple filters before limit."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:50]
        p = p[p['value'] > 20]
        p = p[p['value'] < 80]
        p = p[:10]
        p = p[p['value'] > 30]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:50]
        ds = ds[ds['value'] > 20]
        ds = ds[ds['value'] < 80]
        ds = ds[:10]
        ds = ds[ds['value'] > 30]
        d = ds.to_df()

        expected = [32, 34, 36, 38, 40]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_filter_limit_filter_limit_filter(self):
        """Test: [>5][:40][<100][:20][>30] - Filter first pattern."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[pdf['value'] > 5]
        p = p[:40]
        p = p[p['value'] < 100]
        p = p[:20]
        p = p[p['value'] > 30]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[ds['value'] > 5]
        ds = ds[:40]
        ds = ds[ds['value'] < 100]
        ds = ds[:20]
        ds = ds[ds['value'] > 30]
        d = ds.to_df()

        # values: >5 gives 6-198, [:40] gives 6-84, <100 all qualify,
        # [:20] gives 6-44, >30 gives 32-44 = [32, 34, 36, 38, 40, 42, 44]
        expected = [32, 34, 36, 38, 40, 42, 44]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_offset_with_multi_level(self):
        """Test: [10:30][>20][:10][<60] - Offset with multi-level."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[10:30]  # rows 10-29, values 20-58
        p = p[p['value'] > 20]  # values 22-58
        p = p[:10]  # values 22-40
        p = p[p['value'] < 60]  # all qualify

        ds = DataStore.from_file(self.csv_file)
        ds = ds[10:30]
        ds = ds[ds['value'] > 20]
        ds = ds[:10]
        ds = ds[ds['value'] < 60]
        d = ds.to_df()

        expected = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_chain_one_shot_syntax(self):
        """Test one-shot chaining syntax matches step-by-step."""
        ds = DataStore.from_file(self.csv_file)
        # One-shot chain
        result1 = ds[:50][ds['value'] > 60][:10][ds['value'] > 75].to_df()

        # Step-by-step
        ds2 = DataStore.from_file(self.csv_file)
        ds2 = ds2[:50]
        ds2 = ds2[ds2['value'] > 60]
        ds2 = ds2[:10]
        ds2 = ds2[ds2['value'] > 75]
        result2 = ds2.to_df()

        self.assertEqual(list(result1['value']), list(result2['value']))
        self.assertEqual(list(result1['value']), [76, 78, 80])

    def test_dataframe_source_multi_level(self):
        """Test multi-level on DataFrame-backed DataStore."""
        pdf = pd.DataFrame(
            {
                'id': list(range(100)),
                'value': [i * 2 for i in range(100)],
            }
        )

        p = pdf[:50]
        p = p[p['value'] > 60]
        p = p[:10]
        p = p[p['value'] > 75]

        ds = DataStore.from_df(pdf)
        ds = ds[:50]
        ds = ds[ds['value'] > 60]
        ds = ds[:10]
        ds = ds[ds['value'] > 75]
        d = ds.to_df()

        self.assertEqual(list(p['value']), list(d['value']))
        self.assertEqual(list(d['value']), [76, 78, 80])

    def test_order_limit_filter_order_limit_filter(self):
        """Test: ORDER DESC -> LIMIT -> FILTER -> ORDER ASC -> LIMIT -> FILTER.

        This tests complex multi-level nesting with ORDER BY operations.
        """
        pdf = pd.read_csv(self.csv_file)

        # Pandas step by step
        p = pdf.sort_values('value', ascending=False)  # ORDER DESC: 198, 196, ..., 0
        p = p[:30]  # LIMIT 30: 198, 196, ..., 140
        p = p[p['value'] < 180]  # FILTER: 178, 176, ..., 140 (20 rows)
        p = p.sort_values('value', ascending=True)  # ORDER ASC: 140, 142, ..., 178
        p = p[:10]  # LIMIT 10: 140, 142, ..., 158
        p = p[p['value'] > 145]  # FILTER: 146, 148, ..., 158

        ds = DataStore.from_file(self.csv_file)
        ds = ds.sort_values('value', ascending=False)
        ds = ds[:30]
        ds = ds[ds['value'] < 180]
        ds = ds.sort_values('value', ascending=True)
        ds = ds[:10]
        ds = ds[ds['value'] > 145]
        d = ds.to_df()

        expected = [146, 148, 150, 152, 154, 156, 158]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_order_limit_filter_basic(self):
        """Test: ORDER DESC -> LIMIT -> FILTER."""
        pdf = pd.read_csv(self.csv_file)

        p = pdf.sort_values('value', ascending=False)  # 198, 196, ..., 0
        p = p[:20]  # 198, 196, ..., 160
        p = p[p['value'] < 190]  # 188, 186, ..., 160 (15 rows)

        ds = DataStore.from_file(self.csv_file)
        ds = ds.sort_values('value', ascending=False)
        ds = ds[:20]
        ds = ds[ds['value'] < 190]
        d = ds.to_df()

        expected = list(range(188, 158, -2))  # [188, 186, ..., 160]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_filter_order_limit_filter(self):
        """Test: FILTER -> ORDER -> LIMIT -> FILTER."""
        pdf = pd.read_csv(self.csv_file)

        p = pdf[pdf['value'] > 50]  # 52, 54, ..., 198 (74 rows)
        p = p.sort_values('value', ascending=False)  # 198, 196, ..., 52
        p = p[:30]  # 198, 196, ..., 140
        p = p[p['value'] < 170]  # 168, 166, ..., 140 (15 rows)

        ds = DataStore.from_file(self.csv_file)
        ds = ds[ds['value'] > 50]
        ds = ds.sort_values('value', ascending=False)
        ds = ds[:30]
        ds = ds[ds['value'] < 170]
        d = ds.to_df()

        expected = list(range(168, 138, -2))  # [168, 166, ..., 140]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_limit_order_filter_limit(self):
        """Test: LIMIT -> ORDER -> FILTER -> LIMIT."""
        pdf = pd.read_csv(self.csv_file)

        p = pdf[:50]  # 0, 2, ..., 98 (50 rows)
        p = p.sort_values('value', ascending=False)  # 98, 96, ..., 0
        p = p[p['value'] > 30]  # 98, 96, ..., 32 (34 rows)
        p = p[:10]  # 98, 96, ..., 80

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:50]
        ds = ds.sort_values('value', ascending=False)
        ds = ds[ds['value'] > 30]
        ds = ds[:10]
        d = ds.to_df()

        expected = list(range(98, 78, -2))  # [98, 96, ..., 80]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_order_filter_order_limit(self):
        """Test: ORDER -> FILTER -> ORDER -> LIMIT."""
        pdf = pd.read_csv(self.csv_file)

        p = pdf.sort_values('value', ascending=True)  # 0, 2, ..., 198
        p = p[p['value'] > 100]  # 102, 104, ..., 198 (49 rows)
        p = p.sort_values('value', ascending=False)  # 198, 196, ..., 102
        p = p[:15]  # 198, 196, ..., 170

        ds = DataStore.from_file(self.csv_file)
        ds = ds.sort_values('value', ascending=True)
        ds = ds[ds['value'] > 100]
        ds = ds.sort_values('value', ascending=False)
        ds = ds[:15]
        d = ds.to_df()

        expected = list(range(198, 168, -2))  # [198, 196, ..., 170]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_multi_order_with_filters(self):
        """Test multiple ORDER BY with filters interleaved."""
        pdf = pd.read_csv(self.csv_file)

        p = pdf.sort_values('value', ascending=False)[:40]  # Top 40: 198-120
        p = p[p['value'] < 180][:20]  # < 180, take 20: 178-140
        p = p.sort_values('value', ascending=True)  # Sort ASC: 140-178
        p = p[p['value'] > 150]  # > 150: 152-178
        p = p[:8]  # Take 8: 152-166

        ds = DataStore.from_file(self.csv_file)
        ds = ds.sort_values('value', ascending=False)[:40]
        ds = ds[ds['value'] < 180][:20]
        ds = ds.sort_values('value', ascending=True)
        ds = ds[ds['value'] > 150]
        ds = ds[:8]
        d = ds.to_df()

        expected = [152, 154, 156, 158, 160, 162, 164, 166]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)


class TestWindowFunctions(unittest.TestCase):
    """Test window functions: rolling, shift, diff, etc."""

    @classmethod
    def setUpClass(cls):
        """Create temp file with test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, 'data.csv')
        pd.DataFrame(
            {
                'id': list(range(20)),
                'value': [i * 2 for i in range(20)],  # 0, 2, 4, ..., 38
            }
        ).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_rolling_mean(self):
        """Test rolling().mean()"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].rolling(3).mean())

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].rolling(3).mean())

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            if pd.isna(e):
                self.assertTrue(pd.isna(r))
            else:
                self.assertAlmostEqual(e, r)

    def test_rolling_sum(self):
        """Test rolling().sum()"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].rolling(3).sum())

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].rolling(3).sum())

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            if pd.isna(e):
                self.assertTrue(pd.isna(r))
            else:
                self.assertAlmostEqual(e, r)

    def test_rolling_with_min_periods(self):
        """Test rolling() with min_periods"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].rolling(3, min_periods=1).mean())

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].rolling(3, min_periods=1).mean())

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            self.assertAlmostEqual(e, r)

    def test_shift_positive(self):
        """Test shift() with positive periods"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].shift(1))

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].shift(1))

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            if pd.isna(e):
                self.assertTrue(pd.isna(r))
            else:
                self.assertEqual(e, r)

    def test_shift_negative(self):
        """Test shift() with negative periods"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].shift(-1))

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].shift(-1))

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            if pd.isna(e):
                self.assertTrue(pd.isna(r))
            else:
                self.assertEqual(e, r)

    def test_diff(self):
        """Test diff()"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].diff())

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].diff())

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            if pd.isna(e):
                self.assertTrue(pd.isna(r))
            else:
                self.assertEqual(e, r)

    def test_ewm_mean(self):
        """Test ewm().mean()"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].ewm(span=3).mean())

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].ewm(span=3).mean())

        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            self.assertAlmostEqual(e, r, places=2)

    def test_rank(self):
        """Test rank()"""
        pdf = pd.read_csv(self.csv_file)
        expected = list(pdf['value'].rank())

        ds = DataStore.from_file(self.csv_file)
        result = list(ds['value'].rank())

        self.assertEqual(expected, result)

    def test_rolling_in_chain(self):
        """Test rolling() in a chain with LIMIT and FILTER"""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:15].copy()
        p['rolling_mean'] = p['value'].rolling(3).mean()
        p = p.dropna()
        p = p[p['rolling_mean'] > 10]

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:15]
        ds['rolling_mean'] = ds['value'].rolling(3).mean()
        ds = ds.dropna()
        ds = ds[ds['rolling_mean'] > 10]
        d = ds.to_df()

        self.assertEqual(len(p), len(d))
        self.assertEqual(list(p['value']), list(d['value']))


class TestMixedSqlPandasOperations(unittest.TestCase):
    """Test chains with pandas-only operations in the middle.

    These tests verify that SQL operations before a pandas-only function
    are executed first, then pandas-only operations are applied, and
    subsequent operations continue on the resulting DataFrame.
    """

    @classmethod
    def setUpClass(cls):
        """Create temp file with 100 rows for testing."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.csv_file = os.path.join(cls.temp_dir, 'data.csv')
        pd.DataFrame(
            {
                'id': list(range(100)),
                'value': [i * 2 for i in range(100)],  # 0, 2, 4, ..., 198
            }
        ).to_csv(cls.csv_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp directory."""
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_limit_filter_apply_limit_filter(self):
        """Test: LIMIT -> FILTER -> apply() -> LIMIT -> FILTER.

        apply() is pandas-only, so SQL ops before it execute first,
        then apply() runs on DataFrame, then subsequent ops continue.
        """
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:50]  # LIMIT 50: values 0-98
        p = p[p['value'] > 20]  # FILTER: values 22-98
        p = p.copy()  # Avoid SettingWithCopyWarning
        p['value'] = p['value'].apply(lambda x: x + 1000)  # apply: 1022-1098
        p = p[:15]  # LIMIT 15: first 15 transformed values
        p = p[p['value'] < 1060]  # FILTER: values < 1060

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:50]
        ds = ds[ds['value'] > 20]
        ds['value'] = ds['value'].apply(lambda x: x + 1000)
        ds = ds[:15]
        ds = ds[ds['value'] < 1060]
        d = ds.to_df()

        expected = [1022, 1024, 1026, 1028, 1030, 1032, 1034, 1036, 1038, 1040, 1042, 1044, 1046, 1048, 1050]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_order_limit_apply_filter(self):
        """Test: ORDER -> LIMIT -> apply() -> FILTER."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf.sort_values('value', ascending=False)[:30]  # Top 30: 198-140
        p = p.copy()
        p['value'] = p['value'].apply(lambda x: x * 2)  # Double: 396-280
        p = p[p['value'] > 350]  # > 350: 396, 392, ..., 352

        ds = DataStore.from_file(self.csv_file)
        ds = ds.sort_values('value', ascending=False)[:30]
        ds['value'] = ds['value'].apply(lambda x: x * 2)
        ds = ds[ds['value'] > 350]
        d = ds.to_df()

        expected = list(range(396, 350, -4))  # [396, 392, 388, ..., 352]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_filter_apply_order_limit(self):
        """Test: FILTER -> apply() -> ORDER -> LIMIT."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[pdf['value'] > 50]  # > 50: 52-198
        p = p.copy()
        p['value'] = p['value'].apply(lambda x: -x)  # Negate: -52 to -198
        p = p.sort_values('value', ascending=False)[:10]  # Top 10 (least negative)

        ds = DataStore.from_file(self.csv_file)
        ds = ds[ds['value'] > 50]
        ds['value'] = ds['value'].apply(lambda x: -x)
        ds = ds.sort_values('value', ascending=False)[:10]
        d = ds.to_df()

        expected = [-52, -54, -56, -58, -60, -62, -64, -66, -68, -70]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_limit_apply_limit_apply_filter(self):
        """Test: LIMIT -> apply() -> LIMIT -> apply() -> FILTER.

        Multiple pandas-only operations in the chain.
        """
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:40]  # values 0-78
        p = p.copy()
        p['value'] = p['value'].apply(lambda x: x + 100)  # 100-178
        p = p[:20]  # first 20: 100-138
        p = p.copy()
        p['value'] = p['value'].apply(lambda x: x * 2)  # 200-276
        p = p[p['value'] > 250]  # > 250: 252, 256, 260, 264, 268, 272, 276

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:40]
        ds['value'] = ds['value'].apply(lambda x: x + 100)
        ds = ds[:20]
        ds['value'] = ds['value'].apply(lambda x: x * 2)
        ds = ds[ds['value'] > 250]
        d = ds.to_df()

        expected = [252, 256, 260, 264, 268, 272, 276]
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)

    def test_new_column_with_apply_then_filter(self):
        """Test adding a new column with apply() then filtering on it."""
        pdf = pd.read_csv(self.csv_file)
        p = pdf[:30]
        p = p.copy()
        p['category'] = p['value'].apply(lambda x: 'high' if x > 30 else 'low')
        p = p[p['category'] == 'high']

        ds = DataStore.from_file(self.csv_file)
        ds = ds[:30]
        ds['category'] = ds['value'].apply(lambda x: 'high' if x > 30 else 'low')
        ds = ds[ds['category'] == 'high']
        d = ds.to_df()

        self.assertEqual(len(p), len(d))
        self.assertEqual(list(p['value']), list(d['value']))
        # values 32, 34, 36, ..., 58 (14 values)
        self.assertEqual(list(d['value']), list(range(32, 60, 2)))

    def test_dataframe_source_with_apply(self):
        """Test DataFrame-backed DataStore with apply() in chain."""
        pdf = pd.DataFrame(
            {
                'id': list(range(50)),
                'value': [i * 2 for i in range(50)],
            }
        )

        p = pdf[:25]
        p = p.copy()
        p['value'] = p['value'].apply(lambda x: x + 500)
        p = p[p['value'] > 520]

        ds = DataStore.from_df(pdf)
        ds = ds[:25]
        ds['value'] = ds['value'].apply(lambda x: x + 500)
        ds = ds[ds['value'] > 520]
        d = ds.to_df()

        # values: 522, 524, ..., 548 (14 values)
        expected = list(range(522, 550, 2))
        self.assertEqual(list(p['value']), expected)
        self.assertEqual(list(d['value']), expected)


if __name__ == '__main__':
    unittest.main()
