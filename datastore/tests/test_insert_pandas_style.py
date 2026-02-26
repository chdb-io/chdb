"""
Test cases for pandas-style DataFrame.insert() method.

The insert() method now supports two modes:
1. Pandas-style: insert(loc, column, value, allow_duplicates=False) - inserts column at position (INPLACE)
2. SQL-style: insert(data, **columns) - inserts rows into table (tested in test_insert_update_delete.py)

This file tests the pandas-style column insertion functionality.

NOTE: pandas insert() is an INPLACE operation that returns None.
DataStore insert() matches this behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore


class TestInsertPandasStyle:
    """Test pandas-compatible insert() for column insertion."""

    def test_insert_basic(self):
        """Test basic column insertion at specified position."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df.insert(1, 'B', [4, 5, 6])

        # DataStore (inplace operation like pandas)
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        ds.insert(1, 'B', [4, 5, 6])

        # Compare
        assert list(ds.columns) == list(pd_df.columns)
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_at_position_zero(self):
        """Test inserting column at the beginning (position 0)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df.insert(0, 'First', [10, 20])

        # DataStore (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))
        ds.insert(0, 'First', [10, 20])

        # Compare
        assert list(ds.columns) == ['First', 'A', 'B']
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_at_end(self):
        """Test inserting column at the end."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df.insert(2, 'Last', [5, 6])

        # DataStore (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2], 'B': [3, 4]}))
        ds.insert(2, 'Last', [5, 6])

        # Compare
        assert list(ds.columns) == ['A', 'B', 'Last']
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_with_scalar_value(self):
        """Test inserting column with a scalar value (broadcast to all rows)."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df.insert(1, 'const', 100)

        # DataStore (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        ds.insert(1, 'const', 100)

        # Compare
        assert list(ds.columns) == list(pd_df.columns)
        assert ds._get_df()['const'].tolist() == [100, 100, 100]

    def test_insert_with_allow_duplicates(self):
        """Test inserting column with duplicate name when allow_duplicates=True."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df.insert(1, 'A', [4, 5, 6], allow_duplicates=True)

        # DataStore (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        ds.insert(1, 'A', [4, 5, 6], allow_duplicates=True)

        # Compare - should have two 'A' columns
        assert list(ds.columns) == ['A', 'A']
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_duplicate_name_raises_error(self):
        """Test that inserting duplicate column name without allow_duplicates raises error."""
        # pandas raises ValueError for duplicate column names
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError):
            pd_df.insert(1, 'A', [4, 5, 6])

        # DataStore should raise the same
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        with pytest.raises(ValueError):
            ds.insert(1, 'A', [4, 5, 6])

    def test_insert_with_series(self):
        """Test inserting a pandas Series as the value."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df.insert(1, 'B', pd.Series([4, 5, 6]))

        # DataStore (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        ds.insert(1, 'B', pd.Series([4, 5, 6]))

        # Compare
        assert list(ds.columns) == list(pd_df.columns)
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_with_numpy_array(self):
        """Test inserting a numpy array as the value."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_df.insert(1, 'B', np.array([4, 5, 6]))

        # DataStore (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        ds.insert(1, 'B', np.array([4, 5, 6]))

        # Compare
        assert list(ds.columns) == list(pd_df.columns)
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_returns_none(self):
        """Test that insert returns None (like pandas)."""
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        result = ds.insert(1, 'B', [4, 5, 6])

        # insert() returns None (like pandas)
        assert result is None
        # But the DataStore is modified in place
        assert list(ds.columns) == ['A', 'B']

    def test_insert_invalid_loc_type_raises_type_detection(self):
        """Test that non-int first argument falls back to SQL mode."""
        # When first arg is not int, it's treated as SQL mode
        # which requires table_name
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        with pytest.raises(ValueError, match="Table name required"):
            ds.insert([{"id": 1}])  # List triggers SQL mode, but no table

    def test_insert_missing_column_raises_error(self):
        """Test that missing column argument raises TypeError."""
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        with pytest.raises(TypeError, match="missing required argument: 'column'"):
            ds.insert(1, None, [4, 5, 6])

    def test_insert_missing_value_raises_error(self):
        """Test that missing value argument raises TypeError."""
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3]}))
        with pytest.raises(TypeError, match="missing required argument: 'value'"):
            ds.insert(1, 'B', None)


class TestInsertPandasStyleChained:
    """Test chained insert operations."""

    def test_insert_multiple_columns_sequential(self):
        """Test inserting multiple columns sequentially."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2]})
        pd_df.insert(1, 'B', [3, 4])
        pd_df.insert(2, 'C', [5, 6])

        # DataStore - sequential (inplace)
        ds = DataStore(pd.DataFrame({'A': [1, 2]}))
        ds.insert(1, 'B', [3, 4])
        ds.insert(2, 'C', [5, 6])

        # Compare
        assert list(ds.columns) == ['A', 'B', 'C']
        assert ds._get_df().values.tolist() == pd_df.values.tolist()

    def test_insert_then_filter(self):
        """Test inserting a column then filtering on it."""
        # pandas
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        pd_df.insert(1, 'B', [10, 20, 30, 40])
        pd_result = pd_df[pd_df['B'] > 20]

        # DataStore (inplace then filter)
        ds = DataStore(pd.DataFrame({'A': [1, 2, 3, 4]}))
        ds.insert(1, 'B', [10, 20, 30, 40])
        ds_result = ds[ds['B'] > 20]

        # Compare
        assert list(ds_result.columns) == list(pd_result.columns)
        # Check row count (should have rows where B > 20)
        assert len(ds_result) == len(pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
