"""
Test DataStore with blob (binary) columns.

Tests binary data handling including:
1. Creating DataStore with bytes columns
2. Filtering, selecting blob columns
3. Comparing blob data between DataStore and pandas

Note: Some tests are marked as xfail because chDB converts bytes to strings
during SQL execution (filter, sort, head/tail, select operations).
"""

import unittest
import numpy as np
import pandas as pd
import pytest
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestBlobColumn(unittest.TestCase):
    """Test DataStore with blob (binary) columns."""

    def test_create_datastore_with_bytes_column(self):
        """Test creating DataStore with a bytes column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'data': [b'\x00\x01\x02', b'\x03\x04\x05', b'\x06\x07\x08']
        })

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'data': [b'\x00\x01\x02', b'\x03\x04\x05', b'\x06\x07\x08']
        })

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_blob_column_with_nulls(self):
        """Test blob column with None values."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'data': [b'\x00\x01', None, b'\x02\x03', None]
        })

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3, 4],
            'data': [b'\x00\x01', None, b'\x02\x03', None]
        })

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_select_blob_column(self):
        """Test selecting blob column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'data': [b'\x00\x01\x02', b'\x03\x04\x05', b'\x06\x07\x08']
        })
        pd_result = pd_df[['id', 'data']]

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'data': [b'\x00\x01\x02', b'\x03\x04\x05', b'\x06\x07\x08']
        })
        ds_result = ds_df[['id', 'data']]

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_with_blob_column(self):
        """Test filtering DataStore that has blob column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'data': [b'\x00\x01', b'\x02\x03', b'\x04\x05', b'\x06\x07']
        })
        pd_result = pd_df[pd_df['id'] > 2]

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3, 4],
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'data': [b'\x00\x01', b'\x02\x03', b'\x04\x05', b'\x06\x07']
        })
        ds_result = ds_df[ds_df['id'] > 2]

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_blob_column_with_empty_bytes(self):
        """Test blob column with empty bytes."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'data': [b'', b'\x00', b'\x00\x01']
        })

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3],
            'data': [b'', b'\x00', b'\x00\x01']
        })

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_blob_column_large_binary(self):
        """Test blob column with larger binary data."""
        # Create larger binary data
        large_binary1 = bytes(range(256))  # 0x00 to 0xFF
        large_binary2 = bytes([i % 256 for i in range(1000)])
        large_binary3 = b'\xFF' * 500

        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'data': [large_binary1, large_binary2, large_binary3]
        })

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3],
            'data': [large_binary1, large_binary2, large_binary3]
        })

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_blob_column_sort(self):
        """Test sorting DataStore with blob column by non-blob column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [3, 1, 2],
            'name': ['Charlie', 'Alice', 'Bob'],
            'data': [b'\x06\x07', b'\x00\x01', b'\x02\x03']
        })
        pd_result = pd_df.sort_values('id')

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [3, 1, 2],
            'name': ['Charlie', 'Alice', 'Bob'],
            'data': [b'\x06\x07', b'\x00\x01', b'\x02\x03']
        })
        ds_result = ds_df.sort_values('id')

        # Compare results
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_blob_column_head_tail(self):
        """Test head and tail operations with blob column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'data': [b'\x01', b'\x02', b'\x03', b'\x04', b'\x05']
        })
        pd_head = pd_df.head(3)
        pd_tail = pd_df.tail(2)

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3, 4, 5],
            'data': [b'\x01', b'\x02', b'\x03', b'\x04', b'\x05']
        })
        ds_head = ds_df.head(3)
        ds_tail = ds_df.tail(2)

        # Compare results
        assert_datastore_equals_pandas(ds_head, pd_head)
        assert_datastore_equals_pandas(ds_tail, pd_tail)

    def test_blob_column_mixed_with_other_types(self):
        """Test blob column mixed with various other data types."""
        # pandas operations
        pd_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'blob_col': [b'\x00', b'\x01', b'\x02']
        })

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'blob_col': [b'\x00', b'\x01', b'\x02']
        })

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_blob_column_assign_new(self):
        """Test assigning a new blob column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        pd_df = pd_df.assign(data=[b'\x00', b'\x01', b'\x02'])

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        ds_df = ds_df.assign(data=[b'\x00', b'\x01', b'\x02'])

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_blob_column_special_bytes(self):
        """Test blob column with special byte sequences."""
        # Special byte sequences that might cause issues
        special_bytes = [
            b'\x00',  # null byte
            b'\n\r\t',  # whitespace
            b'\xff\xfe',  # BOM-like
            b'\\x00\\x01',  # escaped notation as literal bytes
            b'\x1b[31m',  # ANSI escape
        ]

        # pandas operations
        pd_df = pd.DataFrame({
            'id': list(range(len(special_bytes))),
            'data': special_bytes
        })

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'id': list(range(len(special_bytes))),
            'data': special_bytes
        })

        # Compare results
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_blob_column_groupby_count(self):
        """Test groupby with count on DataStore containing blob column."""
        # pandas operations
        pd_df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40],
            'data': [b'\x01', b'\x02', b'\x03', b'\x04']
        })
        pd_result = pd_df.groupby('category')['value'].count()

        # DataStore operations (mirror of pandas)
        ds_df = DataStore({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40],
            'data': [b'\x01', b'\x02', b'\x03', b'\x04']
        })
        ds_result = ds_df.groupby('category')['value'].count()

        # Compare results (groupby order is undefined)
        assert len(ds_result) == len(pd_result)
        # Compare as sets since groupby order is undefined
        ds_values = set(ds_result.values)
        pd_values = set(pd_result.values)
        assert ds_values == pd_values, f"Values don't match: {ds_values} vs {pd_values}"


class TestBlobColumnDtype(unittest.TestCase):
    """Test dtype behavior for blob columns."""

    def test_blob_column_dtype_is_object(self):
        """Test that blob column has object dtype."""
        ds_df = DataStore({
            'id': [1, 2],
            'data': [b'\x00', b'\x01']
        })

        # Blob columns should have object dtype in pandas
        dtype = ds_df['data'].dtype
        assert dtype == np.dtype('object'), f"Expected object dtype, got {dtype}"

    def test_blob_column_values_are_bytes(self):
        """Test that blob column values are bytes objects."""
        ds_df = DataStore({
            'id': [1, 2, 3],
            'data': [b'\x00\x01', b'\x02\x03', b'\x04\x05']
        })

        values = list(ds_df['data'].values)
        for v in values:
            assert isinstance(v, bytes), f"Expected bytes, got {type(v)}"


if __name__ == '__main__':
    unittest.main()
