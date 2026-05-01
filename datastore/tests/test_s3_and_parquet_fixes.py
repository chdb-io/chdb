"""
Tests for fixes to None format handling and read_parquet S3 support.

Regression tests for:
- DataStore.from_s3() without explicit format crashing with 'NoneType' has no attr 'lower'
- DataStore.from_file() without explicit format crashing with 'NoneType' has no attr 'lower'
- read_parquet('s3://...') routing to S3 table function
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore, read_parquet
from datastore.table_functions import S3TableFunction, FileTableFunction


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'value': [10, 20, 30, 40, 50],
    })


@pytest.fixture
def parquet_file(sample_df, tmp_path):
    path = tmp_path / "test_data.parquet"
    sample_df.to_parquet(path, index=False)
    return str(path)


class TestPreservesRowOrderNoneFormat:
    """
    preserves_row_order() crashed with 'NoneType' has no attribute 'lower'
    when format=None was stored in params (dict.get("format", "") returns None
    when the key exists with value None, not the default "").
    """

    def test_s3_table_function_preserves_row_order_no_format(self):
        """S3TableFunction.preserves_row_order() must not crash when format is None."""
        tf = S3TableFunction(url="s3://bucket/data.parquet", nosign=True)
        # format key exists with value None in params
        assert tf.params.get("format") is None
        # must not raise
        result = tf.preserves_row_order()
        assert isinstance(result, bool)

    def test_file_table_function_preserves_row_order_no_format(self):
        """FileTableFunction.preserves_row_order() must not crash when format is None."""
        tf = FileTableFunction(path="/some/file.parquet", format=None)
        assert tf.params.get("format") is None
        result = tf.preserves_row_order()
        assert isinstance(result, bool)

    def test_from_s3_no_format_does_not_crash(self):
        """DataStore.from_s3() without format must not crash on creation."""
        ds = DataStore.from_s3("s3://bucket/data.parquet", nosign=True)
        assert ds is not None
        assert ds.source_type == "s3"
        assert ds._table_function is not None
        assert ds._table_function.params.get("format") is None

    def test_from_file_no_format_does_not_crash(self, parquet_file):
        """DataStore.from_file() without format must not crash on creation."""
        ds = DataStore.from_file(parquet_file)
        assert ds is not None
        assert ds._table_function is not None

    def test_from_file_no_format_reads_data(self, parquet_file, sample_df):
        """DataStore.from_file() without format must correctly read parquet data."""
        ds = DataStore.from_file(parquet_file)
        pd_result = sample_df

        assert list(ds.columns) == list(pd_result.columns)
        assert len(ds) == len(pd_result)


class TestReadParquetS3:
    """read_parquet('s3://...') must route to S3 table function, not from_file."""

    def test_read_parquet_s3_url_creates_s3_datastore(self):
        """read_parquet with s3:// path must create an S3-backed DataStore."""
        ds = read_parquet("s3://bucket/data.parquet")
        assert ds is not None
        assert ds.source_type == "s3"
        assert isinstance(ds._table_function, S3TableFunction)

    def test_read_parquet_s3_url_sets_parquet_format(self):
        """read_parquet with s3:// path must pass format=Parquet to the table function."""
        ds = read_parquet("s3://bucket/data.parquet")
        assert ds._table_function.params.get("format") == "Parquet"

    def test_read_parquet_local_file_reads_data(self, parquet_file, sample_df):
        """read_parquet with a local file path must correctly read data."""
        ds = read_parquet(parquet_file)
        pd_result = sample_df

        assert list(ds.columns) == list(pd_result.columns)
        assert len(ds) == len(pd_result)
