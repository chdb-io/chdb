"""
Tests for file path auto-detection feature.

This feature allows users to pass file paths directly to DataStore:
    ds = DataStore('/path/to/data.parquet')  # Works!
    ds = DataStore('data.csv')               # Works!

Instead of the verbose format:
    ds = DataStore('file', path='/path/to/data.parquet')
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datastore import DataStore
from tests.test_utils import get_series


@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.randint(1, 100, 100),
        'score': np.random.uniform(0, 100, 100),
    })


@pytest.fixture
def parquet_file(sample_data, tmp_path):
    """Create a temporary parquet file."""
    path = tmp_path / "test_data.parquet"
    sample_data.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def csv_file(sample_data, tmp_path):
    """Create a temporary CSV file."""
    path = tmp_path / "test_data.csv"
    sample_data.to_csv(path, index=False)
    return str(path)


class TestFilePathAutoDetection:
    """Tests for automatic file path detection in DataStore.__init__."""
    
    def test_parquet_absolute_path(self, parquet_file, sample_data):
        """Test that absolute parquet path is auto-detected."""
        ds = DataStore(parquet_file)
        
        assert ds._table_function is not None
        assert ds.source_type == 'file'
        assert list(ds.columns) == list(sample_data.columns)
        assert ds.shape == sample_data.shape
    
    def test_csv_absolute_path(self, csv_file, sample_data):
        """Test that absolute CSV path is auto-detected."""
        ds = DataStore(csv_file)
        
        assert ds._table_function is not None
        assert ds.source_type == 'file'
        assert list(ds.columns) == list(sample_data.columns)
        assert ds.shape == sample_data.shape
    
    def test_filter_on_parquet(self, parquet_file, sample_data):
        """Test filtering operations on auto-detected parquet file."""
        ds = DataStore(parquet_file)
        pd_df = sample_data
        
        # Filter
        ds_result = ds[ds['value'] > 50]
        pd_result = pd_df[pd_df['value'] > 50]
        
        assert ds_result.shape == pd_result.shape
    
    def test_chained_operations_on_parquet(self, parquet_file, sample_data):
        """Test chained operations on auto-detected parquet file."""
        ds = DataStore(parquet_file)
        pd_df = sample_data
        
        # Chain: filter -> sort -> head
        ds_result = ds[ds['value'] > 30]
        ds_result = ds_result.sort_values('score', ascending=False)
        ds_result = ds_result.head(10)
        
        pd_result = pd_df[pd_df['value'] > 30]
        pd_result = pd_result.sort_values('score', ascending=False)
        pd_result = pd_result.head(10)
        
        assert ds_result.shape == pd_result.shape
        # Check values match (sorted by score descending)
        assert list(ds_result['id']) == list(pd_result['id'])
    
    def test_groupby_on_parquet(self, parquet_file, sample_data):
        """Test groupby operations on auto-detected parquet file."""
        ds = DataStore(parquet_file)
        pd_df = sample_data
        
        # GroupBy sum
        ds_result = ds.groupby('category')['value'].sum()
        pd_result = pd_df.groupby('category')['value'].sum()
        
        ds_series = get_series(ds_result)
        
        # Compare sums (order may differ)
        ds_sums = set(ds_series.values)
        pd_sums = set(pd_result.values)
        
        assert ds_sums == pd_sums
    
    def test_explicit_file_source_still_works(self, parquet_file, sample_data):
        """Test that explicit 'file' source type still works."""
        ds = DataStore('file', path=parquet_file)
        
        assert ds._table_function is not None
        assert ds.source_type == 'file'
        assert ds.shape == sample_data.shape
    
    def test_dataframe_source_unaffected(self, sample_data):
        """Test that DataFrame source is not affected by the change."""
        ds = DataStore(sample_data)
        
        assert ds.source_type == 'dataframe'
        assert ds.shape == sample_data.shape
    
    def test_known_source_types_unaffected(self):
        """Test that known source types are not auto-converted."""
        # These should NOT be treated as file paths
        known_types = ['chdb', 'file', 's3', 'mysql', 'clickhouse', 'remote']
        
        for src_type in known_types:
            ds = DataStore(src_type)
            assert ds.source_type == src_type, f"'{src_type}' should not be auto-converted"


class TestFilePathExtensions:
    """Test various file extensions for auto-detection."""
    
    @pytest.mark.parametrize("extension", [
        '.parquet',
        '.csv',
        '.tsv',
        '.json',
        '.jsonl',
    ])
    def test_extension_detection(self, sample_data, tmp_path, extension):
        """Test that various file extensions trigger auto-detection."""
        path = tmp_path / f"test_data{extension}"
        
        # Write file based on extension
        if extension == '.parquet':
            sample_data.to_parquet(path, index=False)
        elif extension in ('.csv', '.tsv'):
            sep = '\t' if extension == '.tsv' else ','
            sample_data.to_csv(path, index=False, sep=sep)
        elif extension in ('.json', '.jsonl'):
            orient = 'records' if extension == '.jsonl' else 'records'
            lines = extension == '.jsonl'
            sample_data.to_json(path, orient=orient, lines=lines)
        
        ds = DataStore(str(path))
        
        assert ds._table_function is not None, f"Extension {extension} should trigger auto-detection"
        assert ds.source_type == 'file'


class TestPerformanceWithFilePath:
    """Performance-related tests for file path auto-detection."""
    
    def test_predicate_pushdown(self, sample_data, tmp_path):
        """Test that predicate pushdown works with auto-detected paths."""
        # Create larger dataset
        large_data = pd.concat([sample_data] * 100, ignore_index=True)
        path = tmp_path / "large_data.parquet"
        large_data.to_parquet(path, index=False)
        
        ds = DataStore(str(path))
        
        # High selectivity filter
        result = ds[(ds['category'] == 'A') & (ds['value'] > 90)]
        
        expected = large_data[(large_data['category'] == 'A') & (large_data['value'] > 90)]
        
        assert result.shape == expected.shape
