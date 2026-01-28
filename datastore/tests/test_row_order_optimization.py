"""
Tests for row order preservation optimization.

Verifies that:
1. Parquet files with input_format_parquet_preserve_order=1 skip unnecessary ORDER BY
2. Row order is still correctly preserved without explicit ORDER BY
3. Other sources (Python table, non-parquet files) still use ORDER BY for row preservation
"""

import tempfile
import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from datastore.sql_executor import SQLExecutionEngine
from tests.test_utils import assert_datastore_equals_pandas


class TestRowOrderOptimization:
    """Tests for row order preservation optimization."""

    @pytest.fixture
    def parquet_data(self):
        """Create test parquet file."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'id': np.arange(n),
            'value': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C', 'D'], n)
        })
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            parquet_path = f.name
            df.to_parquet(parquet_path)
        
        yield df, parquet_path
        
        import os
        os.unlink(parquet_path)

    def test_parquet_source_preserves_row_order(self, parquet_data):
        """Test that parquet source correctly reports row order preservation."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        engine = SQLExecutionEngine(ds)
        
        # Verify format settings are correct
        assert ds._format_settings.get('input_format_parquet_preserve_order') == 1
        
        # Verify source_preserves_row_order returns True
        assert engine.source_preserves_row_order() is True

    def test_python_source_does_not_preserve_row_order(self):
        """Test that Python() table function does not report row order preservation."""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        ds = DataStore(df)
        engine = SQLExecutionEngine(ds)
        
        # Python() table uses _row_id via connection.query_df
        # source_preserves_row_order should return False for consistency
        assert engine.source_preserves_row_order() is False

    def test_filter_row_order_preserved_parquet(self, parquet_data):
        """Test that filter operations preserve row order for parquet files."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        ds_filtered = ds[ds['value'] > 0]
        pd_filtered = df[df['value'] > 0]
        
        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_filter_row_order_preserved_python(self):
        """Test that filter operations preserve row order for Python() table."""
        np.random.seed(42)
        df = pd.DataFrame({
            'id': np.arange(100),
            'value': np.random.randn(100)
        })
        
        ds = DataStore(df)
        ds_filtered = ds[ds['value'] > 0]
        pd_filtered = df[df['value'] > 0]
        
        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_multiple_filters_row_order(self, parquet_data):
        """Test row order with chained filters."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        ds_filtered = ds[(ds['value'] > -1) & (ds['value'] < 1)]
        pd_filtered = df[(df['value'] > -1) & (df['value'] < 1)]
        
        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_filter_with_sort_row_order(self, parquet_data):
        """Test that explicit sort overrides default row order."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        ds_result = ds[ds['value'] > 0].sort_values('value')
        pd_result = df[df['value'] > 0].sort_values('value')
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_row_order_preserved(self, parquet_data):
        """Test that head() preserves row order."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        ds_result = ds.head(50)
        pd_result = df.head(50)
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_row_order(self, parquet_data):
        """Test that column selection preserves row order."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        ds_result = ds[['id', 'value']]
        pd_result = df[['id', 'value']]
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_computed_column_row_order(self, parquet_data):
        """Test that computed columns preserve row order."""
        df, parquet_path = parquet_data
        
        ds = DataStore.from_file(parquet_path, 'parquet')
        ds['doubled'] = ds['value'] * 2
        df['doubled'] = df['value'] * 2
        
        ds_result = ds[ds['doubled'] > 0]
        pd_result = df[df['doubled'] > 0]
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTableFunctionPreservesRowOrder:
    """Tests for preserves_row_order method on table functions."""

    def test_file_table_function_parquet(self):
        """Test FileTableFunction.preserves_row_order for parquet."""
        from datastore.table_functions import FileTableFunction
        
        tf = FileTableFunction(path='/tmp/test.parquet', format='parquet')
        
        # Without settings, should return False
        assert tf.preserves_row_order(None) is False
        assert tf.preserves_row_order({}) is False
        
        # With preserve_order=1, should return True
        assert tf.preserves_row_order({'input_format_parquet_preserve_order': 1}) is True
        
        # With preserve_order=0, should return False
        assert tf.preserves_row_order({'input_format_parquet_preserve_order': 0}) is False

    def test_file_table_function_csv(self):
        """Test FileTableFunction.preserves_row_order for CSV."""
        from datastore.table_functions import FileTableFunction
        
        tf = FileTableFunction(path='/tmp/test.csv', format='csv')
        
        # CSV files don't have preserve_order setting
        assert tf.preserves_row_order({'input_format_parquet_preserve_order': 1}) is False

    def test_s3_table_function_parquet(self):
        """Test S3TableFunction.preserves_row_order for parquet."""
        from datastore.table_functions import S3TableFunction
        
        tf = S3TableFunction(path='s3://bucket/test.parquet', format='parquet')
        
        # Without settings, should return False
        assert tf.preserves_row_order(None) is False
        
        # With preserve_order=1, should return True
        assert tf.preserves_row_order({'input_format_parquet_preserve_order': 1}) is True

    def test_numbers_table_function(self):
        """Test NumbersTableFunction.preserves_row_order."""
        from datastore.table_functions import NumbersTableFunction
        
        tf = NumbersTableFunction(limit=100)
        
        # numbers() always preserves order
        assert tf.preserves_row_order(None) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
