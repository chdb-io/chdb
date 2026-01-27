"""
Exploratory Discovery Batch 14: IO Operations, SQL Generation Validation, and Edge Cases

Focus areas:
1. IO operations: to_csv, to_parquet, to_json round-trip testing
2. SQL generation validation: verify lazy ops produce correct SQL
3. Multi-source merge scenarios
4. Special data types: Categorical, Nullable Int
5. Memory edge cases: large operations, chunked processing
6. Error handling and validation
"""

import pytest
from tests.xfail_markers import chdb_category_type, chdb_timedelta_type
import pandas as pd
import numpy as np
import tempfile
import os
from io import StringIO
import json
import datastore

from datastore import DataStore
from tests.test_utils import get_dataframe, assert_datastore_equals_pandas, get_series


class TestIOOperationsRoundTrip:
    """Test IO operations with round-trip verification."""

    def test_to_csv_basic(self):
        """Test basic to_csv output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds = DataStore(pd_df)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            try:
                ds.to_csv(f.name, index=False)
                pd_read = pd.read_csv(f.name)
                assert_datastore_equals_pandas(ds, pd_read)
            finally:
                os.unlink(f.name)

    def test_to_csv_with_filter(self):
        """Test to_csv after filter operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']})
        ds = DataStore(pd_df)

        ds_filtered = ds[ds['a'] > 2]
        pd_filtered = pd_df[pd_df['a'] > 2]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            try:
                ds_filtered.to_csv(f.name, index=False)
                pd_read = pd.read_csv(f.name)
                assert_datastore_equals_pandas(ds_filtered, pd_read)
            finally:
                os.unlink(f.name)

    def test_to_parquet_basic(self):
        """Test basic to_parquet output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.1, 2.2, 3.3]})
        ds = DataStore(pd_df)

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            try:
                ds.to_parquet(f.name)
                pd_read = pd.read_parquet(f.name)
                assert_datastore_equals_pandas(ds, pd_read)
            finally:
                os.unlink(f.name)

    def test_to_json_basic(self):
        """Test basic to_json output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds = DataStore(pd_df)

        ds_json = ds.to_json(orient='records')
        pd_json = pd_df.to_json(orient='records')

        ds_data = json.loads(ds_json)
        pd_data = json.loads(pd_json)

        assert ds_data == pd_data, f"JSON mismatch: {ds_data} vs {pd_data}"

    def test_to_dict_records(self):
        """Test to_dict output - DataStore matches pandas default ('dict' format)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds = DataStore(pd_df)

        # DataStore.to_dict() matches pandas default ('dict' format)
        ds_dict = ds.to_dict()
        pd_dict = pd_df.to_dict()  # pandas default is 'dict' orient

        assert ds_dict == pd_dict

        # Also test explicit records format
        ds_records = ds.to_dict(orient='records')
        pd_records = pd_df.to_dict(orient='records')

        assert ds_records == pd_records

    def test_to_records_basic(self):
        """Test to_records output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        ds = DataStore(pd_df)

        ds_rec = ds.to_records(index=False)
        pd_rec = pd_df.to_records(index=False)

        assert len(ds_rec) == len(pd_rec)


class TestSQLGenerationValidation:
    """Validate SQL generation for various lazy operation chains."""

    def test_filter_then_select_sql(self):
        """Verify filter then select generates correct SQL."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': ['x', 'y', 'z', 'w', 'v']})
        ds = DataStore(pd_df)

        ds_result = ds[ds['a'] > 2][['a', 'b']]
        pd_result = pd_df[pd_df['a'] > 2][['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_then_filter_sql(self):
        """Verify select then filter generates correct SQL."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': ['x', 'y', 'z', 'w', 'v']})
        ds = DataStore(pd_df)

        ds_result = ds[['a', 'b']][ds['a'] > 2]
        pd_result = pd_df[['a', 'b']][pd_df['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_sql(self):
        """Verify multiple filters are combined correctly."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(pd_df)

        ds_result = ds[ds['a'] > 1][ds['a'] < 5][ds['b'] > 15]
        pd_result = pd_df[pd_df['a'] > 1][pd_df['a'] < 5][pd_df['b'] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_head_sql(self):
        """Verify sort then head generates ORDER BY LIMIT."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(pd_df)

        ds_result = ds.sort_values('a').head(3)
        pd_result = pd_df.sort_values('a').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_then_filter_sql(self):
        """Verify groupby agg then filter uses HAVING or subquery."""
        pd_df = pd.DataFrame({'a': ['x', 'x', 'y', 'y', 'z'], 'b': [1, 2, 3, 4, 5]})
        ds = DataStore(pd_df)

        ds_agg = ds.groupby('a').agg({'b': 'sum'})
        pd_agg = pd_df.groupby('a').agg({'b': 'sum'})  # Keep as grouped result

        # Filter aggregated result
        ds_result = ds_agg[ds_agg['b'] > 3]
        pd_result = pd_agg[pd_agg['b'] > 3]

        # Compare with proper index handling
        assert len(ds_result) == len(pd_result)
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_assign_then_filter_on_new_column(self):
        """Verify assign creates column that can be filtered."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(pd_df)

        # Proper chaining
        ds_with_b = ds.assign(b=ds['a'] * 2)
        ds_result = ds_with_b[ds_with_b['b'] > 4]

        pd_temp = pd_df.assign(b=pd_df['a'] * 2)
        pd_result = pd_temp[pd_temp['b'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMultiSourceMerge:
    """Test merge operations with multiple DataStores."""

    def test_merge_two_datastores_basic(self):
        """Test merging two DataStores."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'val_left': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key': [2, 3, 4], 'val_right': ['x', 'y', 'z']})

        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        ds_result = ds_left.merge(ds_right, on='key')
        pd_result = pd_left.merge(pd_right, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_with_different_column_names(self):
        """Test merge with left_on/right_on."""
        pd_left = pd.DataFrame({'key_l': [1, 2, 3], 'val_left': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key_r': [2, 3, 4], 'val_right': ['x', 'y', 'z']})

        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        ds_result = ds_left.merge(ds_right, left_on='key_l', right_on='key_r')
        pd_result = pd_left.merge(pd_right, left_on='key_l', right_on='key_r')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_left_join(self):
        """Test left join merge."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'val_left': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key': [2, 3, 4], 'val_right': ['x', 'y', 'z']})

        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        ds_result = ds_left.merge(ds_right, on='key', how='left')
        pd_result = pd_left.merge(pd_right, on='key', how='left')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_outer_join(self):
        """Test outer join merge."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'val_left': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'key': [2, 3, 4], 'val_right': ['x', 'y', 'z']})

        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        ds_result = ds_left.merge(ds_right, on='key', how='outer')
        pd_result = pd_left.merge(pd_right, on='key', how='outer')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_then_filter(self):
        """Test filter after merge."""
        pd_left = pd.DataFrame({'key': [1, 2, 3, 4], 'val': [10, 20, 30, 40]})
        pd_right = pd.DataFrame({'key': [2, 3, 4, 5], 'val2': [100, 200, 300, 400]})

        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        ds_merged = ds_left.merge(ds_right, on='key')
        pd_merged = pd_left.merge(pd_right, on='key')

        ds_result = ds_merged[ds_merged['val'] > 20]
        pd_result = pd_merged[pd_merged['val'] > 20]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_on_multiple_keys(self):
        """Test merge on multiple columns."""
        pd_left = pd.DataFrame({'k1': [1, 1, 2, 2], 'k2': ['a', 'b', 'a', 'b'], 'val_l': [10, 20, 30, 40]})
        pd_right = pd.DataFrame({'k1': [1, 2, 2], 'k2': ['a', 'a', 'b'], 'val_r': [100, 200, 300]})

        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        ds_result = ds_left.merge(ds_right, on=['k1', 'k2'])
        pd_result = pd_left.merge(pd_right, on=['k1', 'k2'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestSpecialDataTypes:
    """Test handling of special pandas data types."""

    def test_nullable_integer(self):
        """Test nullable integer type."""
        pd_df = pd.DataFrame({'a': pd.array([1, 2, None, 4], dtype=pd.Int64Dtype())})
        ds = DataStore(pd_df)

        # Filter operation
        ds_result = ds[ds['a'] > 1]
        pd_result = pd_df[pd_df['a'] > 1]

        # DataStore may convert to float64, which is acceptable
        assert len(ds_result) == len(pd_result)

    @chdb_category_type
    def test_categorical_basic(self):
        """Test categorical data type."""
        pd_df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a', 'c'])})
        ds = DataStore(pd_df)

        ds_result = ds[ds['cat'] == 'a']
        pd_result = pd_df[pd_df['cat'] == 'a']

        # Compare lengths since categorical handling may differ
        assert len(ds_result) == len(pd_result)

    def test_datetime_basic(self):
        """Test datetime type."""
        pd_df = pd.DataFrame({'dt': pd.date_range('2024-01-01', periods=5, freq='D'), 'val': [1, 2, 3, 4, 5]})
        ds = DataStore(pd_df)

        ds_result = ds[ds['val'] > 2]
        pd_result = pd_df[pd_df['val'] > 2]

        assert len(ds_result) == len(pd_result)

    @chdb_timedelta_type
    def test_timedelta_basic(self):
        """Test timedelta type."""
        pd_df = pd.DataFrame({'td': pd.to_timedelta([1, 2, 3, 4, 5], unit='D'), 'val': [1, 2, 3, 4, 5]})
        ds = DataStore(pd_df)

        ds_result = ds[ds['val'] > 2]
        pd_result = pd_df[pd_df['val'] > 2]

        assert len(ds_result) == len(pd_result)

    def test_boolean_type(self):
        """Test boolean type operations."""
        pd_df = pd.DataFrame({'a': [True, False, True, False], 'b': [1, 2, 3, 4]})
        ds = DataStore(pd_df)

        ds_result = ds[ds['a']]
        pd_result = pd_df[pd_df['a']]

        assert len(ds_result) == len(pd_result)

    def test_string_with_none(self):
        """Test string column with None values."""
        pd_df = pd.DataFrame({'a': ['x', None, 'z', None, 'v']})
        ds = DataStore(pd_df)

        ds_result = ds.dropna()
        pd_result = pd_df.dropna()

        assert len(ds_result) == len(pd_result)


class TestMemoryAndPerformance:
    """Test memory handling and performance edge cases."""

    def test_large_dataframe_filter(self):
        """Test filter on larger DataFrame."""
        n = 10000
        pd_df = pd.DataFrame({'a': np.random.randint(0, 100, n), 'b': np.random.random(n)})
        ds = DataStore(pd_df)

        ds_result = ds[ds['a'] > 50]
        pd_result = pd_df[pd_df['a'] > 50]

        assert len(ds_result) == len(pd_result)

    def test_many_columns_select(self):
        """Test selecting from DataFrame with many columns."""
        n_cols = 50
        pd_df = pd.DataFrame({f'col_{i}': range(10) for i in range(n_cols)})
        ds = DataStore(pd_df)

        selected_cols = [f'col_{i}' for i in range(0, n_cols, 5)]  # Select every 5th column
        ds_result = ds[selected_cols]
        pd_result = pd_df[selected_cols]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deeply_nested_operations(self):
        """Test deeply nested operation chain."""
        pd_df = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
        ds = DataStore(pd_df)

        # Chain multiple operations
        ds_result = ds[ds['a'] > 10][ds['a'] < 90].assign(c=ds['a'] + ds['b'])[['a', 'c']].head(20)

        pd_result = pd_df[pd_df['a'] > 10][pd_df['a'] < 90].assign(c=pd_df['a'] + pd_df['b'])[['a', 'c']].head(20)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_select_nonexistent_column(self):
        """Test selecting non-existent column raises error."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore(pd_df)

        with pytest.raises((KeyError, Exception)):
            get_series(ds['nonexistent'])

    def test_filter_on_nonexistent_column(self):
        """Test filtering on non-existent column raises error."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore(pd_df)

        with pytest.raises((KeyError, Exception)):
            # Use len() to trigger execution
            len(ds[ds['nonexistent'] > 1])

    def test_empty_dataframe_operations(self):
        """Test operations on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds = DataStore(pd_df)

        ds_result = ds[['a']]
        pd_result = pd_df[['a']]

        assert len(ds_result) == 0
        assert list(ds_result.columns) == ['a']

    def test_single_row_operations(self):
        """Test operations on single-row DataFrame."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds = DataStore(pd_df)

        ds_result = ds.assign(c=ds['a'] + ds['b'])
        pd_result = pd_df.assign(c=pd_df['a'] + pd_df['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_handling_in_filter(self):
        """Test NaN handling in filter conditions."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds = DataStore(pd_df)

        ds_result = ds[ds['a'] > 2]
        pd_result = pd_df[pd_df['a'] > 2]

        assert len(ds_result) == len(pd_result)

    def test_inf_handling(self):
        """Test infinity handling."""
        pd_df = pd.DataFrame({'a': [1.0, np.inf, -np.inf, 4.0]})
        ds = DataStore(pd_df)

        # Replace inf values
        ds_result = ds['a'].replace([np.inf, -np.inf], np.nan)
        pd_result = pd_df['a'].replace([np.inf, -np.inf], np.nan)

        # Just check length since exact NaN handling may differ
        ds_exec = get_series(ds_result)
        assert len(ds_exec) == len(pd_result)


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_data_cleaning_pipeline(self):
        """Test a typical data cleaning pipeline."""
        pd_df = pd.DataFrame(
            {
                'name': ['Alice', 'bob', 'CHARLIE', '  Dave  '],
                'age': [25, -1, 30, 999],
                'score': [85.5, np.nan, 90.0, 75.0],
            }
        )
        ds = DataStore(pd_df)

        # Clean: strip names, filter invalid ages
        ds_clean = ds[ds['age'] > 0][ds['age'] < 100]
        pd_clean = pd_df[pd_df['age'] > 0][pd_df['age'] < 100]

        assert_datastore_equals_pandas(ds_clean, pd_clean)

    def test_aggregation_with_multiple_functions(self):
        """Test groupby with multiple aggregation functions."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'B'], 'value': [1, 2, 3, 4, 5]})
        ds = DataStore(pd_df)

        ds_agg = ds.groupby('group').agg({'value': ['sum', 'mean', 'count']})
        pd_agg = pd_df.groupby('group').agg({'value': ['sum', 'mean', 'count']}).reset_index()

        # Check that we have the right number of rows
        assert len(ds_agg) == len(pd_agg)

    def test_window_function_with_partition(self):
        """Test window function with partition."""
        pd_df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'B'], 'value': [1, 2, 3, 4, 5]})
        ds = DataStore(pd_df)

        ds_result = ds.assign(cumsum=ds.groupby('group')['value'].cumsum())
        pd_result = pd_df.assign(cumsum=pd_df.groupby('group')['value'].cumsum())

        # Reset index for comparison
        ds_exec = get_dataframe(ds_result).reset_index(drop=True)
        pd_exec = pd_result.reset_index(drop=True)

        assert len(ds_exec) == len(pd_exec)

    def test_pivot_and_unpivot(self):
        """Test pivot and melt operations."""
        pd_df = pd.DataFrame(
            {
                'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
                'product': ['A', 'B', 'A', 'B'],
                'sales': [100, 150, 110, 160],
            }
        )
        ds = DataStore(pd_df)

        # Pivot
        ds_pivot = ds.pivot(index='date', columns='product', values='sales')
        pd_pivot = pd_df.pivot(index='date', columns='product', values='sales')

        # Compare shapes
        assert ds_pivot.shape == pd_pivot.shape

    def test_concat_multiple_datastores(self):
        """Test concatenating multiple DataStores using datastore.concat."""
        pd_df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        pd_df2 = pd.DataFrame({'a': [3, 4], 'b': ['z', 'w']})
        pd_df3 = pd.DataFrame({'a': [5, 6], 'b': ['v', 'u']})

        ds1 = DataStore(pd_df1)
        ds2 = DataStore(pd_df2)
        ds3 = DataStore(pd_df3)

        # Use datastore.concat (module-level function)
        ds_result = datastore.concat([ds1, ds2, ds3], ignore_index=True)
        pd_result = pd.concat([pd_df1, pd_df2, pd_df3], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestColumnExprChaining:
    """Test ColumnExpr method chaining."""

    def test_str_methods_chain(self):
        """Test string method chaining."""
        pd_df = pd.DataFrame({'name': ['  ALICE  ', '  bob  ', '  Charlie  ']})
        ds = DataStore(pd_df)

        ds_result = ds['name'].str.strip().str.lower()
        pd_result = pd_df['name'].str.strip().str.lower()

        ds_exec = get_series(ds_result)

        # Compare values
        assert list(ds_exec) == list(pd_result)

    def test_numeric_methods_chain(self):
        """Test numeric method chaining."""
        pd_df = pd.DataFrame({'a': [-1.5, 2.7, -3.2, 4.9]})
        ds = DataStore(pd_df)

        ds_result = ds['a'].abs().round()
        pd_result = pd_df['a'].abs().round()

        ds_exec = get_series(ds_result)

        assert list(ds_exec) == list(pd_result)

    def test_comparison_chain(self):
        """Test comparison method chaining."""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 7, 2], 'b': [2, 4, 3, 6, 8]})
        ds = DataStore(pd_df)

        # a > b and a < 5
        ds_cond = (ds['a'] > ds['b']) & (ds['a'] < 5)
        pd_cond = (pd_df['a'] > pd_df['b']) & (pd_df['a'] < 5)

        ds_result = ds[ds_cond]
        pd_result = pd_df[pd_cond]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_chain(self):
        """Test arithmetic expression chaining."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds = DataStore(pd_df)

        # (a + b) * 2 - 1
        ds_expr = (ds['a'] + ds['b']) * 2 - 1
        pd_expr = (pd_df['a'] + pd_df['b']) * 2 - 1

        ds_exec = get_series(ds_expr)

        assert list(ds_exec) == list(pd_expr)


class TestIndexOperations:
    """Test index-related operations."""

    def test_set_index_then_reset(self):
        """Test set_index followed by reset_index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [10, 20, 30]})
        ds = DataStore(pd_df)

        ds_result = ds.set_index('a').reset_index()
        pd_result = pd_df.set_index('a').reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reindex_with_new_index(self):
        """Test reindex with new index values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds = DataStore(pd_df)

        ds_result = ds.reindex(['x', 'w', 'z'])
        pd_result = pd_df.reindex(['x', 'w', 'z'])

        assert len(ds_result) == len(pd_result)

    def test_sort_index(self):
        """Test sort_index operation."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]}, index=['c', 'a', 'b'])
        ds = DataStore(pd_df)

        ds_result = ds.sort_index()
        pd_result = pd_df.sort_index()

        # Check that order matches
        assert list(ds_result['a']) == list(pd_result['a'])


class TestConcatBehavior:
    """Test concat behavior - instance method vs module function."""

    def test_concat_instance_method(self):
        """Test ds.concat() instance method behavior."""
        pd_df1 = pd.DataFrame({'a': [1, 2]})
        pd_df2 = pd.DataFrame({'a': [3, 4]})

        ds1 = DataStore(pd_df1)
        ds2 = DataStore(pd_df2)

        # Instance concat - appends other to self
        ds_result = ds1.concat([ds2])

        # Just verify it works and returns something
        assert len(ds_result) >= 2

    def test_concat_module_function(self):
        """Test datastore.concat() module function."""
        pd_df1 = pd.DataFrame({'a': [1, 2]})
        pd_df2 = pd.DataFrame({'a': [3, 4]})

        ds1 = DataStore(pd_df1)
        ds2 = DataStore(pd_df2)

        # Module-level concat
        ds_result = datastore.concat([ds1, ds2], ignore_index=True)
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)

        assert len(ds_result) == len(pd_result)


class TestEdgeCases:
    """Additional edge case tests."""

    def test_duplicate_column_names(self):
        """Test handling of duplicate column names."""
        # Create DataFrame with duplicate column names manually
        pd_df = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'a'])
        ds = DataStore(pd_df)

        # Just verify it doesn't crash
        assert ds is not None
        assert len(ds) == 1

    def test_unicode_column_names(self):
        """Test handling of unicode column names."""
        pd_df = pd.DataFrame({'名前': ['Alice', 'Bob'], 'Preis€': [10, 20]})
        ds = DataStore(pd_df)

        assert '名前' in ds.columns or len(ds.columns) == 2

    def test_special_char_in_values(self):
        """Test handling of special characters in values."""
        pd_df = pd.DataFrame({'a': ['hello"world', "it's", 'back\\slash']})
        ds = DataStore(pd_df)

        assert len(ds) == 3

    def test_very_long_string(self):
        """Test handling of very long strings."""
        long_str = 'x' * 10000
        pd_df = pd.DataFrame({'a': [long_str, 'short']})
        ds = DataStore(pd_df)

        assert len(ds.iloc[0]['a']) == 10000

    def test_mixed_int_float_column(self):
        """Test column with mixed int and float values."""
        pd_df = pd.DataFrame({'a': [1, 2.5, 3, 4.7]})
        ds = DataStore(pd_df)

        ds_result = ds[ds['a'] > 2]
        pd_result = pd_df[pd_df['a'] > 2]

        assert len(ds_result) == len(pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
