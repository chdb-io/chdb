"""
Exploratory Batch 55: IO Output, Metadata Methods, Complex Sort Operations

Focus areas:
1. IO output methods (to_csv, to_json, to_parquet) + lazy operation chains
2. Metadata methods (describe, info, memory_usage, convert_dtypes) + lazy chains
3. select_dtypes edge cases
4. Complex sort operations (multi-column, na_position, key parameter)
5. Index operations with chains
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from io import StringIO

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# ========== IO Output Methods ==========

class TestToCSVChains:
    """Test to_csv with lazy operation chains."""

    def test_to_csv_basic(self):
        """Basic to_csv output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_csv(index=False)
        ds_result = ds_df.to_csv(index=False)

        assert pd_result == ds_result

    def test_to_csv_after_filter(self):
        """to_csv after filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['x', 'y', 'z', 'w']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].to_csv(index=False)
        ds_result = ds_df[ds_df['a'] > 2].to_csv(index=False)

        assert pd_result == ds_result

    def test_to_csv_after_sort(self):
        """to_csv after sort chain."""
        pd_df = pd.DataFrame({'a': [3, 1, 2], 'b': ['z', 'x', 'y']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a').to_csv(index=False)
        ds_result = ds_df.sort_values('a').to_csv(index=False)

        assert pd_result == ds_result

    def test_to_csv_after_assign(self):
        """to_csv after assign chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(b=pd_df['a'] * 2).to_csv(index=False)
        ds_result = ds_df.assign(b=ds_df['a'] * 2).to_csv(index=False)

        assert pd_result == ds_result

    def test_to_csv_columns_param(self):
        """to_csv with columns parameter."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_csv(columns=['a', 'c'], index=False)
        ds_result = ds_df.to_csv(columns=['a', 'c'], index=False)

        assert pd_result == ds_result

    def test_to_csv_na_rep(self):
        """to_csv with na_rep for missing values."""
        pd_df = pd.DataFrame({'a': [1, None, 3], 'b': ['x', 'y', None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_csv(na_rep='NA', index=False)
        ds_result = ds_df.to_csv(na_rep='NA', index=False)

        assert pd_result == ds_result

    def test_to_csv_separator(self):
        """to_csv with custom separator."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_csv(sep='\t', index=False)
        ds_result = ds_df.to_csv(sep='\t', index=False)

        assert pd_result == ds_result

    def test_to_csv_to_file_after_filter(self):
        """to_csv to file after filter chain."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_pd:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_ds:
                try:
                    pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
                    ds_df = DataStore(pd_df)

                    pd_df[pd_df['a'] > 1].to_csv(f_pd.name, index=False)
                    ds_df[ds_df['a'] > 1].to_csv(f_ds.name, index=False)

                    with open(f_pd.name) as f1, open(f_ds.name) as f2:
                        assert f1.read() == f2.read()
                finally:
                    os.unlink(f_pd.name)
                    os.unlink(f_ds.name)


class TestToJSONChains:
    """Test to_json with lazy operation chains."""

    def test_to_json_basic(self):
        """Basic to_json output."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_json(orient='records')
        ds_result = ds_df.to_json(orient='records')

        assert pd_result == ds_result

    def test_to_json_after_filter(self):
        """to_json after filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 1].to_json(orient='records')
        ds_result = ds_df[ds_df['a'] > 1].to_json(orient='records')

        assert pd_result == ds_result

    def test_to_json_orient_split(self):
        """to_json with orient='split'."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_json(orient='split')
        ds_result = ds_df.to_json(orient='split')

        assert pd_result == ds_result

    def test_to_json_orient_index(self):
        """to_json with orient='index'."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_json(orient='index')
        ds_result = ds_df.to_json(orient='index')

        assert pd_result == ds_result

    def test_to_json_orient_columns(self):
        """to_json with orient='columns'."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_json(orient='columns')
        ds_result = ds_df.to_json(orient='columns')

        assert pd_result == ds_result

    def test_to_json_orient_values(self):
        """to_json with orient='values'."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_json(orient='values')
        ds_result = ds_df.to_json(orient='values')

        assert pd_result == ds_result

    def test_to_json_lines(self):
        """to_json with lines=True."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_json(orient='records', lines=True)
        ds_result = ds_df.to_json(orient='records', lines=True)

        assert pd_result == ds_result

    def test_to_json_after_groupby_agg(self):
        """to_json after groupby + agg chain."""
        pd_df = pd.DataFrame({'cat': ['a', 'b', 'a', 'b'], 'val': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.groupby('cat')['val'].sum().reset_index().to_json(orient='records')
        ds_result = ds_df.groupby('cat')['val'].sum().reset_index().to_json(orient='records')

        assert pd_result == ds_result


class TestToParquetChains:
    """Test to_parquet with lazy operation chains."""

    def test_to_parquet_basic(self):
        """Basic to_parquet output."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_pd:
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_ds:
                try:
                    pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
                    ds_df = DataStore(pd_df)

                    pd_df.to_parquet(f_pd.name, index=False)
                    ds_df.to_parquet(f_ds.name, index=False)

                    pd_read = pd.read_parquet(f_pd.name)
                    ds_read = pd.read_parquet(f_ds.name)

                    pd.testing.assert_frame_equal(pd_read, ds_read)
                finally:
                    os.unlink(f_pd.name)
                    os.unlink(f_ds.name)

    def test_to_parquet_after_filter(self):
        """to_parquet after filter chain."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_pd:
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_ds:
                try:
                    pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['w', 'x', 'y', 'z']})
                    ds_df = DataStore(pd_df)

                    pd_df[pd_df['a'] > 2].to_parquet(f_pd.name, index=False)
                    ds_df[ds_df['a'] > 2].to_parquet(f_ds.name, index=False)

                    pd_read = pd.read_parquet(f_pd.name)
                    ds_read = pd.read_parquet(f_ds.name)

                    pd.testing.assert_frame_equal(pd_read, ds_read)
                finally:
                    os.unlink(f_pd.name)
                    os.unlink(f_ds.name)

    def test_to_parquet_after_assign_sort(self):
        """to_parquet after assign + sort chain."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_pd:
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f_ds:
                try:
                    pd_df = pd.DataFrame({'a': [3, 1, 2]})
                    ds_df = DataStore(pd_df)

                    pd_df.assign(b=pd_df['a'] * 2).sort_values('a').to_parquet(f_pd.name, index=False)
                    ds_df.assign(b=ds_df['a'] * 2).sort_values('a').to_parquet(f_ds.name, index=False)

                    pd_read = pd.read_parquet(f_pd.name)
                    ds_read = pd.read_parquet(f_ds.name)

                    pd.testing.assert_frame_equal(pd_read, ds_read)
                finally:
                    os.unlink(f_pd.name)
                    os.unlink(f_ds.name)


# ========== Metadata Methods ==========

class TestDescribeChains:
    """Test describe with lazy operation chains."""

    def test_describe_basic(self):
        """Basic describe output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe()
        ds_result = ds_df.describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_after_filter(self):
        """describe after filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].describe()
        ds_result = ds_df[ds_df['a'] > 2].describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_percentiles(self):
        """describe with custom percentiles."""
        pd_df = pd.DataFrame({'a': range(100)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(percentiles=[0.1, 0.5, 0.9])
        ds_result = ds_df.describe(percentiles=[0.1, 0.5, 0.9])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_include_all(self):
        """describe with include='all'."""
        pd_df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat': ['a', 'b', 'a'],
            'flt': [1.5, 2.5, 3.5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(include='all')
        ds_result = ds_df.describe(include='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_include_number(self):
        """describe with include='number'."""
        pd_df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat': ['a', 'b', 'a'],
            'flt': [1.5, 2.5, 3.5]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(include='number')
        ds_result = ds_df.describe(include='number')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_exclude_object(self):
        """describe with exclude='object'."""
        pd_df = pd.DataFrame({
            'num': [1, 2, 3],
            'cat': ['a', 'b', 'a'],
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(exclude='object')
        ds_result = ds_df.describe(exclude='object')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMemoryUsageChains:
    """Test memory_usage with lazy operation chains."""

    def test_memory_usage_basic(self):
        """Basic memory_usage output."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.memory_usage()
        ds_result = ds_df.memory_usage()

        # Compare values (Index object vs Series)
        pd.testing.assert_series_equal(
            pd.Series(pd_result),
            pd.Series(ds_result)
        )

    def test_memory_usage_deep(self):
        """memory_usage with deep=True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.memory_usage(deep=True)
        ds_result = ds_df.memory_usage(deep=True)

        pd.testing.assert_series_equal(
            pd.Series(pd_result),
            pd.Series(ds_result)
        )

    def test_memory_usage_no_index(self):
        """memory_usage with index=False."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.memory_usage(index=False)
        ds_result = ds_df.memory_usage(index=False)

        pd.testing.assert_series_equal(
            pd.Series(pd_result),
            pd.Series(ds_result)
        )

    def test_memory_usage_after_filter(self):
        """memory_usage after filter chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['w', 'x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].memory_usage()
        ds_result = ds_df[ds_df['a'] > 2].memory_usage()

        # pandas 3.0 uses Arrow-backed strings with different memory accounting
        # Just verify the index column has same memory and string columns are close
        assert pd_result.index.tolist() == ds_result.index.tolist(), "Index names should match"
        # Allow small differences in memory due to Arrow vs object string storage
        for idx in pd_result.index:
            diff = abs(pd_result[idx] - ds_result[idx])
            assert diff <= 2, f"Memory usage for '{idx}' differs too much: {pd_result[idx]} vs {ds_result[idx]}"


class TestConvertDtypesChains:
    """Test convert_dtypes with lazy operation chains."""

    def test_convert_dtypes_basic(self):
        """Basic convert_dtypes output."""
        pd_df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.convert_dtypes()
        ds_result = ds_df.convert_dtypes()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_convert_dtypes_after_filter(self):
        """convert_dtypes after filter chain."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': ['w', 'x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].convert_dtypes()
        ds_result = ds_df[ds_df['a'] > 2].convert_dtypes()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_convert_dtypes_no_infer(self):
        """convert_dtypes with infer_objects=False."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.convert_dtypes(infer_objects=False)
        ds_result = ds_df.convert_dtypes(infer_objects=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSelectDtypesChains:
    """Test select_dtypes with lazy operation chains."""

    def test_select_dtypes_include_int(self):
        """select_dtypes include integer."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.5, 2.5, 3.5],
            'c': ['x', 'y', 'z']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include=['int64'])
        ds_result = ds_df.select_dtypes(include=['int64'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_include_number(self):
        """select_dtypes include 'number'."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.5, 2.5, 3.5],
            'c': ['x', 'y', 'z']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include='number')
        ds_result = ds_df.select_dtypes(include='number')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_exclude_object(self):
        """select_dtypes exclude 'object'."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.5, 2.5, 3.5],
            'c': ['x', 'y', 'z']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(exclude='object')
        ds_result = ds_df.select_dtypes(exclude='object')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_after_filter(self):
        """select_dtypes after filter chain."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [1.5, 2.5, 3.5, 4.5],
            'c': ['w', 'x', 'y', 'z']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].select_dtypes(include='number')
        ds_result = ds_df[ds_df['a'] > 2].select_dtypes(include='number')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_dtypes_then_operation(self):
        """select_dtypes then apply operation."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.5, 2.5, 3.5],
            'c': ['x', 'y', 'z']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.select_dtypes(include='number').sum()
        ds_result = ds_df.select_dtypes(include='number').sum()

        pd.testing.assert_series_equal(
            pd.Series(pd_result),
            pd.Series(ds_result)
        )


# ========== Complex Sort Operations ==========

class TestComplexSort:
    """Test complex sort operations."""

    def test_sort_values_multiple_columns(self):
        """Sort by multiple columns."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 1, 2],
            'b': [4, 3, 2, 1]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values(['a', 'b'])
        ds_result = ds_df.sort_values(['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_mixed_ascending(self):
        """Sort with mixed ascending/descending."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 1, 2],
            'b': [4, 3, 2, 1]
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values(['a', 'b'], ascending=[True, False])
        ds_result = ds_df.sort_values(['a', 'b'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_na_position_first(self):
        """Sort with na_position='first'."""
        pd_df = pd.DataFrame({'a': [3, None, 1, 2, None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', na_position='first')
        ds_result = ds_df.sort_values('a', na_position='first')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_na_position_last(self):
        """Sort with na_position='last' (default)."""
        pd_df = pd.DataFrame({'a': [3, None, 1, 2, None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', na_position='last')
        ds_result = ds_df.sort_values('a', na_position='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_descending_na_first(self):
        """Sort descending with na_position='first'."""
        pd_df = pd.DataFrame({'a': [3, None, 1, 2, None]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', ascending=False, na_position='first')
        ds_result = ds_df.sort_values('a', ascending=False, na_position='first')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_ignore_index(self):
        """Sort with ignore_index=True."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]}, index=[10, 20, 30])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', ignore_index=True)
        ds_result = ds_df.sort_values('a', ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_then_head(self):
        """Sort then head."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a').head(3)
        ds_result = ds_df.sort_values('a').head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_after_filter(self):
        """Sort after filter."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2], 'b': ['e', 'c', 'a', 'd', 'b']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].sort_values('a')
        ds_result = ds_df[ds_df['a'] > 2].sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_basic(self):
        """Sort by index."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[30, 10, 20])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_index()
        ds_result = ds_df.sort_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_index_descending(self):
        """Sort index descending."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]}, index=[10, 20, 30])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_index(ascending=False)
        ds_result = ds_df.sort_index(ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_string_column(self):
        """Sort by string column."""
        pd_df = pd.DataFrame({'a': ['banana', 'apple', 'cherry']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a')
        ds_result = ds_df.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_with_duplicates(self):
        """Sort with duplicate values (stability)."""
        pd_df = pd.DataFrame({
            'a': [2, 1, 2, 1],
            'b': ['d', 'c', 'b', 'a']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a', kind='stable')
        ds_result = ds_df.sort_values('a', kind='stable')

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== Index Operations with Chains ==========

class TestIndexChains:
    """Test index operations with chains."""

    def test_set_index_then_filter(self):
        """set_index then filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a')
        pd_result = pd_result[pd_result['b'] > 4]

        ds_result = ds_df.set_index('a')
        ds_result = ds_result[ds_result['b'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_then_sort_index(self):
        """set_index then sort_index."""
        pd_df = pd.DataFrame({'a': [3, 1, 2], 'b': [6, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a').sort_index()
        ds_result = ds_df.set_index('a').sort_index()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_then_filter(self):
        """reset_index then filter."""
        pd_df = pd.DataFrame({'b': [4, 5, 6]}, index=[1, 2, 3])
        pd_df.index.name = 'a'
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reset_index()
        pd_result = pd_result[pd_result['a'] > 1]

        ds_result = ds_df.reset_index()
        ds_result = ds_result[ds_result['a'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_drop_false(self):
        """set_index with drop=False."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a', drop=False)
        ds_result = ds_df.set_index('a', drop=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_append(self):
        """set_index with append=True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=[10, 20, 30])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('a', append=True)
        ds_result = ds_df.set_index('a', append=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_multiple_columns(self):
        """set_index with multiple columns."""
        pd_df = pd.DataFrame({'a': [1, 2, 1], 'b': ['x', 'y', 'z'], 'c': [7, 8, 9]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index(['a', 'b'])
        ds_result = ds_df.set_index(['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)


# ========== DataFrame Shape and Properties ==========

class TestDataFrameProperties:
    """Test DataFrame properties after operations."""

    def test_shape_after_filter(self):
        """shape after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].shape
        ds_result = ds_df[ds_df['a'] > 2].shape

        assert pd_result == ds_result

    def test_ndim(self):
        """ndim property."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        assert pd_df.ndim == ds_df.ndim == 2

    def test_size_after_filter(self):
        """size after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].size
        ds_result = ds_df[ds_df['a'] > 2].size

        assert pd_result == ds_result

    def test_T_transpose(self):
        """Transpose with .T."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.T
        ds_result = ds_df.T

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transpose_after_filter(self):
        """transpose after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 1].T
        ds_result = ds_df[ds_df['a'] > 1].T

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_property_false(self):
        """empty property when not empty."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        assert pd_df.empty == ds_df.empty == False

    def test_empty_property_true(self):
        """empty property when empty."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df)

        assert pd_df.empty == ds_df.empty == True

    def test_empty_after_filter_all_false(self):
        """empty after filter that removes all rows."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        assert pd_df[pd_df['a'] > 100].empty == ds_df[ds_df['a'] > 100].empty == True


# ========== Misc Edge Cases ==========

class TestMiscEdgeCases:
    """Test miscellaneous edge cases."""

    def test_columns_after_assign(self):
        """columns property after assign."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(b=1).columns.tolist()
        ds_result = ds_df.assign(b=1).columns.tolist()

        assert pd_result == ds_result

    def test_dtypes_after_assign(self):
        """dtypes after assign."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(b=1.5).dtypes
        ds_result = ds_df.assign(b=1.5).dtypes

        pd.testing.assert_series_equal(pd_result, ds_result)

    def test_index_after_filter(self):
        """index after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]}, index=['w', 'x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].index.tolist()
        ds_result = ds_df[ds_df['a'] > 2].index.tolist()

        assert pd_result == ds_result

    def test_values_after_sort(self):
        """values after sort."""
        pd_df = pd.DataFrame({'a': [3, 1, 2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a').values.tolist()
        ds_result = ds_df.sort_values('a').values.tolist()

        assert pd_result == ds_result

    def test_iterrows_basic(self):
        """iterrows iteration."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        ds_df = DataStore(pd_df)

        pd_rows = list(pd_df.iterrows())
        ds_rows = list(ds_df.iterrows())

        assert len(pd_rows) == len(ds_rows)
        for (pd_idx, pd_row), (ds_idx, ds_row) in zip(pd_rows, ds_rows):
            assert pd_idx == ds_idx
            pd.testing.assert_series_equal(pd_row, ds_row)

    def test_itertuples_basic(self):
        """itertuples iteration."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        ds_df = DataStore(pd_df)

        pd_tuples = list(pd_df.itertuples())
        ds_tuples = list(ds_df.itertuples())

        assert pd_tuples == ds_tuples

    def test_itertuples_after_filter(self):
        """itertuples after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_tuples = list(pd_df[pd_df['a'] > 1].itertuples())
        ds_tuples = list(ds_df[ds_df['a'] > 1].itertuples())

        assert pd_tuples == ds_tuples

    def test_keys_method(self):
        """keys method."""
        pd_df = pd.DataFrame({'a': [1], 'b': [2]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.keys().tolist()
        ds_result = ds_df.keys().tolist()

        assert pd_result == ds_result

    def test_items_iteration(self):
        """items iteration."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_items = list(pd_df.items())
        ds_items = list(ds_df.items())

        assert len(pd_items) == len(ds_items)
        for (pd_name, pd_col), (ds_name, ds_col) in zip(pd_items, ds_items):
            assert pd_name == ds_name
            pd.testing.assert_series_equal(pd_col, pd.Series(ds_col))

    def test_get_method(self):
        """get method for column access."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.get('a')
        ds_result = ds_df.get('a')

        # Both should return a Series
        pd.testing.assert_series_equal(pd_result, pd.Series(ds_result))

    def test_get_method_default(self):
        """get method with default for missing column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.get('missing', default='not found')
        ds_result = ds_df.get('missing', default='not found')

        assert pd_result == ds_result == 'not found'

    def test_pop_column(self):
        """pop a column."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore({'a': [1, 2], 'b': [3, 4]})

        pd_popped = pd_df.pop('b')
        ds_popped = ds_df.pop('b')

        pd.testing.assert_series_equal(pd_popped, ds_popped)
        assert pd_df.columns.tolist() == ds_df.columns.tolist() == ['a']

    def test_insert_column(self):
        """insert a column."""
        pd_df = pd.DataFrame({'a': [1, 2], 'c': [5, 6]})
        ds_df = DataStore({'a': [1, 2], 'c': [5, 6]})

        pd_df.insert(1, 'b', [3, 4])
        ds_df.insert(1, 'b', [3, 4])

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestComplexChains:
    """Test complex operation chains with IO/metadata."""

    def test_filter_sort_head_to_csv(self):
        """filter -> sort -> head -> to_csv chain."""
        pd_df = pd.DataFrame({'a': [5, 3, 1, 4, 2], 'b': ['e', 'c', 'a', 'd', 'b']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].sort_values('a').head(2).to_csv(index=False)
        ds_result = ds_df[ds_df['a'] > 2].sort_values('a').head(2).to_csv(index=False)

        assert pd_result == ds_result

    def test_assign_filter_describe(self):
        """assign -> filter -> describe chain."""
        pd_df = pd.DataFrame({'a': range(10)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.assign(b=pd_df['a'] * 2)[pd_df['a'] > 5].describe()
        ds_result = ds_df.assign(b=ds_df['a'] * 2)[ds_df['a'] > 5].describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_sort_to_json(self):
        """groupby -> agg -> sort -> to_json chain."""
        pd_df = pd.DataFrame({
            'cat': ['a', 'b', 'a', 'b'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore(pd_df)

        pd_result = (pd_df.groupby('cat')['val'].sum()
                     .reset_index()
                     .sort_values('val')
                     .to_json(orient='records'))
        ds_result = (ds_df.groupby('cat')['val'].sum()
                     .reset_index()
                     .sort_values('val')
                     .to_json(orient='records'))

        assert pd_result == ds_result

    def test_filter_select_dtypes_sum(self):
        """filter -> select_dtypes -> sum chain."""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [1.5, 2.5, 3.5, 4.5],
            'c': ['w', 'x', 'y', 'z']
        })
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].select_dtypes(include='number').sum()
        ds_result = ds_df[ds_df['a'] > 2].select_dtypes(include='number').sum()

        pd.testing.assert_series_equal(
            pd.Series(pd_result),
            pd.Series(ds_result)
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
