"""
Exploratory Discovery Batch 19 - Module-level pandas API Functions and Edge Cases

Focus areas:
1. Module-level functions from pandas_api.py: merge_asof, merge_ordered,
   wide_to_long, crosstab, json_normalize, factorize, melt advanced
2. Constructor edge cases: various data sources
3. Deep ColumnExpr chaining
4. Complex lazy operation combinations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import DataStore
import sys
sys.path.insert(0, '/Users/auxten/Codes/go/src/github.com/auxten/chdb-ds')

from datastore import DataStore
import datastore as ds
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_dataframe, get_series


class TestMergeAsof:
    """Test ds.merge_asof() function."""

    def test_merge_asof_basic(self):
        """Basic merge_asof with sorted keys."""
        left = pd.DataFrame({
            'time': pd.to_datetime(['2020-01-01 10:00', '2020-01-01 10:05',
                                    '2020-01-01 10:10']),
            'value': [1, 2, 3]
        })
        right = pd.DataFrame({
            'time': pd.to_datetime(['2020-01-01 10:02', '2020-01-01 10:07']),
            'price': [100, 200]
        })

        pd_result = pd.merge_asof(left, right, on='time')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_asof(ds_left, ds_right, on='time')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_asof_direction_backward(self):
        """merge_asof with backward direction (default)."""
        left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
        right = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [10, 20, 30, 60, 70]})

        pd_result = pd.merge_asof(left, right, on='a', direction='backward')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_asof(ds_left, ds_right, on='a', direction='backward')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_asof_direction_forward(self):
        """merge_asof with forward direction."""
        left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
        right = pd.DataFrame({'a': [1, 2, 3, 6, 7, 12], 'right_val': [10, 20, 30, 60, 70, 120]})

        pd_result = pd.merge_asof(left, right, on='a', direction='forward')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_asof(ds_left, ds_right, on='a', direction='forward')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_asof_direction_nearest(self):
        """merge_asof with nearest direction."""
        left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
        right = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [10, 20, 30, 60, 70]})

        pd_result = pd.merge_asof(left, right, on='a', direction='nearest')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_asof(ds_left, ds_right, on='a', direction='nearest')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_asof_with_tolerance(self):
        """merge_asof with tolerance parameter."""
        left = pd.DataFrame({
            'time': pd.to_datetime(['2020-01-01 10:00', '2020-01-01 10:05']),
            'value': [1, 2]
        })
        right = pd.DataFrame({
            'time': pd.to_datetime(['2020-01-01 10:02', '2020-01-01 10:20']),
            'price': [100, 200]
        })

        pd_result = pd.merge_asof(left, right, on='time', tolerance=pd.Timedelta('3min'))

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_asof(ds_left, ds_right, on='time', tolerance=pd.Timedelta('3min'))

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMergeOrdered:
    """Test ds.merge_ordered() function."""

    def test_merge_ordered_basic(self):
        """Basic merge_ordered with fill_method."""
        left = pd.DataFrame({'key': ['a', 'c', 'e'], 'lval': [1, 2, 3]})
        right = pd.DataFrame({'key': ['b', 'c', 'd'], 'rval': [10, 20, 30]})

        pd_result = pd.merge_ordered(left, right, on='key')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_ordered(ds_left, ds_right, on='key')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_ordered_ffill(self):
        """merge_ordered with forward fill."""
        left = pd.DataFrame({'key': ['a', 'c', 'e'], 'lval': [1.0, 2.0, 3.0]})
        right = pd.DataFrame({'key': ['b', 'c', 'd'], 'rval': [10.0, 20.0, 30.0]})

        pd_result = pd.merge_ordered(left, right, on='key', fill_method='ffill')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_ordered(ds_left, ds_right, on='key', fill_method='ffill')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_ordered_left_on_right_on(self):
        """merge_ordered with different column names."""
        left = pd.DataFrame({'left_key': ['a', 'c', 'e'], 'lval': [1, 2, 3]})
        right = pd.DataFrame({'right_key': ['b', 'c', 'd'], 'rval': [10, 20, 30]})

        pd_result = pd.merge_ordered(left, right, left_on='left_key', right_on='right_key')

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_result = ds.merge_ordered(ds_left, ds_right, left_on='left_key', right_on='right_key')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestCrosstab:
    """Test ds.crosstab() function."""

    def test_crosstab_basic(self):
        """Basic crosstab with two series."""
        a = pd.Series(['foo', 'foo', 'bar', 'bar', 'foo', 'foo'], name='A')
        b = pd.Series(['one', 'one', 'two', 'two', 'one', 'two'], name='B')

        pd_result = pd.crosstab(a, b)
        ds_result = ds.crosstab(a, b)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_crosstab_with_values(self):
        """crosstab with values and aggfunc."""
        a = pd.Series(['foo', 'foo', 'bar', 'bar'], name='A')
        b = pd.Series(['one', 'two', 'one', 'two'], name='B')
        c = pd.Series([1, 2, 3, 4], name='C')

        pd_result = pd.crosstab(a, b, values=c, aggfunc='sum')
        ds_result = ds.crosstab(a, b, values=c, aggfunc='sum')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_crosstab_normalize(self):
        """crosstab with normalization."""
        a = pd.Series(['foo', 'foo', 'bar', 'bar', 'foo', 'foo'], name='A')
        b = pd.Series(['one', 'one', 'two', 'two', 'one', 'two'], name='B')

        pd_result = pd.crosstab(a, b, normalize=True)
        ds_result = ds.crosstab(a, b, normalize=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_crosstab_margins(self):
        """crosstab with margins (totals)."""
        a = pd.Series(['foo', 'foo', 'bar', 'bar'], name='A')
        b = pd.Series(['one', 'two', 'one', 'two'], name='B')

        pd_result = pd.crosstab(a, b, margins=True)
        ds_result = ds.crosstab(a, b, margins=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestWideToLong:
    """Test ds.wide_to_long() function."""

    def test_wide_to_long_basic(self):
        """Basic wide_to_long transformation."""
        df = pd.DataFrame({
            'A1970': [1, 2],
            'A1980': [3, 4],
            'B1970': [5, 6],
            'B1980': [7, 8],
            'id': ['x', 'y']
        })

        pd_result = pd.wide_to_long(df, stubnames=['A', 'B'], i='id', j='year')

        ds_df = DataStore(df)
        ds_result = ds.wide_to_long(ds_df, stubnames=['A', 'B'], i='id', j='year')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_wide_to_long_with_sep(self):
        """wide_to_long with separator."""
        df = pd.DataFrame({
            'A_1970': [1, 2],
            'A_1980': [3, 4],
            'id': ['x', 'y']
        })

        pd_result = pd.wide_to_long(df, stubnames=['A'], i='id', j='year', sep='_')

        ds_df = DataStore(df)
        ds_result = ds.wide_to_long(ds_df, stubnames=['A'], i='id', j='year', sep='_')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestFactorize:
    """Test ds.factorize() function."""

    def test_factorize_basic(self):
        """Basic factorize."""
        values = pd.Series(['a', 'b', 'a', 'c', 'b'])

        pd_codes, pd_uniques = pd.factorize(values)
        ds_codes, ds_uniques = ds.factorize(values)

        np.testing.assert_array_equal(pd_codes, ds_codes)

    def test_factorize_with_sort(self):
        """Factorize with sorting."""
        values = pd.Series(['c', 'a', 'b', 'a', 'c'])

        pd_codes, pd_uniques = pd.factorize(values, sort=True)
        ds_codes, ds_uniques = ds.factorize(values, sort=True)

        np.testing.assert_array_equal(pd_codes, ds_codes)

    def test_factorize_with_na(self):
        """Factorize with NA values."""
        values = pd.Series(['a', None, 'b', 'a', None])

        pd_codes, pd_uniques = pd.factorize(values, use_na_sentinel=True)
        ds_codes, ds_uniques = ds.factorize(values, use_na_sentinel=True)

        np.testing.assert_array_equal(pd_codes, ds_codes)


class TestJsonNormalize:
    """Test ds.json_normalize() function."""

    def test_json_normalize_basic(self):
        """Basic json_normalize."""
        data = [
            {'id': 1, 'name': {'first': 'John', 'last': 'Doe'}},
            {'id': 2, 'name': {'first': 'Jane', 'last': 'Smith'}}
        ]

        pd_result = pd.json_normalize(data)
        ds_result = ds.json_normalize(data)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_json_normalize_nested(self):
        """json_normalize with nested records."""
        data = [
            {'state': 'CA', 'cities': [{'name': 'LA'}, {'name': 'SF'}]},
            {'state': 'NY', 'cities': [{'name': 'NYC'}]}
        ]

        pd_result = pd.json_normalize(data, record_path='cities', meta='state')
        ds_result = ds.json_normalize(data, record_path='cities', meta='state')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_json_normalize_max_level(self):
        """json_normalize with max_level."""
        data = [
            {'a': {'b': {'c': 1}}, 'd': 2},
            {'a': {'b': {'c': 3}}, 'd': 4}
        ]

        pd_result = pd.json_normalize(data, max_level=1)
        ds_result = ds.json_normalize(data, max_level=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMeltAdvanced:
    """Test ds.melt() with advanced parameters."""

    def test_melt_ignore_index(self):
        """melt with ignore_index=False."""
        df = pd.DataFrame({
            'A': ['a', 'b'],
            'B': [1, 2],
            'C': [3, 4]
        }, index=['row1', 'row2'])

        pd_result = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'], ignore_index=False)

        ds_df = DataStore(df)
        ds_result = ds.melt(ds_df, id_vars=['A'], value_vars=['B', 'C'], ignore_index=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_var_name_value_name(self):
        """melt with custom var_name and value_name."""
        df = pd.DataFrame({
            'id': [1, 2],
            'var1': [10, 20],
            'var2': [100, 200]
        })

        pd_result = pd.melt(df, id_vars=['id'], var_name='variable_type', value_name='measurement')

        ds_df = DataStore(df)
        ds_result = ds.melt(ds_df, id_vars=['id'], var_name='variable_type', value_name='measurement')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestPivotTableAdvanced:
    """Test ds.pivot_table() with advanced parameters."""

    def test_pivot_table_multi_agg(self):
        """pivot_table with multiple aggregation functions."""
        df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'values': [1.0, 2.0, 3.0, 4.0]
        })

        pd_result = pd.pivot_table(df, values='values', index='A', columns='B', aggfunc=['mean', 'sum'])

        ds_df = DataStore(df)
        ds_result = ds.pivot_table(ds_df, values='values', index='A', columns='B', aggfunc=['mean', 'sum'])

        # Flatten column names for comparison
        pd_result.columns = ['_'.join(map(str, col)).strip() for col in pd_result.columns.values]
        ds_result_df = get_dataframe(ds_result)
        if hasattr(ds_result_df.columns, 'to_flat_index'):
            ds_result_df.columns = ['_'.join(map(str, col)).strip() for col in ds_result_df.columns.to_flat_index()]

        assert_datastore_equals_pandas(ds_result_df, pd_result)

    def test_pivot_table_fill_value(self):
        """pivot_table with fill_value."""
        df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar'],
            'B': ['one', 'two', 'one'],
            'values': [1, 2, 3]
        })

        pd_result = pd.pivot_table(df, values='values', index='A', columns='B', fill_value=0)

        ds_df = DataStore(df)
        ds_result = ds.pivot_table(ds_df, values='values', index='A', columns='B', fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pivot_table_margins(self):
        """pivot_table with margins."""
        df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'values': [1.0, 2.0, 3.0, 4.0]
        })

        pd_result = pd.pivot_table(df, values='values', index='A', columns='B', margins=True)

        ds_df = DataStore(df)
        ds_result = ds.pivot_table(ds_df, values='values', index='A', columns='B', margins=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestConstructorEdgeCases:
    """Test DataStore constructor with various edge cases."""

    def test_constructor_from_series(self):
        """DataStore from pandas Series."""
        ser = pd.Series([1, 2, 3], name='values')

        ds_df = DataStore(ser)
        pd_df = ser.to_frame()

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_from_list_of_lists(self):
        """DataStore from list of lists."""
        data = [[1, 2], [3, 4], [5, 6]]

        pd_df = pd.DataFrame(data, columns=['A', 'B'])
        ds_df = DataStore(data, columns=['A', 'B'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_from_dict_with_different_lengths(self):
        """DataStore from dict with scalar values."""
        data = {'A': 1, 'B': 2, 'C': 3}

        pd_df = pd.DataFrame(data, index=[0])
        ds_df = DataStore(data, index=[0])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_with_index(self):
        """DataStore with custom index."""
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}

        pd_df = pd.DataFrame(data, index=['x', 'y', 'z'])
        ds_df = DataStore(data, index=['x', 'y', 'z'])

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_empty_df(self):
        """DataStore from empty DataFrame."""
        pd_df = pd.DataFrame()
        ds_df = DataStore(pd_df)

        assert len(ds_df) == 0
        assert len(ds_df.columns) == 0

    def test_constructor_columns_only(self):
        """DataStore with columns but no data."""
        pd_df = pd.DataFrame(columns=['A', 'B', 'C'])
        ds_df = DataStore(columns=['A', 'B', 'C'])

        assert len(ds_df) == 0
        assert list(ds_df.columns) == ['A', 'B', 'C']


class TestDeepColumnExprChaining:
    """Test deep ColumnExpr chaining operations."""

    def test_str_chain_multiple_ops(self):
        """Chain multiple string operations."""
        data = {'text': ['  Hello World  ', '  Python  ', '  DataStore  ']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df['text'].str.strip().str.lower().str.replace(' ', '_')
        ds_result = ds_df['text'].str.strip().str.lower().str.replace(' ', '_')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_arithmetic_chain(self):
        """Chain multiple arithmetic operations."""
        data = {'value': [10, 20, 30]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = ((pd_df['value'] + 5) * 2 - 10) / 2
        ds_result = ((ds_df['value'] + 5) * 2 - 10) / 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_chain_str_arithmetic(self):
        """Mix string and length operations."""
        data = {'text': ['hello', 'world', 'test']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df['text'].str.upper().str.len() * 2
        ds_result = ds_df['text'].str.upper().str.len() * 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_after_assign_chain(self):
        """Filter after multiple assigns."""
        data = {'A': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_df = pd_df.assign(B=pd_df['A'] * 2)
        pd_df = pd_df.assign(C=pd_df['B'] + 10)
        pd_result = pd_df[pd_df['C'] > 15]

        ds_df = ds_df.assign(B=ds_df['A'] * 2)
        ds_df = ds_df.assign(C=ds_df['B'] + 10)
        ds_result = ds_df[ds_df['C'] > 15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_then_filter(self):
        """GroupBy aggregation then filter result."""
        data = {'group': ['A', 'A', 'B', 'B', 'C'], 'value': [10, 20, 30, 40, 50]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_agg = pd_df.groupby('group')['value'].sum().reset_index()
        pd_result = pd_agg[pd_agg['value'] > 25]

        ds_agg = ds_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_agg[ds_agg['value'] > 25]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestComplexLazyOpCombinations:
    """Test complex lazy operation combinations."""

    def test_filter_sort_head_select(self):
        """Filter -> sort -> head -> select chain."""
        data = {'A': [5, 3, 1, 4, 2], 'B': [50, 30, 10, 40, 20], 'C': ['e', 'c', 'a', 'd', 'b']}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['A'] > 1].sort_values('B').head(3)[['A', 'C']]
        ds_result = ds_df[ds_df['A'] > 1].sort_values('B').head(3)[['A', 'C']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_and_sorts(self):
        """Multiple filters and sorts."""
        data = {'A': [1, 2, 3, 4, 5], 'B': ['x', 'y', 'x', 'y', 'x'], 'C': [10, 20, 30, 40, 50]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df['A'] > 1][pd_df['B'] == 'x'].sort_values('C', ascending=False)
        ds_result = ds_df[ds_df['A'] > 1][ds_df['B'] == 'x'].sort_values('C', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_drop_rename_chain(self):
        """Assign -> drop -> rename chain."""
        data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.assign(D=pd_df['A'] + pd_df['B']).drop(columns=['C']).rename(columns={'D': 'sum_AB'})
        ds_result = ds_df.assign(D=ds_df['A'] + ds_df['B']).drop(columns=['C']).rename(columns={'D': 'sum_AB'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_then_groupby(self):
        """Concat then groupby."""
        df1 = pd.DataFrame({'group': ['A', 'B'], 'value': [1, 2]})
        df2 = pd.DataFrame({'group': ['A', 'B'], 'value': [3, 4]})

        pd_result = pd.concat([df1, df2]).groupby('group')['value'].sum().reset_index()

        ds1 = DataStore(df1)
        ds2 = DataStore(df2)
        ds_result = ds.concat([ds1, ds2]).groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_then_filter_then_agg(self):
        """Merge -> filter -> aggregate."""
        left = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
        right = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [10, 20, 30]})

        pd_merged = pd.merge(left, right, on='key', how='inner')
        pd_filtered = pd_merged[pd_merged['value1'] > 1]
        pd_result = pd_filtered['value2'].sum()

        ds_left = DataStore(left)
        ds_right = DataStore(right)
        ds_merged = ds_left.merge(ds_right, on='key', how='inner')
        ds_filtered = ds_merged[ds_merged['value1'] > 1]
        ds_result = ds_filtered['value2'].sum()

        # Both should be scalar 20
        assert ds_result == pd_result


class TestInferFreq:
    """Test ds.infer_freq() function."""

    def test_infer_freq_daily(self):
        """Infer daily frequency."""
        idx = pd.date_range('2020-01-01', periods=5, freq='D')

        pd_result = pd.infer_freq(idx)
        ds_result = ds.infer_freq(idx)

        assert pd_result == ds_result

    def test_infer_freq_hourly(self):
        """Infer hourly frequency."""
        idx = pd.date_range('2020-01-01', periods=5, freq='h')

        pd_result = pd.infer_freq(idx)
        ds_result = ds.infer_freq(idx)

        assert pd_result == ds_result

    def test_infer_freq_business_day(self):
        """Infer business day frequency."""
        idx = pd.bdate_range('2020-01-01', periods=5)

        pd_result = pd.infer_freq(idx)
        ds_result = ds.infer_freq(idx)

        assert pd_result == ds_result


class TestUniqueAndValueCounts:
    """Test ds.unique() and ds.value_counts() functions."""

    def test_unique_basic(self):
        """Basic unique function."""
        values = pd.Series([1, 2, 2, 3, 3, 3])

        pd_result = pd.unique(values)
        ds_result = ds.unique(values)

        np.testing.assert_array_equal(sorted(pd_result), sorted(ds_result))

    def test_unique_with_na(self):
        """unique with NA values."""
        values = pd.Series([1, 2, None, 2, None])

        pd_result = pd.unique(values)
        ds_result = ds.unique(values)

        # Both should contain 1, 2, and NaN
        assert len(pd_result) == len(ds_result)

    def test_value_counts_module_function(self):
        """ds.value_counts() module function."""
        values = pd.Series(['a', 'b', 'a', 'c', 'a', 'b'])

        pd_result = pd.value_counts(values)
        ds_result = ds.value_counts(values)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_value_counts_normalize(self):
        """value_counts with normalize."""
        values = pd.Series(['a', 'b', 'a', 'c', 'a', 'b'])

        pd_result = pd.value_counts(values, normalize=True)
        ds_result = ds.value_counts(values, normalize=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArrayFunction:
    """Test ds.array() function."""

    def test_array_basic(self):
        """Basic array creation."""
        pd_result = pd.array([1, 2, 3])
        ds_result = ds.array([1, 2, 3])

        np.testing.assert_array_equal(pd_result, ds_result)

    def test_array_with_dtype(self):
        """array with dtype."""
        pd_result = pd.array([1, 2, 3], dtype='float64')
        ds_result = ds.array([1, 2, 3], dtype='float64')

        np.testing.assert_array_equal(pd_result, ds_result)


class TestDataFrameSeriesConstructors:
    """Test ds.DataFrame() and ds.Series() factory functions."""

    def test_dataframe_factory(self):
        """ds.DataFrame() factory function."""
        pd_result = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_result = ds.DataFrame({'A': [1, 2], 'B': [3, 4]})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_factory(self):
        """ds.Series() factory function."""
        pd_result = pd.Series([1, 2, 3], name='values')
        ds_result = ds.Series([1, 2, 3], name='values')

        # ds.Series should return a ColumnExpr or pandas Series
        ds_result = get_series(ds_result)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsNaFunctions:
    """Test isna/isnull/notna/notnull functions."""

    def test_isna_datastore(self):
        """isna on DataStore."""
        data = {'A': [1, None, 3], 'B': [None, 2, None]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd.isna(pd_df)
        ds_result = ds.isna(ds_df)

        # ds.isna may return a DataFrame, DataStore, or numpy array
        if isinstance(ds_result, np.ndarray):
            ds_result = pd.DataFrame(ds_result, columns=pd_result.columns)
        else:
            ds_result = get_dataframe(ds_result)
        assert_frame_equal(ds_result.reset_index(drop=True), 
                                      pd_result.reset_index(drop=True))

    def test_isnull_series(self):
        """isnull on Series."""
        ser = pd.Series([1, None, 3])

        pd_result = pd.isnull(ser)
        ds_result = ds.isnull(ser)

        assert_series_equal(pd_result, ds_result)

    def test_notna_datastore(self):
        """notna on DataStore."""
        data = {'A': [1, None, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd.notna(pd_df)
        ds_result = ds.notna(ds_df)

        # ds.notna may return a DataFrame, DataStore, or numpy array
        if isinstance(ds_result, np.ndarray):
            ds_result = pd.DataFrame(ds_result, columns=pd_result.columns)
        else:
            ds_result = get_dataframe(ds_result)
        assert_frame_equal(ds_result.reset_index(drop=True), 
                                      pd_result.reset_index(drop=True))

    def test_notnull_series(self):
        """notnull on Series."""
        ser = pd.Series([1, None, 3])

        pd_result = pd.notnull(ser)
        ds_result = ds.notnull(ser)

        assert_series_equal(pd_result, ds_result)


class TestToConversionFunctions:
    """Test to_datetime, to_numeric, to_timedelta functions."""

    def test_to_datetime_basic(self):
        """to_datetime basic usage."""
        dates = ['2020-01-01', '2020-01-02', '2020-01-03']

        pd_result = pd.to_datetime(dates)
        ds_result = ds.to_datetime(dates)

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_to_datetime_format(self):
        """to_datetime with format."""
        dates = ['01/01/2020', '02/01/2020']

        pd_result = pd.to_datetime(dates, format='%d/%m/%Y')
        ds_result = ds.to_datetime(dates, format='%d/%m/%Y')

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_to_numeric_basic(self):
        """to_numeric basic usage."""
        values = pd.Series(['1', '2', '3'])

        pd_result = pd.to_numeric(values)
        ds_result = ds.to_numeric(values)

        assert_series_equal(pd_result, ds_result)

    def test_to_numeric_errors_coerce(self):
        """to_numeric with errors='coerce'."""
        values = pd.Series(['1', 'two', '3'])

        pd_result = pd.to_numeric(values, errors='coerce')
        ds_result = ds.to_numeric(values, errors='coerce')

        assert_series_equal(pd_result, ds_result)

    def test_to_timedelta_basic(self):
        """to_timedelta basic usage."""
        values = ['1 days', '2 days', '3 days']

        pd_result = pd.to_timedelta(values)
        ds_result = ds.to_timedelta(values)

        pd.testing.assert_index_equal(pd_result, ds_result)


class TestRangeFunctions:
    """Test date_range, bdate_range, period_range, timedelta_range, interval_range."""

    def test_date_range_basic(self):
        """date_range basic usage."""
        pd_result = pd.date_range('2020-01-01', periods=5)
        ds_result = ds.date_range('2020-01-01', periods=5)

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_date_range_freq(self):
        """date_range with frequency."""
        pd_result = pd.date_range('2020-01-01', '2020-01-10', freq='2D')
        ds_result = ds.date_range('2020-01-01', '2020-01-10', freq='2D')

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_bdate_range_basic(self):
        """bdate_range basic usage."""
        pd_result = pd.bdate_range('2020-01-01', periods=5)
        ds_result = ds.bdate_range('2020-01-01', periods=5)

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_period_range_basic(self):
        """period_range basic usage."""
        pd_result = pd.period_range('2020-01', periods=3, freq='M')
        ds_result = ds.period_range('2020-01', periods=3, freq='M')

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_timedelta_range_basic(self):
        """timedelta_range basic usage."""
        pd_result = pd.timedelta_range('1 days', periods=5)
        ds_result = ds.timedelta_range('1 days', periods=5)

        pd.testing.assert_index_equal(pd_result, ds_result)

    def test_interval_range_basic(self):
        """interval_range basic usage."""
        pd_result = pd.interval_range(start=0, end=5)
        ds_result = ds.interval_range(start=0, end=5)

        pd.testing.assert_index_equal(pd_result, ds_result)


class TestOptionsFunctions:
    """Test option management functions."""

    def test_set_get_option(self):
        """set_option and get_option."""
        original = ds.get_option('display.max_rows')

        ds.set_option('display.max_rows', 100)
        assert ds.get_option('display.max_rows') == 100

        ds.reset_option('display.max_rows')
        assert ds.get_option('display.max_rows') == original

    def test_option_context(self):
        """option_context manager."""
        original = ds.get_option('display.max_rows')

        with ds.option_context('display.max_rows', 50):
            assert ds.get_option('display.max_rows') == 50

        assert ds.get_option('display.max_rows') == original


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
