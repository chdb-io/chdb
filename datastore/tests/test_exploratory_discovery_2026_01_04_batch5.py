"""
Exploratory Discovery - Batch 5 Tests (2026-01-04)

Focus areas:
1. Chained operations edge cases - complex operation combinations
2. DataFrame method parameter variants - uncommon parameters
3. loc/iloc edge cases
4. Data type boundaries - special value handling
5. merge/join edge cases
"""

import pytest
from tests.xfail_markers import datastore_callable_index
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series, get_value


class TestChainedOperationsEdgeCases:
    """Test edge cases of chained operations"""

    @datastore_callable_index
    def test_filter_assign_filter(self):
        """filter -> assign -> filter"""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df[df['a'] > 1].assign(c=lambda x: x['a'] * 2)[lambda x: x['c'] > 4]
        ds_result = ds[ds['a'] > 1].assign(c=lambda x: x['a'] * 2)[lambda x: x['c'] > 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_filter_groupby(self):
        """groupby -> filter on agg result -> groupby again"""
        df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'subcategory': ['x', 'y', 'x', 'y', 'x', 'y'],
                'value': [10, 20, 30, 40, 50, 60],
            }
        )
        ds = DataStore(df)

        # pandas: first groupby sum, then filter
        pd_agg = df.groupby('category')['value'].sum()
        pd_filtered = pd_agg[pd_agg > 50]

        ds_agg = ds.groupby('category')['value'].sum()
        ds_filtered = ds_agg[ds_agg > 50]

        assert_datastore_equals_pandas(ds_filtered, pd_filtered, check_row_order=False)

    def test_multiple_assign_chain(self):
        """Multiple consecutive assign operations"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df.assign(b=lambda x: x['a'] * 2).assign(c=lambda x: x['b'] + 1).assign(d=lambda x: x['c'] - x['a'])
        ds_result = ds.assign(b=lambda x: x['a'] * 2).assign(c=lambda x: x['b'] + 1).assign(d=lambda x: x['c'] - x['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_head_sort(self):
        """sort -> head -> sort (different column)"""
        df = pd.DataFrame({'a': [5, 3, 1, 4, 2], 'b': [10, 30, 50, 20, 40]})
        ds = DataStore(df)

        pd_result = df.sort_values('a').head(3).sort_values('b')
        ds_result = ds.sort_values('a').head(3).sort_values('b')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_filter_rename(self):
        """rename -> filter -> rename"""
        df = pd.DataFrame({'old_name': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.rename(columns={'old_name': 'new_name'})
        pd_result = pd_result[pd_result['new_name'] > 2]
        pd_result = pd_result.rename(columns={'new_name': 'final_name'})

        ds_result = ds.rename(columns={'old_name': 'new_name'})
        ds_result = ds_result[ds_result['new_name'] > 2]
        ds_result = ds_result.rename(columns={'new_name': 'final_name'})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataFrameMethodVariants:
    """Test DataFrame method parameter variants"""

    def test_drop_duplicates_keep_last(self):
        """drop_duplicates keep='last'"""
        df = pd.DataFrame({'a': [1, 1, 2, 2, 3], 'b': ['first', 'second', 'first', 'second', 'only']})
        ds = DataStore(df)

        pd_result = df.drop_duplicates(subset=['a'], keep='last')
        ds_result = ds.drop_duplicates(subset=['a'], keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self):
        """drop_duplicates keep=False (drop all duplicates)"""
        df = pd.DataFrame({'a': [1, 1, 2, 2, 3], 'b': ['first', 'second', 'first', 'second', 'only']})
        ds = DataStore(df)

        pd_result = df.drop_duplicates(subset=['a'], keep=False)
        ds_result = ds.drop_duplicates(subset=['a'], keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_ffill(self):
        """fillna with method='ffill'"""
        df = pd.DataFrame({'a': [1, np.nan, np.nan, 4, 5], 'b': [10, 20, np.nan, 40, np.nan]})
        ds = DataStore(df)

        pd_result = df.ffill()
        ds_result = ds.ffill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_bfill(self):
        """fillna with method='bfill'"""
        df = pd.DataFrame({'a': [np.nan, 2, np.nan, 4, 5], 'b': [10, np.nan, 30, np.nan, 50]})
        ds = DataStore(df)

        pd_result = df.bfill()
        ds_result = ds.bfill()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_na_position_first(self):
        """sort_values with na_position='first'"""
        df = pd.DataFrame({'a': [3, np.nan, 1, np.nan, 2], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.sort_values('a', na_position='first')
        ds_result = ds.sort_values('a', na_position='first')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_values_multiple_columns_mixed_order(self):
        """sort_values with multiple columns and mixed ascending"""
        df = pd.DataFrame({'a': [1, 1, 2, 2, 3], 'b': [5, 4, 3, 2, 1], 'c': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.sort_values(['a', 'b'], ascending=[True, False])
        ds_result = ds.sort_values(['a', 'b'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_keep_all(self):
        """nlargest with keep='all' (keep all ties)"""
        df = pd.DataFrame({'a': [1, 2, 2, 3, 3], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.nlargest(3, 'a', keep='all')
        ds_result = ds.nlargest(3, 'a', keep='all')

        # nlargest with keep='all' may return more than n rows
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_rank_method_dense(self):
        """rank with method='dense'"""
        df = pd.DataFrame({'a': [3, 1, 3, 2, 1]})
        ds = DataStore(df)

        pd_result = df['a'].rank(method='dense')
        ds_result = ds['a'].rank(method='dense')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rank_method_first(self):
        """rank with method='first'"""
        df = pd.DataFrame({'a': [3, 1, 3, 2, 1]})
        ds = DataStore(df)

        pd_result = df['a'].rank(method='first')
        ds_result = ds['a'].rank(method='first')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestLocIlocEdgeCases:
    """Test loc/iloc edge cases"""

    def test_iloc_negative_index(self):
        """iloc with negative index"""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.iloc[-1]
        ds_result = ds.iloc[-1]

        # Returns Series
        if isinstance(ds_result, pd.DataFrame):
            ds_result = ds_result.iloc[0]
        assert_series_equal(ds_result, pd_result)

    def test_iloc_negative_slice(self):
        """iloc with negative slice"""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.iloc[-3:]
        ds_result = ds.iloc[-3:]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_iloc_step(self):
        """iloc with step"""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [10, 20, 30, 40, 50, 60]})
        ds = DataStore(df)

        pd_result = df.iloc[::2]  # every other row
        ds_result = ds.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_single_column(self):
        """loc with single column selection"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})
        ds = DataStore(df)

        pd_result = df.loc[:, 'b']
        ds_result = ds.loc[:, 'b']

        # May return Series or DataFrame
        if isinstance(ds_result, pd.DataFrame):
            ds_result = ds_result.iloc[:, 0]
        assert_series_equal(
            ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_loc_column_list(self):
        """loc with column list"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'c': [100, 200, 300]})
        ds = DataStore(df)

        pd_result = df.loc[:, ['a', 'c']]
        ds_result = ds.loc[:, ['a', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataTypeBoundaries:
    """Test data type edge cases"""

    def test_very_large_numbers(self):
        """Handle very large numbers"""
        df = pd.DataFrame({'a': [10**15, 10**16, 10**17], 'b': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df[df['a'] > 10**15]
        ds_result = ds[ds['a'] > 10**15]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_float_precision(self):
        """Float precision"""
        df = pd.DataFrame({'a': [0.1 + 0.2, 0.3, 0.30000000001], 'b': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df.round(10)
        ds_result = ds.round(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_int_float_column(self):
        """Column with mixed integers and floats"""
        df = pd.DataFrame({'a': [1, 2.5, 3, 4.5, 5]})
        ds = DataStore(df)

        pd_result = df['a'].sum()
        ds_result = ds['a'].sum()

        # Should return the same sum
        assert abs(float(ds_result) - float(pd_result)) < 0.001

    def test_string_with_special_chars(self):
        """Strings with special characters"""
        df = pd.DataFrame({'a': ["hello'world", 'foo"bar', "test\nline", "tab\there"], 'b': [1, 2, 3, 4]})
        ds = DataStore(df)

        pd_result = df[df['b'] > 1]
        ds_result = ds[ds['b'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_string_vs_null(self):
        """Difference between empty string and NULL"""
        df = pd.DataFrame({'a': ['', None, 'value', ''], 'b': [1, 2, 3, 4]})
        ds = DataStore(df)

        # Count null values
        pd_null_count = df['a'].isna().sum()
        ds_null_count = ds['a'].isna().sum()

        assert int(ds_null_count) == int(pd_null_count)

    def test_boolean_column_operations(self):
        """Boolean column operations"""
        df = pd.DataFrame({'flag': [True, False, True, False], 'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        pd_result = df[df['flag']]
        ds_result = ds[ds['flag']]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMergeJoinEdgeCases:
    """Test merge/join edge cases"""

    def test_merge_with_suffixes(self):
        """merge with custom suffixes"""
        df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'value': [10, 20, 30]})
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)

        pd_result = pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
        ds_result = ds1.merge(ds2, on='key', suffixes=('_left', '_right'))

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_outer(self):
        """outer merge"""
        df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['b', 'c', 'd'], 'other': [20, 30, 40]})
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)

        pd_result = pd.merge(df1, df2, on='key', how='outer')
        ds_result = ds1.merge(ds2, on='key', how='outer')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_on_multiple_columns(self):
        """merge on multiple columns"""
        df1 = pd.DataFrame({'key1': ['a', 'a', 'b'], 'key2': [1, 2, 1], 'value': [10, 20, 30]})
        df2 = pd.DataFrame({'key1': ['a', 'b', 'b'], 'key2': [1, 1, 2], 'other': [100, 200, 300]})
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)

        pd_result = pd.merge(df1, df2, on=['key1', 'key2'])
        ds_result = ds1.merge(ds2, on=['key1', 'key2'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_with_indicator(self):
        """merge with indicator column"""
        df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'value': [1, 2, 3]})
        df2 = pd.DataFrame({'key': ['b', 'c', 'd'], 'other': [20, 30, 40]})
        ds1 = DataStore(df1)
        ds2 = DataStore(df2)

        pd_result = pd.merge(df1, df2, on='key', how='outer', indicator=True)
        ds_result = ds1.merge(ds2, on='key', how='outer', indicator=True)

        # indicator column may be category type
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestAggregationEdgeCases:
    """Test aggregation edge cases"""

    def test_agg_with_named_agg(self):
        """Using named aggregation"""
        df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        pd_result = df.groupby('category').agg(total=('value', 'sum'), average=('value', 'mean'))
        ds_result = ds.groupby('category').agg(total=('value', 'sum'), average=('value', 'mean'))

        # Reset index for comparison
        pd_result = pd_result.reset_index()
        ds_result = get_series(ds_result)
        if hasattr(ds_result, 'reset_index'):
            ds_result = ds_result.reset_index(drop=True) if 'category' in ds_result.columns else ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multiple_agg_same_column(self):
        """Apply multiple aggregation functions to the same column"""
        df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        pd_result = df.groupby('category')['value'].agg(['sum', 'mean', 'std'])
        ds_result = ds.groupby('category')['value'].agg(['sum', 'mean', 'std'])

        # Reset index
        pd_result = pd_result.reset_index()
        ds_result = get_series(ds_result)
        if isinstance(ds_result, pd.DataFrame):
            ds_result = ds_result.reset_index(drop=True) if 'category' in ds_result.columns else ds_result.reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_as_index_false(self):
        """groupby with as_index=False"""
        df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        pd_result = df.groupby('category', as_index=False)['value'].sum()
        ds_result = ds.groupby('category', as_index=False)['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_agg_with_all_nan(self):
        """Aggregate all-NaN column"""
        df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [np.nan, np.nan, np.nan, np.nan]})
        ds = DataStore(df)

        pd_result = df.groupby('category')['value'].sum()
        ds_result = ds.groupby('category')['value'].sum()

        # Sum of all NaN should return 0 or NaN (depends on skipna)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestStringMethodEdgeCases:
    """Test string method edge cases"""

    def test_str_contains_regex(self):
        """str.contains with regex"""
        df = pd.DataFrame({'text': ['hello world', 'HELLO', 'world hello', 'test'], 'id': [1, 2, 3, 4]})
        ds = DataStore(df)

        pd_result = df[df['text'].str.contains(r'hello', regex=True)]
        ds_result = ds[ds['text'].str.contains(r'hello', regex=True)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_contains_case_insensitive(self):
        """str.contains case insensitive"""
        df = pd.DataFrame({'text': ['Hello', 'HELLO', 'hello', 'test'], 'id': [1, 2, 3, 4]})
        ds = DataStore(df)

        pd_result = df[df['text'].str.contains('hello', case=False)]
        ds_result = ds[ds['text'].str.contains('hello', case=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_split_expand(self):
        """str.split with expand=True"""
        df = pd.DataFrame({'text': ['a-b-c', 'd-e-f', 'g-h-i']})
        ds = DataStore(df)

        pd_result = df['text'].str.split('-', expand=True)
        ds_result = ds['text'].str.split('-', expand=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_extract(self):
        """str.extract with regex groups"""
        df = pd.DataFrame({'text': ['user_123', 'user_456', 'admin_789']})
        ds = DataStore(df)

        pd_result = df['text'].str.extract(r'(\w+)_(\d+)')
        ds_result = ds['text'].str.extract(r'(\w+)_(\d+)')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace_regex(self):
        """str.replace with regex"""
        df = pd.DataFrame({'text': ['hello123world', 'test456here', 'no numbers']})
        ds = DataStore(df)

        pd_result = df['text'].str.replace(r'\d+', 'NUM', regex=True)
        ds_result = ds['text'].str.replace(r'\d+', 'NUM', regex=True)

        # Execute if ColumnExpr (lazy evaluation)
        ds_result = get_series(ds_result)
        # Returns Series
        if isinstance(ds_result, pd.DataFrame):
            ds_result = ds_result.iloc[:, 0]
        assert_series_equal(
            ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))


class TestWindowFunctionEdgeCases:
    """Test window function edge cases"""

    def test_rolling_with_min_periods(self):
        """rolling with min_periods < window"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].rolling(window=3, min_periods=1).sum()
        ds_result = ds['value'].rolling(window=3, min_periods=1).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_center(self):
        """rolling with center=True"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].rolling(window=3, center=True).mean()
        ds_result = ds['value'].rolling(window=3, center=True).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_sum(self):
        """expanding sum"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].expanding().sum()
        ds_result = ds['value'].expanding().sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shift_with_fill_value(self):
        """shift with fill_value"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df['value'].shift(2, fill_value=0)
        ds_result = ds['value'].shift(2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMiscellaneousEdgeCases:
    """Other edge cases"""

    def test_pivot_with_fill_value(self):
        """pivot_table with fill_value"""
        df = pd.DataFrame({'row': ['A', 'A', 'B', 'B'], 'col': ['x', 'y', 'x', 'y'], 'value': [1, 2, 3, np.nan]})
        ds = DataStore(df)

        pd_result = df.pivot_table(values='value', index='row', columns='col', fill_value=0)
        ds_result = ds.pivot_table(values='value', index='row', columns='col', fill_value=0)

        # pivot_table result is usually a DataFrame
        ds_result = get_value(ds_result)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_melt_basic(self):
        """melt operation"""
        df = pd.DataFrame({'id': ['A', 'B'], 'value1': [1, 2], 'value2': [3, 4]})
        ds = DataStore(df)

        pd_result = df.melt(id_vars=['id'], value_vars=['value1', 'value2'])
        ds_result = ds.melt(id_vars=['id'], value_vars=['value1', 'value2'])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_stack_unstack(self):
        """stack and unstack"""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds = DataStore(df)

        # stack
        pd_stacked = df.stack()
        ds_stacked = ds.stack()

        # Compare stack result
        ds_stacked = get_series(ds_stacked)

        # stack returns Series with MultiIndex, compare values
        assert len(ds_stacked) == len(pd_stacked)

    def test_clip_both_bounds(self):
        """clip with both lower and upper"""
        df = pd.DataFrame({'value': [-10, 0, 5, 10, 20]})
        ds = DataStore(df)

        pd_result = df['value'].clip(lower=0, upper=10)
        ds_result = ds['value'].clip(lower=0, upper=10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_inclusive(self):
        """between with different inclusive options"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        # inclusive='both' (default)
        pd_result = df[df['value'].between(2, 4)]
        ds_result = ds[ds['value'].between(2, 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_backtick_columns(self):
        """query with column names that need backticks"""
        df = pd.DataFrame({'col with space': [1, 2, 3, 4, 5], 'value': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df.query('`col with space` > 2')
        ds_result = ds.query('`col with space` > 2')

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
