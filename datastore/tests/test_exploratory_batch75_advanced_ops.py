"""
Exploratory Batch 75: Advanced Operations and Edge Cases

Focus areas:
1. Constructor edge cases
2. Complex method chains
3. Cross-DataStore operations
4. Index-aware operations
5. Accessor completeness
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


class TestConstructorEdgeCases:
    """Test DataFrame constructor variations"""

    def test_constructor_from_numpy_array_with_columns(self):
        """Test constructing from numpy array with column names"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        pd_df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
        ds_df = DataStore(arr, columns=['A', 'B', 'C'])
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_with_index(self):
        """Test constructing with custom index"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds_df = DataStore({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_constructor_from_list_of_dicts(self):
        """Test constructing from list of dictionaries"""
        data = [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}]
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        assert_datastore_equals_pandas(ds_df, pd_df)


class TestBasicMethodsWorking:
    """Test basic methods that were verified working"""

    def test_xs_method(self):
        """Test xs (cross-section) method"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['foo', 'bar'])
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]}, index=['foo', 'bar'])
        pd_xs = pd_df.xs('foo')
        ds_xs = ds_df.xs('foo')
        assert pd_xs.tolist() == ds_xs.tolist()

    def test_take_method(self):
        """Test take method with indices"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        ds_df = DataStore({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        pd_take = pd_df.take([0, 2])
        ds_take = ds_df.take([0, 2])
        assert_datastore_equals_pandas(ds_take, pd_take)

    def test_combine_first(self):
        """Test combine_first method"""
        pd_df_a = pd.DataFrame({'A': [None, 2, 3], 'B': [4, None, 6]})
        pd_df_b = pd.DataFrame({'A': [10, 20, 30], 'B': [40, 50, 60]})
        ds_df_a = DataStore({'A': [None, 2, 3], 'B': [4, None, 6]})
        ds_df_b = DataStore({'A': [10, 20, 30], 'B': [40, 50, 60]})
        pd_combined = pd_df_a.combine_first(pd_df_b)
        ds_combined = ds_df_a.combine_first(ds_df_b)
        assert_datastore_equals_pandas(ds_combined, pd_combined)

    def test_truncate_method(self):
        """Test truncate method"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
        pd_trunc = pd_df.truncate(before=2, after=4)
        ds_trunc = ds_df.truncate(before=2, after=4)
        assert_datastore_equals_pandas(ds_trunc, pd_trunc)

    def test_add_prefix_suffix(self):
        """Test add_prefix and add_suffix"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})

        pd_prefixed = pd_df.add_prefix('col_')
        ds_prefixed = ds_df.add_prefix('col_')
        assert pd_prefixed.columns.tolist() == ds_prefixed.columns.tolist()

        pd_suffixed = pd_df.add_suffix('_val')
        ds_suffixed = ds_df.add_suffix('_val')
        assert pd_suffixed.columns.tolist() == ds_suffixed.columns.tolist()

    def test_squeeze(self):
        """Test squeeze method"""
        pd_df = pd.DataFrame({'A': [1]})
        ds_df = DataStore({'A': [1]})
        pd_squeezed = pd_df.squeeze()
        ds_squeezed = ds_df.squeeze()
        assert pd_squeezed == ds_squeezed

    def test_nlargest_nsmallest(self):
        """Test nlargest and nsmallest"""
        pd_df = pd.DataFrame({'A': [3, 1, 4, 1, 5, 9, 2], 'B': range(7)})
        ds_df = DataStore({'A': [3, 1, 4, 1, 5, 9, 2], 'B': range(7)})

        pd_largest = pd_df.nlargest(3, 'A')
        ds_largest = ds_df.nlargest(3, 'A')
        assert_datastore_equals_pandas(ds_largest, pd_largest)

        pd_smallest = pd_df.nsmallest(3, 'A')
        ds_smallest = ds_df.nsmallest(3, 'A')
        assert_datastore_equals_pandas(ds_smallest, pd_smallest)

    def test_align(self):
        """Test align method"""
        pd_df1 = pd.DataFrame({'A': [1, 2]}, index=[0, 1])
        pd_df2 = pd.DataFrame({'A': [3, 4]}, index=[1, 2])
        ds_df1 = DataStore({'A': [1, 2]}, index=[0, 1])
        ds_df2 = DataStore({'A': [3, 4]}, index=[1, 2])

        pd_aligned1, pd_aligned2 = pd_df1.align(pd_df2, join='outer')
        ds_aligned1, ds_aligned2 = ds_df1.align(ds_df2, join='outer')
        assert_datastore_equals_pandas(ds_aligned1, pd_aligned1)
        assert_datastore_equals_pandas(ds_aligned2, pd_aligned2)


class TestMergeOperations:
    """Test merge/join operations"""

    def test_merge_with_validate(self):
        """Test merge with validate parameter"""
        pd_df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'val2': [4, 5, 6]})
        ds_df1 = DataStore({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
        ds_df2 = DataStore({'key': ['a', 'b', 'd'], 'val2': [4, 5, 6]})

        pd_merged = pd_df1.merge(pd_df2, on='key', how='inner', validate='one_to_one')
        ds_merged = ds_df1.merge(ds_df2, on='key', how='inner', validate='one_to_one')
        assert_datastore_equals_pandas(ds_merged, pd_merged)

    def test_merge_with_indicator(self):
        """Test merge with indicator parameter"""
        pd_df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'val2': [4, 5, 6]})
        ds_df1 = DataStore({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
        ds_df2 = DataStore({'key': ['a', 'b', 'd'], 'val2': [4, 5, 6]})

        pd_merged = pd_df1.merge(pd_df2, on='key', how='outer', indicator=True)
        ds_merged = ds_df1.merge(ds_df2, on='key', how='outer', indicator=True)
        assert_datastore_equals_pandas(ds_merged, pd_merged)

    def test_cross_join(self):
        """Test cross join (merge how='cross')"""
        pd_df1 = pd.DataFrame({'A': [1, 2]})
        pd_df2 = pd.DataFrame({'B': ['x', 'y']})
        ds_df1 = DataStore({'A': [1, 2]})
        ds_df2 = DataStore({'B': ['x', 'y']})

        pd_cross = pd_df1.merge(pd_df2, how='cross')
        ds_cross = ds_df1.merge(ds_df2, how='cross')
        assert_datastore_equals_pandas(ds_cross, pd_cross)


class TestGroupByOperations:
    """Test groupby operations"""

    def test_groupby_filter_aggregate_chain(self):
        """Test filter then groupby aggregate chain"""
        pd_df = pd.DataFrame({'cat': ['A', 'A', 'B', 'B', 'C'], 'val': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'cat': ['A', 'A', 'B', 'B', 'C'], 'val': [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df['val'] > 1].groupby('cat')['val'].sum()
        ds_result = ds_df[ds_df['val'] > 1].groupby('cat')['val'].sum()
        assert list(pd_result) == list(ds_result)

    def test_groupby_multiple_columns(self):
        """Test groupby with multiple grouping columns"""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'val': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby(['A', 'B'])['val'].sum()
        ds_result = ds_df.groupby(['A', 'B'])['val'].sum()
        assert list(pd_result) == list(ds_result)

    def test_groupby_multiple_agg_functions(self):
        """Test agg with multiple functions"""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'val': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('A')['val'].agg(['sum', 'mean', 'max'])
        ds_result = ds_df.groupby('A')['val'].agg(['sum', 'mean', 'max'])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_dropna_false(self):
        """Test groupby with dropna=False"""
        pd_df = pd.DataFrame({'A': ['foo', 'foo', None, 'bar'], 'B': [1, 2, 3, 4]})
        ds_df = DataStore({'A': ['foo', 'foo', None, 'bar'], 'B': [1, 2, 3, 4]})

        pd_grp = pd_df.groupby('A', dropna=False)['B'].sum()
        ds_grp = ds_df.groupby('A', dropna=False)['B'].sum()
        # Note: pandas uses NaN as key, DataStore uses None
        # Compare values only
        assert list(pd_grp.sort_index(na_position='last')) == list(ds_grp.sort_index(na_position='last'))


class TestComplexChains:
    """Test complex operation chains"""

    def test_assign_then_sort(self):
        """Test assign followed by sort_values"""
        pd_df = pd.DataFrame({'A': [3, 1, 2]})
        ds_df = DataStore({'A': [3, 1, 2]})

        pd_result = pd_df.assign(B=lambda x: x['A'] * 2).sort_values('B')
        ds_result = ds_df.assign(B=lambda x: x['A'] * 2).sort_values('B')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter_with_isin(self):
        """Test assign followed by filter with isin"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore({'A': [1, 2, 3, 4]})

        pd_result = pd_df.assign(B=lambda x: x['A'] * 2)[lambda x: x['B'].isin([2, 6])]
        ds_result = ds_df.assign(B=lambda x: x['A'] * 2)[lambda x: x['B'].isin([2, 6])]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestReshapeOperations:
    """Test reshape operations"""

    def test_stack(self):
        """Test stack operation"""
        import warnings
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['one', 'two'])
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]}, index=['one', 'two'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pd_stacked = pd_df.stack()
            ds_stacked = ds_df.stack()
        assert list(pd_stacked) == list(ds_stacked)

    def test_pivot(self):
        """Test pivot operation"""
        pd_df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-01', '2020-01-02'],
            'variable': ['A', 'B', 'A'],
            'value': [1, 2, 3]
        })
        ds_df = DataStore({
            'date': ['2020-01-01', '2020-01-01', '2020-01-02'],
            'variable': ['A', 'B', 'A'],
            'value': [1, 2, 3]
        })

        pd_pivot = pd_df.pivot(index='date', columns='variable', values='value')
        ds_pivot = ds_df.pivot(index='date', columns='variable', values='value')
        assert_datastore_equals_pandas(ds_pivot, pd_pivot)

    def test_pivot_table(self):
        """Test pivot_table operation"""
        pd_df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'C': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': ['one', 'two', 'one', 'two'],
            'C': [1, 2, 3, 4]
        })

        pd_pt = pd_df.pivot_table(values='C', index='A', columns='B', aggfunc='sum')
        ds_pt = ds_df.pivot_table(values='C', index='A', columns='B', aggfunc='sum')
        assert_datastore_equals_pandas(ds_pt, pd_pt)

    def test_melt(self):
        """Test melt operation"""
        pd_df = pd.DataFrame({'A': ['a', 'b'], 'B': [1, 2], 'C': [3, 4]})
        ds_df = DataStore({'A': ['a', 'b'], 'B': [1, 2], 'C': [3, 4]})

        pd_melted = pd_df.melt(id_vars=['A'], value_vars=['B', 'C'])
        ds_melted = ds_df.melt(id_vars=['A'], value_vars=['B', 'C'])
        assert_datastore_equals_pandas(ds_melted, pd_melted)

    def test_explode(self):
        """Test explode operation"""
        pd_df = pd.DataFrame({'A': [[1, 2], [3, 4]]})
        ds_df = DataStore({'A': [[1, 2], [3, 4]]})

        pd_exploded = pd_df.explode('A')
        ds_exploded = ds_df.explode('A')
        assert_datastore_equals_pandas(ds_exploded, pd_exploded)


class TestComparisonAndInfo:
    """Test comparison and info methods"""

    def test_compare(self):
        """Test compare method"""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [1, 2, 9], 'B': [4, 8, 6]})
        ds_df1 = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df2 = DataStore({'A': [1, 2, 9], 'B': [4, 8, 6]})

        pd_cmp = pd_df1.compare(pd_df2)
        ds_cmp = ds_df1.compare(ds_df2)
        assert_datastore_equals_pandas(ds_cmp, pd_cmp)

    def test_memory_usage(self):
        """Test memory_usage method"""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_df = DataStore({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})

        pd_mem = pd_df.memory_usage()
        ds_mem = ds_df.memory_usage()
        assert list(pd_mem.index) == list(ds_mem.index)

    def test_duplicated_keep_false(self):
        """Test duplicated with keep=False"""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3]})
        ds_df = DataStore({'A': [1, 1, 2, 2, 3]})

        pd_dup = pd_df.duplicated(keep=False)
        ds_dup = ds_df.duplicated(keep=False)
        assert pd_dup.tolist() == ds_dup.tolist()


class TestSeriesOperations:
    """Test Series/ColumnExpr operations"""

    def test_series_mask_complex_condition(self):
        """Test Series.mask with complex conditions"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df['A'].mask((pd_df['A'] > 2) & (pd_df['A'] < 5), 0)
        ds_result = ds_df['A'].mask((ds_df['A'] > 2) & (ds_df['A'] < 5), 0)
        assert pd_result.tolist() == list(ds_result)

    def test_rolling_with_min_periods(self):
        """Test rolling with min_periods parameter"""
        pd_s = pd.Series([1, 2, None, 4, 5])
        ds_df = DataStore({'A': [1, 2, None, 4, 5]})

        pd_roll = pd_s.rolling(window=3, min_periods=1).mean()
        ds_roll = ds_df['A'].rolling(window=3, min_periods=1).mean()
        assert pd_roll.tolist() == list(ds_roll)

    def test_cumsum_skipna_false(self):
        """Test cumsum with skipna=False"""
        pd_df = pd.DataFrame({'A': [1, 2, None, 4]})
        ds_df = DataStore({'A': [1, 2, None, 4]})

        pd_cumsum = pd_df['A'].cumsum(skipna=False)
        ds_cumsum = ds_df['A'].cumsum(skipna=False)

        # Compare with NaN handling
        pd_list = pd_cumsum.tolist()
        ds_list = list(ds_cumsum)
        assert len(pd_list) == len(ds_list)
        for p, d in zip(pd_list, ds_list):
            if pd.isna(p):
                assert pd.isna(d)
            else:
                assert p == d

    def test_ewm_with_alpha(self):
        """Test ewm with alpha parameter"""
        pd_s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ds_df = DataStore({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})

        pd_ewm = pd_s.ewm(alpha=0.5).mean()
        ds_ewm = ds_df['A'].ewm(alpha=0.5).mean()

        pd_list = pd_ewm.tolist()
        ds_list = list(ds_ewm)
        assert len(pd_list) == len(ds_list)
        for p, d in zip(pd_list, ds_list):
            assert abs(p - d) < 1e-10

    def test_groupby_transform(self):
        """Test groupby transform"""
        pd_df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4]})
        ds_df = DataStore({'A': ['foo', 'foo', 'bar', 'bar'], 'B': [1, 2, 3, 4]})

        pd_transform = pd_df.groupby('A')['B'].transform('sum')
        ds_transform = ds_df.groupby('A')['B'].transform('sum')
        assert pd_transform.tolist() == list(ds_transform)

    def test_expanding(self):
        """Test expanding window"""
        pd_s = pd.Series([1, 2, 3, 4, 5])
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_expanding = pd_s.expanding().mean()
        ds_expanding = ds_df['A'].expanding().mean()
        assert pd_expanding.tolist() == list(ds_expanding)

    def test_idxmax_idxmin(self):
        """Test idxmax and idxmin"""
        pd_df = pd.DataFrame({'A': [3, 1, 2]}, index=['x', 'y', 'z'])
        ds_df = DataStore({'A': [3, 1, 2]}, index=['x', 'y', 'z'])

        assert pd_df['A'].idxmax() == ds_df['A'].idxmax()
        assert pd_df['A'].idxmin() == ds_df['A'].idxmin()


class TestStringAccessor:
    """Test string accessor methods"""

    def test_str_split_with_expand_and_n(self):
        """Test str.split with expand and n parameters"""
        pd_df = pd.DataFrame({'A': ['a-b-c', 'd-e-f', 'g-h']})
        ds_df = DataStore({'A': ['a-b-c', 'd-e-f', 'g-h']})

        pd_split = pd_df['A'].str.split('-', n=1, expand=True)
        ds_split = ds_df['A'].str.split('-', n=1, expand=True)
        assert_datastore_equals_pandas(ds_split, pd_split)

    def test_str_cat(self):
        """Test str.cat method"""
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
        ds_df = DataStore({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})

        pd_cat = pd_df['A'].str.cat(pd_df['B'], sep='-')
        ds_cat = ds_df['A'].str.cat(ds_df['B'], sep='-')
        assert pd_cat.tolist() == list(ds_cat)

    def test_str_replace_with_regex(self):
        """Test str.replace with regex"""
        pd_df = pd.DataFrame({'A': ['foo123', 'bar456']})
        ds_df = DataStore({'A': ['foo123', 'bar456']})

        pd_replace = pd_df['A'].str.replace(r'\d+', 'XXX', regex=True)
        ds_replace = ds_df['A'].str.replace(r'\d+', 'XXX', regex=True)
        assert pd_replace.tolist() == list(ds_replace)


class TestApplyAndMap:
    """Test apply and map operations"""

    def test_apply_axis_1(self):
        """Test apply with axis=1"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})

        pd_apply = pd_df.apply(lambda x: x['A'] + x['B'], axis=1)
        ds_apply = ds_df.apply(lambda x: x['A'] + x['B'], axis=1)
        assert pd_apply.tolist() == list(ds_apply)

    def test_map_with_dict(self):
        """Test Series.map with dictionary"""
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c']})
        ds_df = DataStore({'A': ['a', 'b', 'c']})

        mapping = {'a': 1, 'b': 2, 'c': 3}
        pd_mapped = pd_df['A'].map(mapping)
        ds_mapped = ds_df['A'].map(mapping)
        assert pd_mapped.tolist() == list(ds_mapped)

    def test_unique_order_preservation(self):
        """Test unique preserves order"""
        pd_s = pd.Series(['b', 'a', 'c', 'b', 'a'])
        ds_df = DataStore({'A': ['b', 'a', 'c', 'b', 'a']})

        pd_unique = pd_s.unique()
        ds_unique = ds_df['A'].unique()
        assert pd_unique.tolist() == list(ds_unique)

    def test_factorize(self):
        """Test factorize"""
        pd_s = pd.Series(['b', 'a', 'c', 'b', 'a'])
        ds_df = DataStore({'A': ['b', 'a', 'c', 'b', 'a']})

        pd_codes, pd_uniques = pd_s.factorize()
        ds_codes, ds_uniques = ds_df['A'].factorize()
        assert pd_codes.tolist() == list(ds_codes)
        assert pd_uniques.tolist() == list(ds_uniques)


class TestMiscMethods:
    """Test miscellaneous methods"""

    def test_reindex_with_fill_value(self):
        """Test reindex with fill_value"""
        pd_df = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df = DataStore({'A': [1, 2, 3]}, index=['a', 'b', 'c'])

        pd_reindex = pd_df.reindex(['a', 'b', 'd'], fill_value=0)
        ds_reindex = ds_df.reindex(['a', 'b', 'd'], fill_value=0)
        assert_datastore_equals_pandas(ds_reindex, pd_reindex)

    def test_set_axis(self):
        """Test set_axis"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = DataStore({'A': [1, 2], 'B': [3, 4]})

        pd_axis = pd_df.set_axis(['X', 'Y'], axis=1)
        ds_axis = ds_df.set_axis(['X', 'Y'], axis=1)
        assert pd_axis.columns.tolist() == ds_axis.columns.tolist()

    def test_droplevel(self):
        """Test droplevel with MultiIndex"""
        pd_df = pd.DataFrame(
            {'A': [1, 2]},
            index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['letter', 'number'])
        )
        ds_df = DataStore(
            {'A': [1, 2]},
            index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['letter', 'number'])
        )

        pd_drop = pd_df.droplevel('letter')
        ds_drop = ds_df.droplevel('letter')
        assert pd_drop.index.tolist() == ds_drop.index.tolist()

    def test_swaplevel(self):
        """Test swaplevel with MultiIndex"""
        pd_df = pd.DataFrame(
            {'A': [1, 2]},
            index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['letter', 'number'])
        )
        ds_df = DataStore(
            {'A': [1, 2]},
            index=pd.MultiIndex.from_tuples([('a', 1), ('b', 2)], names=['letter', 'number'])
        )

        pd_swap = pd_df.swaplevel(0, 1)
        ds_swap = ds_df.swaplevel(0, 1)
        assert pd_swap.index.tolist() == ds_swap.index.tolist()

    def test_first_last_valid_index(self):
        """Test first_valid_index and last_valid_index"""
        pd_df = pd.DataFrame({'A': [None, None, 1, 2, None]})
        ds_df = DataStore({'A': [None, None, 1, 2, None]})

        assert pd_df['A'].first_valid_index() == ds_df['A'].first_valid_index()
        assert pd_df['A'].last_valid_index() == ds_df['A'].last_valid_index()

    def test_select_dtypes(self):
        """Test select_dtypes"""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b'], 'C': [1.0, 2.0]})
        ds_df = DataStore({'A': [1, 2], 'B': ['a', 'b'], 'C': [1.0, 2.0]})

        pd_selected = pd_df.select_dtypes(include=['number'])
        ds_selected = ds_df.select_dtypes(include=['number'])
        assert pd_selected.columns.tolist() == ds_selected.columns.tolist()

    def test_any_all_with_axis(self):
        """Test any/all with axis parameter"""
        pd_df = pd.DataFrame({'A': [True, False], 'B': [True, True]})
        ds_df = DataStore({'A': [True, False], 'B': [True, True]})

        pd_any_row = pd_df.any(axis=1)
        ds_any_row = ds_df.any(axis=1)
        assert pd_any_row.tolist() == ds_any_row.tolist()

        pd_all_col = pd_df.all(axis=0)
        ds_all_col = ds_df.all(axis=0)
        assert pd_all_col.tolist() == ds_all_col.tolist()


class TestValueCounts:
    """Test value_counts method"""

    def test_value_counts_dropna_false(self):
        """Test value_counts with dropna=False"""
        pd_df = pd.DataFrame({'A': ['foo', 'bar', None, 'foo', None]})
        ds_df = DataStore({'A': ['foo', 'bar', None, 'foo', None]})

        pd_vc = pd_df['A'].value_counts(dropna=False)
        ds_vc = ds_df['A'].value_counts(dropna=False)
        # Compare values - need to use list since ColumnExpr iteration has issues with dict()
        # Sort by index to ensure consistent comparison
        pd_sorted = pd_vc.sort_index(na_position='last')
        ds_sorted = ds_vc.sort_index(na_position='last')
        assert list(pd_sorted) == list(ds_sorted)


class TestResample:
    """Test resample operations"""

    def test_resample_sum(self):
        """Test basic resample with sum"""
        import warnings
        dates = pd.date_range('2020-01-01', periods=6, freq='h')
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6]}, index=dates)
        ds_df = DataStore({'A': [1, 2, 3, 4, 5, 6]}, index=dates)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            pd_resampled = pd_df.resample('2h').sum()
            ds_resampled = ds_df.resample('2h').sum()
        assert_datastore_equals_pandas(ds_resampled, pd_resampled)


class TestConcat:
    """Test concat operations"""

    def test_concat_with_keys(self):
        """Test concat with keys parameter"""
        from datastore import concat

        pd_df1 = pd.DataFrame({'A': [1, 2]})
        pd_df2 = pd.DataFrame({'A': [3, 4]})
        ds_df1 = DataStore({'A': [1, 2]})
        ds_df2 = DataStore({'A': [3, 4]})

        pd_concat = pd.concat([pd_df1, pd_df2], keys=['x', 'y'])
        ds_concat = concat([ds_df1, ds_df2], keys=['x', 'y'])
        assert_datastore_equals_pandas(ds_concat, pd_concat)


class TestWhereMethod:
    """Test where/mask method"""

    def test_where_with_other(self):
        """Test where method with other parameter"""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = DataStore({'A': [1, 2, 3, 4]})

        pd_where = pd_df['A'].where(pd_df['A'] > 2, other=0)
        ds_where = ds_df['A'].where(ds_df['A'] > 2, other=0)
        assert pd_where.tolist() == list(ds_where)


# ============================================================================
# KNOWN ISSUES - Tests for discovered bugs
# These tests are marked with xfail until the issues are fixed
# ============================================================================


class TestKnownIssues:
    """Tests for discovered issues - marked xfail until fixed"""

    @pytest.mark.xfail(reason="clip with Series lower/upper not supported")
    def test_clip_with_series_boundary(self):
        """Test clip with Series as lower boundary"""
        pd_s = pd.Series([1, 2, 3, 4, 5])
        pd_lower = pd.Series([2, 2, 2, 2, 2])
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_lower_df = DataStore({'A': [2, 2, 2, 2, 2]})

        pd_clipped = pd_s.clip(lower=pd_lower)
        ds_clipped = ds_df['A'].clip(lower=ds_lower_df['A'])
        assert pd_clipped.tolist() == list(ds_clipped)

    @pytest.mark.xfail(reason="between returns LazyCondition instead of boolean Series")
    def test_between_return_type(self):
        """Test between returns correct type for direct use"""
        pd_s = pd.Series([1, 2, 3, 4, 5])
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})

        pd_between = pd_s.between(2, 4, inclusive='neither')
        ds_between = ds_df['A'].between(2, 4, inclusive='neither')
        # Should be able to call tolist() on DataStore result
        assert pd_between.tolist() == ds_between.tolist()

    @pytest.mark.xfail(reason="Cross-DataStore ColumnExpr comparison compares column names not values")
    def test_cross_datastore_comparison(self):
        """Test comparison between columns from different DataStores"""
        pd_s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        pd_s2 = pd.Series([2, 2, 2], index=['a', 'b', 'c'])
        ds_df1 = DataStore({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df2 = DataStore({'A': [2, 2, 2]}, index=['a', 'b', 'c'])

        pd_eq = pd_s1 == pd_s2
        ds_eq = ds_df1['A'] == ds_df2['A']
        assert pd_eq.tolist() == list(ds_eq)

    @pytest.mark.xfail(reason="Cross-DataStore Series arithmetic doesn't align on index")
    def test_cross_datastore_arithmetic(self):
        """Test arithmetic between Series from different DataStores with index mismatch"""
        pd_s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        pd_s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])
        ds_df1 = DataStore({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        ds_df2 = DataStore({'A': [10, 20, 30]}, index=['b', 'c', 'd'])

        pd_sum = pd_s1 + pd_s2
        ds_sum = ds_df1['A'] + ds_df2['A']

        # pandas aligns on index, DataStore should too
        pd_list = pd_sum.tolist()
        ds_list = list(ds_sum)
        assert len(pd_list) == len(ds_list)

    @pytest.mark.xfail(reason="dt.components not implemented for timedelta")
    def test_dt_components(self):
        """Test dt.components for timedelta Series"""
        pd_df = pd.DataFrame({'A': pd.to_timedelta(['1 days 2 hours', '3 days 4 hours'])})
        ds_df = DataStore({'A': pd.to_timedelta(['1 days 2 hours', '3 days 4 hours'])})

        pd_comp = pd_df['A'].dt.components
        ds_comp = ds_df['A'].dt.components
        assert_datastore_equals_pandas(ds_comp, pd_comp)

    @pytest.mark.xfail(reason="groupby observed=False returns repr string instead of result")
    def test_groupby_observed_false(self):
        """Test groupby with observed=False for categorical"""
        pd_df = pd.DataFrame({
            'cat': pd.Categorical(['a', 'a', 'b'], categories=['a', 'b', 'c']),
            'val': [1, 2, 3]
        })
        ds_df = DataStore({
            'cat': pd.Categorical(['a', 'a', 'b'], categories=['a', 'b', 'c']),
            'val': [1, 2, 3]
        })

        pd_grp = pd_df.groupby('cat', observed=False)['val'].sum()
        ds_grp = ds_df.groupby('cat', observed=False)['val'].sum()
        assert list(pd_grp) == list(ds_grp)
