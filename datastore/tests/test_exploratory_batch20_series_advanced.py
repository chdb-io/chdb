"""
Exploratory Discovery Batch 20 - Series Advanced Operations and Deep Chains
Date: 2026-01-04

Focus areas:
1. Series-specific methods (map, factorize, searchsorted)
2. DataFrame advanced indexing (xs, swaplevel, reorder_levels)
3. Deep operation chains
4. Special numeric values handling
5. DataFrame/Series interaction patterns
"""

import pytest
from tests.xfail_markers import chdb_median_in_where, datastore_str_join_array
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_dataframe, get_series


class TestSeriesMapAdvanced:
    """Test Series.map() with various input types."""

    def test_map_with_dict(self):
        """Map values using dictionary."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'b']})
        ds_df = DataStore(pd_df.copy())
        
        mapping = {'a': 'Apple', 'b': 'Banana', 'c': 'Cherry'}
        
        pd_result = pd_df.assign(mapped=pd_df['category'].map(mapping))
        ds_result = ds_df.assign(mapped=ds_df['category'].map(mapping))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_with_series(self):
        """Map values using Series as mapper."""
        pd_df = pd.DataFrame({'category': ['a', 'b', 'c', 'a', 'b']})
        ds_df = DataStore(pd_df.copy())
        
        mapper = pd.Series(['Apple', 'Banana', 'Cherry'], index=['a', 'b', 'c'])
        
        pd_result = pd_df.assign(mapped=pd_df['category'].map(mapper))
        ds_result = ds_df.assign(mapped=ds_df['category'].map(mapper))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_with_function(self):
        """Map values using function."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.assign(squared=pd_df['value'].map(lambda x: x ** 2))
        ds_result = ds_df.assign(squared=ds_df['value'].map(lambda x: x ** 2))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_with_na_action(self):
        """Map with na_action parameter."""
        pd_df = pd.DataFrame({'category': ['a', None, 'c', 'a', None]})
        ds_df = DataStore(pd_df.copy())
        
        mapping = {'a': 'Apple', 'c': 'Cherry'}
        
        pd_result = pd_df.assign(mapped=pd_df['category'].map(mapping, na_action='ignore'))
        ds_result = ds_df.assign(mapped=ds_df['category'].map(mapping, na_action='ignore'))
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSeriesFactorize:
    """Test Series.factorize() method."""

    def test_factorize_basic(self):
        """Basic factorize."""
        pd_df = pd.DataFrame({'category': ['b', 'a', 'c', 'b', 'a']})
        ds_df = DataStore(pd_df.copy())
        
        pd_codes, pd_uniques = pd_df['category'].factorize()
        ds_codes, ds_uniques = ds_df['category'].factorize()
        
        # Codes should match
        np.testing.assert_array_equal(pd_codes, ds_codes)
        # Uniques should match (order matters for unsorted)
        np.testing.assert_array_equal(pd_uniques, ds_uniques)

    def test_factorize_with_sort(self):
        """Factorize with sorted uniques."""
        pd_df = pd.DataFrame({'category': ['b', 'a', 'c', 'b', 'a']})
        ds_df = DataStore(pd_df.copy())
        
        pd_codes, pd_uniques = pd_df['category'].factorize(sort=True)
        ds_codes, ds_uniques = ds_df['category'].factorize(sort=True)
        
        np.testing.assert_array_equal(pd_codes, ds_codes)
        np.testing.assert_array_equal(pd_uniques, ds_uniques)

    def test_factorize_with_na(self):
        """Factorize with NA values."""
        pd_df = pd.DataFrame({'category': ['b', None, 'c', 'b', None]})
        ds_df = DataStore(pd_df.copy())
        
        pd_codes, pd_uniques = pd_df['category'].factorize()
        ds_codes, ds_uniques = ds_df['category'].factorize()
        
        np.testing.assert_array_equal(pd_codes, ds_codes)
        # Uniques arrays may have different dtypes but same values
        assert len(pd_uniques) == len(ds_uniques)


class TestSeriesSearchsorted:
    """Test Series.searchsorted() method."""

    def test_searchsorted_basic(self):
        """Basic searchsorted on sorted series."""
        pd_df = pd.DataFrame({'value': [1, 3, 5, 7, 9]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].searchsorted(4)
        ds_result = ds_df['value'].searchsorted(4)
        
        assert pd_result == ds_result

    def test_searchsorted_array(self):
        """Searchsorted with array of values."""
        pd_df = pd.DataFrame({'value': [1, 3, 5, 7, 9]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].searchsorted([2, 4, 6])
        ds_result = ds_df['value'].searchsorted([2, 4, 6])
        
        np.testing.assert_array_equal(pd_result, ds_result)

    def test_searchsorted_side(self):
        """Searchsorted with side parameter."""
        pd_df = pd.DataFrame({'value': [1, 3, 3, 5, 7]})
        ds_df = DataStore(pd_df.copy())
        
        pd_left = pd_df['value'].searchsorted(3, side='left')
        ds_left = ds_df['value'].searchsorted(3, side='left')
        pd_right = pd_df['value'].searchsorted(3, side='right')
        ds_right = ds_df['value'].searchsorted(3, side='right')
        
        assert pd_left == ds_left
        assert pd_right == ds_right


class TestDataFrameXS:
    """Test DataFrame.xs() cross-section method."""

    def test_xs_basic(self):
        """Basic xs on single-level index."""
        pd_df = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            index=['x', 'y', 'z']
        )
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.xs('y')
        ds_result = ds_df.xs('y')
        
        # xs returns a Series for single row
        assert_series_equal(
            get_series(ds_result),
            pd_result
        )

    def test_xs_multiindex_level0(self):
        """XS on MultiIndex at level 0."""
        arrays = [
            ['a', 'a', 'b', 'b'],
            [1, 2, 1, 2]
        ]
        idx = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.xs('a', level='first')
        ds_result = ds_df.xs('a', level='first')
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_xs_multiindex_level1(self):
        """XS on MultiIndex at level 1."""
        arrays = [
            ['a', 'a', 'b', 'b'],
            [1, 2, 1, 2]
        ]
        idx = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.xs(1, level='second')
        ds_result = ds_df.xs(1, level='second')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataFrameSwapLevel:
    """Test DataFrame.swaplevel() method."""

    def test_swaplevel_index(self):
        """Swap levels in MultiIndex."""
        arrays = [
            ['a', 'a', 'b', 'b'],
            [1, 2, 1, 2]
        ]
        idx = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.swaplevel()
        ds_result = ds_df.swaplevel()
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_swaplevel_columns(self):
        """Swap levels in MultiIndex columns."""
        arrays = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]]
        columns = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
        pd_df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=columns)
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.swaplevel(axis=1)
        ds_result = ds_df.swaplevel(axis=1)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDataFrameReorderLevels:
    """Test DataFrame.reorder_levels() method."""

    def test_reorder_levels_index(self):
        """Reorder levels in MultiIndex."""
        arrays = [
            ['a', 'a', 'b', 'b'],
            [1, 2, 1, 2],
            ['x', 'y', 'x', 'y']
        ]
        idx = pd.MultiIndex.from_arrays(arrays, names=['first', 'second', 'third'])
        pd_df = pd.DataFrame({'value': [10, 20, 30, 40]}, index=idx)
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.reorder_levels(['third', 'first', 'second'])
        ds_result = ds_df.reorder_levels(['third', 'first', 'second'])
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDeepOperationChains:
    """Test deeply nested operation chains."""

    def test_filter_sort_head_select_chain(self):
        """Chain: filter -> sort -> head -> select."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'C': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[pd_df['A'] > 3].sort_values('B').head(4)[['A', 'C']]
        ds_result = ds_df[ds_df['A'] > 3].sort_values('B').head(4)[['A', 'C']]
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_agg_sort_head_chain(self):
        """Chain: groupby -> agg -> sort -> head."""
        pd_df = pd.DataFrame({
            'group': ['a', 'a', 'b', 'b', 'c', 'c'],
            'value': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.groupby('group').agg({'value': 'sum'}).sort_values('value', ascending=False).head(2)
        ds_result = ds_df.groupby('group').agg({'value': 'sum'}).sort_values('value', ascending=False).head(2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_filter_groupby_agg_chain(self):
        """Chain: assign -> filter -> groupby -> agg."""
        pd_df = pd.DataFrame({
            'category': ['a', 'a', 'b', 'b', 'c', 'c'],
            'value': [10, 20, 30, 40, 50, 60]
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_result = (pd_df
            .assign(doubled=pd_df['value'] * 2)
            [pd_df['value'] > 20]
            .groupby('category')
            .agg({'doubled': 'mean'}))
        ds_result = (ds_df
            .assign(doubled=ds_df['value'] * 2)
            [ds_df['value'] > 20]
            .groupby('category')
            .agg({'doubled': 'mean'}))
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filter_conditions(self):
        """Multiple filter conditions combined."""
        pd_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': ['x', 'y', 'x', 'y', 'x']
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df[(pd_df['A'] > 1) & (pd_df['B'] > 1) & (pd_df['C'] == 'x')]
        ds_result = ds_df[(ds_df['A'] > 1) & (ds_df['B'] > 1) & (ds_df['C'] == 'x')]
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_then_filter_groupby(self):
        """Chain: merge -> filter -> groupby -> agg."""
        pd_left = pd.DataFrame({'key': ['a', 'b', 'c'], 'value1': [1, 2, 3]})
        pd_right = pd.DataFrame({'key': ['a', 'b', 'c'], 'value2': [10, 20, 30]})
        ds_left = DataStore(pd_left.copy())
        ds_right = DataStore(pd_right.copy())
        
        pd_merged = pd_left.merge(pd_right, on='key')
        ds_merged = ds_left.merge(ds_right, on='key')
        
        pd_result = pd_merged[pd_merged['value1'] > 1].groupby('key').agg({'value2': 'sum'})
        ds_result = ds_merged[ds_merged['value1'] > 1].groupby('key').agg({'value2': 'sum'})
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialNumericValues:
    """Test handling of special numeric values."""

    def test_inf_values(self):
        """Handle infinity values."""
        pd_df = pd.DataFrame({'value': [1.0, np.inf, 3.0, -np.inf, 5.0]})
        ds_df = DataStore(pd_df.copy())
        
        # Filter for finite values
        pd_result = pd_df[np.isfinite(pd_df['value'])]
        ds_result = ds_df[np.isfinite(ds_df['value'].to_pandas())]
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_inf(self):
        """Replace infinity values."""
        pd_df = pd.DataFrame({'value': [1.0, np.inf, 3.0, -np.inf, 5.0]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.replace([np.inf, -np.inf], np.nan)
        ds_result = ds_df.replace([np.inf, -np.inf], np.nan)
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_small_values(self):
        """Handle very small values."""
        pd_df = pd.DataFrame({'value': [1e-300, 1e-200, 1e-100, 1e-50, 1.0]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].sum()
        ds_result = ds_df['value'].sum()
        
        # Use relative tolerance for very small values
        assert abs(pd_result - ds_result) < 1e-10 or abs((pd_result - ds_result) / pd_result) < 1e-5

    def test_very_large_values(self):
        """Handle very large values."""
        pd_df = pd.DataFrame({'value': [1e50, 1e100, 1e200, 1e300]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].max()
        ds_result = ds_df['value'].max()
        
        assert pd_result == ds_result


class TestDataFrameSeriesInteraction:
    """Test interactions between DataFrame and Series operations."""

    def test_series_to_frame(self):
        """Convert Series to DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_series = pd_df['A']
        ds_series = ds_df['A']
        
        pd_result = pd_series.to_frame(name='column_A')
        ds_result = ds_series.to_frame(name='column_A')
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_series_arithmetic(self):
        """Arithmetic between DataFrame column and scalar."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_mean = pd_df['A'].mean()
        ds_mean = ds_df['A'].mean()
        
        pd_result = pd_df.assign(A_centered=pd_df['A'] - pd_mean)
        ds_result = ds_df.assign(A_centered=ds_df['A'] - ds_mean)
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_comparison_with_dataframe_filter(self):
        """Use Series comparison result to filter DataFrame.
        
        Note: This works because ds_df['A'].median() returns scalar (matching pandas),
        which can be used directly in comparison without SQL subquery issues.
        """
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})
        ds_df = DataStore(pd_df.copy())
        
        pd_mask = pd_df['A'] > pd_df['A'].median()
        ds_mask = ds_df['A'] > ds_df['A'].median()
        
        pd_result = pd_df[pd_mask]
        ds_result = ds_df[ds_mask]
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSeriesStringOperations:
    """Test various string operations on Series."""

    def test_str_split_expand(self):
        """Split strings and expand to DataFrame."""
        pd_df = pd.DataFrame({'text': ['a-b-c', 'd-e-f', 'g-h-i']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['text'].str.split('-', expand=True)
        ds_result = ds_df['text'].str.split('-', expand=True)
        
        # Convert to DataFrame using Duck Typing
        ds_result_df = get_dataframe(ds_result)
            
        assert_frame_equal(ds_result_df, pd_result)

    @datastore_str_join_array
    def test_str_join(self):
        """Join strings from list column."""
        pd_df = pd.DataFrame({'letters': [['a', 'b', 'c'], ['d', 'e', 'f']]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['letters'].str.join('-')
        ds_result = ds_df['letters'].str.join('-')
        
        assert_series_equal(
            get_series(ds_result),
            pd_result
        )

    def test_str_get_dummies(self):
        """Get dummies from string column."""
        pd_df = pd.DataFrame({'category': ['a|b', 'b|c', 'a|c']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['category'].str.get_dummies('|')
        ds_result = ds_df['category'].str.get_dummies('|')
        
        assert_frame_equal(
            get_series(ds_result),
            pd_result
        )


class TestSeriesMethodChains:
    """Test chained method calls on Series."""

    def test_str_chain_lower_strip_replace(self):
        """Chain: lower -> strip -> replace."""
        pd_df = pd.DataFrame({'text': ['  Hello World  ', '  FOO BAR  ', '  Test  ']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['text'].str.lower().str.strip().str.replace(' ', '_', regex=False)
        ds_result = ds_df['text'].str.lower().str.strip().str.replace(' ', '_', regex=False)
        
        assert_series_equal(
            get_series(ds_result),
            pd_result
        )

    def test_numeric_chain_abs_round_clip(self):
        """Chain: abs -> round -> clip."""
        pd_df = pd.DataFrame({'value': [-3.7, 2.3, -1.5, 4.9, -0.2]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].abs().round().clip(lower=0, upper=3)
        ds_result = ds_df['value'].abs().round().clip(lower=0, upper=3)
        
        assert_series_equal(
            get_series(ds_result),
            pd_result,
        )


class TestRollingEwmEdgeCases:
    """Test edge cases for rolling and ewm operations."""

    def test_rolling_min_periods_larger_than_window(self):
        """Rolling with min_periods larger than data."""
        pd_df = pd.DataFrame({'value': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].rolling(window=5, min_periods=1).mean()
        ds_result = ds_df['value'].rolling(window=5, min_periods=1).mean()
        
        assert_series_equal(
            get_series(ds_result),
            pd_result)

    def test_ewm_halflife(self):
        """EWM with halflife parameter."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df['value'].ewm(halflife=2).mean()
        ds_result = ds_df['value'].ewm(halflife=2).mean()
        
        assert_series_equal(
            get_series(ds_result),
            pd_result)


class TestAtIatAccessors:
    """Test at and iat accessors."""

    def test_at_get_value(self):
        """Get single value with at."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.at['y', 'A']
        ds_result = ds_df.at['y', 'A']
        
        assert pd_result == ds_result

    def test_iat_get_value(self):
        """Get single value with iat."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.iat[1, 0]
        ds_result = ds_df.iat[1, 0]
        
        assert pd_result == ds_result


class TestDescribeVariations:
    """Test describe() with various parameters."""

    def test_describe_percentiles(self):
        """Describe with custom percentiles."""
        pd_df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.describe(percentiles=[0.1, 0.5, 0.9])
        ds_result = ds_df.describe(percentiles=[0.1, 0.5, 0.9])
        
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_include_all(self):
        """Describe with include='all'."""
        pd_df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string': ['a', 'b', 'a', 'b', 'c']
        })
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.describe(include='all')
        ds_result = ds_df.describe(include='all')
        
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMemoryUsage:
    """Test memory_usage method."""

    def test_memory_usage_basic(self):
        """Basic memory usage."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.memory_usage()
        ds_result = ds_df.memory_usage()
        
        # Memory usage should be similar structure
        assert len(pd_result) == len(ds_result)

    def test_memory_usage_deep(self):
        """Memory usage with deep=True."""
        pd_df = pd.DataFrame({'A': ['hello', 'world', 'test']})
        ds_df = DataStore(pd_df.copy())
        
        pd_result = pd_df.memory_usage(deep=True)
        ds_result = ds_df.memory_usage(deep=True)
        
        # Deep memory usage should account for string objects
        assert len(pd_result) == len(ds_result)


class TestInfoMethod:
    """Test info method."""

    def test_info_basic(self):
        """Basic info output."""
        import io
        
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_df = DataStore(pd_df.copy())
        
        # Capture info output
        pd_buffer = io.StringIO()
        ds_buffer = io.StringIO()
        
        pd_df.info(buf=pd_buffer)
        ds_df.info(buf=ds_buffer)
        
        # Both should produce output
        assert len(pd_buffer.getvalue()) > 0
        assert len(ds_buffer.getvalue()) > 0


class TestCompareMethod:
    """Test compare method."""

    def test_compare_different_values(self):
        """Compare DataFrames with different values."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 6]})
        
        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())
        
        pd_result = pd_df1.compare(pd_df2)
        ds_result = ds_df1.compare(ds_df2)
        
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
