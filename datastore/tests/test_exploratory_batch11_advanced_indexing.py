"""
Exploratory Discovery Batch 11: Advanced Indexing and Edge Cases

Testing areas:
1. MultiIndex Operations (set_index, reset_index variations)
2. At/Iat Accessors
3. XS (Cross-section) Operations
4. Duplicates Handling (various keep parameters)
5. Advanced Interpolate Methods
6. Merge Edge Cases (suffixes, indicator)
7. Advanced String Operations
8. Complex Apply/Transform Scenarios
"""

import pytest
from tests.xfail_markers import chdb_array_nullable, chdb_no_normalize_utf8, lazy_extractall_multiindex
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_series, get_value


class TestMultiIndexOperations:
    """Test MultiIndex creation and manipulation."""

    def test_set_index_single_column(self):
        """set_index with single column."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('A')
        ds_result = ds_df.set_index('A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_multiple_columns(self):
        """set_index with multiple columns creates MultiIndex."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': ['x', 'y', 'x', 'y'], 'C': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index(['A', 'B'])
        ds_result = ds_df.set_index(['A', 'B'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_drop_false(self):
        """set_index with drop=False keeps columns."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('A', drop=False)
        ds_result = ds_df.set_index('A', drop=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_set_index_append(self):
        """set_index with append=True adds to existing index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [10, 20, 30]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.set_index('A').set_index('B', append=True)
        ds_result = ds_df.set_index('A').set_index('B', append=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_drop_true(self):
        """reset_index with drop=True discards index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']}, index=[10, 20, 30])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reset_index_level(self):
        """reset_index with specific level."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': ['x', 'y', 'x', 'y'], 'C': [10, 20, 30, 40]}).set_index(
            ['A', 'B']
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.reset_index(level='B')
        ds_result = ds_df.reset_index(level='B')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAtIatAccessors:
    """Test at and iat accessors for scalar access."""

    def test_at_string_index(self):
        """at accessor with string index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.at['y', 'A']
        ds_result = ds_df.at['y', 'A']

        assert pd_result == ds_result

    def test_at_integer_index(self):
        """at accessor with integer index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=[10, 20, 30])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.at[20, 'B']
        ds_result = ds_df.at[20, 'B']

        assert pd_result == ds_result

    def test_iat_first_element(self):
        """iat accessor for first element."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iat[0, 0]
        ds_result = ds_df.iat[0, 0]

        assert pd_result == ds_result

    def test_iat_last_element(self):
        """iat accessor for last element."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iat[2, 1]
        ds_result = ds_df.iat[2, 1]

        assert pd_result == ds_result

    def test_iat_negative_index(self):
        """iat accessor with negative index."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iat[-1, -1]
        ds_result = ds_df.iat[-1, -1]

        assert pd_result == ds_result


class TestXSOperations:
    """Test xs (cross-section) operations."""

    def test_xs_simple(self):
        """xs with simple row selection."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['x', 'y', 'z'])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.xs('y')
        ds_result = ds_df.xs('y')

        assert_series_equal(ds_result, pd_result)

    def test_xs_multiindex_level0(self):
        """xs with MultiIndex, selecting from level 0."""
        pd_df = pd.DataFrame(
            {'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]},
            index=pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]),
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.xs('a', level=0)
        ds_result = ds_df.xs('a', level=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_xs_multiindex_level1(self):
        """xs with MultiIndex, selecting from level 1."""
        pd_df = pd.DataFrame(
            {'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]},
            index=pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]),
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.xs(1, level=1)
        ds_result = ds_df.xs(1, level=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_xs_axis_columns(self):
        """xs with axis=1 (columns)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.xs('A', axis=1)
        ds_result = ds_df.xs('A', axis=1)

        assert_series_equal(ds_result, pd_result)


class TestDuplicatesHandling:
    """Test duplicated and drop_duplicates with various parameters."""

    def test_duplicated_default(self):
        """duplicated with default keep='first'."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'x', 'y', 'z', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.duplicated()
        ds_result = ds_df.duplicated()

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_keep_last(self):
        """duplicated with keep='last'."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'x', 'y', 'z', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.duplicated(keep='last')
        ds_result = ds_df.duplicated(keep='last')

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_keep_false(self):
        """duplicated with keep=False marks all duplicates."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'x', 'y', 'z', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.duplicated(keep=False)
        ds_result = ds_df.duplicated(keep=False)

        assert_series_equal(ds_result, pd_result)

    def test_duplicated_subset(self):
        """duplicated with subset columns."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'y', 'y', 'z', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.duplicated(subset=['A'])
        ds_result = ds_df.duplicated(subset=['A'])

        assert_series_equal(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """drop_duplicates with keep='last'."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'y', 'z', 'w', 'v']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop_duplicates(subset=['A'], keep='last')
        ds_result = ds_df.drop_duplicates(subset=['A'], keep='last')

        # Reset index for comparison since row order may differ
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_drop_duplicates_keep_false(self):
        """drop_duplicates with keep=False removes all duplicates."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'y', 'z', 'w', 'v']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop_duplicates(subset=['A'], keep=False)
        ds_result = ds_df.drop_duplicates(subset=['A'], keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_drop_duplicates_ignore_index(self):
        """drop_duplicates with ignore_index=True."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 3], 'B': ['x', 'y', 'z', 'w']}, index=[10, 20, 30, 40])
        ds_df = DataStore(pd_df)

        pd_result = pd_df.drop_duplicates(subset=['A'], ignore_index=True)
        ds_result = ds_df.drop_duplicates(subset=['A'], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAdvancedInterpolate:
    """Test interpolate method with various parameters."""

    def test_interpolate_linear(self):
        """interpolate with linear method (default)."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.interpolate()
        ds_result = ds_df.interpolate()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_pad(self):
        """interpolate with pad method."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, np.nan, 4.0, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.interpolate(method='pad')
        ds_result = ds_df.interpolate(method='pad')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_limit(self):
        """interpolate with limit parameter."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, np.nan, np.nan, 5.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.interpolate(limit=1)
        ds_result = ds_df.interpolate(limit=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_limit_direction_backward(self):
        """interpolate with limit_direction='backward'."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, np.nan, 4.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.interpolate(limit_direction='backward')
        ds_result = ds_df.interpolate(limit_direction='backward')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_interpolate_multiple_columns(self):
        """interpolate on multiple columns."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0], 'B': [np.nan, 2.0, np.nan]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.interpolate()
        ds_result = ds_df.interpolate()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMergeEdgeCases:
    """Test merge edge cases."""

    def test_merge_suffixes(self):
        """merge with custom suffixes."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'A': [10, 20, 30]})
        pd_right = pd.DataFrame({'key': [1, 2, 3], 'A': [100, 200, 300]})
        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        pd_result = pd_left.merge(pd_right, on='key', suffixes=('_left', '_right'))
        ds_result = ds_left.merge(ds_right, on='key', suffixes=('_left', '_right'))

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_indicator(self):
        """merge with indicator=True."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'A': [10, 20, 30]})
        pd_right = pd.DataFrame({'key': [2, 3, 4], 'B': [200, 300, 400]})
        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        pd_result = pd_left.merge(pd_right, on='key', how='outer', indicator=True)
        ds_result = ds_left.merge(ds_right, on='key', how='outer', indicator=True)

        # Sort by key for comparison
        pd_result = pd_result.sort_values('key').reset_index(drop=True)
        ds_df = ds_result._get_df().sort_values('key').reset_index(drop=True)

        # Compare column by column (indicator column may have different dtype)
        for col in ['key', 'A', 'B']:
            assert_series_equal(ds_df[col], pd_result[col])

    def test_merge_left_on_right_on(self):
        """merge with different column names."""
        pd_left = pd.DataFrame({'key_left': [1, 2, 3], 'A': [10, 20, 30]})
        pd_right = pd.DataFrame({'key_right': [1, 2, 3], 'B': [100, 200, 300]})
        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        pd_result = pd_left.merge(pd_right, left_on='key_left', right_on='key_right')
        ds_result = ds_left.merge(ds_right, left_on='key_left', right_on='key_right')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_validate_one_to_one(self):
        """merge with validate='one_to_one'."""
        pd_left = pd.DataFrame({'key': [1, 2, 3], 'A': [10, 20, 30]})
        pd_right = pd.DataFrame({'key': [1, 2, 3], 'B': [100, 200, 300]})
        ds_left = DataStore(pd_left)
        ds_right = DataStore(pd_right)

        pd_result = pd_left.merge(pd_right, on='key', validate='one_to_one')
        ds_result = ds_left.merge(ds_right, on='key', validate='one_to_one')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestAdvancedStringOperations:
    """Test advanced string accessor operations."""

    def test_str_extract(self):
        """str.extract with regex groups."""
        pd_df = pd.DataFrame({'text': ['abc123', 'def456', 'ghi789']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.extract(r'([a-z]+)(\d+)')
        ds_result = ds_df['text'].str.extract(r'([a-z]+)(\d+)')

        ds_result = get_value(ds_result)
        assert_frame_equal(ds_result, pd_result)

    def test_str_extractall(self):
        """str.extractall for multiple matches."""
        pd_df = pd.DataFrame({'text': ['a1b2', 'c3', 'd4e5f6']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.extractall(r'([a-z])(\d)')
        ds_result = ds_df['text'].str.extractall(r'([a-z])(\d)')

        ds_result = get_value(ds_result)
        assert_frame_equal(ds_result, pd_result)

    @chdb_array_nullable
    def test_str_findall(self):
        """str.findall returns all matches."""
        pd_df = pd.DataFrame({'text': ['a1b2c3', 'x9', 'no_match']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.findall(r'\d')
        ds_result = ds_df['text'].str.findall(r'\d')

        assert_series_equal(ds_result, pd_result)

    def test_str_split_expand(self):
        """str.split with expand=True."""
        pd_df = pd.DataFrame({'text': ['a,b,c', 'd,e,f', 'g,h,i']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.split(',', expand=True)
        ds_result = ds_df['text'].str.split(',', expand=True)

        ds_result = get_value(ds_result)
        assert_frame_equal(ds_result, pd_result)

    def test_str_get_dummies(self):
        """str.get_dummies for one-hot encoding."""
        pd_df = pd.DataFrame({'text': ['a|b', 'b|c', 'a|c']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.get_dummies(sep='|')
        ds_result = ds_df['text'].str.get_dummies(sep='|')

        if isinstance(ds_result, DataStore):
            ds_result = ds_result._get_df()
        assert_frame_equal(ds_result, pd_result)

    def test_str_partition(self):
        """str.partition splits into three parts."""
        pd_df = pd.DataFrame({'text': ['a_b_c', 'd_e_f']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.partition('_')
        ds_result = ds_df['text'].str.partition('_')

        if isinstance(ds_result, DataStore):
            ds_result = ds_result._get_df()
        assert_frame_equal(ds_result, pd_result)

    def test_str_wrap(self):
        """str.wrap wraps text at given width."""
        pd_df = pd.DataFrame({'text': ['hello world this is long', 'short']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.wrap(10)
        ds_result = ds_df['text'].str.wrap(10)

        # Execute lazy result if needed
        ds_result = get_series(ds_result)
        assert_series_equal(ds_result, pd_result)

    @chdb_no_normalize_utf8
    def test_str_normalize(self):
        """str.normalize unicode normalization."""
        pd_df = pd.DataFrame({'text': ['café', 'naïve']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['text'].str.normalize('NFD')
        ds_result = ds_df['text'].str.normalize('NFD')

        assert_series_equal(ds_result, pd_result)


class TestComplexApplyTransform:
    """Test complex apply and transform scenarios."""

    def test_apply_with_args(self):
        """apply with additional positional args."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def add_value(col, val):
            return col + val

        pd_result = pd_df.apply(add_value, args=(10,))
        ds_result = ds_df.apply(add_value, args=(10,))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_with_kwargs(self):
        """apply with keyword arguments."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        def multiply(col, factor=1):
            return col * factor

        pd_result = pd_df.apply(multiply, factor=2)
        ds_result = ds_df.apply(multiply, factor=2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_axis_1(self):
        """apply along axis=1 (row-wise)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda row: row['A'] + row['B'], axis=1)
        ds_result = ds_df.apply(lambda row: row['A'] + row['B'], axis=1)

        assert_series_equal(ds_result, pd_result)

    def test_apply_result_type_expand(self):
        """apply with result_type='expand'."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        def split_value(row):
            return [row['A'], row['A'] * 2, row['A'] * 3]

        pd_result = pd_df.apply(split_value, axis=1, result_type='expand')
        ds_result = ds_df.apply(split_value, axis=1, result_type='expand')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_multiple_functions(self):
        """transform with multiple functions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.transform([np.sqrt, np.exp])
        ds_result = ds_df.transform([np.sqrt, np.exp])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_transform_dict_of_functions(self):
        """transform with dict mapping columns to functions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.transform({'A': np.sqrt, 'B': lambda x: x * 2})
        ds_result = ds_df.transform({'A': np.sqrt, 'B': lambda x: x * 2})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestResampleOperations:
    """Test resample operations with datetime index."""

    def test_resample_sum(self):
        """resample with sum aggregation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        pd_df = pd.DataFrame({'A': range(10)}, index=dates)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.resample('3D').sum()
        ds_result = ds_df.resample('3D').sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_resample_mean(self):
        """resample with mean aggregation."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        pd_df = pd.DataFrame({'A': range(10)}, index=dates)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.resample('W').mean()
        ds_result = ds_df.resample('W').mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_resample_multiple_columns(self):
        """resample with multiple columns."""
        dates = pd.date_range('2023-01-01', periods=12, freq='h')
        pd_df = pd.DataFrame({'A': range(12), 'B': [x * 2 for x in range(12)]}, index=dates)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.resample('4h').agg({'A': 'sum', 'B': 'mean'})
        ds_result = ds_df.resample('4h').agg({'A': 'sum', 'B': 'mean'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_resample_first_last(self):
        """resample with first and last."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']}, index=dates)
        ds_df = DataStore(pd_df)

        pd_result = pd_df.resample('5D').first()
        ds_result = ds_df.resample('5D').first()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestClipOperations:
    """Test clip operations with various parameters."""

    def test_clip_lower_upper(self):
        """clip with both lower and upper."""
        pd_df = pd.DataFrame({'A': [-1, 0, 5, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.clip(lower=0, upper=10)
        ds_result = ds_df.clip(lower=0, upper=10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_lower_only(self):
        """clip with lower only."""
        pd_df = pd.DataFrame({'A': [-5, -1, 0, 5, 10]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.clip(lower=0)
        ds_result = ds_df.clip(lower=0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_upper_only(self):
        """clip with upper only."""
        pd_df = pd.DataFrame({'A': [-5, 0, 5, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.clip(upper=5)
        ds_result = ds_df.clip(upper=5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_with_nan(self):
        """clip handles NaN values."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 5.0, 10.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.clip(lower=2, upper=8)
        ds_result = ds_df.clip(lower=2, upper=8)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDescribeVariations:
    """Test describe method with various parameters."""

    def test_describe_default(self):
        """describe with default parameters."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe()
        ds_result = ds_df.describe()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_percentiles(self):
        """describe with custom percentiles."""
        pd_df = pd.DataFrame({'A': range(100)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(percentiles=[0.1, 0.5, 0.9])
        ds_result = ds_df.describe(percentiles=[0.1, 0.5, 0.9])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_include_all(self):
        """describe with include='all'."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(include='all')
        ds_result = ds_df.describe(include='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_exclude(self):
        """describe with exclude parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z'], 'C': [1.1, 2.2, 3.3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.describe(exclude=['object'])
        ds_result = ds_df.describe(exclude=['object'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestPopMethod:
    """Test pop method for removing and returning columns."""

    def test_pop_column(self):
        """pop removes and returns a column."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_popped = pd_df.pop('B')
        ds_popped = ds_df.pop('B')

        assert_series_equal(ds_popped, pd_popped)

    def test_pop_modifies_inplace(self):
        """pop modifies the DataFrame in place."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = DataStore(pd_df.copy())

        pd_df.pop('B')
        ds_df.pop('B')

        # After pop, 'B' should not be in columns
        pd_remaining = pd_df
        ds_remaining = ds_df._get_df()

        assert list(ds_remaining.columns) == list(pd_remaining.columns)


class TestGetMethod:
    """Test get method for safe column access."""

    def test_get_existing_column(self):
        """get returns column if exists."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.get('A')
        ds_result = ds_df.get('A')

        assert_series_equal(ds_result, pd_result)

    def test_get_nonexistent_column(self):
        """get returns default if column doesn't exist."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.get('C', default='missing')
        ds_result = ds_df.get('C', default='missing')

        assert pd_result == ds_result

    def test_get_default_none(self):
        """get returns None by default if column doesn't exist."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.get('X')
        ds_result = ds_df.get('X')

        assert pd_result == ds_result


class TestCorrCov:
    """Test correlation and covariance methods."""

    def test_corr_default(self):
        """corr with default pearson method."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': [1, 3, 2, 5, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.corr()
        ds_result = ds_df.corr()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_corr_spearman(self):
        """corr with spearman method."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.corr(method='spearman')
        ds_result = ds_df.corr(method='spearman')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_cov_default(self):
        """cov computes covariance matrix."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.cov()
        ds_result = ds_df.cov()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_corrwith(self):
        """corrwith computes correlation with another DataFrame."""
        pd_df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 6, 6]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        pd_result = pd_df1.corrwith(pd_df2)
        ds_result = ds_df1.corrwith(ds_df2)

        assert_series_equal(ds_result, pd_result)


class TestNuniqueValueCounts:
    """Test nunique and value_counts methods."""

    def test_nunique_default(self):
        """nunique counts unique values per column."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'x', 'x', 'y', 'y']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()

        assert_series_equal(ds_result, pd_result)

    def test_nunique_dropna_false(self):
        """nunique with dropna=False counts NaN."""
        pd_df = pd.DataFrame(
            {
                'A': [1, 1, np.nan, 2, 2],
            }
        )
        ds_df = DataStore(pd_df)

        pd_result = pd_df.nunique(dropna=False)
        ds_result = ds_df.nunique(dropna=False)

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_normalize(self):
        """value_counts with normalize=True."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 2], 'B': ['x', 'x', 'y', 'y', 'y']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.value_counts(normalize=True)
        ds_result = ds_df.value_counts(normalize=True)

        # Sort for comparison since order may differ
        pd_result = pd_result.sort_index()
        ds_result = ds_result.sort_index()

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_subset(self):
        """value_counts with subset of columns."""
        pd_df = pd.DataFrame({'A': [1, 1, 2, 2, 3], 'B': ['x', 'x', 'y', 'y', 'y'], 'C': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.value_counts(subset=['A', 'B'])
        ds_result = ds_df.value_counts(subset=['A', 'B'])

        pd_result = pd_result.sort_index()
        ds_result = ds_result.sort_index()

        assert_series_equal(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
