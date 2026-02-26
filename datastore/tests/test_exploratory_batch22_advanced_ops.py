"""
Exploratory Discovery Batch 22 - Advanced Operations and Edge Cases

Focus areas:
1. GroupBy cumcount and related operations
2. Index operations with complex scenarios
3. replace/fillna edge cases
4. Complex operation chains with assign/transform
5. Series operations edge cases
6. DataFrame stack/unstack edge cases
7. Merge/concat edge cases
8. String accessor edge cases
"""

import pytest
from tests.xfail_markers import pandas_deprecated_fillna_downcast
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_frame_equal,
    assert_series_equal,
    get_dataframe,
    get_series,
)


class TestGroupByCumcount:
    """Test groupby().cumcount() operation."""

    def test_cumcount_basic(self):
        """Basic cumcount - count within each group."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'A', 'B', 'B'], 'value': [1, 2, 3, 4, 5, 6]})
        pd_result = pd_df.groupby('category').cumcount()

        ds_df = DataStore(pd_df)
        ds_result = ds_df.groupby('category').cumcount()

        # cumcount returns a Series with same index as original DataFrame
        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_cumcount_ascending_false(self):
        """cumcount with ascending=False (reverse within group)."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'A', 'B', 'B'], 'value': [1, 2, 3, 4, 5, 6]})
        pd_result = pd_df.groupby('category').cumcount(ascending=False)

        ds_df = DataStore(pd_df)
        ds_result = ds_df.groupby('category').cumcount(ascending=False)

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_cumcount_multiple_groups(self):
        """cumcount with multiple groupby columns."""
        pd_df = pd.DataFrame(
            {
                'cat1': ['A', 'A', 'A', 'B', 'B', 'B'],
                'cat2': ['X', 'X', 'Y', 'X', 'Y', 'Y'],
                'value': [1, 2, 3, 4, 5, 6],
            }
        )
        pd_result = pd_df.groupby(['cat1', 'cat2']).cumcount()

        ds_df = DataStore(pd_df)
        ds_result = ds_df.groupby(['cat1', 'cat2']).cumcount()

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )


class TestGroupByPipe:
    """Test groupby().pipe() operation."""

    def test_pipe_with_function(self):
        """Test pipe with a simple function."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})

        def get_summary(grp):
            return grp.agg({'value': ['mean', 'sum']})

        pd_result = pd_df.groupby('category').pipe(get_summary)

        ds_df = DataStore(pd_df)
        ds_result = ds_df.groupby('category').pipe(get_summary)

        # Compare results - pipe returns what the function returns
        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )


class TestIndexOperations:
    """Test index operations edge cases."""

    def test_set_index_append(self):
        """set_index with append=True."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        pd_df = pd_df.set_index('A')
        pd_result = pd_df.set_index('B', append=True)

        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = ds_df.set_index('A')
        ds_result = ds_df.set_index('B', append=True)

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_reset_index_level(self):
        """reset_index with level parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        pd_df = pd_df.set_index(['A', 'B'])
        pd_result = pd_df.reset_index(level='A')

        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = ds_df.set_index(['A', 'B'])
        ds_result = ds_df.reset_index(level='A')

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_set_index_drop_false(self):
        """set_index with drop=False keeps column."""
        pd_df = pd.DataFrame(
            {
                'A': [1, 2, 3],
                'B': [4, 5, 6],
            }
        )
        pd_result = pd_df.set_index('A', drop=False)

        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df.set_index('A', drop=False)

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_set_index_verify_integrity(self):
        """set_index with verify_integrity=True on duplicate values."""
        pd_df = pd.DataFrame(
            {
                'A': [1, 1, 3],  # Duplicates
                'B': [4, 5, 6],
            }
        )

        ds_df = DataStore({'A': [1, 1, 3], 'B': [4, 5, 6]})

        # Both should raise ValueError
        with pytest.raises((ValueError, Exception)):
            pd_df.set_index('A', verify_integrity=True)

        with pytest.raises((ValueError, Exception)):
            ds_df.set_index('A', verify_integrity=True)


class TestReplaceFillnaEdgeCases:
    """Test replace and fillna edge cases."""

    def test_replace_regex_pattern(self):
        """replace with regex=True."""
        pd_df = pd.DataFrame(
            {
                'A': ['foo', 'bar', 'foobar', 'baz'],
            }
        )
        pd_result = pd_df.replace(r'^foo.*', 'replaced', regex=True)

        ds_df = DataStore({'A': ['foo', 'bar', 'foobar', 'baz']})
        ds_result = ds_df.replace(r'^foo.*', 'replaced', regex=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_method_bfill_ffill(self):
        """replace with method parameter."""
        pd_df = pd.DataFrame(
            {
                'A': [1, np.nan, 3, np.nan, 5],
            }
        )

        # ffill (forward fill)
        pd_result_ffill = pd_df.ffill()
        ds_df = DataStore({'A': [1, np.nan, 3, np.nan, 5]})
        ds_result_ffill = ds_df.ffill()
        assert_datastore_equals_pandas(ds_result_ffill, pd_result_ffill)

        # bfill (backward fill)
        pd_result_bfill = pd_df.bfill()
        ds_df2 = DataStore({'A': [1, np.nan, 3, np.nan, 5]})
        ds_result_bfill = ds_df2.bfill()
        assert_datastore_equals_pandas(ds_result_bfill, pd_result_bfill)

    def test_fillna_limit(self):
        """fillna with limit parameter."""
        pd_df = pd.DataFrame(
            {
                'A': [1, np.nan, np.nan, np.nan, 5],
            }
        )
        pd_result = pd_df.ffill(limit=1)

        ds_df = DataStore({'A': [1, np.nan, np.nan, np.nan, 5]})
        ds_result = ds_df.ffill(limit=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_deprecated_fillna_downcast
    def test_fillna_downcast(self):
        """fillna with downcast parameter."""
        pd_df = pd.DataFrame(
            {
                'A': [1.0, np.nan, 3.0],
            }
        )
        # downcast='infer' will convert float to int if possible
        pd_result = pd_df.fillna(2, downcast='infer')

        ds_df = DataStore({'A': [1.0, np.nan, 3.0]})
        ds_result = ds_df.fillna(2, downcast='infer')

        # Just check values match, dtype might differ
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict_with_nested(self):
        """replace with nested dict per column."""
        pd_df = pd.DataFrame(
            {
                'A': [1, 2, 3],
                'B': ['a', 'b', 'c'],
            }
        )
        pd_result = pd_df.replace({'A': {1: 100}, 'B': {'a': 'alpha'}})

        ds_df = DataStore({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_result = ds_df.replace({'A': {1: 100}, 'B': {'a': 'alpha'}})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAssignTransformChains:
    """Test complex operation chains with assign/transform."""

    def test_assign_then_transform(self):
        """assign followed by groupby transform."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        pd_df = pd_df.assign(doubled=pd_df['value'] * 2)
        pd_result = pd_df.groupby('category')['doubled'].transform('mean')

        ds_df = DataStore({'category': ['A', 'A', 'B', 'B'], 'value': [10, 20, 30, 40]})
        ds_df = ds_df.assign(doubled=ds_df['value'] * 2)
        ds_result = ds_df.groupby('category')['doubled'].transform('mean')

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_filter_assign_filter(self):
        """Filter -> assign -> filter chain."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        pd_result = pd_df[pd_df['A'] > 1].assign(C=lambda x: x['A'] + x['B'])
        pd_result = pd_result[pd_result['C'] > 35]

        ds_df = DataStore({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        ds_result = ds_df[ds_df['A'] > 1].assign(C=lambda x: x['A'] + x['B'])
        ds_result = ds_result[ds_result['C'] > 35]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_columns_order(self):
        """assign multiple columns - order should be preserved."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        pd_result = pd_df.assign(B=lambda x: x['A'] * 2, C=lambda x: x['A'] * 3, D=lambda x: x['A'] * 4)

        ds_df = DataStore({'A': [1, 2, 3]})
        ds_result = ds_df.assign(B=lambda x: x['A'] * 2, C=lambda x: x['A'] * 3, D=lambda x: x['A'] * 4)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSeriesOperationsEdge:
    """Test Series operations edge cases."""

    def test_series_drop_duplicates_keep_false(self):
        """drop_duplicates with keep=False removes all duplicates."""
        # Mirror pattern: both use DataFrame column access to preserve Series name
        pd_df = pd.DataFrame({'A': [1, 1, 2, 3, 3]})
        pd_result = pd_df['A'].drop_duplicates(keep=False)

        ds_df = DataStore({'A': [1, 1, 2, 3, 3]})
        ds_result = ds_df['A'].drop_duplicates(keep=False)

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_series_nlargest_keep(self):
        """nlargest with keep parameter."""
        # Mirror pattern: both use DataFrame column access to preserve Series name
        pd_df = pd.DataFrame({'A': [1, 3, 3, 5, 5, 2]})
        pd_result = pd_df['A'].nlargest(3, keep='all')

        ds_df = DataStore({'A': [1, 3, 3, 5, 5, 2]})
        ds_result = ds_df['A'].nlargest(3, keep='all')

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_series_nsmallest_keep(self):
        """nsmallest with keep parameter."""
        # Mirror pattern: both use DataFrame column access to preserve Series name
        pd_df = pd.DataFrame({'A': [1, 3, 1, 5, 2, 2]})
        pd_result = pd_df['A'].nsmallest(3, keep='last')

        ds_df = DataStore({'A': [1, 3, 1, 5, 2, 2]})
        ds_result = ds_df['A'].nsmallest(3, keep='last')

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_series_between_inclusive(self):
        """between with inclusive parameter."""
        # Mirror pattern: both use DataFrame column access to preserve Series name
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})

        # inclusive='both' (default)
        pd_result_both = pd_df['A'].between(2, 4, inclusive='both')
        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result_both = ds_df['A'].between(2, 4, inclusive='both')
        # between() returns LazyCondition - use _execute() to get Series
        ds_series = get_series(ds_result_both)
        assert_series_equal(
            ds_series,
            pd_result_both,
        )

        # inclusive='neither'
        pd_result_neither = pd_df['A'].between(2, 4, inclusive='neither')
        ds_df2 = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result_neither = ds_df2['A'].between(2, 4, inclusive='neither')
        # between() returns LazyCondition - use _execute() to get Series
        ds_series2 = get_series(ds_result_neither)
        assert_series_equal(
            ds_series2,
            pd_result_neither,
        )


class TestStackUnstackEdge:
    """Test stack/unstack edge cases."""

    def test_stack_level(self):
        """stack with level parameter."""
        pd_df = pd.DataFrame(
            {
                ('A', 'x'): [1, 2],
                ('A', 'y'): [3, 4],
                ('B', 'x'): [5, 6],
                ('B', 'y'): [7, 8],
            }
        )
        pd_df.columns = pd.MultiIndex.from_tuples([('A', 'x'), ('A', 'y'), ('B', 'x'), ('B', 'y')])
        pd_result = pd_df.stack(level=0)

        ds_df = DataStore(pd_df)
        ds_result = ds_df.stack(level=0)

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_unstack_fill_value(self):
        """unstack with fill_value."""
        pd_df = pd.DataFrame({'A': ['one', 'one', 'two'], 'B': ['a', 'b', 'a'], 'value': [1, 2, 3]})
        pd_df = pd_df.set_index(['A', 'B'])
        pd_result = pd_df.unstack(fill_value=0)

        ds_df = DataStore({'A': ['one', 'one', 'two'], 'B': ['a', 'b', 'a'], 'value': [1, 2, 3]})
        ds_df = ds_df.set_index(['A', 'B'])
        ds_result = ds_df.unstack(fill_value=0)

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )


class TestMergeConcatEdge:
    """Test merge/concat edge cases."""

    def test_merge_indicator(self):
        """merge with indicator=True."""
        pd_left = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        pd_right = pd.DataFrame({'A': [2, 3, 4], 'C': ['x', 'y', 'z']})
        pd_result = pd.merge(pd_left, pd_right, on='A', how='outer', indicator=True)

        ds_left = DataStore({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_right = DataStore({'A': [2, 3, 4], 'C': ['x', 'y', 'z']})
        ds_result = ds_left.merge(ds_right, on='A', how='outer', indicator=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_suffixes(self):
        """merge with custom suffixes."""
        pd_left = pd.DataFrame({'A': [1, 2], 'B': [10, 20]})
        pd_right = pd.DataFrame({'A': [1, 2], 'B': [100, 200]})
        pd_result = pd.merge(pd_left, pd_right, on='A', suffixes=('_left', '_right'))

        ds_left = DataStore({'A': [1, 2], 'B': [10, 20]})
        ds_right = DataStore({'A': [1, 2], 'B': [100, 200]})
        ds_result = ds_left.merge(ds_right, on='A', suffixes=('_left', '_right'))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_keys(self):
        """concat with keys parameter."""
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        pd_result = pd.concat([pd_df1, pd_df2], keys=['first', 'second'])

        ds_df1 = DataStore({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = DataStore({'A': [5, 6], 'B': [7, 8]})
        from datastore import concat as ds_concat

        ds_result = ds_concat([ds_df1, ds_df2], keys=['first', 'second'])

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_concat_ignore_index(self):
        """concat with ignore_index=True."""
        pd_df1 = pd.DataFrame({'A': [1, 2]}, index=[10, 11])
        pd_df2 = pd.DataFrame({'A': [3, 4]}, index=[20, 21])
        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)

        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        from datastore import concat as ds_concat

        ds_result = ds_concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringAccessorEdge:
    """Test string accessor edge cases."""

    def test_str_cat_with_sep(self):
        """str.cat with sep and na_rep."""
        pd_ser = pd.Series(['a', 'b', None, 'd'])
        pd_result = pd_ser.str.cat(sep='-', na_rep='NA')

        ds_df = DataStore({'A': ['a', 'b', None, 'd']})
        ds_result = ds_df['A'].str.cat(sep='-', na_rep='NA')

        # str.cat returns a scalar when no other argument
        assert ds_result == pd_result

    def test_str_cat_with_others(self):
        """str.cat concatenating with another Series."""
        # Mirror pattern: both use DataFrame column access to preserve Series name
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
        pd_result = pd_df['A'].str.cat(pd_df['B'], sep='-')

        ds_df = DataStore({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
        # Get the Series for str.cat
        # ColumnExpr needs to execute first, then use the Series
        ds_result = ds_df['A'].str.cat(get_series(ds_df['B']), sep='-')

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )

    def test_str_get_dummies(self):
        """str.get_dummies - one-hot encoding from string."""
        pd_ser = pd.Series(['a|b', 'b|c', 'c'])
        pd_result = pd_ser.str.get_dummies(sep='|')

        ds_df = DataStore({'A': ['a|b', 'b|c', 'c']})
        ds_result = ds_df['A'].str.get_dummies(sep='|')

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_str_encode_decode(self):
        """str.encode and str.decode."""
        # Mirror pattern: both use DataFrame column access to preserve Series name
        pd_df = pd.DataFrame({'A': ['hello', 'world']})
        pd_encoded = pd_df['A'].str.encode('utf-8')
        pd_decoded = pd_encoded.str.decode('utf-8')

        ds_df = DataStore({'A': ['hello', 'world']})
        ds_encoded = ds_df['A'].str.encode('utf-8')
        ds_decoded = ds_encoded.str.decode('utf-8')

        # Execute ColumnExpr if needed
        ds_series = get_series(ds_decoded)
        assert_series_equal(
            ds_series,
            pd_decoded,
        )


class TestDataFrameMethodsEdge:
    """Test DataFrame methods edge cases."""

    def test_transpose(self):
        """Test DataFrame transpose."""
        pd_df = pd.DataFrame(
            {
                'A': [1, 2, 3],
                'B': [4, 5, 6],
            }
        )
        pd_result = pd_df.T

        ds_df = DataStore({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_result = ds_df.T

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_dot_product(self):
        """DataFrame dot product."""
        pd_df1 = pd.DataFrame([[1, 2], [3, 4]])
        pd_df2 = pd.DataFrame([[5], [6]])
        pd_result = pd_df1.dot(pd_df2)

        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)
        ds_result = ds_df1.dot(ds_df2._get_df())

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_diff_periods(self):
        """diff with periods parameter."""
        pd_df = pd.DataFrame({'A': [1, 3, 6, 10]})

        # periods=1 (default)
        pd_result_1 = pd_df.diff(periods=1)
        ds_df = DataStore({'A': [1, 3, 6, 10]})
        ds_result_1 = ds_df.diff(periods=1)
        assert_datastore_equals_pandas(ds_result_1, pd_result_1)

        # periods=2
        pd_result_2 = pd_df.diff(periods=2)
        ds_df2 = DataStore({'A': [1, 3, 6, 10]})
        ds_result_2 = ds_df2.diff(periods=2)
        assert_datastore_equals_pandas(ds_result_2, pd_result_2)

    def test_pct_change_periods(self):
        """pct_change with periods parameter."""
        pd_df = pd.DataFrame({'A': [10, 15, 20, 25]})

        # periods=1 (default)
        pd_result_1 = pd_df.pct_change(periods=1)
        ds_df = DataStore({'A': [10, 15, 20, 25]})
        ds_result_1 = ds_df.pct_change(periods=1)
        assert_datastore_equals_pandas(ds_result_1, pd_result_1)

        # periods=2
        pd_result_2 = pd_df.pct_change(periods=2)
        ds_df2 = DataStore({'A': [10, 15, 20, 25]})
        ds_result_2 = ds_df2.pct_change(periods=2)
        assert_datastore_equals_pandas(ds_result_2, pd_result_2)

    def test_shift_fill_value(self):
        """shift with fill_value parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4]})
        pd_result = pd_df.shift(periods=2, fill_value=0)

        ds_df = DataStore({'A': [1, 2, 3, 4]})
        ds_result = ds_df.shift(periods=2, fill_value=0)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSpecialValues:
    """Test special value handling."""

    def test_inf_operations(self):
        """Operations with infinity values."""
        pd_df = pd.DataFrame(
            {
                'A': [1, np.inf, -np.inf, 4],
            }
        )

        # replace inf with nan
        pd_result = pd_df.replace([np.inf, -np.inf], np.nan)
        ds_df = DataStore({'A': [1, np.inf, -np.inf, 4]})
        ds_result = ds_df.replace([np.inf, -np.inf], np.nan)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_with_inf(self):
        """clip operations with infinity."""
        pd_df = pd.DataFrame(
            {
                'A': [1, 100, -100, 50],
            }
        )
        pd_result = pd_df.clip(lower=-np.inf, upper=60)

        ds_df = DataStore({'A': [1, 100, -100, 50]})
        ds_result = ds_df.clip(lower=-np.inf, upper=60)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nan_propagation_in_arithmetic(self):
        """NaN propagation in arithmetic operations."""
        pd_df = pd.DataFrame(
            {
                'A': [1, np.nan, 3],
                'B': [4, 5, np.nan],
            }
        )
        pd_result = pd_df['A'] + pd_df['B']

        ds_df = DataStore({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        ds_result = ds_df['A'] + ds_df['B']

        # Execute ColumnExpr/LazySeries if needed
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series,
            pd_result,
        )


class TestGroupByAdvanced:
    """Advanced groupby operations."""

    def test_groupby_sort_false(self):
        """groupby with sort=False preserves original order."""
        pd_df = pd.DataFrame({'category': ['B', 'A', 'B', 'A'], 'value': [1, 2, 3, 4]})
        pd_result = pd_df.groupby('category', sort=False).sum()

        ds_df = DataStore({'category': ['B', 'A', 'B', 'A'], 'value': [1, 2, 3, 4]})
        ds_result = ds_df.groupby('category', sort=False).sum()

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )

    def test_groupby_dropna(self):
        """groupby with dropna parameter."""
        pd_df = pd.DataFrame({'category': ['A', 'B', None, 'A', None], 'value': [1, 2, 3, 4, 5]})

        # dropna=True (default) - exclude NaN groups
        pd_result_true = pd_df.groupby('category', dropna=True).sum()
        ds_df = DataStore({'category': ['A', 'B', None, 'A', None], 'value': [1, 2, 3, 4, 5]})
        ds_result_true = ds_df.groupby('category', dropna=True).sum()

        assert_frame_equal(
            get_dataframe(ds_result_true),
            pd_result_true,
        )

    def test_groupby_agg_list_functions(self):
        """groupby agg with list of functions."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        pd_result = pd_df.groupby('category').agg(['sum', 'mean', 'count'])

        ds_df = DataStore({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_result = ds_df.groupby('category').agg(['sum', 'mean', 'count'])

        assert_frame_equal(
            get_dataframe(ds_result),
            pd_result,
        )


class TestIOEdgeCases:
    """Test I/O operations edge cases."""

    def test_to_csv_with_compression(self, tmp_path):
        """to_csv with compression parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        csv_path = tmp_path / "test.csv.gz"
        pd_df.to_csv(csv_path, compression='gzip', index=False)
        pd_read = pd.read_csv(csv_path, compression='gzip')

        ds_df = DataStore({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        ds_csv_path = tmp_path / "ds_test.csv.gz"
        ds_df.to_csv(ds_csv_path, compression='gzip', index=False)
        ds_read = pd.read_csv(ds_csv_path, compression='gzip')

        assert_frame_equal(ds_read, pd_read)

    def test_to_dict_orient_records(self):
        """to_dict with orient='records'."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
        pd_result = pd_df.to_dict(orient='records')

        ds_df = DataStore({'A': [1, 2], 'B': ['x', 'y']})
        ds_result = ds_df.to_dict(orient='records')

        assert ds_result == pd_result

    def test_to_dict_orient_index(self):
        """to_dict with orient='index'."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']}, index=['row1', 'row2'])
        pd_result = pd_df.to_dict(orient='index')

        ds_df = DataStore(pd_df)
        ds_result = ds_df.to_dict(orient='index')

        assert ds_result == pd_result


class TestRollingExpandingEdge:
    """Test rolling/expanding edge cases."""

    def test_rolling_min_periods(self):
        """rolling with min_periods parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.rolling(window=3, min_periods=1).mean()

        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df.rolling(window=3, min_periods=1).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rolling_center(self):
        """rolling with center=True."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.rolling(window=3, center=True).mean()

        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df.rolling(window=3, center=True).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_expanding_min_periods(self):
        """expanding with min_periods parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.expanding(min_periods=2).sum()

        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df.expanding(min_periods=2).sum()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_ewm_alpha(self):
        """ewm with alpha parameter."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        pd_result = pd_df.ewm(alpha=0.5).mean()

        ds_df = DataStore({'A': [1, 2, 3, 4, 5]})
        ds_result = ds_df.ewm(alpha=0.5).mean()

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
