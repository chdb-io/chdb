"""
Exploratory Batch 76: Boundary Conditions and Method Chains

Focus areas:
1. Self-join operations (merge DataFrame with itself)
2. Complex nested boolean conditions
3. Numeric edge cases (inf, -inf, large numbers)
4. Method chains with type transitions
5. Operations on single-row/single-column DataFrames
6. Cumulative operations edge cases
7. String accessor edge cases with special characters
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestSelfJoinOperations:
    """Test merge/join operations with the same DataFrame"""

    def test_self_join_basic(self):
        """Test basic self-join on same key column"""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['a', 'b', 'c'],
            'value': [10, 20, 30]
        })
        ds_df = DataStore({
            'id': [1, 2, 3],
            'name': ['a', 'b', 'c'],
            'value': [10, 20, 30]
        })

        pd_result = pd_df.merge(pd_df, on='id', suffixes=('_left', '_right'))
        ds_result = ds_df.merge(ds_df, on='id', suffixes=('_left', '_right'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_self_join_different_columns(self):
        """Test self-join using different columns for left and right"""
        pd_df = pd.DataFrame({
            'parent_id': [None, 1, 1, 2],
            'id': [1, 2, 3, 4],
            'name': ['root', 'child1', 'child2', 'grandchild']
        })
        ds_df = DataStore({
            'parent_id': [None, 1, 1, 2],
            'id': [1, 2, 3, 4],
            'name': ['root', 'child1', 'child2', 'grandchild']
        })

        pd_result = pd_df.merge(
            pd_df,
            left_on='parent_id',
            right_on='id',
            how='left',
            suffixes=('', '_parent')
        )
        ds_result = ds_df.merge(
            ds_df,
            left_on='parent_id',
            right_on='id',
            how='left',
            suffixes=('', '_parent')
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_self_join_filter_then_merge(self):
        """Test filtering a DataFrame then merging with original"""
        pd_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })
        ds_df = DataStore({
            'id': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50]
        })

        pd_filtered = pd_df[pd_df['category'] == 'A']
        pd_result = pd_df.merge(pd_filtered, on='id', how='inner', suffixes=('', '_filtered'))

        ds_filtered = ds_df[ds_df['category'] == 'A']
        ds_result = ds_df.merge(ds_filtered, on='id', how='inner', suffixes=('', '_filtered'))

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNestedBooleanConditions:
    """Test complex nested boolean conditions"""

    def test_triple_and_conditions(self):
        """Test A & B & C condition"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6]
        })

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] > 2) & (pd_df['c'] < 5)]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] > 2) & (ds_df['c'] < 5)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_triple_or_conditions(self):
        """Test A | B | C condition"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6]
        })

        pd_result = pd_df[(pd_df['a'] > 4) | (pd_df['b'] > 4) | (pd_df['c'] > 5)]
        ds_result = ds_df[(ds_df['a'] > 4) | (ds_df['b'] > 4) | (ds_df['c'] > 5)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mixed_and_or_conditions(self):
        """Test (A & B) | (C & D) condition"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6],
            'd': [6, 5, 4, 3, 2]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6],
            'd': [6, 5, 4, 3, 2]
        })

        pd_result = pd_df[((pd_df['a'] > 2) & (pd_df['b'] > 2)) | ((pd_df['c'] > 4) & (pd_df['d'] < 4))]
        ds_result = ds_df[((ds_df['a'] > 2) & (ds_df['b'] > 2)) | ((ds_df['c'] > 4) & (ds_df['d'] < 4))]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_deeply_nested_conditions(self):
        """Test ((A & B) | C) & (D | E) condition"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6],
            'd': [6, 5, 4, 3, 2],
            'e': [1, 1, 1, 1, 1]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [2, 3, 4, 5, 6],
            'd': [6, 5, 4, 3, 2],
            'e': [1, 1, 1, 1, 1]
        })

        pd_cond = (((pd_df['a'] > 2) & (pd_df['b'] < 4)) | (pd_df['c'] == 4)) & ((pd_df['d'] > 3) | (pd_df['e'] == 1))
        ds_cond = (((ds_df['a'] > 2) & (ds_df['b'] < 4)) | (ds_df['c'] == 4)) & ((ds_df['d'] > 3) | (ds_df['e'] == 1))

        pd_result = pd_df[pd_cond]
        ds_result = ds_df[ds_cond]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNumericEdgeCases:
    """Test numeric boundary conditions"""

    def test_infinity_values(self):
        """Test operations with infinity values"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, -np.inf, 0.0, np.nan],
            'b': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        ds_df = DataStore({
            'a': [1.0, np.inf, -np.inf, 0.0, np.nan],
            'b': [2.0, 3.0, 4.0, 5.0, 6.0]
        })

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()
        assert pd_result == ds_result or (np.isnan(pd_result) and np.isnan(ds_result))

    def test_infinity_comparison(self):
        """Test comparison with infinity"""
        pd_df = pd.DataFrame({
            'a': [1.0, np.inf, -np.inf, 100.0, np.nan]
        })
        ds_df = DataStore({
            'a': [1.0, np.inf, -np.inf, 100.0, np.nan]
        })

        pd_result = pd_df[pd_df['a'] > 0]
        ds_result = ds_df[ds_df['a'] > 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_very_large_numbers(self):
        """Test arithmetic with very large numbers"""
        large_val = 10**15
        pd_df = pd.DataFrame({
            'a': [large_val, large_val + 1, large_val + 2],
            'b': [1, 2, 3]
        })
        ds_df = DataStore({
            'a': [large_val, large_val + 1, large_val + 2],
            'b': [1, 2, 3]
        })

        pd_result = pd_df['a'] + pd_df['b']
        ds_result = ds_df['a'] + ds_df['b']
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_very_small_numbers(self):
        """Test arithmetic with very small numbers"""
        small_val = 1e-15
        pd_df = pd.DataFrame({
            'a': [small_val, small_val * 2, small_val * 3],
            'b': [1e-15, 2e-15, 3e-15]
        })
        ds_df = DataStore({
            'a': [small_val, small_val * 2, small_val * 3],
            'b': [1e-15, 2e-15, 3e-15]
        })

        pd_result = pd_df['a'] * pd_df['b']
        ds_result = ds_df['a'] * ds_df['b']
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_integer_overflow_boundary(self):
        """Test values near integer overflow boundary"""
        max_int64 = np.iinfo(np.int64).max
        pd_df = pd.DataFrame({
            'a': [max_int64 - 2, max_int64 - 1, max_int64],
            'b': [1, 1, 0]
        })
        ds_df = DataStore({
            'a': [max_int64 - 2, max_int64 - 1, max_int64],
            'b': [1, 1, 0]
        })

        # Subtraction should work fine
        pd_result = pd_df['a'] - pd_df['b']
        ds_result = ds_df['a'] - ds_df['b']
        assert_series_equal(ds_result, pd_result, check_names=False)


class TestSingleRowColumnDataFrames:
    """Test operations on edge-case sized DataFrames"""

    def test_single_row_groupby(self):
        """Test groupby on single-row DataFrame"""
        pd_df = pd.DataFrame({'cat': ['A'], 'val': [10]})
        ds_df = DataStore({'cat': ['A'], 'val': [10]})

        pd_result = pd_df.groupby('cat')['val'].sum()
        ds_result = ds_df.groupby('cat')['val'].sum()
        assert list(pd_result) == list(ds_result)

    def test_single_column_operations(self):
        """Test various operations on single-column DataFrame"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        # Filter
        pd_filtered = pd_df[pd_df['a'] > 2]
        ds_filtered = ds_df[ds_df['a'] > 2]
        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

        # Sort
        pd_sorted = pd_df.sort_values('a', ascending=False)
        ds_sorted = ds_df.sort_values('a', ascending=False)
        assert_datastore_equals_pandas(ds_sorted, pd_sorted)

    def test_single_cell_dataframe(self):
        """Test operations on 1x1 DataFrame"""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore({'a': [42]})

        assert pd_df['a'].iloc[0] == ds_df['a'].iloc[0]
        assert pd_df['a'].sum() == ds_df['a'].sum()
        assert pd_df['a'].mean() == ds_df['a'].mean()


class TestCumulativeOperationsEdgeCases:
    """Test cumulative operations with edge cases"""

    def test_cumsum_with_nulls(self):
        """Test cumsum handling of null values"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})

        pd_result = pd_df['a'].cumsum()
        ds_result = ds_df['a'].cumsum()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cummax_with_nulls(self):
        """Test cummax handling of null values"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 2.0, np.nan]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, 2.0, np.nan]})

        pd_result = pd_df['a'].cummax()
        ds_result = ds_df['a'].cummax()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cummin_with_nulls(self):
        """Test cummin handling of null values"""
        pd_df = pd.DataFrame({'a': [5.0, np.nan, 3.0, 4.0, np.nan]})
        ds_df = DataStore({'a': [5.0, np.nan, 3.0, 4.0, np.nan]})

        pd_result = pd_df['a'].cummin()
        ds_result = ds_df['a'].cummin()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cumsum_single_element(self):
        """Test cumsum on single element"""
        pd_df = pd.DataFrame({'a': [42.0]})
        ds_df = DataStore({'a': [42.0]})

        pd_result = pd_df['a'].cumsum()
        ds_result = ds_df['a'].cumsum()
        assert_series_equal(ds_result, pd_result, check_names=False)


class TestMethodChainsWithTypeTransitions:
    """Test method chains that involve type changes"""

    def test_filter_groupby_agg_filter_chain(self):
        """Test filter -> groupby -> agg -> filter chain"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'C', 'C'],
            'val': [1, 2, 3, 4, 5, 6]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B', 'C', 'C'],
            'val': [1, 2, 3, 4, 5, 6]
        })

        pd_result = (pd_df[pd_df['val'] > 1]
                     .groupby('cat')['val']
                     .sum())
        ds_result = (ds_df[ds_df['val'] > 1]
                     .groupby('cat')['val']
                     .sum())
        # Compare values only since groupby result format may differ
        assert sorted(pd_result.tolist()) == sorted(list(ds_result))

    def test_assign_filter_sort_chain(self):
        """Test assign -> filter -> sort chain"""
        pd_df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })
        ds_df = DataStore({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1]
        })

        pd_result = (pd_df
                     .assign(c=lambda x: x['a'] + x['b'])
                     [lambda x: x['c'] > 5]
                     .sort_values('c'))
        ds_result = (ds_df
                     .assign(c=lambda x: x['a'] + x['b'])
                     [lambda x: x['c'] > 5]
                     .sort_values('c'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_assign_chain(self):
        """Test multiple consecutive assign operations"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = (pd_df
                     .assign(b=lambda x: x['a'] * 2)
                     .assign(c=lambda x: x['b'] + 1)
                     .assign(d=lambda x: x['c'] ** 2))
        ds_result = (ds_df
                     .assign(b=lambda x: x['a'] * 2)
                     .assign(c=lambda x: x['b'] + 1)
                     .assign(d=lambda x: x['c'] ** 2))
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringAccessorEdgeCases:
    """Test string accessor with special characters and edge cases"""

    def test_str_with_empty_strings(self):
        """Test string operations with empty strings"""
        pd_df = pd.DataFrame({'s': ['hello', '', 'world', '', 'test']})
        ds_df = DataStore({'s': ['hello', '', 'world', '', 'test']})

        pd_result = pd_df['s'].str.len()
        ds_result = ds_df['s'].str.len()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_str_upper_with_mixed(self):
        """Test upper with mixed case and special characters"""
        pd_df = pd.DataFrame({'s': ['Hello', 'WORLD', 'tEsT', '123', 'a!@#']})
        ds_df = DataStore({'s': ['Hello', 'WORLD', 'tEsT', '123', 'a!@#']})

        pd_result = pd_df['s'].str.upper()
        ds_result = ds_df['s'].str.upper()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_str_contains_special_chars(self):
        """Test str.contains with special regex characters"""
        pd_df = pd.DataFrame({'s': ['a.b', 'a*b', 'a+b', 'a?b', 'normal']})
        ds_df = DataStore({'s': ['a.b', 'a*b', 'a+b', 'a?b', 'normal']})

        # Using literal matching (regex=False)
        pd_result = pd_df['s'].str.contains('.', regex=False)
        ds_result = ds_df['s'].str.contains('.', regex=False)
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_str_strip_whitespace(self):
        """Test strip operations with various whitespace"""
        pd_df = pd.DataFrame({'s': ['  hello  ', '\tworld\t', '\n test\n', 'no_space', '   ']})
        ds_df = DataStore({'s': ['  hello  ', '\tworld\t', '\n test\n', 'no_space', '   ']})

        pd_result = pd_df['s'].str.strip()
        ds_result = ds_df['s'].str.strip()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_str_replace_empty_pattern(self):
        """Test replace with empty replacement"""
        pd_df = pd.DataFrame({'s': ['hello', 'world', 'test']})
        ds_df = DataStore({'s': ['hello', 'world', 'test']})

        pd_result = pd_df['s'].str.replace('o', '')
        ds_result = ds_df['s'].str.replace('o', '')
        assert_series_equal(ds_result, pd_result, check_names=False)


class TestShiftDiffEdgeCases:
    """Test shift and diff with edge cases"""

    def test_shift_negative_periods(self):
        """Test shift with negative periods (shift backward)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].shift(-1)
        ds_result = ds_df['a'].shift(-1)
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_shift_periods_larger_than_data(self):
        """Test shift with periods larger than data length"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore({'a': [1, 2, 3]})

        pd_result = pd_df['a'].shift(5)
        ds_result = ds_df['a'].shift(5)
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_diff_with_nulls(self):
        """Test diff handling of null values"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, 4.0, np.nan]})
        ds_df = DataStore({'a': [1.0, np.nan, 3.0, 4.0, np.nan]})

        pd_result = pd_df['a'].diff()
        ds_result = ds_df['a'].diff()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_diff_negative_periods(self):
        """Test diff with negative periods"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].diff(-1)
        ds_result = ds_df['a'].diff(-1)
        assert_series_equal(ds_result, pd_result, check_names=False)


class TestFilterWithNullValues:
    """Test filtering behavior with null values"""

    def test_filter_equals_null(self):
        """Test filtering where column equals NULL"""
        pd_df = pd.DataFrame({'a': [1, 2, None, 4, None]})
        ds_df = DataStore({'a': [1, 2, None, 4, None]})

        # In pandas, comparison with None returns False, so no rows match
        pd_result = pd_df[pd_df['a'] == pd_df['a']]  # NaN != NaN
        ds_result = ds_df[ds_df['a'] == ds_df['a']]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_isna(self):
        """Test filtering with isna()"""
        pd_df = pd.DataFrame({'a': [1, 2, None, 4, None]})
        ds_df = DataStore({'a': [1, 2, None, 4, None]})

        pd_result = pd_df[pd_df['a'].isna()]
        ds_result = ds_df[ds_df['a'].isna()]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_notna(self):
        """Test filtering with notna()"""
        pd_df = pd.DataFrame({'a': [1, 2, None, 4, None]})
        ds_df = DataStore({'a': [1, 2, None, 4, None]})

        pd_result = pd_df[pd_df['a'].notna()]
        ds_result = ds_df[ds_df['a'].notna()]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggregationWithAllNulls:
    """Test aggregation on columns with all null values"""

    def test_sum_all_nulls(self):
        """Test sum on column with all null values"""
        pd_df = pd.DataFrame({'a': [None, None, None]}, dtype=float)
        ds_df = DataStore({'a': [None, None, None]})

        pd_result = pd_df['a'].sum()
        ds_result = ds_df['a'].sum()
        assert pd_result == ds_result

    def test_mean_all_nulls(self):
        """Test mean on column with all null values"""
        pd_df = pd.DataFrame({'a': [None, None, None]}, dtype=float)
        ds_df = DataStore({'a': [None, None, None]})

        pd_result = pd_df['a'].mean()
        ds_result = ds_df['a'].mean()
        # Both should be NaN
        assert np.isnan(pd_result) and np.isnan(ds_result)

    def test_count_all_nulls(self):
        """Test count on column with all null values"""
        pd_df = pd.DataFrame({'a': [None, None, None]})
        ds_df = DataStore({'a': [None, None, None]})

        pd_result = pd_df['a'].count()
        ds_result = ds_df['a'].count()
        assert pd_result == ds_result


class TestMergeWithDuplicateKeys:
    """Test merge operations with duplicate key values"""

    def test_merge_one_to_many(self):
        """Test merge with one-to-many relationship"""
        pd_df1 = pd.DataFrame({
            'key': ['a', 'b', 'c'],
            'val1': [1, 2, 3]
        })
        pd_df2 = pd.DataFrame({
            'key': ['a', 'a', 'b', 'b', 'b', 'c'],
            'val2': [10, 20, 30, 40, 50, 60]
        })

        ds_df1 = DataStore({
            'key': ['a', 'b', 'c'],
            'val1': [1, 2, 3]
        })
        ds_df2 = DataStore({
            'key': ['a', 'a', 'b', 'b', 'b', 'c'],
            'val2': [10, 20, 30, 40, 50, 60]
        })

        pd_result = pd_df1.merge(pd_df2, on='key')
        ds_result = ds_df1.merge(ds_df2, on='key')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_many_to_many(self):
        """Test merge with many-to-many relationship (Cartesian product per key)"""
        pd_df1 = pd.DataFrame({
            'key': ['a', 'a', 'b'],
            'val1': [1, 2, 3]
        })
        pd_df2 = pd.DataFrame({
            'key': ['a', 'a', 'b'],
            'val2': [10, 20, 30]
        })

        ds_df1 = DataStore({
            'key': ['a', 'a', 'b'],
            'val1': [1, 2, 3]
        })
        ds_df2 = DataStore({
            'key': ['a', 'a', 'b'],
            'val2': [10, 20, 30]
        })

        pd_result = pd_df1.merge(pd_df2, on='key')
        ds_result = ds_df1.merge(ds_df2, on='key')
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRankOperations:
    """Test rank operations with edge cases"""

    def test_rank_with_ties(self):
        """Test rank with tied values"""
        pd_df = pd.DataFrame({'a': [3, 1, 2, 2, 3]})
        ds_df = DataStore({'a': [3, 1, 2, 2, 3]})

        pd_result = pd_df['a'].rank()
        ds_result = ds_df['a'].rank()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_rank_with_nulls(self):
        """Test rank handling of null values"""
        pd_df = pd.DataFrame({'a': [3.0, np.nan, 2.0, np.nan, 1.0]})
        ds_df = DataStore({'a': [3.0, np.nan, 2.0, np.nan, 1.0]})

        pd_result = pd_df['a'].rank()
        ds_result = ds_df['a'].rank()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_rank_method_min(self):
        """Test rank with method='min'"""
        pd_df = pd.DataFrame({'a': [3, 1, 2, 2, 3]})
        ds_df = DataStore({'a': [3, 1, 2, 2, 3]})

        pd_result = pd_df['a'].rank(method='min')
        ds_result = ds_df['a'].rank(method='min')
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_rank_method_max(self):
        """Test rank with method='max'"""
        pd_df = pd.DataFrame({'a': [3, 1, 2, 2, 3]})
        ds_df = DataStore({'a': [3, 1, 2, 2, 3]})

        pd_result = pd_df['a'].rank(method='max')
        ds_result = ds_df['a'].rank(method='max')
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_rank_ascending_false(self):
        """Test rank with ascending=False"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore({'a': [1, 2, 3, 4, 5]})

        pd_result = pd_df['a'].rank(ascending=False)
        ds_result = ds_df['a'].rank(ascending=False)
        assert_series_equal(ds_result, pd_result, check_names=False)
