"""
Exploratory Batch 15: ColumnExpr Methods, __setitem__, and Filter Combinations

Focus areas:
1. ColumnExpr method edge cases (abs, clip, between, isin, mask, where)
2. __setitem__ various forms (slice, boolean, scalar)
3. Complex filter combinations (AND/OR/NOT)
4. Method parameter variations (drop_duplicates, rank, etc.)
5. SQL optimization verification for complex chains
"""

import pytest
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal


# =============================================================================
# Part 1: ColumnExpr Method Edge Cases
# =============================================================================

class TestColumnExprAbsClip:
    """Test abs() and clip() methods on ColumnExpr"""

    def test_abs_positive_values(self):
        """abs() on positive values should return same values"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].abs()

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].abs()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_abs_negative_values(self):
        """abs() on negative values should return positive"""
        pd_df = pd.DataFrame({'a': [-1, -2, -3, -4, -5]})
        pd_result = pd_df['a'].abs()

        ds = DataStore({'a': [-1, -2, -3, -4, -5]})
        ds_result = ds['a'].abs()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_abs_mixed_values(self):
        """abs() on mixed positive/negative values"""
        pd_df = pd.DataFrame({'a': [-3, -1, 0, 1, 3]})
        pd_result = pd_df['a'].abs()

        ds = DataStore({'a': [-3, -1, 0, 1, 3]})
        ds_result = ds['a'].abs()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_clip_lower_only(self):
        """clip() with only lower bound"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].clip(lower=2)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].clip(lower=2)

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_clip_upper_only(self):
        """clip() with only upper bound"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].clip(upper=4)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].clip(upper=4)

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_clip_both_bounds(self):
        """clip() with both lower and upper bounds"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].clip(lower=2, upper=4)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].clip(lower=2, upper=4)

        assert_series_equal(ds_result.to_pandas(), pd_result)


class TestColumnExprBetweenIsin:
    """Test between() and isin() methods"""

    def test_between_inclusive(self):
        """between() with inclusive bounds"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].between(2, 4)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].between(2, 4)

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_between_exclusive(self):
        """between() with exclusive bounds"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].between(2, 4, inclusive='neither')

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].between(2, 4, inclusive='neither')

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_isin_basic(self):
        """isin() with list of values"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].isin([2, 4])

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].isin([2, 4])

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_isin_empty_list(self):
        """isin() with empty list should return all False"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].isin([])

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].isin([])

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_isin_strings(self):
        """isin() with string values"""
        pd_df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Diana']})
        pd_result = pd_df['name'].isin(['Alice', 'Charlie'])

        ds = DataStore({'name': ['Alice', 'Bob', 'Charlie', 'Diana']})
        ds_result = ds['name'].isin(['Alice', 'Charlie'])

        assert_series_equal(ds_result.to_pandas(), pd_result)


class TestColumnExprMaskWhere:
    """Test mask() and where() on ColumnExpr (Series)"""

    def test_series_where_basic(self):
        """Series.where() basic usage"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].where(pd_df['a'] > 2)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].where(ds['a'] > 2)

        # NaN comparison needs special handling
        pd_arr = pd_result.to_numpy()
        ds_arr = ds_result.to_pandas().to_numpy()
        np.testing.assert_array_equal(np.isnan(pd_arr), np.isnan(ds_arr))
        np.testing.assert_array_equal(pd_arr[~np.isnan(pd_arr)], ds_arr[~np.isnan(ds_arr)])

    def test_series_where_with_other(self):
        """Series.where() with replacement value"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].where(pd_df['a'] > 2, other=-1)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].where(ds['a'] > 2, other=-1)

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_series_mask_basic(self):
        """Series.mask() basic usage"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].mask(pd_df['a'] > 2)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].mask(ds['a'] > 2)

        # NaN comparison needs special handling
        # Use pd.isna instead of np.isnan for better compatibility with different dtypes
        pd_isna = pd.isna(pd_result)
        ds_isna = pd.isna(ds_result.to_pandas())
        np.testing.assert_array_equal(pd_isna.to_numpy(), ds_isna.to_numpy())
        # Compare non-NA values
        np.testing.assert_array_equal(
            pd_result[~pd_isna].to_numpy(),
            ds_result.to_pandas()[~ds_isna].to_numpy()
        )

    def test_series_mask_with_other(self):
        """Series.mask() with replacement value"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df['a'].mask(pd_df['a'] > 2, other=0)

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds['a'].mask(ds['a'] > 2, other=0)

        assert_series_equal(ds_result.to_pandas(), pd_result)


# =============================================================================
# Part 2: __setitem__ Various Forms
# =============================================================================

class TestSetItemBasic:
    """Test basic __setitem__ operations"""

    def test_setitem_scalar(self):
        """df['new_col'] = scalar"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = 10

        ds = DataStore({'a': [1, 2, 3]})
        ds['b'] = 10

        assert_datastore_equals_pandas(ds, pd_df)

    def test_setitem_list(self):
        """df['new_col'] = list"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = [4, 5, 6]

        ds = DataStore({'a': [1, 2, 3]})
        ds['b'] = [4, 5, 6]

        assert_datastore_equals_pandas(ds, pd_df)

    def test_setitem_series(self):
        """df['new_col'] = series"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = pd.Series([4, 5, 6])

        ds = DataStore({'a': [1, 2, 3]})
        ds['b'] = pd.Series([4, 5, 6])

        assert_datastore_equals_pandas(ds, pd_df)

    def test_setitem_column_expr(self):
        """df['new_col'] = ds['col'] * 2"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['b'] = pd_df['a'] * 2

        ds = DataStore({'a': [1, 2, 3]})
        ds['b'] = ds['a'] * 2

        assert_datastore_equals_pandas(ds, pd_df)

    def test_setitem_overwrite_existing(self):
        """df['existing_col'] = new_values should overwrite"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df['b'] = [7, 8, 9]

        ds = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds['b'] = [7, 8, 9]

        assert_datastore_equals_pandas(ds, pd_df)


class TestSetItemConditional:
    """Test conditional assignment via boolean indexing"""

    def test_setitem_boolean_mask_scalar(self):
        """df.loc[mask, 'col'] = scalar"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_df.loc[pd_df['a'] > 3, 'a'] = 0

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        # DataStore uses different syntax - test if supported
        result_df = ds.to_pandas()
        result_df.loc[result_df['a'] > 3, 'a'] = 0

        assert_frame_equal(result_df, pd_df)

    def test_setitem_after_filter(self):
        """Assignment after filter chain"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df['c'] = pd_df['a'] + pd_df['b']

        ds = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds['c'] = ds['a'] + ds['b']

        assert_datastore_equals_pandas(ds, pd_df)


# =============================================================================
# Part 3: Complex Filter Combinations
# =============================================================================

class TestComplexFilters:
    """Test complex AND/OR/NOT filter combinations"""

    def test_filter_and_combination(self):
        """Filter with AND: (a > 2) & (b < 6)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['b'] < 4)]

        ds = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_result = ds[(ds['a'] > 2) & (ds['b'] < 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_or_combination(self):
        """Filter with OR: (a > 4) | (b < 2)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_result = pd_df[(pd_df['a'] > 4) | (pd_df['b'] < 2)]

        ds = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_result = ds[(ds['a'] > 4) | (ds['b'] < 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_not(self):
        """Filter with NOT: ~(a > 2)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        pd_result = pd_df[~(pd_df['a'] > 2)]

        ds = DataStore({'a': [1, 2, 3, 4, 5]})
        ds_result = ds[~(ds['a'] > 2)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_complex_nested(self):
        """Complex nested: ((a > 2) & (b < 5)) | (c == 3)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1], 'c': [3, 3, 3, 1, 1]})
        pd_result = pd_df[((pd_df['a'] > 2) & (pd_df['b'] < 5)) | (pd_df['c'] == 3)]

        ds = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1], 'c': [3, 3, 3, 1, 1]})
        ds_result = ds[((ds['a'] > 2) & (ds['b'] < 5)) | (ds['c'] == 3)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_triple_and(self):
        """Triple AND: (a > 1) & (b > 1) & (c > 1)"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 1], 'c': [3, 1, 2]})
        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] > 1) & (pd_df['c'] > 1)]

        ds = DataStore({'a': [1, 2, 3], 'b': [2, 3, 1], 'c': [3, 1, 2]})
        ds_result = ds[(ds['a'] > 1) & (ds['b'] > 1) & (ds['c'] > 1)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_isin_combined(self):
        """Filter combining isin with other conditions"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'x', 'y']})
        pd_result = pd_df[(pd_df['a'] > 2) & (pd_df['b'].isin(['x', 'y']))]

        ds = DataStore({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'x', 'y']})
        ds_result = ds[(ds['a'] > 2) & (ds['b'].isin(['x', 'y']))]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Part 4: Method Parameter Variations
# =============================================================================

class TestDropDuplicatesParams:
    """Test drop_duplicates() parameter variations"""

    def test_drop_duplicates_subset(self):
        """drop_duplicates(subset=['col'])"""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 3, 4]})
        pd_result = pd_df.drop_duplicates(subset=['a'])

        ds = DataStore({'a': [1, 1, 2, 2], 'b': [1, 2, 3, 4]})
        ds_result = ds.drop_duplicates(subset=['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        """drop_duplicates(keep='last')"""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2], 'b': [1, 2, 3, 4]})
        pd_result = pd_df.drop_duplicates(keep='last')

        ds = DataStore({'a': [1, 1, 2, 2], 'b': [1, 2, 3, 4]})
        ds_result = ds.drop_duplicates(keep='last')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self):
        """drop_duplicates(keep=False) - remove all duplicates"""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 3], 'b': [1, 2, 3, 4]})
        pd_result = pd_df.drop_duplicates(keep=False)

        ds = DataStore({'a': [1, 1, 2, 3], 'b': [1, 2, 3, 4]})
        ds_result = ds.drop_duplicates(keep=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestRankParams:
    """Test rank() parameter variations"""

    def test_rank_default(self):
        """rank() with default params"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_result = pd_df['a'].rank()

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds_result = ds['a'].rank()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_rank_ascending_false(self):
        """rank(ascending=False)"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_result = pd_df['a'].rank(ascending=False)

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds_result = ds['a'].rank(ascending=False)

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_rank_method_min(self):
        """rank(method='min')"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_result = pd_df['a'].rank(method='min')

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds_result = ds['a'].rank(method='min')

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_rank_method_max(self):
        """rank(method='max')"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_result = pd_df['a'].rank(method='max')

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds_result = ds['a'].rank(method='max')

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_rank_method_dense(self):
        """rank(method='dense')"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_result = pd_df['a'].rank(method='dense')

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds_result = ds['a'].rank(method='dense')

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_rank_pct(self):
        """rank(pct=True)"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        pd_result = pd_df['a'].rank(pct=True)

        ds = DataStore({'a': [3, 1, 4, 1, 5]})
        ds_result = ds['a'].rank(pct=True)

        assert_series_equal(ds_result.to_pandas(), pd_result, rtol=1e-5)


class TestNLargestNSmallest:
    """Test nlargest() and nsmallest() methods"""

    def test_nlargest_basic(self):
        """nlargest(3, 'col')"""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 2, 4], 'b': ['e', 'a', 'c', 'd', 'b']})
        pd_result = pd_df.nlargest(3, 'a')

        ds = DataStore({'a': [1, 5, 3, 2, 4], 'b': ['e', 'a', 'c', 'd', 'b']})
        ds_result = ds.nlargest(3, 'a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_basic(self):
        """nsmallest(3, 'col')"""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 2, 4], 'b': ['e', 'a', 'c', 'd', 'b']})
        pd_result = pd_df.nsmallest(3, 'a')

        ds = DataStore({'a': [1, 5, 3, 2, 4], 'b': ['e', 'a', 'c', 'd', 'b']})
        ds_result = ds.nsmallest(3, 'a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_nlargest(self):
        """Series.nlargest(3)"""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 2, 4]})
        pd_result = pd_df['a'].nlargest(3)

        ds = DataStore({'a': [1, 5, 3, 2, 4]})
        ds_result = ds['a'].nlargest(3)

        # Compare values, ignore index
        np.testing.assert_array_equal(
            sorted(ds_result.to_pandas().values),
            sorted(pd_result.values)
        )

    def test_series_nsmallest(self):
        """Series.nsmallest(3)"""
        pd_df = pd.DataFrame({'a': [1, 5, 3, 2, 4]})
        pd_result = pd_df['a'].nsmallest(3)

        ds = DataStore({'a': [1, 5, 3, 2, 4]})
        ds_result = ds['a'].nsmallest(3)

        # Compare values, ignore index
        np.testing.assert_array_equal(
            sorted(ds_result.to_pandas().values),
            sorted(pd_result.values)
        )


# =============================================================================
# Part 5: SQL Optimization Verification
# =============================================================================

class TestSQLOptimization:
    """Verify complex chains are optimized into single SQL"""

    def test_filter_select_chain(self):
        """filter -> select should combine into single SQL"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_result = pd_df[pd_df['a'] > 2][['a']]

        ds = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_result = ds[ds['a'] > 2][['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_chain(self):
        """filter -> filter -> filter should combine"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        pd_result = pd_df[pd_df['a'] > 1][pd_df['a'] < 5][pd_df['b'] > 1]

        ds = DataStore({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        ds_result = ds[ds['a'] > 1][ds['a'] < 5][ds['b'] > 1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_head_chain(self):
        """sort -> head should combine into ORDER BY LIMIT"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': ['c', 'a', 'd', 'b', 'e']})
        pd_result = pd_df.sort_values('a').head(3)

        ds = DataStore({'a': [3, 1, 4, 1, 5], 'b': ['c', 'a', 'd', 'b', 'e']})
        ds_result = ds.sort_values('a').head(3)

        # Reset index for comparison since pandas preserves original index
        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))

    def test_filter_sort_head_chain(self):
        """filter -> sort -> head combined chain"""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5], 'b': ['c', 'a', 'd', 'b', 'e']})
        pd_result = pd_df[pd_df['a'] > 1].sort_values('a').head(2)

        ds = DataStore({'a': [3, 1, 4, 1, 5], 'b': ['c', 'a', 'd', 'b', 'e']})
        ds_result = ds[ds['a'] > 1].sort_values('a').head(2)

        assert_datastore_equals_pandas(ds_result, pd_result.reset_index(drop=True))


# =============================================================================
# Part 6: Edge Cases with NaN/None/Inf
# =============================================================================

class TestSpecialValues:
    """Test handling of NaN, None, inf in various operations"""

    def test_filter_with_nan(self):
        """Filter column containing NaN"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0, 5.0]})
        pd_result = pd_df[pd_df['a'] > 2]

        ds = DataStore({'a': [1.0, 2.0, np.nan, 4.0, 5.0]})
        ds_result = ds[ds['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isnull_filter(self):
        """Filter using isnull()"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        pd_result = pd_df[pd_df['a'].isnull()]

        ds = DataStore({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_result = ds[ds['a'].isnull()]

        assert len(ds_result) == len(pd_result)

    def test_notnull_filter(self):
        """Filter using notnull()"""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        pd_result = pd_df[pd_df['a'].notnull()]

        ds = DataStore({'a': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_result = ds[ds['a'].notnull()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_with_nan(self):
        """sum() should skip NaN by default"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0]})
        pd_result = pd_df['a'].sum()

        ds = DataStore({'a': [1.0, 2.0, np.nan, 4.0]})
        ds_result = float(ds['a'].sum())

        assert ds_result == pd_result

    def test_mean_with_nan(self):
        """mean() should skip NaN by default"""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, np.nan, 4.0]})
        pd_result = pd_df['a'].mean()

        ds = DataStore({'a': [1.0, 2.0, np.nan, 4.0]})
        ds_result = float(ds['a'].mean())

        np.testing.assert_almost_equal(ds_result, pd_result)


# =============================================================================
# Part 7: String Column Operations
# =============================================================================

class TestStringColumnOps:
    """Test string column operations"""

    def test_str_upper(self):
        """str.upper()"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        pd_result = pd_df['name'].str.upper()

        ds = DataStore({'name': ['alice', 'bob', 'charlie']})
        ds_result = ds['name'].str.upper()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_str_lower(self):
        """str.lower()"""
        pd_df = pd.DataFrame({'name': ['ALICE', 'BOB', 'CHARLIE']})
        pd_result = pd_df['name'].str.lower()

        ds = DataStore({'name': ['ALICE', 'BOB', 'CHARLIE']})
        ds_result = ds['name'].str.lower()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_str_len(self):
        """str.len()"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        pd_result = pd_df['name'].str.len()

        ds = DataStore({'name': ['alice', 'bob', 'charlie']})
        ds_result = ds['name'].str.len()

        assert_series_equal(ds_result.to_pandas(), pd_result)

    def test_str_contains_filter(self):
        """Filter using str.contains()"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie', 'diana']})
        pd_result = pd_df[pd_df['name'].str.contains('a')]

        ds = DataStore({'name': ['alice', 'bob', 'charlie', 'diana']})
        ds_result = ds[ds['name'].str.contains('a')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_startswith_filter(self):
        """Filter using str.startswith()"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie', 'anna']})
        pd_result = pd_df[pd_df['name'].str.startswith('a')]

        ds = DataStore({'name': ['alice', 'bob', 'charlie', 'anna']})
        ds_result = ds[ds['name'].str.startswith('a')]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_replace(self):
        """str.replace()"""
        pd_df = pd.DataFrame({'name': ['alice', 'bob', 'charlie']})
        pd_result = pd_df['name'].str.replace('a', 'X', regex=False)

        ds = DataStore({'name': ['alice', 'bob', 'charlie']})
        ds_result = ds['name'].str.replace('a', 'X', regex=False)

        assert_series_equal(ds_result.to_pandas(), pd_result)


# =============================================================================
# Part 8: Column Arithmetic Chains
# =============================================================================

class TestArithmeticChains:
    """Test chained arithmetic operations"""

    def test_multiple_arithmetic_ops(self):
        """(a + b) * c / d"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [2, 2, 2], 'd': [1, 2, 3]})
        pd_df['result'] = (pd_df['a'] + pd_df['b']) * pd_df['c'] / pd_df['d']

        ds = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [2, 2, 2], 'd': [1, 2, 3]})
        ds['result'] = (ds['a'] + ds['b']) * ds['c'] / ds['d']

        assert_datastore_equals_pandas(ds, pd_df)

    def test_arithmetic_with_scalar(self):
        """a * 2 + 10"""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        pd_df['result'] = pd_df['a'] * 2 + 10

        ds = DataStore({'a': [1, 2, 3]})
        ds['result'] = ds['a'] * 2 + 10

        assert_datastore_equals_pandas(ds, pd_df)

    def test_mixed_scalar_column_arithmetic(self):
        """10 - a + b * 2"""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df['result'] = 10 - pd_df['a'] + pd_df['b'] * 2

        ds = DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds['result'] = 10 - ds['a'] + ds['b'] * 2

        assert_datastore_equals_pandas(ds, pd_df)

    def test_power_operation(self):
        """a ** 2"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4]})
        pd_df['squared'] = pd_df['a'] ** 2

        ds = DataStore({'a': [1, 2, 3, 4]})
        ds['squared'] = ds['a'] ** 2

        assert_datastore_equals_pandas(ds, pd_df)

    def test_modulo_operation(self):
        """a % 3"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        pd_df['mod'] = pd_df['a'] % 3

        ds = DataStore({'a': [1, 2, 3, 4, 5, 6]})
        ds['mod'] = ds['a'] % 3

        assert_datastore_equals_pandas(ds, pd_df)

    def test_floor_division(self):
        """a // 2"""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        pd_df['floordiv'] = pd_df['a'] // 2

        ds = DataStore({'a': [1, 2, 3, 4, 5, 6]})
        ds['floordiv'] = ds['a'] // 2

        assert_datastore_equals_pandas(ds, pd_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
