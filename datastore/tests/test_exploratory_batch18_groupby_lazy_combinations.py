"""
Exploratory Discovery Batch 18 - GroupBy Advanced Operations and LazyOp Combinations

Focus areas:
1. GroupBy transform/filter/apply edge cases
2. Rolling/Expanding with grouped data
3. Multiple aggregate functions on same column
4. LazyWhere/LazyMask chaining
5. Complex operation chains with groupby
6. Time series methods with DataStore
"""

import pytest
from tests.xfail_markers import (
    chdb_mask_dtype_nullable,
    pandas_version_no_include_groups,
    pandas_version_first_last_offset_warning,
)
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_dataframe, get_series


class TestGroupByTransform:
    """Test groupby transform edge cases."""

    def test_transform_with_lambda(self):
        """Test groupby transform with lambda function."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].transform(lambda x: x - x.mean())
        ds_result = ds_df.groupby('category')['value'].transform(lambda x: x - x.mean())

        # Series name may differ
        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_transform_with_builtin_func(self):
        """Test groupby transform with builtin function name."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].transform('mean')
        ds_result = ds_df.groupby('category')['value'].transform('mean')

        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_transform_normalize(self):
        """Test groupby transform for normalization."""
        pd_df = pd.DataFrame({'group': ['X', 'X', 'Y', 'Y'], 'score': [10.0, 20.0, 30.0, 40.0]})
        ds_df = DataStore(pd_df.copy())

        def normalize(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0

        pd_result = pd_df.groupby('group')['score'].transform(normalize)
        ds_result = ds_df.groupby('group')['score'].transform(normalize)

        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_transform_cumulative(self):
        """Test groupby transform with cumsum."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].transform('cumsum')
        ds_result = ds_df.groupby('category')['value'].transform('cumsum')

        ds_series = get_series(ds_result)
        assert_series_equal(
            ds_series.reset_index(drop=True), pd_result.reset_index(drop=True))


class TestGroupByFilter:
    """Test groupby filter edge cases."""

    def test_filter_by_size(self):
        """Test groupby filter by group size."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').filter(lambda x: len(x) >= 3)
        ds_result = ds_df.groupby('category').filter(lambda x: len(x) >= 3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_sum(self):
        """Test groupby filter by group sum."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 10, 20]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').filter(lambda x: x['value'].sum() > 5)
        ds_result = ds_df.groupby('category').filter(lambda x: x['value'].sum() > 5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_all_filtered(self):
        """Test groupby filter when all groups are filtered out."""
        pd_df = pd.DataFrame({'category': ['A', 'B'], 'value': [1, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').filter(lambda x: len(x) > 10)
        ds_result = ds_df.groupby('category').filter(lambda x: len(x) > 10)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByApply:
    """Test groupby apply edge cases.

    Note: The include_groups parameter was added in pandas 2.2.0.
    Tests that use include_groups are skipped on older pandas versions.
    """

    @pandas_version_no_include_groups
    def test_apply_returns_scalar(self):
        """Test groupby apply that returns scalar per group."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        # include_groups parameter was added in pandas 2.2.0
        pd_result = pd_df.groupby('category').apply(lambda x: x['value'].sum(), include_groups=False)
        ds_result = ds_df.groupby('category').apply(lambda x: x['value'].sum())

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    @pandas_version_no_include_groups
    def test_apply_returns_series(self):
        """Test groupby apply that returns Series per group."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').apply(lambda x: x['value'].head(1), include_groups=False)
        ds_result = ds_df.groupby('category').apply(lambda x: x['value'].head(1))

        # Just check it runs without error and has correct shape
        ds_df_result = get_series(ds_result)
        assert len(ds_df_result) == len(pd_result)


class TestGroupByMultipleAggs:
    """Test multiple aggregations on same column."""

    def test_agg_multiple_funcs_list(self):
        """Test agg with list of functions on single column."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].agg(['sum', 'mean', 'count'])
        ds_result = ds_df.groupby('category')['value'].agg(['sum', 'mean', 'count'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_dict_multiple_cols(self):
        """Test agg with dict specifying different funcs per column."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value1': [1, 2, 3, 4], 'value2': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').agg({'value1': 'sum', 'value2': 'mean'})
        ds_result = ds_df.groupby('category').agg({'value1': 'sum', 'value2': 'mean'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_agg_named_tuple_syntax(self):
        """Test agg with named aggregation tuple syntax."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').agg(total=('value', 'sum'), average=('value', 'mean'))
        ds_result = ds_df.groupby('category').agg(total=('value', 'sum'), average=('value', 'mean'))

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByWithRolling:
    """Test groupby combined with rolling operations."""

    def test_groupby_rolling_mean(self):
        """Test groupby with rolling mean."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'A', 'B', 'B', 'B'], 'value': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category')['value'].rolling(2).mean().reset_index(level=0, drop=True)
        ds_result = ds_df.groupby('category')['value'].rolling(2).mean()

        # Rolling with groupby returns pandas Series
        ds_series = ds_result if isinstance(ds_result, pd.Series) else ds_result
        assert len(ds_series) == len(pd_result)


class TestSeriesWhere:
    """Test Series where/mask operations."""

    def test_series_where_simple(self):
        """Test simple Series where operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].where(pd_df['a'] > 2)
        ds_result = ds_df['a'].where(ds_df['a'] > 2)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_series_where_with_other(self):
        """Test Series where with replacement value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].where(pd_df['a'] > 2, -1)
        ds_result = ds_df['a'].where(ds_df['a'] > 2, -1)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    @chdb_mask_dtype_nullable
    def test_series_mask_simple(self):
        """Test simple Series mask operation.

        Note: DataStore returns nullable Int64 (better semantics, preserves integer type),
        while pandas returns float64 (because numpy int64 can't represent NaN).
        This is a design difference, not a bug.
        """
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].mask(pd_df['a'] > 2)
        ds_result = ds_df['a'].mask(ds_df['a'] > 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_mask_with_other(self):
        """Test Series mask with replacement value."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].mask(pd_df['a'] > 2, 0)
        ds_result = ds_df['a'].mask(ds_df['a'] > 2, 0)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)


class TestDataFrameComparison:
    """Test DataFrame comparison operators - discovered bug."""

    #
    def test_df_gt_scalar(self):
        """Test DataFrame > scalar comparison."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df > 2
        ds_result = ds_df > 2

        assert_datastore_equals_pandas(ds_result, pd_result)

    #
    def test_df_where(self):
        """Test DataFrame where operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.where(pd_df > 2)
        ds_result = ds_df.where(ds_df > 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    #
    def test_df_mask(self):
        """Test DataFrame mask operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.mask(pd_df > 2)
        ds_result = ds_df.mask(ds_df > 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArgsortMethods:
    """Test argsort and related methods."""

    def test_argsort_ascending(self):
        """Test argsort ascending order."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].argsort()
        ds_result = ds_df['a'].argsort()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_argmin(self):
        """Test argmin method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].argmin()
        ds_result = ds_df['a'].argmin()

        assert ds_result == pd_result

    def test_argmax(self):
        """Test argmax method."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].argmax()
        ds_result = ds_df['a'].argmax()

        assert ds_result == pd_result


class TestColumnExprAgg:
    """Test ColumnExpr agg method variations."""

    def test_agg_with_single_func(self):
        """Test agg with single function string."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].agg('sum')
        ds_result = ds_df['a'].agg('sum')

        assert ds_result == pd_result

    def test_agg_with_list(self):
        """Test agg with list of functions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].agg(['sum', 'mean', 'min', 'max'])
        ds_result = ds_df['a'].agg(['sum', 'mean', 'min', 'max'])

        # Check values match
        pd_dict = pd_result.to_dict()
        ds_exec = get_series(ds_result)
        ds_dict = ds_exec.to_dict()
        for key in pd_dict:
            assert abs(pd_dict[key] - ds_dict[key]) < 1e-6


class TestExpandingMethods:
    """Test expanding window operations."""

    def test_expanding_sum(self):
        """Test expanding sum."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].expanding().sum()
        ds_result = ds_df['a'].expanding().sum()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_expanding_mean(self):
        """Test expanding mean."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].expanding().mean()
        ds_result = ds_df['a'].expanding().mean()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_expanding_min_periods(self):
        """Test expanding with min_periods."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].expanding(min_periods=3).mean()
        ds_result = ds_df['a'].expanding(min_periods=3).mean()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)


class TestEwmMethods:
    """Test exponentially weighted window operations."""

    def test_ewm_mean_span(self):
        """Test ewm mean with span parameter."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].ewm(span=3).mean()
        ds_result = ds_df['a'].ewm(span=3).mean()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_ewm_mean_alpha(self):
        """Test ewm mean with alpha parameter."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].ewm(alpha=0.5).mean()
        ds_result = ds_df['a'].ewm(alpha=0.5).mean()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)


class TestRollingEdgeCases:
    """Test rolling window edge cases."""

    def test_rolling_min_periods(self):
        """Test rolling with min_periods."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].rolling(3, min_periods=1).mean()
        ds_result = ds_df['a'].rolling(3, min_periods=1).mean()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_rolling_center(self):
        """Test rolling with center=True."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].rolling(3, center=True).mean()
        ds_result = ds_df['a'].rolling(3, center=True).mean()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_rolling_std(self):
        """Test rolling std."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].rolling(3).std()
        ds_result = ds_df['a'].rolling(3).std()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)


class TestCombineOperations:
    """Test combine and combine_first operations."""

    def test_combine_first_basic(self):
        """Test combine_first basic usage."""
        pd_df1 = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan, 2, np.nan]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]})
        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_with_pandas_df(self):
        """Test combine_first with pandas DataFrame."""
        pd_df1 = pd.DataFrame({'a': [1, np.nan, 3]})
        pd_df2 = pd.DataFrame({'a': [10, 20, 30]})
        ds_df1 = DataStore(pd_df1.copy())

        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(pd_df2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAlignOperations:
    """Test align operations."""

    def test_align_inner(self):
        """Test align with inner join."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'b': [4, 5, 6]}, index=[1, 2, 3])
        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_left, pd_right = pd_df1.align(pd_df2, join='inner')
        ds_left, ds_right = ds_df1.align(ds_df2, join='inner')

        assert_datastore_equals_pandas(ds_left, pd_left)
        assert_datastore_equals_pandas(ds_right, pd_right)

    def test_align_outer(self):
        """Test align with outer join."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]}, index=[0, 1, 2])
        pd_df2 = pd.DataFrame({'b': [4, 5, 6]}, index=[1, 2, 3])
        ds_df1 = DataStore(pd_df1.copy())
        ds_df2 = DataStore(pd_df2.copy())

        pd_left, pd_right = pd_df1.align(pd_df2, join='outer')
        ds_left, ds_right = ds_df1.align(ds_df2, join='outer')

        assert_datastore_equals_pandas(ds_left, pd_left)
        assert_datastore_equals_pandas(ds_right, pd_right)


class TestComplexChains:
    """Test complex operation chains."""

    def test_filter_groupby_agg(self):
        """Test filter -> groupby -> agg chain."""
        pd_df = pd.DataFrame(
            {'category': ['A', 'A', 'B', 'B', 'A'], 'value': [1, 2, 3, 4, 5], 'flag': [True, False, True, True, True]}
        )
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[pd_df['flag']].groupby('category')['value'].sum()
        ds_result = ds_df[ds_df['flag']].groupby('category')['value'].sum()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_assign_filter_sort(self):
        """Test assign -> filter -> sort chain."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.assign(c=pd_df['a'] + pd_df['b']).query('c > 20').sort_values('c')
        ds_result = ds_df.assign(c=ds_df['a'] + ds_df['b']).query('c > 20').sort_values('c')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_head_sort(self):
        """Test groupby -> head -> sort chain."""
        pd_df = pd.DataFrame({'category': ['A', 'A', 'A', 'B', 'B', 'B'], 'value': [3, 1, 2, 6, 4, 5]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.groupby('category').head(2).sort_values(['category', 'value'])
        ds_result = ds_df.groupby('category').head(2).sort_values(['category', 'value'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters(self):
        """Test multiple filter operations."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1], 'c': ['x', 'y', 'x', 'y', 'x']})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df[(pd_df['a'] > 1) & (pd_df['b'] < 5) & (pd_df['c'] == 'x')]
        ds_result = ds_df[(ds_df['a'] > 1) & (ds_df['b'] < 5) & (ds_df['c'] == 'x')]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestModeUnique:
    """Test mode and unique operations."""

    def test_mode_single_value(self):
        """Test mode with single most frequent value."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 2]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].mode()
        ds_result = ds_df['a'].mode()

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_mode_multiple_values(self):
        """Test mode with multiple equally frequent values."""
        pd_df = pd.DataFrame({'a': [1, 1, 2, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].mode()
        ds_result = ds_df['a'].mode()

        # Both should return all modes
        ds_series = get_series(ds_result)
        assert len(ds_series) == len(pd_result)

    def test_unique(self):
        """Test unique values."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].unique()
        ds_result = ds_df['a'].unique()

        # Compare as sets since order may differ
        pd_set = set(pd_result)
        ds_exec = get_series(ds_result)
        ds_set = set(ds_exec)
        assert pd_set == ds_set

    def test_nunique(self):
        """Test nunique count."""
        pd_df = pd.DataFrame({'a': [1, 2, 2, 3, 3, 3, np.nan]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].nunique()
        ds_result = ds_df['a'].nunique()

        assert ds_result == pd_result


class TestMapApply:
    """Test map and apply operations."""

    def test_map_with_dict(self):
        """Test map with dictionary."""
        pd_df = pd.DataFrame({'a': ['cat', 'dog', 'bird']})
        ds_df = DataStore(pd_df.copy())

        mapping = {'cat': 1, 'dog': 2, 'bird': 3}
        pd_result = pd_df['a'].map(mapping)
        ds_result = ds_df['a'].map(mapping)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_map_with_function(self):
        """Test map with function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].map(lambda x: x * 2)
        ds_result = ds_df['a'].map(lambda x: x * 2)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_apply_with_function(self):
        """Test apply with function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df['a'].apply(lambda x: x**2)
        ds_result = ds_df['a'].apply(lambda x: x**2)

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)


class TestFirstLastOffset:
    """Test first/last with offset parameter.

    Note: DataFrame.first()/last() with offset was deprecated in pandas 2.1.0.
    In older pandas versions, no FutureWarning is emitted.
    """

    @pandas_version_first_last_offset_warning
    def test_first_with_offset(self):
        """Test first with DateOffset."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        ds_df = DataStore(pd_df.copy())

        with pytest.warns(FutureWarning):
            pd_result = pd_df.first('3D')
        with pytest.warns(FutureWarning):
            ds_result = ds_df.first('3D')

        assert_datastore_equals_pandas(ds_result, pd_result)

    @pandas_version_first_last_offset_warning
    def test_last_with_offset(self):
        """Test last with DateOffset."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=pd.date_range('2023-01-01', periods=5, freq='D'))
        ds_df = DataStore(pd_df.copy())

        with pytest.warns(FutureWarning):
            pd_result = pd_df.last('3D')
        with pytest.warns(FutureWarning):
            ds_result = ds_df.last('3D')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestBetweenTime:
    """Test between_time and at_time methods."""

    def test_between_time(self):
        """Test between_time filter."""
        dates = pd.date_range('2023-01-01', periods=24, freq='h')
        pd_df = pd.DataFrame({'a': range(24)}, index=dates)
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.between_time('09:00', '17:00')
        ds_result = ds_df.between_time('09:00', '17:00')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_at_time(self):
        """Test at_time filter."""
        dates = pd.date_range('2023-01-01', periods=48, freq='h')
        pd_df = pd.DataFrame({'a': range(48)}, index=dates)
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.at_time('12:00')
        ds_result = ds_df.at_time('12:00')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTruncate:
    """Test truncate method."""

    def test_truncate_before_after(self):
        """Test truncate with before and after."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.truncate(before=1, after=3)
        ds_result = ds_df.truncate(before=1, after=3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_truncate_datetime_index(self):
        """Test truncate with datetime index."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]}, index=dates)
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.truncate(before='2023-01-02', after='2023-01-04')
        ds_result = ds_df.truncate(before='2023-01-02', after='2023-01-04')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGetMethod:
    """Test get method variations."""

    def test_get_existing_column(self):
        """Test get with existing column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.get('a')
        ds_result = ds_df.get('a')

        ds_series = get_series(ds_result)
        assert_series_equal(ds_series, pd_result)

    def test_get_non_existing_column(self):
        """Test get with non-existing column returns default."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df.copy())

        pd_result = pd_df.get('nonexistent', default='default_value')
        ds_result = ds_df.get('nonexistent', default='default_value')

        assert pd_result == ds_result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
