"""
Test GroupBy aggregation function SQL mapping completeness and correctness.

Scenario 7 (P1): Verify all aggregation functions in groupby context produce
results consistent with pandas in both named_agg and dict_agg modes.

Covers:
- All agg functions: count, sum, mean, std, var, min, max, first, last, nunique
- named_agg mode: agg(alias=('col', 'func'))
- dict_agg mode: agg({'col': 'func'}) and agg({'col': ['func1', 'func2']})
- first/last row-order semantics (argMin/argMax + rowNumberInAllBlocks)
- nunique -> uniqExact mapping
- var -> varSamp mapping (sample variance, ddof=1)
- Multi-column groupby + multi-agg combinations
- Computed columns as groupby key or agg target
"""

import unittest

import numpy as np
import pandas as pd

import datastore as ds
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_frame_equal,
    assert_series_equal,
    get_dataframe,
    get_series,
)
from tests.xfail_markers import chdb_python_table_rownumber_nondeterministic


class TestAggFuncSQLMapping(unittest.TestCase):
    """Test that all agg functions map correctly to SQL and match pandas."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'A', 'B', 'B', 'B'],
                'value': [1, 2, 2, 3, 3, 4],
                'score': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_count_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'value': 'count'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'count'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'value': 'sum'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'sum'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mean_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'score': 'mean'})
        ds_result = self.ds_df.groupby('category').agg({'score': 'mean'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_std_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'score': 'std'})
        ds_result = self.ds_df.groupby('category').agg({'score': 'std'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_dict_agg(self):
        """var must use varSamp (sample variance, ddof=1) to match pandas."""
        pd_result = self.df.groupby('category').agg({'score': 'var'})
        ds_result = self.ds_df.groupby('category').agg({'score': 'var'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_min_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'value': 'min'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'min'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_max_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'value': 'max'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'max'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_first_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'value': 'first'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'first'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_last_dict_agg(self):
        pd_result = self.df.groupby('category').agg({'value': 'last'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'last'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nunique_dict_agg(self):
        """nunique must map to uniqExact in SQL."""
        pd_result = self.df.groupby('category').agg({'value': 'nunique'})
        ds_result = self.ds_df.groupby('category').agg({'value': 'nunique'})
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNamedAgg(unittest.TestCase):
    """Test named aggregation mode: agg(alias=('col', 'func'))."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'A', 'B', 'B', 'B'],
                'value': [1, 2, 2, 3, 3, 4],
                'score': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_count_named_agg(self):
        pd_result = self.df.groupby('category').agg(cnt=('value', 'count'))
        ds_result = self.ds_df.groupby('category').agg(cnt=('value', 'count'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_named_agg(self):
        pd_result = self.df.groupby('category').agg(total=('value', 'sum'))
        ds_result = self.ds_df.groupby('category').agg(total=('value', 'sum'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_mean_named_agg(self):
        pd_result = self.df.groupby('category').agg(avg=('score', 'mean'))
        ds_result = self.ds_df.groupby('category').agg(avg=('score', 'mean'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_std_named_agg(self):
        pd_result = self.df.groupby('category').agg(s=('score', 'std'))
        ds_result = self.ds_df.groupby('category').agg(s=('score', 'std'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_named_agg(self):
        """var must use varSamp (sample variance, ddof=1) to match pandas."""
        pd_result = self.df.groupby('category').agg(v=('score', 'var'))
        ds_result = self.ds_df.groupby('category').agg(v=('score', 'var'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_min_named_agg(self):
        pd_result = self.df.groupby('category').agg(mi=('value', 'min'))
        ds_result = self.ds_df.groupby('category').agg(mi=('value', 'min'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_max_named_agg(self):
        pd_result = self.df.groupby('category').agg(ma=('value', 'max'))
        ds_result = self.ds_df.groupby('category').agg(ma=('value', 'max'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_first_named_agg(self):
        pd_result = self.df.groupby('category').agg(f=('value', 'first'))
        ds_result = self.ds_df.groupby('category').agg(f=('value', 'first'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_last_named_agg(self):
        pd_result = self.df.groupby('category').agg(l=('value', 'last'))
        ds_result = self.ds_df.groupby('category').agg(l=('value', 'last'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nunique_named_agg(self):
        """nunique must map to uniqExact in SQL."""
        pd_result = self.df.groupby('category').agg(nu=('value', 'nunique'))
        ds_result = self.ds_df.groupby('category').agg(nu=('value', 'nunique'))
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAllAggsCombined(unittest.TestCase):
    """Test all agg functions combined in a single agg call."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'A', 'B', 'B', 'B'],
                'value': [1, 2, 2, 3, 3, 4],
                'score': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    @chdb_python_table_rownumber_nondeterministic
    def test_all_named_agg_combined(self):
        """All agg functions in one named_agg call."""
        pd_result = self.df.groupby('category').agg(
            cnt=('value', 'count'),
            total=('value', 'sum'),
            avg=('score', 'mean'),
            s=('score', 'std'),
            v=('score', 'var'),
            mi=('value', 'min'),
            ma=('value', 'max'),
            f=('value', 'first'),
            l=('value', 'last'),
            nu=('value', 'nunique'),
        )
        ds_result = self.ds_df.groupby('category').agg(
            cnt=('value', 'count'),
            total=('value', 'sum'),
            avg=('score', 'mean'),
            s=('score', 'std'),
            v=('score', 'var'),
            mi=('value', 'min'),
            ma=('value', 'max'),
            f=('value', 'first'),
            l=('value', 'last'),
            nu=('value', 'nunique'),
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_func_dict_agg(self):
        """Multiple functions per column in dict format."""
        pd_result = self.df.groupby('category').agg(
            {'value': ['count', 'sum', 'mean', 'std', 'var', 'min', 'max']}
        )
        ds_result = self.ds_df.groupby('category').agg(
            {'value': ['count', 'sum', 'mean', 'std', 'var', 'min', 'max']}
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_col_multi_func_dict_agg(self):
        """Multiple columns, each with different functions."""
        pd_result = self.df.groupby('category').agg(
            {'value': ['sum', 'count'], 'score': ['mean', 'std']}
        )
        ds_result = self.ds_df.groupby('category').agg(
            {'value': ['sum', 'count'], 'score': ['mean', 'std']}
        )
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDirectAggMethods(unittest.TestCase):
    """Test direct groupby().method() calls."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'category': ['A', 'A', 'A', 'B', 'B', 'B'],
                'value': [1, 2, 2, 3, 3, 4],
                'score': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_groupby_count(self):
        pd_result = self.df.groupby('category').count()
        ds_result = self.ds_df.groupby('category').count()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_sum(self):
        pd_result = self.df.groupby('category').sum()
        ds_result = self.ds_df.groupby('category').sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_mean(self):
        pd_result = self.df.groupby('category').mean()
        ds_result = self.ds_df.groupby('category').mean()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_std(self):
        pd_result = self.df.groupby('category').std()
        ds_result = self.ds_df.groupby('category').std()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_var(self):
        """var() must return sample variance (ddof=1) matching pandas."""
        pd_result = self.df.groupby('category').var()
        ds_result = self.ds_df.groupby('category').var()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_min(self):
        pd_result = self.df.groupby('category').min()
        ds_result = self.ds_df.groupby('category').min()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_max(self):
        pd_result = self.df.groupby('category').max()
        ds_result = self.ds_df.groupby('category').max()
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_first(self):
        pd_result = self.df.groupby('category').first()
        ds_result = self.ds_df.groupby('category').first()
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_groupby_last(self):
        pd_result = self.df.groupby('category').last()
        ds_result = self.ds_df.groupby('category').last()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMultiColumnGroupby(unittest.TestCase):
    """Test multi-column groupby with various agg combinations."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'dept': ['A', 'A', 'B', 'B', 'A', 'B'],
                'role': ['eng', 'mgr', 'eng', 'mgr', 'eng', 'eng'],
                'salary': [100, 200, 150, 250, 120, 180],
                'bonus': [10, 20, 15, 25, 12, 18],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_multi_col_groupby_named_agg(self):
        pd_result = self.df.groupby(['dept', 'role']).agg(
            avg_sal=('salary', 'mean'),
            total=('salary', 'sum'),
            cnt=('salary', 'count'),
        )
        ds_result = self.ds_df.groupby(['dept', 'role']).agg(
            avg_sal=('salary', 'mean'),
            total=('salary', 'sum'),
            cnt=('salary', 'count'),
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_col_groupby_dict_agg(self):
        pd_result = self.df.groupby(['dept', 'role']).agg(
            {'salary': 'sum', 'bonus': 'mean'}
        )
        ds_result = self.ds_df.groupby(['dept', 'role']).agg(
            {'salary': 'sum', 'bonus': 'mean'}
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_col_groupby_sum(self):
        pd_result = self.df.groupby(['dept', 'role']).sum()
        ds_result = self.ds_df.groupby(['dept', 'role']).sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_col_groupby_multi_func(self):
        pd_result = self.df.groupby(['dept', 'role']).agg(
            {'salary': ['sum', 'mean', 'count'], 'bonus': ['min', 'max']}
        )
        ds_result = self.ds_df.groupby(['dept', 'role']).agg(
            {'salary': ['sum', 'mean', 'count'], 'bonus': ['min', 'max']}
        )
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComputedColumnGroupby(unittest.TestCase):
    """Test computed columns as groupby key or agg target."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'],
                'value': [10, 20, 30, 40, 50, 60],
                'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_computed_column_as_groupby_key(self):
        """Computed column (value // 30) as groupby key."""
        self.df['bucket'] = self.df['value'] // 30
        self.ds_df['bucket'] = self.ds_df['value'] // 30

        pd_result = self.df.groupby('bucket').agg({'score': 'mean'})
        ds_result = self.ds_df.groupby('bucket').agg({'score': 'mean'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_computed_column_as_agg_target(self):
        """Computed column (value * 2) as agg target."""
        self.df['doubled'] = self.df['value'] * 2
        self.ds_df['doubled'] = self.ds_df['value'] * 2

        pd_result = self.df.groupby('name').agg({'doubled': 'sum'})
        ds_result = self.ds_df.groupby('name').agg({'doubled': 'sum'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_computed_column_named_agg(self):
        """Computed column in named_agg context."""
        self.df['doubled'] = self.df['value'] * 2
        self.ds_df['doubled'] = self.ds_df['value'] * 2

        pd_result = self.df.groupby('name').agg(
            d_sum=('doubled', 'sum'),
            d_mean=('doubled', 'mean'),
        )
        ds_result = self.ds_df.groupby('name').agg(
            d_sum=('doubled', 'sum'),
            d_mean=('doubled', 'mean'),
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_computed_key_and_target(self):
        """Both groupby key and agg target are computed columns."""
        self.df['bucket'] = self.df['value'] // 30
        self.df['doubled'] = self.df['score'] * 2
        self.ds_df['bucket'] = self.ds_df['value'] // 30
        self.ds_df['doubled'] = self.ds_df['score'] * 2

        pd_result = self.df.groupby('bucket').agg(
            total=('doubled', 'sum'),
            avg=('doubled', 'mean'),
        )
        ds_result = self.ds_df.groupby('bucket').agg(
            total=('doubled', 'sum'),
            avg=('doubled', 'mean'),
        )
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFirstLastRowOrder(unittest.TestCase):
    """Test first/last preserve row order via argMin/argMax + rowNumberInAllBlocks."""

    @chdb_python_table_rownumber_nondeterministic
    def test_first_preserves_insertion_order(self):
        """first() should return the first value by insertion order."""
        df = pd.DataFrame(
            {'g': ['A', 'A', 'A', 'B', 'B', 'B'], 'v': [10, 20, 30, 40, 50, 60]}
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g')['v'].first()
        ds_result = ds_df.groupby('g')['v'].first()
        assert_series_equal(ds_result, pd_result, check_names=False)

    @chdb_python_table_rownumber_nondeterministic
    def test_last_preserves_insertion_order(self):
        """last() should return the last value by insertion order."""
        df = pd.DataFrame(
            {'g': ['A', 'A', 'A', 'B', 'B', 'B'], 'v': [10, 20, 30, 40, 50, 60]}
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g')['v'].last()
        ds_result = ds_df.groupby('g')['v'].last()
        assert_series_equal(ds_result, pd_result, check_names=False)

    @chdb_python_table_rownumber_nondeterministic
    def test_first_last_in_named_agg(self):
        """first/last work correctly via named_agg syntax."""
        df = pd.DataFrame(
            {
                'g': ['X', 'X', 'Y', 'Y', 'Z', 'Z'],
                'a': [1, 2, 3, 4, 5, 6],
                'b': ['p', 'q', 'r', 's', 't', 'u'],
            }
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg(
            first_a=('a', 'first'),
            last_a=('a', 'last'),
            first_b=('b', 'first'),
            last_b=('b', 'last'),
        )
        ds_result = ds_df.groupby('g').agg(
            first_a=('a', 'first'),
            last_a=('a', 'last'),
            first_b=('b', 'first'),
            last_b=('b', 'last'),
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_python_table_rownumber_nondeterministic
    def test_first_last_large_data(self):
        """first/last with large dataset - deterministic row ordering."""
        n = 5000
        categories = ['A', 'B', 'C', 'D', 'E'] * (n // 5)
        values = list(range(n))

        df = pd.DataFrame({'g': categories, 'v': values})
        ds_df = ds.DataFrame(df.copy())

        pd_first = df.groupby('g')['v'].first().sort_index()
        ds_first = get_series(ds_df.groupby('g')['v'].first()).sort_index()
        assert_series_equal(ds_first, pd_first, check_names=False)

        pd_last = df.groupby('g')['v'].last().sort_index()
        ds_last = get_series(ds_df.groupby('g')['v'].last()).sort_index()
        assert_series_equal(ds_last, pd_last, check_names=False)


class TestNuniqueMapping(unittest.TestCase):
    """Test nunique -> uniqExact mapping in all agg contexts."""

    def test_nunique_basic(self):
        df = pd.DataFrame({'g': ['A', 'A', 'B', 'B', 'B'], 'v': [1, 1, 2, 3, 3]})
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'nunique'})
        ds_result = ds_df.groupby('g').agg({'v': 'nunique'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nunique_with_nulls(self):
        """nunique should exclude NaN by default (matching pandas dropna=True)."""
        df = pd.DataFrame(
            {
                'g': ['A', 'A', 'A', 'B', 'B', 'B'],
                'v': [1.0, np.nan, 1.0, 2.0, 2.0, np.nan],
            }
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'nunique'})
        ds_result = ds_df.groupby('g').agg({'v': 'nunique'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nunique_all_unique(self):
        df = pd.DataFrame({'g': ['A', 'A', 'A', 'B', 'B'], 'v': [1, 2, 3, 4, 5]})
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'nunique'})
        ds_result = ds_df.groupby('g').agg({'v': 'nunique'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nunique_named_agg(self):
        df = pd.DataFrame({'g': ['A', 'A', 'B', 'B', 'B'], 'v': [1, 1, 2, 3, 3]})
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg(distinct_count=('v', 'nunique'))
        ds_result = ds_df.groupby('g').agg(distinct_count=('v', 'nunique'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nunique_with_strings(self):
        df = pd.DataFrame(
            {
                'g': ['A', 'A', 'A', 'B', 'B', 'B'],
                'v': ['x', 'y', 'x', 'p', 'q', 'q'],
            }
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'nunique'})
        ds_result = ds_df.groupby('g').agg({'v': 'nunique'})
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestVarSampMapping(unittest.TestCase):
    """Test var -> varSamp mapping (sample variance, ddof=1)."""

    def test_var_sample_variance(self):
        """Verify var uses sample variance (ddof=1), not population variance."""
        df = pd.DataFrame(
            {
                'g': ['A', 'A', 'A', 'B', 'B', 'B'],
                'v': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'var'})
        ds_result = ds_df.groupby('g').agg({'v': 'var'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_single_element_groups(self):
        """var of single element group should be NaN (pandas behavior)."""
        df = pd.DataFrame({'g': ['A', 'B', 'C'], 'v': [10.0, 20.0, 30.0]})
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'var'})
        ds_result = ds_df.groupby('g').agg({'v': 'var'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_two_element_groups(self):
        df = pd.DataFrame(
            {'g': ['A', 'A', 'B', 'B'], 'v': [10.0, 20.0, 30.0, 50.0]}
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg({'v': 'var'})
        ds_result = ds_df.groupby('g').agg({'v': 'var'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_named_agg(self):
        df = pd.DataFrame(
            {
                'g': ['A', 'A', 'A', 'B', 'B', 'B'],
                'v': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').agg(variance=('v', 'var'))
        ds_result = ds_df.groupby('g').agg(variance=('v', 'var'))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_var_direct_method(self):
        """groupby().var() direct method call."""
        df = pd.DataFrame(
            {
                'g': ['A', 'A', 'A', 'B', 'B', 'B'],
                'v': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                'w': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            }
        )
        ds_df = ds.DataFrame(df.copy())

        pd_result = df.groupby('g').var()
        ds_result = ds_df.groupby('g').var()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAsIndexFalse(unittest.TestCase):
    """Test as_index=False returns group keys as columns."""

    def setUp(self):
        self.df = pd.DataFrame(
            {'category': ['A', 'A', 'B', 'B', 'B'], 'value': [10, 20, 30, 40, 50]}
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_as_index_false_dict_agg(self):
        pd_result = self.df.groupby('category', as_index=False).agg({'value': 'sum'})
        ds_result = self.ds_df.groupby('category', as_index=False).agg({'value': 'sum'})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_as_index_false_named_agg(self):
        pd_result = self.df.groupby('category', as_index=False).agg(
            total=('value', 'sum')
        )
        ds_result = self.ds_df.groupby('category', as_index=False).agg(
            total=('value', 'sum')
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_as_index_false_direct_method(self):
        pd_result = self.df.groupby('category', as_index=False).sum()
        ds_result = self.ds_df.groupby('category', as_index=False).sum()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnSelection(unittest.TestCase):
    """Test groupby with column selection."""

    def setUp(self):
        self.df = pd.DataFrame(
            {
                'g': ['A', 'A', 'B', 'B'],
                'a': [1, 2, 3, 4],
                'b': [10, 20, 30, 40],
                'c': [100, 200, 300, 400],
            }
        )
        self.ds_df = ds.DataFrame(self.df.copy())

    def test_column_selection_single(self):
        pd_result = self.df.groupby('g')['a'].sum()
        ds_result = self.ds_df.groupby('g')['a'].sum()
        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_column_selection_multiple(self):
        pd_result = self.df.groupby('g')[['a', 'b']].sum()
        ds_result = self.ds_df.groupby('g')[['a', 'b']].sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_selection_mean(self):
        pd_result = self.df.groupby('g')[['a', 'b']].mean()
        ds_result = self.ds_df.groupby('g')[['a', 'b']].mean()
        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    unittest.main()
