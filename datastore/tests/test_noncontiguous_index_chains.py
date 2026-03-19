"""
Tests for non-contiguous index operation chains.

Verifies that DataFrames with non-contiguous indices (produced by step slicing,
dropna, filter, sample, etc.) work correctly through chDB SQL execution.

Key concern: _prepare_df_for_chdb() must reset non-contiguous indices before
SQL execution and restore them in results.

Mirror Pattern: All tests compare DataStore behavior with pandas behavior.
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


class TestStepSliceThenSQLOps:
    """df[::2] followed by SQL operations - verify data matches pandas."""

    def test_step2_then_filter(self):
        """Step slice then filter: chDB must handle non-contiguous index."""
        data = {'a': list(range(20)), 'b': [x * 10 for x in range(20)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        pd_result = pd_result[pd_result['a'] > 5]

        ds_result = ds[::2]
        ds_result = ds_result[ds_result['a'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step2_then_sort_values(self):
        """Step slice then sort: verify data and order match pandas."""
        data = {'a': [9, 1, 7, 3, 5, 8, 2, 6, 4, 0], 'b': list(range(10))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2].sort_values('a')
        ds_result = ds[::2].sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step2_then_column_selection(self):
        """Step slice then column selection."""
        data = {'a': list(range(10)), 'b': list(range(10, 20)), 'c': list(range(20, 30))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2][['a', 'c']]
        ds_result = ds[::2][['a', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step2_then_arithmetic_column(self):
        """Step slice then add arithmetic column via assign."""
        data = {'a': list(range(10)), 'b': list(range(10, 20))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2].assign(c=lambda df: df['a'] + df['b'])
        ds_result = ds[::2].assign(c=lambda df: df['a'] + df['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step2_then_aggregation_sum(self):
        """Step slice then aggregation: result should be scalar."""
        data = {'a': list(range(10))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_sum = pd_df[::2]['a'].sum()
        ds_sum = ds[::2]['a'].sum()

        assert ds_sum == pd_sum, f"Sum mismatch: {ds_sum} != {pd_sum}"

    def test_step2_then_aggregation_mean(self):
        """Step slice then mean aggregation."""
        data = {'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_mean = pd_df[::2]['a'].mean()
        ds_mean = ds[::2]['a'].mean()

        assert abs(ds_mean - pd_mean) < 1e-10, f"Mean mismatch: {ds_mean} != {pd_mean}"

    def test_step2_then_groupby_agg(self):
        """Step slice then groupby+agg."""
        data = {
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2].groupby('category')['value'].sum()
        ds_result = ds[::2].groupby('category')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_step2_then_len(self):
        """Step slice then len(): basic sanity check."""
        data = {'a': list(range(10))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        assert len(ds[::2]) == len(pd_df[::2])
        assert len(ds[::3]) == len(pd_df[::3])
        assert len(ds[1::4]) == len(pd_df[1::4])


class TestOffsetStepSliceThenChainedOps:
    """df[1::3] followed by filter + assign + sort."""

    def test_offset_step_then_filter_assign_sort(self):
        """df[1::3] -> filter -> assign -> sort: full chain."""
        data = {
            'x': list(range(30)),
            'y': [i * 2.5 for i in range(30)],
            'group': ['A', 'B', 'C'] * 10,
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        # Step slice with offset
        pd_sliced = pd_df[1::3]
        ds_sliced = ds[1::3]

        # Filter
        pd_filtered = pd_sliced[pd_sliced['x'] > 10]
        ds_filtered = ds_sliced[ds_sliced['x'] > 10]

        # Assign new column
        pd_assigned = pd_filtered.assign(z=lambda df: df['x'] * 2 + df['y'])
        ds_assigned = ds_filtered.assign(z=lambda df: df['x'] * 2 + df['y'])

        # Sort
        pd_sorted = pd_assigned.sort_values('z')
        ds_sorted = ds_assigned.sort_values('z')

        assert_datastore_equals_pandas(ds_sorted, pd_sorted)

    def test_offset_step_then_filter(self):
        """df[1::3] -> filter."""
        data = {'val': list(range(20))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[1::3]
        pd_result = pd_result[pd_result['val'] > 5]

        ds_result = ds[1::3]
        ds_result = ds_result[ds_result['val'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_offset_step_then_assign(self):
        """df[1::3] -> assign."""
        data = {'a': list(range(15)), 'b': list(range(15, 30))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[1::3].assign(c=lambda df: df['a'] - df['b'])
        ds_result = ds[1::3].assign(c=lambda df: df['a'] - df['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_offset_step_then_sort(self):
        """df[1::3] -> sort_values."""
        data = {'a': [5, 3, 8, 1, 9, 2, 7, 4, 6, 0, 11, 10]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[1::3].sort_values('a')
        ds_result = ds[1::3].sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_offset_step_then_multiple_filters(self):
        """df[2::3] -> filter -> filter: chained filters on non-contiguous index."""
        data = {'a': list(range(30)), 'b': [x % 5 for x in range(30)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[2::3]
        pd_result = pd_result[pd_result['a'] > 5]
        pd_result = pd_result[pd_result['b'] < 4]

        ds_result = ds[2::3]
        ds_result = ds_result[ds_result['a'] > 5]
        ds_result = ds_result[ds_result['b'] < 4]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_various_step_sizes(self):
        """Test different step sizes all produce correct results."""
        data = {'val': list(range(50))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        for start in [0, 1, 2]:
            for step in [2, 3, 5, 7]:
                pd_result = pd_df[start::step]
                ds_result = ds[start::step]
                assert_datastore_equals_pandas(
                    ds_result, pd_result,
                    msg=f"Failed for [{start}::{step}]"
                )


class TestDropNAThenGroupbyAgg:
    """dropna() produces non-contiguous index, then groupby + agg."""

    def test_dropna_then_groupby_sum(self):
        """dropna -> groupby -> sum."""
        data = {
            'category': ['A', 'B', None, 'A', 'B', 'A', None, 'B', 'A', 'B'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna(subset=['category']).groupby('category')['value'].sum()
        ds_result = ds.dropna(subset=['category']).groupby('category')['value'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dropna_then_groupby_mean(self):
        """dropna -> groupby -> mean."""
        data = {
            'group': ['X', 'Y', 'X', None, 'Y', 'X', 'Y', None, 'X', 'Y'],
            'score': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna().groupby('group')['score'].mean()
        ds_result = ds.dropna().groupby('group')['score'].mean()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dropna_then_groupby_agg_multiple(self):
        """dropna -> groupby -> agg with multiple functions."""
        data = {
            'cat': ['A', 'B', 'A', None, 'B', 'A', 'B', None],
            'val1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'val2': [10, 20, 30, 40, 50, 60, 70, 80],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna().groupby('cat').agg({'val1': 'sum', 'val2': 'mean'})
        ds_result = ds.dropna().groupby('cat').agg({'val1': 'sum', 'val2': 'mean'})

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_dropna_then_filter(self):
        """dropna -> filter: non-contiguous index preserved through filter."""
        data = {
            'a': [1, None, 3, 4, None, 6, 7, 8, None, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna()
        pd_result = pd_result[pd_result['b'] > 50]

        ds_result = ds.dropna()
        ds_result = ds_result[ds_result['b'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_sort(self):
        """dropna -> sort_values."""
        data = {
            'name': ['c', None, 'a', 'b', None, 'e', 'd'],
            'score': [30, 20, 10, 40, 50, 60, 70],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna().sort_values('name')
        ds_result = ds.dropna().sort_values('name')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_assign(self):
        """dropna -> assign: new column on non-contiguous index."""
        data = {
            'x': [1, None, 3, None, 5, 6],
            'y': [10, 20, 30, 40, 50, 60],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna().assign(z=lambda df: df['y'] * 2)
        ds_result = ds.dropna().assign(z=lambda df: df['y'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_values_column_then_chain(self):
        """dropna on value column -> filter -> assign."""
        data = {
            'group': ['A', 'B', 'A', 'B', 'A', 'B'],
            'value': [10.0, np.nan, 30.0, np.nan, 50.0, 60.0],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna(subset=['value'])
        pd_result = pd_result[pd_result['value'] > 20]
        pd_result = pd_result.assign(doubled=lambda df: df['value'] * 2)

        ds_result = ds.dropna(subset=['value'])
        ds_result = ds_result[ds_result['value'] > 20]
        ds_result = ds_result.assign(doubled=lambda df: df['value'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFilterProducesNonContiguousIndex:
    """Boolean filter produces non-contiguous index, then chain ops."""

    def test_filter_then_filter_then_sort(self):
        """filter -> filter -> sort: doubly-filtered non-contiguous index."""
        data = {'a': list(range(20)), 'b': [x % 3 for x in range(20)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['a'] > 5]
        pd_result = pd_result[pd_result['b'] != 0]
        pd_result = pd_result.sort_values('a', ascending=False)

        ds_result = ds[ds['a'] > 5]
        ds_result = ds_result[ds_result['b'] != 0]
        ds_result = ds_result.sort_values('a', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_groupby_count(self):
        """filter -> groupby -> count."""
        data = {
            'dept': ['ENG', 'HR', 'ENG', 'HR', 'ENG', 'HR', 'ENG', 'HR'] * 3,
            'salary': [100, 50, 120, 60, 80, 40, 110, 70] * 3,
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['salary'] > 60].groupby('dept')['salary'].count()
        ds_result = ds[ds['salary'] > 60].groupby('dept')['salary'].count()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_filter_then_assign_then_filter(self):
        """filter -> assign -> filter: multiple non-contiguous index ops."""
        data = {
            'x': list(range(15)),
            'y': [i * 3 for i in range(15)],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['x'] > 3]
        pd_result = pd_result.assign(z=lambda df: df['x'] + df['y'])
        pd_result = pd_result[pd_result['z'] > 30]

        ds_result = ds[ds['x'] > 3]
        ds_result = ds_result.assign(z=lambda df: df['x'] + df['y'])
        ds_result = ds_result[ds_result['z'] > 30]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSampleThenChainedOps:
    """sample() produces non-contiguous index + chain operations."""

    def test_sample_then_sort(self):
        """sample -> sort: verify sorted result matches pandas values."""
        data = {'a': list(range(100)), 'b': [x * 2 for x in range(100)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        # Use fixed random_state for reproducibility
        pd_result = pd_df.sample(n=20, random_state=42).sort_values('a')
        ds_result = ds.sample(n=20, random_state=42).sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sample_then_filter(self):
        """sample -> filter: non-contiguous index through filter."""
        data = {'val': list(range(100))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.sample(n=50, random_state=123)
        pd_result = pd_result[pd_result['val'] > 50]
        pd_result = pd_result.sort_values('val')

        ds_result = ds.sample(n=50, random_state=123)
        ds_result = ds_result[ds_result['val'] > 50]
        ds_result = ds_result.sort_values('val')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sample_then_assign(self):
        """sample -> assign: add column on sampled non-contiguous index."""
        data = {'x': list(range(50)), 'y': [i * 10 for i in range(50)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.sample(n=15, random_state=99).assign(
            z=lambda df: df['x'] + df['y']
        )
        pd_result = pd_result.sort_values('x')
        ds_result = ds.sample(n=15, random_state=99).assign(
            z=lambda df: df['x'] + df['y']
        )
        ds_result = ds_result.sort_values('x')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sample_then_groupby(self):
        """sample -> groupby: verify aggregation on sampled data."""
        data = {
            'cat': ['A', 'B', 'C'] * 30,
            'val': list(range(90)),
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.sample(n=60, random_state=7).groupby('cat')['val'].sum()
        ds_result = ds.sample(n=60, random_state=7).groupby('cat')['val'].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_sample_frac_then_sort(self):
        """sample(frac=) -> sort: fractional sampling."""
        data = {'a': list(range(40))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.sample(frac=0.3, random_state=55).sort_values('a')
        ds_result = ds.sample(frac=0.3, random_state=55).sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestMixedNonContiguousIndexSources:
    """Combine multiple sources of non-contiguous indices."""

    def test_step_slice_then_dropna_then_filter(self):
        """step_slice -> dropna -> filter: triple non-contiguous source."""
        data = {
            'a': [1, None, 3, 4, None, 6, 7, 8, None, 10, 11, 12],
            'b': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]  # non-contiguous
        pd_result = pd_result.dropna()  # more non-contiguous
        pd_result = pd_result[pd_result['b'] > 300]  # even more

        ds_result = ds[::2]
        ds_result = ds_result.dropna()
        ds_result = ds_result[ds_result['b'] > 300]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_step_slice(self):
        """filter -> step_slice: filter first then step slice."""
        data = {'a': list(range(20)), 'b': [x % 4 for x in range(20)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['b'] < 3][::2]
        ds_result = ds[ds['b'] < 3][::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_then_step_slice_then_sort(self):
        """dropna -> step_slice -> sort."""
        data = {
            'val': [5, None, 3, 8, None, 1, 7, None, 2, 6, None, 4, 9, None, 0],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna()[::2].sort_values('val')
        ds_result = ds.dropna()[::2].sort_values('val')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_slice_then_filter_then_groupby(self):
        """step_slice -> filter -> groupby: complex chain."""
        data = {
            'group': ['A', 'B', 'C', 'A', 'B', 'C'] * 5,
            'val': list(range(30)),
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        pd_result = pd_result[pd_result['val'] > 5]
        pd_result = pd_result.groupby('group')['val'].mean()

        ds_result = ds[::2]
        ds_result = ds_result[ds_result['val'] > 5]
        ds_result = ds_result.groupby('group')['val'].mean()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestNonContiguousIndexColumnExpressions:
    """Verify column expressions work correctly on non-contiguous index DataFrames."""

    def test_step_slice_column_arithmetic(self):
        """Arithmetic expression on step-sliced DataFrame."""
        data = {'a': list(range(10)), 'b': list(range(10, 20))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_sliced = pd_df[::3]
        ds_sliced = ds[::3]

        pd_result = pd_sliced['a'] + pd_sliced['b']
        ds_result = ds_sliced['a'] + ds_sliced['b']

        assert list(ds_result.values) == list(pd_result.values)
        assert list(ds_result.index) == list(pd_result.index)

    def test_dropna_column_comparison(self):
        """Comparison expression on dropna'd DataFrame."""
        data = {'val': [1, None, 3, None, 5, 6, None, 8]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_clean = pd_df.dropna()
        ds_clean = ds.dropna()

        pd_result = pd_clean['val'] > 4
        ds_result = ds_clean['val'] > 4

        assert list(ds_result.values) == list(pd_result.values)
        assert list(ds_result.index) == list(pd_result.index)

    def test_filter_column_multiply(self):
        """Multiply expression on filtered DataFrame."""
        data = {'x': [1, 2, 3, 4, 5, 6, 7, 8]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_filtered = pd_df[pd_df['x'] > 3]
        ds_filtered = ds[ds['x'] > 3]

        pd_result = pd_filtered['x'] * 10
        ds_result = ds_filtered['x'] * 10

        assert list(ds_result.values) == list(pd_result.values)
        assert list(ds_result.index) == list(pd_result.index)

    def test_step_slice_string_operations(self):
        """String accessor on step-sliced DataFrame."""
        data = {'name': ['alice', 'bob', 'charlie', 'david', 'eve', 'frank']}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]['name'].str.upper()
        ds_result = ds[::2]['name'].str.upper()

        assert list(ds_result.values) == list(pd_result.values)
        assert list(ds_result.index) == list(pd_result.index)


class TestNonContiguousIndexDataIntegrity:
    """Verify data integrity: values at each index position are correct."""

    def test_step_slice_values_match_original_positions(self):
        """After df[::2], values at index 0,2,4,... must match original rows."""
        data = {'id': list(range(10)), 'val': [x * 100 for x in range(10)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        # Verify each row's data matches
        ds_df = ds_result.to_df() if hasattr(ds_result, 'to_df') else pd.DataFrame(ds_result)
        for idx in pd_result.index:
            assert ds_df.loc[idx, 'id'] == pd_result.loc[idx, 'id'], (
                f"id mismatch at index {idx}"
            )
            assert ds_df.loc[idx, 'val'] == pd_result.loc[idx, 'val'], (
                f"val mismatch at index {idx}"
            )

    def test_offset_step_values_match(self):
        """After df[1::3], values at correct positions."""
        data = {'id': list(range(15)), 'name': [f'item_{i}' for i in range(15)]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[1::3]
        ds_result = ds[1::3]

        assert_datastore_equals_pandas(ds_result, pd_result)

        # Verify specific values
        ds_df = ds_result.to_df() if hasattr(ds_result, 'to_df') else pd.DataFrame(ds_result)
        expected_indices = [1, 4, 7, 10, 13]
        assert list(ds_df.index) == expected_indices
        assert list(ds_df['id']) == expected_indices

    def test_filter_preserves_correct_values(self):
        """After filter, remaining rows have correct original values."""
        data = {
            'id': list(range(10)),
            'val': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['val'] > 50]
        ds_result = ds[ds['val'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

        ds_df = ds_result.to_df() if hasattr(ds_result, 'to_df') else pd.DataFrame(ds_result)
        # Rows with val > 50 are at original indices 5,6,7,8,9
        assert list(ds_df.index) == [5, 6, 7, 8, 9]
        assert list(ds_df['val']) == [55, 65, 75, 85, 95]

    def test_dropna_preserves_correct_values(self):
        """After dropna, remaining rows have correct original values."""
        data = {
            'id': [0, 1, 2, 3, 4, 5],
            'val': [10, None, 30, None, 50, 60],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna()
        ds_result = ds.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_chained_ops_preserve_values(self):
        """Complex chain: verify final values are exactly correct."""
        data = {
            'id': list(range(20)),
            'score': [i * 5 for i in range(20)],
            'group': ['X', 'Y'] * 10,
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        # Step slice -> filter -> sort
        pd_result = pd_df[::2]  # ids: 0,2,4,6,8,10,12,14,16,18
        pd_result = pd_result[pd_result['score'] > 20]  # ids: 6,8,10,12,14,16,18
        pd_result = pd_result.sort_values('score', ascending=False)

        ds_result = ds[::2]
        ds_result = ds_result[ds_result['score'] > 20]
        ds_result = ds_result.sort_values('score', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCases:
    """Edge cases for non-contiguous index handling."""

    def test_single_row_after_filter(self):
        """Filter to single row: non-contiguous with just one element."""
        data = {'a': [1, 2, 3, 4, 5]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['a'] == 3]
        ds_result = ds[ds['a'] == 3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_after_filter(self):
        """Filter to empty: no rows remain."""
        data = {'a': [1, 2, 3]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[pd_df['a'] > 100]
        ds_result = ds[ds['a'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_equals_length(self):
        """Step equals length: only first row returned."""
        data = {'a': [10, 20, 30, 40, 50]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::5]
        ds_result = ds[::5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_large_step_on_small_df(self):
        """Step larger than DataFrame."""
        data = {'a': [1, 2, 3]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::100]
        ds_result = ds[::100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_consecutive_step_slices(self):
        """Apply step slicing twice."""
        data = {'a': list(range(100))}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::3][::2]
        ds_result = ds[::3][::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all_valid(self):
        """dropna on DataFrame with no NaN: index should be contiguous."""
        data = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna()
        ds_result = ds.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all_nan(self):
        """dropna on DataFrame where all rows have NaN."""
        data = {'a': [None, None, None], 'b': [1, 2, 3]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df.dropna()
        ds_result = ds.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_noncontiguous_with_duplicate_values(self):
        """Non-contiguous index with duplicate values in data columns."""
        data = {'a': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

        # Also test sort on duplicates
        pd_sorted = pd_df[::2].sort_values('a')
        ds_sorted = ds[::2].sort_values('a')

        assert_datastore_equals_pandas(ds_sorted, pd_sorted)
