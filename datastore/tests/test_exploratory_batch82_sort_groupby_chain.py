"""
Exploratory Batch 82: Sort -> GroupBy Chain Operations

Focus areas:
1. sort_values() followed by groupby().first() / groupby().last()
2. Order preservation in chained operations
3. GroupBy aggregation type consistency (nunique returns uint64 vs int64)
4. Complex chain operations with sort and groupby

Discovered Issues:
- sort_values().groupby().first() does not preserve sort order in DataStore
- groupby().nunique() returns uint64 in DataStore vs int64 in pandas
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


class TestSortGroupbyFirstLast:
    """Test sort_values() -> groupby().first()/last() chains"""

    def test_sort_groupby_first_basic(self):
        """Sort by value then get first per group - basic case"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [10, 5, 20, 15]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [10, 5, 20, 15]
        })

        # Sort ascending, then get first per group
        pd_result = pd_df.sort_values('val').groupby('cat').first()
        ds_result = ds_df.sort_values('val').groupby('cat').first()

        # pandas: A=5, B=15 (smallest values per category)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_first_with_dates(self):
        """Sort by date then get first per group"""
        pd_df = pd.DataFrame({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'date': pd.to_datetime([
                '2024-01-02', '2024-01-01',  # A: earliest is 01-01
                '2024-01-03', '2024-01-01',  # B: earliest is 01-01
                '2024-01-02', '2024-01-01'   # C: earliest is 01-01
            ]),
            'value': [15, 10, 25, 20, 35, 30]
        })
        ds_df = DataStore({
            'category': ['A', 'A', 'B', 'B', 'C', 'C'],
            'date': pd.to_datetime([
                '2024-01-02', '2024-01-01',
                '2024-01-03', '2024-01-01',
                '2024-01-02', '2024-01-01'
            ]),
            'value': [15, 10, 25, 20, 35, 30]
        })

        # Sort by date, get first row per category
        pd_result = pd_df.sort_values('date').groupby('category').first()
        ds_result = ds_df.sort_values('date').groupby('category').first()

        # pandas result: A value=10, B value=20, C value=30 (all from 01-01)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_last(self):
        """Sort ascending then get last per group (should be max)"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 3, 2, 4]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 3, 2, 4]
        })

        # Sort ascending, get last = max per group
        pd_result = pd_df.sort_values('val').groupby('cat').last()
        ds_result = ds_df.sort_values('val').groupby('cat').last()

        # pandas: A=3, B=4
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending_groupby_first(self):
        """Sort descending then get first per group (should be max)"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 3, 2, 4]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 3, 2, 4]
        })

        # Sort descending, get first = max per group
        pd_result = pd_df.sort_values('val', ascending=False).groupby('cat').first()
        ds_result = ds_df.sort_values('val', ascending=False).groupby('cat').first()

        # pandas: A=3, B=4
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_multiple_cols_groupby_first(self):
        """Sort by multiple columns then get first per group"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'priority': [2, 1, 1, 2, 1],
            'val': [100, 200, 300, 400, 500]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'priority': [2, 1, 1, 2, 1],
            'val': [100, 200, 300, 400, 500]
        })

        # Sort by priority then val, get first per cat
        pd_result = pd_df.sort_values(['priority', 'val']).groupby('cat').first()
        ds_result = ds_df.sort_values(['priority', 'val']).groupby('cat').first()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupbyAggregationTypes:
    """Test groupby aggregation return types"""

    @pytest.mark.xfail(reason="DataStore returns uint64, pandas returns int64")
    def test_groupby_nunique_dtype(self):
        """GroupBy nunique should return int64 like pandas"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'val': [1, 1, 2, 3, 3]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'val': [1, 1, 2, 3, 3]
        })

        pd_result = pd_df.groupby('cat')['val'].nunique()
        ds_result = ds_df.groupby('cat')['val'].nunique()

        # Values should match
        assert list(pd_result.values) == list(ds_result.values)
        # Dtype should be int64 (not uint64)
        assert pd_result.dtype == ds_result.dtype, \
            f"dtype mismatch: pandas={pd_result.dtype}, DataStore={ds_result.dtype}"

    def test_groupby_nunique_values(self):
        """GroupBy nunique values should match pandas"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'val': [1, 1, 2, 3, 3]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'val': [1, 1, 2, 3, 3]
        })

        pd_result = pd_df.groupby('cat')['val'].nunique()
        ds_result = ds_df.groupby('cat')['val'].nunique()

        # Values should match (A=2, B=1)
        assert list(pd_result.values) == list(ds_result.values)

    def test_groupby_count_dtype(self):
        """GroupBy count should return int64 like pandas"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'B'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B', 'B'],
            'val': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df.groupby('cat')['val'].count()
        ds_result = ds_df.groupby('cat')['val'].count()

        assert list(pd_result.values) == list(ds_result.values)

    def test_groupby_size_values(self):
        """GroupBy size values should match pandas"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B', 'B'],
            'val': [1, 2, 3, 4, 5]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B', 'B'],
            'val': [1, 2, 3, 4, 5]
        })

        pd_result = pd_df.groupby('cat').size()
        ds_result = ds_df.groupby('cat').size()

        assert list(pd_result.values) == list(ds_result.values)


class TestComplexSortGroupbyChains:
    """Test complex chains involving sort and groupby"""

    def test_filter_sort_groupby_first(self):
        """Filter -> Sort -> GroupBy -> First chain"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
            'flag': [True, False, True, True, True, False],
            'val': [10, 20, 5, 30, 15, 25]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
            'flag': [True, False, True, True, True, False],
            'val': [10, 20, 5, 30, 15, 25]
        })

        # Filter flagged, sort by val, get first per cat
        pd_result = (pd_df[pd_df['flag']]
                     .sort_values('val')
                     .groupby('cat')
                     .first())
        ds_result = (ds_df[ds_df['flag']]
                     .sort_values('val')
                     .groupby('cat')
                     .first())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_nth(self):
        """Sort -> GroupBy -> nth(n) chain"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [30, 10, 20, 60, 40, 50]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [30, 10, 20, 60, 40, 50]
        })

        # Sort ascending, get 2nd smallest per group (nth(1))
        pd_result = pd_df.sort_values('val').groupby('cat').nth(1)
        ds_result = ds_df.sort_values('val').groupby('cat').nth(1)

        # A: sorted vals are 10,20,30 -> nth(1) is 20
        # B: sorted vals are 40,50,60 -> nth(1) is 50
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_head(self):
        """Sort -> GroupBy -> head(n) chain"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [30, 10, 20, 60, 40, 50]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B', 'B'],
            'val': [30, 10, 20, 60, 40, 50]
        })

        # Sort ascending, get top 2 smallest per group
        pd_result = pd_df.sort_values('val').groupby('cat').head(2)
        ds_result = ds_df.sort_values('val').groupby('cat').head(2)

        # Sort both results by val for comparison (head doesn't preserve order)
        pd_sorted = pd_result.sort_values('val').reset_index(drop=True)
        ds_sorted = ds_result.sort_values('val').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_sorted, pd_sorted)

    def test_assign_sort_groupby_first(self):
        """Assign -> Sort -> GroupBy -> First chain"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [10, 20, 30, 40]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [10, 20, 30, 40]
        })

        # Assign computed column, sort by it, get first per group
        pd_result = (pd_df
                     .assign(val_squared=lambda x: x['val'] ** 2)
                     .sort_values('val_squared')
                     .groupby('cat')
                     .first())
        ds_result = (ds_df
                     .assign(val_squared=lambda x: x['val'] ** 2)
                     .sort_values('val_squared')
                     .groupby('cat')
                     .first())

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupbyWithoutSort:
    """Test groupby first/last without prior sort (original order)"""

    def test_groupby_first_no_sort(self):
        """GroupBy first without sort uses original row order"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('cat').first()
        ds_result = ds_df.groupby('cat').first()

        # Without sort: A=1, B=3 (first occurrence in original order)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_last_no_sort(self):
        """GroupBy last without sort uses original row order"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 2, 3, 4]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'B', 'B'],
            'val': [1, 2, 3, 4]
        })

        pd_result = pd_df.groupby('cat').last()
        ds_result = ds_df.groupby('cat').last()

        # Without sort: A=2, B=4 (last occurrence in original order)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestEdgeCasesWithNulls:
    """Test sort -> groupby chains with NULL values"""

    def test_sort_groupby_first_with_nulls(self):
        """Sort -> GroupBy -> First with NULL values in sort column"""
        pd_df = pd.DataFrame({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'val': [np.nan, 10, 5, 20, np.nan]
        })
        ds_df = DataStore({
            'cat': ['A', 'A', 'A', 'B', 'B'],
            'val': [np.nan, 10, 5, 20, np.nan]
        })

        # Sort with na_position='last' (default), get first per group
        pd_result = pd_df.sort_values('val').groupby('cat').first()
        ds_result = ds_df.sort_values('val').groupby('cat').first()

        # A: sorted is 5, 10, NaN -> first is 5
        # B: sorted is 20, NaN -> first is 20
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_groupby_first_nulls_in_group(self):
        """Sort -> GroupBy -> First with NULL values in group column"""
        pd_df = pd.DataFrame({
            'cat': ['A', None, 'A', 'B', None],
            'val': [10, 20, 5, 30, 15]
        })
        ds_df = DataStore({
            'cat': ['A', None, 'A', 'B', None],
            'val': [10, 20, 5, 30, 15]
        })

        # Sort by val, groupby cat (with dropna=True by default)
        pd_result = pd_df.sort_values('val').groupby('cat').first()
        ds_result = ds_df.sort_values('val').groupby('cat').first()

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
