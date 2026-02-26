"""
Deep edge case tests based on bug patterns discovered during fixes.

These tests explore combinations and edge cases that might reveal deeper issues.
Each test is designed to probe specific interaction patterns that caused previous bugs.

Bug patterns explored:
1. Nested LIMIT/WHERE with various orderings
2. GroupBy with NULL values and various aggregations
3. JOIN + chained operations
4. Mixed SQL/Pandas segments with complex expressions
5. Sort stability with duplicates and NaN
6. Column alias conflicts in complex queries
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from datastore import DataStore


class TestNestedLimitWherePatterns:
    """Test various nested LIMIT/WHERE patterns that can break SQL generation."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                'id': list(range(100)),
                'value': [i * 2 for i in range(100)],
                'category': ['A', 'B', 'C'] * 33 + ['A'],
            }
        )

    def test_limit_where_limit_where_order(self, sample_df):
        """Test: LIMIT -> WHERE -> LIMIT -> WHERE"""
        pdf = sample_df.copy()
        p = pdf[:50]
        p = p[p['value'] > 30]
        p = p[:20]
        p = p[p['value'] > 50]

        ds = DataStore.from_df(sample_df)
        ds = ds[:50]
        ds = ds[ds['value'] > 30]
        ds = ds[:20]
        ds = ds[ds['value'] > 50]
        d = ds.to_df()

        assert list(p['value']) == list(d['value'])

    def test_where_limit_where_limit_order(self, sample_df):
        """Test: WHERE -> LIMIT -> WHERE -> LIMIT"""
        pdf = sample_df.copy()
        p = pdf[pdf['value'] > 20]
        p = p[:30]
        p = p[p['value'] > 50]
        p = p[:10]

        ds = DataStore.from_df(sample_df)
        ds = ds[ds['value'] > 20]
        ds = ds[:30]
        ds = ds[ds['value'] > 50]
        ds = ds[:10]
        d = ds.to_df()

        assert list(p['value']) == list(d['value'])

    def test_triple_nested_limit_where(self, sample_df):
        """Test: LIMIT -> WHERE -> LIMIT -> WHERE -> LIMIT -> WHERE"""
        pdf = sample_df.copy()
        p = pdf[:80]
        p = p[p['value'] > 20]
        p = p[:50]
        p = p[p['value'] > 40]
        p = p[:30]
        p = p[p['value'] > 60]

        ds = DataStore.from_df(sample_df)
        ds = ds[:80]
        ds = ds[ds['value'] > 20]
        ds = ds[:50]
        ds = ds[ds['value'] > 40]
        ds = ds[:30]
        ds = ds[ds['value'] > 60]
        d = ds.to_df()

        assert list(p['value']) == list(d['value'])

    def test_limit_with_offset_and_where(self, sample_df):
        """Test: slice with start:stop and WHERE"""
        pdf = sample_df.copy()
        p = pdf[10:50]
        p = p[p['value'] > 40]

        ds = DataStore.from_df(sample_df)
        ds = ds[10:50]
        ds = ds[ds['value'] > 40]
        d = ds.to_df()

        assert list(p['value']) == list(d['value'])

    def test_multiple_slices_with_offset(self, sample_df):
        """Test: [10:50][5:20] chained slices"""
        pdf = sample_df.copy()
        p = pdf[10:50]
        p = p[5:20]

        ds = DataStore.from_df(sample_df)
        ds = ds[10:50]
        ds = ds[5:20]
        d = ds.to_df()

        assert list(p['value']) == list(d['value'])


class TestGroupByWithNullAndEdgeCases:
    """Test GroupBy operations with NULL values and edge cases."""

    @pytest.fixture
    def df_with_nulls(self):
        return pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'B', None, None],
                'value': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan],
                'count_col': [10, 20, 30, 40, 50, 60, 70],
            }
        )
    def test_groupby_count_null_category_pandas_default(self, df_with_nulls):
        """Test that pandas default excludes NULL while DataStore includes it.

        This documents the known difference:
        - pandas groupby(dropna=True): excludes NULL keys
        - SQL GROUP BY: includes NULL keys
        - DataStore: follows SQL semantics (includes NULL)
        """
        pdf = df_with_nulls.copy()
        # pandas default: dropna=True, excludes NULL
        pd_result = pdf.groupby('category').count()

        ds = DataStore.from_df(df_with_nulls)
        ds_result = ds.groupby('category').count().to_df()

        # This will fail: pandas has 2 groups (A, B), DataStore has 3 (A, B, None)
        assert set(pd_result.index) == set(ds_result.index)

    def test_groupby_count_with_null_category_dropna_false(self, df_with_nulls):
        """Test count() with NULL values - using dropna=False to include NULL groups."""
        pdf = df_with_nulls.copy()
        # Use dropna=False on both to include NULL groups
        pd_result = pdf.groupby('category', dropna=False).count()

        ds = DataStore.from_df(df_with_nulls)
        ds_result = ds.groupby('category', dropna=False).count().to_df()

        # Both should now include NULL category
        pd_categories = set(pd_result.index.tolist())
        ds_categories = set(ds_result.index.tolist())

        # Normalize None/NaN for comparison
        # pandas 3.0 may use np.nan, pd.NA, or None for missing values
        pd_has_null = any(pd.isna(cat) for cat in pd_categories)
        ds_has_null = any(pd.isna(cat) for cat in ds_categories)

        assert pd_has_null == ds_has_null, "NULL category handling mismatch"
        # Compare non-null categories
        pd_non_null = {c for c in pd_categories if not pd.isna(c)}
        ds_non_null = {c for c in ds_categories if not pd.isna(c)}
        assert pd_non_null == ds_non_null

    def test_groupby_sum_with_nan_values(self, df_with_nulls):
        """Test sum() with NaN values in aggregated column."""
        pdf = df_with_nulls.copy()
        pd_result = pdf.groupby('category')['value'].sum()

        ds = DataStore.from_df(df_with_nulls)
        ds_result = ds.groupby('category')['value'].sum()

        # Compare results
        for cat in pd_result.index:
            pd_val = pd_result.loc[cat]
            ds_val = ds_result[cat]
            if pd.isna(pd_val):
                assert pd.isna(ds_val)
            else:
                assert abs(pd_val - ds_val) < 0.001
    def test_groupby_multiple_agg_null_pandas_default(self, df_with_nulls):
        """Test that pandas default excludes NULL in multiple aggregations."""
        pdf = df_with_nulls.copy()
        pd_result = pdf.groupby('category').agg({'value': ['sum', 'mean', 'count'], 'count_col': ['sum', 'count']})

        ds = DataStore.from_df(df_with_nulls)
        ds_result = (
            ds.groupby('category').agg({'value': ['sum', 'mean', 'count'], 'count_col': ['sum', 'count']}).to_df()
        )

        # This will fail: pandas has 2 groups, DataStore has 3 (includes None)
        assert len(pd_result) == len(ds_result)

    def test_groupby_multiple_agg_with_nulls_dropna_false(self, df_with_nulls):
        """Test multiple aggregations with NULL/NaN values - using dropna=False."""
        pdf = df_with_nulls.copy()
        # Use dropna=False on both to include NULL keys
        pd_result = pdf.groupby('category', dropna=False).agg(
            {'value': ['sum', 'mean', 'count'], 'count_col': ['sum', 'count']}
        )

        ds = DataStore.from_df(df_with_nulls)
        ds_result = (
            ds.groupby('category', dropna=False).agg({'value': ['sum', 'mean', 'count'], 'count_col': ['sum', 'count']}).to_df()
        )

        # Verify structure - both should have 3 groups (A, B, None)
        assert len(pd_result) == len(ds_result)

    def test_groupby_size_vs_count_with_nulls(self, df_with_nulls):
        """Test that size() includes NaN while count() excludes NaN."""
        pdf = df_with_nulls.copy()

        # Get pandas results
        pd_size = pdf.groupby('category').size()
        pd_count = pdf.groupby('category')['value'].count()

        ds = DataStore.from_df(df_with_nulls)
        ds_size = ds.groupby('category').size()
        ds_count = ds.groupby('category')['value'].count()

        # size() should be >= count() because size includes NaN rows
        for cat in ['A', 'B']:
            assert ds_size[cat] >= ds_count[cat]
            assert pd_size[cat] >= pd_count[cat]


class TestJoinWithChainedOperations:
    """Test JOIN operations followed by various chained operations."""

    @pytest.fixture
    def temp_files(self):
        temp_dir = tempfile.mkdtemp()
        users_file = os.path.join(temp_dir, 'users.csv')
        orders_file = os.path.join(temp_dir, 'orders.csv')

        pd.DataFrame(
            {
                'user_id': [1, 2, 3, 4, 5],
                'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                'age': [25, 30, 35, 40, 45],
            }
        ).to_csv(users_file, index=False)

        pd.DataFrame(
            {
                'user_id': [1, 1, 2, 3, 3, 3, 4],
                'amount': [100, 200, 150, 300, 400, 50, 250],
                'product': ['A', 'B', 'A', 'C', 'A', 'B', 'C'],
            }
        ).to_csv(orders_file, index=False)

        yield users_file, orders_file

        import shutil

        shutil.rmtree(temp_dir)

    def test_join_then_filter(self, temp_files):
        """Test JOIN followed by filter."""
        users_file, orders_file = temp_files

        users = DataStore.from_file(users_file)
        orders = DataStore.from_file(orders_file)

        ds = users.join(orders, on='user_id')
        ds = ds[ds['amount'] > 150]
        result = ds.to_df()

        assert len(result) > 0
        assert all(result['amount'] > 150)

    def test_join_then_groupby(self, temp_files):
        """Test JOIN followed by groupby."""
        users_file, orders_file = temp_files

        users = DataStore.from_file(users_file)
        orders = DataStore.from_file(orders_file)

        ds = users.join(orders, on='user_id')
        result = ds.groupby('name')['amount'].sum()

        # Alice should have 300 (100 + 200)
        assert result['Alice'] == 300

    def test_join_then_apply_then_filter(self, temp_files):
        """Test JOIN -> apply -> filter chain."""
        users_file, orders_file = temp_files

        users = DataStore.from_file(users_file)
        orders = DataStore.from_file(orders_file)

        ds = users.join(orders, on='user_id')
        ds['discount'] = ds['amount'].apply(lambda x: x * 0.1 if x > 200 else 0)
        ds = ds[ds['discount'] > 0]
        result = ds.to_df()

        assert len(result) > 0
        assert all(result['amount'] > 200)

    def test_join_then_sort_then_limit(self, temp_files):
        """Test JOIN -> sort -> limit chain."""
        users_file, orders_file = temp_files

        users = DataStore.from_file(users_file)
        orders = DataStore.from_file(orders_file)

        ds = users.join(orders, on='user_id')
        ds = ds.sort('amount', ascending=False)
        ds = ds[:3]
        result = ds.to_df()

        assert len(result) == 3
        # Should be in descending order
        amounts = list(result['amount'])
        assert amounts == sorted(amounts, reverse=True)


class TestSortStabilityEdgeCases:
    """Test sort stability with duplicates, NaN, and complex scenarios."""

    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame(
            {
                'key': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
                'value': [1, 1, 2, 2, 1, 1, 2, 2],
                'order_id': [1, 2, 3, 4, 5, 6, 7, 8],  # Original order tracker
            }
        )

    def test_stable_sort_preserves_original_order(self, df_with_duplicates):
        """Test that stable sort preserves original order for equal keys."""
        pdf = df_with_duplicates.copy()
        pd_sorted = pdf.sort_values('key', kind='stable')

        ds = DataStore.from_df(df_with_duplicates)
        ds_sorted = ds.sort('key').to_df()

        # For rows with same key, original order should be preserved
        a_rows_pd = pd_sorted[pd_sorted['key'] == 'A']['order_id'].tolist()
        a_rows_ds = ds_sorted[ds_sorted['key'] == 'A']['order_id'].tolist()

        assert a_rows_pd == a_rows_ds

    def test_sort_with_nan_values(self):
        """Test sorting with NaN values."""
        df = pd.DataFrame(
            {
                'key': [3.0, np.nan, 1.0, np.nan, 2.0],
                'value': ['c', 'x', 'a', 'y', 'b'],
            }
        )

        pd_sorted = df.sort_values('key')
        ds = DataStore.from_df(df)
        ds_sorted = ds.sort('key').to_df()

        # NaN should be at the end
        assert pd.isna(pd_sorted['key'].iloc[-1])
        assert pd.isna(pd_sorted['key'].iloc[-2])

    def test_multi_column_sort_stability(self):
        """Test multi-column sort stability."""
        df = pd.DataFrame(
            {
                'a': ['X', 'X', 'Y', 'Y', 'X', 'Y'],
                'b': [1, 2, 1, 2, 1, 1],
                'original_pos': [0, 1, 2, 3, 4, 5],
            }
        )

        pd_sorted = df.sort_values(['a', 'b'], kind='stable')

        ds = DataStore.from_df(df)
        ds_sorted = ds.sort(['a', 'b']).to_df()

        # Same positions should have same order
        assert list(pd_sorted['original_pos']) == list(ds_sorted['original_pos'])


class TestColumnAliasConflicts:
    """Test scenarios where column aliases might conflict."""

    def test_filter_on_column_then_agg_same_column(self):
        """Test: filter on column X, then aggregate to create column X."""
        df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'B'],
                'value': [100, 200, 300, 400, 500],
            }
        )

        pdf = df[df['value'] > 150]
        pd_result = pdf.groupby('category')['value'].sum()

        ds = DataStore.from_df(df)
        ds = ds[ds['value'] > 150]
        ds_result = ds.groupby('category')['value'].sum()

        assert pd_result['A'] == ds_result['A']
        assert pd_result['B'] == ds_result['B']

    def test_multiple_where_same_column(self):
        """Test multiple WHERE conditions on same column."""
        df = pd.DataFrame(
            {
                'value': list(range(100)),
            }
        )

        pdf = df[(df['value'] > 20) & (df['value'] < 80)]

        ds = DataStore.from_df(df)
        ds = ds[(ds['value'] > 20) & (ds['value'] < 80)]
        result = ds.to_df()

        assert len(pdf) == len(result)
        assert list(pdf['value']) == list(result['value'])

    def test_computed_column_same_name_as_existing(self):
        """Test creating computed column with same name as existing column."""
        df = pd.DataFrame(
            {
                'value': [1, 2, 3, 4, 5],
            }
        )

        # Overwrite 'value' column
        ds = DataStore.from_df(df)
        ds['value'] = ds['value'] * 10
        result = ds.to_df()

        assert list(result['value']) == [10, 20, 30, 40, 50]


class TestComplexExpressionChains:
    """Test complex expression chains that might break SQL generation."""

    def test_arithmetic_then_comparison_then_groupby(self):
        """Test: arithmetic expression -> comparison -> groupby."""
        df = pd.DataFrame(
            {
                'category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'price': [100, 200, 150, 250, 300, 50],
                'quantity': [10, 5, 8, 4, 2, 20],
            }
        )

        # pandas reference
        pdf = df.copy()
        pdf['total'] = pdf['price'] * pdf['quantity']
        pdf = pdf[pdf['total'] > 500]
        pd_result = pdf.groupby('category')['total'].sum()

        # DataStore
        ds = DataStore.from_df(df)
        ds['total'] = ds['price'] * ds['quantity']
        ds = ds[ds['total'] > 500]
        ds_result = ds.groupby('category')['total'].sum()

        for cat in pd_result.index:
            assert pd_result[cat] == ds_result[cat]

    def test_string_operations_chain(self):
        """Test chained string operations."""
        df = pd.DataFrame(
            {
                'text': ['  Hello World  ', 'foo BAR', 'Test String'],
            }
        )

        # pandas reference
        pdf = df.copy()
        pd_result = pdf['text'].str.strip().str.lower()

        # DataStore
        ds = DataStore.from_df(df)
        ds_result = ds['text'].str.strip().str.lower().to_pandas()

        assert list(pd_result) == list(ds_result)

    def test_nested_function_calls(self):
        """Test nested function calls."""
        df = pd.DataFrame(
            {
                'value': [-5.5, 3.3, -2.2, 4.4],
            }
        )

        # abs(round(value))
        ds = DataStore.from_df(df)
        result = ds['value'].round().abs().to_pandas()

        expected = df['value'].round().abs()
        np.testing.assert_array_almost_equal(result.values, expected.values)


class TestEmptyDataFrameEdgeCases:
    """Test operations on empty DataFrames."""

    def test_filter_to_empty_then_operations(self):
        """Test operations after filtering to empty result."""
        df = pd.DataFrame(
            {
                'value': [1, 2, 3],
                'category': ['A', 'B', 'C'],
            }
        )

        ds = DataStore.from_df(df)
        ds = ds[ds['value'] > 100]  # Results in empty

        result = ds.to_df()
        assert len(result) == 0

    def test_empty_groupby(self):
        """Test groupby on empty DataFrame."""
        df = pd.DataFrame(
            {
                'value': [1, 2, 3],
                'category': ['A', 'B', 'C'],
            }
        )

        ds = DataStore.from_df(df)
        ds = ds[ds['value'] > 100]  # Results in empty

        # This should not raise an error
        try:
            result = ds.groupby('category')['value'].sum()
            # Result might be empty Series or have special handling
        except Exception as e:
            pytest.fail(f"groupby on empty df raised: {e}")


class TestIndexPreservation:
    """Test that index is preserved correctly through operations."""

    def test_filter_preserves_index(self):
        """Test that filter preserves original index."""
        df = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
            },
            index=[100, 200, 300, 400, 500],
        )

        pd_filtered = df[df['value'] > 25]

        ds = DataStore.from_df(df)
        ds_filtered = ds[ds['value'] > 25].to_df()

        assert list(pd_filtered.index) == list(ds_filtered.index)

    def test_sort_preserves_index(self):
        """Test that sort preserves original index mapping."""
        df = pd.DataFrame(
            {
                'value': [30, 10, 40, 20],
            },
            index=['c', 'a', 'd', 'b'],
        )

        pd_sorted = df.sort_values('value')

        ds = DataStore.from_df(df)
        ds_sorted = ds.sort('value').to_df()

        assert list(pd_sorted.index) == list(ds_sorted.index)

    def test_slice_preserves_index(self):
        """Test that slice preserves original index."""
        df = pd.DataFrame(
            {
                'value': [10, 20, 30, 40, 50],
            },
            index=['a', 'b', 'c', 'd', 'e'],
        )

        pd_sliced = df[1:4]

        ds = DataStore.from_df(df)
        ds_sliced = ds[1:4].to_df()

        assert list(pd_sliced.index) == list(ds_sliced.index)


class TestLargeOperationChains:
    """Test very long chains of operations."""

    def test_ten_operations_chain(self):
        """Test a chain of 10 operations."""
        df = pd.DataFrame(
            {
                'id': list(range(1000)),
                'value': np.random.randint(0, 100, 1000),
                'category': np.random.choice(['A', 'B', 'C'], 1000),
            }
        )

        # Build a complex chain
        pdf = df.copy()
        pdf = pdf[pdf['value'] > 10]
        pdf = pdf[:800]
        pdf = pdf[pdf['value'] < 90]
        pdf = pdf[:600]
        pdf = pdf[pdf['category'] == 'A']
        pdf = pdf[:400]
        pdf = pdf[pdf['value'] > 20]
        pdf = pdf[:200]
        pdf = pdf[pdf['value'] < 80]
        pdf = pdf[:100]

        ds = DataStore.from_df(df)
        ds = ds[ds['value'] > 10]
        ds = ds[:800]
        ds = ds[ds['value'] < 90]
        ds = ds[:600]
        ds = ds[ds['category'] == 'A']
        ds = ds[:400]
        ds = ds[ds['value'] > 20]
        ds = ds[:200]
        ds = ds[ds['value'] < 80]
        ds = ds[:100]

        result = ds.to_df()

        assert len(pdf) == len(result)
        assert list(pdf['value']) == list(result['value'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
