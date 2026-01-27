"""
Tests for known issues documented in todo.md

This file verifies the current status of all known issues.
Each issue is either:
- Marked with xfail (bug to be fixed)
- Verified as fixed (test passes)

NO WORKAROUND TESTS - if it's a bug, we fix it, not teach users to avoid it.

Run with: pytest tests/test_known_issues_verification.py -v
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


# =============================================================================
# Import xfail markers
# =============================================================================
from tests.xfail_markers import (
    # chDB limitations
    chdb_category_type,
    chdb_timedelta_type,
    chdb_nat_returns_nullable_int,
    # DataStore bugs
    bug_extractall_multiindex,
    bug_where_computed_column,
    # DataStore limitations
    limit_callable_index,
    limit_query_variable_scope,
    limit_datastore_no_invert,
)


# =============================================================================
# SECTION 1: Data Types - SQL Operations
# =============================================================================


class TestDataTypesSQLOperations:
    """Tests for data type issues during SQL operations."""

    def test_categorical_type_read_only_works(self):
        """Categorical type works for read-only access (no SQL execution)."""
        df = pd.DataFrame({'cat_col': pd.Categorical(['a', 'b', 'c']), 'value': [1, 2, 3]})
        ds = DataStore(df)

        # Read-only access works (no SQL execution triggered)
        assert list(ds['cat_col']) == ['a', 'b', 'c']

    @chdb_category_type
    def test_categorical_type_fails_in_sql(self):
        """Categorical type fails when SQL operations are needed."""
        df = pd.DataFrame({'cat_col': pd.Categorical(['a', 'b', 'c']), 'value': [1, 2, 3]})
        ds = DataStore(df)

        # SQL operation (filter) fails
        result = ds[ds['value'] > 1]
        len(result)  # Triggers SQL execution

    def test_timedelta_type_read_only_works(self):
        """Timedelta type works for read-only access."""
        df = pd.DataFrame({'td_col': pd.to_timedelta(['1 day', '2 days', '3 days']), 'value': [1, 2, 3]})
        ds = DataStore(df)

        # Read-only access works
        pd_values = df['td_col'].tolist()
        ds_values = list(ds['td_col'])
        assert pd_values == ds_values

    @chdb_timedelta_type
    def test_timedelta_type_fails_in_sql(self):
        """Timedelta type fails when SQL operations are needed."""
        df = pd.DataFrame({'td_col': pd.to_timedelta(['1 day', '2 days', '3 days']), 'value': [1, 2, 3]})
        ds = DataStore(df)

        # SQL operation (filter) fails
        result = ds[ds['value'] > 1]
        len(result)

    def test_numpy_array_columns_read_only_works(self):
        """Numpy array columns are preserved for read-only access."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        df = pd.DataFrame({'arr_col': [arr1, arr2], 'id': [1, 2]})
        ds = DataStore(df)

        ds_values = list(ds['arr_col'])
        # Arrays should be preserved for read-only access
        assert isinstance(ds_values[0], np.ndarray), f"Expected ndarray, got {type(ds_values[0])}"
        np.testing.assert_array_equal(ds_values[0], arr1)
        np.testing.assert_array_equal(ds_values[1], arr2)

    # NOTE: Array inside Nullable test is in test_chdb_dtype_differences.py::test_raw_sql_split_without_ifnull_fails
    # The issue occurs when SQL functions return Array type on nullable columns (e.g., splitByWhitespace)


# =============================================================================
# SECTION 2: Fixed Issues (Verification Tests)
# =============================================================================


class TestFixedIssues:
    """Tests verifying issues that are now FIXED."""

    def test_invert_operator_on_column_works(self):
        """~ invert operator on column now works."""
        df = pd.DataFrame({'a': [True, False, True]})
        ds = DataStore(df)

        pd_result = ~df['a']
        ds_result = ~ds['a']

        assert pd_result.tolist() == list(ds_result)

    def test_sum_of_all_nan_returns_zero(self):
        """Sum of all NaN now returns 0.0 same as pandas."""
        df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
        ds = DataStore(df)

        pd_sum = df['a'].sum()
        ds_sum = ds['a'].sum()

        # Both should return 0.0
        assert pd_sum == 0.0
        # numpy scalar repr differs between versions:
        # - older: '0.0'
        # - newer: 'np.float64(0.0)'
        assert float(ds_sum) == 0.0

    def test_index_preserved_after_filter(self):
        """Index is now preserved after SQL execution."""
        df = pd.DataFrame({'a': [1, 2, 3]}, index=['x', 'y', 'z'])
        ds = DataStore(df)

        pd_filtered = df[df['a'] > 1]
        ds_filtered = ds[ds['a'] > 1]

        assert pd_filtered.index.tolist() == ds_filtered.index.tolist()

    def test_index_preserved_after_sort(self):
        """Index is preserved after sort operation."""
        df = pd.DataFrame({'a': [3, 1, 2]}, index=['x', 'y', 'z'])
        ds = DataStore(df)

        pd_sorted = df.sort_values('a')
        ds_sorted = ds.sort_values('a')

        assert pd_sorted.index.tolist() == ds_sorted.index.tolist()

    def test_filter_with_lazy_assigned_column(self):
        """Filter with lazy assigned column now works."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df.copy())

        # Assign computed column
        ds['b'] = ds['a'] * 2

        # Filter on computed column should work
        ds_filtered = ds[ds['b'] > 5]

        # Verify
        assert len(ds_filtered) == 3
        assert list(ds_filtered['b']) == [6, 8, 10]

    def test_simple_alias_chain_works(self):
        """Simple assign + filter chain works now."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        ds = DataStore(df.copy())

        ds['value'] = ds['value'] * 2
        ds_filtered = ds[ds['value'] > 25]

        # Compare with pandas
        pdf = df.copy()
        pdf['value'] = pdf['value'] * 2
        pdf_filtered = pdf[pdf['value'] > 25]

        assert len(ds_filtered) == len(pdf_filtered)
        assert list(ds_filtered['value']) == pdf_filtered['value'].tolist()


# =============================================================================
# SECTION 3: P1 Bugs (High Priority - To Be Fixed)
# =============================================================================


class TestP1Bugs:
    """P1 bugs that should be fixed soon."""

    def test_null_string_comparison_fixed(self):
        """FIXED: != None now returns correct rows matching pandas behavior.

        In pandas, element-wise comparison with None:
        - col != None returns True for ALL rows (every element differs from singleton None)
        - col == None returns False for ALL rows

        This is different from .notna()/.isna() which check for NA values.
        """
        df = pd.DataFrame({'s': ['abc', 'def', None]})
        ds = DataStore(df)

        # != None returns all 3 rows (matches pandas)
        pd_result = df[df['s'] != None]  # noqa: E711
        ds_result = ds[ds['s'] != None]  # noqa: E711
        assert len(ds_result) == len(pd_result)  # Both return 3

        # == None returns 0 rows (matches pandas)
        pd_eq_result = df[df['s'] == None]  # noqa: E711
        ds_eq_result = ds[ds['s'] == None]  # noqa: E711
        assert len(ds_eq_result) == len(pd_eq_result)  # Both return 0

    @bug_where_computed_column
    def test_where_computed_column_bug(self):
        """BUG: where() with lazy assigned column fails."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df.copy())

        # Assign computed column
        ds['b'] = ds['a'] * 2

        # where() with computed column should work like pandas
        pd_df = df.copy()
        pd_df['b'] = pd_df['a'] * 2
        pd_result = pd_df.where(pd_df['b'] > 5)

        ds_result = ds.where(ds['b'] > 5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    @limit_query_variable_scope
    def test_query_variable_scope_bug(self):
        """BUG: query() with @variable has scope issues."""
        df = pd.DataFrame({'value': [10, 20, 30, 40]})
        ds = DataStore(df)

        threshold = 25

        pd_result = df.query('value > @threshold')
        ds_result = ds.query('value > @threshold')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_apply_method_call_fixed(self):
        """FIXED: groupby.apply() with lambda calling methods now works."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [1, 2, 3, 4]})
        ds = DataStore(df)

        pd_result = df.groupby('group')['value'].apply(lambda x: x.sum())
        ds_result = ds.groupby('group')['value'].apply(lambda x: x.sum())

        assert pd_result.tolist() == list(ds_result)


# =============================================================================
# SECTION 4: P2 Bugs (Medium Priority)
# =============================================================================


class TestP2Bugs:
    """P2 bugs to be fixed."""

    def test_extractall_multiindex_bug(self):
        """BUG: extractall MultiIndex is lost."""
        df = pd.DataFrame({'text': ['a1b2', 'c3d4']})
        ds = DataStore(df)

        pd_result = df['text'].str.extractall(r'(\d)')
        ds_result = ds['text'].str.extractall(r'(\d)')

        # Both should return MultiIndex
        assert isinstance(pd_result.index, pd.MultiIndex)
        assert isinstance(ds_result.index, pd.MultiIndex)


# =============================================================================
# SECTION 5: P3 Issues (Low Priority / Design Decisions)
# =============================================================================


class TestP3Issues:
    """P3 issues - lower priority or design decisions."""

    @limit_callable_index
    def test_callable_as_index_not_supported(self):
        """Callable as index is not supported."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds = DataStore(df)

        pd_result = df[lambda x: x['a'] > 2]
        ds_result = ds[lambda x: x['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_unstack_now_available(self):
        """unstack() now available on ColumnExpr."""
        df = pd.DataFrame({'cat1': ['A', 'A', 'B', 'B'], 'cat2': ['X', 'Y', 'X', 'Y'], 'value': [1, 2, 3, 4]})
        ds = DataStore(df)

        # pandas
        pd_grouped = df.groupby(['cat1', 'cat2'])['value'].sum()
        pd_result = pd_grouped.unstack()

        # DataStore
        ds_grouped = ds.groupby(['cat1', 'cat2'])['value'].sum()
        ds_result = ds_grouped.unstack()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    @limit_datastore_no_invert
    def test_dataframe_invert_not_supported(self):
        """~ds (entire DataFrame invert) not supported."""
        df = pd.DataFrame({'a': [True, False, True]})
        ds = DataStore(df)

        # Entire DataFrame invert fails
        result = ~ds
        len(result)


# =============================================================================
# SECTION 6: Dtype Differences (Values Correct)
# =============================================================================


class TestDtypeDifferences:
    """Tests where values are correct but dtype differs - NOT bugs."""

    @chdb_nat_returns_nullable_int
    def test_dtype_difference_dt_with_nat(self):
        """dt accessor with NaT returns different dtype (values correct)."""
        df = pd.DataFrame({'dt': pd.to_datetime(['2020-01-01', None, '2020-03-01'])})
        ds = DataStore(df)

        pd_year = df['dt'].dt.year
        ds_year = ds['dt'].dt.year

        # Check dtypes match exactly (this will fail - dtype differs)
        assert pd_year.dtype == ds_year.dtype


# =============================================================================
# Run
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
