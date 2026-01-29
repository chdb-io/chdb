"""
Exploratory Tests for DataStore Edge Cases and Boundary Conditions

Tests discovered through systematic exploration of:
1. Complex chained operations (filter -> groupby -> agg -> sort)
2. Boundary conditions (empty DataFrame, single row, large duplicates)
3. pandas API compatibility for common operations
4. Type combinations (int + float, string + None)

Following Mirror Code Pattern: each DataStore operation has a corresponding pandas
operation for comparison.
"""

import unittest
import numpy as np
import pandas as pd

import datastore as ds
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_frame_equal,
    assert_series_equal,
)


class TestEmptyDataFrameOperations(unittest.TestCase):
    """Test operations on empty DataFrames."""

    def test_empty_dataframe_creation(self):
        """Test creating empty DataFrame with columns."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = ds.DataFrame({'A': [], 'B': []})

        assert list(ds_df.columns) == list(pd_df.columns)
        assert len(ds_df) == len(pd_df) == 0

    def test_empty_dataframe_filter(self):
        """Test filtering that results in empty DataFrame."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

        pd_result = pd_df[pd_df['A'] > 100]
        ds_result = ds_df[ds_df['A'] > 100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dataframe_groupby(self):
        """Test groupby on empty DataFrame."""
        pd_df = pd.DataFrame({'group': [], 'value': []})
        ds_df = ds.DataFrame({'group': [], 'value': []})

        # Filter to empty then groupby
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_empty_dataframe_sort_values(self):
        """Test sorting empty DataFrame."""
        pd_df = pd.DataFrame({'A': [], 'B': []})
        ds_df = ds.DataFrame({'A': [], 'B': []})

        pd_result = pd_df.sort_values('A')
        ds_result = ds_df.sort_values('A')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSingleRowOperations(unittest.TestCase):
    """Test operations on single-row DataFrames."""

    def test_single_row_filter_match(self):
        """Test filter that keeps the single row."""
        pd_df = pd.DataFrame({'A': [5], 'B': ['x']})
        ds_df = ds.DataFrame({'A': [5], 'B': ['x']})

        pd_result = pd_df[pd_df['A'] >= 5]
        ds_result = ds_df[ds_df['A'] >= 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_filter_no_match(self):
        """Test filter that removes the single row."""
        pd_df = pd.DataFrame({'A': [5], 'B': ['x']})
        ds_df = ds.DataFrame({'A': [5], 'B': ['x']})

        pd_result = pd_df[pd_df['A'] > 5]
        ds_result = ds_df[ds_df['A'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_groupby(self):
        """Test groupby with single row."""
        pd_df = pd.DataFrame({'group': ['A'], 'value': [10]})
        ds_df = ds.DataFrame({'group': ['A'], 'value': [10]})

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_single_row_agg(self):
        """Test aggregation functions on single row."""
        pd_df = pd.DataFrame({'A': [5.0]})
        ds_df = ds.DataFrame({'A': [5.0]})

        # Test mean on single value
        pd_mean = pd_df['A'].mean()
        ds_mean = ds_df['A'].mean()
        np.testing.assert_almost_equal(float(ds_mean), float(pd_mean))

        # Test std on single value
        pd_std = pd_df['A'].std()
        ds_std = ds_df['A'].std()
        # pandas returns NaN for std of single value with default ddof=1
        assert pd.isna(pd_std) == pd.isna(ds_std)


class TestManyDuplicatesOperations(unittest.TestCase):
    """Test operations with many duplicate values."""

    def test_duplicates_in_groupby(self):
        """Test groupby when all values in group column are the same."""
        pd_df = pd.DataFrame({
            'group': ['A'] * 100,
            'value': list(range(100))
        })
        ds_df = ds.DataFrame({
            'group': ['A'] * 100,
            'value': list(range(100))
        })

        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_duplicates_in_sort_stability(self):
        """Test that sort is stable for duplicate values."""
        pd_df = pd.DataFrame({
            'A': [1, 1, 1, 2, 2, 2],
            'B': ['a', 'b', 'c', 'd', 'e', 'f']
        })
        ds_df = ds.DataFrame({
            'A': [1, 1, 1, 2, 2, 2],
            'B': ['a', 'b', 'c', 'd', 'e', 'f']
        })

        # pandas stable sort preserves order within groups
        pd_result = pd_df.sort_values('A', kind='stable')
        ds_result = ds_df.sort_values('A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_value_counts_with_ties(self):
        """Test value_counts when multiple values have same count."""
        pd_df = pd.DataFrame({'A': ['x', 'y', 'x', 'y', 'z', 'z']})
        ds_df = ds.DataFrame({'A': ['x', 'y', 'x', 'y', 'z', 'z']})

        pd_result = pd_df['A'].value_counts()
        ds_result = ds_df['A'].value_counts()

        # value_counts order for ties is implementation-dependent
        # Compare values but not order
        assert_series_equal(ds_result, pd_result, check_index=False)


class TestChainedOperations(unittest.TestCase):
    """Test complex chained operations."""

    def setUp(self):
        self.data = {
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'C'],
            'subcategory': ['x', 'x', 'y', 'y', 'x', 'y', 'x', 'y'],
            'value': [10, 20, 30, 40, 50, 60, 70, 80],
            'weight': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = ds.DataFrame(self.data)

    def test_filter_then_groupby_then_sort(self):
        """Test filter -> groupby -> sort chain."""
        pd_result = (
            self.pd_df[self.pd_df['value'] > 20]
            .groupby('category')['value']
            .sum()
            .reset_index()
            .sort_values('value', ascending=False)
        )
        ds_result = (
            self.ds_df[self.ds_df['value'] > 20]
            .groupby('category')['value']
            .sum()
            .reset_index()
            .sort_values('value', ascending=False)
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multiple_filters_chained(self):
        """Test multiple filter operations chained together."""
        pd_result = self.pd_df[
            (self.pd_df['category'] == 'A') & (self.pd_df['value'] > 20)
        ]
        ds_result = self.ds_df[
            (self.ds_df['category'] == 'A') & (self.ds_df['value'] > 20)
        ]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_groupby_multiple_aggs(self):
        """Test filter -> groupby with multiple aggregations."""
        pd_result = (
            self.pd_df[self.pd_df['value'] >= 30]
            .groupby('category')
            .agg({'value': 'sum', 'weight': 'mean'})
            .reset_index()
        )
        ds_result = (
            self.ds_df[self.ds_df['value'] >= 30]
            .groupby('category')
            .agg({'value': 'sum', 'weight': 'mean'})
            .reset_index()
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_then_filter_result(self):
        """Test groupby -> filter on aggregated result."""
        # Filter groups where sum > 50
        pd_grouped = self.pd_df.groupby('category')['value'].sum().reset_index()
        pd_result = pd_grouped[pd_grouped['value'] > 50]

        ds_grouped = self.ds_df.groupby('category')['value'].sum().reset_index()
        ds_result = ds_grouped[ds_grouped['value'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_sort_then_head(self):
        """Test sort -> head chain."""
        pd_result = self.pd_df.sort_values('value', ascending=False).head(3)
        ds_result = self.ds_df.sort_values('value', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_then_tail(self):
        """Test sort -> tail chain."""
        pd_result = self.pd_df.sort_values('value', ascending=True).tail(3)
        ds_result = self.ds_df.sort_values('value', ascending=True).tail(3)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestTypeHandling(unittest.TestCase):
    """Test type combinations and conversions."""

    def test_int_float_mixed_column(self):
        """Test operations on columns with mixed int and float."""
        pd_df = pd.DataFrame({'A': [1, 2.5, 3, 4.5]})
        ds_df = ds.DataFrame({'A': [1, 2.5, 3, 4.5]})

        pd_result = pd_df['A'].sum()
        ds_result = ds_df['A'].sum()

        np.testing.assert_almost_equal(float(ds_result), float(pd_result))

    def test_string_with_none(self):
        """Test string column with None values."""
        pd_df = pd.DataFrame({'A': ['a', None, 'c', None]})
        ds_df = ds.DataFrame({'A': ['a', None, 'c', None]})

        # Count non-null
        pd_count = pd_df['A'].count()
        ds_count = ds_df['A'].count()
        assert int(ds_count) == int(pd_count) == 2

    def test_numeric_with_nan(self):
        """Test numeric operations with NaN values."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0, np.nan, 5.0]})
        ds_df = ds.DataFrame({'A': [1.0, np.nan, 3.0, np.nan, 5.0]})

        # Sum should skip NaN
        pd_sum = pd_df['A'].sum()
        ds_sum = ds_df['A'].sum()
        np.testing.assert_almost_equal(float(ds_sum), float(pd_sum))

        # Mean should skip NaN
        pd_mean = pd_df['A'].mean()
        ds_mean = ds_df['A'].mean()
        np.testing.assert_almost_equal(float(ds_mean), float(pd_mean))

    def test_boolean_column_operations(self):
        """Test operations on boolean columns."""
        pd_df = pd.DataFrame({'A': [True, False, True, True, False]})
        ds_df = ds.DataFrame({'A': [True, False, True, True, False]})

        # Sum of booleans counts True values
        pd_sum = pd_df['A'].sum()
        ds_sum = ds_df['A'].sum()
        assert int(ds_sum) == int(pd_sum) == 3

        # Mean of booleans gives ratio of True
        pd_mean = pd_df['A'].mean()
        ds_mean = ds_df['A'].mean()
        np.testing.assert_almost_equal(float(ds_mean), float(pd_mean))


class TestNullHandling(unittest.TestCase):
    """Test NULL/None/NaN handling across operations."""

    def test_filter_with_null(self):
        """Test filtering with NULL values."""
        pd_df = pd.DataFrame({'A': [1, None, 3, None, 5], 'B': ['a', 'b', 'c', 'd', 'e']})
        ds_df = ds.DataFrame({'A': [1, None, 3, None, 5], 'B': ['a', 'b', 'c', 'd', 'e']})

        # Filter rows where A is not null
        pd_result = pd_df[pd_df['A'].notna()]
        ds_result = ds_df[ds_df['A'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_with_null_in_group(self):
        """Test groupby when group column has NULL values."""
        pd_df = pd.DataFrame({
            'group': ['A', 'B', None, 'A', None],
            'value': [1, 2, 3, 4, 5]
        })
        ds_df = ds.DataFrame({
            'group': ['A', 'B', None, 'A', None],
            'value': [1, 2, 3, 4, 5]
        })

        # pandas groupby excludes NULL groups by default
        pd_result = pd_df.groupby('group')['value'].sum().reset_index()
        ds_result = ds_df.groupby('group')['value'].sum().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_fillna_basic(self):
        """Test fillna with scalar value."""
        pd_df = pd.DataFrame({'A': [1.0, np.nan, 3.0, np.nan]})
        ds_df = ds.DataFrame({'A': [1.0, np.nan, 3.0, np.nan]})

        pd_result = pd_df['A'].fillna(0)
        ds_result = ds_df['A'].fillna(0)

        assert_series_equal(ds_result, pd_result)

    def test_dropna_basic(self):
        """Test dropna to remove rows with NULL."""
        pd_df = pd.DataFrame({
            'A': [1, None, 3],
            'B': ['x', 'y', None]
        })
        ds_df = ds.DataFrame({
            'A': [1, None, 3],
            'B': ['x', 'y', None]
        })

        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSelectAndProjection(unittest.TestCase):
    """Test column selection and projection operations."""

    def test_select_single_column(self):
        """Test selecting a single column."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

        pd_result = pd_df[['A']]
        ds_result = ds_df[['A']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_multiple_columns(self):
        """Test selecting multiple columns."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

        pd_result = pd_df[['A', 'C']]
        ds_result = ds_df[['A', 'C']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_columns_different_order(self):
        """Test selecting columns in different order."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

        pd_result = pd_df[['C', 'A', 'B']]
        ds_result = ds_df[['C', 'A', 'B']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_columns(self):
        """Test dropping columns."""
        pd_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        ds_df = ds.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

        pd_result = pd_df.drop(columns=['B'])
        ds_result = ds_df.drop(columns=['B'])

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAssignOperations(unittest.TestCase):
    """Test assign and column creation operations."""

    def test_assign_scalar(self):
        """Test assigning a scalar value as new column."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = ds.DataFrame({'A': [1, 2, 3]})

        pd_result = pd_df.assign(B=10)
        ds_result = ds_df.assign(B=10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_computed(self):
        """Test assigning a computed column."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = ds.DataFrame({'A': [1, 2, 3]})

        pd_result = pd_df.assign(B=pd_df['A'] * 2)
        ds_result = ds_df.assign(B=ds_df['A'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple(self):
        """Test assigning multiple columns at once."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = ds.DataFrame({'A': [1, 2, 3]})

        pd_result = pd_df.assign(B=pd_df['A'] * 2, C=pd_df['A'] + 10)
        ds_result = ds_df.assign(B=ds_df['A'] * 2, C=ds_df['A'] + 10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_chained(self):
        """Test chained assign operations."""
        pd_df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = ds.DataFrame({'A': [1, 2, 3]})

        pd_result = pd_df.assign(B=pd_df['A'] * 2).assign(C=lambda x: x['B'] + 1)
        ds_result = ds_df.assign(B=ds_df['A'] * 2).assign(C=lambda x: x['B'] + 1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComparisonOperations(unittest.TestCase):
    """Test comparison and boolean operations."""

    def test_eq_comparison(self):
        """Test equality comparison."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 2, 1]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 2, 1]})

        pd_result = pd_df['A'] == 2
        ds_result = ds_df['A'] == 2

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_ne_comparison(self):
        """Test not-equal comparison."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 2, 1]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 2, 1]})

        pd_result = pd_df['A'] != 2
        ds_result = ds_df['A'] != 2

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_string_comparison(self):
        """Test string equality comparison."""
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c', 'b', 'a']})
        ds_df = ds.DataFrame({'A': ['a', 'b', 'c', 'b', 'a']})

        pd_result = pd_df['A'] == 'b'
        ds_result = ds_df['A'] == 'b'

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_combined_conditions(self):
        """Test combined boolean conditions."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})

        pd_result = pd_df[(pd_df['A'] > 2) & (pd_df['B'] < 4)]
        ds_result = ds_df[(ds_df['A'] > 2) & (ds_df['B'] < 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_or_condition(self):
        """Test OR condition."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df[(pd_df['A'] < 2) | (pd_df['A'] > 4)]
        ds_result = ds_df[(ds_df['A'] < 2) | (ds_df['A'] > 4)]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsinOperation(unittest.TestCase):
    """Test isin operation."""

    def test_isin_with_list(self):
        """Test isin with list of values."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df['A'].isin([1, 3, 5])]
        ds_result = ds_df[ds_df['A'].isin([1, 3, 5])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_with_strings(self):
        """Test isin with string values."""
        pd_df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e']})
        ds_df = ds.DataFrame({'A': ['a', 'b', 'c', 'd', 'e']})

        pd_result = pd_df[pd_df['A'].isin(['a', 'c', 'e'])]
        ds_result = ds_df[ds_df['A'].isin(['a', 'c', 'e'])]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isin_empty_list(self):
        """Test isin with empty list."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df[pd_df['A'].isin([])]
        ds_result = ds_df[ds_df['A'].isin([])]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNlargestNsmallest(unittest.TestCase):
    """Test nlargest and nsmallest operations."""

    def test_nlargest_basic(self):
        """Test nlargest basic usage."""
        pd_df = pd.DataFrame({'A': [5, 1, 3, 2, 4], 'B': ['e', 'a', 'c', 'b', 'd']})
        ds_df = ds.DataFrame({'A': [5, 1, 3, 2, 4], 'B': ['e', 'a', 'c', 'b', 'd']})

        pd_result = pd_df.nlargest(3, 'A')
        ds_result = ds_df.nlargest(3, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_basic(self):
        """Test nsmallest basic usage."""
        pd_df = pd.DataFrame({'A': [5, 1, 3, 2, 4], 'B': ['e', 'a', 'c', 'b', 'd']})
        ds_df = ds.DataFrame({'A': [5, 1, 3, 2, 4], 'B': ['e', 'a', 'c', 'b', 'd']})

        pd_result = pd_df.nsmallest(3, 'A')
        ds_result = ds_df.nsmallest(3, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_n_larger_than_rows(self):
        """Test nlargest when n is larger than number of rows."""
        pd_df = pd.DataFrame({'A': [5, 1, 3]})
        ds_df = ds.DataFrame({'A': [5, 1, 3]})

        pd_result = pd_df.nlargest(10, 'A')
        ds_result = ds_df.nlargest(10, 'A')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUniqueAndNunique(unittest.TestCase):
    """Test unique and nunique operations."""

    def test_unique_integers(self):
        """Test unique on integer column."""
        pd_series = pd.Series([1, 2, 2, 3, 1, 3, 4])
        ds_series = ds.Series([1, 2, 2, 3, 1, 3, 4])

        pd_result = sorted(pd_series.unique())
        ds_result = sorted(ds_series.unique())

        np.testing.assert_array_equal(ds_result, pd_result)

    def test_unique_strings(self):
        """Test unique on string column."""
        pd_series = pd.Series(['a', 'b', 'a', 'c', 'b'])
        ds_series = ds.Series(['a', 'b', 'a', 'c', 'b'])

        pd_result = sorted(pd_series.unique())
        ds_result = sorted(ds_series.unique())

        np.testing.assert_array_equal(ds_result, pd_result)

    def test_nunique(self):
        """Test nunique (number of unique values)."""
        pd_df = pd.DataFrame({'A': [1, 2, 2, 3, 1, 3, 4]})
        ds_df = ds.DataFrame({'A': [1, 2, 2, 3, 1, 3, 4]})

        pd_result = pd_df['A'].nunique()
        ds_result = ds_df['A'].nunique()

        assert int(ds_result) == int(pd_result) == 4


class TestRenameOperations(unittest.TestCase):
    """Test rename operations."""

    def test_rename_columns_dict(self):
        """Test renaming columns with dict."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df = ds.DataFrame({'A': [1, 2], 'B': [3, 4]})

        pd_result = pd_df.rename(columns={'A': 'X', 'B': 'Y'})
        ds_result = ds_df.rename(columns={'A': 'X', 'B': 'Y'})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_partial(self):
        """Test renaming only some columns."""
        pd_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        ds_df = ds.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})

        pd_result = pd_df.rename(columns={'A': 'X'})
        ds_result = ds_df.rename(columns={'A': 'X'})

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestConcatOperations(unittest.TestCase):
    """Test concat operations."""

    def test_concat_two_dataframes(self):
        """Test concatenating two DataFrames."""
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        ds_df1 = ds.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = ds.DataFrame({'A': [5, 6], 'B': [7, 8]})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_empty_dataframe(self):
        """Test concatenating with empty DataFrame."""
        pd_df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        pd_df2 = pd.DataFrame({'A': [], 'B': []})
        ds_df1 = ds.DataFrame({'A': [1, 2], 'B': [3, 4]})
        ds_df2 = ds.DataFrame({'A': [], 'B': []})

        pd_result = pd.concat([pd_df1, pd_df2], ignore_index=True)
        ds_result = ds.concat([ds_df1, ds_df2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestDescribeAndInfo(unittest.TestCase):
    """Test describe and info operations."""

    def test_describe_basic(self):
        """Test describe on numeric columns."""
        pd_df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0], 'B': [5.0, 4.0, 3.0, 2.0, 1.0]})
        ds_df = ds.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0], 'B': [5.0, 4.0, 3.0, 2.0, 1.0]})

        pd_result = pd_df.describe()
        ds_result = ds_df.describe()

        # Compare statistics values
        for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            for col in ['A', 'B']:
                np.testing.assert_almost_equal(
                    float(ds_result.loc[stat, col]),
                    float(pd_result.loc[stat, col]),
                    decimal=4,
                    err_msg=f"Mismatch at {stat}, {col}"
                )


class TestClipOperation(unittest.TestCase):
    """Test clip operation."""

    def test_clip_both_bounds(self):
        """Test clip with both lower and upper bounds."""
        pd_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ds_series = ds.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        pd_result = pd_series.clip(lower=3, upper=7)
        ds_result = ds_series.clip(lower=3, upper=7)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_lower_only(self):
        """Test clip with only lower bound."""
        pd_series = pd.Series([1, 2, 3, 4, 5])
        ds_series = ds.Series([1, 2, 3, 4, 5])

        pd_result = pd_series.clip(lower=3)
        ds_result = ds_series.clip(lower=3)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_upper_only(self):
        """Test clip with only upper bound."""
        pd_series = pd.Series([1, 2, 3, 4, 5])
        ds_series = ds.Series([1, 2, 3, 4, 5])

        pd_result = pd_series.clip(upper=3)
        ds_result = ds_series.clip(upper=3)

        assert_series_equal(ds_result, pd_result, check_names=False)


if __name__ == '__main__':
    unittest.main()

    def test_notin_empty_list(self):
        """Test negated isin with empty list (should return all rows)."""
        pd_df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        ds_df = ds.DataFrame({'A': [1, 2, 3, 4, 5]})

        pd_result = pd_df[~pd_df['A'].isin([])]
        ds_result = ds_df[~ds_df['A'].isin([])]

        assert_datastore_equals_pandas(ds_result, pd_result)
