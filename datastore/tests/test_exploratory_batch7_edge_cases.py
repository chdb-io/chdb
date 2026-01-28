"""
Exploratory tests for Edge Cases - Batch 7

Focus areas:
1. Empty DataFrame operations
2. Type conversion boundaries
3. Index/Column selection edge cases
4. Groupby special cases
5. NULL/NaN handling boundaries

Uses Mirror Code Pattern: test DataStore vs pandas for correctness.
"""

import pytest
from tests.xfail_markers import chdb_duplicate_column_rename
import pandas as pd
import numpy as np

from datastore import DataStore
from tests.test_utils import get_dataframe, assert_datastore_equals_pandas, assert_series_equal, get_series


# ============================================================================
# Section 1: Empty DataFrame Operations
# ============================================================================


class TestEmptyDataFrameOperations:
    """Tests for operations on empty DataFrames."""

    def setup_method(self):
        """Create empty test DataFrames."""
        self.empty_df = pd.DataFrame({'a': [], 'b': [], 'c': []})
        self.empty_ds = DataStore(self.empty_df)

    def test_empty_filter_result(self):
        """Filter that returns empty result."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds = DataStore(df)

        pd_result = df[df['a'] > 100]
        ds_result = ds[ds['a'] > 100]

        assert len(ds_result) == 0
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_groupby_sum(self):
        """Groupby sum on empty DataFrame."""
        pd_result = self.empty_df.groupby('a')['b'].sum()
        ds_result = self.empty_ds.groupby('a')['b'].sum()

        assert len(ds_result) == 0

    def test_empty_groupby_count(self):
        """Groupby count on empty DataFrame."""
        pd_result = self.empty_df.groupby('a')['b'].count()
        ds_result = self.empty_ds.groupby('a')['b'].count()

        assert len(ds_result) == 0

    def test_empty_groupby_mean(self):
        """Groupby mean on empty DataFrame."""
        pd_result = self.empty_df.groupby('a')['b'].mean()
        ds_result = self.empty_ds.groupby('a')['b'].mean()

        assert len(ds_result) == 0

    def test_empty_value_counts(self):
        """value_counts on empty column."""
        df = pd.DataFrame({'a': []})
        ds = DataStore(df)

        pd_result = df['a'].value_counts()
        ds_result = ds['a'].value_counts()

        assert len(ds_result) == 0

    def test_empty_head(self):
        """head() on empty DataFrame."""
        pd_result = self.empty_df.head(5)
        ds_result = self.empty_ds.head(5)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_sort_values(self):
        """sort_values on empty DataFrame."""
        pd_result = self.empty_df.sort_values('a')
        ds_result = self.empty_ds.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_dropna(self):
        """dropna on empty DataFrame."""
        pd_result = self.empty_df.dropna()
        ds_result = self.empty_ds.dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_fillna(self):
        """fillna on empty DataFrame."""
        pd_result = self.empty_df.fillna(0)
        ds_result = self.empty_ds.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_column_selection(self):
        """Column selection on empty DataFrame."""
        pd_result = self.empty_df[['a', 'b']]
        ds_result = self.empty_ds[['a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_returns_empty_then_aggregate(self):
        """Filter that returns empty, then aggregate."""
        df = pd.DataFrame({'cat': ['A', 'B', 'C'], 'val': [1, 2, 3]})
        ds = DataStore(df)

        pd_filtered = df[df['val'] > 100]
        ds_filtered = ds[ds['val'] > 100]

        pd_result = pd_filtered['val'].sum()
        ds_result = ds_filtered['val'].sum()

        # Direct comparison - ColumnExpr supports __eq__
        assert pd_result == 0
        assert ds_result == 0

    def test_empty_distinct(self):
        """distinct on empty DataFrame."""
        pd_result = self.empty_df.drop_duplicates()
        ds_result = self.empty_ds.drop_duplicates()

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Section 2: Single Row DataFrame Operations
# ============================================================================


class TestSingleRowOperations:
    """Tests for operations on single-row DataFrames."""

    def setup_method(self):
        """Create single-row test data."""
        self.single_df = pd.DataFrame({'a': [1], 'b': [2], 'c': ['x']})
        self.single_ds = DataStore(self.single_df)

    def test_single_row_groupby(self):
        """Groupby on single row."""
        pd_result = self.single_df.groupby('c')['a'].sum().reset_index()
        ds_result = self.single_ds.groupby('c')['a'].sum()
        assert len(ds_result) == 1

    def test_single_row_head(self):
        """head(n) where n > row count."""
        pd_result = self.single_df.head(10)
        ds_result = self.single_ds.head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_tail(self):
        """tail(n) where n > row count."""
        pd_result = self.single_df.tail(10)
        ds_result = self.single_ds.tail(10)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_nlargest(self):
        """nlargest where n > row count."""
        pd_result = self.single_df.nlargest(5, 'a')
        ds_result = self.single_ds.nlargest(5, 'a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_nsmallest(self):
        """nsmallest where n > row count."""
        pd_result = self.single_df.nsmallest(5, 'a')
        ds_result = self.single_ds.nsmallest(5, 'a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_sample(self):
        """sample on single row."""
        pd_result = self.single_df.sample(frac=1.0, random_state=42)
        ds_result = self.single_ds.sample(frac=1.0, random_state=42)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_rank(self):
        """rank on single row."""
        pd_result = self.single_df['a'].rank()
        ds_result = self.single_ds['a'].rank()
        assert len(ds_result) == 1
        assert ds_result.iloc[0] == 1.0


# ============================================================================
# Section 3: NULL/NaN Handling Edge Cases
# ============================================================================


class TestNullHandlingEdgeCases:
    """Tests for NULL/NaN handling edge cases."""

    def setup_method(self):
        """Create test data with NaN values."""
        self.df_with_nan = pd.DataFrame(
            {'a': [1, None, 3, None, 5], 'b': [None, 2, None, 4, None], 'c': ['x', 'y', None, 'z', None]}
        )
        self.ds_with_nan = DataStore(self.df_with_nan)

    def test_filter_on_null_column(self):
        """Filter where condition column has NaN."""
        pd_result = self.df_with_nan[self.df_with_nan['a'] > 2]
        ds_result = self.ds_with_nan[self.ds_with_nan['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_filter(self):
        """Filter using isna()."""
        pd_result = self.df_with_nan[self.df_with_nan['a'].isna()]
        ds_result = self.ds_with_nan[self.ds_with_nan['a'].isna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notna_filter(self):
        """Filter using notna()."""
        pd_result = self.df_with_nan[self.df_with_nan['a'].notna()]
        ds_result = self.ds_with_nan[self.ds_with_nan['a'].notna()]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_any(self):
        """dropna with how='any'."""
        pd_result = self.df_with_nan.dropna(how='any')
        ds_result = self.ds_with_nan.dropna(how='any')

        assert len(ds_result) == len(pd_result)

    def test_dropna_all(self):
        """dropna with how='all'."""
        df = pd.DataFrame({'a': [1, None, None], 'b': [2, None, None], 'c': [3, 4, None]})
        ds = DataStore(df)

        pd_result = df.dropna(how='all')
        ds_result = ds.dropna(how='all')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset(self):
        """dropna with subset parameter."""
        pd_result = self.df_with_nan.dropna(subset=['a'])
        ds_result = self.ds_with_nan.dropna(subset=['a'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_scalar(self):
        """fillna with scalar value."""
        pd_result = self.df_with_nan.fillna(0)
        ds_result = self.ds_with_nan.fillna(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict(self):
        """fillna with dict values per column."""
        fill_values = {'a': 0, 'b': -1, 'c': 'missing'}
        pd_result = self.df_with_nan.fillna(fill_values)
        ds_result = self.ds_with_nan.fillna(fill_values)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sum_with_nan(self):
        """Sum column with NaN values."""
        pd_result = self.df_with_nan['a'].sum()
        ds_result = self.ds_with_nan['a'].sum()

        assert ds_result == pd_result

    def test_mean_with_nan(self):
        """Mean column with NaN values."""
        pd_result = self.df_with_nan['a'].mean()
        ds_result = self.ds_with_nan['a'].mean()

        assert ds_result == pd_result

    def test_count_with_nan(self):
        """Count column with NaN values (should exclude NaN)."""
        pd_result = self.df_with_nan['a'].count()
        ds_result = self.ds_with_nan['a'].count()

        assert ds_result == pd_result

    def test_all_nan_column_operations(self):
        """Operations on column that is all NaN."""
        df = pd.DataFrame({'a': [None, None, None]})
        ds = DataStore(df)

        pd_sum = df['a'].sum()
        ds_sum = ds['a'].sum()

        assert pd_sum == 0
        assert ds_sum == 0 or pd.isna(ds_sum)


# ============================================================================
# Section 4: Column Selection Edge Cases
# ============================================================================


class TestColumnSelectionEdgeCases:
    """Tests for column selection edge cases."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': ['x', 'y', 'z']})
        self.ds = DataStore(self.df)

    def test_select_single_column_as_list(self):
        """Select single column using list returns DataFrame."""
        pd_result = self.df[['a']]
        ds_result = self.ds[['a']]

        # DataStore is lazy, just verify results match
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_single_column_as_string(self):
        """Select single column using string returns Series-like."""
        pd_result = self.df['a']
        ds_result = self.ds['a']

        assert_series_equal(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_select_columns_in_different_order(self):
        """Select columns in different order than original."""
        pd_result = self.df[['c', 'a', 'b']]
        ds_result = self.ds[['c', 'a', 'b']]

        assert list(ds_result.columns) == ['c', 'a', 'b']
        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_duplicate_column_rename
    def test_select_duplicate_columns(self):
        """Select same column multiple times."""
        pd_result = self.df[['a', 'a', 'b']]
        ds_result = self.ds[['a', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_all_columns_explicitly(self):
        """Select all columns explicitly by name."""
        pd_result = self.df[['a', 'b', 'c']]
        ds_result = self.ds[['a', 'b', 'c']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_with_space_in_name(self):
        """Select column with space in name."""
        df = pd.DataFrame({'col with space': [1, 2, 3], 'normal': [4, 5, 6]})
        ds = DataStore(df)

        pd_result = df[['col with space']]
        ds_result = ds[['col with space']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_with_special_chars(self):
        """Select column with special characters."""
        df = pd.DataFrame({'col-with-dash': [1, 2, 3], 'col_underscore': [4, 5, 6]})
        ds = DataStore(df)

        pd_result = df[['col-with-dash', 'col_underscore']]
        ds_result = ds[['col-with-dash', 'col_underscore']]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Section 5: Groupby Edge Cases
# ============================================================================


class TestGroupbyEdgeCases:
    """Tests for groupby edge cases."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame(
            {'cat': ['A', 'B', 'A', 'B', 'A', 'C'], 'val': [1, 2, 3, 4, 5, 6], 'val2': [10, 20, 30, 40, 50, 60]}
        )
        self.ds = DataStore(self.df)

    def test_groupby_single_group(self):
        """Groupby where all rows belong to same group."""
        df = pd.DataFrame({'cat': ['A', 'A', 'A'], 'val': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df.groupby('cat')['val'].sum().reset_index()
        ds_result = ds.groupby('cat')['val'].sum()

        assert len(ds_result) == 1

    def test_groupby_all_unique_groups(self):
        """Groupby where each row is its own group."""
        df = pd.DataFrame({'cat': ['A', 'B', 'C'], 'val': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df.groupby('cat')['val'].sum().reset_index()
        ds_result = ds.groupby('cat')['val'].sum()

        assert len(ds_result) == 3

    def test_groupby_multiple_agg_functions(self):
        """Groupby with multiple aggregation functions."""
        pd_result = self.df.groupby('cat')['val'].agg(['sum', 'mean', 'count']).reset_index()
        ds_result = self.ds.groupby('cat')['val'].agg(['sum', 'mean', 'count'])

        assert 'sum' in ds_result.columns or ('val', 'sum') in ds_result.columns
        assert len(ds_result) == 3

    def test_groupby_dict_agg(self):
        """Groupby with dict aggregation (different aggs per column)."""
        pd_result = self.df.groupby('cat').agg({'val': 'sum', 'val2': 'mean'}).reset_index()
        ds_result = self.ds.groupby('cat').agg({'val': 'sum', 'val2': 'mean'})

        assert 'val' in ds_result.columns or ('val', 'sum') in ds_result.columns
        assert 'val2' in ds_result.columns or ('val2', 'mean') in ds_result.columns

    def test_groupby_size(self):
        """Groupby size (count rows per group)."""
        pd_result = self.df.groupby('cat').size().reset_index(name='size')
        ds_result = self.ds.groupby('cat').size()

        assert len(ds_result) == 3

    def test_groupby_first_last(self):
        """Groupby first and last."""
        pd_first = self.df.groupby('cat')['val'].first()
        ds_first = self.ds.groupby('cat')['val'].first()

        assert len(ds_first) == len(pd_first)

        pd_last = self.df.groupby('cat')['val'].last()
        ds_last = self.ds.groupby('cat')['val'].last()

        assert len(ds_last) == len(pd_last)

    def test_groupby_nth_positive(self):
        """Groupby nth with positive index."""
        pd_result = self.df.groupby('cat').nth(0).reset_index()
        ds_result = self.ds.groupby('cat').nth(0)

        assert len(ds_result) == 3

    def test_groupby_on_filtered_data(self):
        """Groupby after filter."""
        pd_filtered = self.df[self.df['val'] > 2]
        pd_result = pd_filtered.groupby('cat')['val'].sum().reset_index()

        ds_filtered = self.ds[self.ds['val'] > 2]
        ds_result = ds_filtered.groupby('cat')['val'].sum()

        assert len(ds_result) == 3


# ============================================================================
# Section 6: Arithmetic Edge Cases
# ============================================================================


class TestArithmeticEdgeCases:
    """Tests for arithmetic edge cases."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50], 'c': [0, 1, 0, 2, 0]})
        self.ds = DataStore(self.df)

    def test_division_by_zero(self):
        """Division by zero should produce inf or NaN."""
        pd_result = self.df['a'] / self.df['c']
        ds_result = self.ds['a'] / self.ds['c']

        assert np.isinf(pd_result.iloc[0]) or pd.isna(pd_result.iloc[0])
        assert np.isinf(ds_result.iloc[0]) or pd.isna(ds_result.iloc[0])

    def test_negative_number_operations(self):
        """Operations with negative numbers."""
        df = pd.DataFrame({'a': [-1, -2, 3], 'b': [1, -2, -3]})
        ds = DataStore(df)

        pd_result = df['a'] * df['b']
        ds_result = ds['a'] * ds['b']

        assert_series_equal(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_float_precision(self):
        """Float precision in operations."""
        df = pd.DataFrame({'a': [0.1, 0.2, 0.3]})
        ds = DataStore(df)

        pd_result = df['a'] + df['a'] + df['a']
        ds_result = ds['a'] + ds['a'] + ds['a']

        assert all(abs(ds_result.iloc[i] - pd_result.iloc[i]) < 0.0001 for i in range(len(pd_result)))

    def test_integer_overflow(self):
        """Large integer operations."""
        df = pd.DataFrame({'a': [10**18, 10**18, 10**18]})
        ds = DataStore(df)

        pd_result = df['a'] + df['a']
        ds_result = ds['a'] + ds['a']

        assert ds_result.iloc[0] == 2 * 10**18

    def test_modulo_operation(self):
        """Modulo operation."""
        pd_result = self.df['b'] % 3
        ds_result = self.ds['b'] % 3

        assert_series_equal(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_power_operation(self):
        """Power/exponent operation."""
        pd_result = self.df['a'] ** 2
        ds_result = self.ds['a'] ** 2

        assert_series_equal(ds_result.reset_index(drop=True), pd_result.reset_index(drop=True))


# ============================================================================
# Section 7: String Operations Edge Cases
# ============================================================================


class TestStringOperationsEdgeCases:
    """Tests for string operations edge cases."""

    def setup_method(self):
        """Create test data with strings."""
        self.df = pd.DataFrame(
            {'text': ['hello', 'WORLD', 'Hello World', '', '  spaces  ', None], 'num': [1, 2, 3, 4, 5, 6]}
        )
        self.ds = DataStore(self.df)

    def test_str_lower(self):
        """String lower case."""
        pd_result = self.df['text'].str.lower()
        ds_result = self.ds['text'].str.lower()

        for i in range(len(pd_result)):
            if pd.notna(pd_result.iloc[i]):
                assert ds_result.iloc[i] == pd_result.iloc[i]

    def test_str_upper(self):
        """String upper case."""
        pd_result = self.df['text'].str.upper()
        ds_result = self.ds['text'].str.upper()

        for i in range(len(pd_result)):
            if pd.notna(pd_result.iloc[i]):
                assert ds_result.iloc[i] == pd_result.iloc[i]

    def test_str_strip(self):
        """String strip whitespace."""
        pd_result = self.df['text'].str.strip()
        ds_result = self.ds['text'].str.strip()

        assert ds_result.iloc[4] == 'spaces'

    def test_str_len(self):
        """String length."""
        pd_result = self.df['text'].str.len()
        ds_result = self.ds['text'].str.len()

        for i in range(len(pd_result)):
            if pd.notna(pd_result.iloc[i]):
                assert ds_result.iloc[i] == pd_result.iloc[i]

    def test_str_contains_simple(self):
        """String contains pattern."""
        pd_result = self.df[self.df['text'].str.contains('o', na=False)]
        ds_result = self.ds[self.ds['text'].str.contains('o', na=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_startswith(self):
        """String startswith."""
        pd_result = self.df[self.df['text'].str.startswith('H', na=False)]
        ds_result = self.ds[self.ds['text'].str.startswith('H', na=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_endswith(self):
        """String endswith."""
        pd_result = self.df[self.df['text'].str.endswith('d', na=False)]
        ds_result = self.ds[self.ds['text'].str.endswith('d', na=False)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_string_handling(self):
        """Operations on empty strings."""
        pd_result = self.df[self.df['text'] == '']
        ds_result = self.ds[self.ds['text'] == '']

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Section 8: Type Conversion Edge Cases
# ============================================================================


class TestTypeConversionEdgeCases:
    """Tests for type conversion edge cases."""

    def test_int_to_float(self):
        """Convert int column to float."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df['a'].astype(float)
        ds_result = ds['a'].astype(float)

        assert ds_result.dtype == float

    def test_float_to_int(self):
        """Convert float column to int."""
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        ds = DataStore(df)

        pd_result = df['a'].astype(int)
        ds_result = ds['a'].astype(int)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_int_to_string(self):
        """Convert int column to string."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore(df)

        pd_result = df['a'].astype(str)
        ds_result = ds['a'].astype(str)

        assert list(ds_result) == ['1', '2', '3']

    def test_string_to_int_valid(self):
        """Convert valid numeric strings to int."""
        df = pd.DataFrame({'a': ['1', '2', '3']})
        ds = DataStore(df)

        pd_result = df['a'].astype(int)
        ds_result = ds['a'].astype(int)

        assert list(ds_result) == [1, 2, 3]

    def test_bool_to_int(self):
        """Convert bool column to int."""
        df = pd.DataFrame({'a': [True, False, True]})
        ds = DataStore(df)

        pd_result = df['a'].astype(int)
        ds_result = ds['a'].astype(int)

        assert list(ds_result) == [1, 0, 1]


# ============================================================================
# Section 9: Slice/Range Edge Cases
# ============================================================================


class TestSliceEdgeCases:
    """Tests for slice/range edge cases."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'a': list(range(10)), 'b': list(range(10, 20))})
        self.ds = DataStore(self.df)

    def test_head_zero(self):
        """head(0) should return empty DataFrame."""
        pd_result = self.df.head(0)
        ds_result = self.ds.head(0)

        assert len(ds_result) == 0

    # Negative head/tail now supported
    def test_head_negative(self):
        """head with negative number (pandas returns all but last n)."""
        pd_result = self.df.head(-2)
        ds_result = self.ds.head(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_zero(self):
        """tail(0) should return empty DataFrame."""
        pd_result = self.df.tail(0)
        ds_result = self.ds.tail(0)

        assert len(ds_result) == 0

    def test_tail_negative(self):
        """tail with negative number (pandas returns all but first n)."""
        pd_result = self.df.tail(-2)
        ds_result = self.ds.tail(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_beyond_length(self):
        """Slice beyond DataFrame length."""
        pd_result = self.df[5:100]
        ds_result = self.ds[5:100]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_with_step(self):
        """Slice with step parameter."""
        pd_result = self.df[::2]
        ds_result = self.ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_negative_start(self):
        """Slice with negative start."""
        pd_result = self.df[-3:]
        ds_result = self.ds[-3:]

        assert_datastore_equals_pandas(ds_result, pd_result)

    # Negative head/tail now supported
    def test_slice_negative_end(self):
        """Slice with negative end."""
        pd_result = self.df[:-3]
        ds_result = self.ds[:-3]

        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================================
# Section 10: Sort Edge Cases
# ============================================================================


class TestSortEdgeCases:
    """Tests for sort edge cases."""

    def setup_method(self):
        """Create test data."""
        self.df = pd.DataFrame({'a': [3, 1, 2, 1, 3], 'b': [1, 2, 3, 4, 5], 'c': ['z', 'a', 'b', 'c', 'd']})
        self.ds = DataStore(self.df)

    def test_sort_ascending(self):
        """Sort ascending."""
        pd_result = self.df.sort_values('a')
        ds_result = self.ds.sort_values('a')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending(self):
        """Sort descending."""
        pd_result = self.df.sort_values('a', ascending=False)
        ds_result = self.ds.sort_values('a', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_multiple_columns(self):
        """Sort by multiple columns."""
        pd_result = self.df.sort_values(['a', 'b'])
        ds_result = self.ds.sort_values(['a', 'b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_mixed_ascending(self):
        """Sort with mixed ascending/descending."""
        pd_result = self.df.sort_values(['a', 'b'], ascending=[True, False])
        ds_result = self.ds.sort_values(['a', 'b'], ascending=[True, False])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_with_null(self):
        """Sort column with NULL values."""
        df = pd.DataFrame({'a': [3, None, 1, None, 2]})
        ds = DataStore(df)

        pd_result = df.sort_values('a')
        ds_result = ds.sort_values('a')

        non_null = ds_result[ds_result['a'].notna()]['a'].tolist()
        assert non_null == sorted(non_null)

    def test_sort_strings(self):
        """Sort string column."""
        pd_result = self.df.sort_values('c')
        ds_result = self.ds.sort_values('c')

        assert_datastore_equals_pandas(ds_result, pd_result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestNegativeHeadTail:
    """Tests for negative n in head() and tail() methods."""

    def setup_method(self):
        """Set up test data."""
        self.data = {'a': list(range(1, 11)), 'b': list('abcdefghij')}
        self.df = pd.DataFrame(self.data)
        self.ds = DataStore(self.data)

    def test_head_negative_basic(self):
        """head(-3) should return all rows except the last 3."""
        pd_result = self.df.head(-3)
        ds_result = self.ds.head(-3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_negative_basic(self):
        """tail(-3) should return all rows except the first 3."""
        pd_result = self.df.tail(-3)
        ds_result = self.ds.tail(-3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_negative_equals_length(self):
        """head(-n) where n equals length should return empty DataFrame."""
        pd_result = self.df.head(-10)
        ds_result = self.ds.head(-10)
        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_tail_negative_equals_length(self):
        """tail(-n) where n equals length should return empty DataFrame."""
        pd_result = self.df.tail(-10)
        ds_result = self.ds.tail(-10)
        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_head_negative_exceeds_length(self):
        """head(-n) where n exceeds length should return empty DataFrame."""
        pd_result = self.df.head(-15)
        ds_result = self.ds.head(-15)
        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_tail_negative_exceeds_length(self):
        """tail(-n) where n exceeds length should return empty DataFrame."""
        pd_result = self.df.tail(-15)
        ds_result = self.ds.tail(-15)
        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_head_negative_one(self):
        """head(-1) should return all rows except the last one."""
        pd_result = self.df.head(-1)
        ds_result = self.ds.head(-1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_negative_one(self):
        """tail(-1) should return all rows except the first one."""
        pd_result = self.df.tail(-1)
        ds_result = self.ds.tail(-1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_positive_still_works(self):
        """Ensure positive head() still works correctly."""
        pd_result = self.df.head(3)
        ds_result = self.ds.head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_positive_still_works(self):
        """Ensure positive tail() still works correctly."""
        pd_result = self.df.tail(3)
        ds_result = self.ds.tail(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_zero(self):
        """head(0) should return empty DataFrame."""
        pd_result = self.df.head(0)
        ds_result = self.ds.head(0)
        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_head_tail_chain(self):
        """Test chaining head and tail operations."""
        # pandas: first get rows 0-6 (head(-3)), then last 3 of those
        pd_result = self.df.head(-3).tail(3)
        ds_result = self.ds.head(-3).tail(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_head_chain(self):
        """Test chaining tail and head operations."""
        # pandas: first get rows 3-9 (tail(-3)), then first 3 of those
        pd_result = self.df.tail(-3).head(3)
        ds_result = self.ds.tail(-3).head(3)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_negative_preserves_index(self):
        """head(-n) should preserve original index values."""
        pd_result = self.df.head(-3)
        ds_result = self.ds.head(-3)
        # Check that index values match
        pd_idx = pd_result.index.tolist()
        ds_idx = get_dataframe(ds_result).index.tolist()
        assert pd_idx == ds_idx, f"Index mismatch: {pd_idx} vs {ds_idx}"

    def test_tail_negative_preserves_index(self):
        """tail(-n) should preserve original index values."""
        pd_result = self.df.tail(-3)
        ds_result = self.ds.tail(-3)
        # Check that index values match
        pd_idx = pd_result.index.tolist()
        ds_idx = get_dataframe(ds_result).index.tolist()
        assert pd_idx == ds_idx, f"Index mismatch: {pd_idx} vs {ds_idx}"
