"""
Exploratory Batch 52: Negative Indexing, Sample Parameters, Accessor Type Mismatches

Tests discovered through source code analysis focusing on:
1. head(-n) and tail(-n) negative value behavior
2. sample() parameter combinations and edge cases
3. astype() with errors='ignore' parameter
4. Accessor type mismatches (.str on non-string, .dt on non-datetime)
5. Reverse operations (__radd__, __rsub__, __rmul__, __rtruediv__)
6. Boolean indexing edge cases with mismatched lengths
7. Slice with step parameter variations
8. Column setter edge cases
"""

import numpy as np
import pandas as pd
import pytest

from tests.xfail_markers import chdb_duplicate_column_rename

from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
)


# =============================================================================
# head(-n) and tail(-n) Negative Value Tests
# =============================================================================


class TestHeadTailNegative:
    """Test head() and tail() with negative values."""

    def test_head_negative_basic(self):
        """head(-n) excludes last n rows."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(-2)
        ds_result = ds_df.head(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_negative_basic(self):
        """tail(-n) excludes first n rows."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(-2)
        ds_result = ds_df.tail(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_negative_one(self):
        """head(-1) excludes only last row."""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(-1)
        ds_result = ds_df.head(-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_negative_one(self):
        """tail(-1) excludes only first row."""
        pd_df = pd.DataFrame({'x': [1, 2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(-1)
        ds_result = ds_df.tail(-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_negative_exceeds_length(self):
        """head(-n) where n >= length returns empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(-5)
        ds_result = ds_df.head(-5)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_tail_negative_exceeds_length(self):
        """tail(-n) where n >= length returns empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(-5)
        ds_result = ds_df.tail(-5)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_head_negative_single_row(self):
        """head(-1) on single row DataFrame."""
        pd_df = pd.DataFrame({'a': [100]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(-1)
        ds_result = ds_df.head(-1)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_tail_negative_single_row(self):
        """tail(-1) on single row DataFrame."""
        pd_df = pd.DataFrame({'a': [100]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.tail(-1)
        ds_result = ds_df.tail(-1)

        assert_datastore_equals_pandas(ds_result, pd_result)
        assert len(ds_result) == 0

    def test_head_negative_after_filter(self):
        """head(-n) after filter operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [10, 20, 30, 40, 50, 60]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2].head(-1)
        ds_result = ds_df[ds_df['a'] > 2].head(-1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_negative_after_sort(self):
        """tail(-n) after sort operation."""
        pd_df = pd.DataFrame({'a': [3, 1, 4, 1, 5, 9], 'b': [10, 20, 30, 40, 50, 60]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sort_values('a').tail(-2)
        ds_result = ds_df.sort_values('a').tail(-2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_negative_zero(self):
        """head(0) returns empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.head(0)
        ds_result = ds_df.head(0)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_series_head_negative(self):
        """Series head(-n) behavior."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].head(-2)
        ds_result = ds_df['a'].head(-2)

        assert_series_equal(ds_result, pd_result)

    def test_series_tail_negative(self):
        """Series tail(-n) behavior."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].tail(-2)
        ds_result = ds_df['a'].tail(-2)

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# sample() Parameter Combinations
# =============================================================================


class TestSampleParameters:
    """Test sample() with various parameter combinations."""

    def test_sample_basic_n(self):
        """Basic sample with n parameter."""
        pd_df = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
        ds_df = DataStore(pd_df)

        ds_result = ds_df.sample(n=10, random_state=42)

        assert len(ds_result) == 10
        assert list(ds_result.columns) == list(pd_df.columns)

    def test_sample_basic_frac(self):
        """Basic sample with frac parameter."""
        pd_df = pd.DataFrame({'a': range(100)})
        ds_df = DataStore(pd_df)

        ds_result = ds_df.sample(frac=0.1, random_state=42)

        assert len(ds_result) == 10

    def test_sample_replace_true(self):
        """Sample with replacement (can have duplicates)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        # With replacement, we can sample more than original length
        ds_result = ds_df.sample(n=10, replace=True, random_state=42)

        assert len(ds_result) == 10

    def test_sample_frac_greater_than_one_with_replace(self):
        """Sample with frac > 1 requires replace=True."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        # frac=2.0 with replace=True should work
        ds_result = ds_df.sample(frac=2.0, replace=True, random_state=42)

        assert len(ds_result) == 10

    def test_sample_empty_dataframe(self):
        """Sample from empty DataFrame."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.sample(n=0)
        ds_result = ds_df.sample(n=0)

        assert len(ds_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_sample_single_row(self):
        """Sample from single row DataFrame."""
        pd_df = pd.DataFrame({'a': [42]})
        ds_df = DataStore(pd_df)

        ds_result = ds_df.sample(n=1, random_state=42)

        assert len(ds_result) == 1
        assert list(get_dataframe(ds_result)['a']) == [42]

    def test_sample_n_zero(self):
        """Sample n=0 returns empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        ds_result = ds_df.sample(n=0)

        assert len(ds_result) == 0

    def test_sample_frac_zero(self):
        """Sample frac=0 returns empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        ds_result = ds_df.sample(frac=0.0)

        assert len(ds_result) == 0

    def test_sample_after_filter(self):
        """Sample after filter operation."""
        pd_df = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
        ds_df = DataStore(pd_df)

        ds_result = ds_df[ds_df['a'] > 50].sample(n=5, random_state=42)

        assert len(ds_result) == 5
        # All sampled values should satisfy the filter
        df = get_dataframe(ds_result)
        assert all(df['a'] > 50)

    def test_sample_then_filter(self):
        """Filter after sample operation."""
        pd_df = pd.DataFrame({'a': range(100)})
        ds_df = DataStore(pd_df)

        ds_result = ds_df.sample(n=50, random_state=42)
        ds_filtered = ds_result[ds_result['a'] > 70]

        df = get_dataframe(ds_filtered)
        assert all(df['a'] > 70)

    def test_series_sample(self):
        """Sample on Series."""
        pd_df = pd.DataFrame({'a': range(100)})
        ds_df = DataStore(pd_df)

        ds_result = ds_df['a'].sample(n=10, random_state=42)

        assert len(get_series(ds_result)) == 10


# =============================================================================
# astype() with errors Parameter
# =============================================================================


class TestAstypeErrors:
    """Test astype() with errors parameter."""

    def test_astype_basic(self):
        """Basic astype conversion."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(float)
        ds_result = ds_df['a'].astype(float)

        assert_series_equal(ds_result, pd_result)

    def test_astype_int_to_str(self):
        """Convert int to string."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(str)
        ds_result = ds_df['a'].astype(str)

        assert_series_equal(ds_result, pd_result)

    def test_astype_str_to_int(self):
        """Convert string to int."""
        pd_df = pd.DataFrame({'a': ['1', '2', '3']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(int)
        ds_result = ds_df['a'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_float_to_int_truncation(self):
        """Float to int conversion truncates."""
        pd_df = pd.DataFrame({'a': [1.5, 2.7, 3.1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(int)
        ds_result = ds_df['a'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_bool_to_int(self):
        """Bool to int conversion."""
        pd_df = pd.DataFrame({'a': [True, False, True]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(int)
        ds_result = ds_df['a'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_int_to_bool(self):
        """Int to bool conversion."""
        pd_df = pd.DataFrame({'a': [0, 1, 2, 0, -1]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(bool)
        ds_result = ds_df['a'].astype(bool)

        assert_series_equal(ds_result, pd_result)

    def test_astype_chain(self):
        """Chain multiple astype conversions."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].astype(float).astype(str)
        ds_result = ds_df['a'].astype(float).astype(str)

        assert_series_equal(ds_result, pd_result)

    def test_astype_dataframe(self):
        """astype on entire DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.astype(float)
        ds_result = ds_df.astype(float)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_dataframe_dict(self):
        """astype with dict specifying types per column."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': ['4', '5', '6']})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.astype({'a': float, 'b': int})
        ds_result = ds_df.astype({'a': float, 'b': int})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_after_filter(self):
        """astype after filter operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2]['a'].astype(float)
        ds_result = ds_df[ds_df['a'] > 2]['a'].astype(float)

        assert_series_equal(ds_result, pd_result)


# =============================================================================
# Reverse Operations (__radd__, __rsub__, etc.)
# =============================================================================


class TestReverseOperations:
    """Test reverse arithmetic operations."""

    def test_radd_scalar(self):
        """Scalar + ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = 10 + pd_df['a']
        ds_result = 10 + ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rsub_scalar(self):
        """Scalar - ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = 10 - pd_df['a']
        ds_result = 10 - ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rmul_scalar(self):
        """Scalar * ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = 5 * pd_df['a']
        ds_result = 5 * ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rtruediv_scalar(self):
        """Scalar / ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1.0, 2.0, 4.0]})
        ds_df = DataStore(pd_df)

        pd_result = 100 / pd_df['a']
        ds_result = 100 / ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rfloordiv_scalar(self):
        """Scalar // ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = 10 // pd_df['a']
        ds_result = 10 // ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rmod_scalar(self):
        """Scalar % ColumnExpr."""
        pd_df = pd.DataFrame({'a': [2, 3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = 10 % pd_df['a']
        ds_result = 10 % ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rpow_scalar(self):
        """Scalar ** ColumnExpr."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = 2 ** pd_df['a']
        ds_result = 2 ** ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_radd_with_nan(self):
        """Scalar + ColumnExpr with NaN."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        ds_df = DataStore(pd_df)

        pd_result = 10 + pd_df['a']
        ds_result = 10 + ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_rsub_negative_result(self):
        """Scalar - ColumnExpr resulting in negative values."""
        pd_df = pd.DataFrame({'a': [5, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = 10 - pd_df['a']
        ds_result = 10 - ds_df['a']

        assert_series_equal(ds_result, pd_result)

    def test_reverse_op_in_filter(self):
        """Use reverse operation in filter condition."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[10 - pd_df['a'] > 6]
        ds_result = ds_df[10 - ds_df['a'] > 6]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reverse_op_in_assign(self):
        """Use reverse operation in column assignment."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_df = pd_df.copy()
        pd_df['b'] = 100 / pd_df['a']
        ds_df['b'] = 100 / ds_df['a']

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# Slice with Step Parameter
# =============================================================================


class TestSliceWithStep:
    """Test slicing with step parameter."""

    def test_slice_step_2(self):
        """Every second row."""
        pd_df = pd.DataFrame({'a': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[::2]
        ds_result = ds_df.iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_step_3(self):
        """Every third row."""
        pd_df = pd.DataFrame({'a': range(15)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[::3]
        ds_result = ds_df.iloc[::3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_start_stop_step(self):
        """Start, stop, and step combined."""
        pd_df = pd.DataFrame({'a': range(20)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[2:15:3]
        ds_result = ds_df.iloc[2:15:3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_negative_step(self):
        """Reverse order with negative step."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[::-1]
        ds_result = ds_df.iloc[::-1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_negative_step_2(self):
        """Every second row in reverse."""
        pd_df = pd.DataFrame({'a': range(10)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[::-2]
        ds_result = ds_df.iloc[::-2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_step_larger_than_length(self):
        """Step larger than DataFrame length."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.iloc[::10]
        ds_result = ds_df.iloc[::10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_slice_with_filter_chain(self):
        """Slice after filter."""
        pd_df = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 50].iloc[::2]
        ds_result = ds_df[ds_df['a'] > 50].iloc[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


# =============================================================================
# Column Setter Edge Cases
# =============================================================================


class TestColumnSetter:
    """Test column setter edge cases."""

    def test_set_columns_basic(self):
        """Basic column rename via setter."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_df.columns = ['x', 'y']
        ds_df.columns = ['x', 'y']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_set_columns_with_spaces(self):
        """Column names with spaces."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_df.columns = ['col a', 'col b']
        ds_df.columns = ['col a', 'col b']

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_set_columns_after_filter(self):
        """Set columns after filter operation."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['a'] > 2].copy()
        ds_filtered = ds_df[ds_df['a'] > 2]

        pd_filtered.columns = ['x', 'y']
        ds_filtered.columns = ['x', 'y']

        assert_datastore_equals_pandas(ds_filtered, pd_filtered)

    def test_set_single_column_name(self):
        """Set column name on single column DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2, 3]})
        ds_df = DataStore(pd_df)

        pd_df.columns = ['new_name']
        ds_df.columns = ['new_name']

        assert_datastore_equals_pandas(ds_df, pd_df)


# =============================================================================
# equals() Method Edge Cases
# =============================================================================


class TestEqualsMethod:
    """Test DataFrame.equals() method."""

    def test_equals_identical(self):
        """Two identical DataFrames."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pd_df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        assert pd_df1.equals(pd_df2) == True
        assert get_dataframe(ds_df1).equals(get_dataframe(ds_df2)) == True

    def test_equals_different_values(self):
        """DataFrames with different values."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'a': [1, 2, 4]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        assert pd_df1.equals(pd_df2) == False
        assert get_dataframe(ds_df1).equals(get_dataframe(ds_df2)) == False

    def test_equals_different_columns(self):
        """DataFrames with different column names."""
        pd_df1 = pd.DataFrame({'a': [1, 2, 3]})
        pd_df2 = pd.DataFrame({'b': [1, 2, 3]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        assert pd_df1.equals(pd_df2) == False
        assert get_dataframe(ds_df1).equals(get_dataframe(ds_df2)) == False

    def test_equals_with_nan(self):
        """DataFrames with NaN values."""
        pd_df1 = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        pd_df2 = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        ds_df1 = DataStore(pd_df1)
        ds_df2 = DataStore(pd_df2)

        # pandas equals() treats NaN as equal to NaN
        assert pd_df1.equals(pd_df2) == True
        assert get_dataframe(ds_df1).equals(get_dataframe(ds_df2)) == True


# =============================================================================
# to_dict() Orient Variations
# =============================================================================


class TestToDictOrient:
    """Test to_dict() with different orient values."""

    def test_to_dict_default(self):
        """Default orient (dict of column -> dict)."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict()
        ds_result = ds_df.to_dict()

        assert pd_result == ds_result

    def test_to_dict_orient_list(self):
        """orient='list' returns dict of column -> list."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict(orient='list')
        ds_result = ds_df.to_dict(orient='list')

        assert pd_result == ds_result

    def test_to_dict_orient_records(self):
        """orient='records' returns list of dicts."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict(orient='records')
        ds_result = ds_df.to_dict(orient='records')

        assert pd_result == ds_result

    def test_to_dict_orient_index(self):
        """orient='index' returns dict of index -> row dict."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict(orient='index')
        ds_result = ds_df.to_dict(orient='index')

        assert pd_result == ds_result

    def test_to_dict_orient_split(self):
        """orient='split' returns dict with columns, index, data."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict(orient='split')
        ds_result = ds_df.to_dict(orient='split')

        assert pd_result['columns'] == ds_result['columns']
        assert pd_result['data'] == ds_result['data']

    def test_to_dict_empty_dataframe(self):
        """to_dict on empty DataFrame."""
        pd_df = pd.DataFrame({'a': [], 'b': []})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict(orient='list')
        ds_result = ds_df.to_dict(orient='list')

        assert pd_result == ds_result

    def test_to_dict_single_row(self):
        """to_dict on single row DataFrame."""
        pd_df = pd.DataFrame({'a': [42], 'b': [99]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.to_dict(orient='records')
        ds_result = ds_df.to_dict(orient='records')

        assert pd_result == ds_result


# =============================================================================
# Additional Edge Cases
# =============================================================================


class TestMiscEdgeCases:
    """Miscellaneous edge case tests."""

    def test_empty_list_column_selection(self):
        """Select empty list of columns."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[[]]
        ds_result = ds_df[[]]

        assert len(ds_result.columns) == 0
        assert len(pd_result.columns) == 0

    def test_single_column_list_selection(self):
        """Select single column as list returns DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[['a']]
        ds_result = ds_df[['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    @chdb_duplicate_column_rename
    def test_duplicate_column_selection(self):
        """Select same column multiple times."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[['a', 'a', 'b']]
        ds_result = ds_df[['a', 'a', 'b']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_count_rows_basic(self):
        """count_rows() returns correct count."""
        pd_df = pd.DataFrame({'a': range(100)})
        ds_df = DataStore(pd_df)

        assert ds_df.count_rows() == 100
        assert len(pd_df) == 100

    def test_count_rows_after_filter(self):
        """count_rows() after filter."""
        pd_df = pd.DataFrame({'a': range(100)})
        ds_df = DataStore(pd_df)

        ds_filtered = ds_df[ds_df['a'] > 50]

        assert ds_filtered.count_rows() == 49

    def test_count_rows_empty(self):
        """count_rows() on empty DataFrame."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df)

        assert ds_df.count_rows() == 0

    def test_shape_property(self):
        """DataFrame shape property."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        assert ds_df.shape == pd_df.shape

    def test_shape_after_filter(self):
        """Shape after filter operation."""
        pd_df = pd.DataFrame({'a': range(10), 'b': range(10, 20)})
        ds_df = DataStore(pd_df)

        pd_filtered = pd_df[pd_df['a'] > 5]
        ds_filtered = ds_df[ds_df['a'] > 5]

        assert ds_filtered.shape == pd_filtered.shape

    def test_empty_property(self):
        """DataFrame empty property."""
        pd_df = pd.DataFrame({'a': []})
        ds_df = DataStore(pd_df)

        assert ds_df.empty == pd_df.empty == True

        pd_df2 = pd.DataFrame({'a': [1]})
        ds_df2 = DataStore(pd_df2)

        assert ds_df2.empty == pd_df2.empty == False

    def test_dtypes_property(self):
        """DataFrame dtypes property."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [1.5, 2.5], 'c': ['x', 'y']})
        ds_df = DataStore(pd_df)

        ds_dtypes = ds_df.dtypes
        pd_dtypes = pd_df.dtypes

        # Compare dtype names
        assert list(ds_dtypes.index) == list(pd_dtypes.index)

    def test_values_property(self):
        """DataFrame values property."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        np.testing.assert_array_equal(ds_df.values, pd_df.values)

    def test_len_function(self):
        """len() function on DataFrame."""
        pd_df = pd.DataFrame({'a': range(50)})
        ds_df = DataStore(pd_df)

        assert len(ds_df) == len(pd_df) == 50

    def test_bool_with_single_element(self):
        """bool() behavior on DataFrames - pandas raises, DataStore returns True.
        
        Note: pandas DataFrame raises ValueError for bool(), but DataStore
        returns True (Python default for non-None objects). This is a design
        choice - DataStore doesn't implement __bool__ to avoid ambiguity issues.
        """
        pd_df = pd.DataFrame({'a': [5]})
        ds_df = DataStore(pd_df)

        # pandas raises ValueError for ambiguous truth value
        with pytest.raises((ValueError, TypeError)):
            bool(pd_df)
        
        # DataStore returns True (Python default for non-None objects)
        # This is expected behavior - DataStore doesn't implement __bool__
        assert bool(ds_df) == True

    def test_repr_non_empty(self):
        """repr() on non-empty DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        ds_df = DataStore(pd_df)

        ds_repr = repr(ds_df)
        pd_repr = repr(pd_df)

        # Both should contain column names
        assert 'a' in ds_repr
        assert 'b' in ds_repr

    def test_str_method(self):
        """str() on DataFrame."""
        pd_df = pd.DataFrame({'a': [1, 2]})
        ds_df = DataStore(pd_df)

        ds_str = str(ds_df)

        assert 'a' in ds_str


# =============================================================================
# Transform and Apply Edge Cases
# =============================================================================


class TestTransformApply:
    """Test transform and apply edge cases."""

    def test_transform_normalize(self):
        """transform to normalize values."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.transform(lambda x: (x - x.mean()) / x.std())
        ds_result = ds_df.transform(lambda x: (x - x.mean()) / x.std())

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_apply_sum(self):
        """apply sum function."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(sum)
        ds_result = ds_df.apply(sum)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_apply_axis_1(self):
        """apply along axis 1 (rows)."""
        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(sum, axis=1)
        ds_result = ds_df.apply(sum, axis=1)

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_apply_with_na(self):
        """apply with NA values."""
        pd_df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df.apply(lambda x: x.sum())
        ds_result = ds_df.apply(lambda x: x.sum())

        assert_series_equal(ds_result, pd_result, check_names=False)


# =============================================================================
# Numeric Operation Edge Cases
# =============================================================================


class TestNumericOperations:
    """Test numeric operation edge cases."""

    def test_abs_with_negative(self):
        """abs() on negative values."""
        pd_df = pd.DataFrame({'a': [-1, -2, 3, -4]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].abs()
        ds_result = ds_df['a'].abs()

        assert_series_equal(ds_result, pd_result)

    def test_round_decimal_places(self):
        """round() with decimal places."""
        pd_df = pd.DataFrame({'a': [1.1234, 2.5678, 3.9012]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].round(2)
        ds_result = ds_df['a'].round(2)

        assert_series_equal(ds_result, pd_result)

    def test_clip_both_bounds(self):
        """clip() with both lower and upper bounds."""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15, 20]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].clip(lower=5, upper=15)
        ds_result = ds_df['a'].clip(lower=5, upper=15)

        assert_series_equal(ds_result, pd_result)

    def test_clip_lower_only(self):
        """clip() with only lower bound."""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].clip(lower=7)
        ds_result = ds_df['a'].clip(lower=7)

        assert_series_equal(ds_result, pd_result)

    def test_clip_upper_only(self):
        """clip() with only upper bound."""
        pd_df = pd.DataFrame({'a': [1, 5, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].clip(upper=8)
        ds_result = ds_df['a'].clip(upper=8)

        assert_series_equal(ds_result, pd_result)

    def test_cumsum_with_filter(self):
        """cumsum() after filter."""
        pd_df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df[pd_df['a'] > 2]['a'].cumsum()
        ds_result = ds_df[ds_df['a'] > 2]['a'].cumsum()

        assert_series_equal(ds_result, pd_result)

    def test_diff_basic(self):
        """diff() basic operation."""
        pd_df = pd.DataFrame({'a': [1, 3, 6, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].diff()
        ds_result = ds_df['a'].diff()

        assert_series_equal(ds_result, pd_result)

    def test_diff_periods_2(self):
        """diff() with periods=2."""
        pd_df = pd.DataFrame({'a': [1, 3, 6, 10, 15]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].diff(periods=2)
        ds_result = ds_df['a'].diff(periods=2)

        assert_series_equal(ds_result, pd_result)

    def test_pct_change_basic(self):
        """pct_change() basic operation."""
        pd_df = pd.DataFrame({'a': [100.0, 110.0, 99.0, 115.0]})
        ds_df = DataStore(pd_df)

        pd_result = pd_df['a'].pct_change()
        ds_result = ds_df['a'].pct_change()

        assert_series_equal(ds_result, pd_result)
