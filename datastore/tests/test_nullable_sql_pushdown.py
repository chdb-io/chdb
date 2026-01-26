"""
Test Nullable type SQL pushdown optimizations.

This test file verifies that DataStore correctly handles nullable types
(Nullable Boolean, Nullable Int64, etc.) with SQL pushdown where possible.

Exploration context (2026-01-06):
- chDB now correctly supports nullable boolean/integer types
- SQL pushdown is possible for nullable types in many scenarios
- IS NULL / IS NOT NULL conditions work with SQL pushdown

Known dtype differences:
- When filtering removes all NA values, chDB may return non-nullable dtypes
  (e.g., int64 instead of Int64, bool instead of boolean)
- This is acceptable as long as values match

Version-specific notes (Python 3.8 / pandas 2.0.x):
- Nullable boolean type handling in SQL differs in older pandas versions
- chDB type conversion for Bool/String supertype issues in older versions
- These tests may be skipped on older pandas versions
"""

import unittest
import pandas as pd
import numpy as np

from datastore import DataStore
from datastore.lazy_ops import LazyWhere, LazyMask
from tests.test_utils import assert_datastore_equals_pandas
from tests.xfail_markers import (
    pandas_version_nullable_bool_sql,
    pandas_version_nullable_int_dtype,
    skip_if_old_pandas,
)


class TestNullableBooleanSQLPushdown(unittest.TestCase):
    """Test SQL pushdown for nullable boolean columns.

    Note: Nullable boolean SQL pushdown has issues in older pandas + chDB combinations
    where String/Bool type conversion fails. Tests are skipped on pandas < 2.1.
    """

    @skip_if_old_pandas("Nullable boolean SQL pushdown has type conversion issues in pandas < 2.1")
    def test_where_nullable_bool_other_true(self):
        """Test where with nullable bool column, other=True uses SQL pushdown."""
        df = pd.DataFrame(
            {
                'bool_col': pd.array([True, False, pd.NA, True], dtype='boolean'),
            }
        )
        ds = DataStore(df)

        # Verify SQL pushdown is used
        lazy_where = LazyWhere(ds['bool_col']._expr == False, True)
        lazy_where._datastore = ds
        self.assertEqual(lazy_where.execution_engine(), 'chDB')

        # Verify result matches pandas (values match, dtype may differ due to chDB)
        pd_result = df.where(df['bool_col'] == False, True)
        ds_result = ds.where(ds['bool_col'] == False, True)
        # dtype may differ: chDB returns bool, pandas returns boolean
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable boolean SQL pushdown has type conversion issues in pandas < 2.1")
    def test_where_nullable_bool_other_false(self):
        """Test where with nullable bool column, other=False uses SQL pushdown."""
        df = pd.DataFrame(
            {
                'bool_col': pd.array([True, False, pd.NA, True], dtype='boolean'),
            }
        )
        ds = DataStore(df)

        # Verify SQL pushdown is used
        lazy_where = LazyWhere(ds['bool_col']._expr == True, False)
        lazy_where._datastore = ds
        self.assertEqual(lazy_where.execution_engine(), 'chDB')

        # Verify result matches pandas
        pd_result = df.where(df['bool_col'] == True, False)
        ds_result = ds.where(ds['bool_col'] == True, False)
        # dtype may differ: chDB returns bool, pandas returns boolean
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable boolean SQL pushdown has type conversion issues in pandas < 2.1")
    def test_mask_nullable_bool_other_true(self):
        """Test mask with nullable bool column, other=True uses SQL pushdown."""
        df = pd.DataFrame(
            {
                'bool_col': pd.array([True, False, pd.NA, True], dtype='boolean'),
            }
        )
        ds = DataStore(df)

        # Verify SQL pushdown is used
        lazy_mask = LazyMask(ds['bool_col']._expr == True, True)
        lazy_mask._datastore = ds
        self.assertEqual(lazy_mask.execution_engine(), 'chDB')

        # Verify result matches pandas
        pd_result = df.mask(df['bool_col'] == True, True)
        ds_result = ds.mask(ds['bool_col'] == True, True)
        # dtype may differ: chDB returns bool, pandas returns boolean
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable boolean SQL pushdown has type conversion issues in pandas < 2.1")
    def test_filter_nullable_bool(self):
        """Test filter on nullable boolean column."""
        df = pd.DataFrame({'bool_col': pd.array([True, False, pd.NA, True], dtype='boolean'), 'val': [1, 2, 3, 4]})
        ds = DataStore(df)

        # Filter where bool_col is True
        pd_result = df[df['bool_col'] == True]
        ds_result = ds[ds['bool_col'] == True]
        # dtype may differ: filtering removes NA, chDB returns non-nullable
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable boolean SQL pushdown has type conversion issues in pandas < 2.1")
    def test_filter_nullable_bool_with_na(self):
        """Test that NA values in bool column don't match True or False."""
        df = pd.DataFrame({'bool_col': pd.array([True, False, pd.NA, True], dtype='boolean'), 'val': [1, 2, 3, 4]})
        ds = DataStore(df)

        # Row with NA should not appear when filtering for True
        ds_result = ds[ds['bool_col'] == True]
        self.assertEqual(len(ds_result), 2)  # Only rows 0 and 3

        # Row with NA should not appear when filtering for False
        ds_result = ds[ds['bool_col'] == False]
        self.assertEqual(len(ds_result), 1)  # Only row 1


class TestNullableInt64SQLPushdown(unittest.TestCase):
    """Test SQL pushdown for nullable Int64 columns.

    Note: Nullable Int64 dtype preservation differs between pandas versions.
    In older pandas + chDB combinations, Int64 may be converted to float64.
    Tests that require exact dtype match are skipped on pandas < 2.1.
    """

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_where_nullable_int64_other_zero(self):
        """Test where with nullable Int64 column, other=0 uses SQL pushdown."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        # Verify SQL pushdown is used
        lazy_where = LazyWhere(ds['int_col']._expr > 2, 0)
        lazy_where._datastore = ds
        self.assertEqual(lazy_where.execution_engine(), 'chDB')

        # Verify result matches pandas
        pd_result = df.where(df['int_col'] > 2, 0)
        ds_result = ds.where(ds['int_col'] > 2, 0)
        # dtype may differ: chDB returns int64, pandas returns Int64
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_where_nullable_int64_other_none(self):
        """Test where with nullable Int64 column, other=None uses SQL pushdown."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        # Verify SQL pushdown is used
        lazy_where = LazyWhere(ds['int_col']._expr > 2, None)
        lazy_where._datastore = ds
        self.assertEqual(lazy_where.execution_engine(), 'chDB')

        # Verify result matches pandas
        pd_result = df.where(df['int_col'] > 2, None)
        ds_result = ds.where(ds['int_col'] > 2, None)
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_filter_nullable_int64_comparison(self):
        """Test filter with comparison on nullable Int64 column."""
        df = pd.DataFrame({'int_col': pd.array([1, 2, pd.NA, 4], dtype='Int64'), 'val': [10, 20, 30, 40]})
        ds = DataStore(df)

        # Filter where int_col > 2
        pd_result = df[df['int_col'] > 2]
        ds_result = ds[ds['int_col'] > 2]
        # dtype may differ: filtering removes NA, chDB returns non-nullable
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_nullable_int64_na_excluded(self):
        """Test that NA values in Int64 column are excluded from comparisons."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        # NA row should not appear (NA > 2 is NA, not True)
        ds_result = ds[ds['int_col'] > 2]
        self.assertEqual(len(ds_result), 1)  # Only row 3 (value 4)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_groupby_sum_nullable_int64(self):
        """Test groupby sum correctly handles NA values in nullable Int64."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'int_col': pd.array([1, pd.NA, 3, 4], dtype='Int64')})
        ds = DataStore(df)

        pd_result = df.groupby('group')['int_col'].sum()
        ds_result = ds.groupby('group')['int_col'].sum()

        # Both should correctly ignore NA: A=1, B=7
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_mean_nullable_int64(self):
        """Test groupby mean correctly handles NA values in nullable Int64."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'int_col': pd.array([1, pd.NA, 3, 4], dtype='Int64')})
        ds = DataStore(df)

        pd_result = df.groupby('group')['int_col'].mean()
        ds_result = ds.groupby('group')['int_col'].mean()

        # Both should correctly ignore NA: A=1.0 (only 1 value), B=3.5
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_groupby_count_nullable_int64(self):
        """Test groupby count correctly handles NA values in nullable Int64."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'int_col': pd.array([1, pd.NA, 3, 4], dtype='Int64')})
        ds = DataStore(df)

        pd_result = df.groupby('group')['int_col'].count()
        ds_result = ds.groupby('group')['int_col'].count()

        # Both should exclude NA: A=1, B=2
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIsNullIsNotNullConditions(unittest.TestCase):
    """Test IS NULL / IS NOT NULL conditions with SQL pushdown.

    Note: These tests use nullable Int64 columns. When filtering removes all NA values,
    the dtype may differ between pandas (Int64) and chDB (int64/float64).
    Tests that require exact dtype match are skipped on pandas < 2.1.
    """

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_isnull_filter(self):
        """Test isnull() filter works correctly."""
        df = pd.DataFrame({'int_col': pd.array([1, 2, pd.NA, 4, pd.NA], dtype='Int64'), 'val': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df[df['int_col'].isnull()]
        ds_result = ds[ds['int_col'].isnull()]
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_notnull_filter(self):
        """Test notnull() filter works correctly."""
        df = pd.DataFrame({'int_col': pd.array([1, 2, pd.NA, 4, pd.NA], dtype='Int64'), 'val': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        pd_result = df[df['int_col'].notnull()]
        ds_result = ds[ds['int_col'].notnull()]
        # dtype may differ: filtering for notnull removes NA, chDB returns non-nullable
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_isna_alias(self):
        """Test isna() is alias for isnull()."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        pd_result = df[df['int_col'].isna()]
        ds_result = ds[ds['int_col'].isna()]
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_notna_alias(self):
        """Test notna() is alias for notnull()."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, pd.NA, 4], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        pd_result = df[df['int_col'].notna()]
        ds_result = ds[ds['int_col'].notna()]
        # dtype may differ: filtering for notna removes NA, chDB returns non-nullable
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_isnull_and_comparison(self):
        """Test isnull() combined with comparison filter."""
        df = pd.DataFrame({'int_col': pd.array([1, 2, pd.NA, 4, pd.NA], dtype='Int64'), 'val': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        # NOT NULL AND > 2
        pd_result = df[df['int_col'].notnull() & (df['int_col'] > 2)]
        ds_result = ds[ds['int_col'].notnull() & (ds['int_col'] > 2)]
        # dtype may differ: filtering removes NA, chDB returns non-nullable
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_isnull_or_condition(self):
        """Test isnull() combined with OR condition."""
        df = pd.DataFrame({'int_col': pd.array([1, 2, pd.NA, 4, pd.NA], dtype='Int64'), 'val': [10, 20, 30, 40, 50]})
        ds = DataStore(df)

        # IS NULL OR val > 40
        pd_result = df[df['int_col'].isnull() | (df['val'] > 40)]
        ds_result = ds[ds['int_col'].isnull() | (ds['val'] > 40)]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullableFloat64SQLPushdown(unittest.TestCase):
    """Test SQL pushdown for nullable Float64 columns."""

    def test_where_nullable_float64_other_zero(self):
        """Test where with nullable Float64 column, other=0.0 uses SQL pushdown."""
        df = pd.DataFrame(
            {
                'float_col': pd.array([1.5, 2.5, pd.NA, 4.5], dtype='Float64'),
            }
        )
        ds = DataStore(df)

        # Verify result matches pandas
        pd_result = df.where(df['float_col'] > 2.0, 0.0)
        ds_result = ds.where(ds['float_col'] > 2.0, 0.0)
        # dtype may differ: chDB returns float64, pandas returns Float64
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_nullable_float64(self):
        """Test filter on nullable Float64 column."""
        df = pd.DataFrame({'float_col': pd.array([1.5, 2.5, pd.NA, 4.5], dtype='Float64'), 'val': [10, 20, 30, 40]})
        ds = DataStore(df)

        pd_result = df[df['float_col'] > 2.0]
        ds_result = ds[ds['float_col'] > 2.0]
        # dtype may differ: filtering removes NA, chDB returns non-nullable
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullableChainedOperations(unittest.TestCase):
    """Test chained operations with nullable types.

    Note: These tests involve nullable Int64 columns in complex chains.
    Dtype preservation differs between pandas versions.
    """

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_filter_then_agg_nullable_int64(self):
        """Test filter followed by aggregation on nullable Int64."""
        df = pd.DataFrame({'group': ['A', 'A', 'B', 'B', 'A'], 'int_col': pd.array([1, pd.NA, 3, 4, 5], dtype='Int64')})
        ds = DataStore(df)

        # Filter out NA values, then groupby sum
        pd_result = df[df['int_col'].notnull()].groupby('group')['int_col'].sum()
        ds_result = ds[ds['int_col'].notnull()].groupby('group')['int_col'].sum()
        assert_datastore_equals_pandas(ds_result, pd_result)

    @skip_if_old_pandas("Nullable Int64 dtype preservation differs in pandas < 2.1")
    def test_filter_by_comparison_then_isnull_check(self):
        """Test filter by comparison then check for nulls."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, pd.NA, 4, pd.NA], dtype='Int64'),
                'other_col': pd.array([pd.NA, 10, pd.NA, 20, pd.NA], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        # Filter rows where int_col > 0 (excludes NA)
        # Then check which have null other_col
        pd_filtered = df[df['int_col'] > 0]
        ds_filtered = ds[ds['int_col'] > 0]

        pd_result = pd_filtered[pd_filtered['other_col'].isnull()]
        ds_result = ds_filtered[ds_filtered['other_col'].isnull()]
        # dtype may differ: filtering removes NA from int_col
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNullableEdgeCases(unittest.TestCase):
    """Test edge cases with nullable types."""

    def test_all_na_column(self):
        """Test column with all NA values."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([pd.NA, pd.NA, pd.NA], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        # All should be null
        ds_result = ds[ds['int_col'].isnull()]
        self.assertEqual(len(ds_result), 3)

        # None should be not null
        ds_result = ds[ds['int_col'].notnull()]
        self.assertEqual(len(ds_result), 0)

    def test_no_na_values(self):
        """Test column with no NA values."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([1, 2, 3], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        # None should be null
        ds_result = ds[ds['int_col'].isnull()]
        self.assertEqual(len(ds_result), 0)

        # All should be not null
        ds_result = ds[ds['int_col'].notnull()]
        self.assertEqual(len(ds_result), 3)

    def test_empty_dataframe_nullable(self):
        """Test empty DataFrame with nullable column."""
        df = pd.DataFrame(
            {
                'int_col': pd.array([], dtype='Int64'),
            }
        )
        ds = DataStore(df)

        ds_result = ds[ds['int_col'].isnull()]
        self.assertEqual(len(ds_result), 0)


if __name__ == '__main__':
    unittest.main()
