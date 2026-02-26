"""
Exploratory Batch 40: Eval/Query Chains + Dtype Coercion + Update/Assign Expressions

Focus areas:
1. eval() and query() with lazy chains (filter, groupby, sort)
2. clip() + between() + where() combinations
3. astype() in chain operations - dtype coercion edge cases
4. assign() with complex expressions and dependencies
5. update() operations with various scenarios

Discovery method: Architecture-based exploratory testing
"""

import numpy as np
import pandas as pd
import pytest
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_frame_equal, assert_series_equal, get_dataframe, get_series


class TestEvalQueryChains:
    """Test eval() and query() methods with lazy operation chains."""

    @pytest.fixture
    def ds(self):
        data = {
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'c': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        }
        return DataStore(data), pd.DataFrame(data)

    def test_query_then_filter(self, ds):
        """Query followed by boolean filter."""
        ds_df, pd_df = ds

        pd_result = pd_df.query('a > 3')[pd_df.query('a > 3')['b'] < 80]
        ds_result = ds_df.query('a > 3')[ds_df.query('a > 3')['b'] < 80]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_query(self, ds):
        """Boolean filter followed by query."""
        ds_df, pd_df = ds

        pd_result = pd_df[pd_df['a'] > 2].query('b < 70')
        ds_result = ds_df[ds_df['a'] > 2].query('b < 70')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_then_groupby_column_name(self, ds):
        """Query followed by groupby on a column name."""
        ds_df, pd_df = ds

        pd_queried = pd_df.query('a > 2')
        pd_result = pd_queried.groupby('a')['b'].sum().reset_index()

        ds_queried = ds_df.query('a > 2')
        ds_result = ds_queried.groupby('a')['b'].sum().reset_index()

        # Sort for comparison (groupby order not guaranteed)
        pd_result = pd_result.sort_values('a').reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values('a').reset_index(drop=True)
        assert_frame_equal(ds_result_df, pd_result)

    def test_query_then_sort_head(self, ds):
        """Query followed by sort and head."""
        ds_df, pd_df = ds

        pd_result = pd_df.query('a > 3').sort_values('b', ascending=False).head(3)
        ds_result = ds_df.query('a > 3').sort_values('b', ascending=False).head(3)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_eval_creates_column_then_filter(self, ds):
        """Eval creates column, then filter on it."""
        ds_df, pd_df = ds

        pd_df2 = pd_df.eval('d = a + b')
        pd_result = pd_df2[pd_df2['d'] > 50]

        ds_df2 = ds_df.eval('d = a + b')
        ds_result = ds_df2[ds_df2['d'] > 50]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_creates_column_then_groupby_original(self, ds):
        """Eval creates column, then groupby on original column."""
        ds_df, pd_df = ds

        pd_df2 = pd_df.eval('d = a + b')
        pd_result = pd_df2.groupby('a')['d'].sum().reset_index()

        ds_df2 = ds_df.eval('d = a + b')
        ds_result = ds_df2.groupby('a')['d'].sum().reset_index()

        # Sort for comparison
        pd_result = pd_result.sort_values('a').reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values('a').reset_index(drop=True)
        assert_frame_equal(ds_result_df, pd_result)

    def test_chained_query_calls(self, ds):
        """Multiple chained query calls."""
        ds_df, pd_df = ds

        pd_result = pd_df.query('a > 2').query('b < 80').query('c > 4.0')
        ds_result = ds_df.query('a > 2').query('b < 80').query('c > 4.0')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_in_operator(self, ds):
        """Query with 'in' operator."""
        ds_df, pd_df = ds

        pd_result = pd_df.query('a in [1, 3, 5, 7, 9]')
        ds_result = ds_df.query('a in [1, 3, 5, 7, 9]')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_not_in_operator(self, ds):
        """Query with 'not in' operator."""
        ds_df, pd_df = ds

        pd_result = pd_df.query('a not in [1, 2, 3]')
        ds_result = ds_df.query('a not in [1, 2, 3]')

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestClipBetweenWhereChains:
    """Test clip(), between() and where() in combination chains."""

    @pytest.fixture
    def ds(self):
        data = {
            'x': [-5, -2, 0, 3, 7, 10, 15, 20, 25, 30],
            'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'z': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        }
        return DataStore(data), pd.DataFrame(data)

    def test_clip_basic(self, ds):
        """Basic clip operation."""
        ds_df, pd_df = ds

        pd_result = pd_df['x'].clip(lower=0, upper=20)
        ds_result = ds_df['x'].clip(lower=0, upper=20)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_clip_then_filter(self, ds):
        """Clip followed by filter."""
        ds_df, pd_df = ds

        pd_df2 = pd_df.copy()
        pd_df2['x_clipped'] = pd_df['x'].clip(lower=0, upper=20)
        pd_result = pd_df2[pd_df2['x_clipped'] > 5]

        ds_df2 = ds_df.copy()
        ds_df2['x_clipped'] = ds_df['x'].clip(lower=0, upper=20)
        ds_result = ds_df2[ds_df2['x_clipped'] > 5]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_basic(self, ds):
        """Basic between operation."""
        ds_df, pd_df = ds

        pd_result = pd_df['x'].between(0, 15)
        ds_result = ds_df['x'].between(0, 15)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_between_inclusive(self, ds):
        """Between with inclusive parameter."""
        ds_df, pd_df = ds

        pd_result = pd_df['x'].between(0, 15, inclusive='neither')
        ds_result = ds_df['x'].between(0, 15, inclusive='neither')

        assert_series_equal(get_series(ds_result), pd_result)

    def test_between_as_filter(self, ds):
        """Use between result as filter."""
        ds_df, pd_df = ds

        pd_result = pd_df[pd_df['x'].between(0, 15)]
        ds_result = ds_df[ds_df['x'].between(0, 15)]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_where_then_clip(self, ds):
        """Where followed by clip."""
        ds_df, pd_df = ds

        pd_result = pd_df['x'].where(pd_df['x'] > 0, 0).clip(upper=15)
        ds_result = ds_df['x'].where(ds_df['x'] > 0, 0).clip(upper=15)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_clip_where_filter_chain(self, ds):
        """Clip -> where -> filter chain."""
        ds_df, pd_df = ds

        pd_df2 = pd_df.copy()
        pd_df2['x_proc'] = pd_df['x'].clip(lower=-3, upper=25).where(pd_df['y'] > 3, -999)
        pd_result = pd_df2[pd_df2['x_proc'] != -999]

        ds_df2 = ds_df.copy()
        ds_df2['x_proc'] = ds_df['x'].clip(lower=-3, upper=25).where(ds_df['y'] > 3, -999)
        ds_result = ds_df2[ds_df2['x_proc'] != -999]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_then_where(self, ds):
        """Between used in where condition."""
        ds_df, pd_df = ds

        pd_result = pd_df['z'].where(pd_df['x'].between(0, 15), 0)
        ds_result = ds_df['z'].where(ds_df['x'].between(0, 15), 0)

        assert_series_equal(get_series(ds_result), pd_result)


class TestAstypeChains:
    """Test astype() in chain operations - dtype coercion edge cases."""

    @pytest.fixture
    def ds(self):
        data = {
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
            'bool_col': [True, False, True, False, True],
        }
        return DataStore(data), pd.DataFrame(data)

    def test_astype_int_to_float(self, ds):
        """Convert int to float."""
        ds_df, pd_df = ds

        pd_result = pd_df['int_col'].astype(float)
        ds_result = ds_df['int_col'].astype(float)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_float_to_int(self, ds):
        """Convert float to int (truncation)."""
        ds_df, pd_df = ds

        pd_result = pd_df['float_col'].astype(int)
        ds_result = ds_df['float_col'].astype(int)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_int_to_str(self, ds):
        """Convert int to string."""
        ds_df, pd_df = ds

        pd_result = pd_df['int_col'].astype(str)
        ds_result = ds_df['int_col'].astype(str)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_bool_to_int(self, ds):
        """Convert bool to int."""
        ds_df, pd_df = ds

        pd_result = pd_df['bool_col'].astype(int)
        ds_result = ds_df['bool_col'].astype(int)

        assert_series_equal(get_series(ds_result), pd_result)

    def test_astype_then_filter(self, ds):
        """Astype followed by filter."""
        ds_df, pd_df = ds

        pd_df2 = pd_df.copy()
        pd_df2['float_as_int'] = pd_df['float_col'].astype(int)
        pd_result = pd_df2[pd_df2['float_as_int'] > 2]

        ds_df2 = ds_df.copy()
        ds_df2['float_as_int'] = ds_df['float_col'].astype(int)
        ds_result = ds_df2[ds_df2['float_as_int'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_astype(self, ds):
        """Filter followed by astype."""
        ds_df, pd_df = ds

        pd_filtered = pd_df[pd_df['int_col'] > 2]
        pd_result = pd_filtered['float_col'].astype(int)

        ds_filtered = ds_df[ds_df['int_col'] > 2]
        ds_result = ds_filtered['float_col'].astype(int)

        # Reset index for comparison (filter changes index)
        assert_series_equal(
            get_series(ds_result).reset_index(drop=True), pd_result.reset_index(drop=True))

    def test_astype_multiple_columns_dict(self, ds):
        """Astype with dict for multiple columns."""
        ds_df, pd_df = ds

        pd_result = pd_df.astype({'int_col': float, 'float_col': int})
        ds_result = ds_df.astype({'int_col': float, 'float_col': int})

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_chain_conversion(self, ds):
        """Chain multiple astype conversions."""
        ds_df, pd_df = ds

        pd_result = pd_df['int_col'].astype(float).astype(str)
        ds_result = ds_df['int_col'].astype(float).astype(str)

        assert_series_equal(get_series(ds_result), pd_result)


class TestAssignExpressions:
    """Test assign() with complex expressions and dependencies."""

    @pytest.fixture
    def ds(self):
        data = {
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500],
        }
        return DataStore(data), pd.DataFrame(data)

    def test_assign_single_column(self, ds):
        """Assign a single new column."""
        ds_df, pd_df = ds

        pd_result = pd_df.assign(d=lambda x: x['a'] + x['b'])
        ds_result = ds_df.assign(d=lambda x: x['a'] + x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_multiple_columns(self, ds):
        """Assign multiple new columns."""
        ds_df, pd_df = ds

        pd_result = pd_df.assign(d=lambda x: x['a'] + x['b'], e=lambda x: x['b'] * 2)
        ds_result = ds_df.assign(d=lambda x: x['a'] + x['b'], e=lambda x: x['b'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_overwrite_column(self, ds):
        """Assign overwrites existing column."""
        ds_df, pd_df = ds

        pd_result = pd_df.assign(a=lambda x: x['a'] * 2)
        ds_result = ds_df.assign(a=lambda x: x['a'] * 2)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_scalar(self, ds):
        """Assign column with scalar value."""
        ds_df, pd_df = ds

        pd_result = pd_df.assign(d=100)
        ds_result = ds_df.assign(d=100)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_series(self, ds):
        """Assign column with Series value."""
        ds_df, pd_df = ds

        new_col = pd.Series([5, 4, 3, 2, 1])
        pd_result = pd_df.assign(d=new_col)
        ds_result = ds_df.assign(d=new_col)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self, ds):
        """Assign followed by filter."""
        ds_df, pd_df = ds

        pd_result = pd_df.assign(d=lambda x: x['a'] + x['b']).query('d > 30')
        ds_result = ds_df.assign(d=lambda x: x['a'] + x['b']).query('d > 30')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_groupby_simple(self, ds):
        """Assign followed by simple groupby on original column."""
        ds_df, pd_df = ds

        # First assign a new column
        pd_assigned = pd_df.assign(d=lambda x: x['a'] * 10)
        ds_assigned = ds_df.assign(d=lambda x: x['a'] * 10)

        # Then groupby on original column 'a' and aggregate new column 'd'
        pd_result = pd_assigned.groupby('a')['d'].sum().reset_index()
        ds_result = ds_assigned.groupby('a')['d'].sum().reset_index()

        # Sort for comparison
        pd_result = pd_result.sort_values('a').reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values('a').reset_index(drop=True)
        assert_frame_equal(ds_result_df, pd_result)

    def test_chained_assign(self, ds):
        """Chain multiple assign calls."""
        ds_df, pd_df = ds

        pd_result = pd_df.assign(d=lambda x: x['a'] * 2).assign(e=lambda x: x['d'] + x['b'])
        ds_result = ds_df.assign(d=lambda x: x['a'] * 2).assign(e=lambda x: x['d'] + x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_dependent_columns(self, ds):
        """Assign columns that depend on each other (within same assign)."""
        ds_df, pd_df = ds

        # In pandas, columns are evaluated in order, but each lambda sees original df
        pd_result = pd_df.assign(d=lambda x: x['a'] * 2, e=lambda x: x['a'] * 3)  # Uses original a, not modified
        ds_result = ds_df.assign(d=lambda x: x['a'] * 2, e=lambda x: x['a'] * 3)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestUpdateOperations:
    """Test update() operations with various scenarios."""

    @pytest.fixture
    def ds(self):
        data = {
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
        }
        return DataStore(data), pd.DataFrame(data)

    def test_update_with_dataframe(self, ds):
        """Update with another DataFrame."""
        ds_df, pd_df = ds

        update_data = pd.DataFrame({'a': [100, 200]}, index=[0, 1])

        pd_df_copy = pd_df.copy()
        pd_df_copy.update(update_data)

        ds_df_copy = ds_df.copy()
        ds_df_copy.update(update_data)

        assert_datastore_equals_pandas(ds_df_copy, pd_df_copy)

    def test_update_overwrites_only_provided(self, ds):
        """Update only overwrites values that are provided."""
        ds_df, pd_df = ds

        update_data = pd.DataFrame({'a': [100, np.nan, 300]}, index=[0, 1, 2])

        pd_df_copy = pd_df.copy()
        pd_df_copy.update(update_data)

        ds_df_copy = ds_df.copy()
        ds_df_copy.update(update_data)

        assert_datastore_equals_pandas(ds_df_copy, pd_df_copy)

    def test_update_with_overwrite_false(self, ds):
        """Update with overwrite=False only fills NaN."""
        ds_df, pd_df = ds

        # Add NaN to original
        pd_df_with_nan = pd_df.copy()
        pd_df_with_nan.loc[0, 'a'] = np.nan

        ds_df_with_nan = DataStore(pd_df_with_nan)

        update_data = pd.DataFrame({'a': [100, 200]}, index=[0, 1])

        pd_df_with_nan.update(update_data, overwrite=False)
        ds_df_with_nan.update(update_data, overwrite=False)

        assert_datastore_equals_pandas(ds_df_with_nan, pd_df_with_nan)


class TestComplexChainScenarios:
    """Test complex scenarios combining multiple operations."""

    @pytest.fixture
    def ds(self):
        np.random.seed(42)
        data = {
            'id': list(range(1, 21)),
            'category': ['A', 'B', 'C', 'D'] * 5,
            'value': list(np.random.randint(10, 100, 20)),
            'score': list(np.random.random(20) * 100),
        }
        return DataStore(data), pd.DataFrame(data)

    def test_query_filter_groupby_chain(self, ds):
        """Query -> filter -> groupby chain on existing column."""
        ds_df, pd_df = ds

        pd_result = (
            pd_df.query('value > 20')[pd_df.query('value > 20')['score'] > 30]
            .groupby('category')['value']
            .mean()
            .reset_index()
        )

        ds_result = (
            ds_df.query('value > 20')[ds_df.query('value > 20')['score'] > 30]
            .groupby('category')['value']
            .mean()
            .reset_index()
        )

        # Sort for comparison
        pd_result = pd_result.sort_values('category').reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values('category').reset_index(drop=True)
        assert_frame_equal(ds_result_df, pd_result)

    def test_filter_eval_sort_head_chain(self, ds):
        """Filter -> eval -> sort -> head chain."""
        ds_df, pd_df = ds

        pd_result = (
            pd_df[pd_df['value'] > 30].eval('combined = value + score').sort_values('combined', ascending=False).head(5)
        )

        ds_result = (
            ds_df[ds_df['value'] > 30].eval('combined = value + score').sort_values('combined', ascending=False).head(5)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_multiple_column_ops_chain(self, ds):
        """Chain of column selection, rename, drop operations."""
        ds_df, pd_df = ds

        pd_result = pd_df[['id', 'category', 'value']].rename(columns={'value': 'val'}).query('val > 40')

        ds_result = ds_df[['id', 'category', 'value']].rename(columns={'value': 'val'}).query('val > 40')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_astype_groupby_agg_simple_chain(self, ds):
        """Astype -> groupby -> simple agg chain."""
        ds_df, pd_df = ds

        pd_df2 = pd_df.copy()
        pd_df2['score_int'] = pd_df['score'].astype(int)
        pd_result = pd_df2.groupby('category')['score_int'].mean().reset_index()

        ds_df2 = ds_df.copy()
        ds_df2['score_int'] = ds_df['score'].astype(int)
        ds_result = ds_df2.groupby('category')['score_int'].mean().reset_index()

        # Sort for comparison
        pd_result = pd_result.sort_values('category').reset_index(drop=True)
        ds_result_df = get_dataframe(ds_result).sort_values('category').reset_index(drop=True)
        assert_frame_equal(ds_result_df, pd_result)


class TestEdgeCasesEmptyAndSingleRow:
    """Edge cases with empty and single-row DataFrames."""

    def test_query_returns_empty(self):
        """Query that returns empty result."""
        data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.query('a > 100')
        ds_result = ds_df.query('a > 100')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_on_empty_df(self):
        """Eval on empty DataFrame."""
        data = {'a': [], 'b': []}
        pd_df = pd.DataFrame(data).astype({'a': int, 'b': int})
        ds_df = DataStore(data)

        pd_result = pd_df.eval('c = a + b')
        ds_result = ds_df.eval('c = a + b')

        assert len(ds_result) == 0
        assert len(pd_result) == 0

    def test_assign_on_single_row(self):
        """Assign on single-row DataFrame."""
        data = {'a': [1], 'b': [2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.assign(c=lambda x: x['a'] + x['b'])
        ds_result = ds_df.assign(c=lambda x: x['a'] + x['b'])

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_on_empty(self):
        """Clip on empty DataFrame."""
        data = {'a': []}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df['a'].clip(lower=0, upper=10)
        ds_result = ds_df['a'].clip(lower=0, upper=10)

        assert len(pd_result) == 0
        assert len(get_series(ds_result)) == 0

    def test_between_all_false(self):
        """Between where all values are outside range."""
        data = {'a': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df['a'].between(100, 200)
        ds_result = ds_df['a'].between(100, 200)

        assert pd_result.sum() == 0
        assert get_series(ds_result).astype(bool).sum() == 0

    def test_between_all_true(self):
        """Between where all values are inside range."""
        data = {'a': [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df['a'].between(0, 10)
        ds_result = ds_df['a'].between(0, 10)

        assert pd_result.sum() == 5
        assert get_series(ds_result).astype(bool).sum() == 5


class TestNullableTypesWithOperations:
    """Test nullable types with eval/query/assign operations."""

    @pytest.fixture
    def ds_nullable(self):
        data = {
            'a': pd.array([1, 2, None, 4, 5], dtype=pd.Int64Dtype()),
            'b': pd.array([10, None, 30, 40, 50], dtype=pd.Int64Dtype()),
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        return ds_df, pd_df

    def test_query_with_nullable_column(self, ds_nullable):
        """Query on nullable column."""
        ds_df, pd_df = ds_nullable

        pd_result = pd_df.query('a > 2')
        ds_result = ds_df.query('a > 2')

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_with_nullable_input(self, ds_nullable):
        """Assign using nullable column in expression."""
        ds_df, pd_df = ds_nullable

        pd_result = pd_df.assign(c=lambda x: x['a'].fillna(0) + x['b'].fillna(0))
        ds_result = ds_df.assign(c=lambda x: x['a'].fillna(0) + x['b'].fillna(0))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_on_nullable(self, ds_nullable):
        """Clip on nullable column (NAs should remain, dtype preserved)."""
        ds_df, pd_df = ds_nullable

        pd_result = pd_df['a'].clip(lower=2, upper=4)
        ds_result = ds_df['a'].clip(lower=2, upper=4)

        # Full comparison including dtype (fixed via dtype_correction)
        assert_datastore_equals_pandas(ds_result, pd_result)
