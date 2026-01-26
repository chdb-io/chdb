"""
Heuristic edge case tests - discovered through exploratory testing.
These tests cover less common pandas operations to ensure DataStore compatibility.
"""

from tests.test_utils import assert_frame_equal, assert_series_equal, get_dataframe
import pandas as pd
import numpy as np
import pytest
import datastore as ds


class TestCrosstab:
    """Test crosstab with DataStore columns."""

    def test_crosstab_basic(self):
        df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar', 'foo'],
            'B': ['one', 'two', 'one', 'two', 'one'],
        })
        ds_df = ds.DataStore(df)

        pd_result = pd.crosstab(df['A'], df['B'])
        ds_result = ds.crosstab(ds_df['A'], ds_df['B'])

        # Compare values (index/column names may differ)
        assert_frame_equal(
            ds_result.to_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))


class TestExplode:
    """Test explode operation."""

    def test_explode_list_column(self):
        df = pd.DataFrame({'A': [[1, 2], [3, 4]], 'B': ['x', 'y']})
        ds_df = ds.DataStore(df)

        pd_result = df.explode('A')
        ds_result = ds_df.explode('A')

        assert_frame_equal(
            ds_result.to_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))


class TestAssign:
    """Test assign with various inputs."""

    def test_assign_lambda(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds.DataStore(df)

        pd_result = df.assign(C=lambda x: x['A'] + x['B'])
        ds_result = ds_df.assign(C=lambda x: x['A'] + x['B'])

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_assign_scalar(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        ds_df = ds.DataStore(df)

        pd_result = df.assign(B=10)
        ds_result = ds_df.assign(B=10)

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestTransform:
    """Test transform operation."""

    def test_transform_lambda(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds.DataStore(df)

        pd_result = df[['A', 'B']].transform(lambda x: x * 2)
        ds_result = ds_df[['A', 'B']].transform(lambda x: x * 2)

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestAgg:
    """Test agg with various inputs."""

    def test_agg_multiple_functions(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        ds_df = ds.DataStore(df)

        pd_result = df.agg(['sum', 'mean'])
        ds_result = ds_df.agg(['sum', 'mean'])

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_agg_dict(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        ds_df = ds.DataStore(df)

        pd_result = df.agg({'A': 'sum', 'B': 'mean'})
        ds_result = ds_df.agg({'A': 'sum', 'B': 'mean'})

        assert_series_equal(ds_result, pd_result)


class TestEvalQuery:
    """Test eval and query operations."""

    def test_eval_expression(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds.DataStore(df)

        pd_result = df.eval('C = A + B')
        ds_result = ds_df.eval('C = A + B')

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_query_filter(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        ds_df = ds.DataStore(df)

        pd_result = df.query('A > 2')
        ds_result = ds_df.query('A > 2')

        assert_frame_equal(
            ds_result.to_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))


class TestPipe:
    """Test pipe operation."""

    def test_pipe_function(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        ds_df = ds.DataStore(df)

        def double_col_a(df):
            # Handle both DataStore and DataFrame
            is_datastore = isinstance(df, ds.DataStore)
            result = get_dataframe(df).copy()
            result['A'] = result['A'] * 2
            return ds.DataStore(result) if is_datastore else result

        pd_result = df.pipe(double_col_a)
        ds_result = ds_df.pipe(double_col_a)

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestGroupbyVariants:
    """Test groupby with various options."""

    def test_groupby_as_index_false(self):
        """groupby with as_index=False should return group key as column, not index."""
        df = pd.DataFrame({
            'A': ['foo', 'foo', 'bar', 'bar'],
            'B': [1, 2, 3, 4]
        })
        ds_df = ds.DataStore(df)

        pd_result = df.groupby('A', as_index=False).sum()
        ds_result = ds_df.groupby('A', as_index=False).sum()

        assert_frame_equal(
            ds_result.to_df().reset_index(drop=True),
            pd_result.reset_index(drop=True))


class TestRollingExpanding:
    """Test rolling and expanding operations.

    Note: rolling/expanding return pandas DataFrame directly (pass-through to pandas).
    """

    def test_rolling_mean(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.rolling(window=2).mean()
        ds_result = ds_df.rolling(window=2).mean()

        # rolling returns pandas DataFrame directly
        assert_frame_equal(ds_result, pd_result)

    def test_expanding_sum(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0, 5.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.expanding().sum()
        ds_result = ds_df.expanding().sum()

        # expanding returns pandas DataFrame directly
        assert_frame_equal(ds_result, pd_result)


class TestCumulativeFunctions:
    """Test cumulative functions."""

    def test_cumsum(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        ds_df = ds.DataStore(df)

        pd_result = df.cumsum()
        ds_result = ds_df.cumsum()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_cumprod(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4]})
        ds_df = ds.DataStore(df)

        pd_result = df.cumprod()
        ds_result = ds_df.cumprod()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_cummax(self):
        df = pd.DataFrame({'A': [1, 3, 2, 4]})
        ds_df = ds.DataStore(df)

        pd_result = df.cummax()
        ds_result = ds_df.cummax()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_cummin(self):
        df = pd.DataFrame({'A': [4, 2, 3, 1]})
        ds_df = ds.DataStore(df)

        pd_result = df.cummin()
        ds_result = ds_df.cummin()

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestShiftDiff:
    """Test shift and diff operations."""

    def test_shift(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.shift(1)
        ds_result = ds_df.shift(1)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_shift_negative(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0, 4.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.shift(-1)
        ds_result = ds_df.shift(-1)

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_diff(self):
        df = pd.DataFrame({'A': [1.0, 3.0, 6.0, 10.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.diff()
        ds_result = ds_df.diff()

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestRank:
    """Test rank operation."""

    def test_rank_default(self):
        df = pd.DataFrame({'A': [3.0, 1.0, 4.0, 1.0, 5.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.rank()
        ds_result = ds_df.rank()

        assert_frame_equal(get_dataframe(ds_result), pd_result)

    def test_rank_method_min(self):
        df = pd.DataFrame({'A': [3.0, 1.0, 4.0, 1.0, 5.0]})
        ds_df = ds.DataStore(df)

        pd_result = df.rank(method='min')
        ds_result = ds_df.rank(method='min')

        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestPctChange:
    """Test pct_change operation."""

    def test_pct_change(self):
        df = pd.DataFrame({'A': [10.0, 11.0, 12.1, 13.31]})
        ds_df = ds.DataStore(df)

        pd_result = df.pct_change()
        ds_result = ds_df.pct_change()

        assert_frame_equal(get_dataframe(ds_result), pd_result)
