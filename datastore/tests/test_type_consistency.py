"""
Test type consistency between DataStore and pandas.

This module tests that DataStore operations return the correct types:
- DataFrame operations return DataStore (equivalent to pd.DataFrame)
- Series operations return ColumnExpr (equivalent to pd.Series)
- Scalar aggregations (without groupby) return scalars (matches pandas)
- GroupBy aggregations return Series/DataFrame via ColumnExpr

Design Decision: Scalar Returns for Non-Grouped Aggregations
============================================================
ds['col'].sum() returns scalar (numpy.int64, numpy.float64, etc.)
This matches pandas behavior: df['col'].sum() returns scalar.

For SQL building in agg(), use col() from datastore.expressions:
  ds.groupby('x').agg(avg=col('value').mean())  # Use col() for SQL
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from datastore.column_expr import ColumnExpr


class TestIndexingTypeConsistency:
    """Test that indexing operations return correct types."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': ['x', 'y', 'z']})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': ['x', 'y', 'z']})

    def test_single_column_returns_columnexpr(self, ds, df):
        """ds['col'] should return ColumnExpr (like pd.Series)."""
        ds_result = ds['a']
        pd_result = df['a']

        assert isinstance(ds_result, ColumnExpr), f"Expected ColumnExpr, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)

    def test_multi_column_returns_datastore(self, ds, df):
        """ds[['col1', 'col2']] should return DataStore (like pd.DataFrame)."""
        ds_result = ds[['a', 'b']]
        pd_result = df[['a', 'b']]

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_boolean_indexing_returns_datastore(self, ds, df):
        """ds[condition] should return DataStore (like pd.DataFrame)."""
        ds_result = ds[ds['a'] > 1]
        pd_result = df[df['a'] > 1]

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_slice_returns_datastore(self, ds, df):
        """ds[:n] should return DataStore (like pd.DataFrame)."""
        ds_result = ds[:2]
        pd_result = df[:2]

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_integer_column_name_returns_columnexpr(self):
        """ds[0] should return ColumnExpr when column name is integer."""
        ds = DataStore({0: [1, 2, 3], 1: [4, 5, 6]})
        df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})

        ds_result = ds[0]
        pd_result = df[0]

        assert isinstance(ds_result, ColumnExpr), f"Expected ColumnExpr, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)


class TestAggregationReturnsScalar:
    """Test that scalar aggregations (without groupby) return scalars like pandas."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0], 'cat': ['A', 'A', 'B', 'B', 'B']})

    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0], 'cat': ['A', 'A', 'B', 'B', 'B']}
        )

    # ========== No groupby - returns scalar (matches pandas) ==========

    def test_sum_returns_scalar(self, ds, df):
        """ds['col'].sum() returns scalar to match pandas."""
        ds_result = ds['a'].sum()
        pd_result = df['a'].sum()
        assert type(ds_result) == type(pd_result), f"Expected {type(pd_result).__name__}, got {type(ds_result).__name__}"
        assert ds_result == pd_result

    def test_mean_returns_scalar(self, ds, df):
        """ds['col'].mean() returns scalar to match pandas."""
        ds_result = ds['a'].mean()
        pd_result = df['a'].mean()
        assert type(ds_result) == type(pd_result), f"Expected {type(pd_result).__name__}, got {type(ds_result).__name__}"
        np.testing.assert_almost_equal(ds_result, pd_result)

    def test_min_returns_scalar(self, ds, df):
        """ds['col'].min() returns scalar to match pandas."""
        ds_result = ds['a'].min()
        pd_result = df['a'].min()
        assert type(ds_result) == type(pd_result), f"Expected {type(pd_result).__name__}, got {type(ds_result).__name__}"
        assert ds_result == pd_result

    def test_max_returns_scalar(self, ds, df):
        """ds['col'].max() returns scalar to match pandas."""
        ds_result = ds['a'].max()
        pd_result = df['a'].max()
        assert type(ds_result) == type(pd_result), f"Expected {type(pd_result).__name__}, got {type(ds_result).__name__}"
        assert ds_result == pd_result

    def test_count_returns_scalar(self, ds, df):
        """ds['col'].count() returns scalar to match pandas."""
        ds_result = ds['a'].count()
        pd_result = df['a'].count()
        assert type(ds_result) == type(pd_result), f"Expected {type(pd_result).__name__}, got {type(ds_result).__name__}"
        assert ds_result == pd_result

    # ========== With groupby - also returns ColumnExpr ==========

    def test_groupby_sum_returns_columnexpr(self, ds):
        """ds.groupby('cat')['col'].sum() returns ColumnExpr."""
        result = ds.groupby('cat')['a'].sum()
        assert isinstance(result, ColumnExpr), f"Expected ColumnExpr, got {type(result).__name__}"


class TestColumnExprBehavesLikeScalar:
    """Test that ColumnExpr aggregations behave like scalars."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3]})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3]})

    def test_add_to_aggregation(self, ds, df):
        """ds['col'].sum() + 10 should work and give correct result."""
        ds_result = ds['a'].sum() + 10
        pd_result = df['a'].sum() + 10
        assert ds_result == pd_result == 16

    def test_subtract_from_aggregation(self, ds, df):
        """ds['col'].sum() - 1 should work."""
        ds_result = ds['a'].sum() - 1
        pd_result = df['a'].sum() - 1
        assert ds_result == pd_result == 5

    def test_multiply_aggregation(self, ds, df):
        """ds['col'].sum() * 2 should work."""
        ds_result = ds['a'].sum() * 2
        pd_result = df['a'].sum() * 2
        assert ds_result == pd_result == 12

    def test_divide_aggregation(self, ds, df):
        """ds['col'].sum() / 2 should work."""
        ds_result = ds['a'].sum() / 2
        pd_result = df['a'].sum() / 2
        assert ds_result == pd_result == 3.0

    def test_int_coercion(self, ds, df):
        """int(ds['col'].sum()) should work."""
        ds_result = int(ds['a'].sum())
        pd_result = int(df['a'].sum())
        assert ds_result == pd_result == 6

    def test_float_coercion(self, ds, df):
        """float(ds['col'].mean()) should work."""
        ds_result = float(ds['a'].mean())
        pd_result = float(df['a'].mean())
        assert ds_result == pd_result == 2.0

    def test_compare_to_scalar(self, ds, df):
        """ds['col'].sum() > 5 should work."""
        ds_result = ds['a'].sum() > 5
        pd_result = df['a'].sum() > 5
        assert ds_result == pd_result == True

    def test_compare_less_than(self, ds, df):
        """ds['col'].sum() < 10 should work."""
        ds_result = ds['a'].sum() < 10
        pd_result = df['a'].sum() < 10
        assert ds_result == pd_result == True

    def test_repr_shows_value(self, ds, df):
        """repr(ds['col'].sum()) should show the value."""
        ds_repr = repr(ds['a'].sum())
        pd_repr = repr(df['a'].sum())
        # Both should contain '6'
        assert '6' in ds_repr
        assert '6' in pd_repr


class TestAggregationValueCorrectness:
    """Test that aggregation values match pandas."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0]})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0]})

    def test_sum_value(self, ds, df):
        """Sum values should match."""
        ds_val = int(ds['a'].sum())
        pd_val = int(df['a'].sum())
        assert ds_val == pd_val == 15

    def test_mean_value(self, ds, df):
        """Mean values should match."""
        ds_val = float(ds['a'].mean())
        pd_val = float(df['a'].mean())
        assert ds_val == pd_val == 3.0

    def test_min_value(self, ds, df):
        """Min values should match."""
        ds_val = int(ds['a'].min())
        pd_val = int(df['a'].min())
        assert ds_val == pd_val == 1

    def test_max_value(self, ds, df):
        """Max values should match."""
        ds_val = int(ds['a'].max())
        pd_val = int(df['a'].max())
        assert ds_val == pd_val == 5

    def test_count_value(self, ds, df):
        """Count values should match."""
        ds_val = int(ds['a'].count())
        pd_val = int(df['a'].count())
        assert ds_val == pd_val == 5

    def test_std_value(self, ds, df):
        """Std values should match."""
        ds_val = float(ds['a'].std())
        pd_val = float(df['a'].std())
        assert np.isclose(ds_val, pd_val)

    def test_var_value(self, ds, df):
        """Var values should match."""
        ds_val = float(ds['a'].var())
        pd_val = float(df['a'].var())
        assert np.isclose(ds_val, pd_val)

    def test_median_value(self, ds, df):
        """Median values should match."""
        ds_val = float(ds['a'].median())
        pd_val = float(df['a'].median())
        assert ds_val == pd_val == 3.0

    def test_prod_value(self, ds, df):
        """Prod values should match."""
        ds_val = int(ds['a'].prod())
        pd_val = int(df['a'].prod())
        assert ds_val == pd_val == 120


class TestDataFrameOperationTypes:
    """Test that DataFrame-level operations return correct types."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3], 'b': [4, 5, 6]})

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    def test_head_returns_datastore(self, ds, df):
        """ds.head() should return DataStore."""
        ds_result = ds.head()
        pd_result = df.head()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_tail_returns_datastore(self, ds, df):
        """ds.tail() should return DataStore."""
        ds_result = ds.tail()
        pd_result = df.tail()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_dropna_returns_datastore(self, ds, df):
        """ds.dropna() should return DataStore."""
        ds_result = ds.dropna()
        pd_result = df.dropna()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_assign_returns_datastore(self, ds, df):
        """ds.assign(c=1) should return DataStore."""
        ds_result = ds.assign(c=1)
        pd_result = df.assign(c=1)

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_sample_returns_datastore(self, ds, df):
        """ds.sample(2) should return DataStore."""
        ds_result = ds.sample(2)
        pd_result = df.sample(2)

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_describe_returns_datastore(self, ds, df):
        """ds.describe() should return DataStore."""
        ds_result = ds.describe()
        pd_result = df.describe()

        assert isinstance(ds_result, DataStore), f"Expected DataStore, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.DataFrame)

    def test_dataframe_sum_returns_series(self, ds, df):
        """ds.sum() (DataFrame-level) should return Series."""
        ds_result = ds.sum()
        pd_result = df.sum()

        # DataStore.sum() returns Series (not DataStore)
        assert isinstance(ds_result, pd.Series), f"Expected Series, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)

    def test_dataframe_mean_returns_series(self, ds, df):
        """ds.mean() (DataFrame-level) should return Series."""
        ds_result = ds.mean()
        pd_result = df.mean()

        assert isinstance(ds_result, pd.Series), f"Expected Series, got {type(ds_result).__name__}"
        assert isinstance(pd_result, pd.Series)


class TestSQLBuildingWithCol:
    """Test that SQL building uses col() from datastore.expressions."""

    @pytest.fixture
    def ds(self):
        return DataStore({'a': [1, 2, 3, 4, 5], 'cat': ['A', 'A', 'B', 'B', 'B']})

    def test_col_aggregation_has_to_sql_for_sql(self, ds):
        """col() aggregation should have to_sql() for SQL building."""
        from datastore.expressions import col
        result = col('a').sum()
        # col().sum() returns an Expression (AggregateFunction), not ColumnExpr
        # It should have to_sql() for SQL generation
        assert hasattr(result, 'to_sql'), "col() aggregation should have to_sql()"
        sql = result.to_sql()
        assert 'sum' in sql.lower(), f"Expected sum in SQL, got: {sql}"

    def test_col_aggregation_can_use_as_alias(self, ds):
        """col() aggregation should support .as_() for SQL aliases."""
        from datastore.expressions import col
        result = col('a').sum()
        # Should have as_ method for aliasing
        assert hasattr(result, 'as_'), "col() aggregation should have as_() method"
    
    def test_scalar_agg_matches_pandas(self, ds):
        """ds['col'].sum() returns scalar matching pandas behavior."""
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'cat': ['A', 'A', 'B', 'B', 'B']})
        
        ds_result = ds['a'].sum()
        pd_result = df['a'].sum()
        
        # Should be scalar, not ColumnExpr
        assert not isinstance(ds_result, ColumnExpr), "ds['col'].sum() should return scalar"
        assert type(ds_result) == type(pd_result)
        assert ds_result == pd_result
