"""
Tests for DataStore.from_df() and DataStore.from_dataframe() factory methods.
"""

import sys
from pathlib import Path

# Add parent directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_frame_equal


class TestFromDfBasics:
    """Test basic from_df functionality."""

    def test_from_df_creates_datastore(self):
        """Test that from_df returns a DataStore instance."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        ds = DataStore.from_df(df)
        assert isinstance(ds, DataStore)

    def test_from_df_preserves_data(self):
        """Test that from_df preserves the original data."""
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        ds = DataStore.from_df(df)
        result = ds.to_df()

        assert_frame_equal(result, df)

    def test_from_df_with_name(self):
        """Test from_df with optional name parameter."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_df(df, name='test_data')

        # Name should appear in explain output
        assert ds._original_source_desc is not None
        assert 'test_data' in ds._original_source_desc

    def test_from_df_without_name(self):
        """Test from_df without name parameter uses 'unnamed'."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_df(df)

        assert 'unnamed' in ds._original_source_desc

    def test_from_df_invalid_input(self):
        """Test from_df raises error for non-DataFrame input."""
        with pytest.raises(TypeError):
            DataStore.from_df([1, 2, 3])

        with pytest.raises(TypeError):
            DataStore.from_df({'a': [1, 2, 3]})

        with pytest.raises(TypeError):
            DataStore.from_df("not a dataframe")

    def test_from_df_schema_inferred(self):
        """Test that schema is inferred from DataFrame dtypes."""
        df = pd.DataFrame({'int_col': [1, 2, 3], 'float_col': [1.0, 2.0, 3.0], 'str_col': ['a', 'b', 'c']})
        ds = DataStore.from_df(df)

        assert ds._schema is not None
        assert 'int_col' in ds._schema
        assert 'float_col' in ds._schema
        assert 'str_col' in ds._schema


class TestFromDataframeAlias:
    """Test from_dataframe alias."""

    def test_from_dataframe_is_alias(self):
        """Test that from_dataframe is an alias for from_df."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        ds1 = DataStore.from_df(df)
        ds2 = DataStore.from_dataframe(df)

        assert_frame_equal(ds1.to_df(), ds2.to_df())

    def test_from_dataframe_with_name(self):
        """Test from_dataframe with name parameter."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        ds = DataStore.from_dataframe(df, name='my_data')

        assert 'my_data' in ds._original_source_desc


class TestFromDfFiltering:
    """Test filtering operations on DataStore from DataFrame."""

    def test_filter_with_condition(self):
        """Test filtering with a condition."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})
        ds = DataStore.from_df(df)

        result = ds.filter(ds.age > 26).to_df()

        assert len(result) == 2
        assert 'Alice' not in result['name'].values
        assert 'Bob' in result['name'].values
        assert 'Charlie' in result['name'].values

    def test_filter_multiple_conditions(self):
        """Test filtering with multiple conditions."""
        df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                'age': [25, 30, 35, 28],
                'country': ['USA', 'UK', 'USA', 'UK'],
            }
        )
        ds = DataStore.from_df(df)

        result = ds.filter(ds.age > 26).filter(ds.country == 'USA').to_df()

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Charlie'

    def test_filter_with_and(self):
        """Test filtering with AND operator."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35], 'active': [True, True, False]})
        ds = DataStore.from_df(df)

        result = ds.filter((ds.age > 20) & (ds.active == True)).to_df()

        assert len(result) == 2
        assert 'Charlie' not in result['name'].values


class TestFromDfColumnOperations:
    """Test column operations on DataStore from DataFrame."""

    def test_column_assignment(self):
        """Test assigning new columns."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        ds = DataStore.from_df(df)

        ds['doubled'] = ds.value * 2
        result = ds.to_df()

        assert 'doubled' in result.columns
        assert list(result['doubled']) == [20, 40, 60]

    def test_column_selection(self):
        """Test selecting specific columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        ds = DataStore.from_df(df)

        result = ds.select('a', 'b').to_df()

        assert list(result.columns) == ['a', 'b']
        assert 'c' not in result.columns

    def test_add_prefix(self):
        """Test adding prefix to columns."""
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        ds = DataStore.from_df(df)

        ds = ds.add_prefix('col_')
        result = ds.to_df()

        assert 'col_x' in result.columns
        assert 'col_y' in result.columns


class TestFromDfSQLOperations:
    """Test SQL operations on DataStore from DataFrame."""

    def test_sql_filter(self):
        """Test SQL filter using .sql() method."""
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        ds = DataStore.from_df(df)

        result = ds.sql('age > 26').to_df()

        assert len(result) == 1
        assert result.iloc[0]['name'] == 'Bob'

    def test_sql_full_query(self):
        """Test full SQL query using .sql() method."""
        df = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie'], 'age': [25, 30, 35]})
        ds = DataStore.from_df(df)

        result = ds.sql('SELECT name FROM __df__ WHERE age >= 30').to_df()

        assert len(result) == 2
        assert 'name' in result.columns

    def test_sql_with_order(self):
        """Test SQL query with ORDER BY."""
        df = pd.DataFrame({'name': ['Charlie', 'Alice', 'Bob'], 'age': [35, 25, 30]})
        ds = DataStore.from_df(df)

        result = ds.sql('SELECT * FROM __df__ ORDER BY age').to_df()

        assert list(result['name']) == ['Alice', 'Bob', 'Charlie']


class TestFromDfChainedOperations:
    """Test chained operations on DataStore from DataFrame."""

    def test_complex_chain(self):
        """Test complex chain of operations."""
        df = pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
                'age': [25, 30, 35, 28],
                'salary': [50000, 60000, 70000, 55000],
            }
        )
        ds = DataStore.from_df(df, name='employees')

        # Filter -> Add column -> Filter again -> Select
        ds = ds.filter(ds.age > 25)
        ds['bonus'] = ds.salary * 0.1
        ds = ds.filter(ds.bonus > 5500)

        result = ds.to_df()

        assert len(result) == 2  # Bob and Charlie
        assert 'bonus' in result.columns

    def test_mixed_sql_pandas(self):
        """Test mixing SQL and pandas operations."""
        df = pd.DataFrame({'id': [1, 2, 3, 4], 'value': [100, 200, 300, 400]})
        ds = DataStore.from_df(df)

        # SQL filter -> Pandas operation -> SQL filter
        ds = ds.sql('value > 150')
        ds['doubled'] = ds.value * 2
        ds = ds.sql('doubled > 500')

        result = ds.to_df()

        assert len(result) == 2  # values 300, 400 -> doubled 600, 800


class TestFromDfProperties:
    """Test DataFrame-like properties on DataStore from DataFrame."""

    def test_shape(self):
        """Test shape property."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5]})
        ds = DataStore.from_df(df)

        assert ds.shape == (5, 2)

    def test_columns(self):
        """Test columns property."""
        df = pd.DataFrame({'first': [1], 'second': [2], 'third': [3]})
        ds = DataStore.from_df(df)

        assert list(ds.columns) == ['first', 'second', 'third']

    def test_head(self):
        """Test head method."""
        df = pd.DataFrame({'x': range(100)})
        ds = DataStore.from_df(df)

        result = ds.head(5).to_df()

        assert len(result) == 5
        assert list(result['x']) == [0, 1, 2, 3, 4]

    def test_tail(self):
        """Test tail method."""
        df = pd.DataFrame({'x': range(100)})
        ds = DataStore.from_df(df)

        result = ds.tail(5).to_df()

        assert len(result) == 5


class TestFromDfEmptyDataFrame:
    """Test handling of empty DataFrames."""

    def test_empty_dataframe(self):
        """Test from_df with empty DataFrame."""
        df = pd.DataFrame({'a': [], 'b': []})
        ds = DataStore.from_df(df)

        result = ds.to_df()

        assert len(result) == 0
        assert list(result.columns) == ['a', 'b']

    def test_empty_after_filter(self):
        """Test filtering to empty result."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ds = DataStore.from_df(df)

        result = ds.filter(ds.value > 100).to_df()

        assert len(result) == 0


class TestFromDfDataTypes:
    """Test different data types in DataFrame."""

    def test_various_dtypes(self):
        """Test DataFrame with various data types."""
        df = pd.DataFrame(
            {
                'int_col': [1, 2, 3],
                'float_col': [1.5, 2.5, 3.5],
                'str_col': ['a', 'b', 'c'],
                'bool_col': [True, False, True],
            }
        )
        ds = DataStore.from_df(df)

        result = ds.to_df()

        assert_frame_equal(result, df)

    def test_datetime_column(self):
        """Test DataFrame with datetime column."""
        df = pd.DataFrame(
            {'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']), 'value': [100, 200, 300]}
        )
        ds = DataStore.from_df(df)

        result = ds.to_df()

        assert len(result) == 3

    def test_nullable_column(self):
        """Test DataFrame with nullable values."""
        df = pd.DataFrame({'name': ['Alice', None, 'Charlie'], 'age': [25, 30, None]})
        ds = DataStore.from_df(df)

        result = ds.to_df()

        assert len(result) == 3


class TestFromDfExplain:
    """Test explain() functionality for from_df DataStores."""

    def test_explain_shows_source(self):
        """Test that explain shows DataFrame source."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_df(df, name='test_source')

        # explain() prints to stdout, just make sure it doesn't error
        ds.explain()

    def test_explain_shows_operations(self):
        """Test that explain shows operations."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        ds = DataStore.from_df(df)
        ds = ds.filter(ds.value > 1)
        ds['doubled'] = ds.value * 2

        # explain() should work without errors
        ds.explain()


class TestFromDfImmutability:
    """Test that operations are immutable."""

    def test_filter_immutable(self):
        """Test that filter returns new DataStore."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds1 = DataStore.from_df(df)
        ds2 = ds1.filter(ds1.a > 1)

        # Original should be unchanged
        assert len(ds1.to_df()) == 3
        assert len(ds2.to_df()) == 2

    def test_select_immutable(self):
        """Test that select returns new DataStore."""
        df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        ds1 = DataStore.from_df(df)
        ds2 = ds1.select('a', 'b')

        # Original should have all columns
        assert 'c' in ds1.to_df().columns
        assert 'c' not in ds2.to_df().columns


class TestFromDfRepr:
    """Test string representation."""

    def test_str_representation(self):
        """Test __str__ representation."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_df(df)

        result = str(ds)
        assert '1' in result
        assert '2' in result
        assert '3' in result

    def test_repr_representation(self):
        """Test __repr__ representation."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        ds = DataStore.from_df(df)

        result = repr(ds)
        assert 'a' in result
