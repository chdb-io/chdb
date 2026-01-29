"""
Kaggle Pandas Compatibility Test Suite
======================================

Tests common pandas operations found in Kaggle notebooks, comparing
datastore behavior with pandas for API consistency.

Based on analysis of popular Kaggle notebooks including:
- Topic 1. Exploratory Data Analysis with Pandas (kashnitsky)
- Comprehensive Data Analysis with Pandas (prashant111)
- EDA: Exploratory Data Analysis notebook (udutta)
- And patterns from top 100+ Kaggle notebooks

Design Principle:
    Tests use natural execution triggers (.values, __eq__, len(), repr())
    following the lazy execution design principle.
    Avoid explicit _execute() calls - use natural triggers instead.
"""

from tests.test_utils import get_dataframe, get_series
import pytest
import pandas as pd
import numpy as np
import datastore as ds


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_df():
    """Simple DataFrame for basic tests."""
    return {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50], 'C': ['a', 'b', 'c', 'd', 'e']}


@pytest.fixture
def groupby_df():
    """DataFrame for groupby tests."""
    return {
        'category': ['A', 'B', 'A', 'B', 'A', 'B'],
        'value': [10, 20, 30, 40, 50, 60],
        'count': [1, 2, 3, 4, 5, 6],
    }


@pytest.fixture
def na_df():
    """DataFrame with missing values."""
    return {
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
    }


@pytest.fixture
def str_df():
    """DataFrame with strings."""
    return {'text': ['hello', 'WORLD', 'Python', 'pandas', 'DataStore']}


@pytest.fixture
def dup_df():
    """DataFrame with duplicates."""
    return {'A': [1, 2, 2, 3, 3, 3], 'B': ['a', 'b', 'b', 'c', 'c', 'c']}


@pytest.fixture
def merge_df1():
    """First DataFrame for merge tests."""
    return {'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]}


@pytest.fixture
def merge_df2():
    """Second DataFrame for merge tests."""
    return {'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]}


# ============================================================================
# Category 1: DataFrame Creation and Basic IO
# ============================================================================


class TestDataFrameCreation:
    """Tests for DataFrame creation operations."""

    def test_create_from_dict(self, simple_df):
        """Test creating DataFrame from dict."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        assert list(ds_df.columns) == list(pd_df.columns)
        assert ds_df.shape == pd_df.shape
        np.testing.assert_array_equal(ds_df.to_pandas().values, pd_df.values)

    def test_create_with_custom_index(self):
        """Test creating DataFrame with custom index."""
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        index = ['a', 'b', 'c']

        pd_df = pd.DataFrame(data, index=index)
        ds_df = ds.DataStore.from_df(pd.DataFrame(data, index=index))

        assert list(ds_df.index) == list(pd_df.index)
        np.testing.assert_array_equal(ds_df.to_pandas().values, pd_df.values)

    def test_create_series(self):
        """Test creating Series."""
        pd_series = pd.Series([1, 2, 3, 4], name='test_series')
        ds_series = ds.Series([1, 2, 3, 4], name='test_series')

        assert ds_series.name == pd_series.name
        np.testing.assert_array_equal(ds_series.values, pd_series.values)

    def test_read_csv(self, tmp_path):
        """Test reading CSV file."""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        pd.DataFrame(
            {
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [25, 30, 35],
                'city': ['New York', 'London', 'Paris'],
            }
        ).to_csv(csv_path, index=False)

        pd_df = pd.read_csv(csv_path)
        ds_df = ds.read_csv(str(csv_path))

        assert list(ds_df.columns) == list(pd_df.columns)
        assert ds_df.shape == pd_df.shape


# ============================================================================
# Category 2: Basic DataFrame Inspection
# ============================================================================


class TestDataFrameInspection:
    """Tests for DataFrame inspection operations."""

    def test_head(self, simple_df):
        """Test head() operation."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_head = pd_df.head(3)
        ds_head = ds_df.head(3)

        np.testing.assert_array_equal(ds_head.to_pandas().values, pd_head.values)

    def test_tail(self, simple_df):
        """Test tail() operation."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_tail = pd_df.tail(2)
        ds_tail = ds_df.tail(2)

        np.testing.assert_array_equal(ds_tail.to_pandas().values, pd_tail.values)

    def test_shape(self, simple_df):
        """Test shape property."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        assert ds_df.shape == pd_df.shape

    def test_columns(self, simple_df):
        """Test columns property."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        assert list(ds_df.columns) == list(pd_df.columns)

    def test_dtypes(self, simple_df):
        """Test dtypes property."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        # Compare dtype dict keys
        assert set(ds_df.dtypes.to_dict().keys()) == set(pd_df.dtypes.to_dict().keys())


# ============================================================================
# Category 3: Column Selection and Indexing
# ============================================================================


class TestColumnSelection:
    """Tests for column selection operations."""

    def test_select_single_column(self, simple_df):
        """Test selecting single column with []."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_col = pd_df['A']
        ds_col = ds_df['A']

        np.testing.assert_array_equal(ds_col.values, pd_col.values)

    def test_select_multiple_columns(self, simple_df):
        """Test selecting multiple columns with []."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_cols = pd_df[['A', 'B']]
        ds_cols = ds_df[['A', 'B']]

        np.testing.assert_array_equal(ds_cols.to_pandas().values, pd_cols.values)

    def test_loc_select_rows(self, simple_df):
        """Test loc for selecting rows by label."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        # loc returns native pandas DataFrame - this is expected behavior
        pd_result = pd_df.loc[0:2, ['A', 'B']]
        ds_result = ds_df.loc[0:2, ['A', 'B']]

        # Both return pandas DataFrame
        assert isinstance(ds_result, pd.DataFrame)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_iloc_select_rows(self, simple_df):
        """Test iloc for selecting rows by position."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        # iloc returns native pandas DataFrame - this is expected behavior
        pd_result = pd_df.iloc[0:2, 0:2]
        ds_result = ds_df.iloc[0:2, 0:2]

        # Both return pandas DataFrame
        assert isinstance(ds_result, pd.DataFrame)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)


# ============================================================================
# Category 4: Filtering and Boolean Indexing
# ============================================================================


class TestFiltering:
    """Tests for filtering operations."""

    def test_boolean_filter_single(self, simple_df):
        """Test boolean filter with single condition."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df[pd_df['A'] > 2]
        ds_result = ds_df[ds_df['A'] > 2]

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_boolean_filter_and(self, simple_df):
        """Test boolean filter with AND condition."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df[(pd_df['A'] > 2) & (pd_df['B'] < 50)]
        ds_result = ds_df[(ds_df['A'] > 2) & (ds_df['B'] < 50)]

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_boolean_filter_or(self, simple_df):
        """Test boolean filter with OR condition."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df[(pd_df['A'] < 2) | (pd_df['A'] > 4)]
        ds_result = ds_df[(ds_df['A'] < 2) | (ds_df['A'] > 4)]

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_isin_filter(self, simple_df):
        """Test isin() filter."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df[pd_df['A'].isin([1, 3, 5])]
        ds_result = ds_df[ds_df['A'].isin([1, 3, 5])]

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 5: Missing Value Handling
# ============================================================================


class TestMissingValues:
    """Tests for missing value handling."""

    def test_isna(self, na_df):
        """Test isna() detection."""
        pd_df = pd.DataFrame(na_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(na_df))

        pd_result = pd_df.isna()
        ds_result = ds_df.isna()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_fillna(self, na_df):
        """Test fillna() with scalar."""
        pd_df = pd.DataFrame(na_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(na_df))

        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_dropna(self, na_df):
        """Test dropna() to remove rows with missing values."""
        pd_df = pd.DataFrame(na_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(na_df))

        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 6: GroupBy and Aggregation
# ============================================================================


class TestGroupBy:
    """Tests for groupby operations."""

    def test_groupby_sum(self, groupby_df):
        """Test groupby().sum()."""
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        pd_result = pd_df.groupby('category')['value'].sum()
        ds_result = ds_df.groupby('category')['value'].sum()

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_groupby_mean(self, groupby_df):
        """Test groupby().mean()."""
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        pd_result = pd_df.groupby('category')['value'].mean()
        ds_result = ds_df.groupby('category')['value'].mean()

        np.testing.assert_allclose(ds_result.values, pd_result.values)

    def test_groupby_count(self, groupby_df):
        """Test groupby().count()."""
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        pd_result = pd_df.groupby('category')['value'].count()
        ds_result = ds_df.groupby('category')['value'].count()

        np.testing.assert_array_equal(ds_result.values, pd_result.values)
        # Verify dtype matches pandas (int64)
        assert ds_result.dtype == pd_result.dtype, f"dtype mismatch: {ds_result.dtype} vs {pd_result.dtype}"

    def test_groupby_agg_multiple(self, groupby_df):
        """Test groupby().agg() with multiple functions."""
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        pd_result = pd_df.groupby('category', sort=True)['value'].agg(['sum', 'mean', 'count'])
        ds_result = ds_df.groupby('category', sort=True)['value'].agg(['sum', 'mean', 'count'])

        # Both should return DataFrame with same columns and values
        ds_result_df = get_dataframe(ds_result)
        assert list(ds_result_df.columns) == list(
            pd_result.columns
        ), f"Column mismatch: {list(ds_result_df.columns)} vs {list(pd_result.columns)}"
        np.testing.assert_allclose(ds_result_df.values, pd_result.values, rtol=1e-5)


    def test_groupby_agg_nunique(self, groupby_df):
        """Test groupby().agg() with nunique function.
        
        This verifies the fix for: 'nunique' function name not mapped to SQL 'uniqExact()'.
        Previously, groupby().agg({'col': 'nunique'}) would fail with:
        "Code: 46. DB::Exception: Function with name 'nunique' does not exist"
        """
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        # Test agg with dict syntax
        pd_result = pd_df.groupby('category', sort=True).agg({'value': 'nunique'})
        ds_result = ds_df.groupby('category', sort=True).agg({'value': 'nunique'})

        ds_result_df = get_dataframe(ds_result)
        assert list(ds_result_df.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result_df.values, pd_result.values)

    def test_groupby_agg_nunique_multiple_cols(self):
        """Test groupby().agg() with nunique on multiple columns."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value1': [1, 1, 2, 3, 3, 3],
            'value2': ['x', 'x', 'y', 'x', 'y', 'z']
        })
        ds_df = ds.DataStore.from_df(pd_df)

        # Test agg with multiple columns
        pd_result = pd_df.groupby('group', sort=True).agg({
            'value1': 'nunique',
            'value2': 'nunique'
        })
        ds_result = ds_df.groupby('group', sort=True).agg({
            'value1': 'nunique',
            'value2': 'nunique'
        })

        ds_result_df = get_dataframe(ds_result)
        assert list(ds_result_df.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result_df.values, pd_result.values)

    def test_groupby_agg_mixed_with_nunique(self):
        """Test groupby().agg() mixing nunique with other agg functions."""
        pd_df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [1, 2, 2, 3, 3, 3]
        })
        ds_df = ds.DataStore.from_df(pd_df)

        # Test agg with list of functions including nunique
        pd_result = pd_df.groupby('group', sort=True)['value'].agg(['sum', 'nunique', 'count'])
        ds_result = ds_df.groupby('group', sort=True)['value'].agg(['sum', 'nunique', 'count'])

        ds_result_df = get_dataframe(ds_result)
        assert list(ds_result_df.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result_df.values, pd_result.values)


# ============================================================================
# Category 7: Sorting
# ============================================================================


class TestSorting:
    """Tests for sorting operations."""

    def test_sort_values_ascending(self, simple_df):
        """Test sort_values() ascending."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df.sort_values('A')
        ds_result = ds_df.sort_values('A')

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_sort_values_descending(self, simple_df):
        """Test sort_values() descending."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df.sort_values('B', ascending=False)
        ds_result = ds_df.sort_values('B', ascending=False)

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_sort_values_multiple_columns(self, groupby_df):
        """Test sort_values() with multiple columns."""
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        pd_result = pd_df.sort_values(['category', 'value'])
        ds_result = ds_df.sort_values(['category', 'value'])

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 8: Statistical Operations
# ============================================================================


class TestStatistics:
    """Tests for statistical operations."""

    def test_sum(self, simple_df):
        """Test column sum()."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df['A'].sum()
        ds_result = ds_df['A'].sum()

        assert ds_result == pd_result

    def test_mean(self, simple_df):
        """Test column mean()."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df['B'].mean()
        ds_result = ds_df['B'].mean()

        assert ds_result == pd_result

    def test_min_max(self, simple_df):
        """Test min() and max()."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        assert ds_df['A'].min() == pd_df['A'].min()
        assert ds_df['A'].max() == pd_df['A'].max()

    def test_describe(self, simple_df):
        """Test describe() summary statistics."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df[['A', 'B']].describe()
        ds_result = ds_df[['A', 'B']].describe()

        # Compare shape and index
        assert ds_result.to_pandas().shape == pd_result.shape

    def test_value_counts(self, groupby_df):
        """Test value_counts()."""
        pd_df = pd.DataFrame(groupby_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(groupby_df))

        pd_result = pd_df['category'].value_counts()
        ds_result = ds_df['category'].value_counts()

        # Sort both by index for comparison
        np.testing.assert_array_equal(ds_result.to_pandas().sort_index().values, pd_result.sort_index().values)


# ============================================================================
# Category 9: Data Transformation
# ============================================================================


class TestTransformation:
    """Tests for data transformation operations."""

    def test_rename_columns(self, simple_df):
        """Test rename() columns."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df.rename(columns={'A': 'col_A', 'B': 'col_B'})
        ds_result = ds_df.rename(columns={'A': 'col_A', 'B': 'col_B'})

        assert list(ds_result.columns) == list(pd_result.columns)

    def test_drop_columns(self, simple_df):
        """Test drop() columns."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df.drop(columns=['C'])
        ds_result = ds_df.drop(columns=['C'])

        assert list(ds_result.columns) == list(pd_result.columns)

    def test_reset_index(self, simple_df):
        """Test reset_index()."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df.reset_index(drop=True)
        ds_result = ds_df.reset_index(drop=True)

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_apply_lambda(self, simple_df):
        """Test apply() with lambda."""
        pd_df = pd.DataFrame(simple_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))

        pd_result = pd_df['A'].apply(lambda x: x * 2)
        ds_result = ds_df['A'].apply(lambda x: x * 2)

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 10: String Operations
# ============================================================================


class TestStringOperations:
    """Tests for string operations."""

    def test_str_lower(self, str_df):
        """Test str.lower()."""
        pd_df = pd.DataFrame(str_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(str_df))

        pd_result = pd_df['text'].str.lower()
        ds_result = ds_df['text'].str.lower()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_str_upper(self, str_df):
        """Test str.upper()."""
        pd_df = pd.DataFrame(str_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(str_df))

        pd_result = pd_df['text'].str.upper()
        ds_result = ds_df['text'].str.upper()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_str_contains(self, str_df):
        """Test str.contains()."""
        pd_df = pd.DataFrame(str_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(str_df))

        pd_result = pd_df['text'].str.contains('a')
        ds_result = ds_df['text'].str.contains('a')

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_str_len(self, str_df):
        """Test str.len()."""
        pd_df = pd.DataFrame(str_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(str_df))

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)
        # Verify dtype is int64 (not uint64)
        assert ds_result.to_pandas().dtype == pd_result.dtype


# ============================================================================
# Category 11: DateTime Operations
# ============================================================================


class TestDateTimeOperations:
    """Tests for datetime operations."""

    @pytest.fixture
    def dt_df(self):
        """DataFrame with dates."""
        return {'date': pd.date_range('2024-01-01', periods=5)}

    def test_dt_year(self, dt_df):
        """Test dt.year."""
        pd_df = pd.DataFrame(dt_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dt_df))

        pd_result = pd_df['date'].dt.year
        ds_result = ds_df['date'].dt.year

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_dt_month(self, dt_df):
        """Test dt.month."""
        pd_df = pd.DataFrame(dt_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dt_df))

        pd_result = pd_df['date'].dt.month
        ds_result = ds_df['date'].dt.month

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_dt_day(self, dt_df):
        """Test dt.day."""
        pd_df = pd.DataFrame(dt_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dt_df))

        pd_result = pd_df['date'].dt.day
        ds_result = ds_df['date'].dt.day

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_dt_dayofweek(self, dt_df):
        """Test dt.dayofweek."""
        pd_df = pd.DataFrame(dt_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dt_df))

        pd_result = pd_df['date'].dt.dayofweek
        ds_result = ds_df['date'].dt.dayofweek

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 12: Merge and Join Operations
# ============================================================================


class TestMergeJoin:
    """Tests for merge and join operations."""

    def test_merge_inner(self, merge_df1, merge_df2):
        """Test merge() inner join."""
        pd_df1 = pd.DataFrame(merge_df1)
        pd_df2 = pd.DataFrame(merge_df2)
        ds_df1 = ds.DataStore.from_df(pd.DataFrame(merge_df1))
        ds_df2 = ds.DataStore.from_df(pd.DataFrame(merge_df2))

        pd_result = pd.merge(pd_df1, pd_df2, on='key', how='inner')
        ds_result = ds.merge(ds_df1, ds_df2, on='key', how='inner')

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_merge_left(self, merge_df1, merge_df2):
        """Test merge() left join."""
        pd_df1 = pd.DataFrame(merge_df1)
        pd_df2 = pd.DataFrame(merge_df2)
        ds_df1 = ds.DataStore.from_df(pd.DataFrame(merge_df1))
        ds_df2 = ds.DataStore.from_df(pd.DataFrame(merge_df2))

        pd_result = pd.merge(pd_df1, pd_df2, on='key', how='left')
        ds_result = ds.merge(ds_df1, ds_df2, on='key', how='left')

        # Handle NaN comparison
        pd_vals = pd_result.fillna(-999).values
        ds_vals = ds_result.to_pandas().fillna(-999).values
        np.testing.assert_array_equal(ds_vals, pd_vals)

    def test_concat_vertical(self, merge_df1):
        """Test concat() vertical concatenation."""
        pd_df = pd.DataFrame(merge_df1)
        ds_df = ds.DataStore.from_df(pd.DataFrame(merge_df1))

        pd_result = pd.concat([pd_df, pd_df], ignore_index=True)
        ds_result = ds.concat([ds_df, ds_df], ignore_index=True)

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 13: Unique and Duplicates
# ============================================================================


class TestUniqueDuplicates:
    """Tests for unique and duplicate operations."""

    def test_unique(self, dup_df):
        """Test unique() values."""
        pd_df = pd.DataFrame(dup_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dup_df))

        pd_result = sorted(pd_df['A'].unique().tolist())
        ds_result = sorted(ds_df['A'].unique().tolist())

        assert ds_result == pd_result

    def test_nunique(self, dup_df):
        """Test nunique() count."""
        pd_df = pd.DataFrame(dup_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dup_df))

        pd_result = pd_df['A'].nunique()
        ds_result = ds_df['A'].nunique()

        assert ds_result == pd_result

    def test_drop_duplicates(self, dup_df):
        """Test drop_duplicates()."""
        pd_df = pd.DataFrame(dup_df)
        ds_df = ds.DataStore.from_df(pd.DataFrame(dup_df))

        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Category 14: to_pandas() API Consistency
# ============================================================================


class TestToPandasAPI:
    """Tests for to_pandas() API consistency with Polars/Dask conventions."""

    def test_datastore_to_pandas(self, simple_df):
        """Test DataStore.to_pandas() method."""
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))
        pd_df = pd.DataFrame(simple_df)

        result = ds_df.to_pandas()
        assert isinstance(result, pd.DataFrame)
        np.testing.assert_array_equal(result.values, pd_df.values)

    def test_column_expr_to_pandas(self, simple_df):
        """Test ColumnExpr.to_pandas() method."""
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))
        pd_df = pd.DataFrame(simple_df)

        result = ds_df['A'].to_pandas()
        assert isinstance(result, pd.Series)
        np.testing.assert_array_equal(result.values, pd_df['A'].values)

    def test_lazy_series_to_pandas(self, simple_df):
        """Test LazySeries.to_pandas() method."""
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))
        pd_df = pd.DataFrame(simple_df)

        # value_counts returns LazySeries
        result = ds_df['C'].value_counts().to_pandas()
        assert isinstance(result, pd.Series)

    def test_to_pandas_after_operations(self, simple_df):
        """Test to_pandas() after chained operations."""
        ds_df = ds.DataStore.from_df(pd.DataFrame(simple_df))
        pd_df = pd.DataFrame(simple_df)

        ds_result = ds_df[ds_df['A'] > 2].to_pandas()
        pd_result = pd_df[pd_df['A'] > 2]

        np.testing.assert_array_equal(ds_result.values, pd_result.values)
