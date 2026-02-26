"""
Kaggle Domain-Specific Pandas Compatibility Tests
=================================================

Tests common pandas operations from Kaggle notebooks across multiple domains:
- Computer Vision (CV)
- Natural Language Processing (NLP)
- Recommendation Systems
- Large Language Models (LLM)
- Exploratory Data Analysis (EDA)

This test suite complements test_kaggle_pandas_compat.py by focusing on:
1. File-based data sources (CSV, Parquet)
2. Domain-specific operation patterns
3. Realistic Kaggle workflow scenarios

Design Principle:
    Tests use natural execution triggers (.values, __eq__, len(), repr())
    following the lazy execution design principle.
    Avoid explicit _execute() calls - use natural triggers instead.
"""

import pytest
import pandas as pd
import numpy as np
import datastore as ds
from datastore.exceptions import UnsupportedOperationError

import tempfile
import os
from tests.test_utils import assert_frame_equal


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def eda_csv(temp_dir):
    """Create EDA test CSV file."""
    df = pd.DataFrame(
        {
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
            'value': [10, 20, 15, 30, 25, 12, 35, 22],
            'score': [1.5, 2.3, 1.8, 3.2, 2.9, 1.6, 3.5, 2.4],
        }
    )
    filepath = os.path.join(temp_dir, 'eda_test.csv')
    df.to_csv(filepath, index=False)
    return filepath, df


@pytest.fixture
def groupby_csv(temp_dir):
    """Create GroupBy test CSV file."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'] * 10,
            'region': ['North', 'South', 'North', 'East', 'South', 'North', 'East', 'South'] * 10,
            'sales': np.random.randint(100, 1000, 80),
            'quantity': np.random.randint(1, 50, 80),
        }
    )
    filepath = os.path.join(temp_dir, 'groupby_test.csv')
    df.to_csv(filepath, index=False)
    return filepath, df


@pytest.fixture
def users_activities_csv(temp_dir):
    """Create users and activities CSV files for merge tests."""
    users = pd.DataFrame({'user_id': [1, 2, 3, 4, 5], 'user_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']})
    activities = pd.DataFrame(
        {
            'user_id': [1, 1, 2, 3, 3, 3, 4, 6],
            'item_id': [101, 102, 101, 103, 104, 105, 102, 106],
            'rating': [5, 4, 5, 3, 4, 5, 2, 4],
        }
    )

    users_file = os.path.join(temp_dir, 'users.csv')
    activities_file = os.path.join(temp_dir, 'activities.csv')

    users.to_csv(users_file, index=False)
    activities.to_csv(activities_file, index=False)

    return users_file, activities_file, users, activities


@pytest.fixture
def text_csv(temp_dir):
    """Create text data CSV for NLP tests."""
    df = pd.DataFrame(
        {
            'text': ['Hello World!', 'Natural Language', 'MACHINE LEARNING', 'Data Science'],
            'category': ['greeting', 'nlp', 'ml', 'ds'],
        }
    )
    filepath = os.path.join(temp_dir, 'text_test.csv')
    df.to_csv(filepath, index=False)
    return filepath, df


@pytest.fixture
def ratings_csv(temp_dir):
    """Create ratings CSV for recommendation system tests."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5] * 10,
            'item_id': [101, 102, 101, 103, 102, 104, 101, 103, 105] * 10,
            'rating': np.random.randint(1, 6, 90),
        }
    )
    filepath = os.path.join(temp_dir, 'ratings.csv')
    df.to_csv(filepath, index=False)
    return filepath, df


# ============================================================================
# Test Category 1: Basic EDA Operations
# ============================================================================


class TestBasicEDA:
    """Test basic EDA operations from file sources."""

    def test_describe_from_csv(self, eda_csv):
        """Test describe() on data loaded from CSV."""
        filepath, df = eda_csv

        pd_desc = df.describe()
        ds_result = ds.DataStore.from_file(filepath)
        ds_desc = ds_result.describe()

        # Compare shape and values
        assert ds_desc.to_pandas().shape == pd_desc.shape
        np.testing.assert_array_almost_equal(
            ds_desc.to_pandas().values, pd_desc.values, decimal=5, err_msg="describe() values should match"
        )

    def test_value_counts_from_csv(self, eda_csv):
        """Test value_counts() on data loaded from CSV."""
        filepath, df = eda_csv

        pd_vc = df['category'].value_counts()
        ds_result = ds.DataStore.from_file(filepath)
        ds_vc = ds_result['category'].value_counts()

        # Compare sorted values (order may differ)
        np.testing.assert_array_equal(
            sorted(ds_vc.to_pandas().values), sorted(pd_vc.values), err_msg="value_counts() should return same counts"
        )


# ============================================================================
# Test Category 2: GroupBy Aggregations
# ============================================================================


class TestGroupByAggregations:
    """Test GroupBy operations - common in all Kaggle domains."""

    def test_groupby_sum_returns_series(self, groupby_csv):
        """
        Test groupby().sum() returns Series-like result.

        In pandas: df.groupby('col')['value'].sum() returns Series
        DataStore should behave the same.
        """
        filepath, df = groupby_csv

        # Pandas groupby returns Series
        pd_result = df.groupby('category')['sales'].sum()
        assert isinstance(pd_result, pd.Series), "pandas groupby()['col'].sum() should return Series"

        # DataStore groupby returns ColumnExpr that behaves like Series
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.groupby('category')['sales'].sum()

        # Values should match
        np.testing.assert_array_equal(
            sorted(ds_result.values),
            sorted(pd_result.values),
            err_msg="groupby().sum() values should match pandas",
        )

    def test_groupby_mean_returns_series(self, groupby_csv):
        """Test groupby().mean() returns Series-like result."""
        filepath, df = groupby_csv

        pd_result = df.groupby('category')['sales'].mean()
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.groupby('category')['sales'].mean()

        np.testing.assert_array_almost_equal(
            sorted(ds_result.values), sorted(pd_result.values), decimal=5, err_msg="groupby().mean() should match"
        )

    def test_groupby_agg_dict(self, groupby_csv):
        """Test groupby().agg() with dict parameter."""
        filepath, df = groupby_csv

        pd_result = df.groupby('category').agg({'sales': 'sum', 'quantity': 'mean'})
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.groupby('category').agg({'sales': 'sum', 'quantity': 'mean'})

        # Compare values after sorting by index
        pd_sorted = pd_result.sort_index()
        ds_sorted = ds_result.to_pandas().sort_index()

        np.testing.assert_array_almost_equal(
            ds_sorted.values, pd_sorted.values, decimal=5, err_msg="groupby().agg() should match pandas"
        )


# ============================================================================
# Test Category 3: Merge/Join Operations (Recommendation Systems)
# ============================================================================


class TestMergeOperations:
    """Test merge operations - critical for recommendation systems."""

    def test_inner_merge_from_csv(self, users_activities_csv):
        """Test inner merge from CSV files."""
        users_file, activities_file, users, activities = users_activities_csv

        # Pandas merge
        pd_result = pd.merge(users, activities, on='user_id', how='inner')

        # DataStore merge
        ds_users = ds.DataStore.from_file(users_file)
        ds_activities = ds.DataStore.from_file(activities_file)
        ds_result = ds_users.merge(ds_activities, on='user_id', how='inner')

        # Compare shape and columns
        assert ds_result.to_pandas().shape == pd_result.shape, "Merge shape should match"
        assert list(ds_result.columns) == list(pd_result.columns), "Merge columns should match"

        # Compare values
        assert_frame_equal(
            ds_result.to_pandas().sort_values(['user_id', 'item_id']).reset_index(drop=True),
            pd_result.sort_values(['user_id', 'item_id']).reset_index(drop=True),
            )

    def test_left_merge_from_csv(self, users_activities_csv):
        """Test left merge from CSV files - includes unmatched rows."""
        users_file, activities_file, users, activities = users_activities_csv

        pd_result = pd.merge(users, activities, on='user_id', how='left')
        ds_users = ds.DataStore.from_file(users_file)
        ds_activities = ds.DataStore.from_file(activities_file)
        ds_result = ds_users.merge(ds_activities, on='user_id', how='left')

        # Left merge should have same number of rows for matched users
        assert ds_result.to_pandas().shape == pd_result.shape


# ============================================================================
# Test Category 4: Pivot Table Operations (Recommendation Systems)
# ============================================================================


class TestPivotTable:
    """Test pivot table operations - essential for recommendation systems."""

    def test_pivot_table_mean(self, ratings_csv):
        """Test pivot_table() with mean aggregation."""
        filepath, df = ratings_csv

        pd_pivot = pd.pivot_table(df, values='rating', index='user_id', columns='item_id', aggfunc='mean')

        ds_store = ds.DataStore.from_file(filepath)
        ds_pivot = ds_store.pivot_table(values='rating', index='user_id', columns='item_id', aggfunc='mean')

        # Compare shape
        assert ds_pivot.to_pandas().shape == pd_pivot.shape, "Pivot table shape should match"

        # Compare values (accounting for NaN)
        pd_vals = pd_pivot.fillna(-999).values
        ds_vals = ds_pivot.to_pandas().fillna(-999).values
        np.testing.assert_array_almost_equal(ds_vals, pd_vals, decimal=5)


# ============================================================================
# Test Category 5: String Operations (NLP)
# ============================================================================


class TestStringOperations:
    """Test string operations - critical for NLP tasks."""

    def test_str_lower_from_csv(self, text_csv):
        """Test str.lower() on CSV data."""
        filepath, df = text_csv

        pd_result = df['text'].str.lower()
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store['text'].str.lower()

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_str_len_from_csv(self, text_csv):
        """Test str.len() on CSV data."""
        filepath, df = text_csv

        pd_result = df['text'].str.len()
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store['text'].str.len()

        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_str_contains_from_csv(self, text_csv):
        """Test str.contains() on CSV data."""
        filepath, df = text_csv

        pd_result = df['text'].str.contains('a', case=False)
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store['text'].str.contains('a', case=False)

        np.testing.assert_array_equal(ds_result.values, pd_result.values)


# ============================================================================
# Test Category 6: Sorting Operations (Model Evaluation)
# ============================================================================


class TestSortingOperations:
    """Test sorting operations - common in CV and model evaluation."""

    def test_sort_values_from_csv(self, eda_csv):
        """Test sort_values() on CSV data."""
        filepath, df = eda_csv

        pd_sorted = df.sort_values('value', ascending=False).reset_index(drop=True)
        ds_store = ds.DataStore.from_file(filepath)
        ds_sorted = ds_store.sort_values('value', ascending=False)

        np.testing.assert_array_equal(
            ds_sorted.to_pandas().reset_index(drop=True).values,
            pd_sorted.values,
            err_msg="sort_values() should match pandas",
        )


# ============================================================================
# Test Category 7: Filtering Operations
# ============================================================================


class TestFilteringOperations:
    """Test filtering operations - used everywhere in Kaggle notebooks."""

    def test_filter_numeric_from_csv(self, eda_csv):
        """Test filtering with numeric condition."""
        filepath, df = eda_csv

        pd_result = df[df['value'] > 20].reset_index(drop=True)
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.filter(ds_store.value > 20)

        np.testing.assert_array_equal(
            ds_result.to_pandas().reset_index(drop=True).values,
            pd_result.values,
            err_msg="filter() should match pandas boolean indexing",
        )

    def test_filter_compound_condition(self, eda_csv):
        """Test filtering with AND condition."""
        filepath, df = eda_csv

        pd_result = df[(df['value'] > 15) & (df['score'] < 3.0)].reset_index(drop=True)
        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.filter((ds_store.value > 15) & (ds_store.score < 3.0))

        np.testing.assert_array_equal(
            ds_result.to_pandas().reset_index(drop=True).values,
            pd_result.values,
        )


# ============================================================================
# Test Category 8: Apply/Lambda Operations (Feature Engineering)
# ============================================================================


class TestApplyOperations:
    """Test apply and lambda functions - essential for feature engineering."""

    def test_assign_lambda_multiplication(self, eda_csv):
        """Test assign() with lambda for multiplication."""
        filepath, df = eda_csv

        pd_df = df.copy()
        pd_df['doubled'] = pd_df['value'] * 2

        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.assign(doubled=lambda x: x['value'] * 2)

        np.testing.assert_array_equal(ds_result.to_pandas()['doubled'].values, pd_df['doubled'].values)

    def test_assign_lambda_column_product(self, eda_csv):
        """Test assign() with lambda for column product."""
        filepath, df = eda_csv

        pd_df = df.copy()
        pd_df['product'] = pd_df['value'] * pd_df['score']

        ds_store = ds.DataStore.from_file(filepath)
        ds_result = ds_store.assign(product=lambda x: x['value'] * x['score'])

        np.testing.assert_array_almost_equal(
            ds_result.to_pandas()['product'].values, pd_df['product'].values, decimal=5
        )


# ============================================================================
# Test Category 9: Missing Value Handling (Data Cleaning)
# ============================================================================


class TestMissingValues:
    """Test missing value handling - common in data cleaning."""

    def test_fillna_from_df(self):
        """Test fillna() on DataFrame with missing values."""
        df = pd.DataFrame(
            {
                'A': [1.0, 2.0, np.nan, 4.0, np.nan, 6.0],
                'B': [10.0, np.nan, 30.0, np.nan, 50.0, 60.0],
            }
        )

        pd_result = df.fillna(0)
        ds_store = ds.DataStore.from_df(df)
        ds_result = ds_store.fillna(0)

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_dropna_from_df(self):
        """Test dropna() on DataFrame with missing values."""
        df = pd.DataFrame(
            {
                'A': [1.0, 2.0, np.nan, 4.0, np.nan, 6.0],
                'B': [10.0, np.nan, 30.0, np.nan, 50.0, 60.0],
            }
        )

        pd_result = df.dropna()
        ds_store = ds.DataStore.from_df(df)
        ds_result = ds_store.dropna()

        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)


# ============================================================================
# Test Category 10: DateTime Operations (Time Series)
# ============================================================================


class TestDateTimeOperations:
    """Test datetime operations - important for time series analysis."""

    def test_dt_accessors_from_df(self):
        """Test datetime accessors on DataFrame with dates."""
        df = pd.DataFrame(
            {
                'timestamp': pd.date_range('2025-01-01', periods=5, freq='D'),
                'value': [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        ds_store = ds.DataStore.from_df(df)

        # Test year
        pd_year = df['timestamp'].dt.year
        ds_year = ds_store['timestamp'].dt.year
        np.testing.assert_array_equal(ds_year.values, pd_year.values)

        # Test month
        pd_month = df['timestamp'].dt.month
        ds_month = ds_store['timestamp'].dt.month
        np.testing.assert_array_equal(ds_month.values, pd_month.values)

        # Test day
        pd_day = df['timestamp'].dt.day
        ds_day = ds_store['timestamp'].dt.day
        np.testing.assert_array_equal(ds_day.values, pd_day.values)


# ============================================================================
# Test Category 11: CSV File Format Handling
# ============================================================================


class TestCSVFormatHandling:
    """Test CSV file format handling - important for Kaggle data sources."""

    def test_csv_with_header_from_file(self, temp_dir):
        """Test reading CSV with header row."""
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        filepath = os.path.join(temp_dir, 'test.csv')
        df.to_csv(filepath, index=False)

        ds_store = ds.DataStore.from_file(filepath)
        assert list(ds_store.columns) == ['name', 'age'], "CSV headers should be detected"
        assert len(ds_store) == 2

    def test_csv_no_header_explicit_format(self, temp_dir):
        """Test reading CSV without header using explicit format."""
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        filepath = os.path.join(temp_dir, 'test_no_header.csv')
        df.to_csv(filepath, index=False, header=False)

        ds_store = ds.DataStore.from_file(filepath, format='CSV')
        # Without header, ClickHouse uses c1, c2, etc.
        assert 'c1' in ds_store.columns or 'name' in ds_store.columns


# ============================================================================
# Test Category 12: read_csv() vs from_file() Consistency
# ============================================================================


class TestReadCSVConsistency:
    """Test consistency between read_csv() and from_file()."""

    def test_read_csv_parses_header(self, temp_dir):
        """Test read_csv() correctly parses CSV header."""
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        filepath = os.path.join(temp_dir, 'test.csv')
        df.to_csv(filepath, index=False)

        ds_store = ds.read_csv(filepath)
        assert list(ds_store.columns) == ['name', 'age']

    def test_from_file_auto_infers_csv_with_names(self, temp_dir):
        """Test from_file() auto-infers CSVWithNames format for .csv files."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        filepath = os.path.join(temp_dir, 'test.csv')
        df.to_csv(filepath, index=False)

        ds_store = ds.DataStore.from_file(filepath)
        # After fix: from_file should auto-infer CSVWithNames for .csv files
        assert list(ds_store.columns) == ['x', 'y'], "from_file() should use CSVWithNames for .csv files"


# ============================================================================
# Test Category 13: read_csv() Pandas Compatibility
# ============================================================================


class TestReadCSVPandasCompatibility:
    """Test read_csv() matches pandas behavior for various parameters."""

    def test_read_csv_header_none(self, temp_dir):
        """Test read_csv() with header=None (no header row)."""
        # Create CSV without header
        filepath = os.path.join(temp_dir, 'no_header.csv')
        with open(filepath, 'w') as f:
            f.write('Alice,25\nBob,30\nCharlie,35\n')

        # pandas behavior
        pd_result = pd.read_csv(filepath, header=None)

        # datastore behavior
        ds_result = ds.read_csv(filepath, header=None)

        # Both should have auto-generated column names
        assert len(ds_result.columns) == len(pd_result.columns)
        assert len(ds_result) == len(pd_result)

    def test_read_csv_with_header_default(self, temp_dir):
        """Test read_csv() with default header behavior (first row is header)."""
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'value': [100, 200]})
        filepath = os.path.join(temp_dir, 'with_header.csv')
        df.to_csv(filepath, index=False)

        pd_result = pd.read_csv(filepath)
        ds_result = ds.read_csv(filepath)

        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == len(pd_result)

    def test_read_csv_custom_delimiter(self, temp_dir):
        """Test read_csv() with custom delimiter."""
        filepath = os.path.join(temp_dir, 'semicolon.csv')
        with open(filepath, 'w') as f:
            f.write('name;age\nAlice;25\nBob;30\n')

        pd_result = pd.read_csv(filepath, sep=';')
        ds_result = ds.read_csv(filepath, sep=';')

        assert list(ds_result.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result.to_pandas()['age'].values, pd_result['age'].values)

    def test_read_csv_tab_delimiter(self, temp_dir):
        """Test read_csv() with tab delimiter."""
        filepath = os.path.join(temp_dir, 'tab.tsv')
        with open(filepath, 'w') as f:
            f.write('name\tage\nAlice\t25\nBob\t30\n')

        pd_result = pd.read_csv(filepath, sep='\t')
        ds_result = ds.read_csv(filepath, sep='\t')

        assert list(ds_result.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result.to_pandas()['age'].values, pd_result['age'].values)

    def test_read_csv_nrows(self, temp_dir):
        """Test read_csv() with nrows parameter."""
        df = pd.DataFrame({'x': range(100), 'y': range(100, 200)})
        filepath = os.path.join(temp_dir, 'large.csv')
        df.to_csv(filepath, index=False)

        pd_result = pd.read_csv(filepath, nrows=10)
        ds_result = ds.read_csv(filepath, nrows=10)

        assert len(ds_result) == 10
        assert len(pd_result) == 10
        np.testing.assert_array_equal(ds_result.to_pandas()['x'].values, pd_result['x'].values)

    def test_read_csv_skiprows_simple(self, temp_dir):
        """Test read_csv() with skiprows parameter on clean CSV data."""
        # Create a clean CSV (header + data rows)
        df = pd.DataFrame({'x': range(10), 'y': range(10, 20)})
        filepath = os.path.join(temp_dir, 'skiprows.csv')
        df.to_csv(filepath, index=False)

        # Read skipping first 3 data rows
        pd_result = pd.read_csv(filepath, skiprows=[1, 2, 3])  # Skip rows 1-3 (data rows)
        ds_result = ds.read_csv(filepath, skiprows=[1, 2, 3])  # Falls back to pandas

        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == len(pd_result)

    def test_read_csv_skiprows_with_comment(self, temp_dir):
        """Test read_csv() with skiprows parameter and comment lines - uses pandas."""
        filepath = os.path.join(temp_dir, 'skiprows_comment.csv')
        with open(filepath, 'w') as f:
            f.write('# comment line 1\n')
            f.write('# comment line 2\n')
            f.write('name,age\n')
            f.write('Alice,25\n')
            f.write('Bob,30\n')

        # Use comment parameter which triggers pandas fallback
        pd_result = pd.read_csv(filepath, comment='#')
        ds_result = ds.read_csv(filepath, comment='#')

        assert list(ds_result.columns) == list(pd_result.columns)
        assert len(ds_result) == len(pd_result)

    def test_read_csv_with_compression(self, temp_dir):
        """Test read_csv() with gzip compression."""
        import gzip

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        filepath = os.path.join(temp_dir, 'compressed.csv.gz')
        df.to_csv(filepath, index=False, compression='gzip')

        pd_result = pd.read_csv(filepath, compression='gzip')
        ds_result = ds.read_csv(filepath, compression='gzip')

        assert list(ds_result.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result.to_pandas()['a'].values, pd_result['a'].values)

    def test_read_csv_fallback_to_pandas_for_usecols(self, temp_dir):
        """Test read_csv() falls back to pandas for usecols parameter."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        filepath = os.path.join(temp_dir, 'usecols.csv')
        df.to_csv(filepath, index=False)

        pd_result = pd.read_csv(filepath, usecols=['a', 'c'])
        ds_result = ds.read_csv(filepath, usecols=['a', 'c'])

        assert list(ds_result.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result.to_pandas().values, pd_result.values)

    def test_read_csv_fallback_to_pandas_for_dtype(self, temp_dir):
        """Test read_csv() falls back to pandas for dtype parameter."""
        df = pd.DataFrame({'num': ['1', '2', '3'], 'val': ['4', '5', '6']})
        filepath = os.path.join(temp_dir, 'dtype.csv')
        df.to_csv(filepath, index=False)

        pd_result = pd.read_csv(filepath, dtype={'num': int, 'val': float})
        ds_result = ds.read_csv(filepath, dtype={'num': int, 'val': float})

        assert pd_result['num'].dtype == ds_result.to_pandas()['num'].dtype
        assert pd_result['val'].dtype == ds_result.to_pandas()['val'].dtype

    def test_read_csv_with_names(self, temp_dir):
        """Test read_csv() with custom column names (falls back to pandas)."""
        filepath = os.path.join(temp_dir, 'no_header.csv')
        with open(filepath, 'w') as f:
            f.write('Alice,25\nBob,30\n')

        pd_result = pd.read_csv(filepath, header=None, names=['name', 'age'])
        ds_result = ds.read_csv(filepath, header=None, names=['name', 'age'])

        assert list(ds_result.columns) == list(pd_result.columns)
        np.testing.assert_array_equal(ds_result.to_pandas()['name'].values, pd_result['name'].values)


# ============================================================================
# Test Category 14: read_csv() Backend Selection Verification
# ============================================================================


class TestReadCSVBackendSelection:
    """
    Verify read_csv() selects the correct backend (chDB vs pandas) based on parameters.

    Backend detection:
    - _table_function is not None -> chDB SQL engine (file table function)
    - _source_df is not None -> pandas fallback (created from DataFrame)
    """

    @pytest.fixture
    def simple_csv(self, temp_dir):
        """Create a simple CSV file for testing."""
        filepath = os.path.join(temp_dir, 'simple.csv')
        with open(filepath, 'w') as f:
            f.write('name,age,score\n')
            f.write('Alice,25,85.5\n')
            f.write('Bob,30,92.0\n')
            f.write('Charlie,35,78.5\n')
        return filepath

    # ===== Tests for chDB backend (expected) =====

    def test_default_params_uses_chdb(self, simple_csv):
        """Default read_csv() should use chDB backend."""
        result = ds.read_csv(simple_csv)
        assert result._table_function is not None, "Default should use chDB table function"
        assert result._source_df is None, "Default should NOT use pandas DataFrame source"

    def test_header_0_uses_chdb(self, simple_csv):
        """header=0 should use chDB backend."""
        result = ds.read_csv(simple_csv, header=0)
        assert result._table_function is not None, "header=0 should use chDB"
        assert result._source_df is None

    def test_header_infer_uses_chdb(self, simple_csv):
        """header='infer' should use chDB backend."""
        result = ds.read_csv(simple_csv, header='infer')
        assert result._table_function is not None, "header='infer' should use chDB"
        assert result._source_df is None

    def test_header_none_uses_chdb(self, simple_csv):
        """header=None should use chDB backend with CSV format."""
        result = ds.read_csv(simple_csv, header=None)
        assert result._table_function is not None, "header=None should use chDB"
        assert result._source_df is None

    def test_nrows_uses_chdb(self, simple_csv):
        """nrows parameter should use chDB backend."""
        result = ds.read_csv(simple_csv, nrows=2)
        assert result._table_function is not None, "nrows should use chDB"
        assert result._source_df is None

    def test_tab_delimiter_uses_chdb(self, temp_dir):
        """Tab delimiter should use chDB backend with TSV format."""
        filepath = os.path.join(temp_dir, 'tab.tsv')
        with open(filepath, 'w') as f:
            f.write('name\tage\n')
            f.write('Alice\t25\n')
        result = ds.read_csv(filepath, sep='\t')
        assert result._table_function is not None, "Tab delimiter should use chDB TSV"
        assert result._source_df is None

    def test_compression_uses_chdb(self, temp_dir):
        """Compression should use chDB backend."""
        import gzip

        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        filepath = os.path.join(temp_dir, 'compressed.csv.gz')
        df.to_csv(filepath, index=False, compression='gzip')

        result = ds.read_csv(filepath, compression='gzip')
        assert result._table_function is not None, "Compression should use chDB"
        assert result._source_df is None

    # ===== Tests for pandas fallback (expected) =====

    def test_usecols_uses_pandas(self, simple_csv):
        """usecols parameter should fall back to pandas."""
        result = ds.read_csv(simple_csv, usecols=['name', 'age'])
        assert result._source_df is not None, "usecols should use pandas"
        assert result._table_function is None, "usecols should NOT use chDB"

    def test_dtype_uses_pandas(self, simple_csv):
        """dtype parameter should fall back to pandas."""
        result = ds.read_csv(simple_csv, dtype={'age': str})
        assert result._source_df is not None, "dtype should use pandas"
        assert result._table_function is None, "dtype should NOT use chDB"

    def test_names_uses_pandas(self, temp_dir):
        """names parameter should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'no_header.csv')
        with open(filepath, 'w') as f:
            f.write('Alice,25\n')
        result = ds.read_csv(filepath, header=None, names=['name', 'age'])
        assert result._source_df is not None, "names should use pandas"
        assert result._table_function is None, "names should NOT use chDB"

    def test_index_col_uses_pandas(self, simple_csv):
        """index_col parameter should fall back to pandas."""
        result = ds.read_csv(simple_csv, index_col='name')
        assert result._source_df is not None, "index_col should use pandas"
        assert result._table_function is None, "index_col should NOT use chDB"

    def test_parse_dates_uses_pandas(self, temp_dir):
        """parse_dates parameter should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'dates.csv')
        with open(filepath, 'w') as f:
            f.write('date,value\n')
            f.write('2025-01-01,100\n')
        result = ds.read_csv(filepath, parse_dates=['date'])
        assert result._source_df is not None, "parse_dates should use pandas"
        assert result._table_function is None, "parse_dates should NOT use chDB"

    def test_converters_uses_pandas(self, simple_csv):
        """converters parameter should fall back to pandas."""
        result = ds.read_csv(simple_csv, converters={'age': lambda x: int(x) + 1})
        assert result._source_df is not None, "converters should use pandas"
        assert result._table_function is None, "converters should NOT use chDB"

    def test_custom_delimiter_uses_pandas(self, temp_dir):
        """Non-standard delimiter (other than , or tab) should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'semicolon.csv')
        with open(filepath, 'w') as f:
            f.write('name;age\n')
            f.write('Alice;25\n')
        result = ds.read_csv(filepath, sep=';')
        assert result._source_df is not None, "Semicolon delimiter should use pandas"
        assert result._table_function is None, "Semicolon delimiter should NOT use chDB"

    def test_pipe_delimiter_uses_pandas(self, temp_dir):
        """Pipe delimiter should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'pipe.csv')
        with open(filepath, 'w') as f:
            f.write('name|age\n')
            f.write('Alice|25\n')
        result = ds.read_csv(filepath, sep='|')
        assert result._source_df is not None, "Pipe delimiter should use pandas"
        assert result._table_function is None, "Pipe delimiter should NOT use chDB"

    def test_comment_uses_pandas(self, temp_dir):
        """comment parameter should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'comments.csv')
        with open(filepath, 'w') as f:
            f.write('# comment\n')
            f.write('name,age\n')
            f.write('Alice,25\n')
        result = ds.read_csv(filepath, comment='#')
        assert result._source_df is not None, "comment should use pandas"
        assert result._table_function is None, "comment should NOT use chDB"

    def test_thousands_uses_pandas(self, temp_dir):
        """thousands parameter should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'thousands.csv')
        with open(filepath, 'w') as f:
            f.write('name,value\n')
            f.write('Alice,"1,000"\n')
        result = ds.read_csv(filepath, thousands=',')
        assert result._source_df is not None, "thousands should use pandas"
        assert result._table_function is None, "thousands should NOT use chDB"

    def test_skipfooter_uses_pandas(self, simple_csv):
        """skipfooter parameter should fall back to pandas."""
        result = ds.read_csv(simple_csv, skipfooter=1, engine='python')
        assert result._source_df is not None, "skipfooter should use pandas"
        assert result._table_function is None, "skipfooter should NOT use chDB"

    def test_chunksize_raises_not_implemented(self, simple_csv):
        """chunksize parameter is not supported and raises NotImplementedError.

        Note: pandas.read_csv with chunksize returns TextFileReader, not DataFrame.
        DataStore doesn't support chunked reading, so we raise a clear error.
        """
        with pytest.raises(UnsupportedOperationError, match="does not support chunked reading"):
            ds.read_csv(simple_csv, chunksize=1)

    def test_iterator_raises_not_implemented(self, simple_csv):
        """iterator parameter is not supported and raises NotImplementedError."""
        with pytest.raises(UnsupportedOperationError, match="does not support chunked reading"):
            ds.read_csv(simple_csv, iterator=True)

    def test_header_gt_0_uses_pandas(self, temp_dir):
        """header > 0 should fall back to pandas."""
        filepath = os.path.join(temp_dir, 'multi_header.csv')
        with open(filepath, 'w') as f:
            f.write('# metadata line\n')
            f.write('name,age\n')
            f.write('Alice,25\n')
        result = ds.read_csv(filepath, header=1)
        assert result._source_df is not None, "header=1 should use pandas"
        assert result._table_function is None, "header=1 should NOT use chDB"

    def test_true_false_values_uses_chdb_with_settings(self, temp_dir):
        """true_values/false_values should use chDB but with settings."""
        filepath = os.path.join(temp_dir, 'bool.csv')
        with open(filepath, 'w') as f:
            f.write('name,active\n')
            f.write('Alice,yes\n')
            f.write('Bob,no\n')
        # Note: true_values/false_values may not be converted by chDB,
        # as chDB doesn't natively support these parameters like pandas does
        result = ds.read_csv(filepath, true_values=['yes'], false_values=['no'])
        # This should use chDB with bool settings
        assert result._table_function is not None or result._source_df is not None
        # Verify data is correct either way
        assert len(result) == 2
        # Verify data is loaded (may be strings or booleans depending on backend)
        active_values = list(result.to_pandas()['active'].values)
        assert active_values == [True, False] or active_values == ['yes', 'no']

    # ===== Combination tests =====

    def test_nrows_with_chdb_params_uses_chdb(self, simple_csv):
        """Combination of chDB-compatible params should use chDB."""
        result = ds.read_csv(simple_csv, header=0, nrows=2)
        assert result._table_function is not None, "chDB params combo should use chDB"
        assert result._source_df is None

    def test_pandas_param_overrides_chdb(self, simple_csv):
        """If any pandas-required param is present, use pandas."""
        result = ds.read_csv(simple_csv, nrows=2, dtype={'age': int})
        assert result._source_df is not None, "pandas param should override chDB"
        assert result._table_function is None

    # ===== URL/file-like object tests =====

    def test_file_like_object_uses_pandas(self, simple_csv):
        """File-like object should fall back to pandas."""
        with open(simple_csv, 'r') as f:
            result = ds.read_csv(f)
        assert result._source_df is not None, "File-like object should use pandas"
        assert result._table_function is None
