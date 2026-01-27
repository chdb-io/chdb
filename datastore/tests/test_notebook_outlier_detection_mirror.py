"""
Mirror Test for Outlier Detection Notebook
==========================================

Tests pandas operations found in the Outlier Detection notebook,
comparing DataStore behavior with pandas for API consistency.

Operations tested from the notebook:
- DataFrame creation with numeric columns
- head() for basic inspection
- Column selection with list notation df[['col1', 'col2']]
- Chained operations: df[['col1', 'col2']].head()
- describe() for numeric statistics
- map() for value replacement
- shape property
- Column assignment with new values

Design Principle:
    Tests use natural execution triggers following the lazy execution design.
    Mirror Code Pattern: DataStore and pandas operations must be identical.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


# ============================================================================
# Fixtures with Outlier Detection data structure
# ============================================================================


@pytest.fixture
def outlier_df():
    """
    Sample data mimicking outlier detection dataset structure.

    Note: DataFrame is created from a 2D numpy array. DataStore automatically
    handles the memory layout issue with chDB's Python() table function.
    """
    np.random.seed(42)

    # Generate normally distributed data (like in the notebook)
    data = np.random.normal(size=(100, 2))

    # Introduce outliers into the data
    outliers = np.random.uniform(low=-10, high=10, size=(10, 2))

    # Combine the data
    combined_data = np.vstack([data, outliers])

    # Create a binary label indicating whether each point is an outlier
    labels = np.zeros(110)
    labels[100:] = 1  # The last 10 points are outliers

    # Create a DataFrame from the data
    df = pd.DataFrame(combined_data, columns=['Feature 1', 'Feature 2'])
    df['Outlier'] = labels

    return df


@pytest.fixture
def outlier_df_with_predictions(outlier_df):
    """Dataset with prediction column added (as in notebook after model prediction)."""
    df = outlier_df.copy()
    # Simulate model predictions (in notebook: clf.predict returns 1 or -1)
    np.random.seed(42)
    predictions = np.random.choice([1, -1], size=len(df), p=[0.9, 0.1])
    df['Predicted Outlier'] = predictions
    return df


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicInspection:
    """Tests for basic DataFrame inspection operations from the notebook."""

    def test_head_default(self, outlier_df):
        """Test df.head() - from notebook cell showing first 5 rows."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df.head()
        ds_result = ds.head()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_with_n(self, outlier_df):
        """Test df.head(n) with custom n."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df.head(10)
        ds_result = ds.head(10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shape_property(self, outlier_df):
        """Test shape property."""
        ds = DataStore(outlier_df)
        assert ds.shape == outlier_df.shape

    def test_columns_values(self, outlier_df):
        """Test columns.values property."""
        ds = DataStore(outlier_df)
        np.testing.assert_array_equal(ds.columns.values, outlier_df.columns.values)


class TestColumnSelection:
    """Tests for column selection operations from the notebook."""

    def test_single_column_selection(self, outlier_df):
        """Test df['Feature 1'] - single column selection."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df['Feature 1']
        ds_result = ds['Feature 1']
        assert_series_equal(ds_result, pd_result)

    def test_multi_column_selection(self, outlier_df):
        """Test df[['Feature 1', 'Feature 2']] - from notebook model fitting."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[['Feature 1', 'Feature 2']]
        ds_result = ds[['Feature 1', 'Feature 2']]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_columns_then_head(self, outlier_df):
        """Test df[['Feature 1', 'Feature 2']].head() - chained operation."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[['Feature 1', 'Feature 2']].head()
        ds_result = ds[['Feature 1', 'Feature 2']].head()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_feature_selection_shape(self, outlier_df):
        """Test shape after column selection."""
        ds = DataStore(outlier_df)
        pd_selected = outlier_df[['Feature 1', 'Feature 2']]
        ds_selected = ds[['Feature 1', 'Feature 2']]
        assert ds_selected.shape == pd_selected.shape


class TestDescribeOperations:
    """Tests for describe() operations."""

    def test_describe_full_dataframe(self, outlier_df):
        """Test describe() on full DataFrame."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df.describe()
        ds_result = ds.describe()
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_describe_selected_columns(self, outlier_df):
        """Test describe() on selected columns - df[['Feature 1', 'Feature 2']].describe()."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[['Feature 1', 'Feature 2']].describe()
        ds_result = ds[['Feature 1', 'Feature 2']].describe()
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestMapOperation:
    """Tests for map() operation from the notebook."""

    def test_map_basic(self, outlier_df_with_predictions):
        """Test basic map operation."""
        ds = DataStore(outlier_df_with_predictions)
        pd_df = outlier_df_with_predictions.copy()

        # The notebook maps: 1 -> 0, -1 -> 1
        pd_result = pd_df['Predicted Outlier'].map({1: 0, -1: 1})
        ds_result = ds['Predicted Outlier'].map({1: 0, -1: 1})
        assert_series_equal(ds_result, pd_result)

    def test_map_and_assign(self, outlier_df_with_predictions):
        """Test map with column assignment - the pattern from the notebook."""
        pd_df = outlier_df_with_predictions.copy()
        ds_df = DataStore(outlier_df_with_predictions)

        # pandas operation
        pd_df['Predicted Outlier'] = pd_df['Predicted Outlier'].map({1: 0, -1: 1})

        # DataStore operation (mirror)
        ds_df['Predicted Outlier'] = ds_df['Predicted Outlier'].map({1: 0, -1: 1})

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestDataFrameCreation:
    """Tests for DataFrame creation patterns from the notebook."""

    def test_create_from_dict(self):
        """Test DataFrame creation from dict - most reliable pattern."""
        np.random.seed(42)
        data = np.random.normal(size=(50, 2))

        pd_df = pd.DataFrame({'Feature 1': data[:, 0], 'Feature 2': data[:, 1]})
        ds_df = DataStore({'Feature 1': data[:, 0], 'Feature 2': data[:, 1]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_create_with_additional_column(self):
        """Test creating DataFrame and adding a column."""
        np.random.seed(42)
        data = np.random.normal(size=(50, 2))

        pd_df = pd.DataFrame({'Feature 1': data[:, 0], 'Feature 2': data[:, 1]})
        pd_df['Label'] = np.zeros(50)

        ds_df = DataStore({'Feature 1': data[:, 0], 'Feature 2': data[:, 1]})
        ds_df['Label'] = np.zeros(50)

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestNumpyIntegration:
    """Tests for numpy integration patterns from the notebook."""

    def test_create_df_from_dict_with_numpy_columns(self):
        """Test DataFrame creation from dict with numpy array columns."""
        np.random.seed(42)
        data = np.random.normal(size=(20, 2))
        outliers = np.random.uniform(low=-10, high=10, size=(5, 2))
        combined = np.vstack([data, outliers])

        # Create from dict (column by column) - the reliable way
        pd_df = pd.DataFrame({'Feature 1': combined[:, 0], 'Feature 2': combined[:, 1]})
        ds_df = DataStore({'Feature 1': combined[:, 0], 'Feature 2': combined[:, 1]})

        assert_datastore_equals_pandas(ds_df, pd_df)

    def test_numpy_zeros_as_column(self):
        """Test adding numpy zeros as a column."""
        np.random.seed(42)
        data = np.random.normal(size=(25, 2))
        labels = np.zeros(25)
        labels[20:] = 1

        pd_df = pd.DataFrame({'Feature 1': data[:, 0], 'Feature 2': data[:, 1]})
        pd_df['Outlier'] = labels

        ds_df = DataStore({'Feature 1': data[:, 0], 'Feature 2': data[:, 1]})
        ds_df['Outlier'] = labels

        assert_datastore_equals_pandas(ds_df, pd_df)


class TestStatisticalOperations:
    """Tests for statistical operations that could be used with outlier detection."""

    def test_mean_per_column(self, outlier_df):
        """Test column means."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[['Feature 1', 'Feature 2']].mean()
        ds_result = ds[['Feature 1', 'Feature 2']].mean()
        assert_series_equal(ds_result, pd_result, rtol=1e-5)

    def test_std_per_column(self, outlier_df):
        """Test column standard deviations."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[['Feature 1', 'Feature 2']].std()
        ds_result = ds[['Feature 1', 'Feature 2']].std()
        assert_series_equal(ds_result, pd_result, rtol=1e-5)

    def test_min_max_per_column(self, outlier_df):
        """Test column min and max."""
        ds = DataStore(outlier_df)

        pd_min = outlier_df[['Feature 1', 'Feature 2']].min()
        ds_min = ds[['Feature 1', 'Feature 2']].min()
        assert_series_equal(ds_min, pd_min)

        pd_max = outlier_df[['Feature 1', 'Feature 2']].max()
        ds_max = ds[['Feature 1', 'Feature 2']].max()
        assert_series_equal(ds_max, pd_max)


class TestBooleanIndexing:
    """Tests for boolean indexing used in outlier analysis."""

    def test_filter_outliers(self, outlier_df):
        """Test filtering to get only outliers."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[outlier_df['Outlier'] == 1]
        ds_result = ds[ds['Outlier'] == 1]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_non_outliers(self, outlier_df):
        """Test filtering to get non-outliers."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df[outlier_df['Outlier'] == 0]
        ds_result = ds[ds['Outlier'] == 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_count_outliers(self, outlier_df):
        """Test counting outliers using boolean indexing."""
        ds = DataStore(outlier_df)
        pd_count = len(outlier_df[outlier_df['Outlier'] == 1])
        ds_count = len(ds[ds['Outlier'] == 1])
        assert ds_count == pd_count


class TestValueCounts:
    """Tests for value_counts operation on label column."""

    def test_outlier_value_counts(self, outlier_df):
        """Test value_counts on Outlier column."""
        ds = DataStore(outlier_df)
        pd_result = outlier_df['Outlier'].value_counts()
        ds_result = ds['Outlier'].value_counts()
        # value_counts order may vary for ties, so check_row_order=False
        assert_series_equal(ds_result, pd_result, check_index=False)
