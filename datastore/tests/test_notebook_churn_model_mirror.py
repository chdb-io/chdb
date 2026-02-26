"""
Mirror Test for Churn Model Notebook
====================================

Tests pandas operations found in the Churn model.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Operations tested from the notebook:
- isnull().sum() for checking missing values
- value_counts(normalize=True) for ratio calculations
- corr() for correlation matrix
- to_numpy() for array conversion
- to_list() for list conversion
- to_json(orient='records') for JSON export
- Column selection and filtering
- List comprehension on columns
- apply() with lambda functions
- Basic statistics: sum(), max(), value_counts()

Design Principle:
    Tests use natural execution triggers following the lazy execution design.
    Mirror Code Pattern: DataStore and pandas operations must be identical.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
)


# ============================================================================
# Fixtures with Churn-like data structure
# ============================================================================


@pytest.fixture
def churn_df():
    """Sample data mimicking telecom churn dataset structure."""
    np.random.seed(42)
    return pd.DataFrame({
        'Churn': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        'AccountWeeks': [128, 107, 137, 84, 75, 118, 121, 147, 141, 65],
        'ContractRenewal': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        'DataPlan': [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        'DataUsage': [2.7, 3.7, 0.0, 0.0, 0.0, 0.0, 2.03, 2.33, 0.0, 0.0],
        'CustServCalls': [1, 1, 0, 2, 3, 0, 3, 4, 0, 4],
        'DayMins': [265.1, 161.6, 243.4, 299.4, 166.7, 223.4, 218.2, 157.0, 258.6, 129.1],
        'DayCalls': [110, 123, 114, 71, 113, 98, 88, 79, 84, 121],
        'MonthlyCharge': [89.0, 82.0, 52.0, 57.0, 41.0, 57.0, 88.0, 84.0, 58.0, 36.0],
        'OverageFee': [9.87, 9.78, 6.06, 3.10, 7.42, 8.86, 8.01, 8.74, 6.55, 10.69],
        'RoamMins': [10.0, 13.7, 12.2, 6.6, 10.1, 12.7, 15.8, 7.1, 11.2, 5.0],
    })


@pytest.fixture
def churn_extended_df():
    """Extended churn data for more comprehensive testing."""
    np.random.seed(123)
    n = 100

    # Generate imbalanced churn data (typical ~85% retain, ~15% churn)
    churn_values = np.random.choice([0, 1], n, p=[0.85, 0.15])

    return pd.DataFrame({
        'Churn': churn_values,
        'AccountWeeks': np.random.randint(50, 200, n),
        'ContractRenewal': np.random.choice([0, 1], n, p=[0.1, 0.9]),
        'DataPlan': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'DataUsage': np.random.exponential(1.5, n) * np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'CustServCalls': np.random.poisson(1.5, n),
        'DayMins': np.random.normal(180, 50, n),
        'DayCalls': np.random.randint(60, 140, n),
        'MonthlyCharge': np.random.uniform(30, 100, n),
        'OverageFee': np.random.uniform(3, 15, n),
        'RoamMins': np.random.exponential(10, n),
    })


# ============================================================================
# Test Classes
# ============================================================================


class TestMissingValueChecks:
    """Tests for missing value check operations from the notebook."""

    def test_isnull_sum(self, churn_df):
        """Test data.isnull().sum() pattern for checking missing values."""
        ds = DataStore(churn_df)

        # pandas
        pd_result = churn_df.isnull().sum()

        # DataStore
        ds_result = ds.isnull().sum()

        assert_series_equal(ds_result, pd_result)

    def test_isnull_sum_with_nulls(self):
        """Test isnull().sum() with actual null values."""
        df = pd.DataFrame({
            'A': [1, 2, None, 4, None],
            'B': [None, 2, 3, None, 5],
            'C': [1, 2, 3, 4, 5],
        })
        ds = DataStore(df)

        pd_result = df.isnull().sum()
        ds_result = ds.isnull().sum()

        assert_series_equal(ds_result, pd_result)


class TestValueCountsOperations:
    """Tests for value_counts operations from the notebook."""

    def test_value_counts_basic(self, churn_df):
        """Test basic value_counts() on a column."""
        ds = DataStore(churn_df)

        pd_result = churn_df['Churn'].value_counts()
        ds_result = ds['Churn'].value_counts()

        assert_series_equal(ds_result, pd_result)

    def test_value_counts_normalize(self, churn_df):
        """Test value_counts(normalize=True) for ratio calculation."""
        ds = DataStore(churn_df)

        pd_result = churn_df['Churn'].value_counts(normalize=True)
        ds_result = ds['Churn'].value_counts(normalize=True)

        assert_series_equal(ds_result, pd_result, check_dtype=False, rtol=1e-5)

    def test_value_counts_normalize_values(self, churn_df):
        """Test accessing .values from normalized value_counts - pattern from notebook."""
        ds = DataStore(churn_df)

        # This is the pattern: churn_ratios = data['Churn'].value_counts(normalize=True).values
        pd_ratios = churn_df['Churn'].value_counts(normalize=True).values
        ds_ratios = get_series(ds['Churn'].value_counts(normalize=True)).values

        np.testing.assert_allclose(ds_ratios, pd_ratios, rtol=1e-5)


class TestCorrelationMatrix:
    """Tests for correlation matrix operations from the notebook."""

    def test_corr_full(self, churn_df):
        """Test corr() correlation matrix."""
        ds = DataStore(churn_df)

        pd_result = churn_df.corr()
        ds_result = ds.corr()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_corr_column_slice(self, churn_df):
        """Test corr().iloc[:, :1] pattern for correlation with target."""
        ds = DataStore(churn_df)

        # Pattern from notebook: mapping = data.corr().iloc[:, :1]
        pd_result = churn_df.corr().iloc[:, :1]
        ds_result = get_dataframe(ds.corr()).iloc[:, :1]

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestDataConversions:
    """Tests for data conversion operations from the notebook."""

    def test_to_numpy_series(self, churn_df):
        """Test to_numpy() on a Series - pattern from notebook for target extraction."""
        ds = DataStore(churn_df)

        # Pattern: target = data['Churn'].to_numpy()
        pd_target = churn_df['Churn'].to_numpy()
        ds_target = get_series(ds['Churn']).to_numpy()

        np.testing.assert_array_equal(ds_target, pd_target)

    def test_to_json_records(self, churn_df):
        """Test to_json(orient='records') for chart data export."""
        ds = DataStore(churn_df)

        # Pattern from notebook: chart_data.datasets.layer00 = data.to_json(orient='records')
        pd_json = churn_df.to_json(orient='records')
        ds_json = get_dataframe(ds).to_json(orient='records')

        # Parse and compare as lists to handle potential ordering differences
        import json
        pd_data = json.loads(pd_json)
        ds_data = json.loads(ds_json)

        assert len(pd_data) == len(ds_data)
        # Compare sorted by all columns to be order-agnostic
        pd_sorted = sorted(pd_data, key=lambda x: tuple(sorted(x.items())))
        ds_sorted = sorted(ds_data, key=lambda x: tuple(sorted(x.items())))
        assert pd_sorted == ds_sorted


class TestColumnOperations:
    """Tests for column operations from the notebook."""

    def test_column_selection(self, churn_df):
        """Test single column selection."""
        ds = DataStore(churn_df)

        pd_result = churn_df['Churn']
        ds_result = ds['Churn']

        assert_series_equal(ds_result, pd_result)

    def test_columns_list_comprehension(self, churn_df):
        """Test column list comprehension pattern from notebook."""
        ds = DataStore(churn_df)

        # Pattern: features = [col for col in data.columns if col != 'Churn']
        pd_features = [col for col in churn_df.columns if col != 'Churn']
        ds_features = [col for col in ds.columns if col != 'Churn']

        assert pd_features == ds_features

    def test_column_subset_selection(self, churn_df):
        """Test selecting subset of columns using list."""
        ds = DataStore(churn_df)

        # Pattern: features = data[features]
        feature_cols = [col for col in churn_df.columns if col != 'Churn']

        pd_result = churn_df[feature_cols]
        ds_result = ds[feature_cols]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAggregations:
    """Tests for aggregation operations from the notebook."""

    def test_sum_over_series(self, churn_df):
        """Test sum() on value_counts result - total calculation."""
        ds = DataStore(churn_df)

        # Pattern: total = sum(data['Churn'].value_counts())
        pd_total = sum(churn_df['Churn'].value_counts())
        ds_total = sum(get_series(ds['Churn'].value_counts()))

        assert pd_total == ds_total

    def test_max_value_counts(self, churn_df):
        """Test max() on value_counts - majority class calculation."""
        ds = DataStore(churn_df)

        # Pattern: majority = data['Churn'].value_counts().max()
        pd_majority = churn_df['Churn'].value_counts().max()
        ds_majority = get_series(ds['Churn'].value_counts()).max()

        assert pd_majority == ds_majority


class TestApplyOperations:
    """Tests for apply operations from the notebook."""

    def test_apply_round(self):
        """Test apply with np.round lambda - pattern from notebook."""
        df = pd.DataFrame({
            'Values': [1.234, 2.567, 3.891, 4.123, 5.678],
        })
        ds = DataStore(df)

        # Pattern: up['CustServCalls'] = up['CustServCalls'].apply(lambda row: np.round(row, 0))
        pd_result = df['Values'].apply(lambda row: np.round(row, 0))
        ds_result = ds['Values'].apply(lambda row: np.round(row, 0))

        assert_series_equal(ds_result, pd_result, check_dtype=False)


class TestSortOperations:
    """Tests for sort operations from the notebook."""

    def test_sort_values_ascending_false(self, churn_df):
        """Test sort_values with ascending=False."""
        ds = DataStore(churn_df)

        pd_result = churn_df.sort_values(by='DayMins', ascending=False)
        ds_result = ds.sort_values(by='DayMins', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFeatureImportancePattern:
    """Tests for feature importance patterns from the notebook."""

    def test_create_importance_df(self):
        """Test creating DataFrame from list of tuples - zip pattern."""
        features = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9],
        })
        importances = [0.3, 0.5, 0.2]

        # Pattern from notebook:
        # importances = pd.DataFrame(
        #     list(zip(features.columns, model.feature_importances_)),
        #     columns=["feature", "importance"],
        # )
        pd_result = pd.DataFrame(
            list(zip(features.columns, importances)),
            columns=["feature", "importance"],
        )

        ds_features = DataStore(features)
        ds_result = DataStore(pd.DataFrame(
            list(zip(ds_features.columns, importances)),
            columns=["feature", "importance"],
        ))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_get_top_features(self):
        """Test sort and iloc for top N feature extraction."""
        importances = pd.DataFrame({
            'feature': ['A', 'B', 'C', 'D', 'E'],
            'importance': [0.1, 0.3, 0.25, 0.15, 0.2],
        })
        ds = DataStore(importances)

        # Pattern: bal = importances.sort_values(by='importance', ascending=False)['feature'].iloc[:5].to_list()
        pd_result = importances.sort_values(by='importance', ascending=False)['feature'].iloc[:5].to_list()

        ds_sorted = get_dataframe(ds.sort_values(by='importance', ascending=False))
        ds_result = ds_sorted['feature'].iloc[:5].to_list()

        assert pd_result == ds_result


class TestImbalancedDataCalculations:
    """Tests for imbalanced data calculations from the notebook."""

    def test_baseline_calculation(self, churn_extended_df):
        """Test baseline calculation for imbalanced data."""
        ds = DataStore(churn_extended_df)

        # Pattern from notebook:
        # total = sum(data['Churn'].value_counts())
        # majority = data['Churn'].value_counts().max()
        # imbalanced_baseline = majority / total
        pd_total = sum(churn_extended_df['Churn'].value_counts())
        pd_majority = churn_extended_df['Churn'].value_counts().max()
        pd_baseline = pd_majority / pd_total

        ds_vc = get_series(ds['Churn'].value_counts())
        ds_total = sum(ds_vc)
        ds_majority = ds_vc.max()
        ds_baseline = ds_majority / ds_total

        np.testing.assert_almost_equal(ds_baseline, pd_baseline, decimal=10)


class TestDataFrameInfoLike:
    """Tests for info-like operations from the notebook."""

    def test_dtypes(self, churn_df):
        """Test dtypes property."""
        ds = DataStore(churn_df)

        pd_dtypes = churn_df.dtypes
        ds_dtypes = ds.dtypes

        # Complete dtype comparison using pd.testing.assert_series_equal
        pd.testing.assert_series_equal(pd_dtypes, ds_dtypes)

    def test_shape(self, churn_df):
        """Test shape property."""
        ds = DataStore(churn_df)
        assert ds.shape == churn_df.shape


class TestUpscaledDataOperations:
    """Tests mimicking operations on SMOTE-upscaled data."""

    def test_assign_column(self):
        """Test assigning a new column to DataFrame."""
        df = pd.DataFrame({
            'A': [1.1, 2.2, 3.3, 4.4, 5.5],
            'B': [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        ds = DataStore(df.copy())

        target = [0, 1, 0, 1, 0]

        # Pattern: up['Churn'] = r_target
        df['Churn'] = target
        ds['Churn'] = target

        assert_datastore_equals_pandas(ds, df)

    def test_items_iteration(self, churn_df):
        """Test iterating over columns using items()."""
        ds = DataStore(churn_df)

        # Pattern from notebook: for i, (k, v) in enumerate(data.items(), 1):
        pd_items = list(churn_df.items())
        ds_items = list(ds.items())

        assert len(pd_items) == len(ds_items)

        for (pd_name, pd_series), (ds_name, ds_series) in zip(pd_items, ds_items):
            assert pd_name == ds_name
            assert_series_equal(ds_series, pd_series)
