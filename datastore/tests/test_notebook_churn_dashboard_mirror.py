"""
Mirror Test for Churn Dashboard Notebook
========================================

Tests pandas operations found in the Churn Dashboard.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Operations tested from the notebook:
- DataFrame creation with column names from numpy arrays
- Column list comprehension filtering
- Column assignment with series/lists
- apply() with np.ceil lambda
- groupby with multiple aggregations and reset_index
- loc accessor for value extraction
- rank() method with dense method
- sort_values() with reset_index(drop=True)
- iloc[0].to_dict() pattern
- DataFrame copy()
- values[0] scalar extraction
- to_json(orient='records') export

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
# Fixtures mimicking Churn Dashboard data patterns
# ============================================================================


@pytest.fixture
def churn_data():
    """Sample churn data similar to TELECOM_CHURN dataset."""
    np.random.seed(42)
    return pd.DataFrame({
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
        'Churn': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    })


@pytest.fixture
def cluster_data():
    """Sample clustered data for cluster analysis tests."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(20),
        'feature2': np.random.randn(20),
        'cluster': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        'Churn': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    })


@pytest.fixture
def feature_importance_data():
    """Feature importance data for testing."""
    return pd.DataFrame({
        'feature': ['CustServCalls', 'DayMins', 'ContractRenewal', 'DataUsage', 'MonthlyCharge'],
        'importance': [0.25, 0.20, 0.18, 0.15, 0.12],
    })


# ============================================================================
# Test Classes
# ============================================================================


class TestDataFrameCreationFromNumpy:
    """Tests for DataFrame creation from numpy arrays - pattern from notebook."""

    def test_dataframe_from_numpy_with_columns(self):
        """Test pd.DataFrame(numpy_array, columns=[...]) pattern."""
        feature_names = ['A', 'B', 'C']
        np_array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])

        # Pattern from notebook: dataset = pd.DataFrame(dataset, columns=feature_names)
        pd_result = pd.DataFrame(np_array, columns=feature_names)
        ds_result = DataStore(pd.DataFrame(np_array, columns=feature_names))

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dataframe_from_list_of_tuples_with_columns(self):
        """Test pd.DataFrame(list(zip(...)), columns=[...]) pattern."""
        features = ['feat1', 'feat2', 'feat3']
        importances = [0.3, 0.5, 0.2]

        # Pattern: importances = pd.DataFrame(list(zip(features.columns, model.feature_importances_)), columns=['feature', 'importance'])
        pd_result = pd.DataFrame(
            list(zip(features, importances)),
            columns=['feature', 'importance']
        )
        ds_result = DataStore(pd.DataFrame(
            list(zip(features, importances)),
            columns=['feature', 'importance']
        ))

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnListComprehension:
    """Tests for column list comprehension patterns from notebook."""

    def test_column_filter_comprehension(self, churn_data):
        """Test [col for col in data.columns if col != 'Churn'] pattern."""
        ds = DataStore(churn_data)

        # Pattern: feature_names = [col for col in data.columns if col != 'Churn']
        pd_features = [col for col in churn_data.columns if col != 'Churn']
        ds_features = [col for col in ds.columns if col != 'Churn']

        assert pd_features == ds_features

    def test_column_selection_from_list(self, churn_data):
        """Test data[feature_names] with column list selection."""
        ds = DataStore(churn_data)

        feature_names = [col for col in churn_data.columns if col != 'Churn']

        # Pattern: features = data[feature_names]
        pd_result = churn_data[feature_names]
        ds_result = ds[feature_names]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnAssignment:
    """Tests for column assignment patterns from notebook."""

    def test_assign_target_column(self, churn_data):
        """Test df['Churn'] = target pattern for adding target column."""
        # Create data without Churn column
        df = churn_data.drop('Churn', axis=1).copy()
        ds = DataStore(df.copy())

        target = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]

        # Pattern: up['Churn'] = r_target
        df['Churn'] = target
        ds['Churn'] = target

        assert_datastore_equals_pandas(ds, df)

    def test_assign_cluster_column(self, churn_data):
        """Test df['cluster'] = kmeans.labels_ pattern."""
        df = churn_data.copy()
        ds = DataStore(df.copy())

        labels = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]

        # Pattern: dataset['cluster'] = kmeans.labels_
        df['cluster'] = labels
        ds['cluster'] = labels

        assert_datastore_equals_pandas(ds, df)


class TestApplyWithCeil:
    """Tests for apply() with np.ceil - pattern from notebook."""

    def test_apply_ceil(self):
        """Test df['col'].apply(lambda row: np.ceil(row)) pattern."""
        df = pd.DataFrame({
            'CustServCalls': [1.2, 2.7, 0.1, 3.9, 4.5],
        })
        ds = DataStore(df.copy())

        # Pattern: dataset['CustServCalls'] = dataset['CustServCalls'].apply(lambda row: np.ceil(row))
        pd_result = df['CustServCalls'].apply(lambda row: np.ceil(row))
        ds_result = ds['CustServCalls'].apply(lambda row: np.ceil(row))

        assert_series_equal(ds_result, pd_result)

    def test_apply_ceil_and_assign(self):
        """Test apply and assign in-place pattern."""
        df = pd.DataFrame({
            'CustServCalls': [1.2, 2.7, 0.1, 3.9, 4.5],
        })
        ds = DataStore(df.copy())

        # Pattern: dataset['CustServCalls'] = dataset['CustServCalls'].apply(lambda row: np.ceil(row))
        df['CustServCalls'] = df['CustServCalls'].apply(lambda row: np.ceil(row))
        ds['CustServCalls'] = ds['CustServCalls'].apply(lambda row: np.ceil(row))

        assert_datastore_equals_pandas(ds, df)


class TestGroupbyMultipleAggregations:
    """Tests for groupby with multiple aggregations - pattern from notebook."""

    def test_groupby_agg_mean_count(self, cluster_data):
        """Test data.groupby('col')['target'].agg(['mean', 'count']) pattern."""
        ds = DataStore(cluster_data)

        # Pattern: churn_by_quartile = temp_df.groupby('quartile')['churn'].agg(['mean', 'count'])
        pd_result = cluster_data.groupby('cluster')['Churn'].agg(['mean', 'count'])
        ds_result = ds.groupby('cluster')['Churn'].agg(['mean', 'count'])

        # GroupBy with agg can have different row order, use check_row_order=False
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_groupby_agg_reset_index(self, cluster_data):
        """Test groupby().agg().reset_index() pattern."""
        ds = DataStore(cluster_data)

        # Pattern: churn_by_quartile = temp_df.groupby('quartile')['churn'].agg(['mean', 'count']).reset_index()
        pd_result = cluster_data.groupby('cluster')['Churn'].agg(['mean', 'count']).reset_index()
        ds_result = ds.groupby('cluster')['Churn'].agg(['mean', 'count']).reset_index()

        # GroupBy result order is implementation-dependent
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


class TestLocAccessor:
    """Tests for loc accessor patterns from notebook."""

    def test_loc_idxmax(self, feature_importance_data):
        """Test df.loc[df['col'].idxmax()] pattern."""
        ds = DataStore(feature_importance_data)

        # Pattern: highest_risk_quartile = churn_by_quartile.loc[churn_by_quartile['mean'].idxmax()]
        pd_idx = feature_importance_data['importance'].idxmax()
        ds_idx = get_series(ds['importance']).idxmax()

        pd_result = feature_importance_data.loc[pd_idx]
        ds_result = get_dataframe(ds).loc[ds_idx]

        # Compare as Series
        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_loc_scalar_extraction(self, feature_importance_data):
        """Test df.loc[idx, 'col'] for scalar extraction."""
        ds = DataStore(feature_importance_data)

        # Pattern: highest_churn_rate = highest_risk_quartile['mean']
        pd_idx = feature_importance_data['importance'].idxmax()
        pd_result = feature_importance_data.loc[pd_idx, 'importance']

        ds_idx = get_series(ds['importance']).idxmax()
        ds_df = get_dataframe(ds)
        ds_result = ds_df.loc[ds_idx, 'importance']

        assert pd_result == ds_result


class TestRankMethod:
    """Tests for rank() method - pattern from notebook."""

    def test_rank_dense_ascending_false(self, feature_importance_data):
        """Test df['col'].rank(method='dense', ascending=False) pattern."""
        ds = DataStore(feature_importance_data)

        # Pattern: enhanced_importances['importance_rank'] = enhanced_importances['importance'].rank(method='dense', ascending=False).astype(int)
        pd_result = feature_importance_data['importance'].rank(method='dense', ascending=False).astype(int)
        ds_result = get_series(ds['importance']).rank(method='dense', ascending=False).astype(int)

        pd.testing.assert_series_equal(ds_result, pd_result)

    def test_rank_dense_assign(self, feature_importance_data):
        """Test rank assignment pattern."""
        df = feature_importance_data.copy()
        ds = DataStore(df.copy())

        # Pattern: enhanced_importances['importance_rank'] = enhanced_importances['importance'].rank(method='dense', ascending=False).astype(int)
        df['importance_rank'] = df['importance'].rank(method='dense', ascending=False).astype(int)

        ds_df = get_dataframe(ds)
        ds_df['importance_rank'] = ds_df['importance'].rank(method='dense', ascending=False).astype(int)
        ds_result = DataStore(ds_df)

        assert_datastore_equals_pandas(ds_result, df)


class TestSortValuesResetIndex:
    """Tests for sort_values with reset_index - pattern from notebook."""

    def test_sort_values_reset_index_drop(self, feature_importance_data):
        """Test sort_values('col').reset_index(drop=True) pattern."""
        ds = DataStore(feature_importance_data)

        # Pattern: enhanced_importances = enhanced_importances.sort_values('importance_rank').reset_index(drop=True)
        pd_result = feature_importance_data.sort_values('importance').reset_index(drop=True)
        ds_result = ds.sort_values('importance').reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending_reset_index(self, feature_importance_data):
        """Test sort_values(ascending=False).reset_index(drop=True) pattern."""
        ds = DataStore(feature_importance_data)

        pd_result = feature_importance_data.sort_values('importance', ascending=False).reset_index(drop=True)
        ds_result = ds.sort_values('importance', ascending=False).reset_index(drop=True)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestIlocToDict:
    """Tests for iloc[0].to_dict() pattern from notebook."""

    def test_iloc_first_row_to_dict(self, feature_importance_data):
        """Test df.iloc[0].to_dict() pattern."""
        ds = DataStore(feature_importance_data)

        # Pattern: top = dataframe.iloc[0].to_dict()
        pd_result = feature_importance_data.iloc[0].to_dict()
        ds_result = get_dataframe(ds).iloc[0].to_dict()

        assert pd_result == ds_result

    def test_iloc_top3_to_dict(self, cluster_data):
        """Test df.iloc[:3].to_dict(orient='records') pattern."""
        ds = DataStore(cluster_data)

        # Pattern: top3 = same_cluster.sort_values(by='similarity_score', ascending=False).iloc[:3].to_dict(orient='records')
        pd_result = cluster_data.iloc[:3].to_dict(orient='records')
        ds_result = get_dataframe(ds).iloc[:3].to_dict(orient='records')

        assert pd_result == ds_result


class TestDataFrameCopy:
    """Tests for DataFrame copy() - pattern from notebook."""

    def test_copy(self, churn_data):
        """Test df.copy() pattern."""
        ds = DataStore(churn_data)

        # Pattern: same_cluster = dataset.copy()
        pd_copy = churn_data.copy()
        ds_copy = get_dataframe(ds.copy())

        assert_datastore_equals_pandas(ds_copy, pd_copy)

    def test_copy_modify_original(self, churn_data):
        """Test that copy is independent of original."""
        df = churn_data.copy()
        ds = DataStore(df.copy())

        # Make copies
        pd_copy = df.copy()
        ds_copy = ds.copy()

        # Modify originals
        df['new_col'] = 1
        ds['new_col'] = 1

        # Check copies are unmodified
        assert 'new_col' not in pd_copy.columns
        assert 'new_col' not in ds_copy.columns


class TestValuesScalarExtraction:
    """Tests for .values[0] scalar extraction - pattern from notebook."""

    def test_values_scalar(self, cluster_data):
        """Test df[condition]['col'].values[0] pattern."""
        ds = DataStore(cluster_data)

        # Pattern: c1 = dataframe[dataframe['cluster'] == 0]['churn_rate'].values[0]
        # First calculate churn rate per cluster
        churn_rate = cluster_data.groupby('cluster')['Churn'].mean().reset_index()
        churn_rate.columns = ['cluster', 'churn_rate']

        ds_churn_rate = ds.groupby('cluster')['Churn'].mean().reset_index()
        ds_churn_rate.columns = ['cluster', 'churn_rate']

        pd_value = churn_rate[churn_rate['cluster'] == 0]['churn_rate'].values[0]
        ds_value = get_dataframe(ds_churn_rate[ds_churn_rate['cluster'] == 0])['churn_rate'].values[0]

        np.testing.assert_almost_equal(pd_value, ds_value)


class TestToJsonRecords:
    """Tests for to_json(orient='records') - pattern from notebook."""

    def test_to_json_records(self, feature_importance_data):
        """Test df.to_json(orient='records') pattern."""
        ds = DataStore(feature_importance_data)

        # Pattern: chart_importances.datasets.layer00 = importances.to_json(orient='records')
        import json

        pd_json = feature_importance_data.to_json(orient='records')
        ds_json = get_dataframe(ds).to_json(orient='records')

        # Parse and compare
        pd_data = json.loads(pd_json)
        ds_data = json.loads(ds_json)

        assert pd_data == ds_data


class TestCorrelationColumn:
    """Tests for correlation patterns from notebook."""

    def test_corr_series(self, churn_data):
        """Test series.corr(other_series) pattern."""
        ds = DataStore(churn_data)

        # Pattern: correlation = feature_data_clean.corr(churn_data_clean)
        pd_corr = churn_data['AccountWeeks'].corr(churn_data['Churn'])
        ds_corr = get_series(ds['AccountWeeks']).corr(get_series(ds['Churn']))

        np.testing.assert_almost_equal(pd_corr, ds_corr, decimal=10)


class TestMeanStatistics:
    """Tests for mean() statistics patterns from notebook."""

    def test_filtered_mean(self, churn_data):
        """Test df[df['col'] == value]['other'].mean() pattern."""
        ds = DataStore(churn_data)

        # Pattern: churned_mean = data_df[data_df['Churn'] == 1][feature_name].mean()
        pd_mean = churn_data[churn_data['Churn'] == 1]['DayMins'].mean()
        ds_mean = ds[ds['Churn'] == 1]['DayMins'].mean()

        # Trigger execution and compare
        ds_mean_val = float(ds_mean)
        np.testing.assert_almost_equal(pd_mean, ds_mean_val, decimal=5)

    def test_overall_mean(self, churn_data):
        """Test df['col'].mean() pattern."""
        ds = DataStore(churn_data)

        # Pattern: overall_mean = feature_data.mean()
        pd_mean = churn_data['DayMins'].mean()
        ds_mean = ds['DayMins'].mean()

        ds_mean_val = float(ds_mean)
        np.testing.assert_almost_equal(pd_mean, ds_mean_val, decimal=5)


class TestStdStatistics:
    """Tests for std() statistics patterns from notebook."""

    def test_std(self, churn_data):
        """Test df['col'].std() pattern."""
        ds = DataStore(churn_data)

        # Pattern: overall_std = feature_data.std()
        pd_std = churn_data['DayMins'].std()
        ds_std = ds['DayMins'].std()

        ds_std_val = float(ds_std)
        np.testing.assert_almost_equal(pd_std, ds_std_val, decimal=5)


class TestVarStatistics:
    """Tests for var() variance patterns from notebook."""

    def test_var(self, churn_data):
        """Test df['col'].var() pattern."""
        ds = DataStore(churn_data)

        # Pattern: if len(feature_data_clean) > 1 and feature_data_clean.var() > 0
        pd_var = churn_data['DayMins'].var()
        ds_var = ds['DayMins'].var()

        ds_var_val = float(ds_var)
        np.testing.assert_almost_equal(pd_var, ds_var_val, decimal=5)


class TestPercentileCalculations:
    """Tests for percentile/quantile calculations from notebook."""

    def test_qcut_quartiles(self, churn_data):
        """Test pd.qcut for quartile creation."""
        # Pattern: quartiles = pd.qcut(feature_data_clean, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        pd_quartiles = pd.qcut(churn_data['DayMins'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # DataStore doesn't directly support qcut, this is a pandas-only operation
        # Test that we can feed qcut results back into DataStore
        df = churn_data.copy()
        df['quartile'] = pd_quartiles

        ds = DataStore(df)
        assert 'quartile' in ds.columns


class TestNullHandling:
    """Tests for null handling patterns from notebook."""

    def test_notna_mask(self):
        """Test pd.notna() for creating valid masks."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, None, 4.0, None],
            'B': [None, 2.0, 3.0, None, 5.0],
        })
        ds = DataStore(df)

        # Pattern: valid_mask = pd.notna(feature_data) & pd.notna(churn_data)
        pd_mask = pd.notna(df['A'])
        ds_mask = get_series(ds['A']).notna()

        pd.testing.assert_series_equal(ds_mask, pd_mask)

    def test_dropna(self):
        """Test dropna() for cleaning data."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, None, 4.0, None],
            'B': [None, 2.0, 3.0, None, 5.0],
        })
        ds = DataStore(df)

        # Pattern: churned_customers_feature = data_df[data_df['Churn'] == 1][feature_name].dropna()
        pd_result = df['A'].dropna()
        ds_result = ds['A'].dropna()

        assert_series_equal(ds_result, pd_result)


class TestUniqueCount:
    """Tests for unique value counting from notebook."""

    def test_unique(self, churn_data):
        """Test df['col'].unique() pattern."""
        ds = DataStore(churn_data)

        # Pattern: len(feature_data.unique()) > 2
        pd_unique = churn_data['Churn'].unique()
        ds_unique = get_series(ds['Churn']).unique()

        np.testing.assert_array_equal(sorted(ds_unique), sorted(pd_unique))


class TestChainedOperations:
    """Tests for chained operations patterns from notebook."""

    def test_filter_groupby_agg_reset(self, cluster_data):
        """Test chained filter -> groupby -> agg -> reset_index."""
        ds = DataStore(cluster_data)

        # Chained operation from notebook pattern
        pd_result = cluster_data[cluster_data['cluster'].isin([0, 1])].groupby('cluster')['Churn'].mean().reset_index()
        ds_result = ds[ds['cluster'].isin([0, 1])].groupby('cluster')['Churn'].mean().reset_index()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_sort_iloc_todict_chain(self, feature_importance_data):
        """Test sort_values -> iloc -> to_dict chain."""
        ds = DataStore(feature_importance_data)

        # Pattern: top = dataframe.iloc[0].to_dict()
        pd_sorted = feature_importance_data.sort_values('importance', ascending=False)
        pd_result = pd_sorted.iloc[0].to_dict()

        ds_sorted = ds.sort_values('importance', ascending=False)
        ds_result = get_dataframe(ds_sorted).iloc[0].to_dict()

        assert pd_result == ds_result


class TestDataTypeConversions:
    """Tests for data type conversion patterns from notebook."""

    def test_astype_int(self, churn_data):
        """Test df['col'].astype(int) pattern."""
        ds = DataStore(churn_data)

        # Pattern: target = data['Churn'].astype(int).to_numpy()
        pd_result = churn_data['Churn'].astype(int)
        ds_result = ds['Churn'].astype(int)

        assert_series_equal(ds_result, pd_result)

    def test_astype_int_to_numpy(self, churn_data):
        """Test df['col'].astype(int).to_numpy() pattern."""
        ds = DataStore(churn_data)

        # Pattern: target = data['Churn'].astype(int).to_numpy()
        pd_result = churn_data['Churn'].astype(int).to_numpy()
        ds_result = get_series(ds['Churn'].astype(int)).to_numpy()

        np.testing.assert_array_equal(ds_result, pd_result)
