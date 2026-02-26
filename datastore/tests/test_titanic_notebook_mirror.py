"""
Mirror Test for Titanic Data Science Solutions Notebook
=======================================================

Tests pandas operations found in the titanic-data-science-solutions.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Notebook source: https://www.kaggle.com/code/titanic-data-science-solutions

Operations tested from notebook:
- head(), tail() - Data preview
- columns.values - Get column names
- describe(), describe(include=['O']) - Statistics
- groupby().mean().sort_values() - Aggregation with sorting
- str.extract() - String accessor regex extraction
- crosstab() - Cross tabulation
- replace() - Value replacement
- map() - Value mapping
- fillna() - Fill null values
- drop() - Drop columns
- astype() - Type conversion
- dropna() - Drop NA values
- median() - Median calculation
- .loc[] - Location-based assignment
- pd.cut(), pd.qcut() - Binning operations

Design Principle:
    Tests use natural execution triggers following the lazy execution design.
    Avoid explicit _execute() calls - use natural triggers instead.
    Follow Mirror Code Pattern: pandas operations mirrored in DataStore.
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal
from tests.xfail_markers import pandas_version_cut_array_protocol


# ============================================================================
# Fixtures
# ============================================================================


def dataset_path(filename: str) -> str:
    """Get path to test dataset."""
    import os
    return os.path.join(os.path.dirname(__file__), "dataset", filename)


@pytest.fixture
def titanic_pd_df():
    """Load Titanic dataset as pandas DataFrame."""
    return pd.read_csv(dataset_path("Titanic-Dataset.csv"))


@pytest.fixture
def titanic_ds(titanic_pd_df):
    """Create DataStore from Titanic dataset."""
    return DataStore(titanic_pd_df.copy())


# ============================================================================
# Test Classes - Following notebook structure
# ============================================================================


class TestDataPreview:
    """Tests for basic data preview operations."""

    def test_head_default(self, titanic_pd_df, titanic_ds):
        """head() - default 5 rows."""
        pd_result = titanic_pd_df.head()
        ds_result = titanic_ds.head()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_head_n(self, titanic_pd_df, titanic_ds):
        """head(n) - specific number of rows."""
        pd_result = titanic_pd_df.head(10)
        ds_result = titanic_ds.head(10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_default(self, titanic_pd_df, titanic_ds):
        """tail() - default 5 rows."""
        pd_result = titanic_pd_df.tail()
        ds_result = titanic_ds.tail()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_columns_values(self, titanic_pd_df, titanic_ds):
        """columns.values - get column names as array."""
        pd_result = titanic_pd_df.columns.values
        ds_result = titanic_ds.columns.values
        np.testing.assert_array_equal(ds_result, pd_result)

    def test_shape(self, titanic_pd_df, titanic_ds):
        """shape property."""
        pd_result = titanic_pd_df.shape
        ds_result = titanic_ds.shape
        assert ds_result == pd_result


class TestDescribeStatistics:
    """Tests for describe() operations."""

    def test_describe_numeric(self, titanic_pd_df, titanic_ds):
        """describe() - numeric columns statistics."""
        pd_result = titanic_pd_df.describe()
        ds_result = titanic_ds.describe()
        # Use assert_datastore_equals_pandas for proper comparison
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_describe_object(self, titanic_pd_df, titanic_ds):
        """describe(include=['O']) - object columns statistics."""
        pd_result = titanic_pd_df.describe(include=['O'])
        ds_result = titanic_ds.describe(include=['O'])
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupbyOperations:
    """Tests for groupby operations from the notebook."""

    def test_groupby_pclass_survived_mean_sort(self, titanic_pd_df, titanic_ds):
        """
        train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
        From cell-19 of notebook.
        """
        pd_result = (
            titanic_pd_df[['Pclass', 'Survived']]
            .groupby(['Pclass'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        ds_result = (
            titanic_ds[['Pclass', 'Survived']]
            .groupby(['Pclass'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_sex_survived_mean_sort(self, titanic_pd_df, titanic_ds):
        """
        train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
        From cell-20 of notebook.
        """
        pd_result = (
            titanic_pd_df[["Sex", "Survived"]]
            .groupby(['Sex'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        ds_result = (
            titanic_ds[["Sex", "Survived"]]
            .groupby(['Sex'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_sibsp_survived_mean_sort(self, titanic_pd_df, titanic_ds):
        """
        train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
        From cell-21 of notebook.
        """
        pd_result = (
            titanic_pd_df[["SibSp", "Survived"]]
            .groupby(['SibSp'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        ds_result = (
            titanic_ds[["SibSp", "Survived"]]
            .groupby(['SibSp'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_parch_survived_mean_sort(self, titanic_pd_df, titanic_ds):
        """
        train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
        From cell-22 of notebook.
        """
        pd_result = (
            titanic_pd_df[["Parch", "Survived"]]
            .groupby(['Parch'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        ds_result = (
            titanic_ds[["Parch", "Survived"]]
            .groupby(['Parch'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


class TestDropOperations:
    """Tests for drop operations from the notebook."""

    def test_drop_ticket_cabin(self, titanic_pd_df, titanic_ds):
        """
        train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
        From cell-32 of notebook.
        """
        pd_result = titanic_pd_df.drop(['Ticket', 'Cabin'], axis=1)
        ds_result = titanic_ds.drop(['Ticket', 'Cabin'], axis=1)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_name_passengerid(self, titanic_pd_df, titanic_ds):
        """
        train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
        From cell-40 of notebook.
        """
        pd_result = titanic_pd_df.drop(['Name', 'PassengerId'], axis=1)
        ds_result = titanic_ds.drop(['Name', 'PassengerId'], axis=1)
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestStringExtract:
    """Tests for string extraction operations."""

    def test_str_extract_title(self, titanic_pd_df, titanic_ds):
        """
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+).', expand=False)
        From cell-34 of notebook.
        """
        pd_result = titanic_pd_df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
        ds_result = titanic_ds['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

        # Natural execution trigger via len()
        assert len(ds_result) == len(pd_result)
        # Compare values
        assert_series_equal(ds_result, pd_result, check_names=False)


class TestReplaceOperations:
    """Tests for replace operations from the notebook."""

    def test_replace_single_value(self, titanic_pd_df):
        """Test replace single value."""
        # Extract titles first
        pd_df = titanic_pd_df.copy()
        pd_df['Title'] = pd_df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)

        ds = DataStore(titanic_pd_df.copy())
        ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

        # Replace Mlle with Miss
        pd_df['Title'] = pd_df['Title'].replace('Mlle', 'Miss')
        ds['Title'] = ds['Title'].replace('Mlle', 'Miss')

        # Compare entire DataFrame with Title column
        assert_datastore_equals_pandas(ds, pd_df)

    def test_replace_list_values(self, titanic_pd_df):
        """
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess',...], 'Rare')
        From cell-36 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Title'] = pd_df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)

        ds = DataStore(titanic_pd_df.copy())
        ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

        pd_df['Title'] = pd_df['Title'].replace(rare_titles, 'Rare')
        ds['Title'] = ds['Title'].replace(rare_titles, 'Rare')

        # Compare entire DataFrame with Title column
        assert_datastore_equals_pandas(ds, pd_df)


class TestMapOperations:
    """Tests for map operations from the notebook."""

    def test_map_sex_to_numeric(self, titanic_pd_df):
        """
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        From cell-42 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        assert_datastore_equals_pandas(ds, pd_df)

    def test_map_embarked_to_numeric(self, titanic_pd_df):
        """
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        From cell-67 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        # Fill NaN first to avoid issues with astype(int)
        pd_df['Embarked'] = pd_df['Embarked'].fillna('S')
        pd_df['Embarked'] = pd_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Embarked'] = ds['Embarked'].fillna('S')
        ds['Embarked'] = ds['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        assert_datastore_equals_pandas(ds, pd_df)


class TestFillnaOperations:
    """Tests for fillna operations from the notebook."""

    def test_fillna_with_mode(self, titanic_pd_df):
        """
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        From cell-65 of notebook.
        """
        # Get mode
        freq_port = titanic_pd_df.Embarked.dropna().mode()[0]

        pd_df = titanic_pd_df.copy()
        pd_df['Embarked'] = pd_df['Embarked'].fillna(freq_port)

        ds = DataStore(titanic_pd_df.copy())
        ds['Embarked'] = ds['Embarked'].fillna(freq_port)

        assert_datastore_equals_pandas(ds, pd_df)

    def test_fillna_with_median(self, titanic_pd_df):
        """
        test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
        From cell-69 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        median_fare = pd_df['Fare'].dropna().median()
        pd_df['Fare'] = pd_df['Fare'].fillna(median_fare)

        ds = DataStore(titanic_pd_df.copy())
        ds['Fare'] = ds['Fare'].fillna(median_fare)

        assert_datastore_equals_pandas(ds, pd_df)


class TestAggregationOperations:
    """Tests for aggregation operations."""

    def test_dropna_median(self, titanic_pd_df, titanic_ds):
        """
        guess_df = dataset[...]['Age'].dropna()
        age_guess = guess_df.median()
        From cell-48 of notebook.
        """
        pd_result = titanic_pd_df['Age'].dropna().median()
        ds_result = titanic_ds['Age'].dropna().median()
        # Force execution by using float()
        ds_val = float(ds_result)
        assert abs(ds_val - pd_result) < 1e-5

    def test_mode_single_column(self, titanic_pd_df, titanic_ds):
        """
        freq_port = train_df.Embarked.dropna().mode()[0]
        From cell-64 of notebook.
        """
        pd_result = titanic_pd_df.Embarked.dropna().mode()[0]
        ds_result = titanic_ds['Embarked'].dropna().mode()[0]
        assert ds_result == pd_result


class TestFeatureEngineering:
    """Tests for feature engineering operations from the notebook."""

    def test_create_familysize(self, titanic_pd_df):
        """
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        From cell-56 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1

        ds = DataStore(titanic_pd_df.copy())
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        # Compare entire DataFrame
        assert_datastore_equals_pandas(ds, pd_df)

    def test_create_isalone(self, titanic_pd_df):
        """
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        From cell-58 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        pd_df['IsAlone'] = 0
        pd_df.loc[pd_df['FamilySize'] == 1, 'IsAlone'] = 1

        ds = DataStore(titanic_pd_df.copy())
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        ds['IsAlone'] = 0
        ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1

        # Compare entire DataFrame
        assert_datastore_equals_pandas(ds, pd_df)

    def test_create_age_class(self, titanic_pd_df):
        """
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
        From cell-62 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Age*Class'] = pd_df['Age'] * pd_df['Pclass']

        ds = DataStore(titanic_pd_df.copy())
        ds['Age*Class'] = ds['Age'] * ds['Pclass']

        # Compare entire DataFrame
        assert_datastore_equals_pandas(ds, pd_df)


class TestGroupbyFamilySizeSurvived:
    """Tests for FamilySize groupby operations from notebook."""

    def test_familysize_survived_groupby_mean_sort(self, titanic_pd_df):
        """
        train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
        From cell-56 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1

        ds = DataStore(titanic_pd_df.copy())
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        pd_result = (
            pd_df[['FamilySize', 'Survived']]
            .groupby(['FamilySize'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        ds_result = (
            ds[['FamilySize', 'Survived']]
            .groupby(['FamilySize'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_isalone_survived_groupby_mean(self, titanic_pd_df):
        """
        train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
        From cell-58 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        pd_df['IsAlone'] = 0
        pd_df.loc[pd_df['FamilySize'] == 1, 'IsAlone'] = 1

        ds = DataStore(titanic_pd_df.copy())
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        ds['IsAlone'] = 0
        ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1

        pd_result = (
            pd_df[['IsAlone', 'Survived']]
            .groupby(['IsAlone'], as_index=False)
            .mean()
        )
        ds_result = (
            ds[['IsAlone', 'Survived']]
            .groupby(['IsAlone'], as_index=False)
            .mean()
        )

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestComplexChainOperations:
    """Tests for complex chained operations from the notebook."""

    @pytest.mark.xfail(reason="Bug: Multiple replace() on same column causes SQL generation error in groupby")
    def test_title_survived_groupby_mean(self, titanic_pd_df):
        """
        train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        From cell-36 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Title'] = pd_df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        pd_df['Title'] = pd_df['Title'].replace(rare_titles, 'Rare')
        pd_df['Title'] = pd_df['Title'].replace('Mlle', 'Miss')
        pd_df['Title'] = pd_df['Title'].replace('Ms', 'Miss')
        pd_df['Title'] = pd_df['Title'].replace('Mme', 'Mrs')

        ds = DataStore(titanic_pd_df.copy())
        ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        ds['Title'] = ds['Title'].replace(rare_titles, 'Rare')
        ds['Title'] = ds['Title'].replace('Mlle', 'Miss')
        ds['Title'] = ds['Title'].replace('Ms', 'Miss')
        ds['Title'] = ds['Title'].replace('Mme', 'Mrs')

        pd_result = (
            pd_df[['Title', 'Survived']]
            .groupby(['Title'], as_index=False)
            .mean()
        )
        ds_result = (
            ds[['Title', 'Survived']]
            .groupby(['Title'], as_index=False)
            .mean()
        )

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_embarked_survived_groupby_mean_sort(self, titanic_pd_df):
        """
        train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)
        From cell-65 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        freq_port = pd_df.Embarked.dropna().mode()[0]
        pd_df['Embarked'] = pd_df['Embarked'].fillna(freq_port)

        ds = DataStore(titanic_pd_df.copy())
        ds['Embarked'] = ds['Embarked'].fillna(freq_port)

        pd_result = (
            pd_df[['Embarked', 'Survived']]
            .groupby(['Embarked'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )
        ds_result = (
            ds[['Embarked', 'Survived']]
            .groupby(['Embarked'], as_index=False)
            .mean()
            .sort_values(by='Survived', ascending=False)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


class TestCrosstabOperation:
    """Tests for crosstab operation from the notebook."""

    @pytest.mark.xfail(reason="Bug: pd.crosstab not yet implemented in DataStore")
    def test_crosstab_title_sex(self, titanic_pd_df):
        """
        pd.crosstab(train_df['Title'], train_df['Sex'])
        From cell-34 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Title'] = pd_df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)

        ds = DataStore(titanic_pd_df.copy())
        ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

        pd_result = pd.crosstab(pd_df['Title'], pd_df['Sex'])
        # DataStore does not support pd.crosstab yet
        ds_result = pd.crosstab(ds['Title'], ds['Sex'])

        pd.testing.assert_frame_equal(ds_result, pd_result)


class TestCutQcutOperations:
    """Tests for pd.cut and pd.qcut operations from the notebook."""

    @pandas_version_cut_array_protocol
    def test_pd_cut_age_band(self, titanic_pd_df):
        """
        train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
        From cell-50 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['AgeBand'] = pd.cut(pd_df['Age'], 5)

        ds = DataStore(titanic_pd_df.copy())
        ds['AgeBand'] = pd.cut(ds['Age'], 5)

        assert_datastore_equals_pandas(ds, pd_df)

    @pandas_version_cut_array_protocol
    def test_pd_qcut_fare_band(self, titanic_pd_df):
        """
        train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
        From cell-71 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['FareBand'] = pd.qcut(pd_df['Fare'], 4)

        ds = DataStore(titanic_pd_df.copy())
        ds['FareBand'] = pd.qcut(ds['Fare'], 4)

        assert_datastore_equals_pandas(ds, pd_df)


class TestLocOperations:
    """Tests for .loc[] operations from the notebook."""

    def test_loc_conditional_assignment_simple(self, titanic_pd_df):
        """
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        From cell-52 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df.loc[pd_df['Age'] <= 16, 'Age'] = 0

        ds = DataStore(titanic_pd_df.copy())
        ds.loc[ds['Age'] <= 16, 'Age'] = 0

        assert_datastore_equals_pandas(ds, pd_df)

    def test_loc_conditional_assignment_range(self, titanic_pd_df):
        """
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        From cell-52 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df.loc[(pd_df['Age'] > 16) & (pd_df['Age'] <= 32), 'Age'] = 1

        ds = DataStore(titanic_pd_df.copy())
        ds.loc[(ds['Age'] > 16) & (ds['Age'] <= 32), 'Age'] = 1

        assert_datastore_equals_pandas(ds, pd_df)


class TestMultiColumnSelection:
    """Tests for multi-column selection patterns."""

    def test_select_multiple_columns(self, titanic_pd_df, titanic_ds):
        """Select multiple columns."""
        pd_result = titanic_pd_df[['Pclass', 'Survived', 'Sex']]
        ds_result = titanic_ds[['Pclass', 'Survived', 'Sex']]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_loc_slice_columns(self, titanic_pd_df):
        """
        train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
        From cell-62 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Age*Class'] = pd_df['Age'] * pd_df['Pclass']

        ds = DataStore(titanic_pd_df.copy())
        ds['Age*Class'] = ds['Age'] * ds['Pclass']

        pd_result = pd_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
        ds_result = ds.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestAsTypeOperations:
    """Tests for astype operations from the notebook."""

    def test_astype_int(self, titanic_pd_df):
        """
        dataset['Age'] = dataset['Age'].astype(int)
        From cell-48 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Age'] = pd_df['Age'].fillna(0).astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Age'] = ds['Age'].fillna(0).astype(int)

        assert_datastore_equals_pandas(ds, pd_df)

    def test_map_astype_chain(self, titanic_pd_df):
        """
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        From cell-42 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        assert_datastore_equals_pandas(ds, pd_df)


class TestFilteringOperations:
    """Tests for filtering operations common in data analysis."""

    def test_filter_age_condition(self, titanic_pd_df, titanic_ds):
        """Filter rows by age condition."""
        pd_result = titanic_pd_df[titanic_pd_df['Age'] > 30]
        ds_result = titanic_ds[titanic_ds['Age'] > 30]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_compound_condition(self, titanic_pd_df, titanic_ds):
        """Filter with compound conditions."""
        pd_result = titanic_pd_df[(titanic_pd_df['Sex'] == 'female') & (titanic_pd_df['Survived'] == 1)]
        ds_result = titanic_ds[(titanic_ds['Sex'] == 'female') & (titanic_ds['Survived'] == 1)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_with_dropna(self, titanic_pd_df, titanic_ds):
        """
        guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
        From cell-48 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        pd_result = pd_df[(pd_df['Sex'] == 'male') & (pd_df['Pclass'] == 1)]['Age'].dropna()
        ds_result = ds[(ds['Sex'] == 'male') & (ds['Pclass'] == 1)]['Age'].dropna()

        assert_series_equal(ds_result, pd_result, check_names=False)


class TestNullHandling:
    """Tests for null value handling operations."""

    def test_notnull_astype(self, titanic_pd_df):
        """Test notnull().astype(int) pattern."""
        pd_df = titanic_pd_df.copy()
        pd_df['Has_Cabin'] = pd_df['Cabin'].notnull().astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Has_Cabin'] = ds['Cabin'].notnull().astype(int)

        assert_datastore_equals_pandas(ds, pd_df)

    def test_isnull_sum(self, titanic_pd_df, titanic_ds):
        """Test isnull().sum() pattern."""
        pd_result = titanic_pd_df['Age'].isnull().sum()
        ds_result = titanic_ds['Age'].isnull().sum()
        assert ds_result == pd_result

    def test_isna_sum(self, titanic_pd_df, titanic_ds):
        """Test isna().sum() pattern."""
        pd_result = titanic_pd_df['Cabin'].isna().sum()
        ds_result = titanic_ds['Cabin'].isna().sum()
        assert ds_result == pd_result


class TestCompleteWorkflow:
    """Test complete workflow from notebook sections."""

    def test_complete_data_wrangling_workflow(self, titanic_pd_df):
        """
        Test complete data wrangling workflow from notebook:
        1. Drop columns
        2. Extract title
        3. Replace titles
        4. Map values
        5. Create FamilySize
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # 1. Drop columns
        pd_df = pd_df.drop(['Ticket', 'Cabin'], axis=1)
        ds = ds.drop(['Ticket', 'Cabin'], axis=1)

        # 2. Extract title
        pd_df['Title'] = pd_df.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
        ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

        # 3. Replace titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        pd_df['Title'] = pd_df['Title'].replace(rare_titles, 'Rare')
        ds['Title'] = ds['Title'].replace(rare_titles, 'Rare')

        pd_df['Title'] = pd_df['Title'].replace('Mlle', 'Miss')
        ds['Title'] = ds['Title'].replace('Mlle', 'Miss')

        pd_df['Title'] = pd_df['Title'].replace('Ms', 'Miss')
        ds['Title'] = ds['Title'].replace('Ms', 'Miss')

        pd_df['Title'] = pd_df['Title'].replace('Mme', 'Mrs')
        ds['Title'] = ds['Title'].replace('Mme', 'Mrs')

        # 4. Map Sex to numeric
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # 5. Create FamilySize
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        # Compare entire DataFrame
        assert_datastore_equals_pandas(ds, pd_df)


# ============================================================================
# Additional tests from titanic-data-science-solutions.ipynb notebook
# ============================================================================


class TestAgeGuessingPattern:
    """
    Tests for the age guessing pattern from cell-48 of notebook.
    This is a complex pattern that iterates over Sex/Pclass combinations
    to fill Age NaN values.
    """

    def test_filter_sex_pclass_age_median(self, titanic_pd_df):
        """
        Pattern: guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
                 age_guess = guess_df.median()
        From cell-48 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        # Convert Sex to numeric first (as done in notebook)
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # Test one combination: male (0) and Pclass 1
        pd_result = pd_df[(pd_df['Sex'] == 0) & (pd_df['Pclass'] == 1)]['Age'].dropna().median()
        ds_result = ds[(ds['Sex'] == 0) & (ds['Pclass'] == 1)]['Age'].dropna().median()

        assert abs(float(ds_result) - pd_result) < 1e-5

    def test_age_guessing_all_combinations(self, titanic_pd_df):
        """
        Test age guessing for all Sex/Pclass combinations (2x3=6 combinations).
        From cell-48 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

        ds = DataStore(titanic_pd_df.copy())
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        for i in range(0, 2):  # Sex: 0=male, 1=female
            for j in range(0, 3):  # Pclass: 1, 2, 3
                pd_guess = pd_df[(pd_df['Sex'] == i) & (pd_df['Pclass'] == j+1)]['Age'].dropna().median()
                ds_guess = ds[(ds['Sex'] == i) & (ds['Pclass'] == j+1)]['Age'].dropna().median()

                # Handle potential NaN if no matching rows
                if pd.isna(pd_guess):
                    assert pd.isna(ds_guess) or float(ds_guess) != float(ds_guess)  # NaN check
                else:
                    assert abs(float(ds_guess) - pd_guess) < 1e-5, \
                        f"Mismatch for Sex={i}, Pclass={j+1}: DS={ds_guess}, PD={pd_guess}"


class TestAgeBandGroupBy:
    """Tests for AgeBand groupby operations from cell-50 of notebook."""

    @pytest.mark.xfail(reason="chDB does not support CATEGORY dtype from pd.cut in groupby operations")
    def test_ageband_survived_groupby_mean_sort(self, titanic_pd_df):
        """
        train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
        train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()
            .sort_values(by='AgeBand', ascending=True)
        From cell-50 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['AgeBand'] = pd.cut(pd_df['Age'], 5)

        ds = DataStore(titanic_pd_df.copy())
        ds['AgeBand'] = pd.cut(ds['Age'], 5)

        # observed=False is deprecated for categorical groupby; will default to True in future
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The default of observed=False is deprecated')
            pd_result = (
                pd_df[['AgeBand', 'Survived']]
                .groupby(['AgeBand'], as_index=False)
                .mean()
                .sort_values(by='AgeBand', ascending=True)
            )
        ds_result = (
            ds[['AgeBand', 'Survived']]
            .groupby(['AgeBand'], as_index=False)
            .mean()
            .sort_values(by='AgeBand', ascending=True)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


class TestFareBandGroupBy:
    """Tests for FareBand groupby operations from cell-71 of notebook."""

    @pytest.mark.xfail(reason="chDB does not support CATEGORY dtype from pd.qcut in groupby operations")
    def test_fareband_survived_groupby_mean_sort(self, titanic_pd_df):
        """
        train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
        train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()
            .sort_values(by='FareBand', ascending=True)
        From cell-71 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        pd_df['FareBand'] = pd.qcut(pd_df['Fare'], 4)

        ds = DataStore(titanic_pd_df.copy())
        ds['FareBand'] = pd.qcut(ds['Fare'], 4)

        # observed=False is deprecated for categorical groupby; will default to True in future
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The default of observed=False is deprecated')
            pd_result = (
                pd_df[['FareBand', 'Survived']]
                .groupby(['FareBand'], as_index=False)
                .mean()
                .sort_values(by='FareBand', ascending=True)
            )
        ds_result = (
            ds[['FareBand', 'Survived']]
            .groupby(['FareBand'], as_index=False)
            .mean()
            .sort_values(by='FareBand', ascending=True)
        )

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


class TestMLDataPreparation:
    """
    Tests for ML model data preparation patterns from cell-77 of notebook.
    This involves splitting features (X) and target (Y) from the dataset.
    """

    def test_drop_survived_for_features(self, titanic_pd_df):
        """
        X_train = train_df.drop("Survived", axis=1)
        From cell-77 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # First prepare the data similar to notebook
        pd_df = pd_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
        ds = ds.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

        pd_result = pd_df.drop("Survived", axis=1)
        ds_result = ds.drop("Survived", axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_target_column(self, titanic_pd_df, titanic_ds):
        """
        Y_train = train_df["Survived"]
        From cell-77 of notebook.
        """
        pd_result = titanic_pd_df["Survived"]
        ds_result = titanic_ds["Survived"]

        assert_series_equal(ds_result, pd_result, check_names=False)

    def test_train_test_split_shapes(self, titanic_pd_df):
        """
        Test that X_train.shape, Y_train.shape produce correct shapes.
        From cell-77 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # Prepare data
        pd_df = pd_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
        ds = ds.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

        pd_X = pd_df.drop("Survived", axis=1)
        pd_Y = pd_df["Survived"]

        ds_X = ds.drop("Survived", axis=1)
        ds_Y = ds["Survived"]

        assert ds_X.shape == pd_X.shape
        assert len(ds_Y) == len(pd_Y)


class TestDataFrameCreation:
    """
    Tests for creating new DataFrames from scratch.
    Similar to patterns used in cell-81 (coeff_df) and cell-97 (models).
    """

    def test_create_dataframe_from_columns(self, titanic_pd_df):
        """
        coeff_df = pd.DataFrame(train_df.columns.delete(0))
        From cell-81 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # Create DataFrame from columns (without first column)
        pd_columns = pd_df.columns.delete(0)
        ds_columns = ds.columns.delete(0)

        # Compare the column values
        np.testing.assert_array_equal(ds_columns, pd_columns)

    def test_create_model_comparison_dataframe(self):
        """
        Test creating a comparison DataFrame similar to cell-97.
        models = pd.DataFrame({
            'Model': ['Model1', 'Model2'],
            'Score': [80.0, 85.0]
        })
        """
        pd_df = pd.DataFrame({
            'Model': ['LogisticRegression', 'RandomForest', 'SVM'],
            'Score': [80.36, 86.76, 83.84]
        })
        ds = DataStore({
            'Model': ['LogisticRegression', 'RandomForest', 'SVM'],
            'Score': [80.36, 86.76, 83.84]
        })

        assert_datastore_equals_pandas(ds, pd_df)

    def test_dataframe_sort_values(self):
        """
        models.sort_values(by='Score', ascending=False)
        From cell-97 of notebook.
        """
        pd_df = pd.DataFrame({
            'Model': ['LogisticRegression', 'RandomForest', 'SVM'],
            'Score': [80.36, 86.76, 83.84]
        })
        ds = DataStore({
            'Model': ['LogisticRegression', 'RandomForest', 'SVM'],
            'Score': [80.36, 86.76, 83.84]
        })

        pd_result = pd_df.sort_values(by='Score', ascending=False)
        ds_result = ds.sort_values(by='Score', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


class TestSubmissionDataFrame:
    """
    Tests for creating submission DataFrame pattern from cell-98 of notebook.
    """

    def test_create_submission_dataframe(self, titanic_pd_df):
        """
        submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
        From cell-98 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # Simulate Y_pred as simple array
        y_pred = np.array([0, 1, 0, 1, 1] * (len(pd_df) // 5 + 1))[:len(pd_df)]

        pd_submission = pd.DataFrame({
            "PassengerId": pd_df["PassengerId"],
            "Survived": y_pred
        })
        ds_submission = DataStore({
            "PassengerId": ds["PassengerId"],
            "Survived": y_pred
        })

        assert_datastore_equals_pandas(ds_submission, pd_submission)


class TestMultipleLocAssignments:
    """
    Tests for multiple consecutive loc[] assignments.
    From cell-52 of notebook (Age banding).
    """

    def test_multiple_loc_age_banding(self, titanic_pd_df):
        """
        Test multiple consecutive loc[] conditional assignments for age banding.
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        ...
        From cell-52 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        # Fill NaN first
        pd_df['Age'] = pd_df['Age'].fillna(pd_df['Age'].median())

        ds = DataStore(titanic_pd_df.copy())
        ds['Age'] = ds['Age'].fillna(float(titanic_pd_df['Age'].median()))

        # Multiple consecutive loc assignments
        pd_df.loc[pd_df['Age'] <= 16, 'Age'] = 0
        pd_df.loc[(pd_df['Age'] > 16) & (pd_df['Age'] <= 32), 'Age'] = 1
        pd_df.loc[(pd_df['Age'] > 32) & (pd_df['Age'] <= 48), 'Age'] = 2
        pd_df.loc[(pd_df['Age'] > 48) & (pd_df['Age'] <= 64), 'Age'] = 3
        pd_df.loc[pd_df['Age'] > 64, 'Age'] = 4

        ds.loc[ds['Age'] <= 16, 'Age'] = 0
        ds.loc[(ds['Age'] > 16) & (ds['Age'] <= 32), 'Age'] = 1
        ds.loc[(ds['Age'] > 32) & (ds['Age'] <= 48), 'Age'] = 2
        ds.loc[(ds['Age'] > 48) & (ds['Age'] <= 64), 'Age'] = 3
        ds.loc[ds['Age'] > 64, 'Age'] = 4

        assert_datastore_equals_pandas(ds, pd_df)

    def test_multiple_loc_fare_banding(self, titanic_pd_df):
        """
        Test multiple consecutive loc[] conditional assignments for fare banding.
        From cell-73 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # Fare banding
        pd_df.loc[pd_df['Fare'] <= 7.91, 'Fare'] = 0
        pd_df.loc[(pd_df['Fare'] > 7.91) & (pd_df['Fare'] <= 14.454), 'Fare'] = 1
        pd_df.loc[(pd_df['Fare'] > 14.454) & (pd_df['Fare'] <= 31), 'Fare'] = 2
        pd_df.loc[pd_df['Fare'] > 31, 'Fare'] = 3

        ds.loc[ds['Fare'] <= 7.91, 'Fare'] = 0
        ds.loc[(ds['Fare'] > 7.91) & (ds['Fare'] <= 14.454), 'Fare'] = 1
        ds.loc[(ds['Fare'] > 14.454) & (ds['Fare'] <= 31), 'Fare'] = 2
        ds.loc[ds['Fare'] > 31, 'Fare'] = 3

        assert_datastore_equals_pandas(ds, pd_df)


class TestShapeAfterOperations:
    """
    Tests for shape property after various operations.
    From cell-32 of notebook showing before/after shapes.
    """

    def test_shape_after_drop(self, titanic_pd_df):
        """
        print("Before", train_df.shape, test_df.shape)
        train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
        "After", train_df.shape
        From cell-32 of notebook.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        # Before shapes
        assert ds.shape == pd_df.shape

        # After drop
        pd_df = pd_df.drop(['Ticket', 'Cabin'], axis=1)
        ds = ds.drop(['Ticket', 'Cabin'], axis=1)

        # After shapes
        assert ds.shape == pd_df.shape
        assert ds.shape[1] == pd_df.shape[1]  # Check column count specifically

    def test_shape_after_adding_columns(self, titanic_pd_df):
        """
        Test shape changes after adding new columns.
        """
        pd_df = titanic_pd_df.copy()
        ds = DataStore(titanic_pd_df.copy())

        initial_cols = pd_df.shape[1]

        # Add FamilySize
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        assert ds.shape[1] == pd_df.shape[1] == initial_cols + 1

        # Add IsAlone
        pd_df['IsAlone'] = 0
        ds['IsAlone'] = 0

        assert ds.shape[1] == pd_df.shape[1] == initial_cols + 2


class TestCombineDataFramesList:
    """
    Tests for combining DataFrames in a list pattern.
    combine = [train_df, test_df] from cell-4 of notebook.
    """

    def test_dataframe_list_iteration(self, titanic_pd_df):
        """
        Test iterating over a list of DataFrames/DataStores.
        for dataset in combine:
            dataset['NewCol'] = ...
        From cell-34, 36, 38, etc. of notebook.
        """
        pd_df1 = titanic_pd_df.head(100).copy()
        pd_df2 = titanic_pd_df.tail(100).copy()

        ds1 = DataStore(pd_df1.copy())
        ds2 = DataStore(pd_df2.copy())

        pd_combine = [pd_df1, pd_df2]
        ds_combine = [ds1, ds2]

        # Apply operation to both
        for pd_dataset in pd_combine:
            pd_dataset['FamilySize'] = pd_dataset['SibSp'] + pd_dataset['Parch'] + 1

        for ds_dataset in ds_combine:
            ds_dataset['FamilySize'] = ds_dataset['SibSp'] + ds_dataset['Parch'] + 1

        # Verify both DataStores match pandas
        assert_datastore_equals_pandas(ds_combine[0], pd_combine[0])
        assert_datastore_equals_pandas(ds_combine[1], pd_combine[1])
