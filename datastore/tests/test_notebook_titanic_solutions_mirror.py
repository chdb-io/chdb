"""
Mirror Test for Titanic Data Science Solutions Notebook
========================================================

Tests pandas operations found in the titanic-data-science-solutions.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Based on: https://www.kaggle.com/code/startupsci/titanic-data-science-solutions

Operations tested:
- DataFrame construction and basic info
- str.extract() for regex pattern extraction
- crosstab() for cross-tabulation
- map() with dictionary for categorical encoding
- replace() with list of values
- loc[] with conditional assignment
- pd.cut() for binning continuous values
- pd.qcut() for quantile-based binning
- drop() columns
- fillna() with mode values
- Column arithmetic and feature engineering
- groupby with multiple aggregations
- sort_values()

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
    assert_frame_equal,
)
import os


# ============================================================================
# Test Data Path
# ============================================================================


def dataset_path(filename: str) -> str:
    """Get path to test dataset."""
    return os.path.join(os.path.dirname(__file__), "dataset", filename)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def titanic_pd_df():
    """Load Titanic dataset as pandas DataFrame."""
    return pd.read_csv(dataset_path("Titanic-Dataset.csv"))


@pytest.fixture
def titanic_ds(titanic_pd_df):
    """Load Titanic dataset as DataStore."""
    return DataStore.from_file(dataset_path("Titanic-Dataset.csv"))


@pytest.fixture
def sample_df_with_names():
    """Sample data with Name column for str.extract() testing."""
    return pd.DataFrame({
        'Name': [
            'Braund, Mr. Owen Harris',
            'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',
            'Heikkinen, Miss. Laina',
            'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
            'Allen, Mr. William Henry',
            'Moran, Mr. James',
        ],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male'],
        'Survived': [0, 1, 1, 1, 0, 0],
    })


# ============================================================================
# Test: String Extract (str.extract)
# ============================================================================


class TestStrExtract:
    """Tests for str.extract() operation as used in the notebook."""

    def test_str_extract_title(self, sample_df_with_names):
        """
        Extract Title from Name using regex.

        From notebook: dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
        """
        # pandas
        pd_df = sample_df_with_names.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # DataStore
        ds = DataStore(sample_df_with_names.copy())
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        assert_datastore_equals_pandas(ds, pd_df)

    def test_str_extract_title_from_titanic(self, titanic_pd_df, titanic_ds):
        """
        Extract Title from real Titanic data.
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Compare Title column
        assert_series_equal(
            ds['Title'],
            pd_df['Title'],
            check_names=True
        )


# ============================================================================
# Test: Crosstab
# ============================================================================


class TestCrosstab:
    """Tests for pd.crosstab() operation as used in the notebook."""

    def test_crosstab_title_sex(self, sample_df_with_names):
        """
        Create crosstab of Title vs Sex.

        From notebook: pd.crosstab(train_df['Title'], train_df['Sex'])
        """
        # pandas - create Title first
        pd_df = sample_df_with_names.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_result = pd.crosstab(pd_df['Title'], pd_df['Sex'])

        # DataStore - use pandas crosstab with DataStore columns
        ds = DataStore(sample_df_with_names.copy())
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # crosstab uses values directly, triggers execution
        ds_result = pd.crosstab(ds['Title'].values, ds['Sex'].values)
        ds_result.index.name = 'Title'
        ds_result.columns.name = 'Sex'

        assert_frame_equal(ds_result, pd_result)


# ============================================================================
# Test: Map with Dictionary
# ============================================================================


class TestMapDict:
    """Tests for map() with dictionary as used in the notebook."""

    def test_map_sex_to_numeric(self, titanic_pd_df):
        """
        Map Sex to numeric values.

        From notebook: dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # Compare Sex column
        assert_series_equal(
            ds['Sex'],
            pd_df['Sex'],
            check_names=True
        )

    def test_map_embarked_to_numeric(self, titanic_pd_df):
        """
        Map Embarked to numeric values.

        From notebook: dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        """
        # pandas - fill NA first
        pd_df = titanic_pd_df.copy()
        pd_df['Embarked'] = pd_df['Embarked'].fillna('S')
        pd_df['Embarked'] = pd_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Embarked'] = ds['Embarked'].fillna('S')
        ds['Embarked'] = ds['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        assert_series_equal(
            ds['Embarked'],
            pd_df['Embarked'],
            check_names=True
        )

    def test_map_title_to_numeric(self, sample_df_with_names):
        """
        Map Title to numeric values.

        From notebook: title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        """
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3}

        # pandas
        pd_df = sample_df_with_names.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_df['Title'] = pd_df['Title'].map(title_mapping)
        pd_df['Title'] = pd_df['Title'].fillna(0)

        # DataStore
        ds = DataStore(sample_df_with_names.copy())
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Title'].map(title_mapping)
        ds['Title'] = ds['Title'].fillna(0)

        assert_series_equal(
            ds['Title'],
            pd_df['Title'],
            check_names=True
        )


# ============================================================================
# Test: Replace with List
# ============================================================================


class TestReplaceList:
    """Tests for replace() with list of values as used in the notebook."""

    def test_replace_rare_titles(self, sample_df_with_names):
        """
        Replace multiple titles with 'Rare'.

        From notebook: dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', ...], 'Rare')
        """
        # pandas
        pd_df = sample_df_with_names.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_df['Title'] = pd_df['Title'].replace(['Mr', 'Mrs'], 'Common')

        # DataStore
        ds = DataStore(sample_df_with_names.copy())
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Title'].replace(['Mr', 'Mrs'], 'Common')

        assert_series_equal(
            ds['Title'],
            pd_df['Title'],
            check_names=True
        )

    def test_replace_single_value(self, sample_df_with_names):
        """
        Replace single value.

        From notebook: dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        """
        # pandas
        pd_df = sample_df_with_names.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_df['Title'] = pd_df['Title'].replace('Miss', 'Ms')

        # DataStore
        ds = DataStore(sample_df_with_names.copy())
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Title'].replace('Miss', 'Ms')

        assert_series_equal(
            ds['Title'],
            pd_df['Title'],
            check_names=True
        )


# ============================================================================
# Test: loc[] with Conditional Assignment
# ============================================================================


class TestLocConditionalAssignment:
    """Tests for loc[] with conditional assignment as used in the notebook."""

    def test_loc_age_band_assignment(self, titanic_pd_df):
        """
        Assign Age bands using loc[].

        From notebook:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Age'] = pd_df['Age'].fillna(pd_df['Age'].median())
        pd_df.loc[pd_df['Age'] <= 16, 'AgeBand'] = 0
        pd_df.loc[(pd_df['Age'] > 16) & (pd_df['Age'] <= 32), 'AgeBand'] = 1
        pd_df.loc[(pd_df['Age'] > 32) & (pd_df['Age'] <= 48), 'AgeBand'] = 2
        pd_df.loc[(pd_df['Age'] > 48) & (pd_df['Age'] <= 64), 'AgeBand'] = 3
        pd_df.loc[pd_df['Age'] > 64, 'AgeBand'] = 4

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Age'] = ds['Age'].fillna(ds['Age'].median())
        ds.loc[ds['Age'] <= 16, 'AgeBand'] = 0
        ds.loc[(ds['Age'] > 16) & (ds['Age'] <= 32), 'AgeBand'] = 1
        ds.loc[(ds['Age'] > 32) & (ds['Age'] <= 48), 'AgeBand'] = 2
        ds.loc[(ds['Age'] > 48) & (ds['Age'] <= 64), 'AgeBand'] = 3
        ds.loc[ds['Age'] > 64, 'AgeBand'] = 4

        assert_series_equal(
            ds['AgeBand'],
            pd_df['AgeBand'],
            check_names=True
        )


# ============================================================================
# Test: pd.cut() for Binning
# ============================================================================


class TestPdCut:
    """Tests for pd.cut() binning as used in the notebook."""

    def test_cut_age_bands(self, titanic_pd_df):
        """
        Create Age bands using pd.cut().

        From notebook: train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Age'] = pd_df['Age'].fillna(pd_df['Age'].median())
        pd_df['AgeBand'] = pd.cut(pd_df['Age'], 5)

        # DataStore - cut is not directly supported, use pandas operation
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Age'] = ds['Age'].fillna(ds['Age'].median())
        # For cut, we need to execute and use pandas
        ds_df = ds.to_df()
        ds_df['AgeBand'] = pd.cut(ds_df['Age'], 5)

        # Compare AgeBand categories
        assert list(pd_df['AgeBand'].cat.categories) == list(ds_df['AgeBand'].cat.categories)

    def test_qcut_fare_bands(self, titanic_pd_df):
        """
        Create Fare bands using pd.qcut().

        From notebook: train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['FareBand'] = pd.qcut(pd_df['Fare'], 4)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_df = ds.to_df()
        ds_df['FareBand'] = pd.qcut(ds_df['Fare'], 4)

        # Compare FareBand categories
        assert list(pd_df['FareBand'].cat.categories) == list(ds_df['FareBand'].cat.categories)


# ============================================================================
# Test: Drop Columns
# ============================================================================


class TestDropColumns:
    """Tests for drop() columns as used in the notebook."""

    def test_drop_single_column(self, titanic_pd_df):
        """
        Drop a single column.

        From notebook: train_df = train_df.drop(['Ticket'], axis=1)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_result = pd_df.drop(['Ticket'], axis=1)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_result = ds.drop(['Ticket'], axis=1)

        assert list(ds_result.columns) == list(pd_result.columns)
        assert 'Ticket' not in ds_result.columns

    def test_drop_multiple_columns(self, titanic_pd_df):
        """
        Drop multiple columns.

        From notebook: train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_result = pd_df.drop(['Ticket', 'Cabin'], axis=1)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_result = ds.drop(['Ticket', 'Cabin'], axis=1)

        assert list(ds_result.columns) == list(pd_result.columns)
        assert 'Ticket' not in ds_result.columns
        assert 'Cabin' not in ds_result.columns

    def test_drop_name_passengerid(self, titanic_pd_df):
        """
        Drop Name and PassengerId columns.

        From notebook: train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_result = pd_df.drop(['Name', 'PassengerId'], axis=1)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_result = ds.drop(['Name', 'PassengerId'], axis=1)

        assert list(ds_result.columns) == list(pd_result.columns)


# ============================================================================
# Test: fillna with mode
# ============================================================================


class TestFillnaMode:
    """Tests for fillna() with mode() as used in the notebook."""

    def test_fillna_embarked_mode(self, titanic_pd_df):
        """
        Fill Embarked NaN with mode.

        From notebook:
        freq_port = train_df.Embarked.dropna().mode()[0]
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        freq_port = pd_df['Embarked'].dropna().mode()[0]
        pd_df['Embarked'] = pd_df['Embarked'].fillna(freq_port)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        # Use the same mode value
        ds['Embarked'] = ds['Embarked'].fillna(freq_port)

        # No NaN should remain
        assert ds['Embarked'].isna().sum() == 0
        assert pd_df['Embarked'].isna().sum() == 0

        # Compare values
        assert_series_equal(
            ds['Embarked'],
            pd_df['Embarked'],
            check_names=True
        )


# ============================================================================
# Test: Feature Engineering
# ============================================================================


class TestFeatureEngineering:
    """Tests for feature engineering as used in the notebook."""

    def test_create_family_size(self, titanic_pd_df):
        """
        Create FamilySize = SibSp + Parch + 1.

        From notebook: dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        assert_series_equal(
            ds['FamilySize'],
            pd_df['FamilySize'],
            check_names=True
        )

    def test_create_is_alone(self, titanic_pd_df):
        """
        Create IsAlone feature.

        From notebook:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        pd_df['IsAlone'] = 0
        pd_df.loc[pd_df['FamilySize'] == 1, 'IsAlone'] = 1

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        ds['IsAlone'] = 0
        ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1

        assert_series_equal(
            ds['IsAlone'],
            pd_df['IsAlone'],
            check_names=True
        )

    def test_create_age_times_class(self, titanic_pd_df):
        """
        Create Age*Class feature.

        From notebook: dataset['Age*Class'] = dataset.Age * dataset.Pclass
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Age'] = pd_df['Age'].fillna(pd_df['Age'].median())
        pd_df['Age*Class'] = pd_df['Age'] * pd_df['Pclass']

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Age'] = ds['Age'].fillna(ds['Age'].median())
        ds['Age*Class'] = ds['Age'] * ds['Pclass']

        assert_series_equal(
            ds['Age*Class'],
            pd_df['Age*Class'],
            check_names=True,
            check_dtype=False  # May have different int/float types
        )


# ============================================================================
# Test: GroupBy with Sort
# ============================================================================


class TestGroupBySorted:
    """Tests for groupby with sort_values as used in the notebook."""

    def test_groupby_pclass_survived_mean_sorted(self, titanic_pd_df):
        """
        GroupBy Pclass, mean Survived, sorted descending.

        From notebook: train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
                       .sort_values(by='Survived', ascending=False)
        """
        # pandas
        pd_result = titanic_pd_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
        pd_result = pd_result.sort_values(by='Survived', ascending=False)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_result = ds[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
        ds_result = ds_result.sort_values(by='Survived', ascending=False)

        # Compare with row order check since we explicitly sorted
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_sex_survived_mean_sorted(self, titanic_pd_df):
        """
        GroupBy Sex, mean Survived, sorted descending.

        From notebook: train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
                       .sort_values(by='Survived', ascending=False)
        """
        # pandas
        pd_result = titanic_pd_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
        pd_result = pd_result.sort_values(by='Survived', ascending=False)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_result = ds[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
        ds_result = ds_result.sort_values(by='Survived', ascending=False)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_sibsp_survived_mean_sorted(self, titanic_pd_df):
        """
        GroupBy SibSp, mean Survived, sorted descending.

        From notebook: train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
                       .sort_values(by='Survived', ascending=False)

        Note: Uses kind='stable' because SibSp=5 and SibSp=8 both have Survived=0.0,
        creating a tie. Without stable sort, the order is platform-dependent.
        """
        # pandas
        pd_result = titanic_pd_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
        pd_result = pd_result.sort_values(by='Survived', ascending=False, kind='stable')

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_result = ds[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
        ds_result = ds_result.sort_values(by='Survived', ascending=False, kind='stable')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)

    def test_groupby_familysize_survived_mean_sorted(self, titanic_pd_df):
        """
        GroupBy FamilySize, mean Survived, sorted descending.

        From notebook:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
            .sort_values(by='Survived', ascending=False)

        Note: Uses kind='stable' because FamilySize=8 and FamilySize=11 both have Survived=0.0,
        creating a tie. Without stable sort, the order is platform-dependent.
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        pd_result = pd_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
        pd_result = pd_result.sort_values(by='Survived', ascending=False, kind='stable')

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        ds_result = ds[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
        ds_result = ds_result.sort_values(by='Survived', ascending=False, kind='stable')

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=True)


# ============================================================================
# Test: GroupBy with as_index=False
# ============================================================================


class TestGroupByAsIndexFalse:
    """Tests for groupby with as_index=False parameter."""

    def test_groupby_title_survived_mean(self, sample_df_with_names):
        """
        GroupBy Title, mean Survived with as_index=False.

        From notebook: train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        """
        # pandas
        pd_df = sample_df_with_names.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_result = pd_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

        # DataStore
        ds = DataStore(sample_df_with_names.copy())
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds_result = ds[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

        # as_index=False means Title should be a column, not index
        assert 'Title' in list(ds_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ============================================================================
# Test: describe(include=['O'])
# ============================================================================


class TestDescribeObject:
    """Tests for describe(include=['O']) as used in the notebook."""

    def test_describe_object_columns(self, titanic_pd_df):
        """
        Describe object columns only.

        From notebook: train_df.describe(include=['O'])
        """
        # pandas
        pd_result = titanic_pd_df.describe(include=['O'])

        # DataStore - describe proxies to pandas
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds_df = ds.to_df()
        ds_result = ds_df.describe(include=['O'])

        assert_frame_equal(ds_result, pd_result)


# ============================================================================
# Test: Complete Workflow - Title Processing Chain
# ============================================================================


class TestTitleProcessingWorkflow:
    """Tests for the complete Title processing workflow from the notebook."""

    def test_full_title_workflow(self, titanic_pd_df):
        """
        Complete Title processing workflow:
        1. Extract Title from Name
        2. Replace rare titles with 'Rare'
        3. Map to numeric values
        4. GroupBy analysis
        """
        # Define title mapping
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                       'Rev', 'Sir', 'Jonkheer', 'Dona']

        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Title'] = pd_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        pd_df['Title'] = pd_df['Title'].replace(rare_titles, 'Rare')
        pd_df['Title'] = pd_df['Title'].replace('Mlle', 'Miss')
        pd_df['Title'] = pd_df['Title'].replace('Ms', 'Miss')
        pd_df['Title'] = pd_df['Title'].replace('Mme', 'Mrs')
        pd_df['Title'] = pd_df['Title'].map(title_mapping)
        pd_df['Title'] = pd_df['Title'].fillna(0)

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Title'].replace(rare_titles, 'Rare')
        ds['Title'] = ds['Title'].replace('Mlle', 'Miss')
        ds['Title'] = ds['Title'].replace('Ms', 'Miss')
        ds['Title'] = ds['Title'].replace('Mme', 'Mrs')
        ds['Title'] = ds['Title'].map(title_mapping)
        ds['Title'] = ds['Title'].fillna(0)

        assert_series_equal(
            ds['Title'],
            pd_df['Title'],
            check_names=True,
            check_dtype=False  # int/float differences acceptable
        )


# ============================================================================
# Test: Fare Binning Workflow
# ============================================================================


class TestFareBinningWorkflow:
    """Tests for Fare binning workflow from the notebook."""

    def test_fare_binning_conditional_assignment(self, titanic_pd_df):
        """
        Fare binning using loc[] conditional assignment.

        From notebook:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        etc.
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df['Fare'] = pd_df['Fare'].fillna(pd_df['Fare'].median())
        pd_df['FareBand'] = pd_df['Fare'].copy()
        pd_df.loc[pd_df['Fare'] <= 7.91, 'FareBand'] = 0
        pd_df.loc[(pd_df['Fare'] > 7.91) & (pd_df['Fare'] <= 14.454), 'FareBand'] = 1
        pd_df.loc[(pd_df['Fare'] > 14.454) & (pd_df['Fare'] <= 31), 'FareBand'] = 2
        pd_df.loc[pd_df['Fare'] > 31, 'FareBand'] = 3

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds['Fare'] = ds['Fare'].fillna(ds['Fare'].median())
        ds['FareBand'] = ds['Fare']
        ds.loc[ds['Fare'] <= 7.91, 'FareBand'] = 0
        ds.loc[(ds['Fare'] > 7.91) & (ds['Fare'] <= 14.454), 'FareBand'] = 1
        ds.loc[(ds['Fare'] > 14.454) & (ds['Fare'] <= 31), 'FareBand'] = 2
        ds.loc[ds['Fare'] > 31, 'FareBand'] = 3

        assert_series_equal(
            ds['FareBand'],
            pd_df['FareBand'],
            check_names=True,
            check_dtype=False
        )


# ============================================================================
# Test: Combined Feature Engineering Workflow
# ============================================================================


class TestCombinedFeatureWorkflow:
    """Tests for combined feature engineering workflow."""

    def test_complete_feature_engineering(self, titanic_pd_df):
        """
        Complete feature engineering workflow as in notebook:
        1. Drop Ticket, Cabin
        2. Create Title feature
        3. Create FamilySize
        4. Create IsAlone
        5. Create Age*Class
        """
        # pandas
        pd_df = titanic_pd_df.copy()
        pd_df = pd_df.drop(['Ticket', 'Cabin'], axis=1)
        pd_df['Age'] = pd_df['Age'].fillna(pd_df['Age'].median())
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        pd_df['IsAlone'] = 0
        pd_df.loc[pd_df['FamilySize'] == 1, 'IsAlone'] = 1
        pd_df['Age*Class'] = pd_df['Age'] * pd_df['Pclass']

        # DataStore
        ds = DataStore.from_file(dataset_path("Titanic-Dataset.csv"))
        ds = ds.drop(['Ticket', 'Cabin'], axis=1)
        ds['Age'] = ds['Age'].fillna(ds['Age'].median())
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        ds['IsAlone'] = 0
        ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
        ds['Age*Class'] = ds['Age'] * ds['Pclass']

        # Compare columns match
        assert set(ds.columns) == set(pd_df.columns)

        # Compare key features
        assert_series_equal(ds['FamilySize'], pd_df['FamilySize'], check_names=True)
        assert_series_equal(ds['IsAlone'], pd_df['IsAlone'], check_names=True)
        assert_series_equal(ds['Age*Class'], pd_df['Age*Class'], check_names=True, check_dtype=False)
