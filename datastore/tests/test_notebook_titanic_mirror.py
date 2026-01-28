"""
Mirror Test for Titanic Data Science Solutions Notebook
=======================================================

Tests pandas operations found in the titanic-data-science-solutions.ipynb notebook,
comparing DataStore behavior with pandas for API consistency.

Operations tested from the notebook:
- Basic inspection: columns, shape, head(), tail(), info-like operations
- describe() for numeric and object columns
- groupby().mean().sort_values() pattern
- str.extract() for extracting titles from names
- map() for value replacement
- fillna() for handling missing values
- value_counts() and mode()
- crosstab() equivalent operations
- pd.cut() for binning continuous values
- pd.qcut() for quantile-based binning
- loc[] for conditional assignment
- drop() for removing columns
- Arithmetic operations for feature creation

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
# Fixtures with Titanic-like data structure
# ============================================================================


@pytest.fixture
def titanic_df():
    """Sample data mimicking Titanic dataset structure."""
    return pd.DataFrame({
        'PassengerId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Survived': [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        'Pclass': [3, 1, 3, 1, 3, 3, 1, 3, 3, 2],
        'Name': [
            'Braund, Mr. Owen Harris',
            'Cumings, Mrs. John Bradley',
            'Heikkinen, Miss. Laina',
            'Futrelle, Mrs. Jacques Heath',
            'Allen, Mr. William Henry',
            'Moran, Mr. James',
            'McCarthy, Mr. Timothy J',
            'Palsson, Master. Gosta Leonard',
            'Johnson, Mrs. Oscar',
            'Nasser, Mrs. Nicholas'
        ],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'],
        'Age': [22.0, 38.0, 26.0, 35.0, 35.0, np.nan, 54.0, 2.0, 27.0, 14.0],
        'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
        'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2 3101282', '113803', '373450',
                   '330877', '17463', '349909', '347742', '237736'],
        'Fare': [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan, np.nan, np.nan],
        'Embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C'],
    })


@pytest.fixture
def titanic_extended_df():
    """Extended Titanic-like data for more comprehensive testing."""
    np.random.seed(42)
    n = 100
    
    # Generate Age with NaN values
    age_values = np.random.uniform(1, 80, n)
    age_mask = np.random.random(n) < 0.2
    age_values = np.where(age_mask, np.nan, age_values)
    
    # Generate Fare with NaN values
    fare_values = np.random.exponential(30, n)
    fare_mask = np.random.random(n) < 0.01
    fare_values = np.where(fare_mask, np.nan, fare_values)
    
    # Generate Embarked with NaN values (use None for string column)
    embarked_choices = np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09])
    embarked_mask = np.random.random(n) < 0.02
    embarked_values = [None if mask else val for val, mask in zip(embarked_choices, embarked_mask)]
    
    return pd.DataFrame({
        'PassengerId': range(1, n + 1),
        'Survived': np.random.choice([0, 1], n, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
        'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
        'Age': age_values,
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.68, 0.23, 0.05, 0.02, 0.015, 0.005]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.76, 0.13, 0.08, 0.015, 0.01, 0.003, 0.002]),
        'Fare': fare_values,
        'Embarked': embarked_values,
    })


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicInspection:
    """Tests for basic DataFrame inspection operations from the notebook."""

    def test_columns_values(self, titanic_df):
        """Test print(train_df.columns.values) equivalent."""
        ds = DataStore(titanic_df)
        # columns.values returns numpy array in pandas
        np.testing.assert_array_equal(ds.columns.values, titanic_df.columns.values)

    def test_head_default(self, titanic_df):
        """Test head() with default 5 rows."""
        ds = DataStore(titanic_df)
        pd_result = titanic_df.head()
        ds_result = ds.head()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_tail_default(self, titanic_df):
        """Test tail() with default 5 rows."""
        ds = DataStore(titanic_df)
        pd_result = titanic_df.tail()
        ds_result = ds.tail()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_shape_property(self, titanic_df):
        """Test shape property."""
        ds = DataStore(titanic_df)
        assert ds.shape == titanic_df.shape


class TestDescribeOperations:
    """Tests for describe() operations from the notebook."""

    def test_describe_numeric(self, titanic_df):
        """Test describe() for numeric columns - default behavior."""
        ds = DataStore(titanic_df)
        pd_result = titanic_df.describe()
        ds_result = ds.describe()
        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_describe_object(self, titanic_df):
        """Test describe(include=['O']) for object columns."""
        ds = DataStore(titanic_df)
        pd_result = titanic_df.describe(include=['O'])
        ds_result = ds.describe(include=['O'])
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestGroupByPivotFeatures:
    """Tests for groupby operations used in feature analysis from notebook."""

    def test_groupby_pclass_survived_mean(self, titanic_df):
        """Test: train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()"""
        ds = DataStore(titanic_df)

        # pandas
        pd_result = titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

        # DataStore
        ds_result = ds[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_groupby_sex_survived_mean(self, titanic_df):
        """Test: train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()"""
        ds = DataStore(titanic_df)

        pd_result = titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
        ds_result = ds[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_groupby_sibsp_survived_mean(self, titanic_df):
        """Test: train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()"""
        ds = DataStore(titanic_df)

        pd_result = titanic_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
        ds_result = ds[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_groupby_parch_survived_mean(self, titanic_df):
        """Test: train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()"""
        ds = DataStore(titanic_df)

        pd_result = titanic_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
        ds_result = ds[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_groupby_mean_sort_values(self, titanic_df):
        """Test: groupby().mean().sort_values(by='Survived', ascending=False)"""
        ds = DataStore(titanic_df)

        # pandas
        pd_result = (titanic_df[['Pclass', 'Survived']]
                     .groupby(['Pclass'], as_index=False)
                     .mean()
                     .sort_values(by='Survived', ascending=False))

        # DataStore
        ds_result = (ds[['Pclass', 'Survived']]
                     .groupby(['Pclass'], as_index=False)
                     .mean()
                     .sort_values(by='Survived', ascending=False))

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestStringExtraction:
    """Tests for str.extract() operation used to extract titles from names."""

    def test_str_extract_title(self, titanic_df):
        """Test: dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)"""
        ds = DataStore(titanic_df)

        # pandas - extract title from name
        pd_result = titanic_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

        # DataStore
        ds_result = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_str_extract_creates_new_column(self, titanic_df):
        """Test assigning str.extract result to new column."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Extract title and assign to new column
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Compare the new column
        assert_datastore_equals_pandas(ds['Title'], pd_df['Title'])


class TestMapReplacement:
    """Tests for map() operation used for value replacement."""

    def test_map_sex_to_numeric(self, titanic_df):
        """Test: dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        pd_result = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
        ds_result = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_map_embarked_to_numeric(self, titanic_df):
        """Test: dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # fillna first to avoid NaN issues
        pd_df['Embarked'] = pd_df['Embarked'].fillna('S')
        ds['Embarked'] = ds['Embarked'].fillna('S')

        pd_result = pd_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        ds_result = ds['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestFillnaOperations:
    """Tests for fillna() operations from notebook."""

    def test_fillna_age_with_median(self, titanic_df):
        """Test filling Age NaN with median."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        age_median = pd_df['Age'].median()

        pd_df['Age'] = pd_df['Age'].fillna(age_median)
        ds['Age'] = ds['Age'].fillna(age_median)

        # Verify no NaN
        assert ds['Age'].isna().sum() == pd_df['Age'].isna().sum()
        assert ds['Age'].isna().sum() == 0

        # Verify values match
        assert_datastore_equals_pandas(ds['Age'], pd_df['Age'])

    def test_fillna_embarked_with_mode(self, titanic_extended_df):
        """Test filling Embarked NaN with mode (most frequent value)."""
        pd_df = titanic_extended_df.copy()
        ds = DataStore(titanic_extended_df)

        # Get mode - mode() returns a Series, take first value
        freq_port = pd_df['Embarked'].mode()[0]

        pd_df['Embarked'] = pd_df['Embarked'].fillna(freq_port)
        ds['Embarked'] = ds['Embarked'].fillna(freq_port)

        assert_datastore_equals_pandas(ds['Embarked'], pd_df['Embarked'])


class TestValueCountsAndMode:
    """Tests for value_counts() and mode() operations."""

    def test_value_counts(self, titanic_df):
        """Test value_counts() operation."""
        ds = DataStore(titanic_df)

        pd_result = titanic_df['Sex'].value_counts()
        ds_result = ds['Sex'].value_counts()

        # Sort both for comparison since value_counts order may differ
        assert_datastore_equals_pandas(ds_result.sort_index(), pd_result.sort_index())

    def test_dropna_mode(self, titanic_extended_df):
        """Test: train_df.Embarked.dropna().mode()[0]"""
        ds = DataStore(titanic_extended_df)

        pd_mode = titanic_extended_df['Embarked'].dropna().mode()[0]
        ds_mode = ds['Embarked'].dropna().mode()[0]

        assert ds_mode == pd_mode


class TestDropOperation:
    """Tests for drop() operation to remove columns."""

    def test_drop_single_column(self, titanic_df):
        """Test: train_df.drop(['Ticket'], axis=1)"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        pd_result = pd_df.drop(['Ticket'], axis=1)
        ds_result = ds.drop(['Ticket'], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_multiple_columns(self, titanic_df):
        """Test: train_df.drop(['Ticket', 'Cabin'], axis=1)"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        pd_result = pd_df.drop(['Ticket', 'Cabin'], axis=1)
        ds_result = ds.drop(['Ticket', 'Cabin'], axis=1)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestArithmeticFeatureCreation:
    """Tests for arithmetic operations used to create new features."""

    def test_family_size_creation(self, titanic_df):
        """Test: dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        assert_datastore_equals_pandas(ds['FamilySize'], pd_df['FamilySize'])

    def test_age_class_interaction(self, titanic_df):
        """Test: dataset['Age*Class'] = dataset.Age * dataset.Pclass"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Fill Age NaN first
        age_median = pd_df['Age'].median()
        pd_df['Age'] = pd_df['Age'].fillna(age_median)
        ds['Age'] = ds['Age'].fillna(age_median)

        pd_df['Age*Class'] = pd_df['Age'] * pd_df['Pclass']
        ds['Age*Class'] = ds['Age'] * ds['Pclass']

        assert_datastore_equals_pandas(ds['Age*Class'], pd_df['Age*Class'])


class TestIsAloneFeature:
    """Tests for creating IsAlone feature."""

    def test_is_alone_creation(self, titanic_df):
        """Test creating IsAlone feature with loc[] assignment."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Create FamilySize
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        # Create IsAlone using loc
        pd_df['IsAlone'] = 0
        pd_df.loc[pd_df['FamilySize'] == 1, 'IsAlone'] = 1

        ds['IsAlone'] = 0
        ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1

        assert_datastore_equals_pandas(ds['IsAlone'], pd_df['IsAlone'])


class TestGroupByWithNewFeatures:
    """Tests for groupby operations on newly created features."""

    def test_family_size_survived_groupby(self, titanic_df):
        """Test groupby on FamilySize feature."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Create FamilySize
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        # Groupby
        pd_result = (pd_df[['FamilySize', 'Survived']]
                     .groupby(['FamilySize'], as_index=False)
                     .mean()
                     .sort_values(by='Survived', ascending=False))

        ds_result = (ds[['FamilySize', 'Survived']]
                     .groupby(['FamilySize'], as_index=False)
                     .mean()
                     .sort_values(by='Survived', ascending=False))

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)

    def test_is_alone_survived_groupby(self, titanic_df):
        """Test groupby on IsAlone feature."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Create FamilySize and IsAlone
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        pd_df['IsAlone'] = 0
        pd_df.loc[pd_df['FamilySize'] == 1, 'IsAlone'] = 1

        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
        ds['IsAlone'] = 0
        ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1

        # Groupby
        pd_result = pd_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
        ds_result = ds[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

        assert_datastore_equals_pandas(ds_result, pd_result, rtol=1e-5)


class TestTitleProcessing:
    """Tests for title extraction and processing workflow from notebook."""

    def test_title_extraction_and_replacement(self, titanic_df):
        """Test full title extraction and replacement workflow."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Extract title
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Replace rare titles with 'Rare'
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        pd_df['Title'] = pd_df['Title'].replace(rare_titles, 'Rare')
        ds['Title'] = ds['Title'].replace(rare_titles, 'Rare')

        # Replace equivalent titles
        pd_df['Title'] = pd_df['Title'].replace('Mlle', 'Miss')
        pd_df['Title'] = pd_df['Title'].replace('Ms', 'Miss')
        pd_df['Title'] = pd_df['Title'].replace('Mme', 'Mrs')

        ds['Title'] = ds['Title'].replace('Mlle', 'Miss')
        ds['Title'] = ds['Title'].replace('Ms', 'Miss')
        ds['Title'] = ds['Title'].replace('Mme', 'Mrs')

        assert_datastore_equals_pandas(ds['Title'], pd_df['Title'])

    def test_title_to_numeric_mapping(self, titanic_df):
        """Test mapping titles to numeric values."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Extract title
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Map titles to numeric
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        pd_df['Title'] = pd_df['Title'].map(title_mapping)
        pd_df['Title'] = pd_df['Title'].fillna(0)

        ds['Title'] = ds['Title'].map(title_mapping)
        ds['Title'] = ds['Title'].fillna(0)

        assert_datastore_equals_pandas(ds['Title'], pd_df['Title'])


class TestLocConditionalAssignment:
    """Tests for loc[] conditional assignment operations from notebook."""

    def test_loc_age_banding(self, titanic_df):
        """Test age banding using loc[] conditional assignment."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Fill NaN first
        age_median = pd_df['Age'].median()
        pd_df['Age'] = pd_df['Age'].fillna(age_median)
        ds['Age'] = ds['Age'].fillna(age_median)

        # Convert Age to int
        pd_df['Age'] = pd_df['Age'].astype(int)
        ds['Age'] = ds['Age'].astype(int)

        # Age banding
        pd_df.loc[pd_df['Age'] <= 16, 'AgeBand'] = 0
        pd_df.loc[(pd_df['Age'] > 16) & (pd_df['Age'] <= 32), 'AgeBand'] = 1
        pd_df.loc[(pd_df['Age'] > 32) & (pd_df['Age'] <= 48), 'AgeBand'] = 2
        pd_df.loc[(pd_df['Age'] > 48) & (pd_df['Age'] <= 64), 'AgeBand'] = 3
        pd_df.loc[pd_df['Age'] > 64, 'AgeBand'] = 4

        ds.loc[ds['Age'] <= 16, 'AgeBand'] = 0
        ds.loc[(ds['Age'] > 16) & (ds['Age'] <= 32), 'AgeBand'] = 1
        ds.loc[(ds['Age'] > 32) & (ds['Age'] <= 48), 'AgeBand'] = 2
        ds.loc[(ds['Age'] > 48) & (ds['Age'] <= 64), 'AgeBand'] = 3
        ds.loc[ds['Age'] > 64, 'AgeBand'] = 4

        assert_datastore_equals_pandas(ds['AgeBand'], pd_df['AgeBand'])

    def test_loc_fare_banding(self, titanic_df):
        """Test fare banding using loc[] conditional assignment."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Fare banding
        pd_df.loc[pd_df['Fare'] <= 7.91, 'FareBand'] = 0
        pd_df.loc[(pd_df['Fare'] > 7.91) & (pd_df['Fare'] <= 14.454), 'FareBand'] = 1
        pd_df.loc[(pd_df['Fare'] > 14.454) & (pd_df['Fare'] <= 31), 'FareBand'] = 2
        pd_df.loc[pd_df['Fare'] > 31, 'FareBand'] = 3

        ds.loc[ds['Fare'] <= 7.91, 'FareBand'] = 0
        ds.loc[(ds['Fare'] > 7.91) & (ds['Fare'] <= 14.454), 'FareBand'] = 1
        ds.loc[(ds['Fare'] > 14.454) & (ds['Fare'] <= 31), 'FareBand'] = 2
        ds.loc[ds['Fare'] > 31, 'FareBand'] = 3

        assert_datastore_equals_pandas(ds['FareBand'], pd_df['FareBand'])


class TestComprehensiveTitanicWorkflow:
    """Test complete workflow from the notebook."""

    def test_mini_titanic_pipeline(self, titanic_df):
        """Test a mini version of the complete Titanic preprocessing pipeline."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Step 1: Drop Ticket and Cabin
        pd_df = pd_df.drop(['Ticket', 'Cabin'], axis=1)
        ds = ds.drop(['Ticket', 'Cabin'], axis=1)

        # Step 2: Extract Title
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Step 3: Map Title to numeric
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}
        pd_df['Title'] = pd_df['Title'].map(title_mapping).fillna(5)
        ds['Title'] = ds['Title'].map(title_mapping).fillna(5)

        # Step 4: Drop Name
        pd_df = pd_df.drop(['Name'], axis=1)
        ds = ds.drop(['Name'], axis=1)

        # Step 5: Convert Sex to numeric
        pd_df['Sex'] = pd_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
        ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        # Step 6: Fill Age with median
        age_median = pd_df['Age'].median()
        pd_df['Age'] = pd_df['Age'].fillna(age_median)
        ds['Age'] = ds['Age'].fillna(age_median)

        # Step 7: Create FamilySize
        pd_df['FamilySize'] = pd_df['SibSp'] + pd_df['Parch'] + 1
        ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1

        # Step 8: Create IsAlone
        pd_df['IsAlone'] = (pd_df['FamilySize'] == 1).astype(int)
        ds['IsAlone'] = (ds['FamilySize'] == 1).astype(int)

        # Step 9: Convert Embarked to numeric
        pd_df['Embarked'] = pd_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0).astype(int)
        ds['Embarked'] = ds['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0).astype(int)

        # Final comparison (allow dtype differences for nullable int columns)
        assert_datastore_equals_pandas(ds, pd_df, check_nullable_dtype=False)


class TestCrosstabEquivalent:
    """Tests for crosstab-equivalent operations."""

    def test_title_sex_groupby(self, titanic_df):
        """Test title-sex cross tabulation equivalent using groupby."""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Extract title
        pd_df['Title'] = pd_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
        ds['Title'] = ds['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Groupby Title and Sex, count
        pd_result = pd_df.groupby(['Title', 'Sex']).size()
        ds_result = ds.groupby(['Title', 'Sex']).size()

        # Sort for comparison
        assert_datastore_equals_pandas(
            ds_result.sort_index(),
            pd_result.sort_index()
        )


class TestDropnaOperations:
    """Tests for dropna() operations used in notebook."""

    def test_dropna_single_column(self, titanic_df):
        """Test dropna() on single column for age guessing."""
        ds = DataStore(titanic_df)

        # Get Age without NaN
        pd_result = titanic_df['Age'].dropna()
        ds_result = ds['Age'].dropna()

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_median(self, titanic_extended_df):
        """Test: dataset['Age'].dropna().median()"""
        ds = DataStore(titanic_extended_df)

        pd_median = titanic_extended_df['Age'].dropna().median()
        ds_median = ds['Age'].dropna().median()

        np.testing.assert_almost_equal(ds_median, pd_median, decimal=5)


class TestAsTypeConversions:
    """Tests for astype() type conversion operations."""

    def test_age_astype_int(self, titanic_df):
        """Test: dataset['Age'].astype(int)"""
        pd_df = titanic_df.copy()
        ds = DataStore(titanic_df)

        # Fill NaN first (astype(int) fails on NaN)
        age_median = pd_df['Age'].median()
        pd_df['Age'] = pd_df['Age'].fillna(age_median)
        ds['Age'] = ds['Age'].fillna(age_median)

        pd_result = pd_df['Age'].astype(int)
        ds_result = ds['Age'].astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sex_map_astype_int(self, titanic_df):
        """Test: map().astype(int) chain."""
        ds = DataStore(titanic_df)

        pd_result = titanic_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
        ds_result = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestNotebookStatistics:
    """Tests for statistical operations used in notebook analysis."""

    def test_survival_rate_by_pclass(self, titanic_extended_df):
        """Test computing survival rate by Pclass."""
        ds = DataStore(titanic_extended_df)

        pd_result = titanic_extended_df.groupby('Pclass')['Survived'].mean()
        ds_result = ds.groupby('Pclass')['Survived'].mean()

        # Compare values
        for idx in pd_result.index:
            np.testing.assert_almost_equal(ds_result[idx], pd_result[idx], decimal=5)

    def test_survival_rate_by_sex(self, titanic_extended_df):
        """Test computing survival rate by Sex."""
        ds = DataStore(titanic_extended_df)

        pd_result = titanic_extended_df.groupby('Sex')['Survived'].mean()
        ds_result = ds.groupby('Sex')['Survived'].mean()

        for idx in pd_result.index:
            np.testing.assert_almost_equal(ds_result[idx], pd_result[idx], decimal=5)
