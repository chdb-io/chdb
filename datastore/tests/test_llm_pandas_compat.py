"""
LLM/NLP Pandas Compatibility Test Suite
=======================================

Tests pandas operations commonly used in Kaggle LLM and NLP competitions,
comparing datastore behavior with pandas for API consistency.

Based on analysis of popular Kaggle LLM/NLP notebooks including:
- LLM Classification Finetuning Competition
- LLM - Detect AI Generated Text Competition
- NLP text preprocessing notebooks
- Transformers and fine-tuning workflows

Common LLM/NLP data preparation patterns tested:
1. Loading multiple datasets and combining them
2. Text preprocessing (cleaning, lowercasing, deduplication)
3. Train/test splitting with flags
4. String operations for text cleaning
5. Duplicate detection and removal
6. Missing value handling in text columns
7. Feature engineering for text data
8. Creating submission DataFrames

Design Principle:
    Tests use natural execution triggers (.values, __eq__, len(), repr())
    following the lazy execution design principle.
    Avoid explicit _execute() calls - use natural triggers instead.
"""

from tests.test_utils import assert_frame_equal, assert_series_equal, get_series
import pytest
import pandas as pd
import numpy as np
import datastore as ds
import tempfile
import os


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def text_df_data():
    """DataFrame with text data for preprocessing tests."""
    return {
        'text': ['Hello World!', 'MACHINE Learning.', 'Deep-Learning', 'NLP Tasks?', 'Transformers!!!'],
        'id': [1, 2, 3, 4, 5],
    }


@pytest.fixture
def dup_df_data():
    """DataFrame with duplicates for duplicate detection tests."""
    return {
        'id': [1, 2, 3, 2, 4, 3],
        'text': ['a', 'b', 'c', 'b', 'd', 'c'],
        'label': [0, 1, 0, 1, 1, 0],
    }


@pytest.fixture
def na_text_df_data():
    """DataFrame with missing text values."""
    return {
        'text': ['Hello', None, 'World', np.nan, 'Test'],
        'id': [1, 2, 3, 4, 5],
    }


@pytest.fixture
def train_test_csv_files():
    """Create temporary train and test CSV files for loading tests."""
    train_data = pd.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'text': ['Hello world', 'Machine learning', 'Deep learning', 'NLP tasks', 'Transformers'],
            'label': [0, 1, 1, 0, 1],
        }
    )
    test_data = pd.DataFrame(
        {
            'id': [6, 7, 8],
            'text': ['Test sentence', 'Another test', 'Final test'],
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, 'train_llm.csv')
        test_path = os.path.join(tmpdir, 'test_llm.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        yield train_path, test_path


# ============================================================================
# Category 1: Multi-Dataset Loading and Combining (Very Common in LLM Tasks)
# ============================================================================


class TestMultiDatasetLoading:
    """Tests for loading and combining multiple datasets."""

    def test_concat_with_flag(self, train_test_csv_files):
        """Test loading train/test CSV, adding flag column, and concatenating."""
        train_path, test_path = train_test_csv_files

        # Pandas version
        pd_train = pd.read_csv(train_path)
        pd_test = pd.read_csv(test_path)
        pd_train['train_test'] = 1
        pd_test['train_test'] = 0
        pd_result = pd.concat([pd_train, pd_test], ignore_index=True)

        # DataStore version
        ds_train = ds.read_csv(train_path)
        ds_test = ds.read_csv(test_path)
        # Convert to pandas for column assignment and concat
        ds_train_df = ds_train.to_pandas()
        ds_test_df = ds_test.to_pandas()
        ds_train_df['train_test'] = 1
        ds_test_df['train_test'] = 0
        ds_result = pd.concat([ds_train_df, ds_test_df], ignore_index=True)

        assert pd_result.equals(ds_result)


# ============================================================================
# Category 2: Text Preprocessing Operations (Core NLP Pattern)
# ============================================================================


class TestTextPreprocessing:
    """Tests for text preprocessing operations."""

    def test_lowercase(self, text_df_data):
        """Test converting text to lowercase with str.lower()."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df['text'].str.lower()
        ds_result = ds_df['text'].str.lower()

        ds_result = get_series(ds_result)

        assert_series_equal(pd_result, ds_result)

    def test_remove_punctuation_regex(self, text_df_data):
        """Test removing punctuation with str.replace() regex."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df['text'].str.replace(r'[^\w\s]', '', regex=True)
        ds_result = ds_df['text'].str.replace(r'[^\w\s]', '', regex=True)

        ds_result = get_series(ds_result)

        assert_series_equal(pd_result, ds_result)

    def test_strip_whitespace(self):
        """Test stripping whitespace with str.strip()."""
        data = {'text': ['  hello  ', ' world ', '  test  ']}
        pd_df = pd.DataFrame(data)
        ds_df = ds.DataFrame(data)

        pd_result = pd_df['text'].str.strip()
        ds_result = ds_df['text'].str.strip()

        ds_result = get_series(ds_result)

        assert_series_equal(pd_result, ds_result)

    def test_text_length(self, text_df_data):
        """Test getting text length with str.len()."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df['text'].str.len()
        ds_result = ds_df['text'].str.len()

        ds_result = get_series(ds_result)

        np.testing.assert_array_equal(pd_result.values, ds_result.values)


# ============================================================================
# Category 3: Duplicate Detection and Removal (Data Quality)
# ============================================================================


class TestDuplicateDetection:
    """Tests for duplicate detection and removal."""

    def test_duplicated_subset(self, dup_df_data):
        """Test detecting duplicates in specific column."""
        pd_df = pd.DataFrame(dup_df_data)
        ds_df = ds.DataFrame(dup_df_data)

        pd_result = pd_df.duplicated(subset=['id'])
        ds_result = ds_df.duplicated(subset=['id'])

        ds_result = get_series(ds_result)

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_duplicated_sum(self, dup_df_data):
        """Test counting duplicates with duplicated().sum()."""
        pd_df = pd.DataFrame(dup_df_data)
        ds_df = ds.DataFrame(dup_df_data)

        pd_result = pd_df.duplicated(subset=['id']).sum()
        ds_result = ds_df.duplicated(subset=['id'])

        ds_result = get_series(ds_result).sum()

        assert pd_result == ds_result


# ============================================================================
# Category 4: Missing Value Handling in Text Columns
# ============================================================================


class TestMissingValueHandling:
    """Tests for missing value handling in text columns."""

    def test_dropna_subset(self, na_text_df_data):
        """Test dropping rows with missing text using dropna(subset=)."""
        pd_df = pd.DataFrame(na_text_df_data)
        ds_df = ds.DataFrame(na_text_df_data)

        pd_result = pd_df.dropna(subset=['text'])
        ds_result = ds_df.dropna(subset=['text'])

        ds_result = get_series(ds_result)

        assert_frame_equal(pd_result.reset_index(drop=True), ds_result.reset_index(drop=True))

    def test_fillna_text(self, na_text_df_data):
        """Test filling missing text with empty string."""
        pd_df = pd.DataFrame(na_text_df_data)
        ds_df = ds.DataFrame(na_text_df_data)

        pd_df_copy = pd_df.copy()
        pd_df_copy['text'] = pd_df_copy['text'].fillna('')
        ds_result = ds_df.fillna({'text': ''})

        ds_result = get_series(ds_result)

        assert_frame_equal(pd_df_copy, ds_result)


# ============================================================================
# Category 5: Feature Engineering for Text Data
# ============================================================================


class TestTextFeatureEngineering:
    """Tests for feature engineering on text data."""

    def test_word_count(self, text_df_data):
        """Test counting words with str.split().str.len()."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df['text'].str.split().str.len()
        ds_result = ds_df['text'].str.split().str.len()

        ds_result = get_series(ds_result)

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_contains_case_insensitive(self, text_df_data):
        """Test checking if text contains word with str.contains(case=False)."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df['text'].str.contains('learning', case=False)
        ds_result = ds_df['text'].str.contains('learning', case=False)

        ds_result = get_series(ds_result)

        np.testing.assert_array_equal(pd_result.values, ds_result.values)

    def test_extract_first_word(self, text_df_data):
        """Test extracting first word with str.split().str[0]."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df['text'].str.split().str[0]
        ds_result = ds_df['text'].str.split().str[0]

        ds_result = get_series(ds_result)

        np.testing.assert_array_equal(pd_result.values, ds_result.values)


# ============================================================================
# Category 6: Creating Submission DataFrames (Competition Pattern)
# ============================================================================


class TestSubmissionCreation:
    """Tests for creating submission DataFrames."""

    def test_create_submission(self):
        """Test creating submission DataFrame from predictions."""
        predictions = [0, 1, 1, 0, 1]
        ids = [1, 2, 3, 4, 5]

        pd_submission = pd.DataFrame({'id': ids, 'prediction': predictions})
        ds_submission = ds.DataFrame({'id': ids, 'prediction': predictions})

        ds_submission = get_series(ds_submission)

        assert_frame_equal(pd_submission, ds_submission)


# ============================================================================
# Category 7: Sampling and Data Splitting
# ============================================================================


class TestSamplingAndSplitting:
    """Tests for sampling and data splitting operations."""

    def test_sample_n(self, text_df_data):
        """Test sampling n rows with sample(n=3, random_state=42)."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df.sample(n=3, random_state=42)
        ds_result = ds_df.sample(n=3, random_state=42)

        ds_result = get_series(ds_result)

        assert_frame_equal(pd_result.reset_index(drop=True), ds_result.reset_index(drop=True))

    def test_sample_frac(self, text_df_data):
        """Test sampling fraction of rows with sample(frac=0.6)."""
        pd_df = pd.DataFrame(text_df_data)
        ds_df = ds.DataFrame(text_df_data)

        pd_result = pd_df.sample(frac=0.6, random_state=42)
        ds_result = ds_df.sample(frac=0.6, random_state=42)

        ds_result = get_series(ds_result)

        assert_frame_equal(pd_result.reset_index(drop=True), ds_result.reset_index(drop=True))
