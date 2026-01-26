"""
Tests for slice step support: df[::step], df[start:stop:step]

Mirror Pattern: All tests compare DataStore behavior with pandas behavior.
"""

import pandas as pd
import pytest
from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


class TestSliceStepBasic:
    """Basic slice step operations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataStore and pandas DataFrame."""
        data = {'a': list(range(10)), 'b': list(range(10, 20))}
        return DataStore(data), pd.DataFrame(data)

    def test_step_2_from_start(self, sample_data):
        """Test df[::2] - every 2nd row starting from index 0."""
        ds, pd_df = sample_data

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_2_from_index_1(self, sample_data):
        """Test df[1::2] - every 2nd row starting from index 1."""
        ds, pd_df = sample_data

        pd_result = pd_df[1::2]
        ds_result = ds[1::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_3(self, sample_data):
        """Test df[::3] - every 3rd row."""
        ds, pd_df = sample_data

        pd_result = pd_df[::3]
        ds_result = ds[::3]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_with_start_and_stop(self, sample_data):
        """Test df[1:8:2] - from index 1 to 8, step 2."""
        ds, pd_df = sample_data

        pd_result = pd_df[1:8:2]
        ds_result = ds[1:8:2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_5(self, sample_data):
        """Test df[::5] - every 5th row."""
        ds, pd_df = sample_data

        pd_result = pd_df[::5]
        ds_result = ds[::5]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSliceStepReverse:
    """Reverse slice operations with negative step."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataStore and pandas DataFrame."""
        data = {'a': list(range(10)), 'b': list(range(10, 20))}
        return DataStore(data), pd.DataFrame(data)

    def test_reverse_all(self, sample_data):
        """Test df[::-1] - reverse all rows."""
        ds, pd_df = sample_data

        pd_result = pd_df[::-1]
        ds_result = ds[::-1]

        # Note: reverse slicing preserves original index in pandas
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reverse_step_2(self, sample_data):
        """Test df[::-2] - reverse every 2nd row."""
        ds, pd_df = sample_data

        pd_result = pd_df[::-2]
        ds_result = ds[::-2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reverse_with_bounds(self, sample_data):
        """Test df[8:2:-2] - reverse from 8 to 2, step -2."""
        ds, pd_df = sample_data

        pd_result = pd_df[8:2:-2]
        ds_result = ds[8:2:-2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_reverse_with_start_only(self, sample_data):
        """Test df[5::-1] - reverse from index 5 to beginning."""
        ds, pd_df = sample_data

        pd_result = pd_df[5::-1]
        ds_result = ds[5::-1]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSliceStepEdgeCases:
    """Edge cases for slice step operations."""

    def test_step_larger_than_length(self):
        """Test step larger than DataFrame length."""
        data = {'a': [1, 2, 3]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::10]
        ds_result = ds[::10]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_on_single_row(self):
        """Test step slicing on single row DataFrame."""
        data = {'a': [1]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_on_empty_dataframe(self):
        """Test step slicing on empty DataFrame."""
        data = {'a': []}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_1_no_effect(self):
        """Test df[::1] - step of 1 should return all rows."""
        data = {'a': [1, 2, 3, 4, 5]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::1]
        ds_result = ds[::1]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_start_beyond_length(self):
        """Test start index beyond DataFrame length."""
        data = {'a': [1, 2, 3]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[10::2]
        ds_result = ds[10::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSliceStepChained:
    """Test slice step combined with other operations."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataStore and pandas DataFrame."""
        data = {'a': list(range(10)), 'b': list(range(10, 20))}
        return DataStore(data), pd.DataFrame(data)

    def test_step_then_filter(self, sample_data):
        """Test df[::2] followed by filtering."""
        ds, pd_df = sample_data

        pd_result = pd_df[::2]
        pd_result = pd_result[pd_result['a'] > 2]

        ds_result = ds[::2]
        ds_result = ds_result[ds_result['a'] > 2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_then_select_columns(self, sample_data):
        """Test df[::2] followed by column selection."""
        ds, pd_df = sample_data

        pd_result = pd_df[::2][['a']]
        ds_result = ds[::2][['a']]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_then_step(self, sample_data):
        """Test filtering followed by df[::2]."""
        ds, pd_df = sample_data

        pd_result = pd_df[pd_df['a'] > 2][::2]
        ds_result = ds[ds['a'] > 2][::2]

        assert_datastore_equals_pandas(ds_result, pd_result)


class TestSliceStepMultipleColumns:
    """Test slice step with various column types."""

    def test_step_with_string_column(self):
        """Test step slicing with string columns."""
        data = {'name': ['a', 'b', 'c', 'd', 'e'], 'value': [1, 2, 3, 4, 5]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_with_float_column(self):
        """Test step slicing with float columns."""
        data = {'value': [1.1, 2.2, 3.3, 4.4, 5.5]}
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[::2]
        ds_result = ds[::2]

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_step_with_mixed_types(self):
        """Test step slicing with mixed column types."""
        data = {
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
        }
        ds = DataStore(data)
        pd_df = pd.DataFrame(data)

        pd_result = pd_df[1::2]
        ds_result = ds[1::2]

        assert_datastore_equals_pandas(ds_result, pd_result)
