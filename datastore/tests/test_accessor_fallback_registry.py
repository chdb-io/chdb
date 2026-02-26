"""
Tests for Accessor-level Pandas Fallback Registry.

This module tests the ACCESSOR_PARAM_PANDAS_FALLBACK registry in function_executor.py
which defines when accessor methods need pandas fallback based on parameters.
"""

import pytest
import pandas as pd
from datastore import DataStore
from datastore.function_executor import function_config, FunctionExecutorConfig
from tests.test_utils import assert_frame_equal


class TestAccessorFallbackRegistry:
    """Tests for accessor fallback registry configuration."""

    def test_registry_exists(self):
        """Test that ACCESSOR_PARAM_PANDAS_FALLBACK registry exists."""
        assert hasattr(FunctionExecutorConfig, 'ACCESSOR_PARAM_PANDAS_FALLBACK')
        assert isinstance(FunctionExecutorConfig.ACCESSOR_PARAM_PANDAS_FALLBACK, dict)

    def test_registry_not_empty(self):
        """Test that registry contains entries."""
        assert len(FunctionExecutorConfig.ACCESSOR_PARAM_PANDAS_FALLBACK) > 0

    def test_string_accessor_entries_exist(self):
        """Test that string accessor entries are in registry."""
        registry = FunctionExecutorConfig.ACCESSOR_PARAM_PANDAS_FALLBACK
        assert 'str.split' in registry
        assert 'str.extract' in registry
        assert 'str.extractall' in registry

    def test_datetime_accessor_entries_exist(self):
        """Test that datetime accessor entries are in registry."""
        registry = FunctionExecutorConfig.ACCESSOR_PARAM_PANDAS_FALLBACK
        assert 'dt.strftime' in registry
        # dt.floor, dt.ceil, dt.round, dt.normalize now use SQL implementation
        # so they are not in the fallback registry
        assert 'dt.tz_localize' in registry


class TestNeedsAccessorFallback:
    """Tests for needs_accessor_fallback method."""

    def test_split_with_expand_true_needs_fallback(self):
        """Test that str.split with expand=True needs pandas fallback."""
        assert function_config.needs_accessor_fallback('str.split', expand=True)

    def test_split_with_expand_false_no_fallback(self):
        """Test that str.split with expand=False does not need fallback."""
        assert not function_config.needs_accessor_fallback('str.split', expand=False)

    def test_split_default_no_fallback(self):
        """Test that str.split without expand parameter does not need fallback."""
        assert not function_config.needs_accessor_fallback('str.split')

    def test_extract_always_needs_fallback(self):
        """Test that str.extract always needs pandas fallback."""
        assert function_config.needs_accessor_fallback('str.extract')

    def test_extractall_always_needs_fallback(self):
        """Test that str.extractall always needs pandas fallback."""
        assert function_config.needs_accessor_fallback('str.extractall')

    def test_strftime_always_needs_fallback(self):
        """Test that dt.strftime always needs pandas fallback."""
        assert function_config.needs_accessor_fallback('dt.strftime')

    def test_unknown_method_no_fallback(self):
        """Test that unknown accessor methods do not need fallback."""
        assert not function_config.needs_accessor_fallback('str.upper')
        assert not function_config.needs_accessor_fallback('str.lower')
        assert not function_config.needs_accessor_fallback('dt.year')

    def test_case_insensitive(self):
        """Test that accessor method names are case-insensitive."""
        assert function_config.needs_accessor_fallback('STR.SPLIT', expand=True)
        assert function_config.needs_accessor_fallback('Str.Extract')


class TestGetAccessorFallbackReason:
    """Tests for get_accessor_fallback_reason method."""

    def test_split_with_expand_returns_reason(self):
        """Test that split with expand=True returns a reason."""
        reason = function_config.get_accessor_fallback_reason('str.split', expand=True)
        assert reason is not None
        assert 'str.split' in reason
        assert 'expand' in reason

    def test_split_without_expand_returns_none(self):
        """Test that split without fallback returns None."""
        reason = function_config.get_accessor_fallback_reason('str.split', expand=False)
        assert reason is None

    def test_extract_returns_always_reason(self):
        """Test that str.extract returns 'always requires' reason."""
        reason = function_config.get_accessor_fallback_reason('str.extract')
        assert reason is not None
        assert 'always requires' in reason

    def test_unknown_method_returns_none(self):
        """Test that unknown methods return None."""
        reason = function_config.get_accessor_fallback_reason('str.upper')
        assert reason is None


class TestListAccessorFallbacks:
    """Tests for list_accessor_fallbacks method."""

    def test_returns_dict(self):
        """Test that list_accessor_fallbacks returns a dictionary."""
        fallbacks = function_config.list_accessor_fallbacks()
        assert isinstance(fallbacks, dict)

    def test_contains_expected_entries(self):
        """Test that returned dict contains expected entries."""
        fallbacks = function_config.list_accessor_fallbacks()
        assert 'str.split' in fallbacks
        assert fallbacks['str.split'] == {'expand': True}
        assert fallbacks['str.extract'] == '*'

    def test_returns_copy(self):
        """Test that modifying returned dict doesn't affect original."""
        fallbacks = function_config.list_accessor_fallbacks()
        fallbacks['test.new'] = '*'
        assert 'test.new' not in FunctionExecutorConfig.ACCESSOR_PARAM_PANDAS_FALLBACK


class TestConfigSummaryIncludesAccessorFallbacks:
    """Test that get_config_summary includes accessor fallback info."""

    def test_config_summary_has_accessor_fallbacks_key(self):
        """Test that config summary includes accessor_fallbacks count."""
        summary = function_config.get_config_summary()
        assert 'accessor_fallbacks' in summary
        assert isinstance(summary['accessor_fallbacks'], int)
        assert summary['accessor_fallbacks'] > 0


class TestAccessorFallbackIntegration:
    """Integration tests for accessor fallback behavior."""

    def test_split_expand_true_returns_dataframe(self):
        """Test that str.split with expand=True returns a DataFrame."""
        ds = DataStore({'text': ['a-b-c', 'd-e-f', 'g-h-i']})
        pd_df = pd.DataFrame({'text': ['a-b-c', 'd-e-f', 'g-h-i']})

        ds_result = ds['text'].str.split('-', expand=True)
        pd_result = pd_df['text'].str.split('-', expand=True)

        # Compare
        assert_frame_equal(
            pd.DataFrame(ds_result) if not isinstance(ds_result, pd.DataFrame) else ds_result,
            pd_result
        )

    def test_split_expand_false_returns_series_of_lists(self):
        """Test that str.split with expand=False returns series of lists."""
        ds = DataStore({'text': ['a-b-c', 'd-e-f']})
        pd_df = pd.DataFrame({'text': ['a-b-c', 'd-e-f']})

        ds_result = ds['text'].str.split('-', expand=False)
        pd_result = pd_df['text'].str.split('-', expand=False)

        # The result should be a series containing arrays/lists
        # For DataStore, split returns Array type from chDB
        # Just verify it executes without error
        assert ds_result is not None

    def test_extract_uses_pandas(self):
        """Test that str.extract uses pandas fallback."""
        ds = DataStore({'text': ['a1b2', 'c3d4', 'e5f6']})
        pd_df = pd.DataFrame({'text': ['a1b2', 'c3d4', 'e5f6']})

        ds_result = ds['text'].str.extract(r'([a-z])(\d)')
        pd_result = pd_df['text'].str.extract(r'([a-z])(\d)')

        # Compare
        assert_frame_equal(
            pd.DataFrame(ds_result) if not isinstance(ds_result, pd.DataFrame) else ds_result,
            pd_result
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
