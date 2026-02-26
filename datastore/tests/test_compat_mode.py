"""
Tests for CompatMode configuration infrastructure.

Tests the compat_mode config dimension: get/set/validation, auto-setting of
execution engine, DataStoreConfig property access, and convenience methods.
"""

import unittest

from datastore.config import (
    CompatMode,
    get_compat_mode,
    set_compat_mode,
    is_performance_mode,
    use_performance_mode,
    use_pandas_compat,
    get_execution_engine,
    ExecutionEngine,
    set_execution_engine,
    config,
)


class TestCompatModeConfig(unittest.TestCase):
    """Test compat_mode config get/set/validation."""

    def setUp(self):
        """Reset to default pandas mode before each test."""
        set_compat_mode(CompatMode.PANDAS)
        set_execution_engine(ExecutionEngine.AUTO)

    def tearDown(self):
        """Reset to default pandas mode after each test."""
        set_compat_mode(CompatMode.PANDAS)
        set_execution_engine(ExecutionEngine.AUTO)

    def test_default_mode_is_pandas(self):
        """Default compat mode should be 'pandas'."""
        assert get_compat_mode() == CompatMode.PANDAS
        assert get_compat_mode() == "pandas"

    def test_set_performance_mode(self):
        """Setting performance mode should change the compat mode."""
        set_compat_mode(CompatMode.PERFORMANCE)
        assert get_compat_mode() == CompatMode.PERFORMANCE
        assert get_compat_mode() == "performance"

    def test_set_pandas_mode(self):
        """Setting pandas mode should change the compat mode."""
        set_compat_mode(CompatMode.PERFORMANCE)
        set_compat_mode(CompatMode.PANDAS)
        assert get_compat_mode() == CompatMode.PANDAS

    def test_is_performance_mode_returns_correct_value(self):
        """is_performance_mode() should reflect current mode."""
        assert not is_performance_mode()
        set_compat_mode(CompatMode.PERFORMANCE)
        assert is_performance_mode()
        set_compat_mode(CompatMode.PANDAS)
        assert not is_performance_mode()

    def test_invalid_mode_raises_value_error(self):
        """Invalid mode string should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            set_compat_mode("invalid_mode")
        assert "Invalid compat mode" in str(ctx.exception)
        # Verify mode didn't change
        assert get_compat_mode() == CompatMode.PANDAS

    def test_performance_mode_auto_sets_execution_engine_to_chdb(self):
        """Setting performance mode should auto-set execution engine to chDB."""
        assert get_execution_engine() == ExecutionEngine.AUTO
        set_compat_mode(CompatMode.PERFORMANCE)
        assert get_execution_engine() == ExecutionEngine.CHDB

    def test_pandas_mode_does_not_change_execution_engine(self):
        """Setting pandas mode should not change the execution engine."""
        set_execution_engine(ExecutionEngine.AUTO)
        set_compat_mode(CompatMode.PANDAS)
        assert get_execution_engine() == ExecutionEngine.AUTO

    def test_use_performance_mode_convenience(self):
        """use_performance_mode() convenience function should work."""
        use_performance_mode()
        assert is_performance_mode()
        assert get_compat_mode() == CompatMode.PERFORMANCE

    def test_use_pandas_compat_convenience(self):
        """use_pandas_compat() convenience function should work."""
        set_compat_mode(CompatMode.PERFORMANCE)
        use_pandas_compat()
        assert not is_performance_mode()
        assert get_compat_mode() == CompatMode.PANDAS


class TestDataStoreConfigCompatMode(unittest.TestCase):
    """Test compat_mode via DataStoreConfig instance (config object)."""

    def setUp(self):
        set_compat_mode(CompatMode.PANDAS)
        set_execution_engine(ExecutionEngine.AUTO)

    def tearDown(self):
        set_compat_mode(CompatMode.PANDAS)
        set_execution_engine(ExecutionEngine.AUTO)

    def test_config_compat_mode_property_get(self):
        """config.compat_mode should return current mode."""
        assert config.compat_mode == CompatMode.PANDAS

    def test_config_compat_mode_property_set(self):
        """config.compat_mode = 'performance' should set mode."""
        config.compat_mode = CompatMode.PERFORMANCE
        assert config.compat_mode == CompatMode.PERFORMANCE
        assert is_performance_mode()

    def test_config_set_compat_mode_method(self):
        """config.set_compat_mode() should work."""
        config.set_compat_mode(CompatMode.PERFORMANCE)
        assert config.compat_mode == CompatMode.PERFORMANCE

    def test_config_use_performance_mode(self):
        """config.use_performance_mode() convenience method."""
        config.use_performance_mode()
        assert is_performance_mode()
        assert config.compat_mode == CompatMode.PERFORMANCE

    def test_config_use_pandas_compat(self):
        """config.use_pandas_compat() convenience method."""
        config.use_performance_mode()
        config.use_pandas_compat()
        assert not is_performance_mode()
        assert config.compat_mode == CompatMode.PANDAS


class TestCompatModeExports(unittest.TestCase):
    """Test that compat_mode symbols are exported from datastore package."""

    def test_imports_from_datastore(self):
        """All compat_mode symbols should be importable from datastore."""
        from datastore import (
            CompatMode as CM,
            set_compat_mode as scm,
            get_compat_mode as gcm,
            is_performance_mode as ipm,
            use_performance_mode as upm,
            use_pandas_compat as upc,
        )
        assert CM.PANDAS == "pandas"
        assert CM.PERFORMANCE == "performance"
        assert callable(scm)
        assert callable(gcm)
        assert callable(ipm)
        assert callable(upm)
        assert callable(upc)


if __name__ == "__main__":
    unittest.main()
