"""
Tests for chdb_settings configuration.

Verifies:
- Default settings are applied to chdb connections
- User can override settings before any connection
- Settings are frozen after first connection
- Settings propagate through both per-DataStore and global executor paths

Note: chdb server-level settings (e.g. memory_worker_correct_memory_tracker) are
fixed for the lifetime of the process once the first chdb.connect() initializes the
engine. Tests that verify non-default server settings must run in a subprocess to
get a fresh chdb engine.
"""

import importlib
import subprocess
import sys
import textwrap
import pytest
from datastore.config import get_chdb_settings, set_chdb_setting
from datastore.connection import Connection
from datastore.executor import get_executor, reset_executor

# datastore.__init__ exports `config` (a DataStoreConfig instance), which shadows
# the `datastore.config` module. Use importlib to get the actual module object.
_config_module = importlib.import_module('datastore.config')


def _reset_chdb_settings_state():
    """Reset chdb_settings and frozen flag on the real module."""
    _config_module._chdb_settings = {
        'memory_worker_correct_memory_tracker': 1,
        'max_server_memory_usage': 0,
        'max_server_memory_usage_to_ram_ratio': 0,
    }
    _config_module._chdb_settings_frozen = False


def _run_chdb_in_subprocess(script: str) -> None:
    """Run a Python script in a fresh subprocess for isolated chdb engine state.

    Server-level settings are baked into the chdb engine on first connect and
    cannot be changed within the same process.  A subprocess guarantees a fresh
    engine so we can verify non-default values actually take effect.
    """
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


@pytest.fixture(autouse=True)
def reset_chdb_state():
    """Reset chdb_settings and executor state before each test."""
    reset_executor()
    _reset_chdb_settings_state()
    yield
    reset_executor()
    _reset_chdb_settings_state()


class TestChdbSettingsDefaults:
    """Verify default chdb_settings values."""

    def test_default_settings(self):
        settings = get_chdb_settings()
        assert settings == {
            'memory_worker_correct_memory_tracker': 1,
            'max_server_memory_usage': 0,
            'max_server_memory_usage_to_ram_ratio': 0,
        }

    def test_default_applied_to_connection(self):
        conn = Connection(':memory:')
        conn.connect()
        try:
            r = conn._conn.query(
                "SELECT value FROM system.server_settings "
                "WHERE name = 'memory_worker_correct_memory_tracker'",
                'CSV',
            )
            assert '"1"' in str(r)
        finally:
            conn.close()

    def test_default_memory_limits_applied_to_connection(self):
        conn = Connection(':memory:')
        conn.connect()
        try:
            r = conn._conn.query(
                "SELECT name, value FROM system.server_settings "
                "WHERE name IN ('max_server_memory_usage', 'max_server_memory_usage_to_ram_ratio') "
                "ORDER BY name",
                'CSV',
            )
            result = str(r)
            assert '"max_server_memory_usage","0"' in result
            assert '"max_server_memory_usage_to_ram_ratio","0"' in result
        finally:
            conn.close()


class TestChdbSettingsOverride:
    """Verify user can override settings before connection."""

    def test_set_chdb_setting_before_connect(self):
        set_chdb_setting('memory_worker_correct_memory_tracker', 0)
        assert get_chdb_settings()['memory_worker_correct_memory_tracker'] == 0

    def test_override_applied_to_connection(self):
        _run_chdb_in_subprocess("""
            from datastore.config import set_chdb_setting
            from datastore.connection import Connection

            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

            conn = Connection(':memory:')
            conn.connect()
            try:
                r = conn._conn.query(
                    "SELECT value FROM system.server_settings "
                    "WHERE name = 'memory_worker_correct_memory_tracker'",
                    'CSV',
                )
                assert '"0"' in str(r), f"Expected '0' but got: {r}"
            finally:
                conn.close()
        """)

    def test_add_custom_setting(self):
        set_chdb_setting('max_server_memory_usage', 1073741824)
        settings = get_chdb_settings()
        assert settings['max_server_memory_usage'] == 1073741824
        assert settings['memory_worker_correct_memory_tracker'] == 1


class TestChdbSettingsFrozen:
    """Verify settings are frozen after first connection."""

    def test_frozen_after_connect(self):
        conn = Connection(':memory:')
        conn.connect()
        conn.close()

        with pytest.raises(RuntimeError, match="cannot be changed after a connection"):
            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

    def test_frozen_after_global_executor_connect(self):
        executor = get_executor()
        executor._ensure_connected()

        with pytest.raises(RuntimeError, match="cannot be changed after a connection"):
            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

    def test_not_frozen_before_any_connect(self):
        assert _config_module._chdb_settings_frozen is False
        set_chdb_setting('memory_worker_correct_memory_tracker', 0)
        set_chdb_setting('memory_worker_correct_memory_tracker', 1)

    def test_get_chdb_settings_returns_copy(self):
        settings = get_chdb_settings()
        settings['memory_worker_correct_memory_tracker'] = 999
        assert get_chdb_settings()['memory_worker_correct_memory_tracker'] == 1


class TestChdbSettingsDataStoreConfig:
    """Verify DataStoreConfig interface for chdb_settings."""

    def test_config_chdb_settings_property(self):
        from datastore.config import config
        assert config.chdb_settings == {
            'memory_worker_correct_memory_tracker': 1,
            'max_server_memory_usage': 0,
            'max_server_memory_usage_to_ram_ratio': 0,
        }

    def test_config_set_chdb_setting(self):
        from datastore.config import config
        config.set_chdb_setting('memory_worker_correct_memory_tracker', 0)
        assert config.chdb_settings['memory_worker_correct_memory_tracker'] == 0

    def test_config_set_chdb_setting_frozen(self):
        from datastore.config import config
        conn = Connection(':memory:')
        conn.connect()
        conn.close()

        with pytest.raises(RuntimeError, match="cannot be changed after a connection"):
            config.set_chdb_setting('memory_worker_correct_memory_tracker', 0)


class TestChdbSettingsGlobalExecutor:
    """Verify settings propagate through global executor path."""

    def test_global_executor_uses_default_settings(self):
        executor = get_executor()
        executor._ensure_connected()

        r = executor.connection._conn.query(
            "SELECT value FROM system.server_settings "
            "WHERE name = 'memory_worker_correct_memory_tracker'",
            'CSV',
        )
        assert '"1"' in str(r)

    def test_global_executor_uses_overridden_settings(self):
        _run_chdb_in_subprocess("""
            from datastore.config import set_chdb_setting
            from datastore.executor import get_executor

            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

            executor = get_executor()
            executor._ensure_connected()

            r = executor.connection._conn.query(
                "SELECT value FROM system.server_settings "
                "WHERE name = 'memory_worker_correct_memory_tracker'",
                'CSV',
            )
            assert '"0"' in str(r), f"Expected '0' but got: {r}"
        """)


class TestChdbSettingsDataStoreInternal:
    """Verify settings propagate through DataStore's internal connection creation."""

    def test_default_setting_via_datastore_dataframe_query(self):
        """DataStore with in-memory DataFrame uses global executor path."""
        import pandas as pd
        from datastore import DataStore

        pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        ds = DataStore(pd_df)

        # Trigger execution (goes through global executor → Connection.connect())
        pd_result = pd_df[pd_df['a'] > 1]
        ds_result = ds[ds['a'] > 1]

        assert list(ds_result['a']) == list(pd_result['a'])
        assert list(ds_result['b']) == list(pd_result['b'])

        # Verify default setting was applied
        executor = get_executor()
        executor._ensure_connected()
        r = executor.connection._conn.query(
            "SELECT value FROM system.server_settings "
            "WHERE name = 'memory_worker_correct_memory_tracker'",
            'CSV',
        )
        assert '"1"' in str(r)

        # Verify settings are now frozen
        with pytest.raises(RuntimeError, match="cannot be changed after a connection"):
            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

    def test_override_setting_via_datastore_dataframe_query(self):
        """Override setting before DataStore query, verify it takes effect."""
        _run_chdb_in_subprocess("""
            import pandas as pd
            from datastore import DataStore
            from datastore.config import set_chdb_setting
            from datastore.executor import get_executor

            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

            pd_df = pd.DataFrame({'x': [1, 2, 3]})
            ds = DataStore(pd_df)

            pd_result = pd_df[pd_df['x'] > 1]
            ds_result = ds[ds['x'] > 1]
            assert list(ds_result['x']) == list(pd_result['x'])

            executor = get_executor()
            executor._ensure_connected()
            r = executor.connection._conn.query(
                "SELECT value FROM system.server_settings "
                "WHERE name = 'memory_worker_correct_memory_tracker'",
                'CSV',
            )
            assert '"0"' in str(r), f"Expected '0' but got: {r}"
        """)

    def test_default_setting_via_datastore_file_source(self, tmp_path):
        """DataStore with file source creates per-instance connection."""
        import pandas as pd
        from datastore import DataStore

        csv_path = tmp_path / "test.csv"
        pd.DataFrame({'a': [1, 2], 'b': [3, 4]}).to_csv(csv_path, index=False)

        ds = DataStore("file", path=str(csv_path))
        pd_result = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        assert list(ds['a']) == list(pd_result['a'])
        assert list(ds['b']) == list(pd_result['b'])

        # Verify setting was applied on per-DataStore connection
        r = ds._executor.connection._conn.query(
            "SELECT value FROM system.server_settings "
            "WHERE name = 'memory_worker_correct_memory_tracker'",
            'CSV',
        )
        assert '"1"' in str(r)

        # Verify frozen after DataStore internally created connection
        with pytest.raises(RuntimeError, match="cannot be changed after a connection"):
            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

    def test_override_setting_via_datastore_file_source(self, tmp_path):
        """Override setting to 0, verify per-DataStore file connection picks it up."""
        import pandas as pd
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({'a': [10, 20], 'b': [30, 40]}).to_csv(csv_path, index=False)

        _run_chdb_in_subprocess(f"""
            import pandas as pd
            from datastore import DataStore
            from datastore.config import set_chdb_setting

            set_chdb_setting('memory_worker_correct_memory_tracker', 0)

            ds = DataStore("file", path="{csv_path}")
            pd_result = pd.DataFrame({{'a': [10, 20], 'b': [30, 40]}})

            assert list(ds['a']) == list(pd_result['a'])
            assert list(ds['b']) == list(pd_result['b'])

            r = ds._executor.connection._conn.query(
                "SELECT value FROM system.server_settings "
                "WHERE name = 'memory_worker_correct_memory_tracker'",
                'CSV',
            )
            assert '"0"' in str(r), f"Expected '0' but got: {{r}}"
        """)


class TestChdbSettingsConnectionString:
    """Verify connection string is built correctly."""

    def test_connection_string_default(self):
        conn = Connection(':memory:')
        conn_str = conn._build_connection_string()
        assert ':memory:?' in conn_str
        assert 'memory_worker_correct_memory_tracker=1' in conn_str
        assert 'max_server_memory_usage=0' in conn_str
        assert 'max_server_memory_usage_to_ram_ratio=0' in conn_str

    def test_connection_string_override(self):
        set_chdb_setting('memory_worker_correct_memory_tracker', 0)
        conn = Connection(':memory:')
        conn_str = conn._build_connection_string()
        assert 'memory_worker_correct_memory_tracker=0' in conn_str

    def test_connection_string_extra_setting(self):
        set_chdb_setting('some_custom_setting', 42)
        conn = Connection(':memory:')
        conn_str = conn._build_connection_string()
        assert 'some_custom_setting=42' in conn_str
        assert 'memory_worker_correct_memory_tracker=1' in conn_str

    def test_connection_string_with_file_database(self):
        conn = Connection('test.db')
        conn_str = conn._build_connection_string()
        assert 'test.db?' in conn_str
        assert 'memory_worker_correct_memory_tracker=1' in conn_str
