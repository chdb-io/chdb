import subprocess
import sys
import textwrap
import types
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def test_import_and_basic_query_without_adbc_driver_manager():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "adbc_driver_manager" or fullname.startswith("adbc_driver_manager."):
                    raise ImportError(f"blocked {fullname}")
                return None


        sys.meta_path.insert(0, Blocker())

        import chdb

        result = chdb.query("SELECT 1", "CSV")
        assert str(result).strip() == "1"
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_pyproject_keeps_base_dependencies_and_defines_adbc_extra():
    project_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())

    assert pyproject["project"]["dependencies"] == [
        "chdb-core>=26.7.0",
        "pandas>=2.1.0",
        "pyarrow>=13.0.0",
    ]

    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert optional_dependencies["adbc"] == [
        "chdb-core>=26.7.0",
        "adbc-driver-manager>=1.11.0; python_version >= '3.10'",
    ]


def test_adbc_adapter_uses_chdb_core_library(monkeypatch):
    calls = {}

    def fake_connect(**kwargs):
        calls["kwargs"] = kwargs
        return "connection"

    fake_chdb = types.SimpleNamespace(
        _chdb=types.SimpleNamespace(__file__="/tmp/chdb-core/_chdb.abi3.so")
    )
    fake_manager_dbapi = types.SimpleNamespace(connect=fake_connect)
    fake_manager = types.SimpleNamespace(dbapi=fake_manager_dbapi)

    monkeypatch.setitem(sys.modules, "chdb", fake_chdb)
    monkeypatch.setitem(sys.modules, "adbc_driver_manager", fake_manager)
    monkeypatch.setitem(sys.modules, "adbc_driver_manager.dbapi", fake_manager_dbapi)

    from adbc_driver_chdb import dbapi

    connection = dbapi.connect("chdb://")

    assert connection == "connection"
    assert calls["kwargs"] == {
        "driver": "/tmp/chdb-core/_chdb.abi3.so",
        "entrypoint": "chdb_adbc_init",
        "autocommit": True,
        "db_kwargs": {"uri": "chdb://"},
    }
