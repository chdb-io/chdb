"""Unit tests for chdb.deploy — no ClickHouse server required.

Covers the pieces that translate a Python function into ClickHouse
executable-UDF artifacts (type mapping, source extraction, script and XML
generation, naming), the datastore.config connection registry, and the
`from chdb import func` entry-point wiring.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

import chdb
import chdb.deploy as deploy

# `from datastore import config` would yield the DataStoreConfig facade
# instance (it shadows the module name in the package namespace); resolve the
# module itself through sys.modules after importing it.
import datastore.config

dsconfig = sys.modules["datastore.config"]


@pytest.fixture(autouse=True)
def clean_connection_registry():
    dsconfig.clear_connections()
    yield
    dsconfig.clear_connections()


# ---------------------------------------------------------------------------
# Connection registry (datastore/config.py)
# ---------------------------------------------------------------------------


class TestConnectionRegistry:
    def test_register_and_get(self):
        conn = dsconfig.register_connection(
            "demo", host="localhost", port=8125, password="secret"
        )
        assert dsconfig.get_connection("demo") is conn
        assert conn.http_url == "http://localhost:8125"
        assert not conn.supports_udf_deploy()

    def test_first_registered_becomes_default(self):
        dsconfig.register_connection("a", host="host-a")
        dsconfig.register_connection("b", host="host-b")
        assert dsconfig.get_connection().name == "a"

    def test_default_flag_overrides(self):
        dsconfig.register_connection("a", host="host-a")
        dsconfig.register_connection("b", host="host-b", default=True)
        assert dsconfig.get_connection().name == "b"
        dsconfig.set_default_connection("a")
        assert dsconfig.get_connection().name == "a"

    def test_password_hidden_from_repr(self):
        conn = dsconfig.register_connection(
            "secret", host="localhost", password="hunter2"
        )
        assert "hunter2" not in repr(conn)
        assert conn.password == "hunter2"

    def test_ipv6_host_bracketed_in_url(self):
        conn = dsconfig.register_connection("v6", host="::1", port=8123)
        assert conn.http_url == "http://[::1]:8123"

    def test_secure_url_and_deploy_channel(self):
        conn = dsconfig.register_connection(
            "cloud",
            host="example.clickhouse.cloud",
            port=8443,
            secure=True,
            udf_scripts_dir="/tmp/s",
            udf_config_dir="/tmp/c",
        )
        assert conn.http_url == "https://example.clickhouse.cloud:8443"
        assert conn.supports_udf_deploy()

    def test_unregister_and_errors(self):
        with pytest.raises(KeyError, match="No default"):
            dsconfig.get_connection()
        dsconfig.register_connection("a", host="host-a")
        with pytest.raises(KeyError, match="Unknown ClickHouse connection"):
            dsconfig.get_connection("nope")
        assert dsconfig.unregister_connection("a") is True
        assert dsconfig.unregister_connection("a") is False
        with pytest.raises(KeyError, match="No default"):
            dsconfig.get_connection()
        assert dsconfig.list_connections() == []

    def test_rejects_empty_name_or_host(self):
        with pytest.raises(ValueError):
            dsconfig.register_connection("", host="h")
        with pytest.raises(ValueError):
            dsconfig.register_connection("x", host="")


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------


class TestTypeResolution:
    def test_from_annotations(self):
        def fn(a: int, b: float, c: str, d: bool) -> float:
            return 0.0

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [
            ("a", "Int64"),
            ("b", "Float64"),
            ("c", "String"),
            ("d", "Bool"),
        ]
        assert ret == "Float64"

    def test_unannotated_falls_back_to_string(self):
        def fn(a, b):
            return a

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [("a", "String"), ("b", "String")]
        assert ret == "String"

    def test_explicit_strings_win_over_annotations(self):
        def fn(a: int) -> int:
            return a

        args, ret = deploy._resolve_types(fn, ["UInt32"], "UInt64")
        assert args == [("a", "UInt32")]
        assert ret == "UInt64"

    def test_sqltypes_objects(self):
        from chdb.sqltypes import INT64, STRING

        def fn(a, b):
            return b

        args, ret = deploy._resolve_types(fn, [INT64, STRING], STRING)
        assert args == [("a", "Int64"), ("b", "String")]
        assert ret == "String"

    def test_arg_count_mismatch_raises(self):
        def fn(a, b):
            return a

        with pytest.raises(ValueError, match="parameters"):
            deploy._resolve_types(fn, ["Int64"], None)

    def test_quoted_annotations_resolved_to_real_types(self):
        # equivalent to `from __future__ import annotations` in user code:
        # the raw annotation is the string "int", not the int type
        def fn(a: "int", b: "float") -> "str":
            return ""

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [("a", "Int64"), ("b", "Float64")]
        assert ret == "String"

    def test_keyword_only_parameter_rejected(self):
        def fn(a: int, *, flag: bool) -> int:
            return a

        with pytest.raises(ValueError, match="keyword-only"):
            deploy._resolve_types(fn, None, None)

    def test_var_args_rejected(self):
        def fn(*values: int) -> int:
            return sum(values)

        with pytest.raises(ValueError, match="keyword-only"):
            deploy._resolve_types(fn, None, None)

    def test_decimal_and_temporal_converters(self):
        assert deploy._converter_name("Decimal(38, 10)") == "_parse_decimal"
        assert deploy._converter_name("Decimal128(10)") == "_parse_decimal"
        assert deploy._converter_name("Date") == "_parse_date"
        assert deploy._converter_name("Date32") == "_parse_date"
        assert deploy._converter_name("DateTime64(3)") == "_parse_datetime"
        assert deploy._converter_name("Nullable(DateTime)") == "_parse_datetime"


# ---------------------------------------------------------------------------
# Source extraction, script generation, XML generation
# ---------------------------------------------------------------------------


def _run_script(script_body: str, stdin: str) -> str:
    code = script_body.split("#!/usr/bin/env python3\n", 1)[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


class TestArtifactGeneration:
    def test_source_strips_decorator_lines(self):
        source = deploy._function_source(_decorated_sample)
        assert source.startswith("def _decorated_sample")
        assert "pytest" not in source

    def test_numeric_script_roundtrip(self):
        def add_tax(price: float, rate: float) -> float:
            return round(price * (1 + rate), 2)

        script = deploy._generate_script(
            "add_tax", deploy._function_source(add_tax), ["Float64", "Float64"]
        )
        assert _run_script(script, "100\t0.13\n10\t0\n") == "113.0\n10.0\n"

    def test_string_escapes_and_bool_output(self):
        def has_tab(text: str) -> bool:
            return "\t" in text

        script = deploy._generate_script(
            "has_tab", deploy._function_source(has_tab), ["String"]
        )
        assert _run_script(script, "a\\tb\nplain\n") == "true\nfalse\n"

    def test_null_passthrough(self):
        def maybe(value: str) -> str:
            return "yes" if value is None else value

        script = deploy._generate_script(
            "maybe", deploy._function_source(maybe), ["Nullable(String)"]
        )
        assert _run_script(script, "\\N\nhello\n") == "yes\nhello\n"

    def test_none_result_becomes_null(self):
        def swallow(value: str):
            return None

        script = deploy._generate_script(
            "swallow", deploy._function_source(swallow), ["String"]
        )
        assert _run_script(script, "x\n") == "\\N\n"

    def test_decimal_precision_preserved(self):
        def echo_decimal(value) -> str:
            return str(value)

        script = deploy._generate_script(
            "echo_decimal",
            deploy._function_source(echo_decimal),
            ["Decimal(38, 10)"],
        )
        big = "1234567890123456789.0123456789"
        assert _run_script(script, big + "\n") == big + "\n"

    def test_date_and_datetime_parsed(self):
        def year_of(d) -> int:
            return d.year

        script = deploy._generate_script(
            "year_of", deploy._function_source(year_of), ["Date32"]
        )
        assert _run_script(script, "2024-03-15\n") == "2024\n"

        def hour_of(ts) -> int:
            return ts.hour

        script = deploy._generate_script(
            "hour_of", deploy._function_source(hour_of), ["DateTime64(3)"]
        )
        assert _run_script(script, "2024-03-15 07:30:15.123\n") == "7\n"

    def test_xml_structure(self):
        xml = deploy._generate_config_xml(
            "chdb_nb_abc123_deadbeef",
            "chdb_nb_abc123_deadbeef.py",
            [("price", "Float64"), ("city", "String")],
            "Float64",
        )
        function = ET.fromstring(xml).find("function")
        assert function.findtext("type") == "executable"
        assert function.findtext("execute_direct") == "1"
        assert function.findtext("name") == "chdb_nb_abc123_deadbeef"
        assert function.findtext("return_type") == "Float64"
        assert function.findtext("format") == "TabSeparated"
        assert function.findtext("command") == "chdb_nb_abc123_deadbeef.py"
        arguments = function.findall("argument")
        assert [(a.findtext("name"), a.findtext("type")) for a in arguments] == [
            ("price", "Float64"),
            ("city", "String"),
        ]


@pytest.mark.parametrize("marker", [None])
def _decorated_sample(marker):  # pragma: no cover - source fixture only
    return marker


# ---------------------------------------------------------------------------
# Naming and decorator argument validation
# ---------------------------------------------------------------------------


class TestNamingAndValidation:
    def test_session_id_shape(self):
        assert len(deploy.session_id()) == 6
        assert deploy.session_id() == deploy._SESSION_ID

    def test_permanent_requires_deploy(self):
        with pytest.raises(ValueError, match="permanent=True requires"):
            deploy.func(permanent=True)

    def test_deploy_without_connection_raises_keyerror(self):
        def orphan(x: int) -> int:
            return x

        with pytest.raises(KeyError, match="No default ClickHouse connection"):
            deploy.deploy(orphan)

    def test_missing_channel_message_mentions_cloud(self, monkeypatch):
        dsconfig.register_connection("q", host="localhost", port=1)
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: False)

        def fn(x: int) -> int:
            return x

        with pytest.raises(RuntimeError, match="ClickHouse Cloud"):
            deploy.deploy(fn, "q")

    def test_session_name_stable_for_same_code(self, monkeypatch):
        dsconfig.register_connection("q", host="localhost", port=1)
        seen = []

        def fake_exists(conn, name):
            seen.append(name)
            return True  # short-circuit: name resolution only

        monkeypatch.setattr(deploy, "_function_exists", fake_exists)

        def fn(x: int) -> int:
            return x + 1

        first = deploy.deploy(fn, "q")
        second = deploy.deploy(fn, "q")
        assert first.remote_name == second.remote_name
        assert first.remote_name.startswith(f"chdb_nb_{deploy.session_id()}_")
        assert first.skipped and second.skipped

    def test_async_function_rejected(self):
        async def fetchy(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="async"):
            deploy.deploy(fetchy)

    def test_undeploy_rejects_non_identifier_names(self):
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir="/tmp/s", udf_config_dir="/tmp/c",
        )
        with pytest.raises(ValueError, match="Invalid UDF name"):
            deploy.undeploy("../etc/passwd", "q")
        with pytest.raises(ValueError, match="Invalid UDF name"):
            deploy.undeploy("/absolute", "q")

    def test_failed_deploy_leaves_no_artifacts(self, tmp_path, monkeypatch):
        scripts = tmp_path / "scripts"
        configs = tmp_path / "configs"
        scripts.mkdir()
        configs.mkdir()
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir=str(scripts), udf_config_dir=str(configs),
        )
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: False)

        def boom(connection):
            raise RuntimeError("reload failed")

        monkeypatch.setattr(deploy, "_reload_functions", boom)

        def fn(x: int) -> int:
            return x

        with pytest.raises(RuntimeError, match="reload failed"):
            deploy.deploy(fn, "q")
        assert list(scripts.iterdir()) == []
        assert list(configs.iterdir()) == []

    def test_failed_deploy_restores_preexisting_artifacts(
        self, tmp_path, monkeypatch
    ):
        scripts = tmp_path / "scripts"
        configs = tmp_path / "configs"
        scripts.mkdir()
        configs.mkdir()
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir=str(scripts), udf_config_dir=str(configs),
        )
        old_script = scripts / "fixed_name.py"
        old_config = configs / "fixed_name_function.xml"
        old_script.write_text("# original script")
        old_script.chmod(0o700)  # deliberately private permissions
        old_config.write_text("<functions>original</functions>")

        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: False)
        reloads = []

        def boom(connection):
            reloads.append(connection.name)
            raise RuntimeError("reload failed")

        monkeypatch.setattr(deploy, "_reload_functions", boom)

        def fn(x: int) -> int:
            return x

        with pytest.raises(RuntimeError, match="reload failed"):
            deploy.deploy(fn, "q", permanent=True, name="fixed_name")
        # pre-existing artifacts restored, not deleted
        assert old_script.read_text() == "# original script"
        assert old_config.read_text() == "<functions>original</functions>"
        # ... with their original permission bits, not a hardcoded 0o755
        assert (old_script.stat().st_mode & 0o7777) == 0o700
        # no temp files left behind
        assert sorted(p.name for p in scripts.iterdir()) == ["fixed_name.py"]
        assert sorted(p.name for p in configs.iterdir()) == [
            "fixed_name_function.xml"
        ]
        # cleanup attempted a second reload so the server drops the stale UDF
        assert len(reloads) == 2

    def test_permanent_uses_function_name(self, monkeypatch):
        dsconfig.register_connection("q", host="localhost", port=1)
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: True)

        def my_scorer(x: float) -> float:
            return x

        deployment = deploy.deploy(my_scorer, "q", permanent=True)
        assert deployment.remote_name == "my_scorer"
        assert deployment.permanent


# ---------------------------------------------------------------------------
# Entry-point wiring
# ---------------------------------------------------------------------------


class TestEntryPoint:
    def test_chdb_func_is_extended(self):
        import inspect

        parameters = inspect.signature(chdb.func).parameters
        assert "deploy" in parameters
        assert "permanent" in parameters

    def test_self_repair_rebinds_after_clobber(self):
        original = chdb.func
        try:
            chdb.func = None
            deploy._install()
            assert chdb.func is deploy.func
        finally:
            chdb.func = original

    def test_install_after_udf_import_does_not_recurse(self):
        """Regression: _install() rebinds chdb.udf.func to deploy's func; the
        decorator's local-registration delegation must keep pointing at the
        genuine chdb.udf decorator, not recurse into itself."""
        import chdb.udf  # ensure the module is loaded so _install patches it

        deploy._install()
        assert chdb.udf.func is deploy.func

        @chdb.func(return_type="Int64")
        def _udf_deploy_recursion_probe(a: int) -> int:
            return a + 1

        try:
            result = chdb.query(
                "SELECT _udf_deploy_recursion_probe(41)", "CSV"
            ).bytes()
            assert b"42" in result
        finally:
            chdb.drop_function("_udf_deploy_recursion_probe")

    def test_local_only_decorator_still_registers_locally(self):
        @chdb.func(return_type="Int64")
        def _udf_deploy_unit_add(a: int, b: int) -> int:
            return a + b

        try:
            result = chdb.query(
                "SELECT _udf_deploy_unit_add(20, 22)", "CSV"
            ).bytes()
            assert b"42" in result
        finally:
            chdb.drop_function("_udf_deploy_unit_add")
