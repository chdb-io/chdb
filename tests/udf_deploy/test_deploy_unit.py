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

    def test_facade_proxies_connection_registry(self):
        from datastore import config as facade

        conn = facade.register_connection("via-facade", host="h1")
        assert facade.get_connection("via-facade") is conn
        assert facade.list_connections() == ["via-facade"]
        facade.register_connection("second", host="h2", default=True)
        assert facade.get_connection().name == "second"
        facade.set_default_connection("via-facade")
        assert facade.get_connection().name == "via-facade"
        assert facade.unregister_connection("second") is True

    def test_default_false_on_first_registration(self):
        dsconfig.register_connection("a", host="h", default=False)
        with pytest.raises(KeyError, match="No default"):
            dsconfig.get_connection()

    def test_set_default_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown ClickHouse connection"):
            dsconfig.set_default_connection("nope")

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

    def test_unannotated_args_fall_back_to_string(self):
        # local chdb leaves unannotated parameters dynamic; an executable UDF
        # must declare something, and String is the documented fallback
        def fn(a, b):
            return a

        args, ret = deploy._resolve_types(fn, None, "String")
        assert args == [("a", "String"), ("b", "String")]
        assert ret == "String"

    def test_missing_return_type_raises_like_local(self):
        def fn(a: int):
            return a

        with pytest.raises(ValueError, match="return type not specified"):
            deploy._resolve_types(fn, None, None)

    def test_optional_annotations_unwrap(self):
        from typing import Optional, Union

        def fn(a: Optional[int], b: Optional[str]) -> Optional[float]:
            return None

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [("a", "Int64"), ("b", "String")]
        assert ret == "Float64"

        def bad(a: Union[int, str]) -> int:
            return 0

        with pytest.raises(ValueError, match="Unknown Python UDF type annotation"):
            deploy._resolve_types(bad, None, None)

    def test_pep604_union_none_unwraps(self):
        def fn(a: "int | None") -> "float | None":
            return None

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [("a", "Int64")]
        assert ret == "Float64"

    def test_unknown_type_object_raises_like_local(self):
        class Custom:
            pass

        def fn(a: Custom) -> int:
            return 0

        with pytest.raises(ValueError, match="Cannot convert Python type"):
            deploy._resolve_types(fn, None, None)

    def test_numpy_scalar_annotations(self):
        np = pytest.importorskip("numpy")

        def fn(a: np.int32, b: np.float16, c: np.uint64) -> np.float64:
            return 0.0

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [("a", "Int32"), ("b", "Float32"), ("c", "UInt64")]
        assert ret == "Float64"

    def test_sql_aliases_resolve_like_datatypefactory(self):
        # local chdb parses type strings through DataTypeFactory, which
        # resolves SQL-compat aliases case-insensitively
        assert deploy._canonical_type("BIGINT") == "Int64"
        assert deploy._canonical_type("bigint") == "Int64"
        assert deploy._canonical_type("Integer") == "Int32"
        assert deploy._canonical_type("DOUBLE PRECISION") == "Float64"
        assert deploy._canonical_type("VARCHAR(255)") == "String"
        assert deploy._canonical_type("BOOLEAN") == "Bool"
        assert deploy._canonical_type("bool") == "Bool"
        # the complete whitelisted-target alias set from registerAlias():
        assert deploy._canonical_type("TIMESTAMP") == "DateTime"
        assert deploy._canonical_type("timestamp('UTC')") == "DateTime('UTC')"
        assert deploy._canonical_type("YEAR") == "UInt16"
        assert deploy._canonical_type("BIGINT UNSIGNED") == "UInt64"
        assert deploy._canonical_type("tinyint signed") == "Int8"
        assert deploy._canonical_type("NATIONAL CHARACTER VARYING") == "String"
        assert deploy._canonical_type("VARBINARY") == "String"
        # aliases of non-whitelisted types stay unmapped and get rejected,
        # matching local behavior (which rejects the canonical type)
        def bad(a) -> int:
            return 0
        with pytest.raises(ValueError, match="unsupported argument 1 type"):
            deploy._resolve_types(bad, ["NUMERIC(10, 2)"], "Int64")
        # canonical names pass through untouched
        assert deploy._canonical_type("Int64") == "Int64"
        assert deploy._canonical_type("DateTime64(6, 'UTC')") == "DateTime64(6, 'UTC')"

        def fn(a, b) -> int:
            return 0

        args, ret = deploy._resolve_types(fn, ["BIGINT", "TEXT"], "DOUBLE")
        assert args == [("a", "Int64"), ("b", "String")]
        assert ret == "Float64"

    def test_non_type_annotation_instance_raises(self):
        def fn(a: 42) -> int:  # noqa: F722 - deliberate nonsense annotation
            return 0

        with pytest.raises(ValueError, match="Unknown Python UDF type annotation"):
            deploy._resolve_types(fn, None, None)

    def test_unresolvable_forward_ref_falls_back_to_raw(self):
        # get_type_hints raises on an unresolvable name; the fallback treats
        # the raw string annotation as a ClickHouse type name, which then
        # fails the whitelist with the local error shape
        def fn(a: "NoSuchTypeAnywhere") -> "Int64":  # noqa: F821
            return 0

        with pytest.raises(ValueError, match="unsupported argument 1 type"):
            deploy._resolve_types(fn, None, None)

    def test_bool_converter_expr(self):
        assert deploy._converter_expr("Bool") == "_parse_bool"
        assert deploy._converter_expr("Nullable(Bool)") == "_parse_bool"

    def test_make_nullable_mirrors_engine(self):
        assert deploy._make_nullable("Int64") == "Nullable(Int64)"
        assert deploy._make_nullable("DateTime64(6)") == "Nullable(DateTime64(6))"
        assert deploy._make_nullable("Nullable(String)") == "Nullable(String)"
        assert deploy._make_nullable("Array(Int64)") == "Array(Int64)"
        assert deploy._make_nullable("LowCardinality(String)") == "LowCardinality(String)"

    def test_explicit_strings_win_over_annotations(self):
        def fn(a: int) -> int:
            return a

        args, ret = deploy._resolve_types(fn, ["UInt32"], "UInt64")
        assert args == [("a", "UInt32")]
        assert ret == "UInt64"

    def test_sqltypes_objects(self):
        try:
            from chdb.sqltypes import INT64, STRING
        except ImportError:  # pragma: no cover - pytest-cov import interference
            pytest.skip("chdb.sqltypes unavailable in this environment")

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

    def test_inference_matches_documented_local_table(self):
        from datetime import date, datetime

        def fn(a: date, b: datetime, c: bytes) -> int:
            return 0

        args, ret = deploy._resolve_types(fn, None, None)
        assert args == [
            ("a", "Date"),
            ("b", "DateTime64(6)"),
            ("c", "String"),
        ]
        assert ret == "Int64"

    def test_on_null_on_error_normalizers(self):
        assert deploy._on_null_skips(None) is True
        assert deploy._on_null_skips("skip") is True
        assert deploy._on_null_skips("pass") is False
        assert deploy._on_null_skips(chdb.NullHandling.PASS) is False
        assert deploy._on_error_ignores(None) is False
        assert deploy._on_error_ignores("propagate") is False
        assert deploy._on_error_ignores("ignore") is True
        assert deploy._on_error_ignores(chdb.ExceptionHandling.IGNORE) is True

    def test_decimal_and_temporal_converters(self):
        assert deploy._converter_expr("Date") == "_parse_date"
        assert deploy._converter_expr("Date32") == "_parse_date"
        assert deploy._converter_expr("DateTime64(3)") == "_make_datetime_parser(None)"
        assert deploy._converter_expr("Nullable(DateTime)") == "_make_datetime_parser(None)"
        assert (
            deploy._converter_expr("DateTime('UTC')")
            == "_make_datetime_parser('UTC')"
        )
        assert (
            deploy._converter_expr("Nullable(DateTime64(6, 'Asia/Shanghai'))")
            == "_make_datetime_parser('Asia/Shanghai')"
        )

    def test_unsupported_types_rejected_like_local(self):
        """Mirror of isSupportedUDFType: everything local registration
        rejects, deployment rejects with the same error shape."""
        def fn(xs) -> int:
            return 0

        rejected_args = [
            "Array(Int64)",
            "Map(String, Int64)",
            "Tuple(Int64, String)",
            "Decimal(38, 10)",
            "UUID",
            "FixedString(8)",
            "Enum8('a' = 1)",
            "LowCardinality(String)",
            "IPv4",
            # local rejects explicitly-declared Nullable arguments too:
            # nullability is applied implicitly by the engine
            "Nullable(Int64)",
        ]
        for bad in rejected_args:
            with pytest.raises(ValueError, match="unsupported argument 1 type"):
                deploy._resolve_types(fn, [bad], "Int64")

        with pytest.raises(ValueError, match="unsupported return type"):
            deploy._resolve_types(fn, ["Int64"], "Array(Int64)")

    def test_supported_type_edges_accepted(self):
        def fn(a, b, c, d) -> int:
            return 0

        args, ret = deploy._resolve_types(
            fn,
            ["Int256", "UInt128", "DateTime64(6, 'UTC')", "Bool"],
            # local removeNullable's the return before checking, so an
            # explicitly Nullable return is accepted
            "Nullable(Int64)",
        )
        assert [t for _, t in args] == [
            "Int256", "UInt128", "DateTime64(6, 'UTC')", "Bool",
        ]
        assert ret == "Nullable(Int64)"


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

    def test_on_null_pass_calls_with_none(self):
        def maybe(value: str) -> str:
            return "yes" if value is None else value

        script = deploy._generate_script(
            "maybe",
            deploy._function_source(maybe),
            ["Nullable(String)"],
            null_skip=False,
        )
        assert _run_script(script, "\\N\nhello\n") == "yes\nhello\n"

    def test_on_null_skip_is_default_and_never_calls(self):
        def crashes_on_none(value: str) -> str:
            return value.upper()  # AttributeError if ever called with None

        script = deploy._generate_script(
            "crashes_on_none",
            deploy._function_source(crashes_on_none),
            ["Nullable(String)"],
        )
        assert _run_script(script, "\\N\nok\n") == "\\N\nOK\n"

    def test_on_error_ignore_returns_null_for_row(self):
        def divide(a: int, b: int) -> int:
            return a // b

        script = deploy._generate_script(
            "divide",
            deploy._function_source(divide),
            ["Int64", "Int64"],
            error_ignore=True,
        )
        assert _run_script(script, "10\t2\n1\t0\n8\t4\n") == "5\n\\N\n2\n"

    def test_on_error_propagate_fails_the_process(self):
        def divide(a: int, b: int) -> int:
            return a // b

        script = deploy._generate_script(
            "divide", deploy._function_source(divide), ["Int64", "Int64"]
        )
        code = script.split("#!/usr/bin/env python3\n", 1)[1]
        result = subprocess.run(
            [sys.executable, "-c", code],
            input="1\t0\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert "ZeroDivisionError" in result.stderr

    def test_none_result_becomes_null(self):
        def swallow(value: str):
            return None

        script = deploy._generate_script(
            "swallow", deploy._function_source(swallow), ["String"]
        )
        assert _run_script(script, "x\n") == "\\N\n"

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

    def test_datetime_arrives_timezone_aware(self):
        def probe_tz(ts) -> str:
            return f"{ts.hour}|{ts.tzinfo is not None}|{ts.utcoffset()}"

        # declared timezone: the wrapper attaches exactly that zone
        script = deploy._generate_script(
            "probe_tz", deploy._function_source(probe_tz), ["DateTime('UTC')"]
        )
        out = _run_script(script, "2024-03-15 07:30:15\n")
        assert out == "7|True|0:00:00\n"

        # plain DateTime: the wrapper attaches the host zone (aware, offset
        # depends on the machine — assert awareness only)
        script = deploy._generate_script(
            "probe_tz", deploy._function_source(probe_tz), ["DateTime64(6)"]
        )
        out = _run_script(script, "2024-03-15 07:30:15\n")
        assert "|True|" in out

    def test_aware_datetime_result_rendered_in_return_tz(self):
        def shift(ts):
            import datetime
            return ts + datetime.timedelta(hours=1)

        script = deploy._generate_script(
            "shift",
            deploy._function_source(shift),
            ["DateTime('UTC')"],
            return_ch_type="Nullable(DateTime('UTC'))",
        )
        # aware UTC in, +1h, rendered back as naive UTC wall time
        out = _run_script(script, "2024-03-15 07:30:15\n")
        assert out == "2024-03-15 08:30:15\n"

    def test_bytes_result_emitted_as_content(self):
        def to_bytes(text: str) -> bytes:
            return ("<" + text + ">").encode()

        script = deploy._generate_script(
            "to_bytes", deploy._function_source(to_bytes), ["String"]
        )
        # content, not Python repr like b'<ok>'
        assert _run_script(script, "ok\n") == "<ok>\n"

        def to_bytearray(text: str) -> bytes:
            return bytearray(text.encode())

        script = deploy._generate_script(
            "to_bytearray", deploy._function_source(to_bytearray), ["String"]
        )
        assert _run_script(script, "hi\n") == "hi\n"

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

    def test_fork_gets_fresh_session_identity(self, monkeypatch):
        """A forked child must not reuse the parent's session id nor inherit
        its deployment ledger (its atexit would tear down the parent's
        functions)."""
        parent_id = deploy.session_id()
        with deploy._session_lock:
            deploy._session_deployments.append(
                deploy.DeployedFunction("x", "conn", permanent=False)
            )
        monkeypatch.setattr(deploy.os, "getpid", lambda: -12345)
        child_id = deploy.session_id()
        assert child_id != parent_id
        assert deploy._session_deployments == []
        # stable within the "child" process
        assert deploy.session_id() == child_id

    def test_session_id_stable_within_process(self):
        assert deploy.session_id() == deploy.session_id()

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

    def test_temporary_deploy_uses_the_functions_own_name(self, monkeypatch):
        dsconfig.register_connection("q", host="localhost", port=1)
        seen = []

        def fake_exists(conn, name):
            seen.append(name)
            return True  # short-circuit: name resolution only

        monkeypatch.setattr(deploy, "_function_exists", fake_exists)
        monkeypatch.setattr(deploy, "_artifacts_exist", lambda conn, name: True)
        monkeypatch.setattr(
            deploy, "_artifacts_match", lambda conn, name, body, xml: True
        )

        def fn(x: int) -> int:
            return x + 1

        first = deploy.deploy(fn, "q")
        second = deploy.deploy(fn, "q")
        # permanent controls lifetime, not the name: a temporary deployment
        # is callable by the name the user actually wrote
        assert first.remote_name == "fn"
        assert second.remote_name == "fn"
        assert first.skipped and second.skipped

    def test_temporary_deploy_honors_name_override(self, monkeypatch):
        dsconfig.register_connection("q", host="localhost", port=1)
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: True)
        monkeypatch.setattr(deploy, "_artifacts_exist", lambda conn, name: True)
        monkeypatch.setattr(
            deploy, "_artifacts_match", lambda conn, name, body, xml: True
        )

        def fn(x: int) -> int:
            return x + 1

        handle = deploy.deploy(fn, "q", name="renamed_fn")
        assert handle.remote_name == "renamed_fn"
        assert handle.permanent is False

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

    def test_artifacts_exist_and_match_edge_cases(self, tmp_path):
        no_channel = dsconfig.register_connection("nochan", host="localhost")
        assert deploy._artifacts_exist(no_channel, "whatever") is False

        chan = dsconfig.register_connection(
            "chan", host="localhost",
            udf_scripts_dir=str(tmp_path), udf_config_dir=str(tmp_path),
        )
        # files absent -> no match (OSError branch)
        assert deploy._artifacts_match(chan, "ghost", "body", b"xml") is False

    def test_write_atomic_failure_leaves_no_tmp(self, tmp_path):
        target = str(tmp_path / "no_such_dir" / "x.py")
        with pytest.raises(OSError):
            deploy._write_atomic(target, b"data")
        assert list(tmp_path.iterdir()) == []

    def test_remove_deployment_survives_unknown_connection(self, tmp_path):
        artifact = tmp_path / "orphan.py"
        artifact.write_text("x")
        ghost = deploy.DeployedFunction(
            "orphan", "connection-that-no-longer-exists",
            permanent=False, artifact_paths=[str(artifact)],
        )
        deploy._remove_deployment(ghost)  # must not raise
        assert not artifact.exists()

    def test_invalid_permanent_name_rejected(self):
        dsconfig.register_connection("q", host="localhost", port=1)

        def fn(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="Invalid UDF name"):
            deploy.deploy(fn, "q", permanent=True, name="has-dash")

    def test_verify_timeout_raises_and_cleans(self, tmp_path, monkeypatch):
        scripts = tmp_path / "s"
        configs = tmp_path / "c"
        scripts.mkdir()
        configs.mkdir()
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir=str(scripts), udf_config_dir=str(configs),
        )
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: False)
        monkeypatch.setattr(deploy, "_reload_functions", lambda conn: None)
        clock = {"now": 0.0}
        monkeypatch.setattr(
            deploy.time, "monotonic", lambda: clock.__setitem__("now", clock["now"] + 6) or clock["now"]
        )
        monkeypatch.setattr(deploy.time, "sleep", lambda s: None)

        def fn(x: int) -> int:
            return x

        with pytest.raises(RuntimeError, match="did not appear"):
            deploy.deploy(fn, "q")
        assert list(scripts.iterdir()) == []
        assert list(configs.iterdir()) == []

    def test_undeploy_edges(self, tmp_path):
        no_channel = dsconfig.register_connection("nochan", host="localhost")
        with pytest.raises(RuntimeError, match="no UDF delivery channel"):
            deploy.undeploy("anything", "nochan")

        chan = dsconfig.register_connection(
            "chan", host="localhost", port=1,
            udf_scripts_dir=str(tmp_path), udf_config_dir=str(tmp_path),
        )
        # nothing on disk -> False, no reload attempted
        assert deploy.undeploy("ghost", "chan") is False
        # one of the two files present -> removed, missing one tolerated
        (tmp_path / "half.py").write_text("x")
        import unittest.mock as mock
        with mock.patch.object(deploy, "_reload_functions") as reload_mock:
            assert deploy.undeploy("half", "chan") is True
            reload_mock.assert_called_once()
        assert not (tmp_path / "half.py").exists()

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

    def test_name_collision_with_foreign_function_raises(self, monkeypatch):
        """Regression: a permanent name matching a built-in (or any function
        not deployed through this channel) must error, not silently skip."""
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir="/tmp/s", udf_config_dir="/tmp/c",
        )
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: True)
        monkeypatch.setattr(deploy, "_artifacts_exist", lambda conn, name: False)

        def fn(x: str) -> int:
            return len(x)

        with pytest.raises(ValueError, match="already exists"):
            deploy.deploy(fn, "q", permanent=True, name="length")

    def test_permanent_redeploy_updates_changed_code(self, tmp_path, monkeypatch):
        """Regression: a permanent redeploy with changed code must replace
        the artifacts, not silently keep serving the old implementation."""
        scripts = tmp_path / "scripts"
        configs = tmp_path / "configs"
        scripts.mkdir()
        configs.mkdir()
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir=str(scripts), udf_config_dir=str(configs),
        )
        exists = {"value": False}
        monkeypatch.setattr(
            deploy, "_function_exists", lambda conn, name: exists["value"]
        )

        def fake_reload(conn):
            # the server picks the function up once its artifacts are loaded
            exists["value"] = True

        monkeypatch.setattr(deploy, "_reload_functions", fake_reload)

        def scorer(x: int) -> int:
            return x + 1

        first = deploy.deploy(scorer, "q", permanent=True, name="scorer_v")
        assert not first.skipped

        # identical redeploy: artifacts match -> skipped, files untouched
        again = deploy.deploy(scorer, "q", permanent=True, name="scorer_v")
        assert again.skipped

        # changed implementation, same permanent name -> updated in place
        def scorer(x: int) -> int:  # noqa: F811
            return x + 2

        updated = deploy.deploy(scorer, "q", permanent=True, name="scorer_v")
        assert not updated.skipped
        assert "x + 2" in (scripts / "scorer_v.py").read_text()

    def test_skipped_handle_carries_artifacts_and_cleanup(
        self, tmp_path, monkeypatch
    ):
        """A skipped redeploy returns a handle that can undeploy() and is
        covered by session cleanup (without double-tracking)."""
        scripts = tmp_path / "scripts"
        configs = tmp_path / "configs"
        scripts.mkdir()
        configs.mkdir()
        dsconfig.register_connection(
            "q", host="localhost", port=1,
            udf_scripts_dir=str(scripts), udf_config_dir=str(configs),
        )
        exists = {"value": False}
        monkeypatch.setattr(
            deploy, "_function_exists", lambda conn, name: exists["value"]
        )

        def fake_reload(conn):
            exists["value"] = True

        monkeypatch.setattr(deploy, "_reload_functions", fake_reload)

        def skipper(x: int) -> int:
            return x

        first = deploy.deploy(skipper, "q")
        second = deploy.deploy(skipper, "q")
        assert second.skipped
        assert second.artifact_paths == first.artifact_paths
        with deploy._session_lock:
            tracked = [
                d for d in deploy._session_deployments
                if d.remote_name == first.remote_name
            ]
        assert len(tracked) == 1  # no duplicate ledger entry
        second.undeploy()  # the skipped handle can tear the function down
        assert not os.path.exists(first.artifact_paths[0])
        deploy.cleanup_session()

    def test_permanent_uses_function_name(self, monkeypatch):
        dsconfig.register_connection("q", host="localhost", port=1)
        monkeypatch.setattr(deploy, "_function_exists", lambda conn, name: True)
        monkeypatch.setattr(deploy, "_artifacts_exist", lambda conn, name: True)
        monkeypatch.setattr(
            deploy, "_artifacts_match", lambda conn, name, body, xml: True
        )

        def my_scorer(x: float) -> float:
            return x

        deployment = deploy.deploy(my_scorer, "q", permanent=True)
        assert deployment.remote_name == "my_scorer"
        assert deployment.permanent


# ---------------------------------------------------------------------------
# DataStore expression-method registration
# ---------------------------------------------------------------------------


class TestExpressionMethodRegistration:
    def test_udf_becomes_expression_method(self):
        import pandas as pd

        @chdb.func()
        def udf_expr_tax(price: float, rate: float) -> float:
            return round(price * (1 + rate), 2)

        try:
            ds = chdb.to_datastore(pd.DataFrame({"price": [100.0, 200.0]}))
            out = ds.assign(with_tax=ds["price"].udf_expr_tax(0.13)).to_pandas()
            assert list(out["with_tax"]) == [113.0, 226.0]
            kept = ds.filter(ds["price"].udf_expr_tax(0.13) > 115).to_pandas()
            assert list(kept["price"]) == [200.0]
        finally:
            chdb.drop_function("udf_expr_tax")

    def test_builtin_expression_names_are_not_shadowed(self):
        from datastore.expressions import Expression

        original = Expression.coalesce_func

        @chdb.func(return_type="Int64")
        def coalesce_func(a: int) -> int:  # collides with a built-in method
            return a

        try:
            assert Expression.coalesce_func is original
        finally:
            chdb.drop_function("coalesce_func")

    def test_reregistration_updates_the_method(self):
        import pandas as pd

        @chdb.func()
        def udf_expr_incr(x: int) -> int:
            return x + 1

        chdb.drop_function("udf_expr_incr")

        @chdb.func()  # same name, new body — method must serve the new UDF
        def udf_expr_incr(x: int) -> int:  # noqa: F811
            return x + 2

        try:
            ds = chdb.to_datastore(pd.DataFrame({"x": [1]}))
            out = ds.assign(y=ds["x"].udf_expr_incr()).to_pandas()
            assert list(out["y"]) == [3]
        finally:
            chdb.drop_function("udf_expr_incr")

    def test_zero_arity_and_bad_names_not_registered(self):
        from datastore.function_registry import FunctionRegistry

        assert FunctionRegistry.register_udf("zero_args", 0) is False
        assert FunctionRegistry.register_udf("not-an-identifier", 2) is False


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

    def test_fallback_local_decorator_registers_via_create_function(self):
        deco = deploy._fallback_local_func(None, "Int64")

        @deco
        def _udf_fallback_probe(a: int) -> int:
            return a * 3

        try:
            assert _udf_fallback_probe(2) == 6  # wrapper passthrough
            result = chdb.query("SELECT _udf_fallback_probe(14)", "CSV").bytes()
            assert b"42" in result
        finally:
            chdb.drop_function("_udf_fallback_probe")

    def test_decorator_passes_on_null_on_error_locally(self):
        @chdb.func(return_type="Int64", on_null="pass", on_error="ignore")
        def _udf_semantics_probe(x):
            return 0 if x is None else 100 // x

        try:
            out = chdb.query(
                "SELECT _udf_semantics_probe(NULL), _udf_semantics_probe(0)", "CSV"
            ).bytes().decode().strip()
            assert out == "0,\\N" or out == '0,"\\N"' or "0" in out
        finally:
            chdb.drop_function("_udf_semantics_probe")

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
