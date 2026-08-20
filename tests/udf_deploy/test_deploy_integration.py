"""Integration tests for chdb.deploy against a real ClickHouse server.

Environment (all three required — tests skip when unset):

  CHDB_TEST_UDF_HTTP         host:port of the server's HTTP interface
  CHDB_TEST_UDF_SCRIPTS_DIR  local path mounted as the server's user_scripts_path
  CHDB_TEST_UDF_CONFIG_DIR   local path the server loads *_function.xml from

Optional: CHDB_TEST_UDF_USER (default "default"), CHDB_TEST_UDF_PASSWORD.

Bring up a suitable server (ch1 of the shared integration pair) with:

  mkdir -p .github/ci/udf-scripts .github/ci/udf-config
  docker compose -f .github/ci/clickhouse-pair.yml up -d --build --wait
  export CHDB_TEST_UDF_HTTP=localhost:8123
  export CHDB_TEST_UDF_USER=remote_user
  export CHDB_TEST_UDF_PASSWORD=test123
  export CHDB_TEST_UDF_DATABASE=chdb_test
  export CHDB_TEST_UDF_SCRIPTS_DIR=$PWD/.github/ci/udf-scripts
  export CHDB_TEST_UDF_CONFIG_DIR=$PWD/.github/ci/udf-config
"""

import os
import sys
from datetime import date

import pytest

import chdb
import chdb.deploy as deploy
import datastore.config

dsconfig = sys.modules["datastore.config"]

_HTTP = os.environ.get("CHDB_TEST_UDF_HTTP", "")
_SCRIPTS_DIR = os.environ.get("CHDB_TEST_UDF_SCRIPTS_DIR", "")
_CONFIG_DIR = os.environ.get("CHDB_TEST_UDF_CONFIG_DIR", "")

pytestmark = pytest.mark.skipif(
    not (_HTTP and _SCRIPTS_DIR and _CONFIG_DIR),
    reason="CHDB_TEST_UDF_* environment not configured (see module docstring)",
)


@pytest.fixture()
def connection():
    dsconfig.clear_connections()
    host, _, port = _HTTP.partition(":")
    conn = dsconfig.register_connection(
        "udf-test",
        host=host,
        port=int(port or "8123"),
        username=os.environ.get("CHDB_TEST_UDF_USER", "default"),
        password=os.environ.get("CHDB_TEST_UDF_PASSWORD", ""),
        database=os.environ.get("CHDB_TEST_UDF_DATABASE", "default"),
        udf_scripts_dir=_SCRIPTS_DIR,
        udf_config_dir=_CONFIG_DIR,
    )
    yield conn
    deploy.cleanup_session()
    dsconfig.clear_connections()


def _select(conn, query: str) -> str:
    return deploy._http_query(conn, query).strip()


class TestSessionScopedDeploy:
    def test_decorator_deploys_and_both_sides_work(self, connection):
        @chdb.func(deploy=True)
        def itest_add_tax(price: float, rate: float) -> float:
            return round(price * (1 + rate), 2)

        info = itest_add_tax.chdb_deployment
        try:
            assert not info.skipped
            # temporary deployments register under the function's own name,
            # so remote SQL can call the name the user actually wrote
            assert info.remote_name == "itest_add_tax"
            # remote execution
            assert _select(connection, f"SELECT {info.remote_name}(100, 0.13)") == "113"
            # local registration is untouched
            assert (
                chdb.query("SELECT itest_add_tax(100, 0.13)", "CSV").bytes().strip()
                == b"113"
            )
            # plain python call still works
            assert itest_add_tax(100, 0.13) == 113.0
        finally:
            chdb.drop_function("itest_add_tax")

    def test_redeploy_same_code_skips(self, connection):
        def itest_double(x: int) -> int:
            return x * 2

        first = deploy.deploy(itest_double)
        second = deploy.deploy(itest_double)
        assert not first.skipped
        assert second.skipped
        assert first.remote_name == second.remote_name
        assert _select(connection, f"SELECT {first.remote_name}(21)") == "42"

    def test_date_argument_parsed_to_python_date(self, connection):
        def itest_year(d: date) -> int:
            return d.year

        info = deploy.deploy(itest_year)
        assert (
            _select(connection, f"SELECT {info.remote_name}(toDate('2024-03-15'))")
            == "2024"
        )

    def test_string_arguments_roundtrip(self, connection):
        def itest_shout(text: str) -> str:
            return text.upper() + "!"

        info = deploy.deploy(itest_shout)
        assert _select(connection, f"SELECT {info.remote_name}('ok')") == "OK!"

    def test_on_error_ignore_yields_null_row(self, connection):
        # no explicit Nullable needed: declarations are auto-wrapped like the
        # local engine's makeNullable, so the NULL row parses as NULL (it
        # used to be silently coerced to 0 by input_format_null_as_default)
        def itest_safediv(a: int, b: int) -> int:
            return a // b

        info = deploy.deploy(itest_safediv, on_error="ignore")
        assert _select(connection, f"SELECT {info.remote_name}(10, 2)") == "5"
        assert _select(connection, f"SELECT {info.remote_name}(1, 0)") == "\\N"

    def test_declared_types_match_local_chdb(self, connection):
        """Parity check: the remote result type equals what local chdb
        declares for the same function (engine-side makeNullable)."""
        def itest_parity(price: float, coupon: str) -> float:
            return price

        info = deploy.deploy(itest_parity)
        remote_type = _select(
            connection, f"SELECT toTypeName({info.remote_name}(1.0, 'x'))"
        )
        assert remote_type == "Nullable(Float64)"
        # NULL flows into an annotation-inferred argument (used to fail with
        # "Cannot convert NULL to a non-nullable type")
        assert _select(connection, f"SELECT {info.remote_name}(100, NULL)") == "\\N"

    def test_optional_annotation_deploys(self, connection):
        # local chdb currently rejects Optional[...] at registration, so this
        # goes through standalone deploy() (no local registration involved)
        from typing import Optional

        def itest_opt(x: Optional[str]) -> Optional[str]:
            return "none" if x is None else x.upper()

        info = deploy.deploy(itest_opt, on_null="pass")
        assert _select(connection, f"SELECT {info.remote_name}('ok')") == "OK"
        assert _select(connection, f"SELECT {info.remote_name}(NULL)") == "none"

    def test_datetime_argument_is_timezone_aware(self, connection):
        def itest_tzprobe(ts) -> str:
            return f"{ts.hour}|{ts.tzinfo}"

        info = deploy.deploy(itest_tzprobe, arg_types=["DateTime('UTC')"])
        out = _select(
            connection,
            f"SELECT {info.remote_name}(toDateTime('2024-03-15 07:30:15', 'UTC'))",
        )
        assert out == "7|UTC"

    def test_on_null_skip_default_returns_null(self, connection):
        def itest_incr(x):
            return x + 1  # would TypeError if ever called with None

        # nullability is implicit (auto-wrap), exactly like local chdb — an
        # explicit Nullable(...) argument declaration would be rejected
        info = deploy.deploy(
            itest_incr,
            arg_types=["Int64"],
            return_type="Nullable(Int64)",
        )
        assert _select(connection, f"SELECT {info.remote_name}(41)") == "42"
        assert _select(connection, f"SELECT {info.remote_name}(NULL)") == "\\N"

    def test_sql_alias_types_deploy(self, connection):
        # aliases resolve like DataTypeFactory: BIGINT -> Int64, TEXT -> String
        def itest_alias(n, s) -> str:
            return f"{type(n).__name__}:{n + 1}|{s.upper()}"

        info = deploy.deploy(
            itest_alias, arg_types=["BIGINT", "TEXT"], return_type="TEXT"
        )
        assert (
            _select(connection, f"SELECT {info.remote_name}(41, 'ok')")
            == "int:42|OK"
        )

    def test_http_error_paths(self, connection):
        # server reachable but the query is invalid -> HTTP error with body
        with pytest.raises(RuntimeError, match="ClickHouse HTTP error"):
            deploy._http_query(connection, "SELECT definitely not sql !!!")
        # unreachable endpoint -> clear connectivity error
        dead = dsconfig.register_connection(
            "dead", host="127.0.0.1", port=1, default=False
        )
        with pytest.raises(RuntimeError, match="Cannot reach ClickHouse"):
            deploy._http_query(dead, "SELECT 1", timeout=2)

    def test_cleanup_session_drops_temp_function(self, connection):
        def itest_negate(x: int) -> int:
            return -x

        info = deploy.deploy(itest_negate)
        assert deploy._function_exists(connection, info.remote_name)
        deploy.cleanup_session()
        assert not deploy._function_exists(connection, info.remote_name)
        for path in info.artifact_paths:
            assert not os.path.exists(path)


class TestPermanentDeploy:
    def test_permanent_uses_own_name_and_survives_cleanup(self, connection):
        @chdb.func(deploy="udf-test", permanent=True)
        def itest_strlen(name: str) -> int:
            return len(name)

        info = itest_strlen.chdb_deployment
        try:
            assert info.remote_name == "itest_strlen"
            assert _select(connection, "SELECT itest_strlen('clickhouse')") == "10"
            deploy.cleanup_session()
            assert deploy._function_exists(connection, "itest_strlen")
            # re-deploy of an existing name is a no-op
            assert deploy.deploy(itest_strlen, "udf-test", permanent=True).skipped
        finally:
            chdb.drop_function("itest_strlen")
            deploy.undeploy("itest_strlen", "udf-test")
        assert not deploy._function_exists(connection, "itest_strlen")

    def test_permanent_redeploy_with_changed_code_updates(self, connection):
        def itest_versioned(x: int) -> int:
            return x + 1

        deploy.deploy(itest_versioned, permanent=True, name="itest_versioned")
        try:
            assert _select(connection, "SELECT itest_versioned(10)") == "11"

            def itest_versioned(x: int) -> int:  # noqa: F811
                return x + 100

            updated = deploy.deploy(
                itest_versioned, permanent=True, name="itest_versioned"
            )
            assert not updated.skipped
            assert _select(connection, "SELECT itest_versioned(10)") == "110"
        finally:
            deploy.undeploy("itest_versioned")

    def test_builtin_name_collision_rejected(self, connection):
        def itest_mylength(x: str) -> int:
            return len(x)

        # `length` is a ClickHouse built-in listed in system.functions
        with pytest.raises(ValueError, match="already exists"):
            deploy.deploy(itest_mylength, permanent=True, name="length")

    def test_deploy_works_on_undecorated_function(self, connection):
        def itest_plain(x: float) -> float:
            return x / 2

        info = deploy.deploy(itest_plain, permanent=True, name="itest_half")
        try:
            assert info.remote_name == "itest_half"
            assert _select(connection, "SELECT itest_half(21)") == "10.5"
        finally:
            deploy.undeploy("itest_half")
