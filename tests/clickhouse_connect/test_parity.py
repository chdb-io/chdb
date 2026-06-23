"""
Output-parity matrix: chDB backend vs a real ClickHouse server (HTTP backend).

This is the chDB analogue of the chdb-node Layer-2 parity suite (PR #49): the byte/value
compatibility claim is only credible if the *same* clickhouse-connect call returns the
*same* result whether it runs against an embedded chDB engine or a real ClickHouse server
over HTTP. Each case below builds one client per backend and asserts the results match.

The HTTP side is configured from the environment and the module is skipped when no server
is reachable (so the suite still runs chDB-only in environments without a server, and gates
parity in CI where a server is provisioned):

    CLICKHOUSE_CONNECT_TEST_HOST       (default: localhost)
    CLICKHOUSE_CONNECT_TEST_PORT       (default: 8123)
    CLICKHOUSE_CONNECT_TEST_USER       (default: default)
    CLICKHOUSE_CONNECT_TEST_PASSWORD   (default: empty)

chDB ships the same ClickHouse engine version, so the matrix is version-aligned by
construction. Known, intentional divergences are marked ``xfail`` with a reason and still
run (so a divergence that silently disappears is flagged).
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

chdb = pytest.importorskip("chdb")
clickhouse_connect = pytest.importorskip("clickhouse_connect")

import clickhouse_connect.driver.registry as _cc_registry  # noqa: E402

if "chdb" not in _cc_registry.available_backend_names():
    pytest.skip("chdb backend is not registered with clickhouse-connect", allow_module_level=True)


def _http_config():
    return {
        "host": os.getenv("CLICKHOUSE_CONNECT_TEST_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_CONNECT_TEST_PORT", "8123")),
        "username": os.getenv("CLICKHOUSE_CONNECT_TEST_USER", "default"),
        "password": os.getenv("CLICKHOUSE_CONNECT_TEST_PASSWORD", ""),
    }


def _server_reachable() -> bool:
    try:
        c = clickhouse_connect.get_client(**_http_config())
        try:
            return c.ping()
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return False


if not _server_reachable():
    pytest.skip(
        "No ClickHouse server reachable for parity (set CLICKHOUSE_CONNECT_TEST_HOST/PORT)",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def http_client():
    c = clickhouse_connect.get_client(**_http_config())
    yield c
    c.close()


@pytest.fixture
def chdb_client():
    c = clickhouse_connect.get_client(backend="chdb")
    yield c
    c.close()


# (id, sql) — each runs on both backends; result_rows must match exactly.
_QUERY_CASES = [
    ("primitives", "SELECT toInt32(13) AS i, toString('user_1') AS s, toFloat64(3.14) AS f"),
    ("nullable", "SELECT CAST(NULL AS Nullable(Int64)) AS n, CAST(7 AS Nullable(Int64)) AS v"),
    ("low_cardinality", "SELECT CAST('user_2' AS LowCardinality(String)) AS lc"),
    ("dates", "SELECT toDate('2026-05-19') AS d, toDateTime('2026-05-19 10:30:00', 'UTC') AS dt"),
    ("decimal", "SELECT toDecimal64(123.456, 3) AS dec"),
    ("array", "SELECT [1, 2, 3]::Array(UInt32) AS arr"),
    ("map", "SELECT map('user_1', 13, 'user_2', 79) AS m"),
    ("tuple", "SELECT tuple(1, 'a', 3.14) AS t"),
    ("enum", "SELECT CAST('a' AS Enum8('a' = 1, 'b' = 2)) AS e"),
    ("uuid", "SELECT toUUID('550e8400-e29b-41d4-a716-446655440000') AS u"),
    ("ipv4_ipv6", "SELECT toIPv4('127.0.0.1') AS v4, toIPv6('::1') AS v6"),
    ("fixed_string", "SELECT toFixedString('xyz', 4) AS fs"),
    ("multi_row", "SELECT number FROM numbers(5)"),
    ("aggregate", "SELECT count(), sum(number), avg(number) FROM numbers(100)"),
    ("group_by", "SELECT number % 3 AS g, count() AS c FROM numbers(30) GROUP BY g ORDER BY g"),
    ("empty", "SELECT 1 WHERE 0"),
    ("datetime64", "SELECT toDateTime64('2026-05-19 10:30:00.123456', 6, 'UTC') AS dt"),
]


@pytest.mark.parametrize("case_id,sql", _QUERY_CASES, ids=[c[0] for c in _QUERY_CASES])
def test_query_parity(http_client, chdb_client, case_id, sql):
    http_rows = http_client.query(sql).result_rows
    chdb_rows = chdb_client.query(sql).result_rows
    assert chdb_rows == http_rows, f"{case_id}: chdb={chdb_rows!r} http={http_rows!r}"


@pytest.mark.parametrize("case_id,sql", _QUERY_CASES, ids=[c[0] for c in _QUERY_CASES])
def test_column_names_parity(http_client, chdb_client, case_id, sql):
    assert chdb_client.query(sql).column_names == http_client.query(sql).column_names


def test_parameter_binding_parity(http_client, chdb_client):
    sql = "SELECT {x:Int32} AS x, {name:String} AS name"
    params = {"x": 13, "name": "user_1"}
    assert chdb_client.query(sql, parameters=params).result_rows == http_client.query(sql, parameters=params).result_rows


def test_decimal_value_parity(http_client, chdb_client):
    sql = "SELECT toDecimal64(123.456, 3) AS dec"
    (chdb_dec,) = chdb_client.query(sql).result_rows[0]
    (http_dec,) = http_client.query(sql).result_rows[0]
    assert chdb_dec == http_dec == Decimal("123.456")


def _make_table(client, name):
    client.command(f"DROP TABLE IF EXISTS {name}")
    client.command(f"CREATE TABLE {name} (id UInt32, name String, v Float64) ENGINE = MergeTree ORDER BY id")


def test_insert_round_trip_parity(http_client, chdb_client):
    rows = [[13, "user_1", 1.5], [79, "user_2", 2.5], [103, "user_3", 3.5]]
    out = {}
    for label, client in (("http", http_client), ("chdb", chdb_client)):
        table = f"parity_insert_{label}"
        _make_table(client, table)
        client.insert(table, rows, column_names=["id", "name", "v"])
        out[label] = client.query(f"SELECT id, name, v FROM {table} ORDER BY id").result_rows
        client.command(f"DROP TABLE IF EXISTS {table}")
    assert out["chdb"] == out["http"] == [tuple(r) for r in rows]


def test_query_df_parity(http_client, chdb_client):
    pd = pytest.importorskip("pandas")
    sql = "SELECT number AS n, toString(number) AS s FROM numbers(10)"
    http_df = http_client.query_df(sql)
    chdb_df = chdb_client.query_df(sql)
    # Compare values column-by-column; dtype representation may differ (the documented
    # Arrow-vs-Native divergence the design proposal calls out), values must not.
    assert list(chdb_df["n"]) == list(http_df["n"])
    assert list(chdb_df["s"]) == list(http_df["s"])
    pd.testing.assert_series_equal(
        chdb_df["n"].astype("int64"), http_df["n"].astype("int64"), check_names=False
    )
