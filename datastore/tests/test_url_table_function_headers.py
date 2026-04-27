import chdb
import pytest

from datastore.exceptions import DataStoreError
from datastore.table_functions import UrlTableFunction


def assert_sql_parses(sql: str) -> None:
    """
    Assert that ``sql`` is accepted by ClickHouse's SQL parser.

    Uses ``EXPLAIN AST``, a pure parser-level operation that does not run
    the query, do schema inference, or perform any network IO. ClickHouse
    raises ``RuntimeError`` ("Code: ... Syntax error: ...") for any parser
    failure, which we re-raise as a pytest failure.
    """
    full_sql = f"EXPLAIN AST SELECT * FROM {sql}"
    try:
        chdb.query(full_sql, "CSV")
    except Exception as e:
        pytest.fail(
            f"Generated SQL failed to parse via EXPLAIN AST.\n"
            f"  SQL:   {sql}\n"
            f"  Error: {e}"
        )


class TestUrlTableFunctionHeaders:

    def test_to_sql_no_headers_unchanged(self):
        tf = UrlTableFunction(url="https://example.com/d.parquet", format="Parquet")
        sql = tf.to_sql()
        assert sql == "url('https://example.com/d.parquet', 'Parquet')"
        assert " HEADERS(" not in sql
        assert "headers(" not in sql
        assert_sql_parses(sql)

    def test_to_sql_with_headers_inside_url_call(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["User-Agent: test-agent"],
        )
        sql = tf.to_sql()
        assert "headers('User-Agent'='test-agent')" in sql
        assert " HEADERS(" not in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_pad_structure_auto(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["X-Token: abc"],
        )
        sql = tf.to_sql()
        assert "'Parquet', 'auto', headers('X-Token'='abc')" in sql
        assert " HEADERS(" not in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_pad_format_and_structure_auto(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            headers=["X-Token: abc"],
        )
        sql = tf.to_sql()
        assert sql.count("'auto'") >= 2
        assert "headers('X-Token'='abc')" in sql
        assert " HEADERS(" not in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_dict_input(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers={"Authorization": "Bearer xyz"},
        )
        sql = tf.to_sql()
        assert "headers('Authorization'='Bearer xyz')" in sql
        assert " HEADERS(" not in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_multiple_entries(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["X-A: 1", "X-B: 2"],
        )
        sql = tf.to_sql()
        assert "'X-A'='1'" in sql
        assert "'X-B'='2'" in sql
        assert " HEADERS(" not in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_missing_url_raises(self):
        tf = UrlTableFunction(format="Parquet", headers=["X-A: 1"])
        with pytest.raises(DataStoreError):
            tf.to_sql()

    def test_to_sql_headers_dict_value_with_single_quote_is_escaped(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers={"X-Note": "it's bad"},
        )
        sql = tf.to_sql()
        assert "headers('X-Note'='it''s bad')" in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_list_value_with_single_quote_is_escaped(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["X-Note: it's bad"],
        )
        sql = tf.to_sql()
        assert "headers('X-Note'='it''s bad')" in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_dict_key_with_single_quote_is_escaped(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers={"X-O'Reilly": "v"},
        )
        sql = tf.to_sql()
        assert "headers('X-O''Reilly'='v')" in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_single_string_parsed_as_kv(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers="Authorization: Bearer xyz",
        )
        sql = tf.to_sql()
        assert "headers('Authorization'='Bearer xyz')" in sql
        # Must NOT degenerate into a single string literal.
        assert "headers('Authorization: Bearer xyz')" not in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_single_string_without_colon_raises(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers="malformed",
        )
        with pytest.raises(DataStoreError, match="invalid header"):
            tf.to_sql()

    def test_to_sql_headers_list_missing_colon_raises(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["Authorization Bearer xyz"],
        )
        with pytest.raises(DataStoreError, match="invalid header"):
            tf.to_sql()

    def test_to_sql_headers_list_non_string_element_raises(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=[123],
        )
        with pytest.raises(DataStoreError, match="invalid header"):
            tf.to_sql()

    def test_to_sql_headers_unsupported_type_raises(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=42,
        )
        with pytest.raises(DataStoreError, match="headers must be"):
            tf.to_sql()

    def test_to_sql_headers_value_with_colon(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["X-Forwarded-Host: x.com:8080"],
        )
        sql = tf.to_sql()
        assert "headers('X-Forwarded-Host'='x.com:8080')" in sql
        assert_sql_parses(sql)

    def test_to_sql_headers_value_with_url(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers={"X-Original-URL": "https://x.com:8080/path?a=1"},
        )
        sql = tf.to_sql()
        assert "headers('X-Original-URL'='https://x.com:8080/path?a=1')" in sql
        assert_sql_parses(sql)

    def test_to_sql_structure_without_format_uses_auto_format(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            structure="id UInt32, name String",
        )
        sql = tf.to_sql()
        assert sql == (
            "url('https://example.com/d.parquet', 'auto', "
            "'id UInt32, name String')"
        )
        assert_sql_parses(sql)

    def test_to_sql_structure_and_headers_without_format(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            structure="id UInt32, name String",
            headers=["X-Token: abc"],
        )
        sql = tf.to_sql()
        # structure must be preserved at slot 3; only format gets 'auto'.
        assert (
            "'auto', 'id UInt32, name String', headers('X-Token'='abc')"
            in sql
        )
        # 'auto' should appear only once (for format), NOT overwrite structure.
        assert sql.count("'auto'") == 1
        assert_sql_parses(sql)

    def test_to_sql_format_only_no_padding(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
        )
        sql = tf.to_sql()
        assert sql == "url('https://example.com/d.parquet', 'Parquet')"
        assert "'auto'" not in sql
        assert_sql_parses(sql)

    def test_to_sql_url_only(self):
        tf = UrlTableFunction(url="https://example.com/d.parquet")
        sql = tf.to_sql()
        assert sql == "url('https://example.com/d.parquet')"
        assert_sql_parses(sql)


@pytest.mark.parametrize(
    "kwargs",
    [
        # baseline shapes
        {"url": "https://x.com/d.parquet"},
        {"url": "https://x.com/d.parquet", "format": "Parquet"},
        {"url": "https://x.com/d.parquet", "structure": "id UInt32"},
        {
            "url": "https://x.com/d.parquet",
            "format": "Parquet",
            "structure": "id UInt32, name String",
        },
        # headers via dict
        {
            "url": "https://x.com/d.parquet",
            "format": "Parquet",
            "headers": {"K": "V"},
        },
        {
            "url": "https://x.com/d.parquet",
            "format": "Parquet",
            "headers": {"K1": "V1", "K2": "V2"},
        },
        # headers via list
        {
            "url": "https://x.com/d.parquet",
            "format": "Parquet",
            "headers": ["K: V"],
        },
        {
            "url": "https://x.com/d.parquet",
            "format": "Parquet",
            "headers": ["K1: V1", "K2: V2"],
        },
        # headers via single string
        {
            "url": "https://x.com/d.parquet",
            "format": "Parquet",
            "headers": "K: V",
        },
        # headers without format / structure (slot padding cases)
        {
            "url": "https://x.com/d.parquet",
            "headers": {"K": "V"},
        },
        {
            "url": "https://x.com/d.parquet",
            "structure": "id UInt32",
            "headers": {"K": "V"},
        },
        # values containing characters that previously broke things
        {
            "url": "https://x.com/d.parquet",
            "headers": {"X-Note": "it's bad"},
        },
        {
            "url": "https://x.com/d.parquet",
            "headers": ["X-URL: https://y.com:8080/path?a=1&b=2"],
        },
        {
            "url": "https://x.com/d.parquet",
            "headers": {"X-Auth": "Basic dXNlcjpwYXNz=="},
        },
    ],
    ids=lambda kw: ",".join(sorted(kw.keys() - {"url"})) or "url-only",
)
def test_generated_sql_parses_via_chdb(kwargs):
    sql = UrlTableFunction(**kwargs).to_sql()
    assert_sql_parses(sql)


def test_assert_sql_parses_rejects_invalid_sql():
    bad_sql = "url('x') HEADERS('Authorization: Bearer xxx')"
    with pytest.raises(pytest.fail.Exception):
        assert_sql_parses(bad_sql)
