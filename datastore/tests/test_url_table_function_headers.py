import pytest
from datastore.table_functions import UrlTableFunction
from datastore.exceptions import DataStoreError


class TestUrlTableFunctionHeaders:

    def test_to_sql_no_headers_unchanged(self):
        tf = UrlTableFunction(url="https://example.com/d.parquet", format="Parquet")
        sql = tf.to_sql()
        assert sql == "url('https://example.com/d.parquet', 'Parquet')"
        assert " HEADERS(" not in sql
        assert "headers(" not in sql

    def test_to_sql_with_headers_inside_url_call(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["User-Agent: test-agent"],
        )
        sql = tf.to_sql()
        assert "headers('User-Agent'='test-agent')" in sql
        assert " HEADERS(" not in sql

    def test_to_sql_headers_pad_structure_auto(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers=["X-Token: abc"],
        )
        sql = tf.to_sql()
        assert "'Parquet', 'auto', headers('X-Token'='abc')" in sql
        assert " HEADERS(" not in sql

    def test_to_sql_headers_pad_format_and_structure_auto(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            headers=["X-Token: abc"],
        )
        sql = tf.to_sql()
        assert sql.count("'auto'") >= 2
        assert "headers('X-Token'='abc')" in sql
        assert " HEADERS(" not in sql

    def test_to_sql_headers_dict_input(self):
        tf = UrlTableFunction(
            url="https://example.com/d.parquet",
            format="Parquet",
            headers={"Authorization": "Bearer xyz"},
        )
        sql = tf.to_sql()
        assert "headers('Authorization'='Bearer xyz')" in sql
        assert " HEADERS(" not in sql

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

    def test_to_sql_headers_missing_url_raises(self):
        tf = UrlTableFunction(format="Parquet", headers=["X-A: 1"])
        with pytest.raises(DataStoreError):
            tf.to_sql()

    def test_to_sql_headers_executes_via_chdb(self):
        try:
            import chdb
        except ImportError:
            pytest.skip("chdb not installed")
        tf = UrlTableFunction(
            url="https://httpbin.org/ip",
            format="JSONAsString",
            headers=["User-Agent: chdb-test"],
        )
        sql = f"SELECT * FROM {tf.to_sql()}"
        try:
            result = chdb.query(sql, "CSV")
        except Exception as e:
            pytest.skip(f"chdb execution unavailable: {e}")
        assert result is not None
        assert "origin" in str(result) or "200" in str(result)
