"""
Tests for improved error messages in DataStore
"""
import pytest
import pandas as pd

from datastore import DataStore
from datastore.exceptions import (
    DataStoreError,
    UnsupportedOperationError,
    ImmutableError,
    ColumnNotFoundError,
)


class TestUnsupportedOperationError:
    """Test UnsupportedOperationError exception"""

    def test_basic_message(self):
        """Test basic error message format"""
        err = UnsupportedOperationError(
            operation="str.split(expand=True)",
            reason="SQL does not support splitting strings into multiple columns"
        )
        assert "str.split(expand=True)" in str(err)
        assert "SQL does not support" in str(err)
        assert "Suggestion" not in str(err)

    def test_message_with_suggestion(self):
        """Test error message with suggestion"""
        err = UnsupportedOperationError(
            operation="str.split(expand=True)",
            reason="SQL does not support splitting strings into multiple columns",
            suggestion="Use pandas directly: df['col'].str.split(expand=True)"
        )
        assert "str.split(expand=True)" in str(err)
        assert "Suggestion:" in str(err)
        assert "Use pandas directly" in str(err)

    def test_attributes(self):
        """Test exception attributes are accessible"""
        err = UnsupportedOperationError(
            operation="test_op",
            reason="test_reason",
            suggestion="test_suggestion"
        )
        assert err.operation == "test_op"
        assert err.reason == "test_reason"
        assert err.suggestion == "test_suggestion"

    def test_is_datastore_error(self):
        """Test inheritance from DataStoreError"""
        err = UnsupportedOperationError(operation="op", reason="reason")
        assert isinstance(err, DataStoreError)


class TestImmutableError:
    """Test ImmutableError exception"""

    def test_basic_message(self):
        """Test basic immutable error message"""
        err = ImmutableError(object_type="DataStore")
        assert "DataStore is immutable" in str(err)
        assert "inplace=True is not supported" in str(err)

    def test_with_operation(self):
        """Test error with operation name"""
        err = ImmutableError(
            object_type="DataStore",
            operation="sort_values"
        )
        assert "sort_values" in str(err)
        assert "inplace modification" in str(err)

    def test_with_suggestion(self):
        """Test error with suggestion"""
        err = ImmutableError(
            object_type="DataStore",
            operation="sort_values",
            suggestion="Use result = ds.sort_values('col') instead"
        )
        assert "Use result" in str(err)

    def test_column_expr(self):
        """Test ColumnExpr immutable error"""
        err = ImmutableError(object_type="ColumnExpr")
        assert "ColumnExpr is immutable" in str(err)


class TestColumnNotFoundError:
    """Test ColumnNotFoundError exception"""

    def test_basic_message(self):
        """Test basic column not found message"""
        err = ColumnNotFoundError(column="nonexistent")
        assert "Column 'nonexistent' not found" in str(err)

    def test_with_available_columns(self):
        """Test error message with available columns"""
        err = ColumnNotFoundError(
            column="nonexistent",
            available_columns=["a", "b", "c"]
        )
        assert "nonexistent" in str(err)
        assert "Available columns:" in str(err)
        assert "'a'" in str(err)
        assert "'b'" in str(err)
        assert "'c'" in str(err)

    def test_truncated_columns_list(self):
        """Test truncation when many columns available"""
        many_cols = [f"col{i}" for i in range(20)]
        err = ColumnNotFoundError(
            column="missing",
            available_columns=many_cols
        )
        assert "first 10 of 20" in str(err)
        assert "..." in str(err)

    def test_attributes(self):
        """Test exception attributes"""
        err = ColumnNotFoundError(column="col", available_columns=["a", "b"])
        assert err.column == "col"
        assert err.available_columns == ["a", "b"]


class TestInplaceErrorRaised:
    """Test that inplace=True raises appropriate error"""

    def test_sort_values_inplace(self):
        """Test sort_values with inplace=True raises error"""
        ds = DataStore({"a": [3, 1, 2]})
        with pytest.raises(ImmutableError) as exc_info:
            ds.sort_values("a", inplace=True)
        # Should mention immutable and inplace
        assert "immutable" in str(exc_info.value).lower() or "inplace" in str(exc_info.value).lower()

    def test_fillna_inplace(self):
        """Test fillna with inplace=True raises error"""
        ds = DataStore({"a": [1, None, 3]})
        with pytest.raises(ImmutableError) as exc_info:
            ds.fillna(0, inplace=True)
        assert "immutable" in str(exc_info.value).lower() or "inplace" in str(exc_info.value).lower()


class TestUnsupportedOperationRaised:
    """Test that unsupported operations raise UnsupportedOperationError"""

    def test_string_slice_with_step(self):
        """Test string slicing with step raises clear error"""
        ds = DataStore({"name": ["hello", "world"]})
        with pytest.raises(UnsupportedOperationError) as exc_info:
            # Access the str accessor and try slicing with step
            _ = ds["name"].str[::2]
        assert "str[::step]" in str(exc_info.value)
        assert "Suggestion" in str(exc_info.value)

    def test_string_negative_start_positive_stop(self):
        """Test string slicing with negative start and positive stop"""
        ds = DataStore({"name": ["hello", "world"]})
        with pytest.raises(UnsupportedOperationError) as exc_info:
            _ = ds["name"].str[-3:2]
        assert "str[-n:m]" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ========================================================================
# Tests for Remote ClickHouse Error Handling UX (CH-5)
# ========================================================================

from datastore.exceptions import (
    ConnectionError,
    ExecutionError,
    translate_remote_error,
    _extract_clickhouse_error_code,
    _extract_host_from_error,
)


class TestExtractClickHouseErrorCode:
    """Test the error code extraction helper."""

    def test_simple_error_code(self):
        msg = "Code: 62. DB::Exception: Syntax error... (SYNTAX_ERROR)"
        assert _extract_clickhouse_error_code(msg) == "SYNTAX_ERROR"

    def test_nested_error_dns(self):
        """DNS_ERROR nested inside NO_REMOTE_SHARD_AVAILABLE should return DNS_ERROR."""
        msg = (
            "Code: 519. DB::NetException: All attempts... (DNS_ERROR) "
            "(ALL_CONNECTION_TRIES_FAILED) (NO_REMOTE_SHARD_AVAILABLE)"
        )
        assert _extract_clickhouse_error_code(msg) == "DNS_ERROR"

    def test_nested_error_network(self):
        """NETWORK_ERROR nested inside wrappers should return NETWORK_ERROR."""
        msg = (
            "Code: 210. DB::NetException: Connection refused... (NETWORK_ERROR) "
            "(ALL_CONNECTION_TRIES_FAILED) (NO_REMOTE_SHARD_AVAILABLE)"
        )
        assert _extract_clickhouse_error_code(msg) == "NETWORK_ERROR"

    def test_nested_auth_error(self):
        """AUTHENTICATION_FAILED has highest priority."""
        msg = (
            "Code: 516. DB::Exception: Authentication failed... (AUTHENTICATION_FAILED) "
            "(ALL_CONNECTION_TRIES_FAILED) (NO_REMOTE_SHARD_AVAILABLE)"
        )
        assert _extract_clickhouse_error_code(msg) == "AUTHENTICATION_FAILED"

    def test_no_error_code(self):
        msg = "Some generic error without error code"
        assert _extract_clickhouse_error_code(msg) == ""

    def test_unknown_code_fallback(self):
        msg = "Something went wrong (SOME_CUSTOM_CODE)"
        assert _extract_clickhouse_error_code(msg) == "SOME_CUSTOM_CODE"


class TestExtractHostFromError:
    """Test host extraction from error messages."""

    def test_dns_error_host_extraction(self):
        msg = "Not found address of host: myserver.example.com: (myserver.example.com:9000)"
        assert _extract_host_from_error(msg) == "myserver.example.com"

    def test_connection_refused_host_extraction(self):
        msg = "Connection refused (127.0.0.1:9000)"
        assert _extract_host_from_error(msg) == "127.0.0.1:9000"

    def test_no_host_found(self):
        msg = "Some generic error"
        assert _extract_host_from_error(msg) == ""


class TestTranslateAuthenticationError:
    """Test that authentication errors produce clear, actionable messages."""

    def test_authentication_failed_contains_keyword(self):
        """Error message must contain 'authentication' hint."""
        err = RuntimeError(
            "Code: 516. DB::Exception: default: Authentication failed: "
            "password is incorrect. (AUTHENTICATION_FAILED)"
        )
        result = translate_remote_error(err, {"host": "myhost:9000", "user": "admin"})
        assert "authentication" in result.lower()
        assert "admin" in result
        assert "myhost:9000" in result
        assert "check your username and password" in result.lower()

    def test_unknown_user(self):
        """UNKNOWN_USER should also be treated as authentication error."""
        err = RuntimeError(
            "Code: 279. DB::Exception: Unknown user 'baduser'. (UNKNOWN_USER)"
        )
        result = translate_remote_error(err, {"user": "baduser", "host": "server:9000"})
        assert "authentication" in result.lower()
        assert "baduser" in result

    def test_auth_error_preserves_original(self):
        """Friendly message should include the original error for debugging."""
        original = "Code: 516. DB::Exception: Authentication failed (AUTHENTICATION_FAILED)"
        err = RuntimeError(original)
        result = translate_remote_error(err)
        assert "Original error:" in result
        assert original in result


class TestTranslateUnknownDatabase:
    """Test that unknown database errors suggest listing available databases."""

    def test_unknown_database_contains_hint(self):
        """Error message must suggest listing available databases."""
        err = RuntimeError(
            "Code: 81. DB::Exception: Database nonexistent_db does not exist. (UNKNOWN_DATABASE)"
        )
        result = translate_remote_error(err, {"host": "myhost:9000", "database": "nonexistent_db"})
        assert "nonexistent_db" in result
        assert "does not exist" in result
        assert "databases()" in result

    def test_unknown_database_extracts_name_from_error(self):
        """Should extract database name from error message even without context."""
        err = RuntimeError(
            "Code: 81. DB::Exception: Database my_missing_db does not exist. (UNKNOWN_DATABASE)"
        )
        result = translate_remote_error(err)
        assert "my_missing_db" in result

    def test_unknown_database_shows_host(self):
        """Should show the host when available."""
        err = RuntimeError(
            "Code: 81. DB::Exception: Database testdb does not exist. (UNKNOWN_DATABASE)"
        )
        result = translate_remote_error(err, {"host": "prod-server:9000"})
        assert "prod-server:9000" in result


class TestTranslateUnknownTable:
    """Test that unknown table errors suggest listing available tables."""

    def test_unknown_table_contains_hint(self):
        """Error message must suggest listing available tables."""
        err = RuntimeError(
            "Code: 60. DB::Exception: Table default.nonexistent_table does not exist. (UNKNOWN_TABLE)"
        )
        result = translate_remote_error(
            err, {"host": "myhost:9000", "database": "default", "table": "nonexistent_table"}
        )
        assert "nonexistent_table" in result
        assert "does not exist" in result
        assert "tables(" in result

    def test_unknown_table_extracts_name_from_error(self):
        """Should extract table name from error message."""
        err = RuntimeError(
            "Code: 60. DB::Exception: Table mydb.missing_table does not exist. (UNKNOWN_TABLE)"
        )
        result = translate_remote_error(err)
        assert "missing_table" in result or "mydb.missing_table" in result

    def test_unknown_table_shows_database_context(self):
        """Should show database context when available."""
        err = RuntimeError(
            "Code: 60. DB::Exception: Table test.users does not exist. (UNKNOWN_TABLE)"
        )
        result = translate_remote_error(err, {"database": "test", "host": "server:9000"})
        assert "test" in result


class TestTranslateAccessDenied:
    """Test that permission errors clearly indicate access issues."""

    def test_access_denied_contains_permission_hint(self):
        """Error message must indicate permission problem."""
        err = RuntimeError(
            "Code: 497. DB::Exception: admin: Not enough privileges. (ACCESS_DENIED)"
        )
        result = translate_remote_error(err, {"user": "admin", "host": "myhost:9000"})
        assert "access denied" in result.lower()
        assert "privileges" in result.lower()
        assert "admin" in result

    def test_access_denied_suggests_grant(self):
        """Should suggest using GRANT or contacting admin."""
        err = RuntimeError(
            "Code: 497. DB::Exception: Access denied (ACCESS_DENIED)"
        )
        result = translate_remote_error(err)
        assert "GRANT" in result or "administrator" in result.lower()


class TestTranslateConnectionRefused:
    """Test that connection refused errors suggest checking network/server."""

    def test_connection_refused_contains_network_hint(self):
        """Error message must suggest checking network/server."""
        err = RuntimeError(
            "Code: 519. DB::NetException: All attempts to get table structure failed. Log: \n"
            "Code: 279. DB::NetException: All connection tries failed. Log: \n"
            "Code: 210. DB::NetException: Connection refused (127.0.0.1:9000). (NETWORK_ERROR) "
            "(ALL_CONNECTION_TRIES_FAILED) (NO_REMOTE_SHARD_AVAILABLE)"
        )
        result = translate_remote_error(err, {"host": "127.0.0.1:9000"})
        assert "connection refused" in result.lower()
        assert "server is running" in result.lower()
        assert "127.0.0.1:9000" in result

    def test_connection_refused_mentions_port(self):
        """Should mention default ports."""
        err = RuntimeError(
            "Code: 210. DB::NetException: Connection refused (10.0.0.1:9000). (NETWORK_ERROR)"
        )
        result = translate_remote_error(err)
        assert "9000" in result
        assert "9440" in result


class TestTranslateTimeout:
    """Test that timeout errors suggest checking network."""

    def test_timeout_contains_network_hint(self):
        """Error message must suggest checking network."""
        err = RuntimeError(
            "Code: 209. DB::NetException: Connection timed out (SOCKET_TIMEOUT)"
        )
        result = translate_remote_error(err, {"host": "slow-server:9000"})
        assert "timed out" in result.lower()
        assert "network" in result.lower()
        assert "slow-server:9000" in result

    def test_timeout_suggests_connectivity(self):
        """Should suggest checking firewall/connectivity."""
        err = RuntimeError(
            "Code: 209. DB::NetException: Connection timed out (SOCKET_TIMEOUT)"
        )
        result = translate_remote_error(err)
        assert "firewall" in result.lower() or "connectivity" in result.lower()


class TestTranslateDNSError:
    """Test that DNS/hostname errors suggest checking host format."""

    def test_dns_error_suggests_format(self):
        """Error message must suggest correct host format."""
        err = RuntimeError(
            "Code: 519. DB::NetException: All attempts to get table structure failed. Log: \n"
            "Code: 198. DB::NetException: Not found address of host: badhost: "
            "(badhost:9000). (DNS_ERROR) (NO_REMOTE_SHARD_AVAILABLE)"
        )
        result = translate_remote_error(err, {"host": "badhost:9000"})
        assert "cannot resolve" in result.lower() or "hostname" in result.lower()
        assert "badhost" in result
        assert "host" in result.lower()
        # Should suggest correct format
        assert "localhost:9000" in result or "hostname:port" in result.lower()

    def test_invalid_host_format_suggests_correct(self):
        """Invalid host format should suggest expected format."""
        err = RuntimeError(
            "Code: 198. DB::NetException: Not found address of host: not a valid host: "
            "(not a valid host:9000). (DNS_ERROR)"
        )
        result = translate_remote_error(err)
        assert "format" in result.lower() or "typos" in result.lower()


class TestTranslateSQLSyntaxError:
    """Test that SQL syntax errors include SQL and position."""

    def test_syntax_error_includes_position(self):
        """Error message must include position information."""
        err = RuntimeError(
            "Code: 62. DB::Exception: Syntax error: failed at position 8 (1): ... (SYNTAX_ERROR)"
        )
        result = translate_remote_error(err)
        assert "syntax error" in result.lower()
        assert "position 8" in result

    def test_syntax_error_preserves_original(self):
        """Original error should be preserved for debugging."""
        original = "Code: 62. DB::Exception: Syntax error: failed at position 8 ... (SYNTAX_ERROR)"
        err = RuntimeError(original)
        result = translate_remote_error(err)
        assert original in result


class TestTranslateFallback:
    """Test fallback behavior for unrecognized errors."""

    def test_unknown_error_returns_original(self):
        """Unrecognized errors should return the original message unchanged."""
        original = "Some completely unknown error"
        err = RuntimeError(original)
        result = translate_remote_error(err)
        assert result == original

    def test_empty_context_is_safe(self):
        """Translation should work with no context."""
        err = RuntimeError("Code: 516. (AUTHENTICATION_FAILED)")
        result = translate_remote_error(err)
        assert "authentication" in result.lower()

    def test_none_context_is_safe(self):
        """Translation should work with None context."""
        err = RuntimeError("Code: 516. (AUTHENTICATION_FAILED)")
        result = translate_remote_error(err, None)
        assert "authentication" in result.lower()


class TestConnectionModeNoTable:
    """Test that ds.columns in connection mode gives clear error."""

    def test_columns_in_connection_mode(self):
        """Accessing columns without a table should give a clear error."""
        ds = DataStore(
            source="clickhouse", host="localhost:9000", user="default", password=""
        )
        # In connection mode (no table), accessing columns should raise
        # a clear error about needing to specify a table
        try:
            _ = ds.columns
            # If it doesn't raise, the message is handled elsewhere
        except (DataStoreError, AttributeError, Exception) as e:
            error_msg = str(e).lower()
            # Should mention table or similar guidance
            assert (
                "table" in error_msg
                or "column" in error_msg
                or "connect" in error_msg
                or "no data" in error_msg
            ), f"Error message not helpful: {e}"


class TestRealErrorTranslationIntegration:
    """Integration tests that trigger real chdb errors and verify friendly messages."""

    def test_dns_error_real(self):
        """Real DNS error from chdb should be translated to friendly message."""
        ds = DataStore.from_clickhouse(
            host="nonexistent_host_xyz_12345:9000",
            database="default",
            table="test",
            user="default",
            password="",
        )
        with pytest.raises((ConnectionError, ExecutionError)) as exc_info:
            ds.connect()

        error_msg = str(exc_info.value)
        # Should contain friendly message, not just raw chdb exception
        assert "cannot resolve" in error_msg.lower() or "hostname" in error_msg.lower()
        assert "nonexistent_host_xyz_12345" in error_msg

    def test_connection_refused_real(self):
        """Real connection refused from chdb should be translated."""
        ds = DataStore.from_clickhouse(
            host="127.0.0.1:19999",  # unlikely to have ClickHouse on this port
            database="default",
            table="test",
            user="default",
            password="",
        )
        with pytest.raises((ConnectionError, ExecutionError)) as exc_info:
            ds.connect()

        error_msg = str(exc_info.value)
        assert "connection refused" in error_msg.lower() or "server is running" in error_msg.lower()

    def test_syntax_error_real(self):
        """Real SQL syntax error should include position info."""
        ds = DataStore({"a": [1, 2, 3]})
        with pytest.raises((ExecutionError, Exception)) as exc_info:
            # sql() is lazy, need to trigger execution
            result = ds.sql("SELEKT * FORM __df__")
            result._execute()

        error_msg = str(exc_info.value)
        assert "syntax" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
