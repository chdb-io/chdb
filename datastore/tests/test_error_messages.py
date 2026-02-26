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
