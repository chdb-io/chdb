"""
Test ClickHouse-specific data type round-trip through DataStore.

CH-4: Verify that ClickHouse-specific types are correctly preserved
when data passes through ClickHouse table -> DataStore -> pandas DataFrame.

Types tested:
- Decimal(18,4): precision preservation
- FixedString(16): trailing NULL byte handling
- Enum8: string mapping
- UUID: format and type preservation
- IPv4 / IPv6: address preservation
- LowCardinality(String): transparent handling
- Nullable(Int64): NULL handling with pandas nullable dtype
- DateTime64(3, 'UTC'): sub-second precision and timezone
- UInt64: large value precision (> 2^53)
- Map(String, String): dict mapping
- Array(String): array mapping
"""

import ipaddress
import uuid

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore


class TestClickHouseTypeRoundTrip:
    """Verify ClickHouse-specific data types preserve data through DataStore."""

    def _setup_table(self, table_name, ddl, insert_sql):
        """Create a DataStore with test table, insert data, return (ds, conn)."""
        ds = DataStore(table=table_name)
        ds.connect()
        conn = ds._connection
        conn.execute(ddl)
        if isinstance(insert_sql, list):
            for sql in insert_sql:
                conn.execute(sql)
        else:
            conn.execute(insert_sql)
        return ds, conn

    def _read_df(self, ds):
        """Read table via DataStore and return pandas DataFrame."""
        result = ds.select("*").execute()
        return result.to_df()

    def _cleanup(self, ds, conn, table_name):
        """Drop table and close connection."""
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        except Exception:
            pass
        ds.close()

    # -- Decimal(18,4) --

    def test_decimal_precision_preserved(self):
        """Decimal(18,4) values should preserve precision for normal-range values."""
        ds, conn = self._setup_table(
            "t_decimal",
            "CREATE TABLE t_decimal (val Decimal(18,4)) ENGINE = Memory",
            "INSERT INTO t_decimal VALUES (123.4567), (0.0001), (-999.9999), (0)",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert float(values[0]) == pytest.approx(123.4567)
            assert float(values[1]) == pytest.approx(0.0001)
            assert float(values[2]) == pytest.approx(-999.9999)
            assert float(values[3]) == pytest.approx(0.0)
        finally:
            self._cleanup(ds, conn, "t_decimal")

    def test_decimal_maps_to_float64(self):
        """Decimal(18,4) maps to float64 dtype in pandas."""
        ds, conn = self._setup_table(
            "t_decimal_dt",
            "CREATE TABLE t_decimal_dt (val Decimal(18,4)) ENGINE = Memory",
            "INSERT INTO t_decimal_dt VALUES (1.0)",
        )
        try:
            df = self._read_df(ds)
            assert df["val"].dtype == np.float64
        finally:
            self._cleanup(ds, conn, "t_decimal_dt")

    def test_decimal_large_value_precision_loss(self):
        """Decimal(18,4) with very large values may lose precision via float64.

        This documents the known limitation: Decimal -> float64 conversion
        cannot represent all Decimal(18,4) values exactly.
        99999999999999.9999 becomes 100000000000000.0 in float64.
        """
        ds, conn = self._setup_table(
            "t_decimal_lg",
            "CREATE TABLE t_decimal_lg (val Decimal(18,4)) ENGINE = Memory",
            "INSERT INTO t_decimal_lg VALUES (99999999999999.9999)",
        )
        try:
            df = self._read_df(ds)
            val = float(df["val"].iloc[0])
            # float64 cannot represent 99999999999999.9999 exactly
            # It rounds to 100000000000000.0
            assert val == pytest.approx(100000000000000.0)
        finally:
            self._cleanup(ds, conn, "t_decimal_lg")

    # -- FixedString(16) --

    def test_fixedstring_has_trailing_null_bytes(self):
        """FixedString(N) pads short values with NULL bytes.

        This is standard ClickHouse behavior. Values shorter than N
        are right-padded with null bytes to exactly N characters.
        """
        ds, conn = self._setup_table(
            "t_fstr",
            "CREATE TABLE t_fstr (val FixedString(8)) ENGINE = Memory",
            "INSERT INTO t_fstr VALUES ('hello'), ('a'), ('12345678')",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            # Short values are padded with null bytes
            assert values[0] == "hello\x00\x00\x00"
            assert len(values[0]) == 8
            assert values[1] == "a\x00\x00\x00\x00\x00\x00\x00"
            assert len(values[1]) == 8
            # Exact-length value has no padding
            assert values[2] == "12345678"
            assert len(values[2]) == 8
        finally:
            self._cleanup(ds, conn, "t_fstr")

    def test_fixedstring_dtype_is_object(self):
        """FixedString maps to a string-like dtype (object or StringDtype)."""
        ds, conn = self._setup_table(
            "t_fstr_dt",
            "CREATE TABLE t_fstr_dt (val FixedString(4)) ENGINE = Memory",
            "INSERT INTO t_fstr_dt VALUES ('test')",
        )
        try:
            df = self._read_df(ds)
            assert df["val"].dtype == object or pd.api.types.is_string_dtype(df["val"])
            assert isinstance(df["val"].iloc[0], str)
        finally:
            self._cleanup(ds, conn, "t_fstr_dt")

    def test_fixedstring_strippable(self):
        """FixedString values can be cleaned by stripping null bytes."""
        ds, conn = self._setup_table(
            "t_fstr_strip",
            "CREATE TABLE t_fstr_strip (val FixedString(16)) ENGINE = Memory",
            "INSERT INTO t_fstr_strip VALUES ('hello'), ('world')",
        )
        try:
            df = self._read_df(ds)
            cleaned = [v.rstrip("\x00") for v in df["val"]]
            assert cleaned == ["hello", "world"]
        finally:
            self._cleanup(ds, conn, "t_fstr_strip")

    # -- Enum8 --

    def test_enum8_maps_to_string(self):
        """Enum8 values are returned as their string labels."""
        ds, conn = self._setup_table(
            "t_enum",
            "CREATE TABLE t_enum (val Enum8('apple'=1, 'banana'=2, 'cherry'=3)) ENGINE = Memory",
            "INSERT INTO t_enum VALUES ('apple'), ('banana'), ('cherry'), ('apple')",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert values == ["apple", "banana", "cherry", "apple"]
            assert df["val"].dtype == object or pd.api.types.is_string_dtype(df["val"])
        finally:
            self._cleanup(ds, conn, "t_enum")

    # -- UUID --

    def test_uuid_preserved_as_uuid_objects(self):
        """UUID values are returned as Python uuid.UUID objects."""
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        nil_uuid = "00000000-0000-0000-0000-000000000000"

        ds, conn = self._setup_table(
            "t_uuid",
            "CREATE TABLE t_uuid (val UUID) ENGINE = Memory",
            f"INSERT INTO t_uuid VALUES ('{test_uuid}'), ('{nil_uuid}')",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert isinstance(values[0], uuid.UUID)
            assert isinstance(values[1], uuid.UUID)
            assert str(values[0]) == test_uuid
            assert str(values[1]) == nil_uuid
        finally:
            self._cleanup(ds, conn, "t_uuid")

    # -- IPv4 --

    def test_ipv4_preserved_as_ipaddress(self):
        """IPv4 values are returned as ipaddress.IPv4Address objects."""
        ds, conn = self._setup_table(
            "t_ipv4",
            "CREATE TABLE t_ipv4 (val IPv4) ENGINE = Memory",
            "INSERT INTO t_ipv4 VALUES ('192.168.1.1'), ('10.0.0.1'), ('255.255.255.255')",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert isinstance(values[0], ipaddress.IPv4Address)
            assert str(values[0]) == "192.168.1.1"
            assert str(values[1]) == "10.0.0.1"
            assert str(values[2]) == "255.255.255.255"
        finally:
            self._cleanup(ds, conn, "t_ipv4")

    # -- IPv6 --

    def test_ipv6_preserved_as_ipaddress(self):
        """IPv6 values are returned as ipaddress.IPv6Address objects."""
        ds, conn = self._setup_table(
            "t_ipv6",
            "CREATE TABLE t_ipv6 (val IPv6) ENGINE = Memory",
            "INSERT INTO t_ipv6 VALUES ('::1'), ('2001:db8::1'), ('fe80::1')",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert isinstance(values[0], ipaddress.IPv6Address)
            assert str(values[0]) == "::1"
            assert str(values[1]) == "2001:db8::1"
            assert str(values[2]) == "fe80::1"
        finally:
            self._cleanup(ds, conn, "t_ipv6")

    # -- LowCardinality(String) --

    def test_lowcardinality_string_transparent(self):
        """LowCardinality(String) should be indistinguishable from String."""
        ds, conn = self._setup_table(
            "t_lowcard",
            "CREATE TABLE t_lowcard (val LowCardinality(String)) ENGINE = Memory",
            "INSERT INTO t_lowcard VALUES ('hello'), ('world'), ('hello')",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert values == ["hello", "world", "hello"]
            assert df["val"].dtype == object or pd.api.types.is_string_dtype(df["val"])
            assert isinstance(values[0], str)
        finally:
            self._cleanup(ds, conn, "t_lowcard")

    # -- Nullable(Int64) --

    def test_nullable_int64_with_values(self):
        """Nullable(Int64) with non-null values preserves integers."""
        ds, conn = self._setup_table(
            "t_nullable_val",
            "CREATE TABLE t_nullable_val (val Nullable(Int64)) ENGINE = Memory",
            "INSERT INTO t_nullable_val VALUES (42), (-1), (0)",
        )
        try:
            df = self._read_df(ds)

            assert int(df["val"].iloc[0]) == 42
            assert int(df["val"].iloc[1]) == -1
            assert int(df["val"].iloc[2]) == 0
        finally:
            self._cleanup(ds, conn, "t_nullable_val")

    def test_nullable_int64_with_nulls(self):
        """Nullable(Int64) with NULL uses pandas nullable Int64 dtype."""
        ds, conn = self._setup_table(
            "t_nullable_null",
            "CREATE TABLE t_nullable_null (val Nullable(Int64)) ENGINE = Memory",
            "INSERT INTO t_nullable_null VALUES (42), (NULL), (-1)",
        )
        try:
            df = self._read_df(ds)

            # pandas nullable Int64 dtype preserves integer type with NA support
            assert df["val"].dtype == pd.Int64Dtype()

            assert int(df["val"].iloc[0]) == 42
            assert pd.isna(df["val"].iloc[1])
            assert int(df["val"].iloc[2]) == -1
        finally:
            self._cleanup(ds, conn, "t_nullable_null")

    # -- DateTime64(3, 'UTC') --

    def test_datetime64_subsecond_precision(self):
        """DateTime64(3, 'UTC') preserves millisecond precision."""
        ds, conn = self._setup_table(
            "t_dt64",
            "CREATE TABLE t_dt64 (val DateTime64(3, 'UTC')) ENGINE = Memory",
            "INSERT INTO t_dt64 VALUES ('2024-01-15 10:30:45.123'), ('2024-06-30 23:59:59.999')",
        )
        try:
            df = self._read_df(ds)

            ts0 = pd.Timestamp(df["val"].iloc[0])
            ts1 = pd.Timestamp(df["val"].iloc[1])

            # Verify date/time components
            assert ts0.year == 2024
            assert ts0.month == 1
            assert ts0.day == 15
            assert ts0.hour == 10
            assert ts0.minute == 30
            assert ts0.second == 45
            # Millisecond precision: 123ms = 123000 microseconds
            assert ts0.microsecond == 123000

            assert ts1.hour == 23
            assert ts1.minute == 59
            assert ts1.second == 59
            assert ts1.microsecond == 999000
        finally:
            self._cleanup(ds, conn, "t_dt64")

    def test_datetime64_timezone_preserved(self):
        """DateTime64(3, 'UTC') preserves UTC timezone."""
        ds, conn = self._setup_table(
            "t_dt64_tz",
            "CREATE TABLE t_dt64_tz (val DateTime64(3, 'UTC')) ENGINE = Memory",
            "INSERT INTO t_dt64_tz VALUES ('2024-01-15 10:30:45.000')",
        )
        try:
            df = self._read_df(ds)

            # Should have timezone-aware datetime dtype
            dtype_str = str(df["val"].dtype)
            assert "datetime64" in dtype_str
            assert "UTC" in dtype_str

            ts = pd.Timestamp(df["val"].iloc[0])
            assert ts.tz is not None
            assert str(ts.tz) == "UTC"
        finally:
            self._cleanup(ds, conn, "t_dt64_tz")

    # -- UInt64 (large values) --

    def test_uint64_preserves_precision_above_2_53(self):
        """UInt64 values > 2^53 should not lose precision.

        2^53 + 1 = 9007199254740993 is the first integer not representable
        in float64. UInt64 dtype preserves exact values.
        """
        ds, conn = self._setup_table(
            "t_uint64",
            "CREATE TABLE t_uint64 (val UInt64) ENGINE = Memory",
            "INSERT INTO t_uint64 VALUES (9007199254740993), (18446744073709551615)",
        )
        try:
            df = self._read_df(ds)

            # UInt64 dtype preserves exact values
            assert df["val"].dtype == np.dtype("uint64")

            val0 = df["val"].iloc[0]
            val1 = df["val"].iloc[1]

            assert int(val0) == 9007199254740993
            assert int(val1) == 18446744073709551615
        finally:
            self._cleanup(ds, conn, "t_uint64")

    # -- Map(String, String) --

    def test_map_returns_dict(self):
        """Map(String, String) values are returned as Python dicts."""
        ds, conn = self._setup_table(
            "t_map",
            "CREATE TABLE t_map (val Map(String, String)) ENGINE = Memory",
            "INSERT INTO t_map VALUES (map('key1', 'val1', 'key2', 'val2')), (map('only', 'one')), (map())",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            assert isinstance(values[0], dict)
            assert values[0] == {"key1": "val1", "key2": "val2"}

            assert isinstance(values[1], dict)
            assert values[1] == {"only": "one"}

            assert isinstance(values[2], dict)
            assert values[2] == {}
        finally:
            self._cleanup(ds, conn, "t_map")

    # -- Array(String) --

    def test_array_returns_ndarray(self):
        """Array(String) values are returned as numpy arrays."""
        ds, conn = self._setup_table(
            "t_arr",
            "CREATE TABLE t_arr (val Array(String)) ENGINE = Memory",
            "INSERT INTO t_arr VALUES (['one', 'two', 'three']), (['single']), ([])",
        )
        try:
            df = self._read_df(ds)
            values = df["val"].tolist()

            # Array values come as numpy arrays
            assert list(values[0]) == ["one", "two", "three"]
            assert list(values[1]) == ["single"]
            assert list(values[2]) == []
        finally:
            self._cleanup(ds, conn, "t_arr")

    # -- Combined round-trip --

    def test_all_types_single_table(self):
        """All ClickHouse-specific types can coexist in one table."""
        ds, conn = self._setup_table(
            "t_all_types",
            """CREATE TABLE t_all_types (
                decimal_val Decimal(18,4),
                fixed_str FixedString(8),
                enum_val Enum8('x'=1, 'y'=2),
                uuid_val UUID,
                ipv4_val IPv4,
                ipv6_val IPv6,
                lowcard_str LowCardinality(String),
                nullable_int Nullable(Int64),
                dt64_val DateTime64(3, 'UTC'),
                uint64_val UInt64,
                map_val Map(String, String),
                arr_val Array(String)
            ) ENGINE = Memory""",
            """INSERT INTO t_all_types VALUES (
                1.5,
                'test',
                'x',
                '550e8400-e29b-41d4-a716-446655440000',
                '127.0.0.1',
                '::1',
                'low',
                100,
                '2024-01-01 00:00:00.500',
                42,
                map('a', 'b'),
                ['hello']
            )""",
        )
        try:
            df = self._read_df(ds)

            assert len(df) == 1
            assert len(df.columns) == 12

            row = df.iloc[0]
            assert float(row["decimal_val"]) == pytest.approx(1.5)
            assert row["fixed_str"].rstrip("\x00") == "test"
            assert row["enum_val"] == "x"
            assert str(row["uuid_val"]) == "550e8400-e29b-41d4-a716-446655440000"
            assert str(row["ipv4_val"]) == "127.0.0.1"
            assert str(row["ipv6_val"]) == "::1"
            assert row["lowcard_str"] == "low"
            assert int(row["nullable_int"]) == 100
            assert pd.Timestamp(row["dt64_val"]).microsecond == 500000
            assert int(row["uint64_val"]) == 42
            assert row["map_val"] == {"a": "b"}
            assert list(row["arr_val"]) == ["hello"]
        finally:
            self._cleanup(ds, conn, "t_all_types")
