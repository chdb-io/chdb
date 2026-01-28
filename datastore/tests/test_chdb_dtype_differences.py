"""
Test verifying dtype preservation between chDB and pandas.

These tests verify that chDB correctly preserves dtypes when data passes
through chDB's Python() table function:

1. Float columns with NaN: preserved as float64
2. Integer columns with None: preserved as float64
3. Datetime columns: preserved as datetime64[ns]

NOTE: These tests were previously marked xfail due to dtype conversion issues
in older chDB versions. As of 2026-01-06, chDB correctly preserves dtypes.
"""

import numpy as np
import pandas as pd
import pytest
from tests.xfail_markers import chdb_array_nullable

import chdb


class TestChDBDtypeDifferences:
    """Verify dtype preservation between chDB output and pandas."""

    def test_float_nan_dtype_preservation(self):
        """
        Verify chDB preserves float64 columns containing NaN.

        Original: float64 with NaN (numpy.nan)
        After chDB: float64 (preserved)
        """
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        assert df["a"].dtype == np.float64

        result = chdb.query("SELECT * FROM Python(df)", "DataFrame")

        # chDB returns Float64Dtype() instead of float64
        assert result["a"].dtype == np.float64

    def test_integer_none_dtype_preservation(self):
        """
        Verify chDB preserves columns with None.

        Note: pandas converts int-like list with None to float64 (not int64),
        and chDB preserves this as float64.
        """
        df = pd.DataFrame({"a": [1, 2, None, 4]})
        original_dtype = df["a"].dtype  # float64 (because of None)

        result = chdb.query("SELECT * FROM Python(df)", "DataFrame")

        # chDB returns Float64Dtype()
        assert result["a"].dtype == original_dtype

    def test_datetime_timezone_preservation(self):
        """
        Verify chDB preserves naive datetime columns.

        Original: datetime64[ns] (naive)
        After chDB: datetime64[ns] (preserved)
        """
        df = pd.DataFrame({"dt": pd.to_datetime(["2021-01-01", "2021-01-02"])})
        assert df["dt"].dtype == "datetime64[ns]"

        result = chdb.query("SELECT * FROM Python(df)", "DataFrame")

        # chDB adds timezone
        assert result["dt"].dtype == df["dt"].dtype


class TestChDBArrayNullableLimitation:
    """Document ClickHouse's Array(T) in Nullable limitation and workaround."""

    def test_datastore_split_preserves_none_via_pandas_fallback(self):
        """
        DataStore's str.split() may use pandas fallback execution,
        which preserves None values like pandas does.

        This is actually the CORRECT pandas-compatible behavior.
        """
        from datastore import DataStore

        df = pd.DataFrame({'text': ['hello world', None, 'foo bar']})

        # pandas behavior
        pd_result = df['text'].str.split()

        # DataStore behavior (may use pandas fallback)
        ds = DataStore.from_df(df)
        ds_result = ds['text'].str.split().to_pandas()

        # Both should preserve None for NULL values
        assert pd_result.iloc[1] is None
        assert ds_result.iloc[1] is None

    @chdb_array_nullable
    def test_raw_sql_split_without_ifnull_fails(self):
        """
        Using splitByWhitespace directly in SQL without ifNull fails
        with 'Nested type Array(String) cannot be inside Nullable type' error.

        This documents the ClickHouse limitation that requires the ifNull workaround.
        """
        df = pd.DataFrame({'text': ['hello world', None, 'foo bar']})

        # This should fail: no ifNull wrapper
        result = chdb.query("SELECT splitByWhitespace(text) FROM Python(df)", 'DataFrame')
        assert len(result) == 3
