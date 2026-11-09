"""
Conditions assigned as columns (``ds[col] = cond``, ``assign(col=cond)``,
``assign(col=lambda d: cond)``).

pandas mode pushes a condition to SQL only when every input column can't hold
pandas NA (never NULL, or NULL only for NaN/NaT); the SQL result is a bool
column like pandas. Conditions over nullable inputs run in pandas.
Performance mode always pushes them and returns SQL's native result.
"""

import chdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datastore import DataStore
from datastore.config import (
    CompatMode,
    ExecutionEngine,
    set_compat_mode,
    set_execution_engine,
)
from tests.test_utils import assert_frame_equal, get_dataframe


@pytest.fixture
def performance_mode():
    set_compat_mode(CompatMode.PERFORMANCE)
    yield
    set_compat_mode(CompatMode.PANDAS)
    set_execution_engine(ExecutionEngine.AUTO)


@pytest.fixture(autouse=True)
def pandas_mode_after_test():
    yield
    set_compat_mode(CompatMode.PANDAS)
    set_execution_engine(ExecutionEngine.AUTO)


def _frame():
    return pd.DataFrame(
        {
            "i": [1, 5, 3, 7],
            "f": [1.0, np.nan, 3.0, 7.0],
            "b": [True, False, True, False],
        }
    )


def _nullable_frame():
    return pd.DataFrame(
        {
            "n": pd.array([1, None, 3, 7], dtype="Int64"),
            "nb": pd.array([True, None, False, True], dtype="boolean"),
        }
    )


@pytest.fixture
def non_null_parquet(tmp_path):
    """Parquet whose columns are not Nullable in ClickHouse; f holds a NaN value."""
    path = str(tmp_path / "non_null.parquet")
    schema = pa.schema(
        [
            pa.field("i", pa.int64(), nullable=False),
            pa.field("f", pa.float64(), nullable=False),
            pa.field("b", pa.bool_(), nullable=False),
        ]
    )
    frame = _frame()
    table = pa.Table.from_arrays(
        [pa.array(frame[name].tolist(), type=schema.field(name).type) for name in frame],
        schema=schema,
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def nullable_parquet(tmp_path):
    path = str(tmp_path / "nullable.parquet")
    pd.DataFrame({"n": pd.array([1, None, 3, 7], dtype="Int64")}).to_parquet(path)
    return path


NON_NULLABLE_CONDITIONS = [
    pytest.param(lambda d: d["i"] > 2, '"i" > 2', id="int_gt"),
    pytest.param(lambda d: d["f"] > 2, '"f" > 2', id="float_nan_gt"),
    pytest.param(lambda d: d["f"] != 3, '"f" != 3', id="float_nan_ne"),
    pytest.param(lambda d: ~(d["f"] != 3), '"f" != 3', id="not_float_nan_ne"),
    pytest.param(lambda d: d["b"] & (d["i"] > 2), '"i" > 2', id="bool_and"),
    pytest.param(lambda d: (d["f"] > 2) | d["b"], '"f" > 2', id="float_or_bool"),
    pytest.param(lambda d: ~d["b"], '"b"', id="not_bool"),
]

ENTRY_POINTS = ["setitem", "assign", "assign_callable"]


def _assign(ds, how, cond):
    if how == "setitem":
        ds["c"] = cond(ds)
        return ds
    if how == "assign":
        return ds.assign(c=cond(ds))
    return ds.assign(c=cond)


class TestPandasMode:
    @pytest.mark.parametrize("how", ENTRY_POINTS)
    @pytest.mark.parametrize("cond, sql_fragment", NON_NULLABLE_CONDITIONS)
    def test_non_nullable_condition_is_pushed_and_bool(
        self, non_null_parquet, how, cond, sql_fragment
    ):
        ds_result = _assign(DataStore.from_file(non_null_parquet), how, cond)
        sql = ds_result.to_sql(execution_format=True)
        assert sql_fragment in sql
        assert "toBool(" in sql

        pd_result = _assign(pd.read_parquet(non_null_parquet), how, cond)
        assert_frame_equal(get_dataframe(ds_result), pd_result)
        assert_frame_equal(chdb.query(sql, "DataFrame"), pd_result)

    @pytest.mark.parametrize("how", ENTRY_POINTS)
    @pytest.mark.parametrize("cond, sql_fragment", NON_NULLABLE_CONDITIONS)
    def test_dataframe_source_matches_pandas_dtype(self, how, cond, sql_fragment):
        ds_result = _assign(DataStore(_frame()), how, cond)
        assert_frame_equal(get_dataframe(ds_result), _assign(_frame(), how, cond))

    @pytest.mark.parametrize("how", ENTRY_POINTS)
    @pytest.mark.parametrize(
        "cond",
        [
            pytest.param(lambda d: d["n"] > 2, id="int64_gt"),
            pytest.param(lambda d: d["n"] != 3, id="int64_ne"),
            pytest.param(lambda d: d["nb"] & (d["n"] > 2), id="boolean_and"),
            pytest.param(lambda d: ~d["nb"], id="boolean_not"),
        ],
    )
    def test_nullable_extension_condition_runs_in_pandas(self, how, cond):
        ds_result = _assign(DataStore(_nullable_frame()), how, cond)
        assert_frame_equal(get_dataframe(ds_result), _assign(_nullable_frame(), how, cond))

    @pytest.mark.parametrize("how", ENTRY_POINTS)
    def test_nullable_sql_column_runs_in_pandas(self, nullable_parquet, how):
        def cond(d):
            return d["n"] > 2

        ds_result = _assign(DataStore.from_file(nullable_parquet), how, cond)
        assert '"n" > 2' not in ds_result.to_sql(execution_format=True)
        pd_result = _assign(pd.read_parquet(nullable_parquet), how, cond)
        assert_frame_equal(get_dataframe(ds_result), pd_result)


class TestPerformanceMode:
    @pytest.mark.parametrize("how", ENTRY_POINTS)
    @pytest.mark.parametrize(
        "cond, sql_fragment",
        [
            pytest.param(lambda d: d["i"] > 2, '"i" > 2 AS "c"', id="int_gt"),
            pytest.param(
                lambda d: d["b"] & (d["i"] > 2),
                '("b" = 1 AND "i" > 2) AS "c"',
                id="bool_and",
            ),
        ],
    )
    def test_non_nullable_condition_is_pushed_natively(
        self, non_null_parquet, performance_mode, how, cond, sql_fragment
    ):
        ds_result = _assign(DataStore.from_file(non_null_parquet), how, cond)
        sql = ds_result.to_sql(execution_format=True)
        assert sql_fragment in sql
        assert "toBool(" not in sql

        expected = _assign(pd.read_parquet(non_null_parquet), how, cond)
        expected["c"] = expected["c"].astype("uint8")
        assert_frame_equal(get_dataframe(ds_result), expected)

    @pytest.mark.parametrize("how", ENTRY_POINTS)
    def test_nullable_condition_is_pushed_with_sql_nulls(
        self, nullable_parquet, performance_mode, how
    ):
        def cond(d):
            return d["n"] > 2

        ds_result = _assign(DataStore.from_file(nullable_parquet), how, cond)
        assert '"n" > 2 AS "c"' in ds_result.to_sql(execution_format=True)

        expected = _assign(pd.read_parquet(nullable_parquet), how, cond)
        expected["c"] = pd.array([0, None, 1, 1], dtype="UInt8")
        assert_frame_equal(get_dataframe(ds_result), expected)

    def test_float_nan_comparison_gives_null(self, performance_mode):
        ds = DataStore(_frame())
        ds["c"] = ds["f"] > 2
        result = get_dataframe(ds)
        assert result["c"].dtype == "UInt8"
        assert result["c"].isna().tolist() == [False, True, False, False]
        assert result["c"].fillna(0).astype(int).tolist() == [0, 0, 1, 1]
