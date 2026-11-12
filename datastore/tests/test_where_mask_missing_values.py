"""
where()/mask() follow pandas where the condition is missing: a pd.NA in the
mask is replaced by both methods, and a comparison with NaN is False (True
for !=) as pandas computes it. Series and DataFrame methods, displayed or
assigned, on DataFrame and Parquet sources.
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.config import CompatMode, ExecutionEngine, set_compat_mode, set_execution_engine
from datastore.core import _keeps_read_columns, _may_read, _reads_within
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


def _frame():
    return pd.DataFrame(
        {
            "v": [1, 2, 3, 4],
            "w": [5, 6, 7, 8],
            "m": pd.array([True, None, False, True], dtype="boolean"),
            "n": pd.array([1, None, 3, 4], dtype="Int64"),
            "f": [1.0, np.nan, 3.0, 4.0],
        }
    )


@pytest.fixture(params=["dataframe", "parquet"])
def make_ds(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(_frame())
    path = str(tmp_path / "frame.parquet")
    _frame().to_parquet(path)
    return lambda: DataStore.from_file(path)


CONDITIONS = {
    "boolean_column": lambda s: s["m"],
    "int64_comparison": lambda s: s["n"] > 1,
    "float_comparison": lambda s: s["f"] > 1,
    "float_not_equal": lambda s: s["f"] != 3,
    "int64_and_int": lambda s: (s["n"] > 1) & (s["v"] > 1),
    "int64_or_float": lambda s: (s["n"] > 3) | (s["f"] > 2),
    "isna": lambda s: s["f"].isna(),
}
SERIES_OPS = {
    "where": lambda s, c: s["v"].where(c, 0),
    "mask": lambda s, c: s["v"].mask(c, 0),
    "mask_na": lambda s, c: s["v"].mask(c),
}
FRAME_OPS = {
    "where": lambda s, c: s[["v", "w"]].where(c, 0, axis=0),
    "mask": lambda s, c: s[["v", "w"]].mask(c, 0, axis=0),
}


@pytest.mark.parametrize("op", sorted(SERIES_OPS))
@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_series_method(make_ds, cond, op):
    d = make_ds()
    result = SERIES_OPS[op](d, CONDITIONS[cond](d))
    p = _frame()
    expected = SERIES_OPS[op](p, CONDITIONS[cond](p))
    assert_series_equal(get_series(result), expected)


@pytest.mark.parametrize("op", sorted(SERIES_OPS))
@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_series_method_assigned(make_ds, cond, op):
    d = make_ds()
    d["r"] = SERIES_OPS[op](d, CONDITIONS[cond](d))
    p = _frame()
    p["r"] = SERIES_OPS[op](p, CONDITIONS[cond](p))
    assert_datastore_equals_pandas(d, p)


@pytest.mark.parametrize("op", sorted(FRAME_OPS))
@pytest.mark.parametrize("cond", sorted(CONDITIONS))
def test_frame_method(make_ds, cond, op):
    d = make_ds()
    result = FRAME_OPS[op](d, CONDITIONS[cond](d))
    p = _frame()
    expected = FRAME_OPS[op](p, CONDITIONS[cond](p))
    assert_datastore_equals_pandas(result, expected)


FULL_FRAME_OPS = {
    "where": lambda s, c: s.where(c, 0),
    "mask": lambda s, c: s.mask(c, 0),
}


@pytest.mark.parametrize("op", sorted(FULL_FRAME_OPS))
@pytest.mark.parametrize("cond", ["int64_comparison", "float_comparison", "float_not_equal"])
def test_whole_frame_method(make_ds, cond, op, tmp_path):
    def numeric(s):
        return s[["v", "w", "n", "f"]]

    d = make_ds()
    result = FULL_FRAME_OPS[op](numeric(d), CONDITIONS[cond](d))
    p = _frame()
    expected = FULL_FRAME_OPS[op](numeric(p), CONDITIONS[cond](p))
    assert_datastore_equals_pandas(result, expected)


@pytest.fixture
def performance_mode():
    set_compat_mode(CompatMode.PERFORMANCE)
    yield
    set_compat_mode(CompatMode.PANDAS)
    set_execution_engine(ExecutionEngine.AUTO)


def test_performance_mode_keeps_sql_null_semantics(performance_mode):
    """Performance mode computes mask() in SQL as is: a NULL condition keeps the value."""
    d = DataStore(_frame())
    result = d["v"].mask(d["n"] > 1, 0)
    assert result._exec_mode == "expr"
    assert result._expr.to_sql() == 'if("n" > 1,0,"v")'
    assert list(get_series(result)) == [1, 2, 0, 0]


class TestColumnsReadHelpers:
    """Unknown columns (None) read as 'not only these' and 'maybe one of these'."""

    def test_reads_within(self):
        assert _reads_within({"a"}, ["a", "b"])
        assert not _reads_within({"a", "c"}, ["a", "b"])
        assert not _reads_within(None, ["a", "b"])

    def test_may_read(self):
        assert _may_read({"a"}, {"a"})
        assert not _may_read({"a"}, {"b"})
        assert _may_read(None, {"b"})
        assert not _may_read(None, set())

    def test_keeps_read_columns_unknown(self):
        assert not _keeps_read_columns(("assign", "a"), None, rowwise=True)
        assert not _keeps_read_columns(("select", {"a"}, True, {"b"}), None, rowwise=True)
        assert not _keeps_read_columns(("select", {"a"}, False, set()), None, rowwise=True)
        assert _keeps_read_columns(("rows",), None, rowwise=True)
        assert not _keeps_read_columns(("rows",), None, rowwise=False)
