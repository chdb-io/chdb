"""
Tests for lazy metadata access on NON-pristine but SQL-pushdownable pipelines.

Extends the pristine-source optimization (test_pristine_metadata_optimization.py)
to filtered / projected pipelines: `.columns` resolves via a ``SELECT ... LIMIT 0``
schema probe, `.dtypes` via a ``LIMIT 1`` probe (LIMIT 0 degrades string columns to
``object``), and `.shape` / `.index` via ``COUNT(*)``, instead of materializing the
full result through ``_execute()``. Non-pushdownable pipelines, remote sources, and
custom/group indexes keep the full-execution fallback.

Mirrors the mock + SQL-capture style of test_pristine_metadata_optimization.py.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore


def _make_parquet(tmp_path):
    df = pd.DataFrame({
        "id": range(1000),
        "value": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
    })
    path = str(tmp_path / "events.parquet")
    df.to_parquet(path, index=False)
    return path, df


class TestNonPristinePushdownMetadata:
    """A filtered (non-pristine, SQL-pushdownable) pipeline must read metadata
    without materializing the result via _execute()."""

    @pytest.fixture
    def filtered(self, tmp_path):
        path, df = _make_parquet(tmp_path)
        ds = DataStore.from_file(path)
        ds2 = ds[ds["value"] > 0]                 # pushdownable filter -> non-pristine
        pd_result = df[df["value"] > 0]
        return ds2, pd_result

    def test_precondition_non_pristine_but_pushdownable(self, filtered):
        ds2, _ = filtered
        assert ds2._get_source_df_if_pristine() is None
        assert ds2._is_pristine_sql_source() is False
        assert ds2._can_sql_pushdown() is True

    def test_columns_uses_limit0_not_execute(self, filtered):
        ds2, pd_result = filtered
        ds2.connect()
        sqls = []
        original_execute = ds2._executor.execute

        def capture_sql(sql, *args, **kwargs):
            sqls.append(sql)
            return original_execute(sql, *args, **kwargs)

        with patch.object(ds2, "_execute", wraps=ds2._execute) as mock_exec, \
                patch.object(ds2._executor, "execute", side_effect=capture_sql):
            cols = ds2.columns
            mock_exec.assert_not_called()

        assert list(cols) == list(pd_result.columns)
        assert any("LIMIT 0" in s for s in sqls), f"expected a LIMIT 0 probe, got: {sqls}"

    def test_dtypes_uses_limit1_not_execute(self, filtered):
        ds2, pd_result = filtered
        ds2.connect()
        sqls = []
        original_execute = ds2._executor.execute

        def capture_sql(sql, *args, **kwargs):
            sqls.append(sql)
            return original_execute(sql, *args, **kwargs)

        with patch.object(ds2, "_execute", wraps=ds2._execute) as mock_exec, \
                patch.object(ds2._executor, "execute", side_effect=capture_sql):
            dtypes = ds2.dtypes
            mock_exec.assert_not_called()

        assert list(dtypes.index) == list(pd_result.columns)
        # .dtypes probes with LIMIT 1 (LIMIT 0 would degrade string columns to
        # object); see TestDtypesProbeLimit1.
        assert any("LIMIT 1" in s for s in sqls), f"expected a LIMIT 1 probe, got: {sqls}"

    def test_shape_uses_count_not_execute(self, filtered):
        ds2, pd_result = filtered
        with patch.object(ds2, "count_rows", wraps=ds2.count_rows) as mock_count, \
                patch.object(ds2, "_execute", wraps=ds2._execute) as mock_exec:
            shape = ds2.shape
            mock_count.assert_called()
            mock_exec.assert_not_called()
        assert shape == pd_result.shape

    def test_index_is_rangeindex_from_count_not_execute(self, filtered):
        ds2, pd_result = filtered
        with patch.object(ds2, "count_rows", wraps=ds2.count_rows) as mock_count, \
                patch.object(ds2, "_execute", wraps=ds2._execute) as mock_exec:
            idx = ds2.index
            mock_count.assert_called()
            mock_exec.assert_not_called()
        assert isinstance(idx, pd.RangeIndex)
        assert len(idx) == len(pd_result)

    def test_metadata_access_does_not_cache(self, filtered):
        ds2, _ = filtered
        assert ds2._cached_result is None
        _ = ds2.columns
        _ = ds2.dtypes
        _ = ds2.shape
        _ = ds2.index
        assert ds2._cached_result is None, "metadata access must not materialize the result"


class TestProjectionColumnsCorrectness:
    """A column-reducing projection is NOT reflected in the LIMIT 0 probe SQL
    (it's a post-SQL lazy op), so the probe's columns differ from the projected
    set and `.columns` falls back to execution. It must still report the
    projected columns -- regression guard for the lazy-metadata optimization."""

    def test_projection_columns_correct(self, tmp_path):
        path, df = _make_parquet(tmp_path)
        ds = DataStore.from_file(path)[["id", "category"]]
        pd_result = df[["id", "category"]]
        assert list(ds.columns) == list(pd_result.columns) == ["id", "category"]


class TestIndexFallback:
    """`.index` only takes the RangeIndex shortcut for default-indexed pipelines;
    a custom (set_index) index must fall back and stay correct."""

    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "id": [10, 20, 30, 40, 50],
            "score": [90.5, 85.0, 92.3, 78.0, 95.5],
        })

    def test_set_index_does_not_use_rangeindex_shortcut(self, tmp_path, df):
        path = str(tmp_path / "idx.parquet")
        df.to_parquet(path, index=False)
        ds = DataStore.from_file(path).set_index("id")
        pd_result = df.set_index("id")
        idx = ds.index
        # must reflect the real index values (10..50), not a degenerate RangeIndex(0..n)
        assert list(idx) == list(pd_result.index)


class TestOffsetRowCount:
    """An OFFSET-without-LIMIT pipeline (e.g. ``ds[k:]``) must report the
    POST-offset row count for ``.shape`` / ``len(.index)`` / ``count_rows()``.

    Regression guard: ``count_rows()`` only short-circuits to execution on LIMIT,
    and its flat / subquery COUNT paths drop OFFSET (``_build_count_subquery``
    clears ``_offset_value``), so an offset-only pipeline over-counts by the
    offset unless ``count_rows()`` falls back to execution. The optimization must
    match both pandas and full execution."""

    @pytest.fixture
    def offset_pipeline(self, tmp_path):
        path, df = _make_parquet(tmp_path)        # 1000 rows
        return path, df.iloc[5:]                   # OFFSET 5, no LIMIT -> 995 rows

    def test_count_rows_respects_offset(self, offset_pipeline):
        path, pd_result = offset_pipeline
        assert DataStore.from_file(path)[5:].count_rows() == len(pd_result) == 995

    def test_shape_respects_offset_matches_pandas_and_full(self, offset_pipeline):
        path, pd_result = offset_pipeline
        assert DataStore.from_file(path)[5:].shape == pd_result.shape  # (995, 3)
        # internal invariant: optimized shape == full-execution shape
        full = DataStore.from_file(path)[5:]._get_df()
        assert DataStore.from_file(path)[5:].shape == full.shape

    def test_index_length_respects_offset(self, offset_pipeline):
        path, pd_result = offset_pipeline
        ds_idx = DataStore.from_file(path)[5:].index
        full_idx = DataStore.from_file(path)[5:]._get_df().index
        assert len(ds_idx) == len(pd_result) == 995
        assert list(ds_idx) == list(full_idx)  # optimized index == full execution


class TestSortValuesIndex:
    """``sort_values()`` (ORDER BY) must NOT take the ``RangeIndex(0..n)`` shortcut:
    a sorted result preserves the original positional labels, which the shortcut
    destroys. ``.index`` must match both full execution and pandas.

    Regression guard for the lazy-metadata optimization: ``count_rows()`` strips
    ORDER BY for its flat count, so ``.index`` would otherwise return a degenerate
    ``RangeIndex`` with the wrong labels."""

    def test_sort_values_index_matches_full_and_pandas(self, tmp_path):
        df = pd.DataFrame({
            "age": [5, 3, 8, 1, 9, 2, 7, 0, 6, 4],
            "name": [f"n{i}" for i in range(10)],
        })
        path = str(tmp_path / "sort.parquet")
        df.to_parquet(path, index=False)
        pd_result = df.sort_values("age")

        opt_index = list(DataStore.from_file(path).sort_values("age").index)
        full_index = list(DataStore.from_file(path).sort_values("age")._get_df().index)

        # primary invariant: the optimization must not disagree with full execution
        assert opt_index == full_index, (opt_index, full_index)
        # and it must match pandas' order-preserved labels (not RangeIndex 0..n)
        assert opt_index == list(pd_result.index), (opt_index, list(pd_result.index))
        assert opt_index != list(range(len(df))), "must not be a degenerate RangeIndex"


class TestDtypesProbeLimit1:
    """`.dtypes` on a SQL-pushdownable pipeline probes with ``LIMIT 1`` (not
    ``LIMIT 0``). An empty (LIMIT 0) result degrades string columns to numpy
    ``object``; one row lets chDB report the real pandas dtype (e.g. StringDtype).

    The optimized dtypes MUST equal full execution (``_get_df().dtypes``) for
    every type, and the access must not materialize the full result. Regression
    guard for BUG#3 (string columns reported as ``object``)."""

    @pytest.fixture
    def typed(self, tmp_path):
        df = pd.DataFrame({
            "i":  pd.array([1, 2, 3, 4, 5], dtype="int64"),
            "f":  [1.5, 2.5, 3.5, 4.5, 5.5],
            "s":  ["alpha", "beta", "gamma", "delta", "eps"],
            "b":  [True, False, True, False, True],
            "dt": pd.to_datetime(
                ["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01", "2021-05-01"]
            ),
        })
        path = str(tmp_path / "typed.parquet")
        df.to_parquet(path, index=False)
        return path, df

    def _flt(self, path):
        return DataStore.from_file(path)[DataStore.from_file(path)["i"] >= 2]

    def test_filtered_dtypes_match_full_execution(self, typed):
        path, _ = typed
        opt = dict(self._flt(path).dtypes)
        full = dict(self._flt(path)._get_df().dtypes)
        assert opt == full, (opt, full)

    def test_string_column_is_not_object(self, typed):
        # BUG#3: a LIMIT 0 probe degrades the string column to object; LIMIT 1
        # reports the real dtype, matching full execution.
        path, _ = typed
        opt = dict(self._flt(path).dtypes)
        full = dict(self._flt(path)._get_df().dtypes)
        assert opt["s"] != object, f"string column degraded to object: {opt['s']}"
        assert opt["s"] == full["s"]

    def test_dtypes_uses_limit1_probe_not_execute(self, typed):
        path, _ = typed
        ds = self._flt(path)
        ds.connect()
        sqls = []
        original = ds._executor.execute
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec, \
                patch.object(
                    ds._executor, "execute",
                    side_effect=lambda s, *a, **k: (sqls.append(s), original(s, *a, **k))[1],
                ):
            dtypes = ds.dtypes
            mock_exec.assert_not_called()
        assert any("LIMIT 1" in s for s in sqls), f"expected a LIMIT 1 probe, got: {sqls}"
        assert dtypes["s"] != object
        assert ds._cached_result is None, "dtypes access must not materialize the result"

    def test_sort_then_dtypes_match_full(self, typed):
        path, _ = typed
        opt = dict(DataStore.from_file(path).sort_values("f").dtypes)
        full = dict(DataStore.from_file(path).sort_values("f")._get_df().dtypes)
        assert opt == full, (opt, full)
        assert opt["s"] != object

    def test_head_then_dtypes_match_full(self, typed):
        path, _ = typed
        opt = dict(DataStore.from_file(path).head(3).dtypes)
        full = dict(DataStore.from_file(path).head(3)._get_df().dtypes)
        assert opt == full, (opt, full)
        assert opt["s"] != object

    def test_empty_result_dtypes_match_full(self, typed):
        path, _ = typed
        flt = lambda: DataStore.from_file(path)[DataStore.from_file(path)["i"] > 10 ** 9]
        opt = dict(flt().dtypes)
        full = dict(flt()._get_df().dtypes)
        # genuinely empty -> chDB degrades strings to object in BOTH paths; the
        # invariant is still optimized == full execution.
        assert opt == full, (opt, full)


class TestDtypesNullableInteger:
    """A nullable integer column's pandas dtype is null-PRESENCE dependent: chDB
    yields numpy ``int64`` when the result has no nulls and pandas ``Int64`` when
    it does. A bounded probe can't see nulls elsewhere, so the ``.dtypes`` path
    runs a COUNT(nulls) check: no nulls -> trust the LIMIT 1 probe (optimization
    kept); nulls present -> fall back to full execution. Either way ``.dtypes``
    must equal ``_get_df().dtypes``."""

    def _write(self, tmp_path, values):
        df = pd.DataFrame({
            "ni": pd.array(values, dtype="Int64"),
            "s": [f"r{i}" for i in range(len(values))],
        })
        path = str(tmp_path / "ni.parquet")
        df.to_parquet(path, index=False)
        return path

    def test_nullable_int_with_nulls_falls_back_and_matches_full(self, tmp_path):
        path = self._write(tmp_path, [1, None, 3, None, 5])
        # filter on the OTHER column so the null rows of `ni` are kept
        flt = lambda: DataStore.from_file(path)[DataStore.from_file(path)["s"] != "zzz"]
        opt = dict(flt().dtypes)
        full = dict(flt()._get_df().dtypes)
        assert opt == full, (opt, full)
        assert str(opt["ni"]) == "Int64", "nulls present -> nullable extension dtype"

    def test_nullable_int_without_nulls_keeps_optimization(self, tmp_path):
        path = self._write(tmp_path, [1, None, 3, None, 5])
        # filter keeps only non-null rows of `ni`
        full = dict(
            DataStore.from_file(path)[DataStore.from_file(path)["ni"] >= 3]._get_df().dtypes
        )
        ds = DataStore.from_file(path)[DataStore.from_file(path)["ni"] >= 3]
        with patch.object(ds, "_execute", wraps=ds._execute) as mock_exec:
            opt = dict(ds.dtypes)
            # COUNT(nulls)==0 -> probe trusted, no full materialization
            mock_exec.assert_not_called()
        assert opt == full, (opt, full)
        assert str(opt["ni"]) == "int64", "no nulls -> plain numpy dtype"


class TestDtypesColumnAlteringFallback:
    """Column-altering ops (astype / rename / drop / assign) are applied post-SQL
    and are not in the probe; ``.dtypes`` must fall back to execution and still
    report correct per-column dtypes."""

    @pytest.fixture
    def base(self, tmp_path):
        df = pd.DataFrame({
            "id": pd.array([1, 2, 3], dtype="int64"),
            "val": [10.0, 20.0, 30.0],
            "name": ["x", "y", "z"],
        })
        path = str(tmp_path / "base.parquet")
        df.to_parquet(path, index=False)
        return path, df

    def test_astype_dtypes_match_full(self, base):
        path, _ = base
        opt = dict(DataStore.from_file(path).astype({"id": "float64"}).dtypes)
        full = dict(DataStore.from_file(path).astype({"id": "float64"})._get_df().dtypes)
        assert opt == full, (opt, full)
        assert str(opt["id"]) == "float64"

    def test_rename_dtypes_match_full(self, base):
        path, _ = base
        opt = dict(DataStore.from_file(path).rename(columns={"val": "value"}).dtypes)
        full = dict(
            DataStore.from_file(path).rename(columns={"val": "value"})._get_df().dtypes
        )
        assert opt == full, (opt, full)
        assert "value" in opt and "val" not in opt

    def test_drop_dtypes_match_full(self, base):
        path, _ = base
        opt = dict(DataStore.from_file(path).drop(columns=["name"]).dtypes)
        full = dict(DataStore.from_file(path).drop(columns=["name"])._get_df().dtypes)
        assert opt == full, (opt, full)
        assert "name" not in opt
