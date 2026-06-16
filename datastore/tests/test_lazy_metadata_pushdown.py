"""
Tests for lazy metadata access on NON-pristine but SQL-pushdownable pipelines.

Extends the pristine-source optimization (test_pristine_metadata_optimization.py)
to filtered / projected pipelines: `.columns` / `.dtypes` must resolve via a
``SELECT ... LIMIT 0`` schema probe, and `.shape` / `.index` via ``COUNT(*)``,
instead of materializing the full result through ``_execute()``. Non-pushdownable
pipelines and custom/group indexes keep the full-execution fallback.

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

    def test_dtypes_uses_limit0_not_execute(self, filtered):
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
        assert any("LIMIT 0" in s for s in sqls), f"expected a LIMIT 0 probe, got: {sqls}"

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
