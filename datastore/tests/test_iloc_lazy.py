"""
iloc[] row and column selection stays lazy (issue #769).

``ds.iloc[rows]``, ``ds.iloc[:, cols]`` and ``ds.iloc[rows, cols]`` select a
DataFrame, so they must return a lazy DataStore instead of executing the
pipeline and handing back a pandas DataFrame. Single rows and scalars are
still pandas objects.
"""

import pandas as pd
import pytest

import datastore.core as core
from datastore import DataStore
from tests.test_utils import assert_series_equal


@pytest.fixture
def execute_calls(monkeypatch):
    calls = []
    original = core.DataStore._execute

    def counting_execute(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(core.DataStore, "_execute", counting_execute)
    return calls


def _frame():
    return pd.DataFrame(
        {
            "a": [5, 2, 3, 4, 1, 6],
            "b": ["u", "v", "w", "x", "y", "z"],
            "c": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        }
    )


@pytest.fixture(params=["DataFrame", "file"])
def source(request, tmp_path):
    pd_df = _frame()
    if request.param == "DataFrame":
        return pd_df, lambda: DataStore(pd_df)
    path = str(tmp_path / "t.parquet")
    pd_df.to_parquet(path)
    return pd_df, lambda: DataStore.from_file(path)


def _assert_frame(ds_result, pd_result):
    assert isinstance(ds_result, DataStore)
    pd.testing.assert_frame_equal(ds_result.to_pandas(), pd_result)


class TestILocLazy:
    @pytest.mark.parametrize(
        "key",
        [slice(1, 3), (slice(None), [0]), [0, 2]],
        ids=["rows_slice", "column_list", "row_list"],
    )
    def test_issue_chain(self, source, execute_calls, key):
        pd_df, make = source
        ds = make()

        ds_result = ds.iloc[key]
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        _assert_frame(ds_result, pd_df.iloc[key])

    @pytest.mark.parametrize(
        "key",
        [
            slice(-2, None),
            slice(None, None, -2),
            slice(4, 1),
            [True, False, True, False, False, True],
            (slice(1, 4), [2, 0]),
            ([3, 0], slice(1, None)),
            (slice(None), slice(None, None, 2)),
        ],
        ids=[
            "negative_start",
            "negative_step",
            "empty_slice",
            "boolean_mask",
            "rows_and_columns",
            "row_list_column_slice",
            "column_step",
        ],
    )
    def test_keys_match_pandas(self, source, execute_calls, key):
        pd_df, make = source

        ds_result = make().iloc[key]
        assert len(execute_calls) == 0

        _assert_frame(ds_result, pd_df.iloc[key])

    def test_chain_after_iloc(self, source, execute_calls):
        pd_df, make = source

        ds_result = make().iloc[[0, 2, 3, 5], [0, 2]].sort_values("a", ascending=False).head(2)
        assert len(execute_calls) == 0

        pd_result = pd_df.iloc[[0, 2, 3, 5], [0, 2]].sort_values("a", ascending=False).head(2)
        _assert_frame(ds_result, pd_result)

    def test_head_keeps_row_labels(self, source):
        pd_df, make = source

        _assert_frame(make().iloc[1:5].head(2), pd_df.iloc[1:5].head(2))
        _assert_frame(make().iloc[::2].head(2), pd_df.iloc[::2].head(2))

    def test_iloc_after_sort_and_filter(self, execute_calls):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds[ds["a"] > 1].sort_values("c", ascending=False).iloc[1:3]
        assert len(execute_calls) == 0

        pd_result = pd_df[pd_df["a"] > 1].sort_values("c", ascending=False).iloc[1:3]
        _assert_frame(ds_result, pd_result)

    def test_custom_index_is_kept(self, execute_calls):
        pd_df = _frame()
        pd_df.index = [10, 20, 30, 40, 50, 60]
        ds = DataStore(pd_df)

        ds_result = ds.iloc[[4, 1], [1, 2]]
        assert len(execute_calls) == 0

        _assert_frame(ds_result, pd_df.iloc[[4, 1], [1, 2]])

    def test_single_column_is_lazy(self, source, execute_calls):
        pd_df, make = source
        ds = make()

        ds_result = ds.iloc[:, 2]
        assert not isinstance(ds_result, (DataStore, pd.Series))
        assert len(execute_calls) == 0
        assert_series_equal(ds_result, pd_df.iloc[:, 2])

        assert_series_equal(make().iloc[1:4, 0], pd_df.iloc[1:4, 0])

    def test_single_row_and_scalar(self, source):
        pd_df, make = source

        ds_row = make().iloc[2]
        assert isinstance(ds_row, pd.Series)
        assert_series_equal(ds_row, pd_df.iloc[2])
        assert make().iloc[-1, 1] == pd_df.iloc[-1, 1]

    def test_duplicate_column_positions_match_pandas(self):
        pd_df = _frame()

        # ds[cols] can't repeat a column, so this one runs on pandas
        ds_result = DataStore(pd_df).iloc[:, [0, 0]]
        pd.testing.assert_frame_equal(ds_result, pd_df.iloc[:, [0, 0]])

    def test_out_of_bounds_raises(self):
        ds = DataStore(_frame())

        with pytest.raises(IndexError):
            ds.iloc[:, [5]]
        with pytest.raises(IndexError):
            ds.iloc[[10]].to_pandas()
