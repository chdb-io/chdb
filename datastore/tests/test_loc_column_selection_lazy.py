"""
loc[:, columns] stays lazy (issue #716).

``ds.loc[:, cols]`` is the same column selection as ``ds[cols]``, so it must
return a lazy DataStore (or a ColumnExpr for a single label) instead of
executing the pipeline and handing back a pandas DataFrame.
"""

import pandas as pd
import pytest

import datastore.core as core
from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal


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
        {"region": ["n", "s", "n"], "price": [1, 2, 4], "qty": [4, 5, 6]}
    )


class TestLocColumnSelectionLazy:
    def test_issue_chain(self, execute_calls):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, ["region", "price"]]
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        pd_result = pd_df.loc[:, ["region", "price"]]
        assert_datastore_equals_pandas(ds_result, pd_result)
        pd.testing.assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_chain_after_selection(self, execute_calls):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = (
            ds.loc[:, ["region", "price"]]
            .sort_values("price", ascending=False)
            .head(2)
        )
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        pd_result = (
            pd_df.loc[:, ["region", "price"]]
            .sort_values("price", ascending=False)
            .head(2)
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_single_label_is_lazy_series(self, execute_calls):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, "price"]
        assert not isinstance(ds_result, (DataStore, pd.Series))
        assert len(execute_calls) == 0

        assert_series_equal(ds_result, pd_df.loc[:, "price"])

    def test_index_of_labels(self, execute_calls):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, pd_df.columns[1:]]
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        assert_datastore_equals_pandas(ds_result, pd_df.loc[:, pd_df.columns[1:]])

    def test_empty_list(self, execute_calls):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, []]
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        pd_result = pd_df.loc[:, []]
        assert list(ds_result.columns) == []
        pd.testing.assert_index_equal(ds_result.to_pandas().index, pd_result.index)

    def test_custom_index_is_kept(self, execute_calls):
        pd_df = _frame()
        pd_df.index = [10, 20, 30]
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, ["qty", "region"]]
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        assert_datastore_equals_pandas(
            ds_result, pd_df.loc[:, ["qty", "region"]], check_index=True
        )

    def test_file_source(self, tmp_path, execute_calls):
        path = str(tmp_path / "sales.parquet")
        _frame().to_parquet(path)
        ds = DataStore.from_file(path)

        ds_result = ds.loc[:, ["qty", "region"]]
        assert isinstance(ds_result, DataStore)
        assert len(execute_calls) == 0

        pd_result = pd.read_parquet(path).loc[:, ["qty", "region"]]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_boolean_column_mask_matches_pandas(self):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, [True, False, True]]
        pd_result = pd_df.loc[:, [True, False, True]]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_duplicate_labels_match_pandas(self):
        pd_df = _frame()
        ds = DataStore(pd_df)

        ds_result = ds.loc[:, ["price", "price"]]
        pd_result = pd_df.loc[:, ["price", "price"]]
        assert_datastore_equals_pandas(ds_result, pd_result)
