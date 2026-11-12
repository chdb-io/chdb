"""
A group key that is explicitly selected after groupby() is aggregated like
any other selected column, as pandas does: ``groupby('g')['g'].sum()`` sums
``g`` within each group. Every case runs on a DataFrame source and on a file
source, under the default and the pandas execution engine.
"""

import itertools

import pandas as pd
import pytest

from datastore import DataStore
from datastore.config import ExecutionEngine, set_execution_engine
from tests.test_utils import assert_datastore_equals_pandas

_counter = itertools.count()

DATA = {
    "g": [1, 2, 1, 3, 2, 1],
    "h": [5, 5, 6, 6, 5, 5],
    "v": [10, 20, 30, 40, 50, 60],
}


@pytest.fixture(params=["auto", "pandas"])
def engine(request):
    set_execution_engine(
        ExecutionEngine.PANDAS if request.param == "pandas" else ExecutionEngine.AUTO
    )
    yield
    set_execution_engine(ExecutionEngine.AUTO)


@pytest.fixture(params=["dataframe", "file"])
def source(request, tmp_path, engine):
    pdf = pd.DataFrame(DATA)
    if request.param == "dataframe":
        return pdf, DataStore(pdf.copy())
    path = str(tmp_path / f"t{next(_counter)}.parquet")
    pdf.to_parquet(path)
    return pdf, DataStore.from_file(path)


class TestSeriesOfSelectedKey:
    @pytest.mark.parametrize("func", ["sum", "count", "mean", "min", "max"])
    def test_aggregate_selected_key(self, source, func):
        pdf, ds = source
        pd_result = getattr(pdf.groupby("g")["g"], func)()
        ds_result = getattr(ds.groupby("g")["g"], func)()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_selected_key_of_two_keys(self, source):
        pdf, ds = source
        pd_result = pdf.groupby(["g", "h"])["g"].sum()
        ds_result = ds.groupby(["g", "h"])["g"].sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_selected_key_as_index_false(self, source):
        pdf, ds = source
        pd_result = pdf.groupby("g", as_index=False)["g"].sum()
        ds_result = ds.groupby("g", as_index=False)["g"].sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_selected_key_after_filter(self, source):
        pdf, ds = source
        pd_result = pdf[pdf["v"] > 10].groupby("g")["g"].sum()
        ds_result = ds[ds["v"] > 10].groupby("g")["g"].sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_other_column_unchanged(self, source):
        pdf, ds = source
        pd_result = pdf.groupby("g")["v"].sum()
        ds_result = ds.groupby("g")["v"].sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


class TestFrameWithSelectedKey:
    def test_sum_selected_key_and_column(self, source):
        pdf, ds = source
        pd_result = pdf.groupby("g")[["g", "v"]].sum()
        ds_result = ds.groupby("g")[["g", "v"]].sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_sum_selected_key_and_column_as_index_false(self, source):
        pdf, ds = source
        pd_result = pdf.groupby("g", as_index=False)[["g", "v"]].sum()
        ds_result = ds.groupby("g", as_index=False)[["g", "v"]].sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_mean_selected_second_key(self, source):
        pdf, ds = source
        pd_result = pdf.groupby(["g", "h"])[["h", "v"]].mean()
        ds_result = ds.groupby(["g", "h"])[["h", "v"]].mean()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_sum_selected_key_then_sort(self, source):
        pdf, ds = source
        pd_result = pdf.groupby("g")[["g", "v"]].sum().sort_values("v")
        ds_result = ds.groupby("g")[["g", "v"]].sum().sort_values("v")
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_unselected_keys_still_excluded(self, source):
        pdf, ds = source
        pd_result = pdf.groupby("g").sum()
        ds_result = ds.groupby("g").sum()
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)
