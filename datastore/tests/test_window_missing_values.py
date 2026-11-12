"""
Window methods (cumulative, shift/diff, rank, grouped transforms) over a
nullable column keep pandas' missing-value semantics, and window clauses
take column expressions without evaluating them.
"""

import pandas as pd
import pytest

import datastore.core as core
from datastore import DataStore, F
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


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
            "g": ["x", "y", "x", "y", "x", "y"],
            "v": pd.array([1, None, 3, 4, None, 2], dtype="Int64"),
        }
    )


@pytest.fixture(params=["dataframe", "parquet"])
def make_ds(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(_frame())
    path = str(tmp_path / "nullable.parquet")
    _frame().to_parquet(path)
    return lambda: DataStore.from_file(path)


# pandas 2 ranks a masked array by the values under its mask, so NA rows get
# arbitrary ranks; pandas 3 keeps them NA.
PANDAS2_MASKED_RANK = pytest.mark.skipif(
    int(pd.__version__.split(".")[0]) < 3, reason="pandas 2 ranks the values under the NA mask"
)

VALUES = [
    pytest.param(lambda d: d.groupby("g")["v"].cumsum(), id="groupby_cumsum"),
    pytest.param(lambda d: d.groupby("g")["v"].cummax(), id="groupby_cummax"),
    pytest.param(lambda d: d["v"].cumsum(), id="cumsum"),
    pytest.param(lambda d: d["v"].cummax(), id="cummax"),
    pytest.param(lambda d: d["v"].shift(), id="shift"),
    pytest.param(lambda d: d["v"].diff(), id="diff"),
    pytest.param(lambda d: d["v"].rank(method="min"), id="rank_min", marks=PANDAS2_MASKED_RANK),
    pytest.param(
        lambda d: d["v"].rank(method="dense", ascending=False),
        id="rank_dense_desc",
        marks=PANDAS2_MASKED_RANK,
    ),
    pytest.param(lambda d: d["v"].rank(method="first"), id="rank_first", marks=PANDAS2_MASKED_RANK),
    pytest.param(lambda d: d.groupby("g")["v"].transform("sum"), id="transform_sum"),
    pytest.param(lambda d: d.groupby("g")["v"].transform("mean"), id="transform_mean"),
    pytest.param(lambda d: d.groupby("g")["v"].transform("count"), id="transform_count"),
]


@pytest.mark.parametrize("make_value", VALUES)
def test_value(make_ds, make_value):
    assert_series_equal(get_series(make_value(make_ds())), make_value(_frame()), check_names=False)


@pytest.mark.parametrize("make_value", VALUES)
def test_assigned(make_ds, make_value):
    def chain(d):
        return d.assign(w=make_value(d))

    assert_datastore_equals_pandas(chain(make_ds()), chain(_frame()))


@pytest.mark.parametrize("make_value", [v for v in VALUES if not v.id.startswith(("groupby", "transform"))])
def test_building_does_not_execute(make_ds, make_value, execute_calls):
    d = make_ds()
    d.assign(w=make_value(d))
    assert len(execute_calls) == 0


class TestWindowClauseColumnExpressions:
    def test_order_and_partition_by_column_expressions(self, execute_calls):
        d = DataStore({"value": [3, 1, 2], "g": ["a", "a", "b"]})
        window = F.row_number().over(partition_by=d["g"], order_by=[d["value"]])
        assert len(execute_calls) == 0
        assert window.to_sql() == 'row_number() OVER (PARTITION BY "g" ORDER BY "value")'

    def test_assign_window_ordered_by_column_expression(self, execute_calls):
        d = DataStore({"value": [3, 1, 2]})
        ds_result = d.assign(rn=F.row_number().over(order_by=d["value"]))
        assert len(execute_calls) == 0
        pdf = pd.DataFrame({"value": [3, 1, 2]})
        pd_result = pdf.assign(rn=pdf["value"].rank(method="first").astype("uint64"))
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_column_value_without_sql_expression_is_rejected(self, execute_calls):
        from datastore.exceptions import ValidationError

        d = DataStore({"value": [3, 1, 2]})
        with pytest.raises(ValidationError):
            F.row_number().over(order_by=d["value"].cumsum())
        assert len(execute_calls) == 0
