"""
groupby() on a categorical key with SQL expression aggregates
(``agg(total=col('v').sum())``) matches pandas: groups in category order,
categories without rows included with observed=False (0 for sum, NaN for
mean), and a CategoricalIndex. Each case runs on a DataFrame source and on
a file source whose key is made categorical in the pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.expressions import col
from tests.test_utils import assert_datastore_equals_pandas

CATEGORIES = ["c", "b", "a", "d"]
FRAME = pd.DataFrame(
    {
        "g": pd.Categorical(["b", "a", "b", "a", "b"], categories=CATEGORIES),
        "v": [1.0, 2.0, 3.0, 4.0, 5.0],
        "w": [10, 20, 30, 40, 50],
    }
)


@pytest.fixture(params=["dataframe", "file"])
def source(request, tmp_path):
    if request.param == "dataframe":
        return lambda: DataStore(FRAME.copy())
    path = str(tmp_path / "t.parquet")
    FRAME.assign(g=FRAME["g"].astype(str)).to_parquet(path)
    return lambda: DataStore.from_file(path).astype({"g": pd.CategoricalDtype(CATEGORIES)})


def _grouped(df, observed):
    return df.groupby("g", observed=observed)


# (DataStore aggregate expression, pandas computation on the grouped frame)
REDUCTIONS = [
    pytest.param(lambda: col("v").sum(), lambda g: g["v"].sum(), id="sum"),
    pytest.param(lambda: col("v").mean(), lambda g: g["v"].mean(), id="mean"),
    pytest.param(lambda: col("v").min(), lambda g: g["v"].min(), id="min"),
    pytest.param(lambda: col("v").max(), lambda g: g["v"].max(), id="max"),
    pytest.param(lambda: col("w").count(), lambda g: g.size(), id="count"),
    pytest.param(lambda: (col("v") * 2).sum(), lambda g: g["v"].sum() * 2, id="sum_of_expression"),
    pytest.param(
        lambda: col("v").sum() / col("w").count(), lambda g: g["v"].sum() / g.size(), id="ratio"
    ),
    pytest.param(lambda: col("v").max() - col("v").min(), lambda g: g["v"].max() - g["v"].min(), id="range"),
]


@pytest.mark.parametrize("observed", [True, False])
@pytest.mark.parametrize("make_expr, pandas_value", REDUCTIONS)
def test_expression_aggregate(source, observed, make_expr, pandas_value):
    ds_result = source().groupby("g", observed=observed).agg(total=make_expr())
    pd_result = pandas_value(_grouped(FRAME, observed)).rename("total").to_frame()
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)
    assert isinstance(ds_result.to_pandas().index, pd.CategoricalIndex)


@pytest.mark.parametrize("observed", [True, False])
def test_several_outputs(source, observed):
    ds_result = source().groupby("g", observed=observed).agg(
        total=col("v").sum(), top=col("w").max()
    )
    g = _grouped(FRAME, observed)
    pd_result = pd.DataFrame({"total": g["v"].sum(), "top": g["w"].max()})
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("observed", [True, False])
def test_as_index_false(source, observed):
    ds_result = source().groupby("g", observed=observed, as_index=False).agg(
        total=col("v").sum()
    )
    pd_result = (
        _grouped(FRAME, observed)["v"].sum().rename("total").reset_index()
    )
    assert_datastore_equals_pandas(ds_result, pd_result)


@pytest.mark.parametrize("observed", [True, False])
def test_aggregate_inside_case_when(source, observed):
    ds = source()
    ds_result = ds.groupby("g", observed=observed).agg(
        flag=ds.when(col("v").sum() > 6, 1).otherwise(0)
    )
    sums = _grouped(FRAME, observed)["v"].sum()
    flags = np.where(sums > 6, 1, 0)
    pd_result = pd.DataFrame({"flag": flags}, index=sums.index)
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("observed", [True, False])
def test_group_key_in_output(source, observed):
    ds_result = source().groupby("g", observed=observed).agg(key=col("g"), total=col("v").sum())
    sums = _grouped(FRAME, observed)["v"].sum()
    pd_result = pd.DataFrame({"key": sums.index.to_series(index=sums.index), "total": sums})
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("observed", [True, False])
def test_as_index_false_output_named_like_key(source, observed):
    # pandas keeps the output and drops the key
    ds_result = source().groupby("g", observed=observed, as_index=False).agg(g=col("v").sum())
    pd_result = FRAME.groupby("g", observed=observed, as_index=False).agg(g=("v", "sum"))
    assert_datastore_equals_pandas(ds_result, pd_result)


def test_key_named_like_value_column(tmp_path):
    frame = FRAME.rename(columns={"g": "__value__"})
    ds_result = DataStore(frame.copy()).groupby("__value__", observed=True).agg(total=col("v").sum())
    pd_result = frame.groupby("__value__", observed=True).agg(total=("v", "sum"))
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)


@pytest.mark.parametrize("observed", [True, False])
def test_nullable_integer_values(observed):
    frame = FRAME.assign(n=pd.array([1, None, 3, 4, None], dtype="Int64"))
    ds_result = DataStore(frame.copy()).groupby("g", observed=observed).agg(total=col("n").sum())
    pd_result = frame.groupby("g", observed=observed)["n"].sum().rename("total").to_frame()
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)
