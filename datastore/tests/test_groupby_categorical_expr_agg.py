"""
groupby() on a categorical key with SQL expression aggregates
(``agg(total=col('v').sum())``) matches pandas: groups in category order,
categories without rows included with observed=False (0 for sum, NaN for
mean), and a CategoricalIndex. Each case runs on a DataFrame source and on
a file source whose key is made categorical in the pipeline.
"""

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
