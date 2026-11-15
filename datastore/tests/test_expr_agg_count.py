"""
``col(x).count()`` in an expression aggregate counts the non-missing values
of ``x``, as pandas ``count()`` does, not every row. Runs on DataFrame,
Parquet and CSV sources, with categorical keys (aggregates in pandas) and
plain keys (in SQL).
"""

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from datastore.expressions import col
from tests.test_utils import assert_datastore_equals_pandas

KEYS = ["a", "a", "b", "b", "c"]
VALUES = {
    "float": [1.0, np.nan, 3.0, 4.0, np.nan],
    "nullable_int": pd.array([1, None, 3, 4, None], dtype="Int64"),
    "str": ["x", None, "y", "z", None],
}


@pytest.fixture(params=["dataframe", "parquet", "csv"])
def kind(request):
    return request.param


@pytest.mark.parametrize("categorical", [True, False], ids=["categorical", "plain"])
@pytest.mark.parametrize("column", list(VALUES))
def test_count_skips_missing_values(kind, categorical, column, tmp_path):
    df = pd.DataFrame({"k": KEYS, "w": VALUES[column], "v": [1, 2, 3, 4, 5]})
    if kind == "dataframe":
        ds = DataStore(df.copy())
    else:
        path = str(tmp_path / f"t.{kind}")
        df.to_parquet(path) if kind == "parquet" else df.to_csv(path, index=False)
        ds = DataStore.from_file(path)
    if categorical:
        df = df.astype({"k": "category"})
        ds = ds.astype({"k": "category"})

    ds_result = ds.groupby("k", observed=True).agg(
        c=col("w").count(), share=col("w").count() / col("v").count()
    )

    grouped = df.groupby("k", observed=True)
    pd_result = pd.DataFrame(
        {"c": grouped["w"].count(), "share": grouped["w"].count() / grouped["v"].count()}
    )
    assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)
