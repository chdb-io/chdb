"""
A DataFrame with a repeated column label: operations on another, unique
column match pandas instead of failing on the repeated label (its lookup
gives a DataFrame, and chDB can't read repeated labels).
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.expression_evaluator import ExpressionEvaluator
from datastore.expressions import Field
from datastore.functions import Function
from tests.test_utils import assert_series_equal, get_series


def _frame():
    df = pd.DataFrame([[1, 2, 3, "x"], [4, 5, None, "y"], [7, 8, 9, "x"]], columns=["A", "A", "B", "g"])
    df["B"] = df["B"].astype("Int64")
    return df


OPERATIONS = {
    "abs": lambda s: s["B"].abs(),
    "mask": lambda s: s["B"].mask(s["B"] > 3, 0),
    "where": lambda s: s["B"].where(s["B"] > 3, 0),
    "cumsum": lambda s: s["B"].cumsum(),
    "shift": lambda s: s["B"].shift(1),
    "rank": lambda s: s["B"].rank(),
    "groupby_sum": lambda s: s.groupby("g")["B"].sum(),
}


@pytest.mark.parametrize("name", sorted(OPERATIONS))
def test_operation_on_unique_column(name):
    result = OPERATIONS[name](DataStore(_frame()))
    expected = OPERATIONS[name](_frame())
    assert_series_equal(get_series(result), expected)


def test_evaluator_function_on_unique_column():
    df = _frame()
    result = ExpressionEvaluator(df, None).evaluate(Function("abs", Field("B")))
    assert_series_equal(result, df["B"].abs(), check_names=False)
