"""
Comparing two executed values (aggregation results, method results) gives
the pandas result: a bool for scalars, an element-wise Series otherwise.
"""

import operator

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_series_equal, get_series

OPERATORS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]


def _frames():
    return pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": [3, 2, 1]})


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize(
    "reduce",
    [
        pytest.param(lambda s: s.agg("sum"), id="agg_sum"),
        pytest.param(lambda s: s.aggregate("max"), id="aggregate_max"),
    ],
)
def test_scalar_results_compare_to_bool(op, reduce):
    left, right = _frames()
    pd_result = op(reduce(left["a"]), reduce(right["a"]))
    ds_left, ds_right = DataStore(left.copy()), DataStore(right.copy())
    ds_result = op(reduce(ds_left["a"]), reduce(ds_right["a"]))
    assert bool(ds_result) == bool(pd_result)


@pytest.mark.parametrize("op", OPERATORS)
def test_series_results_compare_element_wise(op):
    left, right = _frames()
    pd_result = op(left["a"].cumsum(), right["a"].cumsum())
    ds_left, ds_right = DataStore(left.copy()), DataStore(right.copy())
    ds_result = op(ds_left["a"].cumsum(), ds_right["a"].cumsum())
    assert_series_equal(get_series(ds_result), pd_result)
