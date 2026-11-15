"""
ColumnExpr and LazySeries are immutable: every method that takes ``inplace``
rejects ``inplace=True`` instead of returning a new value the caller drops,
and the column is left as it was.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.exceptions import ImmutableError
from datastore.lazy_result import LazySeries
from tests.test_utils import assert_datastore_equals_pandas, assert_series_equal, get_series


def _frame():
    return pd.DataFrame({"a": [3.0, None, 1.0, 3.0]})


CALLS = {
    "where": lambda s: s.where(s > 1, 0, inplace=True),
    "mask": lambda s: s.mask(s > 1, 0, inplace=True),
    "fillna": lambda s: s.fillna(0, inplace=True),
    "ffill": lambda s: s.ffill(inplace=True),
    "bfill": lambda s: s.bfill(inplace=True),
    "interpolate": lambda s: s.interpolate(inplace=True),
    "clip": lambda s: s.clip(0, 2, inplace=True),
    "replace": lambda s: s.replace(3.0, 9.0, inplace=True),
    "dropna": lambda s: s.dropna(inplace=True),
    "drop_duplicates": lambda s: s.drop_duplicates(inplace=True),
    "sort_values": lambda s: s.sort_values(inplace=True),
    "sort_index": lambda s: s.sort_index(inplace=True),
    "reset_index": lambda s: s.reset_index(inplace=True),
}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_inplace_true_is_rejected(name):
    d = DataStore(_frame())
    with pytest.raises(ImmutableError):
        CALLS[name](d["a"])
    assert_datastore_equals_pandas(d, _frame())


NOT_INPLACE = {
    "where": lambda s: s.where(s > 1, 0, inplace=False),
    "mask": lambda s: s.mask(s > 1, 0, inplace=False),
    "clip": lambda s: s.clip(0, 2, inplace=False),
    "replace": lambda s: s.replace(3.0, 9.0, inplace=False),
    "dropna": lambda s: s.dropna(inplace=False),
    "drop_duplicates": lambda s: s.drop_duplicates(inplace=False),
    "sort_index": lambda s: s.sort_index(ascending=False, inplace=False),
}


@pytest.mark.parametrize("name", sorted(NOT_INPLACE))
def test_inplace_false_matches_pandas(name):
    d = DataStore(_frame())
    result = NOT_INPLACE[name](d["a"])
    expected = NOT_INPLACE[name](_frame()["a"])
    assert_series_equal(get_series(result), expected)


def test_lazy_series_rejects_inplace():
    d = DataStore(_frame())
    with pytest.raises(ImmutableError):
        LazySeries(d["a"], "where", d["a"] > 1, 0, inplace=True)
