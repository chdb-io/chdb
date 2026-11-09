"""
Datetime literals in SQL keep their full precision: a numpy datetime64 or
pandas Timestamp with nanoseconds selects the same rows as in pandas.
"""

import operator

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas

LITERALS = [
    pytest.param(np.datetime64("2023-01-01T00:00:00.000000001"), id="datetime64_ns"),
    pytest.param(pd.Timestamp("2023-01-01 00:00:00.000000001"), id="timestamp_ns"),
    pytest.param(np.datetime64("2023-01-01T00:00:00.000001"), id="datetime64_us"),
]

OPERATORS = [operator.eq, operator.ne, operator.gt, operator.ge, operator.lt]


def _frame():
    return pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:00:00.000000001",
                    "2023-01-01 00:00:00.000000002",
                    "2023-01-01 00:00:00.000001",
                ],
                format="ISO8601",
            ),
            "v": [0, 1, 2, 3],
        }
    )


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize("literal", LITERALS)
def test_filter(op, literal):
    def chain(df):
        return df[op(df["t"], literal)]

    assert_datastore_equals_pandas(chain(DataStore(_frame())), chain(_frame()))


@pytest.mark.parametrize("literal", LITERALS)
def test_assign_comparison(literal):
    def chain(df):
        return df.assign(after=lambda d: d["t"] > literal)

    assert_datastore_equals_pandas(chain(DataStore(_frame())), chain(_frame()))
