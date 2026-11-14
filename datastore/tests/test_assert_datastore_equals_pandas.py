"""
Tests for the assert_datastore_equals_pandas() test helper itself.

check_index=True used to be accepted and ignored, so tests relying on it
passed even when DataStore returned the wrong row labels (#771).
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


def test_check_index_rejects_different_labels():
    expected = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
    result = DataStore(pd.DataFrame({"a": [1, 2, 3]}))

    with pytest.raises(AssertionError, match="index"):
        assert_datastore_equals_pandas(result, expected, check_index=True)
    # Labels are not compared by default
    assert_datastore_equals_pandas(result, expected)


def test_check_index_accepts_matching_labels():
    pd_df = pd.DataFrame({"a": [3, 1, 2]}, index=[10, 20, 30])
    ds_df = DataStore(pd_df.copy())

    assert_datastore_equals_pandas(ds_df.sort_values("a"), pd_df.sort_values("a"), check_index=True)


def test_check_index_series():
    pd_df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
    ds_df = DataStore(pd_df.copy())

    assert_datastore_equals_pandas(ds_df["a"], pd_df["a"], check_index=True)
    with pytest.raises(AssertionError, match="index"):
        assert_datastore_equals_pandas(ds_df["a"], pd_df["a"].set_axis([0, 1, 2]), check_index=True)


def test_check_index_without_row_order():
    pd_df = pd.DataFrame({"k": ["b", "a", "b"], "v": [1, 2, 3]})
    ds_df = DataStore(pd_df.copy())

    pd_result = pd_df.groupby("k").sum()
    ds_result = ds_df.groupby("k").sum()

    # Labels are compared regardless of their order
    assert_datastore_equals_pandas(ds_result, pd_result.iloc[::-1], check_index=True, check_row_order=False)
    with pytest.raises(AssertionError, match="index"):
        assert_datastore_equals_pandas(
            ds_result, pd_result.set_axis(["x", "y"]), check_index=True, check_row_order=False
        )
