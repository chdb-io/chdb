import pandas as pd
import pytest

from datastore import DataStore
from datastore.executor import get_executor


@pytest.mark.parametrize("execution_format", [False, True])
@pytest.mark.parametrize("constructor", [DataStore, DataStore.from_df])
def test_dataframe_filter_sql_can_be_replayed(execution_format, constructor):
    frame = pd.DataFrame({"a": [3, 1, 2]}, index=[30, 10, 20])
    ds = constructor(frame)

    sql = ds[ds["a"] > 1].to_sql(execution_format=execution_format)
    result = get_executor().query_dataframe(sql, frame)

    pd.testing.assert_frame_equal(result, frame[frame["a"] > 1])


def test_dataframe_assignment_sql_keeps_source_and_expression():
    frame = pd.DataFrame({"a": [3, 1, 2]})
    ds = DataStore(frame)
    ds["b"] = ds["a"] * 2

    result = get_executor().query_dataframe(ds.to_sql(execution_format=True), frame)

    pd.testing.assert_frame_equal(result, frame.assign(b=frame["a"] * 2))


def test_dataframe_sql_before_pandas_step_reads_source_only():
    frame = pd.DataFrame({"a": [3, 1, 2]})
    ds = DataStore(frame).sort_values("a")

    result = get_executor().query_dataframe(ds.to_sql(execution_format=True), frame)

    pd.testing.assert_frame_equal(result, frame)
