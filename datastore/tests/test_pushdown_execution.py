"""Route a compiled SQL segment to a remote ClickHouse server.

These tests use a recording executor instead of a live server: what matters here
is which SQL leaves the planner, that it is executed exactly once, that the
returned rows are the result of the chain, and that a remote failure is never
retried locally.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.pushdown import (
    REMOTE_CLICKHOUSE,
    RemoteSource,
    SegmentExecutorError,
    SegmentResult,
    SqlSegmentExecutor,
)

HOST = "clickhouse:9000"
DATABASE = "demo"
TABLE = "events"

SCHEMA = {"channel": "String", "revenue": "Float64", "event_type": "String"}

REMOTE_ROWS = pd.DataFrame(
    {"channel": ["paid", "organic"], "revenue": [900.0, 400.0]}
)


class RecordingExecutor(SqlSegmentExecutor):
    """Captures the SQL it is handed and returns a fixed result."""

    def __init__(self, frame=REMOTE_ROWS, metrics=None, database=DATABASE):
        self.frame = frame
        self.metrics = metrics or {"query_id": "gc-test-1", "rows_read": 50_000_000}
        self.database = database
        self.calls = []

    def accepts(self, source):
        return source.host == HOST and source.database == self.database

    def execute(self, sql, source):
        self.calls.append((sql, source))
        return SegmentResult(frame=self.frame.copy(), metrics=self.metrics)


class FailingExecutor(RecordingExecutor):
    def execute(self, sql, source):
        self.calls.append((sql, source))
        raise RuntimeError("remote query rejected")


def remote_store(executor=None, table=TABLE, database=DATABASE):
    """A remote-backed DataStore with a known schema, so no server is contacted."""
    store = DataStore("clickhouse", host=HOST, database=database, table=table)
    store._schema = dict(SCHEMA)
    if executor is not None:
        store.set_sql_segment_executor(executor)
    return store


def chain(store):
    """Filter, project, group, sort - a chain the planner keeps in one segment."""
    return (
        store[store["event_type"] == "purchase"][["channel", "revenue"]]
        .groupby("channel", as_index=False)
        .agg({"revenue": "sum"})
        .sort_values("revenue", ascending=False)
    )


def test_remote_segment_reads_the_table_directly():
    executor = RecordingExecutor()
    result = chain(remote_store(executor)).to_pandas()

    assert len(executor.calls) == 1
    sql, source = executor.calls[0]
    assert "remote(" not in sql
    assert f'"{DATABASE}"."{TABLE}"' in sql
    assert "GROUP BY" in sql
    assert source == RemoteSource(HOST, DATABASE, TABLE, secure=False)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), REMOTE_ROWS)


def test_local_path_still_uses_the_remote_table_function():
    sql = chain(remote_store()).to_sql(execution_format=True)
    assert "remote(" in sql
    assert f'"{DATABASE}"."{TABLE}"' not in sql


def test_sql_preview_shows_what_the_remote_server_will_run():
    sql = chain(remote_store(RecordingExecutor())).to_sql(execution_format=True)
    assert "remote(" not in sql
    assert f'"{DATABASE}"."{TABLE}"' in sql


def explain_text(store, capsys):
    store.explain()
    return capsys.readouterr().out


def test_explain_names_the_engine_that_will_run_the_segment(capsys):
    plan = explain_text(chain(remote_store(RecordingExecutor())), capsys)

    assert f'"{DATABASE}"."{TABLE}"' in plan
    assert "[ClickHouse]" in plan
    assert "[chDB]" not in plan


def test_explain_stays_local_without_an_executor(capsys):
    plan = explain_text(chain(remote_store()), capsys)

    assert "remote(" in plan
    assert "[chDB]" in plan
    assert "[ClickHouse]" not in plan


def test_declined_source_is_not_pushed_down():
    executor = RecordingExecutor(database="other")
    store = remote_store(executor)
    assert store._pushdown_for_first_segment() == (None, None)
    assert executor.calls == []


def test_remote_failure_is_not_retried_locally():
    executor = FailingExecutor()
    with pytest.raises(RuntimeError, match="remote query rejected"):
        chain(remote_store(executor)).to_pandas()
    assert len(executor.calls) == 1


def test_trace_records_target_sql_and_metrics():
    executor = RecordingExecutor()
    store = chain(remote_store(executor))
    store.to_pandas()

    trace = store.last_pushdown_trace
    assert trace is not None
    assert trace.target == REMOTE_CLICKHOUSE
    assert trace.source.qualified_name() == f"{DATABASE}.{TABLE}"
    assert trace.result_rows == len(REMOTE_ROWS)
    assert trace.metrics["query_id"] == "gc-test-1"
    assert "remote(" not in trace.sql


def test_executor_returning_a_plain_frame_is_accepted():
    class FrameOnlyExecutor(RecordingExecutor):
        def execute(self, sql, source):
            self.calls.append((sql, source))
            return self.frame.copy()

    executor = FrameOnlyExecutor()
    result = chain(remote_store(executor)).to_pandas()
    pd.testing.assert_frame_equal(result.reset_index(drop=True), REMOTE_ROWS)


def test_executor_returning_junk_is_rejected():
    class JunkExecutor(RecordingExecutor):
        def execute(self, sql, source):
            self.calls.append((sql, source))
            return [{"channel": "paid"}]

    with pytest.raises(SegmentExecutorError):
        chain(remote_store(JunkExecutor())).to_pandas()


def test_binding_requires_the_executor_contract():
    with pytest.raises(TypeError):
        remote_store().set_sql_segment_executor(object())
