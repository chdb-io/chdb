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


NATIVE_ZOO = """
SELECT
    toDecimal64(1.25, 2)                    AS dec,
    toDateTime('2026-01-01 10:00:00')       AS dt,
    toDateTime64('2026-01-01 10:00:00.123', 3) AS dt64,
    toDate('2026-01-02')                    AS d,
    'abc'                                   AS s,
    toUInt64(18446744073709551615)          AS u64,
    toInt32(-5)                             AS i32,
    toFloat32(1.5)                          AS f32,
    toNullable(toUInt8(3))                  AS n,
    CAST(NULL AS Nullable(Int64))           AS nn,
    ['a', 'b']                              AS arr,
    toUUID('6f9619ff-8b86-d011-b42d-00c04fc964ff') AS uid
"""


def test_native_bytes_convert_exactly_like_local_execution():
    """A foreign result must land with the dtypes local execution produces."""
    import chdb

    from datastore.pushdown import frame_from_native

    local = chdb.query(NATIVE_ZOO, "DataFrame")
    converted = frame_from_native(chdb.query(NATIVE_ZOO, "Native").bytes())

    assert [str(dtype) for dtype in converted.dtypes] == [
        str(dtype) for dtype in local.dtypes
    ]
    pd.testing.assert_frame_equal(converted, local)


def test_an_empty_result_converts_to_an_empty_frame():
    from datastore.pushdown import frame_from_native

    assert frame_from_native(b"").empty


def test_apply_does_not_warn_about_a_parameter_pandas_deprecated():
    """A default apply() must not pass convert_dtype, which pandas 2.1 deprecated."""
    import warnings

    store = DataStore(pd.DataFrame({"a": [1, 2, 3]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = store["a"].apply(lambda value: value * 2)
        values = list(result.to_pandas() if hasattr(result, "to_pandas") else result)

    assert values == [2, 4, 6]
    assert [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "convert_dtype" in str(warning.message)
    ] == []



class PlanRecorder:
    """Collects the placement reports a plan publishes while executing."""

    def __init__(self):
        self.reports = []

    def __call__(self, placements):
        self.reports.append([placement.as_dict() for placement in placements])

    @property
    def last(self):
        return self.reports[-1] if self.reports else []


@pytest.fixture
def recorder():
    from datastore.pushdown import set_plan_observer

    recorder = PlanRecorder()
    set_plan_observer(recorder)
    yield recorder
    set_plan_observer(None)


def test_a_pushed_down_segment_is_reported_as_running_on_the_server(recorder):
    chain(remote_store(RecordingExecutor())).to_pandas()

    assert len(recorder.last) == 1
    segment = recorder.last[0]
    assert segment["kind"] == "sql"
    assert segment["engine"] == "remote_clickhouse"
    assert segment["reasonCode"] == "sql_pushed_to_source"
    assert f'"{DATABASE}"."{TABLE}"' in segment["sql"]
    assert len(segment["ops"]) == 4


def test_the_same_chain_on_a_local_frame_is_reported_as_local(recorder):
    """No executor, no server: the report must not claim the work went remote."""
    store = DataStore(
        pd.DataFrame(
            {
                "channel": ["paid", "organic", "paid"],
                "revenue": [1.0, 2.0, 3.0],
                "event_type": ["purchase"] * 3,
            }
        )
    )
    chain(store).to_pandas()

    engines = {segment["engine"] for segment in recorder.last}
    assert "remote_clickhouse" not in engines
    assert "local_chdb" in engines
    local = next(s for s in recorder.last if s["engine"] == "local_chdb")
    assert local["reasonCode"] in {"sql_on_source_locally", "sql_on_returned_frame"}
    assert local["detail"]


def test_a_pandas_tail_is_reported_with_the_reason_it_stayed_local(recorder):
    store = remote_store(RecordingExecutor())
    filtered = store[store["event_type"] == "purchase"][["channel", "revenue"]]
    # groupby().apply() carries a Python callable, so the planner cannot compile
    # it and the chain ends in a pandas segment.
    filtered.groupby("channel").apply(lambda frame: frame.head(1)).to_pandas()

    assert [segment["engine"] for segment in recorder.last] == [
        "remote_clickhouse",
        "pandas",
    ]
    tail = recorder.last[1]
    assert tail["kind"] == "pandas"
    assert tail["reasonCode"] == "python_callable"
    assert tail["detail"]
    assert tail["sql"] is None


def test_an_observer_that_raises_cannot_break_a_query():
    from datastore.pushdown import set_plan_observer

    def explode(_placements):
        raise RuntimeError("observer is broken")

    set_plan_observer(explode)
    try:
        result = chain(remote_store(RecordingExecutor())).to_pandas()
    finally:
        set_plan_observer(None)

    assert len(result) == len(REMOTE_ROWS)
