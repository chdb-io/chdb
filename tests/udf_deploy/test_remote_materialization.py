"""plan_remote_materialization: one server-side statement, or the reason why not.

Unit tests only — no server, no execution. The planner and the remote-target
compiler are exercised on pipelines whose schema lookups are stubbed out.
"""

import pandas as pd
import pytest

import chdb
from datastore.core import DataStore
from datastore.pushdown import REMOTE_CLICKHOUSE
from datastore import udf as udf_registry


@pytest.fixture()
def remote_events(monkeypatch):
    ds = DataStore.from_clickhouse("clickhouse:9000", "demo", "events")
    # No server in unit tests: stub every structure lookup the builders use.
    monkeypatch.setattr(
        DataStore,
        "schema",
        lambda self: {"revenue": "Float64", "channel": "String"},
    )
    monkeypatch.setattr(
        DataStore,
        "_get_all_column_names",
        lambda self: ["revenue", "channel"],
    )
    return ds


def _bind(name, fn, remote=None):
    udf_registry.bind_local(fn, name, arity=2, local_name=name, arg_types=None)
    if remote:
        udf_registry.bind_remote(fn, name, connection=remote, remote_name=name)


@pytest.fixture(autouse=True)
def _clean_bindings():
    yield
    udf_registry._BINDINGS.pop("mat_tax", None)


def test_in_memory_frame_is_refused():
    ds = chdb.to_datastore(pd.DataFrame({"price": [1.0, 2.0]}))
    verdict = ds.assign(c=ds["price"] * 2).plan_remote_materialization()
    assert verdict["eligible"] is False
    assert "in-memory DataFrame" in verdict["reason"]
    assert verdict["sql"] is None


def test_pandas_only_operation_is_refused(remote_events, monkeypatch):
    # Which ops classify as pandas-only is the planner's own moving target;
    # this pins the refusal branch itself against a plan WITH a pandas segment.
    from datastore import query_planner as qp

    split_plan = type(
        "FakePlan",
        (),
        {
            "segments": [
                qp.ExecutionSegment(segment_type="sql", ops=[], is_first_segment=True),
                qp.ExecutionSegment(segment_type="pandas", ops=[]),
            ]
        },
    )()
    monkeypatch.setattr(
        qp.QueryPlanner, "plan_segments", lambda self, *a, **k: split_plan
    )
    ds = remote_events
    verdict = ds.filter(ds["channel"] == "web").plan_remote_materialization()
    assert verdict["eligible"] is False
    assert "pandas-only" in verdict["reason"]


def test_plain_sql_pipeline_compiles_for_the_server(remote_events):
    ds = remote_events
    pipe = ds.filter(ds["channel"] == "web").assign(t=ds["revenue"] * 1.13)
    verdict = pipe.plan_remote_materialization()
    assert verdict["eligible"] is True, verdict["reason"]
    # sources render as the server's own table, never remote()
    assert '"demo"."events"' in verdict["sql"]
    assert "remote(" not in verdict["sql"]
    assert '"channel" = \'web\'' in verdict["sql"]


def test_undeployed_udf_is_refused(remote_events):
    def mat_tax(revenue: float, rate: float) -> float:
        return revenue * (1 + rate)

    _bind("mat_tax", mat_tax)  # local only, never deployed
    ds = remote_events
    call = udf_registry.UdfCall(
        udf_registry._BINDINGS["mat_tax"], ds["revenue"]._expr, 0.13
    )
    pipe = ds.assign(t=call)
    verdict = pipe.plan_remote_materialization()
    assert verdict["eligible"] is False
    assert "not deployed" in verdict["reason"]


def test_deployed_udf_compiles_by_its_remote_name(remote_events):
    def mat_tax(revenue: float, rate: float) -> float:
        return revenue * (1 + rate)

    _bind("mat_tax", mat_tax, remote="demo")
    ds = remote_events
    call = udf_registry.UdfCall(
        udf_registry._BINDINGS["mat_tax"], ds["revenue"]._expr, 0.13
    )
    pipe = ds.assign(t=call)
    verdict = pipe.plan_remote_materialization()
    assert verdict["eligible"] is True, verdict["reason"]
    assert "mat_tax" in verdict["sql"]
    assert '"demo"."events"' in verdict["sql"]
    assert verdict["udfs"] and verdict["udfs"][0]["name"] == "mat_tax"


def test_groupby_by_expression_is_refused_not_broken(remote_events, monkeypatch):
    # The flat SQL form references a kernel-internal __groupby_temp_* column
    # the server has never heard of; a statement the server would refuse must
    # never leave here marked eligible.
    monkeypatch.setattr(DataStore, "schema", lambda self: {"event_time": "DateTime", "user_id": "UInt64", "revenue": "Float64", "channel": "String"})
    monkeypatch.setattr(DataStore, "_get_all_column_names", lambda self: ["event_time", "user_id", "revenue", "channel"])
    ds = remote_events
    pipe = ds.groupby(ds["event_time"].toDate()).agg(n=("user_id", "count"))
    verdict = pipe.plan_remote_materialization()
    assert verdict["eligible"] is False
    assert "temporary column" in verdict["reason"]
    assert verdict["sql"] is None


def test_groupby_by_column_stays_eligible(remote_events):
    ds = remote_events
    pipe = ds.groupby("channel").agg(n=("revenue", "count"))
    verdict = pipe.plan_remote_materialization()
    assert verdict["eligible"] is True, verdict["reason"]
    assert "__groupby_temp_" not in verdict["sql"]
    assert '"demo"."events"' in verdict["sql"]
