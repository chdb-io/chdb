"""plan_remote_materialization: one server-side statement, or the reason why not.

Unit tests only — no server, no execution. Structure lookups the builders use
are stubbed out.
"""

import pandas as pd
import pytest

import chdb
from datastore.core import DataStore


@pytest.fixture()
def remote_events(monkeypatch):
    ds = DataStore.from_clickhouse("clickhouse:9000", "demo", "events")
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


def test_in_memory_frame_is_refused():
    ds = chdb.to_datastore(pd.DataFrame({"price": [1.0, 2.0]}))
    verdict = ds.assign(c=ds["price"] * 2).plan_remote_materialization()
    assert verdict["eligible"] is False
    assert "in-memory DataFrame" in verdict["reason"]
    assert verdict["sql"] is None


def test_pandas_only_pipeline_is_refused(remote_events, monkeypatch):
    monkeypatch.setattr(DataStore, "_is_fully_sql_pushable", lambda self: False)
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
