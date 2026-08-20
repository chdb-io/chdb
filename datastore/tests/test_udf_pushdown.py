"""Applying a deployed Python UDF compiles into the query.

A UDF is the one kind of Python a chain does not have to come home for: the
engine can call it. What these tests pin down is that applying one produces SQL
rather than a pandas fallback, that the name in that SQL belongs to whichever
engine is being compiled for, and that an ordinary callable is still left alone.
"""

import pandas as pd
import pytest

from datastore import DataStore
from datastore.function_registry import FunctionRegistry
from datastore.pushdown import (
    LOCAL_CHDB,
    REMOTE_CLICKHOUSE,
    SegmentResult,
    SqlSegmentExecutor,
    compiling_for,
)
from datastore.udf import (
    UdfBinding,
    bind_local,
    bind_remote,
    binding_for,
    binding_named,
    clear_bindings,
)

FRAME = pd.DataFrame({"revenue": [100.0, 200.0, 300.0], "channel": ["a", "b", "a"]})

def _house_rate():
    """A call the translator will not follow, which is the point.

    Every rule in this file has to stay a function: one that can be said in SQL
    is said in SQL now, and would never reach the deployment machinery these
    tests are about.
    """
    return 0.9




def register(name, fn, arity=1):
    """Bind a callable as a UDF the way chdb.deploy does, without deploying."""
    bind_local(fn, name, arity)
    FunctionRegistry.register_udf(name, arity, doc=f"test udf {name}")
    return fn


@pytest.fixture(autouse=True)
def clean_bindings():
    yield
    clear_bindings()


def test_applying_a_udf_compiles_into_the_query():
    def scale_udf(value):
        return value * _house_rate()

    register("scale_udf", scale_udf)
    store = DataStore(FRAME)

    sql = store.assign(net=store["revenue"].apply(scale_udf)).to_sql(
        execution_format=True
    )

    assert 'scale_udf("revenue")' in sql


def test_apply_and_the_column_method_compile_the_same_way():
    """The two spellings are the same expression, so they must agree."""

    def bonus_udf(value):
        return value + _house_rate()

    register("bonus_udf", bonus_udf)
    store = DataStore(FRAME)

    applied = store.assign(x=store["revenue"].apply(bonus_udf)).to_sql(
        execution_format=True
    )
    called = store.assign(x=store["revenue"].bonus_udf()).to_sql(execution_format=True)

    assert applied == called


def test_an_ordinary_callable_still_goes_to_pandas():
    """Nothing here may guess a SQL translation for arbitrary Python."""
    store = DataStore(FRAME)

    sql = store.assign(doubled=store["revenue"].apply(lambda v: v * 2)).to_sql(
        execution_format=True
    )

    assert "doubled" not in sql


def test_the_deployed_name_is_used_only_when_compiling_for_the_server():
    def tax_udf(value):
        return value * _house_rate()

    register("tax_udf", tax_udf)
    bind_remote(tax_udf, "tax_udf", "demo", "chdb_udf_9f2a_c31d")
    store = DataStore(FRAME)

    def compiled():
        return store.assign(net=store["revenue"].apply(tax_udf)).to_sql(
            execution_format=True
        )

    local_sql = compiled()
    with compiling_for(REMOTE_CLICKHOUSE):
        remote_sql = compiled()
    after = compiled()

    assert 'tax_udf("revenue")' in local_sql
    assert 'chdb_udf_9f2a_c31d("revenue")' in remote_sql
    assert "tax_udf" not in remote_sql
    # The block restores the target, so the next compile is local again.
    assert after == local_sql


def test_a_udf_that_was_never_deployed_has_no_name_on_the_server():
    """This is what stops a planner from pushing a call the server cannot run."""
    binding = bind_local(None, "local_only_udf", 1)

    assert binding.name_for(LOCAL_CHDB) == "local_only_udf"
    assert binding.name_for(REMOTE_CLICKHOUSE) is None
    assert binding.runs_on(LOCAL_CHDB)
    assert not binding.runs_on(REMOTE_CLICKHOUSE)


def test_one_deployment_needs_no_connection_but_two_do():
    binding = UdfBinding("shared_udf", 1, local_name="shared_udf")
    binding.remote_names["demo"] = "deployed_on_demo"

    assert binding.name_for(REMOTE_CLICKHOUSE) == "deployed_on_demo"

    binding.remote_names["other"] = "deployed_on_other"

    # Two candidates and no way to choose is not an answer.
    assert binding.name_for(REMOTE_CLICKHOUSE) is None
    assert binding.name_for(REMOTE_CLICKHOUSE, "other") == "deployed_on_other"


def test_a_binding_is_found_by_the_callable_or_by_its_name():
    def named_udf(value):
        return value

    bind_local(named_udf, "named_udf", 1)

    assert binding_for(named_udf) is binding_named("named_udf")
    assert binding_for(lambda v: v) is None
    assert binding_for(len) is None


def test_extra_arguments_reach_the_call():
    def rate_udf(value, rate):
        return value * rate * _house_rate()

    register("rate_udf", rate_udf, arity=2)
    store = DataStore(FRAME)

    sql = store.assign(net=store["revenue"].apply(rate_udf, args=(0.5,))).to_sql(
        execution_format=True
    )

    assert 'rate_udf("revenue",0.5)' in sql.replace(", ", ",")


def test_the_values_are_the_ones_the_udf_produces():
    """Compiling into SQL is only worth anything if the answer is unchanged."""
    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64")
    def recognized_revenue(value):
        return value * 0.92 * _house_rate()

    store = DataStore(FRAME)
    result = store.assign(
        net=store["revenue"].apply(recognized_revenue)
    ).to_pandas()

    expected = FRAME["revenue"].map(recognized_revenue)
    pd.testing.assert_series_equal(
        result["net"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# A call the server has no name for keeps its segment at home
# ---------------------------------------------------------------------------


class RecordingExecutor(SqlSegmentExecutor):
    """Stands in for a bound ClickHouse: records what it was asked to run.

    ``throughput`` is what the link has been measured to carry. None - the
    default - is an unmeasured link, which is the state a fresh session is in.
    """

    target = REMOTE_CLICKHOUSE

    def __init__(self, throughput=None):
        self.calls = []
        self.throughput = throughput

    def observed_throughput_bytes_per_s(self):
        return self.throughput

    def accepts(self, source):
        return True

    def execute(self, sql, source):
        self.calls.append(sql)
        return SegmentResult(
            frame=pd.DataFrame({"channel": ["a"], "net": [1.0]}), metrics={}
        )


def remote_store(executor=None):
    store = DataStore("clickhouse", host="ch:9000", database="demo", table="events")
    store._schema = {"revenue": "Float64", "channel": "String", "event_type": "String"}
    if executor is not None:
        store.set_sql_segment_executor(executor)
    return store


def udf_chain(store):
    purchases = store[store["event_type"] == "purchase"]
    return (
        purchases.assign(net=purchases["revenue"].apply(_recognized))
        .groupby("channel")
        .agg({"net": "sum"})
    )


def _recognized(value):  # replaced per test by register()
    return value * _house_rate()


def test_an_undeployed_udf_keeps_only_itself_at_home():
    """The call stays, the filter still goes: they end up in different segments.

    remote() is a distributed read, so a UDF left in the source statement would
    be sent to a shard that has never heard of it. Splitting there costs nothing
    - the server still does the scan and the filter, and only the call comes
    home.
    """
    register("_recognized", _recognized)
    executor = RecordingExecutor()

    sql = udf_chain(remote_store(executor)).to_sql(execution_format=True)

    assert "_recognized" not in sql
    assert "remote(" not in sql
    assert '"demo"."events"' in sql
    assert "WHERE" in sql


def test_the_same_chain_is_pushed_down_once_the_udf_is_deployed():
    """Deployed and on a link slow enough that moving the rows costs more."""
    register("_recognized", _recognized)
    bind_remote(_recognized, "_recognized", "demo", "chdb_udf_7c1a_9b3e")
    executor = RecordingExecutor(throughput=1_000_000)  # 1 MB/s

    udf_chain(remote_store(executor)).to_pandas()

    assert len(executor.calls) == 1
    pushed = executor.calls[0]
    assert 'chdb_udf_7c1a_9b3e("revenue")' in pushed
    assert "remote(" not in pushed
    assert "_recognized" not in pushed


def test_the_reason_names_the_function_that_kept_the_segment_home():
    register("_recognized", _recognized)
    executor = RecordingExecutor()
    store = remote_store(executor)

    code, sentence = store._pushdown_blocked_by_udf(
        udf_chain(store)._lazy_ops, executor
    )

    assert code.value == "udf_not_deployed"
    assert "_recognized" in sentence
    assert "not deployed" in sentence


def test_nothing_blocks_a_chain_without_a_udf():
    executor = RecordingExecutor()
    store = remote_store(executor)
    plain = store[store["event_type"] == "purchase"].head(5)

    assert store._pushdown_blocked_by_udf(plain._lazy_ops, executor) is None


def test_a_server_that_lost_the_function_keeps_the_segment_home():
    """A deployment record proves it was shipped, not that it is still there."""

    class ForgetfulServer(RecordingExecutor):
        def resolves_function(self, name, source):
            return False

    register("_recognized", _recognized)
    bind_remote(_recognized, "_recognized", "demo", "chdb_udf_gone")
    executor = ForgetfulServer()
    store = remote_store(executor)

    code, sentence = store._pushdown_blocked_by_udf(
        udf_chain(store)._lazy_ops, executor
    )
    udf_chain(remote_store(executor)).to_sql(execution_format=True)

    assert code.value == "udf_missing_on_server"
    assert "chdb_udf_gone" in sentence
    assert executor.calls == []


def test_an_executor_that_cannot_check_is_trusted():
    """Preflight catches a stale deployment; it is not a new way to fail."""

    class Unsure(RecordingExecutor):
        def resolves_function(self, name, source):
            return None

    class Broken(RecordingExecutor):
        def resolves_function(self, name, source):
            raise RuntimeError("system.functions is unreachable")

    register("_recognized", _recognized)
    bind_remote(_recognized, "_recognized", "demo", "chdb_udf_ok")

    for executor in (Unsure(1_000_000), Broken(1_000_000), RecordingExecutor(1_000_000)):
        store = remote_store(executor)
        assert (
            store._pushdown_blocked_by_udf(udf_chain(store)._lazy_ops, executor)
            is None
        )


def test_calling_a_udf_with_the_wrong_number_of_arguments_stays_home():
    """The server reports an executable UDF's name and nothing about its shape."""

    def two_arg_udf(value, rate):
        return value * rate * _house_rate()

    register("two_arg_udf", two_arg_udf, arity=2)
    bind_remote(two_arg_udf, "two_arg_udf", "demo", "chdb_udf_two")
    executor = RecordingExecutor()
    store = remote_store(executor)
    chain = store.assign(net=store["revenue"].apply(two_arg_udf))

    code, sentence = store._pushdown_blocked_by_udf(chain._lazy_ops, executor)

    assert code.value == "udf_arity_mismatch"
    assert "takes 2 argument(s) but is called with 1" in sentence


def test_explain_shows_the_split_an_undeployed_udf_forces(capsys):
    register("_recognized", _recognized)
    udf_chain(remote_store(RecordingExecutor())).explain()

    plan = capsys.readouterr().out

    # The source segment goes to the server; the call is described separately.
    assert "[ClickHouse]" in plan
    assert "chDB" in plan


def test_a_udf_call_survives_being_aliased_and_rebuilt():
    """Copying an expression must not turn a bound call into a plain function."""
    from copy import copy

    from datastore.udf import UdfCall, udf_calls_in

    register("_recognized", _recognized)
    store = DataStore(FRAME)
    call = store["revenue"].apply(_recognized)._expr

    assert isinstance(copy(call), UdfCall)
    assert isinstance(call.as_("net"), UdfCall)
    assert isinstance(call.rebuild_with_args(list(call.args)), UdfCall)
    # And it is still findable once the chain is built.
    chain = store.assign(net=store["revenue"].apply(_recognized))
    assert [c.binding.logical_name for c in udf_calls_in(chain._lazy_ops)] == [
        "_recognized"
    ]


def test_the_report_names_the_function_and_the_name_it_ran_under():
    """A reader matching hashes against their own code is not a report."""
    from datastore.pushdown import set_plan_observer

    reports = []
    set_plan_observer(lambda placements: reports.append(
        [p.as_dict() for p in placements]
    ))
    try:
        register("_recognized", _recognized)
        bind_remote(_recognized, "_recognized", "demo", "chdb_nb_3f99a3_585a1cb9")
        udf_chain(remote_store(RecordingExecutor(throughput=1_000_000))).to_pandas()
    finally:
        set_plan_observer(None)

    assert reports
    segment = reports[-1][0]
    assert segment["engine"] == "remote_clickhouse"
    assert segment["udfs"] == [
        {"name": "_recognized", "deployedAs": "chdb_nb_3f99a3_585a1cb9"}
    ]


def test_a_local_run_reports_the_function_under_its_own_name():
    from datastore.pushdown import set_plan_observer

    reports = []
    set_plan_observer(lambda placements: reports.append(
        [p.as_dict() for p in placements]
    ))
    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64")
    def reported_locally(value):
        return value * _house_rate()

    try:
        store = DataStore(FRAME)
        store.assign(net=store["revenue"].apply(reported_locally)).to_pandas()
    finally:
        set_plan_observer(None)

    sql_segments = [
        segment for report in reports for segment in report if segment["kind"] == "sql"
    ]
    assert sql_segments
    assert {"name": "reported_locally"} in sql_segments[-1]["udfs"]


def test_a_segment_without_a_udf_reports_none():
    from datastore.pushdown import set_plan_observer

    reports = []
    set_plan_observer(lambda placements: reports.append(
        [p.as_dict() for p in placements]
    ))
    try:
        DataStore(FRAME).head(2).to_pandas()
    finally:
        set_plan_observer(None)

    assert all(segment["udfs"] == [] for report in reports for segment in report)


def test_arguments_are_converted_to_the_declared_types():
    """A Decimal column against a Float64 declaration must work either way."""
    import chdb

    @chdb.func(arg_types=["Float64"], return_type="Float64")
    def declared_float(value):
        return value * _house_rate()

    frame = pd.DataFrame({"revenue": [100, 250], "channel": ["a", "b"]})
    store = DataStore(frame)

    sql = store.assign(net=store["revenue"].apply(declared_float)).to_sql(
        execution_format=True
    )

    assert 'declared_float(CAST("revenue" AS Float64))' in sql
    # And the conversion is not just cosmetic: the call has to run.
    result = store.assign(net=store["revenue"].apply(declared_float)).to_pandas()
    assert list(result["net"]) == [90.0, 225.0]


def test_a_udf_without_declared_types_is_called_as_written():
    from datastore.udf import UdfCall, bind_local

    binding = bind_local(None, "untyped_udf", 1)
    call = UdfCall(binding, __import__("datastore").expressions.Field("revenue"))

    assert call.to_sql() == 'untyped_udf("revenue")'
