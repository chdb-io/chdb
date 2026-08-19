"""Focused tests for structured pushdown explanations.

These tests exercise planner metadata only.  They intentionally do not execute
the resulting DataStore so this first slice can be adopted by a remote notebook
adapter without changing execution routing.
"""

from datastore.conditions import BinaryCondition
from datastore.expressions import Field, Literal
from datastore.lazy_ops import LazyApply, LazyGroupByAgg, LazyRelationalOp
from datastore.query_planner import (
    PUSHDOWN_REASON_DETAILS,
    PushdownReasonCode,
    QueryPlanner,
)


def test_supported_sql_operation_has_stable_decision_code():
    planner = QueryPlanner()
    op = LazyRelationalOp(
        "WHERE",
        "value_filter",
        condition=BinaryCondition(">", Field("value"), Literal(10)),
    )

    decision = planner.explain_op_pushdown(op, schema={"value": "Int64"})

    assert decision.eligible is True
    assert decision.op_type == "WHERE"
    assert decision.reason_code is PushdownReasonCode.SQL_SUPPORTED
    assert decision.detail
    assert decision.as_dict() == {
        "op_index": None,
        "op_type": "WHERE",
        "eligible": True,
        "reason_code": PushdownReasonCode.SQL_SUPPORTED.value,
        "detail": PUSHDOWN_REASON_DETAILS[PushdownReasonCode.SQL_SUPPORTED],
    }


def test_python_callable_has_explicit_pandas_reason():
    planner = QueryPlanner()
    op = LazyApply(lambda frame: frame, "identity")

    decision = planner.explain_op_pushdown(op)

    assert decision.eligible is False
    assert decision.op_type == "APPLY"
    assert decision.reason_code is PushdownReasonCode.PYTHON_CALLABLE
    assert decision.detail == PUSHDOWN_REASON_DETAILS[PushdownReasonCode.PYTHON_CALLABLE]


def test_reason_code_details_have_a_complete_one_to_one_mapping():
    assert set(PUSHDOWN_REASON_DETAILS) == set(PushdownReasonCode)
    assert all(PUSHDOWN_REASON_DETAILS[code] for code in PushdownReasonCode)


def test_order_sensitive_and_meaningless_sorts_have_distinct_reasons():
    planner = QueryPlanner()
    order = LazyRelationalOp("ORDER BY", "sort", fields=[Field("value")])
    meaningless_aggregation = LazyGroupByAgg(
        groupby_cols=["category"], agg_func="sum"
    )
    order_sensitive_aggregation = LazyGroupByAgg(
        groupby_cols=["category"], agg_func="first"
    )

    meaningless = planner.explain_op_pushdown(
        order, following_ops=[meaningless_aggregation]
    )
    order_sensitive = planner.explain_op_pushdown(
        order_sensitive_aggregation, preceding_ops=[order]
    )

    assert meaningless.reason_code is PushdownReasonCode.MEANINGLESS_SORT_BEFORE_AGGREGATION
    assert order_sensitive.reason_code is PushdownReasonCode.ORDER_DEPENDENT_AGGREGATION


def test_execution_plan_keeps_decisions_aligned_with_segments():
    planner = QueryPlanner()
    ops = [
        LazyRelationalOp(
            "WHERE",
            "value_filter",
            condition=BinaryCondition(">", Field("value"), Literal(10)),
        ),
        LazyApply(lambda frame: frame, "identity"),
        LazyRelationalOp("LIMIT", "top_rows", limit_value=5),
    ]

    plan = planner.plan_segments(ops, has_sql_source=True)
    explanation = plan.explain()

    assert [segment.segment_type for segment in plan.segments] == [
        "sql",
        "pandas",
        "sql",
    ]
    assert [item["op_index"] for item in explanation] == [0, 1, 2]
    assert [item["segment_type"] for item in explanation] == [
        "sql",
        "pandas",
        "sql",
    ]
    assert [item["reason_code"] for item in explanation] == [
        PushdownReasonCode.SQL_SUPPORTED.value,
        PushdownReasonCode.PYTHON_CALLABLE.value,
        PushdownReasonCode.SQL_SUPPORTED.value,
    ]
