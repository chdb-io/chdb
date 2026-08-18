"""Focused tests for structured pushdown explanations.

These tests exercise planner metadata only.  They intentionally do not execute
the resulting DataStore so this first slice can be adopted by a remote notebook
adapter without changing execution routing.
"""

from datastore.conditions import BinaryCondition
from datastore.expressions import Field, Literal
from datastore.lazy_ops import LazyApply, LazyRelationalOp
from datastore.query_planner import QueryPlanner


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
    assert decision.reason_code == "sql_supported"
    assert decision.detail
    assert decision.as_dict() == {
        "op_index": None,
        "op_type": "WHERE",
        "eligible": True,
        "reason_code": "sql_supported",
        "detail": "WHERE is supported by the SQL planner",
    }


def test_python_callable_has_explicit_pandas_reason():
    planner = QueryPlanner()
    op = LazyApply(lambda frame: frame, "identity")

    decision = planner.explain_op_pushdown(op)

    assert decision.eligible is False
    assert decision.op_type == "APPLY"
    assert decision.reason_code == "python_callable"
    assert "Python callable" in decision.detail


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
        "sql_supported",
        "python_callable",
        "sql_supported",
    ]

